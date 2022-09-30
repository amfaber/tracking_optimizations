#![allow(warnings)]
use futures;
use futures_intrusive;
type my_dtype = f32;
use pollster::FutureExt;
use std::collections::VecDeque;
use wgpu::{util::DeviceExt, Buffer};
use crate::kernels;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Params{
    pub pic_dims: [u32; 2],
    pub gauss_dims: [u32; 2],
    pub constant_dims: [u32; 2],
    pub circle_dims: [u32; 2],
    pub max_iterations: u32,
    pub shift_threshold: f32,
    pub minmass: f32,
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        ::std::mem::size_of::<T>(),
    )
}


pub async fn execute_gpu<T: Iterator<Item = Vec<my_dtype>>>(mut frames: T, dims: [u32; 2]) -> anyhow::Result<Vec<Vec<my_dtype>>>{
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptionsBase{
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })
    .block_on()
    .ok_or(anyhow::anyhow!("Couldn't create the adapter"))?;

    let (device, queue) = adapter
    .request_device(&Default::default(), None)
    .block_on()?;

    
    let common_header = include_str!("shaders/params.wgsl");

    let shaders = [
        include_str!("shaders/preprocess.wgsl"),
        include_str!("shaders/centers.wgsl"),
        include_str!("shaders/walk.wgsl"),
    ];

    let workgroup_size = [16, 16, 1];
    let workgroups: [u32; 2] = 
        dims.iter().zip(workgroup_size)
        .map(|(&x, size)| (x + size - 1) / size).collect::<Vec<u32>>().try_into().unwrap();

    let preprocess_source = |source| -> String {
        let mut result = String::new();
        result.push_str(common_header);
        result.push_str(source);
        result.replace("@workgroup_size(_)",
        format!("@workgroup_size({}, {}, {})", workgroup_size[0], workgroup_size[1], workgroup_size[2]).as_str())
    };


    let shaders = shaders.iter().map(|&source|{
        let shader_source = preprocess_source(source);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        shader
    }).collect::<Vec<_>>();

    let slice_size = dims.iter().product::<u32>() as usize * std::mem::size_of::<my_dtype>();
    let size = slice_size as wgpu::BufferAddress;

    let mut staging_buffers = Vec::new();
    for i in 0..2{
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Staging {}", i).as_str()),
            size: 2 * size,
            usage: wgpu::BufferUsages::MAP_WRITE 
            | wgpu::BufferUsages::COPY_SRC 
            | wgpu::BufferUsages::COPY_DST 
            | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        staging_buffers.push(staging_buffer);
    }
    let mut free_staging_buffers = staging_buffers.iter().collect::<Vec<&wgpu::Buffer>>();
    let mut in_use_staging_buffers = VecDeque::new();
    
    let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let sigma = 1.;
    let gaussian_kernel = kernels::Kernel::tp_gaussian(sigma, 4.);
    let gaussian_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&gaussian_kernel.data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let noise_size = 9u32;
    let constant_kernel = kernels::Kernel::rolling_average([noise_size, noise_size]);
    let constant_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&constant_kernel.data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let processed_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let centers_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 2 * size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let r = 9;
    let circle_kernel = kernels::Kernel::circle_mask(r);
    let circle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&circle_kernel.data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let masses_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size*2,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let params = Params{
        pic_dims: dims,
        gauss_dims: gaussian_kernel.size,
        constant_dims: constant_kernel.size,
        circle_dims: [noise_size, noise_size],
        max_iterations: 100,
        shift_threshold: 0.6,
        minmass: 500.,
    };

    let param_buffer = unsafe{
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Width Buffer"),
            contents: any_as_u8_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM
        })
    };

    let compute_pipelines = shaders.iter().map(|shader|{
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: shader,
        entry_point: "main",
        })
    }).collect::<Vec<_>>();

    let bind_group_layouts = compute_pipelines.iter().map(|pipeline|{
        pipeline.get_bind_group_layout(0)
    }).collect::<Vec<_>>();

    // let bind_group_layout = compute_pipeline.get_bind_group_layout(0);

    let bind_group_entries = [
        vec![
            &param_buffer,
            &frame_buffer,
            &gaussian_buffer, 
            &constant_buffer,
            &processed_buffer, 
        ],
        vec![
            &param_buffer,    
            &circle_buffer, 
            &processed_buffer, 
            &centers_buffer,
            &masses_buffer,
        ],
        vec![
            &param_buffer,
            &processed_buffer, 
            &centers_buffer,
            &masses_buffer,
            &result_buffer,
        ],
    ];

    let bind_group_entries = bind_group_entries
        .iter().map(|group| group.iter().enumerate().map(|(i, &buffer)|
            wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buffer.as_entire_binding()}).collect::<Vec<_>>()
        )
        .collect::<Vec<_>>();
    
    
    let bind_groups = bind_group_layouts.iter().zip(bind_group_entries.iter())
        .map(|(layout, entries)|
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries[..],
        })).collect::<Vec<_>>();
    
    let submit_work = |staging_buffer: &wgpu::Buffer, frame: &[my_dtype]| {
        queue.write_buffer(&staging_buffer, 0, bytemuck::cast_slice(&frame[..]));
        let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(staging_buffer, 0,
            &frame_buffer, 0, size);
        
        compute_pipelines.iter().zip(bind_groups.iter()).for_each(|(pipeline, bind_group)|{
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.set_pipeline(pipeline);
            cpass.dispatch_workgroups(workgroups[0], workgroups[1], 1);
        });

        encoder.copy_buffer_to_buffer(&result_buffer, 0,
            staging_buffer, 0, 2 * size);
        
        queue.submit(Some(encoder.finish()))
    };

    let mut output: Vec<Vec<f32>> = Vec::new();
    
    let frame = frames.next().unwrap();
    let staging_buffer = free_staging_buffers.pop().unwrap();
    in_use_staging_buffers.push_back(staging_buffer);
    let mut old_submission = submit_work(staging_buffer, &frame);

    for frame in frames{
        let staging_buffer = free_staging_buffers.pop().unwrap();
        in_use_staging_buffers.push_back(staging_buffer);
        let new_submission = submit_work(staging_buffer, &frame);
        
        let finished_staging_buffer = in_use_staging_buffers.pop_front().unwrap();
        let mut buffer_slice = finished_staging_buffer.slice(..);
        let (sender, receiver) = 
                futures_intrusive::channel::shared::oneshot_channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        device.poll(wgpu::Maintain::WaitForSubmissionIndex(old_submission));
        receiver.receive().await.unwrap();
        // device.poll(wgpu::Maintain::Wait);
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        output.push(result);
        drop(data);
        finished_staging_buffer.unmap();
        free_staging_buffers.push(finished_staging_buffer);
        old_submission = new_submission;

    }
    let finished_staging_buffer = in_use_staging_buffers.pop_front().unwrap();
    let mut buffer_slice = finished_staging_buffer.slice(..);
    let (sender, receiver) = 
            futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(old_submission));
    receiver.receive().await.unwrap();
    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    output.push(result);
    drop(data);
    finished_staging_buffer.unmap();
    // dbg!(in_use_staging_buffers.len());
    // dbg!(free_staging_buffers.len());
    Ok(output)
}