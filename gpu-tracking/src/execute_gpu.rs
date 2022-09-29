#![allow(warnings)]
use futures;
use futures_intrusive;
type my_dtype = f32;
use pollster::FutureExt;
use std::collections::VecDeque;
use wgpu::{util::DeviceExt, Buffer};
use crate::kernels;

#[derive(Clone, Copy)]
pub struct Params{
    pub pic_dims: [u32; 2],
    pub gauss_dims: [u32; 2],
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

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("first shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader1.wgsl").into()),
    });

    let slice_size = dims.iter().product::<u32>() as usize * std::mem::size_of::<my_dtype>();
    let size = slice_size as wgpu::BufferAddress;

    let mut staging_buffers = Vec::new();
    
    for i in 0..2{
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Staging {}", i).as_str()),
            size,
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
    dbg!(gaussian_kernel.size);

    let gaussian_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&gaussian_kernel.data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params = Params{
        pic_dims: dims,
        gauss_dims: gaussian_kernel.size,
    };
    let param_buffer = unsafe{
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Width Buffer"),
            contents: any_as_u8_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM
        })
    };


    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: "main",
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: frame_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: gaussian_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: param_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: result_buffer.as_entire_binding(),
        },
        ],
    });
    
    let workgroup_size = 16;
    let workgroups: [u32; 2] = 
        dims.iter().map(|&x| (x + workgroup_size - 1) / workgroup_size).collect::<Vec<u32>>().try_into().unwrap();
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    
    
    let submit_work = |staging_buffer: &wgpu::Buffer, frame: &[my_dtype]| {
        queue.write_buffer(&staging_buffer, 0, bytemuck::cast_slice(&frame[..]));
        let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(staging_buffer, 0, &frame_buffer, 0, size);
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("first compute pass");
            cpass.dispatch_workgroups(workgroups[0], workgroups[1], 1); // Number of cells to run, the (x,y,z) size of item being processed
        }
        encoder.copy_buffer_to_buffer(&result_buffer, 0, staging_buffer, 0, size);
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