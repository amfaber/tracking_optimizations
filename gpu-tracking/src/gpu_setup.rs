use std::{collections::HashMap, fs::File, io::Read};

use crate::{execute_gpu::TrackingParams, kernels, my_dtype};

use pollster::FutureExt;
use wgpu::{Buffer, Device, self, util::DeviceExt};

pub struct GpuParams{
    pub pic_dims: [u32; 2],
    pub composite_dims: [u32; 2],
    pub sigma: f32,
    // pub constant_dims: [u32; 2],
    pub circle_dims: [u32; 2],
    pub dilation_dims: [u32; 2],
    pub max_iterations: u32,
    pub shift_threshold: f32,
    pub minmass: f32,
}
pub struct GpuBuffers{
    pub staging_buffers: Vec<Buffer>,
    pub frame_buffer: Buffer,
    // pub composite_buffer: Buffer,
    // pub gauss_1d_buffer: Buffer,
    pub processed_buffer: Buffer,
    pub centers_buffer: Buffer,
    // pub circle_buffer: Buffer,
    pub masses_buffer: Buffer,
    pub result_buffer: Buffer,
    pub param_buffer: Buffer,
    pub max_rows: Buffer,
}

pub struct GpuState{
    pub device: Device,
    pub queue: wgpu::Queue,
    pub buffers: GpuBuffers,
    pub pipelines: Vec<(String, wgpu::ComputePipeline)>,
    pub bind_groups: HashMap<String, (u32, wgpu::BindGroup)>,
    pub workgroups: [u32; 2],
    pub pic_size: usize,
    pub result_read_depth: u64,
    pub pic_byte_size: u64,
}

fn gpuparams_from_tracking_params(params: TrackingParams, pic_dims: [u32; 2]) -> GpuParams {
    let kernel_size = params.smoothing_size;
    let circle_size = params.diameter;
    let dilation_size = (2. * params.separation as f32 / (2 as f32).sqrt()) as u32;
    GpuParams{
        pic_dims,
        composite_dims: [kernel_size, kernel_size],
        sigma: params.noise_size,
        // constant_dims: [kernel_size, kernel_size],
        circle_dims: [circle_size, circle_size],
        dilation_dims: [dilation_size, dilation_size],
        max_iterations: params.max_iterations,
        shift_threshold: 0.6,
        minmass: params.minmass,
    }
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        ::std::mem::size_of::<T>(),
    )
}

pub fn setup_buffers(tracking_params: &TrackingParams,
    device: &wgpu::Device,
    size: u64,
    dims: &[u32; 2],
    ) -> GpuBuffers{
    let result_buffer_depth = if tracking_params.characterize { 7 } else { 3 };
    let params = gpuparams_from_tracking_params(tracking_params.clone(), *dims);
    let mut staging_buffers = Vec::new();
    for i in 0..2{
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Staging {}", i).as_str()),
            size: ((result_buffer_depth + tracking_params.cpu_processed as u64) * size) as u64,
            usage: wgpu::BufferUsages::COPY_SRC 
            | wgpu::BufferUsages::COPY_DST 
            | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        staging_buffers.push(staging_buffer);
    }
    // let mut free_staging_buffers = staging_buffers.iter().collect::<Vec<&wgpu::Buffer>>();
    // let mut in_use_staging_buffers = VecDeque::new();
    let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let sigma = tracking_params.noise_size as f32;
    // let gaussian_kernel = kernels::Kernel::tp_gaussian(sigma, 4.);
    // let composite_kernel = kernels::Kernel::composite_kernel(sigma, params.composite_dims);
    // let composite_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //     label: None,
    //     contents: bytemuck::cast_slice(&composite_kernel.data),
    //     usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    // });

    // let noise_size = tracking_params.noise_size;
    // let constant_kernel = kernels::Kernel::rolling_average([noise_size, noise_size]);
    // let constant_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //     label: None,
    //     contents: bytemuck::cast_slice(&constant_kernel.data),
    //     usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    // });

    let processed_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let centers_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (2 * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    // let r = tracking_params.diameter / 2;
    // let circle_kernel = kernels::Kernel::circle_mask(r);
    // let circle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //     label: None,
    //     contents: bytemuck::cast_slice(&circle_kernel.data),
    //     usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    // });

    let masses_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (result_buffer_depth * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    

    let param_buffer = unsafe{
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Width Buffer"),
            contents: any_as_u8_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM
        })
    };

    // let gauss_1d_kernel = kernels::Kernel::gauss_1d(sigma, params.composite_dims[0]);
    // let gauss_1d_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //     label: None,
    //     contents: bytemuck::cast_slice(&gauss_1d_kernel.data),
    //     usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    // });

    let max_rows = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });


    GpuBuffers{
        staging_buffers,
        frame_buffer,
        // composite_buffer,
        // gauss_1d_buffer,
        processed_buffer,
        centers_buffer,
        // circle_buffer,
        masses_buffer,
        result_buffer,
        param_buffer,
        max_rows,
    }
}



pub fn setup_state(tracking_params: &TrackingParams, dims: &[u32; 2], debug: bool) -> GpuState{
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptionsBase{
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })
    .block_on()
    .ok_or(anyhow::anyhow!("Couldn't create the adapter")).unwrap();

    let mut desc = wgpu::DeviceDescriptor::default();
    desc.features = wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
    // desc.limits.max_compute_invocations_per_workgroup = 1024;
    let (device, queue) = adapter
    .request_device(&desc, None)
    .block_on().unwrap();

    
    let common_header = include_str!("shaders/params.wgsl");


    let shaders = HashMap::from([
        // ("proprocess_backup", "src/shaders/another_backup_preprocess.wgsl"),
        ("centers", include_str!("shaders/centers.wgsl")),
        // ("centers_outside_parens", include_str!("shaders/centers_outside_parens.wgsl")),
        ("max_rows", include_str!("shaders/max_rows.wgsl")),
        ("walk", include_str!("shaders/walk.wgsl")),
        ("walk_cols", include_str!("shaders/walk_cols.wgsl")),
        ("preprocess_rows", include_str!("shaders/preprocess_rows.wgsl")),
        ("preprocess_cols", include_str!("shaders/preprocess_cols.wgsl")),
    ]);

    // let shaders = shaders.iter().map(|(&name, shader)| {
    //     let mut shader_file = File::open(shader).unwrap();
    //     let mut shader_string = String::new();
    //     shader_file.read_to_string(&mut shader_string).unwrap();
    //     (name, shader_string)
    // }).collect::<HashMap<_, _>>();

    let workgroup_size = [16, 16, 1];
    let workgroups: [u32; 2] = 
        dims.iter().zip(workgroup_size)
        .map(|(&x, size)| (x + size - 1) / size).collect::<Vec<u32>>().try_into().unwrap();

    let preprocess_source = |source| -> String {
        let mut result = String::new();
        result.push_str(common_header);
        result.push_str(source);
        if tracking_params.characterize{
            result = result.replace("//_feat_char_", "");
        }
        result.replace("@workgroup_size(_)",
        format!("@workgroup_size({}, {}, {})", workgroup_size[0], workgroup_size[1], workgroup_size[2]).as_str())
    };


    let shaders = shaders.iter().map(|(&name, source)|{
        let shader_source = preprocess_source(source);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        (name, shader)
    }).collect::<HashMap<_, _>>();
    let pic_size = dims.iter().product::<u32>() as usize;
    let result_read_depth: u64 = match debug{
        false => match tracking_params.characterize{
            false => 3,
            true => 7,
        },
        true => 2,
    };
    let slice_size = pic_size * std::mem::size_of::<my_dtype>();
    let size = slice_size as wgpu::BufferAddress;
    
    
    let buffers = setup_buffers(&tracking_params, &device, size, dims);

    let pipelines = match debug{
        // false => vec![
        //     // ("rows", &shaders[0]),
        //     // ("finish", &shaders[0]),
        //     ("preprocess", 0, &shaders["preprocess_backup"]),
        //     ("centers", 0, &shaders["centers"]),
        //     ("walk", 0, &shaders["walk"]),
        // ],
        _ => vec![
            ("pp_rows", 0, &shaders["preprocess_rows"]),
            ("pp_cols", 0, &shaders["preprocess_cols"]),
            ("centers", 0, &shaders["centers"]),
            // ("centers", 0, &shaders["centers_outside_parens"]),
            ("max_row", 0, &shaders["max_rows"]),
            ("walk", 0, &shaders["walk_cols"]),
        ],
    };

    let compute_pipelines = pipelines.iter().map(|(name, group, shader)|{
        (name.to_string(), device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: shader,
        entry_point: "main",
        }))
    }).collect::<Vec<_>>();

    let bind_group_layouts = compute_pipelines.iter()
    .zip(pipelines.iter())
    .map(|((name, pipeline), (entry, group, shader))|{
        (name.as_str(), (*group, pipeline.get_bind_group_layout(*group)))
    }).collect::<HashMap<_, _>>();


    let bind_group_entries = 
        HashMap::from([
        ("pp_rows", vec![
            Some((0, &buffers.param_buffer)),
            Some((1, &buffers.frame_buffer)),
            // (2, &buffers.gauss_1d_buffer),
            Some((3, &buffers.centers_buffer)),
        ]),
        ("pp_cols", vec![
            Some((0, &buffers.param_buffer)),
            // (1, &buffers.gauss_1d_buffer),
            Some((2, &buffers.centers_buffer)),
            Some((3, &buffers.processed_buffer)), 
        ]),
        ("centers", vec![
            Some((0, &buffers.param_buffer)),    
            Some((2, &buffers.processed_buffer)), 
            Some((3, &buffers.centers_buffer)),
            Some((4, &buffers.masses_buffer)),
            if (tracking_params.characterize) {Some((5, &buffers.frame_buffer))} else {None},
            if (tracking_params.characterize) {Some((6, &buffers.result_buffer))} else {None},
        ]),
        ("max_row", vec![
            Some((0, &buffers.param_buffer)),
            Some((1, &buffers.processed_buffer)), 
            Some((2, &buffers.max_rows)),
        ]),
        ("walk", vec![
            Some((0, &buffers.param_buffer)),
            Some((1, &buffers.processed_buffer)), 
            Some((2, &buffers.max_rows)),
            Some((3, &buffers.centers_buffer)),
            Some((4, &buffers.masses_buffer)),
            Some((5, &buffers.result_buffer)),
        ]),
    ]);

    let bind_group_entries = bind_group_entries
        .iter().map(|(&name, group)| (name, group.iter().flatten().map(|(i, buffer)|
            wgpu::BindGroupEntry {
            binding: *i as u32,
            resource: buffer.as_entire_binding()}).collect::<Vec<_>>())
        )
        .collect::<HashMap<_, _>>();
    
    
    let bind_groups = bind_group_layouts.iter()
        .map(|(&name, (group, layout))|{
        let entries = &bind_group_entries[name];
        (name.to_string(), (*group, device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries[..],
        })))}
    ).collect::<HashMap<_, _>>();

    GpuState{
        device,
        queue,
        buffers,
        pipelines: compute_pipelines,
        bind_groups,
        workgroups,
        pic_size,
        result_read_depth,
        pic_byte_size: size,
    }
}

