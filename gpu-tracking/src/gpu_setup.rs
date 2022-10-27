use std::{collections::HashMap, fs::File, io::Read};

use crate::{execute_gpu::TrackingParams, my_dtype};

use pollster::FutureExt;
use wgpu::{Buffer, Device, self, util::DeviceExt};
use winit::window::Window;
use bytemuck;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}



impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}


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
    pub atomic_buffer: Buffer,
    pub particles_buffer: Buffer,
    pub atomic_filtered_buffer: Buffer,
}

pub struct RenderState{
    pub surface: wgpu::Surface,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub texture: wgpu::Texture,
    pub config: wgpu::SurfaceConfiguration,
}

impl RenderState{
    pub fn new(dims: &[u32; 2], device: &Device, window: &Window, surface: wgpu::Surface, adapter: &wgpu::Adapter) -> Self{

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(adapter)[0],
            width: dims[1],
            height: dims[0],
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rendering/render.wgsl").into()),
        });

        let vertices = [
            Vertex { position: [-1.0, -1.0, 0.0], tex_coords: [0.0, 0.0] },
            Vertex { position: [1.0, -1.0, 0.0], tex_coords: [1.0, 0.0] },
            Vertex { position: [1.0, 1.0, 0.0], tex_coords: [1.0, 1.0] },
            Vertex { position: [-1.0, 1.0, 0.0], tex_coords: [0.0, 1.0] },
        ];

        let indices: &[u16] = &[0, 1, 2, 2, 3, 0];

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: dims[1],
                height: dims[0],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.create_view(&wgpu::TextureViewDescriptor::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&device.create_sampler(
                        &wgpu::SamplerDescriptor {
                            address_mode_u: wgpu::AddressMode::ClampToEdge,
                            address_mode_v: wgpu::AddressMode::ClampToEdge,
                            address_mode_w: wgpu::AddressMode::ClampToEdge,
                            mag_filter: wgpu::FilterMode::Linear,
                            min_filter: wgpu::FilterMode::Nearest,
                            mipmap_filter: wgpu::FilterMode::Nearest,
                            ..Default::default()
                        }
                    )),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = indices.len() as u32;

        Self{
            surface,
            pipeline: render_pipeline,
            bind_group: diffuse_bind_group,
            bind_group_layout: texture_bind_group_layout,
            vertex_buffer,
            index_buffer,
            num_indices: 6,
            texture,
            config,
        }
    }
}

pub struct GpuState{
    pub device: Device,
    pub queue: wgpu::Queue,
    pub buffers: GpuBuffers,
    pub pipelines: Vec<(String, (wgpu::ComputePipeline, [u32; 3]))>,
    pub bind_groups: HashMap<String, (u32, wgpu::BindGroup)>,
    // pub workgroups: [u32; 2],
    pub pic_size: usize,
    pub result_read_depth: u64,
    pub pic_byte_size: u64,
    pub render_state: Option<RenderState>,
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
            size: size + 4,
            // size: ((result_buffer_depth + tracking_params.cpu_processed as u64) * size) as u64,
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

    // let sigma = tracking_params.noise_size as f32;
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

    let atomic_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let atomic_filtered_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let particles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
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
        atomic_buffer,
        particles_buffer,
        atomic_filtered_buffer,
    }
}




pub fn setup_state(tracking_params: &TrackingParams, dims: &[u32; 2], debug: bool, window: Option<&Window>) -> GpuState{
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { window.map(|window| instance.create_surface(window)) };
    let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptionsBase{
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: surface.as_ref(),
    })
    .block_on()
    .ok_or(anyhow::anyhow!("Couldn't create the adapter")).unwrap();

    let mut desc = wgpu::DeviceDescriptor::default();
    desc.features = wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
    // desc.limits.max_compute_invocations_per_workgroup = 1024;
    let (device, queue) = adapter
    .request_device(&desc, None)
    .block_on().unwrap();
    
    let render_state = surface.zip(window).map(|(surface, window)| {
        let surface_size = window.inner_size();
        let surface_config = 
            wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface.get_supported_formats(&adapter)[0],
                width: surface_size.width,
                height: surface_size.height,
                present_mode: wgpu::PresentMode::Fifo,
                // alpha_mode: wgpu::CompositeAlphaMode::Auto,
            };
            surface.configure(&device, &surface_config);
            // dims: &[u32; 2], device: &Device, window: &Window, surface: wgpu::Surface, adapter: &wgpu::Adapter
        RenderState::new(&dims, &device, &window, surface, &adapter)
    });



    // let render_texture = wgpu::


    let common_header = include_str!("shaders/params.wgsl");
    
    let workgroup_size_2d = [16u32, 16, 1];
    let wg_dims = [dims[0], dims[1], 1];

    let shaders = HashMap::from([
        ("max_rows", (include_str!("shaders/max_rows.wgsl"), wg_dims, workgroup_size_2d)),
        ("extract_max", (include_str!("shaders/extract_max.wgsl"), wg_dims, workgroup_size_2d)),
        ("preprocess_rows", (include_str!("shaders/preprocess_rows.wgsl"), wg_dims, workgroup_size_2d)),
        ("preprocess_cols", (include_str!("shaders/preprocess_cols.wgsl"), wg_dims, workgroup_size_2d)),
        ("walk", (include_str!("shaders/walk.wgsl"), [10000, 1, 1], [256, 1, 1])),
        ("characterize", (include_str!("shaders/characterize.wgsl"), [10000, 1, 1], [256, 1, 1])),
    ]);


    // let shaders = HashMap::from([
    //     ("max_rows", ("src/shaders/max_rows.wgsl", wg_dims, workgroup_size_2d)),
    //     ("extract_max", ("src/shaders/extract_max.wgsl", wg_dims, workgroup_size_2d)),
    //     ("preprocess_rows", ("src/shaders/preprocess_rows.wgsl", wg_dims, workgroup_size_2d)),
    //     ("preprocess_cols", ("src/shaders/preprocess_cols.wgsl", wg_dims, workgroup_size_2d)),
    //     ("walk", ("src/shaders/walk.wgsl", [10000, 1, 1], [256, 1, 1])),
    //     ("characterize", ("src/shaders/characterize.wgsl", [10000, 1, 1], [256, 1, 1])),
    // ]);

    // let shaders = shaders.into_iter().map(|(name, (shader, dims, group_size))| {
    //     let mut shader_file = File::open(shader).expect(format!("{} not found", shader).as_str());
    //     let mut shader_string = String::new();
    //     shader_file.read_to_string(&mut shader_string).unwrap();
    //     (name, (shader_string, dims, group_size))
    // }).collect::<HashMap<_, _>>();

    let n_workgroups = |dims: &[u32; 3], wgsize: &[u32; 3]| { 
        let mut n_workgroups = [0, 0, 0];
        for i in 0..3 {
            n_workgroups[i] = (dims[i] + wgsize[i] - 1) / wgsize[i];
        }
        n_workgroups
    };

    

    let preprocess_source = |source, wg_size: &[u32; 3]| -> String {
        let mut result = String::new();
        result.push_str(common_header);
        result.push_str(source);
        // if tracking_params.characterize{
        //     result = result.replace("//_feat_char_", "");
        // }
        result.replace("@workgroup_size(_)",
        format!("@workgroup_size({}, {}, {})", wg_size[0], wg_size[1], wg_size[2]).as_str())
    };


    let shaders = shaders.iter().map(|(&name, (source, dims, group_size))|{
        let shader_source = preprocess_source(source, group_size);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        (name, (shader, n_workgroups(dims, group_size)))
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

    let pipelines: Vec<Option<(&str, &(wgpu::ShaderModule, [u32; 3]))>> = match debug{
        // false => vec![
        //     // ("rows", &shaders[0]),
        //     // ("finish", &shaders[0]),
        //     ("preprocess", 0, &shaders["preprocess_backup"]),
        //     ("centers", 0, &shaders["centers"]),
        //     ("walk", 0, &shaders["walk"]),
        // ],
        _ => vec![
            Some(("pp_rows", &shaders["preprocess_rows"])),
            Some(("pp_cols", &shaders["preprocess_cols"])),
            Some(("max_row", &shaders["max_rows"])),
            Some(("extract_max", &shaders["extract_max"])),
            Some(("walk", &shaders["walk"])),
            if (tracking_params.characterize) {Some(("characterize", &shaders["characterize"]))} else {None},
            // ("centers", 0, &shaders["centers"]),
            // ("centers", 0, &shaders["centers_outside_parens"]),
            // ("max_row", 0, &shaders["max_rows"]),
            // ("walk", 0, &shaders["walk_cols"]),
        ],
    };

    let compute_pipelines = pipelines.iter().flatten().map(|(name, (shader, wg_n))|{
        (name.to_string(), (device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: shader,
        entry_point: "main",
        }), *wg_n))
    }).collect::<Vec<_>>();

    let bind_group_layouts = compute_pipelines.iter()
    // .zip(pipelines.iter())
    .map(|(name, (pipeline, wg_n))|{
        (name.as_str(), (0, pipeline.get_bind_group_layout(0)))
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
            if tracking_params.characterize {Some((5, &buffers.frame_buffer))} else {None},
            if tracking_params.characterize {Some((6, &buffers.result_buffer))} else {None},
        ]),
        ("max_row", vec![
            Some((0, &buffers.param_buffer)),
            Some((1, &buffers.processed_buffer)), 
            Some((2, &buffers.max_rows)),
        ]),
        ("extract_max", vec![
            Some((0, &buffers.param_buffer)),
            Some((1, &buffers.processed_buffer)), 
            Some((2, &buffers.max_rows)),
            Some((3, &buffers.atomic_buffer)),
            Some((4, &buffers.particles_buffer))
            // Some((3, &buffers.centers_buffer)),
            // Some((4, &buffers.masses_buffer)),
            // Some((5, &buffers.result_buffer)),
            ]),
        ("walk", vec![
            Some((0, &buffers.param_buffer)),
            Some((1, &buffers.processed_buffer)), 
            Some((2, &buffers.particles_buffer)),
            Some((3, &buffers.atomic_buffer)),
            Some((4, &buffers.atomic_filtered_buffer)),
            Some((5, &buffers.result_buffer)),
        ]),

        ("characterize", vec![
            Some((0, &buffers.param_buffer)),
            Some((1, &buffers.processed_buffer)), 
            Some((2, &buffers.frame_buffer)),
            // Some((3, &buffers.atomic_buffer)),
            Some((3, &buffers.atomic_filtered_buffer)),
            Some((4, &buffers.result_buffer)),
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
        // workgroups,
        pic_size,
        result_read_depth,
        pic_byte_size: size,
        render_state,
    }
}

