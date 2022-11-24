use std::{collections::HashMap, fs::File, io::Read, rc::Rc};

use crate::{my_dtype};

use pollster::FutureExt;
use wgpu::{Buffer, Device, self, util::{DeviceExt, BufferInitDescriptor}, ComputePipeline, BindGroupLayout};
use wgpu_fft::{self, fft::{FftPlan, FftPass}, FullComputePass, infer_compute_bindgroup_layout};

#[derive(Clone)]
pub struct TrackpyParams{
    
}

#[derive(Clone)]
pub struct LogParams{
    pub min_sigma: my_dtype,
    pub max_sigma: my_dtype,
    pub n_sigma: usize,
    pub log_spacing: bool,
}

#[derive(Clone)]
pub enum Style{
    Trackpy(TrackpyParams),
    Log(LogParams),
}


#[derive(Clone)]
pub struct TrackingParams{
    pub diameter: u32,
    pub minmass: f32,
    pub maxsize: f32,
    pub separation: u32,
    pub noise_size: f32,
    pub smoothing_size: u32,
    pub threshold: f32,
    pub invert: bool,
    pub percentile: f32,
    pub topn: u32,
    pub preprocess: bool,
    pub max_iterations: u32,
    pub characterize: bool,
    pub filter_close: bool,
    pub search_range: Option<my_dtype>,
    pub memory: Option<usize>,
    // pub cpu_processed: bool,
    pub sig_radius: Option<my_dtype>,
    pub bg_radius: Option<my_dtype>,
    pub gap_radius: Option<my_dtype>,
    pub varcheck: Option<my_dtype>,
    pub style: Style,
}

impl Default for TrackingParams{
    fn default() -> Self {
        TrackingParams{
            diameter: 9,
            minmass: 0.,
            maxsize: 0.0,
            separation: 11,
            noise_size: 1.,
            smoothing_size: 9,
            threshold: 0.0,
            invert: false,
            percentile: 0.,
            topn: 0,
            preprocess: true,
            max_iterations: 10,
            characterize: false,
            filter_close: true,
            search_range: None,
            memory: None,
            // cpu_processed: true,
            sig_radius: None,
            bg_radius: None,
            gap_radius: None,
            varcheck: None,
            style: Style::Trackpy(TrackpyParams{

            })
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
    pub margin: i32,
    pub var_factor: f32,
}
pub struct TrackpyGpuBuffers{
    pub staging_buffers: Vec<Buffer>,
    pub frame_buffer: Buffer,
    pub result_buffer: Buffer,
    pub atomic_buffer: Buffer,
    pub particles_buffer: Buffer,
    pub atomic_filtered_buffer: Buffer,
    
    pub processed_buffer: Buffer,
    pub centers_buffer: Buffer,
    pub masses_buffer: Buffer,
    pub max_rows: Buffer,
    pub param_buffer: Buffer,
}

pub struct LogGpuBuffers{
    pub staging_buffers: Vec<Buffer>,
    pub frame_buffer: Buffer,
    pub result_buffer: Buffer,
    pub atomic_buffer: Buffer,
    pub particles_buffer: Buffer,
    pub atomic_filtered_buffer: Buffer,
    pub param_buffer: Buffer,
    
    pub raw_padded: (Buffer, (FullComputePass, [u32; 4]), FftPass<2>),
    pub logspace_buffers: Vec<(Buffer, Vec<FullComputePass>, FftPass<2>)>,
    pub filter_buffers: Vec<(Buffer, FftPass<2>, FullComputePass)>,
    pub laplacian_buffer: (Buffer, FftPass<2>),
    pub unpadded_laplace: (Buffer, [u32; 2]),
    pub global_max: Buffer,
}


pub struct TrackpyGpuState{
    pub order: Vec<String>,
    pub buffers: TrackpyGpuBuffers,
}


pub struct LogGpuState{
    pub fftplan: FftPlan,
    pub buffers: LogGpuBuffers,
    pub sigmas: Vec<my_dtype>,
}

pub enum GpuBuffers{
    Trackpy(TrackpyGpuBuffers),
    Log(LogGpuBuffers)
}

pub struct GpuState{
    pub device: Device,
    pub queue: wgpu::Queue,
    // pub bind_groups: HashMap<String, Vec<wgpu::BindGroup>>,
    // pub pipelines: HashMap<String, (ComputePipeline, wgpu::BindGroupLayout, [u32; 3])>,
    pub passes: HashMap<String, Vec<FullComputePass>>,
    pub pic_size: usize,
    pub result_read_depth: u64,
    pub pic_byte_size: u64,
    pub flavor: GpuStateFlavor,
}

pub enum GpuStateFlavor{
    Trackpy(TrackpyGpuState),
    Log(LogGpuState),
}

// pub enum GpuStateFlavor{
//     Trackpy(TrackpyGpuState),
//     Log(LogGpuState),
// }

impl GpuState{
    pub fn staging_buffers(&self) -> &Vec<Buffer>{
        match self.flavor{
            GpuStateFlavor::Trackpy(ref flavor) => &flavor.buffers.staging_buffers,
            GpuStateFlavor::Log(ref flavor) => &flavor.buffers.staging_buffers,
        }
    }
    
    pub fn atomic_filtered_buffer(&self) -> &Buffer{
        match self.flavor{
            GpuStateFlavor::Trackpy(ref flavor) => &flavor.buffers.atomic_filtered_buffer,
            GpuStateFlavor::Log(ref flavor) => &flavor.buffers.atomic_filtered_buffer,
        }
    }

    pub fn frame_buffer(&self) -> &Buffer{
        match self.flavor{
            GpuStateFlavor::Trackpy(ref flavor) => &flavor.buffers.frame_buffer,
            GpuStateFlavor::Log(ref flavor) => &flavor.buffers.frame_buffer,
        }
    }

    pub fn result_buffer(&self) -> &Buffer{
        match self.flavor{
            GpuStateFlavor::Trackpy(ref flavor) => &flavor.buffers.result_buffer,
            GpuStateFlavor::Log(ref flavor) => &flavor.buffers.result_buffer,
        }
    }

    pub fn atomic_buffer(&self) -> &Buffer{
        match self.flavor{
            GpuStateFlavor::Trackpy(ref flavor) => &flavor.buffers.atomic_buffer,
            GpuStateFlavor::Log(ref flavor) => &flavor.buffers.atomic_buffer,
        }
    }

    pub fn particles_buffer(&self) -> &Buffer{
        match self.flavor{
            GpuStateFlavor::Trackpy(ref flavor) => &flavor.buffers.particles_buffer,
            GpuStateFlavor::Log(ref flavor) => &flavor.buffers.particles_buffer,
        }
    }


    fn setup_buffers(&self){

        match self.flavor{
            GpuStateFlavor::Trackpy(ref flavor) => {

            },
            GpuStateFlavor::Log(ref flavor) => {
                let buffers = &flavor.buffers;
                
                let mut encoder = self.device.create_command_encoder(&Default::default());
                let (pad_pass, push_constants) = flavor.fftplan.pad_pass(&self.device, &buffers.unpadded_laplace.0, &buffers.laplacian_buffer.0, &buffers.unpadded_laplace.1);
                pad_pass.execute(&mut encoder, bytemuck::cast_slice(&push_constants));


                let laplace_fft = &buffers.laplacian_buffer.1;
                laplace_fft.execute(&mut encoder, false, false);


                let gauss_size = (flavor.sigmas[flavor.sigmas.len() - 1] * 4. + 0.5) as u32 * 2 + 1;
                let iterator = buffers.filter_buffers.iter().zip(self.passes["init_gauss"].iter()).zip(&flavor.sigmas);
                for (((_filter_buffer, fft, lap_convolve), init), sigma) in iterator{
                    // let push_constants = panic!("figure out push_constants. Limits to the gaussians are still missing");
                    let push_constants = (*sigma, gauss_size, gauss_size);
                    
                    init.execute(&mut encoder, unsafe{ any_as_u8_slice(&push_constants) });

                    fft.execute(&mut encoder, false, false);
                    lap_convolve.execute(&mut encoder, &[]);
                }
                self.queue.submit(Some(encoder.finish()));
            }
        }
    }
}


fn gpuparams_from_tracking_params(params: &TrackingParams, pic_dims: [u32; 2]) -> GpuParams {
    let kernel_size = params.smoothing_size;
    let circle_size = params.diameter;
    let dilation_size = (2. * params.separation as f32 / (2 as f32).sqrt()) as u32;
    let margin = vec![params.diameter / 2, params.separation / 2 - 1, params.smoothing_size / 2].iter().max().unwrap().clone() as i32;

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
        margin,
        var_factor: params.varcheck.unwrap_or(0.0),
    }
}

pub unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        ::std::mem::size_of::<T>(),
    )
}

pub fn create_buffers(tracking_params: &TrackingParams,
    device: &wgpu::Device,
    size: u64,
    dims: &[u32; 2],
    fftplan: Option<&FftPlan>,
    ) -> GpuBuffers {
    
    let result_buffer_depth = if tracking_params.characterize { 7 } else { 3 };
    let mut staging_buffers = Vec::new();
    for i in 0..2{
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Staging {}", i).as_str()),
            size: 8*size,
            // size: ((result_buffer_depth + tracking_params.cpu_processed as u64) * size) as u64,
            usage: wgpu::BufferUsages::COPY_SRC 
            | wgpu::BufferUsages::COPY_DST 
            | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        staging_buffers.push(staging_buffer);
    }
    
    let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
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
    let params = gpuparams_from_tracking_params(&tracking_params, *dims);

    let param_buffer = unsafe{
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Width Buffer"),
            contents: any_as_u8_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM
        })
    };
    
    match &tracking_params.style{
    Style::Trackpy(styleparams) => {
        
        let processed_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        let centers_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (2 * size) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        
        let masses_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        
    
        let max_rows = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        GpuBuffers::Trackpy(TrackpyGpuBuffers{
            staging_buffers,
            frame_buffer,
            processed_buffer,
            centers_buffer,
            masses_buffer,
            result_buffer,
            param_buffer,
            max_rows,
            atomic_buffer,
            particles_buffer,
            atomic_filtered_buffer,
        })
    },
    Style::Log(styleparams) => {
        // let GpuStateFlavor::Log(flavor) = flavor else { panic!() };

        let fftplan = fftplan.unwrap();

        let laplacian: [f32; 9] = [
            -1., -1., -1.,
            -1.,  8., -1.,
            -1., -1., -1.,
        ];

        let laplacian_buffer = device.create_buffer_init(&BufferInitDescriptor{
            label: None,
            contents: bytemuck::cast_slice(&laplacian),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let padded_laplacian_buffer = fftplan.create_buffer(device, false);

        // let mut encoder = device.create_command_encoder(&Default::default());
        // fftplan.pad(device, &mut encoder, &laplacian_buffer, &padded_laplacian_buffer, &[3, 3]);
        
        let laplacian_fft_pass = fftplan.fft_pass(&padded_laplacian_buffer, device);

        let filter_buffers: Vec<_> = (0..styleparams.n_sigma).map(|_|{
            let buffer = fftplan.create_buffer(device, false);
            let pass = fftplan.fft_pass(&buffer, device);
            let laplace_convolve = fftplan.inplace_spectral_convolution_pass(device, &buffer, &padded_laplacian_buffer);
            (buffer, pass, laplace_convolve)
            // buffer
        }).collect();

        let raw_padded = fftplan.create_buffer(device, false);
        let raw_pad_pass = fftplan.pad_pass(device, &frame_buffer, &raw_padded, dims);
        let raw_fft_pass = fftplan.fft_pass(&raw_padded, device);
        let raw_padded = (raw_padded, raw_pad_pass, raw_fft_pass);


        let logspace_buffers: Vec<_> = (0..3)
        .map(|_|{
            let log_space_buffer = fftplan.create_buffer(device, false);
            let fftpass = fftplan.fft_pass(&log_space_buffer, device);
            let convolutionpasses = filter_buffers.iter().map(|(filterbuffer, _fftpass, _lap_conv)|{
                fftplan.spectral_convolution_pass(device, &raw_padded.0, filterbuffer, &log_space_buffer)
            }).collect();
            (log_space_buffer, convolutionpasses, fftpass)
            // buffer
        }).collect();

        let global_max = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });


        GpuBuffers::Log(LogGpuBuffers{
            staging_buffers,
            frame_buffer,
            result_buffer,
            atomic_buffer,
            particles_buffer,
            atomic_filtered_buffer,
            param_buffer,

            raw_padded,
            logspace_buffers,
            filter_buffers,
            laplacian_buffer: (padded_laplacian_buffer, laplacian_fft_pass),
            unpadded_laplace: (laplacian_buffer, [3, 3]),
            global_max,
        })
    }
    }
}



pub fn setup_state(
    tracking_params: &TrackingParams,
    dims: &[u32; 2],
    debug: bool,
    characterize_new_points: bool,
    ) -> GpuState {
    
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptionsBase{
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })
    .block_on().expect("Couldn't create adapter");

    let mut desc = wgpu::DeviceDescriptor::default();
    desc.features = wgpu::Features::MAPPABLE_PRIMARY_BUFFERS | wgpu::Features::PUSH_CONSTANTS;
    desc.limits.max_push_constant_size = 16;
    // desc.limits.max_compute_invocations_per_workgroup = 1024;
    let (device, queue) = adapter
    .request_device(&desc, None)
    .block_on().expect("Couldn't create device and queue");
    
    
    let workgroup_size2d = [16u32, 16, 1];
    let workgroup_size1d = [256u32, 1, 1];

    let n_workgroups = |dims: &[u32; 3], wgsize: &[u32; 3]| { 
        let mut n_workgroups = [0, 0, 0];
        for i in 0..3 {
            n_workgroups[i] = (dims[i] + wgsize[i] - 1) / wgsize[i];
        }
        n_workgroups
    };

    let result_read_depth: u64 = match debug{
        false => match tracking_params.characterize{
            false => 3,
            true => 7,
        },
        true => 2,
    };
    
    let pic_size = dims.iter().product::<u32>() as usize;
    let slice_size = pic_size * std::mem::size_of::<my_dtype>();
    let size = slice_size as wgpu::BufferAddress;
    

    let preprocess_source = |source, wg_size: &[u32; 3], common_header| -> String {
        let mut result = String::new();
        result.push_str(common_header);
        result.push_str(source);
        if characterize_new_points{
            result = result.replace("//_feat_characterize_points ", "");
        }
        if tracking_params.varcheck.is_some(){
            result = result.replace("//_feat_varcheck ", "");
        }
        if let Style::Log(ref params) = tracking_params.style{
            result = result.replace("//_feat_LoG ", "");
        }
        result.replace("@workgroup_size(_)",
        format!("@workgroup_size({}, {}, {})", wg_size[0], wg_size[1], wg_size[2]).as_str())
    };

    let wg_dims = [dims[0], dims[1], 1];
    let common_header = include_str!("shaders/params.wgsl");
    
    let (flavor, shaders, common_header) = match &tracking_params.style{
        
    Style::Trackpy(params) => {
            
        let mut shaders = HashMap::from([
            ("max_rows", (include_str!("shaders/max_rows.wgsl"), wg_dims, workgroup_size2d)),
            ("extract_max", (include_str!("shaders/extract_max.wgsl"), wg_dims, workgroup_size2d)),
            ("preprocess_rows", (include_str!("shaders/preprocess_rows.wgsl"), wg_dims, workgroup_size2d)),
            ("preprocess_cols", (include_str!("shaders/preprocess_cols.wgsl"), wg_dims, workgroup_size2d)),
            ("walk", (include_str!("shaders/walk.wgsl"), [10000, 1, 1], workgroup_size1d)),
            ("characterize", (include_str!("shaders/characterize.wgsl"), [10000, 1, 1], workgroup_size1d)),
        ]);
            
        let order = vec![
            Some("preprocess_rows".to_string()),
            Some("preprocess_cols".to_string()),
            if (!characterize_new_points) { Some("max_rows".to_string()) } else {None},
            if (!characterize_new_points) { Some("extract_max".to_string()) } else {None},
            if (!characterize_new_points) { Some("walk".to_string()) } else {None},
            if (tracking_params.characterize | characterize_new_points) { Some("characterize".to_string()) } else {None},
        ];

        let order = order.into_iter().flatten().collect();
        
        let GpuBuffers::Trackpy(buffers) = create_buffers(&tracking_params, &device, size * 2, dims, None) else {unreachable!()};
        
        let flavor = GpuStateFlavor::Trackpy(TrackpyGpuState {
            order,
            buffers,
        });


        (flavor, shaders, common_header)
    },
    Style::Log(params) => {
        // let common_header = "";

        let fftshaders = wgpu_fft::fft::compile_shaders(&device, 
            Some(&workgroup_size2d),
            Some(&workgroup_size1d));
        let gauss_size = (params.max_sigma * 4. + 0.5) as u32 * 2 + 1;
        let fft_shape = wgpu_fft::fft::get_shape(dims, &[gauss_size, gauss_size]);
        let fftplan = Some(wgpu_fft::fft::FftPlan::create(&fft_shape, fftshaders, &device, &queue));
        let init_wg_dims = [gauss_size, gauss_size, 1];
        
        let shaders = HashMap::from([

            ("init_gauss", (include_str!("shaders/log_style/init_gauss.wgsl"), init_wg_dims, workgroup_size1d)),
            ("logspace_max", (include_str!("shaders/log_style/logspace_max.wgsl"), wg_dims, workgroup_size1d)),
            ("walk", (include_str!("shaders/walk.wgsl"), [10000, 1, 1], workgroup_size1d)),
            // ("characterize", (include_str!("shaders/characterize.wgsl"), [10000, 1, 1], workgroup_size1d)),
        ]);
        
        
        let GpuBuffers::Log(buffers) = create_buffers(&tracking_params, &device, size, dims, fftplan.as_ref()) else {unreachable!()};
        
        let sigmas = if params.log_spacing{
            let mut start = params.min_sigma.log(10.);
            let end = params.max_sigma.log(10.);
            let diff = (end - start) / params.n_sigma as my_dtype;
            let mut out = (0..(params.n_sigma - 1)).map(|_| {let temp = start; start += diff; temp.powf(10.)}).collect::<Vec<_>>();
            out.push(start);
            out
        } else{
            let mut start = params.min_sigma;
            let end = params.max_sigma;
            let diff = (end - start) / params.n_sigma as my_dtype;
            let mut out = (0..(params.n_sigma - 1)).map(|_| {let temp = start; start += diff; temp}).collect::<Vec<_>>();
            out.push(start);
            out
        };
        let flavor = GpuStateFlavor::Log(LogGpuState{
            fftplan: fftplan.unwrap(),
            buffers,
            sigmas,
        });
        (flavor, shaders, common_header)
    },
    };
    
    

    /* Dynamically load shaders to avoid recompilation when debugging. This requires the shaders to be
    in the directory layout as in the repo.
    let shaders = HashMap::from([
        ("max_rows", ("src/shaders/max_rows.wgsl", wg_dims, workgroup_size2d)),
        ("extract_max", ("src/shaders/extract_max.wgsl", wg_dims, workgroup_size2d)),
        ("preprocess_rows", ("src/shaders/preprocess_rows.wgsl", wg_dims, workgroup_size2d)),
        ("preprocess_cols", ("src/shaders/preprocess_cols.wgsl", wg_dims, workgroup_size2d)),
        ("walk", ("src/shaders/walk.wgsl", [10000, 1, 1], [256, 1, 1])),
        ("characterize", ("src/shaders/characterize.wgsl", [10000, 1, 1], [256, 1, 1])),
    ]);

    let shaders = shaders.into_iter().map(|(name, (shader, dims, group_size))| {
        let mut shader_file = File::open(shader).expect(format!("{} not found", shader).as_str());
        let mut shader_string = String::new();
        shader_file.read_to_string(&mut shader_string).unwrap();
        (name, (shader_string, dims, group_size))
    }).collect::<HashMap<_, _>>();
    */

    let shaders = shaders.into_iter().map(|(name, (source, dims, group_size))|{
        let shader_source = preprocess_source(source, &group_size, common_header);
        let bindgrouplayout = infer_compute_bindgroup_layout(&device, &shader_source);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        (name, (shader, n_workgroups(&dims, &group_size), bindgrouplayout))
    }).collect::<HashMap<_, _>>();
    
    let pipelines = shaders.into_iter().map(|(name, (shader, wg_n, bindgrouplayout))|{
        // let dummypipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        //     label: None,
        //     layout: None,
        //     module: &shader,
        //     entry_point: "main",
        //     });
        // let bindgrouplayout = dummypipeline.get_bind_group_layout(0);
        let pipelinelayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: None,
            bind_group_layouts: &[&bindgrouplayout],
            push_constant_ranges: &[wgpu::PushConstantRange{
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: None,
            layout: Some(&pipelinelayout),
            module: &shader,
            entry_point: "main",
        });

        (name.to_string(), (Rc::new(pipeline), bindgrouplayout, wg_n))
    }).collect::<HashMap<_, _>>();


    
    
    
    let bind_group_entries = match flavor{
    GpuStateFlavor::Trackpy(ref flavor) => {
        let buffers = &flavor.buffers;
            // let bind_group_entries = 
            HashMap::from([
            ("preprocess_rows", vec![vec![
                Some((0, &buffers.param_buffer)),
                Some((1, &buffers.frame_buffer)),
                Some((3, &buffers.centers_buffer)),
            ]]),
            ("preprocess_cols", vec![vec![
                Some((0, &buffers.param_buffer)),
                Some((2, &buffers.centers_buffer)),
                Some((3, &buffers.processed_buffer)), 
            ]]),
            // ("centers", vec![
            //     Some((0, &buffers.param_buffer)),    
            //     Some((2, &buffers.processed_buffer)), 
            //     Some((3, &buffers.centers_buffer)),
            //     Some((4, &buffers.masses_buffer)),
            //     if tracking_params.characterize {Some((5, &buffers.frame_buffer))} else {None},
            //     if tracking_params.characterize {Some((6, &buffers.result_buffer))} else {None},
            // ]),
            ("max_rows", vec![vec![
                Some((0, &buffers.param_buffer)),
                Some((1, &buffers.processed_buffer)), 
                Some((2, &buffers.max_rows)),
            ]]),
            ("extract_max", vec![vec![
                Some((0, &buffers.param_buffer)),
                Some((1, &buffers.processed_buffer)), 
                Some((2, &buffers.max_rows)),
                Some((3, &buffers.atomic_buffer)),
                Some((4, &buffers.particles_buffer))
            ]]),
            ("walk", vec![vec![
                Some((0, &buffers.param_buffer)),
                Some((1, &buffers.processed_buffer)), 
                Some((2, &buffers.particles_buffer)),
                Some((3, &buffers.atomic_buffer)),
                Some((4, &buffers.atomic_filtered_buffer)),
                Some((5, &buffers.result_buffer)),
                Some((6, &buffers.frame_buffer)),
                // if tracking_params.varcheck.is_some() {Some((6, &buffers.frame_buffer))} else {None},
            ]]),

            ("characterize", vec![vec![
                Some((0, &buffers.param_buffer)),
                Some((1, &buffers.processed_buffer)), 
                Some((2, &buffers.frame_buffer)),
                Some((3, &buffers.atomic_filtered_buffer)),
                Some((4, &buffers.result_buffer)),
            ]]),
                
        ])
    },

    GpuStateFlavor::Log(ref flavor) => {
        let buffers = &flavor.buffers;
        HashMap::from([
            ("init_gauss", 
                buffers.filter_buffers.iter().map(|(buffer)|{
                    vec![
                        Some((0, &flavor.fftplan.gpu_side_params)),
                        Some((1, &buffer.0)),
                    ]
                }).collect()),
            
            ("logspace_max", (0..3).map(|i| {
                vec![
                    Some((0, &flavor.fftplan.gpu_side_params)),
                    Some((1, &buffers.param_buffer)),
                    Some((2, &buffers.logspace_buffers[(i-1) % 3].0)),
                    Some((3, &buffers.logspace_buffers[i].0)),
                    Some((4, &buffers.logspace_buffers[(i+1) % 3].0)),
                    Some((5, &buffers.atomic_buffer)),
                    Some((6, &buffers.particles_buffer)),
                    Some((7, &buffers.global_max)),
                    ]}).collect::<Vec<_>>()),
            
            ("walk", buffers.logspace_buffers.iter().map(|(logspace_buffer, _, __)| {
            vec![
                Some((0, &buffers.param_buffer)),
                Some((1, logspace_buffer)), 
                Some((2, &buffers.particles_buffer)),
                Some((3, &buffers.atomic_buffer)),
                Some((4, &buffers.atomic_filtered_buffer)),
                Some((5, &buffers.result_buffer)),
                Some((6, &buffers.frame_buffer)),
                // if tracking_params.varcheck.is_some() {Some((6, &buffers.frame_buffer))} else {None},
                Some((7, &flavor.fftplan.gpu_side_params)),
            ]}).collect::<Vec<_>>()),

            // ("characterize", )
        ])
    }
    };

    let bind_group_entries = bind_group_entries
        .iter().map(|(&name, bind_entries_vector)| {
            let bind_entries_vector = bind_entries_vector.iter().map(|bind_entries|{
                bind_entries.iter().flatten().map(|(i, buffer)|
                    wgpu::BindGroupEntry {
                    binding: *i as u32,
                    resource: buffer.as_entire_binding()}).collect::<Vec<_>>()
                }).collect::<Vec<_>>();
                (name, bind_entries_vector)
            })
        .collect::<HashMap<_, _>>();
    
    
    let passes = pipelines.into_iter()
        .map(|(name, (pipeline, layout, wg_n))|{
        let entries = &bind_group_entries[name.as_str()];
        let passes_for_name = entries.iter().map(|entry|{
            let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entry[..],
            });
            let pass = FullComputePass{
                pipeline: Rc::clone(&pipeline),
                bindgroup,
                wg_n: wg_n.clone(),
            };
            pass
        }).collect::<Vec<_>>();
        
        (name, passes_for_name)
    }).collect::<HashMap<_, _>>();

    // let passes = ;

    // let pipelines = pipelines.into_iter().map(|(name, (computepipeline, bindgrouplayout, wg_n))|{
    //     (name, computepipeline)
    // }).collect::<HashMap<_,_>>();

    // match buffers{
    // GpuBuffers::Trackpy(buffers) => {
    //     let main_pipeline = vec![
    //         Some(&passes["preprocess_rows"][0]),
    //         // Some(&passes["preprocess_cols"][0]),
    //         // // Some(("pp_rows", &shaders["preprocess_rows"])),
    //         // // Some(("pp_cols", &shaders["preprocess_cols"])),
    //         // if (!characterize_new_points) {Some(&passes["max_rows"][0])} else {None},
    //         // if (!characterize_new_points) {Some(&passes["extract_max"][0])} else {None},
    //         // if (!characterize_new_points) {Some(&passes["walk"][0])} else {None},
    //         // if (tracking_params.characterize | characterize_new_points) {Some(&passes["characterize"][0])} else {None},
    //     ];
    
    //     // let main_pipeline = main_pipeline.into_iter().flatten().collect();
    
    

    let state = GpuState{
        device,
        queue,
        passes,
        // bind_groups,
        // pipelines,
        pic_byte_size: size,
        pic_size,
        result_read_depth,
        flavor,
        // passes: HashMap::new(),
    };
    // if let GpuStateFlavor::Log(ref flavor) = state.flavor{
    state.setup_buffers();

    // if let GpuStateFlavor::Log(flavor) = &state.flavor{
    //     let buffers = &flavor.buffers;
    //     let encoder = state.device.create_command_encoder(&Default::default());
    //     let copy_size = flavor.fftplan.params.dims;
    //     let copy_size = (copy_size[0] * copy_size[1]) as u64;
    //     inspect_buffer(&buffers.filter_buffers[0].0,
    //         &buffers.staging_buffers[0],
    //         &state.queue,
    //         encoder,
    //         &state.device,
    //         copy_size,
    //         "testing/dump.bin")
    //     }
    state

}

pub fn inspect_buffers<P: AsRef<std::path::Path>>(
    buffers_to_inspect: &[&wgpu::Buffer],
    mappable_buffer: &wgpu::Buffer,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    device: &wgpu::Device,
    // copy_size: u64,
    file_path: P,
    ) -> ! {
    
    let path = file_path.as_ref().to_owned();
    queue.submit(Some(encoder.finish()));
    
    for (i, &buffer) in buffers_to_inspect.iter().enumerate(){
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(mappable_buffer, 0, None);
        encoder.copy_buffer_to_buffer(buffer, 0, mappable_buffer, 0, buffer.size());
        queue.submit(Some(encoder.finish()));
        let slice = mappable_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range()[..].to_vec();
        let mut path = path.clone();
        path.push(format!("dump{}.bin", i));
        std::fs::write(&path, &data[..buffer.size() as usize]);
        drop(slice);
        mappable_buffer.unmap();
    }

    // let (sender, recv) = std::sync::mpsc::channel();
    // slice.map_async(wgpu::MapMode::Read, move |res|{sender.send(res);});
    // let idk = recv.recv().unwrap().unwrap();
    panic!("intended panic")
}