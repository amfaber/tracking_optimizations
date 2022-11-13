#![allow(warnings)]
use wgpu;
use std::{collections::HashMap, hash::Hash};
use bytemuck;

// #[derive(bytemuck::Pod, bytemuck::Zeroable)]
// #[repr(C)]
#[derive(Debug)]
pub struct FftParams{
    pub dims: [u32; 2],
    pub stage: u32,
    pub current_dim: u32,
    pub inverse: f32,
}

static fftparams_size_checker: [u8; 20] = [0; std::mem::size_of::<FftParams>()];


impl FftParams{
    pub fn new(dims: &[u32; 2]) -> Self{
        FftParams { dims: dims.clone(), stage: 1, current_dim: 0, inverse: 1. }
    }
    
    pub fn set_stage(&mut self, stage: u32){
        self.stage = stage;
    }

    pub fn set_dim(&mut self, dim: u32){
        self.current_dim = dim;
    }

    pub fn next_stage(&mut self){
        self.stage += 1;
    }
    
    pub fn reset(&mut self){
        self.stage = 1;
    }

    pub fn write_to_buffer(&self, queue: &wgpu::Queue, buffer: &wgpu::Buffer){
        let slice = unsafe{ std::slice::from_raw_parts(self as *const FftParams as *const u8, std::mem::size_of::<Self>()) };
        queue.write_buffer(buffer, 0, slice);
    }

    pub fn invert(&mut self){
        self.inverse = self.inverse * -1.;
    }
}


pub struct FftPlan<'a>{
    pub pipelines: HashMap<String, (wgpu::ComputePipeline, [u32; 3], wgpu::BindGroupLayout)>,
    pub exponents: [u32; 2],
    pub params: FftParams,
    pub gpu_side_params: wgpu::Buffer,
    pub twiddle_factors: [wgpu::Buffer; 2],
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
}

impl<'a> FftPlan<'a>{

    pub fn create(
        dims: &[u32; 2],
        shaders: HashMap<String,
        (wgpu::ShaderModule, [u32; 3])>,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        ) -> Self{
        assert_eq!(dims[0]-1 & dims[0], 0);
        assert_eq!(dims[1]-1 & dims[1], 0);

        let (n, m) = {
            let mut i = 0;
            let n = loop{
                if i == 32{
                    panic!();
                }
                if (dims[0] >> i & 1) == 1{
                    break i;
                }
                i += 1;
            };
            let mut i = 0;
            let m = loop{
                if i == 32{
                    panic!();
                }
                if (dims[0] >> i & 1) == 1{
                    break i;
                }
                i += 1;
            };
            (n, m)
        };


        let params = FftParams::new(dims);
        

        let n_workgroups = |dims: &[u32; 3], wgsize: &[u32; 3]| { 
            let mut n_workgroups = [0, 0, 0];
            for i in 0..3 {
                n_workgroups[i] = (dims[i] + wgsize[i] - 1) / wgsize[i];
            }
            n_workgroups
        };

        
        let (gpu_side_params, twiddle_factors) = {
            let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_READ;
            (
                device.create_buffer(&wgpu::BufferDescriptor{
                    label: None,
                    size: std::mem::size_of::<FftParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                [
                    device.create_buffer(&wgpu::BufferDescriptor{
                        label: None,
                        size: (dims[0] * std::mem::size_of::<f32>() as u32) as u64,
                        usage,
                        mapped_at_creation: false,
                    }),
                    device.create_buffer(&wgpu::BufferDescriptor{
                        label: None,
                        size: (dims[1] * std::mem::size_of::<f32>() as u32) as u64,
                        usage,
                        mapped_at_creation: false,
                }),
                ]
            )
        };

        let shapes = HashMap::from([
          ("fft", [dims[0] / 2, dims[1] / 2, 1]),
          ("twiddle_setup", [dims[0] / 2 + dims[1] / 2, 1, 1]),
          ("pad", [dims[0], dims[1], 1]),
        ]);

        let pipelines = shaders.iter().map(|(name, (shader, wg_size))| {
            let shape = shapes[name.as_str()];
            let wg_n = n_workgroups(&shape, wg_size);
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
                label: None,
                layout: None,
                module: shader,
                entry_point: "main",
            });
            let layout = pipeline.get_bind_group_layout(0);
            (name.to_string(), (pipeline, wg_n, layout))
        }).collect::<HashMap<_, _>>();

        // Twiddle setup
        {
            // let (shader, wg_size) = &shaders["twiddle_setup"];
            // let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            //     label: None,
            //     layout: None,
            //     module: shader,
            //     entry_point: "main"
            // });

            let twiddle_size = dims[0] / 2 + dims[1] / 2;
            let (pipeline, wg_n, bind_group_layout) = &pipelines["twiddle_setup"];
            // let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group =
                device.create_bind_group(
                    &wgpu::BindGroupDescriptor{
                        label: None,
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry{
                            binding: 0 as u32,
                            resource: gpu_side_params.as_entire_binding(),
                        },
                            wgpu::BindGroupEntry{
                            binding: 1 as u32,
                            resource: twiddle_factors[0].as_entire_binding(),
                        },
                            wgpu::BindGroupEntry{
                            binding: 2 as u32,
                            resource: twiddle_factors[1].as_entire_binding(),
                        },
                        ],
                    }
                );
            
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            params.write_to_buffer(queue, &gpu_side_params);
            // let wg_n = n_workgroups(&[twiddle_size as u32, 1, 1], wg_size);
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.set_pipeline(&pipeline);
                cpass.dispatch_workgroups(wg_n[0], wg_n[1], wg_n[2])
            }
            queue.submit(Some(encoder.finish()));
        };

        Self{
            pipelines,
            exponents: [n, m],
            params,
            gpu_side_params,
            twiddle_factors,
            device,
            queue,
        }

        
    }

    pub fn fft(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        inverse: bool,
        normalize: bool,
        ){
        
        self.params.inverse = if inverse {-1.} else {1.};
        // self.params.reset();

        let bind_group_entries = [
            ("fft0", vec![
                (0u32, &self.gpu_side_params),
                (1, input),
                (2, &self.twiddle_factors[0]),
            ]),
            ("fft1", vec![
                (0, &self.gpu_side_params),
                (1, input),
                (2, &self.twiddle_factors[1]),
            ]),
            ("normalize", vec![
                (0, &self.gpu_side_params),
                (1, input),
            ]),

        ];
        
        let bind_group_entries = bind_group_entries.iter().map(|(name, entries)|{
            let entries = entries.iter().map(|(binding, buffer)|{
                wgpu::BindGroupEntry{
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                }
            }).collect::<Vec<_>>();
            (name.to_string(), entries)
        }).collect::<HashMap<_,_>>();


        let (pipeline, wg_n, layout) = &self.pipelines["fft"];
        let bindgroup = self.device.create_bind_group(&wgpu::BindGroupDescriptor{
                    label: None,
                    layout,
                    entries: &bind_group_entries["fft0"],
                });
                
        for p in 1..self.exponents[0]{
            self.params.set_stage(p);
            self.params.write_to_buffer(self.queue, &self.gpu_side_params);
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_bind_group(0, &bindgroup, &[]);
                cpass.set_pipeline(pipeline);
                cpass.dispatch_workgroups(wg_n[0], wg_n[1], wg_n[2]);
            };
        }

        let bindgroup = self.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: None,
            layout,
            entries: &bind_group_entries["fft1"],
        });

        self.params.set_dim(1);

        for p in 1..self.exponents[1]{
            self.params.set_stage(p);
            self.params.write_to_buffer(self.queue, &self.gpu_side_params);
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_bind_group(0, &bindgroup, &[]);
                cpass.set_pipeline(pipeline);
                cpass.dispatch_workgroups(wg_n[0], wg_n[1], wg_n[2]);
            };
        }
        
        // let (pipeline, wg_n, layout) = &self.pipelines["normalize"];
        // let bindgroup = self.device.create_bind_group(&wgpu::BindGroupDescriptor{
        //     label: None,
        //     layout,
        //     entries: &bind_group_entries["normalize"],
        // });
        
        // if normalize{
        //     let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        //     cpass.set_bind_group(0, &bindgroup, &[]);
        //     cpass.set_pipeline(pipeline);
        //     cpass.dispatch_workgroups(wg_n[0], wg_n[1], wg_n[2]);
        //     // cpass.set_bind_group(0, bind_group, offsets)
        // }
    }


    pub fn pad(&self, encoder: &mut wgpu::CommandEncoder, input: &wgpu::Buffer, output: &wgpu::Buffer){
        self.params.write_to_buffer(self.queue, &self.gpu_side_params);
        let bind_group_entries = [
            ("pad", vec![
                (0, &self.gpu_side_params),
                (1, input),
                (2, output),
            ]),

        ];
        let bind_group_entries = bind_group_entries.iter().map(|(name, entries)|{
            let entries = entries.iter().map(|(binding, buffer)|{
                wgpu::BindGroupEntry{
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                }
            }).collect::<Vec<_>>();
            (name.to_string(), entries)
        }).collect::<HashMap<_,_>>();

        let (pipeline, wg_n, layout) = &self.pipelines["pad"];

        let bindgroup = self.device.create_bind_group(&wgpu::BindGroupDescriptor{
                    label: None,
                    layout,
                    entries: &bind_group_entries["pad"],
        });
        
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            dbg!(wg_n);
            cpass.set_bind_group(0, &bindgroup, &[]);
            cpass.set_pipeline(pipeline);
            cpass.dispatch_workgroups(wg_n[0], wg_n[1], wg_n[2]);
        }

    }
}







pub fn compile_shaders(device: &wgpu::Device, workgroup_size2d: Option<&[u32; 3]>, workgroup_size1d: Option<&[u32; 3]>) -> HashMap<String, (wgpu::ShaderModule, [u32; 3])>{
    
    

    // let workgroup_size_2d = [16u32, 16, 1];
    let workgroup_size2d = workgroup_size2d.unwrap_or(&[16u32, 16, 1]).clone();
    let workgroup_size1d =  workgroup_size1d.unwrap_or(&[256u32, 1, 1]).clone();
    // let wg_dims = [dims[0], dims[1], 1];

    let common_header = include_str!("shaders/common_header.wgsl");

    let shaders = HashMap::from([
        ("fft", (include_str!("shaders/fft.wgsl"), workgroup_size2d)),
        ("twiddle_setup", (include_str!("shaders/twiddle_setup.wgsl"), workgroup_size1d)),
        ("pad", (include_str!("shaders/pad.wgsl"), workgroup_size2d)),
    ]);

    let preprocess_source = |source, wg_size: &[u32; 3]| -> String {
        let mut result = String::new();
        result.push_str(common_header);
        result.push_str(source);
        result.replace("@workgroup_size(_)",
        format!("@workgroup_size({}, {}, {})", wg_size[0], wg_size[1], wg_size[2]).as_str())
    };


    let shaders = shaders.iter()
        .map(|(&name, (source, group_size))|{
        let shader_source = preprocess_source(source, group_size);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        (name.to_string(), (shader, group_size.clone()))
    }).collect::<HashMap<_, _>>();

    shaders
}
