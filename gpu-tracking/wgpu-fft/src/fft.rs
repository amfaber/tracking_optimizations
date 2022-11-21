#![allow(warnings)]
use wgpu::{self, BufferBinding, ComputePass};
use std::{collections::HashMap, hash::Hash, char::ParseCharError, rc::Rc};
use bytemuck;
use crate::FullComputePass;

// #[derive(bytemuck::Pod, bytemuck::Zeroable)]
// #[repr(C)]
#[derive(Debug)]
pub struct FftParams{
    pub dims: [u32; 2],
    pub stage: u32,
    pub current_dim: u32,
    pub inverse: u32,
}



pub struct FftPass<const N: usize>{
    passes: [FullComputePass; N],
    exponents: [u32; N],
    normalizer: FullComputePass,
}


impl<const N: usize> FftPass<N>{
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, inverse: bool, normalize: bool){
        let inverseu32 = if inverse {1u32} else {0};
        for (dim, (pass, n)) in self.passes.iter().zip(self.exponents.iter()).enumerate(){
            for p in 1..=*n{
                // let mut cpass = encoder.begin_compute_pass(&Default::default());
                let push_constants = [p, dim as u32, inverseu32];
                // cpass.set_push_constants(0, bytemuck::cast_slice(&push_constants));
                pass.execute(encoder, bytemuck::cast_slice(&push_constants));
            }
        }

        if normalize{
            // let cpass = encoder.begin_compute_pass(&Default::default());
            self.normalizer.execute(encoder, &[]);
        }

    }
}

// static fftparams_size_checker: [u8; 20] = [0; std::mem::size_of::<FftParams>()];

pub fn array_map<T: Default, const N: usize, F: Fn(&T, &T) -> T>(arr1: &[T; N], arr2: &[T; N], function: F) -> [T; N]{
    let mut result: [T; N] = core::array::from_fn(|_| T::default());
    
    result.iter_mut().zip(arr1.iter().zip(arr2.iter())).for_each(|(r, (a, b))|{
        *r = function(a, b)
    });

    result
}

impl FftParams{
    pub fn new(dims: &[u32; 2]) -> Self{
        FftParams { dims: dims.clone(), stage: 1, current_dim: 0, inverse: 0 }
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

    pub fn write_shape_to_buffer(&self, queue: &wgpu::Queue, buffer: &wgpu::Buffer){
        let slice = unsafe{ std::slice::from_raw_parts(self as *const FftParams as *const u8, std::mem::size_of::<u32>()*2) };
        queue.write_buffer(buffer, 0, slice);
    }

}


pub struct FftPlan{
    pub pipelines: HashMap<String, (Rc<wgpu::ComputePipeline>, [u32; 3], wgpu::BindGroupLayout)>,
    pub exponents: [u32; 2],
    pub params: FftParams,
    pub gpu_side_params: wgpu::Buffer,
    pub twiddle_factors: [wgpu::Buffer; 2],
    // pub device: &'a wgpu::Device,
    // pub queue: &'a wgpu::Queue,
}

impl FftPlan{

    pub fn create(
        dims: &[u32; 2],
        shaders: HashMap<String,
        (wgpu::ShaderModule, [u32; 3])>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ) -> Self{
        assert_eq!(dims[0].count_ones(), 1);
        assert_eq!(dims[1].count_ones(), 1);
        
        let n = dims[0].trailing_zeros();
        let m = dims[1].trailing_zeros();

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
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                [
                    device.create_buffer(&wgpu::BufferDescriptor{
                        label: None,
                        size: std::cmp::max((dims[0] * std::mem::size_of::<f32>() as u32) as u64, 8),
                        usage,
                        mapped_at_creation: false,
                    }),
                    device.create_buffer(&wgpu::BufferDescriptor{
                        label: None,
                        size: std::cmp::max((dims[1] * std::mem::size_of::<f32>() as u32) as u64, 8),
                        usage,
                        mapped_at_creation: false,
                }),
                ]
            )
        };

        let shapes = HashMap::from([
          ("fft0",              [dims[0] / 2, dims[1], 1]),
          ("fft1",              [dims[0], dims[1] / 2, 1]),
          ("twiddle_setup",     [dims[0] / 2 + dims[1] / 2, 1, 1]),
          ("pad",               [dims[0], dims[1], 1]),
          ("normalize",         [dims[0], dims[1], 1]),
          ("multiply",          [dims[0]*dims[1], 1, 1]),
          ("multiply_inplace",  [dims[0]*dims[1], 1, 1]),
        ]);
        // dbg!(&shapes);

        let pipelines = shaders.iter().map(|(name, (shader, wg_size))| {
            shapes.get(name.as_str()).expect(format!("{} not found", name).as_str());
            let shape = shapes[name.as_str()];
            let wg_n = n_workgroups(&shape, wg_size);
            let dummy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
                label: None,
                layout: None,
                module: shader,
                entry_point: "main",
            });
            
            let bindgrouplayout = dummy_pipeline.get_bind_group_layout(0);
            // bindgrouplayout
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
                label: None,
                bind_group_layouts: &[&bindgrouplayout],
                push_constant_ranges: &[wgpu::PushConstantRange{
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
                label: None,
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: "main",
            });
            (name.to_string(), (Rc::new(pipeline), wg_n, bindgrouplayout))
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
            params.write_shape_to_buffer(queue, &gpu_side_params);
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
            // device,
            // queue,
        }

        
    }

    pub fn fft_pass(
        &self,
        // encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        device: &wgpu::Device,
        // inverse: bool,
        // normalize: bool,
        ) -> FftPass<2> {
        
        // self.params.inverse = if inverse { 1 } else { 0 };
        // self.params.reset();

        let bind_group_entries = [
            ("fft0", vec![
                (0, &self.gpu_side_params),
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
        // let mut order = [(0, "fft0"), (1, "fft1")];


        // self.params.current_dim = 0;
        let bind_group_entries = bind_group_entries.iter().map(|(name, entries)|{
            let entries = entries.iter().map(|(binding, buffer)|{
                wgpu::BindGroupEntry{
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                }
            }).collect::<Vec<_>>();
            (name.to_string(), entries)
        }).collect::<HashMap<_,_>>();

        let (pipeline0, wg_n0, layout) = &self.pipelines["fft0"];
        let pipeline0 = Rc::clone(pipeline0);
        let bindgroup0 = device.create_bind_group(&wgpu::BindGroupDescriptor{
                    label: None,
                    layout,
                    entries: &bind_group_entries["fft0"],
                });

        
        // dbg!(&self.params.current_dim);
        // for p in 1..=self.exponents[0]{
        //     // dbg!(p);
        //     let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        //     cpass.set_bind_group(0, &bindgroup, &[]);
        //     cpass.set_pipeline(pipeline);
        //     let mut push_constants = vec![p, self.params.current_dim, self.params.inverse];
        //     // push_constants.extend(bytemuck::cast_slice(&[self.params.inverse]));
        //     let push_constants = bytemuck::cast_slice::<u32, u8>(&push_constants);
        //     cpass.set_push_constants(0, push_constants);
        //     cpass.dispatch_workgroups(wg_n[0], wg_n[1], wg_n[2]);
        // }
        
        // self.params.current_dim = 1;
        let (pipeline1, wg_n1, layout) = &self.pipelines["fft1"];
        let pipeline1 = Rc::clone(pipeline1);
        let bindgroup1 = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: None,
            layout,
            entries: &bind_group_entries["fft1"],
        });

        // for p in 1..=self.exponents[1]{
        //     // dbg!(p);
        //     let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        //     cpass.set_bind_group(0, &bindgroup, &[]);
        //     cpass.set_pipeline(pipeline);
        //     let mut push_constants = vec![p, self.params.current_dim];
        //     push_constants.extend(bytemuck::cast_slice(&[self.params.inverse]));
        //     let push_constants = bytemuck::cast_slice::<u32, u8>(&push_constants);
        //     cpass.set_push_constants(0, push_constants);
        //     cpass.dispatch_workgroups(wg_n[0], wg_n[1], wg_n[2]);
        // }
        
        // if normalize{
        let (pipelinen, wg_nn, layout) = &self.pipelines["normalize"];
        let pipelinen = Rc::clone(pipelinen);
        let bindgroupn = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: None,
            layout,
            entries: &bind_group_entries["normalize"],
        });

        FftPass{
            passes: [
                FullComputePass{
                    pipeline: pipeline0,
                    wg_n: wg_n0.clone(),
                    bindgroup: bindgroup0,
                },
                FullComputePass{
                    pipeline: pipeline1,
                    wg_n: wg_n1.clone(),
                    bindgroup: bindgroup1,
                },

            ],
            exponents: self.exponents.clone(),
            normalizer: FullComputePass {
                pipeline: pipelinen,
                wg_n: wg_nn.clone(),
                bindgroup: bindgroupn,
            },
        }
        // let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        // cpass.set_bind_group(0, &bindgroup, &[]);
        // cpass.set_pipeline(pipeline);
        // cpass.dispatch_workgroups(wg_n[0], wg_n[1], wg_n[2]);
            // cpass.set_bind_group(0, bind_group, offsets)
        // }
        // todo!()
    }

    pub fn create_buffer(&self, device: &wgpu::Device, mappable: bool) -> wgpu::Buffer {

        let size = self.params.dims.iter().product::<u32>() as u64 * std::mem::size_of::<f32>() as u64 * 2;
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        if mappable{
            usage |= wgpu::BufferUsages::MAP_READ
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        });

        buffer
    }

    pub fn pad_pass(&self,
        device: &wgpu::Device,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        input_shape: &[u32; 2]
        ) -> (FullComputePass, [u32; 4]) {
        // self.params.write_shape_to_buffer(self.queue, &self.gpu_side_params);
        let bind_group_entries = [
            ("pad", vec![
                // (0, &self.gpu_side_params),
                (0, input),
                (1, output),
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

        let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor{
                    label: None,
                    layout,
                    entries: &bind_group_entries["pad"],
        });
        

        let pass = FullComputePass {
            bindgroup,
            wg_n: wg_n.clone(),
            pipeline: Rc::clone(pipeline)
        };
        let push_constants = [input_shape[0], input_shape[1], self.params.dims[0], self.params.dims[1]];

        (pass, push_constants)
        // {
        //     let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        //     // dbg!(wg_n);
        //     cpass.set_bind_group(0, &bindgroup, &[]);
        //     cpass.set_pipeline(pipeline);
        //     let push_constants = [input_shape[0], input_shape[1], self.params.dims[0], self.params.dims[1]];
        //     cpass.set_push_constants(0, bytemuck::cast_slice(&[push_constants]));
        //     cpass.dispatch_workgroups(wg_n[0], wg_n[1], wg_n[2]);
        // }

    }

    pub fn inplace_spectral_convolution_pass(&self, device: &wgpu::Device, modified_inplace: &wgpu::Buffer, factor: &wgpu::Buffer)
    -> FullComputePass{

        let bind_group_entries = [
            ("multiply_inplace", vec![
                (0, modified_inplace),
                (1, factor),
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

        let (pipeline, wg_n, layout) = &self.pipelines["multiply_inplace"];
        let pipeline = Rc::clone(pipeline);
        let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor{
                    label: None,
                    layout,
                    entries: &bind_group_entries["multiply_inplace"],
        });
        

        FullComputePass { bindgroup, wg_n: wg_n.clone(), pipeline }

    }

    pub fn spectral_convolution_pass(&self, device: &wgpu::Device, input1: &wgpu::Buffer, input2: &wgpu::Buffer, output: &wgpu::Buffer)
    -> FullComputePass{

        let bind_group_entries = [
            ("multiply", vec![
                // (0, &self.gpu_side_params),
                (0, input1),
                (1, input2),
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

        let (pipeline, wg_n, layout) = &self.pipelines["multiply"];
        let pipeline = Rc::clone(pipeline);
        let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor{
                    label: None,
                    layout,
                    entries: &bind_group_entries["multiply"],
        });
        

        FullComputePass { bindgroup, wg_n: wg_n.clone(), pipeline }

    }



}


pub fn get_shape(image_shape: &[u32; 2], filter_shape: &[u32; 2]) -> [u32; 2]{
    let output_shape = array_map(image_shape, filter_shape, |a, b| a + b - 1);
    let output_shape = output_shape.map(|size| {
        if size.count_ones() == 1{
            size
        } else {
            1 << (32 - size.leading_zeros())
        }
    });
    output_shape
}



pub fn compile_shaders(device: &wgpu::Device, workgroup_size2d: Option<&[u32; 3]>, workgroup_size1d: Option<&[u32; 3]>) -> HashMap<String, (wgpu::ShaderModule, [u32; 3])>{
    
    

    // let workgroup_size_2d = [16u32, 16, 1];
    let workgroup_size2d = workgroup_size2d.unwrap_or(&[16u32, 16, 1]).clone();
    let workgroup_size1d =  workgroup_size1d.unwrap_or(&[256u32, 1, 1]).clone();
    // let wg_dims = [dims[0], dims[1], 1];

    let common_header = include_str!("shaders/common_header.wgsl");

    let fft_source = include_str!("shaders/fft.wgsl");

    let shaders = HashMap::from([
        ("fft0", (fft_source, workgroup_size2d)),
        // ("fft", (fft_source, workgroup_size2d)),
        ("fft1", (fft_source, workgroup_size2d)),
        ("twiddle_setup", (include_str!("shaders/twiddle_setup.wgsl"), workgroup_size1d)),
        ("pad", (include_str!("shaders/pad.wgsl"), workgroup_size2d)),
        ("normalize", (include_str!("shaders/normalize.wgsl"), workgroup_size2d)),
        ("multiply_inplace", (include_str!("shaders/multiply_inplace.wgsl"), workgroup_size1d)),
        ("multiply", (include_str!("shaders/multiply.wgsl"), workgroup_size1d)),
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
