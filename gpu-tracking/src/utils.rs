
use wgpu_fft::FullComputePass;
use std::rc::Rc;
use wgpu::{self, util::DeviceExt};


pub unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        ::std::mem::size_of::<T>(),
    )
}

pub struct SeparableConvolution<const N: usize>{
    input_pass: FullComputePass,
    passes: [FullComputePass; 2],
}

pub enum Dispatcher{
    Direct([u32; 3]),
    Indirect{
        dispatcher: wgpu::Buffer,
        resetter: wgpu::Buffer,
    }
}

impl Dispatcher{
    pub fn new_direct(dims: &[u32; 3], wgsize: &[u32; 3]) -> Self { 
        let mut n_workgroups = [0, 0, 0];
        for i in 0..3 {
            n_workgroups[i] = (dims[i] + wgsize[i] - 1) / wgsize[i];
        }
        Self::Direct(n_workgroups)
    }

    pub fn new_indirect(device: &wgpu::Device, default: wgpu::util::DispatchIndirect) -> Self {
        let dispatcher = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: None,
            contents: default.as_bytes(),
            usage: wgpu::BufferUsages::STORAGE
        });
        
    }
}

impl<const N: usize> SeparableConvolution<N>{
    pub fn new<'a>(
        device: &wgpu::Device,
        pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, [u32; 3]),
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        temp_buffer: &wgpu::Buffer,
        additional_buffers: impl IntoIterator<Item = &'a wgpu::Buffer>,
    ) -> Self{
        let mut bind_group_entries_output_first = vec![
            wgpu::BindGroupEntry{
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            },
            
            wgpu::BindGroupEntry{
                binding: 1,
                resource: temp_buffer.as_entire_binding(),
            },
        ];
        let mut bind_group_entries_temp_first = vec![
            wgpu::BindGroupEntry{
                binding: 0,
                resource: temp_buffer.as_entire_binding(),
            },

            wgpu::BindGroupEntry{
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ];

        let first_out = if N % 2 == 0{
            temp_buffer
        } else {
            output_buffer
        };
        
        let mut bind_group_entries_input = vec![
            wgpu::BindGroupEntry{
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry{
                binding: 1,
                resource: first_out.as_entire_binding(),
            },
        ];

        for (i, additional_buffer) in additional_buffers.into_iter().enumerate(){
            bind_group_entries_output_first.push(
                wgpu::BindGroupEntry{
                    binding: i as u32 + 2,
                    resource: additional_buffer.as_entire_binding(),
                }
            );

            bind_group_entries_temp_first.push(
                wgpu::BindGroupEntry{
                    binding: i as u32 + 2,
                    resource: additional_buffer.as_entire_binding(),
                }
            );

            bind_group_entries_input.push(
                wgpu::BindGroupEntry{
                    binding: i as u32 + 2,
                    resource: additional_buffer.as_entire_binding(),
                }
            );
        }

        let bind_group_output_first = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: None,
            layout: &pipeline.1,
            entries: &bind_group_entries_output_first[..],
        });

        let bind_group_temp_first = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: None,
            layout: &pipeline.1,
            entries: &bind_group_entries_temp_first[..],
        });

        let bind_group_input = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: None,
            layout: &pipeline.1,
            entries: &bind_group_entries_input[..],
        });


        let full_output_first = FullComputePass{
            pipeline: Rc::clone(&pipeline.0),
            bindgroup: bind_group_output_first,
            wg_n: pipeline.2.clone(),
        };

        let full_temp_first = FullComputePass{
            pipeline: Rc::clone(&pipeline.0),
            bindgroup: bind_group_temp_first,
            wg_n: pipeline.2.clone(),
        };

        let full_input = FullComputePass{
            pipeline: Rc::clone(&pipeline.0),
            bindgroup: bind_group_input,
            wg_n: pipeline.2.clone(),
        };
        
        let base = if N % 2 == 0{
            [full_temp_first, full_output_first]
        } else{
            [full_output_first, full_temp_first]
        };

        Self{
            passes: base,
            input_pass: full_input,
        }
    }

    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, push_constants: &[u8]){
        let input = std::iter::once(&self.input_pass);
        for (dim, pass) in input.chain(self.passes.iter().cycle().take(N-1)).enumerate(){
            let push_constants = Vec::from_iter(unsafe{ any_as_u8_slice(&(dim as u32)).iter().chain(push_constants.iter()).cloned() });
            pass.execute(encoder, &push_constants[..]);
        }
    }
}


pub struct Laplace<const N: usize>{
    pass: SeparableConvolution<N>,
}



impl<const N: usize> Laplace<N>{
    pub fn new<'a>(
        device: &wgpu::Device,
        pipeline: &(Rc<wgpu::ComputePipeline>, wgpu::BindGroupLayout, [u32; 3]),
        input_buffer: &wgpu::Buffer,
        output_buffer: &'a wgpu::Buffer,
        temp1: &wgpu::Buffer,
        temp2: &wgpu::Buffer,
        additional_buffers: impl IntoIterator<Item = &'a wgpu::Buffer>,
    ) -> Self{
        let additional_buffers = [output_buffer].into_iter().chain(additional_buffers.into_iter());
        let sep = SeparableConvolution::<N>::new(
            device,
            pipeline,
            input_buffer,
            temp1,
            temp2,
            additional_buffers
        );
        // let mut bind_group_entries = vec![
        //     wgpu::BindGroupEntry{
        //         binding: 0,
        //         resource: input_buffer.as_entire_binding(),
        //     },
            
        //     wgpu::BindGroupEntry{
        //         binding: 1,
        //         resource: output_buffer.as_entire_binding(),
        //     },
        // ];

        // for (i, additional_buffer) in additional_buffers.into_iter().enumerate(){
        //     bind_group_entries.push(
        //         wgpu::BindGroupEntry{
        //             binding: i as u32 + 2,
        //             resource: additional_buffer.as_entire_binding(),
        //         }
        //     );
        // }

        // let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
        //     label: None,
        //     layout: &pipeline.1,
        //     entries: &bind_group_entries[..],
        // });


        // let full = FullComputePass{
        //     pipeline: Rc::clone(&pipeline.0),
        //     bindgroup: bind_group,
        //     wg_n: pipeline.2.clone(),
        // };

        Self{
            pass: sep,
        }
    }

    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, push_constants: &[u8]){

        for diff_dim in 0..(N as u32){
            let push_constants = Vec::from_iter(unsafe{ any_as_u8_slice(&(diff_dim)).iter().chain(push_constants.iter()).cloned() });
            self.pass.execute(encoder, &push_constants[..]);
        }
    }
}






pub fn inspect_buffers<P: AsRef<std::path::Path>>(
    buffers_to_inspect: &[&wgpu::Buffer],
    mappable_buffer: &wgpu::Buffer,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    device: &wgpu::Device,
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
        dbg!(std::env::current_dir().unwrap());
        dbg!(&path);
        std::fs::write(&path, &data[..buffer.size() as usize]).unwrap();
        drop(slice);
        mappable_buffer.unmap();
    }

    panic!("intended panic")
}