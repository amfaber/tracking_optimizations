use std::rc::Rc;

pub mod fft;

pub struct FullComputePass{
    pub bindgroup: wgpu::BindGroup,
    pub wg_n: [u32; 3],
    pub pipeline: Rc<wgpu::ComputePipeline>,
}


impl FullComputePass{
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, push_constants: &[u8]){
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_bind_group(0, &self.bindgroup, &[]);
        cpass.set_pipeline(&self.pipeline);
        if push_constants.len() > 0{
            cpass.set_push_constants(0, push_constants);
        }
        cpass.dispatch_workgroups(self.wg_n[0], self.wg_n[1], self.wg_n[2]);
    }
}

#[cfg(test)]
mod tests {
    // bytemuck::cast_slice()
}
//     use std::time::Instant;

//     use crate::fft;
//     use wgpu::{self};
//     use pollster::FutureExt;
//     use realfft::{self, num_complex::Complex32};
//     // use bytemuck;

//     struct GpuState{
//         _instance: wgpu::Instance,
//         _adapter: wgpu::Adapter,
//         device: wgpu::Device,
//         queue: wgpu::Queue,
//     }

//     impl GpuState{
//         fn new() -> Self{
//             let instance = wgpu::Instance::new(wgpu::Backends::all());
//             let adapter = instance.request_adapter(&wgpu::RequestAdapterOptionsBase{
//                 power_preference: wgpu::PowerPreference::HighPerformance,
//                 force_fallback_adapter: false,
//                 compatible_surface: None,
//             }).block_on().unwrap();

//             let mut desc = wgpu::DeviceDescriptor::default();
//             desc.features = wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
//             desc.features |= wgpu::Features::PUSH_CONSTANTS;
//             desc.limits.max_push_constant_size = 128;
//             // desc.limits.max_compute_invocations_per_workgroup = 512;
//             // desc.limits.max_compute_workgroup_size_x = 512;
            
//             let (device, queue) = adapter
//             .request_device(&desc, None).block_on().unwrap();
            
//             Self{
//                 _instance: instance,
//                 _adapter: adapter,
//                 device,
//                 queue,
//             }
//         }
//     }

//     #[test]
//     fn gpu_speed() {
//         let shape = [300, 300];
//         let data = (0..shape[0] * shape[1]).map(|_i| _i as f32).collect::<Vec<_>>();
//         let size = std::mem::size_of::<f32>() * data.len();
//         let size = size as u64;
//         let state = GpuState::new();
//         let shaders =
//          fft::compile_shaders(&state.device, None, None);

//         let input = state.device.create_buffer(&wgpu::BufferDescriptor {
//             label: None,
//             size,
//             usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_READ,
//             mapped_at_creation: false,
//         });
        
//         let padded_shape = fft::get_shape(&shape, &[1, 1]);

//         let mut plan = fft::FftPlan::create(&padded_shape, shaders, &state.device, &state.queue);
//         let padded = plan.create_buffer(false);
//         // let padded = state.device.create_buffer(&wgpu::BufferDescriptor {
//         //     label: None,
//         //     size: size * 2,
//         //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC 
//         //     // | wgpu::BufferUsages::MAP_READ
//         //     ,
//         //     mapped_at_creation: false,
//         // });
//         // let test = data.as_ptr() as *const u8;
//         // let copy_slice = unsafe{ std::slice::from_raw_parts(data.as_ptr() as *const u8, size as usize) };
//         let copy_slice = bytemuck::cast_slice::<f32, u8>(&data);
        
//         state.queue.write_buffer(&input, 0, copy_slice);
        
//         let mut encoder = state.device.create_command_encoder(&Default::default());
//         plan.pad(&mut encoder, &input, &padded, &[shape[0], shape[1]]);
//         state.queue.submit(Some(encoder.finish()));

//         let n = 250;
//         let now = Instant::now();
//         for _i in 0..n{
//             let mut encoder = state.device.create_command_encoder(&Default::default());
//             // plan.fft(&mut encoder, &padded, false, false);
//             // plan.fft(&mut encoder, &padded, true, true);
//             state.queue.submit(Some(encoder.finish()));
//         }
//         // let slice = plan.twiddle_factors[0].slice(..);
//         // let slice = padded.slice(..);
//         // slice.map_async(wgpu::MapMode::Read, |_| {});
//         state.device.poll(wgpu::Maintain::Wait);
//         dbg!(now.elapsed().as_nanos() as f64 / 1_000_000_000.);
//         // // dbg!(now.elapsed().as_nanos() as f64 / (2. * n as f64));
//         // let read_data = slice.get_mapped_range();
//         // std::fs::write("out.bin", &read_data[..]).unwrap();

//     }
//     #[test]
//     fn gpu_save() {
//         let shape = [16, 16];
//         let data = (0..shape[0] * shape[1]).map(|_i| _i as f32).collect::<Vec<_>>();
//         let size = std::mem::size_of::<f32>() * data.len();
//         let size = size as u64;
//         let state = GpuState::new();
//         let shaders =
//          fft::compile_shaders(&state.device, None, None);

//         let input = state.device.create_buffer(&wgpu::BufferDescriptor {
//             label: None,
//             size,
//             usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_READ,
//             mapped_at_creation: false,
//         });
        
//         let padded_shape = fft::get_shape(&shape, &[1, 1]);

//         let mut plan = fft::FftPlan::create(&padded_shape, shaders, &state.device, &state.queue);
//         let padded = plan.create_buffer(true);

//         let copy_slice = bytemuck::cast_slice::<f32, u8>(&data);
        
//         state.queue.write_buffer(&input, 0, copy_slice);
        
//         let mut encoder = state.device.create_command_encoder(&Default::default());
//         plan.pad(&mut encoder, &input, &padded, &[shape[0], shape[1]]);
//         state.queue.submit(Some(encoder.finish()));

//         let mut encoder = state.device.create_command_encoder(&Default::default());
//         // plan.fft(&mut encoder, &padded, false, false);
//         // plan.fft(&mut encoder, &padded, true, true);
//         state.queue.submit(Some(encoder.finish()));
//         let slice = padded.slice(..);
//         slice.map_async(wgpu::MapMode::Read, |_| {});
//         state.device.poll(wgpu::Maintain::Wait);
//         let read_data = slice.get_mapped_range();
//         std::fs::write("out.bin", &read_data[..]).unwrap();

//     }

//     #[test]
//     fn rustfft_test() {
//         let mut real_planner = realfft::RealFftPlanner::<f32>::new();
//         let r2c = real_planner.plan_fft_forward(1024);
//         let shape = [1024, 1024];
//         let mut data = (0..shape[0]*shape[1]).map(|_i| _i as f32).collect::<Vec<_>>();
//         let mut transposed = (0..shape[0]*shape[1]).map(|_i| 0 as f32).collect::<Vec<_>>();
//         let mut output = (0..shape[0]*shape[1]).map(|_i| Complex32::new(0., 0.)).collect::<Vec<_>>();
//         let now = Instant::now();
//         for _i in 0..500{
//             (0..shape[0]).for_each(|idx| r2c.process(data[idx*shape[1]..(idx+1)*shape[1]].as_mut(),
//             output[idx*(shape[1]/2+1)..(idx+1)*(shape[1]/2+1)].as_mut()).unwrap());
//             transpose::transpose(&data, &mut transposed, 1024, 1024);
//             (0..shape[0]).for_each(|idx| r2c.process(data[idx*shape[1]..(idx+1)*shape[1]].as_mut(),
//             output[idx*(shape[1]/2+1)..(idx+1)*(shape[1]/2+1)].as_mut()).unwrap());
//         }
//         dbg!(now.elapsed().as_nanos() as f64 / 1_000_000_000.);
//     }

//     #[test]
//     fn nextpow2() {
//         let img_shape = [32, 33];
//         let flt_shape = [1, 1];
//         let idk = fft::get_shape(&img_shape, &flt_shape);
//         dbg!(idk);
//     }
// }



