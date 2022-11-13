pub mod fft;
pub mod convolutions;



#[cfg(test)]
mod tests {
    use crate::fft;
    use wgpu::{self, Queue};
    use pollster::FutureExt;
    use bytemuck;

    struct GpuState{
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    impl GpuState{
        fn new() -> Self{
            let instance = wgpu::Instance::new(wgpu::Backends::all());
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptionsBase{
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            }).block_on().unwrap();

            let mut desc = wgpu::DeviceDescriptor::default();
            desc.features = wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
            
            let (device, queue) = adapter
            .request_device(&desc, None).block_on().unwrap();
            
            Self{
                instance,
                adapter,
                device,
                queue,
            }
        }
    }

    #[test]
    fn it_works() {
        let data = (0..256*256).map(|i| i as f32).collect::<Vec<_>>();
        let size = std::mem::size_of::<f32>() * data.len();
        let size = size as u64;
        let state = GpuState::new();
        let shaders = fft::compile_shaders(&state.device, None, None);

        let shape = [256, 256];
        
        
        let input = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        let padded = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size * 2,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // let test = data.as_ptr() as *const u8;
        let copy_slice = unsafe{ std::slice::from_raw_parts(data.as_ptr() as *const u8, size as usize) };
        
        state.queue.write_buffer(&input, 0, copy_slice);
        let mut plan = fft::FftPlan::create(&shape, shaders, &state.device, &state.queue);
        
        let mut encoder = state.device.create_command_encoder(&Default::default());
        plan.pad(&mut encoder, &input, &padded);
        // plan.fft(&mut encoder, &padded, false, false);
        state.queue.submit(Some(encoder.finish()));


        // let slice = plan.twiddle_factors[0].slice(..);
        let slice = padded.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        state.device.poll(wgpu::Maintain::Wait);
        let read_data = slice.get_mapped_range();
        std::fs::write("out.bin", &read_data[..]).unwrap();

    }
}
