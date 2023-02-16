use wgpu;
use gpu_tracking::execute_gpu::path_to_iter;
use ndarray::{Array, Axis};
use pollster::FutureExt;

fn main() {
    let path = "testing/easy_test_data.tif";
    let (provider, dims) = path_to_iter(&path, None).unwrap();
    let data = provider.into_iter().flat_map(|vec| vec.unwrap().into_iter()).collect::<Vec<_>>();
    let array = Array::from_shape_vec([data.len() / (dims[0] * dims[1]) as usize, dims[0] as usize, dims[1] as usize], data).unwrap();

    let instance = wgpu::Instance::new(
        wgpu::InstanceDescriptor{
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        }
    );
    
    
    let adapter = instance.request_adapter(
        &wgpu::RequestAdapterOptionsBase{
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }
    ).block_on().unwrap();

    let mut desc = wgpu::DeviceDescriptor::default();
    // desc.features = wgpu::Features::MAPPABLE_PRIMARY_BUFFERS | wgpu::Features::PUSH_CONSTANTS;
    desc.limits.max_push_constant_size = 16;
    desc.limits.max_storage_buffers_per_shader_stage = 12;
    let (device, queue) = adapter
    .request_device(&desc, None)
    .block_on().unwrap();

    let shape = array.shape();

    let buffer = device.create_buffer(&wgpu::BufferDescriptor{
        label: None,
        size: (shape[1] * shape[2]) as u64 * 4,
        usage: 
        wgpu::BufferUsages::COPY_DST
        // wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC 
        // | wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
            ,
        mapped_at_creation: false,
    });

    let buffer2 = device.create_buffer(&wgpu::BufferDescriptor{
        label: None,
        size: (shape[1] * shape[2]) as u64 * 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });


    let now = std::time::Instant::now();

    // let bufslice = buffer.slice(..);
    // let (sender, receiver) = std::sync::mpsc::channel();
    // for frame in array.axis_iter(Axis(0)){
    //     let sender = sender.clone();
    //     bufslice.map_async(wgpu::MapMode::Write, move |res| {sender.send(res).unwrap();});
    //     queue.submit(None);
    //     // device.poll(wgpu::MaintainBase::Wait);
    //     receiver.recv().unwrap().unwrap();

    //     let mut mapped = bufslice.get_mapped_range_mut();
    //     mapped.clone_from_slice(bytemuck::cast_slice(frame.as_slice().unwrap()));
    //     drop(mapped);
    //     buffer.unmap();

    //     let mut encoder = device.create_command_encoder(&Default::default());
    //     encoder.copy_buffer_to_buffer(&buffer, 0, &buffer2, 0, buffer.size());
    //     queue.submit(Some(encoder.finish()));
    //     device.poll(wgpu::MaintainBase::Wait);
    // }

    for frame in array.axis_iter(Axis(0)){
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(frame.as_slice().unwrap()));
        queue.submit(None);
    }
    device.poll(wgpu::MaintainBase::Wait);
    // let elapsed2 = now.elapsed().as_secs_f64();
    // dbg!(elapsed2);
    // queue.submit(None);
    // device.poll(wgpu::MaintainBase::Wait);

    let elapsed = now.elapsed().as_secs_f64();
    dbg!(elapsed);
    // dbg!(array.shape());
}
