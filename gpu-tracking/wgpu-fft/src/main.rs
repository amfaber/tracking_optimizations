use wgpu_fft::fft;


use std::time::Instant;

use wgpu::{self, util::DeviceExt};
use pollster::FutureExt;
// use bytemuck;

struct GpuState{
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
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
        desc.features |= wgpu::Features::PUSH_CONSTANTS;
        desc.limits.max_push_constant_size = 128;
        // desc.limits.max_compute_invocations_per_workgroup = 512;
        // desc.limits.max_compute_workgroup_size_x = 512;
        
        let (device, queue) = adapter
        .request_device(&desc, None).block_on().unwrap();
        
        Self{
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
        }
    }
}

fn _gpu_speed(shape: [u32; 2]) {
    // let shape = [300, 300];
    let data = (0..shape[0] * shape[1]).map(|_i| _i as f32).collect::<Vec<_>>();
    let size = std::mem::size_of::<f32>() * data.len();
    let size = size as u64;
    let state = GpuState::new();
    let shaders =
        fft::compile_shaders(&state.device, None, None);

    let input = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let padded_shape = fft::get_shape(&shape, &[1, 1]);

    let mut plan = fft::FftPlan::create(&padded_shape, shaders, &state.device, &state.queue);
    let padded = plan.create_buffer(false);
    // let padded = state.device.create_buffer(&wgpu::BufferDescriptor {
    //     label: None,
    //     size: size * 2,
    //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC 
    //     // | wgpu::BufferUsages::MAP_READ
    //     ,
    //     mapped_at_creation: false,
    // });
    // let test = data.as_ptr() as *const u8;
    // let copy_slice = unsafe{ std::slice::from_raw_parts(data.as_ptr() as *const u8, size as usize) };
    let copy_slice = bytemuck::cast_slice::<f32, u8>(&data);
    
    state.queue.write_buffer(&input, 0, copy_slice);
    
    let mut encoder = state.device.create_command_encoder(&Default::default());
    plan.pad(&mut encoder, &input, &padded, &[shape[0], shape[1]]);
    state.queue.submit(Some(encoder.finish()));

    let n = 250;
    let now = Instant::now();
    for _i in 0..n{
        let mut encoder = state.device.create_command_encoder(&Default::default());
        plan.fft(&mut encoder, &padded, false, false);
        plan.fft(&mut encoder, &padded, true, true);
        state.queue.submit(Some(encoder.finish()));
    }
    // let slice = plan.twiddle_factors[0].slice(..);
    // let slice = padded.slice(..);
    // slice.map_async(wgpu::MapMode::Read, |_| {});
    state.device.poll(wgpu::Maintain::Wait);
    dbg!(now.elapsed().as_nanos() as f64 / 1_000_000_000.);
    // // dbg!(now.elapsed().as_nanos() as f64 / (2. * n as f64));
    // let read_data = slice.get_mapped_range();
    // std::fs::write("out.bin", &read_data[..]).unwrap();

}
fn _gpu_save(shape: [u32; 2], do_fft: bool) {
    let shape = [524u32, 800];
    let data = std::fs::read("grey_lion.bin").unwrap();
    // let data = unsafe{
    //     data.chunks(4)
    //     .map(|chunk| std::mem::transmute::<[u8;4], f32>(chunk.try_into().unwrap())).collect::<Vec<_>>()
    // };
    let data: Vec<_> = data.into_iter().map(|e| e as f32).collect();
    assert_eq!(data.len() as u32, shape[0] * shape[1]);
    // let data = (0..shape[0] * shape[1]).map(|_i| _i as f32).collect::<Vec<_>>();
    let state = GpuState::new();
    let shaders =
        fft::compile_shaders(&state.device, None, None);
    
    // let radius = 10;
    // const radius: usize = 10;
    // const diameter: usize = 2*radius + 1;
    // const area: usize = diameter.pow(2);
    // let filter = [1./(area as f32); area];
    // let filtershape = [diameter as u32; 2];
    
    let filter: [f32; 9] = [
         0., -1., -0.,
        -1.,  4., -1.,
         0., -1., -0.,
    ];

    let filtershape = [3; 2];

    let input = state.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        contents: bytemuck::cast_slice(&data),
    });

    let filterbuffer = state.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        contents: bytemuck::cast_slice(&filter),
    });
    

    let padded_shape = fft::get_shape(&shape, &filtershape);

    let mut plan = fft::FftPlan::create(&padded_shape, shaders, &state.device, &state.queue);
    let padded = plan.create_buffer(true);
    let filterpadded = plan.create_buffer(false);

    let copy_slice = bytemuck::cast_slice::<f32, u8>(&data);
    
    state.queue.write_buffer(&input, 0, copy_slice);
    
    let mut encoder = state.device.create_command_encoder(&Default::default());
    plan.pad(&mut encoder, &input, &padded, &[shape[0], shape[1]]);
    plan.pad(&mut encoder, &filterbuffer, &filterpadded, &filtershape);
    state.queue.submit(Some(encoder.finish()));

    if do_fft{
        let mut encoder = state.device.create_command_encoder(&Default::default());
        plan.fft(&mut encoder, &padded, false, false);
        plan.fft(&mut encoder, &filterpadded, false, false);
        plan.spectral_convolution(&mut encoder, &padded, &filterpadded);
        plan.fft(&mut encoder, &padded, true, true);
        state.queue.submit(Some(encoder.finish()));
    }
    let slice = padded.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    state.device.poll(wgpu::Maintain::Wait);
    let read_data = slice.get_mapped_range();
    std::fs::write("out.bin", &read_data[..]).unwrap();

}

use clap::Parser;

#[derive(Parser, Debug)]
struct Args{
    shape_rows: u32,
    shape_cols: u32,

    #[arg(long)]
    fft: Option<bool>,
}

fn main(){
    let args = Args::parse();
    let do_fft = args.fft.unwrap_or(true);
    _gpu_save([args.shape_rows, args.shape_cols], do_fft)
}