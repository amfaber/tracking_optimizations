#![allow(warnings)]
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
    let padded = plan.create_buffer(&state.device, false);
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
    // plan.pad(&state.device, &mut encoder, &input, &padded, &[shape[0], shape[1]]);
    // state.queue.submit(Some(encoder.finish()));

    let n = 250;
    let now = Instant::now();
    let fftpass = plan.fft_pass(&padded, &state.device);
    for _i in 0..n{
        let mut encoder = state.device.create_command_encoder(&Default::default());
        fftpass.execute(&mut encoder, false, false);
        fftpass.execute(&mut encoder, true, true);
        // plan.fft(&mut encoder, &padded, true, true);
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


pub fn inspect_buffer(
    buffer_to_inspect: &wgpu::Buffer,
    mappable_buffer: &wgpu::Buffer,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    device: &wgpu::Device,
    copy_size: u64,
    file_path: &str,
    ) -> !{
    encoder.copy_buffer_to_buffer(&buffer_to_inspect, 0, mappable_buffer, 0, copy_size);
    let slice = mappable_buffer.slice(..);
    let (sender, recv) = std::sync::mpsc::channel();
    queue.submit(Some(encoder.finish()));
    slice.map_async(wgpu::MapMode::Read, move |res|{sender.send(res);});
    device.poll(wgpu::Maintain::Wait);
    let idk = recv.recv().unwrap().unwrap();
    let data = slice.get_mapped_range()[..].to_vec();
    std::fs::write(file_path, &data);
    drop(slice);
    mappable_buffer.unmap();
    panic!("intended panic")
}



fn _gpu_save(argshape: [u32; 2], do_fft: bool) {
    // let shape = [524u32, 800];
    // let data = std::fs::read("grey_lion.bin").unwrap();
    // let data: Vec<_> = data.into_iter().map(|e| e as f32).collect();

    let shape = argshape;
    let data = (0..shape[0] * shape[1]).map(|_i| _i as f32).collect::<Vec<_>>();
    assert_eq!(data.len() as u32, shape[0] * shape[1]);
    let state = GpuState::new();
    let shaders =
        fft::compile_shaders(&state.device, None, None);
    
    // let radius = 10;
    // const radius: usize = 10;
    // const diameter: usize = 2*radius + 1;
    // const area: usize = diameter.pow(2);
    // let filter = [1./(area as f32); area];
    // let filtershape = [diameter as u32; 2];
    
    // let filter: [f32; 9] = [
    //      0., -1., -0.,
    //     -1.,  4., -1.,
    //      0., -1., -0.,
    // ];
    // let filtershape = [3; 2];

    // let filter = [1f32; 65536];
    // let filtershape = [256, 256];

    let filtershape = argshape;
    let filter = (0..filtershape[0]*filtershape[1]).map(|i| 1./(filtershape[0]*filtershape[1]) as f32).collect::<Vec<_>>();
    // let filter = (0..filtershape[0]*filtershape[1]).map(|i| i as f32).collect::<Vec<_>>();


    let input = state.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        contents: bytemuck::cast_slice(&data),
    });

    let filterbuffer = state.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | 
        wgpu::BufferUsages::MAP_READ,
        contents: bytemuck::cast_slice(&filter),
    });
    

    let padded_shape = fft::get_shape(&shape, &filtershape);

    let plan = fft::FftPlan::create(&padded_shape, shaders, &state.device, &state.queue);
    let padded = plan.create_buffer(&state.device, true);
    let filterpadded = plan.create_buffer(&state.device, true);

    let copy_slice = bytemuck::cast_slice::<f32, u8>(&data);
    
    state.queue.write_buffer(&input, 0, copy_slice);
    
    let mut encoder = state.device.create_command_encoder(&Default::default());
    
    let (pass, push_constants) = plan.pad_pass(&state.device, &input, &padded, &[shape[0], shape[1]]);
    inspect_buffer(
        &padded,
        &filterpadded,
        &state.queue,
        encoder,
        &state.device,
        (padded_shape[0]*padded_shape[1]*8) as u64,
        "dump.bin");
    
    pass.execute(&mut encoder, bytemuck::cast_slice(&push_constants));
    let (pass, push_constants) = plan.pad_pass(&state.device, &filterbuffer, &filterpadded, &filtershape);
    pass.execute(&mut encoder, bytemuck::cast_slice(&push_constants));
    state.queue.submit(Some(encoder.finish()));

    let fft_pass_filter = plan.fft_pass(&filterpadded, &state.device);
    let fft_pass_image = plan.fft_pass(&padded, &state.device);

    let convolution = plan.inplace_spectral_convolution_pass(&state.device, &padded, &filterpadded);
    
    let padded2 = plan.create_buffer(&state.device, true);
    let filter_convolve = plan.inplace_spectral_convolution_pass(&state.device, &filterpadded, &padded2);

    let copy_size = plan.params.dims[0] * plan.params.dims[1] * 8;
    if do_fft{
        let mut encoder = state.device.create_command_encoder(&Default::default());
        fft_pass_filter.execute(&mut encoder, false, false);
        // fft_pass_filter.execute(&mut encoder, true, true);
        fft_pass_image.execute(&mut encoder, false, false);
        encoder.copy_buffer_to_buffer(&filterpadded, 0, &padded2, 0, copy_size as u64);
        filter_convolve.execute(&mut encoder, &[]);
        convolution.execute(&mut encoder, &[]);
        fft_pass_image.execute(&mut encoder, true, true);
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
    // _gpu_save([args.shape_rows, args.shape_cols], do_fft)
    let state = GpuState::new();
    let source = include_str!("shaders/fft.wgsl");
    let module = state.device.create_shader_module(wgpu::ShaderModuleDescriptor{
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let entries = [
        wgpu::BindGroupLayoutEntry{
            binding: 0,
            ty: wgpu::BindingType::Buffer{
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            visibility: wgpu::ShaderStages::COMPUTE,
            count: None,
        },
        wgpu::BindGroupLayoutEntry{
            binding: 1,
            ty: wgpu::BindingType::Buffer{
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            visibility: wgpu::ShaderStages::COMPUTE,
            count: None,
        },
        // wgpu::BindGroupLayoutEntry{
        //     binding: 2,
        //     ty: wgpu::BindingType::Buffer{
        //         ty: wgpu::BufferBindingType::Storage { read_only: true },
        //         has_dynamic_offset: false,
        //         min_binding_size: None,
        //     },
        //     visibility: wgpu::ShaderStages::COMPUTE,
        //     count: None,
        // },

    ];
    
    let bind_group_layouts = state.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: None,
        entries: entries.as_slice()
    });

    let pipelinelayout = state.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
        label: None,
        bind_group_layouts: &[&bind_group_layouts],
        push_constant_ranges: &[wgpu::PushConstantRange{
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..16,
        }],
    });

    let idk = state.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
        label: None,
        layout: Some(&pipelinelayout),
        module: &module,
        entry_point: "main",
    });

}