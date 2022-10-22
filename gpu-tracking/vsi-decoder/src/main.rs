#![allow(warnings)]
use vsi_decoder::MinimalETSParser;
use std::{fs::File, io::Read};
use bencher::black_box;
use pollster::FutureExt;

fn _main(){
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptionsBase{
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })
    .block_on()
    .unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).block_on().unwrap();
    
    let buffer = device.create_buffer(&wgpu::BufferDescriptor{
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let atomic = device.create_buffer(&wgpu::BufferDescriptor{
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor{
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = {
        let mut file = File::open("src/shaders/shader.wgsl").unwrap();
        let mut string = String::new();
        file.read_to_string(&mut string).unwrap();
        device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: None,
            source: wgpu::ShaderSource::Wgsl(string.into()),
        })
    };

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
        label: None,
        layout: None,
        module: &shader,
        entry_point: "main",
    });

    let bind_group_entries = vec![
        wgpu::BindGroupEntry{
            binding: 0,
            resource: buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry{
            binding: 1,
            resource: atomic.as_entire_binding(),
        },
    ];
    
    let bind_group_layout = pipeline.get_bind_group_layout(0);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: None,
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
        label: None,
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
            label: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buffer, 0, &out_buffer, 0, 1024);
    queue.submit(Some(encoder.finish()));

    let mut buffer_slice = out_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_|{});
    device.poll(wgpu::Maintain::Wait);
    let data = buffer_slice.get_mapped_range();
    let dataf32 = bytemuck::cast_slice::<u8, f32>(&data);
    
    dbg!(&dataf32[..53]);
    drop(data);
    out_buffer.unmap();
}

fn main() {
    
    let mut file = File::open("../tiff_vsi/vsi dummy/_Process_9747_/stack1/frame_t_0.ets").unwrap();
    let mut parser = MinimalETSParser::new(&mut file).unwrap();
    let mut iter = parser.iterate_channel(file.try_clone().unwrap(), 1);
    let mut iter2 = parser.iterate_channel(file, 1);
    let now = std::time::Instant::now();
    // dbg!(iter.next());
    // for channel1 in iter{
    //     black_box(channel1);
    // }
    for (channel1, channel2) in iter.zip(iter2){
        black_box(channel1);
        black_box(channel2);
    }
    let elapsed = now.elapsed().as_micros() as f64 / 1e6;
    println!("Elapsed: {}", elapsed);
    // iter2.next();
}
