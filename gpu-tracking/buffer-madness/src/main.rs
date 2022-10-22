use std::{fs::OpenOptions, io::Write};

use buffer_madness;
use pollster::FutureExt;
// use futures_intrusive;
use bytemuck;

fn main() {
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptionsBase{
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })
    .block_on().unwrap();

    let mut desc = wgpu::DeviceDescriptor::default();
    desc.features = wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
    let (device, queue) = adapter
    .request_device(&desc, None)
    .block_on().unwrap();

    let buffers = buffer_madness::buffers::make_buffers(&device);

    let shader = include_str!("shaders/shader.wgsl");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
        label: Some("shader"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });

    let pipelines = [("fill", &shader), ("idk", &shader)];

    let pipelines = pipelines.iter().map(|(entry, shader)|{
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: None,
            layout: None,
            module: shader,
            entry_point: entry,
        })
    }).collect::<Vec<_>>();

    let bind_group_layouts = pipelines.iter().map(|pipeline|{
        pipeline.get_bind_group_layout(0)
    }).collect::<Vec<_>>();


    let bind_group_entries = [
        vec![
            (0, &buffers.intermediate),
            // (1, buffers.output),
        ],
        vec![
            (0, &buffers.intermediate),
            (1, &buffers.output),
        ],
    ];


    let bind_groups = bind_group_layouts.iter().zip(bind_group_entries).map(|(layout, entries)|{
        let entries = entries.iter().map(|(binding, buffer)|{
            wgpu::BindGroupEntry{
                binding: *binding,
                resource: buffer.as_entire_binding(),
            }
        }).collect::<Vec<_>>();
        device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: None,
            layout,
            entries: &entries[..],
        })
    }).collect::<Vec<_>>();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
        label: Some("encoder"),
    });

    pipelines.iter().zip(bind_groups.iter()).for_each(|(pipeline, bind_group)|{
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.set_pipeline(pipeline);
        cpass.dispatch_workgroups(2, 2, 1);
    });

    queue.submit(Some(encoder.finish()));
    let buffer_slice = buffers.output.slice(..);
    // let (sender, receiver) = 
    //         futures_intrusive::channel::shared::oneshot_channel();

    // buffer_slice.map_async(wgpu::MapMode::Read, move |v|{sender.send(v).unwrap()});
    buffer_slice.map_async(wgpu::MapMode::Read, move |_|{});
    // dbg!("before poll");
    device.poll(wgpu::Maintain::Wait);
    // dbg!("after poll");
    // receiver.receive().block_on().unwrap().unwrap();
    let data = buffer_slice.get_mapped_range().to_vec();
    // let mut options = OpenOptions::new().create(true);
    // options.truncate(true);
    let mut file = OpenOptions::new().write(true).create(true).truncate(true).open("output.bin").unwrap();
    file.write(&data).unwrap();
    
    // let data = bytemuck::cast_slice::<u8, f32>(&data);
    // dbg!(data);
}
