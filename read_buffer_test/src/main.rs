use std::time::Instant;

use wgpu::{self, util::DeviceExt};
use pollster::FutureExt;
use futures_intrusive;

fn main() {
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }).block_on().unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).block_on().unwrap();
    let cpu_buffer = vec![5u8; 1_000_000];
    let gpu_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gpu_buffer"),
        contents: &cpu_buffer,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });
    queue.submit(None);
    let buffer_slice = gpu_buffer.slice(..);
    let (sender, receiver) = 
            futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);
    receiver.receive().block_on().unwrap().unwrap();
    let now = Instant::now();
    let data = buffer_slice.get_mapped_range();
    let result = data.to_vec();
    let elapsed = now.elapsed().as_nanos();
    println!("Elapsed: {} ms", elapsed as f64 / 1_000_000.0);
    dbg!(cpu_buffer[0]);
    dbg!(result[0]);

    
}
