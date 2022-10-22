use wgpu::{self, Buffer};
// use wgpu::util::DeviceExt;
pub struct Buffers{
    pub intermediate: Buffer,
    pub output: Buffer,
}


pub fn make_buffers(device: &wgpu::Device) -> Buffers{
    let size = 1024 * 4;
    let intermediate = device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("intermediate"),
        size: 2 * size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let output = device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("output"),
        size: size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    
    Buffers{
        intermediate,
        output,
    }
}