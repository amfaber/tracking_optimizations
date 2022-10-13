use crate::{execute_gpu::TrackingParams, kernels};

use wgpu::{Buffer, Device, self, util::DeviceExt};

pub struct GpuParams{
    pub pic_dims: [u32; 2],
    pub composite_dims: [u32; 2],
    // pub constant_dims: [u32; 2],
    pub circle_dims: [u32; 2],
    pub dilation_dims: [u32; 2],
    pub max_iterations: u32,
    pub shift_threshold: f32,
    pub minmass: f32,
}
pub struct GpuBuffers{
    pub staging_buffers: Vec<Buffer>,
    pub frame_buffer: Buffer,
    pub composite_buffer: Buffer,
    pub gauss_1d_buffer: Buffer,
    pub processed_buffer: Buffer,
    pub centers_buffer: Buffer,
    pub circle_buffer: Buffer,
    pub masses_buffer: Buffer,
    pub result_buffer: Buffer,
    pub param_buffer: Buffer,
    pub max_rows: Buffer,
}

fn gpuparams_from_tracking_params(params: TrackingParams, pic_dims: [u32; 2]) -> GpuParams {
    let kernel_size = params.smoothing_size;
    let circle_size = params.diameter;
    let dilation_size = (2. * params.separation as f32 / (2 as f32).sqrt()) as u32;
    GpuParams{
        pic_dims,
        composite_dims: [kernel_size, kernel_size],
        // constant_dims: [kernel_size, kernel_size],
        circle_dims: [circle_size, circle_size],
        dilation_dims: [dilation_size, dilation_size],
        max_iterations: params.max_iterations,
        shift_threshold: 0.6,
        minmass: params.minmass,
    }
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        ::std::mem::size_of::<T>(),
    )
}

pub fn setup_buffers(tracking_params: &TrackingParams,
    device: &wgpu::Device,
    n_result_columns: u64,
    size: u64,
    dims: &[u32; 2],
    ) -> GpuBuffers{
    let params = gpuparams_from_tracking_params(tracking_params.clone(), *dims);
    let mut staging_buffers = Vec::new();
    for i in 0..2{
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Staging {}", i).as_str()),
            size: (n_result_columns * size) as u64,
            usage: wgpu::BufferUsages::COPY_SRC 
            | wgpu::BufferUsages::COPY_DST 
            | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        staging_buffers.push(staging_buffer);
    }
    // let mut free_staging_buffers = staging_buffers.iter().collect::<Vec<&wgpu::Buffer>>();
    // let mut in_use_staging_buffers = VecDeque::new();
    let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let sigma = tracking_params.noise_size as f32;
    // let gaussian_kernel = kernels::Kernel::tp_gaussian(sigma, 4.);
    let composite_kernel = kernels::Kernel::composite_kernel(sigma, params.composite_dims);
    let composite_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&composite_kernel.data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // let noise_size = tracking_params.noise_size;
    // let constant_kernel = kernels::Kernel::rolling_average([noise_size, noise_size]);
    // let constant_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //     label: None,
    //     contents: bytemuck::cast_slice(&constant_kernel.data),
    //     usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    // });

    let processed_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let centers_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (2 * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let r = tracking_params.diameter / 2;
    let circle_kernel = kernels::Kernel::circle_mask(r);
    let circle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&circle_kernel.data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let masses_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n_result_columns * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    

    let param_buffer = unsafe{
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Width Buffer"),
            contents: any_as_u8_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM
        })
    };

    let gauss_1d_kernel = kernels::Kernel::gauss_1d(sigma, params.composite_dims[0]);
    let gauss_1d_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&gauss_1d_kernel.data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let max_rows = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });


    GpuBuffers{
        staging_buffers,
        frame_buffer,
        composite_buffer,
        gauss_1d_buffer,
        processed_buffer,
        centers_buffer,
        circle_buffer,
        masses_buffer,
        result_buffer,
        param_buffer,
        max_rows,
    }
}
