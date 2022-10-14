#![allow(warnings)]
use futures;
use futures_intrusive;
type my_dtype = f32;
use ndarray::{ArrayBase, ViewRepr, Array2, ArrayView2, ArrayView3};
use pollster::FutureExt;
use std::{collections::VecDeque, time::Instant, io::{BufRead, Write, Read}, fs::{self, File}};
use wgpu::{util::DeviceExt, Buffer};
use crate::{kernels, into_slice::IntoSlice, buffer_setup, linking::ReturnDistance};
use std::collections::HashMap;
use kd_tree;
use rayon::prelude::*;

type inner_output = f32;
type output_type = Vec<inner_output>;

// #[derive(Clone, Copy)]
// pub struct GpuParams{
//     pub pic_dims: [u32; 2],
//     pub gauss_dims: [u32; 2],
//     pub constant_dims: [u32; 2],
//     pub circle_dims: [u32; 2],
//     pub max_iterations: u32,
//     pub shift_threshold: f32,
//     pub minmass: f32,
// }

// pub struct Arguments{
//     pub diameter: u32,
//     pub minmass: Option<f32>,
//     pub max_size: Option<f32>,
//     pub separation: Option<u32>,
//     pub noise_size: Option<u32>,
//     pub smoothing_size: Option<u32>,
//     pub threshold: Option<f32>,
//     pub invert: Option<bool>,
//     pub percentile: Option<f32>,
//     pub topn: Option<u32>,
//     pub preprocess: Option<bool>,
//     pub max_iterations: Option<u32>,
//     pub characterize: Option<bool>,
// }

#[derive(Clone, Copy)]
pub struct TrackingParams{
    pub diameter: u32,
    pub minmass: f32,
    pub maxsize: f32,
    pub separation: u32,
    pub noise_size: u32,
    pub smoothing_size: u32,
    pub threshold: f32,
    pub invert: bool,
    pub percentile: f32,
    pub topn: u32,
    pub preprocess: bool,
    pub max_iterations: u32,
    pub characterize: bool,
    pub filter_close: bool,
}

impl Default for TrackingParams{
    fn default() -> Self {
        TrackingParams{
            diameter: 9,
            minmass: 0.,
            maxsize: 0.0,
            separation: 11,
            noise_size: 1,
            smoothing_size: 9,
            threshold: 0.0,
            invert: false,
            percentile: 0.,
            topn: 0,
            preprocess: true,
            max_iterations: 10,
            characterize: false,
            filter_close: true,
        }
    }
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        ::std::mem::size_of::<T>(),
    )
}

fn get_work(finished_staging_buffer: &Buffer,
    device: &wgpu::Device,
    old_submission: wgpu::SubmissionIndex,
    wait_gpu_time: &mut f32,
    output: &mut Vec<inner_output>,
    pic_size: usize,
    n_result_columns: u64,
    frame_index: usize,
    debug: bool,
    separation: my_dtype,
    filter: bool,
    ) -> () {
    let mut buffer_slice = finished_staging_buffer.slice(..);
    let (sender, receiver) = 
            futures_intrusive::channel::shared::oneshot_channel();
    
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    let now = Instant::now();
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(old_submission));
    (*wait_gpu_time) += now.elapsed().as_millis() as f32 / 1000.;
    receiver.receive().block_on().unwrap();
    let data = buffer_slice.get_mapped_range();
    let dataf32 = bytemuck::cast_slice::<u8, f32>(&data);
    let mut positions = Vec::new();
    let mut properties = Vec::new();
    let mut idx = 0usize;
    match debug{
        _ => {
            for i in 0..pic_size{
                let mass = dataf32[i];
                if mass != 0.0{
                    positions.push(([dataf32[i+pic_size], dataf32[i+2*pic_size]], (idx, mass)));
                    // properties.push(mass);
                    for j in 3..n_result_columns as usize{
                        properties.push(dataf32[i + j * pic_size]);
                    }
                    idx += 1;
                }
            }
            if filter{
                // let sep2 = separation.powi(2);
                let tree = kd_tree::KdTree::build_by_ordered_float(positions);
                let relevant_points = tree.iter().map(|query| {
                    let positions = query.0;
                    let (idx, mass) = query.1;
                    let neighbors = tree.within_radius(query, separation);
                    // dbg!(query);
                    // dbg!(&neighbors);
                    // let item = neighbor.item;
                    // let neighbor_positions = item.0;
                    // let (neighbor_idx, neighbor_mass) = item.1;
                    let mut keep_point = true;
                    for neighbor in neighbors{
                        let (neighbor_idx, neighbor_mass) = neighbor.1;
                        if neighbor_idx != idx{
                            if mass < neighbor_mass{
                                keep_point = false;
                                break;
                            } else if mass == neighbor_mass{
                                if idx > neighbor_idx{
                                    keep_point = false;
                                    break;
                                }
                            }
                        }
                    }
                    // if neighbor.squared_distance < sep2{
                    //     if mass < neighbor_mass{
                    //         keep_point = false;
                    //     } else if mass == neighbor_mass{
                    //         if idx > neighbor_idx{
                    //             keep_point = false;
                    //         }
                    //     }
                    // }
                    if keep_point{
                        Some((idx, positions, mass))
                    } else {
                        // dbg!("here");
                        None
                    }
                }).flatten()
                .collect::<Vec<_>>();
                // dbg!(relevant_points.len());
                // dbg!(tree.len());
                for point in relevant_points{
                    let (idx, positions, mass) = point;
                    output.push(frame_index as my_dtype);
                    output.push(mass);
                    output.push(positions[0]);
                    output.push(positions[1]);
                    for j in 0..n_result_columns as usize - 3{
                        output.push(properties[idx * (n_result_columns as usize - 3) + j]);
                    }
                }
            } else {
                for point in positions{
                    let (positions, (idx, mass)) = point;
                    output.push(frame_index as my_dtype);
                    output.push(mass);
                    output.push(positions[0]);
                    output.push(positions[1]);
                    for j in 0..n_result_columns as usize - 3{
                        output.push(properties[idx * (n_result_columns as usize - 3) + j]);
                    }
                }
            }

        },
        true => {
            for i in 0..pic_size*n_result_columns as usize{
                output.push(dataf32[i]);
            }
        },
    }
    // let elapsed = now.elapsed().as_nanos() as inner_output / 1_000_000.;
    // result.push(result);
    drop(data);
    finished_staging_buffer.unmap();
    
}




pub fn execute_gpu<'a, T: Iterator<Item = impl IntoSlice>>(
    mut frames: T,
    dims: &[u32; 2],
    tracking_params: TrackingParams,
    debug: bool,
    ) -> (output_type, (usize, usize)){
    
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptionsBase{
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })
    .block_on()
    .ok_or(anyhow::anyhow!("Couldn't create the adapter")).unwrap();

    let mut desc = wgpu::DeviceDescriptor::default();
    desc.features = wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
    let (device, queue) = adapter
    .request_device(&desc, None)
    .block_on().unwrap();

    
    let common_header = include_str!("shaders/params.wgsl");

    // let shaders = [
    //     include_str!("shaders/preprocess.wgsl"),
    //     include_str!("shaders/another_backup_preprocess.wgsl"),
    //     include_str!("shaders/centers.wgsl"),
    //     include_str!("shaders/walk.wgsl"),
    // ];

    let shaders = HashMap::from([
        // ("preprocess", "src/shaders/preprocess.wgsl"),
        ("proprocess_backup", "src/shaders/another_backup_preprocess.wgsl"),
        ("centers", "src/shaders/centers.wgsl"),
        ("centers_outside_parens", "src/shaders/centers.wgsl"),
        ("max_rows", "src/shaders/max_rows.wgsl"),
        ("walk", "src/shaders/walk.wgsl"),
        ("walk_cols", "src/shaders/walk_cols.wgsl"),
        ("preprocess_rows", "src/shaders/preprocess_rows.wgsl"),
        ("preprocess_cols", "src/shaders/preprocess_cols.wgsl"),
    ]);

    let shaders = shaders.iter().map(|(&name, shader)| {
        let mut shader_file = File::open(shader).unwrap();
        let mut shader_string = String::new();
        shader_file.read_to_string(&mut shader_string).unwrap();
        (name, shader_string)
    }).collect::<HashMap<_, _>>();

    let workgroup_size = [16, 16, 1];
    let workgroups: [u32; 2] = 
        dims.iter().zip(workgroup_size)
        .map(|(&x, size)| (x + size - 1) / size).collect::<Vec<u32>>().try_into().unwrap();

    let preprocess_source = |source| -> String {
        let mut result = String::new();
        result.push_str(common_header);
        result.push_str(source);
        result.replace("@workgroup_size(_)",
        format!("@workgroup_size({}, {}, {})", workgroup_size[0], workgroup_size[1], workgroup_size[2]).as_str())
    };


    let shaders = shaders.iter().map(|(&name, source)|{
        let shader_source = preprocess_source(source);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        (name, shader)
    }).collect::<HashMap<_, _>>();
    let pic_size = dims.iter().product::<u32>() as usize;
    let n_result_columns: u64 = match debug{
        _ => 3,
        true => 1,
    };
    let slice_size = pic_size * std::mem::size_of::<my_dtype>();
    let size = slice_size as wgpu::BufferAddress;
    
    
    let buffers = buffer_setup::setup_buffers(&tracking_params, &device, n_result_columns, size, dims);

    let pipelines = match debug{
        // false => vec![
        //     // ("rows", &shaders[0]),
        //     // ("finish", &shaders[0]),
        //     ("preprocess", 0, &shaders["preprocess_backup"]),
        //     ("centers", 0, &shaders["centers"]),
        //     ("walk", 0, &shaders["walk"]),
        // ],
        _ => vec![
            ("pp_rows", 0, &shaders["preprocess_rows"]),
            ("pp_cols", 0, &shaders["preprocess_cols"]),
            // ("centers", 0, &shaders["centers"]),
            ("centers", 0, &shaders["centers_outside_parens"]),
            ("max_row", 0, &shaders["max_rows"]),
            ("walk", 0, &shaders["walk_cols"]),
        ],
    };

    let compute_pipelines = pipelines.iter().map(|(name, group, shader)|{
        (*name, device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: shader,
        entry_point: "main",
        }))
    }).collect::<Vec<_>>();

    let bind_group_layouts = compute_pipelines.iter()
    .zip(pipelines.iter())
    .map(|((name, pipeline), (entry, group, shader))|{
        (*name, (*group, pipeline.get_bind_group_layout(*group)))
    }).collect::<HashMap<_, _>>();


    let bind_group_entries = match debug{
        // false => HashMap::from([
        //     ("preprocess", vec![ // another_backup_preprocess.wgsl
        //         (0, &buffers.param_buffer),
        //         (1, &buffers.frame_buffer),
        //         (2, &buffers.composite_buffer),
        //         (3, &buffers.processed_buffer), 
        //         ]),
        //     ("centers", vec![
        //         (0, &buffers.param_buffer),    
        //         (1, &buffers.circle_buffer), 
        //         (2, &buffers.processed_buffer), 
        //         (3, &buffers.centers_buffer),
        //         (4, &buffers.masses_buffer),
        //     ]),
        //     ("walk", vec![
        //         (0, &buffers.param_buffer),
        //         (1, &buffers.processed_buffer), 
        //         (2, &buffers.centers_buffer),
        //         (3, &buffers.masses_buffer),
        //         (4, &buffers.result_buffer),
        //     ]),
        // ]),
        _ => HashMap::from([
            ("pp_rows", vec![
                (0, &buffers.param_buffer),
                (1, &buffers.frame_buffer),
                (2, &buffers.gauss_1d_buffer),
                (3, &buffers.centers_buffer),
            ]),
            ("pp_cols", vec![
                (0, &buffers.param_buffer),
                (1, &buffers.gauss_1d_buffer),
                (2, &buffers.centers_buffer),
                (3, &buffers.processed_buffer), 
            ]),
            ("centers", vec![
                (0, &buffers.param_buffer),    
                (1, &buffers.circle_buffer), 
                (2, &buffers.processed_buffer), 
                (3, &buffers.centers_buffer),
                (4, &buffers.masses_buffer),
            ]),
            ("max_row", vec![
                (0, &buffers.param_buffer),
                (1, &buffers.processed_buffer), 
                (2, &buffers.max_rows),
            ]),
            ("walk", vec![
                (0, &buffers.param_buffer),
                (1, &buffers.processed_buffer), 
                (2, &buffers.max_rows),
                (3, &buffers.centers_buffer),
                (4, &buffers.masses_buffer),
                (5, &buffers.result_buffer),
            ]),
        ])
    };

    let bind_group_entries = bind_group_entries
        .iter().map(|(&name, group)| (name, group.iter().map(|(i, buffer)|
            wgpu::BindGroupEntry {
            binding: *i as u32,
            resource: buffer.as_entire_binding()}).collect::<Vec<_>>())
        )
        .collect::<HashMap<_, _>>();
    
    
    let bind_groups = bind_group_layouts.iter()//.zip(bind_group_entries.iter())
        .map(|(&name, (group, layout))|{
        let entries = &bind_group_entries[name];
        (name, (*group, device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries[..],
        })))}
        ).collect::<HashMap<_, _>>();
    
    let submit_work = |staging_buffer: &wgpu::Buffer, frame: &[my_dtype]| {
        queue.write_buffer(&staging_buffer, 0, bytemuck::cast_slice(frame));
        let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        
        encoder.copy_buffer_to_buffer(staging_buffer, 0,
            &buffers.frame_buffer, 0, size);
            
        compute_pipelines.iter().for_each(|(name, pipeline)|{
            let (group, bind_group) = &bind_groups[name];
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_bind_group(*group, bind_group, &[]);
            cpass.set_pipeline(pipeline);
            cpass.dispatch_workgroups(workgroups[0], workgroups[1], 1);
        });
        let output_buffer = match debug {
            false => &buffers.result_buffer,
            // true => &buffers.result_buffer,
            true => &buffers.result_buffer,
        };
        encoder.copy_buffer_to_buffer(output_buffer, 0,
            staging_buffer, 0, n_result_columns * size);
        
        encoder.clear_buffer(&buffers.result_buffer, 0, None);
        let index = queue.submit(Some(encoder.finish()));
        // let mut encoder =
        //     device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // queue.submit(Some(encoder.finish()));
        index
    };
    
    let mut wait_gpu_time = 0.;

    // let mut output: Vec<Vec<f32>> = Vec::new();
    let mut output: output_type = Vec::new();
    
    let mut free_staging_buffers = buffers.staging_buffers.iter().collect::<Vec<&wgpu::Buffer>>();
    let mut in_use_staging_buffers = VecDeque::new();
    let frame = frames.next().unwrap();
    let frame_slice = frame.into_slice();

    let mut frame_index = 0;
    let staging_buffer = free_staging_buffers.pop().unwrap();
    in_use_staging_buffers.push_back(staging_buffer);
    // device.start_capture();
    let mut old_submission = submit_work(staging_buffer, frame.into_slice());

    // let stdin = std::io::stdin();
    // let mut lock = stdin.lock();
    // let mut _buf = String::new();
    // lock.read_line(&mut _buf);

    for frame in frames{
        let frame = frame;
        let staging_buffer = free_staging_buffers.pop().unwrap();
        in_use_staging_buffers.push_back(staging_buffer);
        let new_submission = submit_work(staging_buffer, &frame.into_slice());
        
        let finished_staging_buffer = in_use_staging_buffers.pop_front().unwrap();
        
        get_work(finished_staging_buffer,
            &device, 
            old_submission, 
            &mut wait_gpu_time,
            &mut output,
            pic_size,
            n_result_columns,
            frame_index,
            debug,
            tracking_params.separation as f32,
            tracking_params.filter_close,
        );


        free_staging_buffers.push(finished_staging_buffer);
        old_submission = new_submission;
        frame_index += 1;
    }
    dbg!(wait_gpu_time);
    let finished_staging_buffer = in_use_staging_buffers.pop_front().unwrap();
    
    get_work(finished_staging_buffer,
        &device, 
        old_submission, 
        &mut wait_gpu_time,
        &mut output,
        pic_size,
        n_result_columns,
        frame_index,
        debug,
        tracking_params.separation as f32,
        tracking_params.filter_close,
    );


    let n_result_columns = n_result_columns as usize + 1;
    let shape = (output.len() / n_result_columns, n_result_columns);
    (output, shape)
}

pub fn execute_ndarray(array: &ArrayView3<my_dtype>, params: TrackingParams, debug: bool) -> Array2<my_dtype> {
    if !array.is_standard_layout(){
        panic!("Array is not standard layout");
    }
    let axisiter = array.axis_iter(ndarray::Axis(0));
    let dims = array.shape();
    let (res, res_dims) = execute_gpu(axisiter, &[dims[1] as u32, dims[2] as u32], params, debug);
    let res = Array2::from_shape_vec(res_dims, res).unwrap();
    res
}