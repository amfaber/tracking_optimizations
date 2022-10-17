#![allow(warnings)]
use futures;
use futures_intrusive::{self, channel};
type my_dtype = f32;
use ndarray::{ArrayBase, ViewRepr, Array2, ArrayView2, ArrayView3};
use pollster::FutureExt;
use std::{collections::VecDeque, time::Instant, io::{BufRead, Write, Read}, fs::{self, File}, sync::mpsc::Sender};
use wgpu::{util::DeviceExt, Buffer};
use crate::{kernels, into_slice::IntoSlice, buffer_setup, linking::{ReturnDistance, Linker}};
use std::collections::HashMap;
use kd_tree;
use rayon::prelude::*;

type inner_output = f32;
type output_type = Vec<inner_output>;
type channel_type = Option<(Vec<([f32; 2], (usize, f32))>, Vec<f32>, usize)>;
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
    pub noise_size: f32,
    pub smoothing_size: u32,
    pub threshold: f32,
    pub invert: bool,
    pub percentile: f32,
    pub topn: u32,
    pub preprocess: bool,
    pub max_iterations: u32,
    pub characterize: bool,
    pub filter_close: bool,
    pub search_range: Option<my_dtype>,
    pub memory: Option<usize>,
}

impl Default for TrackingParams{
    fn default() -> Self {
        TrackingParams{
            diameter: 9,
            minmass: 0.,
            maxsize: 0.0,
            separation: 11,
            noise_size: 1.,
            smoothing_size: 9,
            threshold: 0.0,
            invert: false,
            percentile: 0.,
            topn: 0,
            preprocess: true,
            max_iterations: 10,
            characterize: false,
            filter_close: true,
            search_range: None,
            memory: None,
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
    // output: &mut Vec<inner_output>,
    pic_size: usize,
    result_read_depth: u64,
    frame_index: usize,
    debug: bool,
    separation: my_dtype,
    filter: bool,
    job_sender: &Sender<channel_type>,
    ) -> () {
    let mut buffer_slice = finished_staging_buffer.slice(..);
    let (sender, receiver) = 
            futures_intrusive::channel::shared::oneshot_channel();
    
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    let now = Instant::now();
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(old_submission));
    (*wait_gpu_time) += now.elapsed().as_nanos() as f32 / 1_000_000_000.;
    receiver.receive().block_on().unwrap();
    let data = buffer_slice.get_mapped_range();
    let dataf32 = bytemuck::cast_slice::<u8, f32>(&data);
    let mut positions = Vec::new();
    let mut properties = Vec::new();
    let mut idx = 0usize;
    match debug{
        false => {
            for i in 0..pic_size{
                let mass = dataf32[i];
                if mass != 0.0{
                    positions.push(([dataf32[i+pic_size], dataf32[i+2*pic_size]], (idx, mass)));
                    for j in 3..result_read_depth as usize{
                        properties.push(dataf32[i + j * pic_size]);
                    }
                    idx += 1;
                }
            }
        },
        true => {
            for i in 0..pic_size*result_read_depth as usize{
                properties.push(dataf32[i]);
            }
        },
    }
    job_sender.send(Some((positions, properties, frame_index))).unwrap();
    // post_process(positions, properties, output, frame_index, result_read_depth, filter, separation, debug);
    drop(data);
    finished_staging_buffer.unmap();
    
}
// (Vec<([f32; 2], (usize, f32))>,Vec<f32>,Vec<my_dtype>,usize,u64,bool,my_dtype,bool)
fn post_process(
    positions: Vec<([f32; 2], (usize, f32))>,
    properties: Vec<f32>,
    output: &mut Vec<my_dtype>,
    frame_index: usize,
    result_read_depth: u64,
    filter: bool,
    separation: my_dtype,
    debug: bool,
    linker: Option<&mut Linker>,
    ) -> () {
    if debug{
        output.extend(properties);
        return;
    }
    if filter{
        let tree = kd_tree::KdTree::build_by_ordered_float(positions);
        let relevant_points = tree.iter().map(|query| {
            let positions = query.0;
            let (idx, mass) = query.1;
            let neighbors = tree.within_radius(query, separation);
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
            if keep_point{
                Some((positions, (idx, mass)))
            } else {
                None
            }
        }).flatten()
        .collect::<Vec<_>>();
        match linker{
            Some(linker) =>{
                let part_ids = linker.advance(&relevant_points);
                for (point, part_id) in relevant_points.iter().zip(part_ids.iter()){
                    let (positions, (idx, mass)) = point;
                    output.push(frame_index as my_dtype);
                    output.push(*mass);
                    output.push(positions[0]);
                    output.push(positions[1]);
                    output.push(*part_id as my_dtype);
                    for j in 0..result_read_depth as usize - 3{
                        output.push(properties[*idx * (result_read_depth as usize - 3) + j]);
                    }
                }
            },
            None => {
                for point in relevant_points.iter(){
                    let (positions, (idx, mass)) = point;
                    output.push(frame_index as my_dtype);
                    output.push(*mass);
                    output.push(positions[0]);
                    output.push(positions[1]);
                    for j in 0..result_read_depth as usize - 3{
                        output.push(properties[*idx * (result_read_depth as usize - 3) + j]);
                    }
                }
            }
        }

    } else {
        for point in positions{
            let (positions, (idx, mass)) = point;
            output.push(frame_index as my_dtype);
            output.push(mass);
            output.push(positions[0]);
            output.push(positions[1]);
            for j in 0..result_read_depth as usize - 3{
                output.push(properties[idx * (result_read_depth as usize - 3) + j]);
            }
        }
    }
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
    desc.limits.max_compute_invocations_per_workgroup = 1024;
    let (device, queue) = adapter
    .request_device(&desc, None)
    .block_on().unwrap();

    
    let common_header = include_str!("shaders/params.wgsl");


    let shaders = HashMap::from([
        // ("proprocess_backup", "src/shaders/another_backup_preprocess.wgsl"),
        ("centers", "src/shaders/centers.wgsl"),
        ("centers_outside_parens", "src/shaders/centers_outside_parens.wgsl"),
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
        if tracking_params.characterize{
            result = result.replace("//_feat_char_", "");
        }
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
    let result_read_depth: u64 = match debug{
        false => match tracking_params.characterize{
            false => 3,
            true => 7,
        },
        true => 2,
    };
    let slice_size = pic_size * std::mem::size_of::<my_dtype>();
    let size = slice_size as wgpu::BufferAddress;
    
    
    let buffers = buffer_setup::setup_buffers(&tracking_params, &device, size, dims);

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
            ("centers", 0, &shaders["centers"]),
            // ("centers", 0, &shaders["centers_outside_parens"]),
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
                Some((0, &buffers.param_buffer)),
                Some((1, &buffers.frame_buffer)),
                // (2, &buffers.gauss_1d_buffer),
                Some((3, &buffers.centers_buffer)),
            ]),
            ("pp_cols", vec![
                Some((0, &buffers.param_buffer)),
                // (1, &buffers.gauss_1d_buffer),
                Some((2, &buffers.centers_buffer)),
                Some((3, &buffers.processed_buffer)), 
            ]),
            ("centers", vec![
                Some((0, &buffers.param_buffer)),    
                Some((2, &buffers.processed_buffer)), 
                Some((3, &buffers.centers_buffer)),
                Some((4, &buffers.masses_buffer)),
                if (tracking_params.characterize) {Some((5, &buffers.frame_buffer))} else {None},
                if (tracking_params.characterize) {Some((6, &buffers.result_buffer))} else {None},
            ]),
            ("max_row", vec![
                Some((0, &buffers.param_buffer)),
                Some((1, &buffers.processed_buffer)), 
                Some((2, &buffers.max_rows)),
            ]),
            ("walk", vec![
                Some((0, &buffers.param_buffer)),
                Some((1, &buffers.processed_buffer)), 
                Some((2, &buffers.max_rows)),
                Some((3, &buffers.centers_buffer)),
                Some((4, &buffers.masses_buffer)),
                Some((5, &buffers.result_buffer)),
            ]),
        ])
    };

    let bind_group_entries = bind_group_entries
        .iter().map(|(&name, group)| (name, group.iter().flatten().map(|(i, buffer)|
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
            true => &buffers.centers_buffer,
        };
        encoder.copy_buffer_to_buffer(output_buffer, 0,
            staging_buffer, 0, result_read_depth * size);
        
        let index = queue.submit(Some(encoder.finish()));
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(&buffers.result_buffer, 0, None);
        queue.submit(Some(encoder.finish()));
        index
    };
    
    let mut wait_gpu_time = 0.;

    let (inp_sender,
        inp_receiver) = std::sync::mpsc::channel::<channel_type>();
    let handle = {
        let filter = tracking_params.filter_close;
        let separation = tracking_params.separation as my_dtype;
        // let mut linker = match tracking_params.search_range{
        //     Some(range) => Some(Linker::new(range, tracking_params.memory.unwrap_or(0))),
        //     None => None,
        // };
        let mut linker = tracking_params.search_range
        .map(|range| Linker::new(range, tracking_params.memory.unwrap_or(0)));

        std::thread::spawn(move ||{
            let mut output: output_type = Vec::new();
            loop{
                match inp_receiver.recv().unwrap(){
                    None => break,
                    Some(inp) => {
                        let (positions, properties, frame_index) = inp;
                        post_process(positions, properties, &mut output, frame_index, result_read_depth, filter, separation, debug, linker.as_mut())
                    }
                }
            }
            output
        })
    };
    
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
            // &mut output,
            pic_size,
            result_read_depth,
            frame_index,
            debug,
            tracking_params.separation as f32,
            tracking_params.filter_close,
            &inp_sender,
        );


        free_staging_buffers.push(finished_staging_buffer);
        old_submission = new_submission;
        frame_index += 1;
    }
    let finished_staging_buffer = in_use_staging_buffers.pop_front().unwrap();
    
    get_work(finished_staging_buffer,
        &device, 
        old_submission, 
        &mut wait_gpu_time,
        // &mut output,
        pic_size,
        result_read_depth,
        frame_index,
        debug,
        tracking_params.separation as f32,
        tracking_params.filter_close,
        &inp_sender,
    );
    dbg!(wait_gpu_time);
    inp_sender.send(None);
    let output = handle.join().unwrap();

    let mut result_read_depth = result_read_depth as usize + 1;
    if tracking_params.search_range.is_some(){ result_read_depth += 1; }

    let shape = (output.len() / result_read_depth, result_read_depth);
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