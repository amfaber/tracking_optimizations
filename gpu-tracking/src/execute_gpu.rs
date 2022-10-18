#![allow(warnings)]
use futures;
use futures_intrusive::{self, channel};
type my_dtype = f32;
use ndarray::{ArrayBase, ViewRepr, Array2, ArrayView2, ArrayView3, Axis, Array1, Array, ArrayView, Ix, Ix2, IntoDimension, ArrayViewD, s, Dimension};
use pollster::FutureExt;
use std::{collections::VecDeque, time::Instant, io::{BufRead, Write, Read}, fs::{self, File}, sync::mpsc::Sender, ops::Add};
use wgpu::{util::DeviceExt, Buffer, SubmissionIndex};
use crate::{kernels, into_slice::IntoSlice, gpu_setup::{self, GpuState}, linking::{ReturnDistance, Linker}};
use std::collections::HashMap;
use kd_tree;
use rayon::prelude::*;
use ndarray_stats::{QuantileExt, MaybeNanExt};
use bencher::black_box;

type inner_output = f32;
type output_type = Vec<inner_output>;
type channel_type = Option<(Vec<([f32; 2], (usize, f32))>, Option<Vec<f32>>, usize, Option<Vec<my_dtype>>)>;

#[derive(Clone)]
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
    pub cpu_processed: bool,
    pub sig_radius: Option<my_dtype>,
    pub bg_radius: Option<my_dtype>,
    pub gap_radius: Option<my_dtype>,
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
            cpu_processed: true,
            sig_radius: None,
            bg_radius: None,
            gap_radius: None,
        }
    }
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        ::std::mem::size_of::<T>(),
    )
}

fn submit_work(
    frame: &[my_dtype],
    staging_buffer: &wgpu::Buffer,
    state: &GpuState,
    tracking_params: &TrackingParams,
    debug: bool,
    ) -> SubmissionIndex {
    state.queue.write_buffer(staging_buffer, 0, bytemuck::cast_slice(frame));
    let mut encoder =
    state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    
    encoder.copy_buffer_to_buffer(staging_buffer, 0,
        &state.buffers.frame_buffer, 0, state.pic_byte_size);
        
    state.pipelines.iter().for_each(|(name, pipeline)|{
        let (group, bind_group) = &state.bind_groups[name];
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_bind_group(*group, bind_group, &[]);
        cpass.set_pipeline(pipeline);
        cpass.dispatch_workgroups(state.workgroups[0], state.workgroups[1], 1);
    });
    let output_buffer = match debug {
        false => &state.buffers.result_buffer,
        true => &state.buffers.centers_buffer,
    };
    encoder.copy_buffer_to_buffer(output_buffer, 0,
        staging_buffer, 0, state.result_read_depth * state.pic_byte_size);
    if tracking_params.cpu_processed {
        encoder.copy_buffer_to_buffer(&state.buffers.processed_buffer, 0,
            &staging_buffer, state.result_read_depth * state.pic_byte_size, state.pic_byte_size);
    }
    let index = state.queue.submit(Some(encoder.finish()));
    let mut encoder =
        state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.clear_buffer(&state.buffers.result_buffer, 0, None);
    state.queue.submit(Some(encoder.finish()));
    index
}


fn get_work(
    finished_staging_buffer: &Buffer,
    state: &GpuState,
    tracking_params: &TrackingParams,
    old_submission: wgpu::SubmissionIndex,
    wait_gpu_time: &mut Option<f64>,
    frame_index: usize,
    debug: bool,
    job_sender: &Sender<channel_type>,
    circle_inds: &Vec<[i32; 2]>,
    dims: &[u32; 2],
    ) -> () {
    let mut buffer_slice = finished_staging_buffer.slice(..);
    let (sender, receiver) = 
            futures_intrusive::channel::shared::oneshot_channel();
    
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    let now = wait_gpu_time.as_ref().map(|_| Instant::now());
    state.device.poll(wgpu::Maintain::WaitForSubmissionIndex(old_submission));
    wait_gpu_time.as_mut().map(|t| *t += now.unwrap().elapsed().as_secs_f64());
    // (*wait_gpu_time) += now.elapsed().as_nanos() as f64 / 1_000_000_000.;
    receiver.receive().block_on().unwrap();
    let data = buffer_slice.get_mapped_range();
    let dataf32 = bytemuck::cast_slice::<u8, f32>(&data);
    let mut positions: Vec<([f32; 2], (usize, f32))> = Vec::with_capacity(1000);
    let mut properties = if tracking_params.characterize || debug {
        Some(Vec::with_capacity(1000*(state.result_read_depth-3) as usize))
    } else {
        None
    };
    let mut idx = 0usize;
    
    let pic_size = state.pic_size;
    let result_read_depth = state.result_read_depth;
    
    let (mut neighborhoods, processed_offset) = if tracking_params.cpu_processed {
        (Some(Vec::with_capacity(1000 * circle_inds.len())), Some(result_read_depth as usize))
    } else {
        (None, None)
    };
    
    
    let shape = [dataf32.len()/dims.iter().product::<u32>() as usize, dims[0] as usize, dims[1] as usize];
    let data_array = ArrayView::from_shape(shape, dataf32).unwrap();
    let masses = data_array.index_axis(Axis(0), 0);
    let centersx = data_array.index_axis(Axis(0), 1);
    let centersy = data_array.index_axis(Axis(0), 2);
    let gpu_properties = (3..result_read_depth).map(|i| data_array.index_axis(Axis(0), i as usize)).collect::<Vec<_>>();
    let processed = processed_offset.map(|processed_offset| {data_array.index_axis(Axis(0), processed_offset)});
    match debug{
        false => {
            for (i, &mass) in masses.indexed_iter(){
                if mass != 0.0{
                    unsafe{
                        positions.push(([*centersx.uget(i), *centersy.uget(i)], (idx, mass)));
                    }
                    
                    if tracking_params.cpu_processed{
                        // let idx = i.into_dimension().as_array_view();
                        for ind in circle_inds.iter(){
                            // let curidx = i + (*ind as usize) + processed_offset * pic_size;
                            let curidx = ((i.0 as i32 + ind[0]) as usize, (i.1 as i32 + ind[1]) as usize);
                            neighborhoods.as_mut().unwrap().push(*processed.unwrap().get(curidx).unwrap_or(&my_dtype::NAN));
                        }
                    }
                    if tracking_params.characterize{
                        for prop in gpu_properties.iter(){
                            unsafe{
                                properties.as_mut().unwrap().push(*prop.uget(i));
                            }
                        }
                    }
                    idx += 1;
                }
            }
        },
        true => {
            for i in 0..pic_size*result_read_depth as usize{
                properties.as_mut().unwrap().push(dataf32[i]);
            }
        },
    }
    job_sender.send(Some((positions, properties, frame_index, neighborhoods))).unwrap();
    // post_process(positions, properties, output, frame_index, result_read_depth, filter, separation, debug);
    drop(data);
    finished_staging_buffer.unmap();
    
}

pub fn column_names(params: TrackingParams) -> Vec<(String, String)>{
    let mut names = vec![("frame", "int"), ("mass", "float"), ("y", "float"), ("x", "float")];

    if params.search_range.is_some(){
        names.push(("particle", "int"));
    }

    if params.cpu_processed{
        names.push(("Rg", "float"));
        names.push(("ecc", "float"));
        names.push(("signal", "float"));
    }

    if params.sig_radius.is_some(){
        names.push(("raw_mass", "float"));
    }

    if params.bg_radius.is_some(){
        names.push(("raw_bg_median", "float"));
        names.push(("raw_mass_corrected", "float"));
    }
    
    if params.characterize{
        names.push(("Rg_gpu", "float"));
        names.push(("raw_mass_gpu", "float"));
        names.push(("signal_gpu", "float"));
        names.push(("ecc_gpu", "float"));
    }
    names.iter().map(|(name, t)| (name.to_string(), t.to_string())).collect()
}


pub struct PostProcessKernels{
    r2: Array2<my_dtype>,
    sin: Array2<my_dtype>,
    cos: Array2<my_dtype>,
    raw_sig_inds: Option<Vec<[i32; 2]>>,
    raw_bg_inds: Option<Vec<[i32; 2]>>,
}

fn post_process<A: IntoSlice>(
    positions: Vec<([f32; 2], (usize, f32))>,
    properties: Option<Vec<f32>>,
    output: &mut Vec<my_dtype>,
    frame_index: usize,
    result_read_depth: u64,
    debug: bool,
    linker: Option<&mut Linker>,
    neighborhoods: Option<Vec<f32>>,
    neighborhood_size: usize,
    kernels: Option<&PostProcessKernels>,
    middle_most: usize,
    frame: Option<A>,
    dims: &[u32; 2],
    tracking_params: &TrackingParams,
    ) -> () {
    if debug{
        output.extend(properties.unwrap());
        return;
    }

    let N = positions.len();
    let tree = kd_tree::KdTree::build_by_ordered_float(positions);
    let relevant_points = tree.iter().map(|query| {
        let positions = query.0;
        let (idx, mass) = query.1;
        let neighbors = tree.within_radius(query, tracking_params.separation as my_dtype);
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
    }).flatten().collect::<Vec<_>>();

    let cpu_properties = neighborhoods.map(|neighborhoods|{
        let neighborhoods = ndarray::Array::from_shape_vec((N, neighborhood_size), neighborhoods).unwrap();
        let kernels = kernels.unwrap();
        let Rg = (&neighborhoods * &kernels.r2).sum_axis(Axis(1));
        let sin = (&neighborhoods * &kernels.sin).sum_axis(Axis(1)).mapv(|x| x.powi(2));
        let cos = (&neighborhoods * &kernels.cos).sum_axis(Axis(1)).mapv(|x| x.powi(2));
        let ecc = (sin + cos).mapv(|x| x.sqrt());
        let middle_most = neighborhoods.map_axis(Axis(1), |x| x[middle_most]);
        let signal = neighborhoods.map_axis(Axis(1), |x|
            x.iter().max_by(|&&a, &b| {
                (a).partial_cmp(b).unwrap()
            }).cloned().unwrap());
        (Rg, ecc, middle_most, signal)
    });

    let raw_properties = frame.map(|inner| {
        let frame = inner.into_slice();
        let dims = [dims[0] as usize, dims[1] as usize];
        let frame = ArrayView::from_shape(dims, frame).unwrap();
        let kernels = kernels.unwrap();
        let raw_sig_inds = kernels.raw_sig_inds.as_ref().unwrap();
        let raw_bg_inds = kernels.raw_bg_inds.as_ref();
        let mut sig = Vec::with_capacity(N * raw_sig_inds.len());
        let mut bg = raw_bg_inds.map(|_| Vec::with_capacity(N * raw_sig_inds.len()));
        
        for (position, (idx, mass)) in relevant_points.iter() {
            let middle_index = [position[0].round() as i32, position[1].round() as i32];
            for ind in raw_sig_inds{
                let curidx = [(middle_index[0] + ind[0]) as usize, (middle_index[1] + ind[1]) as usize];
                sig.push(*frame.get(curidx).unwrap_or(&my_dtype::NAN));
            }
            bg.as_mut().map(|bg|{
                for ind in raw_bg_inds.unwrap(){
                    let curidx = [(middle_index[0] + ind[0]) as usize, (middle_index[1] + ind[1]) as usize];
                    bg.push(*frame.get(curidx).unwrap_or(&my_dtype::NAN));
                }
            });
        }
        let sig = ndarray::Array::from_shape_vec((relevant_points.len(), raw_sig_inds.len()), sig).unwrap();
        let mut bg = bg.map(|bg| ndarray::Array::from_shape_vec((relevant_points.len(), raw_bg_inds.unwrap().len()), bg).unwrap());
        let sig_sums = sig.fold_axis_skipnan(Axis(1),
        noisy_float::NoisyFloat::new(0 as my_dtype), |&acc, &next|{ (acc + next) }).mapv(|x| x.raw());
        let bg_medians = bg.map(|mut bg|{
            bg.quantile_axis_skipnan_mut(Axis(1), noisy_float::types::n64(0.5), &ndarray_stats::interpolate::Linear).unwrap()});
        let corrected = bg_medians.as_ref().map(|bg_medians|{
             &sig_sums - bg_medians * raw_sig_inds.len() as f32});
        (sig_sums, bg_medians, corrected)
    });

    let part_ids = linker.map(|linker| linker.advance(&relevant_points));

    for (idx_after_filter, point) in relevant_points.iter().enumerate(){
        let (positions, (idx_before_filter, mass)) = point;
        output.push(frame_index as my_dtype);
        output.push(*mass);
        output.push(positions[0]);
        output.push(positions[1]);

        part_ids.as_ref().map(|part_ids| output.push(part_ids[idx_after_filter] as my_dtype));

        cpu_properties.as_ref().map(|cpu_properties|{
            let (Rg, ecc, middle_most, signal) = cpu_properties;
            let idx = *idx_before_filter;
            output.push((Rg[idx] / mass).sqrt());
            output.push(ecc[idx] / (mass - middle_most[idx]));
            output.push(signal[idx]);
        });

        raw_properties.as_ref().map(|raw_properties|{
            let idx = idx_after_filter;
            let (sig_sums, bg_medians, corrected) = raw_properties;
            output.push(sig_sums[idx]);
            bg_medians.as_ref().map(|bg_medians| output.push(bg_medians[idx]));
            corrected.as_ref().map(|corrected| output.push(corrected[idx]));
        });
        
        properties.as_ref().map(|properties|{
            let idx = *idx_before_filter;
            for j in 0..result_read_depth as usize - 3{
                output.push(properties[idx * (result_read_depth as usize - 3) + j]);
            }
        });
    }
}


pub fn execute_gpu<A: IntoSlice + Send, T: Iterator<Item = A>>(
    mut frames: T,
    dims: &[u32; 2],
    tracking_params: TrackingParams,
    debug: bool,
    verbosity: u32,
    ) -> (output_type, Vec<(String, String)>){
    let state = gpu_setup::setup_state(&tracking_params, dims, debug);

    let (circle_inds, middle_most) = kernels::circle_inds((tracking_params.diameter as i32 / 2) as f32);
    let mut wait_gpu_time = if verbosity > 0 {Some(0.) } else {None};

    let (inp_sender,
        inp_receiver) = std::sync::mpsc::channel::<channel_type>();
    
    let (frame_sender,
        frame_receiver) = std::sync::mpsc::channel::<Option<A>>();
    
    let output = std::thread::scope(|scope|{
        let handle = {
            let filter = tracking_params.filter_close;
            let separation = tracking_params.separation as my_dtype;
            let neighborhood_size = circle_inds.len();
            let radius = tracking_params.diameter as i32 / 2;
            let thread_tracking_params = tracking_params.clone();
            let mut linker = tracking_params.search_range
            .map(|range| Linker::new(range, tracking_params.memory.unwrap_or(0)));
            
            scope.spawn(move ||{
                let kernels = Some(
                    PostProcessKernels{
                        r2: ndarray::Array::from_shape_vec((1, neighborhood_size), kernels::r2_in_circle(radius)).unwrap(),
                        sin: ndarray::Array::from_shape_vec((1, neighborhood_size), kernels::sin_in_circle(radius)).unwrap(),
                        cos: ndarray::Array::from_shape_vec((1, neighborhood_size), kernels::cos_in_circle(radius)).unwrap(),
                        raw_sig_inds: tracking_params.sig_radius.map(|radius| kernels::circle_inds(radius).0),
                        raw_bg_inds: tracking_params.bg_radius.zip(tracking_params.sig_radius).zip(tracking_params.gap_radius)
                        .map(|((bg_radius, sig_radius), gap_radius)| kernels::annulus_inds(bg_radius, sig_radius + gap_radius)),
                    }
                );
                let mut output: output_type = Vec::new();
                let mut thread_sleep = if verbosity > 0 {Some(0.)} else {None};
                loop{
                    let now = thread_sleep.map(|_| std::time::Instant::now());
                    match inp_receiver.recv().unwrap(){
                        None => break,
                        Some(inp) => {
                            thread_sleep.as_mut().map(|thread_sleep| *thread_sleep += now.unwrap().elapsed().as_nanos() as f64 / 1e9);
                            let (positions, properties,
                                frame_index, neighborhoods) = inp;
                            let frame = frame_receiver.recv().unwrap();
                            post_process(positions, properties, &mut output, frame_index,
                                state.result_read_depth, debug, linker.as_mut(),
                                neighborhoods, neighborhood_size, kernels.as_ref(), middle_most, frame, dims, &thread_tracking_params);
                        }
                    }
                }
                thread_sleep.map(|thread_sleep| println!("Thread sleep: {} s", thread_sleep));
                output
            })
        };
        
        let mut free_staging_buffers = state.buffers.staging_buffers.iter().collect::<Vec<&wgpu::Buffer>>();
        let mut in_use_staging_buffers = VecDeque::new();
        let frame = frames.next().unwrap();
        let mut frame_index = 0;
        let staging_buffer = free_staging_buffers.pop().unwrap();
        in_use_staging_buffers.push_back(staging_buffer);
        let mut old_submission =
            submit_work(frame.into_slice(), staging_buffer, &state, &tracking_params, debug);
        
        let send_frame = tracking_params.sig_radius.is_some();
        frame_sender.send(if send_frame{Some(frame)} else {None}).unwrap();

        for frame in frames{
            let staging_buffer = free_staging_buffers.pop().unwrap();
            in_use_staging_buffers.push_back(staging_buffer);
            let new_submission = submit_work(frame.into_slice(), staging_buffer, &state, &tracking_params, debug);
            frame_sender.send(if send_frame{Some(frame)} else {None}).unwrap();
            
            let finished_staging_buffer = in_use_staging_buffers.pop_front().unwrap();
            
            get_work(finished_staging_buffer,
                &state, 
                &tracking_params,
                old_submission, 
                &mut wait_gpu_time,
                frame_index,
                debug,
                &inp_sender,
                &circle_inds,
                &dims,
            );


            free_staging_buffers.push(finished_staging_buffer);
            old_submission = new_submission;
            frame_index += 1;
        }
        let finished_staging_buffer = in_use_staging_buffers.pop_front().unwrap();
        
        get_work(finished_staging_buffer,
            &state, 
            &tracking_params,
            old_submission, 
            &mut wait_gpu_time,
            frame_index,
            debug,
            &inp_sender,
            &circle_inds,
            &dims,
        );
        dbg!(wait_gpu_time);
        inp_sender.send(None);
        handle.join().unwrap()
    });

    // let mut n_result_columns = state.result_read_depth as usize + 1;
    // if tracking_params.cpu_processed { n_result_columns += 3};
    // tracking_params.sig_radius.map(|_| n_result_columns += 1);
    // tracking_params.bg_radius.map(|_| n_result_columns += 2);
    // tracking_params.search_range.map(|_| n_result_columns += 1);

    // let shape = (output.len() / n_result_columns, n_result_columns);
    (output, column_names(tracking_params))
}

pub fn execute_ndarray(array: &ArrayView3<my_dtype>, params: TrackingParams, debug: bool, verbosity: u32) -> (Array2<my_dtype>, Vec<(String, String)>) {
    if !array.is_standard_layout(){
        panic!("Array is not standard layout");
    }
    let axisiter = array.axis_iter(ndarray::Axis(0));
    let dims = array.shape();
    let (res, column_names) = execute_gpu(axisiter, &[dims[1] as u32, dims[2] as u32], params, debug, verbosity);
    let res_len = res.len();
    let shape = (res_len / column_names.len(), column_names.len());
    let res = Array2::from_shape_vec(shape, res)
        .expect(format!("Could not convert to ndarray. Shape is ({}, {}) but length is {}", shape.0, shape.1, &res_len).as_str());
    (res, column_names)
}