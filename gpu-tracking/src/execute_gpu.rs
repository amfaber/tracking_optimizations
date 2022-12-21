use bencher::black_box;
use bytemuck::{Pod, Zeroable};
use futures_intrusive;
type my_dtype = f32;
type my_dtype_u = u32;
use image::Frame;
use ndarray::{Array2, ArrayView3, Axis, ArrayView, Array3, ArrayView2};
use num_traits::Pow;
use pollster::FutureExt;
use tiff::decoder::Decoder;
use std::{collections::{VecDeque, HashMap}, time::Instant, sync::mpsc::Sender, path::Path, fs::File, io::Write, marker::PhantomData, f32::consts::PI};
use wgpu::{Buffer, SubmissionIndex};
use crate::{kernels, into_slice::IntoSlice,
    gpu_setup::{
    self, GpuState, ParamStyle, TrackingParams, GpuStateFlavor
}, linking::{
    Linker, FrameSubsetter, ReturnDistance
}, decoderiter::{
    IterDecoder, self
}, utils::{
    any_as_u8_slice, inspect_buffers,
}};
use kd_tree::{self, KdPoint};
use ndarray_stats::{QuantileExt, MaybeNanExt};

type inner_output = my_dtype;
type output_type = Vec<inner_output>;
type channel_type = Option<(Vec<ResultRow>, usize, Option<Vec<my_dtype>>, Option<f32>)>;



#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
struct ResultRow{
    pub x: my_dtype,
    pub y: my_dtype,
    pub mass: my_dtype,
    pub r: my_dtype,
    pub log_space_intensity: my_dtype,
    pub Rg: my_dtype,
    pub raw_mass: my_dtype,
    pub signal: my_dtype,
    pub ecc: my_dtype,
}

const N_RESULT_COLUMNS: usize = 9;
const MY_DTYPE_SIZE: usize = std::mem::size_of::<my_dtype>();


impl ResultRow{

    pub fn insert_in_output(&self, output: &mut Vec<my_dtype>, r: bool, characterize: bool){
        output.push(self.x);
        output.push(self.y);
        output.push(self.mass);
        if r{
            output.push(self.r);
        }
        if characterize{
            output.push(self.Rg);
            output.push(self.raw_mass);
            output.push(self.signal);
            output.push(self.ecc);
        }
    }
}

impl KdPoint for ResultRow
{    
    type Scalar = f32;
    type Dim = typenum::U2;
    fn at(&self, k: usize) -> Self::Scalar{
        match k{
            0 => self.x,
            1 => self.y,
            _ => panic!("Invalid index"),
        }
    }
}


fn submit_work(
    frame: &[my_dtype],
    staging_buffer: &wgpu::Buffer,
    state: &GpuState,
    tracking_params: &TrackingParams,
    debug: bool,
    positions: Option<Vec<[my_dtype; 2]>>,
    ) -> SubmissionIndex {
    // let atomic_filtered_buffer = state.common;
    // let frame_buffer = &state.common_buffers.frame_buffer;
    // let result_buffer = &state.common_buffers.result_buffer;
    // let particles_buffer = &state.common_buffers.particles_buffer;
    // let atomic_buffer = &state.common_buffers.atomic_buffer;
    // let atomic_filtered_buffer = &state.common_buffers.atomic_filtered_buffer;
    
    let common_buffers = &state.common_buffers;

    let mut encoder =
    state.device.create_command_encoder(&Default::default());
    state.queue.write_buffer(staging_buffer, 0, bytemuck::cast_slice(frame));

    
    encoder.copy_buffer_to_buffer(staging_buffer, 0,
        &common_buffers.frame_buffer, 0, state.pic_byte_size);

    match state.flavor{
        GpuStateFlavor::Trackpy{ ref order, .. } => {

            if let Some(ref positions) = positions {
                let positions = positions.iter().map(|pos| ResultRow{
                    x: pos[0],
                    y: pos[1],
                    mass: 0.,
                    log_space_intensity: 0.0,
                    r: 0.,
                    Rg: 0.,
                    raw_mass: 0.,
                    signal: 0.,
                    ecc: 0.,
                }).collect::<Vec<_>>();
                state.queue.write_buffer(staging_buffer, state.pic_byte_size, bytemuck::cast_slice(&positions));
                state.queue.write_buffer(&common_buffers.atomic_filtered_buffer, 0, bytemuck::cast_slice(&[positions.len() as u32]));
                encoder.copy_buffer_to_buffer(staging_buffer, state.pic_byte_size,
                    &common_buffers.result_buffer, 0, state.pic_byte_size);
            };
            // let buffers = &flavor.buffers;
            
            
            // state.pipelines.iter().for_each(|(name, (pipeline, wg_n))|{
            //     let (group, bind_group) = &state.bind_groups[name];
            //     let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            //     cpass.set_bind_group(*group, bind_group, &[]);
            //     cpass.set_pipeline(pipeline);
            //     cpass.dispatch_workgroups(wg_n[0], wg_n[1], wg_n[2]);
            // });
            
            for cpassname in order.iter(){
                let fullpass = &state.passes[cpassname][0];
                fullpass.execute(&mut encoder, &[]);
            }
            
        },
        GpuStateFlavor::Log{ ref buffers, ref radii, .. } => {
            
            state.passes["preprocess_rows"][0].execute(&mut encoder, &[]);
            state.passes["preprocess_cols"][0].execute(&mut encoder, &[]);
            
            // let fft = false;
            // if fft{
            //     let modder = buffers.logspace_buffers.len();

            //     let (pad_raw, push_constants) = &buffers.raw_padded.1;
            //     pad_raw.execute(&mut encoder, bytemuck::cast_slice(push_constants));
    
            //     let fft_raw = &buffers.raw_padded.2;
            //     fft_raw.execute(&mut encoder, false, false);
    
            //     encoder.clear_buffer(&buffers.logspace_buffers[2].0, 0, None);
                
            //     let n_sigma = buffers.filter_buffers.len();
            //     let mut iterator = 0..n_sigma;
                
            //     let i = iterator.next().unwrap();
            //     let convolution = &buffers.logspace_buffers[i % modder].1[i];
            //     convolution.execute(&mut encoder, &[]);
                
            //     let ifft = &buffers.logspace_buffers[i % modder].2;
            //     ifft.execute(&mut encoder, true, true);
            //     let mut edge: i32 = -1;
                
            //     for i in iterator{
            //         let convolution = &buffers.logspace_buffers[i % modder].1[i];
            //         convolution.execute(&mut encoder, &[]);
                    
            //         let ifft = &buffers.logspace_buffers[i % modder].2;
            //         ifft.execute(&mut encoder, true, true);
                    
            //         let find_max = &state.passes["logspace_max"][(i-1) % 3];
            //         let sigma = sigmas[i-1];
            //         let push_constants_tuple = (edge, sigma);
            //         let push_constants = unsafe{ any_as_u8_slice(&push_constants_tuple) };
            //         find_max.execute(&mut encoder, push_constants);
                    
            //         // let walk = &state.passes["walk"][(i-1) % 3];
            //         // walk.execute(&mut encoder, &[]);
            //         // encoder.clear_buffer(&buffers.particles_buffer, 0, None);
            //         // encoder.clear_buffer(&buffers.atomic_buffer, 0, None);
                    
            //         edge = 0;
            //     }
    
                
            //     // encoder.clear_buffer(&buffers.logspace_buffers[n_sigma % modder].0, 0, None);
                
            //     edge = 1;
                
            //     let find_max = &state.passes["logspace_max"][(n_sigma - 1) % modder];
            //     let sigma = sigmas[n_sigma - 1];
            //     let push_constants_tuple = (edge, sigma);
            //     let push_constants = unsafe{ any_as_u8_slice(&push_constants_tuple) };
            //     find_max.execute(&mut encoder, push_constants);
                
            //     encoder.copy_buffer_to_buffer(&buffers.global_max, 0, staging_buffer, 4+state.pic_byte_size, 4);
            //     encoder.clear_buffer(&buffers.global_max, 0, None);
            // } else {
                // encoder.clear_buffer(&buffers.logspace_buffers[2].0, 0, None);
                
            let n_radii = radii.len();
            let mut iterator = radii.iter().enumerate();
            
            let (i, radius) = iterator.next().unwrap();
            let modder = buffers.logspace_buffers.len();
            let convolution = &buffers.logspace_buffers[i % modder].1;
            let sigma = radius / (2 as my_dtype).sqrt();
            convolution.execute(&mut encoder, bytemuck::cast_slice(&[sigma]));
            
            let mut edge = -1;
            
            for (i, radius) in iterator{
                let convolution = &buffers.logspace_buffers[i % modder].1;
                let sigma = radius / (2 as my_dtype).sqrt();
                convolution.execute(&mut encoder, bytemuck::cast_slice(&[sigma]));
                
                let find_max = &state.passes["logspace_max"][(i-1) % 3];
                let prev_radius = radii[i-1];
                let push_constants_tuple = (edge, prev_radius);
                let push_constants = unsafe{ any_as_u8_slice(&push_constants_tuple) };
                find_max.execute(&mut encoder, push_constants);
                
                let walk = &state.passes["walk"][(i-1) % 3];
                walk.execute(&mut encoder, &[]);
                encoder.clear_buffer(&common_buffers.particles_buffer, 0, None);
                encoder.clear_buffer(&common_buffers.atomic_buffer, 0, None);
                edge = 0;
            }
            
            edge = 1;
            
            let find_max = &state.passes["logspace_max"][(n_radii - 1) % modder];
            let radius = radii[n_radii - 1];
            let push_constants_tuple = (edge, radius);
            let push_constants = unsafe{ any_as_u8_slice(&push_constants_tuple) };
            find_max.execute(&mut encoder, push_constants);
            
            let walk = &state.passes["walk"][(n_radii - 1) % modder];
            walk.execute(&mut encoder, &[]);
            encoder.clear_buffer(&common_buffers.particles_buffer, 0, None);
            encoder.clear_buffer(&common_buffers.atomic_buffer, 0, None);
            // }
            
            // let to_print = buffers.logspace_buffers.iter().map(|tup| &tup.0).collect::<Vec<_>>();
            // let to_print = vec![&common_buffers.particles_buffer, &common_buffers.atomic_buffer];
            // let to_print = vec![&common_buffers.result_buffer, &common_buffers.atomic_filtered_buffer];
            // let to_print = vec![&common_buffers.processed_buffer];
            // inspect_buffers(to_print.as_slice(),
            //     &common_buffers.staging_buffers[0],
            //     &state.queue,
            //     encoder,
            //     &state.device,
            //     // copy_size,
            //     "testing");

        }
    }

    let output_buffer = &common_buffers.result_buffer;
    
    encoder.copy_buffer_to_buffer(&common_buffers.atomic_filtered_buffer, 0,
        staging_buffer, 0, 4);
    encoder.copy_buffer_to_buffer(output_buffer, 0,
        staging_buffer, 4, state.pic_byte_size);
    
    let index = state.queue.submit(Some(encoder.finish()));
    let mut encoder =
        state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.clear_buffer(&common_buffers.result_buffer, 0, None);
    encoder.clear_buffer(&common_buffers.atomic_buffer, 0, None);
    encoder.clear_buffer(&common_buffers.atomic_filtered_buffer, 0, None);
    encoder.clear_buffer(&common_buffers.particles_buffer, 0, None);
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
    job_sender: Option<&Sender<channel_type>>,
    circle_inds: &Vec<[i32; 2]>,
    dims: &[u32; 2],
    ) -> Option<(Vec<ResultRow>, usize)> {
    // println!("getting frame: {}", frame_index);
    let buffer_slice = finished_staging_buffer.slice(..);
    let (sender, receiver) = 
            futures_intrusive::channel::shared::oneshot_channel();
    
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    let now = wait_gpu_time.as_ref().map(|_| Instant::now());
    state.device.poll(wgpu::Maintain::WaitForSubmissionIndex(old_submission));
    wait_gpu_time.as_mut().map(|t| *t += now.unwrap().elapsed().as_secs_f64());
    receiver.receive().block_on().unwrap().unwrap();
    let data = buffer_slice.get_mapped_range();
    let mut idx = 0usize;
    
    let neighborhoods = None;
    let n_parts = *bytemuck::from_bytes::<u32>(&data[..MY_DTYPE_SIZE]);
    let particle_data_endpoint = (n_parts as usize)*MY_DTYPE_SIZE*N_RESULT_COLUMNS + MY_DTYPE_SIZE;
    let results = bytemuck::cast_slice::<u8, ResultRow>(&data[MY_DTYPE_SIZE..particle_data_endpoint]).to_vec();
    let global_max = match tracking_params.style{
        ParamStyle::Trackpy{ .. } => None,
        ParamStyle::Log{ .. } => Some(*bytemuck::from_bytes::<f32>(&data[particle_data_endpoint..(particle_data_endpoint+MY_DTYPE_SIZE)]))
    };

    drop(data);
    finished_staging_buffer.unmap();

    let out = match job_sender{
        Some(job_sender) => {
            job_sender.send(Some((results, frame_index, neighborhoods, global_max))).unwrap();
            None
        },
        None => Some((results, frame_index)),
    };

    out
}

pub fn column_names(params: TrackingParams) -> Vec<(String, String)>{
    let mut names = vec![("frame", "int"), ("y", "float"), ("x", "float"), ("mass", "float")];

    if let ParamStyle::Log{ .. } = params.style{
        names.push(("r", "float"))
    }
    
    if params.characterize{
        names.push(("Rg", "float"));
        names.push(("raw", "float"));
        names.push(("signal", "float"));
        names.push(("ecc", "float"));
    }

    if params.search_range.is_some(){
        names.push(("particle", "int"));
    }
    
    if params.sig_radius.is_some() | matches!(params.style, ParamStyle::Log{ .. }){
        names.push(("raw_mass", "float"));
        names.push(("raw_bg_median", "float"));
        names.push(("raw_mass_corrected", "float"));
    }

    
    names.iter().map(|(name, t)| (name.to_string(), t.to_string())).collect()
}

pub struct PostProcessKernels<F1, F2>
where F1: Fn(my_dtype) -> (Vec<[i32; 2]>, usize), F2: Fn(my_dtype) -> Vec<[i32; 2]>{
    // r2: Array2<my_dtype>,
    // sin: Array2<my_dtype>,
    // cos: Array2<my_dtype>,
    
    raw_sig_inds: FloatMemoizer<(Vec<[i32; 2]>, usize), F1>,
    raw_bg_inds: FloatMemoizer<Vec<[i32; 2]>, F2>,
}

struct FloatMemoizer<O, F>
where F: Fn(my_dtype) -> O{
    map: HashMap<my_dtype_u, O>,
    function: Box<F>,
}

impl<O, F> FloatMemoizer<O, F>
where F: Fn(my_dtype) -> O{
    pub fn new(function: F) -> Self{
        Self{
            map: HashMap::new(),
            function: Box::new(function),
        }
    }

    pub fn call(&mut self, args: my_dtype) -> &O{
        self.map.entry(args.to_bits()).or_insert((self.function)(args))
    }
}


fn filter_close_trackpy(results: &Vec<ResultRow>, separation: my_dtype) -> Vec<ResultRow>{
    
    let tree = kd_tree::KdIndexTree::build_by_ordered_float(&results);
    let output = results.iter().enumerate()
    .map(|(idx, query)| {
        let mass = query.mass;
        let neighbors = tree.within_radius_rd(query, separation);
        // let neighbors = tree.within_radius(query, tracking_params.separation as my_dtype);
        let mut keep_point = true;
        for (&neighbor_idx, distance) in neighbors{
            let neighbor_mass = results[neighbor_idx].mass;
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
            Some(*query)
        } else {
            None
        }
    }).flatten().collect::<Vec<_>>();

    return output
}

fn prune_blobs_log(results: &Vec<ResultRow>, overlap_threshold: my_dtype) -> Vec<ResultRow> {

    fn disk_overlap(d: my_dtype, r1: my_dtype, r2: my_dtype) -> my_dtype{
        
        fn make_acos(d: my_dtype, r1: my_dtype, r2: my_dtype, r1s: my_dtype, r2s: my_dtype) -> my_dtype{
            let mut ratio: my_dtype = (d.pow(2) + r1s - r2s) / (2f32 * d * r1);
            ratio = ratio.clamp(-1., 1.);
            return ratio.acos()
        }
        let r1s = r1.pow(2);
        let r2s = r2.pow(2);
        let acos1 = make_acos(d, r1, r2, r1s, r2s);
        let acos2 = make_acos(d, r2, r1, r2s, r1s);

        let a = -d + r2 + r1;
        let b =  d - r2 + r1;
        let c =  d + r2 - r1;
        let d =  d + r2 + r1;
        let area = r1s + acos1 + r2s * acos2 - 0.5 * (a * b * c * d).abs().sqrt();

        let overlap = area / (PI * r1s.min(r2s));
        return overlap
    }

    fn blob_overlap(blob1: &ResultRow, blob2: &ResultRow, distance: my_dtype) -> my_dtype{
        if distance > blob1.r + blob2.r{
            return 0.
        } else if distance <= (blob1.r - blob2.r).abs() {
            return 1.
        } else {
            return disk_overlap(distance, blob1.r, blob2.r)
        }
    }
    let tree = kd_tree::KdIndexTree::build_by_ordered_float(&results);
    // let mut output = Vec::new();

    let mut to_keep: Vec<_> = results.iter().map(|ele| Some(ele)).collect();

    // This is pretty much a copy-paste from skimage's feature.blob's _prune_blobs.
    // It is not invariant to the order of the blobs, which in turn can make it non-deterministic
    // as we have no guarantees on the ordering of the blobs received from the gpu.
    // It can be made deterministic by first sorting the results.

    for idx in 0..results.len(){
        let query = match to_keep[idx]{
            None => continue,
            Some(query) => query,
        };

        let query_radius = query.r * 2.;
        let neighbors = tree.within_radius_rd(query, query_radius);
        for (&neighbor_idx, distance) in neighbors{
            if neighbor_idx == idx{
                continue
            }
            let neighbor = match to_keep[neighbor_idx]{
                Some(neighbor) => neighbor,
                None => continue,
            };
            if blob_overlap(query, neighbor, distance) > overlap_threshold{
                if query.r > neighbor.r{
                    to_keep[neighbor_idx] = None
                } else {
                    to_keep[idx] = None
                }
            }
        }
    }

    let output: Vec<_> = to_keep.into_iter().flatten().cloned().collect();
    output
}

fn post_process<A: IntoSlice>(
    results: Vec<ResultRow>,
    output: &mut Vec<my_dtype>,
    frame_index: usize,
    debug: bool,
    linker: Option<&mut Linker>,
    neighborhoods: Option<Vec<f32>>,
    neighborhood_size: usize,
    kernels: Option<&mut PostProcessKernels<impl Fn(my_dtype) -> (Vec<[i32; 2]>, usize), impl Fn(my_dtype) -> Vec<[i32; 2]>>>,
    middle_most: usize,
    frame: Option<A>,
    dims: &[u32; 2],
    tracking_params: &TrackingParams,
    ) -> () {
    
    let N = results.len();
    let relevant_points = match tracking_params.style{
        ParamStyle::Trackpy{filter_close, separation, ..} => {
            if filter_close{
                filter_close_trackpy(&results, separation as my_dtype)
            } else {
                results
            }
        },
        ParamStyle::Log{ overlap_threshold, ..} => {
            if overlap_threshold < 1.{
                prune_blobs_log(&results, overlap_threshold)
                
            } else {
                results
            }
        },
        // true => {
        //     filter_close_trackpy(&results, tracking_params.separation as my_dtype)
        // },
        // false => {
        //     results
        // }
    };
    

    // let cpu_properties = neighborhoods.map(|neighborhoods|{
    //     let neighborhoods = ndarray::Array::from_shape_vec((N, neighborhood_size), neighborhoods).unwrap();
    //     let kernels = kernels.unwrap();
    //     let Rg = (&neighborhoods * &kernels.r2).sum_axis(Axis(1));
    //     let sin = (&neighborhoods * &kernels.sin).sum_axis(Axis(1)).mapv(|x| x.powi(2));
    //     let cos = (&neighborhoods * &kernels.cos).sum_axis(Axis(1)).mapv(|x| x.powi(2));
    //     let ecc = (sin + cos).mapv(|x| x.sqrt());
    //     let middle_most = neighborhoods.map_axis(Axis(1), |x| x[middle_most]);
    //     let signal = neighborhoods.map_axis(Axis(1), |x|
    //         x.iter().max_by(|&&a, &b| {
    //             (a).partial_cmp(b).unwrap()
    //         }).cloned().unwrap());
    //     (Rg, ecc, middle_most, signal)
    // });

    let raw_properties = frame.map(|inner| {
        
        let frame = inner.into_slice();
        let dims = [dims[0] as usize, dims[1] as usize];
        let frame = ArrayView::from_shape(dims, frame).unwrap();
        let kernels = kernels.unwrap();

        // let raw_sig_inds = kernels.raw_sig_inds.as_ref().unwrap();
        // let raw_bg_inds = kernels.raw_bg_inds.as_ref();
        let (mut raw_sig_inds, mut raw_bg_inds) = match tracking_params.style{
            ParamStyle::Trackpy{ .. } => {
                let radius = tracking_params.sig_radius.unwrap();
                let (raw_sig_inds, _) = kernels.raw_sig_inds.call(radius);
                let raw_bg_inds = kernels.raw_bg_inds.call(radius);
                (Some(raw_sig_inds), Some(raw_bg_inds))
            },
            ParamStyle::Log { .. } => {
                (None, None)
            }
        };
        // let mut sig = Vec::with_capacity(N * raw_sig_inds.len());
        // let mut bg = raw_bg_inds.map(|_| Vec::with_capacity(N * raw_sig_inds.len()));
        // let mut bg = Vec::with_capacity(N * raw_bg_inds.len());
        let mut sig = Vec::new();
        let mut bg = Vec::new();

        let mut sig_sums = Vec::new();
        let mut bg_medians = Vec::new();
        let mut correcteds = Vec::new();


        for row in relevant_points.iter() {
            match tracking_params.style{
                ParamStyle::Trackpy{ .. } => {},
                ParamStyle::Log{ .. } => {
                    let (inner_raw_sig_inds, _) = kernels.raw_sig_inds.call(row.r);
                    let inner_raw_bg_inds = kernels.raw_bg_inds.call(row.r);
                    raw_sig_inds = Some(inner_raw_sig_inds);
                    raw_bg_inds = Some(inner_raw_bg_inds);
                },
            }
            let raw_sig_inds = raw_sig_inds.unwrap();
            let raw_bg_inds = raw_bg_inds.unwrap();
            let middle_index = [row.x.round() as i32, row.y.round() as i32];
            for ind in raw_sig_inds{
                let curidx = [(middle_index[0] + ind[0]) as usize, (middle_index[1] + ind[1]) as usize];
                match frame.get(curidx){
                    Some(val) => {
                        sig.push(*val);
                    },
                    None => {}
                };
                // sig.push(*frame.get(curidx).unwrap_or(&my_dtype::NAN));
            }
            // bg.as_mut().map(|bg|{
            for ind in raw_bg_inds{
                let curidx = [(middle_index[0] + ind[0]) as usize, (middle_index[1] + ind[1]) as usize];
                match frame.get(curidx){
                    Some(val) => {
                        bg.push(*val);
                    },
                    None => {}
                };
            }
            let sig_sum: my_dtype = sig.iter().sum();
            bg.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let bg_median = if bg.len() == 0{
                0.
            } else {
                bg[bg.len()/2]
            };
            let corrected = sig_sum - bg_median * raw_sig_inds.len() as f32;

            sig_sums.push(sig_sum);
            bg_medians.push(bg_median);
            correcteds.push(corrected);
            sig.clear();
            bg.clear();
        }
        
        (sig_sums, bg_medians, correcteds)
    });

    let part_ids = linker.map(|linker| linker.advance(&relevant_points));
    let include_r = match tracking_params.style{
        ParamStyle::Log{ .. } => true,
        ParamStyle::Trackpy{ .. } => false,
    };
    for (idx, row) in relevant_points.iter().enumerate(){
        output.push(frame_index as my_dtype);
        row.insert_in_output(output, include_r, tracking_params.characterize);
        // output.extend_from_slice(row.to_slice(tracking_params.characterize, tracking_params.style));
        part_ids.as_ref().map(|part_ids| output.push(part_ids[idx] as my_dtype));
        raw_properties.as_ref().map(|raw_properties|{
            output.push(raw_properties.0[idx]);
            output.push(raw_properties.1[idx]);
            output.push(raw_properties.2[idx]);
            // raw_properties.1.as_ref().map(|bg_medians| output.push(bg_medians[idx]));
            // raw_properties.2.as_ref().map(|corrected| output.push(corrected[idx]));
        });
    }
}


pub fn execute_gpu<A: IntoSlice + Send, T: Iterator<Item = A>>(
    mut frames: T,
    dims: &[u32; 2],
    tracking_params: TrackingParams,
    debug: bool,
    verbosity: u32,
    mut pos_iter: Option<impl Iterator<Item = (usize, Vec<[my_dtype; 2]>)>>,
    // all_frames: Option<&Array3<my_dtype>>,
    ) -> (output_type, Vec<(String, String)>){
    let default_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        default_panic(panic_info);
        std::process::exit(1);
    }));
    
    let state = gpu_setup::setup_state(&tracking_params, dims, debug, pos_iter.is_some());

    let (circle_inds, middle_most) = kernels::circle_inds((0 as i32 / 2) as f32);
    // let (circle_inds, middle_most) = kernels::circle_inds((tracking_params.diameter as i32 / 2) as f32);
    let mut wait_gpu_time = if verbosity > 0 { Some(0.) } else {None};

    let (inp_sender,
        inp_receiver) = std::sync::mpsc::channel();
        // inp_receiver) = std::sync::mpsc::channel::<channel_type>();
    
    let (frame_sender,
        frame_receiver) = std::sync::mpsc::channel::<Option<A>>();

    let send_frame = tracking_params.sig_radius.is_some() |
         matches!(tracking_params.style, ParamStyle::Log{ .. });
    
    
    let output = std::thread::scope(|scope|{
        let handle = {
            let neighborhood_size = circle_inds.len();
            let thread_tracking_params = tracking_params.clone();
            let mut linker = tracking_params.search_range
                .map(|range| Linker::new(range, tracking_params.memory.unwrap_or(0)));
            let thread_send_frame = send_frame.clone();
            scope.spawn(move ||{
                let mut kernels = if send_frame {
                    let raw_sig_inds = FloatMemoizer::new(kernels::circle_inds);
                    let raw_bg_inds = FloatMemoizer::new(|radius| {
                            kernels::annulus_inds(tracking_params.bg_radius.unwrap_or(2.0*radius), 
                            tracking_params.gap_radius.unwrap_or(0.) + radius)});
                    Some(
                        PostProcessKernels{
                            raw_sig_inds,
                            raw_bg_inds,
                        }
                    )
                } else {
                    None
                };
                let mut output: output_type = Vec::with_capacity(100000);
                let mut thread_sleep = if verbosity > 0 {Some(0.)} else {None};
                loop{
                    let now = thread_sleep.map(|_| std::time::Instant::now());
                    match inp_receiver.recv().unwrap(){
                        None => break,
                        Some(inp) => {
                            thread_sleep.as_mut().map(|thread_sleep| *thread_sleep += now.unwrap().elapsed().as_nanos() as f64 / 1e9);
                            let (results, frame_index,
                                neighborhoods, global_max) = inp;
                            let frame = frame_receiver.recv().unwrap();
                            post_process(results, &mut output, frame_index, debug, linker.as_mut(),
                                neighborhoods, neighborhood_size, kernels.as_mut(), middle_most, frame, dims, &thread_tracking_params);
                        }
                    }
                }
                thread_sleep.map(|thread_sleep| println!("Thread sleep: {} s", thread_sleep));
                output
            })
        };
        
        let mut free_staging_buffers = state.common_buffers.staging_buffers.iter().collect::<Vec<&wgpu::Buffer>>();
        
        let mut in_use_staging_buffers = VecDeque::new();
        let (frame, positions, mut frame_index) = match pos_iter{
            Some(ref mut iter) => {
                let (frame_to_take, positions) = iter.next().unwrap();
                let frame = frames.by_ref().skip(frame_to_take).next().unwrap();
                (frame, Some(positions), frame_to_take)
            },
            None => {
                (frames.next().unwrap(), None, 0)
            }
        };
        let staging_buffer = free_staging_buffers.pop().unwrap();
        in_use_staging_buffers.push_back(staging_buffer);
        let mut old_submission =
            submit_work(frame.into_slice(), staging_buffer, &state, &tracking_params, debug, positions);
        
        
        frame_sender.send(if send_frame{Some(frame)} else {None}).unwrap();

        loop{
            let (frame, positions, frame_to_take) = match pos_iter{
                Some(ref mut iter) => {
                    let (frame_to_take, positions) = match iter.next(){
                        Some((frame_to_take, positions)) => (frame_to_take, positions),
                        None => break,
                    };
                    let frame = match frames.by_ref().skip(frame_to_take - frame_index - 1).next(){
                        Some(frame) => frame,
                        None => {
                            inp_sender.send(None);
                            handle.join();
                            panic!("Frame {} not found", frame_to_take);
                        },
                    };
                    (frame, Some(positions), Some(frame_to_take))
                },
                None => {
                    match frames.next(){
                        Some(frame) => (frame, None, None),
                        None => break,
                    }
                }
            };
            let staging_buffer = free_staging_buffers.pop().unwrap();
            in_use_staging_buffers.push_back(staging_buffer);
            let new_submission = submit_work(frame.into_slice(), staging_buffer, &state, &tracking_params, debug, positions);
            frame_sender.send(if send_frame {Some(frame)} else {None}).unwrap();
            
            let finished_staging_buffer = in_use_staging_buffers.pop_front().unwrap();
            
            get_work(finished_staging_buffer,
                &state, 
                &tracking_params,
                old_submission, 
                &mut wait_gpu_time,
                frame_index,
                debug,
                Some(&inp_sender),
                &circle_inds,
                &dims,
            );


            free_staging_buffers.push(finished_staging_buffer);
            old_submission = new_submission;
            frame_index = frame_to_take.unwrap_or(frame_index + 1);
        }
        let finished_staging_buffer = in_use_staging_buffers.pop_front().unwrap();
        
        get_work(finished_staging_buffer,
            &state, 
            &tracking_params,
            old_submission, 
            &mut wait_gpu_time,
            frame_index,
            debug,
            Some(&inp_sender),
            &circle_inds,
            &dims,
        );
        wait_gpu_time.map(|wait_gpu_time| println!("Wait GPU time: {} s", wait_gpu_time));
        inp_sender.send(None).unwrap();
        handle.join().unwrap()
    });

    (output, column_names(tracking_params))
}

// pub fn sequential_execute<A: IntoSlice + Send, T: Iterator<Item = A>>(
//     mut frames: T,
//     dims: &[u32; 2],
//     tracking_params: TrackingParams,
//     debug: bool,
//     verbosity: u32,
//     ) -> (output_type, Vec<(String, String)>){
//     let state = gpu_setup::setup_state(&tracking_params, dims, debug, false);
    
//     let staging_buffer = &state.staging_buffers()[0];
//     let mut output = Vec::new();
    
//     let mut linker = tracking_params.search_range.map(|search_range| {
//         Linker::new(search_range, tracking_params.memory.unwrap_or(0))
//     });
//     for (frame_index, frame) in frames.enumerate(){
//         let frame = frame.into_slice();
//         let submission = submit_work(frame, staging_buffer, &state, &tracking_params, debug, None);
//         let (results, frame_index) = get_work(staging_buffer, &state, &tracking_params, submission, &mut None, frame_index, debug, None, &vec![], &dims).unwrap();
        
//         if debug{
//             let buffer_slice = state.buffers.processed_buffer.slice(..);
//             buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
//             state.device().poll(wgpu::Maintain::Wait);
//             let data = buffer_slice.get_mapped_range();
//             // let dataf32 = bytemuck::cast_slice::<u8, f32>(&data[..]);
//             let image = data[..].to_vec();
//             std::fs::OpenOptions::new().write(true).create(true).open("debug_image.bin").unwrap().write_all(&image).unwrap();
//             // .unwrap().write_all(bytemuck::cast_slice(&image)).unwrap();
//             drop(data);
//             state.buffers.processed_buffer.unmap();
//         }

//         post_process::<A>(results, &mut output, frame_index, debug, linker.as_mut(), None, 0, None, 0, None, dims, &tracking_params)

//     }
//     let names = column_names(tracking_params);
//     (output, names)
// }

pub fn execute_ndarray<'a>(
    array: &ArrayView3<my_dtype>,
    params: TrackingParams,
    debug: bool,
    verbosity: u32,
    pos_array: Option<&'a ArrayView2<'a, my_dtype>>,
    ) -> (Array2<my_dtype>, Vec<(String, String)>) {
    if !array.is_standard_layout(){
        panic!("Array is not standard layout");
    }
    let axisiter = array.axis_iter(ndarray::Axis(0));
    let dims = array.shape();
    let mut pos_iter = pos_array.map(|pos_array| FrameSubsetter::new(pos_array, 0, (1, 2)));
    
    let (res, column_names) = 
        // if debug {
        //     sequential_execute(axisiter, &[dims[1] as u32, dims[2] as u32], params, debug, verbosity)
        // } else {
            execute_gpu(axisiter, &[dims[1] as u32, dims[2] as u32], params, debug, verbosity, pos_iter);
        // };
    let res_len = res.len();
    let shape = (res_len / column_names.len(), column_names.len());
    let res = Array2::from_shape_vec(shape, res)
        .expect(format!("Could not convert to ndarray. Shape is ({}, {}) but length is {}", shape.0, shape.1, &res_len).as_str());
    (res, column_names)
}

pub fn execute_file(path: &str,
    channel: Option<usize>,
    params: TrackingParams,
    debug: bool,
    verbosity: u32,
    mut pos_iter: Option<impl Iterator<Item = (usize, Vec<[my_dtype; 2]>)>>,
    ) -> (Array2<my_dtype>, Vec<(String, String)>) {
    let path = Path::new(path);
    let ext = path.extension().unwrap().to_ascii_lowercase();
    let mut file = File::open(path).expect("Could not open file");
    let (res, column_names) = match ext.to_str().unwrap() {
        "tif" | "tiff" => {
            let file = File::open(path).expect("Could not open file");
            // panic!("custom panic");
            let mut decoder = Decoder::new(file).unwrap();
            let (width, height) = decoder.dimensions().unwrap();
            let dims = [height, width];
            let iterator = IterDecoder::from(decoder);
            let n_frames = if debug {1} else {usize::MAX};
            let iterator = iterator.take(n_frames);
            let (res, column_names) = 
                // if debug {
                //     sequential_execute(iterator, &dims, params, debug, verbosity)
                // } else {
                    execute_gpu(iterator, &dims, params, debug, verbosity, pos_iter);
                // };
            (res, column_names)
        },
        "ets" => {
            let parser = decoderiter::MinimalETSParser::new(&mut file).unwrap();
            let dims = [parser.dims[0] as u32, parser.dims[1] as u32];
            let iterator = parser.iterate_channel(
                file, channel.unwrap_or(0))
                .flatten().map(|vec| vec.into_iter().map(|x| x as f32).collect::<Vec<_>>());
            let n_frames = if debug {1} else {usize::MAX};
            let iterator = iterator.take(n_frames);
            let (res, column_names) = 
                // if debug{
                //     sequential_execute(iterator, &dims, params, debug, verbosity)
                // } else{
                    execute_gpu(iterator, &dims, params, debug, verbosity, pos_iter);
                // };
            (res, column_names)
        },
        _ => panic!("File extension '{}' not supported", ext.to_str().unwrap()),
    };

    let res_len = res.len();
    let shape = (res_len / column_names.len(), column_names.len());
    let res = Array2::from_shape_vec(shape, res)
        .expect(format!("Could not convert to ndarray. Shape is ({}, {}) but length is {}", shape.0, shape.1, &res_len).as_str());
    (res, column_names)
}