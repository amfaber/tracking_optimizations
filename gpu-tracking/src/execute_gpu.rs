use bencher::black_box;
use bytemuck::{Pod, Zeroable};
use futures_intrusive;
type my_dtype = f32;
use ndarray::{Array2, ArrayView3, Axis, ArrayView};
use pollster::FutureExt;
use tiff::decoder::Decoder;
use std::{collections::VecDeque, time::Instant, sync::mpsc::Sender, path::Path, fs::File, num::NonZeroU32};
use wgpu::{Buffer, SubmissionIndex};
use crate::{kernels, into_slice::IntoSlice, gpu_setup::{self, GpuState}, linking::Linker, decoderiter::{IterDecoder, self}};
use kd_tree::{self, KdPoint};
use ndarray_stats::{QuantileExt, MaybeNanExt};
use winit::window::Window;

type inner_output = my_dtype;
type output_type = Vec<inner_output>;
type channel_type = Option<(Vec<ResultRow>, usize)>;

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
    // pub cpu_processed: bool,
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
            // cpu_processed: true,
            sig_radius: None,
            bg_radius: None,
            gap_radius: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ResultRow{
    pub x: f32,
    pub y: f32,
    pub mass: f32,
    pub Rg: f32,
    pub raw_mass: f32,
    pub signal: f32,
    pub ecc: f32,
}

impl ResultRow{
    pub fn to_slice(&self, characterize: bool) -> &[f32]{
        let ptr = self as *const Self as *const f32;
        let len = if characterize {7} else {3};
        unsafe{std::slice::from_raw_parts(ptr, len)}
    }
}

unsafe impl Zeroable for ResultRow {}

unsafe impl Pod for ResultRow{}

impl KdPoint for ResultRow{
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
    ) -> SubmissionIndex {
    state.queue.write_buffer(staging_buffer, 0, bytemuck::cast_slice(frame));
    let mut encoder =
    state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    
    encoder.copy_buffer_to_buffer(staging_buffer, 0,
        &state.buffers.frame_buffer, 0, state.pic_byte_size);
        
    state.pipelines.iter().for_each(|(name, (pipeline, wg_n))|{
        let (group, bind_group) = &state.bind_groups[name];
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_bind_group(*group, bind_group, &[]);
        cpass.set_pipeline(pipeline);
        cpass.dispatch_workgroups(wg_n[0], wg_n[1], wg_n[2]);
    });
    let output_buffer = match debug {
        false => &state.buffers.result_buffer,
        true => &state.buffers.result_buffer,
    };
    encoder.copy_buffer_to_buffer(&state.buffers.atomic_filtered_buffer, 0,
        staging_buffer, 0, 4);
    encoder.copy_buffer_to_buffer(output_buffer, 0,
        staging_buffer, 4, state.pic_byte_size);
    
    let index = state.queue.submit(Some(encoder.finish()));
    let mut encoder =
        state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.clear_buffer(&state.buffers.result_buffer, 0, None);
    encoder.clear_buffer(&state.buffers.atomic_buffer, 0, None);
    encoder.clear_buffer(&state.buffers.atomic_filtered_buffer, 0, None);
    encoder.clear_buffer(&state.buffers.particles_buffer, 0, None);
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
    // circle_inds: &Vec<[i32; 2]>,
    dims: &[u32; 2],
    ) -> Option<(Vec<ResultRow>, usize)> {
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
    
    
    let n_parts = *bytemuck::from_bytes::<u32>(&data[..4]);
    let results = bytemuck::cast_slice::<u8, ResultRow>(&data[4..(n_parts as usize)*4*7 + 4]).to_vec();
    
    drop(data);
    finished_staging_buffer.unmap();
    
    if let Some(sender) = job_sender{
        sender.send(Some((results, frame_index))).unwrap();
        return None;
    }
    return Some((results, frame_index));
    
}

pub fn column_names(params: TrackingParams) -> Vec<(String, String)>{
    let mut names = vec![("frame", "int"), ("y", "float"), ("x", "float"), ("mass", "float")];
    
    if params.characterize{
        names.push(("Rg", "float"));
        names.push(("raw", "float"));
        names.push(("signal", "float"));
        names.push(("ecc", "float"));
    }

    if params.search_range.is_some(){
        names.push(("particle", "int"));
    }
    
    if params.sig_radius.is_some(){
        names.push(("raw_mass", "float"));
    }

    if params.bg_radius.is_some(){
        names.push(("raw_bg_median", "float"));
        names.push(("raw_mass_corrected", "float"));
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
    results: Vec<ResultRow>,
    output: &mut Vec<my_dtype>,
    frame_index: usize,
    // result_read_depth: u64,
    debug: bool,
    linker: Option<&mut Linker>,
    // neighborhoods: Option<Vec<f32>>,
    // neighborhood_size: usize,
    kernels: Option<&PostProcessKernels>,
    middle_most: Option<usize>,
    frame: Option<A>,
    dims: &[u32; 2],
    tracking_params: &TrackingParams,
    ) -> (Vec<ResultRow>, Option<Vec<usize>>) {

    let N = results.len();
    let tree = kd_tree::KdIndexTree::build_by_ordered_float(&results);
    let relevant_points = results.iter().enumerate()
    .map(|(idx, query)| {
        let mass = query.mass;
        let neighbors = tree.within_radius(query, tracking_params.separation as my_dtype);
        let mut keep_point = true;
        for &neighbor_idx in neighbors{
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
        let raw_sig_inds = kernels.raw_sig_inds.as_ref().unwrap();
        let raw_bg_inds = kernels.raw_bg_inds.as_ref();
        let mut sig = Vec::with_capacity(N * raw_sig_inds.len());
        let mut bg = raw_bg_inds.map(|_| Vec::with_capacity(N * raw_sig_inds.len()));
        
        for row in relevant_points.iter() {
            let middle_index = [row.x.round() as i32, row.y as i32];
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
        let bg = bg.map(|bg| ndarray::Array::from_shape_vec((relevant_points.len(), raw_bg_inds.unwrap().len()), bg).unwrap());
        let sig_sums = sig.fold_axis_skipnan(Axis(1),
        noisy_float::NoisyFloat::new(0 as my_dtype), |&acc, &next|{ acc + next }).mapv(|x| x.raw());
        let bg_medians = bg.map(|mut bg|{
            bg.quantile_axis_skipnan_mut(Axis(1), noisy_float::types::n64(0.5), &ndarray_stats::interpolate::Linear).unwrap()});
        let corrected = bg_medians.as_ref().map(|bg_medians|{
             &sig_sums - bg_medians * raw_sig_inds.len() as f32});
        (sig_sums, bg_medians, corrected)
    });

    let part_ids = linker.map(|linker| linker.advance(&relevant_points));
    
    for (idx, row) in relevant_points.iter().enumerate(){
        output.push(frame_index as my_dtype);
        output.extend_from_slice(row.to_slice(tracking_params.characterize));
        part_ids.as_ref().map(|part_ids| output.push(part_ids[idx] as my_dtype));
        raw_properties.as_ref().map(|raw_properties|{
            output.push(raw_properties.0[idx]);
            raw_properties.1.as_ref().map(|bg_medians| output.push(bg_medians[idx]));
            raw_properties.2.as_ref().map(|corrected| output.push(corrected[idx]));
        });

    }
    (relevant_points, part_ids)
}


pub fn execute_gpu<A: IntoSlice + Send, T: Iterator<Item = A>>(
    mut frames: T,
    dims: &[u32; 2],
    tracking_params: TrackingParams,
    debug: bool,
    verbosity: u32,
    ) -> (output_type, Vec<(String, String)>){
    let state = gpu_setup::setup_state(&tracking_params, dims, debug, None);

    let (circle_inds, middle_most) = kernels::circle_inds((tracking_params.diameter as i32 / 2) as f32);
    let mut wait_gpu_time = if verbosity > 0 {Some(0.) } else {None};

    let (inp_sender,
        inp_receiver) = std::sync::mpsc::channel::<channel_type>();
    
    let (frame_sender,
        frame_receiver) = std::sync::mpsc::channel::<Option<A>>();
    
    let output = std::thread::scope(|scope|{
        let handle = {
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
                let mut output: output_type = Vec::with_capacity(100000);
                let mut thread_sleep = if verbosity > 0 {Some(0.)} else {None};
                loop{
                    let now = thread_sleep.map(|_| std::time::Instant::now());
                    match inp_receiver.recv().unwrap(){
                        None => break,
                        Some(inp) => {
                            thread_sleep.as_mut().map(|thread_sleep| *thread_sleep += now.unwrap().elapsed().as_nanos() as f64 / 1e9);
                            let (results, frame_index) = inp;
                            let frame = frame_receiver.recv().unwrap();
                            post_process(results, &mut output, frame_index,
                                debug, linker.as_mut(),
                                kernels.as_ref(), Some(middle_most), frame, dims, &thread_tracking_params);
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
                Some(&inp_sender),
                // &circle_inds,
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
            Some(&inp_sender),
            // &circle_inds,
            &dims,
        );
        wait_gpu_time.map(|wait_gpu_time| println!("Wait GPU time: {} s", wait_gpu_time));
        inp_sender.send(None).unwrap();
        handle.join().unwrap()
    });

    (output, column_names(tracking_params))
}

pub fn show_gpu<A: IntoSlice + Send, T: Iterator<Item = A>>(
    mut frames: T,
    dims: &[u32; 2],
    tracking_params: TrackingParams,
    debug: bool,
    verbosity: u32,
    window: Option<&Window>,
    ) -> (){
    
    
    
    let state = gpu_setup::setup_state(&tracking_params, dims, debug, window.as_deref());
    let staging_buffer = &state.buffers.staging_buffers[0];
    let mut output = Vec::new();

    let mut linker = Linker::new(tracking_params.search_range.expect("Search range must be supplied"), tracking_params.memory.unwrap_or(0));
    
    let render_state = state.render_state.as_ref().unwrap();
    for (frame_index, frame) in frames.enumerate(){
        let submission = submit_work(frame.into_slice(), staging_buffer, &state, &tracking_params, debug);
        let (results, neighborhoods) = get_work(staging_buffer,
            &state,
            &tracking_params,
            submission,
            &mut None, 0,
            debug,
            None,
            &dims).unwrap();
        
        let (relevant_points, tracks) = post_process::<A>(results,
            &mut output,
            frame_index,
            // result_read_depth,
            debug,
            Some(&mut linker),
            None,
            None,
            None,
            dims,
            &tracking_params);
        
        let output = match render_state.surface.get_current_texture(){
            Ok(output) => output,
            Err(_) => {
                render_state.surface.configure(&state.device, &render_state.config);
                render_state.surface.get_current_texture().unwrap()
            },
        };
        // dbg!(&output);

        let view = output.texture.create_view(&Default::default());
        let mut encoder = state.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &state.buffers.processed_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(NonZeroU32::new(4 * dims[1]).unwrap()),
                    rows_per_image: None,
                },
            },
            wgpu::ImageCopyTexture {
                texture: &render_state.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: dims[1],
                height: dims[0],
                depth_or_array_layers: 1,
            },
        );


        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                ..Default::default()
            });
            // render_pass.set_pipeline(&render_state.pipeline);
            // render_pass.set_bind_group(0, &render_state.bind_group, &[]);
            // render_pass.set_vertex_buffer(0, render_state.vertex_buffer.slice(..));
            // render_pass.set_index_buffer(render_state.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            // render_pass.draw_indexed(0..6, 0, 0..1);
        }
        state.queue.submit(Some(encoder.finish()));
        output.present();
    }
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

pub fn execute_file(path: &str,
    channel: Option<usize>,
    params: TrackingParams,
    debug: bool,
    verbosity: u32,
    window: Option<&Window>,
    // ) -> (Array2<my_dtype>, Vec<(String, String)>) {
    ) -> () {
    let path = Path::new(path);
    let ext = path.extension().unwrap().to_ascii_lowercase();
    let mut file = File::open(path).expect("Could not open file");
    // let (res, column_names) = match ext.to_str().unwrap() {
    match ext.to_str().unwrap() {
        "tif" | "tiff" => {
            let file = File::open(path).expect("Could not open file");
            // panic!("custom panic");
            let mut decoder = Decoder::new(file).unwrap();
            let (width, height) = decoder.dimensions().unwrap();
            let dims = [height, width];
            let iterator = IterDecoder::from(decoder);
            let n_frames = if debug {1} else {usize::MAX};
            let iterator = iterator.take(n_frames);
            // let (res, column_names) = execute_gpu(iterator, &dims, params, debug, verbosity);
            // (res, column_names)
            show_gpu(iterator, &dims, params, debug, verbosity, window);
        },
        "ets" => {
            let parser = decoderiter::MinimalETSParser::new(&mut file).unwrap();
            let dims = [parser.dims[0] as u32, parser.dims[1] as u32];
            let iterator = parser.iterate_channel(
                file, channel.unwrap_or(0))
                .flatten().map(|vec| vec.into_iter().map(|x| x as f32).collect::<Vec<_>>());
            let n_frames = if debug {1} else {usize::MAX};
            let iterator = iterator.take(n_frames);
            let (res, column_names) = execute_gpu(iterator, &dims, params, debug, verbosity);
            // (res, column_names)
        },
        _ => panic!("File extension '{}' not supported", ext.to_str().unwrap()),
    };

    // let res_len = res.len();
    // let shape = (res_len / column_names.len(), column_names.len());
    // let res = Array2::from_shape_vec(shape, res)
    //     .expect(format!("Could not convert to ndarray. Shape is ({}, {}) but length is {}", shape.0, shape.1, &res_len).as_str());
    // (res, column_names)
}