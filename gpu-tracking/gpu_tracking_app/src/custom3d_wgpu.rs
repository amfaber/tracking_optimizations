use std::{sync::Arc, collections::{HashMap, HashSet, BTreeMap}, path::PathBuf, ops::RangeInclusive, rc::{Rc, Weak}, cell::RefCell};

use eframe::egui_wgpu;
use egui::{self, TextStyle};
use epaint;
use bytemuck;
use thiserror::Error;
use gpu_tracking::{gpu_setup::{TrackingParams, ParamStyle}, execute_gpu::path_to_iter, progressfuture::{ProgressFuture, PollResult}};
use ndarray::{Array, Array2, Axis, ArrayView1};
use crate::{colormaps, texture::ColormapRenderResources};
use kd_tree;
use std::fmt::Write;
use uuid::Uuid;
use anyhow;
use rfd;
use strum::IntoEnumIterator;
use csv;
use ndarray_csv::Array2Reader;
use image::save_buffer;

// use ndarray_stats::histogram::{HistogramExt, strategies::Auto, Bins, Edges, Grid, GridBuilder};
// use ordered_float;

type WorkerType = ProgressFuture<Result<RecalculateResult, anyhow::Error>, (usize, Option<usize>), RecalculateJob>;

fn ignore_result<R>(_res: R){}

trait ColorMap{
    fn call(&self, t: f32) -> epaint::Color32;
}

impl ColorMap for [f32; 120]{
    fn call(&self, t: f32) -> epaint::Color32{
        let t = t.clamp(0.0, 1.0);
        let t29 = t * 29.;
        let ind = t29 as usize;
        let leftover = t29 - ind as f32;
        let color_view: &[[f32; 4]] = bytemuck::cast_slice(self);
        
        let start = color_view[ind];
        let end = if leftover == 0.0{
            [0., 0., 0., 0.]
        } else {
            color_view[ind + 1]
        };
        let mut out = [0; 4];
        for ((o, s), e) in out.iter_mut().zip(start.iter()).zip(end.iter()){
            *o = ((s + leftover * (e - s)) * 255.) as u8;
        }
        epaint::Color32::from_rgba_unmultiplied(out[0], out[1], out[2], out[3])
    }
}

type FileProvider = Box<dyn gpu_tracking::decoderiter::FrameProvider<
		Frame = Vec<f32>,
		FrameIter = Box<dyn Iterator<Item = Result<Vec<f32>, gpu_tracking::error::Error>>>
	>>;

struct ProviderDimension((FileProvider, [u32; 2]));

impl ProviderDimension{
    fn to_array(&self, frame_idx: usize) -> anyhow::Result<Array2<f32>>{
        let (provider, dims) = &self.0;
        let frame = provider.get_frame(frame_idx)?;
        let frame = Array::from_shape_vec([dims[0] as usize, dims[1] as usize], frame).unwrap();
        Ok(frame)
    }
    // fn dims(&self) -> &[u32; 2]{
    //     &self.0.1
    // }
    fn len(&self) -> usize{
        self.0.0.len(None)
    }
}

#[derive(Clone, PartialEq)]
enum DataMode{
    Off,
    Immediate,
    Range(std::ops::RangeInclusive<usize>),
    Full,
}

impl DataMode{
    // fn get_range(&self) -> &RangeInclusive<usize>{
    //     match self{
    //         Self::Range(range) => range,
    //         _ => panic!("Tried to get range where there is none"),
    //     }
    // }
}

pub struct AppWrapper{
    apps: Vec<Rc<RefCell<Custom3d>>>,
    opens: Vec<bool>,
    frame_count: usize,
}

impl AppWrapper{
    pub fn new<'a>(_cc: &'a eframe::CreationContext<'a>) -> Option<Self> {
        let apps = vec![
            Rc::new(RefCell::new(Custom3d::new()?)),
        ];
        let opens = vec![true];
        
        Some(Self{apps, opens, frame_count: 0})
    }
}

impl eframe::App for AppWrapper{
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // if self.frame_count == 4{
        //     frame.request_pixels();
        // }
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    let response = ui.button("New");
                    if response.clicked(){
                        self.apps.push(Rc::new(RefCell::new(Custom3d::new().unwrap())));
                        self.opens.push(true);
                    }
                    if ui.button("capture").clicked(){
                        frame.request_pixels();
                    }
                    let mut adds = Vec::new();
                    let mut removes = Vec::new();
                    for (i, (app, open)) in self.apps.iter_mut().zip(self.opens.iter_mut()).enumerate(){
                        // app.borrow_mut()
                        let mut app = app.borrow_mut();
                        egui::containers::Window::new("")
                            .id(egui::Id::new(app.uuid.as_u128()))
                            .title_bar(true)
                            .open(open)
                            .show(ui.ctx(), |ui|{
                                app.entry_point(ui, frame);
                                ui.horizontal(|ui| {
                                    let response = ui.button("Clone");
                                    if response.clicked() && matches!(app.result_status, ResultStatus::Valid | ResultStatus::Static){
                                        adds.push((i, app.uuid));
                                    }
                                    if response.secondary_clicked(){
                                        app.other_apps.clear();
                                        app.update_circles();
                                    }
                                    if ui.button("Copy python command").clicked(){
                                        ui.output_mut(|output| output.copied_text = app.input_state.to_py_command());
                                    }
                                    if ui.button("Output data to csv").clicked() | app.save_pending{
                                        app.output_csv();
                                    }
                                    ui.add(egui::widgets::TextEdit::singleline(&mut app.input_state.output_path)
                                        .code_editor()
                                        .hint_text("Save file path (must be .csv)"));
                                });
                            });
                        // drop(app);
                    }
                    for (i, open) in self.opens.iter().enumerate(){
                        if !*open{
                            removes.push(i);
                        }
                    }
                    for (i, uuid) in adds{
                        let app = self.apps.iter().find(|app| app.borrow().uuid == uuid).unwrap();
                        let new_app = Rc::new(RefCell::new(app.borrow().clone()));
                        let mut new_app_mut = new_app.borrow_mut();
                        // let new_app_mut = Rc::get_mut(&mut new_app).unwrap();
                        new_app_mut.setup_gpu_after_clone(ui, frame);
                        new_app_mut.other_apps.push(Rc::downgrade(app));
                        drop(new_app_mut);
                        self.apps.insert(i + 1, new_app);
                        self.opens.insert(i + 1, true);
                    }
                    for i in removes{
                        self.apps.remove(i);
                        self.opens.remove(i);
                    }
                })
        });
        self.frame_count += 1;
    }

    fn post_rendering(&mut self, _window_size_px: [u32; 2], frame: &eframe::Frame) {
        // dbg!("hi");
        if let Some(data) = frame.frame_pixels(){
            // std::fs::write("from_gpu.bin", &data).expect("failed to write");
            image::save_buffer("from_gpu.png", &data, 1200, 1000, image::ColorType::Rgba8).expect("save failed");
        }
        // frame.storage
    }
}

struct RecalculateJob{
    path: PathBuf,
    tracking_params: TrackingParams,
    channel: Option<usize>,
    // result_sender: Sender<anyhow::Result<RecalculateResult>>,
}

pub struct Custom3d {
    worker: WorkerType,
    progress: Option<usize>,
    plot_radius_fallback: f32,
    static_dataset: bool,
    recently_updated: bool,
	frame_provider: Option<ProviderDimension>,
    vid_len: Option<usize>,
    frame_idx: usize,
    other_apps: Vec<Weak<RefCell<Custom3d>>>,
    // last_render: Option<std::time::Instant>,
    // gpu_tracking_state: gpu_tracking::gpu_setup::GpuState,
    results: Option<Array2<f32>>,
    result_names: Option<Vec<(String, String)>>,
    circles_to_plot: Vec<(Array2<f32>, Option<Weak<RefCell<Custom3d>>>)>,
    tracking_params: gpu_tracking::gpu_setup::TrackingParams,
    mode: DataMode,
    path: Option<PathBuf>,
    channel: Option<usize>,
    output_path: PathBuf,
    save_pending: bool,
    particle_hash: Option<BTreeMap<usize, Vec<(usize, [f32; 2], usize)>>>,
    // row_range: Option<Vec<(usize, usize)>>,
    alive_particles: Option<BTreeMap<usize, Vec<(usize, Option<usize>)>>>,
    cumulative_particles: Option<BTreeMap<usize, HashSet<usize>>>,
    
    circle_kdtree: Option<kd_tree::KdTree<([f32; 2], (usize, usize))>>,

    r_col: Option<usize>,
    x_col: Option<usize>,
    y_col: Option<usize>,
    frame_col: Option<usize>,
    particle_col: Option<usize>,

    circle_color: egui::Color32,
    image_cmap: colormaps::KnownMaps,
    line_cmap: colormaps::KnownMaps,
    line_cmap_bounds: Option<RangeInclusive<f32>>,
    track_colors: TrackColors,
    all_tracks: bool,
    // line_cmap_end: f32,

    zoom_box_start: Option<egui::Pos2>,
    cur_asp: egui::Vec2,
    texture_zoom_level: egui::Rect,
    databounds: Option<egui::Rect>,

    uuid: Uuid,
    result_status: ResultStatus,
    // job_sender: Sender<RecalculateJob>,
    // result_receiver: Option<Receiver<anyhow::Result<RecalculateResult>>>,

    input_state: InputState,
    needs_update: NeedsUpdate,

    playback: Playback,
}

#[derive(Clone)]
enum Playback{
    FPS((f32, std::time::Instant)),
    Off,
}

impl Playback{
    fn should_frame_advance(&mut self, ui: &mut egui::Ui) -> bool{
        match self{
            Self::FPS((fps, last_advance)) => {
                ui.ctx().request_repaint();
                
                if last_advance.elapsed().as_micros() as f32 / 1_000_000. > 1./(*fps){
                    *last_advance = std::time::Instant::now();
                    true
                } else {
                    false
                }
            },
            Self::Off => false,
        }
    }
}

impl Custom3d{
    fn clone(&self) -> Self{
        
        let mode = self.mode.clone();
        let cur_asp = self.cur_asp;
        
        let uuid = Uuid::new_v4();
        
        let params = self.tracking_params.clone();
        
        let line_cmap = self.line_cmap.clone();


        let texture_zoom_level = self.texture_zoom_level;

        // let (job_sender, job_receiver) = std::sync::mpsc::channel();
        
        // std::thread::spawn(move ||{
        //     loop{
        //         match job_receiver.recv(){
        //             Ok(RecalculateJob { path, channel, tracking_params, result_sender }) => {
        //                 let result = RecalculateResult::from(
        //                     gpu_tracking::execute_gpu::execute_file(&path, channel, tracking_params.clone(), 0, None, None, None).into()
        //                 );
        //                 match result_sender.send(result){
        //                     Ok(()) => (),
        //                     Err(_) => break,
        //                 };
        //             },
        //             Err(_) => break
        //         }
        //     }
        // });
        let worker = ProgressFuture::new(|job, progress, interrupt|{
            match job{
                RecalculateJob { path, tracking_params, channel } => {
                    RecalculateResult::from(
                        gpu_tracking::execute_gpu::execute_file(&path, channel, tracking_params.clone(), 0, None, Some(interrupt), Some(progress))
                    )
                }
            }
        });
        
        let input_state = self.input_state.clone();
        
        let frame_idx = self.frame_idx;
        let image_cmap = self.image_cmap.clone();
        
        let channel = self.channel.clone();
        let frame_provider = self.path.as_ref().and_then(|path| {
            match path_to_iter(path, channel){
                Ok(res) => Some(ProviderDimension(res)),
                Err(_) => None
            }
        });
        let vid_len = self.vid_len.clone();

        let needs_update = self.needs_update.clone();

        let playback = self.playback.clone();

        let circle_color = self.circle_color.clone();
        let output_path = self.output_path.clone();
        let other_apps = self.other_apps.clone();
        
        
        let out = Self{
            worker,
            progress: self.progress.clone(),
            plot_radius_fallback: self.plot_radius_fallback.clone(),
            static_dataset: self.static_dataset.clone(),
            recently_updated: self.recently_updated.clone(),
            other_apps,
            frame_provider,
            vid_len,
            frame_idx,
            // last_render: None,
            results: self.results.clone(),
            result_names: self.result_names.clone(),
            circles_to_plot: self.circles_to_plot.clone(),
            tracking_params: params,
            mode,
            path: self.path.clone(),
            channel,
            output_path,
            save_pending: self.save_pending.clone(),
            particle_hash: self.particle_hash.clone(),
            alive_particles: self.alive_particles.clone(),
            // row_range: self.row_range.clone(),
            cumulative_particles: self.cumulative_particles.clone(),
            
            circle_kdtree: self.circle_kdtree.clone(),

            r_col: self.r_col.clone(),
            x_col: self.x_col.clone(),
            y_col: self.y_col.clone(),
            frame_col: self.frame_col.clone(),
            particle_col: self.particle_col.clone(),

            circle_color,
            image_cmap,
            line_cmap,
            line_cmap_bounds: self.line_cmap_bounds.clone(),
            track_colors: self.track_colors.clone(),
            all_tracks: self.all_tracks.clone(),

            zoom_box_start: None,
            cur_asp,
            texture_zoom_level,
            databounds: self.databounds.clone(),

            uuid,
            result_status: self.result_status.clone(),
            // result_receiver: None,
            // job_sender,

            input_state,
            needs_update,

            playback
        };
        out
        
    }
}

#[derive(Clone, PartialEq)]
enum Style{
    Trackpy,
    Log,
}

#[derive(Clone)]
struct InputState{
    path: String,
    channel: String,
    output_path: String,
    frame_idx: String,
    datamode: DataMode,
    range_start: String,
    range_end: String,
    fps: String,

    style: Style,
    all_options: bool,
    color_options: bool,

    plot_radius_fallback: String,
    
    // Trackpy
    diameter: String,
    separation: String,
    filter_close: bool,

    // Log
    min_radius: String,
    max_radius: String,
    n_radii: String,
    log_spacing: bool,
    overlap_threshold: String,
    
    minmass: String,
    max_iterations: String,
    characterize: bool,
    search_range: String,
    memory: String,
    doughnut_correction: bool,
    snr: String,
    minmass_snr: String,
    illumination_sigma: String,
    adaptive_background: String,
    shift_threshold: String,
    noise_size: String,
    smoothing_size: String,
    illumination_correction_per_frame: bool,
}

impl Default for InputState{
    fn default() -> Self{
        Self{
            path: String::new(),
            channel: String::new(),
            output_path: String::new(),
            frame_idx: "0".to_string(),
            datamode: DataMode::Immediate,
            range_start: "0".to_string(),
            range_end: "10".to_string(),
            fps: "30".to_string(),

            style: Style::Trackpy,
            all_options: false,
            color_options: false,

            plot_radius_fallback: "4.5".to_string(),

            diameter: "9".to_string(),
            separation: "10".to_string(),
            filter_close: true,
            
            min_radius: "2.3".to_string(),
            max_radius: "3.5".to_string(),
            n_radii: "10".to_string(),
            log_spacing: false,
            overlap_threshold: "0.0".to_string(),
            
            minmass: "0.0".to_string(),
            max_iterations: "10".to_string(),
            characterize: true,
            search_range: "10".to_string(),
            memory: String::new(),
            doughnut_correction: true,
            snr: "1.5".to_string(),
            minmass_snr: "0.3".to_string(),
            illumination_sigma: String::new(),
            adaptive_background: String::new(),
            shift_threshold: "0.6".to_string(),
            noise_size: "1.0".to_string(),
            smoothing_size: String::new(),
            illumination_correction_per_frame: false,
        }
    }
}

impl InputState{
    fn to_trackingparams(&self) -> TrackingParams{
        let diameter = self.diameter.parse::<u32>().ok();
        let separation = self.separation.parse::<u32>().ok();
        let filter_close = self.filter_close;
        
        let min_radius = self.min_radius.parse::<f32>().ok();
        let max_radius = self.max_radius.parse::<f32>().ok();
        let n_radii = self.n_radii.parse::<usize>().ok();
        let log_spacing = self.log_spacing;
        let overlap_threshold = self.overlap_threshold.parse::<f32>().ok();
        
        let minmass = self.minmass.parse::<f32>().ok();
        let max_iterations = self.max_iterations.parse::<u32>().ok();
        let characterize = self.characterize;
        let search_range = self.search_range.parse::<f32>().ok();
        let memory = self.memory.parse::<usize>().ok();
        let doughnut_correction = self.doughnut_correction;
        let snr = self.snr.parse::<f32>().ok();
        let minmass_snr = self.minmass_snr.parse::<f32>().ok();
        let illumination_sigma = self.illumination_sigma.parse::<f32>().ok();
        let adaptive_background = self.adaptive_background.parse::<usize>().ok();
        let shift_threshold = self.shift_threshold.parse::<f32>().ok();
        let noise_size = self.noise_size.parse::<f32>().ok();
        let smoothing_size = self.smoothing_size.parse::<u32>().ok();
        let illumination_correction_per_frame = self.illumination_correction_per_frame;

        let (style, include_r_in_output, _smoothing_size_default) = match self.style{
            Style::Trackpy => {
                let diameter = diameter.unwrap_or(9);
                (ParamStyle::Trackpy{
                    diameter,
                    separation: separation.unwrap_or(diameter),
                    filter_close,
                    maxsize: 0.0,
                    invert: false,
                    percentile: 0.0,
                    topn: 0,
                    preprocess: true,
                    threshold: 0.0,
                }, false, diameter)
            },
            Style::Log => {
                let max_radius = max_radius.unwrap_or(3.5);
                let ss_default = ((max_radius + 0.5) as u32) * 2 + 1;
                (ParamStyle::Log{
                    min_radius: min_radius.unwrap_or(2.2),
                    max_radius,
                    n_radii: n_radii.unwrap_or(10),
                    log_spacing,
                    overlap_threshold: overlap_threshold.unwrap_or(0.0),
                }, true, ss_default)
            },
        };
        TrackingParams{
            style,
            minmass: minmass.unwrap_or(0.0),
            max_iterations: max_iterations.unwrap_or(10),
            characterize,
            search_range,
            memory,
            doughnut_correction,
            bg_radius: None,
            gap_radius: None,
            snr,
            minmass_snr,
            truncate_preprocessed: true,
            illumination_sigma,
            adaptive_background,
            include_r_in_output,
            shift_threshold: shift_threshold.unwrap_or(0.6),
            linker_reset_points: None,
            keys: None,
            noise_size: noise_size.unwrap_or(1.0),
            // smoothing_size: Some(smoothing_size.unwrap_or(smoothing_size_default)),
            smoothing_size,
            illumination_correction_per_frame,
        }
    }

    fn to_py_command(&self) -> String{
        let mut output = "gpu_tracking.".to_string();
        match self.style{
            Style::Trackpy => {
                output.push_str("batch(\n\t");
                self.diameter.parse::<u32>().ok().map(|val| write!(output, "{},\n\t", val));
                self.separation.parse::<u32>().ok().map(|val| write!(output, "separation = {},\n\t", val));
                if !self.filter_close { write!(output, "filter_close = False,\n\t").unwrap() };
            },
            Style::Log => {
                output.push_str("LoG(\n\t");
                self.min_radius.parse::<f32>().ok().map(|val| write!(output, "{},\n\t", val));
                self.max_radius.parse::<f32>().ok().map(|val| write!(output, "{},\n\t", val));
                self.n_radii.parse::<usize>().ok().map(|val| write!(output, "n_radii = {},\n\t", val));
                if self.log_spacing { write!(output, "log_spacing = True,\n\t").unwrap() };
                self.overlap_threshold.parse::<f32>().ok().map(|val| write!(output, "overlap_threshold = {},\n\t", val));
            },
        };
        self.channel.parse::<usize>().ok().map(|val| write!(output, "channel = {},\n\t", val));
        
        self.minmass.parse::<f32>().ok().map(|val| write!(output, "minmass = {},\n\t", val));
        self.max_iterations.parse::<u32>().ok().map(|val| if val != 10 {write!(output, "max_iterations = {},\n\t", val).unwrap()});
        self.search_range.parse::<f32>().ok().map(|val| write!(output, "search_range = {},\n\t", val));
        self.memory.parse::<usize>().ok().map(|val| write!(output, "memory = {},\n\t", val));
        self.snr.parse::<f32>().ok().map(|val| write!(output, "snr = {},\n\t", val));
        self.minmass_snr.parse::<f32>().ok().map(|val| write!(output, "minmass_snr = {},\n\t", val));
        self.illumination_sigma.parse::<f32>().ok().map(|val| write!(output, "illumination_sigma = {},\n\t", val));
        self.adaptive_background.parse::<usize>().ok().map(|val| write!(output, "adaptive_background = {},\n\t", val));
        self.shift_threshold.parse::<f32>().ok().map(|val| write!(output, "shift_threshold = {},\n\t", val));
        self.noise_size.parse::<f32>().ok().map(|val| write!(output, "noise_size = {},\n\t", val));
        self.smoothing_size.parse::<u32>().ok().map(|val| write!(output, "smoothing_size = {},\n\t", val));
        if self.characterize { write!(output, "characterize = True,\n\t").unwrap() };
        if self.doughnut_correction { write!(output, "doughnut_correction = True,\n\t").unwrap() };
        if self.illumination_correction_per_frame { write!(output, "illumination_correction_per_frame = True,\n\t").unwrap() };
        output.pop();
        output.push(')');
        output
    }
}

enum FrameChange{
    Next,
    Previous,
    Input,
    Resubmit,
}

impl FrameChange{
    fn from_scroll(scroll: egui::Vec2) -> Option<Self>{
        match scroll.y.partial_cmp(&0.0){
            Some(std::cmp::Ordering::Equal) | None => {
                None
            },
            Some(std::cmp::Ordering::Greater) => {
                Some(Self::Next)
            },
            Some(std::cmp::Ordering::Less) => {
                Some(Self::Previous)
            }
        }
    }
}

fn normalize_rect(rect: egui::Rect) -> egui::Rect {
    let min = egui::Pos2{
        x: std::cmp::min_by(rect.min.x, rect.max.x, |a: &f32, b: &f32| a.partial_cmp(b).unwrap()),
        y: std::cmp::min_by(rect.min.y, rect.max.y, |a: &f32, b: &f32| a.partial_cmp(b).unwrap()),
    };
    let max = egui::Pos2{
        x: std::cmp::max_by(rect.min.x, rect.max.x, |a: &f32, b: &f32| a.partial_cmp(b).unwrap()),
        y: std::cmp::max_by(rect.min.y, rect.max.y, |a: &f32, b: &f32| a.partial_cmp(b).unwrap()),
        };
    egui::Rect{min, max}
}

impl Custom3d {
    pub fn setup_gpu_after_clone(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame){
        if self.frame_provider.is_none(){
            return
        }
        let wgpu_render_state = frame.wgpu_render_state().unwrap();
        ignore_result(self.create_gpu_resource(wgpu_render_state));
        ignore_result(self.update_frame(ui, FrameChange::Resubmit));
        self.resize(ui, self.texture_zoom_level, self.databounds.unwrap());
    }

    fn create_gpu_resource(&mut self, wgpu_render_state: &egui_wgpu::RenderState) -> anyhow::Result<()>{
        let provider = self.frame_provider.as_ref().unwrap();
        let frame = provider.to_array(self.frame_idx)?;
        let frame_view = frame.view();
        let resources = ColormapRenderResources::new(wgpu_render_state, &frame_view, self.image_cmap.get_map());
        wgpu_render_state
            .renderer
            .write()
            .paint_callback_resources
            .entry::<HashMap<Uuid, ColormapRenderResources>>()
            .or_insert(HashMap::new())
            .insert(self.uuid, resources);
        Ok(())
    }

    fn load_csv(&mut self) -> anyhow::Result<()>{
        let mut reader = csv::Reader::from_path(&self.input_state.path)?;
        let header = reader.headers()?;
        let header: Vec<_> = header.iter().map(|field_name|{
            (field_name.to_string(), "float".to_string())
        }).collect();
        let results = reader.deserialize_array2_dynamic::<f32>()?;
        self.r_col = header.iter().position(|element|{
            element.0 == "r"
        });
        self.x_col = header.iter().position(|element|{
            element.0 == "x"
        });
        self.y_col = header.iter().position(|element|{
            element.0 == "y"
        });
        self.frame_col = header.iter().position(|element|{
            element.0 == "frame"
        });
        self.particle_col = header.iter().position(|element|{
            element.0 == "particle"
        });
        self.result_names = Some(header);
        self.results = Some(results);
        self.path = Some(self.input_state.path.clone().into());
        self.result_status = ResultStatus::Static;
        Ok(())
    }
    
    pub fn setup_new_path(&mut self, wgpu_render_state: &egui_wgpu::RenderState) -> anyhow::Result<()>{
        self.channel = self.input_state.channel.parse::<usize>().ok();
        let (provider, dims) = match path_to_iter(&self.input_state.path, self.channel){
            Ok(res) => {
                self.path = Some(self.input_state.path.clone().into());
                self.static_dataset = false;
                res
            },
            Err(_) => {
                // self.path = None;
                self.frame_provider = None;
                match self.load_csv(){
                    Ok(()) => {
                        self.static_dataset = true;
                        return Ok(())
                    },
                    Err(e) => {
                        self.static_dataset = false;
                        return Err(e)
                    }
                };
                // self.static_dataset = true;
                // return Ok(())
            }
        };
        let provider_dimension = ProviderDimension((provider, dims));
        self.vid_len = Some(provider_dimension.len());
        self.line_cmap_bounds = match self.mode{
            DataMode::Range(ref range) => {
                Some(*range.start() as f32..=(*range.end() as f32))
            },
            _ => {
                Some(0.0..=((self.vid_len.unwrap() - 1) as f32))
            }
        };
        self.frame_provider = Some(provider_dimension);
        
        self.frame_idx = 0;

        self.create_gpu_resource(wgpu_render_state)?;
        
        let mut cur_asp = egui::Vec2{
            x: dims[1] as f32,
            y: dims[0] as f32,
        };
        cur_asp = cur_asp / cur_asp.max_elem();
        
        let databounds = egui::Rect::from_x_y_ranges(
            0.0..=(dims[0] - 1) as f32,
            0.0..=(dims[1] - 1) as f32,
        );
        
        self.databounds = Some(databounds);
        self.cur_asp = cur_asp;
        self.result_status = ResultStatus::Processing;
        self.particle_hash = None;
        // self.row_range = None;
        self.input_state.frame_idx = self.frame_idx.to_string();
        ignore_result(self.recalculate());
        Ok(())
    }
    
    pub fn new() -> Option<Self> {
        // Get the WGPU render state from the eframe creation context. This can also be retrieved
        // from `eframe::Frame` when you don't have a `CreationContext` available.
        let mode = DataMode::Immediate;
        let cur_asp = egui::Vec2{
            x: 1.0,
            y: 1.0,
        };
        
        let uuid = Uuid::new_v4();
        
        
        let line_cmap = colormaps::KnownMaps::inferno;

        let texture_zoom_level = zero_one_rect();

        // let (job_sender, job_receiver) = std::sync::mpsc::channel();
        
        // std::thread::spawn(move ||{
        //     loop{
        //         match job_receiver.recv(){
        //             Ok(RecalculateJob { path, channel, tracking_params, result_sender }) => {
        //                 let result = RecalculateResult::from(
        //                     gpu_tracking::execute_gpu::execute_file(&path, channel, tracking_params.clone(), 0, None, None, None).into()
        //                 );
        //                 result_sender.send(result).expect("Main thread lost");
        //             },
        //             Err(_) => break
        //         }
        //     }
        // });
        
        let worker = ProgressFuture::new(|job, progress, interrupt|{
            match job{
                RecalculateJob { path, tracking_params, channel } => {
                    RecalculateResult::from(
                        gpu_tracking::execute_gpu::execute_file(&path, channel, tracking_params.clone(), 0, None, Some(interrupt), Some(progress))
                    )
                }
            }
        });
        
        let input_state = InputState::default();
        let params = input_state.to_trackingparams();
        
        let frame_idx = 0;
        let image_cmap = colormaps::KnownMaps::viridis;
        
        let needs_update = NeedsUpdate::default();

        let line_cmap_bounds = None;

        let playback = Playback::Off;

        let circle_color = egui::Color32::from_rgb(255, 255, 255);

        let track_colors = TrackColors::Local;
        
        let out = Self{
            worker,
            progress: None,
            plot_radius_fallback: 4.5,
            static_dataset: false,
            recently_updated: false,
            other_apps: Vec::new(),
            channel: None,
            frame_provider: None,
            vid_len: None,
            frame_idx,
            // last_render: None,
            results: None,
            result_names: None,
            circles_to_plot: Vec::new(),
            tracking_params: params,
            mode,
            path: None,
            output_path: PathBuf::from(""),
            save_pending: false,
            
            particle_hash: None,
            // row_range: None,
            alive_particles: None,
            circle_kdtree: None,
            cumulative_particles: None,

            r_col: None,
            x_col: None,
            y_col: None,
            frame_col: None,
            particle_col: None,

            circle_color,
            image_cmap,
            line_cmap,
            line_cmap_bounds,
            track_colors,
            all_tracks: false,

            zoom_box_start: None,
            cur_asp,
            texture_zoom_level,
            databounds: None,

            uuid,
            result_status: ResultStatus::Processing,
            // result_receiver: None,
            // job_sender,

            input_state,
            needs_update,

            playback
        };
        Some(out)
    }
}
#[derive(Error, Debug)]
#[error("Another job was submitted before the previous one was retrieved")]
struct StillWaitingError;


#[derive(Default, Clone)]
struct NeedsUpdate{
    datamode: bool,
    params: bool,
}
impl NeedsUpdate{
    fn any(&self) -> bool{
        self.datamode | self.params
    }
}

#[derive(Debug, Clone)]
enum ResultStatus{
    Valid,
    Processing,
    TooOld,
    Static,
}

#[derive(Debug, Clone, PartialEq)]
enum TrackColors{
    Local,
    Global,
}

impl TrackColors{
    fn to_str(&self) -> &'static str{
        match self{
            Self::Local => "Local",
            Self::Global => "Global",
        }
    }
}

struct RecalculateResult{
    results: Array2<f32>,
    result_names: Vec<(&'static str, &'static str)>,
    r_col: Option<usize>,
    x_col: Option<usize>,
    y_col: Option<usize>,
    frame_col: Option<usize>,
    particle_col: Option<usize>,
}

impl RecalculateResult{
    fn from<E: std::error::Error + Send + Sync + 'static>(input: Result<(Array2<f32>, Vec<(&'static str, &'static str)>), E>) -> anyhow::Result<Self>{
        let (results, result_names) = input?;
        let r_col = result_names.iter().position(|element|{
            element.0 == "r"
        });
        let x_col = result_names.iter().position(|element|{
            element.0 == "x"
        });
        let y_col = result_names.iter().position(|element|{
            element.0 == "y"
        });
        let frame_col = result_names.iter().position(|element|{
            element.0 == "frame"
        });
        let particle_col = result_names.iter().position(|element|{
            element.0 == "particle"
        });
        Ok(Self{
            results,
            result_names,
            r_col,
            x_col,
            y_col,
            frame_col,
            particle_col,
        })
    }
}


impl Custom3d {

    fn results_to_circles_to_plot(&self, frame: usize) -> Option<Array2<f32>>{
        match self.result_status{
            ResultStatus::Processing | ResultStatus::TooOld => {
                return None
            }
            ResultStatus::Static | ResultStatus::Valid => ()
        }
        let results = self.results.as_ref()?;

        if let Some(alive) = &self.alive_particles{
            let mut out = Array2::zeros((0, results.shape()[1]));
            for part in alive.get(&frame)?{
                if let Some(row) = part.1{
                    out.push_row(results.index_axis(Axis(0), row)).unwrap();
                }
            }
            Some(out)
        } else {
            Some(results.clone())
        }
    }

    fn update_circles(&mut self){
        self.circles_to_plot.clear();
        if let Some(circles) = self.results_to_circles_to_plot(self.frame_idx){
            self.circles_to_plot.push((circles, None))
        }

        for owner in self.other_apps.iter(){
            if let Some(alive_owner) = owner.upgrade(){
                if let Some(circles) = alive_owner.borrow().results_to_circles_to_plot(self.frame_idx){
                    self.circles_to_plot.push((circles, Some(Weak::clone(owner))));
                }
            }
        }
        
        
        let kdtree = kd_tree::KdTree::build_by_ordered_float(self.circles_to_plot.iter().enumerate().flat_map(|(idx, (array, _owner))| {
            let y_col = self.y_col.clone().unwrap();
            let x_col = self.x_col.clone().unwrap();
            array.axis_iter(Axis(0)).enumerate().map(move |(i, row)| {
                ([row[y_col], row[x_col]], (idx, i))
            })
        }).collect());
        
        if !kdtree.is_empty(){
            self.circle_kdtree = Some(kdtree);
        } else {
            self.circle_kdtree = None;
        }
    }
    
    fn update_from_recalculate(&mut self, result: anyhow::Result<RecalculateResult>) -> anyhow::Result<()>{
        let result = result?;
        let result_names: Vec<_> = result.result_names.into_iter().map(|(s1, s2)| (s1.to_string(), s2.to_string())).collect();
        // self.result_receiver = None;
        self.results = Some(result.results);
        self.result_names = Some(result_names);
        self.r_col = result.r_col;
        self.x_col = result.x_col;
        self.y_col = result.y_col;
        self.frame_col = result.frame_col;
        self.particle_col = result.particle_col;
        self.result_status = ResultStatus::Valid;
        self.recently_updated = true;
        self.update_lines();
        self.update_circles();
        Ok(())
    }

    fn update_lines(&mut self){
        match (&self.mode, self.particle_col){
            (DataMode::Immediate, _) | (_, None) => {
                self.particle_hash = None;
                self.alive_particles = None;
                self.cumulative_particles = None;
            },
            (_, Some(particle_col)) => {
                let mut particle_hash = BTreeMap::new();
                // let mut row_range: Vec<(usize, usize)> = Vec::new();
                let mut alive_particles = BTreeMap::new();
                let mut cumulative = BTreeMap::new();
                for (i, row) in self.results.as_ref().unwrap().axis_iter(Axis(0)).enumerate(){
                    let part_id = row[particle_col] as usize;
                    let frame = row[self.frame_col.unwrap()] as usize;
                    let to_insert = (frame, [row[self.x_col.unwrap()], row[self.y_col.unwrap()]], i);
                    particle_hash.entry(part_id).or_insert(Vec::new()).push(to_insert);
                }

                for (part_id, vec) in particle_hash.iter_mut(){
                    let mut prev_frame = None;
                    for (frame, _pos, row) in vec{
                        if let Some(prev) = prev_frame{
                            for inbetween in prev..*frame{
                                alive_particles.entry(inbetween).or_insert(Vec::new()).push((*part_id, None))
                            }
                        }
                        alive_particles.entry(*frame).or_insert(Vec::new()).push((*part_id, Some(*row)));
                        prev_frame = Some(*frame);
                    }
                }
                let mut prev_entry: Option<&HashSet<_>> = None;
                for (frame, vec) in &alive_particles{
                    if let Some(prev) = prev_entry{
                        let mut next_set = prev.clone();
                        next_set.extend(vec.iter().map(|(part_id, _row)| *part_id));
                        cumulative.insert(*frame, next_set);
                    } else {
                        cumulative.insert(*frame, vec.iter().map(|(part_id, _row)| *part_id).collect());
                    }
                    prev_entry = Some(&cumulative[frame]);
                }

                self.particle_hash = Some(particle_hash);
                self.alive_particles = Some(alive_particles);
                self.cumulative_particles = Some(cumulative);
                
                
                // let mut to_debug: Vec<_> = self.particle_hash.as_ref().unwrap().iter().collect();
                // to_debug.sort_by(|a, b| {
                //     let outer = a.1[0].0.cmp(&b.1[0].0);
                //     match outer{
                //         std::cmp::Ordering::Equal => {
                //             a.1[0].1[0].partial_cmp(&b.1[0].1[0]).unwrap()
                //         },
                //         _ => outer
                //     }
                // });
                
            }
        }
    }

    fn poll_result(&mut self) -> anyhow::Result<()>{
        self.recently_updated = false;
        match self.worker.poll()?{
            PollResult::Done(res) => {
                match self.result_status{
                    ResultStatus::Processing => {
                        self.update_from_recalculate(res)?;
                    },
                    ResultStatus::TooOld => {
                        // self.worker.interrupt();
                        self.recalculate()?;
                    },
                    ResultStatus::Valid | ResultStatus::Static => unreachable!(),
                }
            },
            PollResult::Pending(prog) => {
                self.progress = Some(prog.0)
            },
            PollResult::NoJobRunning => (),
        };
        // match self.result_receiver.as_ref(){
        //     Some(recv) => {
        //         match recv.try_recv(){
        //             Ok(res) => {
        //                 match self.result_status{
        //                     ResultStatus::Processing => {
        //                         self.update_from_recalculate(res)?;
        //                     },
        //                     ResultStatus::TooOld => {
        //                         self.result_receiver = None;
        //                         self.recalculate()?;
        //                     },
        //                     ResultStatus::Valid | ResultStatus::Static => unreachable!(),
        //                 }
                        
        //             },
        //             Err(std::sync::mpsc::TryRecvError::Empty) => {
                        
        //             },
        //             Err(std::sync::mpsc::TryRecvError::Disconnected) => {panic!("Thread lost")},
        //         }
        //     },
        //     None => {
        //         if !matches!(self.result_status, ResultStatus::Valid){
        //             assert!(matches!(self.mode, DataMode::Off))
        //         }
        //         // assert!(matches!(self.result_status, ResultStatus::Valid))
        //     }
        // }
        Ok(())
    }

    fn update_datamode(&mut self, ui: &mut egui::Ui){
        let old = self.mode.clone();
        let try_block = || -> anyhow::Result<()>{
            let start = &self.input_state.range_start;
            let end = &self.input_state.range_end;
            match &self.input_state.datamode{
                DataMode::Immediate | DataMode::Full | DataMode::Off if self.mode != self.input_state.datamode => {
                    self.line_cmap_bounds = Some(0.0..=((self.vid_len.unwrap() - 1) as f32));
                    self.mode = self.input_state.datamode.clone()
                },
                DataMode::Range(_) => {
                    let (start, mut end) = (start.parse::<usize>()?, end.parse::<usize>()?);
                    let len = self.vid_len.unwrap();
                    if end >= len{
                        end = len - 1;
                        self.input_state.range_end = end.to_string();
                    }
                    let range = start..=end;
                    if range.is_empty(){
                        return Err(anyhow::Error::msg("empty range"))
                    }
                    let new = DataMode::Range(range);
                    if self.mode == new{
                        return Ok(())
                    }
                    self.mode = DataMode::Range(start..=end);
                    self.line_cmap_bounds = Some((start as f32)..=(end as f32));
                    ignore_result(self.update_frame(ui, FrameChange::Resubmit));
                },
                _ => {
                    return Ok(())
                }
            };
            ignore_result(self.recalculate());
            Ok(())
        }();
        if try_block.is_err(){
            self.mode = old;
        }
    }

    fn output_csv(&mut self){
        let mut succeeded = false;
        match self.result_status{
            ResultStatus::Processing | ResultStatus::TooOld => {
                self.save_pending = true;
                return
            },
            ResultStatus::Valid | ResultStatus::Static => {
                self.save_pending = false;
            }
        }
        match self.results{
            Some(ref results) => {
                let pathbuf = PathBuf::from(&self.input_state.output_path);
                let pathbuf = if let Some("csv") = pathbuf.extension().map(|osstr| osstr.to_str().unwrap_or("")){
                    pathbuf
                } else {
                    let dialog = rfd::FileDialog::new()
                        .add_filter("csv", &["csv"])
                        .set_directory(std::env::current_dir().unwrap())
                        .save_file();
                    if let Some(path) = dialog{
                        path
                    } else {
                        PathBuf::from("")
                    }
                };

                self.input_state.output_path = pathbuf.clone().into_os_string().into_string().unwrap();
                let writer = csv::Writer::from_path(pathbuf);
            
                if let Ok(mut writer) = writer{
                    let header = self.result_names.as_ref().unwrap().iter().map(|(name, _ty)|{
                        name
                    });
                    writer.write_record(header).unwrap();
                    for row in results.axis_iter(Axis(0)){
                        writer.write_record(row.iter().map(|num| num.to_string())).unwrap();
                    }
                    succeeded = true;
                }
            },
            None => self.input_state.output_path = "No results to save".to_string()
        }
        if !succeeded{
            self.input_state.output_path = "Save failed".to_string();
        }
    }

    fn update_state(&mut self, ui: &mut egui::Ui){
        if self.needs_update.datamode{
            self.update_datamode(ui);
        }

        if self.needs_update.params{
            self.tracking_params = self.input_state.to_trackingparams();
            ignore_result(self.recalculate());
        }

        self.needs_update = NeedsUpdate::default();
    }

    fn entry_point(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame){
        ui.horizontal(|ui|{
            let browse_clicked = ui.button("Browse").clicked();
            if browse_clicked{
                self.path = rfd::FileDialog::new()
                    .set_directory(std::env::current_dir().unwrap())
                    .add_filter("Supported video formats", &["tif", "tiff", "vsi", "ets"])
                    .add_filter("CSV dataset", &["csv"])
                    .pick_file();
                if let Some(ref path) = self.path{
                    self.input_state.path = path.clone().into_os_string().into_string().unwrap(); 
                }
            }
            let textedit = egui::widgets::TextEdit::singleline(&mut self.input_state.path)
                .code_editor()
                // .desired_width(f32::INFINITY)
                .hint_text("Video file path");
            let path_changed = ui.add(textedit).changed();
            
            let channel_changed = if !self.static_dataset{
                let textedit = egui::widgets::TextEdit::singleline(&mut self.input_state.channel)
                    .hint_text("vsi/ets channel");
                ui.add(textedit).changed()
            } else {
                false
            };
            
            if path_changed | browse_clicked | channel_changed{
                let wgpu_render_state = frame.wgpu_render_state().unwrap();
                self.update_state(ui);
                match self.setup_new_path(wgpu_render_state){
                    Ok(_) => {},
                    Err(_) => self.path = None,
                };
            }
        });
        if !self.static_dataset{
            self.get_input(ui, frame);
        } else {
            self.color_options(ui);
            ui.horizontal(|ui|{
                ui.label("Plot diameter");
                let changed = ui.add(egui::widgets::TextEdit::singleline(
                    &mut self.input_state.plot_radius_fallback
                ).desired_width(25.)).changed();
                if changed{
                    match self.input_state.plot_radius_fallback.parse::<f32>(){
                        Ok(radius) => self.plot_radius_fallback = radius,
                        Err(_) => ()
                    }
                }
            });
        }
    }

    fn color_options(&mut self, ui: &mut egui::Ui){
        ui.horizontal(|ui|{
            ui.label("Circles:");
            ui.color_edit_button_srgba(&mut self.circle_color);

            ui.label("Image:");
            egui::ComboBox::from_id_source(self.uuid.as_u128() + 1).selected_text(self.image_cmap.get_name())
                .show_ui(ui, |ui|{ 
                    if colormap_dropdown(ui, &mut self.image_cmap){
                        self.set_image_cmap(ui)
                    };
                }
            );

            ui.label("Tracks:");
            egui::ComboBox::from_id_source(self.uuid.as_u128() + 2).selected_text(self.line_cmap.get_name())
                .show_ui(ui, |ui|{ 
                    if colormap_dropdown(ui, &mut self.line_cmap){
                        self.set_image_cmap(ui)
                    };
                }
            );
            
            egui::ComboBox::from_id_source(self.uuid.as_u128() + 3).selected_text(self.track_colors.to_str())
                .show_ui(ui, |ui|{ 
                    ui.selectable_value(&mut self.track_colors, TrackColors::Local, "Local");
                    ui.selectable_value(&mut self.track_colors, TrackColors::Global, "Global");
                }
            );
            
        });
    }
    
    fn get_input(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame){
        // ui.vertical(|ui|{
        ui.horizontal(|ui|{
            match self.playback{
                Playback::FPS(_) => {
                    if ui.button("Pause").clicked(){
                        self.playback = Playback::Off
                    };
                },
                Playback::Off => {
                    if ui.button("Play").clicked(){
                        let fps = match self.input_state.fps.parse::<f32>(){
                            Ok(fps) => fps,
                            Err(_) => 30.,
                        };
                        self.playback = Playback::FPS((fps, std::time::Instant::now()))
                    };
                },
            }
            let fps_changed = ui.add(egui::widgets::TextEdit::singleline(
                &mut self.input_state.fps
            ).desired_width(30.)).changed();
            
            if fps_changed & matches!(self.playback, Playback::FPS(_)){
                let fps = match self.input_state.fps.parse::<f32>(){
                    Ok(fps) => fps,
                    Err(_) => 30.,
                };
                self.playback = Playback::FPS((fps, std::time::Instant::now()))
            };
            ui.label("fps")
        });
        ui.horizontal(|ui|{
            let mut changed = false;
            changed |= ui.selectable_value(&mut self.input_state.datamode, DataMode::Off, "Off").clicked();
            changed |= ui.selectable_value(&mut self.input_state.datamode, DataMode::Immediate, "One").clicked();
            changed |= ui.selectable_value(&mut self.input_state.datamode, DataMode::Full, "All").clicked();
            changed |= ui.selectable_value(&mut self.input_state.datamode, DataMode::Range(0..=1), "Range").clicked();
            
            ui.label("Showing frame:");
            let frame_changed = ui.add(egui::widgets::TextEdit::singleline(
                &mut self.input_state.frame_idx
            ).desired_width(30.)).changed();
            
            if frame_changed{
                ignore_result(self.update_frame(ui, FrameChange::Input));
            }
            
            if self.input_state.datamode == DataMode::Range(0..=1){
                ui.label("in range");
                changed |= ui.add(egui::widgets::TextEdit::singleline(
                    &mut self.input_state.range_start
                ).desired_width(30.)).changed();
                ui.label("to");
                changed |= ui.add(egui::widgets::TextEdit::singleline(
                    &mut self.input_state.range_end
                ).desired_width(30.)).changed();
            }
            
            if changed{
                self.needs_update.datamode = true;
            }
            
        });
        
        
        self.tracking_input(ui);
        
        let mut submit_click = false;
        ui.horizontal(|ui| {
            submit_click = ui.add_enabled(self.frame_provider.is_some() & self.needs_update.any(), egui::widgets::Button::new("Submit")).clicked();
            let response = ui.button("🏠🔍");
            if response.clicked() && self.frame_provider.is_some(){
                self.reset_zoom(ui);
            }
            if ui.add(egui::SelectableLabel::new(self.input_state.color_options, "Color Options")).clicked(){
                self.input_state.color_options = !self.input_state.color_options;
            }
            if ui.add(egui::SelectableLabel::new(self.all_tracks, "Tracks for all particles")).clicked(){
                self.all_tracks = !self.all_tracks;
            }
            
        });
        if self.input_state.color_options{
            self.color_options(ui);
        }
        
        if submit_click | ui.ctx().input(|inp| inp.key_down(egui::Key::Enter)){
            self.update_state(ui);
        }
        
        
        match self.frame_provider{
            Some(_) => self.show(ui, frame),
            None => {},
        }
        // });
    }

    fn tracking_input(&mut self, ui: &mut egui::Ui){
        let mut changed = false;
        ui.horizontal(|ui|{
            changed |= ui.selectable_value(&mut self.input_state.style, Style::Trackpy, "Trackpy").clicked();
            changed |= ui.selectable_value(&mut self.input_state.style, Style::Log, "LoG").clicked();
            ui.add_space(20.0);
            if ui.add(egui::SelectableLabel::new(self.input_state.all_options, "All Options")).clicked(){
                self.input_state.all_options = !self.input_state.all_options;
            }
        });
        ui.horizontal(|ui|
            match self.input_state.style{
                Style::Trackpy => {
                    ui.label("Diameter");
                    changed |= ui.add(egui::widgets::TextEdit::singleline(
                        &mut self.input_state.diameter
                    ).desired_width(25.)).changed();
                    ui.add_space(10.0);
                    
                },
                Style::Log => {
                    ui.label("Minimum Radius");
                    changed |= ui.add(egui::widgets::TextEdit::singleline(
                        &mut self.input_state.min_radius
                    ).desired_width(25.)).changed();
                    ui.add_space(10.0);
                    
                    ui.label("Maximum Radius");
                    changed |= ui.add(egui::widgets::TextEdit::singleline(
                        &mut self.input_state.max_radius
                    ).desired_width(25.)).changed();
                    ui.add_space(10.0);
                },
            }
        );

        ui.horizontal(|ui|{
            ui.label("SNR");
            changed |= ui.add(egui::widgets::TextEdit::singleline(
                &mut self.input_state.snr
            ).desired_width(25.)).changed();
            ui.add_space(10.0);
            
            ui.label("Area SNR");
            changed |= ui.add(egui::widgets::TextEdit::singleline(
                &mut self.input_state.minmass_snr
            ).desired_width(25.)).changed();
            ui.add_space(10.0);
            
            ui.label("Tracking Search Range");
            changed |= ui.add(egui::widgets::TextEdit::singleline(
                &mut self.input_state.search_range
            ).desired_width(25.)).changed();
            ui.add_space(10.0);
            
            ui.label("Tracking memory");
            changed |= ui.add(egui::widgets::TextEdit::singleline(
                &mut self.input_state.memory
            ).desired_width(25.)).changed();
            ui.add_space(10.0);
        });

        if self.input_state.all_options{
            ui.horizontal(|ui|
                match self.input_state.style{
                    Style::Trackpy => {
                        ui.label("Separation");
                        changed |= ui.add(egui::widgets::TextEdit::singleline(
                            &mut self.input_state.separation
                        ).desired_width(25.)).changed();
                        ui.add_space(10.0);
                
                        if ui.add(egui::SelectableLabel::new(self.input_state.filter_close, "Filter Close")).clicked(){
                            self.input_state.filter_close = !self.input_state.filter_close;
                            changed = true
                        }
                        ui.add_space(10.0);
                
                    },
                    Style::Log => {
                        ui.label("Number of radii");
                        changed |= ui.add(egui::widgets::TextEdit::singleline(
                            &mut self.input_state.n_radii
                        ).desired_width(25.)).changed();
                        ui.add_space(10.0);
                
                        if ui.add(egui::SelectableLabel::new(self.input_state.log_spacing, "Logarithmic spacing of radii")).clicked(){
                            self.input_state.log_spacing = !self.input_state.log_spacing;
                            changed = true
                        }
                        ui.add_space(10.0);
                
                        ui.label("Maximum blob overlap");
                        changed |= ui.add(egui::widgets::TextEdit::singleline(
                            &mut self.input_state.overlap_threshold
                        ).desired_width(25.)).changed();
                        ui.add_space(10.0);
                    },
                }
            );

            ui.horizontal(|ui|{
                ui.label("Illumination Sigma");
                changed |= ui.add(egui::widgets::TextEdit::singleline(
                    &mut self.input_state.illumination_sigma
                ).desired_width(25.)).changed();
                ui.add_space(10.0);
        
                ui.label("Adaptive Background");
                changed |= ui.add(egui::widgets::TextEdit::singleline(
                    &mut self.input_state.adaptive_background
                ).desired_width(25.)).changed();
                ui.add_space(10.0);
        
                ui.label("Smoothing (Boxcar) Size");
                changed |= ui.add(egui::widgets::TextEdit::singleline(
                    &mut self.input_state.smoothing_size
                ).desired_width(25.)).changed();
                ui.add_space(10.0);
        
                if ui.add(egui::SelectableLabel::new(self.input_state.illumination_correction_per_frame, "Illumination Correct Per Frame")).clicked(){
                    self.input_state.illumination_correction_per_frame = !self.input_state.illumination_correction_per_frame;
                    changed = true
                }
                ui.add_space(10.0);
            });
    
            ui.horizontal(|ui|{
                ui.label("Minmass");
                changed |= ui.add(egui::widgets::TextEdit::singleline(
                    &mut self.input_state.minmass
                ).desired_width(25.)).changed();
                ui.add_space(10.0);
        
                if ui.add(egui::SelectableLabel::new(self.input_state.characterize, "Characterize")).clicked(){
                    self.input_state.characterize = !self.input_state.characterize;
                    changed = true
                }
                ui.add_space(10.0);
        
                if ui.add(egui::SelectableLabel::new(self.input_state.doughnut_correction, "Doughnut Correction")).clicked(){
                    self.input_state.doughnut_correction = !self.input_state.doughnut_correction;
                    changed = true
                }
                ui.add_space(10.0);
        
                ui.label("Noise Size");
                changed |= ui.add(egui::widgets::TextEdit::singleline(
                    &mut self.input_state.noise_size
                ).desired_width(25.)).changed();
                ui.add_space(10.0);
        
                ui.label("Shift Threshold");
                changed |= ui.add(egui::widgets::TextEdit::singleline(
                    &mut self.input_state.shift_threshold
                ).desired_width(25.)).changed();
                ui.add_space(10.0);
        
                ui.label("Max Iterations");
                changed |= ui.add(egui::widgets::TextEdit::singleline(
                    &mut self.input_state.max_iterations
                ).desired_width(25.)).changed();
                ui.add_space(10.0);
            });
        }

        if changed{
            self.needs_update.params = true;
        }
    }

    fn reset_zoom(&mut self, ui: &mut egui::Ui){
        let dims = egui::Vec2{
            x: self.frame_provider.as_ref().unwrap().0.1[1] as f32,
            y: self.frame_provider.as_ref().unwrap().0.1[0] as f32,
        };
        self.cur_asp = dims / dims.max_elem();
        let databounds = egui::Rect{
            min: egui::Pos2::ZERO,
            max: egui::Pos2{
                x: dims.y - 1.0,
                y: dims.x - 1.0,
            }
        };
        self.resize(ui, zero_one_rect(), databounds);
    }

    fn show(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame){
        ui.horizontal(|ui|{
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                let change = FrameChange::from_scroll(ui.ctx().input(|inp| inp.scroll_delta));
                self.custom_painting(ui, change);
            });
        });
    }

    fn recalculate(&mut self) -> anyhow::Result<()>{
        match &self.mode{
            DataMode::Off => {
                self.result_status = ResultStatus::Processing;
                return Ok(())
            }
            DataMode::Immediate => {
                self.tracking_params.keys = Some(vec![self.frame_idx]);
            },
            DataMode::Range(range) => {
                self.tracking_params.keys = Some(range.clone().collect());
            }
            DataMode::Full => {
                self.tracking_params.keys = None;
            }
        }
        if self.static_dataset{
            match self.mode{
                DataMode::Off => {},
                _ => self.result_status = ResultStatus::Static,
            }
            return Ok(())
        }
        // if self.result_receiver.is_some(){
        //     self.result_status = ResultStatus::TooOld;
        //     return Err(StillWaitingError.into())
        // }
        if self.worker.n_jobs() > 0{
            self.result_status = ResultStatus::TooOld;
            self.worker.interrupt();
            return Err(StillWaitingError.into())
        }
        
        let tracking_params = self.tracking_params.clone();
        let path = self.path.as_ref().cloned().unwrap();
        let channel = self.channel.clone();
        // let (result_sender, result_receiver) =  std::sync::mpsc::channel();
        // self.result_receiver = Some(result_receiver);

        self.worker.submit_new(
            RecalculateJob{
                path,
                channel,
                tracking_params
            }
        ).expect("thread crash");
        // self.job_sender.send(RecalculateJob{
        //     path,
        //     channel,
        //     result_sender,
        //     tracking_params
        // }).expect("Thread lost");

        self.result_status = ResultStatus::Processing;
        self.particle_hash = None;
        self.alive_particles = None;
        self.cumulative_particles = None;
        self.particle_col = None;
        self.progress = None;
        Ok(())
    }

    fn resize(&mut self, ui: &mut egui::Ui, size: egui::Rect, databounds: egui::Rect){
        self.texture_zoom_level = size.clone();
        self.databounds = Some(databounds.clone());
        let uuid = self.uuid;
        let cb = egui_wgpu::CallbackFn::new()
            .prepare(move |_device, queue, _encoder, paint_callback_resources|{
                let resources: &mut HashMap<Uuid, ColormapRenderResources> = paint_callback_resources.get_mut().unwrap();
                let resources = resources.get_mut(&uuid).unwrap();
                resources.resize(queue, &size);
                Vec::new()
            });
        ui.painter().add(egui::PaintCallback{
            rect: egui::Rect::EVERYTHING,
            callback: Arc::new(cb),
        });
    }

    fn set_image_cmap(&mut self, ui: &mut egui::Ui){
        let uuid = self.uuid;
        let cmap = self.image_cmap.get_map();
        let cb = egui_wgpu::CallbackFn::new()
            .prepare(move |_device, queue, _encoder, paint_callback_resources|{
                let resources: &mut HashMap<Uuid, ColormapRenderResources> = paint_callback_resources.get_mut().unwrap();
                let resources = resources.get_mut(&uuid).unwrap();
                resources.set_cmap(queue, &cmap);
                Vec::new()
            });
        ui.painter().add(egui::PaintCallback{
            rect: egui::Rect::EVERYTHING,
            callback: Arc::new(cb),
        });
    }
    
    fn update_frame(&mut self, ui: &mut egui::Ui, direction: FrameChange) -> anyhow::Result<()>{
        if self.frame_provider.is_none(){
            return Ok(())
        }
        let new_index = match direction{
            FrameChange::Next => self.frame_idx + 1,
            FrameChange::Previous => {
                if self.frame_idx != 0 { self.frame_idx - 1 } else { 0 }
            },
            FrameChange::Input => {
                self.input_state.frame_idx.parse::<usize>()?
            },
            FrameChange::Resubmit => self.frame_idx
        };

        let new_index = match self.mode{
            DataMode::Range(ref range) => {
                new_index.clamp(*range.start(), *range.end())
            },
            _ => new_index.min(self.vid_len.unwrap()-1)
        };
        
        
        if new_index == self.frame_idx && !matches!(direction, FrameChange::Resubmit){
            return Ok(())
        }
        self.frame_idx = new_index;
        self.input_state.frame_idx = self.frame_idx.to_string();
        let array = self.frame_provider.as_ref().unwrap().to_array(self.frame_idx)?;
        match self.mode{
            DataMode::Immediate => {
                ignore_result(self.recalculate());
                if !self.other_apps.is_empty(){
                    self.update_circles();
                }
            },
            _ => {
                self.update_circles();
            },
        }
        
        let uuid = self.uuid;

        let cb = egui_wgpu::CallbackFn::new()
            .prepare(move |_device, queue, _encoder, paint_callback_resources|{
                let resources: &mut HashMap<Uuid, ColormapRenderResources> = paint_callback_resources.get_mut().unwrap();
                let resources = resources.get_mut(&uuid).unwrap();
                let array_view = array.view();
                resources.update_texture(queue, &array_view);
                Vec::new()
            });
        ui.painter().add(egui::PaintCallback{
            rect: egui::Rect::EVERYTHING,
            callback: Arc::new(cb),
        });
        Ok(())
    }


    unsafe fn get_owner(&self, owner: &Option<Weak<RefCell<Self>>>) -> Option<&Self>{
        match owner{
            Some(inner) => {
                let owner = unsafe { inner.upgrade().map(|inner| &*(&*inner.borrow() as *const Custom3d)) };
                owner
            },
            None => Some(self),
        }
    }
    
    fn result_dependent_plotting(&mut self, ui: &mut egui::Ui, rect: egui::Rect, hover_pos: Option<egui::Pos2>){
        for (circles_to_plot, owner) in self.circles_to_plot.iter(){
            let owner = unsafe{ self.get_owner(owner) };
            let owner = owner.map(|inner| (inner, inner.result_status.clone()));
            if let Some((owner, ResultStatus::Valid | ResultStatus::Static)) = owner{
                let circle_plotting = circles_to_plot.axis_iter(Axis(0)).map(|rrow|{
                    epaint::Shape::circle_stroke(
                        data_to_screen_coords_vec2([rrow[owner.x_col.unwrap()], rrow[owner.y_col.unwrap()]].into(), &rect, &self.databounds.as_ref().unwrap()),
                        {
                            let r = owner.point_radius(rrow);
                            data_radius_to_screen(r, &rect, &self.databounds.as_ref().unwrap())
                        },
                        (1., owner.circle_color)
                    )
                });
                ui.painter_at(rect).extend(circle_plotting);

                
                if let (Some(particle_hash), Some(alive_particles), Some(cumulative_particles)) =
                    (&owner.particle_hash, &owner.alive_particles, &owner.cumulative_particles){
                    let cmap = owner.line_cmap.get_map();
                    let databounds = self.databounds.clone().unwrap();
                    let frame_idx = self.frame_idx;
                    let iter: Box<dyn Iterator<Item = &usize>> = if self.all_tracks{
                        Box::new(cumulative_particles[&frame_idx].iter().map(|part_id| part_id))
                    } else {
                        Box::new(alive_particles[&frame_idx].iter().map(|(part_id, _row)| part_id))
                    };                    
                    let line_plotting = iter.map(|part_id|{
                        let particle_vec = &particle_hash[part_id];
                        let bounds = match self.track_colors{
                            TrackColors::Local => {
                                let track_len = particle_vec.len();
                                let local_max = if track_len > 2{
                                    particle_vec[track_len-2].0 as f32
                                } else {
                                    (particle_vec[track_len-1].0 + 1) as f32
                                };
                                let local_bounds = (particle_vec[0].0 as f32)..=local_max;
                                local_bounds
                            },
                            TrackColors::Global => {
                                self.line_cmap_bounds.clone().unwrap()
                            },
                        };
                        particle_vec.windows(2).flat_map(move |window|{
                            let start = window[0];
                            let end = window[1];
                            if end.0 <= frame_idx && end.0 - start.0 == 1{
                                let t = inverse_lerp(start.0 as f32, bounds.start().clone(), bounds.end().clone());
                                Some(epaint::Shape::line_segment([
                                    data_to_screen_coords_vec2(start.1.into(), &rect, &databounds),
                                    data_to_screen_coords_vec2(end.1.into(), &rect, &databounds),
                                ], (1., cmap.call(t))))
                            } else {
                                None
                            }
                        })

                    }).flatten();
                    ui.painter_at(rect).extend(line_plotting);
                }

            }
            
        }

        if let (Some(hover), Some(tree)) = (hover_pos, &self.circle_kdtree){
            let hover_data = [
                lerp(inverse_lerp(hover.y, rect.min.y, rect.max.y), self.databounds.as_ref().unwrap().min.x, self.databounds.as_ref().unwrap().max.x),
                lerp(inverse_lerp(hover.x, rect.min.x, rect.max.x), self.databounds.as_ref().unwrap().min.y, self.databounds.as_ref().unwrap().max.y),
            ];
            
            let nearest = tree.nearest(&hover_data);
            if let Some(nearest) = nearest{
                let mut cutoff = 0.02 * (self.databounds.as_ref().unwrap().max.x - self.databounds.as_ref().unwrap().min.x);
                let row = self.circles_to_plot[nearest.item.1.0].0.index_axis(Axis(0), nearest.item.1.1);
                let owner = unsafe{ self.get_owner(&self.circles_to_plot[nearest.item.1.0].1) };
                if let Some(owner) = owner{
                    let point_radius = owner.point_radius(row);
                    cutoff = std::cmp::max_by(point_radius, cutoff, |a, b| a.partial_cmp(b).unwrap()).powi(2);
                    if nearest.squared_distance < cutoff{
                        let mut label_text = String::new();
                        let iter = owner.result_names.as_ref().unwrap().iter().enumerate();
                        for (i, (value_name, _)) in iter{
                            if value_name != &"frame"{
                                write!(label_text, "{value_name}: {}\n", row[i]).unwrap();
                            }
                        }
                        label_text.pop();
                        let label = ui.fonts(|fonts| fonts.layout_no_wrap(
                            label_text,
                            TextStyle::Body.resolve(ui.style()),
                            egui::Color32::from_rgb(0, 0, 0),
                        ));
                        // let label = epaint::text::Fonts::layout_no_wrap(
                        // );
                
                        let mut screen_pos = egui::Pos2{
                            x: 10. + lerp(inverse_lerp(nearest.item.0[1], self.databounds.as_ref().unwrap().min.y, self.databounds.as_ref().unwrap().max.y), rect.min.x, rect.max.x),
                            y: lerp(inverse_lerp(nearest.item.0[0], self.databounds.as_ref().unwrap().min.x, self.databounds.as_ref().unwrap().max.x), rect.min.y, rect.max.y)
                            - (label.rect.max.y - label.rect.min.y) * 0.5,
                        };
                        let expansion = 4.0;
                        let mut screen_rect = label.rect.translate(screen_pos.to_vec2()).expand(expansion);
                        screen_pos = screen_pos + egui::Vec2{x: 0.0, y: 
                            float_max(rect.min.y - screen_rect.min.y + 1.0, 0.0) 
                        };
                
                        screen_pos = screen_pos + egui::Vec2{x: 0.0, y: 
                            float_min(rect.max.y - screen_rect.max.y - 1.0, 0.0) 
                        };
                        screen_rect = label.rect.translate(screen_pos.to_vec2()).expand(expansion);
                        if !rect.contains_rect(screen_rect){
                            screen_pos = screen_pos + egui::Vec2{ x: -20.0 - screen_rect.width() + 2.0 * expansion, y: 0.0};
                            screen_rect = label.rect.translate(screen_pos.to_vec2()).expand(expansion);
                        }
                        ui.painter_at(rect).add(
                            epaint::Shape::rect_filled(
                                screen_rect, 2., epaint::Color32::from_rgba_unmultiplied(255, 255, 255, 50)
                            )
                        );
                        ui.painter_at(rect).add(epaint::Shape::galley(screen_pos, label));
                    }
                }
            }
        }
    }
    
    fn custom_painting(&mut self, ui: &mut egui::Ui, direction: Option<FrameChange>) {
        let size = egui::Vec2::splat(600.0) * self.cur_asp;
        let (rect, response) =
            ui.allocate_exact_size(size, egui::Sense::drag());
        
        if self.poll_result().is_err(){
            return
        }

        let mut need_to_update_cirles = false;
        for other in self.other_apps.iter(){
            if let Some(alive_other) = other.upgrade(){
                need_to_update_cirles |= alive_other.borrow().recently_updated;
            }
        }
        if need_to_update_cirles{
            self.update_circles()
        }
        
        if let (Some(direction), true) = (direction, response.hovered()){
            ignore_result(self.update_frame(ui, direction));
        }

        if self.playback.should_frame_advance(ui){
           ignore_result(self.update_frame(ui, FrameChange::Next));
            // ui.ctx().request_repaint();
        }
        
        let uuid = self.uuid;

        let cb = egui_wgpu::CallbackFn::new()
            .paint(move |_info, render_pass, paint_callback_resources| {
                let resources: &HashMap<Uuid, ColormapRenderResources> = paint_callback_resources.get().unwrap();
                let resources = resources.get(&uuid).unwrap();
                resources.paint(render_pass);
            });

        let callback = egui::PaintCallback {
            rect,
            callback: Arc::new(cb),
        };

        ui.painter().add(callback);
        
        if let Some(pos) = response.interact_pointer_pos(){
            // let input = ui.ctx().input();
            // let primary_clicked = input.pointer.primary_clicked();
            // drop(input);
            let primary_clicked = ui.ctx().input(|inp| inp.pointer.primary_clicked());
            if response.drag_started() && primary_clicked{
                self.zoom_box_start = Some(rect.clamp(pos));
            }
            
            if let Some(start) = self.zoom_box_start{
                let pos = rect.clamp(pos);
                let this_rect = normalize_rect(egui::Rect::from_two_pos(start, pos));
                ui.painter_at(rect).rect_stroke(this_rect, 0.0, (1., self.circle_color));
                if response.drag_released() {
                    let this_asp = this_rect.max - this_rect.min;
                    if this_asp.x != 0.0 && this_asp.y != 0.0{
                        self.cur_asp = this_asp / this_asp.max_elem();

                        let t = inverse_lerp_rect(&rect, &this_rect);
                        let texture_zoom_level = lerp_rect(&self.texture_zoom_level, &t);
                        let t = egui::Rect{
                            min: egui::Pos2{ x: t.min.y, y: t.min.x },
                            max: egui::Pos2{ x: t.max.y, y: t.max.x },
                        };
                        let databounds = lerp_rect(&self.databounds.as_ref().unwrap(), &t);
                        self.resize(ui, texture_zoom_level, databounds);
                        self.zoom_box_start = None;
                    }
                }
            } else {
                if response.drag_released(){
                    self.reset_zoom(ui)
                }
            }
        }

        self.result_dependent_plotting(ui, rect, response.hover_pos());
        
        match self.result_status{
            ResultStatus::Valid | ResultStatus::Static => {
            },
            ResultStatus::Processing => {
                match self.mode{
                    DataMode::Off | DataMode::Immediate => {},
                    DataMode::Full => { 
                        ui.put(rect,
                            egui::widgets::ProgressBar::new(
                                self.progress.unwrap_or(0) as f32 / self.vid_len.unwrap() as f32
                            ).desired_width(180.));
                    },
                    DataMode::Range(ref range) => { 
                        let prog = (self.progress.unwrap_or(0) as f32 - *range.start() as f32) 
                                    / (range.end() - range.start()) as f32;
                        // dbg!(prog);
                        ui.put(rect,
                            egui::widgets::ProgressBar::new(
                                prog
                            ).desired_width(180.));
                    },
                }
                ui.ctx().request_repaint();
            }
            ResultStatus::TooOld => {
                match self.mode{
                    DataMode::Off | DataMode::Immediate => {},
                    _ => { ui.put(rect, egui::widgets::Spinner::new().size(60.)); },
                };
                ui.ctx().request_repaint();
            }
        }
    }

    fn point_radius(&self, arrayrow: ArrayView1<f32>) -> f32{
        let point_radius = match self.r_col{
            Some(r_col) => {
               arrayrow[r_col] 
            },
            None => {
                if self.static_dataset{
                    self.plot_radius_fallback
                } else {
                    match self.tracking_params.style{
                        gpu_tracking::gpu_setup::ParamStyle::Trackpy {diameter, .. } => {
                            diameter as f32 / 2.0
                        }
                        gpu_tracking::gpu_setup::ParamStyle::Log { min_radius, .. } => {
                            min_radius
                        }
                    }
                }
            }
        };
        point_radius
    }
}

fn float_max(a: f32, b: f32) -> f32{
    std::cmp::max_by(a, b, |a, b| a.partial_cmp(b).unwrap())
}

fn float_min(a: f32, b: f32) -> f32{
    std::cmp::min_by(a, b, |a, b| a.partial_cmp(b).unwrap())
}

fn lerp(t: f32, min: f32, max: f32) -> f32{
    min + t * (max - min)
}

fn inverse_lerp(dat: f32, min: f32, max: f32) -> f32{
    (dat - min) / (max - min)
}

fn inverse_lerp_rect(outer: &egui::Rect, inner: &egui::Rect) -> egui::Rect{
    egui::Rect{
        min: ((inner.min - outer.min) / (outer.max - outer.min)).to_pos2(),
        max: ((inner.max - outer.min) / (outer.max - outer.min)).to_pos2(),
    }
}

fn lerp_rect(outer: &egui::Rect, t: &egui::Rect) -> egui::Rect{
    egui::Rect{
        min: (outer.min + t.min.to_vec2() * (outer.max - outer.min)),
        max: (outer.min + t.max.to_vec2() * (outer.max - outer.min)),
    }
}

fn zero_one_rect() -> egui::Rect{
    egui::Rect::from_x_y_ranges(0.0..=1.0, 0.0..=1.0)
}

fn data_to_screen_coords_vec2(vec2: egui::Vec2, rect: &egui::Rect, databounds: &egui::Rect) -> egui::Pos2 {
    let t = egui::Vec2::new(inverse_lerp(vec2.x, databounds.min.y, databounds.max.y),
        inverse_lerp(vec2.y, databounds.min.x, databounds.max.x));
    rect.lerp(t)
}

fn data_radius_to_screen(radius: f32, rect: &egui::Rect, databounds: &egui::Rect) -> f32 {
    let t = radius / (databounds.max.y - databounds.min.y);
    t * (rect.max.x - rect.min.x) 
}
fn colormap_dropdown(ui: &mut egui::Ui, input: &mut colormaps::KnownMaps) -> bool{
    let cmapiter = colormaps::KnownMaps::iter();
    let mut clicked = false;
    for cmap in cmapiter{
        clicked |= ui.selectable_value(input, cmap, cmap.get_name()).clicked();
    };
    clicked
}
