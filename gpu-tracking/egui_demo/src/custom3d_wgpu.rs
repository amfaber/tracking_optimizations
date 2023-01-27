use std::{num::NonZeroU64, sync::{Arc, mpsc::{Receiver, Sender}}, collections::HashMap, path::PathBuf, ops::RangeInclusive};

use eframe::{
    egui_wgpu::{self, wgpu},
    wgpu::util::DeviceExt,
};
use egui::{self, TextStyle, Widget};
use epaint;
use bytemuck;
use thiserror::Error;
use gpu_tracking::{gpu_setup::{TrackingParams, ParamStyle}, execute_gpu::path_to_iter};
use ndarray::{Array, Array2, Axis, ArrayView1};
use crate::{colormaps, texture::ColormapRenderResources};
use kd_tree;
use std::fmt::Write;
use uuid::Uuid;
use anyhow;
// use ordered_float::OrderedFloat;

trait PanicGet{
    type Output;
    fn t(self) -> Self::Output;
    fn r(&self) -> &Self::Output;
}

impl<T> PanicGet for Option<T>{
    type Output = T;

    fn t(self) -> Self::Output{
        self.unwrap()
    }

    fn r(&self) -> &Self::Output{
        self.as_ref().unwrap()
    }
}


impl<T, E: std::fmt::Debug> PanicGet for Result<T, E>{
    type Output = T;

    fn t(self) -> Self::Output{
        self.unwrap()
    }

    fn r(&self) -> &Self::Output{
        self.as_ref().unwrap()
    }
}

trait ColorMap{
    fn call(&self, t: f32) -> epaint::Color32;
}

impl ColorMap for [f32; 120]{
    fn call(&self, t: f32) -> epaint::Color32{
        let t29 = t * 29.;
        let ind = t29 as usize;
        let leftover = t29 - ind as f32;
        let color_view: &[[f32; 4]] = bytemuck::cast_slice(self);
        
        let start = color_view[ind];
        let end = color_view[ind + 1];
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
    fn dims(&self) -> &[u32; 2]{
        &self.0.1
    }
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
    fn get_range(&self) -> &RangeInclusive<usize>{
        match self{
            Self::Range(range) => range,
            _ => panic!("Tried to get range where there is none"),
        }
    }
}

pub struct AppWrapper{
    apps: Vec<Custom3d>,
    opens: Vec<bool>,
}

impl AppWrapper{
    pub fn new<'a>(cc: &'a eframe::CreationContext<'a>) -> Option<Self> {
        // let wgpu_render_state = cc.wgpu_render_state.as_ref()?;
        let apps = vec![
            Custom3d::new()?,
        ];
        let opens = vec![true];
        Some(Self{apps, opens})
    }
}

impl eframe::App for AppWrapper{
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    let response = ui.button("New");
                    if response.clicked(){
                        self.apps.push(Custom3d::new().unwrap());
                        self.opens.push(true);
                    }
                    let mut adds = Vec::new();
                    let mut removes = Vec::new();
                    for (i, (app, open)) in self.apps.iter_mut().zip(self.opens.iter_mut()).enumerate(){
                        egui::containers::Window::new(format!("{}", app.uuid.as_u128()))
                            .title_bar(true)
                            .open(open)
                            .show(ui.ctx(), |ui|{
                                app.get_input(ui, frame);
                                ui.horizontal(|ui| {
                                    let response = ui.button("Clone");
                                    if response.clicked(){
                                        adds.push((i, app.uuid));
                                    }
                                });
                            });
                    }
                    for (i, open) in self.opens.iter().enumerate(){
                        if !*open{
                            removes.push(i);
                        }
                    }
                    for (i, uuid) in adds{
                        let app = self.apps.iter().find(|app| app.uuid == uuid).unwrap();
                        let mut new_app = app.clone();
                        new_app.setup_gpu_after_clone(ui, frame);
                        self.apps.insert(i + 1, new_app);
                        self.opens.insert(i + 1, true);
                    }
                    for i in removes{
                        self.apps.remove(i);
                        self.opens.remove(i);
                    }
                })
        });
    }
}

struct RecalculateJob{
    path: PathBuf,
    tracking_params: TrackingParams,
    result_sender: Sender<anyhow::Result<RecalculateResult>>,
}

pub struct Custom3d {
    // input_path: String,
    
	frame_provider: Option<ProviderDimension>,
    vid_len: Option<usize>,
    frame_idx: usize,
    last_render: Option<std::time::Instant>,
    // gpu_tracking_state: gpu_tracking::gpu_setup::GpuState,
    results: Option<Array2<f32>>,
    result_names: Option<Vec<(&'static str, &'static str)>>,
    circles_to_plot: Option<Array2<f32>>,
    tracking_params: gpu_tracking::gpu_setup::TrackingParams,
    mode: DataMode,
    path: Option<PathBuf>,
    particle_hash: Option<HashMap<usize, Vec<(usize, [f32; 2])>>>,
    circle_kdtree: Option<kd_tree::KdTree<([f32; 2], usize)>>,

    r_col: Option<usize>,
    x_col: Option<usize>,
    y_col: Option<usize>,
    frame_col: Option<usize>,
    particle_col: Option<usize>,

    image_cmap: [f32; 120],
    line_cmap: [f32; 120],
    line_cmap_bounds: RangeInclusive<f32>,
    // line_cmap_end: f32,

    zoom_box_start: Option<egui::Pos2>,
    cur_asp: egui::Vec2,
    texture_zoom_level: egui::Rect,
    databounds: Option<egui::Rect>,

    uuid: Uuid,
    result_status: ResultStatus,
    job_sender: Sender<RecalculateJob>,
    result_receiver: Option<Receiver<anyhow::Result<RecalculateResult>>>,

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
                    // dbg!(1_000_000. / last_advance.elapsed().as_micros() as f32);
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

impl Clone for Custom3d{
    fn clone(&self) -> Self{
        
        let mode = self.mode.clone();
        let cur_asp = self.cur_asp;
        
        let uuid = Uuid::new_v4();
        
        let params = self.tracking_params.clone();
        
        let line_cmap = self.line_cmap;

        let texture_zoom_level = self.texture_zoom_level;

        let (job_sender, job_receiver) = std::sync::mpsc::channel();
        
        std::thread::spawn(move ||{
            loop{
                match job_receiver.recv(){
                    Ok(RecalculateJob { path, tracking_params, result_sender }) => {
                        // let generator = || gpu_tracking::execute_gpu::path_to_iter(&path, None);
                        // let result = RecalculateResult::from(gpu_tracking::execute_gpu::execute_provider(generator, tracking_params.clone(), 0, None).unwrap());
                        let result = RecalculateResult::from(
                            gpu_tracking::execute_gpu::execute_file(&path, None, tracking_params.clone(), 0, None).into()
                        );
                        result_sender.send(result).expect("Main thread lost");
                    },
                    Err(_) => break
                }
            }
        });
        
        let input_state = self.input_state.clone();
        
        let frame_idx = self.frame_idx;
        let image_cmap = self.image_cmap;

        let frame_provider = self.path.as_ref().map(|path| ProviderDimension(path_to_iter(path, None).unwrap()));
        let vid_len = self.vid_len.clone();

        let needs_update = self.needs_update.clone();

        let playback = self.playback.clone();
        
        let out = Self{
            
            frame_provider,
            vid_len,
            frame_idx,
            last_render: None,
            results: self.results.clone(),
            result_names: self.result_names.clone(),
            circles_to_plot: self.circles_to_plot.clone(),
            tracking_params: params,
            mode,
            path: self.path.clone(),
            particle_hash: self.particle_hash.clone(),
            circle_kdtree: self.circle_kdtree.clone(),

            r_col: self.r_col.clone(),
            x_col: self.x_col.clone(),
            y_col: self.y_col.clone(),
            frame_col: self.frame_col.clone(),
            particle_col: self.particle_col.clone(),

            image_cmap,
            line_cmap,
            line_cmap_bounds: self.line_cmap_bounds.clone(),

            zoom_box_start: None,
            cur_asp,
            texture_zoom_level,
            databounds: self.databounds.clone(),

            uuid,
            result_status: self.result_status.clone(),
            result_receiver: None,
            job_sender,

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
    frame_idx: String,
    datamode: DataMode,
    range_start: String,
    range_end: String,
    fps: String,

    style: Style,
    all_options: bool,
    
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
            frame_idx: "0".to_string(),
            datamode: DataMode::Immediate,
            range_start: "0".to_string(),
            range_end: "10".to_string(),
            fps: "30".to_string(),

            style: Style::Trackpy,
            all_options: false,

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
            smoothing_size: "9".to_string(),
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

        let (style, include_r_in_output, smoothing_size_default) = match self.style{
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
            memory: Some(memory.unwrap_or(10)),
            doughnut_correction,
            bg_radius: None,
            gap_radius: None,
            snr: Some(snr.unwrap_or(1.5)),
            minmass_snr: Some(minmass_snr.unwrap_or(0.3)),
            truncate_preprocessed: false,
            illumination_sigma,
            adaptive_background,
            include_r_in_output,
            shift_threshold: shift_threshold.unwrap_or(0.6),
            linker_reset_points: None,
            keys: None,
            noise_size: noise_size.unwrap_or(1.0),
            smoothing_size: Some(smoothing_size.unwrap_or(smoothing_size_default)),
            illumination_correction_per_frame,
        }
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
        let wgpu_render_state = frame.wgpu_render_state().unwrap();
        self.create_gpu_resource(wgpu_render_state);
        self.update_frame(ui, FrameChange::Resubmit);
        self.resize(ui, self.texture_zoom_level, self.databounds.unwrap());
    }

    fn create_gpu_resource(&mut self, wgpu_render_state: &egui_wgpu::RenderState) -> anyhow::Result<()>{
        let provider = self.frame_provider.as_ref().unwrap();
        let frame = provider.to_array(self.frame_idx)?;
        let frame_view = frame.view();
        let resources = ColormapRenderResources::new(wgpu_render_state, &frame_view, self.image_cmap);
        wgpu_render_state
            .renderer
            .write()
            .paint_callback_resources
            .entry::<HashMap<Uuid, ColormapRenderResources>>()
            .or_insert(HashMap::new())
            .insert(self.uuid, resources);
        Ok(())
    }

    pub fn setup_new_path(&mut self, wgpu_render_state: &egui_wgpu::RenderState) -> anyhow::Result<()>{
        let (provider, dims) = match path_to_iter(&self.input_state.path, None){
            Ok(res) => {
                self.path = Some(self.input_state.path.clone().into());
                res
            },
            Err(e) => {
                self.path = None;
                return Err(e.into())
            }
        };
        let provider_dimension = ProviderDimension((provider, dims));
        self.vid_len = Some(provider_dimension.len());
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
        self.input_state.frame_idx = self.frame_idx.to_string();
        self.recalculate();
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
        
        let mut params = gpu_tracking::gpu_setup::TrackingParams::default();
        params.snr = Some(1.3);
        params.search_range = Some(10.);
        
        let line_cmap = colormaps::MAPS["inferno"];

        let texture_zoom_level = zero_one_rect();

        let (job_sender, job_receiver) = std::sync::mpsc::channel();
        
        std::thread::spawn(move ||{
            loop{
                match job_receiver.recv(){
                    Ok(RecalculateJob { path, tracking_params, result_sender }) => {
                        // let generator = || gpu_tracking::execute_gpu::path_to_iter(&path, None);
                        // let result = RecalculateResult::from(gpu_tracking::execute_gpu::execute_provider(generator, tracking_params.clone(), 0, None).unwrap());
                        let result = RecalculateResult::from(
                            gpu_tracking::execute_gpu::execute_file(&path, None, tracking_params.clone(), 0, None).into()
                        );
                        result_sender.send(result).expect("Main thread lost");
                    },
                    Err(_) => break
                }
            }
        });
        
        // let settings_visibility = SettingsVisibility::default();
        let input_state = InputState::default();
        
        let frame_idx = 0;
        let image_cmap = colormaps::MAPS["viridis"];
        
        let needs_update = NeedsUpdate::default();

        let line_cmap_bounds = 0.0..=999.0;

        let playback = Playback::Off;
        
        let out = Self{
            
            frame_provider: None,
            vid_len: None,
            frame_idx,
            last_render: None,
            results: None,
            result_names: None,
            circles_to_plot: None,
            tracking_params: params,
            mode,
            path: None,
            particle_hash: None,
            circle_kdtree: None,

            r_col: None,
            x_col: None,
            y_col: None,
            frame_col: None,
            particle_col: None,

            image_cmap,
            line_cmap,
            line_cmap_bounds,

            zoom_box_start: None,
            cur_asp,
            texture_zoom_level,
            databounds: None,

            uuid,
            result_status: ResultStatus::Processing,
            result_receiver: None,
            job_sender,

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

    fn update_circles_and_lines(&mut self){
        if self.results.is_none() | matches!(self.result_status, ResultStatus::Processing){
            return
        }
        match (&self.particle_hash, &self.mode, self.particle_col){
            (_, DataMode::Immediate, _) => {}
            (None, _, Some(particle_col)) => {
                self.particle_hash = Some(HashMap::new());
                for row in self.results.as_ref().unwrap().axis_iter(Axis(0)){
                    let part_id = row[particle_col];
                    let to_insert = (row[self.frame_col.unwrap()] as usize, [row[self.x_col.unwrap()], row[self.y_col.unwrap()]]);
                    self.particle_hash.as_mut().unwrap().entry(part_id as usize).or_insert(Vec::new()).push(to_insert);
                }
                
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
            _ => {}
        }
        
        let mut subsetter = gpu_tracking::linking::FrameSubsetter::new(
            self.results.as_ref().unwrap().view(),
            Some(0),
            (1, 2),
            None,
            gpu_tracking::linking::SubsetterType::Agnostic,
        );

        if let Some((circles_to_plot, circle_kdtree)) = subsetter.find_map(|ele| {
            match ele{
                Ok((Some(i), gpu_tracking::linking::SubsetterOutput::Agnostic(res))) if i == self.frame_idx => {
                    let kdtree = kd_tree::KdTree::build_by_ordered_float(res.axis_iter(Axis(0)).enumerate().map(|(i, row)| {
                        ([row[self.y_col.unwrap()], row[self.x_col.unwrap()]], i)
                    }).collect());
                    Some((Some(res), Some(kdtree)))
                },
                _ => None
            }
        }) {
            self.circles_to_plot = circles_to_plot;
            self.circle_kdtree = circle_kdtree;
        }
    }
    
    fn update_from_recalculate(&mut self, result: anyhow::Result<RecalculateResult>) -> anyhow::Result<()>{
        let result = result?;
        self.result_receiver = None;
        self.results = Some(result.results);
        self.result_names = Some(result.result_names);
        self.r_col = result.r_col;
        self.x_col = result.x_col;
        self.y_col = result.y_col;
        self.frame_col = result.frame_col;
        self.particle_col = result.particle_col;
        self.result_status = ResultStatus::Valid;
        self.update_circles_and_lines();
        Ok(())
    }

    fn poll_result(&mut self) -> anyhow::Result<()>{
        match self.result_receiver.as_ref(){
            Some(recv) => {
                match recv.try_recv(){
                    Ok(res) => {
                        match self.result_status{
                            ResultStatus::Processing => {
                                self.update_from_recalculate(res)?;
                            },
                            ResultStatus::TooOld => {
                                self.result_receiver = None;
                                self.recalculate()?;
                            },
                            ResultStatus::Valid => unreachable!(),
                        }
                        
                    },
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        
                    },
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {panic!("Thread lost")},
                }
            },
            None => {
                if !matches!(self.result_status, ResultStatus::Valid){
                    assert!(matches!(self.mode, DataMode::Off))
                }
                // assert!(matches!(self.result_status, ResultStatus::Valid))
            }
        }
        Ok(())
    }

    fn update_datamode(&mut self, ui: &mut egui::Ui){
        let old = self.mode.clone();
        let try_block = || -> anyhow::Result<()>{
            let start = &self.input_state.range_start;
            let end = &self.input_state.range_end;
            match &self.input_state.datamode{
                DataMode::Immediate | DataMode::Full | DataMode::Off if self.mode != self.input_state.datamode => {
                    self.line_cmap_bounds = 0.0..=(self.vid_len.unwrap() as f32);
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
                    self.line_cmap_bounds = (start as f32)..=(end as f32);
                    self.update_frame(ui, FrameChange::Resubmit);
                },
                _ => {
                    return Ok(())
                }
            };
            self.recalculate();
            Ok(())
        }();
        if try_block.is_err(){
            self.mode = old;
        }
    }

    fn update_state(&mut self, ui: &mut egui::Ui){
        if self.needs_update.datamode{
            self.update_datamode(ui);
        }

        if self.needs_update.params{
            self.tracking_params = self.input_state.to_trackingparams();
            self.recalculate();
        }

        self.needs_update = NeedsUpdate::default();
    }

    fn get_input(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame){
        ui.vertical(|ui|{
            let textedit = egui::widgets::TextEdit::singleline(&mut self.input_state.path)
                .code_editor()
                .desired_width(600.0)
                .hint_text("Video file path");
            let response = ui.add(textedit);
            if response.changed(){
                let wgpu_render_state = frame.wgpu_render_state().unwrap();
                self.update_state(ui);
                match self.setup_new_path(wgpu_render_state){
                    Ok(_) => {},
                    Err(_) => self.path = None,
                };
            }

            let submit_click = ui.add_enabled(self.path.is_some() & self.needs_update.any(), egui::widgets::Button::new("Submit")).clicked();
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
                    self.update_frame(ui, FrameChange::Input);
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
            
            if submit_click | ui.ctx().input().key_down(egui::Key::Enter){
                self.update_state(ui);
            }

            
            
            match self.path{
                Some(_) => self.show(ui, frame),
                None => {},
            }
        });
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

    fn show(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame){
        let response = ui.button("🏠🔍");
        if response.clicked(){
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

        
        egui::Frame::canvas(ui.style()).show(ui, |ui| {
            let change = FrameChange::from_scroll(ui.ctx().input().scroll_delta);
            self.custom_painting(ui, change);
        });

        self.last_render = Some(std::time::Instant::now());
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
        if self.result_receiver.is_some(){
            self.result_status = ResultStatus::TooOld;
            return Err(StillWaitingError.into())
        }
        let tracking_params = self.tracking_params.clone();
        let path = self.path.as_ref().cloned().unwrap();
        let (result_sender, result_receiver) =  std::sync::mpsc::channel();
        self.result_receiver = Some(result_receiver);
        self.job_sender.send(RecalculateJob{
            path,
            result_sender,
            tracking_params
        }).expect("Thread lost");

        self.result_status = ResultStatus::Processing;
        self.particle_hash = None;
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
    
    fn update_frame(&mut self, ui: &mut egui::Ui, direction: FrameChange) -> anyhow::Result<()>{
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
                self.recalculate();
            },
            _ => {
                self.update_circles_and_lines();
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

    
    fn result_dependent_plotting(&mut self, ui: &mut egui::Ui, rect: egui::Rect, response: egui::Response){
        let circle_plotting = self.circles_to_plot.as_ref().unwrap().axis_iter(Axis(0)).map(|rrow|{
            epaint::Shape::circle_stroke(
                data_to_screen_coords_vec2([rrow[self.x_col.unwrap()], rrow[self.y_col.unwrap()]].into(), &rect, &self.databounds.as_ref().unwrap()),
                {
                    let r = self.point_radius(rrow);
                    data_radius_to_screen(r, &rect, &self.databounds.as_ref().unwrap())
                },
                (1., epaint::Color32::from_rgb(255, 255, 255))
            )
        });
        ui.painter_at(rect).extend(circle_plotting);


        match self.particle_hash{
            Some(ref particle_hash) => {
                let line_plotting = self.circles_to_plot.as_ref().unwrap().axis_iter(Axis(0)).map(|rrow|{
                    let particle = rrow[self.particle_col.unwrap()];
                    let particle_vec = &particle_hash[&(particle as usize)];
                    particle_vec.windows(2).flat_map(|window|{
                        let start = window[0];
                        let end = window[1];
                        if end.0 <= self.frame_idx{
                            let t = inverse_lerp(start.0 as f32, *self.line_cmap_bounds.start(), *self.line_cmap_bounds.end());
                            Some(epaint::Shape::line_segment([
                                data_to_screen_coords_vec2(start.1.into(), &rect, &self.databounds.as_ref().unwrap()),
                                data_to_screen_coords_vec2(end.1.into(), &rect, &self.databounds.as_ref().unwrap()),
                            ], (1., self.line_cmap.call(t))))
                        } else {
                            None
                        }
                    })
        
                }).flatten();
                ui.painter_at(rect).extend(line_plotting);
            }
            None => {}
        }

        if let Some(hover) = response.hover_pos(){

            let hover_data = [
                lerp(inverse_lerp(hover.y, rect.min.y, rect.max.y), self.databounds.as_ref().unwrap().min.y, self.databounds.as_ref().unwrap().max.y),
                lerp(inverse_lerp(hover.x, rect.min.x, rect.max.x), self.databounds.as_ref().unwrap().min.x, self.databounds.as_ref().unwrap().max.x),
            ];
            
            let nearest = self.circle_kdtree.as_ref().unwrap().nearest(&hover_data);
            if let Some(nearest) = nearest{
                let mut cutoff = 0.02 * (self.databounds.as_ref().unwrap().max.x - self.databounds.as_ref().unwrap().min.x);
                let row = self.circles_to_plot.as_ref().unwrap().index_axis(Axis(0), nearest.item.1);
                let point_radius = self.point_radius(row);
                cutoff = std::cmp::max_by(point_radius, cutoff, |a, b| a.partial_cmp(b).unwrap()).powi(2);
                if nearest.squared_distance < cutoff{
                    let mut label_text = String::new();
                    let iter = self.result_names.as_ref().unwrap().iter().enumerate();
                    for (i, (value_name, _)) in iter{
                        if value_name != &"frame"{
                            write!(label_text, "{value_name}: {}\n", row[i]).unwrap();
                        }
                    }
                    label_text.pop();
                    let label = epaint::text::Fonts::layout_no_wrap(
                        &*ui.fonts(),
                        label_text,
                        TextStyle::Body.resolve(ui.style()),
                        egui::Color32::from_rgb(0, 0, 0),
                    );
                    
                    let mut screen_pos = egui::Pos2{
                        x: 10. + lerp(inverse_lerp(nearest.item.0[1], self.databounds.as_ref().unwrap().min.x, self.databounds.as_ref().unwrap().max.x), rect.min.x, rect.max.x),
                        y: lerp(inverse_lerp(nearest.item.0[0], self.databounds.as_ref().unwrap().min.y, self.databounds.as_ref().unwrap().max.y), rect.min.y, rect.max.y)
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
    
    fn custom_painting(&mut self, ui: &mut egui::Ui, direction: Option<FrameChange>) {
        let size = egui::Vec2::splat(600.0) * self.cur_asp;
        let (rect, response) =
            ui.allocate_exact_size(size, egui::Sense::drag());
        
        self.poll_result();
        
        if let (Some(direction), true) = (direction, response.hovered()){
            self.update_frame(ui, direction);
        }

        if self.playback.should_frame_advance(ui){
            self.update_frame(ui, FrameChange::Next);
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
            if response.drag_started(){
                self.zoom_box_start = Some(rect.clamp(pos));
            }
            let pos = rect.clamp(pos);
            let this_rect = normalize_rect(egui::Rect::from_two_pos(self.zoom_box_start.unwrap(), pos));
            ui.painter_at(rect).rect_stroke(this_rect, 0.0, (1., epaint::Color32::from_rgb(255, 255, 255)));
            if response.drag_released() {
                self.cur_asp = this_rect.max - this_rect.min;
                if self.cur_asp.x == 0.0{
                    self.cur_asp.x = 1.0;
                }
                if self.cur_asp.y == 0.0{
                    self.cur_asp.y = 1.0;
                }
                self.cur_asp = self.cur_asp / self.cur_asp.max_elem();

                let t = inverse_lerp_rect(&rect, &this_rect);
                let texture_zoom_level = lerp_rect(&self.texture_zoom_level, &t);
                let databounds = lerp_rect(&self.databounds.as_ref().unwrap(), &t);
                self.resize(ui, texture_zoom_level, databounds);
            }
        }

        match self.result_status{
            ResultStatus::Valid => {
                self.result_dependent_plotting(ui, rect, response);
            },
            ResultStatus::Processing => {
                match self.mode{
                    DataMode::Off | DataMode::Immediate => {},
                    _ => { ui.put(rect, egui::widgets::Spinner::new().size(60.)); },
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
                match self.tracking_params.style{
                    gpu_tracking::gpu_setup::ParamStyle::Trackpy {diameter, .. } => {
                        diameter as f32 / 2.0
                    }
                    _ => panic!("we shouldn't get here")
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

// fn remap_rect(from: &egui::Rect, to: &egui::Rect) -> egui::Rect{
//     lerp_rect(to, &inverse_lerp_rect(to, from))
// }

fn zero_one_rect() -> egui::Rect{
    egui::Rect::from_x_y_ranges(0.0..=1.0, 0.0..=1.0)
}

fn data_to_screen_coords_vec2(vec2: egui::Vec2, rect: &egui::Rect, databounds: &egui::Rect) -> egui::Pos2 {
    let t = egui::Vec2::new(inverse_lerp(vec2.x, databounds.min.x, databounds.max.x), inverse_lerp(vec2.y, databounds.min.y, databounds.max.y));
    rect.lerp(t)
}

fn data_radius_to_screen(radius: f32, rect: &egui::Rect, databounds: &egui::Rect) -> f32 {
    let t = radius / (databounds.max.x - databounds.min.x);
    // dbg!(t);
    t * (rect.max.x - rect.min.x) 
}


