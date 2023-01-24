use std::{num::NonZeroU64, sync::Arc, collections::HashMap};

use eframe::{
    egui_wgpu::{self, wgpu},
    wgpu::util::DeviceExt,
};
use egui;
use epaint;
use bytemuck;
use emath;
use ndarray::{Array, Array2, Axis};
use crate::colormaps;

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
            *o = ((s + t * (e - s)) * 255.) as u8;
        }
        epaint::Color32::from_rgba_unmultiplied(out[0], out[1], out[2], out[3])
    }
}

use crate::texture::ColormapRenderResources;

type FileProvider = Box<dyn gpu_tracking::decoderiter::FrameProvider<
		Frame = Vec<f32>,
		FrameIter = Box<dyn Iterator<Item = Result<Vec<f32>, gpu_tracking::error::Error>>>
	>>;

struct ProviderDimension((FileProvider, [u32; 2]));

impl ProviderDimension{
    fn to_array(&self, frame_idx: usize) -> Array2<f32>{
        let (provider, dims) = &self.0;
        let frame = provider.get_frame(frame_idx).unwrap();
        let frame = Array::from_shape_vec([dims[0] as usize, dims[1] as usize], frame).unwrap();
        frame
    }
}

enum DataMode{
    Immediate,
    Range(std::ops::RangeInclusive<usize>),
    Full,
}

pub struct Custom3d {
	frame_provider: Option<ProviderDimension>,
    frame_idx: usize,
    last_render: Option<std::time::Instant>,
    gpu_tracking_state: gpu_tracking::gpu_setup::GpuState,
    results: Option<Array2<f32>>,
    result_names: Option<Vec<(&'static str, &'static str)>>,
    circles_to_plot: Option<Array2<f32>>,
    tracking_params: gpu_tracking::gpu_setup::TrackingParams,
    mode: DataMode,
    path: &'static str,
    particle_hash: Option<HashMap<usize, Vec<(usize, [f32; 2])>>>,

    r_col: Option<usize>,
    x_col: Option<usize>,
    y_col: Option<usize>,
    frame_col: Option<usize>,
    particle_col: Option<usize>,

    line_cmap: [f32; 120],
    line_cmap_end: f32,
}

impl Custom3d {
    pub fn new<'a>(cc: &'a eframe::CreationContext<'a>) -> Option<Self> {
        // Get the WGPU render state from the eframe creation context. This can also be retrieved
        // from `eframe::Frame` when you don't have a `CreationContext` available.
        let mode = DataMode::Range(0..=100);
        let path = r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\testing\easy_test_data.tif";
        let generator = || gpu_tracking::execute_gpu::path_to_iter(path, None);
        let wgpu_render_state = cc.wgpu_render_state.as_ref().unwrap();
        let (provider, dims) = generator().unwrap();
        let frame_idx = 0;
        let frame = provider.get_frame(frame_idx).unwrap();
        let frame = Array::from_shape_vec([dims[0] as usize, dims[1] as usize], frame).unwrap();
        let frame_view = frame.view();
        let resources = crate::texture::ColormapRenderResources::new(wgpu_render_state, &frame_view);
        wgpu_render_state
            .renderer
            .write()
            .paint_callback_resources
            .insert(resources);
        let mut params = gpu_tracking::gpu_setup::TrackingParams::default();
        params.snr = Some(1.3);
        params.search_range = Some(10.);
        // if let gpu_tracking::gpu_setup::ParamStyle::Trackpy{ref mut diameter, ..} = params.style{
            // *diameter = 11;
        // }
        let gpu_state = gpu_tracking::gpu_setup::setup_state(&params, &dims, false).unwrap();
        
        let line_cmap = colormaps::MAPS["inferno"];
        
        let mut out = Self{
            frame_provider: Some(ProviderDimension((provider, dims))),
            frame_idx,
            last_render: None,
            gpu_tracking_state: gpu_state,
            results: None,
            result_names: None,
            circles_to_plot: None,
            tracking_params: params,
            mode,
            path,
            particle_hash: None,

            r_col: None,
            x_col: None,
            y_col: None,
            frame_col: None,
            particle_col: None,

            line_cmap,
            line_cmap_end: 100.,
        };
        out.recalculate();
        Some(out)
    }
}

impl eframe::App for Custom3d {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        ui.label("The triangle is being painted using ");
                        ui.hyperlink_to("WGPU", "https://wgpu.rs");
                        ui.label(" (Portable Rust graphics API awesomeness)");
                    });
                    ui.label("It's not a very impressive demo, but it shows you can embed 3D inside of egui.");
                    let idk = ui.button("test");
                    if idk.clicked(){
                        self.next_frame(ui);
                    }
                    egui::Frame::canvas(ui.style()).show(ui, |ui| {
                        self.custom_painting(ui);
                    });
                    ui.label("Drag to rotate!");
                    // if let Some(time) = self.last_render{
                    //     let fps = 1./(time.elapsed().as_micros() as f64 / 1_000_000.);
                    //     dbg!(fps);
                    //     dbg!(self.frame_idx);
                    // }
                    self.last_render = Some(std::time::Instant::now());
                    
                });
        });
    }
}

impl Custom3d {

    fn recalculate(&mut self) {
        let generator = || gpu_tracking::execute_gpu::path_to_iter(self.path, None);
        match &self.mode{
            DataMode::Immediate => {
                self.tracking_params.keys = Some(vec![self.frame_idx]);
                self.results = None;
            },
            DataMode::Range(range) => {
                self.tracking_params.keys = Some(range.clone().collect());
            }
            DataMode::Full => {
                self.tracking_params.keys = None;
            }
        }
        match self.results{
            Some(_) => {},
            None => {
                let (results, result_names) = gpu_tracking::execute_gpu::execute_provider(generator, self.tracking_params.clone(), 0, None).unwrap();
                self.results = Some(results);
                self.result_names = Some(result_names);
                self.r_col = self.result_names.as_ref().unwrap().iter().position(|element|{
                    element.0 == "r"
                });
                self.x_col = self.result_names.as_ref().unwrap().iter().position(|element|{
                    element.0 == "x"
                });
                self.y_col = self.result_names.as_ref().unwrap().iter().position(|element|{
                    element.0 == "y"
                });
                self.frame_col = self.result_names.as_ref().unwrap().iter().position(|element|{
                    element.0 == "frame"
                });
                self.particle_col = self.result_names.as_ref().unwrap().iter().position(|element|{
                    element.0 == "particle"
                });
            },
        }

        match (&self.particle_hash, &self.mode){
            (_, DataMode::Immediate) => {}
            (None, _) => {
                self.particle_hash = Some(HashMap::new());
                for row in self.results.as_ref().unwrap().axis_iter(Axis(0)){
                    let part_id = row[self.particle_col.unwrap()];
                    let to_insert = (row[self.frame_col.unwrap()] as usize, [row[self.x_col.unwrap()], row[self.y_col.unwrap()]]);
                    self.particle_hash.as_mut().unwrap().entry(part_id as usize).or_insert(Vec::new()).push(to_insert);
                }
            }
            _ => {}
        }
        
        let mut circles_to_plot = gpu_tracking::linking::FrameSubsetter::new(
            self.results.as_ref().unwrap().view(),
            Some(0),
            (1, 2),
            None,
            gpu_tracking::linking::SubsetterType::Agnostic,
        );

        self.circles_to_plot = circles_to_plot.find_map(|ele| {
            match ele{
                Ok((Some(i), gpu_tracking::linking::SubsetterOutput::Agnostic(res))) if i == self.frame_idx => {
                    Some(res)
                },
                _ => None
            }
        });
    }
    
    fn next_frame(&mut self, ui: &mut egui::Ui){
        self.frame_idx += 1;
        // match self.mode{
        //     DataMode::Immediate => {
        self.recalculate();
        //     },
        //     _ => {}
        // }
        let array = self.frame_provider.as_ref().unwrap().to_array(self.frame_idx);

        let cb = egui_wgpu::CallbackFn::new()
            .prepare(move |_device, queue, _encoder, paint_callback_resources|{
                let resources: &mut ColormapRenderResources = paint_callback_resources.get_mut().unwrap();
                let array_view = array.view();
                resources.update_texture(queue, &array_view);
                Vec::new()
            });
        ui.painter().add(egui::PaintCallback{
            rect: egui::Rect::EVERYTHING,
            callback: Arc::new(cb),
        });
    }

    
    
    fn custom_painting(&mut self, ui: &mut egui::Ui) {
        let (rect, response) =
            ui.allocate_exact_size(egui::Vec2::splat(600.0), egui::Sense::drag());

        let cb = egui_wgpu::CallbackFn::new()
            .paint(move |_info, render_pass, paint_callback_resources| {
                let resources: &crate::texture::ColormapRenderResources = paint_callback_resources.get().unwrap();
                resources.paint(render_pass);
            });

        let callback = egui::PaintCallback {
            rect,
            callback: Arc::new(cb),
        };

        let databounds = egui::Rect::from_x_y_ranges(
            0.0..=(self.frame_provider.as_ref().unwrap().0.1[0] - 1) as f32,
            0.0..=(self.frame_provider.as_ref().unwrap().0.1[1] - 1) as f32,
        );
        
        ui.painter().add(callback);
        
        let circle_plotting = self.circles_to_plot.as_ref().unwrap().axis_iter(Axis(0)).map(|rrow|{
            epaint::Shape::circle_stroke(
                data_to_screen_coords_vec2([rrow[self.x_col.unwrap()], rrow[self.y_col.unwrap()]].into(), &rect, &databounds),
                {
                    let r = match self.r_col{
                        Some(r_col) => {
                           rrow[r_col] 
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
                    data_radius_to_screen(r, &rect, &databounds)
                },
                (1., epaint::Color32::from_rgb(255, 255, 255))
            )
        });
        ui.painter_at(rect).extend(circle_plotting);


        let line_plotting = self.circles_to_plot.as_ref().unwrap().axis_iter(Axis(0)).map(|rrow|{
            let particle = rrow[self.particle_col.unwrap()];
            let particle_vec = &self.particle_hash.as_ref().unwrap()[&(particle as usize)];
            particle_vec.windows(2).flat_map(|window|{
                let start = window[0];
                let end = window[1];
                if end.0 <= self.frame_idx{
                    let t = start.0 as f32 / self.line_cmap_end;
                    Some(epaint::Shape::line_segment([
                        data_to_screen_coords_vec2(start.1.into(), &rect, &databounds),
                        data_to_screen_coords_vec2(end.1.into(), &rect, &databounds),
                    ], (1., self.line_cmap.call(t))))
                } else {
                    None
                }
            })
            
        }).flatten();
        ui.painter_at(rect).extend(line_plotting);
        // ui.painter_at(rect).circle_stroke(
        //     data_to_screen_coords_vec2([255.5, 255.5].into(), &rect, &databounds),
        //     data_radius_to_screen(256., &rect, &databounds),
        //     (2., epaint::Color32::from_rgb(255, 255, 255))
        // );
        // for i in 0..normer{
        //     // ui.painter_at(rect).add(
        //     //     epaint::Shape::circle_stroke([i as f32, i as f32].into(), 30., (1., epaint::Color32::from_rgb(0, 0, 0)))
        //     // );
        // }
    }
}

fn inverse_lerp(dat: f32, min: f32, max: f32) -> f32{
    (dat - min) / (max - min)
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


