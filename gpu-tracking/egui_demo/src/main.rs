// use egui;
use eframe;
use egui_demo;
fn main() {
    let options = eframe::NativeOptions {
        drag_and_drop_support: true,

        initial_window_size: Some([800., 800.].into()),
        renderer: eframe::Renderer::Wgpu,

        ..Default::default()
    };
    eframe::run_native(
        "egui demo app",
        options,
        Box::new(|cc| Box::new(egui_demo::custom3d_wgpu::Custom3d::new(cc).unwrap())),
    )
}
