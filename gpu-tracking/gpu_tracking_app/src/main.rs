// use egui;
use eframe;
use gpu_tracking_app;
fn main() {
    let options = eframe::NativeOptions {
        drag_and_drop_support: true,
        // maximized: true,

        initial_window_size: Some([1200., 1000.].into()),
        renderer: eframe::Renderer::Wgpu,

        ..Default::default()
    };
    eframe::run_native(
        "gpu_tracking",
        options,
        Box::new(|cc| Box::new(gpu_tracking_app::custom3d_wgpu::AppWrapper::test(cc).unwrap())),
    ).unwrap();
}
