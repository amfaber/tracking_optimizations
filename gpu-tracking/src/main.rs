#![allow(warnings)]
use gpu_tracking::execute_gpu::TrackingParams;
use pollster::FutureExt;
use winit::event_loop;
use std::io::Write;
use std::{fs, time::Instant};
use tiff::decoder::{Decoder, DecodingResult};
use gpu_tracking::{execute_gpu::{self, execute_gpu, execute_ndarray}};
use gpu_tracking::decoderiter::IterDecoder;
use futures;
use futures_intrusive;
pub type my_dtype = f32;
use std::path;
use clap::Parser;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[derive(Parser, Debug)]
struct Args{
    input: Option<String>,
    #[arg(short, long)]
    debug: Option<bool>,
    #[arg(short, long)]
    filter: Option<bool>,
    #[arg(short, long)]
    characterize: Option<bool>,
}


fn main() -> anyhow::Result<()> {
    let args: Args = Args::parse();
    let now_top = Instant::now();
    let path = args.input.unwrap_or("../emily_tracking/sample_vids/s_20.tif".to_string());
    let debug = args.debug.unwrap_or(false);
    let filter = args.filter.unwrap_or(true);
    let characterize = args.characterize.unwrap_or(true);
    let params = TrackingParams{
        diameter: 9,
        minmass: 0.,
        separation: 10,
        search_range: Some(9.),
        filter_close: filter,
        characterize,
        ..Default::default()
    };

    let event_loop = EventLoop::new();
    
    // event_loop.run(move |event, _, control_flow|{
        //     *control_flow = ControlFlow::Wait;
        
        //     match event {
            //         Event::WindowEvent {
                //             event: WindowEvent::CloseRequested,
    //             window_id,
    //         } if window_id == window.id() => *control_flow = ControlFlow::Exit,
    //         _ => (),
    //     }
    // });
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    
    let now = Instant::now();
    // let (results, column_names) =
    //     execute_gpu::execute_file(&path, Some(1), params, debug, 1, Some(window));
    // let (results, column_names) =
        execute_gpu::execute_file(&path, Some(1), params, debug, 1, Some(&window));
    let function_time = now.elapsed().as_millis() as f64 / 1000.;
    dbg!(function_time);
    
    
    // if debug{
    //     let mut file = fs::OpenOptions::new().write(true).create(true).truncate(true).open("test").unwrap();
    //     let raw_bytes = unsafe{std::slice::from_raw_parts(results.as_ptr() as *const u8, results.len() * std::mem::size_of::<my_dtype>())};
    //     file.write_all(raw_bytes).unwrap();
    // }

    Ok(())
}
