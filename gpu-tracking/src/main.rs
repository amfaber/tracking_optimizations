#![allow(warnings)]
use gpu_tracking::gpu_setup::Style;
use ndarray::{Array2, s};
use pollster::FutureExt;
use std::io::Write;
use std::{fs, time::Instant};
use tiff::decoder::{Decoder, DecodingResult};
use gpu_tracking::{
    execute_gpu::{self, execute_gpu, execute_ndarray},
    decoderiter::IterDecoder,
    gpu_setup::{TrackingParams, LogParams, TrackpyParams}};
use futures;
use futures_intrusive;
pub type my_dtype = f32;
use std::path;
use clap::Parser;

#[derive(Parser, Debug)]
struct Args{
    // #[arg(short, long)]
    input: Option<String>,
    #[arg(short, long)]
    debug: Option<bool>,
    #[arg(short, long)]
    filter: Option<bool>,
    #[arg(short, long)]
    characterize: Option<bool>,
    // #[arg(short, long)]
    // processed_cpu: Option<bool>,
}

// macro_rules! test {
//     // (@helper $inside:ident; $y:ident $($body:tt)*) => {
//     //     {
//     //         println!("inside func {}", $y);
//     //         let mut $inside = 1;
//     //         $($body)*
//     //     }
//     // };

//     (fn $name:ident($($args:tt)*) -> $outtype:ty => $params:ident $body:block) => {
//         fn $name($($args)*, y: i32) -> $outtype {
//             let mut $params = 1;
//             $body
//         }
//     }
// }


// test!(
//     fn idk(x: i32, u: usize) -> i32 => inside{
        
//         inside
//     }
// );

fn main() -> anyhow::Result<()> {
    let args: Args = Args::parse();
    let now_top = Instant::now();
    // let path = args.input.unwrap_or("../emily_tracking/sample_vids/s_20.tif".to_string());
    let path = args.input.unwrap_or("testing/scuffed_blobs_1C.tif".to_string());
    let debug = args.debug.unwrap_or(false);
    let filter = args.filter.unwrap_or(true);
    let characterize = args.characterize.unwrap_or(false);
    // let processed_cpu = args.processed_cpu.unwrap_or(false);
    let file = fs::File::open(&path).expect("didn't find the file");
    let mut decoder = Decoder::new(file).expect("Can't create decoder");
    let (width, height) = decoder.dimensions().unwrap();
    let dims = [height, width];
    // // // dbg!(dims);
    let mut decoderiter = IterDecoder::from(decoder);
    // let all_frames = decoderiter.collect::<Vec<_>>();
    // let all_views = all_frames.iter().map(|x| x.view()).collect::<Vec<_>>();
    // let arr = ndarray::stack(ndarray::Axis(0), &all_views).unwrap();


    // let results = execute_ndarray(&arr.view(), TrackingParams::default(), true);
    let params = TrackingParams{
        diameter: 9,
        minmass: 800.,
        separation: 10,
        filter_close: filter,
        // search_range: Some(9.),
        // characterize,
        // cpu_processed: processed_cpu,
        // sig_radius: Some(3.),
        // bg_radius: Some((60 as f32).sqrt()),
        // gap_radius: Some(0.5),
        // varcheck: Some(1.),
        style: Style::Log(LogParams{
            min_sigma: 10.,
            max_sigma: 50.,
            n_sigma: 10,
            log_spacing: false,
        }),
        ..Default::default()
    };
    // let mut decoderiter = match debug {
    //         true => decoderiter.take(1),
    //         false => decoderiter.take(usize::MAX)
    //     };
    // let now = Instant::now();
    // let points = vec![(0usize, vec![[300f32, 300f32], [200f32, 200f32]]), (200usize, vec![[300f32, 300f32], [200f32, 200f32]])];
    // let (results, column_names) = execute_gpu::execute_file(&path, Some(1), params, debug, 1, Some(points.into_iter()));
    // // let (results, shape) = execute_gpu(&mut decoderiter, &dims, params, debug, 1);
    // dbg!(&results);
    // let function_time = now.elapsed().as_millis() as f64 / 1000.;
    // dbg!(function_time);
    let now = Instant::now();
    let points = vec![0f32, 300f32, 300f32, 0f32, 200f32, 200f32, 200f32, 300f32, 300f32, 2000f32, 200f32, 200f32];
    let points = Array2::from_shape_vec((4, 3), points).unwrap();
    let (results, column_names) = execute_gpu::execute_gpu(decoderiter, &dims, params, debug, 1, None::<std::vec::IntoIter<(usize, Vec<[my_dtype; 2]>)>>);
    // let (results, column_names) = execute_gpu::execute_ndarray(&arr.view(), params, debug, 1, None);
    // let (results, shape) = execute_gpu(&mut decoderiter, &dims, params, debug, 1);
    // dbg!(&results.slice(s![..600, ..]));
    let function_time = now.elapsed().as_millis() as f64 / 1000.;
    dbg!(function_time);
    dbg!(&results.len());
    
    std::fs::write("testing/dump.bin", bytemuck::cast_slice(&results));
    // if debug{
        // let mut file = fs::OpenOptions::new().write(true).create(true).truncate(true).open("test").unwrap();
        // let raw_bytes = unsafe{std::slice::from_raw_parts(results.as_ptr() as *const u8, results.len() * std::mem::size_of::<my_dtype>())};
        // file.write_all(raw_bytes).unwrap();
    // }

    // let total = now_top.elapsed().as_millis() as f64 / 1000.;
    // dbg!(total);
    Ok(())
}
