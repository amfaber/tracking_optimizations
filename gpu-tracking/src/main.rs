#![allow(warnings)]
use pollster::FutureExt;
use std::io::Write;
use std::{fs, time::Instant};
use tiff::decoder::{Decoder, DecodingResult};
use gpu_tracking::{execute_gpu::execute_gpu};
use gpu_tracking::decoderiter::IterDecoder;
use futures;
use futures_intrusive;
use image::{self, EncodableLayout};
pub type my_dtype = f32;
use std::path;
use clap::Parser;

#[derive(Parser, Debug)]
struct Args{
    // #[arg(short, long)]
    input: Option<String>,
}


fn main() -> anyhow::Result<()> {
    let args: Args = Args::parse();
    let now_top = Instant::now();
    let path = match args.input{
        Some(input) => input,
        None => "../emily_tracking/sample_vids/s_20.tif".to_string()
    };
    // let path = r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\grey_lion.tiff";
    // let path = r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\tester.tiff";
    let file = fs::File::open(path).expect("didn't find the file");
    let mut decoder = Decoder::new(file).expect("Can't create decoder");
    let (width, height) = decoder.dimensions().unwrap();
    let dims = [height, width];
    // dbg!(dims);
    let mut decoderiter = IterDecoder::from(decoder);
    let now = Instant::now();
    let results = execute_gpu(decoderiter, dims);
    let function_time = now.elapsed().as_millis() as f64 / 1000.;
    dbg!(function_time);
    
    let mut file = fs::OpenOptions::new().write(true).create(true).truncate(true).open("test").unwrap();
    // file.write(&results[0].as_bytes()).unwrap();
    // let test = results as *const [u8]; 
    file.write(&results.as_bytes()).unwrap();

    let total = now_top.elapsed().as_millis() as f64 / 1000.;
    dbg!(total);
    Ok(())
}

fn _main(){
    let path = path::PathBuf::from(r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\src\shaders");
    let shaders = path.read_dir().expect("Could not read directory")
    .filter_map(|entry_res| {
        entry_res.ok().and_then(|entry|{
            entry.file_name().to_str().and_then(|s| {
                match !s.starts_with("_") && s.ends_with(".wgsl") {
                    true => Some(s.to_string()),
                    false => None,
                }
            })
        })
    })
    .collect::<Vec<_>>();
    dbg!(shaders);
}
