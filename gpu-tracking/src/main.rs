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

fn main() -> anyhow::Result<()> {
    let now_top = Instant::now();
    let path = r"C:\Users\andre\Documents\tracking_optimizations\emily_tracking\sample_vids\s_20.tif";
    // let path = r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\grey_lion.tiff";
    // let path = r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\tester.tiff";
    let file = fs::File::open(path).expect("didn't find the file");
    let mut decoder = Decoder::new(file).expect("Can't create decoder");
    let (width, height) = decoder.dimensions().unwrap();
    let dims = [height, width];
    // dbg!(dims);
    let mut decoderiter = IterDecoder::from(decoder);
    let now = Instant::now();
    let pictures = execute_gpu(decoderiter, dims).block_on()?;
    
    dbg!(now.elapsed().as_millis() as f64 / 1000.);
    let coords = pictures.iter()
    .map(|frame_coords| {
        // let mut vec = Vec::with_capacity(1000);
        let mut vec = Vec::new();
        let iter = frame_coords.into_iter()
        .enumerate().step_by(2)
        .filter_map(|(i, &u_part)|{
            let v_part = frame_coords[i+1];
            match u_part != 0.0 || v_part != 0.0 {
                true => Some([u_part, v_part]),
                false => None,
            }
        });
        vec.extend(iter);
        vec
    });
    // let test = coords.size_hint();
    let coords = coords.collect::<Vec<_>>();
    // dbg!(coords);

    // dbg!(coords[0].len());

    
    // dbg!(pictures[0].iter().filter(|&ele| *ele != 0. as my_dtype).count() / 2);


    // let mut img = image::ImageBuffer::new(width, height);
    // for (x, y, pixel) in img.enumerate_pixels_mut() {
    //     *pixel = image::Luma([pictures[0][y as usize * width as usize + x as usize] as u8]);
    // }
    // img.save("test.tiff")?;

    // let mut file = fs::OpenOptions::new().write(true).create(true).open("test").unwrap();
    // // file.write(&pictures[0].as_bytes()).unwrap();
    // file.write(coords[0].iter().flat_map(|ele| *ele).collect::<Vec<_>>()[..].as_bytes()).unwrap();

    let total = now_top.elapsed().as_millis() as f64 / 1000.;
    dbg!(total);
    Ok(())
}

