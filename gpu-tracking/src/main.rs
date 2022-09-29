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



fn main() -> anyhow::Result<()> {
    // let path = r"C:\Users\andre\Documents\tracking_optimizations\emily_tracking\sample_vids\s_20.tif";
    // let path = r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\grey_lion.tiff";
    let path = r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\tester.tiff";
    let file = fs::File::open(path).expect("didn't find the file");
    let mut decoder = Decoder::new(file).expect("Can't create decoder");
    let (width, height) = decoder.dimensions().unwrap();
    let dims = [height, width];
    dbg!(dims);
    let mut decoderiter = IterDecoder::from(decoder);
    let mut i = 0;
    let now = Instant::now();
    // let pic = decoderiter.next().unwrap();
    // dbg!(pic.len());
    let pictures = execute_gpu(decoderiter, dims).block_on()?;
    dbg!(now.elapsed().as_millis() as f64 / 1000.);
    // let mut img = image::ImageBuffer::new(width, height);
    // for (x, y, pixel) in img.enumerate_pixels_mut() {
        // *pixel = image::Luma([pictures[0][y as usize * width as usize + x as usize] as u8]);
    // }

    let mut file = fs::OpenOptions::new().write(true).create(true).open("test").unwrap();
    file.write(&pictures[0].as_bytes()).unwrap();
    
    Ok(())
}

