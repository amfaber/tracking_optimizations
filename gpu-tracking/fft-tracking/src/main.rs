use fft_tracking::decode_iter::IterDecoder;
use std::time::Instant;
use std::fs;
use tiff::decoder::{Decoder};
use clap::Parser;
#[derive(Parser, Debug)]
struct Args{
    // #[arg(short, long)]
    input: Option<String>,
}



fn main() {
    let args: Args = Args::parse();
    let now_top = Instant::now();
    let path = match args.input{
        Some(input) => input,
        None => "../../emily_tracking/sample_vids/s_20.tif".to_string()
    };
    // let path = r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\grey_lion.tiff";
    // let path = r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\tester.tiff";
    let file = fs::File::open(path).expect("didn't find the file");
    let mut decoder = Decoder::new(file).expect("Can't create decoder");
    let (width, height) = decoder.dimensions().unwrap();
    let dims = [height, width];
    // // dbg!(dims);
    let mut decoderiter = IterDecoder::from(decoder).take(1);
    
}
