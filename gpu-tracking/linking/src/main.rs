use linking::linking;
use std::{fs::{File, self}, collections::{VecDeque, HashMap, hash_map::Entry}, io::Write};
use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
use ndarray::Array2;
use std::time::Instant;


fn main() {
    let file = File::open("trackpy_reference/located.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let array_read: Array2<f32> = reader.deserialize_array2_dynamic().unwrap();
    // dbg!(array_read);
    // for i in 0..100{
    let mut frame_iter = linking::FrameSubsetter::new(&array_read, 2);
    
    let now = Instant::now();
    let results = linking::link_all(frame_iter, 9., 0);
    let elapsed = now.elapsed().as_millis() as f64 / 1000.0;
    dbg!(elapsed);
    
    let results = results.into_iter()
    .map(|(frame_idx, coords, part_id)| [frame_idx as f32, coords[0], coords[1], part_id as f32]).flatten().collect::<Vec<_>>();
    results.as_slice();
    let raw_bytes = unsafe{std::slice::from_raw_parts(results.as_ptr() as *const u8, results.len() * std::mem::size_of::<f32>())};
    let mut file = fs::OpenOptions::new().write(true).create(true).truncate(true).open("test").unwrap();
    file.write_all(raw_bytes).unwrap();
}
