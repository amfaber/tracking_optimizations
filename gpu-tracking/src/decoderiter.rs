use tiff::decoder::{Decoder, DecodingResult};
use ndarray::{Array2};
use crate::my_dtype;

// type item_type = Vec<my_dtype>;
type item_type = Array2<my_dtype>;

pub struct IterDecoder<R: std::io::Read + std::io::Seek>{
    decoder: Decoder<R>,
    // pub dims: [u32; 2],
    pub dims: (usize, usize),
    first: bool
}


impl <R: std::io::Read + std::io::Seek> From::<Decoder<R>> for IterDecoder<R>{
    fn from(mut decoder: Decoder<R>) -> IterDecoder<R>{
        let (width, height) = decoder.dimensions().unwrap();
        // let dims = [height, width];
        let dims = (height as usize, width as usize);
        IterDecoder {
            decoder,
            dims,
            first: true
        }
    }
}

impl<R: std::io::Read + std::io::Seek> IterDecoder<R>{
    fn _treat_data<T: Into<my_dtype> + Copy>(&self, res: Vec<T>) -> item_type{
        let data = res.iter().map(|&x| x.into()).collect::<Vec<my_dtype>>();
        let data = ndarray::Array::from_shape_vec(self.dims, data).unwrap();
        data
    }
}

impl <R: std::io::Read + std::io::Seek> Iterator for IterDecoder<R>{
    type Item = item_type;
    fn next(&mut self) -> Option<Self::Item>{
        if self.first{
            self.first = false;
            match self.decoder.read_image().unwrap(){
                DecodingResult::U16(res) => {
                    return Some(self._treat_data(res));
                },
                DecodingResult::U8(res) => {
                    return Some(self._treat_data(res))
                },
                _ => panic!("Wrong bit depth")
            }
        };
        match self.decoder.more_images(){
            true => {
                self.decoder.next_image().unwrap();
                match self.decoder.read_image().unwrap(){
                    DecodingResult::U16(res) => {
                        return Some(self._treat_data(res))
                    },
                    DecodingResult::U8(res) => {
                        return Some(self._treat_data(res))
                    },
                    _ => panic!("Wrong bit depth")
                }
            },
            false => {
                return None
            }
        };
    }
}