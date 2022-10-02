#![allow(warnings)]
use tiff::decoder::{Decoder, DecodingResult};
use crate::my_dtype;

pub struct IterDecoder<R: std::io::Read + std::io::Seek>{
    decoder: Decoder<R>,
    first: bool,
    pub data: Option<Vec<my_dtype>>,
}


impl <'a, R: std::io::Read + std::io::Seek> From::<Decoder<R>> for IterDecoder<R>{
    fn from(decoder: Decoder<R>) -> IterDecoder<R>{
        IterDecoder { decoder, first: true, data: None }
    }
}

impl IterDecoder<R>{
    // fn _treat_data(res: )
}


impl<R: std::io::Read + std::io::Seek> Iterator for IterDecoder<R>{
    type Item = &'a[my_dtype];

    fn next(&mut self) -> Option<Self::Item>{
        if self.first{
            self.first = false;
            match self.decoder.read_image().unwrap(){
                DecodingResult::U16(res) => {
                    return Some(out)
                },
                DecodingResult::U8(res) => {
                    return Some(res.iter().map(|&x| x as my_dtype).collect())
                },
                _ => panic!("Wrong bit depth")
            }
        };

        match self.decoder.more_images(){
            true => {
                self.decoder.next_image().unwrap();
                match self.decoder.read_image().unwrap(){
                    DecodingResult::U16(res) => {
                        return Some(res.iter().map(|&x| x as my_dtype).collect())
                    },
                    DecodingResult::U8(res) => {
                        return Some(res.iter().map(|&x| x as my_dtype).collect())
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