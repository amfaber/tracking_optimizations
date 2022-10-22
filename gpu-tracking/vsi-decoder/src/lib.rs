use std::{collections::HashMap, io::{Seek, Read, self, SeekFrom}};

use byteorder::{ReadBytesExt, LittleEndian};

#[derive(Debug, Clone)]
pub struct MinimalETSParser{//<R: Read + Seek>{
    // pub reader: R,
    pub offsets: HashMap<usize, Vec<Option<u64>>>,
    pub dims: [usize; 2],
    dataType: ETSDataType,
}

pub enum ETSDataBuffer<'a>{
    I8(&'a mut [i8]),
    U8(&'a mut [u8]),
    I16(&'a mut [i16]),
    U16(&'a mut [u16]),
    I32(&'a mut [i32]),
    U32(&'a mut [u32]),
    F32(&'a mut [f32]),
}

#[derive(Debug, Clone, Copy)]

enum ETSDataType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
}

impl ETSDataType {
    pub fn size(&self) -> usize {
        match self {
            ETSDataType::I8 => 1,
            ETSDataType::U8 => 1,
            ETSDataType::I16 => 2,
            ETSDataType::U16 => 2,
            ETSDataType::I32 => 4,
            ETSDataType::U32 => 4,
            ETSDataType::F32 => 4,
        }
    }
    pub fn from_code(code: u32) -> ETSDataType {
        match code {
            1 => ETSDataType::I8,
            2 => ETSDataType::U8,
            3 => ETSDataType::I16,
            4 => ETSDataType::U16,
            5 => ETSDataType::I32,
            6 => ETSDataType::U32,
            9 => ETSDataType::F32,
            _ => panic!("Unknown ETS data type code: {}", code),
        }
    }
}


impl MinimalETSParser{
    pub fn new<R: Read + Seek>(reader: &mut R) -> io::Result<Self>{
        reader.seek(SeekFrom::Start(0))?;
        let mut buf4 = vec![0; 4];
        // let mut buf8 = vec![0; 8];
        reader.read_exact(&mut buf4)?;
        assert_eq!("SIS\x00".as_bytes(), &buf4[..]);
        reader.seek(SeekFrom::Current(8))?;
        let ndimensions = reader.read_u32::<LittleEndian>()?;
        let additionalHeaderOffset = reader.read_u64::<LittleEndian>()?;
        reader.seek(SeekFrom::Current(8))?;
        let usedChunkOffset = reader.read_u64::<LittleEndian>()?;
        let nUsedChunks = reader.read_u32::<LittleEndian>()?;
        
        reader.seek(SeekFrom::Start(additionalHeaderOffset.into()))?;
        reader.read_exact(&mut buf4)?;
        assert_eq!("ETS\x00".as_bytes(), &buf4[..]);
        reader.seek(SeekFrom::Current(4))?;

        let dataType = ETSDataType::from_code(reader.read_u32::<LittleEndian>()?);

        reader.seek(SeekFrom::Current(16))?;
        let dims = [reader.read_u32::<LittleEndian>()? as usize, reader.read_u32::<LittleEndian>()? as usize];

        reader.seek(SeekFrom::Start(usedChunkOffset))?;
        let mut offsets = HashMap::new();
        for _ in 0..nUsedChunks{
            reader.seek(SeekFrom::Current(4))?;
            let coord = (0..ndimensions).map(|_| reader.read_u32::<LittleEndian>().unwrap() as usize).collect::<Vec<_>>();
            let entry = offsets.entry(coord[4]).or_insert(Vec::new());
            if coord[2] + 1 == entry.len(){
                entry.push(Some(reader.read_u64::<LittleEndian>().unwrap()));
            }
            else if coord[2] + 1 > entry.len(){
                entry.resize(coord[2] + 1, None);
                entry[coord[2]] = Some(reader.read_u64::<LittleEndian>().unwrap());
            }
            else {
                entry[coord[2]] = Some(reader.read_u64::<LittleEndian>().unwrap());
            }
            reader.seek(SeekFrom::Current(8))?;
        }

        let parser = MinimalETSParser{
            // reader,
            offsets,
            dims,
            dataType,
        };
        Ok(parser)
    }

    // pub fn u32_with_buf(reader: &mut R, buf: &mut [u8]) -> io::Result<u32> {
    //     reader.read_exact(buf)?;
    //     Ok(u32::from_le_bytes(buf.try_into().unwrap()))
    // }

    pub fn iterate_channel<R: Read + Seek>(&self, reader: R, channel: usize) -> ETSIterator<R>{
        let read_size = self.dims[0] * self.dims[1] * self.dataType.size();
        let offsets = self.offsets.get(&channel).unwrap().clone();
        ETSIterator{
            reader,
            // parser: self.reader,
            current: 0,
            offsets,
            read_size,
        }
    }
}



pub struct ETSIterator<R: Read + Seek>{
    reader: R,
    current: usize,
    offsets: Vec<Option<u64>>,
    read_size: usize,
}

impl<R: Read + Seek> ETSIterator<R>{
    pub fn new(reader: R, offsets: Vec<Option<u64>>, read_size: usize) -> Self{
        ETSIterator{
            reader,
            current: 0,
            offsets,
            read_size,
        }
    }

    pub fn len(&self) -> usize{
        self.offsets.iter().flatten().count()
    }
}

impl<R: Read + Seek> Iterator for ETSIterator<R>{
    type Item = Option<Vec<u16>>;
    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.offsets.get(self.current)?;
        match offset{
            Some(offset) => {
                self.reader.seek(SeekFrom::Start(*offset)).unwrap();
                
                #[cfg(target_endian = "big")]
                let mut buf = vec![0u8; self.read_size];
                #[cfg(target_endian = "little")]
                let mut buf = vec![0u16; self.read_size / 2];
                // println!("u16 buf.len {}", buf.len());
                #[cfg(target_endian = "little")]
                let byte_buf = unsafe{std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 2)};
                // println!("u8 buf.len {}", buf.len());
                
                self.reader.read_exact(byte_buf).unwrap();
                
                #[cfg(target_endian = "big")]
                let buf = buf.chunks_exact(2).map(|chunk| {
                    u16::from_le_bytes(chunk.try_into().unwrap())
                }).collect::<Vec<_>>();
                self.current += 1;
                Some(Some(buf))
            },
            None => {
                self.current += 1;
                Some(None)
            }
        }
    }
}