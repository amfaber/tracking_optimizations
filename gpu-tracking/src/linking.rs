// #![allow(warnings)]
// #![allow(const_item_mutation)]
use kd_tree::{KdPoint, KdTree, KdIndexTree};
use std::cmp::Ordering;
use ndarray::{ArrayView2, Array2, Array, Axis, s};
use kd_tree;
use std::{collections::{HashMap, HashSet, VecDeque}, iter::FromIterator};
type float = f32;
use typenum::{self, U2};
use num_traits;
use crate::execute_gpu::ResultRow;
use std::panic;

pub enum SubsetterOutput{
    Linking(Vec<[float; 2]>),
    Characterization(Vec<ResultRow>),
    Agnostic(Array2<float>),
}

impl SubsetterOutput{
    fn len(&self) -> usize{
        match self{
            Self::Linking(vec) => vec.len(),
            Self::Characterization(vec) => vec.len(),
            Self::Agnostic(array) => array.len(),
        }
    }
}

#[derive(Clone)]
pub enum SubsetterType{
    Linking,
    Characterization,
    Agnostic,
}

#[derive(Clone)]
pub struct FrameSubsetter<'a>{
    pub frame_col: Option<usize>,
    pub r_col: Option<usize>,
    pub array: ArrayView2<'a, float>,
    pub positions: (usize, usize),
    idx: usize,
    cur_frame: float,
    ty: SubsetterType,
    // iter: ndarray::iter::Iter<'a, float, >,
}

impl<'a> FrameSubsetter<'a>{
    pub fn new(
        array: ArrayView2<'a, float>,
        frame_col: Option<usize>,
        positions: (usize, usize),
        r_col: Option<usize>,
        ty: SubsetterType,
    ) -> FrameSubsetter<'a>{
        FrameSubsetter{
            frame_col,
            array,
            positions,
            idx: 0,
            cur_frame: 0.0,
            r_col,
            ty,
        }
    }
    
    pub fn into_linking_iter(self)
    -> impl Iterator<Item = crate::error::Result<(Option<usize>, Vec<[float; 2]>)>> + 'a
    {
        let idk = self.map(|subsetter_element|{
            let out = subsetter_element.map(|(frame, subset_outputter)| {
                let vec = match subset_outputter{
                    SubsetterOutput::Linking(vec) => vec,
                    _ => unreachable!()
                };
                (frame, vec)
            });
            out
        });
        idk
    }
}

impl<'a> Iterator for FrameSubsetter<'a>{
    // type Item = ArrayView2<'a, float>;
    type Item = crate::error::Result<(Option<usize>, SubsetterOutput)>; 
    fn next(&mut self) -> Option<Self::Item> {
        // let prev_idx = self.idx;
        let mut output = match self.ty{
            SubsetterType::Linking => SubsetterOutput::Linking(Vec::new()),
            SubsetterType::Characterization => SubsetterOutput::Characterization(Vec::new()),
            SubsetterType::Agnostic => SubsetterOutput::Agnostic(Array::zeros((0, self.array.shape()[1]))),
        };
        // let mut output = Vec::new();
        match self.frame_col{
            Some(frame_col) => {
                loop{
                    let frame = self.array.get(ndarray::Ix2(self.idx, frame_col));
                    match frame{
                        Some(frame) => {
                            let frame = *frame;
                            if frame > self.cur_frame{
                                let out = (Some(self.cur_frame as usize), output);
                                self.cur_frame = frame;
                                // let result = Some(self.array.slice(s![prev_idx..self.idx, ..]));
                                return Some(Ok(out));
                            }
                            if frame < self.cur_frame{
                                return Some(Err(crate::error::Error::NonSortedCharacterization))
                            }
                        },
                        None => {
                            if output.len() > 0{
                                return Some(Ok((Some(self.cur_frame as usize), output)));
                            }
                            return None;
                        }
                    }
                    let x = self.array[[self.idx, self.positions.0]];
                    let y = self.array[[self.idx, self.positions.1]];
                    match output{
                        SubsetterOutput::Linking(ref mut vec) => vec.push([x, y]),
                        SubsetterOutput::Characterization(ref mut vec) => {
                            let r_row = match self.r_col{
                                Some(r_col) => {
                                    ResultRow{
                                        x,
                                        y, 
                                        r: self.array[[self.idx, r_col]],
                                        ..Default::default()
                                    }
                                },
                                None => {
                                    ResultRow{
                                        x,
                                        y, 
                                        ..Default::default()
                                    }
                                },
                            };
                            vec.push(r_row);
                        },
                        SubsetterOutput::Agnostic(ref mut array) => {
                            array.append(Axis(0), self.array.slice(s![self.idx..self.idx+1, ..])).unwrap();
                        },
                    }
                    self.idx += 1;
                }
            },
            None => {
                match output{
                    SubsetterOutput::Linking(ref mut vec) => {
                        for row in self.array.rows(){
                            vec.push([row[self.positions.0], row[self.positions.1]]);
                        }
                    },
                    SubsetterOutput::Characterization(ref mut vec) => {
                        match self.r_col{
                            Some(r_col) => {
                                for row in self.array.rows(){
                                    let r_row = ResultRow{
                                        x: row[self.positions.0],
                                        y: row[self.positions.1],
                                        r: row[r_col],
                                        ..Default::default()
                                    };
                                    vec.push(r_row);
                                }
                            },
                            None => {
                                for row in self.array.rows(){
                                    let r_row = ResultRow{
                                        x: row[self.positions.0],
                                        y: row[self.positions.1],
                                        ..Default::default()
                                    };
                                    vec.push(r_row);
                                }
                            }
                        }
                    },
                    SubsetterOutput::Agnostic(ref mut array) => {
                        *array = self.array.into_owned()
                    }
                }
                return Some(Ok((None, output)))
            }
        }
    }
}


pub trait ReturnDistance2<T, N, R>{
    fn within_radius_rd2(&self, query: &impl KdPoint<Scalar = T::Scalar, Dim = N>,
        radius: T::Scalar,
    ) -> Vec<(&R, T::Scalar)>
    where
        T: KdPoint<Dim = N>;
    
}

impl<'a, T: KdPoint, N: typenum::marker_traits::Unsigned> ReturnDistance2<T, N, usize> for KdIndexTree<'a, T>{
    fn within_radius_rd2(&self, query: &impl KdPoint<Scalar = T::Scalar, Dim = N>,
        radius: T::Scalar,
    ) -> Vec<(&usize, <T as KdPoint>::Scalar)>
    where
        T: KdPoint<Dim = N>,
        {
            let r2 = radius * radius;
            let results = self.within_by_cmp(|item, k| {
                let coord = item.at(k);
                if coord < query.at(k) - radius {
                    Ordering::Less
                } else if coord > query.at(k) + radius {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            });
            let results = results.into_iter().filter_map(|item| {
                let mut distance = <T::Scalar as num_traits::Zero>::zero();
                for k in 0..N::to_usize() {
                    let diff = self.item(*item).at(k) - query.at(k);
                    distance += diff * diff;
                }
                match distance < r2{
                    true => Some((item, distance)),
                    false => None,
                }
            }).collect::<Vec<_>>();
            // todo!()
            results
        }
}

impl<T: KdPoint, N: typenum::marker_traits::Unsigned> ReturnDistance2<T, N, T> for KdTree<T>{
    fn within_radius_rd2(&self, query: &impl KdPoint<Scalar = T::Scalar, Dim = N>,
        radius: T::Scalar,
    ) -> Vec<(&T, T::Scalar)>
    where
        T: KdPoint<Dim = N>,
        {
            let r2 = radius * radius;
            let results = self.within_by_cmp(|item, k| {
                let coord = item.at(k);
                if coord < query.at(k) - radius {
                    Ordering::Less
                } else if coord > query.at(k) + radius {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            });
            let results = results.into_iter().filter_map(|item| {
                let mut distance = <T::Scalar as num_traits::Zero>::zero();
                for k in 0..N::to_usize() {
                    let diff = item.at(k) - query.at(k);
                    distance += diff * diff;
                }
                match distance < r2{
                    true => Some((item, distance)),
                    false => None,
                }
            }).collect::<Vec<_>>();
            // todo!()
            results
            // todo!()
        }
}

#[derive(Debug)]
pub struct ReuseVecofVec<T>{
    pub vecs: Vec<Vec<T>>,
    pub len: usize,
}

impl<T> ReuseVecofVec<T>{
    pub fn new() -> ReuseVecofVec<T>{
        ReuseVecofVec{
            vecs: Vec::new(),
            len: 0,
        }
    }
    
    pub fn set_size(&mut self, size: usize){
        if size > self.len{
            for _ in self.len..size{
                self.vecs.push(Vec::new());
            }
        }
        self.len = size;
    }
    pub fn with_size(size: usize) -> ReuseVecofVec<T>{
        let mut output = ReuseVecofVec::new();
        output.set_size(size);
        output
    }
    
    pub fn push_to(&mut self, idx: usize, item: T){
        if idx >= self.len{
            panic!("Index out of bounds");
        }
        self.vecs[idx].push(item);
    }

    pub fn iter(&mut self) -> std::iter::Take<std::slice::Iter<Vec<T>>>{
        self.vecs.iter().take(self.len)
    }
    pub fn iter_mut(&mut self) -> std::iter::Take<std::slice::IterMut<Vec<T>>>{
        self.vecs.iter_mut().take(self.len)
    }

    pub fn clear(&mut self){
        for vec in self.iter_mut(){
            vec.clear();
        }
    }
}

impl<T> std::ops::Index<usize> for ReuseVecofVec<T>{
    type Output = Vec<T>;
    fn index(&self, idx: usize) -> &Self::Output{
        &self.vecs[idx]
    }
}

#[derive(Debug)]
pub struct Linker{
    pub search_range: float,
    pub memory: usize,
    pub prev: Vec<([float; 2], usize)>,
    pub prev_N: usize,
    pub src_to_dest: ReuseVecofVec<(usize, float)>,
    pub dest_to_src: ReuseVecofVec<(usize, float)>,
    pub visited: [Vec<bool>; 2],
    pub memory_vec: Option<VecDeque<Vec<([float; 2],usize)>>>,
    pub frame_idx: usize,
    pub part_idx: usize,
    pub starting_frame: Vec<DurationBookkeep>,
}

#[derive(Debug)]
pub struct DurationBookkeep{
    pub start: usize,
    pub duration: usize,
}

impl DurationBookkeep{
    fn new(start: usize) -> Self{
        Self {
            start,
            duration: 0,
        }
    }
}

impl Linker{
    pub fn new(search_range: float, memory: usize) -> Linker{
        let mem_init: Option<VecDeque<Vec<([f32; 2],usize)>>> = if memory > 0{
            let mut init = VecDeque::new();
            (0..memory).for_each(|_| init.push_back(Vec::new()));
            Some(init)
        } else{
            None
        };

        Linker{
            search_range,
            memory,
            prev: Vec::new(),
            prev_N: 0,
            src_to_dest: ReuseVecofVec::new(),
            dest_to_src: ReuseVecofVec::new(),
            visited: [Vec::new(), Vec::new()],
            memory_vec: mem_init,
            frame_idx: 0,
            part_idx: 0,
            starting_frame: Vec::new(),
        }
    }

    pub fn reset(&mut self){
        *self = Self::new(self.search_range, self.memory)
    }

    pub fn connect<T: KdPoint<Scalar = float, Dim = U2> + std::fmt::Debug>(&mut self, frame1: &[T], frame2: &[T]) -> (Vec<usize>, Vec<usize>){

        let prev: Vec<_>= frame1.iter().map(|ele| {
            let out = ([ele.at(0), ele.at(1)], self.part_idx);
            self.part_idx += 1;
            out
        }).collect();
        let (result, _memory) = link(
            &prev,
            frame2,
            &mut self.src_to_dest,
            &mut self.dest_to_src,
            &mut self.visited,
            None,
            self.search_range,
            &mut self.part_idx,
        );
        let first = prev.into_iter().map(|(_, idx)| idx).collect();
        (first, result)
    }

    pub fn advance<T: KdPoint<Scalar = float, Dim = U2> + std::fmt::Debug>(&mut self, frame: &[T]) 
            -> Vec<usize> {
        let N = frame.len();
        let memory_start_idx = match self.memory_vec{
            Some(ref memvec) =>{
                let mut memset: HashSet<_> = HashSet::from_iter(self.prev.iter().map(|ele| ele.1));
                for entry in memvec.iter().flatten(){
                    match memset.contains(&(*entry).1){
                        true => {},
                        false => {
                            self.prev.push(*entry);
                            memset.insert(entry.1);
                        },
                    }
                }
                Some(self.prev_N)
            }
            None => None,
        };
        let prev_part_idx = self.part_idx;
        
        let (result, memory) =
            link(
                &self.prev,
                &frame,
                &mut self.src_to_dest,
                &mut self.dest_to_src,
                &mut self.visited,
                memory_start_idx,
                self.search_range,
                &mut self.part_idx,
            );

        for _ in 0..(self.part_idx - prev_part_idx){
            self.starting_frame.push(DurationBookkeep::new(self.frame_idx));
        }
        
        self.prev = frame.iter().zip(result.iter()).map(|(a, b)| ([a.at(0), a.at(1)], *b)).collect::<Vec<_>>();
        match self.memory_vec{
            Some(ref mut memvec) => {
                memvec.push_back(memory);
                let leaving_pool = memvec.pop_front().unwrap();
                for entry in leaving_pool{
                    let start_frame = self.starting_frame.get_mut(entry.1).unwrap();
                    start_frame.duration = (self.frame_idx - start_frame.start) - memvec.len();
                }
            }
            None => {
                let leaving_pool = memory;
                for entry in leaving_pool{
                    let start_frame = self.starting_frame.get_mut(entry.1).unwrap();
                    start_frame.duration = self.frame_idx - start_frame.start;
                }
            },
        }
        self.prev_N = N;
        self.frame_idx += 1;
        result
    }
    
    pub fn finish(mut self) -> Vec<DurationBookkeep> {
        if let Some(memvec) = self.memory_vec{
            for (i, vec) in memvec.iter().enumerate(){
                for entry in vec{
                    let start_frame = self.starting_frame.get_mut(entry.1).unwrap();
                    start_frame.duration = (self.frame_idx - start_frame.start) - (memvec.len() - i);
                }
            }
        }
        for entry in self.prev{
            let start_frame = self.starting_frame.get_mut(entry.1).unwrap();
            start_frame.duration = self.frame_idx - start_frame.start;
        }
        self.starting_frame
    }

}


pub fn link<T: KdPoint<Scalar = float, Dim = U2> + std::fmt::Debug>(
    src: &Vec<([float; 2], usize)>,
    dest: &[T],
    src_to_dest: &mut ReuseVecofVec<(usize, float)>,
    dest_to_src: &mut ReuseVecofVec<(usize, float)>,
    visited: &mut [Vec<bool>; 2],
    memory_start_idx: Option<usize>,
    radius: float,
    counter: &mut usize,
    ) -> (Vec<usize>, Vec<([float; 2], usize)>){
    

    let tree = kd_tree::KdIndexTree::build_by_ordered_float(dest);
    let dest_points_near_source = src.iter().map(|point| {
            tree.within_radius_rd2(point, radius)
    });
        
    src_to_dest.set_size(src.len());
    dest_to_src.set_size(dest.len());
    src_to_dest.clear();
    dest_to_src.clear();

    
    for (source, dest_points) in dest_points_near_source.enumerate(){
        for dest in dest_points{
            src_to_dest.push_to(source, (*dest.0, dest.1));
            dest_to_src.push_to(*dest.0, (source, dest.1));
        }
    }
    
    #[inline]
    fn recurse(
        src_to_dest: &ReuseVecofVec<(usize, float)>,
        dest_to_src: &ReuseVecofVec<(usize, float)>,
        node: (usize, usize),
        visited: &mut [Vec<bool>; 2],
        path: &mut [Vec<usize>; 2]) -> (){
        if visited[node.1][node.0]{
            return;
        }
        visited[node.1][node.0] = true;
        path[node.1].push(node.0);

        match node.1{
            0 => {
                for &dest_node in &src_to_dest[node.0]{
                    recurse(src_to_dest, dest_to_src, (dest_node.0, 1), visited, path);
                }
            },
            1 => {
                for &source_node in &dest_to_src[node.0]{
                    recurse(src_to_dest, dest_to_src, (source_node.0, 0), visited, path);
                }
            },
            _ => panic!("Invalid node type"),
        }
    }

    if src.len() > visited[0].len(){
        for _ in visited[0].len()..src.len(){
            visited[0].push(false);
        }
    }

    if dest.len() > visited[1].len(){
        for _ in visited[1].len()..dest.len(){
            visited[1].push(false);
        }
    }

    for val in visited[0].iter_mut(){
        *val = false;
    }
    
    for val in visited[1].iter_mut(){
        *val = false;
    }
    
    let mut paths = Vec::new();
    let mut output = vec![None; dest.len()];
    for i in 0..dest.len(){
        let mut path = [Vec::new(), Vec::new()];
        recurse(&src_to_dest, &dest_to_src, (i, 1), visited, &mut path);
        if path[1].len() == 1{
            if path[0].len() == 0{

            } else if path[0].len() == 1{
                output[i] = Some(src[path[0][0]].1);
            }
            else{
                paths.push(path);
            }
        }
        else if path[1].len() > 0{
            paths.push(path);
        }
    }
    #[inline]
    fn recurse2(progress: usize,
        path: &[Vec<usize>; 2],
        score: &mut float,
        best: &mut float,
        used: &mut HashMap<usize, usize>,
        output: &mut Vec<Option<usize>>,
        dest_to_src: &ReuseVecofVec<(usize, float)>,
        default_score: float,
        sources: &Vec<([float; 2], usize)>,
        nulls: &mut HashSet<usize>,
        ){
        let current_dest = path[1][progress];
        let dest_to_src_with_null = dest_to_src[current_dest].iter()
            .map(|ele| (Some(sources[ele.0].1), ele.1)).chain(std::iter::once((None, default_score)));
        for src in dest_to_src_with_null{
            if *score > *best{
                return;
            }
            *score += src.1;
            let src_ident = src.0;
            match src_ident{
                Some(src_ident) => {
                    if used.contains_key(&src_ident){
                        *score -= src.1;
                        continue;
                    }
                    used.insert(src_ident, current_dest);
                },
                None => {
                    nulls.insert(current_dest);
                },
            }

            if progress == path[1].len() - 1{
                if *score < *best{
                    *best = *score;
                    for (key, val) in used.iter(){
                        output[*val] = Some(*key);
                    }
                    for val in nulls.iter(){
                        output[*val] = None;
                    }
                }
            } else {
                recurse2(progress + 1, path, score, best, used, output, dest_to_src, default_score, sources, nulls);
            }
            match src_ident{
                Some(src_ident) => {
                    used.remove(&src_ident);
                },
                None => {
                    nulls.remove(&current_dest);
                },
            }
            *score -= src.1;
        }
        ()
    }
    
    for path in paths{
        let mut used = HashMap::new();
        let mut nulls = HashSet::new();
        let score = &mut 0.0;
        let best = &mut f32::INFINITY.clone();
        recurse2(0, &path, score, best, &mut used, &mut output,
            dest_to_src, radius * radius, &src, &mut nulls)
    }

    let output = output.into_iter().map(|ele| match ele{ Some(val) => val, None => { let old = *counter; *counter += 1; old } }).collect::<Vec<_>>();

    let used_sources: HashSet<_> = HashSet::from_iter(output.iter().cloned());
    let unused_sources = match memory_start_idx{
        Some(idx) => src.iter().take(idx).filter(|ele| !used_sources.contains(&ele.1)).map(|ele| *ele).collect::<Vec<_>>(),
        None => src.iter().filter(|ele| !used_sources.contains(&ele.1)).map(|ele| *ele).collect::<Vec<_>>(),
    };

    (output, unused_sources)
}

// pub fn link_all<T>(frame_iter: T, radius: float, memory: usize) -> Vec<usize>
//     where T: Iterator<Item = Vec<([float; 2])>>{
//     let mut prev: Vec<([float; 2], usize)> = Vec::new();
//     let mut prev_N = 0;
//     let mut src_to_dest = ReuseVecofVec::new();
//     let mut dest_to_src = ReuseVecofVec::new();
//     let mut visited = [Vec::new(), Vec::new()];
//     let mut memory_vec: VecDeque<Vec<([f32; 2],usize)>> = VecDeque::new();
//     (0..memory).for_each(|_| memory_vec.push_back(Vec::new()));
//     let mut results = Vec::new();
//     let mut total_tracks = 0;

//     for frame in frame_iter{
//         let N = frame.len();
//         if memory > 0{
//             let mut memset: HashSet<_> = HashSet::from_iter(prev.iter().map(|ele| ele.1));
//             for entry in memory_vec.iter().flatten(){
//                 match memset.contains(&(*entry).1){
//                     true => {},
//                     false => {
//                         prev.push(*entry);
//                         memset.insert(entry.1);
//                     },
//                 }
//             }
//         }
//         let memory_start_idx = match memory{
//             0 => None,
//             _ => Some(prev_N)
//         };
//         let (result, memory) = 
//             link(&prev,
//                 &frame,
//                 &mut src_to_dest,
//                 &mut dest_to_src,
//                 &mut visited,
//                 memory_start_idx,
//                 radius,
//                 &mut total_tracks,
//         );
//         prev = frame.iter().zip(result.iter()).map(|(a, b)| (*a, *b)).collect::<Vec<_>>();
//         results.extend(result.into_iter());
//         memory_vec.push_front(memory);
//         memory_vec.pop_back();
//         prev_N = N;
//     }
//     results

// }


pub fn linker_all(frame_iter: impl Iterator<Item = crate::error::Result<(Option<usize>, Vec<[float; 2]>)>>, radius: float, memory: usize) -> crate::error::Result<Vec<usize>>{

    let mut linker = Linker::new(radius, memory);
    let mut results = Vec::new();
    for frame in frame_iter{
        let frame = frame?;
        let result = linker.advance(&frame.1);
        results.extend(result.into_iter());
    }
    Ok(results)
}

enum LR{
    Left,
    Right,
}

struct LRContainer{
    item: (Option<usize>, Vec<[float; 2]>),
    side: LR,
}

impl LRContainer{
    fn other(self, new_item: (Option<usize>, Vec<[float; 2]>)) -> Self{
        match self.side{
            LR::Left => LRContainer{
                item: new_item,
                side: LR::Right,
            },
            LR::Right => LRContainer{
                item: new_item,
                side: LR::Left,
            },
        }
    }

    fn cmp(&self, other: &(Option<usize>, Vec<[float; 2]>)) -> std::cmp::Ordering{
        self.item.0.unwrap().cmp(other.0.as_ref().unwrap())
    }
    
    fn into_inner(self) -> (Option<usize>, Vec<[float; 2]>){
        self.item
    }

    fn finish(self, other: (Option<usize>, Vec<[float; 2]>)) -> ((Option<usize>, Vec<[float; 2]>), (Option<usize>, Vec<[float; 2]>)){
        match self.side{
            LR::Left => {
                (self.into_inner(), other)
            },
            LR::Right => {
                (other, self.into_inner())
            },
        }
    }

    fn fill(&self, results1: &mut Vec<usize>, results2: &mut Vec<usize>, linker: &mut Linker){
        match self.side{
            LR::Left => {
                fill_base(&self.item, results1, linker);
            },
            LR::Right => {
                fill_base(&self.item, results2, linker);
            },
        }
    }

    fn fill_other(&self, other: &(Option<usize>, Vec<[f32; 2]>), results1: &mut Vec<usize>, results2: &mut Vec<usize>, linker: &mut Linker){
        match self.side{
            LR::Left => {
                fill_base(other, results2, linker);
            },
            LR::Right => {
                fill_base(other, results1, linker);
            },
        }
    }
}

fn fill_base(item: &(Option<usize>, Vec<[f32; 2]>), result: &mut Vec<usize>, linker: &mut Linker){
    for _ in 0..item.1.len(){
        result.push(linker.part_idx);
        linker.part_idx += 1;
    }
}


fn sync_iterators(
    frame_iter1: &mut impl Iterator<Item = crate::error::Result<(Option<usize>, Vec<[float; 2]>)>>,
    frame_iter2: &mut impl Iterator<Item = crate::error::Result<(Option<usize>, Vec<[float; 2]>)>>,
    to_hit: LRContainer,
    linker: &mut Linker,
    results1: &mut Vec<usize>,
    results2: &mut Vec<usize>,
    ) -> crate::error::Result<Option<((Option<usize>, Vec<[float; 2]>), (Option<usize>, Vec<[float; 2]>))>>{
    let next_item = match to_hit.side{
        LR::Left => {
            frame_iter2.next()
        },
        LR::Right => {
            frame_iter1.next()
        },
    };

    let next_item = match next_item{
        Some(inner) => inner?,
        None => {
            to_hit.fill(results1, results2, linker);
            return Ok(None)
        }
    };
    match to_hit.cmp(&next_item){
        std::cmp::Ordering::Greater => {
            to_hit.fill_other(&next_item, results1, results2, linker);
            return sync_iterators(frame_iter1, frame_iter2, to_hit, linker, results1, results2)
        },
        std::cmp::Ordering::Equal => {
            return Ok(Some(to_hit.finish(next_item)))
        },
        std::cmp::Ordering::Less => {
            to_hit.fill_other(&next_item, results1, results2, linker);
            to_hit.fill(results1, results2, linker);
            return sync_iterators(frame_iter1, frame_iter2, to_hit.other(next_item), linker, results1, results2)
        },
    }
}

pub fn connect_all(
    mut frame_iter1: impl Iterator<Item = crate::error::Result<(Option<usize>, Vec<[float; 2]>)>>,
    mut frame_iter2: impl Iterator<Item = crate::error::Result<(Option<usize>, Vec<[float; 2]>)>>,
    radius: float,
    ) -> crate::error::Result<(Vec<usize>, Vec<usize>)>{

    let mut linker = Linker::new(radius, 0);
    let mut results1 = Vec::new();
    let mut results2 = Vec::new();
    loop{
        let frame1 = frame_iter1.next();
        let frame2 = frame_iter2.next();

        match (frame1, frame2){
            (Some(Ok(frame1)), Some(Ok(frame2))) => {
                let both_frames = match frame1.0.unwrap().cmp(&frame2.0.unwrap()){
                    std::cmp::Ordering::Equal => {
                        Some((frame1, frame2))
                    },
                    std::cmp::Ordering::Greater => {
                        let to_hit = LRContainer { item: frame1, side: LR::Left };
                        fill_base(&frame2, &mut results2, &mut linker);
                        sync_iterators(&mut frame_iter1, &mut frame_iter2, to_hit, &mut linker, &mut results1, &mut results2)?
                    },
                    std::cmp::Ordering::Less => {
                        let to_hit = LRContainer { item: frame2, side: LR::Right };
                        fill_base(&frame1, &mut results1, &mut linker);
                        sync_iterators(&mut frame_iter1, &mut frame_iter2, to_hit, &mut linker, &mut results1, &mut results2)?
                    },
                };
                if let Some((frame1, frame2)) = both_frames{
                    let (result1, result2) = linker.connect(&frame1.1, &frame2.1);
                    results1.extend(result1.into_iter());
                    results2.extend(result2.into_iter());
                }
            },
            (Some(Ok(frame1)), None) => {
                fill_base(&frame1, &mut results1, &mut linker)
            },
            (None, Some(Ok(frame2))) => {
                fill_base(&frame2, &mut results2, &mut linker)
            },
            (None, None) => {
                break
            }
            (Some(Err(e)), _) | (_, Some(Err(e))) => {
                return Err(e)
            }
        }
    }
    Ok((results1, results2))
}