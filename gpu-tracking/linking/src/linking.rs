#![allow(warnings)]
use ndarray::{Array2, ArrayView2, s};
use kd_tree;
use bencher::black_box;
use std::{collections::{HashMap, HashSet, VecDeque, hash_map::Entry}, default, iter::FromIterator};

type float = f32;

pub struct FrameSubsetter<'a>{
    pub frame_col: usize,
    pub array: &'a Array2<float>,
    idx: usize,
    cur_frame: float,
    // iter: ndarray::iter::Iter<'a, float, >,
}

impl FrameSubsetter<'_>{
    pub fn new(array: &Array2<float>, frame_col: usize) -> FrameSubsetter{
        FrameSubsetter{
            frame_col,
            array,
            idx: 0,
            cur_frame: 0.0,
        }
    }
}

impl<'a> Iterator for FrameSubsetter<'a>{
    // type Item = ArrayView2<'a, float>;
    type Item = Vec<([float; 2], usize)>; 
    fn next(&mut self) -> Option<Self::Item> {
        let prev_idx = self.idx;
        let mut output = Vec::new();
        loop{
            let frame = self.array.get(ndarray::Ix2(self.idx, self.frame_col));
            match frame{
                Some(frame) => {
                    let frame = *frame;
                    if frame != self.cur_frame{
                        self.cur_frame = frame;
                        // let result = Some(self.array.slice(s![prev_idx..self.idx, ..]));
                        return Some(output);
                    }
                },
                None => {
                    if output.len() > 0{
                        return Some(output);
                    }
                    return None;
                }
            }
            output.push(([self.array[[self.idx, 0]], self.array[[self.idx, 1]]], self.idx));
            self.idx += 1;
        }
    }
}

use kd_tree::{KdPoint, KdTree};
use std::cmp::Ordering;
use typenum;
use num_traits;

trait ReturnDistance<T, N>{
    fn within_radius_rd(&self, query: &impl KdPoint<Scalar = T::Scalar, Dim = N>,
        radius: T::Scalar,
    ) -> Vec<(&T, T::Scalar)>
    where
        T: KdPoint<Dim = N>;
    
}

impl<T: KdPoint, N: typenum::marker_traits::Unsigned> ReturnDistance<T, N> for KdTree<T>{
    fn within_radius_rd(&self, query: &impl KdPoint<Scalar = T::Scalar, Dim = N>,
        radius: T::Scalar,
    ) -> Vec<(&T, T::Scalar)>
    where
        T: KdPoint<Dim = N>,
        {
            let r2 = radius * radius;
            let mut results = self.within_by_cmp(|item, k| {
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
        }
}

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


pub fn link(
    src: Vec<([float; 2], usize)>,
    dest: Vec<([float; 2], usize)>,
    src_to_dest: &mut ReuseVecofVec<(usize, float)>,
    dest_to_src: &mut ReuseVecofVec<(usize, float)>,
    visited: &mut [Vec<bool>; 2],
    memory_start_idx: Option<usize>,
    radius: float,
    ) -> (Vec<([float; 2], usize)>, Vec<([float; 2], usize)>){
    
    let dest_0_idx = dest.iter().enumerate().map(|(i, ele)| (ele.0, i)).collect::<Vec<_>>();

    let tree = kd_tree::KdTree::build_by_ordered_float(dest_0_idx);
    let dest_points_near_source = src.iter().map(|point| tree.within_radius_rd(point, radius));
    // let mut source: Vec<Vec<(usize, float)>> = vec![Vec::new(); src.len()];
    // let mut dest: Vec<Vec<(usize, float)>> = vec![Vec::new(); tree.len()];
    src_to_dest.set_size(src.len());
    dest_to_src.set_size(tree.len());
    src_to_dest.clear();
    dest_to_src.clear();

    // let dest: Vec<Vec<usize>> = Vec::new();

    for (source, dest_points) in dest_points_near_source.enumerate(){
        for dest in dest_points{
            src_to_dest.push_to(source, (dest.0.1, dest.1));
            dest_to_src.push_to(dest.0.1, (source, dest.1));
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

    // let mut visited = [vec![false; src.len()], vec![false; tree.len()]];
    // let mut visited = [vec![false; 2]; std::cmp::max(source.len(), dest.len())];
    if src.len() > visited[0].len(){
        for _ in visited[0].len()..src.len(){
            visited[0].push(false);
        }
    }

    if tree.len() > visited[1].len(){
        for _ in visited[1].len()..tree.len(){
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
    let mut output = vec![0usize; tree.len()];
    // let mut debug = 637usize;
    for i in 0..tree.len(){
        // if dest[i].1 == debug{
        //     debug = i;
        // }
        let mut path = [Vec::new(), Vec::new()];
        recurse(&src_to_dest, &dest_to_src, (i, 1), visited, &mut path);
        if path[1].len() == 1{
            if path[0].len() == 0{
                output[i] = dest[path[1][0]].1;
                // continue;
            } else if path[0].len() == 1{
                output[i] = src[path[0][0]].1;
                // continue;
            }
            else{
                paths.push(path);
            }
        }
        else if path[1].len() > 0{
            paths.push(path);
        }
    }

    fn recurse2(progress: usize,
        path: &[Vec<usize>; 2],
        score: &mut float,
        best: &mut float,
        used: &mut HashMap<usize, usize>,
        output: &mut Vec<usize>,
        dest_to_src: &ReuseVecofVec<(usize, float)>,
        default_score: float,
        sources: &Vec<([float; 2], usize)>,
        destinations: &Vec<([float; 2], usize)>,
        ){
        let current_dest = path[1][progress];
        let dest_to_src_with_null = dest_to_src[current_dest].iter()
            .map(|ele| (sources[ele.0].1, ele.1)).chain(std::iter::once((destinations[current_dest].1, default_score)));
        // let mut score = score_ref;
        for src in dest_to_src_with_null{
            if *score > *best{
                // *score -= src.1;
                return;
            }
            *score += src.1;
            let src_ident = src.0;
            if used.contains_key(&src_ident){
                *score -= src.1;
                continue;
            }
            used.insert(src_ident, current_dest);
            if progress == path[1].len() - 1{
                if *score < *best{
                    *best = *score;
                    for (key, val) in used.iter(){
                        // assert_eq!(output[*val], -1);
                        output[*val] = *key;
                    }
                }
            } else {
                recurse2(progress + 1, path, score, best, used, output, dest_to_src, default_score, sources, destinations);
            }
            used.remove(&src_ident);
            *score -= src.1;
        }
        ()
    }
    
    // let score_inp = 0.0;
    // let score_inp = &mut 0.0;
    for path in paths{
        // for val in &path[1]{
        //     if *val == debug{
        //         println!("here");
        //     }
        // }
        let mut used = HashMap::new();
        let score = &mut 0.0;
        let best = &mut f32::MAX;
        recurse2(0, &path, score, best, &mut used, &mut output,
            dest_to_src, radius * radius, &src, &dest)
    }

    // let output: Vec<_> = output.iter().map(|ele| *ele as usize).collect();
    let unused_sources = match memory_start_idx{
        Some(idx) => {
            let used_sources: HashSet<_> = HashSet::from_iter(output.iter().cloned());
            src.iter().take(idx).filter(|ele| !used_sources.contains(&ele.1)).map(|ele| *ele).collect::<Vec<_>>()
        },
        None => Vec::new(),
    };

    let output = output.iter().zip(dest.iter())
    .map(|(a, b)| (b.0, *a as usize)).collect::<Vec<_>>();
    
    (output, unused_sources)
}
pub fn link_all<T>(mut frame_iter: T, radius: float, memory: usize) -> Vec<(usize, [float; 2], i32)>
    where T: Iterator<Item = Vec<([float; 2], usize)>>{
    let mut frame_iter = frame_iter.enumerate();
    let (i, mut prev) = frame_iter.next().unwrap();
    let mut prev_N = prev.len();
    let mut src_to_dest = ReuseVecofVec::new();
    let mut dest_to_src = ReuseVecofVec::new();
    let mut visited = [Vec::new(), Vec::new()];
    let mut memory_vec: VecDeque<Vec<([f32; 2],usize)>> = VecDeque::new();
    (0..memory).for_each(|_| memory_vec.push_back(Vec::new()));
    let mut results = Vec::new();
    results.extend(prev.iter().map(|(a, b)| (i, *a, *b)));

    for (i, frame) in frame_iter{
        // let new_prev = frame.clone();
        let N = frame.len();
        if memory > 0{
            let mut memset: HashSet<_> = HashSet::from_iter(prev.iter().map(|ele| ele.1));
            for entry in memory_vec.iter().flatten(){
                match memset.contains(&(*entry).1){
                    true => {},
                    false => {
                        prev.push(*entry);
                        memset.insert(entry.1);
                    },
                }
                // prev.extend(mem)
            }
        }
        let memory_start_idx = match memory{
            0 => None,
            _ => Some(prev_N)
        };
        // dbg!(prev.len());
        let (result, memory) = 
            link(prev,
                frame,
                &mut src_to_dest,
                &mut dest_to_src,
                &mut visited,
                memory_start_idx,
                radius,
        );
        prev = result.clone();
        results.extend(result.into_iter().map(|(a, b)| (i, a, b)));
        memory_vec.push_front(memory);
        memory_vec.pop_back();
        prev_N = N;
        // dbg!(i);
        // i += 1;
        // break;
    }
    // dbg!(results);
    let mut all_particles = HashMap::new();
    let mut counter = 0;
    let results = results.into_iter()
    .map(|(frame_idx, coords, part_id)| {
        match all_particles.entry(part_id){
            Entry::Occupied(entry) => {
                (frame_idx, coords, *entry.get())
            },
            Entry::Vacant(entry) => {
                entry.insert(counter);
                counter += 1;
                (frame_idx, coords, counter - 1)
            }
        }
    }).collect::<Vec<_>>();
    results

}