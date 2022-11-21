#![allow(warnings)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod decoderiter;
pub mod execute_gpu;
pub mod kernels;
pub mod into_slice;
pub mod slice_wrapper;
pub type my_dtype = f32;
pub mod gpu_setup;
pub mod linking;
// #[cfg(feature = "python")]
// pub mod python_bindings;
use linking::FrameSubsetter;
use ndarray::Array2;
use crate::{execute_gpu::{execute_ndarray}, decoderiter::{MinimalETSParser}, gpu_setup::{TrackingParams, TrackpyParams, LogParams}};
use ndarray::Array;
use std::{fs::File, collections::HashMap};

#[cfg(feature = "python")]
use pyo3::{prelude::*, types::PyDict};
#[cfg(feature = "python")]
use numpy::{self, PyArray2, PyReadonlyArray3, PyReadonlyArray2, IntoPyArray, PyArray1,};


macro_rules! not_implemented {
    ($name:ident) => {
        if $name.is_some(){
            panic!("{} is not implemented", stringify!($name));
        }
    };
    
    ($name:ident, $($names:ident), +) => {
        not_implemented!($name);
        not_implemented!($($names), +);
    };
}

macro_rules! make_args {
    (
    // $(#[$m:meta])*
    $vis:vis fn $name:ident <$($generics:tt),*> ({$($preargs:tt)*}, {$($postargs:tt)*}) -> $outtype:ty => $params:ident $body:block
    ) => {
        // $(#[$m])*
        #[pyfunction]
        $vis fn $name<$($generics),*>($($preargs)*
            diameter: u32,
            minmass: Option<my_dtype>,
            maxsize: Option<my_dtype>,
            separation: Option<u32>,
            noise_size: Option<f32>,
            smoothing_size: Option<u32>,
            threshold: Option<my_dtype>,
            invert: Option<bool>,
            percentile: Option<my_dtype>,
            topn: Option<u32>,
            preprocess: Option<bool>,
            max_iterations: Option<u32>,
            characterize: Option<bool>,
            filter_close: Option<bool>,
            search_range: Option<my_dtype>,
            memory: Option<usize>,
            // cpu_processed: Option<bool>,
            sig_radius: Option<my_dtype>,
            bg_radius: Option<my_dtype>,
            gap_radius: Option<my_dtype>,
            varcheck: Option<my_dtype>,
            $($postargs)*
        ) -> $outtype {
            not_implemented!(maxsize, threshold, invert, percentile,
                topn, preprocess);
            
            
            let minmass = minmass.unwrap_or(0.);
            let maxsize = maxsize.unwrap_or(f32::INFINITY);
            let separation = separation.unwrap_or(diameter + 1);
            let noise_size = noise_size.unwrap_or(1.);
            let smoothing_size = smoothing_size.unwrap_or(diameter);
            let threshold = threshold.unwrap_or(1./255.);
            let invert = invert.unwrap_or(false);
            let percentile = percentile.unwrap_or(64.);
            let topn = topn.unwrap_or(u32::MAX);
            let preprocess = preprocess.unwrap_or(true);
            let max_iterations = max_iterations.unwrap_or(10);
            let characterize = characterize.unwrap_or(false);
            let filter_close = filter_close.unwrap_or(true);
            // let cpu_processed = cpu_processed.unwrap_or(false);
            let gap_radius = bg_radius.map(|_| gap_radius.unwrap_or(0.));

            let style = gpu_setup::Style::Trackpy(
                TrackpyParams{}
            );
    
            // neither search_range nor memory are unwrapped as linking is optional on the Rust side.
    
            let $params = TrackingParams {
                diameter,
                minmass,
                maxsize,
                separation,
                noise_size,
                smoothing_size,
                threshold,
                invert,
                percentile,
                topn,
                preprocess,
                max_iterations,
                characterize,
                filter_close,
                search_range,
                memory,
                // cpu_processed,
                sig_radius,
                bg_radius,
                gap_radius,
                varcheck,
                style
            };
            $body
        }
    };
}

// #[cfg(feature = "python")]


#[cfg(feature = "python")]
make_args!(
    fn batch_rust<'py>(
    {
        py: Python<'py>,
        pyarr: PyReadonlyArray3<my_dtype>,
    },
    {
        points_to_characterize: Option<PyReadonlyArray2<my_dtype>>,
    }
    ) -> (&'py PyArray2<my_dtype>, Py<PyAny>) => params{
        let array = pyarr.as_array();
        let points_rust_array = points_to_characterize.as_ref().map(|arr| arr.as_array());
        let (res, columns) = execute_ndarray(&array, params, false, 0, points_rust_array.as_ref());
        (res.into_pyarray(py), columns.into_py(py))
    }
);

#[cfg(feature = "python")]
make_args!(
    fn batch_file_rust<'py>(
        {
            py: Python<'py>,
            filename: String,
        },
        {
            channel: Option<usize>,
            points_to_characterize: Option<PyReadonlyArray2<my_dtype>>,
        }
        ) -> (&'py PyArray2<my_dtype>, Py<PyAny>) => params {

        let points_rust_array = points_to_characterize.as_ref().map(|arr| arr.as_array());
        let mut pos_iter = points_rust_array.as_ref().map(|points| 
            FrameSubsetter::new(points, 0, (1, 2)));

        let (res, columns) = execute_gpu::execute_file(
            &filename, channel, params, false, 0, pos_iter);
        (res.into_pyarray(py), columns.into_py(py))
    }
);



#[cfg(feature = "python")]
#[pymodule]
fn gpu_tracking(_py: Python, m: &PyModule) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(batch_rust, m)?)?;

    m.add_function(wrap_pyfunction!(batch_file_rust, m)?)?;

    // fn batch_py<'py>(
    //     py: Python<'py>,
    //     pyarr: PyReadonlyArray3<my_dtype>,
    //     diameter: u32,
    //     minmass: Option<my_dtype>,
    //     maxsize: Option<my_dtype>,
    //     separation: Option<u32>,
    //     noise_size: Option<f32>,
    //     smoothing_size: Option<u32>,
    //     threshold: Option<my_dtype>,
    //     invert: Option<bool>,
    //     percentile: Option<my_dtype>,
    //     topn: Option<u32>,
    //     preprocess: Option<bool>,
    //     max_iterations: Option<u32>,
    //     characterize: Option<bool>,
    //     filter_close: Option<bool>,
    //     search_range: Option<my_dtype>,
    //     memory: Option<usize>,
    //     // cpu_processed: Option<bool>,
    //     sig_radius: Option<my_dtype>,
    //     bg_radius: Option<my_dtype>,
    //     gap_radius: Option<my_dtype>,
    //     points_to_characterize: Option<PyReadonlyArray2<my_dtype>>,
    //     ) -> (&'py PyArray2<my_dtype>, Py<PyAny>) {
        
    //     not_implemented!(maxsize, threshold, invert, percentile,
    //         topn, preprocess);
        
        
    //     let minmass = minmass.unwrap_or(0.);
    //     let maxsize = maxsize.unwrap_or(f32::INFINITY);
    //     let separation = separation.unwrap_or(diameter + 1);
    //     let noise_size = noise_size.unwrap_or(1.);
    //     let smoothing_size = smoothing_size.unwrap_or(diameter);
    //     let threshold = threshold.unwrap_or(1./255.);
    //     let invert = invert.unwrap_or(false);
    //     let percentile = percentile.unwrap_or(64.);
    //     let topn = topn.unwrap_or(u32::MAX);
    //     let preprocess = preprocess.unwrap_or(true);
    //     let max_iterations = max_iterations.unwrap_or(10);
    //     let characterize = characterize.unwrap_or(false);
    //     let filter_close = filter_close.unwrap_or(true);
    //     // let cpu_processed = cpu_processed.unwrap_or(false);
    //     let gap_radius = bg_radius.map(|_| gap_radius.unwrap_or(0.));

    //     // neither search_range nor memory are unwrapped as linking is optional on the Rust side.

    //     let params = TrackingParams {
    //         diameter,
    //         minmass,
    //         maxsize,
    //         separation,
    //         noise_size,
    //         smoothing_size,
    //         threshold,
    //         invert,
    //         percentile,
    //         topn,
    //         preprocess,
    //         max_iterations,
    //         characterize,
    //         filter_close,
    //         search_range,
    //         memory,
    //         // cpu_processed,
    //         sig_radius,
    //         bg_radius,
    //         gap_radius,
    //     }; 
        
    //     let array = pyarr.as_array();
    //     let points_rust_array = points_to_characterize.as_ref().map(|arr| arr.as_array());
    //     let (res, columns) = execute_ndarray(&array, params, false, 0, points_rust_array.as_ref());
    //     (res.into_pyarray(py), columns.into_py(py))
    // }

    // #[pyfn(m)]
    // #[pyo3(name = "batch_file_rust")]
    // fn batch_by_filename<'py>(
    //     py: Python<'py>,
    //     filename: String,
    //     diameter: u32,
    //     channel: Option<usize>,
    //     minmass: Option<my_dtype>,
    //     maxsize: Option<my_dtype>,
    //     separation: Option<u32>,
    //     noise_size: Option<f32>,
    //     smoothing_size: Option<u32>,
    //     threshold: Option<my_dtype>,
    //     invert: Option<bool>,
    //     percentile: Option<my_dtype>,
    //     topn: Option<u32>,
    //     preprocess: Option<bool>,
    //     max_iterations: Option<u32>,
    //     characterize: Option<bool>,
    //     filter_close: Option<bool>,
    //     search_range: Option<my_dtype>,
    //     memory: Option<usize>,
    //     // cpu_processed: Option<bool>,
    //     sig_radius: Option<my_dtype>,
    //     bg_radius: Option<my_dtype>,
    //     gap_radius: Option<my_dtype>,
    //     points_to_characterize: Option<PyReadonlyArray2<my_dtype>>,
    //     ) -> (&'py PyArray2<my_dtype>, Py<PyAny>) {
        
    //     not_implemented!(maxsize, threshold, invert, percentile,
    //         topn, preprocess);
        
        
    //     let minmass = minmass.unwrap_or(0.);
    //     let maxsize = maxsize.unwrap_or(f32::INFINITY);
    //     let separation = separation.unwrap_or(diameter + 1);
    //     let noise_size = noise_size.unwrap_or(1.);
    //     let smoothing_size = smoothing_size.unwrap_or(diameter);
    //     let threshold = threshold.unwrap_or(1./255.);
    //     let invert = invert.unwrap_or(false);
    //     let percentile = percentile.unwrap_or(64.);
    //     let topn = topn.unwrap_or(u32::MAX);
    //     let preprocess = preprocess.unwrap_or(true);
    //     let max_iterations = max_iterations.unwrap_or(10);
    //     let characterize = characterize.unwrap_or(false);
    //     let filter_close = filter_close.unwrap_or(true);
    //     // let cpu_processed = cpu_processed.unwrap_or(false);
    //     let gap_radius = bg_radius.map(|_| gap_radius.unwrap_or(0.));

    //     // neither search_range nor memory are unwrapped as linking is optional on the Rust side.

    //     let params = TrackingParams {
    //         diameter,
    //         minmass,
    //         maxsize,
    //         separation,
    //         noise_size,
    //         smoothing_size,
    //         threshold,
    //         invert,
    //         percentile,
    //         topn,
    //         preprocess,
    //         max_iterations,
    //         characterize,
    //         filter_close,
    //         search_range,
    //         memory,
    //         // cpu_processed,
    //         sig_radius,
    //         bg_radius,
    //         gap_radius,
    //     };

    //     let points_rust_array = points_to_characterize.as_ref().map(|arr| arr.as_array());
    //     let mut pos_iter = points_rust_array.as_ref().map(|points| 
    //         FrameSubsetter::new(points, 0, (1, 2)));

    //     let (res, columns) = execute_gpu::execute_file(
    //         &filename, channel, params, false, 0, pos_iter);
    //     (res.into_pyarray(py), columns.into_py(py))
    // }


    #[pyfn(m)]
    #[pyo3(name = "link_rust")]
    fn link_py<'py>(py: Python<'py>, pyarr: PyReadonlyArray2<my_dtype>,
        search_range: my_dtype,
        memory: Option<usize>) -> &'py PyArray1<usize> {
        let memory = memory.unwrap_or(0);
        let array = pyarr.as_array();
        let frame_iter = linking::FrameSubsetter::new(&array, 0, (1, 2));
        let res = linking::linker_all(frame_iter, search_range, memory);
        res.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "parse_ets")]
    fn parse_ets<'py>(py: Python<'py>, path: &str) -> &'py PyDict{
        // HashMap<usize, &'py PyArray3<u16>>
        let mut file = File::open(path).unwrap();
        let parser = MinimalETSParser::new(&mut file).unwrap();
        // let mut n_frames = vec![0];
        let output = PyDict::new(py);
        for channel in parser.offsets.keys(){
            let iter = parser.iterate_channel(file.try_clone().unwrap(), *channel);
            let n_frames = iter.len();
            let mut vec = Vec::with_capacity(n_frames * parser.dims.iter().product::<usize>());
            vec.extend(iter.flatten().flatten());
            let array = Array::from_shape_vec((n_frames, parser.dims[1], parser.dims[0]), vec).unwrap();
            let array = array.into_pyarray(py);
            output.set_item(*channel, array).unwrap();
        }
        output
    }

    Ok(())
}