#![allow(warnings)]

pub mod decoderiter;
pub mod execute_gpu;
pub mod kernels;
pub mod into_slice;
pub mod slice_wrapper;
pub type my_dtype = f32;
pub mod gpu_setup;
pub mod linking;

use ndarray::prelude::*;
use ndarray;
use crate::{execute_gpu::{execute_ndarray, TrackingParams}};
// use pyo3::PyList;
// #[cfg(feature = "python")]
use numpy::ndarray::{ArrayD, ArrayViewD, Array2, Array3, ArrayBase};
// #[cfg(feature = "python")]
use numpy::{IntoPyArray, PyReadonlyArray3, PyReadonlyArray2, PyArray2, PyArray3, PyArray1};
// #[cfg(feature = "python")]
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

// #[cfg(feature = "python")]
use pyo3::{prelude::*, types::PyList};
// #[cfg(feature = "python")]
#[pymodule]
fn gpu_tracking(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "batch")]
    fn batch_py<'py>(
        py: Python<'py>,
        pyarr: PyReadonlyArray3<my_dtype>,
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
        cpu_processed: Option<bool>,
        sig_radius: Option<my_dtype>,
        bg_radius: Option<my_dtype>,
        gap_radius: Option<my_dtype>,
        ) ->  (&'py PyArray2<my_dtype>, Py<PyAny>) {
        
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
        let cpu_processed = cpu_processed.unwrap_or(false);
        // let sig_radius = sig_radius;
        // let bg_radius = bg_radius;
        let gap_radius = bg_radius.map(|_| gap_radius.unwrap_or(0.));
        // let gap_radius = gap_radius;

        // neither search_range nor memory are unwrapped as linking is optional on the Rust side.

        let params = TrackingParams {
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
            cpu_processed,
            sig_radius,
            bg_radius,
            gap_radius,
        }; 
        
        let array = pyarr.as_array();
        let (res, columns) = execute_gpu::execute_ndarray(&array, params, false);
        (res.into_pyarray(py), columns.into_py(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "link")]
    fn link_py<'py>(py: Python<'py>, pyarr: PyReadonlyArray2<my_dtype>,
        search_range: my_dtype,
        memory: Option<usize>) -> &'py PyArray1<usize> {
        let memory = memory.unwrap_or(0);
        let array = pyarr.as_array();
        let frame_iter = linking::FrameSubsetter::new(&array, 0, (2, 3));
        let res = linking::linker_all(frame_iter, search_range, memory);
        res.into_pyarray(py)
    }

    Ok(())
}