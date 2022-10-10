#![allow(warnings)]

pub mod decoderiter;
pub mod execute_gpu;
pub mod kernels;
pub mod into_slice;
pub mod slice_wrapper;
pub type my_dtype = f32;
pub mod buffer_setup;

use ndarray::prelude::*;
use ndarray;
use crate::{execute_gpu::{execute_ndarray, TrackingParams}};

#[cfg(feature = "python")]
use numpy::ndarray::{ArrayD, ArrayViewD, Array2, Array3, ArrayBase};
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyReadonlyArray3, PyArray2, PyArray3};
#[cfg(feature = "python")]
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

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
#[pymodule]
fn gpu_tracking(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "execute")]
    fn execute_py<'py>(py: Python<'py>, pyarr: PyReadonlyArray3<my_dtype>) -> &'py PyArray2<my_dtype> {
        let array = pyarr.as_array();
        let res = execute_gpu::execute_ndarray(&array, TrackingParams::default(), false);
        res.into_pyarray(py)
    }



    #[pyfn(m)]
    #[pyo3(name = "batch")]
    fn batch_py<'py>(
        py: Python<'py>,
        pyarr: PyReadonlyArray3<my_dtype>,
        diameter: u32,
        minmass: Option<my_dtype>,
        maxsize: Option<my_dtype>,
        separation: Option<u32>,
        noise_size: Option<u32>,
        smoothing_size: Option<u32>,
        threshold: Option<my_dtype>,
        invert: Option<bool>,
        percentile: Option<my_dtype>,
        topn: Option<u32>,
        preprocess: Option<bool>,
        max_iterations: Option<u32>,
        characterize: Option<bool>,
        ) -> &'py PyArray2<my_dtype> {
        
        not_implemented!(maxsize, threshold, invert, percentile,
            topn, preprocess, characterize);
        
        
        let minmass = minmass.unwrap_or(0.);
        let maxsize = maxsize.unwrap_or(f32::INFINITY);
        let separation = separation.unwrap_or(diameter + 1);
        let noise_size = noise_size.unwrap_or(1);
        let smoothing_size = smoothing_size.unwrap_or(diameter);
        let threshold = threshold.unwrap_or(1./255.);
        let invert = invert.unwrap_or(false);
        let percentile = percentile.unwrap_or(64.);
        let topn = topn.unwrap_or(u32::MAX);
        let preprocess = preprocess.unwrap_or(true);
        let max_iterations = max_iterations.unwrap_or(10);
        let characterize = characterize.unwrap_or(false);

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
        }; 
        
        let array = pyarr.as_array();
        let res = execute_gpu::execute_ndarray(&array, params, false);
        res.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "composite")]
    fn composite_py<'py>(py: Python<'py>, sigma: my_dtype, size: u32) -> &'py PyArray2<my_dtype> {
        let res = kernels::Kernel::composite_kernel(sigma, [size, size]);
        let arr = ndarray::Array::from_shape_vec((size as usize, size as usize), res.data).unwrap();
        arr.into_pyarray(py)
        // todo!()
        // res.into_pyarray(py)
    }

    Ok(())
}