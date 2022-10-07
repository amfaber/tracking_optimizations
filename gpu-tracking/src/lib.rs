#![allow(warnings)]
use pyo3::prelude::*;
pub mod decoderiter;
pub mod execute_gpu;
pub mod kernels;
pub mod into_slice;
pub mod slice_wrapper;
pub type my_dtype = f32;
pub mod buffer_setup;

use numpy::ndarray::{ArrayD, ArrayViewD, Array2, Array3, ArrayBase};
use numpy::{IntoPyArray, PyReadonlyArray3, PyArray2, PyArray3};
use ndarray::prelude::*;
use crate::{execute_gpu::{execute_ndarray, TrackingParams}};



#[pymodule]
fn gpu_tracking(_py: Python, m: &PyModule) -> PyResult<()> {
    

    #[pyfn(m)]
    #[pyo3(name = "execute")]
    fn execute_py<'py>(py: Python<'py>, pyarr: PyReadonlyArray3<my_dtype>) -> &'py PyArray2<my_dtype> {
        let array = pyarr.as_array();
        let res = execute_gpu::execute_ndarray(&array, TrackingParams::default());
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
        
        
        
        
        let array = pyarr.as_array();
        let res = execute_gpu::execute_ndarray(&array, TrackingParams::default());
        res.into_pyarray(py)
    }



    Ok(())
}