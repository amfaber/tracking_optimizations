#![allow(warnings)]
use pyo3::prelude::*;
pub mod decoderiter;
pub mod execute_gpu;
pub mod kernels;
pub mod into_slice;
pub mod slice_wrapper;
pub type my_dtype = f32;

use numpy::ndarray::{ArrayD, ArrayViewD, Array2, Array3, ArrayBase};
use numpy::{IntoPyArray, PyReadonlyArray3, PyArray2, PyArray3};
use ndarray::prelude::*;
use crate::{execute_gpu::execute_ndarray};



#[pymodule]
fn gpu_tracking(_py: Python, m: &PyModule) -> PyResult<()> {
    

    #[pyfn(m)]
    #[pyo3(name = "execute")]
    fn execute_py<'py>(py: Python<'py>, pyarr: PyReadonlyArray3<my_dtype>) -> &'py PyArray2<my_dtype> {
        let array = pyarr.as_array();
        let res = execute_gpu::execute_ndarray(&array);
        res.into_pyarray(py)
    }


    Ok(())
}