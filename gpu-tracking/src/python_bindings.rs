use std::{fs::File, path::PathBuf};

use crate::{my_dtype, linking::{FrameSubsetter, SubsetterType, SubsetterOutput}};
use ndarray::Array;
use pyo3::{prelude::*, types::{PyDict, PyList}};
// #[cfg(feature = "python")]
use numpy::{IntoPyArray, PyReadonlyArray3, PyReadonlyArray2, PyArray2, PyArray1, PyArrayDyn, PyArray3};
use crate::{execute_gpu::{execute_ndarray, execute_file, path_to_iter, mean_from_iter}, decoderiter::MinimalETSParser, gpu_setup::{TrackingParams, ParamStyle}, linking, error::Error};

impl std::convert::From::<Error> for PyErr{
	fn from(err: Error) -> PyErr{
		match err{
			Error::GpuAdapterError | Error::GpuDeviceError(_)  => {
				pyo3::exceptions::PyConnectionError::new_err(err.to_string())
			},

            Error::ThreadError => {
				pyo3::exceptions::PyBaseException::new_err(err.to_string())
            },
			
            Error::InvalidFileName { .. } |
			Error::FileNotFound{ .. } => {
				pyo3::exceptions::PyFileNotFoundError::new_err(err.to_string())
            },

            Error::DimensionMismatch{ .. } |
            Error::EmptyIterator |
            Error::NonSortedCharacterization |
            Error::FrameOutOfBounds{ .. } |
            Error::NonStandardArrayLayout |
            Error::UnsupportedFileformat { .. } |
            Error::ArrayDimensionsError{ .. } |
            Error::NoExtensionError{ .. } |
            Error::ReadError |
            Error::FrameOOB |
            Error::CastError
             => {
				pyo3::exceptions::PyValueError::new_err(err.to_string())
			},
			
		}
	}
}

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
            maxsize: Option<my_dtype>,
            separation: Option<u32>,
            noise_size: Option<f32>,
            smoothing_size: Option<u32>,
            threshold: Option<my_dtype>,
            invert: Option<bool>,
            percentile: Option<my_dtype>,
            topn: Option<u32>,
            preprocess: Option<bool>,
            filter_close: Option<bool>,

            minmass: Option<my_dtype>,
            max_iterations: Option<u32>,
            characterize: Option<bool>,
            search_range: Option<my_dtype>,
            memory: Option<usize>,
            doughnut_correction: Option<bool>,
            bg_radius: Option<my_dtype>,
            gap_radius: Option<my_dtype>,
            snr: Option<my_dtype>,
            minmass_snr: Option<my_dtype>,
            truncate_preprocessed: Option<bool>,
            correct_illumination: Option<bool>,
            illumination_sigma: Option<my_dtype>,
            adaptive_background: Option<usize>,
            shift_threshold: Option<my_dtype>,
            linker_reset_points: Option<Vec<usize>>,
            keys: Option<Vec<usize>>,
            
            $($postargs)*
        ) -> $outtype {
            not_implemented!(maxsize, threshold, invert, percentile,
                topn, preprocess);
            
            let maxsize = maxsize.unwrap_or(f32::INFINITY);
            let separation = separation.unwrap_or(diameter + 1);
            let noise_size = noise_size.unwrap_or(1.);
            let smoothing_size = smoothing_size.unwrap_or(diameter);
            let threshold = threshold.unwrap_or(1./255.);
            let invert = invert.unwrap_or(false);
            let percentile = percentile.unwrap_or(64.);
            let topn = topn.unwrap_or(u32::MAX);
            let preprocess = preprocess.unwrap_or(true);
            let filter_close = filter_close.unwrap_or(true);
                
            let minmass = minmass.unwrap_or(0.);
            let max_iterations = max_iterations.unwrap_or(10);
            let characterize = characterize.unwrap_or(false);
            let gap_radius = bg_radius.map(|_| gap_radius.unwrap_or(0.));
            let truncate_preprocessed = truncate_preprocessed.unwrap_or(true);
            // let adaptive_background = adaptive_background.unwrap_or(false);
            let shift_threshold = shift_threshold.unwrap_or(0.6);

            let doughnut_correction = doughnut_correction.unwrap_or(false);
            
            let illumination_sigma = match illumination_sigma{
                Some(val) => Some(val),
                None => {
                    if correct_illumination.unwrap_or(false){
                        Some(30.)
                    } else {
                        None
                    }
                }
            };
    
            let mut $params = TrackingParams {
                style: ParamStyle::Trackpy{
                    diameter,

                    maxsize,
                    separation,
                    noise_size,
                    smoothing_size,
                    threshold,
                    invert,
                    percentile,
                    topn,
                    preprocess,
                    filter_close,
                },
                minmass,
                max_iterations,
                characterize,
                search_range,
                memory,
                // cpu_processed,
                doughnut_correction,
                bg_radius,
                gap_radius,
                snr,
                minmass_snr,
                truncate_preprocessed,
                // correct_illumination,
                illumination_sigma,
                adaptive_background,
                include_r_in_output: false,
                shift_threshold,
                linker_reset_points,
                keys,
            };
            $body
        }
    };
}

macro_rules! make_log_args {
    (
    // $(#[$m:meta])*
    $vis:vis fn $name:ident <$($generics:tt),*> ({$($preargs:tt)*}, {$($postargs:tt)*}) -> $outtype:ty => $params:ident $body:block
    ) => {
        // $(#[$m])*
        #[pyfunction]
        $vis fn $name<$($generics),*>($($preargs)*

            min_radius: my_dtype,
            max_radius: my_dtype,
            n_radii: Option<usize>,
            log_spacing: Option<bool>,
            // prune_blobs: Option<bool>,
            overlap_threshold: Option<my_dtype>,

            minmass: Option<my_dtype>,
            max_iterations: Option<u32>,
            characterize: Option<bool>,
            search_range: Option<my_dtype>,
            memory: Option<usize>,
            doughnut_correction: Option<bool>,
            bg_radius: Option<my_dtype>,
            gap_radius: Option<my_dtype>,
            snr: Option<my_dtype>,
            minmass_snr: Option<my_dtype>,
            truncate_preprocessed: Option<bool>,
            correct_illumination: Option<bool>,
            illumination_sigma: Option<my_dtype>,
            adaptive_background: Option<usize>,
            shift_threshold: Option<my_dtype>,
            linker_reset_points: Option<Vec<usize>>,
            keys: Option<Vec<usize>>,
            
            $($postargs)*
        ) -> $outtype {
            
            
            let n_radii = n_radii.unwrap_or(10);
            let log_spacing = log_spacing.unwrap_or(false);
            let overlap_threshold = overlap_threshold.unwrap_or(1.);

            let minmass = minmass.unwrap_or(0.);
            let max_iterations = max_iterations.unwrap_or(10);
            let characterize = characterize.unwrap_or(false);
            let gap_radius = bg_radius.map(|_| gap_radius.unwrap_or(0.));
            let truncate_preprocessed = truncate_preprocessed.unwrap_or(true);
            // let adaptive_background = adaptive_background.unwrap_or(false);
            let shift_threshold = shift_threshold.unwrap_or(0.6);
            
            let doughnut_correction = doughnut_correction.unwrap_or(false);

            let illumination_sigma = match illumination_sigma{
                Some(val) => Some(val),
                None => {
                    if correct_illumination.unwrap_or(false){
                        Some(30.)
                    } else {
                        None
                    }
                }
            };
    
            let mut $params = TrackingParams {
                style: ParamStyle::Log{
                    min_radius,
                    max_radius,
                    n_radii,
                    log_spacing,
                    overlap_threshold,
                },
                minmass,
                max_iterations,
                characterize,
                search_range,
                memory,
                // cpu_processed,
                doughnut_correction,
                bg_radius,
                gap_radius,
                snr,
                minmass_snr,
                truncate_preprocessed,
                illumination_sigma,
                adaptive_background,
                include_r_in_output: true,
                shift_threshold,
                linker_reset_points,
                keys,
            };
            $body
        }
    };
}


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
make_args!(
    fn batch_rust<'py>(
    {
        py: Python<'py>,
        pyarr: PyReadonlyArray3<my_dtype>,
    },
    {
        debug: Option<bool>,
    }
    ) -> PyResult<(&'py PyArray2<my_dtype>, Py<PyAny>)> => params{
        let debug = debug.unwrap_or(false);
        let array = pyarr.as_array();
        let (res, columns) = execute_ndarray(&array, params, debug, 0, None)?;
        Ok((res.into_pyarray(py), columns.into_py(py)))
    }
);

// #[cfg(feature = "python")]
make_args!(
    fn batch_file_rust<'py>(
        {
            py: Python<'py>,
            filename: String,
        },
        {
            channel: Option<usize>,
            debug: Option<bool>,
        }
        ) -> PyResult<(&'py PyArray2<my_dtype>, Py<PyAny>)> => params {
        let debug = debug.unwrap_or(false);

        let (res, columns) = execute_file(
            &filename, channel, params, debug, 0, None)?;
        Ok((res.into_pyarray(py), columns.into_py(py)))
    }
);

make_args!(
    fn characterize_rust<'py>(
    {
        py: Python<'py>,
        pyarr: PyReadonlyArray3<my_dtype>,
        points_to_characterize: PyReadonlyArray2<my_dtype>,
        points_has_frames: bool,
        points_has_r: bool,
    },
    {
        debug: Option<bool>,
    }
    ) -> PyResult<(&'py PyArray2<my_dtype>, Py<PyAny>)> => params{
        let debug = debug.unwrap_or(false);
        let array = pyarr.as_array();
        
        params.characterize = true;
        if points_has_r{
            params.include_r_in_output = true;
        }
        if let ParamStyle::Trackpy{ ref mut filter_close, .. } = params.style{
            *filter_close = false;
        }
        
        let inp = Some((points_to_characterize.as_array(), points_has_frames, points_has_r));
        let (res, columns) = execute_ndarray(&array, params, debug, 0, inp)?;
        Ok((res.into_pyarray(py), columns.into_py(py)))
    }
);

make_args!(
    fn characterize_file_rust<'py>(
        {
            py: Python<'py>,
            filename: String,
            points_to_characterize: PyReadonlyArray2<my_dtype>,
            points_has_frames: bool,
            points_has_r: bool,
        },
        {
            channel: Option<usize>,
            debug: Option<bool>,
        }
        ) -> PyResult<(&'py PyArray2<my_dtype>, Py<PyAny>)> => params {
        let debug = debug.unwrap_or(false);

        params.characterize = true;
        let inp = Some((points_to_characterize.as_array(), points_has_frames, points_has_r));
        if points_has_r{
            params.include_r_in_output = true;
        }
        if let ParamStyle::Trackpy{ ref mut filter_close, .. } = params.style{
            *filter_close = false;
        }
        
        let (res, columns) = execute_file(&filename, channel, params, debug, 0, inp)?;
        Ok((res.into_pyarray(py), columns.into_py(py)))
    }
);

make_log_args!(
    fn log_rust<'py>(
    {
        py: Python<'py>,
        pyarr: PyReadonlyArray3<my_dtype>,
    },
    {
        debug: Option<bool>,
    }
    ) -> PyResult<(&'py PyArray2<my_dtype>, Py<PyAny>)> => params{
    let debug = debug.unwrap_or(false);
    let array = pyarr.as_array();
    let (res, columns) = execute_ndarray(&array, params, debug, 0, None)?;
    Ok((res.into_pyarray(py), columns.into_py(py)))
}
);

make_log_args!(
    fn log_file_rust<'py>(
        {
            py: Python<'py>,
            filename: String,
        },
        {
            channel: Option<usize>,
            // points_to_characterize: Option<PyReadonlyArray2<my_dtype>>,
            debug: Option<bool>,
        }
        ) -> PyResult<(&'py PyArray2<my_dtype>, Py<PyAny>)> => params {
        let debug = debug.unwrap_or(false);
        // let points_rust_array = points_to_characterize.as_ref().map(|arr| arr.as_array());

        let (res, columns) = execute_file(
            &filename, channel, params, debug, 0, None)?;
        Ok((res.into_pyarray(py), columns.into_py(py)))
    }
);


#[pymodule]
fn gpu_tracking(_py: Python, m: &PyModule) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(batch_rust, m)?)?;

    m.add_function(wrap_pyfunction!(batch_file_rust, m)?)?;

    m.add_function(wrap_pyfunction!(log_rust, m)?)?;
    
    m.add_function(wrap_pyfunction!(log_file_rust, m)?)?;
    
    m.add_function(wrap_pyfunction!(characterize_rust, m)?)?;
    
    m.add_function(wrap_pyfunction!(characterize_file_rust, m)?)?;

    #[pyfn(m)]
    #[pyo3(name = "load")]
    fn load<'py>(py: Python<'py>, path: &str, keys: Option<Vec<usize>>, channel: Option<usize>) -> PyResult<&'py PyArray3<my_dtype>>{
        let path = PathBuf::from(path);
        let (provider, dims) = path_to_iter(&path, channel)?;
        let mut output = Vec::new();
        let mut n_frames = 0;
        match keys{
            Some(keys) => {
                if keys.iter().enumerate().all(|(idx, &key)| idx == key){
                    let image_iter = provider.into_iter().take(keys.len());
                    for image in image_iter{
                        let image = image?;
                        output.extend(image.into_iter());
                        n_frames += 1;
                    }
                } else {                
                    for key in keys{
                        let image = provider.get_frame(key)
                            .map_err(|err|{
                                match err{
                                    Error::FrameOOB => Error::FrameOutOfBounds { vid_len: provider.len(Some(key)), problem_idx: key },
                                    _ => err,
                                }
                            })?;
                        output.extend(image.into_iter());
                        n_frames += 1;
                    }
                }
            },
            None => {
                for image in provider.into_iter(){
                    let image = image?;
                    output.extend(image.into_iter());
                    n_frames += 1;
                }
            }
        }
        let arr = Array::from_shape_vec([n_frames, dims[0] as usize, dims[1] as usize], output).unwrap();
        let pyarr = arr.into_pyarray(py);
        Ok(pyarr)
    }
    
    #[pyfn(m)]
    #[pyo3(name = "link_rust")]
    fn link_py<'py>(py: Python<'py>, pyarr: PyReadonlyArray2<my_dtype>,
        search_range: my_dtype,
        memory: Option<usize>) -> PyResult<&'py PyArray1<usize>> {
        let memory = memory.unwrap_or(0);
        let array = pyarr.as_array();
        let frame_iter = linking::FrameSubsetter::new(array, Some(0), (1, 2), None, SubsetterType::Linking)
            .into_linking_iter();
        // let frame_iter = frame_iter.map(|subsetter_element|{
        //     let out = subsetter_element.map(|(frame, subset_outputter)| {
        //         let vec = match subset_outputter{
        //             SubsetterOutput::Linking(vec) => vec,
        //             _ => unreachable!()
        //         };
        //         (frame, vec)
        //     });
        //     out
        // });
        let res = linking::linker_all(frame_iter, search_range, memory)?;
        Ok(res.into_pyarray(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "connect_rust")]
    fn connect<'py>(py: Python<'py>, pyarr1: PyReadonlyArray2<my_dtype>, pyarr2: PyReadonlyArray2<my_dtype>, search_range: my_dtype)
    -> PyResult<(&'py PyArray1<usize>, &'py PyArray1<usize>)>{
        let array1 = pyarr1.as_array();
        let array2 = pyarr2.as_array();
        let frame_iter1 = linking::FrameSubsetter::new(array1, Some(0), (1, 2), None, SubsetterType::Linking)
            .into_linking_iter();
        let frame_iter2 = linking::FrameSubsetter::new(array2, Some(0), (1, 2), None, SubsetterType::Linking)
            .into_linking_iter();
        let (res1, res2) = linking::connect_all(frame_iter1, frame_iter2, search_range)?;
        Ok((res1.into_pyarray(py), res2.into_pyarray(py)))
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

    #[pyfn(m)]
    #[pyo3(name = "parse_ets_with_keys")]
    fn parse_ets_with_keys<'py>(py: Python<'py>, path: &str, keys: Vec<usize>, channel: Option<usize>) -> PyResult<&'py PyArray3<u16>>{
        let mut file = File::open(path).unwrap();
        let parser = MinimalETSParser::new(&mut file).unwrap();
        let channel = channel.unwrap_or(0);
        let mut iter = parser.iterate_channel(file.try_clone().unwrap(), channel);
        let n_frames = keys.len();
        let mut vec = Vec::with_capacity(n_frames * parser.dims.iter().product::<usize>());
        for key in keys{
            iter.seek(key);
            vec.extend(iter.next().unwrap().into_iter().flatten());
        }
        let array = Array::from_shape_vec((n_frames, parser.dims[1], parser.dims[0]), vec).unwrap();
        let array = array.into_pyarray(py);
        Ok(array)
    }

    #[pyfn(m)]
    #[pyo3(name = "mean_from_disk")]
    fn mean_from_disk<'py>(py: Python<'py>, path: &str, channel: Option<usize>) -> PyResult<&'py PyArray2<f32>>{
        let path = PathBuf::from(path);
        let (provider, dims) = path_to_iter(&path, channel)?;
        // let iter = (0..)
        //     .map(|i| provider.get_frame(i))
        //     .take_while(|res| !matches!(res, Err(crate::error::Error::FrameOOB)));
        let iter = provider.into_iter();
        let mean_arr = mean_from_iter(iter, &dims, channel)?;
        // let idk = provider.get_frame(0);
        // let mut n_frames = 0;
        // let mut mean_vec = iter.fold(vec![0f32; (dims[0] * dims[1]) as usize], |mut acc, ele|{
        //     for (a, e) in acc.iter_mut().zip(ele.iter()){
        //         *a += e
        //     }
        //     n_frames += 1;
        //     acc
        // });
        // let n_frames = n_frames as f32;
        // for e in mean_vec.iter_mut(){
        //     *e /= n_frames;
        // }
        // let mean_arr = Array::from_shape_vec((dims[0] as usize, dims[1] as usize), mean_vec).unwrap();
        
        Ok(mean_arr.into_pyarray(py))
    }
    
    
    Ok(())
}