use anyhow;
use thiserror;
use wgpu;

#[derive(Debug, thiserror::Error)]
pub enum Error{
	#[error("could not get gpu adapter")]
	GpuAdapterError,
	
	#[error("could not get gpu device")]
	GpuDeviceError(#[from] wgpu::RequestDeviceError),
	
	#[error("could not find file '{filename}'")]
	FileNotFound{
		source: std::io::Error,
		filename: std::path::PathBuf,
	},

	#[error("dimension mismatch. provided dimensions are ({}, {}), but frame {} contains {} pixels", .dimensions[0], .dimensions[1], idx, frame_len)]
	DimensionMismatch{
		idx: usize,
		frame_len: usize,
		dimensions: [u32; 2],
	},

	#[error("the provided video doesn't contain any frames.")]
	EmptyIterator,

	#[error("the points provided for characterization aren't sorted in ascending frame number.")]
	NonSortedCharacterization,

	#[error("Requested characterization for a frame outside the length
 of the video. Video length is {}, requested characterization in frame {}", vid_len, problem_idx)]
	FrameOutOfBounds{
		vid_len: usize,
		problem_idx: usize,
	},
	
	#[error("unexpected threading error. this will require some more debugging.")]
	ThreadError,
	// ThreadError{
	// 	source: Box<dyn std::error::Error + Send>,
	// },
	
	#[error("the passed array is not in standard memory layout. perhaps it is a view of a larger array?")]
	NonStandardArrayLayout,

	
	#[error("could not determine extension of file '{filename}'")]
	NoExtensionError{
		filename: std::path::PathBuf,
	},

	#[error("unsupported file format: {extension}")]
	UnsupportedFileformat{
		extension: String,
	},

	#[error("unsupported characterize array dimensions.
accepted dimensions: (Nx2) or (Nx3). received: {:?}", dims)]
	ArrayDimensionsError{
		dims: Vec<usize>,
	}
}

pub type Result<T> = std::result::Result<T, Error>;

