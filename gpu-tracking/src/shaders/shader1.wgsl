struct dims {
  nrows: u32,
  ncols: u32,
};

struct params{
  pic_nrows: u32,
  pic_ncols: u32,
  gauss_nrows: u32,
  gauss_ncols: u32,
}

@group(0) @binding(0)
var<storage, read> picture: array<f32>; // this is used as both input and output for convenience

@group(0) @binding(1)
var<storage, read> gaussian: array<f32>;

@group(0) @binding(2)
var<uniform> params: params;

@group(0) @binding(3)
var<storage, read_write> out_picture: array<f32>;

fn convolve_gauss(u: u32, v: u32, size: dims) -> f32 {
  var result = 0.0;
  let start_u = u - size.nrows / 2u;
  let start_v = v - size.ncols / 2u;
  // let start_u = u;
  // let start_v = v;
  for (var i: u32 = 0u; i < size.nrows; i = i + 1u) {
    for (var j: u32 = 0u; j < size.ncols; j = j + 1u) {
      let pic_u = clamp(start_u + i, 0u, params.pic_nrows - 1u);
      let pic_v = clamp(start_v + j, 0u, params.pic_ncols - 1u);
      let pic_idx = pic_u * params.pic_ncols + pic_v;
      let gauss_idx = i * size.ncols + j;
      result = result + picture[pic_idx] * gaussian[gauss_idx];
    }
  }
  return result;
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if ((global_id.x >= params.pic_nrows) || (global_id.y >= params.pic_ncols)) {
    return;
  }
  let idx = global_id.x * params.pic_ncols + global_id.y;
  
  convolve_gauss(global_id.x, global_id.y, dims(params.gauss_nrows, params.gauss_ncols));
  out_picture[idx] = convolve_gauss(global_id.x, global_id.y, dims(params.gauss_nrows, params.gauss_ncols));
  // out_picture[idx] = f32(global_id.x);
}