struct params{
  pic_nrows: i32,
  pic_ncols: i32,
  gauss_nrows: i32,
  gauss_ncols: i32,
}

@group(0) @binding(0)
var<storage, read> picture: array<f32>; // this is used as both input and output for convenience

@group(0) @binding(1)
var<storage, read> gaussian: array<f32>;

@group(0) @binding(2)
var<uniform> params: params;

@group(0) @binding(3)
var<storage, read_write> out_picture: array<f32>;

fn convolve_gauss(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32) -> f32 {

  var result = 0.0;
  let start_u = u - kernel_rows / 2;
  let start_v = v - kernel_cols / 2;
  let i32pic_rows = i32(params.pic_nrows);
  // let start_u = u;
  // let start_v = v;
  for (var i: i32 = 0; i < kernel_rows; i = i + 1) {
    for (var j: i32 = 0; j < kernel_cols; j = j + 1) {
      let pic_u = clamp(start_u + i, 0, i32(params.pic_nrows) - 1);
      let pic_v = clamp(start_v + j, 0, params.pic_ncols - 1);
      let pic_idx = pic_u * params.pic_ncols + pic_v;
      let gauss_idx = i * kernel_cols + j;
      result = result + picture[pic_idx] * gaussian[gauss_idx];
    }
  }
  return result;
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if ((global_id.x >= u32(params.pic_nrows)) || (global_id.y >= u32(params.pic_ncols))) {
    return;
  }
  let idx = global_id.x * u32(params.pic_ncols) + global_id.y;
  
  // if global_id.x == (params.pic_nrows - 1u) {
  //   out_picture[idx] = convolve_gauss(global_id.x, global_id.y, dims(params.gauss_nrows, params.gauss_ncols));
  // };
  // convolve_gauss(global_id.x, global_id.y, dims(params.gauss_nrows, params.gauss_ncols));
  out_picture[idx] = convolve_gauss(i32(global_id.x), i32(global_id.y), params.gauss_nrows, params.gauss_ncols);
  // out_picture[idx] = f32(global_id.x);
}