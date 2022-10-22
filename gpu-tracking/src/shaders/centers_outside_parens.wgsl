@group(0) @binding(0)
var<uniform> params: params;

@group(0) @binding(1)
var <storage, read_write> circle_mask: array<f32>;

@group(0) @binding(2)
var<storage, read_write> processed_buffer: array<f32>;

@group(0) @binding(3)
var<storage, read_write> centers: array<vec2<f32>>;

@group(0) @binding(4)
var<storage, read_write> masses: array<f32>;


fn get_center(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32) -> vec3<f32>{
  var result = vec3<f32>(0.0, 0.0, 0.0);
  let start_u = u - kernel_rows / 2;
  let start_v = v - kernel_cols / 2;
  var pic_along_row = start_u * params.pic_ncols + start_v;
  var pic_along_col = start_u * params.pic_ncols + start_v;
  var kernel_idx = 0;
  var counters: vec2<f32>;
  var idx_along_row: i32;
  var idx_along_col: i32;

  for (var i: i32 = 0; i < kernel_rows; i = i + 1) {
    counters[0] = 0.;
    counters[1] = 0.;
    for (var j: i32 = 0; j < kernel_cols; j = j + 1) {
      if (circle_mask[kernel_idx] == 1.){
        idx_along_row = (start_u + i) * params.pic_ncols + start_v + j;
        idx_along_col = (start_u + j) * params.pic_ncols + start_v + i;
        counters[0] += processed_buffer[idx_along_row];
        counters[1] += processed_buffer[idx_along_col];
        result[2] += processed_buffer[idx_along_row];
      }
      kernel_idx += 1;
    }
    result[0] += counters[0] * f32(i);
    result[1] += counters[1] * f32(i);
  }
  if (result[2] == 0.){
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  let r = f32((kernel_rows - 1) / 2);
  result[0] = result[0] / result[2] - r;
  result[1] = result[1] / result[2] - r;
  return result;
}

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  let u = i32(global_id.x);
  let v = i32(global_id.y);
  let kernel_rows = params.circle_nrows;
  let kernel_cols = params.circle_ncols;
  let idx = u * params.pic_ncols + v;

  if (u - kernel_rows / 2 < 0 || u + kernel_rows / 2 >= params.pic_nrows || v - kernel_cols / 2 < 0 || v + kernel_cols / 2 >= params.pic_ncols) {
    centers[idx] = vec2<f32>(0.0, 0.0);
    return;
  }
  let centerout = get_center(i32(global_id.x), i32(global_id.y),
  params.circle_nrows, params.circle_ncols);
  centers[idx] = vec2<f32>(centerout[0], centerout[1]);
  masses[idx] = centerout[2];
}