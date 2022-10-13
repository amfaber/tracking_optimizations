struct params{
  pic_nrows: i32,
  pic_ncols: i32,
  gauss_nrows: i32,
  gauss_ncols: i32,
  constant_nrows: i32,
  constant_ncols: i32,
  circle_nrows: i32,
  circle_ncols: i32,
  max_iterations: u32,
  threshold: f32,
  minmass: f32,
}
@group(0) @binding(0)
@group(1) @binding(0)
@group(2) @binding(0)
var<uniform> params: params;

@group(0) @binding(1)
var<storage, read> picture: array<f32>; // this is used as both input and output for convenience

@group(0) @binding(2)
var<storage, read> gaussian: array<f32>;

@group(0) @binding(3)
var<storage, read> constant_kernel: array<f32>;

@group(1) @binding(1)
var <storage, read_write> circle_mask: array<f32>;

@group(0) @binding(4)
@group(1) @binding(2)
@group(2) @binding(1)
var<storage, read_write> processed_buffer: array<f32>;

@group(1) @binding(3)
@group(2) @binding(2)
var<storage, read_write> centers: array<vec2<f32>>;

@group(1) @binding(4)
@group(2) @binding(3)
var<storage, read_write> masses: array<f32>;

@group(2) @binding(4)
var<storage, read_write> results: array<vec2<f32>>;

fn convolve_gauss(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32) -> f32 {

  var result = 0.0;
  let start_u = u - kernel_rows / 2;
  let start_v = v - kernel_cols / 2;
  for (var i: i32 = 0; i < kernel_rows; i = i + 1) {
    for (var j: i32 = 0; j < kernel_cols; j = j + 1) {
      // let pic_u = clamp(start_u + i, 0, i32(params.pic_nrows) - 1);
      // let pic_v = clamp(start_v + j, 0, params.pic_ncols - 1);
      let pic_u = start_u + i;
      let pic_v = start_v + j;
      if (pic_u < 0 || pic_u >= params.pic_nrows || pic_v < 0 || pic_v >= params.pic_ncols) {
        continue;
      }
      let pic_idx = pic_u * params.pic_ncols + pic_v;
      let kernel_idx = i * kernel_cols + j;
      result = result + picture[pic_idx] * gaussian[kernel_idx];
    }
  }
  return result;
}

fn convolve_constant(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32) -> f32 {
  var result = 0.0;
  var kernel_sum = 0.0;
  let start_u = u - kernel_rows / 2;
  let start_v = v - kernel_cols / 2;
  for (var i: i32 = 0; i < kernel_rows; i = i + 1) {
    for (var j: i32 = 0; j < kernel_cols; j = j + 1) {
      // let pic_u = clamp(start_u + i, 0, i32(params.pic_nrows) - 1);
      // let pic_v = clamp(start_v + j, 0, params.pic_ncols - 1);
      let pic_u = start_u + i;
      let pic_v = start_v + j;
      if (pic_u < 0 || pic_u >= params.pic_nrows || pic_v < 0 || pic_v >= params.pic_ncols) {
        continue;
      }
      let pic_idx = pic_u * params.pic_ncols + pic_v;
      let kernel_idx = i * kernel_cols + j;
      result += picture[pic_idx] * constant_kernel[kernel_idx];
      kernel_sum += constant_kernel[kernel_idx];
    }
  }
  return result / kernel_sum;
}

fn get_center(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32) -> vec3<f32>{
  var result = vec3<f32>(0.0, 0.0, 0.0);
  let start_u = u - kernel_rows / 2;
  let start_v = v - kernel_cols / 2;
  for (var i: i32 = 0; i < kernel_rows; i = i + 1) {
    for (var j: i32 = 0; j < kernel_cols; j = j + 1) {
      // let pic_u = clamp(start_u + i, 0, i32(params.pic_nrows) - 1);
      // let pic_v = clamp(start_v + j, 0, params.pic_ncols - 1);
      let pic_u = start_u + i;
      let pic_v = start_v + j;
      if (pic_u < 0 || pic_u >= params.pic_nrows || pic_v < 0 || pic_v >= params.pic_ncols) {
        continue;
      }
      let pic_idx = pic_u * params.pic_ncols + pic_v;
      let kernel_idx = i * kernel_cols + j;
      let relevance = processed_buffer[pic_idx] * circle_mask[kernel_idx];
      result += vec3<f32>(relevance * f32(i),
                          relevance * f32(j),
                          relevance);
    }
  }
  let r = f32((kernel_rows - 1) / 2);
  result[0] = result[0] / result[2] - r;
  result[1] = result[1] / result[2] - r;
  return result;
}

fn is_max(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32) -> bool {
  let center = processed_buffer[u * params.pic_ncols + v];
  let start_u = u - kernel_rows / 2;
  let start_v = v - kernel_cols / 2;
  for (var i: i32 = 0; i < kernel_rows; i = i + 1) {
    for (var j: i32 = 0; j < kernel_cols; j = j + 1) {
      let pic_u = start_u + i;
      let pic_v = start_v + j;
      if (pic_u < 0 || pic_u >= params.pic_nrows || pic_v < 0 || pic_v >= params.pic_ncols) {
        continue;
      }
      let pic_idx = pic_u * params.pic_ncols + pic_v;
      if (processed_buffer[pic_idx] > center) {
        return false;
      }
    }
  }
  return true;
}

fn walk(u: i32, v: i32) -> vec3<f32> {
  var adjust_u = 0;
  var adjust_v = 0;
  var center: vec2<f32>;
  var idx: i32;
  for (var i: u32 = 0u; i < params.max_iterations; i = i + 1u) {
    idx = (u + adjust_u) * params.pic_ncols + v + adjust_v;
    center = centers[idx];
    var changed = false;
    if (center[0] > params.threshold){
      adjust_u += 1;
      changed = true;
    } else if (center[0] < -params.threshold){
      adjust_u -= 1;
      changed = true;
    }
    if (center[1] > params.threshold){
      adjust_v += 1;
      changed = true;
    } else if (center[1] < -params.threshold){
      adjust_v -= 1;
      changed = true;
    }
    if (~changed){
      break;
    }
  }
  let final_coords = vec2<f32>(f32(u + adjust_u)+ center[0], f32(v + adjust_v));
  return vec3<f32>(final_coords[0], final_coords[1], masses[idx]);
}

@compute @workgroup_size(16, 16)
fn main_preprocess(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  let idx = global_id.x * u32(params.pic_ncols) + global_id.y;
  
  processed_buffer[idx] = convolve_gauss(i32(global_id.x), i32(global_id.y), params.gauss_nrows, params.gauss_ncols);
  processed_buffer[idx] -= convolve_constant(i32(global_id.x), i32(global_id.y), params.constant_nrows, params.constant_ncols);
  if (processed_buffer[idx] < 1./255.) {
    processed_buffer[idx] = 0.0;
  }
}

@compute @workgroup_size(16, 16)
fn main_centers(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  let idx = global_id.x * u32(params.pic_ncols) + global_id.y;

  let centerout = get_center(i32(global_id.x), i32(global_id.y), params.constant_nrows, params.constant_ncols);
  centers[idx] = vec2<f32>(centerout[0], centerout[1]);
  masses[idx] = centerout[2];
}

@compute @workgroup_size(16, 16)
fn main_walk(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  let idx = global_id.x * u32(params.pic_ncols) + global_id.y;

  if (is_max(i32(global_id.x), i32(global_id.y), params.constant_nrows, params.constant_ncols)) {
    let walk_out = walk(i32(global_id.x), i32(global_id.y));
    if (walk_out[2] > params.minmass){
      results[idx] = vec2<f32>(walk_out[0], walk_out[1]);
    }
    // results[idx] = walk(i32(global_id.x), i32(global_id.y));
  }
}