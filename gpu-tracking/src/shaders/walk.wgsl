@group(0) @binding(0)
var<uniform> params: params;

@group(0) @binding(1)
var<storage, read_write> processed_buffer: array<f32>;

@group(0) @binding(2)
var<storage, read_write> centers: array<vec2<f32>>;

@group(0) @binding(3)
var<storage, read_write> masses: array<f32>;

@group(0) @binding(4)
var<storage, read_write> results: array<f32>;

fn is_max(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32) -> bool {
  let center = processed_buffer[u * params.pic_ncols + v];

  let start_u = u - i32(f32(kernel_rows) / 2. - 0.5);
  let start_v = v - i32(f32(kernel_cols) / 2. - 0.5);
  for (var i: i32 = 0; i < kernel_rows; i = i + 1) {
    for (var j: i32 = 0; j < kernel_cols; j = j + 1) {
      let pic_u = start_u + i;
      let pic_v = start_v + j;
      if (pic_u < 0 || pic_u >= params.pic_nrows || pic_v < 0 || pic_v >= params.pic_ncols) {
        return false;
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
  // var adjust_u = 0;
  // var adjust_v = 0;
  var u = u;
  var v = v;
  var center: vec2<f32>;
  var idx: i32;
  var changed = false;
  for (var i: u32 = 0u; i < params.max_iterations; i = i + 1u) {

    idx = u * params.pic_ncols + v;
    center = centers[idx];
    changed = false;
    if (center[0] > params.threshold && u < params.pic_nrows - 1) {
      u += 1;
      changed = true;
    } else if (center[0] < -params.threshold && u > 0) {
      u -= 1;
      changed = true;
    }
    if (center[1] > params.threshold && v < params.pic_ncols - 1) {
      v += 1;
      changed = true;
    } else if (center[1] < -params.threshold && v > 0) {
      v -= 1;
      changed = true;
    }
    if (~changed){
      break;
    }
  }
  let final_coords = vec2<f32>(
    f32(u) + center[0], // + 1000. * f32(changed),
    f32(v) + center[1], // + 1000. * f32(changed),
    );
  return vec3<f32>(final_coords[0], final_coords[1], masses[idx]);
}

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  let idx = global_id.x * u32(params.pic_ncols) + global_id.y;

  if (is_max(i32(global_id.x), i32(global_id.y), params.dilation_nrows, params.dilation_ncols)) {
    // results[idx] = 1.;
    let walk_out = walk(i32(global_id.x), i32(global_id.y));
    if (walk_out[2] > params.minmass){
      let pic_size = u32(params.pic_nrows * params.pic_ncols);
      results[idx + pic_size * 0u] = walk_out[2];
      results[idx + pic_size * 1u] = walk_out[0];
      results[idx + pic_size * 2u] = walk_out[1];
    }
  }
}