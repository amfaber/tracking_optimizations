struct ParticleLocation{
    x: i32,
    y: i32,
    r: f32,
    log_space_max: f32,
}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> processed_buffer: array<f32>;

@group(0) @binding(2)
var<storage, read> max_rows: array<f32>;

@group(0) @binding(3)
var<storage, read_write> n_particles: atomic<u32>;

@group(0) @binding(4)
var<storage, read_write> max_points: array<ParticleLocation>;

fn is_max(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32) -> bool {
  let center = processed_buffer[u * params.pic_ncols + v];
  var pic_u = u - i32(f32(kernel_rows) / 2. - 0.5);
  var pic_idx: i32;
  for (var i: i32 = 0; i < kernel_rows; i = i + 1) {
    // if (pic_u < 0 || pic_u >= params.pic_nrows || pic_v < 0 || pic_v >= params.pic_ncols) {
    //   return false;
    // }
    if (pic_u >= 0 && pic_u < params.pic_nrows){
      pic_idx = pic_u * params.pic_ncols + v;
      if (max_rows[pic_idx] > center) {
        return false;
      }
    }
    pic_u += 1;
  }
  return true;
}


@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  let u = i32(global_id.x);
  let v = i32(global_id.y);
  if (u < params.margin || u >= params.pic_nrows - params.margin || v < params.margin || v >= params.pic_ncols - params.margin) {
    return;
  }

  let idx = global_id.x * u32(params.pic_ncols) + global_id.y;

  if (is_max(i32(global_id.x), i32(global_id.y), params.dilation_nrows, params.dilation_ncols)) {
    let part = atomicAdd(&n_particles, 1u);
    max_points[part] = ParticleLocation(u, v, -1.0, -1.0);
  }
}