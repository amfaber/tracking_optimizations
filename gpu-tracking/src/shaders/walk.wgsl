struct ParticleLocation{
    x: i32,
    y: i32,
    r: f32,
    log_space: f32,
}

struct Shape{
    nrows: u32,
    ncols: u32,
}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> processed_buffer: array<f32>;

@group(0) @binding(2)
var<storage, read> particles: array<ParticleLocation>;

@group(0) @binding(3)
var<storage, read> n_particles: u32;

@group(0) @binding(4)
var<storage, read_write> n_particles_filtered: atomic<u32>;

@group(0) @binding(5)
var<storage, read_write> results: array<f32>;

@group(0) @binding(6)
var<storage, read> raw_frame: array<f32>;

// //_feat_LoG @group(0) @binding(7)
// //_feat_LoG var<uniform> shape: Shape;


fn get_center(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32, transform: vec2<i32>) -> vec3<f32>{
  let rint = (kernel_rows - 1) / 2;
  let r = f32(rint);
  let r2 = r * r;
  var centers_and_mass = vec3<f32>(0.0, 0.0, 0.0);
  let middle_idx = u * params.pic_ncols + v;
  for (var i: i32 = -rint; i <= rint; i = i + 1) {
    let x = f32(i);
    let x2 = x*x;
    for (var j: i32 = -rint; j <= rint; j = j + 1) {
      let y = f32(j);
      let y2 = y*y;
      var mask = x2 + y2;
      if (mask > r2) {
        continue;
      }
      let pic_u = u + i;
      let pic_v = v + j;
      if (pic_u < 0 || pic_u >= params.pic_nrows || pic_v < 0 || pic_v >= params.pic_ncols) {
        continue;
      }
      // //_feat_LoG let pic_u = (pic_u + transform[0]) % i32(shape.nrows);
      // //_feat_LoG let pic_v = (pic_v + transform[1]) % i32(shape.ncols);
      let pic_idx = pic_u * params.pic_ncols + pic_v;
      let data = processed_buffer[pic_idx];

      centers_and_mass[0] += data * x;
      centers_and_mass[1] += data * y;
      centers_and_mass[2] += data;
    }
  }
  centers_and_mass[0] /= centers_and_mass[2];
  centers_and_mass[1] /= centers_and_mass[2];
  return centers_and_mass;
}

fn variance_check(u: f32, v: f32, kernel_rows: i32, kernel_cols: i32) -> bool{
  let u = i32(round(u));
  let v = i32(round(v));

  let radiusu = kernel_rows / 2;
  let radiusv = kernel_cols / 2;
  
  var mean = 0.0;
  var counter = 0.0;
  for (var i: i32 = -radiusu; i <= radiusu; i = i + 1) {
    let pic_u = u + i;
    if (pic_u < 0 || pic_u >= params.pic_nrows) {
      continue;
    }
    for (var j: i32 = -radiusv; j <= radiusv; j = j + 1) {
      let pic_v = v + j;
      if (pic_v < 0 || pic_v >= params.pic_ncols) {
        continue;
      }
      let pic_idx = pic_u * params.pic_ncols + pic_v;
      mean += raw_frame[pic_idx];
      counter += 1.0;
    }
  }
  mean /= counter;

  var variance = 0.0;
  for (var i: i32 = -radiusu; i <= radiusu; i = i + 1) {
    let pic_u = u + i;
    if (pic_u < 0 || pic_u >= params.pic_nrows) {
      continue;
    }
    for (var j: i32 = -radiusv; j <= radiusv; j = j + 1) {
      let pic_v = v + j;
      if (pic_v < 0 || pic_v >= params.pic_ncols) {
        continue;
      }
      let pic_idx = pic_u * params.pic_ncols + pic_v;
      variance += (raw_frame[pic_idx] - mean) * (raw_frame[pic_idx] - mean);
    }
  }
  variance /= counter - 1.0;
  // variance = sqrt(variance);
  let stdev = sqrt(variance);
  if (processed_buffer[u * params.pic_ncols + v] > stdev * params.var_factor) {
    return true;
  } else {
    return false;
  }
}

fn walk(argpicuv: vec2<i32>, r: f32, transform: vec2<i32>) -> vec3<f32> {
  // var adjust_u = 0;
  // var adjust_v = 0;
  var picuv = argpicuv;
  var changed = false;
  var center_and_mass: vec3<f32>;

  var radiusu: i32;
  var radiusv: i32;
  var kernel_rows: i32;
  var kernel_cols: i32;
  if r > 0.0 {
    radiusu = i32(ceil(r));
    radiusv = i32(ceil(r));
    kernel_rows = radiusu * 2 + 1;
    kernel_cols = radiusv * 2 + 1;

  } else {
    radiusu = params.circle_nrows / 2;
    radiusv = params.circle_ncols / 2;
    kernel_rows = params.circle_nrows;
    kernel_cols = params.circle_ncols;
  }

  for (var i: u32 = 0u; i < params.max_iterations; i = i + 1u) {
    // idx = u * params.pic_ncols + v;
    center_and_mass = get_center(picuv[0], picuv[1], kernel_rows, kernel_cols, transform);
    changed = false;
    if ((center_and_mass[0] > params.threshold) && (picuv[0] < params.pic_nrows - 1 - radiusu)) {
      picuv[0] += 1;
      changed = true;
    } else if (center_and_mass[0] < -params.threshold && picuv[0] > radiusu) {
      picuv[0] -= 1;
      changed = true;
    }
    if ((center_and_mass[1] > params.threshold) && (picuv[1] < params.pic_ncols - 1 - radiusv)) {
      picuv[1] += 1;
      changed = true;
    } else if ((center_and_mass[1] < -params.threshold) && (picuv[1] > radiusv)) {
      picuv[1] -= 1;
      changed = true;
    }
    if (!changed){
      break;
    }
  }
  let final_coords = vec3<f32>(
    f32(picuv[0]) + center_and_mass[0], // + 1000. * f32(changed),
    f32(picuv[1]) + center_and_mass[1], // + 1000. * f32(changed),
    center_and_mass[2],
    // f32(changed)
    );
  return final_coords;
}

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= n_particles) {
    return;
  }
  // let idx = global_id.x * u32(params.pic_ncols) + global_id.y;
  let part_idx = global_id.x;
  let particle_location = particles[part_idx];
  let picuv = vec2<i32>(particle_location.x, particle_location.y);

  let pad_offset = vec2<i32>(0, 0);
  // //_feat_LoG let pad_offset = vec2<i32>((i32(shape.nrows) - params.pic_nrows) / 2, (i32(shape.ncols) - params.pic_ncols));

  let r = particle_location.r;
  let log_space = particle_location.log_space;

  let final_coords = walk(picuv, r, pad_offset);

  let varcheck = true;

  //_feat_varcheck var kernel_cols: i32;
  //_feat_varcheck var kernel_rows: i32;
  //_feat_varcheck if r > 0.0 {
  //_feat_varcheck   kernel_rows = i32(ceil(r)) * 2 + 1;
  //_feat_varcheck   kernel_cols = i32(ceil(r)) * 2 + 1;
  //_feat_varcheck } else {
  //_feat_varcheck   kernel_rows = params.preprocess_nrows;
  //_feat_varcheck   kernel_cols = params.preprocess_ncols;
  //_feat_varcheck }
  //_feat_varcheck let varcheck = variance_check(final_coords[0], final_coords[1], kernel_rows, kernel_cols);

  let minmass = params.minmass;
  // let minmass = -10000000.;
  if (final_coords[2] > minmass && varcheck) {
    let part_id = atomicAdd(&n_particles_filtered, 1u);
    let n_res = 9u;
    results[part_id * n_res + 0u] = final_coords[0];
    results[part_id * n_res + 1u] = final_coords[1];
    results[part_id * n_res + 2u] = final_coords[2];
    if r>0.0 {
      results[part_id * n_res + 3u] = r;
      results[part_id * n_res + 4u] = log_space;
    }

  }
  // let pic_size = u32(params.pic_nrows * params.pic_ncols);
  // if (is_max(i32(global_id.x), i32(global_id.y), params.dilation_nrows, params.dilation_ncols)) {
  //   // results[idx] = 1.;
  //   let walk_out = walk(i32(global_id.x), i32(global_id.y));
  //   if (walk_out[2] > params.minmass){
  //     results[idx + pic_size * 0u] = walk_out[2];
  //     results[idx + pic_size * 1u] = walk_out[0];
  //     results[idx + pic_size * 2u] = walk_out[1];
  //   }
  // }
}