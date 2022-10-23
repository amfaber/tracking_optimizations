// struct params{
//   pic_nrows: i32,
//   pic_ncols: i32,
//   preprocess_nrows: i32,
//   preprocess_ncols: i32,
//   sigma2: f32,
//   // constant_nrows: i32,
//   // constant_ncols: i32,
//   circle_nrows: i32,
//   circle_ncols: i32,
//   dilation_nrows: i32,
//   dilation_ncols: i32,
//   max_iterations: u32,
//   threshold: f32,
//   minmass: f32,
// }


@group(0) @binding(0)
var<uniform> params: params;

@group(0) @binding(1)
var<storage, read_write> processed_buffer: array<f32>;

@group(0) @binding(2)
var<storage, read_write> particles: array<vec2<i32>>;

@group(0) @binding(3)
var<storage, read> n_particles: u32;

@group(0) @binding(4)
var<storage, read_write> n_particles_filtered: atomic<u32>;

@group(0) @binding(5)
var<storage, read_write> results: array<f32>;

fn get_center(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32) -> vec3<f32>{
  let rint = (kernel_rows - 1) / 2;
  let r = f32(rint);
  let r2 = r * r;
  var centers_and_mass = vec3<f32>(0.0, 0.0, 0.0);
  // var mass = 0.0;
  // var result = vec3<f32>(0.0, 0.0, 0.0);
  let middle_idx = u * params.pic_ncols + v;
  //_feat_char_let pic_size = params.pic_nrows * params.pic_ncols;
  // let start_u = u - kernel_rows / 2;
  // let start_v = v - kernel_cols / 2;
  var Rg = 0.0;
  // var raw_mass = 0.0;
  // var signal = 0.0;
  // var ecc_sin = 0.0;
  // var ecc_cos = 0.0;
  for (var i: i32 = -rint; i <= rint; i = i + 1) {
    let x = f32(i);
    let x2 = x*x;
    for (var j: i32 = -rint; j <= rint; j = j + 1) {
      // let yint = j;
      let y = f32(j);
      let y2 = y*y;
      var mask = x2 + y2;
      if (mask > r2) {
        continue;
      }
      // let pic_u = clamp(start_u + i, 0, i32(params.pic_nrows) - 1);
      // let pic_v = clamp(start_v + j, 0, params.pic_ncols - 1);
      let pic_u = u + i;
      let pic_v = v + j;
      if (pic_u < 0 || pic_u >= params.pic_nrows || pic_v < 0 || pic_v >= params.pic_ncols) {
        continue;
      }
      let pic_idx = pic_u * params.pic_ncols + pic_v;
      let data = processed_buffer[pic_idx];

      centers_and_mass[0] += data * x;
      centers_and_mass[1] += data * y;
      centers_and_mass[2] += data;

      //_feat_char_Rg += data*mask;
      //_feat_char_raw_mass += raw_frame[pic_idx];
      //_feat_char_signal = max(signal, data);
      //_feat_char_let theta = atan2(y, x);
      //_feat_char_ecc_sin += data * sin(2.*theta);
      //_feat_char_ecc_cos += data * cos(2.*theta);
    }
  }
  // ecc_sin *= ecc_sin;
  // ecc_cos *= ecc_cos;
  // let ecc = sqrt(ecc_sin + ecc_cos) / (mass - processed_buffer[middle_idx]);
  // let Rg = sqrt(Rg / mass);

  //_feat_char_results_buffer[middle_idx + pic_size * 3] = Rg;
  //_feat_char_results_buffer[middle_idx + pic_size * 4] = raw_mass;
  //_feat_char_results_buffer[middle_idx + pic_size * 5] = signal;
  //_feat_char_results_buffer[middle_idx + pic_size * 6] = ecc;

  centers_and_mass[0] /= centers_and_mass[2];
  centers_and_mass[1] /= centers_and_mass[2];
  // centers_buffer[middle_idx] = centers_and_mass;
  // masses[middle_idx] = mass;
  return centers_and_mass;
}

fn walk(part_idx: u32) -> vec3<f32> {
  // var adjust_u = 0;
  // var adjust_v = 0;
  var picuv = particles[part_idx];
  var changed = false;
  var center_and_mass: vec3<f32>;
  for (var i: u32 = 0u; i < params.max_iterations; i = i + 1u) {

    // idx = u * params.pic_ncols + v;

    center_and_mass = get_center(picuv[0], picuv[1], params.circle_nrows, params.circle_ncols);
    changed = false;
    if (center_and_mass[0] > params.threshold && picuv[0] < params.pic_nrows - 1) {
      picuv[0] += 1;
      changed = true;
    } else if (center_and_mass[0] < -params.threshold && picuv[0] > 0) {
      picuv[0] -= 1;
      changed = true;
    }
    if (center_and_mass[1] > params.threshold && picuv[1] < params.pic_ncols - 1) {
      picuv[1] += 1;
      changed = true;
    } else if (center_and_mass[1] < -params.threshold && picuv[1] > 0) {
      picuv[1] -= 1;
      changed = true;
    }
    if (~changed){
      break;
    }
  }
  let final_coords = vec3<f32>(
    f32(picuv[0]) + center_and_mass[0], // + 1000. * f32(changed),
    f32(picuv[1]) + center_and_mass[1], // + 1000. * f32(changed),
    center_and_mass[2]
    );
  return final_coords;
}

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= n_particles) {
    return;
  }
  // let idx = global_id.x * u32(params.pic_ncols) + global_id.y;
  let final_coords = walk(global_id.x);
  
  if (final_coords[2] > params.minmass) {
    let part_id = atomicAdd(&n_particles_filtered, 1u);
    let n_res_cols = 7u;
    results[part_id * n_res_cols + 0u] = final_coords[0];
    results[part_id * n_res_cols + 1u] = final_coords[1];
    results[part_id * n_res_cols + 2u] = final_coords[2];
    // centers_buffer[global_id.x] = final_coords;
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