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
var<storage, read> processed_buffer: array<f32>;

@group(0) @binding(2)
var<storage, read> raw_frame: array<f32>;

@group(0) @binding(3)
var<storage, read> n_particles: u32;

@group(0) @binding(4)
var<storage, read_write> results: array<f32>;


fn characterize(part_idx: u32, kernel_rows: i32, kernel_cols: i32){
  let rint = (kernel_rows - 1) / 2;
  let r = f32(rint);
  let r2 = r * r;
  let u = i32(round(results[part_idx * 7u + 0u]));
  let v = i32(round(results[part_idx * 7u + 1u]));
  let mass = results[part_idx * 7u + 2u];

  let middle_idx = u * params.pic_ncols + v;
  let pic_size = params.pic_nrows * params.pic_ncols;
//   let start_u = u - kernel_rows / 2;
//   let start_v = v - kernel_cols / 2;
  var Rg = 0.0;
  var raw_mass = 0.0;
  var signal = 0.0;
  var ecc_sin = 0.0;
  var ecc_cos = 0.0;
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

    //   centers[0] += data * x;
    //   centers[1] += data * y;
    //   mass += data;

      Rg += data*mask;
      raw_mass += raw_frame[pic_idx];
      signal = max(signal, data);
      let theta = atan2(y, x);
      ecc_sin += data * sin(2.*theta);
      ecc_cos += data * cos(2.*theta);
    }
  }
  ecc_sin *= ecc_sin;
  ecc_cos *= ecc_cos;
  let ecc = sqrt(ecc_sin + ecc_cos) / (mass - processed_buffer[middle_idx]);
  let Rg = sqrt(Rg / mass);

  results[part_idx * 7u + 3u] = Rg;
  results[part_idx * 7u + 4u] = raw_mass;
  results[part_idx * 7u + 5u] = signal;
  results[part_idx * 7u + 6u] = ecc;

//   centers[0] /= mass;
//   centers[1] /= mass;
//   centers_buffer[middle_idx] = centers;
//   masses[middle_idx] = mass;
}

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= n_particles) {
        return;
    }

    // let u = i32(round(results[idx*3u + 0u]));
    // let v = i32(round(results[idx*3u + 1u]));
    characterize(idx, params.circle_nrows, params.circle_ncols);

}