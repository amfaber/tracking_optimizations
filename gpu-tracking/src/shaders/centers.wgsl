@group(0) @binding(0)
var<uniform> params: params;

// @group(0) @binding(1)
// var <storage, read> circle_mask: array<f32>;

@group(0) @binding(2)
var<storage, read> processed_buffer: array<f32>;

@group(0) @binding(3)
var<storage, read_write> centers_buffer: array<vec2<f32>>;

@group(0) @binding(4)
var<storage, read_write> masses: array<f32>;

//_feat_char_@group(0) @binding(5)
//_feat_char_var <storage, read> raw_frame: array<f32>;

//_feat_char_@group(0) @binding(6)
//_feat_char_var<storage, read_write> results_buffer: array<f32>;


fn get_center(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32){
  let rint = (kernel_rows - 1) / 2;
  let r = f32(rint);
  let r2 = r * r;
  var centers = vec2<f32>(0.0, 0.0);
  var mass = 0.0;
  // var result = vec3<f32>(0.0, 0.0, 0.0);
  let middle_idx = u * params.pic_ncols + v;
  //_feat_char_let pic_size = params.pic_nrows * params.pic_ncols;
  // let start_u = u - kernel_rows / 2;
  // let start_v = v - kernel_cols / 2;
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

      centers[0] += data * x;
      centers[1] += data * y;
      mass += data;

      //_feat_char_Rg += data*mask;
      //_feat_char_raw_mass += raw_frame[pic_idx];
      //_feat_char_signal = max(signal, data);
      //_feat_char_let theta = atan2(y, x);
      //_feat_char_ecc_sin += data * sin(2.*theta);
      //_feat_char_ecc_cos += data * cos(2.*theta);
    }
  }
  ecc_sin *= ecc_sin;
  ecc_cos *= ecc_cos;
  let ecc = sqrt(ecc_sin + ecc_cos) / (mass - processed_buffer[middle_idx]);
  let Rg = sqrt(Rg / mass);

  //_feat_char_results_buffer[middle_idx + pic_size * 3] = Rg;
  //_feat_char_results_buffer[middle_idx + pic_size * 4] = raw_mass;
  //_feat_char_results_buffer[middle_idx + pic_size * 5] = signal;
  //_feat_char_results_buffer[middle_idx + pic_size * 6] = ecc;

  centers[0] /= mass;
  centers[1] /= mass;
  centers_buffer[middle_idx] = centers;
  masses[middle_idx] = mass;
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
    centers_buffer[idx] = vec2<f32>(0.0, 0.0);
    masses[idx] = 0.0;
    return;
  }

  get_center(i32(global_id.x), i32(global_id.y),
  params.circle_nrows, params.circle_ncols);
  // centers[idx] = vec2<f32>(centerout[0], centerout[1]);
  // masses[idx] = centerout[2];
}