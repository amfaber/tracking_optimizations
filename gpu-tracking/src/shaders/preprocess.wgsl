@group(0) @binding(0)
var<uniform> params: params;

@group(0) @binding(1)
var<storage, read> picture: array<f32>; // this is used as both input and output for convenience

@group(0) @binding(2)
var<storage, read> gauss1d: array<f32>;

@group(0) @binding(3)
var<storage, read_write> temp: array<vec2<f32>>;

// @group(0) @binding(4)
// var<storage, read_write> temp_constant: array<f32>;

@group(0) @binding(4)
var<storage, read_write> processed_buffer: array<f32>;

fn convolve_rows(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32) -> vec2<f32> {
  var gauss_sum = 0.0;
  var constant_sum = 0.0;
  // let kernel_increment = 1.0 / f32(kernel_rows * kernel_cols);
  // let start_u = u - kernel_rows / 2;
  let start_v = v - kernel_cols / 2;
  for (var j: i32 = 0; j < kernel_cols; j = j + 1) {
    // let pic_u = clamp(start_u + i, 0, i32(params.pic_nrows) - 1);
    // let pic_v = clamp(start_v + j, 0, i32(params.pic_ncols) - 1);
    
    
    // let pic_u = start_u + i;
    let pic_v = start_v + j;

    // var pic_u = start_u + i;
    // var pic_v = start_v + j;
    // if (pic_u < 0){
    //   pic_u = -pic_u + -1;
    // }
    // if (pic_v < 0){
    //   pic_v = -pic_v + -1;
    // }
    // if (pic_u >= i32(params.pic_nrows)){
    //   pic_u = 2 * i32(params.pic_nrows) - pic_u - 1;
    // }
    // if (pic_v >= i32(params.pic_ncols)){
    //   pic_v = 2 * i32(params.pic_ncols) - pic_v - 1;
    // }

    // if (pic_u < 0 || pic_u >= params.pic_nrows || pic_v < 0 || pic_v >= params.pic_ncols) {
    //   continue;
    // }
    let pic_idx = u + pic_v;
    gauss_sum += gauss1d[j] * picture[pic_idx];
    constant_sum += picture[pic_idx];
  }

  return vec2<f32>(gauss_sum, constant_sum);
}

fn convolve_finish(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32) -> f32 {
  var gauss_sum = 0.0;
  var constant_sum = 0.0;
  var pic_u = u - kernel_rows / 2;
  let u_increment = params.pic_ncols;
  for (var i: i32 = 0; i < kernel_rows; i = i + 1) {
    let pic_idx = pic_u + v;
    gauss_sum += gauss1d[i] * temp[pic_idx][0];
    constant_sum += temp[pic_idx][1];
    pic_u += u_increment;
  }
  return gauss_sum - constant_sum/f32(kernel_rows * kernel_cols);
}


@compute @workgroup_size(_)
fn rows(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  let idx = global_id.x * u32(params.pic_ncols) + global_id.y;
  
  let rows_out = convolve_rows(i32(global_id.x), i32(global_id.y), params.composite_nrows, params.composite_ncols);
  temp[idx] = rows_out;
  // if (processed_buffer[idx] < 1./255.) {
  //   processed_buffer[idx] = 0.0;
  // }
}

@compute @workgroup_size(_)
fn finish(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  let idx = global_id.x * u32(params.pic_ncols) + global_id.y;
  
  let result = convolve_finish(i32(global_id.x), i32(global_id.y), params.composite_nrows, params.composite_ncols);
  processed_buffer[idx] = result;
  if (result > 1./255.) {
    processed_buffer[idx] = result;
  }
}

