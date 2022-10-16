@group(0) @binding(0)
var<uniform> params: params;

@group(0) @binding(1)
var<storage, read> gauss1d: array<f32>;

@group(0) @binding(2)
var<storage, read> temp: array<vec2<f32>>;

@group(0) @binding(3)
var<storage, read_write> processed_buffer: array<f32>;


@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  let u = i32(global_id.x);
  let v = i32(global_id.y);
  var gauss_sum = 0.0;
  var constant_sum = 0.0;
  let kernel_rows = params.composite_ncols;
  let kernel_cols = params.composite_nrows;
  let start_u = u - kernel_rows / 2;
  var pic_u: i32;
  var pic_idx: i32;
  var do_gauss: bool;
  for (var i: i32 = 0; i < kernel_rows; i = i + 1) {
    pic_u = start_u + i;
    do_gauss = true;
    if (pic_u < 0){
        pic_u = -pic_u + -1;
        do_gauss = false;
      }
    else if (pic_u >= i32(params.pic_nrows)){
        pic_u = 2 * i32(params.pic_nrows) - pic_u - 1;
        do_gauss = false;
      }
    pic_idx = pic_u * params.pic_ncols + v;
    if (do_gauss){
      gauss_sum += gauss1d[i] * temp[pic_idx][0];
    }
    constant_sum += temp[pic_idx][1];
  }
  
  let result = gauss_sum - constant_sum/f32(kernel_rows * kernel_cols);
  if (result < 0.0){
    processed_buffer[u * params.pic_ncols + v] = 0.0;
  }
  else{
    processed_buffer[u * params.pic_ncols + v] = result;
  }
}

