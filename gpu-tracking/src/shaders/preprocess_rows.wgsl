@group(0) @binding(0)
var<uniform> params: params;

@group(0) @binding(1)
var<storage, read> picture: array<f32>;

@group(0) @binding(2)
var<storage, read> gauss1d: array<f32>;

@group(0) @binding(3)
var<storage, read_write> temp: array<vec2<f32>>;


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
  let pic_v = v - kernel_cols / 2;
  var pic_idx = u * params.pic_ncols + pic_v;
  for (var j: i32 = 0; j < kernel_cols; j = j + 1) {
    gauss_sum += gauss1d[j] * picture[pic_idx];
    constant_sum += picture[pic_idx];
    pic_idx += 1;
  }
  
  temp[u * params.pic_ncols + v] = vec2(gauss_sum, constant_sum);
}