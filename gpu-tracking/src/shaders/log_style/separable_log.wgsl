struct PushConstants{
  dim: u32,
  sigma: f32,
}

var<push_constant> pc: PushConstants;

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;


@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  var gauss_sum = 0.0;
  let dim = pc.dim;
  let sigma = pc.sigma;
  let sigma2 = sigma * sigma;
  let rint = i32(4 * sigma + 0.5);
  var u: i32;
  var v: i32;
  var u_stride: i32;
  if dim == 0 {
    u = i32(global_id.x);
    v = i32(global_id.y);
    u_stride = params.pic_ncols;
  } else {
    u = i32(global_id.y);
    v = i32(global_id.x);
    u_stride = 1;
  }

  var gauss_norm = 0.0;
  let pic_idx_base = i32(u) * u_stride + i32(v);
  for (var i: i32 = -rint; i <= rint; i = i + 1) {
    let x = f32(i);
    let pic_u = u + i;
    if (pic_u < 0) || (pic_u >= i32(params.pic_ncols)) {
      continue;
    }
    let pic_idx = pic_idx_base + i * u_stride;
    let x2 = x * x;
    let gauss_deriv_2 = exp(-x2 / (2.0 * sigma2)) * (x2 - sigma2);
    gauss_sum += gauss_deriv_2 * input[pic_idx];
  }
  
  let sqrt2pi = 2.5066282746310002;
  let norm = sqrt2pi * sigma2 * sigma2 * sigma;
  let result = gauss_sum/gauss_norm;
  output[pic_idx_base] = result;
}