@group(0) @binding(0)
var<uniform> params: params;

@group(0) @binding(1)
var <storage, read_write> circle_mask: array<f32>;

@group(0) @binding(2)
var<storage, read_write> processed_buffer: array<f32>;

@group(0) @binding(3)
var<storage, read_write> centers: array<vec2<f32>>;

@group(0) @binding(4)
var<storage, read_write> masses: array<f32>;


fn get_center(u: i32, v: i32, kernel_rows: i32, kernel_cols: i32) -> vec3<f32>{
  var result = vec3<f32>(0.0, 0.0, 0.0);
  let start_u = u - kernel_rows / 2;
  let start_v = v - kernel_cols / 2;
  for (var i: i32 = 0; i < kernel_rows; i = i + 1) {
    for (var j: i32 = 0; j < kernel_cols; j = j + 1) {
      // let pic_u = clamp(start_u + i, 0, i32(params.pic_nrows) - 1);
      // let pic_v = clamp(start_v + j, 0, params.pic_ncols - 1);
      let pic_u = start_u + i;
      let pic_v = start_v + j;
      if (pic_u < 0 || pic_u >= params.pic_nrows || pic_v < 0 || pic_v >= params.pic_ncols) {
        continue;
      }
      let pic_idx = pic_u * params.pic_ncols + pic_v;
      let kernel_idx = i * kernel_cols + j;
      let relevance = processed_buffer[pic_idx] * circle_mask[kernel_idx];
      result += vec3<f32>(relevance * f32(i),
                          relevance * f32(j),
                          relevance);
    }
  }
  let r = f32((kernel_rows - 1) / 2);
  result[0] = result[0] / result[2] - r;
  result[1] = result[1] / result[2] - r;
  return result;
}

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
    return;
  }
  let idx = global_id.x * u32(params.pic_ncols) + global_id.y;

  let centerout = get_center(i32(global_id.x), i32(global_id.y),
  params.circle_nrows, params.circle_ncols);
  centers[idx] = vec2<f32>(centerout[0], centerout[1]);
  masses[idx] = centerout[2];
}