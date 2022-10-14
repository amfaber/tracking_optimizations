@group(0) @binding(0)
var<uniform> params: params;

@group(0) @binding(1)
var <storage, read_write> circle_mask: array<u32>;

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
  var pic_along_row = start_u * params.cols + start_v;
  var pic_along_col = start_v * params.rows + start_u;
  var kernel_idx = 0;


  for (var i: i32 = 0; i < kernel_rows; i = i + 1) {
    var counters = vec3<f32>(0, 0);
    for (var j: i32 = 0; j < kernel_cols; j = j + 1) {
      if (circle_mask[kernel_idx] == 1u){
        counters[0] += processed_buffer[pic_along_row];
        counters[1] += processed_buffer[pic_along_col];
        result[2] += processed_buffer[pic_along_row];
      }
      pic_along_row += 1;
      pic_along_col += params.rows;
      kernel_idx += 1;
    }
    result[0] += counters[0] * f32(i);
    result[1] += counters[1] * f32(i);
    pic_along_row += params.cols;
    pic_along_col += 1;
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