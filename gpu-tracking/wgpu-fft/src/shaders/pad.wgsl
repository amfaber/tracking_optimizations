// struct Shape{
//     nrows: u32,
//     ncols: u32,
//     // stage: u32,
//     // current_dim: u32,
//     // inverse: f32,
// }

// @group(0) @binding(0)
// var<uniform> params: Shape;

struct PushConstants{
    input_nrows: u32,
    input_ncols: u32,
    output_nrows: u32,
    output_ncols: u32,
}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<vec2<f32>>;

var<push_constant> pc: PushConstants;


@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= pc.input_nrows || global_id.y >= pc.input_ncols) {
        return;
    }
    let padded_nrows = (pc.output_nrows - pc.input_nrows) / 2u;
    let padded_ncols = (pc.output_ncols - pc.input_ncols) / 2u;

    let input_idx = global_id.x * pc.input_ncols + global_id.y;
    let output_idx = (global_id.x + padded_nrows) * pc.output_ncols + (global_id.y + padded_ncols);
    output[output_idx] = vec2<f32>(input[input_idx], 0.0);

}
