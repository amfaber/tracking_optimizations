struct fftparams{
    nrows: u32,
    ncols: u32,
    stage: u32,
    current_dim: u32,
    inverse: f32,
}

@group(0) @binding(0)
var<uniform> params: fftparams;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<vec2<f32>>;

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.nrows || global_id.y >= params.ncols) {
        return;
    }
    let idx = global_id.x * params.ncols + global_id.y;
    output[idx] = vec2<f32>(input[idx], 0.0);
}
