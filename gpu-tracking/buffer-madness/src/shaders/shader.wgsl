@group(0) @binding(0)
var<storage, read_write> buffer: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16)
fn fill(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u = global_id.x;
    let v = global_id.y;
    let nrows = 32u;
    let ncols = 32u;
    if (u >= nrows || v >= ncols) {
        return;
    }
    buffer[u * ncols + v] = vec2<f32>(f32(u), f32(v));
}
@compute @workgroup_size(16, 16)
fn idk(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u = global_id.x;
    let v = global_id.y;
    let nrows = 32u;
    let ncols = 32u;
    if (u >= nrows || v >= ncols) {
        return;
    }
    output[u * ncols + v] = buffer[u * ncols + v][1];

}

