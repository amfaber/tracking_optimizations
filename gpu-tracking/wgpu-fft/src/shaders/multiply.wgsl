@group(0) @binding(0)
var<storage, read> inbuffer1: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read> inbuffer2: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read_write> outbuffer: array<vec2<f32>>;

fn complex_multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let k1 = b[0] * (a[0] + a[1]);
    let k2 = a[0] * (b[1] - b[0]);
    let k3 = a[1] * (b[0] + b[1]);
    return vec2<f32>(k1 - k3, k1 + k2);
}

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= arrayLength(&outbuffer)) {
        return;
    }
    let index = global_id.x;
    outbuffer[index] = complex_multiply(inbuffer1[index], inbuffer2[index]);
}