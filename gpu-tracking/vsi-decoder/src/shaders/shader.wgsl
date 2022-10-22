@group(0) @binding(0)
var<storage, read_write> out: array<f32>;

@group(0) @binding(1)
var<storage, read_write> a: atomic<i32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
    let i: u32 = global_id.x;
    // var idk: atomic<u32>;
    if (i % 5u == 0u ){
        let idx = atomicAdd(&a, 1);
        out[idx] = f32(i);
    }
}