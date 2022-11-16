struct Shape{
    nrows: u32,
    ncols: u32,
    // stage: u32,
    // current_dim: u32,
    // inverse: f32,
}

struct PushConstants{
    stage: u32,
    current_dim: u32,
    inverse: u32,
}


@group(0) @binding(0)
var<uniform> shape: Shape;

@group(0) @binding(1)
var<storage, read_write> data: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read> twiddles: array<vec2<f32>>;

var<push_constant> pc: PushConstants;

// @group(0) @binding(3)
// var<storage, read_write> twiddle1: array<vec2<f32>>;

fn complex_multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let k1 = b[0] * (a[0] + a[1]);
    let k2 = a[0] * (b[1] - b[0]);
    let k3 = a[1] * (b[0] + b[1]);
    return vec2<f32>(k1 - k3, k1 + k2);
}



@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
    let nrows = shape.nrows;
    let ncols = shape.ncols;
    let p = pc.stage;
    let current_dim = pc.current_dim;
    let inverse = bool(pc.inverse);

    var i: u32;
    var j: u32;
    var N: u32; 
    var M: u32;
    var istride: u32;
    var jpos_idx: u32;
    if current_dim == 0u{
        i = global_id.x;
        j = global_id.y;
        N = nrows;
        M = ncols;
        istride = ncols;
        jpos_idx = j;
    } else {
        i = global_id.y;
        j = global_id.x;
        N = ncols;
        M = nrows;
        istride = 1u;
        let jstride = nrows;
        jpos_idx = j*jstride;
    }

    let halfN = N >> 1u;

    if i >= halfN || j >= M {
        return;
    }
    // if (i != 0u){
    //     return;
    // }

    // var l: u32;
    // var q: u32;
    // var k: u32;
    
    let l = 1u << p;
    let q = (i * (N >> p)) & (halfN - 1u);
    let k = i >> (p - 1u);
    let s = l >> 1u;
    let u = i & (s - 1u);
    
    var w = twiddles[q];
    // w[1] = -w[1];

    let kl = k*l;
    let upper_butter_idx = istride*(kl + u) + jpos_idx;
    let lower_butter_idx = istride*(kl + u + s) + jpos_idx;
    let upper_butter = data[upper_butter_idx];
    let lower_butter = data[lower_butter_idx];

    let w_lower_butter = complex_multiply(w, lower_butter);
    data[upper_butter_idx] = upper_butter + w_lower_butter;
    data[lower_butter_idx] = upper_butter - w_lower_butter;


    // let l = 1u << p;
    // let k = i / s;
    // let q = (i * (N >> p)) & ((N >> 1u) - 1u);

    // if !inverse{
    // } else {
    // }

}


