struct fftparams{
    nrows: u32,
    ncols: u32,
    stage: u32,
    current_dim: u32,
    inverse: f32,
}

@group(0) @binding(0)
var<uniform> fftparams: fftparams;

@group(0) @binding(1)
var<storage, read_write> data: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read_write> twiddles: array<vec2<f32>>;

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
    if global_id.x > fftparams.nrows || global_id.y > fftparams.ncols {
        return;
    }
    let nrows = fftparams.nrows;
    let ncols = fftparams.ncols;
    let p = fftparams.stage;
    let current_dim = fftparams.current_dim;
    let inverse = fftparams.inverse;

    var i: u32;
    var j: u32;
    var N: u32; 
    var istride: u32;
    var jpos_idx: u32;
    if current_dim == 0u{
        i = global_id.x;
        j = global_id.y;
        N = nrows;
        istride = nrows;
        jpos_idx = j;
    } else {
        i = global_id.y;
        j = global_id.x;
        N = ncols;
        istride = 1u;
        let jstride = nrows;
        jpos_idx = j*jstride;
    }


    let l = 1u << p;
    let s = l >> 1u;
    let k = i / s;
    let u = i & s - 1u;
    let q = (i * (N >> p)) & ((N >> 1u) - 1u);
    let w = twiddles[q] * inverse;
    let kl = k*l;
    let upper_butter = istride*(kl + u) + jpos_idx;
    let lower_butter = istride*(kl + u + s) + jpos_idx;
    let tau = complex_multiply(w, data[lower_butter]);
    data[lower_butter] = data[upper_butter] - tau;
    data[upper_butter] = data[upper_butter] + tau;

}

    // i + 1;

