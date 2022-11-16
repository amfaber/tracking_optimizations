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
var<storage, read_write> twiddles0: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read_write> twiddles1: array<vec2<f32>>;


@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
    let pi = -3.14159265358979323846;
    if (global_id.x < arrayLength(&twiddles0)){
        let i = global_id.x;
        let N = f32(fftparams.nrows);
        // let twiddle = vec2<f32>(cos(1.), sin(1.));
        let twiddle = vec2<f32>(cos(2.*pi*f32(i)/N), sin(2.*pi*f32(i)/N));
        twiddles0[i] = twiddle;
        return;
    }
    if (global_id.x < arrayLength(&twiddles0) + arrayLength(&twiddles1)){
        let i = global_id.x - arrayLength(&twiddles0);
        let N = f32(fftparams.ncols);
        let twiddle = vec2<f32>(cos(2.*pi*f32(i)/N), sin(2.*pi*f32(i)/N));
        twiddles1[i] = twiddle;
        return;
    }
}

