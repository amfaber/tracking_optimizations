struct Shape{
    nrows: u32,
    ncols: u32,
}

@group(0) @binding(0)
var<uniform> shape : Shape;

@group(0) @binding(1)
var<storage, read_write> data: array<vec2<f32>>;

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
    if (global_id.x >= shape.nrows || global_id.y >= shape.ncols) {
        return;
    }
    let idx = global_id.x * shape.ncols + global_id.y;

    data[idx] /= f32(shape.nrows*shape.ncols);
}