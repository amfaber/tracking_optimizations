struct Shape{
    nrows: u32,
    ncols: u32,
}

struct PushConstants{
    edge: i32,
    sigma: f32,
}

struct ParticleLocation{
    x: i32,
    y: i32,
    r: f32,
    log_space: f32,
}

@group(0) @binding(0)
var<uniform> shape: Shape;

@group(0) @binding(1)
var<uniform> params: Params;

@group(0) @binding(2)
var<storage, read> bottom: array<vec2<f32>>;

@group(0) @binding(3)
var<storage, read> middle: array<vec2<f32>>;

@group(0) @binding(4)
var<storage, read> top: array<vec2<f32>>;

@group(0) @binding(5)
var<storage, read_write> n_particles: atomic<u32>;

@group(0) @binding(6)
var<storage, read_write> particles: array<ParticleLocation>;

@group(0) @binding(7)
var<storage, read_write> global_max: atomic<i32>;

var<push_constant> pc: PushConstants;

fn is_max_in_plane(i: i32, j: i32, plane: u32, center: f32, row_transform: i32, col_transform: i32) -> bool{
    for (var loop_i = -1; loop_i < 2; loop_i++){
        for (var loop_j = -1; loop_j < 2; loop_j++){
            let x = i + loop_i;
            let y = j + loop_j;
            if (x < 0 || x >= params.pic_nrows || y >= params.pic_ncols){
                continue;
            }
            let idx = (((x + row_transform) % i32(shape.nrows)) * i32(params.pic_ncols) + (y + col_transform) % i32(shape.ncols));
            var neighbor: f32;
            if plane == 0u{
                neighbor = bottom[idx][0];
            }
            else if plane == 1u{
                neighbor = middle[idx][0];
            }
            else if plane == 2u{
                neighbor = top[idx][0];
            }
            // if neighbor - center > -0.5{
            //     return false;
            // }
            if (neighbor >= center){
                return false;
            }
        }
    }
    return true;
}

@compute @workgroup_size(_)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= u32(params.pic_nrows) || global_id.y >= u32(params.pic_ncols)) {
        return;
    }
    
    let i = i32(global_id[0]);
    let j = i32(global_id[1]);
    
    let row_transform = (i32(shape.nrows) - params.pic_nrows) / 2;
    let col_transform = (i32(shape.ncols) - params.pic_ncols) / 2;

    let center = middle[((i + row_transform) % i32(shape.nrows)) * i32(shape.ncols) + ((j + col_transform) % i32(shape.ncols))][0];
    if (center <= 1.0){
        return;
    }

    let is_max = is_max_in_plane(i, j, 1u, center, row_transform, col_transform);
    if !is_max {
        return;
    }

    if (pc.edge != -1){
        let is_max = is_max_in_plane(i, j, 0u, center, row_transform, col_transform);
        if !is_max {
            return;
        }
    }
    if (pc.edge != 1){
        let is_max = is_max_in_plane(i, j, 2u, center, row_transform, col_transform);
        if !is_max {
            return;
        }
    }

    let idx = atomicAdd(&n_particles, 1u);
    atomicMax(&global_max, i32(center));
    let sqrt2 = 1.4142135623730951;
    particles[idx] = ParticleLocation(i, j, pc.sigma*sqrt2, center);

}