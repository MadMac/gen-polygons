// Compute shader for fitness scoring.
//
// One invocation per pixel. Each invocation reads the goal pixel and the
// rendered pixel, computes the absolute RGB difference in linear-RGB
// (textures are sRGB-formatted so the hardware does the decode on read),
// scales to an int, and atomicAdds into a shared accumulator.
//
// The CPU then reads back one u32 and normalises into the fitness value
// the GA consumes. Lower accumulator = closer match = higher fitness.

struct FitnessParams {
    image_width: u32;
    image_height: u32;
    pad0: u32;
    pad1: u32;
};

struct FitnessResult {
    value: atomic<u32>;
};

[[group(0), binding(0)]]
var<uniform> params: FitnessParams;

[[group(0), binding(1)]]
var goal_texture: texture_2d<f32>;

[[group(0), binding(2)]]
var rendered_texture: texture_2d<f32>;

[[group(0), binding(3)]]
var<storage, read_write> fitness_result: FitnessResult;

// Per-pixel diff is in [0, 3.0] (sum of three channels in [0,1]).
// Scale to integer so atomicAdd<u32> can accumulate without floats.
// Max accumulator: image_pixels * 3 * 1000; for 256x256 that's ~196M, fits in u32.

[[stage(compute), workgroup_size(8, 8, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= params.image_width || y >= params.image_height) {
        return;
    }

    let goal = textureLoad(goal_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
    let rendered = textureLoad(rendered_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
    let diff = abs(goal.r - rendered.r) + abs(goal.g - rendered.g) + abs(goal.b - rendered.b);

    atomicAdd(&fitness_result.value, u32(diff * 1000.0));
}
