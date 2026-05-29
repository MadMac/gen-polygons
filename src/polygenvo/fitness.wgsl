// Compute shader for fitness scoring.
//
// One invocation per pixel. Each invocation reads the goal pixel and the
// rendered pixel as linear-RGB (textures are sRGB-formatted so the
// hardware does the decode on read), converts each to CIELAB, computes
// the ΔE76 perceptual distance, normalises and atomicAdds into the
// shared accumulator.
//
// Why CIELAB: ΔE76 distance in Lab space is approximately uniform in
// human perception. Summed-RGB diff over-weights bright colours and
// is blind to chroma vs. luminance imbalance. Same compute pattern,
// same single-u32 readback.

struct FitnessParams {
    image_width: u32,
    image_height: u32,
    pad0: u32,
    pad1: u32,
}

struct FitnessResult {
    value: atomic<u32>,
}

@group(0) @binding(0)
var<uniform> params: FitnessParams;

@group(0) @binding(1)
var goal_texture: texture_2d<f32>;

@group(0) @binding(2)
var rendered_texture: texture_2d<f32>;

@group(0) @binding(3)
var<storage, read_write> fitness_result: FitnessResult;

// Linear-RGB (sRGB primaries, D65) -> CIE XYZ
fn linear_rgb_to_xyz(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        rgb.r * 0.4124564 + rgb.g * 0.3575761 + rgb.b * 0.1804375,
        rgb.r * 0.2126729 + rgb.g * 0.7151522 + rgb.b * 0.0721750,
        rgb.r * 0.0193339 + rgb.g * 0.1191920 + rgb.b * 0.9503041
    );
}

// CIE XYZ (D65) -> CIELAB
fn xyz_to_lab(xyz: vec3<f32>) -> vec3<f32> {
    let xn = xyz.x / 0.95047;
    let yn = xyz.y / 1.00000;
    let zn = xyz.z / 1.08883;
    let fx = select((7.787 * xn) + (16.0 / 116.0), pow(xn, 1.0 / 3.0), xn > 0.008856);
    let fy = select((7.787 * yn) + (16.0 / 116.0), pow(yn, 1.0 / 3.0), yn > 0.008856);
    let fz = select((7.787 * zn) + (16.0 / 116.0), pow(zn, 1.0 / 3.0), zn > 0.008856);
    return vec3<f32>(
        116.0 * fy - 16.0,
        500.0 * (fx - fy),
        200.0 * (fy - fz)
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= params.image_width || y >= params.image_height) {
        return;
    }

    let goal_rgb = textureLoad(goal_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
    let rendered_rgb = textureLoad(rendered_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;

    let goal_lab = xyz_to_lab(linear_rgb_to_xyz(goal_rgb));
    let rendered_lab = xyz_to_lab(linear_rgb_to_xyz(rendered_rgb));

    let d = goal_lab - rendered_lab;
    let delta_e = sqrt(d.x * d.x + d.y * d.y + d.z * d.z);

    let normalized = clamp(delta_e / 250.0, 0.0, 1.0);
    atomicAdd(&fitness_result.value, u32(normalized * 1000.0));
}
