// Compute shader for fitness scoring + residual-error binning.
//
// One invocation per pixel. Each invocation reads the goal and rendered pixels
// as linear-RGB (textures are sRGB-formatted so the hardware decodes on read),
// converts each to CIELAB, and takes the ΔE76 perceptual distance. The
// normalised per-pixel error is reduced within each 8×8 workgroup in shared
// memory and added to the score accumulator with a single atomicAdd per
// workgroup (truncation once per 64 px instead of once per px). Each pixel also
// bins its error into a GRID_DIM×GRID_DIM grid for error-guided placement.

const GRID_DIM: u32 = 16u;        // MUST match ERROR_GRID_DIM in main.rs
const GRID_CELLS: u32 = 256u;     // GRID_DIM * GRID_DIM
const GRID_SCALE: f32 = 1000.0;   // grid magnitudes are used only relatively
const WG_PIXELS: u32 = 64u;       // workgroup_size 8*8

struct FitnessParams {
    image_width: u32,
    image_height: u32,
    scale: u32,   // FITNESS_SCALE
    pad1: u32,
}

struct SlotResult {
    score: atomic<u32>,
    grid: array<atomic<u32>, GRID_CELLS>,
}

@group(0) @binding(0)
var<uniform> params: FitnessParams;

@group(0) @binding(1)
var goal_texture: texture_2d<f32>;

@group(0) @binding(2)
var rendered_texture: texture_2d<f32>;

@group(0) @binding(3)
var<storage, read_write> result: SlotResult;

var<workgroup> partials: array<f32, WG_PIXELS>;

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
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let x = global_id.x;
    let y = global_id.y;
    let in_bounds = x < params.image_width && y < params.image_height;

    var normalized = 0.0;
    if (in_bounds) {
        let goal_rgb = textureLoad(goal_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
        let rendered_rgb = textureLoad(rendered_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
        let goal_lab = xyz_to_lab(linear_rgb_to_xyz(goal_rgb));
        let rendered_lab = xyz_to_lab(linear_rgb_to_xyz(rendered_rgb));
        let d = goal_lab - rendered_lab;
        let delta_e = sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        // ΔE76 between primary-saturated colours peaks ~230; normalise by 250
        // into [0,1].
        normalized = clamp(delta_e / 250.0, 0.0, 1.0);

        // Bin into the coarse error grid (cell row 0 = top of image).
        let gx = (x * GRID_DIM) / params.image_width;
        let gy = (y * GRID_DIM) / params.image_height;
        let cell = gy * GRID_DIM + gx;
        atomicAdd(&result.grid[cell], u32(normalized * GRID_SCALE));
    }

    // Workgroup reduction: sum the 64 normalised values, one atomicAdd by lane 0.
    partials[lid] = normalized;
    workgroupBarrier();
    if (lid == 0u) {
        var sum = 0.0;
        for (var i = 0u; i < WG_PIXELS; i = i + 1u) {
            sum = sum + partials[i];
        }
        atomicAdd(&result.score, u32(sum * f32(params.scale)));
    }
}
