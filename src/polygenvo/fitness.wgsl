// Compute shader for fitness scoring + residual-error binning.
//
// One invocation per pixel. The goal's CIELAB is precomputed once on the CPU
// (binding 1); each invocation reads it directly, converts only the rendered
// pixel (linear-RGB → CIELAB; the render target is sRGB so textureLoad decodes
// on read), and takes the perceptual ΔE2000 distance between them. The
// normalised per-pixel error is reduced within each 8×8 workgroup in shared
// memory and added to the score accumulator with a single atomicAdd per
// workgroup (truncation once per 64 px instead of once per px). Each pixel also
// bins its error into a GRID_DIM×GRID_DIM grid for error-guided placement.

const GRID_DIM: u32 = 16u;        // MUST match ERROR_GRID_DIM in main.rs
const GRID_CELLS: u32 = 256u;     // GRID_DIM * GRID_DIM
const GRID_SCALE: f32 = 1000.0;   // grid magnitudes are used only relatively
const WG_PIXELS: u32 = 64u;       // workgroup_size 8*8
// ΔE2000 normaliser into [0,1]. ~100 is the ΔE2000 between black and white
// (ΔL'=100, S_L≈1 at L̄=50). This divisor sets sensitivity, not correctness —
// it's the one eyeball-tunable knob (lower = more sensitive to large diffs).
const DE2000_NORM: f32 = 100.0;
const PI: f32 = 3.14159265358979;

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

// Goal CIELAB, precomputed once on the CPU (see `goal_to_lab` in fitness.rs),
// row-major as `[L, a, b, 0]` per pixel. Read directly — no per-dispatch
// re-conversion of the static goal.
@group(0) @binding(1)
var<storage, read> goal_lab_buf: array<vec4<f32>>;

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

fn deg2rad(d: f32) -> f32 { return d * PI / 180.0; }

// x^7 via multiplications. CIEDE2000 needs two of these per pixel; the integer
// power avoids `pow`'s exp2/log2 pair in the dominant 512² hot path with no
// accuracy loss.
fn pow7(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    return x * x2 * x4;
}

// CIEDE2000 perceptual colour difference between two CIELAB colours
// (kL = kC = kH = 1). Standard closed form: chroma adjustment `G`, the
// C'/h' terms, the T/SL/SC/SH weights, and the RT hue-rotation interaction.
// Perceptually far more uniform than the ΔE76 it replaces, so (1+λ) selection
// favours matches that look closer to a human rather than just numerically.
fn delta_e2000(lab1: vec3<f32>, lab2: vec3<f32>) -> f32 {
    let l1 = lab1.x; let a1 = lab1.y; let b1 = lab1.z;
    let l2 = lab2.x; let a2 = lab2.y; let b2 = lab2.z;

    let c1 = sqrt(a1 * a1 + b1 * b1);
    let c2 = sqrt(a2 * a2 + b2 * b2);
    let c_bar = (c1 + c2) * 0.5;
    let c_bar7 = pow7(c_bar);
    let g = 0.5 * (1.0 - sqrt(c_bar7 / (c_bar7 + 6103515625.0))); // 25^7 = 6.103...e9

    let a1p = a1 * (1.0 + g);
    let a2p = a2 * (1.0 + g);
    let c1p = sqrt(a1p * a1p + b1 * b1);
    let c2p = sqrt(a2p * a2p + b2 * b2);

    // Hues in degrees, wrapped to [0, 360). atan2(0,0) is undefined in the
    // shader IR (GLSL.std.450 / SPIR-V), and neutral/black pixels (a=b=0 — e.g.
    // every uncovered region of a candidate render) hit exactly that. Guard it:
    // a zero-chroma colour has no meaningful hue, so pin it to 0 (the CIEDE2000
    // convention) rather than rely on driver behaviour.
    var h1p = 0.0;
    if (a1p != 0.0 || b1 != 0.0) {
        h1p = degrees(atan2(b1, a1p));
        if (h1p < 0.0) { h1p = h1p + 360.0; }
    }
    var h2p = 0.0;
    if (a2p != 0.0 || b2 != 0.0) {
        h2p = degrees(atan2(b2, a2p));
        if (h2p < 0.0) { h2p = h2p + 360.0; }
    }

    let dlp = l2 - l1;
    let dcp = c2p - c1p;

    let cp_prod = c1p * c2p;
    var dhp = 0.0;
    if (cp_prod != 0.0) {
        var diff = h2p - h1p;
        if (diff > 180.0) { diff = diff - 360.0; }
        else if (diff < -180.0) { diff = diff + 360.0; }
        dhp = diff;
    }
    let dHp = 2.0 * sqrt(cp_prod) * sin(deg2rad(dhp) * 0.5);

    let lp_bar = (l1 + l2) * 0.5;
    let cp_bar = (c1p + c2p) * 0.5;
    var hp_bar = h1p + h2p;
    if (cp_prod != 0.0) {
        if (abs(h1p - h2p) > 180.0) {
            if (hp_bar < 360.0) { hp_bar = (h1p + h2p + 360.0) * 0.5; }
            else { hp_bar = (h1p + h2p - 360.0) * 0.5; }
        } else {
            hp_bar = (h1p + h2p) * 0.5;
        }
    }

    let t = 1.0
        - 0.17 * cos(deg2rad(hp_bar - 30.0))
        + 0.24 * cos(deg2rad(2.0 * hp_bar))
        + 0.32 * cos(deg2rad(3.0 * hp_bar + 6.0))
        - 0.20 * cos(deg2rad(4.0 * hp_bar - 63.0));

    let d_theta = 30.0 * exp(-pow((hp_bar - 275.0) / 25.0, 2.0));
    let cp_bar7 = pow7(cp_bar);
    let rc = 2.0 * sqrt(cp_bar7 / (cp_bar7 + 6103515625.0));
    let lp_bar_m50_sq = (lp_bar - 50.0) * (lp_bar - 50.0);
    let sl = 1.0 + (0.015 * lp_bar_m50_sq) / sqrt(20.0 + lp_bar_m50_sq);
    let sc = 1.0 + 0.045 * cp_bar;
    let sh = 1.0 + 0.015 * cp_bar * t;
    let rt = -sin(deg2rad(2.0 * d_theta)) * rc;

    let term_l = dlp / sl;
    let term_c = dcp / sc;
    let term_h = dHp / sh;
    return sqrt(term_l * term_l + term_c * term_c + term_h * term_h + rt * term_c * term_h);
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
        // Goal Lab is precomputed (binding 1); only the rendered pixel needs
        // converting. The rendered texture is sRGB-formatted, so textureLoad
        // returns linear RGB (hardware decode).
        let goal_lab = goal_lab_buf[y * params.image_width + x].xyz;
        let rendered_rgb = textureLoad(rendered_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
        let rendered_lab = xyz_to_lab(linear_rgb_to_xyz(rendered_rgb));
        // Perceptual ΔE2000, normalised into [0,1].
        normalized = clamp(delta_e2000(goal_lab, rendered_lab) / DE2000_NORM, 0.0, 1.0);

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
