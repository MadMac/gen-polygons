// Canonical CIE color-space primitives, prepended to fitness.wgsl and
// softraster_tiled.wgsl at pipeline-build time via `gpu::with_color_prelude`.
//
// The fitness evaluator and the differentiable soft-rasterizer must agree on
// how a colour maps to CIELAB: the gradient polish minimises Lab-MSE through
// this exact pipeline, and the elitist gate keeps a polish only if the ΔE2000
// fitness (also using this pipeline) confirms it. A single definition here
// prevents the optimiser's objective and the gate's objective from drifting
// apart. Keep in sync with the f32 Rust copies in fitness.rs.

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
