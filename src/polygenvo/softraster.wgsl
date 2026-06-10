// Differentiable soft-rasterizer. FORWARD pass: per pixel, soft-composite all
// triangles over black in linear RGB, convert to CIELAB, store per-pixel Lab.
// Mirrors the CPU reference in softras_ref.rs (the equality test is the bar).

struct Params {
    width: u32,
    height: u32,
    num_tris: u32,
    tau: f32,
};
@group(0) @binding(0) var<uniform> params: Params;

// Triangle params, tightly packed f32: triangle t, vertex k, component c lives
// at index t*18 + k*6 + c. Per-vertex component layout: [cx, cy, r, g, b, a].
@group(0) @binding(1) var<storage, read> tri_params: array<f32>;

// Per-pixel output CIELAB as [L, a, b, 0], row-major (y * width + x).
@group(0) @binding(2) var<storage, read_write> out_lab: array<vec4<f32>>;

fn srgb_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) { return c / 12.92; }
    return pow((c + 0.055) / 1.055, 2.4);
}

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

fn pixel_to_clip(px: u32, py: u32) -> vec2<f32> {
    let cx = (f32(px) + 0.5) / f32(params.width) * 2.0 - 1.0;
    let cy = 1.0 - (f32(py) + 0.5) / f32(params.height) * 2.0;
    return vec2<f32>(cx, cy);
}

fn edge_sd(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let e = b - a;
    let len = length(e);
    if (len == 0.0) { return -3.4e38; }
    return ((-e.y) * (p.x - a.x) + e.x * (p.y - a.y)) / len;
}

@compute @workgroup_size(8, 8, 1)
fn forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.width || py >= params.height) { return; }
    let p = pixel_to_clip(px, py);
    var c = vec3<f32>(0.0, 0.0, 0.0);
    for (var t: u32 = 0u; t < params.num_tris; t = t + 1u) {
        let base = t * 18u;
        let v0 = vec2<f32>(tri_params[base + 0u], tri_params[base + 1u]);
        let v1 = vec2<f32>(tri_params[base + 6u], tri_params[base + 7u]);
        let v2 = vec2<f32>(tri_params[base + 12u], tri_params[base + 13u]);
        let d0 = edge_sd(p, v0, v1);
        let d1 = edge_sd(p, v1, v2);
        let d2 = edge_sd(p, v2, v0);
        let d = min(d0, min(d1, d2));
        let cov = 1.0 / (1.0 + exp(-d / params.tau));
        let det = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
        var l0 = 1.0 / 3.0;
        var l1 = 1.0 / 3.0;
        var l2 = 1.0 / 3.0;
        if (abs(det) >= 1e-12) {
            l0 = ((v1.y - v2.y) * (p.x - v2.x) + (v2.x - v1.x) * (p.y - v2.y)) / det;
            l1 = ((v2.y - v0.y) * (p.x - v2.x) + (v0.x - v2.x) * (p.y - v2.y)) / det;
            l2 = 1.0 - l0 - l1;
        }
        let r = l0 * tri_params[base + 2u]  + l1 * tri_params[base + 8u]  + l2 * tri_params[base + 14u];
        let g = l0 * tri_params[base + 3u]  + l1 * tri_params[base + 9u]  + l2 * tri_params[base + 15u];
        let b = l0 * tri_params[base + 4u]  + l1 * tri_params[base + 10u] + l2 * tri_params[base + 16u];
        let a = l0 * tri_params[base + 5u]  + l1 * tri_params[base + 11u] + l2 * tri_params[base + 17u];
        let src_a = cov * a;
        let lin = vec3<f32>(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b));
        c = src_a * lin + (1.0 - src_a) * c;
    }
    let lab = xyz_to_lab(linear_rgb_to_xyz(c));
    out_lab[py * params.width + px] = vec4<f32>(lab, 0.0);
}
