//! CPU reference soft-rasterizer: the golden oracle for the WGSL differentiable
//! rasterizer. Forward Lab-MSE loss + analytic gradient + Adam, in plain f64 for
//! finite-difference accuracy. Test-only — the production path is on-device
//! (`gradient.rs`/`softraster.wgsl`). Mirrors the hard renderer's pixel/clip
//! mapping, color space, and OVER composite so "GPU == this" is a meaningful bar.

/// Pixel-center clip coords for pixel (px, py) in a W×H image. Row 0 = top;
/// clip space is y-up to match shader.wgsl.
pub(crate) fn pixel_to_clip(px: u32, py: u32, w: u32, h: u32) -> (f64, f64) {
    let cx = (px as f64 + 0.5) / w as f64 * 2.0 - 1.0;
    let cy = 1.0 - (py as f64 + 0.5) / h as f64 * 2.0;
    (cx, cy)
}

/// Signed distance from clip point `p` to the half-plane of CCW edge a→b,
/// positive on the interior (left) side. For a CCW triangle, the min over the
/// three edges is the signed distance to the triangle (positive inside).
pub(crate) fn edge_signed_dist(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let ex = b.0 - a.0;
    let ey = b.1 - a.1;
    let len = (ex * ex + ey * ey).sqrt();
    if len == 0.0 {
        return f64::NEG_INFINITY;
    }
    // Left-normal of edge a→b is (-ey, ex); positive for interior points (CCW).
    ((-ey) * (p.0 - a.0) + ex * (p.1 - a.1)) / len
}

/// Signed distance to a CCW triangle (positive inside).
pub(crate) fn tri_signed_dist(p: (f64, f64), v: &[(f64, f64); 3]) -> f64 {
    let d0 = edge_signed_dist(p, v[0], v[1]);
    let d1 = edge_signed_dist(p, v[1], v[2]);
    let d2 = edge_signed_dist(p, v[2], v[0]);
    d0.min(d1).min(d2)
}

/// Soft coverage approximation σ(d/τ) — approaches a hard step function as τ → 0.
pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}
fn linear_rgb_to_xyz(r: f64, g: f64, b: f64) -> [f64; 3] {
    [
        r * 0.4124564 + g * 0.3575761 + b * 0.1804375,
        r * 0.2126729 + g * 0.7151522 + b * 0.0721750,
        r * 0.0193339 + g * 0.1191920 + b * 0.9503041,
    ]
}
fn xyz_to_lab(xyz: [f64; 3]) -> [f64; 3] {
    let f = |t: f64| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let fx = f(xyz[0] / 0.95047);
    let fy = f(xyz[1] / 1.00000);
    let fz = f(xyz[2] / 1.08883);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}
fn rgb_to_lab(r: f64, g: f64, b: f64) -> [f64; 3] {
    let lin = [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)];
    xyz_to_lab(linear_rgb_to_xyz(lin[0], lin[1], lin[2]))
}

/// One triangle's optimizable params, draw order. v[i] = [cx, cy, r, g, b, a].
pub(crate) type ParamTri = [[f64; 6]; 3];

/// Barycentric coords of clip point p w.r.t. triangle vertices (v0,v1,v2).
fn barycentric(p: (f64, f64), v0: (f64, f64), v1: (f64, f64), v2: (f64, f64)) -> (f64, f64, f64) {
    let d = (v1.1 - v2.1) * (v0.0 - v2.0) + (v2.0 - v1.0) * (v0.1 - v2.1);
    if d.abs() < 1e-12 {
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }
    let l0 = ((v1.1 - v2.1) * (p.0 - v2.0) + (v2.0 - v1.0) * (p.1 - v2.1)) / d;
    let l1 = ((v2.1 - v0.1) * (p.0 - v2.0) + (v0.0 - v2.0) * (p.1 - v2.1)) / d;
    (l0, l1, 1.0 - l0 - l1)
}

/// Soft-composite the scene over black and return the linear-space premultiplied
/// RGB at pixel (px,py). `tau` is the coverage temperature.
fn forward_pixel_rgb(scene: &[ParamTri], px: u32, py: u32, w: u32, h: u32, tau: f64) -> [f64; 3] {
    let p = pixel_to_clip(px, py, w, h);
    let mut c = [0.0f64; 3]; // composited linear RGB over black
    for t in scene {
        let v = [(t[0][0], t[0][1]), (t[1][0], t[1][1]), (t[2][0], t[2][1])];
        let d = tri_signed_dist(p, &v);
        let cov = sigmoid(d / tau);
        let (l0, l1, l2) = barycentric(p, v[0], v[1], v[2]);
        // Per-vertex color, barycentric-interpolated, then to linear RGB.
        let mut rgb = [0.0f64; 3];
        let mut a = 0.0f64;
        for (k, &lk) in [l0, l1, l2].iter().enumerate() {
            rgb[0] += lk * t[k][2];
            rgb[1] += lk * t[k][3];
            rgb[2] += lk * t[k][4];
            a += lk * t[k][5];
        }
        let src_a = cov * a;
        let lin = [srgb_to_linear(rgb[0]), srgb_to_linear(rgb[1]), srgb_to_linear(rgb[2])];
        for ch in 0..3 {
            c[ch] = src_a * lin[ch] + (1.0 - src_a) * c[ch];
        }
    }
    c
}

/// Per-pixel composited color converted to CIELAB (linear-RGB -> XYZ -> Lab).
/// Exposed so the GPU forward test can compare against this oracle.
pub(crate) fn forward_pixel_lab(scene: &[ParamTri], px: u32, py: u32, w: u32, h: u32, tau: f64) -> [f64; 3] {
    let lin = forward_pixel_rgb(scene, px, py, w, h, tau);
    xyz_to_lab(linear_rgb_to_xyz(lin[0], lin[1], lin[2]))
}

/// Total Lab-MSE loss of the scene vs a goal given as row-major CIELAB per pixel
/// (matches fitness.rs' precomputed goal-Lab).
pub(crate) fn forward_loss(
    scene: &[ParamTri],
    goal_lab: &[[f64; 3]],
    w: u32,
    h: u32,
    tau: f64,
) -> f64 {
    let mut sum = 0.0;
    for py in 0..h {
        for px in 0..w {
            let lab = forward_pixel_lab(scene, px, py, w, h, tau);
            let g = goal_lab[(py * w + px) as usize];
            for ch in 0..3 {
                let dlt = lab[ch] - g[ch];
                sum += dlt * dlt;
            }
        }
    }
    sum / (w * h) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_dist_positive_inside_ccw_triangle() {
        // CCW triangle around the origin.
        let v = [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)];
        assert!(tri_signed_dist((0.0, -0.1), &v) > 0.0, "centre is inside");
        assert!(tri_signed_dist((2.0, 2.0), &v) < 0.0, "far point is outside");
    }

    #[test]
    fn coverage_is_half_on_the_edge() {
        // On the boundary the signed distance is ~0, so sigmoid(d/τ) ≈ 0.5.
        let v = [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)];
        // Midpoint of the bottom edge lies on the boundary.
        let cov = sigmoid(tri_signed_dist((0.0, -0.5), &v) / 0.01);
        assert!((cov - 0.5).abs() < 0.05, "coverage on edge ≈ 0.5, got {cov}");
    }

    #[test]
    fn pixel_to_clip_corners() {
        // Pixel (0,0) center → (-0.75, 0.75); pixel (3,3) center → (0.75, -0.75).
        let (cx, cy) = pixel_to_clip(0, 0, 4, 4);
        assert!((cx - (-0.75)).abs() < 1e-12 && (cy - 0.75).abs() < 1e-12,
                "top-left pixel center (-0.75, 0.75), got ({cx},{cy})");
        let (cx, cy) = pixel_to_clip(3, 3, 4, 4);
        assert!((cx - 0.75).abs() < 1e-12 && (cy - (-0.75)).abs() < 1e-12,
                "bottom-right pixel center (0.75, -0.75), got ({cx},{cy})");
    }

    #[test]
    fn single_triangle_covers_centre_pixels() {
        let w = 16; let h = 16;
        let tri: ParamTri = [
            [-0.8, -0.8, 0.9, 0.1, 0.1, 1.0],
            [ 0.8, -0.8, 0.9, 0.1, 0.1, 1.0],
            [ 0.0,  0.8, 0.9, 0.1, 0.1, 1.0],
        ];
        let centre = forward_pixel_rgb(&[tri], 8, 8, w, h, 0.01);
        assert!(centre[0] > 0.2, "centre should be reddish, got {centre:?}");
        let corner = forward_pixel_rgb(&[tri], 0, 0, w, h, 0.01);
        assert!(corner.iter().all(|&c| c < 0.05), "corner outside -> black, got {corner:?}");
    }

    #[test]
    fn soft_converges_toward_hard_as_tau_shrinks() {
        let v = [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)];
        let inside = (0.0, -0.1);
        let outside = (0.9, 0.9);
        let soft = sigmoid(tri_signed_dist(inside, &v) / 0.2);
        let sharp = sigmoid(tri_signed_dist(inside, &v) / 0.005);
        assert!(sharp > soft, "interior coverage sharpens toward 1 as τ shrinks");
        let sharp_out = sigmoid(tri_signed_dist(outside, &v) / 0.005);
        assert!(sharp_out < 0.01, "exterior coverage -> 0 as τ shrinks");
    }
}
