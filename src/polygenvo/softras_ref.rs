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
}
