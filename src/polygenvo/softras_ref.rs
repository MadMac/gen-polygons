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

/// Soft-composite the scene over black and return the composited linear RGB
/// (straight-alpha OVER) at pixel (px,py). `tau` is the coverage temperature.
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

/// Derivative of `srgb_to_linear` w.r.t. its argument.
fn srgb_to_linear_grad(c: f64) -> f64 {
    if c <= 0.04045 {
        1.0 / 12.92
    } else {
        (2.4 / 1.055) * ((c + 0.055) / 1.055).powf(1.4)
    }
}

/// `∂lab/∂xyz` Jacobian (3x3, row = lab channel, col = xyz channel).
fn lab_jacobian(xyz: [f64; 3]) -> [[f64; 3]; 3] {
    // Whitepoint divisors for X, Y, Z.
    let xn = [0.95047, 1.00000, 1.08883];
    // f'(t) where t = xyz_ch / wn_ch; chain in 1/wn for ∂f/∂xyz_ch.
    let mut dfx = [0.0f64; 3]; // ∂f(xyz_ch/wn)/∂xyz_ch
    for ch in 0..3 {
        let t = xyz[ch] / xn[ch];
        let fp = if t > 0.008856 {
            (1.0 / 3.0) * t.powf(-2.0 / 3.0)
        } else {
            7.787
        };
        dfx[ch] = fp / xn[ch];
    }
    // fx depends on X (idx 0), fy on Y (idx 1), fz on Z (idx 2).
    // L = 116*fy - 16  -> ∂L/∂Y = 116*dfx[1]
    // a = 500*(fx - fy) -> ∂a/∂X = 500*dfx[0], ∂a/∂Y = -500*dfx[1]
    // b = 200*(fy - fz) -> ∂b/∂Y = 200*dfx[1], ∂b/∂Z = -200*dfx[2]
    [
        [0.0, 116.0 * dfx[1], 0.0],
        [500.0 * dfx[0], -500.0 * dfx[1], 0.0],
        [0.0, 200.0 * dfx[1], -200.0 * dfx[2]],
    ]
}

/// `∂xyz/∂(linear rgb)` is the constant `linear_rgb_to_xyz` matrix (row = xyz, col = rgb).
const RGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

/// Barycentric weights plus their Jacobian w.r.t. the six vertex-position
/// scalars (v0.x, v0.y, v1.x, v1.y, v2.x, v2.y). Returns (l, dl) where
/// `dl[k][j]` = ∂l_k / ∂(pos scalar j). Degenerate triangles get zero gradient.
fn barycentric_with_grad(
    p: (f64, f64),
    v0: (f64, f64),
    v1: (f64, f64),
    v2: (f64, f64),
) -> ([f64; 3], [[f64; 6]; 3]) {
    let d = (v1.1 - v2.1) * (v0.0 - v2.0) + (v2.0 - v1.0) * (v0.1 - v2.1);
    if d.abs() < 1e-12 {
        return ([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], [[0.0; 6]; 3]);
    }
    // Numerators.
    let n0 = (v1.1 - v2.1) * (p.0 - v2.0) + (v2.0 - v1.0) * (p.1 - v2.1);
    let n1 = (v2.1 - v0.1) * (p.0 - v2.0) + (v0.0 - v2.0) * (p.1 - v2.1);
    let l0 = n0 / d;
    let l1 = n1 / d;
    let l2 = 1.0 - l0 - l1;

    // Partials of d w.r.t. (v0x, v0y, v1x, v1y, v2x, v2y).
    // d = (v1y - v2y)(v0x - v2x) + (v2x - v1x)(v0y - v2y)
    let dd = [
        v1.1 - v2.1,                       // ∂d/∂v0x
        v2.0 - v1.0,                       // ∂d/∂v0y
        -(v0.1 - v2.1),                    // ∂d/∂v1x  : from (v2x - v1x)(v0y - v2y), ∂/∂v1x = -(v0y-v2y)
        v0.0 - v2.0,                       // ∂d/∂v1y  : from (v1y - v2y)(v0x - v2x), ∂/∂v1y = (v0x-v2x)
        (v0.1 - v2.1) - (v1.1 - v2.1),     // ∂d/∂v2x  : ∂[(v1y-v2y)(v0x-v2x)]/∂v2x = -(v1y-v2y); ∂[(v2x-v1x)(v0y-v2y)]/∂v2x = (v0y-v2y)
        (v1.0 - v2.0) - (v0.0 - v2.0),     // ∂d/∂v2y  : ∂[(v1y-v2y)(v0x-v2x)]/∂v2y = -(v0x-v2x); ∂[(v2x-v1x)(v0y-v2y)]/∂v2y = -(v2x-v1x)
    ];

    // Partials of n0 = (v1y - v2y)(p.x - v2x) + (v2x - v1x)(p.y - v2y)
    let dn0 = [
        0.0,                                                   // v0x
        0.0,                                                   // v0y
        -(p.1 - v2.1),                                         // v1x : ∂(v2x - v1x)/∂v1x * (p.y - v2y)
        p.0 - v2.0,                                            // v1y : ∂(v1y - v2y)/∂v1y * (p.x - v2x)
        -(v1.1 - v2.1) + (p.1 - v2.1),                         // v2x : ∂[(v1y-v2y)(p.x-v2x)]/∂v2x = -(v1y-v2y); ∂[(v2x-v1x)(p.y-v2y)]/∂v2x = (p.y-v2y)
        -(p.0 - v2.0) - (v2.0 - v1.0),                         // v2y : ∂[(v1y-v2y)(p.x-v2x)]/∂v2y = -(p.x-v2x); ∂[(v2x-v1x)(p.y-v2y)]/∂v2y = -(v2x-v1x)
    ];

    // Partials of n1 = (v2y - v0y)(p.x - v2x) + (v0x - v2x)(p.y - v2y)
    // ∂n1/∂v0x = (p.y - v2y)
    // ∂n1/∂v0y = -(p.x - v2x)
    // ∂n1/∂v2x = (v2y - v0y)*(-1) + (-1)*(p.y - v2y) = -(v2y - v0y) - (p.y - v2y)
    // ∂n1/∂v2y = (1)*(p.x - v2x) + (v0x - v2x)*(-1)  = (p.x - v2x) - (v0x - v2x)
    let dn1 = [
        p.1 - v2.1,                          // v0x
        -(p.0 - v2.0),                       // v0y
        0.0,                                 // v1x
        0.0,                                 // v1y
        -(v2.1 - v0.1) - (p.1 - v2.1),       // v2x
        (p.0 - v2.0) - (v0.0 - v2.0),        // v2y
    ];

    let mut dl = [[0.0f64; 6]; 3];
    for j in 0..6 {
        // l0 = n0/d ; l1 = n1/d (quotient rule)
        let dl0 = (dn0[j] * d - n0 * dd[j]) / (d * d);
        let dl1 = (dn1[j] * d - n1 * dd[j]) / (d * d);
        dl[0][j] = dl0;
        dl[1][j] = dl1;
        dl[2][j] = -dl0 - dl1;
    }
    ([l0, l1, l2], dl)
}

/// Gradient of a single edge's signed distance (CCW edge a→b) w.r.t.
/// (a.x, a.y, b.x, b.y). Mirrors `edge_signed_dist`.
fn edge_signed_dist_grad(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> [f64; 4] {
    let ex = b.0 - a.0;
    let ey = b.1 - a.1;
    let len2 = ex * ex + ey * ey;
    let len = len2.sqrt();
    if len == 0.0 {
        return [0.0; 4];
    }
    // num = (-ey)(p.x - a.x) + ex(p.y - a.y)
    let num = (-ey) * (p.0 - a.0) + ex * (p.1 - a.1);
    // Partials of ex, ey: ex = b.x - a.x, ey = b.y - a.y.
    // d(.)/d(a.x): ∂ex=-1 ∂ey=0 ; d(.)/d(a.y): ∂ex=0 ∂ey=-1
    // d(.)/d(b.x): ∂ex=1  ∂ey=0 ; d(.)/d(b.y): ∂ex=0 ∂ey=1
    // num = -ey*(p.x-a.x) + ex*(p.y-a.y)
    // ∂num/∂a.x = -∂ey*(p.x-a.x) + ey*1 + ∂ex*(p.y-a.y)
    //   with ∂ex=-1,∂ey=0: = 0 + ey - (p.y-a.y) = ... compute per-var below.
    // len = sqrt(ex^2+ey^2) -> ∂len = (ex*∂ex + ey*∂ey)/len
    let dnum = [
        // a.x: ∂ex=-1, ∂ey=0 ; num = -ey*(p.x-a.x)+ex*(p.y-a.y)
        // ∂num = -(∂ey)*(p.x-a.x) - ey*∂(p.x-a.x) + (∂ex)*(p.y-a.y)
        //      = -(0)*(.) - ey*(-1) + (-1)*(p.y-a.y) = ey - (p.y - a.y)
        ey - (p.1 - a.1),
        // a.y: ∂ex=0, ∂ey=-1 ; ∂(p.y-a.y)=-1
        // ∂num = -(-1)*(p.x-a.x) - ey*0 + 0*(p.y-a.y) ... wait ex part: ∂ex=0
        //      = (p.x-a.x) - ex
        (p.0 - a.0) - ex,
        // b.x: ∂ex=1, ∂ey=0
        // ∂num = -0*(.) - 0 + 1*(p.y-a.y) = (p.y - a.y)
        p.1 - a.1,
        // b.y: ∂ex=0, ∂ey=1
        // ∂num = -1*(p.x-a.x) - 0 + 0 = -(p.x - a.x)
        -(p.0 - a.0),
    ];
    // len = sqrt(ex^2+ey^2); ∂len/∂var = (ex*∂ex + ey*∂ey)/len.
    let dlen = [
        -ex / len, // a.x: ∂ex=-1
        -ey / len, // a.y: ∂ey=-1
        ex / len,  // b.x: ∂ex=1
        ey / len,  // b.y: ∂ey=1
    ];
    let mut out = [0.0f64; 4];
    for (j, o) in out.iter_mut().enumerate() {
        *o = (dnum[j] * len - num * dlen[j]) / len2;
    }
    out
}

/// Analytic ∂(forward_loss)/∂params, same shape as `scene` (one ParamTri of
/// gradients per triangle: [d/dcx, d/dcy, d/dr, d/dg, d/db, d/da] per vertex).
pub(crate) fn grad_loss(scene: &[ParamTri], goal_lab: &[[f64; 3]], w: u32, h: u32, tau: f64) -> Vec<ParamTri> {
    let n = (w * h) as f64;
    let m = scene.len();
    let mut grad = vec![[[0.0f64; 6]; 3]; m];

    for py in 0..h {
        for px in 0..w {
            let p = pixel_to_clip(px, py, w, h);
            let goal = goal_lab[(py * w + px) as usize];

            // ---- Forward pass over triangles, storing per-triangle state. ----
            let mut c = [0.0f64; 3]; // running composited linear RGB (over black)
            // Per-triangle stored state for the backward pass.
            let mut st_src_a = vec![0.0f64; m];
            let mut st_lin = vec![[0.0f64; 3]; m];
            let mut st_below = vec![[0.0f64; 3]; m]; // C just before triangle t applied

            for (t, tri) in scene.iter().enumerate() {
                st_below[t] = c;
                let v = [(tri[0][0], tri[0][1]), (tri[1][0], tri[1][1]), (tri[2][0], tri[2][1])];
                let d = tri_signed_dist(p, &v);
                let cov = sigmoid(d / tau);
                let (l0, l1, l2) = barycentric(p, v[0], v[1], v[2]);
                let l = [l0, l1, l2];
                let mut rgb = [0.0f64; 3];
                let mut a = 0.0f64;
                for k in 0..3 {
                    rgb[0] += l[k] * tri[k][2];
                    rgb[1] += l[k] * tri[k][3];
                    rgb[2] += l[k] * tri[k][4];
                    a += l[k] * tri[k][5];
                }
                let src_a = cov * a;
                let lin = [srgb_to_linear(rgb[0]), srgb_to_linear(rgb[1]), srgb_to_linear(rgb[2])];
                st_src_a[t] = src_a;
                st_lin[t] = lin;
                for ch in 0..3 {
                    c[ch] = src_a * lin[ch] + (1.0 - src_a) * c[ch];
                }
            }
            let c_final = c;

            // ---- Suffix transmittance T_t = Π_{j>t} (1 - src_a_j). ----
            let mut suffix_t = vec![1.0f64; m];
            // Walk from the top down: T_{m-1} = 1, T_{t} = T_{t+1} * (1 - src_a_{t+1}).
            for t in (0..m).rev() {
                if t + 1 < m {
                    suffix_t[t] = suffix_t[t + 1] * (1.0 - st_src_a[t + 1]);
                }
            }

            // ---- ∂L/∂C_final (3-vector). ----
            let xyz = linear_rgb_to_xyz(c_final[0], c_final[1], c_final[2]);
            let lab = xyz_to_lab(xyz);
            // ∂L/∂lab
            let mut dl_dlab = [0.0f64; 3];
            for ch in 0..3 {
                dl_dlab[ch] = (2.0 / n) * (lab[ch] - goal[ch]);
            }
            // ∂lab/∂xyz
            let jl = lab_jacobian(xyz);
            // ∂L/∂xyz = dl_dlab^T · jl
            let mut dl_dxyz = [0.0f64; 3];
            for j in 0..3 {
                for i in 0..3 {
                    dl_dxyz[j] += dl_dlab[i] * jl[i][j];
                }
            }
            // ∂xyz/∂C = RGB_TO_XYZ ; ∂L/∂C = dl_dxyz^T · RGB_TO_XYZ
            let mut dl_dc = [0.0f64; 3];
            for j in 0..3 {
                for i in 0..3 {
                    dl_dc[j] += dl_dxyz[i] * RGB_TO_XYZ[i][j];
                }
            }

            // ---- Backprop into each triangle's params. ----
            for (t, tri) in scene.iter().enumerate() {
                let src_a = st_src_a[t];
                let lin = st_lin[t];
                let below = st_below[t];
                let tt = suffix_t[t];

                // ∂L/∂src_a_t = dot(dL/dC, T_t*(lin - C_below))
                let mut dl_dsrc_a = 0.0f64;
                for ch in 0..3 {
                    dl_dsrc_a += dl_dc[ch] * tt * (lin[ch] - below[ch]);
                }
                // ∂L/∂lin_t[ch] = dL/dC[ch] * T_t * src_a
                let mut dl_dlin = [0.0f64; 3];
                for ch in 0..3 {
                    dl_dlin[ch] = dl_dc[ch] * tt * src_a;
                }

                // Recompute the per-triangle forward locals + geometry jacobians.
                let v = [(tri[0][0], tri[0][1]), (tri[1][0], tri[1][1]), (tri[2][0], tri[2][1])];
                let (l, dl) = barycentric_with_grad(p, v[0], v[1], v[2]);
                let mut rgb = [0.0f64; 3];
                let mut a = 0.0f64;
                for k in 0..3 {
                    rgb[0] += l[k] * tri[k][2];
                    rgb[1] += l[k] * tri[k][3];
                    rgb[2] += l[k] * tri[k][4];
                    a += l[k] * tri[k][5];
                }
                let d = tri_signed_dist(p, &v);
                let cov = sigmoid(d / tau);
                let dcov_dd = cov * (1.0 - cov) / tau;

                // src_a = cov * a -> ∂src_a/∂cov = a, ∂src_a/∂a = cov
                let dl_dcov = dl_dsrc_a * a;
                let dl_da = dl_dsrc_a * cov; // ∂L/∂(interpolated alpha a)

                // lin[ch] = srgb_to_linear(rgb[ch]) ; ∂lin/∂rgb = srgb'(rgb)
                let mut dl_drgb = [0.0f64; 3];
                for ch in 0..3 {
                    dl_drgb[ch] = dl_dlin[ch] * srgb_to_linear_grad(rgb[ch]);
                }

                // --- Color comps (2,3,4): rgb[ch] = Σ_k l_k * col_{k,ch}. ---
                // --- Alpha comp (5): a = Σ_k l_k * a_k. ---
                for k in 0..3 {
                    grad[t][k][2] += dl_drgb[0] * l[k];
                    grad[t][k][3] += dl_drgb[1] * l[k];
                    grad[t][k][4] += dl_drgb[2] * l[k];
                    grad[t][k][5] += dl_da * l[k];
                }

                // --- Position comps (0,1) via two routes. ---
                // Route A: coverage edge distance (argmin edge only).
                // d_t = min over 3 edges; route gradient to the argmin edge.
                let d0 = edge_signed_dist(p, v[0], v[1]);
                let d1 = edge_signed_dist(p, v[1], v[2]);
                let d2 = edge_signed_dist(p, v[2], v[0]);
                // Identify argmin edge and its two endpoint vertex indices.
                let (edge_grad, va, vb) = if d0 <= d1 && d0 <= d2 {
                    (edge_signed_dist_grad(p, v[0], v[1]), 0usize, 1usize)
                } else if d1 <= d2 {
                    (edge_signed_dist_grad(p, v[1], v[2]), 1usize, 2usize)
                } else {
                    (edge_signed_dist_grad(p, v[2], v[0]), 2usize, 0usize)
                };
                // ∂L/∂d via coverage: dl_dcov * dcov_dd. edge_grad = ∂d/∂(a.x,a.y,b.x,b.y).
                let dl_dd = dl_dcov * dcov_dd;
                grad[t][va][0] += dl_dd * edge_grad[0];
                grad[t][va][1] += dl_dd * edge_grad[1];
                grad[t][vb][0] += dl_dd * edge_grad[2];
                grad[t][vb][1] += dl_dd * edge_grad[3];

                // Route B: barycentric weights depend on vertex positions; they
                // feed rgb (-> lin) and a (-> src_a). ∂L/∂l_k accumulated from
                // both, then routed through dl[k][j] to position scalars.
                let mut dl_dl = [0.0f64; 3]; // ∂L/∂l_k
                for k in 0..3 {
                    // via rgb: rgb[ch] += l_k * col_{k,ch}
                    dl_dl[k] += dl_drgb[0] * tri[k][2]
                        + dl_drgb[1] * tri[k][3]
                        + dl_drgb[2] * tri[k][4];
                    // via alpha: a += l_k * a_k
                    dl_dl[k] += dl_da * tri[k][5];
                }
                // dl[k][j]: j in 0..6 = (v0x,v0y,v1x,v1y,v2x,v2y).
                for vert in 0..3 {
                    for comp in 0..2 {
                        let g: f64 = (0..3).map(|k| dl_dl[k] * dl[k][vert * 2 + comp]).sum();
                        grad[t][vert][comp] += g;
                    }
                }
            }
        }
    }

    grad
}

pub(crate) struct AdamCfg { pub steps: usize, pub lr: f64, pub tau_start: f64, pub tau_end: f64 }

/// Run Adam on the scene params minimizing `forward_loss`. τ anneals
/// geometrically from tau_start to tau_end. Returns the optimized scene.
pub(crate) fn adam_polish(mut scene: Vec<ParamTri>, goal_lab: &[[f64; 3]], w: u32, h: u32, cfg: &AdamCfg) -> Vec<ParamTri> {
    let n = scene.len();
    let mut m = vec![[[0.0f64; 6]; 3]; n];
    let mut v = vec![[[0.0f64; 6]; 3]; n];
    let (b1, b2, eps) = (0.9, 0.999, 1e-8);
    for s in 0..cfg.steps {
        let frac = if cfg.steps > 1 { s as f64 / (cfg.steps - 1) as f64 } else { 0.0 };
        let tau = cfg.tau_start * (cfg.tau_end / cfg.tau_start).powf(frac);
        let g = grad_loss(&scene, goal_lab, w, h, tau);
        let t = (s + 1) as f64;
        for tri in 0..n { for vert in 0..3 { for c in 0..6 {
            let gr = g[tri][vert][c];
            m[tri][vert][c] = b1 * m[tri][vert][c] + (1.0 - b1) * gr;
            v[tri][vert][c] = b2 * v[tri][vert][c] + (1.0 - b2) * gr * gr;
            let mh = m[tri][vert][c] / (1.0 - b1.powf(t));
            let vh = v[tri][vert][c] / (1.0 - b2.powf(t));
            scene[tri][vert][c] -= cfg.lr * mh / (vh.sqrt() + eps);
            if c < 2 { scene[tri][vert][c] = scene[tri][vert][c].clamp(-1.0, 1.0); }
            else { scene[tri][vert][c] = scene[tri][vert][c].clamp(0.0, 1.0); }
        }}}
    }
    scene
}

#[cfg(test)]
mod tests {
    // The finite-difference gradient check is the task spec, kept verbatim;
    // its helper signature and explicit index loops are intentional.
    #![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    use super::*;

    /// Central finite-difference of `forward_loss` w.r.t. one scalar param.
    fn fd_grad(scene: &[ParamTri], goal_lab: &[[f64; 3]], w: u32, h: u32, tau: f64,
               tri: usize, vert: usize, comp: usize) -> f64 {
        let eps = 1e-4;
        let mut sp = scene.to_vec();
        sp[tri][vert][comp] += eps;
        let mut sm = scene.to_vec();
        sm[tri][vert][comp] -= eps;
        (forward_loss(&sp, goal_lab, w, h, tau) - forward_loss(&sm, goal_lab, w, h, tau)) / (2.0 * eps)
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let w = 12; let h = 12;
        let goal_lab: Vec<[f64; 3]> = (0..w * h).map(|_| rgb_to_lab(0.5, 0.5, 0.5)).collect();
        let scene: Vec<ParamTri> = vec![[
            [-0.4, -0.3, 0.7, 0.2, 0.6, 0.8],
            [ 0.5, -0.4, 0.2, 0.7, 0.3, 0.8],
            [ 0.1,  0.6, 0.4, 0.4, 0.9, 0.8],
        ]];
        let tau = 0.15;
        let analytic = grad_loss(&scene, &goal_lab, w, h, tau);
        for tri in 0..scene.len() {
            for vert in 0..3 {
                for comp in 0..6 {
                    let fd = fd_grad(&scene, &goal_lab, w, h, tau, tri, vert, comp);
                    let a = analytic[tri][vert][comp];
                    let scale = fd.abs().max(a.abs()).max(1e-3);
                    assert!(
                        (fd - a).abs() / scale < 1e-2,
                        "grad mismatch tri{tri} v{vert} c{comp}: analytic {a}, fd {fd}"
                    );
                }
            }
        }
    }

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
        assert!(centre[0] > 0.5, "centre should be reddish, got {centre:?}");
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

    /// ParamTri scene -> genome Vec<Vertex> (z=0). Vertex order = draw order.
    fn scene_to_genome(scene: &[ParamTri]) -> Vec<crate::genome::Vertex> {
        let mut g = Vec::with_capacity(scene.len() * 3);
        for tri in scene {
            for vrt in tri {
                g.push(crate::genome::Vertex {
                    position: [vrt[0] as f32, vrt[1] as f32, 0.0],
                    color: [vrt[2] as f32, vrt[3] as f32, vrt[4] as f32, vrt[5] as f32],
                });
            }
        }
        g
    }

    /// Bake a GoalImage to row-major f64 CIELAB (mirrors fitness.rs goal_to_lab).
    fn goal_image_to_lab_f64(goal: &crate::goal::GoalImage, w: u32, h: u32) -> Vec<[f64; 3]> {
        let _ = (w, h); // dimensions implicit in goal.pixels iterator
        let mut out = Vec::with_capacity((goal.pixels.width() * goal.pixels.height()) as usize);
        for p in goal.pixels.pixels() {
            out.push(rgb_to_lab(p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0));
        }
        out
    }

    #[test]
    fn adam_polish_lowers_loss_on_misplaced_triangle() {
        let w = 24; let h = 24;
        let goal_lab: Vec<[f64;3]> = (0..w*h).map(|_| rgb_to_lab(0.2, 0.6, 0.9)).collect();
        // Triangle with slightly wrong color near center — both position and color
        // gradients are active, so Adam converges fast.
        let scene: Vec<ParamTri> = vec![[
            [-0.4, -0.4, 0.5, 0.5, 0.5, 1.0],
            [ 0.4, -0.4, 0.5, 0.5, 0.5, 1.0],
            [ 0.0,  0.4, 0.5, 0.5, 0.5, 1.0],
        ]];
        let cfg = AdamCfg { steps: 60, lr: 0.05, tau_start: 0.3, tau_end: 0.02 };
        let before = forward_loss(&scene, &goal_lab, w, h, cfg.tau_end);
        let after_scene = adam_polish(scene, &goal_lab, w, h, &cfg);
        let after = forward_loss(&after_scene, &goal_lab, w, h, cfg.tau_end);
        assert!(after < before * 0.9, "polish should cut loss >=10%: {before} -> {after}");
    }

    #[test]
    fn milestone1_polish_improves_hard_de2000() {
        use crate::fitness::FitnessCalc;
        use crate::test_support::{init_test_wgpu, make_solid_goal};

        let size = 64u32;
        let goal = make_solid_goal(size, [50, 150, 230]);
        let (device, queue) = init_test_wgpu();
        let calc = FitnessCalc::new_for_test(device, queue, &goal, 1);

        let scene: Vec<ParamTri> = vec![[
            [-0.9, -0.9, 0.196, 0.588, 0.902, 1.0],
            [-0.6, -0.9, 0.196, 0.588, 0.902, 1.0],
            [-0.9, -0.6, 0.196, 0.588, 0.902, 1.0],
        ]];
        let goal_lab = goal_image_to_lab_f64(&goal, size, size);
        let before = calc.fitness_of(&scene_to_genome(&scene));

        let cfg = AdamCfg { steps: 80, lr: 0.05, tau_start: 0.3, tau_end: 0.02 };
        let polished = adam_polish(scene, &goal_lab, size, size, &cfg);
        let after = calc.fitness_of(&scene_to_genome(&polished));

        println!("milestone1: hard ΔE2000 fitness before={before} after={after}");
        assert!(after > before, "hard ΔE2000 fitness must improve: {before} -> {after}");
    }
}
