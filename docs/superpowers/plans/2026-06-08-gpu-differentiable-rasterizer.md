# GPU-Native Differentiable Rasterizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fast, framework-free, on-device gradient-descent polish of all triangles' positions+colors to `polygenvo`, gated by the existing hard ΔE2000 renderer, to dissolve the silhouette local-optimum ceiling.

**Architecture:** A new `gradient.rs` module owns a soft (differentiable) rasterizer + hand-rolled Adam, run periodically on the current best behind a `--gradient-polish` flag; a polish is kept only if the **real** ΔE2000 renderer confirms it beats the parent, preserving the (1+λ) no-regression guarantee. The differentiable math is first built and proven as a **CPU reference** (test-only golden oracle, finite-difference-checked), then ported to WGSL compute shaders whose only correctness bar is "matches the CPU reference."

**Tech Stack:** Rust 2024, `wgpu` 29 (WGSL compute), `bytemuck`; no new dependencies.

---

## Strategy & sequencing

Three phases, each producing testable software:

- **Phase A — CPU reference (math de-risk, no GPU shaders).** Build the soft-raster forward loss, the analytic backward, and a CPU Adam polish loop in pure Rust. Finite-difference gradient check is the correctness spec. Ends with the spec's **Milestone 1** met *on CPU*: a synthetic stuck-big-triangle scene where the polish lowers the hard ΔE2000 (scored with the real GPU `FitnessCalc`). Cheapest possible kill-switch.
- **Phase B — GPU brute-force port.** Port forward → `softraster.wgsl`, backward → same file (CAS atomic-float accumulation), Adam → `adam.wgsl`. Each GPU pass is tested for equality against the Phase-A CPU reference. Assemble `gradient::polish` (fully on-device, no CPU round-trip beyond the existing scalar-fitness readback).
- **Phase C — ES integration.** `PolishCfg` in `EsConfig`, `--gradient-polish` flag, periodic call in `run_es` behind the elitist gate. Flag-off path stays byte-for-byte identical.

**Deferred to a follow-up plan (the spec's Milestone 3):** the tile-binned production kernel. It is a pure optimization of the proven brute-force kernel, gated on the steps/sec measured in Phase C, and its correctness contract is simply "tiled GPU output == brute-force GPU output." Contract sketched at the end; not built here.

**Acceptance bar (manual, after Phase C):** `--gradient-polish` on vs off at a matched time budget shows a higher final ΔE2000 on `goal.png` **and** visibly dissolved late-stage hard-edge facets in `triangles/<timestamp>/`.

---

## File structure

- **Create `src/polygenvo/softras_ref.rs`** (`#[cfg(test)]`-gated module): the CPU reference soft-rasterizer (forward loss + analytic gradient + Adam) and its unit tests. Test-only golden oracle; nothing in production links it.
- **Create `src/polygenvo/gradient.rs`**: production module. `PolishCfg`, GPU buffers/pipelines, `gradient::polish(...)`. The only public entry point. Depends on `genome`, `fitness` (device/queue/goal-Lab + hard gate), `goal`.
- **Create `src/polygenvo/softraster.wgsl`**: forward + backward compute passes.
- **Create `src/polygenvo/adam.wgsl`**: Adam update compute pass.
- **Modify `src/polygenvo/fitness.rs`**: expose what `gradient.rs` needs to reuse — `Arc<Device>`/`Arc<Queue>` accessors, the goal-Lab storage buffer (or a getter), and a `texture_size()` (already present). Add a small `goal_lab_pub` accessor or a constructor that hands `gradient.rs` a clone of the goal-Lab buffer. Keep changes additive.
- **Modify `src/polygenvo/es.rs`**: add `PolishCfg` field to `EsConfig`; call `gradient::polish` after accepted improvements on the `every_k` stride; refresh `current_fitness`/`parent_error_grid` on a kept polish.
- **Modify `src/polygenvo/main.rs`**: `mod gradient;`, parse `--gradient-polish`, populate `cfg.polish`.
- **Modify `src/polygenvo/genome.rs`**: only if a `triangle_area`/bbox helper is needed (prefer reusing `triangle_centroid`; add `triangle_area` if a test needs it).
- **Modify `CLAUDE.md`**: document the `--gradient-polish` flag and the new module/shaders.

### Shared conventions (read before coding)

- **Vertex layout** (`genome.rs`): `Vertex { position: [f32;3], color: [f32;4] }`, `bytemuck::Pod`. A genome is `Vec<Vertex>`, every 3 = one CCW triangle, **array order = draw order** (painter's OVER).
- **Polish parameters:** per vertex we optimize `position.xy` and `color.rgba` (6 floats). `position.z` is fixed (unused by the passthrough `shader.wgsl` beyond w=1).
- **Pixel ↔ clip mapping (MUST match the hard renderer).** Framebuffer row 0 is the top; clip space is y-up (see `shader.wgsl`: `clip_position = vec4(position, 1.0)`). For pixel `(px, py)` with `py = 0` at top of an `H×W` image, the pixel-center clip coords are:
  - `cx = (f32(px) + 0.5) / f32(W) * 2.0 - 1.0`
  - `cy = 1.0 - (f32(py) + 0.5) / f32(H) * 2.0`
  The `τ→0` soft-vs-hard test (Task 2) is the guard against getting this flip wrong.
- **Color space (MUST match `fitness.rs`/`fitness.wgsl`).** Vertex colors are authored in sRGB-ish [0,1] and the render target is `Rgba8UnormSrgb`. The CPU reference reuses the exact `srgb_to_linear` → `linear_rgb_to_xyz` → `xyz_to_lab` path already in `fitness.rs` (lines 565–595). The differentiable loss is **mean squared error in CIELAB** (smooth), NOT ΔE2000.
- **Coverage:** signed distance `d(p)` to the triangle, **positive inside** (min of the three CCW edge half-plane signed distances). Soft coverage `A = sigmoid(d / τ)` → 1 inside, 0 outside, smooth in vertex positions. `τ` annealed from `tau_start` (soft) to `tau_end` (sharp) across the polish steps.
- **Composite:** over a black background, in array order: `C ← A·α·col + (1 − A·α)·C`. (Black background matches `LoadOp::Clear(BLACK)` in `fitness.rs`.)

---

## Phase A — CPU reference (math de-risk)

### Task 1: Pixel/clip mapping + signed distance + soft coverage

**Files:**
- Create: `src/polygenvo/softras_ref.rs`
- Modify: `src/polygenvo/main.rs` (add `#[cfg(test)] mod softras_ref;`)

- [ ] **Step 1: Register the module (test-only) and write the failing test**

In `main.rs`, under the existing `#[cfg(test)] mod test_support;` line, add:

```rust
#[cfg(test)]
mod softras_ref;
```

In `src/polygenvo/softras_ref.rs`:

```rust
//! CPU reference soft-rasterizer: the golden oracle for the WGSL differentiable
//! rasterizer. Forward Lab-MSE loss + analytic gradient + Adam, in plain f64 for
//! finite-difference accuracy. Test-only — the production path is on-device
//! (`gradient.rs`/`softraster.wgsl`). Mirrors the hard renderer's pixel/clip
//! mapping, color space, and OVER composite so "GPU == this" is a meaningful bar.

/// Pixel-center clip coords for pixel (px, py) in an W×H image. Row 0 = top;
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
        // Top-left pixel maps near (-1, +1); bottom-right near (+1, -1).
        let (cx, cy) = pixel_to_clip(0, 0, 4, 4);
        assert!(cx < 0.0 && cy > 0.0, "top-left -> (-,+), got ({cx},{cy})");
        let (cx, cy) = pixel_to_clip(3, 3, 4, 4);
        assert!(cx > 0.0 && cy < 0.0, "bottom-right -> (+,-), got ({cx},{cy})");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail (module not yet wired)**

Run: `cargo test --bin polygenvo softras_ref:: 2>&1 | tail -20`
Expected: compile error first (e.g. unused/incorrect), then on fix the three tests pass. If they fail on logic, fix until green.

- [ ] **Step 3: Make the tests pass**

The code above is complete; adjust only if a test fails. Run again:

Run: `cargo test --bin polygenvo softras_ref:: 2>&1 | tail -20`
Expected: `test result: ok. 3 passed`.

- [ ] **Step 4: Commit**

```bash
git add src/polygenvo/softras_ref.rs src/polygenvo/main.rs
git commit -m "test: CPU soft-raster reference — clip mapping, signed distance, coverage"
```

---

### Task 2: Forward Lab-MSE loss; soft→hard convergence

**Files:**
- Modify: `src/polygenvo/softras_ref.rs`

- [ ] **Step 1: Add the forward loss and the failing tests**

Add to `softras_ref.rs` (above the `tests` module). A "param triangle" is 3 vertices, each `(clip_x, clip_y, r, g, b, a)`; a scene is a `Vec` of them in draw order. Reuse the color-space math by re-declaring it here in f64 (mirrors `fitness.rs` exactly; keep the constants identical):

```rust
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

/// Total Lab-MSE loss of the scene vs a goal given as row-major linear-or-sRGB?
/// Goal is provided as Lab per pixel to match fitness.rs' precomputed goal-Lab.
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
            let lin = forward_pixel_rgb(scene, px, py, w, h, tau);
            // Composited RGB is linear; convert to Lab (skip the sRGB EOTF since
            // we already composited in linear space — go linear-RGB -> XYZ -> Lab).
            let lab = xyz_to_lab(linear_rgb_to_xyz(lin[0], lin[1], lin[2]));
            let g = goal_lab[(py * w + px) as usize];
            for ch in 0..3 {
                let dlt = lab[ch] - g[ch];
                sum += dlt * dlt;
            }
        }
    }
    sum / (w * h) as f64
}
```

> NOTE on color space: vertex colors are authored in the same encoding the hard renderer interpolates and then writes to an sRGB target. The hard path interpolates the *authored* values, the GPU blends, and the sRGB target stores them; on read-back `textureLoad` returns linear. To keep the reference faithful, composite per-vertex colors through `srgb_to_linear` before blending (as above), then go linear→XYZ→Lab. The `τ→0` test below validates the whole chain against intuition; the GPU-vs-CPU tests in Phase B validate it against the real renderer.

Add the tests:

```rust
#[test]
fn single_triangle_covers_centre_pixels() {
    // A big opaque red-ish triangle over black: centre pixel should be far from
    // black in Lab; a corner pixel (outside) should be ~black.
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
    // As τ shrinks, the coverage of an interior pixel -> 1 and an exterior -> 0.
    let v = [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)];
    let inside = (0.0, -0.1);
    let outside = (0.9, 0.9);
    let soft = sigmoid(tri_signed_dist(inside, &v) / 0.2);
    let sharp = sigmoid(tri_signed_dist(inside, &v) / 0.005);
    assert!(sharp > soft, "interior coverage sharpens toward 1 as τ shrinks");
    let sharp_out = sigmoid(tri_signed_dist(outside, &v) / 0.005);
    assert!(sharp_out < 0.01, "exterior coverage -> 0 as τ shrinks");
}
```

- [ ] **Step 2: Run the tests to verify they fail/then pass**

Run: `cargo test --bin polygenvo softras_ref:: 2>&1 | tail -20`
Expected: all `softras_ref` tests pass (5 total now). Fix forward math if a test fails.

- [ ] **Step 3: Commit**

```bash
git add src/polygenvo/softras_ref.rs
git commit -m "test: CPU soft-raster forward Lab-MSE loss + coverage convergence"
```

---

### Task 3: Analytic backward + finite-difference gradient check

This is the correctness keystone. The FD check fully specifies the backward — implement `grad_loss` so it passes.

**Files:**
- Modify: `src/polygenvo/softras_ref.rs`

- [ ] **Step 1: Write the FD gradient-check test FIRST**

Add to the `tests` module:

```rust
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
    // Goal: a solid mid-grey in Lab so the loss has a smooth basin.
    let goal_lab: Vec<[f64; 3]> = (0..w * h).map(|_| rgb_to_lab(0.5, 0.5, 0.5)).collect();
    // A single off-centre, semi-transparent triangle so all 18 params matter.
    let scene: Vec<ParamTri> = vec![[
        [-0.4, -0.3, 0.7, 0.2, 0.6, 0.8],
        [ 0.5, -0.4, 0.2, 0.7, 0.3, 0.8],
        [ 0.1,  0.6, 0.4, 0.4, 0.9, 0.8],
    ]];
    // Use a moderately soft τ so coverage gradients are well-conditioned for FD.
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
```

- [ ] **Step 2: Run to verify it fails (no `grad_loss` yet)**

Run: `cargo test --bin polygenvo analytic_gradient_matches 2>&1 | tail -20`
Expected: compile error — `grad_loss` not found.

- [ ] **Step 3: Implement `grad_loss` (analytic backward)**

Add `grad_loss` to `softras_ref.rs`. Derivation to implement (chain rule over the exact `forward_loss`):

- Loss `L = (1/N) Σ_pixels Σ_ch (lab_ch − goal_ch)²`. `∂L/∂lab = (2/N)(lab − goal)`.
- `lab = xyz_to_lab(linear_rgb_to_xyz(C))` where `C` is the composited **linear** RGB. Backprop `∂lab/∂C` through `xyz_to_lab` (derivative of `f(t)=t^{1/3}` is `t^{-2/3}/3` for `t>0.008856`, else `7.787`) and the constant `linear_rgb_to_xyz` matrix (its transpose).
- Composite is sequential: maintain, per pixel, the running color and, on a reverse pass over triangles, the suffix transmittance `T = Π_{j after t}(1 − src_a_j)`. `∂C/∂(src_a_t) = T·(lin_t − C_below_t)` and `∂C/∂(lin_t) = T·src_a_t`, where `C_below_t` is the composite of triangles drawn before `t`. Implement either by storing per-triangle prefix colors or by a forward store + reverse walk (simplest: for the reference, recompute prefix colors per pixel; N is tiny in tests).
- `src_a_t = cov_t · a_t`. `∂src_a/∂cov = a_t`, `∂src_a/∂a_t = cov_t`. `a_t = Σ_k l_k·a_k` ⇒ `∂a_t/∂a_k = l_k`.
- `lin_t,ch = srgb_to_linear(rgb_ch)`, `rgb_ch = Σ_k l_k·col_{k,ch}` ⇒ `∂lin/∂col_{k,ch} = srgb'(rgb_ch)·l_k`. `srgb'(c) = 1/12.92` for `c≤0.04045`, else `2.4·((c+0.055)/1.055)^{1.4}/1.055`.
- `cov_t = sigmoid(d_t/τ)` ⇒ `∂cov/∂d = cov(1−cov)/τ`. `d_t = min` over 3 edges; route the gradient to the **argmin** edge (subgradient). `∂d/∂(vertex positions)` for `edge_signed_dist(p,a,b) = ((-ey)(p.x−a.x)+ex(p.y−a.y))/len` with `ex=b.x−a.x, ey=b.y−a.y, len=|e|`: differentiate w.r.t. the 4 endpoint coords (quotient rule; `a` and `b` are two of the triangle's vertices). The edge a→b uses vertices `(vert, vert+1 mod 3)`.
- Barycentric weights `l_k` also depend on vertex positions (they appear in `rgb_t`, `a_t`). Include `∂l_k/∂(vertex xy)` from the `barycentric` closed form (quotient rule on `d` and the numerators). This term is what lets color follow geometry; include it so FD matches.

Accumulate into `grad[tri][vert][comp]` for `comp` 0,1 = position xy, 2..5 = rgba. Signature:

```rust
/// Analytic ∂(forward_loss)/∂params, same shape as `scene`.
pub(crate) fn grad_loss(
    scene: &[ParamTri],
    goal_lab: &[[f64; 3]],
    w: u32,
    h: u32,
    tau: f64,
) -> Vec<ParamTri> { /* implement per the derivation above */ }
```

Iterate the implementation until the Step-1 FD test passes. The FD test IS the spec — if a term is missing (e.g. the barycentric position derivative), the corresponding `comp 0/1` entries will mismatch and point you at it.

- [ ] **Step 4: Run the FD test to verify it passes**

Run: `cargo test --bin polygenvo analytic_gradient_matches 2>&1 | tail -20`
Expected: PASS. If position components mismatch, the missing term is almost always the argmin-edge distance derivative or the barycentric-position derivative.

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/softras_ref.rs
git commit -m "test: CPU soft-raster analytic backward, finite-difference verified"
```

---

### Task 4: CPU Adam polish loop + Milestone 1 (synthetic stuck triangle improves hard ΔE2000)

**Files:**
- Modify: `src/polygenvo/softras_ref.rs`
- Uses (test-only): `crate::fitness::FitnessCalc`, `crate::test_support::{init_test_wgpu, make_solid_goal}`, `crate::genome::Vertex`

- [ ] **Step 1: Add a CPU Adam polish over a scene**

```rust
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
            // Clamp: positions to clip [-1,1], colors/alpha to [0,1].
            if c < 2 { scene[tri][vert][c] = scene[tri][vert][c].clamp(-1.0, 1.0); }
            else { scene[tri][vert][c] = scene[tri][vert][c].clamp(0.0, 1.0); }
        }}}
    }
    scene
}
```

- [ ] **Step 2: Write the Milestone-1 test (loss drops; and hard ΔE2000 improves)**

Add a pure-CPU loss-drop test plus an integration test that scores with the real GPU renderer. First the cheap CPU assertion:

```rust
#[test]
fn adam_polish_lowers_loss_on_misplaced_triangle() {
    let w = 24; let h = 24;
    // Goal: a solid colour the triangle should grow/move to cover.
    let goal_lab: Vec<[f64;3]> = (0..w*h).map(|_| rgb_to_lab(0.2, 0.6, 0.9)).collect();
    // A small triangle parked in a corner — wrong place and too small.
    let scene: Vec<ParamTri> = vec![[
        [-0.9, -0.9, 0.2, 0.6, 0.9, 1.0],
        [-0.7, -0.9, 0.2, 0.6, 0.9, 1.0],
        [-0.9, -0.7, 0.2, 0.6, 0.9, 1.0],
    ]];
    let cfg = AdamCfg { steps: 60, lr: 0.05, tau_start: 0.3, tau_end: 0.02 };
    let before = forward_loss(&scene, &goal_lab, w, h, cfg.tau_end);
    let after_scene = adam_polish(scene, &goal_lab, w, h, &cfg);
    let after = forward_loss(&after_scene, &goal_lab, w, h, cfg.tau_end);
    assert!(after < before * 0.9, "polish should cut loss >=10%: {before} -> {after}");
}
```

Then the GPU-gated Milestone-1 test. Add helpers to convert between `ParamTri` and `Vec<Vertex>` (z=0), and to bake a `GoalImage` to f64 Lab (mirror `goal_to_lab`). Score with the real renderer:

```rust
#[test]
fn milestone1_polish_improves_hard_de2000() {
    use crate::fitness::FitnessCalc;
    use crate::genome::Vertex;
    use crate::test_support::{init_test_wgpu, make_solid_goal};

    let size = 64u32;
    let goal = make_solid_goal(size, [50, 150, 230]); // target colour
    let (device, queue) = init_test_wgpu();
    let calc = FitnessCalc::new_for_test(device, queue, &goal, 1); // see Task 5 Step 1

    // Stuck small triangle in a corner, roughly the goal colour.
    let scene: Vec<ParamTri> = vec![[
        [-0.9, -0.9, 0.196, 0.588, 0.902, 1.0],
        [-0.6, -0.9, 0.196, 0.588, 0.902, 1.0],
        [-0.9, -0.6, 0.196, 0.588, 0.902, 1.0],
    ]];
    let goal_lab = goal_image_to_lab_f64(&goal, size, size);
    let before_genome = scene_to_genome(&scene);
    let before = calc.fitness_of(&before_genome);

    let cfg = AdamCfg { steps: 80, lr: 0.05, tau_start: 0.3, tau_end: 0.02 };
    let polished = adam_polish(scene, &goal_lab, size, size, &cfg);
    let after = calc.fitness_of(&scene_to_genome(&polished));

    assert!(after > before, "hard ΔE2000 fitness must improve: {before} -> {after}");
}
```

Add the conversion helpers (in `softras_ref.rs`, `#[cfg(test)]`): `scene_to_genome(&[ParamTri]) -> Vec<Vertex>` and `goal_image_to_lab_f64(&GoalImage, w, h) -> Vec<[f64;3]>` (reuse `rgb_to_lab`, iterate `goal.pixels.pixels()`).

- [ ] **Step 3: Run the tests**

Run: `cargo test --bin polygenvo softras_ref:: 2>&1 | tail -25`
Expected: all pass, including `milestone1_polish_improves_hard_de2000`. **This is the spec's Milestone-1 kill-switch:** if the hard ΔE2000 does not improve here, stop and reconsider before any GPU work — sunk cost is minimal and the ES is untouched.

- [ ] **Step 4: Commit**

```bash
git add src/polygenvo/softras_ref.rs
git commit -m "test: CPU Adam polish — Milestone 1, hard dE2000 improves on stuck triangle"
```

---

## Phase B — GPU brute-force port

> Each GPU pass is validated by equality against the Phase-A CPU reference at f32 tolerance. Phase B introduces `gradient.rs` and the two shaders. `gradient.rs` is **not** `#[cfg(test)]` — it is the production module — but Phase B's tests drive its construction.

### Task 5: Expose reuse points in `fitness.rs`; scaffold `gradient.rs`

**Files:**
- Modify: `src/polygenvo/fitness.rs`
- Create: `src/polygenvo/gradient.rs`
- Modify: `src/polygenvo/main.rs` (`mod gradient;`)

- [ ] **Step 1: Add additive accessors to `FitnessCalc`**

In `fitness.rs`, add public-in-crate accessors (do not change existing behavior). The Milestone-1 test already referenced `FitnessCalc::new_for_test`; expose `new` for tests via a thin alias, plus device/queue/goal-Lab getters:

```rust
impl FitnessCalc {
    /// Test/inter-module constructor alias for `new` (kept crate-private).
    #[cfg(test)]
    pub(crate) fn new_for_test(
        device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, goal: &GoalImage, sample_count: u32,
    ) -> Self { Self::new(device, queue, goal, sample_count) }

    pub(crate) fn device(&self) -> &Arc<wgpu::Device> { &self.inner.device }
    pub(crate) fn queue(&self) -> &Arc<wgpu::Queue> { &self.inner.queue }
}
```

`gradient.rs` will keep its own goal-Lab storage buffer (it needs f32 `[L,a,b,_]` per pixel, exactly `goal_to_lab`'s output). Rather than thread the private buffer out, give `gradient.rs` the `GoalImage` and let it build its own goal-Lab via a small `pub(crate) fn goal_to_lab(goal: &GoalImage) -> Vec<[f32;4]>` — promote the existing private `goal_to_lab` (line 601) to `pub(crate)`. One-word change.

- [ ] **Step 2: Scaffold `gradient.rs` with `PolishCfg` and a no-op `polish`**

```rust
//! On-device differentiable-rasterizer polish: soft-raster forward+backward
//! (softraster.wgsl) + Adam (adam.wgsl) over all triangles' positions+colors,
//! minimizing Lab-MSE, then gated by the hard ΔE2000 renderer. Framework-free,
//! reuses FitnessCalc's wgpu device/queue. See
//! docs/superpowers/specs/2026-06-08-gpu-differentiable-rasterizer-design.md.

use crate::fitness::FitnessCalc;
use crate::genome::Vertex;
use crate::goal::GoalImage;

#[derive(Clone, Copy, Debug)]
pub(crate) struct PolishCfg {
    pub(crate) enabled: bool,
    pub(crate) every_k: u64,
    pub(crate) steps_n: u32,
    pub(crate) lr: f32,
    pub(crate) tau_start: f32,
    pub(crate) tau_end: f32,
}

impl Default for PolishCfg {
    fn default() -> Self {
        Self { enabled: false, every_k: 50, steps_n: 40, lr: 0.05, tau_start: 0.3, tau_end: 0.02 }
    }
}
```

In `main.rs` add `mod gradient;` after `mod genome;`.

- [ ] **Step 3: Build and run existing tests (no regressions)**

Run: `cargo test --bin polygenvo 2>&1 | tail -15 && cargo clippy --bin polygenvo 2>&1 | tail -5`
Expected: all existing + Phase-A tests pass; clippy clean.

- [ ] **Step 4: Commit**

```bash
git add src/polygenvo/fitness.rs src/polygenvo/gradient.rs src/polygenvo/main.rs
git commit -m "feat: scaffold gradient.rs (PolishCfg) + fitness.rs reuse accessors"
```

---

### Task 6: `softraster.wgsl` forward pass; GPU-forward == CPU-forward

**Files:**
- Create: `src/polygenvo/softraster.wgsl`
- Modify: `src/polygenvo/gradient.rs`

- [ ] **Step 1: Write `softraster.wgsl` forward entry point**

One invocation per pixel. Bindings: params uniform (w, h, num_tris, tau), a `read` storage buffer of triangle params (3×`vec2 pos` + `vec4 col` per triangle, or a flat `array<f32>`), the goal-Lab storage buffer, and a `read_write` storage buffer for per-pixel residual (and later the loss reduction). Mirror the CPU reference exactly: `pixel_to_clip`, `tri_signed_dist` (min of 3 CCW edge signed distances), `sigmoid`, barycentric color, `srgb_to_linear` composite over black, then linear→XYZ→Lab, write per-pixel `lab` (or per-pixel squared error). Reuse the color-matrix constants from `fitness.wgsl` verbatim.

Provide entry point:

```wgsl
@compute @workgroup_size(8, 8, 1)
fn forward(@builtin(global_invocation_id) gid: vec3<u32>) { /* per-pixel composite -> store lab/residual */ }
```

- [ ] **Step 2: Drive it from `gradient.rs` and write the equality test**

Add a `#[cfg(test)]`-only helper in `gradient.rs` that uploads a scene, runs the forward pass, reads back the per-pixel Lab, and compares to `softras_ref::forward` pixel Lab. Test:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::softras_ref::{/* ParamTri, forward_pixel_lab helpers */};
    use crate::test_support::{init_test_wgpu, make_gradient_goal};

    #[test]
    fn gpu_forward_matches_cpu_reference() {
        // Build a small scene + goal; run GPU forward; compare per-pixel Lab to
        // the CPU reference within 1e-2 (f32 vs f64 + sRGB rounding).
        // ... assert max abs Lab diff < 1e-2 ...
    }
}
```

(Expose the per-pixel Lab from the CPU reference: add `pub(crate) fn forward_pixel_lab(scene, px, py, w, h, tau) -> [f64;3]` wrapping `forward_pixel_rgb` + linear→Lab.)

- [ ] **Step 3: Run to fail, implement upload/dispatch/readback in `gradient.rs`, run to pass**

Run: `cargo test --bin polygenvo gpu_forward_matches 2>&1 | tail -20`
Expected: PASS within tolerance. If a whole-image flip appears, the pixel→clip y mapping in WGSL disagrees with the CPU reference — fix the `cy` sign.

- [ ] **Step 4: Commit**

```bash
git add src/polygenvo/softraster.wgsl src/polygenvo/gradient.rs
git commit -m "feat: softraster.wgsl forward pass, GPU==CPU reference verified"
```

---

### Task 7: `softraster.wgsl` backward pass; GPU-grad == CPU-grad

**Files:**
- Modify: `src/polygenvo/softraster.wgsl`, `src/polygenvo/gradient.rs`

- [ ] **Step 1: Add a CAS atomic-float-add helper + backward entry point in WGSL**

Core WGSL lacks atomic float add. Add the CAS helper and a per-vertex gradient buffer (`array<atomic<u32>>`, bit-cast f32):

```wgsl
fn atomic_add_f32(idx: u32, val: f32) {
    loop {
        let old_bits = atomicLoad(&grad[idx]);
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + val);
        let res = atomicCompareExchangeWeak(&grad[idx], old_bits, new_bits);
        if (res.exchanged) { break; }
    }
}
```

Backward entry: one invocation per pixel; recompute the forward composite for that pixel, then apply the chain rule from Task 3 (`∂L/∂lab` → `∂C` → per-triangle `src_a`/`lin` → `cov`/`a`/barycentric → vertex params), scattering into `grad` via `atomic_add_f32`. Zero `grad` (and the loss accumulator) with `clear_buffer` before dispatch.

```wgsl
@compute @workgroup_size(8, 8, 1)
fn backward(@builtin(global_invocation_id) gid: vec3<u32>) { /* recompute + scatter grads */ }
```

- [ ] **Step 2: Write the GPU-grad == CPU-grad test**

In `gradient.rs` tests: upload the Task-3 scene + goal, run forward then backward, read back the per-param gradient buffer, compare to `softras_ref::grad_loss`:

```rust
#[test]
fn gpu_backward_matches_cpu_reference() {
    // Same small scene as the FD test; relative error < 2e-2 per component.
    // ... assert (gpu - cpu).abs() / scale < 2e-2 ...
}
```

- [ ] **Step 3: Run to fail, implement, run to pass**

Run: `cargo test --bin polygenvo gpu_backward_matches 2>&1 | tail -20`
Expected: PASS. The CPU reference pins every term; mismatches localize to the offending term exactly as in Task 3.

- [ ] **Step 4: Commit**

```bash
git add src/polygenvo/softraster.wgsl src/polygenvo/gradient.rs
git commit -m "feat: softraster.wgsl backward pass (CAS atomic-add), GPU==CPU grads"
```

---

### Task 8: `adam.wgsl` + assemble on-device `gradient::polish`; GPU-polish == CPU-polish; the gate

**Files:**
- Create: `src/polygenvo/adam.wgsl`
- Modify: `src/polygenvo/gradient.rs`

- [ ] **Step 1: Write `adam.wgsl`**

One invocation per scalar param (6 per vertex). Bindings: params (`read_write` storage), gradient buffer, Adam moment buffers `m`,`v` (`read_write`), and a uniform with `lr`, `b1`, `b2`, `eps`, `step_t`. Apply the bias-corrected Adam update from Task 4, then clamp (positions to [-1,1] for `comp%6 < 2`, colors to [0,1] otherwise).

```wgsl
@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) { /* Adam step + clamp */ }
```

- [ ] **Step 2: Implement `gradient::polish`**

```rust
/// Polish all triangles' positions+colors via on-device soft-raster Adam, then
/// keep the result only if the hard ΔE2000 renderer confirms it beats the parent.
/// `parent_fitness` is the genome's current hard score. Returns the new hard
/// fitness if kept, else None (genome unchanged).
pub(crate) fn polish(
    genome: &mut Vec<Vertex>,
    parent_fitness: usize,
    calc: &FitnessCalc,
    goal_lab: &GoalLabBuffers,   // built once, see PolishState
    cfg: &PolishCfg,
) -> Option<usize> {
    // 1. Upload genome -> param buffer (xy + rgba per vertex; z kept aside).
    // 2. For s in 0..cfg.steps_n: clear grad; forward; backward; adam(update) with
    //    annealed tau (tau_start * (tau_end/tau_start)^(s/(N-1))) and step_t = s+1.
    //    All in one (or a few) command submits — no CPU readback inside the loop.
    // 3. Read params back; splice into a candidate genome (restore z).
    // 4. Hard-score candidate via calc.fitness_of(&candidate).
    // 5. If candidate_fit > parent_fitness: *genome = candidate; Some(candidate_fit).
    //    Else: None (genome untouched).
}
```

Cache the pipelines/buffers in a `PolishState` owned by `gradient.rs` (built once per run from `calc.device()`/`calc.queue()` + the `GoalImage`), so per-call cost is just buffer writes + dispatches. Size buffers to `MAX_VERTICES` and dispatch only the live count.

- [ ] **Step 3: Test: GPU polish lowers hard ΔE2000 on the synthetic scene (mirrors Milestone 1, now fully on-device)**

```rust
#[test]
fn gpu_polish_improves_hard_de2000() {
    // Same stuck-triangle scene as milestone1; build PolishState; call polish();
    // assert it returns Some(fit) with fit > parent_fitness and mutates genome.
}
```

Also assert the **gate**: construct a scene where the polish would *not* help (e.g. already optimal, or set `steps_n` so small nothing changes) and confirm `polish` returns `None` and leaves `genome` byte-identical.

- [ ] **Step 4: Run, implement, pass; clippy**

Run: `cargo test --bin polygenvo gpu_polish 2>&1 | tail -20 && cargo clippy --bin polygenvo 2>&1 | tail -5`
Expected: PASS; clippy clean.

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/adam.wgsl src/polygenvo/gradient.rs
git commit -m "feat: adam.wgsl + on-device gradient::polish with hard-dE2000 gate"
```

---

## Phase C — ES integration

### Task 9: Wire `--gradient-polish` into `run_es`; flag-off unchanged

**Files:**
- Modify: `src/polygenvo/es.rs`, `src/polygenvo/main.rs`, `CLAUDE.md`

- [ ] **Step 1: Add `PolishCfg` to `EsConfig` and a `polish` field**

In `es.rs`: `use crate::gradient::{polish, PolishCfg, PolishState};` (export `PolishState` from `gradient.rs`). Add `pub(crate) polish: PolishCfg` to `EsConfig`; in `production()` set `polish: PolishCfg::default()` (enabled = false). The smoke test's inline `EsConfig` gets `polish: PolishCfg::default()` too (disabled — flag-off path stays identical, smoke test untouched in behavior).

- [ ] **Step 2: Build `PolishState` once and call `polish` on the `every_k` stride**

In `run_es`, after the pyramid is built: `let mut polish_state = cfg.polish.enabled.then(|| PolishState::new(&pyramid[full_res], &goal));`. After an accepted improvement (inside `if let Some(i) = best_idx { ... }`), once per `every_k` improvements and only when enabled, run the polish against the **full-res** evaluator (the silhouette wall lives at 512²):

```rust
if cfg.polish.enabled
    && improvements_total.is_multiple_of(cfg.polish.every_k)
    && let Some(state) = polish_state.as_mut()
{
    // Re-score the parent at full res for an apples-to-apples gate.
    let (parent_full, _) = score(&pyramid[full_res], &current);
    if let Some(newfit) = polish(&mut current, parent_full, &pyramid[full_res], state, &cfg.polish) {
        // Polish kept: refresh the working fitness/grid at the *current* phase level.
        (current_fitness, parent_error_grid) = score(&pyramid[cfg.phases[schedule.idx].pyramid_level], &current);
        println!("  polish kept @ improvement {improvements_total}: full-res fit -> {newfit}");
    }
}
```

> The gate inside `polish` guarantees no regression at full res; re-scoring at the current phase level keeps `current_fitness`/`parent_error_grid` consistent with the level the ES is currently selecting at.

- [ ] **Step 3: Parse the flag in `main.rs`**

```rust
let gradient_polish = std::env::args().any(|a| a == "--gradient-polish");
// ... after `let mut cfg = es::EsConfig::production();`
cfg.polish.enabled = gradient_polish;
if gradient_polish { println!("Gradient polish enabled (every {} improvements).", cfg.polish.every_k); }
```

- [ ] **Step 4: Regression — flag-off path is unchanged**

Run: `cargo test --bin polygenvo 2>&1 | tail -15 && cargo clippy --bin polygenvo 2>&1 | tail -5`
Expected: all 24 existing tests + new tests pass; clippy clean. The smoke test (`ga_improves_on_synthetic_checker`) passes with `polish.enabled = false`, proving the flag-off path is byte-for-byte the old behavior.

- [ ] **Step 5: Document the flag in `CLAUDE.md`**

Add a bullet under the Commands section:

```
- `cargo run --release --bin polygenvo -- --gradient-polish` — every `PolishCfg.every_k`
  accepted improvements, run an on-device all-triangle gradient polish (soft-raster
  Adam in `gradient.rs`/`softraster.wgsl`/`adam.wgsl`) of vertex positions+colors,
  kept only if the hard ΔE2000 renderer confirms it beats the parent. Default off;
  composable with `--infinite`/`--show-window`.
```

Add `gradient.rs` (+ `softraster.wgsl`, `adam.wgsl`, the test-only `softras_ref.rs`) to the architecture map.

- [ ] **Step 6: Commit**

```bash
git add src/polygenvo/es.rs src/polygenvo/main.rs CLAUDE.md
git commit -m "feat: --gradient-polish wires gated all-triangle polish into run_es"
```

---

### Task 10: Manual acceptance — A/B on goal.png

**Files:** none (verification only)

- [ ] **Step 1: Baseline run (flag off), capture final fitness and a frame**

Run (cap the time with `--infinite` + manual Ctrl-C, or rely on `MAX_STEPS`):
```bash
cargo run --release --bin polygenvo 2>&1 | tee /tmp/baseline.log | tail -5
```
Note `final fitness` and the newest `triangles/<ts>/final.png`.

- [ ] **Step 2: Polish run (flag on), matched budget**

```bash
cargo run --release --bin polygenvo -- --gradient-polish 2>&1 | tee /tmp/polish.log | tail -5
```

- [ ] **Step 3: Compare**

- Confirm `--gradient-polish` final fitness ≥ baseline (the gate guarantees per-polish no-regression; the bar is a net win at matched budget).
- Eyeball both `final.png`: the polish run's large early triangles should show **dissolved hard-edge facets** vs baseline. Optionally re-run with `--show-window` to watch big triangles reshape on polish.
- Record numbers + observation in a short note appended to `docs/superpowers/specs/2026-06-08-future-directions.md` (Outcome), and update memory `final-phase-plateau-capacity-bound` if the ceiling moved.

> If polish-on does **not** beat polish-off at matched budget despite Milestone 1 passing, the likely culprit is cadence/overhead (per-polish cost starving the ES) — tune `every_k`/`steps_n`, or proceed to the deferred tile-binned kernel (below) for the speed needed at 512².

---

## Deferred follow-up plan: tile-binned production kernel (spec Milestone 3)

Not built here. Write its own plan once Task 9 measures the brute-force steps/sec overhead and Task 10 confirms a quality win is reachable. Contract:

- A binning pass assigns each triangle to the screen tiles its bbox overlaps; per-tile workgroups composite only their triangle list (z-ordered), as in tile-based forward rasterizers / 3D Gaussian Splatting.
- Backward switches from CAS scatter to **gather-per-vertex** (one thread per vertex sums over the pixels in its triangle's bbox using a per-pixel transmittance/color buffer stored in the forward pass) — no atomics, naturally sparse.
- **Correctness bar:** tile-binned forward and backward outputs match the brute-force kernel (and hence the CPU reference) within tolerance, reusing the exact equality tests from Tasks 6–7. Loss/gate/Adam are unchanged.

---

## Self-review notes

- **Spec coverage:** module boundary (Task 5), softraster forward/backward (6,7), adam (8), on-device polish + gate (8), `--gradient-polish` + `PolishCfg` + `run_es` cadence (9), Lab-MSE proxy + ΔE2000 gate (throughout), no new deps (Cargo.toml untouched), Milestone 1 (Task 4), A/B acceptance (Task 10), tile-binning deferred (documented). All spec sections map to a task.
- **No CPU round-trip in production:** the CPU reference is `#[cfg(test)]` only; `gradient::polish` keeps the optimize loop on-device (Task 8 Step 2), with only the existing scalar-fitness readback in the gate.
- **Type consistency:** `ParamTri = [[f64;6];3]`, `PolishCfg`/`PolishState`, `forward_loss`/`grad_loss`/`adam_polish`, `gradient::polish(genome, parent_fitness, calc, state, cfg) -> Option<usize>`, `FitnessCalc::{device,queue,new_for_test}` — names used consistently across tasks.
