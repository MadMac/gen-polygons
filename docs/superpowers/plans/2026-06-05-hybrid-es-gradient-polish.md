# Hybrid ES + Gradient-Descent Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Periodically polish the ES's current-best genome with gradient descent (vertex positions + colors) through a differentiable soft rasterizer, to fix the "stuck big triangle" local optima that elitist mutation cannot.

**Architecture:** A new isolated `gradient.rs` module. Every *K* accepted ES improvements, it selects the top-*M* big/high-error triangles, hard-renders the rest as a fixed base image, soft-rasterizes the *M* triangles over that base in `burn` (autodiff), runs Adam on their positions+colors against an MSE-in-Lab loss, writes the result back, then re-scores with the exact ΔE2000 renderer and keeps the change only if it improves (elitist gate). Off by default; enabled with `--gradient-polish`.

**Tech Stack:** Rust 2024, `wgpu` (existing hard renderer), `burn` + `burn-autodiff` on the `wgpu` backend (new), `image`, `bytemuck`.

**Reference spec:** [docs/superpowers/specs/2026-06-05-hybrid-es-gradient-polish-design.md](../specs/2026-06-05-hybrid-es-gradient-polish-design.md)

> **Note on `burn` API churn:** the `burn` tensor-op names/signatures below are written against burn ~0.17 and **must be confirmed against the version pinned in Task 0.1**. The *math* (edge functions, coverage, barycentric colour, composite, Lab, MSE, Adam) is exact and version-independent; if an op name differs in the pinned version, adjust the call, not the math. The TDD steps will surface any mismatch immediately.

---

## Milestone 0 — Dependencies & burn smoke test (de-risk the toolchain)

### Task 0.1: Add `burn` and prove autodiff works on the wgpu backend

**Files:**
- Modify: `Cargo.toml`
- Create: `src/polygenvo/gradient.rs`
- Modify: `src/polygenvo/main.rs` (register the module)

- [ ] **Step 1: Add the dependency**

In `Cargo.toml` under `[dependencies]`, add (confirm the latest stable `burn` on crates.io first; pin exactly):

```toml
burn = { version = "0.17", default-features = false, features = ["wgpu", "autodiff", "std"] }
```

- [ ] **Step 2: Register the module**

In `src/polygenvo/main.rs`, add alongside the other `mod` declarations:

```rust
mod gradient;
```

- [ ] **Step 3: Write the failing smoke test**

Create `src/polygenvo/gradient.rs` with only:

```rust
//! Gradient-descent polish of the ES best via a differentiable soft rasterizer (burn).

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, Wgpu};
    use burn::tensor::Tensor;

    type B = Autodiff<Wgpu>;

    #[test]
    fn burn_autodiff_grad_of_x_squared_is_2x() {
        let device = Default::default();
        // f(x) = sum(x^2); df/dx = 2x. At x=3 -> grad 6.
        let x = Tensor::<B, 1>::from_floats([3.0], &device).require_grad();
        let y = x.clone().powf_scalar(2.0).sum();
        let grads = y.backward();
        let gx = x.grad(&grads).unwrap();
        let g: f32 = gx.into_data().to_vec::<f32>().unwrap()[0];
        assert!((g - 6.0).abs() < 1e-4, "expected grad 6, got {g}");
    }
}
```

- [ ] **Step 4: Run it; confirm it compiles and passes**

Run: `cargo test --bin polygenvo gradient::tests::burn_autodiff_grad_of_x_squared_is_2x -- --nocapture`
Expected: PASS. If op names differ (`powf_scalar`, `require_grad`, `into_data().to_vec`), fix them here and record the correct forms — every later task reuses them.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/polygenvo/main.rs src/polygenvo/gradient.rs
git commit -m "feat: add burn (wgpu+autodiff) dependency and autodiff smoke test"
```

---

## Milestone 1 — Standalone differentiable polish (THE DE-RISK GATE)

> Everything in Milestone 1 is standalone and testable without touching `run_es`. If Task 1.7 does not show the hard ΔE2000 improving, **stop here** — minimal sunk cost, ES untouched.

### Task 1.1: `triangle_area` helper

**Files:**
- Modify: `src/polygenvo/genome.rs` (add helper after `triangle_centroid`, ~line 54)
- Test: in `genome.rs` `#[cfg(test)] mod tests`

- [ ] **Step 1: Write the failing test**

In `genome.rs` tests module:

```rust
#[test]
fn triangle_area_matches_known_value() {
    // Right triangle with legs 0.3 and 0.6 -> area = 0.5*0.3*0.6 = 0.09.
    let genome = vec![
        Vertex { position: [0.0, 0.0, 0.0], color: [0.0; 4] },
        Vertex { position: [0.3, 0.0, 0.0], color: [0.0; 4] },
        Vertex { position: [0.0, 0.6, 0.0], color: [0.0; 4] },
    ];
    assert!((triangle_area(&genome, 0) - 0.09).abs() < 1e-6);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --bin polygenvo genome::tests::triangle_area_matches_known_value`
Expected: FAIL — `triangle_area` not found.

- [ ] **Step 3: Implement the helper**

In `genome.rs` after `triangle_centroid`:

```rust
/// Unsigned area of triangle `t` in clip space. Used to bias the polish subset
/// toward the big early triangles fine refinement can't reshape.
pub(crate) fn triangle_area(genome: &[Vertex], t: usize) -> f32 {
    let b = t * 3;
    let (x0, y0) = (genome[b].position[0], genome[b].position[1]);
    let (x1, y1) = (genome[b + 1].position[0], genome[b + 1].position[1]);
    let (x2, y2) = (genome[b + 2].position[0], genome[b + 2].position[1]);
    0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs()
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --bin polygenvo genome::tests::triangle_area_matches_known_value`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/genome.rs
git commit -m "feat: add triangle_area helper for polish subset selection"
```

### Task 1.2: Differentiable types and the pixel grid

**Files:**
- Modify: `src/polygenvo/gradient.rs`

This task defines the backend alias, a `PolishParams` struct holding the differentiable tensors for the subset, and a cached pixel-coordinate grid in clip space. No behavior yet beyond construction; the test asserts grid shape and corner coordinates.

- [ ] **Step 1: Write the failing test**

Add to `gradient.rs` tests:

```rust
#[test]
fn pixel_grid_has_clip_corners() {
    let device = Default::default();
    let g = super::pixel_grid::<super::B>(4, 4, &device); // (xs, ys), each [H*W]
    let xs: Vec<f32> = g.0.into_data().to_vec().unwrap();
    let ys: Vec<f32> = g.1.into_data().to_vec().unwrap();
    // Row-major H*W; pixel centres in clip space [-1,1], y flipped (row 0 = top).
    assert!((xs[0] - (-0.75)).abs() < 1e-5, "top-left x");
    assert!((ys[0] - 0.75).abs() < 1e-5, "top-left y (row 0 = top)");
    assert!((xs[15] - 0.75).abs() < 1e-5, "bottom-right x");
    assert!((ys[15] - (-0.75)).abs() < 1e-5, "bottom-right y");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --bin polygenvo gradient::tests::pixel_grid_has_clip_corners`
Expected: FAIL — `pixel_grid` not found.

- [ ] **Step 3: Implement backend alias + pixel grid**

At the top of `gradient.rs` (outside the test module):

```rust
use burn::backend::{Autodiff, Wgpu};
use burn::tensor::{Tensor, TensorData};

pub(crate) type B = Autodiff<Wgpu>;
type Device = <Wgpu as burn::tensor::backend::Backend>::Device;

/// Clip-space (x, y) coordinate of each pixel centre, row-major over H*W, with
/// row 0 = top of the image (clip y flipped) to match the fitness shader's
/// pixel→clip convention. Returns (xs, ys), each a 1-D tensor of length H*W.
fn pixel_grid<Bk: burn::tensor::backend::Backend>(
    h: usize,
    w: usize,
    device: &Bk::Device,
) -> (Tensor<Bk, 1>, Tensor<Bk, 1>) {
    let mut xs = Vec::with_capacity(h * w);
    let mut ys = Vec::with_capacity(h * w);
    for row in 0..h {
        for col in 0..w {
            let u = (col as f32 + 0.5) / w as f32; // [0,1] left->right
            let v = (row as f32 + 0.5) / h as f32; // [0,1] top->bottom
            xs.push(u * 2.0 - 1.0);
            ys.push(1.0 - v * 2.0);
        }
    }
    (
        Tensor::from_data(TensorData::new(xs, [h * w]), device),
        Tensor::from_data(TensorData::new(ys, [h * w]), device),
    )
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --bin polygenvo gradient::tests::pixel_grid_has_clip_corners`
Expected: PASS. (If `Tensor::from_data`/`TensorData::new` differ in the pinned burn, adjust to the form confirmed in Task 0.1.)

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/gradient.rs
git commit -m "feat: gradient module backend alias + clip-space pixel grid"
```

### Task 1.3: Soft coverage for one triangle (forward)

**Files:**
- Modify: `src/polygenvo/gradient.rs`

Coverage `A(p) = sigmoid(d0/τ)·sigmoid(d1/τ)·sigmoid(d2/τ)`, where `d_i` is the signed distance from pixel `p` to edge `i` of a CCW triangle (positive inside). Edge `i` from `a→b`: signed area term `e = (b.x-a.x)(p.y-a.y) - (b.y-a.y)(p.x-a.x)`; signed distance `d = e / |b-a|`. The product of three sigmoids → ~1 well inside, ~0 outside, smooth across edges, differentiable in the vertices.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn soft_coverage_inside_high_outside_low() {
    let device = Default::default();
    let (xs, ys) = super::pixel_grid::<super::B>(8, 8, &device);
    // CCW triangle covering the centre.
    let tri = [[-0.5f32, -0.5], [0.5, -0.5], [0.0, 0.5]];
    let a = super::soft_coverage::<super::B>(&xs, &ys, tri, 0.02, &device);
    let v: Vec<f32> = a.into_data().to_vec().unwrap();
    // Pixel nearest centre (clip ~0,0) is row 4 col 4 -> index 36; well inside.
    assert!(v[36] > 0.9, "centre coverage {} should be ~1", v[36]);
    // Corner pixel (top-left, index 0) is far outside.
    assert!(v[0] < 0.1, "corner coverage {} should be ~0", v[0]);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --bin polygenvo gradient::tests::soft_coverage_inside_high_outside_low`
Expected: FAIL — `soft_coverage` not found.

- [ ] **Step 3: Implement**

```rust
use burn::tensor::activation::sigmoid;

/// One edge's signed distance to every pixel: e/|b-a|, positive on the interior
/// (CCW) side. `a`, `b` are clip-space vertex positions.
fn edge_signed_dist<Bk: burn::tensor::backend::Backend>(
    xs: &Tensor<Bk, 1>,
    ys: &Tensor<Bk, 1>,
    a: [f32; 2],
    b: [f32; 2],
) -> Tensor<Bk, 1> {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
    // e = dx*(p.y-a.y) - dy*(p.x-a.x)
    let e = ys.clone().sub_scalar(a[1]).mul_scalar(dx) - xs.clone().sub_scalar(a[0]).mul_scalar(dy);
    e.div_scalar(len)
}

/// Soft coverage A(p) in [0,1] for a CCW triangle, length H*W. Differentiable in
/// the vertices via `edge_signed_dist` (kept constant-fold here; real gradient
/// flow is added in Task 1.5 where vertices are tensor params).
fn soft_coverage<Bk: burn::tensor::backend::Backend>(
    xs: &Tensor<Bk, 1>,
    ys: &Tensor<Bk, 1>,
    tri: [[f32; 2]; 3],
    tau: f32,
    _device: &Bk::Device,
) -> Tensor<Bk, 1> {
    let d0 = edge_signed_dist(xs, ys, tri[0], tri[1]);
    let d1 = edge_signed_dist(xs, ys, tri[1], tri[2]);
    let d2 = edge_signed_dist(xs, ys, tri[2], tri[0]);
    sigmoid(d0.div_scalar(tau)) * sigmoid(d1.div_scalar(tau)) * sigmoid(d2.div_scalar(tau))
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --bin polygenvo gradient::tests::soft_coverage_inside_high_outside_low`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/gradient.rs
git commit -m "feat: soft-rasteriser coverage (product of edge sigmoids)"
```

### Task 1.4: Forward render of M soft triangles over a fixed base → loss

**Files:**
- Modify: `src/polygenvo/gradient.rs`

This task introduces the *differentiable* parameterisation. The subset's positions and colors are held in one parameter tensor `params` of shape `[M*3, 6]` (per vertex: x, y, r, g, b, a). The forward:

1. For each of the M triangles, build edge distances from the *param tensor* (so gradients flow to positions), compute coverage `A` (length N=H*W), barycentric colours from the *param tensor* colours, and composite over the running image `C` (3 channels, linear RGB): `C = A·α·col + (1−A·α)·C`.
2. Initialise `C` to the fixed `base` image (constant tensor `[N,3]`).
3. Loss = mean squared error vs `goal` (`[N,3]`, linear RGB for now; Lab swapped in Task 1.6).

Barycentric colour: with edge functions `e0,e1,e2` (unnormalised, from `edge_signed_dist`×len), `λ_i = e_i / (e0+e1+e2)`; `col = λ0·c_v0 + λ1·c_v1 + λ2·c_v2` where `c_vi` are the three vertex colours. (λ for vertex i uses the edge *opposite* vertex i; index carefully per the code below.) Outside pixels have `A≈0`, so out-of-range λ are masked by the composite.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn forward_loss_drops_when_triangle_matches_goal() {
    let device = Default::default();
    let (h, w) = (16, 16);
    let n = h * w;
    // Goal: a red triangle over black; base: all black. Param triangle starts
    // grey; loss should be lower when its colour is set to red.
    let goal = vec![0.0f32; n * 3];
    let base = vec![0.0f32; n * 3];
    let tri = [[-0.5f32, -0.5], [0.5, -0.5], [0.0, 0.5]];
    let grey = super::forward_loss::<super::B>(&base, &goal, &[tri], &[[0.5; 6]; 3], 0.02, h, w, &device);
    // Goal pixels under the triangle are red:
    let mut goal_red = goal.clone();
    // (test helper fills triangle interior with red; see note) -- approximate by
    // making the whole goal red, base black, param red -> near-zero loss vs grey.
    let goal_all_red: Vec<f32> = (0..n).flat_map(|_| [1.0, 0.0, 0.0]).collect();
    let red = super::forward_loss::<super::B>(&base, &goal_all_red, &[tri], &[[1.0, 0.0, 0.0, 1.0, /*pad*/ 0.0, 0.0]; 3], 0.02, h, w, &device);
    assert!(red < grey, "matching colour should lower loss: red {red} vs grey {grey}");
}
```

> Note: the param layout per vertex is `[x, y, r, g, b, a]`; the test above passes colour in slots 2..6. Keep the helper signature taking `tris: &[[[f32;2];3]]` (positions) and `cols: &[[f32;4];3]` per triangle (RGBA) for clarity — adjust the test to that shape when implementing. The assertion (matching colour ⇒ lower loss) is the invariant that matters.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --bin polygenvo gradient::tests::forward_loss_drops_when_triangle_matches_goal`
Expected: FAIL — `forward_loss` not found.

- [ ] **Step 3: Implement the forward + loss**

```rust
/// Unnormalised edge function e = dx*(p.y-a.y) - dy*(p.x-a.x) (∝ 2·signed area).
fn edge_fn<Bk: burn::tensor::backend::Backend>(
    xs: &Tensor<Bk, 1>, ys: &Tensor<Bk, 1>, a: [f32; 2], b: [f32; 2],
) -> Tensor<Bk, 1> {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    ys.clone().sub_scalar(a[1]).mul_scalar(dx) - xs.clone().sub_scalar(a[0]).mul_scalar(dy)
}

/// Composite M soft triangles (positions `tris`, colours `cols` RGBA) over a
/// fixed `base` (linear RGB, length N*3 row-major), returning MSE vs `goal`.
/// NOTE: this first version takes plain f32 geometry/colours (no grad). Task 1.5
/// replaces the inputs with a tracked param tensor so gradients flow.
fn forward_loss<Bk: burn::tensor::backend::Backend>(
    base: &[f32], goal: &[f32],
    tris: &[[[f32; 2]; 3]], cols: &[[[f32; 4]; 3]],
    tau: f32, h: usize, w: usize, device: &Bk::Device,
) -> f32 {
    let n = h * w;
    let (xs, ys) = pixel_grid::<Bk>(h, w, device);
    // Running image, [N,3].
    let mut c = Tensor::<Bk, 2>::from_data(TensorData::new(base.to_vec(), [n, 3]), device);
    for (tri, col) in tris.iter().zip(cols.iter()) {
        let cov = soft_coverage::<Bk>(&xs, &ys, *tri, tau, device); // [N]
        // Barycentric weights from unnormalised edge functions.
        let e0 = edge_fn(&xs, &ys, tri[1], tri[2]); // opposite v0
        let e1 = edge_fn(&xs, &ys, tri[2], tri[0]); // opposite v1
        let e2 = edge_fn(&xs, &ys, tri[0], tri[1]); // opposite v2
        let denom = (e0.clone() + e1.clone() + e2.clone()).clamp(1e-6, f32::INFINITY);
        let (l0, l1, l2) = (e0 / denom.clone(), e1 / denom.clone(), e2 / denom);
        // Per-channel interpolated colour [N,3], times alpha [N].
        let alpha = cov.clone().mul_scalar(/* mean alpha */ (col[0][3] + col[1][3] + col[2][3]) / 3.0);
        let mut chans = Vec::with_capacity(3);
        for ch in 0..3 {
            let cc = l0.clone().mul_scalar(col[0][ch]) + l1.clone().mul_scalar(col[1][ch]) + l2.clone().mul_scalar(col[2][ch]);
            chans.push(cc); // [N]
        }
        let col_n = Tensor::stack::<2>(chans, 1); // [N,3]
        let a3 = alpha.clone().reshape([n, 1]); // broadcast over 3 channels
        c = col_n.mul(a3.clone()) + c.mul(a3.mul_scalar(-1.0).add_scalar(1.0));
    }
    let goal_t = Tensor::<Bk, 2>::from_data(TensorData::new(goal.to_vec(), [n, 3]), device);
    let diff = c - goal_t;
    let mse = diff.clone().mul(diff).mean();
    mse.into_data().to_vec::<f32>().unwrap()[0]
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --bin polygenvo gradient::tests::forward_loss_drops_when_triangle_matches_goal`
Expected: PASS. (Confirm `Tensor::stack`, `clamp`, `reshape`, broadcasting semantics against the pinned burn; adjust call forms if needed.)

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/gradient.rs
git commit -m "feat: soft composite of M triangles over fixed base + MSE loss"
```

### Task 1.5: Tracked params + one Adam optimisation step lowers the loss

**Files:**
- Modify: `src/polygenvo/gradient.rs`

Replace the plain-f32 geometry/colour inputs with a single tracked tensor `params: Tensor<B, 2>` of shape `[M*3, 6]` (`require_grad`). Rebuild the forward to read positions/colours by slicing `params` (so `.backward()` produces gradients for every vertex coordinate and colour). Implement **manual Adam** over the raw tensor (avoids burn's module-centric optimiser).

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn adam_step_lowers_loss_on_color_mismatch() {
    let device = Default::default();
    let (h, w) = (16, 16);
    let n = h * w;
    let base = vec![0.0f32; n * 3];
    let goal: Vec<f32> = (0..n).flat_map(|_| [1.0, 0.0, 0.0]).collect(); // all red
    let tri = [[-0.9f32, -0.9], [0.9, -0.9], [0.0, 0.9]]; // big, covers most
    // Start grey, opaque.
    let init = super::pack_params(&[tri], &[[[0.5, 0.5, 0.5, 1.0]; 3]]);
    let mut opt = super::PolishState::new(init, h, w, &device);
    let l0 = opt.loss(&base, &goal, 0.02);
    for _ in 0..30 { opt.adam_step(&base, &goal, 0.02, 0.05); }
    let l1 = opt.loss(&base, &goal, 0.02);
    assert!(l1 < l0 * 0.5, "Adam should cut loss substantially: {l0} -> {l1}");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --bin polygenvo gradient::tests::adam_step_lowers_loss_on_color_mismatch`
Expected: FAIL — `pack_params` / `PolishState` not found.

- [ ] **Step 3: Implement tracked params + manual Adam**

```rust
/// Flatten triangles into the param row layout [M*3, 6] = per vertex [x,y,r,g,b,a].
pub(crate) fn pack_params(tris: &[[[f32; 2]; 3]], cols: &[[[f32; 4]; 3]]) -> Vec<f32> {
    let mut v = Vec::with_capacity(tris.len() * 3 * 6);
    for (tri, col) in tris.iter().zip(cols.iter()) {
        for k in 0..3 {
            v.extend_from_slice(&[tri[k][0], tri[k][1], col[k][0], col[k][1], col[k][2], col[k][3]]);
        }
    }
    v
}

/// Holds the tracked params and Adam moments for one polish.
pub(crate) struct PolishState {
    params: Tensor<B, 2>, // [M*3, 6], require_grad
    m: Tensor<B, 2>,
    v: Tensor<B, 2>,
    t: i32,
    n_tris: usize,
    h: usize,
    w: usize,
    device: Device,
}

impl PolishState {
    pub(crate) fn new(flat: Vec<f32>, h: usize, w: usize, device: &Device) -> Self {
        let rows = flat.len() / 6;
        let params = Tensor::<B, 2>::from_data(TensorData::new(flat, [rows, 6]), device).require_grad();
        let m = params.clone().detach().mul_scalar(0.0);
        let v = m.clone();
        Self { params, m, v, t: 0, n_tris: rows / 3, h, w, device: device.clone() }
    }

    /// Differentiable forward → scalar loss tensor (reads geometry/colour from `params`).
    fn loss_tensor(&self, base: &[f32], goal: &[f32], tau: f32) -> Tensor<B, 1> {
        let n = self.h * self.w;
        let (xs, ys) = pixel_grid::<B>(self.h, self.w, &self.device);
        let mut c = Tensor::<B, 2>::from_data(TensorData::new(base.to_vec(), [n, 3]), &self.device);
        for ti in 0..self.n_tris {
            // Slice the 3 vertex rows for triangle ti: rows [ti*3 .. ti*3+3].
            let rows = self.params.clone().slice([(ti * 3)..(ti * 3 + 3), 0..6]); // [3,6]
            // Pull scalar positions for the (constant-per-pixel) edge geometry by
            // reading via tensor ops so grads flow: build edge distances from the
            // param rows. Implement edge fns directly on param scalars:
            let p = rows; // [3,6]
            // Helper closures over param rows -> per-pixel tensors:
            let coord = |idx: usize, comp: usize| p.clone().slice([idx..idx + 1, comp..comp + 1]).reshape([1]);
            let (ax, ay) = (coord(0, 0), coord(0, 1));
            let (bx, by) = (coord(1, 0), coord(1, 1));
            let (cx, cy) = (coord(2, 0), coord(2, 1));
            let cov = soft_coverage_t(&xs, &ys, &ax, &ay, &bx, &by, &cx, &cy, tau);
            let (l0, l1, l2) = bary_t(&xs, &ys, &ax, &ay, &bx, &by, &cx, &cy);
            let alpha_mean = (coord(0, 5) + coord(1, 5) + coord(2, 5)).div_scalar(3.0); // [1]
            let alpha = cov.mul(alpha_mean.unsqueeze_dim(0).repeat_dim(0, n)); // [N]
            let mut chans = Vec::with_capacity(3);
            for ch in 0..3 {
                let comp = 2 + ch;
                let cc = l0.clone().mul(coord(0, comp).unsqueeze_dim(0).repeat_dim(0, n))
                    + l1.clone().mul(coord(1, comp).unsqueeze_dim(0).repeat_dim(0, n))
                    + l2.clone().mul(coord(2, comp).unsqueeze_dim(0).repeat_dim(0, n));
                chans.push(cc);
            }
            let col_n = Tensor::stack::<2>(chans, 1); // [N,3]
            let a3 = alpha.reshape([n, 1]);
            c = col_n.mul(a3.clone()) + c.mul(a3.mul_scalar(-1.0).add_scalar(1.0));
        }
        let goal_t = Tensor::<B, 2>::from_data(TensorData::new(goal.to_vec(), [n, 3]), &self.device);
        let diff = c - goal_t;
        diff.clone().mul(diff).mean()
    }

    pub(crate) fn loss(&self, base: &[f32], goal: &[f32], tau: f32) -> f32 {
        self.loss_tensor(base, goal, tau).into_data().to_vec::<f32>().unwrap()[0]
    }

    /// One manual Adam step (lr fixed; betas 0.9/0.999, eps 1e-8).
    pub(crate) fn adam_step(&mut self, base: &[f32], goal: &[f32], tau: f32, lr: f32) {
        self.t += 1;
        let loss = self.loss_tensor(base, goal, tau);
        let grads = loss.backward();
        let g = self.params.grad(&grads).expect("params grad");
        let g = Tensor::<B, 2>::from_inner(g); // lift inner grad to autodiff tensor for math
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        self.m = self.m.clone().mul_scalar(b1) + g.clone().mul_scalar(1.0 - b1);
        self.v = self.v.clone().mul_scalar(b2) + g.clone().mul(g).mul_scalar(1.0 - b2);
        let mhat = self.m.clone().div_scalar(1.0 - b1.powi(self.t));
        let vhat = self.v.clone().div_scalar(1.0 - b2.powi(self.t));
        let update = mhat.div(vhat.sqrt().add_scalar(eps)).mul_scalar(lr);
        // Detach to make the updated params a fresh leaf, then re-track.
        self.params = (self.params.clone().detach() - update.detach()).require_grad();
    }

    /// Read the optimised params back out as flat [M*3*6] f32.
    pub(crate) fn to_flat(&self) -> Vec<f32> {
        self.params.clone().detach().into_data().to_vec::<f32>().unwrap()
    }
}
```

Add the two tensor-valued helpers `soft_coverage_t` and `bary_t` (edge distance / barycentric built from per-vertex scalar tensors so gradients reach positions). They mirror Task 1.3/1.4 math but take the vertex coordinate tensors instead of f32:

```rust
fn edge_dist_t(
    xs: &Tensor<B, 1>, ys: &Tensor<B, 1>,
    ax: &Tensor<B, 1>, ay: &Tensor<B, 1>, bx: &Tensor<B, 1>, by: &Tensor<B, 1>,
) -> Tensor<B, 1> {
    let n = xs.dims()[0];
    let dx = bx.clone().sub(ax.clone()).repeat_dim(0, n);
    let dy = by.clone().sub(ay.clone()).repeat_dim(0, n);
    let pax = xs.clone().sub(ax.clone().repeat_dim(0, n));
    let pay = ys.clone().sub(ay.clone().repeat_dim(0, n));
    // e = dx*(p.y-a.y) - dy*(p.x-a.x); not length-normalised (tau absorbs scale).
    dy.clone().mul(pax).neg().add(dx.mul(pay))
}

fn soft_coverage_t(
    xs: &Tensor<B, 1>, ys: &Tensor<B, 1>,
    ax: &Tensor<B, 1>, ay: &Tensor<B, 1>, bx: &Tensor<B, 1>, by: &Tensor<B, 1>,
    cx: &Tensor<B, 1>, cy: &Tensor<B, 1>, tau: f32,
) -> Tensor<B, 1> {
    let d0 = edge_dist_t(xs, ys, ax, ay, bx, by);
    let d1 = edge_dist_t(xs, ys, bx, by, cx, cy);
    let d2 = edge_dist_t(xs, ys, cx, cy, ax, ay);
    sigmoid(d0.div_scalar(tau)) * sigmoid(d1.div_scalar(tau)) * sigmoid(d2.div_scalar(tau))
}

fn bary_t(
    xs: &Tensor<B, 1>, ys: &Tensor<B, 1>,
    ax: &Tensor<B, 1>, ay: &Tensor<B, 1>, bx: &Tensor<B, 1>, by: &Tensor<B, 1>,
    cx: &Tensor<B, 1>, cy: &Tensor<B, 1>,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    let e0 = edge_dist_t(xs, ys, bx, by, cx, cy); // opposite v0
    let e1 = edge_dist_t(xs, ys, cx, cy, ax, ay); // opposite v1
    let e2 = edge_dist_t(xs, ys, ax, ay, bx, by); // opposite v2
    let denom = (e0.clone() + e1.clone() + e2.clone()).clamp(1e-6, f32::INFINITY);
    (e0 / denom.clone(), e1 / denom.clone(), e2 / denom)
}
```

> The two-path duplication (f32 helpers from Task 1.3/1.4 vs the tensor helpers here) is intentional during build-up; after Task 1.5 passes, delete the now-unused f32 `soft_coverage`/`forward_loss`/`edge_fn`/`edge_signed_dist` to keep one path. Do that deletion as part of this task's Step 5.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --bin polygenvo gradient::tests::adam_step_lowers_loss_on_color_mismatch`
Expected: PASS. Then delete the dead f32-only helpers and run `cargo clippy --bin polygenvo` clean.

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/gradient.rs
git commit -m "feat: tracked soft-raster params + manual Adam polish step"
```

### Task 1.6: MSE loss in CIELAB (match the ES metric direction)

**Files:**
- Modify: `src/polygenvo/gradient.rs`

Swap the linear-RGB MSE for MSE in CIELAB so the polish optimises the same perceptual quantity the ES selects on. Add a differentiable `linear_rgb_to_lab_t([N,3]) -> [N,3]` (same matrix/constants as `fitness.rs`/`fitness.wgsl`; the Lab cube-root is `powf_scalar(1.0/3.0)` guarded for the small-value branch). The `goal` passed in is already Lab (the caller will pass `goal_lab`); convert only the rendered `c` to Lab before the diff.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn lab_of_mid_grey_is_neutral() {
    let device = Default::default();
    // Linear-RGB mid grey -> a,b ~ 0, L ~ 50-76. Just assert a,b near 0.
    let g = vec![0.21f32, 0.21, 0.21];
    let lab = super::linear_rgb_to_lab_t(
        burn::tensor::Tensor::<super::B, 2>::from_data(
            burn::tensor::TensorData::new(g, [1, 3]), &device));
    let v: Vec<f32> = lab.into_data().to_vec().unwrap();
    assert!(v[1].abs() < 1.0 && v[2].abs() < 1.0, "neutral grey should have a,b ~ 0, got {v:?}");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --bin polygenvo gradient::tests::lab_of_mid_grey_is_neutral`
Expected: FAIL — `linear_rgb_to_lab_t` not found.

- [ ] **Step 3: Implement Lab conversion + switch the loss**

```rust
/// Differentiable linear-RGB [N,3] -> CIELAB [N,3] (same constants as fitness.wgsl).
pub(crate) fn linear_rgb_to_lab_t(rgb: Tensor<B, 2>) -> Tensor<B, 2> {
    let n = rgb.dims()[0];
    let r = rgb.clone().slice([0..n, 0..1]);
    let g = rgb.clone().slice([0..n, 1..2]);
    let b = rgb.slice([0..n, 2..3]);
    let x = r.clone().mul_scalar(0.4124564) + g.clone().mul_scalar(0.3575761) + b.clone().mul_scalar(0.1804375);
    let y = r.clone().mul_scalar(0.2126729) + g.clone().mul_scalar(0.7151522) + b.clone().mul_scalar(0.0721750);
    let z = r.mul_scalar(0.0193339) + g.mul_scalar(0.119_192) + b.mul_scalar(0.9503041);
    // f(t): t>0.008856 ? t^(1/3) : 7.787t + 16/116. Blend via a mask to stay differentiable.
    let f = |t: Tensor<B, 2>| {
        let cube = t.clone().clamp(1e-6, f32::INFINITY).powf_scalar(1.0 / 3.0);
        let lin = t.clone().mul_scalar(7.787).add_scalar(16.0 / 116.0);
        let mask = t.greater_elem(0.008856).float(); // 1.0 where cube branch
        cube.mul(mask.clone()) + lin.mul(mask.mul_scalar(-1.0).add_scalar(1.0))
    };
    let fx = f(x.div_scalar(0.95047));
    let fy = f(y.div_scalar(1.0));
    let fz = f(z.div_scalar(1.08883));
    let l = fy.clone().mul_scalar(116.0).sub_scalar(16.0);
    let a = (fx - fy.clone()).mul_scalar(500.0);
    let bb = (fy - fz).mul_scalar(200.0);
    Tensor::cat(vec![l, a, bb], 1) // [N,3]
}
```

In `loss_tensor`, after building `c` (linear RGB `[N,3]`), replace the diff line with:

```rust
let c_lab = linear_rgb_to_lab_t(c);
let goal_t = Tensor::<B, 2>::from_data(TensorData::new(goal.to_vec(), [n, 3]), &self.device); // goal is LAB
let diff = c_lab - goal_t;
diff.clone().mul(diff).mean()
```

Update the Task 1.5 test's `goal` to Lab, or keep its linear-RGB assertion lenient (matching colour still lowers Lab loss). Adjust the 1.5 test if needed so it still passes.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --bin polygenvo gradient::tests::lab_of_mid_grey_is_neutral gradient::tests::adam_step_lowers_loss_on_color_mismatch`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/gradient.rs
git commit -m "feat: CIELAB MSE loss for the polish (matches ES metric)"
```

### Task 1.7: THE GATE — synthetic misplaced-big-triangle, polish lowers hard ΔE2000

**Files:**
- Modify: `src/polygenvo/gradient.rs`
- Uses: `crate::fitness::FitnessCalc`, `crate::test_support::{init_test_wgpu, make_*_goal}`

End-to-end on the *real* hard renderer: construct a goal with a clear shape, a genome whose one big triangle is deliberately misplaced, render the base (everything else) with the hard renderer, run the polish on the big triangle, splice it back, and assert the **hard ΔE2000 score improves**. This is the stop/go gate from the spec.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn polish_improves_hard_fitness_on_misplaced_triangle() {
    use crate::fitness::FitnessCalc;
    use crate::test_support::{init_test_wgpu, make_solid_goal};
    let (device, queue) = init_test_wgpu();
    // Goal: solid teal. A single big triangle that should cover the canvas in
    // teal but is mis-coloured (red). Polishing its colour must raise fitness.
    let goal = make_solid_goal(64, [0, 128, 128]);
    let calc = FitnessCalc::new_for_test(device, queue, &goal, 1); // see note
    let mut genome = vec![
        crate::genome::Vertex { position: [-1.0, -1.0, 0.0], color: [1.0, 0.0, 0.0, 1.0] },
        crate::genome::Vertex { position: [3.0, -1.0, 0.0], color: [1.0, 0.0, 0.0, 1.0] },
        crate::genome::Vertex { position: [-1.0, 3.0, 0.0], color: [1.0, 0.0, 0.0, 1.0] },
    ];
    let before = calc.fitness_of(&genome);
    let kept = super::polish(&mut genome, &calc, &super::PolishCfg::for_test());
    let after = calc.fitness_of(&genome);
    assert!(kept, "polish should be kept");
    assert!(after > before, "hard fitness must improve: {before} -> {after}");
}
```

> Note: `polish` (Task 2.x signature) and `FitnessCalc::new_for_test` / `goal_lab` access are introduced here for the standalone test. To keep Milestone 1 self-contained, implement a minimal `polish(genome, calc, cfg)` now that: (1) selects the single largest triangle as the subset, (2) gets the base by hard-rendering the genome minus that triangle (Task 2.3 render-to-buffer — pull that helper forward to here), (3) gets `goal_lab` from `calc`, (4) optimises, writes back, and (5) re-scores and reverts unless improved. Milestone 2 generalises subset size and wires cadence.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --bin polygenvo gradient::tests::polish_improves_hard_fitness_on_misplaced_triangle`
Expected: FAIL — `polish` / `PolishCfg` / `render_to_buffer` not found.

- [ ] **Step 3: Implement `render_to_buffer`, `goal_lab` access, and a minimal `polish`**

In `fitness.rs`, refactor `snapshot` to share an in-memory render, and expose the goal Lab + a single-genome fitness already present (`fitness_of`). Add:

```rust
// fitness.rs (impl FitnessCalc)
/// Render `vertices` to an in-memory linear-RGB f32 buffer [H*W*3], row-major.
/// Shares the render+readback path with `snapshot` (factor the common code into
/// a private `render_rgba8(&self, &[Vertex]) -> Vec<u8>` and convert sRGB->linear here).
pub(crate) fn render_linear_rgb(&self, vertices: &[Vertex]) -> Vec<f32> {
    let rgba8 = self.render_rgba8(vertices); // factored from snapshot
    let mut out = Vec::with_capacity((self.inner.texture_size * self.inner.texture_size * 3) as usize);
    for px in rgba8.chunks_exact(4) {
        for &c in &px[..3] {
            let s = c as f32 / 255.0;
            out.push(if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) });
        }
    }
    out
}

/// The precomputed goal CIELAB as [H*W*3] row-major (L,a,b per pixel) for the
/// polish loss. Store the `goal_lab: Vec<[f32;4]>` baked in `new` on the inner
/// struct and expose it flattened to [L,a,b] here.
pub(crate) fn goal_lab_lab3(&self) -> Vec<f32> {
    self.inner.goal_lab.iter().flat_map(|p| [p[0], p[1], p[2]]).collect()
}
```

(Store `goal_lab` on `FitnessCalcInner` in `new` instead of dropping it after upload. Add `new_for_test` only if `new` is not already test-accessible — `FitnessCalc::new` is `pub(crate)`, so tests can call it directly; drop `new_for_test` from the test and use `FitnessCalc::new(device, queue, &goal, 1)`.)

In `gradient.rs`:

```rust
pub(crate) struct PolishCfg {
    pub(crate) subset_m: usize,
    pub(crate) steps_n: usize,
    pub(crate) lr: f32,
    pub(crate) tau: f32,
}
impl PolishCfg {
    pub(crate) fn for_test() -> Self { Self { subset_m: 1, steps_n: 60, lr: 0.05, tau: 0.02 } }
}

/// Polish the top-`subset_m` triangles of `genome` against the goal; keep the
/// result only if the exact ΔE2000 fitness improves. Returns whether it was kept.
pub(crate) fn polish(genome: &mut Vec<Vertex>, calc: &FitnessCalc, cfg: &PolishCfg) -> bool {
    use crate::genome::triangle_area;
    let n = genome.len() / 3;
    if n == 0 { return false; }
    let before = calc.fitness_of(genome);
    // Subset = largest `subset_m` triangles by area (Milestone 2 adds error weighting).
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| triangle_area(genome, b).partial_cmp(&triangle_area(genome, a)).unwrap());
    idx.truncate(cfg.subset_m.min(n));
    // Base = genome without the subset triangles, hard-rendered to linear RGB.
    let subset: std::collections::HashSet<usize> = idx.iter().copied().collect();
    let base_genome: Vec<Vertex> = (0..n)
        .filter(|t| !subset.contains(t))
        .flat_map(|t| genome[t * 3..t * 3 + 3].iter().copied())
        .collect();
    let size = calc.texture_size() as usize;
    let base = calc.render_linear_rgb(&base_genome);
    let goal_lab = calc.goal_lab_lab3();
    // Pack subset params, optimise.
    let tris: Vec<[[f32; 2]; 3]> = idx.iter().map(|&t| {
        let b = t * 3;
        [[genome[b].position[0], genome[b].position[1]],
         [genome[b + 1].position[0], genome[b + 1].position[1]],
         [genome[b + 2].position[0], genome[b + 2].position[1]]]
    }).collect();
    let cols: Vec<[[f32; 4]; 3]> = idx.iter().map(|&t| {
        let b = t * 3;
        [genome[b].color, genome[b + 1].color, genome[b + 2].color]
    }).collect();
    let device = Default::default();
    let mut state = PolishState::new(pack_params(&tris, &cols), size, size, &device);
    for _ in 0..cfg.steps_n { state.adam_step(&base, &goal_lab, cfg.tau, cfg.lr); }
    // Write optimised params back into a candidate genome.
    let flat = state.to_flat();
    let mut candidate = genome.clone();
    for (si, &t) in idx.iter().enumerate() {
        for k in 0..3 {
            let r = (si * 3 + k) * 6;
            let b = t * 3 + k;
            candidate[b].position[0] = flat[r].clamp(-1.0, 1.0);
            candidate[b].position[1] = flat[r + 1].clamp(-1.0, 1.0);
            for c in 0..4 { candidate[b].color[c] = flat[r + 2 + c].clamp(0.0, 1.0); }
        }
    }
    // Elitist gate.
    let after = calc.fitness_of(&candidate);
    if after > before { *genome = candidate; true } else { false }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --bin polygenvo gradient::tests::polish_improves_hard_fitness_on_misplaced_triangle -- --nocapture`
Expected: PASS — `after > before`. **This is the stop/go gate.** If it fails after reasonable tuning of `steps_n`/`lr`/`tau`, halt and report; do not proceed to Milestone 2.

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/gradient.rs src/polygenvo/fitness.rs
git commit -m "feat: end-to-end polish improves hard fitness (de-risk gate)"
```

---

## Milestone 2 — Wire into the ES behind `--gradient-polish`

### Task 2.1: Error-weighted subset selection

**Files:**
- Modify: `src/polygenvo/gradient.rs` (extend `polish` to take the parent error grid)
- Test: `src/polygenvo/gradient.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn subset_prefers_large_high_error_triangles() {
    // Two triangles: a big one in a high-error cell, a small one in a zero-error
    // cell. With subset_m=1, selection must pick the big high-error one (index 0).
    let chosen = super::select_subset(
        /*areas*/ &[0.5, 0.01],
        /*errors*/ &[10.0, 0.0],
        1,
    );
    assert_eq!(chosen, vec![0]);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --bin polygenvo gradient::tests::subset_prefers_large_high_error_triangles`
Expected: FAIL — `select_subset` not found.

- [ ] **Step 3: Implement**

```rust
/// Indices of the top-`m` triangles by `area * (1 + error)`, descending.
pub(crate) fn select_subset(areas: &[f32], errors: &[f32], m: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..areas.len()).collect();
    let score = |t: usize| areas[t] * (1.0 + errors[t]);
    idx.sort_by(|&a, &b| score(b).partial_cmp(&score(a)).unwrap());
    idx.truncate(m.min(areas.len()));
    idx
}
```

Then change `polish` to accept `error_grid: &[u32]` and compute a per-triangle error by sampling the grid cell at each triangle's centroid (reuse `ERROR_GRID_DIM`, the clip→cell mapping is the inverse of `cell_to_clip`). Replace the area-only sort in `polish` with `select_subset(&areas, &errors, cfg.subset_m)`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --bin polygenvo gradient::tests::subset_prefers_large_high_error_triangles`
Expected: PASS. Re-run the Task 1.7 gate test (now `polish` takes an error grid — pass `&vec![1u32; GRID_CELLS]`).

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/gradient.rs
git commit -m "feat: area*error subset selection for polish"
```

### Task 2.2: `PolishCfg` in `EsConfig`, off by default

**Files:**
- Modify: `src/polygenvo/es.rs` (`EsConfig`, `production`, the smoke-test config)
- Modify: `src/polygenvo/gradient.rs` (`PolishCfg` gains `enabled`, `every_k`, `production`/`disabled`)

- [ ] **Step 1: Write the failing test**

```rust
// in es.rs tests
#[test]
fn production_has_polish_disabled_by_default() {
    let cfg = EsConfig::production();
    assert!(!cfg.polish.enabled, "gradient polish must be opt-in");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --bin polygenvo es::tests::production_has_polish_disabled_by_default`
Expected: FAIL — no `polish` field.

- [ ] **Step 3: Implement**

In `gradient.rs`:

```rust
#[derive(Clone)]
pub(crate) struct PolishCfg {
    pub(crate) enabled: bool,
    pub(crate) every_k: u64,   // run after every K accepted improvements
    pub(crate) subset_m: usize,
    pub(crate) steps_n: usize,
    pub(crate) lr: f32,
    pub(crate) tau: f32,
}
impl PolishCfg {
    pub(crate) fn disabled() -> Self {
        Self { enabled: false, every_k: 200, subset_m: 32, steps_n: 40, lr: 0.02, tau: 0.02 }
    }
    pub(crate) fn for_test() -> Self { Self { enabled: true, every_k: 1, subset_m: 1, steps_n: 60, lr: 0.05, tau: 0.02 } }
}
```

In `es.rs`: add `pub(crate) polish: crate::gradient::PolishCfg` to `EsConfig`; set `polish: PolishCfg::disabled()` in `production()`; add `polish: PolishCfg::disabled()` to the smoke-test config literal.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --bin polygenvo es::tests::production_has_polish_disabled_by_default`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/es.rs src/polygenvo/gradient.rs
git commit -m "feat: PolishCfg in EsConfig, disabled by default"
```

### Task 2.3: Call polish from `run_es` every K improvements

**Files:**
- Modify: `src/polygenvo/es.rs` (the accept branch in the main loop, ~line 370–387)

- [ ] **Step 1: Write the failing test**

```rust
// in es.rs tests — polish-enabled smoke run must still satisfy the no-regression invariant.
#[test]
fn es_with_polish_does_not_regress() {
    let goal = make_checker_goal(32);
    let (device, queue) = init_test_wgpu();
    let test_phases = vec![Phase {
        cap: 6, pyramid_level: 0,
        initial_sigma_pos: 0.1, initial_sigma_col: 0.1, initial_sigma_grad: 0.1,
    }];
    let mut polish = crate::gradient::PolishCfg::disabled();
    polish.enabled = true; polish.every_k = 2; polish.subset_m = 2; polish.steps_n = 5;
    let result = run_es(device, queue, goal, EsConfig {
        phases: test_phases, max_steps: 20, lambda: 4,
        snapshot_every: None, stop_flag: None, polish,
    });
    assert!(result.final_fitness >= result.initial_fitness, "polish must not regress fitness");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --bin polygenvo es::tests::es_with_polish_does_not_regress`
Expected: FAIL — `EsConfig` has no `polish` field usage in `run_es` yet (and won't compile until Step 3).

- [ ] **Step 3: Implement the call site**

In `run_es`, in the `if let Some(i) = best_idx { ... }` accept block, after `improvements_total += 1;`, add:

```rust
if cfg.polish.enabled && improvements_total % cfg.polish.every_k == 0 {
    let calc = &pyramid[phase.pyramid_level];
    let kept = crate::gradient::polish(&mut current, calc, &parent_error_grid, &cfg.polish);
    if kept {
        (current_fitness, parent_error_grid) = score(calc, &current);
    }
}
```

(`score` already exists and returns `(usize, Vec<u32>)`.) Ensure `polish`'s signature is `polish(&mut Vec<Vertex>, &FitnessCalc, &[u32], &PolishCfg) -> bool`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --bin polygenvo es::tests::es_with_polish_does_not_regress`
Expected: PASS (the elitist gate guarantees no regression).

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/es.rs src/polygenvo/gradient.rs
git commit -m "feat: run gradient polish every K improvements (gated)"
```

### Task 2.4: `--gradient-polish` CLI flag

**Files:**
- Modify: `src/polygenvo/main.rs`

- [ ] **Step 1: Write the failing test**

```rust
// in main.rs tests (add a #[cfg(test)] mod if none) — flag parsing only.
#[test]
fn parses_gradient_polish_flag() {
    let args = vec!["polygenvo".to_string(), "--gradient-polish".to_string()];
    assert!(super::has_flag(&args, "--gradient-polish"));
    assert!(!super::has_flag(&["polygenvo".to_string()], "--gradient-polish"));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --bin polygenvo main::tests::parses_gradient_polish_flag` (or the module path the test lands in)
Expected: FAIL — `has_flag` not found (or add it next to the existing `arg_value`).

- [ ] **Step 3: Implement**

In `main.rs`, add a `has_flag` helper beside `arg_value`:

```rust
fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}
```

In `main`, after building the production config:

```rust
let mut cfg = EsConfig::production();
if has_flag(&args, "--gradient-polish") {
    cfg.polish.enabled = true;
}
```

(Wire `args` from `std::env::args().collect()` as `arg_value` already does.)

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --bin polygenvo parses_gradient_polish_flag`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/main.rs
git commit -m "feat: --gradient-polish CLI flag enables the polish"
```

### Task 2.5: Full verification pass

- [ ] **Step 1: Clippy + full test suite**

Run: `cargo clippy --bin polygenvo` → clean.
Run: `cargo test --bin polygenvo` → all green (existing 23 + new tests).

- [ ] **Step 2: Build release**

Run: `cargo build --release --bin polygenvo` → builds.

- [ ] **Step 3: Commit any fixups**

```bash
git add -A && git commit -m "chore: clippy/test fixups for gradient polish" || echo "nothing to fix"
```

---

## Milestone 3 — Validation (manual A/B, per CLAUDE.md)

### Task 3.1: A/B run on goal.png

- [ ] **Step 1: Baseline (~150s)**

Run: `timeout 152 stdbuf -oL ./target/release/polygenvo > /tmp/base.log 2>&1`
Record final fitness (`tail -1`) and keep the final snapshot frame.

- [ ] **Step 2: With polish (~150s)**

Run: `timeout 152 stdbuf -oL ./target/release/polygenvo --gradient-polish > /tmp/polish.log 2>&1`
Record final fitness and the final snapshot frame.

- [ ] **Step 3: Compare**

Compare final ΔE2000 (higher = better) and steps/sec (polish overhead), and eyeball the two final frames for reduction in hard-edged background facets — the symptom this targets. Note results in the commit message of any follow-up tuning.

- [ ] **Step 4: Decision**

If polish improves the frame and fitness is neutral-or-better, tune `every_k`/`subset_m`/`steps_n` for the throughput/quality trade and consider enabling by default. If it does not help despite the Task 1.7 gate passing, leave it opt-in and document the finding in the spec's status.

---

## Self-review notes (coverage vs spec)

- Module boundary (`gradient.rs`, single entry `polish`, flag-gated): Tasks 1.x, 2.2–2.4. ✓
- Subset polish over fixed hard-rendered base: Task 1.7 (`render_linear_rgb`), 2.1 (selection). ✓
- Soft rasterizer (coverage + barycentric colour + composite): Tasks 1.3–1.5. ✓
- MSE-in-Lab loss: Task 1.6. ✓
- Adam optimiser: Task 1.5. ✓
- Elitist re-score gate (no regression): Tasks 1.7, 2.3 (+ test `es_with_polish_does_not_regress`). ✓
- Interleaved cadence (every K improvements): Task 2.3. ✓
- burn dep + wgpu backend: Task 0.1. ✓
- Milestone-1 stop/go gate: Task 1.7. ✓
- Tests: forward convergence/coverage (1.3), forward loss (1.4), Adam lowers loss (1.5), Lab (1.6), hard-fitness gate (1.7), subset (2.1), no-regression (2.3), flag (2.4); existing suite stays green. ✓
- A/B validation: Task 3.1. ✓
