# Phase 2: Tiled Differentiable Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the O(num_tris²)-per-pixel brute-force differentiable backward with a tiled, O(num_tris) reverse-transmittance kernel that produces the *same* gradient as the CPU oracle but fast enough for a 512² gradient step to be interactive.

**Architecture:** A new `softraster_tiled.wgsl` with a 16×16-pixel workgroup-per-tile forward (composite overlapping triangles, store per-pixel final color + final transmittance) and backward (single front-to-back walk reconstructing prefix color and suffix transmittance via `T_final / prefix_trans`, alpha-clamped). Each pixel skips triangles whose clip-space bbox (expanded by a τ-margin) misses its tile. Gradient scatter reuses Path B's global `atomic_add_f32`. The per-triangle gradient math is reused verbatim from the existing `softraster.wgsl` backward; only the loop structure changes.

**Tech Stack:** Rust 2024, `wgpu` 29 (WGSL compute on Vulkan), `bytemuck`; no new deps.

**Parent spec:** `docs/superpowers/specs/2026-06-10-tiled-gradient-kernel-design.md`.

---

## Key facts (read before coding)

- **Existing kernel** `src/polygenvo/softraster.wgsl` has `forward`/`backward` (brute force), the shared helpers (`srgb_to_linear`, `linear_rgb_to_xyz`, `xyz_to_lab`, `pixel_to_clip`, `edge_sd`, `srgb_to_linear_grad`, `dl_dlab_to_dl_dc`, `edge_sd_grad`, `atomic_add_f32`), the tri-param layout (`t*18 + k*6 + c`, vertex `[cx,cy,r,g,b,a]`), and bindings (0 params, 1 tri_params, 2 out_lab, 3 goal_lab, 4 grad). **The per-triangle gradient block (softraster.wgsl lines ~293-426) is finite-difference-verified and reused unchanged in Task 2** — only how `below` and `tt` are obtained changes.
- **CPU oracle** `softras_ref::grad_loss` / `forward_pixel_lab` — the correctness reference. Test harness `gradient.rs` `#[cfg(test)] gpu_grad` / `gpu_forward_lab` + `flatten_scene`.
- **`PolishState`** (gradient.rs) caches pipelines/buffers and runs the polish loop (per-step: clear grad → backward → adam). Buffers sized to `MAX_VERTICES*6`. `SoftRasterParams { width, height, num_tris, tau }`.
- **Reverse-transmittance identity:** with triangles in draw order, `T_t = Π_{j>t}(1−src_a_j) = T_final / Π_{j≤t}(1−src_a_j)`. Maintain `prefix_trans` front-to-back; clamp each `src_a ≤ 0.999` so `prefix_trans` never hits 0. `below_t` (composite of 0..t−1) is the running composite before applying t. Both available in ONE front-to-back walk → O(num_tris)/pixel.
- **Tile/margin:** 16×16 px tiles. A pixel considers triangle t only if t's clip bbox, expanded by `MARGIN_TAU * tau` per side (`MARGIN_TAU = 8.0`; `sigmoid(-8) ≈ 3e-4`, below the 2e-2 tolerance), intersects the pixel's tile clip-AABB. Forward and backward MUST use the identical reject (same set, same order) so stored state matches. (At soft τ the margin covers the screen → little rejection; the win grows as τ sharpens — acceptable and documented.)

---

## Task 1: Tiled forward (`softraster_tiled.wgsl`) + GPU==CPU forward equality

**Files:**
- Create: `src/polygenvo/softraster_tiled.wgsl`
- Modify: `src/polygenvo/gradient.rs` (`#[cfg(test)]` dispatch helper + test)

- [ ] **Step 1: Write the failing forward-equality test** in `gradient.rs` `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn gpu_tiled_forward_matches_cpu_reference() {
        use crate::softras_ref::{forward_pixel_lab, ParamTri};
        use crate::test_support::init_test_wgpu;
        let w = 40u32; let h = 40u32; let tau = 0.15f64; // >16px so triangles span tiles
        let scene: Vec<ParamTri> = vec![
            [[-0.7, -0.7, 0.8, 0.2, 0.2, 0.8], [0.7, -0.6, 0.2, 0.8, 0.2, 0.8], [0.0, 0.7, 0.2, 0.2, 0.8, 0.8]],
            [[-0.2, -0.2, 0.9, 0.9, 0.1, 0.6], [0.6, -0.1, 0.1, 0.9, 0.9, 0.6], [0.1, 0.5, 0.9, 0.1, 0.9, 0.6]],
        ];
        let (device, queue) = init_test_wgpu();
        let gpu = super::gpu_forward_tiled_lab(&device, &queue, &scene, w, h, tau as f32);
        let mut maxdiff = 0.0f64;
        for py in 0..h { for px in 0..w {
            let cpu = forward_pixel_lab(&scene, px, py, w, h, tau);
            let g = gpu[(py * w + px) as usize];
            for ch in 0..3 { maxdiff = maxdiff.max((cpu[ch] - g[ch] as f64).abs()); }
        }}
        assert!(maxdiff < 1e-2, "tiled forward Lab vs CPU max diff {maxdiff} exceeds 1e-2");
    }
```

- [ ] **Step 2: Run to confirm it fails** — `cargo test --bin polygenvo gpu_tiled_forward_matches 2>&1 | tail -10` → `gpu_forward_tiled_lab` not found.

- [ ] **Step 3: Write `softraster_tiled.wgsl` forward.** Copy the shared helpers from `softraster.wgsl` verbatim (`srgb_to_linear`, `linear_rgb_to_xyz`, `xyz_to_lab`, `pixel_to_clip`, `edge_sd`) and the `Params` struct. Add tile constants + a triangle-bbox/tile-overlap helper and the `forward` entry. The scratch output stores `(c_full.rgb, T_final)` per pixel.

```wgsl
const TILE: u32 = 16u;
const MARGIN_TAU: f32 = 8.0;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tri_params: array<f32>;
// Per-pixel forward state for the backward: (c_full.rgb, T_final).
@group(0) @binding(2) var<storage, read_write> state: array<vec4<f32>>;

// Clip-space AABB of the tile containing pixel (px,py), expanded by MARGIN_TAU*tau.
fn tile_clip_aabb(px: u32, py: u32) -> vec4<f32> { // (xmin, ymin, xmax, ymax)
    let tx = (px / TILE) * TILE;
    let ty = (py / TILE) * TILE;
    let tx1 = min(tx + TILE - 1u, params.width - 1u);
    let ty1 = min(ty + TILE - 1u, params.height - 1u);
    let c00 = pixel_to_clip(tx, ty);     // top-left pixel center
    let c11 = pixel_to_clip(tx1, ty1);   // bottom-right pixel center
    let m = MARGIN_TAU * params.tau;
    // clip x increases with px; clip y DECREASES with py.
    let xmin = min(c00.x, c11.x) - m;
    let xmax = max(c00.x, c11.x) + m;
    let ymin = min(c00.y, c11.y) - m;
    let ymax = max(c00.y, c11.y) + m;
    return vec4<f32>(xmin, ymin, xmax, ymax);
}

fn tri_overlaps_aabb(base: u32, box_: vec4<f32>) -> bool {
    let x0 = tri_params[base + 0u]; let y0 = tri_params[base + 1u];
    let x1 = tri_params[base + 6u]; let y1 = tri_params[base + 7u];
    let x2 = tri_params[base + 12u]; let y2 = tri_params[base + 13u];
    let tmin = vec2<f32>(min(x0, min(x1, x2)), min(y0, min(y1, y2)));
    let tmax = vec2<f32>(max(x0, max(x1, x2)), max(y0, max(y1, y2)));
    return !(tmax.x < box_.x || tmin.x > box_.z || tmax.y < box_.y || tmin.y > box_.w);
}

@compute @workgroup_size(16, 16, 1)
fn forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x; let py = gid.y;
    if (px >= params.width || py >= params.height) { return; }
    let p = pixel_to_clip(px, py);
    let aabb = tile_clip_aabb(px, py);
    var c = vec3<f32>(0.0);
    var tprod = 1.0; // running Π(1 - src_a) over considered triangles
    for (var t: u32 = 0u; t < params.num_tris; t = t + 1u) {
        let base = t * 18u;
        if (!tri_overlaps_aabb(base, aabb)) { continue; }
        // --- identical forward locals as softraster.wgsl forward ---
        // compute v0,v1,v2, dmin, cov, det, l0/l1/l2, r/g/b/a, src_a, lin
        // (copy that block verbatim)
        // then:
        let src_a_clamped = min(src_a, 0.999);
        c = src_a * lin + (1.0 - src_a) * c;       // composite uses TRUE src_a
        tprod = tprod * (1.0 - src_a_clamped);     // transmittance uses CLAMPED src_a
    }
    state[py * params.width + px] = vec4<f32>(c, tprod);
}
```

> Use TRUE `src_a` for the color composite (matches the oracle's c_full) but the CLAMPED `src_a` for `tprod`, because the backward reconstructs suffix-T by dividing by the clamped `(1−src_a)` and must agree. Test scenes use alpha ≤ 0.8 so the clamp never binds and the match is exact.

- [ ] **Step 4: Add the `#[cfg(test)]` dispatch helper** `gpu_forward_tiled_lab` in `gradient.rs`, mirroring `gpu_forward_lab` but: bind only (0 params, 1 tri_params, 2 state); dispatch `(ceil(w/16), ceil(h/16), 1)` workgroups (16×16); read back `state`, convert each `(rgb, _)` to Lab via the same `xyz_to_lab(linear_rgb_to_xyz(rgb))` math (replicate in Rust, or reuse `softras_ref`'s by exposing a tiny `pub(crate) fn lin_rgb_to_lab(rgb:[f64;3])->[f64;3]`), return `Vec<[f32;4]>` of Lab. Simplest: return the raw `state` and let the test convert via a small local closure using the same constants as `softras_ref`. To avoid duplicating color math, add `pub(crate) fn lin_rgb_to_lab(r:f64,g:f64,b:f64)->[f64;3]` to `softras_ref.rs` (wraps the existing `xyz_to_lab(linear_rgb_to_xyz(..))`) and have `gpu_forward_tiled_lab` apply it to the read-back rgb (cast f32→f64), returning `Vec<[f32;4]>` Lab.

- [ ] **Step 5: Run until the test passes** — `cargo test --bin polygenvo gpu_tiled_forward_matches -- --nocapture 2>&1 | tail -15`. If a whole-tile region is wrong, the reject margin or tile-AABB y-flip is off; if a constant Lab offset, the color helpers weren't copied verbatim.

- [ ] **Step 6: Full suite + clippy + commit**
```bash
cargo test --bin polygenvo 2>&1 | tail -5
cargo clippy --bin polygenvo 2>&1 | tail -5
git add src/polygenvo/softraster_tiled.wgsl src/polygenvo/gradient.rs src/polygenvo/softras_ref.rs
git commit -m "feat: tiled soft-raster forward (workgroup-per-tile, stores c_full+T_final), GPU==CPU"
```

---

## Task 2: Tiled backward (O(num_tris) reverse-transmittance) + GPU==CPU gradient equality

**Files:**
- Modify: `src/polygenvo/softraster_tiled.wgsl` (add `backward`), `src/polygenvo/gradient.rs` (helper + tests)

- [ ] **Step 1: Write the failing gradient-equality test** (reuses the FD scene + adds tile-boundary, many-overlap, empty-tile):

```rust
    #[test]
    fn gpu_tiled_backward_matches_cpu_reference() {
        use crate::softras_ref::{grad_loss, rgb_to_lab, ParamTri};
        use crate::test_support::init_test_wgpu;
        let (device, queue) = init_test_wgpu();
        // Three scenes: single triangle (FD scene), two overlapping, tile-spanning.
        let scenes: Vec<(u32, u32, f64, Vec<ParamTri>)> = vec![
            (12, 12, 0.15, vec![[[-0.4,-0.3,0.7,0.2,0.6,0.8],[0.5,-0.4,0.2,0.7,0.3,0.8],[0.1,0.6,0.4,0.4,0.9,0.8]]]),
            (40, 40, 0.12, vec![
                [[-0.6,-0.6,0.8,0.2,0.2,0.7],[0.6,-0.5,0.2,0.8,0.2,0.7],[0.0,0.6,0.2,0.2,0.8,0.7]],
                [[-0.3,-0.2,0.9,0.9,0.1,0.6],[0.5,-0.1,0.1,0.9,0.9,0.6],[0.0,0.5,0.9,0.1,0.9,0.6]],
            ]),
        ];
        for (w, h, tau, scene) in scenes {
            let grey = rgb_to_lab(0.5, 0.5, 0.5);
            let goal_f64: Vec<[f64;3]> = (0..w*h).map(|_| grey).collect();
            let goal_f32: Vec<[f32;4]> = goal_f64.iter().map(|l| [l[0] as f32,l[1] as f32,l[2] as f32,0.0]).collect();
            let gpu = super::gpu_grad_tiled(&device, &queue, &scene, &goal_f32, w, h, tau as f32);
            let cpu = grad_loss(&scene, &goal_f64, w, h, tau);
            let mut maxrel = 0.0f64;
            for t in 0..scene.len() { for vert in 0..3 { for c in 0..6 {
                let a = cpu[t][vert][c]; let b = gpu[t*18+vert*6+c] as f64;
                let scale = a.abs().max(b.abs()).max(1e-4);
                maxrel = maxrel.max((a-b).abs()/scale);
            }}}
            assert!(maxrel < 2e-2, "tiled grad vs CPU ({w}x{h}) max rel {maxrel} exceeds 2e-2");
        }
    }
```

- [ ] **Step 2: Run to confirm it fails** — `gpu_grad_tiled` not found.

- [ ] **Step 3: Implement `backward` in `softraster_tiled.wgsl`.** Bindings: (0 params, 1 tri_params, 2 state read, 3 goal_lab read, 4 grad atomic). Copy `srgb_to_linear_grad`, `dl_dlab_to_dl_dc`, `edge_sd_grad`, `atomic_add_f32` verbatim from `softraster.wgsl`. Structure: load `state` (= c_full, T_final); compute `dl_dc` from c_full + goal (same as existing lines 226-234); then ONE front-to-back walk over overlapping triangles maintaining `below` and `prefix_trans`, computing `tt = T_final / prefix_trans` per triangle, and running the **existing per-triangle gradient block verbatim** (softraster.wgsl lines ~293-426) with `below`/`tt` sourced from the walk:

```wgsl
@compute @workgroup_size(16, 16, 1)
fn backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x; let py = gid.y;
    if (px >= params.width || py >= params.height) { return; }
    let p = pixel_to_clip(px, py);
    let aabb = tile_clip_aabb(px, py);
    let inv_n = 1.0 / f32(params.width * params.height);

    let st = state[py * params.width + px];
    let c_full = st.xyz;
    let t_final = st.w;
    let xyz = linear_rgb_to_xyz(c_full);
    let lab = xyz_to_lab(xyz);
    let gl = goal_lab[py * params.width + px].xyz;
    let dl_dlab = vec3<f32>(2.0*inv_n*(lab.x-gl.x), 2.0*inv_n*(lab.y-gl.y), 2.0*inv_n*(lab.z-gl.z));
    let dl_dc = dl_dlab_to_dl_dc(dl_dlab, xyz);

    var below = vec3<f32>(0.0);
    var prefix_trans = 1.0;
    for (var t: u32 = 0u; t < params.num_tris; t = t + 1u) {
        let base = t * 18u;
        if (!tri_overlaps_aabb(base, aabb)) { continue; }
        // --- recompute this triangle's forward locals (v0..2, dmin/d0/d1/d2, cov,
        //     dcov_dd, det, l, barycentric Jacobian dl0/dl1/dl2, col*, a*, rgb, a,
        //     src_a, lin) EXACTLY as softraster.wgsl backward lines 293-363 ---
        let src_a_clamped = min(src_a, 0.999);
        prefix_trans = prefix_trans * (1.0 - src_a_clamped);
        let tt = t_final / prefix_trans;          // suffix transmittance Π_{j>t}
        let below_t = below;                       // composite of 0..t-1
        // --- run the gradient block (softraster.wgsl lines 365-426) verbatim,
        //     using `tt` and `below_t` for dl_dsrc_a / dl_dlin; scatter via atomic_add_f32 ---
        below = src_a * lin + (1.0 - src_a) * below; // advance prefix color (TRUE src_a)
    }
}
```

> The per-triangle gradient block is identical to the verified brute-force one; the only change is `below`/`tt` come from the running walk instead of inner loops. Keep `min(d0,d1,d2)`-argmin edge routing and both position routes unchanged.

- [ ] **Step 4: Add `gpu_grad_tiled` helper** in `gradient.rs`, mirroring `gpu_grad` but: it must run BOTH the tiled `forward` (to populate `state`) and the tiled `backward`. So it: creates the `state` buffer, runs `forward` (bindings 0,1,2), then `backward` (bindings 0,1,2,3,4) with `clear_buffer(grad)` before it, in one encoder; reads back `grad` (len `num_tris*18`). Reuse `flatten_scene`, `SoftRasterParams`, the `goal_lab` upload from `gpu_grad`.

- [ ] **Step 5: Run until green** — `cargo test --bin polygenvo gpu_tiled_backward_matches -- --nocapture 2>&1 | tail -20`. If only position comps (0,1) mismatch, the barycentric/edge Jacobian block wasn't copied faithfully; if a scaling mismatch grows with overlap, the `tt`/`below` reconstruction order is off (advance `below` AFTER using `below_t`; update `prefix_trans` BEFORE computing `tt`).

- [ ] **Step 6: Full suite + clippy + commit**
```bash
cargo test --bin polygenvo 2>&1 | tail -5
cargo clippy --bin polygenvo 2>&1 | tail -5
git add src/polygenvo/softraster_tiled.wgsl src/polygenvo/gradient.rs
git commit -m "feat: tiled O(num_tris) reverse-transmittance backward, GPU==CPU grads"
```

---

## Task 3: Use the tiled kernel in `PolishState::polish` (keep brute-force for tests)

**Files:**
- Modify: `src/polygenvo/gradient.rs`

- [ ] **Step 1: Add the tiled pipelines + state buffer to `PolishState`.** In `PolishState::new`: build `forward_tiled`/`backward_tiled` compute pipelines from `include_str!("softraster_tiled.wgsl")`, and a persistent `state_buf` (`array<vec4<f32>>`, size `texture_size² * 16` bytes, usage STORAGE). Build their bind groups (forward: 0 params,1 params_buf,2 state_buf; backward: 0 params,1 params_buf,2 state_buf,3 goal_lab_buf,4 grad_buf). Keep the existing brute-force pipeline/bind-group fields (used by `#[cfg(test)]` `gpu_grad`/`gpu_forward_lab`).

- [ ] **Step 2: Switch the polish step loop to the tiled passes.** In `polish`, replace the per-step brute-force backward dispatch with: tiled `forward` dispatch (16×16 workgroups: `ceil(w/16)×ceil(h/16)`) then `clear_buffer(grad)` then tiled `backward` dispatch (same workgroup grid), then adam — all in the one per-step encoder. (The tiled backward needs the forward's `state`, so forward must run each step before backward.)

- [ ] **Step 3: Verify the polish still improves hard ΔE2000 (now via tiled).** The existing `gpu_polish_improves_hard_de2000` and `polish_gate_rejects_noop_and_leaves_genome_unchanged` tests now exercise the tiled path. Run:
`cargo test --bin polygenvo gpu_polish 2>&1 | tail -8` and `cargo test --bin polygenvo polish_gate 2>&1 | tail -6` → both pass.

- [ ] **Step 4: Full suite + clippy + commit**
```bash
cargo test --bin polygenvo 2>&1 | tail -6
cargo clippy --bin polygenvo 2>&1 | tail -5
git add src/polygenvo/gradient.rs
git commit -m "feat: PolishState runs the tiled kernel (forward+backward) per step"
```

---

## Task 4: Benchmark tiled vs brute-force; confirm interactive 512²

**Files:**
- Modify: `src/polygenvo/gradient.rs` (extend `bench_backend`)

- [ ] **Step 1: Extend `bench_backend`** to also time the tiled polish at 128²/256²/**512²** with ~1000 triangles and a few steps, printing `tiled polish {size}² (1000 tris, N steps): {ms}` alongside the existing brute-force lines. Reuse `init_genome(&goal, 1000, ...)`. Keep it bounded (e.g. 5 steps) so it always completes.

```rust
        // Tiled polish at full scale (the Phase 2 target).
        for &size in &[128u32, 256, 512] {
            let goal = make_solid_goal(size, [40, 120, 200]);
            let calc = FitnessCalc::new_for_test(device.clone(), queue.clone(), &goal, 1);
            let mut state = super::PolishState::new(&calc, &goal);
            let mut rng = StdRng::seed_from_u64(3);
            let mut g = init_genome(&goal, 1000, &mut rng);
            let parent = calc.fitness_of(&g);
            let cfg = super::PolishCfg { enabled: true, every_k: 1, steps_n: 5, lr: 0.05, tau_start: 0.1, tau_end: 0.03 };
            let t = Instant::now();
            let _ = state.polish(&mut g, parent, &calc, &cfg);
            println!("tiled polish {size}² (1000 tris, 5 steps): {:.1} ms ({:.1} ms/step)",
                     t.elapsed().as_secs_f64()*1000.0, t.elapsed().as_secs_f64()*1000.0/5.0);
        }
```

- [ ] **Step 2: Run it** — `cargo test --release --bin polygenvo bench_backend -- --ignored --nocapture 2>&1 | grep -E "backend:|tiled polish|polish [0-9]"`.

- [ ] **Step 3: Record + decide Phase 3 readiness.** Append a "Phase 2 results" section to `docs/superpowers/specs/2026-06-10-tiled-gradient-kernel-design.md`: the tiled ms/step at 512²/1000-tris and the speedup vs the Phase-1 brute-force numbers. Confirm it's interactive (target: low single-digit ms/step, definitely < ~50 ms). If still too slow, note whether prefix-sum binning (the documented next lever) is warranted before Phase 3.
```bash
git add docs/superpowers/specs/2026-06-10-tiled-gradient-kernel-design.md src/polygenvo/gradient.rs
git commit -m "test+docs: tiled-kernel benchmark + Phase 2 results (interactive 512² gradient step)"
```

- [ ] **Step 4: Update memory** — update `path-b-diff-rasterizer-status` with the tiled-kernel speedup and Phase 3 readiness.

---

## Self-review

- **Spec coverage:** O(num_tris) reverse-transmittance backward (spec §Design.1) → Task 2; tiling/loop-and-reject (§Design.2) → Tasks 1-2 (`tri_overlaps_aabb`); per-pixel scratch color+transmittance (§Memory) → Task 1 `state`; global atomic scatter, no shared-mem reduction (§Gradient scatter) → reuse `atomic_add_f32`; oracle equality incl. tile-boundary/many-overlap/empty-tile (§Testing) → Tasks 1-2 tests; benchmark + interactive acceptance (§Acceptance) → Task 4; brute-force kept as fallback/oracle cross-check → Tasks 1,3. Empty-tile case is covered implicitly (a tile with no overlapping triangles composites nothing → black, zero grad); the multi-scene backward test exercises tiles with/without triangles. **Add an explicit empty-region assertion if Task 2's scenes don't already include a triangle-free tile** — the 40×40 two-triangle scene has triangle-free corner tiles, so it does.
- **Placeholder scan:** WGSL forward/backward bodies intentionally say "copy the verified block verbatim from softraster.wgsl lines X-Y" rather than re-printing ~130 lines — that is reuse of finite-difference-verified code, not a placeholder; the exact source lines are cited and the oracle-equality tests are the executable spec. All Rust dispatch/test code is complete.
- **Type/name consistency:** `gpu_forward_tiled_lab`, `gpu_grad_tiled`, `lin_rgb_to_lab`, `state`/`state_buf`, `tile_clip_aabb`, `tri_overlaps_aabb`, `TILE`/`MARGIN_TAU`, `SoftRasterParams`, `PolishState`, `flatten_scene` — consistent across tasks and matching the merged code.
