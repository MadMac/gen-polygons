# Tiled Differentiable Kernel (Phase 2) — Design

**Date:** 2026-06-10
**Binary:** `polygenvo`
**Status:** Approved for planning
**Parent:** `2026-06-10-end-to-end-gradient-optimizer-design.md` (this is its Phase 2)

## Problem

The merged differentiable polish is **compute-bound**: its backward pass is
O(num_tris²) per pixel (it recomputes the prefix composite and suffix transmittance
for every triangle, every pixel). Phase 1's benchmark proved this is *not* a backend
artifact — the brute-force polish is ~equal on GL and Vulkan (1334 vs 1269 ms at 128²
for 100 tris/10 steps), so atomic contention is **not** the bottleneck; the O(num_tris²)
work is. At 512² with ~1000 triangles it is minutes per gradient step — unusable for the
every-step end-to-end loop (Phase 3).

## Goal

A GPU forward/backward kernel that computes the **same gradient** as the existing CPU
oracle (`softras_ref::grad_loss`) but fast enough that a 512² gradient step over ~1000
triangles is interactive (low single-digit ms), via two compounding wins.

## Decisions locked during brainstorming

- **Build the full tiled kernel upfront** (best scalability toward bigger images).
- **Drop the shared-memory gradient reduction** the parent spec floated — Phase-1 data
  shows atomics are not the bottleneck, so plain global atomic scatter is fine once the
  work shrinks.
- **Listless loop-and-reject tiling** first (each tile loops triangles, bbox-rejects
  non-overlappers); explicit prefix-sum binning is a documented future lever if
  much-larger images make the O(tiles×num_tris) reject loop dominate.
- **Reuse** the CPU oracle (`softras_ref::grad_loss`) as the correctness reference and
  the existing `gpu_grad`/`gpu_forward_lab` equality-test harness.

## Design

### Two compounding wins

1. **O(num_tris) reverse-transmittance backward** (the primary algorithmic fix). The
   forward stores, per pixel, the final composited color **and** the final transmittance
   `T_final = Π(1 − src_a_j)`. The backward walks each pixel's triangles **once in
   reverse draw order**, maintaining a running suffix transmittance reconstructed
   incrementally (`T` updated by dividing out each triangle's `(1 − src_a)` as it moves
   past it — **alpha-clamped** to `≤ ~0.999` to avoid div-by-zero on near-opaque
   triangles, exactly as 3D Gaussian Splatting does). This replaces the O(num_tris²)
   prefix/suffix recompute with O(num_tris).
2. **Tiling** bounds *which* triangles a pixel considers. A **workgroup per 16×16 pixel
   tile** cooperatively loops the triangle list, cheaply **bbox-rejects** non-overlappers,
   and only composites / back-props the triangles whose bbox intersects the tile → per-
   pixel cost becomes O(triangles-in-tile), not O(all triangles).

Together: O(num_tris²)/pixel → O(triangles-in-tile)/pixel.

### Gradient scatter

Keep Path B's global `atomic_add_f32` (CAS on the u32 bit-pattern). Contention is now low
(only a tile's pixels touch a tile's triangles' vertices), and Phase 1 proved atomics
weren't the bottleneck. No shared-memory reduction.

### Memory

Per-pixel scratch: final color (vec3) + final transmittance (f32) = one `vec4<f32>` per
pixel = H×W×16 bytes (bounded; **not** the rejected H×W×num_tris wall). The forward
writes it; the backward reads it.

### Components / files

- **Create** `src/polygenvo/softraster_tiled.wgsl` — tiled `forward` (composite +
  write per-pixel state) and `backward` (reverse-transmittance, atomic scatter) entries.
  Reuses the same color-space / coverage / barycentric helpers as `softraster.wgsl`
  (factor the shared WGSL helpers or duplicate verbatim, matching the existing style).
- **Keep** `src/polygenvo/softraster.wgsl` (brute-force entries) as the oracle
  cross-check / fallback.
- **Modify** `src/polygenvo/gradient.rs` — `PolishState` gains the tiled pipelines, the
  per-pixel scratch buffer, and tile-count uniforms; the polish loop dispatches the
  tiled entries (`ceil(W/16)×ceil(H/16)` workgroups for the tile passes). Add tiled
  `#[cfg(test)]` dispatch helpers mirroring `gpu_forward_lab`/`gpu_grad`.
- **Reuse** `adam.wgsl` unchanged and `softras_ref.rs` (oracle) unchanged.

### Data flow (one gradient step, tiled)

1. Upload current params (as today).
2. **Tiled forward:** per tile, composite overlapping triangles front-to-back; write
   per-pixel final color + transmittance to scratch; (Lab residual vs goal as today).
3. **Tiled backward:** per tile, reverse-walk overlapping triangles using the stored
   per-pixel state; scatter ∂Lab-MSE/∂params via `atomic_add_f32`.
4. **Adam** update (unchanged).

## Testing

- **Equality vs oracle:** the existing `gpu_grad`-style harness, but driving the tiled
  entries, must match `softras_ref::grad_loss` within the existing 2e-2 relative
  tolerance — for the single-triangle FD scene **and** new cases:
  - a triangle **spanning a tile boundary** (its bbox covers ≥2 tiles),
  - a pixel **covered by many overlapping triangles** (exercises the reverse-transmittance
    chain and the clamp),
  - an **empty tile** (no triangles) produces zero gradient / black.
- **Forward equality:** tiled `forward` per-pixel Lab matches `forward_pixel_lab` (and
  the brute-force GPU forward) within 1e-2, incl. tile boundaries.
- **Benchmark:** extend `bench_backend` to time the **tiled** forward+backward at
  128²/256²/512² with ~1000 triangles; assert (informally, via printout) an
  order-of-magnitude speedup over brute-force and a low-ms 512² step.
- **Regression:** existing suite stays green; clippy clean.

## Acceptance

Tiled GPU gradient matches the CPU oracle on all the above cases, and the benchmark
shows a 512²/~1000-triangle gradient step is interactive (low single-digit ms) — fast
enough to unblock the Phase 3 end-to-end loop.

## Risks & mitigations

- **Reverse-transmittance alpha-clamp accuracy** vs the exact oracle → clamp threshold
  tuned high (≤0.999); 2e-2 tolerance + oracle tests guard it; test scenes use alpha<1.
- **Tile-boundary correctness** (a triangle contributing to multiple tiles; a pixel's
  triangle set) → explicit spanning/overlap edge-case tests are the guard.
- **Listless reject loop is O(tiles×num_tris)** in bbox tests → fine to ~10k triangles
  (~1e7 cheap tests/pass); explicit prefix-sum binning is the documented next lever for
  much larger images.
- **Per-pixel scratch must match forward/backward exactly** (color + transmittance
  semantics) → the forward and backward share one WGSL definition of the stored layout;
  the oracle equality test catches any mismatch.

## Out of scope

- Explicit prefix-sum / sorted tile binning (future scalability lever).
- The end-to-end run loop (Phase 3 — separate plan; this kernel is its prerequisite).
- Changing the gradient math or the Lab-MSE proxy (identical to the merged/validated
  version; only the algorithm changes).
