# GPU-Native Differentiable Rasterizer — Design

**Date:** 2026-06-08
**Binary:** `polygenvo`
**Status:** Approved for planning

## Problem

The (1+λ)-ES is at a **practical quality ceiling** documented in
[2026-06-08-future-directions.md](2026-06-08-future-directions.md): a large
triangle committed early gets a wrong silhouette that hill-climbing can't repair.
Root cause, well established over five neutral experiments: repairing a wrong big
triangle is a **downhill-first** move that elitist (1+λ) rejects, and under a
**hard** rasterizer the loss is **discontinuous in vertex positions**, so there is
no gradient pointing at the fix.

Of the two scoped paths forward, **Path A** (incremental/dirty-region fitness) is a
throughput win but **quality-neutral by construction** — given `--infinite`, more
steps/sec mostly reaches the same wall sooner; it does not address the silhouette
ceiling. **Path B** (a differentiable rasterizer) directly dissolves that wall:
gradients move all of a triangle's vertices coherently downhill in one step, with no
elitist rejection and no discontinuity. Path B is therefore the chosen project.

The shelved branch `feat/hybrid-es-gradient-polish`
([design](2026-06-05-hybrid-es-gradient-polish-design.md)) already **validated the
math** (+57% ΔE2000 at a matched gate) but was a no-op in pipeline because it used
per-triangle, **framework-mediated** (`burn`) GPU dispatch (~24 s/call). The lesson:
the rasterizer must be **framework-free, batched/vectorized, and on the existing
wgpu device** (no CPU round-trip).

## Decisions locked during brainstorming

- **Role:** hybrid-first — keep the ES as the driver for discrete decisions
  (add/split/delete/z-order/coarse-to-fine schedule); add a fast, **all-triangle**,
  on-device gradient polish of continuous params (positions + per-vertex RGBA),
  gated by the existing hard ΔE2000 renderer. Build the kernel so flipping to an
  end-to-end gradient optimizer later is only a *scheduling* change.
- **Differentiable loss = MSE in CIELAB** (smooth); the exact **ΔE2000 hard
  renderer stays the acceptance gate**. Proxy moves it, gate certifies it.
- **Optimizer = hand-rolled Adam** in a small on-device compute pass (~6 floats per
  vertex). No CPU round-trip beyond the tiny scalar-fitness readback the hard
  renderer already does.
- **Kernel sequencing:** brute-force pixel-parallel kernel first (prove the math
  behind the gate), then tile-binned kernel for the all-triangle 512² speed bar.
- **Acceptance bar:** a measurable ΔE2000 improvement over `master` on `goal.png`
  at a matched budget **and** visibly dissolved late-stage hard-edge facets.
- **No new dependencies** (no `burn`); preserves the repo's framework-free ethos.

## Design

### Module boundary

New framework-free module **[`gradient.rs`](../../../src/polygenvo/gradient.rs)**
with one entry point:

```
gradient::polish(genome: &mut Vec<Vertex>, calc: &FitnessCalc, cfg: &PolishCfg) -> bool  // true if kept
```

- Reuses `FitnessCalc`'s `Arc<Device>`/`Arc<Queue>`, the **precomputed goal-Lab
  buffer** (in [fitness.rs](../../../src/polygenvo/fitness.rs)), the `Vertex` layout
  ([genome.rs](../../../src/polygenvo/genome.rs)), and the existing hard ΔE2000
  renderer for the gate.
- `es.rs` is the only caller. Flag off ⇒ `run_es` is byte-for-byte today.
- Nothing depends on `gradient.rs`.

### New shaders

- **`softraster.wgsl`** — fused differentiable rasterizer:
  - **Forward** compute pass: soft-composite all triangles in array order via OVER
    (`C ← A·α·col + (1 − A·α)·C`), coverage `A = sigmoid(−signed_dist(p,t)/τ)`
    (SoftRas-style, differentiable in vertex positions), per-vertex color by
    barycentric interpolation; convert `C` → CIELAB differentiably; per-pixel
    Lab-MSE residual vs goal-Lab. `τ` annealed soft→sharp across the N steps.
  - **Backward** compute pass: accumulate ∂(Lab-MSE)/∂params. Gradient flows
    through the z-ordered OVER chain via suffix transmittance.
- **`adam.wgsl`** — Adam update over the param buffer; clamps positions to clip
  space and colors/alpha to [0,1].

### GPU buffers owned by `gradient.rs`

Params (working copy of genome continuous params), gradients, Adam moments (m, v),
and a per-pixel residual/color scratch (H×W floats — cheap; **not** the
per-triangle-per-pixel memory wall the shelved spec rejected).

### Data flow, one `polish` call

1. Upload current genome → param buffer.
2. Loop N steps: forward → backward → Adam, annealing `τ`.
3. Write params back into a **candidate** genome.
4. **Gate:** re-score candidate with the real ΔE2000 hard renderer; keep only if it
   beats the parent, else discard and restore. Preserves the (1+λ) no-regression
   guarantee and catches both approximations (soft≠hard, draw-order).

### Known WGSL constraint (decide at implementation)

Core WGSL has **no atomic float add**. The backward scatter of per-vertex gradients
needs float accumulation. Two routes:

- **(a)** `atomicCompareExchangeWeak` CAS-loop on a `u32` bit-cast of `f32` — simple,
  contention-heavy. Use for the Milestone-1 brute-force kernel (correctness first).
- **(b)** gather-per-vertex — one thread per vertex loops the pixels in its
  triangle's bbox using a per-pixel transmittance/color buffer stored in the forward
  pass; no atomics, naturally sparse. Adopt for the tile-binned production kernel.

### Integration & cadence

- `PolishCfg` inside `EsConfig` (`enabled`, `every_k`, `steps_n`, `lr`, `tau_start`,
  `tau_end`), default **off**; new CLI flag **`--gradient-polish`** in
  [main.rs](../../../src/polygenvo/main.rs) (parsed via the existing `arg_value`
  style).
- In [es.rs](../../../src/polygenvo/es.rs) `run_es`: after an accepted improvement
  whose count is a multiple of `every_k`, call `gradient::polish`. On a kept polish,
  refresh `current_fitness` and `parent_error_grid`. All-triangle ⇒ **no** subset
  selection (simpler than the shelved design — no top-M area×error pick, no fixed
  background split).

### Graduation to end-to-end (designed-in, not built)

The kernel produces gradients for all triangles every call, so end-to-end is a later
**scheduling** change (gradient every step on the whole genome instead of every-K
behind the gate), not a new pipeline. Not built now; nothing here blocks it.

## Milestones (de-risk in order)

1. **Brute-force kernel, standalone.** Synthetic stuck-big-triangle scene over a
   known background; run soft-raster + Adam; assert the **hard ΔE2000 improves** and
   the triangle visibly moves. Cheap kill-switch — ES untouched.
2. **Wire into `run_es`** behind `--gradient-polish`; verify the gate never regresses
   and measure steps/sec overhead vs flag-off.
3. **Tile-binned kernel** for the all-triangle 512² speed bar (binning pass +
   per-tile z-ordered composite; same loss/gate/Adam; gather-per-vertex backward).
4. **A/B on `goal.png`** at a matched budget: ΔE2000 vs `master` **and** eyeball the
   late-stage facets.

## Testing

- **Unit** (in `gradient.rs` `#[cfg(test)] mod tests`): soft forward → hard render as
  `τ → 0` on a single triangle; **finite-difference gradient check** on a tiny
  few-pixel/one-triangle case — the primary correctness guard for the backward pass.
- **Integration:** the Milestone-1 synthetic stuck-big-triangle scene — polish lowers
  the hard ΔE2000.
- **Regression:** existing tests + smoke test stay green (flag-off path unchanged);
  `cargo clippy --bin polygenvo` kept clean.

## Out of scope

- Path A (dirty-region fitness) — separate, quality-neutral effort.
- End-to-end gradient optimizer (designed-in as a later scheduling change only).
- Tiled full-genome **store-and-replay** autodiff (the per-triangle-per-pixel memory
  wall; rejected).
- `burn`/any framework or a separate wgpu device + CPU round-trip.

## Outcome (implemented 2026-06-10, branch `feat/gpu-differentiable-rasterizer`)

The full design was built CPU-reference-first and is **correct and validated**, but the
brute-force kernel is **not yet a net in-loop win** — production speed/cadence work is
deferred (as the plan anticipated). What landed and what we learned:

**Validated (the mechanism works):**
- CPU reference soft-rasterizer: forward Lab-MSE + analytic backward, **finite-difference
  verified** (`softras_ref.rs`).
- WGSL forward matches the CPU reference to **5.2e-5**; WGSL backward (CAS atomic-float
  scatter) matches the FD-verified CPU gradient to **5.6e-6** (all 6 params/vertex).
- **Milestone 1**: on a synthetic stuck-big-triangle scene, the polish lifts the real hard
  ΔE2000 fitness **484461 → 517980** (GPU) — matching the CPU reference (517983) almost
  exactly. The gate correctly rejects no-ops, leaving the genome byte-identical.
- Wired behind `--gradient-polish` (default **off**); the flag-off ES path is byte-for-byte
  unchanged (smoke test green).

**Why it isn't a live win yet (A/B on a downscaled `goal.png`):**
- **128²**: the first polish never returns within 60 s. The backward is **O(num_tris²) per
  pixel** (it recomputes the prefix composite and suffix transmittance per triangle) AND the
  CAS atomic-float scatter into only `num_tris*18` slots is **contention-bound** at realistic
  pixel/triangle counts. Effectively hangs.
- **64²**: the polish completes but at ~**120 steps/s** vs ~1300 baseline, and **every polish
  is rejected by the ΔE2000 gate**. During active coarse-phase evolution the genome is not
  "stuck" like the synthetic scene, so the Lab-MSE proxy's gains don't beat the hard ΔE2000
  parent at full res → pure overhead, no quality gain.

**Conclusion / next steps (a follow-up plan):**
1. **Tiled / gather-per-vertex backward kernel** (the deferred Milestone-3): bin triangles to
   tiles, cache per-triangle forward state to drop O(num_tris²) → O(num_tris), and replace the
   CAS scatter with gather-per-vertex (one thread per vertex over its bbox) to eliminate atomic
   contention. This is the prerequisite for any 512² use.
2. **Cadence/targeting**: only polish on a genuine *full-resolution plateau* (not during coarse
   evolution), and consider targeting the stuck/large triangles rather than all-triangle every
   time, so polishes have headroom to beat the gate.
3. **Proxy alignment**: the soft Lab-MSE proxy beats the ΔE2000 gate easily on grossly-wrong
   geometry but not on near-converged genomes; revisit if (1)+(2) still under-deliver.

The branch is a correct, tested foundation (35 tests green, clippy clean) that the tiled-kernel
plan builds on; it is not yet enabled for production runs.
