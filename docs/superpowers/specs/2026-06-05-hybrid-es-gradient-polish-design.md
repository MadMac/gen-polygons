# Hybrid ES + Gradient-Descent Polish — Design

**Date:** 2026-06-05
**Binary:** `polygenvo`
**Status:** Approved for planning

## Problem

The (1+λ)-ES hits a quality ceiling that is **not** a representation or capacity limit:
a large triangle committed during the coarse phase, in a region that looked fine at
128², becomes a **permanent local optimum** once its silhouette turns out wrong at 512².
Three contained, elitist-compatible operators were tried this session and all failed
(targeted-split repair, scale-relative vertex nudges, affine/resize) — each fitness-neutral
or slightly worse, none fixing the hard-edged facets.

They fail for one root reason, now well established: **fixing a wrong big triangle is a
downhill-first move.** The instant you reshape/shrink/delete it you lose the coverage it
provided; the compensating detail only pays off several steps later. Elitist (1+λ) accepts
a step only if it is *immediately* better, so it rejects the first step of any real repair.
No mutation operator can get past this — the limit is the acceptance rule plus the fact
that, under a *hard* rasterizer, the loss is **discontinuous in vertex positions** (a vertex
move flips pixel coverage in jumps), so the search has no signal pointing toward the fix.

**Gradient descent through a differentiable (soft) rasterizer dissolves exactly this wall:**
the gradient of the loss w.r.t. a triangle's three vertices points straight at the reshape
that lowers error, and all three vertices move coherently in one step — no elitist rejection,
no "downhill first."

## Approach

Keep the (1+λ)-ES as the driver for the **discrete** decisions it is good at (add / split /
delete / z-order / coarse-to-fine schedule), and add a **periodic gradient-descent polish**
of the **continuous** parameters (vertex positions + colors) of the current best, through a
soft rasterizer with autodiff. The ES proposes structure; gradient descent refines geometry
and colour where a hard rasterizer gives no usable signal.

Decisions locked during brainstorming:

- **Scope:** positions *and* colors (the only version that fixes silhouettes; colors-only
  would not break the wall).
- **Autodiff:** a pure-Rust framework — **`burn`** on its **wgpu** backend (matches the
  existing wgpu/GL setup, stays cross-platform; `candle` leans CUDA/Metal). Autodiff derives
  the backward pass; we do not hand-roll gradient math.
- **Cadence:** interleaved — every *K* accepted improvements, polish the current best, then
  hand it back to the ES, so big triangles get reshaped while still fixable.
- **Differentiable region:** **subset polish** — only the top-*M* "stuck" triangles are
  differentiated; the rest are a fixed background. A full-genome, full-res soft rasterizer is
  memory-infeasible (coverage is one value per pixel per triangle: 512²×~1000 ≈ 2.6×10⁸
  floats forward, ×autodiff intermediates = many GB).

### Alternatives rejected

- **Colors/alpha-only gradient descent** (analytic, no soft rasterizer): much smaller, but
  does not touch silhouettes — would not break the wall this is for.
- **Simulated-annealing / accept-worse acceptance:** the cheaper escape mechanism; a real
  alternative, but the user chose the gradient route for its larger payoff and because it
  gives a *directed* fix rather than a stochastic one.
- **Full-genome polish (coarse-res or tiled):** coarse-res loses fine-detail signal; tiled
  full-res adds large bookkeeping complexity (triangles spanning tiles). Subset polish bounds
  memory and directly targets the big-triangle problem.

## Design

### Module boundary

A new isolated module **[`gradient.rs`](../../../src/polygenvo/gradient.rs)** with one entry point:

```
gradient::polish(genome: &mut Vec<Vertex>, goal_lab: &[[f32;4]],
                 hard: &FitnessCalc, cfg: &PolishCfg) -> bool   // true if kept
```

`es.rs` is the only caller. With the feature flag off, `run_es` is byte-for-byte today's
behavior. `gradient.rs` depends on `burn` + `genome` + the goal Lab; nothing depends on it.

burn runs on its **own** wgpu device (cubecl), separate from the app's `FitnessCalc` device,
so the polish does a **CPU round-trip**: upload the subset's params and the fixed base image
to burn tensors, optimize, read the params back. Polish is periodic (every *K* improvements),
so the round-trip is amortized.

### The polish step

1. **Select subset.** Top-*M* triangles by `area × local-error` (reuses the residual error
   grid; needs a small `triangle_area` helper in `genome.rs` — the one prototyped and reverted
   earlier this session). M ≈ 16–64 (tunable). These are the stuck big triangles.
2. **Fixed base.** Hard-render the genome **without** the subset into one RGBA image, using
   the existing rasterizer path. This is the non-differentiable background.
3. **Differentiable composite.** Soft-rasterize the M subset triangles **over** the base, in
   their draw order (see soft rasterizer below).
4. **Optimize.** Adam (burn's optimizer), N ≈ 20–50 steps, on the subset's vertex positions
   and per-vertex RGBA, minimizing **mean-squared error in CIELAB** vs the goal (goal Lab is
   already precomputed in `fitness.rs`; the soft composite is converted to Lab differentiably).
5. **Write back + elitist gate.** Splice the optimized params into the genome at their
   original indices, **re-score the whole genome with the real hard ΔE2000 renderer**, and
   keep the polish **only if it beats the pre-polish fitness**. Otherwise discard and restore.
   This preserves the (1+λ) no-regression guarantee exactly and is the safety net for the two
   approximations below.

### Soft rasterizer (burn tensor ops, autodiff)

Per subset-triangle `t`, per pixel `p`:

- **Coverage** `A(p,t) = sigmoid(-signed_dist(p,t) / τ)` — SoftRas-style; `signed_dist` is the
  signed distance to the triangle (negative inside), differentiable in the vertex positions.
  `τ` is a temperature (start softer for gradient flow, optionally anneal toward sharp).
- **Color** by barycentric interpolation of the triangle's three vertex colors at `p` —
  differentiable in the colors (matches the hard renderer's per-vertex gradient shading).
- **Composite** over the fixed base in draw order:
  `C ← A·α·col + (1 − A·α)·C`, sequentially over the M triangles (M small → cheap graph).
- Convert `C` → CIELAB differentiably (the Lab cube-root is differentiable) for the loss.

### Two deliberate approximations (both caught by the gate)

- **Soft ≠ hard.** The soft renderer blurs edges; step 5 re-scores with the exact renderer, so
  a polish that only looked good under soft-raster is rejected.
- **Draw order.** Compositing the subset *on top of* the base ignores that a subset triangle
  may originally sit behind later ones. Step 5 rejects any case where this misleads us.

### Integration & cadence

- A `PolishCfg` in `EsConfig` (with `enabled: bool`, `every_k`, `subset_m`, `steps_n`,
  `lr`, `tau`). Default **off**; enabled by a new CLI flag **`--gradient-polish`** in `main.rs`.
- In `run_es`, after an accepted improvement, if `enabled` and the improvement count is a
  multiple of `every_k`, call `gradient::polish`. On a kept polish, refresh `current_fitness`
  and `parent_error_grid` (the polish already re-scored with the hard renderer).

### Dependencies & risks (eyes-open)

- **New deps:** `burn`, `burn-autodiff`, `burn-wgpu`. Heavy; breaks the repo's framework-free
  ethos (accepted). Pin versions; document in `Cargo.toml` and CLAUDE.md.
- **burn wgpu maturity** is an unknown → de-risked by Milestone 1 below before any ES wiring.
- **Fidelity gap / rejected polishes:** if soft-raster polish rarely improves the hard score,
  the work is wasted. Measured directly by Milestone 1.
- **Performance:** per-polish soft-raster (M·H·W) + N Adam steps + CPU round-trip; periodic, so
  amortized, but burn init and per-polish overhead could still be significant. Measure steps/sec
  with the flag on vs off.

## Milestones (de-risk before committing to the full integration)

1. **Prove the core, standalone.** A synthetic scene with one deliberately-misplaced big
   triangle over a known background; run the soft-raster + Adam polish; assert the **hard
   ΔE2000 improves** and the triangle visibly moves toward correct. If this does not convince,
   stop here cheaply — minimal sunk cost, ES untouched.
2. **Wire into `run_es`** behind `--gradient-polish`; verify the elitist gate never regresses
   fitness and measure steps/sec overhead.
3. **A/B on `goal.png`** (~150 s, flag off vs on): compare final ΔE2000 and eyeball the
   late-stage hard-edge facets — the symptom this is meant to fix.

## Testing

- **Unit:** soft-raster forward converges to the hard render as `τ → 0` on a single triangle;
  finite-difference gradient check on a tiny case (a few pixels, one triangle).
- **Integration:** the Milestone-1 synthetic stuck-big-triangle scenario as a test — polish
  lowers the hard ΔE2000.
- **Regression:** existing 24 tests stay green; the gradient path is behind the flag, so the
  default ES, the smoke test, and CI-less validation are unaffected.

## Out of scope

- Full-genome or tiled differentiable rendering (memory/complexity — subset polish only).
- Replacing the ES with end-to-end gradient descent (structure decisions stay discrete/ES).
- Colors-only polish (does not fix silhouettes).
- Sharing one wgpu device between the app and burn (CPU round-trip is acceptable for periodic
  polish; revisit only if profiling shows it dominates).
