# ES Plateau Improvements — Design

**Date:** 2026-05-29
**Binary:** `polygenvo`
**Status:** Approved for planning

## Problem

The (1+λ)-ES in [main.rs](../../../src/polygenvo/main.rs) "improves then crawls": fitness
climbs quickly early, then improvements become rare and tiny and visible progress
effectively stalls. The goal is to raise the final-quality ceiling **and** reach good
results faster (both throughput and search quality matter).

### Root causes identified in the current code

1. **One shared `sigma` for position, color, and alpha.** Position is clip-space
   `[-1,1]`; color/alpha are `[0,1]`. A single step size cannot suit both, and the 1/5
   success rule shrinks all of them in lockstep. ([main.rs:604-676](../../../src/polygenvo/main.rs))
2. **Uniform perturbations, not Gaussian.** `random_range(-sigma..sigma)` has no tail, so
   the search cannot make the occasional larger exploratory jump ES relies on.
3. **Blind triangle placement.** The `add` operator drops triangles at random centers;
   late in a run, residual error is concentrated in specific regions, so random placement
   almost never helps and structural progress stalls.
4. **Serial GPU evaluation throttles throughput.** Each candidate does its own
   `submit → poll(wait) → readback` ([main.rs:374-387](../../../src/polygenvo/main.rs)); the
   GPU idles between the λ candidates of every step.
5. **Coarse fitness quantization.** Per-pixel ΔE is truncated to a `×1000` u32
   ([fitness.wgsl:84](../../../src/polygenvo/fitness.wgsl)); tiny-but-real improvements can
   round to zero and be rejected.

## Approach

Keep the (1+λ)-ES, phase schedule, goal pyramid, and selection logic unchanged. Land four
targeted changes underneath them. Each is independently testable; the existing smoke test
remains the regression guard.

This is the "tune the engine" approach chosen over (a) adding a population/annealing escape
mechanism and (b) replacing the search core with CMA-ES. CMA-ES in particular assumes a
**fixed problem dimension**, which conflicts with the variable-length genome (triangles are
added and deleted), so it was rejected as high-complexity for uncertain gain. An escape
mechanism can be added later if a capacity/trap plateau appears after these fixes.

## Confirmed facts

- `goal.png` is **512×512**. The pyramid (`build_pyramid`) is therefore 128² / 256² / 512².
- The largest accumulator spans 262,144 pixels, bounding the fitness scale at
  `2³² / 262144 ≈ 16384`. We use **`FITNESS_SCALE = 8192`** (8× finer than today's 1000,
  with comfortable headroom).
- `rand 0.10` ships no normal distribution; Gaussian sampling uses a small **Box-Muller**
  helper rather than adding `rand_distr`, consistent with the project's tight dependency set.

## Components

### 1. Batched GPU evaluation

The largest throughput lever. Algorithmically identical to the current loop (same
candidates, same selection) — only evaluation is batched, so there is no behavioral change.

- **New API:** `FitnessCalc::fitness_of_batch(&[&[Vertex]]) -> Vec<Eval>` where
  `struct Eval { score: usize, error_grid: Vec<u32> }` (grid length `G²`). The existing
  `fitness_of(&[Vertex]) -> usize` becomes a thin batch-of-1 wrapper returning `.score`, so
  initial scoring, `snapshot`, and any other callers are unchanged.
- **Vertex buffer:** sized `LAMBDA × MAX_VERTICES × sizeof(Vertex)` (6 × 450 × 28 ≈ 75 KB).
  Candidate `i` is written at byte offset `i × MAX_VERTICES × sizeof(Vertex)`. `sizeof(Vertex)`
  is 28 (4-aligned), so every per-candidate offset is a legal vertex-buffer offset
  (`COPY_BUFFER_ALIGNMENT` = 4). Per-candidate vertex counts are tracked so each `draw`
  call uses the candidate's actual length.
- **Single encoder:** loop `i in 0..batch_len`: a render pass (clear → draw candidate `i`)
  into **one reused render target**, then a compute pass scoring candidate `i` into output
  slot `i`. Within a single command buffer, passes execute in submission order with
  automatic resource barriers, so reusing one render target across candidates is correct.
- **Slot selection:** the compute dispatch selects its output slot per dispatch via a 4-byte
  **immediate constant** (preferred; avoids 256-byte dynamic-offset padding). A dynamic
  uniform offset is an acceptable fallback if immediate data proves awkward under wgpu 29.
- **Readback:** one `MAP_READ` buffer holds `batch_len × (1 score u32 + G² grid u32)`.
  Map once, poll once, parse all results. The λ sync points of today collapse to **one**.
- **Capacity:** with `LAMBDA = 6`, `MAX_VERTICES = 450`, `G = 16`: vertex buffer ≈ 75 KB,
  accumulator+grid readback = 6 × (1 + 256) × 4 B ≈ 6 KB. Negligible.

### 2. Decoupled Gaussian self-adaptive step sizes

- **Gaussian perturbations.** `mutate` replaces uniform `(-σ,σ)` jitter with `N(0,σ)` via a
  Box-Muller helper. Most samples are small (fine refinement) with a tail for occasional
  larger jumps.
- **Two step sizes.** `σ_pos` drives vertex-position nudges; `σ_col` drives recolor and
  alpha mutations. Structural operators (add / delete / z-swap / relocate) have no step size.
- **Per-type 1/5 rule.** During the λ loop, classify each candidate by the operator it used
  — *positional* (vertex nudge), *chromatic* (recolor, alpha), or *structural* (everything
  else). Count, per category, how many candidates were generated and how many **beat the
  parent** (`f > current_fitness`; note multiple candidates may beat the parent even though
  only the best is selected). Over `SIGMA_WINDOW` steps, adapt `σ_pos` and `σ_col`
  independently toward a ~20% beat-the-parent rate using the existing multipliers
  (×1.15 above target, ×0.85 below), each with its own clamp range. Structural candidates do
  not feed σ adaptation.
- **Phase fields.** `Phase::initial_sigma` splits into `initial_sigma_pos` and
  `initial_sigma_col`. The final-phase plateau "sigma restart" restarts **both**.

### 3. Error-guided placement

- **Error grid output.** The fitness compute pass bins each pixel's ΔE into a `G × G` grid
  (`G = ERROR_GRID_DIM = 16`) per candidate. Cell index is
  `(x * G / width, y * G / height)`. The grid uses a modest scale (its magnitudes are used
  only relatively, for roulette weighting), accumulated with `atomicAdd` into the slot's
  grid region.
- **Parent residual cache.** When a candidate is accepted, cache its `error_grid` as
  `parent_error_grid`. It is initialized from the first scoring of the seed genome and
  refreshes whenever the parent changes — free, since it rides on the batched readback. On
  phase promotion / regrow the parent is re-scored, refreshing the grid for the new level.
- **Error-guided operators.**
  - `add`: sample a grid cell by **roulette proportional to cell error**, place the new
    triangle's center at a random clip-space point inside that cell, color sampled from the
    goal at that point (reusing `sample_goal_color`).
  - `relocate` (new operator): pick an existing triangle and move its center to a
    roulette-sampled high-error cell, recycling triangles that are not earning their keep.
  Operator probabilities are retuned to introduce `relocate` and keep total structural
  probability roughly as today; exact splits are a tuning detail for the plan.

### 4. Finer fitness accumulation

- **Workgroup reduction.** Each 8×8 workgroup sums its 64 per-pixel ΔE values in
  `var<workgroup>` shared memory (with a barrier), then a single thread `atomicAdd`s the
  scaled workgroup sum into the score accumulator. Truncation happens once per 64 pixels
  instead of per pixel (~64× less quantization noise), so small real improvements stop
  rounding to zero.
- **Scale.** `FITNESS_SCALE = 8192`. `S` and `G` are passed in the `params` uniform's two
  spare `u32` slots (currently `pad0`/`pad1`), so the shader and the Rust normalizer
  (`max_total = texture_size² × S`) share a single source of truth — no duplicated constant
  across the `include_str!` boundary.

## Data flow (per step)

```
parent, σ_pos, σ_col, parent_error_grid
  → generate λ candidates (each tagged: positional | chromatic | structural;
      add/relocate sample cells from parent_error_grid)
  → fitness_of_batch(candidates)  →  one submit, one readback  →  λ Eval { score, error_grid }
  → pick best score; if best > current_fitness:
        accept candidate, current_fitness = best, parent_error_grid = best.error_grid
  → update per-type beat-the-parent counters
  → every SIGMA_WINDOW steps: adapt σ_pos, σ_col independently (1/5 rule)
  → every PLATEAU_WINDOW steps (after PHASE_MIN_STEPS): promote phase, or restart both σ
```

## Error handling

- Failure model unchanged: panic on `map_async`/`poll` error, matching current code.
- Debug assertions: every candidate ≤ `MAX_VERTICES` vertices; batch length ≤ `LAMBDA`.
- Variable per-candidate vertex counts are tracked explicitly; each `draw` uses the
  candidate's own count (a genome that shrank via delete still renders correctly).

## Testing

The smoke test is the only regression guard; these changes touch `run_es`, `FitnessCalc`,
and both `.wgsl` files, so `cargo test --bin polygenvo` must pass before the work is
considered safe.

- **Update the existing smoke test** `ga_improves_on_synthetic_checker` for the split
  `Phase` sigma fields. It keeps its assertions (`0 < final ≤ 1_000_000`, `final ≥ initial`)
  and `snapshot_every: None`.
- **Batch consistency test:** `fitness_of_batch(&[g, g, g])` returns three equal scores,
  each equal to `fitness_of(g)` for the same genome — guards the riskiest new code.
- **Grid invariant test:** the sum of a candidate's `error_grid` cells ≈ its `score`
  accumulator (modulo the score/grid scale ratio), validating the binning.
- **Box-Muller sanity test:** over a large sample, mean ≈ 0 and standard deviation ≈ σ.
- **Manual validation (per CLAUDE.md):** run `cargo run --release --bin polygenvo`; compare
  steps/sec and fitness-vs-time against current `master`, and eyeball `triangles/` frames.

## Tunables added / changed

- `const FITNESS_SCALE: u32 = 8192;`
- `const ERROR_GRID_DIM: u32 = 16;`
- Per-type sigma clamp ranges (`σ_pos`, `σ_col`).
- `Phase { triangles, pyramid_level, initial_sigma_pos, initial_sigma_col }` — both
  production `PHASES` and the smoke test's inline `Phase` updated.

## Out of scope

- Population / island restarts or annealing acceptance (Approach 2).
- CMA-ES or differential-evolution search core (Approach 3).
- Raising `MAX_VERTICES` / per-phase triangle caps — the diagnosis is diminishing returns,
  not a hard capacity wall. Revisit only if a capacity plateau appears after these changes.
- CI. There is still no CI; validation remains the smoke test plus manual eyeballing.
