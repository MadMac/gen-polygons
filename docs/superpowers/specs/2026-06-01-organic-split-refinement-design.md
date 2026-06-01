# Organic Split-Driven Refinement — Design

**Date:** 2026-06-01
**Binary:** `polygenvo`
**Status:** Approved for planning

## Problem

The progressive triangle ramp (geometric phases up to `MAX_TRIANGLES`, see
[2026-06-01-progressive-triangle-ramp-design.md](2026-06-01-progressive-triangle-ramp-design.md))
did not raise the fitness ceiling. Two instrumented 140s runs on the 512² Van Gogh `goal.png`
(MAX_TRIANGLES=10000) showed:

- **Every promotion crashes fitness**, scaling with batch size: 150→240 dropped −47k,
  384→615 −92k, 984→1575 −122k. All are 512²→512² transitions, so resolution is not the
  variable — the cause is `grow_genome` appending a large batch of triangles on promotion.
- **The search deletes the surplus every phase.** The triangle count decays to exactly the
  ¾ `min_triangles` floor each phase (461=¾·615, 1182=¾·1575, 1890=¾·2520).
- **The ceiling does not move:** peak fitness was 960,970 at ~384 triangles and 961,188 at
  ~2520 triangles — identical. Going from 150 to 2520 triangles bought nothing.

An error-guided variant of `grow_genome` (placing the batch in high-error cells instead of
uniformly at random) was tried and **disproven**: re-scored fitness after each promotion was
within noise of the random baseline, and the count still collapsed to the floor. Late in
refinement the residual-error grid is diffuse, so "error-guided" ≈ random.

**Root cause:** `grow_genome` is the only place triangles enter the genome *without* passing
`(1+λ)` selection. It dumps many large triangles unconditionally; dumped triangles overwrite
already-good pixels (a flat semi-transparent blob cannot match detailed residual), so fitness
crashes and the search spends the phase deleting them. The `(1+λ)`-ES with single-triangle
mutations saturates around 150–400 triangles at ~961k and cannot exploit more capacity when
it is handed all at once.

## Approach

Make capacity grow **organically and quality-gated**: a triangle is added only when a
mutation candidate that adds it passes selection (strictly improves fitness). The vehicle is
a new `split` operator that subdivides an existing triangle into smaller ones covering the
same area, recoloured from the goal — so growth *refines* the image instead of disrupting it.

Alternatives considered and rejected during brainstorming:
- **Lower `MAX_TRIANGLES` to ~300** (accept the capacity limit). Pragmatic and stable, but
  abandons the goal of capturing fine detail.
- **Augment the dump with a split operator** (keep batch growth, add split for cleanup). The
  disruptive dumps — the user's actual complaint — would remain.

This design keeps the `(1+λ)`-ES, the goal pyramid, the fitness shader, the σ self-adaptation,
and the plateau-driven promotion *trigger*. It changes how the genome grows.

## Design

### 1. Organic growth model

`grow_genome` and its promotion-time call are **removed**. The genome changes only through
mutation candidates evaluated by `fitness_of_batch` and accepted by `(1+λ)` selection. Nothing
can add triangles unconditionally, so no promotion or growth event can crash fitness or pile
on un-vetted triangles.

### 2. The `split` operator

A new structural mutation (returns `OpKind::Structural`, carries no step size), selected from
a slice of the `mutate()` op distribution:

1. **Choose a triangle, biased to high error.** Roulette-select an error-grid cell with
   `sample_error_cell(error_grid, rng)`, map it to clip space, and split the existing triangle
   whose centroid is nearest that point. If the genome is empty, no-op.
2. **Subdivide into 4 by edge midpoints.** For parent vertices `v0,v1,v2` with edge midpoints
   `m01,m12,m20`, emit the 4 children `(v0,m01,m20)`, `(m01,v1,m12)`, `(m20,m12,v2)`,
   `(m01,m12,m20)`. These exactly tile the parent and preserve its CCW winding (so none are
   back-face culled). **Replace the parent in place** at its array index, so the children
   inherit the parent's z/draw position and the alpha-blend order is preserved.
3. **Inherit alpha, recolour RGB from the goal.** Each child keeps the parent's alpha; its RGB
   is sampled from the goal at the child's own centroid via `sample_goal_color`.
4. **Respect the cap.** No-op if the genome is at the phase cap (a split adds 3 triangles, so
   it is also a no-op if `count + 3 > cap`).

Because the children tile the same area with the same alpha, the render is identical to the
parent **except** for finer colour resolution. In a flat region the children get ~identical
colours → ~neutral → selection drops it (flat areas need no detail). Where the goal varies
under the triangle, the four locally-coloured children lower per-pixel ΔE → strict improvement
→ accepted. Detail therefore accretes exactly where the image is wrong, gated by fitness.

### 3. Phases and promotion

With growth organic, per-phase triangle *targets* are obsolete; phases revert to a
coarse-to-fine **pyramid + σ schedule** plus a capacity cap.

- **`Phase` carries `cap`, `pyramid_level`, `initial_sigma_pos`, `initial_sigma_col`** — the
  `triangles` *target* field becomes `cap` (a ceiling, not a fill target).
- **`production_phases()` (the geometric count generator) is removed.** The schedule is ~4
  phases: 128² → 256² → 512² → 512²-fine. Early coarse phases get a small cap (a few hundred)
  so the search does not waste triangles on a blurry target; the **final phase's
  `cap = MAX_TRIANGLES`** (the existing 10000 knob is preserved).
- **Promotion (same plateau trigger) advances the pyramid level, raises the cap, and resets
  σ — and nothing else.** No `grow_genome`, no genome surgery. It re-scores at the new level
  exactly as today, so a promotion no longer perturbs the image; it only sharpens evaluation
  and unlocks capacity.
- **`min_triangles` becomes a small absolute floor of 8** (was ¾×target); `delete` prunes down
  to it.
- **Initial genome** still seeds via `init_genome` at a small starting count (the first phase's
  cap, capped to a small number) — the correct random cold start with no image yet.

### 4. Operator mix and seed-radius scaling

`mutate()` gains a `split` slice. Proposed weights (tunable):

| op | now | new |
|---|---|---|
| positional nudge | 40 | 38 |
| recolour | 25 | 24 |
| alpha | 13 | 12 |
| **split (new)** | – | **10** |
| z-swap | 8 | 5 |
| add (inject) | 6 | 5 |
| relocate | 4 | 3 |
| delete | 4 | 3 |

`add` (error-guided injection of a *new* triangle into an uncovered high-error region) stays —
it complements `split`: `add` introduces coverage for unmodeled features, `split` refines
coverage that exists. Both are fitness-gated and both respect the cap.

**Seed-radius scaling.** Fresh triangles from `add` seed at a fixed radius 0.2, which is huge
at high counts. Scale to the current triangle scale:
`radius = (0.2 * sqrt(START_TRIS as f32 / count as f32)).clamp(0.02, 0.2)` (≈0.2 at ~40
triangles, ≈0.013 at 10000), where `START_TRIS` is the first phase's starting count. `split`
needs no radius — it inherits the parent's geometry.

### 5. Out of scope

No change to the `(1+λ)` selection rule, the fitness shader, the σ self-adaptation math, the
pyramid construction, the plateau-detection trigger, or the render pipeline.

## Testing

**Unit tests (GPU-free, seeded RNG)** for `split`:
- **Tiling/containment:** the 4 children's combined signed area ≈ the parent's, and every
  child vertex lies within the parent triangle.
- **Winding preserved:** each child's signed area has the same sign as the parent (CCW).
- **Alpha inherited:** every child vertex's alpha equals the parent's.
- **Recolour:** on a non-uniform goal the children get differing RGB; on a uniform goal they
  get ~identical RGB.
- **Cap respected:** `mutate` forced into the split branch grows the genome by exactly 3 below
  the cap, and is a no-op at the cap.

**Phase schedule test:** caps are non-decreasing and the final phase's `cap == MAX_TRIANGLES`.

**Integration guard:** the existing `ga_improves_on_synthetic_checker` smoke test stays
(updated for the `Phase { cap, .. }` field); still asserts fitness improves and never regresses.

**Cleanup:** delete `grow_genome`, its promotion call, the `production_phases` generator, and
the `production_phases_schedule` test.

**Behavioral validation (the real proof):** a `cargo run --release` compared against the
baseline log for (1) no fitness drop at promotions, (2) the triangle count climbing smoothly
and *holding* instead of decaying to a floor, (3) whether the ceiling clears ~961k, and (4)
cleaner snapshots in `triangles/`.
