# Progressive Triangle-Count Ramp — Design

**Date:** 2026-06-01
**Binary:** `polygenvo`
**Status:** Approved for planning

## Problem

The (1+λ)-ES plateaus near **964k–965k similarity** at its final phase (150 triangles,
512² pyramid level). An A/B on the 512² Van Gogh `goal.png` confirmed both master and the
plateau-improvements branch flatten at roughly the same ceiling, and eyeballing the frame
shows composition and colour regions are captured but fine facial detail is absent. That is
the signature of a **representation-capacity** limit, not a search limit: 150 alpha-blended
triangles cannot resolve fine detail in a 512² portrait.

The hard cap is [main.rs:13](../../../src/polygenvo/main.rs#L13) — `MAX_VERTICES = 450`
(150 triangles) — and the schedule that climbs to it is the hand-written 4-entry `PHASES`
array at [main.rs:601-606](../../../src/polygenvo/main.rs#L601-L606). The goal is to let the
ES keep adding triangles progressively to a much higher ceiling (~1000 for now), with that
ceiling controlled by a single, easily-changeable constant.

## Approach

Keep the (1+λ)-ES, mutation operators, fitness shader, σ-adaptation math, and
plateau-driven promotion **unchanged**. The promotion mechanism at
[main.rs:1110-1173](../../../src/polygenvo/main.rs#L1110-L1173) already walks `phase_idx`
through `cfg.phases`, grows the genome with `grow_genome`, re-scores, and resets σ on each
promotion — it simply needs more phases to walk through and a larger vertex buffer to grow
into.

This change is therefore purely **capacity + schedule shape**:

1. Derive the buffer cap from a triangle-denominated constant so one knob governs the
   ceiling.
2. Keep the existing 4 coarse-to-fine phases as a fixed warmup, and **auto-generate** the
   higher-count phases above 150 up to the cap with a geometric growth rule.

Alternatives considered and rejected:
- **Hand-write every phase up to 1000.** Gives full per-phase control of σ and pyramid
  level, but the ceiling is no longer "one knob" — raising it means editing the list by
  hand. Rejected against the easily-changeable requirement.
- **Arithmetic (+150) growth.** Smooth and even, but many promotions and slow to reach high
  counts; more total runtime for little benefit. Geometric concentrates bigger relative
  jumps early where added capacity helps most.

## Design

### 1. Single source of truth for the cap

Replace the hard-coded `MAX_VERTICES` with a triangle-denominated cap that everything
derives from:

```rust
const MAX_TRIANGLES: usize = 1000;          // the one knob you change
const MAX_VERTICES: usize = MAX_TRIANGLES * 3;
const PHASE_GROWTH: f32 = 1.6;              // geometric multiplier for auto phases
```

`MAX_VERTICES` continues to feed the vertex-buffer sizing in `FitnessCalc::new` and the
`verts.len() <= MAX_VERTICES` assert in `fitness_of_batch` exactly as today. At 1000
triangles the per-candidate vertex buffer is 3000 × 28 B, ×`LAMBDA` (6) ≈ 504 KB —
negligible. No buffer-layout, batching, or shader changes are required; the existing
`MAX_VERTICES`-derived sizing just scales.

### 2. Warmup phases stay hand-tuned; the high end is generated

Keep the existing 4 phases as a `WARMUP_PHASES` const — the coarse-to-fine pyramid climb
40 → 80 → 120 → 150, carrying their per-phase σ and `pyramid_level` (that part already
works and sets up the pyramid traversal). Add a generator that, starting from the last
warmup count (150), appends geometric phases up to `MAX_TRIANGLES`, all at the finest
pyramid level and the finest hand-tuned σ:

```
fn production_phases() -> Vec<Phase>:
    debug_assert!(MAX_TRIANGLES >= last WARMUP_PHASES count)   // coherence guard
    phases = WARMUP_PHASES.to_vec()
    n = last warmup count (150)
    loop:
        n = ceil(n * PHASE_GROWTH)
        if n >= 0.85 * MAX_TRIANGLES: break    // avoid a near-duplicate final phase
        push Phase { triangles: n, finest pyramid level, finest σ }
    push Phase { triangles: MAX_TRIANGLES, finest level, finest σ }   // exact cap
```

`EsConfig::production()` calls `production_phases()` instead of `PHASES.to_vec()`.

**Resulting schedule with the defaults:** 40 → 80 → 120 → 150 → 240 → 384 → 615 → 1000.
Bumping `MAX_TRIANGLES` to 1500 automatically extends the tail; lowering it shortens it.

**Design notes:**
- *Pyramid level for auto phases* is the finest (full-res, level 2). The hand-tuned phases
  already reach it at 120 triangles, so everything above stays there.
- *Initial σ for auto phases* reuses the finest hand-tuned values (σ_pos 0.05 / σ_col
  0.04). The 1/5-success rule re-adapts σ within each phase, so the seed matters little.
- *The 0.85 snap-to-cap rule* prevents emitting a near-duplicate penultimate phase (e.g.
  984 immediately before 1000); the final jump (615 → 1000, ×1.63) stays consistent with
  the geometric ratio.
- *`MAX_TRIANGLES` governs the high-end ceiling, not the warmup floor.* The 4 warmup phases
  are fixed; the cap only extends/truncates the generated tail above 150. The
  `debug_assert` fails loudly if a future edit sets the cap below the warmup ceiling.

### 3. Runtime implications — no logic change

Each phase requires `PHASE_MIN_STEPS` (400) before it can promote, so 4 extra phases add
≥1,600 steps of floor before the cap is reached — well under `MAX_STEPS = 500_000`, so that
constant is unchanged. Per-step cost: rendering 1000 triangles × `LAMBDA` candidates at
512² is cheap on the GPU, and fitness is per-pixel (triangle-count independent). Genome
`insert`/`remove` in `mutate` stay O(n) on a ≤3000-element `Vec` — trivial.

### 4. Out of scope

No change to mutation operators, the fitness shader, σ-adaptation math, pyramid
construction, or the selection logic. This is purely a capacity + schedule-shape change.

## Testing

The smoke test `ga_improves_on_synthetic_checker` builds its own single-element `phases`
Vec (6 triangles) and never calls `production_phases()`, so it is untouched; `MAX_VERTICES
= 3000` trivially accommodates 6 triangles. Per project convention, run `cargo test --bin
polygenvo` after the change as the regression guard. Substantive validation remains a
manual run (`cargo run --release --bin polygenvo`) eyeballing the snapshots in `triangles/`
to confirm the ceiling actually rises past ~965k as triangle count grows.

A small unit test on `production_phases()` is worth adding: assert the schedule is
monotonically increasing, starts with the 4 warmup phases, ends exactly at `MAX_TRIANGLES`,
and contains no duplicate counts — this pins the generator's contract cheaply (no GPU
required).
