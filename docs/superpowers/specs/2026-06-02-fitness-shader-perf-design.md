# Fitness-shader performance — design / follow-up

Status: **proposed** (deferred out of the 2026-06-02 readability refactor, which was
kept score-preserving). Implement and validate as its own change.

## Why

Per the `batched-eval-perf-characteristic` memory, the final 512² phase is **GPU
per-pixel-bound**: batching and CPU-sync optimisations don't help there because the
bottleneck is the work the compute shader does per pixel, not dispatch/readback
overhead. So the lever is the shader's per-pixel cost. These changes shift fitness
scores slightly (the goal conversion rounds differently), so they need re-validation
against the smoke test plus an eyeball run — that's why they're split out of the pure
refactor.

## 1. Precompute goal CIELAB once per pyramid level (headline)

Today [fitness.wgsl](../../../src/polygenvo/fitness.wgsl) converts **both** the goal
and the rendered pixel sRGB→linear→XYZ→Lab on every dispatch:

```
let goal_lab     = xyz_to_lab(linear_rgb_to_xyz(goal_rgb));      // constant per level
let rendered_lab = xyz_to_lab(linear_rgb_to_xyz(rendered_rgb));  // varies per candidate
```

The goal is fixed for the lifetime of a `FitnessCalc`, so its Lab is recomputed
λ × steps times for no reason — including 3 `pow(x, 1/3)` (cube roots) per pixel.

**Approach:** bake the goal to Lab once in `FitnessCalc::new` and have the scoring
shader read it directly.

- Add a `goal_lab` storage buffer (or `Rgba32Float` texture), `width*height` `vec3<f32>`.
- One-time compute pass (or a tiny dedicated shader) at construction fills it from the
  existing `goal_texture`. Alternatively compute Lab on the CPU and upload — the goal
  image is already in RAM, and this keeps the per-frame shader free of a goal-decode path.
- The scoring shader's per-pixel work drops to: load goal-Lab, convert the rendered
  pixel, take ΔE76. Roughly halves the colour-conversion math at the dominant cost point.

**Risk / validation:** scores change in the low bits (goal Lab now quantised once at
build instead of recomputed in `Rgba8UnormSrgb` precision each pass). Confirm
`cargo test --bin polygenvo` still passes (the smoke test asserts non-regression, not
exact values) and that a ~1–2 min run still climbs into the ~960k region with a clean
portrait. Keep the `ERROR_GRID_DIM`/`GRID_DIM` mirroring contract intact.

## 2. Cheaper result-buffer clear (small, allocation-free)

`fitness_of_batch` zeroes the result buffer every step by allocating a heap vec and
uploading it ([fitness.rs](../../../src/polygenvo/fitness.rs), the
`vec![0u8; SLOT_STRIDE * LAMBDA]` + `write_buffer`). Replace with
`encoder.clear_buffer(&result_buffer, 0, None)` at the top of the encoder — no
per-step allocation, no CPU→GPU copy of zeros. Tiny but free.

## 3. Note only — genome edit cost

`mutate`'s `split`/`insert`/`remove` are O(n) memmoves over up to ~30k vertices
(`MAX_TRIANGLES = 10000`). Negligible against GPU cost today, but worth revisiting if
`MAX_TRIANGLES` grows substantially. Not worth changing now.

## Out of scope / rejected

- Don't attack dispatch/readback structure for the final phase (proven not the
  bottleneck — see `batched-eval-perf-characteristic`).
- Don't reintroduce unconditional batch triangle growth (see
  `final-phase-plateau-capacity-bound`).
