# Organic Split-Driven Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace unconditional batch triangle growth with a fitness-gated `split` operator so the genome grows only where added detail improves the fit — eliminating the promotion "dump/mess" and the count-collapse.

**Architecture:** Add a `split_triangle` helper (midpoint-4 subdivision, recoloured from the goal) and a `grow_by_split` mutation that replaces one high-error triangle in place. Wire it into `mutate()`'s op distribution. Then restructure phases so `Phase` carries a `cap` (not a fill target), promotion stops calling `grow_genome` (which is deleted along with `production_phases`), and `add`'s seed radius shrinks as the count grows.

**Tech Stack:** Rust (edition 2024), `wgpu` 29, single binary `polygenvo`. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-06-01-organic-split-refinement-design.md](../specs/2026-06-01-organic-split-refinement-design.md)

---

## File Structure

All changes are in `src/polygenvo/main.rs` (the project keeps the whole binary in one file; this change follows that pattern).

- `split_triangle` — pure geometry helper near the other seeding helpers (`error_seeded_triangle`).
- `grow_by_split` — mutation helper beside `split_triangle`.
- `mutate()` — op-distribution rebalance + new `split` branch + radius-scaled `add`.
- `Phase` / `PHASES` / `EsConfig::production` / `run_es` — phase restructure.
- `mod tests` — new unit tests; remove the obsolete `production_phases_schedule` test.

---

## Task 1: `split_triangle` geometry helper

A pure function that subdivides a triangle into 4 midpoint children that tile it, preserve CCW winding, inherit the parent alpha, and recolour each child from the goal at its centroid. Unused until Task 2, so it carries a temporary `#[allow(dead_code)]`.

**Files:**
- Modify: `src/polygenvo/main.rs` — add `split_triangle` after `error_seeded_triangle` (after line 822)
- Test: `src/polygenvo/main.rs` — `mod tests`

- [ ] **Step 1: Write the failing tests**

Add these tests inside the `mod tests` block, immediately before the `ga_improves_on_synthetic_checker` test (currently line 1408). They reference helpers `tri_signed_area` and `make_gradient_goal` defined in the same block:

```rust
    fn tri_signed_area(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
        0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
    }

    // Per-column gradient: every distinct x maps to a distinct R channel, so two
    // points with different x always get different colours.
    fn make_gradient_goal(size: u32) -> GoalImage {
        let mut buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let v = (x * 255 / (size - 1)) as u8;
                buf.put_pixel(x, y, Rgba([v, 128, 255 - v, 255]));
            }
        }
        GoalImage { goal_image: buf }
    }

    #[test]
    fn split_triangle_tiles_and_preserves_winding() {
        let goal = make_solid_goal(64, [100, 150, 200]);
        let v0 = Vertex { position: [-0.5, -0.5, 0.0], color: [0.1, 0.2, 0.3, 0.5] };
        let v1 = Vertex { position: [0.5, -0.5, 0.0], color: [0.1, 0.2, 0.3, 0.5] };
        let v2 = Vertex { position: [0.0, 0.5, 0.0], color: [0.1, 0.2, 0.3, 0.5] };
        let parent_area = tri_signed_area(v0.position, v1.position, v2.position);
        assert!(parent_area > 0.0, "test fixture must be CCW");

        let children = split_triangle(v0, v1, v2, &goal);
        assert_eq!(children.len(), 12, "4 child triangles = 12 vertices");

        let mut total = 0.0;
        for t in 0..4 {
            let b = t * 3;
            let area = tri_signed_area(children[b].position, children[b + 1].position, children[b + 2].position);
            assert!(area > 0.0, "child {t} must keep CCW winding (got area {area})");
            total += area;
        }
        assert!((total - parent_area).abs() < 1e-5, "children must tile parent: {total} vs {parent_area}");
    }

    #[test]
    fn split_triangle_inherits_alpha() {
        let goal = make_solid_goal(64, [100, 150, 200]);
        let a = 0.42_f32;
        let v0 = Vertex { position: [-0.5, -0.5, 0.0], color: [0.1, 0.2, 0.3, a] };
        let v1 = Vertex { position: [0.5, -0.5, 0.0], color: [0.4, 0.5, 0.6, a] };
        let v2 = Vertex { position: [0.0, 0.5, 0.0], color: [0.7, 0.8, 0.9, a] };
        let children = split_triangle(v0, v1, v2, &goal);
        for (i, v) in children.iter().enumerate() {
            assert_eq!(v.color[3], a, "child vertex {i} alpha must equal parent alpha");
        }
    }

    #[test]
    fn split_triangle_recolours_from_goal() {
        // Non-uniform goal: child colours must differ (detail captured).
        let grad = make_gradient_goal(64);
        let v0 = Vertex { position: [-0.6, -0.5, 0.0], color: [0.0, 0.0, 0.0, 0.5] };
        let v1 = Vertex { position: [0.6, -0.5, 0.0], color: [0.0, 0.0, 0.0, 0.5] };
        let v2 = Vertex { position: [0.0, 0.6, 0.0], color: [0.0, 0.0, 0.0, 0.5] };
        let kids = split_triangle(v0, v1, v2, &grad);
        let reds: Vec<f32> = (0..4).map(|t| kids[t * 3].color[0]).collect();
        assert!(reds.iter().any(|&r| (r - reds[0]).abs() > 1e-3), "non-uniform goal: child colours must differ, got {reds:?}");

        // Uniform goal: all children share one colour (the neutral case).
        let solid = make_solid_goal(64, [10, 20, 30]);
        let kids2 = split_triangle(v0, v1, v2, &solid);
        for t in 0..4 {
            let c = kids2[t * 3].color;
            assert!((c[0] - kids2[0].color[0]).abs() < 1e-6, "uniform goal: child {t} colour must match");
        }
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --bin polygenvo split_triangle`
Expected: FAIL — compile error `cannot find function 'split_triangle' in this scope`.

- [ ] **Step 3: Implement `split_triangle`**

Add this function immediately after `error_seeded_triangle` (after its closing `}` at line 822):

```rust
/// Subdivide a CCW triangle into 4 midpoint children that exactly tile it.
/// Children keep the parent's winding and alpha; each child's RGB is sampled
/// from the goal at the child's own centroid, so a split adds colour resolution
/// where the goal varies under the triangle (and is ~neutral where it doesn't).
/// Returns 12 vertices = 4 triangles. Temporary `allow(dead_code)`: wired into
/// `mutate` in the next task.
#[allow(dead_code)]
fn split_triangle(v0: Vertex, v1: Vertex, v2: Vertex, goal: &GoalImage) -> [Vertex; 12] {
    let alpha = v0.color[3];
    let mid = |a: &Vertex, b: &Vertex| -> [f32; 3] {
        [
            (a.position[0] + b.position[0]) * 0.5,
            (a.position[1] + b.position[1]) * 0.5,
            0.0,
        ]
    };
    let m01 = mid(&v0, &v1);
    let m12 = mid(&v1, &v2);
    let m20 = mid(&v2, &v0);
    // Build a child from three positions, recoloured from the goal at its centroid.
    let child = |p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]| -> [Vertex; 3] {
        let cx = (p0[0] + p1[0] + p2[0]) / 3.0;
        let cy = (p0[1] + p1[1] + p2[1]) / 3.0;
        let color = sample_goal_color(goal, cx, cy, alpha);
        [
            Vertex { position: p0, color },
            Vertex { position: p1, color },
            Vertex { position: p2, color },
        ]
    };
    // Three corner children + one centre child, all CCW (verified against a
    // CCW parent v0,v1,v2).
    let c0 = child(v0.position, m01, m20);
    let c1 = child(v1.position, m12, m01);
    let c2 = child(v2.position, m20, m12);
    let c3 = child(m01, m12, m20);
    let mut out = [Vertex { position: [0.0; 3], color: [0.0; 4] }; 12];
    out[0..3].copy_from_slice(&c0);
    out[3..6].copy_from_slice(&c1);
    out[6..9].copy_from_slice(&c2);
    out[9..12].copy_from_slice(&c3);
    out
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --bin polygenvo split_triangle`
Expected: PASS — `split_triangle_tiles_and_preserves_winding`, `split_triangle_inherits_alpha`, `split_triangle_recolours_from_goal` all ok.

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/main.rs
git commit -m "feat: split_triangle midpoint-4 subdivision helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `grow_by_split` + wire `split` into `mutate`, radius scaling

Add the mutation that picks a high-error triangle and replaces it with its 4 children, wire it into the op distribution, drop the `allow(dead_code)`, and scale `add`'s seed radius with the triangle count.

**Files:**
- Modify: `src/polygenvo/main.rs` — add `grow_by_split` after `split_triangle`; rebalance `mutate` op ranges; add `INITIAL_TRIANGLES` const
- Test: `src/polygenvo/main.rs` — `mod tests`

- [ ] **Step 1: Write the failing test**

Add this test inside `mod tests`, immediately after the `split_triangle_recolours_from_goal` test:

```rust
    #[test]
    fn grow_by_split_respects_cap() {
        let goal = make_solid_goal(64, [100, 150, 200]);
        let mut rng = StdRng::seed_from_u64(99);
        let grid = vec![1u32; GRID_CELLS]; // flat error -> any triangle eligible
        let mut genome = init_genome(&goal, 5, &mut rng); // 5 triangles = 15 verts

        // Below cap: one split replaces 1 triangle with 4 -> net +3 triangles.
        grow_by_split(&mut genome, &goal, &grid, 100, &mut rng);
        assert_eq!(genome.len() / 3, 8, "split should grow 5 -> 8 triangles");

        // At/over cap: n + 3 > cap -> no-op.
        let before = genome.clone();
        grow_by_split(&mut genome, &goal, &grid, 9, &mut rng); // 8 + 3 = 11 > 9
        assert_eq!(genome, before, "split must be a no-op when it would exceed the cap");
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --bin polygenvo grow_by_split_respects_cap`
Expected: FAIL — compile error `cannot find function 'grow_by_split' in this scope`.

- [ ] **Step 3a: Add `INITIAL_TRIANGLES` constant**

Add this constant right after the `PHASE_GROWTH` constant (line 21):

```rust
// Cold-start triangle count (also the reference count for add's seed-radius
// scaling: a fresh triangle shrinks toward the current triangle scale as the
// genome grows).
const INITIAL_TRIANGLES: usize = 40;
```

- [ ] **Step 3b: Implement `grow_by_split`**

Add this function immediately after `split_triangle` (and delete the `#[allow(dead_code)]` line above `split_triangle`, since `grow_by_split` now uses it):

```rust
/// Replace one triangle (chosen near a high-error cell) with its 4 midpoint
/// children, growing the genome by 3 triangles. No-op if the genome is empty or
/// a split would exceed `max_triangles` (a split adds 3). This is the only
/// growth path: it is applied as a mutation candidate, so `(1+λ)` selection
/// keeps it only when the added detail improves fitness.
fn grow_by_split(
    genome: &mut Vec<Vertex>,
    goal: &GoalImage,
    error_grid: &[u32],
    max_triangles: usize,
    rng: &mut impl Rng,
) {
    let n = genome.len() / 3;
    if n == 0 || n + 3 > max_triangles {
        return;
    }
    // Bias toward error: pick the triangle whose centroid is nearest a
    // roulette-selected high-error cell centre.
    let cell = sample_error_cell(error_grid, rng);
    let (tx, ty) = cell_to_clip(cell, 0.5, 0.5);
    let mut best_t = 0usize;
    let mut best_d = f32::INFINITY;
    for t in 0..n {
        let b = t * 3;
        let cx = (genome[b].position[0] + genome[b + 1].position[0] + genome[b + 2].position[0]) / 3.0;
        let cy = (genome[b].position[1] + genome[b + 1].position[1] + genome[b + 2].position[1]) / 3.0;
        let d = (cx - tx) * (cx - tx) + (cy - ty) * (cy - ty);
        if d < best_d {
            best_d = d;
            best_t = t;
        }
    }
    let b = best_t * 3;
    // Read the parent (Copy) before splicing, then replace it in place so the 4
    // children inherit its z/draw position (alpha-blend order preserved).
    let children = split_triangle(genome[b], genome[b + 1], genome[b + 2], goal);
    genome.splice(b..b + 3, children);
}
```

- [ ] **Step 3c: Rebalance the `mutate` op distribution and add the `split` branch**

In `mutate`, the match arms currently use ranges `0..=39`, `40..=64`, `65..=77`, `78..=85`, `86..=91`, `92..=95`, `96..=99`. Re-range them and insert a `split` arm so the weights become nudge 38 / recolour 24 / alpha 12 / split 10 / z-swap 5 / add 5 / relocate 3 / delete 3.

Change the nudge arm header `0..=39 =>` to:

```rust
        0..=37 => {
```

Change the recolour arm header `40..=64 =>` to:

```rust
        38..=61 => {
```

Change the alpha arm header `65..=77 =>` to:

```rust
        62..=73 => {
```

Insert a NEW split arm immediately after the alpha arm's closing `}` and before the z-swap arm (`78..=85 =>`):

```rust
        74..=83 => {
            // Split a high-error triangle into 4 midpoint children — the only
            // growth path, gated by selection (see `grow_by_split`).
            grow_by_split(&mut child, goal, error_grid, max_triangles, rng);
            OpKind::Structural
        }
```

Change the z-swap arm header `78..=85 =>` to:

```rust
        84..=88 => {
```

Change the add arm header `86..=91 =>` to:

```rust
        89..=93 => {
```

Inside that add arm, replace the seed line `let tri = error_seeded_triangle(goal, error_grid, rng, 0.2);` with radius scaling:

```rust
                let seed_radius =
                    (0.2 * (INITIAL_TRIANGLES as f32 / n as f32).sqrt()).clamp(0.02, 0.2);
                let tri = error_seeded_triangle(goal, error_grid, rng, seed_radius);
```

Change the relocate arm header `92..=95 =>` to:

```rust
        94..=96 => {
```

The final delete arm stays `_ =>` but update its comment `// Delete one triangle (op in 96..=99).` to:

```rust
            // Delete one triangle (op in 97..=99).
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --bin polygenvo`
Expected: PASS — `grow_by_split_respects_cap` plus all existing tests (including the three `split_triangle_*` and the smoke test) pass. Also confirm no `dead_code` warning for `split_triangle` (it is now used by `grow_by_split`).

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/main.rs
git commit -m "feat: split mutation operator + count-scaled add radius

grow_by_split replaces a high-error triangle with its 4 midpoint children;
wired into mutate as the sole, selection-gated growth path. add's seed
radius now shrinks as the genome grows.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Phase restructure — caps replace targets, remove batch growth

Turn `Phase.triangles` into `Phase.cap`, replace the geometric `production_phases` generator and `WARMUP_PHASES` with a 4-phase coarse-to-fine `PHASES` schedule whose final cap is `MAX_TRIANGLES`, delete `grow_genome` and `PHASE_GROWTH` and the compile-time tail guard, and make promotion stop growing the genome. This is one atomic change — the code does not compile until all of it lands.

**Files:**
- Modify: `src/polygenvo/main.rs` — constants block, `Phase`, `PHASES`, `production_phases` (delete), `grow_genome` (delete), `EsConfig::production`, `run_es` (init, per-step caps, promotion), smoke test, remove `production_phases_schedule` test, add `phase_caps` test

- [ ] **Step 1: Replace the `production_phases_schedule` test with a phase-schedule test**

Delete the entire `production_phases_schedule` test (currently lines 1375-1406) and replace it with:

```rust
    #[test]
    fn phase_caps_are_monotonic_and_reach_max() {
        let caps: Vec<usize> = PHASES.iter().map(|p| p.cap).collect();
        assert!(
            caps.windows(2).all(|w| w[1] >= w[0]),
            "phase caps must be non-decreasing: {caps:?}"
        );
        assert_eq!(
            *caps.last().unwrap(),
            MAX_TRIANGLES,
            "final phase cap must be the global triangle ceiling"
        );
    }
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `cargo test --bin polygenvo phase_caps`
Expected: FAIL — compile errors: `PHASES` not found and `Phase` has no field `cap` (both introduced below).

- [ ] **Step 3a: Update the constants block**

Replace the `MAX_TRIANGLES` doc comment and delete `PHASE_GROWTH`. The block at lines 12-21 currently reads:

```rust
// Triangle-count ceiling — the one knob that governs capacity. Raising it
// extends the auto-generated phase tail (see `production_phases`) and the
// vertex-buffer capacity below; lowering it shortens the tail.
const MAX_TRIANGLES: usize = 10000;

// Vertex buffer capacity (in vertices). 3 vertices per triangle.
const MAX_VERTICES: usize = MAX_TRIANGLES * 3;

// Geometric growth multiplier for the auto-generated high-count phases.
const PHASE_GROWTH: f32 = 1.6;

// Cold-start triangle count (also the reference count for add's seed-radius
// scaling: a fresh triangle shrinks toward the current triangle scale as the
// genome grows).
const INITIAL_TRIANGLES: usize = 40;
```

Replace it with (drops `PHASE_GROWTH`, keeps `INITIAL_TRIANGLES` from Task 2):

```rust
// Triangle-count ceiling — the one knob that governs capacity. It is the final
// phase's cap (see PHASES) and the vertex-buffer capacity below. The genome
// grows toward it organically via the fitness-gated `split` operator.
const MAX_TRIANGLES: usize = 10000;

// Vertex buffer capacity (in vertices). 3 vertices per triangle.
const MAX_VERTICES: usize = MAX_TRIANGLES * 3;

// Cold-start triangle count (also the reference count for add's seed-radius
// scaling: a fresh triangle shrinks toward the current triangle scale as the
// genome grows).
const INITIAL_TRIANGLES: usize = 40;
```

- [ ] **Step 3b: Rename the `Phase.triangles` field to `cap`**

The `Phase` struct (lines 600-607) currently reads:

```rust
#[derive(Clone)]
pub struct Phase {
    triangles: usize,
    pyramid_level: usize,
    // Initial step sizes for this phase, self-adapted by per-type 1/5 rules.
    initial_sigma_pos: f32,
    initial_sigma_col: f32,
}
```

Replace `triangles: usize,` with:

```rust
    // Maximum triangles the genome may grow to in this phase (a ceiling, not a
    // fill target — growth is organic via `split`).
    cap: usize,
```

- [ ] **Step 3c: Replace `WARMUP_PHASES` + guard + `production_phases` with the new `PHASES`**

Delete lines 609-662 (the `WARMUP_PHASES` const, the `const _: () = assert!(...)` guard, and the entire `production_phases` function) and replace them with:

```rust
// Coarse-to-fine schedule: pyramid level + initial σ per phase, with a capacity
// cap that rises to MAX_TRIANGLES at the finest level. Promotion advances this
// schedule on plateau; the genome grows toward each cap via `split`.
const PHASES: &[Phase] = &[
    Phase { cap: 300,           pyramid_level: 0, initial_sigma_pos: 0.30, initial_sigma_col: 0.20 }, // 128² coarse
    Phase { cap: 800,           pyramid_level: 1, initial_sigma_pos: 0.18, initial_sigma_col: 0.12 }, // 256² medium
    Phase { cap: 2000,          pyramid_level: 2, initial_sigma_pos: 0.10, initial_sigma_col: 0.08 }, // 512² fine
    Phase { cap: MAX_TRIANGLES, pyramid_level: 2, initial_sigma_pos: 0.05, initial_sigma_col: 0.04 }, // 512² finest
];
```

- [ ] **Step 3d: Point `EsConfig::production` at `PHASES`**

In `EsConfig::production` (line 673), replace:

```rust
            phases: production_phases(),
```

with:

```rust
            phases: PHASES.to_vec(),
```

- [ ] **Step 3e: Delete `grow_genome`**

Delete the entire `grow_genome` function (lines 751-758, the doc comment through the closing `}`).

- [ ] **Step 3f: Update `run_es` init to use `INITIAL_TRIANGLES` and `.cap`**

In `run_es`, replace the init line (line 1014):

```rust
    let mut current = init_genome(&goal, cfg.phases[phase_idx].triangles, &mut rng);
```

with (cold-start at INITIAL_TRIANGLES, never above the first phase's cap):

```rust
    let mut current = init_genome(&goal, INITIAL_TRIANGLES.min(cfg.phases[phase_idx].cap), &mut rng);
```

In the starting-phase `println!` (line 1030), replace `cfg.phases[phase_idx].triangles,` with:

```rust
        cfg.phases[phase_idx].cap,
```

- [ ] **Step 3g: Update the per-step cap/floor**

The per-step block (lines 1063-1067) currently reads:

```rust
        // Hold the genome near this phase's target. Allow ~25% shrinkage so
        // add/delete can shuffle the composition, but don't let add grow past
        // the phase's target — that's what phase promotion is for.
        let max_triangles = phase.triangles;
        let min_triangles = (phase.triangles * 3 / 4).max(8);
```

Replace it with:

```rust
        // `split`/`add` may grow the genome up to this phase's cap; `delete`
        // may prune down to a small absolute floor. Growth is organic and
        // selection-gated, so there is no fill target to hold near.
        let max_triangles = phase.cap;
        let min_triangles = 8;
```

- [ ] **Step 3h: Remove the `grow_genome` call from promotion**

The promotion block (lines 1173-1177) currently reads:

```rust
                phase_idx += 1;
                let new_phase = &cfg.phases[phase_idx];
                grow_genome(&mut current, new_phase.triangles, &goal, &mut rng);
                sigma_pos = new_phase.initial_sigma_pos;
                sigma_col = new_phase.initial_sigma_col;
```

Replace it with (no genome surgery on promotion — just unlock capacity + reset σ):

```rust
                phase_idx += 1;
                let new_phase = &cfg.phases[phase_idx];
                // No genome growth on promotion: it only raises the cap and
                // sharpens evaluation. The genome grows organically via `split`.
                sigma_pos = new_phase.initial_sigma_pos;
                sigma_col = new_phase.initial_sigma_col;
```

In the promotion `println!` (line 1196), replace `new_phase.triangles,` with:

```rust
                    new_phase.cap,
```

- [ ] **Step 3i: Update the smoke test's `Phase` literal**

In `ga_improves_on_synthetic_checker` (lines 1415-1420), replace `triangles: 6,` with:

```rust
            cap: 6,
```

- [ ] **Step 4: Run the full suite to verify everything passes**

Run: `cargo test --bin polygenvo`
Expected: PASS — `phase_caps_are_monotonic_and_reach_max`, the three `split_triangle_*`, `grow_by_split_respects_cap`, and the remaining existing tests (including `ga_improves_on_synthetic_checker`) all pass. No `production_phases` / `grow_genome` / `PHASE_GROWTH` references remain (grep to confirm: `grep -nE 'production_phases|grow_genome|PHASE_GROWTH|WARMUP_PHASES|\.triangles' src/polygenvo/main.rs` returns nothing).

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/main.rs
git commit -m "feat: organic growth — caps replace targets, drop batch growth

Phase.triangles -> Phase.cap; PHASES is a 4-phase coarse-to-fine schedule
capped at MAX_TRIANGLES. Promotion no longer dumps triangles (grow_genome
and production_phases deleted); the genome grows only via the split operator.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Manual validation (after all tasks)

The unit tests pin the split mechanism and the schedule; the behavioral fix is only observable from a run. This is the required eyeball check.

1. Release build: `cargo build --release --bin polygenvo`
2. From a directory with `goal.png` and `triangles/`: `cargo run --release --bin polygenvo` (or capture: `timeout 140 ./target/release/polygenvo > /tmp/polyrun_split.log 2>&1`)
3. In the log, confirm vs. the batch-growth baseline (`/tmp/polyrun.log`): (a) `→ Phase` re-scored fitness no longer drops 50–120k (promotion no longer perturbs the genome), (b) the `tris` count climbs smoothly and *holds* instead of decaying to a ¾ floor, (c) whether peak fitness clears the old ~961k, and (d) eyeball the latest `triangles/imageN.png` for cleaner, finer detail than the muddy batch-growth frames.

---

## Self-review notes

- **Spec coverage:** §1 organic growth (remove grow_genome dump) → Task 3 (3e, 3h). §2 split operator (high-error pick, midpoint-4 in place, alpha inherit, recolour, cap) → Task 1 (`split_triangle`) + Task 2 (`grow_by_split` + wiring). §3 phases (cap field, ~4 phases, final cap = MAX_TRIANGLES, promotion raises cap only, min floor 8, init small) → Task 3. §4 op mix (38/24/12/10/5/5/3/3) + seed-radius scaling → Task 2. §5 out-of-scope respected (no selection/shader/σ-math/pyramid/trigger changes). Testing § → Task 1 geometry tests, Task 2 cap test, Task 3 schedule test + smoke test retained.
- **Placeholder scan:** none — every code step shows complete code and exact ranges; commands have expected output.
- **Type/name consistency:** `split_triangle(Vertex,Vertex,Vertex,&GoalImage) -> [Vertex;12]` used identically in Task 1 tests, `grow_by_split`, and the geometry. `grow_by_split(&mut Vec<Vertex>, &GoalImage, &[u32], usize, &mut impl Rng)` matches its call in `mutate`. `Phase.cap` is used in `PHASES`, `EsConfig`, `run_es`, the smoke test, and the schedule test. `INITIAL_TRIANGLES` defined in Task 2, reused in Task 3's init. `error_seeded_triangle` signature (`..., max_radius: f32`) is unchanged — Task 2 only changes the argument passed.
- **Op-range arithmetic:** 0..=37 (38) + 38..=61 (24) + 62..=73 (12) + 74..=83 (10) + 84..=88 (5) + 89..=93 (5) + 94..=96 (3) + 97..=99 (3) = 100. ✓
- **Net-vertex check:** `splice(b..b+3, [12 verts])` removes 3, inserts 12 → +9 vertices = +3 triangles per split; cap guard `n + 3 > max` blocks overflow. ✓
