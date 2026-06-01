# Progressive Triangle-Count Ramp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the (1+λ)-ES progressively grow the triangle count to a single, easily-changeable ceiling (~1000) by keeping the hand-tuned warmup phases and auto-generating geometric high-count phases above them.

**Architecture:** Replace the hard-coded `MAX_VERTICES` with a triangle-denominated `MAX_TRIANGLES` constant that everything derives from. Keep the existing 4 coarse-to-fine phases as a fixed `WARMUP_PHASES` const; add a `production_phases()` generator that appends geometric phases (×`PHASE_GROWTH`) from the last warmup count up to `MAX_TRIANGLES`, all at the finest pyramid level and σ. The existing plateau-driven promotion logic walks the longer schedule unchanged.

**Tech Stack:** Rust (edition 2024), `wgpu` 29, single binary `polygenvo`. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-06-01-progressive-triangle-ramp-design.md](../specs/2026-06-01-progressive-triangle-ramp-design.md)

---

## File Structure

All changes are in one file — the project keeps the whole binary in a single ~1370-line `main.rs`, and this change is small enough that following that pattern is correct.

- Modify: `src/polygenvo/main.rs`
  - Constants block near the top (capacity knobs).
  - `PHASES` const → renamed to `WARMUP_PHASES`; new `production_phases()` fn beside it.
  - `EsConfig::production()` — call the generator.
  - `mod tests` — one new GPU-free unit test pinning the generator's contract.

---

## Task 1: Capacity constants + rename PHASES → WARMUP_PHASES

Pure refactor. Introduces the single-knob cap and renames the phase const without changing runtime behavior (production still uses only the warmup phases at this point). The build and existing tests must stay green.

**Files:**
- Modify: `src/polygenvo/main.rs:12-13` (constants), `src/polygenvo/main.rs:601-606` (`PHASES`), `src/polygenvo/main.rs:618` (`EsConfig::production`)

- [ ] **Step 1: Replace the `MAX_VERTICES` constant block with triangle-denominated knobs**

Find this block at the top of the file:

```rust
// Vertex buffer capacity (in vertices). 450 vertices = 150 triangles.
const MAX_VERTICES: usize = 450;
```

Replace it with:

```rust
// Triangle-count ceiling — the one knob that governs capacity. Raising it
// extends the auto-generated phase tail (see `production_phases`) and the
// vertex-buffer capacity below; lowering it shortens the tail.
const MAX_TRIANGLES: usize = 1000;

// Vertex buffer capacity (in vertices). 3 vertices per triangle.
const MAX_VERTICES: usize = MAX_TRIANGLES * 3;

// Geometric growth multiplier for the auto-generated high-count phases.
const PHASE_GROWTH: f32 = 1.6;
```

- [ ] **Step 2: Rename the `PHASES` const to `WARMUP_PHASES`**

Find this block (around line 601):

```rust
const PHASES: &[Phase] = &[
    Phase { triangles: 40,  pyramid_level: 0, initial_sigma_pos: 0.30, initial_sigma_col: 0.20 }, // 128² coarse
    Phase { triangles: 80,  pyramid_level: 1, initial_sigma_pos: 0.18, initial_sigma_col: 0.12 }, // 256² medium
    Phase { triangles: 120, pyramid_level: 2, initial_sigma_pos: 0.10, initial_sigma_col: 0.08 }, // 512² fine
    Phase { triangles: 150, pyramid_level: 2, initial_sigma_pos: 0.05, initial_sigma_col: 0.04 }, // 512² finer
];
```

Replace it with:

```rust
// Hand-tuned coarse-to-fine warmup phases (the pyramid climb). The production
// schedule keeps these verbatim, then `production_phases` appends geometric
// high-count phases above the last warmup count up to MAX_TRIANGLES.
const WARMUP_PHASES: &[Phase] = &[
    Phase { triangles: 40,  pyramid_level: 0, initial_sigma_pos: 0.30, initial_sigma_col: 0.20 }, // 128² coarse
    Phase { triangles: 80,  pyramid_level: 1, initial_sigma_pos: 0.18, initial_sigma_col: 0.12 }, // 256² medium
    Phase { triangles: 120, pyramid_level: 2, initial_sigma_pos: 0.10, initial_sigma_col: 0.08 }, // 512² fine
    Phase { triangles: 150, pyramid_level: 2, initial_sigma_pos: 0.05, initial_sigma_col: 0.04 }, // 512² finer
];
```

- [ ] **Step 3: Update the only consumer of the renamed const**

In `EsConfig::production()`, find:

```rust
            phases: PHASES.to_vec(),
```

Replace with:

```rust
            phases: WARMUP_PHASES.to_vec(),
```

- [ ] **Step 4: Build and run the test suite — expect green**

Run: `cargo test --bin polygenvo`
Expected: compiles cleanly; `test result: ok.` with all existing tests passing (`gaussian_has_zero_mean_and_unit_std`, `batch_scores_match_single`, `error_grid_tracks_residual`, `sample_error_cell_favours_high_error`, `sample_error_cell_uniform_when_empty`, `cell_to_clip_stays_in_cell_bounds`, `ga_improves_on_synthetic_checker`). No `unused`/`unknown ident` warnings for `PHASES`.

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/main.rs
git commit -m "refactor: derive vertex cap from MAX_TRIANGLES, rename PHASES to WARMUP_PHASES

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Add `production_phases()` generator and wire it in

Adds the geometric high-count tail above the warmup phases, capped at `MAX_TRIANGLES`. TDD: the contract is a pure function over the constants, so it's tested without a GPU.

**Files:**
- Test: `src/polygenvo/main.rs` — `mod tests` (add `production_phases_schedule`)
- Create (fn): `src/polygenvo/main.rs` — `production_phases()` immediately after the `WARMUP_PHASES` const
- Modify: `src/polygenvo/main.rs` — `EsConfig::production()` to call the generator

- [ ] **Step 1: Write the failing test**

Add this test inside the `mod tests` block, immediately after the `cell_to_clip_stays_in_cell_bounds` test:

```rust
    #[test]
    fn production_phases_schedule() {
        let phases = production_phases();
        let counts: Vec<usize> = phases.iter().map(|p| p.triangles).collect();

        // Starts with the four hand-tuned warmup phases, verbatim.
        assert_eq!(&counts[..4], &[40, 80, 120, 150]);

        // With the default constants (MAX_TRIANGLES=1000, PHASE_GROWTH=1.6) the
        // auto tail is geometric ×1.6 with the penultimate value snapped to the cap.
        assert_eq!(counts, vec![40, 80, 120, 150, 240, 384, 615, 1000]);

        // Strictly increasing: no duplicates, no shrinkage.
        assert!(
            counts.windows(2).all(|w| w[1] > w[0]),
            "schedule not strictly increasing: {counts:?}"
        );

        // Ends exactly at the cap.
        assert_eq!(*counts.last().unwrap(), MAX_TRIANGLES);

        // Auto phases inherit the finest warmup phase's pyramid level and σ.
        let finest = WARMUP_PHASES.last().unwrap();
        for p in &phases[4..] {
            assert_eq!(p.pyramid_level, finest.pyramid_level);
            assert_eq!(p.initial_sigma_pos, finest.initial_sigma_pos);
            assert_eq!(p.initial_sigma_col, finest.initial_sigma_col);
        }
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --bin polygenvo production_phases_schedule`
Expected: FAIL — compile error `cannot find function 'production_phases' in this scope`.

- [ ] **Step 3: Implement `production_phases()` and wire it into production**

Add this function immediately after the `WARMUP_PHASES` const definition (before `pub struct EsConfig`):

```rust
/// Build the production phase schedule: the hand-tuned `WARMUP_PHASES`, then
/// geometric high-count phases growing by `PHASE_GROWTH` from the last warmup
/// count up to `MAX_TRIANGLES`. The auto phases sit at the finest warmup
/// pyramid level and reuse its σ (the 1/5-rule re-adapts σ within each phase).
/// The penultimate value is snapped to the cap when it lands within 15% of it,
/// so the schedule never ends with a near-duplicate phase.
fn production_phases() -> Vec<Phase> {
    let finest = WARMUP_PHASES
        .last()
        .expect("WARMUP_PHASES must be non-empty");
    debug_assert!(
        MAX_TRIANGLES >= finest.triangles,
        "MAX_TRIANGLES {} is below the warmup ceiling {}",
        MAX_TRIANGLES,
        finest.triangles
    );

    let mut phases = WARMUP_PHASES.to_vec();
    let mk = |n: usize| Phase {
        triangles: n,
        pyramid_level: finest.pyramid_level,
        initial_sigma_pos: finest.initial_sigma_pos,
        initial_sigma_col: finest.initial_sigma_col,
    };

    // Snap-to-cap threshold: stop generating geometric phases once the next one
    // would land within 15% of the cap, then append the exact cap instead.
    let snap = (MAX_TRIANGLES as f32 * 0.85) as usize;
    let mut n = finest.triangles;
    loop {
        n = (n as f32 * PHASE_GROWTH).ceil() as usize;
        if n >= snap {
            break;
        }
        phases.push(mk(n));
    }
    // Append the exact cap, unless the cap equals the warmup ceiling (no tail).
    if MAX_TRIANGLES > finest.triangles {
        phases.push(mk(MAX_TRIANGLES));
    }
    phases
}
```

Then update `EsConfig::production()` to use it. Find:

```rust
            phases: WARMUP_PHASES.to_vec(),
```

Replace with:

```rust
            phases: production_phases(),
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test --bin polygenvo production_phases_schedule`
Expected: PASS — `test tests::production_phases_schedule ... ok`.

Then run the full suite to confirm nothing regressed:

Run: `cargo test --bin polygenvo`
Expected: `test result: ok.` with all tests passing (8 total now, including the new one and the `ga_improves_on_synthetic_checker` smoke test).

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/main.rs
git commit -m "feat: auto-generate geometric high-count phases up to MAX_TRIANGLES

Schedule becomes 40 -> 80 -> 120 -> 150 -> 240 -> 384 -> 615 -> 1000
with the default constants; the ceiling is a single MAX_TRIANGLES knob.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Manual validation (after both tasks)

The unit test pins the schedule contract, but the actual ceiling lift is only observable by running the simulation. This is the project's required eyeball check — not part of an automated task.

1. Release build: `cargo build --release --bin polygenvo`
2. From a directory containing `goal.png` and a `triangles/` dir: `cargo run --release --bin polygenvo`
3. Watch the log: phase promotions should now climb past 150 (`→ Phase N | 240 triangles …`, then 384, 615, 1000) and the reported fitness should rise above the prior ~965k plateau as the count grows. Eyeball the latest `triangles/imageN.png` snapshots for finer detail than the 150-triangle frames.

---

## Self-review notes

- **Spec coverage:** §1 cap-from-one-constant → Task 1. §2 warmup-kept + auto-generated tail + 0.85 snap + finest level/σ + debug_assert guard → Task 2 (`production_phases`). §3 runtime (no logic change) → no task needed (promotion loop untouched). §4 out-of-scope respected (no operator/shader/σ-math changes). Testing §: generator unit test (Task 2 Step 1) + smoke test green (Task 2 Step 4) + manual run (above).
- **Placeholder scan:** none — every code step shows complete code; commands have expected output.
- **Type consistency:** `production_phases() -> Vec<Phase>` matches `EsConfig.phases: Vec<Phase>`; `Phase` fields (`triangles`, `pyramid_level`, `initial_sigma_pos`, `initial_sigma_col`) match the struct at `main.rs:592-599`; `MAX_TRIANGLES`/`PHASE_GROWTH`/`WARMUP_PHASES` defined in Task 1 are the exact identifiers used in Task 2.
- **Edge case checked:** if `MAX_TRIANGLES == 150` (warmup ceiling), the loop breaks immediately (240 ≥ snap 127) and the cap is not appended (`MAX_TRIANGLES > finest.triangles` is false) → schedule is just the warmup phases, no duplicate. If `MAX_TRIANGLES < 150`, the `debug_assert` fires in debug builds.
