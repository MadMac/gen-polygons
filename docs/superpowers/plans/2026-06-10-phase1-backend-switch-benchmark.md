# Phase 1: Backend Switch (GL→Vulkan) + Benchmark + Decision Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `polygenvo`'s GPU device off the slow GL backend onto Vulkan (with a safe fallback + env override), then benchmark the existing pipeline and the merged brute-force polish on both backends so the data can scope the remaining work (tiled kernel, end-to-end loop).

**Architecture:** `gpu::init_wgpu` selects a preferred backend (default `PRIMARY` = Vulkan on Linux), falling back to GL if no adapter is found, overridable via `POLYGENVO_BACKEND`. The full test suite already routes through `init_wgpu` (via `init_test_wgpu`), so switching it re-validates everything on the new backend. A `#[ignore]`d benchmark test prints fitness-scoring and polish timings at 128²/256²/512² so the same binary can be run under each backend and compared.

**Tech Stack:** Rust 2024, `wgpu` 29, `bytemuck`; no new dependencies.

**Why this is its own plan:** The design (`docs/superpowers/specs/2026-06-10-end-to-end-gradient-optimizer-design.md`) is explicitly **measure-first** — the Vulkan numbers determine how much tiled-kernel work is actually needed and its parameters (tile size, whether shared-memory reduction is required). The tiled-kernel and end-to-end-loop plans are written **after** Task 4 produces those numbers.

---

## File structure

- **Modify `src/polygenvo/gpu.rs`**: backend selection helper + fallback + `POLYGENVO_BACKEND` override; log the selected backend. One clear responsibility (device bring-up) preserved.
- **Modify `src/polygenvo/window.rs`** (line ~333): mirror the same backend selection for the surface-compatible device, so headless and windowed agree.
- **Modify `src/polygenvo/gradient.rs`** (`#[cfg(test)]` tests): add the `#[ignore]`d `bench_backend` timing test (it already has `FitnessCalc` + `PolishState` access).
- **No new deps; no production behavior change** beyond which GPU backend is chosen.

### Shared facts (read before coding)

- Current `gpu::init_wgpu` ([gpu.rs](../../../src/polygenvo/gpu.rs)) builds `wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::GL, flags, backend_options, memory_budget_thresholds, display })` then `request_adapter` (HighPerformance, no surface) then `request_device`. **Preserve the exact `InstanceDescriptor` field set** — only `backends` changes.
- `test_support::init_test_wgpu` = `block_on(gpu::init_wgpu())`, so every test runs on the selected backend. The GPU smoke test `es::tests::ga_improves_on_synthetic_checker` is the real cross-pipeline validation.
- `FitnessCalc::new_for_test(device, queue, &goal, sample_count)`, `fitness_of_batch(&[&[Vertex]]) -> Vec<Eval>`, `texture_size()`; `genome::init_genome(&goal, n_triangles, &mut rng)`; `gradient::PolishState::new(&calc, &goal)` + `polish(&mut genome, parent_fitness, &calc, &cfg)`; `gradient::PolishCfg`.

---

## Task 1: Backend selection with fallback + env override

**Files:**
- Modify: `src/polygenvo/gpu.rs`

- [ ] **Step 1: Write the failing test**

Add to `src/polygenvo/gpu.rs` (new `#[cfg(test)] mod tests`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn init_wgpu_returns_a_working_device() {
        let (device, _queue) = block_on(init_wgpu());
        // A trivial allocation proves the device is live on whatever backend was picked.
        let _b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
    }

    #[test]
    fn preferred_backends_honors_env_override() {
        // Default (unset) is PRIMARY; explicit "gl" selects GL.
        assert_eq!(backends_from_env(Some("gl")), wgpu::Backends::GL);
        assert_eq!(backends_from_env(Some("vulkan")), wgpu::Backends::VULKAN);
        assert_eq!(backends_from_env(None), wgpu::Backends::PRIMARY);
        assert_eq!(backends_from_env(Some("garbage")), wgpu::Backends::PRIMARY);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --bin polygenvo gpu::tests 2>&1 | tail -15`
Expected: compile error — `backends_from_env` not found.

- [ ] **Step 3: Implement backend selection + fallback**

Rewrite `gpu.rs` so `init_wgpu` tries the preferred backend then falls back to GL. Keep the existing `InstanceDescriptor` field set verbatim (only `backends` varies) and the existing `RequestAdapterOptions`/`DeviceDescriptor`.

```rust
//! wgpu device/queue bring-up shared by production and tests.

use std::sync::Arc;

/// Map an optional `POLYGENVO_BACKEND` value to a backend set. Unset/unknown =
/// PRIMARY (Vulkan on Linux, Metal on macOS, DX12 on Windows).
fn backends_from_env(v: Option<&str>) -> wgpu::Backends {
    match v {
        Some("gl") => wgpu::Backends::GL,
        Some("vulkan") => wgpu::Backends::VULKAN,
        Some("metal") => wgpu::Backends::METAL,
        Some("dx12") => wgpu::Backends::DX12,
        _ => wgpu::Backends::PRIMARY,
    }
}

fn preferred_backends() -> wgpu::Backends {
    backends_from_env(std::env::var("POLYGENVO_BACKEND").ok().as_deref())
}

async fn try_backend(backends: wgpu::Backends) -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: Default::default(),
        display: Default::default(),
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok()?;
    let info = adapter.get_info();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
        .ok()?;
    println!("wgpu backend: {:?} — {}", info.backend, info.name);
    Some((Arc::new(device), Arc::new(queue)))
}

pub(crate) async fn init_wgpu() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let preferred = preferred_backends();
    if let Some(dq) = try_backend(preferred).await {
        return dq;
    }
    eprintln!("preferred wgpu backend {preferred:?} unavailable — falling back to GL");
    try_backend(wgpu::Backends::GL)
        .await
        .expect("no suitable wgpu adapter (GL fallback also failed)")
}
```

> If `wgpu::InstanceDescriptor`/`DeviceDescriptor` field names differ from the snippet, mirror the EXACT fields from the pre-change `gpu.rs` (which compiles) — only change `backends`.

- [ ] **Step 4: Run the gpu tests**

Run: `cargo test --bin polygenvo gpu::tests -- --nocapture 2>&1 | tail -15`
Expected: both pass; the `--nocapture` output shows a `wgpu backend: Vulkan — …` line (or GL on a host without Vulkan).

- [ ] **Step 5: Run the FULL suite on the new backend (the real validation)**

Run: `cargo test --bin polygenvo 2>&1 | tail -8`
Expected: all 35 existing tests + the 2 new ones pass on Vulkan. If the GPU smoke test or any gradient/fitness test fails on Vulkan, that is a blocking finding — report it (do not silence).

- [ ] **Step 6: Clippy + commit**

Run: `cargo clippy --bin polygenvo 2>&1 | tail -5`
```bash
git add src/polygenvo/gpu.rs
git commit -m "feat: select Vulkan/PRIMARY backend with GL fallback + POLYGENVO_BACKEND override"
```

---

## Task 2: Mirror backend selection in the window path

**Files:**
- Modify: `src/polygenvo/window.rs` (the `wgpu::Instance::new` at ~line 332-333)

- [ ] **Step 1: Make a small helper visible to window.rs**

In `gpu.rs`, expose the selector for reuse: change `fn preferred_backends()` to `pub(crate) fn preferred_backends()`.

- [ ] **Step 2: Use it in window.rs**

In `src/polygenvo/window.rs`, replace `backends: wgpu::Backends::GL,` (line ~333) with `backends: crate::gpu::preferred_backends(),`. Leave the rest of `init_window` (surface creation, surface-compatible adapter request) unchanged — Vulkan supports surfaces on Linux.

- [ ] **Step 3: Build + test (headless validation)**

Run: `cargo build --bin polygenvo 2>&1 | tail -5 && cargo test --bin polygenvo 2>&1 | tail -5`
Expected: compiles; suite green. (The windowed path itself needs a display and is validated manually in Step 4.)

- [ ] **Step 4: Manual windowed smoke (needs a display; skip in headless CI)**

Run (only if a display is available): `cargo run --release --bin polygenvo -- --show-window --goal /tmp/goal64.png`
Expected: a window opens and renders the evolving best; the startup log shows the Vulkan backend. Close the window → graceful stop. If the window path fails on Vulkan on this host, note it and set `POLYGENVO_BACKEND=gl` is the documented workaround.

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/gpu.rs src/polygenvo/window.rs
git commit -m "feat: window path uses the same backend selection as headless"
```

---

## Task 3: Benchmark harness (`#[ignore]`d timing test)

**Files:**
- Modify: `src/polygenvo/gradient.rs` (`#[cfg(test)] mod tests`)

- [ ] **Step 1: Add the benchmark test**

Add to `gradient.rs`'s `#[cfg(test)] mod tests`. It is `#[ignore]`d (timings, not an assertion) and prints ms per backend so the same binary can be compared across `POLYGENVO_BACKEND` values. It bounds polish cost (small triangle count, small `steps_n`, sizes 128/256 only) so it can never hang the suite; fitness scoring is timed at 128/256/512.

```rust
    #[test]
    #[ignore = "benchmark; run with --release -- --ignored --nocapture"]
    fn bench_backend() {
        use crate::fitness::FitnessCalc;
        use crate::genome::init_genome;
        use crate::test_support::make_solid_goal;
        use rand::{rngs::StdRng, SeedableRng};
        use std::time::Instant;

        let (device, queue) = crate::test_support::init_test_wgpu();
        println!("--- bench_backend (set POLYGENVO_BACKEND=gl|vulkan to compare) ---");

        // Fitness scoring throughput at each resolution (the core ES per-step cost).
        for &size in &[128u32, 256, 512] {
            let goal = make_solid_goal(size, [40, 120, 200]);
            let calc = FitnessCalc::new_for_test(device.clone(), queue.clone(), &goal, 1);
            let mut rng = StdRng::seed_from_u64(1);
            let g = init_genome(&goal, 200, &mut rng);
            // warmup
            let _ = calc.fitness_of_batch(&[g.as_slice()]);
            let iters = 50;
            let t = Instant::now();
            for _ in 0..iters {
                let _ = calc.fitness_of_batch(&[g.as_slice()]);
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            println!("fitness {size}² (200 tris): {ms:.3} ms/score");
        }

        // Brute-force polish cost at bounded sizes/steps (128/256 only; small genome).
        for &size in &[128u32, 256] {
            let goal = make_solid_goal(size, [40, 120, 200]);
            let calc = FitnessCalc::new_for_test(device.clone(), queue.clone(), &goal, 1);
            let mut state = super::PolishState::new(&calc, &goal);
            let mut rng = StdRng::seed_from_u64(2);
            let mut g = init_genome(&goal, 100, &mut rng);
            let parent = calc.fitness_of(&g);
            let cfg = super::PolishCfg { enabled: true, every_k: 1, steps_n: 10, lr: 0.05, tau_start: 0.3, tau_end: 0.05 };
            let t = Instant::now();
            let _ = state.polish(&mut g, parent, &calc, &cfg);
            println!("polish {size}² (100 tris, 10 steps): {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        }
    }
```

- [ ] **Step 2: Verify it compiles and runs (ignored by default)**

Run: `cargo test --bin polygenvo 2>&1 | tail -5`
Expected: suite green; `bench_backend` shows as ignored (not run by default).

- [ ] **Step 3: Commit**

```bash
git add src/polygenvo/gradient.rs
git commit -m "test: add ignored bench_backend timing harness (fitness + polish, per backend)"
```

---

## Task 4: Run the benchmark, record results, decide kernel scope (analysis)

**Files:** none (produces a results note)

- [ ] **Step 1: Benchmark Vulkan (default) — release build**

Run: `POLYGENVO_BACKEND=vulkan cargo test --release --bin polygenvo bench_backend -- --ignored --nocapture 2>&1 | tee /tmp/bench_vulkan.txt | grep -E "backend:|fitness|polish"`

- [ ] **Step 2: Benchmark GL for comparison**

Run: `POLYGENVO_BACKEND=gl cargo test --release --bin polygenvo bench_backend -- --ignored --nocapture 2>&1 | tee /tmp/bench_gl.txt | grep -E "backend:|fitness|polish"`

- [ ] **Step 3: Record + decide**

Append a short results table (fitness ms/score and polish ms at each size, GL vs Vulkan) to the design spec's a new "Phase 1 results" section in `docs/superpowers/specs/2026-06-10-end-to-end-gradient-optimizer-design.md`. Then make the scope call and write it down:
- If Vulkan makes the **brute-force polish** tolerable at 256²/512² (e.g. polish ms is in the low hundreds, not seconds), the tiled kernel may be **deferrable** — the end-to-end loop could ship on the brute-force kernel first.
- If polish is still too slow at 512² on Vulkan, the **tiled kernel is required** — and the fitness/polish ms give the budget the tiled kernel must beat.

Commit the results:
```bash
git add docs/superpowers/specs/2026-06-10-end-to-end-gradient-optimizer-design.md
git commit -m "docs: Phase 1 backend benchmark results + kernel-scope decision"
```

- [ ] **Step 4: Update project memory**

Update `path-b-diff-rasterizer-status` memory with the backend decision and the measured GL-vs-Vulkan speedup, so the next session starts from the data.

---

## After Phase 1 (separate plans, written from the data)

- **Phase 2 — Tiled differentiable kernel** (only if Task 4 says it's needed): tile-binned forward + shared-memory-reduction backward in WGSL, GPU==CPU-oracle tested, beating the brute-force budget measured here.
- **Phase 3 — End-to-end run loop**: gradient as the primary per-step refiner of all triangles, ES narrowed to structural ops, best-ever snapshot net, behind a flag; A/B vs `master` at 512².

Each gets its own brainstorm-confirmed scope (parameters now known) → plan → build.

---

## Self-review

- **Spec coverage (Phase 1 portion):** backend switch + measure-first (spec §Architecture.1 / Milestone 1) → Tasks 1, 2, 4; window-path consistency (spec risks) → Task 2; benchmark harness → Task 3; decision gate that scopes the kernel → Task 4. Tiled kernel (Milestone 2) and end-to-end loop (Milestone 3) are deferred to their own plans **by design** (measure-first) — not gaps.
- **Placeholder scan:** none — every code/command step is concrete. Task 4 is an analysis task (no code) by nature; its outputs are explicit commands + a results commit.
- **Type/name consistency:** `backends_from_env`/`preferred_backends`/`try_backend`/`init_wgpu` in gpu.rs; `PolishState::new`/`polish`, `PolishCfg`, `FitnessCalc::{new_for_test,fitness_of_batch,fitness_of}`, `init_genome` used consistently and matching the merged code.
