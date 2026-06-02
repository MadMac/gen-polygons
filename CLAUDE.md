# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Experimental playground for `wgpu`, shaders, and genetic algorithms. The goal is to approximate a raster image (`goal.png` at the repo root) by evolving a population of colored triangles. There is a small test suite (per-module unit tests plus one GPU smoke test — a 30-step ES on a 32×32 synthetic checker) but no CI — substantive changes are still evaluated by running the simulation and eyeballing output frames in `triangles/`.

## Commands

- `cargo build --release --bin polygenvo` — release build; effectively required because the ES does many GPU renders per step.
- `cargo run --release --bin polygenvo` — runs the simulation; needs `goal.png` (or `--goal <path>`) and `triangles/` in CWD.
- `cargo run --release --bin polygenvo -- --goal <path>` — approximate a different image instead of the default `goal.png`. Accepts `--goal path` or `--goal=path` (parsed by `arg_value` in `main.rs`); composable with the other flags.
- `cargo run --release --bin polygenvo -- --infinite` — same, but drops the `MAX_STEPS` ceiling and runs until Ctrl-C. A `ctrlc` handler in `main.rs` flips `EsConfig.stop_flag` (an `Arc<AtomicBool>`); `run_es` checks it each step and exits cleanly via the normal final-snapshot/summary path instead of being hard-killed mid-step.
- `cargo run --release --bin polygenvo -- --show-window` — opens a live window (winit) that renders the current best candidate, refreshed on each accepted improvement (throttled to ~display rate). Composable with `--infinite`. Closing the window stops the run gracefully (same final-snapshot/summary path as Ctrl-C). Needs a display server; in windowed mode the wgpu device is created by `window::init_window` (so the adapter is surface-compatible) instead of `gpu::init_wgpu`.
- `cargo build --bin polygenvo` — debug build; useful when iterating on Rust code paths.
- `cargo test --bin polygenvo` — runs the test suite (per-module unit tests + the GPU smoke test `es::tests::ga_improves_on_synthetic_checker`). Fast (~0.1s) and requires a working wgpu adapter on the host.

Runtime requirements for `polygenvo`:
- `goal.png` (or the file named by `--goal <path>`) must exist in the working directory (square RGBA PNG; `texture_size` is taken from its width).
- Snapshots go to a fresh per-run subfolder `triangles/<local-timestamp>/` (e.g. `triangles/2026-06-02_12-56-43/`) that `run_es` creates on startup via `create_dir_all` (so `triangles/` need not pre-exist). `run_timestamp()` in `es.rs` names it in the user's local timezone via `chrono::Local::now()`. Frames are `imageN.png`/`final.png` inside it. `triangles/` is gitignored.

## Binaries

Only one binary lives in the repo: [`polygenvo`](src/polygenvo/main.rs). When the user says "the app" or "the simulation," they mean `polygenvo`.

## High-level architecture (polygenvo)

[main.rs](src/polygenvo/main.rs) is a thin entry point — `env_logger::init → goal::load_goal_image(--goal or "goal.png") → block_on(gpu::init_wgpu()) → es::run_es(EsConfig::production())` — that just declares the modules below. The code is split into single-responsibility modules under `src/polygenvo/`, layered low → high (each only depends on the ones above it):

- **[goal.rs](src/polygenvo/goal.rs)** — `GoalImage` (RGBA8 wrapper with a compact `Debug`), `load_goal_image`, `downsample_goal` (Lanczos, for pyramid levels), `sample_goal_color` (clip-space point → goal pixel; image y is flipped).
- **[genome.rs](src/polygenvo/genome.rs)** — `Vertex { position: [f32;3], color: [f32;4] }` matching the wgpu vertex layout exactly (`bytemuck::Pod`, so the genome is `cast_slice`'d straight to a vertex buffer). A genome is a `Vec<Vertex>` interpreted as a `TriangleList`, so length stays a multiple of 3. Holds the capacity constants (`MAX_TRIANGLES`/`MAX_VERTICES`/`INITIAL_TRIANGLES`), `triangle_centroid`, `seeded_triangle` (the shared CCW-triangle builder — uniform and error-guided seeding both call it, differing only in how the centre is chosen; CCW for `front_face: Ccw, cull_mode: Back`), `init_genome`, and `split_triangle` (4-way midpoint subdivision).
- **[gpu.rs](src/polygenvo/gpu.rs)** — `init_wgpu` (GL backend, high-performance adapter) → `Arc<Device>` / `Arc<Queue>`. Used by the headless path; `--show-window` instead brings up the device via `window::init_window` (surface-compatible adapter).
- **[window.rs](src/polygenvo/window.rs)** — the `--show-window` live viewer (winit 0.30, single-threaded). `init_window` creates the window + a surface-compatible wgpu device and returns a `WindowObserver` implementing `es::StepObserver`. `run_es` calls `on_step` each step; it pumps window events non-blockingly (`pump_app_events`) and re-renders the best to the swapchain on improvement (own pipeline reusing `shader.wgsl`/`Vertex::desc()`, prefers a non-blocking present mode + `MIN_PRESENT_INTERVAL` throttle so the search never stalls on a present). `CloseRequested` makes `on_step` return `false`, ending the loop via the normal final-snapshot path.
- **[fitness.rs](src/polygenvo/fitness.rs)** — the GPU evaluator. `FitnessCalc::fitness_of_batch(&[&[Vertex]]) -> Vec<Eval>` renders each candidate into an offscreen `Rgba8UnormSrgb` target (real OVER alpha blending; MSAA only on the finest level), then dispatches [fitness.wgsl](src/polygenvo/fitness.wgsl) to score it and emit a 16×16 residual-error grid. The compute shader converts rendered+goal pixels linear-RGB → CIE XYZ → CIELAB, takes per-pixel ΔE76, and `atomicAdd`s a workgroup-reduced sum (scaled by `FITNESS_SCALE`) into one `u32`. Rust maps the accumulator to a similarity score in `[0, 1_000_000]`, **higher = better**. `FitnessCalc` is `Clone`-by-`Arc`. GPU-layout constants (`LAMBDA`, `FITNESS_SCALE`, `ERROR_GRID_DIM`/`GRID_CELLS`, slot strides, `MSAA_SAMPLE_COUNT`) and `build_pyramid` live here.
- **[variation.rs](src/polygenvo/variation.rs)** — the mutation operators. `mutate` picks one operator from a named weighted table (`OPERATORS`, weights summing to 100; `pick_op` roulette) and applies it, returning `(child, OpKind)`. Operators: vertex nudge, recolour, alpha nudge, `split` (the only growth path — fitness-gated subdivision via `grow_by_split`), z-swap, error-seeded add, relocate, delete. `OpKind` (Positional/Chromatic/Structural) classes which step size an op exercises; `StepSizes { pos, col }` carries the two σ. Also `gaussian` (Box-Muller), `sample_error_cell`, `cell_to_clip`.
- **[es.rs](src/polygenvo/es.rs)** — the search driver. `pub(crate) fn run_es(device, queue, goal, cfg: EsConfig) -> EsResult` runs a **(1+λ)-ES** over the coarse-to-fine `PHASES` schedule: each step generates `cfg.lambda` candidates, scores them in one batch, accepts the best improver. Two extracted structs own the fiddly state — **`OneFifthRule`** (current σ pair + the 1/5-success-rule window, with a single `reset_window` so the per-type tallies can't drift) and **`PhaseSchedule`** (phase index, in-phase step count, plateau detection via `check_plateau`). On plateau, promotion raises the cap and re-scores at a finer pyramid level; the genome grows organically via `split` (no batch growth). ES tunables (`SIGMA_*`, `SIGMA_WINDOW`, `PHASE_MIN_STEPS`, `PLATEAU_WINDOW`, `PLATEAU_ACCEPTS`, `MAX_STEPS`, `MIN_TRIANGLES`) live here.

Both `run_es` and the smoke test drive the same code path. The coarse-to-fine schedule is `const PHASES: &[Phase]` (each phase: `cap`, `pyramid_level`, `initial_sigma_pos`, `initial_sigma_col`); `run_es` takes phases via `EsConfig.phases` — production uses `PHASES.to_vec()`, tests use a single-element `Vec`.

Tunable constants now live next to the module that owns them (genome capacity in `genome.rs`, GPU/grid layout in `fitness.rs`, ES schedule/σ in `es.rs`) rather than one block at the top of `main.rs`. Tests are a `#[cfg(test)] mod tests` in each module, with shared fixtures in [test_support.rs](src/polygenvo/test_support.rs).

## Shader pipeline

Both shaders are loaded via `include_str!` and fed to `wgpu` as `ShaderSource::Wgsl`:

- [shader.wgsl](src/polygenvo/shader.wgsl) — trivial passthrough vertex+fragment for rendering the triangle list.
- [fitness.wgsl](src/polygenvo/fitness.wgsl) — compute shader; CIELAB ΔE76 per pixel + `atomicAdd` into a single `u32`.

Edits to either `.wgsl` file take effect on the next `cargo build` because `include_str!` paths are tracked for rebuilds.

WGSL syntax is **current (post-1.0)**: `@location(0)`, `@vertex`/`@fragment`/`@compute`, `@group(N) @binding(M)`, comma-separated struct fields. There is no `build.rs`, no `shaderc`, no SPIR-V path.

## Dependency versions

`Cargo.toml` is on a current ecosystem snapshot: `wgpu = "29"`, `image = "0.25"`, `env_logger = "0.11"`, `rand = "0.10"`, `bytemuck = "1"`, `futures = "0.3"`, `log = "0.4"`, `ctrlc = "3"` (graceful Ctrl-C for `--infinite`), `chrono = "0.4"` (`default-features = false, features = ["clock"]`, for local-timezone snapshot-folder timestamps), `winit = "0.30"` (the `--show-window` live viewer). Rust edition is `"2024"`.

`rand 0.10` notes for editing: use `rand::rng()` (not `thread_rng()`) and `rng.random_range(a..b)` (not `gen_range`). The `gen` keyword is reserved in edition 2024; if a method name collides it must be escaped as `r#gen`.

## Conventions when editing

- Tunables live next to their owning module (see the architecture map). When changing one, check any `PHASES` entries that override per-phase values (`initial_sigma_pos`/`initial_sigma_col` are per-phase; `LAMBDA` and `MAX_STEPS` are global). `LAMBDA` is defined in `fitness.rs` because it sizes the GPU batch buffers; `cfg.lambda` must stay ≤ it (asserted in `fitness_of_batch`).
- The fitness direction is **higher = better**. Code that compares fitnesses uses `>` for "improvement".
- Tests are the regression guard — the GPU smoke test covers `run_es`/`FitnessCalc`/the `.wgsl` pipeline, and the per-module unit tests cover the pure logic (`OneFifthRule`/`PhaseSchedule`, the operator table, geometry helpers). Run `cargo test --bin polygenvo` (and `cargo clippy --bin polygenvo`, which is kept clean) before assuming a change is safe.
- PNG snapshots are gated on `cfg.snapshot_every`: production sets `Some(SNAPSHOT_EVERY_IMPROVEMENT)`, the smoke test sets `None`. When adding new snapshot sites, gate them too.
