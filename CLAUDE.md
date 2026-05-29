# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Experimental playground for `wgpu`, shaders, and genetic algorithms. The goal is to approximate a raster image (`goal.png` at the repo root) by evolving a population of colored triangles. There is one tiny smoke test (a 30-step ES on a 32×32 synthetic checker) but no CI — substantive changes are still evaluated by running the simulation and eyeballing output frames in `triangles/`.

## Commands

- `cargo build --release --bin polygenvo` — release build; effectively required because the ES does many GPU renders per step.
- `cargo run --release --bin polygenvo` — runs the simulation; needs `goal.png` and `triangles/` in CWD.
- `cargo build --bin polygenvo` — debug build; useful when iterating on Rust code paths.
- `cargo test --bin polygenvo` — runs the smoke test (`ga_improves_on_synthetic_checker`). Fast (~0.1s) and requires a working wgpu adapter on the host.

Runtime requirements for `polygenvo`:
- `goal.png` must exist in the working directory (square RGBA PNG; `texture_size` is taken from its width).
- The `triangles/` directory must exist — the binary writes `triangles/imageN.png` snapshots and will panic on `save()` if the directory is missing. It is gitignored.

## Binaries

Only one binary lives in the repo: [`polygenvo`](src/polygenvo/main.rs). When the user says "the app" or "the simulation," they mean `polygenvo`.

## High-level architecture (polygenvo)

[main.rs](src/polygenvo/main.rs) is ~970 lines and wires together three layers:

1. **Genome** — `Vertex { position: [f32;3], color: [f32;4] }` matching the wgpu vertex layout exactly (it's `bytemuck::Pod`, so the genome buffer is `cast_slice`'d straight to a vertex buffer). A genome is a `Vec<Vertex>` interpreted as a `TriangleList`, so length must stay a multiple of 3. Triangles are seeded by sampling colours from the goal at random clip-space points: see `random_color_seeded_triangle` (centres in `(-0.9, 0.9)`, radii scaled by `max_radius`, CCW winding for `front_face: Ccw, cull_mode: Back`).

2. **Fitness (`FitnessCalc`)** — `fitness_of(&[Vertex])` does a full wgpu render of the triangles into an offscreen `Rgba8UnormSrgb` texture, then dispatches [fitness.wgsl](src/polygenvo/fitness.wgsl) to score it against the goal. The compute shader converts both rendered and goal pixels through linear-RGB → CIE XYZ → CIELAB, takes the per-pixel ΔE76 distance, normalises by 250, scales by 1000, and `atomicAdd`s into a single `u32` accumulator. Rust then maps the accumulator to a similarity score in `[0, 1_000_000]` where **higher = better fit**. `FitnessCalc` is `Clone`-by-`Arc` (the inner struct holds the device/queue/pipelines/buffers behind `Arc`); cloning is cheap and shares all GPU resources.

3. **ES driver** — Hand-rolled **(1+λ)-ES** with a coarse-to-fine phase schedule. Each step generates `LAMBDA` mutated candidates from the current parent (`mutate(...)` applies position/colour/alpha jitter and occasional add/delete-triangle operations), evaluates them on a goal pyramid, and accepts the best if it beats the parent. Sigma self-adapts via the **1/5 success rule** evaluated over `SIGMA_WINDOW` steps. Phase promotion fires when the last `PLATEAU_WINDOW` steps produced fewer than `PLATEAU_ACCEPTS` improvements; promotion bumps the triangle count and switches to a finer pyramid level. No `genevo`, no crossover, no population.

The ES loop is **extracted into `pub fn run_es(device, queue, goal, cfg: EsConfig) -> EsResult`** so both production and the smoke test exercise the same code. `main()` is a thin wrapper: `env_logger::init → load_goal_image("goal.png") → block_on(init_wgpu()) → run_es(EsConfig::production())`.

Tunable constants at the top of [main.rs](src/polygenvo/main.rs):
- `MAX_VERTICES`, `LAMBDA`, `SIGMA_WINDOW`, `PHASE_MIN_STEPS`, `PLATEAU_WINDOW`, `PLATEAU_ACCEPTS`, `SNAPSHOT_EVERY_IMPROVEMENT`, `MAX_STEPS`.

The coarse-to-fine schedule is `const PHASES: &[Phase] = &[...]` with `triangles`, `pyramid_level`, and `initial_sigma` per phase. `run_es` takes its phases via `EsConfig.phases: Vec<Phase>` — production uses `PHASES.to_vec()`; the smoke test uses a single-element `Vec`.

## Shader pipeline

Both shaders are loaded via `include_str!` and fed to `wgpu` as `ShaderSource::Wgsl`:

- [shader.wgsl](src/polygenvo/shader.wgsl) — trivial passthrough vertex+fragment for rendering the triangle list.
- [fitness.wgsl](src/polygenvo/fitness.wgsl) — compute shader; CIELAB ΔE76 per pixel + `atomicAdd` into a single `u32`.

Edits to either `.wgsl` file take effect on the next `cargo build` because `include_str!` paths are tracked for rebuilds.

WGSL syntax is **current (post-1.0)**: `@location(0)`, `@vertex`/`@fragment`/`@compute`, `@group(N) @binding(M)`, comma-separated struct fields. There is no `build.rs`, no `shaderc`, no SPIR-V path.

## Dependency versions

`Cargo.toml` is on a current ecosystem snapshot: `wgpu = "29"`, `image = "0.25"`, `env_logger = "0.11"`, `rand = "0.10"`, `bytemuck = "1"`, `futures = "0.3"`, `log = "0.4"`. Rust edition is `"2024"`.

`rand 0.10` notes for editing: use `rand::rng()` (not `thread_rng()`) and `rng.random_range(a..b)` (not `gen_range`). The `gen` keyword is reserved in edition 2024; if a method name collides it must be escaped as `r#gen`.

## Conventions when editing

- When changing tunables, verify both the `const` at the top and any `PHASES` entries that override per-phase values (`initial_sigma` is per-phase; `LAMBDA` and `MAX_STEPS` are global).
- The fitness direction is **higher = better**. Code that compares fitnesses uses `>` for "improvement".
- The smoke test is the only regression guard — if you change `run_es`, `FitnessCalc`, or either `.wgsl`, run `cargo test --bin polygenvo` before assuming the change is safe.
- PNG snapshots are gated on `cfg.snapshot_every`: production sets `Some(SNAPSHOT_EVERY_IMPROVEMENT)`, the smoke test sets `None`. When adding new snapshot sites, gate them too.
