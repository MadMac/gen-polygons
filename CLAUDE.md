# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Experimental playground for `wgpu`, shaders, and genetic algorithms. The goal is to approximate a raster image (`goal.png` at the repo root) by evolving a population of colored triangles. There is no test suite, no CI, no public API — changes are evaluated by running the simulation and eyeballing output frames in `triangles/`.

## Commands

- `cargo build` — also compiles shaders (see "Shader pipeline" below).
- `cargo run --release --bin polygenvo` — the active binary; release mode is essentially required because the GA does many GPU renders per generation.
- `cargo run --bin polygenvo` — debug build; slow but useful when iterating on Rust code paths.

Runtime requirements for `polygenvo`:
- `goal.png` must exist in the working directory (square RGBA PNG; `texture_size` is taken from its width).
- The `triangles/` directory must exist — the binary writes `triangles/imageN.png` snapshots and will panic on `save()` if the directory is missing. It is gitignored.

## Binaries

Cargo defines four binaries in [Cargo.toml](Cargo.toml), but only one is actually live:

- [`polygenvo`](src/polygenvo/main.rs) — the active binary. Genetic algorithm (via `genevo`) + `wgpu` rendering + GPU/CPU fitness scoring against `goal.png`. **This is what you should be reading and editing.**
- [`genevoalgo`](src/genevoalgo/main.rs) — earlier standalone `genevo` experiment with no rendering; fitness is a meaningless sum of vertex components. Kept as a reference for the bare `genevo` wiring.
- [`polygen`](src/polygen/main.rs) — early `wgpu` render-to-texture experiment, no GA. Renders an empty vertex buffer and writes `image.png`. Reference only.
- [`genalgo`](src/genalgo/main.rs) — abandoned `oxigen`-based attempt; the file is almost entirely commented out and `main()` prints "Not implemented!".

When the user says "the app" or "the simulation," they mean `polygenvo`.

## High-level architecture (polygenvo)

The single ~1300-line [main.rs](src/polygenvo/main.rs) wires together three layers:

1. **Genome** — `Vertex { position: [f32;3], color: [f32;4] }` matching the wgpu vertex layout exactly (it's `bytemuck::Pod`, so the genome buffer is `cast_slice`'d straight to a vertex buffer). A genome is a `Vec<Vertex>` interpreted as a `TriangleList`, so length must stay a multiple of 3. The `Pictures` `GenomeBuilder` produces vertices in `(-0.4, 0.4)` — recent commits intentionally shrank this range to force smaller, more detailed triangles, so do not "fix" it back to `(-1.0, 1.0)` without understanding the trade-off.

2. **Fitness (`FitnessCalc`)** — the GA's `fitness_of` does a full wgpu render of the genome's triangles into an offscreen texture, then scores it against `goal.png`. Two paths:
   - **GPU compute** (`use_gpu_fitness = true`, default when the pipeline initializes): dispatches [fitness.wgsl](src/polygenvo/fitness.wgsl), a multi-scale comparison combining color difference, Sobel edge structure, and SSIM-style local statistics. Parameters are packed into a `FitnessParams` uniform; weights are passed in as `u32`s scaled by 1000.
   - **CPU fallback**: pixel-by-pixel RGB diff with aggressive subsampling (`sample_step` of 3–4, only sampling the top-left ¾ region for speed).
   Both paths add a per-vertex-count bonus to bias the GA toward keeping more triangles (prevents premature simplification). `FitnessCalc` holds `&` references to `device`/`queue`, and its hand-written `Clone` impl rebuilds the render pipeline and output buffer rather than sharing them — required because `genevo` clones the fitness function into selection/reinsertion operators.

3. **GA driver** — `genevo::simulate(...)` with `MaximizeSelector`, `UniformCrossBreeder`, `BreederValueMutator`, `ElitistReinserter`. The main loop in `fn main()` steps the simulation, tracks fitness history in a `VecDeque`, and **adaptively adjusts `current_mutation_rate`** based on recent improvement rate (raise on stagnation, lower on fast improvement). Note: the mutation rate variable is updated but the `BreederValueMutator` is constructed once before the loop — changing the variable mid-run does not feed back into the mutator. This is a known gap; verify before claiming an adaptive-mutation change actually takes effect.

The simulation terminates on either `FitnessLimit` or the in-loop convergence check (small fitness range across the last 10 generations with high mutation rate). `GENERATION_LIMIT` and `PHASE_DURATION` constants exist but are not wired into the termination condition in the current code path.

Tunable constants live at the top of `polygenvo/main.rs`: `INITIAL_VERTICES`, `MAX_VERTICES`, `VERTICES_INCREMENT`, `POPULATION_SIZE`, `GENERATION_LIMIT`, `PHASE_DURATION`. The vertex-range literals (`-0.4..0.4`) and triangle-size bounds in the `BreederValueMutator` min/max `Vertex`es must be kept in sync — they're duplicated across `Pictures::build_genome`, `RandomValueMutation`, and the mutator setup in `main`.

## Shader pipeline

[build.rs](build.rs) globs `src/**/*.{vert,frag,comp}` and uses `shaderc` to compile each to a sibling `.spv` file at build time (with `cargo:rerun-if-changed`). The committed `.vert`/`.frag` files are GLSL; the `.spv` outputs are gitignored.

**However**, `polygenvo` does not load the SPIR-V — it `include_str!`s [shader.wgsl](src/polygenvo/shader.wgsl) (trivial passthrough vertex+fragment) and [fitness.wgsl](src/polygenvo/fitness.wgsl) (compute) at compile time and feeds them to wgpu as `ShaderSource::Wgsl`. So:
- Edits to `.wgsl` files take effect on the next `cargo build` (they trigger a recompile because `include_str!` is tracked).
- Edits to `.vert`/`.frag` files compile via `build.rs` but are **dead code** for the active binary. Don't bother updating them unless you're also rewiring the binary to consume SPIR-V.

The WGSL syntax in this repo is the **pre-1.0 wgpu 0.12 dialect** (`[[location(0)]]` attributes, `;` field separators in structs, `textureLoad` with `i32` coords). It will not parse with current wgpu/naga. Match the existing style when editing.

## Dependency versions to be aware of

`Cargo.toml` is pinned to an old generation of the ecosystem: `wgpu = 0.12`, `winit = 0.26`, `cgmath = 0.17`, `image = 0.24`. APIs differ substantially from current versions — when in doubt, mirror patterns from the existing code rather than what current docs suggest. `winit` is imported but the event loop in `polygenvo` is entirely commented out; the binary runs headless.
