# gen-polygons

Approximate a raster image by evolving a population of coloured, semi-transparent
triangles. Starting from a handful of random triangles, an evolution strategy nudges
their positions, colours, and opacities — and occasionally adds, splits, or removes
them — until the rendered overlap of triangles resembles the target image.

It is an experimental playground for **`wgpu`**, **compute shaders**, and **genetic
algorithms**: every fitness evaluation is a real GPU render scored by a CIELAB
compute pass, so the search runs at thousands of candidate evaluations per second.

The one binary is **`polygenvo`**.

```
goal.png                 evolved triangles                rendered result
┌───────────┐            ╱╲    ╱╲  ╱╲                     ┌───────────┐
│  target   │   ───▶    ╱  ╲  ╱  ╲╱  ╲    ───▶            │ ~960k/1M  │
│  raster   │          ╱____╲╱________╲                   │ similarity│
└───────────┘          (hundreds, alpha-blended)          └───────────┘
```

## How it works

1. **Genome.** A candidate is a `Vec<Vertex>` interpreted as a GPU `TriangleList`
   (3 vertices per triangle, each carrying an RGBA colour). Triangles are seeded by
   sampling the goal image's colour at the triangle's centre.

2. **Fitness on the GPU.** Each candidate is rendered into an offscreen texture with
   real OVER alpha blending, then a compute shader compares it to the goal pixel by
   pixel. Both images are converted linear-RGB → CIE XYZ → **CIELAB**, and the
   per-pixel **ΔE76** perceptual distance is summed into a single similarity score in
   `[0, 1_000_000]` where **higher = better**. The same pass also emits a coarse
   16×16 grid of where the error is concentrated, used to steer where new triangles
   are placed. All λ candidates of a generation are scored in a single GPU submit.

3. **The search — a (1+λ)-ES.** Each step mutates the current best parent into λ
   children, scores them, and keeps the best one if it beats the parent. Mutation
   picks one operator from a weighted table: nudge a vertex, recolour, change opacity,
   **split** a high-error triangle into four (the only way the genome grows — and only
   if it improves the fit), swap draw order, add a triangle in a high-error region,
   relocate one, or delete one. Step sizes self-adapt via the **1/5 success rule**.

4. **Coarse-to-fine.** The goal is downsampled into a pyramid (128² → 256² → 512²).
   The search starts coarse and cheap; when it plateaus it promotes to a finer level
   and a higher triangle cap. Detail emerges organically rather than all at once.

There is no crossover and no population — just a parent, its mutated challengers, and
a perceptual GPU scorer. (CMA-ES was considered and rejected: it assumes a fixed
problem dimension, which clashes with a genome whose triangle count changes.)

## Build & run

Requires a working Rust toolchain (edition 2024) and a GPU/driver that exposes a
`wgpu` GL adapter.

```sh
# Run the simulation (release is effectively required — many GPU renders per step)
cargo run --release --bin polygenvo

# Fast iteration on Rust code paths
cargo build --bin polygenvo

# Tests: per-module unit tests + a GPU smoke test (~0.1s, needs a wgpu adapter)
cargo test --bin polygenvo
```

**Runtime requirements (relative to the working directory):**

- **`goal.png`** must exist — a square RGBA PNG. Its width sets the working resolution
  (512×512 is the tuned size).
- **`triangles/`** receives progress snapshots (`image0.png`, `imageN.png`,
  `final.png`). It is created automatically and is gitignored.

The run prints a progress line each second (step rate, triangle count, current
fitness, step sizes) and snapshots the best result periodically. On the bundled
512² goal it climbs to roughly **955k–965k / 1,000,000** and grows to several hundred
triangles within a couple of minutes; let it run longer to refine further. Stop it
with `Ctrl-C` and inspect the frames in `triangles/`.

## Project layout

Everything lives under [`src/polygenvo/`](src/polygenvo/), split into
single-responsibility modules layered low → high:

| Module | Responsibility |
|---|---|
| [`goal.rs`](src/polygenvo/goal.rs) | the target image: loading, Lanczos downsampling, colour sampling |
| [`genome.rs`](src/polygenvo/genome.rs) | `Vertex`, triangle seeding, midpoint subdivision, capacity constants |
| [`gpu.rs`](src/polygenvo/gpu.rs) | `wgpu` device/queue bring-up |
| [`fitness.rs`](src/polygenvo/fitness.rs) | the GPU evaluator: render pipeline, scoring compute pass, the pyramid |
| [`variation.rs`](src/polygenvo/variation.rs) | mutation operators and the weighted operator table |
| [`es.rs`](src/polygenvo/es.rs) | the (1+λ)-ES driver, step-size adaptation, phase schedule |
| [`main.rs`](src/polygenvo/main.rs) | thin entry point |

Two WGSL shaders are compiled in via `include_str!` (no build script, no SPIR-V):
[`shader.wgsl`](src/polygenvo/shader.wgsl) renders the triangles, and
[`fitness.wgsl`](src/polygenvo/fitness.wgsl) does the CIELAB ΔE76 scoring + error grid.

## Tuning

The interesting knobs live next to the module that owns them:

- **`es.rs`** — the `PHASES` schedule (per-phase triangle cap, pyramid level, initial
  step sizes), the 1/5-rule window, and plateau-promotion thresholds.
- **`fitness.rs`** — `LAMBDA` (candidates per generation), `FITNESS_SCALE`, and the
  16×16 error-grid dimension.
- **`genome.rs`** — `MAX_TRIANGLES`, `INITIAL_TRIANGLES`.
- **`variation.rs`** — the `OPERATORS` weight table that sets how often each mutation
  is tried.

See [CLAUDE.md](CLAUDE.md) for a deeper architecture tour and editing conventions, and
[`docs/superpowers/`](docs/superpowers/) for design notes (including a documented
performance follow-up for the fitness shader).

## Status

A personal experiment, not a polished tool — no CLI flags or configuration files yet;
behaviour is changed by editing the constants above. Substantive changes are validated
by running it and eyeballing the frames in `triangles/`.
