# wgpu modernization (0.12 → 29)

**Date:** 2026-05-27
**Branch:** `wgpu-modernize` (off `master` @ `3fefcf9`)
**Status:** Design approved, ready for implementation plan

## Goal

Migrate `polygenvo` (the only active binary) from `wgpu` 0.12 to the latest stable `wgpu` (29.x). Pure API/syntax migration — preserve the existing (1+λ)-ES algorithm, fitness shader semantics, and runtime behavior. No perf restructuring, no algorithmic changes.

## Motivation

The codebase is pinned to a 4-year-old wgpu generation. The WGSL files use a pre-1.0 dialect that current `naga` will not parse. Modernization unblocks future use of newer wgpu features (pipeline cache, timestamp queries, render bundles) and removes a stale ecosystem dependency. Speed gains are explicitly **not** a goal — the hot path is GPU-sync-bound, and recent profiling-driven work in Tier 1 / Tier 2 already optimized the algorithm above the API layer.

## Non-goals

The following are **out of scope** for this branch and must not be mixed in:

- Batching multiple genomes per submit
- Persistent staging buffer / pipelined readback (current code does `poll(Wait)` per evaluation)
- Splitting the binary into `[lib]` + `[bin]`
- Re-introducing `winit` or any window-display path (binary stays headless)
- Re-architecting `(1+λ)`-ES into a different evolution scheme
- Re-writing `fitness.wgsl` to a different fitness metric
- Any change to `CLAUDE.md` beyond a final pass at the end to correct stale facts

## Approach

Two commits on `wgpu-modernize` branch, both atomic:

1. **Commit 1 — cleanup + testability + smoke test on wgpu 0.12.** Delete dead binaries and dead shader pipeline, drop unused deps, refactor for testability, add smoke test. Smoke test passes on wgpu 0.12. This commit verifies the baseline.
2. **Commit 2 — atomic wgpu 0.12 → 29 migration.** Bump dep, rewrite both `.wgsl` files, update every wgpu API call site in `main.rs`. Smoke test still passes on wgpu 29. Release build of `polygenvo` succeeds.

This sequencing gives a clean bisect point at the migration boundary and uses the smoke test as a regression guard across the API boundary.

## Commit 1 — Cleanup + testability

### Deletions

- `src/genalgo/` — abandoned `oxigen` experiment, `main()` only prints "Not implemented!"
- `src/genevoalgo/` — earlier standalone `genevo` reference experiment
- `src/polygen/` — earlier wgpu render-to-texture reference experiment
- `src/polygenvo/shader.vert`, `src/polygenvo/shader.frag` — GLSL source for dead pipeline
- `src/polygenvo/shader.vert.spv`, `src/polygenvo/shader.frag.spv` — gitignored build artifacts (no commit needed but should not be re-generated)
- `build.rs` — the entire shaderc-based GLSL → SPIR-V pipeline is dead code; only WGSL is consumed by `polygenvo` via `include_str!`. Delete the file entirely.

### Cargo.toml

- Remove three `[[bin]]` entries: `polygen`, `genalgo`, `genevoalgo`. Keep only `polygenvo`.
- Remove the entire `[build-dependencies]` section (`anyhow`, `fs_extra`, `glob`, `shaderc`) — `build.rs` is gone.
- Remove `genevo = "0.7"` — unused after `polygenvo` migrated to its own (1+λ)-ES in Tier 2; no remaining binary references it.
- Bump `rand = "0.8"` → `rand = "0.10"`. Now possible because the `genevo ^0.8` constraint is gone.
- Keep all other deps as committed in `3fefcf9` (post-bundle state).

### Source refactor (in `src/polygenvo/main.rs`)

Extract the ES loop and its setup into a callable function so tests can exercise it without depending on `goal.png` on disk:

```rust
pub struct EsConfig {
    pub phases: Vec<Phase>,           // matches the const PHASES schedule shape
    pub max_steps: u64,
    pub lambda: usize,
    pub snapshot_every: Option<u64>,  // None disables PNG snapshots in triangles/
}

pub struct EsResult {
    pub initial_fitness: usize,       // matches fitness_of() return type
    pub final_fitness: usize,
    pub steps_run: u64,
}

fn run_es(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    goal: GoalImage,
    cfg: EsConfig,
) -> EsResult { /* moved from main */ }
```

`main()` becomes a thin wrapper:

1. `env_logger::init()`
2. Open `goal.png` into a `GoalImage`
3. Initialize wgpu (`Instance`, `Adapter`, `Device`, `Queue` wrapped in `Arc`)
4. Construct a production `EsConfig` (constants pulled from the existing tunables at the top of `main.rs`)
5. Call `run_es(...)`
6. Print final fitness

The ES uses a multi-phase coarse-to-fine schedule defined in `const PHASES: &[Phase]` (line 476 of `main.rs`), where each `Phase` has `triangles`, `pyramid_level`, and `initial_sigma`. Production constructs `EsConfig { phases: PHASES.to_vec(), ... }`; the smoke test passes a single-element `Vec<Phase>`. Other tunables (`MAX_VERTICES`, `SIGMA_WINDOW`, `PLATEAU_WINDOW`, `SNAPSHOT_EVERY_IMPROVEMENT`) stay as private module-level constants — only the four fields above are surfaced because they are what the smoke test needs to override.

**Fitness direction.** `FitnessCalc::fitness_of` (line 308) returns `usize` in `[0, 1_000_000]` where **higher = better fit**: the GPU sums per-pixel ΔE in CIELAB, then Rust converts to `1 - sum / max_total` and scales to `1_000_000`. The smoke test assertion direction is `final_fitness >= initial_fitness`.

### rand 0.8 → 0.10 migration

Approximate call-site count: ~20 in `polygenvo/main.rs` (combined `thread_rng` + `gen_range` matches). Mechanical renames:

- `thread_rng()` → `rand::rng()`
- `rng.gen_range(a..b)` → `rng.random_range(a..b)`
- `rand::prelude::*` import line stays; trait names are unchanged (`Rng`, `SeedableRng`)
- Verify no `.gen::<T>()` bare calls (none expected post the genevoalgo deletion, but worth grepping)

### Smoke test

Added as `#[cfg(test)] mod tests` at the bottom of `src/polygenvo/main.rs`. Single test:

```rust
#[test]
fn ga_improves_on_synthetic_checker() {
    let goal = make_checker_goal(32);                    // 32×32 RGBA checker
    let (device, queue) = futures::executor::block_on(init_test_wgpu());
    let test_phases = vec![Phase { triangles: 6, pyramid_level: 0, initial_sigma: 0.1 }];
    let result = run_es(device, queue, goal, EsConfig {
        phases: test_phases,
        max_steps: 30,
        lambda: 4,
        snapshot_every: None,
    });
    assert!(result.steps_run > 0);
    assert!(result.final_fitness <= 1_000_000);
    // higher fitness = better fit (see FitnessCalc::fitness_of doc).
    assert!(result.final_fitness >= result.initial_fitness);
}
```

Helpers added in the same module:
- `make_checker_goal(size: u32) -> GoalImage` — constructs an `ImageBuffer<Rgba<u8>, _>` with a black/white checker pattern, wraps in `GoalImage`. No disk I/O.
- `init_test_wgpu() -> (Arc<Device>, Arc<Queue>)` — async helper that mirrors `main()`'s wgpu init but with no surface and minimal feature/limit requests.

**Caveat:** the test requires a working wgpu adapter on the host. On a Linux workstation with Vulkan (current environment) this is fine. On a headless CI machine it would need `WGPU_BACKEND=gl` or `llvmpipe`. There is no CI in this project; this is acceptable.

### Verification of commit 1

- `cargo build --release --bin polygenvo` succeeds
- `cargo test --bin polygenvo` runs and passes (the new smoke test)
- `cargo run --release --bin polygenvo` (manual, with `goal.png` present) still produces output frames in `triangles/`

## Commit 2 — wgpu 0.12 → 29 atomic migration

### Cargo.toml

- `wgpu = { version = "0.12.0", features = ["spirv"] }` → `wgpu = "29"`. Drop the `spirv` feature: confirmed via grep that `main.rs` uses `ShaderSource::Wgsl(include_str!(...))` for both `shader.wgsl` and `fitness.wgsl`, and no `ShaderSource::SpirV` or related call sites exist.
- No other dependency changes in this commit

### WGSL rewrite

Both `src/polygenvo/shader.wgsl` and `src/polygenvo/fitness.wgsl` migrated from pre-1.0 syntax. **No semantic changes.** Syntax-only mapping:

| Pre-1.0 | Current WGSL |
|---|---|
| `[[location(N)]]` | `@location(N)` |
| `[[builtin(X)]]` | `@builtin(X)` |
| `[[stage(vertex)]]` | `@vertex` |
| `[[stage(fragment)]]` | `@fragment` |
| `[[stage(compute), workgroup_size(8,8,1)]]` | `@compute @workgroup_size(8, 8, 1)` |
| `[[group(0), binding(0)]]` | `@group(0) @binding(0)` |
| struct field `;` separator | `,` separator |

Algorithmic code (RGB → XYZ → Lab, ΔE76, atomic add) is unchanged. `textureLoad(t, vec2<i32>(...), 0)` syntax is unchanged. `select(false, true, cond)` argument order is unchanged. `atomic<u32>`, `var<storage, read_write>`, `var<uniform>` are unchanged.

### main.rs wgpu API call-site updates

The following changes are expected based on documented wgpu evolution. Exact list to be confirmed by attempting `cargo build` and resolving compile errors at implementation time.

**Instance / Adapter / Device construction:**
- `Instance::new(Backends::all())` → `Instance::new(&InstanceDescriptor { backends, flags: InstanceFlags::default(), backend_options: BackendOptions::default() })`
- `DeviceDescriptor` field renames: `features` → `required_features`, `limits` → `required_limits`; add `memory_hints: MemoryHints::default()`
- `adapter.request_device(&desc, None).await` → `adapter.request_device(&desc).await` (second `trace_path` arg removed)
- Verify `Features` flag names — `MAPPABLE_PRIMARY_BUFFERS` etc. mostly stable but worth grepping

**Textures:**
- `TextureDescriptor`: add `view_formats: &[]`
- Verify render target format (`Rgba8UnormSrgb`) is still usable as `texture_2d<f32>` sample binding in fitness compute — if naga rejects, set `view_formats: &[TextureFormat::Rgba8Unorm]` and create an unorm view for the storage binding, or unify on a single format

**Buffers:**
- `BufferDescriptor` unchanged
- `util::DeviceExt::create_buffer_init` still works
- `Buffer::slice(..).map_async(MapMode::Read, |result| ...)`: callback type is now `FnOnce(Result<(), BufferAsyncError>) + Send + 'static`. The existing channel-based wait pattern continues to work; just adjust the callback signature.

**Pipelines:**
- `ShaderModuleDescriptor`: drop the `flags` field
- `RenderPipelineDescriptor`: wrap module+entry+buffers in `VertexState`/`FragmentState` with `compilation_options: PipelineCompilationOptions::default()`; add `multiview: None`, `cache: None`
- `ComputePipelineDescriptor`: add `compilation_options`, `cache`

**Passes:**
- `RenderPassDescriptor`: `color_attachments: &[Some(RenderPassColorAttachment { .. })]` (already `Option`-wrapped in 0.14+); add `timestamp_writes: None`, `occlusion_query_set: None`
- `ComputePassDescriptor`: add `timestamp_writes: None`

**Polling:**
- `device.poll(Maintain::Wait)` → `device.poll(PollType::Wait).unwrap()` (renamed in 23; `poll` now returns `Result<MaintainResult>`)

**Bind groups:**
- `BindGroupLayoutEntry`, `BindGroupEntry` shapes mostly unchanged
- `BufferBindingType::Uniform` simplified (no `dynamic` field)

### Texture-format / sRGB watch-out

The render target is `Rgba8UnormSrgb` and the compute shader binds it as `texture_2d<f32>`. In wgpu 0.12 this worked silently. Current wgpu validates view formats more strictly. Mitigation paths if validation rejects:

1. Add `view_formats: &[TextureFormat::Rgba8UnormSrgb]` to the texture descriptor (probably sufficient).
2. If a separate linear view is needed for the compute binding, add `Rgba8Unorm` to `view_formats` and create a second `TextureView` with `format: Some(Rgba8Unorm)` for the compute bind group.

Behavior must be preserved: the fitness shader's comment states "textures are sRGB-formatted so the hardware does the decode on read" — `textureLoad` of `Rgba8UnormSrgb` returns linear values. Whatever path keeps that semantic is the correct choice.

### Verification of commit 2

- `cargo build --release --bin polygenvo` succeeds
- `cargo test --bin polygenvo` passes the smoke test
- `cargo run --release --bin polygenvo` (manual, with `goal.png` present) produces output frames in `triangles/` that show fitness improving across generations (eyeballed against pre-migration baseline)

## Risks

- **Naga rejection of WGSL** beyond the syntax migration. If naga complains about a missing attribute, undeclared resource, or workgroup size constraint, fix inline — these are usually one-line corrections.
- **Validation hard-failures at pipeline creation** that were silent in 0.12. Examples: missing `TextureUsages::COPY_SRC` on a texture later used in `copy_texture_to_buffer`, bind group layout mismatch between shader resources and Rust-side `BindGroupLayoutEntry`. The smoke test surfaces these immediately rather than at GA-runtime.
- **Texture format / sRGB view validation** (described above). Mitigation paths documented.
- **`poll(Wait)` now returning `Result`** — every call site must `.unwrap()` or propagate. Two call sites exist today (both `inner.device.poll(wgpu::Maintain::Wait)` in `main.rs`); both need updating.
- **Smoke test requires a working wgpu adapter on the dev machine.** Confirmed available on the current Linux workstation via Vulkan.

## Post-merge cleanup (separate, non-blocking)

Refresh `CLAUDE.md` to reflect:
- Line count of `polygenvo/main.rs` (~803, not 1300)
- `genevo` is no longer a dependency; the GA uses a hand-rolled (1+λ)-ES
- CPU fitness fallback no longer exists (removed in Tier 1)
- Fitness shader is CIELAB ΔE76 (not multi-scale SSIM as the doc currently describes)
- Build script no longer exists; shaders are WGSL-only via `include_str!`
- Only one binary remains (`polygenvo`)

This is a docs-only follow-up commit on this branch or a separate one — author's preference at merge time.
