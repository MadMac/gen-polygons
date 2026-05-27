# wgpu modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `polygenvo` (the only active binary) from `wgpu` 0.12 to the latest stable `wgpu` (29.x). Pure API/syntax migration — preserve the (1+λ)-ES algorithm and fitness shader semantics; no perf restructuring.

**Architecture:** Two atomic commits on branch `wgpu-modernize`. Commit 1 deletes dead code, drops `genevo`, bumps `rand` to 0.10, extracts the ES loop into a `run_es(...)` function, and adds a smoke test — all still on wgpu 0.12. Commit 2 atomically migrates wgpu (0.12 → 29) and both WGSL files (pre-1.0 dialect → current). The smoke test from commit 1 acts as the regression guard across the API boundary.

**Tech Stack:** Rust 2024, wgpu (0.12 → 29.x), WGSL (pre-1.0 → current), `rand` 0.10, `image` 0.25, `bytemuck`, hand-rolled (1+λ)-ES (no `genevo` after this branch).

**Reference spec:** [docs/superpowers/specs/2026-05-27-wgpu-modernization-design.md](../specs/2026-05-27-wgpu-modernization-design.md)

**Branch state at plan start:** `wgpu-modernize` checked out off `master @ 3fefcf9`. Spec already committed as `09b69d9`.

---

## Phase 1 — Cleanup, refactor, smoke test (still on wgpu 0.12)

Tasks 1–5 share a single commit at the end (Task 5). Do **not** create intermediate commits inside Phase 1 — the spec requires Phase 1 to land as one atomic commit. If a task fails midway, fix it before moving on.

### Task 1: Delete dead binaries and dead shader pipeline

**Files:**
- Delete: `src/genalgo/` (entire directory)
- Delete: `src/genevoalgo/` (entire directory)
- Delete: `src/polygen/` (entire directory)
- Delete: `src/polygenvo/shader.vert`
- Delete: `src/polygenvo/shader.frag`
- Delete: `src/polygenvo/shader.vert.spv`
- Delete: `src/polygenvo/shader.frag.spv`
- Delete: `build.rs`
- Modify: `Cargo.toml` — remove three `[[bin]]` entries and entire `[build-dependencies]` section

- [ ] **Step 1: Delete the three dead binary source directories**

```bash
rm -rf src/genalgo src/genevoalgo src/polygen
```

- [ ] **Step 2: Delete the dead `.vert`/`.frag`/`.spv` files**

```bash
rm -f src/polygenvo/shader.vert src/polygenvo/shader.frag \
      src/polygenvo/shader.vert.spv src/polygenvo/shader.frag.spv
```

- [ ] **Step 3: Delete `build.rs`**

```bash
rm -f build.rs
```

- [ ] **Step 4: Edit `Cargo.toml` to remove the three dead `[[bin]]` entries and `[build-dependencies]`**

After this edit, the file should contain `[package]`, `[dependencies]`, and exactly one `[[bin]]` block for `polygenvo`. Remove these blocks:

```toml
[build-dependencies]
anyhow = "1"
fs_extra = "1"
glob = "0.3"
shaderc = "0.10.1"

[[bin]]
name = "polygen"
path = "src/polygen/main.rs"

[[bin]]
name = "genalgo"
path = "src/genalgo/main.rs"

[[bin]]
name = "genevoalgo"
path = "src/genevoalgo/main.rs"
```

Keep the `[[bin]]` entry for `polygenvo`.

- [ ] **Step 5: Verify the build still compiles**

```bash
cargo build --bin polygenvo
```

Expected: succeeds with no errors. Warnings about dead code in `Vertex`/`check` are pre-existing and acceptable.

**Do not commit yet.** Move to Task 2.

---

### Task 2: Drop `genevo` and bump `rand` 0.8 → 0.10

**Files:**
- Modify: `Cargo.toml` — remove `genevo`, bump `rand` to `0.10`
- Modify: `src/polygenvo/main.rs` — update ~20 `rand` call sites

The `genevo` dependency is no longer referenced by any binary after Task 1 (Tier 2 commit replaced the GA with a hand-rolled (1+λ)-ES). Removing it lifts the `rand ^0.8` constraint, freeing the bump to 0.10.

- [ ] **Step 1: Edit `Cargo.toml`**

Remove the line:
```toml
genevo = "0.7"
```

Change:
```toml
rand = "0.8"
```
to:
```toml
rand = "0.10"
```

- [ ] **Step 2: Update `rand` API call sites in `src/polygenvo/main.rs`**

Mechanical renames. Use `grep -n "thread_rng\|gen_range" src/polygenvo/main.rs` to find all sites.

Renames:
- `thread_rng()` → `rand::rng()` (1 site, around line 661)
- `rng.gen_range(a..b)` → `rng.random_range(a..b)` (~19 sites in `random_color_seeded_triangle`, `mutate`, etc.)

The `use rand::prelude::*;` import line stays — `Rng` and `SeedableRng` trait names are unchanged. If `rand::prelude::*` doesn't bring in the right traits in 0.10, replace with explicit imports: `use rand::{Rng, rngs::ThreadRng};`.

- [ ] **Step 3: Verify build**

```bash
cargo build --bin polygenvo
```

Expected: succeeds. If there are residual errors about distribution traits (`Standard`, `Uniform`), they likely indicate a missed call site or an import change needed for 0.10. Resolve them.

**Do not commit yet.** Move to Task 3.

---

### Task 3: Extract `run_es` function and define `EsConfig`/`EsResult`

**Files:**
- Modify: `src/polygenvo/main.rs` — extract the body of `main()` (currently starting around line 632) into a new function `run_es`, define two new public structs.

The goal is to make the ES loop callable from a test without depending on `goal.png` on disk. Keep all existing module-level constants (`MAX_VERTICES`, `SIGMA_WINDOW`, `PLATEAU_WINDOW`, etc.). Only surface the values the smoke test needs to override as `EsConfig` fields.

- [ ] **Step 1: Add `EsConfig` and `EsResult` structs**

Background (verified against current code): the ES uses a multi-phase coarse-to-fine schedule defined as `const PHASES: &[Phase] = &[ ... ]` at line 476 of `main.rs`. The `Phase` struct (line 469) has three fields: `triangles: usize`, `pyramid_level: usize`, `initial_sigma: f32`. The ES initializes from `PHASES[0]` and promotes through phases as plateaus are detected. Also: `fitness_of` returns `usize` in `[0, 1_000_000]` where **higher = better fit** (line 308 doc comment + the `if f > best_fit` comparison at line 711). The `EsConfig` and `EsResult` types below reflect these realities.

Place these after the existing `const PHASES` declaration (around line 482, before `fn downsample_goal`):

```rust
pub struct EsConfig {
    pub phases: Vec<Phase>,
    pub max_steps: u64,
    pub lambda: usize,
    pub snapshot_every: Option<u64>,
}

impl EsConfig {
    fn production() -> Self {
        Self {
            phases: PHASES.to_vec(),
            max_steps: MAX_STEPS,
            lambda: LAMBDA,
            snapshot_every: Some(SNAPSHOT_EVERY_IMPROVEMENT),
        }
    }
}

pub struct EsResult {
    pub initial_fitness: usize,
    pub final_fitness: usize,
    pub steps_run: u64,
}
```

Add `#[derive(Clone)]` to `struct Phase` at line 469 so `PHASES.to_vec()` compiles.

- [ ] **Step 2: Extract the ES loop into `run_es`**

Define a new function with this signature:

```rust
fn run_es(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    goal: GoalImage,
    cfg: EsConfig,
) -> EsResult {
    // (moved from main, minus env_logger::init, goal.png load, wgpu init)
    ...
}
```

The function body is the current contents of `main()` starting around line 659 (after the wgpu init and goal-load blocks), with these adjustments:
- Use `cfg.phases` (a `Vec<Phase>`) in place of `PHASES`. Wherever the current code references `PHASES[i]`, use `cfg.phases[i]`. Wherever it references `PHASES.len()`, use `cfg.phases.len()`.
- Use `cfg.max_steps` instead of the `MAX_STEPS` constant
- Use `cfg.lambda` instead of the `LAMBDA` constant
- Gate PNG snapshot writes on `cfg.snapshot_every`: if `None`, skip both the `triangles/` directory creation (current line 692) and any `snapshot()` calls inside the loop. If `Some(n)`, snapshot every `n` improvements as the production code currently does.
- At function start (after the initial parent evaluation at current line 668), save that value as `initial_fitness: usize`. The `current_fitness` variable already exists; just capture its initial value before entering the loop.
- Track `steps_run` as the loop counter at function exit. `step` already exists as a `u64` in the loop; return it as `steps_run`.
- At function end, return `EsResult { initial_fitness, final_fitness: current_fitness, steps_run: step }`.

If the existing code passes any of those constants into helper functions as arguments, thread the `cfg` value through instead.

- [ ] **Step 3: Reduce `main()` to a thin wrapper**

`main()` should now look approximately like:

```rust
fn main() {
    env_logger::init();
    let goal = load_goal_image("goal.png");
    let (device, queue) = block_on(init_wgpu());
    let cfg = EsConfig::production();
    let result = run_es(device, queue, goal, cfg);
    println!(
        "Done. Initial fitness: {:.2}, final fitness: {:.2}, steps: {}",
        result.initial_fitness, result.final_fitness, result.steps_run
    );
}
```

Extract the goal-loading code and the wgpu-init code into helpers `load_goal_image(path: &str) -> GoalImage` and `init_wgpu() -> impl Future<Output = (Arc<wgpu::Device>, Arc<wgpu::Queue>)>`. These already exist as inline blocks in the current `main()` — turn them into named functions so they can be reused by `init_test_wgpu` in Task 4.

- [ ] **Step 4: Verify build and that nothing broke at runtime**

```bash
cargo build --bin polygenvo
```

Expected: succeeds. The refactor is structural only — behavior is preserved.

Optionally (recommended), do a brief smoke run to confirm the refactor is correct:

```bash
cargo run --release --bin polygenvo 2>&1 | head -20
```

(Requires `goal.png` and `triangles/` in the working directory.) Expected: process starts, prints fitness numbers, makes progress. Kill it with Ctrl-C after a few seconds.

**Do not commit yet.** Move to Task 4.

---

### Task 4: Add smoke test helpers and the test

**Files:**
- Modify: `src/polygenvo/main.rs` — add `#[cfg(test)] mod tests` block at the bottom

- [ ] **Step 1: Append the test module to `src/polygenvo/main.rs`**

Add this block as the last thing in the file:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    fn make_checker_goal(size: u32) -> GoalImage {
        // Construct a black/white checker pattern at the requested resolution.
        let mut buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(size, size);
        let cell = (size / 4).max(1);  // 4×4 logical cells; min 1px
        for y in 0..size {
            for x in 0..size {
                let on = ((x / cell) + (y / cell)) % 2 == 0;
                let v = if on { 255 } else { 0 };
                buf.put_pixel(x, y, Rgba([v, v, v, 255]));
            }
        }
        GoalImage { goal_image: buf }
    }

    fn init_test_wgpu() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        // Mirrors init_wgpu() from Task 3 but constructed inline for the test
        // so it can run without main.rs's environment setup. Adjust the
        // Instance/Adapter/Device construction to match whatever init_wgpu()
        // uses in this codebase.
        block_on(init_wgpu())
    }

    #[test]
    fn ga_improves_on_synthetic_checker() {
        let goal = make_checker_goal(32);
        let (device, queue) = init_test_wgpu();
        // Single-phase config for the test. pyramid_level: 0 means full-res
        // (no downsampling); for a 32×32 goal this is fine.
        let test_phases = vec![Phase {
            triangles: 6,
            pyramid_level: 0,
            initial_sigma: 0.1,
        }];
        let result = run_es(
            device,
            queue,
            goal,
            EsConfig {
                phases: test_phases,
                max_steps: 30,
                lambda: 4,
                snapshot_every: None,
            },
        );
        assert!(
            result.steps_run > 0,
            "ES loop must run at least one step"
        );
        // fitness_of returns usize in [0, 1_000_000] where HIGHER = better fit.
        assert!(
            result.final_fitness <= 1_000_000,
            "fitness out of expected range: {}",
            result.final_fitness
        );
        assert!(
            result.final_fitness >= result.initial_fitness,
            "fitness should not regress: initial={}, final={}",
            result.initial_fitness,
            result.final_fitness
        );
    }
}
```

If `block_on` isn't already imported at module scope (it is — see line 1: `use futures::executor::block_on;`), the test inherits it via `use super::*;`.

`GoalImage` has a field named `goal_image` based on the existing struct (verified in spec); if the field is private to the module, leave the test inside the same module so it can still construct one. (`#[cfg(test)] mod tests` is inside the same crate's module tree, so private fields are accessible.)

- [ ] **Step 2: Build the test target**

```bash
cargo build --tests --bin polygenvo
```

Expected: compiles.

**Do not commit yet.** Move to Task 5.

---

### Task 5: Verify Phase 1 baseline passes, then commit Phase 1

**Files:** none modified (verification + commit only)

- [ ] **Step 1: Run the smoke test on wgpu 0.12**

```bash
cargo test --bin polygenvo -- --nocapture
```

Expected: `ga_improves_on_synthetic_checker` passes. If it fails with `fitness should not regress`, the (1+λ)-ES may stall at very small lambda/sigma — try increasing `max_steps` to 60 or `lambda` to 6, but the assertion direction must remain `final >= initial` (higher = better).

If the test panics with a wgpu validation error or texture creation failure, the refactor in Task 3 likely missed something (e.g., didn't propagate `texture_size` correctly, or `build_pyramid` is being called with the wrong arguments). Fix Task 3 before continuing.

Note on `build_pyramid`: in the current code (line 659 of `main.rs`), `build_pyramid(&device, &queue, &goal_image)` returns a `Vec<FitnessCalc>` indexed by pyramid level. The test config uses `pyramid_level: 0`, which is the coarsest level (smallest texture) — for a 32×32 goal, level 0 is the 32×32 image itself if downsampling is no-op. Verify by reading `build_pyramid`'s body that this works for a 32×32 input. If `build_pyramid` requires the input to be a power of 2 ≥ some minimum, bump the test goal size accordingly (e.g., 64 or 128).

- [ ] **Step 2: Confirm release build still works**

```bash
cargo build --release --bin polygenvo
```

Expected: succeeds.

- [ ] **Step 3: Commit Phase 1**

```bash
git status
git add -A
git diff --cached --stat
git commit -m "$(cat <<'EOF'
refactor: drop dead code, bump rand 0.8 -> 0.10, add smoke test

- Delete unused binaries (genalgo, genevoalgo, polygen) and their dirs
- Delete dead .vert/.frag/.spv pipeline + build.rs (only WGSL used)
- Drop genevo dep (replaced by hand-rolled (1+λ)-ES in Tier 2)
- Bump rand 0.8 -> 0.10 (no longer constrained by genevo)
- Extract run_es() + EsConfig/EsResult so the ES is testable
- Add ga_improves_on_synthetic_checker smoke test on a 32x32 checker

Smoke test passes on wgpu 0.12; serves as the regression guard for
the upcoming wgpu 0.12 -> 29 migration.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Expected: one commit added to `wgpu-modernize`. Run `git log --oneline -3` to verify.

---

## Phase 2 — Atomic wgpu 0.12 → 29 migration

Tasks 6–8 share a single commit at the end (Task 8). Phase 2 is the actual wgpu migration; the smoke test from Phase 1 must pass on the new wgpu.

### Task 6: Rewrite both WGSL files to current syntax

**Files:**
- Modify: `src/polygenvo/shader.wgsl` — replace pre-1.0 syntax
- Modify: `src/polygenvo/fitness.wgsl` — replace pre-1.0 syntax

Syntax-only; no semantic changes. Both files become text inputs to `naga` once `wgpu` is bumped — they must parse with current naga.

- [ ] **Step 1: Replace the contents of `src/polygenvo/shader.wgsl`** with:

```wgsl
// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color);
}
```

- [ ] **Step 2: Replace the contents of `src/polygenvo/fitness.wgsl`** with:

```wgsl
// Compute shader for fitness scoring.
//
// One invocation per pixel. Each invocation reads the goal pixel and the
// rendered pixel as linear-RGB (textures are sRGB-formatted so the
// hardware does the decode on read), converts each to CIELAB, computes
// the ΔE76 perceptual distance, normalises and atomicAdds into the
// shared accumulator.
//
// Why CIELAB: ΔE76 distance in Lab space is approximately uniform in
// human perception. Summed-RGB diff over-weights bright colours and
// is blind to chroma vs. luminance imbalance. Same compute pattern,
// same single-u32 readback.

struct FitnessParams {
    image_width: u32,
    image_height: u32,
    pad0: u32,
    pad1: u32,
}

struct FitnessResult {
    value: atomic<u32>,
}

@group(0) @binding(0)
var<uniform> params: FitnessParams;

@group(0) @binding(1)
var goal_texture: texture_2d<f32>;

@group(0) @binding(2)
var rendered_texture: texture_2d<f32>;

@group(0) @binding(3)
var<storage, read_write> fitness_result: FitnessResult;

// Linear-RGB (sRGB primaries, D65) -> CIE XYZ
fn linear_rgb_to_xyz(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        rgb.r * 0.4124564 + rgb.g * 0.3575761 + rgb.b * 0.1804375,
        rgb.r * 0.2126729 + rgb.g * 0.7151522 + rgb.b * 0.0721750,
        rgb.r * 0.0193339 + rgb.g * 0.1191920 + rgb.b * 0.9503041
    );
}

// CIE XYZ (D65) -> CIELAB
fn xyz_to_lab(xyz: vec3<f32>) -> vec3<f32> {
    let xn = xyz.x / 0.95047;
    let yn = xyz.y / 1.00000;
    let zn = xyz.z / 1.08883;
    let fx = select((7.787 * xn) + (16.0 / 116.0), pow(xn, 1.0 / 3.0), xn > 0.008856);
    let fy = select((7.787 * yn) + (16.0 / 116.0), pow(yn, 1.0 / 3.0), yn > 0.008856);
    let fz = select((7.787 * zn) + (16.0 / 116.0), pow(zn, 1.0 / 3.0), zn > 0.008856);
    return vec3<f32>(
        116.0 * fy - 16.0,
        500.0 * (fx - fy),
        200.0 * (fy - fz)
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= params.image_width || y >= params.image_height) {
        return;
    }

    let goal_rgb = textureLoad(goal_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
    let rendered_rgb = textureLoad(rendered_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;

    let goal_lab = xyz_to_lab(linear_rgb_to_xyz(goal_rgb));
    let rendered_lab = xyz_to_lab(linear_rgb_to_xyz(rendered_rgb));

    let d = goal_lab - rendered_lab;
    let delta_e = sqrt(d.x * d.x + d.y * d.y + d.z * d.z);

    let normalized = clamp(delta_e / 250.0, 0.0, 1.0);
    atomicAdd(&fitness_result.value, u32(normalized * 1000.0));
}
```

Notes on the diff:
- `[[location(N)]]` → `@location(N)`, `[[builtin(X)]]` → `@builtin(X)`
- `[[stage(vertex)]]` / `[[stage(fragment)]]` / `[[stage(compute), workgroup_size(...)]]` → `@vertex` / `@fragment` / `@compute @workgroup_size(...)`
- `[[group(0), binding(0)]]` → `@group(0) @binding(0)`
- Struct field separators `;` → `,`
- Function return-position attribute: `-> [[location(0)]] T` → `-> @location(0) T`
- All algorithmic code unchanged

- [ ] **Step 3: Do not build yet.** `cargo build` will fail because wgpu hasn't been bumped — current wgpu 0.12 still expects the old WGSL syntax. The build is broken in this exact intermediate state. That's expected; proceed to Task 7.

**Do not commit yet.** Move to Task 7.

---

### Task 7: Bump `wgpu` to 29 and update every API call site

**Files:**
- Modify: `Cargo.toml` — bump `wgpu`, drop `spirv` feature
- Modify: `src/polygenvo/main.rs` — update every wgpu API call site to current API

This is the biggest task. Approach: bump `wgpu`, run `cargo build`, fix each compile error in turn. The error messages are precise and well-named. Don't try to anticipate every change up front — let `rustc` lead.

- [ ] **Step 1: Edit `Cargo.toml`**

Change:
```toml
wgpu = { version = "0.12.0", features = [ "spirv" ] }
```
to:
```toml
wgpu = "29"
```

The `spirv` feature is dropped because `main.rs` only uses `wgpu::ShaderSource::Wgsl(include_str!(...))` (no `ShaderSource::SpirV`).

- [ ] **Step 2: Run `cargo build --bin polygenvo` and fix errors in order**

```bash
cargo build --bin polygenvo 2>&1 | head -120
```

Apply the following recipes as errors point at each site. These are the migrations documented in the spec; the executor should also consult current wgpu docs at <https://docs.rs/wgpu/latest/wgpu/> when in doubt.

**Recipe — `Instance` construction.** Find the `wgpu::Instance::new(...)` call site. Replace:

```rust
let instance = wgpu::Instance::new(wgpu::Backends::all());
```

with:

```rust
let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
    backends: wgpu::Backends::all(),
    flags: wgpu::InstanceFlags::default(),
    backend_options: wgpu::BackendOptions::default(),
});
```

(Field names may have evolved — if `cargo build` complains about a missing field like `dx12_shader_compiler` or `gles_minor_version`, add it with `Default::default()`. If a field is unrecognized, remove it.)

**Recipe — `DeviceDescriptor` and `request_device`.** Find the `adapter.request_device(...)` call site. Replace:

```rust
let (device, queue) = adapter
    .request_device(
        &wgpu::DeviceDescriptor {
            label: Some("device"),
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
        },
        None,
    )
    .await
    .unwrap();
```

with:

```rust
let (device, queue) = adapter
    .request_device(&wgpu::DeviceDescriptor {
        label: Some("device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    })
    .await
    .unwrap();
```

The second argument (`trace_path: Option<&Path>`) is gone; `trace` is a field on the descriptor in 29.x. If `Trace::Off` is not the right variant name in 29.x, check the type definition.

**Recipe — `TextureDescriptor`.** Find all `wgpu::TextureDescriptor { ... }` literals (there are at least two: the fitness render target and the goal texture upload target). Add the field `view_formats: &[]` to each. Example:

```rust
let texture = device.create_texture(&wgpu::TextureDescriptor {
    label: Some("Fitness Render Target"),
    size: wgpu::Extent3d { width: texture_size, height: texture_size, depth_or_array_layers: 1 },
    mip_level_count: 1,
    sample_count: 1,
    dimension: wgpu::TextureDimension::D2,
    format: target_format,
    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC,
    view_formats: &[],  // ← new field
});
```

**Recipe — `ShaderModuleDescriptor`.** Find the two `wgpu::ShaderModuleDescriptor { ... }` literals (one for `shader.wgsl`, one for `fitness.wgsl`). The current code has a `flags` field; this field is removed in current wgpu. If the current code does not have `flags` (look at lines 129–132 and 222–225), no change needed here; only the way `create_shader_module` is called might differ. In wgpu 29, `create_shader_module` takes `&ShaderModuleDescriptor` directly:

```rust
let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("Render Shader"),
    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
});
```

Note: in some wgpu versions `create_shader_module` took `&Descriptor` (reference), in current it takes the descriptor by value. The compiler error will be unambiguous.

**Recipe — `RenderPipelineDescriptor`** (vertex/fragment states). The current code (line ~138–174) uses:

```rust
vertex: wgpu::VertexState {
    module: &render_shader,
    entry_point: "vs_main",
    buffers: &[Vertex::desc()],
},
fragment: Some(wgpu::FragmentState {
    module: &render_shader,
    entry_point: "fs_main",
    targets: &[wgpu::ColorTargetState { ... }],
}),
```

Update to:

```rust
vertex: wgpu::VertexState {
    module: &render_shader,
    entry_point: Some("vs_main"),
    compilation_options: wgpu::PipelineCompilationOptions::default(),
    buffers: &[Vertex::desc()],
},
fragment: Some(wgpu::FragmentState {
    module: &render_shader,
    entry_point: Some("fs_main"),
    compilation_options: wgpu::PipelineCompilationOptions::default(),
    targets: &[Some(wgpu::ColorTargetState { ... })],   // ← Option-wrapped
}),
```

Note: `entry_point` is `Option<&str>` in current wgpu (was `&str` in 0.12). `targets` items are `Option<ColorTargetState>` in current wgpu (were bare in 0.12).

Also add to the `RenderPipelineDescriptor` body (alongside `multiview: None`):

```rust
cache: None,
```

**Recipe — `ComputePipelineDescriptor`.** Find the compute pipeline creation. Add:

```rust
compilation_options: wgpu::PipelineCompilationOptions::default(),
cache: None,
```

`entry_point` becomes `Option<&str>`: `entry_point: Some("main")`.

**Recipe — `RenderPassDescriptor`.** Find `begin_render_pass(...)` call(s). Update `color_attachments` to be `Option`-wrapped if not already:

```rust
color_attachments: &[Some(wgpu::RenderPassColorAttachment { ... })],
```

Add the two new fields:

```rust
timestamp_writes: None,
occlusion_query_set: None,
```

**Recipe — `ComputePassDescriptor`.** Find `begin_compute_pass(...)`. Add:

```rust
timestamp_writes: None,
```

**Recipe — `device.poll(Maintain::Wait)`.** Two call sites in `main.rs` (around line 367 and 447). Replace each:

```rust
inner.device.poll(wgpu::Maintain::Wait);
```

with:

```rust
inner.device.poll(wgpu::PollType::Wait).unwrap();
```

(`Maintain` was renamed to `PollType` in wgpu 23+; `poll` now returns `Result<MaintainResult, PollError>`.)

**Recipe — `Buffer::slice(..).map_async(...)` future → callback.** This is a hard API break, not just a signature tweak. The current code (around lines 365–368 and 411–420) uses the wgpu 0.12 pattern where `map_async` returns a future:

```rust
// OLD (wgpu 0.12) — DELETE this:
let slice = inner.fitness_readback.slice(..);
let mapping = slice.map_async(wgpu::MapMode::Read);
inner.device.poll(wgpu::Maintain::Wait);
block_on(mapping).unwrap();
```

In current wgpu, `map_async` takes a callback and returns `()`. Replace each such block with the channel pattern:

```rust
// NEW — wgpu 29:
let slice = inner.fitness_readback.slice(..);
let (sender, receiver) = std::sync::mpsc::channel();
slice.map_async(wgpu::MapMode::Read, move |result| {
    sender.send(result).ok();
});
inner.device.poll(wgpu::PollType::Wait).unwrap();
receiver.recv().unwrap().unwrap();  // outer unwrap = channel; inner = map result
```

`futures::executor::block_on` is no longer needed for the readback path. Remove the `block_on(mapping)` call. The `use futures::executor::block_on;` import is still needed for the `block_on(instance.request_adapter(...))` and `block_on(adapter.request_device(...))` calls in the wgpu-init helper.

**Recipe — `Operations.store: bool` → `StoreOp` enum.** Find every `wgpu::Operations { load: ..., store: true }` (one instance in the render pass at line 333–336). Replace:

```rust
ops: wgpu::Operations {
    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
    store: true,
},
```

with:

```rust
ops: wgpu::Operations {
    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
    store: wgpu::StoreOp::Store,
},
```

**Recipe — `compute_pass.dispatch(x, y, z)` → `dispatch_workgroups(x, y, z)`.** Find the dispatch call (line 352). Rename:

```rust
// OLD:
compute_pass.dispatch(wg, wg, 1);
// NEW:
compute_pass.dispatch_workgroups(wg, wg, 1);
```

Arguments are unchanged.

**Recipe — `BindGroupLayoutEntry` `BufferBindingType::Uniform`.** The variant may have simplified. If the compiler complains about a `dynamic` field, remove it. Stable form in 29:

```rust
ty: wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Uniform,
    has_dynamic_offset: false,
    min_binding_size: None,
},
```

**Recipe — texture-format / sRGB sample binding.** If pipeline creation or shader binding fails with a complaint about sampling `Rgba8UnormSrgb` as `texture_2d<f32>`, add to the render-target `TextureDescriptor`:

```rust
view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
```

If that still fails, the compute binding likely needs an `Rgba8Unorm` view of the sRGB texture. Add `Rgba8Unorm` to `view_formats` and create the compute-side view with `format: Some(wgpu::TextureFormat::Rgba8Unorm)`. Only do this if the simpler fix above doesn't work.

- [ ] **Step 3: Iterate until `cargo build --bin polygenvo` succeeds**

```bash
cargo build --bin polygenvo
```

Expected: succeeds. If there are residual errors not covered by the recipes above, consult `https://docs.rs/wgpu/29/wgpu/` and apply the analogous fix.

**Do not commit yet.** Move to Task 8.

---

### Task 8: Verify smoke test + manual run, then commit Phase 2

**Files:** none modified (verification + commit only)

- [ ] **Step 1: Run the smoke test on the new wgpu**

```bash
cargo test --bin polygenvo -- --nocapture
```

Expected: `ga_improves_on_synthetic_checker` passes. If it fails:
- `pipeline creation failed: ...` — usually a bind-group / shader signature mismatch. Re-read the WGSL bindings and compare to the Rust-side `BindGroupLayoutEntry` array.
- `validation error: ...` — usually a missing usage flag (`TextureUsages::COPY_SRC`, `STORAGE_BINDING`, etc.) or a format mismatch. Add the missing flag.
- `Texture view formats: ...` — apply the texture-format watch-out recipe from Task 7.
- `final_fitness > initial_fitness` — algorithmic regression. Re-check Task 3's extraction: did `cfg.lambda` and `cfg.initial_sigma` reach the actual ES inner loop, or are constants still being used somewhere?

- [ ] **Step 2: Confirm release build**

```bash
cargo build --release --bin polygenvo
```

Expected: succeeds, no warnings beyond pre-existing dead-code ones.

- [ ] **Step 3: Manual run with `goal.png`**

```bash
ls goal.png triangles/ 2>&1
cargo run --release --bin polygenvo 2>&1 | head -30
```

Let it run for ~30 seconds. Expected: prints fitness numbers that decrease over steps; writes some `triangles/imageN.png` snapshots. Kill with Ctrl-C. Open one of the recent snapshots and confirm it looks like a triangle-approximation in progress (not all-black, not random noise).

If the snapshot looks broken (all black, all white, single-color block), investigate before committing. Likely causes: wrong texture-format conversion, bind-group binding mismatch, or sRGB encoding mismatch.

- [ ] **Step 4: Commit Phase 2**

```bash
git status
git add -A
git diff --cached --stat
git commit -m "$(cat <<'EOF'
deps: migrate wgpu 0.12 -> 29 + WGSL to current syntax

Atomic migration of the only consumer (polygenvo) and its two WGSL
files to current wgpu APIs and current naga-compliant WGSL.

WGSL: pre-1.0 [[attribute]] syntax -> @attribute, struct field ; -> ,
wgpu: InstanceDescriptor, DeviceDescriptor required_*/memory_hints,
TextureDescriptor view_formats, RenderPipelineDescriptor compilation_
options/cache, Option-wrapped color_attachments, timestamp_writes,
occlusion_query_set, Maintain -> PollType, drop spirv feature.

Smoke test (ga_improves_on_synthetic_checker) added in the previous
commit passes on the new wgpu, confirming the algorithm is preserved
across the API migration.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Expected: a second commit on `wgpu-modernize`. Run `git log --oneline -3` to verify two new commits (Phase 1 + Phase 2) on top of `09b69d9` (the spec).

---

## Post-merge follow-up (separate, non-blocking)

Refresh `CLAUDE.md` to reflect the post-Tier-1/2 + post-modernization state. This is **not** part of this plan — it's a tracked follow-up:

- Line count of `polygenvo/main.rs` (~803, not 1300)
- `genevo` is no longer a dependency; the GA uses a hand-rolled (1+λ)-ES
- CPU fitness fallback no longer exists (removed in Tier 1)
- Fitness shader is CIELAB ΔE76 (not multi-scale SSIM)
- `build.rs` and the `.vert/.frag` pipeline no longer exist; shaders are WGSL-only via `include_str!`
- Only one binary remains (`polygenvo`)
- wgpu version reference updated to current
- WGSL syntax reference updated (no longer pre-1.0)

Land this as a separate commit either on `wgpu-modernize` before merging or as a follow-up on `master`. Author's call at merge time.
