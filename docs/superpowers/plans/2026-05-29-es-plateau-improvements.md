# ES Plateau Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break the "improves then crawls" plateau in the `polygenvo` (1+λ)-ES by batching GPU evaluation, decoupling Gaussian self-adaptive step sizes, guiding triangle placement by residual error, and accumulating fitness at finer precision.

**Architecture:** The (1+λ)-ES structure, phase schedule, goal pyramid, and selection logic in [src/polygenvo/main.rs](../../../src/polygenvo/main.rs) are preserved. Four changes land underneath them: (1) `FitnessCalc` evaluates all λ candidates in one GPU submit/readback via one bind group per output slot; (2) the fitness compute shader reduces per-workgroup before a single `atomicAdd` at scale 8192 and bins a 16×16 error grid; (3) `mutate` uses two Gaussian step sizes (`σ_pos`, `σ_col`) each adapted by its own 1/5 success rule; (4) `add`/`relocate` operators sample high-error grid cells.

**Tech Stack:** Rust (edition 2024), `wgpu = "29"`, WGSL compute/render shaders, `rand = "0.10"`, `bytemuck`, `image = "0.25"`.

---

## Conventions for every task

- Work on branch `es-plateau-improvements` (already created; spec committed there).
- Build/test the GPU binary with `cargo test --bin polygenvo` — it needs a working wgpu adapter on the host (same as the existing smoke test).
- Every commit message ends with the trailer:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
- The smoke test `ga_improves_on_synthetic_checker` is the regression guard. It must pass at the end of every task.
- Fitness direction is **higher = better**; comparisons use `>` for improvement.

---

## Task 1: Box-Muller Gaussian helper

Pure-Rust, no GPU. Provides `N(0,σ)` sampling used by `mutate` in Task 3. Isolated and unit-testable first.

**Files:**
- Modify: `src/polygenvo/main.rs` (add `gaussian` fn near the mutation helpers, ~line 600; add test in the existing `mod tests`)

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `src/polygenvo/main.rs` (after the existing `use` lines in that module):

```rust
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn gaussian_has_zero_mean_and_unit_std() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 100_000;
        let samples: Vec<f32> = (0..n).map(|_| gaussian(&mut rng, 1.0)).collect();
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let var: f32 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std = var.sqrt();
        assert!(mean.abs() < 0.05, "mean {mean} not ~0");
        assert!((std - 1.0).abs() < 0.05, "std {std} not ~1");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --bin polygenvo gaussian_has_zero_mean_and_unit_std`
Expected: compile error — `cannot find function gaussian in this scope`.

- [ ] **Step 3: Write minimal implementation**

Add above `fn mutate(` in `src/polygenvo/main.rs`:

```rust
/// One sample from N(0, sigma) via the Box-Muller transform. `rand 0.10` ships no
/// normal distribution and we avoid adding `rand_distr`, so we derive it from two
/// uniforms. Gaussian (vs the previous uniform) perturbations give the ES both
/// fine refinement (most mass near 0) and an occasional larger exploratory jump
/// (the tail) — the previous uniform jitter had no tail.
fn gaussian(rng: &mut impl Rng, sigma: f32) -> f32 {
    let u1: f32 = rng.random_range(1e-7_f32..1.0); // avoid ln(0)
    let u2: f32 = rng.random_range(0.0_f32..1.0);
    let mag = (-2.0 * u1.ln()).sqrt();
    mag * (std::f32::consts::TAU * u2).cos() * sigma
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --bin polygenvo gaussian_has_zero_mean_and_unit_std`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/polygenvo/main.rs
git commit -m "feat: add Box-Muller gaussian sampler for ES mutation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Batched GPU evaluation + finer fitness + error grid

The largest structural change, and the throughput lever. These three concerns share the same WGSL `main()`, bind group, and readback layout, so they land together to avoid rewriting those structures three times. Behaviourally the selection is identical to today — only evaluation is batched.

**Files:**
- Modify: `src/polygenvo/fitness.wgsl` (rewrite `FitnessParams`, add per-slot result struct + workgroup reduction + grid binning)
- Modify: `src/polygenvo/main.rs` (new constants; `Eval` struct; `FitnessCalcInner` fields; `FitnessCalc::new`; `fitness_of_batch`; `fitness_of` becomes a wrapper; batch the λ loop in `run_es`)

### 2a — New constants and `Eval`

- [ ] **Step 1: Add constants near the top of `main.rs`**

After `const MAX_STEPS: u64 = 500_000;` (line ~32) add:

```rust
// Per-pixel ΔE accumulator scale. Bounded by u32: largest pyramid level is
// 512² = 262144 px, and 262144 * FITNESS_SCALE must stay < 2^32, so the safe
// ceiling is ~16384. 8192 is 8× finer than the previous 1000 with headroom.
// Passed to the shader via the params uniform so the Rust normaliser and the
// shader share one source of truth.
const FITNESS_SCALE: u32 = 8192;

// Coarse residual-error grid emitted by the fitness pass for error-guided
// placement. MUST equal `GRID_DIM` in fitness.wgsl (WGSL array sizes must be
// compile-time constants, so the value is mirrored rather than passed).
const ERROR_GRID_DIM: u32 = 16;
const GRID_CELLS: usize = (ERROR_GRID_DIM * ERROR_GRID_DIM) as usize; // 256

// Per-candidate GPU output: one score u32 + GRID_CELLS grid u32. Storage-buffer
// binding offsets must be 256-aligned, so each slot is padded to SLOT_STRIDE.
const SLOT_PAYLOAD: u64 = 4 + (GRID_CELLS as u64) * 4; // 1028 bytes
const SLOT_STRIDE: u64 = SLOT_PAYLOAD.div_ceil(256) * 256; // 1280 bytes
```

- [ ] **Step 2: Add `use std::num::NonZeroU64;` to the imports**

At the top of `main.rs`, alongside the other `use std::...` lines:

```rust
use std::num::NonZeroU64;
```

- [ ] **Step 3: Add the `Eval` struct**

Above `struct FitnessCalcInner {` (line ~80) add:

```rust
/// Result of scoring one candidate: the similarity score in [0, 1_000_000]
/// (higher = better) plus the coarse residual-error grid (length GRID_CELLS,
/// row-major, cell row 0 = top of the image) used to guide triangle placement.
#[derive(Clone, Debug)]
pub struct Eval {
    pub score: usize,
    pub error_grid: Vec<u32>,
}
```

### 2b — Rewrite the compute shader

- [ ] **Step 4: Rewrite `src/polygenvo/fitness.wgsl`**

Replace the whole file with:

```wgsl
// Compute shader for fitness scoring + residual-error binning.
//
// One invocation per pixel. Each invocation reads the goal and rendered pixels
// as linear-RGB (textures are sRGB-formatted so the hardware decodes on read),
// converts each to CIELAB, and takes the ΔE76 perceptual distance. The
// normalised per-pixel error is reduced within each 8×8 workgroup in shared
// memory and added to the score accumulator with a single atomicAdd per
// workgroup (truncation once per 64 px instead of once per px). Each pixel also
// bins its error into a GRID_DIM×GRID_DIM grid for error-guided placement.

const GRID_DIM: u32 = 16u;        // MUST match ERROR_GRID_DIM in main.rs
const GRID_CELLS: u32 = 256u;     // GRID_DIM * GRID_DIM
const GRID_SCALE: f32 = 1000.0;   // grid magnitudes are used only relatively
const WG_PIXELS: u32 = 64u;       // workgroup_size 8*8

struct FitnessParams {
    image_width: u32,
    image_height: u32,
    scale: u32,   // FITNESS_SCALE
    pad1: u32,
}

struct SlotResult {
    score: atomic<u32>,
    grid: array<atomic<u32>, GRID_CELLS>,
}

@group(0) @binding(0)
var<uniform> params: FitnessParams;

@group(0) @binding(1)
var goal_texture: texture_2d<f32>;

@group(0) @binding(2)
var rendered_texture: texture_2d<f32>;

@group(0) @binding(3)
var<storage, read_write> result: SlotResult;

var<workgroup> partials: array<f32, WG_PIXELS>;

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
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let x = global_id.x;
    let y = global_id.y;
    let in_bounds = x < params.image_width && y < params.image_height;

    var normalized = 0.0;
    if (in_bounds) {
        let goal_rgb = textureLoad(goal_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
        let rendered_rgb = textureLoad(rendered_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
        let goal_lab = xyz_to_lab(linear_rgb_to_xyz(goal_rgb));
        let rendered_lab = xyz_to_lab(linear_rgb_to_xyz(rendered_rgb));
        let d = goal_lab - rendered_lab;
        let delta_e = sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        // ΔE76 between primary-saturated colours peaks ~230; normalise by 250
        // into [0,1].
        normalized = clamp(delta_e / 250.0, 0.0, 1.0);

        // Bin into the coarse error grid (cell row 0 = top of image).
        let gx = (x * GRID_DIM) / params.image_width;
        let gy = (y * GRID_DIM) / params.image_height;
        let cell = gy * GRID_DIM + gx;
        atomicAdd(&result.grid[cell], u32(normalized * GRID_SCALE));
    }

    // Workgroup reduction: sum the 64 normalised values, one atomicAdd by lane 0.
    partials[lid] = normalized;
    workgroupBarrier();
    if (lid == 0u) {
        var sum = 0.0;
        for (var i = 0u; i < WG_PIXELS; i = i + 1u) {
            sum = sum + partials[i];
        }
        atomicAdd(&result.score, u32(sum * f32(params.scale)));
    }
}
```

### 2c — Rewrite `FitnessCalc` for batched evaluation

- [ ] **Step 5: Change `FitnessCalcInner` fields**

In `struct FitnessCalcInner` (line ~80) replace the vertex/compute/fitness fields. Replace:

```rust
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: u64,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    fitness_buffer: wgpu::Buffer,
    fitness_readback: wgpu::Buffer,
```

with:

```rust
    vertex_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    // One bind group per output slot; binding 3 is offset into result_buffer.
    slot_bind_groups: Vec<wgpu::BindGroup>,
    result_buffer: wgpu::Buffer,
    result_readback: wgpu::Buffer,
```

- [ ] **Step 6: Update the vertex buffer + result buffers + bind groups in `FitnessCalc::new`**

In `FitnessCalc::new`, replace the vertex-buffer block (lines ~179-187):

```rust
        // Genome size is constant per run; MAX_VERTICES gives headroom for any
        // future growth phase. Filled per call via queue.write_buffer.
        let vertex_capacity = (MAX_VERTICES as u64) * std::mem::size_of::<Vertex>() as u64;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: vertex_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
```

with (vertex buffer now holds LAMBDA candidates back-to-back):

```rust
        // Vertex buffer holds LAMBDA candidates back-to-back; candidate i lives at
        // byte offset i * MAX_VERTICES * sizeof(Vertex). sizeof(Vertex) is 28 (4-
        // aligned), so every per-candidate offset is a legal vertex-buffer offset.
        let per_candidate_bytes = (MAX_VERTICES as u64) * std::mem::size_of::<Vertex>() as u64;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: per_candidate_bytes * LAMBDA as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
```

Then replace the params/fitness buffer block (lines ~239-257):

```rust
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fitness Params"),
            contents: bytemuck::cast_slice(&[texture_size, texture_size, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let fitness_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fitness Accumulator"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fitness_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fitness Readback"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
```

with:

```rust
        // scale travels in params so the Rust normaliser and shader agree.
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fitness Params"),
            contents: bytemuck::cast_slice(&[texture_size, texture_size, FITNESS_SCALE, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // One result buffer holding LAMBDA slots of (score u32 + grid u32[GRID_CELLS]),
        // each slot padded to SLOT_STRIDE for storage-binding offset alignment.
        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fitness Results"),
            size: SLOT_STRIDE * LAMBDA as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let result_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fitness Readback"),
            size: SLOT_STRIDE * LAMBDA as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
```

- [ ] **Step 7: Replace the single bind group with one bind group per slot**

Replace the bind-group block (lines ~259-289):

```rust
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fitness Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&goal_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &fitness_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
```

with:

```rust
        // One bind group per output slot. Bindings 0–2 (params, goal, rendered
        // target) are shared; binding 3 is the result buffer offset to slot i.
        // The shader always writes to "its" SlotResult at element 0 — the binding
        // offset selects the slot, so no per-dispatch slot index is needed.
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let slot_size = NonZeroU64::new(SLOT_PAYLOAD).unwrap();
        let slot_bind_groups: Vec<wgpu::BindGroup> = (0..LAMBDA)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Fitness Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &params_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&goal_texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &result_buffer,
                                offset: i as u64 * SLOT_STRIDE,
                                size: Some(slot_size),
                            }),
                        },
                    ],
                })
            })
            .collect();
```

- [ ] **Step 8: Update the `FitnessCalc { inner: Arc::new(FitnessCalcInner { ... }) }` constructor tail**

Replace the field list (lines ~291-306) to match the new fields:

```rust
        FitnessCalc {
            inner: Arc::new(FitnessCalcInner {
                device,
                queue,
                texture_size,
                render_pipeline,
                texture,
                texture_view,
                vertex_buffer,
                compute_pipeline,
                slot_bind_groups,
                result_buffer,
                result_readback,
            }),
        }
```

- [ ] **Step 9: Replace `fitness_of` with `fitness_of_batch` + a thin wrapper**

Replace the entire `fn fitness_of(&self, vertices: &[Vertex]) -> usize { ... }` (lines ~310-393) with:

```rust
    /// Score `batch` candidates in a single GPU submit + readback. For each
    /// candidate i: render it into the shared target, then run the compute pass
    /// to write slot i. Within one command buffer, passes execute in order with
    /// automatic barriers, so reusing one render target across candidates is
    /// safe. Returns one `Eval` per candidate. `batch.len()` must be ≤ LAMBDA.
    fn fitness_of_batch(&self, batch: &[&[Vertex]]) -> Vec<Eval> {
        let inner = &*self.inner;
        assert!(
            batch.len() <= LAMBDA,
            "batch of {} exceeds LAMBDA {}",
            batch.len(),
            LAMBDA
        );
        let per_candidate_bytes = (MAX_VERTICES as u64) * std::mem::size_of::<Vertex>() as u64;

        // Upload all candidate vertices; zero the whole result buffer.
        for (i, verts) in batch.iter().enumerate() {
            assert!(
                verts.len() <= MAX_VERTICES,
                "genome of {} vertices exceeds MAX_VERTICES {}",
                verts.len(),
                MAX_VERTICES
            );
            let bytes: &[u8] = bytemuck::cast_slice(verts);
            inner
                .queue
                .write_buffer(&inner.vertex_buffer, i as u64 * per_candidate_bytes, bytes);
        }
        let zeros = vec![0u8; (SLOT_STRIDE * LAMBDA as u64) as usize];
        inner.queue.write_buffer(&inner.result_buffer, 0, &zeros);

        let mut encoder = inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Fitness Encoder"),
            });

        for (i, verts) in batch.iter().enumerate() {
            let num_vertices = verts.len() as u32;
            let vb_offset = i as u64 * per_candidate_bytes;
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Fitness Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &inner.texture_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                render_pass.set_pipeline(&inner.render_pipeline);
                render_pass.set_vertex_buffer(0, inner.vertex_buffer.slice(vb_offset..));
                render_pass.draw(0..num_vertices, 0..1);
            }
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Fitness Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&inner.compute_pipeline);
                compute_pass.set_bind_group(0, &inner.slot_bind_groups[i], &[]);
                let wg = (inner.texture_size + 7) / 8;
                compute_pass.dispatch_workgroups(wg, wg, 1);
            }
        }

        encoder.copy_buffer_to_buffer(
            &inner.result_buffer,
            0,
            &inner.result_readback,
            0,
            SLOT_STRIDE * LAMBDA as u64,
        );
        inner.queue.submit(iter::once(encoder.finish()));

        let slice = inner.result_readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        inner.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let max_total = (inner.texture_size as f64).powi(2) * FITNESS_SCALE as f64;
        let evals = {
            let data = slice.get_mapped_range();
            batch
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let base = (i as u64 * SLOT_STRIDE) as usize;
                    let raw = u32::from_le_bytes([
                        data[base],
                        data[base + 1],
                        data[base + 2],
                        data[base + 3],
                    ]);
                    let similarity = (1.0 - raw as f64 / max_total).max(0.0);
                    let score = (similarity * 1_000_000.0) as usize;
                    let grid: Vec<u32> = (0..GRID_CELLS)
                        .map(|c| {
                            let o = base + 4 + c * 4;
                            u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]])
                        })
                        .collect();
                    Eval {
                        score,
                        error_grid: grid,
                    }
                })
                .collect::<Vec<_>>()
        };
        inner.result_readback.unmap();
        evals
    }

    /// Score a single candidate. Thin wrapper over `fitness_of_batch`. After
    /// Task 4 `run_es` scores via the batch path directly, so this is exercised
    /// mainly by tests — `allow(dead_code)` keeps non-test builds warning-free.
    #[allow(dead_code)]
    fn fitness_of(&self, vertices: &[Vertex]) -> usize {
        self.fitness_of_batch(&[vertices])[0].score
    }
```

### 2d — Batch the λ loop in `run_es`

- [ ] **Step 10: Replace the candidate-generation loop in `run_es`**

Replace the `// (1+λ): produce λ candidates...` block (lines ~777-789):

```rust
        // (1+λ): produce λ candidates, keep the best if it beats the parent.
        let mut best_idx: Option<usize> = None;
        let mut best_fit = current_fitness;
        let mut candidates: Vec<Vec<Vertex>> = Vec::with_capacity(cfg.lambda);
        for _ in 0..cfg.lambda {
            let c = mutate(&current, sigma, min_triangles, max_triangles, &goal, &mut rng);
            let f = calc.fitness_of(&c);
            if f > best_fit {
                best_fit = f;
                best_idx = Some(candidates.len());
            }
            candidates.push(c);
        }
```

with:

```rust
        // (1+λ): produce λ candidates and evaluate them all in one GPU submit.
        let mut candidates: Vec<Vec<Vertex>> = Vec::with_capacity(cfg.lambda);
        for _ in 0..cfg.lambda {
            candidates.push(mutate(&current, sigma, min_triangles, max_triangles, &goal, &mut rng));
        }
        let cand_refs: Vec<&[Vertex]> = candidates.iter().map(|c| c.as_slice()).collect();
        let evals = calc.fitness_of_batch(&cand_refs);
        let mut best_idx: Option<usize> = None;
        let mut best_fit = current_fitness;
        for (i, e) in evals.iter().enumerate() {
            if e.score > best_fit {
                best_fit = e.score;
                best_idx = Some(i);
            }
        }
```

### 2e — Tests

- [ ] **Step 11: Add batch-consistency and grid-binning tests**

Add to `mod tests` in `src/polygenvo/main.rs`:

```rust
    fn make_solid_goal(size: u32, rgb: [u8; 3]) -> GoalImage {
        let mut buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(size, size);
        for y in 0..size {
            for x in 0..size {
                buf.put_pixel(x, y, Rgba([rgb[0], rgb[1], rgb[2], 255]));
            }
        }
        GoalImage { goal_image: buf }
    }

    #[test]
    fn batch_scores_match_single() {
        let goal = make_checker_goal(32);
        let (device, queue) = init_test_wgpu();
        let calc = FitnessCalc::new(device, queue, &goal);
        let mut rng = StdRng::seed_from_u64(7);
        let g = init_genome(&goal, 6, &mut rng);
        let single = calc.fitness_of(&g);
        let batch = calc.fitness_of_batch(&[g.as_slice(), g.as_slice(), g.as_slice()]);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].score, single);
        assert_eq!(batch[1].score, single);
        assert_eq!(batch[2].score, single);
        assert_eq!(batch[0].error_grid.len(), GRID_CELLS);
    }

    #[test]
    fn error_grid_tracks_residual() {
        // Empty genome renders black. Against a white goal every cell has large
        // error; against a black goal the error is ~0.
        let (device, queue) = init_test_wgpu();
        let white = make_solid_goal(32, [255, 255, 255]);
        let black = make_solid_goal(32, [0, 0, 0]);
        let calc_white = FitnessCalc::new(device.clone(), queue.clone(), &white);
        let calc_black = FitnessCalc::new(device, queue, &black);
        let empty: Vec<Vertex> = Vec::new();
        let ew = &calc_white.fitness_of_batch(&[empty.as_slice()])[0];
        let eb = &calc_black.fitness_of_batch(&[empty.as_slice()])[0];
        let sum_white: u64 = ew.error_grid.iter().map(|&w| w as u64).sum();
        let sum_black: u64 = eb.error_grid.iter().map(|&w| w as u64).sum();
        assert!(ew.error_grid.iter().all(|&w| w > 0), "white goal: every cell should have error");
        assert!(sum_white > sum_black * 10, "white {sum_white} should dwarf black {sum_black}");
    }
```

- [ ] **Step 12: Run the GPU tests + smoke test**

Run: `cargo test --bin polygenvo`
Expected: PASS — `gaussian_has_zero_mean_and_unit_std`, `batch_scores_match_single`, `error_grid_tracks_residual`, and `ga_improves_on_synthetic_checker` all pass.

- [ ] **Step 13: Commit**

```bash
git add src/polygenvo/fitness.wgsl src/polygenvo/main.rs
git commit -m "feat: batched GPU evaluation, finer fitness, error grid

Score all LAMBDA candidates in one submit/readback via one bind group
per output slot. Workgroup-reduce before a single atomicAdd at scale
8192. Emit a 16x16 residual-error grid per candidate.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Decoupled Gaussian self-adaptive step sizes

Split the single `sigma` into `σ_pos` (vertex positions) and `σ_col` (colour + alpha), make perturbations Gaussian, and adapt each by its own 1/5 success rule using a per-category "beat the parent" rate.

**Files:**
- Modify: `src/polygenvo/main.rs` (`OpKind` enum; per-type sigma clamps; `Phase` struct + `PHASES`; `mutate` signature/body; `run_es` sigma state, classification, adaptation, restart, logs; smoke-test `Phase`)

- [ ] **Step 1: Add `OpKind` and per-type sigma clamp constants**

After the `GRID_CELLS` / `SLOT_STRIDE` constants block, add:

```rust
// Per-type self-adapted step-size clamps. Position lives in clip-space [-1,1];
// colour/alpha in [0,1], so they get independent ranges.
const SIGMA_POS_MIN: f32 = 0.005;
const SIGMA_POS_MAX: f32 = 0.5;
const SIGMA_COL_MIN: f32 = 0.003;
const SIGMA_COL_MAX: f32 = 0.4;

/// Which step size a mutation exercises, for per-type 1/5-rule adaptation.
#[derive(Copy, Clone, Debug, PartialEq)]
enum OpKind {
    Positional, // vertex nudge -> sigma_pos
    Chromatic,  // recolour / alpha -> sigma_col
    Structural, // add / delete / z-swap / relocate -> no step size
}
```

- [ ] **Step 2: Split the `Phase` struct and `PHASES`**

Replace the `Phase` struct (lines ~490-496):

```rust
#[derive(Clone)]
pub struct Phase {
    triangles: usize,
    pyramid_level: usize,
    // Initial sigma for this phase. Self-adapted by the 1/5 rule from here.
    initial_sigma: f32,
}
```

with:

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

Replace `PHASES` (lines ~498-503):

```rust
const PHASES: &[Phase] = &[
    Phase { triangles: 40,  pyramid_level: 0, initial_sigma: 0.25 },  // 128² coarse
    Phase { triangles: 80,  pyramid_level: 1, initial_sigma: 0.15 },  // 256² medium
    Phase { triangles: 120, pyramid_level: 2, initial_sigma: 0.10 },  // 512² fine
    Phase { triangles: 150, pyramid_level: 2, initial_sigma: 0.05 },  // 512² finer
];
```

with:

```rust
const PHASES: &[Phase] = &[
    Phase { triangles: 40,  pyramid_level: 0, initial_sigma_pos: 0.30, initial_sigma_col: 0.20 }, // 128² coarse
    Phase { triangles: 80,  pyramid_level: 1, initial_sigma_pos: 0.18, initial_sigma_col: 0.12 }, // 256² medium
    Phase { triangles: 120, pyramid_level: 2, initial_sigma_pos: 0.10, initial_sigma_col: 0.08 }, // 512² fine
    Phase { triangles: 150, pyramid_level: 2, initial_sigma_pos: 0.05, initial_sigma_col: 0.04 }, // 512² finer
];
```

- [ ] **Step 3: Rewrite `mutate` to take two sigmas, use Gaussian, and return `OpKind`**

Replace the whole `fn mutate(...) -> Vec<Vertex> { ... }` (lines ~604-676) with:

```rust
/// Apply one random mutation to a clone of `parent`, returning the child and the
/// `OpKind` it exercised (for per-type step-size adaptation). Positional nudges
/// use `sigma_pos`; recolour/alpha use `sigma_col`; both are Gaussian. Structural
/// changes (add/delete/z-swap) happen rarely and carry no step size.
fn mutate(
    parent: &[Vertex],
    sigma_pos: f32,
    sigma_col: f32,
    min_triangles: usize,
    max_triangles: usize,
    goal: &GoalImage,
    rng: &mut impl Rng,
) -> (Vec<Vertex>, OpKind) {
    let mut child = parent.to_vec();
    let n = child.len() / 3;
    if n == 0 {
        // Pathological: rebuild from scratch.
        return (init_genome(goal, min_triangles, rng), OpKind::Structural);
    }

    let op = rng.random_range(0u32..100);
    let kind = match op {
        0..=39 => {
            // Nudge a single vertex of one triangle (Gaussian, sigma_pos).
            let t = rng.random_range(0..n);
            let v = rng.random_range(0..3);
            let idx = t * 3 + v;
            child[idx].position[0] = (child[idx].position[0] + gaussian(rng, sigma_pos)).clamp(-1.0, 1.0);
            child[idx].position[1] = (child[idx].position[1] + gaussian(rng, sigma_pos)).clamp(-1.0, 1.0);
            OpKind::Positional
        }
        40..=64 => {
            // Recolour all three vertices of one triangle (RGB, Gaussian, sigma_col).
            let t = rng.random_range(0..n);
            let dr = gaussian(rng, sigma_col);
            let dg = gaussian(rng, sigma_col);
            let db = gaussian(rng, sigma_col);
            for v in 0..3 {
                let c = &mut child[t * 3 + v].color;
                c[0] = (c[0] + dr).clamp(0.0, 1.0);
                c[1] = (c[1] + dg).clamp(0.0, 1.0);
                c[2] = (c[2] + db).clamp(0.0, 1.0);
            }
            OpKind::Chromatic
        }
        65..=79 => {
            // Nudge the alpha of one triangle (Gaussian, sigma_col).
            let t = rng.random_range(0..n);
            let da = gaussian(rng, sigma_col);
            for v in 0..3 {
                let a = &mut child[t * 3 + v].color[3];
                *a = (*a + da).clamp(0.0, 1.0);
            }
            OpKind::Chromatic
        }
        80..=89 => {
            // Swap z-order with a neighbouring triangle.
            if n > 1 {
                let t = rng.random_range(0..n - 1);
                for v in 0..3 {
                    child.swap(t * 3 + v, (t + 1) * 3 + v);
                }
            }
            OpKind::Structural
        }
        90..=94 => {
            // Add a new colour-seeded triangle at a random z position.
            if n < max_triangles {
                let tri = random_color_seeded_triangle(goal, rng, 0.2);
                let insert_at = rng.random_range(0..=n) * 3;
                for (offset, vert) in tri.iter().enumerate() {
                    child.insert(insert_at + offset, *vert);
                }
            }
            OpKind::Structural
        }
        _ => {
            // Delete one triangle.
            if n > min_triangles {
                let t = rng.random_range(0..n);
                for _ in 0..3 {
                    child.remove(t * 3);
                }
            }
            OpKind::Structural
        }
    };
    (child, kind)
}
```

- [ ] **Step 4: Update `run_es` sigma initialisation**

Replace (lines ~736-737):

```rust
    let mut current = init_genome(&goal, cfg.phases[phase_idx].triangles, &mut rng);
    let mut sigma = cfg.phases[phase_idx].initial_sigma;
```

with:

```rust
    let mut current = init_genome(&goal, cfg.phases[phase_idx].triangles, &mut rng);
    let mut sigma_pos = cfg.phases[phase_idx].initial_sigma_pos;
    let mut sigma_col = cfg.phases[phase_idx].initial_sigma_col;
```

- [ ] **Step 5: Update the starting-phase log line**

Replace the first `println!("Phase {} | ... σ={:.3} ...")` (lines ~741-749) with:

```rust
    println!(
        "Phase {} | {} triangles | level {} ({}²) | σ_pos={:.3} σ_col={:.3} | starting fitness {}",
        phase_idx,
        cfg.phases[phase_idx].triangles,
        cfg.phases[phase_idx].pyramid_level,
        pyramid[cfg.phases[phase_idx].pyramid_level].inner.texture_size,
        sigma_pos,
        sigma_col,
        current_fitness
    );
```

- [ ] **Step 6: Replace the per-window accept counters with per-type counters**

Replace (lines ~753-755):

```rust
    let mut accepts_in_sigma_window: u64 = 0;
    let mut steps_in_sigma_window: u64 = 0;
```

with:

```rust
    // Per-type 1/5 rule: count candidates generated and how many beat the parent,
    // separately for positional and chromatic mutations, over SIGMA_WINDOW steps.
    let mut steps_in_sigma_window: u64 = 0;
    let mut pos_gen: u64 = 0;
    let mut pos_better: u64 = 0;
    let mut col_gen: u64 = 0;
    let mut col_better: u64 = 0;
```

- [ ] **Step 7: Generate candidates with two sigmas and classify them**

Replace the candidate-generation + selection block written in Task 2 Step 10 (the `let mut candidates ... best_idx = Some(i)` block) with:

```rust
        // (1+λ): produce λ candidates and evaluate them all in one GPU submit.
        let mut candidates: Vec<Vec<Vertex>> = Vec::with_capacity(cfg.lambda);
        let mut kinds: Vec<OpKind> = Vec::with_capacity(cfg.lambda);
        for _ in 0..cfg.lambda {
            let (child, kind) = mutate(
                &current, sigma_pos, sigma_col, min_triangles, max_triangles, &goal, &mut rng,
            );
            candidates.push(child);
            kinds.push(kind);
        }
        let cand_refs: Vec<&[Vertex]> = candidates.iter().map(|c| c.as_slice()).collect();
        let evals = calc.fitness_of_batch(&cand_refs);
        let mut best_idx: Option<usize> = None;
        let mut best_fit = current_fitness;
        for (i, e) in evals.iter().enumerate() {
            match kinds[i] {
                OpKind::Positional => pos_gen += 1,
                OpKind::Chromatic => col_gen += 1,
                OpKind::Structural => {}
            }
            if e.score > current_fitness {
                match kinds[i] {
                    OpKind::Positional => pos_better += 1,
                    OpKind::Chromatic => col_better += 1,
                    OpKind::Structural => {}
                }
            }
            if e.score > best_fit {
                best_fit = e.score;
                best_idx = Some(i);
            }
        }
```

- [ ] **Step 8: Update the accept block to drop the old window counter**

In the accept block (lines ~791-799) remove the `accepts_in_sigma_window += 1;` line. The block becomes:

```rust
        let mut accepted = false;
        if let Some(i) = best_idx {
            current = candidates.swap_remove(i);
            current_fitness = best_fit;
            accepts_in_plateau_window += 1;
            improvements_total += 1;
            accepted = true;
        }
        steps_in_sigma_window += 1;
        step += 1;
        phase_step += 1;
```

- [ ] **Step 9: Replace the 1/5 adaptation block with per-type adaptation**

Replace the `// 1/5 success rule` block (lines ~804-814):

```rust
        // 1/5 success rule: maintain ~20% acceptance rate.
        if steps_in_sigma_window >= SIGMA_WINDOW {
            let rate = accepts_in_sigma_window as f32 / steps_in_sigma_window as f32;
            if rate > 0.2 {
                sigma = (sigma * 1.15).min(0.5);
            } else if rate < 0.2 {
                sigma = (sigma * 0.85).max(0.005);
            }
            steps_in_sigma_window = 0;
            accepts_in_sigma_window = 0;
        }
```

with:

```rust
        // Per-type 1/5 success rule: adapt each step size toward a ~20%
        // beat-the-parent rate, independently, over SIGMA_WINDOW steps.
        if steps_in_sigma_window >= SIGMA_WINDOW {
            if pos_gen > 0 {
                let rate = pos_better as f32 / pos_gen as f32;
                if rate > 0.2 {
                    sigma_pos = (sigma_pos * 1.15).min(SIGMA_POS_MAX);
                } else if rate < 0.2 {
                    sigma_pos = (sigma_pos * 0.85).max(SIGMA_POS_MIN);
                }
            }
            if col_gen > 0 {
                let rate = col_better as f32 / col_gen as f32;
                if rate > 0.2 {
                    sigma_col = (sigma_col * 1.15).min(SIGMA_COL_MAX);
                } else if rate < 0.2 {
                    sigma_col = (sigma_col * 0.85).max(SIGMA_COL_MIN);
                }
            }
            steps_in_sigma_window = 0;
            pos_gen = 0;
            pos_better = 0;
            col_gen = 0;
            col_better = 0;
        }
```

- [ ] **Step 10: Update the periodic progress log**

Replace the periodic `println!("step ... σ={:.3} ...")` (lines ~825-836) with:

```rust
        if last_log.elapsed().as_secs_f32() >= 1.0 {
            println!(
                "step {:>6} | phase {} | tris {:>3} | σ_pos={:.3} σ_col={:.3} | fit {:>7} | improvements {} | {:.1}/s",
                step,
                phase_idx,
                current.len() / 3,
                sigma_pos,
                sigma_col,
                current_fitness,
                improvements_total,
                step as f64 / started.elapsed().as_secs_f64()
            );
            last_log = Instant::now();
        }
```

- [ ] **Step 11: Update phase-promotion sigma reset + log**

In the promotion branch, replace `sigma = new_phase.initial_sigma;` (line ~849) with:

```rust
                sigma_pos = new_phase.initial_sigma_pos;
                sigma_col = new_phase.initial_sigma_col;
```

Replace the promotion `println!("→ Phase {} | ... σ={:.3} ...")` (lines ~853-861) with:

```rust
                println!(
                    "→ Phase {} | {} triangles | level {} ({}²) | σ_pos={:.3} σ_col={:.3} | re-scored fitness {}",
                    phase_idx,
                    new_phase.triangles,
                    new_phase.pyramid_level,
                    pyramid[new_phase.pyramid_level].inner.texture_size,
                    sigma_pos,
                    sigma_col,
                    current_fitness
                );
```

- [ ] **Step 12: Update the final-phase sigma-restart branch + log**

Replace the `} else if plateaued {` branch body (lines ~867-879):

```rust
            } else if plateaued {
                // No further phases to promote to. Kick σ back to this phase's
                // initial_sigma so the search re-explores instead of grinding
                // at near-zero step size. Reset phase_step so the next plateau
                // evaluation waits another PHASE_MIN_STEPS + PLATEAU_WINDOW.
                let old_sigma = sigma;
                sigma = phase.initial_sigma;
                phase_step = 0;
                println!(
                    "⤴ Sigma restart (no further phases) | σ {:.3} → {:.3}",
                    old_sigma, sigma
                );
            }
```

with:

```rust
            } else if plateaued {
                // No further phases to promote to. Kick both σ back to this
                // phase's initial sizes so the search re-explores instead of
                // grinding at near-zero step size. Reset phase_step so the next
                // plateau evaluation waits another PHASE_MIN_STEPS + PLATEAU_WINDOW.
                let (old_pos, old_col) = (sigma_pos, sigma_col);
                sigma_pos = phase.initial_sigma_pos;
                sigma_col = phase.initial_sigma_col;
                phase_step = 0;
                println!(
                    "⤴ Sigma restart (no further phases) | σ_pos {:.3}→{:.3} σ_col {:.3}→{:.3}",
                    old_pos, sigma_pos, old_col, sigma_col
                );
            }
```

- [ ] **Step 13: Update the smoke-test `Phase`**

In `ga_improves_on_synthetic_checker`, replace the `test_phases` `Phase` (lines ~944-948):

```rust
        let test_phases = vec![Phase {
            triangles: 6,
            pyramid_level: 0,
            initial_sigma: 0.1,
        }];
```

with:

```rust
        let test_phases = vec![Phase {
            triangles: 6,
            pyramid_level: 0,
            initial_sigma_pos: 0.1,
            initial_sigma_col: 0.1,
        }];
```

- [ ] **Step 14: Run all tests**

Run: `cargo test --bin polygenvo`
Expected: PASS — all four tests still pass with the decoupled-sigma `mutate`.

- [ ] **Step 15: Commit**

```bash
git add src/polygenvo/main.rs
git commit -m "feat: decoupled Gaussian self-adaptive step sizes

Split sigma into sigma_pos and sigma_col with Gaussian perturbations,
each adapted by its own 1/5 success rule on a per-category beat-the-
parent rate.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Error-guided placement

Cache the accepted parent's error grid and use it to bias the `add` operator and a new `relocate` operator toward high-residual regions.

**Files:**
- Modify: `src/polygenvo/main.rs` (`sample_error_cell`, `cell_to_clip`, `error_seeded_triangle` helpers; `mutate` signature + add/relocate ops; `run_es` grid caching; tests)

- [ ] **Step 1: Write failing tests for the placement helpers**

Add to `mod tests`:

```rust
    #[test]
    fn sample_error_cell_favours_high_error() {
        // Cell 2 dominates; with a fixed seed it should be picked almost always.
        let grid = vec![0u32, 0, 100, 0];
        let mut rng = StdRng::seed_from_u64(1);
        let hits = (0..1000).filter(|_| sample_error_cell(&grid, &mut rng) == 2).count();
        assert!(hits >= 999, "dominant cell chosen {hits}/1000");
    }

    #[test]
    fn sample_error_cell_uniform_when_empty() {
        // All-zero grid -> uniform fallback over the four cells.
        let grid = vec![0u32; 4];
        let mut rng = StdRng::seed_from_u64(2);
        let mut counts = [0usize; 4];
        for _ in 0..4000 {
            counts[sample_error_cell(&grid, &mut rng)] += 1;
        }
        assert!(counts.iter().all(|&c| c > 700), "not roughly uniform: {counts:?}");
    }

    #[test]
    fn cell_to_clip_stays_in_cell_bounds() {
        // For ERROR_GRID_DIM=16, cell 0 is the top-left; its clip-x spans
        // [-1, -1 + 2/16] and clip-y spans [1 - 2/16, 1].
        let g = ERROR_GRID_DIM as f32;
        let (cx, cy) = cell_to_clip(0, 0.5, 0.5);
        assert!(cx >= -1.0 && cx <= -1.0 + 2.0 / g, "cx {cx} out of cell 0");
        assert!(cy <= 1.0 && cy >= 1.0 - 2.0 / g, "cy {cy} out of cell 0");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --bin polygenvo sample_error_cell`
Expected: compile error — `cannot find function sample_error_cell` / `cell_to_clip`.

- [ ] **Step 3: Add the placement helpers**

Above `fn mutate(` in `src/polygenvo/main.rs` add:

```rust
/// Roulette-select a grid cell index with probability proportional to its error
/// weight. Falls back to uniform when the grid is all-zero (e.g. a perfect match).
fn sample_error_cell(grid: &[u32], rng: &mut impl Rng) -> usize {
    let total: u64 = grid.iter().map(|&w| w as u64).sum();
    if total == 0 {
        return rng.random_range(0..grid.len());
    }
    let mut pick = rng.random_range(0..total);
    for (i, &w) in grid.iter().enumerate() {
        let w = w as u64;
        if pick < w {
            return i;
        }
        pick -= w;
    }
    grid.len() - 1
}

/// Map an error-grid cell plus intra-cell jitter (`jx`, `jy` in [0,1]) to a
/// clip-space point in [-1,1]². Cell row 0 is the top of the image, so clip y is
/// flipped to match (the fitness shader bins with row 0 = top).
fn cell_to_clip(cell: usize, jx: f32, jy: f32) -> (f32, f32) {
    let g = ERROR_GRID_DIM as f32;
    let gx = (cell % ERROR_GRID_DIM as usize) as f32;
    let gy = (cell / ERROR_GRID_DIM as usize) as f32;
    let u = (gx + jx) / g; // [0,1] across the image width
    let v = (gy + jy) / g; // [0,1] top→bottom
    (u * 2.0 - 1.0, 1.0 - v * 2.0)
}

/// Like `random_color_seeded_triangle`, but the centre is drawn from a high-error
/// grid cell rather than uniformly across the canvas.
fn error_seeded_triangle(
    goal: &GoalImage,
    error_grid: &[u32],
    rng: &mut impl Rng,
    max_radius: f32,
) -> [Vertex; 3] {
    let cell = sample_error_cell(error_grid, rng);
    let (cx, cy) = cell_to_clip(cell, rng.random_range(0.0..1.0), rng.random_range(0.0..1.0));
    let radius = rng.random_range(max_radius * 0.3..max_radius);
    let alpha = rng.random_range(0.25_f32..0.75);
    let color = sample_goal_color(goal, cx, cy, alpha);
    let base = rng.random_range(0.0_f32..std::f32::consts::TAU);
    let third = std::f32::consts::TAU / 3.0;
    let mk = |theta: f32| Vertex {
        position: [cx + radius * theta.cos(), cy + radius * theta.sin(), 0.0],
        color,
    };
    [mk(base), mk(base + third), mk(base + 2.0 * third)]
}
```

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `cargo test --bin polygenvo sample_error_cell cell_to_clip`
Expected: PASS (3 tests: both `sample_error_cell_*` and `cell_to_clip_stays_in_cell_bounds`).

- [ ] **Step 5: Add `error_grid` to `mutate` and wire in error-guided add + relocate**

Replace the `fn mutate(...)` signature line and the `add` / `delete` arms. First, change the signature (the parameter list written in Task 3 Step 3):

```rust
fn mutate(
    parent: &[Vertex],
    sigma_pos: f32,
    sigma_col: f32,
    min_triangles: usize,
    max_triangles: usize,
    goal: &GoalImage,
    error_grid: &[u32],
    rng: &mut impl Rng,
) -> (Vec<Vertex>, OpKind) {
```

Then replace the `80..=89` (z-swap), `90..=94` (add), and `_` (delete) arms with the retuned operator set that introduces `relocate`:

```rust
        78..=85 => {
            // Swap z-order with a neighbouring triangle.
            if n > 1 {
                let t = rng.random_range(0..n - 1);
                for v in 0..3 {
                    child.swap(t * 3 + v, (t + 1) * 3 + v);
                }
            }
            OpKind::Structural
        }
        86..=91 => {
            // Add a new triangle seeded in a high-error region.
            if n < max_triangles {
                let tri = error_seeded_triangle(goal, error_grid, rng, 0.2);
                let insert_at = rng.random_range(0..=n) * 3;
                for (offset, vert) in tri.iter().enumerate() {
                    child.insert(insert_at + offset, *vert);
                }
            }
            OpKind::Structural
        }
        92..=95 => {
            // Relocate an existing triangle's centroid to a high-error cell and
            // recolour it to that region — recycles triangles that aren't helping.
            let t = rng.random_range(0..n);
            let base = t * 3;
            let cell = sample_error_cell(error_grid, rng);
            let (tx, ty) =
                cell_to_clip(cell, rng.random_range(0.0..1.0), rng.random_range(0.0..1.0));
            let ccx = (child[base].position[0]
                + child[base + 1].position[0]
                + child[base + 2].position[0])
                / 3.0;
            let ccy = (child[base].position[1]
                + child[base + 1].position[1]
                + child[base + 2].position[1])
                / 3.0;
            let (dx, dy) = (tx - ccx, ty - ccy);
            let col = sample_goal_color(goal, tx, ty, child[base].color[3]);
            for v in 0..3 {
                child[base + v].position[0] = (child[base + v].position[0] + dx).clamp(-1.0, 1.0);
                child[base + v].position[1] = (child[base + v].position[1] + dy).clamp(-1.0, 1.0);
                child[base + v].color[0] = col[0];
                child[base + v].color[1] = col[1];
                child[base + v].color[2] = col[2];
            }
            OpKind::Structural
        }
        _ => {
            // Delete one triangle (op in 96..=99).
            if n > min_triangles {
                let t = rng.random_range(0..n);
                for _ in 0..3 {
                    child.remove(t * 3);
                }
            }
            OpKind::Structural
        }
```

- [ ] **Step 6: Capture the parent error grid in `run_es` (initial scoring)**

Replace the initial scoring line (line ~738):

```rust
    let mut current_fitness = pyramid[cfg.phases[phase_idx].pyramid_level].fitness_of(&current);
```

with:

```rust
    let mut current_fitness;
    let mut parent_error_grid: Vec<u32>;
    {
        let mut e = pyramid[cfg.phases[phase_idx].pyramid_level].fitness_of_batch(&[current.as_slice()]);
        let ev = e.swap_remove(0);
        current_fitness = ev.score;
        parent_error_grid = ev.error_grid;
    }
```

- [ ] **Step 7: Pass the grid into `mutate` and refresh it on accept**

Replace the candidate loop + accept written in Task 3 (Steps 7 and 8). The generation loop's `mutate` call gains `&parent_error_grid`:

```rust
        // (1+λ): produce λ candidates and evaluate them all in one GPU submit.
        let mut candidates: Vec<Vec<Vertex>> = Vec::with_capacity(cfg.lambda);
        let mut kinds: Vec<OpKind> = Vec::with_capacity(cfg.lambda);
        for _ in 0..cfg.lambda {
            let (child, kind) = mutate(
                &current, sigma_pos, sigma_col, min_triangles, max_triangles, &goal,
                &parent_error_grid, &mut rng,
            );
            candidates.push(child);
            kinds.push(kind);
        }
        let cand_refs: Vec<&[Vertex]> = candidates.iter().map(|c| c.as_slice()).collect();
        let evals = calc.fitness_of_batch(&cand_refs);
        let mut best_idx: Option<usize> = None;
        let mut best_fit = current_fitness;
        for (i, e) in evals.iter().enumerate() {
            match kinds[i] {
                OpKind::Positional => pos_gen += 1,
                OpKind::Chromatic => col_gen += 1,
                OpKind::Structural => {}
            }
            if e.score > current_fitness {
                match kinds[i] {
                    OpKind::Positional => pos_better += 1,
                    OpKind::Chromatic => col_better += 1,
                    OpKind::Structural => {}
                }
            }
            if e.score > best_fit {
                best_fit = e.score;
                best_idx = Some(i);
            }
        }
```

And the accept block caches the winner's grid (read it before `swap_remove`, which only reorders `candidates`, not `evals`):

```rust
        let mut accepted = false;
        if let Some(i) = best_idx {
            parent_error_grid = evals[i].error_grid.clone();
            current = candidates.swap_remove(i);
            current_fitness = best_fit;
            accepts_in_plateau_window += 1;
            improvements_total += 1;
            accepted = true;
        }
        steps_in_sigma_window += 1;
        step += 1;
        phase_step += 1;
```

- [ ] **Step 8: Refresh the grid on phase promotion re-score**

In the promotion branch, replace the re-score line (line ~851):

```rust
                current_fitness = pyramid[new_phase.pyramid_level].fitness_of(&current);
```

with:

```rust
                {
                    let mut e = pyramid[new_phase.pyramid_level].fitness_of_batch(&[current.as_slice()]);
                    let ev = e.swap_remove(0);
                    current_fitness = ev.score;
                    parent_error_grid = ev.error_grid;
                }
```

- [ ] **Step 9: Run all tests**

Run: `cargo test --bin polygenvo`
Expected: PASS — all tests (`gaussian_*`, `batch_scores_match_single`, `error_grid_tracks_residual`, `sample_error_cell_*`, `cell_to_clip_*`, `ga_improves_on_synthetic_checker`).

- [ ] **Step 10: Commit**

```bash
git add src/polygenvo/main.rs
git commit -m "feat: error-guided triangle placement and relocate operator

Cache the accepted parent's residual-error grid and bias the add
operator + a new relocate operator toward high-error cells via roulette.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Manual validation

No code; this is the CLAUDE.md "run the sim and eyeball the frames" guard, comparing against `master`.

**Files:** none.

- [ ] **Step 1: Build release**

Run: `cargo build --release --bin polygenvo`
Expected: clean build.

- [ ] **Step 2: Confirm runtime prerequisites**

Run: `ls goal.png && mkdir -p triangles`
Expected: `goal.png` exists; `triangles/` present.

- [ ] **Step 3: Run a bounded comparison run**

Run: `cargo run --release --bin polygenvo`
Let it run a few minutes (Ctrl-C to stop). Note from the log:
- steps/sec — expect a multiple of `master` (batched evaluation removes per-candidate GPU sync).
- fitness climb and that `σ_pos`/`σ_col` adapt independently in the log lines.

- [ ] **Step 4: Eyeball the output frames**

Open `triangles/final.png` and the latest `triangles/imageN.png` against `goal.png`. Expect detail in previously-blurry regions (error-guided placement) and continued improvement past where `master` flat-lined.

- [ ] **Step 5: Record findings**

Note steps/sec and final fitness vs `master` in the PR / branch description. If σ collapses or fitness regresses, revisit the per-type clamp constants (`SIGMA_*_MIN/MAX`) and operator probability split in `mutate` — both are flagged as tunables in the spec.

---

## Self-review notes (author)

- **Spec coverage:** Batched eval (Task 2c/2d), workgroup reduction + scale 8192 in params (Task 2a/2b), error grid (Task 2b + Task 4), decoupled Gaussian self-adaptive sigmas + per-type 1/5 (Task 1 + Task 3), error-guided add + relocate (Task 4), Phase split + tunables (Task 3 Step 2, Task 2a), all tests from the spec (gaussian, batch consistency, grid invariant, roulette/placement) present, manual validation (Task 5). ✓
- **Deviation from spec:** slot selection uses one bind group per slot (offset binding) instead of an immediate constant — the spec explicitly allowed a fallback "if immediate data proves awkward under wgpu 29"; this avoids the `PUSH_CONSTANTS` feature and uses only APIs already in the file. `ERROR_GRID_DIM` is mirrored as a WGSL `const GRID_DIM` (array sizes must be compile-time in WGSL) rather than passed in params; only `FITNESS_SCALE` travels in params. Both noted inline.
- **Type consistency:** `Eval { score, error_grid }`, `fitness_of_batch(&[&[Vertex]]) -> Vec<Eval>`, `mutate(...) -> (Vec<Vertex>, OpKind)`, `OpKind::{Positional,Chromatic,Structural}`, `sample_error_cell(&[u32], rng) -> usize`, `cell_to_clip(usize, f32, f32) -> (f32,f32)` used consistently across tasks. `mutate`'s signature gains `error_grid` in Task 4; Task 4 restates the full signature and arms.
