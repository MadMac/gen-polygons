use futures::executor::block_on;
use image::{ImageBuffer, Rgba};
use rand::prelude::*;
use std::fmt;
use std::iter;
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

// Triangle-count ceiling — the one knob that governs capacity. Raising it
// extends the auto-generated phase tail (see `production_phases`) and the
// vertex-buffer capacity below; lowering it shortens the tail.
const MAX_TRIANGLES: usize = 10000;

// Vertex buffer capacity (in vertices). 3 vertices per triangle.
const MAX_VERTICES: usize = MAX_TRIANGLES * 3;

// Geometric growth multiplier for the auto-generated high-count phases.
const PHASE_GROWTH: f32 = 1.6;

// (1+λ)-ES: number of mutated candidates evaluated per step.
const LAMBDA: usize = 6;

// 1/5 success rule: re-evaluate σ this often.
const SIGMA_WINDOW: u64 = 50;

// Minimum steps in a phase before promotion is considered.
const PHASE_MIN_STEPS: u64 = 400;

// Promote when the last PLATEAU_WINDOW steps yielded fewer than this many
// successful improvements.
const PLATEAU_WINDOW: u64 = 100;
const PLATEAU_ACCEPTS: u64 = 5;

// Snapshot a PNG every N steps that produced a successful improvement.
const SNAPSHOT_EVERY_IMPROVEMENT: u64 = 100;

// Hard cap on total ES steps (sanity).
const MAX_STEPS: u64 = 500_000;

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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq, PartialOrd)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 4],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

pub struct GoalImage {
    goal_image: image::ImageBuffer<image::Rgba<u8>, std::vec::Vec<u8>>,
}

impl Clone for GoalImage {
    fn clone(&self) -> GoalImage {
        GoalImage {
            goal_image: self.goal_image.clone(),
        }
    }
}

impl fmt::Debug for GoalImage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Empty debug")
    }
}

/// Result of scoring one candidate: the similarity score in [0, 1_000_000]
/// (higher = better) plus the coarse residual-error grid (length GRID_CELLS,
/// row-major, cell row 0 = top of the image) used to guide triangle placement.
#[derive(Clone, Debug)]
pub struct Eval {
    pub score: usize,
    pub error_grid: Vec<u32>,
}

struct FitnessCalcInner {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture_size: u32,
    render_pipeline: wgpu::RenderPipeline,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    vertex_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    // One bind group per output slot; binding 3 is offset into result_buffer.
    slot_bind_groups: Vec<wgpu::BindGroup>,
    result_buffer: wgpu::Buffer,
    result_readback: wgpu::Buffer,
}

#[derive(Clone)]
struct FitnessCalc {
    inner: Arc<FitnessCalcInner>,
}

impl fmt::Debug for FitnessCalc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FitnessCalc({0}x{0})", self.inner.texture_size)
    }
}

impl FitnessCalc {
    fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, goal_image: &GoalImage) -> Self {
        let texture_size = goal_image.goal_image.width();
        let target_format = wgpu::TextureFormat::Rgba8UnormSrgb;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Fitness Render Target"),
            size: wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&Default::default());

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fitness Render Pipeline"),
            layout: Some(&render_pipeline_layout),
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
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState {
                        alpha: wgpu::BlendComponent::OVER,
                        color: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: true,
            },
            multiview_mask: None,
            cache: None,
        });

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

        let goal_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Goal Texture"),
            size: wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Matches the render target so sampled values land in the same
            // colour space when the compute shader reads them.
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &goal_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            goal_image.goal_image.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(texture_size * 4),
                rows_per_image: Some(texture_size),
            },
            wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
        );
        let goal_texture_view = goal_texture.create_view(&Default::default());

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fitness Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("fitness.wgsl").into()),
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fitness Compute Pipeline"),
            layout: None,
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

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
    }
}

impl FitnessCalc {
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

    /// Render `vertices` and save the result as a PNG. Uses the same render
    /// pipeline as `fitness_of` but copies the texture back to the CPU and
    /// writes it through the `image` crate. A fresh readback buffer is
    /// allocated per call because snapshots are infrequent.
    fn snapshot(&self, vertices: &[Vertex], path: &Path) {
        let inner = &*self.inner;
        let num_vertices = vertices.len() as u32;
        let vertex_bytes: &[u8] = bytemuck::cast_slice(vertices);
        inner.queue.write_buffer(&inner.vertex_buffer, 0, vertex_bytes);

        let texture_size = inner.texture_size;
        let bytes_per_pixel = 4u32;
        let unpadded_bpr = bytes_per_pixel * texture_size;
        let padded_bpr = (unpadded_bpr + 255) & !255;
        let output_buffer = inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Snapshot Readback"),
            size: (padded_bpr * texture_size) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = inner.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Snapshot Encoder"),
        });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Snapshot Render Pass"),
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
            render_pass.set_vertex_buffer(0, inner.vertex_buffer.slice(..));
            render_pass.draw(0..num_vertices, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &inner.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(texture_size),
                },
            },
            wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
        );
        inner.queue.submit(iter::once(encoder.finish()));

        let slice = output_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        inner.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();

        // Strip row padding back to unpadded_bpr per row.
        let mut tight = Vec::with_capacity((unpadded_bpr * texture_size) as usize);
        for row in 0..texture_size {
            let start = (row * padded_bpr) as usize;
            let end = start + unpadded_bpr as usize;
            tight.extend_from_slice(&data[start..end]);
        }
        drop(data);
        output_buffer.unmap();

        let img = ImageBuffer::<Rgba<u8>, _>::from_raw(texture_size, texture_size, tight)
            .expect("snapshot buffer size mismatch");
        img.save(path).expect("snapshot write failed");
    }
}

// ---- (1+λ)-ES support: phases, mutation operators, initial seeding ----

#[derive(Clone)]
pub struct Phase {
    triangles: usize,
    pyramid_level: usize,
    // Initial step sizes for this phase, self-adapted by per-type 1/5 rules.
    initial_sigma_pos: f32,
    initial_sigma_col: f32,
}

// Hand-tuned coarse-to-fine warmup phases (the pyramid climb). The production
// schedule keeps these verbatim, then `production_phases` appends geometric
// high-count phases above the last warmup count up to MAX_TRIANGLES.
const WARMUP_PHASES: &[Phase] = &[
    Phase { triangles: 40,  pyramid_level: 0, initial_sigma_pos: 0.30, initial_sigma_col: 0.20 }, // 128² coarse
    Phase { triangles: 80,  pyramid_level: 1, initial_sigma_pos: 0.18, initial_sigma_col: 0.12 }, // 256² medium
    Phase { triangles: 120, pyramid_level: 2, initial_sigma_pos: 0.10, initial_sigma_col: 0.08 }, // 512² fine
    Phase { triangles: 150, pyramid_level: 2, initial_sigma_pos: 0.05, initial_sigma_col: 0.04 }, // 512² finer
];

// Compile-time coherence guard: the cap must be at least the warmup ceiling, or
// the auto-generated tail would be empty/nonsensical. Unlike `debug_assert!`,
// this also fires in release builds (the binary always runs `--release`).
const _: () = assert!(
    MAX_TRIANGLES >= WARMUP_PHASES[WARMUP_PHASES.len() - 1].triangles,
    "MAX_TRIANGLES is below the WARMUP_PHASES ceiling",
);

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

    let mut phases = WARMUP_PHASES.to_vec();
    let make_phase = |n: usize| Phase {
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
        phases.push(make_phase(n));
    }
    // Append the exact cap, unless the cap equals the warmup ceiling (no tail).
    if MAX_TRIANGLES > finest.triangles {
        phases.push(make_phase(MAX_TRIANGLES));
    }
    phases
}

pub struct EsConfig {
    pub phases: Vec<Phase>,
    pub max_steps: u64,
    pub lambda: usize,
    pub snapshot_every: Option<u64>,
}

impl EsConfig {
    fn production() -> Self {
        Self {
            phases: production_phases(),
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

/// Downsample the goal image to the given square size using a Lanczos filter.
fn downsample_goal(full: &GoalImage, size: u32) -> GoalImage {
    if size == full.goal_image.width() {
        return full.clone();
    }
    let dyn_img = image::DynamicImage::ImageRgba8(full.goal_image.clone());
    let resized = dyn_img.resize_exact(size, size, image::imageops::FilterType::Lanczos3).into_rgba8();
    GoalImage { goal_image: resized }
}

/// Build one `FitnessCalc` per pyramid level. Level indices match `Phase::pyramid_level`.
fn build_pyramid(device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>, goal: &GoalImage) -> Vec<FitnessCalc> {
    let full = goal.goal_image.width();
    let sizes = [full / 4, full / 2, full];
    sizes
        .iter()
        .map(|&s| {
            let g = downsample_goal(goal, s);
            FitnessCalc::new(device.clone(), queue.clone(), &g)
        })
        .collect()
}

/// Sample the goal image at a clip-space point to seed a triangle's colour.
/// Clip space `(-1, -1)` maps to top-left of the image (image y is flipped).
fn sample_goal_color(goal: &GoalImage, cx: f32, cy: f32, alpha: f32) -> [f32; 4] {
    let w = goal.goal_image.width();
    let h = goal.goal_image.height();
    let px = (((cx.clamp(-1.0, 1.0) + 1.0) * 0.5) * (w - 1) as f32) as u32;
    let py = (((1.0 - cy.clamp(-1.0, 1.0)) * 0.5) * (h - 1) as f32) as u32;
    let p = goal.goal_image.get_pixel(px.min(w - 1), py.min(h - 1));
    [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0, alpha]
}

/// Generate one triangle centred on `(cx, cy)` in clip space with the colour
/// sampled from the goal at that point. Vertices are placed in CCW order so
/// the rasteriser (front_face: Ccw, cull_mode: Back) keeps the triangle.
fn random_color_seeded_triangle(goal: &GoalImage, rng: &mut impl Rng, max_radius: f32) -> [Vertex; 3] {
    let cx = rng.random_range(-0.9_f32..0.9);
    let cy = rng.random_range(-0.9_f32..0.9);
    let radius = rng.random_range(max_radius * 0.3..max_radius);
    let alpha = rng.random_range(0.25_f32..0.75);
    let color = sample_goal_color(goal, cx, cy, alpha);

    let base = rng.random_range(0.0_f32..std::f32::consts::TAU);
    let third = std::f32::consts::TAU / 3.0;
    let mk = |theta: f32| Vertex {
        position: [cx + radius * theta.cos(), cy + radius * theta.sin(), 0.0],
        color,
    };
    // CCW (with wgpu's y-up clip space).
    [mk(base), mk(base + third), mk(base + 2.0 * third)]
}

fn init_genome(goal: &GoalImage, n_triangles: usize, rng: &mut impl Rng) -> Vec<Vertex> {
    let mut genome = Vec::with_capacity(n_triangles * 3);
    for _ in 0..n_triangles {
        let tri = random_color_seeded_triangle(goal, rng, 0.3);
        genome.extend_from_slice(&tri);
    }
    genome
}

/// Grow `genome` until it has exactly `target_triangles * 3` vertices by
/// appending new colour-seeded triangles. No-op if already at or above target.
fn grow_genome(genome: &mut Vec<Vertex>, target_triangles: usize, goal: &GoalImage, rng: &mut impl Rng) {
    while genome.len() / 3 < target_triangles {
        let tri = random_color_seeded_triangle(goal, rng, 0.2);
        genome.extend_from_slice(&tri);
    }
}

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
    error_grid: &[u32],
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
        65..=77 => {
            // Nudge the alpha of one triangle (Gaussian, sigma_col).
            let t = rng.random_range(0..n);
            let da = gaussian(rng, sigma_col);
            for v in 0..3 {
                let a = &mut child[t * 3 + v].color[3];
                *a = (*a + da).clamp(0.0, 1.0);
            }
            OpKind::Chromatic
        }
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
            // Move + clamp first, then recolour from the triangle's actual
            // post-clamp centroid: near the border a vertex can clamp, so the
            // landed centroid differs from the target (tx, ty) and we want the
            // colour of where the triangle actually ends up.
            for v in 0..3 {
                child[base + v].position[0] = (child[base + v].position[0] + dx).clamp(-1.0, 1.0);
                child[base + v].position[1] = (child[base + v].position[1] + dy).clamp(-1.0, 1.0);
            }
            let acx = (child[base].position[0]
                + child[base + 1].position[0]
                + child[base + 2].position[0])
                / 3.0;
            let acy = (child[base].position[1]
                + child[base + 1].position[1]
                + child[base + 2].position[1])
                / 3.0;
            let col = sample_goal_color(goal, acx, acy, child[base].color[3]);
            for v in 0..3 {
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
    };
    (child, kind)
}

fn load_goal_image(path: &str) -> GoalImage {
    let goal_image = GoalImage {
        goal_image: image::open(path)
            .unwrap_or_else(|e| panic!("failed to open goal image at {path}: {e}"))
            .into_rgba8(),
    };
    println!(
        "Loaded {} ({}x{})",
        path,
        goal_image.goal_image.width(),
        goal_image.goal_image.height()
    );
    goal_image
}

pub async fn init_wgpu() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL,
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
        .expect("no suitable wgpu adapter");
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
        .expect("device init failed");
    (Arc::new(device), Arc::new(queue))
}

pub fn run_es(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    goal: GoalImage,
    cfg: EsConfig,
) -> EsResult {
    let pyramid = build_pyramid(&device, &queue, &goal);
    let full_res = pyramid.len() - 1; // index of full-resolution level (for snapshots)

    let mut rng = rand::rng();

    // ---- Phase 0: initialise the genome at the first phase's triangle count ----
    let mut phase_idx: usize = 0;
    let mut current = init_genome(&goal, cfg.phases[phase_idx].triangles, &mut rng);
    let mut sigma_pos = cfg.phases[phase_idx].initial_sigma_pos;
    let mut sigma_col = cfg.phases[phase_idx].initial_sigma_col;
    let mut current_fitness;
    let mut parent_error_grid: Vec<u32>;
    {
        let mut e = pyramid[cfg.phases[phase_idx].pyramid_level].fitness_of_batch(&[current.as_slice()]);
        let ev = e.swap_remove(0);
        current_fitness = ev.score;
        parent_error_grid = ev.error_grid;
    }
    let initial_fitness = current_fitness;

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

    // ---- ES state ----
    let mut step: u64 = 0;
    let mut phase_step: u64 = 0;
    // Per-type 1/5 rule: count candidates generated and how many beat the parent,
    // separately for positional and chromatic mutations, over SIGMA_WINDOW steps.
    let mut steps_in_sigma_window: u64 = 0;
    let mut pos_gen: u64 = 0;
    let mut pos_better: u64 = 0;
    let mut col_gen: u64 = 0;
    let mut col_better: u64 = 0;
    let mut accepts_in_plateau_window: u64 = 0;
    let mut improvements_total: u64 = 0;
    let started = Instant::now();
    let mut last_log = Instant::now();

    // Trigger a snapshot of the initial state at full resolution so the
    // triangles/ directory has something to compare against.
    if let Some(_) = cfg.snapshot_every {
        let _ = std::fs::create_dir_all("triangles");
        pyramid[full_res].snapshot(&current, Path::new("triangles/image0.png"));
    }

    while step < cfg.max_steps {
        let phase = &cfg.phases[phase_idx];
        let calc = &pyramid[phase.pyramid_level];
        // Hold the genome near this phase's target. Allow ~25% shrinkage so
        // add/delete can shuffle the composition, but don't let add grow past
        // the phase's target — that's what phase promotion is for.
        let max_triangles = phase.triangles;
        let min_triangles = (phase.triangles * 3 / 4).max(8);

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

        // Snapshot occasionally on improvement.
        if let Some(snap_every) = cfg.snapshot_every {
            if accepted && improvements_total > 0 && improvements_total % snap_every == 0 {
                let path_buf = format!("triangles/image{}.png", step);
                pyramid[full_res].snapshot(&current, Path::new(&path_buf));
            }
        }

        // Periodic progress log (rate-limited so output stays readable).
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

        // Phase promotion. Two conditions: minimum time spent in this phase AND
        // a plateau (few accepts in the last PLATEAU_WINDOW). The plateau check
        // only kicks in once the rolling window has filled.
        if phase_step >= PHASE_MIN_STEPS && phase_step % PLATEAU_WINDOW == 0 {
            let plateaued = accepts_in_plateau_window < PLATEAU_ACCEPTS;
            accepts_in_plateau_window = 0;
            if plateaued && phase_idx + 1 < cfg.phases.len() {
                phase_idx += 1;
                let new_phase = &cfg.phases[phase_idx];
                grow_genome(&mut current, new_phase.triangles, &goal, &mut rng);
                sigma_pos = new_phase.initial_sigma_pos;
                sigma_col = new_phase.initial_sigma_col;
                // Re-score against the new (possibly higher-resolution) pyramid level.
                {
                    let mut e = pyramid[new_phase.pyramid_level].fitness_of_batch(&[current.as_slice()]);
                    let ev = e.swap_remove(0);
                    current_fitness = ev.score;
                    parent_error_grid = ev.error_grid;
                }
                phase_step = 0;
                // Clear the 1/5-rule window so the first adaptation window after the
                // transition doesn't mix stale old-phase stats with new-phase candidates.
                steps_in_sigma_window = 0;
                pos_gen = 0;
                pos_better = 0;
                col_gen = 0;
                col_better = 0;
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
                // Snapshot the new phase's starting frame.
                if let Some(_) = cfg.snapshot_every {
                    let path_buf = format!("triangles/image{}_phase{}.png", step, phase_idx);
                    pyramid[full_res].snapshot(&current, Path::new(&path_buf));
                }
            } else if plateaued {
                // No further phases to promote to. Kick both σ back to this
                // phase's initial sizes so the search re-explores instead of
                // grinding at near-zero step size. Reset phase_step so the next
                // plateau evaluation waits another PHASE_MIN_STEPS + PLATEAU_WINDOW.
                let (old_pos, old_col) = (sigma_pos, sigma_col);
                sigma_pos = phase.initial_sigma_pos;
                sigma_col = phase.initial_sigma_col;
                phase_step = 0;
                // Clear the 1/5-rule window so the first adaptation window after the
                // restart doesn't mix stale stats with post-restart candidates.
                steps_in_sigma_window = 0;
                pos_gen = 0;
                pos_better = 0;
                col_gen = 0;
                col_better = 0;
                println!(
                    "⤴ Sigma restart (no further phases) | σ_pos {:.3}→{:.3} σ_col {:.3}→{:.3}",
                    old_pos, sigma_pos, old_col, sigma_col
                );
            }
        }
    }

    println!(
        "Done. {} steps in {:.1}s, {} improvements, final fitness {}",
        step,
        started.elapsed().as_secs_f64(),
        improvements_total,
        current_fitness
    );
    if let Some(_) = cfg.snapshot_every {
        pyramid[full_res].snapshot(&current, Path::new("triangles/final.png"));
    }

    EsResult {
        initial_fitness,
        final_fitness: current_fitness,
        steps_run: step,
    }
}

fn main() {
    env_logger::init();
    let goal = load_goal_image("goal.png");
    let (device, queue) = block_on(init_wgpu());
    let cfg = EsConfig::production();
    let result = run_es(device, queue, goal, cfg);
    println!(
        "Done. Initial fitness: {}, final fitness: {}, steps: {}",
        result.initial_fitness, result.final_fitness, result.steps_run
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

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
        // Reuse the main init_wgpu helper; same backends/preferences as production.
        block_on(init_wgpu())
    }

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

    #[test]
    fn production_phases_schedule() {
        let phases = production_phases();
        let counts: Vec<usize> = phases.iter().map(|p| p.triangles).collect();

        // Starts with the four hand-tuned warmup phases, verbatim.
        assert_eq!(&counts[..WARMUP_PHASES.len()], &[40, 80, 120, 150]);

        // With the default constants (MAX_TRIANGLES=10000, PHASE_GROWTH=1.6) the
        // auto tail is geometric ×1.6 with the penultimate value snapped to the cap.
        assert_eq!(
            counts,
            vec![40, 80, 120, 150, 240, 384, 615, 984, 1575, 2520, 4032, 6452, 10000]
        );

        // Strictly increasing: no duplicates, no shrinkage.
        assert!(
            counts.windows(2).all(|w| w[1] > w[0]),
            "schedule not strictly increasing: {counts:?}"
        );

        // Ends exactly at the cap.
        assert_eq!(*counts.last().unwrap(), MAX_TRIANGLES);

        // Auto phases inherit the finest warmup phase's pyramid level and σ.
        let finest = WARMUP_PHASES.last().unwrap();
        for p in &phases[WARMUP_PHASES.len()..] {
            assert_eq!(p.pyramid_level, finest.pyramid_level);
            assert_eq!(p.initial_sigma_pos, finest.initial_sigma_pos);
            assert_eq!(p.initial_sigma_col, finest.initial_sigma_col);
        }
    }

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

    #[test]
    fn ga_improves_on_synthetic_checker() {
        let goal = make_checker_goal(32);
        let (device, queue) = init_test_wgpu();
        // Single-phase config for the test. pyramid_level 0 is the coarsest
        // level (build_pyramid sizes = [full/4, full/2, full]); for a 32×32
        // goal this evaluates at 8×8 — fast and plenty for a smoke test.
        let test_phases = vec![Phase {
            triangles: 6,
            pyramid_level: 0,
            initial_sigma_pos: 0.1,
            initial_sigma_col: 0.1,
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
        // A stuck-at-zero result usually means the GPU pipeline returned
        // garbage (e.g., a bind-group or texture-format mismatch silently
        // produced an empty render) — that's the most likely silent failure
        // mode of the wgpu migration, so guard against it explicitly.
        assert!(
            result.final_fitness > 0,
            "fitness stuck at zero — pipeline likely broken"
        );
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
