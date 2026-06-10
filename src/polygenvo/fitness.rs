//! GPU fitness evaluator: renders a genome's triangles into an offscreen target
//! and scores it against the goal with a CIELAB ΔE76 compute pass. Also owns the
//! coarse residual-error grid used to guide triangle placement, and the
//! coarse-to-fine pyramid of evaluators.

use crate::genome::{Vertex, MAX_VERTICES};
use crate::goal::{downsample_goal, GoalImage};
use image::{ImageBuffer, Rgba};
use std::fmt;
use std::iter;
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// (1+λ)-ES batch capacity: λ candidates are uploaded and scored in one GPU
/// submit, so it also sizes the per-candidate vertex/result buffers below. Lives
/// here (rather than in `es`) because it is structurally a GPU-buffer dimension;
/// `es` reads it for its default config.
pub(crate) const LAMBDA: usize = 6;

// Per-pixel ΔE accumulator scale. Bounded by u32: largest pyramid level is
// 512² = 262144 px, and 262144 * FITNESS_SCALE must stay < 2^32, so the safe
// ceiling is ~16384. 8192 is 8× finer than the previous 1000 with headroom.
// Passed to the shader via the params uniform so the Rust normaliser and the
// shader share one source of truth.
pub(crate) const FITNESS_SCALE: u32 = 8192;

// Coarse residual-error grid emitted by the fitness pass for error-guided
// placement. MUST equal `GRID_DIM` in fitness.wgsl (WGSL array sizes must be
// compile-time constants, so the value is mirrored rather than passed).
pub(crate) const ERROR_GRID_DIM: u32 = 16;
pub(crate) const GRID_CELLS: usize = (ERROR_GRID_DIM * ERROR_GRID_DIM) as usize; // 256

// Per-candidate GPU output: one score u32 + GRID_CELLS grid u32. Storage-buffer
// binding offsets must be 256-aligned, so each slot is padded to SLOT_STRIDE.
const SLOT_PAYLOAD: u64 = 4 + (GRID_CELLS as u64) * 4; // 1028 bytes
const SLOT_STRIDE: u64 = SLOT_PAYLOAD.div_ceil(256) * 256; // 1280 bytes

/// MSAA sample count for the finest pyramid level. Only the full-resolution
/// level gets MSAA: there the per-pixel fitness compute dominates, so the extra
/// render cost is a small fraction of step time. The coarse levels are
/// render-bound (tiny compute, heavy translucent overdraw), where 4× MSAA
/// roughly halves throughput — so they stay at 1× for fast exploration.
/// Geometric edge AA only; alpha-to-coverage stays off because the triangles
/// already use real OVER alpha blending.
pub(crate) const MSAA_SAMPLE_COUNT: u32 = 4;

/// Result of scoring one candidate: the similarity score in [0, 1_000_000]
/// (higher = better) plus the coarse residual-error grid (length GRID_CELLS,
/// row-major, cell row 0 = top of the image) used to guide triangle placement.
#[derive(Clone, Debug)]
pub(crate) struct Eval {
    pub(crate) score: usize,
    pub(crate) error_grid: Vec<u32>,
}

struct FitnessCalcInner {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture_size: u32,
    render_pipeline: wgpu::RenderPipeline,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    // Multisampled render target, present only on MSAA levels. When `Some`, the
    // render pass draws into it and resolves into `texture`; when `None`, it
    // renders straight into `texture` (1× — resolve requires a >1× source).
    msaa_view: Option<wgpu::TextureView>,
    vertex_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    // One bind group per output slot; binding 3 is offset into result_buffer.
    slot_bind_groups: Vec<wgpu::BindGroup>,
    result_buffer: wgpu::Buffer,
    result_readback: wgpu::Buffer,
}

#[derive(Clone)]
pub(crate) struct FitnessCalc {
    inner: Arc<FitnessCalcInner>,
}

impl fmt::Debug for FitnessCalc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FitnessCalc({0}x{0})", self.inner.texture_size)
    }
}

impl FitnessCalc {
    fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        goal_image: &GoalImage,
        sample_count: u32,
    ) -> Self {
        let texture_size = goal_image.pixels.width();
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

        // Multisampled colour target, only when this level uses MSAA. The
        // render pass draws into it and resolves into `texture` (single-sample)
        // at end-of-pass; only RENDER_ATTACHMENT usage is needed since nothing
        // reads it directly. At 1× there is no MSAA target — the render pass
        // draws straight into `texture`.
        let msaa_view = (sample_count > 1).then(|| {
            let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Fitness MSAA Render Target"),
                size: wgpu::Extent3d {
                    width: texture_size,
                    height: texture_size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            msaa_texture.create_view(&Default::default())
        });

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
                count: sample_count,
                mask: !0,
                // Off: triangles use real OVER alpha blending, so a2c would
                // double-count alpha and dither translucent interiors.
                alpha_to_coverage_enabled: false,
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

        // Precompute the goal's CIELAB once, on the CPU, into a storage buffer
        // the scoring shader reads directly. The goal is fixed for this
        // evaluator's lifetime, so re-converting it sRGB→linear→XYZ→Lab on every
        // dispatch (λ × steps times, including cube roots) is pure waste. A
        // storage buffer — rather than an Rgba32Float texture — sidesteps the
        // filterable-float sample-type mismatch the auto-derived bind-group
        // layout (`layout: None`) would otherwise hit. `goal_to_lab` mirrors the
        // shader's exact math, so scores only shift in the low bits.
        let goal_lab: Vec<[f32; 4]> = goal_to_lab(goal_image);
        let goal_lab_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Goal CIELAB"),
            contents: bytemuck::cast_slice(&goal_lab),
            usage: wgpu::BufferUsages::STORAGE,
        });

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
                            resource: goal_lab_buffer.as_entire_binding(),
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
                msaa_view,
                vertex_buffer,
                compute_pipeline,
                slot_bind_groups,
                result_buffer,
                result_readback,
            }),
        }
    }

    /// The square edge length (px) this evaluator scores at. Used by callers for
    /// logging without exposing the private `inner`.
    pub(crate) fn texture_size(&self) -> u32 {
        self.inner.texture_size
    }

    /// Shared wgpu device. Exposed for gradient.rs (Task 6+).
    #[allow(dead_code)] // used by gradient.rs (Task 6+)
    pub(crate) fn device(&self) -> &std::sync::Arc<wgpu::Device> { &self.inner.device }

    /// Shared wgpu queue. Exposed for gradient.rs (Task 6+).
    #[allow(dead_code)] // used by gradient.rs (Task 6+)
    pub(crate) fn queue(&self) -> &std::sync::Arc<wgpu::Queue> { &self.inner.queue }

    /// Test/inter-module constructor alias for the private `new`.
    #[cfg(test)]
    pub(crate) fn new_for_test(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        goal: &GoalImage,
        sample_count: u32,
    ) -> Self {
        Self::new(device, queue, goal, sample_count)
    }
}

impl FitnessCalcInner {
    /// Colour-attachment wiring for a render pass. On MSAA levels, draw into the
    /// multisampled target and resolve into `texture` (discarding the MS buffer);
    /// at 1× draw straight into `texture` (resolve needs a >1× source).
    fn color_attachment(
        &self,
    ) -> (&wgpu::TextureView, Option<&wgpu::TextureView>, wgpu::StoreOp) {
        match &self.msaa_view {
            Some(ms) => (ms, Some(&self.texture_view), wgpu::StoreOp::Discard),
            None => (&self.texture_view, None, wgpu::StoreOp::Store),
        }
    }
}

impl FitnessCalc {
    /// Score `batch` candidates in a single GPU submit + readback. For each
    /// candidate i: render it into the shared target, then run the compute pass
    /// to write slot i. Within one command buffer, passes execute in order with
    /// automatic barriers, so reusing one render target across candidates is
    /// safe. Returns one `Eval` per candidate. `batch.len()` must be ≤ LAMBDA.
    pub(crate) fn fitness_of_batch(&self, batch: &[&[Vertex]]) -> Vec<Eval> {
        let inner = &*self.inner;
        assert!(
            batch.len() <= LAMBDA,
            "batch of {} exceeds LAMBDA {}",
            batch.len(),
            LAMBDA
        );
        let per_candidate_bytes = (MAX_VERTICES as u64) * std::mem::size_of::<Vertex>() as u64;

        // Upload all candidate vertices.
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

        let mut encoder = inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Fitness Encoder"),
            });
        // Zero the result buffer on the GPU before the compute passes atomicAdd
        // into it — no per-step heap alloc + CPU→GPU copy of zeros. Ordered
        // before the passes below by the encoder's automatic barriers.
        encoder.clear_buffer(&inner.result_buffer, 0, None);

        for (i, verts) in batch.iter().enumerate() {
            let num_vertices = verts.len() as u32;
            let vb_offset = i as u64 * per_candidate_bytes;
            {
                let (view, resolve_target, store) = inner.color_attachment();
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Fitness Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store,
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
                let wg = inner.texture_size.div_ceil(8);
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

    /// Score a single candidate. Thin wrapper over `fitness_of_batch`. `run_es`
    /// scores via the batch path directly, so this is exercised mainly by tests —
    /// `allow(dead_code)` keeps non-test builds warning-free.
    #[allow(dead_code)]
    pub(crate) fn fitness_of(&self, vertices: &[Vertex]) -> usize {
        self.fitness_of_batch(&[vertices])[0].score
    }

    /// Render `vertices` and save the result as a PNG. Uses the same render
    /// pipeline as `fitness_of` but copies the texture back to the CPU and
    /// writes it through the `image` crate. A fresh readback buffer is
    /// allocated per call because snapshots are infrequent.
    pub(crate) fn snapshot(&self, vertices: &[Vertex], path: &Path) {
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
            let (view, resolve_target, store) = inner.color_attachment();
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Snapshot Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store,
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

/// Standard sRGB EOTF (gamma-decode) for one channel in `[0,1]`. Matches what
/// `Rgba8UnormSrgb` hardware decode applies when the shader loads the render
/// target, so the CPU-baked goal Lab lands in the same colour space.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear-RGB (sRGB primaries, D65) → CIE XYZ. Same matrix as `fitness.wgsl`.
fn linear_rgb_to_xyz(r: f32, g: f32, b: f32) -> [f32; 3] {
    [
        r * 0.4124564 + g * 0.3575761 + b * 0.1804375,
        r * 0.2126729 + g * 0.7151522 + b * 0.0721750,
        r * 0.0193339 + g * 0.119_192 + b * 0.9503041,
    ]
}

/// CIE XYZ (D65) → CIELAB. Same constants as `fitness.wgsl`.
fn xyz_to_lab(xyz: [f32; 3]) -> [f32; 3] {
    let f = |t: f32| {
        if t > 0.008856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    let fx = f(xyz[0] / 0.95047);
    let fy = f(xyz[1] / 1.00000);
    let fz = f(xyz[2] / 1.08883);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// Bake the goal image to row-major CIELAB, one `[L, a, b, 0]` per pixel (the
/// trailing 0 keeps the storage element 16-byte aligned for the shader's
/// `array<vec4<f32>>`). Runs once per `FitnessCalc`; mirrors the shader's
/// sRGB→linear→XYZ→Lab path so scores only shift in the low bits.
pub(crate) fn goal_to_lab(goal: &GoalImage) -> Vec<[f32; 4]> {
    goal
        .pixels
        .pixels()
        .map(|p| {
            let r = srgb_to_linear(p[0] as f32 / 255.0);
            let g = srgb_to_linear(p[1] as f32 / 255.0);
            let b = srgb_to_linear(p[2] as f32 / 255.0);
            let lab = xyz_to_lab(linear_rgb_to_xyz(r, g, b));
            [lab[0], lab[1], lab[2], 0.0]
        })
        .collect()
}

/// Build one `FitnessCalc` per pyramid level. Level indices match `Phase::pyramid_level`.
pub(crate) fn build_pyramid(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    goal: &GoalImage,
) -> Vec<FitnessCalc> {
    let full = goal.pixels.width();
    let sizes = [full / 4, full / 2, full];
    let last = sizes.len() - 1;
    sizes
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let g = downsample_goal(goal, s);
            // MSAA only on the finest (full-res) level; coarse levels stay 1×.
            let sample_count = if i == last { MSAA_SAMPLE_COUNT } else { 1 };
            FitnessCalc::new(device.clone(), queue.clone(), &g, sample_count)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{init_genome, Vertex};
    use crate::test_support::{init_test_wgpu, make_checker_goal, make_solid_goal};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn batch_scores_match_single() {
        let goal = make_checker_goal(32);
        let (device, queue) = init_test_wgpu();
        // 4× here so this test also guards the MSAA render+resolve path; the
        // batch-vs-single equality below holds at any sample count.
        let calc = FitnessCalc::new(device, queue, &goal, MSAA_SAMPLE_COUNT);
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
        let calc_white = FitnessCalc::new(device.clone(), queue.clone(), &white, 1);
        let calc_black = FitnessCalc::new(device, queue, &black, 1);
        let empty: Vec<Vertex> = Vec::new();
        let ew = &calc_white.fitness_of_batch(&[empty.as_slice()])[0];
        let eb = &calc_black.fitness_of_batch(&[empty.as_slice()])[0];
        let sum_white: u64 = ew.error_grid.iter().map(|&w| w as u64).sum();
        let sum_black: u64 = eb.error_grid.iter().map(|&w| w as u64).sum();
        assert!(ew.error_grid.iter().all(|&w| w > 0), "white goal: every cell should have error");
        assert!(sum_white > sum_black * 10, "white {sum_white} should dwarf black {sum_black}");
    }
}
