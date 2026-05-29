use futures::executor::block_on;
use image::{ImageBuffer, Rgba};
use rand::prelude::*;
use std::fmt;
use std::iter;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

// Vertex buffer capacity (in vertices). 450 vertices = 150 triangles.
const MAX_VERTICES: usize = 450;

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

struct GoalImage {
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

struct FitnessCalcInner {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture_size: u32,
    render_pipeline: wgpu::RenderPipeline,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: u64,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    fitness_buffer: wgpu::Buffer,
    fitness_readback: wgpu::Buffer,
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
        });
        let texture_view = texture.create_view(&Default::default());

        let render_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fitness Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState {
                        alpha: wgpu::BlendComponent::OVER,
                        color: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
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
            multiview: None,
        });

        // Genome size is constant per run; MAX_VERTICES gives headroom for any
        // future growth phase. Filled per call via queue.write_buffer.
        let vertex_capacity = (MAX_VERTICES as u64) * std::mem::size_of::<Vertex>() as u64;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: vertex_capacity,
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
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &goal_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            goal_image.goal_image.as_raw(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(texture_size * 4),
                rows_per_image: NonZeroU32::new(texture_size),
            },
            wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
        );
        let goal_texture_view = goal_texture.create_view(&Default::default());

        let compute_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Fitness Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("fitness.wgsl").into()),
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fitness Compute Pipeline"),
            layout: None,
            module: &compute_shader,
            entry_point: "main",
        });

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

        FitnessCalc {
            inner: Arc::new(FitnessCalcInner {
                device,
                queue,
                texture_size,
                render_pipeline,
                texture,
                texture_view,
                vertex_buffer,
                vertex_capacity,
                compute_pipeline,
                compute_bind_group,
                fitness_buffer,
                fitness_readback,
            }),
        }
    }
}

impl FitnessCalc {
    /// Render `vertices` into the internal texture, run the compute shader to
    /// compute the per-pixel diff against the goal, and return a fitness in
    /// `[0, 1_000_000]` where higher = closer match.
    fn fitness_of(&self, vertices: &[Vertex]) -> usize {
        let inner = &*self.inner;
        let num_vertices = vertices.len() as u32;
        let vertex_bytes: &[u8] = bytemuck::cast_slice(vertices);

        assert!(
            (vertex_bytes.len() as u64) <= inner.vertex_capacity,
            "Genome ({} bytes) exceeds preallocated vertex buffer ({} bytes)",
            vertex_bytes.len(),
            inner.vertex_capacity
        );

        inner.queue.write_buffer(&inner.vertex_buffer, 0, vertex_bytes);
        inner.queue.write_buffer(&inner.fitness_buffer, 0, &0u32.to_le_bytes());

        let mut encoder = inner.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Fitness Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fitness Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &inner.texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&inner.render_pipeline);
            render_pass.set_vertex_buffer(0, inner.vertex_buffer.slice(..));
            render_pass.draw(0..num_vertices, 0..1);
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fitness Compute Pass"),
            });
            compute_pass.set_pipeline(&inner.compute_pipeline);
            compute_pass.set_bind_group(0, &inner.compute_bind_group, &[]);
            let wg = (inner.texture_size + 7) / 8;
            compute_pass.dispatch(wg, wg, 1);
        }

        encoder.copy_buffer_to_buffer(
            &inner.fitness_buffer,
            0,
            &inner.fitness_readback,
            0,
            std::mem::size_of::<u32>() as u64,
        );

        inner.queue.submit(iter::once(encoder.finish()));

        let slice = inner.fitness_readback.slice(..);
        let mapping = slice.map_async(wgpu::MapMode::Read);
        inner.device.poll(wgpu::Maintain::Wait);
        block_on(mapping).unwrap();
        let raw = {
            let data = slice.get_mapped_range();
            u32::from_le_bytes([data[0], data[1], data[2], data[3]])
        };
        inner.fitness_readback.unmap();

        // Per-pixel: ΔE76 normalised to [0,1] and scaled to u32 by ×1000 (see fitness.wgsl).
        let max_total = (inner.texture_size as f64).powi(2) * 1000.0;
        let similarity = (1.0 - raw as f64 / max_total).max(0.0);
        (similarity * 1_000_000.0) as usize
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
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &inner.texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&inner.render_pipeline);
            render_pass.set_vertex_buffer(0, inner.vertex_buffer.slice(..));
            render_pass.draw(0..num_vertices, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &inner.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(padded_bpr),
                    rows_per_image: NonZeroU32::new(texture_size),
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
        let mapping = slice.map_async(wgpu::MapMode::Read);
        inner.device.poll(wgpu::Maintain::Wait);
        block_on(mapping).unwrap();
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
    // Initial sigma for this phase. Self-adapted by the 1/5 rule from here.
    initial_sigma: f32,
}

const PHASES: &[Phase] = &[
    Phase { triangles: 40,  pyramid_level: 0, initial_sigma: 0.25 },  // 128² coarse
    Phase { triangles: 80,  pyramid_level: 1, initial_sigma: 0.15 },  // 256² medium
    Phase { triangles: 120, pyramid_level: 2, initial_sigma: 0.10 },  // 512² fine
    Phase { triangles: 150, pyramid_level: 2, initial_sigma: 0.05 },  // 512² finer
];

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

/// Apply one random mutation to a clone of `parent`. Operator probabilities
/// are roughly tuned for polygon-image evolution: small local changes dominate,
/// structural changes (add/delete/z-swap) happen rarely.
fn mutate(parent: &[Vertex], sigma: f32, min_triangles: usize, max_triangles: usize, goal: &GoalImage, rng: &mut impl Rng) -> Vec<Vertex> {
    let mut child = parent.to_vec();
    let n = child.len() / 3;
    if n == 0 {
        // Pathological: rebuild from scratch.
        return init_genome(goal, min_triangles, rng);
    }

    let op = rng.random_range(0u32..100);
    match op {
        0..=39 => {
            // Nudge a single vertex of one triangle.
            let t = rng.random_range(0..n);
            let v = rng.random_range(0..3);
            let idx = t * 3 + v;
            let dx = rng.random_range(-sigma..sigma);
            let dy = rng.random_range(-sigma..sigma);
            child[idx].position[0] = (child[idx].position[0] + dx).clamp(-1.0, 1.0);
            child[idx].position[1] = (child[idx].position[1] + dy).clamp(-1.0, 1.0);
        }
        40..=64 => {
            // Recolour all three vertices of one triangle (RGB).
            let t = rng.random_range(0..n);
            let dr = rng.random_range(-sigma..sigma);
            let dg = rng.random_range(-sigma..sigma);
            let db = rng.random_range(-sigma..sigma);
            for v in 0..3 {
                let c = &mut child[t * 3 + v].color;
                c[0] = (c[0] + dr).clamp(0.0, 1.0);
                c[1] = (c[1] + dg).clamp(0.0, 1.0);
                c[2] = (c[2] + db).clamp(0.0, 1.0);
            }
        }
        65..=79 => {
            // Nudge the alpha of one triangle.
            let t = rng.random_range(0..n);
            let da = rng.random_range(-sigma..sigma);
            for v in 0..3 {
                let a = &mut child[t * 3 + v].color[3];
                *a = (*a + da).clamp(0.0, 1.0);
            }
        }
        80..=89 => {
            // Swap z-order with a neighbouring triangle.
            if n > 1 {
                let t = rng.random_range(0..n - 1);
                for v in 0..3 {
                    child.swap(t * 3 + v, (t + 1) * 3 + v);
                }
            }
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
        }
        _ => {
            // Delete one triangle.
            if n > min_triangles {
                let t = rng.random_range(0..n);
                for _ in 0..3 {
                    child.remove(t * 3);
                }
            }
        }
    }
    child
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

async fn init_wgpu() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let instance = wgpu::Instance::new(wgpu::Backends::GL);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("no suitable wgpu adapter");
    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .await
        .expect("device init failed");
    (Arc::new(device), Arc::new(queue))
}

fn run_es(
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
    let mut sigma = cfg.phases[phase_idx].initial_sigma;
    let mut current_fitness = pyramid[cfg.phases[phase_idx].pyramid_level].fitness_of(&current);
    let initial_fitness = current_fitness;

    println!(
        "Phase {} | {} triangles | level {} ({}²) | σ={:.3} | starting fitness {}",
        phase_idx,
        cfg.phases[phase_idx].triangles,
        cfg.phases[phase_idx].pyramid_level,
        pyramid[cfg.phases[phase_idx].pyramid_level].inner.texture_size,
        sigma,
        current_fitness
    );

    // ---- ES state ----
    let mut step: u64 = 0;
    let mut phase_step: u64 = 0;
    let mut accepts_in_sigma_window: u64 = 0;
    let mut steps_in_sigma_window: u64 = 0;
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

        let mut accepted = false;
        if let Some(i) = best_idx {
            current = candidates.swap_remove(i);
            current_fitness = best_fit;
            accepts_in_sigma_window += 1;
            accepts_in_plateau_window += 1;
            improvements_total += 1;
            accepted = true;
        }
        steps_in_sigma_window += 1;
        step += 1;
        phase_step += 1;

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
                "step {:>6} | phase {} | tris {:>3} | σ={:.3} | fit {:>7} | improvements {} | {:.1}/s",
                step,
                phase_idx,
                current.len() / 3,
                sigma,
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
                sigma = new_phase.initial_sigma;
                // Re-score against the new (possibly higher-resolution) pyramid level.
                current_fitness = pyramid[new_phase.pyramid_level].fitness_of(&current);
                phase_step = 0;
                println!(
                    "→ Phase {} | {} triangles | level {} ({}²) | σ={:.3} | re-scored fitness {}",
                    phase_idx,
                    new_phase.triangles,
                    new_phase.pyramid_level,
                    pyramid[new_phase.pyramid_level].inner.texture_size,
                    sigma,
                    current_fitness
                );
                // Snapshot the new phase's starting frame.
                if let Some(_) = cfg.snapshot_every {
                    let path_buf = format!("triangles/image{}_phase{}.png", step, phase_idx);
                    pyramid[full_res].snapshot(&current, Path::new(&path_buf));
                }
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

