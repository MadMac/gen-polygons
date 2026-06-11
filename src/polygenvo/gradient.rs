//! On-device differentiable-rasterizer polish: soft-raster forward+backward
//! (softraster_tiled.wgsl) + Adam (adam.wgsl) over all triangles' positions+colors,
//! minimizing Lab-MSE, then gated by the hard ΔE2000 renderer. Framework-free,
//! reuses FitnessCalc's wgpu device/queue. See
//! docs/superpowers/specs/2026-06-08-gpu-differentiable-rasterizer-design.md.

#[derive(Clone, Copy, Debug)]
pub(crate) struct PolishCfg {
    pub(crate) enabled: bool,
    pub(crate) every_k: u64,
    pub(crate) steps_n: u32,
    pub(crate) lr: f32,
    pub(crate) tau_start: f32,
    pub(crate) tau_end: f32,
}

impl Default for PolishCfg {
    fn default() -> Self {
        Self { enabled: false, every_k: 50, steps_n: 40, lr: 0.05, tau_start: 0.3, tau_end: 0.02 }
    }
}

// Adam optimizer hyperparameters (standard defaults; not exposed in PolishCfg
// because they're rarely worth tuning — lr/tau/steps in PolishCfg are the knobs).
const ADAM_B1: f32 = 0.9;
const ADAM_B2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

/// Uniform params for the `softraster_tiled.wgsl` forward/backward passes.
/// `#[repr(C)]` + Pod so bytemuck casts it straight to the uniform buffer the
/// shader reads; `tiles_x` maps a pixel to its tile.
/// Padded to a 16-byte multiple (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftRasterParams {
    width: u32,
    height: u32,
    num_tris: u32,
    tau: f32,
    tiles_x: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Uniform params for `binning.wgsl` (count/scan/fill/sort). Matches the WGSL
/// `BinParams` struct field-for-field: 8 × u32/f32 = 32 bytes. `#[repr(C)]` + Pod
/// so bytemuck casts it straight to the uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BinParams {
    num_tris: u32,
    tiles_x: u32,
    tiles_y: u32,
    width: u32,
    height: u32,
    tau: f32,
    list_cap: u32,
    _pad: u32,
}

/// Uniform params for the `adam.wgsl` update pass. Matches the WGSL `AdamParams`
/// struct field-for-field (16 bytes header + two u32 pads → 32 bytes total).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamUniform {
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    step_t: u32,
    num_params: u32,
    pad0: u32,
    pad1: u32,
}

/// Scene-dependent binning dimensions for `BinResources::write_params` (groups
/// the per-step fields so the call doesn't exceed clippy's argument limit).
#[derive(Clone, Copy)]
struct BinDims {
    num_tris: u32,
    tiles_x: u32,
    tiles_y: u32,
    width: u32,
    height: u32,
    tau: f32,
}

/// Cached `binning.wgsl` pipelines + buffers. Both the production `PolishState`
/// and the `#[cfg(test)]` helpers own one of these so the binning pass sequence
/// (clear counts → count → scan → reset counts → fill → sort_tiles) is written
/// the same way everywhere. `record` appends the six commands to an encoder;
/// callers then bind `offsets_buf`(5)/`list_buf`(6) on the forward/backward
/// tiled bind groups. The bind group references an externally supplied
/// `tri_params` buffer (so it tracks the same triangle data the kernel reads).
struct BinResources {
    bp_buf: wgpu::Buffer,
    counts_buf: wgpu::Buffer,
    offsets_buf: wgpu::Buffer,
    list_buf: wgpu::Buffer,
    overflow_buf: wgpu::Buffer,
    list_cap: u32,
    count_pipeline: wgpu::ComputePipeline,
    scan_pipeline: wgpu::ComputePipeline,
    fill_pipeline: wgpu::ComputePipeline,
    sort_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl BinResources {
    /// Build the binning pipelines + buffers. `num_tiles` sizes `tile_counts`
    /// (atomic u32) and `tile_offsets` (num_tiles+1); `list_cap` sizes
    /// `tile_list`. `tri_params` is the externally owned triangle-param storage
    /// buffer the count/fill passes read.
    fn new(
        device: &wgpu::Device,
        tri_params: &wgpu::Buffer,
        num_tiles: u64,
        list_cap: u32,
    ) -> Self {
        let bp_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bin Params"),
            size: std::mem::size_of::<BinParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bin TileCounts"),
            size: num_tiles * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bin TileOffsets"),
            size: (num_tiles + 1) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let list_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bin TileList"),
            size: (list_cap as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let overflow_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bin Overflow"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Binning Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("binning.wgsl").into()),
        });
        // Explicit shared layout (all six bindings); the four entries each use a
        // subset, so one bind group drives every pipeline.
        let storage = |ro: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Binning BGL"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(1, storage(true)),
                entry(2, storage(false)),
                entry(3, storage(false)),
                entry(4, storage(false)),
                entry(5, storage(false)),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Binning Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let make_pipeline = |ep: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(ep),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let count_pipeline = make_pipeline("count");
        let scan_pipeline = make_pipeline("scan");
        let fill_pipeline = make_pipeline("fill");
        let sort_pipeline = make_pipeline("sort_tiles");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binning Bind Group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tri_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: list_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: overflow_buf.as_entire_binding() },
            ],
        });

        Self {
            bp_buf,
            counts_buf,
            offsets_buf,
            list_buf,
            overflow_buf,
            list_cap,
            count_pipeline,
            scan_pipeline,
            fill_pipeline,
            sort_pipeline,
            bind_group,
        }
    }

    /// Write the BinParams uniform for this step (host-side; ordered before the
    /// next submit). `list_cap` and the `_pad` are filled from this `BinResources`,
    /// so the caller passes the scene-dependent fields via `BinDims`.
    fn write_params(&self, queue: &wgpu::Queue, dims: BinDims) {
        let bp = BinParams {
            num_tris: dims.num_tris,
            tiles_x: dims.tiles_x,
            tiles_y: dims.tiles_y,
            width: dims.width,
            height: dims.height,
            tau: dims.tau,
            list_cap: self.list_cap,
            _pad: 0,
        };
        queue.write_buffer(&self.bp_buf, 0, bytemuck::bytes_of(&bp));
    }

    /// Record the binning pass sequence into `encoder`: clear counts (+overflow)
    /// → count → scan → reset counts → fill → sort_tiles. Leaves `offsets_buf`
    /// (exclusive prefix sum, total at [num_tiles]) and `list_buf` (per-tile
    /// triangle indices, draw order) populated for the current BinParams.
    fn record(&self, encoder: &mut wgpu::CommandEncoder, num_tris: u32, tiles_x: u32, tiles_y: u32) {
        encoder.clear_buffer(&self.counts_buf, 0, None);
        encoder.clear_buffer(&self.overflow_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.count_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_tris.max(1).div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scan_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // Reset counts to 0 — fill reuses tile_counts as the per-tile cursor.
        encoder.clear_buffer(&self.counts_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fill"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fill_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_tris.max(1).div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sort_tiles"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sort_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(tiles_x * tiles_y, 1, 1);
        }
    }
}

/// Cached pipelines + buffers for the on-device gradient polish. Built once per
/// goal; `polish` then only writes the genome + uniforms and dispatches. The
/// `params_buf` doubles as backward's `tri_params` and Adam's `params`; the
/// `grad_buf` is written (atomic f32-via-u32) by backward and re-read as plain
/// f32 by Adam.
///
/// The production `polish` loop uses the TILED pipelines (`forward_tiled_pipeline` /
/// `backward_tiled_pipeline`) for the per-step forward+backward.
pub(crate) struct PolishState {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    width: u32,
    height: u32,
    params_buf: wgpu::Buffer,
    grad_buf: wgpu::Buffer,
    adam_m_buf: wgpu::Buffer,
    adam_v_buf: wgpu::Buffer,
    sr_params_buf: wgpu::Buffer,
    adam_params_buf: wgpu::Buffer,
    _goal_lab_buf: wgpu::Buffer,
    adam_pipeline: wgpu::ComputePipeline,
    adam_bind_group: wgpu::BindGroup,
    readback_buf: wgpu::Buffer,
    // Tiled forward+backward pipelines (used by the production polish loop).
    forward_tiled_pipeline: wgpu::ComputePipeline,
    backward_tiled_pipeline: wgpu::ComputePipeline,
    // Per-pixel state buffer: vec4<f32> = (c_full.rgb, T_final), w*h elements.
    // Only directly accessed here during construction; used indirectly via bind groups.
    #[allow(dead_code)]
    state_buf: wgpu::Buffer,
    forward_tiled_bind_group: wgpu::BindGroup,
    backward_tiled_bind_group: wgpu::BindGroup,
    // Per-tile triangle-list binning (rebuilt each polish step).
    bin: BinResources,
    tiles_x: u32,
    tiles_y: u32,
}

impl PolishState {
    /// Build the pipelines + buffers, sized to the maximum genome
    /// (`MAX_VERTICES * 6` scalar params). The goal-Lab storage buffer is filled
    /// once from `FitnessCalc::goal_to_lab`.
    pub(crate) fn new(calc: &crate::fitness::FitnessCalc, goal: &crate::goal::GoalImage) -> Self {
        use wgpu::util::DeviceExt;
        let device = calc.device().clone();
        let queue = calc.queue().clone();
        let size = calc.texture_size();
        let width = size;
        let height = size;

        let max_params = (crate::genome::MAX_VERTICES * 6) as u64;
        let param_bytes = max_params * 4;

        // Goal Lab as [L, a, b, 0] per pixel — exactly backward's goal_lab layout.
        let goal_lab = crate::fitness::goal_to_lab(goal);
        let goal_lab_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Polish GoalLab"),
            contents: bytemuck::cast_slice(&goal_lab),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Polish Params"),
            size: param_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grad_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Polish Grad"),
            size: param_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let adam_m_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Polish AdamM"),
            size: param_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let adam_v_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Polish AdamV"),
            size: param_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sr_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Polish SoftRasterParams"),
            size: std::mem::size_of::<SoftRasterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let adam_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Polish AdamUniform"),
            size: std::mem::size_of::<AdamUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Polish Readback"),
            size: param_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let adam_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Polish Adam Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("adam.wgsl").into()),
        });

        let adam_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Polish Adam Pipeline"),
            layout: None,
            module: &adam_shader,
            entry_point: Some("update"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let adam_bgl = adam_pipeline.get_bind_group_layout(0);
        let adam_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Polish Adam Bind Group"),
            layout: &adam_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: adam_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grad_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: adam_m_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: adam_v_buf.as_entire_binding() },
            ],
        });

        // ---- Tiled pipelines (softraster_tiled.wgsl) ----
        let tiled_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Polish Tiled Shader"),
            source: wgpu::ShaderSource::Wgsl(
                crate::gpu::with_color_prelude(include_str!("softraster_tiled.wgsl")).into(),
            ),
        });
        let forward_tiled_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Polish Tiled Forward Pipeline"),
                layout: None,
                module: &tiled_shader,
                entry_point: Some("forward"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let backward_tiled_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Polish Tiled Backward Pipeline"),
                layout: None,
                module: &tiled_shader,
                entry_point: Some("backward"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Persistent per-pixel state buffer: one vec4<f32> per pixel (16 bytes each).
        let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Polish TiledState"),
            size: (width as u64) * (height as u64) * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // ---- Binning resources (rebuilt each polish step) ----
        // num_tiles is fixed by the texture size; the per-tile lists are rebuilt
        // each step (positions move). list_cap = num_tiles*1024 sizes the total
        // triangle-tile-incidence buffer (at 512²: 1024 tiles → ~1.05M u32 ≈ 4 MiB).
        // Realistic late-stage genomes (10000 *small* triangles) need only tens of
        // entries/tile; the generous headroom also covers denser/larger-triangle
        // mid-run genomes before splitting. The `overflow` flag in BinResources
        // guards the pathological case (total incidences exceeding the cap).
        let tiles_x = width.div_ceil(16);
        let tiles_y = height.div_ceil(16);
        let num_tiles = (tiles_x * tiles_y) as u64;
        let list_cap = (num_tiles * 1024).max(1) as u32;
        let bin = BinResources::new(&device, &params_buf, num_tiles, list_cap);

        // Tiled forward bind group: 0=sr_params, 1=params_buf (tri_params),
        //   2=state, 5=tile_offsets, 6=tile_list.
        let fwd_tiled_bgl = forward_tiled_pipeline.get_bind_group_layout(0);
        let forward_tiled_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Polish Tiled Forward Bind Group"),
            layout: &fwd_tiled_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sr_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bin.offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bin.list_buf.as_entire_binding(),
                },
            ],
        });

        // Tiled backward bind group: 0=sr_params, 1=params_buf, 2=state,
        //   3=goal_lab_buf, 4=grad_buf, 5=tile_offsets, 6=tile_list.
        let bwd_tiled_bgl = backward_tiled_pipeline.get_bind_group_layout(0);
        let backward_tiled_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Polish Tiled Backward Bind Group"),
            layout: &bwd_tiled_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sr_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: goal_lab_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bin.offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bin.list_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            width,
            height,
            params_buf,
            grad_buf,
            adam_m_buf,
            adam_v_buf,
            sr_params_buf,
            adam_params_buf,
            _goal_lab_buf: goal_lab_buf,
            adam_pipeline,
            adam_bind_group,
            readback_buf,
            forward_tiled_pipeline,
            backward_tiled_pipeline,
            state_buf,
            forward_tiled_bind_group,
            backward_tiled_bind_group,
            bin,
            tiles_x,
            tiles_y,
        }
    }

    /// Run `cfg.steps_n` backward+Adam steps fully on-device starting from
    /// `genome`, then keep the result only if the hard ΔE2000 renderer
    /// (`calc.fitness_of`) confirms it beats `parent_fitness`. On accept, mutates
    /// `genome` and returns `Some(new_fitness)`; otherwise leaves `genome`
    /// untouched and returns `None`.
    /// Run `cfg.steps_n` binned gradient (soft-raster + Adam) steps over all
    /// vertices' positions+colors and return the optimized genome — NO gate. The
    /// shared core of `polish` (gated) and `polish_ungated`. Caller guards empty.
    fn apply_gradient_steps(
        &mut self,
        genome: &[crate::genome::Vertex],
        cfg: &PolishCfg,
    ) -> Vec<crate::genome::Vertex> {
        let n_verts = genome.len();
        let num_tris = (n_verts / 3) as u32;
        let num_params = (n_verts * 6) as u32;

        // Flatten genome → [pos.x, pos.y, r, g, b, a] per vertex; save z for rebuild.
        let mut flat: Vec<f32> = Vec::with_capacity(n_verts * 6);
        let mut zs: Vec<f32> = Vec::with_capacity(n_verts);
        for vtx in genome.iter() {
            flat.push(vtx.position[0]);
            flat.push(vtx.position[1]);
            flat.push(vtx.color[0]);
            flat.push(vtx.color[1]);
            flat.push(vtx.color[2]);
            flat.push(vtx.color[3]);
            zs.push(vtx.position[2]);
        }
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&flat));

        // Zero Adam moments for this run.
        {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Polish Moment Clear"),
            });
            enc.clear_buffer(&self.adam_m_buf, 0, None);
            enc.clear_buffer(&self.adam_v_buf, 0, None);
            self.queue.submit(std::iter::once(enc.finish()));
        }

        for s in 0..cfg.steps_n {
            let frac = if cfg.steps_n > 1 {
                s as f32 / (cfg.steps_n - 1) as f32
            } else {
                0.0
            };
            let tau = cfg.tau_start * (cfg.tau_end / cfg.tau_start).powf(frac);
            let sr = SoftRasterParams {
                width: self.width,
                height: self.height,
                num_tris,
                tau,
                tiles_x: self.width.div_ceil(16),
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            self.queue.write_buffer(&self.sr_params_buf, 0, bytemuck::bytes_of(&sr));
            let ap = AdamUniform {
                lr: cfg.lr,
                b1: ADAM_B1,
                b2: ADAM_B2,
                eps: ADAM_EPS,
                step_t: s + 1,
                num_params,
                pad0: 0,
                pad1: 0,
            };
            self.queue.write_buffer(&self.adam_params_buf, 0, bytemuck::bytes_of(&ap));
            // Binning uniform for this step (positions move, so re-bin each step).
            self.bin.write_params(
                &self.queue,
                BinDims {
                    num_tris,
                    tiles_x: self.tiles_x,
                    tiles_y: self.tiles_y,
                    width: self.width,
                    height: self.height,
                    tau,
                },
            );

            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Polish Step Encoder"),
            });
            // 0. Binning: rebuild per-tile triangle lists for the current params.
            self.bin.record(&mut enc, num_tris, self.tiles_x, self.tiles_y);
            // 1. Tiled forward: populate per-pixel state = (c_full.rgb, T_final).
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Polish Tiled Forward Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.forward_tiled_pipeline);
                pass.set_bind_group(0, &self.forward_tiled_bind_group, &[]);
                pass.dispatch_workgroups(
                    self.width.div_ceil(16),
                    self.height.div_ceil(16),
                    1,
                );
            }
            // 2. Clear grad buffer (before backward, after forward).
            enc.clear_buffer(&self.grad_buf, 0, None);
            // 3. Tiled backward: O(num_tris) reverse-transmittance gradient scatter.
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Polish Tiled Backward Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.backward_tiled_pipeline);
                pass.set_bind_group(0, &self.backward_tiled_bind_group, &[]);
                pass.dispatch_workgroups(
                    self.width.div_ceil(16),
                    self.height.div_ceil(16),
                    1,
                );
            }
            // 4. Adam update.
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Polish Adam Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.adam_pipeline);
                pass.set_bind_group(0, &self.adam_bind_group, &[]);
                pass.dispatch_workgroups(num_params.div_ceil(64), 1, 1);
            }
            // Per-step submit: the sr-params/adam uniforms change each step, and write_buffer
            // can't be recorded into an encoder; wgpu orders write_buffer before the next
            // submit, so each step sees its own uniforms.
            self.queue.submit(std::iter::once(enc.finish()));
        }

        // Read back the optimized params.
        let read_bytes = (num_params * 4) as u64;
        {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Polish Readback Encoder"),
            });
            enc.copy_buffer_to_buffer(&self.params_buf, 0, &self.readback_buf, 0, read_bytes);
            self.queue.submit(std::iter::once(enc.finish()));
        }
        let slice = self.readback_buf.slice(0..read_bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let out: Vec<f32> = {
            let data = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&data);
            floats[..num_params as usize].to_vec()
        };
        self.readback_buf.unmap();

        // Rebuild candidate genome from read-back params + saved z.
        let candidate: Vec<crate::genome::Vertex> = (0..n_verts)
            .map(|i| {
                let b = i * 6;
                crate::genome::Vertex {
                    position: [out[b], out[b + 1], zs[i]],
                    color: [out[b + 2], out[b + 3], out[b + 4], out[b + 5]],
                }
            })
            .collect();

        candidate
    }

    /// Gated polish: apply gradient steps to a copy; keep only if the hard ΔE2000
    /// renderer confirms it beats `parent_fitness` (the (1+λ) no-regression gate).
    pub(crate) fn polish(
        &mut self,
        genome: &mut Vec<crate::genome::Vertex>,
        parent_fitness: usize,
        calc: &crate::fitness::FitnessCalc,
        cfg: &PolishCfg,
    ) -> Option<usize> {
        if genome.is_empty() {
            return None;
        }
        let candidate = self.apply_gradient_steps(genome, cfg);
        let cand_fit = calc.fitness_of(&candidate);
        if cand_fit > parent_fitness {
            *genome = candidate;
            Some(cand_fit)
        } else {
            None
        }
    }

    /// Ungated apply (for the gradient-primary quality probe): apply gradient
    /// steps and keep the result unconditionally — no elitist gate. The caller
    /// tracks best-ever by hard ΔE2000 across calls.
    #[cfg(test)]
    pub(crate) fn polish_ungated(
        &mut self,
        genome: &mut Vec<crate::genome::Vertex>,
        cfg: &PolishCfg,
    ) {
        if genome.is_empty() {
            return;
        }
        *genome = self.apply_gradient_steps(genome, cfg);
    }
}

#[cfg(test)]
#[path = "gradient_tests.rs"]
mod gradient_tests;
