//! On-device differentiable-rasterizer polish: soft-raster forward+backward
//! (softraster.wgsl) + Adam (adam.wgsl) over all triangles' positions+colors,
//! minimizing Lab-MSE, then gated by the hard ΔE2000 renderer. Framework-free,
//! reuses FitnessCalc's wgpu device/queue. See
//! docs/superpowers/specs/2026-06-08-gpu-differentiable-rasterizer-design.md.

#[allow(dead_code)] // fields used in Task 8/9
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

/// Uniform params for the softraster forward pass. `#[repr(C)]` + Pod so
/// bytemuck can cast it straight to the 16-byte uniform buffer the shader reads.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftRasterParams {
    width: u32,
    height: u32,
    num_tris: u32,
    tau: f32,
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

/// Cached pipelines + buffers for the on-device gradient polish. Built once per
/// goal; `polish` then only writes the genome + uniforms and dispatches. The
/// `params_buf` doubles as backward's `tri_params` and Adam's `params`; the
/// `grad_buf` is written (atomic f32-via-u32) by backward and re-read as plain
/// f32 by Adam.
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
    backward_pipeline: wgpu::ComputePipeline,
    adam_pipeline: wgpu::ComputePipeline,
    backward_bind_group: wgpu::BindGroup,
    adam_bind_group: wgpu::BindGroup,
    readback_buf: wgpu::Buffer,
}

#[allow(dead_code)] // wired into the ES loop in Task 9
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

        let sr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Polish SoftRaster Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("softraster.wgsl").into()),
        });
        let adam_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Polish Adam Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("adam.wgsl").into()),
        });

        let backward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Polish Backward Pipeline"),
            layout: None,
            module: &sr_shader,
            entry_point: Some("backward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let adam_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Polish Adam Pipeline"),
            layout: None,
            module: &adam_shader,
            entry_point: Some("update"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // backward references bindings 0,1,3,4 — the auto layout includes only those.
        let backward_bgl = backward_pipeline.get_bind_group_layout(0);
        let backward_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Polish Backward Bind Group"),
            layout: &backward_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sr_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: goal_lab_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grad_buf.as_entire_binding() },
            ],
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
            backward_pipeline,
            adam_pipeline,
            backward_bind_group,
            adam_bind_group,
            readback_buf,
        }
    }

    /// Run `cfg.steps_n` backward+Adam steps fully on-device starting from
    /// `genome`, then keep the result only if the hard ΔE2000 renderer
    /// (`calc.fitness_of`) confirms it beats `parent_fitness`. On accept, mutates
    /// `genome` and returns `Some(new_fitness)`; otherwise leaves `genome`
    /// untouched and returns `None`.
    pub(crate) fn polish(
        &mut self,
        genome: &mut Vec<crate::genome::Vertex>,
        parent_fitness: usize,
        calc: &crate::fitness::FitnessCalc,
        cfg: &PolishCfg,
    ) -> Option<usize> {
        let n_verts = genome.len();
        if n_verts == 0 {
            return None;
        }
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
            let sr = SoftRasterParams { width: self.width, height: self.height, num_tris, tau };
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

            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Polish Step Encoder"),
            });
            enc.clear_buffer(&self.grad_buf, 0, None);
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Polish Backward Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.backward_pipeline);
                pass.set_bind_group(0, &self.backward_bind_group, &[]);
                pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
            }
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
            // submit, so each step sees its own uniforms. (Batching is the deferred tiled kernel.)
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

        let cand_fit = calc.fitness_of(&candidate);
        if cand_fit > parent_fitness {
            *genome = candidate;
            Some(cand_fit)
        } else {
            None
        }
    }
}

/// Flatten a scene (slice of `ParamTri`) into a `Vec<f32>` in the layout the
/// shaders expect (t*18 + k*6 + c). Pushes a dummy `0.0` when the scene is empty
/// so the resulting storage buffer is non-zero sized (wgpu/WGSL requirement).
#[cfg(test)]
fn flatten_scene(scene: &[crate::softras_ref::ParamTri]) -> Vec<f32> {
    let mut flat: Vec<f32> = Vec::with_capacity(scene.len() * 18);
    for tri in scene {
        for vert in tri {
            for &comp in vert {
                flat.push(comp as f32);
            }
        }
    }
    // at least one element so the storage buffer is non-zero sized
    if flat.is_empty() {
        flat.push(0.0);
    }
    flat
}

/// Run the `softraster.wgsl` forward pass on the GPU and return per-pixel Lab
/// as `Vec<[f32; 4]>` (L, a, b, 0), row-major (y * width + x). Test-only: the
/// production polish path (Task 8) never reads Lab back; this function exists
/// solely to prove GPU == CPU reference within 1e-2.
#[cfg(test)]
pub(crate) fn gpu_forward_lab(
    device: &std::sync::Arc<wgpu::Device>,
    queue: &std::sync::Arc<wgpu::Queue>,
    scene: &[crate::softras_ref::ParamTri],
    w: u32,
    h: u32,
    tau: f32,
) -> Vec<[f32; 4]> {
    use wgpu::util::DeviceExt;

    // 1. Flatten scene -> Vec<f32>: t*18 + k*6 + c.
    let num_tris = scene.len() as u32;
    let flat = flatten_scene(scene);

    // 2. Params uniform (16 bytes).
    let params = SoftRasterParams { width: w, height: h, num_tris, tau };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SoftRaster Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // 3. tri_params storage buffer (read).
    let tri_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SoftRaster TriParams"),
        contents: bytemuck::cast_slice(&flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // 4. out_lab storage buffer (read_write, w*h*16 bytes).
    let out_size = (w * h * 16) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SoftRaster OutLab"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 5. Readback buffer.
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SoftRaster Readback"),
        size: out_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // 6. Compute pipeline.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SoftRaster Forward Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("softraster.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SoftRaster Forward Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("forward"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // 7. Bind group from auto-derived layout.
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SoftRaster Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: tri_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    // 8. Encode + dispatch.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SoftRaster Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SoftRaster Forward Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = w.div_ceil(8);
        let wg_y = h.div_ceil(8);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &readback_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    // 9. Map + read back.
    let slice = readback_buf.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();

    let pixels: Vec<[f32; 4]> = {
        let data = slice.get_mapped_range();
        // Each pixel is 4 f32s = 16 bytes; bytemuck cast from &[u8] to &[f32].
        let floats: &[f32] = bytemuck::cast_slice(&data);
        floats
            .chunks_exact(4)
            .map(|ch| [ch[0], ch[1], ch[2], ch[3]])
            .collect()
    };
    readback_buf.unmap();
    pixels
}

/// Run the `softraster.wgsl` backward pass on the GPU and return the flat
/// per-param gradient (len num_tris*18, layout t*18 + k*6 + c). Test-only: the
/// production polish path feeds the grad buffer straight into Adam on-device;
/// this reads it back solely to prove GPU == CPU reference within rel 2e-2.
#[cfg(test)]
pub(crate) fn gpu_grad(
    device: &std::sync::Arc<wgpu::Device>,
    queue: &std::sync::Arc<wgpu::Queue>,
    scene: &[crate::softras_ref::ParamTri],
    goal_lab: &[[f32; 4]],
    w: u32,
    h: u32,
    tau: f32,
) -> Vec<f32> {
    use wgpu::util::DeviceExt;

    let num_tris = scene.len() as u32;
    let flat = flatten_scene(scene);

    let params = SoftRasterParams { width: w, height: h, num_tris, tau };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SoftRaster Grad Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let tri_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SoftRaster Grad TriParams"),
        contents: bytemuck::cast_slice(&flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let goal_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SoftRaster Grad GoalLab"),
        contents: bytemuck::cast_slice(goal_lab),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let grad_len = (num_tris.max(1) * 18) as u64; // .max(1): non-zero-sized buffer even for an empty scene
    let grad_size = grad_len * 4;
    let grad_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SoftRaster Grad Accum"),
        size: grad_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SoftRaster Grad Readback"),
        size: grad_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SoftRaster Backward Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("softraster.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SoftRaster Backward Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("backward"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // backward references bindings 0,1,3,4 — the auto layout includes only those.
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SoftRaster Grad Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tri_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: goal_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: grad_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SoftRaster Grad Encoder"),
    });
    encoder.clear_buffer(&grad_buf, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SoftRaster Backward Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&grad_buf, 0, &readback_buf, 0, grad_size);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let out: Vec<f32> = {
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        floats.to_vec()
    };
    readback_buf.unmap();
    out
}

#[cfg(test)]
mod tests {
    use crate::softras_ref::{forward_pixel_lab, ParamTri};
    use crate::test_support::init_test_wgpu;

    #[test]
    fn gpu_forward_matches_cpu_reference() {
        let w = 16u32;
        let h = 16u32;
        let tau = 0.1f64;
        let scene: Vec<ParamTri> = vec![
            [
                [-0.5, -0.5, 0.8, 0.2, 0.2, 0.9],
                [0.5, -0.5, 0.2, 0.8, 0.2, 0.9],
                [0.0, 0.6, 0.2, 0.2, 0.8, 0.9],
            ],
            [
                [-0.2, -0.2, 0.9, 0.9, 0.1, 0.6],
                [0.6, -0.1, 0.1, 0.9, 0.9, 0.6],
                [0.1, 0.5, 0.9, 0.1, 0.9, 0.6],
            ],
        ];
        let (device, queue) = init_test_wgpu();
        let gpu = super::gpu_forward_lab(&device, &queue, &scene, w, h, tau as f32);
        let mut maxdiff = 0.0f64;
        for py in 0..h {
            for px in 0..w {
                let cpu = forward_pixel_lab(&scene, px, py, w, h, tau);
                let g = gpu[(py * w + px) as usize];
                for ch in 0..3 {
                    maxdiff = maxdiff.max((cpu[ch] - g[ch] as f64).abs());
                }
            }
        }
        println!("GPU forward Lab vs CPU max diff: {maxdiff}");
        assert!(maxdiff < 1e-2, "GPU forward Lab vs CPU max diff {maxdiff} exceeds 1e-2");
    }

    #[test]
    fn gpu_backward_matches_cpu_reference() {
        use crate::softras_ref::grad_loss;
        let w = 12u32;
        let h = 12u32;
        let tau = 0.15f64;
        let scene: Vec<ParamTri> = vec![[
            [-0.4, -0.3, 0.7, 0.2, 0.6, 0.8],
            [0.5, -0.4, 0.2, 0.7, 0.3, 0.8],
            [0.1, 0.6, 0.4, 0.4, 0.9, 0.8],
        ]];
        let grey = crate::softras_ref::rgb_to_lab(0.5, 0.5, 0.5);
        let goal_f64: Vec<[f64; 3]> = (0..w * h).map(|_| grey).collect();
        let goal_f32: Vec<[f32; 4]> = goal_f64
            .iter()
            .map(|l| [l[0] as f32, l[1] as f32, l[2] as f32, 0.0])
            .collect();
        let (device, queue) = init_test_wgpu();
        let gpu = super::gpu_grad(&device, &queue, &scene, &goal_f32, w, h, tau as f32);
        let cpu = grad_loss(&scene, &goal_f64, w, h, tau);
        let mut maxrel = 0.0f64;
        for t in 0..scene.len() {
            for vert in 0..3 {
                for c in 0..6 {
                    let a = cpu[t][vert][c];
                    let b = gpu[t * 18 + vert * 6 + c] as f64;
                    let scale = a.abs().max(b.abs()).max(1e-4);
                    maxrel = maxrel.max((a - b).abs() / scale);
                }
            }
        }
        println!("GPU grad vs CPU max rel err: {maxrel}");
        assert!(maxrel < 2e-2, "GPU grad vs CPU max rel err {maxrel} exceeds 2e-2");
    }

    #[test]
    fn gpu_polish_improves_hard_de2000() {
        use crate::fitness::FitnessCalc;
        use crate::test_support::{init_test_wgpu, make_solid_goal};
        let size = 64u32;
        let goal = make_solid_goal(size, [50, 150, 230]);
        let (device, queue) = init_test_wgpu();
        let calc = FitnessCalc::new_for_test(device, queue, &goal, 1);
        let mut state = super::PolishState::new(&calc, &goal);
        // stuck small corner triangle, ~goal colour
        let mut genome = vec![
            crate::genome::Vertex { position: [-0.9, -0.9, 0.0], color: [0.196, 0.588, 0.902, 1.0] },
            crate::genome::Vertex { position: [-0.6, -0.9, 0.0], color: [0.196, 0.588, 0.902, 1.0] },
            crate::genome::Vertex { position: [-0.9, -0.6, 0.0], color: [0.196, 0.588, 0.902, 1.0] },
        ];
        let parent = calc.fitness_of(&genome);
        let before = genome.clone();
        let cfg = super::PolishCfg { enabled: true, every_k: 1, steps_n: 80, lr: 0.05, tau_start: 0.3, tau_end: 0.02 };
        let kept = state.polish(&mut genome, parent, &calc, &cfg);
        let newfit = kept.expect("polish should improve hard fitness on the stuck triangle");
        println!("hard fitness before={parent} after={newfit}");
        assert!(newfit > parent, "kept fitness {newfit} must beat parent {parent}");
        assert!(genome != before, "kept polish must mutate the genome");
    }

    #[test]
    fn polish_gate_rejects_noop_and_leaves_genome_unchanged() {
        use crate::fitness::FitnessCalc;
        use crate::test_support::{init_test_wgpu, make_solid_goal};
        let size = 64u32;
        let goal = make_solid_goal(size, [50, 150, 230]);
        let (device, queue) = init_test_wgpu();
        let calc = FitnessCalc::new_for_test(device, queue, &goal, 1);
        let mut state = super::PolishState::new(&calc, &goal);
        let mut genome = vec![
            crate::genome::Vertex { position: [-0.9, -0.9, 0.0], color: [0.196, 0.588, 0.902, 1.0] },
            crate::genome::Vertex { position: [-0.6, -0.9, 0.0], color: [0.196, 0.588, 0.902, 1.0] },
            crate::genome::Vertex { position: [-0.9, -0.6, 0.0], color: [0.196, 0.588, 0.902, 1.0] },
        ];
        let parent = calc.fitness_of(&genome);
        let before = genome.clone();
        // steps_n = 0: no optimization happens, candidate == parent, gate must reject.
        let cfg = super::PolishCfg { enabled: true, every_k: 1, steps_n: 0, lr: 0.05, tau_start: 0.3, tau_end: 0.02 };
        let kept = state.polish(&mut genome, parent, &calc, &cfg);
        assert!(kept.is_none(), "no-op polish must be rejected by the gate");
        assert_eq!(genome, before, "rejected polish must leave the genome byte-identical");
    }
}
