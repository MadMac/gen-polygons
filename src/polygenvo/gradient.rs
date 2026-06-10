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

/// Uniform params for the softraster forward pass. `#[repr(C)]` + Pod so
/// bytemuck can cast it straight to the 16-byte uniform buffer the shader reads.
#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftRasterParams {
    width: u32,
    height: u32,
    num_tris: u32,
    tau: f32,
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
    let mut flat: Vec<f32> = Vec::with_capacity(scene.len() * 18);
    for tri in scene {
        for vert in tri {
            for &comp in vert {
                flat.push(comp as f32);
            }
        }
    }
    // Ensure at least one element so the storage buffer is non-zero sized.
    if flat.is_empty() {
        flat.push(0.0);
    }

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
    let mut flat: Vec<f32> = Vec::with_capacity(scene.len() * 18);
    for tri in scene {
        for vert in tri {
            for &comp in vert {
                flat.push(comp as f32);
            }
        }
    }
    if flat.is_empty() {
        flat.push(0.0);
    }

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

    let grad_len = (num_tris.max(1) * 18) as u64;
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
}
