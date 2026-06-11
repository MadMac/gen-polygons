use super::*;

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

/// Run the `softraster_tiled.wgsl` forward pass on the GPU (16×16 workgroup-per-tile)
/// and return per-pixel Lab as `Vec<[f32; 4]>` (L, a, b, 0), row-major. Test-only:
/// exists solely to prove GPU tiled forward == CPU oracle within 1e-2 Lab.
#[cfg(test)]
pub(crate) fn gpu_forward_tiled_lab(
    device: &std::sync::Arc<wgpu::Device>,
    queue: &std::sync::Arc<wgpu::Queue>,
    scene: &[crate::softras_ref::ParamTri],
    w: u32,
    h: u32,
    tau: f32,
) -> Vec<[f32; 4]> {
    use wgpu::util::DeviceExt;

    let num_tris = scene.len() as u32;
    let flat = flatten_scene(scene);
    let tiles_x = w.div_ceil(16);
    let tiles_y = h.div_ceil(16);
    let num_tiles = (tiles_x * tiles_y) as u64;
    let list_cap = (num_tris.max(1) * (tiles_x * tiles_y)).max(1);

    // params uniform
    let params = SoftRasterParams {
        width: w,
        height: h,
        num_tris,
        tau,
        tiles_x,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("TiledFwd Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // tri_params storage (read)
    let tri_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("TiledFwd TriParams"),
        contents: bytemuck::cast_slice(&flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Binning: build the per-tile lists for this scene before the forward.
    let bin = BinResources::new(device, &tri_buf, num_tiles, list_cap);
    bin.write_params(queue, BinDims { num_tris, tiles_x, tiles_y, width: w, height: h, tau });

    // state storage: vec4<f32> per pixel = (c_full.rgb, T_final)
    let state_size = (w * h * 16) as u64;
    let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TiledFwd State"),
        size: state_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TiledFwd Readback"),
        size: state_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // pipeline from softraster_tiled.wgsl, entry "forward"
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("TiledFwd Shader"),
        source: wgpu::ShaderSource::Wgsl(
            crate::gpu::with_color_prelude(include_str!("softraster_tiled.wgsl")).into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("TiledFwd Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("forward"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bgl = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("TiledFwd Bind Group"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tri_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: state_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bin.offsets_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bin.list_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("TiledFwd Encoder"),
    });
    // Populate tile_offsets/tile_list for this scene before the forward.
    bin.record(&mut encoder, num_tris, tiles_x, tiles_y);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("TiledFwd Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
    }
    encoder.copy_buffer_to_buffer(&state_buf, 0, &readback_buf, 0, state_size);
    queue.submit(std::iter::once(encoder.finish()));

    // read back
    let slice = readback_buf.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| { sender.send(result).ok(); });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();

    // Convert each pixel's (lin_rgb, _) to Lab via softras_ref::lin_rgb_to_lab.
    let pixels: Vec<[f32; 4]> = {
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        floats
            .chunks_exact(4)
            .map(|ch| {
                let lab = crate::softras_ref::lin_rgb_to_lab(
                    ch[0] as f64,
                    ch[1] as f64,
                    ch[2] as f64,
                );
                [lab[0] as f32, lab[1] as f32, lab[2] as f32, 0.0]
            })
            .collect()
    };
    readback_buf.unmap();
    pixels
}

/// Run the `softraster_tiled.wgsl` forward+backward passes on the GPU and return
/// the flat per-param gradient (len num_tris*18, layout t*18 + k*6 + c). The
/// tiled backward needs the forward's `state`, so both run in one encoder:
/// forward (bindings 0,1,2) populates `state`, then `clear_buffer(grad)`, then
/// backward (bindings 0,1,2,3,4). Test-only: proves the tiled O(num_tris) grad
/// matches the CPU reference within rel 2e-2.
#[cfg(test)]
pub(crate) fn gpu_grad_tiled(
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
    let tiles_x = w.div_ceil(16);
    let tiles_y = h.div_ceil(16);
    let num_tiles = (tiles_x * tiles_y) as u64;
    let list_cap = (num_tris.max(1) * (tiles_x * tiles_y)).max(1);

    let params = SoftRasterParams {
        width: w,
        height: h,
        num_tris,
        tau,
        tiles_x,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("TiledGrad Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let tri_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("TiledGrad TriParams"),
        contents: bytemuck::cast_slice(&flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Binning: build the per-tile lists for this scene before forward+backward.
    let bin = BinResources::new(device, &tri_buf, num_tiles, list_cap);
    bin.write_params(queue, BinDims { num_tris, tiles_x, tiles_y, width: w, height: h, tau });
    let goal_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("TiledGrad GoalLab"),
        contents: bytemuck::cast_slice(goal_lab),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Per-pixel forward state: vec4<f32> = (c_full.rgb, T_final).
    let state_size = (w * h * 16) as u64;
    let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TiledGrad State"),
        size: state_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let grad_len = (num_tris.max(1) * 18) as u64; // .max(1): non-zero-sized buffer even for an empty scene
    let grad_size = grad_len * 4;
    let grad_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TiledGrad Accum"),
        size: grad_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TiledGrad Readback"),
        size: grad_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Tiled Shader"),
        source: wgpu::ShaderSource::Wgsl(
            crate::gpu::with_color_prelude(include_str!("softraster_tiled.wgsl")).into(),
        ),
    });
    let fwd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Tiled Forward Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("forward"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bwd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Tiled Backward Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("backward"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // forward bindings: 0 params, 1 tri_params, 2 state.
    let fwd_bgl = fwd_pipeline.get_bind_group_layout(0);
    let fwd_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Tiled Forward Bind Group"),
        layout: &fwd_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tri_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: state_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bin.offsets_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bin.list_buf.as_entire_binding() },
        ],
    });
    // backward bindings: 0 params, 1 tri_params, 2 state, 3 goal_lab, 4 grad, 5 offsets, 6 list.
    let bwd_bgl = bwd_pipeline.get_bind_group_layout(0);
    let bwd_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Tiled Backward Bind Group"),
        layout: &bwd_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tri_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: state_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: goal_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: grad_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bin.offsets_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bin.list_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("TiledGrad Encoder"),
    });
    // Populate tile_offsets/tile_list for this scene before forward+backward.
    bin.record(&mut encoder, num_tris, tiles_x, tiles_y);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tiled Forward Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&fwd_pipeline);
        pass.set_bind_group(0, &fwd_bg, &[]);
        pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
    }
    encoder.clear_buffer(&grad_buf, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tiled Backward Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&bwd_pipeline);
        pass.set_bind_group(0, &bwd_bg, &[]);
        pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
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

/// Run the full `binning.wgsl` pipeline (clear → count → scan → reset-counts →
/// fill → sort_tiles) on the GPU for `scene` and read back `(tile_offsets, tile_list)`.
/// `tile_offsets` has length `num_tiles + 1` (exclusive prefix sum, total at [n]);
/// the returned `tile_list` is truncated to `offsets[num_tiles]` (the live entries).
/// Test-only: proves the binned per-tile lists match the CPU expectation.
#[cfg(test)]
pub(crate) fn gpu_bin(
    device: &std::sync::Arc<wgpu::Device>,
    queue: &std::sync::Arc<wgpu::Queue>,
    scene: &[crate::softras_ref::ParamTri],
    w: u32,
    h: u32,
    tau: f32,
) -> (Vec<u32>, Vec<u32>) {
    use wgpu::util::DeviceExt;

    let num_tris = scene.len() as u32;
    let flat = flatten_scene(scene);
    let tiles_x = w.div_ceil(16);
    let tiles_y = h.div_ceil(16);
    let num_tiles = (tiles_x * tiles_y) as u64;
    // Generous list capacity; for the test's tiny scene this is small but safe.
    let list_cap = (num_tris.max(1) * (tiles_x * tiles_y)).max(1);

    let tri_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Bin TriParams"),
        contents: bytemuck::cast_slice(&flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bin = BinResources::new(device, &tri_buf, num_tiles, list_cap);
    bin.write_params(queue, BinDims { num_tris, tiles_x, tiles_y, width: w, height: h, tau });

    let offsets_len = num_tiles + 1;
    let off_bytes = offsets_len * 4;
    let list_bytes = (list_cap as u64) * 4;
    let off_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Bin Offsets Readback"),
        size: off_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let list_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Bin List Readback"),
        size: list_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let overflow_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Bin Overflow Readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Binning Encoder"),
    });
    bin.record(&mut encoder, num_tris, tiles_x, tiles_y);
    encoder.copy_buffer_to_buffer(&bin.offsets_buf, 0, &off_readback, 0, off_bytes);
    encoder.copy_buffer_to_buffer(&bin.list_buf, 0, &list_readback, 0, list_bytes);
    encoder.copy_buffer_to_buffer(&bin.overflow_buf, 0, &overflow_readback, 0, 4);
    queue.submit(std::iter::once(encoder.finish()));

    // Read back offsets.
    let offsets: Vec<u32> = {
        let slice = off_readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).ok();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let v: &[u32] = bytemuck::cast_slice(&data);
        v.to_vec()
    };
    off_readback.unmap();

    let total = offsets[num_tiles as usize] as usize;
    let list: Vec<u32> = {
        let slice = list_readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).ok();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let v: &[u32] = bytemuck::cast_slice(&data);
        v[..total].to_vec()
    };
    list_readback.unmap();

    // Assert the tile_list never exceeded its allocated capacity.
    let overflow_val: u32 = {
        let slice = overflow_readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).ok();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let v: &[u32] = bytemuck::cast_slice(&data);
        v[0]
    };
    overflow_readback.unmap();
    assert_eq!(overflow_val, 0, "tile_list capacity ({list_cap}) overflowed during binning");

    (offsets, list)
}

#[cfg(test)]
mod tests {
    use crate::softras_ref::{forward_pixel_lab, ParamTri};
    use crate::test_support::init_test_wgpu;

    /// Rust mirror of `binning.wgsl`'s `tri_tile_range` + count/fill: for each tile,
    /// the (sorted) set of triangle indices whose clip-bbox+8τ overlaps that tile.
    /// MUST use identical clip→pixel→tile math as the shader, or the test is moot.
    pub(crate) fn cpu_expected_tile_lists(
        scene: &[ParamTri],
        w: u32,
        h: u32,
        tau: f32,
        tiles_x: u32,
        tiles_y: u32,
    ) -> Vec<Vec<u32>> {
        const MARGIN_TAU: f32 = 8.0;
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); (tiles_x * tiles_y) as usize];
        for (t, tri) in scene.iter().enumerate() {
            let (x0, y0) = (tri[0][0] as f32, tri[0][1] as f32);
            let (x1, y1) = (tri[1][0] as f32, tri[1][1] as f32);
            let (x2, y2) = (tri[2][0] as f32, tri[2][1] as f32);
            let m = MARGIN_TAU * tau;
            let cxmin = x0.min(x1).min(x2) - m;
            let cxmax = x0.max(x1).max(x2) + m;
            let cymin = y0.min(y1).min(y2) - m;
            let cymax = y0.max(y1).max(y2) + m;
            let wf = w as f32;
            let hf = h as f32;
            // clip -> pixel (matches tri_tile_range exactly, including y-flip).
            let pxmin = (cxmin + 1.0) * 0.5 * wf - 0.5;
            let pxmax = (cxmax + 1.0) * 0.5 * wf - 0.5;
            let pymin = (1.0 - cymax) * 0.5 * hf - 0.5; // cymax (top) -> smallest py
            let pymax = (1.0 - cymin) * 0.5 * hf - 0.5;
            let txi = (pxmin.floor() as i32 / 16).clamp(0, tiles_x as i32 - 1);
            let txa = (pxmax.floor() as i32 / 16).clamp(0, tiles_x as i32 - 1);
            let tyi = (pymin.floor() as i32 / 16).clamp(0, tiles_y as i32 - 1);
            let tya = (pymax.floor() as i32 / 16).clamp(0, tiles_y as i32 - 1);
            for ty in tyi..=tya {
                for tx in txi..=txa {
                    let tile = (ty as u32) * tiles_x + tx as u32;
                    out[tile as usize].push(t as u32);
                }
            }
        }
        for cell in &mut out {
            cell.sort_unstable();
        }
        out
    }

    #[test]
    fn gpu_binning_matches_cpu_expectation() {
        let w = 48u32;
        let h = 48u32;
        let tau = 0.05f32; // tiles_x = tiles_y = 3
        let scene: Vec<ParamTri> = vec![
            [[-0.9, -0.9, 0., 0., 0., 1.], [-0.6, -0.9, 0., 0., 0., 1.], [-0.9, -0.6, 0., 0., 0., 1.]], // corner
            [[-0.2, -0.2, 0., 0., 0., 1.], [0.3, -0.1, 0., 0., 0., 1.], [0.0, 0.3, 0., 0., 0., 1.]], // centre
        ];
        let (device, queue) = init_test_wgpu();
        let (offsets, list) = super::gpu_bin(&device, &queue, &scene, w, h, tau);
        let tiles_x = w.div_ceil(16);
        let tiles_y = h.div_ceil(16);
        // CPU expectation: per tile, indices whose clip-bbox+8τ overlaps tile.
        let expect = cpu_expected_tile_lists(&scene, w, h, tau, tiles_x, tiles_y);
        assert_eq!(offsets[0], 0, "offsets[0] must be 0 (exclusive scan)");
        for tile in 0..(tiles_x * tiles_y) as usize {
            let off = offsets[tile] as usize;
            let end = offsets[tile + 1] as usize;
            // offsets are the exclusive prefix sum: per-tile count == expected count.
            assert_eq!(
                end - off,
                expect[tile].len(),
                "tile {tile} count {} != expected {}",
                end - off,
                expect[tile].len()
            );
            let mut got: Vec<u32> = list[off..end].to_vec();
            // already sorted strictly ascending; assert so:
            assert!(
                got.windows(2).all(|w| w[0] < w[1]),
                "tile {tile} not sorted ascending: {got:?}"
            );
            got.sort_unstable();
            assert_eq!(got, expect[tile], "tile {tile} list mismatch");
        }
    }

    #[test]
    fn gpu_tiled_forward_matches_cpu_reference() {
        let w = 40u32; let h = 40u32; let tau = 0.15f64; // >16px so triangles span tiles
        let scene: Vec<ParamTri> = vec![
            [[-0.7, -0.7, 0.8, 0.2, 0.2, 0.8], [0.7, -0.6, 0.2, 0.8, 0.2, 0.8], [0.0, 0.7, 0.2, 0.2, 0.8, 0.8]],
            [[-0.2, -0.2, 0.9, 0.9, 0.1, 0.6], [0.6, -0.1, 0.1, 0.9, 0.9, 0.6], [0.1, 0.5, 0.9, 0.1, 0.9, 0.6]],
        ];
        let (device, queue) = init_test_wgpu();
        let gpu = super::gpu_forward_tiled_lab(&device, &queue, &scene, w, h, tau as f32);
        let mut maxdiff = 0.0f64;
        for py in 0..h { for px in 0..w {
            let cpu = forward_pixel_lab(&scene, px, py, w, h, tau);
            let g = gpu[(py * w + px) as usize];
            for ch in 0..3 { maxdiff = maxdiff.max((cpu[ch] - g[ch] as f64).abs()); }
        }}
        println!("tiled forward GPU vs CPU max Lab diff: {maxdiff}");
        assert!(maxdiff < 1e-2, "tiled forward Lab vs CPU max diff {maxdiff} exceeds 1e-2");
    }

    #[test]
    fn gpu_tiled_backward_matches_cpu_reference() {
        use crate::softras_ref::{grad_loss, rgb_to_lab, ParamTri};
        use crate::test_support::init_test_wgpu;
        let (device, queue) = init_test_wgpu();
        // Two scenes: single triangle (FD scene); two overlapping, tile-spanning.
        let scenes: Vec<(u32, u32, f64, Vec<ParamTri>)> = vec![
            (12, 12, 0.15, vec![[[-0.4,-0.3,0.7,0.2,0.6,0.8],[0.5,-0.4,0.2,0.7,0.3,0.8],[0.1,0.6,0.4,0.4,0.9,0.8]]]),
            (40, 40, 0.12, vec![
                [[-0.6,-0.6,0.8,0.2,0.2,0.7],[0.6,-0.5,0.2,0.8,0.2,0.7],[0.0,0.6,0.2,0.2,0.8,0.7]],
                [[-0.3,-0.2,0.9,0.9,0.1,0.6],[0.5,-0.1,0.1,0.9,0.9,0.6],[0.0,0.5,0.9,0.1,0.9,0.6]],
            ]),
        ];
        for (w, h, tau, scene) in scenes {
            let grey = rgb_to_lab(0.5, 0.5, 0.5);
            let goal_f64: Vec<[f64;3]> = (0..w*h).map(|_| grey).collect();
            let goal_f32: Vec<[f32;4]> = goal_f64.iter().map(|l| [l[0] as f32,l[1] as f32,l[2] as f32,0.0]).collect();
            let gpu = super::gpu_grad_tiled(&device, &queue, &scene, &goal_f32, w, h, tau as f32);
            let cpu = grad_loss(&scene, &goal_f64, w, h, tau);
            let mut maxrel = 0.0f64;
            for t in 0..scene.len() { for vert in 0..3 { for c in 0..6 {
                let a = cpu[t][vert][c]; let b = gpu[t*18+vert*6+c] as f64;
                let scale = a.abs().max(b.abs()).max(1e-4);
                maxrel = maxrel.max((a-b).abs()/scale);
            }}}
            println!("tiled grad GPU vs CPU ({w}x{h}) max rel err: {maxrel}");
            assert!(maxrel < 2e-2, "tiled grad vs CPU ({w}x{h}) max rel {maxrel} exceeds 2e-2");
        }
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
    #[ignore = "benchmark; run with --release -- --ignored --nocapture"]
    fn bench_backend() {
        use crate::fitness::FitnessCalc;
        use crate::genome::init_genome;
        use crate::test_support::make_solid_goal;
        use rand::{rngs::StdRng, SeedableRng};
        use std::time::Instant;

        let (device, queue) = crate::test_support::init_test_wgpu();
        println!("--- bench_backend (set POLYGENVO_BACKEND=gl|vulkan to compare) ---");

        // Fitness scoring throughput at each resolution (the core ES per-step cost).
        for &size in &[128u32, 256, 512] {
            let goal = make_solid_goal(size, [40, 120, 200]);
            let calc = FitnessCalc::new_for_test(device.clone(), queue.clone(), &goal, 1);
            let mut rng = StdRng::seed_from_u64(1);
            let g = init_genome(&goal, 200, &mut rng);
            let _ = calc.fitness_of_batch(&[g.as_slice()]); // warmup
            let iters = 50;
            let t = Instant::now();
            for _ in 0..iters {
                let _ = calc.fitness_of_batch(&[g.as_slice()]);
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            println!("fitness {size}² (200 tris): {ms:.3} ms/score");
        }

        // Binned polish at the Phase-2.5 target: 512², sharp τ=0.03 (the late-stage
        // refinement regime), at 1000 and 3000 triangles. Compare against the
        // Phase-2 listless tiled kernel (1143 ms/step at 512²/1000-tris, sharp τ).
        // Both counts' triangle-tile incidences fit list_cap (num_tiles*1024 ≈ 1.05M
        // at 512²): 1000 large tris ≈ 290k, 3000 ≈ 870k — no overflow, valid timing.
        // `init_genome` makes LARGE triangles (radius ~0.3) that cover most tiles —
        // unrealistic for a refined genome and it defeats tiling/binning. `shrink`
        // pulls each triangle's vertices toward its centroid to simulate the small
        // triangles a real late-stage (split-refined) genome has, where binning's
        // per-tile reduction actually applies.
        let shrink = |g: &mut Vec<crate::genome::Vertex>, f: f32| {
            for tri in g.chunks_mut(3) {
                let cx = (tri[0].position[0] + tri[1].position[0] + tri[2].position[0]) / 3.0;
                let cy = (tri[0].position[1] + tri[1].position[1] + tri[2].position[1]) / 3.0;
                for v in tri.iter_mut() {
                    v.position[0] = cx + (v.position[0] - cx) * f;
                    v.position[1] = cy + (v.position[1] - cy) * f;
                }
            }
        };
        for &(ntris, f, label) in &[(1000usize, 1.0f32, "large"), (1000, 0.15, "small")] {
            let size = 512u32;
            let goal = make_solid_goal(size, [40, 120, 200]);
            let calc = FitnessCalc::new_for_test(device.clone(), queue.clone(), &goal, 1);
            let mut state = super::PolishState::new(&calc, &goal);
            let mut rng = StdRng::seed_from_u64(3);
            let mut g = init_genome(&goal, ntris, &mut rng);
            shrink(&mut g, f);
            let parent = calc.fitness_of(&g);
            let cfg = super::PolishCfg {
                enabled: true, every_k: 1, steps_n: 3, lr: 0.05, tau_start: 0.03, tau_end: 0.03,
            };
            let t = Instant::now();
            let _ = state.polish(&mut g, parent, &calc, &cfg);
            println!("binned polish 512² ({ntris} tris, {label}, sharp τ=0.03): {:.1} ms/step", t.elapsed().as_secs_f64() * 1000.0 / 3.0);
        }
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
