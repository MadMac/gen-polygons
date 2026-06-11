# Prefix-Sum Tile Binning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the differentiable kernel's per-pixel O(num_tris) reject scan with real per-tile triangle lists (count → prefix-sum → fill → per-tile sort), so each pixel iterates only the triangles in its tile — reaching interactive speed at the full triangle budget.

**Architecture:** A new `binning.wgsl` (count / scan / fill / sort_tiles compute entries) builds, each gradient step, a contiguous per-tile list of triangle indices in draw order. `softraster_tiled.wgsl`'s forward/backward iterate `tile_list[off..off+cnt]` instead of looping all triangles. The math is unchanged, so the existing GPU==CPU-oracle tests are the integration guard.

**Tech Stack:** Rust 2024, `wgpu` 29 (WGSL compute on Vulkan), `bytemuck`; no new deps.

**Parent spec:** `docs/superpowers/specs/2026-06-10-tile-binning-design.md`.

---

## Key facts (read first)

- **Run all tests single-threaded:** `cargo test --bin polygenvo -- --test-threads=1` (Vulkan SIGSEGVs under parallel test).
- `softraster_tiled.wgsl`: 16×16 tiles. `Params { width, height, num_tris, tau }` at binding 0; `tri_params` (binding 1); `forward` uses (0,1,2=state); `backward` uses (0,1,2=state,3=goal_lab,4=grad). Both have `for (var t: u32 = 0u; t < params.num_tris ...) { if (!tri_overlaps_aabb(base, aabb)) { continue; } ... }` (forward line ~126, backward line ~270). `pixel_to_clip`, `tile_clip_aabb`, `tri_overlaps_aabb`, `MARGIN_TAU=8.0`, `TILE=16u` already exist there.
- `gradient.rs`: `PolishState` caches pipelines/buffers; `polish` per-step loop (around lines 320-390) does: write uniforms → tiled forward dispatch → `clear_buffer(grad)` → tiled backward dispatch → adam. `#[cfg(test)]` helpers `gpu_forward_tiled_lab` / `gpu_grad_tiled` build their own pipelines+buffers and run the tiled passes; `SoftRasterParams { width,height,num_tris,tau }` (Pod); `flatten_scene`.
- **Tile id of pixel (px,py):** `tile = (py/16) * tiles_x + (px/16)`, `tiles_x = ceil(width/16)`.
- **Triangle → tile range** (must include exactly the tiles whose `tile_clip_aabb` (expanded by 8τ) overlaps the triangle bbox — equivalent to expanding the *triangle* clip-bbox by 8τ and converting to tiles). Pixel from clip: `px(cx) = (cx+1)/2*W - 0.5`; `py(cy) = (1-cy)/2*H - 0.5` (note y-flip: larger cy → smaller py). This mapping is the one risk area — a shared `tri_tile_range` helper is used by both `count` and `fill`.

---

## Task 1: `binning.wgsl` (count/scan/fill/sort) + binning unit test

**Files:**
- Create: `src/polygenvo/binning.wgsl`
- Modify: `src/polygenvo/gradient.rs` (`#[cfg(test)]` `gpu_bin` helper + test)

- [ ] **Step 1: Write `binning.wgsl`.**

```wgsl
// Prefix-sum tile binning for the differentiable kernel. Builds, per 16×16 tile,
// a contiguous list of triangle indices (draw order) so the soft-raster passes
// iterate only a tile's triangles. Re-run each gradient step (positions move).

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
@group(0) @binding(0) var<uniform> bp: BinParams;
@group(0) @binding(1) var<storage, read> tri_params: array<f32>;
@group(0) @binding(2) var<storage, read_write> tile_counts: array<atomic<u32>>; // count, then fill cursor
@group(0) @binding(3) var<storage, read_write> tile_offsets: array<u32>;          // exclusive scan; [num_tiles]=total
@group(0) @binding(4) var<storage, read_write> tile_list: array<u32>;
@group(0) @binding(5) var<storage, read_write> overflow: array<atomic<u32>>;      // [0] set if list_cap exceeded

const TILE: u32 = 16u;
const MARGIN_TAU: f32 = 8.0;

// Inclusive tile range (tx0,tx1,ty0,ty1) covered by triangle `base`'s clip bbox
// expanded by MARGIN_TAU*tau. Shared by count and fill so their sets match exactly.
fn tri_tile_range(base: u32) -> vec4<u32> {
    let x0 = tri_params[base + 0u]; let y0 = tri_params[base + 1u];
    let x1 = tri_params[base + 6u]; let y1 = tri_params[base + 7u];
    let x2 = tri_params[base + 12u]; let y2 = tri_params[base + 13u];
    let m = MARGIN_TAU * bp.tau;
    let cxmin = min(x0, min(x1, x2)) - m;
    let cxmax = max(x0, max(x1, x2)) + m;
    let cymin = min(y0, min(y1, y2)) - m;
    let cymax = max(y0, max(y1, y2)) + m;
    let w = f32(bp.width); let h = f32(bp.height);
    // clip -> pixel. x increases with cx; y increases as cy decreases.
    let pxmin = (cxmin + 1.0) * 0.5 * w - 0.5;
    let pxmax = (cxmax + 1.0) * 0.5 * w - 0.5;
    let pymin = (1.0 - cymax) * 0.5 * h - 0.5; // cymax (top) -> smallest py
    let pymax = (1.0 - cymin) * 0.5 * h - 0.5;
    let txi = clamp(i32(floor(pxmin)) / 16, 0, i32(bp.tiles_x) - 1);
    let txa = clamp(i32(floor(pxmax)) / 16, 0, i32(bp.tiles_x) - 1);
    let tyi = clamp(i32(floor(pymin)) / 16, 0, i32(bp.tiles_y) - 1);
    let tya = clamp(i32(floor(pymax)) / 16, 0, i32(bp.tiles_y) - 1);
    return vec4<u32>(u32(txi), u32(txa), u32(tyi), u32(tya));
}

@compute @workgroup_size(64)
fn count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= bp.num_tris) { return; }
    let r = tri_tile_range(t * 18u);
    for (var ty = r.z; ty <= r.w; ty = ty + 1u) {
        for (var tx = r.x; tx <= r.y; tx = tx + 1u) {
            atomicAdd(&tile_counts[ty * bp.tiles_x + tx], 1u);
        }
    }
}

// Single-workgroup exclusive scan over tile_counts -> tile_offsets. Each thread
// sums a serial chunk, threads share partial sums, then re-walk to write offsets.
@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = bp.tiles_x * bp.tiles_y;
    let nthreads = 256u;
    let chunk = (n + nthreads - 1u) / nthreads;
    let start = lid.x * chunk;
    let end = min(start + chunk, n);
    var partial: u32 = 0u;
    for (var i = start; i < end; i = i + 1u) { partial = partial + atomicLoad(&tile_counts[i]); }
    var shared: u32; // declared as workgroup var below; see note
    // (workgroup array<u32,256> `sh`; sh[lid]=partial; barrier; thread 0 exclusive-scans sh;
    //  barrier; base=sh[lid]; re-walk chunk writing tile_offsets[i]=base+running.)
    // Implement with a `var<workgroup> sh: array<u32, 256>;` at module scope.
    // After computing each thread's `base` (exclusive prefix of partials):
    var running = start_base_placeholder; // = base
    for (var i = start; i < end; i = i + 1u) {
        tile_offsets[i] = running;
        running = running + atomicLoad(&tile_counts[i]);
    }
    if (lid.x == nthreads - 1u) { tile_offsets[n] = running; } // total at [n] (size num_tiles+1)
}

@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= bp.num_tris) { return; }
    let r = tri_tile_range(t * 18u);
    for (var ty = r.z; ty <= r.w; ty = ty + 1u) {
        for (var tx = r.x; tx <= r.y; tx = tx + 1u) {
            let tile = ty * bp.tiles_x + tx;
            let slot = tile_offsets[tile] + atomicAdd(&tile_counts[tile], 1u); // counts reset to 0 before fill
            if (slot < bp.list_cap) { tile_list[slot] = t; }
            else { atomicStore(&overflow[0], 1u); }
        }
    }
}

// One workgroup per tile: insertion-sort the tile's slice ascending by triangle index.
@compute @workgroup_size(1)
fn sort_tiles(@builtin(workgroup_id) wid: vec3<u32>) {
    let tile = wid.x;
    let n = bp.tiles_x * bp.tiles_y;
    if (tile >= n) { return; }
    let off = tile_offsets[tile];
    let cnt = tile_offsets[tile + 1u] - off;
    for (var i = 1u; i < cnt; i = i + 1u) {
        let key = tile_list[off + i];
        var j = i;
        loop {
            if (j == 0u) { break; }
            if (tile_list[off + j - 1u] <= key) { break; }
            tile_list[off + j] = tile_list[off + j - 1u];
            j = j - 1u;
        }
        tile_list[off + j] = key;
    }
}
```

> Implementation notes for the `scan` entry (finish the sketch): add `var<workgroup> sh: array<u32, 256>;` at module scope. Body: compute `partial` (sum of the thread's chunk via `atomicLoad`); `sh[lid.x] = partial; workgroupBarrier();` then thread 0 does an in-place exclusive scan of `sh` (running sum) **and** stores the grand total; `workgroupBarrier();` `let base = sh[lid.x];` then re-walk the chunk writing `tile_offsets[i] = base + running` (running starts 0, += count each step). Last thread writes the total to `tile_offsets[n]`. Keep `tile_counts` intact during scan (read via `atomicLoad`), and have the host **reset `tile_counts` to 0 between scan and fill** (the fill reuses it as the cursor). `tile_offsets` is sized `num_tiles + 1`.

- [ ] **Step 2: Add a `#[cfg(test)] gpu_bin` helper + binning test in `gradient.rs`.** The helper builds BinParams + buffers, runs clear→count→scan→reset-counts→fill→sort_tiles, and reads back `tile_offsets` (len num_tiles+1) and `tile_list` (len = total). Test on a small scene; compare each tile's sorted slice to a CPU-replicated expected set (triangles whose clip-bbox+8τ overlaps the tile), and assert offsets are the exclusive prefix sum of counts.

```rust
    #[test]
    fn gpu_binning_matches_cpu_expectation() {
        use crate::softras_ref::ParamTri;
        use crate::test_support::init_test_wgpu;
        let w = 48u32; let h = 48u32; let tau = 0.05f32; // tiles_x = tiles_y = 3
        let scene: Vec<ParamTri> = vec![
            [[-0.9,-0.9,0.,0.,0.,1.],[-0.6,-0.9,0.,0.,0.,1.],[-0.9,-0.6,0.,0.,0.,1.]], // corner
            [[-0.2,-0.2,0.,0.,0.,1.],[0.3,-0.1,0.,0.,0.,1.],[0.0,0.3,0.,0.,0.,1.]],     // centre
        ];
        let (device, queue) = init_test_wgpu();
        let (offsets, list) = super::gpu_bin(&device, &queue, &scene, w, h, tau);
        let tiles_x = w.div_ceil(16); let tiles_y = h.div_ceil(16);
        // CPU expectation: per tile, indices whose clip-bbox+8τ overlaps tile clip-aabb.
        let expect = super::tests::cpu_expected_tile_lists(&scene, w, h, tau, tiles_x, tiles_y);
        for tile in 0..(tiles_x*tiles_y) as usize {
            let off = offsets[tile] as usize; let end = offsets[tile+1] as usize;
            let mut got: Vec<u32> = list[off..end].to_vec();
            // already sorted ascending; assert so:
            assert!(got.windows(2).all(|w| w[0] < w[1]), "tile {tile} not sorted ascending: {got:?}");
            got.sort_unstable();
            assert_eq!(got, expect[tile], "tile {tile} list mismatch");
        }
    }
```

Add a `cpu_expected_tile_lists` test helper that mirrors `tri_tile_range` in Rust (same clip→pixel→tile math) and returns `Vec<Vec<u32>>` (sorted) per tile.

- [ ] **Step 3: Run to fail, implement `gpu_bin`, run to pass.**
Run: `cargo test --bin polygenvo gpu_binning_matches -- --test-threads=1 --nocapture 2>&1 | tail -20`
Expected: PASS. If a tile is off by one row/col, the y-flip in `tri_tile_range`/`cpu_expected_tile_lists` disagrees; if offsets wrong, the scan exclusive/total handling is off.

- [ ] **Step 4: Full suite + clippy + commit.**
```bash
cargo test --bin polygenvo -- --test-threads=1 2>&1 | tail -5
cargo clippy --bin polygenvo 2>&1 | tail -5
git add src/polygenvo/binning.wgsl src/polygenvo/gradient.rs
git commit -m "feat: prefix-sum tile binning (count/scan/fill/sort), GPU==CPU lists"
```

---

## Task 2: forward/backward iterate `tile_list`; oracle tests still pass

**Files:**
- Modify: `src/polygenvo/softraster_tiled.wgsl`, `src/polygenvo/gradient.rs`

- [ ] **Step 1: Add tile bindings to `softraster_tiled.wgsl`** and extend `Params` with `tiles_x` (so the shader can find a pixel's tile). Add:
```wgsl
@group(0) @binding(5) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> tile_list: array<u32>;
```
and add `tiles_x: u32` to `Params` (after `tau`; update the Rust `SoftRasterParams` to match — add `tiles_x` and a pad if needed for 16-byte alignment; it becomes width,height,num_tris,tau,tiles_x,_pad,_pad,_pad → 32 bytes).

- [ ] **Step 2: Replace the triangle loop header in BOTH `forward` and `backward`.** Where each currently does:
```wgsl
    for (var t: u32 = 0u; t < params.num_tris; t = t + 1u) {
        let base = t * 18u;
        if (!tri_overlaps_aabb(base, aabb)) { continue; }
```
change to iterate the pixel's tile list:
```wgsl
    let tile = (py / 16u) * params.tiles_x + (px / 16u);
    let lo = tile_offsets[tile];
    let hi = tile_offsets[tile + 1u];
    for (var ii: u32 = lo; ii < hi; ii = ii + 1u) {
        let t = tile_list[ii];
        let base = t * 18u;
```
(Drop the `aabb`/`tri_overlaps_aabb` use; the list already restricts to overlapping triangles in draw order. `tile_clip_aabb`/`tri_overlaps_aabb` may become dead — remove them if unused, or leave if the listless path is kept for tests; prefer removing to avoid dead code.) The per-triangle composite/gradient body is otherwise unchanged. Verify the loop still maintains `below`/`prefix_trans`/`tprod` exactly as before (now over the tile's list, which is in draw order — same as iterating all triangles and skipping non-overlappers).

- [ ] **Step 3: Update the `#[cfg(test)]` `gpu_forward_tiled_lab` and `gpu_grad_tiled` helpers** to run the binning passes first (reuse the `gpu_bin` machinery / a shared `run_binning(encoder, ...)` helper) and bind `tile_offsets`+`tile_list` (bindings 5,6) to the forward/backward bind groups. They must produce the tile lists for the SAME scene before the forward/backward dispatch.

- [ ] **Step 4: Run the oracle equality tests through the binned path.**
Run: `cargo test --bin polygenvo gpu_tiled_forward_matches gpu_tiled_backward_matches -- --test-threads=1 --nocapture 2>&1 | tail -20`
Expected: BOTH still pass (forward Lab < 1e-2; backward rel < 2e-2). These are the integration guard — binning changed how the set is gathered, not the math. If they fail, the tile-id computation or the binding wiring is wrong (or the `tri_tile_range` set differs from the old `tri_overlaps_aabb` set — they must be equivalent).

- [ ] **Step 5: Full suite + clippy + commit.**
```bash
cargo test --bin polygenvo -- --test-threads=1 2>&1 | tail -5
cargo clippy --bin polygenvo 2>&1 | tail -5
git add src/polygenvo/softraster_tiled.wgsl src/polygenvo/gradient.rs
git commit -m "feat: tiled forward/backward iterate per-tile lists (binned), GPU==CPU holds"
```

---

## Task 3: wire binning into `PolishState::polish` + benchmark

**Files:**
- Modify: `src/polygenvo/gradient.rs`

- [ ] **Step 1: Add binning pipelines + buffers to `PolishState`.** In `new`: build `count`/`scan`/`fill`/`sort_tiles` pipelines from `include_str!("binning.wgsl")`; create `tile_counts` (num_tiles atomic u32), `tile_offsets` (num_tiles+1 u32), `tile_list` (capacity `LIST_CAP` — compute as `num_tiles * SOME_FACTOR` or a fixed generous value; document; STORAGE), `overflow` (1×u32, STORAGE|COPY_SRC for host check or just log), and a `bin_params_buf` (BinParams uniform). num_tiles from `width/height` (texture_size). Build the bind groups. Also add `tile_offsets`/`tile_list` to the forward/backward tiled bind groups (bindings 5,6).

- [ ] **Step 2: Run binning each step in `polish`.** Before the tiled forward dispatch, in the per-step encoder: write `bin_params_buf` (tau for this step); `clear_buffer(tile_counts)`; `count` dispatch (`num_tris.div_ceil(64)`); `scan` dispatch (1 workgroup); `clear_buffer(tile_counts)` again (reset cursor); `fill` dispatch (`num_tris.div_ceil(64)`); `sort_tiles` dispatch (`num_tiles` workgroups). Then the existing forward → clear grad → backward → adam. (All one encoder per step; the passes are ordered by the encoder.)

- [ ] **Step 3: Verify polish still works (now binned).**
Run: `cargo test --bin polygenvo gpu_polish_improves_hard_de2000 polish_gate -- --test-threads=1 --nocapture 2>&1 | tail -10`
Expected: both pass (polish improves hard ΔE2000; gate rejects no-op).

- [ ] **Step 4: Extend `bench_backend`** with binned polish at 512² with **1000 and 10000** triangles (replace/augment the Phase-2 tiled-polish loop). Keep `steps_n` small (e.g. 3) and τ sharp (0.03).
```rust
        for &ntris in &[1000usize, 10000] {
            let size = 512u32;
            let goal = make_solid_goal(size, [40, 120, 200]);
            let calc = FitnessCalc::new_for_test(device.clone(), queue.clone(), &goal, 1);
            let mut state = super::PolishState::new(&calc, &goal);
            let mut rng = StdRng::seed_from_u64(4);
            let mut g = init_genome(&goal, ntris, &mut rng);
            let parent = calc.fitness_of(&g);
            let cfg = super::PolishCfg { enabled: true, every_k: 1, steps_n: 3, lr: 0.05, tau_start: 0.03, tau_end: 0.03 };
            let t = Instant::now();
            let _ = state.polish(&mut g, parent, &calc, &cfg);
            println!("binned polish 512² ({ntris} tris): {:.1} ms/step", t.elapsed().as_secs_f64()*1000.0/3.0);
        }
```

- [ ] **Step 5: Run the benchmark, record results, decide Phase 3 readiness.**
Run: `cargo test --release --bin polygenvo bench_backend -- --ignored --test-threads=1 --nocapture 2>&1 | grep -E "backend:|binned polish"`
Append a "Phase 2.5 results" section to `docs/superpowers/specs/2026-06-10-tile-binning-design.md` with the ms/step at 1000 and 10000 tris and the speedup vs the listless ~1143 ms/step. Confirm interactive (target tens-of-ms; definitely not multi-second at 10k).
```bash
cargo test --bin polygenvo -- --test-threads=1 2>&1 | tail -5
cargo clippy --bin polygenvo 2>&1 | tail -5
git add src/polygenvo/gradient.rs docs/superpowers/specs/2026-06-10-tile-binning-design.md
git commit -m "feat: PolishState bins per step + benchmark; Phase 2.5 results"
```

- [ ] **Step 6: Update memory** — record the binned ms/step and Phase 3 readiness in `path-b-diff-rasterizer-status`.

---

## Self-review

- **Spec coverage:** count/scan/fill/sort (spec §Binning passes) → Task 1; forward/backward list iteration (§Forward/backward change) → Task 2; PolishState integration + per-step binning (§Integration) → Task 3; binning unit test + scan check (§Testing) → Task 1; oracle equality through binned path (§Testing primary guard) → Task 2 Step 4; benchmark at 1000/10000 (§Testing/Acceptance) → Task 3; overflow flag (§Risks) → Task 1 `overflow` + Task 3 buffer; shared count/fill mapping (§Risks) → `tri_tile_range` used by both. All spec points mapped.
- **Placeholder scan:** the `scan` WGSL is given as a sketch with a precise prose completion (shared-mem exclusive scan) — the engineer completes it against the binning unit test, which is the executable spec; all other code is complete. No TBD/TODO left as work-substitutes.
- **Type/name consistency:** `BinParams`/`tri_tile_range`/`tile_counts`/`tile_offsets`/`tile_list`/`overflow`, `gpu_bin`, `cpu_expected_tile_lists`, `SoftRasterParams` (now +tiles_x), `Params.tiles_x`, bindings 5=tile_offsets/6=tile_list on the tiled passes (5=tile_offsets/... in binning.wgsl is a separate bind group) — consistent within each shader's own binding space. NOTE: binning.wgsl and softraster_tiled.wgsl have independent binding numberings; both use `tile_offsets`/`tile_list` by name but in separate pipelines/bind groups.
