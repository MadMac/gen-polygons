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
    let txi = clamp(i32(floor(pxmin)) / i32(TILE), 0, i32(bp.tiles_x) - 1);
    let txa = clamp(i32(floor(pxmax)) / i32(TILE), 0, i32(bp.tiles_x) - 1);
    let tyi = clamp(i32(floor(pymin)) / i32(TILE), 0, i32(bp.tiles_y) - 1);
    let tya = clamp(i32(floor(pymax)) / i32(TILE), 0, i32(bp.tiles_y) - 1);
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
var<workgroup> sh: array<u32, 256>;

@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = bp.tiles_x * bp.tiles_y;
    let nthreads = 256u;
    let chunk = (n + nthreads - 1u) / nthreads;
    let start = lid.x * chunk;
    let end = min(start + chunk, n);
    var partial: u32 = 0u;
    for (var i = start; i < end; i = i + 1u) { partial = partial + atomicLoad(&tile_counts[i]); }
    sh[lid.x] = partial;
    workgroupBarrier();
    // Thread 0 does an in-place exclusive scan of the per-thread partials.
    if (lid.x == 0u) {
        var acc: u32 = 0u;
        for (var k = 0u; k < nthreads; k = k + 1u) {
            let v = sh[k];
            sh[k] = acc;
            acc = acc + v;
        }
    }
    workgroupBarrier();
    // Each thread re-walks its chunk writing exclusive offsets, based at sh[lid].
    let base = sh[lid.x];
    var running: u32 = base;
    for (var i = start; i < end; i = i + 1u) {
        tile_offsets[i] = running;
        running = running + atomicLoad(&tile_counts[i]);
    }
    // Last thread writes the grand total at [n] (tile_offsets is sized num_tiles+1).
    if (lid.x == nthreads - 1u) { tile_offsets[n] = running; }
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

// TODO(perf): @workgroup_size(1) serial insertion sort per tile — fine for small tiles; replace with a parallel sort if per-tile counts grow large.
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
