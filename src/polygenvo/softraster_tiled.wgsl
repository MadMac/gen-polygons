// Tiled differentiable soft-rasterizer. Each pixel iterates only its tile's
// triangle list (built by binning.wgsl: tile_offsets/tile_list, draw order). The
// forward entry stores per-pixel (c_full.rgb, T_final) into a `state` storage
// buffer so the backward can reconstruct prefix/suffix state in a single
// O(tile_count) walk instead of the O(num_tris²) brute-force.

struct Params {
    width: u32,
    height: u32,
    num_tris: u32,
    tau: f32,
    tiles_x: u32,
}
@group(0) @binding(0) var<uniform> params: Params;

// Triangle params, tightly packed f32: triangle t, vertex k, component c lives
// at index t*18 + k*6 + c. Per-vertex component layout: [cx, cy, r, g, b, a].
@group(0) @binding(1) var<storage, read> tri_params: array<f32>;

// Per-pixel forward state for the backward: (c_full.rgb, T_final).
@group(0) @binding(2) var<storage, read_write> state: array<vec4<f32>>;

// Goal CIELAB as [L, a, b, 0], row-major. Read by `backward`.
@group(0) @binding(3) var<storage, read> goal_lab: array<vec4<f32>>;

// Per-param gradient accumulator: array<atomic<u32>> of length num_tris*18,
// holding f32 gradients bit-cast to u32, layout t*18 + k*6 + c. Read_write by
// `backward`. Cleared to 0 by the host before the pass.
@group(0) @binding(4) var<storage, read_write> grad: array<atomic<u32>>;

// Per-tile triangle lists from binning.wgsl. `tile_offsets[tile..tile+1]` bounds
// the slice of `tile_list` (triangle indices, draw order) for this tile.
@group(0) @binding(5) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> tile_list: array<u32>;

const TILE: u32 = 16u;

// ---------------------------------------------------------------------------
// Soft-raster helpers. linear_rgb_to_xyz / xyz_to_lab come from the prepended
// color.wgsl prelude (gpu::with_color_prelude); srgb_to_linear is local.
// ---------------------------------------------------------------------------

fn srgb_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) { return c / 12.92; }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn pixel_to_clip(px: u32, py: u32) -> vec2<f32> {
    let cx = (f32(px) + 0.5) / f32(params.width) * 2.0 - 1.0;
    let cy = 1.0 - (f32(py) + 0.5) / f32(params.height) * 2.0;
    return vec2<f32>(cx, cy);
}

fn edge_sd(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let e = b - a;
    let len = length(e);
    if (len == 0.0) { return -3.4e38; }
    return ((-e.y) * (p.x - a.x) + e.x * (p.y - a.y)) / len;
}

// ---------------------------------------------------------------------------
// Forward entry: composite overlapping triangles, store (c_full.rgb, T_final).
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16, 1)
fn forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x; let py = gid.y;
    if (px >= params.width || py >= params.height) { return; }
    let p = pixel_to_clip(px, py);
    var c = vec3<f32>(0.0, 0.0, 0.0);
    var tprod = 1.0; // running Π(1 - src_a_clamped) over considered triangles
    let tile = (py / TILE) * params.tiles_x + (px / TILE);
    let lo = tile_offsets[tile];
    let hi = tile_offsets[tile + 1u];
    for (var ii: u32 = lo; ii < hi; ii = ii + 1u) {
        let t = tile_list[ii];
        let base = t * 18u;
        // --- forward per-pixel locals block ---
        let v0 = vec2<f32>(tri_params[base + 0u], tri_params[base + 1u]);
        let v1 = vec2<f32>(tri_params[base + 6u], tri_params[base + 7u]);
        let v2 = vec2<f32>(tri_params[base + 12u], tri_params[base + 13u]);
        let d0 = edge_sd(p, v0, v1);
        let d1 = edge_sd(p, v1, v2);
        let d2 = edge_sd(p, v2, v0);
        let d = min(d0, min(d1, d2));
        let cov = 1.0 / (1.0 + exp(-d / params.tau));
        let det = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
        var l0 = 1.0 / 3.0;
        var l1 = 1.0 / 3.0;
        var l2 = 1.0 / 3.0;
        if (abs(det) >= 1e-12) {
            l0 = ((v1.y - v2.y) * (p.x - v2.x) + (v2.x - v1.x) * (p.y - v2.y)) / det;
            l1 = ((v2.y - v0.y) * (p.x - v2.x) + (v0.x - v2.x) * (p.y - v2.y)) / det;
            l2 = 1.0 - l0 - l1;
        }
        let r = l0 * tri_params[base + 2u]  + l1 * tri_params[base + 8u]  + l2 * tri_params[base + 14u];
        let g = l0 * tri_params[base + 3u]  + l1 * tri_params[base + 9u]  + l2 * tri_params[base + 15u];
        let b = l0 * tri_params[base + 4u]  + l1 * tri_params[base + 10u] + l2 * tri_params[base + 16u];
        let a = l0 * tri_params[base + 5u]  + l1 * tri_params[base + 11u] + l2 * tri_params[base + 17u];
        let src_a = cov * a;
        let lin = vec3<f32>(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b));
        // --- end verbatim block ---
        let src_a_clamped = min(src_a, 0.999); // cap keeps prefix_trans > 0 (backward divides T_final/prefix_trans)
        c = src_a * lin + (1.0 - src_a) * c;           // composite uses TRUE src_a
        tprod = tprod * (1.0 - src_a_clamped);         // transmittance uses CLAMPED src_a
    }
    state[py * params.width + px] = vec4<f32>(c, tprod);
}

// ---------------------------------------------------------------------------
// Backward gradient helpers.
// ---------------------------------------------------------------------------

// Atomic f32 add via CAS on the u32 bit pattern (core WGSL has no atomic float add).
fn atomic_add_f32(idx: u32, val: f32) {
    var old_bits = atomicLoad(&grad[idx]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + val);
        let res = atomicCompareExchangeWeak(&grad[idx], old_bits, new_bits);
        if (res.exchanged) { break; }
        old_bits = res.old_value;
    }
}

// Derivative of srgb_to_linear w.r.t. its argument.
fn srgb_to_linear_grad(c: f32) -> f32 {
    if (c <= 0.04045) { return 1.0 / 12.92; }
    return (2.4 / 1.055) * pow((c + 0.055) / 1.055, 1.4);
}

// ∂L/∂C (linear RGB) from ∂L/∂lab, chaining through xyz_to_lab and
// linear_rgb_to_xyz. xyz = forward XYZ at this pixel.
fn dl_dlab_to_dl_dc(dl_dlab: vec3<f32>, xyz: vec3<f32>) -> vec3<f32> {
    let xn = vec3<f32>(0.95047, 1.00000, 1.08883);
    // f'(t)/wn for each channel, t = xyz_ch / wn_ch.
    var dfx = vec3<f32>(0.0);
    for (var ch: u32 = 0u; ch < 3u; ch = ch + 1u) {
        let t = xyz[ch] / xn[ch];
        var fp = 7.787;
        if (t > 0.008856) { fp = (1.0 / 3.0) * pow(t, -2.0 / 3.0); }
        dfx[ch] = fp / xn[ch];
    }
    // ∂lab/∂xyz Jacobian rows: L=116*fy-16, a=500*(fx-fy), b=200*(fy-fz).
    // jl[i][j] = ∂lab_i/∂xyz_j.
    // dl_dxyz = dl_dlab^T · jl.
    var dl_dxyz = vec3<f32>(0.0);
    // column 0 (X): only a depends on X -> jl[1][0]=500*dfx[0]
    dl_dxyz[0] = dl_dlab[1] * (500.0 * dfx[0]);
    // column 1 (Y): L=116*dfx[1], a=-500*dfx[1], b=200*dfx[1]
    dl_dxyz[1] = dl_dlab[0] * (116.0 * dfx[1])
               + dl_dlab[1] * (-500.0 * dfx[1])
               + dl_dlab[2] * (200.0 * dfx[1]);
    // column 2 (Z): only b depends on Z -> jl[2][2]=-200*dfx[2]
    dl_dxyz[2] = dl_dlab[2] * (-200.0 * dfx[2]);

    // ∂xyz/∂C = RGB_TO_XYZ (= linear_rgb_to_xyz matrix). dl_dc = dl_dxyz^T · M.
    // M[i][j] = ∂xyz_i/∂C_j.
    var dl_dc = vec3<f32>(0.0);
    dl_dc[0] = dl_dxyz[0] * 0.4124564 + dl_dxyz[1] * 0.2126729 + dl_dxyz[2] * 0.0193339;
    dl_dc[1] = dl_dxyz[0] * 0.3575761 + dl_dxyz[1] * 0.7151522 + dl_dxyz[2] * 0.1191920;
    dl_dc[2] = dl_dxyz[0] * 0.1804375 + dl_dxyz[1] * 0.0721750 + dl_dxyz[2] * 0.9503041;
    return dl_dc;
}

// Gradient of an edge's signed distance (CCW edge a->b) w.r.t.
// (a.x, a.y, b.x, b.y). Mirrors edge_signed_dist_grad.
fn edge_sd_grad(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> vec4<f32> {
    let ex = b.x - a.x;
    let ey = b.y - a.y;
    let len2 = ex * ex + ey * ey;
    let len = sqrt(len2);
    if (len == 0.0) { return vec4<f32>(0.0); }
    let num = (-ey) * (p.x - a.x) + ex * (p.y - a.y);
    let dnum = vec4<f32>(
        ey - (p.y - a.y),
        (p.x - a.x) - ex,
        p.y - a.y,
        -(p.x - a.x)
    );
    let dlen = vec4<f32>(-ex / len, -ey / len, ex / len, ey / len);
    return (dnum * len - num * dlen) / len2;
}

// ---------------------------------------------------------------------------
// Backward entry: ONE front-to-back walk over tile-overlapping triangles,
// reconstructing C_below (prefix color) and suffix transmittance tt via
// tt = T_final / prefix_trans. The per-triangle gradient block matches the
// finite-difference-verified CPU reference (softras_ref.rs `grad_loss`); only
// how `below`/`tt` are obtained changes (O(num_tris) vs brute-force O(num_tris²)).
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16, 1)
fn backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x; let py = gid.y;
    if (px >= params.width || py >= params.height) { return; }
    let p = pixel_to_clip(px, py);
    let inv_n = 1.0 / f32(params.width * params.height);

    // Forward state stored by `forward`: (c_full.rgb, T_final).
    let st = state[py * params.width + px];
    let c_full = st.xyz;
    let t_final = st.w;

    // ---- ∂L/∂C_full. ----
    let xyz = linear_rgb_to_xyz(c_full);
    let lab = xyz_to_lab(xyz);
    let gl = goal_lab[py * params.width + px].xyz;
    let dl_dlab = vec3<f32>(
        2.0 * inv_n * (lab.x - gl.x),
        2.0 * inv_n * (lab.y - gl.y),
        2.0 * inv_n * (lab.z - gl.z)
    );
    let dl_dc = dl_dlab_to_dl_dc(dl_dlab, xyz);

    // ---- Single front-to-back walk; reconstruct below/tt per triangle. ----
    var below = vec3<f32>(0.0);
    var prefix_trans = 1.0; // accumulates Π_{j=0..t}(1 - src_a_clamped_j) as we process triangle t
    let tile = (py / TILE) * params.tiles_x + (px / TILE);
    let lo = tile_offsets[tile];
    let hi = tile_offsets[tile + 1u];
    for (var ii: u32 = lo; ii < hi; ii = ii + 1u) {
        let t = tile_list[ii];
        let base = t * 18u;
        let v0 = vec2<f32>(tri_params[base + 0u], tri_params[base + 1u]);
        let v1 = vec2<f32>(tri_params[base + 6u], tri_params[base + 7u]);
        let v2 = vec2<f32>(tri_params[base + 12u], tri_params[base + 13u]);

        // Recompute this triangle's forward locals + barycentric grad.
        let d0 = edge_sd(p, v0, v1);
        let d1 = edge_sd(p, v1, v2);
        let d2 = edge_sd(p, v2, v0);
        let dmin = min(d0, min(d1, d2));
        let cov = 1.0 / (1.0 + exp(-dmin / params.tau));
        let dcov_dd = cov * (1.0 - cov) / params.tau;

        // Barycentric weights + Jacobian dl[k][j], j = (v0x,v0y,v1x,v1y,v2x,v2y).
        let det = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
        var l = vec3<f32>(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        var dl0: array<f32, 6>;
        var dl1: array<f32, 6>;
        var dl2: array<f32, 6>;
        for (var j: u32 = 0u; j < 6u; j = j + 1u) {
            dl0[j] = 0.0; dl1[j] = 0.0; dl2[j] = 0.0;
        }
        if (abs(det) >= 1e-12) {
            let nn0 = (v1.y - v2.y) * (p.x - v2.x) + (v2.x - v1.x) * (p.y - v2.y);
            let nn1 = (v2.y - v0.y) * (p.x - v2.x) + (v0.x - v2.x) * (p.y - v2.y);
            let ll0 = nn0 / det;
            let ll1 = nn1 / det;
            l = vec3<f32>(ll0, ll1, 1.0 - ll0 - ll1);

            // ∂det/∂(v0x,v0y,v1x,v1y,v2x,v2y)
            var dd: array<f32, 6>;
            dd[0] = v1.y - v2.y;
            dd[1] = v2.x - v1.x;
            dd[2] = -(v0.y - v2.y);
            dd[3] = v0.x - v2.x;
            dd[4] = (v0.y - v2.y) - (v1.y - v2.y);
            dd[5] = (v1.x - v2.x) - (v0.x - v2.x);
            // ∂nn0/∂...
            var dn0: array<f32, 6>;
            dn0[0] = 0.0;
            dn0[1] = 0.0;
            dn0[2] = -(p.y - v2.y);
            dn0[3] = p.x - v2.x;
            dn0[4] = -(v1.y - v2.y) + (p.y - v2.y);
            dn0[5] = -(p.x - v2.x) - (v2.x - v1.x);
            // ∂nn1/∂...
            var dn1: array<f32, 6>;
            dn1[0] = p.y - v2.y;
            dn1[1] = -(p.x - v2.x);
            dn1[2] = 0.0;
            dn1[3] = 0.0;
            dn1[4] = -(v2.y - v0.y) - (p.y - v2.y);
            dn1[5] = (p.x - v2.x) - (v0.x - v2.x);
            let det2 = det * det;
            for (var j: u32 = 0u; j < 6u; j = j + 1u) {
                let g0 = (dn0[j] * det - nn0 * dd[j]) / det2;
                let g1 = (dn1[j] * det - nn1 * dd[j]) / det2;
                dl0[j] = g0;
                dl1[j] = g1;
                dl2[j] = -g0 - g1;
            }
        }

        // Interpolated rgb and alpha.
        let col0 = vec3<f32>(tri_params[base + 2u], tri_params[base + 3u], tri_params[base + 4u]);
        let col1 = vec3<f32>(tri_params[base + 8u], tri_params[base + 9u], tri_params[base + 10u]);
        let col2 = vec3<f32>(tri_params[base + 14u], tri_params[base + 15u], tri_params[base + 16u]);
        let a0 = tri_params[base + 5u];
        let a1 = tri_params[base + 11u];
        let a2 = tri_params[base + 17u];
        let rgb = l.x * col0 + l.y * col1 + l.z * col2;
        let a = l.x * a0 + l.y * a1 + l.z * a2;
        let src_a = cov * a;
        let lin = vec3<f32>(srgb_to_linear(rgb.x), srgb_to_linear(rgb.y), srgb_to_linear(rgb.z));

        // --- reconstruct below_t / tt from the running walk (replaces inner loops) ---
        let src_a_clamped = min(src_a, 0.999); // cap keeps prefix_trans > 0 (backward divides T_final/prefix_trans)
        prefix_trans = prefix_trans * (1.0 - src_a_clamped); // update BEFORE computing tt
        let tt = t_final / prefix_trans;                     // suffix transmittance Π_{j>t}
        let below_t = below;                                 // composite of 0..t-1

        // ∂L/∂src_a and ∂L/∂lin.
        var dl_dsrc_a = 0.0;
        for (var ch: u32 = 0u; ch < 3u; ch = ch + 1u) {
            dl_dsrc_a = dl_dsrc_a + dl_dc[ch] * tt * (lin[ch] - below_t[ch]);
        }
        var dl_dlin = vec3<f32>(0.0);
        for (var ch: u32 = 0u; ch < 3u; ch = ch + 1u) {
            dl_dlin[ch] = dl_dc[ch] * tt * src_a;
        }

        // src_a = cov*a.
        let dl_dcov = dl_dsrc_a * a;
        let dl_da = dl_dsrc_a * cov;

        // ∂L/∂rgb = ∂L/∂lin * srgb'(rgb).
        let dl_drgb = vec3<f32>(
            dl_dlin.x * srgb_to_linear_grad(rgb.x),
            dl_dlin.y * srgb_to_linear_grad(rgb.y),
            dl_dlin.z * srgb_to_linear_grad(rgb.z)
        );

        // Color comps (2,3,4) and alpha comp (5): per vertex k, weight l_k.
        let lk = array<f32, 3>(l.x, l.y, l.z);
        for (var k: u32 = 0u; k < 3u; k = k + 1u) {
            let kb = base + k * 6u;
            atomic_add_f32(kb + 2u, dl_drgb.x * lk[k]);
            atomic_add_f32(kb + 3u, dl_drgb.y * lk[k]);
            atomic_add_f32(kb + 4u, dl_drgb.z * lk[k]);
            atomic_add_f32(kb + 5u, dl_da * lk[k]);
        }

        // Position comps (0,1) — Route A: coverage via argmin-edge distance.
        var eg: vec4<f32>;
        var va: u32;
        var vb: u32;
        if (d0 <= d1 && d0 <= d2) {
            eg = edge_sd_grad(p, v0, v1); va = 0u; vb = 1u;
        } else if (d1 <= d2) {
            eg = edge_sd_grad(p, v1, v2); va = 1u; vb = 2u;
        } else {
            eg = edge_sd_grad(p, v2, v0); va = 2u; vb = 0u;
        }
        let dl_dd = dl_dcov * dcov_dd;
        atomic_add_f32(base + va * 6u + 0u, dl_dd * eg.x);
        atomic_add_f32(base + va * 6u + 1u, dl_dd * eg.y);
        atomic_add_f32(base + vb * 6u + 0u, dl_dd * eg.z);
        atomic_add_f32(base + vb * 6u + 1u, dl_dd * eg.w);

        // Route B: barycentric weights depend on positions, feeding rgb and alpha.
        // ∂L/∂l_k from both routes.
        var dl_dl: array<f32, 3>;
        dl_dl[0] = dl_drgb.x * col0.x + dl_drgb.y * col0.y + dl_drgb.z * col0.z + dl_da * a0;
        dl_dl[1] = dl_drgb.x * col1.x + dl_drgb.y * col1.y + dl_drgb.z * col1.z + dl_da * a1;
        dl_dl[2] = dl_drgb.x * col2.x + dl_drgb.y * col2.y + dl_drgb.z * col2.z + dl_da * a2;
        // Route ∂L/∂l_k through dl[k][vert*2+comp] to position scalars.
        for (var vert: u32 = 0u; vert < 3u; vert = vert + 1u) {
            for (var comp: u32 = 0u; comp < 2u; comp = comp + 1u) {
                let j = vert * 2u + comp;
                let gp = dl_dl[0] * dl0[j] + dl_dl[1] * dl1[j] + dl_dl[2] * dl2[j];
                atomic_add_f32(base + vert * 6u + comp, gp);
            }
        }

        // Advance prefix color AFTER using below_t (TRUE src_a, matches forward).
        below = src_a * lin + (1.0 - src_a) * below;
    }
}
