// Adam optimizer update over the flat param buffer (one invocation per scalar
// param). Reads the gradient buffer (written by softraster_tiled.wgsl `backward`,
// re-read here as plain f32 — same bytes), updates bias-corrected moments, steps,
// and clamps: positions (param index %6 < 2) to [-1,1], colors/alpha to [0,1].

struct AdamParams {
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    step_t: u32,      // 1-based Adam step for bias correction
    num_params: u32,
    pad0: u32,
    pad1: u32,
}
@group(0) @binding(0) var<uniform> ap: AdamParams;
@group(0) @binding(1) var<storage, read_write> params: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ap.num_params) { return; }
    let g = grad[i];
    let mi = ap.b1 * m[i] + (1.0 - ap.b1) * g;
    let vi = ap.b2 * v[i] + (1.0 - ap.b2) * g * g;
    m[i] = mi;
    v[i] = vi;
    let t = f32(ap.step_t);
    let mh = mi / (1.0 - pow(ap.b1, t));
    let vh = vi / (1.0 - pow(ap.b2, t));
    var p = params[i] - ap.lr * mh / (sqrt(vh) + ap.eps);
    let comp = i % 6u;
    if (comp < 2u) { p = clamp(p, -1.0, 1.0); } else { p = clamp(p, 0.0, 1.0); }
    params[i] = p;
}
