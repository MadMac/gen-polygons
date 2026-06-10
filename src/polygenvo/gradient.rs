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
