//! wgpu device/queue bring-up shared by production and tests.

use std::sync::Arc;

/// Map an optional `POLYGENVO_BACKEND` value to a backend set. Unset/unknown =
/// PRIMARY (Vulkan on Linux, Metal on macOS, DX12 on Windows).
fn backends_from_env(v: Option<&str>) -> wgpu::Backends {
    match v {
        Some("gl") => wgpu::Backends::GL,
        Some("vulkan") => wgpu::Backends::VULKAN,
        Some("metal") => wgpu::Backends::METAL,
        Some("dx12") => wgpu::Backends::DX12,
        _ => wgpu::Backends::PRIMARY,
    }
}

pub(crate) fn preferred_backends() -> wgpu::Backends {
    backends_from_env(std::env::var("POLYGENVO_BACKEND").ok().as_deref())
}

/// Prepend the canonical CIE color primitives (`color.wgsl`) to a shader body,
/// so the fitness evaluator (`fitness.wgsl`) and the soft-rasterizer
/// (`softraster_tiled.wgsl`) share one definition of the linear-RGB→XYZ→Lab
/// pipeline. They must agree for the elitist polish gate to hold. The prelude is
/// declarations only (no `enable`/`requires` directives), so prepending it is
/// always valid; the shared fns then precede any use in the body.
pub(crate) fn with_color_prelude(body: &str) -> String {
    format!("{}\n{}", include_str!("color.wgsl"), body)
}

async fn try_backend(backends: wgpu::Backends) -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: Default::default(),
        display: Default::default(),
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok()?;
    let info = adapter.get_info();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
        .ok()?;
    println!("wgpu backend: {:?} — {}", info.backend, info.name);
    Some((Arc::new(device), Arc::new(queue)))
}

pub(crate) async fn init_wgpu() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let preferred = preferred_backends();
    if let Some(dq) = try_backend(preferred).await {
        return dq;
    }
    eprintln!("preferred wgpu backend {preferred:?} unavailable — falling back to GL");
    try_backend(wgpu::Backends::GL)
        .await
        .expect("no suitable wgpu adapter (GL fallback also failed)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn init_wgpu_returns_a_working_device() {
        let (device, _queue) = block_on(init_wgpu());
        let _b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
    }

    #[test]
    fn preferred_backends_honors_env_override() {
        assert_eq!(backends_from_env(Some("gl")), wgpu::Backends::GL);
        assert_eq!(backends_from_env(Some("vulkan")), wgpu::Backends::VULKAN);
        assert_eq!(backends_from_env(None), wgpu::Backends::PRIMARY);
        assert_eq!(backends_from_env(Some("garbage")), wgpu::Backends::PRIMARY);
    }
}
