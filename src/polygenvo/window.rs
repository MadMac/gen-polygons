//! Live viewer for `--show-window`: opens an OS window and renders the current
//! best genome on every accepted improvement, so a run can be watched in real
//! time instead of only via the PNG snapshots in `triangles/`.
//!
//! Threading model: single-threaded. `run_es` stays the loop driver and calls
//! `WindowObserver::on_step` (a [`StepObserver`]) once per step; that pumps
//! pending window events without blocking ([`pump_app_events`] with a zero
//! timeout) and re-renders only when the best changed. This avoids sharing the
//! wgpu device across threads (the GL backend is awkward to use that way).
//!
//! The GPU device is created *here* (not in `gpu::init_wgpu`) because the
//! adapter must be chosen `compatible_surface`, and the surface needs the
//! window — which only exists once the event loop has resumed.

use std::iter;
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowId};

use crate::es::StepObserver;
use crate::fitness::MSAA_SAMPLE_COUNT;
use crate::genome::{Vertex, MAX_VERTICES};

// The initial window is sized from the goal image (the render is square and
// resolution-independent), clamped to this range so a tiny goal still opens a
// usable window and a huge one doesn't open off-screen. It stays freely
// resizable afterwards.
const MIN_WINDOW_SIZE: u32 = 256;
const MAX_WINDOW_SIZE: u32 = 1024;

// Cap on-screen refresh to ~display rate. The ES improves far faster than the
// eye (or monitor) can follow, so without this a burst of improvements would
// flood the loop with presents; throttling here keeps the search at full speed
// while still showing essentially every visible change.
const MIN_PRESENT_INTERVAL: Duration = Duration::from_millis(16);

/// Holds the GPU handles plus the live observer. `main` passes `device`/`queue`
/// on to `run_es` (so the ES and the viewer share one device) and the
/// `observer` into `run_es`'s observer slot.
pub(crate) struct WindowInit {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) observer: WindowObserver,
}

/// winit `ApplicationHandler`: creates the window on `resumed` and records
/// window events into plain fields (no rendering happens in the callbacks —
/// `WindowObserver::on_step` acts on these after each pump).
#[derive(Default)]
struct WindowApp {
    window: Option<Arc<Window>>,
    // Initial window edge length in logical pixels (square), derived from the goal.
    initial_size: u32,
    close_requested: bool,
    resized: Option<(u32, u32)>,
    redraw_requested: bool,
}

impl ApplicationHandler for WindowApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("polygenvo — best candidate")
                .with_inner_size(LogicalSize::new(self.initial_size, self.initial_size));
            let window = event_loop
                .create_window(attrs)
                .expect("failed to create window");
            self.window = Some(Arc::new(window));
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.close_requested = true,
            WindowEvent::Resized(size) => self.resized = Some((size.width, size.height)),
            WindowEvent::RedrawRequested => self.redraw_requested = true,
            _ => {}
        }
    }
}

/// Owns the surface + a render pipeline targeting the surface's (sRGB) format,
/// and draws the genome to the swapchain. This mirrors the render half of
/// `FitnessCalc::snapshot` (same passthrough `shader.wgsl`, CCW + back-cull,
/// BLACK clear, `MSAA_SAMPLE_COUNT`× MSAA resolved into the frame) but presents
/// to a window instead of reading back to a PNG, and runs no compute pass.
struct Presenter {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    // Multisampled colour target resolved into each frame; `None` at 1× (when
    // the surface format doesn't support MSAA). Recreated on resize.
    sample_count: u32,
    msaa_view: Option<wgpu::TextureView>,
    last_present: Option<Instant>,
}

impl Presenter {
    fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
        sample_count: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Window Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Window Render Pipeline Layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Window Render Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        alpha: wgpu::BlendComponent::OVER,
                        color: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Window Vertex Buffer"),
            size: (MAX_VERTICES * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        surface.configure(&device, &config);
        let msaa_view =
            (sample_count > 1).then(|| make_msaa_view(&device, &config, sample_count));
        Self {
            device,
            queue,
            surface,
            config,
            pipeline,
            vertex_buffer,
            sample_count,
            msaa_view,
            last_present: None,
        }
    }

    /// True once enough time has passed since the last present to draw another
    /// frame without exceeding the display refresh budget.
    fn ready_to_present(&self) -> bool {
        self.last_present
            .is_none_or(|t| t.elapsed() >= MIN_PRESENT_INTERVAL)
    }

    /// Reconfigure the swapchain after a window resize (ignores zero-area sizes,
    /// e.g. when minimised).
    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        if self.sample_count > 1 {
            self.msaa_view = Some(make_msaa_view(&self.device, &self.config, self.sample_count));
        }
    }

    /// Render `genome` to the next surface frame and present it.
    fn present(&mut self, genome: &[Vertex]) {
        let n = genome.len().min(MAX_VERTICES);
        if n > 0 {
            self.queue
                .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&genome[..n]));
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                // Swapchain stale (often right after a resize): rebuild and retry once.
                self.surface.configure(&self.device, &self.config);
                match self.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    _ => return,
                }
            }
            // Timeout / Occluded / Validation: skip this frame.
            _ => return,
        };

        let frame_view = frame.texture.create_view(&Default::default());
        // With MSAA, draw into the multisampled target and resolve into the
        // frame; at 1× draw straight into the frame.
        let (view, resolve_target) = match &self.msaa_view {
            Some(msaa) => (msaa, Some(&frame_view)),
            None => (&frame_view, None),
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Window Encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Window Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..n as u32, 0..1);
        }
        self.queue.submit(iter::once(encoder.finish()));
        frame.present();
        self.last_present = Some(Instant::now());
    }
}

/// Build a multisampled colour target matching the surface size/format, for
/// MSAA passes that resolve into the swapchain frame.
fn make_msaa_view(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    sample_count: u32,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Window MSAA Target"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&Default::default())
}

/// The live observer handed to `run_es`. Each step it pumps window events and,
/// on improvement (or an OS-requested redraw), re-renders the best candidate.
pub(crate) struct WindowObserver {
    event_loop: EventLoop<()>,
    app: WindowApp,
    presenter: Presenter,
}

impl StepObserver for WindowObserver {
    fn on_step(&mut self, best: &[Vertex], improved: bool) -> bool {
        self.app.redraw_requested = false;
        let status = self
            .event_loop
            .pump_app_events(Some(Duration::ZERO), &mut self.app);
        if matches!(status, PumpStatus::Exit(_)) || self.app.close_requested {
            return false;
        }
        if let Some((w, h)) = self.app.resized.take() {
            self.presenter.resize(w, h);
        }
        // Redraw requests from the OS (expose/resize) are always honoured; new
        // bests are throttled to the display rate so the search never stalls on
        // a present (which under a vsync present-mode would block the loop).
        if self.app.redraw_requested || (improved && self.presenter.ready_to_present()) {
            self.presenter.present(best);
        }
        true
    }
}

/// Open the live window and bring up a surface-compatible wgpu device. Returns
/// the shared device/queue plus the observer that `run_es` drives. Panics on any
/// setup failure (the feature is opt-in via `--show-window`).
pub(crate) fn init_window(goal_size: u32) -> WindowInit {
    let mut event_loop = EventLoop::new().expect("failed to create event loop");

    // Same backend selection as the headless path. On GLES/Wayland the instance
    // needs the platform display handle up front to be able to present.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: crate::gpu::preferred_backends(),
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: Some(Box::new(event_loop.owned_display_handle())),
    });

    // Pump once so `resumed` runs and creates the window, sized from the goal.
    let mut app = WindowApp {
        initial_size: goal_size.clamp(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE),
        ..Default::default()
    };
    let _ = event_loop.pump_app_events(Some(Duration::ZERO), &mut app);
    let window = app
        .window
        .clone()
        .expect("window was not created on the first event-loop pump");

    // Window-handle-only surface: the display handle already lives on the
    // instance (passing it again here would be rejected by wgpu).
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::from_window_without_display(
            window.clone(),
        ))
        .expect("failed to create window surface");

    let adapter = futures::executor::block_on(instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        },
    ))
    .expect("no wgpu adapter compatible with the window surface");

    let (device, queue) = futures::executor::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("window device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        },
    ))
    .expect("device init failed");
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // Pick an sRGB surface format so on-screen colours match the offscreen
    // Rgba8UnormSrgb render the fitness shader scores against.
    let size = window.inner_size();
    let (w, h) = (size.width.max(1), size.height.max(1));
    let mut config = surface
        .get_default_config(&adapter, w, h)
        .expect("surface is not supported by the adapter");
    let caps = surface.get_capabilities(&adapter);
    if let Some(srgb) = caps.formats.iter().copied().find(|f| f.is_srgb()) {
        config.format = srgb;
    }
    // Prefer a non-blocking present mode (Mailbox, else Immediate) so a present
    // never stalls the ES loop on vsync; fall back to the default (Fifo) if the
    // surface offers neither. The `ready_to_present` throttle still caps the
    // on-screen rate regardless.
    for mode in [wgpu::PresentMode::Mailbox, wgpu::PresentMode::Immediate] {
        if caps.present_modes.contains(&mode) {
            config.present_mode = mode;
            break;
        }
    }

    // Use MSAA to match the anti-aliased PNG snapshots, but only if the chosen
    // surface format actually supports it (else fall back to single-sample).
    let sample_count = if adapter
        .get_texture_format_features(config.format)
        .flags
        .sample_count_supported(MSAA_SAMPLE_COUNT)
    {
        MSAA_SAMPLE_COUNT
    } else {
        1
    };

    let presenter = Presenter::new(device.clone(), queue.clone(), surface, config, sample_count);

    WindowInit {
        device,
        queue,
        observer: WindowObserver {
            event_loop,
            app,
            presenter,
        },
    }
}
