//! Shared fixtures for the per-module `#[cfg(test)]` suites: synthetic goals and
//! a wgpu bring-up helper. Compiled only under `cfg(test)`.

use crate::goal::GoalImage;
use crate::gpu::init_wgpu;
use futures::executor::block_on;
use image::{ImageBuffer, Rgba};
use std::sync::Arc;

/// Black/white checker pattern at the requested resolution (4×4 logical cells).
pub(crate) fn make_checker_goal(size: u32) -> GoalImage {
    let mut buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(size, size);
    let cell = (size / 4).max(1); // 4×4 logical cells; min 1px
    for y in 0..size {
        for x in 0..size {
            let on = ((x / cell) + (y / cell)) % 2 == 0;
            let v = if on { 255 } else { 0 };
            buf.put_pixel(x, y, Rgba([v, v, v, 255]));
        }
    }
    GoalImage { pixels: buf }
}

/// Uniform `rgb` fill at the requested resolution.
pub(crate) fn make_solid_goal(size: u32, rgb: [u8; 3]) -> GoalImage {
    let mut buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(size, size);
    for y in 0..size {
        for x in 0..size {
            buf.put_pixel(x, y, Rgba([rgb[0], rgb[1], rgb[2], 255]));
        }
    }
    GoalImage { pixels: buf }
}

/// Per-column gradient: every distinct x maps to a distinct R channel, so two
/// points with different x always get different colours.
pub(crate) fn make_gradient_goal(size: u32) -> GoalImage {
    let mut buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let v = (x * 255 / (size - 1)) as u8;
            buf.put_pixel(x, y, Rgba([v, 128, 255 - v, 255]));
        }
    }
    GoalImage { pixels: buf }
}

/// Reuse the production `init_wgpu` helper; same backends/preferences.
pub(crate) fn init_test_wgpu() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    block_on(init_wgpu())
}

/// Signed area of a triangle (positive = CCW in clip space).
pub(crate) fn tri_signed_area(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
}
