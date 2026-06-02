//! The goal image we're approximating, plus its sampling/downsampling helpers.

use image::{ImageBuffer, Rgba};
use std::fmt;

/// The target raster the ES approximates. Wraps an RGBA8 buffer so we can give
/// it a compact `Debug` (the raw pixel buffer is huge) and keep the
/// goal-sampling helpers in one place.
#[derive(Clone)]
pub(crate) struct GoalImage {
    pub(crate) pixels: ImageBuffer<Rgba<u8>, Vec<u8>>,
}

impl fmt::Debug for GoalImage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GoalImage({}x{})", self.pixels.width(), self.pixels.height())
    }
}

pub(crate) fn load_goal_image(path: &str) -> GoalImage {
    let pixels = image::open(path)
        .unwrap_or_else(|e| panic!("failed to open goal image at {path}: {e}"))
        .into_rgba8();
    println!("Loaded {} ({}x{})", path, pixels.width(), pixels.height());
    GoalImage { pixels }
}

/// Downsample the goal image to the given square size using a Lanczos filter.
pub(crate) fn downsample_goal(full: &GoalImage, size: u32) -> GoalImage {
    if size == full.pixels.width() {
        return full.clone();
    }
    let dyn_img = image::DynamicImage::ImageRgba8(full.pixels.clone());
    let resized = dyn_img
        .resize_exact(size, size, image::imageops::FilterType::Lanczos3)
        .into_rgba8();
    GoalImage { pixels: resized }
}

/// Sample the goal image at a clip-space point to seed a triangle's colour,
/// using bilinear interpolation between the four surrounding texels so seeded
/// and split triangles pick up the goal's local colour ramp rather than
/// snapping to one pixel. Clip space `(-1, -1)` maps to top-left of the image
/// (image y is flipped).
pub(crate) fn sample_goal_color(goal: &GoalImage, cx: f32, cy: f32, alpha: f32) -> [f32; 4] {
    let w = goal.pixels.width();
    let h = goal.pixels.height();
    // Fractional pixel coordinates in [0, w-1] / [0, h-1].
    let fx = ((cx.clamp(-1.0, 1.0) + 1.0) * 0.5) * (w - 1) as f32;
    let fy = ((1.0 - cy.clamp(-1.0, 1.0)) * 0.5) * (h - 1) as f32;
    let x0 = fx.floor() as u32;
    let y0 = fy.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;

    let texel = |x: u32, y: u32| {
        let p = goal.pixels.get_pixel(x, y);
        [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]
    };
    let c00 = texel(x0, y0);
    let c10 = texel(x1, y0);
    let c01 = texel(x0, y1);
    let c11 = texel(x1, y1);

    let mut out = [0.0_f32; 4];
    for ch in 0..3 {
        let top = c00[ch] + (c10[ch] - c00[ch]) * tx;
        let bot = c01[ch] + (c11[ch] - c01[ch]) * tx;
        out[ch] = top + (bot - top) * ty;
    }
    out[3] = alpha;
    out
}
