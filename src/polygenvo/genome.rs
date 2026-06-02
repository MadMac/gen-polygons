//! The genome: a `Vec<Vertex>` interpreted as a CCW `TriangleList`, plus the
//! geometry helpers that seed and subdivide triangles.

use crate::goal::{sample_goal_color, GoalImage};
use rand::prelude::*;

/// Triangle-count ceiling — the one knob that governs capacity. It is the final
/// phase's cap (see PHASES) and the vertex-buffer capacity. The genome grows
/// toward it organically via the fitness-gated `split` operator.
pub(crate) const MAX_TRIANGLES: usize = 10000;

/// Vertex buffer capacity (in vertices). 3 vertices per triangle.
pub(crate) const MAX_VERTICES: usize = MAX_TRIANGLES * 3;

/// Cold-start triangle count (also the reference count for add's seed-radius
/// scaling: a fresh triangle shrinks toward the current triangle scale as the
/// genome grows).
pub(crate) const INITIAL_TRIANGLES: usize = 40;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq, PartialOrd)]
pub(crate) struct Vertex {
    pub(crate) position: [f32; 3],
    pub(crate) color: [f32; 4],
}

impl Vertex {
    pub(crate) fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Centroid (clip-space x, y) of the `t`-th triangle in a genome slice.
pub(crate) fn triangle_centroid(genome: &[Vertex], t: usize) -> (f32, f32) {
    let b = t * 3;
    let cx = (genome[b].position[0] + genome[b + 1].position[0] + genome[b + 2].position[0]) / 3.0;
    let cy = (genome[b].position[1] + genome[b + 1].position[1] + genome[b + 2].position[1]) / 3.0;
    (cx, cy)
}

/// Generate one triangle centred on `(cx, cy)` in clip space, with a random
/// radius (within `max_radius`) and alpha, coloured from the goal at the centre.
/// Vertices are placed in CCW order so the rasteriser (front_face: Ccw,
/// cull_mode: Back) keeps the triangle. Both uniform seeding (`init_genome`) and
/// error-guided seeding (`variation`) share this builder — they differ only in
/// how the centre `(cx, cy)` is chosen.
pub(crate) fn seeded_triangle(
    goal: &GoalImage,
    cx: f32,
    cy: f32,
    max_radius: f32,
    rng: &mut impl Rng,
) -> [Vertex; 3] {
    let radius = rng.random_range(max_radius * 0.3..max_radius);
    let alpha = rng.random_range(0.25_f32..0.75);
    let base = rng.random_range(0.0_f32..std::f32::consts::TAU);
    let third = std::f32::consts::TAU / 3.0;
    // Each vertex samples the goal at its own position (sharing one alpha), so a
    // triangle is born carrying the goal's local colour ramp — the rasteriser
    // interpolates between the three, giving a gradient rather than a flat fill.
    let mk = |theta: f32| {
        let px = cx + radius * theta.cos();
        let py = cy + radius * theta.sin();
        Vertex {
            position: [px, py, 0.0],
            color: sample_goal_color(goal, px, py, alpha),
        }
    };
    // CCW (with wgpu's y-up clip space).
    [mk(base), mk(base + third), mk(base + 2.0 * third)]
}

pub(crate) fn init_genome(goal: &GoalImage, n_triangles: usize, rng: &mut impl Rng) -> Vec<Vertex> {
    let mut genome = Vec::with_capacity(n_triangles * 3);
    for _ in 0..n_triangles {
        let cx = rng.random_range(-0.9_f32..0.9);
        let cy = rng.random_range(-0.9_f32..0.9);
        let tri = seeded_triangle(goal, cx, cy, 0.3, rng);
        genome.extend_from_slice(&tri);
    }
    genome
}

/// Subdivide a CCW triangle into 4 midpoint children that exactly tile it.
/// Children keep the parent's winding and alpha; each child *vertex* samples the
/// goal's RGB at its own position, so a split adds colour resolution where the
/// goal varies under the triangle (and is ~neutral where it doesn't) — and each
/// child is itself a gradient, not a flat fill.
/// Returns 12 vertices = 4 triangles.
pub(crate) fn split_triangle(v0: Vertex, v1: Vertex, v2: Vertex, goal: &GoalImage) -> [Vertex; 12] {
    let alpha = v0.color[3];
    let mid = |a: &Vertex, b: &Vertex| -> [f32; 3] {
        [
            (a.position[0] + b.position[0]) * 0.5,
            (a.position[1] + b.position[1]) * 0.5,
            0.0,
        ]
    };
    let m01 = mid(&v0, &v1);
    let m12 = mid(&v1, &v2);
    let m20 = mid(&v2, &v0);
    // Build a child from three positions; each vertex is recoloured from the goal
    // at its own position (sharing the parent's alpha), so the child interpolates.
    let child = |p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]| -> [Vertex; 3] {
        let vtx = |p: [f32; 3]| Vertex { position: p, color: sample_goal_color(goal, p[0], p[1], alpha) };
        [vtx(p0), vtx(p1), vtx(p2)]
    };
    // Three corner children + one centre child, all CCW (verified against a
    // CCW parent v0,v1,v2).
    let c0 = child(v0.position, m01, m20);
    let c1 = child(v1.position, m12, m01);
    let c2 = child(v2.position, m20, m12);
    let c3 = child(m01, m12, m20);
    let mut out = [Vertex { position: [0.0; 3], color: [0.0; 4] }; 12];
    out[0..3].copy_from_slice(&c0);
    out[3..6].copy_from_slice(&c1);
    out[6..9].copy_from_slice(&c2);
    out[9..12].copy_from_slice(&c3);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{make_gradient_goal, make_solid_goal, tri_signed_area};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn triangle_centroid_is_mean_of_vertices() {
        let genome = vec![
            Vertex { position: [0.0, 0.0, 0.0], color: [0.0; 4] },
            Vertex { position: [0.3, 0.0, 0.0], color: [0.0; 4] },
            Vertex { position: [0.0, 0.6, 0.0], color: [0.0; 4] },
        ];
        let (cx, cy) = triangle_centroid(&genome, 0);
        assert!((cx - 0.1).abs() < 1e-6, "cx {cx}");
        assert!((cy - 0.2).abs() < 1e-6, "cy {cy}");
    }

    #[test]
    fn split_triangle_tiles_and_preserves_winding() {
        let goal = make_solid_goal(64, [100, 150, 200]);
        let v0 = Vertex { position: [-0.5, -0.5, 0.0], color: [0.1, 0.2, 0.3, 0.5] };
        let v1 = Vertex { position: [0.5, -0.5, 0.0], color: [0.1, 0.2, 0.3, 0.5] };
        let v2 = Vertex { position: [0.0, 0.5, 0.0], color: [0.1, 0.2, 0.3, 0.5] };
        let parent_area = tri_signed_area(v0.position, v1.position, v2.position);
        assert!(parent_area > 0.0, "test fixture must be CCW");

        let children = split_triangle(v0, v1, v2, &goal);
        assert_eq!(children.len(), 12, "4 child triangles = 12 vertices");

        let mut total = 0.0;
        for t in 0..4 {
            let b = t * 3;
            let area = tri_signed_area(children[b].position, children[b + 1].position, children[b + 2].position);
            assert!(area > 0.0, "child {t} must keep CCW winding (got area {area})");
            total += area;
        }
        assert!((total - parent_area).abs() < 1e-5, "children must tile parent: {total} vs {parent_area}");
    }

    #[test]
    fn split_triangle_inherits_alpha() {
        let goal = make_solid_goal(64, [100, 150, 200]);
        let a = 0.42_f32;
        let v0 = Vertex { position: [-0.5, -0.5, 0.0], color: [0.1, 0.2, 0.3, a] };
        let v1 = Vertex { position: [0.5, -0.5, 0.0], color: [0.4, 0.5, 0.6, a] };
        let v2 = Vertex { position: [0.0, 0.5, 0.0], color: [0.7, 0.8, 0.9, a] };
        let children = split_triangle(v0, v1, v2, &goal);
        for (i, v) in children.iter().enumerate() {
            assert_eq!(v.color[3], a, "child vertex {i} alpha must equal parent alpha");
        }
    }

    #[test]
    fn split_triangle_recolours_from_goal() {
        // Non-uniform goal: child colours must differ (detail captured).
        let grad = make_gradient_goal(64);
        let v0 = Vertex { position: [-0.6, -0.5, 0.0], color: [0.0, 0.0, 0.0, 0.5] };
        let v1 = Vertex { position: [0.6, -0.5, 0.0], color: [0.0, 0.0, 0.0, 0.5] };
        let v2 = Vertex { position: [0.0, 0.6, 0.0], color: [0.0, 0.0, 0.0, 0.5] };
        let kids = split_triangle(v0, v1, v2, &grad);
        let reds: Vec<f32> = (0..4).map(|t| kids[t * 3].color[0]).collect();
        assert!(reds.iter().any(|&r| (r - reds[0]).abs() > 1e-3), "non-uniform goal: child colours must differ, got {reds:?}");

        // Uniform goal: all children share one colour (the neutral case).
        let solid = make_solid_goal(64, [10, 20, 30]);
        let kids2 = split_triangle(v0, v1, v2, &solid);
        for t in 0..4 {
            let c = kids2[t * 3].color;
            assert!((c[0] - kids2[0].color[0]).abs() < 1e-6, "uniform goal: child {t} colour must match");
        }
    }

    #[test]
    fn split_child_vertices_form_a_gradient() {
        // make_gradient_goal maps x -> R, so a child spanning width should have
        // vertices with differing R: each child is itself a gradient, not flat.
        let grad = make_gradient_goal(64);
        let v0 = Vertex { position: [-0.6, -0.5, 0.0], color: [0.0, 0.0, 0.0, 0.5] };
        let v1 = Vertex { position: [0.6, -0.5, 0.0], color: [0.0, 0.0, 0.0, 0.5] };
        let v2 = Vertex { position: [0.0, 0.6, 0.0], color: [0.0, 0.0, 0.0, 0.5] };
        let kids = split_triangle(v0, v1, v2, &grad);
        // The centre child (verts 9..12) spans the full width, so its three
        // vertices sit at distinct x and must carry distinct R.
        let reds = [kids[9].color[0], kids[10].color[0], kids[11].color[0]];
        assert!(
            reds.iter().any(|&r| (r - reds[0]).abs() > 1e-3),
            "centre child vertices must form a gradient, got {reds:?}"
        );
    }

    #[test]
    fn seeded_triangle_vertices_sample_their_own_position() {
        // Over a non-uniform goal the three vertices land at distinct positions,
        // so they pick up distinct colours (a seeded gradient).
        let grad = make_gradient_goal(64);
        let mut rng = StdRng::seed_from_u64(7);
        let tri = seeded_triangle(&grad, 0.0, 0.0, 0.5, &mut rng);
        let reds = [tri[0].color[0], tri[1].color[0], tri[2].color[0]];
        assert!(
            reds.iter().any(|&r| (r - reds[0]).abs() > 1e-3),
            "seeded triangle vertices must sample their own position, got {reds:?}"
        );
        // Alpha stays shared across the triangle.
        assert_eq!(tri[0].color[3], tri[1].color[3]);
        assert_eq!(tri[1].color[3], tri[2].color[3]);
    }
}
