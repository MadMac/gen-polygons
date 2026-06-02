//! Variation operators for the (1+λ)-ES: Gaussian perturbations, error-guided
//! seeding/relocation, the fitness-gated `split` growth path, and the top-level
//! `mutate` that dispatches one random operator per candidate.

use crate::fitness::ERROR_GRID_DIM;
use crate::genome::{
    init_genome, seeded_triangle, split_triangle, triangle_centroid, Vertex, INITIAL_TRIANGLES,
};
use crate::goal::{sample_goal_color, GoalImage};
use rand::prelude::*;

/// Which step size a mutation exercises, for per-type 1/5-rule adaptation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum OpKind {
    Positional, // vertex nudge -> sigma.pos
    Chromatic,  // recolour / alpha -> sigma.col
    Structural, // add / delete / z-swap / relocate -> no step size
}

/// The two self-adapted Gaussian step sizes a `mutate` call uses: `pos` for
/// vertex nudges (clip space), `col` for recolour/alpha (colour space). They
/// travel together because every mutation needs both and the 1/5 rule adapts
/// them as a pair.
#[derive(Copy, Clone, Debug)]
pub(crate) struct StepSizes {
    pub(crate) pos: f32,
    pub(crate) col: f32,
}

/// The mutation operators `mutate` dispatches between. Selection probabilities
/// live in `OPERATORS` as explicit weights rather than being implied by the
/// boundaries of a `0..100` range match.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Op {
    NudgeVertex,
    Recolor,
    NudgeAlpha,
    Split,
    SwapZ,
    AddTriangle,
    Relocate,
    Delete,
}

impl Op {
    /// The step-size class this operator exercises (independent of whether the
    /// operator's body ends up a no-op under the current genome bounds).
    fn kind(self) -> OpKind {
        match self {
            Op::NudgeVertex => OpKind::Positional,
            Op::Recolor | Op::NudgeAlpha => OpKind::Chromatic,
            Op::Split | Op::SwapZ | Op::AddTriangle | Op::Relocate | Op::Delete => {
                OpKind::Structural
            }
        }
    }
}

/// `(operator, weight)` selection table; weights need not sum to any particular
/// value — `pick_op` normalises by their total. These mirror the previous
/// hand-tuned `0..100` ranges (38/24/12/10/5/5/3/3).
const OPERATORS: &[(Op, u32)] = &[
    (Op::NudgeVertex, 38),
    (Op::Recolor, 24),
    (Op::NudgeAlpha, 12),
    (Op::Split, 10),
    (Op::SwapZ, 5),
    (Op::AddTriangle, 5),
    (Op::Relocate, 3),
    (Op::Delete, 3),
];

/// Roulette-select an operator with probability proportional to its weight.
fn pick_op(rng: &mut impl Rng) -> Op {
    let total: u32 = OPERATORS.iter().map(|&(_, w)| w).sum();
    let mut pick = rng.random_range(0..total);
    for &(op, w) in OPERATORS {
        if pick < w {
            return op;
        }
        pick -= w;
    }
    OPERATORS.last().unwrap().0
}

/// One sample from N(0, sigma) via the Box-Muller transform. `rand 0.10` ships no
/// normal distribution and we avoid adding `rand_distr`, so we derive it from two
/// uniforms. Gaussian (vs the previous uniform) perturbations give the ES both
/// fine refinement (most mass near 0) and an occasional larger exploratory jump
/// (the tail) — the previous uniform jitter had no tail.
fn gaussian(rng: &mut impl Rng, sigma: f32) -> f32 {
    let u1: f32 = rng.random_range(1e-7_f32..1.0); // avoid ln(0)
    let u2: f32 = rng.random_range(0.0_f32..1.0);
    let mag = (-2.0 * u1.ln()).sqrt();
    mag * (std::f32::consts::TAU * u2).cos() * sigma
}

/// Roulette-select a grid cell index with probability proportional to its error
/// weight. Falls back to uniform when the grid is all-zero (e.g. a perfect match).
fn sample_error_cell(grid: &[u32], rng: &mut impl Rng) -> usize {
    let total: u64 = grid.iter().map(|&w| w as u64).sum();
    if total == 0 {
        return rng.random_range(0..grid.len());
    }
    let mut pick = rng.random_range(0..total);
    for (i, &w) in grid.iter().enumerate() {
        let w = w as u64;
        if pick < w {
            return i;
        }
        pick -= w;
    }
    grid.len() - 1
}

/// Map an error-grid cell plus intra-cell jitter (`jx`, `jy` in [0,1]) to a
/// clip-space point in [-1,1]². Cell row 0 is the top of the image, so clip y is
/// flipped to match (the fitness shader bins with row 0 = top).
fn cell_to_clip(cell: usize, jx: f32, jy: f32) -> (f32, f32) {
    let g = ERROR_GRID_DIM as f32;
    let gx = (cell % ERROR_GRID_DIM as usize) as f32;
    let gy = (cell / ERROR_GRID_DIM as usize) as f32;
    let u = (gx + jx) / g; // [0,1] across the image width
    let v = (gy + jy) / g; // [0,1] top→bottom
    (u * 2.0 - 1.0, 1.0 - v * 2.0)
}

/// Like a uniformly-seeded triangle, but the centre is drawn from a high-error
/// grid cell rather than uniformly across the canvas. Delegates triangle
/// construction to `genome::seeded_triangle`.
fn error_seeded_triangle(
    goal: &GoalImage,
    error_grid: &[u32],
    rng: &mut impl Rng,
    max_radius: f32,
) -> [Vertex; 3] {
    let cell = sample_error_cell(error_grid, rng);
    let (cx, cy) = cell_to_clip(cell, rng.random_range(0.0..1.0), rng.random_range(0.0..1.0));
    seeded_triangle(goal, cx, cy, max_radius, rng)
}

/// Replace one triangle (chosen near a high-error cell) with its 4 midpoint
/// children, growing the genome by 3 triangles. No-op if the genome is empty or
/// a split would exceed `max_triangles` (a split adds 3). This is the only
/// growth path: it is applied as a mutation candidate, so `(1+λ)` selection
/// keeps it only when the added detail improves fitness.
fn grow_by_split(
    genome: &mut Vec<Vertex>,
    goal: &GoalImage,
    error_grid: &[u32],
    max_triangles: usize,
    rng: &mut impl Rng,
) {
    let n = genome.len() / 3;
    if n == 0 || n + 3 > max_triangles {
        return;
    }
    // Bias toward error: pick the triangle whose centroid is nearest a
    // roulette-selected high-error cell centre.
    let cell = sample_error_cell(error_grid, rng);
    let (tx, ty) = cell_to_clip(cell, 0.5, 0.5);
    let mut best_t = 0usize;
    let mut best_d = f32::INFINITY;
    for t in 0..n {
        let (cx, cy) = triangle_centroid(genome, t);
        let d = (cx - tx) * (cx - tx) + (cy - ty) * (cy - ty);
        if d < best_d {
            best_d = d;
            best_t = t;
        }
    }
    let b = best_t * 3;
    // Read the parent (Copy) before splicing, then replace it in place so the 4
    // children inherit its z/draw position (alpha-blend order preserved).
    let children = split_triangle(genome[b], genome[b + 1], genome[b + 2], goal);
    genome.splice(b..b + 3, children);
}

/// Apply one random mutation to a clone of `parent`, returning the child and the
/// `OpKind` it exercised (for per-type step-size adaptation). Positional nudges
/// use `sigma_pos`; recolour/alpha use `sigma_col`; both are Gaussian. Structural
/// changes (add/delete/z-swap) happen rarely and carry no step size.
pub(crate) fn mutate(
    parent: &[Vertex],
    sigmas: StepSizes,
    min_triangles: usize,
    max_triangles: usize,
    goal: &GoalImage,
    error_grid: &[u32],
    rng: &mut impl Rng,
) -> (Vec<Vertex>, OpKind) {
    let mut child = parent.to_vec();
    let n = child.len() / 3;
    if n == 0 {
        // Pathological: rebuild from scratch.
        return (init_genome(goal, min_triangles, rng), OpKind::Structural);
    }

    let op = pick_op(rng);
    match op {
        Op::NudgeVertex => {
            // Nudge a single vertex of one triangle (Gaussian, sigma_pos).
            let t = rng.random_range(0..n);
            let v = rng.random_range(0..3);
            let idx = t * 3 + v;
            child[idx].position[0] = (child[idx].position[0] + gaussian(rng, sigmas.pos)).clamp(-1.0, 1.0);
            child[idx].position[1] = (child[idx].position[1] + gaussian(rng, sigmas.pos)).clamp(-1.0, 1.0);
        }
        Op::Recolor => {
            // Recolour all three vertices of one triangle (RGB, Gaussian, sigmas.col).
            let t = rng.random_range(0..n);
            let dr = gaussian(rng, sigmas.col);
            let dg = gaussian(rng, sigmas.col);
            let db = gaussian(rng, sigmas.col);
            for v in 0..3 {
                let c = &mut child[t * 3 + v].color;
                c[0] = (c[0] + dr).clamp(0.0, 1.0);
                c[1] = (c[1] + dg).clamp(0.0, 1.0);
                c[2] = (c[2] + db).clamp(0.0, 1.0);
            }
        }
        Op::NudgeAlpha => {
            // Nudge the alpha of one triangle (Gaussian, sigmas.col).
            let t = rng.random_range(0..n);
            let da = gaussian(rng, sigmas.col);
            for v in 0..3 {
                let a = &mut child[t * 3 + v].color[3];
                *a = (*a + da).clamp(0.0, 1.0);
            }
        }
        Op::Split => {
            // Split a high-error triangle into 4 midpoint children — the only
            // growth path, gated by selection (see `grow_by_split`).
            grow_by_split(&mut child, goal, error_grid, max_triangles, rng);
        }
        Op::SwapZ => {
            // Swap z-order with a neighbouring triangle.
            if n > 1 {
                let t = rng.random_range(0..n - 1);
                for v in 0..3 {
                    child.swap(t * 3 + v, (t + 1) * 3 + v);
                }
            }
        }
        Op::AddTriangle => {
            // Add a new triangle seeded in a high-error region.
            if n < max_triangles {
                let seed_radius =
                    (0.2 * (INITIAL_TRIANGLES as f32 / n as f32).sqrt()).clamp(0.02, 0.2);
                let tri = error_seeded_triangle(goal, error_grid, rng, seed_radius);
                let insert_at = rng.random_range(0..=n) * 3;
                for (offset, vert) in tri.iter().enumerate() {
                    child.insert(insert_at + offset, *vert);
                }
            }
        }
        Op::Relocate => {
            // Relocate an existing triangle's centroid to a high-error cell and
            // recolour it to that region — recycles triangles that aren't helping.
            let t = rng.random_range(0..n);
            let base = t * 3;
            let cell = sample_error_cell(error_grid, rng);
            let (tx, ty) =
                cell_to_clip(cell, rng.random_range(0.0..1.0), rng.random_range(0.0..1.0));
            let (ccx, ccy) = triangle_centroid(&child, t);
            let (dx, dy) = (tx - ccx, ty - ccy);
            // Move + clamp first, then recolour from the triangle's actual
            // post-clamp centroid: near the border a vertex can clamp, so the
            // landed centroid differs from the target (tx, ty) and we want the
            // colour of where the triangle actually ends up.
            for v in 0..3 {
                child[base + v].position[0] = (child[base + v].position[0] + dx).clamp(-1.0, 1.0);
                child[base + v].position[1] = (child[base + v].position[1] + dy).clamp(-1.0, 1.0);
            }
            let (acx, acy) = triangle_centroid(&child, t);
            let col = sample_goal_color(goal, acx, acy, child[base].color[3]);
            for v in 0..3 {
                child[base + v].color[0] = col[0];
                child[base + v].color[1] = col[1];
                child[base + v].color[2] = col[2];
            }
        }
        Op::Delete => {
            // Delete one triangle.
            if n > min_triangles {
                let t = rng.random_range(0..n);
                for _ in 0..3 {
                    child.remove(t * 3);
                }
            }
        }
    }
    (child, op.kind())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::GRID_CELLS;
    use crate::genome::init_genome;
    use crate::test_support::make_solid_goal;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashSet;

    #[test]
    fn operator_weights_sum_to_100_and_map_to_expected_kinds() {
        let total: u32 = OPERATORS.iter().map(|&(_, w)| w).sum();
        assert_eq!(total, 100, "operator weights should sum to 100");
        assert_eq!(Op::NudgeVertex.kind(), OpKind::Positional);
        assert_eq!(Op::Recolor.kind(), OpKind::Chromatic);
        assert_eq!(Op::NudgeAlpha.kind(), OpKind::Chromatic);
        for op in [Op::Split, Op::SwapZ, Op::AddTriangle, Op::Relocate, Op::Delete] {
            assert_eq!(op.kind(), OpKind::Structural, "{op:?} should be structural");
        }
    }

    #[test]
    fn pick_op_reaches_every_operator() {
        let mut rng = StdRng::seed_from_u64(123);
        let seen: HashSet<Op> = (0..10_000).map(|_| pick_op(&mut rng)).collect();
        assert_eq!(seen.len(), OPERATORS.len(), "all operators should be reachable");
    }

    #[test]
    fn gaussian_has_zero_mean_and_unit_std() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 100_000;
        let samples: Vec<f32> = (0..n).map(|_| gaussian(&mut rng, 1.0)).collect();
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let var: f32 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std = var.sqrt();
        assert!(mean.abs() < 0.05, "mean {mean} not ~0");
        assert!((std - 1.0).abs() < 0.05, "std {std} not ~1");
    }

    #[test]
    fn sample_error_cell_favours_high_error() {
        // Cell 2 dominates; with a fixed seed it should be picked almost always.
        let grid = vec![0u32, 0, 100, 0];
        let mut rng = StdRng::seed_from_u64(1);
        let hits = (0..1000).filter(|_| sample_error_cell(&grid, &mut rng) == 2).count();
        assert!(hits >= 999, "dominant cell chosen {hits}/1000");
    }

    #[test]
    fn sample_error_cell_uniform_when_empty() {
        // All-zero grid -> uniform fallback over the four cells.
        let grid = vec![0u32; 4];
        let mut rng = StdRng::seed_from_u64(2);
        let mut counts = [0usize; 4];
        for _ in 0..4000 {
            counts[sample_error_cell(&grid, &mut rng)] += 1;
        }
        assert!(counts.iter().all(|&c| c > 700), "not roughly uniform: {counts:?}");
    }

    #[test]
    fn cell_to_clip_stays_in_cell_bounds() {
        // For ERROR_GRID_DIM=16, cell 0 is the top-left; its clip-x spans
        // [-1, -1 + 2/16] and clip-y spans [1 - 2/16, 1].
        let g = ERROR_GRID_DIM as f32;
        let (cx, cy) = cell_to_clip(0, 0.5, 0.5);
        assert!(cx >= -1.0 && cx <= -1.0 + 2.0 / g, "cx {cx} out of cell 0");
        assert!(cy <= 1.0 && cy >= 1.0 - 2.0 / g, "cy {cy} out of cell 0");
    }

    #[test]
    fn grow_by_split_respects_cap() {
        let goal = make_solid_goal(64, [100, 150, 200]);
        let mut rng = StdRng::seed_from_u64(99);
        let grid = vec![1u32; GRID_CELLS]; // flat error -> any triangle eligible
        let mut genome = init_genome(&goal, 5, &mut rng); // 5 triangles = 15 verts

        // Below cap: one split replaces 1 triangle with 4 -> net +3 triangles.
        grow_by_split(&mut genome, &goal, &grid, 100, &mut rng);
        assert_eq!(genome.len() / 3, 8, "split should grow 5 -> 8 triangles");

        // At/over cap: n + 3 > cap -> no-op.
        let before = genome.clone();
        grow_by_split(&mut genome, &goal, &grid, 9, &mut rng); // 8 + 3 = 11 > 9
        assert_eq!(genome, before, "split must be a no-op when it would exceed the cap");
    }
}
