//! The (1+λ)-ES search driver: coarse-to-fine phase schedule, per-type
//! self-adapted step sizes (1/5 success rule), and plateau-triggered promotion.

use crate::fitness::{build_pyramid, FitnessCalc, LAMBDA};
use crate::genome::{init_genome, Vertex, INITIAL_TRIANGLES, MAX_TRIANGLES};
use crate::goal::GoalImage;
use crate::variation::{mutate, OpKind, StepSizes};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

// Absolute floor on triangle count: `delete` will not prune below this.
const MIN_TRIANGLES: usize = 8;

// 1/5 success rule: re-evaluate σ this often.
const SIGMA_WINDOW: u64 = 50;

// Minimum steps in a phase before promotion is considered.
const PHASE_MIN_STEPS: u64 = 400;

// Promote when the last PLATEAU_WINDOW steps yielded fewer than this many
// successful improvements.
const PLATEAU_WINDOW: u64 = 100;
const PLATEAU_ACCEPTS: u64 = 5;

// Snapshot a PNG every N steps that produced a successful improvement.
const SNAPSHOT_EVERY_IMPROVEMENT: u64 = 100;

// Hard cap on total ES steps (sanity).
const MAX_STEPS: u64 = 500_000;

// Per-type self-adapted step-size clamps. Position lives in clip-space [-1,1];
// colour/alpha in [0,1], so they get independent ranges.
const SIGMA_POS_MIN: f32 = 0.005;
const SIGMA_POS_MAX: f32 = 0.5;
const SIGMA_COL_MIN: f32 = 0.003;
const SIGMA_COL_MAX: f32 = 0.4;
// Single-vertex (gradient) recolour lives in the same [0,1] colour space as
// whole-triangle recolour, so it reuses the colour clamps but adapts separately.
const SIGMA_GRAD_MIN: f32 = 0.003;
const SIGMA_GRAD_MAX: f32 = 0.4;

#[derive(Clone)]
pub(crate) struct Phase {
    // Maximum triangles the genome may grow to in this phase (a ceiling, not a
    // fill target — growth is organic via `split`).
    cap: usize,
    pyramid_level: usize,
    // Initial step sizes for this phase, self-adapted by per-type 1/5 rules.
    initial_sigma_pos: f32,
    initial_sigma_col: f32,
    // Gradient (single-vertex recolour) σ starts level with the colour σ and
    // diverges from it under adaptation.
    initial_sigma_grad: f32,
}

// Coarse-to-fine schedule: pyramid level + initial σ per phase, with a capacity
// cap that rises to MAX_TRIANGLES at the finest level. Promotion advances this
// schedule on plateau; the genome grows toward each cap via `split`.
const PHASES: &[Phase] = &[
    Phase { cap: 300,           pyramid_level: 0, initial_sigma_pos: 0.30, initial_sigma_col: 0.20, initial_sigma_grad: 0.20 }, // 128² coarse
    Phase { cap: 800,           pyramid_level: 1, initial_sigma_pos: 0.18, initial_sigma_col: 0.12, initial_sigma_grad: 0.12 }, // 256² medium
    Phase { cap: 2000,          pyramid_level: 2, initial_sigma_pos: 0.10, initial_sigma_col: 0.08, initial_sigma_grad: 0.08 }, // 512² fine
    Phase { cap: MAX_TRIANGLES, pyramid_level: 2, initial_sigma_pos: 0.05, initial_sigma_col: 0.04, initial_sigma_grad: 0.04 }, // 512² finest
];

pub(crate) struct EsConfig {
    pub(crate) phases: Vec<Phase>,
    pub(crate) max_steps: u64,
    pub(crate) lambda: usize,
    pub(crate) snapshot_every: Option<u64>,
    // When set, the loop stops at the end of the current step once this flips to
    // `true` (a Ctrl-C handler in `main` drives it for `--infinite` runs), then
    // falls through to the normal final-snapshot/summary path.
    pub(crate) stop_flag: Option<Arc<AtomicBool>>,
}

impl EsConfig {
    pub(crate) fn production() -> Self {
        Self {
            phases: PHASES.to_vec(),
            max_steps: MAX_STEPS,
            lambda: LAMBDA,
            snapshot_every: Some(SNAPSHOT_EVERY_IMPROVEMENT),
            stop_flag: None,
        }
    }
}

pub(crate) struct EsResult {
    pub(crate) initial_fitness: usize,
    pub(crate) final_fitness: usize,
    pub(crate) steps_run: u64,
}

/// Per-type self-adapted step sizes plus the 1/5 success rule that drives them.
/// Owns the current σ pair and the rolling beat-the-parent tallies. The single
/// `reset_window` is the one place those tallies are cleared — promotion and the
/// terminal σ-restart both route through it, so the reset sites can't drift
/// apart (the bug the old three hand-written reset blocks invited).
struct OneFifthRule {
    sigmas: StepSizes,
    window_steps: u64,
    pos_gen: u64,
    pos_better: u64,
    col_gen: u64,
    col_better: u64,
    grad_gen: u64,
    grad_better: u64,
}

impl OneFifthRule {
    fn new(phase: &Phase) -> Self {
        let mut r = Self {
            sigmas: StepSizes { pos: 0.0, col: 0.0, grad: 0.0 },
            window_steps: 0,
            pos_gen: 0,
            pos_better: 0,
            col_gen: 0,
            col_better: 0,
            grad_gen: 0,
            grad_better: 0,
        };
        r.restart(phase);
        r
    }

    /// Reset σ to a phase's initial sizes and clear the window (on promotion or
    /// terminal restart).
    fn restart(&mut self, phase: &Phase) {
        self.sigmas = StepSizes {
            pos: phase.initial_sigma_pos,
            col: phase.initial_sigma_col,
            grad: phase.initial_sigma_grad,
        };
        self.reset_window();
    }

    /// Clear the rolling 1/5-rule window — the single counter-reset path.
    fn reset_window(&mut self) {
        self.window_steps = 0;
        self.pos_gen = 0;
        self.pos_better = 0;
        self.col_gen = 0;
        self.col_better = 0;
        self.grad_gen = 0;
        self.grad_better = 0;
    }

    /// Tally one generated candidate (and whether it beat the parent) against
    /// its step-size class. Structural ops carry no step size and are ignored.
    fn record(&mut self, kind: OpKind, improved: bool) {
        match kind {
            OpKind::Positional => {
                self.pos_gen += 1;
                self.pos_better += improved as u64;
            }
            OpKind::Chromatic => {
                self.col_gen += 1;
                self.col_better += improved as u64;
            }
            OpKind::Gradient => {
                self.grad_gen += 1;
                self.grad_better += improved as u64;
            }
            OpKind::Structural => {}
        }
    }

    /// Advance the window by one ES step; when it fills, adapt each σ toward a
    /// ~20% beat-the-parent rate (grow above it, shrink below) and clear the
    /// window. A class is left untouched if it produced no candidates.
    fn end_step(&mut self) {
        self.window_steps += 1;
        if self.window_steps < SIGMA_WINDOW {
            return;
        }
        if self.pos_gen > 0 {
            let rate = self.pos_better as f32 / self.pos_gen as f32;
            self.sigmas.pos = adapt_sigma(self.sigmas.pos, rate, SIGMA_POS_MIN, SIGMA_POS_MAX);
        }
        if self.col_gen > 0 {
            let rate = self.col_better as f32 / self.col_gen as f32;
            self.sigmas.col = adapt_sigma(self.sigmas.col, rate, SIGMA_COL_MIN, SIGMA_COL_MAX);
        }
        if self.grad_gen > 0 {
            let rate = self.grad_better as f32 / self.grad_gen as f32;
            self.sigmas.grad = adapt_sigma(self.sigmas.grad, rate, SIGMA_GRAD_MIN, SIGMA_GRAD_MAX);
        }
        self.reset_window();
    }
}

/// One 1/5-rule step on a single σ: grow ×1.15 above a 20% success rate, shrink
/// ×0.85 below it, clamped to `[min, max]`.
fn adapt_sigma(sigma: f32, rate: f32, min: f32, max: f32) -> f32 {
    if rate > 0.2 {
        (sigma * 1.15).min(max)
    } else if rate < 0.2 {
        (sigma * 0.85).max(min)
    } else {
        sigma
    }
}

/// Coarse-to-fine phase position and the plateau detector that promotes it:
/// tracks the current phase index, steps spent in it, and accepts since the last
/// plateau check.
struct PhaseSchedule {
    idx: usize,
    phase_step: u64,
    accepts: u64,
}

impl PhaseSchedule {
    fn new() -> Self {
        Self { idx: 0, phase_step: 0, accepts: 0 }
    }

    /// Record one ES step and whether it improved the parent.
    fn record(&mut self, accepted: bool) {
        self.phase_step += 1;
        self.accepts += accepted as u64;
    }

    /// At a plateau-check boundary (enough time in-phase, on a `PLATEAU_WINDOW`
    /// stride), return whether the phase has stagnated and clear the accept
    /// count. Returns `None` away from a boundary.
    fn check_plateau(&mut self) -> Option<bool> {
        if self.phase_step >= PHASE_MIN_STEPS && self.phase_step.is_multiple_of(PLATEAU_WINDOW) {
            let plateaued = self.accepts < PLATEAU_ACCEPTS;
            self.accepts = 0;
            Some(plateaued)
        } else {
            None
        }
    }

    /// Advance to the next phase, restarting the in-phase step counter.
    fn advance(&mut self) {
        self.idx += 1;
        self.phase_step = 0;
    }

    /// Restart the in-phase step counter without advancing (terminal σ-restart).
    fn restart_phase(&mut self) {
        self.phase_step = 0;
    }
}

/// Score one genome against `calc`, returning `(fitness, residual error grid)`.
fn score(calc: &FitnessCalc, genome: &[Vertex]) -> (usize, Vec<u32>) {
    let mut evals = calc.fitness_of_batch(&[genome]);
    let ev = evals.swap_remove(0);
    (ev.score, ev.error_grid)
}

/// One-line phase banner: `<label> | cap … | level …² | σ … | fitness …`.
fn log_phase(label: &str, phase: &Phase, level_size: u32, sigmas: StepSizes, fitness: usize) {
    println!(
        "{label} | cap {} | level {} ({}²) | σ_pos={:.3} σ_col={:.3} σ_grad={:.3} | fitness {}",
        phase.cap, phase.pyramid_level, level_size, sigmas.pos, sigmas.col, sigmas.grad, fitness
    );
}

pub(crate) fn run_es(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    goal: GoalImage,
    cfg: EsConfig,
) -> EsResult {
    let pyramid = build_pyramid(&device, &queue, &goal);
    let full_res = pyramid.len() - 1; // index of full-resolution level (for snapshots)

    let mut rng = rand::rng();

    // ---- Phase 0: initialise the genome (INITIAL_TRIANGLES, capped at phase 0's ceiling) ----
    let mut schedule = PhaseSchedule::new();
    let mut sigma = OneFifthRule::new(&cfg.phases[schedule.idx]);
    let mut current = init_genome(
        &goal,
        INITIAL_TRIANGLES.min(cfg.phases[schedule.idx].cap),
        &mut rng,
    );
    let (mut current_fitness, mut parent_error_grid) =
        score(&pyramid[cfg.phases[schedule.idx].pyramid_level], &current);
    let initial_fitness = current_fitness;

    log_phase(
        &format!("Phase {}", schedule.idx),
        &cfg.phases[schedule.idx],
        pyramid[cfg.phases[schedule.idx].pyramid_level].texture_size(),
        sigma.sigmas,
        current_fitness,
    );

    let mut step: u64 = 0;
    let mut improvements_total: u64 = 0;
    let started = Instant::now();
    let mut last_log = Instant::now();

    // Trigger a snapshot of the initial state at full resolution so the
    // triangles/ directory has something to compare against.
    if cfg.snapshot_every.is_some() {
        let _ = std::fs::create_dir_all("triangles");
        pyramid[full_res].snapshot(&current, Path::new("triangles/image0.png"));
    }

    while step < cfg.max_steps {
        // Graceful stop (e.g. Ctrl-C in `--infinite` mode): leave the loop at a
        // step boundary so the final snapshot and summary below still run.
        if cfg.stop_flag.as_ref().is_some_and(|f| f.load(Ordering::Relaxed)) {
            println!("Stop requested — halting at step {step}.");
            break;
        }
        let phase = &cfg.phases[schedule.idx];
        let calc = &pyramid[phase.pyramid_level];
        // `split`/`add` may grow the genome up to this phase's cap; `delete` may
        // prune down to MIN_TRIANGLES. Growth is organic and selection-gated, so
        // there is no fill target to hold near.
        let max_triangles = phase.cap;

        // (1+λ): produce λ candidates and evaluate them all in one GPU submit.
        let mut candidates: Vec<Vec<Vertex>> = Vec::with_capacity(cfg.lambda);
        let mut kinds: Vec<OpKind> = Vec::with_capacity(cfg.lambda);
        for _ in 0..cfg.lambda {
            let (child, kind) = mutate(
                &current,
                sigma.sigmas,
                MIN_TRIANGLES,
                max_triangles,
                &goal,
                &parent_error_grid,
                &mut rng,
            );
            candidates.push(child);
            kinds.push(kind);
        }
        let cand_refs: Vec<&[Vertex]> = candidates.iter().map(|c| c.as_slice()).collect();
        let evals = calc.fitness_of_batch(&cand_refs);

        // Select the best improver over the parent; tally every candidate for
        // the per-type 1/5 rule (against the parent's fitness, pre-acceptance).
        let mut best_idx: Option<usize> = None;
        let mut best_fit = current_fitness;
        for (i, e) in evals.iter().enumerate() {
            sigma.record(kinds[i], e.score > current_fitness);
            if e.score > best_fit {
                best_fit = e.score;
                best_idx = Some(i);
            }
        }

        let accepted = best_idx.is_some();
        if let Some(i) = best_idx {
            parent_error_grid = evals[i].error_grid.clone();
            current = candidates.swap_remove(i);
            current_fitness = best_fit;
            improvements_total += 1;
        }
        step += 1;
        sigma.end_step();
        schedule.record(accepted);

        // Snapshot occasionally on improvement.
        if let Some(snap_every) = cfg.snapshot_every
            && accepted
            && improvements_total > 0
            && improvements_total.is_multiple_of(snap_every)
        {
            let path_buf = format!("triangles/image{}.png", step);
            pyramid[full_res].snapshot(&current, Path::new(&path_buf));
        }

        // Periodic progress log (rate-limited so output stays readable).
        if last_log.elapsed().as_secs_f32() >= 1.0 {
            println!(
                "step {:>6} | phase {} | tris {:>3} | σ_pos={:.3} σ_col={:.3} σ_grad={:.3} | fit {:>7} | improvements {} | {:.1}/s",
                step,
                schedule.idx,
                current.len() / 3,
                sigma.sigmas.pos,
                sigma.sigmas.col,
                sigma.sigmas.grad,
                current_fitness,
                improvements_total,
                step as f64 / started.elapsed().as_secs_f64()
            );
            last_log = Instant::now();
        }

        // Phase promotion / terminal σ-restart, gated on a plateau (few accepts
        // in the last window after enough time in-phase). `check_plateau` resets
        // its accept tally on every boundary; we act only when it stagnates.
        if let Some(true) = schedule.check_plateau() {
            if schedule.idx + 1 < cfg.phases.len() {
                schedule.advance();
                let new_phase = &cfg.phases[schedule.idx];
                // No genome growth on promotion: it only raises the cap and
                // sharpens evaluation. The genome grows organically via `split`.
                sigma.restart(new_phase);
                // Re-score against the new (possibly higher-resolution) level.
                (current_fitness, parent_error_grid) =
                    score(&pyramid[new_phase.pyramid_level], &current);
                log_phase(
                    &format!("→ Phase {}", schedule.idx),
                    new_phase,
                    pyramid[new_phase.pyramid_level].texture_size(),
                    sigma.sigmas,
                    current_fitness,
                );
                if cfg.snapshot_every.is_some() {
                    let path_buf = format!("triangles/image{}_phase{}.png", step, schedule.idx);
                    pyramid[full_res].snapshot(&current, Path::new(&path_buf));
                }
            } else {
                // No further phases: kick σ back to this phase's initial sizes so
                // the search re-explores instead of grinding at near-zero step
                // size, and restart the in-phase clock for the next plateau check.
                let old = sigma.sigmas;
                sigma.restart(phase);
                schedule.restart_phase();
                println!(
                    "⤴ Sigma restart (no further phases) | σ_pos {:.3}→{:.3} σ_col {:.3}→{:.3} σ_grad {:.3}→{:.3}",
                    old.pos, sigma.sigmas.pos, old.col, sigma.sigmas.col, old.grad, sigma.sigmas.grad
                );
            }
        }
    }

    println!(
        "Done. {} steps in {:.1}s, {} improvements, final fitness {}",
        step,
        started.elapsed().as_secs_f64(),
        improvements_total,
        current_fitness
    );
    if cfg.snapshot_every.is_some() {
        pyramid[full_res].snapshot(&current, Path::new("triangles/final.png"));
    }

    EsResult {
        initial_fitness,
        final_fitness: current_fitness,
        steps_run: step,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{init_test_wgpu, make_checker_goal};

    fn test_phase(sp: f32, sc: f32) -> Phase {
        Phase {
            cap: 100,
            pyramid_level: 0,
            initial_sigma_pos: sp,
            initial_sigma_col: sc,
            initial_sigma_grad: sc,
        }
    }

    #[test]
    fn adapt_sigma_grows_shrinks_and_clamps() {
        // Above 20%: grow ×1.15, clamped to max.
        assert!((adapt_sigma(0.1, 0.5, 0.01, 1.0) - 0.115).abs() < 1e-6);
        assert_eq!(adapt_sigma(0.95, 1.0, 0.01, 1.0), 1.0, "grow clamps at max");
        // Below 20%: shrink ×0.85, clamped to min.
        assert!((adapt_sigma(0.1, 0.0, 0.01, 1.0) - 0.085).abs() < 1e-6);
        assert_eq!(adapt_sigma(0.011, 0.0, 0.01, 1.0), 0.01, "shrink clamps at min");
        // Exactly 20%: unchanged.
        assert_eq!(adapt_sigma(0.1, 0.2, 0.01, 1.0), 0.1);
    }

    #[test]
    fn one_fifth_rule_grows_pos_on_success_window() {
        let mut r = OneFifthRule::new(&test_phase(0.1, 0.1));
        assert_eq!((r.sigmas.pos, r.sigmas.col), (0.1, 0.1));
        // A full window of always-improving positional candidates grows σ_pos;
        // σ_col is untouched (no chromatic candidates were recorded).
        for _ in 0..SIGMA_WINDOW {
            r.record(OpKind::Positional, true);
            r.end_step();
        }
        assert!(r.sigmas.pos > 0.1, "σ_pos should grow on a high success rate");
        assert_eq!(r.sigmas.col, 0.1, "σ_col untouched without chromatic candidates");
        assert_eq!(r.window_steps, 0, "window resets after adapting");
    }

    #[test]
    fn one_fifth_rule_shrinks_col_on_failure_and_reset_clears() {
        let mut r = OneFifthRule::new(&test_phase(0.1, 0.1));
        for _ in 0..SIGMA_WINDOW {
            r.record(OpKind::Chromatic, false);
            r.end_step();
        }
        assert!(r.sigmas.col < 0.1, "σ_col should shrink on a zero success rate");

        // reset_window zeroes every rolling tally.
        r.record(OpKind::Positional, true);
        r.record(OpKind::Gradient, true);
        r.reset_window();
        assert_eq!(
            (
                r.window_steps,
                r.pos_gen,
                r.pos_better,
                r.col_gen,
                r.col_better,
                r.grad_gen,
                r.grad_better
            ),
            (0, 0, 0, 0, 0, 0, 0)
        );
    }

    #[test]
    fn one_fifth_rule_adapts_grad_independently() {
        let mut r = OneFifthRule::new(&test_phase(0.1, 0.1));
        // A full window of always-improving gradient candidates grows σ_grad
        // alone; σ_pos and σ_col are untouched (no candidates of their class).
        for _ in 0..SIGMA_WINDOW {
            r.record(OpKind::Gradient, true);
            r.end_step();
        }
        assert!(r.sigmas.grad > 0.1, "σ_grad should grow on a high success rate");
        assert_eq!(r.sigmas.pos, 0.1, "σ_pos untouched without positional candidates");
        assert_eq!(r.sigmas.col, 0.1, "σ_col untouched without chromatic candidates");
    }

    #[test]
    fn phase_schedule_detects_plateau_at_first_boundary() {
        // With zero accepts, the first plateau check fires exactly at
        // PHASE_MIN_STEPS (the first PLATEAU_WINDOW stride >= the minimum) and
        // reports a plateau. Earlier strides (100/200/300) return None because
        // they precede PHASE_MIN_STEPS.
        let mut s = PhaseSchedule::new();
        let mut first_plateau_at = None;
        for i in 1..=PHASE_MIN_STEPS {
            s.record(false);
            if s.check_plateau() == Some(true) {
                first_plateau_at = Some(i);
                break;
            }
        }
        assert_eq!(first_plateau_at, Some(PHASE_MIN_STEPS));
    }

    #[test]
    fn phase_caps_are_monotonic_and_reach_max() {
        let caps: Vec<usize> = PHASES.iter().map(|p| p.cap).collect();
        assert!(
            caps.windows(2).all(|w| w[1] > w[0]),
            "phase caps must be strictly increasing: {caps:?}"
        );
        assert_eq!(
            *caps.last().unwrap(),
            MAX_TRIANGLES,
            "final phase cap must be the global triangle ceiling"
        );
    }

    #[test]
    fn ga_improves_on_synthetic_checker() {
        let goal = make_checker_goal(32);
        let (device, queue) = init_test_wgpu();
        // Single-phase config for the test. pyramid_level 0 is the coarsest
        // level (build_pyramid sizes = [full/4, full/2, full]); for a 32×32
        // goal this evaluates at 8×8 — fast and plenty for a smoke test.
        let test_phases = vec![Phase {
            cap: 6,
            pyramid_level: 0,
            initial_sigma_pos: 0.1,
            initial_sigma_col: 0.1,
            initial_sigma_grad: 0.1,
        }];
        let result = run_es(
            device,
            queue,
            goal,
            EsConfig {
                phases: test_phases,
                max_steps: 30,
                lambda: 4,
                snapshot_every: None,
                stop_flag: None,
            },
        );
        assert!(
            result.steps_run > 0,
            "ES loop must run at least one step"
        );
        // fitness_of returns usize in [0, 1_000_000] where HIGHER = better fit.
        // A stuck-at-zero result usually means the GPU pipeline returned
        // garbage (e.g., a bind-group or texture-format mismatch silently
        // produced an empty render) — that's the most likely silent failure
        // mode of the wgpu migration, so guard against it explicitly.
        assert!(
            result.final_fitness > 0,
            "fitness stuck at zero — pipeline likely broken"
        );
        assert!(
            result.final_fitness <= 1_000_000,
            "fitness out of expected range: {}",
            result.final_fitness
        );
        assert!(
            result.final_fitness >= result.initial_fitness,
            "fitness should not regress: initial={}, final={}",
            result.initial_fitness,
            result.final_fitness
        );
    }
}
