# Future Directions — Quality Ceiling & Paths Forward

**Date:** 2026-06-08
**Binary:** `polygenvo`
**Status:** Notes for a future session (no code attached)

## TL;DR

The (1+λ)-ES approach is near its **practical quality ceiling**. The wins worth
keeping were landed (perceptual metric + structure-first schedule). Exceeding the
ceiling now requires one of two **multi-day architectural projects**, not another
tweak — both scoped below.

## What's banked on `master`

- **ΔE2000 perceptual fitness** + precomputed goal-LAB + bilinear goal sampling.
  (`fitness.wgsl`/`fitness.rs`/`goal.rs`; see memory `fitness-metric-de2000`.)
- **Gradual coarse-to-fine cap ramp** (48→…→10000, plateau-gated) — fixed
  "detail too early / slow general similarity". (`es.rs` `PHASES`; memory
  `coarse-to-fine-cap-ramp`.)

## What's shelved (not on `master`)

- **Hybrid ES + gradient-descent polish** on branch
  `feat/hybrid-es-gradient-polish` (adds the `burn` dependency). Mechanism
  validated (+57 % ΔE2000 at a matched-resolution gate) but a **no-op
  in-pipeline**: cheap enough to not hang ⇒ can't beat the 512² acceptance gate;
  aggressive enough to help ⇒ ~24 s/call, starves the ES and freezes the window.
  Full write-up in `docs/superpowers/specs/2026-06-05-hybrid-es-gradient-polish-design.md`
  (Outcome section) and `…/plans/2026-06-05-hybrid-es-gradient-polish.md`.

## The ceiling finding

Five independent attempts to break the local-optimum ceiling — where a big
triangle committed early gets a wrong silhouette that hill-climbing can't fix —
**all came out neutral**: targeted-split repair, scale-relative vertex nudges,
affine/resize operator, gradient polish, and iterated-local-search (ILS) kicks.

Root cause (well established): fixing a wrong big triangle is a **downhill-first**
move. Elitist (1+λ) accepts a step only if it's *immediately* better, so it
rejects the first step of any real repair; and under a *hard* rasterizer the loss
is **discontinuous in vertex positions**, so there's no gradient pointing at the
fix. Local operators can't get around either fact. The only changes that actually
moved quality this session were *conceptual* (the metric, the schedule).

ILS specifics (so we don't repeat it): kicking the biggest triangles is either
unrecoverably destructive (−10 % fitness, no recovery in budget) or, if gentle,
no escape. A `best-ever` safety net keeps it from regressing but it stays neutral.

## Two paths forward (pick one, scope deliberately)

### Path A — Incremental ("dirty-region") fitness  *(robust, quality-neutral win)*

The 512² phase is GPU per-pixel-bound, yet every candidate re-renders and
re-scores the **whole** image even when a mutation touched one small triangle.
Re-render + re-score only the changed triangle's bounding box, keep a running
score, apply the delta — the way `primitive` (Fogleman) is fast.

- **Payoff:** potentially 5–50× more steps/sec for local mutations ⇒ strictly
  more search ⇒ strictly better-or-equal result, with **no quality tradeoff**.
- **Cost/risk:** real rewrite of the fitness pipeline. Alpha-OVER ordering
  complicates incrementality (changing triangle T forces re-composite of
  everything drawn after T within its bbox) — bounded for late/small triangles
  (the common case), less so for early big ones. Start by measuring the average
  bbox of accepted mutations to bound the expected speedup before committing.

### Path B — GPU-native differentiable rasterizer  *(highest ceiling, largest effort)*

Replace the shelved burn-subset polish with a hand-written **soft rasterizer +
backward pass in WGSL** (no framework), producing gradients for **all** triangles
cheaply on the existing wgpu device — so the silhouette wall genuinely dissolves
(all vertices move coherently downhill, no elitist rejection).

- **Payoff:** the real quality step-change; how modern image-triangulation is done.
- **Cost/risk:** research-grade. The shelved branch proves the *math* works and is
  a reference for coverage/barycentric/composite/Lab; the lesson learned is that
  per-triangle, framework-mediated GPU dispatch is too slow — this path must be
  **batched/vectorized and framework-free** to be viable. Keep it on the existing
  wgpu device (no CPU round-trip).

## Recommendation

If the goal is "more of the same, faster and better" → **Path A**. If it's "break
the silhouette ceiling" → **Path B**. Either is a fresh, deliberately-scoped
session (brainstorm → spec → plan), not an incremental edit. Until then, the
current `master` is a good, honest resting point.
