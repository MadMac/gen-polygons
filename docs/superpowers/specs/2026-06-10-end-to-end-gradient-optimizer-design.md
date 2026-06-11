# End-to-End Gradient Optimizer (+ Tiled Kernel, Vulkan) — Design

**Date:** 2026-06-10
**Binary:** `polygenvo`
**Status:** Approved for planning

## Problem & goal

The user's goal is the **best possible approximation of 512×512 images (and larger
later)**. The (1+λ)-ES is at the silhouette local-optimum ceiling (~942k–961k): big
triangles committed early get a wrong silhouette that elitist hill-climbing can't
repair. The merged Path-B branch built and **validated** a GPU differentiable
soft-rasterizer (CPU-reference-verified forward/backward, on-device Adam polish), but
its `--gradient-polish` is **not a live win**: (a) the brute-force backward is
O(num_tris²)/pixel and the CAS atomic-float scatter is contention-bound on the **GL
backend** → effectively hangs at 512²; (b) more fundamentally, the periodic **elitist
gate that protects the ES also rejects the gradient's reshaping** during active
evolution, so it never gets to dissolve the ceiling.

This project makes the differentiable rasterizer actually deliver quality at 512²+ by
addressing all three: the **backend**, the **kernel performance**, and the
**search strategy**.

## Decisions locked during brainstorming

- **Scope:** design the whole quality path (kernel + strategy), built in measure-gated
  phases.
- **Strategy = end-to-end gradient-primary.** Gradient descent continuously refines
  **all** triangles' positions + per-vertex colors every step (no per-step elitist
  gate blocking it); the ES handles only discrete/structural moves. A periodic
  **best-ever snapshot** is the safety net instead of a strict per-step gate.
- **Backend → Vulkan/PRIMARY**, switched **measure-first** (benchmark before/after; it
  may itself resolve much of the perf problem and re-scope the kernel work).
- **Reuse** the merged `softras_ref.rs` CPU oracle as the golden correctness reference
  for every kernel variant.

## Architecture

### 1. Backend: GL → PRIMARY (measure first)

`gpu::init_wgpu` currently forces `wgpu::Backends::GL`
([gpu.rs](../../../src/polygenvo/gpu.rs)). GL's compute/storage-atomics path is far
slower than Vulkan and is the likely real cause of the polish "hang" (atomic-CAS
contention on GL). Switch the headless device to wgpu's default high-performance
backends (Vulkan on the Linux dev box). Keep the `--show-window` path
([window.rs](../../../src/polygenvo/window.rs)) selecting a **surface-compatible**
adapter as it does today. **First action is a benchmark** (existing pipeline + the
merged brute-force polish at 128²/256²/512², GL vs Vulkan) — the data decides how much
kernel rework is genuinely needed.

### 2. Optimizer split (end-to-end)

- **Gradient (continuous):** every step, soft-rasterize the whole genome at the
  current pyramid level, backprop Lab-MSE, and Adam-update **all** vertex positions +
  per-vertex RGBA. No per-step elitist gate.
- **ES (discrete/structural only):** error-seeded **add**, **split**, **delete**,
  **z-order**, and **relocate** — the placement moves gradient can't make (gradients
  vanish far from a triangle's edges, so coarse placement stays the ES's job). These
  remain **hard-ΔE2000-gated** (cheap, discrete, no-regression).
- **Safety net:** a periodic **best-ever snapshot** (by hard ΔE2000) restored only on
  a *sustained* regression window — replaces the strict per-step gate that blocked the
  gradient, while still preventing runaway divergence.

### 3. Coarse-to-fine retained

Gradient runs at the current pyramid level with the soft-coverage temperature **τ
annealed sharp** over the run, so structure forms at coarse resolution and detail
accrues at 512² — same philosophy as today's `PHASES` schedule
([es.rs](../../../src/polygenvo/es.rs)).

### 4. Tiled differentiable kernel

Replaces the brute-force `softraster.wgsl` passes (kept as the reference/fallback):

- **Forward:** tile-binned compositing (each triangle assigned to the screen tiles its
  bbox overlaps); store only **per-pixel** state needed for backward (final color +
  the data to reconstruct per-triangle contributions) — **H×W memory, not
  H×W×triangles** (avoids the rejected memory wall).
- **Backward:** a workgroup per tile loops over only the triangles touching that tile,
  accumulating per-vertex gradients via **workgroup shared-memory reduction** (few/no
  global atomics) → drops O(num_tris²)/pixel to ~O(num_tris·overlap) and removes the
  atomic-contention wall. Vulkan provides the shared memory / subgroup ops this needs.
- **Correctness bar:** every variant tested **GPU == the CPU oracle** (`softras_ref.rs`),
  reusing the existing equality-test harness.

### 5. Run loop (`run_es` evolution)

Each step interleaves: a **gradient sub-step** (forward→backward→Adam on the whole
genome at the current level) + an **occasional gated structure op**; with periodic
**best-ever checkpointing**, plateau-driven **coarse-to-fine promotion** (as today),
and the existing snapshot/`--show-window`/`--infinite` machinery. Introduced behind a
flag first; becomes the default once it wins the A/B.

## Components / files

- **Modify** [gpu.rs](../../../src/polygenvo/gpu.rs) — backend selection (PRIMARY) +
  a benchmark harness/bin or test for the measurement step.
- **Modify** [softraster.wgsl](../../../src/polygenvo/softraster.wgsl) (or add
  `softraster_tiled.wgsl`) — tiled forward/backward; keep the brute-force entries as
  the oracle-tested fallback.
- **Modify** [gradient.rs](../../../src/polygenvo/gradient.rs) — tiled
  dispatch/binning, `PolishState`→a continuous `GradientRefiner` used every step (not
  just periodic), best-ever snapshot logic.
- **Modify** [es.rs](../../../src/polygenvo/es.rs) — the end-to-end run loop: gradient
  every step + structure-only ES ops + best-ever net; new tunables.
- **Modify** [variation.rs](../../../src/polygenvo/variation.rs) — narrow the operator
  table to structural ops (positional/chromatic vertex moves are now the gradient's
  job; keep add/split/delete/z/relocate). Possibly keep the old table behind the flag
  for the A/B.
- **Modify** [main.rs](../../../src/polygenvo/main.rs) + [CLAUDE.md](../../../CLAUDE.md)
  — flag + docs.
- **Reuse** `softras_ref.rs` (oracle) unchanged.

## Measure-gated milestones (avoid over-building)

1. **Backend switch + benchmark.** Vulkan vs GL on the existing pipeline and the merged
   brute-force polish at 128²/256²/512². Validate `--show-window` + smoke test on
   Vulkan. **Decide kernel scope from the numbers.**
2. **Tiled forward+backward kernel.** GPU==CPU oracle tests; benchmark vs brute-force.
3. **End-to-end run loop.** Gradient-primary + structural ES + best-ever net, behind a
   flag. Loop tests (improves hard ΔE2000 on the synthetic stuck scene; best-ever net
   never regresses below checkpoint).
4. **A/B vs current `master` at 512².** Measured ΔE2000 win + visible facet
   dissolution + completes in sane wall-clock → make it the default.
5. *(Stretch)* validate scaling on a larger (e.g. 1024²) image.

## Testing

- **Kernel:** GPU == CPU oracle (`softras_ref`) for tiled forward/backward, incl.
  triangles spanning tile boundaries.
- **Loop:** gradient-primary improves hard ΔE2000 on the synthetic stuck-big-triangle
  scene; best-ever net never returns a result worse than its checkpoint; structure ops
  stay gated.
- **Regression:** existing 35 tests stay green; the GPU smoke test re-validated on the
  Vulkan backend.
- **A/B:** scripted before/after at a matched wall-clock budget on `goal.png`.

## Risks & mitigations

- **Backend switch breaks a host / the window surface path** → keep window adapter
  selection; validate headless + windowed + smoke test on Vulkan; fall back to GL if a
  host lacks Vulkan (optional `--backend` override).
- **End-to-end loses strict per-step no-regression** → best-ever snapshot net;
  soft-loss gradient descent rarely regresses; structure ops still gated.
- **Lab-MSE vs ΔE2000 proxy gap** (no gate now) → acceptable; documented tuning lever;
  revisit only if A/B underdelivers.
- **Tiling edge-cases** (triangles spanning tiles, empty tiles) → standard but fiddly;
  the CPU-oracle equality tests are the guard.
- **Vanishing gradients for far placement** → by design the ES retains add/split/
  delete/relocate for coarse placement.

## Phase 1 results (2026-06-10, branch `feat/end-to-end-gradient-optimizer`)

Backend switched to PRIMARY (Vulkan selected on AMD RX 7800 XT / RADV); full suite (37
tests) green on Vulkan. Benchmark (`bench_backend`, ignored test), GL vs Vulkan:

| op | size | GL | Vulkan |
|---|---|---|---|
| fitness (200 tris) | 128² | 0.182 ms | 0.141 ms |
| fitness | 256² | 0.204 ms | 0.158 ms |
| fitness | 512² | 0.314 ms | 0.252 ms |
| brute-force polish (100 tris, 10 steps) | 128² | 1334 ms | 1269 ms |
| brute-force polish (100 tris, 10 steps) | 256² | **crashed** | 8195 ms |

**Findings:**
- **Vulkan is the right backend** and is kept: fitness scoring is slightly faster, and
  GL *crashed* on the 256² polish (less robust for heavy compute). Fitness scoring
  (~0.15–0.31 ms/score) is **not** a bottleneck on either backend.
- **The brute-force polish is compute-bound, not backend-bound.** It is ~identical on GL
  and Vulkan at 128² (1334 vs 1269 ms) — the earlier "hang" was the O(num_tris²)-per-pixel
  backward growing with triangle count (and ×`steps_n`×many polishes), *not* GL atomic
  slowness. ~127 ms/step at 128² and ~820 ms/step at 256² for only 100 triangles; at
  512² with 1000+ triangles this is minutes/step. Unusable for an every-step end-to-end
  loop.

**Decision (gate):** The **tiled kernel (Phase 2) is required** — Vulkan alone does not
make the polish viable. The budget it must beat: bring the per-gradient-step cost at 512²
with ~1000 triangles down from minutes to interactive (target: comparable to a few
fitness scores, i.e. low single-digit ms). Phase 3 (end-to-end loop) depends on Phase 2.
Next: a dedicated brainstorm→plan for the tiled forward/backward kernel, now that the
budget and the confirmed need are known.

## Quality probe result — DECISIVE NEGATIVE (2026-06-11, branch `experiment/gradient-quality-probe`)

Before building the full end-to-end loop, a lean probe asked the gating question:
*can ungated gradient descent push a plateaued baseline genome past the hard-ΔE2000
ceiling?* (`es::tests::gradient_primary_quality_probe`: baseline ES on a 128² downscale
of `goal.png` to plateau, then ungated `PolishState::polish_ungated` in chunks, tracking
best-ever by hard ΔE2000.)

**Result: gradient descent CATASTROPHICALLY DEGRADES the converged genome.** Hard ΔE2000
fitness fell 926116 → ~590000 (−36%) over the chunks and *never once* matched, let alone
beat, baseline (best-ever delta = 0). `/tmp/probe_baseline.png` ≡ `/tmp/probe_polished.png`
(best stayed at baseline).

**Why (now understood):** the **soft Lab-MSE proxy is misaligned with hard ΔE2000 for a
near-optimal genome.** The ES optimized the genome for the *hard* renderer; minimizing the
*soft* (τ-blurred) loss moves it *away* from that hard optimum. Milestone 1 (Path B) only
showed improvement because a grossly-misplaced single triangle had enormous headroom where
soft and hard agreed — but the silhouette ceiling is a *near-optimal local minimum*, exactly
where soft and hard diverge. This is also why the hybrid gate rejected every polish during
active evolution. Fundamental tension: sharp τ aligns soft≈hard but saturates the sigmoid
(no gradient); soft τ gives gradient but misaligns from hard. No sweet spot delivered for
*refinement* (probe annealed τ 0.05→0.02).

**Conclusion: the differentiable-rasterizer path does not achieve the quality goal.**
Gradient-primary does not break the silhouette ceiling — it breaks the genome. The full
end-to-end loop is NOT worth building on this proxy. The lean probe (a few hundred lines)
correctly gated this *before* the large rewrite. Banked infrastructure (Phases 1/2/2.5:
Vulkan backend, validated tiled+binned differentiable kernel, CPU oracle) remains correct
and reusable; only the *premise* that gradient-on-soft-Lab-MSE refines quality is refuted.
Any revival needs a fundamentally better-aligned differentiable objective (e.g. a
differentiable approximation of ΔE2000, or hard-renderer-in-the-loop gradient estimation),
not more kernel speed.

## Out of scope

- Replacing the ES's structural role entirely (placement stays discrete/ES).
- A differentiable ΔE2000 loss (keep the smooth Lab-MSE proxy).
- Multi-GPU / non-square images (square-only, as today).
