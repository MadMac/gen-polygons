# Prefix-Sum Tile Binning (Phase 2.5) — Design

**Date:** 2026-06-10
**Binary:** `polygenvo`
**Status:** Approved for planning
**Parent:** `2026-06-10-tiled-gradient-kernel-design.md` (Phase 2 results: tiled kernel
is ~100×+ over brute force but ~1.1 s/step at 512²/1000-tris — not interactive).

## Problem

The tiled differentiable kernel still loops **all** triangles per pixel with a
clip-bbox reject (`tri_overlaps_aabb`), so per-pixel cost is O(num_tris) and the
accuracy-required `8τ` margin limits rejection. Result: ~1.1 s/step at 512²/1000-tris,
~10× worse at the 10k-triangle budget — not interactive.

## Goal

Make per-pixel cost O(triangles-actually-in-tile) by building real per-tile triangle
lists once per gradient step, so each pixel iterates only its tile's triangles. The
gradient math is unchanged, so the CPU-oracle equality tests remain the correctness bar.

## Decisions locked during brainstorming

- **Approach:** count → exclusive prefix-sum → atomic-fill → per-tile sort (no capacity
  cap, no full radix sort). The per-tile sort restores draw order (OVER is order-dependent).
- **Re-bin every gradient step** (positions move each step, so lists must be rebuilt).
- **Single-workgroup scan** (each thread handles a serial chunk + shared-memory scan of
  partials); fine for 512²'s 1024 tiles and scales to larger via per-thread chunking.
  Multi-block scan is a future lever only for very large images.
- **Same bbox + 8τ margin** as the listless tiled kernel → identical contributing set →
  the existing GPU==CPU oracle tests are the integration guard.

## Architecture

16×16-pixel tiles; `num_tiles = ceil(W/16) * ceil(H/16)`. New `binning.wgsl` with four
compute entries, plus updates to `softraster_tiled.wgsl` and `gradient.rs`.

### Buffers (owned by `PolishState`, sized once in `new`)

- `tile_counts`: `array<atomic<u32>>`, len `num_tiles` (also reused as the fill cursor,
  reset between count and fill — or a separate `fill_cursor` of the same length).
- `tile_offsets`: `array<u32>`, len `num_tiles` (exclusive scan of counts) plus a
  one-element `total` (or `num_tiles+1` layout).
- `tile_list`: `array<u32>` of triangle indices, pre-sized to a generous capacity
  `LIST_CAP` (CPU-set; worst case num_tris×num_tiles ≈ 40 MB at the 10k/512² ceiling —
  fine to preallocate). An `atomic<u32>` overflow flag / `total` is checked host-side.
- A small `BinParams` uniform: `num_tris`, `tiles_x`, `tiles_y`, `tau` (+ width/height as
  needed to map triangle bbox → tile range; the margin is `8τ`).

### Binning passes (per gradient step, before forward)

1. **clear_counts** — zero `tile_counts` (and `fill_cursor`); via `clear_buffer` or a
   trivial kernel.
2. **count** — `@workgroup_size(64)`, one thread per triangle: clip bbox, expand by `8τ`,
   clamp to `[0,tiles_x) × [0,tiles_y)`, `atomicAdd(1)` to each covered tile's count.
3. **scan** — single workgroup: exclusive prefix-sum over `tile_counts` into
   `tile_offsets`; write `total` (sum) to a known slot. Per-thread serial chunks +
   shared-memory scan of partials so it handles `num_tiles > workgroup_size`.
4. **fill** — one thread per triangle: for each covered tile, slot =
   `tile_offsets[tile] + atomicAdd(fill_cursor[tile], 1)`; if `slot < LIST_CAP` write the
   triangle index, else set the overflow flag. (Unsorted.)
5. **sort_tiles** — one workgroup per tile: insertion/bitonic-sort `tile_list[off..off+cnt)`
   ascending by triangle index → restores draw order.

### Forward / backward change (`softraster_tiled.wgsl`)

Add `tile_offsets` + `tile_list` bindings. Replace the `for t in 0..num_tris { if
!tri_overlaps_aabb ... }` loop with: compute the pixel's tile id, read
`off = tile_offsets[tile]`, `cnt` (= next offset − off, or a stored count), and iterate
`for i in off..off+cnt { let t = tile_list[i]; ... }`. The per-triangle composite /
reverse-transmittance gradient body is otherwise **unchanged**. (The forward's `T_final`
and the backward's reconstruction use the same tile-list triangle set, so they stay
consistent.)

### Integration (`gradient.rs`)

`PolishState` gains the binning pipelines + buffers. The per-step polish loop becomes:
clear_counts → count → scan → fill → sort_tiles → tiled forward → clear grad → tiled
backward → adam. The `#[cfg(test)]` `gpu_grad_tiled` / `gpu_forward_tiled_lab` helpers
also run the binning passes first (so the oracle tests exercise the binned path).

## Testing

- **Binning unit test:** bin a known small scene; read back `tile_offsets` + `tile_list`;
  assert each tile's slice equals exactly the set of triangles whose bbox+8τ overlaps that
  tile, in ascending index order (CPU-replicated expectation). Include an exclusive-scan
  correctness assertion (offsets[i] == Σ counts[0..i]).
- **Oracle equality (primary guard):** the existing `gpu_grad_tiled` /
  `gpu_forward_tiled_lab` tests, re-pointed through the binned path, still match
  `softras_ref` (forward Lab < 1e-2; backward grad rel < 2e-2), incl. tile-boundary and
  empty-tile scenes. Binning changes HOW the per-tile set is gathered, not the math.
- **Benchmark:** extend `bench_backend` with binned polish at 512² with 1000 AND 10000
  triangles; record ms/step and the speedup vs the listless ~1.1 s/step.
- **Regression:** full suite green (`--test-threads=1` for the Vulkan parallel-test
  SIGSEGV); clippy clean.

## Acceptance

Binned kernel matches the oracle on all cases; benchmark shows an order-of-magnitude
speedup over the listless tiled kernel at 512² (target: tens-of-ms/step), and is tractable
(not multi-second) at the 10k-triangle budget — fast enough that real 512² end-to-end runs
(Phase 3) are feasible.

## Risks & mitigations

- **Scan off-by-one** (exclusive vs inclusive) → the binning unit test asserts
  `offsets[i] == prefix sum`.
- **`tile_list` overflow** → generous `LIST_CAP` + an overflow flag checked host-side
  (panic/log if exceeded); document the worst case.
- **Draw-order restoration** → the per-tile sort; the binning test checks ascending order.
- **Per-step binning overhead** → count (O(num_tris·tiles_per_tri)) + scan (O(num_tiles)) +
  fill + sort (small per-tile lists) — all cheap vs the per-pixel savings; the benchmark
  confirms net win.
- **`count` vs `fill` must use the identical bbox→tile mapping** (same margin, same clamp)
  or slots mismatch → factor the mapping into one shared WGSL helper used by both.

## Phase 2.5 results (2026-06-10, branch `feat/tile-binning`)

Built: `binning.wgsl` (count → exclusive scan → fill → per-tile sort), GPU==CPU-verified
per-tile lists; forward/backward iterate `tile_list[off..off+cnt]`; `PolishState` re-bins
each step. All oracle tests pass **unchanged** through the binned path (forward Lab
2.2e-5, backward rel 1.3e-5), polish improves hard ΔE2000 (484461→517938), 40 tests green.

**Benchmark (Vulkan, AMD RX 7800 XT), binned polish, 512², sharp τ=0.03:**

| genome | ms/step |
|---|---|
| 1000 tris, **large** (`init_genome`, radius ~0.3) | 1306 |
| 1000 tris, **small** (shrunk ×0.15 — realistic refined genome) | **790** |
| (Phase-2 listless tiled, 1000 large, for reference) | 1143 |

**Findings:**
- **Binning works and helps — but modestly (~1.65× on realistic small triangles), not
  the hoped order-of-magnitude.** With `init_genome`'s large triangles it gives ~nothing
  (1306 vs 1143) because each triangle covers ~290 of 1024 tiles → ~283 tris/tile.
- **The 8τ margin is the fundamental limiter.** At τ=0.03 the margin is 0.24 clip, so even
  a *small* triangle's tile footprint is ~16 tiles → ~16 tris/tile at 1000 tris. The
  per-pixel backward (heavy gradient block) over ~16 triangles, ×262k pixels, ×per-step
  binning + submit overhead, still costs ~0.8 s/step.
- **The interactive acceptance bar (tens-of-ms) is NOT met.** 512² differentiable polish
  remains ~hundreds-of-ms to ~1 s per step even with binning + small triangles.

**Decision (gate): the differentiable-rasterizer path is not reaching interactive 512²
on this approach.** Binning is a real, banked improvement but not the finish line.
Remaining levers, none guaranteed and each its own effort: (a) shrink the margin (e.g.
5τ — risks accuracy vs the oracle), (b) batch all N steps into one submit (cut per-step
overhead), (c) lighten the per-triangle backward, (d) polish a *subset* of triangles, or
(e) run gradient at coarse resolution. **Recommendation: pause the speed chase and re-probe
quality** — run a Phase-3 end-to-end probe at a tolerable scale (small image / capped tris)
to learn whether gradient-primary actually beats baseline *at all* before investing more in
speed. If the quality win isn't there, further kernel optimization is wasted.

## Out of scope

- Multi-block GPU scan for very large images (future lever; single-workgroup scan now).
- Full radix sort-based binning (not needed; per-tile sort of small lists suffices).
- The Phase 3 end-to-end run loop (separate; this unblocks it).
