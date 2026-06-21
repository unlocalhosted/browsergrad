# GPU Puzzles Handoff

Source: https://github.com/srush/gpu-puzzles

## BrowserGrad Slice

First browser-safe slice:

- Treat GPU Puzzles as CUDA-shaped kernel pedagogy, not as a CUDA runtime clone.
- Use `docs/internal/gpu-puzzles.profile.json` for JS rubric routing,
  capability gates, allowed tests, and fixture layout.
- Use `@unlocalhosted/browsergrad-kernels` for:
  - `simulateCuda1DGrid()` to run small 1D map/zip/guard puzzle callbacks and
    report thread/block traces plus out-of-bounds reads/writes.
  - `referenceSaxpy()` and `referenceExclusiveScan()` when GPU Puzzles overlap
    with CS149 A3 CUDA concept fixtures.
  - `createBrowsergradKernelRubric(ctx)` when comparing WGSL output tensors
    against references.
- Treat this as the first HipScript-inspired slice: CUDA-shaped kernel
  semantics are explicit in JS today, then can lower into WGSL/WebGPU once the
  simulator trace path is boring and trusted.
- Prefer `defineCuda1DProgram()` for new fixtures that should share one
  source between simulator checks and generated WGSL.

## Upstream Signals

Portable browser-safe concepts:

- Puzzle 1 map: one thread per vector position, add 10.
- Puzzle 2 zip: one thread per position, combine two arrays.
- Puzzle 3 guard: launch more threads than data and require explicit bounds
  checks.
- Early puzzles require simple indexing, arithmetic, loops, and guards; this is
  a good fit for JS/WGSL rubric fixtures.

Native/external-heavy concepts:

- Real Numba CUDA execution and Colab GPU setup remain external.
- Browser rubrics should preserve CUDA-shaped thinking while surfacing that
  native performance and occupancy are not being graded in-browser.

## Platform Handoff

Crafting Attention should:

1. Load `gpu-puzzles.profile.json` and route through
   `runAssignmentJavascriptProfile()`.
2. Register `_bg_cuda_concepts` with `simulateCuda1DGrid()` and kernel
   rubric helpers. The runner should reject the profile before executing the
   rubric if this oracle is missing.
3. Start with `puzzle_map`, `puzzle_zip`, and `puzzle_guard` fixtures.
4. Report guard failures using `violations` from `simulateCuda1DGrid()` instead
   of browser/runtime exceptions.
5. Keep advanced shared-memory/performance puzzles behind later WGSL or external
   CUDA runners.

## Required Fixture Shape

- Small arrays for map/zip/guard inputs and expected outputs.
- Thread/block config per puzzle.
- Optional expected memory trace snapshots for guard diagnostics.

## Known Gaps

- No native CUDA/Numba runner bridge yet.
- No shared-memory simulator yet.
- No visual kernel stepper yet; current slice returns traces suitable for one.
