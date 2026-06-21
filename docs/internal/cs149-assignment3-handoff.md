# CS149 Assignment 3 Handoff

Source: https://github.com/stanford-cs149/asst3

## BrowserGrad Slice

First browser-safe slice:

- Treat CS149 A3 as CUDA concept coverage with explicit native-runner escape
  hatches.
- Use `docs/internal/cs149-assignment3.profile.json` for JS rubric routing,
  allowed tests, timeouts, and capability gates.
- Use `@unlocalhosted/browsergrad-kernels` for:
  - `referenceSaxpy()` to check SAXPY correctness.
  - `referenceExclusiveScan()` to check scan fixtures.
  - `simulateCuda1DGrid()` to check 1D thread/block indexing, guard behavior,
    and out-of-bounds access before native CUDA execution exists.

## Upstream Signals

Portable browser-safe concepts:

- SAXPY is a one-dimensional elementwise CUDA kernel.
- Scan correctness can be tested with small deterministic arrays.
- Renderer ordering and memory-bounds checks can start as small trace/snapshot
  fixtures before real GPU rendering performance enters the path.

Native/external-heavy concepts:

- CUDA compilation, device timing, renderer throughput, and full image-output
  grading remain external or future WebGPU-specific work.
- Browser rubrics should not fake CUDA occupancy, SM scheduling, or native
  memory coalescing.

## Platform Handoff

Crafting Attention should:

1. Load `cs149-assignment3.profile.json` and route through
   `runAssignmentJavascriptProfile()`.
2. Register `_bg_cuda_concept_oracles` with the CUDA-concept helpers.
3. Start with `saxpy_correctness`, `exclusive_scan_correctness`, and
   `kernel_memory_bounds`.
4. Keep `performance_rubric_smoke` informational until WebGPU/native timing
   fixtures are calibrated.
5. Keep native CUDA as an explicit external runner, not hidden BrowserGrad
   behavior.

## Required Fixture Shape

- SAXPY fixtures: scalar `a`, arrays `x` and `y`, expected `a*x+y`.
- Scan fixtures: input array and expected exclusive prefix sums.
- Memory-bounds fixtures: launch shape, input/output lengths, expected
  violations.

## Known Gaps

- No native CUDA runner bridge yet.
- No renderer-specific image oracle yet.
- No calibrated browser performance rubric yet.
