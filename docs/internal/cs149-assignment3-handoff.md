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
  - `referenceFindRepeats()` to check repeated-adjacent output fixtures from
    the scan/find-repeats part of the assignment.
  - `referenceOrderedCircleRender()` to check renderer ordering fixtures with
    deterministic normalized circle geometry and ordered alpha blending.
  - `simulateCuda1DGrid()` to check 1D thread/block indexing, guard behavior,
    and out-of-bounds access before native CUDA execution exists.
- Proven BrowserGrad route:
  - `packages/browsergrad-runtime/tests/assignment-javascript-profile-e2e.test.ts`
    loads this profile, registers `_bg_cuda_concepts`, and passes
    `saxpy_correctness`, `exclusive_scan_correctness`, and
    `renderer_ordering_correctness`, and `kernel_memory_bounds` through the
    generic kernel concept reference.

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
2. Register `_bg_cuda_concepts` with the CUDA-concept helpers.
3. Start with `saxpy_correctness`, `exclusive_scan_correctness`, and
   `renderer_ordering_correctness`, and `kernel_memory_bounds`.
4. Add `find_repeats` fixtures through `referenceFindRepeats()` when authoring
   scan labs.
5. Keep `performance_rubric_smoke` informational until WebGPU/native timing
   fixtures are calibrated.
6. Keep native CUDA as an explicit external runner, not hidden BrowserGrad
   behavior.

## Required Fixture Shape

- SAXPY fixtures: scalar `a`, arrays `x` and `y`, expected `a*x+y`.
- Scan fixtures: input array and expected exclusive prefix sums.
- Find-repeats fixtures: sorted/input array and expected indexes `i` where
  `input[i] === input[i + 1]`.
- Renderer-ordering fixtures: image size, background RGB, ordered circles with
  normalized center/radius/color/alpha, and expected flat RGB pixels.
- Memory-bounds fixtures: launch shape, input/output lengths, expected
  violations.

## Known Gaps

- No native CUDA runner bridge yet.
- No renderer-specific image oracle yet.
- No calibrated browser performance rubric yet.
