# CS149 Assignment 1 Handoff

Source: https://github.com/stanford-cs149/asst1

## BrowserGrad Slice

First browser-safe slice:

- Treat CS149 A1 as a CPU/SIMD systems benchmark, not a BrowserGrad core
  identity.
- Use `docs/internal/cs149-assignment1.profile.json` for profile metadata,
  allowed tests, JS rubric path, and capability gates.
- Use `@unlocalhosted/browsergrad-primitives` for:
  - `simulation.simulateVectorizedClampedExp()` to check clamped exponentiation output,
    tail masks, vector instruction traces, and active-lane utilization.
  - `simulation.simulateVectorizedArraySum()` to check vector-width array-sum reductions
    and horizontal reduction rounds.
  - `simulation.partitionStaticWork()` to check contiguous or cyclic static work
    decomposition for Mandelbrot/K-means-style thread labs.

## Upstream Signals

Portable browser-safe concepts:

- SIMD execution within a core and multicore decomposition are the core
  learning goals.
- Program 2 uses CS149's fake vector intrinsics rather than raw AVX2, making a
  deterministic JS simulator pedagogically honest.
- `clampedExpVector` must handle arbitrary `N` and `VECTOR_WIDTH`, including
  tail lanes.
- `arraySumVector` may assume `N % VECTOR_WIDTH == 0` and should use vector
  reductions.
- Program 1 asks for static thread decomposition without synchronization.

Native/external-heavy concepts:

- Real myth-machine performance graphs, AVX2 timing, ISPC compiler output, and
  C++ wall-clock speedup should remain external-runner checks.
- Browser rubrics should not pretend to reproduce Intel Hyper-Threading,
  frequency scaling, or native ISPC code generation.

## Platform Handoff

Crafting Attention should:

1. Load the profile and show readiness through
   `createAssignmentPreflightReport()`.
   For execution, call `runAssignmentJavascriptProfile()` with the imported JS
   rubric and `_bg_cpu_parallelism`; BrowserGrad has an e2e test proving this
   exact profile-driven route.
2. Treat `simd-simulator`, `pthreads-simulator`, `ispc-simulator`, and
   `performance-rubric` as simulated/browser-safe teaching capabilities.
3. Mount a JS rubric that calls the simulator helpers and reports structured
   assertion failures for:
   - wrong clamped exponentiation output,
   - eager/tail-unsafe vector loads,
   - poor or unexplained lane utilization,
   - wrong horizontal reduction,
   - unbalanced or hard-coded static work partitioning.
4. Keep native C++/ISPC timing behind `native-cpp-external` or
   `ispc-external` runner paths.

## Required Fixture Shape

- Small numeric arrays for `values`, `exponents`, expected clamped outputs, and
  expected sums.
- Thread decomposition fixtures with `items`, `workers`, optional `chunkSize`,
  and expected ranges.
- Optional snapshot fixtures for expected simulator traces.

## Known Gaps

- No browser-native ISPC compiler path yet.
- No native C++ runner bridge yet.
- No real performance timing rubric yet; current browser path checks
  deterministic behavior and explanatory simulator stats.
