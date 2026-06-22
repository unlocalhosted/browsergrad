# CUDA Compatibility Spine

BrowserGrad compiler is a real-GPU compatibility compiler, not a CPU simulator.
CUDA-shaped source lowers into semantic IR, then each primitive gets one of four
lowering decisions:

- `native`: direct WGSL/WebGPU lowering runs on real GPU.
- `gpu-polyfill`: one or more WGSL/WebGPU passes implement CUDA semantics on real GPU.
- `cpu-reference`: CPU reference exists for correctness/traces, but GPU lowering is not available.
- `unsupported`: no honest implementation yet; emit diagnostic.

Every CUDA gap should map to a semantic family:

- `frontend`: C/CUDA syntax, macros, helpers, types.
- `memory`: global/shared/constant/local memory spaces.
- `atomic`: CUDA atomics and memory ordering.
- `texture`: texture/surface objects and reads/writes.
- `subgroup`: warp intrinsics and cooperative groups.
- `library`: cuRAND, cuFFT, and reusable device-library islands.
- `runtime`: streams, events, dynamic parallelism, launch orchestration.
- `feature`: browser/WebGPU capability gates such as `shader-f16` or subgroups.
- `safety`: invalid or unsafe kernels rejected before dispatch.

Public APIs:

- `getCudaFeatureRegistry()` returns known feature records.
- `describeCudaDiagnostic(diagnostic)` maps compiler diagnostics to semantic features.
- `createCudaLoweringPlan(diagnostics)` summarizes whether a kernel can run on GPU,
  requires GPU polyfill, has CPU reference coverage, or is unsupported.

Rule: do not add assignment-specific fixes. Add semantic primitives, reference
truth, WGSL lowering, browser tests, and corpus audit evidence.
