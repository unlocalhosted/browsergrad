# PRD-019 — BrowserGrad GPU Compiler

## Problem Statement

BrowserGrad can run handwritten WGSL kernels and a small `Kernel1D` IR through
real WebGPU, but learners still cannot write CUDA/HIP-shaped kernels in the
browser and get a native-feeling compile/run/debug loop. Without that compiler
layer, GPU Puzzles, CS149, CS336 systems, and future ML-kernel labs either use
handwritten WGSL or stay simulator-only.

The goal is not to clone a Linux CUDA box. The goal is browser-native GPU
programming that is honest, inspectable, and ambitious: CUDA-lite source lowers
to BrowserGrad Kernel IR, emits WGSL, runs a deterministic CPU reference, then
dispatches on WebGPU.

## Solution

Add `@unlocalhosted/browsergrad-compiler`, a generic compiler package that owns
parsing, analysis, Kernel IR, reference execution, WGSL emission, and WebGPU
handoff. Keep `@unlocalhosted/browsergrad-kernels` as the runtime substrate for
devices, typed storage/uniform bindings, feature detection, and dispatch.

Default path: BrowserGrad-owned CUDA-lite -> Kernel IR -> WGSL -> WebGPU.
HipScript-style LLVM/chipStar/clspv/Tint compatibility remains a future power
backend, not the first dependency.

## User Stories

1. As a learner, I want to write SAXPY in CUDA-lite syntax, so that I can run one kernel against both a CPU reference and real WebGPU in the browser.
2. As a systems-lab author, I want to use shared memory and `__syncthreads()` in a tiled matmul lab, so that I can teach GPU locality without shipping a native CUDA toolchain.
3. As a platform author, I want compiler diagnostics with source spans, so that unsupported features, divergent barriers, missing feature gates, and unsafe pointer writes become clear learner feedback.
4. As a future compiler agent, I want a generic Kernel IR, so that I can extend toward 2D/3D kernels, atomics, f16, subgroups, and optional LLVM compatibility without renaming the public surface around one assignment.

## Research Dossier

- Repo exploration: `@unlocalhosted/browsergrad-kernels` already ships
  `runThreadGrid()`, `defineKernel1DProgram()`, `emitKernel1DProgramWgsl()`,
  `runKernel1DProgramWebGpu()`, real WebGPU tests, and a device/pipeline
  substrate. PRD-019 builds on that seam.
- Repo exploration: `docs/platform/kernel-lab-foundation.md` already chooses a
  browser-native kernel core around WebGPU/WGSL and treats HipScript/gpu.cpp as
  design references rather than mandatory dependencies.
- HipScript: https://lights0123.com/blog/2025/01/07/hip-script/ proves CUDA/HIP
  can run in browser by chaining chipStar, clspv, and Tint, but that path is
  heavyweight for the default educational loop.
- WGSL/WebGPU: https://www.w3.org/TR/WGSL/ is the browser-native shader target.
  It exposes workgroup memory, barriers, atomics, f16, and subgroup builtins in
  the language/spec surface.
- Chrome WebGPU subgroups: https://developer.chrome.com/blog/new-in-webgpu-134
  documents subgroup availability and explicit WGSL enablement.
- Chrome WebGPU compatibility mode:
  https://developer.chrome.com/blog/new-in-webgpu-146 documents the broader
  compatibility path and its restricted feature posture.
- WASM threads: https://emscripten.org/docs/porting/pthreads.html documents the
  COOP/COEP and `SharedArrayBuffer` requirement for threaded compiler bundles.
- JSPI: https://v8.dev/blog/jspi documents the future-friendly async bridge for
  large WASM applications.
- WebNN: https://www.w3.org/TR/webnn/ is valuable for graph/inference islands,
  not for custom kernel authoring.

## Grill Decisions

1. Question: Should the default path be LLVM/HipScript-style or BrowserGrad IR?
   Recommended answer: BrowserGrad IR first.
   Decision: Ship a small CUDA-lite compiler first; keep LLVM as optional
   future backend.
2. Question: Should v0 syntax be CUDA-lite, Triton-like, or WGSL macros?
   Recommended answer: CUDA-lite.
   Decision: CUDA-lite best fits GPU Puzzles, CS149, CS336 systems, and
   HipScript compatibility pressure.
3. Question: Should Pyodide be required for compiler labs?
   Recommended answer: No.
   Decision: Compiler/reference/WebGPU paths are pure TS/JS. Python rubrics may
   call them through platform bridges later.
4. Question: Should missing f16/subgroups silently fall back?
   Recommended answer: No.
   Decision: Emit deterministic feature diagnostics so labs teach the real GPU
   capability boundary.

## Novelty Reach

- Novel idea selected: lockstep CPU execution from the same Kernel IR as WGSL
  emission. This gives correctness, traces, and barrier teaching before real
  GPU dispatch.
- Novel idea selected: feature-specialized compile diagnostics for browser
  primitives (`shader-f16`, subgroups, compatibility mode) instead of pretending
  every browser is CUDA.
- Novel idea selected: keep CUDA/HIP-like syntax as a frontend over BrowserGrad
  IR. This lets future Triton-like or LLVM backends reuse the same reference,
  diagnostics, and platform capability vocabulary.

## Implementation Decisions

- Add public compiler APIs:
  `parseCudaLite`, `analyzeCudaLite`, `lowerCudaLiteToKernelIr`,
  `emitKernelIrWgsl`, `compileCudaLiteKernel`,
  `runCompiledKernelReference`, and `runCompiledKernelWebGpu`.
- Add generic kernel APIs in `browsergrad-kernels`:
  `detectKernelFeatures`, `defineWgslKernelProgram`, and
  `runWgslKernelProgram`.
- Support CUDA-lite v0: `__global__ void`, pointer/scalar params, builtins
  `threadIdx/blockIdx/blockDim/gridDim`, declarations, assignment, array
  indexing, `if`, canonical `for`, fixed `__shared__` arrays,
  `__syncthreads()`, scalar `__device__` helpers, selected warp/cooperative
  group primitives, dynamic shared memory with launch metadata, simple runtime
  calls in CPU reference, `min/max/sqrtf/expf/logf`, and common atomics.
- Emit explicit diagnostics for broad C++, templates, classes, unsupported
  runtime orchestration, divergent barriers, const-pointer writes, unsupported
  atomics, and unavailable f16/subgroup features.
- Add examples: SAXPY, guarded map, and shared-memory tiled matmul.
- Add platform capability vocabulary: `cuda-lite-compiler`.

## Testing Decisions

- Unit tests cover parser/analyzer diagnostics, SAXPY compilation, shared-memory
  tiled matmul, WGSL emission, feature gates, and the CPU reference interpreter.
- Browser tests compile SAXPY and tiled matmul, dispatch them on real WebGPU,
  and compare against the CPU reference.
- Kernel package tests cover generic WGSL binding validation, feature detection,
  and real WebGPU typed storage/uniform dispatch.
- Required gates:
  `pnpm --filter @unlocalhosted/browsergrad-kernels test`,
  `pnpm --filter @unlocalhosted/browsergrad-kernels test:browser`,
  `pnpm --filter @unlocalhosted/browsergrad-compiler test`,
  `pnpm --filter @unlocalhosted/browsergrad-compiler test:browser`,
  plus build/typecheck for both packages.

## Expansion Tracks

- CUDA/HIP/C++ compatibility grows by semantic families, not by course-specific
  patches: frontend, memory, atomic, texture, subgroup, library, runtime,
  feature, and safety.
- Browser LLVM/chipStar/clspv/Tint remains a power backend candidate once the
  BrowserGrad IR path owns diagnostics, traces, and WebGPU dispatch truth.
- Triton-compatible syntax can become another frontend over the same Kernel IR.
- Broad runtime APIs advance through explicit orchestration plans: device
  launches, sync, peer copies, streams/events, then host-side multi-dispatch.
- Pyodide assignment wiring and CraftingAttention platform e2e sit above this
  compiler package; they should consume generic compiler/runtime capabilities.

## Further Notes

The v0 compiler is intentionally small. It should feel more native than a
simulator because it reaches real WebGPU, but more teachable than a giant
browser LLVM bundle because every stage is inspectable: source, AST, Kernel IR,
WGSL, CPU trace, and GPU output.
