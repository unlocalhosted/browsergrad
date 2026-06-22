# Kernel Lab Foundation

BrowserGrad should grow a tiny browser-native kernel core first, then mature it
into more powerful GPU-programming features. The goal is progressive
compatibility: start with the clean learning model of buffers, kernels,
dispatch, synchronization, correctness oracles, and performance feedback, then
keep expanding toward CUDA/Triton-style experimentation wherever the browser can
support it honestly.

## Decision

Build a foundational `kernel-lab` layer around WebGPU/WGSL and BrowserGrad
rubrics. Treat projects like `gpu.cpp` and HipScript as design references, not
hard dependencies.

## Clean Parts To Steal

From `gpu.cpp`:

- Small nouns and verbs: context, tensor/buffer, kernel, bindings, dispatch,
  copy-to-host.
- Low boilerplate API where resource setup is explicit and dispatch is obvious.
- Native Dawn runner as an optional test/benchmark companion.
- Fast iteration and examples that expose the GPU model directly.

From HipScript:

- CUDA/HIP teaching concepts can map to WebGPU: grid, block, thread index,
  shared memory, barriers, kernel launch.
- A CUDA-like language can start with a small supported subset while keeping the
  door open to broader compatibility as compiler/runtime pieces mature.
- Browser compilation is possible, but a full LLVM toolchain is heavy and should
  stay optional or future-facing.

## Initial Core

The first stable core should include:

- `KernelContext`: owns a WebGPU device/queue and cache.
- `KernelBuffer`: typed storage buffer with explicit upload/download.
- `KernelProgram`: WGSL source plus workgroup metadata and bind layout.
- `dispatchKernel`: validates bindings, dispatches, awaits completion.
- `KernelOracle`: CPU/JS reference for correctness checks.
- `KernelRubric`: structured assertion helpers for output tolerance, dispatch
  errors, forbidden APIs, and timing envelopes.

First shipped BrowserGrad primitive: `@unlocalhosted/browsergrad-kernels`
exports `createKernelRubric()`, a CPU-only assertion collector for JS/WebGPU
rubrics. It checks tensor shape/value closeness with tolerances, records compact
failure details, and can forward directly into BrowserGrad JS rubric callbacks.
The same package now exports the BrowserGrad-owned thread-grid executor
`runThreadGrid()`, which gives GPU Puzzles/CS149-style labs thread/block
traces, guard diagnostics, and deterministic outputs before WGSL lowering.
`simulateCuda1DGrid()` remains a compatibility alias for assignments that teach
CUDA vocabulary directly.
The next shipped primitive is `defineKernel1DProgram()`: one small Kernel1D
program description can run through `runKernel1DProgramReference()` and
`emitKernel1DProgramWgsl()`, then dispatch through `runKernel1DProgramWebGpu()`
when a browser adapter exists. It now supports scalar params and `outputRead`
expressions for SAXPY-like kernels. `defineCuda1DProgram()` and friends remain
compatibility aliases. This keeps the HipScript direction alive by making
CUDA/HIP-like syntax a frontend over our core, without making browser LLVM the
first dependency.

PRD-019 adds the next layer: `@unlocalhosted/browsergrad-compiler`, a
BrowserGrad-owned CUDA-lite compiler that parses learner source, validates a
small supported subset, lowers into Kernel IR, emits WGSL, and runs both a
lockstep CPU reference and real WebGPU dispatch. This is the default path for
browser-native GPU programming. A HipScript-style LLVM backend remains a future
power backend, not the first dependency.
Use [CUDA-lite Compiler Architecture](./cuda-lite-compiler-architecture.md) for
low-level extension invariants and evidence gates.
The compiler package includes SAXPY, guarded-map, and tiled-matmul examples;
profiles should declare `cuda-lite-compiler` when they depend on this authoring
path rather than handwritten WGSL.
Platforms can derive that label with `browserGpuCapabilities()` after calling
`detectKernelFeatures()` so `shader-f16`, subgroup, WGSL, and CUDA-lite support
come from one capability environment.
For hot loops, the low-level WGSL runner exposes prepared sequences and resident
storage buffers, while the compiler exposes `prepareCompiledKernelWebGpu()` over
the same execution plans. Platform labs should use the prepared path when launch
shape and scalar params are stable, then materialize only requested readbacks.

This core should be independent from Pyodide. Python assignments may call it
through registered JS modules, but JS/WGSL labs should run without Python.

## Maturation Path

1. WGSL-first kernel labs with CPU oracles.
2. CUDA-shaped 1D program IR with simulator and WGSL lowering.
3. Native Dawn/gpu.cpp-style runner for CI and local benchmarking.
4. Resident buffers plus prepared dispatch sequences for hot-loop timing.
5. Kernel tracing artifacts: source, bindings, workgroups, timing, output
   previews.
6. CUDA-lite syntax for teaching simple kernels via PRD-019.
7. Worker-mesh collectives for distributed systems labs.
8. Pattern-specific kernels such as FlashAttention once the simple core is
   boring and stable.

## Compatibility Posture

The core should be ambitious about compatibility for learning and
experimentation. Learners should be able to tinker with CUDA-like syntax,
Triton-like ideas, handwritten WGSL, and native-runner workflows over time.

The first stable guarantee is smaller: WGSL kernels, explicit buffers,
deterministic oracles, and transparent dispatch. Features like broad CUDA
surface area, Triton-style kernels, GPU libraries, warp intrinsics, and richer
compiler support are expansion targets, not abandoned goals.

Two constraints remain load-bearing:

- Do not hide WebGPU so thoroughly that learners cannot see the real GPU model.
- Do not make Pyodide required for kernel labs.

## Why This Matters

For systems assignments like CS336 assignment 2, a browser-native kernel core
lets BrowserGrad teach the real concepts without pretending a browser is a
Linux CUDA box. We can capability-gate native-only tests, then replace them with
labs that show the same systems ideas through WebGPU, deterministic simulators,
and transparent rubrics.
