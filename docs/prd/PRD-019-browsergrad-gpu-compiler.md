# PRD-019 — BrowserGrad GPU Compiler

## Problem

BrowserGrad can already run handwritten WGSL kernels and a small `Kernel1D`
program IR through real WebGPU. The missing layer is a learner-facing compiler:
students should be able to write CUDA/HIP-shaped kernels in the browser, inspect
what BrowserGrad understood, run a deterministic CPU reference, then dispatch
the same program on WebGPU.

HipScript proves a browser CUDA/HIP path is possible by chaining Clang,
chipStar, clspv, and Tint. BrowserGrad should not make that heavy toolchain the
default. The fast path should be BrowserGrad-owned, inspectable, and small:
CUDA-lite source -> Kernel IR -> WGSL -> WebGPU.

## Research Defaults

- HipScript is the compatibility north star, not the default architecture. Its
  LLVM-derived path can become an optional power backend after the small path is
  solid.
- WebGPU/WGSL are the native browser kernel target. Current browser primitives
  worth exploiting are workgroup memory, barriers, atomics, `shader-f16`,
  subgroups, and WebGPU compatibility mode.
- WASM threads, JSPI, and Memory64 are useful for future heavy compiler bundles,
  especially a browser LLVM worker. They are not required for the v0 compiler.
- WebNN is a future graph/inference island backend. It is not a custom kernel
  authoring target.

## Decision

Add `@unlocalhosted/browsergrad-compiler`, a generic compiler package that owns
parsing, analysis, Kernel IR, reference execution, WGSL emission, and WebGPU
handoff. Keep `@unlocalhosted/browsergrad-kernels` as the runtime substrate for
devices, bindings, and dispatch.

## Public API

- `parseCudaLite(source)` parses CUDA-lite source and returns an AST with source
  spans.
- `analyzeCudaLite(ast, options?)` validates the supported subset and reports
  feature requirements.
- `lowerCudaLiteToKernelIr(ast, options?)` lowers the selected kernel into a
  dimension-agnostic Kernel IR.
- `emitKernelIrWgsl(module, options?)` emits WGSL with feature-gated extensions.
- `compileCudaLiteKernel(source, options?)` returns the analyzed IR, WGSL,
  bindings, diagnostics, and required features.
- `runCompiledKernelReference(compiled, inputs, launch)` executes the IR in a
  lockstep CPU interpreter with block-local shared memory.
- `runCompiledKernelWebGpu(device, compiled, inputs, launch)` dispatches the
  emitted WGSL through `@unlocalhosted/browsergrad-kernels`.

## CUDA-Lite v0

Supported:

- `__global__ void kernel(...)`
- `float*`, `int*`, `uint*`, and `const` pointer params
- scalar params: `int`, `uint`, `float`, `half`
- `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` `.x/.y/.z`
- declarations, assignments, array reads/writes, `if`, and canonical `for`
- arithmetic, comparisons, boolean ops, `min`, `max`, `sqrtf`, `expf`, `logf`
- fixed-size `__shared__` arrays
- `__syncthreads()`
- `atomicAdd` for `i32/u32`

Explicit errors:

- broad C++ syntax, templates, classes, dynamic shared memory, device function
  definitions, pointer arithmetic outside array indexing, divergent barriers,
  writes through `const` pointers, unsupported atomics, and subgroup/f16 use
  when unavailable.

## Implementation Notes

- The compiler package is a real package seam. Parser/analyzer/IR/codegen would
  bloat either runtime or kernels if embedded there.
- Diagnostics must include source spans and stable codes so platform rubrics can
  render helpful learner feedback.
- Reference execution must be independent of WebGPU and Pyodide.
- WGSL emission must keep the GPU model visible: storage buffers, uniforms,
  workgroups, shared memory, and barriers should map directly to inspectable
  source.

## Acceptance

- SAXPY compiles from CUDA-lite source to WGSL and runs through CPU reference
  and real WebGPU.
- A tiled matmul fixture uses `__shared__` memory and `__syncthreads()`.
- `half` and subgroup intrinsics produce deterministic feature diagnostics.
- Generic WGSL runner supports typed storage buffers, uniform bytes, selected
  readbacks, 1D/2D/3D dispatch, and structured shader compile failures.
- Browser tests prove real WebGPU dispatch, not just mock execution.
