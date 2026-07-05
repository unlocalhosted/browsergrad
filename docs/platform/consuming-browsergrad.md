# Consuming BrowserGrad Packages

This is the production consumption guide for npm users, lab platforms, and
agents that need to choose the right package without reading the whole repo.

## Install Matrix

Install only the packages you need:

```sh
npm install @unlocalhosted/browsergrad-runtime pyodide
npm install @unlocalhosted/browsergrad-kernels
npm install @unlocalhosted/browsergrad-compiler
npm install @unlocalhosted/browsergrad-jit
npm install @unlocalhosted/browsergrad-grad
npm install @unlocalhosted/browsergrad-primitives
```

For CUDA-lite/WebGPU compiler consumers, install `kernels` and `compiler`
together. The compiler depends on kernels for WGSL/WebGPU execution.

```sh
npm install @unlocalhosted/browsergrad-kernels@latest @unlocalhosted/browsergrad-compiler@latest
```

## Package Roles

| Need | Package | Primary imports |
| --- | --- | --- |
| Run Python/Pyodide labs in a Worker | `@unlocalhosted/browsergrad-runtime` | `createSession`, `parseManifest`, `assertCompatibleRuntime` |
| Use eager NumPy-backed autograd | `@unlocalhosted/browsergrad-grad` | `installGrad`, `createNodePyodideTarget` |
| Use lazy PyTorch-shaped IR, fusion, VJP, AMP, ONNX | `@unlocalhosted/browsergrad-jit` | `installJit`, `createNodePyodideTarget` |
| Run browser WGSL kernels and JS references | `@unlocalhosted/browsergrad-kernels` | `createDevice`, `kernels`, `reference`, `createWebGpuRealizerBridge` |
| Author generic WGSL programs | `@unlocalhosted/browsergrad-kernels` | `defineWgslKernelProgram`, `prepareWgslKernelProgramSequence`, `createWgslStorageBuffer` |
| Use browser-safe float16 helpers | `@unlocalhosted/browsergrad-kernels` | `createWgslFloat16Array`, `float32ToFloat16Bits`, `float16BitsToFloat32` |
| Use CUDA-shaped teaching references | `@unlocalhosted/browsergrad-kernels` | `runThreadGrid`, `simulateCuda1DGrid`, `referenceSaxpy`, `defineCuda1DProgram` |
| Compile CUDA-lite source to CPU reference/WGSL/WebGPU | `@unlocalhosted/browsergrad-compiler` | `compileCudaLiteKernel`, `compileCudaLiteKernelForWebGpu`, `runCompiledKernelWebGpu` |
| Use small browser-safe ML primitives | `@unlocalhosted/browsergrad-primitives` | `text`, `data`, `evaluation`, `simulation`, `scaling`, `rl` |

## Stable Import Surfaces

Use top-level imports for most consumers:

```ts
import {
  createDevice,
  createWgslFloat16Array,
  defineWgslKernelProgram,
  prepareWgslKernelProgramSequence,
  runThreadGrid,
  createKernelRubric,
} from "@unlocalhosted/browsergrad-kernels";
```

Subpath imports are also public for agents and bundlers that want smaller,
domain-specific entry points:

```ts
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";
import { defineWgslKernelProgram } from "@unlocalhosted/browsergrad-kernels/wgsl_program";
import { createWgslFloat16Array } from "@unlocalhosted/browsergrad-kernels/float16";
import { runThreadGrid } from "@unlocalhosted/browsergrad-kernels/cuda_concepts";
import { defineCuda1DProgram } from "@unlocalhosted/browsergrad-kernels/cuda_program";
import { createKernelRubric } from "@unlocalhosted/browsergrad-kernels/rubric";
```

CUDA-lite compiler import:

```ts
import { createDevice, detectKernelFeatures } from "@unlocalhosted/browsergrad-kernels";
import {
  compileCudaLiteKernelForWebGpu,
  compileCudaLiteOptionsFromKernelFeatures,
  createCudaWebGpuExecutionPlan,
  runCompiledKernelReference,
  runCompiledKernelWebGpu,
} from "@unlocalhosted/browsergrad-compiler";
```

## Agent Selection Rules

Agents should follow these rules before generating code:

- If the user asks for PyTorch-shaped Python in browser, start with `runtime`
  plus `jit`.
- If the user asks for eager teaching autograd, use `grad` instead of `jit`.
- If the user asks for direct WebGPU/WGSL kernels, use `kernels` only.
- If the user asks for CUDA-like source, use `compiler` plus `kernels`; do not
  hand-write WGSL unless the user explicitly asks for WGSL.
- If the user asks for rubrics, references, simulations, or generic ML helpers,
  check `kernels` and `primitives` before inventing local helpers.
- Unsupported PyTorch or CUDA APIs must fail loudly with explicit diagnostics.
  Do not silently emulate unsupported behavior with approximate code.

## Browser And Node Notes

- WebGPU execution requires a browser `GPUDevice`; use Playwright/Chromium for
  automated browser tests.
- Pure JS references, package metadata, and CUDA-lite CPU reference execution can
  run in Node.
- Pyodide packages need `pyodide` as a peer dependency and same-origin Pyodide
  assets in browser deployments.
- `shader-f16` and subgroup features are browser/device capabilities. Use
  `detectKernelFeatures(device)` and pass the resulting facts into compiler
  options with `compileCudaLiteOptionsFromKernelFeatures()`.

## Published Artifact Verification

Release confidence must verify packed tarballs, not just workspace source:

```sh
pnpm -r build
pnpm test:release-packages
```

`test:release-packages` runs `pnpm pack` for kernels and compiler, imports the
packed kernels `dist/` exports, checks public subpath exports, and verifies the
compiler tarball dependency on kernels is a concrete npm version, not
`workspace:*`.

