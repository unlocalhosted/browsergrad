# @unlocalhosted/browsergrad-kernels

[![npm](https://img.shields.io/npm/v/@unlocalhosted/browsergrad-kernels.svg)](https://www.npmjs.com/package/@unlocalhosted/browsergrad-kernels)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

WGSL kernel catalog for browser-side machine learning. Every kernel ships with:

1. A WebGPU implementation (WGSL compute shader + dispatch logic)
2. A pure-JS reference (oracle for conformance + CPU fallback + lesson material)

Independent of any tensor framework. Take what you want; the rest tree-shakes away.

> **Status: v0.1.0.** Six kernels: matmul, softmax, relu, gelu, layernorm, attention. f32-only. Independent package — no dependency on the runtime.

## Install

```sh
npm install @unlocalhosted/browsergrad-kernels
```

No runtime dependencies. `@webgpu/types` is a devDependency for our build; consumers don't need it (modern TS pulls WebGPU types from `lib.dom.d.ts`).

## Usage

```ts
import { createDevice, kernels, tensor } from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();

const A = tensor([2, 3], new Float32Array([1, 2, 3, 4, 5, 6]));
const B = tensor([3, 2], new Float32Array([7, 8, 9, 10, 11, 12]));

const C = await kernels.matmul(device, A, B);
console.log(C.shape);  // [2, 2]
console.log(C.data);   // Float32Array(4) [ 58, 64, 139, 154 ]
```

### Without a GPU — the JS reference

Every kernel has a pure-JS counterpart at `@unlocalhosted/browsergrad-kernels/reference`:

```ts
import { reference, tensor } from "@unlocalhosted/browsergrad-kernels/reference";

const A = tensor([2, 3], new Float32Array([1, 2, 3, 4, 5, 6]));
const B = tensor([3, 2], new Float32Array([7, 8, 9, 10, 11, 12]));

const C = reference.matmul(A, B);  // same surface, runs on CPU
```

The reference is also imported by the WGSL implementations' conformance tests — if a WGSL kernel's output diverges from the JS reference by more than `1e-4`, CI fails.

## Kernels in v0.1.0

| Kernel | Shapes | Notes |
|---|---|---|
| `matmul(device, A, B)` | A: `[M, K]`, B: `[K, N]` → `[M, N]` | Naive triple-loop WGSL, workgroup 8×8. |
| `softmax(device, x)` | any rank; along last axis | Stable (subtracts max). One thread per row. |
| `relu(device, x)` | any shape | Elementwise. |
| `gelu(device, x)` | any shape | Tanh-approximation (GPT-2/BERT variant). |
| `layernorm(device, x, { gamma?, beta?, eps? })` | any rank; along last axis | `gamma`/`beta` shape `[D]`; default `eps=1e-5`. |
| `attention(device, Q, K, V)` | Q, K, V: `[S, D]` → `[S, D]` | Scaled dot-product. Single head, no batching, no mask. |

All kernels are f32. Output buffers are freshly allocated each call.

## Design notes

- **Naive WGSL, not optimized.** Each kernel does the simplest correct thing. A 30-line correct matmul beats a 200-line tiled one with an off-by-one. Tiled and fused variants will land as additive features.
- **Reference impls are lesson material.** `src/reference.ts` is deliberately legible — a learner should be able to read `referenceAttention` and understand how Q·Kᵀ, scaling, softmax, and the V multiply compose. The WGSL versions are next to them for the "now make it fast" lesson.
- **Pipeline caching.** `createDevice` returns a `KernelDevice` that caches `GPUComputePipeline`s by `(kernelName, param-signature)`. Calling the same kernel with the same shape category repeatedly re-uses pipelines.
- **Buffer lifecycle.** v0 allocates fresh GPU buffers per call. Hot loops (training) should use a pre-allocated path; that API lands in v0.2 as `kernel.runOnGpu(gpuBuffers, params)`.
- **No fused attention yet.** v0 attention is four kernels glued together (transpose, matmul, scale+softmax, matmul). A fused FlashAttention-style kernel is on the roadmap — same surface, different cache key.

## API

See [`src/types.ts`](./src/types.ts) and [`src/kernels/`](./src/kernels/). Stability contract:

- Adding optional fields → minor version bump
- Removing fields, adding required params, renaming exports → major version bump
- Anything not exported from `src/index.ts` is private

## Why not WebGL / TensorFlow.js / ONNX Runtime Web?

Those are excellent frameworks. This library is much smaller and exposes the WGSL itself. Use it when:
- You want to *read* the kernel source (lesson material)
- You want a thin layer that doesn't bring its own tensor abstraction
- You want WebGPU specifically (newer, generally faster than WebGL paths, no fallback drama)

If you want a full framework with autograd and a graph compiler, those tools are better.

## License

MIT
