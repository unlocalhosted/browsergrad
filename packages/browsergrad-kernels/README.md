# @unlocalhosted/browsergrad-kernels

[![npm](https://img.shields.io/npm/v/@unlocalhosted/browsergrad-kernels.svg)](https://www.npmjs.com/package/@unlocalhosted/browsergrad-kernels)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

WGSL compute-shader catalog for browser ML. Each kernel ships with a pure-JS reference implementation that doubles as a conformance oracle and a CPU fallback. Also ships the production `WebGpuRealizerBridge` that [`browsergrad-jit`](../browsergrad-jit/) consumes for its WebGPU realizer tier.

Zero tensor-library dependency. Drop in if you just need fast WGSL primitives; layer in jit if you want the full PyTorch shape.

## What's shipped

### Kernels (with JS reference)

| Kernel | Variant | Status |
|---|---|---|
| `matmul` | Naive triple-loop, host-tensor input/output | ✅ |
| `matmulTiled` / `matmulTiledDirect` | 16×16 workgroup-tiled GEMM. **Production path.** | ✅ |
| `softmax` | Stable, along last axis | ✅ |
| `relu`, `gelu` | Elementwise activations | ✅ |
| `layernorm` | Along last axis, optional gamma/beta | ✅ |
| `attention` | Composed 3-kernel SDPA | ✅ |
| `flashAttentionDirect` | Flash Attention v2 forward, online softmax. **Known numerical issue on real Metal — tracked.** | ⚠️ |
| `fusedElementwiseDirect` | Runtime WGSL codegen for arbitrary elementwise chains | ✅ |

### Realizer-tier surface (consumed by jit)

- `createWebGpuRealizerBridge(device)` — production bridge satisfying the `WebGpuBridge` Protocol declared in jit. Opaque integer handles; bridge owns `GPUBuffer` lifetimes; pipeline cache via `runDirect`.
- `runDirect(device, desc, opts)` — `GPUBuffer`-in / `GPUBuffer`-out dispatch. The realizer-tier path; no host round-trip per op.
- `materializeFloat32(device, buffer, byteLength)` — read a `GPUBuffer` back to a `Float32Array` (the single readback at the realize boundary).
- `uploadFloat32(device, data)` — upload a typed array into a fresh `GPUBuffer`.

## Install

```bash
npm install @unlocalhosted/browsergrad-kernels
```

## Quick start

### One-shot kernel (host round-trip)

```ts
import { createDevice, kernels, tensor, matmulTiled } from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();
const A = tensor([2, 3], new Float32Array([1, 2, 3, 4, 5, 6]));
const B = tensor([3, 2], new Float32Array([7, 8, 9, 10, 11, 12]));

const C = await matmulTiled(device, A, B);   // tiled GEMM — production path
console.log(C.shape, C.data);                 // [2, 2], Float32Array(4)
```

### Pure-JS reference (no WebGPU required)

```ts
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";
const C = reference.matmul(A, B);  // identical surface; CPU only
```

### Realizer-tier (chained ops, GPU residency)

```ts
import {
  createDevice,
  matmulTiledDirect,
  materializeFloat32,
  uploadFloat32,
} from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();

const x = uploadFloat32(device, xData);
const w1 = uploadFloat32(device, w1Data);
const w2 = uploadFloat32(device, w2Data);

// (x @ w1) stays on the GPU; only the final readback crosses host.
const mid = matmulTiledDirect(device, x, w1, M, K, N);
const out = matmulTiledDirect(device, mid.buffer, w2, M, N, N);
const result = await materializeFloat32(device, out.buffer, out.byteLength);

mid.buffer.destroy();
out.buffer.destroy();
```

### Hand the bridge to browsergrad-jit

```ts
import { createDevice, createWebGpuRealizerBridge } from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();
const bridge = createWebGpuRealizerBridge(device);

// Expose the bridge to Pyodide
pyodide.registerJsModule("_bg_webgpu_bridge", bridge);
```

```python
# In Python (Pyodide)
import browsergrad_jit as bg
from js import _bg_webgpu_bridge
bg.register_webgpu_bridge(_bg_webgpu_bridge)

out = bg.realize_webgpu(model(x))   # all matmuls + fused chains run on the GPU
```

### Runtime WGSL codegen

```ts
import { generateFusedWgsl, fusedElementwiseDirect } from "@unlocalhosted/browsergrad-kernels";

// Produces a self-contained WGSL compute shader for the chain.
// Hash of the ops list = pipeline cache key.
const wgsl = generateFusedWgsl(
  [
    ["ADD", -1, -2],   // step0 = in0 + in1
    ["EXP", 0, 0],     // step1 = exp(step0)
    ["DIV", 1, -1],    // step2 = step1 / in0
  ],
  2,                    // num inputs
);
```

## Browser testing

```bash
pnpm test:browser
```

Launches Chromium via Playwright with WebGPU enabled. Runs against a real `GPUDevice`. On macOS the browser is headed (Metal driver only exposed when visible); on Linux CI set `BG_BROWSER_HEADLESS=1`.

7 scenarios: adapter info, naive vs tiled matmul, residency contract (3 uploads + 1 readback chained matmul), fused-elementwise codegen output matches NumPy semantics, FA-v2 (known-issue advisory), end-to-end `WebGpuRealizerBridge.matmul`.

Real-WebGPU CI is the only reliable way to catch shader-level bugs — NumPy mocks pass everything green even when the WGSL is wrong. The FA-v2 numerical issue tracked in the changelog was caught this way.

## API stability

| Surface | Stability |
|---|---|
| `kernels.*`, `matmul`, `matmulTiled`, `softmax`, `relu`, `gelu`, `layernorm`, `attention` | Semver-stable across `0.x` |
| `runDirect`, `matmulTiledDirect`, `fusedElementwiseDirect`, `flashAttentionDirect` | Semver-stable |
| `materializeFloat32`, `uploadFloat32` | Semver-stable |
| `createWebGpuRealizerBridge`, `WebGpuRealizerBridge` interface | Semver-stable; new methods added additively |
| `KernelError` | Semver-stable |
| WGSL source strings | **Internal.** Tuned freely. |
| Pipeline cache keys | **Internal.** Same WGSL → same key, but the encoding may change. |

## License

[MIT](LICENSE).
