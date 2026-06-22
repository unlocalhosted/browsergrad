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
| `referenceFlashAttention` / `referenceFlashAttentionBackward` | Pure-JS FlashAttention oracle with output, log-sum-exp, and Q/K/V gradients | ✅ |
| `defineKernel1DProgram` / `runKernel1DProgramReference` / `emitKernel1DProgramWgsl` / `runKernel1DProgramWebGpu` | BrowserGrad-owned 1D kernel IR with reference executor, WGSL lowering, and browser WebGPU dispatch | ✅ |
| `runThreadGrid`, `referenceSaxpy`, `referenceExclusiveScan`, `referenceFindRepeats`, `referenceOrderedCircleRender` | Thread-grid teaching references for GPU Puzzles and CS149 A3 browser rubrics | ✅ |
| `defineCuda1DProgram` / `simulateCuda1DProgram` / `emitCuda1DProgramWgsl` / `runCuda1DProgramWebGpu` / `simulateCuda1DGrid` | CUDA-shaped compatibility aliases for labs and rubrics that teach CUDA vocabulary | ✅ |
| `flashAttentionDirect` | Flash Attention v2 forward, online softmax. **Known numerical issue on real Metal — tracked.** | ⚠️ |
| `fusedElementwiseDirect` | Runtime WGSL codegen for arbitrary elementwise chains | ✅ |

### Realizer-tier surface (consumed by jit)

- `createWebGpuRealizerBridge(device)` — production bridge satisfying the `WebGpuBridge` Protocol declared in jit. Opaque integer handles; bridge owns `GPUBuffer` lifetimes; pipeline cache via `runDirect`.
- `runDirect(device, desc, opts)` — `GPUBuffer`-in / `GPUBuffer`-out dispatch. The realizer-tier path; no host round-trip per op.
- `materializeFloat32(device, buffer, byteLength)` — read a `GPUBuffer` back to a `Float32Array` (the single readback at the realize boundary).
- `uploadFloat32(device, data)` — upload a typed array into a fresh `GPUBuffer`.
- `createWgslStorageBuffer()` / `writeWgslStorageBuffer()` /
  `readWgslStorageBuffer()` — caller-owned resident storage buffers for generic
  WGSL programs. Use `residentBuffers` with `runWgslKernelProgramSequence()` to
  avoid per-call upload and skip readback with `readback: []`.
- `prepareWgslKernelProgramSequence()` — prebuilds pipelines and bind groups
  once, then reruns the same WGSL sequence over resident buffers for hot loops.

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

For CS336 A2-style FlashAttention rubrics, import
`referenceFlashAttention()` and `referenceFlashAttentionBackward()` from the
top-level package. The forward oracle returns `{ output, logSumExp }`, matching
the upstream test's saved-LSE contract; the backward oracle recomputes softmax
probabilities and returns Q/K/V gradients without requiring PyTorch autograd,
Triton, or CUDA.

For GPU Puzzles and CS149 A3-style kernel-concept rubrics, import
`runThreadGrid()`, `referenceSaxpy()`, and `referenceExclusiveScan()`.
`runThreadGrid()` runs a browser-safe 1D thread/block callback, records
per-thread reads/writes, and reports out-of-bounds access instead of hiding
missing guards. It is a correctness and pedagogy oracle, not a native CUDA
performance runner. `simulateCuda1DGrid()` remains as a compatibility alias for
rubrics that intentionally use CUDA vocabulary.

For a durable author-once path, define a tiny BrowserGrad Kernel1D program and
run it through both adapters:

```ts
import {
  createDevice,
  defineKernel1DProgram,
  emitKernel1DProgramWgsl,
  runKernel1DProgramReference,
  runKernel1DProgramWebGpu,
} from "@unlocalhosted/browsergrad-kernels";

const program = defineKernel1DProgram({
  name: "saxpy_guarded",
  inputLength: 4,
  outputLength: 4,
  parameters: { a: 2 },
  launch: { blocks: 1, threadsPerBlock: 8 },
  body: [{
    op: "if",
    condition: { op: "lt", left: { op: "threadId" }, right: { op: "outputLength" } },
    body: [{
      op: "write",
      index: { op: "threadId" },
      value: {
        op: "add",
        left: {
          op: "mul",
          left: { op: "param", name: "a" },
          right: { op: "read", index: { op: "threadId" } },
        },
        right: { op: "outputRead", index: { op: "threadId" } },
      },
    }],
  }],
});

const simulated = runKernel1DProgramReference(program, {
  initialInput: [1, 2, 3, 4],
  initialOutput: [10, 20, 30, 40],
});
const wgsl = emitKernel1DProgramWgsl(program);
const device = await createDevice();
const gpu = await runKernel1DProgramWebGpu(device, program, {
  initialInput: [1, 2, 3, 4],
  initialOutput: [10, 20, 30, 40],
});
```

This is the first HipScript-inspired kernel-authoring spine: BrowserGrad owns
the small IR and reference executor, then CUDA/HIP-like syntax can grow as a
frontend. The shipped path already has explicit grid/thread semantics, scalar
params, input/output buffer reads, deterministic traces, WGSL source generation,
and real browser WebGPU dispatch without shipping a browser LLVM toolchain.

For hot WGSL paths, keep storage buffers resident:

```ts
import {
  createDevice,
  createWgslStorageBuffer,
  defineWgslKernelProgram,
  readWgslStorageBuffer,
  runWgslKernelProgram,
  writeWgslStorageBuffer,
} from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();
const program = defineWgslKernelProgram({
  name: "inc",
  workgroupSize: [4, 1, 1],
  bindings: [{ kind: "storage", name: "x", valueType: "f32" }],
  wgsl: `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&x)) { x[gid.x] = x[gid.x] + 1.0; }
}`,
});
const x = createWgslStorageBuffer(device, {
  valueType: "f32",
  data: new Float32Array([1, 2, 3, 4]),
});
writeWgslStorageBuffer(device, x, new Float32Array([5, 6]), Float32Array.BYTES_PER_ELEMENT);

await runWgslKernelProgram(
  device,
  program,
  { buffers: {}, residentBuffers: { x }, readback: [] },
  { dispatchCount: [4, 1, 1] },
);

const out = await readWgslStorageBuffer(device, x);
```

For repeated dispatches, prepare the sequence once:

```ts
const prepared = await prepareWgslKernelProgramSequence(
  device,
  [{ program, launch: { dispatchCount: [4, 1, 1] } }],
  { buffers: {}, residentBuffers: { x }, readback: [] },
);
await prepared.run();
await prepared.run({ readback: [] });
await prepared.run({ readback: [], awaitCompletion: true });
await prepared.run({ uniforms: { params: new Float32Array([3]) } });
prepared.destroy();
```

Prepared uniform updates rewrite existing uniform buffers and reuse bind groups.
Use `stepUniforms` only when sequence steps need different values for the same
uniform binding name. Use `awaitCompletion: true` for no-readback timing gates
or platform watchdogs that need command completion, not only command submission.

Use `storageMetadata` when one physical storage buffer is viewed through
multiple WGSL value types, or when state needs readback even though no step binds
it:

```ts
await runWgslKernelProgramSequence(
  device,
  [{ program, launch, storageAliases: { floats: "raw" } }],
  {
    buffers: { raw: new Uint32Array(1024), state: new Uint32Array([0]) },
    storageMetadata: {
      raw: { valueType: "u32", compatibleValueTypes: ["f32"] },
      state: "u32",
    },
    readback: ["raw", "state"],
  },
);
```

### Kernel rubric assertions

```ts
import {
  createBrowsergradKernelRubric,
  kernels,
  reference,
} from "@unlocalhosted/browsergrad-kernels";

const rubric = createBrowsergradKernelRubric(ctx);

const actual = await kernels.matmul(device, A, B);
const expected = reference.matmul(A, B);
rubric.assertCloseTensor("matmul_tiny", actual, expected, { atol: 1e-4 });
```

`createKernelRubric()` is CPU-only and works without WebGPU. It records
pass/fail assertions, checks tensor shapes, compares values with absolute and
relative tolerance, and emits compact previews plus first failing index/max
error for learner-facing JS rubrics. Non-finite actual or expected values fail
the comparison instead of slipping through tolerance math.
`kernelRubricFailureToAssertionDetails()` formats structured rubric details into
`expected` / `actual` strings for BrowserGrad-style assertion callbacks.
`createBrowsergradKernelRubric(ctx)` is the convenience adapter for
`runAssignmentJavascriptRubric()` contexts and any compatible assertion target.

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
Use `pnpm test:browser:open` when you want the browser window to stay open for
inspection; quit with `q`.

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
