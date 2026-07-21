# @unlocalhosted/browsergrad-kernels

[![npm](https://img.shields.io/npm/v/@unlocalhosted/browsergrad-kernels.svg)](https://www.npmjs.com/package/@unlocalhosted/browsergrad-kernels)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

WGSL compute-shader catalog for browser ML and systems workloads. Each kernel
ships with a pure-JS reference implementation that acts as a conformance
oracle; reference execution is distinct from device execution. The package also
ships the production `WebGpuRealizerBridge` that
[`browsergrad-jit`](../browsergrad-jit/) consumes for its WebGPU realizer tier.

No tensor-framework dependency. The package depends only on BrowserGrad's
dependency-free semantic-core protocols for verified layout/kernel artifacts.
Drop it in for WGSL primitives; layer in JIT for the PyTorch-shaped surface.

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
| `prepareSemanticViewCopyWgsl` / `runSemanticViewCopyWebGpu` | Verified `view-copy@1.0` lowering over canonical layout/index artifacts, with bit-exact u32 storage and structured guarded padding | ✅ nine-case strict CPU/WebGPU parity on Apple Metal 3; release evidence remains commit-scoped |
| `prepareSemanticGemmWgsl` / `runSemanticGemmWebGpu` | Verified logical GEMM plus independent schedule lowering with cooperative workgroup staging, uniform barriers, and masked boundary tiles | ✅ bit-exact only for semantic-core certified exact f32 inputs; required irregular two-schedule WebGPU evidence |
| `prepareSemanticAttentionWgsl` | Verified attention plus independent online K/V-tile schedule lowering with cooperative staging, uniform barriers, and causal/tail masks before state updates | 🧪 preparation/codegen only; device execution and performance evidence remain separate |
| `rowWiseOnlineAttentionDirect` | Fused row-wise online-softmax attention baseline with strict real-WebGPU parity vs composed reference; not block-tiled FlashAttention. | ✅ |
| `flashAttentionDirect` | Deprecated compatibility alias for `rowWiseOnlineAttentionDirect`. | ⚠️ |
| `fusedElementwiseDirect` | Runtime WGSL codegen for arbitrary elementwise chains | ✅ |
| `runTensorGpuPlan` / `runTensorGpuPlanResident` | Generic tensor-plan executor for primitive f32 BUFFER/LOAD/MATMUL/elementwise/shape/reduce/Conv1d/Conv2d/ConvTranspose2d/Conv3d/LayerNorm-forward-backward/SGD/Adam/AdamW-update steps with GPUBuffer residency; materialized or resident root modes | ✅ |
| `WebGpuRealizerBridge.conv1d*` / `conv2d*` | Resident Conv1d/Conv2d forward plus input/weight/bias backward kernels for jit explicit realization. | ✅ |

### Realizer-tier surface (consumed by jit)

- `createWebGpuRealizerBridge(device)` — production bridge satisfying the `WebGpuBridge` Protocol declared in jit. Opaque integer handles; bridge owns `GPUBuffer` lifetimes; pipeline cache via `runDirect`.
- `runDirect(device, desc, opts)` — `GPUBuffer`-in / `GPUBuffer`-out dispatch. The realizer-tier path; no host round-trip per op.
  By default, owned output buffers come from a per-device reusable output pool
  (`createDevice({ outputBufferPoolSize })`), visible through
  `device.getStats().outputBufferPool*`. Pass `opts.outputBuffer` when the
  caller owns output storage.
- `runTensorGpuPlan(device, plan, inputs)` — generic tensor-IR plan executor.
  It consumes scheduled primitive steps, accepts the snake_case plan payload
  emitted by `browsergrad-jit`, keeps intermediates resident, and materializes
  only the declared root. Current coverage: f32 BUFFER/LOAD/2-D MATMUL,
  scalar elementwise ops, FUSED_ELEMENTWISE runtime WGSL codegen,
  FUSED_SOFTMAX last-axis direct softmax, RESHAPE, PERMUTE, BROADCAST_TO, and
  REDUCE(sum/mean) rank <= 4, plus
  Conv1d/Conv2d/ConvTranspose2d/Conv3d/LayerNorm forward/backward and
  functional SGD/Adam/AdamW updates. SDPA-shaped graphs run as primitive
  tensor-plan steps (`MATMUL` -> scale -> `FUSED_SOFTMAX` -> `MATMUL`), not
  a framework-specific bridge method. The executor uses plan liveness to
  return dead owned direct-dispatch outputs to the reusable output pool before
  the root boundary, destroys uploaded host inputs when they die, and reports
  `earlyReleasedBuffers` / `earlyReleasedBytes` for tests and profiling. This
  is the future framework-runtime direction; per-op bridge methods are
  legacy/interim coverage.
- `runTensorGpuPlanResident(device, plan, inputs)` — same executor, but returns
  an owned resident root `GPUBuffer` instead of reading it back. Inputs may be
  host data or bridge-owned resident handles.
- `createWebGpuRealizerBridge(device).run_tensor_plan(plan, inputs, dtype)` —
  Pyodide bridge entrypoint for the same graph-level executor. Prefer extending
  this path for core framework ops instead of adding new per-op bridge methods.
- `createWebGpuRealizerBridge(device).run_tensor_plan_resident(plan, inputs, dtype)` —
  bridge entrypoint for resident tensor-plan roots; follow-on plans can pass
  the returned handle as an input without CPU readback.
- `materializeFloat32(device, buffer, byteLength)` — read a `GPUBuffer` back to a `Float32Array` (the single readback at the realize boundary).
- `uploadFloat32(device, data)` — upload a typed array into a fresh `GPUBuffer`.
- `createWgslStorageBuffer()` / `writeWgslStorageBuffer()` /
  `readWgslStorageBuffer()` — caller-owned resident storage buffers for generic
  WGSL programs. Use `residentBuffers` with `runWgslKernelProgramSequence()` to
  avoid per-call upload and skip readback with `readback: []`.
- `prepareWgslKernelProgramSequence()` — prebuilds pipelines and bind groups
  once, then reruns the same WGSL sequence over resident buffers for hot loops.
- `getWgslPipelineCacheStats()` / `clearWgslPipelineCache()` — inspect or
  invalidate only the generic WGSL program cache for one device. This is kept
  separate from `device.getStats()` so compiler-prepared sequences have
  attributable cache telemetry.

## Install

```bash
npm install @unlocalhosted/browsergrad-kernels
```

## Public import surface

Most consumers should use the top-level package:

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

These subpaths are stable for bundlers and agents that want smaller,
domain-specific imports:

```ts
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";
import { defineWgslKernelProgram } from "@unlocalhosted/browsergrad-kernels/wgsl_program";
import { createWgslFloat16Array } from "@unlocalhosted/browsergrad-kernels/float16";
import { runThreadGrid } from "@unlocalhosted/browsergrad-kernels/cuda_concepts";
import { defineCuda1DProgram } from "@unlocalhosted/browsergrad-kernels/cuda_program";
import { createKernelRubric } from "@unlocalhosted/browsergrad-kernels/rubric";
import {
  prepareSemanticViewCopyWgsl,
  runSemanticViewCopyWebGpu,
} from "@unlocalhosted/browsergrad-kernels/semantic_view_copy";
```

Do not import private files under `src/` or `dist/` from consumer code.

`PreparedSemanticViewCopyWgsl` values are immutable and accepted only by the
module instance that prepared them. View-copy execution bounds its planned
owned working set, permits one in-flight run per `GPUDevice`, distinguishes
shader/pipeline/validation/memory/device-loss failures, and retains the device
slot after timeout or cancellation until submitted work has cleaned up. Pass an
`AbortSignal` or a bounded `timeoutMs` to `runSemanticViewCopyWebGpu` when the
caller owns a shorter lifecycle.
Release CI verifies the packed tarball exports these entry points before npm
publish.

### Verified semantic view copy

The semantic view-copy surface accepts only opaque verified layout/kernel
artifacts from `@unlocalhosted/browsergrad-semantic-core`. Shared preparation
resolves bindings, proves every guarded access and dense destination write,
and derives one semantic specialization hash. Kernels then lowers the same
canonical index/predicate expressions into a signed-i32 WebGPU profile.

Root allocations are bound at offset zero as `array<u32>`, so ordinary values,
signed zero, infinities, and NaN payloads copy bit-for-bit. Padding initializes
exact fill bits and performs the source load only inside a structured `if`;
the lowerer never uses eager `select`, implicit robust-buffer zeroing, address
clamping, or ignored writes as semantics. Device execution validates storage,
dispatch, workgroup, and binding limits before submission and reports separate
pipeline, validation, memory, device-loss, and execution diagnostics.

The advisory focused lane records a real skip when no adapter exists:

```bash
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:view-copy
```

The evidence gate is deliberately strict and fails on missing adapter/device:

```bash
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:view-copy:required
```

Passing Node tests or packed-tarball checks is not WebGPU conformance evidence.

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

7 scenarios: adapter info, naive vs tiled matmul, residency contract (3 uploads + 1 readback chained matmul), fused-elementwise codegen output matches NumPy semantics, fused row-wise online-attention baseline (known-issue advisory), end-to-end `WebGpuRealizerBridge.matmul`.

Real-WebGPU CI is the only reliable way to catch shader-level bugs — NumPy mocks pass everything green even when the WGSL is wrong. A numerical issue in the direct online-attention baseline, tracked in the changelog, was caught this way.

## API stability

| Surface | Stability |
|---|---|
| `kernels.*`, `matmul`, `matmulTiled`, `softmax`, `relu`, `gelu`, `layernorm`, `attention` | Semver-stable across `0.x` |
| `runDirect`, `matmulTiledDirect`, `fusedElementwiseDirect`, `rowWiseOnlineAttentionDirect` | Semver-stable |
| `flashAttentionDirect` | Deprecated compatibility alias; retained through the documented removal window |
| `materializeFloat32`, `uploadFloat32` | Semver-stable |
| `createWebGpuRealizerBridge`, `WebGpuRealizerBridge` interface | Semver-stable; new methods added additively |
| `KernelError` | Semver-stable |
| WGSL source strings | **Internal.** Tuned freely. |
| Pipeline cache keys | **Internal.** Same WGSL → same key, but the encoding may change. |

## License

[MIT](LICENSE).
