# @unlocalhosted/browsergrad-compiler

CUDA-lite compiler for browser-native GPU labs.

The default path is small and inspectable:

```text
CUDA-lite source -> BrowserGrad Kernel IR -> WGSL -> WebGPU
                     \-> lockstep CPU reference
```

This package is independent of Pyodide. It uses
`@unlocalhosted/browsergrad-kernels` for WGSL dispatch.

Compatibility vocabulary:

- **native lowering**: CUDA primitive lowers directly to WGSL/WebGPU.
- **GPU polyfill lowering**: CUDA primitive lowers into one or more real WebGPU passes.
- **CPU reference**: correctness and teaching trace, not primary runtime.
- **unsupported diagnostic**: no honest lowering exists yet.

## Quick Start

```ts
import { createDevice } from "@unlocalhosted/browsergrad-kernels";
import {
  compileCudaLiteKernel,
  createCudaWebGpuExecutionPlan,
  runCompiledKernelReference,
  runCompiledKernelWebGpu,
} from "@unlocalhosted/browsergrad-compiler";

const source = `
__global__ void saxpy(const float* x, float* y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}`;

const compiled = compileCudaLiteKernel(source, {
  workgroupSize: [8, 1, 1],
});
console.log(compiled.loweringPlan.canRunOnGpu);
const input = {
  buffers: {
    x: new Float32Array([1, 2, 3, 4]),
    y: new Float32Array([10, 20, 30, 40]),
  },
  scalars: { a: 2, n: 4 },
};
const launch = { gridDim: [1, 1, 1], blockDim: [8, 1, 1] } as const;

const reference = runCompiledKernelReference(compiled, input, launch);
const plan = createCudaWebGpuExecutionPlan(compiled, input, launch);
console.log(plan.supported && plan.kind);
const gpu = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);
```

Examples live in `examples/`: SAXPY, guarded map, and shared-memory tiled
matmul. The emitted WGSL is intentionally inspectable so labs can show source,
bindings, workgroup size, shared memory, and barriers directly.

For hot WebGPU paths, pass caller-owned buffers through `residentBuffers` and
set `readback: []`. This keeps data on GPU across compiler-dispatched kernels;
use `readWgslStorageBuffer()` from `@unlocalhosted/browsergrad-kernels` at the
actual materialization boundary.

## CUDA Memory Pools

The compiler lowers simple bump allocators to real WebGPU atomics:

- `DevicePool* pool` with `streamOrderedAllocate(pool, size)` / `deviceAllocate(pool, size)`.
- external device pools referenced as `deviceAllocate(&g_pool, size)` and supplied
  through `memoryPools.g_pool`.
- raw pool form `deviceAllocate(poolBase, offset, poolSize, size)` where `poolBase`
  is a pointer parameter and `offset` is an integer pointer counter.

`DevicePool*` inputs use `memoryPools`:

```ts
const input = {
  buffers: { out: new Float32Array(2) },
  memoryPools: {
    pool: { data: new Uint32Array(2), offset: new Uint32Array([0]) },
  },
  scalars: { n: 2 },
};
```

Pool pointers are byte offsets with `0` as null. Casted accesses such as
`((float*)ptr)[0]` lower to raw pool words for `DevicePool*`, and to the typed
base buffer for raw pointer pools. This is a teaching-grade CUDA allocator
primitive, not a course-specific shim.

## Dynamic Parallelism

Device-side launches parse into Kernel IR. By default they remain compile-time
runtime gaps. For CPU teaching traces, opt in:

```ts
compileCudaLiteKernel(source, {
  kernelName: "parent",
  referenceDynamicParallelism: true,
});
```

`runCompiledKernelReference` executes child kernels against shared buffers and
memory pools.

`runCompiledKernelWebGpu` can host-lift conservative child launches into a real
WebGPU dispatch sequence when the parent launch has one workgroup, the child
block size is host-evaluable, pointer args alias named storage buffers, and
child scalar args come from host-evaluable expressions. Named `DevicePool*`
arguments alias their backing pool data/offset bindings, and positive pointer
offsets such as `out + 1` lower to generated base-offset uniforms so WebGPU
bindings stay whole-buffer and alignment-safe. Inspect this before dispatching:

```ts
const plan = createCudaHostDynamicLaunchPlan(compiled, input, launch);
console.log(plan.supported, plan.reason);
```

Recursive launches, device-derived launch args, per-thread launch queues,
negative pointer offsets, and parent side effects after launch stay
reference-only so GPU output is never silently wrong.

## CUDA Runtime Reference Calls

Some runtime orchestration calls have CPU reference truth before GPU lowering.
Peer copies are byte-accurate over modeled buffers/pools when explicitly enabled:

```ts
compileCudaLiteKernel(source, {
  referenceCudaRuntime: true,
});
```

`cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, bytes, stream)` copies bytes
inside the reference interpreter. `runCompiledKernelWebGpu` can host-lift
conservative peer copies into a typed WebGPU copy dispatch when the call is
single-invocation guarded, source/destination are named `Float32Array`,
`Int32Array`, or `Uint32Array` buffers, offsets are non-negative and
host-evaluable, and byte count is element-aligned. The same peer-copy lift can
compose after a host-lifted child dispatch. Mixed buffer types, pools,
device-derived counts, and side effects after copy remain reference-only.

## Cooperative Grid Sync

`cg::grid_group::sync()` has CPU reference truth and a WebGPU phase-splitting
path for safe top-level uniform barriers:

```ts
compileCudaLiteKernel(source, {
  referenceGridSync: true,
});
```

The CPU reference scheduler runs all blocks in lockstep at grid barriers, so
block-0 reduction patterns see writes from every block. WebGPU runs safe
top-level uniform `grid.sync()` as multiple dispatch phases over shared GPU
buffers. Pure launch-derived locals are replayed in later phases, and shared
memory reuse is allowed when the phase rewrites it before reading. Non-uniform
sync, non-replayable locals, and shared-memory read-before-rewrite stay
reference-only.

Use `createCudaLoweringPlan(diagnostics)` and `describeCudaDiagnostic()` to group
compatibility gaps by semantic family instead of raw parser messages.
Use `createCudaRuntimePlan(compiled)` to see which kernels need host
orchestration (`device-launch`, `device-sync`, `peer-copy`, `grid-sync`) before
trying WebGPU single-dispatch execution.
Use `createCudaWebGpuExecutionPlan(compiled, input, launch, { compileKernel })`
to inspect the exact executable WebGPU plan before running: `single-dispatch`,
`grid-sync-phases`, `host-dynamic-launch`, or `host-peer-copy`. The runner uses
the same plan interface internally, so platform preflight and execution share
one source of truth.

## Browser Testing

```bash
pnpm test:browser
pnpm test:browser:open
```

`test:browser:open` keeps Chromium open for inspection; quit with `q`. If the
Vitest browser watch rerun path reports an orchestrator-session error, restart
the command instead of trusting that rerun.

## Performance Harness

```bash
pnpm --filter @unlocalhosted/browsergrad-compiler bench -- --markdown /tmp/bg-cuda-lite-bench.md
```

The harness emits stable JSON timing for compile, CPU reference, dynamic-launch
planning, and peer-copy planning paths. It is data-only: no fixed threshold is
portable across laptops, browsers, and CI runners. Use median/p95 deltas across
commits to catch regressions before promoting platform perf rubrics.
