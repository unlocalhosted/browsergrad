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
const gpu = await runCompiledKernelWebGpu(await createDevice(), compiled, input, launch);
```

Examples live in `examples/`: SAXPY, guarded map, and shared-memory tiled
matmul. The emitted WGSL is intentionally inspectable so labs can show source,
bindings, workgroup size, shared memory, and barriers directly.

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

Use `createCudaLoweringPlan(diagnostics)` and `describeCudaDiagnostic()` to group
compatibility gaps by semantic family instead of raw parser messages.

## Browser Testing

```bash
pnpm test:browser
pnpm test:browser:open
```

`test:browser:open` keeps Chromium open for inspection; quit with `q`. If the
Vitest browser watch rerun path reports an orchestrator-session error, restart
the command instead of trusting that rerun.
