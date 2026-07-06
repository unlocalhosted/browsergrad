# BrowserGrad Compiler Examples

These examples are canonical CUDA-lite lab slices. Each one should compile
through the same production path:

```text
CUDA-lite source -> semantic Kernel IR -> CPU reference -> WGSL/WebGPU plan
```

## `saxpy.cu`

Minimal 1D guarded kernel. Use it for first compile/run lessons and hot-loop
prepared execution:

```ts
import { createDevice } from "@unlocalhosted/browsergrad-kernels";
import {
  compileCudaLiteKernelForWebGpu,
  createCudaWebGpuExecutionPlan,
  prepareCompiledKernelWebGpu,
  runCompiledKernelReference,
  runCompiledKernelWebGpu,
  summarizeCudaWebGpuExecutionPlan,
} from "@unlocalhosted/browsergrad-compiler";

const compiled = compileCudaLiteKernelForWebGpu(source, {
  workgroupSize: [8, 1, 1],
});
const launch = { gridDim: [1, 1, 1], blockDim: [8, 1, 1] } as const;
const input = {
  buffers: {
    x: new Float32Array([1, 2, 3, 4]),
    y: new Float32Array([10, 20, 30, 40]),
  },
  scalars: { a: 2, n: 4 },
};

const reference = runCompiledKernelReference(compiled, input, launch);
const plan = createCudaWebGpuExecutionPlan(compiled, input, launch);
const summary = summarizeCudaWebGpuExecutionPlan(plan);
if (summary.canRunOnWebGpu) {
  const device = await createDevice();
  const gpu = await runCompiledKernelWebGpu(device, compiled, input, launch);
}
```

For hot paths, keep buffers resident and prepare once:

```ts
const prepared = await prepareCompiledKernelWebGpu(
  device,
  compiled,
  { buffers: {}, residentBuffers: { x, y }, scalars: { a: 2, n: 4 }, readback: [] },
  launch,
);
await prepared.run({ readback: [] });
prepared.destroy();
```

## `guarded-map.cu`

Small branch/guard example. Use it when teaching out-of-bounds protection:
threads beyond `n` do no writes.

## `tiled-matmul.cu`

Shared-memory tiled matrix multiply. Use it when teaching `__shared__` memory
and `__syncthreads()` with a CPU reference trace before WebGPU dispatch.

## Unsupported Diagnostics

Unsupported features must report stable diagnostic codes with source spans.
Platform UI should branch on codes, not prose. Example families:

- `divergent-barrier`: barrier may not be reached uniformly.
- `const-pointer-write`: write through a const pointer.
- `missing-feature-shader-f16`: kernel needs `shader-f16` but adapter lacks it.
- `unsupported-local-pointer`: pointer/address pattern not modeled yet.

Use `formatCudaLiteDiagnostics()` for human-facing output and
`summarizeCudaWebGpuExecutionPlan()` for readiness UI.

