# @unlocalhosted/browsergrad-compiler

CUDA-lite compiler for browser-native GPU labs.

The default path is small and inspectable:

```text
CUDA-lite source -> BrowserGrad Kernel IR -> WGSL -> WebGPU
                     \-> lockstep CPU reference
```

This package is independent of Pyodide. It uses
`@unlocalhosted/browsergrad-kernels` for WGSL dispatch.

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

## Browser Testing

```bash
pnpm test:browser
pnpm test:browser:open
```

`test:browser:open` keeps Chromium open for inspection; quit with `q`. If the
Vitest browser watch rerun path reports an orchestrator-session error, restart
the command instead of trusting that rerun.
