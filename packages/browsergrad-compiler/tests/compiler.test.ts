import { describe, expect, it } from "vitest";
import { createWgslFloat16Array } from "@unlocalhosted/browsergrad-kernels";
import {
  CudaLiteCompilerError,
  analyzeCudaLite,
  compileCudaLiteOptionsFromKernelFeatures,
  createCudaLiteCompileCacheKey,
  createCudaLiteCompilerCache,
  compileCudaLiteKernelForWebGpu,
  compileCudaLiteKernel,
  prepareCompiledKernelWebGpu,
  createCudaGridSyncPhasePlan,
  createCudaHostDynamicLaunchPlan,
  createCudaLaunchValidationDiagnostics,
  createCudaLoweringPlan,
  createCudaPeerCopyPlan,
  createCudaRuntimeCopyPlan,
  createCudaRuntimePlan,
  createCudaWebGpuExecutionPlan,
  cudaLiteWebGpuCompileOptions,
  cudaLiteFeatureOptionsFromKernelFeatures,
  describeCudaDiagnostic,
  formatCudaLiteDiagnostics,
  normalizeCudaWebGpuReadbackNames,
  parseCudaLite,
  runCompiledKernelReference,
  runCompiledKernelWebGpu,
  summarizeCudaWebGpuExecutionPlan,
  validateCudaKernelLaunch,
} from "../src/index";

const SAXPY = `
__global__ void saxpy(const float* x, float* y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}
`;

const TILED_MATMUL = `
__global__ void tiled(const float* A, const float* B, float* C, int N) {
  __shared__ float As[2][2];
  __shared__ float Bs[2][2];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
  float acc = 0.0;
  for (int t = 0; t < N; t += 2) {
    if (row < N && (t + tx) < N) {
      As[ty][tx] = A[row * N + t + tx];
    }
    if (col < N && (t + ty) < N) {
      Bs[ty][tx] = B[(t + ty) * N + col];
    }
    __syncthreads();
    for (int k = 0; k < 2; k++) {
      if ((t + k) < N) {
        acc += As[ty][k] * Bs[k][tx];
      }
    }
    __syncthreads();
  }
  if (row < N && col < N) {
    C[row * N + col] = acc;
  }
}
`;

const LOCAL_ARRAY = `
__global__ void localArray(float* out) {
  int i = threadIdx.x;
  float tmp[2][2];
  tmp[0][0] = (float)i;
  tmp[0][1] = tmp[0][0] + 1.0;
  tmp[1][0] = tmp[0][1] + 1.0;
  tmp[1][1] = tmp[1][0] + 1.0;
  out[i] = tmp[1][1];
}
`;

const DEVICE_POOL_ALLOC = `
__global__ void poolKernel(DevicePool* dp, float* out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  void* ptr = streamOrderedAllocate(dp, sizeof(float));
  if (ptr != nullptr && idx < N) {
    ((float*)ptr)[0] = 3.25f;
    out[idx] = ((float*)ptr)[0];
  }
}
`;

const RAW_POOL_ALLOC = `
__global__ void rawPoolKernel(float* poolBase, size_t* offset, size_t poolSize, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    void* ptr = deviceAllocate(poolBase, offset, poolSize, sizeof(float));
    if (ptr != nullptr) {
      ((float*)ptr)[0] = 4.5f;
    }
  }
}
`;

const DEVICE_POINTER_HELPERS = `
__device__ float loadAt(const float* ptr, int offset) {
  return ptr[offset];
}

__device__ void addAt(float* ptr, int offset, float value) {
  ptr[offset] += value;
}

__global__ void helperKernel(const float* x, float* y, float a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    addAt(y, idx, a * loadAt(x + 1, idx));
  }
}
`;

const SHARED_POINTER_HELPERS = `
__device__ float readTile(float* tile, int offset) {
  return tile[offset];
}

__device__ void writeTile(float* tile, int offset, float value) {
  tile[offset] = value;
}

__global__ void sharedHelper(float* out) {
  __shared__ float tile[4];
  int tid = threadIdx.x;
  writeTile(tile, tid, (float)(tid + 1));
  __syncthreads();
  out[tid] = readTile(tile, 3 - tid);
}
`;

const EXTERNAL_POOL_ALLOC = `
__global__ void externalPoolKernel(float* out) {
  float* ptr = (float*) deviceAllocate(&g_pool, sizeof(float));
  if (ptr != nullptr) {
    ((float*)ptr)[0] = 5.5f;
    out[0] = ((float*)ptr)[0];
  }
}
`;

function floatBits(value: number): number {
  const floats = new Float32Array([value]);
  return new Uint32Array(floats.buffer)[0] ?? 0;
}

describe("CUDA-lite compiler", () => {
  it("parses and compiles SAXPY to WGSL", () => {
    const ast = parseCudaLite(SAXPY);
    const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });

    expect(ast.kernels[0]?.name).toBe("saxpy");
    expect(compiled.ir.params.map((param) => param.name)).toEqual(["x", "y", "a", "n"]);
    expect(compiled.wgsl).toContain("@workgroup_size(8, 1, 1)");
    expect(compiled.wgsl).toContain("var<storage, read> x: array<f32>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> y: array<f32>;");
    expect(compiled.wgsl).toContain("params.a");
    const directPlan = createCudaWebGpuExecutionPlan(
      compiled,
      {
        buffers: { x: new Float32Array(4), y: new Float32Array(4) },
        scalars: { a: 2, n: 4 },
      },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    );
    expect(summarizeCudaWebGpuExecutionPlan(directPlan)).toMatchObject({
      canRunOnWebGpu: true,
      mode: "direct",
      kind: "single-dispatch",
      requiresHostOrchestration: false,
    });
  });

  it("runs SAXPY in the lockstep CPU reference interpreter", () => {
    const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([1, 2, 3, 4]),
          y: new Float32Array([10, 20, 30, 40]),
        },
        scalars: { a: 2, n: 4 },
      },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    );

    expect([...result.buffers.y as Float32Array]).toEqual([12, 24, 36, 48]);
    expect(result.trace.some((thread) => thread.writes.length > 0)).toBe(true);
  });

  it("caches compiled kernels with deterministic option keys and LRU eviction", () => {
    let compileCount = 0;
    const cache = createCudaLiteCompilerCache({
      maxEntries: 2,
      compile(source, options) {
        compileCount++;
        return compileCudaLiteKernel(source, options);
      },
    });

    const first = cache.compile(SAXPY, { workgroupSize: [8, 1, 1] });
    const second = cache.compile(SAXPY, { workgroupSize: [8, 1, 1] });
    const third = cache.compile(SAXPY, { workgroupSize: [4, 1, 1] });
    const fourth = cache.compile(LOCAL_ARRAY, { workgroupSize: [4, 1, 1] });

    expect(second).toBe(first);
    expect(third).not.toBe(first);
    expect(cache.size).toBe(2);
    expect(cache.stats).toEqual({ hits: 1, misses: 3, evictions: 1, entries: 2 });
    expect(compileCount).toBe(3);
    expect(cache.get(SAXPY, { workgroupSize: [8, 1, 1] })).toBeUndefined();
    expect(cache.get(SAXPY, { workgroupSize: [4, 1, 1] })).toBe(third);
    expect(cache.get(LOCAL_ARRAY, { workgroupSize: [4, 1, 1] })).toBe(fourth);
  });

  it("supports default compile options and zero-entry cache mode", () => {
    const defaulted = createCudaLiteCompilerCache({
      compileOptions: { workgroupSize: [8, 1, 1] },
    });
    const compiled = defaulted.compile(SAXPY);
    expect(compiled.ir.workgroupSize).toEqual([8, 1, 1]);
    expect(defaulted.compile(SAXPY)).toBe(compiled);

    const disabled = createCudaLiteCompilerCache({ maxEntries: 0 });
    expect(disabled.compile(SAXPY)).not.toBe(disabled.compile(SAXPY));
    expect(disabled.stats).toEqual({ hits: 0, misses: 2, evictions: 0, entries: 0 });
  });

  it("creates stable compile cache keys independent of option property order", () => {
    expect(createCudaLiteCompileCacheKey(SAXPY, {
      features: { subgroups: true, "shader-f16": true },
      workgroupSize: [8, 1, 1],
    })).toBe(createCudaLiteCompileCacheKey(SAXPY, {
      workgroupSize: [8, 1, 1],
      features: { "shader-f16": true, subgroups: true },
    }));
  });

  it("lowers device helper functions with storage pointer params", () => {
    const compiled = compileCudaLiteKernel(DEVICE_POINTER_HELPERS, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([1, 2, 3, 4]),
          y: new Float32Array([10, 20, 30]),
        },
        scalars: { a: 2, n: 3 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
    expect(compiled.wgsl).toContain("fn bg_ptr_read_f32(buffer: u32, index: u32) -> f32");
    expect(compiled.wgsl).toContain("fn bg_ptr_write_f32(buffer: u32, index: u32, value: f32)");
    expect(compiled.wgsl).toContain("loadAt(0u, (0u + u32(1)), idx");
    expect(compiled.wgsl).toContain("addAt(1u, 0u, idx");
    expect([...result.buffers.y as Float32Array]).toEqual([14, 26, 38]);
  });

  it("rejects writes through const device helper pointer params", () => {
    expect(() => compileCudaLiteKernel(`
__device__ void bad(const float* x) {
  x[0] = 1.0f;
}

__global__ void kernel(const float* x) {
  bad(x);
}
`)).toThrow(CudaLiteCompilerError);
  });

  it("rejects passing const storage pointers to writable helper params", () => {
    expect(() => compileCudaLiteKernel(`
__device__ void addAt(float* ptr) {
  ptr[0] += 1.0f;
}

__global__ void kernel(const float* x) {
  addAt(x);
}
`)).toThrow(CudaLiteCompilerError);
  });

  it("lowers one-dimensional shared memory through helper pointer params", () => {
    const compiled = compileCudaLiteKernel(SHARED_POINTER_HELPERS, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("case 1u: { return tile[index]; }");
    expect(compiled.wgsl).toContain("case 1u: { tile[index] = value; return; }");
    expect(compiled.wgsl).toContain("writeTile(1u, 0u, tid");
    expect([...result.buffers.out as Float32Array]).toEqual([4, 3, 2, 1]);
  });

  it("reports precise diagnostics for unsupported helper pointer arguments", () => {
    const mismatch = analyzeCudaLite(parseCudaLite(`
__device__ void useInt(int* ptr) {}

__global__ void kernel(float* x) {
  useInt(x);
}
`));
    expect(mismatch.diagnostics).toContainEqual(expect.objectContaining({
      code: "unsupported-device-pointer-param",
      message: "device pointer parameter 'ptr' expects int pointer",
    }));

    const multiDimensionalShared = analyzeCudaLite(parseCudaLite(`
__device__ void useTile(float* ptr) {}

__global__ void kernel() {
  __shared__ float tile[2][2];
  useTile(&tile[0][0]);
}
`));
    expect(multiDimensionalShared.diagnostics).toContainEqual(expect.objectContaining({
      code: "unsupported-device-pointer-param",
      message: "device pointer parameter 'ptr' only supports one-dimensional shared arrays",
    }));
  });

  it("lowers fixed thread-local arrays through reference and WGSL", () => {
    const compiled = compileCudaLiteKernel(LOCAL_ARRAY, { workgroupSize: [4, 1, 1] });
    expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
    expect(compiled.wgsl).toContain("var tmp: array<array<f32, 2>, 2>;");

    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );
    expect([...result.buffers.out as Float32Array]).toEqual([3, 4, 5, 6]);
  });

  it("runs a shared-memory tiled matmul reference and emits barriers", () => {
    const compiled = compileCudaLiteKernel(TILED_MATMUL, { workgroupSize: [2, 2, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          A: new Float32Array([1, 2, 3, 4]),
          B: new Float32Array([5, 6, 7, 8]),
          C: new Float32Array(4),
        },
        scalars: { N: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 2, 1] },
    );

    expect([...result.buffers.C as Float32Array]).toEqual([19, 22, 43, 50]);
    expect(compiled.wgsl).toContain("var<workgroup> As: array<array<f32, 2>, 2>;");
    expect(compiled.wgsl).toContain("workgroupBarrier();");
  });

  it("allocates from a DevicePool and writes through casted pool pointers", () => {
    const compiled = compileCudaLiteKernel(DEVICE_POOL_ALLOC, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(2) },
        memoryPools: { dp: { data: new Uint32Array(2), offset: new Uint32Array([0]) } },
        scalars: { N: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<storage, read_write> dp_pool: array<u32>;");
    expect(compiled.wgsl).toContain("fn bg_pool_alloc_dp(size_bytes: u32) -> u32");
    expect([...result.buffers.out as Float32Array]).toEqual([3.25, 3.25]);
    expect([...result.buffers.dp as Uint32Array]).toEqual([floatBits(3.25), floatBits(3.25)]);
  });

  it("allocates from a raw pointer pool with a size_t offset counter", () => {
    const compiled = compileCudaLiteKernel(RAW_POOL_ALLOC, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          poolBase: new Float32Array(2),
          offset: new Uint32Array([0]),
        },
        scalars: { poolSize: 8, N: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn bg_raw_pool_alloc_poolBase_offset(pool_size_bytes: u32, size_bytes: u32) -> u32");
    expect(compiled.wgsl).toContain("var<storage, read_write> offset: array<atomic<u32>>;");
    expect([...result.buffers.poolBase as Float32Array]).toEqual([4.5, 4.5]);
    expect([...result.buffers.offset as Uint32Array]).toEqual([8]);
  });

  it("allocates from an external DevicePool reference", () => {
    const compiled = compileCudaLiteKernel(EXTERNAL_POOL_ALLOC, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(1) },
        memoryPools: { g_pool: { data: new Uint32Array(1), offset: new Uint32Array([0]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<storage, read_write> g_pool_pool: array<u32>;");
    expect(compiled.wgsl).toContain("fn bg_pool_alloc_g_pool(size_bytes: u32) -> u32");
    expect([...result.buffers.out as Float32Array]).toEqual([5.5]);
    expect([...result.buffers.g_pool as Uint32Array]).toEqual([floatBits(5.5)]);
  });

  it("supports unary pointer dereference in scalar expressions", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void derefKernel(const int* n, float* out) {
  if (threadIdx.x < *n) { out[0] = 1.0f; }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          n: new Int32Array([1]),
          out: new Float32Array(1),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("i32(local_id.x) < n[0]");
    expect([...result.buffers.out as Float32Array]).toEqual([1]);
  });

  it("supports local size_t declarations as uint scalars", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void sizeKernel(uint* out) {
  size_t bytes = sizeof(float);
  if (threadIdx.x < 1) { out[0] = bytes; }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var bytes: u32 = u32(4)");
    expect([...result.buffers.out as Uint32Array]).toEqual([4]);
  });

  it("accepts common C integer aliases as CUDA-lite i32/u32 scalars", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void integerAliases(int32_t *signedOut, uint32_t *unsignedOut, signed int n) {
  int idx = threadIdx.x;
  long long signedWide = (long long)idx - 2;
  signed short small = (signed short)n;
  unsigned long long unsignedWide = (unsigned long long)n + (uint64_t)idx;
  uintptr_t ptrValue = (uintptr_t)unsignedWide;
  uint32_t bytes = (uint32_t)sizeof(long);
  int64_t signedAlias = (int64_t)signedWide + (int32_t)small;
  if (idx < 2) {
    signedOut[idx] = signedAlias;
    unsignedOut[idx] = ptrValue + bytes;
  }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          signedOut: new Int32Array(2),
          unsignedOut: new Uint32Array(2),
        },
        scalars: { n: 5 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var signedWide: i32");
    expect(compiled.wgsl).toContain("var unsignedWide: u32");
    expect(compiled.wgsl).toContain("var ptrValue: u32");
    expect(compiled.wgsl).toContain("var bytes: u32 = u32(u32(4))");
    expect([...result.buffers.signedOut as Int32Array]).toEqual([3, 4]);
    expect([...result.buffers.unsignedOut as Uint32Array]).toEqual([9, 10]);
  });

  it("accepts CUDA opaque/index aliases and volatile qualifiers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void cudaAliasKernel(volatile size_type *out, curandState *state, CUtensorMap map, cudaGraphConditionalHandle handle) {
  volatile size_type idx = threadIdx.x;
  if (idx < 1) {
    out[0] = idx + map + handle + state[0];
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          out: new Uint32Array(1),
          state: new Uint32Array([5]),
        },
        scalars: { map: 7, handle: 11 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.ir.params.map((param) => [param.name, param.valueType])).toContainEqual(["map", "uint"]);
    expect([...result.buffers.out as Uint32Array]).toEqual([23]);
  });

  it("returns stable diagnostics for unsupported unsafe cases", () => {
    const constWrite = parseCudaLite(`
__global__ void bad(const float* x) {
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`);
    const constAnalysis = analyzeCudaLite(constWrite);
    expect(constAnalysis.diagnostics.map((diagnostic) => diagnostic.code)).toContain("const-pointer-write");

const unsupportedF32Atomic = parseCudaLite(`
__global__ void bad(float* x) {
  if (threadIdx.x < 1) { atomicCAS(&x[0], 0, 1); }
}`);
    const atomicAnalysis = analyzeCudaLite(unsupportedF32Atomic);
    expect(atomicAnalysis.diagnostics.map((diagnostic) => diagnostic.code)).toContain("unsupported-atomic-f32");

    const divergentBarrier = parseCudaLite(`
__global__ void bad(float* x) {
  if (threadIdx.x < 1) { __syncthreads(); }
}`);
    const barrierAnalysis = analyzeCudaLite(divergentBarrier);
    expect(barrierAnalysis.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-barrier");
  });

  it("hardens symbol and array validation", () => {
    const duplicate = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x, float* x) {
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`));
    expect(duplicate.diagnostics.map((diagnostic) => diagnostic.code)).toContain("duplicate-symbol");

    const localArrayInit = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x) {
  float tmp[2] = 1.0;
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`));
    expect(localArrayInit.diagnostics.map((diagnostic) => diagnostic.code)).toContain("unsupported-local-array-init");

    const localPointer = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x) {
  int* y = &x[0];
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`));
    expect(localPointer.diagnostics.map((diagnostic) => diagnostic.code)).toContain("unsupported-local-pointer");

    const invalidShared = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x) {
  __shared__ float tile[0];
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`));
    expect(invalidShared.diagnostics.map((diagnostic) => diagnostic.code)).toContain("invalid-array-dimension");
  });

  it("reports unguarded writes as warnings, not compiler blockers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void exactLaunch(float* x) {
  x[threadIdx.x] = 1.0;
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.diagnostics).toContainEqual(expect.objectContaining({
      code: "unguarded-write",
      severity: "warning",
    }));
    expect(compiled.loweringPlan.canRunOnGpu).toBe(true);
  });

  it("classifies CUDA compatibility gaps by semantic feature", () => {
    const unsupported = analyzeCudaLite(parseCudaLite(`
__global__ void unsupported(float* x) {
  if (threadIdx.x < 1) { atomicCAS(&x[0], 0, 1); }
}`));
    const plan = createCudaLoweringPlan(unsupported.diagnostics);

    expect(plan.canRunOnGpu).toBe(false);
    expect(plan.referenceAvailable).toBe(true);
    expect(plan.unsupported).toContainEqual(expect.objectContaining({
      code: "unsupported-atomic-f32",
      family: "atomic",
      lowering: "unsupported",
    }));
    expect(describeCudaDiagnostic({
      code: "unsupported-call",
      message: "unsupported CUDA-lite call 'tex2D'",
    })).toMatchObject({ family: "texture" });
    expect(describeCudaDiagnostic({
      code: "unsupported-call",
      message: "unsupported CUDA-lite call 'cudaMemcpyPeerAsync'",
    })).toMatchObject({ family: "runtime" });
  });

  it("rejects semantic gaps before WGSL/runtime execution", () => {
    const unknownSymbol = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x) {
  if (threadIdx.x < 1) { x[0] = missing + 1.0; }
}`));
    expect(unknownSymbol.diagnostics.map((diagnostic) => diagnostic.code)).toContain("unknown-symbol");

    const unsupportedCall = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x) {
  if (threadIdx.x < 1) { x[0] = erff(x[0]); }
}`));
    expect(unsupportedCall.diagnostics.map((diagnostic) => diagnostic.code)).toContain("unsupported-call");

    const runtimeCopy = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* dst, float* src, int device) {
  if (threadIdx.x < 1) { cudaMemcpyPeerAsync(dst, device, src, 0, sizeof(float), 0); }
}`));
    expect(runtimeCopy.diagnostics).toContainEqual(expect.objectContaining({
      code: "unsupported-cuda-runtime",
      severity: "error",
    }));

    const scalarParamWrite = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x, int n) {
  if (threadIdx.x < 1) { n = 2; x[0] = 1.0; }
}`));
    expect(scalarParamWrite.diagnostics.map((diagnostic) => diagnostic.code)).toContain("parameter-assignment");

    const scopedLocal = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x) {
  if (threadIdx.x < 1) { float tmp = 1.0; }
  if (threadIdx.x < 1) { x[0] = tmp; }
}`));
    expect(scopedLocal.diagnostics.map((diagnostic) => diagnostic.code)).toContain("unknown-symbol");

    const badAtomicAddress = analyzeCudaLite(parseCudaLite(`
__global__ void bad(int* x) {
  if (threadIdx.x < 1) { atomicAdd(x[0], 1); }
}`));
    expect(badAtomicAddress.diagnostics.map((diagnostic) => diagnostic.code)).toContain("atomic-address-required");

    const barrierExpression = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x) {
  int ok = __syncthreads();
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`));
    expect(barrierExpression.diagnostics.map((diagnostic) => diagnostic.code)).toContain("barrier-expression");

    const barrierArity = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x) {
  __syncthreads(1);
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`));
    expect(barrierArity.diagnostics.map((diagnostic) => diagnostic.code)).toContain("invalid-call-arity");

    const reservedName = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* var) {
  if (threadIdx.x < 1) { var[0] = 1.0; }
}`));
    expect(reservedName.diagnostics.map((diagnostic) => diagnostic.code)).toContain("reserved-symbol");

    const builtinShadow = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x) {
  float threadIdx = 0.0;
  if (blockIdx.x < 1) { x[0] = threadIdx; }
}`));
    expect(builtinShadow.diagnostics.map((diagnostic) => diagnostic.code)).toContain("reserved-symbol");

    const sideEffectCondition = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x, int n) {
  if ((n = 1) < 2) { x[0] = 1.0; }
}`));
    expect(sideEffectCondition.diagnostics.map((diagnostic) => diagnostic.code)).toContain("side-effect-expression");

    const sideEffectRhs = analyzeCudaLite(parseCudaLite(`
__global__ void bad(float* x) {
  int i = 0;
  if (threadIdx.x < 1) { x[0] = i++; }
}`));
    expect(sideEffectRhs.diagnostics.map((diagnostic) => diagnostic.code)).toContain("side-effect-expression");
  });

  it("rejects parser edge cases with clear errors", () => {
    expect(() => parseCudaLite(`
__global__ void bad(float* x) {
  if (threadIdx.x < 1) { x[0] = 1.2.3; }
}`)).toThrow(/invalid numeric literal/);
  });

  it("lowers standalone C block scopes", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void scoped(float* x) {
  int value = 1;
  {
    int value = 4;
    x[0] = (float)value;
  }
  x[1] = (float)value;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("{\n    var value: i32 = 4;");
    expect([...result.buffers.x as Float32Array]).toEqual([4, 1]);
  });

  it("accepts common CUDA lesson syntax in the CUDA-lite subset", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void lessonSyntax(const float *__restrict__ input, float *output, unsigned int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  unsigned int mask = 0xffffffff;
  if (idx >= n) return;
  if (idx < n) {
    float value = input[idx] + ((input[idx] > 0.0f) ? 0.5f : -0.5f);
    #pragma unroll
    for (int i = 0; i < 2U; i++) {
      if (i == 0) continue;
      value += 1.0f;
    }
    warp >>= 1;
    output[idx] = value + lane + warp + ((mask == 0xffffffff) ? 1.0f : 0.0f);
  }
}`, { workgroupSize: [32, 1, 1] });

    expect(compiled.wgsl).toContain("& 31");
    expect(compiled.wgsl).toContain(">> 5");
    expect(compiled.wgsl).toContain("0xffffffff");
    expect(compiled.wgsl).toContain("select");
    expect(compiled.wgsl).toContain("return;");
    expect(compiled.wgsl).toContain("continue;");
    expect(compiled.ir.params.map((param) => [param.name, param.valueType])).toContainEqual(["n", "uint"]);
  });

  it("lowers canonical CUDA while loops with continue", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void whileLoop(int *out) {
  int i = 0;
  int acc = 0;
  while (i < 5) {
    i++;
    if (i == 2) continue;
    acc += i;
  }
  out[0] = acc;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("while ((i < 5))");
    expect(compiled.wgsl).toContain("continue;");
    expect([...result.buffers.out as Int32Array]).toEqual([13]);
  });

  it("compiles stdout-only teaching kernels as no-op WebGPU programs", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void hello() {
  if (threadIdx.x == 0) {
    printf("hello %d\\n", threadIdx.x);
  }
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgslProgram.bindings).toEqual([]);
    expect(compiled.wgsl).toContain("printf omitted");
  });

  it("lowers scalar __device__ helper functions", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float addOne(float value) {
  return value + 1.0f;
}
__global__ void helperKernel(float *x) {
  if (threadIdx.x < 1) { x[0] = addOne(x[0]); }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([2]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.ir.functions.map((fn) => fn.name)).toEqual(["addOne"]);
    expect(compiled.wgsl).toContain("fn addOne(value_arg: f32");
    expect(compiled.wgsl).toContain("return (value + 1.0);");
    expect([...result.buffers.x as Float32Array]).toEqual([3]);
  });

  it("lowers CUDA warp shuffle helpers to subgroup intrinsics", () => {
    const compiled = compileCudaLiteKernel(`
__inline__ __device__ float warpReduceSum(float val) {
  unsigned int mask = 0xffffffff;
  val += __shfl_down_sync(mask, val, 16, 32);
  return val;
}
__global__ void warpKernel(const float *input, float *output) {
  int laneId = threadIdx.x & 31;
  float val = input[threadIdx.x];
  val = warpReduceSum(val);
  if (laneId == 0) { output[0] = val; }
}`, {
      features: { subgroups: true },
      workgroupSize: [32, 1, 1],
    });

    expect(compiled.wgsl).toContain("enable subgroups;");
    expect(compiled.wgsl).toContain("subgroupShuffleDown(val, u32(16))");
    expect(compiled.wgsl).toContain("warpReduceSum(val, local_id, workgroup_id, num_workgroups)");
  });

  it("lowers CUDA warp vote helpers to subgroup predicates", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void voteKernel(uint *input, uint *out) {
  uint mask = 0xffffffffu;
  out[0] = __any_sync(mask, input[0]);
  out[1] = __all_sync(mask, input[1]);
}`, {
      features: { subgroups: true },
      workgroupSize: [1, 1, 1],
    });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { input: new Uint32Array([7, 0]), out: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("subgroupAny");
    expect(compiled.wgsl).toContain("subgroupAll");
    expect(compiled.ir.requiredFeatures).toContain("subgroups");
    expect([...result.buffers.out as Uint32Array]).toEqual([1, 0]);
  });

  it("lowers cooperative-group block and tiled primitives to WebGPU primitives", () => {
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__global__ void tileReduce(const float *input, float *output) {
  cg::thread_block block = cg::this_thread_block();
  auto tile16 = cg::tiled_partition<16>(block);
  int tid = threadIdx.x;
  float val = input[tid];
  for (int offset = tile16.size() / 2; offset > 0; offset >>= 1) {
    val += tile16.shfl_down(val, offset);
  }
  if (tile16.thread_rank() == 0) { output[0] = val; }
  block.sync();
}`, {
      features: { subgroups: true },
      workgroupSize: [32, 1, 1],
    });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          input: new Float32Array(32).fill(1),
          output: new Float32Array(1),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [32, 1, 1] },
    );

    expect(compiled.wgsl).toContain("enable subgroups;");
    expect(compiled.wgsl).toContain("subgroupShuffleDown(val, u32(offset))");
    expect(compiled.wgsl).toContain("(i32(local_id.x) % 16)");
    expect(compiled.wgsl).toContain("workgroupBarrier();");
    expect([...result.buffers.output as Float32Array]).toEqual([16]);
  });

  it("passes cooperative-group handles through device helper parameters", () => {
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__device__ int block_rank(cg::thread_block block) {
  return block.thread_rank();
}
__device__ int tile_rank(cg::thread_block_tile<8> tile) {
  return tile.thread_rank();
}
__global__ void groupParam(int *out) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<8> tile = cg::tiled_partition<8>(block);
  out[threadIdx.x] = block_rank(block) + tile_rank(tile);
}`, {
      features: { subgroups: true },
      workgroupSize: [2, 1, 1],
    });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn block_rank(local_id: vec3<u32>");
    expect(compiled.wgsl).toContain("fn tile_rank(local_id: vec3<u32>");
    expect([...result.buffers.out as Int32Array]).toEqual([0, 2]);
  });

  it("classifies grid-wide cooperative sync as an explicit runtime gap", () => {
    const analysis = analyzeCudaLite(parseCudaLite(`
namespace cg = cooperative_groups;
__global__ void gridSync(float *x) {
  cg::grid_group grid = cg::this_grid();
  grid.sync();
  if (threadIdx.x < 1) { x[0] = 1.0f; }
}`));

    expect(analysis.diagnostics).toContainEqual(expect.objectContaining({
      code: "unsupported-cooperative-groups",
    }));
  });

  it("compiles host-orchestratable runtime gaps for WebGPU planning", () => {
    const source = `
namespace cg = cooperative_groups;
__global__ void gridSync(float *scratch, float *out) {
  cg::grid_group grid = cg::this_grid();
  scratch[blockIdx.x] = blockIdx.x + 1;
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    out[0] = scratch[0] + scratch[1];
  }
}`;
    expect(() => compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] })).toThrow(CudaLiteCompilerError);
    expect(cudaLiteWebGpuCompileOptions({ referenceGridSync: false })).toMatchObject({
      referenceDynamicParallelism: true,
      referenceGridSync: true,
      referenceCudaRuntime: true,
    });

    const compiled = compileCudaLiteKernelForWebGpu(source, { workgroupSize: [1, 1, 1] });
    expect(compiled.diagnostics).toContainEqual(expect.objectContaining({
      code: "unsupported-cooperative-groups",
      severity: "warning",
    }));
    const plan = createCudaWebGpuExecutionPlan(
      compiled,
      { buffers: { scratch: new Float32Array(2), out: new Float32Array(1) } },
      { gridDim: [2, 1, 1], blockDim: [1, 1, 1] },
      { compileKernel: compileCudaLiteKernelForWebGpu },
    );
    expect(summarizeCudaWebGpuExecutionPlan(plan)).toMatchObject({
      canRunOnWebGpu: true,
      mode: "host-orchestrated",
      kind: "grid-sync-phases",
    });
  });

  it("runs grid-wide cooperative sync in CPU reference when explicitly enabled", async () => {
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__global__ void gridSync(float *scratch, float *out) {
  cg::grid_group grid = cg::this_grid();
  scratch[blockIdx.x] = blockIdx.x + 1;
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    out[0] = scratch[0] + scratch[1];
  }
}`, {
      referenceGridSync: true,
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        scratch: new Float32Array(2),
        out: new Float32Array(1),
      },
    };
    const launch = { gridDim: [2, 1, 1], blockDim: [1, 1, 1] } as const;
    const result = runCompiledKernelReference(compiled, input, launch);

    expect(compiled.loweringPlan.canRunOnGpu).toBe(false);
    expect(compiled.diagnostics).toContainEqual(expect.objectContaining({
      code: "unsupported-cooperative-groups",
      severity: "warning",
    }));
    expect([...result.buffers.out as Float32Array]).toEqual([3]);
    expect(createCudaGridSyncPhasePlan(compiled.ir).supported).toBe(true);
    const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);
    expect(summarizeCudaWebGpuExecutionPlan(webGpuPlan)).toMatchObject({
      canRunOnWebGpu: true,
      mode: "host-orchestrated",
      kind: "grid-sync-phases",
      requiresHostOrchestration: true,
    });
  });

  it("runs cudaMemcpyPeerAsync in CPU reference when explicitly enabled", async () => {
    const compiled = compileCudaLiteKernel(`
__global__ void peerCopy(float *dst, const float *src, int n) {
  if (threadIdx.x == 0) {
    cudaMemcpyPeerAsync(dst + 1, 1, src, 0, sizeof(float) * n, 0);
  }
}`, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          dst: new Float32Array([0, 0, 0, 0]),
          src: new Float32Array([2.5, 3.5]),
        },
        scalars: { n: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics).toContainEqual(expect.objectContaining({
      code: "unsupported-cuda-runtime",
      severity: "warning",
    }));
    expect([...result.buffers.dst as Float32Array]).toEqual([0, 2.5, 3.5, 0]);
    const plan = createCudaPeerCopyPlan(
      compiled,
      {
        buffers: {
          dst: new Float32Array(4),
          src: new Float32Array([2.5, 3.5]),
        },
        scalars: { n: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect(plan.supported).toBe(true);
    expect(plan.copies[0]).toMatchObject({
      dstRoot: "dst",
      dstOffset: 1,
      srcRoot: "src",
      srcOffset: 0,
      elementCount: 2,
      valueType: "float",
    });

    const residentPlan = createCudaPeerCopyPlan(
      compiled,
      {
        buffers: {},
        residentBuffers: {
          dst: { buffer: {} as GPUBuffer, byteLength: 16, valueType: "f32" },
          src: { buffer: {} as GPUBuffer, byteLength: 8, valueType: "f32" },
        },
        scalars: { n: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect(residentPlan.supported).toBe(true);
    expect(residentPlan.copies[0]).toMatchObject({ dstRoot: "dst", srcRoot: "src", elementCount: 2 });

    const shortResidentPlan = createCudaPeerCopyPlan(
      compiled,
      {
        buffers: {},
        residentBuffers: {
          dst: { buffer: {} as GPUBuffer, byteLength: 16, valueType: "f32" },
          src: { buffer: {} as GPUBuffer, byteLength: 4, valueType: "f32" },
        },
        scalars: { n: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect(shortResidentPlan.supported).toBe(false);
  });

  it("runs cudaMemcpy and cudaMemcpyAsync through the host-copy planner", () => {
    const source = `
__global__ void runtimeCopy(float *dst, const float *src, int n) {
  cudaStream_t stream;
  cudaEvent_t event;
  if (threadIdx.x == 0) {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    cudaMemcpy(dst + 1, src, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(dst + 3, src + 1, sizeof(float), cudaMemcpyDefault, stream);
    cudaEventRecord(event, stream);
    cudaEventSynchronize(event);
    cudaStreamSynchronize(stream);
    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        dst: new Float32Array([0, 0, 0, 0]),
        src: new Float32Array([2.5, 3.5]),
      },
      scalars: { n: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const result = runCompiledKernelReference(compiled, input, launch);
    const plan = createCudaRuntimeCopyPlan(compiled, input, launch);
    const runtimePlan = createCudaRuntimePlan(compiled);

    expect([...result.buffers.dst as Float32Array]).toEqual([0, 2.5, 3.5, 3.5]);
    expect(runtimePlan.operations.map((operation) => operation.kind)).toEqual([
      "device-sync",
      "device-sync",
      "runtime-copy",
      "runtime-copy",
      "device-sync",
      "device-sync",
      "device-sync",
      "device-sync",
      "device-sync",
    ]);
    expect(plan.supported).toBe(true);
    expect(plan.copies.map((copy) => ({
      dstOffset: copy.dstOffset,
      srcOffset: copy.srcOffset,
      elementCount: copy.elementCount,
    }))).toEqual([
      { dstOffset: 1, srcOffset: 0, elementCount: 2 },
      { dstOffset: 3, srcOffset: 1, elementCount: 1 },
    ]);

    expect(() => compileCudaLiteKernel(`
__global__ void unsupportedKind(float *dst, const float *src) {
  if (threadIdx.x == 0) {
    cudaMemcpy(dst, src, sizeof(float), cudaMemcpyHostToDevice);
  }
}`, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    })).toThrow("unsupported-cuda-runtime-copy-kind");
  });

  it("explains why unsafe peer-copy lifts stay reference-only", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void peerCopyBad(float *dst, const float *src) {
  if (threadIdx.x == 0) {
    cudaMemcpyPeerAsync(dst, 0, src, 0, sizeof(float), 0);
    dst[0] = 9.0f;
  }
}`, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const plan = createCudaPeerCopyPlan(
      compiled,
      {
        buffers: {
          dst: new Float32Array(1),
          src: new Float32Array([2.5]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(plan.supported).toBe(false);
    expect(plan.reason).toContain("parent side effects after peer copy");
    expect(plan.blocker).toMatchObject({
      code: "unsafe-parent-side-effects",
      message: expect.stringContaining("parent side effects after peer copy"),
    });
  });

  it("summarizes runtime orchestration gaps without course-specific logic", () => {
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__global__ void child(float *x) {
  if (threadIdx.x == 0) { x[0] += 1.0f; }
}
__global__ void parent(float *dst, float *src) {
  cg::thread_block block = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();
  if (threadIdx.x == 0) {
    child<<<1, 1>>>(dst);
    cudaDeviceSynchronize();
    cudaMemcpyPeerAsync(dst, 0, src, 1, sizeof(float), 0);
  }
  block.sync();
  grid.sync();
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      referenceGridSync: true,
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const plan = createCudaRuntimePlan(compiled);

    expect(plan.operations.map((operation) => operation.kind)).toEqual([
      "device-launch",
      "device-sync",
      "runtime-copy",
      "grid-sync",
    ]);
    expect(plan.canRunSingleDispatchWebGpu).toBe(false);
    expect(plan.referenceAvailable).toBe(true);
  });

  it("builds explicit WebGPU execution plans for native dispatch and host lifts", () => {
    const single = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });
    const singlePlan = createCudaWebGpuExecutionPlan(
      single,
      {
        buffers: {
          x: new Float32Array([1, 2]),
          y: new Float32Array([10, 20]),
        },
        scalars: { a: 2, n: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    );
    expect(singlePlan).toMatchObject({ supported: true, kind: "single-dispatch" });
    if (singlePlan.supported) {
      expect(singlePlan.steps).toHaveLength(1);
      expect(singlePlan.input.readback).toContain("y");
    }

    const peer = compileCudaLiteKernel(`
__global__ void peerCopy(float *dst, const float *src, int n) {
  if (threadIdx.x == 0) {
    cudaMemcpyPeerAsync(dst + 1, 1, src, 0, sizeof(float) * n, 0);
  }
}`, {
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const peerPlan = createCudaWebGpuExecutionPlan(
      peer,
      {
        buffers: {
          dst: new Float32Array(4),
          src: new Float32Array([2.5, 3.5]),
        },
        scalars: { n: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect(peerPlan).toMatchObject({ supported: true, kind: "host-copy" });
    if (peerPlan.supported) {
      expect(peerPlan.steps.map((step) => step.program.name)).toEqual(["peerCopy", "bg_peer_copy_float"]);
    }

    const dynamic = compileCudaLiteKernel(`
__global__ void child(float *dst, int n) {
  int idx = threadIdx.x;
  if (idx < n) { dst[idx] += 1.0f; }
}
__global__ void parent(float *x, int n) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(n);
    child<<<grid, block>>>(x, n);
    cudaDeviceSynchronize();
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const dynamicInput = { buffers: { x: new Float32Array([1, 2]) }, scalars: { n: 2 } };
    const dynamicLaunch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const dynamicWithoutCompiler = createCudaWebGpuExecutionPlan(dynamic, dynamicInput, dynamicLaunch);
    expect(dynamicWithoutCompiler).toMatchObject({
      supported: false,
      blockers: [{
        kind: "device-launch",
        code: "dynamic-child-compiler-unavailable",
        message: "dynamic child compiler unavailable for WebGPU host orchestration",
      }],
    });
    if (!dynamicWithoutCompiler.supported) {
      expect(dynamicWithoutCompiler.reason).toContain("dynamic-child-compiler-unavailable");
    }

    const dynamicPlan = createCudaWebGpuExecutionPlan(dynamic, dynamicInput, dynamicLaunch, {
      compileKernel: compileCudaLiteKernel,
    });
    expect(dynamicPlan).toMatchObject({ supported: true, kind: "host-dynamic-launch" });
    if (dynamicPlan.supported) {
      expect(dynamicPlan.steps).toHaveLength(1);
      expect(dynamicPlan.steps[0]?.storageAliases).toEqual({ dst: "x" });
    }
  });

  it("validates launch shape before reference or WebGPU execution", async () => {
    const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });
    const input = {
      buffers: {
        x: new Float32Array([1, 2]),
        y: new Float32Array([10, 20]),
      },
      scalars: { a: 2, n: 2 },
    };
    const badGrid = { gridDim: [0, 1, 1] as const, blockDim: [8, 1, 1] as const };
    const badBlock = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };

    expect(createCudaLaunchValidationDiagnostics(badGrid, compiled.ir.workgroupSize)).toContainEqual(expect.objectContaining({
      code: "launch-grid-dim-invalid",
      message: "launch.gridDim[0] must be a positive integer",
    }));
    const badGridPlan = createCudaWebGpuExecutionPlan(compiled, input, badGrid);
    expect(badGridPlan).toMatchObject({
      supported: false,
      blockers: [{
        kind: "launch",
        code: "launch-grid-dim-invalid",
      }],
    });
    expect(summarizeCudaWebGpuExecutionPlan(badGridPlan)).toMatchObject({
      canRunOnWebGpu: false,
      mode: "unsupported",
      requiresHostOrchestration: false,
      blockers: [expect.objectContaining({ code: "launch-grid-dim-invalid" })],
    });
    expect(() => validateCudaKernelLaunch(badBlock, compiled.ir.workgroupSize)).toThrow(CudaLiteCompilerError);
    expect(() => runCompiledKernelReference(compiled, input, badGrid)).toThrow("launch.gridDim[0] must be a positive integer");
    await expect(runCompiledKernelWebGpu(
      {} as never,
      compiled,
      input,
      badBlock,
    )).rejects.toMatchObject({
      diagnostics: [expect.objectContaining({
        code: "launch-workgroup-mismatch",
      })],
    });
  });

  it("maps logical DevicePool readback names to internal storage bindings", () => {
    const compiled = compileCudaLiteKernel(DEVICE_POOL_ALLOC, { workgroupSize: [2, 1, 1] });
    expect(normalizeCudaWebGpuReadbackNames(compiled, ["dp", "out", "dp"])).toEqual(["dp_pool", "out"]);

    const plan = createCudaWebGpuExecutionPlan(
      compiled,
      {
        buffers: { out: new Float32Array(2) },
        memoryPools: { dp: { data: new Uint32Array(2), offset: new Uint32Array([0]) } },
        scalars: { N: 2 },
        readback: ["dp"],
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(plan).toMatchObject({ supported: true });
    if (plan.supported) expect(plan.input.readback).toEqual(["dp_pool"]);
  });

  it("plans safe top-level grid sync as WebGPU dispatch phases", () => {
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__global__ void gridPhases(float *scratch, float *out) {
  cg::grid_group grid = cg::this_grid();
  scratch[blockIdx.x] = blockIdx.x + 1;
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    out[0] = scratch[0] + scratch[1];
  }
}`, {
      referenceGridSync: true,
      workgroupSize: [1, 1, 1],
    });
    const plan = createCudaGridSyncPhasePlan(compiled.ir);

    expect(plan.supported).toBe(true);
    if (plan.supported) {
      expect(plan.modules).toHaveLength(2);
      expect(plan.modules.map((module) => module.name)).toEqual([
        "gridPhases_grid_phase_0",
        "gridPhases_grid_phase_1",
      ]);
    }
    const executionPlan = createCudaWebGpuExecutionPlan(
      compiled,
      {
        buffers: {
          scratch: new Float32Array(2),
          out: new Float32Array(1),
        },
      },
      { gridDim: [2, 1, 1], blockDim: [1, 1, 1] },
    );
    expect(executionPlan).toMatchObject({ supported: true, kind: "grid-sync-phases" });
    if (executionPlan.supported) expect(executionPlan.steps).toHaveLength(2);
  });

  it("rejects grid sync phase splitting when private locals cross phases", () => {
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__global__ void badGridPhase(float *out) {
  cg::grid_group grid = cg::this_grid();
  float carry = out[blockIdx.x];
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) { out[0] = carry; }
}`, {
      referenceGridSync: true,
      workgroupSize: [1, 1, 1],
    });
    const plan = createCudaGridSyncPhasePlan(compiled.ir);

    expect(plan.supported).toBe(false);
    if (!plan.supported) expect(plan.reason).toContain("private thread state");
  });

  it("plans grid sync phases when shared memory is rewritten after sync", () => {
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__global__ void sharedReuse(float *out) {
  cg::grid_group grid = cg::this_grid();
  __shared__ float tile[2];
  int tid = threadIdx.x;
  tile[tid] = (float)(blockIdx.x * 2 + tid + 1);
  __syncthreads();
  if (tid == 0) { out[blockIdx.x] = tile[0] + tile[1]; }
  grid.sync();
  if (blockIdx.x == 0) {
    tile[tid] = out[tid];
    __syncthreads();
    if (tid == 0) { out[0] = tile[0] + tile[1]; }
  }
}`, {
      referenceGridSync: true,
      workgroupSize: [2, 1, 1],
    });
    const plan = createCudaGridSyncPhasePlan(compiled.ir);

    expect(plan.supported).toBe(true);
    if (plan.supported) expect(plan.modules).toHaveLength(2);
  });

  it("rejects grid sync phases when shared memory is read before rewrite", () => {
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__global__ void sharedCarry(float *out) {
  cg::grid_group grid = cg::this_grid();
  __shared__ float tile[2];
  int tid = threadIdx.x;
  tile[tid] = (float)(tid + 1);
  grid.sync();
  if (tid == 0) { out[0] = tile[0]; }
}`, {
      referenceGridSync: true,
      workgroupSize: [2, 1, 1],
    });
    const plan = createCudaGridSyncPhasePlan(compiled.ir);

    expect(plan.supported).toBe(false);
    if (!plan.supported) expect(plan.reason).toContain("read before rewrite");
  });

  it("supports cooperative-group shuffle variants and linear thread ranks", () => {
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__global__ void tileScan(const float *input, float *output) {
  cg::thread_block block = cg::this_thread_block();
  auto tile8 = cg::tiled_partition<8>(block);
  int rank = tile8.thread_rank();
  float val = input[rank];
  val += tile8.shfl_up(val, 1);
  val += tile8.shfl_xor(val, 2);
  tile8.sync();
  if (rank == 0) { output[0] = val; }
}`, {
      features: { subgroups: true },
      workgroupSize: [8, 4, 1],
    });

    expect(compiled.wgsl).toContain("subgroupShuffleUp(val, u32(1))");
    expect(compiled.wgsl).toContain("subgroupShuffleXor(val, u32(2))");
    expect(compiled.wgsl).toContain("(i32(local_id.x + local_id.y * 8u) % 8)");
    expect(compiled.wgsl).toContain("workgroupBarrier();");
  });

  it("supports namespace-form cooperative-group sync and tile reduce", () => {
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__global__ void namespaceTileReduce(const float *input, float *output) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<8> tile = cg::tiled_partition<8>(block);
  int rank = tile.thread_rank();
  float value = input[rank];
  float sum = cg::reduce(tile, value, cg::plus<float>{});
  if (rank == 0) { output[0] = sum; }
  cg::sync(block);
}`, {
      features: { subgroups: true },
      workgroupSize: [8, 1, 1],
    });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          input: new Float32Array(8).fill(1),
          output: new Float32Array(1),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    );

    expect(compiled.wgsl).toContain("subgroupAdd(value)");
    expect(compiled.wgsl).toContain("workgroupBarrier();");
    expect([...result.buffers.output as Float32Array]).toEqual([8]);
  });

  it("accepts C++ namespace aliases inside kernel bodies", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void kernelLocalNamespace(float *out) {
  namespace cg = cooperative_groups;
  using namespace cooperative_groups;
  out[0] = 1.0f;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Float32Array]).toEqual([1]);
  });

  it("lowers cooperative tile meta group size and rank", () => {
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__global__ void tileMeta(int *out) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<4> tile = cg::tiled_partition<4>(block);
  int rank = tile.thread_rank();
  int lane = threadIdx.x;
  out[lane] = tile.meta_group_size() * 10 + tile.meta_group_rank() + rank * 100;
}`, { workgroupSize: [8, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(8) } },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    );

    expect(compiled.wgsl).toContain("/ 4)");
    expect([...result.buffers.out as Int32Array]).toEqual([20, 120, 220, 320, 21, 121, 221, 321]);
  });

  it("allows loop-local variable names to be reused in independent loop scopes", () => {
    const analysis = analyzeCudaLite(parseCudaLite(`
__global__ void scopedLoops(float *x) {
  for (int s = 1; s > 0; s >>= 1) { x[0] += 1.0f; }
  for (int s = 1; s > 0; s >>= 1) { x[0] += 1.0f; }
}`));

    expect(analysis.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("duplicate-symbol");
  });

  it("lowers C-style multiple for-init declarations", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void multiFor(int *out) {
  int acc = 0;
  for (int i = 0, j = 3; i < 3; i++, j--) {
    acc += i + j;
  }
  out[0] = acc;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var i: i32 = 0;");
    expect(compiled.wgsl).toContain("var j: i32 = 3;");
    expect(compiled.wgsl).toContain("loop {");
    expect(compiled.wgsl).toContain("continuing {");
    expect([...result.buffers.out as Int32Array]).toEqual([9]);
  });

  it("supports bool locals and trailing commas in kernel parameter lists", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void boolKernel(int *data, int N,) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  bool even = (idx % 2 == 0);
  if (idx < N) {
    if (even) { data[idx] += 10; }
    else { data[idx] -= 10; }
  }
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { data: new Int32Array([1, 2, 3, 4]) }, scalars: { N: 4 } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var even: bool = ((idx % 2) == 0);");
    expect([...result.buffers.data as Int32Array]).toEqual([11, -8, 13, -6]);
  });

  it("lowers common CUDA float math builtins to WGSL and reference math", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void mathy(float *x, float *out) {
  int idx = threadIdx.x;
  if (idx < 2) {
    float value = x[idx];
    out[idx] = fabsf(value) +
      floorf(value) +
      ceilf(value) +
      truncf(value) +
      roundf(value) +
      sinf(value) +
      cosf(value) +
      tanf(value) +
      tanhf(value) +
      coshf(value) +
      sqrt(fabsf(value)) +
      sqrtf(fabsf(value)) +
      rsqrtf(fabsf(value) + 1.0f) +
      __saturatef(value) +
      __expf(value) +
      __logf(fabsf(value) + 1.0f) +
      powf(fabsf(value), 2.0f) +
      fminf(value, 1.0f) +
      fmaxf(value, -1.0f) +
      __fdividef(value, 2.0f) +
      fma(value, 2.0f, 1.0f) +
      fmaf(value, -1.0f, 0.5f);
  }
}`, { workgroupSize: [2, 1, 1] });
    const input = new Float32Array([-1.25, 0.6]);
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("abs(value)");
    expect(compiled.wgsl).toContain("floor(value)");
    expect(compiled.wgsl).toContain("ceil(value)");
    expect(compiled.wgsl).toContain("trunc(value)");
    expect(compiled.wgsl).toContain("round(value)");
    expect(compiled.wgsl).toContain("sin(value)");
    expect(compiled.wgsl).toContain("cos(value)");
    expect(compiled.wgsl).toContain("tan(value)");
    expect(compiled.wgsl).toContain("tanh(value)");
    expect(compiled.wgsl).toContain("cosh(value)");
    expect(compiled.wgsl).toContain("sqrt(abs(value))");
    expect(compiled.wgsl).toContain("inverseSqrt((abs(value) + 1.0))");
    expect(compiled.wgsl).toContain("clamp(value, 0.0, 1.0)");
    expect(compiled.wgsl).toContain("pow(abs(value), 2.0)");
    expect(compiled.wgsl).toContain("min(value, 1.0)");
    expect(compiled.wgsl).toContain("max(value, (-1.0))");
    expect(compiled.wgsl).toContain("fma(value, 2.0, 1.0)");
    expect(compiled.wgsl).toContain("fma(value, (-1.0), 0.5)");
    const expected = [...input].map((value) =>
      Math.abs(value) +
      Math.floor(value) +
      Math.ceil(value) +
      Math.trunc(value) +
      Math.round(value) +
      Math.sin(value) +
      Math.cos(value) +
      Math.tan(value) +
      Math.tanh(value) +
      Math.cosh(value) +
      Math.sqrt(Math.abs(value)) +
      Math.sqrt(Math.abs(value)) +
      (1 / Math.sqrt(Math.abs(value) + 1)) +
      Math.min(1, Math.max(0, value)) +
      Math.exp(value) +
      Math.log(Math.abs(value) + 1) +
      Math.pow(Math.abs(value), 2) +
      Math.min(value, 1) +
      Math.max(value, -1) +
      (value / 2) +
      (value * 2 + 1) +
      (value * -1 + 0.5)
    );
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
  });

  it("lowers C math aliases used in CUDA snippets", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void c_math_aliases(float *x, float *out) {
  float value = x[0];
  out[0] = fabs(value) + exp(value) + log(fabs(value) + 1.0f) +
    pow(fabs(value), 2.0f) + fmin(value, 1.0f) + fmax(value, -1.0f) +
    __sinf(value) + __cosf(value) + __tanf(value) + lerp(2.0f, 6.0f, 0.25f);
}`, { workgroupSize: [1, 1, 1] });
    const value = -0.25;
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([value]), out: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const expected = Math.abs(value) + Math.exp(value) + Math.log(Math.abs(value) + 1) +
      Math.pow(Math.abs(value), 2) + Math.min(value, 1) + Math.max(value, -1) +
      Math.sin(value) + Math.cos(value) + Math.tan(value) + 3;

    expect(compiled.wgsl).toContain("abs(value)");
    expect(compiled.wgsl).toContain("exp(value)");
    expect(compiled.wgsl).toContain("pow(abs(value), 2.0)");
    expect(compiled.wgsl).toContain("fma(0.25, (6.0 - 2.0), 2.0)");
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(expected, 5);
  });

  it("lets user device functions shadow CUDA math aliases when CUDA source defines them", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float lerp(float a, float b, float t) {
  return a + b + t;
}
__global__ void shadow_lerp(float *out) {
  out[0] = lerp(2.0f, 6.0f, 0.25f);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn lerp(");
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(8.25, 5);
  });

  it("lowers CUDA integer and assert intrinsics", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void intIntrinsics(int *x, uint *out) {
  int idx = threadIdx.x;
  if (idx < 1) {
    assert(x[0] >= 0);
    out[0] = uint(__clz(uint(x[0])));
    out[1] = uint(__mul24(x[0], 3));
    out[2] = __umul24(uint(x[0]), 4u);
    out[3] = umin(7u, 3u);
    out[4] = uint(ceil_div(x[0], 2));
    out[5] = uint(__ffs(0u));
    out[6] = uint(__ffs(8u));
    out[7] = uint(abs(-7));
    out[8] = UMUL(6u, 7u);
    out[9] = UMAD(6u, 7u, 2u);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Int32Array([5]), out: new Uint32Array(10) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("countLeadingZeros");
    expect(compiled.wgsl).toContain("countTrailingZeros");
    expect(compiled.wgsl).toContain("min(u32(7), u32(3))");
    expect(compiled.wgsl).toContain("0;");
    expect([...result.buffers.out as Uint32Array]).toEqual([29, 15, 20, 3, 3, 0, 4, 7, 42, 44]);
  });

  it("recognizes CUDA/C numeric named constants", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void constants(float* out, uint* kinds) {
  __shared__ float tile[WARP_SIZE / 16];
  if (threadIdx.x < 1) {
    out[0] = INFINITY;
    out[1] = -FLT_MAX;
    out[2] = M_PI;
    out[3] = NAN;
    tile[0] = (NULL == 0) ? 7.0f : 0.0f;
    out[4] = tile[0];
    out[5] = M_SQRT2 * M_2_SQRTPI * 0.5f + M_SQRT1_2;
    kinds[0] = cudaMemcpyDeviceToDevice + cudaStreamNonBlocking;
    kinds[1] = warpSize + WARP_SIZE;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(6), kinds: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const out = [...result.buffers.out as Float32Array];

    expect(compiled.wgsl).toContain("return bitcast<f32>(bits);");
    expect(compiled.wgsl).toContain("3.4028234663852886e38");
    expect(compiled.wgsl).toContain("3.141592653589793");
    expect(out[0]).toBe(Number.POSITIVE_INFINITY);
    expect(out[1]).toBeLessThan(-3e38);
    expect(out[2]).toBeCloseTo(Math.PI, 6);
    expect(Number.isNaN(out[3])).toBe(true);
    expect(out[4]).toBe(7);
    expect(out[5]).toBeCloseTo(Math.SQRT2 * (2 / Math.sqrt(Math.PI)) * 0.5 + Math.SQRT1_2, 6);
    expect([...result.buffers.kinds as Uint32Array]).toEqual([4, 64]);
  });

  it("lowers CUDA cache-hint loads and stores as plain pointer memory ops", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void cache_hint(const float* x, float* y) {
  int idx = threadIdx.x;
  if (idx < 2) {
    const float* base = x + 1;
    float value = __ldcs(base + idx);
    __stcs(y + idx, value + 1.0f);
  }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([0, 2, 4]), y: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bg_ptr_read_f32");
    expect(compiled.wgsl).toContain("bg_ptr_write_f32");
    expect([...result.buffers.y as Float32Array]).toEqual([3, 5]);
  });

  it("lowers CUDA float4 values as scalar storage memory views", () => {
    const compiled = compileCudaLiteKernel(`
__device__ inline float4 add_float4(const float4& a, const float4& b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__global__ void vectorSaxpy(float a, const float4* x, const float4* y, float4* z, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float4 x4 = x[i];
    const float4 y4 = y[i];
    float4 sum = add_float4(x4, y4);
    sum.w = sum.w + a;
    z[i] = make_float4(a * x4.x + y4.x, sum.y, sum.z, sum.w);
  }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
          y: new Float32Array([10, 20, 30, 40, 50, 60, 70, 80]),
          z: new Float32Array(8),
        },
        scalars: { a: 2, n: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<storage, read> x: array<f32>;");
    expect(compiled.wgsl).toContain("vec4<f32>");
    expect([...result.buffers.z as Float32Array]).toEqual([12, 22, 33, 46, 60, 66, 77, 90]);
  });

  it("maps CUDA Packed128 float aliases onto vector storage views", () => {
    const compiled = compileCudaLiteKernel(`
typedef float floatX;
typedef Packed128<floatX> x128;
__global__ void packed128_alias(const float* input, float* output) {
  int idx = threadIdx.x * x128::size;
  x128 value = reinterpret_cast<x128 *>(input + idx)[0];
  x128 next;
  for (int lane = 0; lane < value.size; lane++) {
    next[lane] = value[lane] + 1.0f;
  }
  reinterpret_cast<x128 *>(output + idx)[0] = next;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { input: new Float32Array([1, 2, 3, 4]), output: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("vec4<f32>");
    expect([...result.buffers.output as Float32Array]).toEqual([2, 3, 4, 5]);
  });

  it("supports CUDA vector scalar constructors", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void vector_splat(float4 *out, uint4 *kinds) {
  out[0] = make_float4(2.5f);
  kinds[0] = make_uint4(7u);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(4), kinds: new Uint32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("vec4<f32>(2.5, 2.5, 2.5, 2.5)");
    expect(compiled.wgsl).toContain("vec4<u32>(7, 7, 7, 7)");
    expect([...result.buffers.out as Float32Array]).toEqual([2.5, 2.5, 2.5, 2.5]);
    expect([...result.buffers.kinds as Uint32Array]).toEqual([7, 7, 7, 7]);
  });

  it("maps CUDA byte-vector aliases onto canonical uint vector values", () => {
    const compiled = compileCudaLiteKernel(`
__device__ inline int rgbToInt(float r, float g, float b) {
  return (int)(r + g + b);
}
__global__ void byte_vectors(int *out) {
  uchar4 color = make_uchar4(1, 2, 3, 4);
  out[0] = rgbToInt(color.z, color.y, color.x);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("vec4<u32>(1, 2, 3, 4)");
    expect(compiled.wgsl).toContain("rgbToInt(f32(color.z), f32(color.y), f32(color.x)");
    expect([...result.buffers.out as Int32Array]).toEqual([6]);
  });

  it("lowers CUDA clock_t, clock(), and clock64() to deterministic synthetic counters", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void synthetic_clock(clock_t *out) {
  out[threadIdx.x] = clock();
  out[threadIdx.x + 4] = clock64();
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(8) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("workgroup_id.x * 104729u");
    expect([...result.buffers.out as Uint32Array]).toEqual([0, 1, 2, 3, 0, 1, 2, 3]);
  });

  it("accepts CUDA declarator qualifiers, alignment attrs, and constructor-style vector locals", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void decl_frontend(float const *const input, float *const out, size_t const n) {
  __shared__ __align__(16) float4 tile[1];
  int i = threadIdx.x;
  if (uint(i) < n) {
    alignas(16) float scalar(2.0f);
    float4 value(1.0f, 2.0f, 3.0f, 4.0f);
    tile[0] = value;
    out[i] = input[i] + scalar + tile[0].x + tile[0].w;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          input: new Float32Array([3]),
          out: new Float32Array(1),
        },
        scalars: { n: 1 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<workgroup> tile: array<vec4<f32>, 1>;");
    expect(compiled.wgsl).toContain("vec4<f32>(1.0, 2.0, 3.0, 4.0)");
    expect([...result.buffers.out as Float32Array]).toEqual([10]);
  });

  it("lowers dynamic CUDA vector lane access", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void dynamicLane(float *out, int lane) {
  float4 value = make_float4(2.0f, 4.0f, 6.0f, 8.0f);
  out[0] = vec_at(value, lane);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) }, scalars: { lane: 2 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("[u32(params.lane)]");
    expect([...result.buffers.out as Float32Array]).toEqual([6]);
  });

  it("lowers CUDA half2 values as f16 vector storage views", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void half2Add(const half2 *x, half2 *y) {
  int i = threadIdx.x;
  half2 bias = {__float2half(1.0f), __float2half(2.0f)};
  half2 value = x[i];
  y[i] = make_half2(value.x + bias.x, value.y + bias.y);
}`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: createWgslFloat16Array([3, 5]),
          y: createWgslFloat16Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.ir.requiredFeatures).toContain("shader-f16");
    expect(compiled.wgsl).toContain("enable f16;");
    expect(compiled.wgsl).toContain("vec2<f16>");
    expect(Array.from(result.buffers.y as ArrayLike<number>)).toEqual([4, 7]);
  });

  it("lowers CUDA half2 arithmetic intrinsics", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void half2Ops(const half2 *x, const half2 *y, half2 *out) {
  half2 sum = __hadd2(x[0], y[0]);
  half2 prod = __hmul2(sum, make_half2(__float2half(2.0f), __float2half(0.5f)));
  out[0] = __hmax2(prod, make_half2(__float2half(5.0f), __float2half(5.0f)));
}`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: createWgslFloat16Array([1, 8]),
          y: createWgslFloat16Array([2, 4]),
          out: createWgslFloat16Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("max(");
    expect(Array.from(result.buffers.out as ArrayLike<number>)).toEqual([6, 6]);
  });

  it("lowers CUDA shuffle, fence, and conversion intrinsics", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void intrinsic_pack(half2 *h, float2 *f, float *out) {
  int lane = __shfl_sync(0xffffffff, threadIdx.x, 0);
  __syncwarp(0xffffffff);
  __threadfence();
  half2 value = make_half2(__int2half_rn(lane + 1), __int2half_rn(4));
  h[0] = value;
  f[0] = __half22float2(value);
  out[0] = __fmaf_rn(f[0].x, 2.0f, f[0].y);
}`, { features: { "shader-f16": true, subgroups: true }, workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          h: createWgslFloat16Array(2),
          f: new Float32Array(2),
          out: new Float32Array(1),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.ir.requiredFeatures).toEqual(expect.arrayContaining(["shader-f16", "subgroups"]));
    expect(compiled.wgsl).toContain("subgroupShuffle(");
    expect(compiled.wgsl).toContain("workgroupBarrier()");
    expect(compiled.wgsl).toContain("storageBarrier()");
    expect([...result.buffers.f as Float32Array]).toEqual([1, 4]);
    expect([...result.buffers.out as Float32Array]).toEqual([6]);
  });

  it("lowers warp reduction aliases and half conversion aliases", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void reduction_alias_pack(const float *x, half2 *h, float *out) {
  int i = threadIdx.x;
  float sum = warp_reduce_sum_f32(x[i]);
  float maxv = warpReduceMax(sum);
  float minv = warp_reduce_min(maxv);
  h[0] = __hadd2(__float22half2_rn(make_float2(sum, maxv)), __floats2half2_rn(1.0f, 2.0f));
  h[1] = __float2half2_rn(3.0f);
  out[i] = minv + __half2float(hrsqrt(__float2half_rn(4.0f)));
}`, { features: { "shader-f16": true, subgroups: true }, workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([3]),
          h: createWgslFloat16Array(4),
          out: new Float32Array(1),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.ir.requiredFeatures).toEqual(expect.arrayContaining(["shader-f16", "subgroups"]));
    expect(compiled.wgsl).toContain("subgroupAdd(");
    expect(compiled.wgsl).toContain("subgroupMax(");
    expect(compiled.wgsl).toContain("subgroupMin(");
    expect(compiled.wgsl).toContain("vec2<f16>");
    expect(Array.from(result.buffers.h as ArrayLike<number>)).toEqual([4, 5, 3, 3]);
    expect([...result.buffers.out as Float32Array]).toEqual([3.5]);
  });

  it("feature-gates half2 behind shader-f16", () => {
    expect(() => compileCudaLiteKernel(`
__global__ void half2Gate(half2 *x) {
  x[0] = make_half2(__float2half(1.0f), __float2half(2.0f));
}`, { features: { "shader-f16": false }, workgroupSize: [1, 1, 1] })).toThrow(/half requires WebGPU shader-f16 support/);
  });

  it("lowers reinterpret_cast vector memory views over scalar storage", () => {
    const compiled = compileCudaLiteKernel(`
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
__global__ void addPacked(float *a, float *b, float *c, int n) {
  int idx = 4 * threadIdx.x;
  if ((idx + 3) < n) {
    float4 av = FLOAT4(a[idx]);
    float4 bv = FLOAT4(b[idx]);
    float4 cv;
    cv.x = av.x + bv.x;
    cv.y = av.y + bv.y;
    cv.z = av.z + bv.z;
    cv.w = av.w + bv.w;
    FLOAT4(c[idx]) = cv;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          a: new Float32Array([1, 2, 3, 4]),
          b: new Float32Array([10, 20, 30, 40]),
          c: new Float32Array(4),
        },
        scalars: { n: 4 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("vec4<f32>(a[");
    expect(compiled.wgsl).toContain("c[");
    expect([...result.buffers.c as Float32Array]).toEqual([11, 22, 33, 44]);
  });

  it("lowers local typed storage pointer views without emitting pointer vars", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void addPackedAlias(float *a, float *b, float *c) {
  int idx = 4 * threadIdx.x;
  float4 *ap = reinterpret_cast<float4 *>(&a[idx]);
  float4 *bp = reinterpret_cast<float4 *>(&b[idx]);
  float4 *cp = reinterpret_cast<float4 *>(&c[idx]);
  float4 av = ap[0];
  float4 bv = bp[0];
  float4 cv;
  cv.x = av.x + bv.x;
  cv.y = av.y + bv.y;
  cv.z = av.z + bv.z;
  cv.w = av.w + bv.w;
  cp[0] = cv;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          a: new Float32Array([2, 4, 6, 8]),
          b: new Float32Array([1, 3, 5, 7]),
          c: new Float32Array(4),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).not.toContain("var ap");
    expect(compiled.wgsl).toContain("vec4<f32>(a[");
    expect([...result.buffers.c as Float32Array]).toEqual([3, 7, 11, 15]);
  });

  it("keeps CUDA vector shared arrays as logical vec storage", () => {
    const compiled = compileCudaLiteKernel(`
__device__ inline float4 load_float4(float4* tile, int i) {
  return tile[i];
}
__global__ void sharedVector(const float4* x, float4* y) {
  __shared__ float4 tile[2];
  int tid = threadIdx.x;
  tile[tid] = x[tid];
  __syncthreads();
  float4 swapped = load_float4(tile, 1 - tid);
  y[tid] = make_float4(swapped.x, swapped.y, swapped.z, swapped.w + 1.0f);
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]),
          y: new Float32Array(8),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<workgroup> tile: array<vec4<f32>, 2>;");
    expect(compiled.wgsl).toContain("bg_ptr_read_f32x4");
    expect([...result.buffers.y as Float32Array]).toEqual([10, 20, 30, 41, 1, 2, 3, 5]);
  });

  it("lowers cp.async pointer-form copies to synchronous shared-memory copies", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void async_copy(const float *input, float *output) {
  __shared__ float tile[4];
  if (threadIdx.x < 1) {
    CP_ASYNC_CG(tile, input, 16);
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads();
  output[threadIdx.x] = tile[threadIdx.x] + 1.0f;
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { input: new Float32Array([1, 2, 3, 4]), output: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bg_ptr_write_f32");
    expect(compiled.wgsl).toContain("workgroupBarrier()");
    expect([...result.buffers.output as Float32Array]).toEqual([2, 3, 4, 5]);
  });

  it("lowers CUDA device cuRAND state to deterministic browser RNG helpers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void monteCarloPiKernel(unsigned long long *counts, int totalPoints, unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < totalPoints) {
    curandState_t state;
    curand_init(seed, idx, 0, &state);
    float x = curand_uniform(&state);
    float y = curand_uniform(&state);
    unsigned long long localCount = 0ULL;
    if (x * x + y * y <= 1.0f) { localCount = 1ULL; }
    counts[idx] = localCount;
  }
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { counts: new Uint32Array(4) }, scalars: { totalPoints: 4, seed: 1234 } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn bg_curand_uniform");
    expect(compiled.wgsl).toContain("var state: u32");
    expect([...result.buffers.counts as Uint32Array].every((value) => value === 0 || value === 1)).toBe(true);
  });

  it("supports cufftComplex buffers as interleaved complex64 values", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void magnitudeKernel(cufftComplex *data, float *mag, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float real = data[idx].x;
    float imag = data[idx].y;
    mag[idx] = sqrtf(real * real + imag * imag);
  }
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          data: new Float32Array([3, 4, 5, 12]),
          mag: new Float32Array(2),
        },
        scalars: { N: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<storage, read_write> data: array<vec2<f32>>;");
    expect(compiled.wgsl).toContain("data[idx].x");
    expect([...result.buffers.mag as Float32Array]).toEqual([5, 13]);
  });

  it("supports local cufftComplex values and whole-complex writeback", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void multiplyFreqDomain(cufftComplex *A, const cufftComplex *B, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    cufftComplex a = A[idx];
    cufftComplex b = B[idx];
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    A[idx] = c;
  }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          A: new Float32Array([1, 2, 3, 4]),
          B: new Float32Array([5, 6, 7, 8]),
        },
        scalars: { N: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var c: vec2<f32>");
    expect(compiled.wgsl).toContain("A[idx] = c");
    expect([...result.buffers.A as Float32Array]).toEqual([-7, 16, -11, 52]);
  });

  it("lowers supported inline PTX fma statements", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void asmFma(const float *A, const float *B, float *out) {
  int idx = threadIdx.x;
  float sum = out[idx];
  asm volatile (
    "fma.rn.f32 %0, %1, %2, %0;\\n\\t"
    : "+f"(sum)
    : "f"(A[idx]), "f"(B[idx])
  );
  out[idx] = sum;
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          A: new Float32Array([2, 3]),
          B: new Float32Array([4, 5]),
          out: new Float32Array([10, 20]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("sum = fma(A[idx], B[idx], sum);");
    expect([...result.buffers.out as Float32Array]).toEqual([18, 35]);
  });

  it("parses anonymous CUDA lambda kernel bodies", () => {
    const compiled = compileCudaLiteKernel(`
__global__ (cufftComplex *data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx].x *= 2.0f;
    data[idx].y *= 2.0f;
  }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { data: new Float32Array([1, 2, 3, 4]) },
        scalars: { N: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.ir.name).toBe("anonymous_kernel_1");
    expect([...result.buffers.data as Float32Array]).toEqual([2, 4, 6, 8]);
  });

  it("lowers cudaSurfaceObject_t surf2Dwrite to storage-backed surfaces", () => {
    const compiled = compileCudaLiteKernel(`
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
__global__ void surfaceWrite(cudaSurfaceObject_t outputSurf, int width, int height) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  if (x < width && y < height) {
    float value = tex2D(texRef, (float)x + 0.5f, (float)y + 0.5f);
    surf2Dwrite(value * 2.0f, outputSurf, x * sizeof(float), y, cudaBoundaryModeTrap);
  }
}`, { workgroupSize: [2, 2, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {},
        textures: { texRef: { width: 2, height: 2, data: new Float32Array([1, 2, 3, 4]) } },
        surfaces: { outputSurf: { width: 2, height: 2, data: new Float32Array(4) } },
        scalars: { width: 2, height: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 2, 1] },
    );

    expect(compiled.wgsl).toContain("var<storage, read_write> outputSurf: array<f32>;");
    expect(compiled.wgsl).toContain("bg_surf2dwrite_outputSurf");
    expect([...result.buffers.outputSurf as Float32Array]).toEqual([2, 4, 6, 8]);
  });

  it("lowers f32 atomic max helpers through CAS semantics", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void maxKernel(const float *input, float *result, int N) {
  int idx = threadIdx.x;
  if (idx < N) {
    atomicMaxFloat(result, input[idx]);
  }
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          input: new Float32Array([1, 9, 3, 7]),
          result: new Float32Array([2]),
        },
        scalars: { N: 4 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bg_atomicMax_f32");
    expect([...result.buffers.result as Float32Array]).toEqual([9]);
  });

  it("lowers f32 atomic min and sub helpers through CAS semantics", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomicFloatOps(float *minValue, float *subValue) {
  int idx = threadIdx.x;
  if (idx < 2) {
    atomicMin(&minValue[0], idx == 0 ? 5.0f : 3.0f);
    atomicSub(&subValue[0], idx == 0 ? 1.5f : 2.25f);
  }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          minValue: new Float32Array([10]),
          subValue: new Float32Array([10]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bg_atomicMin_f32");
    expect(compiled.wgsl).toContain("bg_atomicSub_f32");
    expect([...result.buffers.minValue as Float32Array]).toEqual([3]);
    expect([...result.buffers.subValue as Float32Array][0]).toBeCloseTo(6.25);
  });

  it("parses dynamic extern shared memory as a clear unsupported diagnostic", () => {
    const analysis = analyzeCudaLite(parseCudaLite(`
__global__ void dynamicShared(float *x) {
  extern __shared__ float scratch[];
  if (threadIdx.x < 1) { scratch[threadIdx.x] = x[0]; }
}`));

    expect(analysis.diagnostics.map((diagnostic) => diagnostic.code)).toContain("dynamic-shared-memory");
  });

  it("parses device-side kernel launches as a runtime compatibility gap", () => {
    const analysis = analyzeCudaLite(parseCudaLite(`
__global__ void child(float *x) { if (threadIdx.x < 1) { x[0] = 1.0f; } }
__global__ void parent(float *x) {
  if (threadIdx.x < 1) {
    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);
    child<<<grid, block>>>(x);
    cudaDeviceSynchronize();
  }
}`), { kernelName: "parent" });

    expect(analysis.diagnostics).toContainEqual(expect.objectContaining({
      code: "unsupported-dynamic-parallelism",
    }));
  });

  it("runs device-side kernel launches in the CPU reference when explicitly enabled", async () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *x) {
  int idx = threadIdx.x;
  if (idx < 2) { x[idx] += 1.0f; }
}
__global__ void parent(float *x) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(2);
    child<<<grid, block>>>(x);
    cudaDeviceSynchronize();
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([1, 2]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.loweringPlan.canRunOnGpu).toBe(false);
    expect(compiled.diagnostics).toContainEqual(expect.objectContaining({
      code: "unsupported-dynamic-parallelism",
      severity: "warning",
    }));
    expect([...result.buffers.x as Float32Array]).toEqual([2, 3]);
  });

  it("plans host-liftable dynamic launches without running WebGPU", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *dst, int n) {
  int idx = threadIdx.x;
  if (idx < n) { dst[idx] += 1.0f; }
}
__global__ void parent(float *x, int n) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(n);
    child<<<grid, block>>>(x, n);
    cudaDeviceSynchronize();
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const plan = createCudaHostDynamicLaunchPlan(
      compiled,
      { buffers: { x: new Float32Array([1, 2]) }, scalars: { n: 2 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(plan.supported).toBe(true);
    expect(plan.launches).toHaveLength(1);
    expect(plan.launches[0]).toMatchObject({
      gridDim: [1, 1, 1],
      blockDim: [2, 1, 1],
      storageAliases: { dst: "x" },
    });
  });

  it("plans host-expanded per-invocation dynamic launches with builtin coordinates", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *dst, int value) {
  if (threadIdx.x < 1) { dst[0] = (float)value; }
}
__global__ void parent(float *out, int limit) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= limit) return;
  int value = idx;
  if (idx > 1) {
    value = value + 10;
  }
  dim3 grid(1);
  dim3 block(1);
  child<<<grid, block>>>(out + idx, value);
  cudaDeviceSynchronize();
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [4, 1, 1],
    });
    const input = {
      buffers: { out: new Float32Array([0, 0, 0, 0]) },
      scalars: { limit: 3 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const plan = createCudaHostDynamicLaunchPlan(compiled, input, launch);

    expect(plan.supported).toBe(true);
    expect(plan.launches).toHaveLength(3);
    expect(plan.launches.map((item) => item.pointerBaseOffsets.dst ?? 0)).toEqual([0, 1, 2]);
    expect(plan.launches.map((item) => item.input.scalars?.value)).toEqual([0, 1, 12]);

    const executionPlan = createCudaWebGpuExecutionPlan(compiled, input, launch, {
      compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options),
    });
    expect(executionPlan).toMatchObject({
      supported: true,
      kind: "host-dynamic-launch",
    });
    expect(executionPlan.supported && executionPlan.steps).toHaveLength(3);
  });

  it("caps host-expanded dynamic launches before building huge plans", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *dst) {
  if (threadIdx.x < 1) { dst[0] = 1.0f; }
}
__global__ void parent(float *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dim3 grid(1);
  dim3 block(1);
  child<<<grid, block>>>(out + idx);
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [4, 1, 1],
    });
    const plan = createCudaHostDynamicLaunchPlan(
      compiled,
      { buffers: { out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      { maxHostExpandedParentInvocations: 2 },
    );

    expect(plan.supported).toBe(false);
    expect(plan.blocker).toMatchObject({
      code: "too-many-parent-invocations",
      message: "host-expanded dynamic launch needs 4 parent invocations; max is 2",
    });
    expect(() => createCudaHostDynamicLaunchPlan(
      compiled,
      { buffers: { out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      { maxHostExpandedParentInvocations: -1 },
    )).toThrow("maxHostExpandedParentInvocations must be a non-negative integer");
  });

  it("threads host orchestration caps through high-level WebGPU APIs", async () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *dst) {
  if (threadIdx.x < 1) { dst[0] = 1.0f; }
}
__global__ void parent(float *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dim3 grid(1);
  dim3 block(1);
  child<<<grid, block>>>(out + idx);
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [4, 1, 1],
    });
    const input = { buffers: { out: new Float32Array(4) } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const options = { maxHostExpandedParentInvocations: 2 };

    await expect(runCompiledKernelWebGpu({} as never, compiled, input, launch, options))
      .rejects.toThrow("too-many-parent-invocations");
    await expect(prepareCompiledKernelWebGpu({} as never, compiled, input, launch, options))
      .rejects.toThrow("too-many-parent-invocations");
  });

  it("treats host-evaluable inactive dynamic launches as single dispatch", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *dst) {
  if (threadIdx.x < 1) { dst[0] = 1.0f; }
}
__global__ void parent(float *out, int enabled) {
  if (enabled != 0) {
    dim3 grid(1);
    dim3 block(1);
    child<<<grid, block>>>(out);
    cudaDeviceSynchronize();
  }
  out[0] += 2.0f;
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const executionPlan = createCudaWebGpuExecutionPlan(
      compiled,
      { buffers: { out: new Float32Array([0]) }, scalars: { enabled: 0 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      { compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options) },
    );

    expect(executionPlan).toMatchObject({
      supported: true,
      kind: "single-dispatch",
    });
  });

  it("flattens recursive host-dynamic launches with a depth cap", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *dst, int value) {
  if (threadIdx.x < 1) { dst[0] += (float)value; }
}
__global__ void parent(float *out, int n) {
  dim3 grid(1);
  dim3 block(1);
  child<<<grid, block>>>(out, n);
  cudaDeviceSynchronize();
  if (n > 1) {
    parent<<<grid, block>>>(out, n - 1);
    cudaDeviceSynchronize();
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const input = { buffers: { out: new Float32Array([0]) }, scalars: { n: 2 } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const executionPlan = createCudaWebGpuExecutionPlan(compiled, input, launch, {
      compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options),
    });

    expect(executionPlan).toMatchObject({
      supported: true,
      kind: "host-dynamic-launch",
    });
    expect(executionPlan.supported && executionPlan.steps).toHaveLength(2);

    const capped = createCudaWebGpuExecutionPlan(compiled, input, launch, {
      compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options),
      maxHostDynamicLaunchDepth: 1,
    });
    expect(capped).toMatchObject({
      supported: false,
      blockers: [{
        code: "host-dynamic-launch-depth-exceeded",
      }],
    });
    expect(() => createCudaWebGpuExecutionPlan(compiled, input, launch, {
      compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options),
      maxHostDynamicLaunchDepth: -1,
    })).toThrow("maxHostDynamicLaunchDepth must be a non-negative integer");
  });

  it("plans host-liftable dynamic launches with DevicePool aliases", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(DevicePool *childPool, float *dst) {
  if (threadIdx.x < 1) {
    float *ptr = (float*) deviceAllocate(childPool, sizeof(float));
    if (ptr != nullptr) {
      ptr[0] = 6.0f;
      dst[0] = ptr[0];
    }
  }
}
__global__ void parent(DevicePool *pool, float *out) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(1);
    child<<<grid, block>>>(pool, out);
    cudaDeviceSynchronize();
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const pool = { data: new Uint32Array(1), offset: new Uint32Array([0]) };
    const plan = createCudaHostDynamicLaunchPlan(
      compiled,
      { buffers: { out: new Float32Array(1) }, memoryPools: { pool } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(plan.supported).toBe(true);
    expect(plan.launches[0]).toMatchObject({
      storageAliases: {
        childPool_pool: "pool_pool",
        childPool_offset: "pool_offset",
        dst: "out",
      },
    });
    expect(plan.launches[0]?.input.memoryPools?.childPool).toBe(pool);

    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) }, memoryPools: { pool } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect([...result.buffers.out as Float32Array]).toEqual([6]);
    expect([...result.buffers.pool as Uint32Array]).toEqual([floatBits(6)]);
  });

  it("plans host-lifted child launches over DevicePool allocation pointers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *data, int n) {
  int idx = threadIdx.x;
  if (idx < n) { data[idx] = (float)(idx + 1); }
}
__global__ void parent(DevicePool *pool, int n) {
  if (threadIdx.x < 1) {
    float *ptr = (float*) deviceAllocate(pool, n * sizeof(float));
    if (ptr != nullptr) {
      dim3 grid(1);
      dim3 block(n);
      child<<<grid, block>>>(ptr, n);
      cudaDeviceSynchronize();
    }
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const pool = { data: new Uint32Array(4), offset: new Uint32Array([0]) };
    const input = { buffers: {}, scalars: { n: 2 }, memoryPools: { pool } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const plan = createCudaHostDynamicLaunchPlan(compiled, input, launch);

    expect(plan.supported).toBe(true);
    expect(plan.launches[0]).toMatchObject({
      storageAliases: { data: "pool_pool" },
      pointerBaseOffsets: { data: 0 },
    });
    expect(plan.poolOffsetUpdates).toEqual({ pool: 8 });

    const executionPlan = createCudaWebGpuExecutionPlan(compiled, input, launch, {
      compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options),
    });
    expect(executionPlan).toMatchObject({
      supported: true,
      kind: "host-dynamic-launch",
    });
    if (executionPlan.supported) {
      expect(executionPlan.steps).toHaveLength(1);
      expect([...executionPlan.input.buffers.pool_offset as Uint32Array]).toEqual([8]);
      expect(executionPlan.input.storageMetadata?.pool_pool).toEqual({ valueType: "u32", compatibleValueTypes: ["f32", "i32"] });
      expect(executionPlan.input.storageMetadata?.pool_offset).toEqual({ valueType: "u32" });
    }
  });

  it("plans host-expanded dynamic launches over order-stable DevicePool allocations", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *data, int n) {
  int idx = threadIdx.x;
  if (idx < n) { data[idx] = (float)(idx + 1); }
}
__global__ void parent(DevicePool *pool, int n) {
  float *ptr = (float*) deviceAllocate(pool, n * sizeof(float));
  if (ptr != nullptr) {
    dim3 grid(1);
    dim3 block(n);
    child<<<grid, block>>>(ptr, n);
    cudaDeviceSynchronize();
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [4, 1, 1],
    });
    const pool = { data: new Uint32Array(8), offset: new Uint32Array([0]) };
    const input = { buffers: {}, scalars: { n: 2 }, memoryPools: { pool } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const plan = createCudaHostDynamicLaunchPlan(compiled, input, launch);

    expect(plan.supported).toBe(true);
    expect(plan.launches).toHaveLength(4);
    expect(plan.launches.map((item) => item.pointerBaseOffsets.data)).toEqual([0, 2, 4, 6]);
    expect(plan.poolOffsetUpdates).toEqual({ pool: 32 });

    const executionPlan = createCudaWebGpuExecutionPlan(compiled, input, launch, {
      compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options),
    });
    expect(executionPlan).toMatchObject({
      supported: true,
      kind: "host-dynamic-launch",
    });
    if (executionPlan.supported) {
      expect(executionPlan.steps).toHaveLength(4);
      expect([...executionPlan.input.buffers.pool_offset as Uint32Array]).toEqual([32]);
      expect(executionPlan.input.storageMetadata?.pool_pool).toEqual({ valueType: "u32", compatibleValueTypes: ["f32", "i32"] });
      expect(executionPlan.input.storageMetadata?.pool_offset).toEqual({ valueType: "u32" });
    }

    const result = runCompiledKernelReference(compiled, input, launch);
    expect([...result.buffers.pool as Uint32Array]).toEqual([
      floatBits(1), floatBits(2),
      floatBits(1), floatBits(2),
      floatBits(1), floatBits(2),
      floatBits(1), floatBits(2),
    ]);
  });

  it("treats launched __device__ functions as kernel-compatible child entries", () => {
    const source = `
__device__ void childKernel(float *data, int n) {
  int idx = threadIdx.x;
  if (idx < n) { data[idx] = (float)(idx + 1); }
}
__global__ void parent(DevicePool *pool, int n) {
  float *ptr = (float*) deviceAllocate(pool, n * sizeof(float));
  if (ptr != nullptr) {
    dim3 grid(1);
    dim3 block(n);
    childKernel<<<grid, block>>>(ptr, n);
    cudaDeviceSynchronize();
  }
}`;
    const compiled = compileCudaLiteKernel(source, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [2, 1, 1],
    });
    const child = compileCudaLiteKernel(source, {
      kernelName: "childKernel",
      referenceDynamicParallelism: true,
      workgroupSize: [2, 1, 1],
    });
    const input = {
      buffers: {},
      scalars: { n: 2 },
      memoryPools: { pool: { data: new Uint32Array(4), offset: new Uint32Array([0]) } },
    };
    const plan = createCudaHostDynamicLaunchPlan(compiled, input, {
      gridDim: [1, 1, 1],
      blockDim: [2, 1, 1],
    });

    expect(child.ir.name).toBe("childKernel");
    expect(compiled.wgsl).not.toContain("fn childKernel(");
    expect(child.wgsl).not.toContain("fn childKernel(");
    expect(plan.supported).toBe(true);
    expect(plan.launches).toHaveLength(2);
  });

  it("rejects host-expanded DevicePool allocations when child args depend on parent order", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *data, int n, int value) {
  int idx = threadIdx.x;
  if (idx < n) { data[idx] = (float)value; }
}
__global__ void parent(DevicePool *pool, int n) {
  float *ptr = (float*) deviceAllocate(pool, n * sizeof(float));
  if (ptr != nullptr) {
    dim3 grid(1);
    dim3 block(n);
    child<<<grid, block>>>(ptr, n, threadIdx.x);
    cudaDeviceSynchronize();
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [4, 1, 1],
    });
    const input = {
      buffers: {},
      scalars: { n: 2 },
      memoryPools: { pool: { data: new Uint32Array(8), offset: new Uint32Array([0]) } },
    };
    const plan = createCudaHostDynamicLaunchPlan(compiled, input, {
      gridDim: [1, 1, 1],
      blockDim: [4, 1, 1],
    });

    expect(plan).toMatchObject({
      supported: false,
      blocker: { code: "pool-allocation-order-sensitive" },
    });
  });

  it("plans multiple ordered host-liftable dynamic launches", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void addOne(float *dst, int n) {
  int idx = threadIdx.x;
  if (idx < n) { dst[idx] += 1.0f; }
}
__global__ void scaleTwo(float *out, int n) {
  int idx = threadIdx.x;
  if (idx < n) { out[idx] *= 2.0f; }
}
__global__ void parent(float *x, int n) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(n);
    addOne<<<grid, block>>>(x, n);
    cudaDeviceSynchronize();
    scaleTwo<<<grid, block>>>(x, n);
    cudaDeviceSynchronize();
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const plan = createCudaHostDynamicLaunchPlan(
      compiled,
      { buffers: { x: new Float32Array([1, 2]) }, scalars: { n: 2 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(plan.supported).toBe(true);
    expect(plan.launches.map((item) => item.kernel.name)).toEqual(["addOne", "scaleTwo"]);
  });

  it("plans dynamic child launches whose child performs host-liftable peer copy", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *dst, const float *src, int n) {
  if (threadIdx.x == 0) {
    cudaMemcpyPeerAsync(dst + 1, 1, src, 0, sizeof(float) * n, 0);
  }
}
__global__ void parent(float *dst, const float *src, int n) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(1);
    child<<<grid, block>>>(dst, src, n);
    cudaDeviceSynchronize();
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      referenceCudaRuntime: true,
      workgroupSize: [1, 1, 1],
    });
    const input = {
      buffers: {
        dst: new Float32Array([0, 0, 0, 0]),
        src: new Float32Array([2.5, 3.5]),
      },
      scalars: { n: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const plan = createCudaHostDynamicLaunchPlan(compiled, input, launch);
    const result = runCompiledKernelReference(compiled, input, launch);

    expect(plan.supported).toBe(true);
    expect(plan.launches[0]?.kernel.name).toBe("child");
    expect([...result.buffers.dst as Float32Array]).toEqual([0, 2.5, 3.5, 0]);
  });

  it("plans pointer-offset dynamic launches as pointer base uniforms", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *out) {
  if (threadIdx.x < 1) { out[0] = 7.0f; }
}
__global__ void parent(float *out) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(1);
    child<<<grid, block>>>(out + 1);
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const plan = createCudaHostDynamicLaunchPlan(
      compiled,
      { buffers: { out: new Float32Array([0, 0]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(plan.supported).toBe(true);
    expect(plan.launches[0]).toMatchObject({
      pointerBaseOffsets: { out: 1 },
      storageAliases: {},
    });
    const child = compileCudaLiteKernel(compiled.ast.source, {
      kernelName: "child",
      pointerBaseOffsets: plan.launches[0]!.pointerBaseOffsets,
      workgroupSize: [1, 1, 1],
    });
    expect(child.wgsl).toContain("bg_base_out");
  });

  it("keeps negative pointer-offset dynamic launches reference-only", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *out) {
  if (threadIdx.x < 1) { out[0] = 7.0f; }
}
__global__ void parent(float *out) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(1);
    child<<<grid, block>>>(out - 1);
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const plan = createCudaHostDynamicLaunchPlan(
      compiled,
      { buffers: { out: new Float32Array([0, 0]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(plan.supported).toBe(false);
  });

  it("keeps dynamic launches with parent side effects after launch reference-only", async () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *x) {
  if (threadIdx.x < 1) { x[0] = 2.0f; }
}
__global__ void parent(float *x) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(1);
    child<<<grid, block>>>(x);
    x[0] = 3.0f;
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const plan = createCudaHostDynamicLaunchPlan(
      compiled,
      { buffers: { x: new Float32Array([0]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(plan.supported).toBe(false);
    expect(plan.reason).toContain("parent side effects after device-side launch");
    expect(plan.blocker).toMatchObject({
      code: "unsafe-parent-side-effects",
      message: expect.stringContaining("parent side effects after device-side launch"),
    });
    const executionPlan = createCudaWebGpuExecutionPlan(
      compiled,
      { buffers: { x: new Float32Array([0]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      { compileKernel: compileCudaLiteKernel },
    );
    expect(executionPlan).toMatchObject({
      supported: false,
      blockers: [{
        kind: "device-launch",
        code: "unsafe-parent-side-effects",
      }],
    });
    await expect(runCompiledKernelWebGpu(
      {} as never,
      compiled,
      { buffers: { x: new Float32Array([0]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    )).rejects.toThrow("CUDA runtime orchestration is reference-only");
  });

  it("treats standalone cudaDeviceSynchronize as a WebGPU-safe no-op", () => {
    const source = `
__global__ void syncOnly(float *x) {
  if (threadIdx.x < 1) {
    cudaDeviceSynchronize();
    x[0] = 9.0f;
  }
}`;
    expect(() => compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] })).toThrow(CudaLiteCompilerError);
    const compiled = compileCudaLiteKernel(source, {
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([0]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics).toContainEqual(expect.objectContaining({
      code: "unsupported-cuda-runtime",
      severity: "warning",
    }));
    expect(createCudaRuntimePlan(compiled).operations.map((operation) => operation.kind)).toEqual(["device-sync"]);
    expect([...result.buffers.x as Float32Array]).toEqual([9]);
  });

  it("passes pointer-offset arguments into reference dynamic launches", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *out) {
  if (threadIdx.x < 1) { out[0] = 7.0f; }
}
__global__ void parent(float *out) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(1);
    child<<<grid, block>>>(out + 1);
  }
}`, {
      kernelName: "parent",
      referenceDynamicParallelism: true,
      workgroupSize: [1, 1, 1],
    });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array([0, 0]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Float32Array]).toEqual([0, 7]);
  });

  it("lowers named dynamic extern shared memory when launch metadata supplies its size", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void dynamicShared(float *x) {
  extern __shared__ float scratch[];
  int tid = threadIdx.x;
  if (tid < 2) { scratch[tid] = x[tid]; }
  __syncthreads();
  if (tid < 1) { x[0] = scratch[0] + scratch[1]; }
}`, {
      workgroupSize: [2, 1, 1],
      dynamicSharedMemory: { scratch: 2 },
    });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([2, 3]) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<workgroup> scratch: array<f32, 2>;");
    expect([...result.buffers.x as Float32Array]).toEqual([5, 3]);
  });

  it("lowers static shared declarations and scalar local-array initializers", () => {
    const compiled = compileCudaLiteKernel(`
static __device__ __forceinline__ float scale(float x) { return x * 2.0f; }
__global__ void init_arrays(float *out) {
  static __shared__ float shared[2];
  float vals[2][2] = {1.0f, 2.0f, 3.0f};
  int tid = threadIdx.x;
  if (tid < 2) { shared[tid] = vals[tid][0] + vals[tid][1]; }
  __syncthreads();
  if (tid < 1) { out[0] = scale(shared[0] + shared[1]); }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<workgroup> shared: array<f32, 2>;");
    expect(compiled.wgsl).toContain("vals[0][0] = 1.0;");
    expect([...result.buffers.out as Float32Array]).toEqual([12]);
  });

  it("lowers builtin infinity macros emitted by CUDA headers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void builtin_inf(float *out) {
  out[0] = -__builtin_inff();
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bitcast<f32>(0x7f800000u)");
    expect([...result.buffers.out as Float32Array]).toEqual([-Infinity]);
  });

  it("lowers local shared-memory pointer aliases as fixed shared offsets", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void splitShared(float *x) {
  extern __shared__ float sdataA[];
  float* sdataB = sdataA + 2;
  int tid = threadIdx.x;
  if (tid < 2) {
    sdataA[tid] = x[tid];
    sdataB[tid] = x[tid + 2];
  }
  __syncthreads();
  if (tid < 1) { x[0] = sdataA[1] + sdataB[1]; }
}`, {
      workgroupSize: [2, 1, 1],
      dynamicSharedMemory: { sdataA: 4 },
    });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([1, 2, 3, 4]) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).not.toContain("var sdataB");
    expect(compiled.wgsl).toContain("sdataA[((0 + 2)) + (tid)]");
    expect([...result.buffers.x as Float32Array]).toEqual([6, 2, 3, 4]);
  });

  it("supports scalar __shared__ declarations without dynamic shared metadata", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void sharedScalar(int *out) {
  __shared__ int localCount;
  if (threadIdx.x == 0) { localCount = 7; }
  __syncthreads();
  atomicAdd(&localCount, 1);
  __syncthreads();
  if (threadIdx.x == 1) { out[0] = localCount; }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<workgroup> localCount: atomic<i32>;");
    expect(compiled.wgsl).toContain("atomicStore(&localCount, 7)");
    expect(compiled.wgsl).toContain("atomicAdd(&localCount, 1)");
    expect(compiled.wgsl).toContain("atomicLoad(&localCount)");
    expect([...result.buffers.out as Int32Array]).toEqual([9]);
  });

  it("evaluates integer constant expressions in shared array dimensions", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void padded(float *x) {
  __shared__ float tile[16][16 + 1];
  int tid = threadIdx.x;
  if (tid < 1) { tile[0][0] = x[0]; x[0] = tile[0][0]; }
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("array<array<f32, 17>, 16>");
  });

  it("folds sizeof and alignof in integer constant expressions", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void layoutConsts(float *x) {
  constexpr int lanes = sizeof(float4) / sizeof(float);
  __shared__ float tile[alignof(float4) == 16 ? lanes : 1];
  int tid = threadIdx.x;
  if (tid < lanes) {
    tile[tid] = x[tid];
    x[tid] = tile[tid] + (float)alignof(float4);
  }
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([1, 2, 3, 4]) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("array<f32, 4>");
    expect([...result.buffers.x as Float32Array]).toEqual([17, 18, 19, 20]);
  });

  it("expands object-like macro constants before parsing", () => {
    const compiled = compileCudaLiteKernel(`
#define TILE_DIM 16 // trailing comments are ignored
#define PADDED_TILE (TILE_DIM + 1)
__global__ void padded(float *x) {
  __shared__ float tile[TILE_DIM][PADDED_TILE];
  int tid = threadIdx.x * TILE_DIM;
  if (tid < 1) { tile[0][0] = x[0]; x[0] = tile[0][0]; }
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("array<array<f32, 17>, 16>");
    expect(compiled.wgsl).toContain("* 16");
  });

  it("normalizes simple C++ aliases and CUDA kernel qualifiers before parsing", () => {
    const compiled = compileCudaLiteKernel(`
#define WARP_SIZE 32
typedef float scalar_t;
using count_t = unsigned int;
__global__ static void __launch_bounds__(WARP_SIZE * 2) boundedAlias(scalar_t *out, count_t n) {
  static_assert(WARP_SIZE == 32);
  constexpr int TILE = WARP_SIZE / 2;
  __shared__ scalar_t tile[TILE];
  int idx = threadIdx.x;
  if (idx < n && idx < TILE) {
    tile[idx] = (scalar_t)idx;
    out[idx] = tile[idx] + (scalar_t)TILE;
  }
}`, { workgroupSize: [16, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(16) }, scalars: { n: 2 } },
      { gridDim: [1, 1, 1], blockDim: [16, 1, 1] },
    );

    expect(compiled.ir.params.map((param) => [param.name, param.valueType])).toContainEqual(["n", "uint"]);
    expect(compiled.wgsl).toContain("array<f32, 16>");
    expect([...result.buffers.out as Float32Array].slice(0, 3)).toEqual([16, 17, 0]);
  });

  it("supports bounded integer template defaults in kernels and helpers", () => {
    const compiled = compileCudaLiteKernel(`
#define WARP_SIZE 32
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float reduce_default(float value) {
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    value = value + 0.0f;
  }
  return value;
}
template <const int NUM_THREADS = 64>
__global__ void templated(float *out) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float scratch[NUM_WARPS];
  int tid = threadIdx.x;
  if (tid < NUM_WARPS) { scratch[tid] = reduce_default<WARP_SIZE>(float(tid)); }
  __syncthreads();
  if (tid == 0) { out[0] = scratch[1]; }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.ir.sharedDeclarations[0]?.dimensions).toEqual([2]);
    expect(compiled.wgsl).toContain("var<workgroup> scratch: array<f32, 2>;");
    expect([...result.buffers.out as Float32Array]).toEqual([1]);
  });

  it("supports bool template defaults as constant expressions", () => {
    const compiled = compileCudaLiteKernel(`
template <const int TILE = 4, const bool UseBias = true>
__global__ void templatedBool(float *out) {
  __shared__ float scratch[TILE];
  int tid = threadIdx.x;
  if (tid < TILE) scratch[tid] = UseBias ? float(tid + 1) : float(tid);
  __syncthreads();
  if (tid == 0) out[0] = scratch[3];
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.ir.sharedDeclarations[0]?.dimensions).toEqual([4]);
    expect([...result.buffers.out as Float32Array]).toEqual([4]);
  });

  it("accepts C++ if constexpr in templated CUDA helpers", () => {
    const compiled = compileCudaLiteKernel(`
template <const int STEP = 8>
__device__ __forceinline__ int swizzle(int i, int j) {
  if constexpr (STEP == 8) {
    return (((j >> 3) ^ (i >> 2)) % 2) << 3;
  } else {
    return (((j >> 2) ^ (i >> 2)) % 4) << 2;
  }
}
__global__ void constexprIf(int *out) {
  constexpr int WIDTH = 8;
  int scratch[(WIDTH == 8) ? 2 : 1];
  scratch[0] = swizzle<8>(4, 8);
  if (threadIdx.x < 1) out[0] = scratch[0];
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("if ((8 == 8))");
    expect([...result.buffers.out as Int32Array]).toEqual([0]);
  });

  it("expands expression-style function macros before parsing", () => {
    const compiled = compileCudaLiteKernel(`
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
__global__ void macroIndex(const float *input, float *output, int M) {
  int row = threadIdx.x;
  if (row < 1) {
    output[0] = input[IDX2C(row, 0, M)];
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          input: new Float32Array([13]),
          output: new Float32Array(1),
        },
        scalars: { M: 1 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).not.toContain("IDX2C");
    expect([...result.buffers.output as Float32Array]).toEqual([13]);
  });

  it("parses C-style declaration lists as sequential locals", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void declarationList(float *x) {
  int row = threadIdx.y, col = threadIdx.x;
  if (row < 1 && col < 1) { x[0] = row + col + 1.0f; }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([0]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.x as Float32Array]).toEqual([1]);
    expect(compiled.wgsl).toContain("var row");
    expect(compiled.wgsl).toContain("var col");
  });

  it("accepts CUDA launch bounds as kernel metadata", () => {
    const compiled = compileCudaLiteKernel(`
__launch_bounds__(128, 2)
__global__ void bounded(float *x) {
  if (threadIdx.x < 1) { x[0] = 1.0f; }
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.ir.name).toBe("bounded");
    expect(compiled.wgsl).toContain("@workgroup_size(1, 1, 1)");
  });

  it("lowers CUDA constant scalar memory as readonly uniform input", () => {
    const compiled = compileCudaLiteKernel(`
__constant__ float scaleFactor;
__global__ void scale(const float *x, float *y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) { y[idx] = x[idx] * scaleFactor; }
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([1, 2, 3, 4]),
          y: new Float32Array(4),
        },
        constants: { scaleFactor: 3 },
        scalars: { n: 4 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.ir.constants.map((constant) => constant.name)).toEqual(["scaleFactor"]);
    expect(compiled.wgsl).toContain("scaleFactor: f32");
    expect(compiled.wgsl).toContain("params.scaleFactor");
    expect([...result.buffers.y as Float32Array]).toEqual([3, 6, 9, 12]);
  });

  it("lowers CUDA constant arrays as readonly storage inputs", () => {
    const compiled = compileCudaLiteKernel(`
__constant__ float coeffs[2];
__global__ void apply(float *x) {
  int idx = threadIdx.x;
  if (idx < 2) { x[idx] = x[idx] * coeffs[idx]; }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { x: new Float32Array([2, 4]) },
        constants: { coeffs: new Float32Array([10, 20]) },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<storage, read> coeffs: array<f32, 2>");
    expect(compiled.wgslProgram.bindings).toContainEqual(expect.objectContaining({
      name: "coeffs",
      access: "read",
    }));
    expect([...result.buffers.x as Float32Array]).toEqual([20, 80]);
  });

  it("embeds initialized CUDA constant memory", () => {
    const compiled = compileCudaLiteKernel(`
__constant__ float scale = 0.5f;
__constant__ short coeffs[] = {2, 3, 5};
__global__ void apply(float *x, int *out) {
  int idx = threadIdx.x;
  if (idx < 3) {
    x[idx] = x[idx] * scale;
    out[idx] = coeffs[idx];
  }
}`, { workgroupSize: [3, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { x: new Float32Array([4, 8, 12]), out: new Int32Array(3) },
      },
      { gridDim: [1, 1, 1], blockDim: [3, 1, 1] },
    );

    expect(compiled.ir.constants.map((constant) => [constant.name, constant.dimensions])).toEqual([
      ["scale", []],
      ["coeffs", [3]],
    ]);
    expect(compiled.wgsl).toContain("const scale: f32 = 0.5;");
    expect(compiled.wgsl).toContain("const coeffs: array<i32, 3> = array<i32, 3>(2, 3, 5);");
    expect(compiled.wgslProgram.bindings.map((binding) => binding.name)).not.toContain("coeffs");
    expect([...result.buffers.x as Float32Array]).toEqual([2, 4, 6]);
    expect([...result.buffers.out as Int32Array]).toEqual([2, 3, 5]);
  });

  it("lowers CUDA texture references and tex2D reads", () => {
    const compiled = compileCudaLiteKernel(`
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
__global__ void sample(float *out, int width) {
  int x = threadIdx.x;
  if (x < width) {
    out[x] = tex2D(texRef, (float)x + 0.5f, 0.5f);
  }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(2) },
        textures: { texRef: { width: 2, height: 1, data: new Float32Array([4, 8]) } },
        scalars: { width: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.ir.textures.map((texture) => texture.name)).toEqual(["texRef"]);
    expect(compiled.wgsl).toContain("var texRef: texture_2d<f32>;");
    expect(compiled.wgsl).toContain("bg_tex2d_texRef");
    expect(compiled.wgslProgram.bindings).toContainEqual(expect.objectContaining({
      kind: "texture2d",
      name: "texRef",
    }));
    expect([...result.buffers.out as Float32Array]).toEqual([4, 8]);
  });

  it("lowers CUDA texture object params and templated tex2D reads", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void sample(float *out, int width, cudaTextureObject_t tex) {
  int x = threadIdx.x;
  if (x < width) {
    out[x] = tex2D<float>(tex, (float)x + 0.5f, 0.5f);
  }
}`, { workgroupSize: [3, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(3) },
        textures: { tex: { width: 3, height: 1, data: new Float32Array([2, 4, 6]) } },
        scalars: { width: 3 },
      },
      { gridDim: [1, 1, 1], blockDim: [3, 1, 1] },
    );

    expect(compiled.ir.params.find((param) => param.name === "tex")?.valueType).toBe("texture2d");
    expect(compiled.wgsl).toContain("var tex: texture_2d<f32>;");
    expect(compiled.wgsl).toContain("bg_tex2d_tex");
    expect(compiled.wgslProgram.bindings).toContainEqual(expect.objectContaining({
      kind: "texture2d",
      name: "tex",
    }));
    expect([...result.buffers.out as Float32Array]).toEqual([2, 4, 6]);
  });

  it("formats diagnostics with source snippets", () => {
    const analysis = analyzeCudaLite(parseCudaLite(`
__global__ void bad(const float* x) {
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`));
    const formatted = formatCudaLiteDiagnostics(
      `
__global__ void bad(const float* x) {
  if (threadIdx.x < 1) { x[0] = 1.0; }
}`,
      analysis.diagnostics,
    );

    expect(formatted).toContain("ERROR const-pointer-write");
    expect(formatted).toContain("x[0] = 1.0");
    expect(formatted).toContain("^");
  });

  it("hardens reference inputs before execution", () => {
    const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });

    expect(() =>
      runCompiledKernelReference(
        compiled,
        {
          buffers: {
            x: new Float32Array([1, 2, 3, 4]),
            y: new Float32Array([10, 20, 30, 40]),
          },
          scalars: { a: 2 },
        },
        { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
      ),
    ).toThrow(/missing scalar input 'n'/);

    expect(() =>
      runCompiledKernelReference(
        compiled,
        {
          buffers: {
            x: new Int32Array([1, 2, 3, 4]),
            y: new Float32Array([10, 20, 30, 40]),
          },
          scalars: { a: 2, n: 4 },
        },
        { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
      ),
    ).toThrow(/buffer 'x' expects Float32Array/);
  });

  it("feature-gates half and subgroup intrinsics", () => {
    const halfSource = `
__global__ void halfy(half* x) {
  if (threadIdx.x < 1) { x[0] = x[0]; }
}`;
    expect(() => compileCudaLiteKernel(halfSource)).toThrow(CudaLiteCompilerError);
    const halfCompiled = compileCudaLiteKernel(halfSource, {
      features: { "shader-f16": true },
      workgroupSize: [1, 1, 1],
    });
    expect(halfCompiled.wgsl).toContain("enable f16;");
    expect(halfCompiled.wgslProgram.bindings[0]).toMatchObject({ valueType: "f16" });

    const halfScalar = compileCudaLiteKernel(`
__global__ void half_scale(half* x, half a) {
  if (threadIdx.x < 1) { x[0] = x[0] + a; }
}`, {
      features: { "shader-f16": true },
      workgroupSize: [1, 1, 1],
    });
    expect(halfScalar.wgsl).toContain("@align(4) a: f16");
    const halfResult = runCompiledKernelReference(
      halfScalar,
      { buffers: { x: createWgslFloat16Array([1]) }, scalars: { a: 2 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect(Array.from(halfResult.buffers.x as Iterable<number>)).toEqual([3]);
    expect(() =>
      runCompiledKernelReference(
        halfScalar,
        { buffers: { x: new Float32Array([1]) }, scalars: { a: 2 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      ),
    ).toThrow(/expects Float16Array/);

    const subgroupSource = `
__global__ void reduce(float* x) {
  if (threadIdx.x < 1) { x[0] = bg_subgroup_add(x[0]); }
}`;
    expect(() => compileCudaLiteKernel(subgroupSource)).toThrow(CudaLiteCompilerError);
  });

  it("derives compile feature options from kernel feature detection", () => {
    expect(
      cudaLiteFeatureOptionsFromKernelFeatures({
        shaderF16: false,
        subgroups: false,
        compatibilityMode: true,
        features: ["shader-f16", "subgroups"],
      }),
    ).toEqual({ "shader-f16": true, subgroups: true, compatibility: true });

    const detectedOptions = compileCudaLiteOptionsFromKernelFeatures(
      { shaderF16: true, subgroups: true, compatibilityMode: false, features: [] },
      { workgroupSize: [1, 1, 1] },
    );
    expect(detectedOptions).toEqual({
      workgroupSize: [1, 1, 1],
      features: { "shader-f16": true, subgroups: true },
    });

    const halfSource = `
__global__ void halfy(half* x) {
  if (threadIdx.x < 1) { x[0] = x[0]; }
}`;
    expect(compileCudaLiteKernel(halfSource, detectedOptions).wgsl).toContain("enable f16;");

    const overridden = compileCudaLiteOptionsFromKernelFeatures(
      { shaderF16: true, subgroups: true, compatibilityMode: false, features: [] },
      { features: { "shader-f16": false }, workgroupSize: [1, 1, 1] },
    );
    expect(overridden.features).toEqual({ "shader-f16": false, subgroups: true });
    expect(() => compileCudaLiteKernel(halfSource, overridden)).toThrow(CudaLiteCompilerError);
  });

  it("lowers CUDA half conversion builtins and exponent literals", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void half_convert(const half* input, half* output, int* flag) {
  int idx = threadIdx.x;
  if (idx < 1) {
    float value = __half2float(input[idx]);
    float big = 1e6f;
    if (big > 999999.0f) { atomicExch(flag, 1); }
    output[idx] = __float2half(value * 2.0f);
  }
}`, {
      features: { "shader-f16": true },
      workgroupSize: [1, 1, 1],
    });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          input: createWgslFloat16Array([1.5]),
          output: createWgslFloat16Array(1),
          flag: new Int32Array([0]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("f32(input[idx])");
    expect(compiled.wgsl).toContain("f16((value * 2.0))");
    expect(compiled.wgsl).toContain("1e6");
    expect(Array.from(result.buffers.output as Iterable<number>)).toEqual([3]);
    expect([...result.buffers.flag as Int32Array]).toEqual([1]);
  });

  it("lowers scalar CUDA half arithmetic and comparison intrinsics", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void half_ops(const __half* input, half* output, int* flags) {
  int idx = threadIdx.x;
  if (idx < 1) {
    half a = input[0];
    half b = input[1];
    half sum = __hadd(a, b);
    half diff = __hsub(sum, __float2half(0.5f));
    half prod = __hmul(diff, __float2half(2.0f));
    half quot = __hdiv(prod, __float2half(2.0f));
    half neg = __hneg(quot);
    half mixed = __hfma(neg, __float2half(-1.0f), __float2half(0.25f));
    half one = hexp(__float2half(0.0f));
    half capped = __hmax(__hmin(mixed, __float2half(3.0f)), one);
    output[0] = capped;
    if (__hgt(capped, __float2half(1.0f))) { flags[0] = 1; }
    if (__heq(__hsub(capped, __float2half(0.5f)), __float2half(1.0f))) { flags[1] = 1; }
    if (__hne(capped, __float2half(0.5f))) { flags[2] = 1; }
    if (__hge(capped, __float2half(1.5f))) { flags[3] = 1; }
    if (__hlt(__float2half(1.0f), capped)) { flags[4] = 1; }
    if (__hle(capped, __float2half(1.5f))) { flags[5] = 1; }
  }
}`, {
      features: { "shader-f16": true },
      workgroupSize: [1, 1, 1],
    });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          input: createWgslFloat16Array([1.5, 0.25]),
          output: createWgslFloat16Array(1),
          flags: new Int32Array(6),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fma(");
    expect(compiled.wgsl).toContain("f16(exp(f32(f16(0.0))))");
    expect(compiled.wgsl).toContain("min(mixed, f16(3.0))");
    expect(compiled.wgsl).toContain("max(");
    expect(Array.from(result.buffers.output as Iterable<number>)).toEqual([1.5]);
    expect([...result.buffers.flags as Int32Array]).toEqual([1, 1, 1, 1, 1, 1]);
  });

  it("emits atomic storage buffers with explicit load/store operations", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomic_read(int* x) {
  if (threadIdx.x < 1) {
    atomicAdd(&x[0], 1);
    x[1] = x[0];
  }
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("var<storage, read_write> x: array<atomic<i32>>;");
    expect(compiled.wgsl).toContain("atomicAdd(&x[0], 1);");
    expect(compiled.wgsl).toContain("atomicStore(&x[1], atomicLoad(&x[0]));");
  });

  it("supports CUDA float atomicAdd with a WGSL CAS loop", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomic_sum(const float* input, float* result) {
  int idx = threadIdx.x;
  if (idx < 2) { atomicAdd(&result[0], input[idx]); }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          input: new Float32Array([1.5, 2.25]),
          result: new Float32Array([10]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<storage, read_write> result: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("fn bg_atomicAdd_f32");
    expect(compiled.wgsl).toContain("bitcast<f32>(old_bits)");
    expect([...result.buffers.result as Float32Array]).toEqual([13.75]);
  });

  it("supports CUDA float atomicExch through u32 bitcasts", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomic_exchange(float* x, float* out) {
  if (threadIdx.x < 1) { out[0] = atomicExch(&x[0], 7.5f); }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([2.5]),
          out: new Float32Array(1),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bitcast<f32>(atomicExchange(&x[0], bitcast<u32>(7.5)))");
    expect([...result.buffers.x as Float32Array]).toEqual([7.5]);
    expect([...result.buffers.out as Float32Array]).toEqual([2.5]);
  });

  it("supports CUDA pointer-form atomicAdd on integer buffers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomic_count(int* x) {
  if (threadIdx.x < 1) {
    atomicAdd_system(x, 1);
    atomicExch_system(x, 42);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Int32Array([41]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("atomicAdd(&x[0], 1);");
    expect(compiled.wgsl).toContain("atomicExchange(&x[0], 42);");
    expect([...result.buffers.x as Int32Array]).toEqual([42]);
  });

  it("supports CUDA integer atomic exchange and compare-swap", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomic_more(int* x, int* out) {
  if (threadIdx.x == 0) {
    out[0] = atomicExch(&x[0], 7);
    out[1] = atomicCAS(&x[0], 7, 9);
    out[2] = atomicMax(&x[1], 5);
    out[3] = atomicMin(&x[1], 3);
    out[4] = atomicSub(&x[1], 1);
    out[5] = atomicAnd(&x[2], 0x6);
    out[6] = atomicOr(&x[2], 0x8);
    out[7] = atomicXor(&x[2], 0x3);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Int32Array([2, 4, 7]),
          out: new Int32Array(8),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.x as Int32Array]).toEqual([9, 2, 13]);
    expect([...result.buffers.out as Int32Array]).toEqual([2, 7, 4, 5, 3, 7, 6, 14]);
    expect(compiled.wgsl).toContain("atomicExchange(&x[0], 7)");
    expect(compiled.wgsl).toContain("atomicCompareExchangeWeak(&x[0], 7, 9).old_value");
    expect(compiled.wgsl).toContain("atomicMax(&x[1], 5)");
    expect(compiled.wgsl).toContain("atomicMin(&x[1], 3)");
    expect(compiled.wgsl).toContain("atomicSub(&x[1], 1)");
    expect(compiled.wgsl).toContain("atomicAnd(&x[2], 0x6)");
    expect(compiled.wgsl).toContain("atomicOr(&x[2], 0x8)");
    expect(compiled.wgsl).toContain("atomicXor(&x[2], 0x3)");
  });

  it("supports CUDA atomic inc/dec and atomics through pointer aliases", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void alias_atomic(float* scratch, const float* values, uint* out) {
  if (threadIdx.x == 0) {
    float* accum = scratch;
    uint* flag = (uint*)(scratch + 2);
    out[0] = atomicInc(flag, 2);
    out[1] = atomicDec(flag, 2);
    atomicAdd(&accum[0], values[0]);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          scratch: new Float32Array([10, 0, 1]),
          values: new Float32Array([1.5]),
          out: new Uint32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.scratch as Float32Array]).toEqual([11.5, 0, 1]);
    expect([...result.buffers.out as Uint32Array]).toEqual([1, 2]);
    expect(compiled.wgsl).toContain("fn bg_atomicInc_storage_u32");
    expect(compiled.wgsl).toContain("fn bg_atomicDec_storage_u32");
    expect(compiled.wgsl).toContain("bg_atomicInc_storage_u32(&scratch[");
    expect(compiled.wgsl).toContain("bg_atomicAdd_f32(&scratch[");
  });

  it("supports CUDA atomic inc/dec on shared integer memory", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void shared_counter(uint* out) {
  __shared__ uint counter[1];
  if (threadIdx.x == 0) {
    counter[0] = 1;
    out[0] = atomicInc(&counter[0], 1);
    out[1] = atomicDec(&counter[0], 1);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Uint32Array]).toEqual([1, 0]);
    expect(compiled.wgsl).toContain("var<workgroup> counter: array<atomic<u32>, 1>;");
    expect(compiled.wgsl).toContain("bg_atomicInc_workgroup_u32(&counter[0], u32(1))");
    expect(compiled.wgsl).toContain("bg_atomicDec_workgroup_u32(&counter[0], u32(1))");
  });

  it("supports CUDA div_ceil and shared address conversion helpers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void address_math(uint* out, int n) {
  __shared__ float smem[8];
  float regs[2][2];
  float* tile = &smem[4];
  if (threadIdx.x == 0) {
    fill_2D_regs<float, 2, 2>(regs, 3.0f);
    out[0] = uint(div_ceil(n, 4));
    out[1] = __cvta_generic_to_shared(tile);
    out[2] = uint(regs[1][1]);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(3) }, scalars: { n: 17 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Uint32Array]).toEqual([5, 4, 3]);
    expect(compiled.wgsl).toContain("(((params.n + 4) - 1) / 4)");
    expect(compiled.wgsl).toContain("out[1] = u32((0u + u32(4)))");
    expect(compiled.wgsl).toContain("regs[fill_regs_0][fill_regs_1] = 3.0;");
  });

  it("supports mutable CUDA pointer rebasing", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void pointer_rebase(uint* x, uint* out, int offset) {
  x += offset;
  if (threadIdx.x == 0) {
    out[0] = x[0];
    x -= 1;
    out[1] = x[0];
    x++;
    out[2] = *x;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Uint32Array([10, 20, 30, 40]), out: new Uint32Array(3) }, scalars: { offset: 2 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Uint32Array]).toEqual([30, 20, 30]);
    expect(compiled.wgsl).toContain("var bg_x_base: u32 = 0u;");
    expect(compiled.wgsl).toContain("bg_x_base = (bg_x_base + u32(params.offset));");
    expect(compiled.wgsl).toContain("x[(bg_x_base + u32(0))]");
  });
});
