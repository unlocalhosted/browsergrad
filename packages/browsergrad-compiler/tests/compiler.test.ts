import { describe, expect, it } from "vitest";
import {
  CudaLiteCompilerError,
  analyzeCudaLite,
  compileCudaLiteKernel,
  createCudaGridSyncPhasePlan,
  createCudaHostDynamicLaunchPlan,
  createCudaLaunchValidationDiagnostics,
  createCudaLoweringPlan,
  createCudaPeerCopyPlan,
  createCudaRuntimePlan,
  createCudaWebGpuExecutionPlan,
  describeCudaDiagnostic,
  formatCudaLiteDiagnostics,
  normalizeCudaWebGpuReadbackNames,
  parseCudaLite,
  runCompiledKernelReference,
  runCompiledKernelWebGpu,
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

    expect(compiled.wgsl).toContain("var bytes: u32 = 4");
    expect([...result.buffers.out as Uint32Array]).toEqual([4]);
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
  float* y = &x[0];
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
  if (threadIdx.x < 1) { x[0] = sinf(x[0]); }
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
  {
    x[0] = 1.0;
  }
}`)).toThrow(/standalone blocks/);

    expect(() => parseCudaLite(`
__global__ void bad(float* x) {
  if (threadIdx.x < 1) { x[0] = 1.2.3; }
}`)).toThrow(/invalid numeric literal/);
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
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          scratch: new Float32Array(2),
          out: new Float32Array(1),
        },
      },
      { gridDim: [2, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics).toContainEqual(expect.objectContaining({
      code: "unsupported-cooperative-groups",
      severity: "warning",
    }));
    expect([...result.buffers.out as Float32Array]).toEqual([3]);
    expect(createCudaGridSyncPhasePlan(compiled.ir).supported).toBe(true);
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
      "peer-copy",
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
    expect(peerPlan).toMatchObject({ supported: true, kind: "host-peer-copy" });
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
      expect(dynamicPlan.steps).toHaveLength(2);
      expect(dynamicPlan.steps[1]?.storageAliases).toEqual({ dst: "x" });
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
    expect(createCudaWebGpuExecutionPlan(compiled, input, badGrid)).toMatchObject({
      supported: false,
      blockers: [{
        kind: "launch",
        code: "launch-grid-dim-invalid",
      }],
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

  it("allows loop-local variable names to be reused in independent loop scopes", () => {
    const analysis = analyzeCudaLite(parseCudaLite(`
__global__ void scopedLoops(float *x) {
  for (int s = 1; s > 0; s >>= 1) { x[0] += 1.0f; }
  for (int s = 1; s > 0; s >>= 1) { x[0] += 1.0f; }
}`));

    expect(analysis.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("duplicate-symbol");
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
    surf2Dwrite(value * 2.0f, outputSurf, x * sizeof(float), y);
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
    expect(executionPlan.supported && executionPlan.steps).toHaveLength(4);
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
    expect(executionPlan.supported && executionPlan.steps).toHaveLength(4);

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

  it("lowers local shared-memory pointer aliases as fixed shared offsets", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void splitShared(float *x) {
  extern __shared__ float sdataA[];
  float* sdataB = &sdataA[2];
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
    expect(compiled.wgsl).toContain("sdataA[(2) + (tid)]");
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
      { buffers: { x: new Float16Array([1]) }, scalars: { a: 2 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect([...halfResult.buffers.x as Float16Array]).toEqual([3]);
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
          input: new Float16Array([1.5]),
          output: new Float16Array(1),
          flag: new Int32Array([0]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("f32(input[idx])");
    expect(compiled.wgsl).toContain("f16((value * 2.0))");
    expect(compiled.wgsl).toContain("1e6");
    expect([...result.buffers.output as Float16Array]).toEqual([3]);
    expect([...result.buffers.flag as Int32Array]).toEqual([1]);
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
  if (threadIdx.x < 1) { atomicAdd(x, 1); }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Int32Array([41]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("atomicAdd(&x[0], 1);");
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
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Int32Array([2, 4]),
          out: new Int32Array(5),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.x as Int32Array]).toEqual([9, 2]);
    expect([...result.buffers.out as Int32Array]).toEqual([2, 7, 4, 5, 3]);
    expect(compiled.wgsl).toContain("atomicExchange(&x[0], 7)");
    expect(compiled.wgsl).toContain("atomicCompareExchangeWeak(&x[0], 7, 9).old_value");
    expect(compiled.wgsl).toContain("atomicMax(&x[1], 5)");
    expect(compiled.wgsl).toContain("atomicMin(&x[1], 3)");
    expect(compiled.wgsl).toContain("atomicSub(&x[1], 1)");
  });
});
