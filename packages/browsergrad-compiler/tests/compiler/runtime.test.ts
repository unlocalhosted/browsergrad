import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { createWgslFloat16Array } from "@unlocalhosted/browsergrad-kernels";
import {
  type CompiledCudaLiteKernel,
  CudaLiteCompilerError,
  analyzeCudaLite,
  compileCudaLiteOptionsFromKernelFeatures,
  createCudaLiteCompileCacheKey,
  createCudaLiteCompilerCache,
  compileCudaLiteKernelForWebGpu,
  compileCudaLiteKernel,
  emitSemanticKernelIrWgsl,
  prepareCompiledKernelWebGpu,
  canEmitSemanticKernelIrWgsl,
  canRunCompiledKernelSemanticReference,
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
  getCudaFeatureRegistry,
  normalizeCudaWebGpuReadbackNames,
  parseCudaLite,
  projectSemanticHostRuntimeToGpuIr,
  runCompiledKernelReference,
  runCompiledKernelSemanticReference,
  runCompiledKernelWebGpu,
  summarizeCudaWebGpuExecutionPlan,
  validateCudaKernelLaunch,
} from "../../src/index";
import {
  constantBufferInputs,
  cudaWebGpuDefaultReadbackNames,
  deviceGlobalBufferInputs,
} from "../../src/webgpu_inputs";
import { deviceLaunchTreeIsExternallySilent } from "../../src/runtime_elision";
import { packCudaWebGpuUniformParams } from "../../src/webgpu_orchestration";
import { collectSemanticPoolAllocations } from "../../src/semantic_ir";

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

function semanticIr(compiled: CompiledCudaLiteKernel) {
  return compiled.kernelIr;
}

const gammaCoefficients = [
  0.9999999999998099,
  676.5203681218851,
  -1259.1392167224028,
  771.3234287776531,
  -176.6150291621406,
  12.507343278686905,
  -0.13857109526572012,
  9.984369578019572e-6,
  1.5056327351493116e-7,
] as const;

function gammaApprox(value: number): number {
  if (Number.isNaN(value)) return NaN;
  if (value === Infinity) return Infinity;
  if (value === -Infinity) return NaN;
  if (value <= 0 && Number.isInteger(value)) return NaN;
  if (value < 0.5) return Math.PI / (Math.sin(Math.PI * value) * gammaApprox(1 - value));
  const z = value - 1;
  let x = gammaCoefficients[0];
  for (let i = 1; i < gammaCoefficients.length; i++) x += gammaCoefficients[i]! / (z + i);
  const t = z + 7.5;
  return Math.sqrt(2 * Math.PI) * (t ** (z + 0.5)) * Math.exp(-t) * x;
}

function erfApprox(value: number): number {
  if (Number.isNaN(value)) return NaN;
  if (!Number.isFinite(value)) return Math.sign(value);
  const sign = value < 0 ? -1 : 1;
  const absValue = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * absValue);
  const polynomial = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return sign * (1 - polynomial * Math.exp(-absValue * absValue));
}

function roundEven(value: number): number {
  const lower = Math.floor(value);
  const diff = value - lower;
  if (diff < 0.5) return lower;
  if (diff > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

function roundAway(value: number): number {
  if (!Number.isFinite(value) || value === 0) return value;
  return value < 0 ? Math.ceil(value - 0.5) : Math.floor(value + 0.5);
}

const nextafterF32Buffer = new ArrayBuffer(4);
const nextafterF32 = new Float32Array(nextafterF32Buffer);
const nextafterU32 = new Uint32Array(nextafterF32Buffer);

function floatToBits(value: number): number {
  nextafterF32[0] = value;
  return nextafterU32[0] ?? 0;
}

function bitsToFloat(bits: number): number {
  nextafterU32[0] = bits >>> 0;
  return nextafterF32[0] ?? 0;
}

function nextafterApprox(x: number, y: number): number {
  if (Number.isNaN(x) || Number.isNaN(y)) return NaN;
  if (Object.is(x, y) || x === y) return y;
  if (x === 0) return bitsToFloat((y < 0 || Object.is(y, -0)) ? 0x80000001 : 0x00000001);
  const bits = floatToBits(x);
  return x > 0
    ? bitsToFloat((x < y ? bits + 1 : bits - 1) >>> 0)
    : bitsToFloat((x < y ? bits - 1 : bits + 1) >>> 0);
}

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

function expectParseDiagnosticCode(source: string, code: string): void {
  try {
    parseCudaLite(source);
  } catch (error) {
    if (error instanceof CudaLiteCompilerError) {
      expect(error.diagnostics.map((diagnostic) => diagnostic.code)).toContain(code);
      return;
    }
    throw error;
  }
  throw new Error(`Expected parse diagnostic '${code}'`);
}

function collectEmittedDiagnosticCodes(srcDir: string): ReadonlySet<string> {
  const out = new Set<string>();
  const patterns = [
    /error\(\s*"([^"]+)"/gsu,
    /warning\(\s*"([^"]+)"/gsu,
    /featureError\(\s*"([^"]+)"/gsu,
    /code:\s*"([^"]+)"/gu,
    /markUnsafe\(\s*"([^"]+)"/gsu,
    /webGpuBlocker\(\s*[^,]+,\s*"([^"]+)"/gsu,
    /\b(?:blocker|firstBlocker|hostDynamicPlan\.blocker|peerCopyRuntimePlan\.blocker)\?\.code\s*\?\?\s*"([^"]+)"/gu,
  ];
  const blockerTypePatterns = [
    /export type CudaHostDynamicLaunchBlockerCode\s*=([\s\S]*?);/gu,
    /export type CudaPeerCopyBlockerCode\s*=([\s\S]*?);/gu,
  ];
  for (const file of fs.readdirSync(srcDir)) {
    if (!file.endsWith(".ts")) continue;
    const source = fs.readFileSync(path.join(srcDir, file), "utf8");
    for (const pattern of patterns) {
      for (const match of source.matchAll(pattern)) {
        const code = match[1];
        if (code) out.add(code);
      }
    }
    for (const pattern of blockerTypePatterns) {
      for (const match of source.matchAll(pattern)) {
        const block = match[1] ?? "";
        for (const codeMatch of block.matchAll(/"([^"]+)"/gu)) {
          const code = codeMatch[1];
          if (code) out.add(code);
        }
      }
    }
  }
  return out;
}

function compilerSourceText(file: string): string {
  return fs.readFileSync(path.join(packageRoot, "src", file), "utf8");
}

function compilerExampleText(file: string): string {
  return fs.readFileSync(path.join(packageRoot, "examples", file), "utf8");
}

describe("CUDA-lite compiler: Runtime orchestration", () => {
  it("omits unreachable DevicePool parameters and lowers used pools through semantic execution", () => {
      const unused = compileCudaLiteKernel(`
  __global__ void unusedPool(DevicePool *pool, uint *out) {
    if (threadIdx.x < 1) out[0] = 7u;
  }`, { workgroupSize: [1, 1, 1] });
      const used = compileCudaLiteKernel(`
  __global__ void usedPool(DevicePool *pool, uint *out) {
    if (threadIdx.x < 1) {
      uint *value = (uint *)streamOrderedAllocate(pool, sizeof(uint));
      value[0] = 7u;
      out[0] = value[0];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        unused,
        { buffers: { out: new Uint32Array(1) }, memoryPools: { pool: { data: new Uint32Array(1), offset: new Uint32Array([0]) } } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canEmitSemanticKernelIrWgsl(unused.wgslLegalizedKernelIr)).toBe(true);
      expect(canRunCompiledKernelSemanticReference(unused)).toBe(true);
      expect(unused.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(unused.wgsl).not.toContain("pool_pool");
      expect([...result.buffers.out as Uint32Array]).toEqual([7]);
      expect(canEmitSemanticKernelIrWgsl(used.wgslLegalizedKernelIr)).toBe(true);
      expect(used.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(used.wgsl).toContain("pool_pool");
    });

  it("allocates from a DevicePool and writes through casted pool pointers", () => {
      const compiled = compileCudaLiteKernel(DEVICE_POOL_ALLOC, { workgroupSize: [2, 1, 1] });
      const allocations = collectSemanticPoolAllocations(compiled.kernelIr.operations);
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: { out: new Float32Array(2) },
          memoryPools: { dp: { data: new Uint32Array(2), offset: new Uint32Array([0]) } },
          scalars: { N: 2 },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(allocations).toMatchObject([{
        kind: "pool-allocate",
        target: { name: "ptr", pointerRuntimeState: true },
        pool: { kind: "device-pool", name: "dp" },
      }]);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("var<storage, read_write> dp_pool: array<u32>;");
      expect(compiled.wgsl).toContain("fn bg_pool_alloc_dp(size_bytes: u32) -> u32");
      expect([...result.buffers.out as Float32Array]).toEqual([3.25, 3.25]);
      expect([...result.buffers.dp as Uint32Array]).toEqual([floatBits(3.25), floatBits(3.25)]);
    });

  it("allocates from an external DevicePool reference", () => {
      const compiled = compileCudaLiteKernel(EXTERNAL_POOL_ALLOC, { workgroupSize: [1, 1, 1] });
      const allocations = collectSemanticPoolAllocations(compiled.kernelIr.operations);
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: { out: new Float32Array(1) },
          memoryPools: { g_pool: { data: new Uint32Array(1), offset: new Uint32Array([0]) } },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(allocations).toMatchObject([{
        kind: "pool-allocate",
        pool: { kind: "device-pool", name: "g_pool" },
      }]);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("var<storage, read_write> g_pool_pool: array<u32>;");
      expect(compiled.wgsl).toContain("fn bg_pool_alloc_g_pool(size_bytes: u32) -> u32");
      expect([...result.buffers.out as Float32Array]).toEqual([5.5]);
      expect([...result.buffers.g_pool as Uint32Array]).toEqual([floatBits(5.5)]);
    });

  it("keeps modeled CUDA runtime calls wired through compiler stage guards", () => {
      const modeledRuntimeCalls = [
        "cudaDeviceSynchronize",
        "cudaCtxResetPersistingL2Cache",
        "cudaDeviceReset",
        "cudaThreadExit",
        "cudaThreadSynchronize",
        "cudaDeviceGetAttribute",
        "cudaDeviceGetLimit",
        "cudaDeviceSetLimit",
        "cudaThreadGetLimit",
        "cudaThreadSetLimit",
        "cudaDeviceCanAccessPeer",
        "cudaDeviceEnablePeerAccess",
        "cudaDeviceDisablePeerAccess",
        "cudaGetDeviceFlags",
        "cudaSetDeviceFlags",
        "cudaMemGetInfo",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "cudaOccupancyMaxPotentialBlockSize",
        "cudaOccupancyMaxPotentialBlockSizeWithFlags",
        "cudaOccupancyAvailableDynamicSMemPerBlock",
        "cudaDeviceGetCacheConfig",
        "cudaDeviceSetCacheConfig",
        "cudaDeviceGetSharedMemConfig",
        "cudaDeviceSetSharedMemConfig",
        "cudaDeviceGetStreamPriorityRange",
        "cudaThreadGetCacheConfig",
        "cudaThreadSetCacheConfig",
        "cudaThreadExchangeStreamCaptureMode",
        "cudaFree",
        "cudaFreeAsync",
        "cudaMemAdvise",
        "cudaMemPrefetchAsync",
        "cudaStreamAttachMemAsync",
        "cudaStreamCreate",
        "cudaStreamCreateWithFlags",
        "cudaStreamCreateWithPriority",
        "cudaStreamDestroy",
        "cudaStreamGetDevice",
        "cudaStreamGetFlags",
        "cudaStreamGetId",
        "cudaStreamGetPriority",
        "cudaStreamIsCapturing",
        "cudaStreamGetCaptureInfo",
        "cudaStreamGetCaptureInfo_v2",
        "cudaStreamBeginCapture",
        "cudaStreamEndCapture",
        "cudaStreamUpdateCaptureDependencies",
        "cudaGraphCreate",
        "cudaGraphInstantiate",
        "cudaGraphInstantiateWithFlags",
        "cudaGraphUpload",
        "cudaGraphExecUpdate",
        "cudaGraphDestroy",
        "cudaGraphExecDestroy",
        "cudaStreamQuery",
        "cudaStreamSynchronize",
        "cudaStreamWaitEvent",
        "cudaSetDevice",
        "cudaGetDevice",
        "cudaGetDeviceCount",
        "cudaRuntimeGetVersion",
        "cudaDriverGetVersion",
        "cudaProfilerStart",
        "cudaProfilerStop",
        "cudaFuncSetAttribute",
        "cudaFuncSetCacheConfig",
        "cudaFuncSetSharedMemConfig",
        "cudaGetLastError",
        "cudaPeekAtLastError",
        "cudaEventCreate",
        "cudaEventCreateWithFlags",
        "cudaEventDestroy",
        "cudaEventQuery",
        "cudaEventRecord",
        "cudaEventRecordWithFlags",
        "cudaEventSynchronize",
      ];
      const registrySource = compilerSourceText("cuda_runtime_noops.ts");
      const missing = modeledRuntimeCalls
        .filter((call) => !registrySource.includes(`"${call}"`))
        .map((call) => `cuda_runtime_noops.ts:${call}`);

      const stageConsumers = [
        "analyzer.ts",
        "semantic_ir.ts",
        "runtime_plan.ts",
      ];
      const missingConsumers = stageConsumers
        .filter((file) => !compilerSourceText(file).includes("isHostManagedRuntimeNoopCall"))
        .map((file) => `${file}:isHostManagedRuntimeNoopCall`);

      expect(missing).toEqual([]);
      expect(missingConsumers).toEqual([]);
    });

  it("keeps CUDA runtime query-write calls wired through side-effect guards", () => {
      const runtimeQueryWriteCalls = [
        "cudaGetDevice",
        "cudaGetDeviceCount",
        "cudaDeviceGetAttribute",
        "cudaDeviceGetLimit",
        "cudaThreadGetLimit",
        "cudaDeviceCanAccessPeer",
        "cudaGetDeviceFlags",
        "cudaMemGetInfo",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "cudaOccupancyMaxPotentialBlockSize",
        "cudaOccupancyMaxPotentialBlockSizeWithFlags",
        "cudaOccupancyAvailableDynamicSMemPerBlock",
        "cudaDeviceGetCacheConfig",
        "cudaDeviceGetSharedMemConfig",
        "cudaThreadGetCacheConfig",
        "cudaThreadExchangeStreamCaptureMode",
        "cudaDeviceGetStreamPriorityRange",
        "cudaStreamCreate",
        "cudaStreamCreateWithFlags",
        "cudaStreamCreateWithPriority",
        "cudaStreamGetDevice",
        "cudaStreamGetFlags",
        "cudaStreamGetId",
        "cudaStreamGetPriority",
        "cudaStreamIsCapturing",
        "cudaStreamGetCaptureInfo",
        "cudaStreamGetCaptureInfo_v2",
        "cudaStreamEndCapture",
        "cudaGraphCreate",
        "cudaGraphInstantiate",
        "cudaGraphInstantiateWithFlags",
        "cudaGraphExecUpdate",
        "cudaEventCreate",
        "cudaEventCreateWithFlags",
        "cudaRuntimeGetVersion",
        "cudaDriverGetVersion",
        "cudaEventElapsedTime",
      ];
      const registrySource = compilerSourceText("cuda_runtime_queries.ts");
      const missing = runtimeQueryWriteCalls
        .filter((call) => !registrySource.includes(`"${call}"`))
        .map((call) => `cuda_runtime_queries.ts:${call}`);

      const consumerChecks = [
        ["analyzer.ts", "isCudaIntegerRuntimeQueryCall"],
        ["dynamic_launch.ts", "isCudaHostDynamicNoopCall"],
        ["peer_copy.ts", "isCudaPeerCopyHostNoopCall"],
        ["cuda_host_silent_calls.ts", "isCudaRuntimeQueryWriteCall"],
        ["webgpu_orchestration.ts", "isCudaRuntimeQueryWriteCall"],
      ] as const;
      const missingConsumers = consumerChecks
        .filter(([file, helper]) => !compilerSourceText(file).includes(helper))
        .map(([file, helper]) => `${file}:${helper}`);

      expect(missing).toEqual([]);
      expect(missingConsumers).toEqual([]);
    });

  it("rejects semantic gaps before WGSL/runtime execution", () => {
      const unknownSymbol = analyzeCudaLite(parseCudaLite(`
  __global__ void bad(float* x) {
    if (threadIdx.x < 1) { x[0] = missing + 1.0; }
  }`));
      expect(unknownSymbol.diagnostics.map((diagnostic) => diagnostic.code)).toContain("unknown-symbol");

      const unknownWithHint = analyzeCudaLite(parseCudaLite(`
  __global__ void bad(float* x) {
    __shared__ float reduce_smem[1];
    if (threadIdx.x < 1) { x[0] = block_smem[0]; }
  }`));
      expect(unknownWithHint.diagnostics).toContainEqual(expect.objectContaining({
        code: "unknown-symbol",
        message: expect.stringContaining("nearest visible symbol 'reduce_smem'"),
      }));

      const unsupportedCall = analyzeCudaLite(parseCudaLite(`
  __global__ void bad(float* x) {
    if (threadIdx.x < 1) { x[0] = unsupported_math_island(x[0]); }
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
      expect(scalarParamWrite.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("parameter-assignment");

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

  it("runs cudaMemcpyPeer and cudaMemcpyPeerAsync in CPU reference when explicitly enabled", async () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void peerCopy(float *dst, const float *src, int n) {
    if (threadIdx.x == 0) {
      cudaMemcpyPeer(dst, 1, src, 0, sizeof(float));
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
      expect([...result.buffers.dst as Float32Array]).toEqual([2.5, 2.5, 3.5, 0]);
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
      expect(createCudaPeerCopyPlan(
        { ...compiled },
        {
          buffers: {
            dst: new Float32Array(4),
            src: new Float32Array([2.5, 3.5]),
          },
          scalars: { n: 2 },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      ).supported).toBe(true);
      const semanticOnlyPlan = createCudaPeerCopyPlan(
        {
          ...compiled,
          ast: { ...compiled.ast, functions: [], kernels: [] },
          analysis: {
            ...compiled.analysis,
            kernel: { ...compiled.analysis.kernel, body: [] },
            deviceGlobals: [],
          },
        },
        {
          buffers: {
            dst: new Float32Array(4),
            src: new Float32Array([2.5, 3.5]),
          },
          scalars: { n: 2 },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect(semanticOnlyPlan.supported).toBe(true);
      expect(semanticOnlyPlan.copies).toHaveLength(2);
      expect(plan.copies[0]).toMatchObject({
        dstRoot: "dst",
        dstOffset: 0,
        srcRoot: "src",
        srcOffset: 0,
        elementCount: 1,
        valueType: "float",
      });
      expect(plan.copies[1]).toMatchObject({
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
      expect(residentPlan.copies[0]).toMatchObject({ dstRoot: "dst", srcRoot: "src", elementCount: 1 });
      expect(residentPlan.copies[1]).toMatchObject({ dstRoot: "dst", srcRoot: "src", elementCount: 2 });

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
      cudaMemset(dst, 0, sizeof(float) * 4);
      cudaMemcpyPeer(dst, 1, src, 0, sizeof(float));
      cudaMemcpy(dst + 1, src, sizeof(float) * n, cudaMemcpyDeviceToDevice);
      cudaMemcpyAsync(dst + 3, src + 1, sizeof(float), cudaMemcpyDefault, stream);
      cudaMemcpy(dst + 2, src, sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpyAsync(dst + 3, src, sizeof(float), cudaMemcpyDeviceToHost, stream);
      cudaMemcpy(dst + 1, src + 1, sizeof(float), cudaMemcpyHostToHost);
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
          dst: new Float32Array([9, 9, 9, 9]),
          src: new Float32Array([2.5, 3.5]),
        },
        scalars: { n: 2 },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const plan = createCudaRuntimeCopyPlan(compiled, input, launch);
      const runtimePlan = createCudaRuntimePlan(compiled);
      const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);
      const gpuIr = projectSemanticHostRuntimeToGpuIr(compiled.kernelIr);

      expect([...result.buffers.dst as Float32Array]).toEqual([2.5, 3.5, 2.5, 2.5]);
      expect(compiled.wgsl).toBeUndefined();
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(false);
      expect(canEmitSemanticKernelIrWgsl(gpuIr)).toBe(true);
      expect(emitSemanticKernelIrWgsl(gpuIr).wgsl).not.toContain("cudaMemcpy");
      expect(runtimePlan.operations.map((operation) => operation.kind)).toEqual([
        "runtime-copy",
        "runtime-copy",
        "runtime-copy",
        "runtime-copy",
        "runtime-copy",
        "runtime-copy",
        "runtime-copy",
      ]);
      expect(plan.supported).toBe(true);
      expect(plan.copies.map((copy) => copy.kind === "copy"
        ? { kind: copy.kind, dstOffset: copy.dstOffset, srcOffset: copy.srcOffset, elementCount: copy.elementCount }
        : copy.kind === "fill"
          ? { kind: copy.kind, dstOffset: copy.dstOffset, srcOffset: undefined, elementCount: copy.elementCount }
          : { kind: copy.kind, dstOffset: undefined, srcOffset: undefined, elementCount: undefined })).toEqual([
        { kind: "fill", dstOffset: 0, srcOffset: undefined, elementCount: 4 },
        { kind: "copy", dstOffset: 0, srcOffset: 0, elementCount: 1 },
        { kind: "copy", dstOffset: 1, srcOffset: 0, elementCount: 2 },
        { kind: "copy", dstOffset: 3, srcOffset: 1, elementCount: 1 },
        { kind: "copy", dstOffset: 2, srcOffset: 0, elementCount: 1 },
        { kind: "copy", dstOffset: 3, srcOffset: 0, elementCount: 1 },
        { kind: "copy", dstOffset: 1, srcOffset: 1, elementCount: 1 },
      ]);
      expect(webGpuPlan.supported).toBe(true);
      if (webGpuPlan.supported) {
        expect(webGpuPlan.kind).toBe("host-copy");
        expect(webGpuPlan.steps).toHaveLength(8);
      }

      expect(() => compileCudaLiteKernel(`
  __global__ void unsupportedKind(float *dst, const float *src) {
    if (threadIdx.x == 0) {
      cudaMemcpy(dst, src, sizeof(float), 99);
    }
  }`, {
        referenceCudaRuntime: true,
        workgroupSize: [1, 1, 1],
      })).toThrow("unsupported-cuda-runtime-copy-kind");
    });

  it("host-lifts nonzero cudaMemset byte patterns for typed storage buffers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void runtimeFill(unsigned int *bits) {
    cudaStream_t stream;
    if (threadIdx.x == 0) {
      cudaStreamCreate(&stream);
      cudaMemsetAsync(bits + 1, 255, sizeof(unsigned int) * 2, stream);
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }`, {
        referenceCudaRuntime: true,
        workgroupSize: [1, 1, 1],
      });
      const input = {
        buffers: {
          bits: new Uint32Array([0, 0, 0, 0]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const plan = createCudaRuntimeCopyPlan(compiled, input, launch);
      const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);

      expect([...result.buffers.bits as Uint32Array]).toEqual([0, 0xffffffff, 0xffffffff, 0]);
      expect(plan.supported).toBe(true);
      expect(plan.copies).toHaveLength(1);
      expect(plan.copies[0]).toMatchObject({
        kind: "fill",
        dstRoot: "bits",
        dstOffset: 1,
        elementCount: 2,
        valueType: "uint",
        byteValue: 255,
      });
      expect(webGpuPlan.supported).toBe(true);
      if (webGpuPlan.supported) {
        expect(webGpuPlan.kind).toBe("host-copy");
        expect(webGpuPlan.steps).toHaveLength(2);
      }
    });

  it("host-lifts byte-granular cudaMemset ranges for partial storage words", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void runtimeByteFill(unsigned int *bits) {
    cudaStream_t stream;
    if (threadIdx.x == 0) {
      cudaStreamCreate(&stream);
      cudaMemset(bits, 0, 3);
      cudaMemsetAsync(bits + 1, 0x7f, 5, stream);
      cudaStreamDestroy(stream);
    }
  }`, {
        referenceCudaRuntime: true,
        workgroupSize: [1, 1, 1],
      });
      const input = {
        buffers: {
          bits: new Uint32Array([0xffffffff, 0, 0]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const plan = createCudaRuntimeCopyPlan(compiled, input, launch);
      const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);

      expect([...result.buffers.bits as Uint32Array]).toEqual([0xff000000, 0x7f7f7f7f, 0x0000007f]);
      expect(plan.supported).toBe(true);
      expect(plan.copies.map((copy) => copy.kind === "fill-bytes"
        ? { kind: copy.kind, dstRoot: copy.dstRoot, dstByteOffset: copy.dstByteOffset, byteCount: copy.byteCount, byteValue: copy.byteValue }
        : { kind: copy.kind, dstRoot: copy.dstRoot, dstByteOffset: undefined, byteCount: undefined, byteValue: copy.kind === "fill" ? copy.byteValue : undefined })).toEqual([
        { kind: "fill-bytes", dstRoot: "bits", dstByteOffset: 0, byteCount: 3, byteValue: 0 },
        { kind: "fill-bytes", dstRoot: "bits", dstByteOffset: 4, byteCount: 5, byteValue: 0x7f },
      ]);
      expect(webGpuPlan.supported).toBe(true);
      if (webGpuPlan.supported) {
        expect(webGpuPlan.kind).toBe("host-copy");
        expect(webGpuPlan.steps).toHaveLength(3);
        expect(webGpuPlan.steps[1]?.program.name).toBe("bg_peer_fill_bytes");
        expect(webGpuPlan.steps[2]?.program.name).toBe("bg_peer_fill_bytes");
      }
    });

  it("host-lifts byte-wise cudaMemcpy across compatible 32-bit typed buffers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void runtimeCrossTypeCopy(float *dst, const unsigned int *src) {
    if (threadIdx.x == 0) {
      cudaMemcpy(dst + 1, src, sizeof(unsigned int) * 2, cudaMemcpyDeviceToDevice);
    }
  }`, {
        referenceCudaRuntime: true,
        workgroupSize: [1, 1, 1],
      });
      const input = {
        buffers: {
          dst: new Float32Array([0, 0, 0]),
          src: new Uint32Array([0x3f800000, 0x40200000]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const plan = createCudaRuntimeCopyPlan(compiled, input, launch);
      const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);

      expect([...result.buffers.dst as Float32Array]).toEqual([0, 1, 2.5]);
      expect(plan.supported).toBe(true);
      expect(plan.copies).toHaveLength(1);
      expect(plan.copies[0]).toMatchObject({
        kind: "copy",
        dstRoot: "dst",
        srcRoot: "src",
        dstOffset: 1,
        srcOffset: 0,
        elementCount: 2,
        valueType: "uint",
      });
      expect(webGpuPlan.supported).toBe(true);
      if (webGpuPlan.supported) {
        expect(webGpuPlan.kind).toBe("host-copy");
        expect(webGpuPlan.steps).toHaveLength(2);
      }
    });

  it("host-lifts byte-granular cudaMemcpy ranges for partial storage words", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void runtimeByteCopy(unsigned int *dst, const unsigned int *src) {
    cudaStream_t stream;
    if (threadIdx.x == 0) {
      cudaStreamCreate(&stream);
      cudaMemcpy(dst, src, 3, cudaMemcpyDeviceToDevice);
      cudaMemcpyAsync(dst + 1, src, 5, cudaMemcpyDefault, stream);
      cudaStreamDestroy(stream);
    }
  }`, {
        referenceCudaRuntime: true,
        workgroupSize: [1, 1, 1],
      });
      const input = {
        buffers: {
          dst: new Uint32Array([0, 0xffffffff, 0]),
          src: new Uint32Array([0xaabbccdd, 0x11223344]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const plan = createCudaRuntimeCopyPlan(compiled, input, launch);
      const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);

      expect([...result.buffers.dst as Uint32Array]).toEqual([0x00bbccdd, 0xaabbccdd, 0x00000044]);
      expect(plan.supported).toBe(true);
      expect(plan.copies.map((copy) => copy.kind === "copy-bytes"
        ? { kind: copy.kind, dstRoot: copy.dstRoot, srcRoot: copy.srcRoot, dstByteOffset: copy.dstByteOffset, srcByteOffset: copy.srcByteOffset, byteCount: copy.byteCount }
        : { kind: copy.kind, dstRoot: copy.dstRoot, srcRoot: copy.kind === "copy" ? copy.srcRoot : undefined, dstByteOffset: undefined, srcByteOffset: undefined, byteCount: undefined })).toEqual([
        { kind: "copy-bytes", dstRoot: "dst", srcRoot: "src", dstByteOffset: 0, srcByteOffset: 0, byteCount: 3 },
        { kind: "copy-bytes", dstRoot: "dst", srcRoot: "src", dstByteOffset: 4, srcByteOffset: 0, byteCount: 5 },
      ]);
      expect(webGpuPlan.supported).toBe(true);
      if (webGpuPlan.supported) {
        expect(webGpuPlan.kind).toBe("host-copy");
        expect(webGpuPlan.steps).toHaveLength(3);
        expect(webGpuPlan.steps[1]?.program.name).toBe("bg_peer_copy_bytes");
        expect(webGpuPlan.steps[2]?.program.name).toBe("bg_peer_copy_bytes");
      }
    });

  it("host-lifts cudaMemset2D and cudaMemset2DAsync row fills", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void runtimeFill2D(unsigned int *bits) {
    cudaStream_t stream;
    if (threadIdx.x == 0) {
      cudaMemset2D(bits + 1, sizeof(unsigned int) * 3, 255, sizeof(unsigned int) * 2, 2);
      cudaMemset2DAsync(bits + 2, sizeof(unsigned int) * 3, 0, sizeof(unsigned int), 2, stream);
    }
  }`, {
        referenceCudaRuntime: true,
        workgroupSize: [1, 1, 1],
      });
      const input = { buffers: { bits: new Uint32Array(8) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const plan = createCudaRuntimeCopyPlan(compiled, input, launch);
      const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);

      expect([...result.buffers.bits as Uint32Array]).toEqual([0, 0xffffffff, 0, 0, 0xffffffff, 0, 0, 0]);
      expect(plan.supported).toBe(true);
      expect(plan.copies.map((copy) => copy.kind === "fill"
        ? { kind: copy.kind, dstOffset: copy.dstOffset, elementCount: copy.elementCount, byteValue: copy.byteValue }
        : copy.kind === "copy"
          ? { kind: copy.kind, dstOffset: copy.dstOffset, elementCount: copy.elementCount, byteValue: undefined }
          : copy.kind === "fill-bytes"
            ? { kind: copy.kind, dstOffset: undefined, elementCount: undefined, byteValue: copy.byteValue }
            : { kind: copy.kind, dstOffset: undefined, elementCount: undefined, byteValue: undefined })).toEqual([
        { kind: "fill", dstOffset: 1, elementCount: 2, byteValue: 255 },
        { kind: "fill", dstOffset: 4, elementCount: 2, byteValue: 255 },
        { kind: "fill", dstOffset: 2, elementCount: 1, byteValue: 0 },
        { kind: "fill", dstOffset: 5, elementCount: 1, byteValue: 0 },
      ]);
      expect(webGpuPlan.supported).toBe(true);
      if (webGpuPlan.supported) {
        expect(webGpuPlan.kind).toBe("host-copy");
        expect(webGpuPlan.steps).toHaveLength(5);
      }
    });

  it("host-lifts cudaMemcpyToSymbol and cudaMemcpyFromSymbol", () => {
      const compiled = compileCudaLiteKernel(`
  __constant__ unsigned int cBits[4];
  __global__ void runtimeSymbolCopy(unsigned int *dst, const unsigned int *src) {
    cudaStream_t stream;
    if (threadIdx.x == 0) {
      cudaMemcpyToSymbol(cBits, src, sizeof(unsigned int) * 2);
      cudaMemcpyToSymbolAsync(cBits, src + 2, sizeof(unsigned int), sizeof(unsigned int) * 2, cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyFromSymbol(dst, cBits, sizeof(unsigned int) * 3);
      cudaMemcpyFromSymbolAsync(dst + 3, cBits, sizeof(unsigned int), sizeof(unsigned int) * 2, cudaMemcpyDeviceToDevice, stream);
    }
  }`, {
        referenceCudaRuntime: true,
        workgroupSize: [1, 1, 1],
      });
      const input = {
        buffers: {
          dst: new Uint32Array([0, 0, 0, 0]),
          src: new Uint32Array([11, 22, 33]),
        },
        constants: {
          cBits: new Uint32Array([0, 0, 0, 0]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const plan = createCudaRuntimeCopyPlan(compiled, input, launch);
      const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);

      expect([...result.buffers.dst as Uint32Array]).toEqual([11, 22, 33, 33]);
      expect(plan.supported).toBe(true);
      expect(plan.copies.map((copy) => copy.kind === "copy"
        ? { dstRoot: copy.dstRoot, srcRoot: copy.srcRoot, dstOffset: copy.dstOffset, srcOffset: copy.srcOffset, elementCount: copy.elementCount }
        : copy.kind === "fill"
          ? { dstRoot: copy.dstRoot, srcRoot: undefined, dstOffset: copy.dstOffset, srcOffset: undefined, elementCount: copy.elementCount }
          : { dstRoot: copy.dstRoot, srcRoot: undefined, dstOffset: undefined, srcOffset: undefined, elementCount: undefined })).toEqual([
        { dstRoot: "cBits", srcRoot: "src", dstOffset: 0, srcOffset: 0, elementCount: 2 },
        { dstRoot: "cBits", srcRoot: "src", dstOffset: 2, srcOffset: 2, elementCount: 1 },
        { dstRoot: "dst", srcRoot: "cBits", dstOffset: 0, srcOffset: 0, elementCount: 3 },
        { dstRoot: "dst", srcRoot: "cBits", dstOffset: 3, srcOffset: 2, elementCount: 1 },
      ]);
      expect(webGpuPlan.supported).toBe(true);
      if (webGpuPlan.supported) {
        expect(webGpuPlan.kind).toBe("host-copy");
        expect(webGpuPlan.steps).toHaveLength(5);
      }
    });

  it("host-lifts byte-granular cudaMemcpyToSymbol and cudaMemcpyFromSymbol", () => {
      const compiled = compileCudaLiteKernel(`
  __constant__ unsigned int cBits[3];
  __global__ void runtimeByteSymbolCopy(unsigned int *dst, const unsigned int *src) {
    cudaStream_t stream;
    if (threadIdx.x == 0) {
      cudaStreamCreate(&stream);
      cudaMemcpyToSymbol(cBits, src, 3);
      cudaMemcpyToSymbolAsync(cBits, src + 1, 5, 3, cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyFromSymbol(dst, cBits, 5);
      cudaMemcpyFromSymbolAsync(dst + 2, cBits, 4, 2, cudaMemcpyDeviceToDevice, stream);
      cudaStreamDestroy(stream);
    }
  }`, {
        referenceCudaRuntime: true,
        workgroupSize: [1, 1, 1],
      });
      const input = {
        buffers: {
          dst: new Uint32Array([0, 0xffffffff, 0, 0xffffffff]),
          src: new Uint32Array([0xaabbccdd, 0x11223344, 0x55667788]),
        },
        constants: {
          cBits: new Uint32Array([0, 0, 0]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const plan = createCudaRuntimeCopyPlan(compiled, input, launch);
      const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);

      expect([...result.buffers.dst as Uint32Array]).toEqual([0x44bbccdd, 0xffffff33, 0x223344bb, 0xffffffff]);
      expect(plan.supported).toBe(true);
      expect(plan.copies.map((copy) => copy.kind === "copy-bytes"
        ? { kind: copy.kind, dstRoot: copy.dstRoot, srcRoot: copy.srcRoot, dstByteOffset: copy.dstByteOffset, srcByteOffset: copy.srcByteOffset, byteCount: copy.byteCount }
        : { kind: copy.kind, dstRoot: copy.dstRoot, srcRoot: copy.kind === "copy" ? copy.srcRoot : undefined, dstByteOffset: undefined, srcByteOffset: undefined, byteCount: undefined })).toEqual([
        { kind: "copy-bytes", dstRoot: "cBits", srcRoot: "src", dstByteOffset: 0, srcByteOffset: 0, byteCount: 3 },
        { kind: "copy-bytes", dstRoot: "cBits", srcRoot: "src", dstByteOffset: 3, srcByteOffset: 4, byteCount: 5 },
        { kind: "copy-bytes", dstRoot: "dst", srcRoot: "cBits", dstByteOffset: 0, srcByteOffset: 0, byteCount: 5 },
        { kind: "copy-bytes", dstRoot: "dst", srcRoot: "cBits", dstByteOffset: 8, srcByteOffset: 2, byteCount: 4 },
      ]);
      expect(webGpuPlan.supported).toBe(true);
      if (webGpuPlan.supported) {
        expect(webGpuPlan.kind).toBe("host-copy");
        expect(webGpuPlan.steps).toHaveLength(5);
        expect(webGpuPlan.steps.slice(1).map((step) => step.program.name)).toEqual([
          "bg_peer_copy_bytes",
          "bg_peer_copy_bytes",
          "bg_peer_copy_bytes",
          "bg_peer_copy_bytes",
        ]);
      }
    });

  it("host-lifts byte-granular cudaMemsetToSymbol and cudaMemsetToSymbolAsync", () => {
      const compiled = compileCudaLiteKernel(`
  __constant__ unsigned int cBits[3];
  __global__ void runtimeSymbolMemset(unsigned int *dst) {
    cudaStream_t stream;
    if (threadIdx.x == 0) {
      cudaStreamCreate(&stream);
      cudaMemsetToSymbol(cBits, 0, sizeof(unsigned int));
      cudaMemsetToSymbolAsync(cBits, 0xff, 3, sizeof(unsigned int), stream);
      cudaMemcpyFromSymbol(dst, cBits, sizeof(unsigned int) * 3);
      cudaStreamDestroy(stream);
    }
  }`, {
        referenceCudaRuntime: true,
        workgroupSize: [1, 1, 1],
      });
      const input = {
        buffers: {
          dst: new Uint32Array([0, 0, 0]),
        },
        constants: {
          cBits: new Uint32Array([0x11223344, 0x55667788, 0xaabbccdd]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const plan = createCudaRuntimeCopyPlan(compiled, input, launch);
      const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);

      expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
      expect([...result.buffers.dst as Uint32Array]).toEqual([0, 0x55ffffff, 0xaabbccdd]);
      expect(plan.supported).toBe(true);
      expect(plan.copies.map((copy) => copy.kind === "fill"
        ? { kind: copy.kind, dstRoot: copy.dstRoot, dstOffset: copy.dstOffset, elementCount: copy.elementCount, byteValue: copy.byteValue }
        : copy.kind === "fill-bytes"
          ? { kind: copy.kind, dstRoot: copy.dstRoot, dstByteOffset: copy.dstByteOffset, byteCount: copy.byteCount, byteValue: copy.byteValue }
          : { kind: copy.kind, dstRoot: copy.dstRoot, dstOffset: copy.kind === "copy" ? copy.dstOffset : undefined, elementCount: copy.kind === "copy" ? copy.elementCount : undefined, byteValue: undefined })).toEqual([
        { kind: "fill", dstRoot: "cBits", dstOffset: 0, elementCount: 1, byteValue: 0 },
        { kind: "fill-bytes", dstRoot: "cBits", dstByteOffset: 4, byteCount: 3, byteValue: 255 },
        { kind: "copy", dstRoot: "dst", dstOffset: 0, elementCount: 3, byteValue: undefined },
      ]);
      expect(webGpuPlan.supported).toBe(true);
      if (webGpuPlan.supported) {
        expect(webGpuPlan.kind).toBe("host-copy");
        expect(webGpuPlan.steps).toHaveLength(4);
      }
    });

  it("host-lifts cudaMemcpy symbol copies for default-initialized device globals", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int gBits[3];
  __global__ void runtimeDeviceSymbolCopy(unsigned int *dst, const unsigned int *src) {
    if (threadIdx.x == 0) {
      cudaMemcpyToSymbol(gBits, src, sizeof(unsigned int) * 3);
      cudaMemcpyFromSymbol(dst, gBits, sizeof(unsigned int) * 3);
    }
  }`, {
        referenceCudaRuntime: true,
        workgroupSize: [1, 1, 1],
      });
      const input = {
        buffers: {
          dst: new Uint32Array([0, 0, 0]),
          src: new Uint32Array([7, 8, 9]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const plan = createCudaRuntimeCopyPlan(compiled, input, launch);
      const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);

      expect([...result.buffers.dst as Uint32Array]).toEqual([7, 8, 9]);
      expect(plan.supported).toBe(true);
      const semanticGlobalPlan = createCudaRuntimeCopyPlan(
        {
          ...compiled,
          ast: { ...compiled.ast, functions: [], kernels: [] },
          analysis: {
            ...compiled.analysis,
            kernel: { ...compiled.analysis.kernel, body: [] },
            deviceGlobals: [],
          },
        },
        input,
        launch,
      );
      expect(semanticGlobalPlan.supported).toBe(true);
      expect(plan.copies.map((copy) => copy.kind === "copy"
        ? { dstRoot: copy.dstRoot, srcRoot: copy.srcRoot, dstOffset: copy.dstOffset, srcOffset: copy.srcOffset, elementCount: copy.elementCount }
        : copy.kind === "fill"
          ? { dstRoot: copy.dstRoot, srcRoot: undefined, dstOffset: copy.dstOffset, srcOffset: undefined, elementCount: copy.elementCount }
          : { dstRoot: copy.dstRoot, srcRoot: undefined, dstOffset: undefined, srcOffset: undefined, elementCount: undefined })).toEqual([
        { dstRoot: "gBits", srcRoot: "src", dstOffset: 0, srcOffset: 0, elementCount: 3 },
        { dstRoot: "dst", srcRoot: "gBits", dstOffset: 0, srcOffset: 0, elementCount: 3 },
      ]);
      expect(webGpuPlan.supported).toBe(true);
      if (webGpuPlan.supported) {
        expect(webGpuPlan.kind).toBe("host-copy");
        expect(webGpuPlan.steps).toHaveLength(3);
      }
    });

  it("host-lifts cudaMemcpy2D and cudaMemcpy2DAsync row copies", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void runtimeCopy2D(float *dst, const float *src) {
    cudaStream_t stream;
    if (threadIdx.x == 0) {
      cudaMemcpy2D(dst, sizeof(float) * 3, src, sizeof(float) * 3, sizeof(float) * 2, 2, cudaMemcpyHostToDevice);
      cudaMemcpy2DAsync(dst + 1, sizeof(float) * 3, src + 3, sizeof(float) * 3, sizeof(float), 2, cudaMemcpyDeviceToHost, stream);
    }
  }`, {
        referenceCudaRuntime: true,
        workgroupSize: [1, 1, 1],
      });
      const input = {
        buffers: {
          dst: new Float32Array(6),
          src: new Float32Array([1, 2, 99, 3, 4, 99, 5, 6, 99]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const plan = createCudaRuntimeCopyPlan(compiled, input, launch);
      const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);

      expect([...result.buffers.dst as Float32Array]).toEqual([1, 3, 0, 3, 5, 0]);
      expect(plan.supported).toBe(true);
      expect(plan.copies.map((copy) => copy.kind === "copy"
        ? { dstOffset: copy.dstOffset, srcOffset: copy.srcOffset, elementCount: copy.elementCount }
        : copy.kind === "fill"
          ? { dstOffset: copy.dstOffset, srcOffset: undefined, elementCount: copy.elementCount }
          : { dstOffset: undefined, srcOffset: undefined, elementCount: undefined })).toEqual([
        { dstOffset: 0, srcOffset: 0, elementCount: 2 },
        { dstOffset: 3, srcOffset: 3, elementCount: 2 },
        { dstOffset: 1, srcOffset: 3, elementCount: 1 },
        { dstOffset: 4, srcOffset: 6, elementCount: 1 },
      ]);
      expect(webGpuPlan.supported).toBe(true);
      if (webGpuPlan.supported) {
        expect(webGpuPlan.kind).toBe("host-copy");
        expect(webGpuPlan.steps).toHaveLength(5);
      }
    });

  it("host-lifts byte-granular cudaMemcpy2D row copies", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void runtimeByteCopy2D(unsigned int *dst, const unsigned int *src) {
    cudaStream_t stream;
    if (threadIdx.x == 0) {
      cudaMemcpy2D(dst, 5, src, 4, 3, 2, cudaMemcpyDeviceToDevice);
      cudaMemcpy2DAsync(dst + 2, 5, src + 1, 4, 5, 1, cudaMemcpyDefault, stream);
    }
  }`, {
        referenceCudaRuntime: true,
        workgroupSize: [1, 1, 1],
      });
      const input = {
        buffers: {
          dst: new Uint32Array([0, 0xffffffff, 0, 0xffffffff]),
          src: new Uint32Array([0xaabbccdd, 0x11223344, 0x55667788]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const plan = createCudaRuntimeCopyPlan(compiled, input, launch);
      const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);

      expect([...result.buffers.dst as Uint32Array]).toEqual([0x00bbccdd, 0x223344ff, 0x11223344, 0xffffff88]);
      expect(plan.supported).toBe(true);
      expect(plan.copies.map((copy) => copy.kind === "copy-bytes"
        ? { kind: copy.kind, dstRoot: copy.dstRoot, srcRoot: copy.srcRoot, dstByteOffset: copy.dstByteOffset, srcByteOffset: copy.srcByteOffset, byteCount: copy.byteCount }
        : { kind: copy.kind, dstRoot: copy.dstRoot, srcRoot: copy.kind === "copy" ? copy.srcRoot : undefined, dstByteOffset: undefined, srcByteOffset: undefined, byteCount: undefined })).toEqual([
        { kind: "copy-bytes", dstRoot: "dst", srcRoot: "src", dstByteOffset: 0, srcByteOffset: 0, byteCount: 3 },
        { kind: "copy-bytes", dstRoot: "dst", srcRoot: "src", dstByteOffset: 5, srcByteOffset: 4, byteCount: 3 },
        { kind: "copy-bytes", dstRoot: "dst", srcRoot: "src", dstByteOffset: 8, srcByteOffset: 4, byteCount: 5 },
      ]);
      expect(webGpuPlan.supported).toBe(true);
      if (webGpuPlan.supported) {
        expect(webGpuPlan.kind).toBe("host-copy");
        expect(webGpuPlan.steps).toHaveLength(4);
        expect(webGpuPlan.steps.slice(1).map((step) => step.program.name)).toEqual([
          "bg_peer_copy_bytes",
          "bg_peer_copy_bytes",
          "bg_peer_copy_bytes",
        ]);
      }
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
      const detachedPlan = createCudaRuntimePlan({ ...compiled });

      expect(plan.operations.map((operation) => operation.kind)).toEqual([
        "device-launch",
        "runtime-copy",
        "grid-sync",
      ]);
      expect(detachedPlan.operations.map((operation) => operation.kind)).toEqual(plan.operations.map((operation) => operation.kind));
      expect(plan.canRunSingleDispatchWebGpu).toBe(false);
      expect(plan.referenceAvailable).toBe(true);
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

  it("ignores CUDA runtime compatibility calls in unreachable helpers for selected kernels", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void unused_runtime(float *dst, float *src) {
    cudaMemcpy(dst, src, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
  }

  __global__ void selected(float *x) {
    if (threadIdx.x == 0) {
      x[0] += 2.0f;
    }
  }`, { kernelName: "selected", workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            x: new Float32Array([3]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.x as Float32Array]).toEqual([5]);
      expect(compiled.wgsl).not.toContain("unused_runtime");
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
      const input = { buffers: { x: new Float32Array([1, 2]) }, scalars: { n: 2 } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const plan = createCudaHostDynamicLaunchPlan(compiled, input, launch);
      const detachedPlan = createCudaHostDynamicLaunchPlan({ ...compiled }, input, launch);
      const semanticChildPlan = createCudaHostDynamicLaunchPlan({
        ...compiled,
        ast: { ...compiled.ast, kernels: [], functions: [] },
      }, input, launch);

      expect(plan.supported).toBe(true);
      expect(detachedPlan.supported).toBe(true);
      expect(semanticChildPlan.supported).toBe(true);
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
      if (executionPlan.supported) {
        expect(executionPlan.steps.every((step) => step.program.wgsl.includes("browsergrad-semantic-wgsl"))).toBe(true);
      }
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

  it("projects inactive recursive launches through subgroup barrier IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void recursiveBarrierParent(int *out, int enabled) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
    __shared__ int values[64];
    if (enabled == 0) return;
    int warp = threadIdx.x / 32;
    if (warp == 0) {
      values[tile.thread_rank()] = tile.thread_rank();
      tile.sync();
      if (tile.thread_rank() == 0) out[0] = values[31];
      tile.sync();
    }
    block.sync();
    if (threadIdx.x == 0) recursiveBarrierParent<<<1, 64>>>(out, enabled - 1);
  }`, {
        kernelName: "recursiveBarrierParent",
        referenceDynamicParallelism: true,
        workgroupSize: [64, 1, 1],
        features: { subgroups: true },
      });
      const projected = projectSemanticHostRuntimeToGpuIr(compiled.kernelIr);
      const executionPlan = createCudaWebGpuExecutionPlan(
        compiled,
        { buffers: { out: new Int32Array(1) }, scalars: { enabled: 0 } },
        { gridDim: [1, 1, 1], blockDim: [64, 1, 1] },
        { compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options) },
      );

      expect(canEmitSemanticKernelIrWgsl(projected)).toBe(true);
      expect(executionPlan).toMatchObject({ supported: true, kind: "single-dispatch" });
      if (executionPlan.supported) {
        expect(executionPlan.steps[0]?.program.wgsl).toContain("browsergrad-semantic-wgsl");
      }
    });

  it("rejects host-lifted dynamic launches followed by runtime query writes", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void child(int *out) {
    if (threadIdx.x < 1) { out[0] = 7; }
  }
  __global__ void parent(int *out) {
    if (threadIdx.x < 1) {
      child<<<1, 1>>>(out);
      cudaGetDevice(&out[1]);
    }
  }`, {
        kernelName: "parent",
        referenceDynamicParallelism: true,
        workgroupSize: [1, 1, 1],
      });

      const executionPlan = createCudaWebGpuExecutionPlan(
        compiled,
        { buffers: { out: new Int32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
        { compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options) },
      );

      expect(executionPlan).toMatchObject({
        supported: false,
        blockers: [{
          code: "unsafe-parent-side-effects",
        }],
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
      if (executionPlan.supported) {
        expect(executionPlan.steps.every((step) => step.program.wgsl.includes("browsergrad-semantic-wgsl"))).toBe(true);
      }
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

  it("propagates host-evaluable scalar postfix updates into recursive dynamic launches", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void parent(float *out, int max_depth, int depth) {
    if (threadIdx.x < 1) {
      out[depth] = out[depth] + 1.0f;
      depth++;
      if (depth >= max_depth) { return; }
      parent<<<1, 1>>>(out, max_depth, depth);
    }
  }`, {
        kernelName: "parent",
        referenceDynamicParallelism: true,
        workgroupSize: [1, 1, 1],
      });
      const executionPlan = createCudaWebGpuExecutionPlan(
        compiled,
        { buffers: { out: new Float32Array(4) }, scalars: { max_depth: 3, depth: 0 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
        { compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options) },
      );

      expect(executionPlan).toMatchObject({
        supported: true,
        kind: "host-dynamic-launch",
      });
    });

  it("elides externally silent recursive dynamic launch trees", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void log_depth(int depth) {
    printf("%d", depth);
    __syncthreads();
  }
  __global__ void parent(int max_depth, int depth) {
    log_depth(depth);
    depth++;
    if (depth >= max_depth) { return; }
    parent<<<gridDim.x, blockDim.x>>>(max_depth, depth);
  }`, {
        kernelName: "parent",
        referenceDynamicParallelism: true,
        workgroupSize: [256, 1, 1],
      });
      const executionPlan = createCudaWebGpuExecutionPlan(
        compiled,
        { buffers: {}, scalars: { max_depth: 4, depth: 0 } },
        { gridDim: [1, 1, 1], blockDim: [256, 1, 1] },
        { compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options) },
      );

      expect(executionPlan).toMatchObject({
        supported: true,
        kind: "runtime-elided-single-dispatch",
      });
      expect(deviceLaunchTreeIsExternallySilent({ ...compiled })).toBe(true);
    });

  it("does not elide dynamic launches that write through reassigned pointer aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void child(int *out) {
    if (threadIdx.x == 0) {
      int *ptr = nullptr;
      ptr = out;
      atomicAdd(ptr, 1);
    }
  }
  __global__ void parent(int *out) {
    if (threadIdx.x == 0) {
      child<<<1, 1>>>(out);
      cudaDeviceSynchronize();
    }
  }`, {
        kernelName: "parent",
        referenceDynamicParallelism: true,
        workgroupSize: [1, 1, 1],
      });

      const executionPlan = createCudaWebGpuExecutionPlan(
        compiled,
        { buffers: { out: new Int32Array([4]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
        { compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options) },
      );

      expect(executionPlan).toMatchObject({
        supported: true,
        kind: "host-dynamic-launch",
      });
    });

  it("does not elide dynamic launches after conditional pointer alias rebinding", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void child(int *out, int clear_alias) {
    if (threadIdx.x == 0) {
      int *ptr = out;
      if (clear_alias) {
        ptr = nullptr;
      }
      atomicAdd(ptr, 1);
    }
  }
  __global__ void parent(int *out, int clear_alias) {
    if (threadIdx.x == 0) {
      child<<<1, 1>>>(out, clear_alias);
      cudaDeviceSynchronize();
    }
  }`, {
        kernelName: "parent",
        referenceDynamicParallelism: true,
        workgroupSize: [1, 1, 1],
      });

      const executionPlan = createCudaWebGpuExecutionPlan(
        compiled,
        { buffers: { out: new Int32Array([4]) }, scalars: { clear_alias: 0 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
        { compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options) },
      );

      expect(executionPlan).toMatchObject({
        supported: false,
        blockers: [{
          code: "dynamic-child-compile-failed",
        }],
      });
    });

  it("does not elide dynamic launches after conditional pointer alias initialization", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void child(int *out, int use_out) {
    if (threadIdx.x == 0) {
      int *ptr = use_out ? out : out;
      atomicAdd(ptr, 1);
    }
  }
  __global__ void parent(int *out, int use_out) {
    if (threadIdx.x == 0) {
      child<<<1, 1>>>(out, use_out);
      cudaDeviceSynchronize();
    }
  }`, {
        kernelName: "parent",
        referenceDynamicParallelism: true,
        workgroupSize: [1, 1, 1],
      });

      const executionPlan = createCudaWebGpuExecutionPlan(
        compiled,
        { buffers: { out: new Int32Array([4]) }, scalars: { use_out: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
        { compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options) },
      );

      expect(executionPlan).toMatchObject({
        supported: true,
        kind: "host-dynamic-launch",
      });
    });

  it("does not elide dynamic launches after assignment-expression pointer alias initialization", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void child(int *out) {
    if (threadIdx.x == 0) {
      int *tmp = nullptr;
      int *ptr = (tmp = out);
      atomicAdd(ptr, 1);
    }
  }
  __global__ void parent(int *out) {
    if (threadIdx.x == 0) {
      child<<<1, 1>>>(out);
      cudaDeviceSynchronize();
    }
  }`, {
        kernelName: "parent",
        referenceDynamicParallelism: true,
        workgroupSize: [1, 1, 1],
      });

      const executionPlan = createCudaWebGpuExecutionPlan(
        compiled,
        { buffers: { out: new Int32Array([4]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
        { compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options) },
      );

      expect(executionPlan).toMatchObject({
        supported: true,
        kind: "host-dynamic-launch",
      });

      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array([4]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...result.buffers.out as Int32Array]).toEqual([5]);
    });

  it("does not elide dynamic launches after sequence pointer alias initialization", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void child(int *out) {
    if (threadIdx.x == 0) {
      int *tmp = nullptr;
      int *ptr = (tmp = out, tmp);
      atomicAdd(ptr, 1);
    }
  }
  __global__ void parent(int *out) {
    if (threadIdx.x == 0) {
      child<<<1, 1>>>(out);
      cudaDeviceSynchronize();
    }
  }`, {
        kernelName: "parent",
        referenceDynamicParallelism: true,
        workgroupSize: [1, 1, 1],
      });

      const executionPlan = createCudaWebGpuExecutionPlan(
        compiled,
        { buffers: { out: new Int32Array([4]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
        { compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options) },
      );

      expect(executionPlan).toMatchObject({
        supported: true,
        kind: "host-dynamic-launch",
      });

      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array([4]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...result.buffers.out as Int32Array]).toEqual([5]);
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

  it("does not elide dynamic launches that allocate external DevicePool memory", () => {
      const source = `
  __global__ void parentKernel(int N) {
    size_t size = N * sizeof(float);
    float *devBuf = (float*) deviceAllocate(&g_pool, size);
    if (devBuf == nullptr) return;
    dim3 grid((N + 255) / 256);
    dim3 block(256);
    childKernel<<<grid, block>>>(devBuf, N);
  }
  __device__ void childKernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      data[idx] += 3.14f;
    }
  }`;
      const compiled = compileCudaLiteKernel(source, {
        kernelName: "parentKernel",
        referenceDynamicParallelism: true,
        workgroupSize: [2, 1, 1],
      });
      const input = {
        buffers: {},
        scalars: { N: 2 },
        memoryPools: { g_pool: { data: new Uint32Array(4), offset: new Uint32Array([0]) } },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
      const executionPlan = createCudaWebGpuExecutionPlan(compiled, input, launch, {
        compileKernel: (childSource, options = {}) => compileCudaLiteKernel(childSource, options),
      });

      expect(executionPlan).toMatchObject({
        supported: true,
        kind: "host-dynamic-launch",
      });

      const result = runCompiledKernelReference(compiled, input, launch);
      expect([...result.buffers.g_pool as Uint32Array]).toEqual([
        floatBits(3.14), floatBits(3.14), floatBits(3.14), floatBits(3.14),
      ]);
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
      expect(canEmitSemanticKernelIrWgsl(child.wgslLegalizedKernelIr, { pointerBaseOffsets: plan.launches[0]!.pointerBaseOffsets })).toBe(true);
      expect(child.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(child.wgsl).toContain("bg_base_out");
      expect(child.wgsl).toContain("var out__bg_ptr_offset: i32 = i32(bg_uniforms.bg_base_out);");
      expect(child.wgsl).toContain("out[u32((out__bg_ptr_offset + 0))] = 7.0;");

      const staleOffsetKey = compileCudaLiteKernel(compiled.ast.source, {
        kernelName: "child",
        pointerBaseOffsets: { stale: 9 },
        workgroupSize: [1, 1, 1],
      });
      expect(staleOffsetKey.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(staleOffsetKey.wgsl).not.toContain("bg_base_stale");
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

  it("models CUDA stream and event creation as deterministic handle writes", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void stream_event_create_handles(uint *handles, int *statusOut) {
    cudaStream_t stream = 7u;
    cudaStream_t priorityStream = 8u;
    cudaEvent_t event = 9u;
    if (threadIdx.x < 1) {
      int streamStatus = cudaStreamCreate(&stream);
      int priorityStatus = cudaStreamCreateWithPriority(&priorityStream, cudaStreamNonBlocking, 0);
      int eventStatus = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
      handles[0] = stream;
      handles[1] = priorityStream;
      handles[2] = event;
      statusOut[0] = streamStatus + priorityStatus + eventStatus;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { handles: new Uint32Array([99, 99, 99]), statusOut: new Int32Array([-1]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
      expect(compiled.wgsl).toContain("var streamStatus: i32 = 0;");
      expect(compiled.wgsl).toContain("stream = 0u;");
      expect(compiled.wgsl).toContain("var priorityStatus: i32 = 0;");
      expect(compiled.wgsl).toContain("priorityStream = 0u;");
      expect(compiled.wgsl).toContain("var eventStatus: i32 = 0;");
      expect(compiled.wgsl).toContain("event = 0u;");
      expect(createCudaRuntimePlan(compiled).operations.map((operation) => operation.kind).every((kind) => kind === "device-sync")).toBe(true);
      expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { handles: new Uint32Array(3), statusOut: new Int32Array(1) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
      expect([...result.buffers.handles as Uint32Array]).toEqual([0, 0, 0]);
      expect([...result.buffers.statusOut as Int32Array]).toEqual([0]);
    });

  it("models CUDA stream device/id/capture queries as deterministic zero-state writes", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void stream_queries(float *out) {
    cudaStream_t stream;
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusActive;
    cudaGraph_t graph = 7u;
    if (threadIdx.x < 1) {
      int device = -1;
      uint streamId = 9u;
      cudaStreamCaptureStatus captureInfoStatus = cudaStreamCaptureStatusActive;
      cudaStreamCaptureStatus captureInfoV2Status = cudaStreamCaptureStatusActive;
      uint captureId = 4u;
      uint captureIdV2 = 5u;
      uint dependencyCount = 3u;
      uint dependencyCountV2 = 6u;
      cudaGraph_t graphV2 = 8u;
      cudaStreamCreate(&stream);
      int deviceStatus = cudaStreamGetDevice(stream, &device);
      int idStatus = cudaStreamGetId(stream, &streamId);
      int captureStatusResult = cudaStreamIsCapturing(stream, &captureStatus);
      int captureInfoResult = cudaStreamGetCaptureInfo(stream, &captureInfoStatus, &captureId, &graph, NULL, NULL, &dependencyCount);
      int captureInfoV2Result = cudaStreamGetCaptureInfo_v2(stream, &captureInfoV2Status, &captureIdV2, &graphV2, NULL, NULL, &dependencyCountV2);
      cudaStreamDestroy(stream);
      out[0] = (float)(device + (int)streamId + captureStatus + captureInfoStatus + captureInfoV2Status + (int)captureId + (int)captureIdV2 + (int)graph + (int)graphV2 + (int)dependencyCount + (int)dependencyCountV2 + deviceStatus + idStatus + captureStatusResult + captureInfoResult + captureInfoV2Result);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array([-1]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
      expect(compiled.wgsl).toContain("device = 0;");
      expect(compiled.wgsl).toContain("streamId = 0u;");
      expect(compiled.wgsl).toContain("captureStatus = 0;");
      expect(compiled.wgsl).toContain("captureInfoStatus = 0;");
      expect(compiled.wgsl).toContain("captureInfoV2Status = 0;");
      expect(compiled.wgsl).toContain("dependencyCount = 0u;");
      expect(compiled.wgsl).toContain("dependencyCountV2 = 0u;");
      expect(createCudaRuntimePlan(compiled).operations).toEqual([]);
      expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { out: new Float32Array([-1]) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
      expect([...result.buffers.out as Float32Array]).toEqual([0]);
    });

  it("models CUDA stream capture graph lifecycle calls as host-managed no-ops", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void stream_capture_graph(uint *graphOut, int *statusOut) {
    cudaStream_t stream;
    cudaGraph_t graph = 9u;
    if (threadIdx.x < 1) {
      cudaStreamCreate(&stream);
      int begin = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
      int update = cudaStreamUpdateCaptureDependencies(stream, NULL, 0, 0);
      int end = cudaStreamEndCapture(stream, &graph);
      int destroy = cudaGraphDestroy(graph);
      cudaStreamDestroy(stream);
      graphOut[0] = graph;
      statusOut[0] = begin + update + end + destroy;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { graphOut: new Uint32Array([99]), statusOut: new Int32Array([-1]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
      expect(compiled.wgsl).toContain("var begin: i32 = 0;");
      expect(compiled.wgsl).toContain("var update: i32 = 0;");
      expect(compiled.wgsl).toContain("var end: i32 = 0;");
      expect(compiled.wgsl).toContain("graph = 0u;");
      expect(compiled.wgsl).toContain("var destroy: i32 = 0;");
      expect(createCudaRuntimePlan(compiled).operations).toEqual([]);
      expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { graphOut: new Uint32Array([99]), statusOut: new Int32Array([-1]) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
      expect([...result.buffers.graphOut as Uint32Array]).toEqual([0]);
      expect([...result.buffers.statusOut as Int32Array]).toEqual([0]);
    });

  it("models CUDA graph create and instantiate lifecycle calls as host-managed no-ops", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void graph_lifecycle(uint *handles, int *statusOut) {
    cudaGraph_t graph = 7u;
    cudaGraphExec_t exec = 8u;
    cudaGraphExec_t execWithFlags = 9u;
    cudaGraphNode_t errorNode = 10u;
    if (threadIdx.x < 1) {
      int create = cudaGraphCreate(&graph, 0);
      int instantiate = cudaGraphInstantiate(&exec, graph, &errorNode, NULL, 0);
      int instantiateFlags = cudaGraphInstantiateWithFlags(&execWithFlags, graph, 0);
      int destroyExec = cudaGraphExecDestroy(exec);
      int destroyGraph = cudaGraphDestroy(graph);
      handles[0] = graph;
      handles[1] = exec;
      handles[2] = execWithFlags;
      handles[3] = errorNode;
      statusOut[0] = create + instantiate + instantiateFlags + destroyExec + destroyGraph;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { handles: new Uint32Array([99, 99, 99, 99]), statusOut: new Int32Array([-1]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
      expect(compiled.wgsl).toContain("var create: i32 = 0;");
      expect(compiled.wgsl).toContain("graph = 0u;");
      expect(compiled.wgsl).toContain("var instantiate: i32 = 0;");
      expect(compiled.wgsl).toContain("exec = 0u;");
      expect(compiled.wgsl).toContain("errorNode = 0u;");
      expect(compiled.wgsl).toContain("var instantiateFlags: i32 = 0;");
      expect(compiled.wgsl).toContain("execWithFlags = 0u;");
      expect(compiled.wgsl).toContain("var destroyExec: i32 = 0;");
      expect(createCudaRuntimePlan(compiled).operations).toEqual([]);
      expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { handles: new Uint32Array(4), statusOut: new Int32Array(1) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
      expect([...result.buffers.handles as Uint32Array]).toEqual([0, 0, 0, 0]);
      expect([...result.buffers.statusOut as Int32Array]).toEqual([0]);
    });

  it("models CUDA graph upload and exec-update lifecycle calls as host-managed no-ops", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void graph_update_lifecycle(uint *handles, int *statusOut) {
    cudaGraph_t graph = 7u;
    cudaGraphExec_t exec = 8u;
    cudaGraphNode_t errorNode = 10u;
    cudaGraphExecUpdateResult updateResult = cudaGraphExecUpdateErrorTopologyChanged;
    if (threadIdx.x < 1) {
      int create = cudaGraphCreate(&graph, 0);
      int instantiate = cudaGraphInstantiateWithFlags(&exec, graph, 0);
      int upload = cudaGraphUpload(exec, 0);
      int update = cudaGraphExecUpdate(exec, graph, &errorNode, &updateResult);
      int destroyExec = cudaGraphExecDestroy(exec);
      int destroyGraph = cudaGraphDestroy(graph);
      handles[0] = graph;
      handles[1] = exec;
      handles[2] = errorNode;
      handles[3] = updateResult;
      statusOut[0] = create + instantiate + upload + update + destroyExec + destroyGraph + (updateResult == cudaGraphExecUpdateSuccess ? 0 : 100);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { handles: new Uint32Array([99, 99, 99, 99]), statusOut: new Int32Array([-1]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
      expect(compiled.wgsl).toContain("var upload: i32 = 0;");
      expect(compiled.wgsl).toContain("var update: i32 = 0;");
      expect(compiled.wgsl).toContain("errorNode = 0u;");
      expect(compiled.wgsl).toContain("updateResult = 0u;");
      expect(createCudaRuntimePlan(compiled).operations).toEqual([]);
      expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { handles: new Uint32Array(4), statusOut: new Int32Array(1) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
      expect([...result.buffers.handles as Uint32Array]).toEqual([0, 0, 0, 0]);
      expect([...result.buffers.statusOut as Int32Array]).toEqual([0]);
    });

  it("models cudaEventElapsedTime as a zero elapsed-time write", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void event_elapsed(float *out) {
    cudaEvent_t start;
    cudaEvent_t stop;
    if (threadIdx.x < 1) {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      cudaEventRecord(stop);
      float ms = -1.0f;
      int status = -1;
      status = cudaEventElapsedTime(&ms, start, stop);
      cudaEventElapsedTime(&out[2], start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      out[0] = ms;
      out[1] = (float)status;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array([-1, -1, -1]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
      expect(compiled.wgsl).toContain("var status: i32 = -(1);");
      expect(compiled.wgsl).toContain("status = 0;");
      expect(compiled.wgsl).toContain("ms = 0.0;");
      expect(compiled.wgsl).toContain("out[2u] = 0.0;");
      expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { out: new Float32Array(3) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
      expect([...result.buffers.out as Float32Array]).toEqual([0, 0, 0]);
    });

  it("models CUDA stream priority range queries", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void stream_priority_range(int *out) {
    if (threadIdx.x < 1) {
      int least = -7;
      int greatest = -9;
      int status = cudaDeviceGetStreamPriorityRange(&least, &greatest);
      cudaDeviceGetStreamPriorityRange(&out[3], &out[4]);
      out[0] = least;
      out[1] = greatest;
      out[2] = status;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(5).fill(-1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
      expect(compiled.wgsl).toContain("var status: i32 = 0;");
      expect(compiled.wgsl).toContain("least = 0;");
      expect(compiled.wgsl).toContain("greatest = 0;");
      expect(compiled.wgsl).toContain("out[3u] = 0;");
      expect(compiled.wgsl).toContain("out[4u] = 0;");
      expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { out: new Int32Array(5) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
      expect([...result.buffers.out as Int32Array]).toEqual([0, 0, 0, 0, 0]);
    });

  it("models CUDA stream flag and priority queries", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void stream_queries(uint *flagsOut, int *priorityOut) {
    cudaStream_t stream;
    if (threadIdx.x < 1) {
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      uint flags = 99u;
      int priority = -7;
      int flagStatus = -1;
      flagStatus = cudaStreamGetFlags(stream, &flags);
      int priorityStatus = cudaStreamGetPriority(stream, &priority);
      cudaStreamGetFlags(stream, &flagsOut[1]);
      cudaStreamGetPriority(stream, &priorityOut[2]);
      cudaStreamDestroy(stream);
      flagsOut[0] = flags + (uint)flagStatus;
      priorityOut[0] = priority;
      priorityOut[1] = priorityStatus;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            flagsOut: new Uint32Array(2).fill(77),
            priorityOut: new Int32Array(3).fill(-1),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
      expect(compiled.wgsl).toContain("var flagStatus: i32 = -(1);");
      expect(compiled.wgsl).toContain("flagStatus = 0;");
      expect(compiled.wgsl).toMatch(/\bflags = 0u?;/u);
      expect(compiled.wgsl).toContain("priority = 0;");
      expect(compiled.wgsl).toContain("flagsOut[1u] = 0u;");
      expect(compiled.wgsl).toContain("priorityOut[2u] = 0;");
      expect(createCudaWebGpuExecutionPlan(
        compiled,
        { buffers: { flagsOut: new Uint32Array(2), priorityOut: new Int32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      ).supported).toBe(true);
      expect([...result.buffers.flagsOut as Uint32Array]).toEqual([0, 0]);
      expect([...result.buffers.priorityOut as Int32Array]).toEqual([0, 0, 0]);
    });

  it("models deprecated CUDA thread runtime aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void thread_aliases(uint *limits, int *status) {
    if (threadIdx.x < 1) {
      size_t limit = 0;
      int cache = -1;
      int limitStatus = cudaThreadGetLimit(&limit, cudaLimitPrintfFifoSize);
      int cacheStatus = cudaThreadGetCacheConfig(&cache);
      int setLimitStatus = cudaThreadSetLimit(cudaLimitPrintfFifoSize, limit);
      int setCacheStatus = cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
      int syncStatus = cudaThreadSynchronize();
      int exitStatus = cudaThreadExit();
      limits[0] = limit;
      status[0] = cache;
      status[1] = limitStatus + cacheStatus + setLimitStatus + setCacheStatus + syncStatus + exitStatus;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { limits: new Uint32Array([0]), status: new Int32Array([-1, -1]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
      expect(compiled.wgsl).toContain("var limitStatus: i32 = 0;");
      expect(compiled.wgsl).toContain("var cacheStatus: i32 = 0;");
      expect(compiled.wgsl).toContain("var setLimitStatus: i32 = 0;");
      expect(compiled.wgsl).toContain("var setCacheStatus: i32 = 0;");
      expect(compiled.wgsl).toContain("var syncStatus: i32 = 0;");
      expect(compiled.wgsl).toContain("var exitStatus: i32 = 0;");
      expect(createCudaWebGpuExecutionPlan(
        compiled,
        { buffers: { limits: new Uint32Array(1), status: new Int32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      ).supported).toBe(true);
      expect([...result.buffers.limits as Uint32Array]).toEqual([1048576]);
      expect([...result.buffers.status as Int32Array]).toEqual([0, 0]);
    });

  it("rejects nested CUDA runtime query side effects before lowering can drop writes", () => {
      const analysis = analyzeCudaLite(parseCudaLite(`
  __global__ void nested_runtime_query(uint *out) {
    if (threadIdx.x < 1) {
      size_t limit = 99u;
      out[0] = (uint)cudaThreadGetLimit(&limit, cudaLimitPrintfFifoSize) + limit;
    }
  }`));
      const dynamicSmemAnalysis = analyzeCudaLite(parseCudaLite(`
  __global__ void nested_dynamic_smem_query(uint *out) {
    if (threadIdx.x < 1) {
      uint dynamicSmem = 99u;
      out[0] = cudaOccupancyAvailableDynamicSMemPerBlock(&dynamicSmem, nested_dynamic_smem_query, 1, 128) + dynamicSmem;
    }
  }`));

      expect(analysis.diagnostics).toContainEqual(expect.objectContaining({
        code: "side-effect-expression",
        message: expect.stringContaining("side-effecting CUDA runtime calls"),
      }));
      expect(dynamicSmemAnalysis.diagnostics).toContainEqual(expect.objectContaining({
        code: "side-effect-expression",
        message: expect.stringContaining("side-effecting CUDA runtime calls"),
      }));
    });

  it("models CUDA occupancy runtime queries", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void runtime_occupancy(int *out, unsigned int *smemOut) {
    if (threadIdx.x < 1) {
      int active = -1;
      int activeFlags = -1;
      int minGrid = -1;
      int blockSize = -1;
      int minGridFlags = -1;
      int blockSizeFlags = -1;
      unsigned int dynamicSmem = 0u;
      int activeStatus = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, runtime_occupancy, 128, 0);
      int activeFlagsStatus = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&activeFlags, runtime_occupancy, 128, 0, cudaOccupancyDisableCachingOverride);
      int potentialStatus = cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, runtime_occupancy, 0, 0);
      int potentialFlagsStatus = cudaOccupancyMaxPotentialBlockSizeWithFlags(&minGridFlags, &blockSizeFlags, runtime_occupancy, 0, 0, cudaOccupancyDefault);
      int dynamicSmemStatus = cudaOccupancyAvailableDynamicSMemPerBlock(&dynamicSmem, runtime_occupancy, 1, 128);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&out[10], runtime_occupancy, 128, 0);
      cudaOccupancyMaxPotentialBlockSize(&out[11], &out[12], runtime_occupancy, 0, 0);
      cudaOccupancyAvailableDynamicSMemPerBlock(&smemOut[1], runtime_occupancy, 1, 128);
      out[0] = active;
      out[1] = activeStatus;
      out[2] = activeFlags;
      out[3] = activeFlagsStatus;
      out[4] = minGrid;
      out[5] = blockSize;
      out[6] = potentialStatus;
      out[7] = minGridFlags;
      out[8] = blockSizeFlags;
      out[9] = potentialFlagsStatus;
      out[13] = dynamicSmemStatus;
      smemOut[0] = dynamicSmem;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(14).fill(-1), smemOut: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
      expect(compiled.wgsl).toContain("var activeStatus: i32 = 0;");
      expect(compiled.wgsl).toContain("var potentialStatus: i32 = 0;");
      expect(compiled.wgsl).toContain("active = 1;");
      expect(compiled.wgsl).toContain("activeFlags = 1;");
      expect(compiled.wgsl).toContain("minGrid = 1;");
      expect(compiled.wgsl).toContain("blockSize = 256;");
      expect(compiled.wgsl).toContain("minGridFlags = 1;");
      expect(compiled.wgsl).toContain("blockSizeFlags = 256;");
      expect(compiled.wgsl).toContain("dynamicSmem = 49152u;");
      expect(compiled.wgsl).toContain("out[10u] = 1;");
      expect(compiled.wgsl).toContain("out[11u] = 1;");
      expect(compiled.wgsl).toContain("out[12u] = 256;");
      expect(compiled.wgsl).toContain("smemOut[1u] = 49152u;");
      expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { out: new Int32Array(14), smemOut: new Uint32Array(2) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
      expect([...result.buffers.out as Int32Array]).toEqual([1, 0, 1, 0, 1, 256, 0, 1, 256, 0, 1, 1, 256, 0]);
      expect([...result.buffers.smemOut as Uint32Array]).toEqual([49152, 49152]);
    });

  it("validates CUDA graph conditional setters as host-managed scheduler side effects", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void graphCondition(int *input, int *out, cudaGraphConditionalHandle handle) {
    if (threadIdx.x < 1) {
      unsigned int value = input[0] & 1;
      cudaGraphSetConditional(handle, value);
      out[0] = value;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { input: new Int32Array([3]), out: new Int32Array(1) }, scalars: { handle: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics).toContainEqual(expect.objectContaining({
        code: "cuda-graph-conditional-host-orchestration",
        severity: "warning",
      }));
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("let bg_noop_arg_");
      expect([...result.buffers.out as Int32Array]).toEqual([1]);
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
});
export {
  fs,
  path,
  fileURLToPath,
  describe,
  expect,
  it,
  createWgslFloat16Array,
  CudaLiteCompilerError,
  analyzeCudaLite,
  compileCudaLiteOptionsFromKernelFeatures,
  createCudaLiteCompileCacheKey,
  createCudaLiteCompilerCache,
  compileCudaLiteKernelForWebGpu,
  compileCudaLiteKernel,
  emitSemanticKernelIrWgsl,
  prepareCompiledKernelWebGpu,
  canEmitSemanticKernelIrWgsl,
  canRunCompiledKernelSemanticReference,
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
  getCudaFeatureRegistry,
  normalizeCudaWebGpuReadbackNames,
  parseCudaLite,
  runCompiledKernelReference,
  runCompiledKernelSemanticReference,
  runCompiledKernelWebGpu,
  summarizeCudaWebGpuExecutionPlan,
  validateCudaKernelLaunch,
  constantBufferInputs,
  cudaWebGpuDefaultReadbackNames,
  deviceGlobalBufferInputs,
  deviceLaunchTreeIsExternallySilent,
  packCudaWebGpuUniformParams,
  packageRoot,
  semanticIr,
  gammaCoefficients,
  gammaApprox,
  erfApprox,
  roundEven,
  roundAway,
  nextafterF32Buffer,
  nextafterF32,
  nextafterU32,
  floatToBits,
  bitsToFloat,
  nextafterApprox,
  SAXPY,
  TILED_MATMUL,
  LOCAL_ARRAY,
  DEVICE_POOL_ALLOC,
  RAW_POOL_ALLOC,
  DEVICE_POINTER_HELPERS,
  SHARED_POINTER_HELPERS,
  EXTERNAL_POOL_ALLOC,
  floatBits,
  expectParseDiagnosticCode,
  collectEmittedDiagnosticCodes,
  compilerSourceText,
  compilerExampleText,
};

export type {
  CompiledCudaLiteKernel,
};
