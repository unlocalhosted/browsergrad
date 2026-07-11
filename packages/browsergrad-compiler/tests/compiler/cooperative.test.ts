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
  semanticKernelIrWgslPreflightBlocker,
  semanticKernelIrWgslPreflightFailure,
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
} from "../../src/index";
import { lowerAnalyzedCudaLiteToKernelIr } from "../../src/analyzer";
import {
  constantBufferInputs,
  cudaWebGpuDefaultReadbackNames,
  deviceGlobalBufferInputs,
} from "../../src/webgpu_inputs";
import { deviceLaunchTreeIsExternallySilent } from "../../src/runtime_elision";
import { packCudaWebGpuUniformParams } from "../../src/webgpu_orchestration";

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

function backendIr(compiled: CompiledCudaLiteKernel) {
  return lowerAnalyzedCudaLiteToKernelIr(compiled.analysis, {
    workgroupSize: compiled.kernelIr.workgroupSize,
    ...(compiled.dynamicSharedMemory === undefined ? {} : { dynamicSharedMemory: compiled.dynamicSharedMemory }),
  });
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

describe("CUDA-lite compiler: Cooperative execution and matrix tiles", () => {
  it("emits simple shared memory and barriers from semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sharedReverse(float* out) {
    __shared__ float tile[4];
    int tid = threadIdx.x;
    tile[tid] = (float)(tid + 1);
    __syncthreads();
    out[tid] = tile[3 - tid];
  }
  `, { workgroupSize: [4, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var<workgroup> tile: array<f32, 4>;");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([4, 3, 2, 1]);
      expect(semanticResult.trace.some((thread) => thread.sharedWrites.length > 0)).toBe(true);
      expect(semanticResult.trace.some((thread) => thread.sharedReads.length > 0)).toBe(true);
      expect([...result.buffers.out as Float32Array]).toEqual([4, 3, 2, 1]);
    });

  it("scales active-lane byte-root pointer-array differences before barriers", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ uint active_byte_pointer_array_diff_index_helper(uint *counter) {
    atomicAdd(counter, 1u);
    return 1u;
  }

  __global__ void active_lane_byte_root_pointer_array_diff(uchar *bytes, uint *counter, int *summary, int limit) {
    int tid = threadIdx.x;
    for (int step = 0; step < 2; step++) {
      if (tid >= limit) return;
      __syncthreads();
      float *ptrs[2];
      ptrs[0] = reinterpret_cast<float*>(bytes + ((tid + 1) * 4));
      ptrs[1] = reinterpret_cast<float*>(bytes + ((tid + 3) * 4));
      summary[tid] = summary[tid] + (ptrs[active_byte_pointer_array_diff_index_helper(counter + tid)] - reinterpret_cast<float*>(bytes + ((tid + 1) * 4)));
      __syncthreads();
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl.match(/\bactive_byte_pointer_array_diff_index_helper\(/gu) ?? []).toHaveLength(2);
      expect(compiled.wgsl).toMatch(/let bg_pointer_array_index_\d+: u32 = active_byte_pointer_array_diff_index_helper\(/u);
      expect(compiled.wgsl).toMatch(/summary\[.+\] = i32\(\(summary\[.+\] \+ select\(0, select\(.+ \/ 4\), \(ptrs_buffer\[.+\] == 0u\)\), \(ptrs_buffer\[.+\] == 0u\)\)\)\);/u);
    });

  it("runs a shared-memory tiled matmul reference and emits barriers", () => {
      const compiled = compileCudaLiteKernel(TILED_MATMUL, { workgroupSize: [2, 2, 1] });
      const input = {
        buffers: {
          A: new Float32Array([1, 2, 3, 4]),
          B: new Float32Array([5, 6, 7, 8]),
          C: new Float32Array(4),
        },
        scalars: { N: 2 },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 2, 1] as const };
      const result = runCompiledKernelReference(
        compiled,
        input,
        launch,
      );
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect([...result.buffers.C as Float32Array]).toEqual([19, 22, 43, 50]);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect([...semanticResult.buffers.C as Float32Array]).toEqual([19, 22, 43, 50]);
      expect(compiled.wgsl).toContain("var<workgroup> As: array<f32, 4>;");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
    });

  it("schedules uniform cg::sync barriers through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  namespace cg = cooperative_groups;
  __global__ void cgShared(float *out) {
    cg::thread_block block = cg::this_thread_block();
    __shared__ float tile[2][2];
    int tid = threadIdx.x;
    for (int pass = 0; pass < 2; pass++) {
      tile[tid][pass] = (float)(tid + pass);
      cg::sync(block);
      out[tid * 2 + pass] = tile[1 - tid][pass];
      cg::sync(block);
    }
  }`, { workgroupSize: [2, 1, 1] });
      const input = { buffers: { out: new Float32Array(4) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(compiled.kernelIr.barrierUniformity.kernel.verified).toBe(true);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 2, 0, 1]);
      expect(compiled.wgsl).toContain("workgroupBarrier();");
    });

  it("keeps analyzer-proven top-level barriers semantic-direct around unrelated loops", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void stagedTranspose(float *out) {
    __shared__ float tile[4];
    int tid = threadIdx.x;
    for (int i = 0; i < 2; ++i) {
      tile[tid] = (float)(tid + i);
    }
    __syncthreads();
    for (int i = 0; i < 2; ++i) {
      out[tid * 2 + i] = tile[3 - tid] + (float)i;
    }
  }`, { workgroupSize: [4, 1, 1] });
      const input = { buffers: { out: new Float32Array(8) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(compiled.kernelIr.barrierUniformity.kernel.verified).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([4, 5, 3, 4, 2, 3, 1, 2]);
    });

  it("lowers CUDA warp shuffle helpers to workgroup-backed warp intrinsics", () => {
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
      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_down_float_32(val, 16u, 32u, local_id)");
      expect(compiled.wgsl).toContain("fn warpReduceSum(bg_arg_val: f32, local_id: vec3<u32>, workgroup_id: vec3<u32>, num_workgroups: vec3<u32>) -> f32");
      expect(compiled.wgsl).toContain("var val: f32 = bg_arg_val;");
      expect(compiled.wgsl).toContain("val = warpReduceSum(val, local_id, workgroup_id, num_workgroups)");
    });

  it("lowers semantic block reductions as subgroup reductions", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void kernel(float* out, float value) {
    out[0] = blockReduce(value, false, 0.0f);
  }`, { features: { subgroups: true } });

      expect(compiled.wgsl).toContain("out[0] = f32(subgroupAdd(bg_uniforms.value))");
    });

  it("lowers masked warp reductions using the value operand", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void kernel(int* out, int value, unsigned int mask) {
    out[0] = warpReduceSum(mask, value);
  }`, { features: { subgroups: true }, workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(1) }, scalars: { value: 7, mask: 0xffffffff } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("bg_semantic_warp_reduce_sum_i32_32_masked(bg_uniforms.value, bg_uniforms.mask, local_id)");
      expect([...result.buffers.out as Int32Array]).toEqual([7]);
    });

  it("applies masked warp reductions to logical CUDA lanes", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void maskedWarpSum(int* out) {
    int value = threadIdx.x + 1;
    int reduced = warpReduceSum(5u, value);
    if (threadIdx.x == 0) out[0] = reduced;
  }`, { features: { subgroups: true }, workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("(mask_arg & (1u << lane)) != 0u");
      expect([...result.buffers.out as Int32Array]).toEqual([4]);
    });

  it("infers subgroup reduction value types for mixed scalar math", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void kernel(float* out, float value, int n, unsigned int mask) {
    float sum = warpReduceSum(mask, value);
    float total = __reduce_add_sync(mask, value);
    out[0] = (sum + total) / n;
  }`, { features: { subgroups: true }, workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) }, scalars: { value: 4, n: 4, mask: 0xffffffff } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("/ f32(bg_uniforms.n)");
      expect([...result.buffers.out as Float32Array]).toEqual([2]);
    });

  it("runs scalar warp reductions across reference threads", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void reduce(const float* x, float* out) {
    int tid = threadIdx.x;
    float value = x[tid];
    value = warp_reduce_sum_f32(value);
    if (tid == 0) out[0] = value;
  }`, { features: { subgroups: true }, workgroupSize: [32, 1, 1] });
      const input = new Float32Array(Array.from({ length: 32 }, (_unused, index) => index + 1));
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: input, out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [32, 1, 1] },
      );

      expect([...result.buffers.out as Float32Array]).toEqual([528]);
    });

  it("lowers CUDA warp vote helpers to subgroup predicates", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void voteKernel(uint *input, uint *out) {
    uint mask = 0xffffffffu;
    out[0] = __any_sync(mask, input[0]);
    out[1] = __all_sync(mask, input[1]);
    out[2] = __ballot_sync(mask, input[0]);
    out[3] = __popc(out[2]);
    out[4] = __reduce_add_sync(mask, input[0]);
  }`, {
        features: { subgroups: true },
        workgroupSize: [1, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { input: new Uint32Array([7, 0]), out: new Uint32Array(5) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("subgroupAny");
      expect(compiled.wgsl).toContain("subgroupAll");
      expect(compiled.wgsl).toContain("bg_semantic_ballot_32");
      expect(compiled.wgsl).toContain("subgroupAdd(input[0u])");
      expect(compiled.wgsl).toContain("countOneBits");
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(backendIr(compiled).requiredFeatures).toContain("subgroups");
      expect([...result.buffers.out as Uint32Array]).toEqual([1, 0, 1, 1, 7]);
    });

  it("lowers CUDA warp vote and reduce helpers through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void semanticVoteKernel(uint *input, uint *out) {
    int tid = threadIdx.x;
    uint mask = 0xffffffffu;
    out[tid * 4] = __any_sync(mask, input[tid]);
    out[tid * 4 + 1] = __all_sync(mask, input[tid]);
    out[tid * 4 + 2] = __ballot_sync(mask, input[tid]);
    out[tid * 4 + 3] = __reduce_add_sync(mask, input[tid]);
  }`, {
        features: { subgroups: true },
        workgroupSize: [4, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { input: new Uint32Array([0, 1, 0, 1]), out: new Uint32Array(16) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { input: new Uint32Array([0, 1, 0, 1]), out: new Uint32Array(16) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("subgroupAny");
      expect(compiled.wgsl).toContain("subgroupAll");
      expect(compiled.wgsl).toContain("bg_semantic_ballot_32");
      expect(compiled.wgsl).toContain("subgroupAdd");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        1, 0, 10, 2,
        1, 0, 10, 2,
        1, 0, 10, 2,
        1, 0, 10, 2,
      ]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([
        1, 0, 10, 2,
        1, 0, 10, 2,
        1, 0, 10, 2,
        1, 0, 10, 2,
      ]);
    });

  it("lowers legacy CUDA warp vote aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void legacyVoteKernel(uint *input, uint *out) {
    int tid = threadIdx.x;
    out[tid * 3] = __any(input[tid]);
    out[tid * 3 + 1] = __all(input[tid]);
    out[tid * 3 + 2] = __ballot(input[tid]);
  }`, {
        features: { subgroups: true },
        workgroupSize: [4, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { input: new Uint32Array([0, 1, 0, 1]), out: new Uint32Array(12) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { input: new Uint32Array([0, 1, 0, 1]), out: new Uint32Array(12) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("subgroupAny");
      expect(compiled.wgsl).toContain("subgroupAll");
      expect(compiled.wgsl).toContain("bg_semantic_ballot_32");
      expect(backendIr(compiled).requiredFeatures).toContain("subgroups");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        1, 0, 10,
        1, 0, 10,
        1, 0, 10,
        1, 0, 10,
      ]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([
        1, 0, 10,
        1, 0, 10,
        1, 0, 10,
        1, 0, 10,
      ]);
    });

  it("lowers CUDA warp shuffle helpers through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void semanticShuffleKernel(uint *out) {
    int tid = threadIdx.x;
    uint mask = 0xffffffffu;
    uint value = uint(tid) + 10u;
    out[tid * 4] = __shfl_sync(mask, value, 2, 4);
    out[tid * 4 + 1] = __shfl_down_sync(mask, value, 1, 4);
    out[tid * 4 + 2] = __shfl_up_sync(mask, value, 1, 4);
    out[tid * 4 + 3] = __shfl_xor_sync(mask, value, 1, 4);
  }`, {
        features: { subgroups: true },
        workgroupSize: [4, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(16) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(16) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_sync_uint_4");
      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_down_uint_4");
      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_up_uint_4");
      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_xor_uint_4");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        12, 11, 10, 11,
        12, 12, 10, 10,
        12, 13, 11, 13,
        12, 13, 12, 12,
      ]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([
        12, 11, 10, 11,
        12, 12, 10, 10,
        12, 13, 11, 13,
        12, 13, 12, 12,
      ]);
    });

  it("lowers legacy CUDA warp shuffle aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void legacyShuffleKernel(uint *out) {
    int tid = threadIdx.x;
    uint value = uint(tid) + 10u;
    out[tid * 4] = __shfl(value, 2, 4);
    out[tid * 4 + 1] = __shfl_down(value, 1, 4);
    out[tid * 4 + 2] = __shfl_up(value, 1, 4);
    out[tid * 4 + 3] = __shfl_xor(value, 1, 4);
  }`, {
        features: { subgroups: true },
        workgroupSize: [4, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(16) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(16) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_sync_uint_4");
      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_down_uint_4");
      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_up_uint_4");
      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_xor_uint_4");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        12, 11, 10, 11,
        12, 12, 10, 10,
        12, 13, 11, 13,
        12, 13, 12, 12,
      ]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([
        12, 11, 10, 11,
        12, 12, 10, 10,
        12, 13, 11, 13,
        12, 13, 12, 12,
      ]);
    });

  it("lowers CUDA min/max subgroup reductions through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void semanticMinMaxReduceKernel(uint *input, uint *out) {
    int tid = threadIdx.x;
    uint mask = 0xffffffffu;
    out[tid * 2] = __reduce_min_sync(mask, input[tid]);
    out[tid * 2 + 1] = __reduce_max_sync(mask, input[tid]);
  }`, {
        features: { subgroups: true },
        workgroupSize: [4, 1, 1],
      });
      const launch = { gridDim: [1, 1, 1], blockDim: [4, 1, 1] } as const;
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { input: new Uint32Array([9, 4, 7, 2]), out: new Uint32Array(8) } },
        launch,
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { input: new Uint32Array([9, 4, 7, 2]), out: new Uint32Array(8) } },
        launch,
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("subgroupMin");
      expect(compiled.wgsl).toContain("subgroupMax");
      expect([...result.buffers.out as Uint32Array]).toEqual([2, 9, 2, 9, 2, 9, 2, 9]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([2, 9, 2, 9, 2, 9, 2, 9]);
    });

  it("lowers CUDA bitwise subgroup reductions through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void semanticBitwiseReduceKernel(uint *input, uint *out) {
    int tid = threadIdx.x;
    uint mask = 0xffffffffu;
    out[tid * 3] = __reduce_and_sync(mask, input[tid]);
    out[tid * 3 + 1] = __reduce_or_sync(mask, input[tid]);
    out[tid * 3 + 2] = __reduce_xor_sync(mask, input[tid]);
  }`, {
        features: { subgroups: true },
        workgroupSize: [4, 1, 1],
      });
      const launch = { gridDim: [1, 1, 1], blockDim: [4, 1, 1] } as const;
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { input: new Uint32Array([0b1111, 0b1100, 0b1010, 0b0011]), out: new Uint32Array(12) } },
        launch,
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { input: new Uint32Array([0b1111, 0b1100, 0b1010, 0b0011]), out: new Uint32Array(12) } },
        launch,
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bg_semantic_reduce_and_uint_32");
      expect(compiled.wgsl).toContain("bg_semantic_reduce_or_uint_32");
      expect(compiled.wgsl).toContain("bg_semantic_reduce_xor_uint_32");
      expect([...result.buffers.out as Uint32Array]).toEqual([0, 15, 10, 0, 15, 10, 0, 15, 10, 0, 15, 10]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([0, 15, 10, 0, 15, 10, 0, 15, 10, 0, 15, 10]);
    });

  it("lowers CUDA syncthreads predicate collectives through native workgroup memory", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void syncPredicates(int *out) {
    int tid = threadIdx.x;
    int count = __syncthreads_count(tid < 3);
    int all = __syncthreads_and(tid < 4);
    int any = __syncthreads_or(tid == 2);
    out[tid * 3] = count;
    out[tid * 3 + 1] = all;
    out[tid * 3 + 2] = any;
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(12) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(compiled.wgsl).toContain("bg_warp_reduce_sum_uint_4");
      expect(compiled.wgsl).toContain("workgroupBarrier()");
      expect([...result.buffers.out as Int32Array]).toEqual([
        3, 1, 1,
        3, 1, 1,
        3, 1, 1,
        3, 1, 1,
      ]);
    });

  it("lowers CUDA activemask to native WebGPU subgroup ballot", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void activeMaskKernel(uint *out) {
    uint mask = __activemask();
    out[threadIdx.x] = mask;
  }`, {
        features: { subgroups: true },
        workgroupSize: [4, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("// browsergrad-semantic-wgsl: direct semantic IR emission");
      expect(compiled.wgsl).toContain("enable subgroups;");
      expect(compiled.wgsl).toContain("subgroupBallot(true).x");
      expect(backendIr(compiled).requiredFeatures).toContain("subgroups");
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-subgroup");
      expect([...result.buffers.out as Uint32Array]).toEqual([15, 15, 15, 15]);
    });

  it("runs CUDA activemask scalar subgroup fallback in reference mode", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void activeMaskScalar(uint *out) {
    out[0] = __activemask();
  }`, {
        features: { subgroups: true },
        subgroupMode: "scalar",
        workgroupSize: [1, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("out[0] = u32(1u)");
      expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { out: new Uint32Array(1) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
      expect([...result.buffers.out as Uint32Array]).toEqual([1]);
    });

  it("lowers CUDA warp vote helpers with boolean predicates", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void voteBoolKernel(bool *info, int warp_size) {
    int tx = threadIdx.x;
    bool *offs = info + (tx * 3);
    *offs = __any_sync(0xffffffffu, (tx >= (warp_size * 3) / 2));
    *(offs + 1) = (tx >= (warp_size * 3) / 2 ? true : false);
    if (__all_sync(0xffffffffu, (tx >= (warp_size * 3) / 2))) {
      *(offs + 2) = true;
    }
  }`, { features: { subgroups: true }, subgroupMode: "scalar", workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl).toContain("select(0u, 1u, (tx >= ((bg_uniforms.warp_size * 3) / 2)))");
      expect(compiled.wgsl).not.toContain(") != 0) != 0");
    });

  it("runs CUDA warp vote helpers as subgroup collectives in reference mode", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void voteBoolKernel(bool *info, int warp_size) {
    int tx = threadIdx.x;
    bool *offs = info + (tx * 3);
    *offs = __any_sync(0xffffffffu, (tx >= (warp_size * 3) / 2));
    *(offs + 1) = (tx >= (warp_size * 3) / 2 ? true : false);
  }`, { features: { subgroups: true }, workgroupSize: [32, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { info: new Uint32Array(96) }, scalars: { warp_size: 16 } },
        { gridDim: [1, 1, 1], blockDim: [32, 1, 1] },
      );

      expect((result.buffers.info as Uint32Array)[0]).toBe(1);
      expect((result.buffers.info as Uint32Array)[1]).toBe(0);
      expect((result.buffers.info as Uint32Array)[72]).toBe(1);
      expect((result.buffers.info as Uint32Array)[73]).toBe(1);
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
      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_down_float_16(val, u32(offset), 16u, local_id)");
      expect(compiled.wgsl).toContain("i32((local_id.x) % 16u)");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect([...result.buffers.output as Float32Array]).toEqual([16]);
    });

  it("uniformly lowers logical-tile shuffles guarded to the first tile", () => {
      const compiled = compileCudaLiteKernel(`
  namespace cg = cooperative_groups;
  __global__ void firstTileReduce(const float *input, float *output) {
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    int tid = block.thread_rank();
    float val = input[tid];
    if (tid < 32) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        val += tile32.shfl_down(val, offset);
      }
      if (tid == 0) { output[0] = val; }
    }
  }`, {
        features: { subgroups: true },
        workgroupSize: [64, 1, 1],
      });
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { input: new Float32Array(64).fill(1), output: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [64, 1, 1] },
      );

      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("let bg_collective_");
      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_down_float_32(val, u32(offset), 32u, local_id)");
      expect([...result.buffers.output as Float32Array]).toEqual([32]);
    });

  it("accepts const-qualified cooperative-group declarations", () => {
      const compiled = compileCudaLiteKernel(`
  namespace cg = cooperative_groups;
  __global__ void constGroup(int *out) {
    const cg::thread_block block = cg::this_thread_block();
    const auto tile = cg::tiled_partition<4>(block);
    out[threadIdx.x] = tile.thread_rank();
  }`, { features: { subgroups: true }, workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect([...result.buffers.out as Int32Array]).toEqual([0, 1, 2, 3]);
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

      expect(compiled.wgsl).toContain("fn block_rank(block__bg_group_rank: i32, block__bg_group_size: i32");
      expect(compiled.wgsl).toContain("fn tile_rank(tile__bg_group_rank: i32, tile__bg_group_size: i32");
      expect([...result.buffers.out as Int32Array]).toEqual([0, 2]);
    });

  it("lowers shared half2 arrays through device helper pointers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void pair_reduce(half2 *values) {
    if (threadIdx.x == 0) values[0] = values[0] + values[1];
    __syncthreads();
  }
  __global__ void half2_shared_reduce(const half2 *input, half2 *out) {
    __shared__ half2 tile[2];
    tile[threadIdx.x] = input[threadIdx.x];
    __syncthreads();
    pair_reduce(tile);
    if (threadIdx.x == 0) out[0] = tile[0];
  }`, { f16Mode: "f32", workgroupSize: [2, 1, 1] });

      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("var<workgroup> tile: array<vec2<f32>, 2>;");
      expect(compiled.wgsl).toContain("ptr<workgroup, array<vec2<f32>, 2>>");
    });

  it("rejects divergent calls to helpers that contain barriers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void sync_tile(float *tile) { __syncthreads(); }
  __global__ void divergent_helper_barrier(float *out) {
    __shared__ float tile[1];
    if (threadIdx.x == 0) sync_tile(tile);
    out[threadIdx.x] = 1.0f;
  }`, { workgroupSize: [2, 1, 1] });

      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(false);
      expect(semanticKernelIrWgslPreflightBlocker(compiled.kernelIr)).toBe("semantic WGSL does not support shared-memory barrier shape");
      expect(semanticKernelIrWgslPreflightFailure(compiled.kernelIr)).toMatchObject({
        message: "semantic WGSL does not support shared-memory barrier shape",
        span: { line: 3, column: 3 },
      });
    });

  it("classifies grid-wide cooperative sync as host-orchestrated WebGPU lowering", () => {
      const analysis = analyzeCudaLite(parseCudaLite(`
  namespace cg = cooperative_groups;
  __global__ void gridSync(float *x) {
    cg::grid_group grid = cg::this_grid();
    grid.sync();
    if (threadIdx.x < 1) { x[0] = 1.0f; }
  }`));

      expect(analysis.diagnostics).toContainEqual(expect.objectContaining({
        code: "cuda-grid-sync-host-orchestration",
        severity: "warning",
      }));
    });

  it("compiles host-orchestratable grid sync for WebGPU planning", () => {
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
      const directCompiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
      expect(directCompiled.loweringPlan).toMatchObject({
        canDirectLowerToWgsl: false,
        requiresGpuPolyfill: true,
        unsupported: [],
      });
      expect(cudaLiteWebGpuCompileOptions({ referenceGridSync: false })).toMatchObject({
        referenceDynamicParallelism: true,
        referenceGridSync: true,
        referenceCudaRuntime: true,
      });

      const compiled = compileCudaLiteKernelForWebGpu(source, { workgroupSize: [1, 1, 1] });
      expect(compiled.diagnostics).toContainEqual(expect.objectContaining({
        code: "cuda-grid-sync-host-orchestration",
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

  it("emits semantic grid phases with dynamic shared memory and cooperative rank queries", () => {
      const compiled = compileCudaLiteKernel(`
  namespace cg = cooperative_groups;
  __global__ void gridSync(float *out) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    int rank = grid.thread_rank();
    int total = grid.size();
    scratch[tid] = rank < total ? 1.0f : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) scratch[tid] += scratch[tid + stride];
      __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = scratch[0];
    grid.sync();
    if (blockIdx.x == 0) {
      scratch[tid] = tid < gridDim.x ? out[tid] : 0.0f;
      __syncthreads();
      if (tid == 0) out[0] = scratch[0];
    }
  }`,
      { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 16 } });

      const phases = createCudaGridSyncPhasePlan(compiled);
      expect(phases.supported).toBe(true);
      if (!phases.supported) return;
      expect(compiled.kernelIr.memory).toContainEqual(expect.objectContaining({
        name: "scratch",
        addressSpace: "shared",
        dimensions: [16],
      }));
      const wgsl = phases.phases.map((phase) => emitSemanticKernelIrWgsl(phase).wgsl).join("\n");
      expect(wgsl).toContain("var<workgroup> scratch: array<f32, 16>;");
      expect(wgsl).toContain("stride >>= 1");
      expect(wgsl).toContain("num_workgroups.x");
    });

  it("plans grid sync phases through shared cooperative reduction helpers", () => {
      const compiled = compileCudaLiteKernel(`
  namespace cg = cooperative_groups;
  __device__ void reduceBlock(double *sdata, const cg::thread_block &cta) {
    const unsigned int tid = cta.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    sdata[tid] = cg::reduce(tile32, sdata[tid], cg::plus<double>());
    cg::sync(cta);
    double beta = 0.0;
    if (cta.thread_rank() == 0) {
      for (int i = 0; i < blockDim.x; i += tile32.size()) beta += sdata[i];
      sdata[0] = beta;
    }
    cg::sync(cta);
  }
  __global__ void helperGridSync(const float *input, float *out, unsigned int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    extern double __shared__ sdata[];
    sdata[block.thread_rank()] = input[grid.thread_rank()];
    reduceBlock(sdata, block);
    if (block.thread_rank() == 0) out[blockIdx.x] = sdata[0];
    cg::sync(grid);
    if (grid.thread_rank() == 0) out[0] += n - n;
  }`, {
        f64Mode: "f32",
        dynamicSharedMemory: { sdata: 32 },
        workgroupSize: [32, 1, 1],
      });
      const phases = createCudaGridSyncPhasePlan(compiled);
      const plan = createCudaWebGpuExecutionPlan(
        compiled,
        { buffers: { input: new Float32Array(32), out: new Float32Array(1) }, scalars: { n: 32 } },
        { gridDim: [1, 1, 1], blockDim: [32, 1, 1] },
      );

      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(false);
      expect(phases).toMatchObject({ supported: true });
      if (phases.supported) {
        expect(phases.phases).toHaveLength(2);
        expect(canEmitSemanticKernelIrWgsl(phases.phases[0]!)).toBe(true);
        const phaseWgsl = emitSemanticKernelIrWgsl(phases.phases[0]!).wgsl;
        expect(phaseWgsl).toContain("fn reduceBlock");
        expect(phaseWgsl).not.toContain("i32(local_id.x) == 0.0");
      }
      expect(plan).toMatchObject({ supported: true, kind: "grid-sync-phases" });
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

      expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(false);
      expect(compiled.loweringPlan.requiresGpuPolyfill).toBe(true);
      expect(compiled.loweringPlan.unsupported).toEqual([]);
      expect(compiled.diagnostics).toContainEqual(expect.objectContaining({
        code: "cuda-grid-sync-host-orchestration",
        severity: "warning",
      }));
      expect([...result.buffers.out as Float32Array]).toEqual([3]);
      expect(createCudaGridSyncPhasePlan(compiled).supported).toBe(true);
      const webGpuPlan = createCudaWebGpuExecutionPlan(compiled, input, launch);
      expect(summarizeCudaWebGpuExecutionPlan(webGpuPlan)).toMatchObject({
        canRunOnWebGpu: true,
        mode: "host-orchestrated",
        kind: "grid-sync-phases",
        requiresHostOrchestration: true,
      });
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
      const plan = createCudaGridSyncPhasePlan(compiled);
      const detachedPlan = createCudaGridSyncPhasePlan({ ...compiled });

      expect(plan.supported).toBe(true);
      expect(detachedPlan.supported).toBe(true);
      if (plan.supported) {
        expect(plan.phases).toHaveLength(2);
        expect(plan.phases.map((phase) => phase.name)).toEqual([
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

  it("lowers grid cooperative group rank and size across all blocks", () => {
      const compiled = compileCudaLiteKernel(`
  namespace cg = cooperative_groups;
  __global__ void gridRankSize(int *out) {
    cg::grid_group grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = grid.size() * 10 + grid.thread_rank();
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(4) } },
        { gridDim: [2, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("num_workgroups.x");
      expect(compiled.wgsl).toContain("workgroup_id.x");
      expect([...result.buffers.out as Int32Array]).toEqual([40, 41, 42, 43]);
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
      const plan = createCudaGridSyncPhasePlan(compiled);

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
      const plan = createCudaGridSyncPhasePlan(compiled);

      expect(plan.supported).toBe(true);
      if (plan.supported) expect(plan.phases).toHaveLength(2);
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
      const plan = createCudaGridSyncPhasePlan(compiled);

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

      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_up_float_8(val, 1u, 8u, local_id)");
      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_xor_float_8(val, 2u, 8u, local_id)");
      expect(compiled.wgsl).toContain("i32((local_id.x + local_id.y * 8u) % 8u)");
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
  }`, { workgroupSize: [8, 1, 1] });
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

      expect(compiled.wgsl).toContain("bg_warp_reduce_sum_float_8(value, 8u, local_id)");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect(compiled.wgsl).not.toContain("enable subgroups;");
      expect(backendIr(compiled).requiredFeatures).not.toContain("subgroups");
      expect([...result.buffers.output as Float32Array]).toEqual([8]);
    });

  it("lowers cooperative-group inclusive and exclusive tile scans", () => {
      const compiled = compileCudaLiteKernel(`
  namespace cg = cooperative_groups;
  __global__ void tileScan(const int *input, int *output) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(block);
    int lane = tile.thread_rank();
    int value = input[threadIdx.x];
    int inclusive = cg::inclusive_scan(tile, value);
    int exclusive = cg::exclusive_scan(tile, value, cg::plus<int>());
    output[threadIdx.x * 2] = inclusive;
    output[threadIdx.x * 2 + 1] = exclusive + lane;
    output[8 + threadIdx.x] = cg::inclusive_scan(block, value);
  }`, {
        workgroupSize: [4, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            input: new Int32Array([1, 2, 3, 4]),
            output: new Int32Array(12),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("fn bg_cg_inclusive_scan_sum_int_4");
      expect(compiled.wgsl).toContain("fn bg_cg_exclusive_scan_sum_int_4");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect([...result.buffers.output as Int32Array]).toEqual([1, 0, 3, 2, 6, 5, 10, 9, 1, 3, 6, 10]);
    });

  it("lowers cooperative-group binary partitions to predicate masks", () => {
      const compiled = compileCudaLiteKernel(`
  namespace cg = cooperative_groups;
  __global__ void binaryPartition(int *input, int *out) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
    int value = input[threadIdx.x];
    auto part = cg::binary_partition(tile, (value & 1) != 0);
    int sum = cg::reduce(part, value, cg::plus<int>());
    if (part.thread_rank() == 0) {
      out[0] = part.size() + sum;
    }
  }`, {
        features: { subgroups: true },
        workgroupSize: [1, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { input: new Int32Array([3]), out: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(backendIr(compiled).requiredFeatures).toContain("subgroups");
      expect(compiled.wgsl).toContain("subgroupBallot");
      expect(compiled.wgsl).toContain("countOneBits");
      expect(compiled.wgsl).toContain("bg_warp_partition_reduce_sum_int_1(value");
      expect(compiled.wgsl).not.toContain("!= 0) != 0");
      expect([...result.buffers.out as Int32Array]).toEqual([4]);
    });

  it("passes tile cooperative groups through generic device helper reduce params", () => {
      const compiled = compileCudaLiteKernel(`
  namespace cg = cooperative_groups;
  __device__ int reduce_n(int value, cooperative_groups::thread_group tile) {
    return cg::reduce(tile, value, cg::plus<int>());
  }
  __global__ void helperTileReduce(int *out) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(block);
    int value = 1;
    int sum = reduce_n(value, tile);
    if (tile.thread_rank() == 0) { out[0] = sum; }
  }`, {
        features: { subgroups: true },
        workgroupSize: [4, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("reduce_n(value, tile__bg_group_rank, tile__bg_group_size");
      expect(compiled.wgsl).toContain("bg_semantic_cg_reduce_i32_4(value, local_id)");
      expect([...result.buffers.out as Int32Array]).toEqual([4]);
    });

  it("lowers custom vector cooperative reductions through workgroup tile reductions", () => {
      const compiled = compileCudaLiteKernel(`
  namespace cg = cooperative_groups;
  __device__ float2 merge_pair(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
  }
  __global__ void vectorTileReduce(float2 *out) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
    float2 value = out[0];
    float2 total = cg::reduce(tile, value, merge_pair);
    if (threadIdx.x == 0) { out[0] = total; }
  }`, {
        features: { subgroups: true },
        workgroupSize: [1, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array([2, 3]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("fn bg_cg_reduce_merge_pair_float2_32");
      expect(compiled.wgsl).toContain("var<workgroup> bg_cg_reduce_merge_pair_float2_32_scratch");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect(compiled.wgsl).toContain("merge_pair(bg_cg_reduce_merge_pair_float2_32_scratch[bg_linear_rank]");
      const mergeFnIndex = compiled.wgsl.indexOf("fn merge_pair(");
      const reduceHelperIndex = compiled.wgsl.indexOf("fn bg_cg_reduce_merge_pair_float2_32");
      expect(mergeFnIndex).toBeGreaterThanOrEqual(0);
      expect(mergeFnIndex).toBeLessThan(reduceHelperIndex);
      expect([...result.buffers.out as Float32Array]).toEqual([2, 3]);
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
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Int32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
      );

      expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("/ 4u");
      expect([...result.buffers.out as Int32Array]).toEqual([20, 120, 220, 320, 21, 121, 221, 321]);
      expect([...semanticResult.buffers.out as Int32Array]).toEqual([20, 120, 220, 320, 21, 121, 221, 321]);
    });

  it("proves uniform member barriers inside shared reduction loops", () => {
      const compiled = compileCudaLiteKernel(`
  namespace cg = cooperative_groups;
  __global__ void memberBarrierLoop(int *out) {
    __shared__ int values[8];
    cg::thread_block block = cg::this_thread_block();
    uint rank = block.thread_rank();
    values[rank] = 1;
    uint limit = block.size() >> 1;
    while (limit >= 1) {
      block.sync();
      if (rank < limit) values[rank] += values[rank + limit];
      limit >>= 1;
    }
    block.sync();
    if (rank == 0) out[0] = values[0];
  }`, { workgroupSize: [8, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
      );

      expect(compiled.kernelIr.barrierUniformity.kernel).toMatchObject({ verified: true });
      expect(compiled.kernelIr.barrierUniformity.kernel.barrierStatementStarts).toHaveLength(2);
      expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect([...semanticResult.buffers.out as Int32Array]).toEqual([8]);
    });

  it("lowers CUDA shuffle, fence, and conversion intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void intrinsic_pack(half2 *h, float2 *f, float *out, uint *bits) {
    int lane = __shfl_sync(0xffffffff, threadIdx.x, 0);
    __syncwarp(0xffffffff);
    __threadfence();
    __threadfence_block();
    __threadfence_system();
    __nanosleep(8);
    __prof_trigger(1);
    half2 value = make_half2(__int2half_rn(lane + 1), __int2half_rn(4));
    h[0] = value;
    bits[0] = __half2_as_uint(value);
    h[1] = __uint_as_half2(0x40003c00u);
    f[0] = __half22float2(value);
    out[0] = __fmaf_rn(f[0].x, 2.0f, f[0].y);
  }`, { features: { "shader-f16": true, subgroups: true }, workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            h: createWgslFloat16Array(4),
            f: new Float32Array(2),
            out: new Float32Array(1),
            bits: new Uint32Array(1),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(backendIr(compiled).requiredFeatures).toEqual(expect.arrayContaining(["shader-f16", "subgroups"]));
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bg_semantic_warp_shuffle_sync_uint_32(0u, 0u, 32u, local_id)");
      expect(compiled.wgsl).toContain("workgroupBarrier()");
      expect(compiled.wgsl).toContain("storageBarrier()");
      expect(compiled.wgsl).toContain("pack2x16float(vec2<f32>(f32((value).x), f32((value).y)))");
      expect(compiled.wgsl).toContain("vec2<f16>(unpack2x16float(");
      expect([...result.buffers.f as Float32Array]).toEqual([1, 4]);
      expect(Array.from(result.buffers.h as Iterable<number>)).toEqual([1, 4, 1, 2]);
      expect([...result.buffers.bits as Uint32Array]).toEqual([0x44003c00]);
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

      expect(backendIr(compiled).requiredFeatures).toEqual(expect.arrayContaining(["shader-f16", "subgroups"]));
      expect(compiled.wgsl).toContain("bg_warp_reduce_sum_float_32(x[i], 32u, local_id)");
      expect(compiled.wgsl).toContain("bg_warp_reduce_max_float_32(sum, 32u, local_id)");
      expect(compiled.wgsl).toContain("bg_warp_reduce_min_float_32(maxv, 32u, local_id)");
      expect(compiled.wgsl).toContain("vec2<f16>");
      expect(Array.from(result.buffers.h as ArrayLike<number>)).toEqual([4, 5, 3, 3]);
      expect([...result.buffers.out as Float32Array]).toEqual([3.5]);
    });

  it("lowers integer warp reduction aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void reduce_int_alias(const int *x, int *out) {
    int i = threadIdx.x;
    int sum = warp_reduce_sum_i8_i32(x[i]);
    out[i] = warp_reduce_sum_i32_i32(sum);
  }`, { features: { subgroups: true }, workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            x: new Int32Array([7]),
            out: new Int32Array(1),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(backendIr(compiled).requiredFeatures).toContain("subgroups");
      expect(compiled.wgsl).toContain("bg_semantic_warp_reduce_sum_i32_32(x[u32(i)], local_id)");
      expect([...result.buffers.out as Int32Array]).toEqual([7]);
    });

  it("lowers output-only inline PTX warp id statements", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void warpId(uint *out) {
    int idx = threadIdx.x;
    unsigned int warp;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp));
    out[idx] = warp;
  }`, { workgroupSize: [64, 1, 1] });
      const input = { buffers: { out: new Uint32Array(64) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [64, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("/ 32u");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([
        ...new Array(32).fill(0),
        ...new Array(32).fill(1),
      ]);
      expect([...result.buffers.out as Uint32Array]).toEqual([
        ...new Array(32).fill(0),
        ...new Array(32).fill(1),
      ]);
    });

  it("lowers inline PTX cp.async fences as native no-op barriers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void cpAsyncFenceAsm(float *out, float *in) {
    int wait = 0;
    asm volatile("cp.async.commit_group;\\n" ::);
    asm volatile("cp.async.wait_group 0;\\n" ::);
    asm volatile("cp.async.wait_group %0;\\n" :: "n"(wait));
    asm volatile("cp.async.wait_all;\\n" ::);
    out[threadIdx.x] = in[threadIdx.x] + 1.0f;
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2), in: new Float32Array([2, 4]) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("cp.async inline asm fence omitted");
      expect([...result.buffers.out as Float32Array]).toEqual([3, 5]);
    });

  it("lowers inline PTX membar fences as native WebGPU storage barriers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void membarAsm(float *out, float *in) {
    asm volatile("membar.cta;\\n" ::);
    asm volatile("membar.gl;\\n" ::);
    asm volatile("membar.sys;\\n" ::);
    out[threadIdx.x] = in[threadIdx.x] + 1.0f;
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2), in: new Float32Array([6, 8]) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("storageBarrier();");
      expect([...result.buffers.out as Float32Array]).toEqual([7, 9]);
    });

  it("lowers inline PTX bar.sync as native WebGPU workgroup barriers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void barSyncAsm(float *out, float *in) {
    int barrier = 0;
    asm volatile("bar.sync 0;\\n" ::);
    asm volatile("bar.sync %0;\\n" :: "r"(barrier));
    out[threadIdx.x] = in[threadIdx.x] + 1.0f;
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2), in: new Float32Array([10, 12]) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect([...result.buffers.out as Float32Array]).toEqual([11, 13]);
    });

  it("rejects divergent inline PTX bar.sync barriers", () => {
      const analysis = analyzeCudaLite(parseCudaLite(`
  __global__ void divergentBarSync(float *out) {
    if (threadIdx.x == 0) {
      asm volatile("bar.sync 0;\\n" ::);
    }
    out[threadIdx.x] = 1.0f;
  }`));

      expect(analysis.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-barrier");
    });

  it("ignores divergent barriers in unreachable helpers for selected kernels", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void unused_divergent_barrier(float *x) {
    if (threadIdx.x == 0) {
      return;
    }
    __syncthreads();
    x[threadIdx.x] = 0.0f;
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
      expect(compiled.wgsl).not.toContain("unused_divergent_barrier");
    });

  it("ignores grid sync compatibility calls in unreachable helpers for selected kernels", () => {
      const compiled = compileCudaLiteKernel(`
  namespace cg = cooperative_groups;

  __device__ void unused_grid_sync() {
    cg::grid_group grid = cg::this_grid();
    grid.sync();
    cg::sync(grid);
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
      expect(compiled.wgsl).not.toContain("unused_grid_sync");
    });

  it("does not route block cooperative sync through grid-sync phase planning", () => {
      const compiled = compileCudaLiteKernel(`
  namespace cg = cooperative_groups;
  __global__ void child(float *dst) {
    if (threadIdx.x == 0) { dst[0] += 1.0f; }
  }
  __global__ void parent(float *x) {
    cg::thread_block cta = cg::this_thread_block();
    cg::sync(cta);
    if (threadIdx.x == 0) {
      child<<<1, 1>>>(x);
      cudaDeviceSynchronize();
    }
  }`, {
        kernelName: "parent",
        workgroupSize: [1, 1, 1],
      });
      const input = { buffers: { x: new Float32Array([1]) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };

      expect(createCudaRuntimePlan(compiled).operations.map((operation) => operation.kind)).toEqual([
        "device-launch",
        "device-sync",
      ]);
      expect(createCudaWebGpuExecutionPlan(compiled, input, launch, {
        compileKernel: compileCudaLiteKernel,
      })).toMatchObject({
        supported: true,
        kind: "host-dynamic-launch",
      });
    });

  it("uniformizes simple predicated barriers for WGSL validation", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void predicatedBarrier(float *A, float *B, float *C, int N) {
    extern __shared__ float sharedData[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      sharedData[threadIdx.x] = A[idx];
      __syncthreads();
      C[idx] = sharedData[threadIdx.x] + B[idx];
    }
  }`, { workgroupSize: [2, 1, 1], dynamicSharedMemory: { sharedData: 2 } });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            A: new Float32Array([1, 2]),
            B: new Float32Array([10, 20]),
            C: new Float32Array(2),
          },
          scalars: { N: 2 },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("workgroupBarrier();\n    if ((idx < bg_uniforms.N))");
      expect([...result.buffers.C as Float32Array]).toEqual([11, 22]);
    });

  it("uniformizes namespace cooperative-group sync inside predicated regions", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void predicatedCoopNamespaceSync(float *x, int N) {
    cg::thread_block cta = cg::this_thread_block();
    int tid = threadIdx.x;
    if (tid < N) {
      x[tid] = x[tid] + 1.0f;
      cg::sync(cta);
      x[tid] = x[tid] + 2.0f;
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
      expect(compiled.wgsl).toContain("workgroupBarrier();\n    if ((tid < bg_uniforms.N))");
      expect(compiled.wgsl).not.toContain("if ((tid < bg_uniforms.N)) {\n    x[u32(tid)] = (x[u32(tid)] + 1.0);\n    workgroupBarrier();");
    });

  it("uniformizes member cooperative-group sync inside predicated regions", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void predicatedCoopMemberSync(float *x, int N) {
    cg::thread_block cta = cg::this_thread_block();
    int tid = threadIdx.x;
    if (tid < N) {
      x[tid] = x[tid] + 1.0f;
      cta.sync();
      x[tid] = x[tid] + 2.0f;
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
      expect(compiled.wgsl).toContain("workgroupBarrier();\n    if ((tid < bg_uniforms.N))");
      expect(compiled.wgsl).not.toContain("if ((tid < bg_uniforms.N)) {\n    x[u32(tid)] = (x[u32(tid)] + 1.0);\n    workgroupBarrier();");
    });

  it("uniformizes barrier device helpers inside predicated regions", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void helper_with_barrier(uint *out, uint value) {
    __syncthreads();
    out[threadIdx.x] = value + (uint)threadIdx.x;
    __syncthreads();
  }
  __global__ void predicatedBarrierHelper(uint *out) {
    __shared__ uint ready;
    if (threadIdx.x == 0) {
      atomicExch(&ready, 1u);
    }
    __syncthreads();
    if (ready == 1u) {
      helper_with_barrier(out, 7u);
      out[threadIdx.x + 4] = 9u;
    }
    __syncthreads();
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.wgsl).toContain("helper_with_barrier__bg_guarded_barrier");
      expect(compiled.wgsl).toContain("bg_call_active: bool");
      expect(compiled.wgsl).toContain("workgroupBarrier();\n  if (bg_call_active)");
      expect(compiled.wgsl).not.toContain("if ((atomicLoad(&ready) == 1u)) {\n    helper_with_barrier(");
    });

  it("keeps uniform shared-memory barrier helper calls as direct calls", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ void sharedBarrierHelper(float *out) {
    __shared__ float tile[4];
    tile[threadIdx.x] = out[threadIdx.x];
    __syncthreads();
    out[threadIdx.x] = tile[threadIdx.x] + 1.0f;
  }
  __global__ void sharedBarrierCaller(float *out) {
    sharedBarrierHelper(out);
  }`, { workgroupSize: [4, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array([1, 2, 3, 4]) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var<workgroup> tile: array<f32, 4>;");
      expect(compiled.wgsl).toContain("sharedBarrierHelper(0u, 0u, local_id, workgroup_id, num_workgroups);");
      expect(compiled.wgsl).not.toContain("bg_inline_sharedBarrierHelper");
      expect(compiled.kernelIr.barrierUniformity.functions.sharedBarrierHelper).toMatchObject({
        verified: true,
        barrierStatementStarts: [expect.any(Number)],
      });
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([2, 3, 4, 5]);
    });

  it("specializes shared-pointer helper calls inside declaration initializers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint readSharedOffset(uint *data, uint offset) {
    return data[offset];
  }
  __global__ void sharedInitializerCall(uint *out) {
    __shared__ uint values[2];
    uint tid = threadIdx.x;
    values[tid] = tid + 3u;
    __syncthreads();
    uint result = readSharedOffset(values + 1, 0u);
    out[tid] = result;
  }`, { workgroupSize: [2, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.kernelIr.functions.find((fn) => fn.name === "readSharedOffset")?.params[0]).toMatchObject({
        addressSpace: "shared",
        name: "data__bg_shared_ptr",
      });
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([4, 4]);
    });

  it("lowers multi-shared-pointer barrier helpers with cooperative-group params", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void mergeShared(
    uint *dstKey, uint *dstVal,
    uint *srcAKey, uint *srcAVal,
    uint *srcBKey, uint *srcBVal,
    cg::thread_block cta
  ) {
    uint tid = threadIdx.x;
    uint key = tid < 2u ? srcAKey[tid] : srcBKey[tid - 2u];
    uint value = tid < 2u ? srcAVal[tid] : srcBVal[tid - 2u];
    cg::sync(cta);
    dstKey[tid] = key;
    dstVal[tid] = value;
  }
  __global__ void sharedMergeHelper(uint *out) {
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint keys[4];
    __shared__ uint values[4];
    uint tid = threadIdx.x;
    keys[tid] = tid + 10u;
    values[tid] = tid + 20u;
    cg::sync(cta);
    mergeShared(keys, values, keys, values, keys + 2u, values + 2u, cta);
    out[tid] = keys[tid] + values[tid];
  }`, { workgroupSize: [4, 1, 1] });
      const input = { buffers: { out: new Uint32Array(4) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect([...runCompiledKernelSemanticReference(compiled, input, launch).buffers.out as Uint32Array]).toEqual([30, 32, 34, 36]);
      expect(compiled.wgsl).toContain("fn mergeShared(dstKey__bg_shared_ptr: ptr<workgroup, array<u32, 4>>");
      expect(compiled.wgsl).toContain("mergeShared(&keys, 0u, &values, 0u, 0u, 0u, 2u, 2u");
    });

  it("returns values from generic cooperative-group barrier helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ int sumReduction(thread_group group, int *workspace, int value) {
    int lane = group.thread_rank();
    for (int stride = group.size() / 2; stride > 0; stride /= 2) {
      workspace[lane] = value;
      group.sync();
      if (lane < stride) value += workspace[lane + stride];
      group.sync();
    }
    return lane == 0 ? value : -1;
  }
  __global__ void genericGroupReduction(int *out) {
    thread_block block = this_thread_block();
    extern __shared__ int workspace[];
    int result;
    int rank = block.thread_rank();
    result = sumReduction(block, workspace, rank);
    if (rank == 0) out[0] = result;
    block.sync();
    thread_block_tile<2> tile = tiled_partition<2>(block);
    int offset = rank - tile.thread_rank();
    result = sumReduction(tile, workspace + offset, tile.thread_rank());
    if (tile.thread_rank() == 0) out[1 + rank / 2] = result;
  }`, {
        workgroupSize: [4, 1, 1],
        dynamicSharedMemory: { workspace: 4 },
      });
      const input = { buffers: { out: new Int32Array(3) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.kernelIr.operations.filter((operation) => operation.kind === "call").map((operation) => operation.result?.name)).toEqual(["result", "result"]);
      expect([...runCompiledKernelSemanticReference(compiled, input, launch).buffers.out as Int32Array]).toEqual([6, 1, 1]);
      expect(compiled.wgsl).toContain("group__bg_group_rank: i32, group__bg_group_size: i32");
    });

  it("lowers early returns before later barriers into active-lane guards", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void earlyReturnBarrier(float *x, int N) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    if (tid >= N) {
      scratch[tid] = 0.0f;
      return;
    }
    scratch[tid] = x[tid];
    __syncthreads();
    x[tid] = scratch[tid] + 1.0f;
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: new Float32Array([1, 2, 3, 4]) }, scalars: { N: 4 } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toContain("var bg_active_lane: bool = true;");
      expect(compiled.wgsl).toContain("bg_active_lane = false;");
      expect(compiled.wgsl).toContain("workgroupBarrier();\n  if (bg_active_lane)");
      expect([...result.buffers.x as Float32Array]).toEqual([2, 3, 4, 5]);
    });

  it("keeps barriers uniform inside tiled loops after active-lane early returns", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void earlyReturnLoopBarrier(float *x, int N) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    if (tid >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < 2; ++k) {
      scratch[tid] = x[tid] + (float)k;
      __syncthreads();
      acc += scratch[tid];
      __syncthreads();
    }
    x[tid] = acc;
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: new Float32Array([1, 2, 3, 4]) }, scalars: { N: 4 } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("var bg_barrier_loop_active_");
      expect(compiled.wgsl).toContain("var bg_barrier_loop_active_");
      expect(compiled.wgsl).toContain(": bool = bg_active_lane;");
      expect(compiled.wgsl).toContain("for (var bg_barrier_loop_iter_");
      expect(compiled.wgsl).toContain("workgroupBarrier();\n    acc = select(acc, (acc + scratch[tid]), bg_barrier_loop_active_");
      expect(compiled.wgsl).toContain("k = select(k, (k + 1), bg_barrier_loop_active_");
      expect(compiled.wgsl).not.toContain("if (bg_active_lane) {\n  for");
      expect([...result.buffers.x as Float32Array]).toEqual([3, 5, 7, 9]);
    });

  it("lowers loop-internal returns before barriers into loop active-lane guards", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void loopInternalReturnBarrier(float *x, int N) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    for (int k = 0; k < 2; ++k) {
      int idx = tid + k * 4;
      if (idx >= N) return;
      scratch[tid] = x[idx] + (float)k;
      __syncthreads();
      x[idx] = scratch[tid] + 1.0f;
      __syncthreads();
    }
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toContain("var bg_barrier_loop_active_");
      expect(compiled.wgsl).toMatch(/bg_barrier_loop_active_\d+ = false;/u);
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect(compiled.wgsl).not.toContain("if ((idx >= bg_uniforms.N)) {\n      return;");
    });

  it("lowers alternate-branch returns before barriers into active-lane guards", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void alternateReturnBarrier(float *x, int N) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    if (tid < N) {
      scratch[tid] = x[tid];
    } else {
      return;
    }
    __syncthreads();
    x[tid] = scratch[tid] + 1.0f;
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: new Float32Array([1, 2, 3, 4]) }, scalars: { N: 4 } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toContain("var bg_active_lane: bool = true;");
      expect(compiled.wgsl).toContain("bg_active_lane = false;");
      expect(compiled.wgsl).toContain("if (bg_active_lane) {\n    scratch[tid] = f32(x[tid]);\n  }");
      expect(compiled.wgsl).not.toContain("return;");
      expect([...result.buffers.x as Float32Array]).toEqual([2, 3, 4, 5]);
    });

  it("lowers nested returns before barriers into active-lane guards", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void nestedReturnBarrier(float *x, int N) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    if (tid < N) {
      if ((tid + 1) < N) {
        scratch[tid] = x[tid];
      } else {
        return;
      }
    }
    __syncthreads();
    if ((tid + 1) < N) {
      x[tid] = scratch[tid] + 1.0f;
    }
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toContain("var bg_active_lane: bool = true;");
      expect(compiled.wgsl).toContain("bg_active_lane = false;");
      expect(compiled.wgsl).not.toContain("return;");
    });

  it("lowers loop alternate-branch returns before barriers into loop active-lane guards", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void loopAlternateReturnBarrier(float *x, int N) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    for (int k = 0; k < 2; ++k) {
      int idx = tid + k * 4;
      if (idx < N) {
        scratch[tid] = x[idx] + (float)k;
      } else {
        return;
      }
      __syncthreads();
      x[idx] = scratch[tid] + 1.0f;
      __syncthreads();
    }
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toContain("var bg_barrier_loop_active_");
      expect(compiled.wgsl).toMatch(/bg_barrier_loop_active_\d+ = false;/u);
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect(compiled.wgsl).not.toContain("return;");
    });

  it("preserves side effects before loop returns lowered for barriers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void loopReturnSideEffectBarrier(float *x, int N) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    for (int k = 0; k < 2; ++k) {
      int idx = tid + k * 4;
      if (idx >= N) {
        x[tid] = -10.0f - (float)tid;
        return;
      }
      scratch[tid] = x[idx] + (float)k;
      __syncthreads();
      x[idx] = scratch[tid] + 1.0f;
      __syncthreads();
    }
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toContain("x[tid] = f32(((-10.0) - f32(tid)));");
      expect(compiled.wgsl).toMatch(/bg_barrier_loop_active_\d+ = false;/u);
      expect(compiled.wgsl).not.toContain("return;");
    });

  it("preserves vector lane side effects before loop returns lowered for barriers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void vectorReturnSideEffectBarrier(float4 *out, int N) {
    int tid = threadIdx.x;
    for (int k = 0; k < 2; ++k) {
      int idx = tid + k * 4;
      if (idx >= N) {
        out[tid].w = -10.0f - (float)tid;
        return;
      }
      float4 value = out[idx];
      __syncthreads();
      out[idx] = make_float4(value.x + 1.0f, value.y + 2.0f, value.z + 3.0f, value.w + 4.0f);
      __syncthreads();
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toContain("-10.0");
      expect(compiled.wgsl).toMatch(/bg_barrier_loop_active_\d+ = false;/u);
      expect(compiled.wgsl).not.toContain("return;");
    });

  it("preserves pointer alias side effects before loop returns lowered for barriers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void pointerAliasReturnSideEffectBarrier(float *x, int N) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    for (int k = 0; k < 2; ++k) {
      int idx = tid + k * 4;
      float *target = &x[idx];
      if (idx >= N) {
        float *lane = &x[tid];
        *lane = -20.0f - (float)tid;
        return;
      }
      scratch[tid] = *target + (float)k;
      __syncthreads();
      *target = scratch[tid] + 1.0f;
      __syncthreads();
    }
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toContain("-20.0");
      expect(compiled.wgsl).toContain("bg_ptr_write_f32");
      expect(compiled.wgsl).toMatch(/bg_barrier_loop_active_\d+ = false;/u);
    });

  it("preserves shared-memory side effects before returns lowered for barriers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sharedReturnSideEffectBarrier(uint *out, int N) {
    extern __shared__ uint scratch[];
    int tid = threadIdx.x;
    scratch[tid] = 0u;
    __syncthreads();
    if (tid >= N) {
      scratch[tid] = 100u + (uint)tid;
      return;
    }
    __syncthreads();
    out[tid] = scratch[(tid + 1) & 3] + (uint)tid;
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toContain("scratch[tid] =");
      expect(compiled.wgsl).toContain("100u");
      expect(compiled.wgsl).toContain("bg_active_lane = false;");
      expect(compiled.wgsl).not.toContain("return;");
    });

  it("uses uniform dynamic bounds for barrier loops that exceed static smoke caps", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void longBarrierLoop(float *x, int N) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
      scratch[tid] = x[row * blockDim.x + tid];
      __syncthreads();
      x[row * blockDim.x + tid] = scratch[tid] + 1.0f;
    }
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });

      expect(compiled.analysis.barrierUniformity.kernel.verified).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("for (var row");
      expect(compiled.wgsl).toContain("bg_uniforms.N");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
    });

  it("keeps nested predicated barriers uniform after active-lane early returns", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void nestedPredicatedBarrier(float *x, int N) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    if (tid >= N) return;
    for (int k = 0; k < 2; ++k) {
      if (k + 1 < 2) {
        scratch[tid] = x[tid] + (float)k;
        __syncthreads();
      }
      __syncthreads();
    }
    x[tid] = scratch[tid] + 1.0f;
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: new Float32Array([1, 2, 3, 4]) }, scalars: { N: 4 } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain(": bool = bg_active_lane;");
      expect(compiled.wgsl).toContain("if (bg_barrier_loop_active_");
      expect(compiled.wgsl).toContain("&& (((k + 1) < 2)))");
      expect(compiled.wgsl).toContain("workgroupBarrier();\n    }\n    workgroupBarrier();");
      expect(compiled.wgsl).not.toContain("if (bg_active_lane) {\n    if (((k + 1) < 2))");
      expect([...result.buffers.x as Float32Array]).toEqual([2, 3, 4, 5]);
    });

  it("keeps nested predicated barriers uniform without early returns", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void nestedBarrierNoReturn(float *x) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    for (int k = 0; k < 2; ++k) {
      if (k + 1 < 2) {
        scratch[tid] = x[tid];
        __syncthreads();
      }
      __syncthreads();
    }
    x[tid] = scratch[tid] + 1.0f;
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: new Float32Array([1, 2, 3, 4]) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.analysis.barrierUniformity.kernel.verified).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl.match(/workgroupBarrier\(\);/gu)).toHaveLength(2);
      expect([...result.buffers.x as Float32Array]).toEqual([2, 3, 4, 5]);
    });

  it("keeps barriers uniform across predicated if-else branches", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void branchedBarrier(float *x, int flag) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    if (flag != 0) {
      scratch[tid] = x[tid];
      __syncthreads();
    } else {
      scratch[tid] = x[tid] + 1.0f;
      __syncthreads();
    }
    x[tid] = scratch[tid] + 1.0f;
  }`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: new Float32Array([1, 2, 3, 4]) }, scalars: { flag: 1 } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.analysis.barrierUniformity.kernel.verified).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("} else {");
      expect(compiled.wgsl.match(/workgroupBarrier\(\);/gu)).toHaveLength(2);
      expect([...result.buffers.x as Float32Array]).toEqual([2, 3, 4, 5]);
    });

  it("folds singleton thread axes before barrier uniformity analysis", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void singletonZBarrier(float *out, int N) {
    for (int i = blockIdx.x * blockDim.z + threadIdx.z; i < N; i += gridDim.x * blockDim.z) {
      out[i] = 1.0f;
      __syncthreads();
      out[i] += 1.0f;
    }
  }`, { workgroupSize: [32, 1, 1] });

      expect(compiled.wgsl).not.toContain("local_id.z");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
    });

  it("folds single-warp ids before barrier uniformity analysis", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void singleWarpBarrier(float *out) {
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    if (warpId != 0) { out[tid] = 1.0f; }
    __syncthreads();
    if (warpId == 0) { out[tid] = 2.0f; }
  }`, { workgroupSize: [32, 1, 1] });

      expect(compiled.analysis.barrierUniformity.kernel.verified).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
    });

  it("keeps loop barriers uniform after divergent breaks", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void breakBeforeBarrier(float *out, int N) {
    int tid = threadIdx.x;
    for (int i = 0; i < 2; ++i) {
      int idx = tid + i * 4;
      if (idx >= N) { break; }
      out[idx] = out[idx] + 1.0f;
      __syncthreads();
      out[idx] = out[idx] + 1.0f;
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.wgsl).toContain("var bg_loop_active_");
      expect(compiled.wgsl).toContain("bg_loop_active_");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect(compiled.wgsl).not.toContain("break;\n    out");
    });

  it("keeps post-loop barriers uniform after divergent breaks", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void breakBeforePostLoopBarrier(uint *out, int N) {
    int tid = threadIdx.x;
    for (int i = 0; i < 3; ++i) {
      out[tid] = (uint)i;
      if (tid >= N) { break; }
    }
    __syncthreads();
    if (tid < N) {
      out[tid] = out[tid] + 10u;
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-break-before-barrier");
      expect(compiled.wgsl).toContain("var bg_active_lane: bool = true;");
      expect(compiled.wgsl).toContain("bg_active_lane = (bg_active_lane && !((tid >= bg_uniforms.N)));");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect(compiled.wgsl).toContain("if (bg_active_lane) {\n    if ((tid < bg_uniforms.N))");
      expect(compiled.wgsl).not.toContain("if ((tid >= bg_uniforms.N)) {\n      break;");
    });

  it("keeps post-loop barriers uniform after nested divergent breaks", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void nestedBreakBeforePostLoopBarrier(uint *out, int N) {
    int tid = threadIdx.x;
    for (int i = 0; i < 3; ++i) {
      out[tid] = (uint)i;
      if (tid >= N) {
        if (out[tid] >= 0u) { break; }
      }
    }
    __syncthreads();
    if (tid < N) {
      out[tid] = out[tid] + 10u;
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-break-before-barrier");
      expect(compiled.wgsl).toContain("var bg_active_lane: bool = true;");
      expect(compiled.wgsl).toContain("bg_active_lane = (bg_active_lane && !((out[tid] >= 0u)));");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect(compiled.wgsl).toContain("if (bg_active_lane) {\n    if ((tid < bg_uniforms.N))");
      expect(compiled.wgsl).not.toContain("if ((out[tid] >= 0u)) {\n        break;");
    });

  it("keeps post-loop barriers uniform after while and do-while divergent breaks", () => {
      const whileCompiled = compileCudaLiteKernel(`
  __global__ void whileBreakBeforePostLoopBarrier(uint *out, int N) {
    int tid = threadIdx.x;
    int i = 0;
    while (i < 3) {
      out[tid] = (uint)i;
      if (tid >= N) { break; }
      i++;
    }
    __syncthreads();
    if (tid < N) { out[tid] = out[tid] + 10u; }
  }`, { workgroupSize: [4, 1, 1] });
      const doWhileCompiled = compileCudaLiteKernel(`
  __global__ void doWhileBreakBeforePostLoopBarrier(uint *out, int N) {
    int tid = threadIdx.x;
    int i = 0;
    do {
      out[tid] = (uint)i;
      if (tid >= N) { break; }
      i++;
    } while (i < 3);
    __syncthreads();
    if (tid < N) { out[tid] = out[tid] + 10u; }
  }`, { workgroupSize: [4, 1, 1] });

      for (const compiled of [whileCompiled, doWhileCompiled]) {
        expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-break-before-barrier");
        expect(compiled.wgsl).toContain("var bg_active_lane: bool = true;");
        expect(compiled.wgsl).toContain("bg_active_lane = (bg_active_lane && !((tid >= bg_uniforms.N)));");
        expect(compiled.wgsl).toContain("workgroupBarrier();");
        expect(compiled.wgsl).not.toContain("if ((tid >= bg_uniforms.N)) {\n      break;");
      }
    });

  it("rejects divergent continues before later barriers instead of emitting unsafe WGSL", () => {
      const source = `
  __global__ void continueBeforeBarrier(uint *out, int N) {
    int tid = threadIdx.x;
    for (int i = 0; i < 3; ++i) {
      if (tid >= N) { continue; }
      out[tid] = out[tid] + 1u;
      __syncthreads();
      out[tid] = out[tid] + 10u;
    }
  }`;
      const analysis = analyzeCudaLite(parseCudaLite(source), { workgroupSize: [4, 1, 1] });
      const diagnostic = analysis.diagnostics.find((item) => item.code === "divergent-continue-before-barrier");

      expect(diagnostic?.severity).toBe("error");
      expect(() => compileCudaLiteKernel(source, { workgroupSize: [4, 1, 1] })).toThrow(/divergent-continue-before-barrier/u);
    });

  it("allows divergent continues that do not skip later barriers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint sum_valid_lanes(uint value, int N) {
    uint total = 0u;
    for (int k = 0; k < 4; ++k) {
      if (threadIdx.x + k >= N) { continue; }
      total += value + (uint)k;
    }
    __syncthreads();
    return total;
  }

  __global__ void innerContinueBeforeBarrier(uint *out, int N) {
    int tid = threadIdx.x;
    out[tid] = sum_valid_lanes((uint)tid, N);
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("divergent-continue-before-barrier");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
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

  it("lowers subgroup intrinsics through scalar compatibility mode", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void subgroupCompat(float* x) {
    int lane = threadIdx.x % 32;
    float v = warp_reduce_sum_f32(x[threadIdx.x]);
    if (lane == 0) {
      v = bg_subgroup_add(v);
    }
    x[threadIdx.x] = v;
  }`, {
        subgroupMode: "scalar",
        workgroupSize: [32, 1, 1],
      });
      expect(backendIr(compiled).requiredFeatures).not.toContain("subgroups");
      expect(compiled.wgsl).not.toContain("enable subgroups;");
      expect(compiled.wgsl).not.toMatch(/\bsubgroup(?:Add|Max|Min|Shuffle|Ballot|Any|All)/u);
    });

  it("runs divergent subgroup calls as scalar operations in scalar compatibility mode", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void subgroupScalarReference(float* x) {
    int idx = threadIdx.x;
    float v = warp_reduce_sum_f32(x[idx]);
    if ((idx % 32) == 0) {
      v = bg_subgroup_add(v);
    }
    x[idx] = v;
  }`, {
        subgroupMode: "scalar",
        workgroupSize: [4, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: new Float32Array([1, 2, 3, 4]) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect([...result.buffers.x as Float32Array]).toEqual([1, 2, 3, 4]);
    });

  it("keeps native subgroup reductions uniform after active-lane early returns", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void subgroupAfterReturn(float* x, float* out, int n) {
    int idx = threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    float sum = bg_subgroup_add(v);
    if (idx == 0) {
      out[0] = sum;
    }
  }`, {
        features: { subgroups: true },
        workgroupSize: [4, 1, 1],
      });

      expect(compiled.wgsl).toContain("var bg_active_lane: bool = true;");
      expect(compiled.wgsl).not.toContain("return;");
      expect(compiled.wgsl).toContain("sum = select(sum, subgroupAdd(v), bg_active_lane);");
      expect(compiled.wgsl).not.toContain("if (bg_active_lane) {\n    sum = subgroupAdd");
    });

  it("keeps native subgroup reductions assigned to local arrays uniform after active-lane early returns", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void subgroupArrayAfterReturn(float* x, float* out, int n) {
    int idx = threadIdx.x;
    if (idx >= n) return;
    float values[2];
    values[0] = x[idx];
    values[1] = x[idx] + 1.0;
    for (int k = 0; k < 2; k++) {
      values[k] = bg_subgroup_add(values[k]);
    }
    if (idx == 0) {
      out[0] = values[0] + values[1];
    }
  }`, {
      features: { subgroups: true },
      workgroupSize: [4, 1, 1],
    });

      expect(compiled.wgsl).toContain("values[k] = select(values[k], bg_predicated_value_");
      expect(compiled.wgsl).not.toContain("if (bg_active_lane) {\n      values[k] = subgroupAdd");
    });

  it("keeps subgroup local declarations uniform inside predicated branches", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void subgroupVarInBranch(float* x, float* out) {
    int idx = threadIdx.x;
    float partial = bg_subgroup_add(x[idx]);
    if ((threadIdx.x / 32) == 0) {
      float gathered = idx == 0 ? partial : 0.0f;
      float total = bg_subgroup_add(gathered);
      if ((threadIdx.x % 32) == 0) {
        out[0] = total;
      }
    }
  }`, {
        features: { subgroups: true },
        workgroupSize: [32, 1, 1],
      });

      expect(compiled.wgsl).toContain("let bg_subgroup_if_active_");
      expect(compiled.wgsl).toContain("total = select(total, subgroupAdd(gathered), bg_subgroup_if_active_");
      expect(compiled.wgsl).not.toContain("if (((i32(local_id.x) / 32) == 0)) {\n    var gathered");
      expect(compiled.wgsl).not.toContain("if (((i32(local_id.x) / 32) == 0)) {\n    var total");
    });

  it("keeps native subgroup reductions uniform inside data-dependent loops", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void subgroupInDataLoop(float* x, float* out, int count, int stride) {
    int lane = threadIdx.x;
    float acc = 0.0f;
    for (int i = lane; i < count; i += stride) {
      acc += x[i];
      acc = bg_subgroup_add(acc);
    }
    out[lane] = acc;
  }`, {
        features: { subgroups: true },
        workgroupSize: [32, 1, 1],
      });

      expect(compiled.wgsl).toContain("var bg_barrier_loop_active_");
      expect(compiled.wgsl).toContain("acc = select(acc, bg_predicated_value_");
      expect(compiled.wgsl).not.toContain("for (var i: i32 = i32(local_id.x); (i < bg_uniforms.count); i += bg_uniforms.stride)");
    });

  it("avoids nonuniform dynamic bounds for subgroup loops with local lane-derived limits", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void subgroupLocalBound(float* x, float* out) {
    int t = threadIdx.x / 32;
    float acc = x[threadIdx.x];
    for (int i = 0; i <= t; ++i) {
      acc = bg_subgroup_add(acc);
    }
    out[threadIdx.x] = acc;
  }`, {
        features: { subgroups: true },
        workgroupSize: [32, 1, 1],
      });

      expect(compiled.wgsl).toContain("< 256u;");
      expect(compiled.wgsl).not.toContain("t + 1");
    });

  it("keeps vector cooperative reductions uniform after active-lane early returns", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float2 merge_pair(float2 a, float2 b) {
    float2 out;
    out.x = max(a.x, b.x);
    out.y = a.y + b.y;
    return out;
  }
  __global__ void subgroupVectorReduce(float* x, float* out, int n) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= n) return;
    float2 pair = make_float2(x[threadIdx.x], 1.0f);
    float2 total = cg::reduce(warp, pair, merge_pair);
    if (warp.thread_rank() == 0) {
      out[idx] = total.x + total.y;
    }
  }`, {
        features: { subgroups: true },
        workgroupSize: [32, 1, 1],
      });

      expect(compiled.wgsl).toContain("total = select(total, bg_cg_reduce_merge_pair_float2_32(pair");
      expect(compiled.wgsl).toContain("var<workgroup> bg_cg_reduce_merge_pair_float2_32_scratch");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect(compiled.wgsl).not.toContain("if (bg_active_lane) {\n    total = bg_cg_reduce");
    });

  it("keeps subgroup device functions uniform inside predicated branches", () => {
      const compiled = compileCudaLiteKernel(`
  __inline__ __device__ float warpPrefixSum(float val) {
    unsigned mask = 0xffffffff;
    for (int offset = 1; offset < 32; offset <<= 1) {
      float n = __shfl_up_sync(mask, val, offset, 32);
      int laneId = threadIdx.x & 31;
      if (laneId >= offset) {
        val += n;
      }
    }
    return val;
  }
  __global__ void warpScanKernel(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      float val = input[idx];
      float prefix = warpPrefixSum(val);
      output[idx] = prefix;
    }
  }`, {
        features: { subgroups: true },
        workgroupSize: [32, 1, 1],
      });

      expect(compiled.wgsl).toContain("warpPrefixSum");
      expect(compiled.wgsl).toContain("subgroup");
    });

  it("lowers syncwarp and fences through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void syncwarp_fence_semantic(int *out) {
    int tid = threadIdx.x;
    __syncwarp(0xffffffff);
    __threadfence();
    __threadfence_block();
    __threadfence_system();
    __nanosleep(tid);
    __prof_trigger(1);
    out[tid] = tid + 7;
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Int32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.kernelIr.operations.map((operation) => operation.kind)).toEqual([
        "declare",
        "barrier",
        "fence",
        "fence",
        "fence",
        "call",
        "call",
        "store",
      ]);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("workgroupBarrier();");
      expect(compiled.wgsl).toContain("storageBarrier();");
      expect([...result.buffers.out as Int32Array]).toEqual([7, 8, 9, 10]);
      expect([...semanticResult.buffers.out as Int32Array]).toEqual([7, 8, 9, 10]);
    });

  it("lowers WMMA fragments through scalarized cooperative matrix primitives", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void wmma_toy(float* A, float* B, float* C) {
    wmma::fragment<wmma::matrix_a, 2, 2, 2, float, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 2, 2, 2, float, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 2, 2, 2, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 2);
    wmma::load_matrix_sync(b, B, 2);
    wmma::mma_sync(c, a, b, c);
    for (int t = 0; t < c.num_elements; t++) {
      c.x[t] = c.x[t] + 1.0f;
    }
    wmma::store_matrix_sync(C, c, 2, wmma::mem_row_major);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            A: new Float32Array([1, 2, 3, 4]),
            B: new Float32Array([5, 6, 7, 8]),
            C: new Float32Array(4),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.C as Float32Array]).toEqual([20, 23, 44, 51]);
      expect(compiled.wgsl).toContain("var a: array<f32, 4>;");
      expect(compiled.wgsl).toContain("var bg_wmma_sum_");
      expect(compiled.wgsl).toContain("write_f32");
    });

  it("resolves same-named local pointer aliases by source scope for WMMA loads", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void wmma_scoped_alias(float* A, float* C, int first_marker, int second_marker) {
    for (int outer = 0; outer < 1; outer++) {
      int first_sel = first_marker;
      wmma::fragment<wmma::matrix_a, 2, 2, 2, float, wmma::row_major> frag[1];
      for (int i = 0; i < 1; i++) {
        float* tile = A + first_sel * 4;
        wmma::load_matrix_sync(frag[i], tile, 2);
      }
    }
    for (int outer = 0; outer < 1; outer++) {
      int second_sel = second_marker;
      wmma::fragment<wmma::matrix_a, 2, 2, 2, float, wmma::row_major> frag[1];
      for (int i = 0; i < 1; i++) {
        float* tile = A + second_sel * 4;
        wmma::load_matrix_sync(frag[i], tile, 2);
      }
    }
    C[0] = 0.0f;
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
      const firstBlock = compiled.wgsl.slice(
        compiled.wgsl.indexOf("var first_sel"),
        compiled.wgsl.indexOf("var second_sel"),
      );
      expect(firstBlock).toContain("first_sel");
      expect(firstBlock).not.toContain("second_sel");
    });

  it("keeps same-named WMMA pointer aliases when only one declaration needs a handle", () => {
      const compiled = compileCudaLiteKernel(`
  #define C_LAYOUT wmma::mem_row_major
  __global__ void wmma_shared_alias_shadow(float* out) {
    extern __shared__ half shmem[][8 * 16 + 16];
    float *base = (float *)&shmem[0][0];
    {
      const float *tile_ptr = base;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::load_matrix_sync(c, tile_ptr, 128, C_LAYOUT);
    }
    {
      const half *tile_ptr = &shmem[0][0];
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
      wmma::load_matrix_sync(a, tile_ptr, 16 * 8 + 16);
    }
  }`, {
        features: { "shader-f16": true },
        workgroupSize: [256, 1, 1],
        dynamicSharedMemory: { shmem: 256 },
      });

      expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
      expect(compiled.wgsl).toContain("bg_ptr_read_f32");
      expect(compiled.wgsl).toContain("bg_ptr_read_f16");
    });

  it("supports WMMA tf32 precision aliases and fragment lane access", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void wmma_tf32(float* A, float* C) {
    wmma::fragment<wmma::matrix_a, 2, 2, 2, wmma::precision::tf32, wmma::row_major> a;
    wmma::load_matrix_sync(a, A, 2);
    for (int t = 0; t < a.num_elements; t++) {
      a.x[t] = wmma::__float_to_tf32(a.x[t]) + 1.0f;
    }
    wmma::store_matrix_sync(C, a, 2, wmma::mem_row_major);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { A: new Float32Array([1, 2, 3, 4]), C: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.C as Float32Array]).toEqual([2, 3, 4, 5]);
      expect(compiled.wgsl).toContain("a[u32(t)]");
      expect(compiled.wgsl).toContain("f32(a[u32(t)])");
    });

  it("supports WMMA integer matrix operands with int accumulators", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void wmma_imma(uint8_t* A, uint8_t* B, int* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, uint8_t, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_t, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;
    wmma::fill_fragment(c, 1);
    wmma::load_matrix_sync(a, A, 16);
    wmma::load_matrix_sync(b, B, 16);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            A: new Uint32Array(256).fill(1),
            B: new Uint32Array(256).fill(2),
            C: new Int32Array(256),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.C as Int32Array]).toEqual(new Array(256).fill(33));
      expect(compiled.wgsl).toContain("var a: array<u32, 256>;");
      expect(compiled.wgsl).toContain("var c: array<i32, 256>;");
      expect(compiled.wgsl).toContain(": i32 = i32(c[");
      expect(compiled.wgsl).toContain("i32(u32(a[");
      expect(compiled.wgsl).toContain("write_i32");
    });

  it("validates WMMA fragment metadata and f16 requirements", () => {
      const half = analyzeCudaLite(parseCudaLite(`
  __global__ void half_wmma(half* A) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::load_matrix_sync(a, A, 16);
  }`));
      expect(half.requiredFeatures).toContain("shader-f16");

      const bad = analyzeCudaLite(parseCudaLite(`
  __global__ void bad_wmma(float* A) {
    wmma::fragment<wmma::matrix_a, 0, 16, 16, float> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float, wmma::row_major> c;
    wmma::fill_fragment(A, 0.0f);
  }`));
      const codes = bad.diagnostics.map((diagnostic) => diagnostic.code);
      expect(codes).toContain("invalid-wmma-fragment-shape");
      expect(codes).toContain("missing-wmma-fragment-layout");
      expect(codes).toContain("unsupported-wmma-fragment-value-type");
      expect(codes).toContain("unsupported-wmma-fragment-layout");
      expect(codes).toContain("unsupported-wmma-fragment-operand");

      const invalidImma = analyzeCudaLite(parseCudaLite(`
  __global__ void bad_imma(uint8_t* A) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_t, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, uint8_t> c;
    wmma::mma_sync(c, a, b, c);
  }`));
      expect(invalidImma.diagnostics.map((diagnostic) => diagnostic.code)).toContain("unsupported-wmma-fragment-value-type");
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
  lowerAnalyzedCudaLiteToKernelIr,
  constantBufferInputs,
  cudaWebGpuDefaultReadbackNames,
  deviceGlobalBufferInputs,
  deviceLaunchTreeIsExternallySilent,
  packCudaWebGpuUniformParams,
  packageRoot,
  backendIr,
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
