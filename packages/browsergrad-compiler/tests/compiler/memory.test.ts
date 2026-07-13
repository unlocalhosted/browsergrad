import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { createWgslFloat16Array } from "@unlocalhosted/browsergrad-kernels";
import {
  type CompiledCudaLiteKernel,
  CudaLiteCompilerError,
  analyzeCudaLite,
  validateSemanticKernelIr,
  typeCheckSemanticKernelIr,
  legalizeSemanticKernelIrForWgsl,
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
} from "../../src/index";
import { lowerAnalyzedCudaLiteToKernelIr } from "../../src/analyzer";
import {
  constantBufferInputs,
  cudaWebGpuDefaultReadbackNames,
  deviceGlobalBufferInputs,
} from "../../src/webgpu_inputs";
import { deviceLaunchTreeIsExternallySilent } from "../../src/runtime_elision";
import { packCudaWebGpuUniformParams } from "../../src/webgpu_orchestration";
import { semanticMemoryIdFromSymbol } from "../../src/semantic_ids";

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

describe("CUDA-lite compiler: Memory and pointer model", () => {
  it("executes pointer helpers with scalar device-global side effects", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint counter = 0, stopped = 0;
__device__ void record(uint *out) {
  if (!stopped) {
    printf("record %u", counter);
    counter++;
    stopped = 1;
    out[0] = counter;
  }
}
__global__ void helper_global(uint *out) { record(out); }
`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Uint32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...result.buffers.out as Uint32Array]).toEqual([1]);
  });

  it("bitcasts same-width shared pointer helper views through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void write_bits(uint *value) { value[0] = 0xffffffffu; }
__global__ void shared_reinterpret(int *out) {
  __shared__ int scratch[1];
  scratch[0] = 0;
  write_bits((uint *)scratch);
  out[0] = scratch[0];
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("bitcast<i32>(4294967295u)");
    expect([...result.buffers.out as Int32Array]).toEqual([-1]);
  });
  it("runs storage pointer offset updates through semantic reference and WGSL", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void pointerOffset(float* out, const float* input, int stride) {
    int row = threadIdx.x;
    out += row * stride;
    input += row * stride;
    out[1] = input[0] + input[1];
  }
  `, { workgroupSize: [2, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(4),
          input: new Float32Array([1, 2, 10, 20]),
        },
        scalars: { stride: 2 },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("var out__bg_ptr_offset");
      expect(compiled.wgsl).toContain("out[u32((out__bg_ptr_offset + 1))]");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([0, 3, 0, 30]);
      expect([...result.buffers.out as Float32Array]).toEqual([0, 3, 0, 30]);
    });

  it("lowers local storage pointer aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localStoragePointerAlias(float* out, const float* input, int stride) {
    int row = threadIdx.x;
    float* outRow = out + row * stride;
    const float* inRow = input + row * stride;
    outRow[1] = inRow[0] + inRow[1];
  }
  `, { workgroupSize: [2, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(4),
          input: new Float32Array([1, 2, 10, 20]),
        },
        scalars: { stride: 2 },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var outRow:");
      expect(compiled.wgsl).not.toContain("bg_ptr_read_f32");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([0, 3, 0, 30]);
      expect([...result.buffers.out as Float32Array]).toEqual([0, 3, 0, 30]);
    });

  it("lowers address-of storage pointer alias indices through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void storagePointerAddressOfIndex(float* out, const float* input) {
    int row = threadIdx.x;
    const float* base = input + row * 2;
    const float* shifted = &base[1];
    out[row] = shifted[0];
  }
  `, { workgroupSize: [2, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(2),
          input: new Float32Array([1, 2, 10, 20]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var shifted:");
      expect(compiled.wgsl).not.toContain("bg_ptr_read_f32");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([2, 20]);
      expect([...result.buffers.out as Float32Array]).toEqual([2, 20]);
    });

  it("lowers direct address-of storage parameter indices through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void storageParamAddressOfIndex(float* out, const float* input) {
    float* target = &out[1];
    const float* source = &input[2];
    target[0] = source[0];
  }
  `, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array([0, 0]),
          input: new Float32Array([3, 7, 11]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var target:");
      expect(compiled.wgsl).not.toContain("var source:");
      expect(compiled.wgsl).not.toContain("bg_ptr_read_f32");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([0, 11]);
      expect([...result.buffers.out as Float32Array]).toEqual([0, 11]);
    });

  it("lowers same-root storage pointer comparisons through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void storagePointerCompare(uint* out, const float* input) {
    const float* a = &input[1];
    const float* b = &input[3];
    out[0] = a == b ? 1u : 0u;
    out[1] = a != b ? 1u : 0u;
    out[2] = a < b ? 1u : 0u;
    out[3] = b >= a ? 1u : 0u;
  }
  `, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          out: new Uint32Array(4),
          input: new Float32Array([3, 5, 7, 11]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-pointer-pointer-comparison");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var a:");
      expect(compiled.wgsl).not.toContain("var b:");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([0, 1, 1, 1]);
      expect([...result.buffers.out as Uint32Array]).toEqual([0, 1, 1, 1]);
    });

  it("lowers nullable storage pointer conditionals through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void nullableStoragePointer(uint* out, const uint* input, int pick) {
    const uint* maybe = pick != 0 ? &input[1] : NULL;
    out[0] = maybe != NULL ? 1u : 0u;
    out[1] = maybe == nullptr ? 1u : 0u;
    if (maybe != NULL) {
      out[2] = maybe[0];
    }
  }
  `, { workgroupSize: [1, 1, 1] });
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const nonNullInput = {
        buffers: {
          out: new Uint32Array(3),
          input: new Uint32Array([5, 13]),
        },
        scalars: { pick: 1 },
      };
      const nullInput = {
        buffers: {
          out: new Uint32Array(3),
          input: new Uint32Array([5, 13]),
        },
        scalars: { pick: 0 },
      };
      const semanticResult = runCompiledKernelSemanticReference(compiled, nonNullInput, launch);
      const result = runCompiledKernelReference(compiled, nonNullInput, launch);
      const nullResult = runCompiledKernelSemanticReference(compiled, nullInput, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-pointer-conditional");
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-pointer-pointer-comparison");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var maybe:");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1, 0, 13]);
      expect([...result.buffers.out as Uint32Array]).toEqual([1, 0, 13]);
      expect([...nullResult.buffers.out as Uint32Array]).toEqual([0, 1, 0]);
    });

  it("lowers same-root storage pointer differences through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void storagePointerDifference(int* out, const float* input) {
    const float* base = input;
    const float* shifted = &input[3];
    out[0] = shifted - base;
  }
  `, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          out: new Int32Array(1),
          input: new Float32Array([3, 5, 7, 11]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-pointer-difference");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var shifted:");
      expect([...semanticResult.buffers.out as Int32Array]).toEqual([3]);
      expect([...result.buffers.out as Int32Array]).toEqual([3]);
    });

  it("lowers address-of dereferenced storage pointer aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void storagePointerAddressOfDeref(float* out, const float* input) {
    const float* base = input + 1;
    const float* same = &*base;
    out[0] = same[0];
  }
  `, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(1),
          input: new Float32Array([3, 7]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var same:");
      expect(compiled.wgsl).not.toContain("bg_ptr_read_f32");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([7]);
      expect([...result.buffers.out as Float32Array]).toEqual([7]);
    });

  it("lowers branch-merged storage pointer aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void storagePointerBranchMerge(float* out, const float* input, int pick) {
    const float* p = input;
    if (pick != 0) {
      p = input + 1;
    } else {
      p = input + 2;
    }
    out[0] = p[0];
  }
  `, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(1),
          input: new Float32Array([3, 7, 11]),
        },
        scalars: { pick: 0 },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([11]);
      expect([...result.buffers.out as Float32Array]).toEqual([11]);
    });

  it("lowers block-updated storage pointer aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void storagePointerBlockUpdate(float* out, const float* input) {
    const float* p = input;
    {
      p = input + 1;
      p += 1;
    }
    out[0] = p[0];
  }
  `, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(1),
          input: new Float32Array([3, 7, 11]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([11]);
      expect([...result.buffers.out as Float32Array]).toEqual([11]);
    });

  it("lowers device helper functions with storage pointer params", () => {
      const compiled = compileCudaLiteKernel(DEVICE_POINTER_HELPERS, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelSemanticReference(
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
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bg_ptr_read_f32(buffer: u32, index: u32) -> f32");
      expect(compiled.wgsl).toContain("fn bg_ptr_write_f32(buffer: u32, index: u32, value: f32)");
      expect(compiled.wgsl).toContain("fn loadAt(ptr_buffer: u32, ptr_base: u32, offset: i32");
      expect(compiled.wgsl).toContain("return bg_ptr_read_f32(ptr_buffer, u32((i32(ptr_base) + offset)))");
      expect(compiled.wgsl).toContain("addAt(1u, 0u, idx");
      expect([...result.buffers.y as Float32Array]).toEqual([14, 26, 38]);
    });

  it("lowers looped local arithmetic in storage-pointer device helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void sumWindow(float *out, const float *input, int count) {
    float total = 0.0f;
    for (int i = 0; i < count; ++i) {
      total += input[i];
    }
    out[0] = total;
  }

  __global__ void sumWindows(float *out, const float *input) {
    int tid = threadIdx.x;
    sumWindow(out + tid, input + tid * 2, 2);
  }`, { workgroupSize: [2, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(2),
          input: new Float32Array([1, 2, 3, 4]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("fn sumWindow(");
      expect(compiled.wgsl).toContain("for (var i: i32 = 0;");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([3, 7]);
      expect([...result.buffers.out as Float32Array]).toEqual([3, 7]);
    });

  it("lowers local return values from storage-pointer device helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float sumWindow(const float *input, int count) {
    float total = 0.0f;
    for (int i = 0; i < count; ++i) {
      total += input[i];
    }
    return total;
  }

  __global__ void sumWindows(float *out, const float *input) {
    int tid = threadIdx.x;
    out[tid] = sumWindow(input + tid * 2, 2);
  }`, { workgroupSize: [2, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(2),
          input: new Float32Array([1, 2, 3, 4]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("return total;");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([3, 7]);
      expect([...result.buffers.out as Float32Array]).toEqual([3, 7]);
    });

  it("lowers direct kernel pointer dereference writes through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void directDeref(float* result, int* count) {
    *result = 3.5f;
    *count = 2;
  }
  `, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { result: new Float32Array(1), count: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_i32");
      expect(compiled.wgsl).toContain("result[0u] = 3.5;");
      expect(compiled.wgsl).toContain("count[0u] = 2;");
      expect([...result.buffers.result as Float32Array]).toEqual([3.5]);
      expect([...result.buffers.count as Int32Array]).toEqual([2]);
    });

  it("flattens device helper aliases rooted at pointer params", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float loadAlias(const float* inp, int row) {
    const float* x = inp + row * 4;
    return x[1];
  }
  __global__ void aliasedParam(const float* inp, float* out, int row) {
    out[0] = loadAlias(inp, row);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            inp: new Float32Array([1, 2, 3, 4, 50, 60, 70, 80]),
            out: new Float32Array(1),
          },
          scalars: { row: 1 },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toMatch(/bg_ptr_read_f32\(inp_buffer(?:_arg)?/u);
      expect(compiled.wgsl).not.toContain("x[");
      expect([...result.buffers.out as Float32Array]).toEqual([60]);
    });

  it("supports conditional storage pointer arguments to device helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void copyOne(float *target, const float *fallback, const float *source) {
    float value = source != NULL ? source[0] : fallback[0];
    target[0] = value;
  }
  __global__ void conditionalPointer(float *out, const float *fallback, const float *maybeSource, int useSource) {
    copyOne(out, fallback, useSource ? maybeSource : NULL);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Float32Array(1),
            fallback: new Float32Array([2]),
            maybeSource: new Float32Array([7]),
          },
          scalars: { useSource: 1 },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("2u, select(4294967295u, 0u,");
      expect([...result.buffers.out as Float32Array]).toEqual([7]);
    });

  it("decays fixed C array device-helper params to pointer params", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void copyArrayParam(float out[2], const float src[2]) {
    out[0] = src[1];
  }
  __global__ void array_param_decay(float *out) {
    __shared__ float tile[2];
    if (threadIdx.x == 0) {
      tile[1] = 6.25f;
      copyArrayParam(out, tile);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn copyArrayParam(out_buffer: u32, out_base: u32");
      expect(compiled.wgsl).toContain("src__bg_shared_ptr: ptr<workgroup, array<f32, 2>>");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([6.25]);
      expect([...result.buffers.out as Float32Array]).toEqual([6.25]);
    });

  it("lowers device helper local scalar out params as WGSL function pointers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void bounds(float x, float *lo, float *hi) {
    *lo = x - 1.0f;
    *hi = x + 1.0f;
  }

  __global__ void writeBounds(float *out) {
    float lo = 0.0f;
    float hi = 0.0f;
    bounds(3.0f, &lo, &hi);
    out[0] = lo;
    out[1] = hi;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("lo: ptr<function, f32>");
      expect(compiled.wgsl).toContain("*lo =");
      expect([...result.buffers.out as Float32Array]).toEqual([2, 4]);
    });

  it("allows readonly constant tables inside local out-pointer helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __constant__ float weights[2] = {2.0f, 3.0f};
  __device__ float weighted(float value, uint *selected) {
    *selected = value > 1.0f ? 1u : 0u;
    return value * weights[*selected];
  }
  __global__ void constantLocalOut(float *out) {
    uint selected = 0u;
    out[0] = weighted(2.0f, &selected);
    out[1] = (float)selected;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("selected: ptr<function, u32>");
      expect([...result.buffers.out as Float32Array]).toEqual([6, 1]);
    });

  it("lowers vector reinterpret memory-view helpers through semantic WGSL", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ float4 ld_vec(const float* address) {
    return *reinterpret_cast<const float4*>(address);
  }

  __device__ void st_vec(float* address, float4 val) {
    *reinterpret_cast<float4*>(address) = val;
  }

  __global__ void vector_helper(float* out, const float* inp) {
    float4 value = ld_vec(inp);
    value.y += 10.0f;
    st_vec(out, value);
  }`, {
        features: { "shader-f16": true, subgroups: true },
        workgroupSize: [1, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(4), inp: new Float32Array([1, 2, 3, 4]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn ld_vec(");
      expect(compiled.wgsl).toContain("fn st_vec(");
      expect(compiled.wgsl).toContain("fn bg_ptr_read_f32x4(");
      expect(compiled.wgsl).toContain("fn bg_ptr_write_f32x4(");
      expect(compiled.wgsl).not.toContain("address[");
      expect([...result.buffers.out as Float32Array]).toEqual([1, 12, 3, 4]);
    });

  it("emits integer min/max device helpers with integer operands and safe WGSL names", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ int clamp(int x, int lo, int hi) {
    x = max(lo, min(hi, x));
    return x;
  }

  __global__ void clampKernel(int *out) {
    out[0] = clamp(-5, -2, 4);
    out[1] = clamp(8, -2, 4);
  }`, { workgroupSize: [1, 1, 1] });
      const input = { buffers: { out: new Int32Array(2) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("fn bg_clamp(bg_arg_x: i32, lo: i32, hi: i32");
      expect(compiled.wgsl).toContain("var x: i32 = bg_arg_x;");
      expect(compiled.wgsl).toContain("x = max(lo, min(hi, x));");
      expect([...semanticResult.buffers.out as Int32Array]).toEqual([-2, 4]);
      expect([...result.buffers.out as Int32Array]).toEqual([-2, 4]);
    });

  it("lowers CUDA vector swizzle writes through storage vector views", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void vector_storage_swizzle_writes(float* out, uint* ui) {
    float4* view = reinterpret_cast<float4*>(out);
    view[0] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    view[0].xy = make_float2(9.0f, 8.0f);
    view[0].zw += make_float2(1.0f, 2.0f);
    uint4* bits = reinterpret_cast<uint4*>(ui);
    bits[0] = make_uint4(5u, 6u, 7u, 8u);
    bits[0].s210 = make_uint3(11u, 12u, 13u);
    bits[0].xy += make_uint2(1u, 2u);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(4), ui: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(4), ui: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-vector-assignment");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect([...result.buffers.out as Float32Array]).toEqual([9, 8, 4, 6]);
      expect([...result.buffers.ui as Uint32Array]).toEqual([14, 14, 11, 8]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([9, 8, 4, 6]);
      expect([...semanticResult.buffers.ui as Uint32Array]).toEqual([14, 14, 11, 8]);
    });

  it("lowers CUDA vector swizzle writes through helper pointer params", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void set_vector_swizzles(float4* view, uint4* bits) {
    view[0].xy = make_float2(9.0f, 8.0f);
    view[0].zw += make_float2(1.0f, 2.0f);
    bits[0].s210 = make_uint3(11u, 12u, 13u);
    bits[0].xy += make_uint2(1u, 2u);
  }

  __global__ void helper_vector_storage_swizzle_writes(float* out, uint* ui) {
    float4* view = reinterpret_cast<float4*>(out);
    view[0] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    uint4* bits = reinterpret_cast<uint4*>(ui);
    bits[0] = make_uint4(5u, 6u, 7u, 8u);
    set_vector_swizzles(view, bits);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(4), ui: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(4), ui: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-vector-assignment");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect([...result.buffers.out as Float32Array]).toEqual([9, 8, 4, 6]);
      expect([...result.buffers.ui as Uint32Array]).toEqual([14, 14, 11, 8]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([9, 8, 4, 6]);
      expect([...semanticResult.buffers.ui as Uint32Array]).toEqual([14, 14, 11, 8]);
    });

  it("keeps shifted scalar bases aligned when casting helper pointer params to vector lanes", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ void write_lane_offset(float4* out, int idx, float value) {
    out[idx].y = value;
  }

  __global__ void vector_lane_offset(float* out, const float* inp) {
    int idx = threadIdx.x;
    const float4* readView = reinterpret_cast<const float4*>(inp);
    float4* writeView = reinterpret_cast<float4*>(out + 4);
    float4 value = readView[idx];
    write_lane_offset(writeView, idx, value.x + value.w);
  }`, { workgroupSize: [2, 1, 1] });

      expect(compiled.wgsl).toContain("write_lane_offset(0u, 4u, idx");
      expect(compiled.wgsl).not.toContain("write_lane_offset(0u, u32((i32(out_base) + 4)), idx");
      expect(compiled.wgsl).toContain("case 0u: { out[(index + 0u)] = (value).x;");
      expect(compiled.wgsl).toContain("bg_ptr_write_f32x4(out_buffer, (out_base + (u32(idx) * 4u))");
    });

  it("keeps shifted vector bases aligned when casting helper pointer params to scalar lanes", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ float load_scalar_offset(const float* inp, int idx) {
    return inp[idx];
  }

  __global__ void vector_to_scalar_offset(float* out, const float4* inp) {
    int idx = threadIdx.x;
    const float* scalarView = reinterpret_cast<const float*>(inp + 1);
    out[idx] = load_scalar_offset(scalarView, idx);
  }`, { workgroupSize: [2, 1, 1] });

      expect(compiled.wgsl).toMatch(/load_scalar_offset\(1u, .*4.*idx/u);
      expect(compiled.wgsl).toContain("return bg_ptr_read_f32(inp_buffer");
    });

  it("writes scalar helper pointer params through shifted vector-backed storage", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ void write_scalar_offset(float* out, int idx, float value) {
    out[idx] = value;
  }

  __global__ void vector_to_scalar_write_offset(float4* out) {
    int idx = threadIdx.x;
    float* scalarView = reinterpret_cast<float*>(out + 1);
    write_scalar_offset(scalarView, idx, 7.0f + (float)idx);
  }`, { workgroupSize: [2, 1, 1] });

      expect(compiled.wgsl).toMatch(/write_scalar_offset\(0u, .*4.*idx/u);
      expect(compiled.wgsl).toContain("bg_ptr_write_f32(out_buffer");
    });

  it("keeps casted vector pointer arithmetic CAS on pointer helpers", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ uint cas_uint_vector_slot(uint4* slot, uint compare, uint value) {
    uint* lanes = reinterpret_cast<uint*>(slot);
    return atomicCAS(lanes + 2, compare, value);
  }

  __global__ void vector_pointer_array_cas_offset(uint4* out, uint* summary) {
    uint4* slots[2];
    slots[0] = out + 1;
    slots[1] = out + 2;
    summary[0] = cas_uint_vector_slot(slots[0], 21u, 81u);
  }`, { workgroupSize: [1, 1, 1] });
      const out = new Uint32Array(12);
      out[6] = 21;
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out, summary: new Uint32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bg_ptr_atomicCompareExchange_u32(");
      expect(compiled.wgsl).toMatch(/bg_ptr_atomicCompareExchange_u32\(slot_buffer, [^\n]*slot_base[^\n]*2[^\n]*compare, value\)/u);
      expect(compiled.wgsl).not.toContain("atomicCompareExchangeWeak((slot_base + 2u)");
      expect([...result.buffers.out as Uint32Array]).toEqual([0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0]);
      expect([...result.buffers.summary as Uint32Array]).toEqual([21]);
    });

  it("guards active-lane pointer-array assignments without scalar select fallback", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ void add_uint_vector_slot(uint4* slot, uint value) {
    uint* lanes = reinterpret_cast<uint*>(slot);
    atomicAdd(lanes + 1, value);
  }

  __global__ void active_lane_pointer_array_assignment(uint4* out, uint4* shadow, int n) {
    int tid = threadIdx.x;
    if (tid >= n) {
      return;
    }
    __syncthreads();
    uint4* slots[2];
    slots[0] = out + 1;
    slots[1] = shadow + 1;
    add_uint_vector_slot(slots[tid], 5u);
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.wgsl).toContain("if (bg_active_lane) {");
      expect(compiled.wgsl).toContain("slots_buffer[u32(0)] = 0u; slots_base[u32(0)] = (0u + (u32(1) * 4u));");
      expect(compiled.wgsl).toContain("slots_buffer[u32(1)] = 1u; slots_base[u32(1)] = (0u + (u32(1) * 4u));");
      expect(compiled.wgsl).not.toContain("select(slots[");
    });

  it("keeps active-lane pointer-array helper args behind the active guard", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ void write_two_active_selected_ptrs(float *a, float *b, float addA, float addB) {
    a[0] = a[0] + addA;
    b[0] = b[0] + addB;
  }

  __global__ void active_lane_helper_pointer_array_selected_args(float4 *values, int limit, int pickRight) {
    int tid = threadIdx.x;
    float *lanes = reinterpret_cast<float*>(values);
    for (int step = 0; step < 2; step++) {
      if (tid >= limit) return;
      __syncthreads();
      float *ptrs[2];
      ptrs[0] = lanes + tid;
      ptrs[1] = lanes + tid + 4;
      write_two_active_selected_ptrs(ptrs[0], ptrs[pickRight != 0 ? 1 : 0], (float)(10 + step), (float)(20 + step));
      __syncthreads();
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toContain("if (bg_barrier_loop_active_");
      expect(compiled.wgsl).toContain("write_two_active_selected_ptrs");
      expect(compiled.wgsl).not.toContain("select(ptrs");
    });

  it("guards active-lane conditional helper-call pointer-array indices", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ uint active_conditional_pointer_array_index_helper(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __device__ void add_selected_uint_ptr(uint *ptr, uint value) {
    atomicAdd(ptr, value);
  }

  __global__ void active_lane_conditional_helper_pointer_array_index(uint *storage, int limit, int enabled) {
    int tid = threadIdx.x;
    for (int step = 0; step < 2; step++) {
      if (tid >= limit) return;
      __syncthreads();
      uint *ptrs[2];
      ptrs[0] = storage + tid;
      ptrs[1] = storage + tid + 4;
      add_selected_uint_ptr(ptrs[enabled != 0 ? active_conditional_pointer_array_index_helper(storage + tid, (uint)(step + tid + 1)) : 0u], 5u);
      __syncthreads();
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toMatch(/if \(bg_barrier_loop_active_\d+\) \{\n\s+if \(\(bg_uniforms.enabled != 0\)\)/u);
      expect(compiled.wgsl).toContain("active_conditional_pointer_array_index_helper");
      expect(compiled.wgsl!.match(/\bactive_conditional_pointer_array_index_helper\(/gu) ?? []).toHaveLength(2);
      expect(compiled.wgsl).toMatch(/let bg_pointer_array_index_\d+: u32 = active_conditional_pointer_array_index_helper\(/u);
      expect(compiled.wgsl).not.toContain("select(0u, active_conditional_pointer_array_index_helper");
    });

  it("evaluates side-effecting pointer-array assignment indices once", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ uint pointer_array_assignment_index_helper(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __global__ void pointer_array_assignment_index_once(uint *storage) {
    uint *ptrs[2];
    ptrs[0] = storage;
    ptrs[1] = storage + 1;
    ptrs[pointer_array_assignment_index_helper(storage, 1u)] = storage + 2;
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl!.match(/\bpointer_array_assignment_index_helper\(/gu) ?? []).toHaveLength(2);
      expect(compiled.wgsl).toMatch(/var bg__bg_pointer_array_index_\d+_\d+: u32 = pointer_array_assignment_index_helper\(/u);
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { storage: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...result.buffers.storage as Uint32Array]).toEqual([1, 0, 0, 0]);
    });

  it("evaluates side-effecting nested pointer-array target indices once", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ uint nested_pointer_array_target_index_helper(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __global__ void nested_pointer_array_target_index_once(uint *storage) {
    uint *ptrs[2];
    ptrs[0] = storage + 1;
    ptrs[1] = storage + 2;
    ptrs[nested_pointer_array_target_index_helper(storage, 1u)][0] = 5u;
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl!.match(/\bnested_pointer_array_target_index_helper\(/gu) ?? []).toHaveLength(2);
      expect(compiled.wgsl).toMatch(/let bg_pointer_array_index_\d+: u32 = nested_pointer_array_target_index_helper\(/u);
    });

  it("evaluates side-effecting pointer-array comparison indices once", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ uint pointer_array_compare_index_helper(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __global__ void pointer_array_compare_index_once(uint *storage) {
    uint *ptrs[2];
    ptrs[0] = storage + 1;
    ptrs[1] = storage + 2;
    storage[3] = ptrs[pointer_array_compare_index_helper(storage, 1u)] == storage + 2 ? 7u : 9u;
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl!.match(/\bpointer_array_compare_index_helper\(/gu) ?? []).toHaveLength(2);
      expect(compiled.wgsl).toContain("pointer_array_compare_index_helper(0u, 0u, 1u");
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { storage: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...result.buffers.storage as Uint32Array]).toEqual([1, 0, 0, 7]);
    });

  it("evaluates side-effecting pointer-array var-init comparison indices once", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ uint pointer_array_var_init_compare_index_helper(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __global__ void pointer_array_var_init_compare_index_once(uint *storage) {
    uint *ptrs[2];
    ptrs[0] = storage + 1;
    ptrs[1] = storage + 2;
    uint value = ptrs[pointer_array_var_init_compare_index_helper(storage, 1u)] == storage + 2 ? 7u : 9u;
    storage[3] = value;
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl!.match(/\bpointer_array_var_init_compare_index_helper\(/gu) ?? []).toHaveLength(2);
      expect(compiled.wgsl).toMatch(/var bg__bg_condition_test_\d+_\d+: bool = \(pointer_array_var_init_compare_index_helper\(/u);
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { storage: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...result.buffers.storage as Uint32Array]).toEqual([1, 0, 0, 7]);
    });

  it("evaluates side-effecting pointer-array difference indices once", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ uint pointer_array_diff_index_helper(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __global__ void pointer_array_diff_index_once(uint *storage) {
    uint *ptrs[2];
    ptrs[0] = storage + 1;
    ptrs[1] = storage + 3;
    storage[3] = (uint)(ptrs[pointer_array_diff_index_helper(storage, 1u)] - (storage + 1));
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl!.match(/\bpointer_array_diff_index_helper\(/gu) ?? []).toHaveLength(2);
      expect(compiled.wgsl).toContain("pointer_array_diff_index_helper(0u, 0u, 1u");
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { storage: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...result.buffers.storage as Uint32Array]).toEqual([1, 0, 0, 2]);
    });

  it("scales vector pointer differences by vector lane width", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __global__ void vector_pointer_difference(uint4 *out, int *summary) {
    uint4 *left = out + 2;
    uint4 *right = out;
    summary[0] = left - right;
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("summary[0u] = 2;");
    });

  it("evaluates side-effecting vector pointer-array difference indices once", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ uint vector_pointer_array_diff_index_helper(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __global__ void vector_pointer_array_diff_index_once(uint4 *out, uint *counter, int *summary) {
    uint4 *ptrs[2];
    ptrs[0] = out + 1;
    ptrs[1] = out + 3;
    summary[0] = ptrs[vector_pointer_array_diff_index_helper(counter, 1u)] - (out + 1);
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl!.match(/\bvector_pointer_array_diff_index_helper\(/gu) ?? []).toHaveLength(2);
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(16), counter: new Uint32Array(1), summary: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...result.buffers.counter as Uint32Array]).toEqual([1]);
      expect([...result.buffers.summary as Int32Array]).toEqual([2]);
    });

  it("keeps scalar-view pointer differences over vector roots in scalar lanes", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __global__ void scalar_view_vector_pointer_difference(uint4 *out, int *summary) {
    uint *left = reinterpret_cast<uint*>(out + 1);
    uint *right = reinterpret_cast<uint*>(out);
    summary[0] = left - right;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(8), summary: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("summary[0u] = ((1 * 4) - (0 * 4));");
      expect([...result.buffers.summary as Int32Array]).toEqual([4]);
    });

  it("scales inline cast pointer differences over byte roots", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __global__ void byte_root_inline_cast_pointer_difference(uchar *bytes, int *summary) {
    summary[0] = reinterpret_cast<float*>(bytes + 8) - reinterpret_cast<float*>(bytes);
  }`, { workgroupSize: [1, 1, 1] });

      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { bytes: new Uint32Array(4), summary: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...result.buffers.summary as Int32Array]).toEqual([2]);
      expect(compiled.wgsl).not.toMatch(/summary\[0\] = i32\(\(i32\(.+\) - i32\(.+\)\)\);/u);
    });

  it("scales byte-root pointer-array differences with side-effecting indices", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ uint byte_pointer_array_diff_index_helper(uint *counter) {
    atomicAdd(counter, 1u);
    return 1u;
  }

  __global__ void byte_root_pointer_array_diff_index_once(uchar *bytes, uint *counter, int *summary) {
    float *ptrs[2];
    ptrs[0] = reinterpret_cast<float*>(bytes + 4);
    ptrs[1] = reinterpret_cast<float*>(bytes + 12);
    summary[0] = ptrs[byte_pointer_array_diff_index_helper(counter)] - reinterpret_cast<float*>(bytes + 4);
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl!.match(/\bbyte_pointer_array_diff_index_helper\(/gu) ?? []).toHaveLength(2);
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { bytes: new Uint32Array(4), counter: new Uint32Array(1), summary: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...result.buffers.counter as Uint32Array]).toEqual([1]);
      expect([...result.buffers.summary as Int32Array]).toEqual([2]);
    });

  it("scales byte-root vector pointer-array differences by vector byte size", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ uint byte_vector_pointer_array_diff_index_helper(uint *counter) {
    atomicAdd(counter, 1u);
    return 1u;
  }

  __global__ void byte_root_vector_pointer_array_diff_index_once(uchar *bytes, uint *counter, int *summary) {
    float4 *ptrs[2];
    ptrs[0] = reinterpret_cast<float4*>(bytes + 16);
    ptrs[1] = reinterpret_cast<float4*>(bytes + 48);
    summary[0] = ptrs[byte_vector_pointer_array_diff_index_helper(counter)] - reinterpret_cast<float4*>(bytes + 16);
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl!.match(/\bbyte_vector_pointer_array_diff_index_helper\(/gu) ?? []).toHaveLength(2);
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { bytes: new Uint32Array(16), counter: new Uint32Array(1), summary: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect([...result.buffers.counter as Uint32Array]).toEqual([1]);
      expect([...result.buffers.summary as Int32Array]).toEqual([2]);
    });

  it("preserves scalar-to-vector pointer alias byte offsets", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ void bump_roundtrip_vec(float4* out, int idx, float4 delta) {
    float4 value = out[idx];
    out[idx] = make_float4(value.x + delta.x, value.y + delta.y, value.z + delta.z, value.w + delta.w);
  }

  __global__ void vectorScalarVectorAliasRoundtrip(float4* out) {
    int idx = threadIdx.x;
    float* scalarView = reinterpret_cast<float*>(out + 1);
    float4* vecView = reinterpret_cast<float4*>(scalarView);
    float scale = (float)(idx + 1);
    bump_roundtrip_vec(vecView, idx, make_float4(scale, scale * 2.0f, scale * 3.0f, scale * 4.0f));
  }`, { workgroupSize: [2, 1, 1] });

      expect(compiled.wgsl).toMatch(/bump_roundtrip_vec\(0u, [^\n]*1 \* 4[^\n]*, idx/u);
      expect(compiled.wgsl).toContain("bg_ptr_read_f32x4(out_buffer, (out_base + (u32(idx) * 4u)))");
      expect(compiled.wgsl).toContain("bg_ptr_write_f32x4(out_buffer, (out_base + (u32(idx) * 4u)), vec4<f32>");
    });

  it("keeps device-global vector pointer-array entries in flat lanes", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ float3 g_ptr_values[3];

  __device__ float3 sum_global_ptrs(float3 *a, float3 *b, float3 *c) {
    return *a + *b + *c;
  }

  __global__ void deviceGlobalVectorPointerArray(float4 *out) {
    g_ptr_values[0] = make_float3(2.0f, 3.0f, 5.0f);
    g_ptr_values[1] = make_float3(7.0f, 11.0f, 13.0f);
    g_ptr_values[2] = make_float3(17.0f, 19.0f, 23.0f);
    float3 *ptrs[3];
    ptrs[0] = &g_ptr_values[0];
    ptrs[1] = &g_ptr_values[1];
    ptrs[2] = &g_ptr_values[2];
    float3 total = sum_global_ptrs(ptrs[0], ptrs[1], ptrs[2]);
    out[0] = make_float4(*ptrs[2], 1.0f);
    out[1] = make_float4(total, 0.0f);
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl).toContain("ptrs_base[u32(2)] = (u32(2) * 3u);");
      expect(compiled.wgsl).toContain("sum_global_ptrs(ptrs_buffer[u32(0)], ptrs_base[u32(0)], ptrs_buffer[u32(1)], ptrs_base[u32(1)], ptrs_buffer[u32(2)], ptrs_base[u32(2)]");
      expect(compiled.wgsl).not.toContain("ptrs_base[u32(2)] = u32(2);");
    });

  it("reads shared vector pointer helpers through scalar lanes", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ void adjust_storage_alias_lane(float *lanes, int offset, float delta) {
    lanes[offset] = lanes[offset] + delta;
  }

  __global__ void crossSpaceVectorAliasConsistency(float *out) {
    __shared__ float4 shared[2];
    shared[1] = make_float4(50.0f, 60.0f, 70.0f, 80.0f);
    adjust_storage_alias_lane(reinterpret_cast<float*>(shared + 1), 1, 1.5f);
    out[0] = shared[1].y;
  }`, { workgroupSize: [1, 1, 1] });

      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect(compiled.wgsl).toContain("ptr<workgroup, array<vec4<f32>, 8>>");
      expect([...result.buffers.out as Float32Array]).toEqual([61.5]);
    });

  it("lowers scalar shared-memory vector views through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sharedScalarVectorView(float *out) {
    __shared__ float tile[8];
    int lane = threadIdx.x;
    float4 *view = reinterpret_cast<float4 *>(&tile[lane * 4]);
    view[0] = make_float4(float(lane * 10 + 1), float(lane * 10 + 2), float(lane * 10 + 3), float(lane * 10 + 4));
    __syncthreads();
    float4 value = view[0];
    out[lane] = value.w;
  }`, { workgroupSize: [2, 1, 1] });
      const input = { buffers: { out: new Float32Array(2) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("tile[");
      expect(compiled.wgsl).toContain("vec4<f32>(tile[");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([4, 14]);
      expect([...result.buffers.out as Float32Array]).toEqual([4, 14]);
    });

  it("lowers storage parameter rebases through semantic IR offsets", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void storageRebase(uint *data, uint *out) {
    data = data + blockIdx.x * 2;
    out[blockIdx.x * blockDim.x + threadIdx.x] = data[threadIdx.x];
  }`, { workgroupSize: [2, 1, 1] });
      const input = { buffers: { data: new Uint32Array([10, 11, 20, 21]), out: new Uint32Array(4) } };
      const launch = { gridDim: [2, 1, 1] as const, blockDim: [2, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("data__bg_ptr_offset");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([10, 11, 20, 21]);
      expect([...result.buffers.out as Uint32Array]).toEqual([10, 11, 20, 21]);
    });

  it("lowers offset local pointer dereferences through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void offsetPointerDeref(uint *data, uint *out) {
    uint *cursor = data + threadIdx.x * 3;
    out[threadIdx.x] = *cursor;
    out[threadIdx.x + 2] = *(cursor + 1);
  }`, { workgroupSize: [2, 1, 1] });
      const input = { buffers: { data: new Uint32Array([10, 11, 12, 20, 21, 22]), out: new Uint32Array(4) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("data[(local_id.x * 3u)]");
      expect(compiled.wgsl).toContain("data[((local_id.x * 3u) + 1u)]");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([10, 20, 11, 21]);
      expect([...result.buffers.out as Uint32Array]).toEqual([10, 20, 11, 21]);
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

  it("lowers update expressions through device helper pointer params", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void incAt(uint* ptr, uint index) {
    ptr[index]++;
  }

  __global__ void kernel(uint* out) {
    for (uint i = 0; i < 1u; i++) {
      uint4 index = make_uint4(i);
      if (index.x == 99u) {
        out[0] = index.x;
      }
    }
    incAt(out, 1u);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array([5, 7]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.out as Uint32Array]).toEqual([5, 8]);
      expect(compiled.wgsl).toContain("bg_ptr_write_u32(ptr_buffer, u32((i32(ptr_base) + i32(index))), (bg_ptr_read_u32(ptr_buffer, u32((i32(ptr_base) + i32(index)))) + 1u))");
      expect(compiled.wgsl).not.toContain("u32(index * vec4<u32>");
      expect(compiled.wgsl).not.toContain("bg_ptr_read_u32(ptr_buffer, u32((i32(ptr_base) + i32(index)))) =");
    });

  it("emits pointer helpers for local storage pointer aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void aliased_rows(float* out, const float* inp, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
      const float* in_row = inp + row * cols;
      float* out_row = out + row * cols;
      for (int col = 0; col < cols; col++) {
        out_row[col] = in_row[col] + 1.0f;
      }
    }
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Float32Array(6),
            inp: new Float32Array([1, 2, 3, 4, 5, 6]),
          },
          scalars: { rows: 2, cols: 3 },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect([...result.buffers.out as Float32Array]).toEqual([2, 3, 4, 5, 6, 7]);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("bg_ptr_read_f32");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
      expect(compiled.wgsl).toContain("out[u32(((row * bg_uniforms.cols) + col))]");
    });

  it("lowers mutable local pointer aliases declared in for initializers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void pointer_for_init(const float* inp, float* out) {
    if (threadIdx.x == 0) {
      float acc = 0.0f;
      int i = 0;
      for (const float* p = inp; i < 3; ++i, ++p) {
        acc += *p;
      }
      out[0] = acc;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { inp: new Float32Array([2, 3, 5]), out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer-for-init");
      expect(compiled.wgsl).toContain("loop {");
      expect(compiled.wgsl).toContain("continuing {");
      expect([...result.buffers.out as Float32Array]).toEqual([10]);
    });

  it("keeps direct scalar storage updates off pointer helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void incKernel(int *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
      data[i]++;
    }
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { data: new Int32Array([0, 1, 2, 3]) }, scalars: { n: 4 } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect([...result.buffers.data as Int32Array]).toEqual([1, 2, 3, 4]);
      expect(compiled.wgsl).toMatch(/data(?:\[i\]|\[u32\(i\)\]) = \((?:i32\()?data(?:\[i\]|\[u32\(i\)\])\)? \+ 1\)/u);
      expect(compiled.wgsl).not.toContain("bg_ptr_read_i32");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_i32");
    });

  it("lowers one-dimensional shared memory through helper pointer params", () => {
      const compiled = compileCudaLiteKernel(SHARED_POINTER_HELPERS, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("fn readTile(tile__bg_shared_ptr: ptr<workgroup, array<f32, 4>>, tile__bg_shared_ptr_base: u32");
      expect(compiled.wgsl).toContain("return (*tile__bg_shared_ptr)[(tile__bg_shared_ptr_base + u32(offset))]");
      expect(compiled.wgsl).toContain("fn writeTile(tile__bg_shared_ptr: ptr<workgroup, array<f32, 4>>, tile__bg_shared_ptr_base: u32");
      expect(compiled.wgsl).toContain("writeTile(&tile, 0u, tid");
      expect([...result.buffers.out as Float32Array]).toEqual([4, 3, 2, 1]);
    });

  it("supports __device__ arrays as storage-backed device pointer arguments", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float d_CallValue[4];

  __device__ void setCallValue(float* values, int index, float value) {
    values[index] = value;
  }

  __global__ void globals_array(float* out) {
    int i = threadIdx.x;
    setCallValue(d_CallValue, i, (float)i + 0.5f);
    out[i] = d_CallValue[i];
  }`, { workgroupSize: [4, 1, 1] });

      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect([...result.buffers.out as Float32Array]).toEqual([0.5, 1.5, 2.5, 3.5]);
      expect([...result.buffers.d_CallValue as Float32Array]).toEqual([0.5, 1.5, 2.5, 3.5]);
      expect(compiled.wgsl).toContain("var<storage, read_write> d_CallValue: array<f32>;");
      expect(compiled.wgsl).toContain("fn setCallValue(");
      expect(compiled.wgsl).toContain("d_CallValue[");
    });

  it("models initialized __device__ arrays through semantic IR storage bindings", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint seed[2] = {3u, 5u};

  __global__ void initializedDeviceGlobalArray(uint* out) {
    int i = threadIdx.x;
    out[i] = seed[i];
  }`, { workgroupSize: [2, 1, 1] });

      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      const seed = compiled.kernelIr.memory.find((symbol) => symbol.name === "seed");

      expect(seed?.kind).toBe("device-global");
      expect(seed?.initialized).toBe(true);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var<storage, read_write> seed: array<u32>;");
      expect(Object.keys(deviceGlobalBufferInputs(compiled, { buffers: { out: new Uint32Array(2) } }))).toEqual(["seed"]);
      expect([...result.buffers.out as Uint32Array]).toEqual([3, 5]);
      expect([...result.buffers.seed as Uint32Array]).toEqual([3, 5]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([3, 5]);
      expect([...semanticResult.buffers.seed as Uint32Array]).toEqual([3, 5]);
    });

  it("packs wider pointer views over device global byte storage", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uchar gScratch[8];

  __global__ void globalByteFloatOverlay(float* out) {
    if (threadIdx.x == 0) {
      float *value = (float *)&gScratch[0];
      value[0] = 1.0f;
      value[1] = 2.0f;
      out[0] = value[0];
      out[1] = value[1];
    }
  }`, { workgroupSize: [1, 1, 1] });

      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.out as Float32Array]).toEqual([1, 2]);
      expect(result.buffers.gScratch).toBeInstanceOf(Uint32Array);
      expect(compiled.wgsl).toContain("var<storage, read_write> gScratch: array<u32>;");
      expect(compiled.wgsl).toContain("bg_ptr_write_f32");
      expect(compiled.wgsl).toContain(">> 2u");
    });

  it("lowers multi-dimensional shared memory through helper pointer params", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float readTile(float* ptr, int i) {
    return ptr[i];
  }

  __device__ void writeTile(float* ptr, int i, float value) {
    ptr[i] = value;
  }

  __global__ void shared_pointer_2d(float* out) {
    __shared__ float tile[2][3];
    int tid = threadIdx.x;
    if (tid < 6) {
      writeTile(&tile[0][0], tid, 3.0f);
    }
    __syncthreads();
    if (tid < 6) {
      out[tid] = readTile(&tile[0][0], tid);
    }
  }`, { workgroupSize: [6, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(6) } },
        { gridDim: [1, 1, 1], blockDim: [6, 1, 1] },
      );

      expect(compiled.wgsl).toContain("return tile[min((((index) / 3u) % 2u), 1u)][min((index % 3u), 2u)];");
      expect(compiled.wgsl).toContain("tile[min((((index) / 3u) % 2u), 1u)][min((index % 3u), 2u)] = value;");
      expect([...result.buffers.out as Float32Array]).toEqual([3, 3, 3, 3, 3, 3]);
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
    });

  it("lowers mixed local and storage helper pointer calls through local specializations", () => {
      const source = `
  __device__ void writeMaybeLocal(float* ptr, float value) {
    ptr[0] = value;
  }

  __global__ void mixedPointers(float* out, int pickStorage) {
    float scratch[1];
    float* ptrs[1];
    ptrs[0] = &scratch[0];
    if (pickStorage) {
      writeMaybeLocal(out, 2.0f);
    } else {
      writeMaybeLocal(ptrs[0], 1.0f);
    }
    out[1] = scratch[0];
  }`;
      const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
      const localResult = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) }, scalars: { pickStorage: 0 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const storageResult = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) }, scalars: { pickStorage: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-device-pointer-param");
      expect(compiled.wgsl).toContain("fn writeMaybeLocal(ptr_buffer_arg: u32, ptr_base_arg: u32");
      expect(compiled.wgsl).toContain("fn writeMaybeLocal__bg_localptr_ptr(ptr: ptr<function, f32>");
      expect(compiled.wgsl).toContain("writeMaybeLocal(0u, 0u");
      expect(compiled.wgsl).toContain("writeMaybeLocal__bg_localptr_ptr(&scratch[ptrs_base[u32(0)]]");
      expect([...localResult.buffers.out as Float32Array]).toEqual([0, 1]);
      expect([...storageResult.buffers.out as Float32Array]).toEqual([2, 0]);
    });

  it("lowers nested helper calls with local scalar pointer params", () => {
      const source = `
  __device__ uint bump(uint* state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
  }

  __device__ uint bump_twice(uint* state) {
    bump(state);
    return bump(state);
  }

  __global__ void nestedLocalPointer(uint* out) {
    uint state = 7u + threadIdx.x;
    out[threadIdx.x] = bump_twice(&state);
  }`;
      const compiled = compileCudaLiteKernel(source, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-device-pointer-param");
      expect(compiled.wgsl).toContain("fn bump(state: ptr<function, u32>");
      expect(compiled.wgsl).toContain("fn bump_twice(state: ptr<function, u32>");
      expect(compiled.wgsl).toContain("bump(state, local_id");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        (((7 * 1664525 + 1013904223) >>> 0) * 1664525 + 1013904223) >>> 0,
        (((8 * 1664525 + 1013904223) >>> 0) * 1664525 + 1013904223) >>> 0,
      ]);
    });

  it("lowers fixed thread-local arrays through reference and WGSL", () => {
      const compiled = compileCudaLiteKernel(LOCAL_ARRAY, { workgroupSize: [4, 1, 1] });
      expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var tmp: array<array<f32, 2>, 2>;");

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
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([3, 4, 5, 6]);
      expect([...result.buffers.out as Float32Array]).toEqual([3, 4, 5, 6]);
    });

  it("lowers braced thread-local array initializers through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localInit(float* out) {
    int tid = threadIdx.x;
    float vals[2][2] = {1.0f, 2.0f, 3.0f};
    out[tid] = vals[tid][0] + vals[tid][1];
  }
  `, { workgroupSize: [2, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("vals[0u][0u] = 1.0;");
      expect(compiled.wgsl).toContain("vals[1u][0u] = 3.0;");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([3, 3]);
      expect([...result.buffers.out as Float32Array]).toEqual([3, 3]);
    });

  it("lowers scalar-fill thread-local array initializers through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localScalarInit(float* out) {
    int tid = threadIdx.x;
    float vals[2][2] = 1.5f + (float)tid;
    out[tid] = vals[tid][0] + vals[tid][1];
  }
  `, { workgroupSize: [2, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-array-init");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("for (var fill_vals_0");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([3, 5]);
      expect([...result.buffers.out as Float32Array]).toEqual([3, 5]);
    });

  it("fills vector fixed local arrays through semantic reference and WGSL", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void vectorLocalFill(float4* out) {
    int tid = threadIdx.x;
    float4 regs[2];
    fill_1D_regs<float4, 2>(regs, make_float4(1.0f + (float)tid, 2.0f, 3.0f, 4.0f));
    out[tid] = make_float4(regs[tid].x, regs[tid][1], regs[tid].z, regs[tid].w);
  }
  `, { workgroupSize: [2, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-array-fill");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("regs[fill_regs_0] = vec4<f32>");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 2, 3, 4, 2, 2, 3, 4]);
      expect([...result.buffers.out as Float32Array]).toEqual([1, 2, 3, 4, 2, 2, 3, 4]);
    });

  it("lowers integer vector fixed local arrays through semantic IR", () => {
      const intCompiled = compileCudaLiteKernel(`
  __global__ void intVectorLocalArray(int4* out) {
    int tid = threadIdx.x;
    int4 vals[2] = { make_int4(1, 2, 3, 4), make_int4(5, 6, 7, 8) };
    out[tid] = vals[tid];
  }
  `, { workgroupSize: [2, 1, 1] });
      const uintCompiled = compileCudaLiteKernel(`
  __global__ void uintVectorLocalArray(uint4* out) {
    int tid = threadIdx.x;
    uint4 vals[2] = make_uint4(1u + (uint)tid, 2u, 3u, 4u);
    out[tid] = vals[tid];
  }
  `, { workgroupSize: [2, 1, 1] });

      const intSemantic = runCompiledKernelSemanticReference(
        intCompiled,
        { buffers: { out: new Int32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      const uintSemantic = runCompiledKernelSemanticReference(
        uintCompiled,
        { buffers: { out: new Uint32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(intCompiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(intCompiled.wgslLegalizedKernelIr)).toBe(true);
      expect(intCompiled.wgsl).toContain("var vals: array<vec4<i32>, 2>;");
      expect([...intSemantic.buffers.out as Int32Array]).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
      expect(canRunCompiledKernelSemanticReference(uintCompiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(uintCompiled.wgslLegalizedKernelIr)).toBe(true);
      expect(uintCompiled.wgsl).toContain("var vals: array<vec4<u32>, 2>;");
      expect([...uintSemantic.buffers.out as Uint32Array]).toEqual([1, 2, 3, 4, 2, 2, 3, 4]);
    });

  it("fills fixed local arrays through semantic reference and WGSL", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localFill(float* out) {
    int tid = threadIdx.x;
    float regs[2][2];
    fill_2D_regs<float, 2, 2>(regs, 7.0f);
    out[tid] = regs[tid][1];
  }
  `, { workgroupSize: [2, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-array-fill");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("regs[fill_regs_0][fill_regs_1] = 7.0;");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([7, 7]);
      expect([...result.buffers.out as Float32Array]).toEqual([7, 7]);
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

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("0u < u32(n[0u])");
      expect([...result.buffers.out as Float32Array]).toEqual([1]);
    });

  it("supports NULL-initialized local pointers that rebind to storage", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void nullablePointer(float* out, int n) {
    float *p = NULL;
    int i = threadIdx.x;
    if (i < n) {
      p = out + i;
      *p = (float)(i + 1);
    }
  }
  `, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(4) }, scalars: { n: 3 } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("p_buffer");
      expect([...result.buffers.out as Float32Array]).toEqual([1, 2, 3, 0]);
    });

  it("lowers NULL-initialized pointer aliases rebound inside nested loops", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void nestedLoopPointer(float* out) {
    float *p = NULL;
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 1; j++) {
        p = out + i + j;
        *p = 7.0f;
      }
    }
  }
  `, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.kernelIr.operations).toContainEqual(expect.objectContaining({
        kind: "declare",
        target: expect.objectContaining({ name: "p", pointer: true }),
      }));
      expect([...result.buffers.out as Float32Array]).toEqual([7]);
    });

  it("keeps same-named direct storage pointer aliases distinct across C block scopes", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void scopedPointerHandles(float* out) {
    if (threadIdx.x == 0) {
      float *value = &out[0];
      value += 0;
      value[0] = 1.0f;
    }
    if (threadIdx.x == 0) {
      float *value = &out[1];
      value += 0;
      value[0] = out[0] + 1.0f;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var value:");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 2]);
      expect([...result.buffers.out as Float32Array]).toEqual([1, 2]);
    });

  it("supports conditional local read pointers derived from const storage", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void constReadPointer(const float* a, const float* b, float* out, int pick_b) {
    const float *p = pick_b ? (&b[1] + 1) : (&a[0] + 2);
    out[0] = *p;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            a: new Float32Array([1, 2, 3]),
            b: new Float32Array([4, 5, 6]),
            out: new Float32Array(1),
          },
          scalars: { pick_b: 1 },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("select(");
      expect([...result.buffers.out as Float32Array]).toEqual([6]);
    });

  it("allows const storage addresses for read-only device pointer params", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float read_one(const float* p) {
    return p[0];
  }
  __global__ void constReadParam(const float* input, float* out) {
    out[0] = read_one(&input[1]);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { input: new Float32Array([2, 9]), out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("read_one(0u, 1u");
      expect([...result.buffers.out as Float32Array]).toEqual([9]);
    });

  it("decays fixed local arrays into function-local pointer params", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float sum_local(float *values) {
    return values[0] + values[1];
  }
  __global__ void localArrayDecay(float *out) {
    float values[2];
    values[0] = 2.0f;
    values[1] = 5.0f;
    out[0] = sum_local(values);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("fn sum_local(values: ptr<function, array<f32, 2>>");
      expect(compiled.wgsl).toContain("sum_local(&values,");
      expect(compiled.wgsl).toContain("values[0u] = 2.0;");
      expect([...result.buffers.out as Float32Array]).toEqual([7]);
    });

  it("supports fixed local arrays declared inside storage-pointer helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void weighted_pair(const float *input, float *out) {
    float weights[3];
    weights[0] = 1.0f;
    weights[1] = 2.0f;
    weights[2] = 3.0f;
    out[0] = input[0] * weights[0] + input[1] * weights[1] + input[2] * weights[2];
  }
  __global__ void helperLocalArray(const float *input, float *out) {
    if (threadIdx.x < 1) weighted_pair(input, out);
  }`, { workgroupSize: [1, 1, 1] });
      const input = { buffers: { input: new Float32Array([2, 3, 4]), out: new Float32Array(1) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(compiled.wgsl).toContain("var weights: array<f32, 3>;");
      expect([...result.buffers.out as Float32Array]).toEqual([20]);
    });

  it("lowers local pointer aliases into fixed local arrays through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerAlias(float *out) {
    float values[2];
    values[0] = 2.0f;
    values[1] = 5.0f;
    float *p = &values[1];
    out[0] = p[0];
    out[1] = *p;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect([...result.buffers.out as Float32Array]).toEqual([5, 5]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([5, 5]);
    });

  it("lowers explicit shared-memory pointer aliases through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void sharedPointerAlias(float *out) {
  __shared__ float values[2];
  int tid = threadIdx.x;
  float *slot = &values[tid];
  *slot = (float)(tid + 1);
  __syncthreads();
  out[tid] = *slot;
}`, { workgroupSize: [2, 1, 1] });
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const input = { buffers: { out: new Float32Array(2) } };
    const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
    const result = runCompiledKernelReference(compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).not.toContain("var slot:");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 2]);
    expect([...result.buffers.out as Float32Array]).toEqual([1, 2]);
  });

  it("decays fixed local arrays into local pointer aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerDecay(float *out) {
    float values[2];
    values[0] = 3.0f;
    values[1] = 8.0f;
    float *p = values;
    out[0] = p[1];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect([...result.buffers.out as Float32Array]).toEqual([8]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([8]);
    });

  it("lowers NULL-initialized local pointers rebound to fixed local arrays through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerNullRebind(float *out) {
    float values[2];
    values[0] = 3.0f;
    values[1] = 9.0f;
    float *p = NULL;
    p = values;
    out[0] = p[1];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect([...result.buffers.out as Float32Array]).toEqual([9]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([9]);
    });

  it("lowers local pointer assignments into fixed local arrays through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerAssignment(float *out) {
    float values[2];
    values[0] = 4.0f;
    values[1] = 9.0f;
    float *p;
    p = values;
    out[0] = p[1];
    p = &values[0];
    out[1] = *p;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect([...result.buffers.out as Float32Array]).toEqual([9, 4]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([9, 4]);
    });

  it("lowers local pointer alias copies through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerCopy(float *out, int pick) {
    float values[3];
    values[0] = 2.0f;
    values[1] = 5.0f;
    values[2] = 11.0f;
    float *p = &values[2];
    float *q = p;
    float *r = pick != 0 ? q : values;
    out[0] = r[0];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) }, scalars: { pick: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(1) }, scalars: { pick: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var q:");
      expect(compiled.wgsl).not.toContain("var r:");
      expect([...result.buffers.out as Float32Array]).toEqual([11]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([11]);
    });

  it("lowers local pointer arithmetic initializers through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerArithmeticInit(float *out) {
    float values[4];
    values[0] = 2.0f;
    values[1] = 5.0f;
    values[2] = 11.0f;
    values[3] = 17.0f;
    float *a = values + 1;
    float *b = 2 + values;
    float *c = &values[3] - 1;
    out[0] = a[0];
    out[1] = b[0];
    out[2] = c[0];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var a:");
      expect(compiled.wgsl).not.toContain("var b:");
      expect(compiled.wgsl).not.toContain("var c:");
      expect([...result.buffers.out as Float32Array]).toEqual([5, 11, 11]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([5, 11, 11]);
    });

  it("lowers assume-aligned local pointer aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerAssumeAligned(float *out) {
    float values[3];
    values[0] = 2.0f;
    values[1] = 5.0f;
    values[2] = 11.0f;
    float *p = (float*)__builtin_assume_aligned(values + 1, 16);
    out[0] = p[1];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect([...result.buffers.out as Float32Array]).toEqual([11]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([11]);
    });

  it("lowers address-of local pointer alias indices through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerAliasAddress(float *out) {
    float values[4];
    values[0] = 2.0f;
    values[1] = 5.0f;
    values[2] = 11.0f;
    values[3] = 17.0f;
    float *p = values + 1;
    float *q = &p[2];
    out[0] = q[0];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect(compiled.wgsl).not.toContain("var q:");
      expect([...result.buffers.out as Float32Array]).toEqual([17]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([17]);
    });

  it("lowers multidimensional local array pointer aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerMultidim(float *out) {
    float values[2][2];
    values[0][0] = 2.0f;
    values[0][1] = 5.0f;
    values[1][0] = 11.0f;
    values[1][1] = 17.0f;
    float *p = &values[0][1];
    out[0] = p[0];
    out[1] = p[2];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect([...result.buffers.out as Float32Array]).toEqual([5, 17]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([5, 17]);
    });

  it("lowers address-of dereferenced local pointer aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerAddressOfDeref(float *out) {
    float values[3];
    values[0] = 2.0f;
    values[1] = 5.0f;
    values[2] = 11.0f;
    float *p = values + 1;
    float *q = &*p;
    out[0] = q[0];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect(compiled.wgsl).not.toContain("var q:");
      expect([...result.buffers.out as Float32Array]).toEqual([5]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([5]);
    });

  it("lowers local pointer increment and decrement updates through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerUpdates(float *out) {
    float values[3];
    values[0] = 2.0f;
    values[1] = 5.0f;
    values[2] = 11.0f;
    float *p = values;
    p++;
    out[0] = p[0];
    --p;
    out[1] = p[0];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect([...result.buffers.out as Float32Array]).toEqual([5, 2]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([5, 2]);
    });

  it("lowers local pointer conditionals into fixed local arrays through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerConditional(float *out, int pick) {
    float values[2];
    values[0] = 6.0f;
    values[1] = 11.0f;
    float *p = pick != 0 ? &values[1] : values;
    out[0] = p[0];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) }, scalars: { pick: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(1) }, scalars: { pick: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-pointer-conditional");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("select(");
      expect([...result.buffers.out as Float32Array]).toEqual([11]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([11]);
    });

  it("lowers local pointer rebasing into fixed local arrays through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerRebase(float *out, int step) {
    float values[3];
    values[0] = 2.0f;
    values[1] = 7.0f;
    values[2] = 13.0f;
    float *p = values;
    p += step;
    out[0] = p[0];
    p -= 1;
    out[1] = p[0];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) }, scalars: { step: 2 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(2) }, scalars: { step: 2 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-pointer-assignment");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect([...result.buffers.out as Float32Array]).toEqual([13, 7]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([13, 7]);
    });

  it("lowers branch-merged local pointer aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerBranchMerge(float *out, int pick) {
    float values[3];
    values[0] = 2.0f;
    values[1] = 5.0f;
    values[2] = 11.0f;
    float *p = values;
    if (pick != 0) {
      p = &values[2];
    }
    out[0] = p[0];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) }, scalars: { pick: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(1) }, scalars: { pick: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("select(");
      expect([...result.buffers.out as Float32Array]).toEqual([11]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([11]);
    });

  it("lowers block-updated local pointer aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerBlockUpdate(float *out) {
    float values[3];
    values[0] = 2.0f;
    values[1] = 5.0f;
    values[2] = 11.0f;
    float *p = values;
    {
      p = &values[1];
      p += 1;
    }
    out[0] = p[0];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-pointer");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect([...result.buffers.out as Float32Array]).toEqual([11]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([11]);
    });

  it("lowers local pointer alias comparisons through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerCompare(uint *out, int pick) {
    float values[2];
    float *a = values;
    float *b = pick != 0 ? &values[1] : values;
    out[0] = a == b ? 1u : 0u;
    out[1] = a != b ? 1u : 0u;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(2) }, scalars: { pick: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(2) }, scalars: { pick: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-pointer-pointer-comparison");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect([...result.buffers.out as Uint32Array]).toEqual([0, 1]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([0, 1]);
    });

  it("lowers local pointer alias null comparisons through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerNullCompare(uint *out) {
    uint values[2];
    uint *p = values;
    out[0] = p != NULL ? 1u : 0u;
    out[1] = nullptr == p ? 1u : 0u;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-pointer-pointer-comparison");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect([...result.buffers.out as Uint32Array]).toEqual([1, 0]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1, 0]);
    });

  it("lowers nullable local pointer conditionals through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void nullableLocalPointerConditional(uint *out, int pick) {
    uint values[2];
    values[0] = 4u;
    values[1] = 9u;
    uint *p = pick != 0 ? values : NULL;
    out[0] = p != NULL ? 1u : 0u;
    out[1] = p == nullptr ? 1u : 0u;
    if (p != NULL) {
      out[2] = p[1];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(3) }, scalars: { pick: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(3) }, scalars: { pick: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const nullResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(3) }, scalars: { pick: 0 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-pointer-conditional");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var p:");
      expect([...result.buffers.out as Uint32Array]).toEqual([1, 0, 9]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1, 0, 9]);
      expect([...nullResult.buffers.out as Uint32Array]).toEqual([0, 1, 0]);
    });

  it("compares nullable local pointer aliases by validity through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void nullableLocalPointerEquality(uint *out, int pick) {
    uint values[2];
    uint *base = values;
    uint *maybe = pick != 0 ? values : NULL;
    out[0] = maybe == base ? 1u : 0u;
    out[1] = maybe != base ? 1u : 0u;
  }`, { workgroupSize: [1, 1, 1] });
      const nonNullResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(2) }, scalars: { pick: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const nullResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(2) }, scalars: { pick: 0 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-pointer-pointer-comparison");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("var maybe:");
      expect([...nonNullResult.buffers.out as Uint32Array]).toEqual([1, 0]);
      expect([...nullResult.buffers.out as Uint32Array]).toEqual([0, 1]);
    });

  it("lowers local pointer alias ordering comparisons through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerOrder(uint *out) {
    float values[4];
    float *a = &values[1];
    float *b = &values[3];
    out[0] = a < b ? 1u : 0u;
    out[1] = b > a ? 1u : 0u;
    out[2] = a <= a ? 1u : 0u;
    out[3] = b >= a ? 1u : 0u;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-pointer-pointer-comparison");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect([...result.buffers.out as Uint32Array]).toEqual([1, 1, 1, 1]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1, 1, 1, 1]);
    });

  it("lowers local pointer alias differences through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerDifference(int *out, int pick) {
    float values[4];
    float *base = values;
    float *left = pick != 0 ? &values[3] : &values[1];
    out[0] = left - base;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(1) }, scalars: { pick: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Int32Array(1) }, scalars: { pick: 1 } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-pointer-difference");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect([...result.buffers.out as Int32Array]).toEqual([3]);
      expect([...semanticResult.buffers.out as Int32Array]).toEqual([3]);
    });

  it("stores modeled memory pointers in fixed local pointer arrays", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float read_x(float3 *value) {
    return (*value).x;
  }
  __global__ void pointerArray(float *out) {
    __shared__ float3 values[3];
    values[0] = make_float3(2.0f, 4.0f, 6.0f);
    float3 *p[3];
    p[0] = &values[0];
    out[0] = read_x(p[0]);
    out[1] = (*p[0]).z;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).not.toContain("p_buffer");
      expect(compiled.wgsl).toContain("read_x(&values, 0u");
      expect(compiled.wgsl).toContain("out[1u] = values[0u].z;");
      expect([...result.buffers.out as Float32Array]).toEqual([2, 6]);
    });

  it("supports corpus-shaped byte storage reinterpret local pointers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void byteReinterpret(int* out) {
    __shared__ uchar scratch[16];
    int lane = threadIdx.x;
    if (lane == 0) {
      scratch[0] = (uchar)1;
      scratch[1] = (uchar)2;
      scratch[2] = (uchar)3;
      scratch[3] = (uchar)4;
    }
    __syncthreads();
    int *shared_words = (int *)&scratch[0];
    if (lane == 0) {
      shared_words[1] = 9;
      out[0] = shared_words[0];
      out[1] = shared_words[1];
    }
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            out: new Int32Array(2),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(true);
      expect(compiled.wgsl).toContain("var<workgroup> scratch: array<atomic<u32>, 4>;");
      expect(compiled.wgsl).toContain("atomicLoad(&scratch");
      expect(compiled.wgsl).toContain("bg_semantic_packed_shared_u8_store");
      expect([...result.buffers.out as Int32Array]).toEqual([0x04030201, 9]);

      const storageCompiled = compileCudaLiteKernel(`
  __global__ void storageByteReinterpret(const uchar* input, int* out) {
    int4 *lane_ptr = (int4 *)(input + threadIdx.x);
    lane_ptr = (int4 *)((uchar *)lane_ptr + 4);
    if (threadIdx.x == 0) {
      out[0] = (*lane_ptr).x;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const storageResult = runCompiledKernelSemanticReference(
        storageCompiled,
        { buffers: { input: new Uint32Array([0, 0, 0, 0, 1, 2, 3, 4]), out: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect(canRunCompiledKernelSemanticReference(storageCompiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(storageCompiled.wgslLegalizedKernelIr)).toBe(true);
      expect(storageCompiled.wgsl).not.toContain("lane_ptr");
      expect(storageCompiled.wgsl).toContain("bitcast<i32>");
      expect([...storageResult.buffers.out as Int32Array]).toEqual([0x04030201]);
    });

  it("preserves packed byte-vector cast width over direct byte storage", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void packedByteVector(uchar* bytes) {
    uchar* target = bytes + 1;
    *(uchar2*)target = make_uchar2(300u, 511u);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { bytes: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const store = compiled.kernelIr.operations.find((operation) => operation.kind === "store");

      expect(store?.kind === "store" ? store.target.packedByteLanes : undefined).toBe(2);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("& 255u");
      expect([...result.buffers.bytes as Uint32Array]).toEqual([0, 44, 255, 0]);

      const wideCast = compileCudaLiteKernel(`
  __global__ void wideVectorOverBytes(uchar* bytes) {
    *(uint2*)bytes = make_uint2(1u, 2u);
  }`, { workgroupSize: [1, 1, 1] });
      expect(canEmitSemanticKernelIrWgsl(wideCast.wgslLegalizedKernelIr)).toBe(false);
    });

  it("preserves raw word width through assigned local pointer aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void pitchedFloat(uchar* bytes, int pitch) {
    int y = threadIdx.y;
    float* pixel;
    pixel = (float*)(bytes + y * pitch) + threadIdx.x;
    pixel[0] = pixel[0] + 1.0f;
  }`, { workgroupSize: [2, 1, 1] });
      const input = {
        buffers: {
          bytes: new Uint32Array([0, 0, 128, 63, 0, 0, 0, 64]),
        },
        scalars: { pitch: 8 },
      };
      const result = runCompiledKernelSemanticReference(
        compiled,
        input,
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("bitcast<u32>");
      expect([...result.buffers.bytes as Uint32Array]).toEqual([0, 0, 0, 64, 0, 0, 64, 64]);
    });

  it("loads raw words from byte storage into packed local byte vectors", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void rawByteWord(const uchar* bytes, uint* out) {
    uchar4 lanes;
    *(uint32_t*)&lanes = *(uint32_t*)(bytes + 1);
    out[0] = lanes.x;
    out[1] = lanes.y;
    out[2] = lanes.z;
    out[3] = lanes.w;
  }`, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          bytes: new Uint32Array([99, 1, 2, 254, 255, 77]),
          out: new Uint32Array(4),
        },
      };
      const result = runCompiledKernelSemanticReference(
        compiled,
        input,
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("<< 24u");
      expect(compiled.wgsl).toContain("lanes.w =");
      expect([...result.buffers.out as Uint32Array]).toEqual([1, 2, 254, 255]);
    });

  it("packs wider pointer views over storage byte params into byte-offset words", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void storageByteFloatOverlay(uchar *scratch, float *out) {
    if (threadIdx.x == 0) {
      float *value = (float *)&scratch[0];
      value[0] = 1.0f;
      value[1] = 2.0f;
      out[0] = value[0];
      out[1] = value[1];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            scratch: new Uint32Array(8),
            out: new Float32Array(2),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("bg_raw_word_");
      expect(compiled.wgsl).toContain(">> 24u");
      expect([...result.buffers.out as Float32Array]).toEqual([1, 2]);
    });

  it("passes direct byte storage through uchar device helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void bump_storage_byte(uchar *bytes, uint index) {
    bytes[index]++;
  }
  __global__ void storageByteHelper(uchar *bytes, uint *out) {
    bump_storage_byte(bytes, 1u);
    out[0] = bytes[1];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { bytes: new Uint32Array([3, 7]), out: new Uint32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bump_storage_byte");
      expect([...result.buffers.bytes as Uint32Array]).toEqual([3, 8]);
      expect([...result.buffers.out as Uint32Array]).toEqual([8]);
    });

  it("resolves fixed local pointer-array slots into shared vector aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float3 add_pointer_array_values(float3 *a, float3 *b, float3 *c) {
    return *a + *b + *c;
  }
  __global__ void localPointerArray(float *out) {
    __shared__ float3 values[3];
    values[0] = make_float3(1.0f, 2.0f, 3.0f);
    values[1] = make_float3(4.0f, 5.0f, 6.0f);
    values[2] = make_float3(7.0f, 8.0f, 9.0f);
    float3 *v[3];
    v[0] = &values[0];
    v[1] = &values[1];
    v[2] = &values[2];
    float3 sum = add_pointer_array_values(v[0], v[1], v[2]);
    out[0] = sum.x;
    out[1] = sum.y;
    out[2] = sum.z;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect([...result.buffers.out as Float32Array]).toEqual([12, 15, 18]);
    });

  it("scales helper pointer params when byte storage is viewed as wider values", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void write_storage_byte_float(float *value, float *out) {
    value[0] = 1.0f;
    value[1] = 2.0f;
    out[0] = value[0];
    out[1] = value[1];
  }

  __global__ void storageByteFloatHelperOverlay(uchar *scratch, float *out) {
    if (threadIdx.x == 0) {
      write_storage_byte_float((float *)&scratch[0], out);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            scratch: new Uint32Array(2),
            out: new Float32Array(2),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("select(u32(1), (u32(1) * 4u), (value_buffer == 0u))");
      expect([...result.buffers.out as Float32Array]).toEqual([1, 2]);
    });

  it("bitcasts float views over packed shared byte storage", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void byteFloatReinterpret(float* out) {
    __shared__ uchar scratch[4];
    if (threadIdx.x == 0) {
      uint *word = (uint *)&scratch[0];
      word[0] = 0x3f800000u;
      float *value = (float *)&scratch[0];
      out[0] = value[0];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(true);
      expect(compiled.wgsl).toContain("bitcast<f32>(atomicLoad(&scratch");
      expect(compiled.wgsl).toContain("atomicStore(&scratch");
      expect([...result.buffers.out as Float32Array]).toEqual([1]);
    });

  it("lowers sample-shaped vector pointer-array helper flow", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float3 calcNormal(float3 *a, float3 *b, float3 *c) {
    return *a + *b + *c;
  }
  __global__ void pointerArrayTriangle(float4 *out) {
    float3 vertlist[3];
    vertlist[0] = make_float3(1.0f, 2.0f, 3.0f);
    vertlist[1] = make_float3(4.0f, 5.0f, 6.0f);
    vertlist[2] = make_float3(7.0f, 8.0f, 9.0f);
    float3 *v[3];
    v[0] = &vertlist[0];
    v[1] = &vertlist[1];
    v[2] = &vertlist[2];
    float3 n = calcNormal(v[0], v[1], v[2]);
    out[0] = make_float4(*v[0], 1.0f);
    out[1] = make_float4(n, 0.0f);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("a: ptr<function, array<vec3<f32>, 3>>, a_base: u32");
      expect(compiled.wgsl).toContain("(*a)[(a_base + 0u)]");
      expect(compiled.wgsl).toContain("calcNormal(&vertlist, 0u, &vertlist, 1u, &vertlist, 2u");
      expect([...result.buffers.out as Float32Array]).toEqual([1, 2, 3, 1, 12, 15, 18, 0]);
    });

  it("rejects non-pointer assignments to pointer-array elements", () => {
      expect(() => compileCudaLiteKernel(`
  __global__ void badPointerArray() {
    __shared__ float3 values[1];
    float3 *p[1];
    p[0] = values[0];
  }`, { workgroupSize: [1, 1, 1] })).toThrow(CudaLiteCompilerError);
    });

  it("rejects const storage addresses for writable device pointer params", () => {
      expect(() => compileCudaLiteKernel(`
  __device__ void write_one(float* p) {
    p[0] = 1.0f;
  }
  __global__ void constWriteParam(const float* input) {
    write_one(&input[0]);
  }`, { workgroupSize: [1, 1, 1] })).toThrow(CudaLiteCompilerError);
    });

  it("accepts printf pointer and local array arguments as no-op debug output", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void hello(char *name) {
    char buffer[8];
    buffer[0] = 'b';
    buffer[1] = '\\0';
    printf("%s %s\\n", buffer, name);
  }`, { workgroupSize: [1, 1, 1] });

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("printf(");
    });

  it("preserves storage bits for typed pointer writes in reference mode", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void typePunnedFlag(float *scratch) {
    unsigned int *flag = (unsigned int *)(scratch + 2);
    if (threadIdx.x == 0) {
      atomicAdd(flag, 1);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { scratch: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const expected = new Float32Array(new Uint32Array([1]).buffer)[0];

      expect((result.buffers.scratch as Float32Array)[2]).toBe(expected);
    });

  it("specializes device pointer helpers for dynamic shared memory in semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void writeShared(float *values, int index, float value) {
    values[index] = value;
  }

  __global__ void sharedHelper(float *out) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    writeShared(sdata, tid, float(tid + 1));
    __syncthreads();
    out[tid] = sdata[tid];
  }`,
      { workgroupSize: [4, 1, 1], dynamicSharedMemory: { sdata: 4 } });

      const helper = compiled.kernelIr.functions.find((fn) => fn.name === "writeShared");
      expect(helper?.params[0]).toMatchObject({
        name: "values__bg_shared_ptr",
        pointer: true,
        addressSpace: "shared",
        dimensions: [4],
  });

      const legalized = legalizeSemanticKernelIrForWgsl(
        typeCheckSemanticKernelIr(validateSemanticKernelIr(compiled.kernelIr)),
      );
      const semanticWgsl = emitSemanticKernelIrWgsl(legalized).wgsl;
      expect(semanticWgsl).toContain("fn writeShared(values__bg_shared_ptr: ptr<workgroup, array<f32, 4>>");
      expect(semanticWgsl).toContain("writeShared(&sdata");
      expect(semanticWgsl).not.toContain("(*sdata)[");
    });

  it("lowers lexical blocks inside shared-pointer device helpers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void writeScoped(float *values, int index, float value) {
  {
    values[index] = value;
  }
}
__global__ void sharedHelperScoped(float *out) {
  __shared__ float values[2];
  int tid = threadIdx.x;
  writeScoped(&values[0], tid, float(tid + 3));
  __syncthreads();
  out[tid] = values[tid];
}`, { workgroupSize: [2, 1, 1] });
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const input = { buffers: { out: new Float32Array(2) } };
    const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
    const result = runCompiledKernelReference(compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn writeScoped(values__bg_shared_ptr: ptr<workgroup, array<f32, 2>>");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([3, 4]);
    expect([...result.buffers.out as Float32Array]).toEqual([3, 4]);
  });

  it("supports u32-backed bool pointer parameters", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void boolPointer(bool *flags, int *out) {
    int idx = threadIdx.x;
    bool active = flags[idx];
    if (active) {
      out[idx] = 1;
      flags[idx + 2] = false;
    } else {
      out[idx] = 0;
      flags[idx + 2] = true;
    }
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { flags: new Uint32Array([1, 0, 1, 1]), out: new Int32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("var<storage, read_write> flags: array<u32>;");
      expect(compiled.wgsl).toContain("var bg_active: bool = (flags[u32(idx)] != 0u);");
      expect([...result.buffers.out as Int32Array]).toEqual([1, 0]);
      expect([...result.buffers.flags as Uint32Array]).toEqual([1, 0, 0, 1]);
    });

  it("lowers statement remquo storage quotient aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void semanticRemquo(int *quo) {
    int *q = quo + 1;
    remquof(7.0f, 2.0f, q);
    remquo(-7.0f, 2.0f, &quo[2]);
  }
  `, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { quo: new Int32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { quo: new Int32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-remquo-quotient");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_i32");
      expect(compiled.wgsl).not.toContain("var q:");
      expect(compiled.wgsl).toContain("var bg__bg_remquo_dividend_");
      expect(compiled.wgsl).toContain("quo[1u] = select(select(i32(floor((bg__bg_remquo_dividend_");
      expect(compiled.wgsl).toContain("quo[2u] = select(select(i32(floor((bg__bg_remquo_dividend_");
      expect([...semanticResult.buffers.quo as Int32Array]).toEqual([0, 4, -4]);
      expect([...result.buffers.quo as Int32Array]).toEqual([0, 4, -4]);
    });

  it("lowers statement frexp storage exponent aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void semanticFrexp(int *expOut) {
    int *expAlias = expOut + 1;
    frexpf(9.0f, expAlias);
    frexp(0.0f, &expOut[2]);
  }
  `, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { expOut: new Int32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { expOut: new Int32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-frexp-exponent");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_i32");
      expect(compiled.wgsl).not.toContain("var expAlias:");
      expect(compiled.wgsl).toContain("var bg__bg_frexp_value_");
      expect(compiled.wgsl).toContain("expOut[1u] = select((i32(floor(log2(abs(bg__bg_frexp_value_");
      expect(compiled.wgsl).toContain("expOut[2u] = select((i32(floor(log2(abs(bg__bg_frexp_value_");
      expect([...semanticResult.buffers.expOut as Int32Array]).toEqual([0, 4, 0]);
      expect([...result.buffers.expOut as Int32Array]).toEqual([0, 4, 0]);
    });

  it("lowers statement modf storage output aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void semanticModfOut(float *out) {
    float *intpart = out + 1;
    modff(8.125f, intpart);
    modf(-2.25f, &out[2]);
  }
  `, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-modf-intpart");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
      expect(compiled.wgsl).not.toContain("var intpart:");
      expect(compiled.wgsl).toContain("var bg__bg_modf_value_");
      expect(compiled.wgsl).toContain("out[1u] = select(trunc(bg__bg_modf_value_");
      expect(compiled.wgsl).toContain("out[2u] = select(trunc(bg__bg_modf_value_");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([0, 8, -2]);
      expect([...result.buffers.out as Float32Array]).toEqual([0, 8, -2]);
    });

  it("lowers storage-only CUDA sincos output params through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void semanticSincos(float *out) {
    if (threadIdx.x == 0) {
      sincosf(0.25f, &out[0], &out[1]);
      sincospi(0.5f, &out[2], &out[3]);
    }
  }
  `, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-sincos-output");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var bg__bg_sincos_angle");
      expect(compiled.wgsl).toContain("out[0u] = sin(bg__bg_sincos_angle");
      expect(compiled.wgsl).toContain("out[1u] = cos(bg__bg_sincos_angle");
      expect(compiled.wgsl).toContain("(3.141592653589793 * 0.5)");
      const out = [...semanticResult.buffers.out as Float32Array];
      expect(out[0]).toBeCloseTo(Math.sin(0.25), 6);
      expect(out[1]).toBeCloseTo(Math.cos(0.25), 6);
      expect(out[2]).toBeCloseTo(1, 6);
      expect(out[3]).toBeCloseTo(0, 6);
      expect([...result.buffers.out as Float32Array]).toEqual([...semanticResult.buffers.out as Float32Array]);
    });

  it("lowers CUDA sincos output pointer aliases through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void semanticSincosAliases(float *out) {
    float *sinOut = out + 1;
    float *cosOut = out + 2;
    sincosf(0.25f, sinOut, cosOut);
  }
  `, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-sincos-output");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
      expect(compiled.wgsl).not.toContain("var sinOut:");
      expect([...semanticResult.buffers.out as Float32Array][1]).toBeCloseTo(Math.sin(0.25), 6);
      expect([...semanticResult.buffers.out as Float32Array][2]).toBeCloseTo(Math.cos(0.25), 6);
      expect([...result.buffers.out as Float32Array]).toEqual([...semanticResult.buffers.out as Float32Array]);
    });

  it("lowers CUDA math output params through local pointer aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void mathOutAliases(float *out, int *ints) {
    float *intpart = out + 1;
    float *sinOut = out + 2;
    float *cosOut = out + 3;
    int *expOut = ints + 1;
    int *quoOut = ints + 2;
    out[0] = modff(3.75f, intpart);
    sincosf(0.25f, sinOut, cosOut);
    out[4] = frexpf(9.0f, expOut);
    out[5] = remquof(7.0f, 2.0f, quoOut);
    out[6] = (float)ints[1];
    out[7] = (float)ints[2];
  }
  `, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(8), ints: new Int32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(8), ints: new Int32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-frexp-exponent");
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-modf-intpart");
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-sincos-output");
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-remquo-quotient");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_i32");
      expect(compiled.wgsl).not.toContain("var intpart:");
      expect(compiled.wgsl).toContain("var bg__bg_modf_value_");
      expect(compiled.wgsl).toContain("out[1u] = select(trunc(bg__bg_modf_value_");
      expect(compiled.wgsl).toContain("out[0u] = select(select((bg__bg_modf_value_");
      expect(compiled.wgsl).toContain("ints[1u] = select((i32(floor(log2(abs(bg__bg_frexp_value_");
      expect(compiled.wgsl).toContain("out[4u] = select((bg__bg_frexp_value_");
      expect(compiled.wgsl).toContain("ints[2u] = select(select(i32(floor((bg__bg_remquo_dividend_");
      expect(compiled.wgsl).toContain("out[5u] = (bg__bg_remquo_dividend_");
      const out = [...result.buffers.out as Float32Array];
      expect(out[0]).toBeCloseTo(0.75, 6);
      expect(out[1]).toBeCloseTo(3, 6);
      expect(out[2]).toBeCloseTo(Math.sin(0.25), 6);
      expect(out[3]).toBeCloseTo(Math.cos(0.25), 6);
      expect(out[4]).toBeCloseTo(0.5625, 6);
      expect(out[5]).toBeCloseTo(-1, 6);
      expect(out[6]).toBe(4);
      expect(out[7]).toBe(4);
      expect([...result.buffers.ints as Int32Array]).toEqual([0, 4, 4]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([...result.buffers.out as Float32Array]);
      expect([...semanticResult.buffers.ints as Int32Array]).toEqual([...result.buffers.ints as Int32Array]);
    });

  it("lowers dynamic CUDA modf storage outputs through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void dynamicModf(float *out, float x) {
    float *intpart = out + 1;
    out[0] = modff(x, intpart);
    float y = modf(x + out[0], &out[2]);
    out[3] = y;
    out[4] = x;
    out[4] = modff(out[4], &out[4]);
  }
  `, { workgroupSize: [1, 1, 1] });
      const input = { buffers: { out: new Float32Array(5) }, scalars: { x: -3.75 } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-modf-intpart");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("bg_modf(");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
      expect(compiled.wgsl).toContain("bg_uniforms.x");
      expect(compiled.wgsl).toContain("var bg__bg_modf_value_");
      expect([...semanticResult.buffers.out as Float32Array][0]).toBeCloseTo(-0.75, 6);
      expect([...semanticResult.buffers.out as Float32Array][1]).toBeCloseTo(-3, 6);
      expect([...semanticResult.buffers.out as Float32Array][2]).toBeCloseTo(-4, 6);
      expect([...semanticResult.buffers.out as Float32Array][3]).toBeCloseTo(-0.5, 6);
      expect([...semanticResult.buffers.out as Float32Array][4]).toBeCloseTo(-0.75, 6);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([...result.buffers.out as Float32Array]);

      for (const x of [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NaN]) {
        const expected = runCompiledKernelReference(
          compiled,
          { buffers: { out: new Float32Array(5) }, scalars: { x } },
          launch,
        );
        const actual = runCompiledKernelSemanticReference(
          compiled,
          { buffers: { out: new Float32Array(5) }, scalars: { x } },
          launch,
        );
        const expectedOut = [...expected.buffers.out as Float32Array];
        const actualOut = [...actual.buffers.out as Float32Array];
        for (const [index, value] of expectedOut.entries()) {
          const actualValue = actualOut[index]!;
          expect(Number.isNaN(actualValue) && Number.isNaN(value) || Object.is(actualValue, value)).toBe(true);
        }
      }
    });

  it("lowers dynamic CUDA frexp storage outputs through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void dynamicFrexp(float *out, int *ints, float x) {
    int *expOut = ints + 1;
    out[0] = frexpf(x, expOut);
    float mantissa = frexp(x + out[0], &ints[2]);
    out[1] = mantissa;
    out[2] = (float)ints[1];
    out[3] = (float)ints[2];
  }
  `, { workgroupSize: [1, 1, 1] });
      const input = { buffers: { out: new Float32Array(4), ints: new Int32Array(3) }, scalars: { x: -9 } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-frexp-exponent");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("bg_frexp(");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_i32");
      expect(compiled.wgsl).toContain("bg_uniforms.x");
      expect(compiled.wgsl).toContain("var bg__bg_frexp_value_");
      expect([...semanticResult.buffers.out as Float32Array][0]).toBeCloseTo(-0.5625, 6);
      expect([...semanticResult.buffers.out as Float32Array][1]).toBeCloseTo(-0.59765625, 6);
      expect([...semanticResult.buffers.out as Float32Array][2]).toBe(4);
      expect([...semanticResult.buffers.out as Float32Array][3]).toBe(4);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([...result.buffers.out as Float32Array]);
      expect([...semanticResult.buffers.ints as Int32Array]).toEqual([...result.buffers.ints as Int32Array]);

      for (const x of [0, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NaN]) {
        const expected = runCompiledKernelReference(
          compiled,
          { buffers: { out: new Float32Array(4), ints: new Int32Array(3) }, scalars: { x } },
          launch,
        );
        const actual = runCompiledKernelSemanticReference(
          compiled,
          { buffers: { out: new Float32Array(4), ints: new Int32Array(3) }, scalars: { x } },
          launch,
        );
        const expectedOut = [...expected.buffers.out as Float32Array];
        const actualOut = [...actual.buffers.out as Float32Array];
        for (const [index, value] of expectedOut.entries()) {
          const actualValue = actualOut[index]!;
          expect(Number.isNaN(actualValue) && Number.isNaN(value) || Object.is(actualValue, value)).toBe(true);
        }
        expect([...actual.buffers.ints as Int32Array]).toEqual([...expected.buffers.ints as Int32Array]);
      }
    });

  it("lowers dynamic CUDA remquo storage outputs through semantic IR for static divisors", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void dynamicRemquo(float *out, int *ints, float x) {
    int *quoOut = ints + 1;
    out[0] = remquof(x, 2.0f, quoOut);
    float rem = remquo(x + out[0], 2.0f, &ints[2]);
    out[1] = rem;
    out[2] = (float)ints[1];
    out[3] = (float)ints[2];
  }
  `, { workgroupSize: [1, 1, 1] });
      const input = { buffers: { out: new Float32Array(4), ints: new Int32Array(3) }, scalars: { x: 7 } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-remquo-quotient");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_i32");
      expect(compiled.wgsl).toContain("bg_uniforms.x");
      expect(compiled.wgsl).toContain("var bg__bg_remquo_dividend_");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([-1, 0, 4, 3]);
      expect([...semanticResult.buffers.ints as Int32Array]).toEqual([0, 4, 3]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([...result.buffers.out as Float32Array]);
      expect([...semanticResult.buffers.ints as Int32Array]).toEqual([...result.buffers.ints as Int32Array]);
    });

  it("lowers CUDA cache-hint loads and stores as plain pointer memory ops", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void cache_hint(const float* x, float* y) {
    int idx = threadIdx.x;
    if (idx < 2) {
      const float* base = x + 1;
      float value = __ldcs(base + idx);
      float direct = __ldcg(&x[idx]);
      __stcg(y + idx, value + direct + 1.0f);
    }
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: new Float32Array([0, 2, 4]), y: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { x: new Float32Array([0, 2, 4]), y: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cache-hint-address");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var value: f32 = x[");
      expect(compiled.wgsl).toContain("y[");
      expect([...semanticResult.buffers.y as Float32Array]).toEqual([3, 7]);
      expect([...result.buffers.y as Float32Array]).toEqual([3, 7]);
    });

  it("preserves vector reinterpret aliases through CUDA cache-hint loads", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void vector_cache_hint(float* out, const float* inp) {
    int idx = threadIdx.x;
    const float4* view = reinterpret_cast<const float4*>(inp);
    float4 value = __ldcs(&view[idx]);
    out[idx * 4] = value.x;
    out[idx * 4 + 1] = value.y;
    out[idx * 4 + 2] = value.z;
    out[idx * 4 + 3] = value.w;
  }`, { workgroupSize: [2, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(8),
          inp: new Float32Array([11, 22, 33, 44, 20, 40, 60, 80]),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("var value: vec4<f32> = vec4<f32>(inp[");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([11, 22, 33, 44, 20, 40, 60, 80]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([...result.buffers.out as Float32Array]);
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
      const semanticResult = runCompiledKernelSemanticReference(
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

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var<storage, read> x: array<f32>;");
      expect(compiled.wgsl).toContain("vec4<f32>");
      expect([...result.buffers.z as Float32Array]).toEqual([12, 22, 33, 46, 60, 66, 77, 90]);
      expect([...semanticResult.buffers.z as Float32Array]).toEqual([12, 22, 33, 46, 60, 66, 77, 90]);
    });

  it("lowers CUDA float4 storage elements through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void vectorStorageSemantic(const float4* x, float4* y, int n) {
    int i = threadIdx.x;
    if (i < n) {
      float4 value = x[i];
      y[i] = make_float4(value.x + 1.0f, value.y + 2.0f, value.z + 3.0f, value.w + 4.0f);
    }
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            x: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
            y: new Float32Array(8),
          },
          scalars: { n: 2 },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            x: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
            y: new Float32Array(8),
          },
          scalars: { n: 2 },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var value: vec4<f32> = vec4<f32>(x[((u32(i) * 4u) + 0u)], x[((u32(i) * 4u) + 1u)], x[((u32(i) * 4u) + 2u)], x[((u32(i) * 4u) + 3u)]);");
      expect(compiled.wgsl).toMatch(/let bg_vector_store_value_y_\d+: vec4<f32> = vec4<f32>\(f32\(\(value\.x \+ 1\.0\)\)/u);
      expect(compiled.wgsl).toMatch(/let bg_vector_store_base_y_\d+: u32 = \(u32\(i\) \* 4u\);/u);
      expect(compiled.wgsl).toMatch(/y\[\(bg_vector_store_base_y_\d+ \+ 3u\)\] = \(bg_vector_store_value_y_\d+\)\.w;/u);
      expect([...result.buffers.y as Float32Array]).toEqual([2, 4, 6, 8, 6, 8, 10, 12]);
      expect([...semanticResult.buffers.y as Float32Array]).toEqual([2, 4, 6, 8, 6, 8, 10, 12]);
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
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            x: createWgslFloat16Array([3, 5]),
            y: createWgslFloat16Array(2),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(backendIr(compiled).requiredFeatures).toContain("shader-f16");
      expect(compiled.wgsl).toContain("enable f16;");
      expect(compiled.wgsl).toContain("vec2<f16>");
      expect(Array.from(result.buffers.y as ArrayLike<number>)).toEqual([4, 7]);
      expect(Array.from(semanticResult.buffers.y as ArrayLike<number>)).toEqual([4, 7]);
    });

  it("preserves half2 vector arithmetic when writing through device pointer aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void reduce_pair(half2 *v) {
    v[0] = v[0] + v[1];
  }
  __global__ void half2PtrAssign(half2 *x) {
    reduce_pair(x);
  }`, { features: { "shader-f16": true }, f16Mode: "f32", workgroupSize: [1, 1, 1] });

      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bg_ptr_write_f16x2(v_buffer, (v_base + (0u * 2u)), (bg_ptr_read_f16x2(v_buffer, (v_base + (0u * 2u))) + bg_ptr_read_f16x2(v_buffer, (v_base + (1u * 2u)))))");
      expect(compiled.wgsl).not.toMatch(/vec2<f32>\(f32\(\(bg_ptr_read_f16x2/u);
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
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            a: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
            b: new Float32Array([10, 20, 30, 40, 50, 60, 70, 80]),
            c: new Float32Array(8),
          },
          scalars: { n: 8 },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("vec4<f32>(");
      expect(compiled.wgsl).toContain("a[");
      expect(compiled.wgsl).toContain("c[");
      expect([...result.buffers.c as Float32Array]).toEqual([11, 22, 33, 44, 55, 66, 77, 88]);
    });

  it("runs local vector reinterpret casts in the reference interpreter", () => {
      const compiled = compileCudaLiteKernel(`
  #define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
  __global__ void localPacked(float *out) {
    float4 value;
    value.x = 1.0f;
    value.y = 2.0f;
    value.z = 3.0f;
    value.w = 4.0f;
    FLOAT4(out[0]) = FLOAT4(value);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("let bg_vector_store_value_out_");
      expect(compiled.wgsl).not.toMatch(/&value\[0\]\.[xyzw]/u);
      expect([...result.buffers.out as Float32Array]).toEqual([1, 2, 3, 4]);
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

  it("flattens chained scalar-to-vector storage pointer aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void chainedVectorAlias(const float *inp, float *out, int row) {
    const float *x = inp + row * 8;
    const float4 *x_vec = reinterpret_cast<const float4 *>(x);
    float4 v = x_vec[1];
    out[0] = v.x + v.w;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            inp: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80]),
            out: new Float32Array(1),
          },
          scalars: { row: 1 },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("vec4<f32>(inp[");
      expect(compiled.wgsl).not.toContain("x_vec[");
      expect(compiled.wgsl).not.toContain("x[");
      expect([...result.buffers.out as Float32Array]).toEqual([130]);
    });

  it("keeps user params pointer distinct from compiler uniforms", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void paramsBuffer(const float *params, float *out, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
      out[idx] = params[idx] + 1.0f;
    }
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            params: new Float32Array([1, 2, 3, 4]),
            out: new Float32Array(4),
          },
          scalars: { n: 4 },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("var<storage, read> bg_params: array<f32>;");
      expect(compiled.wgsl).toContain("var<uniform> bg_uniforms: Params;");
      expect(compiled.wgsl).toContain("bg_params[u32(idx)]");
      expect(compiled.wgsl).toContain("u32(idx) < u32(bg_uniforms.n)");
      expect([...result.buffers.out as Float32Array]).toEqual([2, 3, 4, 5]);
    });

  it("bitcasts scalar typed views over integer shared backing storage", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sharedOverlay(float *out) {
    extern __shared__ int params[];
    float *scratch = (float*)params;
    int tid = threadIdx.x;
    if (tid < 2) { scratch[tid] = (float)(tid + 1); }
    __syncthreads();
    if (tid == 0) { out[0] = scratch[0] + scratch[1]; }
  }`, { workgroupSize: [2, 1, 1], dynamicSharedMemory: { params: 2 } });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("var<workgroup> bg_params: array<i32, 2>;");
      expect(compiled.wgsl).toContain("bg_params[");
      expect(compiled.wgsl).toContain("bitcast<i32>");
      expect(compiled.wgsl).toContain("bitcast<f32>");
      expect([...result.buffers.out as Float32Array]).toEqual([3]);
    });

  it("bitcasts vector typed views over integer shared backing storage", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sharedVectorOverlay(float *out) {
    extern __shared__ int params[];
    float4 *scratch = (float4*)params;
    if (threadIdx.x == 0) { scratch[0] = make_float4(1.0f, 2.0f, 3.0f, 4.0f); }
    __syncthreads();
    if (threadIdx.x == 0) {
      float4 value = scratch[0];
      out[0] = value.x + value.y + value.z + value.w;
    }
  }`, { workgroupSize: [1, 1, 1], dynamicSharedMemory: { params: 4 } });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("var<workgroup> bg_params: array<i32, 4>;");
      expect(compiled.wgsl).toContain("] = bitcast<i32>((vec4<f32>");
      expect(compiled.wgsl).toContain("vec4<f32>(bitcast<f32>");
      expect([...result.buffers.out as Float32Array]).toEqual([10]);
    });

  it("bitcasts nested local array scalar pointer views over integer carriers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localOverlay(float *out) {
    uint regs[1][2][2];
    regs[0][1][0] = __float_as_uint(3.5f);
    float *view = reinterpret_cast<float *>(&(regs[0][1][0]));
    view[1] = view[0] + 2.0f;
    out[0] = __uint_as_float(regs[0][1][1]);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("bitcast<f32>(regs[");
      expect(compiled.wgsl).toContain("bitcast<u32>");
      expect([...result.buffers.out as Float32Array]).toEqual([5.5]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([5.5]);
    });

  it("packs scalar half pointer views over 32-bit local carriers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localHalfOverlay(uint *out, float *sum) {
    uint regs[1][2];
    regs[0][0] = 0u;
    regs[0][1] = 0u;
    half *view = reinterpret_cast<half *>(&(regs[0][0]));
    view[0] = __float2half(1.0f);
    view[1] = __float2half(2.0f);
    view[2] = __float2half(3.0f);
    view[3] = __float2half(4.0f);
    sum[0] = __half2float(view[0]) + __half2float(view[1]) + __half2float(view[2]) + __half2float(view[3]);
    out[0] = regs[0][0];
    out[1] = regs[0][1];
  }`, { workgroupSize: [1, 1, 1], features: { "shader-f16": true } });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(2), sum: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(2), sum: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("pack2x16float");
      expect(compiled.wgsl).toContain("unpack2x16float");
      expect([...result.buffers.out as Uint32Array]).toEqual([0x40003c00, 0x44004200]);
      expect([...result.buffers.sum as Float32Array]).toEqual([10]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([...result.buffers.out as Uint32Array]);
      expect([...semanticResult.buffers.sum as Float32Array]).toEqual([10]);
    });

  it("bit-copies reinterpreted 128-bit local uint carriers into half storage", () => {
      const compiled = compileCudaLiteKernel(`
  #define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
  __global__ void localUintToHalfPack(half *out) {
    uint regs[4];
    regs[0] = 0x40003c00u;
    regs[1] = 0x44004200u;
    regs[2] = 0x46004500u;
    regs[3] = 0x48004700u;
    LDST128BITS(out[0]) = LDST128BITS(regs[0]);
  }`, { workgroupSize: [1, 1, 1], features: { "shader-f16": true } });
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: createWgslFloat16Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(JSON.stringify(compiled.kernelIr.operations)).toContain('"kind":"copy"');
      expect(JSON.stringify(compiled.kernelIr.operations)).toContain('"bytes":16');
      expect(compiled.wgsl).toContain("bitcast<vec2<f16>>");
      expect(Array.from(result.buffers.out as Iterable<number>)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    });

  it("bitcasts dynamic scalar lanes over local uint vectors", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localVectorScalarBits(float *out) {
    uint4 source = make_uint4(__float_as_uint(1.0f), __float_as_uint(2.0f), __float_as_uint(3.0f), __float_as_uint(4.0f));
    uint4 target;
    for (uint i = 0; i < 4; ++i) {
      float value = reinterpret_cast<float *>(&source)[i];
      reinterpret_cast<float *>(&target)[i] = value + 1.0f;
    }
    out[0] = reinterpret_cast<float *>(&target)[0];
    out[1] = reinterpret_cast<float *>(&target)[1];
    out[2] = reinterpret_cast<float *>(&target)[2];
    out[3] = reinterpret_cast<float *>(&target)[3];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("bitcast<f32>(source[i])");
      expect(compiled.wgsl).toContain("target[i] = bitcast<u32>");
      expect([...result.buffers.out as Float32Array]).toEqual([2, 3, 4, 5]);
    });

  it("folds same-type pointer indexing over local scalar addresses", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void scalar_address_identity(const float4* input, float4* output) {
    float4 value = input[0];
    output[0] = reinterpret_cast<float4*>(&value)[0];
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { input: new Float32Array([1, 2, 3, 4]), output: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("let bg_vector_store_value_output_");
      expect([...result.buffers.output as Float32Array]).toEqual([1, 2, 3, 4]);
    });

  it("packs half pointer views over shared byte storage into 16-bit lanes", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sharedByteHalfOverlay(float *out) {
    __shared__ uchar scratch[4];
    if (threadIdx.x == 0) {
      half *view = (half *)&scratch[0];
      view[0] = __float2half(1.0f);
      view[1] = __float2half(2.0f);
      out[0] = __half2float(view[0]);
      out[1] = __half2float(view[1]);
    }
  }`, { workgroupSize: [1, 1, 1], features: { "shader-f16": true }, f16Mode: "f32" });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("pack2x16float");
      expect(compiled.wgsl).toContain("unpack2x16float");
      expect([...result.buffers.out as Float32Array]).toEqual([1, 2]);
    });

  it("packs bf16 pointer views over shared byte storage into 16-bit lanes", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sharedByteBf16Overlay(float *out) {
    __shared__ uchar scratch[4];
    if (threadIdx.x == 0) {
      __nv_bfloat16 *view = (__nv_bfloat16 *)&scratch[0];
      view[0] = __float2bfloat16(1.0f);
      view[1] = __float2bfloat16(2.0f);
      out[0] = __bfloat162float(view[0]);
      out[1] = __bfloat162float(view[1]);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("<< 16u");
      expect(compiled.wgsl).toContain(">> 16u");
      expect([...result.buffers.out as Float32Array]).toEqual([1, 2]);
    });

  it("preserves unaligned typed pointer views over byte storage", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void write_unaligned_byte_float(float *value, float *out) {
    value[0] = 1.0f;
    value[1] = 2.0f;
    out[0] = value[0];
    out[1] = value[1];
  }

  __global__ void unalignedByteFloatOverlay(uchar *scratch, float *out) {
    if (threadIdx.x == 0) {
      write_unaligned_byte_float((float *)&scratch[1], out);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { scratch: new Uint32Array(3), out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.out as Float32Array]).toEqual([1, 2]);
      expect([...result.buffers.scratch as Uint32Array]).toEqual([0x80000000, 0x0000003f, 0x00000040]);
      expect(compiled.wgsl).toContain("& 255u");
      expect(compiled.wgsl).toContain(">> 24u");
    });

  it("packs vector half pointer views over shared byte storage into byte-offset lanes", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sharedByteHalf2Overlay(float *out) {
    __shared__ uchar scratch[4];
    if (threadIdx.x == 0) {
      half2 *view = (half2 *)&scratch[0];
      view[0] = make_half2(__float2half(1.0f), __float2half(2.0f));
      half2 value = view[0];
      out[0] = __low2float(value);
      out[1] = __high2float(value);
    }
  }`, { workgroupSize: [1, 1, 1], features: { "shader-f16": true }, f16Mode: "f32" });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("bg_semantic_packed_shared_u8_store");
      expect(compiled.wgsl).toContain("unpack2x16float");
      expect(compiled.wgsl).toContain("pack2x16float");
      expect([...result.buffers.out as Float32Array]).toEqual([1, 2]);
    });

  it("packs vector bf16 pointer views over shared byte storage into byte-offset lanes", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sharedByteBf162Overlay(float *out) {
    __shared__ uchar scratch[4];
    if (threadIdx.x == 0) {
      __nv_bfloat162 *view = (__nv_bfloat162 *)&scratch[0];
      view[0] = __halves2bfloat162(__float2bfloat16(1.0f), __float2bfloat16(2.0f));
      __nv_bfloat162 value = view[0];
      out[0] = __bfloat162float(value.x);
      out[1] = __bfloat162float(value.y);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("bg_semantic_packed_shared_u8_store");
      expect(compiled.wgsl).toContain("vec2<f32>(bitcast<f32>");
      expect(compiled.wgsl).toContain("<< 16u");
      expect([...result.buffers.out as Float32Array]).toEqual([1, 2]);
    });

  it("lowers generic pointer dereference lvalues and rebased kernel params", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void derefWrite(float *x) {
    x += 1;
    *x = *x + 3.0f;
    float *p = x + 1;
    *p = *p + 5.0f;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: new Float32Array([1, 2, 3]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("x__bg_ptr_offset = (x__bg_ptr_offset + 1);");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
      expect(compiled.wgsl).not.toContain("var p");
      expect([...result.buffers.x as Float32Array]).toEqual([1, 5, 8]);
    });

  it("emits pointer helpers for local storage-pointer aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void localPointerAliases(float *out, const float *left, const float *right, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      float *dst = out + idx;
      const float *a = left + idx;
      const float *b = right + idx;
      *dst = (float)((float)*a + (float)*b);
    }
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Float32Array(4),
            left: new Float32Array([1, 2, 3, 4]),
            right: new Float32Array([10, 20, 30, 40]),
          },
          scalars: { n: 4 },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("bg_ptr_read_f32");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
      expect([...result.buffers.out as Float32Array]).toEqual([11, 22, 33, 44]);
    });

  it("lowers bool storage-pointer aliases directly over u32-backed storage", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void boolPointerAliases(bool *flags) {
    int idx = threadIdx.x;
    bool *slot = flags + idx;
    *slot = (idx & 1) != 0;
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { flags: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("var<storage, read_write> flags: array<u32>;");
      expect(compiled.wgsl).toContain("flags[u32(idx)] = select(0u, 1u, (u32((idx & 1)) != 0u));");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_bool");
      expect([...result.buffers.flags as Uint32Array]).toEqual([0, 1, 0, 1]);
    });

  it("lowers vector member writes through dereferenced reinterpret views", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void vectorDeref(float *x, float *out) {
    float4 *p = reinterpret_cast<float4 *>(&x[0]);
    (*p).z = (*p).x + (*p).y;
    out[0] = (*p).z;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: new Float32Array([2, 4, 0, 8]), out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).not.toContain("bg_ptr_write_f32x4(0u,");
      expect(compiled.wgsl).not.toMatch(/x\[[^\n]+\] = vec[234]<f32>/u);
      expect(compiled.wgsl).toMatch(/x\[[^\n]+\+ 2u\)\] =/u);
      expect([...result.buffers.x as Float32Array]).toEqual([2, 4, 6, 8]);
      expect([...result.buffers.out as Float32Array]).toEqual([6]);
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
      expect(compiled.wgsl).toContain("fn load_float4(tile__bg_shared_ptr: ptr<workgroup, array<vec4<f32>, 2>>, tile__bg_shared_ptr_base: u32");
      expect(compiled.wgsl).toContain("load_float4(&tile, 0u, (1 - tid)");
      expect([...result.buffers.y as Float32Array]).toEqual([10, 20, 30, 41, 1, 2, 3, 5]);
    });

  it("lowers shared-array decay into a semantic pointer alias", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void shared_array_decay(float* out) {
    __shared__ float tile[4];
    int tid = threadIdx.x;
    float* base = tile + 1;
    base[tid] = out[tid];
    __syncthreads();
    out[tid] = base[1 - tid];
  }`, { workgroupSize: [2, 1, 1] });
      const input = { buffers: { out: new Float32Array([3, 7, 0, 0]) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("var<workgroup> tile: array<f32, 4>;");
      expect(compiled.wgsl).not.toContain("var base");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([7, 3, 0, 0]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([...result.buffers.out as Float32Array]);
    });

  it("lowers cp.async pointer-form copies to synchronous shared-memory copies", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void async_copy(const float *input, float *output) {
    __shared__ float tile[4];
    CP_ASYNC_CG(&tile[0], &input[0], 16);
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
    output[threadIdx.x] = tile[threadIdx.x] + 1.0f;
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { input: new Float32Array([1, 2, 3, 4]), output: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { input: new Float32Array([1, 2, 3, 4]), output: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(JSON.stringify(compiled.kernelIr.operations)).toContain('"kind":"copy"');
      expect(JSON.stringify(compiled.kernelIr.operations)).toContain('"kind":"copy-fence"');
      expect(compiled.wgsl).toContain("tile[0u] = input[0u];");
      expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
      expect(compiled.wgsl).toContain("cp.async fence omitted: CP_ASYNC_WAIT_GROUP");
      expect(compiled.wgsl).toContain("workgroupBarrier()");
      expect([...result.buffers.output as Float32Array]).toEqual([2, 3, 4, 5]);
      expect([...semanticResult.buffers.output as Float32Array]).toEqual([2, 3, 4, 5]);
    });

  it("lowers cp.async numeric shared byte addresses through semantic provenance", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void async_copy_bytes(const float *input, float *output) {
    extern __shared__ float smem[];
    float* tile = smem + 4;
    uint smem_base = __cvta_generic_to_shared(tile);
    uint smem_dst = smem_base + threadIdx.x * 8;
    CP_ASYNC_CG(smem_dst, &input[threadIdx.x * 2], 8);
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
    output[threadIdx.x] = float(tile[threadIdx.x * 2]) + float(tile[threadIdx.x * 2 + 1]);
  }`, { workgroupSize: [2, 1, 1], dynamicSharedMemory: { smem: 8 } });
      const input = {
        buffers: {
          input: new Float32Array([1, 2, 3, 4]),
          output: new Float32Array(2),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(JSON.stringify(compiled.kernelIr.operations)).toContain('"kind":"copy"');
      expect(compiled.wgsl).toContain(" / 4u)");
      expect(compiled.wgsl).not.toContain("f32((u32((4 * 4))");
      expect([...semanticResult.buffers.output as Float32Array]).toEqual([3, 7]);
    });

  it("rejects unproven cp.async shared byte alignment from semantic lowering", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void unaligned_async_copy(const float *input, float *output) {
    extern __shared__ float smem[];
    uint smem_base = __cvta_generic_to_shared(smem);
    CP_ASYNC_CG(smem_base + threadIdx.x, &input[0], 4);
    CP_ASYNC_WAIT_ALL();
    output[0] = smem[0];
  }`, { workgroupSize: [1, 1, 1], dynamicSharedMemory: { smem: 1 } });

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(false);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(false);
      expect(JSON.stringify(compiled.kernelIr.operations)).not.toContain('"kind":"copy"');
    });

  it("lowers cuRAND calls against storage-backed state arrays", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __global__ void initRNG(curandState_t *states, float *out, unsigned int seed) {
    unsigned int tid = threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
    out[tid] = curand_uniform(&states[tid]) + curand_normal(&states[tid]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { states: new Uint32Array(4), out: new Float32Array(4) }, scalars: { seed: 1234 } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("fn bg_curand_init_storage");
      expect(compiled.wgsl).toContain("fn bg_curand_uniform_storage");
      expect(compiled.wgsl).toContain("fn bg_curand_normal_storage");
      expect(compiled.wgsl).toContain("bg_curand_init_storage(bg_uniforms.seed, tid, 0u, &states[tid])");
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.kernelIr.operations.some((operation) => operation.kind === "call" && operation.callee === "curand_init")).toBe(true);
      expect([...result.buffers.out as Float32Array].every((value) => Number.isFinite(value))).toBe(true);
    });

  it("parses dynamic extern shared memory as a clear unsupported diagnostic", () => {
      const analysis = analyzeCudaLite(parseCudaLite(`
  __global__ void dynamicShared(float *x) {
    extern __shared__ float scratch[];
    if (threadIdx.x < 1) { scratch[threadIdx.x] = x[0]; }
  }`));

      expect(analysis.diagnostics.map((diagnostic) => diagnostic.code)).toContain("dynamic-shared-memory");
    });

  it("ignores dynamic extern shared memory in unreachable helpers for selected kernels", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void unused_dynamic_shared(float *x) {
    extern __shared__ float scratch[];
    if (threadIdx.x == 0) {
      scratch[0] = x[0];
    }
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
      expect(compiled.wgsl).not.toContain("unused_dynamic_shared");
    });

  it("ignores unreferenced device globals used only by unreachable helpers for selected kernels", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float unused_state[2];

  __device__ void unused_device_global_helper() {
    unused_state[0] = 7.0f;
    unused_state[1] = unused_state[0] + 1.0f;
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

      expect(backendIr(compiled).deviceGlobals.map((global) => global.name)).not.toContain("unused_state");
      expect(compiled.kernelIr.memory.map((symbol) => symbol.name)).not.toContain("unused_state");
      expect([...result.buffers.x as Float32Array]).toEqual([5]);
      expect(compiled.wgsl).not.toContain("unused_state");
      expect(compiled.wgsl).not.toContain("unused_device_global_helper");
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

  it("keeps dynamic shared scalar bases in scalar lanes for vector pointer helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void adjust(float3 *tile, int index, float bias) {
    float3 value = tile[index];
    tile[index] = make_float3(value.x + bias, value.y + 2.0f * bias, value.z + 3.0f * bias);
  }
  __global__ void dynamicSharedFloat3View(float *out) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    float3 *tile = reinterpret_cast<float3 *>(scratch);
    tile[tid] = make_float3((float)(tid + 1), (float)(tid + 10), (float)(tid + 100));
    __syncthreads();
    adjust(tile, tid, 0.5f + (float)tid);
    __syncthreads();
    float3 value = tile[tid];
    out[tid * 3 + 0] = value.x;
    out[tid * 3 + 1] = value.y;
    out[tid * 3 + 2] = value.z;
  }`, {
        workgroupSize: [2, 1, 1],
        dynamicSharedMemory: { scratch: 6 },
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(6) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("scratch[(index + 0u)] = value.x");
      expect(compiled.wgsl).toContain("bg_ptr_write_f32x3(1u, ((0u + u32(0)) + (u32(tid) * 3u))");
      expect(compiled.wgsl).toContain("bg_ptr_write_f32x3(tile_buffer, (tile_base + (u32(index) * 3u))");
      expect([...result.buffers.out as Float32Array]).toEqual([1.5, 11, 101.5, 3.5, 14, 105.5]);
    });

  it("keeps dynamic shared vector addresses in scalar lanes for pointer arrays", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float3 sum_dynamic_shared3(float3 *a, float3 *b, float3 *c) {
    return *a + *b + *c;
  }
  __global__ void dynamicSharedVectorPointerArray(float4 *out) {
    extern __shared__ float scratch[];
    float3 *values = reinterpret_cast<float3 *>(scratch);
    values[0] = make_float3(3.0f, 5.0f, 7.0f);
    values[1] = make_float3(11.0f, 13.0f, 17.0f);
    values[2] = make_float3(19.0f, 23.0f, 29.0f);
    float3 *ptrs[3];
    ptrs[0] = &values[0];
    ptrs[1] = &values[1];
    ptrs[2] = &values[2];
    float3 total = sum_dynamic_shared3(ptrs[0], ptrs[1], ptrs[2]);
    out[0] = make_float4(*ptrs[0], 1.0f);
    out[1] = make_float4(total, 0.0f);
  }`, {
        workgroupSize: [1, 1, 1],
        dynamicSharedMemory: { scratch: 9 },
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("sum_dynamic_shared3(&scratch, u32((0 * 3)), u32((1 * 3)), u32((2 * 3))");
      expect(compiled.wgsl).toContain("vec3<f32>(scratch[u32((0 * 3))], scratch[(u32((0 * 3)) + 1u)]");
      expect(compiled.wgsl).not.toContain("scratch[(u32((0 * 3)) * 3u)]");
      expect([...result.buffers.out as Float32Array]).toEqual([3, 5, 7, 1, 33, 41, 53, 0]);
    });

  it("keeps chained dynamic shared vector aliases in scalar lanes for pointer arrays", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float3 sum_dynamic_shared_chain3(float3 *a, float3 *b, float3 *c) {
    return *a + *b + *c;
  }
  __global__ void dynamicSharedVectorAliasChainPointerArray(float4 *out) {
    extern __shared__ float scratch[];
    float3 *values = reinterpret_cast<float3 *>(scratch);
    values[0] = make_float3(2.0f, 3.0f, 5.0f);
    values[1] = make_float3(7.0f, 11.0f, 13.0f);
    values[2] = make_float3(17.0f, 19.0f, 23.0f);
    values[3] = make_float3(29.0f, 31.0f, 37.0f);
    float3 *shifted = values + 1;
    float3 *ptrs[3];
    ptrs[0] = &shifted[0];
    ptrs[1] = &shifted[1];
    ptrs[2] = &values[3];
    float3 total = sum_dynamic_shared_chain3(ptrs[0], ptrs[1], ptrs[2]);
    out[0] = make_float4(*ptrs[1], 1.0f);
    out[1] = make_float4(total, 0.0f);
  }`, {
        workgroupSize: [1, 1, 1],
        dynamicSharedMemory: { scratch: 12 },
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("sum_dynamic_shared_chain3(&scratch, u32(((1 * 3) + (0 * 3))), u32(((1 * 3) + (1 * 3))), u32((3 * 3))");
      expect(compiled.wgsl).toContain("vec3<f32>(scratch[u32(((1 * 3) + (1 * 3)))], scratch[(u32(((1 * 3) + (1 * 3))) + 1u)]");
      expect(compiled.wgsl).not.toContain("scratch[(u32(((1 * 3) + (1 * 3))) * 3u)]");
      expect([...result.buffers.out as Float32Array]).toEqual([17, 19, 23, 1, 53, 61, 73, 0]);
    });

  it("lowers dynamic extern shared memory with trailing fixed dimensions", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void dynamicShared2d(float *x) {
    extern __shared__ float scratch[][2];
    int tid = threadIdx.x;
    if (tid < 2) {
      scratch[tid][0] = x[tid];
      scratch[tid][1] = x[tid] + 1.0f;
    }
    __syncthreads();
    if (tid < 1) { x[0] = scratch[0][1] + scratch[1][1]; }
  }`, {
        workgroupSize: [2, 1, 1],
        dynamicSharedMemory: { scratch: 2 },
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { x: new Float32Array([2, 3]) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("var<workgroup> scratch: array<f32, 4>;");
      expect([...result.buffers.x as Float32Array]).toEqual([7, 3]);
    });

  it("preserves conditional helper-call laziness in pointer initializers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint conditional_pointer_init_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __global__ void conditionalPointerInit(uint *storage, uint *out, int enabled) {
    uint *target = storage + (enabled != 0 ? conditional_pointer_init_helper_with_pointer_side_effect(storage, 7u) : 0u);
    target[0] = 9u;
    out[0] = storage[0];
    out[1] = storage[1];
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.wgsl).toContain("conditional_pointer_init_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0u, conditional_pointer_init_helper_with_pointer_side_effect");
    });

  it("guards conditional helper-call pointer initializers inside active-lane predication", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint active_conditional_pointer_init_helper_with_pointer_side_effect(uint *ptr, uint lane, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __global__ void activeConditionalPointerInit(uint *storage, uint *out, int limit, int enabled) {
    int tid = threadIdx.x;
    for (int step = 0; step < 2; step++) {
      if (tid >= limit) return;
      __syncthreads();
      uint *ptr = storage + tid * 2;
      uint *target = ptr + (enabled != 0 ? active_conditional_pointer_init_helper_with_pointer_side_effect(ptr, (uint)tid, (uint)(step + tid + 1)) : 0u);
      target[0] = 9u;
      __syncthreads();
    }
    out[tid] = storage[tid];
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toMatch(/if \(bg_barrier_loop_active_\d+\) \{\n\s+if \(\(bg_uniforms.enabled != 0\)\)/u);
      expect(compiled.wgsl).toContain("_ = bg_ptr_atomicAdd_u32(ptr_buffer, ptr_base, add);");
      expect(compiled.wgsl).toContain("active_conditional_pointer_init_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0u, active_conditional_pointer_init_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("_ = atomicAdd(&bg_storage[(u32((0 + (tid * 2))))");
    });

  it("preserves conditional helper-call laziness in vector pointer member lvalues", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint conditional_vector_member_lvalue_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __global__ void conditionalVectorMemberLvalue(uint4 *storage, uint *out, int enabled) {
    storage[enabled != 0 ? (int)conditional_vector_member_lvalue_helper_with_pointer_side_effect(reinterpret_cast<uint*>(storage), 7u) : 0].y = 9u;
    out[0] = storage[0].x;
    out[1] = storage[0].y;
    out[2] = storage[1].y;
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.wgsl).toContain("conditional_vector_member_lvalue_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0, i32(conditional_vector_member_lvalue_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0u, conditional_vector_member_lvalue_helper_with_pointer_side_effect");
    });

  it("guards conditional helper-call vector pointer member lvalues inside active-lane predication", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint active_conditional_vector_member_lvalue_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __global__ void activeConditionalVectorMemberLvalue(uint4 *storage, int limit, int enabled) {
    int tid = threadIdx.x;
    uint4 *target = storage;
    for (int step = 0; step < 2; step++) {
      if (tid >= limit) return;
      __syncthreads();
      target[enabled != 0 ? (int)active_conditional_vector_member_lvalue_helper_with_pointer_side_effect(reinterpret_cast<uint*>(storage) + tid, (uint)(step + tid + 1)) : 0].y = 9u;
      __syncthreads();
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toMatch(/if \(bg_barrier_loop_active_\d+\) \{\n\s+if \(\(bg_uniforms.enabled != 0\)\)/u);
      expect(compiled.wgsl).toContain("active_conditional_vector_member_lvalue_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0, i32(active_conditional_vector_member_lvalue_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0u, active_conditional_vector_member_lvalue_helper_with_pointer_side_effect");
    });

  it("preserves conditional helper-call laziness in local vector pointer member lvalues", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint conditional_local_vector_member_lvalue_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return 1u;
  }

  __global__ void conditionalLocalVectorMemberLvalue(uint4 *storage, uint *out, int enabled) {
    uint4 *target = storage;
    target[enabled != 0 ? (int)conditional_local_vector_member_lvalue_helper_with_pointer_side_effect(reinterpret_cast<uint*>(storage), 7u) : 0].y = 9u;
    out[0] = storage[0].x;
    out[1] = storage[0].y;
    out[2] = storage[1].y;
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.wgsl).toContain("conditional_local_vector_member_lvalue_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0, i32(conditional_local_vector_member_lvalue_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0u, conditional_local_vector_member_lvalue_helper_with_pointer_side_effect");
    });

  it("lowers dynamic extern shared memory declared inside device helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint reduce_one(uint value) {
    extern __shared__ uint sdata[];
    sdata[threadIdx.x] = value;
    return sdata[threadIdx.x];
  }
  __global__ void helperDynamicShared(uint *out) {
    if (threadIdx.x < 2) { out[threadIdx.x] = reduce_one((uint)(threadIdx.x + 3)); }
  }`, {
        workgroupSize: [2, 1, 1],
        dynamicSharedMemory: { sdata: 2 },
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("var<workgroup> sdata: array<u32, 2>;");
      expect([...result.buffers.out as Uint32Array]).toEqual([3, 4]);
    });

  it("lowers bf16 dynamic extern shared memory when launch metadata supplies its size", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void dynamicBf16(bf16 *out, const bf16 *in) {
    extern __shared__ bf16 params[];
    int tid = threadIdx.x;
    if (tid < 2) { params[tid] = in[tid]; }
    __syncthreads();
    if (tid < 1) { out[0] = params[0] + params[1]; }
  }`, {
        workgroupSize: [2, 1, 1],
        dynamicSharedMemory: { params: 2 },
      });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(2), in: new Float32Array([2, 3]) } },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("var<workgroup> bg_params: array<f32, 2>;");
      expect([...result.buffers.out as Float32Array]).toEqual([5, 0]);
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
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("sdataA[u32((2 + tid))] = x[u32((tid + 2))]");
      expect([...result.buffers.x as Float32Array]).toEqual([6, 2, 3, 4]);
    });

  it("evaluates integer constant expressions in shared array dimensions", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void padded(float *x) {
    __shared__ float tile[16][16 + 1];
    int tid = threadIdx.x;
    if (tid < 1) { tile[0][0] = x[0]; x[0] = tile[0][0]; }
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl).toContain("array<f32, 272>");
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
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: { x: new Float32Array([2, 4]) },
          constants: { coeffs: new Float32Array([10, 20]) },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var<storage, read> coeffs: array<f32>");
      expect(compiled.wgslProgram!.bindings).toContainEqual(expect.objectContaining({
        name: "coeffs",
        access: "read",
      }));
      expect([...semanticResult.buffers.x as Float32Array]).toEqual([20, 80]);
      expect([...result.buffers.x as Float32Array]).toEqual([20, 80]);
    });

  it("casts pointer-alias base and offset index math in WGSL", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __global__ void kernel(unsigned int *out, unsigned int pitch) {
    unsigned int *row = out + (blockIdx.x * pitch);
    int i = threadIdx.x;
    row[i] = 7;
  }`, { workgroupSize: [1, 1, 1] });

      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("(workgroup_id.x * bg_uniforms.pitch)");
      expect(compiled.wgsl).toContain("+ u32(i)");
    });

  it("lowers scalar CUDA vector constants as scalarized readonly storage inputs", () => {
      const compiled = compileCudaLiteKernel(`
  __constant__ float3 collider;
  __global__ void apply(float *out) {
    float3 bias = make_float3(1.0f, 2.0f, 3.0f);
    float3 value = collider + bias;
    out[threadIdx.x] = vec_at(value, threadIdx.x);
  }`, { workgroupSize: [3, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: { out: new Float32Array(3) },
          constants: { collider: new Float32Array([10, 20, 30]) },
        },
        { gridDim: [1, 1, 1], blockDim: [3, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: { out: new Float32Array(3) },
          constants: { collider: new Float32Array([10, 20, 30]) },
        },
        { gridDim: [1, 1, 1], blockDim: [3, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var<storage, read> collider: array<f32>;");
      expect(compiled.wgsl).toContain("vec3<f32>(collider[");
      expect(compiled.wgsl).not.toContain("bg_uniforms.collider");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([11, 22, 33]);
      expect([...result.buffers.out as Float32Array]).toEqual([11, 22, 33]);
    });

  it("lowers CUDA vector constant arrays as scalarized readonly storage inputs", () => {
      const compiled = compileCudaLiteKernel(`
  __constant__ float3 table[2];
  __global__ void apply(float *out) {
    int idx = threadIdx.x;
    float3 value = table[idx];
    out[idx] = value.x + value.y + value.z;
  }`, { workgroupSize: [2, 1, 1] });
      const input = {
        buffers: { out: new Float32Array(2) },
        constants: { table: new Float32Array([1, 2, 3, 4, 5, 6]) },
      };
      const result = runCompiledKernelReference(
        compiled,
        input,
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        input,
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var<storage, read> table: array<f32>;");
      expect(compiled.wgsl).toContain("vec3<f32>(table[((u32(idx) * 3u) + 0u)]");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([6, 15]);
      expect([...result.buffers.out as Float32Array]).toEqual([6, 15]);
    });

  it("decays CUDA constant arrays to readonly device pointer arguments", () => {
      const compiled = compileCudaLiteKernel(`
  __constant__ float coeffs[3];
  __device__ float pick(const float *ptr, int index) {
    return ptr[index];
  }
  __global__ void apply(float *out) {
    out[0] = pick(coeffs, 2);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: { out: new Float32Array(1) },
          constants: { coeffs: new Float32Array([3, 5, 7]) },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("var<storage, read> coeffs: array<f32>");
      expect(compiled.wgsl).toContain("fn pick(");
      expect(compiled.wgsl).toContain("return coeffs[u32(index)]");
      expect([...result.buffers.out as Float32Array]).toEqual([7]);
    });

  it("specializes vector constant roots in device pointer helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __constant__ float4 matrix[1];
  __device__ float4 add_row(const float4 *rows, float4 value) {
    return rows[0] + value;
  }
  __global__ void apply(float *out) {
    float3 value = make_float3(add_row(matrix, make_float4(1.0f, 2.0f, 3.0f, 4.0f)));
    out[0] = value.x;
    out[1] = value.y;
    out[2] = value.z;
  }`, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: { out: new Float32Array(3) },
        constants: { matrix: new Float32Array([10, 20, 30, 40]) },
      };
      const result = runCompiledKernelSemanticReference(
        compiled,
        input,
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const helper = compiled.kernelIr.functions.find((fn) => fn.name === "add_row");
      const matrix = compiled.kernelIr.memory.find((symbol) => symbol.name === "matrix");

      expect(helper?.params[0]?.addressSpace).toBe("constant");
      expect(helper?.params[0]?.pointerMemoryAlias).toBe(semanticMemoryIdFromSymbol(matrix!.id));
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("fn add_row(");
      expect(compiled.wgsl).toContain("vec4<f32>(matrix[");
      expect([...result.buffers.out as Float32Array]).toEqual([11, 22, 33]);
    });

  it("passes local scalar out-pointers through semantic device helpers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ int bounds(float value, float *low, float *high) {
  *low = value - 1.0f;
  *high = value + 1.0f;
  return 1;
}
__global__ void localOut(float *out) {
  float low, high;
  int hit = bounds(4.0f, &low, &high);
  out[0] = low;
  out[1] = high;
  out[2] = hit;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Float32Array(3) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("low: ptr<function, f32>");
    expect(compiled.wgsl).toContain("bounds(4.0, &low, &high");
    expect([...result.buffers.out as Float32Array]).toEqual([3, 5, 1]);
  });

  it("supports local read pointers into CUDA constant arrays", () => {
      const compiled = compileCudaLiteKernel(`
  __constant__ uint table[2][2];
  __global__ void apply(uint *out) {
    uint *row = &table[threadIdx.y][0];
    out[threadIdx.x] = row[threadIdx.x];
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: { out: new Uint32Array(2) },
          constants: { table: new Uint32Array([3, 5, 7, 11]) },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var<storage, read> table: array<u32>;");
      expect(compiled.wgsl).toContain("out[local_id.x] = table[((0u * 2u) + local_id.x)]");
      expect([...result.buffers.out as Uint32Array]).toEqual([3, 5]);

      const oneDimensional = compileCudaLiteKernel(`
  __constant__ uint coeffs[2];
  __global__ void one_dim(uint *out) {
    uint *row = &coeffs[0];
    out[threadIdx.x] = row[threadIdx.x];
  }`, { workgroupSize: [2, 1, 1] });
      const oneDimensionalResult = runCompiledKernelReference(
        oneDimensional,
        {
          buffers: { out: new Uint32Array(2) },
          constants: { coeffs: new Uint32Array([13, 17]) },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );
      expect(oneDimensional.wgsl).toContain("out[local_id.x] = coeffs[local_id.x]");
      expect([...oneDimensionalResult.buffers.out as Uint32Array]).toEqual([13, 17]);

      const write = analyzeCudaLite(parseCudaLite(`
  __constant__ uint table[2][2];
  __global__ void bad() {
    uint *row = &table[threadIdx.y][0];
    row[threadIdx.x] = 1u;
  }`));
      expect(write.diagnostics.map((diagnostic) => diagnostic.code)).toContain("const-pointer-write");
    });

  it("rejects CUDA constant array decay to writable device pointer arguments", () => {
      const analysis = analyzeCudaLite(parseCudaLite(`
  __constant__ float coeffs[3];
  __device__ void write(float *ptr) {
    ptr[0] = 1.0f;
  }
  __global__ void bad() {
    write(coeffs);
  }`));

      expect(analysis.diagnostics).toContainEqual(expect.objectContaining({
        code: "unsupported-device-pointer-param",
      }));
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

      expect(backendIr(compiled).constants.map((constant) => [constant.name, constant.dimensions])).toEqual([
        ["scale", []],
        ["coeffs", [3]],
      ]);
      expect(compiled.wgsl).toContain("const scale: f32 = 0.5;");
      expect(compiled.wgsl).toContain("const coeffs: array<i32, 3> = array<i32, 3>(2, 3, 5);");
      expect(compiled.wgslProgram!.bindings.map((binding) => binding.name)).not.toContain("coeffs");
      expect([...result.buffers.x as Float32Array]).toEqual([2, 4, 6]);
      expect([...result.buffers.out as Int32Array]).toEqual([2, 3, 5]);
    });

  it("keeps initialized CUDA constant arrays embedded instead of requiring storage bindings", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __constant__ short Q[] = {32, 33, 34, 35};
  __global__ void quant(float *out, float value) {
    int idx = threadIdx.x;
    out[idx] = roundf(value / (float)Q[idx]);
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.wgsl).toContain("const Q: array<i32, 4> = array<i32, 4>(32, 33, 34, 35);");
      expect(compiled.wgslProgram!.bindings.map((binding) => binding.name)).not.toContain("Q");
    });

  it("lowers half storage through f32 compatibility mode when shader-f16 is absent", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void halfCompat(half* x, half2* y, half a) {
    if (threadIdx.x < 1) {
      x[0] = __float2half(__half2float(x[0]) + __half2float(a));
      y[0] = __hadd2(y[0], __float2half2_rn(1.0f));
    }
  }`, {
        f16Mode: "f32",
        workgroupSize: [1, 1, 1],
      });
      expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
      expect(compiled.wgsl).not.toContain("enable f16;");
      expect(compiled.wgsl).not.toMatch(/\bf16\b/u);
      expect(compiled.wgsl).toContain("vec2<f32>");
      expect(compiled.wgslProgram!.bindings[0]).toMatchObject({ valueType: "f32" });
      expect(compiled.wgslProgram!.bindings[1]).toMatchObject({ valueType: "f32" });

      const uniforms = packCudaWebGpuUniformParams(compiled, {
        buffers: {
          x: new Float32Array([1.5]),
          y: new Float32Array([3, 5]),
        },
        scalars: { a: 2 },
      });
      expect(new DataView(uniforms.buffer).getFloat32(0, true)).toBe(2);
    });

  it("lowers CUDA fp8 storage conversions through explicit helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void fp8_convert(const uint* input, half* output, uint* encoded, int* as_int) {
    int idx = threadIdx.x;
    if (idx < 1) {
      half e4m3 = __nv_cvt_fp8_to_halfraw(input[0], __NV_E4M3);
      output[0] = e4m3;
      encoded[0] = __nv_cvt_float_to_fp8(__half2float(e4m3), __NV_SATFINITE, __NV_E4M3);
      as_int[0] = __half2int_rz(e4m3);
    }
  }`, {
        features: { "shader-f16": true },
        workgroupSize: [1, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            input: new Uint32Array([0x3c]),
            output: createWgslFloat16Array(1),
            encoded: new Uint32Array(1),
            as_int: new Int32Array(1),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            input: new Uint32Array([0x3c]),
            output: createWgslFloat16Array(1),
            encoded: new Uint32Array(1),
            as_int: new Int32Array(1),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bg_fp8_to_f32");
      expect(compiled.wgsl).toContain("fn bg_f32_to_fp8");
      expect(Array.from(result.buffers.output as Iterable<number>)).toEqual([1.5]);
      expect([...result.buffers.encoded as Uint32Array]).toEqual([0x3c]);
      expect([...result.buffers.as_int as Int32Array]).toEqual([1]);
      expect(Array.from(semanticResult.buffers.output as Iterable<number>)).toEqual([1.5]);
      expect([...semanticResult.buffers.encoded as Uint32Array]).toEqual([0x3c]);
      expect([...semanticResult.buffers.as_int as Int32Array]).toEqual([1]);
    });

  it("lowers CUDA bf16 values as rounded f32 browser storage", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void bf16_convert(const __nv_bfloat16* input, __nv_bfloat16* output, float* as_float, uint* bits) {
    int idx = threadIdx.x;
    if (idx < 1) {
      __nv_bfloat16 a = input[0];
      __nv_bfloat16 b = __float2bfloat16(0.1f);
      __nv_bfloat162 pair = __halves2bfloat162(a, b);
      __nv_bfloat162 rawPair = __uint_as_bfloat162(0x40003fc0u);
      output[0] = __hadd(pair.x, pair.y);
      output[1] = __ushort_as_bfloat16(0x3fc0u);
      output[2] = rawPair.x;
      output[3] = rawPair.y;
      as_float[0] = __bfloat162float(output[0]);
      bits[0] = __bfloat16_as_ushort(output[1]);
      bits[1] = __nv_bfloat16_as_ushort(__float2bfloat16(2.0f));
      bits[2] = __bfloat162_as_uint(rawPair);
      bits[3] = __nv_bfloat162_as_uint(__uint_as_nv_bfloat162(0x40403f80u));
    }
  }`, {
        workgroupSize: [1, 1, 1],
      });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            input: new Float32Array([1.5]),
            output: new Float32Array(4),
            as_float: new Float32Array(1),
            bits: new Uint32Array(4),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            input: new Float32Array([1.5]),
            output: new Float32Array(4),
            as_float: new Float32Array(1),
            bits: new Uint32Array(4),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
      expect(compiled.wgsl).toContain("vec2<f32>");
      expect(compiled.wgsl).toContain("((bitcast<u32>(f32(output[1u])) >> 16u) & 0xffffu)");
      expect(compiled.wgsl).toContain("vec2<f32>(bitcast<f32>((1073758144u & 0x0000ffffu) << 16u), bitcast<f32>(1073758144u & 0xffff0000u))");
      expect([...result.buffers.output as Float32Array][0]).toBeCloseTo(1.6015625);
      expect([...result.buffers.output as Float32Array][1]).toBeCloseTo(1.5);
      expect([...result.buffers.output as Float32Array][2]).toBeCloseTo(1.5);
      expect([...result.buffers.output as Float32Array][3]).toBeCloseTo(2);
      expect([...result.buffers.as_float as Float32Array][0]).toBeCloseTo(1.6015625);
      expect([...result.buffers.bits as Uint32Array]).toEqual([0x3fc0, 0x4000, 0x40003fc0, 0x40403f80]);
      expect([...semanticResult.buffers.output as Float32Array][0]).toBeCloseTo(1.6015625);
      expect([...semanticResult.buffers.output as Float32Array][1]).toBeCloseTo(1.5);
      expect([...semanticResult.buffers.output as Float32Array][2]).toBeCloseTo(1.5);
      expect([...semanticResult.buffers.output as Float32Array][3]).toBeCloseTo(2);
      expect([...semanticResult.buffers.as_float as Float32Array][0]).toBeCloseTo(1.6015625);
      expect([...semanticResult.buffers.bits as Uint32Array]).toEqual([0x3fc0, 0x4000, 0x40003fc0, 0x40403f80]);
    });

  it("supports CUDA cache-hint pointer helpers for bf16 storage", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void bf16_cache_hint(const __nv_bfloat16* input, __nv_bfloat16* output) {
    int idx = threadIdx.x;
    __nv_bfloat16 value = __ldcs(input + idx);
    __stcs(output + idx, __float2bfloat16(__bfloat162float(value) + 1.0f));
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            input: new Float32Array([1.5, 2.5]),
            output: new Float32Array(2),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(compiled.wgsl).toContain("var value: f32 = input[");
      expect(compiled.wgsl).toContain("output[");
      expect([...result.buffers.output as Float32Array]).toEqual([2.5, 3.5]);
    });

  it("passes nullable conditional storage pointers into device helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void maybe_store(float* target, float* fallback, float value) {
    if (target != NULL) {
      target[0] = value;
    } else {
      fallback[0] = value + 1.0f;
    }
  }

  __global__ void conditional_pointer(float* target, float* fallback, int enabled) {
    maybe_store(enabled ? target : NULL, fallback, 3.0f);
  }`, { workgroupSize: [1, 1, 1] });
      const enabled = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            target: new Float32Array(1),
            fallback: new Float32Array(1),
          },
          scalars: { enabled: 1 },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const disabled = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            target: new Float32Array(1),
            fallback: new Float32Array(1),
          },
          scalars: { enabled: 0 },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...enabled.buffers.target as Float32Array]).toEqual([3]);
      expect([...enabled.buffers.fallback as Float32Array]).toEqual([0]);
      expect([...disabled.buffers.target as Float32Array]).toEqual([0]);
      expect([...disabled.buffers.fallback as Float32Array]).toEqual([4]);
      expect(compiled.wgsl).toContain("4294967295u");
      expect(compiled.wgsl).toContain("select(4294967295u, 0u, (bg_uniforms.enabled != 0))");
    });

  it("packs uchar shared-memory pointer helpers into u32 carriers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void bump(uchar* ptr, uint offset) {
    ptr[offset]++;
  }
  __global__ void uchar_shared(uint* out) {
    __shared__ uchar bytes[16];
    ((uint*)bytes)[threadIdx.x] = 0u;
    bump(bytes, threadIdx.x);
    bump(bytes, threadIdx.x + 4u);
    if (threadIdx.x == 0) {
      out[0] = bytes[0] + bytes[4];
    }
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(backendIr(compiled).sharedDeclarations[0]?.valueType).toBe("uchar");
      expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(true);
      expect(compiled.wgsl).toContain("var<workgroup> bytes: array<atomic<u32>, 4>;");
      expect(compiled.wgsl).toContain("fn bg_semantic_packed_shared_u8_add(");
      expect(compiled.wgsl).toContain("atomicCompareExchangeWeak");
      expect(compiled.wgsl).not.toContain("array<u32, 16>");
      expect([...result.buffers.out as Uint32Array]).toEqual([2]);
    });

  it("passes shared array offsets to device pointer helper parameters", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void copy_one(float* out, float* src) {
    out[0] = src[0];
  }
  __global__ void shared_pointer_decay(float* out) {
    __shared__ float tile[4];
    if (threadIdx.x == 0) {
      tile[1] = 3.5f;
      copy_one(out, tile + 1);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("src__bg_shared_ptr: ptr<workgroup, array<f32, 4>>");
      expect(compiled.wgsl).toContain("copy_one(0u, 0u, &tile, 1u");
      expect([...result.buffers.out as Float32Array]).toEqual([3.5]);
    });

  it("supports explicit pointer casts over shared arrays", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void shared_pointer_cast(uint* out) {
    __shared__ uint tile[4];
    if (threadIdx.x == 0) {
      ((uint*)tile)[1] = 9u;
      out[0] = ((uint*)tile)[1];
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(true);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("tile[1u] = 9u;");
      expect(compiled.wgsl).toContain("out[0u] = tile[1u];");
      expect([...result.buffers.out as Uint32Array]).toEqual([9]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([9]);
    });

  it("preserves packed uchar shared aliases through semantic reference and WGSL", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void bump(uchar* bytes, uint index) {
    bytes[index]++;
  }
  __global__ void packed_shared_uchar(uint* out) {
    __shared__ uchar bytes[4];
    if (threadIdx.x == 0) {
      ((uint*)bytes)[0] = 0x04030201u;
      bump(bytes, 1u);
      out[0] = ((uint*)bytes)[0];
      out[1] = bytes[0] + bytes[1] + bytes[2] + bytes[3];
      out[2] = ~63u;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(true);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("var<workgroup> bytes: array<atomic<u32>, 1>;");
      expect(compiled.wgsl).toContain("atomicLoad(&bytes[");
      expect(compiled.wgsl).toContain("atomicStore(&bytes[");
      expect(compiled.wgsl).toContain("bg_semantic_packed_shared_u8_add");
      expect(compiled.wgsl).toContain("~(63u)");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([0x04030301, 11, 0xffffffc0]);
    });

  it("reads scalar device globals as values in device helper truthiness", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __device__ unsigned int flag = 0;
  __device__ unsigned int errors = 0;

  __device__ void check_value(int *data, int expected) {
    if ((data[threadIdx.x] != expected) && (!flag)) {
      errors++;
      flag = 1;
    }
  }

  __global__ void scalar_global_truthiness(int *data) {
    check_value(data, 7);
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl).toContain("!((flag[0u] != 0u))");
      expect(compiled.wgsl).not.toContain("!(1u != 4294967295u)");
    });

  it("marks initial and update roots for for-loop pointer rebinding", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void loop_update_pointer_atomic(uint* left, uint* right, uint* out) {
    uint* ptr = left;
    if (threadIdx.x == 0) {
      for (int i = 0; i < 2; i++, ptr = right + 1) {
        out[i] = atomicAdd(ptr, 1u);
      }
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            left: new Uint32Array([4]),
            right: new Uint32Array([8, 10]),
            out: new Uint32Array(2),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect([...result.buffers.left as Uint32Array]).toEqual([5]);
      expect([...result.buffers.right as Uint32Array]).toEqual([8, 11]);
      expect([...result.buffers.out as Uint32Array]).toEqual([4, 10]);
      expect(compiled.wgsl).toContain("var<storage, read_write> left: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("var<storage, read_write> right: array<atomic<u32>>;");
      expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
      expect(compiled.wgsl).toContain("bg_ptr_atomicAdd_u32(ptr_buffer, u32(i32(ptr_base)), 1u)");
      expect(compiled.wgsl).toContain("ptr_buffer = 1u;");
      expect(compiled.wgsl).toContain("ptr_base = 1u;");
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
