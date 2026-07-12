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

describe("CUDA-lite compiler: Control flow and synchronization", () => {
  it("runs simple for-loops through semantic reference and WGSL paths", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void window3(const float* x, float* y, int stride) {
    int base = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int j = 0; j < 3; j++) {
      int idx = base + j * stride;
      acc = acc + x[idx];
    }
    y[base] = acc;
  }
  `, { workgroupSize: [4, 1, 1] });
      const input = {
        buffers: {
          x: new Float32Array([1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400]),
          y: new Float32Array(4),
        },
        scalars: { stride: 4 },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const referenceResult = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("for (var j: i32 = 0; (u32(j) < 3u); j += 1)");
      expect([...semanticResult.buffers.y as Float32Array]).toEqual([111, 222, 333, 444]);
      expect([...referenceResult.buffers.y as Float32Array]).toEqual([111, 222, 333, 444]);
    });

  it("runs loop break and continue through semantic reference and WGSL paths", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void loopControl(int* out) {
    int acc = 0;
    for (int i = 0; i < 8; i++) {
      if (i == 2) continue;
      if (i == 5) break;
      acc += i;
    }
    out[0] = acc;
  }
  `, { workgroupSize: [1, 1, 1] });
      const input = { buffers: { out: new Int32Array(1) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const referenceResult = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("continue;");
      expect(compiled.wgsl).toContain("break;");
      expect([...semanticResult.buffers.out as Int32Array]).toEqual([8]);
      expect([...referenceResult.buffers.out as Int32Array]).toEqual([8]);
    });

  it("runs do-while continue through semantic reference and WGSL continuing checks", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void doWhileContinue(int* out) {
    int i = 0;
    int acc = 0;
    do {
      i++;
      if (i == 2) continue;
      acc += i;
    } while (i < 4);
    out[0] = acc;
  }
  `, { workgroupSize: [1, 1, 1] });
      const input = { buffers: { out: new Int32Array(1) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const referenceResult = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("continuing {");
      expect(compiled.wgsl).toContain("break if !(");
      expect(compiled.wgsl).toContain("break if !((u32(i) < 4u));");
      expect([...semanticResult.buffers.out as Int32Array]).toEqual([8]);
      expect([...referenceResult.buffers.out as Int32Array]).toEqual([8]);
    });

  it("materializes lazy do-while conditions in the IR continuing region", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void lazyDoWhile(uint* state, uint* out, int enabled) {
    int i = 0;
    do {
      i++;
      if (i < 2) continue;
      out[0] = (uint)i;
    } while (i < 3 && (enabled ? atomicAdd(state, 7u) + 1u : 0u) != 0u);
  }
  `, { workgroupSize: [1, 1, 1] });
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const enabledInput = { buffers: { state: new Uint32Array([2]), out: new Uint32Array(1) }, scalars: { enabled: 1 } };
      const disabledInput = { buffers: { state: new Uint32Array([2]), out: new Uint32Array(1) }, scalars: { enabled: 0 } };
      const enabled = runCompiledKernelSemanticReference(compiled, enabledInput, launch);
      const disabled = runCompiledKernelSemanticReference(compiled, disabledInput, launch);
      const loop = compiled.kernelIr.operations.find((operation) => operation.kind === "loop");

      expect(loop?.kind).toBe("loop");
      expect(loop?.kind === "loop" ? loop.condition : undefined).toBeUndefined();
      expect(loop?.kind === "loop" ? loop.continuing?.length : undefined).toBeGreaterThan(0);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("continuing {");
      expect([...enabled.buffers.state as Uint32Array]).toEqual([16]);
      expect([...enabled.buffers.out as Uint32Array]).toEqual([3]);
      expect([...disabled.buffers.state as Uint32Array]).toEqual([2]);
      expect([...disabled.buffers.out as Uint32Array]).toEqual([0]);
    });

  it("runs early kernel returns through semantic reference and WGSL", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void earlyReturn(uint* out, int n) {
    int idx = threadIdx.x;
    if (idx >= n) return;
    out[idx] = (uint)(idx + 1);
  }
  `, { workgroupSize: [4, 1, 1] });
      const input = {
        buffers: { out: new Uint32Array(4) },
        scalars: { n: 2 },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("return;");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1, 2, 0, 0]);
      expect([...result.buffers.out as Uint32Array]).toEqual([1, 2, 0, 0]);
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
    if (threadIdx.x < 1) { atomicAnd(&x[0], 1); }
  }`);
      const atomicAnalysis = analyzeCudaLite(unsupportedF32Atomic);
      expect(atomicAnalysis.diagnostics.map((diagnostic) => diagnostic.code)).toContain("unsupported-atomic-f32");

      const divergentBarrier = parseCudaLite(`
  __global__ void bad(float* x) {
    if (threadIdx.x < 1) { __syncthreads(); }
  }`);
      const barrierAnalysis = analyzeCudaLite(divergentBarrier);
      expect(barrierAnalysis.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-barrier");

      const divergentReturnBeforeBarrier = parseCudaLite(`
  __global__ void bad(float* x, int n) {
    int idx = threadIdx.x;
    if (idx >= n) return;
    __syncthreads();
    x[idx] = 1.0;
  }`);
      const divergentReturnAnalysis = analyzeCudaLite(divergentReturnBeforeBarrier);
      expect(divergentReturnAnalysis.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(divergentReturnAnalysis.diagnostics.find((diagnostic) => diagnostic.code === "divergent-return-before-barrier")?.severity).toBe("warning");
      const divergentBreakBeforeBarrier = parseCudaLite(`
  __global__ void bad(uint* x, int n) {
    int idx = threadIdx.x;
    for (int i = 0; i < 2; ++i) {
      if (idx >= n) break;
      x[idx] = (uint)i;
    }
    __syncthreads();
    if (idx < n) { x[idx] = x[idx] + 1u; }
  }`);
      const divergentBreakAnalysis = analyzeCudaLite(divergentBreakBeforeBarrier);
      expect(divergentBreakAnalysis.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-break-before-barrier");
      expect(divergentBreakAnalysis.diagnostics.find((diagnostic) => diagnostic.code === "divergent-break-before-barrier")?.severity).toBe("warning");
      expect(() => compileCudaLiteKernel(`
  __global__ void warnOnly(float* x, int n) {
    int idx = threadIdx.x;
    if (idx >= n) return;
    __syncthreads();
    x[idx] = 1.0;
  }`, { workgroupSize: [2, 1, 1] })).not.toThrow();
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

      expect(compiled.wgsl).toContain("while ((u32(i) < 5u))");
      expect(compiled.wgsl).toContain("continue;");
      expect([...result.buffers.out as Int32Array]).toEqual([13]);
    });

  it("lowers CUDA do-while loops and continues through the continuing condition", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void doWhileLoop(int *out) {
    int i = 0;
    int acc = 0;
    do {
      acc += i;
      i++;
    } while (i < 4);
    out[0] = acc;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const withContinue = compileCudaLiteKernel(`
  __global__ void doWhileContinue(int *out) {
    int i = 0;
    int acc = 0;
    do {
      i++;
      if (i == 2) continue;
      acc += i;
    } while (i < 4);
    out[0] = acc;
  }`, { workgroupSize: [1, 1, 1] });
      const continueResult = runCompiledKernelReference(
        withContinue,
        { buffers: { out: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("loop {");
      expect(compiled.wgsl).toContain("break if !((u32(i) < 4u));");
      expect([...result.buffers.out as Int32Array]).toEqual([6]);
      expect(withContinue.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-do-while-continue");
      expect(withContinue.wgsl).toContain("continuing {");
      expect(withContinue.wgsl).toContain("break if !((u32(i) < 4u));");
      expect([...continueResult.buffers.out as Int32Array]).toEqual([8]);
    });

  it("lowers CUDA loop breaks", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void breakLoop(int *out) {
    int acc = 0;
    for (int i = 0; i < 8; i++) {
      if (i == 4) break;
      acc += i;
    }
    out[0] = acc;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("break;");
      expect([...result.buffers.out as Int32Array]).toEqual([6]);
    });

  it("allows loop-local variable names to be reused in independent loop scopes", () => {
      const analysis = analyzeCudaLite(parseCudaLite(`
  __global__ void scopedLoops(float *x) {
    for (int s = 1; s > 0; s >>= 1) { x[0] += 1.0f; }
    for (int s = 1; s > 0; s >>= 1) { x[0] += 1.0f; }
  }`));

      expect(analysis.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("duplicate-symbol");
    });

  it("guards helper calls in predicated compound assignments after loop returns", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint helper_with_pointer_side_effect(uint *ptr, uint lane, uint add) {
    atomicAdd(ptr + lane, add);
    return ptr[lane];
  }

  __global__ void pointerHandleLoopReturnSideEffect(uint4 *left, uint4 *right, uint *out, int limit, int pickRight) {
    int tid = threadIdx.x;
    uint *ptr = NULL;
    uint total = 0u;
    for (int step = 0; step < 2; step++) {
      if (tid >= limit) return;
      __syncthreads();
      ptr = reinterpret_cast<uint*>(left);
      if (((step + pickRight) & 1) != 0) {
        ptr = reinterpret_cast<uint*>(right);
      }
      total += helper_with_pointer_side_effect(ptr, (uint)tid, (uint)(step + tid + 1));
      __syncthreads();
    }
    out[tid] = total;
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toMatch(/if \(bg_barrier_loop_active_\d+\) \{\n\s+total = \(total \+ helper_with_pointer_side_effect/u);
      expect(compiled.wgsl).not.toContain("select(total, (total + helper_with_pointer_side_effect");
    });

  it("preserves conditional expression laziness for helper-call assignment RHS", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint conditional_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return ptr[0];
  }

  __global__ void conditionalHelperAssignment(uint *storage, uint *out, int enabled) {
    uint total = 0u;
    total += enabled != 0 ? conditional_helper_with_pointer_side_effect(storage, 7u) : 0u;
    out[0] = total;
  }`, { workgroupSize: [1, 1, 1] });

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("if ((u32(bg_uniforms.enabled) != 0u))");
      expect(compiled.wgsl).toContain("total += conditional_helper_with_pointer_side_effect");
      expect(compiled.wgsl).toContain("total += 0u;");
      expect(compiled.wgsl).not.toContain("select(0u, conditional_helper_with_pointer_side_effect");

      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const disabled = runCompiledKernelReference(compiled, {
        buffers: { storage: new Uint32Array([2]), out: new Uint32Array(1) },
        scalars: { enabled: 0 },
      }, launch);
      expect([...disabled.buffers.storage as Uint32Array]).toEqual([2]);
      expect([...disabled.buffers.out as Uint32Array]).toEqual([0]);

      const enabled = runCompiledKernelReference(compiled, {
        buffers: { storage: new Uint32Array([2]), out: new Uint32Array(1) },
        scalars: { enabled: 1 },
      }, launch);
      expect([...enabled.buffers.storage as Uint32Array]).toEqual([9]);
      expect([...enabled.buffers.out as Uint32Array]).toEqual([9]);
    });

  it("preserves conditional helper-call laziness inside active-lane predication", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint active_conditional_helper_with_pointer_side_effect(uint *ptr, uint lane, uint add) {
    atomicAdd(ptr + lane, add);
    return ptr[lane];
  }

  __global__ void activeConditionalHelperAssignment(uint *storage, uint *out, int limit, int enabled) {
    int tid = threadIdx.x;
    uint total = 0u;
    for (int step = 0; step < 2; step++) {
      if (tid >= limit) return;
      __syncthreads();
      total += enabled != 0 ? active_conditional_helper_with_pointer_side_effect(storage, (uint)tid, (uint)(step + tid + 1)) : 0u;
      __syncthreads();
    }
    out[tid] = total;
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toMatch(/if \(bg_barrier_loop_active_\d+\) \{\n\s+if \(\(bg_uniforms.enabled != 0\)\)/u);
      expect(compiled.wgsl).toContain("total += active_conditional_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0u, active_conditional_helper_with_pointer_side_effect");
    });

  it("preserves conditional helper-call laziness in local var initializers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint conditional_var_init_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return ptr[0];
  }

  __global__ void conditionalHelperVarInit(uint *storage, uint *out, int enabled) {
    uint total = enabled != 0 ? conditional_var_init_helper_with_pointer_side_effect(storage, 7u) : 0u;
    out[0] = total;
  }`, { workgroupSize: [4, 1, 1] });

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var total: u32;");
      expect(compiled.wgsl).toContain("if ((u32(bg_uniforms.enabled) != 0u))");
      expect(compiled.wgsl).toContain("= conditional_var_init_helper_with_pointer_side_effect");
      expect(compiled.wgsl).toMatch(/total = bg__bg_condition_value_\d+_\d+;/u);
      expect(compiled.wgsl).not.toContain("select(0u, conditional_var_init_helper_with_pointer_side_effect");
    });

  it("preserves nested conditional helper-call laziness in local var initializers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint nested_conditional_var_init_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return ptr[0];
  }

  __global__ void nestedConditionalHelperVarInit(uint *storage, uint *out, int enabled) {
    uint total = 3u + (enabled != 0 ? nested_conditional_var_init_helper_with_pointer_side_effect(storage, 7u) : 0u);
    out[0] = total;
  }`, { workgroupSize: [4, 1, 1] });

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var total: u32;");
      expect(compiled.wgsl).toContain("if ((u32(bg_uniforms.enabled) != 0u))");
      expect(compiled.wgsl).toContain("= nested_conditional_var_init_helper_with_pointer_side_effect");
      expect(compiled.wgsl).toMatch(/total = \(3u \+ bg__bg_condition_value_\d+_\d+\);/u);
      expect(compiled.wgsl).not.toContain("select(0u, nested_conditional_var_init_helper_with_pointer_side_effect");
    });

  it("preserves conditional helper-call var init laziness inside active-lane predication", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint active_conditional_var_init_helper_with_pointer_side_effect(uint *ptr, uint lane, uint add) {
    atomicAdd(ptr + lane, add);
    return ptr[lane];
  }

  __global__ void activeConditionalHelperVarInit(uint *storage, uint *out, int limit, int enabled) {
    int tid = threadIdx.x;
    uint total = 0u;
    for (int step = 0; step < 2; step++) {
      if (tid >= limit) return;
      __syncthreads();
      uint value = enabled != 0 ? active_conditional_var_init_helper_with_pointer_side_effect(storage, (uint)tid, (uint)(step + tid + 1)) : 0u;
      total += value;
      __syncthreads();
    }
    out[tid] = total;
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toMatch(/if \(bg_barrier_loop_active_\d+\) \{\n\s+if \(\(bg_uniforms.enabled != 0\)\)/u);
      expect(compiled.wgsl).toContain("value = active_conditional_var_init_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0u, active_conditional_var_init_helper_with_pointer_side_effect");
    });

  it("preserves nested conditional helper-call var init laziness inside active-lane predication", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint active_nested_conditional_var_init_helper_with_pointer_side_effect(uint *ptr, uint lane, uint add) {
    atomicAdd(ptr + lane, add);
    return ptr[lane];
  }

  __global__ void activeNestedConditionalHelperVarInit(uint *storage, uint *out, int limit, int enabled) {
    int tid = threadIdx.x;
    uint total = 0u;
    for (int step = 0; step < 2; step++) {
      if (tid >= limit) return;
      __syncthreads();
      uint value = (uint)(step + 1) + (enabled != 0 ? active_nested_conditional_var_init_helper_with_pointer_side_effect(storage, (uint)tid, (uint)(step + tid + 1)) : 0u);
      total += value;
      __syncthreads();
    }
    out[tid] = total;
  }`, { workgroupSize: [4, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
      expect(compiled.wgsl).toMatch(/if \(bg_barrier_loop_active_\d+\) \{\n\s+if \(\(bg_uniforms.enabled != 0\)\)/u);
      expect(compiled.wgsl).toContain("value = (bitcast<u32>((step + 1)) + active_nested_conditional_var_init_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0u, active_nested_conditional_var_init_helper_with_pointer_side_effect");
    });

  it("preserves nested conditional helper-call laziness in device return expressions", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint nested_return_conditional_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return ptr[0];
  }

  __device__ uint nested_return_conditional_wrapper(uint *ptr, int enabled) {
    return 3u + (enabled != 0 ? nested_return_conditional_helper_with_pointer_side_effect(ptr, 7u) : 0u);
  }

  __global__ void nestedConditionalHelperReturn(uint *storage, uint *out, int enabled) {
    out[0] = nested_return_conditional_wrapper(storage, enabled);
  }`, { workgroupSize: [1, 1, 1] });

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toMatch(/var bg__bg_condition_value_\d+_\d+: u32;/u);
      expect(compiled.wgsl).toContain("if ((u32(enabled) != 0u))");
      expect(compiled.wgsl).toMatch(/return \(3u \+ bg__bg_condition_value_\d+_\d+\);/u);
      expect(compiled.wgsl).not.toContain("select(0u, nested_return_conditional_helper_with_pointer_side_effect");

      const result = runCompiledKernelSemanticReference(compiled, {
        buffers: { storage: new Uint32Array([2]), out: new Uint32Array(1) },
        scalars: { enabled: 1 },
      }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] });
      expect([...result.buffers.storage as Uint32Array]).toEqual([9]);
      expect([...result.buffers.out as Uint32Array]).toEqual([12]);
    });

  it("preserves nested conditional helper-call laziness in call arguments", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint nested_arg_conditional_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return ptr[0];
  }

  __device__ void nested_arg_sink(uint value, uint *out) {
    out[0] = value;
  }

  __global__ void nestedConditionalHelperArg(uint *storage, uint *out, int enabled) {
    nested_arg_sink(3u + (enabled != 0 ? nested_arg_conditional_helper_with_pointer_side_effect(storage, 7u) : 0u), out);
  }`, { workgroupSize: [1, 1, 1] });

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("if ((u32(bg_uniforms.enabled) != 0u))");
      expect(compiled.wgsl).toMatch(/nested_arg_sink\(\(3u \+ bg__bg_condition_value_\d+_\d+\)/u);
      expect(compiled.wgsl).not.toContain("select(0u, nested_arg_conditional_helper_with_pointer_side_effect");

      const result = runCompiledKernelSemanticReference(compiled, {
        buffers: { storage: new Uint32Array([2]), out: new Uint32Array(1) },
        scalars: { enabled: 1 },
      }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] });
      expect([...result.buffers.storage as Uint32Array]).toEqual([9]);
      expect([...result.buffers.out as Uint32Array]).toEqual([12]);
    });

  it("preserves nested conditional helper-call laziness in branch conditions", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint nested_condition_conditional_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return ptr[0];
  }

  __global__ void nestedConditionalHelperCondition(uint *storage, uint *out, int enabled) {
    if ((3u + (enabled != 0 ? nested_condition_conditional_helper_with_pointer_side_effect(storage, 7u) : 0u)) != 0u) {
      out[0] = 1u;
    } else {
      out[0] = 2u;
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("if ((u32(bg_uniforms.enabled) != 0u))");
      expect(compiled.wgsl).toContain("nested_condition_conditional_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0u, nested_condition_conditional_helper_with_pointer_side_effect");
    });

  it("preserves nested conditional helper-call laziness in loop conditions", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint nested_loop_condition_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return ptr[0];
  }

  __global__ void nestedConditionalHelperLoopCondition(uint *storage, uint *out, int enabled) {
    for (int i = 0; i < 1 && ((3u + (enabled != 0 ? nested_loop_condition_helper_with_pointer_side_effect(storage, 7u) : 0u)) != 0u); ++i) {
      out[0] = 1u;
    }
  }`, { workgroupSize: [1, 1, 1] });

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toMatch(/var bg__bg_short_circuit_\d+_\d+: bool;/u);
      expect(compiled.wgsl).toContain("u32(bg_uniforms.enabled) != 0u");
      expect(compiled.wgsl).toContain("nested_loop_condition_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0u, nested_loop_condition_helper_with_pointer_side_effect");

      const disabled = runCompiledKernelSemanticReference(compiled, {
        buffers: { storage: new Uint32Array([2]), out: new Uint32Array(1) },
        scalars: { enabled: 0 },
      }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] });
      expect([...disabled.buffers.storage as Uint32Array]).toEqual([2]);
      expect([...disabled.buffers.out as Uint32Array]).toEqual([1]);

      const enabled = runCompiledKernelSemanticReference(compiled, {
        buffers: { storage: new Uint32Array([2]), out: new Uint32Array(1) },
        scalars: { enabled: 1 },
      }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] });
      expect([...enabled.buffers.storage as Uint32Array]).toEqual([9]);
      expect([...enabled.buffers.out as Uint32Array]).toEqual([1]);

      const sequenceCompiled = compileCudaLiteKernel(`
  __device__ uint nested_sequence_loop_condition_helper_with_pointer_side_effect(uint *ptr, uint add) {
    atomicAdd(ptr, add);
    return ptr[0];
  }

  __global__ void nestedConditionalHelperSequenceLoopCondition(uint *storage, uint *out, int enabled) {
    for (int i = 0, j = 0; j < 1 && ((enabled != 0 ? nested_sequence_loop_condition_helper_with_pointer_side_effect(storage, 7u) : 0u) != 0u); ++i, ++j) {
      out[0] = 1u;
    }
  }`, { workgroupSize: [4, 1, 1] });

      expect(sequenceCompiled.wgsl).toMatch(/loop \{/u);
      expect(canRunCompiledKernelSemanticReference(sequenceCompiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(sequenceCompiled.wgslLegalizedKernelIr)).toBe(true);
      expect(sequenceCompiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(sequenceCompiled.wgsl).toContain("nested_sequence_loop_condition_helper_with_pointer_side_effect");
      expect(sequenceCompiled.wgsl).not.toContain("select(0u, nested_sequence_loop_condition_helper_with_pointer_side_effect");
    });

  it("keeps duplicate for-loop variable names scoped for signedness", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __global__ void kernel(int *out) {
    for (int k = 0; k < 8; k++) {
      out[k] = k;
    }
    for (unsigned int k = 0; k < 8; k++) {
      out[k] += (int)k;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("for (var k: i32 = 0; (u32(k) < 8u); k += 1)");
      expect(compiled.wgsl).toContain("for (var k: u32 = 0u; (k < 8u); k += 1u)");
      expect([...result.buffers.out as Int32Array]).toEqual([0, 2, 4, 6, 8, 10, 12, 14]);
    });

  it("rejects CUDA device trap instead of silently lowering it as a no-op", () => {
      expect(() => compileCudaLiteKernel(`
  __global__ void trap_kernel(int* out) {
    if (threadIdx.x == 0) { __trap(); }
    out[0] = 1;
  }`, { workgroupSize: [1, 1, 1] })).toThrow(/unsupported-device-trap/u);
    });

  it("models scalar-guarded CUDA traps as launch preconditions", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void guarded_trap(int* out, int C) {
    if (C % (32 * 4) != 0) {
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("bad shape");
      }
      __trap();
    }
    if (threadIdx.x < 1) {
      out[0] = 1;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const okInput = { buffers: { out: new Int32Array(1) }, scalars: { C: 128 } };
      const badInput = { buffers: { out: new Int32Array(1) }, scalars: { C: 127 } };
      const missingScalarInput = { buffers: { out: new Int32Array(1) } };

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-device-trap");
      expect(createCudaWebGpuExecutionPlan(compiled, okInput, launch).supported).toBe(true);
      expect([...runCompiledKernelReference(compiled, okInput, launch).buffers.out as Int32Array]).toEqual([1]);
      const badPlan = createCudaWebGpuExecutionPlan(compiled, badInput, launch);
      if (badPlan.supported) throw new Error("expected bad scalar trap precondition plan to be unsupported");
      expect(badPlan.diagnostics.map((diagnostic) => diagnostic.code)).toContain("cuda-launch-precondition-failed");
      expect(() => runCompiledKernelReference(compiled, badInput, launch)).toThrow(/guarded __trap would execute/u);
      const unknownPlan = createCudaWebGpuExecutionPlan(compiled, missingScalarInput, launch);
      if (unknownPlan.supported) throw new Error("expected unknown scalar trap precondition plan to be unsupported");
      expect(unknownPlan.diagnostics.map((diagnostic) => diagnostic.code)).toContain("cuda-launch-precondition-unknown");
    });

  it("ignores CUDA device trap in unreachable helpers for selected kernels", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void trap_helper() {
    __trap();
  }
  __global__ void reachable_kernel(int* out) {
    out[threadIdx.x] = 7;
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-device-trap");
    });

  it("lowers comma sequence return values in device helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ int sequence_return(int value) {
    int local = value;
    return (local += 2, local * 3);
  }

  __device__ int assignment_return(int value) {
    int local = value;
    return (local += 3, local = local + 4);
  }

  __global__ void sequence_return_kernel(int* out) {
    if (threadIdx.x == 0) {
      out[0] = sequence_return(4);
      out[1] = assignment_return(5);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Int32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-sequence-expression");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("local += 2;");
      expect(compiled.wgsl).toContain("return (local * 3);");
      expect(compiled.wgsl).toContain("local += 3;");
      expect(compiled.wgsl).toContain("local = (local + 4);");
      expect(compiled.wgsl).toContain("return local;");
      expect([...result.buffers.out as Int32Array]).toEqual([18, 12]);
      expect([...semanticResult.buffers.out as Int32Array]).toEqual([18, 12]);
    });

  it("lowers bool storage pointers through u32 WebGPU carriers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void boolStorage(bool *flags, uint *out) {
    bool before = flags[0];
    flags[1] = !before;
    flags[2] = before || flags[1];
    out[0] = flags[1] ? 7u : 3u;
    out[1] = flags[2] ? 11u : 5u;
  }`, { workgroupSize: [1, 1, 1] });
      const input = { buffers: { flags: new Uint32Array([0, 0, 0]), out: new Uint32Array(2) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("flags: array<u32>");
      expect(compiled.wgsl).toContain("flags[1u] = select(0u, 1u, !(before));");
      expect([...result.buffers.flags as Uint32Array]).toEqual([0, 1, 1]);
      expect([...result.buffers.out as Uint32Array]).toEqual([7, 11]);
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
