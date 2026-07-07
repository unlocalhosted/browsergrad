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
} from "../src/index";
import { lowerAnalyzedCudaLiteToKernelIr } from "../src/analyzer";
import {
  constantBufferInputs,
  cudaWebGpuDefaultReadbackNames,
  deviceGlobalBufferInputs,
} from "../src/webgpu_inputs";
import { deviceLaunchTreeIsExternallySilent } from "../src/runtime_elision";
import { packCudaWebGpuUniformParams } from "../src/webgpu_orchestration";

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

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

describe("CUDA-lite compiler", () => {
  it("parses and compiles SAXPY to WGSL", () => {
    const ast = parseCudaLite(SAXPY);
    const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });

    expect(ast.kernels[0]?.name).toBe("saxpy");
    expect(backendIr(compiled).params.map((param) => param.name)).toEqual(["x", "y", "a", "n"]);
    expect(Object.hasOwn(compiled, "ir")).toBe(false);
    expect(Object.hasOwn(compiled, "legacyIr")).toBe(false);
    expect(Object.hasOwn(compiled.loweringPlan, "canRunOnGpu")).toBe(false);
    expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(true);
    expect(compiled.semantic).toMatchObject({
      kind: "cuda-lite-semantic-model",
      kernelName: "saxpy",
    });
    expect(compiled.kernelIr.kind).toBe("semantic-kernel-ir");
    expect(compiled.kernelIr.name).toBe("saxpy");
    expect(compiled.kernelIr.params.map((param) => [param.name, param.addressSpace])).toEqual([
      ["x", "storage"],
      ["y", "storage"],
      ["a", "uniform"],
      ["n", "uniform"],
    ]);
    expect(compiled.kernelIr.operations.map((operation) => operation.kind)).toEqual([
      "declare",
      "branch",
    ]);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    const guardedStore = compiled.kernelIr.operations[1]?.kind === "branch"
      ? compiled.kernelIr.operations[1].consequent[0]
      : undefined;
    expect(guardedStore).toMatchObject({
      kind: "store",
      target: {
        base: "y",
        addressSpace: "storage",
        valueType: "float",
      },
      operator: "=",
    });
    expect(guardedStore?.kind === "store" ? guardedStore.reads.map((read) => read.base) : []).toEqual(["x", "y"]);
    expect("body" in compiled.kernelIr).toBe(false);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("@workgroup_size(8, 1, 1)");
    expect(compiled.wgsl).toContain("var<storage, read> x: array<f32>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> y: array<f32>;");
    expect(compiled.wgsl).toContain("bg_uniforms.a");
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

  it("runs SAXPY through the semantic CPU reference path", () => {
    const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [8, 1, 1] });
    const input = {
      buffers: {
        x: new Float32Array([1, 2, 3, 4]),
        y: new Float32Array([10, 20, 30, 40]),
      },
      scalars: { a: 2, n: 4 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [8, 1, 1] as const };
    const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
    const result = runCompiledKernelReference(
      compiled,
      input,
      launch,
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect([...semanticResult.buffers.y as Float32Array]).toEqual([12, 24, 36, 48]);
    expect([...result.buffers.y as Float32Array]).toEqual([12, 24, 36, 48]);
    expect(result.trace.some((thread) => thread.writes.length > 0)).toBe(true);
  });

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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("continue;");
    expect(compiled.wgsl).toContain("break;");
    expect([...semanticResult.buffers.out as Int32Array]).toEqual([8]);
    expect([...referenceResult.buffers.out as Int32Array]).toEqual([8]);
  });

  it("runs scalar device helper calls through semantic reference and WGSL paths", () => {
    const compiled = compileCudaLiteKernel(`
__device__ int affine(int x, int scale, int bias) {
  int y = x * scale;
  return y + bias;
}
__global__ void helperKernel(int* out, int scale, int bias) {
  int idx = threadIdx.x;
  out[idx] = affine(idx, scale, bias);
}
`, { workgroupSize: [4, 1, 1] });
    const input = {
      buffers: { out: new Int32Array(4) },
      scalars: { scale: 3, bias: 2 },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
    const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
    const referenceResult = runCompiledKernelReference(compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("fn affine(");
    expect(compiled.wgsl).toContain("return (y + bias);");
    expect([...semanticResult.buffers.out as Int32Array]).toEqual([2, 5, 8, 11]);
    expect([...referenceResult.buffers.out as Int32Array]).toEqual([2, 5, 8, 11]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("continuing {");
    expect(compiled.wgsl).toContain("break if !((u32(i) < 4u));");
    expect([...semanticResult.buffers.out as Int32Array]).toEqual([8]);
    expect([...referenceResult.buffers.out as Int32Array]).toEqual([8]);
  });

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

  it("emits simple integer atomicAdd from semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomicCounter(uint* counter, uint* out) {
  atomicAdd(&counter[0], 2u);
  out[0] = counter[0];
}
`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { counter: new Uint32Array([5]), out: new Uint32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { counter: new Uint32Array([5]), out: new Uint32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("counter: array<atomic<u32>>");
    expect(compiled.wgsl).toContain("_ = atomicAdd(&counter[0u], 2u);");
    expect(compiled.wgsl).toContain("out[0u] = atomicLoad(&counter[0u]);");
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([7]);
    expect([...result.buffers.out as Uint32Array]).toEqual([7]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("return;");
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1, 2, 0, 0]);
    expect([...result.buffers.out as Uint32Array]).toEqual([1, 2, 0, 0]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).not.toContain("var p:");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([11]);
    expect([...result.buffers.out as Float32Array]).toEqual([11]);
  });

  it("emits integer read-modify-write atomic statements from semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomicRmw(int* x, int* out) {
  atomicSub(&x[0], 3);
  atomicMin(&x[1], 5);
  atomicMax(&x[2], 8);
  atomicAnd(&x[3], 6);
  atomicOr(&x[4], 3);
  atomicXor(&x[5], 10);
  atomicExch(&x[6], 42);
  atomicCAS(&x[7], 0, 99);
  atomicAdd_system(&x[8], 2);
  for (int i = 0; i < 9; i++) {
    out[i] = x[i];
  }
}
`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        x: new Int32Array([10, 7, 4, 9, 12, 11, 8, 0, 1]),
        out: new Int32Array(9),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
    const result = runCompiledKernelReference(compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("atomicSub(&x[0u], 3)");
    expect(compiled.wgsl).toContain("atomicMin(&x[1u], 5)");
    expect(compiled.wgsl).toContain("atomicMax(&x[2u], 8)");
    expect(compiled.wgsl).toContain("atomicAnd(&x[3u], 6)");
    expect(compiled.wgsl).toContain("atomicOr(&x[4u], 3)");
    expect(compiled.wgsl).toContain("atomicXor(&x[5u], 10)");
    expect(compiled.wgsl).toContain("atomicExchange(&x[6u], 42)");
    expect(compiled.wgsl).toContain("atomicCompareExchangeWeak(&x[7u], 0, 99)");
    expect(compiled.wgsl).toContain("atomicAdd(&x[8u], 2)");
    expect([...semanticResult.buffers.out as Int32Array]).toEqual([7, 5, 8, 0, 15, 1, 42, 99, 3]);
    expect([...result.buffers.out as Int32Array]).toEqual([7, 5, 8, 0, 15, 1, 42, 99, 3]);
  });

  it("emits return-valued integer atomics from semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomicReturns(uint* counter, uint* out) {
  out[0] = atomicAdd(&counter[0], 2u);
  out[1] = atomicCAS(&counter[1], 0u, 44u);
  out[2] = counter[0];
  out[3] = counter[1];
}
`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        counter: new Uint32Array([5, 0]),
        out: new Uint32Array(4),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
    const result = runCompiledKernelReference(compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("counter: array<atomic<u32>>");
    expect(compiled.wgsl).toContain("out[0u] = atomicAdd(&counter[0u], 2u);");
    expect(compiled.wgsl).toContain("out[1u] = atomicCompareExchangeWeak(&counter[1u], 0u, 44u).old_value;");
    expect(compiled.wgsl).toContain("out[2u] = atomicLoad(&counter[0u]);");
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([5, 0, 7, 44]);
    expect([...result.buffers.out as Uint32Array]).toEqual([5, 0, 7, 44]);
  });

  it("keeps canonical lab examples executable through Kernel IR and CPU reference", () => {
    const saxpy = compileCudaLiteKernelForWebGpu(compilerExampleText("saxpy.cu"), {
      workgroupSize: [8, 1, 1],
    });
    expect(saxpy.kernelIr.kind).toBe("semantic-kernel-ir");
    expect(summarizeCudaWebGpuExecutionPlan(createCudaWebGpuExecutionPlan(
      saxpy,
      {
        buffers: {
          x: new Float32Array([1, 2, 3, 4]),
          y: new Float32Array([10, 20, 30, 40]),
        },
        scalars: { a: 2, n: 4 },
      },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    ))).toMatchObject({ canRunOnWebGpu: true, kind: "single-dispatch" });
    const saxpyRef = runCompiledKernelReference(
      saxpy,
      {
        buffers: {
          x: new Float32Array([1, 2, 3, 4]),
          y: new Float32Array([10, 20, 30, 40]),
        },
        scalars: { a: 2, n: 4 },
      },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    );
    expect([...saxpyRef.buffers.y as Float32Array]).toEqual([12, 24, 36, 48]);

    const guarded = compileCudaLiteKernelForWebGpu(compilerExampleText("guarded-map.cu"), {
      workgroupSize: [8, 1, 1],
    });
    const guardedRef = runCompiledKernelReference(
      guarded,
      {
        buffers: {
          input: new Float32Array([-2, 3, -4, 5]),
          output: new Float32Array([99, 99, 99, 99]),
        },
        scalars: { n: 3 },
      },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    );
    expect([...guardedRef.buffers.output as Float32Array]).toEqual([0, 3, 0, 99]);

    const tiled = compileCudaLiteKernelForWebGpu(compilerExampleText("tiled-matmul.cu"), {
      workgroupSize: [2, 2, 1],
    });
    const tiledRef = runCompiledKernelReference(
      tiled,
      {
        buffers: {
          A: new Float32Array([
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
          ]),
          B: new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
          ]),
          C: new Float32Array(16),
        },
        scalars: { N: 4 },
      },
      { gridDim: [2, 2, 1], blockDim: [2, 2, 1] },
    );
    expect([...tiledRef.buffers.C as Float32Array]).toEqual([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
    ]);
    expect(summarizeCudaWebGpuExecutionPlan(createCudaWebGpuExecutionPlan(
      tiled,
      {
        buffers: {
          A: new Float32Array(16),
          B: new Float32Array(16),
          C: new Float32Array(16),
        },
        scalars: { N: 4 },
      },
      { gridDim: [2, 2, 1], blockDim: [2, 2, 1] },
    ))).toMatchObject({ canRunOnWebGpu: true, kind: "single-dispatch" });
  });

  it("uses C-style truncating integer division and remainder in the reference interpreter", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void divmod(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = (i / 2) + (i % 2) * 10;
  }
}
`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Int32Array(4) },
        scalars: { n: 4 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect([...result.buffers.out as Int32Array]).toEqual([0, 10, 1, 11]);
  });

  it("treats scalar kernel parameters as mutable per-thread locals", () => {
    const source = `
__global__ void mutateParams(float* out, float alpha, float beta, int n, bool enabled) {
  beta /= alpha;
  n += 1;
  enabled = !enabled;
  if (threadIdx.x == 0) {
    out[0] = enabled ? -1.0f : beta + (float)n;
  }
}
`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(1) },
        scalars: { alpha: 2, beta: 8, n: 3, enabled: 1 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Float32Array]).toEqual([8]);
    expect(compiled.wgsl).toContain("var beta: f32 = bg_uniforms.beta;");
    expect(compiled.wgsl).toContain("var n: i32 = bg_uniforms.n;");
    expect(compiled.wgsl).toContain("var enabled: bool = (bg_uniforms.enabled != 0u);");
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
    expect(backendIr(compiled).workgroupSize).toEqual([8, 1, 1]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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

    expect(compiled.wgsl).toContain("select(4294967295u, 2u");
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

    expect(compiled.wgsl).toContain("fn copyArrayParam(out_buffer_arg: u32, out_base_arg: u32, src_buffer_arg: u32, src_base_arg: u32");
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

  it("lowers vector reinterpret memory-view helpers through the pointer ABI", () => {
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

    expect(compiled.wgsl).toContain("return bg_ptr_read_f32x4(address_buffer, address_base);");
    expect(compiled.wgsl).toContain("bg_ptr_write_f32x4(address_buffer, address_base, val);");
    expect(compiled.wgsl).toContain("case 1u: { return vec4<f32>(inp[(index + 0u)], inp[(index + 1u)], inp[(index + 2u)], inp[(index + 3u)]); }");
    expect(compiled.wgsl).toContain("case 0u: { out[(index + 0u)] = value.x; out[(index + 1u)] = value.y; out[(index + 2u)] = value.z; out[(index + 3u)] = value.w; return; }");
    expect(compiled.wgsl).not.toContain("address[");
    expect([...result.buffers.out as Float32Array]).toEqual([1, 12, 3, 4]);
  });

  it("accepts s0-s3 CUDA vector lane aliases", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void vector_lane_aliases(float* out) {
  float4 value = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  value.s2 = value.s0 + value.s1;
  out[0] = value.s0;
  out[1] = value.s1;
  out[2] = value.s2;
  out[3] = value.s3;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-vector-member");
    expect(compiled.wgsl).toContain("value.x");
    expect(compiled.wgsl).toContain("value.z");
    expect([...result.buffers.out as Float32Array]).toEqual([1, 2, 3, 4]);
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

    expect(compiled.wgsl).toContain("write_lane_offset(0u, (0u + u32((0 + 4))), idx");
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

    expect(compiled.wgsl).toContain("load_scalar_offset(1u, (0u + u32(((0 + 1) * 4))), idx");
    expect(compiled.wgsl).toContain("return inp[index];");
    expect(compiled.wgsl).toContain("return bg_ptr_read_f32(inp_buffer, (inp_base + u32(idx)));");
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

    expect(compiled.wgsl).toContain("write_scalar_offset(0u, (0u + u32(((0 + 1) * 4))), idx");
    expect(compiled.wgsl).toContain("out[index] = value;");
    expect(compiled.wgsl).toContain("bg_ptr_write_f32(out_buffer, (out_base + u32(idx)), value);");
  });

  it("keeps shifted vector-backed scalar atomics on atomic storage carriers", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__device__ void add_scalar_offset(float* out, int idx, float value) {
  atomicAdd(&out[idx], value);
}

__global__ void vector_to_scalar_atomic_offset(float4* out) {
  int idx = threadIdx.x;
  float* scalarView = reinterpret_cast<float*>(out + 1);
  add_scalar_offset(scalarView, idx, 2.0f + (float)idx);
}`, { workgroupSize: [2, 1, 1] });

    expect(compiled.wgsl).toContain("var<storage, read_write> out: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("add_scalar_offset(0u, (0u + u32(((0 + 1) * 4))), idx");
    expect(compiled.wgsl).toContain("bg_ptr_atomicAdd_f32(out_buffer, (out_base + u32(idx)), value);");
    expect(compiled.wgsl).toContain("case 0u: { return bg_atomicAdd_f32(&out[index], value); }");
  });

  it("keeps shifted vector-backed integer scalar atomics on atomic storage carriers", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__device__ void add_uint_scalar_offset(uint* out, int idx, uint value) {
  atomicAdd(&out[idx], value);
}

__global__ void uint_vector_to_scalar_atomic_offset(uint4* out) {
  int idx = threadIdx.x;
  uint* scalarView = reinterpret_cast<uint*>(out + 1);
  add_uint_scalar_offset(scalarView, idx, 2u + (uint)idx);
}`, { workgroupSize: [2, 1, 1] });

    expect(compiled.wgsl).toContain("var<storage, read_write> out: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("add_uint_scalar_offset(0u, (0u + u32(((0 + 1) * 4))), idx");
    expect(compiled.wgsl).toContain("case 0u: { return atomicAdd(&out[index], value); }");
  });

  it("keeps casted vector pointer arithmetic atomics on pointer helpers", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__device__ void add_uint_vector_slot(uint4* slot, uint value) {
  uint* lanes = reinterpret_cast<uint*>(slot);
  atomicAdd(lanes + 3, value);
}

__global__ void vector_pointer_array_atomic_offset(uint4* out) {
  uint4* slots[2];
  slots[0] = out + 1;
  slots[1] = out + 2;
  add_uint_vector_slot(slots[0], 9u);
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_u32(");
    expect(compiled.wgsl).toContain("bg_ptr_atomicAdd_u32(slot_buffer, ((slot_base + u32((0 * 4))) + u32(3)), value)");
    expect(compiled.wgsl).not.toContain("atomicAdd((slot_base + 3u), value)");
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

    expect(compiled.wgsl).toContain("fn bg_ptr_atomicCompareExchange_u32(");
    expect(compiled.wgsl).toContain("bg_ptr_atomicCompareExchange_u32(slot_buffer, ((slot_base + u32((0 * 4))) + u32(2)), compare, value)");
    expect(compiled.wgsl).not.toContain("atomicCompareExchangeWeak((slot_base + 2u)");
  });

  it("keeps vector pointer-array compound writes on pointer helpers after atomics", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__device__ void atomic_uint_vector_slot(uint4* slot, uint value) {
  uint* lanes = reinterpret_cast<uint*>(slot);
  atomicAdd(lanes + 1, value);
}

__device__ void add_uint_vector_slot(uint4* slot, uint4 value) {
  slot[0] += value;
}

__device__ void add_uint_vector_slot_z(uint4* slot, uint value) {
  slot[0].z += value;
}

__global__ void vector_pointer_array_compound_after_atomic(uint4* out) {
  uint4* slots[2];
  slots[0] = out + 1;
  slots[1] = out + 2;
  atomic_uint_vector_slot(slots[0], 5u);
  add_uint_vector_slot(slots[0], make_uint4(1u, 2u, 3u, 4u));
  add_uint_vector_slot_z(slots[1], 7u);
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_u32(");
    expect(compiled.wgsl).toContain("fn bg_ptr_write_u32x4(");
    expect(compiled.wgsl).toContain("bg_ptr_atomicAdd_u32(slot_buffer");
    expect(compiled.wgsl).toContain("bg_ptr_write_u32x4(slot_buffer");
    expect(compiled.wgsl).toContain("atomicLoad(&out[(index + 0u)]");
    expect(compiled.wgsl).toContain("atomicStore(&out[(index + 0u)]");
    expect(compiled.wgsl).not.toContain("atomicLoad(&out[((u32(index) * 4u) + 0u)]");
    expect(compiled.wgsl).not.toContain("atomicStore(&out[((u32(index) * 4u) + 0u)]");
    expect(compiled.wgsl).not.toContain("slot[");
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
    expect(compiled.wgsl.match(/\bactive_conditional_pointer_array_index_helper\(/gu) ?? []).toHaveLength(2);
    expect(compiled.wgsl).toMatch(/let bg_pointer_array_index_\d+: u32 = active_conditional_pointer_array_index_helper\(/u);
    expect(compiled.wgsl).not.toContain("select(0u, active_conditional_pointer_array_index_helper");
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

    expect(compiled.wgsl.match(/\bpointer_array_assignment_index_helper\(/gu) ?? []).toHaveLength(2);
    expect(compiled.wgsl).toMatch(/let bg_pointer_array_index_\d+: u32 = pointer_array_assignment_index_helper\(/u);
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

    expect(compiled.wgsl.match(/\bnested_pointer_array_target_index_helper\(/gu) ?? []).toHaveLength(2);
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

    expect(compiled.wgsl.match(/\bpointer_array_compare_index_helper\(/gu) ?? []).toHaveLength(2);
    expect(compiled.wgsl).toMatch(/let bg_pointer_array_index_\d+: u32 = pointer_array_compare_index_helper\(/u);
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

    expect(compiled.wgsl.match(/\bpointer_array_var_init_compare_index_helper\(/gu) ?? []).toHaveLength(2);
    expect(compiled.wgsl).toMatch(/let bg_pointer_array_index_\d+: u32 = pointer_array_var_init_compare_index_helper\(/u);
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

    expect(compiled.wgsl.match(/\bpointer_array_diff_index_helper\(/gu) ?? []).toHaveLength(2);
    expect(compiled.wgsl).toMatch(/let bg_pointer_array_index_\d+: u32 = pointer_array_diff_index_helper\(/u);
    expect(compiled.wgsl).toContain("i32((0u + u32(1)))");
    expect(compiled.wgsl).not.toContain("bg_storage +");
  });

  it("scales vector pointer differences by vector lane width", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__global__ void vector_pointer_difference(uint4 *out, int *summary) {
  uint4 *left = out + 2;
  uint4 *right = out;
  summary[0] = left - right;
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toMatch(/summary\[0\] = i32\(\(\(i32\(.+\) - i32\(.+\)\) \/ 4\)\);/u);
    expect(compiled.wgsl).not.toMatch(/summary\[0\] = i32\(\(i32\(.+\) - i32\(.+\)\)\);/u);
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

    expect(compiled.wgsl.match(/\bvector_pointer_array_diff_index_helper\(/gu) ?? []).toHaveLength(2);
    expect(compiled.wgsl).toMatch(/let bg_pointer_array_index_\d+: u32 = vector_pointer_array_diff_index_helper\(/u);
    expect(compiled.wgsl).toMatch(/summary\[0\] = i32\(select\(0, \(\(i32\(.+\) - i32\(.+\)\) \/ 4\),/u);
  });

  it("keeps scalar-view pointer differences over vector roots in scalar lanes", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__global__ void scalar_view_vector_pointer_difference(uint4 *out, int *summary) {
  uint *left = reinterpret_cast<uint*>(out + 1);
  uint *right = reinterpret_cast<uint*>(out);
  summary[0] = left - right;
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toMatch(/summary\[0\] = i32\(\(i32\(.+\) - i32\(.+\)\)\);/u);
    expect(compiled.wgsl).not.toMatch(/summary\[0\] = i32\(\(\(i32\(.+\) - i32\(.+\)\) \/ 4\)\);/u);
  });

  it("scales inline cast pointer differences over byte roots", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__global__ void byte_root_inline_cast_pointer_difference(uchar *bytes, int *summary) {
  summary[0] = reinterpret_cast<float*>(bytes + 8) - reinterpret_cast<float*>(bytes);
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toMatch(/summary\[0\] = i32\(\(\(i32\(.+\) - i32\(.+\)\) \/ 4\)\);/u);
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

    expect(compiled.wgsl.match(/\bbyte_pointer_array_diff_index_helper\(/gu) ?? []).toHaveLength(2);
    expect(compiled.wgsl).toMatch(/summary\[0\] = i32\(select\(0, select\(.+ \/ 4\), \(ptrs_buffer\[.+\] == 0u\)\), \(ptrs_buffer\[.+\] == 0u\)\)\);/u);
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

    expect(compiled.wgsl.match(/\bbyte_vector_pointer_array_diff_index_helper\(/gu) ?? []).toHaveLength(2);
    expect(compiled.wgsl).toMatch(/summary\[0\] = i32\(select\(0, select\(.+ \/ 16\), \(ptrs_buffer\[.+\] == 0u\)\), \(ptrs_buffer\[.+\] == 0u\)\)\);/u);
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

    expect(compiled.wgsl).toContain("bump_roundtrip_vec(0u, (0u + u32((((0 + 1) * 4) + 0))), idx");
    expect(compiled.wgsl).toContain("bg_ptr_read_f32x4(out_buffer, (out_base + (u32(idx) * 4u)))");
    expect(compiled.wgsl).toContain("bg_ptr_write_f32x4(out_buffer, (out_base + (u32(idx) * 4u)), vec4<f32>");
  });

  it("reads shifted vector-backed device global scalar atomics from flat lanes", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__device__ uint4 g_vec[2];

__device__ void add_global_scalar(uint* out, int idx, uint value) {
  atomicAdd(&out[idx], value);
}

__global__ void device_global_vector_to_scalar_atomic(uint* out) {
  int idx = threadIdx.x;
  uint* scalarView = reinterpret_cast<uint*>(g_vec + 1);
  add_global_scalar(scalarView, idx, 5u + (uint)idx);
  out[idx] = scalarView[idx];
}`, { workgroupSize: [2, 1, 1] });

    expect(compiled.wgsl).toContain("var<storage, read_write> g_vec: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("add_global_scalar(1u, (0u + u32(((0 + 1) * 4))), idx");
    expect(compiled.wgsl).toContain("atomicLoad(&g_vec[(u32(((0 + 1) * 4)) + u32(idx))])");
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

    expect(compiled.wgsl).toContain("return vec4<f32>(f32(bg_shared[(u32((index + 0u)) / 4u)]");
    expect(compiled.wgsl).not.toContain("return vec4<f32>(bg_shared[(index + 0u)]");
  });

  it("flattens shared vector scalar atomics to scalar atomic lanes", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__device__ void add_shared_scalar(uint* out, int idx, uint value) {
  atomicAdd(&out[idx], value);
}

__global__ void shared_vector_to_scalar_atomic(uint* out) {
  __shared__ uint4 tile[2];
  int idx = threadIdx.x;
  uint* scalarView = reinterpret_cast<uint*>(tile + 1);
  add_shared_scalar(scalarView, idx, 7u + (uint)idx);
  out[idx] = scalarView[idx];
}`, { workgroupSize: [2, 1, 1] });

    expect(compiled.wgsl).toContain("var<workgroup> tile: array<atomic<u32>, 8>;");
    expect(compiled.wgsl).toContain("add_shared_scalar(1u, (0u + u32(((0 + 1) * 4))), idx");
    expect(compiled.wgsl).toContain("case 1u: { return atomicAdd(&tile[index], value); }");
    expect(compiled.wgsl).toContain("atomicLoad(&tile[(u32(((0 + 1) * 4)) + u32(idx))])");
  });

  it("reads and writes packed shared vector scalar helper lanes", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__device__ void add_shared_float_lane(float* lanes, int idx, float value) {
  lanes[idx] = lanes[idx] + value;
}

__global__ void shared_vector_scalar_lane_write(float* out) {
  __shared__ float4 tile[2];
  if (threadIdx.x == 0) {
    tile[1] = make_float4(5.0f, 6.0f, 7.0f, 8.0f);
    float* scalarView = reinterpret_cast<float*>(tile + 1);
    add_shared_float_lane(scalarView, 1, 0.5f);
    out[0] = tile[1].y;
  }
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("add_shared_float_lane(1u, (0u + u32(((0 + 1) * 4))), 1, 0.5");
    expect(compiled.wgsl).toContain("return f32(tile[(u32(index) / 4u)][(u32(index) % 4u)]);");
    expect(compiled.wgsl).toContain("tile[(u32(index) / 4u)] = vec4<f32>(select((tile[(u32(index) / 4u)]).x");
  });

  it("emits shared float vector scalar atomic helpers for flat lanes", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__device__ void add_shared_float3_scalar(float* out, int idx, float value) {
  atomicAdd(&out[idx], value);
}

__global__ void shared_float3_vector_to_scalar_atomic(float* out) {
  __shared__ float3 tile[2];
  int idx = threadIdx.x;
  float* scalarView = reinterpret_cast<float*>(tile + 1);
  add_shared_float3_scalar(scalarView, idx, 7.0f + (float)idx);
  out[idx] = scalarView[idx];
}`, { workgroupSize: [2, 1, 1] });

    expect(compiled.wgsl).toContain("var<workgroup> tile: array<atomic<u32>, 6>;");
    expect(compiled.wgsl).toContain("fn bg_atomicAdd_f32_workgroup");
    expect(compiled.wgsl).toContain("add_shared_float3_scalar(1u, (0u + u32(((0 + 1) * 3))), idx");
    expect(compiled.wgsl).toContain("case 1u: { return bg_atomicAdd_f32_workgroup(&tile[index], value); }");
    expect(compiled.wgsl).toContain("bitcast<f32>(atomicLoad(&tile[(u32(((0 + 1) * 3)) + u32(idx))]))");
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
    expect(compiled.wgsl).toContain("bg_ptr_write_u32(ptr_buffer, (ptr_base + u32(index)), (bg_ptr_read_u32(ptr_buffer, (ptr_base + u32(index))) + 1u))");
    expect(compiled.wgsl).not.toContain("u32(index * vec4<u32>");
    expect(compiled.wgsl).not.toContain("bg_ptr_read_u32(ptr_buffer, (ptr_base + u32(index))) =");
  });

  it("lowers device helper pointer-param writes fed by atomic return values", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void helper_rmw(uint* counter, float* out) {
  uint* ptr = counter;
  out[0] = atomicSub(ptr, 2);
  out[1] = ptr[0];
}

__global__ void helper_atomic_rmw(uint* counter, float* out) {
  helper_rmw(counter, out);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          counter: new Uint32Array([5]),
          out: new Float32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.counter as Uint32Array]).toEqual([3]);
    expect([...result.buffers.out as Float32Array]).toEqual([5, 3]);
    expect(compiled.wgsl).toContain("bg_ptr_write_f32(out_buffer, (out_base + u32(0)), f32(");
    expect(compiled.wgsl).not.toContain("bg_ptr_read_f32(out_buffer, (out_base + u32(0))) =");
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(compiled.wgsl).toContain("data[i] = (i32(data[i]) + 1)");
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

    expect(compiled.wgsl).toContain("fn bg_ptr_read_f32(buffer: u32, index: u32) -> f32 {\n  return tile[index];");
    expect(compiled.wgsl).toContain("fn bg_ptr_write_f32(buffer: u32, index: u32, value: f32) {\n  tile[index] = value;");
    expect(compiled.wgsl).not.toContain("switch buffer");
    expect(compiled.wgsl).toContain("writeTile(1u, 0u, tid");
    expect([...result.buffers.out as Float32Array]).toEqual([4, 3, 2, 1]);
  });

  it("supports mutable __device__ scalar globals", () => {
    const compiled = compileCudaLiteKernel(`
__device__ static unsigned int numErrors = 2, errorFound = 0;

__global__ void globals_scalar(uint* data, uint* out) {
  if (data[0] != 7u && errorFound == 0u) {
    numErrors += 1u;
    errorFound = 1u;
  }
  out[0] = numErrors;
  out[1] = errorFound;
}`, { workgroupSize: [1, 1, 1] });

    const result = runCompiledKernelReference(
      compiled,
      { buffers: { data: new Uint32Array([5]), out: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { data: new Uint32Array([5]), out: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.ast.deviceGlobals.map((global) => global.name)).toEqual(["numErrors", "errorFound"]);
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([3, 1]);
    expect([...semanticResult.buffers.numErrors as Uint32Array]).toEqual([3]);
    expect([...semanticResult.buffers.errorFound as Uint32Array]).toEqual([1]);
    expect([...result.buffers.out as Uint32Array]).toEqual([3, 1]);
    expect([...result.buffers.numErrors as Uint32Array]).toEqual([3]);
    expect([...result.buffers.errorFound as Uint32Array]).toEqual([1]);
    expect(compiled.wgsl).toContain("var<storage, read_write> numErrors: array<u32>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> errorFound: array<u32>;");
    expect(compiled.wgsl).toContain("numErrors[0u] = (numErrors[0u] + 1u)");
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
    expect(compiled.wgsl).toContain("setCallValue(1u, 0u, i");
    expect(compiled.wgsl).toContain("fn bg_ptr_read_f32(buffer: u32, index: u32) -> f32 {\n  return d_CallValue[index];");
    expect(compiled.wgsl).toContain("fn bg_ptr_write_f32(buffer: u32, index: u32, value: f32) {\n  d_CallValue[index] = value;");
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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

  it("supports atomic operations on __device__ globals", () => {
    const compiled = compileCudaLiteKernel(`
__device__ unsigned int counter = 0;

__global__ void globals_atomic(uint* out) {
  int i = threadIdx.x;
  out[i] = atomicAdd(&counter, 1u);
}`, { workgroupSize: [4, 1, 1] });

    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Uint32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([0, 1, 2, 3]);
    expect([...semanticResult.buffers.counter as Uint32Array]).toEqual([4]);
    expect([...result.buffers.out as Uint32Array]).toEqual([0, 1, 2, 3]);
    expect([...result.buffers.counter as Uint32Array]).toEqual([4]);
    expect(backendIr(compiled).atomicDeviceGlobals).toEqual(["counter"]);
    expect(compiled.wgsl).toContain("var<storage, read_write> counter: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("atomicAdd(&counter[0u], 1u)");
  });

  it("flattens multidimensional __device__ globals through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint gMatrix[2][3];

__global__ void globals_matrix(uint* out) {
  int row = threadIdx.y;
  int col = threadIdx.x;
  gMatrix[row][col] = uint(row * 10 + col);
  out[row * 3 + col] = gMatrix[row][col];
}`, { workgroupSize: [3, 2, 1] });
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Uint32Array(6) } },
      { gridDim: [1, 1, 1], blockDim: [3, 2, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(6) } },
      { gridDim: [1, 1, 1], blockDim: [3, 2, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var<storage, read_write> gMatrix: array<u32>;");
    expect(compiled.wgsl).toContain("* 3u");
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([0, 1, 2, 10, 11, 12]);
    expect([...semanticResult.buffers.gMatrix as Uint32Array]).toEqual([0, 1, 2, 10, 11, 12]);
    expect([...result.buffers.out as Uint32Array]).toEqual([0, 1, 2, 10, 11, 12]);
    expect([...result.buffers.gMatrix as Uint32Array]).toEqual([0, 1, 2, 10, 11, 12]);
  });

  it("supports read-modify-write atomics through device pointer helper parameters to __device__ globals", () => {
    const compiled = compileCudaLiteKernel(`
__device__ int g_i[1];
__device__ float g_f[1];

__device__ void helper_global_rmw(int* xi, float* xf, float* out) {
  out[0] = float(atomicSub(xi, 2));
  out[1] = float(atomicMin(xi, 5));
  out[2] = float(atomicMax(xi, 9));
  out[3] = float(atomicAnd(xi, 6));
  out[4] = float(atomicOr(xi, 10));
  out[5] = float(atomicXor(xi, 3));
  out[6] = atomicSub(xf, 1.5f);
  out[7] = atomicMin(xf, 2.0f);
  out[8] = atomicMax(xf, 5.0f);
  out[9] = float(xi[0]);
  out[10] = xf[0];
}

__global__ void device_global_atomic_rmw(float* out) {
  if (threadIdx.x == 0) {
    helper_global_rmw(g_i, g_f, out);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(11) },
        deviceGlobals: {
          g_i: new Int32Array([10]),
          g_f: new Float32Array([4]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn bg_ptr_atomicSub_i32");
    expect(compiled.wgsl).toContain("case 1u: { return atomicSub(&g_i[index], value); }");
    expect(compiled.wgsl).toContain("case 2u: { return bg_atomicSub_f32(&g_f[index], value); }");
    expect([...result.buffers.out as Float32Array]).toEqual([10, 8, 5, 9, 0, 10, 4, 2.5, 2, 9, 5]);
    expect([...result.buffers.g_i as Int32Array]).toEqual([9]);
    expect([...result.buffers.g_f as Float32Array]).toEqual([5]);
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
    expect([...result.buffers.out as Uint32Array]).toEqual([
      (((7 * 1664525 + 1013904223) >>> 0) * 1664525 + 1013904223) >>> 0,
      (((8 * 1664525 + 1013904223) >>> 0) * 1664525 + 1013904223) >>> 0,
    ]);
  });

  it("lowers fixed thread-local arrays through reference and WGSL", () => {
    const compiled = compileCudaLiteKernel(LOCAL_ARRAY, { workgroupSize: [4, 1, 1] });
    expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("vals[0u][0u] = 1.0;");
    expect(compiled.wgsl).toContain("vals[1u][0u] = 3.0;");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([3, 3]);
    expect([...result.buffers.out as Float32Array]).toEqual([3, 3]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("regs[fill_regs_0][fill_regs_1] = 7.0;");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([7, 7]);
    expect([...result.buffers.out as Float32Array]).toEqual([7, 7]);
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

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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

    expect(compiled.wgsl).toMatch(/var p_\d+_buffer: u32 = 4294967295u;/u);
    expect(compiled.wgsl).toMatch(/var p_\d+_base: u32 = 0u;/u);
    expect([...result.buffers.out as Float32Array]).toEqual([1, 2, 3, 0]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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

    expect(compiled.wgsl).toContain("read_one(0u, (0u + u32(1))");
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

    expect(compiled.wgsl).toContain("fn sum_local(values: ptr<function, f32>");
    expect(compiled.wgsl).toContain("sum_local(&values[0]");
    expect([...result.buffers.out as Float32Array]).toEqual([7]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).not.toContain("var p:");
    expect([...result.buffers.out as Float32Array]).toEqual([5, 5]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([5, 5]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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

    expect(compiled.wgsl).toContain("var p_buffer: array<u32, 3>;");
    expect(compiled.wgsl).toContain("p_buffer[u32(0)] =");
    expect(compiled.wgsl).toContain("read_x(p_buffer[u32(0)], p_base[u32(0)]");
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
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          out: new Int32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toMatch(/var shared_words_\d+_buffer: u32/u);
    expect(compiled.wgsl).toContain("var<workgroup> scratch: array<atomic<u32>, 4>;");
    expect(compiled.wgsl).toContain("atomicLoad(&scratch");
    expect(compiled.wgsl).toContain("atomicOr(&scratch");
    expect([...result.buffers.out as Int32Array]).toEqual([0x04030201, 9]);

    const storageCompiled = compileCudaLiteKernel(`
__global__ void storageByteReinterpret(const uchar* input, int* out) {
  int4 *lane_ptr = (int4 *)(input + threadIdx.x);
  lane_ptr = (int4 *)((uchar *)lane_ptr + 4);
  if (threadIdx.x == 0) {
    out[0] = (*lane_ptr).x;
  }
}`, { workgroupSize: [1, 1, 1] });
    expect(storageCompiled.wgsl).toMatch(/var lane_ptr_\d+_buffer: u32/u);
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

    expect(compiled.wgsl).toContain("bg_ptr_write_f32");
    expect(compiled.wgsl).toContain(">> 2u");
    expect([...result.buffers.out as Float32Array]).toEqual([1, 2]);
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

    expect(compiled.wgsl).toContain("bitcast<f32>(select(");
    expect(compiled.wgsl).toContain("atomicStore(&scratch");
    expect(compiled.wgsl).toContain("atomicOr(&scratch");
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

    expect(compiled.wgsl).toContain("calcNormal(&vertlist[v_base[u32(0)]]");
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

    expect(compiled.wgsl).toContain("var bytes: u32 = 4u");
    expect([...result.buffers.out as Uint32Array]).toEqual([4]);
  });

  it("folds CUDA sizeof type names and C character literals", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void c_layout_literals(uint* out) {
  if (threadIdx.x == 0) {
    out[0] = sizeof(unsigned char);
    out[1] = sizeof(char);
    out[2] = sizeof(float4);
    out[3] = '|';
    out[4] = '\\n';
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(5) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("out[0u] = 1u");
    expect(compiled.wgsl).toContain("out[2u] = 16u");
    expect([...result.buffers.out as Uint32Array]).toEqual([1, 1, 16, 124, 10]);
  });

  it("folds sizeof and alignof for modeled value expressions", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void value_sizeof(uint* out, uchar* bytes, float4* vectors) {
  uchar b = bytes[0];
  float4 v = vectors[0];
  if (threadIdx.x == 0) {
    out[0] = sizeof(b);
    out[1] = alignof(b);
    out[2] = sizeof(v);
    out[3] = sizeof(v.x);
    out[4] = sizeof(vectors[0]);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          out: new Uint32Array(5),
          bytes: new Uint32Array([7]),
          vectors: new Float32Array([1, 2, 3, 4]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-sizeof");
    expect(compiled.wgsl).toContain("out[0] = u32(1)");
    expect(compiled.wgsl).toContain("out[2] = u32(16)");
    expect(compiled.wgsl).toContain("out[3] = u32(4)");
    expect([...result.buffers.out as Uint32Array]).toEqual([1, 1, 16, 4, 16]);
  });

  it("folds sizeof for fixed local shared constant and device-global arrays", () => {
    const compiled = compileCudaLiteKernel(`
__constant__ uint c_coeffs[3] = { 1u, 2u, 3u };
__device__ float g_values[2];
__global__ void array_sizeof(uint* out) {
  uint local[4];
  __shared__ float tile[5];
  if (threadIdx.x == 0) {
    out[0] = sizeof(local);
    out[1] = alignof(local);
    out[2] = sizeof(tile);
    out[3] = sizeof(c_coeffs);
    out[4] = sizeof(g_values);
    out[5] = sizeof(local[0]);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(6) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Uint32Array(6) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-sizeof");
    expect([...result.buffers.out as Uint32Array]).toEqual([16, 4, 20, 12, 8, 4]);
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([16, 4, 20, 12, 8, 4]);
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("out[0u] = 16u;");
    expect(compiled.wgsl).toContain("out[2u] = 20u;");
    expect(compiled.wgsl).toContain("out[3u] = 12u;");
    expect(compiled.wgsl).toContain("out[4u] = 8u;");
  });

  it("accepts common C integer aliases as CUDA-lite i32/u32 scalars", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void integerAliases(int32_t *signedOut, uint32_t *unsignedOut, signed int n) {
  int idx = threadIdx.x;
  long long signedWide = (long long)idx - 2;
  long long int signedWideInt = signedWide + 1;
  signed short small = (signed short)n;
  ptrdiff_t stride = (ptrdiff_t)idx;
  unsigned long long unsignedWide = (unsigned long long)n + (uint64_t)idx;
  unsigned long long int unsignedWideInt = unsignedWide + 1u;
  uintptr_t ptrValue = (uintptr_t)unsignedWide;
  uint32_t bytes = (uint32_t)sizeof(long);
  int64_t signedAlias = (int64_t)signedWideInt + (int32_t)small;
  if (idx < 2) {
    signedOut[idx] = signedAlias + stride;
    unsignedOut[idx] = unsignedWideInt + ptrValue + bytes;
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
    expect(compiled.wgsl).toContain("var stride: i32");
    expect(compiled.wgsl).toContain("var unsignedWide: u32");
    expect(compiled.wgsl).toContain("var ptrValue: u32");
    expect(compiled.wgsl).toContain("var bytes: u32 = u32(4)");
    expect([...result.buffers.signedOut as Int32Array]).toEqual([4, 6]);
    expect([...result.buffers.unsignedOut as Uint32Array]).toEqual([15, 17]);
  });

  it("accepts CUDA opaque/index aliases and volatile qualifiers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void cudaAliasKernel(volatile size_type *out, curandState *state, curandStateSobol64 *sobol, curandDirectionVectors64_t direction, CUtensorMap map, cudaGraphConditionalHandle handle) {
  volatile size_type idx = threadIdx.x;
  if (idx < 1) {
    out[0] = idx + map + handle + state[0] + sobol[0] + direction;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          out: new Uint32Array(1),
          state: new Uint32Array([5]),
          sobol: new Uint32Array([13]),
        },
        scalars: { direction: 17, map: 7, handle: 11 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(backendIr(compiled).params.map((param) => [param.name, param.valueType])).toContainEqual(["map", "uint"]);
    expect(backendIr(compiled).params.map((param) => [param.name, param.valueType])).toContainEqual(["direction", "uint"]);
    expect([...result.buffers.out as Uint32Array]).toEqual([53]);
  });

  it("allows C++ block scopes to shadow outer CUDA symbols", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void scopedShadow(const float *wte, float *out) {
  if (threadIdx.x < 1) {
    float wte = 3.0f;
    out[0] = wte;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { wte: new Float32Array([1]), out: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Float32Array]).toEqual([3]);
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

    const modeledLocalPointer = compileCudaLiteKernel(`
__global__ void ok(float* x, float* out) {
  int i = threadIdx.x;
  float* y = x + i;
  if (i < 1) { out[0] = y[0]; }
}`);
    expect(modeledLocalPointer.loweringPlan.canDirectLowerToWgsl).toBe(true);

    const conditionalLocalPointer = compileCudaLiteKernel(`
__global__ void conditional_local_ptr(const float* a, const float* b, float* out, int flag) {
  int i = threadIdx.x;
  const float* p = flag ? a + i : b + i;
  if (i < 2) { out[i] = p[0]; }
}`, { workgroupSize: [2, 1, 1] });
    const conditionalResult = runCompiledKernelReference(
      conditionalLocalPointer,
      {
        buffers: {
          a: new Float32Array([1, 2]),
          b: new Float32Array([3, 4]),
          out: new Float32Array(2),
        },
        scalars: { flag: 0 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    expect(conditionalLocalPointer.loweringPlan.canDirectLowerToWgsl).toBe(true);
    expect([...conditionalResult.buffers.out as Float32Array]).toEqual([3, 4]);

    const mutableLocalPointer = compileCudaLiteKernel(`
__global__ void mutable_local_ptr(const float* a, float* b, float* out) {
  int i = threadIdx.x;
  const float* p = a + i;
  if (i < 1) {
    out[0] = p[0];
    p = b + i;
    out[1] = p[0];
  }
}`, { workgroupSize: [1, 1, 1] });
    const mutableLocalPointerResult = runCompiledKernelReference(
      mutableLocalPointer,
      {
        buffers: {
          a: new Float32Array([2]),
          b: new Float32Array([5]),
          out: new Float32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect([...mutableLocalPointerResult.buffers.out as Float32Array]).toEqual([2, 5]);
    expect(canRunCompiledKernelSemanticReference(mutableLocalPointer)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(mutableLocalPointer.kernelIr)).toBe(true);
    expect(mutableLocalPointer.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(mutableLocalPointer.wgsl).not.toMatch(/var p_\d+_buffer: u32/u);

    const alignedPointer = compileCudaLiteKernel(`
__global__ void aligned(float* x, float* out) {
  float* y = (float*)__builtin_assume_aligned(x + threadIdx.x, 16);
  if (threadIdx.x < 1) { out[0] = y[0]; }
}`);
    expect(alignedPointer.loweringPlan.canDirectLowerToWgsl).toBe(true);

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
    expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(true);
  });

  it("classifies CUDA compatibility gaps by semantic feature", () => {
    const unsupported = analyzeCudaLite(parseCudaLite(`
__global__ void unsupported(float* x) {
  if (threadIdx.x < 1) { atomicCAS(&x[0], 0, 1); }
}`));
    const plan = createCudaLoweringPlan(unsupported.diagnostics);

    expect(plan.canDirectLowerToWgsl).toBe(false);
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
    expect(describeCudaDiagnostic({
      code: "unsupported-inline-asm",
      message: "only fma.rn.f32, laneid, and bfind.u32 inline PTX are supported in CUDA-lite v0",
    })).toMatchObject({ family: "subgroup", referenceRuns: false });
    expect(describeCudaDiagnostic({
      code: "unsupported-f64",
      message: "double requires f64Mode",
    })).toMatchObject({ family: "feature", referenceRuns: false });
    expect(describeCudaDiagnostic({
      code: "f64-lowered-to-f32",
      message: "double is lowered to f32",
    })).toMatchObject({ family: "feature", gpuRuns: true, referenceRuns: true });
    expect(describeCudaDiagnostic({
      code: "unsupported-cpp-object-model",
      message: "C++ object model declarations require modeled constructors, member calls, and object lifetime before CUDA-lite lowering",
    })).toMatchObject({ family: "frontend", gpuRuns: false, referenceRuns: false });
    expect(describeCudaDiagnostic({
      code: "unsupported-cute-object",
      message: "CuTe C++ object declarations require a modeled tensor/tile object graph before CUDA-lite lowering",
    })).toMatchObject({ family: "frontend", gpuRuns: false, referenceRuns: false });
    expect(describeCudaDiagnostic({
      code: "unsupported-wgmma-tma",
      message: "WGMMA/TMA object pipeline declarations require a modeled async tensor-core pipeline before CUDA-lite lowering",
    })).toMatchObject({ family: "subgroup", gpuRuns: false, referenceRuns: false });
    expect(describeCudaDiagnostic({
      code: "unsupported-dependent-carrier-param",
      message: "dependent C++ carrier parameters require concrete source/context normalization before CUDA-lite lowering",
    })).toMatchObject({ family: "frontend", gpuRuns: false, referenceRuns: false });
  });

  it("keeps every emitted compiler diagnostic code in the compatibility registry", () => {
    const emitted = collectEmittedDiagnosticCodes(path.join(packageRoot, "src"));
    const registered = new Set(getCudaFeatureRegistry().map((feature) => feature.code));
    const missing = [...emitted].filter((code) => !registered.has(code)).sort();

    expect(missing).toEqual([]);
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
    const files = [
      "analyzer.ts",
      "reference.ts",
      "runtime_plan.ts",
      "dynamic_launch.ts",
      "peer_copy.ts",
      "webgpu_orchestration.ts",
      "wgsl.ts",
    ];
    const missing = files.flatMap((file) => {
      const source = compilerSourceText(file);
      return modeledRuntimeCalls
        .filter((call) => !source.includes(`"${call}"`))
        .map((call) => `${file}:${call}`);
    });

    expect(missing).toEqual([]);
  });

  it("reports unsupported C++ CUDA object-model gaps with stable diagnostic codes", () => {
    expectParseDiagnosticCode(`
__global__ void cute(float* out) {
  TiledCopy tiled_copy;
  if (threadIdx.x < 1) { out[0] = 0.0f; }
}`, "unsupported-cute-object");

    expectParseDiagnosticCode(`
__global__ void wgmma(float* out) {
  WgmmaSMem<128, 64>& smem = *reinterpret_cast<WgmmaSMem<128, 64>*>(out);
  if (threadIdx.x < 1) { out[0] = 0.0f; }
}`, "unsupported-wgmma-tma");

    expectParseDiagnosticCode(`
__global__ void carrier(float* out, typename GEMM_Traits::Arguments args) {
  if (threadIdx.x < 1) { out[0] = 0.0f; }
}`, "unsupported-dependent-carrier-param");

    expectParseDiagnosticCode(`
__global__ void opaque(Container<int> **g_container) {
  *g_container = new Vector<int>(4);
}`, "unsupported-cpp-object-model");

    expectParseDiagnosticCode(`
__global__ void stacky(float* out) {
  global_stack<int, 4, 32> stack(out, threadIdx.x);
}`, "unsupported-cpp-object-model");
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
    expect(backendIr(compiled).params.map((param) => [param.name, param.valueType])).toContainEqual(["n", "uint"]);
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

  it("accepts empty CUDA statement bodies", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void emptyWhile(int *latch, int *out) {
  while (latch[0] < 1)
    ;
  out[0] = 7;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { latch: new Int32Array([1]), out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("while ((u32(latch[0u]) < 1u))");
    expect([...result.buffers.out as Int32Array]).toEqual([7]);
  });

  it("passes braced CUDA vector initializer arguments to device helpers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float2 pick_max(float2 a, float2 b) {
  return a.x > b.x ? a : b;
}
__global__ void vectorArg(float *x, float *out) {
  float2 v;
  v.x = 1.0f;
  v.y = 2.0f;
  float2 best = pick_max(v, { x[0], 4.0f });
  out[0] = best.x + best.y;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([3]), out: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("pick_max(v, vec2<f32>(x[0], 4.0)");
    expect([...result.buffers.out as Float32Array]).toEqual([7]);
  });

  it("promotes mixed integer and float scalar expressions for WGSL", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void mixed(float *out, int i, int n) {
  if (threadIdx.x < 1) {
    out[0] = 1.0f / powf(10000.0f, 2 * i / (n * 2.0f));
  }
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("pow(10000.0, (f32((2 * bg_uniforms.i)) / (f32(bg_uniforms.n) * 2.0)))");
  });

  it("requires explicit f64 compatibility mode before lowering double to f32", () => {
    const source = `
__global__ void doubleGap(double *out, double a) {
  double sum = a + 1.25;
  out[0] = sum;
}`;
    expect(() => compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] })).toThrow(/unsupported-f64/u);

    const compiled = compileCudaLiteKernel(source, {
      f64Mode: "f32",
      workgroupSize: [1, 1, 1],
    });
    expect(compiled.diagnostics).toContainEqual(expect.objectContaining({
      code: "f64-lowered-to-f32",
      severity: "warning",
    }));
    expect(compiled.wgsl).toContain("var<storage, read_write> out: array<f32>;");
    expect(compiled.wgsl).toContain("a: f32");

    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) }, scalars: { a: 2 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect([...result.buffers.out as Float32Array]).toEqual([3.25]);
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

  it("parses adjacent C string literals in stdout-only calls", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void hello() {
  printf("hello %d "
         "world\\n", threadIdx.x);
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("printf omitted");
  });

  it("accepts printf pointer and local array arguments as no-op debug output", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void hello(char *name) {
  char buffer[8];
  buffer[0] = 'b';
  buffer[1] = '\\0';
  printf("%s %s\\n", buffer, name);
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("printf omitted");
  });

  it("parses scalar C++ aliases and brace scalar constructors", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void scalarCppIntake(float *out, half *halfOut, std::size_t n) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t total = n;
  if (idx < total) {
    out[idx] = float(idx);
    halfOut[idx] = __half{float(idx) / 32.0f};
  }
}`, {
      features: { "shader-f16": true },
      workgroupSize: [2, 1, 1],
    });

    expect(compiled.wgsl).toContain("var idx: i32");
    expect(compiled.wgsl).toContain("var total: u32");
    expect(compiled.wgsl).toContain("halfOut[u32(idx)] = f16");
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

    expect(backendIr(compiled).functions.map((fn) => fn.name)).toEqual(["addOne"]);
    expect(compiled.wgsl).toContain("fn addOne(value: f32");
    expect(compiled.wgsl).toContain("return (value + 1.0);");
    expect([...result.buffers.x as Float32Array]).toEqual([3]);
  });

  it("does not emit unreachable device helpers into WGSL", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint64_t unused_desc(uint *ptr) {
  uint64_t desc = 0;
  desc |= 1llu << 62;
  return desc;
}
__device__ float addOne(float value) {
  return value + 1.0f;
}
__global__ void helperKernel(float *x) {
  if (threadIdx.x < 1) { x[0] = addOne(x[0]); }
}`, { workgroupSize: [1, 1, 1] });

    expect(backendIr(compiled).functions.map((fn) => fn.name)).toEqual(["addOne"]);
    expect(compiled.wgsl).toContain("fn addOne(value: f32");
    expect(compiled.wgsl).not.toContain("unused_desc");
    expect(compiled.wgsl).not.toContain("<< 62");
  });

  it("does not require shader-f16 for unreachable half helpers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ half2 unused_half2(half2 value) {
  return value;
}
__device__ float addOne(float value) {
  return value + 1.0f;
}
__global__ void helperKernel(float *x) {
  if (threadIdx.x < 1) { x[0] = addOne(x[0]); }
}`, { workgroupSize: [1, 1, 1] });

    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect(compiled.wgsl).toContain("fn addOne(value: f32");
    expect(compiled.wgsl).not.toContain("unused_half2");
  });

  it("does not emit shared declarations from unreachable helpers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void unused_shared_helper(float* out) {
  __shared__ float unusedScratch[1];
  unusedScratch[0] = 9.0f;
  out[0] = unusedScratch[0];
}
__device__ float addOne(float value) {
  return value + 1.0f;
}
__global__ void helperKernel(float *x) {
  if (threadIdx.x < 1) { x[0] = addOne(x[0]); }
}`, { workgroupSize: [1, 1, 1] });

    expect(backendIr(compiled).sharedDeclarations.map((shared) => shared.name)).not.toContain("unusedScratch");
    expect(compiled.wgsl).not.toContain("unusedScratch");
  });

  it("resolves overloaded __device__ helpers by arity", () => {
    const compiled = compileCudaLiteKernel(`
__device__ int pick(int value) {
  return value + 1;
}
__device__ int pick(int left, int right) {
  return left + right;
}
__global__ void overloadKernel(int *out) {
  if (threadIdx.x < 1) {
    out[0] = pick(4);
    out[1] = pick(4, 5);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(backendIr(compiled).functions.map((fn) => fn.name)).toEqual(["pick", "pick"]);
    expect(compiled.wgsl).toContain("fn pick__bg_overload_0(");
    expect(compiled.wgsl).toContain("fn pick__bg_overload_1(");
    expect(compiled.wgsl).toContain("out[0] = i32(pick__bg_overload_0(4");
    expect(compiled.wgsl).toContain("out[1] = i32(pick__bg_overload_1(4, 5");
    expect([...result.buffers.out as Int32Array]).toEqual([5, 9]);
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
    expect(compiled.wgsl).toContain("bg_warp_shuffle_down_float_32(val, 16u, 32u, local_id)");
    expect(compiled.wgsl).toContain("bg_inline_warpReduceSum_");
    expect(compiled.wgsl).not.toContain("val = warpReduceSum(val, local_id, workgroup_id, num_workgroups)");
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

    expect(compiled.wgsl).toContain("out[0] = i32(bg_warp_reduce_sum_int_32(bg_uniforms.value, 32u, local_id))");
    expect([...result.buffers.out as Int32Array]).toEqual([7]);
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
    expect(compiled.wgsl).toContain("subgroupBallot");
    expect(compiled.wgsl).toContain("bg_warp_reduce_sum_uint_32(input[0], 32u, local_id)");
    expect(compiled.wgsl).toContain("countOneBits");
    expect(backendIr(compiled).requiredFeatures).toContain("subgroups");
    expect([...result.buffers.out as Uint32Array]).toEqual([1, 0, 1, 1, 7]);
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
    expect(compiled.wgsl).toContain("bg_warp_shuffle_down_float_16(val, u32(offset), 16u, local_id)");
    expect(compiled.wgsl).toContain("(i32(local_id.x) % 16)");
    expect(compiled.wgsl).toContain("workgroupBarrier();");
    expect([...result.buffers.output as Float32Array]).toEqual([16]);
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

    expect(compiled.wgsl).toContain("fn block_rank(local_id: vec3<u32>");
    expect(compiled.wgsl).toContain("fn tile_rank(local_id: vec3<u32>");
    expect([...result.buffers.out as Int32Array]).toEqual([0, 2]);
  });

  it("parses bare CUDA thread_group helper parameters as block handles", () => {
    const compiled = compileCudaLiteKernel(`
__device__ int sumReduction(thread_group g, int *scratch, int value) {
  int lane = g.thread_rank();
  scratch[lane] = value;
  g.sync();
  return scratch[lane] + g.size();
}
__global__ void bareThreadGroup(int *out) {
  __shared__ int scratch[4];
  cg::thread_block block = cg::this_thread_block();
  auto tile2 = cg::tiled_partition<2>(block);
  out[threadIdx.x] = sumReduction(block, scratch, threadIdx.x);
  out[threadIdx.x + 4] = sumReduction(tile2, scratch, threadIdx.x);
}`, {
      workgroupSize: [4, 1, 1],
    });

    expect(compiled.wgsl).toContain("fn sumReduction");
    expect(compiled.wgsl).toContain("g_tile_size_arg");
    expect(compiled.wgsl).toContain("let g_tile_size: u32 = g_tile_size_arg");
    expect(compiled.wgsl).toContain("let bg_inline_sumReduction_");
    expect(compiled.wgsl).toContain("_g_tile_size: u32 = 4u");
    expect(compiled.wgsl).toContain("_g_tile_size: u32 = 2u");
    expect(compiled.wgsl).toContain("workgroupBarrier();");
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

    expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(false);
    expect(compiled.diagnostics).toContainEqual(expect.objectContaining({
      code: "unsupported-cooperative-groups",
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

    expect([...result.buffers.dst as Float32Array]).toEqual([2.5, 3.5, 2.5, 2.5]);
    expect(runtimePlan.operations.map((operation) => operation.kind)).toEqual([
      "device-sync",
      "device-sync",
      "runtime-copy",
      "runtime-copy",
      "runtime-copy",
      "runtime-copy",
      "runtime-copy",
      "runtime-copy",
      "runtime-copy",
      "device-sync",
      "device-sync",
      "device-sync",
      "device-sync",
      "device-sync",
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
    const detachedPlan = createCudaRuntimePlan({ ...compiled });

    expect(plan.operations.map((operation) => operation.kind)).toEqual([
      "device-launch",
      "device-sync",
      "runtime-copy",
      "grid-sync",
    ]);
    expect(detachedPlan.operations.map((operation) => operation.kind)).toEqual(plan.operations.map((operation) => operation.kind));
    expect(plan.canRunSingleDispatchWebGpu).toBe(false);
    expect(plan.referenceAvailable).toBe(true);
  });

  it("materializes WebGPU input buffers from public compile stages on detached results", () => {
    const compiled = compileCudaLiteKernel(`
__constant__ float coeffs[2];
__device__ float global_state[1];

__global__ void selected(float *out, uint n) {
  if (threadIdx.x < n) {
    out[threadIdx.x] = coeffs[threadIdx.x] + global_state[0];
  }
}`, { kernelName: "selected", workgroupSize: [2, 1, 1] });
    const detached = { ...compiled };
    const input = {
      buffers: {
        out: new Float32Array([0, 0]),
      },
      scalars: {
        n: 2,
      },
      constants: {
        coeffs: new Float32Array([3, 4]),
      },
    };

    expect(backendIr(detached).params.map((param) => param.name)).toEqual(["out", "n"]);
    expect(Object.keys(constantBufferInputs(detached, input))).toEqual(["coeffs"]);
    expect(Object.keys(deviceGlobalBufferInputs(detached, input))).toEqual(["global_state"]);
    expect(cudaWebGpuDefaultReadbackNames(detached)).toEqual(["out", "global_state"]);
    expect(packCudaWebGpuUniformParams(detached, input).byteLength).toBeGreaterThanOrEqual(16);
    expect([...runCompiledKernelReference(detached, input, {
      gridDim: [1, 1, 1],
      blockDim: [2, 1, 1],
    }).buffers.out as Float32Array]).toEqual([3, 4]);
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
    cudaMemcpyPeer(dst, 1, src, 0, sizeof(float));
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
      expect(peerPlan.steps.map((step) => step.program.name)).toEqual(["peerCopy", "bg_peer_copy_float", "bg_peer_copy_float"]);
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

    expect(createCudaLaunchValidationDiagnostics(badGrid, backendIr(compiled).workgroupSize)).toContainEqual(expect.objectContaining({
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
    expect(() => validateCudaKernelLaunch(badBlock, backendIr(compiled).workgroupSize)).toThrow(CudaLiteCompilerError);
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
    const plan = createCudaGridSyncPhasePlan(compiled);
    const detachedPlan = createCudaGridSyncPhasePlan({ ...compiled });

    expect(plan.supported).toBe(true);
    expect(detachedPlan.supported).toBe(true);
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

    expect(compiled.wgsl).toContain("bg_warp_shuffle_up_float_8(val, 1u, 8u, local_id)");
    expect(compiled.wgsl).toContain("bg_warp_shuffle_xor_float_8(val, 2u, 8u, local_id)");
    expect(compiled.wgsl).toContain("(i32(local_id.x + local_id.y * 8u) % 8)");
    expect(compiled.wgsl).toContain("workgroupBarrier();");
  });

  it("supports coalesced-group ballot, shfl, and popcount primitives", () => {
    const compiled = compileCudaLiteKernel(`
namespace cg = cooperative_groups;
__global__ void coalescedVote(uint *flags, uint *out) {
  cg::coalesced_group group = cg::coalesced_threads();
  uint vote = group.ballot(flags[threadIdx.x]);
  uint first = group.shfl(threadIdx.x, 0);
  out[threadIdx.x] = __popc(vote) + first;
}`, {
      features: { subgroups: true },
      workgroupSize: [1, 1, 1],
    });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { flags: new Uint32Array([1]), out: new Uint32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("subgroupBallot");
    expect(compiled.wgsl).toContain("bg_warp_shuffle_sync_int_1");
    expect(compiled.wgsl).toContain("countOneBits");
    expect(backendIr(compiled).requiredFeatures).toContain("subgroups");
    expect([...result.buffers.out as Uint32Array]).toEqual([1]);
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

    expect(compiled.wgsl).toContain("bg_warp_reduce_sum_float_8(value, 8u, local_id)");
    expect(compiled.wgsl).toContain("workgroupBarrier();");
    expect([...result.buffers.output as Float32Array]).toEqual([8]);
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

    expect(compiled.wgsl).toContain("reduce_n(value, 4u");
    expect(compiled.wgsl).toContain("bg_warp_reduce_sum_int_4(value, tile_tile_size, local_id)");
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

  it("lowers prefix and postfix updates to WGSL assignment statements", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void updateLoop(int *out) {
  int acc = 0;
  for (int i = 0, j = 3; i < 3; ++i, j--) {
    acc += i + j;
  }
  out[0] = acc;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("i = (i + 1);");
    expect(compiled.wgsl).toContain("j = (j - 1);");
    expect(compiled.wgsl).not.toMatch(/\\+\\+|--/u);
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
    expect(compiled.wgsl).toContain("var bg_active: bool = (flags[idx] != 0u);");
    expect([...result.buffers.out as Int32Array]).toEqual([1, 0]);
    expect([...result.buffers.flags as Uint32Array]).toEqual([1, 0, 0, 1]);
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
      rint(value) +
      rintf(value) +
      nearbyint(value) +
      nearbyintf(value) +
      sinf(value) +
      sinpif(value) +
      cosf(value) +
      cospif(value) +
      tanf(value) +
      asinf(__saturatef(value)) +
      acosf(__saturatef(value)) +
      atanf(value) +
      asinhf(value) +
      acoshf(fabsf(value) + 1.5f) +
      atanhf(__saturatef(value) * 0.5f) +
      tanhf(value) +
      __tanhf(value) +
      sinhf(value) +
      coshf(value) +
      sqrt(fabsf(value)) +
      sqrtf(fabsf(value)) +
      __fsqrt_rn(fabsf(value)) +
      rsqrt(fabsf(value) + 2.0f) +
      rsqrtf(fabsf(value) + 1.0f) +
      __frsqrt_rn(fabsf(value) + 4.0f) +
      __frcp_rn(value + 3.0f) +
      __saturatef(value) +
      __expf(value) +
      exp2f(value) +
      __exp2f(value) +
      exp10f(value) +
      __exp10f(value) +
      expm1f(value) +
      erfcxf(value) +
      normcdff(value) +
      tgammaf(fabsf(value) + 1.0f) +
      lgammaf(fabsf(value) + 1.0f) +
      __logf(fabsf(value) + 1.0f) +
      log2f(fabsf(value) + 1.0f) +
      __log2f(fabsf(value) + 1.0f) +
      log10f(fabsf(value) + 1.0f) +
      __log10f(fabsf(value) + 1.0f) +
      log1pf(fabsf(value)) +
      cbrtf(value) +
      rcbrtf(value + 2.0f) +
      powf(fabsf(value), 2.0f) +
      __powf(fabsf(value), 3.0f) +
      atan2f(value, 2.0f) +
      hypotf(value, 2.0f) +
      rhypotf(value, 2.0f) +
      norm3df(value, 2.0f, -3.0f) +
      norm4df(value, 2.0f, -3.0f, 4.0f) +
      rnorm3df(value, 2.0f, -3.0f) +
      rnorm4df(value, 2.0f, -3.0f, 4.0f) +
      ldexpf(value, 2) +
      scalblnf(value, 3) +
      scalbnf(value, 1) +
      fmodf(value, 2.0f) +
      remainderf(value, 2.0f) +
      logbf(fabsf(value) + 1.0f) +
      ilogbf(fabsf(value) + 1.0f) +
      fdimf(value, -0.5f) +
      copysignf(value, -2.0f) +
      fdividef(value, 4.0f) +
      (signbitf(value) ? 1.0f : 0.0f) +
      (isgreater(value, -2.0f) ? 1.0f : 0.0f) +
      (isgreaterequal(value, value) ? 1.0f : 0.0f) +
      (isless(value, 2.0f) ? 1.0f : 0.0f) +
      (islessequal(value, value) ? 1.0f : 0.0f) +
      (islessgreater(value, 0.0f) ? 1.0f : 0.0f) +
      fminf(value, 1.0f) +
      fmaxf(value, -1.0f) +
      (isfinite(value) ? 1.0f : 0.0f) +
      (isnormal(value) ? 1.0f : 0.0f) +
      __fdividef(value, 2.0f) +
      __fadd_rn(value, 2.0f) +
      __fsub_rn(value, 2.0f) +
      __fmul_rn(value, 2.0f) +
      __fdiv_rn(value, 2.0f) +
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
    expect(compiled.wgsl).toContain("select(floor(abs(value) + 0.5), -floor(abs(value) + 0.5), (value < 0.0))");
    expect(compiled.wgsl).toContain("fn bg_semantic_round_even_f32(");
    expect(compiled.wgsl).toContain("bg_semantic_round_even_f32(value)");
    expect(compiled.wgsl).toContain("sin(value)");
    expect(compiled.wgsl).toContain("sin(3.141592653589793 * value)");
    expect(compiled.wgsl).toContain("cos(value)");
    expect(compiled.wgsl).toContain("cos(3.141592653589793 * value)");
    expect(compiled.wgsl).toContain("tan(value)");
    expect(compiled.wgsl).toContain("asin(clamp(value, 0.0, 1.0))");
    expect(compiled.wgsl).toContain("acos(clamp(value, 0.0, 1.0))");
    expect(compiled.wgsl).toContain("atan(value)");
    expect(compiled.wgsl).toContain("log(value + sqrt((value * value) + 1.0))");
    expect(compiled.wgsl).toContain("log((abs(value) + 1.5) + sqrt(((abs(value) + 1.5) * (abs(value) + 1.5)) - 1.0))");
    expect(compiled.wgsl).toContain("(0.5 * log((1.0 + (clamp(value, 0.0, 1.0) * 0.5)) / (1.0 - (clamp(value, 0.0, 1.0) * 0.5))))");
    expect(compiled.wgsl).toContain("atan2(value, 2.0)");
    expect(compiled.wgsl).toContain("tanh(value)");
    expect(compiled.wgsl).toContain("(0.5 * (exp(value) - exp(-value)))");
    expect(compiled.wgsl).toContain("(0.5 * (exp(value) + exp(-value)))");
    expect(compiled.wgsl).toContain("sqrt(abs(value))");
    expect(compiled.wgsl).toContain("inverseSqrt((abs(value) + 4.0))");
    expect(compiled.wgsl).toContain("inverseSqrt((abs(value) + 2.0))");
    expect(compiled.wgsl).toContain("inverseSqrt((abs(value) + 1.0))");
    expect(compiled.wgsl).toContain("(1.0 / (value + 3.0))");
    expect(compiled.wgsl).toContain("clamp(value, 0.0, 1.0)");
    expect(compiled.wgsl).toContain("exp2(value)");
    expect(compiled.wgsl).toContain("pow(10.0, value)");
    expect(compiled.wgsl).toContain("(exp(value) - 1.0)");
    expect(compiled.wgsl).toContain("(exp(value * value) * (1.0 - ");
    expect(compiled.wgsl).toContain("(0.5 * (1.0 + ");
    expect(compiled.wgsl).toContain("fn bg_semantic_tgamma_f32(");
    expect(compiled.wgsl).toContain("bg_semantic_tgamma_f32((abs(value) + 1.0))");
    expect(compiled.wgsl).toContain("bg_semantic_lgamma_f32((abs(value) + 1.0))");
    expect(compiled.wgsl).toContain("log2((abs(value) + 1.0))");
    expect(compiled.wgsl).toContain("(log((abs(value) + 1.0)) / 2.302585092994046)");
    expect(compiled.wgsl).toContain("log(1.0 + abs(value))");
    expect(compiled.wgsl).toContain("select(pow(abs(value), 0.3333333333333333), -pow(abs(value), 0.3333333333333333), (value < 0.0))");
    expect(compiled.wgsl).toContain("pow(abs(value), 2.0)");
    expect(compiled.wgsl).toContain("pow(abs(value), 3.0)");
    expect(compiled.wgsl).toContain("sqrt((value * value) + (2.0 * 2.0))");
    expect(compiled.wgsl).toContain("(1.0 / sqrt((value * value) + (2.0 * 2.0)))");
    expect(compiled.wgsl).toContain("sqrt((value * value) + (2.0 * 2.0) + (-(3.0) * -(3.0)))");
    expect(compiled.wgsl).toContain("sqrt((value * value) + (2.0 * 2.0) + (-(3.0) * -(3.0)) + (4.0 * 4.0))");
    expect(compiled.wgsl).toContain("(value * exp2(f32(2)))");
    expect(compiled.wgsl).toContain("(value * exp2(f32(3)))");
    expect(compiled.wgsl).toContain("(value * exp2(f32(1)))");
    expect(compiled.wgsl).toContain("min(value, 1.0)");
    expect(compiled.wgsl).toContain("max(value, -(1.0))");
    expect(compiled.wgsl).toContain("(value - trunc(value / 2.0) * 2.0)");
    expect(compiled.wgsl).toContain("bg_semantic_remainder_f32(value, 2.0)");
    expect(compiled.wgsl).toContain("bg_semantic_logb_f32((abs(value) + 1.0))");
    expect(compiled.wgsl).toContain("bg_semantic_ilogb_i32((abs(value) + 1.0))");
    expect(compiled.wgsl).toContain("max((value - -(0.5)), 0.0)");
    expect(compiled.wgsl).toContain("select(abs(value), -abs(value), ((bitcast<u32>(-(2.0)) & 0x80000000u) != 0u))");
    expect(compiled.wgsl).toContain("(value / 4.0)");
    expect(compiled.wgsl).toContain("((bitcast<u32>(value) & 0x80000000u) != 0u)");
    expect(compiled.wgsl).toContain("!((value) != (value) || (-(2.0)) != (-(2.0))) && ((value) > (-(2.0)))");
    expect(compiled.wgsl).toContain("!((value) != (value) || (value) != (value)) && ((value) >= (value))");
    expect(compiled.wgsl).toContain("!((value) != (value) || (2.0) != (2.0)) && ((value) < (2.0))");
    expect(compiled.wgsl).toContain("!((value) != (value) || (value) != (value)) && ((value) <= (value))");
    expect(compiled.wgsl).toContain("!((value) != (value) || (0.0) != (0.0)) && (((value) < (0.0)) || ((value) > (0.0)))");
    expect(compiled.wgsl).toContain("abs(value) <= 3.4028234663852886e38");
    expect(compiled.wgsl).toContain("abs(value) >= 1.1754943508222875e-38");
    expect(compiled.wgsl).toContain("(value + 2.0)");
    expect(compiled.wgsl).toContain("(value - 2.0)");
    expect(compiled.wgsl).toContain("(value * 2.0)");
    expect(compiled.wgsl).toContain("fma(value, 2.0, 1.0)");
    expect(compiled.wgsl).toContain("fma(value, -(1.0), 0.5)");
    const expected = [...input].map((value) =>
      Math.abs(value) +
      Math.floor(value) +
      Math.ceil(value) +
      Math.trunc(value) +
      roundAway(value) +
      roundEven(value) +
      roundEven(value) +
      roundEven(value) +
      roundEven(value) +
      Math.sin(value) +
      Math.sin(Math.PI * value) +
      Math.cos(value) +
      Math.cos(Math.PI * value) +
      Math.tan(value) +
      Math.asin(Math.min(1, Math.max(0, value))) +
      Math.acos(Math.min(1, Math.max(0, value))) +
      Math.atan(value) +
      Math.asinh(value) +
      Math.acosh(Math.abs(value) + 1.5) +
      Math.atanh(Math.min(1, Math.max(0, value)) * 0.5) +
      Math.tanh(value) +
      Math.tanh(value) +
      Math.sinh(value) +
      Math.cosh(value) +
      Math.sqrt(Math.abs(value)) +
      Math.sqrt(Math.abs(value)) +
      Math.sqrt(Math.abs(value)) +
      (1 / Math.sqrt(Math.abs(value) + 2)) +
      (1 / Math.sqrt(Math.abs(value) + 1)) +
      (1 / Math.sqrt(Math.abs(value) + 4)) +
      (1 / (value + 3)) +
      Math.min(1, Math.max(0, value)) +
      Math.exp(value) +
      (2 ** value) +
      (2 ** value) +
      (10 ** value) +
      (10 ** value) +
      Math.expm1(value) +
      (Math.exp(value * value) * (1 - erfApprox(value))) +
      (0.5 * (1 + erfApprox(value * Math.SQRT1_2))) +
      gammaApprox(Math.abs(value) + 1) +
      Math.log(Math.abs(gammaApprox(Math.abs(value) + 1))) +
      Math.log(Math.abs(value) + 1) +
      Math.log2(Math.abs(value) + 1) +
      Math.log2(Math.abs(value) + 1) +
      Math.log10(Math.abs(value) + 1) +
      Math.log10(Math.abs(value) + 1) +
      Math.log1p(Math.abs(value)) +
      Math.cbrt(value) +
      (1 / Math.cbrt(value + 2)) +
      Math.pow(Math.abs(value), 2) +
      Math.pow(Math.abs(value), 3) +
      Math.atan2(value, 2) +
      Math.hypot(value, 2) +
      (1 / Math.hypot(value, 2)) +
      Math.hypot(value, 2, -3) +
      Math.hypot(value, 2, -3, 4) +
      (1 / Math.hypot(value, 2, -3)) +
      (1 / Math.hypot(value, 2, -3, 4)) +
      (value * 4) +
      (value * 8) +
      (value * 2) +
      (value - Math.trunc(value / 2) * 2) +
      (value - roundEven(value / 2) * 2) +
      Math.floor(Math.log2(Math.abs(value) + 1)) +
      Math.floor(Math.log2(Math.abs(value) + 1)) +
      Math.max(value - -0.5, 0) +
      -Math.abs(value) +
      (value / 4) +
      (value < 0 || Object.is(value, -0) ? 1 : 0) +
      (!Number.isNaN(value) && value > -2 ? 1 : 0) +
      (!Number.isNaN(value) && value >= value ? 1 : 0) +
      (!Number.isNaN(value) && value < 2 ? 1 : 0) +
      (!Number.isNaN(value) && value <= value ? 1 : 0) +
      (!Number.isNaN(value) && value !== 0 ? 1 : 0) +
      Math.min(value, 1) +
      Math.max(value, -1) +
      1 +
      1 +
      (value / 2) +
      (value + 2) +
      (value - 2) +
      (value * 2) +
      (value / 2) +
      (value * 2 + 1) +
      (value * -1 + 0.5)
    );
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
  });

  it("lowers CUDA special float intrinsics through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_float_ops(float *x, float *out) {
  int idx = threadIdx.x;
  float value = x[idx];
  out[idx] =
    __saturatef(value) +
    __powf(fabsf(value), 3.0f) +
    fdividef(value, 4.0f) +
    __fdividef(value, 2.0f) +
    __fadd_rn(value, 2.0f) +
    __fsub_rn(value, 2.0f) +
    __fmul_rn(value, 2.0f) +
    __fdiv_rn(value, 2.0f) +
    copysignf(value, -0.0f);
}`, { workgroupSize: [2, 1, 1] });
    const input = new Float32Array([-1.25, 0.6]);
    const runInput = { buffers: { x: input, out: new Float32Array(2) } };
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      runInput,
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const expected = [...input].map((value) =>
      Math.min(1, Math.max(0, value)) +
      Math.pow(Math.abs(value), 3) +
      (value / 4) +
      (value / 2) +
      (value + 2) +
      (value - 2) +
      (value * 2) +
      (value / 2) -
      Math.abs(value)
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("clamp(value, 0.0, 1.0)");
    expect(compiled.wgsl).toContain("pow(abs(value), 3.0)");
    expect(compiled.wgsl).toContain("(value / 4.0)");
    expect(compiled.wgsl).toContain("(value + 2.0)");
    expect(compiled.wgsl).toContain("select(abs(value), -abs(value)");
    expect([...semanticResult.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...semanticResult.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
  });

  it("lowers CUDA elementary math intrinsics through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_elementary_math(float *x, float *out) {
  int idx = threadIdx.x;
  float value = x[idx];
  out[idx] =
    exp2f(value) +
    __exp2f(value) +
    exp10f(value) +
    __exp10f(value) +
    expm1f(value) +
    log2f(fabsf(value) + 1.0f) +
    __log2f(fabsf(value) + 1.0f) +
    log10f(fabsf(value) + 1.0f) +
    __log10f(fabsf(value) + 1.0f) +
    log1pf(fabsf(value)) +
    sinpif(value) +
    cospif(value) +
    cbrtf(value) +
    rcbrtf(value + 2.0f) +
    __frcp_rn(value + 3.0f);
}`, { workgroupSize: [2, 1, 1] });
    const input = new Float32Array([-0.25, 0.5]);
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const expected = [...input].map((value) =>
      (2 ** value) +
      (2 ** value) +
      (10 ** value) +
      (10 ** value) +
      Math.expm1(value) +
      Math.log2(Math.abs(value) + 1) +
      Math.log2(Math.abs(value) + 1) +
      Math.log10(Math.abs(value) + 1) +
      Math.log10(Math.abs(value) + 1) +
      Math.log1p(Math.abs(value)) +
      Math.sin(Math.PI * value) +
      Math.cos(Math.PI * value) +
      Math.cbrt(value) +
      (1 / Math.cbrt(value + 2)) +
      (1 / (value + 3))
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("exp2(value)");
    expect(compiled.wgsl).toContain("pow(10.0, value)");
    expect(compiled.wgsl).toContain("(exp(value) - 1.0)");
    expect(compiled.wgsl).toContain("log2((abs(value) + 1.0))");
    expect(compiled.wgsl).toContain("(log((abs(value) + 1.0)) / 2.302585092994046)");
    expect(compiled.wgsl).toContain("log(1.0 + abs(value))");
    expect(compiled.wgsl).toContain("sin(3.141592653589793 * value)");
    expect(compiled.wgsl).toContain("cos(3.141592653589793 * value)");
    expect(compiled.wgsl).toContain("select(pow(abs(value), 0.3333333333333333)");
    expect(compiled.wgsl).toContain("(1.0 / select(pow(abs((value + 2.0)), 0.3333333333333333)");
    expect(compiled.wgsl).toContain("(1.0 / (value + 3.0))");
    expect([...semanticResult.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...semanticResult.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
  });

  it("lowers CUDA inverse and hyperbolic math intrinsics through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_hyperbolic_math(float *x, float *out) {
  int idx = threadIdx.x;
  float value = x[idx];
  float unit = __saturatef((value + 1.0f) * 0.5f);
  out[idx] =
    asinf(unit) +
    acosf(unit) +
    sinhf(value) +
    coshf(value) +
    asinhf(value) +
    acoshf(fabsf(value) + 1.5f) +
    atanhf(unit * 0.5f);
}`, { workgroupSize: [2, 1, 1] });
    const input = new Float32Array([-0.25, 0.5]);
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const expected = [...input].map((value) => {
      const unit = Math.min(1, Math.max(0, (value + 1) * 0.5));
      return Math.asin(unit) +
        Math.acos(unit) +
        Math.sinh(value) +
        Math.cosh(value) +
        Math.asinh(value) +
        Math.acosh(Math.abs(value) + 1.5) +
        Math.atanh(unit * 0.5);
    });

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("asin(unit)");
    expect(compiled.wgsl).toContain("acos(unit)");
    expect(compiled.wgsl).toContain("(0.5 * (exp(value) - exp(-value)))");
    expect(compiled.wgsl).toContain("(0.5 * (exp(value) + exp(-value)))");
    expect(compiled.wgsl).toContain("log(value + sqrt((value * value) + 1.0))");
    expect(compiled.wgsl).toContain("log((abs(value) + 1.5) + sqrt(((abs(value) + 1.5) * (abs(value) + 1.5)) - 1.0))");
    expect(compiled.wgsl).toContain("(0.5 * log((1.0 + (unit * 0.5)) / (1.0 - (unit * 0.5))))");
    expect([...semanticResult.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...semanticResult.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
  });

  it("lowers CUDA round-away intrinsics through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_round_math(float *x, float *out) {
  int idx = threadIdx.x;
  float value = x[idx];
  out[idx] = roundf(value) + round(value + 0.25f);
}`, { workgroupSize: [2, 1, 1] });
    const input = new Float32Array([-1.5, 2.5]);
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const roundAway = (value: number) => (value < 0 ? -Math.floor(Math.abs(value) + 0.5) : Math.floor(value + 0.5));
    const expected = [...input].map((value) => roundAway(value) + roundAway(value + 0.25));

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("select(floor(abs(value) + 0.5), -floor(abs(value) + 0.5), (value < 0.0))");
    expect(compiled.wgsl).toContain("select(floor(abs((value + 0.25)) + 0.5), -floor(abs((value + 0.25)) + 0.5), ((value + 0.25) < 0.0))");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual(expected);
    expect([...result.buffers.out as Float32Array]).toEqual(expected);
  });

  it("lowers CUDA round-even intrinsics through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_round_even_math(float *x, float *out) {
  int idx = threadIdx.x;
  float value = x[idx];
  out[idx] = rintf(value) + nearbyintf(value) + rint(value + 1.0f) + nearbyint(value + 1.0f);
}`, { workgroupSize: [4, 1, 1] });
    const input = new Float32Array([2.5, 3.5, -2.5, -3.5]);
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );
    const expected = [...input].map((value) => roundEven(value) + roundEven(value) + roundEven(value + 1) + roundEven(value + 1));

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn bg_semantic_round_even_f32");
    expect(compiled.wgsl).toContain("bg_semantic_round_even_f32(value)");
    expect(compiled.wgsl).toContain("bg_semantic_round_even_f32((value + 1.0))");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual(expected);
    expect([...result.buffers.out as Float32Array]).toEqual(expected);
  });

  it("lowers CUDA scalar decomposition math through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_scalar_decomposition(float *x, float *out) {
  int idx = threadIdx.x;
  float value = x[idx];
  out[idx] =
    ldexpf(value, 2) +
    scalbnf(value, 1) +
    scalblnf(value, 3) +
    fmodf(value, 2.0f) +
    remainderf(value, 2.0f) +
    logbf(fabsf(value) + 1.0f) +
    ilogbf(fabsf(value) + 1.0f) +
    fdimf(value, -0.5f);
}`, { workgroupSize: [2, 1, 1] });
    const input = new Float32Array([-1.25, 0.6]);
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const expected = [...input].map((value) =>
      (value * 4) +
      (value * 2) +
      (value * 8) +
      (value - Math.trunc(value / 2) * 2) +
      (value - roundEven(value / 2) * 2) +
      Math.floor(Math.log2(Math.abs(value) + 1)) +
      Math.floor(Math.log2(Math.abs(value) + 1)) +
      Math.max(value - -0.5, 0)
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("(value * exp2(f32(2)))");
    expect(compiled.wgsl).toContain("(value - trunc(value / 2.0) * 2.0)");
    expect(compiled.wgsl).toContain("bg_semantic_remainder_f32(value, 2.0)");
    expect(compiled.wgsl).toContain("bg_semantic_logb_f32((abs(value) + 1.0))");
    expect(compiled.wgsl).toContain("bg_semantic_ilogb_i32((abs(value) + 1.0))");
    expect(compiled.wgsl).toContain("max((value - -(0.5)), 0.0)");
    expect([...semanticResult.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...semanticResult.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
  });

  it("lowers CUDA hypot and norm intrinsics through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_norm_math(float *x, float *out) {
  int idx = threadIdx.x;
  float value = x[idx];
  out[idx] =
    hypotf(value, 2.0f) +
    rhypotf(value, 2.0f) +
    norm3df(value, 2.0f, -3.0f) +
    norm4df(value, 2.0f, -3.0f, 4.0f) +
    rnorm3df(value, 2.0f, -3.0f) +
    rnorm4df(value, 2.0f, -3.0f, 4.0f);
}`, { workgroupSize: [2, 1, 1] });
    const input = new Float32Array([-1.25, 0.6]);
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const expected = [...input].map((value) =>
      Math.hypot(value, 2) +
      (1 / Math.hypot(value, 2)) +
      Math.hypot(value, 2, -3) +
      Math.hypot(value, 2, -3, 4) +
      (1 / Math.hypot(value, 2, -3)) +
      (1 / Math.hypot(value, 2, -3, 4))
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("sqrt((value * value) + (2.0 * 2.0))");
    expect(compiled.wgsl).toContain("(1.0 / sqrt((value * value) + (2.0 * 2.0)))");
    expect(compiled.wgsl).toContain("sqrt((value * value) + (2.0 * 2.0) + (-(3.0) * -(3.0)))");
    expect(compiled.wgsl).toContain("sqrt((value * value) + (2.0 * 2.0) + (-(3.0) * -(3.0)) + (4.0 * 4.0))");
    expect([...semanticResult.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...semanticResult.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
  });

  it("lowers CUDA numeric conversion intrinsics through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_numeric_conversions(float *x, int *iout, uint *uout, float *fout) {
  float a = x[0];
  float b = x[1];
  iout[0] = lrintf(a);
  iout[1] = llrint(b);
  iout[2] = lroundf(a);
  iout[3] = llround(b);
  iout[4] = __float2int_rz(a);
  iout[5] = __float2int_ru(a);
  iout[6] = __float2int_rd(b);
  uout[0] = __float2uint_rn(3.5f);
  uout[1] = __float2uint_rz(3.9f);
  uout[2] = __float2uint_ru(3.1f);
  uout[3] = __float2uint_rd(3.9f);
  fout[0] = __int2float_rn(iout[5]);
  fout[1] = __uint2float_rn(uout[2]);
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        x: new Float32Array([-2.5, 3.5]),
        iout: new Int32Array(7),
        uout: new Uint32Array(4),
        fout: new Float32Array(2),
      },
    };
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      input,
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([-2.5, 3.5]),
          iout: new Int32Array(7),
          uout: new Uint32Array(4),
          fout: new Float32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("i32(bg_semantic_round_even_f32(a))");
    expect(compiled.wgsl).toContain("i32(select(floor(abs(a) + 0.5)");
    expect(compiled.wgsl).toContain("i32(trunc(a))");
    expect(compiled.wgsl).toContain("u32(max(bg_semantic_round_even_f32(3.5), 0.0))");
    expect(compiled.wgsl).toContain("f32(iout[5u])");
    expect(compiled.wgsl).toContain("f32(uout[2u])");
    expect([...semanticResult.buffers.iout as Int32Array]).toEqual([-2, 4, -3, 4, -2, -2, 3]);
    expect([...semanticResult.buffers.uout as Uint32Array]).toEqual([4, 3, 4, 3]);
    expect([...semanticResult.buffers.fout as Float32Array]).toEqual([-2, 4]);
    expect([...result.buffers.iout as Int32Array]).toEqual([-2, 4, -3, 4, -2, -2, 3]);
    expect([...result.buffers.uout as Uint32Array]).toEqual([4, 3, 4, 3]);
    expect([...result.buffers.fout as Float32Array]).toEqual([-2, 4]);
  });

  it("lowers unordered CUDA float predicates without treating them as unsupported calls", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void float_predicates(float *out) {
  if (threadIdx.x == 0) {
    float nan_value = __uint_as_float(0x7fc00000u);
    out[0] = isunordered(nan_value, 1.0f) ? 1.0f : 0.0f;
    out[1] = islessgreater(nan_value, 1.0f) ? 1.0f : 0.0f;
    out[2] = signbit(-0.0f) ? 1.0f : 0.0f;
    out[3] = isnan(nanf("")) ? 1.0f : 0.0f;
    out[4] = isnanf(nan_value) ? 1.0f : 0.0f;
    out[5] = __isnanf(nan_value) ? 1.0f : 0.0f;
    out[6] = isinff(__builtin_inff()) ? 1.0f : 0.0f;
    out[7] = __isinff(__builtin_inff()) ? 1.0f : 0.0f;
    out[8] = finitef(3.0f) && isfinitef(3.0f) && __finitef(3.0f) ? 1.0f : 0.0f;
    out[9] = finite(__builtin_inff()) ? 1.0f : 0.0f;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(10) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(compiled.wgsl).toContain("(((nan_value) != (nan_value)) || ((1.0) != (1.0)))");
    expect(compiled.wgsl).toContain("!(((nan_value) != (nan_value)) || ((1.0) != (1.0)))");
    expect(compiled.wgsl).toContain("((bitcast<u32>(f32((-0.0))) & 0x80000000u) != 0u)");
    expect(compiled.wgsl).toContain("bitcast<f32>(0x7fc00000u)");
    expect(compiled.wgsl).toContain("(abs(bitcast<f32>(0x7f800000u)) > 3.4028234663852886e38)");
    expect([...result.buffers.out as Float32Array]).toEqual([1, 0, 1, 1, 1, 1, 1, 1, 1, 0]);
  });

  it("lowers CUDA float predicates through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_float_predicates(float *x, float *out) {
  float nan_value = x[0];
  float inf_value = x[1];
  float neg_zero = x[2];
  float normal = x[3];
  float subnormal = x[4];
  out[0] = isnan(nan_value) ? 1.0f : 0.0f;
  out[1] = isinf(inf_value) ? 1.0f : 0.0f;
  out[2] = isfinite(inf_value) ? 1.0f : 0.0f;
  out[3] = finitef(normal) && isfinitef(normal) && __finitef(normal) ? 1.0f : 0.0f;
  out[4] = signbit(neg_zero) && signbitf(-normal) ? 1.0f : 0.0f;
  out[5] = isnormal(normal) ? 1.0f : 0.0f;
  out[6] = isnormal(subnormal) ? 1.0f : 0.0f;
  out[7] = isgreater(normal, 2.0f) ? 1.0f : 0.0f;
  out[8] = isgreaterequal(normal, normal) && isless(2.0f, normal) && islessequal(normal, normal) ? 1.0f : 0.0f;
  out[9] = islessgreater(normal, normal) ? 1.0f : 0.0f;
  out[10] = isunordered(nan_value, normal) ? 1.0f : 0.0f;
}`, { workgroupSize: [1, 1, 1] });
    const input = new Float32Array([NaN, Infinity, -0, 3, 1e-40]);
    const runInput = { buffers: { x: input, out: new Float32Array(11) } };
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      runInput,
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(11) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("select(0u, 1u, ((nan_value) != (nan_value)))");
    expect(compiled.wgsl).toContain("select(0u, 1u, (abs(inf_value) > 3.4028234663852886e38))");
    expect(compiled.wgsl).toContain("select(0u, 1u, ((bitcast<u32>(neg_zero) & 0x80000000u) != 0u))");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1]);
    expect([...result.buffers.out as Float32Array]).toEqual([1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1]);
  });

  it("lowers CUDA float bitcast intrinsics through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_float_bitcasts(uint *bits, int *signedBits, float *out) {
  out[0] = __uint_as_float(bits[0]);
  out[1] = __int_as_float(signedBits[0]);
  out[2] = __builtin_inff();
  out[3] = __builtin_huge_valf();
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        bits: new Uint32Array([0x3f800000]),
        signedBits: new Int32Array([-1082130432]),
        out: new Float32Array(4),
      },
    };
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      input,
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          bits: new Uint32Array([0x3f800000]),
          signedBits: new Int32Array([-1082130432]),
          out: new Float32Array(4),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("bitcast<f32>(bits[0u])");
    expect(compiled.wgsl).toContain("bitcast<f32>(signedBits[0u])");
    expect(compiled.wgsl).toContain("bitcast<f32>(0x7f800000u)");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, -1, Infinity, Infinity]);
    expect([...result.buffers.out as Float32Array]).toEqual([1, -1, Infinity, Infinity]);
  });

  it("uses round-to-even semantics for rint, nearbyint, and remainder", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void rounding(float *x, float *out) {
  int idx = threadIdx.x;
  float value = x[idx];
  out[idx] = rintf(value) + nearbyintf(value) + remainderf(value, 2.0f) + logbf(fabsf(value)) + ilogbf(fabsf(value));
}`, { workgroupSize: [4, 1, 1] });
    const input = new Float32Array([2.5, 3.5, -2.5, 5.75]);
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(compiled.wgsl).toContain("fn bg_semantic_round_even_f32(");
    expect(compiled.wgsl).toContain("fn bg_semantic_remainder_f32(");
    expect(compiled.wgsl).toContain("fn bg_semantic_ilogb_i32(");
    expect([...result.buffers.out as Float32Array]).toEqual([...input].map((value) =>
      roundEven(value) + roundEven(value) + (value - roundEven(value / 2) * 2) +
      Math.floor(Math.log2(Math.abs(value))) + Math.floor(Math.log2(Math.abs(value))),
    ));
  });

  it("lowers CUDA remquo quotient out params", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void remquoKernel(float *out, int *quo) {
  int localQuo = 0;
  float localRem = remquof(7.0f, 2.0f, &localQuo);
  out[0] = localRem;
  quo[0] = localQuo;
  out[1] = remquo(5.0f, 2.0f, &quo[1]);
  remquof(-7.0f, 2.0f, &quo[2]);
}
`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(2), quo: new Int32Array(3) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-remquo-quotient");
    expect(compiled.wgsl).toContain("bg_round_even_f32(7.0 / 2.0)");
    expect(compiled.wgsl).toContain("quo[1] = i32(bg_remquo_quo_");
    expect(compiled.wgsl).toContain("quo[2] = i32(bg_remquo_quo_");
    expect([...result.buffers.out as Float32Array]).toEqual([-1, 1]);
    expect([...result.buffers.quo as Int32Array]).toEqual([4, 2, -4]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).not.toContain("bg_ptr_write_i32");
    expect(compiled.wgsl).not.toContain("var q:");
    expect(compiled.wgsl).toContain("var bg__bg_remquo_dividend_");
    expect(compiled.wgsl).toContain("quo[1u] = select(select(i32(floor((bg__bg_remquo_dividend_");
    expect(compiled.wgsl).toContain("quo[2u] = select(select(i32(floor((bg__bg_remquo_dividend_");
    expect([...semanticResult.buffers.quo as Int32Array]).toEqual([0, 4, -4]);
    expect([...result.buffers.quo as Int32Array]).toEqual([0, 4, -4]);
  });

  it("lowers CUDA special function math through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semanticSpecialMath(float *x, float *out) {
  int idx = threadIdx.x;
  float value = x[idx];
  float positive = fabsf(value) + 1.0f;
  out[idx] =
    erff(value) +
    erfcf(value) +
    erfcxf(value) +
    normcdff(value) +
    tgammaf(positive) +
    lgammaf(positive);
}
`, { workgroupSize: [2, 1, 1] });
    const input = new Float32Array([-0.25, 0.5]);
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: input, out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const expected = [...input].map((value) => {
      const positive = Math.abs(value) + 1;
      const gamma = gammaApprox(positive);
      return erfApprox(value) +
        (1 - erfApprox(value)) +
        (Math.exp(value * value) * (1 - erfApprox(value))) +
        (0.5 * (1 + erfApprox(value * Math.SQRT1_2))) +
        gamma +
        Math.log(Math.abs(gamma));
    });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn bg_semantic_erf_f32(");
    expect(compiled.wgsl).toContain("fn bg_semantic_tgamma_f32(");
    expect(compiled.wgsl).toContain("fn bg_semantic_lgamma_f32(");
    expect(compiled.wgsl).toContain("bg_semantic_erf_f32(value)");
    expect(compiled.wgsl).toContain("bg_semantic_tgamma_f32(positive)");
    expect(compiled.wgsl).toContain("bg_semantic_lgamma_f32(positive)");
    expect([...semanticResult.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...semanticResult.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(expected[0]!, 5);
    expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(expected[1]!, 5);
  });

  it("lowers CUDA nextafter and nexttoward intrinsics", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void nextafterKernel(float *out) {
  out[0] = nextafterf(1.0f, 2.0f);
  out[1] = nextafterf(1.0f, 0.0f);
  out[2] = nextafterf(0.0f, -1.0f);
  out[3] = nexttoward(-1.0f, 0.0f);
  out[4] = nextafterf(__builtin_inff(), 0.0f);
}
`, { workgroupSize: [1, 1, 1] });
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Float32Array(5) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(5) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn bg_semantic_nextafter_f32(");
    expect(compiled.wgsl).toContain("bg_semantic_nextafter_f32(1.0, 2.0)");
    const expected = [
      nextafterApprox(1, 2),
      nextafterApprox(1, 0),
      nextafterApprox(0, -1),
      nextafterApprox(-1, 0),
      nextafterApprox(Infinity, 0),
    ];
    expect([...semanticResult.buffers.out as Float32Array]).toEqual(expected);
    expect([...result.buffers.out as Float32Array]).toEqual(expected);
  });

  it("lowers C math aliases used in CUDA snippets", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void c_math_aliases(float *x, float *out) {
  float value = x[0];
  out[0] = fabs(value) + exp(value) + log(fabs(value) + 1.0f) +
    pow(fabs(value), 2.0f) + fmin(value, 1.0f) + fmax(value, -1.0f) +
    __sinf(value) + __cosf(value) + __tanf(value) + erff(value) + erfcf(value) +
    tgamma(fabs(value) + 1.0f) + lgamma(fabs(value) + 1.0f) + lerp(2.0f, 6.0f, 0.25f);
}`, { workgroupSize: [1, 1, 1] });
    const value = -0.25;
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([value]), out: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const expected = Math.abs(value) + Math.exp(value) + Math.log(Math.abs(value) + 1) +
      Math.pow(Math.abs(value), 2) + Math.min(value, 1) + Math.max(value, -1) +
      Math.sin(value) + Math.cos(value) + Math.tan(value) + erfApprox(value) + (1 - erfApprox(value)) +
      gammaApprox(Math.abs(value) + 1) + Math.log(Math.abs(gammaApprox(Math.abs(value) + 1))) + 3;

    expect(compiled.wgsl).toContain("abs(value)");
    expect(compiled.wgsl).toContain("exp(value)");
    expect(compiled.wgsl).toContain("0.3275911");
    expect(compiled.wgsl).toContain("bg_semantic_tgamma_f32((abs(value) + 1.0))");
    expect(compiled.wgsl).toContain("bg_semantic_lgamma_f32((abs(value) + 1.0))");
    expect(compiled.wgsl).toContain("pow(abs(value), 2.0)");
    expect(compiled.wgsl).toContain("fma(0.25, (6.0 - 2.0), 2.0)");
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(expected, 5);
  });

  it("casts integer CUDA math arguments to WGSL float arguments", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void int_math_arg(int n, float *out) {
  out[0] = sqrtf(n) + expf(n - 2) + fminf(n, 3.5f);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) }, scalars: { n: 4 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Float32Array(1) }, scalars: { n: 4 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("sqrt(f32(bg_uniforms.n))");
    expect(compiled.wgsl).toContain("exp(f32((bg_uniforms.n - 2)))");
    expect(compiled.wgsl).toContain("min(f32(bg_uniforms.n), 3.5)");
    expect([...semanticResult.buffers.out as Float32Array][0]).toBeCloseTo(Math.sqrt(4) + Math.exp(2) + 3.5, 5);
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(Math.sqrt(4) + Math.exp(2) + 3.5, 5);
  });

  it("lowers integer CUDA div_ceil helpers through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void int_div_ceil(int n, uint* out) {
  if (threadIdx.x == 0) {
    out[0] = uint(div_ceil(n, 4));
    out[1] = uint(ceil_div(n + 1, 4));
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(2) }, scalars: { n: 17 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Uint32Array(2) }, scalars: { n: 17 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("(((bg_uniforms.n + 4) - 1) / 4)");
    expect(compiled.wgsl).toContain("((((bg_uniforms.n + 1) + 4) - 1) / 4)");
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([5, 5]);
    expect([...result.buffers.out as Uint32Array]).toEqual([5, 5]);
  });

  it("lowers CUDA math aliases through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_math_aliases(float x, float* out) {
  if (threadIdx.x == 0) {
    out[0] = __sinf(x) + __cosf(x) + __tanhf(x);
    out[1] = rsqrtf(x + 4.0f) + __frsqrt_rn(x + 9.0f) + atan2f(x, 2.0f);
    out[2] = fmaf(x, 2.0f, 1.0f) + __fdividef(x, 2.0f);
    out[3] = lerp(2.0f, 6.0f, x);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(4) }, scalars: { x: 0.25 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Float32Array(4) }, scalars: { x: 0.25 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("inverseSqrt((bg_uniforms.x + 4.0))");
    expect(compiled.wgsl).toContain("atan2(bg_uniforms.x, 2.0)");
    expect(compiled.wgsl).toContain("fma(bg_uniforms.x, 2.0, 1.0)");
    expect(compiled.wgsl).toContain("(bg_uniforms.x / 2.0)");
    expect(compiled.wgsl).toContain("fma(bg_uniforms.x, (6.0 - 2.0), 2.0)");
    for (let i = 0; i < 4; i++) {
      expect([...semanticResult.buffers.out as Float32Array][i]).toBeCloseTo([...result.buffers.out as Float32Array][i]!, 5);
    }
  });

  it("lowers C frexp exponent out params", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void frexpKernel(float *out, int *expOut) {
  int exponent = 0;
  float mantissa = frexp(9.0f, &exponent);
  float storageMantissa = frexpf(10.0f, &expOut[1]);
  out[0] = mantissa;
  out[1] = storageMantissa;
  out[2] = frexpf(12.0f, &expOut[2]);
  expOut[0] = exponent;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(3), expOut: new Int32Array(3) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn bg_frexp(");
    expect(compiled.wgsl).toContain("bg_frexp(9.0, &exponent)");
    expect(compiled.wgsl).toContain("bg_frexp(10.0, &bg_frexp_exp_");
    expect(compiled.wgsl).toContain("expOut[1] = i32(bg_frexp_exp_");
    expect(compiled.wgsl).toContain("out[2] = f32(bg_frexp(12.0, &bg_frexp_exp_");
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(0.5625, 6);
    expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(0.625, 6);
    expect([...result.buffers.out as Float32Array][2]).toBeCloseTo(0.75, 6);
    expect([...result.buffers.expOut as Int32Array]).toEqual([4, 4, 4]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).not.toContain("bg_ptr_write_i32");
    expect(compiled.wgsl).not.toContain("var expAlias:");
    expect(compiled.wgsl).toContain("var bg__bg_frexp_value_");
    expect(compiled.wgsl).toContain("expOut[1u] = select((i32(floor(log2(abs(bg__bg_frexp_value_");
    expect(compiled.wgsl).toContain("expOut[2u] = select((i32(floor(log2(abs(bg__bg_frexp_value_");
    expect([...semanticResult.buffers.expOut as Int32Array]).toEqual([0, 4, 0]);
    expect([...result.buffers.expOut as Int32Array]).toEqual([0, 4, 0]);
  });

  it("lowers C modf integer-part out params", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void modfKernel(float *out) {
  float localInt = 0.0f;
  float localFrac = modff(3.75f, &localInt);
  out[0] = localFrac;
  out[1] = localInt;
  out[2] = modff(-2.25f, &out[3]);
  float assignedInt = 0.0f;
  out[4] = modf(5.5f, &assignedInt);
  out[5] = assignedInt;
  modff(8.125f, &out[6]);
}
`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(7).fill(-99) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-modf-intpart");
    expect(compiled.wgsl).toContain("fn bg_modf(");
    expect(compiled.wgsl).toContain("bg_modf(3.75, &localInt)");
    expect(compiled.wgsl).toContain("out[3] = f32(bg_modf_int_");
    expect(compiled.wgsl).toContain("out[6] = f32(bg_modf_int_");
    expect([...result.buffers.out as Float32Array]).toEqual([0.75, 3, -0.25, -2, 0.5, 5, 8]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
    expect(compiled.wgsl).not.toContain("var intpart:");
    expect(compiled.wgsl).toContain("var bg__bg_modf_value_");
    expect(compiled.wgsl).toContain("out[1u] = select(trunc(bg__bg_modf_value_");
    expect(compiled.wgsl).toContain("out[2u] = select(trunc(bg__bg_modf_value_");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([0, 8, -2]);
    expect([...result.buffers.out as Float32Array]).toEqual([0, 8, -2]);
  });

  it("lowers CUDA sincos output params", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void sincosKernel(float *out) {
  float s = 0.0f;
  float c = 0.0f;
  sincosf(0.25f, &s, &c);
  out[0] = s;
  out[1] = c;
  __sincosf(0.5f, &out[2], &out[3]);
  sincospi(0.5f, &out[4], &out[5]);
}
`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(6) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-sincos-output");
    expect(compiled.wgsl).toContain("let bg_sincos_sin_");
    expect(compiled.wgsl).toContain("sin(0.25)");
    expect(compiled.wgsl).toContain("cos(0.25)");
    expect(compiled.wgsl).toContain("sin((3.141592653589793 * 0.5))");
    const out = [...result.buffers.out as Float32Array];
    expect(out[0]).toBeCloseTo(Math.sin(0.25), 6);
    expect(out[1]).toBeCloseTo(Math.cos(0.25), 6);
    expect(out[2]).toBeCloseTo(Math.sin(0.5), 6);
    expect(out[3]).toBeCloseTo(Math.cos(0.5), 6);
    expect(out[4]).toBeCloseTo(1, 6);
    expect(out[5]).toBeCloseTo(0, 6);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("out[0u] = sin(0.25);");
    expect(compiled.wgsl).toContain("out[1u] = cos(0.25);");
    expect(compiled.wgsl).toContain("out[2u] = sin((3.141592653589793 * 0.5));");
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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

  it("lowers CUDA math output var initializers through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void mathOutVarInits(float *out, int *ints) {
  float *intpart = out + 1;
  int *expOut = ints + 1;
  int *quoOut = ints + 2;
  float frac = modff(-3.75f, intpart);
  float mantissa = frexpf(9.0f, expOut);
  float rem = remquof(7.0f, 2.0f, quoOut);
  out[0] = frac;
  out[2] = mantissa;
  out[3] = rem;
  out[4] = (float)ints[1];
  out[5] = (float)ints[2];
}
`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(6), ints: new Int32Array(3) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Float32Array(6), ints: new Int32Array(3) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-frexp-exponent");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-modf-intpart");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-remquo-quotient");
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
    expect(compiled.wgsl).not.toContain("bg_ptr_write_i32");
    expect(compiled.wgsl).toContain("var bg__bg_modf_value_");
    expect(compiled.wgsl).toContain("out[1u] = select(trunc(bg__bg_modf_value_");
    expect(compiled.wgsl).toContain("var frac: f32 = select(select((bg__bg_modf_value_");
    expect(compiled.wgsl).toContain("ints[1u] = select((i32(floor(log2(abs(bg__bg_frexp_value_");
    expect(compiled.wgsl).toContain("var mantissa: f32 = select((bg__bg_frexp_value_");
    expect(compiled.wgsl).toContain("ints[2u] = select(select(i32(floor((bg__bg_remquo_dividend_");
    expect(compiled.wgsl).toContain("var rem: f32 = (bg__bg_remquo_dividend_");
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).not.toContain("bg_ptr_write_i32");
    expect(compiled.wgsl).toContain("bg_uniforms.x");
    expect(compiled.wgsl).toContain("var bg__bg_remquo_dividend_");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([-1, 0, 4, 3]);
    expect([...semanticResult.buffers.ints as Int32Array]).toEqual([0, 4, 3]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([...result.buffers.out as Float32Array]);
    expect([...semanticResult.buffers.ints as Int32Array]).toEqual([...result.buffers.ints as Int32Array]);
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
    expect(compiled.wgsl).toContain("lerp(2.0, 6.0, 0.25");
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
    out[10] = uint(IMAD(6, 7, -2));
    out[11] = __brev(0x01234567u);
    out[12] = __sad(-20, 7, 3u);
    out[13] = __usad(2u, 9u, 5u);
    out[14] = uint(__mulhi(-2000000000, 3));
    out[15] = __umulhi(0xfedcba98u, 0x12345678u);
    out[16] = __byte_perm(0x00112233u, 0x44556677u, 0x5410u);
    out[17] = uint(__rhadd(-7, 2));
    out[18] = __uhadd(0xffffffffu, 1u);
    out[19] = __urhadd(0xffffffffu, 2u);
    out[20] = uint(__hadd(-7, 2));
    out[21] = __funnelshift_l(0x11223344u, 0x55667788u, 8u);
    out[22] = __funnelshift_lc(0x11223344u, 0x55667788u, 40u);
    out[23] = __funnelshift_r(0x11223344u, 0x55667788u, 8u);
    out[24] = __funnelshift_rc(0x11223344u, 0x55667788u, 32u);
    out[25] = uint(__clzll(0x10u));
    out[26] = uint(__ffsll(0x10u));
    out[27] = uint(__popcll(0xf0f0u));
    out[28] = __brevll(0x000000f0u);
    out[29] = uint(__mul64hi(-2000000000, 3));
    out[30] = __umul64hi(0xfedcba98u, 0x12345678u);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Int32Array([5]), out: new Uint32Array(31) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { x: new Int32Array([5]), out: new Uint32Array(31) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("countLeadingZeros");
    expect(compiled.wgsl).toContain("countTrailingZeros");
    expect(compiled.wgsl).toContain("reverseBits(19088743u)");
    expect(compiled.wgsl).toContain("select((u32(7) - u32(-(20))), (u32(-(20)) - u32(7)), (-(20) >= 7)) + 3u");
    expect(compiled.wgsl).toContain("max(2u, 9u) - min(2u, 9u) + 5u");
    expect(compiled.wgsl).toContain("fn bg_semantic_umulhi_u32(");
    expect(compiled.wgsl).toContain("fn bg_semantic_byte_perm_u32(");
    expect(compiled.wgsl).toContain("(-(7) | 2) - ((-(7) ^ 2) >> 1u)");
    expect(compiled.wgsl).toContain("4294967295u & 1u");
    expect(compiled.wgsl).toContain("4294967295u & 2u");
    expect(compiled.wgsl).toContain("(-(7) & 2) + ((-(7) ^ 2) >> 1u)");
    expect(compiled.wgsl).toContain("fn bg_semantic_funnelshift_l_u32(");
    expect(compiled.wgsl).toContain("bg_semantic_funnelshift_rc_u32(287454020u, 1432778632u, 32u)");
    expect(compiled.wgsl).toContain("countLeadingZeros(16u)) + 32");
    expect(compiled.wgsl).toContain("countOneBits(61680u)");
    expect(compiled.wgsl).toContain("min(7u, 3u)");
    expect(compiled.wgsl).not.toContain("assert omitted");
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([29, 15, 20, 3, 3, 0, 4, 7, 42, 44, 40, 0xe6a2c480, 30, 12, 0xfffffffe, 304062474, 0x66772233, 0xfffffffe, 2147483648, 2147483649, 0xfffffffd, 0x22334455, 0x55667788, 0x88112233, 0x55667788, 59, 5, 8, 0x0f000000, 0xfffffffe, 304062474]);
    expect([...result.buffers.out as Uint32Array]).toEqual([29, 15, 20, 3, 3, 0, 4, 7, 42, 44, 40, 0xe6a2c480, 30, 12, 0xfffffffe, 304062474, 0x66772233, 0xfffffffe, 2147483648, 2147483649, 0xfffffffd, 0x22334455, 0x55667788, 0x88112233, 0x55667788, 59, 5, 8, 0x0f000000, 0xfffffffe, 304062474]);
  });

  it("lowers CUDA float/integer bitcast intrinsics", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void bitcast_intrinsics(float *x, uint *bits, int *signed_bits, float *roundtrip) {
  float value = x[0];
  bits[0] = __float_as_uint(value);
  signed_bits[0] = __float_as_int(value);
  roundtrip[0] = __uint_as_float(bits[0]);
  roundtrip[1] = __int_as_float(signed_bits[0]);
}`, { workgroupSize: [1, 1, 1] });
    const input = new Float32Array([-3.5]);
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: input,
          bits: new Uint32Array(1),
          signed_bits: new Int32Array(1),
          roundtrip: new Float32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {
          x: input,
          bits: new Uint32Array(1),
          signed_bits: new Int32Array(1),
          roundtrip: new Float32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("bitcast<u32>(value)");
    expect(compiled.wgsl).toContain("bitcast<i32>(value)");
    expect(compiled.wgsl).toContain("bitcast<f32>(bits[0u])");
    expect([...semanticResult.buffers.bits as Uint32Array]).toEqual([0xc0600000]);
    expect([...semanticResult.buffers.signed_bits as Int32Array]).toEqual([-1067450368]);
    expect([...semanticResult.buffers.roundtrip as Float32Array]).toEqual([-3.5, -3.5]);
    expect([...result.buffers.bits as Uint32Array]).toEqual([0xc0600000]);
    expect([...result.buffers.signed_bits as Int32Array]).toEqual([-1067450368]);
    expect([...result.buffers.roundtrip as Float32Array]).toEqual([-3.5, -3.5]);
  });

  it("lowers CUDA scalar conversion intrinsics with rounding modes", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void convert_intrinsics(float *x, int *iout, uint *uout, float *fout) {
  float a = x[0];
  float b = x[1];
  iout[0] = __float2int_rn(a);
  iout[1] = __float2int_rz(a);
  iout[2] = __float2int_ru(a);
  iout[3] = __float2int_rd(a);
  iout[4] = __float2int_rn(b);
  uout[0] = __float2uint_rn(3.5f);
  uout[1] = __float2uint_rz(3.9f);
  uout[2] = __float2uint_ru(3.1f);
  uout[3] = __float2uint_rd(3.9f);
  fout[0] = __int2float_rn(iout[3]);
  fout[1] = __uint2float_rn(uout[2]);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([2.5, 3.5]),
          iout: new Int32Array(5),
          uout: new Uint32Array(4),
          fout: new Float32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(compiled.wgsl).toContain("bg_semantic_round_even_f32(a)");
    expect(compiled.wgsl).toContain("i32(trunc(a))");
    expect(compiled.wgsl).toContain("i32(ceil(a))");
    expect(compiled.wgsl).toContain("i32(floor(a))");
    expect(compiled.wgsl).toContain("u32(max(bg_semantic_round_even_f32(3.5), 0.0))");
    expect([...result.buffers.iout as Int32Array]).toEqual([2, 2, 3, 2, 4]);
    expect([...result.buffers.uout as Uint32Array]).toEqual([4, 3, 4, 3]);
    expect([...result.buffers.fout as Float32Array]).toEqual([2, 4]);
  });

  it("lowers C integer rounding math aliases with CUDA tie semantics", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void round_integer_aliases(float *x, int *out) {
  out[0] = lrintf(x[0]);
  out[1] = lrint(x[1]);
  out[2] = llrintf(x[2]);
  out[3] = llrint(x[3]);
  out[4] = lroundf(x[0]);
  out[5] = lround(x[2]);
  out[6] = llroundf(x[3]);
  out[7] = llround(x[4]);
  out[8] = (int)roundf(x[4]);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([2.5, 3.5, -2.5, -3.5, -1.5]),
          out: new Int32Array(9),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(compiled.wgsl).toContain("i32(bg_semantic_round_even_f32(x[0u]))");
    expect(compiled.wgsl).toContain("i32(select(floor(abs(x[0u]) + 0.5)");
    expect(compiled.wgsl).toContain("select(floor(abs(x[4u]) + 0.5)");
    expect([...result.buffers.out as Int32Array]).toEqual([2, 4, -2, -4, 3, -3, -4, -2, -2]);
  });

  it("lowers inverse error and normal-CDF CUDA math aliases to WGSL helpers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void inverse_distribution_math(float *out) {
  out[0] = erfinvf(0.8427008f);
  out[1] = erfinv(-0.8427008f);
  out[2] = erfcinvf(0.1572992f);
  out[3] = normcdfinvf(0.841344746f);
  out[4] = normcdfinv(0.158655254f);
  out[5] = normcdff(1.0f);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(6) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Float32Array(6) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const out = [...result.buffers.out as Float32Array];
    const semanticOut = [...semanticResult.buffers.out as Float32Array];

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn bg_semantic_erfinv_f32(");
    expect(compiled.wgsl).toContain("fn bg_semantic_normcdfinv_f32(");
    expect(compiled.wgsl).toContain("bg_semantic_erfinv_f32(0.8427008)");
    expect(compiled.wgsl).toContain("bg_semantic_erfinv_f32(1.0 - 0.1572992)");
    expect(compiled.wgsl).toContain("bg_semantic_normcdfinv_f32(0.841344746)");
    expect(out[0]).toBeCloseTo(1, 4);
    expect(out[1]).toBeCloseTo(-1, 4);
    expect(out[2]).toBeCloseTo(1, 4);
    expect(out[3]).toBeCloseTo(1, 4);
    expect(out[4]).toBeCloseTo(-1, 4);
    expect(out[5]).toBeCloseTo(0.8413447, 4);
    expect(semanticOut[0]).toBeCloseTo(1, 4);
    expect(semanticOut[1]).toBeCloseTo(-1, 4);
    expect(semanticOut[2]).toBeCloseTo(1, 4);
    expect(semanticOut[3]).toBeCloseTo(1, 4);
    expect(semanticOut[4]).toBeCloseTo(-1, 4);
    expect(semanticOut[5]).toBeCloseTo(0.8413447, 4);
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
    out[6] = isinf(out[0]) ? 1.0f : 0.0f;
    out[7] = isnan(out[3]) ? 1.0f : 0.0f;
    out[8] = isNan(out[3]) ? 1.0f : 0.0f;
    out[9] = CUDART_PI_F + CUDART_PIO2_F + CUDART_SQRT_HALF_F;
    out[10] = CUDART_INF_F;
    out[11] = CUDART_NAN_F;
    out[12] = CUDART_L2E_F + CUDART_LN2_F + CUDART_ONE_F + CUDART_ZERO_F;
    kinds[0] = cudaMemcpyDeviceToDevice + cudaStreamNonBlocking;
    kinds[1] = warpSize + WARP_SIZE;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(13), kinds: new Uint32Array(2) } },
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
    expect(out[6]).toBe(1);
    expect(out[7]).toBe(1);
    expect(out[8]).toBe(1);
    expect(out[9]).toBeCloseTo(Math.PI + (Math.PI / 2) + Math.SQRT1_2, 6);
    expect(out[10]).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isNaN(out[11])).toBe(true);
    expect(out[12]).toBeCloseTo(Math.LOG2E + Math.LN2 + 1, 6);
    expect([...result.buffers.kinds as Uint32Array]).toEqual([4, 64]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var value: f32 = x[");
    expect(compiled.wgsl).toContain("y[");
    expect([...semanticResult.buffers.y as Float32Array]).toEqual([3, 7]);
    expect([...result.buffers.y as Float32Array]).toEqual([3, 7]);
  });

  it("accepts CUDA default kernel parameter initializers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void optional_output(int *data, int width, int *partial_sums = NULL) {
  int idx = threadIdx.x;
  if (idx < width) {
    data[idx] = data[idx] + 1;
  }
  if (partial_sums != NULL && idx == 0) {
    partial_sums[0] = data[0];
  }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          data: new Int32Array([3, 4]),
          partial_sums: new Int32Array(1),
        },
        scalars: { width: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(backendIr(compiled).params.map((param) => param.name)).toEqual(["data", "width", "partial_sums"]);
    expect([...result.buffers.data as Int32Array]).toEqual([4, 5]);
    expect([...result.buffers.partial_sums as Int32Array]).toEqual([4]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var value: vec4<f32> = vec4<f32>(x[((u32(i) * 4u) + 0u)], x[((u32(i) * 4u) + 1u)], x[((u32(i) * 4u) + 2u)], x[((u32(i) * 4u) + 3u)]);");
    expect(compiled.wgsl).toContain("y[((u32(i) * 4u) + 3u)] = (vec4<f32>(f32((value.x + 1.0)), f32((value.y + 2.0)), f32((value.z + 3.0)), f32((value.w + 4.0)))).w;");
    expect([...result.buffers.y as Float32Array]).toEqual([2, 4, 6, 8, 6, 8, 10, 12]);
    expect([...semanticResult.buffers.y as Float32Array]).toEqual([2, 4, 6, 8, 6, 8, 10, 12]);
  });

  it("keeps dynamic local vector lane reads scalar inside casts", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__global__ void reduce_add_sum_kernel(float* dst, const float* src, size_t n, size_t m) {
  const size_t idx = threadIdx.x * 4;
  if (idx < n) {
    float4 acc;
    for (int k = 0; k < 4; ++k) {
      acc[k] = 0.f;
    }
    for (int l = 0; l < m; ++l) {
      float4 s = reinterpret_cast<float4 *>(src + idx + n * l)[0];
      for (int k = 0; k < 4; ++k) {
        acc[k] += s[k];
      }
    }
    for (int k = 0; k < 4; ++k) {
      dst[idx + k] = (float)((float)dst[idx + k] + acc[k]);
    }
  }
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).not.toContain("f32((vec4<f32>");
    expect(compiled.wgsl).toContain("dst[(idx + u32(k))] = f32(f32((f32(dst[(idx + u32(k))]) + acc[u32(k)])));");
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

  it("accepts local C++ aliases and constexpr const declarations", () => {
    const compiled = compileCudaLiteKernel(`
typedef float floatX;
__global__ void local_cpp_shapes(const floatX* input, floatX* output) {
  using x128 = Packed128<floatX>;
  constexpr const int Lanes = 4;
  int idx = threadIdx.x * Lanes;
  x128 value = reinterpret_cast<x128 *>(input + idx)[0];
  output[idx] = value[0] + Lanes;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { input: new Float32Array([2, 3, 4, 5]), output: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("vec4<f32>");
    expect([...result.buffers.output as Float32Array]).toEqual([6, 0, 0, 0]);
  });

  it("runs scalarized wide half pack loads and stores", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void half_pack(const half* input, half* output) {
  int idx = threadIdx.x * 8;
  half packed_in[8];
  half packed_out[8];
  packed_in[0] = input[idx + 0];
  packed_in[1] = input[idx + 1];
  packed_in[2] = input[idx + 2];
  packed_in[3] = input[idx + 3];
  packed_in[4] = input[idx + 4];
  packed_in[5] = input[idx + 5];
  packed_in[6] = input[idx + 6];
  packed_in[7] = input[idx + 7];
  for (int lane = 0; lane < 8; lane++) {
    packed_out[lane] = __float2half(__half2float(packed_in[lane]) + 1.0f);
  }
  output[idx + 0] = packed_out[0];
  output[idx + 1] = packed_out[1];
  output[idx + 2] = packed_out[2];
  output[idx + 3] = packed_out[3];
  output[idx + 4] = packed_out[4];
  output[idx + 5] = packed_out[5];
  output[idx + 6] = packed_out[6];
  output[idx + 7] = packed_out[7];
}`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          input: createWgslFloat16Array([1, 2, 3, 4, 5, 6, 7, 8]),
          output: createWgslFloat16Array(8),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var packed_in: array<f16, 8>;");
    expect(Array.from(result.buffers.output as Iterable<number>)).toEqual([2, 3, 4, 5, 6, 7, 8, 9]);
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

    expect(compiled.wgsl).toContain("vec4<f32>(f32(2.5), f32(2.5), f32(2.5), f32(2.5))");
    expect(compiled.wgsl).toContain("vec4<u32>(u32(7u), u32(7u), u32(7u), u32(7u))");
    expect([...result.buffers.out as Float32Array]).toEqual([2.5, 2.5, 2.5, 2.5]);
    expect([...result.buffers.kinds as Uint32Array]).toEqual([7, 7, 7, 7]);
  });

  it("supports CUDA vector-to-vector conversion constructors", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float3 trim(float4 value) {
  return make_float3(value);
}
__global__ void vector_convert(float4 *input, float3 *out) {
  float4 value = input[0];
  out[0] = trim(value);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { input: new Float32Array([1, 2, 3, 4]), out: new Float32Array(3) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { input: new Float32Array([1, 2, 3, 4]), out: new Float32Array(3) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("vec3<f32>(f32((value).x), f32((value).y), f32((value).z))");
    expect([...result.buffers.out as Float32Array]).toEqual([1, 2, 3]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 2, 3]);
  });

  it("supports CUDA vector conversion constructors across scalar families", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void vector_convert(uint4 *input, float4 *out) {
  uint4 raw = input[0];
  out[0] = make_float4(raw);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { input: new Uint32Array([3, 5, 7, 11]), out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("vec4<f32>(f32(raw.x), f32(raw.y), f32(raw.z), f32(raw.w))");
    expect([...result.buffers.out as Float32Array]).toEqual([3, 5, 7, 11]);
  });

  it("supports CUDA vector constructors with vector prefix and scalar tail", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void pack(float4 *out) {
  float3 xyz = make_float3(2.0f, 3.0f, 5.0f);
  out[0] = make_float4(xyz, 7.0f);
}`, { workgroupSize: [1, 1, 1] });
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

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("vec4<f32>(f32((xyz).x), f32((xyz).y), f32((xyz).z), f32(7.0))");
    expect([...result.buffers.out as Float32Array]).toEqual([2, 3, 5, 7]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([2, 3, 5, 7]);
  });

  it("lowers CUDA helper_math vector operations", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void vector_math(float3 *out, float *scalars) {
  float3 a = make_float3(3.0f, 4.0f, 0.0f);
  float3 b = make_float3(0.0f, 1.0f, 2.0f);
  float3 n = normalize(a);
  float3 c = cross(a, b);
  out[0] = make_float3(n.x + c.x, n.y + c.y, n.z + c.z);
  scalars[0] = dot(a, b);
  scalars[1] = length(a);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(3), scalars: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("normalize(a)");
    expect(compiled.wgsl).toContain("cross(a, b)");
    expect(compiled.wgsl).toContain("dot(a, b)");
    expect([...result.buffers.scalars as Float32Array]).toEqual([4, 5]);
    expect([...result.buffers.out as Float32Array]).toEqual([
      expect.closeTo(8.6),
      expect.closeTo(-5.2),
      3,
    ]);
  });

  it("lowers CUDA helper_math vector lerp without shadowing scalar lerp", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void vector_lerp(float3 *out) {
  float3 a = make_float3(1.0f, 2.0f, 3.0f);
  float3 b = make_float3(5.0f, 10.0f, 15.0f);
  out[0] = lerp(a, b, 0.25f);
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

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fma(vec3<f32>(f32(0.25)");
    expect([...result.buffers.out as Float32Array]).toEqual([2, 4, 6]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([2, 4, 6]);
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

    expect(compiled.wgsl).toContain("vec4<u32>(1u, 2u, 3u, 4u)");
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

    expect(compiled.wgsl).toContain("[u32(bg_uniforms.lane)]");
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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

    expect(compiled.wgsl).toContain("bg_ptr_write_f16x2(v_buffer, (v_base + (u32(0) * 2u)), (bg_ptr_read_f16x2(v_buffer, (v_base + (u32(0) * 2u))) + bg_ptr_read_f16x2(v_buffer, (v_base + (u32(1) * 2u)))))");
    expect(compiled.wgsl).not.toMatch(/vec2<f32>\(f32\(\(bg_ptr_read_f16x2/u);
  });

  it("lowers CUDA half2 arithmetic intrinsics", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void half2Ops(const half2 *x, const half2 *y, half2 *out, float *scalar) {
  half2 sum = __hadd2(x[0], y[0]);
  half2 prod = __hmul2(sum, make_half2(__float2half(2.0f), __float2half(0.5f)));
  out[0] = __hmax2(prod, make_half2(__float2half(5.0f), __float2half(5.0f)));
  half2 fused = __hfma2(x[0], y[0], make_half2(__float2half(1.0f), __float2half(2.0f)));
  out[1] = x[0] * y[0] + fused;
  scalar[0] = __low2float(fused) + __high2float(fused);
}`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: createWgslFloat16Array([1, 8]),
          y: createWgslFloat16Array([2, 4]),
          out: createWgslFloat16Array(4),
          scalar: new Float32Array(1),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {
          x: createWgslFloat16Array([1, 8]),
          y: createWgslFloat16Array([2, 4]),
          out: createWgslFloat16Array(4),
          scalar: new Float32Array(1),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("max(");
    expect(compiled.wgsl).toContain("fma(");
    expect(Array.from(result.buffers.out as ArrayLike<number>)).toEqual([6, 6, 5, 66]);
    expect([...result.buffers.scalar as Float32Array]).toEqual([37]);
    expect(Array.from(semanticResult.buffers.out as ArrayLike<number>)).toEqual([6, 6, 5, 66]);
    expect([...semanticResult.buffers.scalar as Float32Array]).toEqual([37]);
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
    expect(compiled.wgsl).toContain("bg_warp_shuffle_sync_int_32(0, 0u, 32u, local_id)");
    expect(compiled.wgsl).toContain("workgroupBarrier()");
    expect(compiled.wgsl).toContain("storageBarrier()");
    expect(compiled.wgsl).toContain("pack2x16float(vec2<f32>(f32((value).x), f32((value).y)))");
    expect(compiled.wgsl).toContain("vec2<f16>(unpack2x16float(u32(0x40003c00u)))");
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
    expect(compiled.wgsl).toContain("bg_warp_reduce_sum_int_32(x[i], 32u, local_id)");
    expect([...result.buffers.out as Int32Array]).toEqual([7]);
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

    expect(compiled.wgsl).toContain("= value.x;");
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

  it("coerces integer conditional expressions to WGSL bool predicates", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void integerPredicate(uint *out, int flag) {
  out[0] = flag ? 1u : 0u;
}`, { workgroupSize: [1, 1, 1] });
    const trueResult = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(1) }, scalars: { flag: 1 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const falseResult = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(1) }, scalars: { flag: 0 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("select(0u, 1u, (bg_uniforms.flag != 0))");
    expect([...trueResult.buffers.out as Uint32Array]).toEqual([1]);
    expect([...falseResult.buffers.out as Uint32Array]).toEqual([0]);
  });

  it("keeps hex masks with f digits as integer literals", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void hexMask(uint *out, uint value) {
  uint mask = 0xffffffffu;
  out[0] = (mask != 0xffffffffu) ? 1u : 0u;
  out[1] = (value == 0xffffffffu) ? 7u : 3u;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(2) }, scalars: { value: 0xffffffff } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("4294967295u");
    expect(compiled.wgsl).not.toContain("f32(4294967295u");
    expect([...result.buffers.out as Uint32Array]).toEqual([0, 7]);
  });

  it("lowers signed int hex masks through bit-preserving casts", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void signedHexMask(uint *out) {
  int mask = 0xffffffff;
  out[0] = uint(mask);
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("var mask: i32 = bitcast<i32>(4294967295u);");
    expect(compiled.wgsl).not.toContain("var mask: i32 = 0xffffffff;");
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

  it("casts mixed signedness assignment and modulo expressions", () => {
    const compiled = compileCudaLiteKernel(`
__device__ int rgbToInt(int x) { return x + 7; }
__global__ void signedness(uint *out, int *signedOut, int n) {
  uint tid = threadIdx.x;
  if (tid < 2u) {
    out[tid] = rgbToInt((int)tid);
    signedOut[tid] = tid % n;
  }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(2), signedOut: new Int32Array(2) }, scalars: { n: 2 } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("out[tid] = u32(rgbToInt");
    expect(compiled.wgsl).toContain("(tid % u32(bg_uniforms.n))");
    expect([...result.buffers.out as Uint32Array]).toEqual([7, 8]);
    expect([...result.buffers.signedOut as Int32Array]).toEqual([0, 1]);
  });

  it("casts float expressions stored into unsigned buffers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void floatToUint(const float *src, uint *out) {
  int idx = threadIdx.x;
  if (idx < 2) {
    out[idx] = src[idx] * 255.0f;
  }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { src: new Float32Array([0.5, 2]), out: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("out[u32(idx)] = u32((src[u32(idx)] * 255.0))");
    expect([...result.buffers.out as Uint32Array]).toEqual([127, 510]);
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
    expect(compiled.wgsl).toContain("bg_params[(index + 0u)] = bitcast<i32>(value.x)");
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

    expect(compiled.wgsl).toContain("bitcast<f32>(regs[");
    expect(compiled.wgsl).toContain("bitcast<u32>");
    expect([...result.buffers.out as Float32Array]).toEqual([5.5]);
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

    expect(compiled.wgsl).toContain("pack2x16float");
    expect(compiled.wgsl).toContain("unpack2x16float");
    expect([...result.buffers.out as Uint32Array]).toEqual([0x40003c00, 0x44004200]);
    expect([...result.buffers.sum as Float32Array]).toEqual([10]);
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

    expect(compiled.wgsl).toContain("bg_ptr_read_f16x2");
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

    expect(compiled.wgsl).toContain("bg_ptr_read_f32x2");
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).not.toContain("bg_ptr_read_f32");
    expect(compiled.wgsl).not.toContain("bg_ptr_write_f32");
    expect([...result.buffers.out as Float32Array]).toEqual([11, 22, 33, 44]);
  });

  it("emits pointer helpers for bool storage-pointer aliases", () => {
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

    expect(compiled.wgsl).toContain("fn bg_ptr_write_bool(buffer: u32, index: u32, value: bool)");
    expect(compiled.wgsl).toContain("bg_ptr_write_bool(0u");
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
    expect(compiled.wgsl).toMatch(/x\[[^\n]+ \+ 2u\] =/u);
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
    expect(compiled.wgsl).toContain("bg_ptr_read_f32x4");
    expect([...result.buffers.y as Float32Array]).toEqual([10, 20, 30, 41, 1, 2, 3, 5]);
  });

  it("lowers cp.async pointer-form copies to synchronous shared-memory copies", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void async_copy(const float *input, float *output) {
  __shared__ float tile[4];
  if (threadIdx.x < 1) {
    CP_ASYNC_CG(&tile[0], &input[0], 16);
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
    expect(compiled.wgsl).toContain("cp.async fence omitted: CP_ASYNC_WAIT_GROUP");
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

  it("lowers CUDA double cuRAND uniform to the deterministic browser RNG island", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void curandDoubleKernel(float *out) {
  curandState_t state;
  curand_init(7ULL, threadIdx.x, 0, &state);
  double x = curand_uniform_double(&state);
  out[threadIdx.x] = (float)x;
}`, { workgroupSize: [4, 1, 1], f64Mode: "f32" });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn bg_curand_uniform");
    expect([...result.buffers.out as Float32Array].every((value) => value > 0 && value <= 1)).toBe(true);
  });

  it("lowers CUDA normal cuRAND draws through the deterministic browser RNG island", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void curandNormalKernel(float *out) {
  curandState_t state;
  curand_init(17ULL, threadIdx.x, 3, &state);
  float x = curand_normal(&state);
  double y = curand_normal_double(&state);
  out[threadIdx.x] = x + (float)y;
}`, { workgroupSize: [4, 1, 1], f64Mode: "f32" });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn bg_curand_normal");
    expect([...result.buffers.out as Float32Array].every((value) => Number.isFinite(value))).toBe(true);
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
    expect(compiled.wgsl).toContain("bg_curand_init_storage(u32(bg_uniforms.seed), u32(tid), u32(0), &states[tid])");
    expect([...result.buffers.out as Float32Array].every((value) => Number.isFinite(value))).toBe(true);
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
    expect(compiled.wgsl).toContain("data[u32(idx)].x");
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
    expect(compiled.wgsl).toContain("A[u32(idx)] = c");
    expect([...result.buffers.A as Float32Array]).toEqual([-7, 16, -11, 52]);
  });

  it("passes cufftComplex values through CUDA float2 helper functions", () => {
    const compiled = compileCudaLiteKernel(`
static __device__ __host__ inline float2 ComplexScale(float2 a, float s) {
  return make_float2(a.x * s, a.y * s);
}
static __device__ __host__ inline float2 ComplexMul(float2 a, float2 b) {
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__global__ void pointwise(cufftComplex *a, cufftComplex *b, float scale) {
  int i = threadIdx.x;
  a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          a: new Float32Array([1, 2, 3, 4]),
          b: new Float32Array([5, 6, 7, 8]),
        },
        scalars: { scale: 0.5 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn ComplexMul");
    expect(compiled.wgsl).toContain("a[u32(i)] = ComplexScale");
    expect(compiled.wgsl).not.toContain("f32(vec2<f32>");
    expect([...result.buffers.a as Float32Array]).toEqual([-3.5, 8, -5.5, 26]);
  });

  it("lowers cuComplex helper builtins natively", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void cuComplexHelpers(cuComplex *a, cuFloatComplex *b, float *out) {
  int i = threadIdx.x;
  cuComplex x = make_cuComplex(a[i].x, a[i].y);
  cuFloatComplex y = make_cuFloatComplex(b[i].x, b[i].y);
  cuComplex z = cuCaddf(cuCmulf(x, y), cuConjf(y));
  cuComplex q = cuCdivf(z, make_cuComplex(2.0f, 0.0f));
  a[i] = cuCsubf(q, make_cuComplex(1.0f, -1.0f));
  out[i] = cuCabsf(a[i]) + cuCrealf(a[i]) + cuCimagf(a[i]);
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          a: new Float32Array([1, 2, 3, 4]),
          b: new Float32Array([5, 6, 7, 8]),
          out: new Float32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(compiled.wgsl).toContain("vec2<f32>(f32");
    expect(compiled.wgsl).toContain("fn bg_cuCabsf");
    expect(compiled.wgsl).toContain("fn bg_cuCdivf");
    expect([...result.buffers.a as Float32Array]).toEqual([-2, 6, -3, 23]);
    expect([...result.buffers.out as Float32Array]).toEqual([...Float32Array.from([Math.hypot(-2, 6) + 4, Math.hypot(-3, 23) + 20])]);
  });

  it("uses scaled cuComplex abs/div lowering for large finite values", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void cuComplexRobust(cuComplex *a, cuComplex *b, float *out) {
  int i = threadIdx.x;
  cuComplex q = cuCdivf(a[i], b[i]);
  out[i * 3] = cuCabsf(b[i]) * 1.0e-20f;
  out[i * 3 + 1] = cuCrealf(q) * 1.0e20f;
  out[i * 3 + 2] = cuCimagf(q) * 1.0e20f;
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          a: new Float32Array([1, 1, -1, 2]),
          b: new Float32Array([1.0e20, 1.0e20, 1.0e20, -1.0e20]),
          out: new Float32Array(6),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );
    const out = [...result.buffers.out as Float32Array];

    expect(compiled.wgsl).toContain("fn bg_cuCabsf");
    expect(compiled.wgsl).toContain("fn bg_cuCdivf");
    expect(out[0]).toBeCloseTo(Math.SQRT2, 5);
    expect(out[1]).toBeCloseTo(1, 5);
    expect(out[2]).toBeCloseTo(0, 5);
    expect(out[3]).toBeCloseTo(Math.SQRT2, 5);
    expect(out[4]).toBeCloseTo(-1.5, 5);
    expect(out[5]).toBeCloseTo(0.5, 5);
  });

  it("lowers cuComplex fused multiply-add natively", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void cuComplexFma(cuComplex *a, cuComplex *b, cuComplex *d) {
  int i = threadIdx.x;
  d[i] = cuCfmaf(a[i], b[i], d[i]);
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          a: new Float32Array([1, 2, 3, 4]),
          b: new Float32Array([5, 6, 7, 8]),
          d: new Float32Array([9, 10, 11, 12]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(compiled.wgsl).toContain("d[u32(i)] = vec2<f32>");
    expect(compiled.wgsl).not.toContain("unsupported CUDA-lite call");
    expect([...result.buffers.d as Float32Array]).toEqual([2, 26, 0, 64]);
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

  it("lowers output-only inline PTX lane id statements", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void laneId(int *out) {
  int idx = threadIdx.x;
  unsigned int laneid;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
  out[idx] = laneid;
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("% 32");
    expect([...result.buffers.out as Int32Array]).toEqual([0, 1, 2, 3]);
  });

  it("lowers output-only inline PTX warp id statements", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void warpId(uint *out) {
  int idx = threadIdx.x;
  unsigned int warp;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp));
  out[idx] = warp;
}`, { workgroupSize: [64, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(64) } },
      { gridDim: [1, 1, 1], blockDim: [64, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
    expect(compiled.wgsl).toContain("/ 32");
    expect([...result.buffers.out as Uint32Array]).toEqual([
      ...new Array(32).fill(0),
      ...new Array(32).fill(1),
    ]);
  });

  it("lowers output-only inline PTX lanemask_lt statements", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void laneMaskLt(uint *out) {
  int idx = threadIdx.x;
  unsigned int mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  out[idx] = mask;
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("1u <<");
    expect([...result.buffers.out as Uint32Array]).toEqual([0, 1, 3, 7]);
  });

  it("lowers output-only inline PTX globaltimer statements deterministically", () => {
    const compiled = compileCudaLiteKernel(`
__device__ unsigned long long read_clock() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__global__ void globalTimer(uint *out) {
  out[threadIdx.x] = (uint)read_clock();
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("workgroup_id.x");
    expect(compiled.wgsl).toContain("local_id.x");
    expect([...result.buffers.out as Uint32Array]).toEqual([0, 1, 2, 3]);
  });

  it("lowers inline PTX bfind.u32 statements", () => {
    const compiled = compileCudaLiteKernel(`
__device__ unsigned int bfind(unsigned int word) {
  unsigned int ret;
  asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
  return ret;
}
__global__ void bfindKernel(uint *out, uint *input) {
  int idx = threadIdx.x;
  out[idx] = bfind(input[idx]);
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(4), input: new Uint32Array([0, 1, 16, 0x80000000]) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("countLeadingZeros");
    expect([...result.buffers.out as Uint32Array]).toEqual([0xffffffff, 0, 4, 31]);
  });

  it("lowers CUDA u8x4 SAD intrinsics and inline PTX", () => {
    const compiled = compileCudaLiteKernel(`
__device__ unsigned int sad_ptx(unsigned int a, unsigned int b, unsigned int c) {
  unsigned int ret;
  asm("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;"
      : "=r"(ret)
      : "r"(a), "r"(b), "r"(c));
  return ret;
}
__global__ void sad4(uint *out, uint *a, uint *b) {
  int idx = threadIdx.x;
  out[idx] = __usad4(a[idx], b[idx], 7u);
  out[idx + 2] = sad_ptx(a[idx], b[idx], 5u);
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          out: new Uint32Array(4),
          a: new Uint32Array([0x01020304, 0xff001020]),
          b: new Uint32Array([0x05010108, 0x0f000020]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("0xffu");
    expect([...result.buffers.out as Uint32Array]).toEqual([18, 263, 16, 261]);
  });

  it("lowers adjacent-string inline PTX mma carriers with multiple outputs", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void mmaCarrier(uint *out) {
  uint a0 = 0x3c003c00u;
  uint a1 = 0x3c003c00u;
  uint a2 = 0x3c003c00u;
  uint a3 = 0x3c003c00u;
  uint b0 = 0x40004000u;
  uint b1 = 0x40004000u;
  uint c = 0u;
  uint d = 0u;
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, "
    "{%2, %3}, {%4, %5}, {%6, %7};\\n"
    : "=r"(c), "=r"(d)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c), "r"(d));
  out[0] = c;
  out[1] = d;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("pack2x16float");
    expect([...result.buffers.out as Uint32Array]).toEqual([0x40004000, 0x40004000]);
  });

  it("bitcasts f32 inline PTX mma accumulator carriers stored in integer regs", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void mmaF32Carrier(uint *out, float *asFloat) {
  uint a0 = 0x3c003c00u;
  uint a1 = 0x3c003c00u;
  uint a2 = 0x3c003c00u;
  uint a3 = 0x3c003c00u;
  uint b0 = 0x40004000u;
  uint b1 = 0x40004000u;
  uint d0 = __float_as_uint(1.5f);
  uint d1 = __float_as_uint(2.5f);
  uint d2 = __float_as_uint(3.5f);
  uint d3 = __float_as_uint(4.5f);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\\n"
    : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(d0), "r"(d1), "r"(d2), "r"(d3));
  out[0] = d0;
  asFloat[0] = __uint_as_float(d0);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(1), asFloat: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bitcast<f32>(d0)");
    expect(compiled.wgsl).toContain("d0 = bitcast<u32>");
    expect([...result.buffers.out as Uint32Array]).toEqual([floatBits(5.5)]);
    expect([...result.buffers.asFloat as Float32Array]).toEqual([5.5]);
  });

  it("lowers multi-output ldmatrix carriers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void ldmatrixCarrier(uint *out) {
  uint a = 0u;
  uint b = 0u;
  uint addr = 5u;
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\\n"
    : "=r"(a), "=r"(b)
    : "r"(addr));
  out[0] = a;
  out[1] = b;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("u32(addr) + 0u");
    expect([...result.buffers.out as Uint32Array]).toEqual([5, 7]);
  });

  it("parses inline PTX with empty output sections as an unsupported semantic gap", () => {
    expect(() => compileCudaLiteKernel(`
__global__ void emptyAsm(float *out, float *in) {
  asm volatile("cp.async.commit_group;\\n" ::);
  out[threadIdx.x] = in[threadIdx.x];
}`)).toThrow(/unsupported-inline-asm/u);
  });

  it("parses inline PTX clobber sections as an unsupported semantic gap", () => {
    expect(() => compileCudaLiteKernel(`
__global__ void clobberAsm(float *out, float *in) {
  asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");
  out[threadIdx.x] = in[threadIdx.x];
}`)).toThrow(/unsupported-inline-asm/u);
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

    expect(backendIr(compiled).name).toBe("anonymous_kernel_1");
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

  it("lowers cudaSurfaceObject_t surf3Dwrite to z-linearized storage", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void surfaceWrite3d(cudaSurfaceObject_t outputSurf) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = blockIdx.z;
  surf3Dwrite(float(x + y * 10 + z * 100), outputSurf, x * sizeof(float), y, z);
}`, { workgroupSize: [2, 2, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {},
        surfaces: { outputSurf: { width: 2, height: 2, data: new Float32Array(8) } },
      },
      { gridDim: [1, 1, 2], blockDim: [2, 2, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {},
        surfaces: { outputSurf: { width: 2, height: 2, data: new Float32Array(8) } },
      },
      { gridDim: [1, 1, 2], blockDim: [2, 2, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var z: i32 = i32(workgroup_id.z);");
    expect(compiled.wgsl).toContain("let bg_index = ((bg_z * i32(bg_uniforms.outputSurf_height)) + bg_y) * i32(bg_uniforms.outputSurf_width) + bg_x;");
    expect([...result.buffers.outputSurf as Float32Array]).toEqual([0, 1, 10, 11, 100, 101, 110, 111]);
    expect([...semanticResult.buffers.outputSurf as Float32Array]).toEqual([0, 1, 10, 11, 100, 101, 110, 111]);
  });

  it("lowers cudaSurfaceObject_t surf2DLayeredwrite to z-linearized layer storage", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void surfaceLayeredWrite(cudaSurfaceObject_t outputSurf) {
  surf2DLayeredwrite(23.0f, outputSurf, 1 * sizeof(float), 1, 1);
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("let bg_z = 1;");
    expect(compiled.wgsl).toContain("let bg_index = ((bg_z * i32(bg_uniforms.outputSurf_height)) + bg_y) * i32(bg_uniforms.outputSurf_width) + bg_x;");
    expect(compiled.wgsl).toContain("outputSurf[bg_index] = 23.0;");
  });

  it("lowers cudaSurfaceObject_t surf2DLayeredread to z-linearized layer storage", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float read_layer(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<float>(surfaceArg, 1 * sizeof(float), row, layer);
}

__global__ void surfaceLayeredRead(cudaSurfaceObject_t surf, float *out) {
  if (threadIdx.x == 0) {
    float value = 0.0f;
    surf2DLayeredread(&value, surf, 0, 1, 1);
    out[0] = value;
    out[1] = surf2DLayeredread<float>(surf, 1 * sizeof(float), 1, 1);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(2) },
        surfaces: { surf: { width: 2, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Float32Array]).toEqual([7, 8]);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("bg_sem_surf2dread_surf(0, 1, 1)");
    expect(compiled.wgsl).toContain("bg_sem_surf2dread_surf((1 * 4), 1, 1)");

    const helperCompiled = compileCudaLiteKernel(`
__device__ float read_layer(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<float>(surfaceArg, 1 * sizeof(float), row, layer);
}

__global__ void surfaceLayeredRead(cudaSurfaceObject_t surf, float *out) {
  out[0] = read_layer(surf, 1, 1);
}`, { workgroupSize: [1, 1, 1] });
    const helperResult = runCompiledKernelReference(
      helperCompiled,
      {
        buffers: { out: new Float32Array(1) },
        surfaces: { surf: { width: 2, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const helperSemanticResult = runCompiledKernelSemanticReference(
      helperCompiled,
      {
        buffers: { out: new Float32Array(1) },
        surfaces: { surf: { width: 2, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect(canRunCompiledKernelSemanticReference(helperCompiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(helperCompiled.kernelIr)).toBe(true);
    expect(helperCompiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(helperCompiled.wgsl).toContain("fn bg_sem_surf2dread(surface: u32, x_bytes: i32, y: i32, z: i32) -> f32");
    expect(helperCompiled.wgsl).toContain("read_layer(0u, 1, 1)");
    expect(helperCompiled.wgsl).toContain("bg_sem_surf2dread(surfaceArg, (1 * 4), row, layer)");
    expect([...helperResult.buffers.out as Float32Array]).toEqual([8]);
    expect([...helperSemanticResult.buffers.out as Float32Array]).toEqual([8]);
  });

  it("lowers cudaSurfaceObject_t surf3Dread to z-linearized layer storage", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float read_z(cudaSurfaceObject_t surfaceArg, int row, int z) {
  return surf3Dread<float>(surfaceArg, 1 * sizeof(float), row, z);
}

__global__ void surfaceRead3d(cudaSurfaceObject_t surf, float *out) {
  if (threadIdx.x == 0) {
    float value = 0.0f;
    surf3Dread(&value, surf, 0, 1, 1);
    out[0] = value;
    out[1] = surf3Dread<float>(surf, 1 * sizeof(float), 1, 1);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(2) },
        surfaces: { surf: { width: 2, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Float32Array]).toEqual([7, 8]);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("bg_sem_surf2dread_surf(0, 1, 1)");
    expect(compiled.wgsl).toContain("bg_sem_surf2dread_surf((1 * 4), 1, 1)");

    const helperCompiled = compileCudaLiteKernel(`
__device__ float read_z(cudaSurfaceObject_t surfaceArg, int row, int z) {
  return surf3Dread<float>(surfaceArg, 1 * sizeof(float), row, z);
}

__global__ void surfaceRead3d(cudaSurfaceObject_t surf, float *out) {
  out[0] = read_z(surf, 1, 1);
}`, { workgroupSize: [1, 1, 1] });
    expect(helperCompiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(helperCompiled.wgsl).toContain("bg_sem_surf2dread(surfaceArg, (1 * 4), row, z)");
  });

  it("lowers layered and 3D vector surface reads lane-wise through z-linearized storage", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void surfaceVectorLayeredRead(cudaSurfaceObject_t surf, float *out) {
  if (threadIdx.x == 0) {
    float4 layeredPointer;
    float4 zPointer;
    surf2DLayeredread(&layeredPointer, surf, 0, 0, 1);
    surf3Dread(&zPointer, surf, 0, 0, 1);
    float4 layeredReturn = surf2DLayeredread<float4>(surf, 0, 0, 1);
    float4 zReturn = surf3Dread<float4>(surf, 0, 0, 1);
    out[0] = layeredPointer.x + layeredReturn.x;
    out[1] = layeredPointer.y + layeredReturn.y;
    out[2] = zPointer.z + zReturn.z;
    out[3] = zPointer.w + zReturn.w;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(4) },
        surfaces: { surf: { width: 4, height: 1, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Float32Array(4) },
        surfaces: { surf: { width: 4, height: 1, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...result.buffers.out as Float32Array]).toEqual([10, 12, 14, 16]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([10, 12, 14, 16]);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("layeredPointer = select(vec4<f32>(), vec4<f32>(f32(bg_sem_surf2dread_surf((0 + 0), 0, 1)), f32(bg_sem_surf2dread_surf((0 + 4), 0, 1)), f32(bg_sem_surf2dread_surf((0 + 8), 0, 1)), f32(bg_sem_surf2dread_surf((0 + 12), 0, 1))), (0 >= 0 && (0 % 4) == 0))");
    expect(compiled.wgsl).toContain("var layeredReturn: vec4<f32> = select(vec4<f32>(), vec4<f32>(f32(bg_sem_surf2dread_surf((0 + 0), 0, 1)), f32(bg_sem_surf2dread_surf((0 + 4), 0, 1)), f32(bg_sem_surf2dread_surf((0 + 8), 0, 1)), f32(bg_sem_surf2dread_surf((0 + 12), 0, 1))), (0 >= 0 && (0 % 4) == 0))");

    const helperCompiled = compileCudaLiteKernel(`
__device__ float4 read_layer_vec(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<float4>(surfaceArg, 0, row, layer);
}

__device__ float4 read_z_vec(cudaSurfaceObject_t surfaceArg, int row, int z) {
  return surf3Dread<float4>(surfaceArg, 0, row, z);
}

__global__ void surfaceVectorLayeredRead(cudaSurfaceObject_t surf, float *out) {
  float4 layeredReturn = read_layer_vec(surf, 0, 1);
  float4 zReturn = read_z_vec(surf, 0, 1);
  out[0] = layeredReturn.x + zReturn.x;
}`, { workgroupSize: [1, 1, 1] });
    expect(helperCompiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(helperCompiled.wgsl).toContain("return select(vec4<f32>(), vec4<f32>(f32(bg_sem_surf2dread(surfaceArg, (0 + 0), row, layer)), f32(bg_sem_surf2dread(surfaceArg, (0 + 4), row, layer)), f32(bg_sem_surf2dread(surfaceArg, (0 + 8), row, layer)), f32(bg_sem_surf2dread(surfaceArg, (0 + 12), row, layer))), (0 >= 0 && (0 % 4) == 0))");
    expect(helperCompiled.wgsl).toContain("return select(vec4<f32>(), vec4<f32>(f32(bg_sem_surf2dread(surfaceArg, (0 + 0), row, z)), f32(bg_sem_surf2dread(surfaceArg, (0 + 4), row, z)), f32(bg_sem_surf2dread(surfaceArg, (0 + 8), row, z)), f32(bg_sem_surf2dread(surfaceArg, (0 + 12), row, z))), (0 >= 0 && (0 % 4) == 0))");
  });

  it("lowers vector surface writes through helper params lane-wise in semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void write_surface_vec(cudaSurfaceObject_t surfaceArg, float4 value) {
  surf2Dwrite(value, surfaceArg, 0, 0);
}

__device__ uint read_surface_value(cudaSurfaceObject_t surfaceArg) {
  return surf2Dread<unsigned int>(surfaceArg, 4, 0);
}

__global__ void surfaceVectorHelperRoundtrip(cudaSurfaceObject_t dst, float *out) {
  float4 value = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  write_surface_vec(dst, value);
  out[0] = (float)read_surface_value(dst);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(1) },
        surfaces: {
          dst: { width: 4, height: 1, data: new Float32Array(4) },
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Float32Array(1) },
        surfaces: {
          dst: { width: 4, height: 1, data: new Float32Array(4) },
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn bg_sem_surf2dread(surface: u32, x_bytes: i32, y: i32, z: i32) -> f32");
    expect(compiled.wgsl).toContain("fn bg_sem_surf2dwrite(surface: u32, value: f32, x_bytes: i32, y: i32, z: i32)");
    expect(compiled.wgsl).toContain("var value: vec4<f32> = vec4<f32>(f32(1.0), f32(2.0), f32(3.0), f32(4.0));");
    expect(compiled.wgsl).toContain("write_surface_vec(0u, value)");
    expect(compiled.wgsl).toContain("bg_sem_surf2dwrite(surfaceArg, (value).x, (0 + 0), 0, 0);");
    expect(compiled.wgsl).toContain("bg_sem_surf2dwrite(surfaceArg, (value).w, (0 + 12), 0, 0);");
    expect(compiled.wgsl).toContain("bg_sem_surf2dread(surfaceArg");
    expect(compiled.wgsl).not.toContain("bg_sem_surf2dwrite_surfaceArg");
    expect(compiled.wgsl).not.toContain("; 0;");
    expect([...result.buffers.dst as Float32Array]).toEqual([1, 2, 3, 4]);
    expect([...semanticResult.buffers.dst as Float32Array]).toEqual([1, 2, 3, 4]);
    expect([...result.buffers.out as Float32Array]).toEqual([2]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([2]);
  });

  it("lowers scalar surface write helper params through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void write_value(cudaSurfaceObject_t surfaceArg, float value) {
  surf2Dwrite(value, surfaceArg, 4, 0);
}

__global__ void surfaceHelperWrite(cudaSurfaceObject_t surf) {
  write_value(surf, 17.0f);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {},
        surfaces: { surf: { width: 2, height: 1, data: new Float32Array(2) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {},
        surfaces: { surf: { width: 2, height: 1, data: new Float32Array(2) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn bg_sem_surf2dwrite(surface: u32, value: f32, x_bytes: i32, y: i32, z: i32)");
    expect(compiled.wgsl).toContain("write_value(0u, 17.0)");
    expect(compiled.wgsl).toContain("bg_sem_surf2dwrite(surfaceArg, value, 4, 0, 0);");
    expect([...result.buffers.surf as Float32Array]).toEqual([0, 17]);
    expect([...semanticResult.buffers.surf as Float32Array]).toEqual([0, 17]);
  });

  it("lowers vector surf2Dread pointer-form calls lane-wise", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void surfaceVectorRead(cudaSurfaceObject_t surf, float *out) {
  float4 value;
  surf2Dread(&value, surf, 0, 0);
  out[0] = value.x;
  out[1] = value.y;
  out[2] = value.z;
  out[3] = value.w;
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: { out: new Float32Array(4) },
      surfaces: { surf: { width: 4, height: 1, data: new Float32Array([1, 2, 3, 4]) } },
    };
    const launch = { gridDim: [1, 1, 1], blockDim: [1, 1, 1] } as const;
    const result = runCompiledKernelReference(compiled, input, launch);
    const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("value = select(vec4<f32>(), vec4<f32>(f32(bg_sem_surf2dread_surf((0 + 0), 0, 0)), f32(bg_sem_surf2dread_surf((0 + 4), 0, 0)), f32(bg_sem_surf2dread_surf((0 + 8), 0, 0)), f32(bg_sem_surf2dread_surf((0 + 12), 0, 0))), (0 >= 0 && (0 % 4) == 0))");
    expect([...result.buffers.out as Float32Array]).toEqual([1, 2, 3, 4]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 2, 3, 4]);
  });

  it("preserves templated vector surf2Dread return type in device helpers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float4 read_surface_vec_return(cudaSurfaceObject_t surfaceArg) {
  return surf2Dread<float4>(surfaceArg, 0, 0);
}

__global__ void surfaceHelperVectorRead(cudaSurfaceObject_t surf, float *out) {
  float4 value = read_surface_vec_return(surf);
  out[0] = value.x;
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: { out: new Float32Array(1) },
      surfaces: { surf: { width: 4, height: 1, data: new Float32Array([1, 2, 3, 4]) } },
    };
    const launch = { gridDim: [1, 1, 1], blockDim: [1, 1, 1] } as const;
    const result = runCompiledKernelReference(compiled, input, launch);
    const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("return select(vec4<f32>(), vec4<f32>(f32(bg_sem_surf2dread(surfaceArg, (0 + 0), 0, 0)), f32(bg_sem_surf2dread(surfaceArg, (0 + 4), 0, 0)), f32(bg_sem_surf2dread(surfaceArg, (0 + 8), 0, 0)), f32(bg_sem_surf2dread(surfaceArg, (0 + 12), 0, 0))), (0 >= 0 && (0 % 4) == 0))");
    expect(compiled.wgsl).not.toContain("return f32(vec4<f32>");
    expect([...result.buffers.out as Float32Array]).toEqual([1]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([1]);
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

  it("ignores device-side launches in unreachable helpers for selected kernels", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(float *x) { x[threadIdx.x] += 1.0f; }

__device__ void unused_launch(float *x) {
  dim3 grid(1);
  dim3 block(1);
  child<<<grid, block>>>(x);
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
    expect(backendIr(compiled).name).toBe("selected");
    expect(compiled.wgsl).not.toContain("unused_launch");
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

  it("ignores f64 and unsupported inline asm compatibility gaps in unreachable helpers for selected kernels", () => {
    const compiled = compileCudaLiteKernel(`
__device__ double unused_double(double value) {
  double acc = value + 1.0;
  return acc;
}

__device__ void unused_inline_asm(int *out) {
  asm volatile("cp.async.commit_group;\\n" ::);
  out[0] = 1;
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

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-f64");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
    expect([...result.buffers.x as Float32Array]).toEqual([5]);
    expect(compiled.wgsl).not.toContain("unused_double");
    expect(compiled.wgsl).not.toContain("unused_inline_asm");
  });

  it("ignores unsupported texture and surface calls in unreachable helpers for selected kernels", () => {
    const compiled = compileCudaLiteKernel(`
texture<float, cudaTextureType2D, cudaReadModeElementType> tex;

__device__ void unused_texture_surface(float *out) {
  out[0] = tex2D<double>(tex, 0.0f, 0.0f);
  surf2Dwrite(out[0], out, 0, 0);
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

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-texture");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-surface");
    expect([...result.buffers.x as Float32Array]).toEqual([5]);
    expect(compiled.wgsl).not.toContain("unused_texture_surface");
  });

  it("ignores unreferenced constants used only by unreachable helpers for selected kernels", () => {
    const compiled = compileCudaLiteKernel(`
__constant__ float unused_coeffs[2];

__device__ float unused_constant_helper() {
  return unused_coeffs[0] + unused_coeffs[1];
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

    expect(backendIr(compiled).constants.map((constant) => constant.name)).not.toContain("unused_coeffs");
    expect(compiled.kernelIr.memory.map((symbol) => symbol.name)).not.toContain("unused_coeffs");
    expect([...result.buffers.x as Float32Array]).toEqual([5]);
    expect(compiled.wgsl).not.toContain("unused_coeffs");
    expect(compiled.wgsl).not.toContain("unused_constant_helper");
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

  it("removes unreachable helper bodies from lowered IR for selected kernels", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float unused_helper(float value) {
  return value * 10.0f;
}

__device__ float selected_helper(float value) {
  return value + 2.0f;
}

__global__ void selected(float *x) {
  if (threadIdx.x == 0) {
    x[0] = selected_helper(x[0]);
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

    expect(backendIr(compiled).functions.map((fn) => fn.name)).toEqual(["selected_helper"]);
    expect(compiled.kernelIr.functions.map((fn) => fn.name)).toEqual(["selected_helper"]);
    expect([...result.buffers.x as Float32Array]).toEqual([5]);
    expect(compiled.wgsl).toContain("selected_helper");
    expect(compiled.wgsl).not.toContain("unused_helper");
  });

  it("ignores unreferenced half globals for selected kernels without shader-f16", () => {
    const compiled = compileCudaLiteKernel(`
__constant__ half unused_coeffs[2];
__device__ half unused_state[2];

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

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("missing-feature-shader-f16");
    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect(backendIr(compiled).constants.map((constant) => constant.name)).not.toContain("unused_coeffs");
    expect(backendIr(compiled).deviceGlobals.map((global) => global.name)).not.toContain("unused_state");
    expect(compiled.kernelIr.memory.map((symbol) => symbol.name)).not.toContain("unused_coeffs");
    expect(compiled.kernelIr.memory.map((symbol) => symbol.name)).not.toContain("unused_state");
    expect([...result.buffers.x as Float32Array]).toEqual([5]);
    expect(compiled.wgsl).not.toContain("unused_coeffs");
    expect(compiled.wgsl).not.toContain("unused_state");
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

    expect(compiled.loweringPlan.canDirectLowerToWgsl).toBe(false);
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
    const input = { buffers: { x: new Float32Array([1, 2]) }, scalars: { n: 2 } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const plan = createCudaHostDynamicLaunchPlan(compiled, input, launch);
    const detachedPlan = createCudaHostDynamicLaunchPlan({ ...compiled }, input, launch);

    expect(plan.supported).toBe(true);
    expect(detachedPlan.supported).toBe(true);
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

  it("does not elide dynamic launches that write external buffers through system atomics", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void child(int *out) {
  if (threadIdx.x == 0) {
    atomicSub_system(&out[0], 1);
    atomicMax_system(&out[1], 7);
    atomicCAS_system(&out[2], 3, 5);
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
      { buffers: { out: new Int32Array([9, 2, 3]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      { compileKernel: (source, options = {}) => compileCudaLiteKernel(source, options) },
    );

    expect(executionPlan).toMatchObject({
      supported: true,
      kind: "host-dynamic-launch",
    });
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

    expect(backendIr(child).name).toBe("childKernel");
    expect(compiled.wgsl).not.toContain("fn childKernel(");
    expect(child.wgsl).not.toContain("fn childKernel(");
    expect(plan.supported).toBe(true);
    expect(plan.launches).toHaveLength(2);
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
    expect(canEmitSemanticKernelIrWgsl(child.kernelIr, { pointerBaseOffsets: plan.launches[0]!.pointerBaseOffsets })).toBe(true);
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

  it("treats standalone cudaDeviceSynchronize as a WebGPU-safe no-op", () => {
    const source = `
__device__ float tunable_helper(float x) { return x + 1.0f; }
__global__ void syncOnly(float *x) {
  cudaStream_t stream;
  cudaEvent_t event;
  if (threadIdx.x < 1) {
    int attr = cudaFuncSetAttribute(tunable_helper, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
    int cache = -1;
    int bank = -1;
    int cacheStatus = cudaDeviceGetCacheConfig(&cache);
    int bankStatus = cudaDeviceGetSharedMemConfig(&bank);
    int setCacheStatus = cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    int setBankStatus = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    int funcCacheStatus = cudaFuncSetCacheConfig(tunable_helper, cudaFuncCachePreferL1);
    int funcBankStatus = cudaFuncSetSharedMemConfig(tunable_helper, cudaSharedMemBankSizeFourByte);
    int profilerStart = cudaProfilerStart();
    int profilerStop = cudaProfilerStop();
    cudaStreamCaptureMode captureMode = cudaStreamCaptureModeRelaxed;
    int exchangeMode = cudaThreadExchangeStreamCaptureMode(&captureMode);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0);
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    int streamReady = cudaStreamQuery(stream);
    cudaDeviceSynchronize();
    cudaEventRecord(event, stream);
    int eventRecordWithFlags = cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal);
    cudaStreamWaitEvent(stream, event, 0);
    int eventReady = cudaEventQuery(event);
    int freed = cudaFree(x);
    int freedAsync = cudaFreeAsync(x, stream);
    cudaEventSynchronize(event);
    cudaStreamSynchronize(stream);
    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
    x[0] = tunable_helper(8.0f) + (float)(attr + cache + bank + cacheStatus + bankStatus + setCacheStatus + setBankStatus + funcCacheStatus + funcBankStatus + profilerStart + profilerStop + exchangeMode + captureMode + streamReady + eventRecordWithFlags + eventReady + freed + freedAsync + (int)(cudaEventWaitExternal - cudaEventRecordExternal));
  }
}`;
    const compiled = compileCudaLiteKernel(source, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([0]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
    expect(compiled.wgsl).toContain("var attr: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var cacheStatus: i32 = 0;");
    expect(compiled.wgsl).toContain("var bankStatus: i32 = 0;");
    expect(compiled.wgsl).toContain("var setCacheStatus: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var funcCacheStatus: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var profilerStart: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var profilerStop: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var exchangeMode: i32 = 0;");
    expect(compiled.wgsl).toContain("captureMode = 0;");
    expect(compiled.wgsl).toContain("var streamReady: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var eventRecordWithFlags: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var eventReady: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var freed: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var freedAsync: i32 = i32(0);");
    expect(createCudaRuntimePlan(compiled).operations.map((operation) => operation.kind).every((kind) => kind === "device-sync")).toBe(true);
    expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { x: new Float32Array([0]) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
    expect([...result.buffers.x as Float32Array]).toEqual([9]);
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
    uint captureId = 4u;
    uint dependencyCount = 3u;
    cudaStreamCreate(&stream);
    int deviceStatus = cudaStreamGetDevice(stream, &device);
    int idStatus = cudaStreamGetId(stream, &streamId);
    int captureStatusResult = cudaStreamIsCapturing(stream, &captureStatus);
    int captureInfoResult = cudaStreamGetCaptureInfo(stream, &captureInfoStatus, &captureId, &graph, NULL, NULL, &dependencyCount);
    cudaStreamDestroy(stream);
    out[0] = (float)(device + (int)streamId + captureStatus + captureInfoStatus + (int)captureId + (int)graph + (int)dependencyCount + deviceStatus + idStatus + captureStatusResult + captureInfoResult);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array([-1]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
    expect(compiled.wgsl).toContain("device = 0;");
    expect(compiled.wgsl).toContain("streamId = 0;");
    expect(compiled.wgsl).toContain("captureStatus = 0;");
    expect(compiled.wgsl).toContain("captureInfoStatus = 0;");
    expect(compiled.wgsl).toContain("dependencyCount = 0;");
    expect(createCudaRuntimePlan(compiled).operations.map((operation) => operation.kind).every((kind) => kind === "device-sync")).toBe(true);
    expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { out: new Float32Array([-1]) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
    expect([...result.buffers.out as Float32Array]).toEqual([0]);
  });

  it("treats CUDA unified-memory advice and prefetch calls as WebGPU-safe no-ops", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void unified_memory_hints(float *x, int n) {
  cudaStream_t stream;
  if (threadIdx.x < 1) {
    cudaStreamCreate(&stream);
    int cache = cudaCtxResetPersistingL2Cache();
    int advise = cudaMemAdvise(x, sizeof(float) * n, cudaMemAdviseSetPreferredLocation, 0);
    int prefetch = cudaMemPrefetchAsync(x, sizeof(float) * n, 0, stream);
    int attach = cudaStreamAttachMemAsync(stream, x, sizeof(float) * n, cudaMemAttachSingle);
    cudaStreamDestroy(stream);
    x[0] = 7.0f + (float)(cache + advise + prefetch + attach + (int)(cudaMemAttachGlobal + cudaMemAttachHost - 3u));
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([0]) }, scalars: { n: 1 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
    expect(compiled.wgsl).toContain("var cache: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var advise: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var prefetch: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var attach: i32 = i32(0);");
    expect(createCudaRuntimePlan(compiled).operations.map((operation) => operation.kind).every((kind) => kind === "device-sync")).toBe(true);
    expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { x: new Float32Array([0]) }, scalars: { n: 1 } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
    expect([...result.buffers.x as Float32Array]).toEqual([7]);
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
    expect(compiled.wgsl).toContain("var status: i32 = (-1);");
    expect(compiled.wgsl).toContain("status = 0;");
    expect(compiled.wgsl).toContain("ms = 0.0;");
    expect(compiled.wgsl).toContain("out[2] = f32(0.0);");
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
    expect(compiled.wgsl).toContain("out[3] = i32(0);");
    expect(compiled.wgsl).toContain("out[4] = i32(0);");
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
    expect(compiled.wgsl).toContain("var flagStatus: i32 = (-1);");
    expect(compiled.wgsl).toContain("flagStatus = 0;");
    expect(compiled.wgsl).toMatch(/\bflags = 0u?;/u);
    expect(compiled.wgsl).toContain("priority = 0;");
    expect(compiled.wgsl).toContain("flagsOut[1] = 0u;");
    expect(compiled.wgsl).toContain("priorityOut[2] = i32(0);");
    expect(createCudaWebGpuExecutionPlan(
      compiled,
      { buffers: { flagsOut: new Uint32Array(2), priorityOut: new Int32Array(3) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    ).supported).toBe(true);
    expect([...result.buffers.flagsOut as Uint32Array]).toEqual([0, 0]);
    expect([...result.buffers.priorityOut as Int32Array]).toEqual([0, 0, 0]);
  });

  it("models CUDA device flag queries and setters", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void device_flags(uint *out) {
  if (threadIdx.x < 1) {
    uint flags = 99u;
    int status = -1;
    status = cudaGetDeviceFlags(&flags);
    int setStatus = cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost);
    cudaGetDeviceFlags(&out[3]);
    out[0] = flags;
    out[1] = (uint)status;
    out[2] = (uint)setStatus;
    out[4] = cudaDeviceLmemResizeToMax;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(5).fill(77) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
    expect(compiled.wgsl).toContain("var status: i32 = (-1);");
    expect(compiled.wgsl).toContain("status = 0;");
    expect(compiled.wgsl).toContain("var setStatus: i32 = i32(0);");
    expect(compiled.wgsl).toContain("flags = 0;");
    expect(compiled.wgsl).toContain("out[3] = 0u;");
    expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { out: new Uint32Array(5) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
    expect([...result.buffers.out as Uint32Array]).toEqual([0, 0, 0, 0, 16]);
  });

  it("treats CUDA device selection and last-error status as WebGPU-safe no-ops", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void runtime_status(int *out) {
  if (threadIdx.x < 1) {
    int set = cudaSetDevice(0);
    int reset = cudaDeviceReset();
    int last = cudaGetLastError();
    int peek = cudaPeekAtLastError();
    out[0] = set + reset + last + peek + (last == cudaSuccess ? 0 : cudaErrorInvalidValue);
    out[1] = cudaErrorNotReady - 34;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array([5, 5]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
    expect(compiled.wgsl).toContain("var set: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var reset: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var last: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var peek: i32 = i32(0);");
    expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { out: new Int32Array([5, 5]) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
    expect([...result.buffers.out as Int32Array]).toEqual([0, 0]);
  });

  it("models cudaGetDevice as a status no-op that writes device zero", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void runtime_get_device(int *out) {
  if (threadIdx.x < 1) {
    int local = -1;
    int count = -1;
    int warp = -1;
    int blocks = -1;
    int runtimeVersion = -1;
    int driverVersion = -1;
    int peer = -1;
    int status = cudaGetDevice(&local);
    int countStatus = cudaGetDeviceCount(&count);
    int attrStatus = cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, 0);
    cudaDeviceGetAttribute(&blocks, cudaDevAttrMaxThreadsPerBlock, 0);
    int runtimeStatus = cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);
    int peerStatus = cudaDeviceCanAccessPeer(&peer, 0, 0);
    int enablePeerStatus = cudaDeviceEnablePeerAccess(0, 0);
    int disablePeerStatus = cudaDeviceDisablePeerAccess(0);
    cudaGetDevice(&out[2]);
    cudaGetDeviceCount(&out[3]);
    out[0] = local;
    out[1] = status;
    out[4] = count;
    out[5] = countStatus;
    out[6] = warp;
    out[7] = blocks + attrStatus;
    out[8] = runtimeVersion + runtimeStatus;
    out[9] = driverVersion;
    out[10] = peer + peerStatus + enablePeerStatus + disablePeerStatus;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(11).fill(-1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
    expect(compiled.wgsl).toContain("var status: i32 = 0;");
    expect(compiled.wgsl).toContain("var countStatus: i32 = 0;");
    expect(compiled.wgsl).toContain("var attrStatus: i32 = 0;");
    expect(compiled.wgsl).toContain("var runtimeStatus: i32 = 0;");
    expect(compiled.wgsl).toContain("var peerStatus: i32 = 0;");
    expect(compiled.wgsl).toContain("var enablePeerStatus: i32 = i32(0);");
    expect(compiled.wgsl).toContain("local = 0;");
    expect(compiled.wgsl).toContain("count = 1;");
    expect(compiled.wgsl).toContain("warp = 32;");
    expect(compiled.wgsl).toContain("blocks = 1024;");
    expect(compiled.wgsl).toContain("runtimeVersion = 12000;");
    expect(compiled.wgsl).toContain("driverVersion = 12000;");
    expect(compiled.wgsl).toContain("out[2] = i32(0);");
    expect(compiled.wgsl).toContain("out[3] = i32(1);");
    expect(compiled.wgsl).toContain("peer = 1;");
    expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { out: new Int32Array(11).fill(-1) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
    expect([...result.buffers.out as Int32Array]).toEqual([0, 0, 0, 1, 1, 0, 32, 1024, 12000, 12000, 1]);
  });

  it("models CUDA device limit queries over size_t outputs", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void runtime_limit_query(uint *out) {
  if (threadIdx.x < 1) {
    size_t limit = 0;
    int status = cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
    int setStatus = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, limit);
    out[0] = limit;
    out[1] = (uint)(status + setStatus);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array([0, 9]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
    expect(compiled.wgsl).toContain("var status: i32 = 0;");
    expect(compiled.wgsl).toContain("var setStatus: i32 = i32(0);");
    expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { out: new Uint32Array(2) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
    expect([...result.buffers.out as Uint32Array]).toEqual([1048576, 0]);
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
    expect(compiled.wgsl).toContain("var setLimitStatus: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var setCacheStatus: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var syncStatus: i32 = i32(0);");
    expect(compiled.wgsl).toContain("var exitStatus: i32 = i32(0);");
    expect(createCudaWebGpuExecutionPlan(
      compiled,
      { buffers: { limits: new Uint32Array(1), status: new Int32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    ).supported).toBe(true);
    expect([...result.buffers.limits as Uint32Array]).toEqual([1048576]);
    expect([...result.buffers.status as Int32Array]).toEqual([0, 0]);
  });

  it("models cudaMemGetInfo over size_t outputs", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void runtime_mem_info(uint *out) {
  if (threadIdx.x < 1) {
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    int status = cudaMemGetInfo(&freeBytes, &totalBytes);
    cudaMemGetInfo(&out[2], &out[3]);
    out[0] = freeBytes + (uint)status;
    out[1] = totalBytes;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-cuda-runtime");
    expect(compiled.wgsl).toContain("var status: i32 = 0;");
    expect(compiled.wgsl).toContain("freeBytes = 268435456;");
    expect(compiled.wgsl).toContain("totalBytes = 268435456;");
    expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { out: new Uint32Array(4) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
    expect([...result.buffers.out as Uint32Array]).toEqual([268435456, 268435456, 268435456, 268435456]);
  });

  it("models CUDA occupancy runtime queries", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void runtime_occupancy(int *out) {
  if (threadIdx.x < 1) {
    int active = -1;
    int activeFlags = -1;
    int minGrid = -1;
    int blockSize = -1;
    int minGridFlags = -1;
    int blockSizeFlags = -1;
    int activeStatus = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, runtime_occupancy, 128, 0);
    int activeFlagsStatus = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&activeFlags, runtime_occupancy, 128, 0, cudaOccupancyDisableCachingOverride);
    int potentialStatus = cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, runtime_occupancy, 0, 0);
    int potentialFlagsStatus = cudaOccupancyMaxPotentialBlockSizeWithFlags(&minGridFlags, &blockSizeFlags, runtime_occupancy, 0, 0, cudaOccupancyDefault);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&out[10], runtime_occupancy, 128, 0);
    cudaOccupancyMaxPotentialBlockSize(&out[11], &out[12], runtime_occupancy, 0, 0);
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
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(13).fill(-1) } },
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
    expect(compiled.wgsl).toContain("out[10] = i32(1);");
    expect(compiled.wgsl).toContain("out[11] = i32(1);");
    expect(compiled.wgsl).toContain("out[12] = i32(256);");
    expect(createCudaWebGpuExecutionPlan(compiled, { buffers: { out: new Int32Array(13) } }, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] }).supported).toBe(true);
    expect([...result.buffers.out as Int32Array]).toEqual([1, 0, 1, 0, 1, 256, 0, 1, 256, 0, 1, 1, 256]);
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
    expect(compiled.wgsl).toContain("cudaGraphSetConditional omitted");
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

  it("flattens vector pointer alias chains before dynamic shared scalar atomics", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void add_dynamic_shared_scalar(float *lanes, int lane, float value) {
  atomicAdd(&lanes[lane], value);
}
__global__ void dynamicSharedFloat3ScalarAtomic(float *out) {
  extern __shared__ float scratch[];
  int tid = threadIdx.x;
  float3 *tile = reinterpret_cast<float3 *>(scratch);
  tile[tid] = make_float3((float)(tid + 1), (float)(tid + 10), (float)(tid + 100));
  __syncthreads();
  float *lanes = reinterpret_cast<float *>(tile + 1);
  add_dynamic_shared_scalar(lanes, tid, 0.5f + (float)tid);
  __syncthreads();
  if (tid == 0) {
    out[0] = lanes[0];
    out[1] = lanes[1];
  }
}`, {
      workgroupSize: [2, 1, 1],
      dynamicSharedMemory: { scratch: 6 },
    });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("add_dynamic_shared_scalar(1u, (0u + u32((0 + ((0 + 1) * 3))))");
    expect(compiled.wgsl).toContain("bg_ptr_atomicAdd_f32(lanes_buffer, (lanes_base + u32(lane)), value)");
    expect(compiled.wgsl).not.toContain("add_dynamic_shared_scalar(1u, ((0u + u32(0)) + u32((0 + 1)))");
    expect([...result.buffers.out as Float32Array]).toEqual([2.5, 12.5]);
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

    expect(compiled.wgsl).toContain("ptrs_base[u32(1)] = ((0u + u32(0)) + (u32(1) * 3u));");
    expect(compiled.wgsl).toContain("ptrs_base[u32(2)] = ((0u + u32(0)) + (u32(2) * 3u));");
    expect(compiled.wgsl).not.toContain("ptrs_base[u32(1)] = ((0u + u32(0)) + u32(1));");
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

    expect(compiled.wgsl).toContain("ptrs_base[u32(0)] = ((0u + u32((0 + ((0 + 1) * 3)))) + (u32(0) * 3u));");
    expect(compiled.wgsl).toContain("ptrs_base[u32(1)] = ((0u + u32((0 + ((0 + 1) * 3)))) + (u32(1) * 3u));");
    expect(compiled.wgsl).not.toContain("ptrs_base[u32(1)] = ((0u + u32((0 + ((0 + 1) * 3)))) + u32(1));");
    expect([...result.buffers.out as Float32Array]).toEqual([17, 19, 23, 1, 53, 61, 73, 0]);
  });

  it("lowers CUDA alternate extern shared qualifier order", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void dynamicSharedLateQualifier(float *x) {
  extern double __shared__ scratch[];
  int tid = threadIdx.x;
  if (tid < 2) { scratch[tid] = (double)x[tid]; }
  __syncthreads();
  if (tid < 1) { x[0] = (float)(scratch[0] + scratch[1]); }
}`, {
      workgroupSize: [2, 1, 1],
      dynamicSharedMemory: { scratch: 2 },
      f64Mode: "f32",
    });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([2, 3]) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<workgroup> scratch: array<f32, 2>;");
    expect([...result.buffers.x as Float32Array]).toEqual([5, 3]);
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

    expect(compiled.wgsl).toContain("var<workgroup> scratch: array<array<f32, 2>, 2>;");
    expect([...result.buffers.x as Float32Array]).toEqual([7, 3]);
  });

  it("accepts volatile shared-memory qualifier order", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void volatileShared(float *out) {
  volatile __shared__ float scratch[2];
  int tid = threadIdx.x;
  if (tid < 2) { scratch[tid] = out[tid]; }
  __syncthreads();
  if (tid < 1) { out[0] = scratch[0] + scratch[1]; }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array([2, 3]) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<workgroup> scratch: array<f32, 2>;");
    expect([...result.buffers.out as Float32Array]).toEqual([5, 3]);
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

    expect(compiled.wgsl).toContain("var<workgroup> tile: array<f32, 4>;");
    expect(compiled.wgsl).toContain("sharedBarrierHelper(0u, 0u, local_id, workgroup_id, num_workgroups);");
    expect(compiled.wgsl).not.toContain("bg_inline_sharedBarrierHelper");
  });

  it("keeps shared-atomic loop breaks before later barriers uniform", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void sharedAtomicBreakBeforeBarrier(uint *out) {
  __shared__ uint done;
  if (threadIdx.x == 0) {
    atomicExch(&done, 1u);
  }
  while (1) {
    __syncthreads();
    if (done == 1u) break;
    __syncthreads();
    out[threadIdx.x] = 3u;
  }
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.wgsl).toContain("var bg_loop_active_");
    expect(compiled.wgsl).toContain("workgroupBarrier();\n      bg_loop_active_");
    expect(compiled.wgsl).toContain(" = (bg_loop_active_");
    expect(compiled.wgsl).toContain(" && !((atomicLoad(&done) == 1u)));");
    expect(compiled.wgsl).not.toContain("if ((atomicLoad(&done) == 1u)) {\n      break;\n    }\n    workgroupBarrier();");
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
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.wgsl).toContain("if ((bg_uniforms.enabled != 0))");
    expect(compiled.wgsl).toContain("total += conditional_helper_with_pointer_side_effect");
    expect(compiled.wgsl).toContain("total += 0u;");
    expect(compiled.wgsl).not.toContain("select(0u, conditional_helper_with_pointer_side_effect");
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

    expect(compiled.wgsl).toContain("var total: u32 = 0u;");
    expect(compiled.wgsl).toContain("if ((bg_uniforms.enabled != 0))");
    expect(compiled.wgsl).toContain("total = conditional_var_init_helper_with_pointer_side_effect");
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

    expect(compiled.wgsl).toContain("var total: u32 = 0u;");
    expect(compiled.wgsl).toContain("if ((bg_uniforms.enabled != 0))");
    expect(compiled.wgsl).toContain("total = (3u + nested_conditional_var_init_helper_with_pointer_side_effect");
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
    expect(compiled.wgsl).toContain("value = (u32((step + 1)) + active_nested_conditional_var_init_helper_with_pointer_side_effect");
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
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.wgsl).toMatch(/var bg_return_value_\d+: u32 = 0u;/u);
    expect(compiled.wgsl).toContain("if ((enabled != 0))");
    expect(compiled.wgsl).toContain("bg_return_value_");
    expect(compiled.wgsl).toContain("= (3u + nested_return_conditional_helper_with_pointer_side_effect");
    expect(compiled.wgsl).not.toContain("select(0u, nested_return_conditional_helper_with_pointer_side_effect");
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
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.wgsl).toContain("if ((bg_uniforms.enabled != 0))");
    expect(compiled.wgsl).toContain("nested_arg_sink((3u + nested_arg_conditional_helper_with_pointer_side_effect");
    expect(compiled.wgsl).toContain("nested_arg_sink((3u + 0u)");
    expect(compiled.wgsl).not.toContain("select(0u, nested_arg_conditional_helper_with_pointer_side_effect");
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

    expect(compiled.wgsl).toContain("if ((bg_uniforms.enabled != 0))");
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
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.wgsl).toMatch(/if \(bg_loop_condition_\d+\) \{/u);
    expect(compiled.wgsl).toContain("bg_uniforms.enabled != 0");
    expect(compiled.wgsl).toContain("nested_loop_condition_helper_with_pointer_side_effect");
    expect(compiled.wgsl).not.toContain("select(0u, nested_loop_condition_helper_with_pointer_side_effect");

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
    expect(sequenceCompiled.wgsl).toContain("nested_sequence_loop_condition_helper_with_pointer_side_effect");
    expect(sequenceCompiled.wgsl).not.toContain("select(0u, nested_sequence_loop_condition_helper_with_pointer_side_effect");
  });

  it("preserves conditional helper-call laziness in vector-lane lvalues", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint conditional_vector_lane_lvalue_helper_with_pointer_side_effect(uint *ptr, uint add) {
  atomicAdd(ptr, add);
  return 1u;
}

__global__ void conditionalVectorLaneLvalue(uint *storage, uint4 *out, int enabled) {
  uint4 value = make_uint4(1u, 2u, 3u, 4u);
  value[enabled != 0 ? (int)conditional_vector_lane_lvalue_helper_with_pointer_side_effect(storage, 7u) : 0] = 9u;
  out[0] = value;
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.wgsl).toContain("conditional_vector_lane_lvalue_helper_with_pointer_side_effect");
    expect(compiled.wgsl).not.toContain("select(0, i32(conditional_vector_lane_lvalue_helper_with_pointer_side_effect");
    expect(compiled.wgsl).not.toContain("select(0u, conditional_vector_lane_lvalue_helper_with_pointer_side_effect");
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

  it("guards conditional helper-call vector lane lvalues inside active-lane predication", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint active_conditional_vector_lane_lvalue_helper_with_pointer_side_effect(uint *ptr, uint add) {
  atomicAdd(ptr, add);
  return 1u;
}

__global__ void activeConditionalVectorLaneLvalue(uint *storage, uint4 *out, int limit, int enabled) {
  int tid = threadIdx.x;
  uint4 value = make_uint4(1u, 2u, 3u, 4u);
  for (int step = 0; step < 2; step++) {
    if (tid >= limit) return;
    __syncthreads();
    value[enabled != 0 ? (int)active_conditional_vector_lane_lvalue_helper_with_pointer_side_effect(storage + tid, (uint)(step + tid + 1)) : 0] = 9u;
    __syncthreads();
  }
  out[tid] = value;
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
    expect(compiled.wgsl).toMatch(/if \(bg_barrier_loop_active_\d+\) \{\n\s+if \(\(bg_uniforms.enabled != 0\)\)/u);
    expect(compiled.wgsl).toContain("active_conditional_vector_lane_lvalue_helper_with_pointer_side_effect");
    expect(compiled.wgsl).not.toContain("select(0, i32(active_conditional_vector_lane_lvalue_helper_with_pointer_side_effect");
    expect(compiled.wgsl).not.toContain("select(0u, active_conditional_vector_lane_lvalue_helper_with_pointer_side_effect");
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

  it("preserves conditional helper-call laziness in atomic addresses", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint conditional_atomic_address_helper_with_pointer_side_effect(uint *ptr, uint add) {
  atomicAdd(ptr, add);
  return 1u;
}

__global__ void conditionalAtomicAddress(uint *storage, uint *out, int enabled) {
  uint old = atomicAdd(storage + (enabled != 0 ? conditional_atomic_address_helper_with_pointer_side_effect(storage, 7u) : 0u), 5u);
  out[0] = storage[0];
  out[1] = storage[1];
  out[2] = old;
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.wgsl).toContain("conditional_atomic_address_helper_with_pointer_side_effect");
    expect(compiled.wgsl).not.toContain("select(0u, conditional_atomic_address_helper_with_pointer_side_effect");
  });

  it("guards conditional helper-call atomic addresses inside active-lane predication", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint active_conditional_atomic_address_helper_with_pointer_side_effect(uint *ptr, uint add) {
  atomicAdd(ptr, add);
  return 1u;
}

__global__ void activeConditionalAtomicAddress(uint *storage, int limit, int enabled) {
  int tid = threadIdx.x;
  for (int step = 0; step < 2; step++) {
    if (tid >= limit) return;
    __syncthreads();
    atomicAdd(storage + (enabled != 0 ? active_conditional_atomic_address_helper_with_pointer_side_effect(storage + tid, (uint)(step + tid + 1)) : 0u), 5u);
    __syncthreads();
  }
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
    expect(compiled.wgsl).toMatch(/if \(bg_barrier_loop_active_\d+\) \{\n\s+if \(\(bg_uniforms.enabled != 0\)\)/u);
    expect(compiled.wgsl).toContain("active_conditional_atomic_address_helper_with_pointer_side_effect");
    expect(compiled.wgsl).not.toContain("select(0u, active_conditional_atomic_address_helper_with_pointer_side_effect");
  });

  it("preserves atomic side effects before loop returns lowered for barriers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomicReturnSideEffectBarrier(uint *counter, uint *out, int N) {
  extern __shared__ uint scratch[];
  int tid = threadIdx.x;
  for (int k = 0; k < 2; ++k) {
    int idx = tid + k * 4;
    if (idx >= N) {
      atomicAdd(&counter[0], 1u);
      return;
    }
    scratch[tid] = (uint)idx;
    __syncthreads();
    out[idx] = scratch[tid] + 1u;
    __syncthreads();
  }
}`, { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
    expect(compiled.wgsl).toContain("atomicAdd");
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

  it("preserves surface side effects before returns lowered for barriers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void surfaceReturnSideEffectBarrier(cudaSurfaceObject_t surf, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    surf2Dwrite(100.0f + (float)tid, surf, tid * sizeof(float), 0);
    return;
  }
  __syncthreads();
  surf2Dwrite(1.0f + (float)tid, surf, tid * sizeof(float), 0);
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
    expect(compiled.wgsl).toContain("100.0");
    expect(compiled.wgsl).toContain("bg_surf2dwrite_surf");
    expect(compiled.wgsl).toContain("bg_active_lane = false;");
  });

  it("preserves texture read side effects before returns lowered for barriers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float4 read_return_texture_vec(cudaTextureObject_t texArg) {
  return tex2D<float4>(texArg, 0.5f, 0.5f);
}

__global__ void textureReturnReadSideEffectBarrier(cudaTextureObject_t tex, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float4 value = read_return_texture_vec(tex);
    out[tid] = value.x + value.y + value.z + value.w + (float)tid;
    return;
  }
  __syncthreads();
  out[tid] = 1.0f + (float)tid;
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
    expect(compiled.wgsl).toContain("bg_tex2d_float4_tex");
    expect(compiled.wgsl).toContain("value");
    expect(compiled.wgsl).toContain("bg_active_lane = false;");
  });

  it("preserves atlas texture read side effects before returns lowered for barriers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float read_return_texture_atlas(cudaTextureObject_t texArg) {
  float layered = tex2DLayered<float>(texArg, 0.0f, 1.0f, 1.0f);
  float volume = tex3D<float>(texArg, 2.0f, 1.0f, 1.0f);
  return layered + volume;
}

__global__ void textureAtlasReturnReadSideEffectBarrier(cudaTextureObject_t tex, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    out[tid] = read_return_texture_atlas(tex) + (float)tid;
    return;
  }
  __syncthreads();
  out[tid] = 1.0f + (float)tid;
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
    expect(compiled.wgsl).toContain("var layered: f32 = textureLoad(texArg");
    expect(compiled.wgsl).toContain("var volume: f32 = textureLoad(texArg");
    expect(compiled.wgsl).toContain("bg_active_lane = false;");
  });

  it("preserves deep texture helper vector stores before returns lowered for barriers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float4 read_deep_texture_leaf(cudaTextureObject_t texArg) {
  return tex2D<float4>(texArg, 0.5f, 0.5f);
}

__device__ float4 read_deep_texture_mid(cudaTextureObject_t texArg) {
  return read_deep_texture_leaf(texArg);
}

__device__ float4 read_deep_texture_outer(cudaTextureObject_t texArg) {
  return read_deep_texture_mid(texArg);
}

__global__ void textureDeepHelperVectorStoreBarrier(cudaTextureObject_t tex, float4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float4 value = read_deep_texture_outer(tex);
    out[tid] = make_float4(value.x + (float)tid, value.y, value.z, value.w);
    return;
  }
  __syncthreads();
  out[tid] = make_float4(1.0f + (float)tid, 10.0f + (float)tid, 20.0f + (float)tid, 30.0f + (float)tid);
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
    expect(compiled.wgsl).toContain("bg_tex2d_float4_tex");
    expect(compiled.wgsl).toContain("read_deep_texture_outer");
    expect(compiled.wgsl).toContain("bg_active_lane = false;");
  });

  it("preserves mixed scalar and vector texture stores before returns lowered for barriers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float read_mixed_texture_scalar(cudaTextureObject_t texArg) {
  return tex2D<float>(texArg, 0.5f, 0.5f);
}

__device__ uint4 read_mixed_texture_vec(cudaTextureObject_t texArg) {
  return tex2D<uint4>(texArg, 0.5f, 0.5f);
}

__global__ void textureMixedScalarVectorStoreBarrier(cudaTextureObject_t tex, float4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float scalar = read_mixed_texture_scalar(tex);
    uint4 vec = read_mixed_texture_vec(tex);
    out[tid] = make_float4(scalar + (float)tid, (float)vec.y, (float)vec.z, (float)vec.w);
    return;
  }
  __syncthreads();
  out[tid] = make_float4(1.0f + (float)tid, 10.0f + (float)tid, 20.0f + (float)tid, 30.0f + (float)tid);
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
    expect(compiled.wgsl).toContain("bg_tex2d_f32_tex");
    expect(compiled.wgsl).toContain("bg_tex2d_uint4_tex");
    expect(compiled.wgsl).toContain("bg_active_lane = false;");
  });

  it("preserves texture-fed scalar pointer alias stores before returns lowered for barriers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float read_alias_texture_scalar(cudaTextureObject_t texArg) {
  return tex2D<float>(texArg, 0.5f, 0.5f);
}

__device__ void write_alias_lane(float *scalarOut, int lane, float value) {
  scalarOut[lane * 4 + 1] = value;
}

__global__ void texturePointerAliasStoreBarrier(cudaTextureObject_t tex, float4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float *scalarView = reinterpret_cast<float*>(out);
    write_alias_lane(scalarView, tid, read_alias_texture_scalar(tex) + (float)tid);
    return;
  }
  __syncthreads();
  out[tid] = make_float4(1.0f + (float)tid, 10.0f + (float)tid, 20.0f + (float)tid, 30.0f + (float)tid);
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
    expect(compiled.wgsl).toContain("bg_tex2d_f32_tex");
    expect(compiled.wgsl).toContain("write_alias_lane");
    expect(compiled.wgsl).toContain("bg_active_lane = false;");
  });

  it("preserves texture-fed pointer alias atomics before returns lowered for barriers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint read_alias_texture_uint(cudaTextureObject_t texArg) {
  return tex2D<uint>(texArg, 0.5f, 0.5f);
}

__device__ void atomic_alias_lane(uint *scalarOut, int lane, uint value) {
  atomicAdd(&scalarOut[lane * 4 + 1], value);
}

__global__ void texturePointerAliasAtomicStoreBarrier(cudaTextureObject_t tex, uint4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    uint *scalarView = reinterpret_cast<uint*>(out);
    atomic_alias_lane(scalarView, tid, read_alias_texture_uint(tex) + (uint)tid);
    return;
  }
  __syncthreads();
  out[tid] = make_uint4(1u + (uint)tid, 10u + (uint)tid, 20u + (uint)tid, 30u + (uint)tid);
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
    expect(compiled.wgsl).toContain("bg_tex2d_uint_tex");
    expect(compiled.wgsl).toContain("atomicAdd");
    expect(compiled.wgsl).toContain("atomicStore(&out[((u32(tid) * 4u) + 0u)]");
    expect(compiled.wgsl).toContain("bg_active_lane = false;");
  });

  it("scales atomic vector reads after pointer alias atomics", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint read_atomic_readback_texture_uint(cudaTextureObject_t texArg) {
  return tex2D<uint>(texArg, 0.5f, 0.5f);
}

__device__ void atomic_readback_alias_lane(uint *scalarOut, int lane, uint value) {
  atomicAdd(&scalarOut[lane * 4 + 1], value);
}

__global__ void texturePointerAliasAtomicVectorReadback(cudaTextureObject_t tex, uint4 *out, uint *summary) {
  int tid = threadIdx.x;
  out[tid] = make_uint4(1u + (uint)tid, 10u + (uint)tid, 20u + (uint)tid, 30u + (uint)tid);
  __syncthreads();
  if (tid == 0) {
    uint *scalarView = reinterpret_cast<uint*>(out);
    atomic_readback_alias_lane(scalarView, 1, read_atomic_readback_texture_uint(tex));
  }
  __syncthreads();
  if (tid == 1) {
    uint4 value = out[1];
    summary[0] = value.x + value.y + value.z + value.w;
  }
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.wgsl).toContain("atomicLoad(&out[((u32(1) * 4u) + 0u)]");
    expect(compiled.wgsl).not.toContain("atomicLoad(&out[(1 + 0u)]");
  });

  it("scales atomic vector compound writes through pointer aliases", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint read_atomic_compound_texture_uint(cudaTextureObject_t texArg) {
  return tex2D<uint>(texArg, 0.5f, 0.5f);
}

__device__ void atomic_compound_alias_lane(uint *scalarOut, int lane, uint value) {
  atomicAdd(&scalarOut[lane * 4 + 1], value);
}

__device__ void add_vector_alias(uint4 *vectorOut, int lane, uint4 value) {
  vectorOut[lane] += value;
}

__device__ void add_vector_alias_y(uint4 *vectorOut, int lane, uint value) {
  vectorOut[lane].y += value;
}

__global__ void texturePointerAliasAtomicVectorCompound(cudaTextureObject_t tex, uint4 *out, uint *summary) {
  int tid = threadIdx.x;
  out[tid] = make_uint4(1u + (uint)tid, 10u + (uint)tid, 20u + (uint)tid, 30u + (uint)tid);
  __syncthreads();
  if (tid == 0) {
    uint *scalarView = reinterpret_cast<uint*>(out);
    atomic_compound_alias_lane(scalarView, 1, read_atomic_compound_texture_uint(tex));
    uint4 *vectorView = reinterpret_cast<uint4*>(out);
    add_vector_alias(vectorView, 1, make_uint4(1u, 1u, 1u, 1u));
    add_vector_alias_y(vectorView, 2, 9u);
  }
  __syncthreads();
  if (tid == 1) {
    uint4 value = out[1];
    uint4 laneTwo = out[2];
    summary[0] = (value.x + value.y + value.z + value.w) + 100u * (laneTwo.x + laneTwo.y + laneTwo.z + laneTwo.w);
  }
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.wgsl).toContain("atomicLoad(&out[(index + 0u)]");
    expect(compiled.wgsl).toContain("atomicStore(&out[(index + 0u)]");
    expect(compiled.wgsl).toContain("bg_ptr_read_u32x4(vectorOut_buffer, (vectorOut_base + (u32(lane) * 4u)))");
    expect(compiled.wgsl).toContain("bg_ptr_write_u32x4(vectorOut_buffer, (vectorOut_base + (u32(lane) * 4u)), vec4<u32>");
    expect(compiled.wgsl).not.toContain("vectorOut[");
  });

  it("preserves texture-to-surface side effects before returns lowered for barriers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float4 sample_return_surface_vec(cudaTextureObject_t texArg) {
  return tex2D<float4>(texArg, 0.5f, 0.5f);
}

__device__ void write_return_surface_vec(cudaSurfaceObject_t surfaceArg, int lane, float4 value) {
  surf2Dwrite(value.x + value.y + value.z + value.w + (float)lane, surfaceArg, lane * sizeof(float), 0);
}

__global__ void textureSurfaceReturnSideEffectBarrier(cudaSurfaceObject_t surf, cudaTextureObject_t tex, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float4 value = sample_return_surface_vec(tex);
    write_return_surface_vec(surf, tid, value);
    return;
  }
  __syncthreads();
  surf2Dwrite(1.0f + (float)tid, surf, tid * sizeof(float), 0);
}`, { workgroupSize: [4, 1, 1] });

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("divergent-return-before-barrier");
    expect(compiled.wgsl).toContain("bg_tex2d_float4_tex");
    expect(compiled.wgsl).toContain("bg_surf2dwrite_surf");
    expect(compiled.wgsl).toContain("bg_active_lane = false;");
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

    expect(compiled.wgsl).toContain("for (var bg_barrier_loop_iter_");
    expect(compiled.wgsl).toContain("bg_barrier_loop_iter_");
    expect(compiled.wgsl).toContain("bg_uniforms.N");
    expect(compiled.wgsl).not.toContain("< 256u;");
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

    expect(compiled.wgsl).toContain("if (bg_barrier_loop_active_");
    expect(compiled.wgsl).toContain("&& (((k + 1) < 2)))");
    expect(compiled.wgsl).toContain("workgroupBarrier();\n    }\n    workgroupBarrier();");
    expect(compiled.wgsl).not.toContain("if (((k + 1) < 2)) {\n      workgroupBarrier();");
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

    expect(compiled.wgsl).toContain("if ((bg_uniforms.flag != 0))");
    expect(compiled.wgsl).toContain("if (!((bg_uniforms.flag != 0)))");
    expect(compiled.wgsl).not.toContain("} else {\n    workgroupBarrier();");
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

    expect(compiled.wgsl).toContain("var warpId: i32 = 0;");
    expect(compiled.wgsl).toContain("workgroupBarrier();");
    expect(compiled.wgsl).not.toContain("i32(local_id.x) / 32");
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

    expect(compiled.wgsl).toContain("var<workgroup> bg_shared: array<f32, 2>;");
    expect(compiled.wgsl).toContain("vals[0][0] = 1.0;");
    expect([...result.buffers.out as Float32Array]).toEqual([12]);
  });

  it("accepts host/device constexpr helper qualifiers", () => {
    const compiled = compileCudaLiteKernel(`
__device__ __host__ constexpr unsigned int mix(unsigned int x, unsigned int seed) {
  x += seed;
  x ^= (x >> 9);
  return x;
}
__global__ void host_device_constexpr_helper(uint *out) {
  out[0] = mix(512u, 7u);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn mix");
    expect([...result.buffers.out as Uint32Array]).toEqual([518]);
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
    expect(compiled.wgsl).toContain("bg_ptr_write_f32(1u, ((0u + u32((0 + 2))) + u32(tid))");
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
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(compiled.wgsl).toContain("var<workgroup> localCount: atomic<i32>;");
    expect(compiled.wgsl).toContain("atomicStore(&localCount, i32(7))");
    expect(compiled.wgsl).toContain("atomicAdd(&localCount, 1)");
    expect(compiled.wgsl).toContain("atomicLoad(&localCount)");
    expect([...semanticResult.buffers.out as Int32Array]).toEqual([9]);
    expect([...result.buffers.out as Int32Array]).toEqual([9]);
  });

  it("runs top-level shared integer atomics through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void semantic_shared_atomic(uint *out) {
  __shared__ uint counter[1];
  counter[0] = 1u;
  out[0] = atomicAdd(&counter[0], 2u);
  out[1] = counter[0];
}`, { workgroupSize: [1, 1, 1] });
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var<workgroup> counter: array<atomic<u32>, 1>;");
    expect(compiled.wgsl).toContain("atomicAdd(&counter[0u], 2u)");
    expect(compiled.wgsl).toContain("atomicLoad(&counter[0u])");
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1, 3]);
    expect([...result.buffers.out as Uint32Array]).toEqual([1, 3]);
  });

  it("supports CAS-backed float atomics in shared memory", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void sharedFloatAtomic(float *out) {
  __shared__ float acc[1];
  if (threadIdx.x == 0) { acc[0] = 0.0f; }
  __syncthreads();
  atomicAdd(&acc[0], 1.5f);
  __syncthreads();
  if (threadIdx.x == 0) { out[0] = acc[0]; }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var<workgroup> acc: array<atomic<u32>, 1>;");
    expect(compiled.wgsl).toContain("fn bg_atomicAdd_f32_workgroup");
    expect(compiled.wgsl).toContain("bitcast<f32>(atomicLoad(&acc[0]))");
    expect([...result.buffers.out as Float32Array]).toEqual([3]);
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
    expect(backendIr(compileCudaLiteKernel(`
static __global__ void staticFirst(int *out) {
  out[0] = 1;
}`)).name).toBe("staticFirst");

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

    expect(backendIr(compiled).params.map((param) => [param.name, param.valueType])).toContainEqual(["n", "uint"]);
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

    expect(backendIr(compiled).sharedDeclarations[0]?.dimensions).toEqual([2]);
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

    expect(backendIr(compiled).sharedDeclarations[0]?.dimensions).toEqual([4]);
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

    expect(compiled.wgsl).toContain("scratch[0u] = swizzle");
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

    expect(backendIr(compiled).name).toBe("bounded");
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
    const semanticResult = runCompiledKernelSemanticReference(
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

    expect(backendIr(compiled).constants.map((constant) => constant.name)).toEqual(["scaleFactor"]);
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("scaleFactor: f32");
    expect(compiled.wgsl).toContain("bg_uniforms.scaleFactor");
    expect([...semanticResult.buffers.y as Float32Array]).toEqual([3, 6, 9, 12]);
    expect([...result.buffers.y as Float32Array]).toEqual([3, 6, 9, 12]);
  });

  it("embeds initialized scalar CUDA constants through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__constant__ float scaleFactor = 0.5f;
__global__ void scaleInitialized(float *x, float *y) {
  int idx = threadIdx.x;
  y[idx] = x[idx] * scaleFactor;
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([2, 4, 6, 8]),
          y: new Float32Array(4),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([2, 4, 6, 8]),
          y: new Float32Array(4),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("const scaleFactor: f32 = 0.5;");
    expect(compiled.wgslProgram.bindings.map((binding) => binding.name)).not.toContain("scaleFactor");
    expect([...semanticResult.buffers.y as Float32Array]).toEqual([1, 2, 3, 4]);
    expect([...result.buffers.y as Float32Array]).toEqual([1, 2, 3, 4]);
  });

  it("embeds initialized CUDA constant arrays through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__constant__ int coeffs[3] = {2, 3, 5};
__global__ void constantArrayInitialized(int *out) {
  int idx = threadIdx.x;
  out[idx] = coeffs[idx];
}`, { workgroupSize: [3, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(3) } },
      { gridDim: [1, 1, 1], blockDim: [3, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Int32Array(3) } },
      { gridDim: [1, 1, 1], blockDim: [3, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("const coeffs: array<i32, 3> = array<i32, 3>(2, 3, 5);");
    expect(compiled.wgslProgram.bindings.map((binding) => binding.name)).not.toContain("coeffs");
    expect([...semanticResult.buffers.out as Int32Array]).toEqual([2, 3, 5]);
    expect([...result.buffers.out as Int32Array]).toEqual([2, 3, 5]);
  });

  it("accepts const-qualified CUDA constant arrays", () => {
    const compiled = compileCudaLiteKernel(`
__constant__ const int table[2] = {3, 5};
__global__ void const_table(int *out) {
  if (threadIdx.x < 2) { out[threadIdx.x] = table[threadIdx.x]; }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(backendIr(compiled).constants.map((constant) => constant.name)).toEqual(["table"]);
    expect([...result.buffers.out as Int32Array]).toEqual([3, 5]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var<storage, read> coeffs: array<f32>");
    expect(compiled.wgslProgram.bindings).toContainEqual(expect.objectContaining({
      name: "coeffs",
      access: "read",
    }));
    expect([...semanticResult.buffers.x as Float32Array]).toEqual([20, 80]);
    expect([...result.buffers.x as Float32Array]).toEqual([20, 80]);
  });

  it("flattens multidimensional CUDA constant arrays through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__constant__ uint table[2][3];
__global__ void constant_matrix(uint *out) {
  int row = threadIdx.y;
  int col = threadIdx.x;
  out[row * 3 + col] = table[row][col];
}`, { workgroupSize: [3, 2, 1] });
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Uint32Array(6) },
        constants: { table: new Uint32Array([3, 5, 7, 11, 13, 17]) },
      },
      { gridDim: [1, 1, 1], blockDim: [3, 2, 1] },
    );
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Uint32Array(6) },
        constants: { table: new Uint32Array([3, 5, 7, 11, 13, 17]) },
      },
      { gridDim: [1, 1, 1], blockDim: [3, 2, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var<storage, read> table: array<u32>");
    expect(compiled.wgsl).toContain("* 3u");
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([3, 5, 7, 11, 13, 17]);
    expect([...result.buffers.out as Uint32Array]).toEqual([3, 5, 7, 11, 13, 17]);
  });

  it("casts signed local initializers from unsigned arithmetic in WGSL", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__device__ int edge(unsigned char left, unsigned char right) {
  short delta = right - left;
  return delta;
}
__global__ void kernel(int *out, unsigned int a, unsigned int b) {
  out[0] = edge(a, b);
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("var delta: i32 = i32((right - left));");
  });

  it("casts device-function scalar compound assignments from promoted operands in WGSL", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__device__ int scale_box(unsigned char ul, unsigned char um, float fscale) {
  short Sum = (short)(ul + um) / 2;
  Sum *= fscale;
  return Sum;
}
__global__ void kernel(int *out, unsigned int a, unsigned int b, float scale) {
  out[0] = scale_box(a, b, scale);
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).not.toContain("Sum *= fscale");
    expect(compiled.wgsl).toContain("Sum = i32((f32(Sum) * fscale));");
  });

  it("casts mixed signed and unsigned arithmetic operands in WGSL", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__global__ void kernel(unsigned int *out, unsigned int pitch) {
  int i = threadIdx.x;
  out[(blockIdx.x * pitch) + i] = 7;
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("(workgroup_id.x * bg_uniforms.pitch)");
    expect(compiled.wgsl).toContain("+ u32(i)");
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

  it("casts uncached integer local initializers for float vars", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__global__ void kernel(float *out) {
  float dist = ((int)threadIdx.x - 3) * ((int)threadIdx.x - 3);
  out[0] = dist;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("var dist: f32 = f32(");
    expect([...result.buffers.out as Float32Array]).toEqual([9]);
  });

  it("casts pointer-alias base and offset index math in WGSL", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__global__ void kernel(unsigned int *out, unsigned int pitch) {
  unsigned int *row = out + (blockIdx.x * pitch);
  int i = threadIdx.x;
  row[i] = 7;
}`, { workgroupSize: [1, 1, 1] });

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var<storage, read> table: array<f32>;");
    expect(compiled.wgsl).toContain("vec3<f32>(table[((u32(idx) * 3u) + 0u)]");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([6, 15]);
    expect([...result.buffers.out as Float32Array]).toEqual([6, 15]);
  });

  it("embeds initialized scalar CUDA vector constants", () => {
    const compiled = compileCudaLiteKernel(`
__constant__ float3 metric = {1.0f, 2.0f, 3.0f};
__global__ void vector_const(float *out) {
  if (threadIdx.x == 0) { out[0] = metric.x + metric.y + metric.z; }
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("const metric: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0)");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([6]);
    expect([...result.buffers.out as Float32Array]).toEqual([6]);
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

    expect(compiled.wgsl).toContain("var<storage, read> coeffs: array<f32, 3>");
    expect(compiled.wgsl).toContain("return coeffs[index]");
    expect([...result.buffers.out as Float32Array]).toEqual([7]);
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

    expect(compiled.wgsl).toContain("var<storage, read> table: array<array<u32, 2>, 2>;");
    expect(compiled.wgsl).toMatch(/var row_\d+_buffer: u32 = 1u;/u);
    expect(compiled.wgsl).toMatch(/var row_\d+_base: u32 = \(\(u32\(0\) \* 2u\) \+ u32\(0\)\);/u);
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
    expect(oneDimensional.wgsl).toMatch(/var row_\d+_base: u32 = u32\(0\);/u);
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
    expect(compiled.wgslProgram.bindings.map((binding) => binding.name)).not.toContain("coeffs");
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
    expect(compiled.wgslProgram.bindings.map((binding) => binding.name)).not.toContain("Q");
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
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Float32Array(2) },
        textures: { texRef: { width: 2, height: 1, data: new Float32Array([4, 8]) } },
        scalars: { width: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(backendIr(compiled).textures.map((texture) => texture.name)).toEqual(["texRef"]);
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var texRef: texture_2d<f32>;");
    expect(compiled.wgsl).toContain("textureLoad(texRef");
    expect(compiled.wgslProgram.bindings).toContainEqual(expect.objectContaining({
      kind: "texture2d",
      name: "texRef",
    }));
    expect([...result.buffers.out as Float32Array]).toEqual([4, 8]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([4, 8]);
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
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Float32Array(3) },
        textures: { tex: { width: 3, height: 1, data: new Float32Array([2, 4, 6]) } },
        scalars: { width: 3 },
      },
      { gridDim: [1, 1, 1], blockDim: [3, 1, 1] },
    );

    expect(backendIr(compiled).params.find((param) => param.name === "tex")?.valueType).toBe("texture2d");
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var tex: texture_2d<f32>;");
    expect(compiled.wgsl).toContain("textureLoad(tex");
    expect(compiled.wgslProgram.bindings).toContainEqual(expect.objectContaining({
      kind: "texture2d",
      name: "tex",
    }));
    expect([...result.buffers.out as Float32Array]).toEqual([2, 4, 6]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([2, 4, 6]);
  });

  it("lowers scalar tex2DLod reads through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void sample(float *out, cudaTextureObject_t tex) {
  out[0] = tex2DLod<float>(tex, 1.5f, 0.5f, 0.0f);
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: { out: new Float32Array(1) },
      textures: { tex: { width: 2, height: 1, data: new Float32Array([5, 13]) } },
    };
    const launch = { gridDim: [1, 1, 1], blockDim: [1, 1, 1] } as const;
    const result = runCompiledKernelReference(compiled, input, launch);
    const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("textureLoad(tex");
    expect([...result.buffers.out as Float32Array]).toEqual([13]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([13]);
  });

  it("honors normalized point-wrap texture descriptors in reference and WGSL lowering", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void sample(float *out, int width, int height, cudaTextureObject_t tex) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  out[y * width + x] = tex2D<float>(tex, (x + 1) / (float)width, (y + 1) / (float)height);
}`, {
      workgroupSize: [4, 2, 1],
      textureDescriptors: {
        tex: { normalizedCoords: true, addressMode: ["wrap", "wrap"], filterMode: "point" },
      },
    });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(8) },
        textures: { tex: { width: 4, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
        scalars: { width: 4, height: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 2, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Float32Array(8) },
        textures: { tex: { width: 4, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
        scalars: { width: 4, height: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 2, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr, compiled.textureDescriptors ? { textureDescriptors: compiled.textureDescriptors } : {})).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn bg_sem_tex2d_tex");
    expect(compiled.wgsl).toContain("x * f32(dims.x)");
    expect([...result.buffers.out as Float32Array]).toEqual([6, 7, 8, 5, 2, 3, 4, 1]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([6, 7, 8, 5, 2, 3, 4, 1]);
  });

  it("honors normalized linear-wrap texture descriptors in reference and WGSL lowering", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void sample(float *out, int width, int height, cudaTextureObject_t tex) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  out[y * width + x] = tex2D<float>(tex, x / (float)width, y / (float)height);
}`, {
      workgroupSize: [4, 2, 1],
      textureDescriptors: {
        tex: { normalizedCoords: true, addressMode: ["wrap", "wrap"], filterMode: "linear" },
      },
    });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(8) },
        textures: { tex: { width: 4, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
        scalars: { width: 4, height: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 2, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Float32Array(8) },
        textures: { tex: { width: 4, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
        scalars: { width: 4, height: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 2, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr, compiled.textureDescriptors ? { textureDescriptors: compiled.textureDescriptors } : {})).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn bg_sem_tex2d_tex");
    expect(compiled.wgsl).toContain("let xb = sx - 0.5;");
    expect([...result.buffers.out as Float32Array]).toEqual([4.5, 3.5, 4.5, 5.5, 4.5, 3.5, 4.5, 5.5]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([4.5, 3.5, 4.5, 5.5, 4.5, 3.5, 4.5, 5.5]);
  });

  it("lowers templated uchar tex2D reads", () => {
    const compiled = compileCudaLiteKernel(`
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
__global__ void sample(uint *out) {
  int x = threadIdx.x;
  out[x] = tex2D<unsigned char>(texRef, (float)x + 0.5f, 0.5f);
}`, { workgroupSize: [3, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Uint32Array(3) },
        textures: { texRef: { width: 3, height: 1, data: new Float32Array([2, 127, 255]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [3, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bg_tex2d_uchar_texRef");
    expect([...result.buffers.out as Uint32Array]).toEqual([2, 127, 255]);
  });

  it("lowers templated bf16 and bf162 tex2D reads with native WebGPU f32 storage", () => {
    const compiled = compileCudaLiteKernel(`
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
__global__ void sample(float *out, uint *bits) {
  __nv_bfloat16 scalar = tex2D<__nv_bfloat16>(texRef, 0.5f, 0.5f);
  __nv_bfloat162 pair = tex2D<__nv_bfloat162>(texRef, 0.5f, 0.5f);
  out[0] = __bfloat162float(scalar);
  out[1] = pair.x;
  out[2] = pair.y;
  bits[0] = __bfloat16_as_ushort(scalar);
  bits[1] = __bfloat162_as_uint(pair);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          out: new Float32Array(3),
          bits: new Uint32Array(2),
        },
        textures: { texRef: { width: 1, height: 1, data: new Float32Array([1.1, 2.2, 3.3, 4.4]), channels: 4 } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-texture");
    expect(compiled.wgsl).toContain("bg_tex2d_bf16_texRef");
    expect(compiled.wgsl).toContain("bg_tex2d_bf162_texRef");
    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(1.1015625);
    expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(1.1015625);
    expect([...result.buffers.out as Float32Array][2]).toBeCloseTo(2.203125);
    expect([...result.buffers.bits as Uint32Array]).toEqual([0x3f8d, 0x400d3f8d]);
  });

  it("lowers float4 texture helper returns through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float4 readPixel(cudaTextureObject_t texSrc, float x) {
  return tex2D<float4>(texSrc, x + 0.5f, 0.5f);
}
__global__ void sample(float *out, cudaTextureObject_t tex) {
  int x = threadIdx.x;
  float4 value = readPixel(tex, (float)x);
  out[x * 4 + 0] = value.x;
  out[x * 4 + 1] = value.y;
  out[x * 4 + 2] = value.z;
  out[x * 4 + 3] = value.w;
}`, { workgroupSize: [2, 1, 1] });
    const input = {
      buffers: { out: new Float32Array(8) },
      textures: { tex: { width: 2, height: 1, channels: 4 as const, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
    };
    const launch = { gridDim: [1, 1, 1], blockDim: [2, 1, 1] } as const;
    const result = runCompiledKernelReference(compiled, input, launch);
    const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn readPixel(texSrc: texture_2d<f32>, x: f32) -> vec4<f32>");
    expect(compiled.wgsl).toContain("textureLoad(texSrc");
    expect([...result.buffers.out as Float32Array]).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("passes CUDA texture handles through device helper params", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float sampleAt(cudaTextureObject_t texSrc, float x) {
  return tex2D<float>(texSrc, x + 0.5f, 0.5f);
}
__global__ void sample(float *out, cudaTextureObject_t tex) {
  int x = threadIdx.x;
  out[x] = sampleAt(tex, (float)x);
}`, { workgroupSize: [3, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(3) },
        textures: { tex: { width: 3, height: 1, data: new Float32Array([3, 6, 9]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [3, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Float32Array(3) },
        textures: { tex: { width: 3, height: 1, data: new Float32Array([3, 6, 9]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [3, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn sampleAt(texSrc: texture_2d<f32>");
    expect(compiled.wgsl).toContain("textureLoad(texSrc");
    expect([...result.buffers.out as Float32Array]).toEqual([3, 6, 9]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([3, 6, 9]);
  });

  it("lowers nested CUDA texture helper params through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float sampleInner(cudaTextureObject_t texInner, float x) {
  return tex2D<float>(texInner, x + 0.5f, 0.5f);
}
__device__ float sampleOuter(cudaTextureObject_t texOuter, float x) {
  return sampleInner(texOuter, x);
}
__global__ void sample(float *out, cudaTextureObject_t tex) {
  int x = threadIdx.x;
  out[x] = sampleOuter(tex, (float)x);
}`, { workgroupSize: [3, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(3) },
        textures: { tex: { width: 3, height: 1, data: new Float32Array([4, 8, 12]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [3, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Float32Array(3) },
        textures: { tex: { width: 3, height: 1, data: new Float32Array([4, 8, 12]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [3, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn sampleInner(texInner: texture_2d<f32>");
    expect(compiled.wgsl).toContain("fn sampleOuter(texOuter: texture_2d<f32>");
    expect(compiled.wgsl).toContain("sampleInner(texOuter");
    expect([...result.buffers.out as Float32Array]).toEqual([4, 8, 12]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([4, 8, 12]);
  });

  it("propagates texture descriptors through device helper texture params", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float sampleAt(cudaTextureObject_t texSrc, float x, float y) {
  return tex2D<float>(texSrc, x, y);
}
__global__ void sample(float *out, int width, int height, cudaTextureObject_t tex) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  out[y * width + x] = sampleAt(tex, x / (float)width, y / (float)height);
}`, {
      workgroupSize: [4, 2, 1],
      textureDescriptors: {
        tex: { normalizedCoords: true, addressMode: ["wrap", "wrap"], filterMode: "linear" },
      },
    });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(8) },
        textures: { tex: { width: 4, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
        scalars: { width: 4, height: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 2, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Float32Array(8) },
        textures: { tex: { width: 4, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
        scalars: { width: 4, height: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 2, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr, compiled.textureDescriptors ? { textureDescriptors: compiled.textureDescriptors } : {})).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn sampleAt(texSrc: texture_2d<f32>");
    expect(compiled.wgsl).toContain("fn sampleAt__bg_tex_");
    expect(compiled.wgsl).toContain("bg_sem_tex2d_texSrc_");
    expect([...result.buffers.out as Float32Array]).toEqual([4.5, 3.5, 4.5, 5.5, 4.5, 3.5, 4.5, 5.5]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([4.5, 3.5, 4.5, 5.5, 4.5, 3.5, 4.5, 5.5]);
  });

  it("propagates texture descriptors through nested device helper texture params", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float sampleInner(cudaTextureObject_t texInner, float x, float y) {
  return tex2D<float>(texInner, x, y);
}
__device__ float sampleOuter(cudaTextureObject_t texOuter, float x, float y) {
  return sampleInner(texOuter, x, y);
}
__global__ void sample(float *out, int width, int height, cudaTextureObject_t tex) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  out[y * width + x] = sampleOuter(tex, x / (float)width, y / (float)height);
}`, {
      workgroupSize: [4, 2, 1],
      textureDescriptors: {
        tex: { normalizedCoords: true, addressMode: ["wrap", "wrap"], filterMode: "linear" },
      },
    });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(8) },
        textures: { tex: { width: 4, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
        scalars: { width: 4, height: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 2, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Float32Array(8) },
        textures: { tex: { width: 4, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) } },
        scalars: { width: 4, height: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 2, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr, compiled.textureDescriptors ? { textureDescriptors: compiled.textureDescriptors } : {})).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn sampleInner(texInner: texture_2d<f32>");
    expect(compiled.wgsl).toContain("fn sampleInner__bg_tex_");
    expect(compiled.wgsl).toContain("fn sampleOuter__bg_tex_");
    expect(compiled.wgsl).toContain("bg_sem_tex2d_texInner_");
    expect([...result.buffers.out as Float32Array]).toEqual([4.5, 3.5, 4.5, 5.5, 4.5, 3.5, 4.5, 5.5]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([4.5, 3.5, 4.5, 5.5, 4.5, 3.5, 4.5, 5.5]);
  });

  it("specializes texture descriptor helper params for conflicting descriptors", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float sampleAt(cudaTextureObject_t texSrc, float x, float y) {
  return tex2D<float>(texSrc, x, y);
}
__global__ void sample(float *out, int width, int height, cudaTextureObject_t linearTex, cudaTextureObject_t pointTex) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  int offset = y * width + x;
  out[offset] = sampleAt(linearTex, x / (float)width, y / (float)height);
  out[offset + width * height] = sampleAt(pointTex, (float)x, (float)y);
}`, {
      workgroupSize: [4, 2, 1],
      textureDescriptors: {
        linearTex: { normalizedCoords: true, addressMode: ["wrap", "wrap"], filterMode: "linear" },
        pointTex: { normalizedCoords: false, addressMode: ["clamp", "clamp"], filterMode: "point" },
      },
    });

    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(16) },
        textures: {
          linearTex: { width: 4, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) },
          pointTex: { width: 4, height: 2, data: new Float32Array([11, 12, 13, 14, 15, 16, 17, 18]) },
        },
        scalars: { width: 4, height: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 2, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Float32Array(16) },
        textures: {
          linearTex: { width: 4, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) },
          pointTex: { width: 4, height: 2, data: new Float32Array([11, 12, 13, 14, 15, 16, 17, 18]) },
        },
        scalars: { width: 4, height: 2 },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 2, 1] },
    );
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr, compiled.textureDescriptors ? { textureDescriptors: compiled.textureDescriptors } : {})).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect((compiled.wgsl.match(/fn sampleAt__bg_tex_/g) ?? []).length).toBe(2);
    expect((compiled.wgsl.match(/fn bg_sem_tex2d_texSrc_/g) ?? []).length).toBe(2);
    expect([...result.buffers.out as Float32Array]).toEqual([
      4.5, 3.5, 4.5, 5.5, 4.5, 3.5, 4.5, 5.5,
      11, 12, 13, 14, 15, 16, 17, 18,
    ]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([
      4.5, 3.5, 4.5, 5.5, 4.5, 3.5, 4.5, 5.5,
      11, 12, 13, 14, 15, 16, 17, 18,
    ]);
  });

  it("specializes conflicting texture descriptor helpers in guarded barrier clones", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void guardedSample(cudaTextureObject_t texSrc, float *out, int offset, float x, float y) {
  out[offset] = tex2D<float>(texSrc, x, y);
  __syncthreads();
}
__global__ void sample(float *out, int width, int height, cudaTextureObject_t linearTex, cudaTextureObject_t pointTex) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  if (x < width) {
    int offset = y * width + x;
    guardedSample(linearTex, out, offset, x / (float)width, y / (float)height);
  }
  __syncthreads();
  if (x < width) {
    int offset = y * width + x;
    guardedSample(pointTex, out, offset + width * height, (float)x, (float)y);
  }
}`, {
      workgroupSize: [4, 2, 1],
      textureDescriptors: {
        linearTex: { normalizedCoords: true, addressMode: ["wrap", "wrap"], filterMode: "linear" },
        pointTex: { normalizedCoords: false, addressMode: ["clamp", "clamp"], filterMode: "point" },
      },
    });

    expect(compiled.wgsl).toContain("fn guardedSample__bg_tex_0__bg_guarded_barrier");
    expect(compiled.wgsl).toContain("fn guardedSample__bg_tex_1__bg_guarded_barrier");
    expect(compiled.wgsl).toContain("textureDimensions(texSrc).x");
  });

  it("specializes conflicting texture descriptor helpers feeding pointer-array atomics", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float readDescriptorAtomic(cudaTextureObject_t texSrc, float x, float y) {
  return tex2D<float>(texSrc, x, y);
}
__device__ void addDescriptorAtomic(uint *slot, float value) {
  atomicAdd(slot, (uint)(value * 10.0f));
}
__global__ void sample(uint *out, int width, int height, cudaTextureObject_t linearTex, cudaTextureObject_t pointTex) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  if (x < width) {
    uint *targets[2];
    targets[0] = out;
    targets[1] = out + 1;
    float linearValue = readDescriptorAtomic(linearTex, x / (float)width, y / (float)height);
    addDescriptorAtomic(targets[linearValue > 4.0f ? 1 : 0], linearValue);
    float pointValue = readDescriptorAtomic(pointTex, (float)x, (float)y);
    addDescriptorAtomic(out + 2 + (pointValue > 14.0f ? 1 : 0), pointValue);
  }
}`, {
      workgroupSize: [4, 2, 1],
      textureDescriptors: {
        linearTex: { normalizedCoords: true, addressMode: ["wrap", "wrap"], filterMode: "linear" },
        pointTex: { normalizedCoords: false, addressMode: ["clamp", "clamp"], filterMode: "point" },
      },
    });

    expect(compiled.wgsl).toContain("fn readDescriptorAtomic__bg_tex_0");
    expect(compiled.wgsl).toContain("fn readDescriptorAtomic__bg_tex_1");
    expect(compiled.wgsl).toContain("atomicAdd(");
    expect(compiled.wgsl).toContain("textureDimensions(texSrc).x");
  });

  it("does not apply 2D texture descriptors to cubemap direction coords", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float readCube(cudaTextureObject_t texSrc) {
  return texCubemap<float>(texSrc, 1.0f, 0.0f, 0.0f);
}
__global__ void sample(float *out, cudaTextureObject_t tex) {
  out[0] = readCube(tex);
}`, {
      workgroupSize: [1, 1, 1],
      textureDescriptors: {
        tex: { normalizedCoords: true, addressMode: ["wrap", "wrap"], filterMode: "point" },
      },
    });

    expect(compiled.wgsl).toContain("bg_cube_face");
    expect(compiled.wgsl).not.toContain("fn readCube__bg_tex_0");
    expect(compiled.wgsl).not.toContain("bg_tex2d_f32_texSrc");
  });

  it("lowers CUDA driver texture object aliases as texture params", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void sample(float *out, CUtexObject tex) {
  int x = threadIdx.x;
  out[x] = tex2D<float>(tex, (float)x + 0.5f, 0.5f);
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(2) },
        textures: { tex: { width: 2, height: 1, data: new Float32Array([7, 11]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(backendIr(compiled).params.find((param) => param.name === "tex")?.valueType).toBe("texture2d");
    expect([...result.buffers.out as Float32Array]).toEqual([7, 11]);
  });

  it("lowers typed CUDA tex2D vector reads and vector-scalar math", () => {
    const compiled = compileCudaLiteKernel(`
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
__global__ void sample(float4 *out) {
  float4 t = make_float4(1.0f);
  t += tex2D<float4>(texRef, 0.5f, 0.5f);
  t = t * 0.5f;
  out[0] = t;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Float32Array(4) },
        textures: {
          texRef: { width: 1, height: 1, channels: 4, data: new Float32Array([1, 2, 3, 4]) },
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bg_tex2d_float4_texRef");
    expect(compiled.wgsl).toContain("vec4<f32>(f32(0.5)");
    expect([...result.buffers.out as Float32Array]).toEqual([1, 1.5, 2, 2.5]);
  });

  it("lowers CUDA helper_math vector min/max overloads", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void clampVector(float4 *out) {
  float4 a = make_float4(-1.0f, 2.0f, 8.0f, 300.0f);
  float4 b = fminf(a, make_float4(255.0f));
  out[0] = fmaxf(b, 0.0f);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("min(");
    expect(compiled.wgsl).toContain("max(");
    expect([...result.buffers.out as Float32Array]).toEqual([0, 2, 8, 255]);
  });

  it("lowers vector assignment chains and POD-field aliases", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void vectorChain(float4 *x, float4 *y, float *out) {
  float4 value = make_float4(2.0f, 3.0f, 5.0f, 7.0f);
  x[0] = y[0] = value;
  float4 record = x[0];
  out[0] = record.S + record.X + record.MuByT + record.VBySqrtT;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array(4),
          y: new Float32Array(4),
          out: new Float32Array(1),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.x as Float32Array]).toEqual([2, 3, 5, 7]);
    expect([...result.buffers.y as Float32Array]).toEqual([2, 3, 5, 7]);
    expect([...result.buffers.out as Float32Array]).toEqual([17]);
  });

  it("lowers CUDA alias typed texture reads through integer vector casts", () => {
    const compiled = compileCudaLiteKernel(`
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
__global__ void sample(uint4 *out) {
  out[0] = tex2D<uchar4>(texRef, 0.5f, 0.5f);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Uint32Array(4) },
        textures: {
          texRef: { width: 1, height: 1, channels: 4, data: new Float32Array([1, 2, 3, 255]) },
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bg_tex2d_uint4_texRef");
    expect([...result.buffers.out as Uint32Array]).toEqual([1, 2, 3, 255]);
  });

  it("lowers CUDA texture fetch aliases without repo-specific rewrites", () => {
    const compiled = compileCudaLiteKernel(`
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
__global__ void sample(float4 *vecOut, float *scalarOut) {
  vecOut[0] = tex2DLod<float4>(texRef, 0.5f, 0.5f, 0.0f);
  scalarOut[0] = tex1Dfetch<float>(texRef, 1);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          vecOut: new Float32Array(4),
          scalarOut: new Float32Array(1),
        },
        textures: {
          texRef: { width: 2, height: 1, channels: 4, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) },
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bg_tex2d_float4_texRef");
    expect([...result.buffers.vecOut as Float32Array]).toEqual([1, 2, 3, 4]);
    expect([...result.buffers.scalarOut as Float32Array]).toEqual([5]);
  });

  it("lowers CUDA 1D, layered, 3D, and cubemap texture calls through texture atlas helpers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void sample(float4 *vecOut, float *scalarOut, cudaTextureObject_t tex) {
  scalarOut[0] = tex1D<float>(tex, 1.0f);
  scalarOut[1] = tex2DLayered<float>(tex, 0.0f, 1.0f, 1.0f);
  scalarOut[2] = tex3D<float>(tex, 2.0f, 1.0f, 1.0f);
  scalarOut[3] = texCubemap<float>(tex, 1.0f, 0.0f, 0.0f);
  vecOut[0] = tex1D<float4>(tex, 0.0f);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          vecOut: new Float32Array(4),
          scalarOut: new Float32Array(4),
        },
        textures: {
          tex: {
            width: 4,
            height: 24,
            channels: 4,
            data: new Float32Array(Array.from({ length: 4 * 24 * 4 }, (_, index) => index + 1)),
          },
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bg_cube_face");
    expect(compiled.wgsl).toContain("bg_cube_u");
    expect(compiled.wgsl).toContain("bg_tex2d_float4_tex");
    expect([...result.buffers.scalarOut as Float32Array]).toEqual([5, 33, 41, 21]);
    expect([...result.buffers.vecOut as Float32Array]).toEqual([1, 2, 3, 4]);
  });

  it("lowers CUDA surf2Dread into guarded surface buffer loads", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void readSurface(uint *out, cudaSurfaceObject_t surf) {
  uint value = 0;
  surf2Dread(&value, surf, 4, 0);
  out[0] = value;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Uint32Array(1) },
        surfaces: { surf: { width: 2, height: 1, data: new Float32Array([3, 9]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Uint32Array(1) },
        surfaces: { surf: { width: 2, height: 1, data: new Float32Array([3, 9]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("bg_sem_surf2dread_surf");
    expect([...result.buffers.out as Uint32Array]).toEqual([9]);
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([9]);
  });

  it("lowers templated surf2Dread return-form calls", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void readSurface(uint *out, cudaSurfaceObject_t surf) {
  out[0] = surf2Dread<unsigned int>(surf, 4, 0);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: { out: new Uint32Array(1) },
        surfaces: { surf: { width: 2, height: 1, data: new Float32Array([3, 9]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: { out: new Uint32Array(1) },
        surfaces: { surf: { width: 2, height: 1, data: new Float32Array([3, 9]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("bg_sem_surf2dread_surf");
    expect([...result.buffers.out as Uint32Array]).toEqual([9]);
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([9]);
  });

  it("lowers CUDA driver surface object aliases as surface params", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void writeSurface(CUsurfObject surf) {
  surf2Dwrite(13u, surf, 4, 0);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {},
        surfaces: { surf: { width: 2, height: 1, data: new Float32Array([3, 9]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {},
        surfaces: { surf: { width: 2, height: 1, data: new Float32Array([3, 9]) } },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(backendIr(compiled).params.find((param) => param.name === "surf")?.valueType).toBe("surface2d");
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var<storage, read_write> surf: array<f32>;");
    expect([...result.buffers.surf as Float32Array]).toEqual([3, 13]);
    expect([...semanticResult.buffers.surf as Float32Array]).toEqual([3, 13]);
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
    expect(compiled.wgslProgram.bindings[0]).toMatchObject({ valueType: "f32" });
    expect(compiled.wgslProgram.bindings[1]).toMatchObject({ valueType: "f32" });

    const uniforms = packCudaWebGpuUniformParams(compiled, {
      buffers: {
        x: new Float32Array([1.5]),
        y: new Float32Array([3, 5]),
      },
      scalars: { a: 2 },
    });
    expect(new DataView(uniforms.buffer).getFloat32(0, true)).toBe(2);
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

    expect(compiled.wgsl).toContain("bg_inline_warpPrefixSum_");
    expect(compiled.wgsl).toContain("prefix = bg_inline_warpPrefixSum_");
    expect(compiled.wgsl).not.toContain("if ((idx < bg_uniforms.N)) {\n    var val");
    expect(compiled.wgsl).not.toContain("if ((idx < bg_uniforms.N)) {\n    var prefix");
    expect(compiled.wgsl).toContain("bg_warp_shuffle_up_float_32");
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
    if (big > 999999.0f) { flag[0] = 1; }
    output[idx] = __float2half(value * 2.0f);
    output[idx + 1] = __uint2half_rn(4u);
    flag[idx + 1] = __half_as_ushort(__float2half(1.5f));
    output[idx + 2] = __ushort_as_half(0x3c00u);
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
          output: createWgslFloat16Array(3),
          flag: new Int32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {
          input: createWgslFloat16Array([1.5]),
          output: createWgslFloat16Array(3),
          flag: new Int32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("enable f16;");
    expect(compiled.wgsl).toContain("f32(input[u32(idx)])");
    expect(compiled.wgsl).toContain("f16((value * 2.0))");
    expect(compiled.wgsl).toContain("f16(f32(4u))");
    expect(compiled.wgsl).toContain("pack2x16float(vec2<f32>(f32(f16(1.5)), 0.0)) & 0xffffu");
    expect(compiled.wgsl).toContain("f16(unpack2x16float(15360u).x)");
    expect(compiled.wgsl).toContain("1000000.0");
    expect(Array.from(result.buffers.output as Iterable<number>)).toEqual([3, 4, 1]);
    expect([...result.buffers.flag as Int32Array]).toEqual([1, 0x3e00]);
    expect(Array.from(semanticResult.buffers.output as Iterable<number>)).toEqual([3, 4, 1]);
    expect([...semanticResult.buffers.flag as Int32Array]).toEqual([1, 0x3e00]);
  });

  it("lowers CUDA half-to-int conversion rounding modes", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void half_int_modes(const half* input, int* iout, uint* uout) {
  half a = input[0];
  half b = input[1];
  half c = input[2];
  half d = input[3];
  iout[0] = __half2int_rn(a);
  iout[1] = __half2int_rz(a);
  iout[2] = __half2int_ru(a);
  iout[3] = __half2int_rd(a);
  iout[4] = __half2int_rn(b);
  iout[5] = __half2int_rn(c);
  iout[6] = __half2int_rd(c);
  uout[0] = __half2uint_rn(b);
  uout[1] = __half2uint_rz(d);
  uout[2] = __half2uint_ru(d);
  uout[3] = __half2uint_rd(d);
}`, {
      features: { "shader-f16": true },
      workgroupSize: [1, 1, 1],
    });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          input: createWgslFloat16Array([2.5, 3.5, -2.5, 3.25]),
          iout: new Int32Array(7),
          uout: new Uint32Array(4),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {
          input: createWgslFloat16Array([2.5, 3.5, -2.5, 3.25]),
          iout: new Int32Array(7),
          uout: new Uint32Array(4),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("enable f16;");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(compiled.wgsl).toContain("bg_semantic_round_even_f32(f32(a))");
    expect(compiled.wgsl).toContain("i32(trunc(f32(a)))");
    expect(compiled.wgsl).toContain("i32(ceil(f32(a)))");
    expect(compiled.wgsl).toContain("i32(floor(f32(a)))");
    expect(compiled.wgsl).toContain("u32(max(bg_semantic_round_even_f32(f32(b)), 0.0))");
    expect([...result.buffers.iout as Int32Array]).toEqual([2, 2, 3, 2, 4, -2, -3]);
    expect([...result.buffers.uout as Uint32Array]).toEqual([4, 3, 4, 3]);
    expect([...semanticResult.buffers.iout as Int32Array]).toEqual([2, 2, 3, 2, 4, -2, -3]);
    expect([...semanticResult.buffers.uout as Uint32Array]).toEqual([4, 3, 4, 3]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
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

  it("lowers CUDA bf16 integer conversion rounding modes", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void bf16_int_modes(int* iout, uint* uout, __nv_bfloat16* output) {
  __nv_bfloat16 a = __float2bfloat16(2.5f);
  __nv_bfloat16 b = __float2bfloat16(3.5f);
  __nv_bfloat16 c = __float2bfloat16(-2.5f);
  __nv_bfloat16 d = __float2bfloat16(3.25f);
  iout[0] = __bfloat162int_rn(a);
  iout[1] = __bfloat162int_rz(a);
  iout[2] = __bfloat162int_ru(a);
  iout[3] = __bfloat162int_rd(a);
  iout[4] = __bfloat162int_rn(b);
  iout[5] = __bfloat162int_rn(c);
  iout[6] = __bfloat162int_rd(c);
  uout[0] = __bfloat162uint_rn(b);
  uout[1] = __bfloat162uint_rz(d);
  uout[2] = __bfloat162uint_ru(d);
  uout[3] = __bfloat162uint_rd(d);
  output[0] = __int2bfloat16_rn(iout[2]);
  output[1] = __uint2bfloat16_rn(uout[2]);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          iout: new Int32Array(7),
          uout: new Uint32Array(4),
          output: new Float32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {
          iout: new Int32Array(7),
          uout: new Uint32Array(4),
          output: new Float32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
    expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
    expect(compiled.wgsl).toContain("bg_semantic_round_even_f32(f32(a))");
    expect(compiled.wgsl).toContain("i32(trunc(f32(a)))");
    expect(compiled.wgsl).toContain("u32(max(bg_semantic_round_even_f32(f32(b)), 0.0))");
    expect([...result.buffers.iout as Int32Array]).toEqual([2, 2, 3, 2, 4, -2, -3]);
    expect([...result.buffers.uout as Uint32Array]).toEqual([4, 3, 4, 3]);
    expect([...result.buffers.output as Float32Array]).toEqual([3, 4]);
    expect([...semanticResult.buffers.iout as Int32Array]).toEqual([2, 2, 3, 2, 4, -2, -3]);
    expect([...semanticResult.buffers.uout as Uint32Array]).toEqual([4, 3, 4, 3]);
    expect([...semanticResult.buffers.output as Float32Array]).toEqual([3, 4]);
  });

  it("lowers scalar CUDA bf16 arithmetic through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void bf16_scalar_ops(const __nv_bfloat16* input, __nv_bfloat16* output) {
  __nv_bfloat16 a = input[0];
  __nv_bfloat16 b = input[1];
  output[0] = __hadd(a, b);
  output[1] = __hsub(a, b);
  output[2] = __hmul(a, b);
  output[3] = __hdiv(a, b);
  output[4] = __hfma(a, b, __float2bfloat16(1.0f));
  output[5] = __hmin(a, b);
  output[6] = __hmax(a, b);
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        input: new Float32Array([2, 0.5]),
        output: new Float32Array(7),
      },
    };
    const result = runCompiledKernelReference(
      compiled,
      input,
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {
          input: new Float32Array([2, 0.5]),
          output: new Float32Array(7),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("bitcast<f32>((bitcast<u32>(f32((a + b))) + 0x8000u) & 0xffff0000u)");
    expect(compiled.wgsl).toContain("fma(a, b, bitcast<f32>((bitcast<u32>(f32(1.0)) + 0x8000u) & 0xffff0000u))");
    expect([...result.buffers.output as Float32Array]).toEqual([2.5, 1.5, 1, 4, 2, 0.5, 2]);
    expect([...semanticResult.buffers.output as Float32Array]).toEqual([2.5, 1.5, 1, 4, 2, 0.5, 2]);
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

  it("packs bf16x2 local reinterpret bits for atomic CAS operands", () => {
    const compiled = compileCudaLiteKernelForWebGpu(`
__global__ void bf162_bits(float* data, unsigned int* out) {
  __nv_bfloat162* packed = reinterpret_cast<__nv_bfloat162*>(data);
  __nv_bfloat162 current = packed[0];
  __nv_bfloat162 next = current + __halves2bfloat162((__nv_bfloat16)1.0f, (__nv_bfloat16)2.0f);
  unsigned int currentBits = *reinterpret_cast<unsigned int*>(&current);
  unsigned int nextBits = *reinterpret_cast<unsigned int*>(&next);
  out[0] = atomicCAS((unsigned int*)&packed[0], currentBits, nextBits);
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("atomicCompareExchangeWeak(&data[");
    expect(compiled.wgsl).not.toContain("*&current");
    expect(compiled.wgsl).not.toContain("&vec2<f32>");
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("bitcast<u32>(f32((current).x)) >> 16u");

    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {
          data: new Float32Array([1.5, 2.5]),
          out: new Uint32Array(1),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1069547520]);
    expect([...semanticResult.buffers.data as Float32Array]).toEqual([1.5, 2.5]);
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

  it("lowers CUDA bitwise not and trap no-op control paths", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void bitwise_not_and_trap(int* out) {
  if (threadIdx.x == 99) { __trap(); }
  out[0] = ~(4 - 1);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...result.buffers.out as Int32Array]).toEqual([-4]);
    expect([...semanticResult.buffers.out as Int32Array]).toEqual([-4]);
    expect(compiled.wgsl).toContain("~((4 - 1))");
    expect(compiled.wgsl).toContain("out[0u] = ~((4 - 1));");
    expect(compiled.wgsl).not.toContain("\n    0;\n");
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
    const semanticResult = runCompiledKernelSemanticReference(
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

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("enable f16;");
    expect(compiled.wgsl).toContain("fma(");
    expect(compiled.wgsl).toContain("f16(exp(f32(f16(0.0))))");
    expect(compiled.wgsl).toContain("min(mixed, f16(3.0))");
    expect(compiled.wgsl).toContain("max(");
    expect(Array.from(result.buffers.output as Iterable<number>)).toEqual([1.5]);
    expect([...result.buffers.flags as Int32Array]).toEqual([1, 1, 1, 1, 1, 1]);
    expect(Array.from(semanticResult.buffers.output as Iterable<number>)).toEqual([1.5]);
    expect([...semanticResult.buffers.flags as Int32Array]).toEqual([1, 1, 1, 1, 1, 1]);
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
    expect(compiled.wgsl).toContain("atomicAdd(&x[0");
    expect(compiled.wgsl).toContain("atomicStore(&x[1");
    expect(compiled.wgsl).toContain("atomicLoad(&x[0");
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

  it("supports CUDA double atomicAdd through explicit f32 compatibility lowering", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void addValue(double *result, double value) {
  atomicAdd(result, value);
}
__global__ void atomic_sum(double* result) {
  addValue(result, 1.5);
}`, { workgroupSize: [2, 1, 1], f64Mode: "f32" });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          result: new Float32Array([0]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.diagnostics.some((diagnostic) => diagnostic.code === "f64-lowered-to-f32")).toBe(true);
    expect(compiled.wgsl).toContain("fn bg_atomicAdd_f32");
    expect(compiled.wgsl).not.toContain("fn bg_ptr_atomicAdd_f32");
    expect([...result.buffers.result as Float32Array]).toEqual([3]);
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

  it("stores computed float values back into atomic float storage with u32 carriers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomic_exchange_assign(float* data, float newValue, int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    float oldValue = atomicExch(&data[idx], newValue);
    data[idx] = oldValue + newValue;
  }
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          data: new Float32Array([1, 2, 3, 4]),
        },
        scalars: {
          newValue: 10,
          N: 4,
        },
      },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(compiled.wgsl).toContain("atomicStore(&data[");
    expect(compiled.wgsl).toContain("bitcast<u32>(f32((oldValue + bg_uniforms.newValue)))");
    expect([...result.buffers.data as Float32Array]).toEqual([11, 12, 13, 14]);
  });

  it("drops unused CUDA float atomicExch return values as valid WGSL statements", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomic_exchange_statement(float* x) {
  if (threadIdx.x < 1) { atomicExch(&x[0], 7.5f); }
}`, { workgroupSize: [1, 1, 1] });
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { x: new Float32Array([2.5]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("atomicExchange(&x[0u], bitcast<u32>(7.5));");
    expect(compiled.wgsl).not.toContain("bitcast<f32>(atomicExchange(&x[0u], bitcast<u32>(7.5)));");
    expect([...semanticResult.buffers.x as Float32Array]).toEqual([7.5]);
  });

  it("supports CUDA system-scope float atomics through CAS-backed WGSL helpers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomic_float_system(float* x, float* out) {
  if (threadIdx.x < 1) {
    out[0] = atomicAdd_system(&x[0], 1.5f);
    out[1] = atomicSub_system(&x[0], 0.5f);
    out[2] = atomicMin_system(&x[0], 2.0f);
    out[3] = atomicMax_system(&x[0], 4.0f);
    out[4] = atomicExch_system(&x[0], 6.0f);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Float32Array([2]),
          out: new Float32Array(5),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bg_atomicAdd_f32(&x[0], 1.5)");
    expect(compiled.wgsl).toContain("bg_atomicSub_f32(&x[0], 0.5)");
    expect(compiled.wgsl).toContain("bg_atomicMin_f32(&x[0], 2.0)");
    expect(compiled.wgsl).toContain("bg_atomicMax_f32(&x[0], 4.0)");
    expect(compiled.wgsl).toContain("bitcast<f32>(atomicExchange(&x[0], bitcast<u32>(6.0)))");
    expect([...result.buffers.x as Float32Array]).toEqual([6]);
    expect([...result.buffers.out as Float32Array]).toEqual([2, 3.5, 3, 2, 4]);
  });

  it("supports CUDA pointer-form atomicAdd on integer buffers", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomic_count(int* x) {
  if (threadIdx.x < 1) {
    atomicAdd_system(x, 1);
    atomicExch_system(x, 42);
    atomicAdd(x + 1, 3);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Int32Array([41, 0]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("atomicAdd(&x[0], 1);");
    expect(compiled.wgsl).toContain("atomicExchange(&x[0], 42);");
    expect(compiled.wgsl).toContain("atomicAdd(&x[(0u + u32(1))], 3);");
    expect([...result.buffers.x as Int32Array]).toEqual([42, 3]);
  });

  it("supports atomicAdd through device pointer helper parameters", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void add_i32(int* target, int value) {
  atomicAdd(target, value);
}
__device__ void add_f32(float* target, float value) {
  atomicAdd(target, value);
}
__global__ void helper_atomic(int* xi, float* xf, const float* values) {
  if (threadIdx.x == 0) {
    add_i32(xi + 1, 3);
    add_f32(xf, values[0]);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          xi: new Int32Array([0, 4]),
          xf: new Float32Array([2.5]),
          values: new Float32Array([1.25]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_i32");
    expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_f32");
    expect(compiled.wgsl).toContain("bg_ptr_atomicAdd_i32(target_buffer, target_base, value)");
    expect(compiled.wgsl).toContain("bg_ptr_atomicAdd_f32(target_buffer, target_base, value)");
    expect([...result.buffers.xi as Int32Array]).toEqual([0, 7]);
    expect([...result.buffers.xf as Float32Array]).toEqual([3.75]);
  });

  it("supports atomicAdd through device pointer helper parameters to shared memory", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void add_u32(uint* target, uint value) {
  atomicAdd(target, value);
}
__global__ void helper_shared_atomic(uint* out) {
  __shared__ uint counts[2];
  if (threadIdx.x == 0) {
    counts[0] = 1u;
    add_u32(&counts[0], 4u);
    out[0] = counts[0];
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_u32(");
    expect(compiled.wgsl).toContain("case 1u: { return atomicAdd(&counts[index], value); }");
    expect([...result.buffers.out as Uint32Array]).toEqual([5]);
  });

  it("supports helper pointer atomics over byte-backed storage views", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void add_word(uint* word, uint* out) {
  out[0] = atomicAdd(word, 5u);
  out[1] = word[0];
}
__global__ void helper_byte_atomic(uchar* scratch, uint* out) {
  if (threadIdx.x == 0) {
    add_word((uint*)&scratch[0], out);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          scratch: new Uint32Array([7]),
          out: new Uint32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
    expect(backendIr(compiled).atomicParams).toContain("scratch");
    expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_u32(");
    expect(compiled.wgsl).toContain("case 0u: { return atomicAdd(&scratch[(u32(index)) >> 2u], value); }");
    expect([...result.buffers.out as Uint32Array]).toEqual([7, 12]);
    expect([...result.buffers.scratch as Uint32Array]).toEqual([12]);
  });

  it("supports signed helper atomics over byte-backed storage views", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void signed_byte_ops(int* word, int* out) {
  out[0] = atomicAdd(word, -3);
  out[1] = word[0];
  out[2] = atomicMin(word, -4);
  out[3] = word[0];
  out[4] = atomicMax(word, 9);
  out[5] = word[0];
  out[6] = atomicCAS(word, 9, -2);
  out[7] = word[0];
}
__global__ void signed_byte_atomic(uchar* scratch, int* out) {
  if (threadIdx.x == 0) {
    signed_byte_ops((int*)&scratch[0], out);
    int* direct = (int*)&scratch[4];
    out[8] = atomicExch(direct, -11);
    out[9] = direct[0];
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          scratch: new Uint32Array([10, 22]),
          out: new Int32Array(10),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
    expect(backendIr(compiled).atomicParams).toContain("scratch");
    expect(compiled.wgsl).toContain("case 0u: { return bg_atomicAdd_storage_u32_as_i32(&scratch[(u32(index)) >> 2u], value); }");
    expect(compiled.wgsl).toContain("case 0u: { return bg_atomicMin_storage_u32_as_i32(&scratch[(u32(index)) >> 2u], value); }");
    expect(compiled.wgsl).toContain("case 0u: { return bg_atomicMax_storage_u32_as_i32(&scratch[(u32(index)) >> 2u], value); }");
    expect(compiled.wgsl).toContain("case 0u: { return bg_atomicCompareExchange_storage_u32_as_i32(&scratch[(u32(index)) >> 2u], compare, value); }");
    expect(compiled.wgsl).toContain("bg_atomicExchange_storage_u32_as_i32(&scratch[(u32((4u) + (0u))) >> 2u], (-11))");
    expect([...result.buffers.out as Int32Array]).toEqual([10, 7, 7, -4, -4, 9, 9, -2, 22, -11]);
    expect([...result.buffers.scratch as Uint32Array]).toEqual([0xfffffffe, 0xfffffff5]);
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
    expect(compiled.wgsl).toContain("var<workgroup> bytes: array<atomic<u32>, 4>;");
    expect(compiled.wgsl).toContain("fn bg_ptr_read_u8(");
    expect(compiled.wgsl).toContain("fn bg_ptr_write_u8(");
    expect(compiled.wgsl).toContain("atomicAnd(&bytes");
    expect(compiled.wgsl).toContain("atomicOr(&bytes");
    expect(compiled.wgsl).not.toContain("array<u32, 16>");
    expect([...result.buffers.out as Uint32Array]).toEqual([2]);
  });

  it("supports atomicAdd through shared pointer aliases", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void shared_alias_atomic(float* out) {
  extern __shared__ float shared[];
  float* acc = shared;
  if (threadIdx.x == 0) {
    acc[0] = 0.0f;
    atomicAdd(&acc[0], 1.5f);
    out[0] = acc[0];
  }
}`, { workgroupSize: [1, 1, 1], dynamicSharedMemory: { shared: 1 } });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bg_atomicAdd_f32_workgroup(&bg_shared");
    expect(compiled.wgsl).not.toContain("&shared[");
    expect(compiled.wgsl).not.toContain("&bitcast<f32>");
    expect([...result.buffers.out as Float32Array]).toEqual([1.5]);
  });

  it("supports atomicAdd through direct shared pointer expressions", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void shared_pointer_expr_atomic(float* out) {
  extern __shared__ float shared[];
  if (threadIdx.x == 0) {
    shared[0] = 0.0f;
    atomicAdd(shared + threadIdx.x, 2.5f);
    out[0] = shared[0];
  }
}`, { workgroupSize: [1, 1, 1], dynamicSharedMemory: { shared: 1 } });

    expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
    expect(compiled.wgsl).toContain("bg_atomicAdd_f32_workgroup(&bg_shared");
    expect(compiled.wgsl).not.toContain("&shared[");
  });

  it("loads vector views from atomic float buffers lane-wise", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomic_vector_read(float* scratch, float4* out) {
  if (threadIdx.x == 0) {
    atomicAdd(&scratch[0], 1.0f);
    float4 loaded = reinterpret_cast<float4*>(scratch)[0];
    out[0] = loaded;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          scratch: new Float32Array([0, 2, 3, 4]),
          out: new Float32Array(4),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {
          scratch: new Float32Array([0, 2, 3, 4]),
          out: new Float32Array(4),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn bg_atomicAdd_f32");
    expect(compiled.wgsl).toContain("vec4<f32>(bitcast<f32>(atomicLoad(&scratch");
    expect([...result.buffers.out as Float32Array]).toEqual([1, 2, 3, 4]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 2, 3, 4]);
  });

  it("supports read-modify-write atomics through device pointer helper parameters", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void helper_rmw(int* xi, float* xf, float* out) {
  out[0] = float(atomicSub(xi, 2));
  out[1] = float(atomicMin(xi, 5));
  out[2] = float(atomicMax(xi, 9));
  out[3] = float(atomicAnd(xi, 6));
  out[4] = float(atomicOr(xi, 10));
  out[5] = float(atomicXor(xi, 3));
  out[6] = atomicSub(xf, 1.5f);
  out[7] = atomicMin(xf, 2.0f);
  out[8] = atomicMax(xf, 5.0f);
  out[9] = float(xi[0]);
  out[10] = xf[0];
}
__global__ void helper_atomic_rmw(int* xi, float* xf, float* out) {
  if (threadIdx.x == 0) {
    helper_rmw(xi, xf, out);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          xi: new Int32Array([10]),
          xf: new Float32Array([4]),
          out: new Float32Array(11),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn bg_ptr_atomicSub_i32");
    expect(compiled.wgsl).toContain("fn bg_ptr_atomicMin_i32");
    expect(compiled.wgsl).toContain("fn bg_ptr_atomicMax_i32");
    expect(compiled.wgsl).toContain("fn bg_ptr_atomicAnd_i32");
    expect(compiled.wgsl).toContain("fn bg_ptr_atomicOr_i32");
    expect(compiled.wgsl).toContain("fn bg_ptr_atomicXor_i32");
    expect(compiled.wgsl).toContain("fn bg_ptr_atomicSub_f32");
    expect([...result.buffers.xi as Int32Array]).toEqual([9]);
    expect([...result.buffers.xf as Float32Array]).toEqual([5]);
    expect([...result.buffers.out as Float32Array]).toEqual([10, 8, 5, 9, 0, 10, 4, 2.5, 2, 9, 5]);
  });

  it("supports read-modify-write atomics through device pointer helper parameters to shared memory", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void helper_shared_rmw(int* xi, float* xf, float* out) {
  out[0] = float(atomicSub(xi, 2));
  out[1] = float(atomicMin(xi, 5));
  out[2] = float(atomicMax(xi, 9));
  out[3] = float(atomicAnd(xi, 6));
  out[4] = float(atomicOr(xi, 10));
  out[5] = float(atomicXor(xi, 3));
  out[6] = atomicSub(xf, 1.5f);
  out[7] = atomicMin(xf, 2.0f);
  out[8] = atomicMax(xf, 5.0f);
  out[9] = float(xi[0]);
  out[10] = xf[0];
}
__global__ void helper_shared_atomic_rmw(float* out) {
  __shared__ int xi[1];
  __shared__ float xf[1];
  if (threadIdx.x == 0) {
    xi[0] = 10;
    xf[0] = 4.0f;
    helper_shared_rmw(&xi[0], &xf[0], out);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(11) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn bg_ptr_atomicSub_i32");
    expect(compiled.wgsl).toContain("case 1u: { return atomicSub(&xi[index], value); }");
    expect(compiled.wgsl).toContain("case 2u: { return bg_atomicSub_f32_workgroup(&xf[index], value); }");
    expect([...result.buffers.out as Float32Array]).toEqual([10, 8, 5, 9, 0, 10, 4, 2.5, 2, 9, 5]);
  });

  it("supports pointer-form atomic exchange against shared scalars", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void mark_u32(uint* target, uint value) {
  atomicExch(target, value);
}
__global__ void helper_shared_scalar_exchange(uint* out) {
  __shared__ uint flag;
  if (threadIdx.x == 0) {
    flag = 0u;
    mark_u32(&flag, 7u);
    out[0] = flag;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn bg_ptr_atomicExchange_u32");
    expect(compiled.wgsl).toContain("case 1u: { return atomicExchange(&flag, value); }");
    expect([...result.buffers.out as Uint32Array]).toEqual([7]);
  });

  it("keeps shared scalar atomic pointer parameters distinct from caller shared names", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void mark_u32(uint* flag, uint value) {
  atomicExch(flag, value);
}
__global__ void helper_shared_scalar_collision(uint* out) {
  __shared__ uint flag;
  if (threadIdx.x == 0) {
    flag = 0u;
    mark_u32(&flag, 7u);
    out[0] = flag;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("bg_ptr_atomicExchange_u32(flag_buffer");
    expect(compiled.wgsl).not.toContain("&flag[flag_base]");
    expect([...result.buffers.out as Uint32Array]).toEqual([7]);
  });

  it("lets scalar parameters shadow shared atomic names", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint set_local(uint flag) {
  flag = 9u;
  return flag;
}
__global__ void helper_shared_scalar_param_collision(uint* out) {
  __shared__ uint flag;
  if (threadIdx.x == 0) {
    flag = 0u;
    out[0] = set_local(flag);
    out[1] = flag;
  }
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.wgsl).toContain("flag = 9u");
    expect(compiled.wgsl).not.toContain("atomicStore(&flag, 9u)");
  });

  it("supports pointer-form atomic compare-swap against storage and shared memory", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint cas_u32(uint* target, uint compare, uint value) {
  return atomicCAS(target, compare, value);
}
__global__ void helper_pointer_cas(uint* out, uint* storage) {
  __shared__ uint flag;
  if (threadIdx.x == 0) {
    flag = 3u;
    out[0] = cas_u32(storage, 2u, 9u);
    out[1] = storage[0];
    out[2] = cas_u32(&flag, 3u, 11u);
    out[3] = flag;
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(4), storage: new Uint32Array([2]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect(compiled.wgsl).toContain("fn bg_ptr_atomicCompareExchange_u32");
    expect([...result.buffers.out as Uint32Array]).toEqual([2, 9, 3, 11]);
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

    expect(compiled.wgsl).toContain("copy_one(0u, 0u, 1u, (0u + u32(1))");
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("tile[1u] = 9u;");
    expect(compiled.wgsl).toContain("out[0u] = tile[1u];");
    expect([...result.buffers.out as Uint32Array]).toEqual([9]);
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([9]);
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
    expect(compiled.wgsl).toContain("atomicExchange(&x[0");
    expect(compiled.wgsl).toContain("atomicCompareExchangeWeak(&x[0");
    expect(compiled.wgsl).toContain(".old_value");
    expect(compiled.wgsl).toContain("atomicMax(&x[1");
    expect(compiled.wgsl).toContain("atomicMin(&x[1");
    expect(compiled.wgsl).toContain("atomicSub(&x[1");
    expect(compiled.wgsl).toContain("atomicAnd(&x[2");
    expect(compiled.wgsl).toContain("atomicOr(&x[2");
    expect(compiled.wgsl).toContain("atomicXor(&x[2");
  });

  it("supports CUDA system-scope integer atomic aliases", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomic_system_aliases(int* x, int* out) {
  if (threadIdx.x == 0) {
    out[0] = atomicAdd_system(&x[0], 2);
    out[1] = atomicSub_system(&x[0], 1);
    out[2] = atomicMax_system(&x[0], 5);
    out[3] = atomicMin_system(&x[0], 3);
    out[4] = atomicAnd_system(&x[1], 0x6);
    out[5] = atomicOr_system(&x[1], 0x8);
    out[6] = atomicXor_system(&x[1], 0x3);
    out[7] = atomicInc_system((uint*)&x[2], 2);
    out[8] = atomicDec_system((uint*)&x[2], 2);
    out[9] = atomicExch_system(&x[3], 12);
    out[10] = atomicCAS_system(&x[3], 12, 11);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          x: new Int32Array([4, 7, 1, 9]),
          out: new Int32Array(11),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {
          x: new Int32Array([4, 7, 1, 9]),
          out: new Int32Array(11),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.x as Int32Array]).toEqual([3, 13, 1, 11]);
    expect([...result.buffers.out as Int32Array]).toEqual([4, 6, 5, 5, 7, 6, 14, 1, 2, 9, 12]);
    expect([...semanticResult.buffers.x as Int32Array]).toEqual([3, 13, 1, 11]);
    expect([...semanticResult.buffers.out as Int32Array]).toEqual([4, 6, 5, 5, 7, 6, 14, 1, 2, 9, 12]);
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("out[0u] = atomicAdd(&x[0u], 2);");
    expect(compiled.wgsl).toContain("out[1u] = atomicSub(&x[0u], 1);");
    expect(compiled.wgsl).toContain("out[2u] = atomicMax(&x[0u], 5);");
    expect(compiled.wgsl).toContain("out[3u] = atomicMin(&x[0u], 3);");
    expect(compiled.wgsl).toContain("out[7u] = i32(bg_atomicInc_storage_i32(&x[2u], 2u));");
    expect(compiled.wgsl).toContain("out[8u] = i32(bg_atomicDec_storage_i32(&x[2u], 2u));");
    expect(compiled.wgsl).toContain("out[9u] = atomicExchange(&x[3u], 12);");
    expect(compiled.wgsl).toContain("out[10u] = atomicCompareExchangeWeak(&x[3u], 12, 11).old_value;");
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

    expect([...result.buffers.scratch as Float32Array]).toEqual([11.5, 0, new Float32Array(new Uint32Array([2]).buffer)[0]]);
    expect([...result.buffers.out as Uint32Array]).toEqual([1065353216, 0]);
    expect(compiled.wgsl).toContain("fn bg_atomicInc_storage_f32_as_u32");
    expect(compiled.wgsl).toContain("fn bg_atomicDec_storage_f32_as_u32");
    expect(compiled.wgsl).not.toContain("u32(bitcast<f32>(old_bits))");
    expect(compiled.wgsl).not.toContain("bitcast<u32>(f32(next_value))");
    expect(compiled.wgsl).toContain("bg_atomicInc_storage_f32_as_u32(&scratch[");
    expect(compiled.wgsl).toContain("bg_atomicAdd_f32(&scratch[");
  });

  it("stores vector views through mixed atomic float buffers lane-wise", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void mixed_atomic_vector_store(float* scratch) {
  if (threadIdx.x == 0) {
    uint* flag = (uint*)scratch;
    atomicInc(flag, 8);
    reinterpret_cast<float4*>(scratch + 4)[0] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  }
}`, { workgroupSize: [1, 1, 1] });

    expect(compiled.diagnostics.filter((diagnostic) => diagnostic.severity === "error")).toEqual([]);
    expect(compiled.wgsl).toContain("atomicStore(&scratch[");
    expect(compiled.wgsl).not.toContain("scratch[(0u + u32(4)) +");
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
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Uint32Array]).toEqual([1, 0]);
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1, 0]);
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var<workgroup> counter: array<atomic<u32>, 1>;");
    expect(compiled.wgsl).toContain("out[0u] = bg_atomicInc_workgroup_u32(&counter[0u], 1u);");
    expect(compiled.wgsl).toContain("out[1u] = bg_atomicDec_workgroup_u32(&counter[0u], 1u);");
  });

  it("lowers CUDA atomic inc/dec on storage buffers through semantic IR", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void atomic_inc_dec_storage(uint* data, uint* out) {
  if (threadIdx.x == 0) {
    out[0] = atomicInc(&data[0], 2u);
    out[1] = atomicDec(&data[0], 2u);
    out[2] = data[0];
  }
}`, { workgroupSize: [1, 1, 1] });
    const input = {
      buffers: {
        data: new Uint32Array([1]),
        out: new Uint32Array(3),
      },
    };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
    const result = runCompiledKernelReference(compiled, input, launch);
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {
          data: new Uint32Array([1]),
          out: new Uint32Array(3),
        },
      },
      launch,
    );

    expect([...result.buffers.data as Uint32Array]).toEqual([1]);
    expect([...result.buffers.out as Uint32Array]).toEqual([1, 2, 1]);
    expect([...semanticResult.buffers.data as Uint32Array]).toEqual([1]);
    expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1, 2, 1]);
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("out[0u] = bg_atomicInc_storage_u32(&data[0u], 2u);");
    expect(compiled.wgsl).toContain("out[1u] = bg_atomicDec_storage_u32(&data[0u], 2u);");
  });

  it("supports CUDA atomic inc/dec through helper pointer params", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void helper_inc_dec(uint* counter, uint* out) {
  uint* ptr = counter;
  out[0] = atomicInc(ptr, 2);
  out[1] = atomicDec(ptr, 2);
  out[2] = ptr[0];
}
__device__ void helper_inc_dec_offset(uint* counter, uint* out, int offset) {
  uint* ptr = counter;
  out[offset + 0] = atomicInc(ptr, 2);
  out[offset + 1] = atomicDec(ptr, 2);
  out[offset + 2] = ptr[0];
}
__global__ void helper_atomic_inc_dec(uint* counter, uint* out) {
  __shared__ uint shared_counter[1];
  if (threadIdx.x == 0) {
    helper_inc_dec(counter, out);
    shared_counter[0] = 1;
    helper_inc_dec_offset(&shared_counter[0], out, 3);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          counter: new Uint32Array([1]),
          out: new Uint32Array(6),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.counter as Uint32Array]).toEqual([1]);
    expect([...result.buffers.out as Uint32Array]).toEqual([1, 2, 1, 1, 2, 1]);
    expect(compiled.wgsl).toContain("bg_atomicInc_storage_u32");
    expect(compiled.wgsl).toContain("bg_atomicDec_storage_u32");
    expect(compiled.wgsl).toContain("bg_atomicInc_workgroup_u32");
    expect(compiled.wgsl).toContain("bg_atomicDec_workgroup_u32");
  });

  it("supports CUDA atomic inc/dec through helper pointer params to device globals", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint g_counter[1];

__device__ void helper_global_inc_dec(uint* counter, uint* out) {
  uint* ptr = counter;
  out[0] = atomicInc(ptr, 2);
  out[1] = atomicDec(ptr, 2);
  out[2] = ptr[0];
}

__global__ void helper_global_atomic_inc_dec(uint* out) {
  if (threadIdx.x == 0) {
    helper_global_inc_dec(g_counter, out);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          out: new Uint32Array(3),
        },
        deviceGlobals: {
          g_counter: new Uint32Array([1]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.g_counter as Uint32Array]).toEqual([1]);
    expect([...result.buffers.out as Uint32Array]).toEqual([1, 2, 1]);
    expect(compiled.wgsl).toContain("g_counter: array<atomic<u32>>");
    expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
    expect(compiled.wgsl).not.toContain("var<storage, read_write> out: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("fn bg_ptr_atomicInc_u32");
    expect(compiled.wgsl).toContain("fn bg_ptr_atomicDec_u32");
    expect(compiled.wgsl).toContain("return bg_atomicInc_storage_u32(&g_counter[index], limit);");
    expect(compiled.wgsl).toContain("return bg_atomicDec_storage_u32(&g_counter[index], limit);");
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

    expect(compiled.wgsl).toContain("!(flag[0u] != 0u)");
    expect(compiled.wgsl).not.toContain("!(1u != 4294967295u)");
  });

  it("marks storage atomic after local pointer assignment rebinding", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void assigned_pointer_atomic(uint* counter, uint* out) {
  uint* ptr = NULL;
  if (threadIdx.x == 0) {
    ptr = counter;
    out[0] = atomicAdd(ptr, 1u);
    out[1] = counter[0];
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          counter: new Uint32Array([4]),
          out: new Uint32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.counter as Uint32Array]).toEqual([5]);
    expect([...result.buffers.out as Uint32Array]).toEqual([4, 5]);
    expect(compiled.wgsl).toContain("var<storage, read_write> counter: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
  });

  it("marks all possible atomic roots after branch pointer rebinding", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void branch_assigned_pointer_atomic(uint* left, uint* right, uint* out, int pick_right) {
  uint* ptr = NULL;
  if (pick_right) {
    ptr = right;
  } else {
    ptr = left;
  }
  if (threadIdx.x == 0) {
    out[0] = atomicAdd(ptr, 1u);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          left: new Uint32Array([4]),
          right: new Uint32Array([8]),
          out: new Uint32Array(1),
        },
        scalars: { pick_right: 1 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.left as Uint32Array]).toEqual([4]);
    expect([...result.buffers.right as Uint32Array]).toEqual([9]);
    expect([...result.buffers.out as Uint32Array]).toEqual([8]);
    expect(compiled.wgsl).toContain("var<storage, read_write> left: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> right: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
  });

  it("marks all possible atomic roots after conditional pointer initialization", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void conditional_pointer_atomic(uint* left, uint* right, uint* out, int pick_right) {
  uint* ptr = pick_right ? right : left;
  if (threadIdx.x == 0) {
    out[0] = atomicAdd(ptr, 1u);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          left: new Uint32Array([4]),
          right: new Uint32Array([8]),
          out: new Uint32Array(1),
        },
        scalars: { pick_right: 0 },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.left as Uint32Array]).toEqual([5]);
    expect([...result.buffers.right as Uint32Array]).toEqual([8]);
    expect([...result.buffers.out as Uint32Array]).toEqual([4]);
    expect(compiled.wgsl).toContain("var<storage, read_write> left: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> right: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
  });

  it("marks initial and update roots for for-loop pointer rebinding", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void loop_update_pointer_atomic(uint* left, uint* right, uint* out) {
  uint* ptr = left;
  if (threadIdx.x == 0) {
    for (int i = 0; i < 1; i++, ptr = right) {
      out[0] = atomicAdd(ptr, 1u);
    }
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          left: new Uint32Array([4]),
          right: new Uint32Array([8]),
          out: new Uint32Array(1),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.left as Uint32Array]).toEqual([5]);
    expect([...result.buffers.right as Uint32Array]).toEqual([8]);
    expect([...result.buffers.out as Uint32Array]).toEqual([4]);
    expect(compiled.wgsl).toContain("var<storage, read_write> left: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> right: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
  });

  it("lowers standalone comma expression statements instead of blocking compatibility", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void sequence_stmt(float* out) {
  int i = 0;
  int j = 0;
  if (threadIdx.x == 0) {
    i = 2, j = i + 3, out[0] = (float)j;
    int k = (j = 4, j + 1);
    out[1] = (k = k + 2, (float)k);
  }
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

    expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-sequence-expression");
    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("i = 2;");
    expect(compiled.wgsl).toContain("j = (i + 3);");
    expect(compiled.wgsl).toContain("k = (k + 2);");
    expect([...result.buffers.out as Float32Array]).toEqual([5, 7]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([5, 7]);
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
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("local += 2;");
    expect(compiled.wgsl).toContain("return (local * 3);");
    expect(compiled.wgsl).toContain("local += 3;");
    expect(compiled.wgsl).toContain("local = (local + 4);");
    expect(compiled.wgsl).toContain("return local;");
    expect([...result.buffers.out as Int32Array]).toEqual([18, 12]);
    expect([...semanticResult.buffers.out as Int32Array]).toEqual([18, 12]);
  });

  it("keeps unrelated same-type storage non-atomic for helper pointer atomics", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void add_one(uint* target) {
  atomicAdd(target, 1u);
}

__global__ void exact_helper_atomic(uint* counter, uint* untouched, uint* out) {
  if (threadIdx.x == 0) {
    add_one(counter);
    out[0] = untouched[0];
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          counter: new Uint32Array([4]),
          untouched: new Uint32Array([8]),
          out: new Uint32Array(1),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.counter as Uint32Array]).toEqual([5]);
    expect([...result.buffers.untouched as Uint32Array]).toEqual([8]);
    expect([...result.buffers.out as Uint32Array]).toEqual([8]);
    expect(compiled.wgsl).toContain("var<storage, read_write> counter: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> untouched: array<u32>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
    expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_u32");
    expect(compiled.wgsl).not.toContain("var<storage, read_write> untouched: array<atomic<u32>>;");
  });

  it("ignores unreachable helper atomics when marking device globals atomic", () => {
    const compiled = compileCudaLiteKernel(`
__device__ uint gCounter[1];

__device__ void unused_global_atomic() {
  atomicAdd(&gCounter[0], 1u);
}

__global__ void read_global(uint* out) {
  if (threadIdx.x == 0) {
    out[0] = gCounter[0];
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          out: new Uint32Array(1),
        },
        deviceGlobals: {
          gCounter: new Uint32Array([7]),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Uint32Array]).toEqual([7]);
    expect(backendIr(compiled).atomicDeviceGlobals).not.toContain("gCounter");
    expect(compiled.wgsl).toContain("var<storage, read_write> gCounter: array<u32>;");
    expect(compiled.wgsl).not.toContain("var<storage, read_write> gCounter: array<atomic<u32>>;");
  });

  it("ignores unreachable helper atomics when marking shared storage atomic", () => {
    const compiled = compileCudaLiteKernel(`
__device__ void unused_shared_atomic() {
  __shared__ uint unusedScratch[1];
  atomicAdd(&unusedScratch[0], 1u);
}

__global__ void plain_shared(uint* out) {
  __shared__ uint scratch[1];
  if (threadIdx.x == 0) {
    scratch[0] = 9u;
    out[0] = scratch[0];
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          out: new Uint32Array(1),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Uint32Array]).toEqual([9]);
    expect(backendIr(compiled).atomicShared).not.toContain("unusedScratch");
    expect(backendIr(compiled).atomicShared).not.toContain("scratch");
    expect(compiled.wgsl).toContain("var<workgroup> scratch: array<u32, 1>;");
    expect(compiled.wgsl).not.toContain("var<workgroup> scratch: array<atomic<u32>, 1>;");
    expect(compiled.wgsl).not.toContain("unusedScratch");
  });

  it("marks storage atomic after chained pointer assignment", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void chained_assignment_pointer_atomic(uint* counter, uint* out) {
  uint* a = NULL;
  uint* b = NULL;
  if (threadIdx.x == 0) {
    a = b = counter;
    out[0] = atomicAdd(a, 1u);
    out[1] = b[0];
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          counter: new Uint32Array([4]),
          out: new Uint32Array(2),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.counter as Uint32Array]).toEqual([5]);
    expect([...result.buffers.out as Uint32Array]).toEqual([4, 5]);
    expect(compiled.wgsl).toContain("var<storage, read_write> counter: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
  });

  it("marks storage atomic through local pointer-array elements", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void pointer_array_atomic(uint* counter, uint* untouched, uint* out) {
  uint* ptrs[2];
  if (threadIdx.x == 0) {
    ptrs[0] = counter;
    ptrs[1] = untouched;
    out[0] = atomicAdd(ptrs[0], 1u);
    out[1] = counter[0];
    out[2] = untouched[0];
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      {
        buffers: {
          counter: new Uint32Array([4]),
          untouched: new Uint32Array([8]),
          out: new Uint32Array(3),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.counter as Uint32Array]).toEqual([5]);
    expect([...result.buffers.untouched as Uint32Array]).toEqual([8]);
    expect([...result.buffers.out as Uint32Array]).toEqual([4, 5, 8]);
    expect(compiled.wgsl).toContain("var<storage, read_write> counter: array<atomic<u32>>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> untouched: array<u32>;");
    expect(compiled.wgsl).toContain("var<storage, read_write> out: array<u32>;");
    expect(compiled.wgsl).toContain("fn bg_ptr_atomicAdd_u32(");
    expect(compiled.wgsl).not.toContain("var<storage, read_write> untouched: array<atomic<u32>>;");
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
    expect(compiled.wgsl).toContain("(((bg_uniforms.n + 4) - 1) / 4)");
    expect(compiled.wgsl).toMatch(/var tile_\d+_base: u32 = u32\(4\);/u);
    expect(compiled.wgsl).toMatch(/out\[1\] = u32\(u32\(tile_\d+_base\)\);/u);
    expect(compiled.wgsl).toContain("regs[fill_regs_0][fill_regs_1] = 3.0;");
  });

  it("lowers shared address conversion for multi-dimensional shared lvalues", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void nested_shared_address(uint* out) {
  __shared__ float tile[2][3][4];
  if (threadIdx.x == 0) {
    out[0] = __cvta_generic_to_shared(tile);
    out[1] = __cvta_generic_to_shared(&tile[1][2][3]);
  }
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Uint32Array]).toEqual([0, 23]);
    expect(compiled.wgsl).toContain("out[0] = u32(u32(0u))");
    expect(compiled.wgsl).toContain("out[1] = u32(u32(((u32(1) * 12u) + (u32(2) * 4u) + u32(3))))");
  });

  it("lowers CUDA assignment expression chains as ordered statements", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void chained_assign(float* x, float* out) {
  __shared__ float sdata[4];
  int tid = threadIdx.x;
  float mySum = x[tid];
  sdata[tid] = mySum;
  __syncthreads();
  if (tid == 0) {
    sdata[tid] = mySum = mySum + sdata[tid + 1];
    out[0] = mySum;
    out[1] = sdata[0];
  }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { x: new Float32Array([2, 5]), out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect([...result.buffers.out as Float32Array]).toEqual([7, 7]);
    expect(compiled.wgsl).toContain("mySum = (mySum + sdata[(tid + 1)]);");
    expect(compiled.wgsl).toContain("sdata[tid] = f32(mySum);");
  });

  it("keeps nested updates out of expression contexts", () => {
    expect(() => compileCudaLiteKernel(`
__global__ void bad_update(float* out) {
  int i = 0;
  out[0] = i++;
}`)).toThrow(/side-effect-expression/u);
  });

  it("alpha-renames WGSL reserved and builtin-shadowing CUDA symbols", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void reserved_names(float* array, float* out, float precision) {
  extern __shared__ float shared[];
  float var = array[0];
  float exp = var + precision;
  if (threadIdx.x == 0) {
    shared[0] = exp;
    out[0] = shared[0];
  }
}`, { workgroupSize: [1, 1, 1], dynamicSharedMemory: { shared: 1 } });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { array: new Float32Array([3]), out: new Float32Array(1) }, scalars: { precision: 2 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Float32Array]).toEqual([5]);
    expect(compiled.wgsl).toContain("var<storage, read_write> bg_array: array<f32>;");
    expect(compiled.wgsl).toContain("var<workgroup> bg_shared: array<f32, 1>;");
    expect(compiled.wgsl).toContain("var bg_var: f32 = bg_array[0];");
    expect(compiled.wgsl).toContain("bg_precision: f32,");
    expect(compiled.wgsl).toContain("var bg_exp: f32 = (bg_var + bg_uniforms.bg_precision);");
    expect(compiled.wgsl).not.toContain("var var:");
    expect(compiled.wgsl).not.toContain(" precision:");
  });

  it("supports vector conditionals used by POD-record lowering", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float2 reduce_md(float2 value, float2 other) {
  bool pick = value.x > other.x;
  float2 bigger = pick ? value : other;
  float2 smaller = pick ? other : value;
  float2 result;
  result.y = bigger.y + smaller.y * __expf(smaller.x - bigger.x);
  result.x = bigger.x;
  return result;
}

__global__ void lowered_record(float* out) {
  float2 value = make_float2(out[0], 1.0f);
  float2 other = make_float2(-1.0f, 2.0f);
  __shared__ float2 shared[1];
  shared[0] = reduce_md(value, other);
  if (threadIdx.x == 0) out[0] = shared[0].x + shared[0].y;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array([3]) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(4 + 2 * Math.exp(-4));
    expect(compiled.wgsl).toContain("select(other, value, pick)");
    expect(compiled.wgsl).toContain("var<workgroup> bg_shared: array<vec2<f32>, 1>;");
  });

  it("uses local const integer expressions in later array dimensions", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void local_const_dim(float* out) {
  const int WIDTH = 2;
  __shared__ float scratch[WIDTH];
  scratch[threadIdx.x] = out[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0) out[0] = scratch[0] + scratch[1];
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array([2, 5]) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect([...result.buffers.out as Float32Array]).toEqual([7, 5]);
    expect(compiled.wgsl).toContain("var<workgroup> scratch: array<f32, 2>;");
  });

  it("supports scalar bitwise compound assignments", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void bitwise_compound(int* out) {
  int value = 6;
  value ^= 3;
  value |= 8;
  value &= 14;
  out[0] = value;
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Int32Array(1) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Int32Array]).toEqual([12]);
    expect(compiled.wgsl).toContain("value = (value ^ 3)");
    expect(compiled.wgsl).toContain("value = (value | 8u)");
    expect(compiled.wgsl).toContain("value = (value & 14)");
  });

  it("supports qualified std scalar aliases and functional casts", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void qualified_std_casts(float* out, int q) {
  std::size_t a = std::size_t(q) * 2;
  cuda::std::uint32_t b = cuda::std::uint32_t(a + 1);
  std::ptrdiff_t c = (std::ptrdiff_t)blockIdx.x + 3;
  out[0] = float(a + b + c);
}`, { workgroupSize: [1, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(1) }, scalars: { q: 2 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );

    expect([...result.buffers.out as Float32Array]).toEqual([12]);
    expect(compiled.wgsl).toContain("var a: u32 = (u32(bg_uniforms.q) * 2u);");
    expect(compiled.wgsl).toContain("var b: u32 = u32((a + 1u));");
    expect(compiled.wgsl).toContain("var c: i32 = (i32(workgroup_id.x) + 3);");
  });

  it("lowers vector pack static constructors after source normalization", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void vector_static(float* out) {
  float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 one = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  out[threadIdx.x] = zero[threadIdx.x] + one[threadIdx.x];
}`, { workgroupSize: [4, 1, 1] });
    const result = runCompiledKernelReference(
      compiled,
      { buffers: { out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );
    const semanticResult = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Float32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect([...result.buffers.out as Float32Array]).toEqual([1, 1, 1, 1]);
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 1, 1, 1]);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var zero: vec4<f32> = vec4<f32>(f32(0.0), f32(0.0), f32(0.0), f32(0.0));");
    expect(compiled.wgsl).toContain("out[local_id.x] = (zero[local_id.x] + one[local_id.x]);");
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
    expect(compiled.wgsl).toContain("bg_x_base = (bg_x_base + u32(bg_uniforms.offset));");
    expect(compiled.wgsl).toContain("x[(bg_x_base + u32(0))]");

    const constPointee = compileCudaLiteKernel(`
__global__ void const_pointer_rebase(const uint* x, uint* out, int offset) {
  x += offset;
  if (threadIdx.x == 0) out[0] = *x;
}`, { workgroupSize: [1, 1, 1] });
    const constPointeeResult = runCompiledKernelReference(
      constPointee,
      { buffers: { x: new Uint32Array([10, 20, 30]), out: new Uint32Array(1) }, scalars: { offset: 1 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect([...constPointeeResult.buffers.out as Uint32Array]).toEqual([20]);

    const assignmentRebase = compileCudaLiteKernel(`
__global__ void assign_pointer_rebase(uint* x, uint* out, int offset) {
  x = &x[offset];
  out[0] = x[0];
  x = x + 1;
  out[1] = *x;
}`, { workgroupSize: [1, 1, 1] });
    const assignmentRebaseResult = runCompiledKernelReference(
      assignmentRebase,
      { buffers: { x: new Uint32Array([10, 20, 30]), out: new Uint32Array(2) }, scalars: { offset: 1 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect([...assignmentRebaseResult.buffers.out as Uint32Array]).toEqual([20, 30]);
    expect(assignmentRebase.wgsl).toContain("bg_x_base = (bg_x_base + u32(bg_uniforms.offset));");
    expect(assignmentRebase.wgsl).toContain("bg_x_base = (bg_x_base + u32(1));");

    const nullGuard = compileCudaLiteKernel(`
__global__ void pointer_null_guard(const uint* x, uint* out) {
  if (x != NULL) out[0] = x[0];
  if (x == nullptr) out[1] = 99u;
}`, { workgroupSize: [1, 1, 1] });
    const nullGuardResult = runCompiledKernelReference(
      nullGuard,
      { buffers: { x: new Uint32Array([42]), out: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect([...nullGuardResult.buffers.out as Uint32Array]).toEqual([42, 0]);
    expect(nullGuard.wgsl).toContain("if (true) {");
    expect(nullGuard.wgsl).toContain("if (false) {");

    const pointerIdentity = compileCudaLiteKernel(`
__global__ void pointer_identity(uint* x, uint* y, uint* out) {
  if (x != y) out[0] = 1u;
  if (x == x) out[1] = 2u;
}`, { workgroupSize: [1, 1, 1] });
    const pointerIdentityResult = runCompiledKernelReference(
      pointerIdentity,
      { buffers: { x: new Uint32Array([1]), y: new Uint32Array([1]), out: new Uint32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect([...pointerIdentityResult.buffers.out as Uint32Array]).toEqual([1, 2]);
    expect(pointerIdentity.wgsl).toContain("((0u == 1u) && (0u == 0u))");

    const pointerDistance = compileCudaLiteKernel(`
__global__ void pointer_distance(uint* data, int* out, int left) {
  uint* lptr = data + left;
  uint* rptr = &data[3];
  int nright = rptr - data;
  int width = rptr - lptr;
  out[0] = nright;
  out[1] = width;
}`, { workgroupSize: [1, 1, 1] });
    const pointerDistanceResult = runCompiledKernelReference(
      pointerDistance,
      { buffers: { data: new Uint32Array([10, 20, 30, 40]), out: new Int32Array(2) }, scalars: { left: 1 } },
      { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
    );
    expect([...pointerDistanceResult.buffers.out as Int32Array]).toEqual([3, 2]);
    expect(pointerDistance.wgsl).toContain("i32(");
    expect(pointerDistance.wgsl).toContain("var width: i32 = (i32((0u + u32(3))) - i32((0u + u32((0 + bg_uniforms.left)))));");

    const sharedScalarDistance = compileCudaLiteKernel(`
__global__ void shared_scalar_distance(uint* blocks, uint* out) {
  __shared__ uint start;
  __shared__ uint end;
  __shared__ uint active;
  if (threadIdx.x == 0) {
    start = blocks[0];
    end = blocks[1];
    active = end - start;
    out[0] = active;
  }
}`, { workgroupSize: [1, 1, 1] });
    expect(sharedScalarDistance.wgsl).toContain("bg_active = u32((end - start));");
    expect(sharedScalarDistance.wgsl).not.toContain("select(0, (i32(end) - i32(start))");

    const mismatchedPointerDistance = analyzeCudaLite(parseCudaLite(`
__global__ void bad_pointer_distance(uint* a, float* b, int* out) {
  out[0] = a - b;
}`));
    expect(mismatchedPointerDistance.diagnostics.map((diagnostic) => diagnostic.code)).toContain("unsupported-pointer-difference");

    const constWrite = analyzeCudaLite(parseCudaLite(`
__global__ void bad_const_write(const uint* x) {
  if (threadIdx.x == 0) x[0] = 1u;
}`));
    expect(constWrite.diagnostics.map((diagnostic) => diagnostic.code)).toContain("const-pointer-write");
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
