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

describe("CUDA-lite compiler: Core compiler contracts", () => {
  it("keeps legacy IR and misleading GPU readiness out of public compiler contracts", () => {
      const srcDir = path.join(packageRoot, "src");
      const sources = fs.readdirSync(srcDir)
        .filter((file) => file.endsWith(".ts"))
        .map((file) => [file, compilerSourceText(file)] as const);
      const forbidden = sources.flatMap(([file, source]) => {
        const hits: string[] = [];
        if (/\blegacyIr\b/u.test(source)) hits.push(`${file}:legacyIr`);
        if (/\bcanRunOnGpu\b/u.test(source)) hits.push(`${file}:canRunOnGpu`);
        if (/interface\s+CompiledCudaLiteKernel[\s\S]*?readonly\s+ir\s*:/u.test(source)) hits.push(`${file}:CompiledCudaLiteKernel.ir`);
        return hits;
      });

      expect(forbidden).toEqual([]);
    });

  it("keeps semantic IR traversal and shared expression contracts centralized", () => {
      const srcDir = path.join(packageRoot, "src");
      const duplicates = fs.readdirSync(srcDir)
        .filter((file) => file.endsWith(".ts") && file !== "semantic_ir.ts")
        .flatMap((file) => {
          const source = compilerSourceText(file);
          return [
            "walkSemanticOperations",
            "walkSemanticOperation",
            "walkSemanticExpression",
            "walkSemanticMemoryRef",
          ]
            .filter((name) => new RegExp(`function\\s+${name}\\b`, "u").test(source))
            .map((name) => `${file}:${name}`);
        });

      expect(duplicates).toEqual([]);

      const expressionContractDuplicates = ["semantic_reference.ts", "semantic_wgsl.ts"]
        .flatMap((file) => {
          const source = compilerSourceText(file);
          return [
            "AssignmentOperatorSupported",
            "VectorBinaryOperatorSupported",
            "SurfaceReadValueType",
          ]
            .filter((suffix) => new RegExp(`function\\s+semantic(?:Reference|Wgsl)${suffix}\\b`, "u").test(source))
            .map((suffix) => `${file}:semantic${suffix}`);
        });

      expect(expressionContractDuplicates).toEqual([]);

      const gridSyncPlannerLeaks = ["runtime_plan.ts", "semantic_grid_sync.ts"]
        .flatMap((file) => {
          const source = compilerSourceText(file);
          return [
            "CudaLiteStatement",
            "CudaLiteExpression",
            "KernelIrModule",
            "lowerAnalyzedCudaLiteToKernelIr",
          ]
            .filter((name) => new RegExp(`\\b${name}\\b`, "u").test(source))
            .map((name) => `${file}:${name}`);
        });

      expect(gridSyncPlannerLeaks).toEqual([]);
    });

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

  it("uses semantic kernel IR for fallback reference input validation", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void fallback_validate(float* x) {
    __shared__ float tile[1];
    if (threadIdx.x < 1) { tile[0] = 2.0f; }
    __syncthreads();
    if (threadIdx.x < 1) { x[0] = tile[0]; }
  }`, { workgroupSize: [1, 1, 1] });

      (compiled.analysis.kernel as unknown as { params: unknown[] }).params = [];

      expect(() => runCompiledKernelReference(
        compiled,
        { buffers: { x: new Uint32Array([0]) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      )).toThrow("buffer 'x' expects Float32Array");
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
      expect(localArrayInit.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-array-init");

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
    if (threadIdx.x < 1) { atomicAnd(&x[0], 1); }
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
        code: "unsupported-device-trap",
        message: "__trap cannot be lowered to WebGPU without an explicit device abort contract",
      })).toMatchObject({ family: "safety", gpuRuns: false, referenceRuns: false });
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
      expect(compiled.wgsl).toContain("select");
      expect(compiled.wgsl).toContain("return;");
      expect(compiled.wgsl).toContain("continue;");
      expect(backendIr(compiled).params.map((param) => [param.name, param.valueType])).toContainEqual(["n", "uint"]);
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

  it("compiles stdout-only teaching kernels as no-op WebGPU programs", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void hello() {
    if (threadIdx.x == 0) {
      printf("hello %d\\n", threadIdx.x);
    }
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgslProgram.bindings).toEqual([]);
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("printf omitted");
    });

  it("parses adjacent C string literals in stdout-only calls", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void hello() {
    printf("hello %d "
           "world\\n", threadIdx.x);
  }`, { workgroupSize: [1, 1, 1] });

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("printf omitted");
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

  it("lowers CUDA match_any through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void semanticMatchAnyKernel(uint *input, uint *out) {
    int tid = threadIdx.x;
    out[tid] = __match_any_sync(0xffffffffu, input[tid]);
  }`, {
        features: { subgroups: true },
        workgroupSize: [4, 1, 1],
      });
      const launch = { gridDim: [1, 1, 1], blockDim: [4, 1, 1] } as const;
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { input: new Uint32Array([0, 1, 0, 1]), out: new Uint32Array(4) } },
        launch,
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { input: new Uint32Array([0, 1, 0, 1]), out: new Uint32Array(4) } },
        launch,
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bg_semantic_match_any_uint_32");
      expect([...result.buffers.out as Uint32Array]).toEqual([5, 10, 5, 10]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([5, 10, 5, 10]);
    });

  it("lowers CUDA address-space predicates as native compile-time constants", () => {
      const compiled = compileCudaLiteKernel(`
  __constant__ float caddr[1];
  __global__ void addressPredicates(float *global, int *out) {
    __shared__ float tile[1];
    float local[1];
    if (threadIdx.x == 0) {
      out[0] = __isGlobal(global);
      out[1] = __isGlobal(&global[0]);
      out[2] = __isShared(tile);
      out[3] = __isShared(&tile[0]);
      out[4] = __isConstant(caddr);
      out[5] = __isConstant(&caddr[0]);
      out[6] = __isLocal(local);
      out[7] = __isLocal(&local[0]);
      out[8] = __isShared(global);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: { global: new Float32Array(1), out: new Int32Array(9) },
          constants: { caddr: new Float32Array([3]) },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("// browsergrad-semantic-wgsl: direct semantic IR emission");
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect([...result.buffers.out as Int32Array]).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 0]);
    });

  it("lowers PTX isspacep address predicates as native compile-time constants", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void isspacepPredicates(float *global, int *out) {
    __shared__ float tile[1];
    float local[1];
    int g;
    int s;
    int l;
    int sg;
    asm volatile("isspacep.global %0, %1;" : "=r"(g) : "l"(global));
    asm volatile("isspacep.shared %0, %1;" : "=r"(s) : "l"(tile));
    asm volatile("isspacep.local %0, %1;" : "=r"(l) : "l"(local));
    asm volatile("isspacep.shared %0, %1;" : "=r"(sg) : "l"(global));
    if (threadIdx.x == 0) {
      out[0] = g;
      out[1] = s;
      out[2] = l;
      out[3] = sg;
    }
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { global: new Float32Array(1), out: new Int32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("g = i32(1u)");
      expect(compiled.wgsl).toContain("sg = i32(0u)");
      expect([...result.buffers.out as Int32Array]).toEqual([1, 1, 1, 0]);
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

  it("rejects missing native WebGPU features before pipeline creation", async () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sample(half *out) {
    out[0] = __float2half(1.0f);
  }`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
      const device = { gpu: { features: new Set<string>() } } as never;
      const input = { buffers: { out: createWgslFloat16Array([0]) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };

      await expect(runCompiledKernelWebGpu(device, compiled, input, launch)).rejects.toMatchObject({
        diagnostics: [expect.objectContaining({
          code: "missing-webgpu-device-feature",
          message: "WebGPU device missing required feature(s): shader-f16",
        })],
      });
      await expect(prepareCompiledKernelWebGpu(device, compiled, input, launch)).rejects.toThrow("WebGPU device missing required feature(s): shader-f16");
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
      expect([...runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Int32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      ).buffers.out as Int32Array]).toEqual([9]);
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

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("i += 1;");
      expect(compiled.wgsl).toContain("j -= 1;");
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

      expect(compiled.wgsl).toContain("var even: bool = (u32((idx % 2)) == 0u);");
      expect([...result.buffers.data as Int32Array]).toEqual([11, -8, 13, -6]);
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

  it("lowers CUDA packed dot signed and unsigned intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void dp4a_intrinsics(int *signed_out, uint *unsigned_out) {
    int a = int(0x01ff027fu);
    int b = int(0x0203fe80u);
    int a2 = int(0xfffe007fu);
    uint ua = 0x01020304u;
    uint ub = 0x05060708u;
    uint ua2 = 0x0002007fu;
    uint ub2 = 0x0203fe80u;
    signed_out[0] = __dp4a(a, b, 5);
    signed_out[1] = __dp2a_lo(a2, b, 5);
    signed_out[2] = __dp2a_hi(a2, b, 5);
    unsigned_out[0] = __dp4a(ua, ub, 9u);
    unsigned_out[1] = __dp2a_lo(ua2, ub2, 9u);
    unsigned_out[2] = __dp2a_hi(ua2, ub2, 9u);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { signed_out: new Int32Array(3), unsigned_out: new Uint32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { signed_out: new Int32Array(3), unsigned_out: new Uint32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bg_semantic_dp4a_i32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_dp4a_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_dp2a_i32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_dp2a_u32(");
      expect(compiled.wgsl).toContain("bitcast<i32>(4294836351u)");
      expect([...semanticResult.buffers.signed_out as Int32Array]).toEqual([-16256, -16247, 382]);
      expect([...semanticResult.buffers.unsigned_out as Uint32Array]).toEqual([79, 16773, 394]);
      expect([...result.buffers.signed_out as Int32Array]).toEqual([...semanticResult.buffers.signed_out as Int32Array]);
      expect([...result.buffers.unsigned_out as Uint32Array]).toEqual([...semanticResult.buffers.unsigned_out as Uint32Array]);
    });

  it("lowers CUDA packed byte SIMD intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void packed_byte_simd(uint *out) {
    uint a = 0x10ff807fu;
    uint b = 0x01028081u;
    out[0] = __vadd4(a, b);
    out[1] = __vsub4(a, b);
    out[2] = __vabsdiffu4(a, b);
    out[3] = __vavgu4(a, b);
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

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("fn bg_semantic_vadd4_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vsub4_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vabsdiffu4_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vavgu4_u32(");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([285278208, 268239102, 268238850, 159481984]);
      expect([...result.buffers.out as Uint32Array]).toEqual([...semanticResult.buffers.out as Uint32Array]);
    });

  it("lowers CUDA unsigned saturated SIMD intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void packed_unsigned_saturating_simd(uint *out) {
    uint a = 0x1234ff01u;
    uint b = 0xf0000280u;
    out[0] = __vaddus4(a, b);
    out[1] = __vsubus4(a, b);
    out[2] = __vaddus2(a, b);
    out[3] = __vsubus2(a, b);
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

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("fn bg_semantic_vaddus4_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vsubus4_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vaddus2_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vsubus2_u32(");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([4281663361, 3472640, 4294967295, 64641]);
      expect([...result.buffers.out as Uint32Array]).toEqual([...semanticResult.buffers.out as Uint32Array]);
    });

  it("lowers CUDA signed saturated SIMD intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void packed_signed_saturating_simd(uint *out) {
    uint a4 = 0x7f80ff01u;
    uint b4 = 0x0180017fu;
    uint a2 = 0x7fff8001u;
    uint b2 = 0x00018000u;
    out[0] = __vaddss4(a4, b4);
    out[1] = __vsubss4(a4, b4);
    out[2] = __vaddss2(a2, b2);
    out[3] = __vsubss2(a2, b2);
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

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("fn bg_semantic_vaddss4_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vsubss4_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vaddss2_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vsubss2_u32(");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([2139095167, 2113994370, 2147450880, 2147352577]);
      expect([...result.buffers.out as Uint32Array]).toEqual([...semanticResult.buffers.out as Uint32Array]);
    });

  it("lowers CUDA packed SIMD min/max intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void packed_simd_minmax(uint *out) {
    uint a4 = 0x7f80ff01u;
    uint b4 = 0x0180017fu;
    uint a2 = 0xffff7fffu;
    uint b2 = 0x00018000u;
    out[0] = __vminu4(a4, b4);
    out[1] = __vmaxu4(a4, b4);
    out[2] = __vmins4(a4, b4);
    out[3] = __vmaxs4(a4, b4);
    out[4] = __vminu2(a2, b2);
    out[5] = __vmaxu2(a2, b2);
    out[6] = __vmins2(a2, b2);
    out[7] = __vmaxs2(a2, b2);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("fn bg_semantic_vminu4_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vmaxu4_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vmins4_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vmaxs4_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vminu2_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vmaxu2_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vmins2_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vmaxs2_u32(");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([25166081, 2139160447, 25231105, 2139095423, 98303, 4294934528, 4294934528, 98303]);
      expect([...result.buffers.out as Uint32Array]).toEqual([...semanticResult.buffers.out as Uint32Array]);
    });

  it("lowers CUDA packed SIMD set comparison intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void packed_simd_vset(uint *out) {
    uint same4 = 0x11223344u;
    uint hi4s = 0x7f7f7f7fu;
    uint lo4s = 0x80808080u;
    uint hi4u = 0xffffffffu;
    uint lo4u = 0x01010101u;
    uint same2 = 0x12345678u;
    uint hi2s = 0x7fff7fffu;
    uint lo2s = 0x80008000u;
    uint hi2u = 0xffffffffu;
    uint lo2u = 0x00010001u;
    out[0] = __vseteq4(same4, same4);
    out[1] = __vsetne4(hi4s, lo4s);
    out[2] = __vsetges4(hi4s, lo4s);
    out[3] = __vsetgts4(hi4s, lo4s);
    out[4] = __vsetles4(lo4s, hi4s);
    out[5] = __vsetlts4(lo4s, hi4s);
    out[6] = __vsetgeu4(hi4u, lo4u);
    out[7] = __vsetgtu4(hi4u, lo4u);
    out[8] = __vsetleu4(lo4u, hi4u);
    out[9] = __vsetltu4(lo4u, hi4u);
    out[10] = __vseteq2(same2, same2);
    out[11] = __vsetne2(hi2s, lo2s);
    out[12] = __vsetges2(hi2s, lo2s);
    out[13] = __vsetgts2(hi2s, lo2s);
    out[14] = __vsetles2(lo2s, hi2s);
    out[15] = __vsetlts2(lo2s, hi2s);
    out[16] = __vsetgeu2(hi2u, lo2u);
    out[17] = __vsetgtu2(hi2u, lo2u);
    out[18] = __vsetleu2(lo2u, hi2u);
    out[19] = __vsetltu2(lo2u, hi2u);
    out[20] = __vseteq4(hi4s, lo4s);
    out[21] = __vsetne4(same4, same4);
    out[22] = __vsetgts2(lo2s, hi2s);
    out[23] = __vsetltu2(hi2u, lo2u);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(24) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(24) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("select(0u, 1u");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
      ]);
      expect([...result.buffers.out as Uint32Array]).toEqual([...semanticResult.buffers.out as Uint32Array]);
    });

  it("lowers CUDA packed SIMD compare mask intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void packed_simd_vcmp(uint *out) {
    uint a4 = 0x7f80ff01u;
    uint b4 = 0x0180017fu;
    uint a2 = 0xffff7fffu;
    uint b2 = 0x00018000u;
    out[0] = __vcmpeq4(a4, b4);
    out[1] = __vcmpne4(a4, b4);
    out[2] = __vcmpges4(a4, b4);
    out[3] = __vcmpgeu4(a4, b4);
    out[4] = __vcmpgts4(a4, b4);
    out[5] = __vcmpgtu4(a4, b4);
    out[6] = __vcmples4(a4, b4);
    out[7] = __vcmpleu4(a4, b4);
    out[8] = __vcmplts4(a4, b4);
    out[9] = __vcmpltu4(a4, b4);
    out[10] = __vcmpeq2(a2, b2);
    out[11] = __vcmpne2(a2, b2);
    out[12] = __vcmpges2(a2, b2);
    out[13] = __vcmpgeu2(a2, b2);
    out[14] = __vcmpgts2(a2, b2);
    out[15] = __vcmpgtu2(a2, b2);
    out[16] = __vcmples2(a2, b2);
    out[17] = __vcmpleu2(a2, b2);
    out[18] = __vcmplts2(a2, b2);
    out[19] = __vcmpltu2(a2, b2);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(20) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(20) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("select(0u, 0xffu");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([
        0x00ff0000, 0xff00ffff, 0xffff0000, 0xffffff00, 0xff000000,
        0xff00ff00, 0x00ffffff, 0x00ff00ff, 0x0000ffff, 0x000000ff,
        0x00000000, 0xffffffff, 0x0000ffff, 0xffff0000, 0x0000ffff,
        0xffff0000, 0xffff0000, 0x0000ffff, 0xffff0000, 0x0000ffff,
      ]);
      expect([...result.buffers.out as Uint32Array]).toEqual([...semanticResult.buffers.out as Uint32Array]);
    });

  it("lowers CUDA packed SIMD absolute and SAD intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void packed_simd_abs_sad(uint *out) {
    uint a4 = 0x7f80ff01u;
    uint b4 = 0x8180017fu;
    uint a2 = 0x80017fffu;
    uint b2 = 0x7fff8000u;
    out[0] = __vabs4(a4);
    out[1] = __vabsss4(a4);
    out[2] = __vneg4(a4);
    out[3] = __vnegss4(a4);
    out[4] = __vabsdiffs4(a4, b4);
    out[5] = __vsads4(a4, b4);
    out[6] = __vsadu4(a4, b4);
    out[7] = __vabs2(a2);
    out[8] = __vabsss2(a2);
    out[9] = __vneg2(a2);
    out[10] = __vnegss2(a2);
    out[11] = __vabsdiffs2(a2, b2);
    out[12] = __vsads2(a2, b2);
    out[13] = __vsadu2(a2, b2);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(14) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(14) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("select(0, 256");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([
        0x7f800101, 0x7f7f0101, 0x818001ff, 0x817f01ff,
        0xfe00027e, 0x0000017e, 0x0000017e,
        0x7fff7fff, 0x7fff7fff, 0x7fff8001, 0x7fff8001,
        0xfffeffff, 0x0001fffd, 0x00000003,
      ]);
      expect([...result.buffers.out as Uint32Array]).toEqual([...semanticResult.buffers.out as Uint32Array]);
    });

  it("lowers CUDA packed SIMD average intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void packed_simd_avg(uint *out) {
    uint a4 = 0x7f80ff01u;
    uint b4 = 0x8180017fu;
    uint a2 = 0x80017fffu;
    uint b2 = 0x7fff8000u;
    out[0] = __vhaddu4(a4, b4);
    out[1] = __vavgs4(a4, b4);
    out[2] = __vhaddu2(a2, b2);
    out[3] = __vavgs2(a2, b2);
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

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([
        0x80808040, 0x00800040, 0x80007fff, 0x00000000,
      ]);
      expect([...result.buffers.out as Uint32Array]).toEqual([...semanticResult.buffers.out as Uint32Array]);
    });

  it("lowers CUDA viadd min/max intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void viadd_minmax(uint *out) {
    uint a = 0x0005fffeu;
    uint b = 0x0003fffdu;
    uint c = 0x0007fffcu;
    out[0] = uint(__viaddmax_s32(7, -3, 2));
    out[1] = uint(__viaddmax_s32_relu(-7, 3, -2));
    out[2] = uint(__viaddmin_s32(7, -3, 6));
    out[3] = uint(__viaddmin_s32_relu(-7, 3, -2));
    out[4] = __viaddmax_u32(10u, 5u, 12u);
    out[5] = __viaddmin_u32(10u, 5u, 20u);
    out[6] = __viaddmax_s16x2(a, b, c);
    out[7] = __viaddmax_s16x2_relu(a, b, c);
    out[8] = __viaddmin_s16x2(a, b, c);
    out[9] = __viaddmin_s16x2_relu(a, b, c);
    out[10] = __viaddmax_u16x2(a, b, c);
    out[11] = __viaddmin_u16x2(a, b, c);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(12) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(12) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([
        0x00000004, 0x00000000, 0x00000004, 0x00000000,
        0x0000000f, 0x0000000f, 0x0008fffc, 0x00080000,
        0x0007fffb, 0x00070000, 0x0008fffb, 0x0007fffc,
      ]);
      expect([...result.buffers.out as Uint32Array]).toEqual([...semanticResult.buffers.out as Uint32Array]);
    });

  it("lowers CUDA vimax/vimin relu and tri-minmax intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void viminmax_relu3(uint *out) {
    uint a = 0xfffe8005u;
    uint b = 0x00030004u;
    uint c = 0x0007fff6u;
    out[0] = uint(__vimax_s32_relu(-7, 3));
    out[1] = uint(__vimin_s32_relu(-7, 3));
    out[2] = __vimax_s16x2_relu(a, c);
    out[3] = __vimin_s16x2_relu(a, c);
    out[4] = uint(__vimax3_s32(-7, 3, 2));
    out[5] = uint(__vimax3_s32_relu(-7, -3, -2));
    out[6] = uint(__vimin3_s32(7, -3, 2));
    out[7] = uint(__vimin3_s32_relu(7, -3, 2));
    out[8] = __vimax3_u32(10u, 5u, 12u);
    out[9] = __vimin3_u32(10u, 5u, 12u);
    out[10] = __vimax3_s16x2(a, b, c);
    out[11] = __vimax3_s16x2_relu(a, b, c);
    out[12] = __vimin3_s16x2(a, b, c);
    out[13] = __vimin3_s16x2_relu(a, b, c);
    out[14] = __vimax3_u16x2(a, b, c);
    out[15] = __vimin3_u16x2(a, b, c);
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(16) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(16) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([
        0x00000003, 0x00000000, 0x00070000, 0x00000000,
        0x00000003, 0x00000000, 0xfffffffd, 0x00000000,
        0x0000000c, 0x00000005, 0x00070004, 0x00070004,
        0xfffe8005, 0x00000000, 0xfffefff6, 0x00030004,
      ]);
      expect([...result.buffers.out as Uint32Array]).toEqual([...semanticResult.buffers.out as Uint32Array]);
    });

  it("lowers CUDA vib min/max predicate intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void vib_predicate(uint *out) {
    bool pred = 0;
    bool pred_hi = 0;
    bool pred_lo = 0;
    int s = __vibmax_s32(7, 3, &pred);
    out[0] = uint(s);
    out[1] = pred ? 1u : 0u;
    s = __vibmin_s32(7, -3, &pred);
    out[2] = uint(s);
    out[3] = pred ? 1u : 0u;
    __vibmax_u32(10u, 12u, &pred);
    out[4] = pred ? 1u : 0u;
    uint u = __vibmin_u32(10u, 12u, &pred);
    out[5] = u;
    out[6] = pred ? 1u : 0u;
    uint a = 0xfffe8005u;
    uint b = 0x00030004u;
    out[7] = __vibmax_s16x2(a, b, &pred_hi, &pred_lo);
    out[8] = pred_hi ? 1u : 0u;
    out[9] = pred_lo ? 1u : 0u;
    out[10] = __vibmin_s16x2(a, b, &pred_hi, &pred_lo);
    out[11] = pred_hi ? 1u : 0u;
    out[12] = pred_lo ? 1u : 0u;
    out[13] = __vibmax_u16x2(a, b, &pred_hi, &pred_lo);
    out[14] = pred_hi ? 1u : 0u;
    out[15] = pred_lo ? 1u : 0u;
    out[16] = __vibmin_u16x2(a, b, &pred_hi, &pred_lo);
    out[17] = pred_hi ? 1u : 0u;
    out[18] = pred_lo ? 1u : 0u;
  }`, { workgroupSize: [1, 1, 1] });
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(19) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([
        0x00000007, 0x00000001, 0xfffffffd, 0x00000000,
        0x00000000, 0x0000000a, 0x00000001, 0x00030004,
        0x00000000, 0x00000000, 0xfffe8005, 0x00000001,
        0x00000001, 0xfffe8005, 0x00000001, 0x00000001,
        0x00030004, 0x00000000, 0x00000000,
      ]);
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
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Uint32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("workgroup_id.x * 104729u");
      expect([...result.buffers.out as Uint32Array]).toEqual([0, 1, 2, 3, 0, 1, 2, 3]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([0, 1, 2, 3, 0, 1, 2, 3]);
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

  it("lowers CUDA log-normal cuRAND draws through the deterministic browser RNG island", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void curandLogNormalKernel(float *out) {
    curandState_t state;
    curand_init(23ULL, threadIdx.x, 5, &state);
    float x = curand_log_normal(&state, 0.25f, 0.5f);
    float y = curand_log_normal_double(&state, 0.1f, 0.2f);
    out[threadIdx.x] = x + y;
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("fn bg_curand_log_normal");
      expect(compiled.wgsl).toContain("bg_curand_log_normal(&state, 0.25, 0.5)");
      expect([...result.buffers.out as Float32Array].every((value) => Number.isFinite(value) && value > 0)).toBe(true);
    });

  it("lowers raw CUDA cuRAND draws through the deterministic browser RNG island", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void curandRawKernel(unsigned int *out) {
    curandState_t state;
    curand_init(31ULL, threadIdx.x, 7, &state);
    unsigned int x = curand(&state);
    unsigned int y = curand(&state);
    out[threadIdx.x] = x ^ y;
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("fn bg_curand(state: ptr<function, u32>) -> u32");
      expect(compiled.wgsl).toContain("var x: u32 = u32(bg_curand(&state))");
      expect([...result.buffers.out as Uint32Array].some((value) => value !== 0)).toBe(true);
    });

  it("lowers CUDA cuRAND Poisson draws through semantic IR", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __global__ void curandPoissonKernel(unsigned int *out, unsigned int seed) {
    curandState_t state;
    curand_init(seed, threadIdx.x, 0, &state);
    unsigned int small = curand_poisson(&state, 3.0f);
    unsigned int large = curand_poisson(&state, 72.0f);
    out[threadIdx.x] = small + large;
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(4) }, scalars: { seed: 1357 } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bg_curand_poisson(state: ptr<function, u32>, lambda: f32) -> u32");
      expect(compiled.wgsl).toContain("bg_curand_poisson(&state, 3.0)");
      expect(JSON.stringify(compiled.kernelIr)).toContain("curand_poisson");
      expect([...result.buffers.out as Uint32Array].every((value) => value > 0)).toBe(true);
    });

  it("lowers CUDA cuRAND skipahead through semantic IR", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __global__ void curandSkipaheadKernel(unsigned int *out, unsigned int seed) {
    curandState_t a;
    curandState_t b;
    curand_init(seed, threadIdx.x, 0, &a);
    curand_init(seed, threadIdx.x, 0, &b);
    unsigned int discarded0 = curand(&a);
    unsigned int discarded1 = curand(&a);
    skipahead(2, &b);
    unsigned int nextA = curand(&a);
    unsigned int nextB = curand(&b);
    out[threadIdx.x] = nextA ^ nextB ^ discarded0 ^ discarded1;
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(4) }, scalars: { seed: 9753 } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bg_curand_advance(state: u32, count: u32) -> u32");
      expect(compiled.wgsl).toContain("bg_curand_skipahead(2u, &b)");
      expect(JSON.stringify(compiled.kernelIr)).toContain("skipahead");
      expect([...result.buffers.out as Uint32Array].some((value) => value !== 0)).toBe(true);
    });

  it("lowers cuRAND calls against shared-memory state arrays", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __global__ void sharedRNG(float *out, unsigned int seed) {
    __shared__ curandState_t states[4];
    unsigned int tid = threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
    out[tid] = curand_uniform(&states[tid]) + curand_normal(&states[tid]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(4) }, scalars: { seed: 1234 } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var<workgroup> states: array<u32, 4>;");
      expect(compiled.wgsl).toContain("fn bg_curand_init_workgroup");
      expect(compiled.wgsl).toContain("bg_curand_init_workgroup(bg_uniforms.seed, tid, 0u, &states[tid])");
      expect([...result.buffers.out as Float32Array].every((value) => Number.isFinite(value))).toBe(true);
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

  it("lowers inline PTX fma.rn.f32 literal source statements", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void asmFmaImmediate(const float *A, const float *B, float *out) {
    int idx = threadIdx.x;
    float sum = out[idx];
    float direct;
    float literalMix;
    asm volatile("fma.rn.f32 %0, %1, 2.0f, %0;" : "+f"(sum) : "f"(A[idx]));
    asm volatile("fma.rn.f32 %0, 3.0f, %1, 1.0f;" : "=f"(direct) : "f"(B[idx]));
    asm volatile("fma.rn.f32 %0, 4.0f, 0.5f, %1;" : "=f"(literalMix) : "f"(A[idx]));
    out[idx] = sum + direct + literalMix;
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

      expect(compiled.wgsl).toContain("sum = fma(A[idx], 2.0, sum);");
      expect(compiled.wgsl).toContain("direct = fma(3.0, B[idx], 1.0);");
      expect(compiled.wgsl).toContain("literalMix = fma(4.0, 0.5, A[idx]);");
      expect([...result.buffers.out as Float32Array]).toEqual([31, 47]);
    });

  it("lowers output-only inline PTX lane id statements", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void laneId(int *out) {
    int idx = threadIdx.x;
    unsigned int laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    out[idx] = laneid;
  }`, { workgroupSize: [4, 1, 1] });
      const input = { buffers: { out: new Int32Array(4) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [4, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("& 31u");
      expect([...semanticResult.buffers.out as Int32Array]).toEqual([0, 1, 2, 3]);
      expect([...result.buffers.out as Int32Array]).toEqual([0, 1, 2, 3]);
    });

  it("parses inline PTX clobbers after empty input sections for supported ops", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void laneIdClobber(uint *out) {
    int idx = threadIdx.x;
    unsigned int laneid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid) :: "memory");
    out[idx] = laneid;
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("& 31u");
      expect([...result.buffers.out as Uint32Array]).toEqual([0, 1, 2, 3]);
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

  it("lowers single-percent inline PTX special register aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void singlePercentSpecialRegs(uint *out) {
    int idx = threadIdx.x;
    unsigned int lane;
    unsigned int warp;
    unsigned int mask;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(lane));
    asm volatile("mov.u32 %0, %warpid;" : "=r"(warp));
    asm volatile("mov.u32 %0, %lanemask_lt;" : "=r"(mask));
    out[idx] = lane;
    out[idx + 4] = warp;
    out[idx + 8] = mask;
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(12) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect([...result.buffers.out as Uint32Array]).toEqual([0, 1, 2, 3, 0, 0, 0, 0, 0, 1, 3, 7]);
    });

  it("lowers inline PTX CUDA dimension special registers through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void dimensionSpecialRegs(uint *out) {
    uint tx;
    uint ty;
    uint bx;
    uint by;
    uint ntx;
    uint nty;
    uint nbx;
    uint nby;
    asm volatile("mov.u32 %0, %tid.x;" : "=r"(tx));
    asm volatile("mov.u32 %0, %tid.y;" : "=r"(ty));
    asm volatile("mov.u32 %0, %ctaid.x;" : "=r"(bx));
    asm volatile("mov.u32 %0, %ctaid.y;" : "=r"(by));
    asm volatile("mov.u32 %0, %ntid.x;" : "=r"(ntx));
    asm volatile("mov.u32 %0, %ntid.y;" : "=r"(nty));
    asm volatile("mov.u32 %0, %nctaid.x;" : "=r"(nbx));
    asm volatile("mov.u32 %0, %nctaid.y;" : "=r"(nby));
    uint idx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    out[idx] = tx;
    out[idx + 8] = ty;
    out[idx + 16] = bx;
    out[idx + 24] = by;
    out[idx + 32] = ntx;
    out[idx + 40] = nty;
    out[idx + 48] = nbx;
    out[idx + 56] = nby;
  }`, { workgroupSize: [2, 2, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(64) } },
        { gridDim: [2, 1, 1], blockDim: [2, 2, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0, 1, 0, 1, 0, 1, 0, 1,
        0, 0, 1, 1, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1,
      ]);
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

  it("lowers inline PTX popc.b32 statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int popc_ptx(unsigned int word) {
    unsigned int ret;
    asm volatile("popc.b32 %0, %1;" : "=r"(ret) : "r"(word));
    return ret;
  }
  __global__ void popcKernel(uint *out, uint *input) {
    int idx = threadIdx.x;
    out[idx] = popc_ptx(input[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(4), input: new Uint32Array([0, 1, 0xf0f0f0f0, 0xffffffff]) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("countOneBits");
      expect([...result.buffers.out as Uint32Array]).toEqual([0, 1, 16, 32]);
    });

  it("lowers inline PTX ffs.b32 statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ int ffs_ptx(unsigned int word) {
    int ret;
    asm volatile("ffs.b32 %0, %1;" : "=r"(ret) : "r"(word));
    return ret;
  }
  __global__ void ffsKernel(int *out, uint *input) {
    int idx = threadIdx.x;
    out[idx] = ffs_ptx(input[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Int32Array(4), input: new Uint32Array([0, 1, 8, 0x80000000]) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("countTrailingZeros");
      expect([...result.buffers.out as Int32Array]).toEqual([0, 1, 4, 32]);
    });

  it("lowers inline PTX clz.b32 and brev.b32 statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int clz_ptx(unsigned int word) {
    unsigned int ret;
    asm volatile("clz.b32 %0, %1;" : "=r"(ret) : "r"(word));
    return ret;
  }
  __device__ unsigned int brev_ptx(unsigned int word) {
    unsigned int ret;
    asm volatile("brev.b32 %0, %1;" : "=r"(ret) : "r"(word));
    return ret;
  }
  __global__ void bitKernel(uint *out, uint *input) {
    int idx = threadIdx.x;
    out[idx] = clz_ptx(input[idx]);
    out[idx + 4] = brev_ptx(input[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(8), input: new Uint32Array([0, 1, 0x01234567, 0xffffffff]) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("countLeadingZeros");
      expect(compiled.wgsl).toContain("reverseBits");
      expect([...result.buffers.out as Uint32Array]).toEqual([32, 31, 7, 0, 0, 0x80000000, 0xe6a2c480, 0xffffffff]);
    });

  it("lowers inline PTX bit-scan/count immediate statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int bfind_imm_ptx() {
    unsigned int ret;
    asm volatile("bfind.u32 %0, 0x00008000;" : "=r"(ret));
    return ret;
  }
  __device__ unsigned int ffs_imm_ptx() {
    unsigned int ret;
    asm volatile("ffs.b32 %0, 0x00008000;" : "=r"(ret));
    return ret;
  }
  __device__ unsigned int popc_imm_ptx() {
    unsigned int ret;
    asm volatile("popc.b32 %0, 0xf0f0f00f;" : "=r"(ret));
    return ret;
  }
  __device__ unsigned int clz_imm_ptx() {
    unsigned int ret;
    asm volatile("clz.b32 %0, 0x00008000;" : "=r"(ret));
    return ret;
  }
  __device__ unsigned int brev_imm_ptx() {
    unsigned int ret;
    asm volatile("brev.b32 %0, 0x01234567;" : "=r"(ret));
    return ret;
  }
  __global__ void bitScanImmediateKernel(uint *out) {
    int idx = threadIdx.x;
    out[idx] = bfind_imm_ptx();
    out[idx + 4] = ffs_imm_ptx();
    out[idx + 8] = popc_imm_ptx();
    out[idx + 12] = clz_imm_ptx();
    out[idx + 16] = brev_imm_ptx();
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(20) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("countTrailingZeros");
      expect(compiled.wgsl).toContain("countOneBits");
      expect(compiled.wgsl).toContain("reverseBits");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        15,
        15,
        15,
        15,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        0xe6a2c480,
        0xe6a2c480,
        0xe6a2c480,
        0xe6a2c480,
      ]);
    });

  it("lowers inline PTX prmt.b32 statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int prmt_ptx(unsigned int x, unsigned int y, unsigned int selector) {
    unsigned int ret;
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(y), "r"(selector));
    return ret;
  }
  __global__ void prmtKernel(uint *out, uint *selector) {
    int idx = threadIdx.x;
    out[idx] = prmt_ptx(0x80112233u, 0x445566f7u, selector[idx]);
  }`, { workgroupSize: [5, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(5), selector: new Uint32Array([0x3210, 0x5410, 0x7654, 0x0000, 0xb210]) } },
        { gridDim: [1, 1, 1], blockDim: [5, 1, 1] },
      );

      expect(compiled.wgsl).toContain("& 0xfu");
      expect([...result.buffers.out as Uint32Array]).toEqual([0x80112233, 0x66f72233, 0x445566f7, 0x33333333, 0xff112233]);
    });

  it("lowers inline PTX prmt.b32 immediate selector statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int prmt_5410_ptx(unsigned int x, unsigned int y) {
    unsigned int ret;
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(ret) : "r"(x), "r"(y));
    return ret;
  }
  __device__ unsigned int prmt_b210_ptx(unsigned int x, unsigned int y) {
    unsigned int ret;
    asm volatile("prmt.b32 %0, %1, %2, 0xb210;" : "=r"(ret) : "r"(x), "r"(y));
    return ret;
  }
  __global__ void prmtImmediateKernel(uint *out) {
    int idx = threadIdx.x;
    out[idx] = prmt_5410_ptx(0x80112233u, 0x445566f7u);
    out[idx + 4] = prmt_b210_ptx(0x80112233u, 0x445566f7u);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("21520u");
      expect(compiled.wgsl).toContain("45584u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0x66f72233,
        0x66f72233,
        0x66f72233,
        0x66f72233,
        0xff112233,
        0xff112233,
        0xff112233,
        0xff112233,
      ]);
    });

  it("lowers inline PTX lop3.b32 statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int choose_ptx(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int ret;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xca;" : "=r"(ret) : "r"(a), "r"(b), "r"(c));
    return ret;
  }
  __device__ unsigned int xor_ptx(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int ret;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(ret) : "r"(a), "r"(b), "r"(c), "n"(0x96));
    return ret;
  }
  __global__ void lop3Kernel(uint *out, uint *a, uint *b, uint *c) {
    int idx = threadIdx.x;
    out[idx] = choose_ptx(a[idx], b[idx], c[idx]);
    out[idx + 4] = xor_ptx(a[idx], b[idx], c[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(8),
            a: new Uint32Array([0xffffffff, 0, 0xf0f0f0f0, 0xaaaaaaaa]),
            b: new Uint32Array([0x12345678, 0x12345678, 0x0f0f0f0f, 0x55555555]),
            c: new Uint32Array([0x87654321, 0x87654321, 0xffffffff, 0x33333333]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("202u");
      expect(compiled.wgsl).toContain("0x96");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0x12345678,
        0x87654321,
        0x0f0f0f0f,
        0x11111111,
        0x6aaeeaa6,
        0x95511559,
        0x00000000,
        0xcccccccc,
      ]);
    });

  it("lowers inline PTX lop3.b32 immediate data statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int choose_literal_b_ptx(unsigned int a, unsigned int c) {
    unsigned int ret;
    asm volatile("lop3.b32 %0, %1, 0x55555555, %2, 0xca;" : "=r"(ret) : "r"(a), "r"(c));
    return ret;
  }
  __device__ unsigned int choose_all_literal_ptx() {
    unsigned int ret;
    asm volatile("lop3.b32 %0, 0xffffffff, 0x12345678, 0x87654321, 0xca;" : "=r"(ret));
    return ret;
  }
  __global__ void lop3ImmediateKernel(uint *out, uint *a) {
    int idx = threadIdx.x;
    out[idx] = choose_literal_b_ptx(a[idx], 0x33333333u);
    out[idx + 4] = choose_all_literal_ptx();
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(8), a: new Uint32Array([0xffffffff, 0, 0xf0f0f0f0, 0xaaaaaaaa]) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("1431655765u");
      expect(compiled.wgsl).toContain("202u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0x55555555,
        0x33333333,
        0x53535353,
        0x11111111,
        0x12345678,
        0x12345678,
        0x12345678,
        0x12345678,
      ]);
    });

  it("lowers inline PTX unary/not b32 immediate statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int not_imm_ptx() {
    unsigned int ret;
    asm volatile("not.b32 %0, 0x0f0f0f0f;" : "=r"(ret));
    return ret;
  }
  __device__ int neg_imm_ptx() {
    int ret;
    asm volatile("neg.s32 %0, -5;" : "=r"(ret));
    return ret;
  }
  __device__ int abs_imm_ptx() {
    int ret;
    asm volatile("abs.s32 %0, -8;" : "=r"(ret));
    return ret;
  }
  __global__ void unaryImmediateKernel(uint *out) {
    int idx = threadIdx.x;
    out[idx] = not_imm_ptx();
    out[idx + 4] = (uint)neg_imm_ptx();
    out[idx + 8] = (uint)abs_imm_ptx();
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(12) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("252645135u");
      expect(compiled.wgsl).toContain("4294967291u");
      expect(compiled.wgsl).toContain("4294967288u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0xf0f0f0f0,
        0xf0f0f0f0,
        0xf0f0f0f0,
        0xf0f0f0f0,
        5,
        5,
        5,
        5,
        8,
        8,
        8,
        8,
      ]);
    });

  it("lowers inline PTX shift b32 statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int shl_ptx(unsigned int value, unsigned int shift) {
    unsigned int ret;
    asm volatile("shl.b32 %0, %1, %2;" : "=r"(ret) : "r"(value), "r"(shift));
    return ret;
  }
  __device__ unsigned int shr_u_ptx(unsigned int value, unsigned int shift) {
    unsigned int ret;
    asm volatile("shr.u32 %0, %1, %2;" : "=r"(ret) : "r"(value), "r"(shift));
    return ret;
  }
  __device__ int shr_s_ptx(int value, unsigned int shift) {
    int ret;
    asm volatile("shr.s32 %0, %1, %2;" : "=r"(ret) : "r"(value), "r"(shift));
    return ret;
  }
  __global__ void shiftKernel(uint *out, uint *input, uint *amount) {
    int idx = threadIdx.x;
    out[idx] = shl_ptx(input[idx], amount[idx]);
    out[idx + 5] = shr_u_ptx(input[idx], amount[idx]);
    out[idx + 10] = (uint)shr_s_ptx((int)input[idx], amount[idx]);
  }`, { workgroupSize: [5, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(15),
            input: new Uint32Array([1, 0x80000000, 0xf0000000, 0x7fffffff, 0x12345678]),
            amount: new Uint32Array([0, 1, 4, 31, 32]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [5, 1, 1] },
      );

      expect(compiled.wgsl).toContain("min(");
      expect(compiled.wgsl).toContain(">= 32u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        1,
        0,
        0,
        0x80000000,
        0,
        1,
        0x40000000,
        0x0f000000,
        0,
        0,
        1,
        0xc0000000,
        0xff000000,
        0,
        0,
      ]);
    });

  it("lowers inline PTX shift b32 immediate statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int shl_imm_ptx(unsigned int value) {
    unsigned int ret;
    asm volatile("shl.b32 %0, %1, 4;" : "=r"(ret) : "r"(value));
    return ret;
  }
  __device__ unsigned int shr_u_imm_ptx(unsigned int value) {
    unsigned int ret;
    asm volatile("shr.u32 %0, %1, 4;" : "=r"(ret) : "r"(value));
    return ret;
  }
  __device__ int shr_s_imm_ptx(int value) {
    int ret;
    asm volatile("shr.s32 %0, %1, 4;" : "=r"(ret) : "r"(value));
    return ret;
  }
  __global__ void shiftImmediateKernel(uint *out, uint *input) {
    int idx = threadIdx.x;
    out[idx] = shl_imm_ptx(input[idx]);
    out[idx + 5] = shr_u_imm_ptx(input[idx]);
    out[idx + 10] = (uint)shr_s_imm_ptx((int)input[idx]);
  }`, { workgroupSize: [5, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(15),
            input: new Uint32Array([1, 0x80000000, 0xf0000000, 0x7fffffff, 0x12345678]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [5, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("4u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0x10,
        0,
        0,
        0xfffffff0,
        0x23456780,
        0,
        0x08000000,
        0x0f000000,
        0x07ffffff,
        0x01234567,
        0,
        0xf8000000,
        0xff000000,
        0x07ffffff,
        0x01234567,
      ]);
    });

  it("lowers inline PTX arithmetic b32 statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int add_ptx(unsigned int a, unsigned int b) {
    unsigned int ret;
    asm volatile("add.u32 %0, %1, %2;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __device__ int sub_ptx(int a, int b) {
    int ret;
    asm volatile("sub.s32 %0, %1, %2;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __device__ unsigned int mul_lo_ptx(unsigned int a, unsigned int b) {
    unsigned int ret;
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __device__ unsigned int mad_lo_ptx(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int ret;
    asm volatile("mad.lo.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(a), "r"(b), "r"(c));
    return ret;
  }
  __global__ void arithmeticKernel(uint *out, uint *a, uint *b, uint *c) {
    int idx = threadIdx.x;
    out[idx] = add_ptx(a[idx], b[idx]);
    out[idx + 4] = (uint)sub_ptx((int)a[idx], (int)b[idx]);
    out[idx + 8] = mul_lo_ptx(a[idx], b[idx]);
    out[idx + 12] = mad_lo_ptx(a[idx], b[idx], c[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(16),
            a: new Uint32Array([1, 0xffffffff, 0x80000000, 0x12345678]),
            b: new Uint32Array([2, 2, 2, 0x87654321]),
            c: new Uint32Array([5, 7, 9, 0xffffffff]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain(" + ");
      expect(compiled.wgsl).toContain(" - ");
      expect(compiled.wgsl).toContain(" * ");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        3,
        1,
        0x80000002,
        0x99999999,
        0xffffffff,
        0xfffffffd,
        0x7ffffffe,
        0x8acf1357,
        2,
        0xfffffffe,
        0,
        0x70b88d78,
        7,
        5,
        9,
        0x70b88d77,
      ]);
    });

  it("lowers inline PTX arithmetic b32 immediate statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int add_imm_ptx(unsigned int a) {
    unsigned int ret;
    asm volatile("add.u32 %0, %1, 5;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __device__ int sub_imm_ptx(int a) {
    int ret;
    asm volatile("sub.s32 %0, %1, 7;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __device__ unsigned int mul_lo_imm_ptx(unsigned int a) {
    unsigned int ret;
    asm volatile("mul.lo.u32 %0, %1, 3;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __device__ unsigned int mad_lo_imm_ptx(unsigned int a, unsigned int b) {
    unsigned int ret;
    asm volatile("mad.lo.u32 %0, %1, %2, 11;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __global__ void arithmeticImmediateKernel(uint *out, uint *a, uint *b) {
    int idx = threadIdx.x;
    out[idx] = add_imm_ptx(a[idx]);
    out[idx + 4] = (uint)sub_imm_ptx((int)a[idx]);
    out[idx + 8] = mul_lo_imm_ptx(a[idx]);
    out[idx + 12] = mad_lo_imm_ptx(a[idx], b[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(16),
            a: new Uint32Array([1, 0xffffffff, 0x80000000, 0x12345678]),
            b: new Uint32Array([2, 2, 2, 0x87654321]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("5u");
      expect(compiled.wgsl).toContain("11u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        6,
        4,
        0x80000005,
        0x1234567d,
        0xfffffffa,
        0xfffffff8,
        0x7ffffff9,
        0x12345671,
        3,
        0xfffffffd,
        0x80000000,
        0x369d0368,
        13,
        9,
        11,
        0x70b88d83,
      ]);
    });

  it("lowers inline PTX min/max u32 and s32 statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int min_u_ptx(unsigned int a, unsigned int b) {
    unsigned int ret;
    asm volatile("min.u32 %0, %1, %2;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __device__ unsigned int max_u_ptx(unsigned int a, unsigned int b) {
    unsigned int ret;
    asm volatile("max.u32 %0, %1, %2;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __device__ int min_s_ptx(int a, int b) {
    int ret;
    asm volatile("min.s32 %0, %1, %2;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __device__ int max_s_ptx(int a, int b) {
    int ret;
    asm volatile("max.s32 %0, %1, %2;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __global__ void minMaxKernel(uint *out, uint *a, uint *b) {
    int idx = threadIdx.x;
    out[idx] = min_u_ptx(a[idx], b[idx]);
    out[idx + 4] = max_u_ptx(a[idx], b[idx]);
    out[idx + 8] = (uint)min_s_ptx((int)a[idx], (int)b[idx]);
    out[idx + 12] = (uint)max_s_ptx((int)a[idx], (int)b[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(16),
            a: new Uint32Array([1, 0xffffffff, 0x80000000, 0x7fffffff]),
            b: new Uint32Array([2, 2, 0x7fffffff, 0x80000000]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("min(");
      expect(compiled.wgsl).toContain("max(");
      expect(compiled.wgsl).toContain("bitcast<i32>");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        1,
        2,
        0x7fffffff,
        0x7fffffff,
        2,
        0xffffffff,
        0x80000000,
        0x80000000,
        1,
        0xffffffff,
        0x80000000,
        0x80000000,
        2,
        2,
        0x7fffffff,
        0x7fffffff,
      ]);
    });

  it("lowers inline PTX min/max u32 and s32 immediate statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int min_u_imm_ptx(unsigned int a) {
    unsigned int ret;
    asm volatile("min.u32 %0, %1, 0x7fffffff;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __device__ unsigned int max_u_imm_ptx(unsigned int a) {
    unsigned int ret;
    asm volatile("max.u32 %0, %1, 2;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __device__ int min_s_imm_ptx(int a) {
    int ret;
    asm volatile("min.s32 %0, %1, -8;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __device__ int max_s_imm_ptx(int a) {
    int ret;
    asm volatile("max.s32 %0, %1, 7;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __global__ void minMaxImmediateKernel(uint *out, uint *a) {
    int idx = threadIdx.x;
    out[idx] = min_u_imm_ptx(a[idx]);
    out[idx + 4] = max_u_imm_ptx(a[idx]);
    out[idx + 8] = (uint)min_s_imm_ptx((int)a[idx]);
    out[idx + 12] = (uint)max_s_imm_ptx((int)a[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(16),
            a: new Uint32Array([1, 0xffffffff, 0x80000000, 0x7fffffff]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("2147483647u");
      expect(compiled.wgsl).toContain("4294967288u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        1,
        0x7fffffff,
        0x7fffffff,
        0x7fffffff,
        2,
        0xffffffff,
        0x80000000,
        0x7fffffff,
        0xfffffff8,
        0xfffffff8,
        0x80000000,
        0xfffffff8,
        7,
        7,
        7,
        0x7fffffff,
      ]);
    });

  it("lowers inline PTX signed neg and abs b32 statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ int neg_ptx(int value) {
    int ret;
    asm volatile("neg.s32 %0, %1;" : "=r"(ret) : "r"(value));
    return ret;
  }
  __device__ int abs_ptx(int value) {
    int ret;
    asm volatile("abs.s32 %0, %1;" : "=r"(ret) : "r"(value));
    return ret;
  }
  __global__ void unaryIntKernel(uint *out, uint *input) {
    int idx = threadIdx.x;
    out[idx] = (uint)neg_ptx((int)input[idx]);
    out[idx + 5] = (uint)abs_ptx((int)input[idx]);
  }`, { workgroupSize: [5, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(10),
            input: new Uint32Array([0, 1, 0xffffffff, 0x80000000, 0x7fffffff]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [5, 1, 1] },
      );

      expect(compiled.wgsl).toContain("0u - ");
      expect(compiled.wgsl).toContain("0x80000000u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0,
        0xffffffff,
        1,
        0x80000000,
        0x80000001,
        0,
        1,
        1,
        0x80000000,
        0x7fffffff,
      ]);
    });

  it("lowers inline PTX selp b32 statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int selp_u_ptx(unsigned int a, unsigned int b, unsigned int p) {
    unsigned int ret;
    asm volatile("selp.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(a), "r"(b), "r"(p));
    return ret;
  }
  __device__ int selp_s_ptx(int a, int b, unsigned int p) {
    int ret;
    asm volatile("selp.s32 %0, %1, %2, %3;" : "=r"(ret) : "r"(a), "r"(b), "r"(p));
    return ret;
  }
  __global__ void selectKernel(uint *out, uint *a, uint *b, uint *pred) {
    int idx = threadIdx.x;
    out[idx] = selp_u_ptx(a[idx], b[idx], pred[idx]);
    out[idx + 4] = (uint)selp_s_ptx((int)a[idx], (int)b[idx], pred[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(8),
            a: new Uint32Array([1, 0xffffffff, 0x80000000, 0x12345678]),
            b: new Uint32Array([2, 3, 0x7fffffff, 0x87654321]),
            pred: new Uint32Array([0, 1, 2, 0]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("select(");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        2,
        0xffffffff,
        0x80000000,
        0x87654321,
        2,
        0xffffffff,
        0x80000000,
        0x87654321,
      ]);
    });

  it("lowers inline PTX selp b32 immediate statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int selp_true_imm_ptx(unsigned int b, unsigned int p) {
    unsigned int ret;
    asm volatile("selp.u32 %0, 0xaaaa0001, %1, %2;" : "=r"(ret) : "r"(b), "r"(p));
    return ret;
  }
  __device__ unsigned int selp_false_imm_ptx(unsigned int a, unsigned int p) {
    unsigned int ret;
    asm volatile("selp.u32 %0, %1, 0xbbbb0002, %2;" : "=r"(ret) : "r"(a), "r"(p));
    return ret;
  }
  __device__ int selp_both_imm_ptx(unsigned int p) {
    int ret;
    asm volatile("selp.s32 %0, -8, 7, %1;" : "=r"(ret) : "r"(p));
    return ret;
  }
  __global__ void selectImmediateKernel(uint *out, uint *a, uint *b, uint *pred) {
    int idx = threadIdx.x;
    out[idx] = selp_true_imm_ptx(b[idx], pred[idx]);
    out[idx + 4] = selp_false_imm_ptx(a[idx], pred[idx]);
    out[idx + 8] = (uint)selp_both_imm_ptx(pred[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(12),
            a: new Uint32Array([1, 0xffffffff, 0x80000000, 0x12345678]),
            b: new Uint32Array([2, 3, 0x7fffffff, 0x87654321]),
            pred: new Uint32Array([0, 1, 2, 0]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("2863267841u");
      expect(compiled.wgsl).toContain("3149594626u");
      expect(compiled.wgsl).toContain("4294967288u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        2,
        0xaaaa0001,
        0xaaaa0001,
        0x87654321,
        0xbbbb0002,
        0xffffffff,
        0x80000000,
        0xbbbb0002,
        7,
        0xfffffff8,
        0xfffffff8,
        7,
      ]);
    });

  it("lowers inline PTX cvt b32 statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int cvt_u_s(int value) {
    unsigned int ret;
    asm volatile("cvt.u32.s32 %0, %1;" : "=r"(ret) : "r"(value));
    return ret;
  }
  __device__ int cvt_s_u(unsigned int value) {
    int ret;
    asm volatile("cvt.s32.u32 %0, %1;" : "=r"(ret) : "r"(value));
    return ret;
  }
  __global__ void convertKernel(uint *out, uint *input) {
    int idx = threadIdx.x;
    out[idx] = cvt_u_s((int)input[idx]);
    out[idx + 4] = (uint)cvt_s_u(input[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(8),
            input: new Uint32Array([0, 1, 0xffffffff, 0x80000000]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("u32(");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0,
        1,
        0xffffffff,
        0x80000000,
        0,
        1,
        0xffffffff,
        0x80000000,
      ]);
    });

  it("lowers inline PTX cvt b32 immediate statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int cvt_u_s_imm() {
    unsigned int ret;
    asm volatile("cvt.u32.s32 %0, -1;" : "=r"(ret));
    return ret;
  }
  __device__ int cvt_s_u_imm() {
    int ret;
    asm volatile("cvt.s32.u32 %0, 0x80000000;" : "=r"(ret));
    return ret;
  }
  __global__ void convertImmediateKernel(uint *out) {
    int idx = threadIdx.x;
    out[idx] = cvt_u_s_imm();
    out[idx + 4] = (uint)cvt_s_u_imm();
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Uint32Array(8) } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("4294967295u");
      expect(compiled.wgsl).toContain("2147483648u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0x80000000,
        0x80000000,
        0x80000000,
        0x80000000,
      ]);
    });

  it("lowers inline PTX f32-to-int cvt statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ int cvt_rni_s(float value) {
    int ret;
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(ret) : "f"(value));
    return ret;
  }
  __device__ int cvt_rzi_s(float value) {
    int ret;
    asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(ret) : "f"(value));
    return ret;
  }
  __device__ int cvt_rmi_s(float value) {
    int ret;
    asm volatile("cvt.rmi.s32.f32 %0, %1;" : "=r"(ret) : "f"(value));
    return ret;
  }
  __device__ int cvt_rpi_s(float value) {
    int ret;
    asm volatile("cvt.rpi.s32.f32 %0, %1;" : "=r"(ret) : "f"(value));
    return ret;
  }
  __device__ unsigned int cvt_rpi_u_imm() {
    unsigned int ret;
    asm volatile("cvt.rpi.u32.f32 %0, 2.25;" : "=r"(ret));
    return ret;
  }
  __global__ void convertF32Kernel(uint *out, float *input) {
    int idx = threadIdx.x;
    float value = input[idx];
    out[idx] = (uint)cvt_rni_s(value);
    out[idx + 4] = (uint)cvt_rzi_s(value);
    out[idx + 8] = (uint)cvt_rmi_s(value);
    out[idx + 12] = (uint)cvt_rpi_s(value);
    out[idx + 16] = cvt_rpi_u_imm();
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(20),
            input: new Float32Array([1.5, 2.5, -1.5, -2.5]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("bg_round_even_f32");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        2,
        2,
        0xfffffffe,
        0xfffffffe,
        1,
        2,
        0xffffffff,
        0xfffffffe,
        1,
        2,
        0xfffffffe,
        0xfffffffd,
        2,
        3,
        0xffffffff,
        0xfffffffe,
        3,
        3,
        3,
        3,
      ]);
    });

  it("lowers inline PTX int-to-f32 cvt statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float cvt_rn_f32_u(unsigned int value) {
    float ret;
    asm volatile("cvt.rn.f32.u32 %0, %1;" : "=f"(ret) : "r"(value));
    return ret;
  }
  __device__ float cvt_rn_f32_s(int value) {
    float ret;
    asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(ret) : "r"(value));
    return ret;
  }
  __device__ float cvt_rn_f32_u_imm() {
    float ret;
    asm volatile("cvt.rn.f32.u32 %0, 16777217;" : "=f"(ret));
    return ret;
  }
  __device__ float cvt_rn_f32_s_imm() {
    float ret;
    asm volatile("cvt.rn.f32.s32 %0, -7;" : "=f"(ret));
    return ret;
  }
  __global__ void convertIntToF32Kernel(float *out, uint *uints, int *ints) {
    int idx = threadIdx.x;
    out[idx] = cvt_rn_f32_u(uints[idx]);
    out[idx + 4] = cvt_rn_f32_s(ints[idx]);
    out[idx + 8] = cvt_rn_f32_u_imm();
    out[idx + 12] = cvt_rn_f32_s_imm();
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Float32Array(16),
            uints: new Uint32Array([0, 1, 16777217, 0xffffffff]),
            ints: new Int32Array([-1, -16777217, 2147483647, -2147483648]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect([...result.buffers.out as Float32Array]).toEqual([
        0,
        1,
        16777216,
        4294967296,
        -1,
        -16777216,
        2147483648,
        -2147483648,
        16777216,
        16777216,
        16777216,
        16777216,
        -7,
        -7,
        -7,
        -7,
      ]);
    });

  it("lowers inline PTX f32 arithmetic statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float ptx_add(float value) {
    float ret;
    asm volatile("add.rn.f32 %0, %1, 2.25;" : "=f"(ret) : "f"(value));
    return ret;
  }
  __device__ float ptx_sub(float value) {
    float ret;
    asm volatile("sub.rn.f32 %0, %1, 1.5;" : "=f"(ret) : "f"(value));
    return ret;
  }
  __device__ float ptx_mul(float value) {
    float ret;
    asm volatile("mul.rn.f32 %0, %1, -2.0;" : "=f"(ret) : "f"(value));
    return ret;
  }
  __global__ void f32ArithmeticKernel(float *out, float *input) {
    int idx = threadIdx.x;
    float value = input[idx];
    out[idx] = ptx_add(value);
    out[idx + 4] = ptx_sub(value);
    out[idx + 8] = ptx_mul(value);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Float32Array(12),
            input: new Float32Array([1.5, -2, 4, -8]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect([...result.buffers.out as Float32Array]).toEqual([
        3.75,
        0.25,
        6.25,
        -5.75,
        0,
        -3.5,
        2.5,
        -9.5,
        -3,
        4,
        -8,
        16,
      ]);
    });

  it("lowers inline PTX f32 division statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ float ptx_div(float value) {
    float ret;
    asm volatile("div.rn.f32 %0, %1, 2.0;" : "=f"(ret) : "f"(value));
    return ret;
  }
  __global__ void f32DivisionKernel(float *out, float *input) {
    int idx = threadIdx.x;
    out[idx] = ptx_div(input[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Float32Array(4),
            input: new Float32Array([1, -3, 7, -9]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect([...result.buffers.out as Float32Array]).toEqual([
        0.5,
        -1.5,
        3.5,
        -4.5,
      ]);
    });

  it("lowers inline PTX mov b32 statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int mov_u_ptx(unsigned int value) {
    unsigned int ret;
    asm volatile("mov.u32 %0, %1;" : "=r"(ret) : "r"(value));
    return ret;
  }
  __device__ int mov_s_ptx(int value) {
    int ret;
    asm volatile("mov.s32 %0, %1;" : "=r"(ret) : "r"(value));
    return ret;
  }
  __global__ void moveKernel(uint *out, uint *input) {
    int idx = threadIdx.x;
    out[idx] = mov_u_ptx(input[idx]);
    out[idx + 4] = (uint)mov_s_ptx((int)input[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(8),
            input: new Uint32Array([0, 1, 0xffffffff, 0x80000000]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("u32(");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0,
        1,
        0xffffffff,
        0x80000000,
        0,
        1,
        0xffffffff,
        0x80000000,
      ]);
    });

  it("lowers inline PTX mov b32 immediate statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int mov_u_imm_ptx() {
    unsigned int ret;
    asm volatile("mov.u32 %0, 0xffffffff;" : "=r"(ret));
    return ret;
  }
  __device__ int mov_s_imm_ptx() {
    int ret;
    asm volatile("mov.s32 %0, -2147483648;" : "=r"(ret));
    return ret;
  }
  __global__ void moveImmediateKernel(uint *out) {
    int idx = threadIdx.x;
    out[idx] = mov_u_imm_ptx();
    out[idx + 4] = (uint)mov_s_imm_ptx();
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(8),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("4294967295u");
      expect(compiled.wgsl).toContain("2147483648u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0x80000000,
        0x80000000,
        0x80000000,
        0x80000000,
      ]);
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

  it("parses device-side kernel launches as host-orchestrated WebGPU lowering", () => {
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
        code: "cuda-dynamic-launch-host-orchestration",
        severity: "warning",
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
      expect(compiled.loweringPlan.requiresGpuPolyfill).toBe(true);
      expect(compiled.loweringPlan.unsupported).toEqual([]);
      expect(compiled.diagnostics).toContainEqual(expect.objectContaining({
        code: "cuda-dynamic-launch-host-orchestration",
        severity: "warning",
      }));
      expect([...result.buffers.x as Float32Array]).toEqual([2, 3]);
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

  it("expands object-like macro constants before parsing", () => {
      const compiled = compileCudaLiteKernel(`
  #define TILE_DIM 16 // trailing comments are ignored
  #define PADDED_TILE (TILE_DIM + 1)
  __global__ void padded(float *x) {
    __shared__ float tile[TILE_DIM][PADDED_TILE];
    int tid = threadIdx.x * TILE_DIM;
    if (tid < 1) { tile[0][0] = x[0]; x[0] = tile[0][0]; }
  }`, { workgroupSize: [1, 1, 1] });

      expect(compiled.wgsl).toContain("array<f32, 272>");
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

  it("emits semantic WGSL for lexical blocks with local declarations", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void scopedLocal(float *out) {
  int tid = threadIdx.x;
  {
    float value = (float)tid + 2.0f;
    out[tid] = value;
  }
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("var value: f32");
    expect([...result.buffers.out as Float32Array]).toEqual([2, 3]);
  });

  it("emits semantic WGSL for device helpers with lexical blocks", () => {
    const compiled = compileCudaLiteKernel(`
__device__ float scopedAdd(float value) {
  {
    float result = value + 2.0f;
    return result;
  }
}
__global__ void helperScopedLocal(float *out) {
  out[threadIdx.x] = scopedAdd((float)threadIdx.x);
}`, { workgroupSize: [2, 1, 1] });
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [2, 1, 1] as const };
    const input = { buffers: { out: new Float32Array(2) } };
    const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
    const result = runCompiledKernelReference(compiled, input, launch);

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toContain("fn scopedAdd");
    expect([...semanticResult.buffers.out as Float32Array]).toEqual([2, 3]);
    expect([...result.buffers.out as Float32Array]).toEqual([2, 3]);
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
      expect(compiled.wgsl).toContain("bg_array");
      expect(compiled.wgsl).toContain("bg_shared");
      expect(compiled.wgsl).toContain("bg_var");
      expect(compiled.wgsl).toContain("bg_precision");
      expect(compiled.wgsl).not.toContain("var var:");
      expect(compiled.wgsl).not.toContain(" precision:");
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
      expect(compiled.wgsl).toContain("var a: u32 = (bitcast<u32>(bg_uniforms.q) * 2u);");
      expect(compiled.wgsl).toContain("var b: u32 = u32((a + 1u));");
      expect(compiled.wgsl).toContain("var c: i32 = (bitcast<i32>(workgroup_id.x) + 3);");
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
