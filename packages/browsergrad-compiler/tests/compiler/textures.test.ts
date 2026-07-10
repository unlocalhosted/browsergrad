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

describe("CUDA-lite compiler: Textures and surfaces", () => {
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
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {},
          textures: { texRef: { width: 2, height: 2, data: new Float32Array([1, 2, 3, 4]) } },
          surfaces: { outputSurf: { width: 2, height: 2, data: new Float32Array(4) } },
          scalars: { width: 2, height: 2 },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 2, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var<storage, read_write> outputSurf: array<f32>;");
      expect(compiled.wgsl).toContain("outputSurf[bg_index] = (value * 2.0);");
      expect(compiled.wgsl).not.toContain("bg_surf2dwrite_outputSurf");
      expect([...result.buffers.outputSurf as Float32Array]).toEqual([2, 4, 6, 8]);
      expect([...semanticResult.buffers.outputSurf as Float32Array]).toEqual([2, 4, 6, 8]);
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
      expect(helperCompiled.wgsl).toContain("read_layer(0u, 1, 1, local_id, workgroup_id, num_workgroups)");
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
      expect(compiled.wgsl).toContain("write_surface_vec(0u, value, local_id, workgroup_id, num_workgroups)");
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
      expect(compiled.wgsl).toContain("write_value(0u, 17.0, local_id, workgroup_id, num_workgroups)");
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

  it("lowers typed uint4 surface vector reads and writes through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void surfaceUintVectorReadWrite(cudaSurfaceObject_t surf, uint4 *out) {
    uint4 pointerValue;
    surf2Dread(&pointerValue, surf, 0, 0);
    uint4 returnValue = surf2Dread<uint4>(surf, 0, 0);
    uint4 written = make_uint4(11u, 12u, 13u, 14u);
    surf2Dwrite(written, surf, 0, 0);
    uint4 afterWrite = surf2Dread<uint4>(surf, 0, 0);
    out[0] = pointerValue;
    out[1] = returnValue;
    out[2] = afterWrite;
  }`, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: { out: new Uint32Array(12) },
        surfaces: { surf: { width: 4, height: 1, data: new Float32Array([1, 2, 3, 4]) } },
      };
      const launch = { gridDim: [1, 1, 1], blockDim: [1, 1, 1] } as const;
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("select(vec4<u32>(), vec4<u32>(u32(bg_sem_surf2dread_surf");
      expect(compiled.wgsl).toContain("surf[bg_index] = f32((written).x);");
      expect(compiled.wgsl).not.toContain("bg_surf2dread_surf");
      expect([...result.buffers.out as Uint32Array]).toEqual([1, 2, 3, 4, 1, 2, 3, 4, 11, 12, 13, 14]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([1, 2, 3, 4, 1, 2, 3, 4, 11, 12, 13, 14]);
      expect([...result.buffers.surf as Float32Array]).toEqual([11, 12, 13, 14]);
      expect([...semanticResult.buffers.surf as Float32Array]).toEqual([11, 12, 13, 14]);
    });

  it("lowers half and half2 surface reads and writes with shader-f16", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void surfaceHalfReadWrite(cudaSurfaceObject_t surf, float *out, uint *bits) {
    half scalar = surf2Dread<half>(surf, 0, 0);
    half2 pair;
    surf2Dread(&pair, surf, 0, 0);
    half2 written = __floats2half2_rn(5.5f, 6.5f);
    surf2Dwrite(written, surf, 0, 0);
    half2 after = surf2Dread<half2>(surf, 0, 0);
    out[0] = __half2float(scalar);
    out[1] = pair.x;
    out[2] = pair.y;
    out[3] = after.x;
    out[4] = after.y;
    bits[0] = __half_as_ushort(scalar);
    bits[1] = __half2_as_uint(pair);
    bits[2] = __half2_as_uint(after);
  }`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(5),
          bits: new Uint32Array(3),
        },
        surfaces: { surf: { width: 2, height: 1, data: new Float32Array([1.1, 2.2]) } },
      };
      const launch = { gridDim: [1, 1, 1], blockDim: [1, 1, 1] } as const;
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("enable f16;");
      expect(compiled.wgsl).toContain("f16(bg_sem_surf2dread_surf(0, 0, 0))");
      expect(compiled.wgsl).toContain("vec2<f16>(f16(bg_sem_surf2dread_surf((0 + 0), 0, 0)), f16(bg_sem_surf2dread_surf((0 + 4), 0, 0)))");
      expect(compiled.wgsl).not.toContain("bg_surf2dread_surf");
      expect(backendIr(compiled).requiredFeatures).toContain("shader-f16");
      expect([...result.buffers.out as Float32Array]).toEqual([1.099609375, 1.099609375, 2.19921875, 5.5, 6.5]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([1.099609375, 1.099609375, 2.19921875, 5.5, 6.5]);
      expect([...result.buffers.bits as Uint32Array]).toEqual([0x3c66, 0x40663c66, 0x46804580]);
      expect([...semanticResult.buffers.bits as Uint32Array]).toEqual([0x3c66, 0x40663c66, 0x46804580]);
      expect([...result.buffers.surf as Float32Array]).toEqual([5.5, 6.5]);
      expect([...semanticResult.buffers.surf as Float32Array]).toEqual([5.5, 6.5]);
    });

  it("lowers half and half2 surface reads and writes through f32 compatibility mode", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void surfaceHalfReadWrite(cudaSurfaceObject_t surf, float *out, uint *bits) {
    half scalar = surf2Dread<half>(surf, 0, 0);
    half2 pair;
    surf2Dread(&pair, surf, 0, 0);
    half2 written = __floats2half2_rn(5.5f, 6.5f);
    surf2Dwrite(written, surf, 0, 0);
    half2 after = surf2Dread<half2>(surf, 0, 0);
    out[0] = __half2float(scalar);
    out[1] = pair.x;
    out[2] = pair.y;
    out[3] = after.x;
    out[4] = after.y;
    bits[0] = __half_as_ushort(scalar);
    bits[1] = __half2_as_uint(pair);
    bits[2] = __half2_as_uint(after);
  }`, { f16Mode: "f32", workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(5),
          bits: new Uint32Array(3),
        },
        surfaces: { surf: { width: 2, height: 1, data: new Float32Array([1.1, 2.2]) } },
      };
      const launch = { gridDim: [1, 1, 1], blockDim: [1, 1, 1] } as const;
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("enable f16;");
      expect(compiled.wgsl).not.toContain("f16(");
      expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
      expect([...result.buffers.out as Float32Array]).toEqual([1.099609375, 1.099609375, 2.19921875, 5.5, 6.5]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([1.099609375, 1.099609375, 2.19921875, 5.5, 6.5]);
      expect([...result.buffers.bits as Uint32Array]).toEqual([0x3c66, 0x40663c66, 0x46804580]);
      expect([...semanticResult.buffers.bits as Uint32Array]).toEqual([0x3c66, 0x40663c66, 0x46804580]);
      expect([...result.buffers.surf as Float32Array]).toEqual([5.5, 6.5]);
      expect([...semanticResult.buffers.surf as Float32Array]).toEqual([5.5, 6.5]);
    });

  it("lowers bf16 and bf162 surface reads and writes through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void surfaceBf16ReadWrite(cudaSurfaceObject_t surf, float *out, uint *bits) {
    __nv_bfloat16 scalar = surf2Dread<__nv_bfloat16>(surf, 0, 0);
    __nv_bfloat162 pair;
    surf2Dread(&pair, surf, 0, 0);
    __nv_bfloat162 written = __halves2bfloat162(5.5f, 6.5f);
    surf2Dwrite(written, surf, 0, 0);
    __nv_bfloat162 after = surf2Dread<__nv_bfloat162>(surf, 0, 0);
    out[0] = __bfloat162float(scalar);
    out[1] = pair.x;
    out[2] = pair.y;
    out[3] = after.x;
    out[4] = after.y;
    bits[0] = __bfloat16_as_ushort(scalar);
    bits[1] = __bfloat162_as_uint(pair);
    bits[2] = __bfloat162_as_uint(after);
  }`, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(5),
          bits: new Uint32Array(3),
        },
        surfaces: { surf: { width: 2, height: 1, data: new Float32Array([1.1, 2.2]) } },
      };
      const launch = { gridDim: [1, 1, 1], blockDim: [1, 1, 1] } as const;
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bg_f32_to_bf16_bits_mode(f32(bg_sem_surf2dread_surf(0, 0, 0)), 0u)");
      expect(compiled.wgsl).toContain("vec2<f32>(bitcast<f32>(bg_f32_to_bf16_bits_mode(f32(bg_sem_surf2dread_surf((0 + 0), 0, 0)), 0u) << 16u), bitcast<f32>(bg_f32_to_bf16_bits_mode(f32(bg_sem_surf2dread_surf((0 + 4), 0, 0)), 0u) << 16u))");
      expect(compiled.wgsl).not.toContain("enable f16;");
      expect(compiled.wgsl).not.toContain("bg_surf2dread_surf");
      expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
      expect([...result.buffers.out as Float32Array]).toEqual([1.1015625, 1.1015625, 2.203125, 5.5, 6.5]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([1.1015625, 1.1015625, 2.203125, 5.5, 6.5]);
      expect([...result.buffers.bits as Uint32Array]).toEqual([0x3f8d, 0x400d3f8d, 0x40d040b0]);
      expect([...semanticResult.buffers.bits as Uint32Array]).toEqual([0x3f8d, 0x400d3f8d, 0x40d040b0]);
      expect([...result.buffers.surf as Float32Array]).toEqual([5.5, 6.5]);
      expect([...semanticResult.buffers.surf as Float32Array]).toEqual([5.5, 6.5]);
    });

  it("lowers uchar surface reads through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void surfaceUcharRead(cudaSurfaceObject_t surf, uint *out) {
    out[0] = surf2Dread<unsigned char>(surf, 0, 0);
    out[1] = surf2Dread<unsigned char>(surf, 4, 0);
  }`, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: { out: new Uint32Array(2) },
        surfaces: { surf: { width: 2, height: 1, data: new Float32Array([5, 7]) } },
      };
      const launch = { gridDim: [1, 1, 1], blockDim: [1, 1, 1] } as const;
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("u32(bg_sem_surf2dread_surf(0, 0, 0))");
      expect(compiled.wgsl).toContain("u32(bg_sem_surf2dread_surf(4, 0, 0))");
      expect(compiled.wgsl).not.toContain("bg_surf2dread_surf");
      expect([...result.buffers.out as Uint32Array]).toEqual([5, 7]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([5, 7]);
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
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: { out: new Uint32Array(3) },
          textures: { texRef: { width: 3, height: 1, data: new Float32Array([2, 127, 255]) } },
        },
        { gridDim: [1, 1, 1], blockDim: [3, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("u32(textureLoad(texRef");
      expect(compiled.wgsl).not.toContain("bg_tex2d_uchar_texRef");
      expect([...result.buffers.out as Uint32Array]).toEqual([2, 127, 255]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([2, 127, 255]);
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
      const semanticResult = runCompiledKernelSemanticReference(
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

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-texture");
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bg_f32_to_bf16_bits_mode(f32(textureLoad(texRef");
      expect(compiled.wgsl).not.toContain("bg_tex2d_bf16_texRef");
      expect(compiled.wgsl).not.toContain("bg_tex2d_bf162_texRef");
      expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
      expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(1.1015625);
      expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(1.1015625);
      expect([...result.buffers.out as Float32Array][2]).toBeCloseTo(2.203125);
      expect([...result.buffers.bits as Uint32Array]).toEqual([0x3f8d, 0x400d3f8d]);
      expect([...semanticResult.buffers.out as Float32Array][0]).toBeCloseTo(1.1015625);
      expect([...semanticResult.buffers.out as Float32Array][1]).toBeCloseTo(1.1015625);
      expect([...semanticResult.buffers.out as Float32Array][2]).toBeCloseTo(2.203125);
      expect([...semanticResult.buffers.bits as Uint32Array]).toEqual([0x3f8d, 0x400d3f8d]);
    });

  it("lowers templated half and half2 tex2D reads with shader-f16", () => {
      const compiled = compileCudaLiteKernel(`
  texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
  __global__ void sample(float *out, uint *bits) {
    half scalar = tex2D<half>(texRef, 0.5f, 0.5f);
    half2 pair = tex2D<half2>(texRef, 0.5f, 0.5f);
    out[0] = __half2float(scalar);
    out[1] = pair.x;
    out[2] = pair.y;
    bits[0] = __half_as_ushort(scalar);
    bits[1] = __half2_as_uint(pair);
  }`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(3),
          bits: new Uint32Array(2),
        },
        textures: { texRef: { width: 1, height: 1, data: new Float32Array([1.1, 2.2, 3.3, 4.4]), channels: 4 as const } },
      };
      const launch = { gridDim: [1, 1, 1], blockDim: [1, 1, 1] } as const;
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-texture");
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("enable f16;");
      expect(compiled.wgsl).toContain("f16(textureLoad(texRef");
      expect(compiled.wgsl).not.toContain("bg_tex2d_half_texRef");
      expect(compiled.wgsl).not.toContain("bg_tex2d_half2_texRef");
      expect(backendIr(compiled).requiredFeatures).toContain("shader-f16");
      expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(1.099609375);
      expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(1.099609375);
      expect([...result.buffers.out as Float32Array][2]).toBeCloseTo(2.19921875);
      expect([...result.buffers.bits as Uint32Array]).toEqual([0x3c66, 0x40663c66]);
      expect([...semanticResult.buffers.out as Float32Array][0]).toBeCloseTo(1.099609375);
      expect([...semanticResult.buffers.out as Float32Array][1]).toBeCloseTo(1.099609375);
      expect([...semanticResult.buffers.out as Float32Array][2]).toBeCloseTo(2.19921875);
      expect([...semanticResult.buffers.bits as Uint32Array]).toEqual([0x3c66, 0x40663c66]);
    });

  it("lowers templated half and half2 tex2D reads through f32 compatibility mode", () => {
      const compiled = compileCudaLiteKernel(`
  texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
  __global__ void sample(float *out, uint *bits) {
    half scalar = tex2D<half>(texRef, 0.5f, 0.5f);
    half2 pair = tex2D<half2>(texRef, 0.5f, 0.5f);
    out[0] = __half2float(scalar);
    out[1] = pair.x;
    out[2] = pair.y;
    bits[0] = __half_as_ushort(scalar);
    bits[1] = __half2_as_uint(pair);
  }`, { f16Mode: "f32", workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(3),
          bits: new Uint32Array(2),
        },
        textures: { texRef: { width: 1, height: 1, data: new Float32Array([1.1, 2.2, 3.3, 4.4]), channels: 4 as const } },
      };
      const launch = { gridDim: [1, 1, 1], blockDim: [1, 1, 1] } as const;
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-texture");
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).not.toContain("enable f16;");
      expect(compiled.wgsl).not.toContain("f16(");
      expect(compiled.wgsl).not.toContain("bg_tex2d_half_texRef");
      expect(compiled.wgsl).not.toContain("bg_tex2d_half2_texRef");
      expect(backendIr(compiled).requiredFeatures).not.toContain("shader-f16");
      expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(1.099609375);
      expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(1.099609375);
      expect([...result.buffers.out as Float32Array][2]).toBeCloseTo(2.19921875);
      expect([...result.buffers.bits as Uint32Array]).toEqual([0x3c66, 0x40663c66]);
      expect([...semanticResult.buffers.out as Float32Array][0]).toBeCloseTo(1.099609375);
      expect([...semanticResult.buffers.out as Float32Array][1]).toBeCloseTo(1.099609375);
      expect([...semanticResult.buffers.out as Float32Array][2]).toBeCloseTo(2.19921875);
      expect([...semanticResult.buffers.bits as Uint32Array]).toEqual([0x3c66, 0x40663c66]);
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
      expect(compiled.wgsl).toContain("fn readPixel(texSrc: texture_2d<f32>, x: f32, local_id: vec3<u32>, workgroup_id: vec3<u32>, num_workgroups: vec3<u32>) -> vec4<f32>");
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

  it("lowers cubemap texture reads through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sample(float *out, cudaTextureObject_t tex) {
    out[0] = texCubemap<float>(tex, 1.0f, 0.0f, 0.0f);
  }`, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: { out: new Float32Array(1) },
        textures: { tex: { width: 2, height: 12, channels: 1 as const, data: new Float32Array([7, 0, ...new Array(22).fill(0)]) } },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("bg_cube_face");
      expect([...result.buffers.out as Float32Array]).toEqual([7]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([7]);
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

      expect(compiled.wgsl).toContain("textureLoad(texRef");
      expect(compiled.wgsl).toContain("vec4<f32>(f32(0.5)");
      expect([...result.buffers.out as Float32Array]).toEqual([1, 1.5, 2, 2.5]);
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

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("vec4<u32>(u32((textureLoad(texRef");
      expect(compiled.wgsl).not.toContain("bg_tex2d_uint4_texRef");
      expect([...result.buffers.out as Uint32Array]).toEqual([1, 2, 3, 255]);
    });

  it("lowers typed scalar texture helper reads through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ uint read_uint_tex(cudaTextureObject_t texArg) {
    return tex2D<uint>(texArg, 0.5f, 0.5f);
  }

  __global__ void sample(cudaTextureObject_t tex, uint *out) {
    out[0] = read_uint_tex(tex);
  }`, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: { out: new Uint32Array(1) },
        textures: { tex: { width: 1, height: 1, data: new Float32Array([42]) } },
      };
      const launch = { gridDim: [1, 1, 1], blockDim: [1, 1, 1] } as const;
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("return u32(textureLoad(texArg");
      expect(compiled.wgsl).not.toContain("bg_tex2d_uint_tex");
      expect([...result.buffers.out as Uint32Array]).toEqual([42]);
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([42]);
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

  it("lowers layered and 3D texture atlas reads through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void sample(float *out, cudaTextureObject_t tex) {
    out[0] = tex2DLayered<float>(tex, 0.0f, 1.0f, 1.0f);
    out[1] = tex3D<float>(tex, 2.0f, 1.0f, 1.0f);
  }`, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: { out: new Float32Array(2) },
        textures: {
          tex: { width: 4, height: 4, data: new Float32Array(Array.from({ length: 16 }, (_, index) => index + 1)) },
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.kernelIr)).toBe(true);
      expect(compiled.kernelIr.operations).toEqual(expect.arrayContaining([
        expect.objectContaining({ kind: "store", value: expect.objectContaining({ kind: "texture-read", callee: "tex2DLayered" }) }),
        expect.objectContaining({ kind: "store", value: expect.objectContaining({ kind: "texture-read", callee: "tex3D" }) }),
      ]));
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect([...result.buffers.out as Float32Array]).toEqual([9, 11]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([9, 11]);
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
