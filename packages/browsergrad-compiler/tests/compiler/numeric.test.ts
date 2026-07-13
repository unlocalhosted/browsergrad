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
import {
  constantBufferInputs,
  cudaWebGpuDefaultReadbackNames,
  deviceGlobalBufferInputs,
} from "../../src/webgpu_inputs";
import { deviceLaunchTreeIsExternallySilent } from "../../src/runtime_elision";
import { packCudaWebGpuUniformParams } from "../../src/webgpu_orchestration";

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

describe("CUDA-lite compiler: Numeric types and intrinsics", () => {
  it("shadows mutable by-value kernel params as thread-local values", () => {
    const compiled = compileCudaLiteKernel(`
__global__ void mutableScalarParams(float alpha, float beta, float *out) {
  beta /= alpha;
  beta += (float)threadIdx.x;
  out[threadIdx.x] = beta;
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { scalars: { alpha: 2, beta: 4 }, buffers: { out: new Float32Array(2) } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect(compiled.wgsl).toMatch(/var bg_param_local_beta_\d+: f32 = bg_uniforms\.beta;/u);
    expect([...result.buffers.out as Float32Array]).toEqual([2, 3]);
  });
  it("lowers ignored dynamic frexp results with local exponent side effects", () => {
    const compiled = compileCudaLiteKernel(`
__device__ int ceil_pow2(int n) {
  if (0 == (n & (n - 1))) return n;
  int exp;
  frexp(float(n), &exp);
  return 1 << exp;
}
__global__ void shared_helper_result(int *out, int n) {
  __shared__ uint value;
  if (threadIdx.x == 0) value = ceil_pow2(n);
  __syncthreads();
  out[threadIdx.x] = int(value);
}`, { workgroupSize: [2, 1, 1] });
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { out: new Int32Array(2) }, scalars: { n: 5 } },
      { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
    );

    expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
    expect([...result.buffers.out as Int32Array]).toEqual([8, 8]);
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

  it("lowers CUDA vector swizzle reads as native vector values", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void vector_swizzles(float* out, uint* ui) {
    float4 value = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float2 lo = value.xy;
    float3 mix = value.zyx;
    float4 color = value.rgba;
    uint4 bits = make_uint4(5u, 6u, 7u, 8u);
    uint3 packed = bits.s210;
    out[0] = lo.x + lo.y;
    out[1] = mix.x + mix.y + mix.z;
    out[2] = color.x + color.w;
    ui[0] = packed.x;
    ui[1] = packed.y;
    ui[2] = packed.z;
  }`, { workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(3), ui: new Uint32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(3), ui: new Uint32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-vector-member");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("var lo: vec2<f32> = value.xy;");
      expect(compiled.wgsl).toContain("var mix: vec3<f32> = value.zyx;");
      expect(compiled.wgsl).toContain("var color: vec4<f32> = value.xyzw;");
      expect(compiled.wgsl).toContain("var packed: vec3<u32> = bits.zyx;");
      expect([...result.buffers.out as Float32Array]).toEqual([3, 6, 5]);
      expect([...result.buffers.ui as Uint32Array]).toEqual([7, 6, 5]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([3, 6, 5]);
      expect([...semanticResult.buffers.ui as Uint32Array]).toEqual([7, 6, 5]);
    });

  it("lowers CUDA vector swizzle writes to local vectors", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void vector_swizzle_writes(float* out, uint* ui) {
    float4 value = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float2 next = make_float2(9.0f, 8.0f);
    value.xy = next;
    value.yx = value.zw;
    value.xy += make_float2(1.0f, 2.0f);
    uint4 bits = make_uint4(5u, 6u, 7u, 8u);
    bits.s210 = make_uint3(11u, 12u, 13u);
    bits.s210 += make_uint3(1u, 2u, 3u);
    out[0] = value.x;
    out[1] = value.y;
    out[2] = value.z;
    out[3] = value.w;
    ui[0] = bits.x;
    ui[1] = bits.y;
    ui[2] = bits.z;
    ui[3] = bits.w;
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

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("invalid-assignment-target");
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-vector-assignment");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("value.xy = next;");
      expect(compiled.wgsl).toContain("value.yx = value.zw;");
      expect(compiled.wgsl).toContain("value.xy = value.xy + vec2<f32>(f32(1.0), f32(2.0));");
      expect(compiled.wgsl).toContain("bits.zyx = vec3<u32>(u32(11u), u32(12u), u32(13u));");
      expect(compiled.wgsl).toContain("bits.zyx = bits.zyx + vec3<u32>(u32(1u), u32(2u), u32(3u));");
      expect([...result.buffers.out as Float32Array]).toEqual([5, 5, 3, 4]);
      expect([...result.buffers.ui as Uint32Array]).toEqual([16, 14, 12, 8]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([5, 5, 3, 4]);
      expect([...semanticResult.buffers.ui as Uint32Array]).toEqual([16, 14, 12, 8]);
    });

  it("rejects repeated CUDA vector swizzle write targets", () => {
      expect(() => compileCudaLiteKernel(`
  __global__ void repeated_vector_swizzle(float* out) {
    float4 value = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    value.xx = make_float2(5.0f, 6.0f);
    out[0] = value.x;
  }`, { workgroupSize: [1, 1, 1] })).toThrow(/vector swizzle assignment target cannot repeat lanes/u);
    });

  it("lowers CUDA vector swizzle writes through local vector arrays", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void local_vector_array_swizzles(float* out, uint* ui) {
    float4 values[2];
    values[0] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    values[0].xy = make_float2(9.0f, 8.0f);
    values[0].zw += make_float2(1.0f, 2.0f);
    uint4 bits[1];
    bits[0] = make_uint4(5u, 6u, 7u, 8u);
    bits[0].s210 = make_uint3(11u, 12u, 13u);
    bits[0].xy += make_uint2(1u, 2u);
    out[0] = values[0].x;
    out[1] = values[0].y;
    out[2] = values[0].z;
    out[3] = values[0].w;
    ui[0] = bits[0].x;
    ui[1] = bits[0].y;
    ui[2] = bits[0].z;
    ui[3] = bits[0].w;
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
      expect(compiled.wgsl).toContain("values[0u].xy = vec2<f32>(f32(9.0), f32(8.0));");
      expect([...result.buffers.out as Float32Array]).toEqual([9, 8, 4, 6]);
      expect([...result.buffers.ui as Uint32Array]).toEqual([14, 14, 11, 8]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([9, 8, 4, 6]);
      expect([...semanticResult.buffers.ui as Uint32Array]).toEqual([14, 14, 11, 8]);
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

      const result = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(1) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      expect(compiled.wgsl).toContain("ptr<workgroup, array<vec4<f32>, 8>>");
      expect([...result.buffers.out as Float32Array]).toEqual([6.5]);
    });

  it("lowers vector local-array scalar-fill initializers through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void vectorLocalArrayInit(float4* out) {
    int tid = threadIdx.x;
    float4 vals[2] = make_float4(1.0f + (float)tid, 2.0f, 3.0f, 4.0f);
    out[tid] = make_float4(vals[tid][0], vals[tid].y, vals[tid].z, vals[tid].w);
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

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-local-array-init");
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var vals: array<vec4<f32>, 2>;");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 2, 3, 4, 2, 2, 3, 4]);
      expect([...result.buffers.out as Float32Array]).toEqual([1, 2, 3, 4, 2, 2, 3, 4]);
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
      expect(compiled.wgsl).toContain("var bytes: u32 = bitcast<u32>(4)");
      expect([...result.buffers.signedOut as Int32Array]).toEqual([4, 6]);
      expect([...result.buffers.unsignedOut as Uint32Array]).toEqual([15, 17]);
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

      expect(compiled.wgsl).toContain("pick_max(v, vec2<f32>(f32(x[0u]), f32(4.0))");
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

      expect(semanticIr(compiled).requiredFeatures).not.toContain("shader-f16");
      expect(compiled.wgsl).toContain("fn addOne(value: f32");
      expect(compiled.wgsl).not.toContain("unused_half2");
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

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bg_semantic_round_even_f32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_tgamma_f32(");
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect([...semanticResult.buffers.iout as Int32Array]).toEqual([-2, 4, -3, 4, -2, -2, 3]);
      expect([...semanticResult.buffers.uout as Uint32Array]).toEqual([4, 3, 4, 3]);
      expect([...semanticResult.buffers.fout as Float32Array]).toEqual([-2, 4]);
      expect([...result.buffers.iout as Int32Array]).toEqual([-2, 4, -3, 4, -2, -2, 3]);
      expect([...result.buffers.uout as Uint32Array]).toEqual([4, 3, 4, 3]);
      expect([...result.buffers.fout as Float32Array]).toEqual([-2, 4]);
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
      expect(compiled.wgsl).toContain("var bg__bg_remquo_dividend_");
      expect(compiled.wgsl).toContain("localQuo = select(");
      expect(compiled.wgsl).toContain("quo[1u] = select(");
      expect(compiled.wgsl).toContain("quo[2u] = select(");
      expect([...result.buffers.out as Float32Array]).toEqual([-1, 1]);
      expect([...result.buffers.quo as Int32Array]).toEqual([4, 2, -4]);
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
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
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(3), expOut: new Int32Array(3) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var bg__bg_frexp_value_");
      expect(compiled.wgsl).toContain("expOut[1u] = select(");
      expect(compiled.wgsl).not.toContain("fn bg_frexp(");
      expect([...result.buffers.out as Float32Array][0]).toBeCloseTo(0.5625, 6);
      expect([...result.buffers.out as Float32Array][1]).toBeCloseTo(0.625, 6);
      expect([...result.buffers.out as Float32Array][2]).toBeCloseTo(0.75, 6);
      expect([...result.buffers.expOut as Int32Array]).toEqual([4, 4, 4]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([...result.buffers.out as Float32Array]);
      expect([...semanticResult.buffers.expOut as Int32Array]).toEqual([4, 4, 4]);
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
      expect(compiled.wgsl).toContain("var bg__bg_modf_value_");
      expect(compiled.wgsl).toContain("localInt = select(trunc(");
      expect(compiled.wgsl).toContain("out[3u] = select(trunc(");
      expect(compiled.wgsl).toContain("out[6u] = select(trunc(");
      expect([...result.buffers.out as Float32Array]).toEqual([0.75, 3, -0.25, -2, 0.5, 5, 8]);
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
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("bg__bg_sincos_angle");
      expect(compiled.wgsl).toContain("sin(bg__bg_sincos_angle");
      expect(compiled.wgsl).toContain("cos(bg__bg_sincos_angle");
      expect(compiled.wgsl).toContain("(3.141592653589793 * 0.5)");
      const out = [...result.buffers.out as Float32Array];
      expect(out[0]).toBeCloseTo(Math.sin(0.25), 6);
      expect(out[1]).toBeCloseTo(Math.cos(0.25), 6);
      expect(out[2]).toBeCloseTo(Math.sin(0.5), 6);
      expect(out[3]).toBeCloseTo(Math.cos(0.5), 6);
      expect(out[4]).toBeCloseTo(1, 6);
      expect(out[5]).toBeCloseTo(0, 6);
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
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

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bg_semantic_umulhi_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_byte_perm_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_funnelshift_l_u32(");
      expect(compiled.wgsl).not.toContain("assert omitted");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([29, 15, 20, 3, 3, 0, 4, 7, 42, 44, 40, 0xe6a2c480, 30, 12, 0xfffffffe, 304062474, 0x66772233, 0xfffffffe, 2147483648, 2147483649, 0xfffffffd, 0x22334455, 0x55667788, 0x88112233, 0x55667788, 59, 5, 8, 0x0f000000, 0xfffffffe, 304062474]);
      expect([...result.buffers.out as Uint32Array]).toEqual([29, 15, 20, 3, 3, 0, 4, 7, 42, 44, 40, 0xe6a2c480, 30, 12, 0xfffffffe, 304062474, 0x66772233, 0xfffffffe, 2147483648, 2147483649, 0xfffffffd, 0x22334455, 0x55667788, 0x88112233, 0x55667788, 59, 5, 8, 0x0f000000, 0xfffffffe, 304062474]);
    });

  it("preserves signed result types for shifts with unsigned counts", () => {
    const compiled = compileCudaLiteKernel(`
  __global__ void signedShifts(int *out, int value, uint amount) {
    out[0] = value << amount;
    out[1] = value >> amount;
  }`, { workgroupSize: [1, 1, 1] });
    const input = { buffers: { out: new Int32Array(2) }, scalars: { value: -8, amount: 1 } };
    const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };

    expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
    expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
    expect([...runCompiledKernelSemanticReference(compiled, input, launch).buffers.out as Int32Array]).toEqual([-16, -4]);
    expect(compiled.wgsl).toContain(" << ");
    expect(compiled.wgsl).toContain(" >> ");
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
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

  it("lowers CUDA packed halfword SIMD intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void packed_halfword_simd(uint *out) {
    uint a = 0x10ff807fu;
    uint b = 0x01028081u;
    out[0] = __vadd2(a, b);
    out[1] = __vsub2(a, b);
    out[2] = __vabsdiffu2(a, b);
    out[3] = __vavgu2(a, b);
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("fn bg_semantic_vadd2_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vsub2_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vabsdiffu2_u32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_vavgu2_u32(");
      expect([...semanticResult.buffers.out as Uint32Array]).toEqual([302055680, 268304382, 268238850, 151093376]);
      expect([...result.buffers.out as Uint32Array]).toEqual([...semanticResult.buffers.out as Uint32Array]);
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
      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bg_semantic_erfinv_f32(");
      expect(compiled.wgsl).toContain("fn bg_semantic_normcdfinv_f32(");
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
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(13), kinds: new Uint32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );
      const out = [...result.buffers.out as Float32Array];

      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect([...semanticResult.buffers.kinds as Uint32Array]).toEqual([4, 64]);
      expect(Number.isNaN((semanticResult.buffers.out as Float32Array)[3])).toBe(true);
      expect(compiled.wgsl).toContain("fn bg_f32_inf() -> f32");
      expect(compiled.wgsl).toContain("fn bg_f32_nan() -> f32");
      expect(compiled.wgsl).toContain("bitcast<f32>(0x7f7fffffu)");
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
      expect(compiled.wgsl).toContain("dst[(idx + u32(k))] = f32((f32(dst[(idx + u32(k))]) + acc[u32(k)]));");
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

  it("lowers reinterpreted 128-bit half pack copies through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  #define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
  #define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
  __global__ void half_pack_reinterpret(const half* input, half* output) {
    half pack[8];
    LDST128BITS(pack[0]) = LDST128BITS(input[0]);
    half2 pair = HALF2(pack[2]);
    LDST128BITS(output[0]) = LDST128BITS(pack[0]);
    output[0] = pair.x;
    output[1] = pair.y;
  }`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            input: createWgslFloat16Array([1, 2, 3, 4, 5, 6, 7, 8]),
            output: createWgslFloat16Array(8),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.kernelIr.operations.filter((operation) => operation.kind === "copy")).toHaveLength(2);
      expect(Array.from(result.buffers.output as Iterable<number>)).toEqual([3, 4, 3, 4, 5, 6, 7, 8]);
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("vec3<f32>(f32(value.x), f32(value.y), f32(value.z))");
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("vec4<f32>(f32(xyz.x), f32(xyz.y), f32(xyz.z), f32(7.0))");
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
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

      expect(compiled.wgsl).toContain("vec4<u32>(u32(1u), u32(2u), u32(3u), u32(4u))");
      expect(compiled.wgsl).toContain("rgbToInt(f32(color.z), f32(color.y), f32(color.x)");
      expect([...result.buffers.out as Int32Array]).toEqual([6]);
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
      expect(compiled.wgsl).toContain("var value: vec4<f32> = vec4<f32>(");
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

  it("lowers CUDA half2 arithmetic intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void half2Ops(const half2 *x, const half2 *y, half2 *out, float *scalar) {
    half2 sum = __hadd2_rn(x[0], y[0]);
    half2 back = __hsub2_rn(sum, y[0]);
    half2 prod = __hmul2_rn(sum, make_half2(__float2half(2.0f), __float2half(0.5f)));
    out[0] = __hmax2(prod, make_half2(__float2half(5.0f), __float2half(5.0f)));
    half2 fused = __hfma2_rn(back, y[0], make_half2(__float2half(1.0f), __float2half(2.0f)));
    out[1] = x[0] * y[0] + fused;
    scalar[0] = __low2float(fused) + __high2float(fused) + __low2float(back);
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("max(");
      expect(compiled.wgsl).toContain("fma(");
      expect(Array.from(result.buffers.out as ArrayLike<number>)).toEqual([6, 6, 5, 66]);
      expect([...result.buffers.scalar as Float32Array]).toEqual([38]);
      expect(Array.from(semanticResult.buffers.out as ArrayLike<number>)).toEqual([6, 6, 5, 66]);
      expect([...semanticResult.buffers.scalar as Float32Array]).toEqual([38]);
    });

  it("lowers local half2 vector assignments", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void half2Assignment(half2 *out) {
    half2 left = __floats2half2_rn(2.0f, 3.0f);
    half2 right = __floats2half2_rn(4.0f, 5.0f);
    half2 value = __floats2half2_rn(1.0f, 1.0f);
    value = left * right + value;
    value += __floats2half2_rn(1.0f, 2.0f);
    out[0] = value;
  }`, { f16Mode: "f32", workgroupSize: [1, 1, 1] });
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, { buffers: { out: createWgslFloat16Array(2) } }, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, { buffers: { out: createWgslFloat16Array(2) } }, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("value = ((left * right) + value);");
      expect(compiled.wgsl).toContain("value += vec2<f32>(");
      expect(Array.from(result.buffers.out as ArrayLike<number>)).toEqual([10, 18]);
      expect(Array.from(semanticResult.buffers.out as ArrayLike<number>)).toEqual([10, 18]);
    });

  it("lowers vector multiply and divide assignments", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ void scale(float4 *value) {
    *value /= 2.0f;
    *value *= 3.0f;
  }
  __global__ void vectorCompound(float4 *out) {
    float4 value = make_float4(8.0f, 16.0f, 24.0f, 32.0f);
    value /= 4.0f;
    value *= 2.0f;
    out[0] = value;
    scale(&out[0]);
  }`, { workgroupSize: [1, 1, 1] });
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, { buffers: { out: new Float32Array(4) } }, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, { buffers: { out: new Float32Array(4) } }, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(semanticKernelIrWgslPreflightFailure(compiled.wgslLegalizedKernelIr)).toBeUndefined();
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("value /= vec4<f32>(f32(4.0)");
      expect(compiled.wgsl).toContain("value *= vec4<f32>(f32(2.0)");
      expect(Array.from(result.buffers.out as Float32Array)).toEqual([6, 12, 18, 24]);
      expect(Array.from(semanticResult.buffers.out as Float32Array)).toEqual([6, 12, 18, 24]);
    });

  it("lowers CUDA half and half2 saturating arithmetic intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void halfSat(const half *x, const half2 *v, half *out, half2 *vecOut) {
    half a = x[0];
    half b = x[1];
    out[0] = __hadd_sat(a, b);
    out[1] = __hsub_sat(b, a);
    out[2] = __hmul_sat(a, __float2half(2.0f));
    out[3] = __hfma_sat(a, b, __float2half(0.25f));
    half2 pair = v[0];
    vecOut[0] = __hadd2_sat(pair, __floats2half2_rn(0.5f, 0.75f));
    vecOut[1] = __hsub2_sat(__floats2half2_rn(0.5f, 1.0f), pair);
    vecOut[2] = __hmul2_sat(pair, __floats2half2_rn(2.0f, 0.75f));
    vecOut[3] = __hfma2_sat(pair, __floats2half2_rn(2.0f, 4.0f), __floats2half2_rn(-1.0f, 0.25f));
  }`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          x: createWgslFloat16Array([0.75, 0.5]),
          v: createWgslFloat16Array([0.75, 0.25]),
          out: createWgslFloat16Array(4),
          vecOut: createWgslFloat16Array(8),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(compiled.wgsl).toContain("select(clamp");
      expect(Array.from(result.buffers.out as Iterable<number>)).toEqual([1, 0, 1, 0.625]);
      expect(Array.from(result.buffers.vecOut as Iterable<number>)).toEqual([1, 1, 0, 0.75, 1, 0.1875, 0.5, 1]);
      expect(Array.from(semanticResult.buffers.out as Iterable<number>)).toEqual([1, 0, 1, 0.625]);
      expect(Array.from(semanticResult.buffers.vecOut as Iterable<number>)).toEqual([1, 1, 0, 0.75, 1, 0.1875, 0.5, 1]);
    });

  it("lowers CUDA half2 unary math aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void half2Unary(const half2 *input, half2 *out) {
    half2 signedPair = input[0];
    half2 positivePair = input[1];
    out[0] = __habs2(signedPair);
    out[1] = __hceil2(signedPair);
    out[2] = __hfloor2(signedPair);
    out[3] = __hneg2(signedPair);
    out[4] = __hrcp2(positivePair);
    out[5] = __hrsqrt2(positivePair);
    out[6] = __hsqrt2(positivePair);
    out[7] = __htrunc2(signedPair);
  }`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          input: createWgslFloat16Array([-1.5, 1.25, 4, 16]),
          out: createWgslFloat16Array(16),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const expected = [1.5, 1.25, -1, 2, -2, 1, 1.5, -1.25, 0.25, 0.0625, 0.5, 0.25, 2, 4, -1, 1];

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(Array.from(result.buffers.out as Iterable<number>)).toEqual(expected);
      expect(Array.from(semanticResult.buffers.out as Iterable<number>)).toEqual(expected);
    });

  it("lowers CUDA half2 comparison intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void half2Compare(const half2 *a, const half2 *b, half2 *vec, uint *mask, int *flags) {
    half2 x = a[0];
    half2 y = b[0];
    half2 nx = a[1];
    half2 ny = b[1];
    vec[0] = __heq2(x, y);
    vec[1] = __hne2(x, y);
    vec[2] = __hgt2(x, y);
    vec[3] = __hge2(x, y);
    vec[4] = __hlt2(x, y);
    vec[5] = __hle2(x, y);
    vec[6] = __hequ2(nx, ny);
    vec[7] = __hneu2(nx, ny);
    vec[8] = __hgtu2(nx, ny);
    vec[9] = __hgeu2(nx, ny);
    vec[10] = __hltu2(nx, ny);
    vec[11] = __hleu2(nx, ny);
    vec[12] = __hisnan2(nx);
    vec[13] = __hisnan2(ny);
    mask[0] = __heq2_mask(x, y);
    mask[1] = __hne2_mask(x, y);
    mask[2] = __hgt2_mask(x, y);
    mask[3] = __hge2_mask(x, y);
    mask[4] = __hlt2_mask(x, y);
    mask[5] = __hle2_mask(x, y);
    mask[6] = __hgt2_mask(nx, ny);
    mask[7] = __hequ2_mask(nx, ny);
    mask[8] = __hneu2_mask(nx, ny);
    mask[9] = __hgtu2_mask(nx, ny);
    if (__hbeq2(x, y)) { flags[0] = 1; }
    if (__hbne2(x, y)) { flags[1] = 1; }
    if (__hbgt2(x, y)) { flags[2] = 1; }
    if (__hbge2(x, y)) { flags[3] = 1; }
    if (__hblt2(x, y)) { flags[4] = 1; }
    if (__hble2(x, y)) { flags[5] = 1; }
    if (__hbequ2(nx, ny)) { flags[6] = 1; }
    if (__hbneu2(nx, ny)) { flags[7] = 1; }
    if (__hbgtu2(nx, ny)) { flags[8] = 1; }
    if (__hbgeu2(nx, ny)) { flags[9] = 1; }
    if (__hbltu2(nx, ny)) { flags[10] = 1; }
    if (__hbleu2(nx, ny)) { flags[11] = 1; }
  }`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          a: createWgslFloat16Array([1, 2, NaN, 4]),
          b: createWgslFloat16Array([1, 3, 4, NaN]),
          vec: createWgslFloat16Array(28),
          mask: new Uint32Array(10),
          flags: new Int32Array(12),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const expectedVec = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1];
      const expectedMask = [0x0000ffff, 0xffff0000, 0, 0x0000ffff, 0xffff0000, 0xffffffff, 0, 0xffffffff, 0xffffffff, 0xffffffff];
      const expectedFlags = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1];

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(Array.from(result.buffers.vec as Iterable<number>)).toEqual(expectedVec);
      expect(Array.from(result.buffers.mask as Iterable<number>)).toEqual(expectedMask);
      expect(Array.from(result.buffers.flags as Iterable<number>)).toEqual(expectedFlags);
      expect(Array.from(semanticResult.buffers.vec as Iterable<number>)).toEqual(expectedVec);
      expect(Array.from(semanticResult.buffers.mask as Iterable<number>)).toEqual(expectedMask);
      expect(Array.from(semanticResult.buffers.flags as Iterable<number>)).toEqual(expectedFlags);
    });

  it("lowers CUDA half2 NaN-propagating min/max intrinsics", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void half2MinMaxNan(const half2 *a, const half2 *b, half2 *out, half2 *nanFlags) {
    half2 x = a[0];
    half2 y = b[0];
    half2 nx = a[1];
    half2 ny = b[1];
    out[0] = __hmin2_nan(x, y);
    out[1] = __hmax2_nan(x, y);
    out[2] = __hmin2_nan(nx, ny);
    out[3] = __hmax2_nan(nx, ny);
    nanFlags[0] = __hisnan2(out[2]);
    nanFlags[1] = __hisnan2(out[3]);
  }`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          a: createWgslFloat16Array([1, 4, NaN, 2]),
          b: createWgslFloat16Array([3, 2, 5, NaN]),
          out: createWgslFloat16Array(8),
          nanFlags: createWgslFloat16Array(4),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const out = Array.from(result.buffers.out as Iterable<number>);
      const semanticOut = Array.from(semanticResult.buffers.out as Iterable<number>);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(out.slice(0, 4)).toEqual([1, 2, 3, 4]);
      expect(Number.isNaN(out[4])).toBe(true);
      expect(Number.isNaN(out[5])).toBe(true);
      expect(Number.isNaN(out[6])).toBe(true);
      expect(Number.isNaN(out[7])).toBe(true);
      expect(Array.from(result.buffers.nanFlags as Iterable<number>)).toEqual([1, 1, 1, 1]);
      expect(semanticOut.slice(0, 4)).toEqual([1, 2, 3, 4]);
      expect(Number.isNaN(semanticOut[4])).toBe(true);
      expect(Number.isNaN(semanticOut[5])).toBe(true);
      expect(Number.isNaN(semanticOut[6])).toBe(true);
      expect(Number.isNaN(semanticOut[7])).toBe(true);
      expect(Array.from(semanticResult.buffers.nanFlags as Iterable<number>)).toEqual([1, 1, 1, 1]);
    });

  it("lowers CUDA half2 lane extraction and packing helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void half2LaneHelpers(const half2 *input, half2 *out, half *scalar) {
    half2 x = input[0];
    half2 y = input[1];
    half lo = __low2half(x);
    half hi = __high2half(x);
    scalar[0] = lo;
    scalar[1] = hi;
    out[0] = __halves2half2(hi, lo);
    out[1] = __half2half2(lo);
    out[2] = __low2half2(x);
    out[3] = __high2half2(x);
    out[4] = __lows2half2(x, y);
    out[5] = __highs2half2(x, y);
    out[6] = __lowhigh2highlow(x);
  }`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          input: createWgslFloat16Array([1, 2, 3, 4]),
          out: createWgslFloat16Array(14),
          scalar: createWgslFloat16Array(2),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const expectedOut = [2, 1, 1, 1, 1, 1, 2, 2, 1, 3, 2, 4, 2, 1];

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(Array.from(result.buffers.scalar as Iterable<number>)).toEqual([1, 2]);
      expect(Array.from(result.buffers.out as Iterable<number>)).toEqual(expectedOut);
      expect(Array.from(semanticResult.buffers.scalar as Iterable<number>)).toEqual([1, 2]);
      expect(Array.from(semanticResult.buffers.out as Iterable<number>)).toEqual(expectedOut);
    });

  it("feature-gates half2 behind shader-f16", () => {
      expect(() => compileCudaLiteKernel(`
  __global__ void half2Gate(half2 *x) {
    x[0] = make_half2(__float2half(1.0f), __float2half(2.0f));
  }`, { features: { "shader-f16": false }, workgroupSize: [1, 1, 1] })).toThrow(/half requires WebGPU shader-f16 support/);
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

  it("lowers CUDA cuRAND vector-pair draws through semantic IR", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __global__ void curandPairKernel(float *out, unsigned int seed) {
    curandState_t state;
    curand_init(seed, threadIdx.x, 0, &state);
    float2 normal = curand_normal2(&state);
    float2 logn = curand_log_normal2(&state, 0.2f, 0.4f);
    out[threadIdx.x] = normal.x + normal.y + logn.x + logn.y;
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(4) }, scalars: { seed: 2468 } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bg_curand_normal2(state: ptr<function, u32>) -> vec2<f32>");
      expect(compiled.wgsl).toContain("var normal: vec2<f32> = bg_curand_normal2(&state)");
      expect(compiled.wgsl).toContain("bg_curand_log_normal2(&state, 0.2, 0.4)");
      expect([...result.buffers.out as Float32Array].every((value) => Number.isFinite(value))).toBe(true);
    });

  it("lowers CUDA Philox cuRAND vector4 draws through semantic IR", () => {
      const compiled = compileCudaLiteKernelForWebGpu(`
  __global__ void curandPhilox4Kernel(float *out, unsigned int *ints, unsigned int seed) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, threadIdx.x, 0, &state);
    float4 uni = curand_uniform4(&state);
    float4 normal = curand_normal4(&state);
    float4 logn = curand_log_normal4(&state, 0.2f, 0.4f);
    uint4 pois = curand_poisson4(&state, 4.0f);
    ints[threadIdx.x] = pois.x + pois.y + pois.z + pois.w;
    out[threadIdx.x] = uni.x + uni.y + uni.z + uni.w + normal.x + normal.y + normal.z + normal.w + logn.x + logn.y + logn.z + logn.w + (float)ints[threadIdx.x];
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        { buffers: { out: new Float32Array(4), ints: new Uint32Array(4) }, scalars: { seed: 24680 } },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn bg_curand_uniform4(state: ptr<function, u32>) -> vec4<f32>");
      expect(compiled.wgsl).toContain("fn bg_curand_poisson4(state: ptr<function, u32>, lambda: f32) -> vec4<u32>");
      expect(JSON.stringify(compiled.kernelIr)).toContain("curand_uniform4");
      expect([...result.buffers.out as Float32Array].every((value) => Number.isFinite(value))).toBe(true);
      expect([...result.buffers.ints as Uint32Array].every((value) => value > 0)).toBe(true);
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

      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var<storage, read_write> data: array<f32>;");
      expect(compiled.wgsl).toContain("data[((u32(idx) * 2u) + 0u)]");
      expect(compiled.wgsl).toContain("data[((u32(idx) * 2u) + 1u)]");
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
      expect(compiled.wgsl).toMatch(/let bg_vector_store_value_A_\d+: vec2<f32> = c;/u);
      expect(compiled.wgsl).toMatch(/A\[\(bg_vector_store_base_A_\d+ \+ 1u\)\] = \(bg_vector_store_value_A_\d+\)\.y;/u);
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

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("fn ComplexMul");
      expect(compiled.wgsl).toContain("a[((u32(i) * 2u) + 0u)]");
      expect(compiled.wgsl).toContain("a[((u32(i) * 2u) + 1u)]");
      expect(compiled.wgsl).not.toContain("f32(vec2<f32>");
      expect([...result.buffers.a as Float32Array]).toEqual([-3.5, 8, -5.5, 26]);
    });

  it("lowers direct cufftComplex lane reads and writes through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void complexLanes(cufftComplex *src, cufftComplex *dst, float scale) {
    int i = threadIdx.x;
    dst[i].x = -src[i].x / scale;
    dst[i].y = -src[i].y / scale;
  }`, { workgroupSize: [2, 1, 1] });
      const result = runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            src: new Float32Array([2, -4, 6, -8]),
            dst: new Float32Array(4),
          },
          scalars: { scale: 2 },
        },
        { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("dst[((u32(i) * 2u) + 0u)]");
      expect(compiled.wgsl).toContain("dst[((u32(i) * 2u) + 1u)]");
      expect([...result.buffers.dst as Float32Array]).toEqual([-1, 2, -3, 4]);
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
      expect(compiled.wgsl).toContain("var x: vec2<f32>");
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
      expect(compiled.wgsl).toMatch(/let bg_vector_store_value_d_\d+: vec2<f32> =/u);
      expect(compiled.wgsl).toMatch(/d\[\(bg_vector_store_base_d_\d+ \+ 1u\)\] = \(bg_vector_store_value_d_\d+\)\.y;/u);
      expect(compiled.wgsl).not.toContain("unsupported CUDA-lite call");
      expect([...result.buffers.d as Float32Array]).toEqual([2, 26, 0, 64]);
    });

  it("lowers cuDoubleComplex helpers through f32-compatible native complex ops", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void cuDoubleComplexCompat(cuDoubleComplex *a, cuDoubleComplex *b, float *out) {
    int i = threadIdx.x;
    cuDoubleComplex x = make_cuDoubleComplex(a[i].x, a[i].y);
    cuDoubleComplex y = make_cuDoubleComplex(b[i].x, b[i].y);
    cuDoubleComplex z = cuCadd(cuCmul(x, y), cuConj(y));
    cuDoubleComplex q = cuCdiv(z, make_cuDoubleComplex(2.0, 0.0));
    a[i] = cuCsub(q, make_cuDoubleComplex(1.0, -1.0));
    b[i] = cuCfma(x, y, b[i]);
    out[i] = cuCabs(a[i]) + cuCreal(a[i]) + cuCimag(a[i]);
  }`, { workgroupSize: [2, 1, 1], f64Mode: "f32" });
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

      const codes = compiled.diagnostics.map((diagnostic) => diagnostic.code);
      expect(codes).not.toContain("unsupported-call");
      expect(codes).not.toContain("unsupported-cufft");
      expect(codes).toContain("f64-lowered-to-f32");
      expect(compiled.wgsl).toContain("var x: vec2<f32>");
      expect(compiled.wgsl).toContain("fn bg_cuCabsf");
      expect(compiled.wgsl).toContain("fn bg_cuCdivf");
      expect([...result.buffers.a as Float32Array]).toEqual([-2, 6, -3, 23]);
      expect([...result.buffers.b as Float32Array]).toEqual([-2, 22, -4, 60]);
      expect([...result.buffers.out as Float32Array]).toEqual([...Float32Array.from([Math.hypot(-2, 6) + 4, Math.hypot(-3, 23) + 20])]);
    });

  it("rejects cuDoubleComplex helper lowering without explicit f64 compatibility mode", () => {
      expect(() => compileCudaLiteKernel(`
  __global__ void cuDoubleComplexStrict(cuDoubleComplex *a, cuDoubleComplex *b) {
    int i = threadIdx.x;
    a[i] = cuCadd(a[i], b[i]);
  }`, { workgroupSize: [2, 1, 1] })).toThrow(/unsupported-f64/u);
    });

  it("lowers inline PTX bitwise b32 statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int and_ptx(unsigned int a, unsigned int b) {
    unsigned int ret;
    asm volatile("and.b32 %0, %1, %2;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __device__ unsigned int or_ptx(unsigned int a, unsigned int b) {
    unsigned int ret;
    asm volatile("or.b32 %0, %1, %2;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __device__ unsigned int xor_ptx(unsigned int a, unsigned int b) {
    unsigned int ret;
    asm volatile("xor.b32 %0, %1, %2;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __device__ unsigned int not_ptx(unsigned int a) {
    unsigned int ret;
    asm volatile("not.b32 %0, %1;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __global__ void bitwiseKernel(uint *out, uint *a, uint *b) {
    int idx = threadIdx.x;
    out[idx] = and_ptx(a[idx], b[idx]);
    out[idx + 4] = or_ptx(a[idx], b[idx]);
    out[idx + 8] = xor_ptx(a[idx], b[idx]);
    out[idx + 12] = not_ptx(a[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(16),
            a: new Uint32Array([0xffffffff, 0x12345678, 0xf0f0f0f0, 0xaaaaaaaa]),
            b: new Uint32Array([0, 0x87654321, 0x0f0f0f0f, 0x55555555]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain(" & ");
      expect(compiled.wgsl).toContain(" | ");
      expect(compiled.wgsl).toContain(" ^ ");
      expect(compiled.wgsl).toContain("~(u32");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0,
        0x02244220,
        0,
        0,
        0xffffffff,
        0x97755779,
        0xffffffff,
        0xffffffff,
        0xffffffff,
        0x95511559,
        0xffffffff,
        0xffffffff,
        0,
        0xedcba987,
        0x0f0f0f0f,
        0x55555555,
      ]);
    });

  it("lowers inline PTX bitwise b32 immediate statements", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int and_imm_ptx(unsigned int a) {
    unsigned int ret;
    asm volatile("and.b32 %0, %1, 0x0f0f0f0f;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __device__ unsigned int or_imm_ptx(unsigned int a) {
    unsigned int ret;
    asm volatile("or.b32 %0, %1, 0x11111111;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __device__ unsigned int xor_imm_ptx(unsigned int a) {
    unsigned int ret;
    asm volatile("xor.b32 %0, %1, 0xffffffff;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __global__ void bitwiseImmediateKernel(uint *out, uint *a) {
    int idx = threadIdx.x;
    out[idx] = and_imm_ptx(a[idx]);
    out[idx + 4] = or_imm_ptx(a[idx]);
    out[idx + 8] = xor_imm_ptx(a[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(12),
            a: new Uint32Array([0xffffffff, 0x12345678, 0xf0f0f0f0, 0xaaaaaaaa]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("252645135u");
      expect(compiled.wgsl).toContain("4294967295u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0x0f0f0f0f,
        0x02040608,
        0,
        0x0a0a0a0a,
        0xffffffff,
        0x13355779,
        0xf1f1f1f1,
        0xbbbbbbbb,
        0,
        0xedcba987,
        0x0f0f0f0f,
        0x55555555,
      ]);
    });

  it("lowers inline PTX setp integer comparisons", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int setp_eq_u(unsigned int a, unsigned int b) {
    unsigned int ret;
    asm volatile("setp.eq.u32 %0, %1, %2;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __device__ unsigned int setp_lt_u(unsigned int a, unsigned int b) {
    unsigned int ret;
    asm volatile("setp.lt.u32 %0, %1, %2;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __device__ unsigned int setp_ge_s(int a, int b) {
    unsigned int ret;
    asm volatile("setp.ge.s32 %0, %1, %2;" : "=r"(ret) : "r"(a), "r"(b));
    return ret;
  }
  __global__ void compareKernel(uint *out, uint *a, uint *b) {
    int idx = threadIdx.x;
    out[idx] = setp_eq_u(a[idx], b[idx]);
    out[idx + 4] = setp_lt_u(a[idx], b[idx]);
    out[idx + 8] = setp_ge_s((int)a[idx], (int)b[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(12),
            a: new Uint32Array([1, 0xffffffff, 0x80000000, 4]),
            b: new Uint32Array([1, 2, 0x7fffffff, 5]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.wgsl).toContain("select(0u, 1u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
      ]);
    });

  it("lowers inline PTX setp integer immediate comparisons", () => {
      const compiled = compileCudaLiteKernel(`
  __device__ unsigned int setp_eq_u_imm(unsigned int a) {
    unsigned int ret;
    asm volatile("setp.eq.u32 %0, %1, 0x7fffffff;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __device__ unsigned int setp_lt_u_imm(unsigned int a) {
    unsigned int ret;
    asm volatile("setp.lt.u32 %0, %1, 2;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __device__ unsigned int setp_ge_s_imm(int a) {
    unsigned int ret;
    asm volatile("setp.ge.s32 %0, %1, -8;" : "=r"(ret) : "r"(a));
    return ret;
  }
  __global__ void compareImmediateKernel(uint *out, uint *a) {
    int idx = threadIdx.x;
    out[idx] = setp_eq_u_imm(a[idx]);
    out[idx + 4] = setp_lt_u_imm(a[idx]);
    out[idx + 8] = setp_ge_s_imm((int)a[idx]);
  }`, { workgroupSize: [4, 1, 1] });
      const result = runCompiledKernelReference(
        compiled,
        {
          buffers: {
            out: new Uint32Array(12),
            a: new Uint32Array([1, 0xffffffff, 0x80000000, 4]),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-inline-asm");
      expect(compiled.wgsl).toContain("2147483647u");
      expect(compiled.wgsl).toContain("4294967288u");
      expect([...result.buffers.out as Uint32Array]).toEqual([
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
      ]);
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
      expect(semanticIr(compiled).requiredFeatures).not.toContain("shader-f16");
      expect(compiled.kernelIr.memory.filter((symbol) => symbol.kind === "constant").map((symbol) => symbol.name)).not.toContain("unused_coeffs");
      expect(compiled.kernelIr.memory.filter((symbol) => symbol.kind === "device-global").map((symbol) => symbol.name)).not.toContain("unused_state");
      expect(compiled.kernelIr.memory.map((symbol) => symbol.name)).not.toContain("unused_coeffs");
      expect(compiled.kernelIr.memory.map((symbol) => symbol.name)).not.toContain("unused_state");
      expect([...result.buffers.x as Float32Array]).toEqual([5]);
      expect(compiled.wgsl).not.toContain("unused_coeffs");
      expect(compiled.wgsl).not.toContain("unused_state");
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("if ((u32(bg_uniforms.enabled) != 0u))");
      expect(compiled.wgsl).toMatch(/bg__bg_condition_value_\d+_\d+ = bitcast<i32>\(active_conditional_vector_lane_lvalue_helper_with_pointer_side_effect/u);
      expect(compiled.wgsl).toContain("active_conditional_vector_lane_lvalue_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0, i32(active_conditional_vector_lane_lvalue_helper_with_pointer_side_effect");
      expect(compiled.wgsl).not.toContain("select(0u, active_conditional_vector_lane_lvalue_helper_with_pointer_side_effect");
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

      expect(compiled.kernelIr.memory.find((symbol) => symbol.kind === "shared")?.dimensions).toEqual([2]);
      expect(compiled.wgsl).toContain("var<workgroup> scratch: array<f32, 2>;");
      expect([...result.buffers.out as Float32Array]).toEqual([1]);
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("const metric: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0)");
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([6]);
      expect([...result.buffers.out as Float32Array]).toEqual([6]);
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
      const semanticResult = runCompiledKernelSemanticReference(
        compiled,
        { buffers: { out: new Float32Array(4) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("min(");
      expect(compiled.wgsl).toContain("max(");
      expect([...result.buffers.out as Float32Array]).toEqual([0, 2, 8, 255]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([0, 2, 8, 255]);
    });

  it("lowers vector assignment chains and POD-field aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void vectorChain(float4 *x, float4 *y, float *out) {
    float4 value = make_float4(2.0f, 3.0f, 5.0f, 7.0f);
    x[0] = y[0] = value;
    float4 record = x[0];
    out[0] = record.S + record.X + record.MuByT + record.VBySqrtT;
  }`, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          x: new Float32Array(4),
          y: new Float32Array(4),
          out: new Float32Array(1),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const result = runCompiledKernelReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect([...semanticResult.buffers.x as Float32Array]).toEqual([2, 3, 5, 7]);
      expect([...semanticResult.buffers.y as Float32Array]).toEqual([2, 3, 5, 7]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([17]);
      expect([...result.buffers.x as Float32Array]).toEqual([2, 3, 5, 7]);
      expect([...result.buffers.y as Float32Array]).toEqual([2, 3, 5, 7]);
      expect([...result.buffers.out as Float32Array]).toEqual([17]);
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("enable f16;");
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(semanticIr(compiled).requiredFeatures).not.toContain("shader-f16");
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

  it("lowers scalar CUDA bf16 arithmetic and predicates through semantic IR", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void bf16_scalar_ops(const __nv_bfloat16* input, const float* seed, __nv_bfloat16* output, uint* flags) {
    __nv_bfloat16 a = input[0];
    __nv_bfloat16 b = input[1];
    __nv_bfloat16 nanValue = __float2bfloat16(seed[0]);
    __nv_bfloat16 infValue = __float2bfloat16(seed[1]);
    __nv_bfloat16 c = __float2bfloat16(-1.75f);
    __nv_bfloat16 d = __float2bfloat16(4.0f);
    output[0] = __hadd(a, b);
    output[1] = __hsub(a, b);
    output[2] = __hmul(a, b);
    output[3] = __hdiv(a, b);
    output[4] = __hfma(a, b, __float2bfloat16(1.0f));
    output[5] = __hmin(a, b);
    output[6] = __hmax(a, b);
    output[7] = __hadd_rn(a, b);
    output[8] = __hadd_sat(__float2bfloat16(0.75f), __float2bfloat16(0.5f));
    output[9] = __hsub_sat(__float2bfloat16(0.5f), __float2bfloat16(0.75f));
    output[10] = __hmul_sat(__float2bfloat16(0.75f), __float2bfloat16(2.0f));
    output[11] = __hfma_sat(__float2bfloat16(0.75f), __float2bfloat16(2.0f), __float2bfloat16(-0.25f));
    output[12] = __hfma_relu(__float2bfloat16(0.5f), __float2bfloat16(1.0f), __float2bfloat16(-2.0f));
    output[13] = __hmin_nan(nanValue, b);
    output[14] = __hmax_nan(a, nanValue);
    output[15] = __habs(c);
    output[16] = __hceil(__float2bfloat16(1.25f));
    output[17] = __hfloor(__float2bfloat16(1.75f));
    output[18] = __htrunc(c);
    output[19] = __hrcp(a);
    output[20] = __hrsqrt(d);
    output[21] = __hsqrt(d);
    output[22] = __hneg(b);
    output[23] = hexp(__float2bfloat16(0.0f));
    if (__heq(a, __float2bfloat16(2.0f))) { flags[0] = 1u; }
    if (__hne(a, b)) { flags[1] = 1u; }
    if (__hgt(a, b)) { flags[2] = 1u; }
    if (__hge(a, a)) { flags[3] = 1u; }
    if (__hlt(b, a)) { flags[4] = 1u; }
    if (__hle(b, b)) { flags[5] = 1u; }
    if (__hequ(nanValue, b)) { flags[6] = 1u; }
    if (__hneu(nanValue, b)) { flags[7] = 1u; }
    if (__hgtu(nanValue, b)) { flags[8] = 1u; }
    if (__hgeu(nanValue, b)) { flags[9] = 1u; }
    if (__hltu(nanValue, b)) { flags[10] = 1u; }
    if (__hleu(nanValue, b)) { flags[11] = 1u; }
    if (__hisnan(nanValue)) { flags[12] = 1u; }
    if (__hisinf(infValue)) { flags[13] = 1u; }
  }`, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          input: new Float32Array([2, 0.5]),
          seed: new Float32Array([Number.NaN, Number.POSITIVE_INFINITY]),
          output: new Float32Array(24),
          flags: new Uint32Array(14),
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
            seed: new Float32Array([Number.NaN, Number.POSITIVE_INFINITY]),
            output: new Float32Array(24),
            flags: new Uint32Array(14),
          },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      );

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(semanticIr(compiled).requiredFeatures).not.toContain("shader-f16");
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect([...result.buffers.output as Float32Array]).toEqual([2.5, 1.5, 1, 4, 2, 0.5, 2, 2.5, 1, 0, 1, 1, 0, Number.NaN, Number.NaN, 1.75, 2, 1, -1, 0.5, 0.5, 2, -0.5, 1]);
      expect([...semanticResult.buffers.output as Float32Array]).toEqual([2.5, 1.5, 1, 4, 2, 0.5, 2, 2.5, 1, 0, 1, 1, 0, Number.NaN, Number.NaN, 1.75, 2, 1, -1, 0.5, 0.5, 2, -0.5, 1]);
      expect([...result.buffers.flags as Uint32Array]).toEqual(Array.from({ length: 14 }, () => 1));
      expect([...semanticResult.buffers.flags as Uint32Array]).toEqual(Array.from({ length: 14 }, () => 1));
    });

  it("lowers CUDA bitwise not without treating trap as a no-op primitive", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void bitwise_not(int* out) {
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
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
      half sum = __hadd_rn(a, b);
      half diff = __hsub_rn(sum, __float2half(0.5f));
      half prod = __hmul_rn(diff, __float2half(2.0f));
      half quot = __hdiv_rn(prod, __float2half(2.0f));
      half neg = __hneg(quot);
      half mixed = __hfma_rn(neg, __float2half(-1.0f), __float2half(0.25f));
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("enable f16;");
      expect(Array.from(result.buffers.output as Iterable<number>)).toEqual([1.5]);
      expect([...result.buffers.flags as Int32Array]).toEqual([1, 1, 1, 1, 1, 1]);
      expect(Array.from(semanticResult.buffers.output as Iterable<number>)).toEqual([1.5]);
      expect([...semanticResult.buffers.flags as Int32Array]).toEqual([1, 1, 1, 1, 1, 1]);
    });

  it("lowers scalar CUDA half unordered comparison and NaN predicates", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void half_unordered(const half* input, half* output, int* flags) {
    half finite = input[0];
    half lower = input[1];
    half nanv = input[2];
    half pinf = input[3];
    half ninf = input[4];
    output[0] = __hmin_nan(finite, nanv);
    output[1] = __hmax_nan(nanv, lower);
    if (__hequ(nanv, finite)) { flags[0] = 1; }
    if (__hneu(nanv, finite)) { flags[1] = 1; }
    if (__hgtu(nanv, finite)) { flags[2] = 1; }
    if (__hgeu(nanv, finite)) { flags[3] = 1; }
    if (__hltu(nanv, finite)) { flags[4] = 1; }
    if (__hleu(nanv, finite)) { flags[5] = 1; }
    if (__hgt(finite, lower)) { flags[6] = 1; }
    if (__hisnan(nanv)) { flags[7] = 1; }
    flags[8] = __hisinf(pinf);
    flags[9] = __hisinf(ninf);
  }`, {
        features: { "shader-f16": true },
        workgroupSize: [1, 1, 1],
      });
      const input = {
        buffers: {
          input: createWgslFloat16Array([2, 1, NaN, Infinity, -Infinity]),
          output: createWgslFloat16Array(2),
          flags: new Int32Array(10),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const output = Array.from(result.buffers.output as Iterable<number>);
      const semanticOutput = Array.from(semanticResult.buffers.output as Iterable<number>);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(Number.isNaN(output[0])).toBe(true);
      expect(Number.isNaN(output[1])).toBe(true);
      expect([...result.buffers.flags as Int32Array]).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, -1]);
      expect(Number.isNaN(semanticOutput[0])).toBe(true);
      expect(Number.isNaN(semanticOutput[1])).toBe(true);
      expect([...semanticResult.buffers.flags as Int32Array]).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, -1]);
    });

  it("lowers scalar CUDA half short conversion and bitcast helpers", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void half_short_convert(const half* input, int* out, uint* uout, half* h) {
    half pos = input[0];
    half neg = input[1];
    out[0] = __half2short_rn(pos);
    out[1] = __half2short_rz(pos);
    out[2] = __half2short_ru(pos);
    out[3] = __half2short_rd(pos);
    out[4] = __half2short_rn(neg);
    out[5] = __half2short_rz(neg);
    out[6] = __half2short_ru(neg);
    out[7] = __half2short_rd(neg);
    out[8] = __half_as_short(__float2half(-2.0f));
    uout[0] = __half2ushort_rn(pos);
    uout[1] = __half2ushort_rz(pos);
    uout[2] = __half2ushort_ru(pos);
    uout[3] = __half2ushort_rd(pos);
    h[0] = __short_as_half(0xc000);
    h[1] = __ushort_as_half(0x3c00u);
  }`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          input: createWgslFloat16Array([1.5, -1.5]),
          out: new Int32Array(9),
          uout: new Uint32Array(4),
          h: createWgslFloat16Array(2),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect([...result.buffers.out as Int32Array]).toEqual([2, 1, 2, 1, -2, -1, -1, -2, -16384]);
      expect([...result.buffers.uout as Uint32Array]).toEqual([2, 1, 2, 1]);
      expect(Array.from(result.buffers.h as Iterable<number>)).toEqual([-2, 1]);
      expect([...semanticResult.buffers.out as Int32Array]).toEqual([2, 1, 2, 1, -2, -1, -1, -2, -16384]);
      expect([...semanticResult.buffers.uout as Uint32Array]).toEqual([2, 1, 2, 1]);
      expect(Array.from(semanticResult.buffers.h as Iterable<number>)).toEqual([-2, 1]);
    });

  it("lowers CUDA directed half conversion aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void half_directed_convert(half* out) {
    out[0] = __float2half_rn(2049.0f);
    out[1] = __float2half_rz(2049.0f);
    out[2] = __float2half_ru(2049.0f);
    out[3] = __float2half_rd(2049.0f);
    out[4] = __float2half_rn(-2049.0f);
    out[5] = __float2half_rz(-2049.0f);
    out[6] = __float2half_ru(-2049.0f);
    out[7] = __float2half_rd(-2049.0f);
    out[8] = __int2half_rn(2049);
    out[9] = __int2half_ru(2049);
    out[10] = __int2half_rd(-2049);
    out[11] = __uint2half_ru(2049u);
    out[12] = __short2half_rz(32767);
    out[13] = __short2half_rd(-32767);
    out[14] = __ushort2half_ru(2049u);
    out[15] = __short2half_rn(0xffff);
  }`, { features: { "shader-f16": true }, workgroupSize: [1, 1, 1] });
      const input = { buffers: { out: createWgslFloat16Array(16) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const expected = [2048, 2048, 2050, 2048, -2048, -2048, -2048, -2050, 2048, 2050, -2050, 2050, 32752, -32768, 2050, -1];

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(Array.from(result.buffers.out as Iterable<number>)).toEqual(expected);
      expect(Array.from(semanticResult.buffers.out as Iterable<number>)).toEqual(expected);
    });

  it("lowers CUDA directed bf16 conversion aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void bf16Directed(float *out, uint *bits, int *signedBits) {
    if (threadIdx.x < 1) {
      __nv_bfloat16 prn = __float2bfloat16_rn(257.0f);
      out[0] = __bfloat162float(prn);
      out[1] = __bfloat162float(__float2bfloat16_rz(257.0f));
      out[2] = __bfloat162float(__float2bfloat16_ru(257.0f));
      out[3] = __bfloat162float(__float2bfloat16_rd(257.0f));
      out[4] = __bfloat162float(__float2bfloat16_rn(-257.0f));
      out[5] = __bfloat162float(__float2bfloat16_rz(-257.0f));
      out[6] = __bfloat162float(__float2bfloat16_ru(-257.0f));
      out[7] = __bfloat162float(__float2bfloat16_rd(-257.0f));
      out[8] = __bfloat162float(__int2bfloat16_ru(257));
      out[9] = __bfloat162float(__int2bfloat16_rd(-257));
      out[10] = __bfloat162float(__uint2bfloat16_ru(257u));
      out[11] = __bfloat162float(__short2bfloat16_rn(0xffff));
      out[12] = __bfloat162float(__short2bfloat16_rd(-257));
      out[13] = __bfloat162float(__ushort2bfloat16_ru(257u));
      out[14] = __bfloat162float(__ushort2bfloat16_rd(257u));
      out[15] = __bfloat162float(__ushort_as_bfloat16(0x4000u));
      out[16] = __bfloat162float(__ll2bfloat16_rn((long long)257));
      out[17] = __bfloat162float(__ll2bfloat16_rd((long long)-257));
      out[18] = __bfloat162float(__ull2bfloat16_ru((unsigned long long)257u));
      out[19] = __bfloat162float(__ull2bfloat16_rd((unsigned long long)257u));
      bits[0] = __bfloat16_as_ushort(prn);
      bits[1] = __bfloat16_as_ushort(__ushort_as_bfloat16(0x3f80u));
      bits[2] = __bfloat162ushort_rn(__float2bfloat16_rn(1.5f));
      bits[3] = __bfloat162ushort_rz(__float2bfloat16_rn(1.5f));
      bits[4] = __bfloat162ushort_ru(__float2bfloat16_rn(1.5f));
      bits[5] = __bfloat162ushort_rd(__float2bfloat16_rn(1.5f));
      bits[6] = __bfloat162uchar_rz(__float2bfloat16_rn(255.0f));
      bits[7] = __bfloat162uchar_rz(__float2bfloat16_rn(257.0f));
      bits[8] = __bfloat162ull_rn(__float2bfloat16_rn(1.5f));
      bits[9] = __bfloat162ull_rz(__float2bfloat16_rn(1.5f));
      bits[10] = __bfloat162ull_ru(__float2bfloat16_rn(1.5f));
      bits[11] = __bfloat162ull_rd(__float2bfloat16_rn(1.5f));
      signedBits[0] = __bfloat16_as_short(__short_as_bfloat16(0xbf80));
      signedBits[1] = __bfloat162short_rn(__float2bfloat16_rn(1.5f));
      signedBits[2] = __bfloat162short_rz(__float2bfloat16_rn(1.5f));
      signedBits[3] = __bfloat162short_ru(__float2bfloat16_rn(1.5f));
      signedBits[4] = __bfloat162short_rd(__float2bfloat16_rn(1.5f));
      signedBits[5] = __bfloat162short_rn(__float2bfloat16_rn(-1.5f));
      signedBits[6] = __bfloat162short_rz(__float2bfloat16_rn(-1.5f));
      signedBits[7] = __bfloat162short_ru(__float2bfloat16_rn(-1.5f));
      signedBits[8] = __bfloat162short_rd(__float2bfloat16_rn(-1.5f));
      signedBits[9] = __bfloat162char_rz(__float2bfloat16_rn(255.0f));
      signedBits[10] = __bfloat162char_rz(__float2bfloat16_rn(129.0f));
      signedBits[11] = __bfloat162ll_rn(__float2bfloat16_rn(1.5f));
      signedBits[12] = __bfloat162ll_rz(__float2bfloat16_rn(1.5f));
      signedBits[13] = __bfloat162ll_ru(__float2bfloat16_rn(1.5f));
      signedBits[14] = __bfloat162ll_rd(__float2bfloat16_rn(1.5f));
      signedBits[15] = __bfloat162ll_rn(__float2bfloat16_rn(-1.5f));
      signedBits[16] = __bfloat162ll_rz(__float2bfloat16_rn(-1.5f));
      signedBits[17] = __bfloat162ll_ru(__float2bfloat16_rn(-1.5f));
      signedBits[18] = __bfloat162ll_rd(__float2bfloat16_rn(-1.5f));
    }
  }`, { workgroupSize: [1, 1, 1] });
      const input = {
        buffers: {
          out: new Float32Array(20),
          bits: new Uint32Array(12),
          signedBits: new Int32Array(19),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const expected = [256, 256, 258, 256, -256, -256, -256, -258, 258, -258, 258, -1, -258, 258, 256, 2, 256, -258, 258, 256];

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(compiled.wgsl).toContain("bg_f32_to_bf16_bits_mode");
      expect([...result.buffers.out as Float32Array]).toEqual(expected);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual(expected);
      expect([...result.buffers.bits as Uint32Array]).toEqual([0x4380, 0x3f80, 2, 1, 2, 1, 255, 0, 2, 1, 2, 1]);
      expect([...semanticResult.buffers.bits as Uint32Array]).toEqual([0x4380, 0x3f80, 2, 1, 2, 1, 255, 0, 2, 1, 2, 1]);
      expect([...result.buffers.signedBits as Int32Array]).toEqual([-16512, 2, 1, 2, 1, -2, -1, -1, -2, -1, -127, 2, 1, 2, 1, -2, -1, -1, -2]);
      expect([...semanticResult.buffers.signedBits as Int32Array]).toEqual([-16512, 2, 1, 2, 1, -2, -1, -1, -2, -1, -127, 2, 1, 2, 1, -2, -1, -1, -2]);
    });

  it("lowers CUDA double to bf16 only through explicit f32 compatibility mode", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void bf16DoubleCompat(float *out, double d) {
    if (threadIdx.x < 1) {
      out[0] = __bfloat162float(__double2bfloat16(d));
    }
  }`, { f64Mode: "f32", workgroupSize: [1, 1, 1] });
      const input = {
        buffers: { out: new Float32Array(1) },
        scalars: { d: 257 },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).toContain("f64-lowered-to-f32");
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(compiled.wgsl).toContain("bg_f32_to_bf16_bits_mode");
      expect([...result.buffers.out as Float32Array]).toEqual([256]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([256]);
      expect(() => compileCudaLiteKernel(`
  __global__ void bf16DoubleStrict(float *out) {
    double d = 257.0;
    out[0] = __bfloat162float(__double2bfloat16(d));
  }`, { workgroupSize: [1, 1, 1] })).toThrow(/unsupported-f64/u);
    });

  it("lowers CUDA bf162 lane and vector conversion aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void bf162Move(float *out, uint *bits) {
    if (threadIdx.x < 1) {
      __nv_bfloat162 pair = __floats2bfloat162_rn(257.0f, -257.0f);
      float2 floats = __bfloat1622float2(pair);
      __nv_bfloat162 fromVec = __float22bfloat162_rn(make_float2(257.0f, -257.0f));
      __nv_bfloat162 splat = __bfloat162bfloat162(1.5f);
      __nv_bfloat162 splat2 = __float2bfloat162_rn(257.0f);
      __nv_bfloat162 lo = __low2bfloat162(pair);
      __nv_bfloat162 hi = __high2bfloat162(pair);
      __nv_bfloat162 lows = __lows2bfloat162(pair, fromVec);
      __nv_bfloat162 highs = __highs2bfloat162(pair, fromVec);
      __nv_bfloat162 swapped = __lowhigh2highlow(pair);
      out[0] = floats.x;
      out[1] = floats.y;
      out[2] = __low2float(fromVec);
      out[3] = __high2float(fromVec);
      out[4] = __bfloat162float(__low2bfloat16(pair));
      out[5] = __bfloat162float(__high2bfloat16(pair));
      out[6] = lo.x;
      out[7] = lo.y;
      out[8] = hi.x;
      out[9] = hi.y;
      out[10] = lows.x;
      out[11] = lows.y;
      out[12] = highs.x;
      out[13] = highs.y;
      out[14] = swapped.x;
      out[15] = swapped.y;
      out[16] = splat.x;
      out[17] = splat.y;
      out[18] = splat2.x;
      out[19] = splat2.y;
      bits[0] = __bfloat162_as_uint(pair);
      bits[1] = __bfloat162_as_uint(fromVec);
      bits[2] = __bfloat162_as_uint(splat);
      bits[3] = __bfloat162_as_uint(splat2);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const input = { buffers: { out: new Float32Array(20), bits: new Uint32Array(4) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const expected = [256, -256, 256, -256, 256, -256, 256, 256, -256, -256, 256, 256, -256, -256, -256, 256, 1.5, 1.5, 256, 256];

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(semanticIr(compiled).requiredFeatures).not.toContain("shader-f16");
      expect(compiled.wgsl).toContain("bg_f32_to_bf16_bits_mode");
      expect([...result.buffers.out as Float32Array]).toEqual(expected);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual(expected);
      expect([...result.buffers.bits as Uint32Array]).toEqual([0xc3804380, 0xc3804380, 0x3fc03fc0, 0x43804380]);
      expect([...semanticResult.buffers.bits as Uint32Array]).toEqual([0xc3804380, 0xc3804380, 0x3fc03fc0, 0x43804380]);
    });

  it("lowers CUDA bf162 arithmetic aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void bf162Math(float *out, uint *bits) {
    if (threadIdx.x < 1) {
      __nv_bfloat162 a = __floats2bfloat162_rn(1.5f, -2.25f);
      __nv_bfloat162 b = __floats2bfloat162_rn(2.5f, 4.0f);
      __nv_bfloat162 c = __floats2bfloat162_rn(1.0f, -1.0f);
      __nv_bfloat162 sum = __hadd2(a, b);
      __nv_bfloat162 diff = __hsub2(b, a);
      __nv_bfloat162 prod = __hmul2(a, b);
      __nv_bfloat162 div = __h2div(__floats2bfloat162_rn(4.0f, 9.0f), __floats2bfloat162_rn(2.0f, 3.0f));
      __nv_bfloat162 neg = __hneg2(a);
      __nv_bfloat162 absVal = __habs2(a);
      __nv_bfloat162 fmaValue = __hfma2(a, b, c);
      __nv_bfloat162 sat = __hfma2_sat(__floats2bfloat162_rn(0.75f, 0.25f), __floats2bfloat162_rn(2.0f, 4.0f), __floats2bfloat162_rn(-1.0f, 0.25f));
      __nv_bfloat162 relu = __hfma2_relu(a, b, c);
      __nv_bfloat162 cmadd = __hcmadd(__floats2bfloat162_rn(1.0f, 2.0f), __floats2bfloat162_rn(3.0f, 4.0f), __floats2bfloat162_rn(5.0f, 6.0f));
      __nv_bfloat162 signedPair = __floats2bfloat162_rn(1.25f, -1.75f);
      __nv_bfloat162 positivePair = __floats2bfloat162_rn(4.0f, 16.0f);
      __nv_bfloat162 expPair = __floats2bfloat162_rn(0.0f, 1.0f);
      __nv_bfloat162 exp2Pair = __floats2bfloat162_rn(1.0f, 3.0f);
      __nv_bfloat162 logPair = __floats2bfloat162_rn(1.0f, 4.0f);
      __nv_bfloat162 log10Pair = __floats2bfloat162_rn(1.0f, 10.0f);
      __nv_bfloat162 rintPair = __floats2bfloat162_rn(2.5f, -1.5f);
      out[0] = sum.x;
      out[1] = sum.y;
      out[2] = diff.x;
      out[3] = diff.y;
      out[4] = prod.x;
      out[5] = prod.y;
      out[6] = div.x;
      out[7] = div.y;
      out[8] = neg.x;
      out[9] = neg.y;
      out[10] = absVal.x;
      out[11] = absVal.y;
      out[12] = fmaValue.x;
      out[13] = fmaValue.y;
      out[14] = sat.x;
      out[15] = sat.y;
      out[16] = relu.x;
      out[17] = relu.y;
      out[18] = cmadd.x;
      out[19] = cmadd.y;
      out[20] = h2ceil(signedPair).x;
      out[21] = h2ceil(signedPair).y;
      out[22] = h2floor(signedPair).x;
      out[23] = h2floor(signedPair).y;
      out[24] = h2rcp(positivePair).x;
      out[25] = h2rcp(positivePair).y;
      out[26] = h2rsqrt(positivePair).x;
      out[27] = h2rsqrt(positivePair).y;
      out[28] = h2sqrt(positivePair).x;
      out[29] = h2sqrt(positivePair).y;
      out[30] = h2trunc(signedPair).x;
      out[31] = h2trunc(signedPair).y;
      out[32] = h2exp(expPair).x;
      out[33] = h2exp(expPair).y;
      out[34] = h2exp2(exp2Pair).x;
      out[35] = h2exp2(exp2Pair).y;
      out[36] = h2exp10(expPair).x;
      out[37] = h2exp10(expPair).y;
      out[38] = h2log(logPair).x;
      out[39] = h2log(logPair).y;
      out[40] = h2log2(logPair).x;
      out[41] = h2log2(logPair).y;
      out[42] = h2log10(log10Pair).x;
      out[43] = h2log10(log10Pair).y;
      out[44] = h2sin(expPair).x;
      out[45] = h2sin(expPair).y;
      out[46] = h2cos(expPair).x;
      out[47] = h2cos(expPair).y;
      out[48] = h2tanh(expPair).x;
      out[49] = h2tanh(expPair).y;
      out[50] = h2tanh_approx(expPair).x;
      out[51] = h2tanh_approx(expPair).y;
      out[52] = h2rint(rintPair).x;
      out[53] = h2rint(rintPair).y;
      bits[0] = __bfloat162_as_uint(sum);
      bits[1] = __bfloat162_as_uint(fmaValue);
    }
  }`, { workgroupSize: [1, 1, 1] });
      const input = { buffers: { out: new Float32Array(54), bits: new Uint32Array(2) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);
      const expected = [4, 1.75, 1, 6.25, 3.75, -9, 2, 3, -1.5, 2.25, 1.5, 2.25, 4.75, -10, 0.5, 1, 4.75, 0, 0, 16, 2, -1, 1, -2, 0.25, 0.0625, 0.5, 0.25, 2, 4, 1, -1, 1, 2.71875, 2, 8, 1, 10, 0, 1.3828125, 0, 2, 0, 1, 0, 0.83984375, 1, 0.5390625, 0, 0.76171875, 0, 0.76171875, 2, -2];

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(semanticIr(compiled).requiredFeatures).not.toContain("shader-f16");
      expect(compiled.wgsl).toContain("bg_f32_to_bf16_bits_mode");
      expect([...result.buffers.out as Float32Array]).toEqual(expected);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual(expected);
      expect([...result.buffers.bits as Uint32Array]).toEqual([0x3fe04080, 0xc1204098]);
      expect([...semanticResult.buffers.bits as Uint32Array]).toEqual([0x3fe04080, 0xc1204098]);
    });

  it("lowers CUDA bf162 comparison and minmax aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void bf162Compare(const float *seed, float *out, uint *mask, uint *flags) {
    if (threadIdx.x < 1) {
      float nanValue = seed[0];
      __nv_bfloat162 x = __floats2bfloat162_rn(1.0f, 2.0f);
      __nv_bfloat162 y = __floats2bfloat162_rn(1.0f, 3.0f);
      __nv_bfloat162 nx = __floats2bfloat162_rn(nanValue, 2.0f);
      __nv_bfloat162 ny = __floats2bfloat162_rn(1.0f, nanValue);
      __nv_bfloat162 eq = __heq2(x, y);
      __nv_bfloat162 ne = __hne2(x, y);
      __nv_bfloat162 gt = __hgt2(y, x);
      __nv_bfloat162 ge = __hge2(x, y);
      __nv_bfloat162 lt = __hlt2(x, y);
      __nv_bfloat162 le = __hle2(x, y);
      __nv_bfloat162 unord = __hequ2(nx, ny);
      __nv_bfloat162 isnx = __hisnan2(nx);
      __nv_bfloat162 isny = __hisnan2(ny);
      __nv_bfloat162 mn = __hmin2(x, y);
      __nv_bfloat162 mx = __hmax2(x, y);
      __nv_bfloat162 nmn = __hmin2_nan(nx, ny);
      __nv_bfloat162 nmx = __hmax2_nan(nx, ny);
      __nv_bfloat162 nmnFlags = __hisnan2(nmn);
      __nv_bfloat162 nmxFlags = __hisnan2(nmx);
      out[0] = eq.x;
      out[1] = eq.y;
      out[2] = ne.x;
      out[3] = ne.y;
      out[4] = gt.x;
      out[5] = gt.y;
      out[6] = ge.x;
      out[7] = ge.y;
      out[8] = lt.x;
      out[9] = lt.y;
      out[10] = le.x;
      out[11] = le.y;
      out[12] = unord.x;
      out[13] = unord.y;
      out[14] = isnx.x;
      out[15] = isnx.y;
      out[16] = isny.x;
      out[17] = isny.y;
      out[18] = mn.x;
      out[19] = mn.y;
      out[20] = mx.x;
      out[21] = mx.y;
      out[22] = nmnFlags.x;
      out[23] = nmnFlags.y;
      out[24] = nmxFlags.x;
      out[25] = nmxFlags.y;
      mask[0] = __heq2_mask(x, y);
      mask[1] = __hne2_mask(x, y);
      mask[2] = __hgt2_mask(y, x);
      mask[3] = __hge2_mask(x, y);
      mask[4] = __hequ2_mask(nx, ny);
      mask[5] = __hgtu2_mask(nx, ny);
      if (__hbeq2(x, y)) { flags[0] = 1u; }
      if (__hbne2(x, y)) { flags[1] = 1u; }
      if (__hble2(x, y)) { flags[2] = 1u; }
      if (__hbequ2(nx, ny)) { flags[3] = 1u; }
      if (__hbgtu2(nx, ny)) { flags[4] = 1u; }
    }
  }`, { workgroupSize: [1, 1, 1] });
      const input = { buffers: { seed: new Float32Array([Number.NaN]), out: new Float32Array(26), mask: new Uint32Array(6), flags: new Uint32Array(5) } };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(semanticIr(compiled).requiredFeatures).not.toContain("shader-f16");
      expect([...result.buffers.out as Float32Array]).toEqual([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 2, 1, 3, 1, 1, 1, 1]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 2, 1, 3, 1, 1, 1, 1]);
      expect([...result.buffers.mask as Uint32Array]).toEqual([0x0000ffff, 0xffff0000, 0xffff0000, 0x0000ffff, 0xffffffff, 0xffffffff]);
      expect([...semanticResult.buffers.mask as Uint32Array]).toEqual([0x0000ffff, 0xffff0000, 0xffff0000, 0x0000ffff, 0xffffffff, 0xffffffff]);
      expect([...result.buffers.flags as Uint32Array]).toEqual([0, 0, 1, 1, 1]);
      expect([...semanticResult.buffers.flags as Uint32Array]).toEqual([0, 0, 1, 1, 1]);
    });

  it("lowers scalar CUDA half unary math aliases", () => {
      const compiled = compileCudaLiteKernel(`
  __global__ void half_unary(const half* input, half* output) {
    if (threadIdx.x < 1) {
      half negative = input[0];
      half two = input[1];
      half four = input[2];
      half fractional = input[3];
      output[0] = __habs(negative);
      output[1] = __hceil(fractional);
      output[2] = __hfloor(fractional);
      output[3] = __htrunc(__float2half(-1.75f));
      output[4] = __hrcp(two);
      output[5] = __hrsqrt(four);
      output[6] = __hsqrt(four);
    }
  }`, {
        features: { "shader-f16": true },
        workgroupSize: [1, 1, 1],
      });
      const input = {
        buffers: {
          input: createWgslFloat16Array([-1.5, 2, 4, 1.25]),
          output: createWgslFloat16Array(7),
        },
      };
      const launch = { gridDim: [1, 1, 1] as const, blockDim: [1, 1, 1] as const };
      const result = runCompiledKernelReference(compiled, input, launch);
      const semanticResult = runCompiledKernelSemanticReference(compiled, input, launch);

      expect(canRunCompiledKernelSemanticReference(compiled)).toBe(true);
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect(compiled.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("unsupported-call");
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(Array.from(result.buffers.output as Iterable<number>)).toEqual([1.5, 2, 1, -1, 0.5, 0.5, 2]);
      expect(Array.from(semanticResult.buffers.output as Iterable<number>)).toEqual([1.5, 2, 1, -1, 0.5, 0.5, 2]);
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
      expect(compiled.wgsl).toContain("value ^= 3");
      expect(compiled.wgsl).toContain("value |= 8");
      expect(compiled.wgsl).toContain("value &= 14");
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
      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(true);
      expect([...result.buffers.out as Float32Array]).toEqual([1, 1, 1, 1]);
      expect([...semanticResult.buffers.out as Float32Array]).toEqual([1, 1, 1, 1]);
      expect(compiled.wgsl).toContain("browsergrad-semantic-wgsl");
      expect(compiled.wgsl).toContain("var zero: vec4<f32> = vec4<f32>(f32(0.0), f32(0.0), f32(0.0), f32(0.0));");
      expect(compiled.wgsl).toContain("out[local_id.x] = (zero[local_id.x] + one[local_id.x]);");
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
