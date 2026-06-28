#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import {
  cudaLiteCorpusExecutionFixtureBaseline,
  cudaLiteCorpusExecutionFixtures,
} from "./cuda-lite-corpus-registry.mjs";
import {
  loadAutoCorpusSmokeFixtures,
  loadCorpusExecutionSources,
  verifyCorpusFixtureCheckouts,
} from "./cuda-lite-webgpu-fixtures.mjs";
import {
  markdownReport,
  summarizeReport,
  validateCorpusFixtureBaseline,
} from "./cuda-lite-webgpu-report.mjs";
import {
  findRepoRoot,
  moduleAliases,
  parseAutoCorpusSmokeFeatures,
  parseAutoCorpusSmokeMode,
  parseBundle,
  parseNonNegativeInteger,
} from "./cuda-lite-webgpu-cli.mjs";

const args = new Map();
for (let i = 2; i < process.argv.length; i++) {
  const key = process.argv[i];
  if (key === "--") continue;
  const value = process.argv[i + 1];
  if (!key?.startsWith("--")) continue;
  args.set(key, value?.startsWith("--") || value === undefined ? "true" : value);
  if (value && !value.startsWith("--")) i++;
}

function parseCaseFilters(argv) {
  const values = [];
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === "--case" || arg === "--cases") {
      const value = argv[index + 1];
      if (value && !value.startsWith("--")) {
        values.push(value);
        index++;
      }
      continue;
    }
    if (arg?.startsWith("--case=")) values.push(arg.slice("--case=".length));
    if (arg?.startsWith("--cases=")) values.push(arg.slice("--cases=".length));
  }
  return values.join(",")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

const headed = args.get("--headed") === "true";
const requireWebGpu = args.get("--require-webgpu") === "true";
const requireCorpusFixtures = args.get("--require-corpus-fixtures") === "true";
const markdownPath = args.get("--markdown");
const jsonPath = args.get("--json");
const summaryOnly = args.get("--summary-only") === "true";
const progress = args.get("--progress") === "true";
const progressPath = args.get("--progress-file");
const caseFilters = parseCaseFilters(process.argv.slice(2));
const caseTimeoutMs = parseNonNegativeInteger(args.get("--case-timeout-ms") ?? "0", "--case-timeout-ms");
const bundle = parseBundle(args.get("--bundle") ?? "src");
const autoCorpusSmokeLimit = parseNonNegativeInteger(args.get("--auto-corpus-smoke-limit") ?? "0", "--auto-corpus-smoke-limit");
const autoCorpusSmokeMode = parseAutoCorpusSmokeMode(args.get("--auto-corpus-smoke-mode") ?? "reference");
const autoCorpusSmokeFeatures = parseAutoCorpusSmokeFeatures(args.get("--auto-corpus-smoke-features") ?? "");

const root = findRepoRoot(process.cwd());
const packageRequire = createRequire(path.join(root, "packages/browsergrad-compiler/package.json"));
const { createServer } = await import(pathToFileURL(packageRequire.resolve("vite")).href);
const playwright = await import(pathToFileURL(packageRequire.resolve("playwright")).href);
const compilerNode = await import(pathToFileURL(path.join(root, "packages/browsergrad-compiler/dist/index.js")).href);
const chromium = playwright.chromium ?? playwright.default?.chromium;
if (!chromium) throw new Error("could not load Playwright chromium");
if (requireCorpusFixtures || autoCorpusSmokeLimit > 0) verifyCorpusFixtureCheckouts(root);

const examplesRoot = path.join(root, "packages/browsergrad-compiler/examples");
const sources = {
  saxpy: fs.readFileSync(path.join(examplesRoot, "saxpy.cu"), "utf8"),
  guardedMap: fs.readFileSync(path.join(examplesRoot, "guarded-map.cu"), "utf8"),
  tiledMatmul: fs.readFileSync(path.join(examplesRoot, "tiled-matmul.cu"), "utf8"),
  gridSync: `
namespace cg = cooperative_groups;
__global__ void gridSync(float *scratch, float *out, float scale) {
  cg::grid_group grid = cg::this_grid();
  scratch[blockIdx.x] = ((float)blockIdx.x + 1.0f) * scale;
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    out[0] = scratch[0] + scratch[1];
  }
}`,
  peerCopy: `
__global__ void peerCopy(float *dst, const float *src, int n) {
  if (threadIdx.x == 0) {
    cudaMemcpyPeerAsync(dst + 1, 1, src, 0, sizeof(float) * n, 0);
  }
}`,
  runtimeCopy: `
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
}`,
  dynamicLaunch: `
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
}`,
  dynamicSystemAtomicLaunch: `
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
}`,
  dynamicAliasAtomicLaunch: `
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
}`,
  dynamicConditionalAliasAtomicLaunch: `
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
}`,
  recursiveDynamicLaunch: `
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
}`,
  dynamicPoolLaunch: `
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
}`,
  dynamicPoolExpandedLaunch: `
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
}`,
  dynamicDeviceFunctionPoolLaunch: `
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
}`,
  devicePointerHelpers: `
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
}`,
  sharedPointerHelpers: `
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
}`,
  deviceGlobalStorage: `
__device__ unsigned int counter = 0;
__device__ float values[4];

__device__ void setValue(float* ptr, int index, float value) {
  ptr[index] = value;
}

__global__ void deviceGlobalStorage(float* out, uint* old) {
  int tid = threadIdx.x;
  old[tid] = atomicAdd(&counter, 1u);
  setValue(values, tid, (float)(tid + 1));
  out[tid] = values[tid];
}`,
  deviceGlobalAtomicRmw: `
__device__ int g_i[1];
__device__ float g_f[1];

__device__ void helperGlobalRmw(int *xi, float *xf, float *out) {
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

__global__ void deviceGlobalAtomicRmw(float *out) {
  if (threadIdx.x == 0) {
    helperGlobalRmw(g_i, g_f, out);
  }
}`,
  ptxTileCarrier: `
__global__ void ptxTileCarrier(uint *out) {
  uint addr = 5u;
  uint tile0 = 0u;
  uint tile1 = 0u;
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\\n"
    : "=r"(tile0), "=r"(tile1)
    : "r"(addr));
  uint a0 = 0x3c003c00u;
  uint a1 = 0x3c003c00u;
  uint a2 = 0x3c003c00u;
  uint a3 = 0x3c003c00u;
  uint b0 = 0x40004000u;
  uint b1 = 0x40004000u;
  uint c = tile0;
  uint d = tile1;
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, "
    "{%2, %3}, {%4, %5}, {%6, %7};\\n"
    : "=r"(c), "=r"(d)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c), "r"(d));
  out[0] = c;
  out[1] = d;
}`,
  ptxMmaF32Carrier: `
__global__ void ptxMmaF32Carrier(uint *out) {
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
  out[1] = __float_as_uint(__uint_as_float(d0));
}`,
  localHalfCarrier: `
__global__ void localHalfCarrier(uint *out) {
  uint regs[1][2];
  regs[0][0] = 0u;
  regs[0][1] = 0u;
  half *view = reinterpret_cast<half *>(&(regs[0][0]));
  view[0] = __float2half(1.0f);
  view[1] = __float2half(2.0f);
  view[2] = __float2half(3.0f);
  view[3] = __float2half(4.0f);
  float sum = __half2float(view[0]) + __half2float(view[1]) + __half2float(view[2]) + __half2float(view[3]);
  out[0] = regs[0][0];
  out[1] = regs[0][1];
  out[2] = __float_as_uint(sum);
}`,
  halfCompatF32Mode: `
__global__ void halfCompatF32Mode(float *out, half *x, half2 *y, half a) {
  if (threadIdx.x < 1) {
    half next = __float2half(__half2float(x[0]) + __half2float(a));
    half2 pair = __hadd2(y[0], __floats2half2_rn(1.0f, 2.0f));
    out[0] = __half2float(next);
    out[1] = __low2float(pair);
    out[2] = __high2float(pair);
  }
}`,
  subgroupScalarCompat: `
__global__ void subgroupScalarCompat(float *x) {
  int idx = threadIdx.x;
  float v = warp_reduce_sum_f32(x[idx]);
  if ((idx % 32) == 0) {
    v = bg_subgroup_add(v);
  }
  x[idx] = v;
}`,
  doubleCompatF32Mode: `
__device__ void addValue(double *result, double value) {
  atomicAdd(result, value);
}
__global__ void doubleCompatF32Mode(double* result, double* out, double a) {
  int idx = threadIdx.x;
  if (idx < 2) {
    addValue(result, a);
    out[idx] = a + (double)idx + 1.25;
  }
}`,
  boolPointerStorage: `
__global__ void boolPointerStorage(bool *flags, int *out) {
  int idx = threadIdx.x;
  bool active = flags[idx];
  bool *slot = flags + idx + 2;
  if (active) {
    out[idx] = 1;
    *slot = false;
  } else {
    out[idx] = 0;
    *slot = true;
  }
}`,
  aliasAtomicIncDec: `
__global__ void aliasAtomicIncDec(float *scratch, const float *values, uint *out) {
  if (threadIdx.x == 0) {
    float *accum = scratch;
    uint *flag = (uint *)(scratch + 2);
    out[0] = atomicInc(flag, 2);
    out[1] = atomicDec(flag, 2);
    atomicAdd(&accum[0], values[0]);
  }
}`,
  sharedAtomicIncDec: `
__global__ void sharedAtomicIncDec(uint *out) {
  __shared__ uint counter[1];
  if (threadIdx.x == 0) {
    counter[0] = 1;
    out[0] = atomicInc(&counter[0], 1);
    out[1] = atomicDec(&counter[0], 1);
  }
}`,
  helperAtomicIncDec: `
__device__ void helperIncDec(uint *counter, uint *out) {
  uint *ptr = counter;
  out[0] = atomicInc(ptr, 2);
  out[1] = atomicDec(ptr, 2);
  out[2] = ptr[0];
}
__device__ void helperIncDecOffset(uint *counter, uint *out, int offset) {
  uint *ptr = counter;
  out[offset + 0] = atomicInc(ptr, 2);
  out[offset + 1] = atomicDec(ptr, 2);
  out[offset + 2] = ptr[0];
}
__global__ void helperAtomicIncDec(uint *counter, uint *out) {
  __shared__ uint sharedCounter[1];
  if (threadIdx.x == 0) {
    helperIncDec(counter, out);
    sharedCounter[0] = 1;
    helperIncDecOffset(&sharedCounter[0], out, 3);
  }
}`,
  deviceGlobalAtomicIncDec: `
__device__ uint gCounter[1];

__device__ void helperGlobalIncDec(uint *counter, uint *out) {
  uint *ptr = counter;
  out[0] = atomicInc(ptr, 2);
  out[1] = atomicDec(ptr, 2);
  out[2] = ptr[0];
}

__global__ void deviceGlobalAtomicIncDec(uint *out) {
  if (threadIdx.x == 0) {
    helperGlobalIncDec(gCounter, out);
  }
}`,
  assignedPointerAtomic: `
__global__ void assignedPointerAtomic(uint *counter, uint *out) {
  uint *ptr = NULL;
  if (threadIdx.x == 0) {
    ptr = counter;
    out[0] = atomicAdd(ptr, 1u);
    out[1] = counter[0];
  }
}`,
  branchAssignedPointerAtomic: `
__global__ void branchAssignedPointerAtomic(uint *left, uint *right, uint *out, int pickRight) {
  uint *ptr = NULL;
  if (pickRight) {
    ptr = right;
  } else {
    ptr = left;
  }
  if (threadIdx.x == 0) {
    out[0] = atomicAdd(ptr, 1u);
  }
}`,
  conditionalPointerAtomic: `
__global__ void conditionalPointerAtomic(uint *left, uint *right, uint *out, int pickRight) {
  uint *ptr = pickRight ? right : left;
  if (threadIdx.x == 0) {
    out[0] = atomicAdd(ptr, 1u);
  }
}`,
  chainedAssignmentPointerAtomic: `
__global__ void chainedAssignmentPointerAtomic(uint *counter, uint *out) {
  uint *a = NULL;
  uint *b = NULL;
  if (threadIdx.x == 0) {
    a = b = counter;
    out[0] = atomicAdd(a, 1u);
    out[1] = b[0];
  }
}`,
  pointerArrayAtomic: `
__global__ void pointerArrayAtomic(uint *counter, uint *untouched, uint *out) {
  uint *ptrs[2];
  if (threadIdx.x == 0) {
    ptrs[0] = counter;
    ptrs[1] = untouched;
    out[0] = atomicAdd(ptrs[0], 1u);
    out[1] = counter[0];
    out[2] = untouched[0];
  }
}`,
  systemAtomicAliases: `
__global__ void systemAtomicAliases(int *x, int *out) {
  if (threadIdx.x == 0) {
    out[0] = atomicAdd_system(&x[0], 2);
    out[1] = atomicSub_system(&x[0], 1);
    out[2] = atomicMax_system(&x[0], 5);
    out[3] = atomicMin_system(&x[0], 3);
    out[4] = atomicAnd_system(&x[1], 0x6);
    out[5] = atomicOr_system(&x[1], 0x8);
    out[6] = atomicXor_system(&x[1], 0x3);
    out[7] = atomicInc_system((uint *)&x[2], 2);
    out[8] = atomicDec_system((uint *)&x[2], 2);
    out[9] = atomicExch_system(&x[3], 12);
    out[10] = atomicCAS_system(&x[3], 12, 11);
  }
}`,
  systemFloatAtomics: `
__global__ void systemFloatAtomics(float *x, float *out) {
  if (threadIdx.x == 0) {
    out[0] = atomicAdd_system(&x[0], 1.5f);
    out[1] = atomicSub_system(&x[0], 0.5f);
    out[2] = atomicMin_system(&x[0], 2.0f);
    out[3] = atomicMax_system(&x[0], 4.0f);
    out[4] = atomicExch_system(&x[0], 6.0f);
  }
}`,
  helperAtomicRmw: `
__device__ void helperRmw(int *xi, float *xf, float *out) {
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
__global__ void helperAtomicRmw(int *xi, float *xf, float *out) {
  if (threadIdx.x == 0) {
    helperRmw(xi, xf, out);
  }
}`,
  helperSharedAtomicRmw: `
__device__ void helperSharedRmw(int *xi, float *xf, float *out) {
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
__global__ void helperSharedAtomicRmw(float *out) {
  __shared__ int xi[1];
  __shared__ float xf[1];
  if (threadIdx.x == 0) {
    xi[0] = 10;
    xf[0] = 4.0f;
    helperSharedRmw(&xi[0], &xf[0], out);
  }
}`,
  helperAtomicExchangeCas: `
__device__ uint exchangeU32(uint *target, uint value) {
  return atomicExch(target, value);
}
__device__ uint casU32(uint *target, uint compare, uint value) {
  return atomicCAS(target, compare, value);
}
__global__ void helperAtomicExchangeCas(uint *storage, uint *out) {
  __shared__ uint flag;
  if (threadIdx.x == 0) {
    flag = 3u;
    out[0] = exchangeU32(&flag, 7u);
    out[1] = flag;
    out[2] = casU32(storage, 2u, 9u);
    out[3] = storage[0];
    out[4] = casU32(&flag, 7u, 11u);
    out[5] = flag;
  }
}`,
  surfaceRead: `
__global__ void surfaceRead(uint *out, cudaSurfaceObject_t surf) {
  uint value = 0;
  surf2Dread(&value, surf, 4, 0);
  out[0] = value;
  out[1] = surf2Dread<unsigned int>(surf, 0, 0);
}`,
  surfaceWrite3d: `
__global__ void surfaceWrite3d(cudaSurfaceObject_t outputSurf) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = blockIdx.z;
  surf3Dwrite(float(x + y * 10 + z * 100), outputSurf, x * sizeof(float), y, z);
}`,
  driverSurfaceAlias: `
__global__ void driverSurfaceAlias(CUsurfObject surf) {
  surf2Dwrite(13u, surf, 4, 0);
}`,
  textureFetchLod: `
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
__global__ void textureFetchLod(float4 *vecOut, float *scalarOut) {
  vecOut[0] = tex2DLod<float4>(texRef, 0.5f, 0.5f, 0.0f);
  scalarOut[0] = tex1Dfetch<float>(texRef, 1);
}`,
  textureAtlasHelpers: `
__global__ void textureAtlasHelpers(float4 *vecOut, float *scalarOut, cudaTextureObject_t tex) {
  scalarOut[0] = tex1D<float>(tex, 1.0f);
  scalarOut[1] = tex2DLayered<float>(tex, 0.0f, 1.0f, 1.0f);
  scalarOut[2] = tex3D<float>(tex, 2.0f, 1.0f, 1.0f);
  scalarOut[3] = texCubemap<float>(tex, 1.0f, 0.0f, 0.0f);
  vecOut[0] = tex1D<float4>(tex, 0.0f);
}`,
  textureUchar4: `
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
__global__ void textureUchar4(uint4 *out) {
  out[0] = tex2D<uchar4>(texRef, 0.5f, 0.5f);
}`,
  reciprocalIntrinsic: `
__global__ void reciprocalIntrinsic(float *x, float *out) {
  int idx = threadIdx.x;
  if (idx < 4) {
    out[idx] = __frcp_rn(x[idx] + 1.0f);
  }
}`,
  ...loadCorpusExecutionSources(root),
};
const autoCorpusSmokeFixtures = autoCorpusSmokeLimit > 0
  ? loadAutoCorpusSmokeFixtures(root, autoCorpusSmokeLimit, compilerNode, {
      verifyMode: autoCorpusSmokeMode,
      allowedRequiredFeatures: autoCorpusSmokeFeatures,
    })
  : [];
for (const fixture of autoCorpusSmokeFixtures) {
  sources[fixture.sourceKey] = fixture.source;
}
const expectedCorpusFixtureNames = cudaLiteCorpusExecutionFixtures.map((fixture) => fixture.caseName);

const html = String.raw`<!doctype html>
<html>
  <head><meta charset="utf-8"><title>BrowserGrad CUDA-lite WebGPU e2e</title></head>
  <body>
    <script type="module">
      import {
        createDevice,
        createWgslFloat16Array,
        createWgslStorageBuffer,
        destroyWgslStorageBuffer,
        readWgslStorageBuffer,
      } from "@unlocalhosted/browsergrad-kernels";
      import {
        compileCudaLiteKernelForWebGpu,
        compileCudaLiteKernel,
        createCudaWebGpuExecutionPlan,
        prepareCompiledKernelWebGpu,
        runCompiledKernelReference,
        runCompiledKernelWebGpu,
      } from "@unlocalhosted/browsergrad-compiler";

      const SOURCES = ${JSON.stringify(sources)};
      const CORPUS_FIXTURES = ${JSON.stringify(cudaLiteCorpusExecutionFixtures)};
      const AUTO_CORPUS_SMOKE_FIXTURES = ${JSON.stringify(autoCorpusSmokeFixtures.map(({ source, ...fixture }) => fixture))};
      const EXPECTED_CORPUS_FIXTURE_NAMES = ${JSON.stringify(expectedCorpusFixtureNames)};
      const CORPUS_FIXTURE_BASELINE = ${JSON.stringify(cudaLiteCorpusExecutionFixtureBaseline)};
      const CASE_FILTERS = ${JSON.stringify(caseFilters)};
      const CASE_TIMEOUT_MS = ${caseTimeoutMs};
      const PROGRESS = ${JSON.stringify(progress)};

      window.__bgRunE2e = async () => {
        if (!navigator.gpu) {
          return {
            available: false,
            reason: "navigator.gpu undefined",
            cases: [],
          };
        }
        let device = await createE2eKernelDevice();
        const cases = [];

        const specs = filterCaseSpecs(caseSpecs());
        for (let index = 0; index < specs.length; index++) {
          const spec = specs[index];
          const missingFeatures = missingSpecFeatures(device, spec);
          if (missingFeatures.length > 0) {
            cases.push({
              name: spec.name,
              plan: "missing-features:" + missingFeatures.join(","),
              ok: false,
              maxAbsDiff: Number.POSITIVE_INFINITY,
              output: spec.output,
              corpusId: spec.corpusId,
            });
            emitProgress({ event: "case-fail", index: index + 1, total: specs.length, name: spec.name, missingFeatures });
            continue;
          }
          emitProgress({ event: "case-start", index: index + 1, total: specs.length, name: spec.name });
          const start = performance.now();
          try {
            const result = await withCaseTimeout(runReferenceWebGpuCase(device, spec), spec.name);
            cases.push({
              name: spec.name,
              plan: result.plan,
              ok: result.ok,
              ...(result.skipped ? { skipped: true } : {}),
              maxAbsDiff: result.maxAbsDiff,
              ...(spec.tolerance === undefined ? {} : { tolerance: spec.tolerance }),
              ...(result.firstDiff === undefined ? {} : { firstDiff: result.firstDiff }),
              output: result.output ?? spec.output,
              ...(spec.corpusId === undefined ? {} : { corpusId: spec.corpusId }),
              ...(spec.expectedOutput === undefined ? {} : { expectedOutputPinned: true }),
              ms: round(performance.now() - start),
            });
            emitProgress({ event: "case-pass", index: index + 1, total: specs.length, name: spec.name, ms: cases[cases.length - 1].ms });
          } catch (error) {
            cases.push({
              name: spec.name,
              plan: "threw",
              ok: false,
              maxAbsDiff: Number.POSITIVE_INFINITY,
              output: spec.output,
              error: serializeError(error),
              ...(spec.corpusId === undefined ? {} : { corpusId: spec.corpusId }),
              ms: round(performance.now() - start),
            });
            emitProgress({ event: "case-fail", index: index + 1, total: specs.length, name: spec.name, ms: cases[cases.length - 1].ms, error: serializeError(error) });
            if (isCaseTimeout(error)) {
              device.gpu.destroy();
              device = await createE2eKernelDevice();
            }
          }
        }

        if (shouldRunPreparedSmoke()) {
          emitProgress({ event: "case-start", index: specs.length + 1, total: specs.length + 1, name: "prepared-resident-saxpy" });
          const preparedStart = performance.now();
          cases.push({
            name: "prepared-resident-saxpy",
            plan: "prepared:single-dispatch",
            ...(await runPreparedResidentSaxpy(device)),
            output: "y",
            ms: round(performance.now() - preparedStart),
          });
          emitProgress({ event: "case-pass", index: specs.length + 1, total: specs.length + 1, name: "prepared-resident-saxpy", ms: cases[cases.length - 1].ms });
        }

        const failed = cases.filter((item) => !item.ok);
        const skipped = cases.filter((item) => item.skipped);
        const corpusFixtureCases = cases.filter((item) => item.name.startsWith("corpus:"));
        const corpusFixturePassed = corpusFixtureCases.filter((item) => item.ok);
        const corpusFixtureExpectedOutputCases = corpusFixtureCases.filter((item) => item.expectedOutputPinned);
        const autoCorpusSmokeCases = cases.filter((item) => item.name.startsWith("auto-corpus:"));
        const autoCorpusSmokePassed = autoCorpusSmokeCases.filter((item) => item.ok && !item.skipped);
        const autoCorpusSmokeSkipped = autoCorpusSmokeCases.filter((item) => item.skipped);
        const autoCorpusSmokeCovered = autoCorpusSmokePassed.length + autoCorpusSmokeSkipped.length;
        const loadedCorpusFixtureNames = new Set(corpusFixtureCases.map((item) => item.name));
        const expectedCorpusFixtureNames = filterExpectedCorpusFixtureNames();
        return {
          available: true,
          caseFilters: CASE_FILTERS,
          cases,
          corpusFixtureCases: corpusFixtureCases.length,
          corpusFixturePassed: corpusFixturePassed.length,
          corpusFixtureFailed: corpusFixtureCases.filter((item) => !item.ok).length,
          corpusFixtureExpectedOutputCases: corpusFixtureExpectedOutputCases.length,
          corpusFixtureCasesByCorpus: countByCorpus(corpusFixtureCases),
          corpusFixturePassedByCorpus: countByCorpus(corpusFixturePassed),
          autoCorpusSmokeCases: autoCorpusSmokeCases.length,
          autoCorpusSmokePassed: autoCorpusSmokePassed.length,
          autoCorpusSmokeSkipped: autoCorpusSmokeSkipped.length,
          autoCorpusSmokeCovered,
          autoCorpusSmokeFailed: autoCorpusSmokeCases.filter((item) => !item.ok).length,
          autoCorpusSmokeCasesByCorpus: countByCorpus(autoCorpusSmokeCases),
          autoCorpusSmokePassedByCorpus: countByCorpus(autoCorpusSmokePassed),
          autoCorpusSmokeSkippedByCorpus: countByCorpus(autoCorpusSmokeSkipped),
          autoCorpusSmokeLimit: ${autoCorpusSmokeLimit},
          autoCorpusSmokeMode: ${JSON.stringify(autoCorpusSmokeMode)},
          corpusFixtureBaseline: CORPUS_FIXTURE_BASELINE,
          expectedCorpusFixtureNames,
          missingCorpusFixtureNames: expectedCorpusFixtureNames.filter((name) => !loadedCorpusFixtureNames.has(name)),
          passed: cases.length - failed.length - skipped.length,
          failed: failed.length,
          skipped: skipped.length,
        };
      };

      async function createE2eKernelDevice() {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter) throw new Error("Failed to obtain a WebGPU adapter");
        const requiredFeatures = ["shader-f16", "subgroups"].filter((feature) => adapter.features?.has?.(feature));
        const requiredLimits = {};
        const requestLimit = (name, baseline, desired) => {
          const supported = adapter.limits?.[name];
          if (typeof supported !== "number" || supported <= baseline) return;
          requiredLimits[name] = Math.min(supported, desired);
        };
        requestLimit("maxStorageBuffersPerShaderStage", 8, 16);
        const descriptor = {};
        if (requiredFeatures.length > 0) descriptor.requiredFeatures = requiredFeatures;
        if (Object.keys(requiredLimits).length > 0) descriptor.requiredLimits = requiredLimits;
        const gpuDevice = await adapter.requestDevice(descriptor);
        return createDevice({ device: gpuDevice });
      }

      function withCaseTimeout(promise, name) {
        if (CASE_TIMEOUT_MS <= 0) return promise;
        let timeoutId;
        const timeout = new Promise((_, reject) => {
          timeoutId = setTimeout(() => reject(new Error("case timeout after " + CASE_TIMEOUT_MS + "ms: " + name)), CASE_TIMEOUT_MS);
        });
        return Promise.race([promise, timeout]).finally(() => clearTimeout(timeoutId));
      }

      function isCaseTimeout(error) {
        return String(error?.message ?? error).startsWith("case timeout after ");
      }

      function missingSpecFeatures(device, spec) {
        const features = spec.requiredFeatures ?? [];
        return features.filter((feature) => !device.gpu.features?.has?.(feature));
      }

      function emitProgress(event) {
        if (!PROGRESS) return;
        console.log("__BG_PROGRESS__" + JSON.stringify(event));
      }

      function filterCaseSpecs(specs) {
        if (CASE_FILTERS.length === 0) return specs;
        const filtered = specs.filter((spec) => CASE_FILTERS.some((filter) => spec.name === filter || spec.name.includes(filter)));
        if (filtered.length === 0 && !shouldRunPreparedSmoke()) {
          throw new Error("no CUDA-lite WebGPU e2e cases matched: " + CASE_FILTERS.join(","));
        }
        return filtered;
      }

      function filterExpectedCorpusFixtureNames() {
        if (CASE_FILTERS.length === 0) return EXPECTED_CORPUS_FIXTURE_NAMES;
        return EXPECTED_CORPUS_FIXTURE_NAMES.filter((name) => CASE_FILTERS.some((filter) => name === filter || name.includes(filter)));
      }

      function shouldRunPreparedSmoke() {
        return CASE_FILTERS.length === 0 ||
          CASE_FILTERS.some((filter) => "prepared-resident-saxpy" === filter || "prepared-resident-saxpy".includes(filter));
      }

      function caseSpecs() {
        const cases = [
          {
            name: "example:saxpy",
            source: SOURCES.saxpy,
            options: { workgroupSize: [8, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4]),
                y: new Float32Array([10, 20, 30, 40]),
              },
              scalars: { a: 2, n: 4 },
            }),
            output: "y",
          },
          {
            name: "example:guarded-map",
            source: SOURCES.guardedMap,
            options: { workgroupSize: [8, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
            input: () => ({
              buffers: {
                input: new Float32Array([-1, 2, -3, 4]),
                output: new Float32Array(4),
              },
              scalars: { n: 4 },
            }),
            output: "output",
          },
          {
            name: "example:tiled-matmul",
            source: SOURCES.tiledMatmul,
            options: { workgroupSize: [2, 2, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 2, 1] },
            input: () => ({
              buffers: {
                A: new Float32Array([1, 2, 3, 4]),
                B: new Float32Array([5, 6, 7, 8]),
                C: new Float32Array(4),
              },
              scalars: { N: 2 },
            }),
            output: "C",
          },
          {
            name: "runtime:grid-sync-phases",
            source: SOURCES.gridSync,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [2, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                scratch: new Float32Array(2),
                out: new Float32Array(1),
              },
              scalars: { scale: 2 },
            }),
            output: "out",
          },
          {
            name: "runtime:host-peer-copy",
            source: SOURCES.peerCopy,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                dst: new Float32Array([0, 0, 0, 0]),
                src: new Float32Array([2.5, 3.5]),
              },
              scalars: { n: 2 },
            }),
            output: "dst",
          },
          {
            name: "runtime:host-copy",
            source: SOURCES.runtimeCopy,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                dst: new Float32Array([0, 0, 0, 0]),
                src: new Float32Array([2.5, 3.5]),
              },
              scalars: { n: 2 },
            }),
            output: "dst",
          },
          {
            name: "runtime:host-dynamic-launch",
            source: SOURCES.dynamicLaunch,
            options: { kernelName: "parent", workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2]),
              },
              scalars: { n: 2 },
            }),
            output: "x",
          },
          {
            name: "runtime:host-dynamic-system-atomics",
            source: SOURCES.dynamicSystemAtomicLaunch,
            options: { kernelName: "parent", workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array([9, 2, 3]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [8, 7, 5] },
          },
          {
            name: "runtime:host-dynamic-alias-atomic",
            source: SOURCES.dynamicAliasAtomicLaunch,
            options: { kernelName: "parent", workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array([4]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [5] },
          },
          {
            name: "runtime:host-dynamic-conditional-alias-atomic",
            source: SOURCES.dynamicConditionalAliasAtomicLaunch,
            options: { kernelName: "parent", workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array([4]),
              },
              scalars: { use_out: 1 },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [5] },
          },
          {
            name: "runtime:recursive-host-dynamic-launch",
            source: SOURCES.recursiveDynamicLaunch,
            options: { kernelName: "parent", workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array([0]),
              },
              scalars: { n: 2 },
            }),
            output: "out",
          },
          {
            name: "runtime:pool-pointer-host-dynamic-launch",
            source: SOURCES.dynamicPoolLaunch,
            options: { kernelName: "parent", workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {},
              memoryPools: {
                pool: { data: new Uint32Array(4), offset: new Uint32Array([0]) },
              },
              scalars: { n: 2 },
              readback: ["pool", "pool_offset"],
            }),
            output: "pool",
            offsetOutput: "pool_offset",
            expectedOffset: 8,
          },
          {
            name: "runtime:expanded-pool-pointer-host-dynamic-launch",
            source: SOURCES.dynamicPoolExpandedLaunch,
            options: { kernelName: "parent", workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {},
              memoryPools: {
                pool: { data: new Uint32Array(8), offset: new Uint32Array([0]) },
              },
              scalars: { n: 2 },
              readback: ["pool", "pool_offset"],
            }),
            output: "pool",
            offsetOutput: "pool_offset",
            expectedOffset: 32,
          },
          {
            name: "runtime:launched-device-function-pool-dynamic-launch",
            source: SOURCES.dynamicDeviceFunctionPoolLaunch,
            options: { kernelName: "parentKernel", workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {},
              memoryPools: {
                g_pool: { data: new Uint32Array(4), offset: new Uint32Array([0]) },
              },
              scalars: { N: 2 },
              readback: ["g_pool", "g_pool_offset"],
            }),
            output: "g_pool",
            offsetOutput: "g_pool_offset",
            expectedOffset: 16,
          },
          {
            name: "device-function:pointer-param-helpers",
            source: SOURCES.devicePointerHelpers,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4]),
                y: new Float32Array([10, 20, 30]),
              },
              scalars: { a: 2, n: 3 },
            }),
            output: "y",
          },
          {
            name: "device-function:shared-pointer-helpers",
            source: SOURCES.sharedPointerHelpers,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
            }),
            output: "out",
          },
          {
            name: "device-function:device-global-storage",
            source: SOURCES.deviceGlobalStorage,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
                old: new Uint32Array(4),
              },
            }),
            output: "out",
          },
          {
            name: "device-function:device-global-helper-rmw",
            source: SOURCES.deviceGlobalAtomicRmw,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(11),
              },
              deviceGlobals: {
                g_i: new Int32Array([10]),
                g_f: new Float32Array([4]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [10, 8, 5, 9, 0, 10, 4, 2.5, 2, 9, 5] },
          },
          {
            name: "ptx:tile-carrier",
            source: SOURCES.ptxTileCarrier,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(2),
              },
            }),
            output: "out",
          },
          {
            name: "ptx:mma-f32-carrier",
            source: SOURCES.ptxMmaF32Carrier,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(2),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [0x40b00000, 0x40b00000] },
          },
          {
            name: "ptx:local-half-carrier",
            source: SOURCES.localHalfCarrier,
            options: { workgroupSize: [1, 1, 1], features: { "shader-f16": true } },
            requiredFeatures: ["shader-f16"],
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(3),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [0x40003c00, 0x44004200, 0x41200000] },
          },
          {
            name: "compat:half-f32-mode-scalar-uniform",
            source: SOURCES.halfCompatF32Mode,
            options: { workgroupSize: [1, 1, 1], f16Mode: "f32" },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(3),
                x: new Float32Array([1.5]),
                y: new Float32Array([3, 5]),
              },
              scalars: { a: 2 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [3.5, 4, 7] },
          },
          {
            name: "compat:subgroup-scalar-mode",
            source: SOURCES.subgroupScalarCompat,
            options: { workgroupSize: [4, 1, 1], subgroupMode: "scalar" },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4]),
              },
            }),
            output: "x",
            expectedOutput: { type: "Float32Array", data: [1, 2, 3, 4] },
          },
          {
            name: "compat:double-f32-mode-atomic",
            source: SOURCES.doubleCompatF32Mode,
            options: { workgroupSize: [2, 1, 1], f64Mode: "f32" },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                result: new Float32Array([0]),
                out: new Float32Array(2),
              },
              scalars: { a: 1.5 },
            }),
            output: "result",
            expectedOutput: { type: "Float32Array", data: [3] },
          },
          {
            name: "storage:bool-pointer-alias",
            source: SOURCES.boolPointerStorage,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                flags: new Uint32Array([1, 0, 1, 1]),
                out: new Int32Array(2),
              },
            }),
            output: "flags",
            expectedOutput: { type: "Uint32Array", data: [1, 0, 0, 1] },
          },
          {
            name: "atomic:alias-inc-dec",
            source: SOURCES.aliasAtomicIncDec,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                scratch: new Float32Array([10, 0, 1]),
                values: new Float32Array([1.5]),
                out: new Uint32Array(2),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [1, 2] },
          },
          {
            name: "atomic:shared-inc-dec",
            source: SOURCES.sharedAtomicIncDec,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(2),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [1, 0] },
          },
          {
            name: "atomic:helper-inc-dec",
            source: SOURCES.helperAtomicIncDec,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                counter: new Uint32Array([1]),
                out: new Uint32Array(6),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [1, 2, 1, 1, 2, 1] },
          },
          {
            name: "atomic:device-global-helper-inc-dec",
            source: SOURCES.deviceGlobalAtomicIncDec,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(3),
              },
              deviceGlobals: {
                gCounter: new Uint32Array([1]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [1, 2, 1] },
          },
          {
            name: "atomic:assigned-pointer-rebind",
            source: SOURCES.assignedPointerAtomic,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                counter: new Uint32Array([4]),
                out: new Uint32Array(2),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [4, 5] },
          },
          {
            name: "atomic:branch-assigned-pointer-rebind",
            source: SOURCES.branchAssignedPointerAtomic,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                left: new Uint32Array([4]),
                right: new Uint32Array([8]),
                out: new Uint32Array(1),
              },
              scalars: { pickRight: 1 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [8] },
          },
          {
            name: "atomic:conditional-pointer-rebind",
            source: SOURCES.conditionalPointerAtomic,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                left: new Uint32Array([4]),
                right: new Uint32Array([8]),
                out: new Uint32Array(1),
              },
              scalars: { pickRight: 0 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [4] },
          },
          {
            name: "atomic:chained-assignment-pointer-rebind",
            source: SOURCES.chainedAssignmentPointerAtomic,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                counter: new Uint32Array([4]),
                out: new Uint32Array(2),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [4, 5] },
          },
          {
            name: "atomic:pointer-array-rebind",
            source: SOURCES.pointerArrayAtomic,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                counter: new Uint32Array([4]),
                untouched: new Uint32Array([8]),
                out: new Uint32Array(3),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [4, 5, 8] },
          },
          {
            name: "atomic:system-aliases",
            source: SOURCES.systemAtomicAliases,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                x: new Int32Array([4, 7, 1, 9]),
                out: new Int32Array(11),
              },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [4, 6, 5, 5, 7, 6, 14, 1, 2, 9, 12] },
          },
          {
            name: "atomic:system-float",
            source: SOURCES.systemFloatAtomics,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([2]),
                out: new Float32Array(5),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [2, 3.5, 3, 2, 4] },
          },
          {
            name: "atomic:helper-rmw",
            source: SOURCES.helperAtomicRmw,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                xi: new Int32Array([10]),
                xf: new Float32Array([4]),
                out: new Float32Array(11),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [10, 8, 5, 9, 0, 10, 4, 2.5, 2, 9, 5] },
          },
          {
            name: "atomic:helper-shared-rmw",
            source: SOURCES.helperSharedAtomicRmw,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(11),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [10, 8, 5, 9, 0, 10, 4, 2.5, 2, 9, 5] },
          },
          {
            name: "atomic:helper-exchange-cas",
            source: SOURCES.helperAtomicExchangeCas,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                storage: new Uint32Array([2]),
                out: new Uint32Array(6),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [3, 7, 2, 9, 7, 11] },
          },
          {
            name: "surface:surf2d-read",
            source: SOURCES.surfaceRead,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(2),
              },
              surfaces: {
                surf: { width: 2, height: 1, data: new Float32Array([3, 9]) },
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [9, 3] },
          },
          {
            name: "surface:surf3d-write",
            source: SOURCES.surfaceWrite3d,
            options: { workgroupSize: [2, 2, 1] },
            launch: { gridDim: [1, 1, 2], blockDim: [2, 2, 1] },
            input: () => ({
              buffers: {},
              surfaces: {
                outputSurf: { width: 2, height: 2, data: new Float32Array(8) },
              },
            }),
            output: "outputSurf",
            expectedOutput: { type: "Float32Array", data: [0, 1, 10, 11, 100, 101, 110, 111] },
          },
          {
            name: "surface:driver-alias",
            source: SOURCES.driverSurfaceAlias,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {},
              surfaces: {
                surf: { width: 2, height: 1, data: new Float32Array([3, 9]) },
              },
            }),
            output: "surf",
            expectedOutput: { type: "Float32Array", data: [3, 13] },
          },
          {
            name: "texture:fetch-lod-vector",
            source: SOURCES.textureFetchLod,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                vecOut: new Float32Array(4),
                scalarOut: new Float32Array(1),
              },
              textures: {
                texRef: {
                  width: 2,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
                },
              },
            }),
            output: "vecOut",
            expectedOutput: { type: "Float32Array", data: [1, 2, 3, 4] },
          },
          {
            name: "texture:atlas-helpers",
            source: SOURCES.textureAtlasHelpers,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
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
            }),
            output: "scalarOut",
            expectedOutput: { type: "Float32Array", data: [5, 33, 41, 21] },
          },
          {
            name: "texture:uchar4-read",
            source: SOURCES.textureUchar4,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(4),
              },
              textures: {
                texRef: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([1, 2, 3, 255]),
                },
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [1, 2, 3, 255] },
          },
          {
            name: "intrinsic:reciprocal",
            source: SOURCES.reciprocalIntrinsic,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 3, 7, 15]),
                out: new Float32Array(4),
              },
            }),
            output: "out",
          },
        ];
        for (const fixture of CORPUS_FIXTURES) {
          const source = SOURCES[fixture.sourceKey];
          if (!source) continue;
          cases.push({
            name: fixture.caseName,
            source,
            options: { ...(fixture.options ?? {}), kernelName: fixture.kernelName, workgroupSize: fixture.workgroupSize },
            launch: fixture.launch,
            input: () => materializeFixtureInput(fixture.input),
            output: fixture.output,
            corpusId: fixture.corpusId,
            ...(fixture.expectedOutput === undefined ? {} : { expectedOutput: fixture.expectedOutput }),
            ...(fixture.tolerance === undefined ? {} : { tolerance: fixture.tolerance }),
          });
        }
        for (const fixture of AUTO_CORPUS_SMOKE_FIXTURES) {
          const source = SOURCES[fixture.sourceKey];
          if (!source) continue;
          cases.push({
            name: fixture.caseName,
            source,
            options: { ...(fixture.options ?? {}), kernelName: fixture.kernelName, workgroupSize: fixture.workgroupSize },
            launch: fixture.launch,
            input: (compiled) => syntheticInputForCompiled(compiled),
            corpusId: fixture.corpusId,
            verifyMode: fixture.verifyMode,
            requiredFeatures: fixture.requiredFeatures,
          });
        }
        return cases;
      }

      function countByCorpus(items) {
        const out = {};
        for (const item of items) {
          const corpusId = item.corpusId ?? "unknown";
          out[corpusId] = (out[corpusId] ?? 0) + 1;
        }
        return out;
      }

      function serializeError(error) {
        return {
          name: error?.name ?? "Error",
          message: error?.message ?? String(error),
          stack: error?.stack,
        };
      }

      function materializeFixtureInput(input) {
        const buffers = {};
        for (const [name, spec] of Object.entries(input.buffers ?? {})) {
          buffers[name] = materializeTypedArray(spec);
        }
        const textures = {};
        for (const [name, spec] of Object.entries(input.textures ?? {})) {
          textures[name] = {
            width: spec.width,
            height: spec.height,
            channels: spec.channels,
            data: materializeTypedArray(spec.data),
          };
        }
        return {
          buffers,
          scalars: { ...(input.scalars ?? {}) },
          ...(Object.keys(textures).length === 0 ? {} : { textures }),
        };
      }

      function materializeTypedArray(spec) {
        const Ctor = globalThis[spec.type];
        if (typeof Ctor !== "function") throw new Error("unsupported fixture typed array: " + spec.type);
        if (spec.data !== undefined) return new Ctor(spec.data);
        return new Ctor(spec.length ?? 0);
      }

      function syntheticInputForCompiled(compiled) {
        const scalars = {};
        const buffers = {};
        const constants = {};
        const deviceGlobals = {};
        const memoryPools = {};
        const textures = {};
        const surfaces = {};
        for (const param of compiled.ir.params) {
          if (param.valueType === "surface2d") {
            surfaces[param.name] = { width: 64, height: 64, data: new Float32Array(64 * 64) };
          } else if (param.valueType === "texture2d") {
            textures[param.name] = { width: 64, height: 64, data: new Float32Array(64 * 64) };
          } else if (param.pointer) {
            if (param.valueType === "devicepool") {
              memoryPools[param.name] = { data: new Uint32Array(4096), offset: new Uint32Array([0]) };
            } else {
              buffers[param.name] = syntheticBufferForType(param.valueType, 4096, compiled.f16Mode);
            }
          } else {
            scalars[param.name] = syntheticScalarForName(param.name);
          }
        }
        for (const constant of compiled.ir.constants) {
          constants[constant.name] = constant.dimensions.length === 0 && !isCudaVectorTypeName(constant.valueType)
            ? syntheticScalarForName(constant.name)
            : syntheticBufferForType(constant.valueType, 4096, compiled.f16Mode);
        }
        for (const global of compiled.ir.deviceGlobals) {
          const length = global.dimensions.length === 0
            ? 1
            : global.dimensions.reduce((product, dimension) => product * dimension, 1);
          deviceGlobals[global.name] = syntheticBufferForType(global.valueType, length, compiled.f16Mode);
        }
        for (const texture of compiled.ir.textures) {
          textures[texture.name] = { width: 64, height: 64, data: new Float32Array(64 * 64) };
        }
        for (const poolName of externalDevicePoolNamesFromSource(compiled.ast.source)) {
          memoryPools[poolName] ??= { data: new Uint32Array(4096), offset: new Uint32Array([0]) };
        }
        return { buffers, scalars, constants, deviceGlobals, memoryPools, textures, surfaces };
      }

      function syntheticBufferForType(type, length = 4096, f16Mode = "native") {
        if (type === "int" || /^int[234]$/u.test(type)) return new Int32Array(length);
        if (type === "uint" || /^uint[234]$/u.test(type) || type === "voidptr" || type === "bool") return new Uint32Array(length);
        if ((type === "half" || type === "half2") && f16Mode !== "f32") return createWgslFloat16Array(length);
        return new Float32Array(length);
      }

      function isCudaVectorTypeName(type) {
        return /^(?:float|int|uint)[234]$|^half2$|^bf162$/u.test(type);
      }

      function syntheticScalarForName(name) {
        if (/(?:clock|delay|sleep|spin|wait)/iu.test(name)) return 0;
        if (/^(?:depth|level)$/iu.test(name)) return 0;
        if (/^(?:maxDepth|max_depth|maxLevel|max_level)$/u.test(name)) return 4;
        if (/^(?:left|begin|start|offset)$/u.test(name)) return 0;
        if (/^(?:right|end|len|nLines|nTessPoints)$/u.test(name)) return 64;
        if (/^(?:n|N|num|count|length|totalLen|frontierSize|numSamples|totalThreads|poolSize|size)$/u.test(name)) return 1024;
        if (/^(?:threads|threadsPerBlock|blockSize)$/u.test(name)) return 256;
        if (/^(?:blocks|blocksPerGrid|numBlocks)$/u.test(name)) return 4;
        return 1;
      }

      function externalDevicePoolNamesFromSource(source) {
        return [...source.matchAll(/\b(?:deviceAllocate|streamOrderedAllocate)\s*\(\s*&\s*([A-Za-z_][A-Za-z0-9_]*)/g)]
          .map((match) => match[1])
          .filter(Boolean);
      }

      async function runReferenceWebGpuCase(device, spec) {
        const compiled = compileCudaLiteKernelForWebGpu(spec.source, spec.options);
        const diagnosticSkip = webGpuDiagnosticSkip(compiled);
        if (diagnosticSkip) {
          return {
            ok: false,
            plan: diagnosticSkip,
            maxAbsDiff: Number.POSITIVE_INFINITY,
            output: spec.verifyMode === "dispatch" ? "dispatch" : spec.output,
          };
        }
        const actualInput = spec.input(compiled);
        const plan = createCudaWebGpuExecutionPlan(compiled, actualInput, spec.launch, {
          compileKernel: (childSource, childOptions = {}) => compileCudaLiteKernelForWebGpu(childSource, {
            ...childOptions,
            f16Mode: childOptions.f16Mode ?? compiled.f16Mode,
          }),
        });
        if (!plan.supported) {
          return { ok: false, plan: plan.reason, maxAbsDiff: Number.POSITIVE_INFINITY };
        }
        const resourceLimitBlocker = webGpuResourceLimitBlocker(device.gpu.limits, plan);
        if (resourceLimitBlocker) {
          return {
            ok: false,
            plan: "resource-limits:" + resourceLimitBlocker,
            maxAbsDiff: Number.POSITIVE_INFINITY,
            output: spec.verifyMode === "dispatch" ? "dispatch" : spec.output,
          };
        }
        if (spec.verifyMode === "dispatch") {
          await runCompiledKernelWebGpu(device, compiled, actualInput, spec.launch);
          return { ok: true, plan: plan.kind, maxAbsDiff: 0, output: "dispatch" };
        }
        const actual = await runCompiledKernelWebGpu(device, compiled, actualInput, spec.launch);
        const outputNames = spec.output === undefined ? Object.keys(actual.buffers) : [spec.output];
        if (spec.expectedOutput !== undefined) {
          const comparison = compareOutputs({ buffers: {} }, actual, outputNames, spec);
          if (comparison.ok && spec.offsetOutput) {
            const offset = actual.buffers[spec.offsetOutput]?.[0];
            if (offset !== spec.expectedOffset) {
              return { ok: false, plan: plan.kind, maxAbsDiff: Math.abs(Number(offset ?? NaN) - spec.expectedOffset) };
            }
          }
          return { ...comparison, plan: plan.kind, output: outputNames.join(",") };
        }
        const expectedInput = spec.offsetOutput && actualInput.readback
          ? { ...actualInput, readback: actualInput.readback.filter((name) => name !== spec.offsetOutput) }
          : actualInput;
        const expected = runCompiledKernelReference(compiled, expectedInput, spec.launch);
        const referenceComparison = compareOutputs(expected, expected, outputNames, spec);
        if (!referenceComparison.ok) return { ...referenceComparison, plan: "reference-output-mismatch" };
        const comparison = compareOutputs(expected, actual, outputNames, spec);
        if (comparison.ok && spec.offsetOutput) {
          const offset = actual.buffers[spec.offsetOutput]?.[0];
          if (offset !== spec.expectedOffset) {
            return { ok: false, plan: plan.kind, maxAbsDiff: Math.abs(Number(offset ?? NaN) - spec.expectedOffset) };
          }
        }
        return { ...comparison, plan: plan.kind, output: outputNames.join(",") };
      }

      function compareOutputs(expected, actual, outputNames, spec) {
        let worst = { ok: true, maxAbsDiff: 0 };
        for (const outputName of outputNames) {
          const expectedOutput = spec.expectedOutput === undefined && outputNames.length === 1
            ? expected.buffers[outputName]
            : spec.expectedOutput === undefined
              ? expected.buffers[outputName]
              : materializeTypedArray(spec.expectedOutput);
          const comparison = compareArrays(expectedOutput, actual.buffers[outputName], spec.tolerance);
          if (!comparison.ok) return { ...comparison, output: outputName };
          if (comparison.maxAbsDiff > worst.maxAbsDiff) worst = comparison;
        }
        return worst;
      }

      async function runPreparedResidentSaxpy(device) {
        const compiled = compileCudaLiteKernel(SOURCES.saxpy, { workgroupSize: [8, 1, 1] });
        const launch = { gridDim: [1, 1, 1], blockDim: [8, 1, 1] };
        const x = createWgslStorageBuffer(device, {
          valueType: "f32",
          data: new Float32Array([1, 2, 3, 4]),
          label: "e2e-resident-x",
        });
        const y = createWgslStorageBuffer(device, {
          valueType: "f32",
          data: new Float32Array([10, 20, 30, 40]),
          label: "e2e-resident-y",
        });
        try {
          const prepared = await prepareCompiledKernelWebGpu(
            device,
            compiled,
            {
              buffers: {},
              residentBuffers: { x, y },
              scalars: { a: 2, n: 4 },
              readback: [],
            },
            launch,
          );
          try {
            await prepared.run({ scalars: { a: 3 }, readback: [], awaitCompletion: true });
            const out = await readWgslStorageBuffer(device, y);
            return compareArrays(new Float32Array([13, 26, 39, 52]), out);
          } finally {
            prepared.destroy();
          }
        } finally {
          destroyWgslStorageBuffer(x);
          destroyWgslStorageBuffer(y);
        }
      }

      function webGpuResourceLimitBlocker(limits, plan) {
        for (const step of plan.steps ?? []) {
          const bindings = step.program?.bindings ?? [];
          const storageCount = bindings.filter((binding) => binding.kind === "storage").length;
          const uniformCount = bindings.filter((binding) => binding.kind === "uniform").length;
          const textureCount = bindings.filter((binding) => binding.kind === "texture2d").length;
          if (storageCount > limits.maxStorageBuffersPerShaderStage) {
            return "storage-buffers:" + storageCount + ">" + limits.maxStorageBuffersPerShaderStage;
          }
          if (uniformCount > limits.maxUniformBuffersPerShaderStage) {
            return "uniform-buffers:" + uniformCount + ">" + limits.maxUniformBuffersPerShaderStage;
          }
          if (textureCount > limits.maxSampledTexturesPerShaderStage) {
            return "sampled-textures:" + textureCount + ">" + limits.maxSampledTexturesPerShaderStage;
          }
        }
        return undefined;
      }

      function webGpuDiagnosticSkip(compiled) {
        if (compiled.diagnostics?.some((diagnostic) => diagnostic.code === "divergent-return-before-barrier")) {
          if (compiled.wgsl?.includes("var bg_active_lane: bool = true;")) return undefined;
          return "non-uniform-return-before-barrier";
        }
        return undefined;
      }

      function compareArrays(expected, actual, tolerance = 1e-5) {
        if (!expected || !actual || expected.length !== actual.length) {
          return { ok: false, maxAbsDiff: Number.POSITIVE_INFINITY };
        }
        let maxAbsDiff = 0;
        let firstDiff;
        for (let i = 0; i < expected.length; i++) {
          const diff = Math.abs(Number(expected[i]) - Number(actual[i]));
          if (diff > maxAbsDiff) maxAbsDiff = diff;
          if (firstDiff === undefined && diff > tolerance) {
            firstDiff = {
              index: i,
              expected: Number(expected[i]),
              actual: Number(actual[i]),
              absDiff: diff,
            };
          }
        }
        return {
          ok: maxAbsDiff <= tolerance,
          maxAbsDiff: round(maxAbsDiff),
          ...(firstDiff === undefined
            ? {}
            : {
                firstDiff: {
                  ...firstDiff,
                  absDiff: round(firstDiff.absDiff),
                },
              }),
        };
      }

      function round(value) {
        return Math.round(value * 1000) / 1000;
      }

    </script>
  </body>
</html>`;

const server = await createServer({
  root,
  logLevel: "error",
  server: { host: "127.0.0.1", port: 0 },
  resolve: {
    alias: moduleAliases(root, bundle),
  },
  plugins: [{
    name: "browsergrad-cuda-lite-e2e-page",
    configureServer(viteServer) {
      viteServer.middlewares.use("/__bg_cuda_lite_e2e__", async (req, res) => {
        try {
          const transformed = await viteServer.transformIndexHtml(req.originalUrl ?? "/__bg_cuda_lite_e2e__", html);
          res.statusCode = 200;
          res.setHeader("Content-Type", "text/html; charset=utf-8");
          res.end(transformed);
        } catch (error) {
          res.statusCode = 500;
          res.end(error instanceof Error ? error.stack : String(error));
        }
      });
    },
  }],
});

let browser;
try {
  await server.listen();
  const urls = server.resolvedUrls?.local ?? [];
  const baseUrl = urls[0];
  if (!baseUrl) throw new Error("Vite did not expose a local URL");
  browser = await chromium.launch({
    headless: !headed,
    args: ["--enable-unsafe-webgpu"],
  });
  const page = await browser.newPage();
  if (progress) {
    page.on("console", (message) => {
      const text = message.text();
      if (text.startsWith("__BG_PROGRESS__")) {
        const payload = text.slice("__BG_PROGRESS__".length);
        console.error(payload);
        if (progressPath && progressPath !== "true") {
          fs.appendFileSync(path.resolve(progressPath), `${payload}\n`);
        }
      }
    });
  }
  await page.goto(new URL("/__bg_cuda_lite_e2e__", baseUrl).href);
  const result = await page.evaluate(() => globalThis.__bgRunE2e());
  const report = {
    tool: "browsergrad-cuda-lite-webgpu-e2e",
    bundle,
    userAgent: await page.evaluate(() => navigator.userAgent),
    ...result,
  };
  if (jsonPath && jsonPath !== "true") {
    fs.writeFileSync(path.resolve(jsonPath), JSON.stringify(report, null, 2));
  }
  console.log(JSON.stringify(summaryOnly ? summarizeReport(report) : report, null, 2));
  if (requireWebGpu && !report.available) {
    throw new Error(`WebGPU unavailable: ${report.reason ?? "unknown"}`);
  }
  const filteredRun = caseFilters.length > 0;
  if (report.available && requireCorpusFixtures && !filteredRun && (report.missingCorpusFixtureNames?.length ?? 0) > 0) {
    throw new Error(`Missing corpus execution fixtures from /tmp corpora: ${report.missingCorpusFixtureNames.join(", ")}`);
  }
  if (report.available && requireCorpusFixtures && !filteredRun) {
    validateCorpusFixtureBaseline(report);
  }
  if (report.available && autoCorpusSmokeLimit > 0 && !filteredRun && (report.autoCorpusSmokeCovered ?? report.autoCorpusSmokePassed ?? 0) < autoCorpusSmokeLimit) {
    throw new Error(`Auto corpus smoke baseline failed: ${report.autoCorpusSmokeCovered ?? report.autoCorpusSmokePassed ?? 0}/${autoCorpusSmokeLimit} covered`);
  }
  if (report.available && report.failed > 0) {
    throw new Error(`CUDA-lite WebGPU e2e failed ${report.failed} case(s)`);
  }
  if (markdownPath && markdownPath !== "true") {
    fs.writeFileSync(path.resolve(markdownPath), markdownReport(report));
  }
} finally {
  if (browser) await browser.close();
  await server.close();
}
