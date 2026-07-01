#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import {
  cudaLiteCorpusExecutionFixtureBaseline,
} from "./cuda-lite-corpus-registry.mjs";
import {
  corpusExecutionFixturesForCaseFilters,
  loadAutoCorpusSmokeFixtures,
  loadCorpusExecutionSources,
  verifyCorpusFixtureCheckouts,
} from "./cuda-lite-webgpu-fixtures.mjs";
import {
  markdownReport,
  failureReplayCases,
  summarizeReport,
  validateCorpusFixtureBaseline,
  validateWarmMsMax,
  validateWarmSpeedup,
} from "./cuda-lite-webgpu-report.mjs";
import {
  applyAutoCorpusSmokeShard,
  effectiveAutoCorpusSmokeLimit as resolveEffectiveAutoCorpusSmokeLimit,
  findRepoRoot,
  moduleAliases,
  parseAutoCorpusSmokeFeatures,
  parseAutoCorpusSmokeMode,
  parseAutoCorpusSmokeProfile,
  parseAutoCorpusSmokeShard,
  parseBundle,
  parseCaseFilters,
  parseFlagArgs,
  parseNonNegativeInteger,
} from "./cuda-lite-webgpu-cli.mjs";

const args = parseFlagArgs(process.argv.slice(2));

const headed = args.get("--headed") === "true";
const requireWebGpu = args.get("--require-webgpu") === "true";
const requireCorpusFixtures = args.get("--require-corpus-fixtures") === "true";
const markdownPath = args.get("--markdown");
const jsonPath = args.get("--json");
const summaryOnly = args.get("--summary-only") === "true";
const progress = args.get("--progress") === "true";
const failFast = args.get("--fail-fast") === "true";
const forbidSkips = args.get("--forbid-skips") === "true";
const compileOnly = args.get("--compile-only") === "true";
const progressPath = args.get("--progress-file");
const profileCase = args.get("--profile-case");
const onlyAutoCorpusSmoke = args.get("--auto-corpus-smoke-only") === "true";
const onlyCorpusFixtures = args.get("--corpus-fixtures-only") === "true";
const autoCorpusSmokeCache = args.get("--auto-corpus-smoke-cache") === "true";
const caseFilters = parseCaseFilters(process.argv.slice(2));
const caseTimeoutMs = parseNonNegativeInteger(args.get("--case-timeout-ms") ?? "0", "--case-timeout-ms");
const warmup = parseNonNegativeInteger(args.get("--warmup") ?? "0", "--warmup");
const repeat = parsePositiveInteger(args.get("--repeat") ?? "1", "--repeat");
const expectWarmSpeedupMin = args.has("--expect-warm-speedup-min")
  ? parsePositiveNumber(args.get("--expect-warm-speedup-min"), "--expect-warm-speedup-min")
  : undefined;
const expectWarmMsMax = args.has("--expect-warm-ms-max")
  ? parsePositiveNumber(args.get("--expect-warm-ms-max"), "--expect-warm-ms-max")
  : undefined;
const bundle = parseBundle(args.get("--bundle") ?? "src");
const autoCorpusSmokeLimit = parseNonNegativeInteger(args.get("--auto-corpus-smoke-limit") ?? "0", "--auto-corpus-smoke-limit");
const autoCorpusSmokeMode = parseAutoCorpusSmokeMode(args.get("--auto-corpus-smoke-mode") ?? "reference");
const autoCorpusSmokeProfile = parseAutoCorpusSmokeProfile(args.get("--auto-corpus-smoke-profile") ?? "full");
const autoCorpusSmokeFeatures = parseAutoCorpusSmokeFeatures(args.get("--auto-corpus-smoke-features") ?? "");
const autoCorpusSmokeShard = parseAutoCorpusSmokeShard(args.get("--auto-corpus-smoke-shard") ?? "1/1");
const autoCorpusSmokeReferencePreflight = autoCorpusSmokeProfile !== "fast" && args.get("--no-auto-corpus-reference-preflight") !== "true";
const effectiveAutoCorpusSmokeLimit = resolveEffectiveAutoCorpusSmokeLimit(autoCorpusSmokeLimit, caseFilters);

const root = findRepoRoot(process.cwd());
const packageRequire = createRequire(path.join(root, "packages/browsergrad-compiler/package.json"));
const { createServer } = await import(pathToFileURL(packageRequire.resolve("vite")).href);
const playwright = await import(pathToFileURL(packageRequire.resolve("playwright")).href);
const chromium = playwright.chromium ?? playwright.default?.chromium;
if (!chromium) throw new Error("could not load Playwright chromium");
if (requireCorpusFixtures || effectiveAutoCorpusSmokeLimit > 0) verifyCorpusFixtureCheckouts(root);

const examplesRoot = path.join(root, "packages/browsergrad-compiler/examples");
const corpusExecutionFixtures = onlyAutoCorpusSmoke ? [] : corpusExecutionFixturesForCaseFilters(caseFilters);
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
  dynamicPointerOffsetLaunch: `
__global__ void child(float *dst) {
  int idx = threadIdx.x;
  dst[idx] = dst[idx] + (float)(idx + 1);
}
__global__ void parent(float *out) {
  if (threadIdx.x == 0) {
    dim3 grid(1);
    dim3 block(2);
    child<<<grid, block>>>(out + 1);
    cudaDeviceSynchronize();
  }
}`,
  dynamicOrderedLaunch: `
__global__ void add_child(float *dst, float amount) {
  int idx = threadIdx.x;
  dst[idx] = dst[idx] + amount;
}
__global__ void scale_child(float *dst, float factor) {
  int idx = threadIdx.x;
  dst[idx] = dst[idx] * factor;
}
__global__ void parent(float *out) {
  if (threadIdx.x == 0) {
    dim3 grid(1);
    dim3 block(4);
    add_child<<<grid, block>>>(out, 1.0f);
    cudaDeviceSynchronize();
    scale_child<<<grid, block>>>(out, 2.0f);
    cudaDeviceSynchronize();
    add_child<<<grid, block>>>(out + 1, 3.0f);
    cudaDeviceSynchronize();
  }
}`,
  dynamicChildPeerCopy: `
__global__ void child(float *dst, const float *src, int n) {
  if (threadIdx.x == 0) {
    cudaMemcpyPeerAsync(dst + 1, 1, src, 0, sizeof(float) * n, 0);
  }
}
__global__ void parent(float *dst, const float *src, int n) {
  if (threadIdx.x == 0) {
    dim3 grid(1);
    dim3 block(1);
    child<<<grid, block>>>(dst, src, n);
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
  dynamicPoolAliasLaunch: `
__global__ void child(DevicePool *childPool, float *dst) {
  if (threadIdx.x == 0) {
    float *ptr = (float*) deviceAllocate(childPool, sizeof(float));
    if (ptr != nullptr) {
      ptr[0] = 6.0f;
      dst[0] = ptr[0];
    }
  }
}
__global__ void parent(DevicePool *pool, float *out) {
  if (threadIdx.x == 0) {
    dim3 grid(1);
    dim3 block(1);
    child<<<grid, block>>>(pool, out);
    cudaDeviceSynchronize();
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
  localStoragePointerAliases: `
__global__ void aliasedRows(float* out, const float* inp, int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    const float* inRow = inp + row * cols;
    float* outRow = out + row * cols;
    for (int col = 0; col < cols; col++) {
      outRow[col] = inRow[col] + 1.0f;
    }
  }
}`,
  vectorPointerMemoryView: `
__device__ float4 loadVec(const float* address) {
  return *reinterpret_cast<const float4*>(address);
}

__device__ void storeVec(float* address, float4 value) {
  *reinterpret_cast<float4*>(address) = value;
}

__global__ void vectorPointerMemoryView(float* out, const float* inp) {
  if (threadIdx.x == 0) {
    float4 value = loadVec(inp);
    value.y += 10.0f;
    storeVec(out, value);
  }
}`,
  vectorCastDynamicBaseRead: `
__global__ void vectorCastDynamicBaseRead(float* out, const float* inp) {
  int idx = threadIdx.x;
  const float4* view = reinterpret_cast<const float4*>(inp);
  float4 value = view[idx];
  out[idx * 4 + 0] = value.x + (float)idx;
  out[idx * 4 + 1] = value.y + (float)idx;
  out[idx * 4 + 2] = value.z + (float)idx;
  out[idx * 4 + 3] = value.w + (float)idx;
}`,
  vectorCacheHintDynamicRead: `
__global__ void vectorCacheHintDynamicRead(float* out, const float* inp) {
  int idx = threadIdx.x;
  const float4* view = reinterpret_cast<const float4*>(inp);
  float4 value = __ldcs(&view[idx]);
  out[idx * 4 + 0] = value.x + 10.0f;
  out[idx * 4 + 1] = value.y + 20.0f;
  out[idx * 4 + 2] = value.z + 30.0f;
  out[idx * 4 + 3] = value.w + 40.0f;
}`,
  vectorCacheHintDynamicWrite: `
__global__ void vectorCacheHintDynamicWrite(float* out, const float* inp) {
  int idx = threadIdx.x;
  const float4* readView = reinterpret_cast<const float4*>(inp);
  float4* writeView = reinterpret_cast<float4*>(out);
  float4 value = readView[idx];
  value.x += 100.0f;
  value.y += 200.0f;
  value.z += 300.0f;
  value.w += 400.0f;
  __stcs(&writeView[idx], value);
}`,
  vectorCacheHintDeviceHelper: `
__device__ void bump_vec(float4* out, const float4* inp, int idx) {
  float4 value = __ldcs(&inp[idx]);
  value.x += 1000.0f;
  value.y += 2000.0f;
  value.z += 3000.0f;
  value.w += 4000.0f;
  __stcs(&out[idx], value);
}

__global__ void vectorCacheHintDeviceHelper(float* out, const float* inp) {
  int idx = threadIdx.x;
  const float4* readView = reinterpret_cast<const float4*>(inp);
  float4* writeView = reinterpret_cast<float4*>(out);
  bump_vec(writeView, readView, idx);
}`,
  vectorCacheHintLanePointer: `
__device__ void write_lane(float4* out, const float4* inp, int idx) {
  float value = __ldcs(&inp[idx].z);
  __stcs(&out[idx].y, value + 50.0f);
}

__global__ void vectorCacheHintLanePointer(float* out, const float* inp) {
  int idx = threadIdx.x;
  const float4* readView = reinterpret_cast<const float4*>(inp);
  float4* writeView = reinterpret_cast<float4*>(out);
  write_lane(writeView, readView, idx);
}`,
  vectorLanePointerOffsetHelper: `
__device__ void write_lane_offset(float4* out, int idx, float value) {
  out[idx].y = value;
}

__global__ void vectorLanePointerOffsetHelper(float* out, const float* inp) {
  int idx = threadIdx.x;
  const float4* readView = reinterpret_cast<const float4*>(inp);
  float4* writeView = reinterpret_cast<float4*>(out + 4);
  float4 value = readView[idx];
  write_lane_offset(writeView, idx, value.x + value.w);
}`,
  vectorToScalarPointerOffsetHelper: `
__device__ float load_scalar_offset(const float* inp, int idx) {
  return inp[idx];
}

__global__ void vectorToScalarPointerOffsetHelper(float* out, const float4* inp) {
  int idx = threadIdx.x;
  const float* scalarView = reinterpret_cast<const float*>(inp + 1);
  out[idx] = load_scalar_offset(scalarView, idx);
}`,
  vectorToScalarPointerWriteOffsetHelper: `
__device__ void write_scalar_offset(float* out, int idx, float value) {
  out[idx] = value;
}

__global__ void vectorToScalarPointerWriteOffsetHelper(float4* out) {
  int idx = threadIdx.x;
  float* scalarView = reinterpret_cast<float*>(out + 1);
  write_scalar_offset(scalarView, idx, 7.0f + (float)idx);
}`,
  vectorScalarVectorAliasRoundtrip: `
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
}`,
  vectorToScalarPointerAtomicOffsetHelper: `
__device__ void add_scalar_offset(float* out, int idx, float value) {
  atomicAdd(&out[idx], value);
}

__global__ void vectorToScalarPointerAtomicOffsetHelper(float4* out) {
  int idx = threadIdx.x;
  float* scalarView = reinterpret_cast<float*>(out + 1);
  add_scalar_offset(scalarView, idx, 2.0f + (float)idx);
}`,
  uintVectorToScalarPointerAtomicOffsetHelper: `
__device__ void add_uint_scalar_offset(uint* out, int idx, uint value) {
  atomicAdd(&out[idx], value);
}

__global__ void uintVectorToScalarPointerAtomicOffsetHelper(uint4* out) {
  int idx = threadIdx.x;
  uint* scalarView = reinterpret_cast<uint*>(out + 1);
  add_uint_scalar_offset(scalarView, idx, 2u + (uint)idx);
}`,
  intVectorToScalarPointerAtomicOffsetHelper: `
__device__ void sub_int_scalar_offset(int* out, int idx, int value) {
  atomicSub(&out[idx], value);
}

__global__ void intVectorToScalarPointerAtomicOffsetHelper(int4* out) {
  int idx = threadIdx.x;
  int* scalarView = reinterpret_cast<int*>(out + 1);
  sub_int_scalar_offset(scalarView, idx, 2 + idx);
}`,
  float3VectorToScalarPointerAtomicOffsetHelper: `
__device__ void add_float3_scalar_offset(float* out, int idx, float value) {
  atomicAdd(&out[idx], value);
}

__global__ void float3VectorToScalarPointerAtomicOffsetHelper(float3* out) {
  int idx = threadIdx.x;
  float* scalarView = reinterpret_cast<float*>(out + 1);
  add_float3_scalar_offset(scalarView, idx, 2.0f + (float)idx);
}`,
  uint3VectorScalarExchangeCas: `
__device__ void exchange_cas_uint3_scalar(uint* data, uint* report) {
  report[0] = atomicExch(&data[0], 9u);
  report[1] = atomicCAS(&data[1], 4u, 10u);
  report[2] = data[0];
  report[3] = data[1];
}

__global__ void uint3VectorScalarExchangeCas(uint3* data, uint* report) {
  uint* scalarView = reinterpret_cast<uint*>(data + 1);
  if (threadIdx.x == 0) {
    exchange_cas_uint3_scalar(scalarView, report);
  }
}`,
  float3VectorScalarMinMax: `
__device__ void minmax_float3_scalar(float* data, float* report) {
  report[0] = atomicMin(&data[0], 3.0f);
  report[1] = atomicMax(&data[1], 8.0f);
  report[2] = data[0];
  report[3] = data[1];
}

__global__ void float3VectorScalarMinMax(float3* data, float* report) {
  float* scalarView = reinterpret_cast<float*>(data + 1);
  if (threadIdx.x == 0) {
    minmax_float3_scalar(scalarView, report);
  }
}`,
  sharedFloat3VectorToScalarAtomic: `
__device__ void add_shared_float3_scalar(float* out, int idx, float value) {
  atomicAdd(&out[idx], value);
}

__global__ void sharedFloat3VectorToScalarAtomic(float* out) {
  __shared__ float3 tile[2];
  int idx = threadIdx.x;
  if (idx < 2) {
    float* scalarView = reinterpret_cast<float*>(tile + 1);
    add_shared_float3_scalar(scalarView, idx, 7.0f + (float)idx);
    out[idx] = scalarView[idx];
  }
}`,
  deviceGlobalVectorToScalarAtomic: `
__device__ uint4 g_vec[2];

__device__ void add_global_scalar(uint* out, int idx, uint value) {
  atomicAdd(&out[idx], value);
}

__global__ void deviceGlobalVectorToScalarAtomic(uint* out) {
  int idx = threadIdx.x;
  uint* scalarView = reinterpret_cast<uint*>(g_vec + 1);
  add_global_scalar(scalarView, idx, 5u + (uint)idx);
  out[idx] = scalarView[idx];
}`,
  deviceGlobalFloat3VectorToScalarAtomic: `
__device__ float3 g_f3[2];

__device__ void add_global_float3_scalar(float* out, int idx, float value) {
  atomicAdd(&out[idx], value);
}

__global__ void deviceGlobalFloat3VectorToScalarAtomic(float* out) {
  int idx = threadIdx.x;
  float* scalarView = reinterpret_cast<float*>(g_f3 + 1);
  add_global_float3_scalar(scalarView, idx, 11.0f + (float)idx);
  out[idx] = scalarView[idx];
}`,
  deviceGlobalVectorPointerArray: `
__device__ float3 g_ptr_values[3];

__device__ float3 sum_global_ptrs(float3 *a, float3 *b, float3 *c) {
  return *a + *b + *c;
}

__global__ void deviceGlobalVectorPointerArray(float4 *out) {
  if (threadIdx.x == 0) {
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
  }
}`,
  crossSpaceVectorAliasConsistency: `
__device__ float4 g_alias_vec[2];

__device__ void adjust_storage_alias_lane(float *lanes, int offset, float delta) {
  lanes[offset] = lanes[offset] + delta;
}

__global__ void crossSpaceVectorAliasConsistency(float *out) {
  __shared__ float4 shared[2];
  if (threadIdx.x == 0) {
    float4 local[2];
    local[0] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    local[1] = make_float4(5.0f, 6.0f, 7.0f, 8.0f);
    shared[0] = make_float4(10.0f, 20.0f, 30.0f, 40.0f);
    shared[1] = make_float4(50.0f, 60.0f, 70.0f, 80.0f);
    g_alias_vec[0] = make_float4(100.0f, 200.0f, 300.0f, 400.0f);
    g_alias_vec[1] = make_float4(500.0f, 600.0f, 700.0f, 800.0f);
    local[1].z = local[1].z + 0.5f;
    adjust_storage_alias_lane(reinterpret_cast<float*>(shared + 1), 1, 1.5f);
    adjust_storage_alias_lane(reinterpret_cast<float*>(g_alias_vec + 1), 3, 2.5f);
    out[0] = local[1].z;
    out[1] = shared[1].y;
    out[2] = g_alias_vec[1].w;
  }
}`,
  sharedVectorToScalarAtomic: `
__device__ void add_shared_scalar(uint* out, int idx, uint value) {
  atomicAdd(&out[idx], value);
}

__global__ void sharedVectorToScalarAtomic(uint* out) {
  __shared__ uint4 tile[2];
  int idx = threadIdx.x;
  if (idx < 2) {
    uint* scalarView = reinterpret_cast<uint*>(tile + 1);
    add_shared_scalar(scalarView, idx, 7u + (uint)idx);
    out[idx] = scalarView[idx];
  }
}`,
  sharedInt3VectorToScalarAtomic: `
__device__ void sub_shared_int3_scalar(int* out, int idx, int value) {
  atomicSub(&out[idx], value);
}

__global__ void sharedInt3VectorToScalarAtomic(int* out) {
  __shared__ int3 tile[2];
  int idx = threadIdx.x;
  if (idx < 2) {
    int* scalarView = reinterpret_cast<int*>(tile + 1);
    scalarView[idx] = 20 + idx;
    sub_shared_int3_scalar(scalarView, idx, 3 + idx);
    out[idx] = scalarView[idx];
  }
}`,
  uintVectorScalarExchangeCas: `
__device__ void exchange_cas_scalar(uint* data, uint* report) {
  report[0] = atomicExch(&data[0], 9u);
  report[1] = atomicCAS(&data[1], 4u, 10u);
  report[2] = data[0];
  report[3] = data[1];
}

__global__ void uintVectorScalarExchangeCas(uint4* data, uint* report) {
  uint* scalarView = reinterpret_cast<uint*>(data + 1);
  if (threadIdx.x == 0) {
    exchange_cas_scalar(scalarView, report);
  }
}`,
  floatVectorScalarMinMax: `
__device__ void minmax_scalar(float* data, float* report) {
  report[0] = atomicMin(&data[0], 3.0f);
  report[1] = atomicMax(&data[1], 8.0f);
  report[2] = data[0];
  report[3] = data[1];
}

__global__ void floatVectorScalarMinMax(float4* data, float* report) {
  float* scalarView = reinterpret_cast<float*>(data + 1);
  if (threadIdx.x == 0) {
    minmax_scalar(scalarView, report);
  }
}`,
  sharedVectorScalarExchangeCas: `
__device__ void shared_exchange_cas_scalar(uint* data, uint* report) {
  report[0] = atomicExch(&data[0], 9u);
  report[1] = atomicCAS(&data[1], 4u, 10u);
  report[2] = data[0];
  report[3] = data[1];
}

__global__ void sharedVectorScalarExchangeCas(uint* report) {
  __shared__ uint4 tile[2];
  uint* scalarView = reinterpret_cast<uint*>(tile + 1);
  if (threadIdx.x == 0) {
    scalarView[0] = 3u;
    scalarView[1] = 4u;
    shared_exchange_cas_scalar(scalarView, report);
  }
}`,
  vectorDerefLaneWrite: `
__global__ void vectorDerefLaneWrite(float *x, float *out) {
  float4 *p = reinterpret_cast<float4 *>(&x[0]);
  (*p).z = (*p).x + (*p).y;
  out[0] = (*p).z;
}`,
  localVectorLaneCompound: `
__global__ void localVectorLaneCompound(float *out) {
  float4 value = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  value[2] += value[0] + 5.0f;
  value[0] = value[2] - value[1];
  out[0] = value.x;
  out[1] = value.z;
}`,
  sharedVectorOverlay: `
__global__ void sharedVectorOverlay(float *out) {
  extern __shared__ int params[];
  float4 *scratch = (float4*)params;
  if (threadIdx.x == 0) {
    scratch[0] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float4 value = scratch[0];
    out[0] = value.x + value.y + value.z + value.w;
  }
}`,
  localVectorPointerArray: `
__device__ float3 sum3(float3 *a, float3 *b, float3 *c) {
  return *a + *b + *c;
}

__global__ void localVectorPointerArray(float4 *out) {
  float3 values[3];
  values[0] = make_float3(1.0f, 2.0f, 3.0f);
  values[1] = make_float3(4.0f, 5.0f, 6.0f);
  values[2] = make_float3(7.0f, 8.0f, 9.0f);
  float3 *ptrs[3];
  ptrs[0] = &values[0];
  ptrs[1] = &values[1];
  ptrs[2] = &values[2];
  float3 total = sum3(ptrs[0], ptrs[1], ptrs[2]);
  out[0] = make_float4(*ptrs[0], 1.0f);
  out[1] = make_float4(total, 0.0f);
}`,
  sharedVectorPointerArray: `
__device__ float3 sum_shared3(float3 *a, float3 *b, float3 *c) {
  return *a + *b + *c;
}

__global__ void sharedVectorPointerArray(float4 *out) {
  __shared__ float3 values[3];
  if (threadIdx.x == 0) {
    values[0] = make_float3(1.0f, 2.0f, 3.0f);
    values[1] = make_float3(10.0f, 20.0f, 30.0f);
    values[2] = make_float3(100.0f, 200.0f, 300.0f);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float3 *ptrs[3];
    ptrs[0] = &values[0];
    ptrs[1] = &values[1];
    ptrs[2] = &values[2];
    float3 total = sum_shared3(ptrs[0], ptrs[1], ptrs[2]);
    out[0] = make_float4(*ptrs[1], 1.0f);
    out[1] = make_float4(total, 0.0f);
  }
}`,
  dynamicSharedVectorPointerArray: `
__device__ float3 sum_dynamic_shared3(float3 *a, float3 *b, float3 *c) {
  return *a + *b + *c;
}

__global__ void dynamicSharedVectorPointerArray(float4 *out) {
  extern __shared__ float scratch[];
  float3 *values = reinterpret_cast<float3*>(scratch);
  if (threadIdx.x == 0) {
    values[0] = make_float3(3.0f, 5.0f, 7.0f);
    values[1] = make_float3(11.0f, 13.0f, 17.0f);
    values[2] = make_float3(19.0f, 23.0f, 29.0f);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float3 *ptrs[3];
    ptrs[0] = &values[0];
    ptrs[1] = &values[1];
    ptrs[2] = &values[2];
    float3 total = sum_dynamic_shared3(ptrs[0], ptrs[1], ptrs[2]);
    out[0] = make_float4(*ptrs[0], 1.0f);
    out[1] = make_float4(total, 0.0f);
  }
}`,
  dynamicSharedVectorAliasChainPointerArray: `
__device__ float3 sum_dynamic_shared_chain3(float3 *a, float3 *b, float3 *c) {
  return *a + *b + *c;
}

__global__ void dynamicSharedVectorAliasChainPointerArray(float4 *out) {
  extern __shared__ float scratch[];
  float3 *values = reinterpret_cast<float3*>(scratch);
  if (threadIdx.x == 0) {
    values[0] = make_float3(2.0f, 3.0f, 5.0f);
    values[1] = make_float3(7.0f, 11.0f, 13.0f);
    values[2] = make_float3(17.0f, 19.0f, 23.0f);
    values[3] = make_float3(29.0f, 31.0f, 37.0f);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float3 *shifted = values + 1;
    float3 *ptrs[3];
    ptrs[0] = &shifted[0];
    ptrs[1] = &shifted[1];
    ptrs[2] = &values[3];
    float3 total = sum_dynamic_shared_chain3(ptrs[0], ptrs[1], ptrs[2]);
    out[0] = make_float4(*ptrs[1], 1.0f);
    out[1] = make_float4(total, 0.0f);
  }
}`,
  localArrayVectorScalarRoundtrip: `
__global__ void localArrayVectorScalarRoundtrip(float *out) {
  int tid = threadIdx.x;
  float scratch[12];
  for (int i = 0; i < 12; i++) {
    scratch[i] = (float)(i + 1);
  }
  float4 *vec = reinterpret_cast<float4*>(&scratch[0]);
  float *lanes = reinterpret_cast<float*>(vec + 1);
  float4 *round = reinterpret_cast<float4*>(lanes);
  round[0] = make_float4(round[0].x + (float)tid, round[0].y + 10.0f, round[0].z + 20.0f, round[0].w + 30.0f);
  out[tid * 4 + 0] = scratch[4];
  out[tid * 4 + 1] = scratch[5];
  out[tid * 4 + 2] = scratch[6];
  out[tid * 4 + 3] = scratch[7];
}`,
  sharedVectorHelper: `
__device__ float4 load_shared_float4(float4 *tile, int index) {
  return tile[index];
}

__global__ void sharedVectorHelper(const float4 *x, float4 *out) {
  __shared__ float4 tile[2];
  int tid = threadIdx.x;
  tile[tid] = x[tid];
  __syncthreads();
  float4 other = load_shared_float4(tile, 1 - tid);
  out[tid] = make_float4(other.x + 1.0f, other.y + 2.0f, other.z + 3.0f, other.w + 4.0f);
}`,
  guardedSharedVectorLanes: `
__device__ void write_shared_z(float4 *tile, int index, float value) {
  tile[index].z = value;
}

__global__ void guardedSharedVectorLanes(const float4 *x, float4 *out, int n) {
  __shared__ float4 tile[4];
  int tid = threadIdx.x;
  if (tid >= n) return;
  tile[tid] = x[tid];
  __syncthreads();
  float4 value = tile[tid];
  write_shared_z(tile, tid, value.x + value.y);
  __syncthreads();
  value = tile[tid];
  out[tid] = make_float4(value.x, value.y, value.z, value.w + (float)tid);
}`,
  guardedSharedFloat3Lanes: `
__device__ void write_shared_float3_y(float3 *tile, int index, float value) {
  tile[index].y = value;
}

__global__ void guardedSharedFloat3Lanes(float3 *out, int n) {
  __shared__ float3 tile[4];
  int tid = threadIdx.x;
  if (tid >= n) return;
  tile[tid] = make_float3((float)(tid + 1), (float)(tid + 10), (float)(tid + 100));
  __syncthreads();
  float3 value = tile[tid];
  write_shared_float3_y(tile, tid, value.x + value.z);
  __syncthreads();
  out[tid] = tile[tid];
}`,
  dynamicSharedFloat3View: `
__device__ void adjust_dynamic_shared_float3(float3 *tile, int index, float bias) {
  float3 value = tile[index];
  tile[index] = make_float3(value.x + bias, value.y + 2.0f * bias, value.z + 3.0f * bias);
}

__global__ void dynamicSharedFloat3View(float *out) {
  extern __shared__ float scratch[];
  int tid = threadIdx.x;
  float3 *tile = reinterpret_cast<float3 *>(scratch);
  tile[tid] = make_float3((float)(tid + 1), (float)(tid + 10), (float)(tid + 100));
  __syncthreads();
  adjust_dynamic_shared_float3(tile, tid, 0.5f + (float)tid);
  __syncthreads();
  float3 value = tile[tid];
  out[tid * 3 + 0] = value.x;
  out[tid * 3 + 1] = value.y;
  out[tid * 3 + 2] = value.z;
}`,
  dynamicSharedFloat3ScalarAtomic: `
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
}`,
  dynamicSharedVectorScalarPointerArrayAtomic: `
__global__ void dynamicSharedVectorScalarPointerArrayAtomic(float *out) {
  extern __shared__ float scratch[];
  float3 *values = reinterpret_cast<float3 *>(scratch);
  if (threadIdx.x == 0) {
    values[0] = make_float3(2.0f, 3.0f, 5.0f);
    values[1] = make_float3(7.0f, 11.0f, 13.0f);
    values[2] = make_float3(17.0f, 19.0f, 23.0f);
    values[3] = make_float3(29.0f, 31.0f, 37.0f);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float3 *shifted = values + 1;
    float *lanes = reinterpret_cast<float *>(shifted);
    float *tail = reinterpret_cast<float *>(values + 3);
    float *ptrs[2];
    ptrs[0] = &lanes[1];
    ptrs[1] = tail + 2;
    out[0] = atomicAdd(ptrs[0], 0.5f);
    out[1] = atomicAdd(ptrs[1], 1.5f);
    out[2] = lanes[1];
    out[3] = tail[2];
  }
}`,
  activeLaneGuardedRhs: `
__device__ int bump_counter(int *counter) {
  return atomicAdd(counter, 1);
}

__global__ void activeLaneGuardedRhs(int *counter, int n) {
  int tid = threadIdx.x;
  if (tid >= n) return;
  __syncthreads();
  bump_counter(counter);
  __syncthreads();
  if (tid == 0) {
    counter[1] = counter[0];
  }
}`,
  activeLaneAssignmentGuardedRhs: `
__device__ int bump_counter_for_assignment(int *counter) {
  return atomicAdd(counter, 1);
}

__global__ void activeLaneAssignmentGuardedRhs(int *counter, int *sink, int n) {
  int tid = threadIdx.x;
  if (tid >= n) return;
  __syncthreads();
  sink[tid] = bump_counter_for_assignment(counter);
  __syncthreads();
  if (tid == 0) {
    counter[1] = counter[0];
  }
}`,
  activeLaneVectorAtomicGuardedRhs: `
__global__ void activeLaneVectorAtomicGuardedRhs(float3 *data, float *out, int n) {
  int tid = threadIdx.x;
  float *scalar = reinterpret_cast<float*>(data + 1);
  if (tid >= n) return;
  __syncthreads();
  atomicAdd(&scalar[tid], 1.25f + (float)tid);
  __syncthreads();
  if (tid == 0) {
    out[0] = scalar[0];
    out[1] = scalar[1];
    out[2] = scalar[2];
  }
}`,
  activeLaneCompoundAssignmentGuardedRhs: `
__device__ int bump_counter_for_compound(int *counter) {
  return atomicAdd(counter, 1);
}

__global__ void activeLaneCompoundAssignmentGuardedRhs(int *counter, int *sink, int n) {
  int tid = threadIdx.x;
  if (tid >= n) return;
  __syncthreads();
  sink[tid] += bump_counter_for_compound(counter);
  __syncthreads();
  if (tid == 0) {
    counter[1] = counter[0];
  }
}`,
  earlyReturnLoopBarrier: `
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
}`,
  activeLaneLoopInternalReturnBarrier: `
__global__ void activeLaneLoopInternalReturnBarrier(float *x, int N) {
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
}`,
  activeLaneAlternateReturnBarrier: `
__global__ void activeLaneAlternateReturnBarrier(float *x, int N) {
  extern __shared__ float scratch[];
  int tid = threadIdx.x;
  if (tid < N) {
    scratch[tid] = x[tid];
  } else {
    return;
  }
  __syncthreads();
  x[tid] = scratch[tid] + 1.0f;
}`,
  activeLaneNestedReturnBarrier: `
__global__ void activeLaneNestedReturnBarrier(float *x, int N) {
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
}`,
  activeLaneLoopAlternateReturnBarrier: `
__global__ void activeLaneLoopAlternateReturnBarrier(float *x, int N) {
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
}`,
  activeLaneLoopReturnSideEffectBarrier: `
__global__ void activeLaneLoopReturnSideEffectBarrier(float *x, int N) {
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
}`,
  activeLaneVectorReturnSideEffectBarrier: `
__global__ void activeLaneVectorReturnSideEffectBarrier(float4 *out, int N) {
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
}`,
  activeLanePointerAliasReturnSideEffectBarrier: `
__global__ void activeLanePointerAliasReturnSideEffectBarrier(float *x, int N) {
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
}`,
  activeLaneAtomicReturnSideEffectBarrier: `
__global__ void activeLaneAtomicReturnSideEffectBarrier(uint *counter, uint *out, int N) {
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
}`,
  activeLaneSharedReturnSideEffectBarrier: `
__global__ void activeLaneSharedReturnSideEffectBarrier(uint *out, int N) {
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
}`,
  activeLaneUniformBreakBarrier: `
__global__ void activeLaneUniformBreakBarrier(float *x, int N) {
  extern __shared__ float scratch[];
  int tid = threadIdx.x;
  if (tid >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < 4; ++k) {
    if (k == 2) break;
    scratch[tid] = x[tid] + (float)k;
    __syncthreads();
    acc += scratch[(tid + 1) % N];
    __syncthreads();
  }
  x[tid] = acc;
}`,
  subgroupTruthinessAssignmentScalar: `
__global__ void subgroupTruthinessAssignmentScalar(float *x, int *out) {
  int tid = threadIdx.x;
  bool keep = bg_subgroup_add(x[tid]) > 0.0f;
  out[tid] = keep ? (tid + 1) : 0;
}`,
  frexpOutParams: `
__global__ void frexpOutParams(float *out, int *expOut) {
  int exponent = 0;
  float mantissa = frexp(9.0f, &exponent);
  out[0] = mantissa;
  expOut[0] = exponent;
}`,
  curandStorageState: `
__global__ void curandStorageState(curandState_t *states, float *out, unsigned int seed) {
  unsigned int tid = threadIdx.x;
  curand_init(seed, tid, 0, &states[tid]);
  out[tid] = curand_uniform(&states[tid]) + curand_normal(&states[tid]);
}`,
  fp8ConvertHelpers: `
__global__ void fp8ConvertHelpers(const uint* input, half* output, uint* encoded, int* as_int) {
  if (threadIdx.x == 0) {
    half e4m3 = __nv_cvt_fp8_to_halfraw(input[0], __NV_E4M3);
    output[0] = e4m3;
    encoded[0] = __nv_cvt_float_to_fp8(__half2float(e4m3), __NV_SATFINITE, __NV_E4M3);
    as_int[0] = __half2int_rz(e4m3);
  }
}`,
  bf16CacheHint: `
__global__ void bf16CacheHint(const __nv_bfloat16* input, __nv_bfloat16* output, float* as_float) {
  int idx = threadIdx.x;
  __nv_bfloat16 value = __ldcs(input + idx);
  __nv_bfloat16 next = __float2bfloat16(__bfloat162float(value) + 1.0f);
  __stcs(output + idx, next);
  as_float[idx] = __bfloat162float(output[idx]);
}`,
  cpAsyncSharedCopy: `
__global__ void cpAsyncSharedCopy(const float *input, float *output) {
  __shared__ float tile[4];
  if (threadIdx.x < 1) {
    CP_ASYNC_CG(&tile[0], &input[0], 16);
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads();
  output[threadIdx.x] = tile[threadIdx.x] + 1.0f;
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
  deviceGlobalTruthiness: `
__device__ unsigned int flag = 0;
__device__ unsigned int numErrors = 0;

__device__ void checkValue(int *data, int expected) {
  if ((data[threadIdx.x] != expected) && (!flag)) {
    numErrors++;
    flag = 1;
  }
}

__global__ void deviceGlobalTruthiness(int *data, uint *out) {
  checkValue(data, 7);
  out[0] = numErrors;
  out[1] = flag;
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
  wmmaToy: `
__global__ void wmmaToy(float* A, float* B, float* C) {
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
}`,
  wmmaTf32: `
__global__ void wmmaTf32(float* A, float* C) {
  wmma::fragment<wmma::matrix_a, 2, 2, 2, wmma::precision::tf32, wmma::row_major> a;
  wmma::load_matrix_sync(a, A, 2);
  for (int t = 0; t < a.num_elements; t++) {
    a.x[t] = wmma::__float_to_tf32(a.x[t]) + 1.0f;
  }
  wmma::store_matrix_sync(C, a, 2, wmma::mem_row_major);
}`,
  wmmaImma: `
__global__ void wmmaImma(uint8_t* A, uint8_t* B, int* C) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, uint8_t, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_t, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;
  wmma::fill_fragment(c, 1);
  wmma::load_matrix_sync(a, A, 16);
  wmma::load_matrix_sync(b, B, 16);
  wmma::mma_sync(c, a, b, c);
  wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
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
  half2Ops: `
__global__ void half2Ops(const half2 *x, const half2 *y, half2 *out, float *scalar) {
  half2 sum = __hadd2(x[0], y[0]);
  half2 prod = __hmul2(sum, make_half2(__float2half(2.0f), __float2half(0.5f)));
  out[0] = __hmax2(prod, make_half2(__float2half(5.0f), __float2half(5.0f)));
  half2 fused = __hfma2(x[0], y[0], make_half2(__float2half(1.0f), __float2half(2.0f)));
  out[1] = x[0] * y[0] + fused;
  scalar[0] = __low2float(fused) + __high2float(fused);
}`,
  intrinsicPack: `
__global__ void intrinsicPack(half2 *h, float2 *f, float *out) {
  int lane = __shfl_sync(0xffffffff, threadIdx.x, 0);
  __syncwarp(0xffffffff);
  __threadfence();
  half2 value = make_half2(__int2half_rn(lane + 1), __int2half_rn(4));
  h[0] = value;
  f[0] = __half22float2(value);
  out[0] = __fmaf_rn(f[0].x, 2.0f, f[0].y);
}`,
  complexMagnitude: `
__global__ void complexMagnitude(cufftComplex *data, float *mag, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float real = data[idx].x;
    float imag = data[idx].y;
    mag[idx] = sqrtf(real * real + imag * imag);
  }
}`,
  complexMultiply: `
__global__ void complexMultiply(cufftComplex *A, const cufftComplex *B, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    cufftComplex a = A[idx];
    cufftComplex b = B[idx];
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    A[idx] = c;
  }
}`,
  complexHelperPointwise: `
static __device__ __host__ inline float2 ComplexScale(float2 a, float s) {
  return make_float2(a.x * s, a.y * s);
}
static __device__ __host__ inline float2 ComplexMul(float2 a, float2 b) {
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__global__ void complexHelperPointwise(cufftComplex *a, cufftComplex *b, float scale) {
  int i = threadIdx.x;
  a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
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
  subgroupArrayAfterReturn: `
__global__ void subgroupArrayAfterReturn(float *x, float *out, int n) {
  int idx = threadIdx.x;
  if (idx >= n) return;
  float values[2];
  values[0] = x[idx];
  values[1] = x[idx] + 1.0f;
  for (int k = 0; k < 2; k++) {
    values[k] = bg_subgroup_add(values[k]);
  }
  if (idx == 0) {
    out[0] = values[0] + values[1];
  }
}`,
  cooperativeTileVectorReduce: `
namespace cg = cooperative_groups;
__device__ float2 merge_pair(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
__global__ void cooperativeTileVectorReduce(const float *x, float *out) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<4> tile = cg::tiled_partition<4>(block);
  int lane = tile.thread_rank();
  float value = x[threadIdx.x];
  float scalar = cg::reduce(tile, value, cg::plus<float>());
  float2 pair = make_float2(value, value * 2.0f);
  float2 total = cg::reduce(tile, pair, merge_pair);
  int shuffled = tile.shfl(threadIdx.x + 10, 2);
  if (lane == 0) {
    out[0] = scalar;
    out[1] = total.x;
    out[2] = total.y;
    out[3] = (float)tile.size();
    out[4] = (float)tile.meta_group_size();
    out[5] = (float)tile.meta_group_rank();
    out[6] = (float)shuffled;
  }
}`,
  cooperativeBinaryPartitionReduce: `
namespace cg = cooperative_groups;
__global__ void cooperativeBinaryPartitionReduce(int *out) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<4> tile = cg::tiled_partition<4>(block);
  int value = threadIdx.x + 1;
  auto part = cg::binary_partition(tile, (threadIdx.x & 1) != 0);
  int sum = cg::reduce(part, value, cg::plus<int>());
  if (part.thread_rank() == 0) {
    if ((threadIdx.x & 1) == 0) {
      out[0] = sum;
      out[2] = part.size();
    } else {
      out[1] = sum;
      out[3] = part.size();
    }
  }
}`,
  cooperativeTileVoteMask: `
namespace cg = cooperative_groups;
__global__ void cooperativeTileVoteMask(int *out) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<4> tile = cg::tiled_partition<4>(block);
  int lane = tile.thread_rank();
  bool secondTile = threadIdx.x >= 4;
  if (lane == 0) {
    int base = tile.meta_group_rank() * 3;
    out[base] = tile.any(secondTile) ? 1 : 0;
    out[base + 1] = tile.all(secondTile) ? 1 : 0;
    out[base + 2] = (int)tile.ballot(secondTile);
  }
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
  conditionalVectorScalarPointerAtomic: `
__global__ void conditionalVectorScalarPointerAtomic(uint4 *left, uint4 *right, uint *out, int pickRight) {
  uint *leftScalar = reinterpret_cast<uint*>(left + 1);
  uint *rightScalar = reinterpret_cast<uint*>(right + 1);
  uint *ptr = pickRight ? rightScalar : leftScalar;
  if (threadIdx.x == 0) {
    out[0] = atomicAdd(ptr, 1u);
    out[1] = ptr[0];
  }
}`,
  conditionalFloat3VectorScalarPointerAtomic: `
__global__ void conditionalFloat3VectorScalarPointerAtomic(float3 *left, float3 *right, float *out, int pickRight) {
  float *leftScalar = reinterpret_cast<float*>(left + 1);
  float *rightScalar = reinterpret_cast<float*>(right + 1);
  float *ptr = pickRight ? rightScalar : leftScalar;
  if (threadIdx.x == 0) {
    out[0] = atomicAdd(ptr, 1.5f);
    out[1] = ptr[0];
  }
}`,
  helperConditionalVectorScalarPointerAtomic: `
__device__ void add_vector_scalar_ptr(uint *ptr, uint *out) {
  out[0] = atomicAdd(ptr, 1u);
  out[1] = ptr[0];
}

__global__ void helperConditionalVectorScalarPointerAtomic(uint4 *left, uint4 *right, uint *out, int pickRight) {
  uint *leftScalar = reinterpret_cast<uint*>(left + 1);
  uint *rightScalar = reinterpret_cast<uint*>(right + 1);
  uint *ptr = pickRight ? rightScalar : leftScalar;
  if (threadIdx.x == 0) {
    add_vector_scalar_ptr(ptr, out);
  }
}`,
  pointerArrayVectorScalarAtomic: `
__global__ void pointerArrayVectorScalarAtomic(uint4 *counter, uint4 *untouched, uint *out) {
  uint *ptrs[2];
  if (threadIdx.x == 0) {
    ptrs[0] = reinterpret_cast<uint*>(counter + 1);
    ptrs[1] = reinterpret_cast<uint*>(untouched + 1);
    out[0] = atomicAdd(ptrs[0], 1u);
    out[1] = ptrs[0][0];
    out[2] = ptrs[1][0];
  }
}`,
  pointerArrayFloat3VectorScalarAtomic: `
__global__ void pointerArrayFloat3VectorScalarAtomic(float3 *counter, float3 *untouched, float *out) {
  float *ptrs[2];
  if (threadIdx.x == 0) {
    ptrs[0] = reinterpret_cast<float*>(counter + 1);
    ptrs[1] = reinterpret_cast<float*>(untouched + 1);
    out[0] = atomicAdd(ptrs[0], 1.5f);
    out[1] = ptrs[0][0];
    out[2] = ptrs[1][0];
  }
}`,
  helperPointerArrayVectorScalarAtomic: `
__device__ void add_vector_scalar_array_ptr(uint *ptr, uint *out) {
  out[0] = atomicAdd(ptr, 1u);
  out[1] = ptr[0];
}

__global__ void helperPointerArrayVectorScalarAtomic(uint4 *counter, uint4 *untouched, uint *out) {
  uint *ptrs[2];
  if (threadIdx.x == 0) {
    ptrs[0] = reinterpret_cast<uint*>(counter + 1);
    ptrs[1] = reinterpret_cast<uint*>(untouched + 1);
    add_vector_scalar_array_ptr(ptrs[0], out);
    out[2] = ptrs[1][0];
  }
}`,
  conditionalVectorLanePointerWrite: `
__device__ void write_selected_lane(float *ptr, float delta) {
  ptr[0] = ptr[0] + delta;
}

__global__ void conditionalVectorLanePointerWrite(float4 *left, float4 *right, float *out, int pickRight) {
  float *leftScalar = reinterpret_cast<float*>(left + 1);
  float *rightScalar = reinterpret_cast<float*>(right + 1);
  float *ptr = pickRight ? (rightScalar + 2) : (leftScalar + 1);
  if (threadIdx.x == 0) {
    write_selected_lane(ptr, 0.5f);
    out[0] = leftScalar[1];
    out[1] = rightScalar[2];
  }
}`,
  helperPointerArraySelectedArgs: `
__device__ void write_two_selected_ptrs(float *a, float *b, float *out) {
  a[0] = a[0] + 10.0f;
  b[0] = b[0] + 20.0f;
  out[0] = a[0];
  out[1] = b[0];
}

__global__ void helperPointerArraySelectedArgs(float4 *values, float *out) {
  float *lanes = reinterpret_cast<float*>(values);
  float *ptrs[2];
  ptrs[0] = lanes + 1;
  ptrs[1] = lanes + 6;
  if (threadIdx.x == 0) {
    write_two_selected_ptrs(ptrs[0], ptrs[1], out);
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
  vectorLaneAtomic: `
__device__ void bump_vector_lanes(float4 *view, int idx) {
  atomicAdd(&view[idx].x, 1.5f);
  atomicAdd(&view[idx].w, 4.5f);
}

__global__ void vectorLaneAtomic(float *out) {
  int idx = threadIdx.x;
  float4 *view = reinterpret_cast<float4*>(out);
  bump_vector_lanes(view, idx);
}`,
  atomicFloatExchangeAssign: `
__global__ void atomicFloatExchangeAssign(float *data) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < 4) {
    float newValue = 10.0f;
    float oldValue = atomicExch(&data[idx], newValue);
    data[idx] = oldValue + newValue;
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
  surfaceRead3d: `
__device__ float read_z(cudaSurfaceObject_t surfaceArg, int row, int z) {
  return surf3Dread<float>(surfaceArg, 1 * sizeof(float), row, z);
}

__global__ void surfaceRead3d(cudaSurfaceObject_t surf, float *out) {
  if (threadIdx.x == 0) {
    float value = 0.0f;
    surf3Dread(&value, surf, 0, 1, 1);
    out[0] = value;
    out[1] = read_z(surf, 1, 1);
  }
}`,
  surfaceVectorRead3d: `
__device__ float4 read_vec_z(cudaSurfaceObject_t surfaceArg, int row, int z) {
  return surf3Dread<float4>(surfaceArg, 0, row, z);
}

__global__ void surfaceVectorRead3d(cudaSurfaceObject_t surf, float *out) {
  if (threadIdx.x == 0) {
    float4 pointerValue;
    surf3Dread(&pointerValue, surf, 0, 0, 1);
    float4 returnValue = read_vec_z(surf, 0, 1);
    out[0] = pointerValue.x + returnValue.x;
    out[1] = pointerValue.y + returnValue.y;
    out[2] = pointerValue.z + returnValue.z;
    out[3] = pointerValue.w + returnValue.w;
  }
}`,
  surfaceLayeredWrite: `
__global__ void surfaceLayeredWrite(cudaSurfaceObject_t outputSurf) {
  if (threadIdx.x == 0) {
    surf2DLayeredwrite(23.0f, outputSurf, 1 * sizeof(float), 1, 1);
  }
}`,
  surface1DWrite: `
__global__ void surface1DWrite(cudaSurfaceObject_t outputSurf) {
  if (threadIdx.x == 0) {
    surf1Dwrite(37.0f, outputSurf, 2 * sizeof(float));
  }
}`,
  surface1DVectorWrite: `
__device__ void write_1d_surface_vec(cudaSurfaceObject_t surfaceArg, float4 value, int x) {
  surf1Dwrite(value, surfaceArg, x * sizeof(float));
}

__global__ void surface1DVectorWrite(cudaSurfaceObject_t outputSurf) {
  if (threadIdx.x == 0) {
    write_1d_surface_vec(outputSurf, make_float4(2.0f, 3.0f, 5.0f, 7.0f), 1);
  }
}`,
  surface1DRead: `
__device__ float read_1d_surface(cudaSurfaceObject_t surfaceArg, int x) {
  return surf1Dread<float>(surfaceArg, x * sizeof(float));
}

__global__ void surface1DRead(cudaSurfaceObject_t surf, float *out) {
  if (threadIdx.x == 0) {
    float pointerValue = 0.0f;
    surf1Dread(&pointerValue, surf, 2 * sizeof(float));
    float returnValue = read_1d_surface(surf, 3);
    out[0] = pointerValue;
    out[1] = returnValue;
    out[2] = pointerValue + returnValue;
  }
}`,
  surface1DVectorRead: `
__device__ float4 read_1d_surface_vec(cudaSurfaceObject_t surfaceArg, int x) {
  return surf1Dread<float4>(surfaceArg, x * sizeof(float));
}

__global__ void surface1DVectorRead(cudaSurfaceObject_t surf, float *out) {
  if (threadIdx.x == 0) {
    float4 pointerValue = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    surf1Dread(&pointerValue, surf, 1 * sizeof(float));
    float4 returnValue = read_1d_surface_vec(surf, 4);
    out[0] = pointerValue.x + returnValue.x;
    out[1] = pointerValue.y + returnValue.y;
    out[2] = pointerValue.z + returnValue.z;
    out[3] = pointerValue.w + returnValue.w;
  }
}`,
  surface1DVectorActiveLaneReturn: `
__device__ void write_1d_surface_vec_active(cudaSurfaceObject_t surfaceArg, float base) {
  surf1Dwrite(make_float4(base + 1.0f, base + 2.0f, base + 3.0f, base + 4.0f), surfaceArg, 1 * sizeof(float));
}

__global__ void surface1DVectorActiveLaneReturn(cudaSurfaceObject_t surf, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    write_1d_surface_vec_active(surf, 10.0f + (float)tid);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    float4 value = surf1Dread<float4>(surf, 1 * sizeof(float));
    out[0] = value.x + value.y + value.z + value.w;
  } else {
    out[tid] = 1.0f + (float)tid;
  }
}`,
  surfaceHelperVectorLayeredWrite: `
__device__ void write_layered_vec(cudaSurfaceObject_t surfaceArg, float4 value, int row, int layer) {
  surf2DLayeredwrite(value, surfaceArg, 0, row, layer);
}

__global__ void surfaceHelperVectorLayeredWrite(cudaSurfaceObject_t outputSurf) {
  if (threadIdx.x == 0) {
    write_layered_vec(outputSurf, make_float4(2.0f, 3.0f, 5.0f, 7.0f), 1, 1);
  }
}`,
  surfaceLayeredRead: `
__device__ float read_layer(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<float>(surfaceArg, 1 * sizeof(float), row, layer);
}

__global__ void surfaceLayeredRead(cudaSurfaceObject_t surf, float *out) {
  if (threadIdx.x == 0) {
    float value = 0.0f;
    surf2DLayeredread(&value, surf, 0, 1, 1);
    out[0] = value;
    out[1] = read_layer(surf, 1, 1);
  }
}`,
  surfaceVectorLayeredRead: `
__device__ float4 read_layer_vec(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<float4>(surfaceArg, 0, row, layer);
}

__global__ void surfaceVectorLayeredRead(cudaSurfaceObject_t surf, float *out) {
  if (threadIdx.x == 0) {
    float4 pointerValue;
    surf2DLayeredread(&pointerValue, surf, 0, 0, 1);
    float4 returnValue = read_layer_vec(surf, 0, 1);
    out[0] = pointerValue.x + returnValue.x;
    out[1] = pointerValue.y + returnValue.y;
    out[2] = pointerValue.z + returnValue.z;
    out[3] = pointerValue.w + returnValue.w;
  }
}`,
  surfaceVectorReadActiveLaneReturn: `
__device__ float4 read_layer_vec_active(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<float4>(surfaceArg, 0, row, layer);
}

__global__ void surfaceVectorReadActiveLaneReturn(cudaSurfaceObject_t surf, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float4 value = read_layer_vec_active(surf, 0, 1);
    out[tid] = value.x + value.y + value.z + value.w + (float)tid;
    return;
  }
  __syncthreads();
  if (tid == 0) {
    float4 value;
    surf3Dread(&value, surf, 0, 0, 1);
    out[0] = value.x + value.y + value.z + value.w;
  } else {
    out[tid] = 1.0f + (float)tid;
  }
}`,
  surfaceVectorWriteActiveLaneReturn: `
__device__ void write_layer_vec_active(cudaSurfaceObject_t surfaceArg, int row, int layer, float base) {
  surf2DLayeredwrite(make_float4(base + 1.0f, base + 2.0f, base + 3.0f, base + 4.0f), surfaceArg, 0, row, layer);
}

__global__ void surfaceVectorWriteActiveLaneReturn(cudaSurfaceObject_t surf, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    write_layer_vec_active(surf, 0, 1, 10.0f + (float)tid);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    float4 value;
    surf3Dread(&value, surf, 0, 0, 1);
    out[0] = value.x + value.y + value.z + value.w;
  } else {
    out[tid] = 1.0f + (float)tid;
  }
}`,
  surfaceFloat2VectorActiveLaneReturn: `
__device__ void write_layer_float2_active(cudaSurfaceObject_t surfaceArg, int row, int layer, float base) {
  surf2DLayeredwrite(make_float2(base + 1.0f, base + 2.0f), surfaceArg, 0, row, layer);
}

__device__ float2 read_layer_float2_active(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<float2>(surfaceArg, 0, row, layer);
}

__global__ void surfaceFloat2VectorActiveLaneReturn(cudaSurfaceObject_t surf, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    write_layer_float2_active(surf, 0, 1, 40.0f + (float)tid);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    float2 value = read_layer_float2_active(surf, 0, 1);
    out[0] = value.x + value.y;
  } else {
    out[tid] = 1.0f + (float)tid;
  }
}`,
  surfaceUint2VectorActiveLaneReturn: `
__device__ void write_layer_uint2_active(cudaSurfaceObject_t surfaceArg, int row, int layer, uint base) {
  surf2DLayeredwrite(make_uint2(base + 1u, base + 2u), surfaceArg, 0, row, layer);
}

__device__ uint2 read_layer_uint2_active(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<uint2>(surfaceArg, 0, row, layer);
}

__global__ void surfaceUint2VectorActiveLaneReturn(cudaSurfaceObject_t surf, uint *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    write_layer_uint2_active(surf, 0, 1, 50u + (uint)tid);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    uint2 value = read_layer_uint2_active(surf, 0, 1);
    out[0] = value.x + value.y;
  } else {
    out[tid] = 1u + (uint)tid;
  }
}`,
  surfaceInt2VectorActiveLaneReturn: `
__device__ void write_layer_int2_active(cudaSurfaceObject_t surfaceArg, int row, int layer, int base) {
  surf2DLayeredwrite(make_int2(base + 1, base + 2), surfaceArg, 0, row, layer);
}

__device__ int2 read_layer_int2_active(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<int2>(surfaceArg, 0, row, layer);
}

__global__ void surfaceInt2VectorActiveLaneReturn(cudaSurfaceObject_t surf, int *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    write_layer_int2_active(surf, 0, 1, 60 + tid);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    int2 value = read_layer_int2_active(surf, 0, 1);
    out[0] = value.x + value.y;
  } else {
    out[tid] = 1 + tid;
  }
}`,
  surfaceFloat3VectorActiveLaneReturn: `
__device__ void write_layer_float3_active(cudaSurfaceObject_t surfaceArg, int row, int layer, float base) {
  surf2DLayeredwrite(make_float3(base + 1.0f, base + 2.0f, base + 3.0f), surfaceArg, 0, row, layer);
}

__device__ float3 read_layer_float3_active(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<float3>(surfaceArg, 0, row, layer);
}

__global__ void surfaceFloat3VectorActiveLaneReturn(cudaSurfaceObject_t surf, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    write_layer_float3_active(surf, 0, 1, 20.0f + (float)tid);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    float3 value = read_layer_float3_active(surf, 0, 1);
    out[0] = value.x + value.y + value.z;
  } else {
    out[tid] = 1.0f + (float)tid;
  }
}`,
  surfaceFloat4VectorActiveLaneReturn: `
__device__ void write_layer_float4_active(cudaSurfaceObject_t surfaceArg, int row, int layer, float base) {
  surf2DLayeredwrite(make_float4(base + 1.0f, base + 2.0f, base + 3.0f, base + 4.0f), surfaceArg, 0, row, layer);
}

__device__ float4 read_layer_float4_active(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<float4>(surfaceArg, 0, row, layer);
}

__global__ void surfaceFloat4VectorActiveLaneReturn(cudaSurfaceObject_t surf, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    write_layer_float4_active(surf, 0, 1, 70.0f + (float)tid);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    float4 value = read_layer_float4_active(surf, 0, 1);
    out[0] = value.x + value.y + value.z + value.w;
  } else {
    out[tid] = 1.0f + (float)tid;
  }
}`,
  surfaceMixedScalarVectorActiveLaneReturn: `
__device__ void write_layer_mixed_scalar_active(cudaSurfaceObject_t surfaceArg, float value) {
  surf2DLayeredwrite(value, surfaceArg, 0, 0, 0);
}

__device__ void write_layer_mixed_vec_active(cudaSurfaceObject_t surfaceArg, float4 value) {
  surf2DLayeredwrite(value, surfaceArg, 0, 0, 1);
}

__device__ float read_layer_mixed_scalar_active(cudaSurfaceObject_t surfaceArg) {
  return surf2DLayeredread<float>(surfaceArg, 0, 0, 0);
}

__device__ float4 read_layer_mixed_vec_active(cudaSurfaceObject_t surfaceArg) {
  return surf2DLayeredread<float4>(surfaceArg, 0, 0, 1);
}

__global__ void surfaceMixedScalarVectorActiveLaneReturn(cudaSurfaceObject_t surf, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    write_layer_mixed_scalar_active(surf, 40.0f + (float)tid);
    write_layer_mixed_vec_active(surf, make_float4(70.0f + (float)tid, 71.0f + (float)tid, 72.0f + (float)tid, 73.0f + (float)tid));
    return;
  }
  __syncthreads();
  if (tid == 0) {
    float scalar = read_layer_mixed_scalar_active(surf);
    float4 value = read_layer_mixed_vec_active(surf);
    out[0] = scalar + value.x + value.y + value.z + value.w;
  } else {
    out[tid] = 1.0f + (float)tid;
  }
}`,
  surfacePointerAliasActiveLaneStore: `
__device__ float read_alias_surface_scalar(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<float>(surfaceArg, 0, 0, layer);
}

__device__ void write_surface_alias_lane(float *scalarOut, int lane, float value) {
  scalarOut[lane * 4 + 2] = value;
}

__global__ void surfacePointerAliasActiveLaneStore(cudaSurfaceObject_t surf, float4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float *scalarView = reinterpret_cast<float*>(out);
    write_surface_alias_lane(scalarView, tid, read_alias_surface_scalar(surf, 1) + (float)tid);
    return;
  }
  __syncthreads();
  out[tid] = make_float4(1.0f + (float)tid, 10.0f + (float)tid, 20.0f + (float)tid, 30.0f + (float)tid);
}`,
  surface1DPointerAliasActiveLaneStore: `
__device__ float read_1d_alias_surface_scalar(cudaSurfaceObject_t surfaceArg, int x) {
  return surf1Dread<float>(surfaceArg, x * sizeof(float));
}

__device__ void write_1d_surface_alias_lane(float *scalarOut, int lane, float value) {
  scalarOut[lane * 4 + 1] = value;
}

__global__ void surface1DPointerAliasActiveLaneStore(cudaSurfaceObject_t surf, float4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float *scalarView = reinterpret_cast<float*>(out);
    write_1d_surface_alias_lane(scalarView, tid, read_1d_alias_surface_scalar(surf, 2) + (float)tid);
    return;
  }
  __syncthreads();
  out[tid] = make_float4(1.0f + (float)tid, 10.0f + (float)tid, 20.0f + (float)tid, 30.0f + (float)tid);
}`,
  surfacePointerAliasAtomicActiveLaneStore: `
__device__ float read_atomic_alias_surface_scalar(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<float>(surfaceArg, 0, 0, layer);
}

__device__ void atomic_surface_alias_lane(uint *scalarOut, int lane, uint value) {
  atomicAdd(&scalarOut[lane * 4 + 2], value);
}

__global__ void surfacePointerAliasAtomicActiveLaneStore(cudaSurfaceObject_t surf, uint4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    uint *scalarView = reinterpret_cast<uint*>(out);
    atomic_surface_alias_lane(scalarView, tid, (uint)read_atomic_alias_surface_scalar(surf, 1) + (uint)tid);
    return;
  }
  __syncthreads();
  out[tid] = make_uint4(1u + (uint)tid, 10u + (uint)tid, 20u + (uint)tid, 30u + (uint)tid);
}`,
  surfacePointerAliasAtomicVectorReadback: `
__device__ float read_atomic_readback_surface_scalar(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<float>(surfaceArg, 0, 0, layer);
}

__device__ void atomic_surface_readback_alias_lane(uint *scalarOut, int lane, uint value) {
  atomicAdd(&scalarOut[lane * 4 + 1], value);
}

__global__ void surfacePointerAliasAtomicVectorReadback(cudaSurfaceObject_t surf, uint4 *out, uint *summary) {
  int tid = threadIdx.x;
  out[tid] = make_uint4(1u + (uint)tid, 10u + (uint)tid, 20u + (uint)tid, 30u + (uint)tid);
  __syncthreads();
  if (tid == 0) {
    uint *scalarView = reinterpret_cast<uint*>(out);
    atomic_surface_readback_alias_lane(scalarView, 1, (uint)read_atomic_readback_surface_scalar(surf, 1));
  }
  __syncthreads();
  if (tid == 1) {
    uint4 value = out[1];
    summary[0] = value.x + value.y + value.z + value.w;
  }
}`,
  surfacePointerAliasAtomicVectorCompound: `
__device__ float read_atomic_compound_surface_scalar(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<float>(surfaceArg, 0, 0, layer);
}

__device__ void atomic_surface_compound_alias_lane(uint *scalarOut, int lane, uint value) {
  atomicAdd(&scalarOut[lane * 4 + 1], value);
}

__device__ void add_surface_vector_alias(uint4 *vectorOut, int lane, uint4 value) {
  vectorOut[lane] += value;
}

__device__ void add_surface_vector_alias_y(uint4 *vectorOut, int lane, uint value) {
  vectorOut[lane].y += value;
}

__global__ void surfacePointerAliasAtomicVectorCompound(cudaSurfaceObject_t surf, uint4 *out, uint *summary) {
  int tid = threadIdx.x;
  out[tid] = make_uint4(1u + (uint)tid, 10u + (uint)tid, 20u + (uint)tid, 30u + (uint)tid);
  __syncthreads();
  if (tid == 0) {
    uint *scalarView = reinterpret_cast<uint*>(out);
    atomic_surface_compound_alias_lane(scalarView, 1, (uint)read_atomic_compound_surface_scalar(surf, 1));
    uint4 *vectorView = reinterpret_cast<uint4*>(out);
    add_surface_vector_alias(vectorView, 1, make_uint4(1u, 1u, 1u, 1u));
    add_surface_vector_alias_y(vectorView, 2, 9u);
  }
  __syncthreads();
  if (tid == 1) {
    uint4 value = out[1];
    uint4 laneTwo = out[2];
    summary[0] = (value.x + value.y + value.z + value.w) + 100u * (laneTwo.x + laneTwo.y + laneTwo.z + laneTwo.w);
  }
}`,
  surfaceUint3VectorActiveLaneReturn: `
__device__ void write_layer_uint3_active(cudaSurfaceObject_t surfaceArg, int row, int layer, uint base) {
  surf2DLayeredwrite(make_uint3(base + 1u, base + 2u, base + 3u), surfaceArg, 0, row, layer);
}

__device__ uint3 read_layer_uint3_active(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<uint3>(surfaceArg, 0, row, layer);
}

__global__ void surfaceUint3VectorActiveLaneReturn(cudaSurfaceObject_t surf, uint *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    write_layer_uint3_active(surf, 0, 1, 30u + (uint)tid);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    uint3 value = read_layer_uint3_active(surf, 0, 1);
    out[0] = value.x + value.y + value.z;
  } else {
    out[tid] = 1u + (uint)tid;
  }
}`,
  surfaceInt3VectorActiveLaneReturn: `
__device__ void write_layer_int3_active(cudaSurfaceObject_t surfaceArg, int row, int layer, int base) {
  surf2DLayeredwrite(make_int3(base + 1, base + 2, base + 3), surfaceArg, 0, row, layer);
}

__device__ int3 read_layer_int3_active(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<int3>(surfaceArg, 0, row, layer);
}

__global__ void surfaceInt3VectorActiveLaneReturn(cudaSurfaceObject_t surf, int *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    write_layer_int3_active(surf, 0, 1, 30 + tid);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    int3 value = read_layer_int3_active(surf, 0, 1);
    out[0] = value.x + value.y + value.z;
  } else {
    out[tid] = 1 + tid;
  }
}`,
  surfaceUint4VectorActiveLaneReturn: `
__device__ void write_layer_uint4_active(cudaSurfaceObject_t surfaceArg, int row, int layer, uint base) {
  surf2DLayeredwrite(make_uint4(base + 1u, base + 2u, base + 3u, base + 4u), surfaceArg, 0, row, layer);
}

__device__ uint4 read_layer_uint4_active(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<uint4>(surfaceArg, 0, row, layer);
}

__global__ void surfaceUint4VectorActiveLaneReturn(cudaSurfaceObject_t surf, uint *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    write_layer_uint4_active(surf, 0, 1, 30u + (uint)tid);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    uint4 value = read_layer_uint4_active(surf, 0, 1);
    out[0] = value.x + value.y + value.z + value.w;
  } else {
    out[tid] = 1u + (uint)tid;
  }
}`,
  surfaceInt4VectorActiveLaneReturn: `
__device__ void write_layer_int4_active(cudaSurfaceObject_t surfaceArg, int row, int layer, int base) {
  surf2DLayeredwrite(make_int4(base + 1, base + 2, base + 3, base + 4), surfaceArg, 0, row, layer);
}

__device__ int4 read_layer_int4_active(cudaSurfaceObject_t surfaceArg, int row, int layer) {
  return surf2DLayeredread<int4>(surfaceArg, 0, row, layer);
}

__global__ void surfaceInt4VectorActiveLaneReturn(cudaSurfaceObject_t surf, int *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    write_layer_int4_active(surf, 0, 1, 30 + tid);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    int4 value = read_layer_int4_active(surf, 0, 1);
    out[0] = value.x + value.y + value.z + value.w;
  } else {
    out[tid] = 1 + tid;
  }
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
  textureObjectUint4HelperRead: `
__device__ uint4 read_uint4_tex(cudaTextureObject_t texArg) {
  return tex2D<uint4>(texArg, 0.5f, 0.5f);
}

__global__ void textureObjectUint4HelperRead(cudaTextureObject_t tex, uint4 *out) {
  if (threadIdx.x == 0) {
    out[0] = read_uint4_tex(tex);
  }
}`,
  textureHelperVectorCastCoercion: `
__device__ uint4 read_uint4_for_float(cudaTextureObject_t texArg) {
  return tex2D<uint4>(texArg, 0.5f, 0.5f);
}

__global__ void textureHelperVectorCastCoercion(cudaTextureObject_t tex, float4 *out) {
  if (threadIdx.x == 0) {
    uint4 raw = read_uint4_for_float(tex);
    out[0] = make_float4(raw);
  }
}`,
  textureNestedHelperVectorRead: `
__device__ float4 read_nested_tex_inner(cudaTextureObject_t texArg) {
  return tex2D<float4>(texArg, 0.5f, 0.5f);
}

__device__ float4 read_nested_tex_outer(cudaTextureObject_t texArg) {
  return read_nested_tex_inner(texArg);
}

__global__ void textureNestedHelperVectorRead(cudaTextureObject_t tex, float4 *out) {
  if (threadIdx.x == 0) {
    out[0] = read_nested_tex_outer(tex);
  }
}`,
  textureActiveLaneReturnReadSideEffect: `
__device__ float4 read_return_texture_vec(cudaTextureObject_t texArg) {
  return tex2D<float4>(texArg, 0.5f, 0.5f);
}

__global__ void textureActiveLaneReturnReadSideEffect(cudaTextureObject_t tex, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float4 value = read_return_texture_vec(tex);
    out[tid] = value.x + value.y + value.z + value.w + (float)tid;
    return;
  }
  __syncthreads();
  out[tid] = 1.0f + (float)tid;
}`,
  textureFloat2ActiveLaneStore: `
__device__ float2 read_return_texture_float2(cudaTextureObject_t texArg) {
  return tex2D<float2>(texArg, 0.5f, 0.5f);
}

__global__ void textureFloat2ActiveLaneStore(cudaTextureObject_t tex, float2 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float2 value = read_return_texture_float2(tex);
    out[tid] = make_float2(value.x + (float)tid, value.y + (float)tid);
    return;
  }
  __syncthreads();
  out[tid] = make_float2(1.0f + (float)tid, 10.0f + (float)tid);
}`,
  textureUint2ActiveLaneStore: `
__device__ uint2 read_return_texture_uint2(cudaTextureObject_t texArg) {
  return tex2D<uint2>(texArg, 0.5f, 0.5f);
}

__global__ void textureUint2ActiveLaneStore(cudaTextureObject_t tex, uint2 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    uint2 value = read_return_texture_uint2(tex);
    out[tid] = make_uint2(value.x + (uint)tid, value.y + (uint)tid);
    return;
  }
  __syncthreads();
  out[tid] = make_uint2(1u + (uint)tid, 10u + (uint)tid);
}`,
  textureInt2ActiveLaneStore: `
__device__ int2 read_return_texture_int2(cudaTextureObject_t texArg) {
  return tex2D<int2>(texArg, 0.5f, 0.5f);
}

__global__ void textureInt2ActiveLaneStore(cudaTextureObject_t tex, int2 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    int2 value = read_return_texture_int2(tex);
    out[tid] = make_int2(value.x - tid, value.y - tid);
    return;
  }
  __syncthreads();
  out[tid] = make_int2(1 + tid, -10 - tid);
}`,
  textureFloat3ActiveLaneStore: `
__device__ float3 read_return_texture_float3(cudaTextureObject_t texArg) {
  return tex2D<float3>(texArg, 0.5f, 0.5f);
}

__global__ void textureFloat3ActiveLaneStore(cudaTextureObject_t tex, float3 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float3 value = read_return_texture_float3(tex);
    out[tid] = make_float3(value.x + (float)tid, value.y + (float)tid, value.z + (float)tid);
    return;
  }
  __syncthreads();
  out[tid] = make_float3(1.0f + (float)tid, 10.0f + (float)tid, 100.0f + (float)tid);
}`,
  textureUint3ActiveLaneStore: `
__device__ uint3 read_return_texture_uint3(cudaTextureObject_t texArg) {
  return tex2D<uint3>(texArg, 0.5f, 0.5f);
}

__global__ void textureUint3ActiveLaneStore(cudaTextureObject_t tex, uint3 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    uint3 value = read_return_texture_uint3(tex);
    out[tid] = make_uint3(value.x + (uint)tid, value.y + (uint)tid, value.z + (uint)tid);
    return;
  }
  __syncthreads();
  out[tid] = make_uint3(1u + (uint)tid, 10u + (uint)tid, 100u + (uint)tid);
}`,
  textureInt3ActiveLaneStore: `
__device__ int3 read_return_texture_int3(cudaTextureObject_t texArg) {
  return tex2D<int3>(texArg, 0.5f, 0.5f);
}

__global__ void textureInt3ActiveLaneStore(cudaTextureObject_t tex, int3 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    int3 value = read_return_texture_int3(tex);
    out[tid] = make_int3(value.x - tid, value.y - tid, value.z - tid);
    return;
  }
  __syncthreads();
  out[tid] = make_int3(1 + tid, -10 - tid, 100 + tid);
}`,
  textureFloat4ActiveLaneStore: `
__device__ float4 read_return_texture_float4(cudaTextureObject_t texArg) {
  return tex2D<float4>(texArg, 0.5f, 0.5f);
}

__global__ void textureFloat4ActiveLaneStore(cudaTextureObject_t tex, float4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float4 value = read_return_texture_float4(tex);
    out[tid] = make_float4(value.x + (float)tid, value.y + (float)tid, value.z + (float)tid, value.w + (float)tid);
    return;
  }
  __syncthreads();
  out[tid] = make_float4(1.0f + (float)tid, 10.0f + (float)tid, 20.0f + (float)tid, 30.0f + (float)tid);
}`,
  textureUint4ActiveLaneStore: `
__device__ uint4 read_return_texture_uint4(cudaTextureObject_t texArg) {
  return tex2D<uint4>(texArg, 0.5f, 0.5f);
}

__global__ void textureUint4ActiveLaneStore(cudaTextureObject_t tex, uint4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    uint4 value = read_return_texture_uint4(tex);
    out[tid] = make_uint4(value.x + (uint)tid, value.y + (uint)tid, value.z + (uint)tid, value.w + (uint)tid);
    return;
  }
  __syncthreads();
  out[tid] = make_uint4(1u + (uint)tid, 10u + (uint)tid, 20u + (uint)tid, 30u + (uint)tid);
}`,
  textureInt4ActiveLaneStore: `
__device__ int4 read_return_texture_int4(cudaTextureObject_t texArg) {
  return tex2D<int4>(texArg, 0.5f, 0.5f);
}

__global__ void textureInt4ActiveLaneStore(cudaTextureObject_t tex, int4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    int4 value = read_return_texture_int4(tex);
    out[tid] = make_int4(value.x - tid, value.y - tid, value.z - tid, value.w - tid);
    return;
  }
  __syncthreads();
  out[tid] = make_int4(1 + tid, -10 - tid, 100 + tid, -100 - tid);
}`,
  textureAtlasActiveLaneReturnReadSideEffect: `
__device__ float read_return_texture_atlas(cudaTextureObject_t texArg) {
  float layered = tex2DLayered<float>(texArg, 0.0f, 1.0f, 1.0f);
  float volume = tex3D<float>(texArg, 2.0f, 1.0f, 1.0f);
  return layered + volume;
}

__global__ void textureAtlasActiveLaneReturnReadSideEffect(cudaTextureObject_t tex, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    out[tid] = read_return_texture_atlas(tex) + (float)tid;
    return;
  }
  __syncthreads();
  out[tid] = 1.0f + (float)tid;
}`,
  textureDeepHelperActiveLaneVectorStore: `
__device__ float4 read_deep_texture_leaf(cudaTextureObject_t texArg) {
  return tex2D<float4>(texArg, 0.5f, 0.5f);
}

__device__ float4 read_deep_texture_mid(cudaTextureObject_t texArg) {
  return read_deep_texture_leaf(texArg);
}

__device__ float4 read_deep_texture_outer(cudaTextureObject_t texArg) {
  return read_deep_texture_mid(texArg);
}

__global__ void textureDeepHelperActiveLaneVectorStore(cudaTextureObject_t tex, float4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float4 value = read_deep_texture_outer(tex);
    out[tid] = make_float4(value.x + (float)tid, value.y, value.z, value.w);
    return;
  }
  __syncthreads();
  out[tid] = make_float4(1.0f + (float)tid, 10.0f + (float)tid, 20.0f + (float)tid, 30.0f + (float)tid);
}`,
  textureMixedScalarVectorActiveLaneStore: `
__device__ float read_mixed_texture_scalar(cudaTextureObject_t texArg) {
  return tex2D<float>(texArg, 0.5f, 0.5f);
}

__device__ uint4 read_mixed_texture_vec(cudaTextureObject_t texArg) {
  return tex2D<uint4>(texArg, 0.5f, 0.5f);
}

__global__ void textureMixedScalarVectorActiveLaneStore(cudaTextureObject_t tex, float4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float scalar = read_mixed_texture_scalar(tex);
    uint4 vec = read_mixed_texture_vec(tex);
    out[tid] = make_float4(scalar + (float)tid, (float)vec.y, (float)vec.z, (float)vec.w);
    return;
  }
  __syncthreads();
  out[tid] = make_float4(1.0f + (float)tid, 10.0f + (float)tid, 20.0f + (float)tid, 30.0f + (float)tid);
}`,
  texturePointerAliasActiveLaneStore: `
__device__ float read_alias_texture_scalar(cudaTextureObject_t texArg) {
  return tex2D<float>(texArg, 0.5f, 0.5f);
}

__device__ void write_alias_lane(float *scalarOut, int lane, float value) {
  scalarOut[lane * 4 + 1] = value;
}

__global__ void texturePointerAliasActiveLaneStore(cudaTextureObject_t tex, float4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float *scalarView = reinterpret_cast<float*>(out);
    write_alias_lane(scalarView, tid, read_alias_texture_scalar(tex) + (float)tid);
    return;
  }
  __syncthreads();
  out[tid] = make_float4(1.0f + (float)tid, 10.0f + (float)tid, 20.0f + (float)tid, 30.0f + (float)tid);
}`,
  texturePointerAliasAtomicActiveLaneStore: `
__device__ uint read_alias_texture_uint(cudaTextureObject_t texArg) {
  return tex2D<uint>(texArg, 0.5f, 0.5f);
}

__device__ void atomic_alias_lane(uint *scalarOut, int lane, uint value) {
  atomicAdd(&scalarOut[lane * 4 + 1], value);
}

__global__ void texturePointerAliasAtomicActiveLaneStore(cudaTextureObject_t tex, uint4 *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    uint *scalarView = reinterpret_cast<uint*>(out);
    atomic_alias_lane(scalarView, tid, read_alias_texture_uint(tex) + (uint)tid);
    return;
  }
  __syncthreads();
  out[tid] = make_uint4(1u + (uint)tid, 10u + (uint)tid, 20u + (uint)tid, 30u + (uint)tid);
}`,
  texturePointerAliasAtomicVectorReadback: `
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
}`,
  texturePointerAliasAtomicVectorCompound: `
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
}`,
  textureSurfaceRoundtrip: `
__global__ void textureSurfaceRoundtrip(cudaSurfaceObject_t surf, cudaTextureObject_t tex, float *out) {
  if (threadIdx.x == 0) {
    float4 value = tex2D<float4>(tex, 0.5f, 0.5f);
    out[0] = value.x + value.y;
    surf2Dwrite(value.z + value.w, surf, 0, 0);
  }
}`,
  textureSurfaceVectorHelperRoundtrip: `
__device__ float4 sample_surface_vec(cudaTextureObject_t texArg) {
  return tex2D<float4>(texArg, 0.5f, 0.5f);
}

__device__ void write_surface_vec(cudaSurfaceObject_t surfaceArg, float4 value) {
  surf2Dwrite(value, surfaceArg, 0, 0);
}

__global__ void textureSurfaceVectorHelperRoundtrip(cudaSurfaceObject_t surf, cudaTextureObject_t tex, float *out) {
  if (threadIdx.x == 0) {
    float4 value = sample_surface_vec(tex);
    write_surface_vec(surf, value);
    out[0] = value.x + value.y + value.z + value.w;
  }
}`,
  textureSurfaceActiveLaneReturnSideEffect: `
__device__ float4 sample_return_surface_vec(cudaTextureObject_t texArg) {
  return tex2D<float4>(texArg, 0.5f, 0.5f);
}

__device__ void write_return_surface_vec(cudaSurfaceObject_t surfaceArg, int lane, float4 value) {
  surf2Dwrite(value.x + value.y + value.z + value.w + (float)lane, surfaceArg, lane * sizeof(float), 0);
}

__global__ void textureSurfaceActiveLaneReturnSideEffect(cudaSurfaceObject_t surf, cudaTextureObject_t tex, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float4 value = sample_return_surface_vec(tex);
    write_return_surface_vec(surf, tid, value);
    return;
  }
  __syncthreads();
  surf2Dwrite(1.0f + (float)tid, surf, tid * sizeof(float), 0);
}`,
  textureSurfaceVectorActiveLaneReturn: `
__device__ float4 sample_return_surface_layer_vec(cudaTextureObject_t texArg, int lane) {
  float4 value = tex2D<float4>(texArg, 0.5f, 0.5f);
  return make_float4(value.x + (float)lane, value.y + (float)lane, value.z + (float)lane, value.w + (float)lane);
}

__device__ void write_return_surface_layer_vec(cudaSurfaceObject_t surfaceArg, float4 value, int layer) {
  surf2DLayeredwrite(value, surfaceArg, 0, 0, layer);
}

__global__ void textureSurfaceVectorActiveLaneReturn(cudaSurfaceObject_t surf, cudaTextureObject_t tex, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float4 value = sample_return_surface_layer_vec(tex, tid);
    write_return_surface_layer_vec(surf, value, 1);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    surf2Dwrite(1.0f, surf, 0, 0);
  }
}`,
  textureSurfaceMixedVectorActiveLaneReturn: `
__device__ float sample_return_surface_scalar(cudaTextureObject_t texArg) {
  return tex2D<float>(texArg, 0.5f, 0.5f);
}

__device__ uint4 sample_return_surface_uint4(cudaTextureObject_t texArg) {
  return tex2D<uint4>(texArg, 0.5f, 0.5f);
}

__device__ void write_return_surface_mixed_vec(cudaSurfaceObject_t surfaceArg, float4 value, int layer) {
  surf2DLayeredwrite(value, surfaceArg, 0, 0, layer);
}

__device__ float4 read_return_surface_mixed_vec(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<float4>(surfaceArg, 0, 0, layer);
}

__global__ void textureSurfaceMixedVectorActiveLaneReturn(cudaSurfaceObject_t surf, cudaTextureObject_t tex, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float scalar = sample_return_surface_scalar(tex);
    uint4 vec = sample_return_surface_uint4(tex);
    write_return_surface_mixed_vec(surf, make_float4(scalar + (float)tid, (float)vec.y, (float)vec.z + (float)tid, (float)vec.w), 1);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    float4 value = read_return_surface_mixed_vec(surf, 1);
    out[0] = value.x + value.y + value.z + value.w;
  } else {
    out[tid] = 1.0f + (float)tid;
  }
}`,
  textureSurfaceFloat2VectorActiveLaneReturn: `
__device__ float2 sample_return_surface_float2(cudaTextureObject_t texArg, int lane) {
  float2 value = tex2D<float2>(texArg, 0.5f, 0.5f);
  return make_float2(value.x + (float)lane, value.y + (float)lane);
}

__device__ void write_return_surface_float2(cudaSurfaceObject_t surfaceArg, float2 value, int layer) {
  surf2DLayeredwrite(value, surfaceArg, 0, 0, layer);
}

__device__ float2 read_return_surface_float2(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<float2>(surfaceArg, 0, 0, layer);
}

__global__ void textureSurfaceFloat2VectorActiveLaneReturn(cudaSurfaceObject_t surf, cudaTextureObject_t tex, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float2 value = sample_return_surface_float2(tex, tid);
    write_return_surface_float2(surf, value, 1);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    float2 value = read_return_surface_float2(surf, 1);
    out[0] = value.x + value.y;
  } else {
    out[tid] = 1.0f + (float)tid;
  }
}`,
  textureSurfaceUint2VectorActiveLaneReturn: `
__device__ uint2 sample_return_surface_uint2(cudaTextureObject_t texArg, int lane) {
  uint2 value = tex2D<uint2>(texArg, 0.5f, 0.5f);
  return make_uint2(value.x + (uint)lane, value.y + (uint)lane);
}

__device__ void write_return_surface_uint2(cudaSurfaceObject_t surfaceArg, uint2 value, int layer) {
  surf2DLayeredwrite(value, surfaceArg, 0, 0, layer);
}

__device__ uint2 read_return_surface_uint2(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<uint2>(surfaceArg, 0, 0, layer);
}

__global__ void textureSurfaceUint2VectorActiveLaneReturn(cudaSurfaceObject_t surf, cudaTextureObject_t tex, uint *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    uint2 value = sample_return_surface_uint2(tex, tid);
    write_return_surface_uint2(surf, value, 1);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    uint2 value = read_return_surface_uint2(surf, 1);
    out[0] = value.x + value.y;
  } else {
    out[tid] = 1u + (uint)tid;
  }
}`,
  textureSurfaceInt2VectorActiveLaneReturn: `
__device__ int2 sample_return_surface_int2(cudaTextureObject_t texArg, int lane) {
  int2 value = tex2D<int2>(texArg, 0.5f, 0.5f);
  return make_int2(value.x - lane, value.y - lane);
}

__device__ void write_return_surface_int2(cudaSurfaceObject_t surfaceArg, int2 value, int layer) {
  surf2DLayeredwrite(value, surfaceArg, 0, 0, layer);
}

__device__ int2 read_return_surface_int2(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<int2>(surfaceArg, 0, 0, layer);
}

__global__ void textureSurfaceInt2VectorActiveLaneReturn(cudaSurfaceObject_t surf, cudaTextureObject_t tex, int *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    int2 value = sample_return_surface_int2(tex, tid);
    write_return_surface_int2(surf, value, 1);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    int2 value = read_return_surface_int2(surf, 1);
    out[0] = value.x + value.y;
  } else {
    out[tid] = 1 + tid;
  }
}`,
  textureSurfaceFloat3VectorActiveLaneReturn: `
__device__ float3 sample_return_surface_float3(cudaTextureObject_t texArg, int lane) {
  float3 value = tex2D<float3>(texArg, 0.5f, 0.5f);
  return make_float3(value.x + (float)lane, value.y + (float)lane, value.z + (float)lane);
}

__device__ void write_return_surface_float3(cudaSurfaceObject_t surfaceArg, float3 value, int layer) {
  surf2DLayeredwrite(value, surfaceArg, 0, 0, layer);
}

__device__ float3 read_return_surface_float3(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<float3>(surfaceArg, 0, 0, layer);
}

__global__ void textureSurfaceFloat3VectorActiveLaneReturn(cudaSurfaceObject_t surf, cudaTextureObject_t tex, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float3 value = sample_return_surface_float3(tex, tid);
    write_return_surface_float3(surf, value, 1);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    float3 value = read_return_surface_float3(surf, 1);
    out[0] = value.x + value.y + value.z;
  } else {
    out[tid] = 1.0f + (float)tid;
  }
}`,
  textureSurfaceUint3VectorActiveLaneReturn: `
__device__ uint3 sample_return_surface_uint3(cudaTextureObject_t texArg, int lane) {
  uint3 value = tex2D<uint3>(texArg, 0.5f, 0.5f);
  return make_uint3(value.x + (uint)lane, value.y + (uint)lane, value.z + (uint)lane);
}

__device__ void write_return_surface_uint3(cudaSurfaceObject_t surfaceArg, uint3 value, int layer) {
  surf2DLayeredwrite(value, surfaceArg, 0, 0, layer);
}

__device__ uint3 read_return_surface_uint3(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<uint3>(surfaceArg, 0, 0, layer);
}

__global__ void textureSurfaceUint3VectorActiveLaneReturn(cudaSurfaceObject_t surf, cudaTextureObject_t tex, uint *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    uint3 value = sample_return_surface_uint3(tex, tid);
    write_return_surface_uint3(surf, value, 1);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    uint3 value = read_return_surface_uint3(surf, 1);
    out[0] = value.x + value.y + value.z;
  } else {
    out[tid] = 1u + (uint)tid;
  }
}`,
  textureSurfaceInt3VectorActiveLaneReturn: `
__device__ int3 sample_return_surface_int3(cudaTextureObject_t texArg, int lane) {
  int3 value = tex2D<int3>(texArg, 0.5f, 0.5f);
  return make_int3(value.x - lane, value.y - lane, value.z - lane);
}

__device__ void write_return_surface_int3(cudaSurfaceObject_t surfaceArg, int3 value, int layer) {
  surf2DLayeredwrite(value, surfaceArg, 0, 0, layer);
}

__device__ int3 read_return_surface_int3(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<int3>(surfaceArg, 0, 0, layer);
}

__global__ void textureSurfaceInt3VectorActiveLaneReturn(cudaSurfaceObject_t surf, cudaTextureObject_t tex, int *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    int3 value = sample_return_surface_int3(tex, tid);
    write_return_surface_int3(surf, value, 1);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    int3 value = read_return_surface_int3(surf, 1);
    out[0] = value.x + value.y + value.z;
  } else {
    out[tid] = 1 + tid;
  }
}`,
  textureSurfaceFloat4VectorActiveLaneReturn: `
__device__ float4 sample_return_surface_float4(cudaTextureObject_t texArg, int lane) {
  float4 value = tex2D<float4>(texArg, 0.5f, 0.5f);
  return make_float4(value.x + (float)lane, value.y + (float)lane, value.z + (float)lane, value.w + (float)lane);
}

__device__ void write_return_surface_float4(cudaSurfaceObject_t surfaceArg, float4 value, int layer) {
  surf2DLayeredwrite(value, surfaceArg, 0, 0, layer);
}

__device__ float4 read_return_surface_float4(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<float4>(surfaceArg, 0, 0, layer);
}

__global__ void textureSurfaceFloat4VectorActiveLaneReturn(cudaSurfaceObject_t surf, cudaTextureObject_t tex, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float4 value = sample_return_surface_float4(tex, tid);
    write_return_surface_float4(surf, value, 1);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    float4 value = read_return_surface_float4(surf, 1);
    out[0] = value.x + value.y + value.z + value.w;
  } else {
    out[tid] = 1.0f + (float)tid;
  }
}`,
  textureSurfaceUint4VectorActiveLaneReturn: `
__device__ uint4 sample_return_surface_uint4(cudaTextureObject_t texArg, int lane) {
  uint4 value = tex2D<uint4>(texArg, 0.5f, 0.5f);
  return make_uint4(value.x + (uint)lane, value.y + (uint)lane, value.z + (uint)lane, value.w + (uint)lane);
}

__device__ void write_return_surface_uint4(cudaSurfaceObject_t surfaceArg, uint4 value, int layer) {
  surf2DLayeredwrite(value, surfaceArg, 0, 0, layer);
}

__device__ uint4 read_return_surface_uint4(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<uint4>(surfaceArg, 0, 0, layer);
}

__global__ void textureSurfaceUint4VectorActiveLaneReturn(cudaSurfaceObject_t surf, cudaTextureObject_t tex, uint *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    uint4 value = sample_return_surface_uint4(tex, tid);
    write_return_surface_uint4(surf, value, 1);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    uint4 value = read_return_surface_uint4(surf, 1);
    out[0] = value.x + value.y + value.z + value.w;
  } else {
    out[tid] = 1u + (uint)tid;
  }
}`,
  textureSurfaceInt4VectorActiveLaneReturn: `
__device__ int4 sample_return_surface_int4(cudaTextureObject_t texArg, int lane) {
  int4 value = tex2D<int4>(texArg, 0.5f, 0.5f);
  return make_int4(value.x - lane, value.y - lane, value.z - lane, value.w - lane);
}

__device__ void write_return_surface_int4(cudaSurfaceObject_t surfaceArg, int4 value, int layer) {
  surf2DLayeredwrite(value, surfaceArg, 0, 0, layer);
}

__device__ int4 read_return_surface_int4(cudaSurfaceObject_t surfaceArg, int layer) {
  return surf2DLayeredread<int4>(surfaceArg, 0, 0, layer);
}

__global__ void textureSurfaceInt4VectorActiveLaneReturn(cudaSurfaceObject_t surf, cudaTextureObject_t tex, int *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    int4 value = sample_return_surface_int4(tex, tid);
    write_return_surface_int4(surf, value, 1);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    int4 value = read_return_surface_int4(surf, 1);
    out[0] = value.x + value.y + value.z + value.w;
  } else {
    out[tid] = 1 + tid;
  }
}`,
  textureSurfaceVolumeVectorActiveLaneReturn: `
__device__ float4 sample_return_volume_vec(cudaTextureObject_t texArg, int lane) {
  float4 layered = tex2DLayered<float4>(texArg, 0.0f, 1.0f, 1.0f);
  float4 volume = tex3D<float4>(texArg, 2.0f, 1.0f, 1.0f);
  return make_float4(
    layered.x + volume.x + (float)lane,
    layered.y + volume.y + (float)lane,
    layered.z + volume.z + (float)lane,
    layered.w + volume.w + (float)lane
  );
}

__device__ void write_return_surface_volume_vec(cudaSurfaceObject_t surfaceArg, float4 value) {
  surf3Dwrite(value, surfaceArg, 0, 0, 1);
}

__global__ void textureSurfaceVolumeVectorActiveLaneReturn(cudaSurfaceObject_t surf, cudaTextureObject_t tex, float *out, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    float4 value = sample_return_volume_vec(tex, tid);
    write_return_surface_volume_vec(surf, value);
    return;
  }
  __syncthreads();
  if (tid == 0) {
    float4 value;
    surf3Dread(&value, surf, 0, 0, 1);
    out[0] = value.x + value.y + value.z + value.w;
  } else {
    out[tid] = 1.0f + (float)tid;
  }
}`,
  surfaceHelperDispatchMultipleSurfaces: `
__device__ float read_surface_alias(cudaSurfaceObject_t surfaceArg) {
  float value = 0.0f;
  surf2Dread(&value, surfaceArg, 0, 0);
  return value;
}

__device__ void write_surface_alias(cudaSurfaceObject_t surfaceArg, float value) {
  surf2Dwrite(value, surfaceArg, 4, 0);
}

__global__ void surfaceHelperDispatchMultipleSurfaces(cudaSurfaceObject_t first, cudaSurfaceObject_t second) {
  if (threadIdx.x == 0) {
    float value = read_surface_alias(first);
    write_surface_alias(second, value + 7.0f);
  }
}`,
  surfaceVectorRead: `
__global__ void surfaceVectorRead(cudaSurfaceObject_t surf, float *out) {
  if (threadIdx.x == 0) {
    float4 value;
    surf2Dread(&value, surf, 0, 0);
    out[0] = value.x;
    out[1] = value.y;
    out[2] = value.z;
    out[3] = value.w;
  }
}`,
  surfaceHelperVectorReadMultipleSurfaces: `
__device__ float4 read_surface_vec_pointer(cudaSurfaceObject_t surfaceArg) {
  float4 value;
  surf2Dread(&value, surfaceArg, 0, 0);
  return value;
}

__device__ float4 read_surface_vec_return(cudaSurfaceObject_t surfaceArg) {
  return surf2Dread<float4>(surfaceArg, 0, 0);
}

__global__ void surfaceHelperVectorReadMultipleSurfaces(cudaSurfaceObject_t first, cudaSurfaceObject_t second, float *out) {
  if (threadIdx.x == 0) {
    float4 firstValue = read_surface_vec_pointer(first);
    float4 secondValue = read_surface_vec_return(second);
    out[0] = firstValue.x;
    out[1] = firstValue.y;
    out[2] = firstValue.z;
    out[3] = firstValue.w;
    out[4] = secondValue.x;
    out[5] = secondValue.y;
    out[6] = secondValue.z;
    out[7] = secondValue.w;
  }
}`,
  surfaceActiveLaneReturnSideEffect: `
__global__ void surfaceActiveLaneReturnSideEffect(cudaSurfaceObject_t surf, int N) {
  int tid = threadIdx.x;
  if (tid >= N) {
    surf2Dwrite(100.0f + (float)tid, surf, tid * sizeof(float), 0);
    return;
  }
  __syncthreads();
  surf2Dwrite(1.0f + (float)tid, surf, tid * sizeof(float), 0);
}`,
  reciprocalIntrinsic: `
__global__ void reciprocalIntrinsic(float *x, float *out) {
  int idx = threadIdx.x;
  if (idx < 4) {
    out[idx] = __frcp_rn(x[idx] + 1.0f);
  }
}`,
  ...loadCorpusExecutionSources(root, corpusExecutionFixtures),
};
const allAutoCorpusSmokeFixtures = effectiveAutoCorpusSmokeLimit > 0
  ? loadAutoCorpusSmokeFixtures(root, effectiveAutoCorpusSmokeLimit, await import(pathToFileURL(path.join(root, "packages/browsergrad-compiler/dist/index.js")).href), {
      verifyMode: autoCorpusSmokeMode,
      profile: autoCorpusSmokeProfile,
      allowedRequiredFeatures: autoCorpusSmokeFeatures,
      cache: autoCorpusSmokeCache,
      referencePreflight: autoCorpusSmokeReferencePreflight,
    })
  : [];
const autoCorpusSmokeFixtures = applyAutoCorpusSmokeShard(allAutoCorpusSmokeFixtures, autoCorpusSmokeShard);
for (const fixture of autoCorpusSmokeFixtures) {
  sources[fixture.sourceKey] = fixture.source;
}
const expectedCorpusFixtureNames = corpusExecutionFixtures.map((fixture) => fixture.caseName);

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
        prepareWgslKernelProgramSequence,
        readWgslStorageBuffer,
        runWgslKernelProgramSequence,
      } from "@unlocalhosted/browsergrad-kernels";
      import {
        compileCudaLiteKernelForWebGpu,
        compileCudaLiteKernel,
        createCudaWebGpuExecutionPlan,
        normalizeCudaWebGpuReadback,
        prepareCompiledKernelWebGpu,
        runCompiledKernelReference,
        runCompiledKernelWebGpu,
      } from "@unlocalhosted/browsergrad-compiler";

      const SOURCES = ${JSON.stringify(sources)};
      const CORPUS_FIXTURES = ${JSON.stringify(corpusExecutionFixtures)};
      const AUTO_CORPUS_SMOKE_FIXTURES = ${JSON.stringify(autoCorpusSmokeFixtures.map(({ source, ...fixture }) => fixture))};
      const EXPECTED_CORPUS_FIXTURE_NAMES = ${JSON.stringify(expectedCorpusFixtureNames)};
      const CORPUS_FIXTURE_BASELINE = ${JSON.stringify(cudaLiteCorpusExecutionFixtureBaseline)};
      const CASE_FILTERS = ${JSON.stringify(caseFilters)};
      const CASE_TIMEOUT_MS = ${caseTimeoutMs};
      const WARMUP = ${warmup};
      const REPEAT = ${repeat};
      const COMPILE_ONLY = ${JSON.stringify(compileOnly)};
      const PROGRESS = ${JSON.stringify(progress)};
      const FAIL_FAST = ${JSON.stringify(failFast)};
      const PROFILE_CASE = ${JSON.stringify(profileCase ?? "")};
      const ONLY_AUTO_CORPUS_SMOKE = ${JSON.stringify(onlyAutoCorpusSmoke)};
      const ONLY_CORPUS_FIXTURES = ${JSON.stringify(onlyCorpusFixtures)};
      const DEVICE_RECYCLE_INTERVAL = 128;

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
        const warmupCases = [];

        const specs = filterCaseSpecs(caseSpecs());
        let stop = false;
        for (let warmupIndex = 0; warmupIndex < WARMUP; warmupIndex++) {
          for (let index = 0; index < specs.length; index++) {
            const spec = specs[index];
            const result = await runCaseOnce(device, spec, warmupIndex + 1, index + 1, specs.length, "warmup");
            warmupCases.push(result.caseResult);
            if (result.recreateDevice) {
              device.gpu.destroy();
              device = await createE2eKernelDevice();
            }
            if (!result.caseResult.ok && FAIL_FAST) {
              stop = true;
              break;
            }
          }
          if (stop) break;
        }
        const totalCaseRuns = specs.length * REPEAT;
        for (let repeatIndex = 0; repeatIndex < REPEAT && !stop; repeatIndex++) {
          for (let index = 0; index < specs.length; index++) {
            const spec = specs[index];
            const runIndex = repeatIndex * specs.length + index + 1;
            const result = await runCaseOnce(device, spec, repeatIndex + 1, runIndex, totalCaseRuns, "case");
            cases.push(result.caseResult);
            if (result.recreateDevice) {
              device.gpu.destroy();
              device = await createE2eKernelDevice();
            }
            const completed = result.caseResult;
            if (!completed.ok && FAIL_FAST) {
              stop = true;
              break;
            }
            if (shouldRecycleDevice(runIndex, totalCaseRuns)) {
              emitProgress({ event: "device-recycle", index: runIndex, total: totalCaseRuns });
              device.gpu.destroy();
              device = await createE2eKernelDevice();
            }
          }
        }

        if (!COMPILE_ONLY && cases.every((item) => item.ok) && shouldRunPreparedSmoke()) {
          emitProgress({ event: "case-start", index: specs.length + 1, total: specs.length + 1, name: "prepared-resident-saxpy" });
          const preparedStart = performance.now();
          cases.push({
            name: "prepared-resident-saxpy",
            stage: "prepared_dispatch",
            plan: "prepared:single-dispatch",
            ...(await runPreparedResidentSaxpy(device)),
            output: "y",
            ms: round(performance.now() - preparedStart),
          });
          emitProgress({ event: "case-pass", index: specs.length + 1, total: specs.length + 1, name: "prepared-resident-saxpy", ms: cases[cases.length - 1].ms });
        }

        const failed = cases.filter((item) => !item.ok);
        const warmupFailed = warmupCases.filter((item) => !item.ok);
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
          failedByStage: countByStage(failed),
          autoCorpusSmokeCases: autoCorpusSmokeCases.length,
          autoCorpusSmokePassed: autoCorpusSmokePassed.length,
          autoCorpusSmokeSkipped: autoCorpusSmokeSkipped.length,
          autoCorpusSmokeCovered,
          autoCorpusSmokeFailed: autoCorpusSmokeCases.filter((item) => !item.ok).length,
          autoCorpusSmokeCasesByCorpus: countByCorpus(autoCorpusSmokeCases),
          autoCorpusSmokePassedByCorpus: countByCorpus(autoCorpusSmokePassed),
          autoCorpusSmokeSkippedByCorpus: countByCorpus(autoCorpusSmokeSkipped),
          autoCorpusSmokeLimit: ${effectiveAutoCorpusSmokeLimit},
          autoCorpusSmokeExpectedCovered: ${autoCorpusSmokeFixtures.length},
          autoCorpusSmokeMode: ${JSON.stringify(autoCorpusSmokeMode)},
          autoCorpusSmokeProfile: ${JSON.stringify(autoCorpusSmokeProfile)},
          compileOnly: COMPILE_ONLY,
          warmup: WARMUP,
          warmupCases: warmupCases.length,
          warmupFailed: warmupFailed.length,
          warmupFailedByStage: countByStage(warmupFailed),
          repeat: REPEAT,
          autoCorpusSmokeShard: ${JSON.stringify(autoCorpusSmokeShard)},
          autoCorpusSmokeOnly: ${JSON.stringify(onlyAutoCorpusSmoke)},
          corpusFixtureBaseline: CORPUS_FIXTURE_BASELINE,
          expectedCorpusFixtureNames,
          missingCorpusFixtureNames: expectedCorpusFixtureNames.filter((name) => !loadedCorpusFixtureNames.has(name)),
          passed: cases.length - failed.length - skipped.length,
          failed: failed.length,
          skipped: skipped.length,
        };
      };

      async function runCaseOnce(device, spec, repeatValue, index, total, phase) {
        const missingFeatures = missingSpecFeatures(device, spec);
        if (missingFeatures.length > 0) {
          const caseResult = {
            name: spec.name,
            repeat: repeatValue,
            phase,
            stage: "feature_check",
            plan: "missing-features:" + missingFeatures.join(","),
            ok: false,
            maxAbsDiff: Number.POSITIVE_INFINITY,
            output: spec.output,
            ...(spec.corpusId === undefined ? {} : { corpusId: spec.corpusId }),
          };
          emitProgress({ event: "case-fail", phase, index, total, name: spec.name, repeat: repeatValue, missingFeatures });
          return { caseResult, recreateDevice: false };
        }
        emitProgress({ event: "case-start", phase, index, total, name: spec.name, repeat: repeatValue });
        const start = performance.now();
        try {
          const result = await withCaseTimeout(COMPILE_ONLY ? runCompileOnlyWebGpuCase(device, spec) : runReferenceWebGpuCase(device, spec), spec.name);
          const caseResult = {
            name: spec.name,
            repeat: repeatValue,
            phase,
            stage: result.stage ?? "unknown",
            plan: result.plan,
            ok: result.ok,
            ...(result.skipped ? { skipped: true } : {}),
            maxAbsDiff: result.maxAbsDiff,
            ...(spec.tolerance === undefined ? {} : { tolerance: spec.tolerance }),
            ...(result.firstDiff === undefined ? {} : { firstDiff: result.firstDiff }),
            output: result.output ?? spec.output,
            ...(spec.corpusId === undefined ? {} : { corpusId: spec.corpusId }),
            ...(spec.expectedOutput === undefined ? {} : { expectedOutputPinned: true }),
            ...(result.profile === undefined ? {} : { profile: result.profile }),
            ms: round(performance.now() - start),
          };
          emitProgress(caseResult.ok
            ? { event: "case-pass", phase, index, total, name: spec.name, repeat: repeatValue, ms: caseResult.ms }
            : { event: "case-fail", phase, index, total, name: spec.name, repeat: repeatValue, ms: caseResult.ms, plan: caseResult.plan });
          return { caseResult, recreateDevice: false };
        } catch (error) {
          const caseResult = {
            name: spec.name,
            repeat: repeatValue,
            phase,
            stage: "threw",
            plan: "threw",
            ok: false,
            maxAbsDiff: Number.POSITIVE_INFINITY,
            output: spec.output,
            error: serializeError(error),
            ...(spec.corpusId === undefined ? {} : { corpusId: spec.corpusId }),
            ms: round(performance.now() - start),
          };
          emitProgress({ event: "case-fail", phase, index, total, name: spec.name, repeat: repeatValue, ms: caseResult.ms, error: serializeError(error) });
          return { caseResult, recreateDevice: isCaseTimeout(error) };
        }
      }

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
        requestLimit("maxComputeWorkgroupStorageSize", 16384, 32768);
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

      function shouldRecycleDevice(completed, total) {
        return completed > 0 && completed < total && completed % DEVICE_RECYCLE_INTERVAL === 0;
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
        if (ONLY_AUTO_CORPUS_SMOKE || ONLY_CORPUS_FIXTURES) return false;
        return CASE_FILTERS.length === 0 ||
          CASE_FILTERS.some((filter) => "prepared-resident-saxpy" === filter || "prepared-resident-saxpy".includes(filter));
      }

      function caseSpecs() {
        if (ONLY_AUTO_CORPUS_SMOKE) return autoCorpusSmokeCaseSpecs();
        if (ONLY_CORPUS_FIXTURES) return corpusFixtureCaseSpecs();
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
            name: "runtime:host-dynamic-pointer-offset",
            source: SOURCES.dynamicPointerOffsetLaunch,
            options: { kernelName: "parent", workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array([10, 20, 30, 40]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [10, 21, 32, 40] },
          },
          {
            name: "runtime:host-dynamic-ordered-launches",
            source: SOURCES.dynamicOrderedLaunch,
            options: { kernelName: "parent", workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array([1, 2, 3, 4, 50]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [4, 9, 11, 13, 53] },
          },
          {
            name: "runtime:host-dynamic-child-peer-copy",
            source: SOURCES.dynamicChildPeerCopy,
            options: { kernelName: "parent", workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                dst: new Float32Array([0, 0, 0, 0]),
                src: new Float32Array([2.5, 3.5]),
              },
              scalars: { n: 2 },
            }),
            output: "dst",
            expectedOutput: { type: "Float32Array", data: [0, 2.5, 3.5, 0] },
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
            name: "runtime:pool-alias-host-dynamic-launch",
            source: SOURCES.dynamicPoolAliasLaunch,
            options: { kernelName: "parent", workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: { out: new Float32Array([0]) },
              memoryPools: {
                pool: { data: new Uint32Array(1), offset: new Uint32Array([0]) },
              },
              readback: ["pool", "pool_offset"],
            }),
            output: "pool",
            offsetOutput: "pool_offset",
            expectedOutput: { type: "Uint32Array", data: [1086324736] },
            expectedOffset: 4,
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
            name: "storage:local-pointer-row-alias",
            source: SOURCES.localStoragePointerAliases,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(6),
                inp: new Float32Array([1, 2, 3, 4, 5, 6]),
              },
              scalars: { rows: 2, cols: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [2, 3, 4, 5, 6, 7] },
          },
          {
            name: "storage:vector-pointer-memory-view",
            source: SOURCES.vectorPointerMemoryView,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
                inp: new Float32Array([1, 2, 3, 4]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [1, 12, 3, 4] },
          },
          {
            name: "storage:vector-cast-dynamic-base-read",
            source: SOURCES.vectorCastDynamicBaseRead,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
                inp: new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [1, 2, 3, 4, 11, 21, 31, 41] },
          },
          {
            name: "helpers:vector-cache-hint-dynamic-read",
            source: SOURCES.vectorCacheHintDynamicRead,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
                inp: new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [11, 22, 33, 44, 20, 40, 60, 80] },
          },
          {
            name: "helpers:vector-cache-hint-dynamic-write",
            source: SOURCES.vectorCacheHintDynamicWrite,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
                inp: new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [101, 202, 303, 404, 110, 220, 330, 440] },
          },
          {
            name: "helpers:vector-cache-hint-device-helper",
            source: SOURCES.vectorCacheHintDeviceHelper,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
                inp: new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [1001, 2002, 3003, 4004, 1010, 2020, 3030, 4040] },
          },
          {
            name: "helpers:vector-cache-hint-lane-pointer",
            source: SOURCES.vectorCacheHintLanePointer,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
                inp: new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [0, 53, 0, 0, 0, 80, 0, 0] },
          },
          {
            name: "helpers:vector-lane-pointer-offset-helper",
            source: SOURCES.vectorLanePointerOffsetHelper,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(12),
                inp: new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [0, 0, 0, 0, 0, 5, 0, 0, 0, 50, 0, 0] },
          },
          {
            name: "helpers:vector-to-scalar-pointer-offset-helper",
            source: SOURCES.vectorToScalarPointerOffsetHelper,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(2),
                inp: new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [10, 20] },
          },
          {
            name: "helpers:vector-to-scalar-pointer-write-offset-helper",
            source: SOURCES.vectorToScalarPointerWriteOffsetHelper,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [0, 0, 0, 0, 7, 8, 0, 0] },
          },
          {
            name: "helpers:vector-scalar-vector-alias-roundtrip",
            source: SOURCES.vectorScalarVectorAliasRoundtrip,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array([
                  1, 2, 3, 4,
                  10, 20, 30, 40,
                  100, 200, 300, 400,
                ]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [
              1, 2, 3, 4,
              11, 22, 33, 44,
              102, 204, 306, 408,
            ] },
          },
          {
            name: "helpers:vector-to-scalar-pointer-atomic-offset-helper",
            source: SOURCES.vectorToScalarPointerAtomicOffsetHelper,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [0, 0, 0, 0, 2, 3, 0, 0] },
          },
          {
            name: "helpers:uint-vector-to-scalar-pointer-atomic-offset-helper",
            source: SOURCES.uintVectorToScalarPointerAtomicOffsetHelper,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(8),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [0, 0, 0, 0, 2, 3, 0, 0] },
          },
          {
            name: "helpers:int-vector-to-scalar-pointer-atomic-offset-helper",
            source: SOURCES.intVectorToScalarPointerAtomicOffsetHelper,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array([0, 0, 0, 0, 10, 10, 0, 0]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [0, 0, 0, 0, 8, 7, 0, 0] },
          },
          {
            name: "helpers:float3-vector-to-scalar-pointer-atomic-offset-helper",
            source: SOURCES.float3VectorToScalarPointerAtomicOffsetHelper,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(6),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [0, 0, 0, 2, 3, 0] },
          },
          {
            name: "helpers:uint3-vector-scalar-exchange-cas",
            source: SOURCES.uint3VectorScalarExchangeCas,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                data: new Uint32Array([0, 0, 0, 3, 4, 5]),
                report: new Uint32Array(4),
              },
            }),
            output: "report",
            expectedOutput: { type: "Uint32Array", data: [3, 4, 9, 10] },
          },
          {
            name: "helpers:float3-vector-scalar-min-max",
            source: SOURCES.float3VectorScalarMinMax,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                data: new Float32Array([0, 0, 0, 4, 6, 5]),
                report: new Float32Array(4),
              },
            }),
            output: "report",
            expectedOutput: { type: "Float32Array", data: [4, 6, 3, 8] },
          },
          {
            name: "helpers:shared-float3-vector-to-scalar-atomic",
            source: SOURCES.sharedFloat3VectorToScalarAtomic,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(2),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [7, 8] },
          },
          {
            name: "helpers:shared-int3-vector-to-scalar-atomic",
            source: SOURCES.sharedInt3VectorToScalarAtomic,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array(2),
              },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [17, 17] },
          },
          {
            name: "helpers:device-global-vector-to-scalar-atomic",
            source: SOURCES.deviceGlobalVectorToScalarAtomic,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(2),
              },
              deviceGlobals: {
                g_vec: new Uint32Array(8),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [5, 6] },
          },
          {
            name: "helpers:device-global-float3-vector-to-scalar-atomic",
            source: SOURCES.deviceGlobalFloat3VectorToScalarAtomic,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(2),
              },
              deviceGlobals: {
                g_f3: new Float32Array(6),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [11, 12] },
          },
          {
            name: "helpers:device-global-vector-pointer-array",
            source: SOURCES.deviceGlobalVectorPointerArray,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
              },
              deviceGlobals: {
                g_ptr_values: new Float32Array(9),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [17, 19, 23, 1, 26, 33, 41, 0] },
          },
          {
            name: "storage:cross-space-vector-alias-consistency",
            source: SOURCES.crossSpaceVectorAliasConsistency,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(3),
              },
              deviceGlobals: {
                g_alias_vec: new Float32Array(8),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [7.5, 61.5, 802.5] },
          },
          {
            name: "helpers:shared-vector-to-scalar-atomic",
            source: SOURCES.sharedVectorToScalarAtomic,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(2),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [7, 8] },
          },
          {
            name: "helpers:uint-vector-scalar-exchange-cas",
            source: SOURCES.uintVectorScalarExchangeCas,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                data: new Uint32Array([0, 0, 0, 0, 3, 4, 5, 6]),
                report: new Uint32Array(4),
              },
            }),
            output: "report",
            expectedOutput: { type: "Uint32Array", data: [3, 4, 9, 10] },
          },
          {
            name: "helpers:float-vector-scalar-min-max",
            source: SOURCES.floatVectorScalarMinMax,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                data: new Float32Array([0, 0, 0, 0, 4, 6, 5, 6]),
                report: new Float32Array(4),
              },
            }),
            output: "report",
            expectedOutput: { type: "Float32Array", data: [4, 6, 3, 8] },
          },
          {
            name: "helpers:shared-vector-scalar-exchange-cas",
            source: SOURCES.sharedVectorScalarExchangeCas,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                report: new Uint32Array(4),
              },
            }),
            output: "report",
            expectedOutput: { type: "Uint32Array", data: [3, 4, 9, 10] },
          },
          {
            name: "storage:vector-deref-lane-write",
            source: SOURCES.vectorDerefLaneWrite,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([2, 4, 0, 8]),
                out: new Float32Array(1),
              },
            }),
            output: "x",
            expectedOutput: { type: "Float32Array", data: [2, 4, 6, 8] },
          },
          {
            name: "storage:local-vector-lane-compound",
            source: SOURCES.localVectorLaneCompound,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(2),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [7, 9] },
          },
          {
            name: "storage:shared-vector-overlay",
            source: SOURCES.sharedVectorOverlay,
            options: { workgroupSize: [1, 1, 1], dynamicSharedMemory: { params: 4 } },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(1),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [10] },
          },
          {
            name: "storage:local-vector-pointer-array",
            source: SOURCES.localVectorPointerArray,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [1, 2, 3, 1, 12, 15, 18, 0] },
          },
          {
            name: "storage:shared-vector-pointer-array",
            source: SOURCES.sharedVectorPointerArray,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [10, 20, 30, 1, 111, 222, 333, 0] },
          },
          {
            name: "storage:dynamic-shared-vector-pointer-array",
            source: SOURCES.dynamicSharedVectorPointerArray,
            options: { workgroupSize: [1, 1, 1], dynamicSharedMemory: { scratch: 9 } },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [3, 5, 7, 1, 33, 41, 53, 0] },
          },
          {
            name: "storage:dynamic-shared-vector-alias-chain-pointer-array",
            source: SOURCES.dynamicSharedVectorAliasChainPointerArray,
            options: { workgroupSize: [1, 1, 1], dynamicSharedMemory: { scratch: 12 } },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [17, 19, 23, 1, 53, 61, 73, 0] },
          },
          {
            name: "storage:local-array-vector-scalar-roundtrip",
            source: SOURCES.localArrayVectorScalarRoundtrip,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [5, 16, 27, 38, 6, 16, 27, 38] },
          },
          {
            name: "storage:shared-vector-helper",
            source: SOURCES.sharedVectorHelper,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]),
                out: new Float32Array(8),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [11, 22, 33, 44, 2, 4, 6, 8] },
          },
          {
            name: "storage:guarded-shared-vector-lanes",
            source: SOURCES.guardedSharedVectorLanes,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([
                  1, 2, 30, 4,
                  10, 20, 300, 40,
                  100, 200, 3000, 400,
                  1000, 2000, 30000, 4000,
                ]),
                out: new Float32Array(16),
              },
              scalars: { n: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [
              1, 2, 3, 4,
              10, 20, 30, 41,
              100, 200, 300, 402,
              0, 0, 0, 0,
            ] },
          },
          {
            name: "storage:guarded-shared-float3-lanes",
            source: SOURCES.guardedSharedFloat3Lanes,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(12),
              },
              scalars: { n: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [
              1, 101, 100,
              2, 103, 101,
              3, 105, 102,
              0, 0, 0,
            ] },
          },
          {
            name: "storage:dynamic-shared-float3-view",
            source: SOURCES.dynamicSharedFloat3View,
            options: { workgroupSize: [2, 1, 1], dynamicSharedMemory: { scratch: 6 } },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(6),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [
              1.5, 11, 101.5,
              3.5, 14, 105.5,
            ] },
          },
          {
            name: "helpers:dynamic-shared-float3-scalar-atomic",
            source: SOURCES.dynamicSharedFloat3ScalarAtomic,
            options: { workgroupSize: [2, 1, 1], dynamicSharedMemory: { scratch: 6 } },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(2),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [2.5, 12.5] },
          },
          {
            name: "atomic:dynamic-shared-vector-scalar-pointer-array",
            source: SOURCES.dynamicSharedVectorScalarPointerArrayAtomic,
            options: { workgroupSize: [1, 1, 1], dynamicSharedMemory: { scratch: 12 } },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [11, 37, 11.5, 38.5] },
          },
          {
            name: "control:active-lane-loop-barrier",
            source: SOURCES.earlyReturnLoopBarrier,
            options: { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4]),
              },
              scalars: { N: 3 },
            }),
            output: "x",
            expectedOutput: { type: "Float32Array", data: [3, 5, 7, 4] },
          },
          {
            name: "control:active-lane-loop-internal-return-barrier",
            source: SOURCES.activeLaneLoopInternalReturnBarrier,
            options: { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
              },
              scalars: { N: 6 },
            }),
            output: "x",
            expectedOutput: { type: "Float32Array", data: [2, 3, 4, 5, 7, 8, 7, 8] },
          },
          {
            name: "control:active-lane-alternate-return-barrier",
            source: SOURCES.activeLaneAlternateReturnBarrier,
            options: { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4]),
              },
              scalars: { N: 3 },
            }),
            output: "x",
            expectedOutput: { type: "Float32Array", data: [2, 3, 4, 4] },
          },
          {
            name: "control:active-lane-nested-return-barrier",
            source: SOURCES.activeLaneNestedReturnBarrier,
            options: { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4]),
              },
              scalars: { N: 3 },
            }),
            output: "x",
            expectedOutput: { type: "Float32Array", data: [2, 3, 3, 4] },
          },
          {
            name: "control:active-lane-loop-alternate-return-barrier",
            source: SOURCES.activeLaneLoopAlternateReturnBarrier,
            options: { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
              },
              scalars: { N: 6 },
            }),
            output: "x",
            expectedOutput: { type: "Float32Array", data: [2, 3, 4, 5, 7, 8, 7, 8] },
          },
          {
            name: "control:active-lane-loop-return-side-effect-barrier",
            source: SOURCES.activeLaneLoopReturnSideEffectBarrier,
            options: { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
              },
              scalars: { N: 6 },
            }),
            output: "x",
            expectedOutput: { type: "Float32Array", data: [2, 3, -12, -13, 7, 8, 7, 8] },
          },
          {
            name: "control:active-lane-vector-return-side-effect-barrier",
            source: SOURCES.activeLaneVectorReturnSideEffectBarrier,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array([
                  1, 2, 3, 4,
                  5, 6, 7, 8,
                  9, 10, 11, 12,
                  13, 14, 15, 16,
                  17, 18, 19, 20,
                  21, 22, 23, 24,
                  25, 26, 27, 28,
                  29, 30, 31, 32,
                ]),
              },
              scalars: { N: 6 },
            }),
            output: "out",
            expectedOutput: {
              type: "Float32Array",
              data: [
                2, 4, 6, 8,
                6, 8, 10, 12,
                10, 12, 14, -12,
                14, 16, 18, -13,
                18, 20, 22, 24,
                22, 24, 26, 28,
                25, 26, 27, 28,
                29, 30, 31, 32,
              ],
            },
          },
          {
            name: "control:active-lane-pointer-alias-return-side-effect-barrier",
            source: SOURCES.activeLanePointerAliasReturnSideEffectBarrier,
            options: { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
              },
              scalars: { N: 6 },
            }),
            output: "x",
            expectedOutput: { type: "Float32Array", data: [2, 3, -22, -23, 7, 8, 7, 8] },
          },
          {
            name: "control:active-lane-atomic-return-side-effect-barrier",
            source: SOURCES.activeLaneAtomicReturnSideEffectBarrier,
            options: { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                counter: new Uint32Array([0]),
                out: new Uint32Array(8),
              },
              scalars: { N: 6 },
            }),
            output: "counter",
            expectedOutput: { type: "Uint32Array", data: [2] },
          },
          {
            name: "control:active-lane-shared-return-side-effect-barrier",
            source: SOURCES.activeLaneSharedReturnSideEffectBarrier,
            options: { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(4),
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [0, 1, 105, 0] },
          },
          {
            name: "control:active-lane-guarded-rhs",
            source: SOURCES.activeLaneGuardedRhs,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                counter: new Int32Array([0, 0]),
              },
              scalars: { n: 2 },
            }),
            output: "counter",
            expectedOutput: { type: "Int32Array", data: [2, 2] },
          },
          {
            name: "control:active-lane-assignment-guarded-rhs",
            source: SOURCES.activeLaneAssignmentGuardedRhs,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                counter: new Int32Array([0, 0]),
                sink: new Int32Array(4),
              },
              scalars: { n: 2 },
            }),
            output: "counter",
            expectedOutput: { type: "Int32Array", data: [2, 2] },
          },
          {
            name: "control:active-lane-vector-atomic-guarded-rhs",
            source: SOURCES.activeLaneVectorAtomicGuardedRhs,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                data: new Float32Array([0, 0, 0, 4, 8, 16]),
                out: new Float32Array(3),
              },
              scalars: { n: 2 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [5.25, 10.25, 16] },
          },
          {
            name: "control:active-lane-compound-assignment-guarded-rhs",
            source: SOURCES.activeLaneCompoundAssignmentGuardedRhs,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                counter: new Int32Array([0, 0]),
                sink: new Int32Array([10, 10, 10, 10]),
              },
              scalars: { n: 2 },
            }),
            output: "counter",
            expectedOutput: { type: "Int32Array", data: [2, 2] },
          },
          {
            name: "control:active-lane-uniform-break-barrier",
            source: SOURCES.activeLaneUniformBreakBarrier,
            options: { workgroupSize: [4, 1, 1], dynamicSharedMemory: { scratch: 4 } },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4]),
              },
              scalars: { N: 3 },
            }),
            output: "x",
            expectedOutput: { type: "Float32Array", data: [5, 7, 3, 4] },
          },
          {
            name: "control:subgroup-truthiness-assignment-scalar",
            source: SOURCES.subgroupTruthinessAssignmentScalar,
            options: { workgroupSize: [4, 1, 1], subgroupMode: "scalar" },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([-1, 2, 0, 4]),
                out: new Int32Array(4),
              },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [0, 2, 0, 4] },
          },
          {
            name: "helpers:frexp-out-params",
            source: SOURCES.frexpOutParams,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(1),
                expOut: new Int32Array(1),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [0.5625] },
          },
          {
            name: "helpers:curand-storage-state",
            source: SOURCES.curandStorageState,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                states: new Uint32Array(4),
                out: new Float32Array(4),
              },
              scalars: { seed: 1234 },
            }),
            output: "out",
            tolerance: 0.001,
          },
          {
            name: "helpers:fp8-convert",
            source: SOURCES.fp8ConvertHelpers,
            options: { f16Mode: "f32", workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                input: new Uint32Array([0x3c]),
                output: new Float32Array(1),
                encoded: new Uint32Array(1),
                as_int: new Int32Array(1),
              },
            }),
            output: "encoded",
            expectedOutput: { type: "Uint32Array", data: [0x3c] },
          },
          {
            name: "helpers:bf16-cache-hint",
            source: SOURCES.bf16CacheHint,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                input: new Float32Array([1.5, 2.5]),
                output: new Float32Array(2),
                as_float: new Float32Array(2),
              },
            }),
            output: "as_float",
            expectedOutput: { type: "Float32Array", data: [2.5, 3.5] },
          },
          {
            name: "memory:cp-async-shared-copy",
            source: SOURCES.cpAsyncSharedCopy,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                input: new Float32Array([1, 2, 3, 4]),
                output: new Float32Array(4),
              },
            }),
            output: "output",
            expectedOutput: { type: "Float32Array", data: [2, 3, 4, 5] },
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
            name: "device-function:device-global-truthiness",
            source: SOURCES.deviceGlobalTruthiness,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                data: new Int32Array([0]),
                out: new Uint32Array(2),
              },
              deviceGlobals: {
                flag: new Uint32Array([0]),
                numErrors: new Uint32Array([0]),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [1, 1] },
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
            name: "wmma:toy-real-webgpu",
            source: SOURCES.wmmaToy,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                A: new Float32Array([1, 2, 3, 4]),
                B: new Float32Array([5, 6, 7, 8]),
                C: new Float32Array(4),
              },
            }),
            output: "C",
            expectedOutput: { type: "Float32Array", data: [20, 23, 44, 51] },
          },
          {
            name: "wmma:tf32-real-webgpu",
            source: SOURCES.wmmaTf32,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                A: new Float32Array([1, 2, 3, 4]),
                C: new Float32Array(4),
              },
            }),
            output: "C",
            expectedOutput: { type: "Float32Array", data: [2, 3, 4, 5] },
          },
          {
            name: "wmma:imma-real-webgpu",
            source: SOURCES.wmmaImma,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                A: new Uint32Array(256).fill(1),
                B: new Uint32Array(256).fill(2),
                C: new Int32Array(256),
              },
            }),
            output: "C",
            expectedOutput: { type: "Int32Array", data: Array(256).fill(33) },
          },
          {
            name: "ptx:local-half-carrier",
            source: SOURCES.localHalfCarrier,
            options: { workgroupSize: [1, 1, 1], f16Mode: "f32" },
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
            name: "compat:half2-arithmetic-f32-mode",
            source: SOURCES.half2Ops,
            options: { workgroupSize: [1, 1, 1], f16Mode: "f32" },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 8]),
                y: new Float32Array([2, 4]),
                out: new Float32Array(4),
                scalar: new Float32Array(1),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [6, 6, 5, 66] },
          },
          {
            name: "intrinsic:warp-half-pack",
            source: SOURCES.intrinsicPack,
            options: { workgroupSize: [1, 1, 1], f16Mode: "f32", features: { subgroups: true } },
            requiredFeatures: ["subgroups"],
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                h: new Float32Array(2),
                f: new Float32Array(2),
                out: new Float32Array(1),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [6] },
          },
          {
            name: "complex:cufft-magnitude",
            source: SOURCES.complexMagnitude,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                data: new Float32Array([3, 4, 5, 12]),
                mag: new Float32Array(2),
              },
              scalars: { N: 2 },
            }),
            output: "mag",
            expectedOutput: { type: "Float32Array", data: [5, 13] },
          },
          {
            name: "complex:cufft-local-writeback",
            source: SOURCES.complexMultiply,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                A: new Float32Array([1, 2, 3, 4]),
                B: new Float32Array([5, 6, 7, 8]),
              },
              scalars: { N: 2 },
            }),
            output: "A",
            expectedOutput: { type: "Float32Array", data: [-7, 16, -11, 52] },
          },
          {
            name: "complex:cufft-helper-pointwise",
            source: SOURCES.complexHelperPointwise,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                a: new Float32Array([1, 2, 3, 4]),
                b: new Float32Array([5, 6, 7, 8]),
              },
              scalars: { scale: 0.5 },
            }),
            output: "a",
            expectedOutput: { type: "Float32Array", data: [-3.5, 8, -5.5, 26] },
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
            name: "subgroup:array-after-return",
            source: SOURCES.subgroupArrayAfterReturn,
            options: { workgroupSize: [4, 1, 1], features: { subgroups: true } },
            requiredFeatures: ["subgroups"],
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4]),
                out: new Float32Array(1),
              },
              scalars: { n: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [15] },
          },
          {
            name: "cooperative:tile-vector-reduce",
            source: SOURCES.cooperativeTileVectorReduce,
            options: { workgroupSize: [4, 1, 1], features: { subgroups: true } },
            requiredFeatures: ["subgroups"],
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                x: new Float32Array([1, 2, 3, 4]),
                out: new Float32Array(7),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [10, 10, 20, 4, 1, 0, 12] },
          },
          {
            name: "cooperative:binary-partition-reduce",
            source: SOURCES.cooperativeBinaryPartitionReduce,
            options: { workgroupSize: [4, 1, 1], features: { subgroups: true } },
            requiredFeatures: ["subgroups"],
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array(4),
              },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [4, 6, 2, 2] },
          },
          {
            name: "cooperative:tile-vote-mask",
            source: SOURCES.cooperativeTileVoteMask,
            options: { workgroupSize: [8, 1, 1], features: { subgroups: true } },
            requiredFeatures: ["subgroups"],
            launch: { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array(6),
              },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [0, 0, 0, 1, 1, 15] },
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
            expectedOutput: { type: "Uint32Array", data: [1065353216, 0] },
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
            name: "atomic:conditional-vector-scalar-pointer-rebind",
            source: SOURCES.conditionalVectorScalarPointerAtomic,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                left: new Uint32Array([0, 0, 0, 0, 4, 0, 0, 0]),
                right: new Uint32Array([0, 0, 0, 0, 8, 0, 0, 0]),
                out: new Uint32Array(2),
              },
              scalars: { pickRight: 1 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [8, 9] },
          },
          {
            name: "atomic:conditional-float3-vector-scalar-pointer-rebind",
            source: SOURCES.conditionalFloat3VectorScalarPointerAtomic,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                left: new Float32Array([0, 0, 0, 4, 0, 0]),
                right: new Float32Array([0, 0, 0, 8, 0, 0]),
                out: new Float32Array(2),
              },
              scalars: { pickRight: 1 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [8, 9.5] },
          },
          {
            name: "atomic:pointer-array-vector-scalar-rebind",
            source: SOURCES.pointerArrayVectorScalarAtomic,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                counter: new Uint32Array([0, 0, 0, 0, 4, 0, 0, 0]),
                untouched: new Uint32Array([0, 0, 0, 0, 8, 0, 0, 0]),
                out: new Uint32Array(3),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [4, 5, 8] },
          },
          {
            name: "atomic:pointer-array-float3-vector-scalar-rebind",
            source: SOURCES.pointerArrayFloat3VectorScalarAtomic,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                counter: new Float32Array([0, 0, 0, 4, 0, 0]),
                untouched: new Float32Array([0, 0, 0, 8, 0, 0]),
                out: new Float32Array(3),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [4, 5.5, 8] },
          },
          {
            name: "atomic:helper-conditional-vector-scalar-pointer-rebind",
            source: SOURCES.helperConditionalVectorScalarPointerAtomic,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                left: new Uint32Array([0, 0, 0, 0, 4, 0, 0, 0]),
                right: new Uint32Array([0, 0, 0, 0, 8, 0, 0, 0]),
                out: new Uint32Array(2),
              },
              scalars: { pickRight: 1 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [8, 9] },
          },
          {
            name: "atomic:helper-pointer-array-vector-scalar-rebind",
            source: SOURCES.helperPointerArrayVectorScalarAtomic,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                counter: new Uint32Array([0, 0, 0, 0, 4, 0, 0, 0]),
                untouched: new Uint32Array([0, 0, 0, 0, 8, 0, 0, 0]),
                out: new Uint32Array(3),
              },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [4, 5, 8] },
          },
          {
            name: "storage:conditional-vector-lane-pointer-write",
            source: SOURCES.conditionalVectorLanePointerWrite,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                left: new Float32Array([0, 0, 0, 0, 5, 7, 11, 13]),
                right: new Float32Array([0, 0, 0, 0, 17, 19, 23, 29]),
                out: new Float32Array(2),
              },
              scalars: { pickRight: 1 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [7, 23.5] },
          },
          {
            name: "storage:helper-pointer-array-selected-args",
            source: SOURCES.helperPointerArraySelectedArgs,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                values: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
                out: new Float32Array(2),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [12, 27] },
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
            name: "atomic:vector-lane-helper",
            source: SOURCES.vectorLaneAtomic,
            options: { workgroupSize: [2, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [2, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [1.5, 0, 0, 4.5, 1.5, 0, 0, 4.5] },
          },
          {
            name: "atomic:float-exchange-assign",
            source: SOURCES.atomicFloatExchangeAssign,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                data: new Float32Array([1, 2, 3, 4]),
              },
            }),
            output: "data",
            expectedOutput: { type: "Float32Array", data: [11, 12, 13, 14] },
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
            name: "surface:surf3d-read",
            source: SOURCES.surfaceRead3d,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(2),
              },
              surfaces: {
                surf: { width: 2, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) },
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [7, 8] },
          },
          {
            name: "surface:surf3d-vector-read",
            source: SOURCES.surfaceVectorRead3d,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) },
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [10, 12, 14, 16] },
          },
          {
            name: "surface:layered-write",
            source: SOURCES.surfaceLayeredWrite,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {},
              surfaces: {
                outputSurf: { width: 2, height: 2, data: new Float32Array(8) },
              },
            }),
            output: "outputSurf",
            expectedOutput: { type: "Float32Array", data: [0, 0, 0, 0, 0, 0, 0, 23] },
          },
          {
            name: "surface:layered-read",
            source: SOURCES.surfaceLayeredRead,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(2),
              },
              surfaces: {
                surf: { width: 2, height: 2, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) },
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [7, 8] },
          },
          {
            name: "surface:layered-vector-read",
            source: SOURCES.surfaceVectorLayeredRead,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) },
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [10, 12, 14, 16] },
          },
          {
            name: "surface:vector-read-active-lane-return",
            source: SOURCES.surfaceVectorReadActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [26, 2, 3, 29] },
          },
          {
            name: "surface:vector-write-active-lane-return",
            source: SOURCES.surfaceVectorWriteActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array(8) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [62, 2, 3, 0] },
          },
          {
            name: "surface:float3-vector-active-lane-return",
            source: SOURCES.surfaceFloat3VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 3, height: 1, data: new Float32Array(6) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [75, 2, 3, 0] },
          },
          {
            name: "surface:float2-vector-active-lane-return",
            source: SOURCES.surfaceFloat2VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 2, height: 1, data: new Float32Array(4) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [89, 2, 3, 0] },
          },
          {
            name: "surface:uint2-vector-active-lane-return",
            source: SOURCES.surfaceUint2VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(4),
              },
              surfaces: {
                surf: { width: 2, height: 1, data: new Float32Array(4) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [109, 2, 3, 0] },
          },
          {
            name: "surface:int2-vector-active-lane-return",
            source: SOURCES.surfaceInt2VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array(4),
              },
              surfaces: {
                surf: { width: 2, height: 1, data: new Float32Array(4) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [129, 2, 3, 0] },
          },
          {
            name: "surface:uint3-vector-active-lane-return",
            source: SOURCES.surfaceUint3VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(4),
              },
              surfaces: {
                surf: { width: 3, height: 1, data: new Float32Array(6) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [105, 2, 3, 0] },
          },
          {
            name: "surface:int3-vector-active-lane-return",
            source: SOURCES.surfaceInt3VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array(4),
              },
              surfaces: {
                surf: { width: 3, height: 1, data: new Float32Array(6) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [105, 2, 3, 0] },
          },
          {
            name: "surface:layered-float4-vector-active-lane-return",
            source: SOURCES.surfaceFloat4VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array(8) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [302, 2, 3, 0] },
          },
          {
            name: "surface:layered-mixed-scalar-vector-active-lane-return",
            source: SOURCES.surfaceMixedScalarVectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array(8) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [341, 2, 3, 0] },
          },
          {
            name: "surface:pointer-alias-active-lane-store",
            source: SOURCES.surfacePointerAliasActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(16),
              },
              surfaces: {
                surf: { width: 1, height: 1, data: new Float32Array([2, 5]) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [1, 10, 20, 30, 2, 11, 21, 31, 3, 12, 22, 32, 0, 0, 8, 0] },
          },
          {
            name: "surface:surf1d-pointer-alias-active-lane-store",
            source: SOURCES.surface1DPointerAliasActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(16),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array([2, 3, 5, 7]) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [1, 10, 20, 30, 2, 11, 21, 31, 3, 12, 22, 32, 0, 8, 0, 0] },
          },
          {
            name: "surface:pointer-alias-atomic-active-lane-store",
            source: SOURCES.surfacePointerAliasAtomicActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(16),
              },
              surfaces: {
                surf: { width: 1, height: 1, data: new Float32Array([2, 5]) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [1, 10, 20, 30, 2, 11, 21, 31, 3, 12, 22, 32, 0, 0, 8, 0] },
          },
          {
            name: "surface:pointer-alias-atomic-vector-readback",
            source: SOURCES.surfacePointerAliasAtomicVectorReadback,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(16),
                summary: new Uint32Array(1),
              },
              surfaces: {
                surf: { width: 1, height: 1, data: new Float32Array([2, 5]) },
              },
            }),
            output: "summary",
            expectedOutput: { type: "Uint32Array", data: [70] },
          },
          {
            name: "surface:pointer-alias-atomic-vector-compound",
            source: SOURCES.surfacePointerAliasAtomicVectorCompound,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(16),
                summary: new Uint32Array(1),
              },
              surfaces: {
                surf: { width: 1, height: 1, data: new Float32Array([2, 5]) },
              },
            }),
            output: "summary",
            expectedOutput: { type: "Uint32Array", data: [7874] },
          },
          {
            name: "surface:uint4-vector-active-lane-return",
            source: SOURCES.surfaceUint4VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array(8) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [142, 2, 3, 0] },
          },
          {
            name: "surface:int4-vector-active-lane-return",
            source: SOURCES.surfaceInt4VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array(8) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [142, 2, 3, 0] },
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
            name: "texture:object-uint4-helper-read",
            source: SOURCES.textureObjectUint4HelperRead,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(4),
              },
              textures: {
                tex: {
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
            name: "texture:helper-vector-cast-coercion",
            source: SOURCES.textureHelperVectorCastCoercion,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([3, 5, 7, 11]),
                },
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [3, 5, 7, 11] },
          },
          {
            name: "texture:nested-helper-vector-read",
            source: SOURCES.textureNestedHelperVectorRead,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [2, 3, 5, 7] },
          },
          {
            name: "texture:active-lane-return-read-side-effect",
            source: SOURCES.textureActiveLaneReturnReadSideEffect,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [1, 2, 3, 20] },
          },
          {
            name: "texture:float3-active-lane-store",
            source: SOURCES.textureFloat3ActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(12),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [
              1, 10, 100,
              2, 11, 101,
              3, 12, 102,
              5, 6, 8,
            ] },
          },
          {
            name: "texture:float2-active-lane-store",
            source: SOURCES.textureFloat2ActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [
              1, 10,
              2, 11,
              3, 12,
              5, 6,
            ] },
          },
          {
            name: "texture:uint2-active-lane-store",
            source: SOURCES.textureUint2ActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(8),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [
              1, 10,
              2, 11,
              3, 12,
              5, 6,
            ] },
          },
          {
            name: "texture:int2-active-lane-store",
            source: SOURCES.textureInt2ActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array(8),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([-2, 3, -5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [
              1, -10,
              2, -11,
              3, -12,
              -5, 0,
            ] },
          },
          {
            name: "texture:uint3-active-lane-store",
            source: SOURCES.textureUint3ActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(12),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [
              1, 10, 100,
              2, 11, 101,
              3, 12, 102,
              5, 6, 8,
            ] },
          },
          {
            name: "texture:int3-active-lane-store",
            source: SOURCES.textureInt3ActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array(12),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([-2, 3, -5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [
              1, -10, 100,
              2, -11, 101,
              3, -12, 102,
              -5, 0, -8,
            ] },
          },
          {
            name: "texture:float4-active-lane-store",
            source: SOURCES.textureFloat4ActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(16),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [
              1, 10, 20, 30,
              2, 11, 21, 31,
              3, 12, 22, 32,
              5, 6, 8, 10,
            ] },
          },
          {
            name: "texture:uint4-active-lane-store",
            source: SOURCES.textureUint4ActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(16),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [
              1, 10, 20, 30,
              2, 11, 21, 31,
              3, 12, 22, 32,
              5, 6, 8, 10,
            ] },
          },
          {
            name: "texture:int4-active-lane-store",
            source: SOURCES.textureInt4ActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array(16),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([-2, 3, -5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [
              1, -10, 100, -100,
              2, -11, 101, -101,
              3, -12, 102, -102,
              -5, 0, -8, 4,
            ] },
          },
          {
            name: "texture:atlas-active-lane-return-read-side-effect",
            source: SOURCES.textureAtlasActiveLaneReturnReadSideEffect,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              textures: {
                tex: {
                  width: 4,
                  height: 24,
                  channels: 4,
                  data: new Float32Array(Array.from({ length: 4 * 24 * 4 }, (_, index) => index + 1)),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [1, 2, 3, 77] },
          },
          {
            name: "texture:deep-helper-active-lane-vector-store",
            source: SOURCES.textureDeepHelperActiveLaneVectorStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(16),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [1, 10, 20, 30, 2, 11, 21, 31, 3, 12, 22, 32, 5, 3, 5, 7] },
          },
          {
            name: "texture:mixed-scalar-vector-active-lane-store",
            source: SOURCES.textureMixedScalarVectorActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(16),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [1, 10, 20, 30, 2, 11, 21, 31, 3, 12, 22, 32, 5, 3, 5, 7] },
          },
          {
            name: "texture:pointer-alias-active-lane-store",
            source: SOURCES.texturePointerAliasActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(16),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [1, 10, 20, 30, 2, 11, 21, 31, 3, 12, 22, 32, 0, 5, 0, 0] },
          },
          {
            name: "texture:pointer-alias-atomic-active-lane-store",
            source: SOURCES.texturePointerAliasAtomicActiveLaneStore,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(16),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [1, 10, 20, 30, 2, 11, 21, 31, 3, 12, 22, 32, 0, 5, 0, 0] },
          },
          {
            name: "texture:pointer-alias-atomic-vector-readback",
            source: SOURCES.texturePointerAliasAtomicVectorReadback,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(16),
                summary: new Uint32Array(1),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
            }),
            output: "summary",
            expectedOutput: { type: "Uint32Array", data: [67] },
          },
          {
            name: "texture:pointer-alias-atomic-vector-compound",
            source: SOURCES.texturePointerAliasAtomicVectorCompound,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(16),
                summary: new Uint32Array(1),
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
            }),
            output: "summary",
            expectedOutput: { type: "Uint32Array", data: [7871] },
          },
          {
            name: "texture-surface:roundtrip",
            source: SOURCES.textureSurfaceRoundtrip,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(1),
              },
              surfaces: {
                surf: { width: 2, height: 1, data: new Float32Array([0, 0]) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
            }),
            output: "surf",
            expectedOutput: { type: "Float32Array", data: [12, 0] },
          },
          {
            name: "texture-surface:vector-helper-roundtrip",
            source: SOURCES.textureSurfaceVectorHelperRoundtrip,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(1),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array([0, 0, 0, 0]) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
            }),
            output: "surf",
            expectedOutput: { type: "Float32Array", data: [2, 3, 5, 7] },
          },
          {
            name: "texture-surface:active-lane-return-side-effect",
            source: SOURCES.textureSurfaceActiveLaneReturnSideEffect,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {},
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array(4) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "surf",
            expectedOutput: { type: "Float32Array", data: [1, 2, 3, 20] },
          },
          {
            name: "texture-surface:vector-active-lane-return",
            source: SOURCES.textureSurfaceVectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {},
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array(8) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "surf",
            expectedOutput: { type: "Float32Array", data: [1, 0, 0, 0, 5, 6, 8, 10] },
          },
          {
            name: "texture-surface:mixed-vector-active-lane-return",
            source: SOURCES.textureSurfaceMixedVectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array(8) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [23, 2, 3, 0] },
          },
          {
            name: "texture-surface:float2-vector-active-lane-return",
            source: SOURCES.textureSurfaceFloat2VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 2, height: 1, data: new Float32Array(4) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [11, 2, 3, 0] },
          },
          {
            name: "texture-surface:uint2-vector-active-lane-return",
            source: SOURCES.textureSurfaceUint2VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(4),
              },
              surfaces: {
                surf: { width: 2, height: 1, data: new Float32Array(4) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [11, 2, 3, 0] },
          },
          {
            name: "texture-surface:int2-vector-active-lane-return",
            source: SOURCES.textureSurfaceInt2VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array(4),
              },
              surfaces: {
                surf: { width: 2, height: 1, data: new Float32Array(4) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([-2, 3, -5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [-5, 2, 3, 0] },
          },
          {
            name: "texture-surface:float3-vector-active-lane-return",
            source: SOURCES.textureSurfaceFloat3VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 3, height: 1, data: new Float32Array(6) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [19, 2, 3, 0] },
          },
          {
            name: "texture-surface:uint3-vector-active-lane-return",
            source: SOURCES.textureSurfaceUint3VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(4),
              },
              surfaces: {
                surf: { width: 3, height: 1, data: new Float32Array(6) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [19, 2, 3, 0] },
          },
          {
            name: "texture-surface:int3-vector-active-lane-return",
            source: SOURCES.textureSurfaceInt3VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array(4),
              },
              surfaces: {
                surf: { width: 3, height: 1, data: new Float32Array(6) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([-2, 3, -5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [-13, 2, 3, 0] },
          },
          {
            name: "texture-surface:float4-vector-active-lane-return",
            source: SOURCES.textureSurfaceFloat4VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array(8) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [29, 2, 3, 0] },
          },
          {
            name: "texture-surface:uint4-vector-active-lane-return",
            source: SOURCES.textureSurfaceUint4VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Uint32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array(8) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([2, 3, 5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Uint32Array", data: [29, 2, 3, 0] },
          },
          {
            name: "texture-surface:int4-vector-active-lane-return",
            source: SOURCES.textureSurfaceInt4VectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Int32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array(8) },
              },
              textures: {
                tex: {
                  width: 1,
                  height: 1,
                  channels: 4,
                  data: new Float32Array([-2, 3, -5, 7]),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Int32Array", data: [-9, 2, 3, 0] },
          },
          {
            name: "texture-surface:volume-vector-active-lane-return",
            source: SOURCES.textureSurfaceVolumeVectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array(8) },
              },
              textures: {
                tex: {
                  width: 4,
                  height: 24,
                  channels: 4,
                  data: new Float32Array(Array.from({ length: 4 * 24 * 4 }, (_, index) => index + 1)),
                },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [320, 2, 3, 0] },
          },
          {
            name: "surface:helper-dispatch-multiple-surfaces",
            source: SOURCES.surfaceHelperDispatchMultipleSurfaces,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {},
              surfaces: {
                first: { width: 2, height: 1, data: new Float32Array([5, 6]) },
                second: { width: 2, height: 1, data: new Float32Array([11, 13]) },
              },
            }),
            output: "second",
            expectedOutput: { type: "Float32Array", data: [11, 12] },
          },
          {
            name: "surface:vector-read",
            source: SOURCES.surfaceVectorRead,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array([2, 3, 5, 7]) },
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [2, 3, 5, 7] },
          },
          {
            name: "surface:helper-vector-read-multiple-surfaces",
            source: SOURCES.surfaceHelperVectorReadMultipleSurfaces,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(8),
              },
              surfaces: {
                first: { width: 4, height: 1, data: new Float32Array([2, 3, 5, 7]) },
                second: { width: 4, height: 1, data: new Float32Array([11, 13, 17, 19]) },
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [2, 3, 5, 7, 11, 13, 17, 19] },
          },
          {
            name: "surface:active-lane-return-side-effect",
            source: SOURCES.surfaceActiveLaneReturnSideEffect,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {},
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array(4) },
              },
              scalars: { N: 3 },
            }),
            output: "surf",
            expectedOutput: { type: "Float32Array", data: [1, 2, 3, 103] },
          },
          {
            name: "surface:helper-vector-layered-write",
            source: SOURCES.surfaceHelperVectorLayeredWrite,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {},
              surfaces: {
                outputSurf: { width: 4, height: 2, data: new Float32Array(16) },
              },
            }),
            output: "outputSurf",
            expectedOutput: {
              type: "Float32Array",
              data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 5, 7],
            },
          },
          {
            name: "surface:surf1d-write",
            source: SOURCES.surface1DWrite,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {},
              surfaces: {
                outputSurf: { width: 4, height: 1, data: new Float32Array(4) },
              },
            }),
            output: "outputSurf",
            expectedOutput: { type: "Float32Array", data: [0, 0, 37, 0] },
          },
          {
            name: "surface:surf1d-vector-write",
            source: SOURCES.surface1DVectorWrite,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {},
              surfaces: {
                outputSurf: { width: 6, height: 1, data: new Float32Array(6) },
              },
            }),
            output: "outputSurf",
            expectedOutput: { type: "Float32Array", data: [0, 2, 3, 5, 7, 0] },
          },
          {
            name: "surface:surf1d-read",
            source: SOURCES.surface1DRead,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(3),
              },
              surfaces: {
                surf: { width: 4, height: 1, data: new Float32Array([11, 13, 17, 19]) },
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [17, 19, 36] },
          },
          {
            name: "surface:surf1d-vector-read",
            source: SOURCES.surface1DVectorRead,
            options: { workgroupSize: [1, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 8, height: 1, data: new Float32Array([2, 3, 5, 7, 11, 13, 17, 19]) },
              },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [14, 18, 24, 30] },
          },
          {
            name: "surface:surf1d-vector-active-lane-return",
            source: SOURCES.surface1DVectorActiveLaneReturn,
            options: { workgroupSize: [4, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array(4),
              },
              surfaces: {
                surf: { width: 6, height: 1, data: new Float32Array(6) },
              },
              scalars: { N: 3 },
            }),
            output: "out",
            expectedOutput: { type: "Float32Array", data: [62, 2, 3, 0] },
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
          const spec = autoCorpusSmokeCaseSpec(fixture);
          if (spec) cases.push(spec);
        }
        return cases;
      }

      function corpusFixtureCaseSpecs() {
        const cases = [];
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
        return cases;
      }

      function autoCorpusSmokeCaseSpecs() {
        return AUTO_CORPUS_SMOKE_FIXTURES.map(autoCorpusSmokeCaseSpec).filter(Boolean);
      }

      function autoCorpusSmokeCaseSpec(fixture) {
        const source = SOURCES[fixture.sourceKey];
        if (!source) return undefined;
        return {
          name: fixture.caseName,
          source,
          options: { ...(fixture.options ?? {}), kernelName: fixture.kernelName, workgroupSize: fixture.workgroupSize },
          launch: fixture.launch,
          input: (compiled) => syntheticInputForCompiled(compiled),
          corpusId: fixture.corpusId,
          verifyMode: fixture.verifyMode,
          requiredFeatures: fixture.requiredFeatures,
          comparison: fixture.comparison,
        };
      }

      function countByCorpus(items) {
        const out = {};
        for (const item of items) {
          const corpusId = item.corpusId ?? "unknown";
          out[corpusId] = (out[corpusId] ?? 0) + 1;
        }
        return out;
      }

      function countByStage(items) {
        const out = {};
        for (const item of items) {
          const stage = item.stage ?? "unknown";
          out[stage] = (out[stage] ?? 0) + 1;
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
          if (constant.init !== undefined) continue;
          constants[constant.name] = constant.dimensions.length === 0 && !isCudaVectorTypeName(constant.valueType)
            ? syntheticScalarForName(constant.name)
            : syntheticConstantBufferForType(constant.valueType, constant.name, 4096, compiled.f16Mode);
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
        if (type === "uint" || type === "uchar" || /^uint[234]$/u.test(type) || type === "voidptr" || type === "bool") return new Uint32Array(length);
        if ((type === "half" || type === "half2") && f16Mode !== "f32") return createWgslFloat16Array(length);
        return new Float32Array(length);
      }

      function syntheticConstantBufferForType(type, name, length = 4096, f16Mode = "native") {
        const buffer = syntheticBufferForType(type, length, f16Mode);
        if (buffer instanceof Uint32Array || buffer instanceof Int32Array) {
          buffer.fill(/gridSize|numCells|numBodies|maxParticlesPerCell/iu.test(name) ? 64 : 1);
        } else if (buffer instanceof Float32Array) {
          buffer.fill(/worldOrigin|colliderPos|gravity/iu.test(name) ? 0 : 1);
        }
        return buffer;
      }

      function isCudaVectorTypeName(type) {
        return /^(?:float|int|uint)[234]$|^half2$|^bf162$/u.test(type);
      }

      function syntheticScalarForName(name) {
        if (/^(?:warpSize|warp_size)$/u.test(name)) return 32;
        if (/^(?:nanoseconds|microseconds|milliseconds)$/iu.test(name)) return 0;
        if (/(?:clock|delay|sleep|spin|wait)/iu.test(name)) return 0;
        if (/^(?:depth|level)$/iu.test(name)) return 0;
        if (/^(?:maxDepth|max_depth|maxLevel|max_level)$/u.test(name)) return 4;
        if (/^(?:left|begin|start|offset)$/u.test(name)) return 0;
        if (/^(?:right|end|len|nLines|nTessPoints)$/u.test(name)) return 64;
        if (/^(?:C|cols|columns|channels|nChannels|vocabSize|vocab_size)$/u.test(name)) return 64;
        if (/^(?:n|N|num|count|length|totalLen|frontierSize|numSamples|totalThreads|poolSize|size)$/u.test(name)) return 1024;
        if (/^(?:threads|threadsPerBlock|threads_per_block|blockSize|block_size)$/u.test(name)) return 256;
        if (/^(?:blocks|blocksPerGrid|numBlocks)$/u.test(name)) return 4;
        return 1;
      }

      function externalDevicePoolNamesFromSource(source) {
        return [...source.matchAll(/\b(?:deviceAllocate|streamOrderedAllocate)\s*\(\s*&\s*([A-Za-z_][A-Za-z0-9_]*)/g)]
          .map((match) => match[1])
          .filter(Boolean);
      }

      function inferDynamicSharedMemory(source) {
        const out = {};
        const sharedBeforeType = /extern\s+__shared__\s+(?:(?:__align__|alignas)\s*\([^)]*\)\s*)?(?:float[234]?|double|int[234]?|uint[234]?|char[234]?|uchar[234]?|uint8_t[234]?|int8_t[234]?|unsigned\s+int|signed\s+int|short|unsigned\s+short|unsigned\s+char|char|int|uint|half|__half|bf16|__nv_bfloat16|bool)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*]/g;
        const sharedAfterType = /extern\s+(?:float[234]?|double|int[234]?|uint[234]?|char[234]?|uchar[234]?|uint8_t[234]?|int8_t[234]?|unsigned\s+int|signed\s+int|short|unsigned\s+short|unsigned\s+char|char|int|uint|half|__half|bf16|__nv_bfloat16|bool)\s+__shared__\s+(?:(?:__align__|alignas)\s*\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*]/g;
        for (const match of source.matchAll(sharedBeforeType)) out[match[1]] = 256;
        for (const match of source.matchAll(sharedAfterType)) out[match[1]] = 256;
        return out;
      }

      function nonEmptyDynamicSharedMemory(value) {
        return value && Object.keys(value).length > 0 ? value : undefined;
      }

      async function runReferenceWebGpuCase(device, spec) {
        const profile = shouldProfileCase(spec) ? {} : undefined;
        let phaseStart = performance.now();
        const compiled = compileCudaLiteKernelForWebGpu(spec.source, spec.options);
        if (profile) profile.compileMs = round(performance.now() - phaseStart);
        const diagnosticSkip = webGpuDiagnosticSkip(compiled);
        if (diagnosticSkip) {
          return {
            ok: false,
            stage: "diagnostics",
            plan: diagnosticSkip,
            maxAbsDiff: Number.POSITIVE_INFINITY,
            output: spec.verifyMode === "dispatch" ? "dispatch" : spec.output,
            ...(profile === undefined ? {} : { profile }),
          };
        }
        phaseStart = performance.now();
        const actualInput = spec.input(compiled);
        if (profile) profile.inputMs = round(performance.now() - phaseStart);
        phaseStart = performance.now();
        const plan = createCudaWebGpuExecutionPlan(compiled, actualInput, spec.launch, {
          compileKernel: (childSource, childOptions = {}) => compileCudaLiteKernelForWebGpu(childSource, {
            ...childOptions,
            f16Mode: childOptions.f16Mode ?? compiled.f16Mode,
            dynamicSharedMemory: nonEmptyDynamicSharedMemory(childOptions.dynamicSharedMemory) ?? inferDynamicSharedMemory(childSource),
          }),
        });
        if (profile) profile.planMs = round(performance.now() - phaseStart);
        if (!plan.supported) {
          return { ok: false, stage: "webgpu_plan", plan: plan.reason, maxAbsDiff: Number.POSITIVE_INFINITY, ...(profile === undefined ? {} : { profile }) };
        }
        const resourceLimitBlocker = webGpuResourceLimitBlocker(device.gpu.limits, plan);
        if (resourceLimitBlocker) {
          return {
            ok: false,
            stage: "resource_limits",
            plan: "resource-limits:" + resourceLimitBlocker,
            maxAbsDiff: Number.POSITIVE_INFINITY,
            output: spec.verifyMode === "dispatch" ? "dispatch" : spec.output,
            ...(profile === undefined ? {} : { profile }),
          };
        }
        phaseStart = performance.now();
        const webgpuStart = phaseStart;
        let rawActual;
        if (profile) {
          const prepared = await prepareWgslKernelProgramSequence(device, plan.steps, plan.input);
          profile.prepareMs = round(performance.now() - phaseStart);
          phaseStart = performance.now();
          try {
            rawActual = await prepared.run();
            profile.runMs = round(performance.now() - phaseStart);
          } finally {
            prepared.destroy();
          }
          phaseStart = performance.now();
          const cachedPrepared = await prepareWgslKernelProgramSequence(device, plan.steps, plan.input);
          profile.prepareCachedMs = round(performance.now() - phaseStart);
          cachedPrepared.destroy();
        } else {
          rawActual = await runWgslKernelProgramSequence(device, plan.steps, plan.input);
        }
        const actual = { buffers: normalizeCudaWebGpuReadback(compiled, rawActual.buffers), trace: [] };
        if (profile) profile.webgpuMs = round(performance.now() - webgpuStart);
        if (spec.verifyMode === "dispatch") {
          return { ok: true, stage: "dispatch", plan: plan.kind, maxAbsDiff: 0, output: "dispatch", ...(profile === undefined ? {} : { profile }) };
        }
        const outputNames = spec.output === undefined ? Object.keys(actual.buffers) : [spec.output];
        if (spec.expectedOutput !== undefined) {
          phaseStart = performance.now();
          const comparison = compareOutputs({ buffers: {} }, actual, outputNames, spec);
          if (profile) profile.compareMs = round(performance.now() - phaseStart);
          if (comparison.ok && spec.offsetOutput) {
            const offset = actual.buffers[spec.offsetOutput]?.[0];
            if (offset !== spec.expectedOffset) {
              return { ok: false, stage: "compare", plan: plan.kind, maxAbsDiff: Math.abs(Number(offset ?? NaN) - spec.expectedOffset), ...(profile === undefined ? {} : { profile }) };
            }
          }
          return { stage: "compare", plan: plan.kind, output: outputNames.join(","), ...comparison, ...(profile === undefined ? {} : { profile }) };
        }
        const expectedInput = spec.offsetOutput && actualInput.readback
          ? { ...actualInput, readback: actualInput.readback.filter((name) => name !== spec.offsetOutput) }
          : actualInput;
        phaseStart = performance.now();
        const expected = runCompiledKernelReference(compiled, expectedInput, spec.launch);
        if (profile) profile.referenceMs = round(performance.now() - phaseStart);
        if (spec.comparison?.kind !== "newdelete-stack-pop") {
          const referenceComparison = compareOutputs(expected, expected, outputNames, spec);
          if (!referenceComparison.ok) return { ...referenceComparison, stage: "reference", plan: "reference-output-mismatch", ...(profile === undefined ? {} : { profile }) };
        }
        phaseStart = performance.now();
        const comparison = compareOutputs(expected, actual, outputNames, spec);
        if (profile) profile.compareMs = round(performance.now() - phaseStart);
        if (comparison.ok && spec.offsetOutput) {
          const offset = actual.buffers[spec.offsetOutput]?.[0];
          if (offset !== spec.expectedOffset) {
            return { ok: false, stage: "compare", plan: plan.kind, maxAbsDiff: Math.abs(Number(offset ?? NaN) - spec.expectedOffset), ...(profile === undefined ? {} : { profile }) };
          }
        }
        return { stage: "compare", plan: plan.kind, output: outputNames.join(","), ...comparison, ...(profile === undefined ? {} : { profile }) };
      }

      async function runCompileOnlyWebGpuCase(device, spec) {
        const compiled = compileCudaLiteKernelForWebGpu(spec.source, spec.options);
        const diagnosticSkip = webGpuDiagnosticSkip(compiled);
        if (diagnosticSkip) {
          return {
            ok: false,
            stage: "diagnostics",
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
            dynamicSharedMemory: nonEmptyDynamicSharedMemory(childOptions.dynamicSharedMemory) ?? inferDynamicSharedMemory(childSource),
          }),
        });
        if (!plan.supported) {
          return { ok: false, stage: "webgpu_plan", plan: plan.reason, maxAbsDiff: Number.POSITIVE_INFINITY };
        }
        const resourceLimitBlocker = webGpuResourceLimitBlocker(device.gpu.limits, plan);
        if (resourceLimitBlocker) {
          return {
            ok: false,
            stage: "resource_limits",
            plan: "resource-limits:" + resourceLimitBlocker,
            maxAbsDiff: Number.POSITIVE_INFINITY,
            output: spec.verifyMode === "dispatch" ? "dispatch" : spec.output,
          };
        }
        for (const step of plan.steps ?? []) {
          const wgsl = step.program?.wgsl;
          if (!wgsl) continue;
          const module = device.gpu.createShaderModule({ code: wgsl });
          const info = await module.getCompilationInfo?.();
          const errors = (info?.messages ?? []).filter((message) => message.type === "error");
          if (errors.length > 0) {
            return {
              ok: false,
              stage: "wgsl_compile",
              plan: "wgsl-compile-error",
              maxAbsDiff: Number.POSITIVE_INFINITY,
              output: spec.verifyMode === "dispatch" ? "dispatch" : spec.output,
              error: {
                name: "WGSLCompileError",
                message: errors.map((message) => message.message).join("\\n"),
                messages: errors.map((message) => ({
                  message: message.message,
                  lineNum: message.lineNum,
                  linePos: message.linePos,
                  offset: message.offset,
                  length: message.length,
                })),
              },
            };
          }
        }
        return {
          ok: true,
          stage: "wgsl_compile",
          plan: "compile-only:" + plan.kind,
          maxAbsDiff: 0,
          output: spec.verifyMode === "dispatch" ? "dispatch" : spec.output,
        };
      }

      function shouldProfileCase(spec) {
        return PROFILE_CASE === "true" || PROFILE_CASE === "all" || PROFILE_CASE === spec.name;
      }

      function compareOutputs(expected, actual, outputNames, spec) {
        let worst = { ok: true, maxAbsDiff: 0 };
        for (const outputName of outputNames) {
          const expectedOutput = spec.expectedOutput === undefined && outputNames.length === 1
            ? expected.buffers[outputName]
            : spec.expectedOutput === undefined
              ? expected.buffers[outputName]
              : materializeTypedArray(spec.expectedOutput);
          const comparison = compareArraysWithMode(expectedOutput, actual.buffers[outputName], spec);
          if (!comparison.ok) return { ...comparison, output: outputName };
          if (comparison.maxAbsDiff > worst.maxAbsDiff) worst = comparison;
        }
        return worst;
      }

      function compareArraysWithMode(expected, actual, spec) {
        if (spec.comparison?.kind === "newdelete-stack-pop") return compareNewdeleteStackPop(actual, spec);
        if (spec.comparison?.kind === "int-multiset") return compareIntegerMultiset(expected, actual);
        return compareArrays(expected, actual, spec.tolerance);
      }

      function compareNewdeleteStackPop(actual, spec) {
        const threadCount = (spec.launch?.blockDim ?? [0, 1, 1]).reduce((product, value) => product * Number(value ?? 1), 1);
        if (!actual || threadCount <= 0 || actual.length < threadCount) {
          return { ok: false, maxAbsDiff: Number.POSITIVE_INFINITY };
        }
        const pushCount = Math.ceil(threadCount / 2);
        const expected = new Int32Array(threadCount);
        for (let index = 0; index < pushCount; index++) expected[index] = index;
        for (let index = pushCount; index < threadCount; index++) expected[index] = -1;
        return compareIntegerMultiset(expected, actual.slice(0, threadCount));
      }

      function compareIntegerMultiset(expected, actual) {
        if (!expected || !actual || expected.length !== actual.length) {
          return { ok: false, maxAbsDiff: Number.POSITIVE_INFINITY };
        }
        const expectedCounts = integerCounts(expected);
        const actualCounts = integerCounts(actual);
        let firstDiff;
        let maxAbsDiff = 0;
        for (const key of new Set([...expectedCounts.keys(), ...actualCounts.keys()])) {
          const expectedCount = expectedCounts.get(key) ?? 0;
          const actualCount = actualCounts.get(key) ?? 0;
          const diff = Math.abs(expectedCount - actualCount);
          if (diff > maxAbsDiff) maxAbsDiff = diff;
          if (firstDiff === undefined && diff !== 0) {
            firstDiff = {
              index: Number(key),
              expected: expectedCount,
              actual: actualCount,
              absDiff: diff,
              relDiff: diff / Math.max(expectedCount, actualCount, 1),
            };
          }
        }
        return {
          ok: firstDiff === undefined,
          maxAbsDiff: round(maxAbsDiff),
          ...(firstDiff === undefined
            ? {}
            : {
                firstDiff: {
                  ...firstDiff,
                  absDiff: round(firstDiff.absDiff),
                  relDiff: round(firstDiff.relDiff),
                },
              }),
        };
      }

      function integerCounts(values) {
        const counts = new Map();
        for (const value of values) {
          const key = Number(value);
          counts.set(key, (counts.get(key) ?? 0) + 1);
        }
        return counts;
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
          if (wgslHasActiveLaneReturnLowering(compiled.wgsl)) return undefined;
          return "non-uniform-return-before-barrier";
        }
        return undefined;
      }

      function wgslHasActiveLaneReturnLowering(wgsl) {
        if (typeof wgsl !== "string") return false;
        if (!wgsl.includes("workgroupBarrier();")) return false;
        if (wgsl.includes("var bg_active_lane: bool = true;") && wgsl.includes("bg_active_lane = false;")) return true;
        return wgsl.includes("var bg_barrier_loop_active_") && /bg_barrier_loop_active_\d+ = false;/u.test(wgsl);
      }

      function compareArrays(expected, actual, tolerance = 1e-5, relativeTolerance = 1e-5) {
        if (!expected || !actual || expected.length !== actual.length) {
          return { ok: false, maxAbsDiff: Number.POSITIVE_INFINITY };
        }
        const useRelativeTolerance = isFloatTypedArray(expected) || isFloatTypedArray(actual);
        let maxAbsDiff = 0;
        let maxRelDiff = 0;
        let firstDiff;
        for (let i = 0; i < expected.length; i++) {
          const expectedValue = Number(expected[i]);
          const actualValue = Number(actual[i]);
          const bothNaN = Number.isNaN(expectedValue) && Number.isNaN(actualValue);
          const sameValue = bothNaN || Object.is(expectedValue, actualValue) || expectedValue === actualValue;
          const diff = sameValue ? 0 : Math.abs(expectedValue - actualValue);
          const relativeBase = Math.max(Math.abs(expectedValue), Math.abs(actualValue), 1);
          const relDiff = sameValue ? 0 : diff / relativeBase;
          if (diff > maxAbsDiff) maxAbsDiff = diff;
          if (relDiff > maxRelDiff) maxRelDiff = relDiff;
          const withinTolerance = diff <= tolerance || (useRelativeTolerance && relDiff <= relativeTolerance);
          if (firstDiff === undefined && !withinTolerance) {
            firstDiff = {
              index: i,
              expected: expectedValue,
              actual: actualValue,
              absDiff: diff,
              relDiff,
            };
          }
        }
        return {
          ok: firstDiff === undefined,
          maxAbsDiff: round(maxAbsDiff),
          ...(useRelativeTolerance ? { maxRelDiff: round(maxRelDiff) } : {}),
          ...(firstDiff === undefined
            ? {}
            : {
                firstDiff: {
                  ...firstDiff,
                  absDiff: round(firstDiff.absDiff),
                  relDiff: round(firstDiff.relDiff),
                },
              }),
        };
      }

      function isFloatTypedArray(value) {
        return value instanceof Float32Array || value instanceof Float64Array;
      }

      function round(value) {
        return Math.round(value * 1000000) / 1000000;
      }

    </script>
  </body>
</html>`;

function parsePositiveInteger(raw, flag) {
  const value = Number(raw);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${flag} expects a positive integer`);
  }
  return value;
}

function parsePositiveNumber(raw, flag) {
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${flag} expects a positive number`);
  }
  return value;
}

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
  const expectedAutoCorpusSmokeCovered = autoCorpusSmokeShard.count > 1
    ? autoCorpusSmokeFixtures.length
    : effectiveAutoCorpusSmokeLimit;
  if (report.available && effectiveAutoCorpusSmokeLimit > 0 && !filteredRun && (report.autoCorpusSmokeCovered ?? report.autoCorpusSmokePassed ?? 0) < expectedAutoCorpusSmokeCovered) {
    throw new Error(`Auto corpus smoke baseline failed: ${report.autoCorpusSmokeCovered ?? report.autoCorpusSmokePassed ?? 0}/${expectedAutoCorpusSmokeCovered} covered`);
  }
  if (report.available && forbidSkips && (report.skipped ?? 0) > 0) {
    throw new Error(`CUDA-lite WebGPU e2e skipped ${report.skipped} case(s) with --forbid-skips`);
  }
  if (report.available && (report.warmupFailed ?? 0) > 0) {
    throw new Error(`CUDA-lite WebGPU e2e warmup failed ${report.warmupFailed} case(s)`);
  }
  if (report.available) {
    validateWarmSpeedup(report, expectWarmSpeedupMin);
    validateWarmMsMax(report, expectWarmMsMax);
  }
  if (report.available && report.failed > 0) {
    writeLastFailures(root, report, process.argv.slice(2));
    throw new Error(`CUDA-lite WebGPU e2e failed ${report.failed} case(s)`);
  }
  if (markdownPath && markdownPath !== "true") {
    fs.writeFileSync(path.resolve(markdownPath), markdownReport(report));
  }
} finally {
  if (browser) await browser.close();
  await server.close();
}

function writeLastFailures(root, report, argv) {
  const cases = failureReplayCases(report);
  if (cases.length === 0) return;
  const out = {
    tool: "browsergrad-cuda-lite-last-failures",
    generatedAt: new Date().toISOString(),
    bundle: report.bundle ?? "src",
    cases,
    failedByStage: report.failedByStage ?? {},
    command: [
      "pnpm",
      "--filter",
      "@unlocalhosted/browsergrad-compiler",
      "run",
      "e2e:webgpu:case",
      "--",
      "--cases",
      cases.join(","),
    ],
    originalArgs: argv,
  };
  const target = path.join(root, ".tmp", "cuda-lite-last-failures.json");
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, `${JSON.stringify(out, null, 2)}\n`);
}
