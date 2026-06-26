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

const headed = args.get("--headed") === "true";
const requireWebGpu = args.get("--require-webgpu") === "true";
const requireCorpusFixtures = args.get("--require-corpus-fixtures") === "true";
const markdownPath = args.get("--markdown");
const jsonPath = args.get("--json");
const summaryOnly = args.get("--summary-only") === "true";
const progress = args.get("--progress") === "true";
const progressPath = args.get("--progress-file");
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

        const specs = caseSpecs();
        for (let index = 0; index < specs.length; index++) {
          const spec = specs[index];
          emitProgress({ event: "case-start", index: index + 1, total: specs.length, name: spec.name });
          const start = performance.now();
          try {
            const result = await withCaseTimeout(runReferenceWebGpuCase(device, spec), spec.name);
            cases.push({
              name: spec.name,
              plan: result.plan,
              ok: result.ok,
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

        const failed = cases.filter((item) => !item.ok);
        const corpusFixtureCases = cases.filter((item) => item.name.startsWith("corpus:"));
        const corpusFixturePassed = corpusFixtureCases.filter((item) => item.ok);
        const corpusFixtureExpectedOutputCases = corpusFixtureCases.filter((item) => item.expectedOutputPinned);
        const autoCorpusSmokeCases = cases.filter((item) => item.name.startsWith("auto-corpus:"));
        const autoCorpusSmokePassed = autoCorpusSmokeCases.filter((item) => item.ok);
        const loadedCorpusFixtureNames = new Set(corpusFixtureCases.map((item) => item.name));
        return {
          available: true,
          cases,
          corpusFixtureCases: corpusFixtureCases.length,
          corpusFixturePassed: corpusFixturePassed.length,
          corpusFixtureFailed: corpusFixtureCases.filter((item) => !item.ok).length,
          corpusFixtureExpectedOutputCases: corpusFixtureExpectedOutputCases.length,
          corpusFixtureCasesByCorpus: countByCorpus(corpusFixtureCases),
          corpusFixturePassedByCorpus: countByCorpus(corpusFixturePassed),
          autoCorpusSmokeCases: autoCorpusSmokeCases.length,
          autoCorpusSmokePassed: autoCorpusSmokePassed.length,
          autoCorpusSmokeFailed: autoCorpusSmokeCases.filter((item) => !item.ok).length,
          autoCorpusSmokeCasesByCorpus: countByCorpus(autoCorpusSmokeCases),
          autoCorpusSmokePassedByCorpus: countByCorpus(autoCorpusSmokePassed),
          autoCorpusSmokeLimit: ${autoCorpusSmokeLimit},
          autoCorpusSmokeMode: ${JSON.stringify(autoCorpusSmokeMode)},
          corpusFixtureBaseline: CORPUS_FIXTURE_BASELINE,
          expectedCorpusFixtureNames: EXPECTED_CORPUS_FIXTURE_NAMES,
          missingCorpusFixtureNames: EXPECTED_CORPUS_FIXTURE_NAMES.filter((name) => !loadedCorpusFixtureNames.has(name)),
          passed: cases.length - failed.length,
          failed: failed.length,
        };
      };

      async function createE2eKernelDevice() {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter) throw new Error("Failed to obtain a WebGPU adapter");
        const requiredFeatures = ["shader-f16", "subgroups"].filter((feature) => adapter.features?.has?.(feature));
        const gpuDevice = await adapter.requestDevice({ requiredFeatures });
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

      function emitProgress(event) {
        if (!PROGRESS) return;
        console.log("__BG_PROGRESS__" + JSON.stringify(event));
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
              buffers[param.name] = syntheticBufferForType(param.valueType);
            }
          } else {
            scalars[param.name] = syntheticScalarForName(param.name);
          }
        }
        for (const constant of compiled.ir.constants) {
          constants[constant.name] = constant.dimensions.length === 0 && !isCudaVectorTypeName(constant.valueType)
            ? syntheticScalarForName(constant.name)
            : syntheticBufferForType(constant.valueType);
        }
        for (const global of compiled.ir.deviceGlobals) {
          const length = global.dimensions.length === 0
            ? 1
            : global.dimensions.reduce((product, dimension) => product * dimension, 1);
          deviceGlobals[global.name] = syntheticBufferForType(global.valueType, length);
        }
        for (const texture of compiled.ir.textures) {
          textures[texture.name] = { width: 64, height: 64, data: new Float32Array(64 * 64) };
        }
        for (const poolName of externalDevicePoolNamesFromSource(compiled.ast.source)) {
          memoryPools[poolName] ??= { data: new Uint32Array(4096), offset: new Uint32Array([0]) };
        }
        return { buffers, scalars, constants, deviceGlobals, memoryPools, textures, surfaces };
      }

      function syntheticBufferForType(type, length = 4096) {
        if (type === "int") return new Int32Array(length);
        if (type === "uint" || type === "voidptr" || type === "bool") return new Uint32Array(length);
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
        if (spec.verifyMode === "dispatch") {
          await runCompiledKernelWebGpu(device, compiled, actualInput, spec.launch);
          return { ok: true, plan: plan.kind, maxAbsDiff: 0, output: "dispatch" };
        }
        const expectedInput = spec.offsetOutput && actualInput.readback
          ? { ...actualInput, readback: actualInput.readback.filter((name) => name !== spec.offsetOutput) }
          : actualInput;
        const expected = runCompiledKernelReference(compiled, expectedInput, spec.launch);
        const actual = await runCompiledKernelWebGpu(device, compiled, actualInput, spec.launch);
        const outputNames = spec.output === undefined ? Object.keys(expected.buffers) : [spec.output];
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
  if (report.available && requireCorpusFixtures && (report.missingCorpusFixtureNames?.length ?? 0) > 0) {
    throw new Error(`Missing corpus execution fixtures from /tmp corpora: ${report.missingCorpusFixtureNames.join(", ")}`);
  }
  if (report.available && requireCorpusFixtures) {
    validateCorpusFixtureBaseline(report);
  }
  if (report.available && autoCorpusSmokeLimit > 0 && (report.autoCorpusSmokePassed ?? 0) < autoCorpusSmokeLimit) {
    throw new Error(`Auto corpus smoke baseline failed: ${report.autoCorpusSmokePassed ?? 0}/${autoCorpusSmokeLimit} passed`);
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
