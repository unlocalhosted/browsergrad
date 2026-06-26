#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { corpusById, cudaLiteCorpusExecutionFixtures } from "./cuda-lite-corpus-registry.mjs";

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
const bundle = parseBundle(args.get("--bundle") ?? "src");

const root = findRepoRoot(process.cwd());
const packageRequire = createRequire(path.join(root, "packages/browsergrad-compiler/package.json"));
const { createServer } = await import(pathToFileURL(packageRequire.resolve("vite")).href);
const playwright = await import(pathToFileURL(packageRequire.resolve("playwright")).href);
const chromium = playwright.chromium ?? playwright.default?.chromium;
if (!chromium) throw new Error("could not load Playwright chromium");
if (requireCorpusFixtures) verifyCorpusFixtureCheckouts();

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
  ...loadCorpusExecutionSources(),
};
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
      const EXPECTED_CORPUS_FIXTURE_NAMES = ${JSON.stringify(expectedCorpusFixtureNames)};

      window.__bgRunE2e = async () => {
        if (!navigator.gpu) {
          return {
            available: false,
            reason: "navigator.gpu undefined",
            cases: [],
          };
        }
        const device = await createDevice();
        const cases = [];

        for (const spec of caseSpecs()) {
          const start = performance.now();
          const result = await runReferenceWebGpuCase(device, spec);
          cases.push({
            name: spec.name,
            plan: result.plan,
            ok: result.ok,
            maxAbsDiff: result.maxAbsDiff,
            ...(result.firstDiff === undefined ? {} : { firstDiff: result.firstDiff }),
            output: spec.output,
            ms: round(performance.now() - start),
          });
        }

        const preparedStart = performance.now();
        cases.push({
          name: "prepared-resident-saxpy",
          plan: "prepared:single-dispatch",
          ...(await runPreparedResidentSaxpy(device)),
          output: "y",
          ms: round(performance.now() - preparedStart),
        });

        const failed = cases.filter((item) => !item.ok);
        const corpusFixtureCases = cases.filter((item) => item.name.startsWith("corpus:"));
        const loadedCorpusFixtureNames = new Set(corpusFixtureCases.map((item) => item.name));
        return {
          available: true,
          cases,
          corpusFixtureCases: corpusFixtureCases.length,
          corpusFixturePassed: corpusFixtureCases.filter((item) => item.ok).length,
          corpusFixtureFailed: corpusFixtureCases.filter((item) => !item.ok).length,
          expectedCorpusFixtureNames: EXPECTED_CORPUS_FIXTURE_NAMES,
          missingCorpusFixtureNames: EXPECTED_CORPUS_FIXTURE_NAMES.filter((name) => !loadedCorpusFixtureNames.has(name)),
          passed: cases.length - failed.length,
          failed: failed.length,
        };
      };

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
          });
        }
        return cases;
      }

      function materializeFixtureInput(input) {
        const buffers = {};
        for (const [name, spec] of Object.entries(input.buffers ?? {})) {
          buffers[name] = materializeTypedArray(spec);
        }
        return {
          buffers,
          scalars: { ...(input.scalars ?? {}) },
        };
      }

      function materializeTypedArray(spec) {
        const Ctor = globalThis[spec.type];
        if (typeof Ctor !== "function") throw new Error("unsupported fixture typed array: " + spec.type);
        if (spec.data !== undefined) return new Ctor(spec.data);
        return new Ctor(spec.length ?? 0);
      }

      async function runReferenceWebGpuCase(device, spec) {
        const compiled = compileCudaLiteKernelForWebGpu(spec.source, spec.options);
        const actualInput = spec.input();
        const plan = createCudaWebGpuExecutionPlan(compiled, actualInput, spec.launch, {
          compileKernel: compileCudaLiteKernelForWebGpu,
        });
        if (!plan.supported) {
          return { ok: false, plan: plan.reason, maxAbsDiff: Number.POSITIVE_INFINITY };
        }
        const expectedInput = spec.offsetOutput && actualInput.readback
          ? { ...actualInput, readback: actualInput.readback.filter((name) => name !== spec.offsetOutput) }
          : actualInput;
        const expected = runCompiledKernelReference(compiled, expectedInput, spec.launch);
        const actual = await runCompiledKernelWebGpu(device, compiled, actualInput, spec.launch);
        const comparison = compareArrays(expected.buffers[spec.output], actual.buffers[spec.output]);
        if (comparison.ok && spec.offsetOutput) {
          const offset = actual.buffers[spec.offsetOutput]?.[0];
          if (offset !== spec.expectedOffset) {
            return { ok: false, plan: plan.kind, maxAbsDiff: Math.abs(Number(offset ?? NaN) - spec.expectedOffset) };
          }
        }
        return { ...comparison, plan: plan.kind };
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

      function compareArrays(expected, actual) {
        if (!expected || !actual || expected.length !== actual.length) {
          return { ok: false, maxAbsDiff: Number.POSITIVE_INFINITY };
        }
        let maxAbsDiff = 0;
        let firstDiff;
        for (let i = 0; i < expected.length; i++) {
          const diff = Math.abs(Number(expected[i]) - Number(actual[i]));
          if (diff > maxAbsDiff) maxAbsDiff = diff;
          if (firstDiff === undefined && diff > 1e-5) {
            firstDiff = {
              index: i,
              expected: Number(expected[i]),
              actual: Number(actual[i]),
              absDiff: diff,
            };
          }
        }
        return {
          ok: maxAbsDiff <= 1e-5,
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
    alias: moduleAliases(bundle),
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
  await page.goto(new URL("/__bg_cuda_lite_e2e__", baseUrl).href);
  const result = await page.evaluate(() => globalThis.__bgRunE2e());
  const report = {
    tool: "browsergrad-cuda-lite-webgpu-e2e",
    bundle,
    userAgent: await page.evaluate(() => navigator.userAgent),
    ...result,
  };
  if (requireWebGpu && !report.available) {
    throw new Error(`WebGPU unavailable: ${report.reason ?? "unknown"}`);
  }
  if (report.available && requireCorpusFixtures && (report.missingCorpusFixtureNames?.length ?? 0) > 0) {
    throw new Error(`Missing corpus execution fixtures from /tmp corpora: ${report.missingCorpusFixtureNames.join(", ")}`);
  }
  console.log(JSON.stringify(report, null, 2));
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

function markdownReport(data) {
  const lines = [
    "# BrowserGrad CUDA-lite WebGPU e2e",
    "",
    `Bundle: \`${data.bundle ?? "src"}\``,
    `User agent: \`${data.userAgent ?? "unknown"}\``,
    `Available: \`${data.available}\``,
    "",
  ];
  if (!data.available) {
    lines.push(`Reason: ${data.reason ?? "unknown"}`, "");
    return `${lines.join("\n")}\n`;
  }
  lines.push("| Case | Plan | Output | Max abs diff | OK | ms |");
  lines.push("| --- | --- | --- | ---: | --- | ---: |");
  for (const item of data.cases ?? []) {
    lines.push(`| \`${item.name}\` | \`${item.plan}\` | \`${item.output}\` | ${item.maxAbsDiff} | \`${item.ok}\` | ${item.ms} |`);
  }
  lines.push("", `Passed: \`${data.passed}\`, failed: \`${data.failed}\``, "");
  lines.push(`Corpus fixtures: \`${data.corpusFixturePassed ?? 0}/${data.corpusFixtureCases ?? 0}\``, "");
  return `${lines.join("\n")}\n`;
}

function loadCorpusExecutionSources() {
  const out = {};
  for (const fixture of cudaLiteCorpusExecutionFixtures) {
    const source = loadNormalizedCorpusKernelSource(fixture);
    if (source) out[fixture.sourceKey] = source;
  }
  return out;
}

function loadNormalizedCorpusKernelSource(fixture) {
  const corpus = corpusById(fixture.corpusId);
  const result = spawnSync(process.execPath, [
    path.join(root, "scripts/audit-cuda-lite-corpus.mjs"),
    corpus.path,
    "--emit-kernel-source",
    fixture.relativePath,
    "--kernel-name",
    fixture.kernelName,
  ], {
    cwd: root,
    encoding: "utf8",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`could not load normalized corpus fixture ${fixture.caseName}: ${result.stderr || result.stdout}`);
  }
  return JSON.parse(result.stdout).source;
}

function verifyCorpusFixtureCheckouts() {
  const seen = new Set();
  for (const fixture of cudaLiteCorpusExecutionFixtures) {
    if (seen.has(fixture.corpusId)) continue;
    seen.add(fixture.corpusId);
    const corpus = corpusById(fixture.corpusId);
    const gitDir = path.join(corpus.path, ".git");
    if (!fs.existsSync(gitDir)) {
      throw new Error(`${corpus.id} expected pinned git checkout at ${corpus.path}`);
    }
    const result = spawnSync("git", ["-C", corpus.path, "rev-parse", "HEAD"], {
      cwd: root,
      encoding: "utf8",
    });
    if (result.error) throw result.error;
    if (result.status !== 0) {
      throw new Error(`${corpus.id} could not verify pinned checkout: ${result.stderr}`);
    }
    const actual = result.stdout.trim();
    if (actual !== corpus.commit) {
      throw new Error(`${corpus.id} expected ${corpus.commit}, got ${actual}`);
    }
    const status = spawnSync("git", ["-C", corpus.path, "status", "--short", "--untracked-files=all"], {
      cwd: root,
      encoding: "utf8",
    });
    if (status.error) throw status.error;
    if (status.status !== 0) {
      throw new Error(`${corpus.id} could not verify clean checkout: ${status.stderr}`);
    }
    const dirty = status.stdout.trim();
    if (dirty.length > 0) {
      throw new Error(`${corpus.id} checkout at ${corpus.path} is dirty; clean or refresh it before browser e2e:\n${dirty}`);
    }
  }
}

function findRepoRoot(start) {
  let current = path.resolve(start);
  while (true) {
    if (fs.existsSync(path.join(current, "pnpm-workspace.yaml"))) return current;
    const parent = path.dirname(current);
    if (parent === current) return path.resolve(start);
    current = parent;
  }
}

function parseBundle(value) {
  if (value === "src" || value === "dist") return value;
  throw new Error("--bundle expects src or dist");
}

function moduleAliases(bundleName) {
  const compilerEntry = bundleName === "dist"
    ? path.join(root, "packages/browsergrad-compiler/dist/index.js")
    : path.join(root, "packages/browsergrad-compiler/src/index.ts");
  const kernelsEntry = bundleName === "dist"
    ? path.join(root, "packages/browsergrad-kernels/dist/index.js")
    : path.join(root, "packages/browsergrad-kernels/src/index.ts");
  return {
    "@unlocalhosted/browsergrad-kernels": kernelsEntry,
    "@unlocalhosted/browsergrad-compiler": compilerEntry,
  };
}
