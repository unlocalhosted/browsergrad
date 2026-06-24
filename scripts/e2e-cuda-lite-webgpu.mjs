#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

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

const root = findRepoRoot(process.cwd());
const packageRequire = createRequire(path.join(root, "packages/browsergrad-compiler/package.json"));
const { createServer } = await import(pathToFileURL(packageRequire.resolve("vite")).href);
const playwright = await import(pathToFileURL(packageRequire.resolve("playwright")).href);
const chromium = playwright.chromium ?? playwright.default?.chromium;
if (!chromium) throw new Error("could not load Playwright chromium");

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
  ...loadCorpusExecutionSources(),
};

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
        return {
          available: true,
          cases,
          corpusFixtureCases: corpusFixtureCases.length,
          corpusFixturePassed: corpusFixtureCases.filter((item) => item.ok).length,
          corpusFixtureFailed: corpusFixtureCases.filter((item) => !item.ok).length,
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
        ];
        if (SOURCES.corpusCuda120VectorAddKernel) {
          cases.push({
            name: "corpus:cuda-120:vectorAddKernel",
            source: SOURCES.corpusCuda120VectorAddKernel,
            options: { workgroupSize: [8, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
            input: () => ({
              buffers: {
                A: new Float32Array([1, 2, 3, 4]),
                B: new Float32Array([10, 20, 30, 40]),
                C: new Float32Array(4),
              },
              scalars: { N: 4 },
            }),
            output: "C",
          });
        }
        if (SOURCES.corpusLlmAddBias) {
          cases.push({
            name: "corpus:llm.c:add_bias",
            source: SOURCES.corpusLlmAddBias,
            options: { workgroupSize: [8, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
            input: () => ({
              buffers: {
                out: new Float32Array([1, 2, 3, 4, 5, 6]),
                bias: new Float32Array([0.5, 1.5, -2]),
              },
              scalars: { B: 1, T: 2, OC: 3 },
            }),
            output: "out",
          });
        }
        if (SOURCES.corpusLlmSetVector) {
          cases.push({
            name: "corpus:llm.c:set_vector",
            source: SOURCES.corpusLlmSetVector,
            options: { workgroupSize: [8, 1, 1] },
            launch: { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
            input: () => ({
              buffers: {
                data: new Float32Array(4),
              },
              scalars: { N: 4, value: 7 },
            }),
            output: "data",
          });
        }
        return cases;
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
        for (let i = 0; i < expected.length; i++) {
          const diff = Math.abs(Number(expected[i]) - Number(actual[i]));
          if (diff > maxAbsDiff) maxAbsDiff = diff;
        }
        return { ok: maxAbsDiff <= 1e-5, maxAbsDiff: round(maxAbsDiff) };
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
    alias: {
      "@unlocalhosted/browsergrad-kernels": path.join(root, "packages/browsergrad-kernels/src/index.ts"),
      "@unlocalhosted/browsergrad-compiler": path.join(root, "packages/browsergrad-compiler/src/index.ts"),
    },
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
    userAgent: await page.evaluate(() => navigator.userAgent),
    ...result,
  };
  if (requireWebGpu && !report.available) {
    throw new Error(`WebGPU unavailable: ${report.reason ?? "unknown"}`);
  }
  if (report.available && requireCorpusFixtures && report.corpusFixtureCases === 0) {
    throw new Error("No corpus execution fixtures were loaded from /tmp corpora");
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
  const cuda120 = extractGlobalKernelFromFile(
    "/tmp/CUDA-120-DAYS--CHALLENGE/daily-updates/day-23-Asynchronous-Memory-Copy.md",
    "vectorAddKernel",
  );
  if (cuda120) out.corpusCuda120VectorAddKernel = cuda120;
  const llmAddBias = extractGlobalKernelFromFile(
    "/tmp/browsergrad-corpora/llm.c/dev/cuda/matmul_forward.cu",
    "add_bias",
  );
  if (llmAddBias) out.corpusLlmAddBias = llmAddBias;
  const llmSetVector = extractGlobalKernelFromFile(
    "/tmp/browsergrad-corpora/llm.c/dev/cuda/nccl_all_reduce.cu",
    "set_vector",
  );
  if (llmSetVector) out.corpusLlmSetVector = llmSetVector;
  return out;
}

function extractGlobalKernelFromFile(filePath, name) {
  if (!fs.existsSync(filePath)) return undefined;
  return extractGlobalKernel(fs.readFileSync(filePath, "utf8"), name);
}

function extractGlobalKernel(source, name) {
  const re = new RegExp(`__global__\\s+void\\s+${escapeRegExp(name)}\\s*\\(`, "u");
  const match = re.exec(source);
  if (!match) return undefined;
  const start = match.index;
  const bodyStart = source.indexOf("{", match.index);
  if (bodyStart < 0) return undefined;
  const end = findBalanced(source, bodyStart, "{", "}");
  if (end === undefined) return undefined;
  return source.slice(start, end + 1);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function findBalanced(source, openIndex, open, close) {
  let depth = 0;
  for (let index = openIndex; index < source.length; index++) {
    const char = source[index];
    if (char === open) depth++;
    if (char === close) {
      depth--;
      if (depth === 0) return index;
    }
  }
  return undefined;
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
