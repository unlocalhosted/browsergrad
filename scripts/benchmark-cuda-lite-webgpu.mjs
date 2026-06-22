#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const args = new Map();
for (let i = 2; i < process.argv.length; i++) {
  const key = process.argv[i];
  const value = process.argv[i + 1];
  if (!key?.startsWith("--")) continue;
  args.set(key, value?.startsWith("--") || value === undefined ? "true" : value);
  if (value && !value.startsWith("--")) i++;
}

const runs = positiveInt(args.get("--runs"), 25);
const warmup = positiveInt(args.get("--warmup"), 5);
const length = positiveInt(args.get("--length"), 16_384);
const markdownPath = args.get("--markdown");
const headed = args.get("--headed") === "true";
const requireWebGpu = args.get("--require-webgpu") === "true";

const root = findRepoRoot(process.cwd());
const packageRequire = createRequire(path.join(root, "packages/browsergrad-compiler/package.json"));
const { createServer } = await import(pathToFileURL(packageRequire.resolve("vite")).href);
const playwright = await import(pathToFileURL(packageRequire.resolve("playwright")).href);
const chromium = playwright.chromium ?? playwright.default?.chromium;
if (!chromium) throw new Error("could not load Playwright chromium");

const saxpySource = `
__global__ void saxpy(const float* x, float* y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}
`;

const html = String.raw`<!doctype html>
<html>
  <head><meta charset="utf-8"><title>BrowserGrad CUDA-lite WebGPU bench</title></head>
  <body>
    <script type="module">
      import {
        createDevice,
        createWgslStorageBuffer,
        destroyWgslStorageBuffer,
        readWgslStorageBuffer,
        writeWgslStorageBuffer,
      } from "@unlocalhosted/browsergrad-kernels";
      import {
        compileCudaLiteKernel,
        prepareCompiledKernelWebGpu,
        runCompiledKernelWebGpu,
      } from "@unlocalhosted/browsergrad-compiler";

      const SAXPY = ${JSON.stringify(saxpySource)};

      window.__bgRunBench = async ({ runs, warmup, length }) => {
        if (!navigator.gpu) {
          return {
            available: false,
            reason: "navigator.gpu undefined",
            runs,
            warmup,
            length,
            benchmarks: [],
          };
        }

        const device = await createDevice();
        const compiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [256, 1, 1] });
        const launch = {
          gridDim: [Math.ceil(length / 256), 1, 1],
          blockDim: [256, 1, 1],
        };
        const xData = filledFloat32(length, 1);
        const yInitial = filledFloat32(length, 2);
        const x = createWgslStorageBuffer(device, {
          valueType: "f32",
          data: xData,
          label: "bench-x",
        });
        const y = createWgslStorageBuffer(device, {
          valueType: "f32",
          data: yInitial,
          label: "bench-y",
        });

        try {
          const input = {
            buffers: {},
            residentBuffers: { x, y },
            scalars: { a: 3, n: length },
            readback: [],
          };
          const prepareMs = await timeOnce(async () => {
            const prepared = await prepareCompiledKernelWebGpu(device, compiled, input, launch);
            prepared.destroy();
          });
          const prepared = await prepareCompiledKernelWebGpu(device, compiled, input, launch);
          try {
            const oneShot = await measure("webgpu:saxpy-one-shot-resident", runs, warmup, async () => {
              writeWgslStorageBuffer(device, y, yInitial);
              await runCompiledKernelWebGpu(device, compiled, input, launch);
              await device.gpu.queue.onSubmittedWorkDone();
            });
            const preparedRun = await measure("webgpu:saxpy-prepared-resident", runs, warmup, async () => {
              writeWgslStorageBuffer(device, y, yInitial);
              await prepared.run({ readback: [], awaitCompletion: true });
            });
            let scalar = 4;
            const preparedScalarUpdate = await measure("webgpu:saxpy-prepared-scalar-update", runs, warmup, async () => {
              writeWgslStorageBuffer(device, y, yInitial);
              scalar = scalar === 4 ? 3 : 4;
              await prepared.run({ scalars: { a: scalar }, readback: [], awaitCompletion: true });
            });
            writeWgslStorageBuffer(device, y, yInitial);
            await prepared.run({ scalars: { a: 4 }, readback: [], awaitCompletion: true });
            const out = await readWgslStorageBuffer(device, y);
            const ok = out[0] === 6 && out[length - 1] === 6;
            return {
              available: true,
              runs,
              warmup,
              length,
              benchmarks: [
                { name: "webgpu:prepare-compiled-saxpy", minMs: prepareMs, medianMs: prepareMs, p95Ms: prepareMs, maxMs: prepareMs },
                oneShot,
                preparedRun,
                preparedScalarUpdate,
              ],
              validation: { saxpy: ok },
            };
          } finally {
            prepared.destroy();
          }
        } finally {
          destroyWgslStorageBuffer(x);
          destroyWgslStorageBuffer(y);
        }
      };

      async function measure(name, runs, warmup, fn) {
        for (let i = 0; i < warmup; i++) await fn();
        const samples = [];
        for (let i = 0; i < runs; i++) {
          samples.push(await timeOnce(fn));
        }
        samples.sort((a, b) => a - b);
        return {
          name,
          minMs: round(samples[0] ?? 0),
          medianMs: round(percentile(samples, 0.5)),
          p95Ms: round(percentile(samples, 0.95)),
          maxMs: round(samples[samples.length - 1] ?? 0),
        };
      }

      async function timeOnce(fn) {
        const start = performance.now();
        await fn();
        return round(performance.now() - start);
      }

      function filledFloat32(length, value) {
        const out = new Float32Array(length);
        out.fill(value);
        return out;
      }

      function percentile(sorted, q) {
        if (sorted.length === 0) return 0;
        const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * q) - 1));
        return sorted[index] ?? 0;
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
    name: "browsergrad-cuda-lite-bench-page",
    configureServer(viteServer) {
      viteServer.middlewares.use("/__bg_cuda_lite_bench__", async (req, res) => {
        try {
          const transformed = await viteServer.transformIndexHtml(req.originalUrl ?? "/__bg_cuda_lite_bench__", html);
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
  await page.goto(new URL("/__bg_cuda_lite_bench__", baseUrl).href);
  const result = await page.evaluate(
    ({ runs, warmup, length }) => globalThis.__bgRunBench({ runs, warmup, length }),
    { runs, warmup, length },
  );
  const report = {
    tool: "browsergrad-cuda-lite-webgpu-benchmark",
    userAgent: await page.evaluate(() => navigator.userAgent),
    ...result,
  };
  if (requireWebGpu && !report.available) {
    throw new Error(`WebGPU unavailable: ${report.reason ?? "unknown"}`);
  }
  console.log(JSON.stringify(report, null, 2));
  if (markdownPath && markdownPath !== "true") {
    fs.writeFileSync(path.resolve(markdownPath), markdownReport(report));
  }
} finally {
  if (browser) await browser.close();
  await server.close();
}

function markdownReport(data) {
  const lines = [
    "# BrowserGrad CUDA-lite WebGPU benchmark",
    "",
    `User agent: \`${data.userAgent ?? "unknown"}\``,
    `Available: \`${data.available}\``,
    `Runs: \`${data.runs}\`, warmup: \`${data.warmup}\`, length: \`${data.length}\``,
    "",
  ];
  if (!data.available) {
    lines.push(`Reason: ${data.reason ?? "unknown"}`, "");
    return `${lines.join("\n")}\n`;
  }
  lines.push("| Benchmark | min ms | median ms | p95 ms | max ms |");
  lines.push("| --- | ---: | ---: | ---: | ---: |");
  for (const bench of data.benchmarks) {
    lines.push(`| \`${bench.name}\` | ${bench.minMs} | ${bench.medianMs} | ${bench.p95Ms} | ${bench.maxMs} |`);
  }
  lines.push("", `Validation: \`${JSON.stringify(data.validation ?? {})}\``, "");
  return `${lines.join("\n")}\n`;
}

function positiveInt(value, fallback) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function findRepoRoot(start) {
  let dir = path.resolve(start);
  while (dir !== path.dirname(dir)) {
    if (fs.existsSync(path.join(dir, "pnpm-workspace.yaml"))) return dir;
    dir = path.dirname(dir);
  }
  throw new Error("could not find repo root");
}
