#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { pathToFileURL } from "node:url";

const args = new Map();
for (let i = 2; i < process.argv.length; i++) {
  const key = process.argv[i];
  const value = process.argv[i + 1];
  if (!key?.startsWith("--")) continue;
  args.set(key, value?.startsWith("--") || value === undefined ? "true" : value);
  if (value && !value.startsWith("--")) i++;
}

const runs = positiveInt(args.get("--runs"), 40);
const warmup = positiveInt(args.get("--warmup"), 8);
const markdownPath = args.get("--markdown");
const medianMaxExpectations = parseBenchmarkThresholds(args.get("--expect-median-max"), "--expect-median-max");
const p95MaxExpectations = parseBenchmarkThresholds(args.get("--expect-p95-max"), "--expect-p95-max");
const root = findRepoRoot(process.cwd());
const compilerUrl = pathToFileURL(path.join(root, "packages/browsergrad-compiler/dist/index.js")).href;
const {
  compileCudaLiteKernelForWebGpu,
  compileCudaLiteKernel,
  createCudaHostDynamicLaunchPlan,
  createCudaPeerCopyPlan,
  runCompiledKernelReference,
} = await import(compilerUrl);

function main() {
  const saxpyCompiled = compileCudaLiteKernel(SAXPY, { workgroupSize: [256, 1, 1] });
  const dynamicCompiled = compileCudaLiteKernelForWebGpu(DYNAMIC_OFFSET, {
    kernelName: "parent",
    workgroupSize: [1, 1, 1],
  });
  const peerCopyCompiled = compileCudaLiteKernelForWebGpu(PEER_COPY, {
    workgroupSize: [1, 1, 1],
  });

  const benchmarks = [
    {
      name: "compile:saxpy",
      fn: () => compileCudaLiteKernel(SAXPY, { workgroupSize: [256, 1, 1] }),
    },
    {
      name: "compile:tiled-matmul-shared",
      fn: () => compileCudaLiteKernel(TILED_MATMUL, { workgroupSize: [8, 8, 1] }),
    },
    {
      name: "reference:saxpy-16k",
      fn: () => runCompiledKernelReference(
        saxpyCompiled,
        {
          buffers: {
            x: filledFloat32(16_384, 1),
            y: filledFloat32(16_384, 2),
          },
          scalars: { a: 3, n: 16_384 },
        },
        { gridDim: [64, 1, 1], blockDim: [256, 1, 1] },
      ),
    },
    {
      name: "plan:host-dynamic-pointer-offset",
      fn: () => createCudaHostDynamicLaunchPlan(
        dynamicCompiled,
        { buffers: { out: new Float32Array(2) } },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      ),
    },
    {
      name: "plan:peer-copy",
      fn: () => createCudaPeerCopyPlan(
        peerCopyCompiled,
        {
          buffers: {
            dst: new Float32Array(4),
            src: new Float32Array([2.5, 3.5]),
          },
          scalars: { n: 2 },
        },
        { gridDim: [1, 1, 1], blockDim: [1, 1, 1] },
      ),
    },
  ];

  const measured = benchmarks.map((bench) => measure(bench.name, bench.fn, { runs, warmup }));
  const thresholdChecks = [
    ...assertBenchmarkThresholds(measured, "medianMs", medianMaxExpectations, "--expect-median-max"),
    ...assertBenchmarkThresholds(measured, "p95Ms", p95MaxExpectations, "--expect-p95-max"),
  ];
  const result = {
    tool: "browsergrad-cuda-lite-compiler-benchmark",
    node: process.version,
    runs,
    warmup,
    benchmarks: measured,
    thresholdChecks,
  };

  console.log(JSON.stringify(result, null, 2));
  if (markdownPath && markdownPath !== "true") {
    fs.writeFileSync(path.resolve(markdownPath), markdownReport(result));
  }
}

function measure(name, fn, options) {
  for (let i = 0; i < options.warmup; i++) fn();
  const samples = [];
  for (let i = 0; i < options.runs; i++) {
    const start = performance.now();
    fn();
    samples.push(performance.now() - start);
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

function percentile(sorted, q) {
  if (sorted.length === 0) return 0;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * q) - 1));
  return sorted[index] ?? 0;
}

function round(value) {
  return Math.round(value * 1000) / 1000;
}

function positiveInt(value, fallback) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function parseBenchmarkThresholds(value, flag) {
  if (value === undefined || value === "true") return new Map();
  const out = new Map();
  for (const rawEntry of String(value).split(",")) {
    const entry = rawEntry.trim();
    if (entry.length === 0) continue;
    const separator = entry.lastIndexOf("=");
    if (separator <= 0 || separator === entry.length - 1) {
      throw new Error(`${flag} expects comma-separated benchmark=maxMs entries`);
    }
    const name = entry.slice(0, separator);
    const maxMs = Number(entry.slice(separator + 1));
    if (!Number.isFinite(maxMs) || maxMs <= 0) {
      throw new Error(`${flag} expects positive max ms for ${name}`);
    }
    out.set(name, maxMs);
  }
  return out;
}

function assertBenchmarkThresholds(benchmarks, metric, expectations, flag) {
  const checks = [];
  for (const [name, maxMs] of expectations) {
    const benchmark = benchmarks.find((item) => item.name === name);
    if (!benchmark) throw new Error(`${flag} unknown benchmark: ${name}`);
    const actualMs = benchmark[metric];
    if (typeof actualMs !== "number") throw new Error(`${flag} missing ${metric} for ${name}`);
    const check = { name, metric, actualMs, maxMs, ok: actualMs <= maxMs };
    checks.push(check);
    if (!check.ok) {
      throw new Error(`${flag} failed: ${name} ${metric} ${actualMs}ms > ${maxMs}ms`);
    }
  }
  return checks;
}

function filledFloat32(length, value) {
  const out = new Float32Array(length);
  out.fill(value);
  return out;
}

function markdownReport(data) {
  const lines = [
    "# BrowserGrad CUDA-lite compiler benchmark",
    "",
    `Node: \`${data.node}\``,
    `Runs: \`${data.runs}\`, warmup: \`${data.warmup}\``,
    "",
    "| Benchmark | min ms | median ms | p95 ms | max ms |",
    "| --- | ---: | ---: | ---: | ---: |",
  ];
  for (const bench of data.benchmarks) {
    lines.push(`| \`${bench.name}\` | ${bench.minMs} | ${bench.medianMs} | ${bench.p95Ms} | ${bench.maxMs} |`);
  }
  if ((data.thresholdChecks?.length ?? 0) > 0) {
    lines.push("", "| Threshold | metric | actual ms | max ms | ok |");
    lines.push("| --- | --- | ---: | ---: | --- |");
    for (const check of data.thresholdChecks) {
      lines.push(`| \`${check.name}\` | \`${check.metric}\` | ${check.actualMs} | ${check.maxMs} | \`${check.ok}\` |`);
    }
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

function findRepoRoot(start) {
  let dir = path.resolve(start);
  while (dir !== path.dirname(dir)) {
    if (fs.existsSync(path.join(dir, "pnpm-workspace.yaml"))) return dir;
    dir = path.dirname(dir);
  }
  throw new Error("could not find repo root");
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
  __shared__ float As[8][8];
  __shared__ float Bs[8][8];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
  float acc = 0.0;
  for (int t = 0; t < N; t += 8) {
    if (row < N && (t + tx) < N) { As[ty][tx] = A[row * N + t + tx]; }
    if (col < N && (t + ty) < N) { Bs[ty][tx] = B[(t + ty) * N + col]; }
    __syncthreads();
    for (int k = 0; k < 8; k++) {
      if ((t + k) < N) { acc += As[ty][k] * Bs[k][tx]; }
    }
    __syncthreads();
  }
  if (row < N && col < N) { C[row * N + col] = acc; }
}
`;

const DYNAMIC_OFFSET = `
__global__ void child(float *out) {
  if (threadIdx.x < 1) { out[0] = 7.0f; }
}
__global__ void parent(float *out) {
  if (threadIdx.x < 1) {
    dim3 grid(1);
    dim3 block(1);
    child<<<grid, block>>>(out + 1);
    cudaDeviceSynchronize();
  }
}
`;

const PEER_COPY = `
__global__ void peerCopy(float *dst, const float *src, int n) {
  if (threadIdx.x == 0) {
    cudaMemcpyPeerAsync(dst + 1, 1, src, 0, sizeof(float) * n, 0);
  }
}
`;

main();
