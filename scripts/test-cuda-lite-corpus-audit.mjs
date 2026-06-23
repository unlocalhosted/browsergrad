#!/usr/bin/env node
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "browsergrad-cuda-audit-"));

try {
  fs.writeFileSync(path.join(tmpRoot, "defs.h"), `
typedef unsigned int TColor;

__device__ TColor make_color(float value) {
  return (TColor)value;
}
`);
  fs.writeFileSync(path.join(tmpRoot, "kernel.cuh"), `
__global__ void Copy(TColor *dst) {
  int i = threadIdx.x;
  dst[i] = make_color((float)i);
}
`);
  fs.writeFileSync(path.join(tmpRoot, "main.cu"), `
#include "defs.h"
#include "kernel.cuh"

void launch(TColor *dst) {
  Copy<<<1, 32>>>(dst);
}
`);

  const result = spawnSync("node", ["scripts/audit-cuda-lite-corpus.mjs", tmpRoot, "--details"], {
    cwd: repoRoot,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
  });
  if (result.status !== 0) {
    process.stderr.write(result.stderr);
    process.stderr.write(result.stdout);
    process.exit(result.status ?? 1);
  }
  const report = JSON.parse(result.stdout.slice(result.stdout.indexOf("{")));
  assertEqual(report.summary.totalKernelDefinitions, 1, "total kernel count");
  assertEqual(report.summary.webGpuRunnableOk, 1, "reverse include kernel WebGPU runnable");
  assertEqual(report.summary.hardFail, 0, "reverse include hard gaps");
  console.log("cuda-lite corpus audit tests passed");
} finally {
  fs.rmSync(tmpRoot, { recursive: true, force: true });
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${expected}, got ${actual}`);
  }
}
