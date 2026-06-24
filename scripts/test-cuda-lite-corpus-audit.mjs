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

class DeviceBox {
public:
  __device__ ~DeviceBox() {}
};

__device__ TColor make_color(float value) {
  return (TColor)value;
}

typedef struct {
  float x;
  float y;
} Pair;

__device__ Pair make_pair(float value) {
  Pair pair = {value, value + 1.0f};
  return pair;
}

struct SoftmaxParams {
  float Scale;
  float Offset;
};

namespace cg = cooperative_groups;

__device__ SoftmaxParams prepare_softmax_like(cg::thread_block_tile<32>& tile, const float *input) {
  SoftmaxParams params = {(float)tile.size(), input[tile.thread_rank()]};
  return params;
}

template<class T>
__device__ float sum_range_like(const T* data, size_t count) {
  size_t index = threadIdx.x;
  float accumulator = 0.0f;
  for (size_t i = index; i < count; i += blockDim.x) {
    accumulator += (float)data[i];
  }
  return accumulator;
}

template<typename Td, typename Ts>
__device__ Td cast_value_like(Ts val);

template<>
__device__ float cast_value_like<float, float>(float val) {
  return val;
}
`);
  fs.writeFileSync(path.join(tmpRoot, "palette.h"), `
typedef unsigned int ExternalColor;
`);
  fs.writeFileSync(path.join(tmpRoot, "kernel.cuh"), `
__global__ void Copy(TColor *dst) {
  int i = threadIdx.x;
  dst[i] = make_color((float)i);
}

__global__ void PairCopy(float *dst) {
  Pair pair = make_pair(dst[0]);
  dst[0] = pair.x + pair.y;
}

__global__ void GroupHelper(const float *input, float *out) {
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
  SoftmaxParams params = prepare_softmax_like(tile, input);
  out[threadIdx.x] = params.Scale + params.Offset;
}

__global__ void TemplateHelper(float *out, const float *data, size_t count) {
  out[threadIdx.x] = sum_range_like(data, count);
}

__global__ void TemplateSpecializationHelper(float *out, const float *data) {
  out[threadIdx.x] = cast_value_like<float, float>(data[threadIdx.x]);
}

__global__ void GlobalAlias(ExternalColor *dst) {
  dst[threadIdx.x] = (ExternalColor)threadIdx.x;
}

__global__ void DynamicVectorShared(uchar4 *out) {
  extern __shared__ uchar4 scratch[];
  int i = threadIdx.x;
  scratch[i] = make_uchar4((uint)i, (uint)(i + 1), (uint)(i + 2), (uint)(i + 3));
  out[i] = scratch[i];
}

__global__ void DynamicAlignedByteShared(float *out) {
  extern __shared__ __align__(16) unsigned char bytes[];
  bytes[threadIdx.x] = (unsigned char)threadIdx.x;
  __syncthreads();
  if (threadIdx.x == 0) out[0] = (float)bytes[1];
}

__global__ void DynamicLateQualifierShared(float *out) {
  extern double __shared__ values[];
  values[threadIdx.x] = (double)threadIdx.x;
  __syncthreads();
  if (threadIdx.x == 0) out[0] = (float)values[1];
}
`);
  fs.writeFileSync(path.join(tmpRoot, "main.cu"), `
#include "defs.h"
#include "kernel.cuh"

void launch(TColor *dst) {
  Copy<<<1, 32>>>(dst);
}

void launch_pair(float *dst) {
  PairCopy<<<1, 1>>>(dst);
}

void launch_group(const float *input, float *out) {
  GroupHelper<<<1, 32>>>(input, out);
}

void launch_template(float *out, const float *data, size_t count) {
  TemplateHelper<<<1, 32>>>(out, data, count);
}

void launch_template_specialization(float *out, const float *data) {
  TemplateSpecializationHelper<<<1, 32>>>(out, data);
}

void launch_global_alias(ExternalColor *dst) {
  GlobalAlias<<<1, 32>>>(dst);
}

void launch_dynamic_vector(uchar4 *out) {
  DynamicVectorShared<<<1, 32, 512>>>(out);
}

void launch_dynamic_aligned(float *out) {
  DynamicAlignedByteShared<<<1, 32, 512>>>(out);
}

void launch_dynamic_late(float *out) {
  DynamicLateQualifierShared<<<1, 32, 512>>>(out);
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
  assertEqual(report.summary.totalKernelDefinitions, 9, "total kernel count");
  assertEqual(report.summary.corpusKernelExecution, "compile-only", "corpus execution mode");
  assertEqual(report.summary.corpusExecutionMode, "compile-only", "corpus execution mode alias");
  assertEqual(report.summary.executionTierCounts.compileCodegenOnlyOk, 9, "compile/codegen-only tier count");
  assertEqual(report.summary.executionTierCounts.fixtureBackedExecutedOk, 0, "fixture execution tier count");
  assertEqual(report.summary.executionTierCounts.browserWebGpuExecutedOk, 0, "browser execution tier count");
  assertEqual(report.summary.executionTierCounts.outputVerifiedOk, 0, "output verified tier count");
  assertEqual(report.summary.webGpuCompiledOk, 9, "reverse include kernel WebGPU compiled");
  assertEqual(report.summary.fixtureBackedExecutionOk, 0, "fixture-backed execution count");
  assertEqual(report.summary.webGpuRunnableOk, 9, "reverse include kernel WebGPU runnable");
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
