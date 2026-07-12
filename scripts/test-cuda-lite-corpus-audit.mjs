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

struct MultiDeviceData {
  unsigned int numDevices;
  unsigned int deviceRank;
};

class PeerGroup {
  const MultiDeviceData &data;
  const cg::grid_group &grid;

public:
  __device__ PeerGroup(const MultiDeviceData &data, const cg::grid_group &grid)
      : data(data), grid(grid) {}

  __device__ unsigned int size() const { return data.numDevices * grid.size(); }
  __device__ unsigned int thread_rank() const { return data.deviceRank * grid.size() + grid.thread_rank(); }
  __device__ void sync() const { grid.sync(); }
};

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

template<class T>
__global__ void TypedKernel(T *out) {
  out[threadIdx.x] = (T)threadIdx.x;
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

__global__ void GridSyncDouble(double *out) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  out[threadIdx.x] = out[threadIdx.x] + 1.0;
  cg::sync(grid);
  out[threadIdx.x] = out[threadIdx.x] + 1.0;
}

__global__ void DynamicNoopChild(float *out) {
  out[0] = 1.0f;
}

__global__ void InactiveDynamic(float *out) {
  if (0 != 0) {
    DynamicNoopChild<<<1, 1>>>(out);
    cudaDeviceSynchronize();
  }
  out[0] += 2.0f;
}

__device__ void retirement_fold(float *values, uint tid, const cg::thread_block &cta) {
  if (tid < 2u) values[tid] += values[tid + 2u];
  cg::sync(cta);
  if (tid == 0u) values[0] += values[1];
  cg::sync(cta);
}

__device__ void retirement_partial(const float *input, float *out, uint n, const cg::thread_block &cta) {
  extern __shared__ float partial[];
  uint tid = threadIdx.x;
  uint index = blockIdx.x * blockDim.x + tid;
  partial[tid] = index < n ? input[index] : 0.0f;
  cg::sync(cta);
  retirement_fold(partial, tid, cta);
  if (tid == 0u) out[blockIdx.x] = partial[0];
}

__device__ uint retirement_count = 0u;
__global__ void RetirementReduce(const float *input, float *out, uint n) {
  const cg::thread_block cta = cg::this_thread_block();
  retirement_partial(input, out, n, cta);
  if (gridDim.x > 1u) {
    uint tid = threadIdx.x;
    __shared__ bool amLast;
    extern __shared__ float scratch[];
    __threadfence();
    if (tid == 0u) {
      uint ticket = atomicInc(&retirement_count, gridDim.x);
      amLast = ticket == gridDim.x - 1u;
    }
    cg::sync(cta);
    if (amLast) {
      uint i = tid;
      float sum = 0.0f;
      while (i < gridDim.x) { sum += out[i]; i += blockDim.x; }
      scratch[tid] = sum;
      cg::sync(cta);
      retirement_fold(scratch, tid, cta);
      if (tid == 0u) { out[0] = scratch[0]; retirement_count = 0u; }
    }
  }
}

__global__ void multiGpuConjugateGradient(int *I,
                                          int *J,
                                          float *val,
                                          float *x,
                                          float *Ax,
                                          float *p,
                                          float *r,
                                          double *dot_result,
                                          int nnz,
                                          int N,
                                          float tol,
                                          MultiDeviceData multi_device_data) {
  cg::grid_group grid = cg::this_grid();
  PeerGroup peer_group(multi_device_data, grid);
  for (int i = peer_group.thread_rank(); i < N; i += peer_group.size()) {
    r[i] = 1.0f;
  }
  peer_group.sync();
  if (grid.thread_rank() == 0) *dot_result = 0.0;
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

void launch_typed_float(float *out) {
  TypedKernel<float><<<1, 32>>>(out);
}

void launch_typed_int(int *out) {
  TypedKernel<int><<<1, 32>>>(out);
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

void launch_grid_sync_double(double *out) {
  GridSyncDouble<<<1, 32, 512>>>(out);
}

void launch_inactive_dynamic(float *out) {
  InactiveDynamic<<<1, 1>>>(out);
}

void launch_retirement(const float *input, float *out, uint n) {
  RetirementReduce<<<3, 4, 32>>>(input, out, n);
}

void launch_multi_gpu_like(int *I, int *J, float *val, float *x, float *Ax, float *p, float *r, double *dot_result, int nnz, int N, float tol, MultiDeviceData data) {
  multiGpuConjugateGradient<<<1, 32>>>(I, J, val, x, Ax, p, r, dot_result, nnz, N, tol, data);
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
  assertEqual(report.summary.totalKernelDefinitions, 15, "total kernel count");
  assertEqual(report.summary.corpusKernelExecution, "compile-codegen-only", "corpus execution mode");
  assertEqual(report.summary.corpusExecutionMode, "compile-codegen-only", "corpus execution mode alias");
  assertEqual(report.summary.executionTierCounts.compileCodegenOnlyOk, 15, "compile/codegen-only tier count");
  assertEqual(report.summary.executionTierCounts.planCompiledOk, 15, "plan-compiled tier count");
  assertEqual(report.summary.executionTierCounts.planCompileGaps, 0, "plan-compiled gap count");
  assertEqual(report.summary.executionTierCounts.fixtureBackedExecutedOk, 0, "fixture execution tier count");
  assertEqual(report.summary.executionTierCounts.browserWebGpuExecutedOk, 0, "browser execution tier count");
  assertEqual(report.summary.executionTierCounts.outputVerifiedOk, 0, "output verified tier count");
  assertEqual(report.summary.planCompiledOk, 15, "plan compiled count");
  assertEqual(report.summary.planCompileGaps, 0, "plan compiled gaps");
  assertEqual(report.summary.singleDispatchPlanCompiledOk, 14, "single-dispatch plan compiled count");
  assertEqual(report.summary.hostOrchestratedPlanCompiledOk, 1, "host-orchestrated plan compiled count");
  assertEqual(report.summary.browserExecutedOk, 0, "browser executed count");
  assertEqual(report.summary.outputVerifiedOk, 0, "output verified count");
  assertEqual(report.summary.deprecatedCompilePlanAliases.webGpuRunnableOk, "planCompiledOk", "deprecated runnable alias");
  assertEqual(report.summary.webGpuDirectCompiledOk, 14, "reverse include kernel direct WGSL compiled");
  assertEqual(
    report.summary.semanticIrDirectWgslOk + report.summary.semanticIrHostPlanOk + report.summary.legacyAstDirectWgslFallback,
    report.summary.webGpuDirectCompiledOk,
    "semantic direct, semantic host, and AST fallback coverage partition",
  );
  assertEqual(report.summary.semanticIrHostPlanOk, 0, "semantic host-plan count");
  assertEqual(
    report.summary.semanticIrWebGpuOk,
    report.summary.semanticIrDirectWgslOk + report.summary.semanticIrHostPlanOk,
    "total semantic WebGPU coverage",
  );
  assertEqual(
    Object.values(report.summary.legacyAstDirectWgslBlockers).reduce((total, count) => total + count, 0),
    report.summary.legacyAstDirectWgslFallback,
    "AST direct WGSL blocker coverage partition",
  );
  assertEqual(report.summary.webGpuHostPlanCompiledOk, 1, "reverse include kernel host-plan compiled");
  assertEqual(report.summary.compileCodegenOk, 15, "reverse include kernel compile/codegen count");
  assertEqual(report.summary.compileCodegenGaps, 0, "reverse include kernel compile/codegen gaps");
  assertEqual(report.summary.fixtureBackedExecutionOk, 0, "fixture-backed execution count");
  assertEqual(report.summary.webGpuRunnableOk, undefined, "legacy runnable count omitted from top-level summary");
  assertEqual(report.summary.webGpuTotalOk, undefined, "legacy total count omitted from top-level summary");
  assertEqual(report.summary.webGpuCompiledOk, undefined, "legacy compiled count omitted from top-level summary");
  assertEqual(report.summary.webGpuSingleDispatchOk, undefined, "legacy single-dispatch count omitted from top-level summary");
  assertEqual(report.summary.webGpuLiftedOk, undefined, "legacy lifted count omitted from top-level summary");
  assertEqual(report.summary.webGpuHostOrchestratedOk, undefined, "legacy host-orchestrated count omitted from top-level summary");
  assertEqual(report.summary.hardFail, 0, "reverse include hard gaps");
  assertEqual(report.summary.referenceFallbackOk, 1, "runtime fallbacks compile through host plans");
  assertEqual(report.failures.length, 1, "only reachable strict direct compile has gaps");
  assertEqual(
    report.failures.find((failure) => failure.kernelName === "GridSyncDouble")?.webGpuPlanLiftKind,
    "grid-sync-phases",
    "grid sync fallback host plan kind",
  );
  assertEqual(report.failures.some((failure) => failure.kernelName === "InactiveDynamic"), false, "dead dynamic launch is pruned by semantic IR");

  const emitted = spawnSync("node", [
    "scripts/audit-cuda-lite-corpus.mjs",
    tmpRoot,
    "--emit-kernel-source",
    "kernel.cuh",
    "--kernel-name",
    "Copy",
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
  });
  if (emitted.status !== 0) {
    process.stderr.write(emitted.stderr);
    process.stderr.write(emitted.stdout);
    process.exit(emitted.status ?? 1);
  }
  const emittedSource = JSON.parse(emitted.stdout).source;
  assertIncludes(emittedSource, "make_color", "emitted normalized source includes helper context");
  assertIncludes(emittedSource, "__global__ void Copy", "emitted normalized source includes requested kernel");
  const emittedTypedInt = spawnSync("node", [
    "scripts/audit-cuda-lite-corpus.mjs",
    tmpRoot,
    "--emit-kernel-source",
    "kernel.cuh",
    "--kernel-name",
    "TypedKernel",
    "--kernel-template-args",
    "int",
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
  });
  if (emittedTypedInt.status !== 0) {
    process.stderr.write(emittedTypedInt.stderr);
    process.stderr.write(emittedTypedInt.stdout);
    process.exit(emittedTypedInt.status ?? 1);
  }
  const emittedTypedIntSource = JSON.parse(emittedTypedInt.stdout).source;
  assertIncludes(emittedTypedIntSource, "template<class T = int>", "template arg override updates default");
  assertIncludes(emittedTypedIntSource, "__global__ void TypedKernel(int *out)", "template arg override updates kernel param type");
  const emittedClassAdjacent = spawnSync("node", [
    "scripts/audit-cuda-lite-corpus.mjs",
    tmpRoot,
    "--emit-kernel-source",
    "kernel.cuh",
    "--kernel-name",
    "multiGpuConjugateGradient",
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
  });
  if (emittedClassAdjacent.status !== 0) {
    process.stderr.write(emittedClassAdjacent.stderr);
    process.stderr.write(emittedClassAdjacent.stdout);
    process.exit(emittedClassAdjacent.status ?? 1);
  }
  const emittedClassAdjacentSource = JSON.parse(emittedClassAdjacent.stdout).source;
  assertIncludes(
    emittedClassAdjacentSource,
    "__global__ void multiGpuConjugateGradient",
    "emitted normalized source keeps requested class-adjacent kernel",
  );
  assertNotIncludes(
    emittedClassAdjacentSource,
    "__device__ unsigned int thread_rank() const",
    "emitted normalized source excludes class member device methods",
  );
  console.log("cuda-lite corpus audit tests passed");
} finally {
  fs.rmSync(tmpRoot, { recursive: true, force: true });
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${expected}, got ${actual}`);
  }
}

function assertIncludes(actual, expected, label) {
  if (!String(actual).includes(expected)) {
    throw new Error(`${label}: expected source to include ${expected}`);
  }
}

function assertNotIncludes(actual, expected, label) {
  if (String(actual).includes(expected)) {
    throw new Error(`${label}: expected source not to include ${expected}`);
  }
}
