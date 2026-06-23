#!/usr/bin/env node
import assert from "node:assert/strict";
import { createKernelCompilationUnit, kernelDefinitionName } from "./cuda-lite-source-normalizer.mjs";

const scalarKernel = `
__global__ void add(float *a, float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] + b[idx];
}`;

const vectorSibling = `
__global__ void add4(float *a, float *b, float *c, int N) {
  int idx = threadIdx.x * 4;
  float4 av = FLOAT4(a[idx]);
  c[idx] = av.x + b[idx];
}`;

{
  const source = createKernelCompilationUnit({
    kernel: scalarKernel,
    siblingKernels: [vectorSibling],
    functionDeclarations: ["#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])"],
  });
  assert.match(source, /__global__ void add\(/u);
  assert.doesNotMatch(source, /__global__ void add4\(/u);
  assert.doesNotMatch(source, /FLOAT4/u);
}

{
  const parent = `
__global__ void parent(float *out) {
  child<<<1, 1>>>(out);
}`;
  const child = `
__global__ void child(float *out) {
  out[0] = 1.0f;
}`;
  const unused = `
__global__ void unused(float *out) {
  out[0] = broken_type();
}`;
  const source = createKernelCompilationUnit({
    kernel: parent,
    siblingKernels: [child, unused],
  });
  assert.match(source, /__global__ void child\(/u);
  assert.doesNotMatch(source, /__global__ void unused\(/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(float *out) {
  out[0] = foo(1.0f);
}`,
    deviceFunctions: [
      {
        name: "foo",
        source: "__device__ float foo(float x) { return bar(x) + 1.0f; }",
      },
      {
        name: "bar",
        source: "__device__ float bar(float x) { return x * 2.0f; }",
      },
      {
        name: "baz",
        source: "__device__ float baz(float x) { return x * 4.0f; }",
      },
    ],
  });
  assert.match(source, /float foo/u);
  assert.match(source, /float bar/u);
  assert.doesNotMatch(source, /float baz/u);
}

{
  const launchBoundsKernel = `
__global__ void __launch_bounds__(WARP_SIZE * kTiles)
    bounded(float *out, const float *in, int N) {
  int idx = threadIdx.x;
  if (idx < N) out[idx] = in[idx];
}`;
  assert.equal(kernelDefinitionName(launchBoundsKernel), "bounded");
  const source = createKernelCompilationUnit({
    kernel: launchBoundsKernel,
    definesByName: new Map([
      ["WARP_SIZE", "32"],
      ["N", "999"],
    ]),
  });
  assert.match(source, /#define WARP_SIZE 32/u);
  assert.doesNotMatch(source, /#define N 999/u);
}

{
  const source = createKernelCompilationUnit({
    kernel: `
__global__ void kernel(float *in, float *out, int ld) {
  int row = threadIdx.x;
  out[0] = in[IDX2C(row, 0, ld)];
}`,
    functionDeclarations: [
      "#define IDX2C(i,j,ld) (((j)*(ld))+(i))",
      "#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])",
    ],
  });
  assert.match(source, /#define IDX2C/u);
  assert.doesNotMatch(source, /#define FLOAT4/u);
}

console.log("cuda-lite source normalizer tests ok");
