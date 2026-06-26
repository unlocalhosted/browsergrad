import assert from "node:assert/strict";
export { assert };
import {
  collectCudaLiteContextDefines,
  collectKernelTemplateArguments,
  createKernelCompilationUnit,
  kernelDefinitionName,
  pruneCudaPreprocessorBranches,
} from "./cuda-lite-source-normalizer.mjs";
export {
  collectCudaLiteContextDefines,
  collectKernelTemplateArguments,
  createKernelCompilationUnit,
  kernelDefinitionName,
  pruneCudaPreprocessorBranches,
};

export const scalarKernel = `
__global__ void add(float *a, float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] + b[idx];
}`;

export const vectorSibling = `
__global__ void add4(float *a, float *b, float *c, int N) {
  int idx = threadIdx.x * 4;
  float4 av = FLOAT4(a[idx]);
  c[idx] = av.x + b[idx];
}`;
