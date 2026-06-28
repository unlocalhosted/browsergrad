#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { isWgslFloat16Array } from "../packages/browsergrad-kernels/dist/index.js";
import { syntheticInputForCompiled } from "./cuda-lite-synthetic-input.mjs";

const root = findRepoRoot(process.cwd());
const compiler = await import(pathToFileURL(path.join(root, "packages/browsergrad-compiler/dist/index.js")).href);

const nativeHalf = compiler.compileCudaLiteKernelForWebGpu(`
__global__ void half_kernel(half *x, half2 *y) {
  if (threadIdx.x < 1) {
    x[0] = x[0];
    y[0] = y[0];
  }
}`, {
  features: { "shader-f16": true },
  workgroupSize: [1, 1, 1],
});
const nativeInput = syntheticInputForCompiled(nativeHalf);
assert.equal(isWgslFloat16Array(nativeInput.buffers.x), true, "native half storage should use Float16Array");
assert.equal(isWgslFloat16Array(nativeInput.buffers.y), true, "native half2 storage should use Float16Array lanes");
compiler.runCompiledKernelReference(nativeHalf, nativeInput, { gridDim: [1, 1, 1], blockDim: [1, 1, 1] });

const f32Half = compiler.compileCudaLiteKernelForWebGpu(`
__global__ void half_kernel(half *x, half2 *y) {
  if (threadIdx.x < 1) {
    x[0] = x[0];
    y[0] = y[0];
  }
}`, {
  f16Mode: "f32",
  workgroupSize: [1, 1, 1],
});
const f32Input = syntheticInputForCompiled(f32Half);
assert.equal(f32Input.buffers.x instanceof Float32Array, true, "f32 half compatibility storage should use Float32Array");
assert.equal(f32Input.buffers.y instanceof Float32Array, true, "f32 half2 compatibility storage should use Float32Array lanes");

const vectorInputKernel = compiler.compileCudaLiteKernelForWebGpu(`
__global__ void vector_input(uint4 *u, int4 *i, float4 *f) {
  if (threadIdx.x < 1) {
    u[0] = u[0];
    i[0] = i[0];
    f[0] = f[0];
  }
}`, {
  workgroupSize: [1, 1, 1],
});
const vectorInput = syntheticInputForCompiled(vectorInputKernel);
assert.equal(vectorInput.buffers.u instanceof Uint32Array, true, "uint vector storage should use Uint32Array lanes");
assert.equal(vectorInput.buffers.i instanceof Int32Array, true, "int vector storage should use Int32Array lanes");
assert.equal(vectorInput.buffers.f instanceof Float32Array, true, "float vector storage should use Float32Array lanes");

const snakeCaseSizingKernel = compiler.compileCudaLiteKernelForWebGpu(`
__global__ void snake_case_sizing(float *out, int block_size, int threads_per_block) {
  if (threadIdx.x == 0) {
    out[0] = (float)(block_size + threads_per_block);
  }
}`, {
  workgroupSize: [1, 1, 1],
});
const snakeCaseSizingInput = syntheticInputForCompiled(snakeCaseSizingKernel);
assert.equal(snakeCaseSizingInput.scalars.block_size, 256, "snake_case block_size should use block sizing default");
assert.equal(snakeCaseSizingInput.scalars.threads_per_block, 256, "snake_case threads_per_block should use block sizing default");

console.log("cuda-lite synthetic input tests passed");

function findRepoRoot(start) {
  let dir = path.resolve(start);
  while (dir !== path.dirname(dir)) {
    if (fs.existsSync(path.join(dir, "pnpm-workspace.yaml"))) return dir;
    dir = path.dirname(dir);
  }
  throw new Error("could not find repo root");
}
