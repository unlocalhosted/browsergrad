#!/usr/bin/env node
import assert from "node:assert/strict";
import { inferAutoCorpusWorkgroupSize } from "./cuda-lite-webgpu-fixtures.mjs";

assert.deepEqual(
  inferAutoCorpusWorkgroupSize("__global__ void __launch_bounds__(32 *8 *1) kernel() {}"),
  [256, 1, 1],
);
assert.deepEqual(
  inferAutoCorpusWorkgroupSize("__global__ void __launch_bounds__(1024) kernel() {}"),
  [32, 1, 1],
);
assert.deepEqual(
  inferAutoCorpusWorkgroupSize("__global__ void __launch_bounds__(THREADS) kernel() {}"),
  [32, 1, 1],
);
assert.deepEqual(
  inferAutoCorpusWorkgroupSize("__global__ void kernel() {}"),
  [32, 1, 1],
);

console.log("cuda-lite WebGPU fixture tests ok");
