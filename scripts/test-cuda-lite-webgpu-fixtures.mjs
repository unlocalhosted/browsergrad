#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  applyAutoCorpusSmokeShard,
  effectiveAutoCorpusSmokeLimit,
  parseAutoCorpusSmokeProfile,
  parseAutoCorpusSmokeShard,
  parseCaseFilters,
  parseCommaSeparatedList,
  parseFlagArgs,
  filterCaseNames,
} from "./cuda-lite-webgpu-cli.mjs";
import { cudaLiteCorpusExecutionFixtures } from "./cuda-lite-corpus-registry.mjs";
import {
  autoCorpusSmokeCacheInputHash,
  autoCorpusSmokeCachePath,
  corpusExecutionFixturesForCaseFilters,
  fixtureJsLiteral,
  inferAutoCorpusWorkgroupSize,
  materializeFixtureInput,
} from "./cuda-lite-webgpu-fixtures.mjs";
import {
  failureReplayCases,
  summarizeReport,
  validateWarmMsMax,
  validateWarmSpeedup,
} from "./cuda-lite-webgpu-report.mjs";

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

assert.equal(effectiveAutoCorpusSmokeLimit(0, []), 0);
assert.equal(effectiveAutoCorpusSmokeLimit(0, ["corpus:llm.c:kernel"]), 0);
assert.equal(effectiveAutoCorpusSmokeLimit(12, ["auto-corpus:cuda-samples:file.cu:kernel:1:1"]), 12);
assert.equal(effectiveAutoCorpusSmokeLimit(0, ["auto-corpus:cuda-samples:file.cu:kernel:1:1"]), 64);
assert.deepEqual(parseAutoCorpusSmokeShard(undefined), { index: 1, count: 1 });
assert.deepEqual(parseAutoCorpusSmokeShard("3/8"), { index: 3, count: 8 });
assert.deepEqual(applyAutoCorpusSmokeShard([0, 1, 2, 3, 4, 5, 6, 7], { index: 2, count: 4 }), [1, 5]);
assert.equal(parseAutoCorpusSmokeProfile("fast"), "fast");
assert.equal(parseAutoCorpusSmokeProfile("full"), "full");
assert.deepEqual(parseCommaSeparatedList(" leetcuda, cuda-samples ,, "), ["leetcuda", "cuda-samples"]);
assert.match(autoCorpusSmokeCacheInputHash(process.cwd()), /^[0-9a-f]{16}$/u);
const cacheRoot = fs.mkdtempSync(path.join(os.tmpdir(), "browsergrad-cache-hash-"));
const cacheSrcDir = path.join(cacheRoot, "packages/browsergrad-compiler/src");
fs.mkdirSync(cacheSrcDir, { recursive: true });
fs.writeFileSync(path.join(cacheSrcDir, "index.ts"), "export const value = 1;\n");
const sourceHashA = autoCorpusSmokeCacheInputHash(cacheRoot);
fs.writeFileSync(path.join(cacheSrcDir, "index.ts"), "export const value = 2;\n");
const sourceHashB = autoCorpusSmokeCacheInputHash(cacheRoot);
assert.notEqual(sourceHashA, sourceHashB);
const cachePathA = autoCorpusSmokeCachePath("/tmp/browsergrad", {
  limit: 4,
  verifyMode: "reference",
  profile: "fast",
  corpusIds: new Set(["leetcuda"]),
  allowedRequiredFeatures: new Set(["subgroups", "shader-f16"]),
  inputHash: "abc",
});
const cachePathB = autoCorpusSmokeCachePath("/tmp/browsergrad", {
  limit: 4,
  verifyMode: "reference",
  profile: "fast",
  corpusIds: new Set(["cuda-samples"]),
  allowedRequiredFeatures: new Set(["shader-f16", "subgroups"]),
  inputHash: "def",
});
assert.match(cachePathA, /v5/u);
assert.match(cachePathA, /sig-abc/u);
assert.match(cachePathA, /corpora-leetcuda/u);
assert.match(cachePathA, /features-shader-f16-subgroups/u);
assert.notEqual(cachePathA, cachePathB);

assert.deepEqual(corpusExecutionFixturesForCaseFilters(["example:saxpy"]), []);
assert.equal(corpusExecutionFixturesForCaseFilters([]).length, cudaLiteCorpusExecutionFixtures.length);
const fixtureCaseName = cudaLiteCorpusExecutionFixtures[0]?.caseName;
assert.ok(fixtureCaseName);
assert.deepEqual(
  corpusExecutionFixturesForCaseFilters([fixtureCaseName]).map((fixture) => fixture.caseName),
  [fixtureCaseName],
);
const materializedFixtureInput = materializeFixtureInput({
  buffers: {
    out: { type: "Float32Array", length: 2 },
  },
  constants: {
    scale: 3,
    lut: { type: "Float32Array", data: [1, 2] },
  },
  deviceGlobals: {
    g_state: { type: "Uint32Array", data: [4, 5] },
  },
  textures: {
    tex: { width: 2, height: 1, channels: 1, data: { type: "Float32Array", data: [6, 7] } },
  },
  surfaces: {
    surf: { width: 2, height: 1, data: { type: "Float32Array", data: [8, 9] } },
  },
  memoryPools: {
    pool: {
      data: { type: "Uint32Array", length: 4 },
      offset: { type: "Uint32Array", data: [12] },
    },
  },
  scalars: { n: 2 },
  readback: ["out", "g_state", "pool_offset"],
});
assert.ok(materializedFixtureInput.buffers.out instanceof Float32Array);
assert.equal(materializedFixtureInput.buffers.out.length, 2);
assert.equal(materializedFixtureInput.constants.scale, 3);
assert.deepEqual([...materializedFixtureInput.constants.lut], [1, 2]);
assert.deepEqual([...materializedFixtureInput.deviceGlobals.g_state], [4, 5]);
assert.deepEqual([...materializedFixtureInput.textures.tex.data], [6, 7]);
assert.deepEqual([...materializedFixtureInput.surfaces.surf.data], [8, 9]);
assert.equal(materializedFixtureInput.memoryPools.pool.data.length, 4);
assert.deepEqual([...materializedFixtureInput.memoryPools.pool.offset], [12]);
assert.deepEqual(materializedFixtureInput.scalars, { n: 2 });
assert.deepEqual(materializedFixtureInput.readback, ["out", "g_state", "pool_offset"]);

const nonFiniteLiteral = fixtureJsLiteral({
  values: [NaN, Infinity, -Infinity, -0, undefined],
  omitted: undefined,
});
const nonFiniteRoundTrip = Function(`return (${nonFiniteLiteral});`)();
assert.equal(Number.isNaN(nonFiniteRoundTrip.values[0]), true);
assert.equal(nonFiniteRoundTrip.values[1], Infinity);
assert.equal(nonFiniteRoundTrip.values[2], -Infinity);
assert.equal(Object.is(nonFiniteRoundTrip.values[3], -0), true);
assert.equal(nonFiniteRoundTrip.values[4], null);
assert.equal(Object.hasOwn(nonFiniteRoundTrip, "omitted"), false);

assert.deepEqual(parseCaseFilters(["--case", "storage:vector-deref-lane-write"]), ["storage:vector-deref-lane-write"]);
assert.deepEqual(parseCaseFilters(["--cases=atomic:helper-rmw,storage:shared-vector-overlay"]), ["atomic:helper-rmw", "storage:shared-vector-overlay"]);
assert.deepEqual(parseCaseFilters(["--only", "prepared-resident-saxpy"]), ["prepared-resident-saxpy"]);
assert.deepEqual(parseCaseFilters(["--only=auto-corpus:cuda-samples:file.cu:kernel:1:1"]), ["auto-corpus:cuda-samples:file.cu:kernel:1:1"]);
assert.deepEqual(filterCaseNames([
  "texture-surface:volume-vector-pointer-array-minmax-active-lane-return",
  "texture-surface:volume-vector-pointer-array-minmax-active-lane-return-false-branch",
], ["texture-surface:volume-vector-pointer-array-minmax-active-lane-return"]), [
  "texture-surface:volume-vector-pointer-array-minmax-active-lane-return",
]);
assert.deepEqual(filterCaseNames([
  "texture-surface:volume-vector-pointer-array-minmax-active-lane-return",
  "texture-surface:volume-vector-pointer-array-minmax-active-lane-return-false-branch",
], ["minmax-active-lane-return"]), [
  "texture-surface:volume-vector-pointer-array-minmax-active-lane-return",
  "texture-surface:volume-vector-pointer-array-minmax-active-lane-return-false-branch",
]);
assert.deepEqual([...parseFlagArgs(["--require-webgpu", "--case-timeout-ms=15000", "--repeat", "2"])], [
  ["--require-webgpu", "true"],
  ["--case-timeout-ms", "15000"],
  ["--repeat", "2"],
]);

const summarized = summarizeReport({
  available: true,
  passed: 2,
  failed: 0,
  skipped: 0,
  caseFilters: Array.from({ length: 24 }, (_, index) => `case:${index}`),
  warmup: 1,
  warmupCases: 1,
  warmupFailed: 0,
  cases: [
    { name: "case:a", repeat: 1, stage: "compare", plan: "single-dispatch", ok: true, ms: 20 },
    { name: "case:a", repeat: 2, stage: "compare", plan: "single-dispatch", ok: true, ms: 5 },
  ],
});
assert.deepEqual(summarized.caseFilters, {
  count: 24,
  first: ["case:0", "case:1", "case:2", "case:3", "case:4", "case:5", "case:6", "case:7"],
  last: ["case:16", "case:17", "case:18", "case:19", "case:20", "case:21", "case:22", "case:23"],
});
assert.equal(summarized.warmup, 1);
assert.equal(summarized.warmupCases, 1);
assert.deepEqual(summarized.repeatStats, [{
  name: "case:a",
  coldMs: 20,
  bestWarmMs: 5,
  bestWarmRepeat: 2,
  speedup: 4,
}]);
validateWarmSpeedup({
  cases: [
    { name: "case:a", repeat: 1, ms: 20 },
    { name: "case:a", repeat: 2, ms: 5 },
  ],
}, 2);
validateWarmMsMax({
  cases: [
    { name: "case:a", repeat: 1, ms: 20 },
    { name: "case:a", repeat: 2, ms: 5 },
  ],
}, 10);
assert.throws(
  () => validateWarmSpeedup({
    cases: [
      { name: "case:a", repeat: 1, ms: 20 },
      { name: "case:a", repeat: 2, ms: 19 },
    ],
  }, 2),
  /Warm speedup gate failed/u,
);
assert.throws(
  () => validateWarmMsMax({
    cases: [
      { name: "case:a", repeat: 1, ms: 20 },
      { name: "case:a", repeat: 2, ms: 11 },
    ],
  }, 10),
  /Warm ms gate failed/u,
);
assert.deepEqual(failureReplayCases({
  cases: [
    { name: "case:a", ok: false },
    { name: "case:b", ok: true },
    { name: "case:a", ok: false },
  ],
}), ["case:a"]);

console.log("cuda-lite WebGPU fixture tests ok");
