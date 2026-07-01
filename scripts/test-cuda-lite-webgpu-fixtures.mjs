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
  parseFlagArgs,
} from "./cuda-lite-webgpu-cli.mjs";
import { cudaLiteCorpusExecutionFixtures } from "./cuda-lite-corpus-registry.mjs";
import {
  autoCorpusSmokeCacheInputHash,
  autoCorpusSmokeCachePath,
  corpusExecutionFixturesForCaseFilters,
  inferAutoCorpusWorkgroupSize,
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
  allowedRequiredFeatures: new Set(["subgroups", "shader-f16"]),
  inputHash: "abc",
});
const cachePathB = autoCorpusSmokeCachePath("/tmp/browsergrad", {
  limit: 4,
  verifyMode: "reference",
  profile: "fast",
  allowedRequiredFeatures: new Set(["shader-f16", "subgroups"]),
  inputHash: "def",
});
assert.match(cachePathA, /v4/u);
assert.match(cachePathA, /sig-abc/u);
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

assert.deepEqual(parseCaseFilters(["--case", "storage:vector-deref-lane-write"]), ["storage:vector-deref-lane-write"]);
assert.deepEqual(parseCaseFilters(["--cases=atomic:helper-rmw,storage:shared-vector-overlay"]), ["atomic:helper-rmw", "storage:shared-vector-overlay"]);
assert.deepEqual(parseCaseFilters(["--only", "prepared-resident-saxpy"]), ["prepared-resident-saxpy"]);
assert.deepEqual(parseCaseFilters(["--only=auto-corpus:cuda-samples:file.cu:kernel:1:1"]), ["auto-corpus:cuda-samples:file.cu:kernel:1:1"]);
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
  warmup: 1,
  warmupCases: 1,
  warmupFailed: 0,
  cases: [
    { name: "case:a", repeat: 1, stage: "compare", plan: "single-dispatch", ok: true, ms: 20 },
    { name: "case:a", repeat: 2, stage: "compare", plan: "single-dispatch", ok: true, ms: 5 },
  ],
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
