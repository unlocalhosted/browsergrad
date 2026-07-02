#!/usr/bin/env node
import assert from "node:assert/strict";
import path from "node:path";
import {
  parseVerifyRealWorldCudaArgs,
  verifyRealWorldCudaPlan,
} from "./verify-real-world-cuda.mjs";
import { cudaLiteCorpusExecutionFixtures } from "./cuda-lite-corpus-registry.mjs";

const defaults = parseVerifyRealWorldCudaArgs([]);
assert.equal(defaults.autoCorpusSmokeProfile, "fast");

const defaultPlan = verifyRealWorldCudaPlan(defaults);
const defaultBrowserSteps = defaultPlan.filter((step) => step.label.startsWith("real-world CUDA browser fixture e2e"));
assert.equal(defaultBrowserSteps.length, 2);
for (const step of defaultBrowserSteps) {
  assert.equal(argAfter(step.args, "--auto-corpus-smoke-profile"), "fast");
  assert.ok(step.args.includes("--forbid-skips"));
}

const full = parseVerifyRealWorldCudaArgs([
  "--skip-fetch",
  "--bundle",
  "src",
  "--auto-corpus-smoke-profile",
  "full",
  "--allow-missing-webgpu",
]);
assert.equal(full.autoCorpusSmokeProfile, "full");

const fullPlan = verifyRealWorldCudaPlan(full);
assert.equal(fullPlan.length, 2);
assert.equal(path.basename(fullPlan[0].args[0]), "audit-real-world-cuda-corpora.mjs");
assert.equal(path.basename(fullPlan[1].args[0]), "e2e-cuda-lite-webgpu.mjs");
assert.equal(argAfter(fullPlan[1].args, "--bundle"), "src");
assert.equal(argAfter(fullPlan[1].args, "--auto-corpus-smoke-profile"), "full");
assert.ok(fullPlan[1].args.includes("--summary-only"));
assert.ok(fullPlan[1].args.includes("--forbid-skips"));
assert.equal(fullPlan[1].args.includes("--require-webgpu"), false);

const scoped = parseVerifyRealWorldCudaArgs([
  "--skip-fetch",
  "--corpus",
  "cuda-samples,llm.c",
  "--forbid-skips",
  "--bundle=src",
]);
assert.deepEqual(scoped.only, ["cuda-samples", "llm.c"]);

const scopedPlan = verifyRealWorldCudaPlan(scoped);
assert.deepEqual(allArgsAfter(scopedPlan[0].args, "--only"), ["cuda-samples", "llm.c"]);
const scopedCases = argAfter(scopedPlan[1].args, "--cases").split(",");
assert.ok(scopedCases.includes("corpus:cuda-samples:vectorAdd"));
assert.ok(scopedCases.includes("corpus:llm.c:add_bias"));
assert.ok(scopedCases.includes("auto-corpus:cuda-samples:"));
assert.ok(scopedCases.includes("auto-corpus:llm.c:"));
assert.ok(scopedPlan[1].args.includes("--forbid-skips"));

const scopedNoAutoSmoke = parseVerifyRealWorldCudaArgs([
  "--corpus=cuda-samples",
  "--bundle=src",
  "--auto-corpus-smoke-limit=0",
]);
const scopedNoAutoSmokePlan = verifyRealWorldCudaPlan(scopedNoAutoSmoke);
assert.deepEqual(
  argAfter(scopedNoAutoSmokePlan[1].args, "--cases").split(","),
  cudaLiteCorpusExecutionFixtures
    .filter((fixture) => fixture.corpusId === "cuda-samples")
    .map((fixture) => fixture.caseName),
);

const leetCudaScoped = parseVerifyRealWorldCudaArgs(["--corpus=leetcuda", "--bundle=src"]);
const leetCudaCases = argAfter(verifyRealWorldCudaPlan(leetCudaScoped)[1].args, "--cases").split(",");
assert.ok(leetCudaCases.includes("corpus:LeetCUDA:elementwise_add_f32_kernel"));
assert.ok(!leetCudaCases.includes("corpus:leetcuda:"));

assert.throws(
  () => parseVerifyRealWorldCudaArgs(["--auto-corpus-smoke-profile", "wide"]),
  /--auto-corpus-smoke-profile expects full or fast/u,
);

assert.throws(
  () => parseVerifyRealWorldCudaArgs(["--only", "unknown-corpus"]),
  /unknown CUDA-lite corpus/u,
);

console.log("verify real-world CUDA CLI tests ok");

function argAfter(args, flag) {
  const index = args.indexOf(flag);
  assert.notEqual(index, -1, `${flag} missing`);
  return args[index + 1];
}

function allArgsAfter(args, flag) {
  const values = [];
  for (let index = 0; index < args.length; index++) {
    if (args[index] === flag) values.push(args[index + 1]);
  }
  return values;
}
