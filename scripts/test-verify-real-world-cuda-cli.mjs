#!/usr/bin/env node
import assert from "node:assert/strict";
import path from "node:path";
import {
  parseVerifyRealWorldCudaArgs,
  verifyRealWorldCudaPlan,
} from "./verify-real-world-cuda.mjs";

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

assert.throws(
  () => parseVerifyRealWorldCudaArgs(["--auto-corpus-smoke-profile", "wide"]),
  /--auto-corpus-smoke-profile expects full or fast/u,
);

console.log("verify real-world CUDA CLI tests ok");

function argAfter(args, flag) {
  const index = args.indexOf(flag);
  assert.notEqual(index, -1, `${flag} missing`);
  return args[index + 1];
}
