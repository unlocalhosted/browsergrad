#!/usr/bin/env node
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  parseVerifyRealWorldCudaArgs,
  verifyRealWorldCudaPlan,
} from "./verify-real-world-cuda.mjs";
import { cudaLiteCorpusExecutionFixtures } from "./cuda-lite-corpus-registry.mjs";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const defaults = parseVerifyRealWorldCudaArgs([]);
assert.equal(defaults.autoCorpusSmokeProfile, "fast");

const defaultPlan = verifyRealWorldCudaPlan(defaults);
const defaultBrowserSteps = defaultPlan.filter((step) => step.label.startsWith("real-world CUDA browser fixture e2e"));
assert.equal(defaultBrowserSteps.length, 2);
for (const step of defaultBrowserSteps) {
  assert.equal(argAfter(step.args, "--auto-corpus-smoke-profile"), "fast");
  assert.ok(step.args.includes("--forbid-skips"));
  const cases = argAfter(step.args, "--cases").split(",");
  assert.ok(cudaLiteCorpusExecutionFixtures.every((fixture) => cases.includes(fixture.caseName)));
  assert.ok(cases.includes("auto-corpus:cuda-samples:"));
  assert.ok(cases.includes("auto-corpus:cuda-120:"));
  assert.ok(cases.includes("auto-corpus:llm.c:"));
  assert.ok(cases.includes("auto-corpus:leetcuda:"));
  assert.ok(!cases.includes("example:saxpy"));
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
assert.equal(argAfter(scopedPlan[1].args, "--auto-corpus-smoke-corpora"), "cuda-samples,llm.c");

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
assert.equal(scopedNoAutoSmokePlan[1].args.includes("--auto-corpus-smoke-corpora"), false);

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

const compilerPackage = JSON.parse(readFileSync(
  path.join(repositoryRoot, "packages/browsergrad-compiler/package.json"),
  "utf8",
));
assert.equal(
  compilerPackage.scripts?.["provision:real-world-cuda"],
  "node ../../scripts/provision-real-world-cuda-corpora.mjs",
);

const workflowContracts = [
  {
    file: ".github/workflows/ci.yml",
    provision: "Provision pinned real-world CUDA corpora",
    verify: "Run complete real-world CUDA bundle gate",
  },
  {
    file: ".github/workflows/publish-npm.yml",
    provision: "Provision pinned real-world CUDA corpora",
    verify: "Run compiler real-world CUDA gate",
  },
  {
    file: ".github/workflows/release.yml",
    provision: "Provision pinned real-world CUDA corpora (compiler release)",
    verify: "Real-world CUDA gate (compiler release)",
    condition: "steps.parse.outputs.shortname == 'compiler'",
  },
];
for (const contract of workflowContracts) {
  const workflow = readFileSync(path.join(repositoryRoot, contract.file), "utf8");
  const steps = workflowSteps(workflow);
  const provisionIndex = uniqueStepIndex(steps, contract.provision, contract.file);
  const verifyIndex = uniqueStepIndex(steps, contract.verify, contract.file);
  assert.equal(
    verifyIndex,
    provisionIndex + 1,
    `${contract.file} must provision immediately before real-world verification`,
  );
  assert.ok(
    steps[provisionIndex].body.includes("provision:real-world-cuda"),
    `${contract.file} provisioning step must use the direct compiler package script`,
  );
  assert.ok(
    steps[verifyIndex].body.includes("verify:real-world-cuda")
      && steps[verifyIndex].body.includes("--skip-fetch"),
    `${contract.file} verification must consume the explicitly provisioned checkout`,
  );
  if (contract.condition !== undefined) {
    assert.equal(stepField(steps[provisionIndex], "if"), contract.condition);
    assert.equal(stepField(steps[verifyIndex], "if"), contract.condition);
  }
}

const ciWorkflow = readFileSync(path.join(repositoryRoot, ".github/workflows/ci.yml"), "utf8");
assert.ok(ciWorkflow.includes("bundle: [src, dist]"));
assert.ok(ciWorkflow.includes("--bundle=${{ matrix.bundle }}"));

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

function workflowSteps(workflow) {
  const steps = [];
  let current;
  for (const line of workflow.split(/\r?\n/u)) {
    const match = /^      - name: (.+)$/u.exec(line);
    if (match) {
      current = { name: match[1], body: line };
      steps.push(current);
    } else if (current) {
      current.body += `\n${line}`;
    }
  }
  return steps;
}

function uniqueStepIndex(steps, name, workflow) {
  const indexes = steps.flatMap((step, index) => step.name === name ? [index] : []);
  assert.equal(indexes.length, 1, `${workflow} must contain exactly one ${name} step`);
  return indexes[0];
}

function stepField(step, field) {
  return new RegExp(`^        ${field}: (.+)$`, "mu").exec(step.body)?.[1];
}
