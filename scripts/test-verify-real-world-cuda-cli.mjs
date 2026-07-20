#!/usr/bin/env node
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  executePlan,
  parseVerifyRealWorldCudaArgs,
  verifyRealWorldCudaPlan,
} from "./verify-real-world-cuda.mjs";
import { aggregateBrowserShardReports } from "./real-world-cuda-browser-shard-evidence.mjs";
import { cudaLiteCorpusExecutionFixtures } from "./cuda-lite-corpus-registry.mjs";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const defaults = parseVerifyRealWorldCudaArgs([]);
assert.equal(defaults.autoCorpusSmokeProfile, "fast");
assert.equal(defaults.browserShards, 1);

const defaultPlan = verifyRealWorldCudaPlan(defaults);
const defaultAuditSteps = defaultPlan.filter((step) =>
  step.parallelGroup === "compile-codegen-audits");
assert.equal(defaultAuditSteps.length, 4);
assert.deepEqual(
  defaultAuditSteps.map((step) => argAfter(step.args, "--only")),
  ["cuda-120", "cuda-samples", "llm.c", "leetcuda"],
);
const defaultBrowserSteps = defaultPlan.filter((step) => step.label.startsWith("real-world CUDA browser fixture e2e"));
assert.equal(defaultBrowserSteps.length, 2);
for (const step of defaultBrowserSteps) {
  assert.equal(argAfter(step.args, "--auto-corpus-smoke-profile"), "fast");
  assert.equal(argAfter(step.args, "--auto-corpus-smoke-shard"), "1/1");
  assert.equal(argAfter(step.args, "--json"), step.reportPath);
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
assert.equal(fullPlan.length, 5);
assert.equal(path.basename(fullPlan[0].args[0]), "audit-real-world-cuda-corpora.mjs");
const fullBrowserStep = fullPlan.at(-1);
assert.equal(path.basename(fullBrowserStep.args[0]), "e2e-cuda-lite-webgpu.mjs");
assert.equal(argAfter(fullBrowserStep.args, "--bundle"), "src");
assert.equal(argAfter(fullBrowserStep.args, "--auto-corpus-smoke-profile"), "full");
assert.ok(fullBrowserStep.args.includes("--summary-only"));
assert.ok(fullBrowserStep.args.includes("--forbid-skips"));
assert.equal(fullBrowserStep.args.includes("--require-webgpu"), false);

const sharded = parseVerifyRealWorldCudaArgs([
  "--bundle=src",
  "--browser-shards=2",
  "--case-timeout-ms=9000",
]);
assert.equal(sharded.browserShards, 2);
const shardedPlan = verifyRealWorldCudaPlan(sharded, { browserReportDir: "/tmp/browsergrad-plan-test" });
const shardedBrowserSteps = shardedPlan.filter((step) => step.kind === "browser-e2e");
assert.equal(shardedBrowserSteps.length, 2);
assert.deepEqual(shardedBrowserSteps.map((step) => step.shardIndex), [1, 2]);
assert.ok(shardedBrowserSteps.every((step) => step.shardCount === 2));
assert.ok(shardedBrowserSteps.every((step) => step.parallelGroup === "browser-e2e-src"));
assert.ok(shardedBrowserSteps.every((step) => step.args.includes("--require-webgpu")));
assert.ok(shardedBrowserSteps.every((step) => step.args.includes("--forbid-skips")));
assert.ok(shardedBrowserSteps.every((step) => argAfter(step.args, "--case-timeout-ms") === "9000"));
assert.deepEqual(
  shardedBrowserSteps.map((step) => argAfter(step.args, "--auto-corpus-smoke-shard")),
  ["1/2", "2/2"],
);
const plannedFixtureCases = shardedBrowserSteps.flatMap((step) => step.expectedFixtureCases);
assert.equal(new Set(plannedFixtureCases).size, cudaLiteCorpusExecutionFixtures.length);
assert.deepEqual(new Set(plannedFixtureCases), new Set(cudaLiteCorpusExecutionFixtures.map((fixture) => fixture.caseName)));
assert.deepEqual(
  shardedBrowserSteps[0].expectedFixtureCases,
  cudaLiteCorpusExecutionFixtures.filter((_, index) => index % 2 === 0).map((fixture) => fixture.caseName),
);
assert.deepEqual(
  shardedBrowserSteps[1].expectedFixtureCases,
  cudaLiteCorpusExecutionFixtures.filter((_, index) => index % 2 === 1).map((fixture) => fixture.caseName),
);
for (const step of shardedBrowserSteps) {
  const cases = argAfter(step.args, "--cases").split(",");
  assert.ok(step.expectedFixtureCases.every((caseName) => cases.includes(caseName)));
  assert.ok(["cuda-120", "cuda-samples", "llm.c", "leetcuda"]
    .every((id) => cases.includes(`auto-corpus:${id}:`)));
}

assert.throws(
  () => parseVerifyRealWorldCudaArgs(["--browser-shards=0"]),
  /--browser-shards expects a positive integer/u,
);
assert.throws(
  () => parseVerifyRealWorldCudaArgs(["--browser-shards=9"]),
  /--browser-shards must not exceed 8/u,
);

const scoped = parseVerifyRealWorldCudaArgs([
  "--skip-fetch",
  "--corpus",
  "cuda-samples,llm.c",
  "--forbid-skips",
  "--bundle=src",
]);
assert.deepEqual(scoped.only, ["cuda-samples", "llm.c"]);

const scopedPlan = verifyRealWorldCudaPlan(scoped);
const scopedAuditSteps = scopedPlan.filter((step) =>
  step.parallelGroup === "compile-codegen-audits");
assert.deepEqual(
  scopedAuditSteps.map((step) => argAfter(step.args, "--only")),
  ["cuda-samples", "llm.c"],
);
const scopedBrowserStep = scopedPlan.at(-1);
const scopedCases = argAfter(scopedBrowserStep.args, "--cases").split(",");
assert.ok(scopedCases.includes("corpus:cuda-samples:vectorAdd"));
assert.ok(scopedCases.includes("corpus:llm.c:add_bias"));
assert.ok(scopedCases.includes("auto-corpus:cuda-samples:"));
assert.ok(scopedCases.includes("auto-corpus:llm.c:"));
assert.ok(scopedBrowserStep.args.includes("--forbid-skips"));
assert.equal(argAfter(scopedBrowserStep.args, "--auto-corpus-smoke-corpora"), "cuda-samples,llm.c");

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
assert.equal(scopedNoAutoSmokePlan.at(-1).args.includes("--auto-corpus-smoke-corpora"), false);

const leetCudaScoped = parseVerifyRealWorldCudaArgs(["--corpus=leetcuda", "--bundle=src"]);
const leetCudaCases = argAfter(
  verifyRealWorldCudaPlan(leetCudaScoped).at(-1).args,
  "--cases",
).split(",");
assert.ok(leetCudaCases.includes("corpus:LeetCUDA:elementwise_add_f32_kernel"));
assert.ok(!leetCudaCases.includes("corpus:leetcuda:"));

assert.throws(
  () => parseVerifyRealWorldCudaArgs(["--auto-corpus-smoke-profile", "wide"]),
  /--auto-corpus-smoke-profile expects full or fast/u,
);

const timed = parseVerifyRealWorldCudaArgs(["--timing-json", ".tmp/real-world-timing.json"]);
assert.equal(timed.timingJson, ".tmp/real-world-timing.json");
assert.throws(
  () => parseVerifyRealWorldCudaArgs(["--timing-json", "--bundle=src"]),
  /--timing-json expects a path/u,
);

assert.throws(
  () => parseVerifyRealWorldCudaArgs(["--only", "unknown-corpus"]),
  /unknown CUDA-lite corpus/u,
);

const fakeShardReports = new Map(shardedBrowserSteps.map((step) => {
  const fixtureCases = step.expectedFixtureCases.map((name) => ({
    name,
    ok: true,
    expectedOutputPinned: step.expectedOutputFixtureCases.includes(name),
  }));
  const autoCorpusCase = {
    name: `auto-corpus:cuda-120:fixture-${step.shardIndex}`,
    ok: true,
  };
  return [step.reportPath, {
    bundle: step.bundle,
    available: true,
    cases: [...fixtureCases, autoCorpusCase],
    passed: fixtureCases.length + 1,
    failed: 0,
    skipped: 0,
    warmupFailed: 0,
    autoCorpusSmokeCovered: 1,
    autoCorpusSmokeExpectedCovered: 1,
    autoCorpusSmokeShard: { index: step.shardIndex, count: step.shardCount },
  }];
}));
const aggregateEvidence = aggregateBrowserShardReports(
  shardedPlan,
  (reportPath) => structuredClone(fakeShardReports.get(reportPath)),
);
assert.equal(aggregateEvidence.ok, true);
assert.equal(aggregateEvidence.bundles.length, 1);
assert.equal(aggregateEvidence.bundles[0].expectedFixtureCases, cudaLiteCorpusExecutionFixtures.length);
assert.equal(aggregateEvidence.bundles[0].observedFixtureCases, cudaLiteCorpusExecutionFixtures.length);
assert.equal(aggregateEvidence.bundles[0].expectedAutoCorpusSmokeCases, 2);
assert.equal(aggregateEvidence.bundles[0].observedAutoCorpusSmokeCases, 2);
assert.deepEqual(aggregateEvidence.bundles[0].failures, []);

const missingFixtureReports = new Map(
  [...fakeShardReports].map(([reportPath, report], index) => [
    reportPath,
    index === 0 ? { ...structuredClone(report), cases: report.cases.slice(1) } : structuredClone(report),
  ]),
);
const missingFixtureEvidence = aggregateBrowserShardReports(
  shardedPlan,
  (reportPath) => missingFixtureReports.get(reportPath),
);
assert.equal(missingFixtureEvidence.ok, false);
assert.equal(missingFixtureEvidence.bundles[0].missingFixtureCases.length, 1);

const unexpectedOutputReports = new Map(
  [...fakeShardReports].map(([reportPath, report], index) => {
    const cloned = structuredClone(report);
    if (index === 0) {
      cloned.cases.push({
        name: "corpus:unexpected:output-threshold",
        ok: true,
        expectedOutputPinned: true,
      });
      cloned.passed += 1;
    }
    return [reportPath, cloned];
  }),
);
const unexpectedOutputEvidence = aggregateBrowserShardReports(
  shardedPlan,
  (reportPath) => unexpectedOutputReports.get(reportPath),
);
assert.equal(unexpectedOutputEvidence.ok, false);
assert.deepEqual(
  unexpectedOutputEvidence.bundles[0].unexpectedOutputPinnedFixtureCases,
  ["corpus:unexpected:output-threshold"],
);

const duplicateAutoReports = new Map(
  [...fakeShardReports].map(([reportPath, report]) => {
    const cloned = structuredClone(report);
    cloned.cases.at(-1).name = "auto-corpus:cuda-120:duplicate";
    return [reportPath, cloned];
  }),
);
const duplicateAutoEvidence = aggregateBrowserShardReports(
  shardedPlan,
  (reportPath) => duplicateAutoReports.get(reportPath),
);
assert.equal(duplicateAutoEvidence.ok, false);
assert.deepEqual(duplicateAutoEvidence.bundles[0].duplicateAutoCorpusCases, ["auto-corpus:cuda-120:duplicate"]);

const unavailableAllowedEvidence = aggregateBrowserShardReports(fullPlan, () => ({
  bundle: "src",
  available: false,
  reason: "navigator.gpu undefined",
  cases: [],
}));
assert.equal(unavailableAllowedEvidence.ok, true);
assert.equal(unavailableAllowedEvidence.bundles[0].coverageEvaluated, false);

const schedulerEvents = [];
let activeParallelSteps = 0;
let maxActiveParallelSteps = 0;
let completedParallelSteps = 0;
const schedulerTimings = await executePlan([
  { label: "audit-a", parallelGroup: "audits", args: [] },
  { label: "audit-b", parallelGroup: "audits", args: [] },
  { label: "browser", args: [] },
], async (step, captureOutput) => {
  schedulerEvents.push({ event: "start", label: step.label, captureOutput });
  if (step.parallelGroup !== undefined) {
    activeParallelSteps += 1;
    maxActiveParallelSteps = Math.max(maxActiveParallelSteps, activeParallelSteps);
    await new Promise((resolve) => setImmediate(resolve));
    activeParallelSteps -= 1;
    completedParallelSteps += 1;
  } else {
    assert.equal(activeParallelSteps, 0);
    assert.equal(completedParallelSteps, 2);
  }
  schedulerEvents.push({ event: "end", label: step.label, captureOutput });
  return { label: step.label, ok: true, exitCode: 0 };
});
assert.equal(maxActiveParallelSteps, 2);
assert.deepEqual(schedulerTimings.map((timing) => timing.label), ["audit-a", "audit-b", "browser"]);
assert.deepEqual(
  schedulerEvents.filter(({ event }) => event === "start").map(({ label, captureOutput }) => [label, captureOutput]),
  [["audit-a", true], ["audit-b", true], ["browser", false]],
);

const failedSchedulerSteps = [];
const failedSchedulerTimings = await executePlan([
  { label: "audit-a", parallelGroup: "audits", args: [] },
  { label: "audit-b", parallelGroup: "audits", args: [] },
  { label: "browser", args: [] },
], async (step) => {
  failedSchedulerSteps.push(step.label);
  return { label: step.label, ok: step.label !== "audit-a", exitCode: step.label === "audit-a" ? 1 : 0 };
});
assert.deepEqual(failedSchedulerSteps, ["audit-a", "audit-b"]);
assert.equal(failedSchedulerTimings.length, 2);
assert.equal(failedSchedulerTimings.some((timing) => !timing.ok), true);

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
assert.ok(ciWorkflow.includes("--browser-shards=2"));
assert.ok(ciWorkflow.includes("--timing-json=.tmp/real-world-cuda-timing-${{ matrix.bundle }}.json"));
assert.ok(ciWorkflow.includes("Upload real-world CUDA timing attribution"));

console.log("verify real-world CUDA CLI tests ok");

function argAfter(args, flag) {
  const index = args.indexOf(flag);
  assert.notEqual(index, -1, `${flag} missing`);
  return args[index + 1];
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
