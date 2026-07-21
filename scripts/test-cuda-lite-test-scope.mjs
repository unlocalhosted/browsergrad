#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  commandToString,
  corpusFixtureCasesForFiles,
  fixtureCaseNamesForChangedLines,
  planForChangedFiles,
} from "./cuda-lite-test-scope.mjs";
import {
  atomicScopeCases,
  controlScopeCases,
  runtimeScopeCases,
  storageScopeCases,
  textureScopeCases,
  webGpuSlowSmokeHotCases,
  webGpuSmokeCases,
} from "./cuda-lite-webgpu-smoke-cases.mjs";
import { buildWebGpuSmokeArgs } from "./run-cuda-lite-webgpu-smoke.mjs";
import { buildWebGpuSlowSmokeHotArgs } from "./run-cuda-lite-webgpu-slow-smoke-hot.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const storagePlan = planForChangedFiles([
  "packages/browsergrad-compiler/src/semantic_wgsl_pointers.ts",
]);
assert.deepEqual(storagePlan.scopes.map((scope) => scope.id), [
  "compiler-types",
  "wgsl-storage-pointer",
]);
assert.ok(
  storagePlan.commands.map(commandToString).includes("pnpm --filter @unlocalhosted/browsergrad-compiler run test:wgsl-modules"),
);
assert.ok(
  storagePlan.commands.map(commandToString).some((cmd) => cmd.includes("storage:vector-deref-lane-write")),
);
assert.ok(
  storagePlan.commands.map(commandToString).some((cmd) => cmd.includes("storage:local-pointer-row-alias")),
);
assert.ok(
  storagePlan.commands.map(commandToString).some((cmd) => cmd.includes("helpers:vector-cache-hint-dynamic-read")),
);
assert.ok(
  storagePlan.commands.map(commandToString).some((cmd) => cmd.includes("storage:shared-vector-helper")),
);
assert.ok(
  storagePlan.commands.map(commandToString).some((cmd) => cmd.includes("device-function:pointer-param-helpers")),
);
assert.ok(
  storagePlan.commands.map(commandToString).some((cmd) => cmd.includes("device-function:device-global-helper-rmw")),
);
assert.ok(
  storagePlan.commands.map(commandToString).some((cmd) => cmd.includes("storage:guarded-shared-vector-lanes")),
);
assert.ok(
  storagePlan.commands.map(commandToString).some((cmd) => cmd.includes("helpers:vector-lane-pointer-offset-helper")),
);
assert.ok(
  storagePlan.commands.map(commandToString).some((cmd) => cmd.includes("helpers:vector-to-scalar-pointer-offset-helper")),
);
assert.ok(
  storagePlan.commands.map(commandToString).some((cmd) => cmd.includes("helpers:vector-to-scalar-pointer-write-offset-helper")),
);

const sharedWgslPlan = planForChangedFiles([
  "packages/browsergrad-compiler/src/wgsl_support_helpers.ts",
]);
assert.deepEqual(sharedWgslPlan.scopes.map((scope) => scope.id), [
  "compiler-types",
  "wgsl-shared-emitter",
]);
assert.deepEqual(sharedWgslPlan.commands.map(commandToString), [
  "pnpm --filter @unlocalhosted/browsergrad-compiler run typecheck",
  "pnpm --filter @unlocalhosted/browsergrad-compiler run test:wgsl-modules",
  "pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:smoke",
]);

const rootWgslPlan = planForChangedFiles([
  "packages/browsergrad-compiler/src/semantic_wgsl.ts",
]);
assert.deepEqual(rootWgslPlan.scopes.map((scope) => scope.id), [
  "compiler-types",
  "compiler-unit",
  "wgsl-shared-emitter",
  "wgsl-storage-pointer",
  "wgsl-atomics",
  "wgsl-control-cooperative",
  "wgsl-texture-surface",
]);
assert.ok(rootWgslPlan.commands.map(commandToString).includes("pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:smoke"));

const atomicPlan = planForChangedFiles([
  "packages/browsergrad-compiler/src/semantic_wgsl_atomic_analysis.ts",
]);
assert.deepEqual(atomicPlan.scopes.map((scope) => scope.id), [
  "compiler-types",
  "wgsl-atomics",
]);
assert.ok(
  atomicPlan.commands.map(commandToString).includes("pnpm --filter @unlocalhosted/browsergrad-compiler run test:wgsl-modules"),
);
assert.ok(
  atomicPlan.commands.map(commandToString).some((cmd) => cmd.includes("atomic:helper-rmw")),
);
assert.ok(
  atomicPlan.commands.map(commandToString).some((cmd) => cmd.includes("atomic:alias-inc-dec")),
);
assert.ok(
  atomicPlan.commands.map(commandToString).some((cmd) => cmd.includes("atomic:system-float")),
);
assert.ok(
  atomicPlan.commands.map(commandToString).some((cmd) => cmd.includes("atomic:helper-exchange-cas")),
);

const scopedAtomicPlan = planForChangedFiles([
  "docs/internal/status-detailed.md",
], { scopes: ["atomic"] });
assert.deepEqual(scopedAtomicPlan.scopes.map((scope) => scope.id), [
  "compiler-types",
  "wgsl-atomics",
]);
assert.deepEqual(scopedAtomicPlan.scopes.map((scope) => scope.files), [[], []]);
assert.deepEqual(scopedAtomicPlan.scopes.map((scope) => scope.requested), [true, true]);
assert.ok(scopedAtomicPlan.commands.map(commandToString).some((cmd) => cmd.includes("atomic:helper-rmw")));
assert.equal(scopedAtomicPlan.commands.map(commandToString).some((cmd) => cmd.includes("corpus:")), false);

const scopedTexturePlan = planForChangedFiles([], { scopes: ["texture"] });
assert.deepEqual(scopedTexturePlan.scopes.map((scope) => scope.id), [
  "compiler-types",
  "wgsl-texture-surface",
]);
assert.deepEqual(scopedTexturePlan.scopes.map((scope) => scope.files), [[], []]);
assert.deepEqual(scopedTexturePlan.scopes.map((scope) => scope.requested), [true, true]);
assert.ok(scopedTexturePlan.commands.map(commandToString).some((cmd) => cmd.includes("texture-surface:roundtrip")));
assert.ok(scopedTexturePlan.commands.map(commandToString).some((cmd) => cmd.includes("surface:driver-alias")));
assert.ok(scopedTexturePlan.commands.map(commandToString).some((cmd) => cmd.includes("texture:atlas-helpers")));

const runtimePlan = planForChangedFiles([
  "packages/browsergrad-compiler/src/dynamic_launch.ts",
]);
assert.deepEqual(runtimePlan.scopes.map((scope) => scope.id), [
  "compiler-types",
  "runtime-orchestration",
]);
assert.ok(runtimePlan.commands.map(commandToString).some((cmd) => cmd.includes("runtime:host-dynamic-pointer-offset")));
assert.ok(runtimePlan.commands.map(commandToString).some((cmd) => cmd.includes("runtime:host-dynamic-ordered-launches")));
assert.ok(runtimePlan.commands.map(commandToString).some((cmd) => cmd.includes("runtime:host-dynamic-child-peer-copy")));
assert.ok(runtimePlan.commands.map(commandToString).some((cmd) => cmd.includes("runtime:host-dynamic-system-atomics")));
assert.ok(runtimePlan.commands.map(commandToString).some((cmd) => cmd.includes("runtime:host-dynamic-conditional-alias-atomic")));
assert.ok(runtimePlan.commands.map(commandToString).some((cmd) => cmd.includes("runtime:recursive-host-dynamic-launch")));
assert.ok(runtimePlan.commands.map(commandToString).some((cmd) => cmd.includes("runtime:pool-alias-host-dynamic-launch")));
assert.ok(runtimePlan.commands.map(commandToString).some((cmd) => cmd.includes("runtime:pool-pointer-host-dynamic-launch")));
assert.ok(runtimePlan.commands.map(commandToString).some((cmd) => cmd.includes("runtime:expanded-pool-pointer-host-dynamic-launch")));
assert.ok(runtimePlan.commands.map(commandToString).some((cmd) => cmd.includes("runtime:launched-device-function-pool-dynamic-launch")));

const scopedRuntimePlan = planForChangedFiles([], { scopes: ["runtime"] });
assert.deepEqual(scopedRuntimePlan.scopes.map((scope) => scope.id), [
  "compiler-types",
  "compiler-unit",
  "runtime-orchestration",
]);
assert.ok(scopedRuntimePlan.commands.map(commandToString).some((cmd) => cmd.includes("runtime:host-copy")));

const controlPlan = planForChangedFiles([
  "packages/browsergrad-compiler/src/wgsl_control_analysis.ts",
]);
assert.deepEqual(controlPlan.scopes.map((scope) => scope.id), [
  "compiler-types",
  "wgsl-control-cooperative",
]);
assert.ok(
  controlPlan.commands.map(commandToString).includes("pnpm --filter @unlocalhosted/browsergrad-compiler run test:wgsl-modules"),
);
assert.ok(
  controlPlan.commands.map(commandToString).some((cmd) => cmd.includes("control:active-lane-guarded-rhs")),
);
assert.ok(
  controlPlan.commands.map(commandToString).some((cmd) => cmd.includes("cooperative:tile-vote-mask")),
);

const scriptPlan = planForChangedFiles([
  "scripts/cuda-lite-webgpu-fixtures.mjs",
]);
assert.deepEqual(scriptPlan.scopes.map((scope) => scope.id), [
  "webgpu-fixture-scripts",
]);
assert.deepEqual(scriptPlan.commands.map(commandToString), [
  "pnpm --filter @unlocalhosted/browsergrad-compiler run test:webgpu-fixtures",
  "pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:smoke",
]);

const kernelWgslPlan = planForChangedFiles([
  "packages/browsergrad-kernels/src/wgsl_program.ts",
]);
assert.deepEqual(kernelWgslPlan.scopes.map((scope) => scope.id), [
  "kernels-wgsl-bridge",
]);
assert.deepEqual(kernelWgslPlan.commands.map(commandToString), [
  "pnpm --filter @unlocalhosted/browsergrad-kernels run typecheck",
  "pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:smoke",
]);

const smokePlan = planForChangedFiles([
  "scripts/cuda-lite-webgpu-smoke-cases.mjs",
]);
assert.deepEqual(smokePlan.scopes.map((scope) => scope.id), [
  "webgpu-smoke-scripts",
]);
assert.deepEqual(smokePlan.commands.map(commandToString), [
  "pnpm --filter @unlocalhosted/browsergrad-compiler run test:test-scope",
  "pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:smoke",
  "pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:slow-smoke-hot",
]);

const slowSmokeRunnerPlan = planForChangedFiles([
  "scripts/run-cuda-lite-webgpu-slow-smoke-hot.mjs",
]);
assert.deepEqual(slowSmokeRunnerPlan.scopes.map((scope) => scope.id), [
  "webgpu-smoke-scripts",
]);

const scopedSmokePlan = planForChangedFiles([], { scopes: ["webgpu-smoke"] });
assert.deepEqual(scopedSmokePlan.scopes.map((scope) => scope.id), [
  "webgpu-smoke-scripts",
]);
assert.deepEqual(scopedSmokePlan.scopes.map((scope) => scope.requested), [true]);

const smokeArgs = buildWebGpuSmokeArgs(root, [], {});
assert.equal(smokeArgs.includes("--profile-case"), true);
assert.equal(smokeArgs.filter((arg) => arg === "--case-timeout-ms").length, 1);
assert.equal(smokeArgs.filter((arg) => arg === "--device-recycle-interval").length, 1);
assert.ok(smokeArgs.includes(webGpuSmokeCases.join(",")));
assert.ok(smokeArgs.includes(webGpuSlowSmokeHotCases.join(",")));
assert.ok(buildWebGpuSmokeArgs(root, [], { CUDA_LITE_WEBGPU_SMOKE_DEVICE_RECYCLE_INTERVAL: "16" }).includes("16"));
assert.equal(buildWebGpuSmokeArgs(root, [], { CUDA_LITE_WEBGPU_SMOKE_PROFILE: "none" }).includes("--profile-case"), false);

const profiledSmokeArgs = buildWebGpuSmokeArgs(root, [], { CUDA_LITE_WEBGPU_SMOKE_PROFILE: "all" });
assert.ok(profiledSmokeArgs.includes("--profile-case"));
assert.ok(profiledSmokeArgs.includes("all"));
assert.ok(buildWebGpuSmokeArgs(root, [], { CUDA_LITE_WEBGPU_SMOKE_PROFILE: "true" }).includes("all"));

const slowSmokeHotArgs = buildWebGpuSlowSmokeHotArgs(root);
const slowSmokeHotCasesArg = slowSmokeHotArgs[slowSmokeHotArgs.indexOf("--cases") + 1];
const slowSmokeHotProfileArg = slowSmokeHotArgs[slowSmokeHotArgs.indexOf("--profile-case") + 1];
assert.equal(slowSmokeHotCasesArg, webGpuSlowSmokeHotCases.join(","));
assert.equal(slowSmokeHotProfileArg, slowSmokeHotCasesArg);
assert.ok(slowSmokeHotArgs.includes("--forbid-skips"));
assert.ok(slowSmokeHotArgs.includes("--expect-warm-ms-max"));
assert.match(slowSmokeHotCasesArg, /texture:cubemap-vector-pointer-array-compound/u);
assert.match(slowSmokeHotCasesArg, /texture:cubemap-helper-multi-object-pointer-array-cas-minmax-guarded-rhs/u);
assert.match(slowSmokeHotCasesArg, /surface:surf3d-pointer-alias-atomic-pointer-array-compound-active-lane-return-false-branch/u);

const lastFailurePlan = planForChangedFiles([
  "scripts/run-cuda-lite-last-failures.mjs",
]);
assert.deepEqual(lastFailurePlan.scopes.map((scope) => scope.id), [
  "tool-runner",
]);
assert.deepEqual(lastFailurePlan.commands.map(commandToString), [
  "pnpm --filter @unlocalhosted/browsergrad-compiler run test:tool-lock",
]);

const bugbashStatusPlan = planForChangedFiles([
  "docs/internal/compiler-bugbash-progress.md",
]);
assert.deepEqual(bugbashStatusPlan.scopes.map((scope) => scope.id), [
  "bugbash-status",
]);
assert.deepEqual(bugbashStatusPlan.commands.map(commandToString), [
  "pnpm --filter @unlocalhosted/browsergrad-compiler run test:bugbash-status",
  "pnpm --filter @unlocalhosted/browsergrad-compiler run bugbash:status",
]);

const benchmarkPlan = planForChangedFiles([
  "scripts/benchmark-cuda-lite-webgpu.mjs",
]);
assert.deepEqual(benchmarkPlan.scopes.map((scope) => scope.id), [
  "benchmark-scripts",
]);
assert.deepEqual(benchmarkPlan.commands.map(commandToString), [
  "pnpm --filter @unlocalhosted/browsergrad-compiler run bench -- --skip-build --runs 1 --warmup 0",
  "pnpm --filter @unlocalhosted/browsergrad-compiler run bench:browser -- --skip-build --require-webgpu --runs 1 --warmup 0 --length 256",
]);

const corpusFixturePlan = planForChangedFiles([
  "scripts/cuda-lite-corpus-fixtures-extra-2.mjs",
], { root });
assert.deepEqual(corpusFixturePlan.scopes.map((scope) => scope.id), [
  "corpus-execution-fixture-files",
]);
const corpusFixtureCommands = corpusFixturePlan.commands.map(commandToString);
assert.equal(corpusFixtureCommands[0], "pnpm --filter @unlocalhosted/browsergrad-compiler run test:webgpu-fixtures");
assert.equal(corpusFixtureCommands.some((cmd) => cmd === "pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:fixtures"), false);
assert.ok(corpusFixtureCommands.some((cmd) => cmd.includes("corpus:llm.c:softmax_forward_kernel2")));

const registryPlan = planForChangedFiles([
  "scripts/cuda-lite-corpus-registry.mjs",
], { root });
assert.deepEqual(registryPlan.scopes.map((scope) => scope.id), [
  "corpus-execution-fixture-registry",
]);
assert.deepEqual(registryPlan.commands.map(commandToString), [
  "pnpm --filter @unlocalhosted/browsergrad-compiler run test:webgpu-fixtures",
  "pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:fixtures",
]);

assert.ok(corpusFixtureCasesForFiles([
  "scripts/cuda-lite-corpus-fixtures-extra-2.mjs",
], root).includes("corpus:llm.c:softmax_forward_kernel2"));

assert.deepEqual(
  fixtureCaseNamesForChangedLines(`
export const fixtures = [
  {
    caseName: "first",
    input: () => ({
      value: 1,
    }),
  },
  {
    caseName: "second",
    input: () => ({
      value: 2,
    }),
  },
];
`, [5, 12]),
  ["first", "second"],
);

const realWorldVerifierPlan = planForChangedFiles([
  "scripts/verify-real-world-cuda.mjs",
]);
assert.deepEqual(realWorldVerifierPlan.scopes.map((scope) => scope.id), [
  "real-world-verifier-scripts",
]);
assert.deepEqual(realWorldVerifierPlan.commands.map(commandToString), [
  "pnpm --filter @unlocalhosted/browsergrad-compiler run test:verify-real-world-cli",
  "pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:fast",
]);

const corpusProvisioningPlan = planForChangedFiles([
  "scripts/cuda-lite-corpus-provisioning.mjs",
  "scripts/cuda-lite-corpus-residue-reclaim.py",
  "scripts/test-cuda-lite-corpus-provisioning.mjs",
]);
assert.deepEqual(corpusProvisioningPlan.scopes.map((scope) => scope.id), [
  "corpus-provisioning-scripts",
]);
assert.deepEqual(corpusProvisioningPlan.scopes[0].files, [
  "scripts/cuda-lite-corpus-provisioning.mjs",
  "scripts/cuda-lite-corpus-residue-reclaim.py",
  "scripts/test-cuda-lite-corpus-provisioning.mjs",
]);
assert.deepEqual(corpusProvisioningPlan.commands.map(commandToString), [
  "pnpm --filter @unlocalhosted/browsergrad-compiler run test:corpus-provisioning:run",
]);

const multiWgslPlan = planForChangedFiles([
  "packages/browsergrad-compiler/src/semantic_wgsl_pointers.ts",
  "packages/browsergrad-compiler/src/semantic_wgsl_atomic_analysis.ts",
  "packages/browsergrad-compiler/src/wgsl_control_analysis.ts",
  "packages/browsergrad-compiler/src/wgsl_texture_surface.ts",
]);
const multiWgslCommands = multiWgslPlan.commands.map(commandToString);
const multiWgslE2e = multiWgslCommands.filter((cmd) => cmd.includes("run e2e:webgpu:case"));
assert.equal(multiWgslE2e.length, 1);
assert.ok(multiWgslE2e[0].includes("storage:vector-deref-lane-write"));
assert.ok(multiWgslE2e[0].includes("atomic:helper-rmw"));
assert.ok(multiWgslE2e[0].includes("control:active-lane-loop-barrier"));
assert.ok(multiWgslE2e[0].includes("texture:uchar4-read"));
assert.ok(multiWgslE2e[0].includes("texture-surface:roundtrip"));
assert.ok(multiWgslCommands.some((cmd) => cmd.includes("run e2e:webgpu:hot-case:gate") && cmd.includes("texture-surface:roundtrip")));
assert.ok(multiWgslCommands.some((cmd) => cmd.includes("run e2e:webgpu:hot-case:gate") && cmd.includes("--expect-warm-ms-max 12")));

const fastPlan = planForChangedFiles([
  "packages/browsergrad-compiler/src/analyzer.ts",
], { includeFastCorpus: true });
assert.ok(
  fastPlan.commands.map(commandToString).includes("pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:fast"),
);

const compilePlan = planForChangedFiles([
  "packages/browsergrad-compiler/src/semantic_wgsl_pointers.ts",
], { includeCompileCorpus: true });
assert.ok(
  compilePlan.commands.map(commandToString).includes("pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:compile"),
);

const dedupedFastPlan = planForChangedFiles([
  "packages/browsergrad-compiler/src/analyzer.ts",
  "scripts/verify-real-world-cuda.mjs",
], { includeFastCorpus: true });
assert.equal(
  dedupedFastPlan.commands.map(commandToString)
    .filter((cmd) => cmd === "pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:fast")
    .length,
  1,
);

const docsPlan = planForChangedFiles([
  "docs/internal/status-detailed.md",
]);
assert.equal(docsPlan.scopes.length, 0);
assert.equal(docsPlan.commands.length, 0);

const timingDir = fs.mkdtempSync(path.join(os.tmpdir(), "browsergrad-test-scope-"));
const timingJson = path.join(timingDir, "timing.json");
const timingResult = spawnSync(process.execPath, [
  path.join(root, "scripts/cuda-lite-test-scope.mjs"),
  "--run",
  "--files",
  "docs/internal/status-detailed.md",
  "--timing-json",
  timingJson,
], { cwd: root, encoding: "utf8" });
assert.equal(timingResult.status, 0, timingResult.stderr || timingResult.stdout);
const timingReport = JSON.parse(fs.readFileSync(timingJson, "utf8"));
assert.equal(timingReport.tool, "browsergrad-cuda-lite-test-scope");
assert.equal(timingReport.exitCode, 0);
assert.deepEqual(timingReport.commands, []);

const rawDashJson = spawnSync(process.execPath, [
  path.join(root, "scripts/cuda-lite-test-scope.mjs"),
  "--",
  "--json",
  "--files",
  "packages/browsergrad-compiler/src/semantic_wgsl_atomic_analysis.ts",
], { cwd: root, encoding: "utf8" });
assert.equal(rawDashJson.status, 0, rawDashJson.stderr || rawDashJson.stdout);
assert.match(rawDashJson.stdout, /"wgsl-atomics"/u);

const scopeCliJson = spawnSync(process.execPath, [
  path.join(root, "scripts/cuda-lite-test-scope.mjs"),
  "--json",
  "--files",
  "docs/internal/status-detailed.md",
  "--scope",
  "atomic",
], { cwd: root, encoding: "utf8" });
assert.equal(scopeCliJson.status, 0, scopeCliJson.stderr || scopeCliJson.stdout);
assert.match(scopeCliJson.stdout, /atomic:helper-rmw/u);
const scopeCliPlan = JSON.parse(scopeCliJson.stdout);
assert.deepEqual(scopeCliPlan.scopes.map((scope) => scope.files), [[], []]);
assert.deepEqual(scopeCliPlan.scopes.map((scope) => scope.requested), [true, true]);

const compilerPackage = JSON.parse(fs.readFileSync(path.join(root, "packages/browsergrad-compiler/package.json"), "utf8"));
assert.match(compilerPackage.scripts.build, /pnpm run clean/u);
const expectedSmokeCases = new Set([
  "example:saxpy",
  ...runtimeScopeCases,
  ...storageScopeCases,
  ...atomicScopeCases,
  ...controlScopeCases,
  ...textureScopeCases,
]);
assert.deepEqual(webGpuSmokeCases, [...expectedSmokeCases]);
assert.equal(new Set(webGpuSmokeCases).size, webGpuSmokeCases.length);
assert.equal(new Set(webGpuSlowSmokeHotCases).size, webGpuSlowSmokeHotCases.length);
for (const name of webGpuSlowSmokeHotCases) assert.ok(expectedSmokeCases.has(name), `${name} should be part of WebGPU smoke`);
assert.equal(compilerPackage.scripts["e2e:webgpu:compile"].includes("--skip-build"), false);
assert.equal(compilerPackage.scripts["e2e:webgpu:fast"].includes("--skip-build"), false);
assert.equal(compilerPackage.scripts["e2e:webgpu:smoke"], "node ../../scripts/run-cuda-lite-webgpu-smoke.mjs");
assert.equal(compilerPackage.scripts["e2e:webgpu:slow-smoke-hot"], "node ../../scripts/run-cuda-lite-webgpu-slow-smoke-hot.mjs");
assert.equal(compilerPackage.scripts["bugbash:status"], "node ../../scripts/cuda-lite-bugbash-status.mjs");
assert.match(compilerPackage.scripts["e2e:webgpu:hot-case:gate"], /--warmup 1/u);
assert.match(compilerPackage.scripts["e2e:webgpu:hot-case:gate"], /--expect-warm-ms-max 12/u);
assert.doesNotMatch(compilerPackage.scripts["e2e:webgpu:hot-case:gate"], /--expect-warm-speedup-min/u);
assert.match(compilerPackage.scripts["e2e:webgpu:corpus-hot"], /--forbid-skips/u);
assert.match(compilerPackage.scripts["e2e:webgpu:corpus-hot"], /histogram64Kernel/u);
assert.match(compilerPackage.scripts["e2e:webgpu:corpus-hot"], /scalarProdGPU/u);
assert.match(compilerPackage.scripts["e2e:webgpu:corpus-hot"], /--profile-case/u);
assert.match(compilerPackage.scripts["e2e:webgpu:corpus-hot"], /--warmup 1/u);
assert.match(compilerPackage.scripts["e2e:webgpu:corpus-hot"], /--expect-warm-ms-max 12/u);
assert.match(compilerPackage.scripts["e2e:webgpu:volume-minmax-hot"], /--forbid-skips/u);
assert.match(compilerPackage.scripts["e2e:webgpu:volume-minmax-hot"], /volume-vector-pointer-array-minmax-active-lane-return,/u);
assert.match(compilerPackage.scripts["e2e:webgpu:volume-minmax-hot"], /volume-vector-pointer-array-minmax-active-lane-return-false-branch/u);
assert.match(compilerPackage.scripts["e2e:webgpu:volume-minmax-hot"], /--profile-case/u);
assert.match(compilerPackage.scripts["e2e:webgpu:volume-minmax-hot"], /--expect-warm-ms-max 12/u);
assert.match(compilerPackage.scripts["e2e:webgpu:auto-corpus-hot"], /--auto-corpus-smoke-only/u);
assert.match(compilerPackage.scripts["e2e:webgpu:auto-corpus-hot"], /--cases 'auto-corpus:/u);
assert.match(compilerPackage.scripts["e2e:webgpu:auto-corpus-hot"], /--profile-case 'auto-corpus:/u);
assert.match(compilerPackage.scripts["e2e:webgpu:auto-corpus-hot"], /synchronizedKernel:6:1/u);
assert.match(compilerPackage.scripts["e2e:webgpu:auto-corpus-hot"], /sharedMemoryExample:7:1/u);
assert.match(compilerPackage.scripts["e2e:webgpu:auto-corpus-hot"], /--expect-warm-ms-max 8/u);

console.log("cuda-lite test scope tests ok");
