#!/usr/bin/env node
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { formatStatus, gatherStatus } from "./cuda-lite-bugbash-status.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "browsergrad-bugbash-status-"));
fs.mkdirSync(path.join(tmp, "docs/internal"), { recursive: true });
fs.mkdirSync(path.join(tmp, ".tmp"), { recursive: true });
fs.writeFileSync(path.join(tmp, "docs/internal/compiler-bugbash-progress.md"), `# Compiler Bugbash Progress

Last updated: 2026-07-01T10:00:00Z

## Dashboard

| Field | Current |
| --- | --- |
| Overall status | Active bugbash, not complete |
| Fixed failure movement | Started from 87 failing cases; current focused gate is green |
| Active work item | Fix control probe |
| Skip policy | No added skips |

## Latest Proven Green Gates

- compiler unit suite: 374 passed
- WebGPU smoke: 111 passed / 0 failed / 0 skipped

## Remaining Probe Map

- Texture family:
  - texture active-lane read probe
- Active-lane/control family:
  - keep probing texture reads plus active-lane writes
`);
fs.writeFileSync(path.join(tmp, ".tmp/cuda-lite-last-failures.json"), JSON.stringify({
  tool: "browsergrad-cuda-lite-last-failures",
  cases: ["control:probe"],
  failedByStage: { diagnostics: 1 },
}, null, 2));

const status = gatherStatus(tmp);
assert.equal(status.lastUpdated, "2026-07-01T10:00:00Z");
assert.equal(status.dashboard["Overall status"], "Active bugbash, not complete");
assert.equal(status.progress.activeFailureCount, 1);
assert.equal(status.progress.movement, "Started from 87 failing cases; current focused gate is green");
assert.equal(status.progress.latestUnitGate, "compiler unit suite: 374 passed");
assert.equal(status.progress.latestSmokeGate, "WebGPU smoke: 111 passed / 0 failed / 0 skipped");
assert.equal(status.progress.remainingProbeCount, 2);
assert.deepEqual(status.activeFailures.cases, ["control:probe"]);
assert.equal(status.activeFailures.failedByStage.diagnostics, 1);
assert.deepEqual(status.remainingProbes, [
  "Texture family: texture active-lane read probe",
  "Active-lane/control family: keep probing texture reads plus active-lane writes",
]);
assert.ok(status.nextCommands.includes("pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:last-failures"));
assert.match(formatStatus(status), /Progress:/u);
assert.match(formatStatus(status), /Movement: Started from 87 failing cases/u);
assert.match(formatStatus(status), /Latest smoke proof: WebGPU smoke: 111 passed/u);
assert.match(formatStatus(status), /Active failures: 1/u);
assert.match(formatStatus(status), /control:probe/u);
assert.match(formatStatus(status), /Remaining probes:/u);
assert.match(formatStatus(status), /Texture family: texture active-lane read probe/u);

const cli = spawnSync(process.execPath, [
  path.join(root, "scripts/cuda-lite-bugbash-status.mjs"),
  "--root",
  tmp,
  "--json",
], { cwd: root, encoding: "utf8" });
assert.equal(cli.status, 0, cli.stderr || cli.stdout);
assert.equal(JSON.parse(cli.stdout).activeFailures.cases[0], "control:probe");
assert.equal(JSON.parse(cli.stdout).remainingProbes[0], "Texture family: texture active-lane read probe");

console.log("cuda-lite bugbash status tests ok");
