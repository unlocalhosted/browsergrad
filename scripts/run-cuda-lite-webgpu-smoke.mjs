#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import path from "node:path";
import { findRepoRoot } from "./cuda-lite-webgpu-cli.mjs";
import { webGpuSmokeCases } from "./cuda-lite-webgpu-smoke-cases.mjs";

const root = findRepoRoot(process.cwd());
const args = [
  path.join(root, "scripts/run-cuda-lite-tool.mjs"),
  "e2e:webgpu",
  "--skip-build",
  "--",
  "--require-webgpu",
  "--forbid-skips",
  "--summary-only",
  "--fail-fast",
  "--case-timeout-ms",
  "10000",
  "--cases",
  webGpuSmokeCases.join(","),
  "--profile-case",
  "all",
  "--case-timeout-ms",
  "15000",
  ...process.argv.slice(2),
];

const result = spawnSync(process.execPath, args, { cwd: root, stdio: "inherit" });
if (result.error) throw result.error;
process.exit(result.status ?? 1);
