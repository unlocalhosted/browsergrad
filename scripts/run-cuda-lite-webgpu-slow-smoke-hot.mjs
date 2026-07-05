#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { findRepoRoot } from "./cuda-lite-webgpu-cli.mjs";
import { webGpuSlowSmokeHotCases } from "./cuda-lite-webgpu-smoke-cases.mjs";

export function buildWebGpuSlowSmokeHotArgs(root, extraArgs = []) {
  const cases = webGpuSlowSmokeHotCases.join(",");
  return [
    path.join(root, "scripts/run-cuda-lite-tool.mjs"),
    "e2e:webgpu",
    "--skip-build",
    "--",
    "--require-webgpu",
    "--forbid-skips",
    "--summary-only",
    "--fail-fast",
    "--cases",
    cases,
    "--profile-case",
    cases,
    "--warmup",
    "1",
    "--repeat",
    "2",
    "--expect-warm-ms-max",
    "20",
    "--case-timeout-ms",
    "10000",
    ...extraArgs,
  ];
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const root = findRepoRoot(process.cwd());
  const result = spawnSync(process.execPath, buildWebGpuSlowSmokeHotArgs(root, process.argv.slice(2)), { cwd: root, stdio: "inherit" });
  if (result.error) throw result.error;
  process.exit(result.status ?? 1);
}
