#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { findRepoRoot } from "./cuda-lite-webgpu-cli.mjs";
import { webGpuSlowSmokeHotCases, webGpuSmokeCases } from "./cuda-lite-webgpu-smoke-cases.mjs";

export function buildWebGpuSmokeArgs(root, extraArgs = [], env = process.env) {
  const profileValue = env.CUDA_LITE_WEBGPU_SMOKE_PROFILE ?? "slow";
  const profileCases = profileValue === "slow"
    ? webGpuSlowSmokeHotCases.join(",")
    : profileValue === "true"
      ? "all"
    : profileValue;
  const profileArgs = profileCases && profileCases !== "none" && profileCases !== "false" ? ["--profile-case", profileCases] : [];
  const timeoutMs = env.CUDA_LITE_WEBGPU_SMOKE_TIMEOUT_MS ?? "15000";
  const recycleInterval = env.CUDA_LITE_WEBGPU_SMOKE_DEVICE_RECYCLE_INTERVAL ?? "128";
  return [
    path.join(root, "scripts/run-cuda-lite-tool.mjs"),
    "e2e:webgpu",
    "--skip-build",
    "--",
    "--require-webgpu",
    "--forbid-skips",
    "--summary-only",
    "--fail-fast",
    "--case-timeout-ms",
    timeoutMs,
    "--device-recycle-interval",
    recycleInterval,
    "--cases",
    webGpuSmokeCases.join(","),
    ...profileArgs,
    ...extraArgs,
  ];
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const root = findRepoRoot(process.cwd());
  const result = spawnSync(process.execPath, buildWebGpuSmokeArgs(root, process.argv.slice(2)), { cwd: root, stdio: "inherit" });
  if (result.error) throw result.error;
  process.exit(result.status ?? 1);
}
