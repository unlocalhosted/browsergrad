#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { findRepoRoot } from "./cuda-lite-webgpu-cli.mjs";
import { webGpuSmokeCases } from "./cuda-lite-webgpu-smoke-cases.mjs";

const root = findRepoRoot(process.cwd());
const args = [
  "--filter",
  "@unlocalhosted/browsergrad-compiler",
  "run",
  "e2e:webgpu:case",
  "--",
  "--cases",
  webGpuSmokeCases.join(","),
  "--case-timeout-ms",
  "15000",
  ...process.argv.slice(2),
];

const result = spawnSync("pnpm", args, { cwd: root, stdio: "inherit" });
if (result.error) throw result.error;
process.exit(result.status ?? 1);
