#!/usr/bin/env node
import fs from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { findRepoRoot } from "./cuda-lite-webgpu-cli.mjs";

const root = findRepoRoot(process.cwd());
const failurePath = path.join(root, ".tmp", "cuda-lite-last-failures.json");

if (!fs.existsSync(failurePath)) {
  console.error(`No last-failure file found: ${failurePath}`);
  process.exit(1);
}

const report = JSON.parse(fs.readFileSync(failurePath, "utf8"));
const cases = Array.isArray(report.cases) ? report.cases.filter(Boolean) : [];
if (cases.length === 0) {
  console.error(`No failed cases in: ${failurePath}`);
  process.exit(1);
}

const args = [
  "--filter",
  "@unlocalhosted/browsergrad-compiler",
  "run",
  "e2e:webgpu:case",
  "--",
  "--cases",
  cases.join(","),
  ...process.argv.slice(2),
];
const result = spawnSync("pnpm", args, { cwd: root, stdio: "inherit" });
if (result.error) throw result.error;
if ((result.status ?? 1) === 0) fs.rmSync(failurePath, { force: true });
process.exit(result.status ?? 1);
