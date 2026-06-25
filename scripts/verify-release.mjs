#!/usr/bin/env node

import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const gates = [
  [
    "pnpm",
    [
      "validate:prd",
      "docs/prd/PRD-017-research-gated-prd-workflow.md",
      "docs/prd/PRD-018-lab-core-capability-spine.md",
      "docs/prd/PRD-020-real-world-cuda-compatibility-ladder.md",
    ],
    repoRoot,
  ],
  ["pnpm", ["-r", "run", "build"], repoRoot],
  ["pnpm", ["-r", "run", "typecheck"], repoRoot],
  ["pnpm", ["-r", "run", "test"], repoRoot],
  ["pnpm", ["-r", "run", "lint"], repoRoot],
  ["pnpm", ["--filter", "@unlocalhosted/browsergrad-jit", "test:integration"], repoRoot],
  ["pnpm", ["--filter", "@unlocalhosted/browsergrad-grad", "test:integration"], repoRoot],
  ["pnpm", ["--filter", "@unlocalhosted/browsergrad-runtime", "test:integration"], repoRoot],
  ["pnpm", ["--filter", "@unlocalhosted/browsergrad-kernels", "test:browser"], repoRoot],
  ["pnpm", ["--filter", "@unlocalhosted/browsergrad-compiler", "test:browser"], repoRoot],
  ["pnpm", ["--filter", "@unlocalhosted/browsergrad-compiler", "test:source-normalizer"], repoRoot],
  ["pnpm", ["--filter", "@unlocalhosted/browsergrad-compiler", "test:audit-corpus"], repoRoot],
  ["pnpm", ["test"], resolve(repoRoot, "packages/browsergrad-dogfood")],
];

for (const [cmd, args, cwd] of gates) {
  await run(cmd, args, cwd);
}

function run(cmd, args, cwd) {
  console.log(`\n$ ${cmd} ${args.join(" ")}`);
  return new Promise((resolvePromise, reject) => {
    const child = spawn(cmd, args, { cwd, stdio: "inherit" });
    child.on("error", reject);
    child.on("exit", (code, signal) => {
      if (code === 0) {
        resolvePromise();
        return;
      }
      reject(new Error(`${cmd} ${args.join(" ")} failed with ${signal ?? code}`));
    });
  });
}
