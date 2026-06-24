#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");
const options = parseArgs(process.argv.slice(2));

run("real-world CUDA compile/codegen audit", [
  path.join(scriptDir, "audit-real-world-cuda-corpora.mjs"),
  ...(options.skipFetch ? ["--skip-fetch"] : []),
  "--limit",
  String(options.limit),
]);

run("real-world CUDA browser fixture e2e", [
  path.join(scriptDir, "e2e-cuda-lite-webgpu.mjs"),
  "--require-corpus-fixtures",
  ...(options.requireWebGpu ? ["--require-webgpu"] : []),
]);

console.log("\nreal-world CUDA verification passed");

function parseArgs(args) {
  const options = {
    skipFetch: false,
    requireWebGpu: false,
    limit: 0,
  };
  for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === "--skip-fetch") {
      options.skipFetch = true;
      continue;
    }
    if (arg === "--require-webgpu") {
      options.requireWebGpu = true;
      continue;
    }
    if (arg === "--limit") {
      options.limit = parseLimit(args[++index]);
      continue;
    }
    if (arg?.startsWith("--limit=")) {
      options.limit = parseLimit(arg.slice("--limit=".length));
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      console.log("usage: node scripts/verify-real-world-cuda.mjs [--skip-fetch] [--require-webgpu] [--limit N]");
      process.exit(0);
    }
    throw new Error(`unexpected argument: ${arg}`);
  }
  return options;
}

function parseLimit(raw) {
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error("--limit expects a non-negative integer");
  }
  return value;
}

function run(label, args) {
  console.log(`\n== ${label} ==`);
  const result = spawnSync(process.execPath, args, {
    cwd: root,
    stdio: "inherit",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}
