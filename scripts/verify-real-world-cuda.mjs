#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseAutoCorpusSmokeProfile } from "./cuda-lite-webgpu-cli.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");

export function parseVerifyRealWorldCudaArgs(args) {
  const options = {
    skipFetch: false,
    allowMissingWebGpu: false,
    limit: 0,
    bundle: "both",
    autoCorpusSmokeLimit: 32,
    autoCorpusSmokeMode: "reference",
    autoCorpusSmokeProfile: "fast",
    autoCorpusSmokeFeatures: [],
    caseTimeoutMs: 0,
    benchmarkWebGpu: false,
    benchmarkRuns: 8,
    benchmarkWarmup: 2,
    benchmarkLength: 4096,
    preparedRatioMax: undefined,
    preparedScalarRatioMax: undefined,
    preparedReadbackRatioMax: undefined,
  };
  for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === "--skip-fetch") {
      options.skipFetch = true;
      continue;
    }
    if (arg === "--require-webgpu") {
      options.allowMissingWebGpu = false;
      continue;
    }
    if (arg === "--allow-missing-webgpu") {
      options.allowMissingWebGpu = true;
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
    if (arg === "--bundle") {
      options.bundle = parseBundle(args[++index]);
      continue;
    }
    if (arg?.startsWith("--bundle=")) {
      options.bundle = parseBundle(arg.slice("--bundle=".length));
      continue;
    }
    if (arg === "--auto-corpus-smoke-limit") {
      options.autoCorpusSmokeLimit = parseLimit(args[++index]);
      continue;
    }
    if (arg?.startsWith("--auto-corpus-smoke-limit=")) {
      options.autoCorpusSmokeLimit = parseLimit(arg.slice("--auto-corpus-smoke-limit=".length));
      continue;
    }
    if (arg === "--auto-corpus-smoke-mode") {
      options.autoCorpusSmokeMode = parseAutoCorpusSmokeMode(args[++index]);
      continue;
    }
    if (arg?.startsWith("--auto-corpus-smoke-mode=")) {
      options.autoCorpusSmokeMode = parseAutoCorpusSmokeMode(arg.slice("--auto-corpus-smoke-mode=".length));
      continue;
    }
    if (arg === "--auto-corpus-smoke-profile") {
      options.autoCorpusSmokeProfile = parseAutoCorpusSmokeProfile(args[++index]);
      continue;
    }
    if (arg?.startsWith("--auto-corpus-smoke-profile=")) {
      options.autoCorpusSmokeProfile = parseAutoCorpusSmokeProfile(arg.slice("--auto-corpus-smoke-profile=".length));
      continue;
    }
    if (arg === "--auto-corpus-smoke-features") {
      options.autoCorpusSmokeFeatures = parseFeatureList(args[++index]);
      continue;
    }
    if (arg?.startsWith("--auto-corpus-smoke-features=")) {
      options.autoCorpusSmokeFeatures = parseFeatureList(arg.slice("--auto-corpus-smoke-features=".length));
      continue;
    }
    if (arg === "--case-timeout-ms") {
      options.caseTimeoutMs = parseLimit(args[++index], "--case-timeout-ms");
      continue;
    }
    if (arg?.startsWith("--case-timeout-ms=")) {
      options.caseTimeoutMs = parseLimit(arg.slice("--case-timeout-ms=".length), "--case-timeout-ms");
      continue;
    }
    if (arg === "--benchmark-webgpu") {
      options.benchmarkWebGpu = true;
      continue;
    }
    if (arg === "--benchmark-runs") {
      options.benchmarkRuns = parsePositiveInt(args[++index], "--benchmark-runs");
      continue;
    }
    if (arg?.startsWith("--benchmark-runs=")) {
      options.benchmarkRuns = parsePositiveInt(arg.slice("--benchmark-runs=".length), "--benchmark-runs");
      continue;
    }
    if (arg === "--benchmark-warmup") {
      options.benchmarkWarmup = parsePositiveInt(args[++index], "--benchmark-warmup");
      continue;
    }
    if (arg?.startsWith("--benchmark-warmup=")) {
      options.benchmarkWarmup = parsePositiveInt(arg.slice("--benchmark-warmup=".length), "--benchmark-warmup");
      continue;
    }
    if (arg === "--benchmark-length") {
      options.benchmarkLength = parsePositiveInt(args[++index], "--benchmark-length");
      continue;
    }
    if (arg?.startsWith("--benchmark-length=")) {
      options.benchmarkLength = parsePositiveInt(arg.slice("--benchmark-length=".length), "--benchmark-length");
      continue;
    }
    if (arg === "--expect-prepared-ratio-max") {
      options.preparedRatioMax = parsePositiveNumber(args[++index], "--expect-prepared-ratio-max");
      continue;
    }
    if (arg?.startsWith("--expect-prepared-ratio-max=")) {
      options.preparedRatioMax = parsePositiveNumber(arg.slice("--expect-prepared-ratio-max=".length), "--expect-prepared-ratio-max");
      continue;
    }
    if (arg === "--expect-prepared-scalar-ratio-max") {
      options.preparedScalarRatioMax = parsePositiveNumber(args[++index], "--expect-prepared-scalar-ratio-max");
      continue;
    }
    if (arg?.startsWith("--expect-prepared-scalar-ratio-max=")) {
      options.preparedScalarRatioMax = parsePositiveNumber(arg.slice("--expect-prepared-scalar-ratio-max=".length), "--expect-prepared-scalar-ratio-max");
      continue;
    }
    if (arg === "--expect-prepared-readback-ratio-max") {
      options.preparedReadbackRatioMax = parsePositiveNumber(args[++index], "--expect-prepared-readback-ratio-max");
      continue;
    }
    if (arg?.startsWith("--expect-prepared-readback-ratio-max=")) {
      options.preparedReadbackRatioMax = parsePositiveNumber(arg.slice("--expect-prepared-readback-ratio-max=".length), "--expect-prepared-readback-ratio-max");
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      console.log("usage: node scripts/verify-real-world-cuda.mjs [--skip-fetch] [--require-webgpu] [--allow-missing-webgpu] [--limit N] [--bundle src|dist|both] [--auto-corpus-smoke-limit N] [--auto-corpus-smoke-mode reference|dispatch] [--auto-corpus-smoke-profile fast|full] [--auto-corpus-smoke-features subgroups] [--case-timeout-ms N] [--benchmark-webgpu] [--benchmark-runs N] [--benchmark-warmup N] [--benchmark-length N] [--expect-prepared-ratio-max N] [--expect-prepared-scalar-ratio-max N] [--expect-prepared-readback-ratio-max N]");
      process.exit(0);
    }
    throw new Error(`unexpected argument: ${arg}`);
  }
  return options;
}

export function verifyRealWorldCudaPlan(options) {
  const steps = [
    {
      label: "real-world CUDA compile/codegen audit",
      args: [
        path.join(scriptDir, "audit-real-world-cuda-corpora.mjs"),
        ...(options.skipFetch ? ["--skip-fetch"] : []),
        "--limit",
        String(options.limit),
      ],
    },
  ];
  for (const bundle of browserBundles(options.bundle)) {
    steps.push({
      label: `real-world CUDA browser fixture e2e (${bundle})`,
      args: [
        path.join(scriptDir, "e2e-cuda-lite-webgpu.mjs"),
        "--require-corpus-fixtures",
        "--forbid-skips",
        "--summary-only",
        "--bundle",
        bundle,
        "--auto-corpus-smoke-limit",
        String(options.autoCorpusSmokeLimit),
        "--auto-corpus-smoke-mode",
        options.autoCorpusSmokeMode,
        "--auto-corpus-smoke-profile",
        options.autoCorpusSmokeProfile,
        "--auto-corpus-smoke-features",
        options.autoCorpusSmokeFeatures.join(","),
        "--case-timeout-ms",
        String(options.caseTimeoutMs),
        ...(options.allowMissingWebGpu ? [] : ["--require-webgpu"]),
      ],
    });
    if (options.benchmarkWebGpu) {
      steps.push({
        label: `real-world CUDA browser perf gate (${bundle})`,
        args: [
          path.join(scriptDir, "benchmark-cuda-lite-webgpu.mjs"),
          "--bundle",
          bundle,
          "--runs",
          String(options.benchmarkRuns),
          "--warmup",
          String(options.benchmarkWarmup),
          "--length",
          String(options.benchmarkLength),
          ...ratioArg("--expect-prepared-ratio-max", options.preparedRatioMax),
          ...ratioArg("--expect-prepared-scalar-ratio-max", options.preparedScalarRatioMax),
          ...ratioArg("--expect-prepared-readback-ratio-max", options.preparedReadbackRatioMax),
          ...(options.allowMissingWebGpu ? [] : ["--require-webgpu"]),
        ],
      });
    }
  }
  return steps;
}

function parseBundle(raw) {
  if (raw === "src" || raw === "dist" || raw === "both") return raw;
  throw new Error("--bundle expects src, dist, or both");
}

function parseAutoCorpusSmokeMode(raw) {
  if (raw === "reference" || raw === "dispatch") return raw;
  throw new Error("--auto-corpus-smoke-mode expects reference or dispatch");
}

function parseFeatureList(raw) {
  if (!raw) return [];
  return raw.split(",").map((item) => item.trim()).filter(Boolean);
}

function browserBundles(bundle) {
  return bundle === "both" ? ["src", "dist"] : [bundle];
}

function parseLimit(raw, flag = "--limit") {
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${flag} expects a non-negative integer`);
  }
  return value;
}

function parsePositiveInt(raw, flag) {
  const value = Number(raw);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${flag} expects a positive integer`);
  }
  return value;
}

function parsePositiveNumber(raw, flag) {
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${flag} expects a positive number`);
  }
  return value;
}

function ratioArg(flag, value) {
  return value === undefined ? [] : [flag, String(value)];
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

function main() {
  const options = parseVerifyRealWorldCudaArgs(process.argv.slice(2));
  for (const step of verifyRealWorldCudaPlan(options)) run(step.label, step.args);
  console.log("\nreal-world CUDA verification passed");
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
