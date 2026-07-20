#!/usr/bin/env node
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseAutoCorpusSmokeProfile } from "./cuda-lite-webgpu-cli.mjs";
import {
  cudaLiteCorpora,
  cudaLiteCorpusExecutionFixtures,
} from "./cuda-lite-corpus-registry.mjs";
import { aggregateBrowserShardReports } from "./real-world-cuda-browser-shard-evidence.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");
const CAPTURED_OUTPUT_BYTE_LIMIT = 2 * 1024 * 1024;
const BROWSER_SHARD_LIMIT = 8;

export function parseVerifyRealWorldCudaArgs(args) {
  const options = {
    skipFetch: false,
    allowMissingWebGpu: false,
    limit: 0,
    only: [],
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
    browserShards: 1,
    timingJson: undefined,
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
    if (arg === "--forbid-skips") {
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
    if (arg === "--only" || arg === "--corpus") {
      options.only.push(...parseCorpusList(args[++index]));
      continue;
    }
    if (arg?.startsWith("--only=")) {
      options.only.push(...parseCorpusList(arg.slice("--only=".length)));
      continue;
    }
    if (arg?.startsWith("--corpus=")) {
      options.only.push(...parseCorpusList(arg.slice("--corpus=".length)));
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
    if (arg === "--browser-shards") {
      options.browserShards = parseBrowserShardCount(args[++index]);
      continue;
    }
    if (arg?.startsWith("--browser-shards=")) {
      options.browserShards = parseBrowserShardCount(arg.slice("--browser-shards=".length));
      continue;
    }
    if (arg === "--timing-json") {
      options.timingJson = parseOutputPath(args[++index], "--timing-json");
      continue;
    }
    if (arg?.startsWith("--timing-json=")) {
      options.timingJson = parseOutputPath(arg.slice("--timing-json=".length), "--timing-json");
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      console.log("usage: node scripts/verify-real-world-cuda.mjs [--skip-fetch] [--require-webgpu] [--allow-missing-webgpu] [--forbid-skips] [--limit N] [--only CORPUS[,CORPUS...]] [--corpus CORPUS[,CORPUS...]] [--bundle src|dist|both] [--auto-corpus-smoke-limit N] [--auto-corpus-smoke-mode reference|dispatch] [--auto-corpus-smoke-profile fast|full] [--auto-corpus-smoke-features subgroups] [--case-timeout-ms N] [--browser-shards N] [--benchmark-webgpu] [--benchmark-runs N] [--benchmark-warmup N] [--benchmark-length N] [--expect-prepared-ratio-max N] [--expect-prepared-scalar-ratio-max N] [--expect-prepared-readback-ratio-max N] [--timing-json PATH]");
      process.exit(0);
    }
    throw new Error(`unexpected argument: ${arg}`);
  }
  return options;
}

export function verifyRealWorldCudaPlan(options, context = {}) {
  const selectedCorpusIds = selectedCorpora(options.only);
  const browserReportDir = context.browserReportDir ?? path.join(root, ".tmp", "real-world-cuda-browser-plan");
  const steps = selectedCorpusIds.map((id) => ({
    label: `real-world CUDA compile/codegen audit (${id})`,
    parallelGroup: "compile-codegen-audits",
    args: [
      path.join(scriptDir, "audit-real-world-cuda-corpora.mjs"),
      ...(options.skipFetch ? ["--skip-fetch"] : []),
      "--only",
      id,
      "--limit",
      String(options.limit),
    ],
  }));
  const browserShardCount = effectiveBrowserShardCount(options);
  for (const bundle of browserBundles(options.bundle)) {
    for (let shardIndex = 1; shardIndex <= browserShardCount; shardIndex++) {
      const expectedFixtureCases = browserShardFixtureCases(options.only, shardIndex, browserShardCount);
      const expectedOutputFixtureCases = expectedFixtureCases.filter((caseName) =>
        fixtureByCaseName(caseName)?.expectedOutput !== undefined);
      const reportPath = path.join(browserReportDir, `${bundle}-${shardIndex}-of-${browserShardCount}.json`);
      steps.push({
        kind: "browser-e2e",
        label: `real-world CUDA browser fixture e2e (${bundle}, shard ${shardIndex}/${browserShardCount})`,
        ...(browserShardCount > 1 ? { parallelGroup: `browser-e2e-${bundle}` } : {}),
        bundle,
        shardIndex,
        shardCount: browserShardCount,
        requireWebGpu: !options.allowMissingWebGpu,
        forbidSkips: true,
        caseTimeoutMs: options.caseTimeoutMs,
        reportPath,
        expectedFixtureCases,
        expectedOutputFixtureCases,
        args: [
          path.join(scriptDir, "e2e-cuda-lite-webgpu.mjs"),
          "--require-corpus-fixtures",
          "--forbid-skips",
          "--summary-only",
          "--json",
          reportPath,
          "--bundle",
          bundle,
          ...browserShardCaseFilterArgs(options.only, options.autoCorpusSmokeLimit, shardIndex, browserShardCount),
          "--auto-corpus-smoke-limit",
          String(options.autoCorpusSmokeLimit),
          "--auto-corpus-smoke-mode",
          options.autoCorpusSmokeMode,
          "--auto-corpus-smoke-profile",
          options.autoCorpusSmokeProfile,
          "--auto-corpus-smoke-features",
          options.autoCorpusSmokeFeatures.join(","),
          "--auto-corpus-smoke-shard",
          `${shardIndex}/${browserShardCount}`,
          ...autoCorpusSmokeCorporaArgs(options.only, options.autoCorpusSmokeLimit),
          "--case-timeout-ms",
          String(options.caseTimeoutMs),
          ...(options.allowMissingWebGpu ? [] : ["--require-webgpu"]),
        ],
      });
    }
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

function parseCorpusList(raw) {
  if (!raw) throw new Error("--only/--corpus expects a corpus id");
  const known = new Set(cudaLiteCorpora.map((corpus) => corpus.id));
  const values = raw.split(",").map((item) => item.trim()).filter(Boolean);
  if (values.length === 0) throw new Error("--only/--corpus expects a corpus id");
  for (const value of values) {
    if (!known.has(value)) {
      throw new Error(`unknown CUDA-lite corpus '${value}', expected one of: ${[...known].join(", ")}`);
    }
  }
  return values;
}

function parseFeatureList(raw) {
  if (!raw) return [];
  return raw.split(",").map((item) => item.trim()).filter(Boolean);
}

function browserBundles(bundle) {
  return bundle === "both" ? ["src", "dist"] : [bundle];
}

function selectedCorpora(only) {
  return only.length === 0
    ? cudaLiteCorpora.map((corpus) => corpus.id)
    : [...new Set(only)];
}

function selectedFixtureCases(only) {
  const corpusIds = new Set(selectedCorpora(only));
  return cudaLiteCorpusExecutionFixtures
    .filter((fixture) => corpusIds.has(fixture.corpusId))
    .map((fixture) => fixture.caseName);
}

function fixtureByCaseName(caseName) {
  return cudaLiteCorpusExecutionFixtures.find((fixture) => fixture.caseName === caseName);
}

function effectiveBrowserShardCount(options) {
  const workItems = selectedFixtureCases(options.only).length + options.autoCorpusSmokeLimit;
  return Math.min(options.browserShards, Math.max(1, workItems));
}

function parseBrowserShardCount(raw) {
  const value = parsePositiveInt(raw, "--browser-shards");
  if (value > BROWSER_SHARD_LIMIT) {
    throw new Error(`--browser-shards must not exceed ${BROWSER_SHARD_LIMIT}`);
  }
  return value;
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

function parseOutputPath(raw, flag) {
  if (typeof raw !== "string" || raw.length === 0 || raw.startsWith("-")) {
    throw new Error(`${flag} expects a path`);
  }
  return raw;
}

function ratioArg(flag, value) {
  return value === undefined ? [] : [flag, String(value)];
}

function browserShardFixtureCases(only, shardIndex, shardCount) {
  return selectedFixtureCases(only)
    .filter((_, index) => index % shardCount === shardIndex - 1);
}

function browserShardCaseFilterArgs(only, autoCorpusSmokeLimit, shardIndex, shardCount) {
  const filters = [
    ...browserShardFixtureCases(only, shardIndex, shardCount),
    ...(autoCorpusSmokeLimit > 0
      ? selectedCorpora(only).map((id) => `auto-corpus:${id}:`)
      : []),
  ];
  return ["--cases", filters.join(",")];
}

function autoCorpusSmokeCorporaArgs(only, autoCorpusSmokeLimit) {
  if (only.length === 0 || autoCorpusSmokeLimit === 0) return [];
  return ["--auto-corpus-smoke-corpora", only.join(",")];
}

function run(step, captureOutput) {
  console.log(`\n== ${step.label} ==`);
  const startedNs = process.hrtime.bigint();
  return new Promise((resolve) => {
    const child = spawn(process.execPath, step.args, {
      cwd: root,
      stdio: captureOutput ? ["ignore", "pipe", "pipe"] : "inherit",
    });
    const stdout = { chunks: [], byteLength: 0, truncatedBytes: 0 };
    const stderr = { chunks: [], byteLength: 0, truncatedBytes: 0 };
    let spawnError;
    child.stdout?.on("data", (chunk) => appendCapturedOutput(stdout, chunk));
    child.stderr?.on("data", (chunk) => appendCapturedOutput(stderr, chunk));
    child.on("error", (error) => { spawnError = error; });
    child.on("close", (code, signal) => {
      if (captureOutput) {
        writeCapturedOutput(process.stdout, stdout, step.label, "stdout");
        writeCapturedOutput(process.stderr, stderr, step.label, "stderr");
      }
      resolve(finishTiming(step, startedNs, code ?? 1, signal, spawnError));
    });
  });
}

function appendCapturedOutput(capture, chunk) {
  const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
  const remaining = CAPTURED_OUTPUT_BYTE_LIMIT - capture.byteLength;
  if (remaining > 0) {
    const retained = Buffer.from(
      bytes.byteLength <= remaining ? bytes : bytes.subarray(0, remaining),
    );
    capture.chunks.push(retained);
    capture.byteLength += retained.byteLength;
  }
  capture.truncatedBytes += Math.max(0, bytes.byteLength - Math.max(0, remaining));
}

function writeCapturedOutput(stream, capture, label, streamName) {
  if (capture.byteLength > 0) stream.write(Buffer.concat(capture.chunks, capture.byteLength));
  if (capture.truncatedBytes > 0) {
    stream.write(
      `\n[${label} ${streamName} truncated by ${capture.truncatedBytes} bytes after ` +
        `${CAPTURED_OUTPUT_BYTE_LIMIT}-byte harness limit]\n`,
    );
  }
}

function finishTiming(step, startedNs, exitCode, signal, error) {
  const durationMs = Number(process.hrtime.bigint() - startedNs) / 1_000_000;
  const record = Object.freeze({
    label: step.label,
    parallelGroup: step.parallelGroup ?? null,
    durationMs: Math.round(durationMs * 1000) / 1000,
    exitCode,
    signal: signal ?? null,
    ok: exitCode === 0 && (signal === undefined || signal === null) && error === undefined,
    ...(error === undefined ? {} : { error: String(error.message ?? error) }),
  });
  console.log(`== timing: ${step.label}: ${record.durationMs.toFixed(3)} ms ==`);
  return record;
}

export async function executePlan(steps, runStep = run) {
  const timings = [];
  for (let index = 0; index < steps.length;) {
    const step = steps[index];
    if (step.parallelGroup === undefined) {
      const timing = await runStep(step, false);
      timings.push(timing);
      if (!timing.ok) return timings;
      index += 1;
      continue;
    }
    const group = [];
    while (index < steps.length && steps[index].parallelGroup === step.parallelGroup) {
      group.push(steps[index]);
      index += 1;
    }
    const groupTimings = await Promise.all(group.map((item) => runStep(item, true)));
    timings.push(...groupTimings);
    if (groupTimings.some((timing) => !timing.ok)) return timings;
  }
  return timings;
}

function timingReport(options, timings, totalMs, browserEvidence) {
  return Object.freeze({
    kind: "browsergrad-real-world-cuda-timing",
    version: 1,
    bundle: options.bundle,
    selectedCorpora: options.only.length === 0
      ? cudaLiteCorpora.map((corpus) => corpus.id)
      : [...new Set(options.only)],
    browserShards: options.browserShards,
    ok: timings.every((timing) => timing.ok) && browserEvidence.ok,
    totalMs: Math.round(totalMs * 1000) / 1000,
    steps: timings,
    browserEvidence,
  });
}

function writeTimingReport(outputPath, report) {
  const absolute = path.resolve(root, outputPath);
  mkdirSync(path.dirname(absolute), { recursive: true });
  writeFileSync(absolute, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

async function main() {
  const options = parseVerifyRealWorldCudaArgs(process.argv.slice(2));
  const browserReportDir = mkdtempSync(path.join(os.tmpdir(), "browsergrad-real-world-cuda-browser-"));
  const startedNs = process.hrtime.bigint();
  try {
    const plan = verifyRealWorldCudaPlan(options, { browserReportDir });
    const timings = await executePlan(plan);
    const browserEvidence = aggregateBrowserShardReports(plan);
    const totalMs = Number(process.hrtime.bigint() - startedNs) / 1_000_000;
    const report = timingReport(options, timings, totalMs, browserEvidence);
    console.log(`\n${JSON.stringify(report, null, 2)}`);
    if (options.timingJson !== undefined) writeTimingReport(options.timingJson, report);
    const failure = timings.find((timing) => !timing.ok);
    if (failure !== undefined || !browserEvidence.ok) {
      process.exitCode = failure?.exitCode || 1;
      return;
    }
    console.log("\nreal-world CUDA verification passed");
  } finally {
    rmSync(browserReportDir, { recursive: true, force: true });
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
