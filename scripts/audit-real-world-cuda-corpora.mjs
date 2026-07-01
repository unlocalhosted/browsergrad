#!/usr/bin/env node
import { existsSync, mkdirSync, rmSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { cudaLiteCorpora } from "./cuda-lite-corpus-registry.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");
const options = parseArgs(process.argv.slice(2));
const selected = options.only.size === 0
  ? cudaLiteCorpora
  : cudaLiteCorpora.filter((corpus) => options.only.has(corpus.id));
if (selected.length === 0) {
  console.error(`no matching corpora for --only (${[...options.only].join(", ")})`);
  process.exit(2);
}

const records = [];
const failures = [];

for (const corpus of selected) {
  console.log(`\n== ${corpus.name} @ ${corpus.commit.slice(0, 7)} ==`);
  try {
    if (options.skipFetch) verifyCorpus(corpus);
    else ensureCorpus(corpus);
    const audit = runAudit(corpus, options);
    const summary = audit.summary ?? audit;
    const record = {
      corpus: {
        id: corpus.id,
        name: corpus.name,
        repo: corpus.repo,
        commit: corpus.commit,
        path: corpus.path,
      },
      summary,
      ...(audit.failures === undefined ? {} : { failures: audit.failures }),
    };
    records.push(record);
    console.log(JSON.stringify(record, null, 2));
  } catch (error) {
    failures.push({ corpus: corpus.id, message: String(error?.message ?? error) });
    console.error(String(error?.message ?? error));
  }
}

const aggregate = {
  ok: failures.length === 0,
  corpusCount: records.length,
  corpora: records,
  failures,
};

console.log("\n== aggregate ==");
console.log(JSON.stringify(aggregate, null, 2));

if (failures.length > 0) process.exit(1);

function ensureCorpus(corpus) {
  const gitDir = path.join(corpus.path, ".git");
  if (existsSync(gitDir) && !isUsableGitCheckout(corpus.path)) {
    rmSync(corpus.path, { recursive: true, force: true });
  }
  if (!existsSync(gitDir)) {
    if (existsSync(corpus.path)) {
      throw new Error(`${corpus.path} exists but is not a git checkout`);
    }
    mkdirSync(path.dirname(corpus.path), { recursive: true });
    run("git", ["init", corpus.path], root);
    run("git", ["-C", corpus.path, "remote", "add", "origin", corpus.repo], root);
  } else {
    run("git", ["-C", corpus.path, "remote", "set-url", "origin", corpus.repo], root);
  }
  run("git", ["-C", corpus.path, "fetch", "--depth=1", "origin", corpus.commit], root);
  run("git", ["-C", corpus.path, "checkout", "--detach", corpus.commit], root);
  run("git", ["-C", corpus.path, "reset", "--hard", corpus.commit], root);
  const actual = runCapture("git", ["-C", corpus.path, "rev-parse", "HEAD"], root).trim();
  if (actual !== corpus.commit) {
    throw new Error(`${corpus.id} expected ${corpus.commit}, got ${actual}`);
  }
  assertCorpusClean(corpus);
}

function isUsableGitCheckout(corpusPath) {
  const result = spawnSync("git", ["-C", corpusPath, "rev-parse", "--git-dir"], {
    cwd: root,
    encoding: "utf8",
  });
  return result.status === 0;
}

function verifyCorpus(corpus) {
  const gitDir = path.join(corpus.path, ".git");
  if (!existsSync(gitDir)) {
    throw new Error(`${corpus.id} expected pinned git checkout at ${corpus.path}; run without --skip-fetch first`);
  }
  const actual = runCapture("git", ["-C", corpus.path, "rev-parse", "HEAD"], root).trim();
  if (actual !== corpus.commit) {
    throw new Error(`${corpus.id} expected ${corpus.commit}, got ${actual}; run without --skip-fetch to refresh`);
  }
  assertCorpusClean(corpus);
}

function assertCorpusClean(corpus) {
  const status = runCapture("git", ["-C", corpus.path, "status", "--short", "--untracked-files=all"], root).trim();
  if (status.length > 0) {
    throw new Error(`${corpus.id} checkout at ${corpus.path} is dirty; clean or refresh it before auditing:\n${status}`);
  }
}

function runAudit(corpus, options) {
  const args = [
    path.join(scriptDir, "audit-cuda-lite-corpus.mjs"),
    corpus.path,
    "--limit",
    String(options.limit),
    ...(options.details ? ["--details"] : []),
    ...(options.includeSources ? ["--sources"] : []),
    ...expectationArgs(corpus.expectations),
  ];
  const result = spawnSync(process.execPath, args, {
    cwd: root,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (result.stderr) process.stderr.write(result.stderr);
  const audit = parseAuditResult(result.stdout, corpus.id);
  if (result.status !== 0) {
    throw new Error(`${corpus.id} audit failed expectation checks`);
  }
  return audit;
}

function parseAuditResult(stdout, id) {
  const jsonStart = stdout.indexOf("{");
  const jsonEnd = stdout.lastIndexOf("}");
  if (jsonStart < 0 || jsonEnd < jsonStart) {
    throw new Error(`${id} audit did not emit JSON summary`);
  }
  return JSON.parse(stdout.slice(jsonStart, jsonEnd + 1));
}

function expectationArgs(expectations) {
  const args = [];
  if (expectations.total !== undefined) args.push("--expect-total", String(expectations.total));
  if (expectations.okMin !== undefined) args.push("--expect-ok-min", String(expectations.okMin));
  if (expectations.compileCodegenMin !== undefined) args.push("--expect-compile-codegen-min", String(expectations.compileCodegenMin));
  if (expectations.planCompiledMin !== undefined) args.push("--expect-plan-compiled-min", String(expectations.planCompiledMin));
  if (expectations.webgpuMin !== undefined) args.push("--expect-plan-compiled-min", String(expectations.webgpuMin));
  if (expectations.referenceOnlyMax !== undefined) args.push("--expect-reference-only-max", String(expectations.referenceOnlyMax));
  if (expectations.hardFailMax !== undefined) args.push("--expect-hard-fail-max", String(expectations.hardFailMax));
  return args;
}

function parseArgs(args) {
  const only = new Set();
  let skipFetch = false;
  let limit = 0;
  let details = false;
  let includeSources = false;
  for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === "--only") {
      const value = args[++index];
      if (!value) throw new Error("--only expects a corpus id");
      only.add(value);
      continue;
    }
    if (arg?.startsWith("--only=")) {
      only.add(arg.slice("--only=".length));
      continue;
    }
    if (arg === "--skip-fetch") {
      skipFetch = true;
      continue;
    }
    if (arg === "--details" || arg === "--json") {
      details = true;
      continue;
    }
    if (arg === "--sources") {
      details = true;
      includeSources = true;
      continue;
    }
    if (arg === "--limit") {
      limit = parseLimit(args[++index]);
      continue;
    }
    if (arg?.startsWith("--limit=")) {
      limit = parseLimit(arg.slice("--limit=".length));
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      console.log(`usage: node scripts/audit-real-world-cuda-corpora.mjs [--only ${cudaLiteCorpora.map((corpus) => corpus.id).join("|")}] [--skip-fetch] [--limit N] [--details] [--sources]`);
      process.exit(0);
    }
    throw new Error(`unexpected argument: ${arg}`);
  }
  return { only, skipFetch, limit, details, includeSources };
}

function parseLimit(raw) {
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0) throw new Error("--limit expects a non-negative integer");
  return value;
}

function run(command, args, cwd) {
  const result = spawnSync(command, args, { cwd, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} exited ${result.status}`);
  }
}

function runCapture(command, args, cwd) {
  const result = spawnSync(command, args, { cwd, encoding: "utf8" });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} exited ${result.status}: ${result.stderr}`);
  }
  return result.stdout;
}
