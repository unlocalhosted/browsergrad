#!/usr/bin/env node
import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");
const corpusRoot = process.env.BROWSERGRAD_CUDA_CORPUS_ROOT ??
  "/tmp/browsergrad-corpora";

const corpora = [
  {
    id: "cuda-120",
    name: "CUDA-120-DAYS--CHALLENGE",
    repo: "https://github.com/AdepojuJeremy/CUDA-120-DAYS--CHALLENGE.git",
    commit: "fd2987a2b1a4e506629ae9beb22ee6434da2d414",
    path: "/tmp/CUDA-120-DAYS--CHALLENGE",
    expectations: {
      total: 240,
      okMin: 225,
      webgpuMin: 240,
      referenceOnlyMax: 0,
      hardFailMax: 0,
    },
  },
  {
    id: "cuda-samples",
    name: "NVIDIA/cuda-samples",
    repo: "https://github.com/NVIDIA/cuda-samples.git",
    commit: "b7c5481c556c3fe98db060207ecaa41a4b9a9abc",
    path: path.join(corpusRoot, "cuda-samples"),
    expectations: {
      total: 357,
      webgpuMin: 296,
      hardFailMax: 60,
    },
  },
  {
    id: "llm.c",
    name: "karpathy/llm.c",
    repo: "https://github.com/karpathy/llm.c.git",
    commit: "f1e2ace651495b74ae22d45d1723443fd00ecd3a",
    path: path.join(corpusRoot, "llm.c"),
    expectations: {
      total: 148,
      webgpuMin: 148,
      hardFailMax: 0,
    },
  },
  {
    id: "leetcuda",
    name: "xlite-dev/LeetCUDA",
    repo: "https://github.com/xlite-dev/LeetCUDA.git",
    commit: "c5dde9a653d077d71445bcbf822d4bf13672a69e",
    path: path.join(corpusRoot, "LeetCUDA"),
    expectations: {
      total: 293,
      webgpuMin: 275,
      hardFailMax: 18,
    },
  },
];

const options = parseArgs(process.argv.slice(2));
const selected = options.only.size === 0
  ? corpora
  : corpora.filter((corpus) => options.only.has(corpus.id));
if (selected.length === 0) {
  console.error(`no matching corpora for --only (${[...options.only].join(", ")})`);
  process.exit(2);
}

const records = [];
const failures = [];

for (const corpus of selected) {
  console.log(`\n== ${corpus.name} @ ${corpus.commit.slice(0, 7)} ==`);
  try {
    if (!options.skipFetch) ensureCorpus(corpus);
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
  if (expectations.webgpuMin !== undefined) args.push("--expect-webgpu-min", String(expectations.webgpuMin));
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
      console.log(`usage: node scripts/audit-real-world-cuda-corpora.mjs [--only ${corpora.map((corpus) => corpus.id).join("|")}] [--skip-fetch] [--limit N] [--details] [--sources]`);
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
