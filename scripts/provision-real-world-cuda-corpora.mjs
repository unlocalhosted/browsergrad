#!/usr/bin/env node
import path from "node:path";
import {
  provisionCudaLiteCorpusCheckout,
  verifyCudaLiteCorpusCheckout,
} from "./cuda-lite-corpus-provisioning.mjs";
import { cudaLiteCorpora } from "./cuda-lite-corpus-registry.mjs";

const options = parseArgs(process.argv.slice(2));
const unknown = [...options.only].filter((id) =>
  !cudaLiteCorpora.some((corpus) => corpus.id === id));
if (unknown.length > 0) {
  console.error(`unknown CUDA-lite corpus id(s): ${unknown.join(", ")}`);
  process.exit(2);
}
const selected = options.only.size === 0
  ? cudaLiteCorpora
  : cudaLiteCorpora.filter((corpus) => options.only.has(corpus.id));
if (selected.length === 0) {
  console.error(`no matching corpora for --only (${[...options.only].join(", ")})`);
  process.exit(2);
}

const admissions = [];
const failures = [];
for (const corpus of selected) {
  try {
    const input = { root: path.dirname(corpus.path), corpus };
    const admission = options.skipFetch
      ? verifyCudaLiteCorpusCheckout(input)
      : provisionCudaLiteCorpusCheckout(input);
    admissions.push(admission);
    console.log(JSON.stringify(admission, null, 2));
  } catch (error) {
    const failure = Object.freeze({
      corpus: corpus.id,
      message: String(error?.message ?? error),
    });
    failures.push(failure);
    console.error(failure.message);
  }
}

const aggregate = Object.freeze({
  kind: "browsergrad-cuda-lite-corpus-provisioning-summary",
  version: 1,
  ok: failures.length === 0,
  admissionCount: admissions.length,
  admissions,
  failures,
  corpusAuditExecuted: false,
  browserExecutionObserved: false,
  webgpuExecutionObserved: false,
  productionConformanceAuthorityMinted: false,
  releaseReady: false,
});
console.log(JSON.stringify(aggregate, null, 2));
if (failures.length > 0) process.exit(1);

function parseArgs(args) {
  const only = new Set();
  let skipFetch = false;
  for (let index = 0; index < args.length; index += 1) {
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
    if (arg === "--help" || arg === "-h") {
      console.log(
        `usage: node scripts/provision-real-world-cuda-corpora.mjs ` +
          `[--only ${cudaLiteCorpora.map((corpus) => corpus.id).join("|")}] [--skip-fetch]`,
      );
      process.exit(0);
    }
    throw new Error(`unexpected argument: ${arg}`);
  }
  return { only, skipFetch };
}
