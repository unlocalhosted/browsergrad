import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import {
  corpusById,
  cudaLiteCorpusExecutionFixtures,
} from "./cuda-lite-corpus-registry.mjs";
import { inferDynamicSharedMemory } from "./audit-cuda-lite-corpus-device.mjs";
import {
  syntheticInputForCompiled,
  syntheticLaunchForCompiled,
} from "./cuda-lite-synthetic-input.mjs";

export function loadCorpusExecutionSources(root) {
  const out = {};
  for (const fixture of cudaLiteCorpusExecutionFixtures) {
    const source = loadNormalizedCorpusKernelSource(root, fixture);
    if (source) out[fixture.sourceKey] = source;
  }
  return out;
}

export function loadAutoCorpusSmokeFixtures(root, limit, compiler, options = {}) {
  const verifyMode = options.verifyMode ?? "reference";
  const allowedRequiredFeatures = new Set(options.allowedRequiredFeatures ?? []);
  const explicit = new Set(cudaLiteCorpusExecutionFixtures.map(corpusFixtureKey));
  const candidatesByCorpus = [];
  for (const corpus of Object.values(corpusesById())) {
    const manifest = loadCorpusKernelManifest(root, corpus, true);
    candidatesByCorpus.push({
      corpus,
      candidates: manifest.kernels
        .filter((item) => item.planCompiledOk)
        .filter((item) => !explicit.has(corpusFixtureKey({ corpusId: corpus.id, relativePath: item.file, kernelName: item.kernelName }))),
    });
  }
  const selected = [];
  let cursor = 0;
  while (selected.length < limit && candidatesByCorpus.some((entry) => entry.candidates.length > 0)) {
    const entry = candidatesByCorpus[cursor % candidatesByCorpus.length];
    cursor++;
    const item = entry.candidates.shift();
    if (!item) continue;
    const fixture = createAutoCorpusSmokeFixture(root, entry.corpus, item, compiler, { verifyMode, allowedRequiredFeatures });
    if (fixture) selected.push(fixture);
  }
  return selected;
}

export function verifyCorpusFixtureCheckouts(root) {
  const seen = new Set();
  for (const fixture of cudaLiteCorpusExecutionFixtures) {
    if (seen.has(fixture.corpusId)) continue;
    seen.add(fixture.corpusId);
    const corpus = corpusById(fixture.corpusId);
    const gitDir = path.join(corpus.path, ".git");
    if (!fs.existsSync(gitDir)) {
      throw new Error(`${corpus.id} expected pinned git checkout at ${corpus.path}`);
    }
    const result = spawnSync("git", ["-C", corpus.path, "rev-parse", "HEAD"], {
      cwd: root,
      encoding: "utf8",
    });
    if (result.error) throw result.error;
    if (result.status !== 0) {
      throw new Error(`${corpus.id} could not verify pinned checkout: ${result.stderr}`);
    }
    const actual = result.stdout.trim();
    if (actual !== corpus.commit) {
      throw new Error(`${corpus.id} expected ${corpus.commit}, got ${actual}`);
    }
    const status = spawnSync("git", ["-C", corpus.path, "status", "--short", "--untracked-files=all"], {
      cwd: root,
      encoding: "utf8",
    });
    if (status.error) throw status.error;
    if (status.status !== 0) {
      throw new Error(`${corpus.id} could not verify clean checkout: ${status.stderr}`);
    }
    const dirty = status.stdout.trim();
    if (dirty.length > 0) {
      throw new Error(`${corpus.id} checkout at ${corpus.path} is dirty; clean or refresh it before browser e2e:\n${dirty}`);
    }
  }
}

function createAutoCorpusSmokeFixture(root, corpus, item, compiler, fixtureOptions) {
  const { verifyMode, allowedRequiredFeatures } = fixtureOptions;
  const sourceKey = `autoCorpusSmoke_${safeIdentifier(corpus.id)}_${safeIdentifier(item.file)}_${safeIdentifier(item.kernelName)}_${item.block}_${item.kernel}`;
  const caseName = `auto-corpus:${corpus.id}:${item.file}:${item.kernelName}:${item.block}:${item.kernel}`;
  const source = item.source ?? loadNormalizedCorpusKernelSource(root, {
    corpusId: corpus.id,
    relativePath: item.file,
    kernelName: item.kernelName,
    caseName,
  });
  const dynamicSharedMemory = inferDynamicSharedMemory(source);
  const workgroupSize = inferAutoCorpusWorkgroupSize(source);
  const compileOptions = {
    kernelName: item.kernelName,
    workgroupSize,
    features: { "shader-f16": true, subgroups: true },
    f64Mode: "f32",
    ...(verifyMode === "dispatch" ? { f16Mode: "f32", subgroupMode: "scalar" } : {}),
    ...(Object.keys(dynamicSharedMemory).length === 0 ? {} : { dynamicSharedMemory }),
  };
  try {
    const compiled = compiler.compileCudaLiteKernelForWebGpu(source, compileOptions);
    if (compiled.ir.requiredFeatures.some((feature) => !allowedRequiredFeatures.has(feature))) return undefined;
    const input = syntheticInputForCompiled(compiled);
    if (verifyMode === "reference" && Object.keys(input.buffers).length === 0) return undefined;
    const launch = syntheticLaunchForCompiled(compiled);
    const plan = compiler.createCudaWebGpuExecutionPlan(compiled, input, launch, {
      compileKernel: (childSource, childOptions = {}) => compiler.compileCudaLiteKernelForWebGpu(childSource, {
        ...childOptions,
        features: { "shader-f16": true, subgroups: true, ...(childOptions.features ?? {}) },
        f64Mode: childOptions.f64Mode ?? "f32",
        f16Mode: childOptions.f16Mode ?? compileOptions.f16Mode,
        dynamicSharedMemory: childOptions.dynamicSharedMemory ?? inferDynamicSharedMemory(childSource),
      }),
    });
    if (!plan.supported) return undefined;
    if (verifyMode === "reference") {
      compiler.runCompiledKernelReference(compiled, input, launch);
    }
    return {
      sourceKey,
      caseName,
      corpusId: corpus.id,
      relativePath: item.file,
      kernelName: item.kernelName,
      workgroupSize: compiled.ir.workgroupSize,
      launch,
      options: compileOptions,
      requiredFeatures: compiled.ir.requiredFeatures,
      source,
      verifyMode,
    };
  } catch {
    return undefined;
  }
}

export function inferAutoCorpusWorkgroupSize(source) {
  const launchBounds = /\b__launch_bounds__\s*\(\s*([^,)]+)/u.exec(source)?.[1];
  const threads = launchBounds ? evaluateIntegerExpression(launchBounds) : undefined;
  return threads && threads > 0 && threads <= 256 ? [threads, 1, 1] : [32, 1, 1];
}

function evaluateIntegerExpression(expression) {
  if (!/^[\d\s()+*/%-]+$/u.test(expression)) return undefined;
  try {
    const value = Function(`"use strict"; return (${expression});`)();
    return Number.isInteger(value) ? value : undefined;
  } catch {
    return undefined;
  }
}

function loadCorpusKernelManifest(root, corpus, includeSources = false) {
  const result = spawnSync(process.execPath, [
    path.join(root, "scripts/audit-cuda-lite-corpus.mjs"),
    corpus.path,
    includeSources ? "--kernel-manifest-sources" : "--kernel-manifest",
    "--limit",
    "0",
  ], {
    cwd: root,
    encoding: "utf8",
    maxBuffer: 256 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`could not load corpus kernel manifest for ${corpus.id}: ${result.stderr || result.stdout}`);
  }
  return JSON.parse(result.stdout);
}

function loadNormalizedCorpusKernelSource(root, fixture) {
  const corpus = corpusById(fixture.corpusId);
  const result = spawnSync(process.execPath, [
    path.join(root, "scripts/audit-cuda-lite-corpus.mjs"),
    corpus.path,
    "--emit-kernel-source",
    fixture.relativePath,
    "--kernel-name",
    fixture.kernelName,
  ], {
    cwd: root,
    encoding: "utf8",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`could not load normalized corpus fixture ${fixture.caseName}: ${result.stderr || result.stdout}`);
  }
  return JSON.parse(result.stdout).source;
}

function corpusFixtureKey(fixture) {
  return `${fixture.corpusId}:${fixture.relativePath}:${fixture.kernelName}`;
}

function corpusesById() {
  const out = {};
  for (const fixture of cudaLiteCorpusExecutionFixtures) {
    out[fixture.corpusId] = corpusById(fixture.corpusId);
  }
  return out;
}

function safeIdentifier(value) {
  return String(value).replace(/[^A-Za-z0-9_]+/g, "_").replace(/^_+|_+$/g, "");
}
