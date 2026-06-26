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

export function loadAutoCorpusSmokeFixtures(root, limit, compiler) {
  const explicit = new Set(cudaLiteCorpusExecutionFixtures.map(corpusFixtureKey));
  const candidatesByCorpus = [];
  for (const corpus of Object.values(corpusesById())) {
    const manifest = loadCorpusKernelManifest(root, corpus);
    candidatesByCorpus.push({
      corpus,
      candidates: manifest.kernels
        .filter((item) => item.directLoweringOk && item.planCompiledOk)
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
    const fixture = createAutoCorpusSmokeFixture(root, entry.corpus, item, compiler);
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

function createAutoCorpusSmokeFixture(root, corpus, item, compiler) {
  const sourceKey = `autoCorpusSmoke_${safeIdentifier(corpus.id)}_${safeIdentifier(item.file)}_${safeIdentifier(item.kernelName)}_${item.block}_${item.kernel}`;
  const caseName = `auto-corpus:${corpus.id}:${item.file}:${item.kernelName}:${item.block}:${item.kernel}`;
  const source = loadNormalizedCorpusKernelSource(root, {
    corpusId: corpus.id,
    relativePath: item.file,
    kernelName: item.kernelName,
    caseName,
  });
  const dynamicSharedMemory = inferDynamicSharedMemory(source);
  const options = {
    kernelName: item.kernelName,
    workgroupSize: [32, 1, 1],
    features: { "shader-f16": true, subgroups: true },
    f64Mode: "f32",
    ...(Object.keys(dynamicSharedMemory).length === 0 ? {} : { dynamicSharedMemory }),
  };
  try {
    const compiled = compiler.compileCudaLiteKernelForWebGpu(source, options);
    if (compiled.ir.requiredFeatures.includes("subgroups") || compiled.ir.requiredFeatures.includes("shader-f16")) return undefined;
    if (Object.keys(syntheticInputForCompiled(compiled).buffers).length === 0) return undefined;
    const launch = syntheticLaunchForCompiled(compiled);
    const plan = compiler.createCudaWebGpuExecutionPlan(compiled, syntheticInputForCompiled(compiled), launch, {
      compileKernel: (childSource, childOptions = {}) => compiler.compileCudaLiteKernelForWebGpu(childSource, {
        ...childOptions,
        features: { "shader-f16": true, subgroups: true, ...(childOptions.features ?? {}) },
        f64Mode: childOptions.f64Mode ?? "f32",
        dynamicSharedMemory: childOptions.dynamicSharedMemory ?? inferDynamicSharedMemory(childSource),
      }),
    });
    if (!plan.supported) return undefined;
    compiler.runCompiledKernelReference(compiled, syntheticInputForCompiled(compiled), launch);
    return {
      sourceKey,
      caseName,
      corpusId: corpus.id,
      relativePath: item.file,
      kernelName: item.kernelName,
      workgroupSize: compiled.ir.workgroupSize,
      launch,
      options,
      source,
    };
  } catch {
    return undefined;
  }
}

function loadCorpusKernelManifest(root, corpus) {
  const result = spawnSync(process.execPath, [
    path.join(root, "scripts/audit-cuda-lite-corpus.mjs"),
    corpus.path,
    "--kernel-manifest",
    "--limit",
    "0",
  ], {
    cwd: root,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
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
