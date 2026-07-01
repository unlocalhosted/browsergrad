import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
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

export function corpusExecutionFixturesForCaseFilters(caseFilters = []) {
  if (caseFilters.length === 0) return cudaLiteCorpusExecutionFixtures;
  return cudaLiteCorpusExecutionFixtures.filter((fixture) =>
    caseFilters.some((filter) => fixture.caseName === filter || fixture.caseName.includes(filter)));
}

export function loadCorpusExecutionSources(root, fixtures = cudaLiteCorpusExecutionFixtures) {
  const out = {};
  for (const fixture of fixtures) {
    const source = loadNormalizedCorpusKernelSource(root, fixture);
    if (source) out[fixture.sourceKey] = source;
  }
  return out;
}

export function loadAutoCorpusSmokeFixtures(root, limit, compiler, options = {}) {
  const verifyMode = options.verifyMode ?? "reference";
  const profile = options.profile ?? "full";
  const referencePreflight = options.referencePreflight ?? true;
  const allowedRequiredFeatures = new Set(options.allowedRequiredFeatures ?? []);
  const cachePath = options.cache === true
    ? autoCorpusSmokeCachePath(root, {
        limit,
        verifyMode,
        profile,
        allowedRequiredFeatures,
        inputHash: autoCorpusSmokeCacheInputHash(root),
      })
    : undefined;
  if (cachePath && fs.existsSync(cachePath)) {
    return JSON.parse(fs.readFileSync(cachePath, "utf8"));
  }
  const explicit = new Set(cudaLiteCorpusExecutionFixtures.map(corpusFixtureKey));
  const candidatesByCorpus = [];
  for (const corpus of Object.values(corpusesById())) {
    const manifest = loadCorpusKernelManifest(root, corpus, true);
    candidatesByCorpus.push({
      corpus,
      candidates: manifest.kernels
        .filter((item) => item.planCompiledOk)
        .filter((item) => profile !== "fast" || isFastAutoCorpusCandidate(corpus, item))
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
    const fixture = createAutoCorpusSmokeFixture(root, entry.corpus, item, compiler, { verifyMode, allowedRequiredFeatures, referencePreflight });
    if (fixture) selected.push(fixture);
  }
  if (cachePath) {
    fs.mkdirSync(path.dirname(cachePath), { recursive: true });
    fs.writeFileSync(cachePath, JSON.stringify(selected));
  }
  return selected;
}

export function autoCorpusSmokeCachePath(root, options) {
  const features = [...options.allowedRequiredFeatures].sort().join("-");
  const key = [
    "v4",
    `limit-${options.limit}`,
    `mode-${options.verifyMode}`,
    `profile-${options.profile}`,
    `features-${features || "none"}`,
    `sig-${options.inputHash ?? "unknown"}`,
  ].join("__").replace(/[^A-Za-z0-9_.-]+/gu, "_");
  return path.join(root, "node_modules/.cache/browsergrad/cuda-lite-auto-corpus", `${key}.json`);
}

export function autoCorpusSmokeCacheInputHash(root) {
  const hash = crypto.createHash("sha256");
  for (const line of autoCorpusSmokeCacheInputs(root)) hash.update(`${line}\n`);
  return hash.digest("hex").slice(0, 16);
}

function autoCorpusSmokeCacheInputs(root) {
  const out = [];
  out.push(...hashTreeInputs(root, "packages/browsergrad-compiler/src", /\.ts$/u));
  for (const file of [
    "packages/browsergrad-compiler/package.json",
    "packages/browsergrad-compiler/tsconfig.build.json",
    "packages/browsergrad-compiler/tsconfig.json",
    "scripts/audit-cuda-lite-corpus.mjs",
    "scripts/audit-cuda-lite-corpus-device.mjs",
    "scripts/cuda-lite-corpus-registry.mjs",
    "scripts/cuda-lite-synthetic-input.mjs",
    "scripts/cuda-lite-webgpu-fixtures.mjs",
  ]) {
    out.push(hashFileInput(root, file));
  }
  for (const corpus of Object.values(corpusesById()).sort((left, right) => left.id.localeCompare(right.id))) {
    out.push(`corpus:${corpus.id}:${corpus.commit}`);
  }
  for (const fixture of cudaLiteCorpusExecutionFixtures) {
    out.push(`fixture:${corpusFixtureKey(fixture)}:${fixture.caseName}`);
  }
  return out;
}

function hashTreeInputs(root, relativeDir, filePattern) {
  const dir = path.join(root, relativeDir);
  if (!fs.existsSync(dir)) return [`missing:${relativeDir}`];
  const files = listFiles(dir)
    .map((filePath) => path.relative(root, filePath).split(path.sep).join("/"))
    .filter((relativePath) => filePattern.test(relativePath))
    .sort();
  return files.map((relativePath) => hashFileInput(root, relativePath));
}

function listFiles(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...listFiles(fullPath));
    else if (entry.isFile()) out.push(fullPath);
  }
  return out;
}

function hashFileInput(root, relativePath) {
  const filePath = path.join(root, relativePath);
  if (!fs.existsSync(filePath)) return `missing:${relativePath}`;
  const hash = crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex").slice(0, 16);
  return `file:${relativePath}:${hash}`;
}

function isFastAutoCorpusCandidate(corpus, item) {
  const key = `${corpus.id}:${item.file}:${item.kernelName}`.toLowerCase();
  return !/(?:sgemm|gemm|matmul|matrixmul|histogram|sort|scan|reduce|reduction|dct|eigen|bisect|encoder|transpose|convolution|attention|wmma|mma|fft|montecarlo|pricing|denois|particle|spmv|segment|lineofsight|quicksort|bitonic)/u.test(key);
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
  const { verifyMode, allowedRequiredFeatures, referencePreflight = true } = fixtureOptions;
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
        dynamicSharedMemory: nonEmptyDynamicSharedMemory(childOptions.dynamicSharedMemory) ?? inferDynamicSharedMemory(childSource),
      }),
    });
    if (!plan.supported) return undefined;
    if (!planFitsPortableWebGpuLimits(plan)) return undefined;
    if (verifyMode === "reference" && referencePreflight) {
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
      ...autoCorpusComparisonFor(corpus, item),
    };
  } catch {
    return undefined;
  }
}

function nonEmptyDynamicSharedMemory(value) {
  return value && Object.keys(value).length > 0 ? value : undefined;
}

function planFitsPortableWebGpuLimits(plan) {
  for (const step of plan.steps ?? []) {
    const bindings = step.program?.bindings ?? [];
    const storageCount = bindings.filter((binding) => binding.kind === "storage").length;
    const uniformCount = bindings.filter((binding) => binding.kind === "uniform").length;
    const textureCount = bindings.filter((binding) => binding.kind === "texture2d").length;
    if (storageCount > 10 || uniformCount > 12 || textureCount > 16) return false;
    if (estimateWgslWorkgroupStorageBytes(step.program?.wgsl ?? "") > 32768) return false;
  }
  return true;
}

function estimateWgslWorkgroupStorageBytes(wgsl) {
  let total = 0;
  for (const line of wgsl.split("\n")) {
    const match = /^\s*var<workgroup>\s+\w+\s*:\s*([^;]+);/u.exec(line);
    if (!match) continue;
    total += wgslTypeByteSize(match[1] ?? "");
  }
  return total;
}

function wgslTypeByteSize(type) {
  const compact = type.replace(/\s+/gu, "");
  if (compact.startsWith("array<")) {
    const parts = splitWgslGenericArgs(compact.slice("array<".length, -1));
    const element = parts[0] ?? "";
    const count = Number(parts[1] ?? "0");
    return Number.isFinite(count) && count > 0 ? wgslTypeByteSize(element) * count : 0;
  }
  if (compact.startsWith("atomic<")) {
    return wgslTypeByteSize(compact.slice("atomic<".length, -1));
  }
  const vector = /^vec([234])<(.+)>$/u.exec(compact);
  if (vector) return Number(vector[1]) * wgslTypeByteSize(vector[2] ?? "");
  if (compact === "f16") return 2;
  if (compact === "f32" || compact === "i32" || compact === "u32" || compact === "bool") return 4;
  return 0;
}

function splitWgslGenericArgs(args) {
  const out = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < args.length; i++) {
    const char = args[i];
    if (char === "<") depth++;
    if (char === ">") depth--;
    if (char === "," && depth === 0) {
      out.push(args.slice(start, i));
      start = i + 1;
    }
  }
  out.push(args.slice(start));
  return out.map((item) => item.trim()).filter(Boolean);
}

function autoCorpusComparisonFor(corpus, item) {
  if (
    corpus.id === "cuda-samples" &&
    item.file === "cpp/3_CUDA_Features/newdelete/newdelete.cu" &&
    (item.kernelName === "placementNew" || item.kernelName === "complexVector")
  ) {
    return { comparison: { kind: "newdelete-stack-pop" } };
  }
  return {};
}

export function inferAutoCorpusWorkgroupSize(source) {
  const launchBounds = /\b__launch_bounds__\s*\(\s*([^,)]+)/u.exec(source)?.[1];
  const threads = launchBounds ? evaluateIntegerExpression(launchBounds) : undefined;
  if (threads && threads > 0 && threads <= 256) return [threads, 1, 1];
  const templateThreads = /\b(?:NUM_THREADS|kBlockSize|BLOCK_SIZE|block_size)\s*=\s*([0-9]+)/u.exec(source)?.[1];
  const templateValue = templateThreads ? Number(templateThreads) : undefined;
  if (templateValue && templateValue > 0 && templateValue <= 256) return [templateValue, 1, 1];
  const hardcodedBlockStride = /\bblockIdx\.x\s*\*\s*([0-9]+)\s*\+\s*threadIdx\.x/u.exec(source)?.[1];
  const strideValue = hardcodedBlockStride ? Number(hardcodedBlockStride) : undefined;
  if (strideValue && strideValue > 0 && strideValue <= 256) return [strideValue, 1, 1];
  return [32, 1, 1];
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
