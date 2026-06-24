#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  collectCudaLiteContextDefines,
  collectKernelTemplateArguments,
  createKernelCompilationUnit,
  kernelDefinitionName,
  pruneCudaPreprocessorBranches,
} from "./cuda-lite-source-normalizer.mjs";

const { corpusPathArg, details, expectations, firstFailureLimit, help, includeSources } = parseArgs(process.argv.slice(2));
if (help) {
  console.log("usage: node scripts/audit-cuda-lite-corpus.mjs <corpus-path> [--limit N] [--details] [--sources] [--expect-total N] [--expect-webgpu-min N] [--expect-hard-fail-max N]");
  process.exit(0);
}
if (!corpusPathArg) {
  console.error("usage: node scripts/audit-cuda-lite-corpus.mjs <corpus-path> [--limit N] [--details] [--sources] [--expect-total N] [--expect-webgpu-min N] [--expect-hard-fail-max N]");
  process.exit(2);
}

const corpusRoot = path.resolve(corpusPathArg);
const repoRoot = findRepoRoot(process.cwd());
const compilerUrl = pathToFileURL(path.join(repoRoot, "packages/browsergrad-compiler/dist/index.js")).href;
const {
  compileCudaLiteKernelForWebGpu,
  compileCudaLiteKernel,
  createCudaRuntimePlan,
  createCudaWebGpuExecutionPlan,
  describeCudaDiagnostic,
} = await import(compilerUrl);
const CUDA_HINT_RE = /__global__|cuda[A-Z]|<<<|threadIdx|blockIdx|__shared__/;
const NON_CODE_BLOCK_LANG_RE = /^(?:mermaid|flowchart|graphviz|dot|plantuml|text|txt)$/iu;
const CUDA_SYSTEM_DEFINES = new Map([
  ["UINT_MAX", "0xffffffffu"],
  ["INT_MAX", "2147483647"],
  ["INT_MIN", "(-2147483647 - 1)"],
  ["CUDART_PI_F", "3.141592654f"],
  ["CUDART_2PI_F", "6.283185307f"],
  ["CUDART_PIO2_F", "1.570796327f"],
  ["CUDART_PIO4_F", "0.785398163f"],
]);
const PORTABLE_POINTER_BASE_TYPES = new Set([
  "float",
  "int",
  "uint",
  "half",
  "half2",
  "bf16",
  "bool",
  "float2",
  "float3",
  "float4",
  "int2",
  "int3",
  "int4",
  "uint2",
  "uint3",
  "uint4",
]);
const SEMANTIC_BUILTIN_DEVICE_HELPERS = new Set([
  "min",
  "max",
  "__ldcs",
  "__stcs",
  "__ldg",
  "__stcg",
  "__usad4",
  "vec_at",
  "load128",
  "load128cs",
  "store128",
  "store128cs",
  "store128cg",
  "div_ceil",
  "blockReduce",
]);

const files = listFiles(corpusRoot)
  .filter((file) => /\.(?:md|markdown|cu|cuh|cpp|cc|cxx|h|hpp)$/i.test(file))
  .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
const corpusContext = createCorpusContext(corpusRoot, files);

const results = [];
let codeBlocks = 0;
let cudaBlocks = 0;

for (const file of files) {
  const absolute = path.join(corpusRoot, file);
  const text = corpusContext.read(absolute);
  const directIncludeContext = corpusContext.directSources(absolute).join("\n");
  const reverseIncludeContext = corpusContext.reverseSources(absolute).join("\n");
  const blocks = markdownBlocks(text, file);
  let carriedDefines = new Map();
  codeBlocks += blocks.length;
  for (const [blockIndex, block] of blocks.entries()) {
    if (isNonKernelCodeBlock(block)) continue;
    const directContext = createAuditBlockContext(directIncludeContext, block.code, carriedDefines);
    const reverseContext = reverseIncludeContext.trim().length > 0
      ? createAuditBlockContext(`${directIncludeContext}\n${reverseIncludeContext}`, block.code, carriedDefines)
      : undefined;
    if (CUDA_HINT_RE.test(block.code)) cudaBlocks++;
    const kernels = extractKernelDefinitions(block.code);
    for (const [kernelIndex, rawKernel] of kernels.entries()) {
      const kernelName = kernelDefinitionName(rawKernel);
      const directAttempt = compileKernelFromAuditContext(rawKernel, kernels, kernelName, directContext);
      if (directAttempt.ok) {
        results.push({ file, block: blockIndex + 1, kernel: kernelIndex + 1, ok: true });
        continue;
      }
      if (reverseContext !== undefined && shouldRetryWithReverseContext(directAttempt.error)) {
        const reverseAttempt = compileKernelFromAuditContext(rawKernel, kernels, kernelName, reverseContext);
        if (reverseAttempt.ok) {
          results.push({ file, block: blockIndex + 1, kernel: kernelIndex + 1, ok: true });
          continue;
        }
      }
      const fallback = classifyReferenceFallback(directAttempt.source, kernelName);
      const diagnostic = directAttempt.error?.diagnostics?.[0];
        const feature = diagnostic
          ? describeCudaDiagnostic(diagnostic)
          : undefined;
        results.push({
          file,
          block: blockIndex + 1,
          kernel: kernelIndex + 1,
          kernelName,
          ok: false,
          error: diagnostic?.code ?? error?.name ?? "error",
          family: feature?.family ?? "unknown",
          feature: feature?.label ?? "Unknown compatibility gap",
          lowering: feature?.lowering ?? "unsupported",
          message: diagnostic?.message ?? String(error?.message ?? error).split("\n")[0],
          ...(includeSources ? { source: directAttempt.source } : {}),
          referenceOk: fallback.referenceOk,
          webGpuLiftOk: fallback.webGpuLiftOk,
          webGpuLiftKind: fallback.webGpuLiftKind,
          webGpuLiftBlocker: fallback.webGpuLiftBlocker,
          webGpuLiftBlockerKind: fallback.webGpuLiftBlockerKind,
          webGpuLiftBlockerCode: fallback.webGpuLiftBlockerCode,
        });
    }
    carriedDefines = mergeCarriedDefines(carriedDefines, directContext.blockDefines);
  }
}

function isNonKernelCodeBlock(block) {
  if (NON_CODE_BLOCK_LANG_RE.test(block.lang)) return true;
  if (/^\s*(?:\/\/\s*)?Pseudocode solution\b/iu.test(block.code)) return true;
  return /^\s*(?:flowchart|graph)\s+(?:LR|RL|TB|TD|BT)\b/iu.test(block.code);
}

function createAuditBlockContext(includeContext, blockCode, carriedDefines) {
  const source = `${includeContext}\n${blockCode}`;
  const declarationContext = `${sourceWithoutCudaFunctionBodies(includeContext)}\n${sourceWithoutCudaFunctionBodies(blockCode)}`;
  const blockDefines = collectCudaLiteContextDefines(declarationContext);
  const effectiveDefines = mergeDefineMaps(CUDA_SYSTEM_DEFINES, carriedDefines, blockDefines);
  const recordDeclarations = collectPodRecordDeclarations(source);
  const recordNames = new Set(recordDeclarations.map(recordDeclarationName).filter((name) => name !== undefined));
  return {
    blockDefines,
    effectiveDefines,
    functionDeclarations: collectFunctionDefines(source),
    deviceFunctions: collectPortableDeviceFunctions(source, recordNames, effectiveDefines),
    dynamicLaunchTargets: collectDynamicLaunchTargetDeviceFunctions(source),
    constantDeclarations: collectConstantDeclarations(source),
    deviceGlobalDeclarations: collectDeviceGlobalDeclarations(source),
    textureDeclarations: collectTextureDeclarations(source),
    recordDeclarations,
    sharedDeclarations: collectTranslationUnitSharedDeclarations(declarationContext),
    templateArguments: collectKernelTemplateArguments(source),
  };
}

function compileKernelFromAuditContext(rawKernel, kernels, kernelName, context) {
  const kernel = pruneCudaPreprocessorBranches(rawKernel, context.effectiveDefines);
  const siblingKernels = [
    ...kernels.filter((candidate) => candidate !== rawKernel)
      .map((candidate) => pruneCudaPreprocessorBranches(candidate, context.effectiveDefines)),
    ...context.dynamicLaunchTargets,
  ];
  const source = createKernelCompilationUnit({
    kernel,
    siblingKernels,
    definesByName: context.effectiveDefines,
    templateArgumentsByKernelName: context.templateArguments,
    functionDeclarations: context.functionDeclarations,
    deviceFunctions: context.deviceFunctions,
    constantDeclarations: context.constantDeclarations,
    deviceGlobalDeclarations: context.deviceGlobalDeclarations,
    textureDeclarations: context.textureDeclarations,
    recordDeclarations: context.recordDeclarations,
    sharedDeclarations: context.sharedDeclarations,
  });
  try {
    compileCudaLiteKernel(source, {
      kernelName,
      features: { "shader-f16": true, subgroups: true },
      f64Mode: "f32",
      workgroupSize: [256, 1, 1],
      dynamicSharedMemory: inferDynamicSharedMemory(source),
    });
    return { ok: true, source };
  } catch (error) {
    return { ok: false, source, error };
  }
}

function shouldRetryWithReverseContext(error) {
  const diagnostic = error?.diagnostics?.[0];
  if (diagnostic === undefined) return false;
  if (diagnostic.code === "unknown-symbol") return true;
  if (diagnostic.code === "unsupported-call" && /unsupported CUDA-lite call/u.test(diagnostic.message ?? "")) return true;
  if (diagnostic.code !== "parse-error") return false;
  return /unsupported CUDA-lite type|unknown CUDA-lite symbol|expected type|expected ';'|expected expression/u.test(diagnostic.message ?? "");
}

const failures = results.filter((result) => !result.ok);
const directLoweringOk = results.length - failures.length;
const webGpuLiftedOk = failures.filter((failure) => failure.webGpuLiftOk).length;
const webGpuRunnableOk = directLoweringOk + webGpuLiftedOk;
const summary = {
  files: files.length,
  codeBlocks,
  cudaBlocks,
  totalKernelDefinitions: results.length,
  corpusKernelExecution: "compile-only",
  fixtureBackedExecutionOk: 0,
  webGpuDirectCompiledOk: directLoweringOk,
  webGpuHostPlanCompiledOk: webGpuLiftedOk,
  webGpuCompiledOk: webGpuRunnableOk,
  directLoweringOk,
  strictCompileGaps: failures.length,
  webGpuRunnableOk,
  webGpuHostOrchestratedOk: webGpuLiftedOk,
  ok: directLoweringOk,
  webGpuSingleDispatchOk: directLoweringOk,
  webGpuLiftedOk,
  hostDynamicLiftableOk: failures.filter((failure) => failure.webGpuLiftKind === "host-dynamic-launch").length,
  webGpuTotalOk: webGpuRunnableOk,
  fail: failures.length,
  referenceFallbackOk: failures.filter((failure) => failure.referenceOk).length,
  referenceOnlyOk: failures.filter((failure) => failure.referenceOk && !failure.webGpuLiftOk).length,
  hardFail: failures.filter((failure) => !failure.referenceOk).length,
  errors: countBy(failures, (failure) => failure.error),
  families: countBy(failures, (failure) => failure.family),
  lowering: countBy(failures, (failure) => failure.lowering),
  webGpuLiftBlockers: countBy(
    failures.filter((failure) => failure.referenceOk && !failure.webGpuLiftOk),
    (failure) => failure.webGpuLiftBlockerCode ?? failure.webGpuLiftBlocker ?? "unknown",
  ),
};

if (details) {
  console.log(JSON.stringify({ summary, failures }, null, 2));
} else {
  console.log(JSON.stringify(summary, null, 2));
}
if (!details && failures.length > 0 && firstFailureLimit > 0) {
  console.log("\nfirst direct-lowering gaps:");
  for (const failure of failures.slice(0, firstFailureLimit)) {
    const lift = failure.webGpuLiftOk ? ` [webgpu-lift:${failure.webGpuLiftKind}]` : "";
    const reference = failure.referenceOk ? " [reference-ok]" : "";
    const blocker = failure.referenceOk && !failure.webGpuLiftOk && failure.webGpuLiftBlocker
      ? ` [webgpu-blocker:${failure.webGpuLiftBlockerCode ?? failure.webGpuLiftBlocker}]`
      : "";
    console.log(`${failure.file} block ${failure.block} kernel ${failure.kernel}: ${failure.family}/${failure.error}${lift}${reference}${blocker}: ${failure.message}`);
  }
}

const expectationFailures = checkExpectations(summary, expectations);
if (expectationFailures.length > 0) {
  console.error("\ncoverage expectations failed:");
  for (const failure of expectationFailures) console.error(`- ${failure}`);
  process.exit(1);
}

function classifyReferenceFallback(source, kernelName) {
  try {
    const compiled = compileCudaLiteKernelForWebGpu(source, {
      kernelName,
      features: { "shader-f16": true, subgroups: true },
      workgroupSize: [256, 1, 1],
      dynamicSharedMemory: inferDynamicSharedMemory(source),
    });
    const lift = webGpuLiftFor(compiled);
    return {
      referenceOk: true,
      webGpuLiftOk: lift.kind !== undefined,
      webGpuLiftKind: lift.kind,
      webGpuLiftBlocker: lift.blocker,
      webGpuLiftBlockerKind: lift.blockerKind,
      webGpuLiftBlockerCode: lift.blockerCode,
    };
  } catch (error) {
    return {
      referenceOk: false,
      webGpuLiftOk: false,
      webGpuLiftKind: undefined,
      webGpuLiftBlocker: String(error?.message ?? error).split("\n")[0],
      webGpuLiftBlockerKind: undefined,
      webGpuLiftBlockerCode: undefined,
    };
  }
}

function webGpuLiftFor(compiled) {
  const runtimePlan = createCudaRuntimePlan(compiled);
  const executionPlan = createCudaWebGpuExecutionPlan(
    compiled,
    syntheticInputFor(compiled),
    { gridDim: [1, 1, 1], blockDim: compiled.ir.workgroupSize },
    {
      compileKernel: (childSource, options = {}) => compileCudaLiteKernelForWebGpu(childSource, {
        ...options,
        features: { "shader-f16": true, subgroups: true, ...options.features },
        dynamicSharedMemory: inferDynamicSharedMemory(childSource),
      }),
    },
  );
  if (!executionPlan.supported) {
    const firstBlocker = executionPlan.blockers?.[0];
    return {
      kind: undefined,
      blocker: firstBlocker?.message ?? executionPlan.reason,
      blockerKind: firstBlocker?.kind,
      blockerCode: firstBlocker?.code,
    };
  }
  if (
    executionPlan.kind === "single-dispatch" &&
    runtimePlan.operations.length > 0 &&
    runtimePlan.operations.every((operation) => operation.kind === "device-sync")
  ) {
    return { kind: "device-sync-noop", blocker: undefined };
  }
  if (executionPlan.kind === "single-dispatch") return { kind: undefined, blocker: "no runtime WebGPU lift required" };
  return { kind: executionPlan.kind, blocker: undefined };
}

function syntheticInputFor(compiled) {
  const scalars = {};
  const buffers = {};
  const constants = {};
  const deviceGlobals = {};
  const memoryPools = {};
  const textures = {};
  for (const param of compiled.ir.params) {
    if (param.pointer) {
      if (param.valueType === "devicepool") {
        memoryPools[param.name] = { data: new Uint32Array(4096), offset: new Uint32Array([0]) };
      } else {
        buffers[param.name] = syntheticBufferForType(param.valueType);
      }
    } else if (param.valueType === "texture2d") {
      textures[param.name] = { width: 64, height: 64, data: new Float32Array(64 * 64) };
    } else {
      scalars[param.name] = syntheticScalarForName(param.name);
    }
  }
  for (const constant of compiled.ir.constants) {
    constants[constant.name] = constant.dimensions.length === 0
      ? syntheticScalarForName(constant.name)
      : syntheticBufferForType(constant.valueType);
  }
  for (const global of compiled.ir.deviceGlobals) {
    const length = global.dimensions.length === 0
      ? 1
      : global.dimensions.reduce((product, dimension) => product * dimension, 1);
    deviceGlobals[global.name] = syntheticBufferForType(global.valueType, length);
  }
  for (const texture of compiled.ir.textures) {
    textures[texture.name] = { width: 64, height: 64, data: new Float32Array(64 * 64) };
  }
  for (const poolName of externalDevicePoolNamesFromSource(compiled.ast.source)) {
    memoryPools[poolName] ??= { data: new Uint32Array(4096), offset: new Uint32Array([0]) };
  }
  return { buffers, scalars, constants, deviceGlobals, memoryPools, textures };
}

function externalDevicePoolNamesFromSource(source) {
  return [...source.matchAll(/\b(?:deviceAllocate|streamOrderedAllocate)\s*\(\s*&\s*([A-Za-z_][A-Za-z0-9_]*)/g)]
    .map((match) => match[1])
    .filter(Boolean);
}

function syntheticBufferForType(type, length = 4096) {
  if (type === "int") return new Int32Array(length);
  if (type === "uint" || type === "voidptr" || type === "bool") return new Uint32Array(length);
  return new Float32Array(length);
}

function syntheticScalarForName(name) {
  if (/^(?:n|N|num|count|length|totalLen|frontierSize|numSamples|totalThreads|poolSize)$/u.test(name)) return 1024;
  if (/^(?:threads|threadsPerBlock|blockSize)$/u.test(name)) return 256;
  if (/^(?:blocks|blocksPerGrid|numBlocks)$/u.test(name)) return 4;
  return 1;
}

function listFiles(root, prefix = "") {
  const entries = fs.readdirSync(path.join(root, prefix), { withFileTypes: true });
  const out = [];
  for (const entry of entries) {
    if (entry.name === ".git" || entry.name === "node_modules") continue;
    const relative = path.join(prefix, entry.name);
    if (entry.isDirectory()) out.push(...listFiles(root, relative));
    else if (entry.isFile()) out.push(relative);
  }
  return out;
}

function findRepoRoot(start) {
  let current = path.resolve(start);
  while (true) {
    if (fs.existsSync(path.join(current, "pnpm-workspace.yaml"))) return current;
    const parent = path.dirname(current);
    if (parent === current) return path.resolve(start);
    current = parent;
  }
}

function markdownBlocks(text, file) {
  if (!/\.(?:md|markdown)$/i.test(file)) return [{ lang: path.extname(file).slice(1), code: text }];
  const blocks = [];
  const re = /```([^\n]*)\n([\s\S]*?)```/g;
  let match;
  while ((match = re.exec(text))) {
    blocks.push({ lang: match[1].trim(), code: match[2] });
  }
  return blocks;
}

function stripComments(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/.*$/gm, "");
}

function extractKernelDefinitions(source) {
  const clean = stripComments(source);
  return collectCudaFunctionBodies(clean, "__global__")
    .filter((kernel) => !isPlaceholderKernel(kernel));
}

function collectCudaFunctionBodies(source, marker) {
  let index = 0;
  const bodies = [];
  while (true) {
    const start = source.indexOf(marker, index);
    if (start < 0) break;
    const brace = source.indexOf("{", start);
    const semicolon = source.indexOf(";", start);
    if (brace < 0) break;
    if (semicolon >= 0 && semicolon < brace) {
      index = semicolon + 1;
      continue;
    }
    let depth = 0;
    let end = -1;
    for (let cursor = brace; cursor < source.length; cursor++) {
      if (source[cursor] === "{") depth++;
      else if (source[cursor] === "}") {
        depth--;
        if (depth === 0) {
          end = cursor + 1;
          break;
        }
      }
    }
    if (end < 0) break;
    bodies.push(source.slice(withCudaDeclarationPrefixStart(source, start), end));
    index = end;
  }
  return bodies;
}

function withTemplatePrefixStart(source, start) {
  const prefix = source.slice(0, start);
  const match = /template\s*<[^;{}]*>\s*$/u.exec(prefix);
  return match ? start - match[0].length : start;
}

function withCudaDeclarationPrefixStart(source, start) {
  let cursor = start;
  while (true) {
    const beforeWhitespace = skipBackwardWhitespace(source, cursor);
    const match = /(?:static|inline|__inline__|__forceinline__|__host__)\s*$/u.exec(source.slice(0, beforeWhitespace));
    if (match === null) break;
    const candidate = beforeWhitespace - match[0].trimEnd().length;
    if (candidate < 0 || !isIdentifierBoundary(source[candidate - 1])) break;
    cursor = candidate;
  }
  return withTemplatePrefixStart(source, cursor);
}

function skipBackwardWhitespace(source, index) {
  let cursor = index;
  while (cursor > 0 && /\s/u.test(source[cursor - 1])) cursor--;
  return cursor;
}

function isIdentifierBoundary(char) {
  return char === undefined || !/[A-Za-z0-9_]/u.test(char);
}

function sourceWithoutCudaFunctionBodies(source) {
  const clean = expandCudaQualifierMacros(stripComments(source));
  let out = "";
  let index = 0;
  while (true) {
    const globalStart = clean.indexOf("__global__", index);
    const deviceStart = clean.indexOf("__device__", index);
    const starts = [globalStart, deviceStart].filter((item) => item >= 0);
    const start = starts.length === 0 ? -1 : Math.min(...starts);
    if (start < 0) {
      out += clean.slice(index);
      return out;
    }
    const brace = clean.indexOf("{", start);
    const semicolon = clean.indexOf(";", start);
    if (brace < 0 || (semicolon >= 0 && semicolon < brace)) {
      out += clean.slice(index, semicolon >= 0 ? semicolon + 1 : clean.length);
      index = semicolon >= 0 ? semicolon + 1 : clean.length;
      continue;
    }
    const declarationStart = withCudaDeclarationPrefixStart(clean, start);
    out += clean.slice(index, declarationStart);
    let depth = 0;
    let end = -1;
    for (let cursor = brace; cursor < clean.length; cursor++) {
      if (clean[cursor] === "{") depth++;
      else if (clean[cursor] === "}") {
        depth--;
        if (depth === 0) {
          end = cursor + 1;
          break;
        }
      }
    }
    if (end < 0) return out;
    index = end;
  }
}

function expandCudaQualifierMacros(source) {
  const defines = collectCudaLiteContextDefines(source);
  const replacements = [...defines]
    .filter(([name, value]) =>
      /^[A-Za-z_][A-Za-z0-9_]*$/u.test(name) &&
      /\b__(?:device|host|forceinline)__\b|\b(?:inline|__inline__)\b/u.test(value) &&
      /^[A-Za-z0-9_\s]+$/u.test(value))
    .sort((a, b) => b[0].length - a[0].length);
  if (replacements.length === 0) return source;
  return source
    .split(/\r?\n/u)
    .map((line) => {
      if (/^\s*#/u.test(line)) return line;
      let out = line;
      for (const [name, value] of replacements) {
        out = out.replace(new RegExp(`\\b${escapeRegExp(name)}\\b`, "gu"), value);
      }
      return out;
    })
    .join("\n");
}

function createCorpusContext(root, files) {
  const sourceCache = new Map();
  const includeGraph = new Map();
  const reverseIncludeGraph = new Map();
  const read = (absoluteFile) => {
    const resolved = path.resolve(absoluteFile);
    const cached = sourceCache.get(resolved);
    if (cached !== undefined) return cached;
    let source = "";
    try {
      source = fs.readFileSync(resolved, "utf8");
    } catch {
      // Optional headers outside small educational corpora are ignored.
    }
    sourceCache.set(resolved, source);
    return source;
  };
  const includesFor = (absoluteFile) => {
    const resolvedFile = path.resolve(absoluteFile);
    const cached = includeGraph.get(resolvedFile);
    if (cached !== undefined) return cached;
    const source = read(resolvedFile);
    const includes = [];
    for (const includeName of localIncludeNames(source)) {
      const resolved = resolveLocalInclude(resolvedFile, root, includeName);
      if (!resolved) continue;
      includes.push(resolved);
    }
    const unique = uniqueResolved(includes);
    includeGraph.set(resolvedFile, unique);
    return unique;
  };
  for (const file of files) {
    const absoluteFile = path.resolve(root, file);
    const source = read(absoluteFile);
    for (const includeName of localIncludeNames(source)) {
      const resolved = resolveLocalInclude(absoluteFile, root, includeName);
      if (!resolved) continue;
      const reverse = reverseIncludeGraph.get(resolved) ?? [];
      reverse.push(absoluteFile);
      reverseIncludeGraph.set(resolved, reverse);
    }
    includesFor(absoluteFile);
  }
  return {
    read,
    directSources(absoluteFile) {
      return uniqueStrings(collectTransitiveIncludeSources(path.resolve(absoluteFile), includesFor, read));
    },
    reverseSources(absoluteFile) {
      return uniqueStrings(collectReverseTranslationUnitSources(
        path.resolve(absoluteFile),
        root,
        includesFor,
        reverseIncludeGraph,
        read,
      ));
    },
  };
}

function localIncludeNames(source) {
  return [...source.matchAll(/^\s*#include\s+"([^"]+)"/gm)]
    .map((match) => match[1])
    .filter(Boolean);
}

function collectTransitiveIncludeSources(absoluteFile, includesFor, read, seen = new Set()) {
  if (seen.has(absoluteFile)) return [];
  seen.add(absoluteFile);
  const out = [];
  for (const included of includesFor(absoluteFile)) {
    if (seen.has(included)) continue;
    out.push(...collectTransitiveIncludeSources(included, includesFor, read, seen));
    out.push(read(included));
  }
  return out;
}

function collectReverseTranslationUnitSources(absoluteFile, root, includesFor, reverseIncludeGraph, read) {
  const out = [];
  const seenFiles = new Set([absoluteFile]);
  const pending = [...(reverseIncludeGraph.get(absoluteFile) ?? [])].map((file) => ({ file, target: absoluteFile }));
  while (pending.length > 0 && seenFiles.size <= 32) {
    const item = pending.shift();
    const includer = item?.file;
    const target = item?.target;
    if (includer === undefined || seenFiles.has(includer)) continue;
    if (target === undefined) continue;
    seenFiles.add(includer);
    for (const source of collectPrefixIncludeSources(includer, target, root, includesFor, read)) {
      out.push(source);
    }
    const prefix = sourcePrefixBeforeResolvedInclude(includer, target, root, read);
    if (prefix !== undefined) out.push(prefix);
    for (const next of reverseIncludeGraph.get(includer) ?? []) {
      if (!seenFiles.has(next)) pending.push({ file: next, target: includer });
    }
  }
  return out;
}

function collectPrefixIncludeSources(includer, target, root, includesFor, read) {
  const prefix = sourcePrefixBeforeResolvedInclude(includer, target, root, read);
  if (prefix === undefined) return [];
  const out = [];
  const seen = new Set([target]);
  for (const includeName of localIncludeNames(prefix)) {
    const resolved = resolveLocalInclude(includer, root, includeName);
    if (!resolved || seen.has(resolved)) continue;
    out.push(...collectTransitiveIncludeSources(resolved, includesFor, read, seen));
    out.push(read(resolved));
  }
  return out;
}

function sourcePrefixBeforeResolvedInclude(includer, target, root, read) {
  const source = read(includer);
  for (const match of source.matchAll(/^\s*#include\s+"([^"]+)"/gm)) {
    const includeName = match[1];
    if (!includeName) continue;
    const resolved = resolveLocalInclude(includer, root, includeName);
    if (resolved === target) return source.slice(0, match.index);
  }
  return undefined;
}

function uniqueResolved(files) {
  return [...new Set(files.map((file) => path.resolve(file)))];
}

function uniqueStrings(values) {
  const seen = new Set();
  const out = [];
  for (const value of values) {
    if (value.trim().length === 0 || seen.has(value)) continue;
    seen.add(value);
    out.push(value);
  }
  return out;
}

function resolveLocalInclude(absoluteFile, root, includeName) {
  const candidates = localIncludeCandidates(absoluteFile, root, includeName);
  for (const candidate of candidates) {
    const relative = path.relative(root, candidate);
    if (relative.startsWith("..") || path.isAbsolute(relative)) continue;
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return candidate;
  }
  return undefined;
}

function localIncludeCandidates(absoluteFile, root, includeName) {
  const out = [];
  const seen = new Set();
  const add = (candidate) => {
    const resolved = path.resolve(candidate);
    if (seen.has(resolved)) return;
    seen.add(resolved);
    out.push(resolved);
  };
  let dir = path.dirname(absoluteFile);
  const resolvedRoot = path.resolve(root);
  while (true) {
    add(path.join(dir, includeName));
    add(path.join(dir, "include", includeName));
    add(path.join(dir, "utils", includeName));
    if (dir === resolvedRoot) break;
    const parent = path.dirname(dir);
    if (parent === dir || path.relative(resolvedRoot, parent).startsWith("..")) break;
    dir = parent;
  }
  add(path.join(resolvedRoot, includeName));
  return out;
}

function isPlaceholderKernel(kernel) {
  const signature = kernel.slice(0, kernel.indexOf("{"));
  return /\(\s*\.\.\.\s*\)/u.test(signature) ||
    /\?\?\?/u.test(kernel) ||
    /\bsome[A-Z][A-Za-z0-9_]*\b/u.test(stripComments(kernel));
}

function collectPortableDeviceFunctions(source, recordNames = new Set(), definesByName = new Map()) {
  const clean = expandCudaQualifierMacros(stripComments(source));
  const functions = [];
  let index = 0;
  while (true) {
    const device = clean.indexOf("__device__", index);
    if (device < 0) break;
    let start = withCudaDeclarationPrefixStart(clean, device);
    const brace = clean.indexOf("{", device);
    const semicolon = clean.indexOf(";", device);
    if (brace < 0) break;
    if (semicolon >= 0 && semicolon < brace) {
      index = semicolon + 1;
      continue;
    }
    let depth = 0;
    let end = -1;
    for (let cursor = brace; cursor < clean.length; cursor++) {
      if (clean[cursor] === "{") depth++;
      else if (clean[cursor] === "}") {
        depth--;
        if (depth === 0) {
          end = cursor + 1;
          break;
        }
      }
    }
    if (end < 0) break;
    const fn = clean.slice(start, end);
    const signature = fn.slice(0, fn.indexOf("{"));
    const name = cudaFunctionDefinitionName(signature);
    if (
      name &&
      !SEMANTIC_BUILTIN_DEVICE_HELPERS.has(name) &&
      !sourceLaunchesDeviceFunction(clean, name) &&
      isPortableDeviceFunctionCandidate(signature, fn, name, recordNames, definesByName)
    ) {
      functions.push({ name, source: fn });
    }
    index = end;
  }
  return functions;
}

function collectDynamicLaunchTargetDeviceFunctions(source) {
  const clean = expandCudaQualifierMacros(stripComments(source));
  return collectCudaFunctionBodies(clean, "__device__")
    .filter((fn) => {
      const signature = fn.slice(0, fn.indexOf("{"));
      const name = cudaFunctionDefinitionName(signature);
      return name !== undefined && sourceLaunchesDeviceFunction(clean, name);
    })
    .map((fn) => fn.replace(/\b__device__\b/u, "__global__"));
}

function cudaFunctionDefinitionName(signature) {
  const open = signature.lastIndexOf("(");
  if (open < 0) return undefined;
  const before = signature.slice(0, open).trim();
  if (/\boperator\b/u.test(before)) return undefined;
  const name = /([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^<>]*>)?\s*$/u.exec(before)?.[1];
  if (name === undefined || ["if", "for", "while", "switch", "return"].includes(name)) return undefined;
  return name;
}

function isPortableDeviceFunctionCandidate(signature, source, name, recordNames = new Set(), definesByName = new Map()) {
  return isPortableScalarDeviceFunction(signature, source) ||
    isPortablePointerDeviceFunction(signature, source, name, recordNames, definesByName) ||
    isPortableReferenceDeviceFunction(signature, source, name, recordNames, definesByName);
}

function isPortableScalarDeviceFunction(signature, source) {
  if (/\*/u.test(signature)) return false;
  if (/\bdo\b|reinterpret|__float_as_int|__int_as_float/u.test(source)) return false;
  return true;
}

function isPortablePointerDeviceFunction(signature, source, name, recordNames = new Set(), definesByName = new Map()) {
  if (!/\*/u.test(signature)) return false;
  if (!hasSupportedDeviceReturnShape(signature, name, recordNames, definesByName)) return false;
  if (/\bvoid\s*\*/u.test(signature)) return false;
  const pointerBases = pointerBaseTypes(signature);
  const templateTypeParams = templateTypeParamNames(signature);
  if (pointerBases.length === 0) return false;
  if (!pointerBases.every((type) => isPortableBaseType(type, templateTypeParams, definesByName))) return false;
  if (/\bdo\b|__float_as_int|__int_as_float/u.test(source)) return false;
  return true;
}

function isPortableReferenceDeviceFunction(signature, source, name, recordNames = new Set(), definesByName = new Map()) {
  if (!/&/u.test(signature)) return false;
  if (!hasSupportedDeviceReturnShape(signature, name, recordNames, definesByName)) return false;
  const referenceBases = referenceBaseTypes(signature);
  const templateTypeParams = templateTypeParamNames(signature);
  if (referenceBases.length === 0) return false;
  if (!referenceBases.every((type) => isPortableBaseType(type, templateTypeParams, definesByName))) return false;
  if (/\bdo\b|__float_as_int|__int_as_float/u.test(source)) return false;
  return true;
}

function isPortableBaseType(type, templateTypeParams = new Set(), definesByName = new Map()) {
  const normalized = normalizePortableBaseType(type, definesByName);
  return PORTABLE_POINTER_BASE_TYPES.has(normalized) || templateTypeParams.has(normalized);
}

function templateTypeParamNames(signature) {
  const match = /^\s*template\s*<([^>]*)>/u.exec(signature);
  if (match?.[1] === undefined) return new Set();
  const names = new Set();
  for (const param of match[1].split(",")) {
    const type = /\b(?:typename|class)\s+([A-Za-z_][A-Za-z0-9_]*)/u.exec(param.trim())?.[1];
    if (type !== undefined) names.add(type);
  }
  return names;
}

function pointerBaseTypes(signature) {
  const cleaned = stripCooperativeGroupParamTypes(signature)
    .replace(/^\s*template\s*<[^>]*>\s*/u, " ")
    .replace(/\b(?:const|volatile|__restrict__|__restrict|restrict|static|inline|__forceinline__|__device__|__host__)\b/gu, " ")
    .replace(/\s+/gu, " ");
  const types = [];
  for (const match of cleaned.matchAll(/([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)?)\s*\*/gu)) {
    if (match[1] !== undefined) types.push(match[1].trim());
  }
  return types;
}

function referenceBaseTypes(signature) {
  const cleaned = stripCooperativeGroupParamTypes(signature)
    .replace(/^\s*template\s*<[^>]*>\s*/u, " ")
    .replace(/\b(?:const|volatile|__restrict__|__restrict|restrict|static|inline|__forceinline__|__device__|__host__)\b/gu, " ")
    .replace(/\s+/gu, " ");
  const types = [];
  for (const match of cleaned.matchAll(/([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)?)\s*&/gu)) {
    if (match[1] !== undefined) types.push(match[1].trim());
  }
  return types;
}

function stripCooperativeGroupParamTypes(signature) {
  return signature.replace(
    /\b(?:(?:cg|cooperative_groups)::)?(?:thread_block_tile\s*<[^>]+>|thread_block|grid_group|coalesced_group)\s*&?\s+[A-Za-z_][A-Za-z0-9_]*/gu,
    " ",
  );
}

function normalizePortableBaseType(type, definesByName = new Map(), seen = new Set()) {
  let normalized = type
    .replace(/\b(?:const|volatile|typename|class|struct)\b/gu, " ")
    .replace(/[&*]/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  const vectorCarrier = /^vec([234])\s*<\s*([A-Za-z_][A-Za-z0-9_:]*)\s*>\s*::\s*(Type|float|double|int|uint|half)$/u.exec(normalized);
  if (vectorCarrier?.[1] !== undefined && vectorCarrier[2] !== undefined && vectorCarrier[3] !== undefined) {
    const scalar = vectorCarrier[3] === "Type"
      ? normalizePortableBaseType(definesByName.get(vectorCarrier[2]) ?? vectorCarrier[2], definesByName, new Set([...seen, vectorCarrier[2]]))
      : vectorCarrier[3];
    if (scalar === "float" || scalar === "double") return `float${vectorCarrier[1]}`;
    if (scalar === "int" || scalar === "uint") return `${scalar}${vectorCarrier[1]}`;
    if (scalar === "half" && vectorCarrier[1] === "2") return "half2";
    return scalar;
  }
  const packed = /^Packed128\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>$/u.exec(normalized)?.[1];
  if (packed !== undefined) {
    const elementType = normalizePortableBaseType(definesByName.get(packed) ?? packed, definesByName, new Set([...seen, packed]));
    if (elementType === "float") return "float4";
    if (elementType === "int") return "int4";
    if (elementType === "uint") return "uint4";
    if (elementType === "half") return "half";
    if (elementType === "bf16") return "bf16";
    return normalized;
  }
  const mapped = definesByName.get(normalized);
  if (mapped && /^[A-Za-z_][A-Za-z0-9_]*$/u.test(mapped) && mapped !== normalized && !seen.has(normalized)) {
    normalized = normalizePortableBaseType(mapped, definesByName, new Set([...seen, normalized]));
  }
  if (normalized === "unsigned int" || normalized === "unsigned") return "uint";
  if (normalized === "unsigned short" || normalized === "unsigned short int" || normalized === "uint16_t") return "uint";
  if (normalized === "unsigned char" || normalized === "uchar" || normalized === "uint8_t") return "uint";
  if (normalized === "uint64_t" || normalized === "uint32_t" || normalized === "uintptr_t") return "uint";
  if (normalized === "signed int" || normalized === "signed") return "int";
  if (normalized === "signed short" || normalized === "signed short int" || normalized === "short" || normalized === "short int" || normalized === "int16_t") return "int";
  if (normalized === "signed char" || normalized === "char" || normalized === "int8_t") return "int";
  if (normalized === "int64_t" || normalized === "int32_t") return "int";
  if (normalized === "clock_t") return "uint";
  if (normalized === "size_t") return "uint";
  if (normalized === "ptrdiff_t") return "int";
  if (normalized === "__half") return "half";
  if (normalized === "__nv_bfloat16" || normalized === "nv_bfloat16") return "bf16";
  return normalized;
}

function hasSupportedDeviceReturnShape(signature, name, recordNames = new Set(), definesByName = new Map()) {
  const open = signature.lastIndexOf("(");
  if (open < 0) return false;
  const before = signature.slice(0, open).trim();
  const nameSuffix = new RegExp(`\\b${escapeRegExp(name)}\\s*(?:<[^<>]*>)?\\s*$`, "u");
  if (!nameSuffix.test(before)) return false;
  const returnType = before
    .replace(nameSuffix, "")
    .replace(/^\s*template\s*<[^>]*>\s*/u, " ")
    .replace(/\b(?:static|inline|__inline__|__forceinline__|__device__|__host__|const|volatile)\b/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  if (returnType === "void") return true;
  const normalized = normalizePortableBaseType(returnType, definesByName);
  const templateTypeParams = templateTypeParamNames(signature);
  return PORTABLE_POINTER_BASE_TYPES.has(normalized) || templateTypeParams.has(normalized) || recordNames.has(returnType);
}

function recordDeclarationName(declaration) {
  const match = /\b(?:typedef\s+)?struct(?:\s+(?:__align__|alignas)\s*\([^)]*\))*\s*([A-Za-z_][A-Za-z0-9_]*)?\s*\{[\s\S]*?\}\s*([A-Za-z_][A-Za-z0-9_]*)?\s*;/u.exec(declaration);
  return match?.[2] ?? match?.[1];
}

function sourceLaunchesDeviceFunction(source, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\s*<<\\s*<`, "u").test(source);
}

function collectObjectDefines(source) {
  const defines = new Map();
  for (const line of source.split(/\r?\n/u)) {
    const stripped = stripLineComment(line);
    const match = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)(?!\()\s+(.+?)\s*$/u.exec(stripped);
    if (match !== null) {
      const [, name, value] = match;
      if (name !== undefined && value !== undefined) defines.set(name, value.trim());
      continue;
    }
    const alias = parseSimpleTypeAlias(stripped, defines);
    if (alias !== undefined) {
      defines.set(alias.name, alias.value);
      continue;
    }
    const constant = parseSimpleIntegerConstant(stripped);
    if (constant !== undefined) defines.set(constant.name, constant.value);
  }
  return defines;
}

function parseSimpleTypeAlias(line, defines) {
  const typedefMatch = /^\s*typedef\s+(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*$/u.exec(line);
  if (typedefMatch !== null) {
    const [, sourceType, alias] = typedefMatch;
    const value = normalizeAliasType(sourceType, defines);
    return value && alias ? { name: alias, value } : undefined;
  }
  const usingMatch = /^\s*using\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*;\s*$/u.exec(line);
  if (usingMatch !== null) {
    const [, alias, sourceType] = usingMatch;
    const value = normalizeAliasType(sourceType, defines);
    return value && alias ? { name: alias, value } : undefined;
  }
  return undefined;
}

function normalizeAliasType(sourceType, defines) {
  let type = sourceType
    .replace(/\b(?:const|volatile|__restrict__|__restrict|restrict)\b/gu, "")
    .replace(/\s+/gu, " ")
    .trim();
  if (type.endsWith("*") || type.includes("(") || type.includes(")")) return undefined;
  const packed = /^Packed128\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>$/u.exec(type)?.[1];
  if (packed !== undefined) {
    const elementType = normalizePortableBaseType(defines.get(packed) ?? packed, defines);
    if (elementType === "float") return "float4";
    if (elementType === "int") return "int4";
    if (elementType === "uint") return "uint4";
    return undefined;
  }
  if (type.includes("<") || type.includes(">")) return undefined;
  if (type === "unsigned int" || type === "unsigned") return "uint";
  if (type === "unsigned char" || type === "uchar" || type === "uint8_t") return "uint";
  if (type === "signed int" || type === "signed") return "int";
  if (type === "signed char" || type === "char" || type === "int8_t") return "int";
  if (type === "clock_t") return "uint";
  if (type === "long long" || type === "long" || type === "short" || type === "short int") return "int";
  const mapped = defines.get(type);
  if (mapped && /^[A-Za-z_][A-Za-z0-9_]*$/u.test(mapped)) type = mapped;
  if (type === "uchar2") return "uint2";
  if (type === "uchar3") return "uint3";
  if (type === "uchar4") return "uint4";
  if (type === "char2") return "int2";
  if (type === "char3") return "int3";
  if (type === "char4") return "int4";
  const supported = new Set(["float", "int", "uint", "half", "__half", "bf16", "bool", "float2", "float3", "float4", "half2", "int2", "int3", "int4", "uint2", "uint3", "uint4"]);
  return supported.has(type) ? type : undefined;
}

function parseSimpleIntegerConstant(line) {
  const match = /^\s*((?:(?:static|constexpr|const)\s+)*)(?:(?:cuda\s*::\s*)?std\s*::\s*)?(?:int|uint|unsigned\s+int|size_t|ptrdiff_t|uint32_t|uint64_t|int32_t|int64_t)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z0-9_:,\s()+\-*/%<>&|^?:.]+)\s*;\s*$/u.exec(line);
  if (match === null) return undefined;
  const [, qualifiers, name, value] = match;
  if (!name || !value) return undefined;
  if (!/\b(?:const|constexpr)\b/u.test(qualifiers ?? "")) return undefined;
  return { name, value: value.trim() };
}

function mergeDefineMaps(...maps) {
  const merged = new Map();
  for (const map of maps) {
    for (const [name, value] of map) merged.set(name, value);
  }
  return merged;
}

function mergeCarriedDefines(previous, next) {
  const merged = new Map(previous);
  for (const [name, value] of next) {
    if (/^[A-Z_][A-Z0-9_]{1,}$/u.test(name)) merged.set(name, value);
  }
  return merged;
}

function collectConstantDeclarations(source) {
  const clean = stripComments(source);
  const declarations = [];
  const re = /__constant__\s+[^;]+;/g;
  let match;
  while ((match = re.exec(clean))) declarations.push(match[0]);
  return declarations;
}

function collectDeviceGlobalDeclarations(source) {
  const clean = stripComments(source);
  const declarations = [];
  const re = /(?:\bstatic\s+)?__device__\s+(?:static\s+)?[^;{}]+;/g;
  let match;
  while ((match = re.exec(clean))) {
    const declaration = match[0].trim();
    if (/\b__shared__\b/u.test(declaration)) continue;
    if (/\b__host__\b|\b__global__\b/u.test(declaration)) continue;
    if (/[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*\)\s*;$/u.test(declaration)) continue;
    declarations.push(declaration);
  }
  return declarations;
}

function collectFunctionDefines(source) {
  const out = [];
  for (const line of logicalPreprocessorLines(source)) {
    const stripped = stripLineComment(line);
    if (/^\s*#define\s+[A-Za-z_][A-Za-z0-9_]*\([^)]*\)\s+.+/u.test(stripped)) out.push(stripped.trim());
  }
  return out;
}

function logicalPreprocessorLines(source) {
  const lines = [];
  let current = "";
  for (const raw of source.split(/\r?\n/u)) {
    const line = current.length === 0 ? raw : `${current} ${raw.trimStart()}`;
    if (/\\\s*$/u.test(line)) {
      current = line.replace(/\\\s*$/u, "").trimEnd();
      continue;
    }
    lines.push(line);
    current = "";
  }
  if (current.length > 0) lines.push(current);
  return lines;
}

function collectTextureDeclarations(source) {
  const clean = stripComments(source);
  const declarations = [];
  const re = /texture\s*<[^;]+>\s*[A-Za-z_][A-Za-z0-9_]*\s*;/g;
  let match;
  while ((match = re.exec(clean))) declarations.push(match[0]);
  return declarations;
}

function collectPodRecordDeclarations(source) {
  const clean = stripComments(source);
  const declarations = [];
  const re = /\b(?:typedef\s+)?struct(?:\s+(?:__align__|alignas)\s*\([^)]*\))*\s*(?:[A-Za-z_][A-Za-z0-9_]*)?\s*\{[\s\S]*?\}\s*(?:[A-Za-z_][A-Za-z0-9_]*)?\s*;/g;
  let match;
  while ((match = re.exec(clean))) declarations.push(match[0]);
  return declarations;
}

function collectTranslationUnitSharedDeclarations(source) {
  const clean = stripComments(source);
  const declarations = [];
  const re = /\b(?:extern\s+)?__shared__\s+(?:float|int|unsigned\s+int|uint|half|__half|bool|char|unsigned\s+char|uchar)\s+[A-Za-z_][A-Za-z0-9_]*\s*\[[^\]]*]\s*;/g;
  let match;
  while ((match = re.exec(clean))) declarations.push(match[0]);
  return declarations;
}

function stripLineComment(line) {
  let escaped = false;
  let inString = false;
  for (let index = 0; index < line.length - 1; index++) {
    const char = line[index];
    if (inString) {
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === "\"") inString = false;
      continue;
    }
    if (char === "\"") {
      inString = true;
      continue;
    }
    if (char === "/" && line[index + 1] === "/") return line.slice(0, index);
  }
  return line;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function inferDynamicSharedMemory(source) {
  const out = {};
  const supportedType =
    "(?:float[234]?|double|int[234]?|uint[234]?|char[234]?|uchar[234]?|uint8_t[234]?|int8_t[234]?|unsigned\\s+int|signed\\s+int|short|unsigned\\s+short|unsigned\\s+char|char|int|uint|half|__half|bf16|__nv_bfloat16|bool)";
  const attrs = "(?:(?:__align__|alignas)\\s*\\([^)]*\\)\\s*)*";
  const re = new RegExp(`extern\\s+__shared__\\s+${attrs}${supportedType}\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\[\\s*]`, "g");
  for (const match of source.matchAll(re)) {
    out[match[1]] = 256;
  }
  return out;
}

function countBy(items, keyFn) {
  const out = {};
  for (const item of items) {
    const key = keyFn(item);
    out[key] = (out[key] ?? 0) + 1;
  }
  return out;
}

function parseArgs(args) {
  let corpusPathArg;
  let details = false;
  let includeSources = false;
  const expectations = {};
  let firstFailureLimit = 80;
  let help = false;
  for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === "--") continue;
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
      const value = Number(args[++index]);
      if (!Number.isInteger(value) || value < 0) {
        console.error("--limit expects a non-negative integer");
        process.exit(2);
      }
      firstFailureLimit = value;
      continue;
    }
    if (arg?.startsWith("--limit=")) {
      const value = Number(arg.slice("--limit=".length));
      if (!Number.isInteger(value) || value < 0) {
        console.error("--limit expects a non-negative integer");
        process.exit(2);
      }
      firstFailureLimit = value;
      continue;
    }
    const expectation = parseExpectationArg(arg, args, index);
    if (expectation) {
      expectations[expectation.key] = expectation.value;
      index = expectation.nextIndex;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      help = true;
      continue;
    }
    if (!corpusPathArg) {
      corpusPathArg = arg;
      continue;
    }
    console.error(`unexpected argument: ${arg}`);
    process.exit(2);
  }
  return { corpusPathArg, details, expectations, firstFailureLimit, help, includeSources };
}

function parseExpectationArg(arg, args, index) {
  const specs = {
    "--expect-total": "totalKernelDefinitions",
    "--expect-ok-min": "okMin",
    "--expect-single-dispatch-min": "webGpuSingleDispatchMin",
    "--expect-webgpu-lifted-min": "webGpuLiftedMin",
    "--expect-webgpu-min": "webGpuTotalMin",
    "--expect-reference-fallback-min": "referenceFallbackMin",
    "--expect-reference-only-max": "referenceOnlyMax",
    "--expect-hard-fail-max": "hardFailMax",
  };
  for (const [flag, key] of Object.entries(specs)) {
    if (arg === flag) {
      return { key, value: parseExpectationValue(flag, args[index + 1]), nextIndex: index + 1 };
    }
    if (arg?.startsWith(`${flag}=`)) {
      return { key, value: parseExpectationValue(flag, arg.slice(flag.length + 1)), nextIndex: index };
    }
  }
  return undefined;
}

function parseExpectationValue(flag, rawValue) {
  const value = Number(rawValue);
  if (!Number.isInteger(value) || value < 0) {
    console.error(`${flag} expects a non-negative integer`);
    process.exit(2);
  }
  return value;
}

function checkExpectations(summary, expectations) {
  const failures = [];
  if (expectations.totalKernelDefinitions !== undefined && summary.totalKernelDefinitions !== expectations.totalKernelDefinitions) {
    failures.push(`totalKernelDefinitions expected ${expectations.totalKernelDefinitions}, got ${summary.totalKernelDefinitions}`);
  }
  minExpectation(failures, summary.ok, expectations.okMin, "ok");
  minExpectation(failures, summary.webGpuSingleDispatchOk, expectations.webGpuSingleDispatchMin, "webGpuSingleDispatchOk");
  minExpectation(failures, summary.webGpuLiftedOk, expectations.webGpuLiftedMin, "webGpuLiftedOk");
  minExpectation(failures, summary.webGpuTotalOk, expectations.webGpuTotalMin, "webGpuTotalOk");
  minExpectation(failures, summary.referenceFallbackOk, expectations.referenceFallbackMin, "referenceFallbackOk");
  maxExpectation(failures, summary.referenceOnlyOk, expectations.referenceOnlyMax, "referenceOnlyOk");
  maxExpectation(failures, summary.hardFail, expectations.hardFailMax, "hardFail");
  return failures;
}

function minExpectation(failures, actual, expected, label) {
  if (expected !== undefined && actual < expected) failures.push(`${label} expected >= ${expected}, got ${actual}`);
}

function maxExpectation(failures, actual, expected, label) {
  if (expected !== undefined && actual > expected) failures.push(`${label} expected <= ${expected}, got ${actual}`);
}
