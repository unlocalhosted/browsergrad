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
import { setAuditHelpers } from "./audit-cuda-lite-corpus-context.mjs";
import * as auditCli from "./audit-cuda-lite-corpus-cli.mjs";
import * as auditDevice from "./audit-cuda-lite-corpus-device.mjs";
import * as auditScan from "./audit-cuda-lite-corpus-scan.mjs";
import { syntheticInputForCompiled } from "./cuda-lite-synthetic-input.mjs";

const { parseArgs, checkExpectations } = auditCli;
const { listFiles, findRepoRoot, markdownBlocks, extractKernelDefinitions, createCorpusContext, sourceWithoutCudaFunctionBodies, stripComments } = auditScan;
const { collectPortableDeviceFunctions, collectDynamicLaunchTargetDeviceFunctions, collectConstantDeclarations, collectDeviceGlobalDeclarations, collectFunctionPointerTables, collectFunctionDefines, collectTextureDeclarations, collectPodRecordDeclarations, collectTranslationUnitSharedDeclarations, inferDynamicSharedMemory, countBy, mergeCarriedDefines, mergeDefineMaps, recordDeclarationName, escapeRegExp } = auditDevice;
const auditHelpers = { ...auditCli, ...auditDevice, ...auditScan, collectCudaLiteContextDefines, collectKernelTemplateArguments, createKernelCompilationUnit, kernelDefinitionName, pruneCudaPreprocessorBranches };
setAuditHelpers(auditHelpers);
const { corpusPathArg, details, emitKernelSource, expectations, firstFailureLimit, help, includeSources, kernelManifest, kernelManifestSources } = parseArgs(process.argv.slice(2));
if (help) {
  console.log("usage: node scripts/audit-cuda-lite-corpus.mjs <corpus-path> [--limit N] [--details] [--sources] [--kernel-manifest] [--kernel-manifest-sources] [--emit-kernel-source FILE --kernel-name NAME] [--expect-total N] [--expect-compile-codegen-min N] [--expect-hard-fail-max N]");
  process.exit(0);
}
if (!corpusPathArg) {
  console.error("usage: node scripts/audit-cuda-lite-corpus.mjs <corpus-path> [--limit N] [--details] [--sources] [--kernel-manifest] [--kernel-manifest-sources] [--emit-kernel-source FILE --kernel-name NAME] [--expect-total N] [--expect-compile-codegen-min N] [--expect-hard-fail-max N]");
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
  "double",
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
  "warpReduceSum",
  "warpReduceMax",
  "warpReduceMin",
  "warp_reduce_sum",
  "warp_reduce_max",
  "warp_reduce_min",
  "warp_reduce_sum_f32",
  "warp_reduce_max_f32",
  "warp_reduce_sum_f16",
  "warp_reduce_sum_f16_f16",
  "warp_reduce_sum_f16_f32",
  "warp_reduce_sum_i8_i32",
  "warp_reduce_sum_i32_i32",
  "atomicAdd",
  "atomicSub",
  "atomicMin",
  "atomicMax",
  "atomicAnd",
  "atomicOr",
  "atomicXor",
  "atomicExch",
  "atomicCAS",
]);

const files = listFiles(corpusRoot)
  .filter((file) => /\.(?:md|markdown|cu|cuh|cpp|cc|cxx|h|hpp)$/i.test(file))
  .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
const corpusContext = createCorpusContext(corpusRoot, files);

if (emitKernelSource !== undefined) {
  const source = emitKernelCompilationSource(corpusRoot, corpusContext, emitKernelSource.file, emitKernelSource.kernelName);
  console.log(JSON.stringify({ source }, null, 2));
  process.exit(0);
}

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
    if (CUDA_HINT_RE.test(block.code)) cudaBlocks++;
    const kernels = extractKernelDefinitions(block.code);
    if (kernels.length === 0) {
      carriedDefines = mergeCarriedDefines(carriedDefines, collectCudaLiteContextDefines(block.code));
      continue;
    }
    const directContext = createAuditBlockContext(directIncludeContext, block.code, carriedDefines, corpusContext.globalDefines);
    const reverseContext = reverseIncludeContext.trim().length > 0
      ? createAuditBlockContext(`${directIncludeContext}\n${reverseIncludeContext}`, block.code, carriedDefines, corpusContext.globalDefines)
      : undefined;
    for (const [kernelIndex, rawKernel] of kernels.entries()) {
      const kernelName = kernelDefinitionName(rawKernel);
      const directAttempt = compileKernelFromAuditContext(rawKernel, kernels, kernelName, directContext);
      if (directAttempt.ok) {
        results.push({
          file,
          block: blockIndex + 1,
          kernel: kernelIndex + 1,
          kernelName,
          ok: true,
          directLoweringOk: true,
          compileCodegenOk: true,
          ...((includeSources || kernelManifestSources) ? { source: directAttempt.source } : {}),
        });
        continue;
      }
      if (reverseContext !== undefined && shouldRetryWithReverseContext(directAttempt.error)) {
        const reverseAttempt = compileKernelFromAuditContext(rawKernel, kernels, kernelName, reverseContext);
        if (reverseAttempt.ok) {
          results.push({
            file,
            block: blockIndex + 1,
            kernel: kernelIndex + 1,
            kernelName,
            ok: true,
            directLoweringOk: true,
            compileCodegenOk: true,
            ...((includeSources || kernelManifestSources) ? { source: reverseAttempt.source } : {}),
          });
          continue;
        }
      }
      const fallback = classifyReferenceFallback(directAttempt.source, kernelName);
      const diagnostic = directAttempt.error?.diagnostics?.[0];
      const feature = diagnostic
        ? describeCudaDiagnostic(diagnostic)
        : undefined;
      const error = directAttempt.error;
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
        ...((includeSources || kernelManifestSources) ? { source: directAttempt.source } : {}),
        referenceOk: fallback.referenceOk,
        webGpuPlanLiftOk: fallback.webGpuPlanLiftOk,
        webGpuPlanLiftKind: fallback.webGpuPlanLiftKind,
        webGpuPlanLiftBlocker: fallback.webGpuPlanLiftBlocker,
        webGpuPlanLiftBlockerKind: fallback.webGpuPlanLiftBlockerKind,
        webGpuPlanLiftBlockerCode: fallback.webGpuPlanLiftBlockerCode,
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

function createAuditBlockContext(includeContext, blockCode, carriedDefines, corpusGlobalDefines = new Map()) {
  const source = `${includeContext}\n${blockCode}`;
  const declarationContext = `${sourceWithoutCudaFunctionBodies(includeContext)}\n${sourceWithoutCudaFunctionBodies(blockCode)}`;
  const blockDefines = collectCudaLiteContextDefines(declarationContext);
  const referencedCorpusDefines = reachableCorpusDefines(corpusGlobalDefines, source);
  const effectiveDefines = mergeDefineMaps(CUDA_SYSTEM_DEFINES, referencedCorpusDefines, carriedDefines, blockDefines);
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
    functionPointerTables: collectFunctionPointerTables(source, effectiveDefines),
    templateArguments: collectKernelTemplateArguments(source),
  };
}

function reachableCorpusDefines(defines, source) {
  if (defines.size === 0) return defines;
  const clean = stripComments(source);
  const out = new Map();
  const pending = [];
  for (const name of defines.keys()) {
    if (mentionsIdentifier(clean, name)) pending.push(name);
  }
  while (pending.length > 0) {
    const name = pending.pop();
    if (name === undefined || out.has(name)) continue;
    const value = defines.get(name);
    if (value === undefined) continue;
    out.set(name, value);
    for (const next of defines.keys()) {
      if (!out.has(next) && mentionsIdentifier(String(value), next)) pending.push(next);
    }
  }
  return out;
}

function mentionsIdentifier(source, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\b`, "u").test(source);
}

function compileKernelFromAuditContext(rawKernel, kernels, kernelName, context) {
  const source = sourceFromAuditContext(rawKernel, kernels, kernelName, context);
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

function sourceFromAuditContext(rawKernel, kernels, kernelName, context) {
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
    functionPointerTables: context.functionPointerTables,
  });
  return source;
}

function emitKernelCompilationSource(corpusRoot, corpusContext, relativePath, kernelName) {
  const absolute = path.join(corpusRoot, relativePath);
  if (!fs.existsSync(absolute)) {
    throw new Error(`kernel source file not found: ${relativePath}`);
  }
  const text = corpusContext.read(absolute);
  const directIncludeContext = corpusContext.directSources(absolute).join("\n");
  const reverseIncludeContext = corpusContext.reverseSources(absolute).join("\n");
  const blocks = markdownBlocks(text, relativePath);
  let carriedDefines = new Map();
  for (const block of blocks) {
    if (isNonKernelCodeBlock(block)) continue;
    const directContext = createAuditBlockContext(directIncludeContext, block.code, carriedDefines, corpusContext.globalDefines);
    const reverseContext = reverseIncludeContext.trim().length > 0
      ? createAuditBlockContext(`${directIncludeContext}\n${reverseIncludeContext}`, block.code, carriedDefines, corpusContext.globalDefines)
      : undefined;
    const kernels = extractKernelDefinitions(block.code);
    const rawKernel = kernels.find((candidate) => kernelDefinitionName(candidate) === kernelName);
    if (rawKernel !== undefined) {
      const directAttempt = compileKernelFromAuditContext(rawKernel, kernels, kernelName, directContext);
      if (directAttempt.ok || reverseContext === undefined || !shouldRetryWithReverseContext(directAttempt.error)) {
        return directAttempt.source;
      }
      const reverseAttempt = compileKernelFromAuditContext(rawKernel, kernels, kernelName, reverseContext);
      return reverseAttempt.source;
    }
    carriedDefines = mergeCarriedDefines(carriedDefines, directContext.blockDefines);
  }
  throw new Error(`kernel ${kernelName} not found in ${relativePath}`);
}

function shouldRetryWithReverseContext(error) {
  const diagnostic = error?.diagnostics?.[0];
  if (diagnostic === undefined) return false;
  if (diagnostic.code === "unknown-symbol") return true;
  if (diagnostic.code === "unsupported-call" && /unsupported CUDA-lite call/u.test(diagnostic.message ?? "")) return true;
  if (diagnostic.code !== "parse-error") return false;
  return /unsupported CUDA-lite type|unknown CUDA-lite symbol|expected type|expected ';'|expected expression|array size must be an integer constant expression/u.test(diagnostic.message ?? "");
}

const failures = results.filter((result) => !result.ok);
const directLoweringOk = results.length - failures.length;
const hostPlanCompiledOk = failures.filter((failure) => failure.webGpuPlanLiftOk).length;
const compileCodegenOk = directLoweringOk + hostPlanCompiledOk;
const compileCodegenGaps = results.length - compileCodegenOk;
const compileFeatureProfile = {
  shaderF16: "assumed-available",
  subgroups: "assumed-available",
  f64Mode: "f32-compatibility",
  devicePortability: "fixture-backed-browser-e2e-only",
};
const summary = {
  files: files.length,
  codeBlocks,
  cudaBlocks,
  totalKernelDefinitions: results.length,
  corpusKernelExecution: "compile-codegen-only",
  corpusExecutionMode: "compile-codegen-only",
  compileFeatureProfile,
  executionTierCounts: {
    planCompiledOk: compileCodegenOk,
    planCompileGaps: compileCodegenGaps,
    compileCodegenOnlyOk: compileCodegenOk,
    fixtureBackedExecutedOk: 0,
    browserWebGpuExecutedOk: 0,
    outputVerifiedOk: 0,
  },
  executionTierNotes: {
    planCompiledOk: "Parsed, analyzed, lowered, and emitted direct WGSL or a host-orchestrated WebGPU plan under compileFeatureProfile assumptions.",
    planCompileGaps: "Extracted kernels that did not compile/codegen into direct WGSL or a host-orchestrated WebGPU plan.",
    compileCodegenOnlyOk: "Parsed, analyzed, lowered, and emitted WGSL or host WebGPU plan from pinned corpus source under compileFeatureProfile assumptions.",
    fixtureBackedExecutedOk: "Requires explicit input/output fixtures; this corpus audit does not synthesize them.",
    browserWebGpuExecutedOk: "Covered by separate browser E2E gates, not by this corpus audit.",
    outputVerifiedOk: "Covered only when fixture-backed execution compares readback against expected outputs.",
  },
  deprecatedCompilePlanAliases: {
    webGpuRunnableOk: "planCompiledOk",
    webGpuTotalOk: "planCompiledOk",
    webGpuCompiledOk: "planCompiledOk",
  },
  fixtureBackedExecutionOk: 0,
  browserExecutedOk: 0,
  outputVerifiedOk: 0,
  planCompiledOk: compileCodegenOk,
  planCompileGaps: compileCodegenGaps,
  webGpuDirectCompiledOk: directLoweringOk,
  webGpuHostPlanCompiledOk: hostPlanCompiledOk,
  singleDispatchPlanCompiledOk: directLoweringOk,
  hostOrchestratedPlanCompiledOk: hostPlanCompiledOk,
  compileCodegenOk,
  compileCodegenGaps,
  compileCodegenOnlyOk: compileCodegenOk,
  compileCodegenOnlyGaps: compileCodegenGaps,
  directLoweringOk,
  strictCompileGaps: failures.length,
  ok: directLoweringOk,
  hostDynamicLaunchPlanCompiledOk: failures.filter((failure) => failure.webGpuPlanLiftKind === "host-dynamic-launch").length,
  fail: failures.length,
  referenceFallbackOk: failures.filter((failure) => failure.referenceOk).length,
  referenceOnlyOk: failures.filter((failure) => failure.referenceOk && !failure.webGpuPlanLiftOk).length,
  hardFail: failures.filter((failure) => !failure.referenceOk).length,
  errors: countBy(failures, (failure) => failure.error),
  families: countBy(failures, (failure) => failure.family),
  lowering: countBy(failures, (failure) => failure.lowering),
  webGpuPlanLiftBlockers: countBy(
    failures.filter((failure) => failure.referenceOk && !failure.webGpuPlanLiftOk),
    (failure) => failure.webGpuPlanLiftBlockerCode ?? failure.webGpuPlanLiftBlocker ?? "unknown",
  ),
};

if (details) {
  console.log(JSON.stringify({ summary, failures }, null, 2));
} else if (kernelManifest) {
  console.log(JSON.stringify({ summary, kernels: kernelManifestRows(results, kernelManifestSources) }, null, 2));
} else {
  console.log(JSON.stringify(summary, null, 2));
}
if (!details && !kernelManifest && failures.length > 0 && firstFailureLimit > 0) {
  console.log("\nfirst direct-lowering gaps:");
  for (const failure of failures.slice(0, firstFailureLimit)) {
    const lift = failure.webGpuPlanLiftOk ? ` [webgpu-plan-lift:${failure.webGpuPlanLiftKind}]` : "";
    const reference = failure.referenceOk ? " [reference-ok]" : "";
    const blocker = failure.referenceOk && !failure.webGpuPlanLiftOk && failure.webGpuPlanLiftBlocker
      ? ` [webgpu-plan-blocker:${failure.webGpuPlanLiftBlockerCode ?? failure.webGpuPlanLiftBlocker}]`
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
      f64Mode: "f32",
      workgroupSize: [256, 1, 1],
      dynamicSharedMemory: inferDynamicSharedMemory(source),
    });
    const lift = webGpuLiftFor(compiled);
    return {
      referenceOk: true,
      webGpuPlanLiftOk: lift.kind !== undefined,
      webGpuPlanLiftKind: lift.kind,
      webGpuPlanLiftBlocker: lift.blocker,
      webGpuPlanLiftBlockerKind: lift.blockerKind,
      webGpuPlanLiftBlockerCode: lift.blockerCode,
    };
  } catch (error) {
    return {
      referenceOk: false,
      webGpuPlanLiftOk: false,
      webGpuPlanLiftKind: undefined,
      webGpuPlanLiftBlocker: String(error?.message ?? error).split("\n")[0],
      webGpuPlanLiftBlockerKind: undefined,
      webGpuPlanLiftBlockerCode: undefined,
    };
  }
}

function kernelManifestRows(items, includeSources = false) {
  return items.map((item) => {
    const planCompiledOk = item.ok || item.webGpuPlanLiftOk === true;
    return {
      file: item.file,
      block: item.block,
      kernel: item.kernel,
      kernelName: item.kernelName,
      directLoweringOk: item.ok === true,
      planCompiledOk,
      compileCodegenOk: planCompiledOk,
      ...(item.webGpuPlanLiftKind === undefined ? {} : { webGpuPlanLiftKind: item.webGpuPlanLiftKind }),
      ...(item.webGpuPlanLiftBlockerCode === undefined ? {} : { webGpuPlanLiftBlockerCode: item.webGpuPlanLiftBlockerCode }),
      ...(includeSources && item.source !== undefined ? { source: item.source } : {}),
    };
  });
}

function webGpuLiftFor(compiled) {
  const runtimePlan = createCudaRuntimePlan(compiled);
  const executionPlan = createCudaWebGpuExecutionPlan(
    compiled,
    syntheticInputForCompiled(compiled),
    { gridDim: [1, 1, 1], blockDim: compiled.ir.workgroupSize },
    {
      compileKernel: (childSource, options = {}) => compileCudaLiteKernelForWebGpu(childSource, {
        ...options,
        features: { "shader-f16": true, subgroups: true, ...options.features },
        f64Mode: options.f64Mode ?? "f32",
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
  if (
    executionPlan.kind === "single-dispatch" &&
    runtimePlan.operations.length > 0
  ) {
    return { kind: "host-pruned-runtime", blocker: undefined };
  }
  if (executionPlan.kind === "single-dispatch") return { kind: undefined, blocker: "no runtime WebGPU lift required" };
  return { kind: executionPlan.kind, blocker: undefined };
}
