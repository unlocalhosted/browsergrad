#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const { corpusPathArg, details, expectations, firstFailureLimit, help } = parseArgs(process.argv.slice(2));
if (help) {
  console.log("usage: node scripts/audit-cuda-lite-corpus.mjs <corpus-path> [--limit N] [--details] [--expect-total N] [--expect-webgpu-min N] [--expect-hard-fail-max N]");
  process.exit(0);
}
if (!corpusPathArg) {
  console.error("usage: node scripts/audit-cuda-lite-corpus.mjs <corpus-path> [--limit N] [--details] [--expect-total N] [--expect-webgpu-min N] [--expect-hard-fail-max N]");
  process.exit(2);
}

const corpusRoot = path.resolve(corpusPathArg);
const repoRoot = findRepoRoot(process.cwd());
const compilerUrl = pathToFileURL(path.join(repoRoot, "packages/browsergrad-compiler/dist/index.js")).href;
const {
  compileCudaLiteKernel,
  createCudaRuntimePlan,
  createCudaWebGpuExecutionPlan,
  describeCudaDiagnostic,
} = await import(compilerUrl);
const CUDA_HINT_RE = /__global__|cuda[A-Z]|<<<|threadIdx|blockIdx|__shared__/;
const NON_CODE_BLOCK_LANG_RE = /^(?:mermaid|flowchart|graphviz|dot|plantuml|text|txt)$/iu;

const files = listFiles(corpusRoot)
  .filter((file) => /\.(?:md|markdown|cu|cuh|cpp|cc|cxx|h|hpp)$/i.test(file))
  .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

const results = [];
let codeBlocks = 0;
let cudaBlocks = 0;

for (const file of files) {
  const absolute = path.join(corpusRoot, file);
  const text = fs.readFileSync(absolute, "utf8");
  const blocks = markdownBlocks(text, file);
  let carriedDefines = new Map();
  codeBlocks += blocks.length;
  for (const [blockIndex, block] of blocks.entries()) {
    if (isNonKernelCodeBlock(block)) continue;
    const blockDefines = collectObjectDefines(block.code);
    const blockFunctionDefines = collectFunctionDefines(block.code);
    const blockDeviceFunctions = collectScalarDeviceFunctions(block.code);
    const blockConstants = collectConstantDeclarations(block.code);
    const blockTextures = collectTextureDeclarations(block.code);
    const effectiveDefines = blockDefines.size === 0 ? carriedDefines : blockDefines;
    if (CUDA_HINT_RE.test(block.code)) cudaBlocks++;
    const kernels = extractKernelDefinitions(block.code);
    for (const [kernelIndex, rawKernel] of kernels.entries()) {
      const kernelName = kernelDefinitionName(rawKernel);
      const siblingKernels = kernels.filter((kernel) => kernel !== rawKernel);
      const source = kernelSourceWithContext(rawKernel, siblingKernels, effectiveDefines, blockFunctionDefines, blockDeviceFunctions, blockConstants, blockTextures);
      try {
        compileCudaLiteKernel(source, {
          kernelName,
          features: { "shader-f16": true, subgroups: true },
          workgroupSize: [256, 1, 1],
          dynamicSharedMemory: inferDynamicSharedMemory(source),
        });
        results.push({ file, block: blockIndex + 1, kernel: kernelIndex + 1, ok: true });
      } catch (error) {
        const fallback = classifyReferenceFallback(source, kernelName);
        const diagnostic = error?.diagnostics?.[0];
        const feature = diagnostic
          ? describeCudaDiagnostic(diagnostic)
          : undefined;
        results.push({
          file,
          block: blockIndex + 1,
          kernel: kernelIndex + 1,
          ok: false,
          error: diagnostic?.code ?? error?.name ?? "error",
          family: feature?.family ?? "unknown",
          feature: feature?.label ?? "Unknown compatibility gap",
          lowering: feature?.lowering ?? "unsupported",
          message: String(error?.message ?? error).split("\n")[0],
          referenceOk: fallback.referenceOk,
          webGpuLiftOk: fallback.webGpuLiftOk,
          webGpuLiftKind: fallback.webGpuLiftKind,
          webGpuLiftBlocker: fallback.webGpuLiftBlocker,
          webGpuLiftBlockerKind: fallback.webGpuLiftBlockerKind,
          webGpuLiftBlockerCode: fallback.webGpuLiftBlockerCode,
        });
      }
    }
    carriedDefines = mergeCarriedDefines(carriedDefines, blockDefines);
  }
}

function isNonKernelCodeBlock(block) {
  if (NON_CODE_BLOCK_LANG_RE.test(block.lang)) return true;
  if (/^\s*(?:\/\/\s*)?Pseudocode solution\b/iu.test(block.code)) return true;
  return /^\s*(?:flowchart|graph)\s+(?:LR|RL|TB|TD|BT)\b/iu.test(block.code);
}

const failures = results.filter((result) => !result.ok);
const summary = {
  files: files.length,
  codeBlocks,
  cudaBlocks,
  totalKernelDefinitions: results.length,
  ok: results.length - failures.length,
  webGpuSingleDispatchOk: results.length - failures.length,
  webGpuLiftedOk: failures.filter((failure) => failure.webGpuLiftOk).length,
  hostDynamicLiftableOk: failures.filter((failure) => failure.webGpuLiftKind === "host-dynamic-launch").length,
  webGpuTotalOk: results.length - failures.length + failures.filter((failure) => failure.webGpuLiftOk).length,
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
  console.log("\nfirst failures:");
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
    const compiled = compileCudaLiteKernel(source, {
      kernelName,
      features: { "shader-f16": true, subgroups: true },
      workgroupSize: [256, 1, 1],
      dynamicSharedMemory: inferDynamicSharedMemory(source),
      referenceDynamicParallelism: true,
      referenceGridSync: true,
      referenceCudaRuntime: true,
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
      compileKernel: (childSource, options = {}) => compileCudaLiteKernel(childSource, {
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
  const memoryPools = {};
  for (const param of compiled.ir.params) {
    if (param.pointer) {
      if (param.valueType === "devicepool") {
        memoryPools[param.name] = { data: new Uint32Array(4096), offset: new Uint32Array([0]) };
      } else {
        buffers[param.name] = syntheticBufferForType(param.valueType);
      }
    } else {
      scalars[param.name] = syntheticScalarForName(param.name);
    }
  }
  for (const constant of compiled.ir.constants) {
    constants[constant.name] = constant.dimensions.length === 0
      ? syntheticScalarForName(constant.name)
      : syntheticBufferForType(constant.valueType);
  }
  for (const poolName of externalDevicePoolNamesFromSource(compiled.ast.source)) {
    memoryPools[poolName] ??= { data: new Uint32Array(4096), offset: new Uint32Array([0]) };
  }
  return { buffers, scalars, constants, memoryPools };
}

function externalDevicePoolNamesFromSource(source) {
  return [...source.matchAll(/\b(?:deviceAllocate|streamOrderedAllocate)\s*\(\s*&\s*([A-Za-z_][A-Za-z0-9_]*)/g)]
    .map((match) => match[1])
    .filter(Boolean);
}

function syntheticBufferForType(type) {
  if (type === "int") return new Int32Array(4096);
  if (type === "uint" || type === "voidptr") return new Uint32Array(4096);
  return new Float32Array(4096);
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
  const kernels = [];
  let index = 0;
  while (true) {
    const start = clean.indexOf("__global__", index);
    if (start < 0) break;
    const brace = clean.indexOf("{", start);
    const semicolon = clean.indexOf(";", start);
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
    const kernel = clean.slice(start, end);
    if (!isPlaceholderKernel(kernel)) kernels.push(kernel);
    index = end;
  }
  return kernels;
}

function isPlaceholderKernel(kernel) {
  const signature = kernel.slice(0, kernel.indexOf("{"));
  return /\(\s*\.\.\.\s*\)/u.test(signature) ||
    /\?\?\?/u.test(kernel) ||
    /\bsome[A-Z][A-Za-z0-9_]*\b/u.test(stripComments(kernel));
}

function collectScalarDeviceFunctions(source) {
  const clean = stripComments(source);
  const functions = [];
  let index = 0;
  while (true) {
    const device = clean.indexOf("__device__", index);
    if (device < 0) break;
    let start = device;
    const before = clean.slice(Math.max(0, device - 32), device);
    const inline = /(?:__inline__|inline|__forceinline__)\s*$/u.exec(before);
    if (inline) start = device - inline[0].length;
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
    const name = /(?:float|int|uint|half|unsigned\s+int|void)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/u.exec(signature)?.[1];
    if (name && isPortableScalarDeviceFunction(signature, fn)) functions.push({ name, source: fn });
    index = end;
  }
  return functions;
}

function isPortableScalarDeviceFunction(signature, source) {
  if (/\*/u.test(signature)) return false;
  if (/\bdo\b|reinterpret|static_cast|__float_as_int|__int_as_float/u.test(source)) return false;
  return true;
}

function kernelSourceWithContext(kernel, siblingKernels, definesByName, functionDeclarations, deviceFunctions, constantDeclarations, textureDeclarations) {
  const params = new Set(kernelParamNames(kernel));
  const defines = [...definesByName]
    .filter(([name]) => !params.has(name))
    .map(([name, value]) => `#define ${name} ${value}`);
  const referencedDeviceFunctions = deviceFunctions
    .filter((fn) => new RegExp(`\\b${escapeRegExp(fn.name)}\\s*\\(`, "u").test(kernel))
    .map((fn) => fn.source);
  return `${defines.join("\n")}\n${functionDeclarations.join("\n")}\n${referencedDeviceFunctions.join("\n")}\n${constantDeclarations.join("\n")}\n${textureDeclarations.join("\n")}\n${siblingKernels.join("\n")}\n${kernel}`;
}

function kernelDefinitionName(kernel) {
  return /__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/u.exec(kernel)?.[1];
}

function collectObjectDefines(source) {
  const defines = new Map();
  for (const line of source.split(/\r?\n/u)) {
    const stripped = stripLineComment(line);
    const match = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)(?!\()\s+(.+?)\s*$/u.exec(stripped);
    if (match === null) continue;
    const [, name, value] = match;
    if (name !== undefined && value !== undefined) defines.set(name, value.trim());
  }
  return defines;
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

function collectFunctionDefines(source) {
  const out = [];
  for (const line of source.split(/\r?\n/u)) {
    const stripped = stripLineComment(line);
    if (/^\s*#define\s+[A-Za-z_][A-Za-z0-9_]*\([^)]*\)\s+.+/u.test(stripped)) out.push(stripped.trim());
  }
  return out;
}

function collectTextureDeclarations(source) {
  const clean = stripComments(source);
  const declarations = [];
  const re = /texture\s*<[^;]+>\s*[A-Za-z_][A-Za-z0-9_]*\s*;/g;
  let match;
  while ((match = re.exec(clean))) declarations.push(match[0]);
  return declarations;
}

function kernelParamNames(kernel) {
  const signature = /__global__\s+(?:void\s+[A-Za-z_][A-Za-z0-9_]*\s*)?\(([\s\S]*?)\)\s*\{/u.exec(kernel);
  if (signature === null) return [];
  return signature[1]
    .split(",")
    .map((param) => /([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*$/u.exec(param.trim())?.[1])
    .filter(Boolean);
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
  for (const match of source.matchAll(/extern\s+__shared__\s+\w+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*]/g)) {
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
  return { corpusPathArg, details, expectations, firstFailureLimit, help };
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
