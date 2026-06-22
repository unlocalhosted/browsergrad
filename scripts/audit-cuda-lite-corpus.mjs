#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [corpusPathArg] = process.argv.slice(2);
if (!corpusPathArg) {
  console.error("usage: node scripts/audit-cuda-lite-corpus.mjs <corpus-path>");
  process.exit(2);
}

const corpusRoot = path.resolve(corpusPathArg);
const compilerUrl = pathToFileURL(path.resolve("packages/browsergrad-compiler/dist/index.js")).href;
const { compileCudaLiteKernel, describeCudaDiagnostic } = await import(compilerUrl);
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
    if (isDocumentationDiagramBlock(block)) continue;
    const blockDefines = collectObjectDefines(block.code);
    const blockFunctionDefines = collectFunctionDefines(block.code);
    const blockDeviceFunctions = collectScalarDeviceFunctions(block.code);
    const blockConstants = collectConstantDeclarations(block.code);
    const blockTextures = collectTextureDeclarations(block.code);
    const effectiveDefines = blockDefines.size === 0 ? carriedDefines : blockDefines;
    if (CUDA_HINT_RE.test(block.code)) cudaBlocks++;
    const kernels = extractKernelDefinitions(block.code);
    for (const [kernelIndex, rawKernel] of kernels.entries()) {
      const source = kernelSourceWithContext(rawKernel, effectiveDefines, blockFunctionDefines, blockDeviceFunctions, blockConstants, blockTextures);
      try {
        compileCudaLiteKernel(source, {
          features: { "shader-f16": true, subgroups: true },
          workgroupSize: [256, 1, 1],
          dynamicSharedMemory: inferDynamicSharedMemory(source),
        });
        results.push({ file, block: blockIndex + 1, kernel: kernelIndex + 1, ok: true });
      } catch (error) {
        const referenceOk = canCompileReferenceOnly(source);
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
          referenceOk,
        });
      }
    }
    carriedDefines = mergeCarriedDefines(carriedDefines, blockDefines);
  }
}

function isDocumentationDiagramBlock(block) {
  if (NON_CODE_BLOCK_LANG_RE.test(block.lang)) return true;
  return /^\s*(?:flowchart|graph)\s+(?:LR|RL|TB|TD|BT)\b/iu.test(block.code);
}

const failures = results.filter((result) => !result.ok);
const summary = {
  files: files.length,
  codeBlocks,
  cudaBlocks,
  totalKernelDefinitions: results.length,
  ok: results.length - failures.length,
  fail: failures.length,
  referenceOnlyOk: failures.filter((failure) => failure.referenceOk).length,
  hardFail: failures.filter((failure) => !failure.referenceOk).length,
  errors: countBy(failures, (failure) => failure.error),
  families: countBy(failures, (failure) => failure.family),
  lowering: countBy(failures, (failure) => failure.lowering),
};

console.log(JSON.stringify(summary, null, 2));
if (failures.length > 0) {
  console.log("\nfirst failures:");
  for (const failure of failures.slice(0, 80)) {
    const reference = failure.referenceOk ? " [reference-ok]" : "";
    console.log(`${failure.file} block ${failure.block} kernel ${failure.kernel}: ${failure.family}/${failure.error}${reference}: ${failure.message}`);
  }
}

function canCompileReferenceOnly(source) {
  try {
    compileCudaLiteKernel(source, {
      features: { "shader-f16": true, subgroups: true },
      workgroupSize: [256, 1, 1],
      dynamicSharedMemory: inferDynamicSharedMemory(source),
      referenceDynamicParallelism: true,
      referenceGridSync: true,
    });
    return true;
  } catch {
    return false;
  }
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
  return /\(\s*\.\.\.\s*\)/u.test(signature) || /\?\?\?/u.test(kernel);
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

function kernelSourceWithContext(kernel, definesByName, functionDeclarations, deviceFunctions, constantDeclarations, textureDeclarations) {
  const params = new Set(kernelParamNames(kernel));
  const defines = [...definesByName]
    .filter(([name]) => !params.has(name))
    .map(([name, value]) => `#define ${name} ${value}`);
  const referencedDeviceFunctions = deviceFunctions
    .filter((fn) => new RegExp(`\\b${escapeRegExp(fn.name)}\\s*\\(`, "u").test(kernel))
    .map((fn) => fn.source);
  return `${defines.join("\n")}\n${functionDeclarations.join("\n")}\n${referencedDeviceFunctions.join("\n")}\n${constantDeclarations.join("\n")}\n${textureDeclarations.join("\n")}\n${kernel}`;
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
