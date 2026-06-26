import { requireAuditHelpers } from "./audit-cuda-lite-corpus-context.mjs";
import fs from "node:fs";
import path from "node:path";

const NON_CODE_BLOCK_LANG_RE = /^(?:mermaid|flowchart|graphviz|dot|plantuml|text|txt)$/iu;
const CUDA_SYSTEM_DEFINES = new Map([
  ["UINT_MAX", "0xffffffffu"], ["INT_MAX", "2147483647"], ["INT_MIN", "(-2147483647 - 1)"],
  ["CUDART_PI_F", "3.141592654f"], ["CUDART_2PI_F", "6.283185307f"], ["CUDART_PIO2_F", "1.570796327f"], ["CUDART_PIO4_F", "0.785398163f"],
]);
const PORTABLE_POINTER_BASE_TYPES = new Set(["float", "double", "int", "uint", "half", "half2", "bf16", "bool", "float2", "float3", "float4", "int2", "int3", "int4", "uint2", "uint3", "uint4"]);
const collectCudaLiteContextDefines = (...args) => requireAuditHelpers().collectCudaLiteContextDefines(...args);
const escapeRegExp = (...args) => requireAuditHelpers().escapeRegExp(...args);
const parseSimpleTypeAlias = (...args) => requireAuditHelpers().parseSimpleTypeAlias(...args);
const stripLineComment = (...args) => requireAuditHelpers().stripLineComment(...args);

export function listFiles(root, prefix = "") {
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

export function findRepoRoot(start) {
  let current = path.resolve(start);
  while (true) {
    if (fs.existsSync(path.join(current, "pnpm-workspace.yaml"))) return current;
    const parent = path.dirname(current);
    if (parent === current) return path.resolve(start);
    current = parent;
  }
}

export function markdownBlocks(text, file) {
  if (!/\.(?:md|markdown)$/i.test(file)) return [{ lang: path.extname(file).slice(1), code: text }];
  const blocks = [];
  const re = /```([^\n]*)\n([\s\S]*?)```/g;
  let match;
  while ((match = re.exec(text))) {
    blocks.push({ lang: match[1].trim(), code: match[2] });
  }
  return blocks;
}

export function stripComments(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/.*$/gm, "");
}

export function extractKernelDefinitions(source) {
  const clean = stripComments(source);
  return collectCudaFunctionBodies(clean, "__global__")
    .filter((kernel) => !isPlaceholderKernel(kernel));
}

export function collectCudaFunctionBodies(source, marker) {
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

export function withTemplatePrefixStart(source, start) {
  const prefix = source.slice(0, start);
  const match = /template\s*<[^;{}]*>\s*$/u.exec(prefix);
  return match ? start - match[0].length : start;
}

export function withCudaDeclarationPrefixStart(source, start) {
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

export function skipBackwardWhitespace(source, index) {
  let cursor = index;
  while (cursor > 0 && /\s/u.test(source[cursor - 1])) cursor--;
  return cursor;
}

export function isIdentifierBoundary(char) {
  return char === undefined || !/[A-Za-z0-9_]/u.test(char);
}

export function sourceWithoutCudaFunctionBodies(source) {
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

export function expandCudaQualifierMacros(source) {
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

export function createCorpusContext(root, files) {
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
  const globalDefines = collectCorpusGlobalDefines(files, root, read);
  return {
    globalDefines,
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

export function collectCorpusGlobalDefines(files, root, read) {
  const values = new Map();
  const conflicts = new Set();
  for (const file of files) {
    const defines = new Map();
    for (const raw of read(path.resolve(root, file)).split(/\r?\n/u)) {
      const line = stripLineComment(raw);
      const alias = parseSimpleTypeAlias(line, defines);
      if (alias === undefined) continue;
      defines.set(alias.name, alias.value);
      if (!isPortableGlobalTypeAlias(alias.value)) continue;
      addUniqueCorpusDefine(values, conflicts, alias.name, alias.value);
    }
  }
  return values;
}

function addUniqueCorpusDefine(values, conflicts, name, value) {
  const previous = values.get(name);
  if (previous !== undefined && previous !== value) {
    values.delete(name);
    conflicts.add(name);
    return;
  }
  if (!conflicts.has(name)) values.set(name, value);
}

function isPortableGlobalTypeAlias(value) {
  return PORTABLE_POINTER_BASE_TYPES.has(value) || value === "complex64" || value === "texture2d" || value === "surface2d";
}

export function localIncludeNames(source) {
  return [...source.matchAll(/^\s*#include\s+"([^"]+)"/gm)]
    .map((match) => match[1])
    .filter(Boolean);
}

export function collectTransitiveIncludeSources(absoluteFile, includesFor, read, seen = new Set()) {
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

export function collectReverseTranslationUnitSources(absoluteFile, root, includesFor, reverseIncludeGraph, read) {
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

export function collectPrefixIncludeSources(includer, target, root, includesFor, read) {
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

export function uniqueResolved(files) {
  return [...new Set(files.map((file) => path.resolve(file)))];
}

export function uniqueStrings(values) {
  const seen = new Set();
  const out = [];
  for (const value of values) {
    if (value.trim().length === 0 || seen.has(value)) continue;
    seen.add(value);
    out.push(value);
  }
  return out;
}

export function resolveLocalInclude(absoluteFile, root, includeName) {
  const candidates = localIncludeCandidates(absoluteFile, root, includeName);
  for (const candidate of candidates) {
    const relative = path.relative(root, candidate);
    if (relative.startsWith("..") || path.isAbsolute(relative)) continue;
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return candidate;
  }
  return undefined;
}

export function localIncludeCandidates(absoluteFile, root, includeName) {
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

export function isPlaceholderKernel(kernel) {
  const signature = kernel.slice(0, kernel.indexOf("{"));
  return /\(\s*\.\.\.\s*\)/u.test(signature) ||
    /\?\?\?/u.test(kernel) ||
    /\bsome[A-Z][A-Za-z0-9_]*\b/u.test(stripComments(kernel));
}
