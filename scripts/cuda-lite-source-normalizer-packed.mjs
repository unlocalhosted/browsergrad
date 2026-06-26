import { requireNormalizerHelpers } from "./cuda-lite-source-normalizer-context.mjs";

export const WIDE_PACKED128_TYPES = new Map([
  ["bg_pack128_half8", { scalarType: "half", lanes: 8 }],
  ["bg_pack128_bf168", { scalarType: "bf16", lanes: 8 }],
]);
const collectObjectDefines = (...args) => requireNormalizerHelpers().collectObjectDefines(...args);
const collectTypeAliasDefines = (...args) => requireNormalizerHelpers().collectTypeAliasDefines(...args);
const collectVisiblePointerTypes = (...args) => requireNormalizerHelpers().collectVisiblePointerTypes(...args);
const escapeRegExp = (...args) => requireNormalizerHelpers().escapeRegExp(...args);
const findBalanced = (...args) => requireNormalizerHelpers().findBalanced(...args);
const isMacroIdentifier = (...args) => requireNormalizerHelpers().isMacroIdentifier(...args);
const mergeDefineMaps = (...args) => requireNormalizerHelpers().mergeDefineMaps(...args);
const normalizeTemplateTypeArgument = (...args) => requireNormalizerHelpers().normalizeTemplateTypeArgument(...args);
const splitTopLevel = (...args) => requireNormalizerHelpers().splitTopLevel(...args);

export function normalizeWidePacked128Aliases(source, definesByName = new Map()) {
  const aliases = collectWidePacked128Aliases(source, definesByName);
  if (aliases.size === 0) return source;
  const variables = collectWidePacked128Variables(source, aliases);
  const pointerAliases = new Map();
  const sharedByteNames = collectSharedByteNames(source);
  let out = source;
  for (const [alias, info] of [...aliases].sort((a, b) => b[0].length - a[0].length)) {
    out = out.replace(new RegExp(`\\b${escapeRegExp(alias)}\\s*::\\s*size\\b`, "gu"), String(info.lanes));
  }
  out = rewriteWidePacked128PointerViews(out, aliases, pointerAliases, sharedByteNames);
  out = rewriteWidePacked128LoadDeclarations(out, aliases, variables);
  out = rewriteWidePacked128IndexedLoadDeclarations(out, aliases, pointerAliases, variables);
  out = rewriteWidePacked128ZeroDeclarations(out, aliases, variables);
  out = rewriteWidePacked128PlainDeclarations(out, aliases, variables);
  out = rewriteWidePacked128LoadAssignments(out, variables);
  out = rewriteWidePacked128IndexedStores(out, pointerAliases);
  out = rewriteWidePacked128Stores(out, variables);
  for (const [name, info] of variables) {
    out = out.replace(new RegExp(`\\b${escapeRegExp(name)}\\s*\\.\\s*size\\b`, "gu"), String(info.lanes));
  }
  return out;
}

function collectWidePacked128Aliases(source, definesByName = new Map()) {
  const aliases = new Map();
  const defines = mergeDefineMaps(definesByName, collectTypeAliasDefines(source, definesByName), collectObjectDefines(source));
  for (const [name, value] of defines) {
    if (!isMacroIdentifier(name)) continue;
    const info = WIDE_PACKED128_TYPES.get(value);
    if (info !== undefined) aliases.set(name, info);
  }
  return aliases;
}

function collectWidePacked128Variables(source, aliases) {
  const variables = new Map();
  for (const [alias, info] of aliases) {
    const re = new RegExp(`\\b(?:const\\s+)?${escapeRegExp(alias)}\\s+(?!\\*)([A-Za-z_][A-Za-z0-9_]*)\\b`, "gu");
    for (const match of source.matchAll(re)) {
      const name = match[1];
      if (name !== undefined) variables.set(name, info);
    }
  }
  return variables;
}

function collectSharedByteNames(source) {
  const names = new Set();
  const re = /\bextern\s+__shared__\s+char\s*\*?\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[\s*\])?\s*;/gu;
  for (const match of source.matchAll(re)) {
    if (match[1] !== undefined) names.add(match[1]);
  }
  return names;
}

function rewriteWidePacked128PointerViews(source, aliases, pointerAliases, sharedByteNames) {
  let out = source;
  const sharedViews = new Map();
  for (const [alias, info] of aliases) {
    out = rewriteWidePacked128PointerViewsForAlias(out, alias, info, pointerAliases, sharedByteNames, sharedViews);
  }
  for (const [name, info] of sharedViews) {
    out = out.replace(
      new RegExp(`\\bextern\\s+__shared__\\s+char\\s*\\*?\\s+${escapeRegExp(name)}\\s*(?:\\[\\s*\\])?\\s*;`, "gu"),
      `extern __shared__ ${info.scalarType} ${name}[];`,
    );
  }
  return out;
}

function rewriteWidePacked128PointerViewsForAlias(source, alias, info, pointerAliases, sharedByteNames, sharedViews) {
  const re = new RegExp(`\\b(const\\s+)?${escapeRegExp(alias)}\\s*\\*\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*reinterpret_cast\\s*<\\s*(?:const\\s+)?${escapeRegExp(alias)}\\s*\\*\\s*>\\s*\\(`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const name = match[2];
    const open = source.indexOf("(", match.index);
    const close = findBalanced(source, open, "(", ")");
    const semi = close === undefined ? -1 : source.indexOf(";", close + 1);
    if (name === undefined || open < 0 || close === undefined || semi < 0) {
      re.lastIndex = match.index + 1;
      continue;
    }
    const base = source.slice(open + 1, close).trim();
    const root = pointerBaseIdentifier(base);
    const suffix = source.slice(close + 1, semi).trim();
    const offset = widePacked128PointerViewOffset(suffix);
    pointerAliases.set(name, info);
    if (root !== undefined && sharedByteNames.has(root)) sharedViews.set(root, info);
    out += source.slice(cursor, match.index);
    out += `${match[1] ?? ""}${info.scalarType}* ${name} = ${widePacked128ScalarPointerExpression(base, offset, info, sharedByteNames)};`;
    cursor = semi + 1;
    re.lastIndex = cursor;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function widePacked128PointerViewOffset(suffix) {
  if (suffix.length === 0) return undefined;
  const match = /^\+\s*([\s\S]+)$/u.exec(suffix);
  return match?.[1]?.trim();
}

function widePacked128ScalarPointerExpression(base, offset, info, sharedByteNames) {
  const root = pointerBaseIdentifier(base);
  const scalarBase = root !== undefined && sharedByteNames.has(root)
    ? (widePacked128BytePointerExpression(base, info) ?? base)
    : base;
  if (offset === undefined || offset.length === 0) return scalarBase;
  return `${scalarBase} + ((${offset}) * ${info.lanes})`;
}

function widePacked128BytePointerExpression(base, info) {
  const byteSize = cudaTypeByteSize(info.scalarType) ?? 4;
  const parts = splitTopLevel(base.replace(/^\(([\s\S]*)\)$/u, "$1"), "+").map((part) => part.trim()).filter(Boolean);
  const root = parts[0];
  if (root === undefined || !/^[A-Za-z_][A-Za-z0-9_]*$/u.test(root)) return base;
  if (parts.length === 1) return root;
  const bytes = parts.slice(1).join(" + ").replace(/\bsizeof\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)/gu, (match, type) =>
    String(cudaTypeByteSize(type) ?? (type === "floatX" ? byteSize : match)));
  return `${root} + ((${bytes}) / ${byteSize})`;
}

function rewriteWidePacked128LoadDeclarations(source, aliases, variables) {
  let out = source;
  for (const [alias, info] of aliases) {
    const re = new RegExp(`\\b(?:const\\s+)?${escapeRegExp(alias)}\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*(load128cs|load128)\\s*\\(`, "gu");
    out = replaceBalancedCall(out, re, (match, open, close) => {
      const name = match[1];
      if (name === undefined) return undefined;
      variables.set(name, info);
      const pointer = out.slice(open + 1, close).trim();
      return emitWidePacked128LoadDeclaration(info, name, pointer);
    });
  }
  return out;
}

function rewriteWidePacked128IndexedLoadDeclarations(source, aliases, pointerAliases, variables) {
  let out = source;
  for (const [alias, info] of aliases) {
    const re = new RegExp(`\\b(?:const\\s+)?${escapeRegExp(alias)}\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\[`, "gu");
    out = replaceBalancedIndexStatement(out, re, (current, match, open, close) => {
      const name = match[1];
      const pointer = match[2];
      if (name === undefined || pointer === undefined || pointerAliases.get(pointer) !== info) return undefined;
      variables.set(name, info);
      const index = current.slice(open + 1, close).trim();
      return emitWidePacked128IndexedLoadDeclaration(info, name, pointer, index);
    });
  }
  return out;
}

function rewriteWidePacked128ZeroDeclarations(source, aliases, variables) {
  let out = source;
  for (const [alias, info] of aliases) {
    const re = new RegExp(`\\b(?:const\\s+)?${escapeRegExp(alias)}\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*${escapeRegExp(alias)}\\s*::\\s*zeros\\s*\\(\\s*\\)\\s*;`, "gu");
    out = out.replace(re, (_match, name) => {
      variables.set(name, info);
      return emitWidePacked128ZeroDeclaration(info, name);
    });
  }
  return out;
}

function rewriteWidePacked128PlainDeclarations(source, aliases, variables) {
  let out = source;
  for (const [alias, info] of aliases) {
    const declaration = new RegExp(`\\b(?:const\\s+)?${escapeRegExp(alias)}\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*;`, "gu");
    out = out.replace(declaration, (_match, name) => {
      variables.set(name, info);
      return `${info.scalarType} ${name}[${info.lanes}];`;
    });
  }
  return out;
}

function rewriteWidePacked128LoadAssignments(source, variables) {
  const helper = /\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(load128cs|load128)\s*\(/gu;
  return replaceBalancedCall(source, helper, (match, open, close) => {
    const name = match[1];
    if (name === undefined) return undefined;
    if (!isStandaloneStatementPrefix(source, match.index)) return undefined;
    const info = variables.get(name);
    if (info === undefined) return undefined;
    const pointer = source.slice(open + 1, close).trim();
    return emitWidePacked128LoadAssignments(info, name, pointer);
  });
}

function isStandaloneStatementPrefix(source, index) {
  const start = Math.max(
    source.lastIndexOf(";", index),
    source.lastIndexOf("{", index),
    source.lastIndexOf("}", index),
    source.lastIndexOf("\n", index),
  ) + 1;
  return source.slice(start, index).trim().length === 0;
}

function rewriteWidePacked128IndexedStores(source, pointerAliases) {
  let out = source;
  for (const [pointer, info] of pointerAliases) {
    const re = new RegExp(`\\b${escapeRegExp(pointer)}\\s*\\[`, "gu");
    out = replaceBalancedIndexStatement(out, re, (current, _match, open, close, semi) => {
      const index = current.slice(open + 1, close).trim();
      const rest = current.slice(close + 1, semi).trim();
      const assignment = /^=\s*([A-Za-z_][A-Za-z0-9_]*)\s*$/u.exec(rest);
      const value = assignment?.[1];
      if (value === undefined) return undefined;
      return emitWidePacked128IndexedStore(info, pointer, index, value);
    });
  }
  return out;
}

function rewriteWidePacked128Stores(source, variables) {
  const helper = /\b(store128cs|store128cg|store128)\s*\(/gu;
  return replaceBalancedCall(source, helper, (_match, open, close) => {
    const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim());
    const pointer = args[0];
    const value = args[1];
    if (pointer === undefined || value === undefined) return undefined;
    const info = /^[A-Za-z_][A-Za-z0-9_]*$/u.test(value) ? variables.get(value) : undefined;
    if (info === undefined) return undefined;
    return emitWidePacked128StoreAssignments(info, pointer, value);
  });
}

export function replaceBalancedIndexStatement(source, re, replacer) {
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("[", match.index);
    const close = findBalanced(source, open, "[", "]");
    const semi = close === undefined ? -1 : source.indexOf(";", close + 1);
    if (open < 0 || close === undefined || semi < 0) {
      re.lastIndex = match.index + 1;
      continue;
    }
    const replacement = replacer(source, match, open, close, semi);
    if (replacement === undefined) {
      re.lastIndex = close + 1;
      continue;
    }
    out += source.slice(cursor, match.index);
    out += replacement.endsWith(";") ? replacement : `${replacement};`;
    cursor = semi + 1;
    re.lastIndex = cursor;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

export function replaceBalancedCall(source, re, replacer) {
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("(", match.index);
    const close = findBalanced(source, open, "(", ")");
    if (open < 0 || close === undefined) {
      re.lastIndex = match.index + 1;
      continue;
    }
    const replacement = replacer(match, open, close);
    if (replacement === undefined) {
      re.lastIndex = close + 1;
      continue;
    }
    out += source.slice(cursor, match.index);
    out += replacement;
    cursor = close + 1;
    re.lastIndex = close + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function emitWidePacked128LoadDeclaration(info, name, pointer) {
  return `${info.scalarType} ${name}[${info.lanes}]; ${emitWidePacked128LoadAssignments(info, name, pointer)}`;
}

function emitWidePacked128IndexedLoadDeclaration(info, name, pointer, index) {
  const base = `(${index}) * ${info.lanes}`;
  return `${info.scalarType} ${name}[${info.lanes}]; ${Array.from({ length: info.lanes }, (_unused, lane) =>
    `${name}[${lane}] = ${pointer}[(${base}) + ${lane}]`).join("; ")}`;
}

function emitWidePacked128ZeroDeclaration(info, name) {
  const zero = widePacked128ZeroLiteral(info.scalarType);
  return `${info.scalarType} ${name}[${info.lanes}]; ${Array.from({ length: info.lanes }, (_unused, lane) =>
    `${name}[${lane}] = ${zero}`).join("; ")};`;
}

function emitWidePacked128LoadAssignments(info, name, pointer) {
  return Array.from({ length: info.lanes }, (_unused, lane) =>
    `${name}[${lane}] = ${widePacked128PointerLane(pointer, lane)}`).join("; ");
}

function emitWidePacked128IndexedStore(info, pointer, index, value) {
  const base = `(${index}) * ${info.lanes}`;
  return Array.from({ length: info.lanes }, (_unused, lane) =>
    `${pointer}[(${base}) + ${lane}] = ${value}[${lane}]`).join("; ");
}

function emitWidePacked128StoreAssignments(info, pointer, value) {
  return Array.from({ length: info.lanes }, (_unused, lane) =>
    `${widePacked128PointerLane(pointer, lane)} = ${value}[${lane}]`).join("; ");
}

function widePacked128ZeroLiteral(scalarType) {
  if (scalarType === "half") return "__float2half(0.0f)";
  return "0.0f";
}

function widePacked128PointerLane(pointer, lane) {
  const parts = splitTopLevel(pointer.replace(/^\(([\s\S]*)\)$/u, "$1"), "+").map((part) => part.trim()).filter(Boolean);
  const base = parts[0];
  if (base !== undefined && /^[A-Za-z_][A-Za-z0-9_]*$/u.test(base)) {
    const offset = parts.slice(1).join(" + ");
    return offset.length === 0 ? `${base}[${lane}]` : `${base}[(${offset}) + ${lane}]`;
  }
  return `(${pointer})[${lane}]`;
}

export function normalizeCudaPipelineAsync(source) {
  const alignedSizes = collectCudaAlignedSizeConstants(source);
  let out = source.replace(
    /\b(?:const\s+)?auto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*cuda::aligned_size_t\s*<\s*alignof\s*\(\s*([^)]+?)\s*\)\s*>\s*\(\s*sizeof\s*\(\s*([^)]+?)\s*\)\s*\)\s*;/gu,
    (_match, name, _alignType, sizeType) => `const int ${name} = ${cudaTypeByteSize(sizeType) ?? 4};`,
  );
  out = out.replace(
    /\b__shared__\s+cuda::pipeline_shared_state\s*<[^;]+>\s+[A-Za-z_][A-Za-z0-9_]*\s*;/gu,
    "",
  );
  out = out.replace(
    /\b__shared__\s+(?:cg|cooperative_groups)::block_tile_memory\s*<[^;]+>\s+[A-Za-z_][A-Za-z0-9_]*\s*;/gu,
    "",
  );
  out = out.replace(
    /\bauto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*cg::this_thread_block\s*\(\s*\)\s*;/gu,
    "cg::thread_block $1 = cg::this_thread_block();",
  );
  out = out.replace(
    /\bauto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*cg::this_thread_block\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\)\s*;/gu,
    "cg::thread_block $1 = cg::this_thread_block();",
  );
  out = out.replace(
    /\b(?:const\s+)?auto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^;]*?cuda::pipeline_role::[A-Za-z_][A-Za-z0-9_:]*\s*;/gu,
    "const int $1 = 0;",
  );
  out = out.replace(
    /\bcuda::pipeline\s*<[^;=]+>\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*cuda::make_pipeline\s*\([^;]*\)\s*;/gu,
    "int $1 = 0;",
  );
  out = out.replace(
    /\bauto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*cuda::make_pipeline\s*\([^;]*\)\s*;/gu,
    "int $1 = 0;",
  );
  out = rewriteCudaMemcpyAsync(out, alignedSizes);
  out = out.replace(/\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*producer_acquire\s*\(\s*\)\s*;/gu, "");
  out = out.replace(/\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*producer_commit\s*\(\s*\)\s*;/gu, "CP_ASYNC_COMMIT_GROUP();");
  out = out.replace(/\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*consumer_wait\s*\(\s*\)\s*;/gu, "CP_ASYNC_WAIT_GROUP(0);");
  out = out.replace(/\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*consumer_release\s*\(\s*\)\s*;/gu, "");
  return out;
}

function collectCudaAlignedSizeConstants(source) {
  const constants = new Map();
  const re = /\b(?:const\s+)?auto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*cuda::aligned_size_t\s*<\s*alignof\s*\(\s*([^)]+?)\s*\)\s*>\s*\(\s*sizeof\s*\(\s*([^)]+?)\s*\)\s*\)\s*;/gu;
  for (const match of source.matchAll(re)) {
    const name = match[1];
    const sizeType = match[3];
    const size = cudaTypeByteSize(sizeType);
    if (name !== undefined && size !== undefined) constants.set(name, size);
  }
  return constants;
}

function rewriteCudaMemcpyAsync(source, alignedSizes) {
  return replaceBalancedCall(source, /\bcuda::memcpy_async\s*\(/gu, (_match, open, close) => {
    const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim());
    const dst = args[0];
    const src = args[1];
    const bytes = cudaMemcpyAsyncByteCount(args[2], alignedSizes);
    if (dst === undefined || src === undefined) return undefined;
    return `CP_ASYNC_CG(${dst}, ${src}, ${bytes})`;
  });
}

function cudaMemcpyAsyncByteCount(expression, alignedSizes) {
  if (expression === undefined) return 4;
  const trimmed = expression.trim();
  if (/^[A-Za-z_][A-Za-z0-9_]*$/u.test(trimmed)) return alignedSizes.get(trimmed) ?? trimmed;
  const sizeof = /^sizeof\s*\(\s*([^)]+?)\s*\)$/u.exec(trimmed);
  if (sizeof?.[1] !== undefined) return cudaTypeByteSize(sizeof[1]) ?? trimmed;
  const aligned = /^cuda::aligned_size_t\s*<\s*alignof\s*\(\s*([^)]+?)\s*\)\s*>\s*\(\s*sizeof\s*\(\s*([^)]+?)\s*\)\s*\)$/u.exec(trimmed);
  if (aligned?.[2] !== undefined) return cudaTypeByteSize(aligned[2]) ?? trimmed;
  return trimmed;
}

export function cudaTypeByteSize(type) {
  const normalized = type
    .replace(/\bconst\b|\bvolatile\b|\b__restrict__\b|\b__restrict\b|\brestrict\b/gu, "")
    .replace(/\s+/gu, " ")
    .trim();
  if (normalized === "float4" || normalized === "int4" || normalized === "uint4") return 16;
  if (normalized === "float2" || normalized === "int2" || normalized === "uint2" || normalized === "double") return 8;
  if (normalized === "float" || normalized === "int" || normalized === "uint" || normalized === "unsigned int") return 4;
  if (normalized === "half" || normalized === "__half" || normalized === "bf16" || normalized === "__nv_bfloat16") return 2;
  return undefined;
}

export function normalizePacked128MemoryHelpers(source) {
  let current = source;
  for (let pass = 0; pass < 4; pass++) {
    const next = normalizePacked128MemoryHelpersOnce(current);
    if (next === current) return current;
    current = next;
  }
  return current;
}

function normalizePacked128MemoryHelpersOnce(source) {
  const withScalarLoadAssignments = rewritePacked128ScalarLoadAssignments(source);
  if (withScalarLoadAssignments !== source) return withScalarLoadAssignments;
  const helper = /\b(load128cs|load128|store128cs|store128cg|store128)\s*\(/gu;
  let out = "";
  let cursor = 0;
  let match;
  while ((match = helper.exec(source)) !== null) {
    const name = match[1];
    const open = source.indexOf("(", match.index + name.length);
    const close = findBalanced(source, open, "(", ")");
    if (name === undefined || open < 0 || close === undefined) {
      helper.lastIndex = match.index + 1;
      continue;
    }
    const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim());
    const replacement = packed128HelperReplacement(name, args, source);
    if (replacement === undefined) {
      helper.lastIndex = close + 1;
      continue;
    }
    out += source.slice(cursor, match.index);
    out += replacement;
    cursor = close + 1;
    helper.lastIndex = close + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function rewritePacked128ScalarLoadAssignments(source) {
  const symbols = collectVisiblePointerTypes(source);
  const re = /\b([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]\n;]+)\]\s*=\s*(load128cs|load128)\s*\(/gu;
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const target = match[1];
    const index = match[2]?.trim();
    const helperName = match[3];
    const open = source.indexOf("(", match.index + match[0].length - 1);
    const close = findBalanced(source, open, "(", ")");
    if (target === undefined || index === undefined || helperName === undefined || open < 0 || close === undefined) {
      re.lastIndex = match.index + 1;
      continue;
    }
    const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim());
    const pointer = args[0];
    if (pointer === undefined || pointer.length === 0) {
      re.lastIndex = close + 1;
      continue;
    }
    const targetInfo = packed128ScalarPointerInfo(target, symbols);
    const pointerInfo = packed128ScalarPointerInfo(pointerBaseIdentifier(pointer), symbols);
    if (targetInfo === undefined || pointerInfo === undefined || targetInfo.scalarType !== pointerInfo.scalarType) {
      re.lastIndex = close + 1;
      continue;
    }
    out += source.slice(cursor, match.index);
    out += emitPacked128ScalarLoadAssignments(target, index, pointer, targetInfo.lanes);
    cursor = close + 1;
    re.lastIndex = close + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function packed128ScalarPointerInfo(name, symbols) {
  if (name === undefined) return undefined;
  const type = normalizePackedScalarType(symbols.get(name));
  const size = type === undefined ? undefined : cudaTypeByteSize(type);
  if (type === undefined || size === undefined || size <= 0 || size > 16 || 16 % size !== 0) return undefined;
  if (!["float", "int", "uint", "half", "bf16"].includes(type)) return undefined;
  return { scalarType: type, lanes: 16 / size };
}

function normalizePackedScalarType(type) {
  if (type === undefined) return undefined;
  if (type === "__half") return "half";
  if (type === "__nv_bfloat16") return "bf16";
  if (type === "unsigned int") return "uint";
  return type;
}

function emitPacked128ScalarLoadAssignments(target, index, pointer, lanes) {
  const base = `(${index}) * ${lanes}`;
  return Array.from({ length: lanes }, (_unused, lane) =>
    `${target}[(${base}) + ${lane}] = ${widePacked128PointerLane(pointer, lane)}`).join("; ");
}

function packed128HelperReplacement(name, args, source) {
  const pointer = args[0];
  if (pointer === undefined || pointer.length === 0) return undefined;
  const vectorType = inferPacked128PointerVectorType(pointer, source);
  if (vectorType === undefined) return undefined;
  if (name === "load128" || name === "load128cs") {
    return `reinterpret_cast<${vectorType} *>(${pointer})[0]`;
  }
  const value = args[1];
  if (value === undefined || value.length === 0) return undefined;
  return `(reinterpret_cast<${vectorType} *>(${pointer})[0] = ${value})`;
}

export function normalizeVectorStaticConstructors(source, definesByName = new Map()) {
  const aliases = new Map();
  for (const vectorType of [
    "float2", "float3", "float4", "half2",
    "int2", "int3", "int4", "uint2", "uint3", "uint4",
  ]) {
    aliases.set(vectorType, vectorType);
  }
  for (const [name, value] of definesByName) {
    if (!isMacroIdentifier(name)) continue;
    const vectorType = normalizeTemplateTypeArgument(value, definesByName);
    if (vectorType !== undefined && supportedVectorStaticType(vectorType)) aliases.set(name, vectorType);
  }
  let out = source;
  for (const [alias, vectorType] of [...aliases].sort((a, b) => b[0].length - a[0].length)) {
    out = out.replace(new RegExp(`\\b${escapeRegExp(alias)}\\s*::\\s*zeros\\s*\\(\\s*\\)`, "gu"), vectorSplatConstructor(vectorType, vectorZeroLiteral(vectorType)));
    out = rewriteVectorConstantConstructors(out, alias, vectorType);
  }
  return out;
}

function supportedVectorStaticType(vectorType) {
  return vectorLaneCount(vectorType) !== undefined;
}

function rewriteVectorConstantConstructors(source, alias, vectorType) {
  const re = new RegExp(`\\b${escapeRegExp(alias)}\\s*::\\s*constant\\s*\\(`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("(", match.index);
    const close = findBalanced(source, open, "(", ")");
    if (open < 0 || close === undefined) {
      re.lastIndex = match.index + alias.length;
      continue;
    }
    const value = source.slice(open + 1, close).trim() || vectorZeroLiteral(vectorType);
    out += source.slice(cursor, match.index);
    out += vectorSplatConstructor(vectorType, value);
    cursor = close + 1;
    re.lastIndex = close + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function vectorSplatConstructor(vectorType, value) {
  const lanes = vectorLaneCount(vectorType);
  return `make_${vectorType}(${Array.from({ length: lanes ?? 0 }, () => value).join(", ")})`;
}

function vectorZeroLiteral(vectorType) {
  if (vectorType.startsWith("half")) return "__float2half(0.0f)";
  if (vectorType.startsWith("float")) return "0.0f";
  return "0";
}

export function vectorLaneCount(vectorType) {
  if (vectorType === "half2") return 2;
  const match = /^(?:float|int|uint)([234])$/u.exec(vectorType);
  return match?.[1] === undefined ? undefined : Number(match[1]);
}

function inferPacked128PointerVectorType(pointer, source) {
  const base = pointerBaseIdentifier(pointer);
  if (base === undefined) return undefined;
  const symbols = collectVisiblePointerTypes(source);
  const defines = collectObjectDefines(source);
  let type = symbols.get(base);
  if (type === undefined) return undefined;
  type = normalizeTemplateTypeArgument(defines.get(type) ?? type, defines) ?? type;
  if (type === "float") return "float4";
  if (type === "int") return "int4";
  if (type === "uint") return "uint4";
  if (type === "float4" || type === "int4" || type === "uint4") return type;
  return undefined;
}

export function pointerBaseIdentifier(pointer) {
  const expression = pointer
    .replace(/\breinterpret_cast\s*<[^>]+>\s*\(/gu, "(")
    .replace(/\b(?:const|volatile)\b/gu, " ")
    .trim();
  return /(?:^|[^A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\b/u.exec(expression)?.[1];
}
