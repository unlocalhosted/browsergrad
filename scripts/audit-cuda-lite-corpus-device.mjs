import { requireAuditHelpers } from "./audit-cuda-lite-corpus-context.mjs";

const PORTABLE_POINTER_BASE_TYPES = new Set(["float", "double", "int", "uint", "half", "half2", "bf16", "bool", "float2", "float3", "float4", "int2", "int3", "int4", "uint2", "uint3", "uint4"]);
const SEMANTIC_BUILTIN_DEVICE_HELPERS = new Set(["min", "max", "__ldcs", "__stcs", "__ldg", "__stcg", "__usad4", "vec_at", "load128", "load128cs", "store128", "store128cs", "store128cg", "div_ceil", "blockReduce", "warpReduceSum", "warpReduceMax", "warpReduceMin", "warp_reduce_sum", "warp_reduce_max", "warp_reduce_min", "warp_reduce_sum_f32", "warp_reduce_max_f32", "warp_reduce_sum_f16", "warp_reduce_sum_f16_f16", "warp_reduce_sum_f16_f32", "warp_reduce_sum_i8_i32", "warp_reduce_sum_i32_i32", "atomicAdd", "atomicSub", "atomicMin", "atomicMax", "atomicAnd", "atomicOr", "atomicXor", "atomicExch", "atomicCAS"]);
const collectCudaFunctionBodies = (...args) => requireAuditHelpers().collectCudaFunctionBodies(...args);
const expandCudaQualifierMacros = (...args) => requireAuditHelpers().expandCudaQualifierMacros(...args);
const pruneCudaPreprocessorBranches = (...args) => requireAuditHelpers().pruneCudaPreprocessorBranches(...args);
const stripComments = (...args) => requireAuditHelpers().stripComments(...args);
const withCudaDeclarationPrefixStart = (...args) => requireAuditHelpers().withCudaDeclarationPrefixStart(...args);

export function collectPortableDeviceFunctions(source, recordNames = new Set(), definesByName = new Map()) {
  const helperDefines = mergeDefineMaps(definesByName, new Map([["__CUDACC__", "1"]]));
  const clean = pruneCudaPreprocessorBranches(expandCudaQualifierMacros(stripComments(source)), helperDefines);
  const functions = [];
  const seenSignatures = new Set();
  const maybeAddFunction = (rawFn) => {
    const fn = pruneCudaPreprocessorBranches(rawFn, helperDefines);
    const signature = fn.slice(0, fn.indexOf("{"));
    const name = cudaFunctionDefinitionName(signature);
    const signatureKey = signature.replace(/\s+/gu, " ").trim();
    if (
      name &&
      !seenSignatures.has(signatureKey) &&
      !SEMANTIC_BUILTIN_DEVICE_HELPERS.has(name) &&
      !sourceLaunchesDeviceFunction(clean, name) &&
      isPortableDeviceFunctionCandidate(signature, fn, name, recordNames, definesByName)
    ) {
      seenSignatures.add(signatureKey);
      functions.push({ name, source: fn });
    }
  };
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
    const rawFn = clean.slice(start, end);
    maybeAddFunction(rawFn);
    index = end;
  }
  for (const rawFn of collectTemplatedDeviceFunctionBodies(clean)) maybeAddFunction(rawFn);
  return functions;
}

export function collectTemplatedDeviceFunctionBodies(source) {
  const functions = [];
  const templateRe = /\btemplate\s*<[^;{}]*>\s*(?:(?:static|inline|__inline__|__forceinline__|__host__)\s+)*__device__\b/gu;
  let match;
  while ((match = templateRe.exec(source)) !== null) {
    const start = match.index;
    const device = source.indexOf("__device__", start);
    if (device < 0) continue;
    const brace = source.indexOf("{", device);
    const semicolon = source.indexOf(";", device);
    if (brace < 0) break;
    if (semicolon >= 0 && semicolon < brace) {
      templateRe.lastIndex = semicolon + 1;
      continue;
    }
    const end = findBalanced(source, brace, "{", "}");
    if (end === undefined) break;
    functions.push(source.slice(start, end + 1));
    templateRe.lastIndex = end + 1;
  }
  return functions;
}

export function collectDynamicLaunchTargetDeviceFunctions(source) {
  const clean = expandCudaQualifierMacros(stripComments(source));
  return collectCudaFunctionBodies(clean, "__device__")
    .filter((fn) => {
      const signature = fn.slice(0, fn.indexOf("{"));
      const name = cudaFunctionDefinitionName(signature);
      return name !== undefined && sourceLaunchesDeviceFunction(clean, name);
    })
    .map((fn) => fn.replace(/\b__device__\b/u, "__global__"));
}

export function cudaFunctionDefinitionName(signature) {
  const open = signature.lastIndexOf("(");
  if (open < 0) return undefined;
  const before = signature.slice(0, open).trim();
  if (/\boperator\b/u.test(before)) return undefined;
  const name = /([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^<>]*>)?\s*$/u.exec(before)?.[1];
  if (name === undefined || ["if", "for", "while", "switch", "return"].includes(name)) return undefined;
  return name;
}

export function isPortableDeviceFunctionCandidate(signature, source, name, recordNames = new Set(), definesByName = new Map()) {
  if (isCudaMemberFunctionLike(signature, name)) return false;
  return isPortableScalarDeviceFunction(signature, source) ||
    isPortablePointerDeviceFunction(signature, source, name, recordNames, definesByName) ||
    isPortableReferenceDeviceFunction(signature, source, name, recordNames, definesByName);
}

export function isCudaMemberFunctionLike(signature, name) {
  const open = signature.lastIndexOf("(");
  if (open < 0) return true;
  const before = signature.slice(0, open).trim();
  if (new RegExp(`~\\s*${escapeRegExp(name)}\\s*(?:<[^<>]*>)?\\s*$`, "u").test(before)) return true;
  if (new RegExp(`::\\s*${escapeRegExp(name)}\\s*(?:<[^<>]*>)?\\s*$`, "u").test(before)) return true;
  const nameSuffix = new RegExp(`\\b${escapeRegExp(name)}\\s*(?:<[^<>]*>)?\\s*$`, "u");
  const returnType = before
    .replace(nameSuffix, "")
    .replace(/^\s*template\s*<[^>]*>\s*/u, " ")
    .replace(/\b(?:static|inline|__inline__|__forceinline__|__device__|__host__|constexpr|const|volatile)\b/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  return returnType.length === 0;
}

export function isPortableScalarDeviceFunction(signature, source) {
  if (/\*/u.test(signature)) return false;
  if (/reinterpret|__float_as_int|__int_as_float/u.test(source)) return false;
  return true;
}

export function isPortablePointerDeviceFunction(signature, source, name, recordNames = new Set(), definesByName = new Map()) {
  if (!/\*/u.test(signature)) return false;
  if (!hasSupportedDeviceReturnShape(signature, name, recordNames, definesByName)) return false;
  if (/\bvoid\s*\*/u.test(signature)) return false;
  const pointerBases = pointerBaseTypes(signature);
  const templateTypeParams = templateTypeParamNames(signature);
  if (pointerBases.length === 0) return false;
  if (!pointerBases.every((type) => isPortableRecordAwareBaseType(type, templateTypeParams, recordNames, definesByName))) return false;
  if (/__float_as_int|__int_as_float/u.test(source)) return false;
  return true;
}

export function isPortableReferenceDeviceFunction(signature, source, name, recordNames = new Set(), definesByName = new Map()) {
  if (!/&/u.test(signature)) return false;
  if (!hasSupportedDeviceReturnShape(signature, name, recordNames, definesByName)) return false;
  const referenceBases = referenceBaseTypes(signature);
  const templateTypeParams = templateTypeParamNames(signature);
  if (referenceBases.length === 0) return false;
  if (!referenceBases.every((type) => isPortableRecordAwareBaseType(type, templateTypeParams, recordNames, definesByName))) return false;
  if (/__float_as_int|__int_as_float/u.test(source)) return false;
  return true;
}

export function isPortableBaseType(type, templateTypeParams = new Set(), definesByName = new Map()) {
  const normalized = normalizePortableBaseType(type, definesByName);
  return PORTABLE_POINTER_BASE_TYPES.has(normalized) || templateTypeParams.has(normalized);
}

export function isPortableRecordAwareBaseType(type, templateTypeParams = new Set(), recordNames = new Set(), definesByName = new Map()) {
  return isPortableBaseType(type, templateTypeParams, definesByName) ||
    recordNames.has(normalizePortableBaseType(type, definesByName));
}

export function templateTypeParamNames(signature) {
  const match = /^\s*template\s*<([^>]*)>/u.exec(signature);
  if (match?.[1] === undefined) return new Set();
  const names = new Set();
  for (const param of match[1].split(",")) {
    const type = /\b(?:typename|class)\s+([A-Za-z_][A-Za-z0-9_]*)/u.exec(param.trim())?.[1];
    if (type !== undefined) names.add(type);
  }
  return names;
}

export function pointerBaseTypes(signature) {
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

export function referenceBaseTypes(signature) {
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

export function stripCooperativeGroupParamTypes(signature) {
  return signature.replace(
    /\b(?:(?:cg|cooperative_groups)::)?(?:thread_block_tile\s*<[^>]+>|thread_block|grid_group|coalesced_group)\s*&?\s+[A-Za-z_][A-Za-z0-9_]*/gu,
    " ",
  );
}

export function normalizePortableBaseType(type, definesByName = new Map(), seen = new Set()) {
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

export function hasSupportedDeviceReturnShape(signature, name, recordNames = new Set(), definesByName = new Map()) {
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

export function recordDeclarationName(declaration) {
  const match = /\b(?:template\s*<[^<>]*>\s*)?(?:typedef\s+)?struct(?:\s+(?:__align__|alignas)\s*\([^)]*\))*\s*([A-Za-z_][A-Za-z0-9_]*)?\s*\{[\s\S]*?\}\s*([A-Za-z_][A-Za-z0-9_]*)?\s*;/u.exec(declaration);
  return match?.[2] ?? match?.[1];
}

export function sourceLaunchesDeviceFunction(source, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\s*<<\\s*<`, "u").test(source);
}

export function collectObjectDefines(source) {
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

export function parseSimpleTypeAlias(line, defines) {
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

export function normalizeAliasType(sourceType, defines) {
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

export function parseSimpleIntegerConstant(line) {
  const match = /^\s*((?:(?:static|constexpr|const)\s+)*)(?:(?:cuda\s*::\s*)?std\s*::\s*)?(?:int|uint|unsigned\s+int|size_t|ptrdiff_t|uint32_t|uint64_t|int32_t|int64_t)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z0-9_:,\s()+\-*/%<>&|^?:.]+)\s*;\s*$/u.exec(line);
  if (match === null) return undefined;
  const [, qualifiers, name, value] = match;
  if (!name || !value) return undefined;
  if (!/\b(?:const|constexpr)\b/u.test(qualifiers ?? "")) return undefined;
  return { name, value: value.trim() };
}

export function mergeDefineMaps(...maps) {
  const merged = new Map();
  for (const map of maps) {
    for (const [name, value] of map) merged.set(name, value);
  }
  return merged;
}

export function mergeCarriedDefines(previous, next) {
  const merged = new Map(previous);
  for (const [name, value] of next) {
    if (/^[A-Z_][A-Z0-9_]{1,}$/u.test(name)) merged.set(name, value);
  }
  return merged;
}

export function collectConstantDeclarations(source) {
  const clean = stripComments(source);
  const declarations = [];
  const re = /__constant__\s+[^;]+;/g;
  let match;
  while ((match = re.exec(clean))) declarations.push(match[0]);
  return declarations;
}

export function collectDeviceGlobalDeclarations(source) {
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

export function collectFunctionPointerTables(source, definesByName = new Map()) {
  const clean = stripComments(source);
  const symbols = collectDeviceFunctionPointerSymbols(clean);
  if (symbols.size === 0) return [];
  const tables = collectDeviceFunctionPointerTableDeclarations(clean, definesByName);
  if (tables.size === 0) return [];
  const entriesByHostTable = new Map();
  const copyRe = /cudaMemcpyFromSymbol\s*\(\s*&\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([^\]]+)\s*\]\s*,\s*([A-Za-z_][A-Za-z0-9_]*)/gu;
  let copyMatch;
  while ((copyMatch = copyRe.exec(clean)) !== null) {
    const hostTable = copyMatch[1];
    const rawIndex = copyMatch[2]?.trim();
    const symbolName = copyMatch[3];
    if (hostTable === undefined || rawIndex === undefined || symbolName === undefined) continue;
    const target = symbols.get(symbolName)?.target;
    const index = resolveFunctionPointerTableIndex(rawIndex, definesByName);
    if (target === undefined || index === undefined) continue;
    const entries = entriesByHostTable.get(hostTable) ?? [];
    entries.push({ index, target });
    entriesByHostTable.set(hostTable, entries);
  }
  const out = [];
  for (const [tableName, table] of tables) {
    const hostName = `h_${tableName}`;
    const entries = entriesByHostTable.get(hostName) ?? entriesByHostTable.get(tableName) ?? [];
    if (entries.length === 0) continue;
    out.push({
      tableName,
      aliasName: table.aliasName,
      entries: dedupeFunctionPointerEntries(entries),
    });
  }
  return out;
}

export function collectDeviceFunctionPointerSymbols(source) {
  const symbols = new Map();
  const re = /\b__device__\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*;/gu;
  let match;
  while ((match = re.exec(source)) !== null) {
    if (match[1] !== undefined && match[2] !== undefined && match[3] !== undefined) {
      symbols.set(match[2], { aliasName: match[1], target: match[3] });
    }
  }
  return symbols;
}

export function collectDeviceFunctionPointerTableDeclarations(source, definesByName = new Map()) {
  const tables = new Map();
  const re = /\b__device__\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([^\]]+)\s*\]\s*;/gu;
  let match;
  while ((match = re.exec(source)) !== null) {
    const aliasName = match[1];
    const tableName = match[2];
    const rawLength = match[3]?.trim();
    if (aliasName === undefined || tableName === undefined || rawLength === undefined) continue;
    const length = resolveFunctionPointerTableIndex(rawLength, definesByName);
    tables.set(tableName, { aliasName, length });
  }
  return tables;
}

export function resolveFunctionPointerTableIndex(raw, definesByName = new Map()) {
  const value = raw.trim();
  if (/^[0-9]+$/u.test(value)) return Number(value);
  const mapped = definesByName.get(value);
  if (mapped !== undefined && /^[0-9]+$/u.test(String(mapped).trim())) return Number(String(mapped).trim());
  return undefined;
}

export function dedupeFunctionPointerEntries(entries) {
  return [...new Map(entries.map((entry) => [entry.index, entry])).values()]
    .sort((a, b) => a.index - b.index);
}

export function collectFunctionDefines(source) {
  const out = [];
  for (const line of logicalPreprocessorLines(source)) {
    const stripped = stripLineComment(line);
    if (/^\s*#define\s+[A-Za-z_][A-Za-z0-9_]*\([^)]*\)\s+.+/u.test(stripped)) out.push(stripped.trim());
  }
  return out;
}

export function logicalPreprocessorLines(source) {
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

export function collectTextureDeclarations(source) {
  const clean = stripComments(source);
  const declarations = [];
  const re = /texture\s*<[^;]+>\s*[A-Za-z_][A-Za-z0-9_]*\s*;/g;
  let match;
  while ((match = re.exec(clean))) declarations.push(match[0]);
  return declarations;
}

export function collectPodRecordDeclarations(source) {
  const clean = stripComments(source);
  const declarations = [];
  let match;
  const structRe = /\bstruct\b/g;
  while ((match = structRe.exec(clean))) {
    const structStart = match.index;
    const headerStart = recordStructDeclarationStart(clean, structStart);
    const brace = clean.indexOf("{", structStart + match[0].length);
    const semicolon = clean.indexOf(";", structStart + match[0].length);
    if (brace < 0 || (semicolon >= 0 && semicolon < brace)) {
      structRe.lastIndex = structStart + match[0].length;
      continue;
    }
    const end = findBalanced(clean, brace, "{", "}");
    if (end === undefined) {
      structRe.lastIndex = structStart + match[0].length;
      continue;
    }
    const semi = clean.indexOf(";", end + 1);
    if (semi < 0) {
      structRe.lastIndex = end + 1;
      continue;
    }
    declarations.push(clean.slice(headerStart, semi + 1));
    structRe.lastIndex = semi + 1;
  }
  const bitpackUnionRe = /\bunion\s+[A-Za-z_][A-Za-z0-9_]*\s*\{\s*struct(?:\s+(?:__align__|alignas)\s*\([^)]*\))*\s*\{\s*(?:signed\s+)?short\s+[A-Za-z_][A-Za-z0-9_]*\s*;\s*(?:signed\s+)?short\s+[A-Za-z_][A-Za-z0-9_]*\s*;\s*\}\s*;\s*(?:unsigned\s+int|uint|uint32_t)\s+[A-Za-z_][A-Za-z0-9_]*\s*;\s*\}\s*;/g;
  while ((match = bitpackUnionRe.exec(clean))) declarations.push(match[0]);
  return declarations;
}

export function recordStructDeclarationStart(source, structStart) {
  const before = source.slice(0, structStart);
  const template = /\btemplate\s*<[^<>]*>\s*$/u.exec(before);
  if (template?.index !== undefined) return template.index;
  const typedef = /\btypedef\s+$/u.exec(before);
  if (typedef?.index !== undefined) return typedef.index;
  return structStart;
}

export function findBalanced(source, open, openChar, closeChar) {
  if (open < 0 || source[open] !== openChar) return undefined;
  let depth = 0;
  for (let index = open; index < source.length; index++) {
    const char = source[index];
    if (char === openChar) depth++;
    else if (char === closeChar) {
      depth--;
      if (depth === 0) return index;
    }
  }
  return undefined;
}

export function collectTranslationUnitSharedDeclarations(source) {
  const clean = stripComments(source);
  const declarations = [];
  const re = /\b(?:extern\s+)?__shared__\s+(?:float|int|unsigned\s+int|uint|half|__half|bool|char|unsigned\s+char|uchar)\s+[A-Za-z_][A-Za-z0-9_]*\s*\[[^\]]*]\s*;/g;
  let match;
  while ((match = re.exec(clean))) declarations.push(match[0]);
  return declarations;
}

export function stripLineComment(line) {
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

export function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function inferDynamicSharedMemory(source) {
  const out = {};
  const supportedType =
    "(?:float[234]?|double|int[234]?|uint[234]?|char[234]?|uchar[234]?|uint8_t[234]?|int8_t[234]?|unsigned\\s+int|signed\\s+int|short|unsigned\\s+short|unsigned\\s+char|char|int|uint|half|__half|bf16|__nv_bfloat16|bool)";
  const attrs = "(?:(?:__align__|alignas)\\s*\\([^)]*\\)\\s*)*";
  const sharedBeforeType = new RegExp(`extern\\s+__shared__\\s+${attrs}${supportedType}\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\[\\s*]`, "g");
  const sharedAfterType = new RegExp(`extern\\s+${supportedType}\\s+__shared__\\s+${attrs}([A-Za-z_][A-Za-z0-9_]*)\\s*\\[\\s*]`, "g");
  for (const match of source.matchAll(sharedBeforeType)) {
    out[match[1]] = 256;
  }
  for (const match of source.matchAll(sharedAfterType)) {
    out[match[1]] = 256;
  }
  return out;
}

export function countBy(items, keyFn) {
  const out = {};
  for (const item of items) {
    const key = keyFn(item);
    out[key] = (out[key] ?? 0) + 1;
  }
  return out;
}
