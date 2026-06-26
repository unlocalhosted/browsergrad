import { requireNormalizerHelpers } from "./cuda-lite-source-normalizer-context.mjs";

const canonicalTemplateValueArgument = (...args) => requireNormalizerHelpers().canonicalTemplateValueArgument(...args);
const escapeRegExp = (...args) => requireNormalizerHelpers().escapeRegExp(...args);
const findBalanced = (...args) => requireNormalizerHelpers().findBalanced(...args);
const parseTemplateParam = (...args) => requireNormalizerHelpers().parseTemplateParam(...args);
const rewriteTemplateParam = (...args) => requireNormalizerHelpers().rewriteTemplateParam(...args);
const skipWhitespace = (...args) => requireNormalizerHelpers().skipWhitespace(...args);
const sourceMentionsIdentifier = (...args) => requireNormalizerHelpers().sourceMentionsIdentifier(...args);
const splitTopLevel = (...args) => requireNormalizerHelpers().splitTopLevel(...args);
const splitTopLevelMacroArgs = (...args) => requireNormalizerHelpers().splitTopLevelMacroArgs(...args);
const stripLineComment = (...args) => requireNormalizerHelpers().stripLineComment(...args);
const templateDefaultEnvironment = (...args) => requireNormalizerHelpers().templateDefaultEnvironment(...args);
const templateParamDefaultValue = (...args) => requireNormalizerHelpers().templateParamDefaultValue(...args);
const vectorLaneCount = (...args) => requireNormalizerHelpers().vectorLaneCount(...args);

export function normalizeRank2CallableViews(source) {
  return normalizeRank2TemplateViewParams(normalizeRank2SharedMemoryMdspanViews(source));
}

function normalizeRank2TemplateViewParams(source) {
  let out = "";
  let cursor = 0;
  let index = 0;
  while (index < source.length) {
    const match = /\btemplate\s*</u.exec(source.slice(index));
    if (match === null) break;
    const templateStart = index + match.index;
    const open = source.indexOf("<", templateStart);
    const close = findBalanced(source, open, "<", ">");
    if (close === undefined) break;
    const brace = source.indexOf("{", close + 1);
    const semicolon = source.indexOf(";", close + 1);
    if (brace < 0) break;
    if (semicolon >= 0 && semicolon < brace) {
      index = semicolon + 1;
      continue;
    }
    const signature = source.slice(close + 1, brace);
    if (!/\b__global__\b/u.test(signature)) {
      index = brace + 1;
      continue;
    }
    const end = findBalanced(source, brace, "{", "}");
    if (end === undefined) {
      index = close + 1;
      continue;
    }
    const transformed = normalizeRank2TemplateViewKernel(
      source.slice(open + 1, close),
      signature,
      source.slice(brace + 1, end),
    );
    if (transformed === undefined) {
      index = end + 1;
      continue;
    }
    out += source.slice(cursor, templateStart);
    out += transformed;
    cursor = end + 1;
    index = end + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function normalizeRank2TemplateViewKernel(templateParamsSource, signature, body) {
  const templateParams = splitTopLevel(templateParamsSource).map(parseTemplateParam).filter(Boolean);
  const templateTypeNames = new Set(templateParams.filter((param) => param.kind === "type").map((param) => param.name));
  if (templateTypeNames.size === 0) return undefined;
  const open = signature.lastIndexOf("(");
  const close = signature.lastIndexOf(")");
  if (open < 0 || close < open) return undefined;
  const beforeParams = signature.slice(0, open);
  const params = splitTopLevel(signature.slice(open + 1, close));
  const viewParams = new Map();
  const rewrittenParams = [];
  for (const param of params) {
    const parsed = parseRank2ViewParam(param, templateTypeNames);
    if (!parsed || !rank2ViewParamIsUsed(body, parsed.name)) {
      rewrittenParams.push(param);
      continue;
    }
    viewParams.set(parsed.name, parsed);
    const pointerType = rank2ViewParamIsWritten(body, parsed.name) ? "float" : "const float";
    rewrittenParams.push(`${pointerType} *${parsed.name}, int ${parsed.name}_extent0, int ${parsed.name}_extent1, int ${parsed.name}_stride0, int ${parsed.name}_stride1`);
  }
  if (viewParams.size === 0) return undefined;
  const removedTypes = new Set([...viewParams.values()].map((param) => param.type));
  if ([...templateTypeNames].some((name) => !removedTypes.has(name))) return undefined;
  let rewrittenBody = body;
  for (const name of viewParams.keys()) {
    rewrittenBody = replaceRank2ViewExtentCalls(rewrittenBody, name);
    rewrittenBody = replaceRank2CallableViewAccesses(rewrittenBody, name, {
      base: name,
      stride0: `${name}_stride0`,
      stride1: `${name}_stride1`,
    });
  }
  return `${beforeParams}(${rewrittenParams.join(", ")}) {${rewrittenBody}}`;
}

function parseRank2ViewParam(param, templateTypeNames) {
  const match = /^\s*(?:const\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*&?\s*([A-Za-z_][A-Za-z0-9_]*)\s*$/u.exec(param);
  if (!match?.[1] || !match[2] || !templateTypeNames.has(match[1])) return undefined;
  return { type: match[1], name: match[2] };
}

function rank2ViewParamIsUsed(body, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\s*\\.\\s*extent\\s*\\(\\s*[01]\\s*\\)`, "u").test(body) &&
    new RegExp(`\\b${escapeRegExp(name)}\\s*\\(`, "u").test(body);
}

function rank2ViewParamIsWritten(body, name) {
  let cursor = 0;
  const re = new RegExp(`\\b${escapeRegExp(name)}\\s*\\(`, "gu");
  while (true) {
    const match = re.exec(body.slice(cursor));
    if (match === null) return false;
    const open = cursor + match.index + match[0].lastIndexOf("(");
    const close = findBalanced(body, open, "(", ")");
    if (close === undefined) return false;
    const next = skipWhitespace(body, close + 1);
    if (body[next] === "=" || (body[next + 1] === "=" && /[+\-*/%&|^]/u.test(body[next] ?? ""))) return true;
    cursor = close + 1;
  }
}

function replaceRank2ViewExtentCalls(source, name) {
  const escaped = escapeRegExp(name);
  return source
    .replace(new RegExp(`\\b${escaped}\\s*\\.\\s*extent\\s*\\(\\s*0\\s*\\)`, "gu"), `${name}_extent0`)
    .replace(new RegExp(`\\b${escaped}\\s*\\.\\s*extent\\s*\\(\\s*1\\s*\\)`, "gu"), `${name}_extent1`);
}

function normalizeRank2SharedMemoryMdspanViews(source) {
  const aliases = [];
  const pattern = /\bcuda\s*::\s*shared_memory_mdspan\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*cuda\s*::\s*std\s*::\s*dextents\s*<[^>{}]+>\s*\{\s*([^,{}]+?)\s*,\s*([^{}]+?)\s*\}\s*\)\s*;/gu;
  let out = source.replace(pattern, (_match, name, storage, _rows, cols) => {
    aliases.push({ name, storage, cols: cols.trim() });
    return "";
  });
  for (const alias of aliases) {
    out = replaceRank2CallableViewAccesses(out, alias.name, {
      base: alias.storage,
      stride0: alias.cols,
      stride1: "1",
    });
  }
  return out;
}

function replaceRank2CallableViewAccesses(source, name, layout) {
  let out = "";
  let cursor = 0;
  const re = new RegExp(`\\b${escapeRegExp(name)}\\s*\\(`, "gu");
  while (true) {
    const match = re.exec(source);
    if (match === null) break;
    const open = match.index + match[0].lastIndexOf("(");
    const close = findBalanced(source, open, "(", ")");
    if (close === undefined) break;
    const args = splitTopLevel(source.slice(open + 1, close));
    if (args.length !== 2) {
      re.lastIndex = open + 1;
      continue;
    }
    out += source.slice(cursor, match.index);
    out += `${layout.base}[(${args[0]}) * ${layout.stride0} + (${args[1]}) * ${layout.stride1}]`;
    cursor = close + 1;
    re.lastIndex = close + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

export function collectSyntheticVectorPackDefines(source, definesByName = new Map()) {
  const out = new Map();
  if (!sourceMentionsIdentifier(source, "x128") || definesByName.has("x128")) return out;
  const floatX = normalizeTemplateTypeArgument(definesByName.get("floatX") ?? "float", definesByName);
  if (floatX === "bf16") out.set("x128", "bg_pack128_bf168");
  else if (floatX === "half") out.set("x128", "bg_pack128_half8");
  else out.set("x128", "float4");
  return out;
}

export function normalizeTemplateValueFallbacks(source, definesByName = new Map()) {
  const sourceValueDefaults = templateDefaultEnvironment(source, definesByName);
  return source.replace(/\btemplate\s*<([^<>]*)>/gu, (match, params, offset) => {
    const parsed = splitTopLevel(params).map(parseTemplateParam);
    if (parsed.every((param) => param?.kind !== "value" || templateParamDefaultValue(param) !== undefined)) return match;
    const tail = source.slice(offset + match.length, nextTemplateDeclaration(source, offset + match.length));
    const rewritten = splitTopLevel(params).map((param, index) => {
      const parsedParam = parsed[index];
      if (parsedParam?.kind !== "value" || templateParamDefaultValue(parsedParam) !== undefined) return param.trim();
      const value = sourceValueDefaults.get(parsedParam.name) ??
        canonicalTemplateValueArgument(tail, parsedParam.name, definesByName);
      return value === undefined ? param.trim() : rewriteTemplateParam(param, value, definesByName);
    });
    return `template <${rewritten.join(", ")}>`;
  });
}

function nextTemplateDeclaration(source, start) {
  const rest = source.slice(start);
  const next = rest.search(/\btemplate\s*</u);
  return next > 0 ? start + next : source.length;
}

export function resolveTemplateDefineValue(value, definesByName) {
  const trimmed = value.trim();
  return definesByName.get(trimmed) ?? trimmed;
}

export function normalizeTemplateTypeArgument(arg, definesByName = new Map(), seen = new Set()) {
  if (arg === undefined) return undefined;
  let type = arg.trim()
    .replace(/\b(?:const|volatile|typename|class|struct)\b/gu, " ")
    .replace(/[&*]/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  if (type.length === 0 || type.includes("(") || type.includes(")")) return undefined;
  const vectorCarrier = /^vec([234])\s*<\s*([A-Za-z_][A-Za-z0-9_:]*)\s*>\s*::\s*(Type|float|double|int|uint|half)$/u.exec(type);
  if (vectorCarrier?.[1] !== undefined && vectorCarrier[2] !== undefined && vectorCarrier[3] !== undefined) {
    const scalar = vectorCarrier[3] === "Type"
      ? normalizeTemplateTypeArgument(definesByName.get(vectorCarrier[2]) ?? vectorCarrier[2], definesByName, new Set([...seen, vectorCarrier[2]]))
      : vectorCarrier[3];
    if (scalar === "float" || scalar === "double") return `float${vectorCarrier[1]}`;
    if (scalar === "int" || scalar === "uint") return `${scalar}${vectorCarrier[1]}`;
    if (scalar === "half" && vectorCarrier[1] === "2") return "half2";
    return undefined;
  }
  const packed = /^Packed128\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>$/u.exec(type)?.[1];
  if (packed !== undefined) {
    const elementType = normalizeTemplateTypeArgument(definesByName.get(packed) ?? packed, definesByName, new Set([...seen, packed]));
    if (elementType === "float") return "float4";
    if (elementType === "int") return "int4";
    if (elementType === "uint") return "uint4";
    if (elementType === "half") return "bg_pack128_half8";
    if (elementType === "bf16") return "bg_pack128_bf168";
    return undefined;
  }
  if (type.includes("<") || type.includes(">")) return undefined;
  const mapped = definesByName.get(type);
  if (mapped && /^[A-Za-z_][A-Za-z0-9_]*$/u.test(mapped)) {
    if (!seen.has(type) && mapped !== type) {
      const normalized = normalizeTemplateTypeArgument(mapped, definesByName, new Set([...seen, type]));
      if (normalized !== undefined) return normalized;
      type = mapped;
    }
  }
  if (type === "__half" || type === "half_t") return "half";
  if (type === "__nv_bfloat16" || type === "nv_bfloat16") return "bf16";
  if (type === "double") return "double";
  if (type === "unsigned int" || type === "unsigned") return "uint";
  if (type === "unsigned char" || type === "uchar" || type === "uint8_t") return "uint";
  if (type === "unsigned short" || type === "unsigned short int") return "uint";
  if (type === "signed int" || type === "signed") return "int";
  if (type === "signed short" || type === "signed short int") return "int";
  if (type === "int64_t" || type === "int32_t") return "int";
  if (type === "uint64_t" || type === "uint32_t" || type === "uintptr_t") return "uint";
  if (type === "signed char" || type === "char" || type === "int8_t") return "int";
  if (
    type === "clock_t" ||
    type === "size_t" ||
    type === "size_type" ||
    type === "curandState" ||
    type === "curandState_t" ||
    type === "curandStateSobol32" ||
    type === "curandStateSobol32_t" ||
    type === "curandStateSobol64" ||
    type === "curandStateSobol64_t" ||
    type === "curandDirectionVectors32_t" ||
    type === "curandDirectionVectors64_t" ||
    type === "CUtensorMap" ||
    type === "cudaGraphConditionalHandle" ||
    type === "__nv_fp8_storage_t" ||
    type === "__nv_fp8x2_storage_t" ||
    type === "__nv_fp8x4_storage_t"
  ) return "uint";
  if (/(?:^|_)curand|curand|(?:Direction|Vectors|rngState|State)(?:_|$)/u.test(type)) return "uint";
  if (type === "CUtexObject") return "cudaTextureObject_t";
  if (type === "CUsurfObject") return "cudaSurfaceObject_t";
  if (type === "long long" || type === "long" || type === "short" || type === "short int" || type === "ptrdiff_t") return "int";
  if (type === "uchar2") return "uint2";
  if (type === "uchar3") return "uint3";
  if (type === "uchar4") return "uint4";
  if (type === "char2") return "int2";
  if (type === "char3") return "int3";
  if (type === "char4") return "int4";
  if (type === "XMFLOAT2") return "float2";
  if (type === "XMFLOAT3") return "float3";
  if (type === "XMFLOAT4") return "float4";
  const supported = new Set(["float", "double", "int", "uint", "half", "bf16", "bool", "float2", "float3", "float4", "half2", "int2", "int3", "int4", "uint2", "uint3", "uint4"]);
  return supported.has(type) ? type : undefined;
}

export function normalizeCppTemplateCarrierSyntax(source) {
  const withBoolCarriers = source
    .replace(/\bstd\s*::\s*bool_constant\s*<\s*(true|false|[01])\s*>\s*\{\s*\}/gu, (_match, value) => value === "1" ? "true" : value === "0" ? "false" : value)
    .replace(/\bstd\s*::\s*bool_constant\s*<\s*([A-Za-z_][A-Za-z0-9_]*|[01])\s*>\s*(?=[,)])/gu, (_match, name) => `bool bg_bool_constant_${name}`);
  return rewriteBoolTemplateCarriers(withBoolCarriers);
}

export function normalizeCudaVectorLength(source) {
  const vectorSymbols = collectCudaVectorValueSymbols(source);
  if (vectorSymbols.size === 0 || !/\blength\s*\(/u.test(source)) return source;
  return source.replace(/\blength\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)/gu, (match, name) => {
    if (typeof name !== "string") return match;
    const lanes = vectorSymbols.get(name);
    if (lanes === undefined) return match;
    const terms = ["x", "y", "z", "w"]
      .slice(0, lanes)
      .map((lane) => `${name}.${lane} * ${name}.${lane}`);
    return `sqrtf(${terms.join(" + ")})`;
  });
}

function collectCudaVectorValueSymbols(source) {
  const symbols = new Map();
  const declarationRe = /\b(float[234])\s+([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()/gu;
  let match;
  while ((match = declarationRe.exec(source)) !== null) {
    const type = match[1];
    const name = match[2];
    if (type === undefined || name === undefined) continue;
    symbols.set(name, Number(type.at(-1)));
  }
  return symbols;
}

function rewriteBoolTemplateCarriers(source) {
  const edits = [];
  for (const match of source.matchAll(/\btemplate\s*<([^>]*)>\s*__global__[\s\S]*?\([^)]*\bbool\s+bg_bool_constant_([A-Za-z_][A-Za-z0-9_]*)\b[^)]*\)\s*\{/gu)) {
    const templateParams = match[1] ?? "";
    const name = match[2];
    if (!name || !new RegExp(`\\bbool\\s+${escapeRegExp(name)}\\b`, "u").test(templateParams)) continue;
    const start = match.index ?? 0;
    const brace = source.indexOf("{", start + match[0].length - 1);
    const end = findBalanced(source, brace, "{", "}");
    if (brace < 0 || end === undefined) continue;
    const body = source.slice(brace + 1, end);
    const rewritten = body.replace(new RegExp(`\\b${escapeRegExp(name)}\\b`, "gu"), `bg_bool_constant_${name}`);
    edits.push({ start: brace + 1, end, value: rewritten });
  }
  let out = source;
  for (const edit of edits.reverse()) out = `${out.slice(0, edit.start)}${edit.value}${out.slice(edit.end)}`;
  return out;
}

export function normalizeSimpleStatementMacros(source) {
  const macros = [];
  for (const line of source.split(/\r?\n/u)) {
    const match = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s+(.+?)\s*$/u.exec(stripLineComment(line));
    if (match?.[1] === undefined || match[2] === undefined || match[3] === undefined) continue;
    const body = match[3].trim();
    if (!body.includes(";") || !body.includes("=") || /[#{}\\]/u.test(body)) continue;
    const params = splitTopLevel(match[2]).map((param) => param.trim()).filter(Boolean);
    if (params.length === 0) continue;
    macros.push({ name: match[1], params, body });
  }
  if (macros.length === 0) return source;
  return source.split(/\r?\n/u).map((line) => {
    if (/^\s*#/u.test(line)) return line;
    let out = line;
    for (const macro of macros) out = replaceSimpleStatementMacroCalls(out, macro);
    return out;
  }).join("\n");
}

export function normalizeSideEffectExpressions(source) {
  return normalizePostfixIndexSideEffects(
    normalizePostfixPointerAssignments(normalizePrefixUpdateConditions(normalizeWhilePrefixUpdateConditions(source))),
  );
}

export function normalizeStdMathAliases(source) {
  return source
    .replace(/\bstd\s*::\s*isinf\s*\(/gu, "isinf(")
    .replace(/\bstd\s*::\s*numeric_limits\s*<\s*(?:float|double)\s*>\s*::\s*infinity\s*\(\s*\)/gu, "INFINITY");
}

export function normalizeStaticTemplateConverters(source, definesByName = new Map()) {
  const re = /\b[A-Za-z_][A-Za-z0-9_:]*\s*<[^;{}()]*>\s*::\s*convert\s*\(/gu;
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const openAngle = source.indexOf("<", match.index);
    const closeAngle = findBalanced(source, openAngle, "<", ">");
    const openParen = source.indexOf("(", closeAngle ?? match.index);
    const closeParen = findBalanced(source, openParen, "(", ")");
    if (closeAngle === undefined || openParen < 0 || closeParen === undefined) {
      re.lastIndex = match.index + 1;
      continue;
    }
    const rawType = source.slice(openAngle + 1, closeAngle);
    const valueType = normalizeTemplateTypeArgument(rawType, definesByName);
    const args = splitTopLevel(source.slice(openParen + 1, closeParen));
    if (valueType === undefined || args.length !== 1 || args[0]?.length === 0) {
      re.lastIndex = closeParen + 1;
      continue;
    }
    out += source.slice(cursor, match.index);
    out += `${valueType}(${args[0]})`;
    cursor = closeParen + 1;
    re.lastIndex = closeParen + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

export function stripTemplateRecordDeclarations(source) {
  const re = /\btemplate\s*<[^;{}]*>\s*(?:struct|class)\s+[A-Za-z_][A-Za-z0-9_:]*(?:\s*<[^;{}]*>)?\s*\{/gu;
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("{", match.index);
    const close = findBalanced(source, open, "{", "}");
    if (close === undefined) {
      re.lastIndex = match.index + 1;
      continue;
    }
    let end = close + 1;
    while (/\s/u.test(source[end] ?? "")) end++;
    if (source[end] !== ";") {
      re.lastIndex = close + 1;
      continue;
    }
    const body = source.slice(open + 1, close).replace(/\/\/.*$/gmu, "").replace(/\/\*[\s\S]*?\*\//gu, "").trim();
    if (body.length > 0) {
      re.lastIndex = close + 1;
      continue;
    }
    out += source.slice(cursor, match.index);
    cursor = end + 1;
    re.lastIndex = end + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

export function normalizeVectorCooperativeReductions(source) {
  const re = /\b(float[234]|int[234]|uint[234])\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:cg|cooperative_groups)\s*::\s*reduce\s*\(/gu;
  let out = "";
  let cursor = 0;
  let counter = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const vectorType = match[1];
    const target = match[2];
    const open = source.indexOf("(", match.index);
    const close = findBalanced(source, open, "(", ")");
    if (!vectorType || !target || open < 0 || close === undefined) {
      re.lastIndex = match.index + 1;
      continue;
    }
    let end = close + 1;
    while (/\s/u.test(source[end] ?? "")) end++;
    if (source[end] !== ";") {
      re.lastIndex = close + 1;
      continue;
    }
    const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim());
    if (args.length !== 3 || args.some((arg) => arg.length === 0)) {
      re.lastIndex = close + 1;
      continue;
    }
    const replacement = emitVectorCooperativeReduction(vectorType, target, args[1], args[2], counter++);
    out += source.slice(cursor, match.index);
    out += replacement;
    cursor = end + 1;
    re.lastIndex = end + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function emitVectorCooperativeReduction(vectorType, target, value, op, counter) {
  const lanes = vectorLaneCount(vectorType);
  if (lanes === undefined) return `${vectorType} ${target} = ${value};`;
  const fields = ["x", "y", "z", "w"].slice(0, lanes);
  const offset = `bg_cg_reduce_offset_${counter}`;
  const shuffled = `make_${vectorType}(${fields.map((field) =>
    `__shfl_xor_sync(0xffffffff, ${target}.${field}, ${offset})`).join(", ")})`;
  return [
    `${vectorType} ${target} = ${value};`,
    `for (int ${offset} = 16; ${offset} > 0; ${offset} /= 2) {`,
    `  ${target} = ${op}(${target}, ${shuffled});`,
    "}",
  ].join(" ");
}

export function normalizeCooperativeGroupHelperParams(source) {
  let out = "";
  let cursor = 0;
  const re = /\b__device__\b/gu;
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("{", match.index);
    if (open < 0) {
      re.lastIndex = match.index + 10;
      continue;
    }
    const close = findBalanced(source, open, "{", "}");
    if (close === undefined) {
      re.lastIndex = open + 1;
      continue;
    }
    const signature = source.slice(match.index, open);
    const body = source.slice(open, close + 1);
    if (!/\b(?:cg|cooperative_groups)\s*::\s*reduce\s*\(/u.test(body)) {
      re.lastIndex = close + 1;
      continue;
    }
    const rewritten = rewriteCooperativeGroupReduceSignature(signature, body);
    if (rewritten === signature) {
      re.lastIndex = close + 1;
      continue;
    }
    out += source.slice(cursor, match.index);
    out += rewritten;
    out += body;
    cursor = close + 1;
    re.lastIndex = close + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function rewriteCooperativeGroupReduceSignature(signature, body) {
  const close = signature.lastIndexOf(")");
  if (close < 0) return signature;
  const open = matchingOpenParen(signature, close);
  if (open === undefined) return signature;
  const params = splitTopLevel(signature.slice(open + 1, close));
  let changed = false;
  const rewritten = params.map((param) => {
    const name = parameterName(param);
    if (!name || !new RegExp(`\\b(?:cg|cooperative_groups)\\s*::\\s*reduce\\s*\\(\\s*${escapeRegExp(name)}\\b`, "u").test(body)) {
      return param;
    }
    if (/\b(?:thread_group|thread_block|thread_block_tile|coalesced_group)\b/u.test(param)) return param;
    changed = true;
    return `cooperative_groups::thread_group ${name}`;
  });
  if (!changed) return signature;
  return `${signature.slice(0, open + 1)}${rewritten.join(", ")}${signature.slice(close)}`;
}

export function matchingOpenParen(source, close) {
  let depth = 0;
  for (let i = close; i >= 0; i--) {
    const ch = source[i];
    if (ch === ")") depth++;
    else if (ch === "(") {
      depth--;
      if (depth === 0) return i;
    }
  }
  return undefined;
}

export function parameterName(param) {
  const cleaned = param.replace(/=[\s\S]*$/u, "").trim();
  return /(?:^|[\s*&])([A-Za-z_][A-Za-z0-9_]*)\s*$/u.exec(cleaned)?.[1];
}

export function normalizeForLoopScopedVariables(source) {
  let out = "";
  let cursor = 0;
  let renameIndex = 0;
  const re = /\bfor\s*\(/gu;
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("(", match.index);
    const close = findBalanced(source, open, "(", ")");
    if (open < 0 || close === undefined) {
      re.lastIndex = match.index + 3;
      continue;
    }
    const bodyStart = skipWhitespace(source, close + 1);
    const bodyEnd = source[bodyStart] === "{"
      ? findBalanced(source, bodyStart, "{", "}")
      : source.indexOf(";", bodyStart);
    if (bodyEnd === undefined || bodyEnd < 0) {
      re.lastIndex = close + 1;
      continue;
    }
    const header = source.slice(open + 1, close);
    const parts = splitTopLevel(header, ";");
    if (parts.length !== 3) {
      re.lastIndex = close + 1;
      continue;
    }
    const init = parts[0] ?? "";
    const initMatch = /^(\s*(?:const\s+|volatile\s+)*(?:unsigned\s+int|unsigned|uint|int|float|size_t|ptrdiff_t)\s+)([A-Za-z_][A-Za-z0-9_]*)([\s\S]*)$/u.exec(init);
    if (initMatch?.[1] === undefined || initMatch[2] === undefined || initMatch[3] === undefined) {
      re.lastIndex = close + 1;
      continue;
    }
    const originalName = initMatch[2];
    if (originalName.length <= 1) {
      re.lastIndex = close + 1;
      continue;
    }
    const scopedName = `bg_for_${originalName}_${renameIndex++}`;
    const renamedHeader = [
      `${initMatch[1]}${scopedName}${replaceIdentifier(initMatch[3], originalName, scopedName)}`,
      replaceIdentifier(parts[1] ?? "", originalName, scopedName),
      replaceIdentifier(parts[2] ?? "", originalName, scopedName),
    ].join(";");
    const body = source.slice(bodyStart, bodyEnd + 1);
    out += source.slice(cursor, open + 1);
    out += renamedHeader;
    out += source.slice(close, bodyStart);
    out += replaceIdentifier(body, originalName, scopedName);
    cursor = bodyEnd + 1;
    re.lastIndex = cursor;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

export function replaceIdentifier(source, from, to) {
  return source.replace(new RegExp(`\\b${escapeRegExp(from)}\\b`, "gu"), to);
}

function normalizePrefixUpdateConditions(source) {
  let out = "";
  let cursor = 0;
  const re = /\bif\s*\(/gu;
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("(", match.index);
    const close = findBalanced(source, open, "(", ")");
    if (open < 0 || close === undefined) {
      re.lastIndex = match.index + 2;
      continue;
    }
    const condition = source.slice(open + 1, close).trim();
    const normalized = normalizePrefixUpdateCondition(condition);
    if (normalized === undefined) {
      re.lastIndex = close + 1;
      continue;
    }
    out += source.slice(cursor, match.index);
    out += `${normalized.prologue}\nif (${normalized.condition})`;
    cursor = close + 1;
    re.lastIndex = close + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function normalizeWhilePrefixUpdateConditions(source) {
  let out = "";
  let cursor = 0;
  const re = /\bwhile\s*\(/gu;
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("(", match.index);
    const close = findBalanced(source, open, "(", ")");
    if (open < 0 || close === undefined) {
      re.lastIndex = match.index + 5;
      continue;
    }
    const condition = source.slice(open + 1, close).trim();
    const normalized = normalizePrefixUpdateCondition(condition);
    if (normalized === undefined) {
      re.lastIndex = close + 1;
      continue;
    }
    let bodyStart = close + 1;
    while (/\s/u.test(source[bodyStart] ?? "")) bodyStart++;
    if (source[bodyStart] !== "{") {
      re.lastIndex = close + 1;
      continue;
    }
    const bodyEnd = findBalanced(source, bodyStart, "{", "}");
    if (bodyEnd === undefined) {
      re.lastIndex = close + 1;
      continue;
    }
    const lineStart = source.lastIndexOf("\n", match.index) + 1;
    const indent = /^[ \t]*/u.exec(source.slice(lineStart, match.index))?.[0] ?? "";
    const body = source.slice(bodyStart + 1, bodyEnd);
    out += source.slice(cursor, match.index);
    out += `while (true) {\n${indent}  ${normalized.prologue}\n${indent}  if (!(${normalized.condition})) break;${body}\n${indent}}`;
    cursor = bodyEnd + 1;
    re.lastIndex = bodyEnd + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function normalizePrefixUpdateCondition(condition) {
  const logical = /^(?<op>\+\+|--)\s*(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*&&\s*(?<rest>[\s\S]+)$/u.exec(condition);
  if (logical?.groups?.op && logical.groups.name && logical.groups.rest) {
    return {
      prologue: `${logical.groups.name}${logical.groups.op};`,
      condition: `${logical.groups.name} && ${logical.groups.rest.trim()}`,
    };
  }
  const identifier = /^(?<op>\+\+|--)\s*(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?<rest>(?:[!<>=]=|[<>=])[\s\S]+)$/u.exec(condition);
  if (identifier?.groups?.op && identifier.groups.name && identifier.groups.rest) {
    return {
      prologue: `${identifier.groups.name}${identifier.groups.op};`,
      condition: `${identifier.groups.name} ${identifier.groups.rest.trim()}`,
    };
  }
  const deref = /^(?<op>\+\+|--)\s*\(\s*(?<target>\*[^)]+)\s*\)\s*(?<rest>(?:[!<>=]=|[<>=])[\s\S]+)$/u.exec(condition);
  if (deref?.groups?.op && deref.groups.target && deref.groups.rest) {
    return {
      prologue: `${deref.groups.target.trim()} ${deref.groups.op === "++" ? "+=" : "-="} 1;`,
      condition: `${deref.groups.target.trim()} ${deref.groups.rest.trim()}`,
    };
  }
  return undefined;
}

function normalizePostfixPointerAssignments(source) {
  return source.split(/\r?\n/u).map((line) => {
    const match = /^(\s*)\*(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?<op>\+\+|--)\s*=\s*(?<value>.+?);\s*$/u.exec(line);
    if (match?.groups?.name === undefined || match.groups.op === undefined || match.groups.value === undefined) return line;
    const indent = match[1] ?? "";
    return `${indent}*${match.groups.name} = ${match.groups.value.trim()};\n${indent}${match.groups.name}${match.groups.op};`;
  }).join("\n");
}

function normalizePostfixIndexSideEffects(source) {
  return source.split(/\r?\n/u).map((line) => {
    const trimmed = line.trim();
    if (
      trimmed.length === 0 ||
      trimmed.startsWith("#") ||
      /^(?:if|while|for|switch|return)\b/u.test(trimmed) ||
      (line.match(/;/gu)?.length ?? 0) !== 1 ||
      !/;\s*$/u.test(line)
    ) {
      return line;
    }
    const matches = [...line.matchAll(/\[\s*(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?<op>\+\+|--)\s*\]/gu)];
    if (matches.length !== 1) return line;
    const match = matches[0];
    const name = match?.groups?.name;
    const op = match?.groups?.op;
    if (name === undefined || op === undefined || match.index === undefined) return line;
    const withoutSideEffect = `${line.slice(0, match.index)}[${name}]${line.slice(match.index + match[0].length)}`;
    const identifierUses = [...withoutSideEffect.matchAll(new RegExp(`\\b${escapeRegExp(name)}\\b`, "gu"))];
    if (identifierUses.length !== 1) return line;
    const indent = /^(\s*)/u.exec(line)?.[1] ?? "";
    return `${withoutSideEffect}\n${indent}${name}${op};`;
  }).join("\n");
}

export function normalizeLocalReferenceAliases(source) {
  const aliases = new Map();
  let depth = 0;
  const declarationRe = /^(\s*)((?:const\s+)?(?:unsigned\s+|signed\s+)?[A-Za-z_][A-Za-z0-9_:<>]*(?:\s+[A-Za-z_][A-Za-z0-9_:<>]*)?)\s*&\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+?\[[^\];]+\](?:\s*\.[A-Za-z_][A-Za-z0-9_]*)?)\s*;\s*$/u;
  const assignmentRe = /^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*;\s*$/u;
  return source.split(/\r?\n/u).map((line) => {
    for (const [name, alias] of aliases) {
      if (alias.depth > depth) aliases.delete(name);
    }
    const declaration = declarationRe.exec(line);
    let rewritten = line;
    if (declaration?.[1] !== undefined && declaration[2] !== undefined && declaration[3] !== undefined && declaration[4] !== undefined) {
      const indent = declaration[1];
      const type = declaration[2].replace(/\bconst\b/gu, " ").replace(/\s+/gu, " ").trim();
      const name = declaration[3];
      const target = declaration[4].trim();
      aliases.set(name, { target, depth });
      rewritten = `${indent}${type} ${name} = ${target};`;
    } else {
      const assignment = assignmentRe.exec(line);
      const alias = assignment?.[2] === undefined ? undefined : aliases.get(assignment[2]);
      if (assignment?.[1] !== undefined && assignment[2] !== undefined && assignment[3] !== undefined && alias !== undefined) {
        const indent = assignment[1];
        const name = assignment[2];
        const value = assignment[3].trim();
        rewritten = `${indent}${alias.target} = ${value};\n${indent}${name} = ${value};`;
      }
    }
    depth += braceDelta(line);
    return rewritten;
  }).join("\n");
}

function braceDelta(line) {
  let delta = 0;
  for (const char of stripLineComment(line)) {
    if (char === "{") delta++;
    else if (char === "}") delta--;
  }
  return delta;
}

function replaceSimpleStatementMacroCalls(line, macro) {
  const re = new RegExp(`\\b${escapeRegExp(macro.name)}\\s*\\(`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(line)) !== null) {
    const open = line.indexOf("(", match.index + macro.name.length);
    const close = findBalanced(line, open, "(", ")");
    if (open < 0 || close === undefined) {
      re.lastIndex = match.index + macro.name.length;
      continue;
    }
    const args = splitTopLevelMacroArgs(line.slice(open + 1, close)).map((arg) => arg.trim());
    if (args.length !== macro.params.length) {
      re.lastIndex = close + 1;
      continue;
    }
    const after = line.slice(close + 1).match(/^\s*;/u);
    const expansion = expandSimpleStatementMacro(macro, args);
    out += line.slice(cursor, match.index);
    out += expansion;
    cursor = close + 1 + (after?.[0]?.length ?? 0);
    re.lastIndex = cursor;
  }
  return cursor === 0 ? line : out + line.slice(cursor);
}

function expandSimpleStatementMacro(macro, args) {
  let body = macro.body;
  for (const [index, param] of macro.params.entries()) {
    const arg = args[index] ?? "";
    body = body.replace(new RegExp(`\\b${escapeRegExp(param)}\\b`, "gu"), arg);
  }
  return body.endsWith(";") ? body : `${body};`;
}
