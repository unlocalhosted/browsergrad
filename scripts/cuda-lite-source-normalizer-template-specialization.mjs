import { requireNormalizerHelpers } from "./cuda-lite-source-normalizer-context.mjs";

const balancedParenContents = (...args) => requireNormalizerHelpers().balancedParenContents(...args);
const collectDeclaredIdentifiers = (...args) => requireNormalizerHelpers().collectDeclaredIdentifiers(...args);
const escapeRegExp = (...args) => requireNormalizerHelpers().escapeRegExp(...args);
const evaluateTemplateIntegerExpression = (...args) => requireNormalizerHelpers().evaluateTemplateIntegerExpression(...args);
const findBalanced = (...args) => requireNormalizerHelpers().findBalanced(...args);
const isKnownCarrierAlias = (...args) => requireNormalizerHelpers().isKnownCarrierAlias(...args);
const isMacroIdentifier = (...args) => requireNormalizerHelpers().isMacroIdentifier(...args);
const isTemplateSymbolArgument = (...args) => requireNormalizerHelpers().isTemplateSymbolArgument(...args);
const kernelDefinitionName = (...args) => requireNormalizerHelpers().kernelDefinitionName(...args);
const mergeDefineMaps = (...args) => requireNormalizerHelpers().mergeDefineMaps(...args);
const normalizeConstexprIntegerExpression = (...args) => requireNormalizerHelpers().normalizeConstexprIntegerExpression(...args);
const normalizeTemplateTypeArgument = (...args) => requireNormalizerHelpers().normalizeTemplateTypeArgument(...args);
const resolveTemplateDefineValue = (...args) => requireNormalizerHelpers().resolveTemplateDefineValue(...args);
const splitTopLevel = (...args) => requireNormalizerHelpers().splitTopLevel(...args);
const substituteTemplateTypes = (...args) => requireNormalizerHelpers().substituteTemplateTypes(...args);
const templateHeaders = (...args) => requireNormalizerHelpers().templateHeaders(...args);

export function specializeTemplateFromLaunchContext(source, templateArgumentsByKernelName, definesByName = new Map()) {
  const name = kernelDefinitionName(source);
  if (name === undefined) return source;
  const args = templateArgumentsByKernelName.get(name);
  if (args === undefined || args.length === 0) return rewriteFirstTemplateHeader(source, [], definesByName);
  return rewriteFirstTemplateHeader(source, args, definesByName);
}

export function specializeDeviceFunctionFromCallContext(source, args, definesByName = new Map()) {
  const defaults = templateHeaderDefaultArguments(source, definesByName);
  if ((args === undefined || args.length === 0) && defaults.length === 0) return rewriteFirstTemplateHeader(source, [], definesByName);
  const merged = defaults.map((value, index) => args?.[index] ?? value);
  if (args !== undefined) {
    for (let index = 0; index < args.length; index++) merged[index] = args[index];
  }
  return rewriteFirstTemplateHeader(source, merged, definesByName);
}

function rewriteFirstTemplateHeader(source, args, definesByName = new Map()) {
  const templateStart = source.search(/\btemplate\s*</u);
  if (templateStart < 0) return source;
  const open = source.indexOf("<", templateStart);
  const close = findBalanced(source, open, "<", ">");
  if (close === undefined) return source;
  const params = splitTopLevel(source.slice(open + 1, close));
  if (params.length === 0) return source;
  const parsedParams = params.map(parseTemplateParam);
  const effectiveArgs = canonicalTemplateFallbackArguments(source.slice(close + 1), parsedParams, args, definesByName);
  const typeEnv = templateTypeEnvironment(parsedParams, effectiveArgs, definesByName);
  const valueEnv = templateValueEnvironment(parsedParams, effectiveArgs, definesByName);
  const rewritten = params.map((param, index) => rewriteTemplateParam(param, effectiveArgs[index], definesByName)).join(", ");
  const body = substituteTemplateValues(substituteTemplateTypes(source.slice(close), typeEnv), valueEnv);
  return `${source.slice(0, open + 1)}${rewritten}${body}`;
}

function canonicalTemplateFallbackArguments(sourceTail, parsedParams, args, definesByName = new Map()) {
  const out = [...args];
  for (let index = 0; index < parsedParams.length; index++) {
    const param = parsedParams[index];
    if (param === undefined) continue;
    const hasDefault = templateParamDefaultValue(param) !== undefined;
    const unresolvedSelfArgument = hasConcreteTemplateArgument(out[index]) && String(out[index]).trim() === param.name;
    if (hasDefault && !unresolvedSelfArgument) continue;
    if (hasConcreteTemplateArgument(out[index]) && String(out[index]).trim() !== param.name) continue;
    if (param.kind === "type") {
      if (templateTypeParamUsedAsPointerParam(sourceTail, param.name)) out[index] = canonicalTemplatePointerType(sourceTail, param.name, definesByName);
      else {
        const scalarType = canonicalTemplateScalarType(sourceTail, param.name, definesByName);
        if (scalarType !== undefined) out[index] = scalarType;
      }
      continue;
    }
    const value = canonicalTemplateValueArgument(sourceTail, param.name, definesByName);
    if (value !== undefined) out[index] = value;
  }
  return out;
}

function hasConcreteTemplateArgument(value) {
  return value !== undefined && String(value).trim().length > 0;
}

function templateTypeParamUsedAsPointerParam(sourceTail, name) {
  const signatureEnd = sourceTail.indexOf("{");
  if (signatureEnd < 0) return false;
  const signature = sourceTail.slice(0, signatureEnd);
  const open = signature.indexOf("(");
  const params = balancedParenContents(signature, open);
  if (params === undefined) return false;
  return splitTopLevel(params).some((param) => {
    const cleaned = param
      .replace(/\b(?:const|volatile|__restrict__|__restrict|restrict)\b/gu, " ")
      .replace(/\s+/gu, " ")
      .trim();
    const escaped = escapeRegExp(name);
    return new RegExp(`^${escaped}\\s*(?:\\*|&)|^${escaped}\\s+[*&]`, "u").test(cleaned);
  });
}

function canonicalTemplatePointerType(sourceTail, name, definesByName = new Map()) {
  const mapped = normalizeTemplateTypeArgument(definesByName.get(name), definesByName);
  if (mapped !== undefined) return mapped;
  const pointerParams = templateTypePointerParamNames(sourceTail, name);
  if (/(?:rng|curand|direction)/iu.test(name) || pointerParams.some((param) => /(?:rng|curand|direction|state)/iu.test(param))) return "uint";
  if (pointerParams.some((param) => /(?:count|num|size|len|idx|index|offset|addr|tid|id|mask|flag)/iu.test(param))) return "uint";
  const candidate = normalizeTemplateTypeArgument(definesByName.get("floatX") ?? "float", definesByName);
  return candidate === "half" || candidate === "bf16" ? "float" : candidate ?? "float";
}

function templateTypePointerParamNames(sourceTail, name) {
  const signatureEnd = sourceTail.indexOf("{");
  if (signatureEnd < 0) return [];
  const signature = sourceTail.slice(0, signatureEnd);
  const open = signature.indexOf("(");
  const params = balancedParenContents(signature, open);
  if (params === undefined) return [];
  const escaped = escapeRegExp(name);
  return splitTopLevel(params).flatMap((param) => {
    const cleaned = param
      .replace(/\b(?:const|volatile|__restrict__|__restrict|restrict)\b/gu, " ")
      .replace(/\s+/gu, " ")
      .trim();
    const match = new RegExp(`^${escaped}\\s*(?:\\*|&)\\s*(?:const\\s+|volatile\\s+)*([A-Za-z_][A-Za-z0-9_]*)\\b|^${escaped}\\s+[*&]\\s*(?:const\\s+|volatile\\s+)*([A-Za-z_][A-Za-z0-9_]*)\\b`, "u").exec(cleaned);
    return match?.[1] ?? match?.[2] ?? [];
  });
}

function canonicalTemplateScalarType(sourceTail, name, definesByName = new Map()) {
  const mapped = normalizeTemplateTypeArgument(definesByName.get(name), definesByName);
  if (mapped !== undefined) return mapped;
  const params = templateTypeScalarParamNames(sourceTail, name);
  if (params.length === 0) return undefined;
  if (params.some((param) => /(?:count|num|size|len|idx|index|offset|addr|tid|id|mask|flag)/iu.test(param))) return "uint";
  return undefined;
}

function templateTypeScalarParamNames(sourceTail, name) {
  const signatureEnd = sourceTail.indexOf("{");
  if (signatureEnd < 0) return [];
  const signature = sourceTail.slice(0, signatureEnd);
  const open = signature.indexOf("(");
  const params = balancedParenContents(signature, open);
  if (params === undefined) return [];
  const escaped = escapeRegExp(name);
  return splitTopLevel(params).flatMap((param) => {
    const cleaned = param
      .replace(/\b(?:const|volatile|__restrict__|__restrict|restrict)\b/gu, " ")
      .replace(/\s+/gu, " ")
      .trim();
    if (new RegExp(`^${escaped}\\s*(?:\\*|&)|^${escaped}\\s+[*&]`, "u").test(cleaned)) return [];
    return new RegExp(`^${escaped}\\s+([A-Za-z_][A-Za-z0-9_]*)\\b`, "u").exec(cleaned)?.[1] ?? [];
  });
}

export function canonicalTemplateValueArgument(sourceTail, name, definesByName = new Map()) {
  const escaped = escapeRegExp(name);
  const launchBounds = new RegExp(`\\b__launch_bounds__\\s*\\(\\s*${escaped}\\b`, "u").test(sourceTail);
  const sharedSized = new RegExp(`\\b__shared__\\b[\\s\\S]{0,160}\\[\\s*${escaped}\\s*\\]`, "u").test(sourceTail);
  const sharedPipelineContext = sharedSized ||
    /\b(?:extern\s+)?__shared__\b|cp\.async|ldmatrix|mma\.sync|wgmma|pipeline|smem|shared/iu.test(sourceTail);
  const sharedPipelineFallback = sharedPipelineContext ? canonicalSharedPipelineTemplateValue(name) : undefined;
  if (sharedPipelineFallback !== undefined) return sharedPipelineFallback;
  if (!launchBounds && !sharedSized && !/\bblock/i.test(name)) return undefined;
  const blockSize = normalizeTemplateValueArgument(definesByName.get("block_size") ?? definesByName.get("BLOCK_SIZE") ?? "256", "int");
  return blockSize === undefined ? undefined : String(blockSize);
}

function canonicalSharedPipelineTemplateValue(name) {
  if (/(?:^|_)(?:k)?stage(?:s)?$/iu.test(name) || /(?:^|_)num_?stages?$/iu.test(name)) return "2";
  if (/(?:^|_)(?:a_?|b_?)?pad(?:ding)?$/iu.test(name)) return "0";
  if (/(?:^|_)warp_?swizzle$/iu.test(name)) return "0";
  return undefined;
}

export function rewriteTemplateParam(param, arg, definesByName = new Map()) {
  if (arg === undefined) return param.trim();
  const cleaned = param.trim();
  if (cleaned.length === 0) return cleaned;
  const parsed = parseTemplateParam(cleaned);
  if (parsed === undefined) return cleaned;
  if (parsed.kind === "type") {
    const value = normalizeTemplateTypeArgument(arg, definesByName);
    if (value === undefined) return cleaned;
    if (parsed.defaultStart === undefined) return `${cleaned} = ${value}`;
    return `${cleaned.slice(0, parsed.defaultStart).trimEnd()} = ${value}`;
  }
  if (parsed.kind === "symbol") {
    const value = normalizeTemplateSymbolArgument(arg);
    if (value === undefined) return cleaned;
    if (parsed.defaultStart === undefined) return `${cleaned} = ${value}`;
    return `${cleaned.slice(0, parsed.defaultStart).trimEnd()} = ${value}`;
  }
  const value = normalizeTemplateValueArgument(arg, parsed.valueType);
  const resolvedValue = normalizeTemplateValueArgument(resolveTemplateDefineValue(arg, definesByName), parsed.valueType);
  if (resolvedValue !== undefined) {
    if (parsed.defaultStart === undefined) return `${cleaned} = ${resolvedValue}`;
    return `${cleaned.slice(0, parsed.defaultStart).trimEnd()} = ${resolvedValue}`;
  }
  if (value === undefined) return cleaned;
  if (parsed.defaultStart === undefined) return `${cleaned} = ${value}`;
  return `${cleaned.slice(0, parsed.defaultStart).trimEnd()} = ${value}`;
}

export function parseTemplateParam(param) {
  const withoutDefault = splitTemplateParamDefault(param);
  const head = withoutDefault.head
    .replace(/\b(?:const|constexpr)\b/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  const valueMatch = /^(?:(?:unsigned\s+)?(?:int|long|short)|uint|size_t|ptrdiff_t|bool)\s+([A-Za-z_][A-Za-z0-9_]*)$/u.exec(head);
  if (valueMatch?.[1] !== undefined) {
    return {
      kind: "value",
      name: valueMatch[1],
      valueType: /\bbool\b/u.test(head) ? "bool" : "int",
      defaultStart: withoutDefault.defaultStart,
      source: param,
    };
  }
  const typeMatch = /^(?:typename|class)\s+([A-Za-z_][A-Za-z0-9_]*)$/u.exec(head);
  if (typeMatch?.[1] !== undefined) {
    return { kind: "type", name: typeMatch[1], defaultStart: withoutDefault.defaultStart, source: param };
  }
  const symbolMatch = /^[A-Za-z_][A-Za-z0-9_:]*\s+([A-Za-z_][A-Za-z0-9_]*)$/u.exec(head);
  if (symbolMatch?.[1] !== undefined) {
    return { kind: "symbol", name: symbolMatch[1], defaultStart: withoutDefault.defaultStart, source: param };
  }
  return undefined;
}

function splitTemplateParamDefault(param) {
  let angle = 0;
  let paren = 0;
  let bracket = 0;
  for (let index = 0; index < param.length; index++) {
    const char = param[index];
    if (char === "<") angle++;
    else if (char === ">") angle = Math.max(0, angle - 1);
    else if (char === "(") paren++;
    else if (char === ")") paren = Math.max(0, paren - 1);
    else if (char === "[") bracket++;
    else if (char === "]") bracket = Math.max(0, bracket - 1);
    else if (char === "=" && angle === 0 && paren === 0 && bracket === 0) {
      return { head: param.slice(0, index), defaultStart: index };
    }
  }
  return { head: param, defaultStart: undefined };
}

export function normalizeTemplateValueArgument(arg, valueType) {
  if (arg === undefined) return undefined;
  const value = arg.trim().replace(/[uUlL]+$/u, "");
  if (valueType === "bool") {
    if (value === "true") return "1";
    if (value === "false") return "0";
  }
  if (/^(?:true|false)$/u.test(value)) return value === "true" ? "1" : "0";
  if (/^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$/u.test(value)) return undefined;
  if (!/^[0-9A-Fa-fxX\s()+\-*/%<>&|^?:.]+$/u.test(value)) return undefined;
  return value;
}

export function normalizeTemplateSymbolArgument(arg) {
  if (arg === undefined) return undefined;
  const value = arg.trim();
  return isTemplateSymbolArgument(value) ? value : undefined;
}

export function templateTypeEnvironment(params, args, definesByName = new Map()) {
  const env = new Map();
  for (let index = 0; index < params.length; index++) {
    const param = params[index];
    if (param?.kind !== "type") continue;
    const raw = args[index] ?? templateParamDefaultValue(param);
    const value = normalizeTemplateTypeArgument(raw, definesByName);
    if (value !== undefined) env.set(param.name, value);
    else if (raw !== undefined && isKnownCarrierAlias(raw, definesByName)) env.set(param.name, raw.trim());
  }
  return env;
}

export function templateValueEnvironment(params, args, definesByName = new Map()) {
  const env = new Map();
  for (let index = 0; index < params.length; index++) {
    const param = params[index];
    if (param?.kind !== "value" && param?.kind !== "symbol") continue;
    const raw = args[index] ?? templateParamDefaultValue(param);
    const value = raw === undefined ? undefined : param.kind === "symbol"
      ? normalizeTemplateSymbolArgument(resolveTemplateDefineValue(raw, definesByName))
      : normalizeTemplateValueArgument(resolveTemplateDefineValue(raw, definesByName), param.valueType);
    if (value !== undefined) env.set(param.name, value);
  }
  return env;
}

export function substituteTemplateValues(source, valueEnv) {
  if (valueEnv.size === 0) return source;
  return source.replace(/\b[A-Za-z_][A-Za-z0-9_]*\b/gu, (name) => valueEnv.get(name) ?? name);
}

export function templateDefaultEnvironment(source, definesByName = new Map()) {
  const env = new Map();
  for (const [name, value] of numericTemplateDefines(definesByName)) env.set(name, value);
  for (const header of templateHeaders(source)) {
    for (const paramSource of splitTopLevel(header)) {
      const param = parseTemplateParam(paramSource);
      if (param === undefined) continue;
      const raw = templateParamDefaultValue(param);
      if (raw === undefined) continue;
      const value = param.kind === "type"
        ? normalizeTemplateTypeArgument(raw, definesByName)
        : normalizeTemplateValueArgument(raw, param.valueType);
      if (value !== undefined) env.set(param.name, value);
    }
  }
  return env;
}

export function templateHeaderDefaultArguments(source, definesByName = new Map()) {
  const header = templateHeaders(source)[0];
  if (header === undefined) return [];
  return splitTopLevel(header).map((paramSource) => {
    const param = parseTemplateParam(paramSource);
    if (param === undefined) return undefined;
    const raw = templateParamDefaultValue(param);
    if (raw === undefined) return undefined;
    return param.kind === "type"
      ? normalizeTemplateTypeArgument(raw, definesByName)
      : normalizeTemplateValueArgument(resolveTemplateDefineValue(raw, definesByName), param.valueType);
  });
}

export function templateParamDefaultValue(param) {
  if (param.defaultStart === undefined || param.source === undefined) return undefined;
  return param.source.slice(param.defaultStart + 1).trim();
}

export function numericTemplateDefines(definesByName) {
  const values = new Map();
  for (let pass = 0; pass < definesByName.size; pass++) {
    let changed = false;
    const env = mergeDefineMaps(definesByName, values);
    for (const [name, raw] of definesByName) {
      if (values.has(name)) continue;
      const normalizedSource = normalizeConstexprIntegerExpression(String(raw));
      const direct = normalizeTemplateValueArgument(normalizedSource, "int");
      const evaluated = evaluateTemplateIntegerExpression(normalizedSource, env);
      const normalized = evaluated ?? direct;
      if (normalized !== undefined) {
        values.set(name, normalized);
        changed = true;
      }
    }
    if (!changed) break;
  }
  return values;
}

export function normalizeNumericObjectDefines(source, definesByName, blockedNames = new Set()) {
  const entries = [...numericTemplateDefines(definesByName)]
    .filter(([name]) => isMacroIdentifier(name) && !blockedNames.has(name))
    .sort((a, b) => b[0].length - a[0].length);
  if (entries.length === 0) return source;
  const localBlocked = new Set(blockedNames);
  return source.split(/\r?\n/u).map((line) => {
    if (/^\s*#/u.test(line)) return line;
    const declared = collectDeclaredIdentifiers(line, definesByName);
    let out = line;
    for (const [name, value] of entries) {
      if (localBlocked.has(name) || declared.has(name)) continue;
      out = out.replace(new RegExp(`(?<![.:])\\b${escapeRegExp(name)}\\b(?!\\s*\\()`, "gu"), value);
    }
    for (const name of declared) localBlocked.add(name);
    return out;
  }).join("\n");
}

export function normalizeSupportedTypeDefineReferences(source, definesByName, blockedNames = new Set()) {
  const entries = [...definesByName]
    .map(([name, value]) => [name, normalizeTemplateTypeArgument(value, definesByName)])
    .filter(([name, value]) => value !== undefined && value !== name && isMacroIdentifier(name))
    .filter(([name]) => !new RegExp(`\\b${escapeRegExp(name)}\\s*::`, "u").test(source))
    .sort((a, b) => b[0].length - a[0].length);
  if (entries.length === 0) return source;
  const localBlocked = new Set(blockedNames);
  return source.split(/\r?\n/u).map((line) => {
    if (/^\s*#/u.test(line)) return line;
    const declared = collectDeclaredIdentifiers(line, definesByName);
    let out = line;
    for (const [name, value] of entries) {
      if (localBlocked.has(name) || declared.has(name)) continue;
      out = out.replace(new RegExp(`\\b${escapeRegExp(name)}\\b`, "gu"), value);
    }
    for (const name of declared) localBlocked.add(name);
    return out;
  }).join("\n");
}
