let activeHelpers;

function withNormalizerHelpers(helpers, work) {
  const previous = activeHelpers;
  activeHelpers = helpers;
  try {
    return work();
  } finally {
    activeHelpers = previous;
  }
}

function requireHelpers() {
  if (activeHelpers === undefined) throw new Error("cuda-lite source normalizer helpers were not provided");
  return activeHelpers;
}

const addressTarget = (...args) => requireHelpers().addressTarget(...args);
const balancedParenContents = (...args) => requireHelpers().balancedParenContents(...args);
const escapeRegExp = (...args) => requireHelpers().escapeRegExp(...args);
const findBalanced = (...args) => requireHelpers().findBalanced(...args);
const replaceBalancedCall = (...args) => requireHelpers().replaceBalancedCall(...args);
const skipWhitespace = (...args) => requireHelpers().skipWhitespace(...args);
const splitTopLevel = (...args) => requireHelpers().splitTopLevel(...args);
const stripCommentsAndStrings = (...args) => requireHelpers().stripCommentsAndStrings(...args);

export function normalizeCurandInitOverloads(source, helpers) {
  return withNormalizerHelpers(helpers, () => {
    const helper = /\bcurand_init\s*\(/gu;
    let out = "";
    let cursor = 0;
    let match;
    while ((match = helper.exec(source)) !== null) {
      const open = source.indexOf("(", match.index);
      const close = findBalanced(source, open, "(", ")");
      if (open < 0 || close === undefined) {
        helper.lastIndex = match.index + 1;
        continue;
      }
      const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim()).filter(Boolean);
      if (args.length !== 3 || !/^&/u.test(args[2] ?? "")) {
        helper.lastIndex = close + 1;
        continue;
      }
      out += source.slice(cursor, open + 1);
      out += `${args[0]}, ${args[1]}, 0, ${args[2]}`;
      cursor = close;
      helper.lastIndex = close + 1;
    }
    return cursor === 0 ? source : out + source.slice(cursor);
  });
}

export function normalizeSincosHelpers(source, helpers) {
  return withNormalizerHelpers(helpers, () => {
    const helper = /\b__sincosf\s*\(/gu;
    let out = "";
    let cursor = 0;
    let match;
    while ((match = helper.exec(source)) !== null) {
      const open = source.indexOf("(", match.index);
      const close = findBalanced(source, open, "(", ")");
      if (open < 0 || close === undefined) {
        helper.lastIndex = match.index + 1;
        continue;
      }
      const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim());
      const replacement = sincosReplacement(args);
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
  });
}

export function normalizeDeviceReferenceParams(source, helpers) {
  return withNormalizerHelpers(helpers, () => {
    const functions = scanDeviceReferenceFunctions(source);
    if (functions.length === 0) return source;
    const withReferenceCalls = rewriteDeviceReferenceCalls(source, functions);
    return rewriteDeviceReferenceDefinitions(withReferenceCalls);
  });
}

function rewriteDeviceReferenceDefinitions(source) {
  const functions = scanDeviceReferenceFunctions(source);
  if (functions.length === 0) return source;
  let out = "";
  let cursor = 0;
  for (const fn of functions) {
    out += source.slice(cursor, fn.paramsStart);
    out += fn.params.map((param) => param.reference === undefined
      ? param.source
      : `${param.reference.type} *${param.reference.name}${param.reference.defaultValue ?? ""}`
    ).join(", ");
    out += source.slice(fn.paramsEnd, fn.bodyStart + 1);
    out += rewriteReferenceParamBody(source.slice(fn.bodyStart + 1, fn.bodyEnd), new Set(fn.params
      .map((param) => param.reference?.name)
      .filter(Boolean)));
    cursor = fn.bodyEnd;
  }
  return out + source.slice(cursor);
}

function rewriteDeviceReferenceCalls(source, functions) {
  let out = source;
  for (const fn of functions) {
    const referenceIndexes = fn.params
      .map((param, index) => param.reference === undefined ? -1 : index)
      .filter((index) => index >= 0);
    if (referenceIndexes.length === 0) continue;
    const re = new RegExp(`\\b${escapeRegExp(fn.name)}\\s*\\(`, "gu");
    out = replaceBalancedCall(out, re, (match, open, close) => {
      if (isInsideCommentOrString(out, match.index) || looksLikeFunctionDefinitionCall(out, match.index, close)) return undefined;
      const args = splitTopLevel(out.slice(open + 1, close)).map((arg) => arg.trim());
      if (args.length !== fn.params.length) return undefined;
      for (const index of referenceIndexes) {
        const arg = args[index];
        if (arg === undefined || arg.length === 0 || arg.startsWith("&")) continue;
        args[index] = referenceArgument(arg);
      }
      return `${fn.name}(${args.join(", ")})`;
    });
  }
  return out;
}

function scanDeviceReferenceFunctions(source) {
  return scanDeviceFunctions(source).map((fn) => ({
    ...fn,
    params: fn.params.map((param) => {
      const parsed = parseDeviceReferenceParam(param);
      if (parsed.reference === undefined) return parsed;
      return referenceParamNeedsPointerRewrite(fn.body, parsed.reference.name) ? parsed : { source: param };
    }),
  })).filter((fn) => fn.params.some((param) => param.reference !== undefined));
}

function scanDeviceFunctions(source) {
  const functions = [];
  let index = 0;
  while (index < source.length) {
    const match = /\b__device__\b/u.exec(source.slice(index));
    if (match === null) break;
    const deviceStart = index + match.index;
    const open = source.indexOf("(", deviceStart);
    const close = findBalanced(source, open, "(", ")");
    if (open < 0 || close === undefined) {
      index = deviceStart + "__device__".length;
      continue;
    }
    const signature = source.slice(deviceStart, open);
    const name = /([A-Za-z_][A-Za-z0-9_]*)\s*$/u.exec(signature)?.[1];
    const afterParams = skipWhitespace(source, close + 1);
    if (name === undefined || source[afterParams] !== "{") {
      index = close + 1;
      continue;
    }
    const end = findBalanced(source, afterParams, "{", "}");
    if (end === undefined) {
      index = close + 1;
      continue;
    }
    functions.push({
      name,
      paramsStart: open + 1,
      paramsEnd: close,
      bodyStart: afterParams,
      bodyEnd: end,
      body: source.slice(afterParams + 1, end),
      params: splitTopLevel(source.slice(open + 1, close)),
    });
    index = end + 1;
  }
  return functions;
}

function parseDeviceReferenceParam(source) {
  const defaultMatch = /\s*=\s*[\s\S]*$/u.exec(source);
  const defaultValue = defaultMatch?.[0];
  const core = (defaultValue === undefined ? source : source.slice(0, defaultMatch.index)).trim();
  if (!/(^|[^&])&([^&]|$)/u.test(core)) return { source };
  const match = /^([\s\S]*?)&\s*([A-Za-z_][A-Za-z0-9_]*)\s*$/u.exec(core);
  if (match?.[1] === undefined || match[2] === undefined) return { source };
  const type = match[1].trim();
  if (type.length === 0 || type.includes("*")) return { source };
  if (!isIntegerReferenceRewriteType(type)) return { source };
  return {
    source,
    reference: {
      type,
      name: match[2],
      ...(defaultValue === undefined ? {} : { defaultValue }),
    },
  };
}

function referenceParamNeedsPointerRewrite(body, name) {
  return new RegExp(`\\batomic[A-Za-z0-9_]*\\s*\\([^;{}]*&\\s*${escapeRegExp(name)}\\b`, "u")
    .test(stripCommentsAndStrings(body));
}

function isIntegerReferenceRewriteType(type) {
  const normalized = type
    .replace(/\b(?:volatile|register|__restrict__|restrict)\b/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  if (/\bconst\b/u.test(normalized)) return false;
  return /^(?:u?int|unsigned|signed|unsigned int|signed int|uint32_t|int32_t)$/u.test(normalized);
}

function rewriteReferenceParamBody(body, referenceNames) {
  if (referenceNames.size === 0) return body;
  let out = "";
  let cursor = 0;
  let index = 0;
  while (index < body.length) {
    const skipped = skipTrivia(body, index);
    if (skipped !== index) {
      index = skipped;
      continue;
    }
    if (body[index] === "&" && body[index + 1] !== "&") {
      const nameStart = skipWhitespace(body, index + 1);
      const name = readIdentifierAt(body, nameStart);
      if (name !== undefined && referenceNames.has(name.value)) {
        out += body.slice(cursor, index);
        out += name.value;
        cursor = name.end;
        index = name.end;
        continue;
      }
    }
    const name = readIdentifierAt(body, index);
    if (name !== undefined) {
      if (referenceNames.has(name.value)) {
        out += body.slice(cursor, index);
        out += `(*${name.value})`;
        cursor = name.end;
      }
      index = name.end;
      continue;
    }
    index++;
  }
  return cursor === 0 ? body : out + body.slice(cursor);
}

function referenceArgument(arg) {
  return /^[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\[[^\]]+\]|\.[A-Za-z_][A-Za-z0-9_]*))*$/u.test(arg) ||
    /^\*\s*[A-Za-z_][A-Za-z0-9_]*$/u.test(arg)
    ? `&${arg}`
    : `&(${arg})`;
}

function looksLikeFunctionDefinitionCall(source, nameStart, close) {
  if (source[skipWhitespace(source, close + 1)] !== "{") return false;
  const lineStart = source.lastIndexOf("\n", nameStart) + 1;
  return /\b__device__\b/u.test(source.slice(lineStart, nameStart));
}

function isInsideCommentOrString(source, target) {
  let index = 0;
  while (index < target) {
    const skipped = skipTrivia(source, index);
    if (skipped !== index) {
      if (target < skipped) return true;
      index = skipped;
      continue;
    }
    index++;
  }
  return false;
}

function skipTrivia(source, index) {
  if (source[index] === "/" && source[index + 1] === "/") {
    const end = source.indexOf("\n", index + 2);
    return end < 0 ? source.length : end;
  }
  if (source[index] === "/" && source[index + 1] === "*") {
    const end = source.indexOf("*/", index + 2);
    return end < 0 ? source.length : end + 2;
  }
  if (source[index] === "\"" || source[index] === "'") {
    const quote = source[index];
    let cursor = index + 1;
    while (cursor < source.length) {
      if (source[cursor] === "\\") {
        cursor += 2;
        continue;
      }
      if (source[cursor] === quote) return cursor + 1;
      cursor++;
    }
    return source.length;
  }
  return index;
}

function readIdentifierAt(source, index) {
  const match = /^[A-Za-z_][A-Za-z0-9_]*/u.exec(source.slice(index));
  if (match === null) return undefined;
  return { value: match[0], end: index + match[0].length };
}

export function normalizeBlockReduceHelpers(source, helpers) {
  return withNormalizerHelpers(helpers, () => {
    const helper = /\bblockReduce\s*</gu;
    let out = "";
    let cursor = 0;
    let match;
    while ((match = helper.exec(source)) !== null) {
      const templateStart = source.indexOf("<", match.index);
      const templateEnd = findBalanced(source, templateStart, "<", ">");
      if (templateStart < 0 || templateEnd === undefined) {
        helper.lastIndex = match.index + 1;
        continue;
      }
      const open = skipWhitespace(source, templateEnd + 1);
      const close = source[open] === "(" ? findBalanced(source, open, "(", ")") : undefined;
      if (close === undefined) {
        helper.lastIndex = templateEnd + 1;
        continue;
      }
      const reducer = normalizeBlockReducerName(source.slice(templateStart + 1, templateEnd).trim());
      const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim()).filter(Boolean);
      if (reducer === undefined || args.length === 0) {
        helper.lastIndex = close + 1;
        continue;
      }
      out += source.slice(cursor, match.index);
      out += `${reducer}(${args[0]})`;
      cursor = close + 1;
      helper.lastIndex = close + 1;
    }
    return cursor === 0 ? source : out + source.slice(cursor);
  });
}

export function normalizeAtomicForwarderHelpers(source, helpers) {
  return withNormalizerHelpers(helpers, () => {
    const helpers = collectAtomicForwarderHelpers(source);
    if (helpers.size === 0) return source;
    let out = source;
    for (const [name, builtin] of helpers) {
      out = out.replace(builtin.definition, "");
      out = rewriteSimpleCallName(out, name, builtin.target);
    }
    return out;
  });
}

function collectAtomicForwarderHelpers(source) {
  const helpers = new Map();
  const re = /(?:template\s*<[^<>]*>\s*)?(?:__device__|__host__|__forceinline__|__inline__|inline|static|\s)+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;{}]*)\)\s*;\s*\}/gu;
  for (const match of source.matchAll(re)) {
    const [definition, name, paramsSource, target, argsSource] = match;
    if (!definition || !name || !paramsSource || !target || !argsSource || !isAtomicBuiltinName(target)) continue;
    const params = splitTopLevel(paramsSource).map((param) => /([A-Za-z_][A-Za-z0-9_]*)\s*$/.exec(param.trim())?.[1]).filter(Boolean);
    const args = splitTopLevel(argsSource).map((arg) => arg.trim());
    if (params.length === 0 || params.length !== args.length) continue;
    if (!args.every((arg, index) => arg === params[index])) continue;
    helpers.set(name, { target, definition });
  }
  return helpers;
}

function isAtomicBuiltinName(name) {
  return /^atomic(?:Add|Add_system|Sub|Min|Min_system|Max|Max_system|MaxFloat|And|And_system|Or|Or_system|Xor|Xor_system|Inc|Inc_system|Dec|Dec_system|Exch|Exch_system|CAS|CAS_system)$/u.test(name);
}

function rewriteSimpleCallName(source, name, replacement) {
  return source.replace(new RegExp(`\\b${escapeRegExp(name)}\\s*\\(`, "gu"), `${replacement}(`);
}

export function normalizePointerStoreForwarderHelpers(source, helpers) {
  return withNormalizerHelpers(helpers, () => {
    const helpers = collectPointerStoreForwarderHelpers(source);
    if (helpers.size === 0) return source;
    let out = source;
    for (const [name, helper] of helpers) {
      out = out.replace(helper.definition, "");
      out = rewritePointerStoreForwarderCalls(out, name, helper);
    }
    return out;
  });
}

function collectPointerStoreForwarderHelpers(source) {
  const helpers = new Map();
  const re = /(?:template\s*<[^<>]*>\s*)?(?:__device__|__host__|__forceinline__|__inline__|inline|static|\s)+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*\{\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*;\s*\}/gu;
  for (const match of source.matchAll(re)) {
    const [definition, name, paramsSource, pointerName, valueName] = match;
    if (!definition || !name || !paramsSource || !pointerName || !valueName) continue;
    const params = splitTopLevel(paramsSource).map((param) => ({
      name: /([A-Za-z_][A-Za-z0-9_]*)\s*$/.exec(param.trim())?.[1],
      pointer: /\*/u.test(param),
    }));
    const valueIndex = params.findIndex((param) => param.name === valueName && !param.pointer);
    const pointerIndex = params.findIndex((param) => param.name === pointerName && param.pointer);
    if (valueIndex < 0 || pointerIndex < 0) continue;
    helpers.set(name, { definition, valueIndex, pointerIndex });
  }
  return helpers;
}

function rewritePointerStoreForwarderCalls(source, name, helper) {
  const re = new RegExp(`\\b${escapeRegExp(name)}\\s*\\(`, "gu");
  return replaceBalancedCall(source, re, (_match, open, close) => {
    const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim());
    const value = args[helper.valueIndex];
    const pointer = args[helper.pointerIndex];
    if (!value || !pointer?.startsWith("&")) return undefined;
    const target = pointer.slice(1).trim();
    if (target.length === 0) return undefined;
    return `(${target} = ${value})`;
  });
}

export function normalizeLocalPointerDeviceFunctionCalls(source, helpers) {
  return withNormalizerHelpers(helpers, () => {
    const helpers = collectLocalPointerDeviceFunctionHelpers(source);
    if (helpers.size === 0) return source;
    let out = source;
    for (const [name, helper] of helpers) {
      const rewritten = rewriteLocalPointerDeviceFunctionCalls(out, name, helper);
      if (rewritten !== out) out = rewritten.replace(helper.definition, "");
    }
    return out;
  });
}

function collectLocalPointerDeviceFunctionHelpers(source) {
  const helpers = new Map();
  let index = 0;
  while (index < source.length) {
    const device = source.indexOf("__device__", index);
    if (device < 0) break;
    const start = localDeviceFunctionPrefixStart(source, device);
    const brace = source.indexOf("{", device);
    const semicolon = source.indexOf(";", device);
    if (brace < 0) break;
    if (semicolon >= 0 && semicolon < brace) {
      index = semicolon + 1;
      continue;
    }
    const end = findBalanced(source, brace, "{", "}");
    if (end === undefined) break;
    const definition = source.slice(start, end + 1);
    const signature = source.slice(start, brace);
    const name = localVoidDeviceFunctionName(signature);
    const params = localDeviceFunctionParams(signature);
    if (
      name !== undefined &&
      params.length > 0 &&
      params.some((param) => param.pointer) &&
      isInlineableLocalPointerHelperBody(source.slice(brace + 1, end), params)
    ) {
      helpers.set(name, { definition, params, body: source.slice(brace + 1, end) });
    }
    index = end + 1;
  }
  return helpers;
}

function localDeviceFunctionPrefixStart(source, device) {
  let cursor = device;
  while (true) {
    const beforeWhitespace = skipBackwardWhitespace(source, cursor);
    const match = /(?:template\s*<[^<>]*>|static|inline|__inline__|__forceinline__|__host__|constexpr)\s*$/u.exec(source.slice(0, beforeWhitespace));
    if (match === null) break;
    cursor = beforeWhitespace - match[0].trimEnd().length;
  }
  return cursor;
}

function skipBackwardWhitespace(source, index) {
  let cursor = index;
  while (cursor > 0 && /\s/u.test(source[cursor - 1])) cursor--;
  return cursor;
}

function localVoidDeviceFunctionName(signature) {
  const open = signature.lastIndexOf("(");
  if (open < 0) return undefined;
  const before = signature.slice(0, open)
    .replace(/\b(?:template\s*<[^<>]*>|static|inline|__inline__|__forceinline__|__host__|__device__|constexpr)\b/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  return /^void\s+([A-Za-z_][A-Za-z0-9_]*)$/u.exec(before)?.[1];
}

function localDeviceFunctionParams(signature) {
  const open = signature.lastIndexOf("(");
  const paramsSource = balancedParenContents(signature, open);
  if (paramsSource === undefined || paramsSource.trim().length === 0) return [];
  return splitTopLevel(paramsSource).map(parseLocalDeviceFunctionParam).filter(Boolean);
}

function parseLocalDeviceFunctionParam(param) {
  const cleaned = param
    .replace(/\b(?:const|volatile|__restrict__|__restrict|restrict)\b/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  const name = /([A-Za-z_][A-Za-z0-9_]*)\s*$/.exec(cleaned)?.[1];
  if (name === undefined) return undefined;
  const head = cleaned.slice(0, cleaned.length - name.length).trim();
  const pointer = head.includes("*");
  const type = head.replace(/\*/gu, " ").replace(/\s+/gu, " ").trim();
  if (type.length === 0 || type.includes("&")) return undefined;
  return { name, type, pointer };
}

function isInlineableLocalPointerHelperBody(body, params) {
  if (/\b(?:return|__syncthreads|__syncwarp|asm)\b/u.test(body)) return false;
  for (const param of params.filter((item) => item.pointer)) {
    const uses = body.match(new RegExp(`\\b${escapeRegExp(param.name)}\\b`, "gu")) ?? [];
    const derefs = body.match(new RegExp(`\\*\\s*${escapeRegExp(param.name)}\\b`, "gu")) ?? [];
    if (uses.length !== derefs.length) return false;
  }
  return true;
}

function rewriteLocalPointerDeviceFunctionCalls(source, name, helper) {
  const re = new RegExp(`\\b${escapeRegExp(name)}\\s*\\(`, "gu");
  return replaceBalancedCall(source, re, (_match, open, close) => {
    const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim());
    if (args.length !== helper.params.length) return undefined;
    const pointerTargets = new Map();
    for (const [index, param] of helper.params.entries()) {
      if (!param.pointer) continue;
      const target = addressTarget(args[index]);
      if (target === undefined) return undefined;
      pointerTargets.set(param.name, target);
    }
    return inlineLocalPointerDeviceFunctionBody(helper, args, pointerTargets);
  });
}

function inlineLocalPointerDeviceFunctionBody(helper, args, pointerTargets) {
  const scalarParams = helper.params
    .map((param, index) => ({ param, index }))
    .filter(({ param }) => !param.pointer);
  const locals = scalarParams.map(({ param, index }) => ({
    name: `bg_${param.name}`,
    type: param.type,
    value: args[index] ?? "0",
  }));
  let body = helper.body.trim();
  for (const [name, target] of pointerTargets) {
    body = body.replace(new RegExp(`\\*\\s*${escapeRegExp(name)}\\b`, "gu"), target);
  }
  for (const local of locals) {
    body = body.replace(new RegExp(`\\b${escapeRegExp(local.name.replace(/^bg_/u, ""))}\\b`, "gu"), local.name);
  }
  if ([...pointerTargets.keys()].some((name) => new RegExp(`\\b${escapeRegExp(name)}\\b`, "u").test(body))) return undefined;
  return `{\n${locals.map((local) => `  ${local.type} ${local.name} = ${local.value};`).join("\n")}\n${indentBlock(body, "  ")}\n}`;
}

function indentBlock(source, prefix) {
  return source.split(/\r?\n/u).map((line) => line.trim().length === 0 ? "" : `${prefix}${line}`).join("\n");
}

function normalizeBlockReducerName(name) {
  if (/^(?:warpReduceSum|warp_reduce_sum|warp_reduce_sum_f32|warp_reduce_sum_f16|warp_reduce_sum_f16_f16|warp_reduce_sum_f16_f32)$/u.test(name)) return name;
  if (/^(?:warpReduceMax|warp_reduce_max|warp_reduce_max_f32)$/u.test(name)) return name;
  if (/^(?:warpReduceMin|warp_reduce_min)$/u.test(name)) return name;
  return undefined;
}

function sincosReplacement(args) {
  const [phase, sinTarget, cosTarget] = args;
  const sinLvalue = addressTarget(sinTarget);
  const cosLvalue = addressTarget(cosTarget);
  if (!phase || !sinLvalue || !cosLvalue) return undefined;
  return `${sinLvalue} = sinf(${phase}); ${cosLvalue} = cosf(${phase})`;
}
