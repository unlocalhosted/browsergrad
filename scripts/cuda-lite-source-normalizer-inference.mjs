import { requireNormalizerHelpers } from "./cuda-lite-source-normalizer-context.mjs";

const balancedParenContents = (...args) => requireNormalizerHelpers().balancedParenContents(...args);
const collectObjectDefines = (...args) => requireNormalizerHelpers().collectObjectDefines(...args);
const escapeRegExp = (...args) => requireNormalizerHelpers().escapeRegExp(...args);
const evaluateTemplateIntegerExpression = (...args) => requireNormalizerHelpers().evaluateTemplateIntegerExpression(...args);
const findBalanced = (...args) => requireNormalizerHelpers().findBalanced(...args);
const mergeDefineMaps = (...args) => requireNormalizerHelpers().mergeDefineMaps(...args);
const normalizeTemplateSymbolArgument = (...args) => requireNormalizerHelpers().normalizeTemplateSymbolArgument(...args);
const normalizeTemplateTypeArgument = (...args) => requireNormalizerHelpers().normalizeTemplateTypeArgument(...args);
const normalizeTemplateValueArgument = (...args) => requireNormalizerHelpers().normalizeTemplateValueArgument(...args);
const numericTemplateDefines = (...args) => requireNormalizerHelpers().numericTemplateDefines(...args);
const parseTemplateParam = (...args) => requireNormalizerHelpers().parseTemplateParam(...args);
const resolveTemplateDefineValue = (...args) => requireNormalizerHelpers().resolveTemplateDefineValue(...args);
const scanCudaFuncSetAttributeTemplateReferences = (...args) => requireNormalizerHelpers().scanCudaFuncSetAttributeTemplateReferences(...args);
const skipWhitespace = (...args) => requireNormalizerHelpers().skipWhitespace(...args);
const splitTopLevel = (...args) => requireNormalizerHelpers().splitTopLevel(...args);
const substituteTemplateValues = (...args) => requireNormalizerHelpers().substituteTemplateValues(...args);
const templateArgumentScore = (...args) => requireNormalizerHelpers().templateArgumentScore(...args);
const templateDefaultEnvironment = (...args) => requireNormalizerHelpers().templateDefaultEnvironment(...args);
const templateValueEnvironment = (...args) => requireNormalizerHelpers().templateValueEnvironment(...args);

export function substituteTemplateTypes(source, typeEnv) {
  if (typeEnv.size === 0) return source;
  return source.replace(/\b[A-Za-z_][A-Za-z0-9_]*\b/gu, (name) => typeEnv.get(name) ?? name);
}

export function templateParameterNames(source) {
  const names = [];
  for (const header of templateHeaders(source)) {
    for (const param of splitTopLevel(header)) {
      const parsed = parseTemplateParam(param);
      if (parsed?.name !== undefined) names.push(parsed.name);
    }
  }
  return names;
}

export function templateHeaders(source) {
  const headers = [];
  let index = 0;
  while (index < source.length) {
    const match = /\btemplate\s*</u.exec(source.slice(index));
    if (match === null) break;
    const start = index + match.index;
    const open = source.indexOf("<", start);
    const close = findBalanced(source, open, "<", ">");
    if (close === undefined) break;
    headers.push(source.slice(open + 1, close));
    index = close + 1;
  }
  return headers;
}

export function stripKernelLaunchTemplateArguments(source) {
  const refs = scanTemplatedKernelReferences(source).filter((ref) => ref.kind === "launch");
  if (refs.length === 0) return source;
  let out = "";
  let cursor = 0;
  for (const ref of refs) {
    out += source.slice(cursor, ref.templateStart);
    cursor = ref.templateEnd + 1;
  }
  out += source.slice(cursor);
  return out;
}

export function stripKnownTemplateCallArguments(source, names) {
  let out = source;
  for (const name of names) out = stripTemplateCallArgumentsForName(out, name);
  return out;
}

export function stripTemplateCallArgumentsForName(source, name) {
  const re = new RegExp(`\\b${escapeRegExp(name)}\\s*<`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const templateStart = source.indexOf("<", match.index + name.length);
    const templateEnd = findBalanced(source, templateStart, "<", ">");
    if (templateEnd === undefined) {
      re.lastIndex = match.index + name.length;
      continue;
    }
    const afterTemplate = skipWhitespace(source, templateEnd + 1);
    if (source[afterTemplate] !== "(") {
      re.lastIndex = templateEnd + 1;
      continue;
    }
    out += source.slice(cursor, templateStart);
    cursor = templateEnd + 1;
    re.lastIndex = templateEnd + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

export function scanTemplatedKernelReferences(source) {
  const refs = scanRegexTemplatedKernelLaunches(source);
  let index = 0;
  while (index < source.length) {
    const ident = /[A-Za-z_][A-Za-z0-9_]*/u.exec(source.slice(index));
    if (ident === null) break;
    const nameStart = index + ident.index;
    const name = ident[0];
    index = nameStart + name.length;
    if (name === "template") continue;
    const templateStart = skipWhitespace(source, index);
    if (source[templateStart] !== "<") continue;
    const templateEnd = findBalanced(source, templateStart, "<", ">");
    if (templateEnd === undefined) continue;
    const afterTemplate = skipWhitespace(source, templateEnd + 1);
    if (source.slice(afterTemplate, afterTemplate + 3) === "<<<") {
      refs.push({
        kind: "launch",
        name,
        args: splitTopLevel(source.slice(templateStart + 1, templateEnd)).map((arg) => arg.trim()),
        templateStart,
        templateEnd,
      });
      index = afterTemplate + 3;
      continue;
    }
    index = templateEnd + 1;
  }
  refs.push(...scanCudaFuncSetAttributeTemplateReferences(source));
  return dedupeTemplateRefs(refs);
}

export function scanRegexTemplatedKernelLaunches(source) {
  const refs = [];
  const re = /\b([A-Za-z_][A-Za-z0-9_]*)\s*</gu;
  for (const match of source.matchAll(re)) {
    const name = match[1];
    if (name === undefined || name === "template") continue;
    const templateStart = source.indexOf("<", match.index + name.length);
    const templateEnd = findBalanced(source, templateStart, "<", ">");
    if (templateEnd === undefined) continue;
    const afterTemplate = skipWhitespace(source, templateEnd + 1);
    if (source.slice(afterTemplate, afterTemplate + 3) !== "<<<") continue;
    refs.push({
      kind: "launch",
      name,
      args: splitTopLevel(source.slice(templateStart + 1, templateEnd)).map((arg) => arg.trim()),
      templateStart,
      templateEnd,
    });
  }
  return refs;
}

export function dedupeTemplateRefs(refs) {
  const seen = new Set();
  return refs.filter((ref) => {
    const key = `${ref.kind}:${ref.name}:${ref.templateStart}:${ref.templateEnd}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function collectWrapperPropagatedTemplateArguments(source, definesByName = new Map()) {
  const propagated = [];
  const calls = scanTemplatedCallReferences(source);
  const ordinaryCalls = scanFunctionCallReferences(source);
  const templatedKernels = new Map(scanTemplatedKernelDefinitions(source).map((kernel) => [kernel.name, kernel]));
  for (const fn of scanTemplatedFunctionDefinitions(source)) {
    const fnCalls = [
      ...calls.filter((call) => call.name === fn.name && call.args.length > 0)
        .map((call) => ({ args: call.args, start: call.start, explicit: true })),
      ...ordinaryCalls.filter((call) => call.name === fn.name && call.args.length > 0)
        .map((call) => ({ args: inferTemplatedFunctionCallArgs(fn, call, source, definesByName), start: call.start, explicit: false })),
    ].filter((call) => call.args.some((arg) => arg !== undefined && arg !== ""));
    if (fnCalls.length === 0) continue;
    const kernelRefs = scanTemplatedKernelReferences(fn.body);
    const kernelLaunches = scanKernelLaunchCalls(fn.body);
    if (kernelRefs.length === 0 && kernelLaunches.length === 0) continue;
    for (const call of fnCalls) {
      const env = templateEnvironment(fn.templateParams, call.args, definesByName);
      if (env.size === 0) continue;
      extendEnvironmentWithConstexprs(env, fn.body, definesByName);
      for (const ref of kernelRefs) {
        const args = ref.args.map((arg) => substituteTemplateArgument(arg, env, definesByName));
        if (templateArgumentScore(args) === 0) continue;
        propagated.push({ name: ref.name, args });
      }
      for (const launch of kernelLaunches) {
        const kernel = templatedKernels.get(launch.name);
        if (kernel === undefined) continue;
        const symbols = substituteVisiblePointerTypes(
          collectVisiblePointerTypes(`${fn.signature}\n${fn.body.slice(0, launch.start)}`, definesByName),
          env,
          definesByName,
        );
        const args = inferKernelTemplateArgsFromSymbols(kernel, launch, symbols, definesByName);
        if (templateArgumentScore(args) === 0) continue;
        propagated.push({ name: launch.name, args });
      }
    }
  }
  return propagated;
}

export function collectSymbolWrapperTemplateArguments(source, definesByName = new Map()) {
  const propagated = [];
  const calls = scanTemplatedCallReferences(source);
  for (const fn of scanTemplatedFunctionDefinitions(source)) {
    const symbolParams = fn.templateParams
      .map((param, index) => ({ param, index }))
      .filter((entry) => entry.param.kind === "symbol");
    if (symbolParams.length === 0) continue;
    const wrapperCalls = calls.filter((call) => call.name === fn.name && call.args.length > 0);
    if (wrapperCalls.length === 0) continue;
    const kernelRefs = scanTemplatedKernelReferences(fn.body).filter((ref) =>
      ref.args.some((arg) => symbolParams.some(({ param }) => arg === param.name)));
    if (kernelRefs.length === 0) continue;
    for (const call of wrapperCalls) {
      for (const ref of kernelRefs) {
        const args = ref.args.map((arg) => {
          const entry = symbolParams.find(({ param }) => param.name === arg);
          if (entry === undefined) return substituteTemplateArgument(arg, new Map(), definesByName);
          return normalizeTemplateSymbolArgument(resolveTemplateDefineValue(call.args[entry.index], definesByName));
        });
        if (templateArgumentScore(args) > 0) propagated.push({ name: ref.name, args });
      }
    }
  }
  return propagated;
}

export function collectSimpleSymbolWrapperTemplateArguments(source, definesByName = new Map()) {
  const propagated = [];
  const wrapperRe = /\btemplate\s*<\s*([A-Za-z_][A-Za-z0-9_:]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*>\s*[\s\S]{0,320}?\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{/gu;
  let match;
  while ((match = wrapperRe.exec(source)) !== null) {
    const symbolName = match[2];
    const wrapperName = match[3];
    const brace = source.indexOf("{", match.index);
    const end = findBalanced(source, brace, "{", "}");
    if (symbolName === undefined || wrapperName === undefined || brace < 0 || end === undefined) {
      wrapperRe.lastIndex = match.index + 1;
      continue;
    }
    const body = source.slice(brace + 1, end);
    const refs = scanTemplatedKernelReferences(body).filter((ref) => ref.args.includes(symbolName));
    if (refs.length === 0) {
      wrapperRe.lastIndex = end + 1;
      continue;
    }
    const callRe = new RegExp(`\\b${escapeRegExp(wrapperName)}\\s*<\\s*([A-Za-z_][A-Za-z0-9_:]*)\\s*>\\s*\\(`, "gu");
    for (const call of source.matchAll(callRe)) {
      const value = normalizeTemplateSymbolArgument(resolveTemplateDefineValue(call[1], definesByName));
      if (value === undefined || value === symbolName) continue;
      for (const ref of refs) {
        propagated.push({
          name: ref.name,
          args: ref.args.map((arg) => arg === symbolName ? value : substituteTemplateArgument(arg, new Map(), definesByName)),
        });
      }
    }
    wrapperRe.lastIndex = end + 1;
  }
  return propagated;
}

export function scanFunctionCallReferences(source) {
  const calls = [];
  let index = 0;
  while (index < source.length) {
    const ident = /[A-Za-z_][A-Za-z0-9_]*/u.exec(source.slice(index));
    if (ident === null) break;
    const nameStart = index + ident.index;
    const name = ident[0];
    index = nameStart + name.length;
    if (["if", "for", "while", "switch", "return", "sizeof", "alignof", "template"].includes(name)) continue;
    const open = skipWhitespace(source, index);
    if (source[open] !== "(") continue;
    const close = findBalanced(source, open, "(", ")");
    if (close === undefined) continue;
    const after = skipWhitespace(source, close + 1);
    if (source[after] === "{" || source.slice(after, after + 3) === ">>>") {
      index = close + 1;
      continue;
    }
    const before = source.slice(Math.max(0, nameStart - 24), nameStart);
    if (/\b(?:__global__|__device__|__host__|void|float|int|uint|size_t|ptrdiff_t|bool)\s*$/u.test(before)) {
      index = close + 1;
      continue;
    }
    calls.push({
      name,
      args: splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim()),
      start: nameStart,
    });
    index = close + 1;
  }
  return calls;
}

export function inferTemplatedFunctionCallArgs(fn, call, source, definesByName = new Map()) {
  const env = new Map();
  const params = parseFunctionSignatureParams(fn.signature, fn.templateParams);
  const visibleSource = source.slice(0, call.start);
  const pointerSymbols = collectVisiblePointerTypes(visibleSource, definesByName);
  const valueSymbols = collectVisibleValueTypes(visibleSource, definesByName);
  for (let index = 0; index < params.length; index++) {
    const param = params[index];
    const arg = call.args[index];
    if (param === undefined || arg === undefined || param.templateType === undefined) continue;
    const type = param.pointer
      ? inferArgumentPointerType(arg, pointerSymbols)
      : inferArgumentValueType(arg, valueSymbols, definesByName);
    if (type !== undefined) env.set(param.templateType, type);
  }
  return fn.templateParams.map((param) => param.kind === "type" || param.kind === "value" || param.kind === "symbol" ? env.get(param.name) : undefined);
}

export function collectLaunchInferredTemplateArguments(source, definesByName = new Map()) {
  const inferred = [];
  const kernels = new Map(scanTemplatedKernelDefinitions(source).map((kernel) => [kernel.name, kernel]));
  for (const launch of scanKernelLaunchCalls(source)) {
    const kernel = kernels.get(launch.name);
    if (kernel === undefined) continue;
    const args = inferKernelTemplateArgs(kernel, launch, source, definesByName);
    if (args.some((arg) => arg !== undefined)) {
      inferred.push({ name: launch.name, args: args.map((arg) => arg ?? "") });
    }
  }
  return inferred;
}

export function scanTemplatedKernelDefinitions(source) {
  return scanTemplatedFunctionDefinitions(source)
    .filter((fn) => /\b__global__\b/u.test(fn.signature))
    .map((fn) => ({
      ...fn,
      kernelParams: parseKernelSignatureParams(fn.signature),
    }));
}

export function scanKernelLaunchCalls(source) {
  const calls = [];
  let index = 0;
  while (index < source.length) {
    const match = /([A-Za-z_][A-Za-z0-9_]*)\s*<<</gu.exec(source.slice(index));
    if (match === null) break;
    const name = match[1];
    const nameStart = index + match.index;
    const launchOpen = source.indexOf("<<<", nameStart + name.length);
    const launchClose = findCudaLaunchClose(source, launchOpen);
    if (!name || launchOpen < 0 || launchClose === undefined) {
      index = nameStart + 1;
      continue;
    }
    const argsOpen = skipWhitespace(source, launchClose + 3);
    if (source[argsOpen] !== "(") {
      index = launchClose + 3;
      continue;
    }
    const argsClose = findBalanced(source, argsOpen, "(", ")");
    if (argsClose === undefined) {
      index = argsOpen + 1;
      continue;
    }
    calls.push({
      name,
      args: splitTopLevel(source.slice(argsOpen + 1, argsClose)).map((arg) => arg.trim()),
      start: nameStart,
    });
    index = argsClose + 1;
  }
  return calls;
}

export function findCudaLaunchClose(source, launchOpen) {
  if (launchOpen < 0) return undefined;
  let angle = 0;
  for (let index = launchOpen; index < source.length - 2; index++) {
    const three = source.slice(index, index + 3);
    if (three === "<<<") {
      angle++;
      index += 2;
      continue;
    }
    if (three === ">>>") {
      angle--;
      if (angle === 0) return index;
      index += 2;
    }
  }
  return undefined;
}

export function inferKernelTemplateArgs(kernel, launch, source, definesByName = new Map()) {
  const symbols = collectVisiblePointerTypes(source.slice(0, launch.start), definesByName);
  return inferKernelTemplateArgsFromSymbols(kernel, launch, symbols, definesByName);
}

export function inferKernelTemplateArgsFromSymbols(kernel, launch, symbols, definesByName = new Map()) {
  const env = new Map();
  for (let index = 0; index < kernel.kernelParams.length; index++) {
    const param = kernel.kernelParams[index];
    const arg = launch.args[index];
    if (param === undefined || arg === undefined) continue;
    if (param.templateType !== undefined) {
      const type = inferArgumentPointerType(arg, symbols);
      if (type !== undefined) env.set(param.templateType, type);
    }
    if (param.boolConstantTemplate !== undefined) {
      const value = inferBoolConstantArgument(arg, symbols);
      if (value !== undefined) env.set(param.boolConstantTemplate, value);
    }
  }
  return kernel.templateParams.map((param) => param.kind === "type" || param.kind === "value" || param.kind === "symbol" ? env.get(param.name) : undefined);
}

export function substituteVisiblePointerTypes(symbols, env, definesByName = new Map()) {
  const out = new Map();
  for (const [name, type] of symbols) {
    out.set(name, substituteTemplateArgument(type, env, definesByName));
  }
  return out;
}

export function parseKernelSignatureParams(signature) {
  const open = signature.indexOf("(");
  const params = balancedParenContents(signature, open);
  if (params === undefined || params.trim().length === 0) return [];
  return splitTopLevel(params).map(parseKernelParamForInference);
}

export function parseFunctionSignatureParams(signature, templateParams = []) {
  const open = signature.indexOf("(");
  const params = balancedParenContents(signature, open);
  if (params === undefined || params.trim().length === 0) return [];
  const templateTypeNames = new Set(templateParams.filter((param) => param.kind === "type").map((param) => param.name));
  return splitTopLevel(params).map((param) => parseFunctionParamForInference(param, templateTypeNames));
}

export function parseKernelParamForInference(param) {
  const boolCarrier = /\bstd\s*::\s*bool_constant\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>/u.exec(param)?.[1];
  const withoutQualifiers = param
    .replace(/\b(?:const|volatile|__restrict__|__restrict|restrict)\b/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  const pointer = /^([A-Za-z_][A-Za-z0-9_:]*)\s*\*/u.exec(withoutQualifiers)?.[1];
  return {
    ...(pointer === undefined ? {} : { templateType: pointer }),
    ...(boolCarrier === undefined ? {} : { boolConstantTemplate: boolCarrier }),
  };
}

export function parseFunctionParamForInference(param, templateTypeNames) {
  const withoutQualifiers = param
    .replace(/\b(?:const|volatile|__restrict__|__restrict|restrict)\b/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  const pointer = /^([A-Za-z_][A-Za-z0-9_:]*)\s*\*/u.exec(withoutQualifiers)?.[1];
  if (pointer !== undefined && templateTypeNames.has(pointer)) return { templateType: pointer, pointer: true };
  const scalar = /^([A-Za-z_][A-Za-z0-9_:]*)\s+[A-Za-z_][A-Za-z0-9_]*\b/u.exec(withoutQualifiers)?.[1];
  if (scalar !== undefined && templateTypeNames.has(scalar)) return { templateType: scalar, pointer: false };
  return {};
}

export function collectVisiblePointerTypes(source, initialDefines = new Map()) {
  const symbols = new Map();
  const defines = mergeDefineMaps(initialDefines, collectObjectDefines(source));
  const typePattern = "(?:(?:unsigned|signed)\\s+(?:char|short|int|long)|long\\s+long|[A-Za-z_][A-Za-z0-9_:]*)";
  const re = new RegExp(`\\b(?:(?:const|volatile)\\s+)*(${typePattern})\\s*(?:\\*\\s*(?:(?:const|volatile|__restrict__|__restrict|restrict)\\s+)*)+\\s*([A-Za-z_][A-Za-z0-9_]*)\\b`, "gu");
  for (const match of source.matchAll(re)) {
    const [, type, name] = match;
    if (type && name) symbols.set(name, normalizeTemplateTypeArgument(defines.get(type) ?? type, defines) ?? type);
  }
  return symbols;
}

export function collectVisibleValueTypes(source, initialDefines = new Map()) {
  const symbols = new Map();
  const defines = mergeDefineMaps(initialDefines, collectObjectDefines(source));
  const typePattern = "(?:(?:unsigned|signed)\\s+(?:char|short|int|long)|long\\s+long|[A-Za-z_][A-Za-z0-9_:]*)";
  const re = new RegExp(`\\b(?:const\\s+|constexpr\\s+|static\\s+|volatile\\s+)*(${typePattern})\\s+([A-Za-z_][A-Za-z0-9_]*)\\b(?!\\s*\\()`, "gu");
  for (const match of source.matchAll(re)) {
    const [, rawType, name] = match;
    if (!rawType || !name) continue;
    const normalized = normalizeTemplateTypeArgument(defines.get(rawType) ?? rawType, defines);
    if (normalized !== undefined) symbols.set(name, normalized);
  }
  return symbols;
}

export function inferArgumentPointerType(arg, symbols) {
  const cast = /^\(\s*(?:const\s+)?([A-Za-z_][A-Za-z0-9_:]*)\s*\*\s*\)/u.exec(arg)?.[1];
  if (cast !== undefined) return cast;
  const name = /^[A-Za-z_][A-Za-z0-9_]*/u.exec(arg.trim())?.[0];
  return name === undefined ? undefined : symbols.get(name);
}

export function inferArgumentValueType(arg, symbols, definesByName = new Map()) {
  const expression = arg.trim();
  const cast = /^\(\s*([A-Za-z_][A-Za-z0-9_:]*)\s*\)/u.exec(expression)?.[1];
  if (cast !== undefined) {
    const normalized = normalizeTemplateTypeArgument(definesByName.get(cast) ?? cast, definesByName);
    if (normalized !== undefined) return normalized;
  }
  if (/\b[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?f\b/u.test(expression) || /\b[0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?\b/u.test(expression)) return "float";
  if (/^\s*(?:0x[0-9A-Fa-f]+u?|\d+u)\s*$/u.test(expression)) return "uint";
  const names = [...expression.matchAll(/\b[A-Za-z_][A-Za-z0-9_]*\b/gu)].map((match) => match[0]);
  const types = names
    .map((name) => symbols.get(name) ?? normalizeTemplateTypeArgument(definesByName.get(name) ?? name, definesByName))
    .filter(Boolean);
  if (types.includes("float") || types.includes("half")) return "float";
  if (types.includes("uint")) return "uint";
  if (types.includes("int")) return "int";
  if (/^[0-9A-Fa-fxXuUlL\s()+\-*/%<>=!&|^?:.]+$/u.test(expression)) return "int";
  return undefined;
}

export function inferBoolConstantArgument(arg) {
  const explicit = /\bstd\s*::\s*bool_constant\s*<\s*(true|false)\s*>\s*\{\s*\}/u.exec(arg)?.[1];
  if (explicit !== undefined) return explicit;
  if (/^\s*true\s*$/u.test(arg)) return "true";
  if (/^\s*false\s*$/u.test(arg)) return "false";
  return undefined;
}

export function scanTemplatedFunctionDefinitions(source) {
  const definitions = [];
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
    const name = templatedFunctionName(signature);
    const end = findBalanced(source, brace, "{", "}");
    if (name === undefined || end === undefined) {
      index = close + 1;
      continue;
    }
    definitions.push({
      name,
      signature,
      templateParams: splitTopLevel(source.slice(open + 1, close)).map(parseTemplateParam).filter(Boolean),
      body: source.slice(brace + 1, end),
    });
    index = end + 1;
  }
  return definitions;
}

export function templatedFunctionName(signature) {
  const open = signature.lastIndexOf("(");
  if (open < 0) return undefined;
  const before = signature.slice(0, open).trim();
  return /([A-Za-z_][A-Za-z0-9_]*)\s*$/u.exec(before)?.[1];
}

export function scanTemplatedCallReferences(source) {
  const refs = [];
  let index = 0;
  while (index < source.length) {
    const ident = /[A-Za-z_][A-Za-z0-9_]*/u.exec(source.slice(index));
    if (ident === null) break;
    const nameStart = index + ident.index;
    const name = ident[0];
    index = nameStart + name.length;
    if (name === "template") continue;
    const templateStart = skipWhitespace(source, index);
    if (source[templateStart] !== "<") continue;
    const templateEnd = findBalanced(source, templateStart, "<", ">");
    if (templateEnd === undefined) continue;
    const afterTemplate = skipWhitespace(source, templateEnd + 1);
    if (source.slice(afterTemplate, afterTemplate + 3) === "<<<") {
      index = afterTemplate + 3;
      continue;
    }
    if (source[afterTemplate] !== "(") {
      index = templateEnd + 1;
      continue;
    }
    refs.push({
      name,
      args: splitTopLevel(source.slice(templateStart + 1, templateEnd)).map((arg) => arg.trim()),
      start: nameStart,
    });
    index = afterTemplate + 1;
  }
  return refs;
}

export function collectDeviceFunctionTemplateArguments(sources, names, definesByName = new Map()) {
  const out = new Map();
  const templatedFunctions = new Map();
  for (const source of sources) {
    for (const fn of scanTemplatedFunctionDefinitions(source)) templatedFunctions.set(fn.name, fn);
  }
  for (const source of sources) {
    const env = templateDefaultEnvironment(source, definesByName);
    for (const call of scanTemplatedCallReferences(source)) {
      if (!names.has(call.name)) continue;
      const args = call.args.map((arg) => substituteTemplateArgument(arg, env, definesByName));
      if (templateArgumentScore(args) === 0) continue;
      setTemplateArgumentsIfBetter(out, call.name, args);
    }
    for (const call of scanFunctionCallReferences(source)) {
      if (!names.has(call.name) || out.has(call.name)) continue;
      const fn = templatedFunctions.get(call.name);
      if (fn === undefined) continue;
      const args = inferTemplatedFunctionCallArgs(fn, call, source, definesByName).map((arg) => arg ?? "");
      if (templateArgumentScore(args) === 0) continue;
      setTemplateArgumentsIfBetter(out, call.name, args);
    }
  }
  let changed = true;
  while (changed) {
    changed = false;
    for (const fn of templatedFunctions.values()) {
      const fnArgs = out.get(fn.name);
      if (fnArgs === undefined || templateArgumentScore(fnArgs) === 0) continue;
      const typeEnv = templateEnvironment(fn.templateParams, fnArgs, definesByName);
      const valueEnv = templateValueEnvironment(fn.templateParams, fnArgs, definesByName);
      if (typeEnv.size === 0 && valueEnv.size === 0) continue;
      const body = substituteTemplateValues(substituteTemplateTypes(fn.body, typeEnv), valueEnv);
      for (const call of scanTemplatedCallReferences(body)) {
        if (!names.has(call.name)) continue;
        const args = call.args.map((arg) => substituteTemplateArgument(arg, mergeDefineMaps(definesByName, typeEnv, valueEnv), definesByName));
        if (templateArgumentScore(args) === 0) continue;
        changed = setTemplateArgumentsIfBetter(out, call.name, args) || changed;
      }
      for (const call of scanFunctionCallReferences(body)) {
        if (!names.has(call.name) || out.has(call.name)) continue;
        const target = templatedFunctions.get(call.name);
        if (target === undefined) continue;
        const args = inferTemplatedFunctionCallArgs(target, call, body, mergeDefineMaps(definesByName, typeEnv, valueEnv)).map((arg) => arg ?? "");
        if (templateArgumentScore(args) === 0) continue;
        changed = setTemplateArgumentsIfBetter(out, call.name, args) || changed;
      }
    }
  }
  return out;
}

export function setTemplateArgumentsIfBetter(out, name, args) {
  const previous = out.get(name);
  if (previous !== undefined && templateArgumentScore(previous) >= templateArgumentScore(args)) return false;
  out.set(name, args);
  return true;
}

export function templateEnvironment(params, args, definesByName = new Map()) {
  const env = new Map();
  let argIndex = 0;
  for (const param of params) {
    const arg = args[argIndex++];
    if (!param || arg === undefined) continue;
    if (param.kind === "value") {
      const normalized = normalizeTemplateValueArgument(resolveTemplateDefineValue(arg, definesByName), param.valueType);
      if (normalized !== undefined) env.set(param.name, normalized);
    } else if (param.kind === "symbol") {
      const normalized = normalizeTemplateSymbolArgument(resolveTemplateDefineValue(arg, definesByName));
      if (normalized !== undefined) env.set(param.name, normalized);
    } else if (param.kind === "type" && /^[A-Za-z_][A-Za-z0-9_:]*$/u.test(arg.trim())) {
      env.set(param.name, resolveTemplateTypeAliasArgument(arg, definesByName));
    }
  }
  return env;
}

export function extendEnvironmentWithConstexprs(env, body, definesByName = new Map()) {
  for (const [name, value] of numericTemplateDefines(mergeDefineMaps(definesByName, env))) env.set(name, value);
  let changed = true;
  while (changed) {
    changed = false;
    for (const { name, expression } of scanVisibleIntegerDeclarations(body)) {
      if (env.has(name)) continue;
      const value = evaluateTemplateIntegerExpression(expression, env);
      if (value === undefined) continue;
      env.set(name, value);
      changed = true;
    }
  }
}

export function collectVisibleIntegerConstants(source, definesByName = new Map()) {
  const env = numericTemplateDefines(definesByName);
  for (const [name, value] of numericTemplateDefines(env)) env.set(name, value);
  let changed = true;
  while (changed) {
    changed = false;
    for (const { name, expression } of scanVisibleIntegerDeclarations(source)) {
      if (env.has(name)) continue;
      const value = evaluateTemplateIntegerExpression(expression, env);
      if (value === undefined) continue;
      env.set(name, value);
      changed = true;
    }
  }
  return env;
}

export function scanVisibleIntegerDeclarations(source) {
  return [
    ...scanConstexprIntegerDeclarations(source),
    ...scanCuteIntObjectDeclarations(source),
  ];
}

export function scanConstexprIntegerDeclarations(source) {
  const declarations = [];
  const re = /\b(?:constexpr|const|static)\s+(?:const\s+)?(?:int|uint|unsigned\s+int|size_t|ptrdiff_t|bool)\s+([^;]+);/gu;
  for (const match of source.matchAll(re)) {
    const tail = match[1];
    if (!tail) continue;
    for (const declarator of splitTopLevel(tail)) {
      const parsed = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([\s\S]+?)\s*$/u.exec(declarator);
      if (parsed?.[1] === undefined || parsed[2] === undefined) continue;
      declarations.push({ name: parsed[1], expression: parsed[2].trim() });
    }
  }
  return declarations;
}

export function scanCuteIntObjectDeclarations(source) {
  const declarations = [];
  const re = /\b(?:auto|const\s+auto|constexpr\s+auto|static\s+constexpr\s+auto)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:cute::)?Int\s*<([^<>]+)>\s*\{\s*\}\s*;/gu;
  for (const match of source.matchAll(re)) {
    if (match[1] === undefined || match[2] === undefined) continue;
    declarations.push({ name: match[1], expression: match[2].trim() });
  }
  return declarations;
}

export function substituteTemplateArgument(arg, env, definesByName = new Map()) {
  const value = evaluateTemplateIntegerExpression(arg, env);
  if (value !== undefined) return value;
  return resolveTemplateTypeAliasArgument(
    arg.replace(/\b[A-Za-z_][A-Za-z0-9_]*\b/gu, (name) => env.get(name) ?? name).trim(),
    definesByName,
  );
}

export function resolveTemplateTypeAliasArgument(arg, definesByName) {
  const trimmed = arg.trim();
  if (!/^[A-Za-z_][A-Za-z0-9_:]*$/u.test(trimmed)) return trimmed;
  return normalizeTemplateTypeArgument(definesByName.get(trimmed) ?? trimmed, definesByName) ?? trimmed;
}

export function resolveTemplateArgument(arg, definesByName) {
  const trimmed = resolveTemplateTypeAliasArgument(arg, definesByName);
  if (trimmed === "true" || trimmed === "false") return trimmed;
  if (!/^[A-Za-z_][A-Za-z0-9_:]*$/u.test(trimmed)) return trimmed;
  const resolved = resolveTemplateDefineValue(trimmed, definesByName);
  if (resolved === "true" || resolved === "false") return resolved;
  const value = normalizeTemplateValueArgument(resolved, "int");
  return value ?? trimmed;
}
