import { requireNormalizerHelpers } from "./cuda-lite-source-normalizer-context.mjs";

const SEMANTIC_BUILTIN_DEVICE_HELPERS = new Set([
  "min", "max", "vec_at", "load128", "load128cs", "store128", "store128cs", "store128cg", "div_ceil", "blockReduce",
  "warpReduceSum", "warpReduceMax", "warpReduceMin", "warp_reduce_sum", "warp_reduce_max", "warp_reduce_min",
  "warp_reduce_sum_f32", "warp_reduce_max_f32", "warp_reduce_sum_f16", "warp_reduce_sum_f16_f16", "warp_reduce_sum_f16_f32",
  "warp_reduce_sum_i8_i32", "warp_reduce_sum_i32_i32", "atomicAdd", "atomicSub", "atomicMin", "atomicMax", "atomicAnd", "atomicOr", "atomicXor", "atomicExch", "atomicCAS",
]);

const SEMANTIC_RECORD_CARRIERS = new Set([
  "DevicePool",
]);
const balancedParenContents = (...args) => requireNormalizerHelpers().balancedParenContents(...args);
const collectBitpackedShortUnions = (...args) => requireNormalizerHelpers().collectBitpackedShortUnions(...args);
const collectLaunchInferredTemplateArguments = (...args) => requireNormalizerHelpers().collectLaunchInferredTemplateArguments(...args);
const collectObjectDefines = (...args) => requireNormalizerHelpers().collectObjectDefines(...args);
const collectPodRecordVectorAliases = (...args) => requireNormalizerHelpers().collectPodRecordVectorAliases(...args);
const collectRecordEnumTypeDefines = (...args) => requireNormalizerHelpers().collectRecordEnumTypeDefines(...args);
const collectScalarizedPodRecords = (...args) => requireNormalizerHelpers().collectScalarizedPodRecords(...args);
const collectSimpleSymbolWrapperTemplateArguments = (...args) => requireNormalizerHelpers().collectSimpleSymbolWrapperTemplateArguments(...args);
const collectSymbolWrapperTemplateArguments = (...args) => requireNormalizerHelpers().collectSymbolWrapperTemplateArguments(...args);
const collectVisibleIntegerConstants = (...args) => requireNormalizerHelpers().collectVisibleIntegerConstants(...args);
const collectWrapperPropagatedTemplateArguments = (...args) => requireNormalizerHelpers().collectWrapperPropagatedTemplateArguments(...args);
const escapeRegExp = (...args) => requireNormalizerHelpers().escapeRegExp(...args);
const evaluateIntegerExpression = (...args) => requireNormalizerHelpers().evaluateIntegerExpression(...args);
const findBalanced = (...args) => requireNormalizerHelpers().findBalanced(...args);
const isTemplateSymbolArgument = (...args) => requireNormalizerHelpers().isTemplateSymbolArgument(...args);
const normalizeTemplateTypeArgument = (...args) => requireNormalizerHelpers().normalizeTemplateTypeArgument(...args);
const parseTemplateParam = (...args) => requireNormalizerHelpers().parseTemplateParam(...args);
const resolveTemplateArgument = (...args) => requireNormalizerHelpers().resolveTemplateArgument(...args);
const scanFunctionCallReferences = (...args) => requireNormalizerHelpers().scanFunctionCallReferences(...args);
const scanStructRecordDeclarations = (...args) => requireNormalizerHelpers().scanStructRecordDeclarations(...args);
const scanTemplatedCallReferences = (...args) => requireNormalizerHelpers().scanTemplatedCallReferences(...args);
const scanTemplatedKernelDefinitions = (...args) => requireNormalizerHelpers().scanTemplatedKernelDefinitions(...args);
const scanTemplatedKernelReferences = (...args) => requireNormalizerHelpers().scanTemplatedKernelReferences(...args);
const skipWhitespace = (...args) => requireNormalizerHelpers().skipWhitespace(...args);
const splitTopLevel = (...args) => requireNormalizerHelpers().splitTopLevel(...args);
const stripLineComment = (...args) => requireNormalizerHelpers().stripLineComment(...args);
const stripRecordEnums = (...args) => requireNormalizerHelpers().stripRecordEnums(...args);
const substituteTemplateArgument = (...args) => requireNormalizerHelpers().substituteTemplateArgument(...args);
const templateArgumentScore = (...args) => requireNormalizerHelpers().templateArgumentScore(...args);

export function collectCudaLiteContextDefines(source) {
  return collectObjectDefines(source);
}

export function pruneCudaPreprocessorBranches(source, definesByName = new Map()) {
  const defines = new Map(definesByName);
  const frames = [];
  const out = [];
  const isIncluded = () => frames.every((frame) => frame.include);
  for (const line of source.split(/\r?\n/u)) {
    const directive = /^\s*#\s*(if|ifdef|ifndef|elif|else|endif|define|undef)\b(.*)$/u.exec(line);
    if (directive === null) {
      if (isIncluded()) out.push(line);
      continue;
    }
    const keyword = directive[1];
    const rest = (directive[2] ?? "").trim();
    if (keyword === "if" || keyword === "ifdef" || keyword === "ifndef") {
      const parentInclude = isIncluded();
      const condition = keyword === "ifdef"
        ? defines.has(rest.split(/\s+/u)[0] ?? "")
        : keyword === "ifndef"
          ? !defines.has(rest.split(/\s+/u)[0] ?? "")
          : evaluatePreprocessorCondition(rest, defines);
      if (condition === undefined && parentInclude) {
        frames.push({ parentInclude, include: true, mode: "unknown", taken: false });
      } else {
        frames.push({
          parentInclude,
          include: parentInclude && condition === true,
          mode: "known",
          taken: condition === true,
        });
      }
      continue;
    }
    if (keyword === "elif") {
      const frame = frames.at(-1);
      if (frame === undefined) continue;
      if (frame.mode === "unknown") {
        frame.include = frame.parentInclude;
        continue;
      }
      if (frame.taken) {
        frame.include = false;
        continue;
      }
      const condition = evaluatePreprocessorCondition(rest, defines);
      if (condition === undefined) {
        frame.mode = "unknown";
        frame.include = frame.parentInclude;
      } else {
        frame.include = frame.parentInclude && condition === true;
        frame.taken = condition === true;
      }
      continue;
    }
    if (keyword === "else") {
      const frame = frames.at(-1);
      if (frame === undefined) continue;
      frame.include = frame.mode === "unknown" ? frame.parentInclude : frame.parentInclude && !frame.taken;
      frame.taken = true;
      continue;
    }
    if (keyword === "endif") {
      frames.pop();
      continue;
    }
    if (isIncluded()) {
      if (keyword === "define") {
        const match = /^([A-Za-z_][A-Za-z0-9_]*)(?!\()\s+(.+?)\s*$/u.exec(rest);
        if (match?.[1] !== undefined && match[2] !== undefined) defines.set(match[1], match[2].trim());
      } else if (keyword === "undef") {
        const name = /^([A-Za-z_][A-Za-z0-9_]*)/u.exec(rest)?.[1];
        if (name !== undefined) defines.delete(name);
      }
      out.push(line);
    }
  }
  return out.join("\n");
}

function evaluatePreprocessorCondition(expression, defines) {
  let unknown = false;
  let expr = expression
    .replace(/\bdefined\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)/gu, (_match, name) => defines.has(name) ? "1" : "0")
    .replace(/\bdefined\s+([A-Za-z_][A-Za-z0-9_]*)/gu, (_match, name) => defines.has(name) ? "1" : "0")
    .replace(/\b([A-Za-z_][A-Za-z0-9_]*)\b/gu, (match, name) => {
      const value = defines.get(name);
      if (value === undefined) return "0";
      const normalized = normalizePreprocessorDefineValue(value);
      if (normalized === undefined) {
        unknown = true;
        return match;
      }
      return normalized;
    })
    .replace(/(?<![=!<>])!(?!=)/gu, "!")
    .replace(/\b(0[xX][0-9A-Fa-f]+|[0-9]+)[uUlL]*/gu, "$1");
  if (unknown || /[^0-9A-Fa-fxX\s()+\-*/%<>=!&|^?:.]/u.test(expr)) return undefined;
  expr = expr.replace(/\b0+([0-9])/gu, "$1");
  const value = evaluateIntegerExpression(expr);
  return value === undefined ? undefined : value !== 0;
}

function normalizePreprocessorDefineValue(value) {
  const trimmed = String(value).trim();
  if (/^(?:true|false)$/u.test(trimmed)) return trimmed === "true" ? "1" : "0";
  if (!/^[0-9A-Fa-fxXuUlL\s()+\-*/%<>=!&|^?:.]+$/u.test(trimmed)) return undefined;
  return trimmed.replace(/\b(0[xX][0-9A-Fa-f]+|[0-9]+)[uUlL]*/gu, "$1");
}

export function normalizeFunctionMacro(macro) {
  const cpAsync = semanticCpAsyncMacro(macro);
  return cpAsync ?? macro;
}

export function injectSharedDeclarationsIntoKernel(kernel, sharedDeclarations) {
  if (sharedDeclarations.length === 0) return kernel;
  const needed = sharedDeclarations
    .filter((declaration) => {
      const name = sharedDeclarationName(declaration);
      return name !== undefined &&
        sourceMentionsIdentifier(kernel, name) &&
        !kernelDeclaresSharedName(kernel, name);
    });
  if (needed.length === 0) return kernel;
  const open = kernel.indexOf("{");
  if (open < 0) return kernel;
  return `${kernel.slice(0, open + 1)}\n${needed.map((declaration) => `  ${declaration.trim()}`).join("\n")}${kernel.slice(open + 1)}`;
}

function sharedDeclarationName(declaration) {
  return /\b(?:extern\s+)?__shared__\s+[\w\s]+?\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[/u.exec(declaration)?.[1];
}

function kernelDeclaresSharedName(kernel, name) {
  return new RegExp(`\\b(?:extern\\s+)?__shared__\\s+[\\w\\s]+?\\s+${escapeRegExp(name)}\\s*\\[`, "u").test(kernel);
}

function semanticCpAsyncMacro(macro) {
  const match = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s+([\s\S]+)$/u.exec(macro);
  if (!match) return undefined;
  const [, name, paramsSource, body] = match;
  if (!name || paramsSource === undefined || !body) return undefined;
  const params = splitTopLevel(paramsSource).map((param) => param.trim()).filter(Boolean);
  const bodyLower = body.toLowerCase();
  if (/\bcp\.async\.(?:ca|cg|bulk)\b/u.test(bodyLower) && params.length >= 3) {
    return `#define ${name}(${params.join(", ")}) CP_ASYNC_${bodyLower.includes("cp.async.ca") ? "CA" : bodyLower.includes("cp.async.bulk") ? "BULK" : "CG"}(${params[0]}, ${params[1]}, ${params[2]})`;
  }
  if (/\bcp\.async\.commit_group\b/u.test(bodyLower)) return `#define ${name}() CP_ASYNC_COMMIT_GROUP()`;
  if (/\bcp\.async\.wait_all\b/u.test(bodyLower)) return `#define ${name}() CP_ASYNC_WAIT_ALL()`;
  if (/\bcp\.async\.wait_group\b/u.test(bodyLower) && params.length >= 1) return `#define ${name}(${params[0]}) CP_ASYNC_WAIT_GROUP(${params[0]})`;
  return undefined;
}

export function kernelDefinitionName(kernel) {
  return kernelSignature(kernel)?.name;
}

export function collectKernelTemplateArguments(source) {
  const definesByName = collectObjectDefines(source);
  const candidates = new Map();
  const kernelTemplateParamNames = new Map(
    scanTemplatedKernelDefinitions(source).map((kernel) => [
      kernel.name,
      kernel.templateParams.map((param) => param.name),
    ]),
  );
  const addCandidate = (name, args) => {
    if (args.length === 0) return;
    const normalizedArgs = args.map((arg) => resolveTemplateArgument(arg, definesByName));
    const previous = candidates.get(name);
    if (shouldReplaceTemplateCandidate(previous, normalizedArgs, kernelTemplateParamNames.get(name))) {
      candidates.set(name, normalizedArgs);
    }
  };
  for (const launch of scanTemplatedKernelReferences(source)) {
    const visibleConstants = collectVisibleIntegerConstants(source.slice(0, launch.templateStart), definesByName);
    addCandidate(launch.name, launch.args.map((arg) => substituteTemplateArgument(arg, visibleConstants, definesByName)));
  }
  for (const propagated of collectWrapperPropagatedTemplateArguments(source, definesByName)) {
    addCandidate(propagated.name, propagated.args);
  }
  for (const propagated of collectSymbolWrapperTemplateArguments(source, definesByName)) {
    addCandidate(propagated.name, propagated.args);
  }
  for (const propagated of collectSimpleSymbolWrapperTemplateArguments(source, definesByName)) {
    addCandidate(propagated.name, propagated.args);
  }
  for (const inferred of collectLaunchInferredTemplateArguments(source, definesByName)) {
    addCandidate(inferred.name, inferred.args);
  }
  return candidates;
}

function shouldReplaceTemplateCandidate(previous, next, paramNames = []) {
  if (previous === undefined) return true;
  const nextScore = templateArgumentScore(next);
  const previousScore = templateArgumentScore(previous);
  if (nextScore > previousScore) return true;
  if (nextScore === previousScore && previous.some((arg, index) => isTemplateSymbolDowngrade(arg, next[index]))) return false;
  return nextScore === previousScore &&
    previous.some((arg, index) => isTemplateSymbolRefinement(arg, next[index], paramNames[index]));
}

function isTemplateParamEcho(args, paramNames) {
  return args.some((arg, index) => {
    const paramName = paramNames[index];
    return paramName !== undefined && arg === paramName;
  });
}

function isTemplateSymbolRefinement(previous, next, paramName) {
  if (previous === undefined || next === undefined || previous === next) return false;
  if (paramName !== undefined && previous === paramName && templateArgumentScore([next]) > 0) return true;
  if (isTemplateSymbolArgument(previous) && !isTemplateSymbolArgument(next) && templateArgumentScore([next]) > 0) return true;
  if (!isTemplateSymbolArgument(previous) || !isTemplateSymbolArgument(next)) return false;
  return String(next).startsWith(`${previous}_`);
}

function isTemplateSymbolDowngrade(previous, next) {
  return isTemplateSymbolArgument(previous) &&
    isTemplateSymbolArgument(next) &&
    previous !== next &&
    String(previous).startsWith(`${next}_`);
}

export function referencedDeviceFunctionClosure(kernel, deviceFunctions, definesByName = new Map()) {
  const byName = new Map();
  for (const fn of deviceFunctions) {
    const currentExplicit = isExplicitTemplateSpecialization(fn.source);
    const overloads = byName.get(fn.name) ?? new Map();
    const key = deviceFunctionSignatureShape(fn.source);
    if (!currentExplicit) {
      for (const [candidateKey, candidate] of overloads) {
        if (isExplicitTemplateSpecialization(candidate.source)) overloads.delete(candidateKey);
      }
      overloads.set(key, fn);
      byName.set(fn.name, overloads);
    } else if (overloads.size === 0 || [...overloads.values()].every((candidate) => isExplicitTemplateSpecialization(candidate.source))) {
      overloads.set(key, fn);
      byName.set(fn.name, overloads);
    }
  }
  const included = new Map();
  const pending = [kernel];
  while (pending.length > 0) {
    const source = pending.pop();
    for (const [name, overloadMap] of byName) {
      if (SEMANTIC_BUILTIN_DEVICE_HELPERS.has(name)) continue;
      if (included.has(name) || !sourceCallsFunctionThroughDefines(source, name, definesByName)) continue;
      const overloads = selectDeviceFunctionOverloadsForSource(source, name, [...overloadMap.values()], definesByName);
      included.set(name, overloads);
      for (const fn of overloads) pending.push(fn.source);
    }
  }
  return [...included.values()].flat();
}

export function normalizeFunctionPointerDispatch(source, functionPointerTables = []) {
  const tables = functionPointerTables.filter((table) => table?.tableName && Array.isArray(table.entries) && table.entries.length > 0);
  if (tables.length === 0) return source;
  let out = source;
  const aliases = collectFunctionPointerAliasParams(out, tables);
  out = rewriteFunctionPointerTableAssignments(out, tables);
  out = rewriteFunctionPointerNullComparisons(out, aliases);
  for (const table of tables) out = rewriteFunctionPointerTableCalls(out, table);
  for (const alias of aliases) out = rewriteFunctionPointerVariableCalls(out, alias.name, alias.table);
  for (const table of tables) {
    for (const selector of collectFunctionPointerSelectorLocals(out, table)) {
      out = rewriteFunctionPointerVariableCalls(out, selector.name, table);
    }
  }
  return out;
}

function collectFunctionPointerAliasParams(source, tables) {
  const signature = kernelSignature(source);
  if (signature?.params === undefined) return [];
  const byAlias = new Map();
  for (const table of tables) {
    if (typeof table.aliasName === "string" && table.aliasName.length > 0 && !byAlias.has(table.aliasName)) byAlias.set(table.aliasName, table);
  }
  if (byAlias.size === 0) return [];
  const out = [];
  for (const rawParam of splitTopLevel(signature.params)) {
    const param = rawParam.trim();
    for (const [aliasName, table] of byAlias) {
      const match = new RegExp(`^\\s*(?:const\\s+)?${escapeRegExp(aliasName)}\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*$`, "u").exec(param);
      if (match?.[1] !== undefined) out.push({ name: match[1], table });
    }
  }
  return out;
}

function rewriteFunctionPointerTableAssignments(source, tables) {
  let out = source;
  for (const table of tables) {
    const tableName = escapeRegExp(table.tableName);
    const re = new RegExp(`(^|[;{}]\\s*)([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*${tableName}\\s*\\[\\s*([^\\]]+)\\s*\\]\\s*;`, "gmu");
    out = out.replace(re, (_match, prefix, name, selector) => `${prefix}uint ${name} = ${selector.trim()};`);
  }
  return out;
}

function rewriteFunctionPointerNullComparisons(source, aliases) {
  let out = source;
  for (const alias of aliases) {
    const name = escapeRegExp(alias.name);
    out = out
      .replace(new RegExp(`\\b${name}\\s*!=\\s*NULL\\b`, "gu"), `${alias.name} != 0xffffffffu`)
      .replace(new RegExp(`\\b${name}\\s*==\\s*NULL\\b`, "gu"), `${alias.name} == 0xffffffffu`)
      .replace(new RegExp(`\\bNULL\\s*!=\\s*${name}\\b`, "gu"), `0xffffffffu != ${alias.name}`)
      .replace(new RegExp(`\\bNULL\\s*==\\s*${name}\\b`, "gu"), `0xffffffffu == ${alias.name}`);
  }
  return out;
}

function rewriteFunctionPointerTableCalls(source, table) {
  const re = new RegExp(`\\(\\s*\\*\\s*\\(?\\s*${escapeRegExp(table.tableName)}\\s*\\[`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const bracketOpen = source.indexOf("[", match.index);
    const bracketClose = findBalanced(source, bracketOpen, "[", "]");
    if (bracketClose === undefined) {
      re.lastIndex = match.index + 1;
      continue;
    }
    let scan = skipWhitespace(source, bracketClose + 1);
    let parens = 0;
    while (source[scan] === ")") {
      parens++;
      scan = skipWhitespace(source, scan + 1);
    }
    if (parens === 0) {
      re.lastIndex = bracketClose + 1;
      continue;
    }
    if (source[scan] !== "(") {
      re.lastIndex = scan;
      continue;
    }
    const callClose = findBalanced(source, scan, "(", ")");
    if (callClose === undefined) {
      re.lastIndex = scan + 1;
      continue;
    }
    const selector = source.slice(bracketOpen + 1, bracketClose).trim();
    const args = source.slice(scan + 1, callClose);
    out += source.slice(cursor, match.index);
    out += functionPointerDispatchExpression(table, selector, args);
    cursor = callClose + 1;
    re.lastIndex = callClose + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function collectFunctionPointerSelectorLocals(source, table) {
  const out = [];
  const re = new RegExp(`\\buint\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*([^;]+)\\s*;`, "gu");
  let match;
  while ((match = re.exec(source)) !== null) {
    if (match[1] !== undefined && match[2] !== undefined && source.slice(match.index, re.lastIndex).includes(table.tableName) === false) out.push({ name: match[1] });
  }
  return out;
}

function rewriteFunctionPointerVariableCalls(source, name, table) {
  const re = new RegExp(`\\(\\s*\\*\\s*${escapeRegExp(name)}\\s*\\)\\s*\\(`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const callOpen = source.indexOf("(", re.lastIndex - 1);
    const callClose = findBalanced(source, callOpen, "(", ")");
    if (callOpen < 0 || callClose === undefined) {
      re.lastIndex = match.index + 1;
      continue;
    }
    const args = source.slice(callOpen + 1, callClose);
    out += source.slice(cursor, match.index);
    out += functionPointerDispatchExpression(table, name, args);
    cursor = callClose + 1;
    re.lastIndex = callClose + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function functionPointerDispatchExpression(table, selector, args) {
  const entries = [...table.entries].sort((a, b) => Number(a.index) - Number(b.index));
  if (entries.length === 1) return `${entries[0].target}(${args})`;
  const fallback = entries[entries.length - 1];
  let expression = `${fallback.target}(${args})`;
  for (let index = entries.length - 2; index >= 0; index--) {
    const entry = entries[index];
    expression = `((${selector}) == ${entry.index} ? ${entry.target}(${args}) : ${expression})`;
  }
  return expression;
}

export function stripFunctionPointerDeviceGlobals(source, functionPointerTables = []) {
  const aliases = new Set(functionPointerTables.map((table) => table?.aliasName).filter((alias) => typeof alias === "string" && alias.length > 0));
  if (aliases.size === 0) return source;
  let out = source;
  for (const alias of aliases) {
    const re = new RegExp(`\\b__device__\\s+${escapeRegExp(alias)}\\s+[A-Za-z_][A-Za-z0-9_]*(?:\\s*\\[[^\\]]+\\])?\\s*(?:=\\s*[A-Za-z_][A-Za-z0-9_]*)?\\s*;\\s*`, "gu");
    out = out.replace(re, "");
  }
  return out;
}

export function deviceFunctionSignatureShape(source) {
  const brace = source.indexOf("{");
  return source.slice(0, brace < 0 ? source.length : brace).replace(/\s+/gu, " ").trim();
}

export function selectDeviceFunctionOverloadsForSource(source, name, overloads, definesByName) {
  const arities = sourceFunctionCallAritiesThroughDefines(source, name, definesByName);
  const arityList = arities.size === 0 ? [...new Set(overloads.map((fn) => deviceFunctionParamCount(fn.source)))] : [...arities];
  const selected = [];
  for (const arity of arityList) {
    const matches = overloads.filter((fn) => deviceFunctionParamCount(fn.source) === arity);
    const chosen = matches[matches.length - 1];
    if (chosen !== undefined) selected.push(chosen);
  }
  return selected.length > 0 ? selected : overloads.slice(-1);
}

function deviceFunctionParamCount(source) {
  const signature = source.slice(0, source.indexOf("{") < 0 ? source.length : source.indexOf("{"));
  const open = signature.lastIndexOf("(");
  if (open < 0) return 0;
  const close = findBalanced(signature, open, "(", ")") ?? signature.lastIndexOf(")");
  if (close <= open) return 0;
  const params = splitTopLevel(signature.slice(open + 1, close)).map((param) => param.trim()).filter(Boolean);
  return params.length === 1 && params[0] === "void" ? 0 : params.length;
}

export function sourceFunctionCallAritiesThroughDefines(source, name, definesByName) {
  const cleanSource = stripCommentsAndStrings(source);
  const out = sourceFunctionCallArities(cleanSource, name);
  const visited = new Set();
  const pending = [];
  for (const defineName of definesByName.keys()) {
    if (sourceCallsFunction(cleanSource, defineName)) pending.push(defineName);
  }
  while (pending.length > 0) {
    const defineName = pending.pop();
    if (defineName === undefined || visited.has(defineName)) continue;
    visited.add(defineName);
    const value = definesByName.get(defineName);
    if (value === undefined) continue;
    for (const arity of sourceFunctionCallArities(String(value), name)) out.add(arity);
    for (const next of definesByName.keys()) {
      if (!visited.has(next) && sourceCallsFunction(String(value), next)) pending.push(next);
    }
  }
  return out;
}

export function sourceFunctionCallArities(source, name) {
  const out = new Set();
  for (const call of scanFunctionCallReferences(source)) {
    if (call.name === name) out.add(call.args.length);
  }
  for (const call of scanTemplatedFunctionCallReferences(source)) {
    if (call.name === name) out.add(call.args.length);
  }
  return out;
}

function scanTemplatedFunctionCallReferences(source) {
  const calls = [];
  for (const call of scanTemplatedCallReferences(source)) {
    const templateStart = source.indexOf("<", call.start + call.name.length);
    if (templateStart < 0) continue;
    const templateEnd = findBalanced(source, templateStart, "<", ">");
    if (templateEnd === undefined) continue;
    const open = skipWhitespace(source, templateEnd + 1);
    if (source[open] !== "(") continue;
    const close = findBalanced(source, open, "(", ")");
    if (close === undefined) continue;
    calls.push({
      name: call.name,
      args: splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim()),
      start: call.start,
    });
  }
  return calls;
}

function isExplicitTemplateSpecialization(source) {
  return /^\s*template\s*<\s*>/u.test(source);
}

export function sourceCallsFunctionThroughDefines(source, name, definesByName) {
  const cleanSource = stripCommentsAndStrings(source);
  if (sourceCallsFunction(cleanSource, name)) return true;
  const visited = new Set();
  const pending = [];
  for (const [defineName, value] of definesByName) {
    if (!sourceCallsFunction(cleanSource, defineName)) continue;
    visited.add(defineName);
    pending.push(value);
  }
  while (pending.length > 0) {
    const value = stripCommentsAndStrings(pending.pop());
    if (sourceCallsFunction(value, name) || sourceMentionsIdentifier(value, name)) return true;
    for (const [defineName, nextValue] of definesByName) {
      if (visited.has(defineName) || !sourceMentionsIdentifier(value, defineName)) continue;
      visited.add(defineName);
      pending.push(nextValue);
    }
  }
  return false;
}

export function sourceCallsFunction(source, name) {
  if (new RegExp(`(?:^|[^:A-Za-z0-9_])${escapeRegExp(name)}\\s*\\(`, "u").test(source)) return true;
  if (new RegExp(`(?:^|[^:A-Za-z0-9_])${escapeRegExp(name)}\\s*<[^;{}()]*>\\s*\\(`, "u").test(source)) return true;
  return scanTemplatedCallReferences(source).some((call) => call.name === name);
}

export function stripCommentsAndStrings(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//gu, " ")
    .replace(/\/\/[^\n\r]*/gu, " ")
    .replace(/"(?:\\.|[^"\\])*"/gu, "\"\"")
    .replace(/'(?:\\.|[^'\\])*'/gu, "''");
}

export function sourceLaunchesKernel(source, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\s*<<\\s*<`, "u").test(source) ||
    scanTemplatedKernelReferences(source).some((launch) => launch.name === name && launch.kind === "launch");
}

export function sourceMentionsIdentifier(source, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\b`, "u").test(source);
}

export function reachableDeclarations(declarations, nameOf, source) {
  return declarations.filter((declaration) => {
    const name = nameOf(declaration);
    return name !== undefined && sourceMentionsIdentifier(source, name);
  });
}

export function constantDeclarationName(declaration) {
  return trailingDeclaratorName(declaration.replace(/\b__constant__\b/u, " "));
}

export function deviceGlobalDeclarationName(declaration) {
  return trailingDeclaratorName(declaration.replace(/\b(?:static\s+)?__device__\b/u, " "));
}

export function textureDeclarationName(declaration) {
  return /\b([A-Za-z_][A-Za-z0-9_]*)\s*;\s*$/u.exec(declaration)?.[1];
}

function trailingDeclaratorName(declaration) {
  const withoutInitializer = declaration
    .replace(/=[\s\S]*$/u, "")
    .replace(/;\s*$/u, "")
    .trim();
  return /(?:^|[\s*&])([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])*\s*$/u.exec(withoutInitializer)?.[1];
}

export function isMacroIdentifier(name) {
  return /^[A-Za-z_][A-Za-z0-9_]*$/u.test(name);
}

export function functionDefineName(declaration) {
  return /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/u.exec(declaration)?.[1];
}

export function recordDeclarationName(declaration) {
  const unionMatch = /\bunion\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{/u.exec(declaration);
  if (unionMatch?.[1] !== undefined) return unionMatch[1];
  const match = /\b(?:template\s*<[^<>]*>\s*)?(?:typedef\s+)?struct(?:\s+(?:__align__|alignas)\s*\([^)]*\))*\s*([A-Za-z_][A-Za-z0-9_]*)?\s*\{[\s\S]*?\}\s*([A-Za-z_][A-Za-z0-9_]*)?\s*;/u.exec(declaration);
  return match?.[2] ?? match?.[1];
}

export function podRecordDeclarationVectorAlias(declaration, definesByName) {
  const name = recordDeclarationName(declaration);
  if (name === undefined) return undefined;
  return collectPodRecordVectorAliases(declaration, definesByName)
    .find((record) => record.name === name)?.vectorType;
}

export function scalarizedPodRecordDeclaration(declaration, definesByName) {
  const name = recordDeclarationName(declaration);
  if (name === undefined) return undefined;
  if (SEMANTIC_RECORD_CARRIERS.has(name)) return undefined;
  return collectScalarizedPodRecords(declaration, definesByName)
    .find((record) => record.name === name);
}

export function templatedScalarizedPodRecordDeclaration(declaration, usageSource) {
  const name = recordDeclarationName(declaration);
  if (name === undefined || !new RegExp(`\\b${escapeRegExp(name)}\\s*<`, "u").test(usageSource)) return undefined;
  const record = scanStructRecordDeclarations(declaration).find((candidate) => (candidate.aliasName ?? candidate.tagName) === name);
  if (record?.rawTemplateParams === undefined) return undefined;
  const stripped = stripRecordEnums(record.body).body;
  if (/\(|\)|\b(?:public|private|protected|operator|union)\b/u.test(stripped)) return undefined;
  const templateNames = new Set(splitTopLevel(record.rawTemplateParams).map(parseTemplateParam).filter(Boolean).map((param) => param.name));
  for (const raw of stripped.split(";")) {
    const line = stripLineComment(raw).trim();
    if (line.length === 0) continue;
    if (/[(){}*&]/u.test(line)) return undefined;
    const match = /^(?:const\s+|volatile\s+)*([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)?)\s+([A-Za-z_][A-Za-z0-9_]*)(.*)$/u.exec(line);
    if (match?.[1] === undefined || match[2] === undefined) return undefined;
    if (!templateNames.has(match[1].trim()) && normalizeTemplateTypeArgument(match[1], collectRecordEnumTypeDefines(record.body)) === undefined) return undefined;
  }
  return { name };
}

export function bitpackedShortUnionDeclaration(declaration) {
  const name = recordDeclarationName(declaration);
  if (name === undefined || !/\bunion\b/u.test(declaration)) return undefined;
  return collectBitpackedShortUnions(declaration).find((record) => record.name === name);
}

export function kernelParamNames(kernel) {
  return (kernelSignature(kernel)?.params ?? "")
    .split(",")
    .map((param) => /([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*$/u.exec(param.trim())?.[1])
    .filter(Boolean);
}

function kernelSignature(kernel) {
  const header = kernel.slice(0, kernel.indexOf("{") < 0 ? kernel.length : kernel.indexOf("{"));
  const match = /__global__[\s\S]*?\bvoid\b\s*(?:(?:__launch_bounds__)\s*\([^)]*\)\s*)*(?:static\s+|extern\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(/u.exec(header);
  if (match?.[1] !== undefined && match.index !== undefined) {
    const open = header.indexOf("(", match.index + match[0].length - 1);
    const params = balancedParenContents(header, open);
    return params === undefined ? undefined : { name: match[1], params };
  }
  const globalIndex = header.indexOf("__global__");
  if (globalIndex < 0) return undefined;
  const open = header.indexOf("(", globalIndex + "__global__".length);
  const params = balancedParenContents(header, open);
  return params === undefined ? undefined : { name: undefined, params };
}
