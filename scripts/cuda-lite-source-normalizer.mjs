const SEMANTIC_BUILTIN_DEVICE_HELPERS = new Set([
  "vec_at",
  "load128",
  "load128cs",
  "store128",
  "store128cs",
  "store128cg",
  "div_ceil",
]);

export function createKernelCompilationUnit({
  kernel,
  siblingKernels = [],
  definesByName = new Map(),
  templateArgumentsByKernelName = new Map(),
  functionDeclarations = [],
  deviceFunctions = [],
  constantDeclarations = [],
  textureDeclarations = [],
}) {
  const aliasContext = [
    kernel,
    ...siblingKernels,
    ...functionDeclarations,
    ...deviceFunctions.map((fn) => fn.source),
    ...constantDeclarations,
  ].join("\n");
  const aliasDefines = collectTypeAliasDefines(aliasContext);
  const carrierDefines = collectCarrierMemberDefines(aliasContext, mergeDefineMaps(definesByName, aliasDefines));
  const effectiveDefines = mergeDefineMaps(definesByName, aliasDefines, carrierDefines);
  const specializedKernel = specializeTemplateFromLaunchContext(kernel, templateArgumentsByKernelName, effectiveDefines);
  const params = new Set(kernelParamNames(specializedKernel));
  const referencedDeviceFunctionsRaw = referencedDeviceFunctionClosure(specializedKernel, deviceFunctions, effectiveDefines);
  const referencedDeviceFunctionNames = new Set(referencedDeviceFunctionsRaw.map((fn) => fn.name));
  const deviceTemplateArgumentsByName = collectDeviceFunctionTemplateArguments(
    [
      specializedKernel,
      ...referencedDeviceFunctionsRaw.map((fn) => fn.source),
    ],
    referencedDeviceFunctionNames,
    effectiveDefines,
  );
  const referencedSiblingKernelsRaw = siblingKernels.filter((sibling) => {
    const name = kernelDefinitionName(sibling);
    return name !== undefined && sourceLaunchesKernel(specializedKernel, name);
  });
  const templateNames = new Set([
    ...templateParameterNames(specializedKernel),
    ...referencedSiblingKernelsRaw.flatMap((sibling) => templateParameterNames(sibling)),
    ...referencedDeviceFunctionsRaw.flatMap((fn) => templateParameterNames(fn.source)),
  ]);
  const defines = [...effectiveDefines]
    .filter(([name]) => isMacroIdentifier(name) && !params.has(name) && !templateNames.has(name))
    .map(([name, value]) => `#define ${name} ${value}`);
  const referencedDeviceFunctions = referencedDeviceFunctionsRaw
    .map((fn) => ({
      ...fn,
      source: stripKnownTemplateCallArguments(
        stripKernelLaunchTemplateArguments(specializeDeviceFunctionFromCallContext(
          fn.source,
          deviceTemplateArgumentsByName.get(fn.name),
          effectiveDefines,
        )),
        referencedDeviceFunctionNames,
      ),
    }));
  const referencedSiblingKernels = referencedSiblingKernelsRaw.map((sibling) => stripKernelLaunchTemplateArguments(
    specializeTemplateFromLaunchContext(sibling, templateArgumentsByKernelName, effectiveDefines),
  ));
  const macroScope = [
    specializedKernel,
    ...referencedDeviceFunctions.map((fn) => fn.source),
    ...referencedSiblingKernels,
  ].join("\n");
  const functionMacros = functionDeclarations.filter((declaration) => {
    const name = functionDefineName(declaration);
    return name !== undefined && sourceMentionsIdentifier(macroScope, name);
  });
  const unit = [
    defines.join("\n"),
    functionMacros.map(normalizeFunctionMacro).join("\n"),
    referencedDeviceFunctions.map((fn) => fn.source).join("\n"),
    constantDeclarations.join("\n"),
    textureDeclarations.join("\n"),
    referencedSiblingKernels.join("\n"),
    stripKnownTemplateCallArguments(stripKernelLaunchTemplateArguments(specializedKernel), referencedDeviceFunctionNames),
  ].filter((part) => part.trim().length > 0).join("\n");
  const withCarrierMembers = normalizeCarrierMemberReferences(unit, effectiveDefines);
  return normalizeCppTemplateCarrierSyntax(normalizeSincosHelpers(normalizeSharedMemoryHelpers(normalizePacked128MemoryHelpers(stripSupportedTypeAliasDeclarations(withCarrierMembers)))));
}

export function collectCudaLiteContextDefines(source) {
  return collectObjectDefines(source);
}

function normalizeFunctionMacro(macro) {
  const cpAsync = semanticCpAsyncMacro(macro);
  return cpAsync ?? macro;
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
  const addCandidate = (name, args) => {
    if (args.length === 0) return;
    const normalizedArgs = args.map((arg) => resolveTemplateArgument(arg, definesByName));
    const previous = candidates.get(name);
    if (previous === undefined || templateArgumentScore(normalizedArgs) > templateArgumentScore(previous)) {
      candidates.set(name, normalizedArgs);
    }
  };
  for (const launch of scanTemplatedKernelReferences(source)) {
    addCandidate(launch.name, launch.args);
  }
  for (const propagated of collectWrapperPropagatedTemplateArguments(source, definesByName)) {
    addCandidate(propagated.name, propagated.args);
  }
  for (const inferred of collectLaunchInferredTemplateArguments(source, definesByName)) {
    addCandidate(inferred.name, inferred.args);
  }
  return candidates;
}

function referencedDeviceFunctionClosure(kernel, deviceFunctions, definesByName = new Map()) {
  const byName = new Map(deviceFunctions.map((fn) => [fn.name, fn]));
  const included = new Map();
  const pending = [kernel];
  while (pending.length > 0) {
    const source = pending.pop();
    for (const [name, fn] of byName) {
      if (SEMANTIC_BUILTIN_DEVICE_HELPERS.has(name)) continue;
      if (included.has(name) || !sourceCallsFunctionThroughDefines(source, name, definesByName)) continue;
      included.set(name, fn);
      pending.push(fn.source);
    }
  }
  return [...included.values()];
}

function sourceCallsFunctionThroughDefines(source, name, definesByName) {
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

function sourceCallsFunction(source, name) {
  if (new RegExp(`\\b${escapeRegExp(name)}\\s*\\(`, "u").test(source)) return true;
  return scanTemplatedCallReferences(source).some((call) => call.name === name);
}

function stripCommentsAndStrings(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//gu, " ")
    .replace(/\/\/[^\n\r]*/gu, " ")
    .replace(/"(?:\\.|[^"\\])*"/gu, "\"\"")
    .replace(/'(?:\\.|[^'\\])*'/gu, "''");
}

function sourceLaunchesKernel(source, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\s*<<\\s*<`, "u").test(source) ||
    scanTemplatedKernelReferences(source).some((launch) => launch.name === name && launch.kind === "launch");
}

function sourceMentionsIdentifier(source, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\b`, "u").test(source);
}

function isMacroIdentifier(name) {
  return /^[A-Za-z_][A-Za-z0-9_]*$/u.test(name);
}

function functionDefineName(declaration) {
  return /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/u.exec(declaration)?.[1];
}

function kernelParamNames(kernel) {
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

function specializeTemplateFromLaunchContext(source, templateArgumentsByKernelName, definesByName = new Map()) {
  const name = kernelDefinitionName(source);
  if (name === undefined) return source;
  const args = templateArgumentsByKernelName.get(name);
  if (args === undefined || args.length === 0) return source;
  return rewriteFirstTemplateHeader(source, args, definesByName);
}

function specializeDeviceFunctionFromCallContext(source, args, definesByName = new Map()) {
  const defaults = templateHeaderDefaultArguments(source, definesByName);
  if ((args === undefined || args.length === 0) && defaults.length === 0) return source;
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
  const typeEnv = templateTypeEnvironment(parsedParams, args, definesByName);
  const rewritten = params.map((param, index) => rewriteTemplateParam(param, args[index], definesByName)).join(", ");
  const body = substituteTemplateTypes(source.slice(close), typeEnv);
  return `${source.slice(0, open + 1)}${rewritten}${body}`;
}

function rewriteTemplateParam(param, arg, definesByName = new Map()) {
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

function parseTemplateParam(param) {
  const withoutDefault = splitTemplateParamDefault(param);
  const head = withoutDefault.head
    .replace(/\b(?:const|constexpr)\b/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  const valueMatch = /^(?:(?:unsigned\s+)?(?:int|long|short)|uint|size_t|bool)\s+([A-Za-z_][A-Za-z0-9_]*)$/u.exec(head);
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

function normalizeTemplateValueArgument(arg, valueType) {
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

function templateTypeEnvironment(params, args, definesByName = new Map()) {
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

function templateDefaultEnvironment(source, definesByName = new Map()) {
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

function templateHeaderDefaultArguments(source, definesByName = new Map()) {
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

function templateParamDefaultValue(param) {
  if (param.defaultStart === undefined || param.source === undefined) return undefined;
  return param.source.slice(param.defaultStart + 1).trim();
}

function numericTemplateDefines(definesByName) {
  const values = new Map();
  for (const [name, raw] of definesByName) {
    const normalized = normalizeTemplateValueArgument(String(raw), "int");
    if (normalized !== undefined) values.set(name, normalized);
  }
  return values;
}

function resolveTemplateDefineValue(value, definesByName) {
  const trimmed = value.trim();
  return definesByName.get(trimmed) ?? trimmed;
}

function normalizeTemplateTypeArgument(arg, definesByName = new Map()) {
  if (arg === undefined) return undefined;
  let type = arg.trim()
    .replace(/\b(?:const|volatile|typename|class|struct)\b/gu, " ")
    .replace(/[&*]/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  if (type.length === 0 || type.includes("(") || type.includes(")")) return undefined;
  const packed = /^Packed128\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>$/u.exec(type)?.[1];
  if (packed !== undefined) {
    const elementType = normalizeTemplateTypeArgument(definesByName.get(packed) ?? packed, definesByName);
    if (elementType === "float") return "float4";
    if (elementType === "int") return "int4";
    if (elementType === "uint") return "uint4";
    return undefined;
  }
  if (type.includes("<") || type.includes(">")) return undefined;
  const mapped = definesByName.get(type);
  if (mapped && /^[A-Za-z_][A-Za-z0-9_]*$/u.test(mapped)) {
    const normalized = normalizeTemplateTypeArgument(mapped, definesByName);
    if (normalized !== undefined) return normalized;
    type = mapped;
  }
  if (type === "__half" || type === "half_t") return "half";
  if (type === "unsigned int" || type === "unsigned") return "uint";
  if (type === "unsigned char" || type === "uchar" || type === "uint8_t") return "uint";
  if (type === "signed int" || type === "signed") return "int";
  if (type === "signed char" || type === "char" || type === "int8_t") return "int";
  if (
    type === "clock_t" ||
    type === "size_type" ||
    type === "curandState" ||
    type === "CUtexObject" ||
    type === "CUtensorMap" ||
    type === "cudaGraphConditionalHandle" ||
    type === "__nv_fp8_storage_t"
  ) return "uint";
  if (type === "long long" || type === "long" || type === "short" || type === "short int") return "int";
  if (type === "uchar2") return "uint2";
  if (type === "uchar3") return "uint3";
  if (type === "uchar4") return "uint4";
  if (type === "char2") return "int2";
  if (type === "char3") return "int3";
  if (type === "char4") return "int4";
  const supported = new Set(["float", "int", "uint", "half", "bool", "float2", "float3", "float4", "half2", "int2", "int3", "int4", "uint2", "uint3", "uint4"]);
  return supported.has(type) ? type : undefined;
}

function normalizeCppTemplateCarrierSyntax(source) {
  return source
    .replace(/\bstd\s*::\s*bool_constant\s*<\s*(true|false)\s*>\s*\{\s*\}/gu, "$1")
    .replace(/\bstd\s*::\s*bool_constant\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s*(?=[,)])/gu, (_match, name) => `bool __bg_bool_constant_${name}`);
}

function normalizeCarrierMemberReferences(source, definesByName) {
  const entries = [...definesByName]
    .filter(([name]) => /^[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*$/u.test(name))
    .sort((a, b) => b[0].length - a[0].length);
  if (entries.length === 0) return source;
  let out = source;
  for (const [name, value] of entries) {
    const [owner, member] = name.split("::");
    if (!owner || !member) continue;
    const re = new RegExp(`(?:\\btypename\\s+)?\\b${escapeRegExp(owner)}\\s*::\\s*${escapeRegExp(member)}\\b`, "gu");
    out = out.replace(re, value);
  }
  return out;
}

function normalizeSharedMemoryHelpers(source) {
  return source
    .replace(
      /\bSharedMemory\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\2\s*\.\s*getPointer\s*\(\s*\)\s*;/gu,
      (_match, templateType, _helperName, pointerType, pointerName) =>
        templateType === pointerType ? `extern __shared__ ${templateType} ${pointerName}[];` : _match,
    )
    .replace(
      /\b([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*SharedMemory\s*<\s*\1\s*>\s*\(\s*\)\s*;/gu,
      "extern __shared__ $1 $2[];",
    );
}

function normalizeSincosHelpers(source) {
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
}

function sincosReplacement(args) {
  const [phase, sinTarget, cosTarget] = args;
  const sinLvalue = addressTarget(sinTarget);
  const cosLvalue = addressTarget(cosTarget);
  if (!phase || !sinLvalue || !cosLvalue) return undefined;
  return `${sinLvalue} = sinf(${phase}); ${cosLvalue} = cosf(${phase})`;
}

function addressTarget(expression) {
  if (expression === undefined) return undefined;
  const trimmed = expression.trim();
  if (!trimmed.startsWith("&")) return undefined;
  const target = trimmed.slice(1).trim();
  if (/^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\[[^\]]+\])?$/u.test(target)) return target;
  return undefined;
}

function normalizePacked128MemoryHelpers(source) {
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

function pointerBaseIdentifier(pointer) {
  const expression = pointer
    .replace(/\breinterpret_cast\s*<[^>]+>\s*\(/gu, "(")
    .replace(/\b(?:const|volatile)\b/gu, " ")
    .trim();
  return /(?:^|[^A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\b/u.exec(expression)?.[1];
}

function collectObjectDefines(source) {
  const defines = new Map();
  for (const line of source.split(/\r?\n/u)) {
    const stripped = stripLineComment(line);
    const match = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)(?!\()\s+(.+?)\s*$/u.exec(stripped);
    if (match?.[1] !== undefined && match[2] !== undefined) {
      defines.set(match[1], match[2].trim());
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
  for (const [name, value] of collectCarrierMemberDefines(source, defines)) defines.set(name, value);
  return defines;
}

function collectTypeAliasDefines(source) {
  const defines = new Map();
  for (const line of source.split(/\r?\n/u)) {
    const alias = parseSimpleTypeAlias(stripLineComment(line), defines);
    if (alias !== undefined) defines.set(alias.name, alias.value);
  }
  return defines;
}

function collectCarrierMemberDefines(source, initialDefines = new Map()) {
  const out = new Map();
  const carriers = scanTemplateStructCarriers(source);
  if (carriers.size === 0) return out;
  const aliases = scanCarrierAliases(source);
  for (const alias of aliases) {
    const carrier = carriers.get(alias.templateName);
    if (carrier === undefined) continue;
    const env = templateEnvironment(carrier.params, alias.args, mergeDefineMaps(initialDefines, out));
    if (env.size === 0) continue;
    const memberEnv = new Map(env);
    for (const member of carrier.members) {
      if (member.kind === "type") {
        const substituted = substituteTemplateArgument(member.value, memberEnv, mergeDefineMaps(initialDefines, out));
        const normalized = normalizeTemplateTypeArgument(substituted, mergeDefineMaps(initialDefines, out));
        if (normalized === undefined) continue;
        out.set(`${alias.name}::${member.name}`, normalized);
        memberEnv.set(member.name, normalized);
        continue;
      }
      const value = evaluateTemplateIntegerExpression(member.value, memberEnv);
      if (value === undefined) continue;
      out.set(`${alias.name}::${member.name}`, value);
      memberEnv.set(member.name, value);
    }
  }
  return out;
}

function scanTemplateStructCarriers(source) {
  const carriers = new Map();
  let index = 0;
  while (index < source.length) {
    const match = /\btemplate\s*</u.exec(source.slice(index));
    if (match === null) break;
    const templateStart = index + match.index;
    const open = source.indexOf("<", templateStart);
    const close = findBalanced(source, open, "<", ">");
    if (close === undefined) break;
    const afterTemplate = skipWhitespace(source, close + 1);
    const structMatch = /^(?:struct|class)\s+(?:alignas\s*\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)/u.exec(source.slice(afterTemplate));
    if (structMatch?.[1] === undefined) {
      index = close + 1;
      continue;
    }
    const brace = source.indexOf("{", afterTemplate + structMatch[0].length);
    const end = findBalanced(source, brace, "{", "}");
    if (brace < 0 || end === undefined) {
      index = close + 1;
      continue;
    }
    carriers.set(structMatch[1], {
      name: structMatch[1],
      params: splitTopLevel(source.slice(open + 1, close)).map(parseTemplateParam).filter(Boolean),
      members: scanCarrierMembers(source.slice(brace + 1, end)),
    });
    index = end + 1;
  }
  return carriers;
}

function scanCarrierMembers(body) {
  const members = [];
  for (const match of body.matchAll(/\busing\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);/gu)) {
    const [, name, value] = match;
    if (name && value) members.push({ kind: "type", name, value: value.trim() });
  }
  for (const match of body.matchAll(/\bstatic\s+constexpr\s+(?:const\s+)?(?:int|uint|unsigned\s+int|size_t|bool)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);/gu)) {
    const [, name, value] = match;
    if (name && value) members.push({ kind: "value", name, value: value.trim() });
  }
  return members;
}

function scanCarrierAliases(source) {
  const aliases = [];
  let index = 0;
  while (index < source.length) {
    const match = /\busing\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*<|\btypedef\s+([A-Za-z_][A-Za-z0-9_]*)\s*</u.exec(source.slice(index));
    if (match === null) break;
    const start = index + match.index;
    const usingName = match[1];
    const usingTemplateName = match[2];
    const typedefTemplateName = match[3];
    const templateName = usingTemplateName ?? typedefTemplateName;
    const open = source.indexOf("<", start + match[0].length - 1);
    const close = findBalanced(source, open, "<", ">");
    const semi = close === undefined ? -1 : skipWhitespace(source, close + 1);
    const typedefName = close === undefined ? undefined : /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*;/u.exec(source.slice(close + 1))?.[1];
    const name = usingName ?? typedefName;
    if (name && templateName && close !== undefined && source[semi + (typedefName === undefined ? 0 : typedefName.length)] === ";") {
      aliases.push({
        name,
        templateName,
        args: splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim()),
      });
      index = (typedefName === undefined ? semi : semi + typedefName.length) + 1;
      continue;
    }
    index = start + 1;
  }
  return aliases;
}

function isKnownCarrierAlias(raw, definesByName) {
  const alias = raw.trim();
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/u.test(alias)) return false;
  for (const name of definesByName.keys()) {
    if (name.startsWith(`${alias}::`)) return true;
  }
  return false;
}

function parseSimpleTypeAlias(line, defines) {
  const typedefMatch = /^\s*typedef\s+(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*$/u.exec(line);
  if (typedefMatch !== null) {
    const [, sourceType, alias] = typedefMatch;
    const value = normalizeTemplateTypeArgument(sourceType, defines);
    return value && alias ? { name: alias, value } : undefined;
  }
  const usingMatch = /^\s*using\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*;\s*$/u.exec(line);
  if (usingMatch !== null) {
    const [, alias, sourceType] = usingMatch;
    const value = normalizeTemplateTypeArgument(sourceType, defines);
    return value && alias ? { name: alias, value } : undefined;
  }
  return undefined;
}

function stripSupportedTypeAliasDeclarations(source) {
  const defines = new Map();
  return source
    .split(/\r?\n/u)
    .filter((line) => {
      const alias = parseSimpleTypeAlias(stripLineComment(line), defines);
      if (alias === undefined) return true;
      defines.set(alias.name, alias.value);
      return false;
    })
    .join("\n");
}

function parseSimpleIntegerConstant(line) {
  const match = /^\s*((?:(?:static|constexpr|const)\s+)*)(?:int|uint|unsigned\s+int|size_t)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9A-Fa-fxXuUlL\s()+\-*/%<>&|^?:.]+)\s*;\s*$/u.exec(line);
  if (match === null) return undefined;
  const [, qualifiers, name, value] = match;
  if (!name || !value || !/\b(?:const|constexpr)\b/u.test(qualifiers ?? "")) return undefined;
  return { name, value: value.trim() };
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

function mergeDefineMaps(...maps) {
  const merged = new Map();
  for (const map of maps) {
    for (const [name, value] of map) merged.set(name, value);
  }
  return merged;
}

function substituteTemplateTypes(source, typeEnv) {
  if (typeEnv.size === 0) return source;
  return source.replace(/\b[A-Za-z_][A-Za-z0-9_]*\b/gu, (name) => typeEnv.get(name) ?? name);
}

function templateParameterNames(source) {
  const names = [];
  for (const header of templateHeaders(source)) {
    for (const param of splitTopLevel(header)) {
      const parsed = parseTemplateParam(param);
      if (parsed?.name !== undefined) names.push(parsed.name);
    }
  }
  return names;
}

function templateHeaders(source) {
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

function stripKernelLaunchTemplateArguments(source) {
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

function stripKnownTemplateCallArguments(source, names) {
  let out = source;
  for (const name of names) out = stripTemplateCallArgumentsForName(out, name);
  return out;
}

function stripTemplateCallArgumentsForName(source, name) {
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

function scanTemplatedKernelReferences(source) {
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

function scanRegexTemplatedKernelLaunches(source) {
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

function dedupeTemplateRefs(refs) {
  const seen = new Set();
  return refs.filter((ref) => {
    const key = `${ref.kind}:${ref.name}:${ref.templateStart}:${ref.templateEnd}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function collectWrapperPropagatedTemplateArguments(source, definesByName = new Map()) {
  const propagated = [];
  const calls = scanTemplatedCallReferences(source);
  for (const fn of scanTemplatedFunctionDefinitions(source)) {
    const fnCalls = calls.filter((call) => call.name === fn.name && call.args.length > 0);
    if (fnCalls.length === 0) continue;
    const kernelRefs = scanTemplatedKernelReferences(fn.body);
    if (kernelRefs.length === 0) continue;
    for (const call of fnCalls) {
      const env = templateEnvironment(fn.templateParams, call.args, definesByName);
      if (env.size === 0) continue;
      extendEnvironmentWithConstexprs(env, fn.body);
      for (const ref of kernelRefs) {
        const args = ref.args.map((arg) => substituteTemplateArgument(arg, env, definesByName));
        if (templateArgumentScore(args) === 0) continue;
        propagated.push({ name: ref.name, args });
      }
    }
  }
  return propagated;
}

function collectLaunchInferredTemplateArguments(source, definesByName = new Map()) {
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

function scanTemplatedKernelDefinitions(source) {
  return scanTemplatedFunctionDefinitions(source)
    .filter((fn) => /\b__global__\b/u.test(fn.signature))
    .map((fn) => ({
      ...fn,
      kernelParams: parseKernelSignatureParams(fn.signature),
    }));
}

function scanKernelLaunchCalls(source) {
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

function findCudaLaunchClose(source, launchOpen) {
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

function inferKernelTemplateArgs(kernel, launch, source, definesByName = new Map()) {
  const env = new Map();
  const symbols = collectVisiblePointerTypes(source.slice(0, launch.start), definesByName);
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
  return kernel.templateParams.map((param) => param.kind === "type" || param.kind === "value" ? env.get(param.name) : undefined);
}

function parseKernelSignatureParams(signature) {
  const open = signature.indexOf("(");
  const params = balancedParenContents(signature, open);
  if (params === undefined || params.trim().length === 0) return [];
  return splitTopLevel(params).map(parseKernelParamForInference);
}

function parseKernelParamForInference(param) {
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

function collectVisiblePointerTypes(source, initialDefines = new Map()) {
  const symbols = new Map();
  const defines = mergeDefineMaps(initialDefines, collectObjectDefines(source));
  const re = /\b(?:const\s+)?([A-Za-z_][A-Za-z0-9_:]*)\s*(?:\*\s*)+\s*([A-Za-z_][A-Za-z0-9_]*)\b/gu;
  for (const match of source.matchAll(re)) {
    const [, type, name] = match;
    if (type && name) symbols.set(name, normalizeTemplateTypeArgument(defines.get(type) ?? type, defines) ?? type);
  }
  return symbols;
}

function inferArgumentPointerType(arg, symbols) {
  const cast = /^\(\s*(?:const\s+)?([A-Za-z_][A-Za-z0-9_:]*)\s*\*\s*\)/u.exec(arg)?.[1];
  if (cast !== undefined) return cast;
  const name = /^[A-Za-z_][A-Za-z0-9_]*/u.exec(arg.trim())?.[0];
  return name === undefined ? undefined : symbols.get(name);
}

function inferBoolConstantArgument(arg) {
  const explicit = /\bstd\s*::\s*bool_constant\s*<\s*(true|false)\s*>\s*\{\s*\}/u.exec(arg)?.[1];
  if (explicit !== undefined) return explicit;
  if (/^\s*true\s*$/u.test(arg)) return "true";
  if (/^\s*false\s*$/u.test(arg)) return "false";
  return undefined;
}

function scanTemplatedFunctionDefinitions(source) {
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

function templatedFunctionName(signature) {
  const open = signature.lastIndexOf("(");
  if (open < 0) return undefined;
  const before = signature.slice(0, open).trim();
  return /([A-Za-z_][A-Za-z0-9_]*)\s*$/u.exec(before)?.[1];
}

function scanTemplatedCallReferences(source) {
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
    });
    index = afterTemplate + 1;
  }
  return refs;
}

function collectDeviceFunctionTemplateArguments(sources, names, definesByName = new Map()) {
  const out = new Map();
  for (const source of sources) {
    const env = templateDefaultEnvironment(source, definesByName);
    for (const call of scanTemplatedCallReferences(source)) {
      if (!names.has(call.name)) continue;
      const args = call.args.map((arg) => substituteTemplateArgument(arg, env));
      if (templateArgumentScore(args) === 0) continue;
      const previous = out.get(call.name);
      if (previous === undefined || templateArgumentScore(args) > templateArgumentScore(previous)) {
        out.set(call.name, args);
      }
    }
  }
  return out;
}

function templateEnvironment(params, args, definesByName = new Map()) {
  const env = new Map();
  let argIndex = 0;
  for (const param of params) {
    const arg = args[argIndex++];
    if (!param || arg === undefined) continue;
    if (param.kind === "value") {
      const normalized = normalizeTemplateValueArgument(resolveTemplateDefineValue(arg, definesByName), param.valueType);
      if (normalized !== undefined) env.set(param.name, normalized);
    } else if (param.kind === "type" && /^[A-Za-z_][A-Za-z0-9_:]*$/u.test(arg.trim())) {
      env.set(param.name, resolveTemplateTypeAliasArgument(arg, definesByName));
    }
  }
  return env;
}

function extendEnvironmentWithConstexprs(env, body) {
  const re = /\bconstexpr\s+(?:const\s+)?(?:int|uint|unsigned\s+int|size_t|bool)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);/gu;
  let changed = true;
  while (changed) {
    changed = false;
    for (const match of body.matchAll(re)) {
      const [, name, expression] = match;
      if (!name || !expression || env.has(name)) continue;
      const value = evaluateTemplateIntegerExpression(expression, env);
      if (value === undefined) continue;
      env.set(name, value);
      changed = true;
    }
  }
}

function substituteTemplateArgument(arg, env, definesByName = new Map()) {
  const value = evaluateTemplateIntegerExpression(arg, env);
  if (value !== undefined) return value;
  return resolveTemplateTypeAliasArgument(
    arg.replace(/\b[A-Za-z_][A-Za-z0-9_]*\b/gu, (name) => env.get(name) ?? name).trim(),
    definesByName,
  );
}

function resolveTemplateTypeAliasArgument(arg, definesByName) {
  const trimmed = arg.trim();
  if (!/^[A-Za-z_][A-Za-z0-9_:]*$/u.test(trimmed)) return trimmed;
  return normalizeTemplateTypeArgument(definesByName.get(trimmed) ?? trimmed, definesByName) ?? trimmed;
}

function resolveTemplateArgument(arg, definesByName) {
  const trimmed = resolveTemplateTypeAliasArgument(arg, definesByName);
  if (trimmed === "true" || trimmed === "false") return trimmed;
  if (!/^[A-Za-z_][A-Za-z0-9_:]*$/u.test(trimmed)) return trimmed;
  const resolved = resolveTemplateDefineValue(trimmed, definesByName);
  if (resolved === "true" || resolved === "false") return resolved;
  const value = normalizeTemplateValueArgument(resolved, "int");
  return value ?? trimmed;
}

function evaluateTemplateIntegerExpression(expression, env) {
  const withTypeLayout = expression.replace(/\b(sizeof|alignof)\s*\(\s*([A-Za-z_][A-Za-z0-9_:]*)\s*\)/gu, (_match, op, typeName) => {
    const concreteType = env.get(typeName) ?? typeName;
    const size = op === "sizeof" ? sizeofType(concreteType) : alignofType(concreteType);
    return size === undefined ? `__unknown_${op}_${typeName}` : String(size);
  });
  const substituted = withTypeLayout
    .replace(/\btrue\b/gu, "1")
    .replace(/\bfalse\b/gu, "0")
    .replace(/\b[A-Za-z_][A-Za-z0-9_]*\b/gu, (name) => env.get(name) ?? `__unknown_${name}`);
  if (/__unknown_/u.test(substituted)) return undefined;
  const normalized = substituted.replace(/([0-9])(?:u|U|l|L)+\b/gu, "$1");
  if (!/^[0-9A-Fa-fxX\s()+\-*/%<>=!&|^?:.]+$/u.test(normalized)) return undefined;
  try {
    const result = Function(`"use strict"; return (${normalized});`)();
    if (typeof result === "boolean") return result ? "1" : "0";
    if (Number.isFinite(result) && Number.isInteger(result)) return String(result);
  } catch {
    return undefined;
  }
  return undefined;
}

function sizeofType(typeName) {
  switch (typeName) {
    case "char":
    case "signed char":
    case "unsigned char":
    case "uchar":
    case "int8_t":
    case "uint8_t":
      return 1;
    case "half":
    case "__half":
    case "short":
    case "short int":
    case "unsigned short":
      return 2;
    case "half2":
    case "float":
    case "int":
    case "uint":
    case "unsigned":
    case "unsigned int":
    case "signed":
    case "signed int":
    case "long":
    case "long long":
    case "size_t":
    case "clock_t":
    case "bool":
      return 4;
    case "float2":
    case "int2":
    case "uint2":
    case "cufftComplex":
      return 8;
    case "float3":
    case "int3":
    case "uint3":
      return 12;
    case "float4":
    case "int4":
    case "uint4":
      return 16;
    default:
      return undefined;
  }
}

function alignofType(typeName) {
  if (typeName === "float3" || typeName === "int3" || typeName === "uint3") return 4;
  if (typeName === "char3" || typeName === "uchar3") return 1;
  return sizeofType(typeName);
}

function templateArgumentScore(args) {
  return args.reduce((score, arg) => {
    if (normalizeTemplateTypeArgument(arg) !== undefined) return score + 2;
    if (normalizeTemplateValueArgument(arg, "int") !== undefined) return score + 1;
    return score;
  }, 0);
}

function scanCudaFuncSetAttributeTemplateReferences(source) {
  const refs = [];
  for (const match of source.matchAll(/\bcudaFuncSetAttribute\s*\(/gu)) {
    const open = source.indexOf("(", match.index);
    const close = findBalanced(source, open, "(", ")");
    if (close === undefined) continue;
    const firstArg = splitTopLevel(source.slice(open + 1, close))[0]?.trim();
    if (!firstArg) continue;
    const ref = parseTemplatedCallee(firstArg);
    if (ref !== undefined) refs.push({ ...ref, kind: "attribute", templateStart: -1, templateEnd: -1 });
  }
  return refs;
}

function parseTemplatedCallee(source) {
  const match = /^([A-Za-z_][A-Za-z0-9_]*)\s*</u.exec(source);
  if (match?.[1] === undefined) return undefined;
  const open = source.indexOf("<", match[0].length - 1);
  const close = findBalanced(source, open, "<", ">");
  if (close === undefined) return undefined;
  if (source.slice(skipWhitespace(source, close + 1)).trim().length > 0) return undefined;
  return {
    name: match[1],
    args: splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim()),
  };
}

function splitTopLevel(source) {
  const parts = [];
  let start = 0;
  let angle = 0;
  let paren = 0;
  let bracket = 0;
  let brace = 0;
  for (let index = 0; index < source.length; index++) {
    const char = source[index];
    if (char === "<") angle++;
    else if (char === ">") angle = Math.max(0, angle - 1);
    else if (char === "(") paren++;
    else if (char === ")") paren = Math.max(0, paren - 1);
    else if (char === "[") bracket++;
    else if (char === "]") bracket = Math.max(0, bracket - 1);
    else if (char === "{") brace++;
    else if (char === "}") brace = Math.max(0, brace - 1);
    else if (char === "," && angle === 0 && paren === 0 && bracket === 0 && brace === 0) {
      parts.push(source.slice(start, index).trim());
      start = index + 1;
    }
  }
  const last = source.slice(start).trim();
  if (last.length > 0) parts.push(last);
  return parts;
}

function balancedParenContents(source, open) {
  if (open < 0) return undefined;
  let depth = 0;
  for (let index = open; index < source.length; index++) {
    const char = source[index];
    if (char === "(") depth++;
    else if (char === ")") {
      depth--;
      if (depth === 0) {
        return source.slice(open + 1, index);
      }
    }
  }
  return undefined;
}

function findBalanced(source, open, openChar, closeChar) {
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

function skipWhitespace(source, index) {
  let cursor = index;
  while (cursor < source.length && /\s/u.test(source[cursor] ?? "")) cursor++;
  return cursor;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
