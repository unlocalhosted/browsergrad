const SEMANTIC_BUILTIN_DEVICE_HELPERS = new Set([
  "vec_at",
  "load128",
  "load128cs",
  "store128",
  "store128cs",
  "store128cg",
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
  const specializedKernel = specializeTemplateFromLaunchContext(kernel, templateArgumentsByKernelName, definesByName);
  const params = new Set(kernelParamNames(specializedKernel));
  const referencedDeviceFunctionsRaw = referencedDeviceFunctionClosure(specializedKernel, deviceFunctions, definesByName);
  const referencedSiblingKernelsRaw = siblingKernels.filter((sibling) => {
    const name = kernelDefinitionName(sibling);
    return name !== undefined && sourceLaunchesKernel(specializedKernel, name);
  });
  const templateNames = new Set([
    ...templateParameterNames(specializedKernel),
    ...referencedSiblingKernelsRaw.flatMap((sibling) => templateParameterNames(sibling)),
    ...referencedDeviceFunctionsRaw.flatMap((fn) => templateParameterNames(fn.source)),
  ]);
  const defines = [...definesByName]
    .filter(([name]) => !params.has(name) && !templateNames.has(name))
    .map(([name, value]) => `#define ${name} ${value}`);
  const referencedDeviceFunctions = referencedDeviceFunctionsRaw
    .map((fn) => ({
      ...fn,
      source: stripKernelLaunchTemplateArguments(specializeTemplateFromLaunchContext(fn.source, templateArgumentsByKernelName, definesByName)),
    }));
  const referencedSiblingKernels = referencedSiblingKernelsRaw.map((sibling) => stripKernelLaunchTemplateArguments(
    specializeTemplateFromLaunchContext(sibling, templateArgumentsByKernelName, definesByName),
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
    functionMacros.join("\n"),
    referencedDeviceFunctions.map((fn) => fn.source).join("\n"),
    constantDeclarations.join("\n"),
    textureDeclarations.join("\n"),
    referencedSiblingKernels.join("\n"),
    stripKernelLaunchTemplateArguments(specializedKernel),
  ].filter((part) => part.trim().length > 0).join("\n");
  return normalizeCppTemplateCarrierSyntax(normalizePacked128MemoryHelpers(unit));
}

export function kernelDefinitionName(kernel) {
  return kernelSignature(kernel)?.name;
}

export function collectKernelTemplateArguments(source) {
  const candidates = new Map();
  const addCandidate = (name, args) => {
    if (args.length === 0) return;
    const previous = candidates.get(name);
    if (previous === undefined || templateArgumentScore(args) > templateArgumentScore(previous)) {
      candidates.set(name, args);
    }
  };
  for (const launch of scanTemplatedKernelReferences(source)) {
    addCandidate(launch.name, launch.args);
  }
  for (const propagated of collectWrapperPropagatedTemplateArguments(source)) {
    addCandidate(propagated.name, propagated.args);
  }
  for (const inferred of collectLaunchInferredTemplateArguments(source)) {
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
      if (included.has(name) || !sourceMentionsIdentifierThroughDefines(source, name, definesByName)) continue;
      included.set(name, fn);
      pending.push(fn.source);
    }
  }
  return [...included.values()];
}

function sourceMentionsIdentifierThroughDefines(source, name, definesByName) {
  if (sourceMentionsIdentifier(source, name)) return true;
  const visited = new Set();
  const pending = [];
  for (const [defineName, value] of definesByName) {
    if (!sourceMentionsIdentifier(source, defineName)) continue;
    visited.add(defineName);
    pending.push(value);
  }
  while (pending.length > 0) {
    const value = pending.pop();
    if (sourceMentionsIdentifier(value, name)) return true;
    for (const [defineName, nextValue] of definesByName) {
      if (visited.has(defineName) || !sourceMentionsIdentifier(value, defineName)) continue;
      visited.add(defineName);
      pending.push(nextValue);
    }
  }
  return false;
}

function sourceLaunchesKernel(source, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\s*<<\\s*<`, "u").test(source) ||
    scanTemplatedKernelReferences(source).some((launch) => launch.name === name && launch.kind === "launch");
}

function sourceMentionsIdentifier(source, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\b`, "u").test(source);
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
    const value = normalizeTemplateTypeArgument(args[index] ?? templateParamDefaultValue(param), definesByName);
    if (value !== undefined) env.set(param.name, value);
  }
  return env;
}

function templateParamDefaultValue(param) {
  if (param.defaultStart === undefined || param.source === undefined) return undefined;
  return param.source.slice(param.defaultStart + 1).trim();
}

function normalizeTemplateTypeArgument(arg, definesByName = new Map()) {
  if (arg === undefined) return undefined;
  let type = arg.trim()
    .replace(/\b(?:const|volatile|typename|class|struct)\b/gu, " ")
    .replace(/[&*]/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  if (type.length === 0 || type.includes("<") || type.includes(">") || type.includes("(") || type.includes(")")) return undefined;
  const mapped = definesByName.get(type);
  if (mapped && /^[A-Za-z_][A-Za-z0-9_]*$/u.test(mapped)) type = mapped;
  if (type === "__half") return "half";
  if (type === "unsigned int" || type === "unsigned") return "uint";
  if (type === "unsigned char" || type === "uchar" || type === "uint8_t") return "uint";
  if (type === "signed int" || type === "signed") return "int";
  if (type === "signed char" || type === "char" || type === "int8_t") return "int";
  if (type === "clock_t") return "uint";
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
  type = defines.get(type) ?? type;
  if (type === "float") return "float4";
  if (type === "int") return "int4";
  if (type === "uint") return "uint4";
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
    const match = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)(?!\()\s+([A-Za-z_][A-Za-z0-9_]*)\s*$/u.exec(line);
    if (match?.[1] !== undefined && match[2] !== undefined) defines.set(match[1], match[2]);
  }
  return defines;
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

function scanTemplatedKernelReferences(source) {
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
  return refs;
}

function collectWrapperPropagatedTemplateArguments(source) {
  const propagated = [];
  const calls = scanTemplatedCallReferences(source);
  for (const fn of scanTemplatedFunctionDefinitions(source)) {
    const fnCalls = calls.filter((call) => call.name === fn.name && call.args.length > 0);
    if (fnCalls.length === 0) continue;
    const kernelRefs = scanTemplatedKernelReferences(fn.body);
    if (kernelRefs.length === 0) continue;
    for (const call of fnCalls) {
      const env = templateEnvironment(fn.templateParams, call.args);
      if (env.size === 0) continue;
      extendEnvironmentWithConstexprs(env, fn.body);
      for (const ref of kernelRefs) {
        const args = ref.args.map((arg) => substituteTemplateArgument(arg, env));
        if (templateArgumentScore(args) === 0) continue;
        propagated.push({ name: ref.name, args });
      }
    }
  }
  return propagated;
}

function collectLaunchInferredTemplateArguments(source) {
  const inferred = [];
  const kernels = new Map(scanTemplatedKernelDefinitions(source).map((kernel) => [kernel.name, kernel]));
  for (const launch of scanKernelLaunchCalls(source)) {
    const kernel = kernels.get(launch.name);
    if (kernel === undefined) continue;
    const args = inferKernelTemplateArgs(kernel, launch, source);
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

function inferKernelTemplateArgs(kernel, launch, source) {
  const env = new Map();
  const symbols = collectVisiblePointerTypes(source.slice(0, launch.start));
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

function collectVisiblePointerTypes(source) {
  const symbols = new Map();
  const re = /\b(?:const\s+)?([A-Za-z_][A-Za-z0-9_:]*)\s*(?:\*\s*)+\s*([A-Za-z_][A-Za-z0-9_]*)\b/gu;
  for (const match of source.matchAll(re)) {
    const [, type, name] = match;
    if (type && name) symbols.set(name, type);
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

function templateEnvironment(params, args) {
  const env = new Map();
  let argIndex = 0;
  for (const param of params) {
    const arg = args[argIndex++];
    if (!param || arg === undefined) continue;
    if (param.kind === "value") {
      const normalized = normalizeTemplateValueArgument(arg, param.valueType);
      if (normalized !== undefined) env.set(param.name, normalized);
    } else if (param.kind === "type" && /^[A-Za-z_][A-Za-z0-9_:]*$/u.test(arg.trim())) {
      env.set(param.name, arg.trim());
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

function substituteTemplateArgument(arg, env) {
  const value = evaluateTemplateIntegerExpression(arg, env);
  if (value !== undefined) return value;
  return arg.replace(/\b[A-Za-z_][A-Za-z0-9_]*\b/gu, (name) => env.get(name) ?? name).trim();
}

function evaluateTemplateIntegerExpression(expression, env) {
  const substituted = expression
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

function templateArgumentScore(args) {
  return args.reduce((score, arg) => score + (normalizeTemplateValueArgument(arg, "int") === undefined ? 0 : 1), 0);
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
