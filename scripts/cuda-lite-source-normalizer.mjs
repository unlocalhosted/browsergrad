const SEMANTIC_BUILTIN_DEVICE_HELPERS = new Set([
  "min",
  "max",
  "vec_at",
  "load128",
  "load128cs",
  "store128",
  "store128cs",
  "store128cg",
  "div_ceil",
  "blockReduce",
]);

const WIDE_PACKED128_TYPES = new Map([
  ["__bg_pack128_half8", { scalarType: "half", lanes: 8 }],
  ["__bg_pack128_bf168", { scalarType: "bf16", lanes: 8 }],
]);

const CUDA_BUILTIN_RECORD_DECLARATIONS = [
  "struct cudaExtent { size_t width; size_t height; size_t depth; };",
];

export function createKernelCompilationUnit({
  kernel,
  siblingKernels = [],
  definesByName = new Map(),
  templateArgumentsByKernelName = new Map(),
  functionDeclarations = [],
  deviceFunctions = [],
  constantDeclarations = [],
  deviceGlobalDeclarations = [],
  textureDeclarations = [],
  sharedDeclarations = [],
  recordDeclarations = [],
}) {
  const recordContext = [
    kernel,
    ...siblingKernels,
    ...functionDeclarations,
    ...deviceFunctions.map((fn) => fn.source),
    ...constantDeclarations,
    ...deviceGlobalDeclarations,
  ].join("\n");
  const builtinRecordDeclarations = CUDA_BUILTIN_RECORD_DECLARATIONS.filter((declaration) => {
    const name = recordDeclarationName(declaration);
    return name !== undefined && sourceMentionsIdentifier(recordContext, name);
  });
  const availableRecordDeclarations = [...builtinRecordDeclarations, ...recordDeclarations];
  const aliasContext = [
    kernel,
    ...siblingKernels,
    ...functionDeclarations,
    ...deviceFunctions.map((fn) => fn.source),
    ...constantDeclarations,
    ...deviceGlobalDeclarations,
    ...availableRecordDeclarations,
  ].join("\n");
  const aliasDefines = collectTypeAliasDefines(aliasContext, definesByName);
  const carrierDefines = collectCarrierMemberDefines(aliasContext, mergeDefineMaps(definesByName, aliasDefines));
  const syntheticPackDefines = collectSyntheticVectorPackDefines(aliasContext, mergeDefineMaps(definesByName, aliasDefines, carrierDefines));
  const effectiveDefines = mergeDefineMaps(definesByName, aliasDefines, carrierDefines, syntheticPackDefines);
  const specializedKernel = specializeTemplateFromLaunchContext(kernel, templateArgumentsByKernelName, effectiveDefines);
  const params = new Set(kernelParamNames(specializedKernel));
  const functionDefineBodies = collectFunctionDefineBodies(functionDeclarations);
  const referencedDeviceFunctionsRaw = referencedDeviceFunctionClosure(
    specializedKernel,
    deviceFunctions,
    mergeDefineMaps(effectiveDefines, functionDefineBodies),
  );
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
  const shadowedMacroNames = collectDeclaredIdentifiers([
    specializedKernel,
    ...referencedSiblingKernelsRaw,
    ...referencedDeviceFunctionsRaw.map((fn) => fn.source),
  ].join("\n"), effectiveDefines);
  const defines = [...effectiveDefines]
    .filter(([name, value]) => isMacroIdentifier(name) && !WIDE_PACKED128_TYPES.has(value) && !params.has(name) && !templateNames.has(name) && !shadowedMacroNames.has(name))
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
    ...constantDeclarations,
    ...deviceGlobalDeclarations,
  ].join("\n");
  const referencedRecordDeclarations = availableRecordDeclarations.filter((declaration) => {
    const name = recordDeclarationName(declaration);
    return name !== undefined &&
      sourceMentionsIdentifier(macroScope, name) &&
      (podRecordDeclarationVectorAlias(declaration, effectiveDefines) !== undefined ||
        scalarizedPodRecordDeclaration(declaration, effectiveDefines) !== undefined);
  });
  const functionMacros = functionDeclarations.filter((declaration) => {
    const name = functionDefineName(declaration);
    return name !== undefined && sourceMentionsIdentifier(macroScope, name);
  });
  const kernelWithSharedDeclarations = injectSharedDeclarationsIntoKernel(
    stripKnownTemplateCallArguments(stripKernelLaunchTemplateArguments(specializedKernel), referencedDeviceFunctionNames),
    sharedDeclarations,
  );
  const unit = normalizeLineContinuations([
    defines.join("\n"),
    functionMacros.map(normalizeFunctionMacro).join("\n"),
    referencedDeviceFunctions.map((fn) => fn.source).join("\n"),
    constantDeclarations.join("\n"),
    deviceGlobalDeclarations.join("\n"),
    textureDeclarations.join("\n"),
    referencedRecordDeclarations.join("\n"),
    referencedSiblingKernels.join("\n"),
    kernelWithSharedDeclarations,
  ].filter((part) => part.trim().length > 0).join("\n"));
  const withCarrierMembers = normalizeCarrierMemberReferences(unit, effectiveDefines);
  const withVectorCarrierAliases = normalizeCudaVectorCarrierAliases(withCarrierMembers, effectiveDefines);
  const withDeviceReferences = normalizeDeviceReferenceParams(withVectorCarrierAliases);
  const withScalarizedRecords = normalizeScalarizedPodRecords(withDeviceReferences, effectiveDefines);
  const withPodRecords = normalizePodRecordVectorAliases(withScalarizedRecords, effectiveDefines);
  const withNumericDefines = normalizeNumericObjectDefines(
    withPodRecords,
    effectiveDefines,
    new Set([...params, ...templateNames]),
  );
  const withCudaPipeline = normalizeCudaPipelineAsync(withNumericDefines);
  const withTypeDefines = normalizeSupportedTypeDefineReferences(
    withCudaPipeline,
    effectiveDefines,
    new Set([...params, ...templateNames]),
  );
  const withCooperativeGroupHelpers = normalizeCooperativeGroupHelperParams(withTypeDefines);
  const withVectorCooperativeReductions = normalizeVectorCooperativeReductions(withCooperativeGroupHelpers);
  const withStdMathAliases = normalizeStdMathAliases(withVectorCooperativeReductions);
  const withoutSupportedAliases = stripSupportedEnumDeclarations(stripSupportedTypeAliasDeclarations(withStdMathAliases, effectiveDefines));
  const withVectorConstructors = normalizeVectorStaticConstructors(withoutSupportedAliases, effectiveDefines);
  const withVectorLength = normalizeCudaVectorLength(withVectorConstructors);
  const withWidePacks = normalizeWidePacked128Aliases(withVectorLength, effectiveDefines);
  const withPackedHelpers = normalizePacked128MemoryHelpers(withWidePacks);
  const withSharedHelpers = normalizeSharedMemoryHelpers(withPackedHelpers);
  const withSincosHelpers = normalizeSincosHelpers(withSharedHelpers);
  const withBlockReduce = normalizeBlockReduceHelpers(withSincosHelpers);
  const withAtomicForwarders = normalizeAtomicForwarderHelpers(withBlockReduce);
  const withPointerStoreForwarders = normalizePointerStoreForwarderHelpers(withAtomicForwarders);
  const withLambdas = normalizeSimpleLocalLambdas(withPointerStoreForwarders);
  const withStatementMacros = normalizeSimpleStatementMacros(withLambdas);
  const withSideEffects = normalizeSideEffectExpressions(withStatementMacros);
  const withScopedForVariables = normalizeForLoopScopedVariables(withSideEffects);
  const withTemplateFallbacks = normalizeTemplateValueFallbacks(withScopedForVariables, effectiveDefines);
  return normalizeCppTemplateCarrierSyntax(withTemplateFallbacks);
}

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

function normalizeFunctionMacro(macro) {
  const cpAsync = semanticCpAsyncMacro(macro);
  return cpAsync ?? macro;
}

function injectSharedDeclarationsIntoKernel(kernel, sharedDeclarations) {
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
    addCandidate(launch.name, launch.args);
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
  if (!isTemplateSymbolArgument(previous) || !isTemplateSymbolArgument(next)) return false;
  return String(next).startsWith(`${previous}_`);
}

function isTemplateSymbolDowngrade(previous, next) {
  return isTemplateSymbolArgument(previous) &&
    isTemplateSymbolArgument(next) &&
    previous !== next &&
    String(previous).startsWith(`${next}_`);
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
  if (new RegExp(`(?:^|[^:A-Za-z0-9_])${escapeRegExp(name)}\\s*\\(`, "u").test(source)) return true;
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

function recordDeclarationName(declaration) {
  const match = /\b(?:typedef\s+)?struct(?:\s+(?:__align__|alignas)\s*\([^)]*\))*\s*([A-Za-z_][A-Za-z0-9_]*)?\s*\{[\s\S]*?\}\s*([A-Za-z_][A-Za-z0-9_]*)?\s*;/u.exec(declaration);
  return match?.[2] ?? match?.[1];
}

function podRecordDeclarationVectorAlias(declaration, definesByName) {
  const name = recordDeclarationName(declaration);
  if (name === undefined) return undefined;
  return collectPodRecordVectorAliases(declaration, definesByName)
    .find((record) => record.name === name)?.vectorType;
}

function scalarizedPodRecordDeclaration(declaration, definesByName) {
  const name = recordDeclarationName(declaration);
  if (name === undefined) return undefined;
  return collectScalarizedPodRecords(declaration, definesByName)
    .find((record) => record.name === name);
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
  if (args === undefined || args.length === 0) return rewriteFirstTemplateHeader(source, [], definesByName);
  return rewriteFirstTemplateHeader(source, args, definesByName);
}

function specializeDeviceFunctionFromCallContext(source, args, definesByName = new Map()) {
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
  const pointerParams = templateTypePointerParamNames(sourceTail, name);
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
    const match = new RegExp(`^${escaped}\\s*(?:\\*|&)\\s*([A-Za-z_][A-Za-z0-9_]*)\\b|^${escaped}\\s+[*&]\\s*([A-Za-z_][A-Za-z0-9_]*)\\b`, "u").exec(cleaned);
    return match?.[1] ?? match?.[2] ?? [];
  });
}

function canonicalTemplateScalarType(sourceTail, name, definesByName = new Map()) {
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

function canonicalTemplateValueArgument(sourceTail, name, definesByName = new Map()) {
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

function parseTemplateParam(param) {
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

function normalizeTemplateValueArgument(arg, valueType) {
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

function normalizeTemplateSymbolArgument(arg) {
  if (arg === undefined) return undefined;
  const value = arg.trim();
  return isTemplateSymbolArgument(value) ? value : undefined;
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

function templateValueEnvironment(params, args, definesByName = new Map()) {
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

function substituteTemplateValues(source, valueEnv) {
  if (valueEnv.size === 0) return source;
  return source.replace(/\b[A-Za-z_][A-Za-z0-9_]*\b/gu, (name) => valueEnv.get(name) ?? name);
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

function normalizeNumericObjectDefines(source, definesByName, blockedNames = new Set()) {
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
      out = out.replace(new RegExp(`\\b${escapeRegExp(name)}\\b`, "gu"), value);
    }
    for (const name of declared) localBlocked.add(name);
    return out;
  }).join("\n");
}

function normalizeSupportedTypeDefineReferences(source, definesByName, blockedNames = new Set()) {
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

function collectSyntheticVectorPackDefines(source, definesByName = new Map()) {
  const out = new Map();
  if (!sourceMentionsIdentifier(source, "x128") || definesByName.has("x128")) return out;
  const floatX = normalizeTemplateTypeArgument(definesByName.get("floatX") ?? "float", definesByName);
  if (floatX === "bf16") out.set("x128", "__bg_pack128_bf168");
  else if (floatX === "half") out.set("x128", "__bg_pack128_half8");
  else out.set("x128", "float4");
  return out;
}

function normalizeTemplateValueFallbacks(source, definesByName = new Map()) {
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

function resolveTemplateDefineValue(value, definesByName) {
  const trimmed = value.trim();
  return definesByName.get(trimmed) ?? trimmed;
}

function normalizeTemplateTypeArgument(arg, definesByName = new Map(), seen = new Set()) {
  if (arg === undefined) return undefined;
  let type = arg.trim()
    .replace(/\b(?:const|volatile|typename|class|struct)\b/gu, " ")
    .replace(/[&*]/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  if (type.length === 0 || type.includes("(") || type.includes(")")) return undefined;
  const packed = /^Packed128\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>$/u.exec(type)?.[1];
  if (packed !== undefined) {
    const elementType = normalizeTemplateTypeArgument(definesByName.get(packed) ?? packed, definesByName, new Set([...seen, packed]));
    if (elementType === "float") return "float4";
    if (elementType === "int") return "int4";
    if (elementType === "uint") return "uint4";
    if (elementType === "half") return "__bg_pack128_half8";
    if (elementType === "bf16") return "__bg_pack128_bf168";
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
  const supported = new Set(["float", "int", "uint", "half", "bf16", "bool", "float2", "float3", "float4", "half2", "int2", "int3", "int4", "uint2", "uint3", "uint4"]);
  return supported.has(type) ? type : undefined;
}

function normalizeCppTemplateCarrierSyntax(source) {
  const withBoolCarriers = source
    .replace(/\bstd\s*::\s*bool_constant\s*<\s*(true|false|[01])\s*>\s*\{\s*\}/gu, (_match, value) => value === "1" ? "true" : value === "0" ? "false" : value)
    .replace(/\bstd\s*::\s*bool_constant\s*<\s*([A-Za-z_][A-Za-z0-9_]*|[01])\s*>\s*(?=[,)])/gu, (_match, name) => `bool __bg_bool_constant_${name}`);
  return rewriteBoolTemplateCarriers(withBoolCarriers);
}

function normalizeCudaVectorLength(source) {
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
  for (const match of source.matchAll(/\btemplate\s*<([^>]*)>\s*__global__[\s\S]*?\([^)]*\bbool\s+__bg_bool_constant_([A-Za-z_][A-Za-z0-9_]*)\b[^)]*\)\s*\{/gu)) {
    const templateParams = match[1] ?? "";
    const name = match[2];
    if (!name || !new RegExp(`\\bbool\\s+${escapeRegExp(name)}\\b`, "u").test(templateParams)) continue;
    const start = match.index ?? 0;
    const brace = source.indexOf("{", start + match[0].length - 1);
    const end = findBalanced(source, brace, "{", "}");
    if (brace < 0 || end === undefined) continue;
    const body = source.slice(brace + 1, end);
    const rewritten = body.replace(new RegExp(`\\b${escapeRegExp(name)}\\b`, "gu"), `__bg_bool_constant_${name}`);
    edits.push({ start: brace + 1, end, value: rewritten });
  }
  let out = source;
  for (const edit of edits.reverse()) out = `${out.slice(0, edit.start)}${edit.value}${out.slice(edit.end)}`;
  return out;
}

function normalizeSimpleStatementMacros(source) {
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

function normalizeSideEffectExpressions(source) {
  return normalizePostfixPointerAssignments(normalizePrefixUpdateConditions(source));
}

function normalizeStdMathAliases(source) {
  return source
    .replace(/\bstd\s*::\s*isinf\s*\(/gu, "isinf(")
    .replace(/\bstd\s*::\s*numeric_limits\s*<\s*(?:float|double)\s*>\s*::\s*infinity\s*\(\s*\)/gu, "INFINITY");
}

function normalizeVectorCooperativeReductions(source) {
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
  const offset = `__bg_cg_reduce_offset_${counter}`;
  const shuffled = `make_${vectorType}(${fields.map((field) =>
    `__shfl_xor_sync(0xffffffff, ${target}.${field}, ${offset})`).join(", ")})`;
  return [
    `${vectorType} ${target} = ${value};`,
    `for (int ${offset} = 16; ${offset} > 0; ${offset} /= 2) {`,
    `  ${target} = ${op}(${target}, ${shuffled});`,
    "}",
  ].join(" ");
}

function normalizeCooperativeGroupHelperParams(source) {
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

function matchingOpenParen(source, close) {
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

function parameterName(param) {
  const cleaned = param.replace(/=[\s\S]*$/u, "").trim();
  return /(?:^|[\s*&])([A-Za-z_][A-Za-z0-9_]*)\s*$/u.exec(cleaned)?.[1];
}

function normalizeForLoopScopedVariables(source) {
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
    const scopedName = `__bg_for_${originalName}_${renameIndex++}`;
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

function replaceIdentifier(source, from, to) {
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

function normalizePrefixUpdateCondition(condition) {
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
    const args = splitTopLevel(line.slice(open + 1, close)).map((arg) => arg.trim());
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

function normalizeCudaVectorCarrierAliases(source, definesByName = new Map()) {
  return source.replace(/\b(?:typename\s+)?vec([234])\s*<\s*([A-Za-z_][A-Za-z0-9_:]*)\s*>\s*::\s*(?:Type|float)\b/gu, (match, lanes, rawType) => {
    const scalar = normalizeTemplateTypeArgument(definesByName.get(rawType) ?? rawType, definesByName);
    if (scalar === "float" || scalar === "double") return `float${lanes}`;
    if (scalar === "int") return `int${lanes}`;
    if (scalar === "uint") return `uint${lanes}`;
    return match;
  });
}

function normalizePodRecordVectorAliases(source, definesByName = new Map()) {
  const records = collectPodRecordVectorAliases(source, definesByName);
  if (records.length === 0) return source;
  let out = source;
  for (const record of records) out = out.replace(record.declaration, "");
  for (const record of records) out = rewritePodRecordConstructors(out, record);
  for (const record of records) out = rewritePodRecordMemberAccess(out, record);
  for (const record of records) {
    out = out.replace(new RegExp(`\\b${escapeRegExp(record.name)}\\b`, "gu"), record.vectorType);
  }
  return out;
}

function normalizeScalarizedPodRecords(source, definesByName = new Map()) {
  const records = collectScalarizedPodRecords(source, definesByName)
    .filter((record) => podRecordVectorType(record.fields) === undefined);
  if (records.length === 0) return source;
  let out = source;
  for (const record of records) out = out.replace(record.declaration, "");
  const symbolsByRecord = new Map(records.map((record) => [record.name, new Map()]));
  for (const record of records) out = rewriteScalarizedRecordConstants(out, record, symbolsByRecord.get(record.name));
  const functionExpansions = [];
  for (const record of records) out = rewriteScalarizedRecordFunctionSignatures(out, record, symbolsByRecord.get(record.name), functionExpansions);
  for (const record of records) out = rewriteScalarizedRecordLocalDeclarations(out, record, symbolsByRecord.get(record.name));
  for (const record of records) out = rewriteScalarizedRecordMemberAccess(out, record, symbolsByRecord.get(record.name));
  out = rewriteScalarizedRecordCalls(out, functionExpansions, symbolsByRecord);
  return out;
}

function collectScalarizedPodRecords(source, definesByName = new Map()) {
  const out = [];
  const re = /\b(?:typedef\s+)?struct(?:\s+(?:__align__|alignas)\s*\([^)]*\))*\s*([A-Za-z_][A-Za-z0-9_]*)?\s*\{([\s\S]*?)\}\s*([A-Za-z_][A-Za-z0-9_]*)?\s*;/gu;
  let match;
  while ((match = re.exec(source)) !== null) {
    const declaration = match[0];
    const tagName = match[1];
    const body = match[2] ?? "";
    const aliasName = match[3];
    const name = aliasName ?? tagName;
    if (!name || /\(|\)|\b(?:public|private|protected|operator|union)\b/u.test(body)) continue;
    const fields = parseScalarizedRecordFields(body, definesByName);
    if (fields === undefined || fields.length === 0) continue;
    out.push({ name, fields, declaration });
  }
  return out;
}

function parseScalarizedRecordFields(body, definesByName) {
  const fields = [];
  for (const raw of body.split(";")) {
    const line = stripLineComment(raw).trim();
    if (line.length === 0) continue;
    if (/[(){}*&]/u.test(line)) return undefined;
    const match = /^(?:const\s+|volatile\s+)*([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)?)\s+([A-Za-z_][A-Za-z0-9_]*)(.*)$/u.exec(line);
    if (match?.[1] === undefined || match[2] === undefined) return undefined;
    const valueType = normalizeTemplateTypeArgument(match[1], definesByName);
    if (!isScalarizedRecordFieldType(valueType)) return undefined;
    const dimensions = [];
    const tail = match[3] ?? "";
    const dimRe = /\[\s*([A-Za-z_][A-Za-z0-9_]*|[0-9]+)\s*\]/gu;
    let dimMatch;
    let consumed = "";
    while ((dimMatch = dimRe.exec(tail)) !== null) {
      const rawDim = dimMatch[1];
      const resolved = rawDim === undefined ? undefined : resolveScalarizedRecordArrayDimension(rawDim, definesByName);
      if (!Number.isInteger(resolved) || resolved <= 0) return undefined;
      dimensions.push(resolved);
      consumed += dimMatch[0];
    }
    if (tail.replace(consumed, "").trim().length > 0) return undefined;
    fields.push({ name: match[2], valueType, dimensions });
  }
  return fields;
}

function resolveScalarizedRecordArrayDimension(rawDim, definesByName) {
  const env = numericTemplateDefines(definesByName);
  const value = evaluateTemplateIntegerExpression(rawDim, env);
  if (value === undefined) return undefined;
  const resolved = Number(value);
  return Number.isInteger(resolved) ? resolved : undefined;
}

function isScalarizedRecordFieldType(valueType) {
  return valueType === "float" ||
    valueType === "int" ||
    valueType === "uint" ||
    valueType === "half" ||
    valueType === "bool" ||
    valueType === "float2" ||
    valueType === "float3" ||
    valueType === "float4" ||
    valueType === "int2" ||
    valueType === "int3" ||
    valueType === "int4" ||
    valueType === "uint2" ||
    valueType === "uint3" ||
    valueType === "uint4" ||
    valueType === "half2";
}

function rewriteScalarizedRecordConstants(source, record, symbols) {
  const re = new RegExp(`\\b__constant__\\s+${escapeRegExp(record.name)}\\s+([A-Za-z_][A-Za-z0-9_]*)((?:\\s*\\[[^\\]]+\\])*)\\s*;`, "gu");
  return source.replace(re, (_match, name, outerDimensions) => {
    if (typeof name !== "string") return _match;
    symbols.set(name, { kind: "constant" });
    if (record.fields.length === 1 && record.fields[0]?.dimensions.length) {
      const field = record.fields[0];
      symbols.set(name, { kind: "single-array-constant", field: field.name });
      return `__constant__ ${field.valueType} ${name}${outerDimensions ?? ""}${field.dimensions.map((dim) => `[${dim}]`).join("")};`;
    }
    return record.fields
      .map((field) => `__constant__ ${field.valueType} ${recordFieldName(name, field)}${outerDimensions ?? ""}${field.dimensions.map((dim) => `[${dim}]`).join("")};`)
      .join("\n");
  });
}

function rewriteScalarizedRecordFunctionSignatures(source, record, symbols, functionExpansions) {
  const re = /\b__(?:device|global)__[\s\S]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\(/gu;
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("(", match.index + match[0].length - 1);
    const close = findBalanced(source, open, "(", ")");
    if (close === undefined) {
      re.lastIndex = match.index + 1;
      continue;
    }
    const next = skipWhitespace(source, close + 1);
    if (source[next] !== "{") {
      re.lastIndex = close + 1;
      continue;
    }
    const rawParams = source.slice(open + 1, close);
    const params = splitTopLevel(rawParams).map((param) => param.trim()).filter(Boolean);
    const expansion = [];
    let changed = false;
    const rewritten = params.flatMap((param) => {
      const expanded = expandScalarizedRecordParam(param, record);
      if (expanded === undefined) {
        expansion.push({ kind: "plain" });
        return [param];
      }
      changed = true;
      symbols.set(expanded.name, { kind: expanded.pointer ? "param-pointer" : "param" });
      expansion.push({ kind: "record", recordName: record.name, record });
      return expanded.params;
    });
    if (!changed) {
      re.lastIndex = close + 1;
      continue;
    }
    const name = match[1];
    if (name !== undefined) functionExpansions.push({ name, params: expansion });
    out += source.slice(cursor, open + 1);
    out += rewritten.join(", ");
    cursor = close;
    re.lastIndex = close + 1;
  }
  out += source.slice(cursor);
  return out;
}

function expandScalarizedRecordParam(param, record) {
  const pointerParam = expandScalarizedRecordPointerParam(param, record);
  if (pointerParam !== undefined) return pointerParam;
  const re = new RegExp(`^\\s*((?:(?:const|volatile|__restrict__|__restrict|restrict|__grid_constant__)\\s+)*)${escapeRegExp(record.name)}\\s*(&)?\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*$`, "u");
  const match = re.exec(param);
  if (match === null || match[3] === undefined) return undefined;
  const qualifiers = (match[1] ?? "").replace(/\b(?:__restrict__|__restrict|restrict|__grid_constant__)\b/gu, "").replace(/\s+/gu, " ").trim();
  const byRef = match[2] !== undefined;
  if (byRef && !/\bconst\b/u.test(qualifiers)) return undefined;
  const name = match[3];
  const prefix = qualifiers.length > 0 ? `${qualifiers} ` : "";
  return {
    name,
    pointer: false,
    params: record.fields.map((field) => field.dimensions.length > 0
      ? `${prefix}${field.valueType} *${recordFieldName(name, field)}`
      : `${prefix}${field.valueType} ${recordFieldName(name, field)}`),
  };
}

function expandScalarizedRecordPointerParam(param, record) {
  if (record.fields.some((field) => field.dimensions.length > 0)) return undefined;
  const re = new RegExp(`^\\s*((?:(?:const|volatile|__restrict__|__restrict|restrict)\\s+)*)${escapeRegExp(record.name)}\\s*\\*\\s*((?:(?:const|volatile|__restrict__|__restrict|restrict)\\s+)*)?([A-Za-z_][A-Za-z0-9_]*)\\s*$`, "u");
  const match = re.exec(param);
  if (match === null || match[3] === undefined) return undefined;
  const qualifiers = `${match[1] ?? ""} ${match[2] ?? ""}`.replace(/\b(?:__restrict__|__restrict|restrict)\b/gu, "").replace(/\s+/gu, " ").trim();
  const name = match[3];
  const prefix = qualifiers.length > 0 ? `${qualifiers} ` : "";
  return {
    name,
    pointer: true,
    params: record.fields.map((field) => `${prefix}${field.valueType} *${recordFieldName(name, field)}`),
  };
}

function rewriteScalarizedRecordLocalDeclarations(source, record, symbols) {
  const re = new RegExp(`(^|[;{}]\\s*)${escapeRegExp(record.name)}\\s+([^;]+);`, "gmu");
  return source.replace(re, (match, prefix, declarators) => {
    if (typeof declarators !== "string" || /[=*&(){}]/u.test(declarators)) return match;
    const names = splitTopLevel(declarators).map((item) => item.trim()).filter(Boolean);
    if (names.length === 0) return match;
    const expanded = [];
    for (const rawName of names) {
      const parsed = /^([A-Za-z_][A-Za-z0-9_]*)((?:\s*\[[^\]]+\])*)$/u.exec(rawName);
      if (parsed?.[1] === undefined) return match;
      const name = parsed[1];
      const outerDimensions = parsed[2] ?? "";
      symbols.set(name, { kind: "local" });
      for (const field of record.fields) {
        expanded.push(`${field.valueType} ${recordFieldName(name, field)}${outerDimensions}${field.dimensions.map((dim) => `[${dim}]`).join("")};`);
      }
    }
    return `${prefix}${expanded.join(" ")}`;
  });
}

function rewriteScalarizedRecordMemberAccess(source, record, symbols) {
  let out = source;
  for (const [name, symbol] of symbols) {
    for (const field of record.fields) {
      const re = new RegExp(`\\b${escapeRegExp(name)}\\b((?:\\s*\\[[^\\]]+\\])*)\\s*\\.\\s*${escapeRegExp(field.name)}\\b`, "gu");
      out = out.replace(re, (_match, indexes) => {
        if (symbol.kind === "single-array-constant" && record.fields.length === 1 && field.dimensions.length > 0) {
          return `${name}${indexes ?? ""}`;
        }
        return `${recordFieldName(name, field)}${indexes ?? ""}`;
      });
    }
  }
  return out;
}

function rewriteScalarizedRecordCalls(source, functionExpansions, symbolsByRecord) {
  if (functionExpansions.length === 0) return source;
  let out = source;
  for (const expansion of functionExpansions) out = rewriteScalarizedRecordCallsForFunction(out, expansion, symbolsByRecord);
  return out;
}

function rewriteScalarizedRecordCallsForFunction(source, expansion, symbolsByRecord) {
  const re = new RegExp(`\\b${escapeRegExp(expansion.name)}\\s*\\(`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("(", match.index);
    const close = findBalanced(source, open, "(", ")");
    if (close === undefined) {
      re.lastIndex = match.index + expansion.name.length;
      continue;
    }
    const after = skipWhitespace(source, close + 1);
    if (source[after] === "{") {
      re.lastIndex = close + 1;
      continue;
    }
    const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim());
    const rewritten = [];
    let argIndex = 0;
    let changed = false;
    for (const param of expansion.params) {
      const arg = args[argIndex++] ?? "";
      if (param.kind !== "record") {
        rewritten.push(arg);
        continue;
      }
      const recordSymbols = symbolsByRecord.get(param.recordName);
      const parts = recordSymbols === undefined ? undefined : scalarizedRecordArgumentParts(arg, param.record, recordSymbols);
      if (parts === undefined) {
        rewritten.push(arg);
        continue;
      }
      changed = true;
      rewritten.push(...parts);
    }
    while (argIndex < args.length) rewritten.push(args[argIndex++] ?? "");
    if (!changed) {
      re.lastIndex = close + 1;
      continue;
    }
    out += source.slice(cursor, open + 1);
    out += rewritten.join(", ");
    cursor = close;
    re.lastIndex = close + 1;
  }
  out += source.slice(cursor);
  return out;
}

function scalarizedRecordArgumentParts(arg, record, symbols) {
  const name = arg.trim();
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/u.test(name) || !symbols.has(name)) return undefined;
  const symbol = symbols.get(name);
  return record.fields.map((field) => {
    if (symbol.kind === "single-array-constant" && record.fields.length === 1 && field.dimensions.length > 0) return name;
    return recordFieldName(name, field);
  });
}

function recordFieldName(name, field) {
  return `${name}__${field.name}`;
}

function collectPodRecordVectorAliases(source, definesByName = new Map()) {
  const out = [];
  const re = /\b(?:typedef\s+)?struct(?:\s+(?:__align__|alignas)\s*\([^)]*\))*\s*([A-Za-z_][A-Za-z0-9_]*)?\s*\{([\s\S]*?)\}\s*([A-Za-z_][A-Za-z0-9_]*)?\s*;/gu;
  let match;
  while ((match = re.exec(source)) !== null) {
    const declaration = match[0];
    const tagName = match[1];
    const body = match[2] ?? "";
    const aliasName = match[3];
    const name = aliasName ?? tagName;
    if (!name || /\(|\)|\b(?:public|private|protected|operator)\b/u.test(body)) continue;
    const fields = parsePodRecordFields(body, definesByName);
    if (fields === undefined) continue;
    const vectorType = podRecordVectorType(fields);
    if (vectorType === undefined) continue;
    out.push({ name, fields, vectorType, declaration });
  }
  return out;
}

function parsePodRecordFields(body, definesByName) {
  const fields = [];
  for (const raw of body.split(";")) {
    const line = raw.replace(/\/\/[^\n\r]*/gu, "").trim();
    if (line.length === 0) continue;
    if (/[(){}]/u.test(line)) return undefined;
    const match = /^(?:const\s+|volatile\s+)*([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)?)\s+([A-Za-z_][A-Za-z0-9_]*)$/u.exec(line);
    if (match?.[1] === undefined || match[2] === undefined) return undefined;
    const valueType = normalizeTemplateTypeArgument(match[1], definesByName);
    if (!isPodRecordScalarType(valueType)) return undefined;
    fields.push({ name: match[2], valueType });
  }
  return fields.length >= 2 && fields.length <= 4 ? fields : undefined;
}

function podRecordVectorType(fields) {
  if (fields.length < 2 || fields.length > 4 || fields.some((field) => field.dimensions?.length > 0)) return undefined;
  const first = fields[0]?.valueType;
  if (!first || !fields.every((field) => field.valueType === first)) return undefined;
  if (first === "float") return `float${fields.length}`;
  if (first === "int") return `int${fields.length}`;
  if (first === "uint") return `uint${fields.length}`;
  if (first === "half" && fields.length === 2) return "half2";
  return undefined;
}

function isPodRecordScalarType(valueType) {
  return valueType === "float" || valueType === "int" || valueType === "uint" || valueType === "half";
}

function rewritePodRecordConstructors(source, record) {
  const re = new RegExp(`\\b${escapeRegExp(record.name)}\\s*\\{`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("{", match.index);
    const close = findBalanced(source, open, "{", "}");
    if (close === undefined) {
      re.lastIndex = match.index + record.name.length;
      continue;
    }
    const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim()).filter(Boolean);
    out += source.slice(cursor, match.index);
    out += `make_${record.vectorType}(${args.join(", ")})`;
    cursor = close + 1;
    re.lastIndex = close + 1;
  }
  out += source.slice(cursor);
  return out;
}

function rewritePodRecordMemberAccess(source, record) {
  const variables = collectPodRecordVariableNames(source, record.name);
  if (variables.size === 0) return source;
  let out = source;
  for (const [fieldIndex, field] of record.fields.entries()) {
    const vectorField = ["x", "y", "z", "w"][fieldIndex];
    if (!field?.name || !vectorField) continue;
    for (const variable of variables) {
      const re = new RegExp(`\\b${escapeRegExp(variable)}\\b((?:\\s*\\[[^\\]]+\\])*)\\s*\\.\\s*${escapeRegExp(field.name)}\\b`, "gu");
      out = out.replace(re, (_match, indexes) => `${variable}${indexes}.${vectorField}`);
    }
  }
  return out;
}

function collectPodRecordVariableNames(source, recordName) {
  const names = new Set();
  const declarationRe = new RegExp(`(?:^|[^A-Za-z0-9_])(?:const\\s+)?${escapeRegExp(recordName)}\\s*(?:&\\s*|\\*\\s*)?([A-Za-z_][A-Za-z0-9_]*)`, "gu");
  let match;
  while ((match = declarationRe.exec(source)) !== null) {
    if (match[1] !== undefined && !["struct", "return"].includes(match[1])) names.add(match[1]);
  }
  return names;
}

function normalizeLineContinuations(source) {
  return source.replace(/\\\r?\n\s*/gu, " ");
}

function normalizeSimpleLocalLambdas(source) {
  let out = source;
  let cursor = 0;
  while (cursor < out.length) {
    const match = /\bauto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[\s*&?\s*\]\s*/gu.exec(out.slice(cursor));
    if (match === null) break;
    const declStart = cursor + match.index;
    const name = match[1];
    if (name === undefined) break;
    const paramsOpen = skipWhitespace(out, declStart + match[0].length);
    const paramsClose = findBalanced(out, paramsOpen, "(", ")");
    if (paramsClose === undefined) {
      cursor = declStart + 1;
      continue;
    }
    const bodyOpen = skipWhitespace(out, paramsClose + 1);
    const bodyClose = findBalanced(out, bodyOpen, "{", "}");
    const semi = bodyClose === undefined ? undefined : skipWhitespace(out, bodyClose + 1);
    if (bodyClose === undefined || out[semi] !== ";") {
      cursor = declStart + 1;
      continue;
    }
    const params = splitTopLevel(out.slice(paramsOpen + 1, paramsClose)).map((param) => param.trim()).filter(Boolean);
    const body = out.slice(bodyOpen + 1, bodyClose).trim();
    if (params.length === 0 || params.some((param) => lambdaParamName(param) === undefined) || /\breturn\b/u.test(body)) {
      cursor = semi + 1;
      continue;
    }
    const before = out.slice(0, declStart);
    const after = out.slice(semi + 1);
    out = before + inlineSimpleLambdaCalls(after, name, params, body);
    cursor = before.length;
  }
  return out;
}

function inlineSimpleLambdaCalls(source, name, params, body) {
  const re = new RegExp(`\\b${escapeRegExp(name)}\\s*\\(`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("(", match.index + name.length);
    const close = findBalanced(source, open, "(", ")");
    if (close === undefined) {
      re.lastIndex = match.index + name.length;
      continue;
    }
    const semi = skipWhitespace(source, close + 1);
    if (source[semi] !== ";") {
      re.lastIndex = close + 1;
      continue;
    }
    const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim());
    if (args.length !== params.length) {
      re.lastIndex = close + 1;
      continue;
    }
    out += source.slice(cursor, match.index);
    out += `{ ${params.map((param, index) => `${param} = ${args[index] ?? "0"}`).join("; ")}; ${body} }`;
    cursor = semi + 1;
    re.lastIndex = semi + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function lambdaParamName(param) {
  return /([A-Za-z_][A-Za-z0-9_]*)\s*$/.exec(param.replace(/=[\s\S]*$/u, "").trim())?.[1];
}

function collectDeclaredIdentifiers(source, definesByName = new Map()) {
  const clean = stripCommentsAndStrings(source);
  const out = new Set();
  const re = /(?:^|[;{}\n(])\s*(?:const\s+|constexpr\s+|static\s+|volatile\s+|__shared__\s+|extern\s+)*(?:unsigned\s+|signed\s+|long\s+|short\s+)?([A-Za-z_][A-Za-z0-9_:]*)\s*(?:[*&]\s*)?([A-Za-z_][A-Za-z0-9_]*)\b/gu;
  for (const match of clean.matchAll(re)) {
    const [, rawType, name] = match;
    if (!rawType || !name) continue;
    if (normalizeTemplateTypeArgument(definesByName.get(rawType) ?? rawType, definesByName) === undefined) continue;
    out.add(name);
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

function normalizeDeviceReferenceParams(source) {
  const functions = scanDeviceReferenceFunctions(source);
  if (functions.length === 0) return source;
  const withReferenceCalls = rewriteDeviceReferenceCalls(source, functions);
  return rewriteDeviceReferenceDefinitions(withReferenceCalls);
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

function normalizeBlockReduceHelpers(source) {
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
}

function normalizeAtomicForwarderHelpers(source) {
  const helpers = collectAtomicForwarderHelpers(source);
  if (helpers.size === 0) return source;
  let out = source;
  for (const [name, builtin] of helpers) {
    out = out.replace(builtin.definition, "");
    out = rewriteSimpleCallName(out, name, builtin.target);
  }
  return out;
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

function normalizePointerStoreForwarderHelpers(source) {
  const helpers = collectPointerStoreForwarderHelpers(source);
  if (helpers.size === 0) return source;
  let out = source;
  for (const [name, helper] of helpers) {
    out = out.replace(helper.definition, "");
    out = rewritePointerStoreForwarderCalls(out, name, helper);
  }
  return out;
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

function addressTarget(expression) {
  if (expression === undefined) return undefined;
  const trimmed = expression.trim();
  if (!trimmed.startsWith("&")) return undefined;
  const target = trimmed.slice(1).trim();
  if (/^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\[[^\]]+\])?$/u.test(target)) return target;
  return undefined;
}

function normalizeWidePacked128Aliases(source, definesByName = new Map()) {
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

function replaceBalancedIndexStatement(source, re, replacer) {
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

function replaceBalancedCall(source, re, replacer) {
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

function normalizeCudaPipelineAsync(source) {
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
    /\bauto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*cg::this_thread_block\s*\(\s*\)\s*;/gu,
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

function cudaTypeByteSize(type) {
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

function normalizePacked128MemoryHelpers(source) {
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

function normalizeVectorStaticConstructors(source, definesByName = new Map()) {
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

function vectorLaneCount(vectorType) {
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
  for (const [name, value] of collectEnumIntegerConstants(source, defines)) defines.set(name, value);
  for (const [name, value] of collectCarrierMemberDefines(source, defines)) defines.set(name, value);
  return defines;
}

function collectEnumIntegerConstants(source, initialDefines = new Map()) {
  const constants = new Map();
  const enumRe = /\benum(?:\s+(?:class\s+)?[A-Za-z_][A-Za-z0-9_]*)?\s*\{([\s\S]*?)\}\s*;/gu;
  let match;
  while ((match = enumRe.exec(source)) !== null) {
    const body = match[1] ?? "";
    let nextValue = 0;
    const env = mergeDefineMaps(initialDefines, constants);
    for (const rawEntry of splitTopLevel(body)) {
      const entry = stripLineComment(rawEntry).trim();
      if (entry.length === 0) continue;
      const parsed = /^([A-Za-z_][A-Za-z0-9_]*)(?:\s*=\s*(.+))?$/u.exec(entry);
      if (parsed?.[1] === undefined) continue;
      const name = parsed[1];
      const explicit = parsed[2]?.trim();
      const value = explicit === undefined
        ? String(nextValue)
        : evaluateTemplateIntegerExpression(explicit, env);
      if (value === undefined) continue;
      constants.set(name, value);
      env.set(name, value);
      nextValue = Number(value) + 1;
    }
  }
  return constants;
}

function collectTypeAliasDefines(source, initialDefines = new Map()) {
  const defines = new Map(initialDefines);
  const out = new Map();
  for (const line of source.split(/\r?\n/u)) {
    const alias = parseSimpleTypeAlias(stripLineComment(line), defines);
    if (alias !== undefined) {
      defines.set(alias.name, alias.value);
      out.set(alias.name, alias.value);
    }
  }
  return out;
}

function collectFunctionDefineBodies(functionDeclarations) {
  const defines = new Map();
  for (const declaration of functionDeclarations) {
    const match = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s+([\s\S]+)$/u.exec(declaration);
    if (match?.[1] !== undefined && match[2] !== undefined) defines.set(match[1], match[2].trim());
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
  for (const match of body.matchAll(/\bstatic\s+constexpr\s+(?:const\s+)?(?:int|uint|unsigned\s+int|size_t|ptrdiff_t|bool)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);/gu)) {
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
  const arrayTypedefMatch = /^\s*typedef\s+(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([0-9]+)\s*\]\s*;\s*$/u.exec(line);
  if (arrayTypedefMatch !== null) {
    const [, sourceType, alias, rawLength] = arrayTypedefMatch;
    const value = normalizeArrayAliasType(sourceType, rawLength, defines);
    return value && alias ? { name: alias, value } : undefined;
  }
  const functionPointerTypedefMatch = /^\s*typedef\s+[\s\S]+?\(\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\([^;]*\)\s*;\s*$/u.exec(line);
  if (functionPointerTypedefMatch !== null) {
    const alias = functionPointerTypedefMatch[1];
    return alias ? { name: alias, value: "uint" } : undefined;
  }
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

function normalizeArrayAliasType(sourceType, rawLength, defines) {
  const type = normalizeTemplateTypeArgument(sourceType, defines);
  const length = Number(rawLength);
  if (!Number.isInteger(length) || length < 2 || length > 4) return undefined;
  if (type === "float") return `float${length}`;
  if (type === "int") return `int${length}`;
  if (type === "uint") return `uint${length}`;
  if (type === "half" && length === 2) return "half2";
  return undefined;
}

function stripSupportedTypeAliasDeclarations(source, initialDefines = new Map()) {
  const defines = new Map(initialDefines);
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

function stripSupportedEnumDeclarations(source) {
  return source.replace(/\benum(?:\s+(?:class\s+)?[A-Za-z_][A-Za-z0-9_]*)?\s*\{[\s\S]*?\}\s*;/gu, "");
}

function parseSimpleIntegerConstant(line) {
  const match = /^\s*((?:(?:static|constexpr|const)\s+)*)(?:int|uint|unsigned\s+int|size_t|ptrdiff_t)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9A-Fa-fxXuUlL\s()+\-*/%<>&|^?:.]+)\s*;\s*$/u.exec(line);
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
      extendEnvironmentWithConstexprs(env, fn.body);
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

function collectSymbolWrapperTemplateArguments(source, definesByName = new Map()) {
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

function collectSimpleSymbolWrapperTemplateArguments(source, definesByName = new Map()) {
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

function scanFunctionCallReferences(source) {
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

function inferTemplatedFunctionCallArgs(fn, call, source, definesByName = new Map()) {
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
  const symbols = collectVisiblePointerTypes(source.slice(0, launch.start), definesByName);
  return inferKernelTemplateArgsFromSymbols(kernel, launch, symbols, definesByName);
}

function inferKernelTemplateArgsFromSymbols(kernel, launch, symbols, definesByName = new Map()) {
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

function substituteVisiblePointerTypes(symbols, env, definesByName = new Map()) {
  const out = new Map();
  for (const [name, type] of symbols) {
    out.set(name, substituteTemplateArgument(type, env, definesByName));
  }
  return out;
}

function parseKernelSignatureParams(signature) {
  const open = signature.indexOf("(");
  const params = balancedParenContents(signature, open);
  if (params === undefined || params.trim().length === 0) return [];
  return splitTopLevel(params).map(parseKernelParamForInference);
}

function parseFunctionSignatureParams(signature, templateParams = []) {
  const open = signature.indexOf("(");
  const params = balancedParenContents(signature, open);
  if (params === undefined || params.trim().length === 0) return [];
  const templateTypeNames = new Set(templateParams.filter((param) => param.kind === "type").map((param) => param.name));
  return splitTopLevel(params).map((param) => parseFunctionParamForInference(param, templateTypeNames));
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

function parseFunctionParamForInference(param, templateTypeNames) {
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

function collectVisiblePointerTypes(source, initialDefines = new Map()) {
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

function collectVisibleValueTypes(source, initialDefines = new Map()) {
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

function inferArgumentPointerType(arg, symbols) {
  const cast = /^\(\s*(?:const\s+)?([A-Za-z_][A-Za-z0-9_:]*)\s*\*\s*\)/u.exec(arg)?.[1];
  if (cast !== undefined) return cast;
  const name = /^[A-Za-z_][A-Za-z0-9_]*/u.exec(arg.trim())?.[0];
  return name === undefined ? undefined : symbols.get(name);
}

function inferArgumentValueType(arg, symbols, definesByName = new Map()) {
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
      start: nameStart,
    });
    index = afterTemplate + 1;
  }
  return refs;
}

function collectDeviceFunctionTemplateArguments(sources, names, definesByName = new Map()) {
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
      const previous = out.get(call.name);
      if (previous === undefined || templateArgumentScore(args) > templateArgumentScore(previous)) {
        out.set(call.name, args);
      }
    }
    for (const call of scanFunctionCallReferences(source)) {
      if (!names.has(call.name) || out.has(call.name)) continue;
      const fn = templatedFunctions.get(call.name);
      if (fn === undefined) continue;
      const args = inferTemplatedFunctionCallArgs(fn, call, source, definesByName).map((arg) => arg ?? "");
      if (templateArgumentScore(args) === 0) continue;
      out.set(call.name, args);
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
    } else if (param.kind === "symbol") {
      const normalized = normalizeTemplateSymbolArgument(resolveTemplateDefineValue(arg, definesByName));
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
  const result = evaluateIntegerExpression(normalized);
  return result === undefined ? undefined : String(result);
}

const INTEGER_EXPRESSION_PRECEDENCE = new Map([
  ["||", 1],
  ["&&", 2],
  ["|", 3],
  ["^", 4],
  ["&", 5],
  ["==", 6],
  ["!=", 6],
  ["<", 7],
  ["<=", 7],
  [">", 7],
  [">=", 7],
  ["<<", 8],
  [">>", 8],
  ["+", 9],
  ["-", 9],
  ["*", 10],
  ["/", 10],
  ["%", 10],
]);

function evaluateIntegerExpression(expression) {
  const tokens = tokenizeIntegerExpression(expression);
  if (tokens === undefined) return undefined;
  const parser = {
    index: 0,
    peek: () => tokens[parser.index],
    advance: () => tokens[parser.index++],
  };
  const value = parseIntegerExpression(parser, 0);
  return value !== undefined && parser.index === tokens.length && Number.isFinite(value)
    ? Math.trunc(value)
    : undefined;
}

function tokenizeIntegerExpression(expression) {
  const tokens = [];
  let index = 0;
  const punctuators = ["||", "&&", "==", "!=", "<=", ">=", "<<", ">>", "+", "-", "*", "/", "%", "<", ">", "&", "|", "^", "!", "~", "?", ":", "(", ")"];
  while (index < expression.length) {
    const char = expression[index];
    if (/\s/u.test(char ?? "")) {
      index++;
      continue;
    }
    const number = /^(?:0[xX][0-9A-Fa-f]+|[0-9]+)/u.exec(expression.slice(index))?.[0];
    if (number !== undefined) {
      tokens.push({ kind: "number", value: Number(number) });
      index += number.length;
      continue;
    }
    const op = punctuators.find((item) => expression.startsWith(item, index));
    if (op === undefined) return undefined;
    tokens.push({ kind: "op", value: op });
    index += op.length;
  }
  return tokens;
}

function parseIntegerExpression(parser, minPrecedence) {
  let left = parseIntegerPrefix(parser);
  if (left === undefined) return undefined;
  while (true) {
    const token = parser.peek();
    if (token?.kind !== "op") break;
    if (token.value === "?" && minPrecedence <= 0) {
      parser.advance();
      const consequent = parseIntegerExpression(parser, 0);
      if (consequent === undefined || parser.advance()?.value !== ":") return undefined;
      const alternate = parseIntegerExpression(parser, 0);
      if (alternate === undefined) return undefined;
      left = left !== 0 ? consequent : alternate;
      continue;
    }
    const precedence = INTEGER_EXPRESSION_PRECEDENCE.get(token.value);
    if (precedence === undefined || precedence < minPrecedence) break;
    parser.advance();
    const right = parseIntegerExpression(parser, precedence + 1);
    if (right === undefined) return undefined;
    left = applyIntegerBinaryOperator(token.value, left, right);
    if (left === undefined) return undefined;
  }
  return left;
}

function parseIntegerPrefix(parser) {
  const token = parser.advance();
  if (token === undefined) return undefined;
  if (token.kind === "number") return token.value;
  if (token.kind !== "op") return undefined;
  if (token.value === "(") {
    const value = parseIntegerExpression(parser, 0);
    return value !== undefined && parser.advance()?.value === ")" ? value : undefined;
  }
  if (token.value === "+" || token.value === "-" || token.value === "!" || token.value === "~") {
    const value = parseIntegerPrefix(parser);
    if (value === undefined) return undefined;
    if (token.value === "+") return value;
    if (token.value === "-") return -value;
    if (token.value === "!") return value === 0 ? 1 : 0;
    return ~Math.trunc(value);
  }
  return undefined;
}

function applyIntegerBinaryOperator(operator, left, right) {
  switch (operator) {
    case "||": return left !== 0 || right !== 0 ? 1 : 0;
    case "&&": return left !== 0 && right !== 0 ? 1 : 0;
    case "|": return Math.trunc(left) | Math.trunc(right);
    case "^": return Math.trunc(left) ^ Math.trunc(right);
    case "&": return Math.trunc(left) & Math.trunc(right);
    case "==": return left === right ? 1 : 0;
    case "!=": return left !== right ? 1 : 0;
    case "<": return left < right ? 1 : 0;
    case "<=": return left <= right ? 1 : 0;
    case ">": return left > right ? 1 : 0;
    case ">=": return left >= right ? 1 : 0;
    case "<<": return Math.trunc(left) << Math.trunc(right);
    case ">>": return Math.trunc(left) >> Math.trunc(right);
    case "+": return left + right;
    case "-": return left - right;
    case "*": return left * right;
    case "/": return right === 0 ? undefined : Math.trunc(left / right);
    case "%": return right === 0 ? undefined : Math.trunc(left) % Math.trunc(right);
    default: return undefined;
  }
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
    case "ptrdiff_t":
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
    if (arg === undefined) return score;
    if (normalizeTemplateTypeArgument(arg) !== undefined) return score + 2;
    if (normalizeTemplateValueArgument(arg, "int") !== undefined) return score + 1;
    if (isTemplateSymbolArgument(arg)) return score + 1;
    return score;
  }, 0);
}

function isTemplateSymbolArgument(arg) {
  return /^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$/u.test(String(arg).trim());
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

function splitTopLevel(source, separator = ",") {
  const parts = [];
  let start = 0;
  let angle = 0;
  let paren = 0;
  let bracket = 0;
  let brace = 0;
  const trackAngles = separator !== ";";
  for (let index = 0; index < source.length; index++) {
    const char = source[index];
    if (trackAngles && char === "<") angle++;
    else if (trackAngles && char === ">") angle = Math.max(0, angle - 1);
    else if (char === "(") paren++;
    else if (char === ")") paren = Math.max(0, paren - 1);
    else if (char === "[") bracket++;
    else if (char === "]") bracket = Math.max(0, bracket - 1);
    else if (char === "{") brace++;
    else if (char === "}") brace = Math.max(0, brace - 1);
    else if (char === separator && angle === 0 && paren === 0 && bracket === 0 && brace === 0) {
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
