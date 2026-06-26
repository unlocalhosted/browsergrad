import { requireNormalizerHelpers } from "./cuda-lite-source-normalizer-context.mjs";

const escapeRegExp = (...args) => requireNormalizerHelpers().escapeRegExp(...args);
const findBalanced = (...args) => requireNormalizerHelpers().findBalanced(...args);
const functionDefineName = (...args) => requireNormalizerHelpers().functionDefineName(...args);
const isMacroIdentifier = (...args) => requireNormalizerHelpers().isMacroIdentifier(...args);
const normalizeTemplateTypeArgument = (...args) => requireNormalizerHelpers().normalizeTemplateTypeArgument(...args);
const parseCudaGlobalFunction = (...args) => requireNormalizerHelpers().parseCudaGlobalFunction(...args);
const skipWhitespace = (...args) => requireNormalizerHelpers().skipWhitespace(...args);
const splitTopLevel = (...args) => requireNormalizerHelpers().splitTopLevel(...args);
const stripCommentsAndStrings = (...args) => requireNormalizerHelpers().stripCommentsAndStrings(...args);

export function normalizeLineContinuations(source) {
  return source.replace(/\\\r?\n\s*/gu, " ");
}

export function normalizeEscapedNewlinesOutsideStrings(source) {
  let out = "";
  let inString = false;
  let inChar = false;
  let escaped = false;
  for (let index = 0; index < source.length; index++) {
    const char = source[index];
    const next = source[index + 1];
    if (inString || inChar) {
      out += char;
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (inString && char === "\"") {
        inString = false;
      } else if (inChar && char === "'") {
        inChar = false;
      }
      continue;
    }
    if (char === "\"") {
      inString = true;
      out += char;
      continue;
    }
    if (char === "'") {
      inChar = true;
      out += char;
      continue;
    }
    if (char === "\\" && next === "n") {
      out += "\n";
      index++;
      continue;
    }
    out += char;
  }
  return out;
}

export function normalizeSimpleLocalLambdas(source) {
  let out = source;
  let cursor = 0;
  while (cursor < out.length) {
    const match = /\bauto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[[^\]\n]*\]\s*/gu.exec(out.slice(cursor));
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
    const actualBodyOpen = out[bodyOpen] === "{"
      ? bodyOpen
      : skipWhitespace(out, out.indexOf("{", bodyOpen));
    if (actualBodyOpen < bodyOpen) {
      cursor = declStart + 1;
      continue;
    }
    const bodyClose = findBalanced(out, actualBodyOpen, "{", "}");
    const semi = bodyClose === undefined ? undefined : skipWhitespace(out, bodyClose + 1);
    if (bodyClose === undefined || out[semi] !== ";") {
      cursor = declStart + 1;
      continue;
    }
    const params = splitTopLevel(out.slice(paramsOpen + 1, paramsClose)).map((param) => param.trim()).filter(Boolean);
    const body = out.slice(actualBodyOpen + 1, bodyClose).trim();
    if (params.length === 0 || params.some((param) => lambdaParamName(param) === undefined)) {
      cursor = semi + 1;
      continue;
    }
    const before = out.slice(0, declStart);
    const after = out.slice(semi + 1);
    const returnExpression = simpleReturnLambdaExpression(out.slice(paramsClose + 1, actualBodyOpen).trim(), body, params);
    out = before + (
      returnExpression === undefined
        ? inlineSimpleLambdaCalls(after, name, params, body)
        : inlineSimpleExpressionLambdaCalls(after, name, params, returnExpression)
    );
    cursor = before.length;
  }
  return out;
}

export function normalizeSimpleExpressionMacros(source) {
  const macros = collectSimpleExpressionMacros(source);
  if (macros.length === 0) return source;
  let out = source;
  for (let pass = 0; pass < 8; pass++) {
    const next = out
      .split(/\r?\n/u)
      .map((line) => /^\s*#/u.test(line) ? line : replaceSimpleExpressionMacroCalls(line, macros))
      .join("\n");
    if (next === out) break;
    out = next;
  }
  return out
    .split(/\r?\n/u)
    .filter((line) => !macros.some((macro) => functionDefineName(line) === macro.name))
    .join("\n");
}

export function normalizeDependentCarrierParams(source) {
  if (!/\b(?:typename\s+)?[A-Za-z_][A-Za-z0-9_:]*\s*::\s*Arguments\b/u.test(source)) return source;
  let out = "";
  let cursor = 0;
  const globalRe = /\b__global__\b/gu;
  let match;
  while ((match = globalRe.exec(source)) !== null) {
    const globalStart = match.index;
    if (globalStart < cursor) continue;
    const fn = parseCudaGlobalFunction(source, globalStart);
    if (fn === undefined) {
      globalRe.lastIndex = globalStart + "__global__".length;
      continue;
    }
    const replacement = lowerDependentCarrierParamFunction(fn);
    if (replacement === undefined) {
      globalRe.lastIndex = fn.bodyEnd + 1;
      continue;
    }
    out += source.slice(cursor, fn.headerStart);
    out += replacement;
    cursor = fn.bodyEnd + 1;
    globalRe.lastIndex = fn.bodyEnd + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function lowerDependentCarrierParamFunction(fn) {
  let body = fn.body;
  const newParams = [];
  let changed = false;
  for (const param of fn.params) {
    const match = /^(?:(?:const|volatile)\s+)*(?:typename\s+)?[A-Za-z_][A-Za-z0-9_:]*\s*::\s*Arguments(?:\s+const|\s*&)*\s+([A-Za-z_][A-Za-z0-9_]*)$/u.exec(param.trim());
    const carrierName = match?.[1];
    if (carrierName === undefined) {
      newParams.push(param);
      continue;
    }
    const lowered = lowerDependentCarrierParam(body, carrierName);
    if (lowered === undefined) {
      newParams.push(param);
      continue;
    }
    body = lowered.body;
    newParams.push(...lowered.params);
    changed = true;
  }
  if (!changed) return undefined;
  return [
    `${fn.signaturePrefix}(${newParams.join(", ")}) {`,
    body,
    "}",
  ].join("\n");
}

function lowerDependentCarrierParam(body, carrierName) {
  const pointerFields = collectDependentCarrierPointerFields(body, carrierName);
  const shapeFields = collectDependentCarrierShapeFields(body, carrierName);
  if (pointerFields.size === 0 && shapeFields.size === 0) return undefined;
  const params = [];
  for (const [field, type] of pointerFields) params.push(`${type} *${carrierName}__${field}`);
  for (const shape of shapeFields) {
    for (const index of shape.indexes) params.push(`int ${carrierName}__${shape.field}${index}`);
  }
  let out = body;
  for (const shape of shapeFields) {
    const selectRe = new RegExp(String.raw`\bselect\s*<\s*([^<>]+)\s*>\s*\(\s*${escapeRegExp(carrierName)}\s*\.\s*${escapeRegExp(shape.field)}\s*\)`, "gu");
    out = out.replace(selectRe, (_match, rawIndexes) => {
      const indexes = splitTopLevel(String(rawIndexes)).map((item) => item.trim()).filter(Boolean);
      if (indexes.length === 0 || indexes.some((index) => !/^[0-9]+$/u.test(index))) return _match;
      return `make_shape(${indexes.map((index) => `${carrierName}__${shape.field}${index}`).join(", ")})`;
    });
    const getRe = new RegExp(String.raw`\bget\s*<\s*([0-9]+)\s*>\s*\(\s*${escapeRegExp(carrierName)}\s*\.\s*${escapeRegExp(shape.field)}\s*\)`, "gu");
    out = out.replace(getRe, (_match, index) => `${carrierName}__${shape.field}${index}`);
  }
  for (const field of pointerFields.keys()) {
    const fieldRe = new RegExp(String.raw`\b${escapeRegExp(carrierName)}\s*\.\s*${escapeRegExp(field)}\b`, "gu");
    out = out.replace(fieldRe, `${carrierName}__${field}`);
  }
  return { body: out, params };
}

function collectDependentCarrierPointerFields(body, carrierName) {
  const fields = new Map();
  const re = new RegExp(String.raw`\bmake_gmem_ptr(?:\s*<\s*([^<>]+)\s*>)?\s*\(\s*${escapeRegExp(carrierName)}\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)`, "gu");
  for (const match of body.matchAll(re)) {
    const rawType = match[1] ?? "float";
    const field = match[2];
    const type = normalizeTemplateTypeArgument(rawType) ?? rawType.trim();
    if (field !== undefined && !fields.has(field)) fields.set(field, type);
  }
  return fields;
}

function collectDependentCarrierShapeFields(body, carrierName) {
  const byField = new Map();
  const record = (field, indexes) => {
    const seen = byField.get(field) ?? new Set();
    for (const index of indexes) {
      if (/^[0-9]+$/u.test(index)) seen.add(index);
    }
    if (seen.size > 0) byField.set(field, seen);
  };
  const selectRe = new RegExp(String.raw`\bselect\s*<\s*([^<>]+)\s*>\s*\(\s*${escapeRegExp(carrierName)}\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)`, "gu");
  for (const match of body.matchAll(selectRe)) {
    if (match[1] === undefined || match[2] === undefined) continue;
    record(match[2], splitTopLevel(match[1]).map((item) => item.trim()));
  }
  const getRe = new RegExp(String.raw`\bget\s*<\s*([0-9]+)\s*>\s*\(\s*${escapeRegExp(carrierName)}\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)`, "gu");
  for (const match of body.matchAll(getRe)) {
    if (match[1] === undefined || match[2] === undefined) continue;
    record(match[2], [match[1]]);
  }
  return [...byField].map(([field, indexes]) => ({ field, indexes: [...indexes].sort((a, b) => Number(a) - Number(b)) }));
}

export function normalizeMissingSemicolonAfterMacroAssignment(source) {
  return source.replace(
    /(=\s*[A-Z_][A-Z0-9_]*\s*\([^;\n{}]+\))\s+(?=(?:const\s+|volatile\s+)*(?:float|half|bf16|int|uint|bool|double|float[234]|int[234]|uint[234]|half2)\s+[A-Za-z_][A-Za-z0-9_]*\b)/gu,
    "$1; ",
  );
}

function collectSimpleExpressionMacros(source) {
  const byName = new Map();
  for (const line of source.split(/\r?\n/u)) {
    const macro = parseSimpleExpressionMacro(line);
    if (macro !== undefined) byName.set(macro.name, macro);
  }
  return [...byName.values()];
}

function parseSimpleExpressionMacro(line) {
  const match = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s+([\s\S]+?)\s*$/u.exec(line);
  if (match?.[1] === undefined || match[2] === undefined || match[3] === undefined) return undefined;
  const name = match[1];
  const params = splitTopLevel(match[2]).map((param) => param.trim()).filter(Boolean);
  const body = match[3].trim();
  if (params.length === 0 || params.some((param) => !isMacroIdentifier(param))) return undefined;
  if (!isSimpleExpressionMacroBody(name, body)) return undefined;
  return { name, params, body, declaration: line };
}

function isSimpleExpressionMacroBody(name, body) {
  if (body.includes(";") || /(?:^|[^\w])asm\s+volatile\b/u.test(body)) return false;
  if (/[{}]/u.test(body)) return false;
  if (/\b(?:reinterpret|static|const|dynamic)_cast\s*</u.test(body)) return false;
  if (new RegExp(`\\b${escapeRegExp(name)}\\s*\\(`, "u").test(body)) return false;
  return /^[A-Za-z0-9_.$()[\]\s,+\-*/%<>=!&|^~?:]+$/u.test(body);
}

function replaceSimpleExpressionMacroCalls(line, macros) {
  let out = line;
  for (const macro of macros) out = replaceSimpleExpressionMacroCall(out, macro);
  return out;
}

function replaceSimpleExpressionMacroCall(line, macro) {
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
    out += line.slice(cursor, match.index);
    out += expandSimpleExpressionMacro(macro, args);
    cursor = close + 1;
    re.lastIndex = close + 1;
  }
  return cursor === 0 ? line : out + line.slice(cursor);
}

function expandSimpleExpressionMacro(macro, args) {
  let body = macro.body;
  for (const [index, param] of macro.params.entries()) {
    const arg = args[index] ?? "";
    body = replaceMacroParamReference(body, param, `(${arg})`);
  }
  return body;
}

function replaceMacroParamReference(source, param, replacement) {
  const re = new RegExp(`\\b${escapeRegExp(param)}\\b`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const before = source[match.index - 1];
    if (before === "." || before === ":") {
      re.lastIndex = match.index + param.length;
      continue;
    }
    out += source.slice(cursor, match.index);
    out += replacement;
    cursor = match.index + param.length;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

export function splitTopLevelMacroArgs(source) {
  return splitTopLevel(source, ",", false);
}

export function normalizeLegacyCudaArithmeticMacros(source) {
  const specs = [
    { name: "FMUL", arity: 2, expand: (args) => `((${args[0] ?? "0"}) * (${args[1] ?? "0"}))` },
    { name: "IMUL", arity: 2, expand: (args) => `((${args[0] ?? "0"}) * (${args[1] ?? "0"}))` },
    { name: "IMAD", arity: 3, expand: (args) => `(((${args[0] ?? "0"}) * (${args[1] ?? "0"})) + (${args[2] ?? "0"}))` },
  ];
  let out = source;
  for (const spec of specs) out = replaceLegacyArithmeticMacroCalls(out, spec);
  return out;
}

function replaceLegacyArithmeticMacroCalls(source, spec) {
  const re = new RegExp(`\\b${spec.name}\\s*\\(`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("(", match.index + spec.name.length);
    const close = findBalanced(source, open, "(", ")");
    if (open < 0 || close === undefined) {
      re.lastIndex = match.index + spec.name.length;
      continue;
    }
    const args = splitTopLevelMacroArgs(source.slice(open + 1, close)).map((arg) => arg.trim());
    if (args.length !== spec.arity) {
      re.lastIndex = close + 1;
      continue;
    }
    out += source.slice(cursor, match.index);
    out += spec.expand(args);
    cursor = close + 1;
    re.lastIndex = close + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
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

function simpleReturnLambdaExpression(trailingReturnType, body, params) {
  if (!/\breturn\b/u.test(body)) return undefined;
  if (params.length !== 1) return undefined;
  const paramName = lambdaParamName(params[0]);
  if (paramName === undefined) return undefined;
  const switchExpression = simpleSwitchReturnExpression(body, paramName, trailingReturnType);
  if (switchExpression !== undefined) return switchExpression;
  const direct = /^\s*return\s+([^;{}]+)\s*;\s*$/u.exec(body)?.[1];
  return direct === undefined ? undefined : { params: [paramName], expression: direct.trim() };
}

function simpleSwitchReturnExpression(body, paramName, trailingReturnType) {
  const switchRe = new RegExp(`^\\s*switch\\s*\\(\\s*${escapeRegExp(paramName)}\\s*\\)\\s*\\{([\\s\\S]*?)\\}\\s*return\\s+([^;]+)\\s*;\\s*$`, "u");
  const match = switchRe.exec(body);
  if (match?.[1] === undefined || match[2] === undefined) return undefined;
  const cases = [];
  const caseRe = /\bcase\s+([0-9]+)\s*:\s*return\s+([^;]+)\s*;/gu;
  for (const item of match[1].matchAll(caseRe)) {
    if (item[1] === undefined || item[2] === undefined) continue;
    cases.push({ value: item[1], expression: item[2].trim() });
  }
  if (cases.length === 0) return undefined;
  const fallback = emptyCudaReturnExpression(match[2].trim(), trailingReturnType);
  const expression = cases
    .reverse()
    .reduce((acc, item) => `((${paramName}) == ${item.value} ? ${item.expression} : ${acc})`, fallback);
  return { params: [paramName], expression };
}

function emptyCudaReturnExpression(rawFallback, trailingReturnType) {
  if (rawFallback !== "{}") return rawFallback;
  const type = /->\s*(?:const\s+)?([A-Za-z_][A-Za-z0-9_]*)/u.exec(trailingReturnType)?.[1];
  if (type === "float4") return "make_float4(0.0f, 0.0f, 0.0f, 0.0f)";
  if (type === "float3") return "make_float3(0.0f, 0.0f, 0.0f)";
  if (type === "float2") return "make_float2(0.0f, 0.0f)";
  if (type === "uint4") return "make_uint4(0u, 0u, 0u, 0u)";
  if (type === "uint3") return "make_uint3(0u, 0u, 0u)";
  if (type === "uint2") return "make_uint2(0u, 0u)";
  if (type === "int4") return "make_int4(0, 0, 0, 0)";
  if (type === "int3") return "make_int3(0, 0, 0)";
  if (type === "int2") return "make_int2(0, 0)";
  return "0";
}

function inlineSimpleExpressionLambdaCalls(source, name, params, lambda) {
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
    const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim());
    if (args.length !== params.length) {
      re.lastIndex = close + 1;
      continue;
    }
    out += source.slice(cursor, match.index);
    let expression = lambda.expression;
    for (const [index, param] of lambda.params.entries()) {
      const arg = args[index] ?? "0";
      expression = expression.replace(new RegExp(`\\b${escapeRegExp(param)}\\b`, "gu"), `(${arg})`);
    }
    out += `(${expression})`;
    cursor = close + 1;
    re.lastIndex = close + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

export function normalizeVectorCooperativeShuffles(source) {
  if (!/\.shfl(?:_up|_down|_xor)?\s*\(/u.test(source)) return source;
  const vectorSymbols = collectCudaVectorValueTypes(source);
  if (vectorSymbols.size === 0) return source;
  const re = /\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(shfl(?:_up|_down|_xor)?)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([^;)]+)\)\s*;/gu;
  return source.replace(re, (match, target, group, method, value, offset) => {
    if (typeof target !== "string" || typeof group !== "string" || typeof method !== "string" || typeof value !== "string" || typeof offset !== "string") return match;
    const targetType = vectorSymbols.get(target);
    const valueType = vectorSymbols.get(value);
    if (targetType === undefined || valueType === undefined || targetType !== valueType) return match;
    const lanes = cudaVectorLanes(targetType);
    if (lanes.length === 0) return match;
    return `${target} = make_${targetType}(${lanes.map((lane) => `${group}.${method}(${value}.${lane}, ${offset.trim()})`).join(", ")});`;
  });
}

export function collectCudaVectorValueTypes(source) {
  const symbols = new Map();
  const declarationRe = /\b(float[234]|half2|int[234]|uint[234])\s+([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()/gu;
  let match;
  while ((match = declarationRe.exec(source)) !== null) {
    const type = match[1];
    const name = match[2];
    if (type !== undefined && name !== undefined) symbols.set(name, type);
  }
  return symbols;
}

export function cudaVectorLanes(type) {
  if (type === "half2" || type === "float2" || type === "int2" || type === "uint2") return ["x", "y"];
  if (type === "float3" || type === "int3" || type === "uint3") return ["x", "y", "z"];
  if (type === "float4" || type === "int4" || type === "uint4") return ["x", "y", "z", "w"];
  return [];
}

function lambdaParamName(param) {
  return /([A-Za-z_][A-Za-z0-9_]*)\s*$/.exec(param.replace(/=[\s\S]*$/u, "").trim())?.[1];
}

export function collectDeclaredIdentifiers(source, definesByName = new Map()) {
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

export function sourceUsesMemberName(source, name) {
  return new RegExp(`[.:]\\s*${escapeRegExp(name)}\\b`, "u").test(stripCommentsAndStrings(source));
}

export function normalizeSharedMemoryHelpers(source) {
  return source
    .replace(
      /\bSharedMemory\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\2\s*\.\s*getPointer\s*\(\s*\)\s*;/gu,
      (_match, templateType, _helperName, pointerType, pointerName) =>
        templateType === pointerType ? `extern __shared__ ${templateType} ${pointerName}[];` : _match,
    )
    .replace(
      /\bSharedMemory\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s+([A-Za-z_][A-Za-z0-9_]*)\s*;/gu,
      "extern __shared__ $1 $2[];",
    )
    .replace(
      /\b([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*SharedMemory\s*<\s*\1\s*>\s*\(\s*\)\s*;/gu,
      "extern __shared__ $1 $2[];",
    );
}

export function addressTarget(expression) {
  if (expression === undefined) return undefined;
  const trimmed = expression.trim();
  if (!trimmed.startsWith("&")) return undefined;
  const target = trimmed.slice(1).trim();
  if (/^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\[[^\]]+\])?$/u.test(target)) return target;
  return undefined;
}
