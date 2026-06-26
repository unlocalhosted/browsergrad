import { requireNormalizerHelpers } from "./cuda-lite-source-normalizer-context.mjs";

const collectPodRecordVectorAliases = (...args) => requireNormalizerHelpers().collectPodRecordVectorAliases(...args);
const cudaVectorLanes = (...args) => requireNormalizerHelpers().cudaVectorLanes(...args);
const escapeRegExp = (...args) => requireNormalizerHelpers().escapeRegExp(...args);
const evaluateTemplateIntegerExpression = (...args) => requireNormalizerHelpers().evaluateTemplateIntegerExpression(...args);
const findBalanced = (...args) => requireNormalizerHelpers().findBalanced(...args);
const findCudaLaunchClose = (...args) => requireNormalizerHelpers().findCudaLaunchClose(...args);
const mergeDefineMaps = (...args) => requireNormalizerHelpers().mergeDefineMaps(...args);
const normalizeTemplateTypeArgument = (...args) => requireNormalizerHelpers().normalizeTemplateTypeArgument(...args);
const numericTemplateDefines = (...args) => requireNormalizerHelpers().numericTemplateDefines(...args);
const parseTemplateParam = (...args) => requireNormalizerHelpers().parseTemplateParam(...args);
const podRecordVectorType = (...args) => requireNormalizerHelpers().podRecordVectorType(...args);
const rewritePodRecordConstructors = (...args) => requireNormalizerHelpers().rewritePodRecordConstructors(...args);
const rewritePodRecordMemberAccess = (...args) => requireNormalizerHelpers().rewritePodRecordMemberAccess(...args);
const skipWhitespace = (...args) => requireNormalizerHelpers().skipWhitespace(...args);
const splitTopLevel = (...args) => requireNormalizerHelpers().splitTopLevel(...args);
const stripLineComment = (...args) => requireNormalizerHelpers().stripLineComment(...args);
const templateTypeEnvironment = (...args) => requireNormalizerHelpers().templateTypeEnvironment(...args);

const SEMANTIC_RECORD_CARRIERS = new Set([
  "DevicePool",
]);

export function normalizeCarrierMemberReferences(source, definesByName) {
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

export function normalizeCudaVectorCarrierAliases(source, definesByName = new Map()) {
  return source.replace(/\b(?:typename\s+)?vec([234])\s*<\s*([A-Za-z_][A-Za-z0-9_:]*)\s*>\s*::\s*(Type|float|double|int|uint|half)\b/gu, (match, lanes, rawType, member) => {
    const scalar = member === "Type"
      ? normalizeTemplateTypeArgument(definesByName.get(rawType) ?? rawType, definesByName)
      : member;
    if (scalar === "float" || scalar === "double") return `float${lanes}`;
    if (scalar === "int") return `int${lanes}`;
    if (scalar === "uint") return `uint${lanes}`;
    if (scalar === "half" && lanes === "2") return "half2";
    return match;
  });
}

export function normalizePodRecordVectorAliases(source, definesByName = new Map()) {
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

export function normalizeScalarizedPodRecords(source, definesByName = new Map()) {
  const records = collectScalarizedPodRecords(source, definesByName)
    .filter((record) => podRecordVectorType(record.fields) === undefined);
  if (records.length === 0) return source;
  let out = source;
  for (const record of records) out = out.replace(record.declaration, "");
  for (const record of records) out = normalizeScalarizedRecordTypeUses(out, record);
  for (const record of records) out = normalizeScalarizedRecordEnumConstants(out, record);
  const symbolsByRecord = new Map(records.map((record) => [record.name, new Map()]));
  for (const record of records) out = rewriteScalarizedRecordConstants(out, record, symbolsByRecord.get(record.name));
  const functionExpansions = [];
  for (const record of records) out = rewriteScalarizedRecordFunctionSignatures(out, record, symbolsByRecord.get(record.name), functionExpansions);
  for (const record of records) out = rewriteScalarizedRecordReturnCallDeclarations(out, record, symbolsByRecord.get(record.name), functionExpansions);
  for (const record of records) out = rewriteScalarizedRecordLocalDeclarations(out, record, symbolsByRecord.get(record.name));
  for (const record of records) out = rewriteScalarizedRecordMemcpys(out, record, symbolsByRecord.get(record.name));
  for (const record of records) out = rewriteScalarizedRecordAssignments(out, record, symbolsByRecord.get(record.name));
  for (const record of records) out = rewriteScalarizedRecordMemberAccess(out, record, symbolsByRecord.get(record.name));
  out = rewriteScalarizedRecordCalls(out, functionExpansions, symbolsByRecord);
  return out;
}

function normalizeScalarizedRecordTypeUses(source, record) {
  if (!record.templated) return source;
  const typeRe = new RegExp(`\\b${escapeRegExp(record.name)}\\s*<`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = typeRe.exec(source)) !== null) {
    const open = source.indexOf("<", match.index + record.name.length);
    const close = findBalanced(source, open, "<", ">");
    if (open < 0 || close === undefined) {
      typeRe.lastIndex = match.index + record.name.length;
      continue;
    }
    out += source.slice(cursor, match.index);
    out += record.name;
    cursor = close + 1;
    typeRe.lastIndex = close + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function normalizeScalarizedRecordEnumConstants(source, record) {
  if (record.enumConstants.size === 0) return source;
  let out = source;
  for (const [name, value] of record.enumConstants) {
    const [owner, member] = name.split("::");
    if (!owner || !member) continue;
    out = out.replace(new RegExp(`\\b${escapeRegExp(owner)}\\s*::\\s*${escapeRegExp(member)}\\b`, "gu"), value);
  }
  return out;
}

export function collectScalarizedPodRecords(source, definesByName = new Map()) {
  const out = [];
  for (const recordDeclaration of scanStructRecordDeclarations(source)) {
    const { aliasName, body, declaration, rawTemplateParams, tagName } = recordDeclaration;
    const name = aliasName ?? tagName;
    if (!name || /\(|\)|\b(?:public|private|protected|operator|union)\b/u.test(stripRecordEnums(body).body)) continue;
    if (SEMANTIC_RECORD_CARRIERS.has(name)) continue;
    const templateParams = rawTemplateParams === undefined
      ? []
      : splitTopLevel(rawTemplateParams).map(parseTemplateParam).filter(Boolean);
    const env = templateParams.length === 0
      ? new Map()
      : (collectTemplateRecordTypeEnvironments(source, name, templateParams, definesByName)[0] ?? new Map());
    const recordDefines = mergeDefineMaps(definesByName, env, collectRecordEnumTypeDefines(body), collectRecordEnumIntegerConstants(name, body, definesByName));
    const fields = parseScalarizedRecordFields(stripRecordEnums(body).body, recordDefines);
    if (fields === undefined || fields.length === 0) continue;
    out.push({
      name,
      fields,
      declaration,
      templated: templateParams.length > 0,
      enumConstants: collectRecordEnumIntegerConstants(name, body, recordDefines),
    });
  }
  return out;
}

export function scanStructRecordDeclarations(source) {
  const out = [];
  const re = /\bstruct\b/gu;
  let match;
  while ((match = re.exec(source)) !== null) {
    const structStart = match.index;
    const headerStart = structDeclarationStart(source, structStart);
    const brace = source.indexOf("{", structStart + match[0].length);
    const semicolon = source.indexOf(";", structStart + match[0].length);
    if (brace < 0 || (semicolon >= 0 && semicolon < brace)) {
      re.lastIndex = structStart + match[0].length;
      continue;
    }
    const end = findBalanced(source, brace, "{", "}");
    if (end === undefined) {
      re.lastIndex = structStart + match[0].length;
      continue;
    }
    const semi = source.indexOf(";", end + 1);
    if (semi < 0) {
      re.lastIndex = end + 1;
      continue;
    }
    const header = source.slice(headerStart, brace);
    const rawTemplateParams = /^\s*template\s*<([^<>]*)>/u.exec(header)?.[1];
    const tagName = /\bstruct(?:\s+(?:__align__|alignas)\s*\([^)]*\))*\s*([A-Za-z_][A-Za-z0-9_]*)?\s*$/u.exec(header)?.[1];
    const aliasName = /^\s*([A-Za-z_][A-Za-z0-9_]*)?\s*$/u.exec(source.slice(end + 1, semi))?.[1];
    out.push({
      aliasName,
      body: source.slice(brace + 1, end),
      declaration: source.slice(headerStart, semi + 1),
      rawTemplateParams,
      tagName,
    });
    re.lastIndex = semi + 1;
  }
  return out;
}

function structDeclarationStart(source, structStart) {
  const before = source.slice(0, structStart);
  const template = /\btemplate\s*<[^<>]*>\s*$/u.exec(before);
  if (template?.index !== undefined) return template.index;
  const typedef = /\btypedef\s+$/u.exec(before);
  if (typedef?.index !== undefined) return typedef.index;
  return structStart;
}

function parseScalarizedRecordFields(body, definesByName) {
  const fields = [];
  for (const raw of body.split(";")) {
    const line = stripLineComment(raw).trim();
    if (line.length === 0) continue;
    if (/[(){}&]/u.test(line)) return undefined;
    const pointerMatch = /^(?:const\s+|volatile\s+)*([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)?)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)(.*)$/u.exec(line);
    const scalarMatch = pointerMatch === null
      ? /^(?:const\s+|volatile\s+)*([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)?)\s+([A-Za-z_][A-Za-z0-9_]*)(.*)$/u.exec(line)
      : null;
    const match = pointerMatch ?? scalarMatch;
    if (match?.[1] === undefined || match[2] === undefined) return undefined;
    const valueType = normalizeTemplateTypeArgument(match[1], definesByName);
    if (!isScalarizedRecordFieldType(valueType)) return undefined;
    const pointer = pointerMatch !== null;
    const tail = match[3] ?? "";
    const declarators = pointer || !/^\s*,/u.test(tail)
      ? [`${match[2]}${tail}`]
      : [match[2], ...splitTopLevel(tail.replace(/^\s*,/u, "")).map((item) => item.trim()).filter(Boolean)];
    for (const declarator of declarators) {
      const parsed = /^([A-Za-z_][A-Za-z0-9_]*)([\s\S]*)$/u.exec(declarator.trim());
      if (parsed?.[1] === undefined) return undefined;
      const dimensions = [];
      const dimTail = parsed[2] ?? "";
      const dimRe = /\[\s*([A-Za-z_][A-Za-z0-9_]*|[0-9]+)\s*\]/gu;
      let dimMatch;
      let consumed = "";
      while ((dimMatch = dimRe.exec(dimTail)) !== null) {
        const rawDim = dimMatch[1];
        const resolved = rawDim === undefined ? undefined : resolveScalarizedRecordArrayDimension(rawDim, definesByName);
        if (!Number.isInteger(resolved) || resolved <= 0) return undefined;
        dimensions.push(resolved);
        consumed += dimMatch[0];
      }
      if (dimTail.replace(consumed, "").trim().length > 0) return undefined;
      fields.push({ name: parsed[1], valueType, dimensions, pointer });
    }
  }
  return fields;
}

export function stripRecordEnums(body) {
  const enumTypes = new Set();
  const stripped = body.replace(/\benum\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{[\s\S]*?\}\s*;?/gu, (_match, name) => {
    if (typeof name === "string") enumTypes.add(name);
    return "";
  });
  return { body: stripped, enumTypes };
}

export function collectRecordEnumTypeDefines(body) {
  const defines = new Map();
  for (const name of stripRecordEnums(body).enumTypes) defines.set(name, "int");
  return defines;
}

export function collectRecordEnumIntegerConstants(recordName, body, initialDefines = new Map()) {
  const constants = new Map();
  const enumRe = /\benum\s+(?:[A-Za-z_][A-Za-z0-9_]*)?\s*\{([\s\S]*?)\}\s*;?/gu;
  let match;
  while ((match = enumRe.exec(body)) !== null) {
    let nextValue = 0;
    const env = mergeDefineMaps(initialDefines, constants);
    for (const rawEntry of splitTopLevel(match[1] ?? "")) {
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
      constants.set(`${recordName}::${name}`, value);
      env.set(name, value);
      nextValue = Number(value) + 1;
    }
  }
  return constants;
}

function collectTemplateRecordTypeEnvironments(source, recordName, templateParams, definesByName = new Map()) {
  const envs = [];
  const re = new RegExp(`\\b${escapeRegExp(recordName)}\\s*<`, "gu");
  let match;
  while ((match = re.exec(source)) !== null) {
    const open = source.indexOf("<", match.index + recordName.length);
    const close = findBalanced(source, open, "<", ">");
    if (open < 0 || close === undefined) {
      re.lastIndex = match.index + recordName.length;
      continue;
    }
    const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim());
    const env = templateTypeEnvironment(templateParams, args, definesByName);
    if (env.size > 0) envs.push(env);
    re.lastIndex = close + 1;
  }
  return envs;
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
    valueType === "double" ||
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
    const prefix = source.slice(match.index, open);
    const returnRecord = scalarizedRecordReturnFunction(prefix, match[1], record);
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
    if (returnRecord !== undefined) {
      changed = true;
      rewritten.push(...scalarizedRecordReturnParams(returnRecord.name, record));
    }
    if (!changed) {
      re.lastIndex = close + 1;
      continue;
    }
    const name = match[1];
    if (name !== undefined) functionExpansions.push({
      name,
      params: expansion,
      ...(returnRecord === undefined ? {} : { returnRecordName: record.name, returnRecord: record }),
    });
    out += source.slice(cursor, match.index);
    out += returnRecord === undefined ? source.slice(match.index, open + 1) : scalarizedRecordReturnPrefix(prefix, name, record);
    out += rewritten.join(", ");
    const bodyOpen = next;
    const bodyEnd = findBalanced(source, bodyOpen, "{", "}");
    if (returnRecord !== undefined && bodyEnd !== undefined) {
      out += source.slice(close, bodyOpen + 1);
      out += rewriteScalarizedRecordReturns(source.slice(bodyOpen + 1, bodyEnd), returnRecord.name, record);
      cursor = bodyEnd;
      re.lastIndex = bodyEnd + 1;
      continue;
    }
    cursor = close;
    re.lastIndex = close + 1;
  }
  out += source.slice(cursor);
  return out;
}

function scalarizedRecordReturnFunction(prefix, name, record) {
  if (name === undefined) return undefined;
  const re = new RegExp(`\\b${escapeRegExp(record.name)}\\s+${escapeRegExp(name)}\\s*$`, "u");
  return re.test(prefix) ? { name } : undefined;
}

function scalarizedRecordReturnPrefix(prefix, name, record) {
  const re = new RegExp(`\\b${escapeRegExp(record.name)}\\s+${escapeRegExp(name)}\\s*$`, "u");
  return `${prefix.replace(re, `void ${name}`)}(`;
}

function scalarizedRecordReturnParams(name, record) {
  return record.fields.map((field) => `${field.valueType} *${recordReturnFieldName(name, field)}`);
}

function recordReturnFieldName(name, field) {
  return `${name}__bg_return__${field.name}`;
}

function rewriteScalarizedRecordReturns(body, functionName, record) {
  const re = /\breturn\s+([A-Za-z_][A-Za-z0-9_]*)\s*;/gu;
  return body.replace(re, (_match, name) => {
    if (typeof name !== "string") return _match;
    return record.fields
      .map((field) => `*${recordReturnFieldName(functionName, field)} = ${recordFieldName(name, field)};`)
      .join(" ") + " return;";
  });
}

function rewriteScalarizedRecordReturnCallDeclarations(source, record, symbols, functionExpansions) {
  const recordReturnFunctions = functionExpansions.filter((expansion) => expansion.returnRecordName === record.name);
  if (recordReturnFunctions.length === 0) return source;
  const byName = new Map(recordReturnFunctions.map((expansion) => [expansion.name, expansion]));
  const re = new RegExp(`(^|[;{}]\\s*)${escapeRegExp(record.name)}\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\(`, "gmu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const variable = match[2];
    const callee = match[3];
    if (variable === undefined || callee === undefined || !byName.has(callee)) {
      re.lastIndex = match.index + 1;
      continue;
    }
    const open = source.indexOf("(", match.index + match[0].length - 1);
    const close = findBalanced(source, open, "(", ")");
    if (close === undefined) {
      re.lastIndex = match.index + 1;
      continue;
    }
    const semi = skipWhitespace(source, close + 1);
    if (source[semi] !== ";") {
      re.lastIndex = close + 1;
      continue;
    }
    symbols.set(variable, { kind: "local" });
    const args = splitTopLevel(source.slice(open + 1, close)).map((arg) => arg.trim()).filter(Boolean);
    const declarations = record.fields.map((field) => `${field.valueType} ${recordFieldName(variable, field)};`).join(" ");
    const returnArgs = record.fields.map((field) => `&${recordFieldName(variable, field)}`);
    out += source.slice(cursor, match.index);
    out += `${match[1] ?? ""}${declarations} ${callee}(${[...args, ...returnArgs].join(", ")});`;
    cursor = semi + 1;
    re.lastIndex = semi + 1;
  }
  out += source.slice(cursor);
  return out;
}

function rewriteScalarizedRecordMemcpys(source, record, symbols) {
  if (symbols.size === 0) return source;
  const re = /\bmemcpy\s*\(\s*&\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*sizeof\s*\(\s*\1\s*\)\s*\)\s*;/gu;
  return source.replace(re, (match, target, sourceArray) => {
    if (typeof target !== "string" || typeof sourceArray !== "string") return match;
    const symbol = symbols.get(target);
    if (symbol?.kind !== "local" && symbol?.kind !== "param") return match;
    let offset = 0;
    const statements = [];
    for (const field of record.fields) {
      const lanes = cudaVectorLanes(field.valueType);
      if (lanes.length === 0) {
        statements.push(`${recordFieldName(target, field)} = ${sourceArray}[${offset}];`);
        offset++;
        continue;
      }
      const ctor = `make_${field.valueType}`;
      statements.push(`${recordFieldName(target, field)} = ${ctor}(${lanes.map((_lane, laneIndex) => `${sourceArray}[${offset + laneIndex}]`).join(", ")});`);
      offset += lanes.length;
    }
    return statements.join(" ");
  });
}

function expandScalarizedRecordParam(param, record) {
  const pointerParam = expandScalarizedRecordPointerParam(param, record);
  if (pointerParam !== undefined) return pointerParam;
  const re = new RegExp(`^\\s*((?:(?:const|volatile|__restrict__|__restrict|restrict|__grid_constant__)\\s+)*)${escapeRegExp(record.name)}\\s*(&)?\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*$`, "u");
  const match = re.exec(param);
  if (match === null || match[3] === undefined) return undefined;
  const qualifiers = normalizeScalarizedRecordQualifiers(match[1] ?? "");
  const byRef = match[2] !== undefined;
  if (byRef && !/\bconst\b/u.test(qualifiers)) return undefined;
  const name = match[3];
  const prefix = qualifiers.length > 0 ? `${qualifiers} ` : "";
  return {
    name,
    pointer: false,
    params: record.fields.map((field) => field.pointer || field.dimensions.length > 0
      ? `${prefix}${field.valueType} *${recordFieldName(name, field)}`
      : `${prefix}${field.valueType} ${recordFieldName(name, field)}`),
  };
}

function expandScalarizedRecordPointerParam(param, record) {
  const re = new RegExp(`^\\s*((?:(?:const|volatile|__restrict__|__restrict|restrict)\\s+)*)${escapeRegExp(record.name)}\\s*\\*\\s*((?:(?:const|volatile|__restrict__|__restrict|restrict)\\s+)*)?([A-Za-z_][A-Za-z0-9_]*)\\s*$`, "u");
  const match = re.exec(param);
  if (match === null || match[3] === undefined) return undefined;
  const qualifiers = normalizeScalarizedRecordQualifiers(`${match[1] ?? ""} ${match[2] ?? ""}`);
  const name = match[3];
  const prefix = qualifiers.length > 0 ? `${qualifiers} ` : "";
  return {
    name,
    pointer: true,
    params: record.fields.map((field) => `${prefix}${field.valueType} *${recordFieldName(name, field)}`),
  };
}

function normalizeScalarizedRecordQualifiers(raw) {
  const out = [];
  for (const token of raw.split(/\s+/u).filter(Boolean)) {
    if (token === "__restrict__" || token === "__restrict" || token === "restrict" || token === "__grid_constant__") continue;
    if ((token === "const" || token === "volatile") && !out.includes(token)) out.push(token);
  }
  return out.join(" ");
}

function rewriteScalarizedRecordLocalDeclarations(source, record, symbols) {
  const re = new RegExp(`(^|[;{}]\\s*)${escapeRegExp(record.name)}\\s+([^;]+);`, "gmu");
  return source.replace(re, (match, prefix, declarators) => {
    if (typeof declarators !== "string" || /[*&()]/u.test(declarators)) return match;
    const names = splitTopLevel(declarators).map((item) => item.trim()).filter(Boolean);
    if (names.length === 0) return match;
    const expanded = [];
    for (const rawDeclarator of names) {
      const parsedDeclarator = parseScalarizedRecordDeclarator(rawDeclarator, record);
      if (parsedDeclarator === undefined) return match;
      const { fieldInitializers, name, outerDimensions } = parsedDeclarator;
      symbols.set(name, { kind: "local" });
      for (const [fieldIndex, field] of record.fields.entries()) {
        const initializer = fieldInitializers[fieldIndex];
        expanded.push(`${field.valueType} ${recordFieldName(name, field)}${outerDimensions}${field.dimensions.map((dim) => `[${dim}]`).join("")}${initializer === undefined ? "" : ` = ${initializer}`};`);
      }
    }
    return `${prefix}${expanded.join(" ")}`;
  });
}

function parseScalarizedRecordDeclarator(rawDeclarator, record) {
  const parsed = /^([A-Za-z_][A-Za-z0-9_]*)((?:\s*\[[^\]]+\])*)\s*(?:=\s*(.+))?$/u.exec(rawDeclarator);
  if (parsed?.[1] === undefined) return undefined;
  const initializer = parsed[3]?.trim();
  if (initializer !== undefined && (parsed[2]?.trim().length ?? 0) > 0) return undefined;
  const fieldInitializers = initializer === undefined ? [] : scalarizedRecordFieldInitializers(initializer, record);
  if (fieldInitializers === undefined) return undefined;
  return {
    name: parsed[1],
    outerDimensions: parsed[2] ?? "",
    fieldInitializers,
  };
}

function scalarizedRecordFieldInitializers(initializer, record) {
  const trimmed = initializer.trim();
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    const values = splitTopLevel(trimmed.slice(1, -1)).map((value) => value.trim());
    return record.fields.map((_field, index) => values[index] ?? "0");
  }
  if (/^[A-Za-z_][A-Za-z0-9_]*$/u.test(trimmed)) {
    return record.fields.map((field) => recordFieldName(trimmed, field));
  }
  return undefined;
}

function rewriteScalarizedRecordAssignments(source, record, symbols) {
  let out = source;
  for (const [sourceName, sourceSymbol] of symbols) {
    if (sourceSymbol.kind !== "local" && sourceSymbol.kind !== "param") continue;
    for (const [targetName, targetSymbol] of symbols) {
      if (targetSymbol.kind === "param-pointer") {
        const indexedRe = new RegExp(`\\b${escapeRegExp(targetName)}\\b((?:\\s*\\[[^\\]]+\\])+)\\s*=\\s*${escapeRegExp(sourceName)}\\s*;`, "gu");
        out = out.replace(indexedRe, (_match, indexes) => scalarizedRecordFieldStores(record, targetName, indexes, sourceName));
        const derefRe = new RegExp(`\\*\\s*${escapeRegExp(targetName)}\\s*=\\s*${escapeRegExp(sourceName)}\\s*;`, "gu");
        out = out.replace(derefRe, scalarizedRecordFieldStores(record, targetName, "[0]", sourceName));
        continue;
      }
      if (targetSymbol.kind === "local" || targetSymbol.kind === "param") {
        const localRe = new RegExp(`\\b${escapeRegExp(targetName)}\\s*=\\s*${escapeRegExp(sourceName)}\\s*;`, "gu");
        out = out.replace(localRe, record.fields
          .map((field) => `${recordFieldName(targetName, field)} = ${recordFieldName(sourceName, field)};`)
          .join(" "));
      }
    }
  }
  return out;
}

function scalarizedRecordFieldStores(record, targetName, indexes, sourceName) {
  return record.fields
    .map((field) => `${recordFieldName(targetName, field)}${indexes} = ${recordFieldName(sourceName, field)};`)
    .join(" ");
}

function rewriteScalarizedRecordMemberAccess(source, record, symbols) {
  let out = source;
  for (const [name, symbol] of symbols) {
    for (const field of record.fields) {
      if (symbol.kind === "param-pointer") {
        const arrowRe = new RegExp(`\\b${escapeRegExp(name)}\\s*->\\s*${escapeRegExp(field.name)}\\b`, "gu");
        out = out.replace(arrowRe, field.pointer ? recordFieldName(name, field) : `${recordFieldName(name, field)}[0]`);
      }
      const re = new RegExp(`\\b${escapeRegExp(name)}\\b((?:\\s*\\[[^\\]]+\\])*)\\s*\\.\\s*${escapeRegExp(field.name)}\\b((?:\\s*\\[[^\\]]+\\])*)`, "gu");
      out = out.replace(re, (_match, indexes, fieldIndexes) => {
        if (symbol.kind === "single-array-constant" && record.fields.length === 1 && field.dimensions.length > 0) {
          return `${name}${indexes ?? ""}${fieldIndexes ?? ""}`;
        }
        if (symbol.kind === "param-pointer") {
          if (field.pointer) return `${recordFieldName(name, field)}${fieldIndexes ?? ""}`;
          if (field.dimensions.length > 0) {
            const flattened = flattenScalarizedRecordPointerFieldIndexes(indexes ?? "", fieldIndexes ?? "", field.dimensions);
            if (flattened !== undefined) return `${recordFieldName(name, field)}[${flattened}]`;
          }
        }
        return `${recordFieldName(name, field)}${indexes ?? ""}${fieldIndexes ?? ""}`;
      });
    }
  }
  return out;
}

function flattenScalarizedRecordPointerFieldIndexes(recordIndexes, fieldIndexes, dimensions) {
  const recordParts = bracketIndexParts(recordIndexes);
  const fieldParts = bracketIndexParts(fieldIndexes);
  if (recordParts.length === 0) recordParts.push("0");
  if (recordParts.length > 1) return undefined;
  if (fieldParts.length > dimensions.length) return undefined;
  const fieldStride = dimensions.reduce((product, dimension) => product * dimension, 1);
  const terms = [`(${recordParts[0]}) * ${fieldStride}`];
  for (let index = 0; index < dimensions.length; index++) {
    const value = fieldParts[index] ?? "0";
    const stride = dimensions.slice(index + 1).reduce((product, dimension) => product * dimension, 1);
    terms.push(stride === 1 ? `(${value})` : `((${value}) * ${stride})`);
  }
  return terms.join(" + ");
}

function bracketIndexParts(indexes) {
  const out = [];
  const re = /\[\s*([^\]]+?)\s*\]/gu;
  let match;
  while ((match = re.exec(indexes)) !== null) {
    const value = match[1]?.trim();
    if (value) out.push(value);
  }
  return out;
}

function rewriteScalarizedRecordCalls(source, functionExpansions, symbolsByRecord) {
  if (functionExpansions.length === 0) return source;
  let out = source;
  for (const expansion of functionExpansions) {
    out = rewriteScalarizedRecordCallsForFunction(out, expansion, symbolsByRecord);
    out = rewriteScalarizedRecordLaunchCallsForFunction(out, expansion, symbolsByRecord);
  }
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
    const rewritten = scalarizedRecordRewrittenArgs(args, expansion, symbolsByRecord);
    if (rewritten === undefined) {
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

function rewriteScalarizedRecordLaunchCallsForFunction(source, expansion, symbolsByRecord) {
  const re = new RegExp(`\\b${escapeRegExp(expansion.name)}\\s*<<<`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const launchOpen = source.indexOf("<<<", match.index + expansion.name.length);
    const launchClose = findCudaLaunchClose(source, launchOpen);
    if (launchOpen < 0 || launchClose === undefined) {
      re.lastIndex = match.index + expansion.name.length;
      continue;
    }
    const argsOpen = skipWhitespace(source, launchClose + 3);
    if (source[argsOpen] !== "(") {
      re.lastIndex = launchClose + 3;
      continue;
    }
    const argsClose = findBalanced(source, argsOpen, "(", ")");
    if (argsClose === undefined) {
      re.lastIndex = argsOpen + 1;
      continue;
    }
    const args = splitTopLevel(source.slice(argsOpen + 1, argsClose)).map((arg) => arg.trim());
    const rewritten = scalarizedRecordRewrittenArgs(args, expansion, symbolsByRecord);
    if (rewritten === undefined) {
      re.lastIndex = argsClose + 1;
      continue;
    }
    out += source.slice(cursor, argsOpen + 1);
    out += rewritten.join(", ");
    cursor = argsClose;
    re.lastIndex = argsClose + 1;
  }
  out += source.slice(cursor);
  return out;
}

function scalarizedRecordRewrittenArgs(args, expansion, symbolsByRecord) {
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
  return changed ? rewritten : undefined;
}

function scalarizedRecordArgumentParts(arg, record, symbols) {
  const name = arg.trim();
  const addressed = scalarizedRecordAddressArgumentParts(name, record, symbols);
  if (addressed !== undefined) return addressed;
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/u.test(name) || !symbols.has(name)) return undefined;
  const symbol = symbols.get(name);
  return record.fields.map((field) => {
    if (symbol.kind === "single-array-constant" && record.fields.length === 1 && field.dimensions.length > 0) return name;
    return recordFieldName(name, field);
  });
}

function scalarizedRecordAddressArgumentParts(arg, record, symbols) {
  const addressMatch = /^&\s*([A-Za-z_][A-Za-z0-9_]*)(\s*(?:\[[\s\S]+\])*)\s*$/u.exec(arg);
  if (addressMatch?.[1] !== undefined) {
    const name = addressMatch[1];
    if (!symbols.has(name)) return undefined;
    const indexes = addressMatch[2]?.trim() ?? "";
    return record.fields.map((field) => scalarizedRecordFieldPointerExpression(name, field, indexes));
  }
  const pointerOffsetMatch = /^([A-Za-z_][A-Za-z0-9_]*)\s*\+\s*([\s\S]+)$/u.exec(arg);
  if (pointerOffsetMatch?.[1] !== undefined && pointerOffsetMatch[2] !== undefined) {
    const name = pointerOffsetMatch[1];
    const symbol = symbols.get(name);
    if (symbol?.kind !== "param-pointer") return undefined;
    const offset = pointerOffsetMatch[2].trim();
    return record.fields.map((field) => {
      const fieldName = recordFieldName(name, field);
      if (field.dimensions.length === 0) return `${fieldName} + (${offset})`;
      const fieldStride = field.dimensions.reduce((product, dimension) => product * dimension, 1);
      return `${fieldName} + ((${offset}) * ${fieldStride})`;
    });
  }
  return undefined;
}

function scalarizedRecordFieldPointerExpression(name, field, indexes) {
  const fieldName = recordFieldName(name, field);
  const indexParts = bracketIndexParts(indexes);
  if (field.dimensions.length === 0) {
    return indexParts.length === 0 ? `&${fieldName}` : `&${fieldName}[${indexParts[0]}]`;
  }
  const flattened = flattenScalarizedRecordPointerFieldIndexes(indexes, "", field.dimensions);
  if (flattened !== undefined) return `&${fieldName}[${flattened}]`;
  return indexParts.length === 0 ? `&${fieldName}[0]` : `&${fieldName}${indexes}`;
}

function recordFieldName(name, field) {
  return `${name}__${field.name}`;
}
