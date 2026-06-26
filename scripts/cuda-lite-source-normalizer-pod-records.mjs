import { requireNormalizerHelpers } from "./cuda-lite-source-normalizer-context.mjs";

const escapeRegExp = (...args) => requireNormalizerHelpers().escapeRegExp(...args);
const findBalanced = (...args) => requireNormalizerHelpers().findBalanced(...args);
const normalizeTemplateTypeArgument = (...args) => requireNormalizerHelpers().normalizeTemplateTypeArgument(...args);
const splitTopLevel = (...args) => requireNormalizerHelpers().splitTopLevel(...args);

export function collectPodRecordVectorAliases(source, definesByName = new Map()) {
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

export function podRecordVectorType(fields) {
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

export function rewritePodRecordConstructors(source, record) {
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

export function rewritePodRecordMemberAccess(source, record) {
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

export function normalizeBitpackedShortUnions(source) {
  const unions = collectBitpackedShortUnions(source);
  if (unions.length === 0) return source;
  let out = source;
  for (const union of unions) out = rewriteBitpackedShortUnion(out, union);
  return out;
}

export function collectBitpackedShortUnions(source) {
  const out = [];
  const re = /\bunion\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*struct(?:\s+(?:__align__|alignas)\s*\([^)]*\))*\s*\{\s*(?:signed\s+)?short\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*(?:signed\s+)?short\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*\}\s*;\s*(?:unsigned\s+int|uint|uint32_t)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*\}\s*;/gu;
  let match;
  while ((match = re.exec(source)) !== null) {
    const name = match[1];
    if (name === undefined || match[2] === undefined || match[3] === undefined || match[4] === undefined) continue;
    out.push({
      declaration: match[0],
      highField: match[3],
      lowField: match[2],
      name,
      packedField: match[4],
    });
  }
  return out;
}

function rewriteBitpackedShortUnion(source, union) {
  const variables = collectBitpackedShortUnionVariables(source, union.name);
  if (variables.size === 0) return source.replace(union.declaration, "");
  let out = source.replace(union.declaration, "");
  out = rewriteBitpackedShortUnionDeclarations(out, union.name);
  for (const variable of variables) {
    out = rewriteBitpackedShortUnionFieldWrites(out, variable, union);
    out = rewriteBitpackedShortUnionPackedField(out, variable, union);
    out = rewriteBitpackedShortUnionFieldReads(out, variable, union);
  }
  return out;
}

function collectBitpackedShortUnionVariables(source, unionName) {
  const variables = new Set();
  const re = new RegExp(`\\b${escapeRegExp(unionName)}\\s+([^;]+);`, "gu");
  let match;
  while ((match = re.exec(source)) !== null) {
    const declarators = match[1];
    if (typeof declarators !== "string" || /[=*&()[\]{}]/u.test(declarators)) continue;
    for (const raw of splitTopLevel(declarators).map((item) => item.trim()).filter(Boolean)) {
      if (/^[A-Za-z_][A-Za-z0-9_]*$/u.test(raw)) variables.add(raw);
    }
  }
  return variables;
}

function rewriteBitpackedShortUnionDeclarations(source, unionName) {
  const re = new RegExp(`\\b${escapeRegExp(unionName)}\\s+([^;]+);`, "gu");
  return source.replace(re, (match, declarators) => {
    if (typeof declarators !== "string" || /[=*&()[\]{}]/u.test(declarators)) return match;
    const names = splitTopLevel(declarators).map((item) => item.trim()).filter(Boolean);
    if (names.length === 0 || names.some((name) => !/^[A-Za-z_][A-Za-z0-9_]*$/u.test(name))) return match;
    return `uint ${names.join(", ")};`;
  });
}

function rewriteBitpackedShortUnionFieldWrites(source, variable, union) {
  let out = rewriteBitpackedShortUnionOneFieldWrite(source, variable, union.lowField, "low");
  out = rewriteBitpackedShortUnionOneFieldWrite(out, variable, union.highField, "high");
  return out;
}

function rewriteBitpackedShortUnionOneFieldWrite(source, variable, field, lane) {
  const re = new RegExp(`\\b${escapeRegExp(variable)}\\s*\\.\\s*${escapeRegExp(field)}\\s*=\\s*([^;]+);`, "gu");
  return source.replace(re, (_match, rawValue) => {
    const value = typeof rawValue === "string" ? rawValue.trim() : "0";
    if (lane === "low") {
      return `${variable} = (${variable} & 0xffff0000u) | (uint(${value}) & 0xffffu);`;
    }
    return `${variable} = (${variable} & 0x0000ffffu) | ((uint(${value}) & 0xffffu) << 16);`;
  });
}

function rewriteBitpackedShortUnionPackedField(source, variable, union) {
  const fieldRe = new RegExp(`\\b${escapeRegExp(variable)}\\s*\\.\\s*${escapeRegExp(union.packedField)}\\b`, "gu");
  return source.replace(fieldRe, variable);
}

function rewriteBitpackedShortUnionFieldReads(source, variable, union) {
  let out = source.replace(
    new RegExp(`\\b${escapeRegExp(variable)}\\s*\\.\\s*${escapeRegExp(union.lowField)}\\b`, "gu"),
    `((int(${variable} << 16)) >> 16)`,
  );
  out = out.replace(
    new RegExp(`\\b${escapeRegExp(variable)}\\s*\\.\\s*${escapeRegExp(union.highField)}\\b`, "gu"),
    `(int(${variable}) >> 16)`,
  );
  return out;
}
