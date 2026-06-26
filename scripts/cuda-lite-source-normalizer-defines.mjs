import { requireNormalizerHelpers } from "./cuda-lite-source-normalizer-context.mjs";

const evaluateTemplateIntegerExpression = (...args) => requireNormalizerHelpers().evaluateTemplateIntegerExpression(...args);
const findBalanced = (...args) => requireNormalizerHelpers().findBalanced(...args);
const normalizeTemplateTypeArgument = (...args) => requireNormalizerHelpers().normalizeTemplateTypeArgument(...args);
const parseTemplateParam = (...args) => requireNormalizerHelpers().parseTemplateParam(...args);
const pruneCudaPreprocessorBranches = (...args) => requireNormalizerHelpers().pruneCudaPreprocessorBranches(...args);
const scanTemplatedCallReferences = (...args) => requireNormalizerHelpers().scanTemplatedCallReferences(...args);
const scanTemplatedFunctionDefinitions = (...args) => requireNormalizerHelpers().scanTemplatedFunctionDefinitions(...args);
const skipWhitespace = (...args) => requireNormalizerHelpers().skipWhitespace(...args);
const splitTopLevel = (...args) => requireNormalizerHelpers().splitTopLevel(...args);
const substituteTemplateArgument = (...args) => requireNormalizerHelpers().substituteTemplateArgument(...args);
const templateEnvironment = (...args) => requireNormalizerHelpers().templateEnvironment(...args);

export function collectObjectDefines(source) {
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

export function collectEnumIntegerConstants(source, initialDefines = new Map()) {
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

export function collectTypeAliasDefines(source, initialDefines = new Map()) {
  const defines = new Map(initialDefines);
  const out = new Map();
  for (const line of pruneCudaPreprocessorBranches(source, defines).split(/\r?\n/u)) {
    const alias = parseSimpleTypeAlias(stripLineComment(line), defines);
    if (alias !== undefined) {
      defines.set(alias.name, alias.value);
      out.set(alias.name, alias.value);
    }
  }
  return out;
}

export function collectFunctionDefineBodies(functionDeclarations) {
  const defines = new Map();
  for (const declaration of functionDeclarations) {
    const match = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s+([\s\S]+)$/u.exec(declaration);
    if (match?.[1] !== undefined && match[2] !== undefined) defines.set(match[1], match[2].trim());
  }
  return defines;
}

export function collectCarrierMemberDefines(source, initialDefines = new Map()) {
  const out = new Map();
  const carriers = scanTemplateStructCarriers(source);
  if (carriers.size === 0) return out;
  const aliases = scanCarrierAliases(source);
  for (const alias of aliases) {
    collectCarrierMemberDefinesForAlias(out, carriers, alias, initialDefines);
  }
  for (const alias of scanInstantiatedCarrierAliases(source, initialDefines)) {
    collectCarrierMemberDefinesForAlias(out, carriers, alias, initialDefines);
  }
  return out;
}

function collectCarrierMemberDefinesForAlias(out, carriers, alias, initialDefines) {
  const carrier = carriers.get(alias.templateName);
  if (carrier === undefined) return;
  const env = templateEnvironment(carrier.params, alias.args, mergeDefineMaps(initialDefines, out));
  const memberEnv = new Map(env);
  for (const [index, param] of carrier.params.entries()) {
    if (param?.kind !== "type") continue;
    const arg = alias.args[index];
    if (arg !== undefined && parseTemplateShapeArgument(arg).length > 0) memberEnv.set(param.name, arg);
  }
  if (memberEnv.size === 0) return;
  for (const member of carrier.members) {
    if (member.kind === "type") {
      const substituted = substituteTemplateArgument(member.value, memberEnv, mergeDefineMaps(initialDefines, out));
      const normalized = normalizeTemplateTypeArgument(substituted, mergeDefineMaps(initialDefines, out));
      if (normalized === undefined) continue;
      out.set(`${alias.name}::${member.name}`, normalized);
      memberEnv.set(member.name, normalized);
      continue;
    }
    const value = evaluateCarrierIntegerExpression(member.value, memberEnv);
    if (value === undefined) continue;
    out.set(`${alias.name}::${member.name}`, value);
    memberEnv.set(member.name, value);
  }
}

function evaluateCarrierIntegerExpression(expression, env) {
  const shapeGet = /^\s*get\s*<\s*([0-9]+)\s*>\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*\}\s*\)\s*$/u.exec(expression);
  if (shapeGet?.[1] !== undefined && shapeGet[2] !== undefined) {
    const dims = parseTemplateShapeArgument(env.get(shapeGet[2]) ?? "");
    const value = dims[Number(shapeGet[1])];
    if (value !== undefined) return value;
  }
  return evaluateTemplateIntegerExpression(expression, env);
}

function parseTemplateShapeArgument(raw) {
  const source = raw.trim();
  const open = source.indexOf("make_shape");
  if (open < 0) return [];
  const paren = source.indexOf("(", open);
  const close = findBalanced(source, paren, "(", ")");
  if (paren < 0 || close === undefined) return [];
  const dims = [];
  for (const arg of splitTopLevel(source.slice(paren + 1, close))) {
    const trimmed = arg.trim();
    const underscore = /^_([0-9]+)\s*\{\s*\}$/u.exec(trimmed)?.[1];
    const intWrapper = /^(?:cute::)?Int\s*<\s*([0-9]+)\s*>\s*\{\s*\}$/u.exec(trimmed)?.[1];
    const numeric = /^[0-9]+$/u.test(trimmed) ? trimmed : undefined;
    const value = underscore ?? intWrapper ?? numeric;
    if (value === undefined) return [];
    dims.push(value);
  }
  return dims;
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
  for (const match of body.matchAll(/\b(?:(?:static\s+constexpr)|(?:constexpr\s+static))\s+(?:const\s+)?(?:int|uint|unsigned\s+int|size_t|ptrdiff_t|bool)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);/gu)) {
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

function scanInstantiatedCarrierAliases(source, definesByName = new Map()) {
  const aliases = [];
  const calls = scanTemplatedCallReferences(source);
  if (calls.length === 0) return aliases;
  const functions = scanTemplatedFunctionDefinitions(source);
  for (const fn of functions) {
    const matchingCalls = calls.filter((call) => call.name === fn.name && call.args.length > 0);
    if (matchingCalls.length === 0) continue;
    for (const call of matchingCalls) {
      const env = templateEnvironment(fn.templateParams, call.args, definesByName);
      if (env.size === 0) continue;
      for (const alias of scanCarrierAliases(fn.body)) {
        aliases.push({
          ...alias,
          args: alias.args.map((arg) => substituteTemplateArgument(arg, env, definesByName)),
        });
      }
    }
  }
  return aliases;
}

export function isKnownCarrierAlias(raw, definesByName) {
  const alias = raw.trim();
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/u.test(alias)) return false;
  for (const name of definesByName.keys()) {
    if (name.startsWith(`${alias}::`)) return true;
  }
  return false;
}

export function parseSimpleTypeAlias(line, defines) {
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

export function normalizeArrayAliasType(sourceType, rawLength, defines) {
  const type = normalizeTemplateTypeArgument(sourceType, defines);
  const length = Number(rawLength);
  if (!Number.isInteger(length) || length < 2 || length > 4) return undefined;
  if (type === "float") return `float${length}`;
  if (type === "int") return `int${length}`;
  if (type === "uint") return `uint${length}`;
  if (type === "half" && length === 2) return "half2";
  return undefined;
}

export function stripSupportedTypeAliasDeclarations(source, initialDefines = new Map()) {
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

export function stripSupportedEnumDeclarations(source) {
  return source.replace(/\benum(?:\s+(?:class\s+)?[A-Za-z_][A-Za-z0-9_]*)?\s*\{[\s\S]*?\}\s*;/gu, "");
}

function parseSimpleIntegerConstant(line) {
  const match = /^\s*((?:(?:static|constexpr|const)\s+)*)(?:(?:cuda\s*::\s*)?std\s*::\s*)?(?:int|uint|unsigned\s+int|size_t|ptrdiff_t|uint32_t|uint64_t|int32_t|int64_t)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z0-9_:,\s()+\-*/%<>&|^?:.]+)\s*;\s*$/u.exec(line);
  if (match === null) return undefined;
  const [, qualifiers, name, value] = match;
  if (!name || !value || !/\b(?:const|constexpr)\b/u.test(qualifiers ?? "")) return undefined;
  return { name, value: normalizeConstexprIntegerExpression(value) };
}

export function normalizeConstexprIntegerExpression(raw) {
  let out = raw
    .replace(/\b(?:cuda\s*::\s*)?std\s*::\s*(?:size_t|ptrdiff_t|uint32_t|uint64_t|int32_t|int64_t)\s*\(/gu, "(")
    .replace(/\(\s*(?:(?:cuda\s*::\s*)?std\s*::\s*)?(?:size_t|ptrdiff_t|uint32_t|uint64_t|int32_t|int64_t)\s*\)/gu, "")
    .trim();
  let cursor = 0;
  while (cursor < out.length) {
    const match = /\bcmax\s*\(/gu.exec(out.slice(cursor));
    if (match === null) break;
    const start = cursor + match.index;
    const open = out.indexOf("(", start);
    const close = findBalanced(out, open, "(", ")");
    if (open < 0 || close === undefined) break;
    const args = splitTopLevel(out.slice(open + 1, close));
    if (args.length !== 2 || args[0] === undefined || args[1] === undefined) {
      cursor = close + 1;
      continue;
    }
    const left = args[0].trim();
    const right = args[1].trim();
    out = `${out.slice(0, start)}((${left}) > (${right}) ? (${left}) : (${right}))${out.slice(close + 1)}`;
    cursor = start;
  }
  return out;
}

export function stripLineComment(line) {
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

export function mergeDefineMaps(...maps) {
  const merged = new Map();
  for (const map of maps) {
    for (const [name, value] of map) merged.set(name, value);
  }
  return merged;
}
