import { requireNormalizerHelpers } from "./cuda-lite-source-normalizer-context.mjs";

const normalizeTemplateTypeArgument = (...args) => requireNormalizerHelpers().normalizeTemplateTypeArgument(...args);
const normalizeTemplateValueArgument = (...args) => requireNormalizerHelpers().normalizeTemplateValueArgument(...args);

export function evaluateTemplateIntegerExpression(expression, env) {
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

export function evaluateIntegerExpression(expression) {
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

export function tokenizeIntegerExpression(expression) {
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

export function parseIntegerExpression(parser, minPrecedence) {
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

export function parseIntegerPrefix(parser) {
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

export function applyIntegerBinaryOperator(operator, left, right) {
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

export function sizeofType(typeName) {
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

export function alignofType(typeName) {
  if (typeName === "float3" || typeName === "int3" || typeName === "uint3") return 4;
  if (typeName === "char3" || typeName === "uchar3") return 1;
  return sizeofType(typeName);
}

export function templateArgumentScore(args) {
  return args.reduce((score, arg) => {
    if (arg === undefined) return score;
    if (normalizeTemplateTypeArgument(arg) !== undefined) return score + 2;
    if (normalizeTemplateValueArgument(arg, "int") !== undefined) return score + 1;
    if (isTemplateSymbolArgument(arg)) return score + 1;
    return score;
  }, 0);
}

export function isTemplateSymbolArgument(arg) {
  return /^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$/u.test(String(arg).trim());
}

export function scanCudaFuncSetAttributeTemplateReferences(source) {
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

export function parseTemplatedCallee(source) {
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

export function splitTopLevel(source, separator = ",", trackAnglesOverride = undefined) {
  const parts = [];
  let start = 0;
  let angle = 0;
  let paren = 0;
  let bracket = 0;
  let brace = 0;
  const trackAngles = trackAnglesOverride ?? separator !== ";";
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

export function balancedParenContents(source, open) {
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

export function findBalanced(source, open, openChar, closeChar) {
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

export function skipWhitespace(source, index) {
  let cursor = index;
  while (cursor < source.length && /\s/u.test(source[cursor] ?? "")) cursor++;
  return cursor;
}

export function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
