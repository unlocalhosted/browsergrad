import { SCHEMA_DIAGNOSTIC_CODES, schemaError } from "./diagnostics.js";
import { type DecodeLimits, resolveDecodeLimits } from "./limits.js";

export type JsonPrimitive = null | boolean | string | number;
export type JsonValue = JsonPrimitive | JsonArray | JsonObject;
export type JsonArray = readonly JsonValue[];
export interface JsonObject {
  readonly [key: string]: JsonValue;
}

const UTF8 = new TextEncoder();
const UTF8_FATAL = new TextDecoder("utf-8", { fatal: true });

/** Primary untrusted-wire entrypoint: budget bytes before UTF-8 decoding. */
export function decodeWireJson(
  bytes: Uint8Array,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): JsonValue {
  const limits = resolveDecodeLimits(options.limits);
  checkDocumentBytes(bytes.byteLength, limits);
  let source: string;
  try {
    source = UTF8_FATAL.decode(new Uint8Array(bytes));
  } catch {
    throw schemaError(
      SCHEMA_DIAGNOSTIC_CODES.invalidJson,
      "wire document is not valid UTF-8",
      { path: "$" },
    );
  }
  return new WireJsonParser(source, limits).parse();
}

/** Trusted in-memory helper. Untrusted bytes must use decodeWireJson. */
export function parseWireJson(
  source: string,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): JsonValue {
  const limits = resolveDecodeLimits(options.limits);
  const documentBytes = UTF8.encode(source).byteLength;
  checkDocumentBytes(documentBytes, limits);
  return new WireJsonParser(source, limits).parse();
}

export function assertJsonValue(
  value: unknown,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): asserts value is JsonValue {
  const limits = resolveDecodeLimits(options.limits);
  const stack: Array<{ readonly value: unknown; readonly path: string; readonly depth: number }> = [
    { value, path: "$", depth: 1 },
  ];
  const seen = new WeakSet<object>();
  let nodes = 0;
  let stringBytes = 0;

  while (stack.length > 0) {
    const current = stack.pop();
    if (current === undefined) break;
    nodes += 1;
    if (nodes > limits.maxNodes) resourceLimit("nodes", nodes, limits.maxNodes, current.path);
    if (current.depth > limits.maxDepth) resourceLimit("depth", current.depth, limits.maxDepth, current.path);

    const item = current.value;
    if (item === null || typeof item === "boolean") continue;
    if (typeof item === "string") {
      assertValidUnicode(item, current.path);
      stringBytes += UTF8.encode(item).byteLength;
      if (stringBytes > limits.maxStringBytes) resourceLimit("string bytes", stringBytes, limits.maxStringBytes, current.path);
      continue;
    }
    if (typeof item === "number") {
      if (!Number.isSafeInteger(item) || Object.is(item, -0)) {
        throw schemaError(
          SCHEMA_DIAGNOSTIC_CODES.unsafeNumber,
          "semantic JSON numbers must be safe integers and must not be negative zero",
          { path: current.path },
        );
      }
      continue;
    }
    if (typeof item !== "object") nonCanonicalValue(current.path, `unsupported ${typeof item} value`);
    if (seen.has(item)) nonCanonicalValue(current.path, "cycles and shared object references are not canonical JSON trees");
    seen.add(item);

    if (Array.isArray(item)) {
      if (item.length > limits.maxArrayLength) resourceLimit("array length", item.length, limits.maxArrayLength, current.path);
      const ownKeys = Reflect.ownKeys(item);
      if (ownKeys.length !== item.length + 1 || ownKeys.some((key) => {
        if (key === "length") return false;
        if (typeof key !== "string" || !/^(?:0|[1-9][0-9]*)$/u.test(key)) return true;
        const index = Number(key);
        return !Number.isSafeInteger(index) || index < 0 || index >= item.length || String(index) !== key;
      })) {
        nonCanonicalValue(current.path, "arrays must not have symbols or named properties");
      }
      const descriptors = Object.getOwnPropertyDescriptors(item);
      for (let index = item.length - 1; index >= 0; index -= 1) {
        const descriptor = descriptors[String(index)];
        if (descriptor === undefined) nonCanonicalValue(`${current.path}[${index}]`, "sparse arrays are not canonical JSON");
        if (descriptor.enumerable !== true || !("value" in descriptor)) {
          nonCanonicalValue(`${current.path}[${index}]`, "array elements must be enumerable data properties without accessors");
        }
        stack.push({ value: descriptor.value, path: `${current.path}[${index}]`, depth: current.depth + 1 });
      }
      continue;
    }

    const prototype = Object.getPrototypeOf(item);
    if (prototype !== Object.prototype && prototype !== null) nonCanonicalValue(current.path, "class instances are not canonical JSON objects");
    const keys = Reflect.ownKeys(item);
    if (keys.some((key) => typeof key !== "string")) nonCanonicalValue(current.path, "JSON object keys must be strings");
    if (keys.length > limits.maxObjectProperties) resourceLimit("object properties", keys.length, limits.maxObjectProperties, current.path);
    const descriptors = Object.getOwnPropertyDescriptors(item);
    for (let index = keys.length - 1; index >= 0; index -= 1) {
      const key = keys[index];
      if (typeof key !== "string") continue;
      assertValidUnicode(key, `${current.path}.[key]`);
      stringBytes += UTF8.encode(key).byteLength;
      if (stringBytes > limits.maxStringBytes) resourceLimit("string bytes", stringBytes, limits.maxStringBytes, current.path);
      const descriptor = descriptors[key];
      if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
        nonCanonicalValue(childPath(current.path, key), "JSON properties must be enumerable data properties without accessors");
      }
      stack.push({ value: descriptor.value, path: childPath(current.path, key), depth: current.depth + 1 });
    }
  }
}

export function deepFreezeJson<T extends JsonValue>(value: T): T {
  const stack: object[] = [];
  if (typeof value === "object" && value !== null) stack.push(value);
  const seen = new WeakSet<object>();
  while (stack.length > 0) {
    const current = stack.pop();
    if (current === undefined || seen.has(current)) continue;
    seen.add(current);
    for (const child of Object.values(current)) {
      if (typeof child === "object" && child !== null) stack.push(child);
    }
    Object.freeze(current);
  }
  return value;
}

export function isJsonObject(value: JsonValue): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

class WireJsonParser {
  private offset = 0;
  private nodes = 0;
  private stringBytes = 0;

  constructor(
    private readonly source: string,
    private readonly limits: DecodeLimits,
  ) {}

  parse(): JsonValue {
    this.skipWhitespace();
    const value = this.parseValue("$", 1);
    this.skipWhitespace();
    if (this.offset !== this.source.length) this.invalid("unexpected trailing data", "$", this.offset);
    return value;
  }

  private parseValue(path: string, depth: number): JsonValue {
    this.consumeNode(path, depth);
    const char = this.source[this.offset];
    if (char === '"') return this.parseString(path);
    if (char === "{") return this.parseObject(path, depth);
    if (char === "[") return this.parseArray(path, depth);
    if (char === "t" && this.consumeLiteral("true")) return true;
    if (char === "f" && this.consumeLiteral("false")) return false;
    if (char === "n" && this.consumeLiteral("null")) return null;
    if (char === "-" || (char !== undefined && /[0-9]/u.test(char))) return this.parseNumber(path);
    this.invalid("expected a JSON value", path, this.offset);
  }

  private parseObject(path: string, depth: number): JsonObject {
    this.offset += 1;
    this.skipWhitespace();
    const result = Object.create(null) as Record<string, JsonValue>;
    const keys = new Set<string>();
    let propertyCount = 0;
    if (this.source[this.offset] === "}") {
      this.offset += 1;
      return result;
    }
    while (this.offset < this.source.length) {
      if (this.source[this.offset] !== '"') this.invalid("object key must be a JSON string", path, this.offset);
      const keyOffset = this.offset;
      const key = this.parseString(`${path}.[key]`);
      if (keys.has(key)) {
        throw schemaError(
          SCHEMA_DIAGNOSTIC_CODES.duplicateKey,
          `duplicate object key ${JSON.stringify(key)}`,
          { path: childPath(path, key), offset: keyOffset },
        );
      }
      keys.add(key);
      propertyCount += 1;
      if (propertyCount > this.limits.maxObjectProperties) resourceLimit("object properties", propertyCount, this.limits.maxObjectProperties, path);
      this.skipWhitespace();
      if (this.source[this.offset] !== ":") this.invalid("expected : after object key", childPath(path, key), this.offset);
      this.offset += 1;
      this.skipWhitespace();
      result[key] = this.parseValue(childPath(path, key), depth + 1);
      this.skipWhitespace();
      const separator = this.source[this.offset];
      if (separator === "}") {
        this.offset += 1;
        return result;
      }
      if (separator !== ",") this.invalid("expected , or } in object", path, this.offset);
      this.offset += 1;
      this.skipWhitespace();
    }
    this.invalid("unterminated object", path, this.offset);
  }

  private parseArray(path: string, depth: number): JsonArray {
    this.offset += 1;
    this.skipWhitespace();
    const result: JsonValue[] = [];
    if (this.source[this.offset] === "]") {
      this.offset += 1;
      return result;
    }
    while (this.offset < this.source.length) {
      if (result.length >= this.limits.maxArrayLength) resourceLimit("array length", result.length + 1, this.limits.maxArrayLength, path);
      result.push(this.parseValue(`${path}[${result.length}]`, depth + 1));
      this.skipWhitespace();
      const separator = this.source[this.offset];
      if (separator === "]") {
        this.offset += 1;
        return result;
      }
      if (separator !== ",") this.invalid("expected , or ] in array", path, this.offset);
      this.offset += 1;
      this.skipWhitespace();
    }
    this.invalid("unterminated array", path, this.offset);
  }

  private parseString(path: string): string {
    this.offset += 1;
    let result = "";
    while (this.offset < this.source.length) {
      const code = this.source.charCodeAt(this.offset);
      if (code === 0x22) {
        this.offset += 1;
        this.stringBytes += UTF8.encode(result).byteLength;
        if (this.stringBytes > this.limits.maxStringBytes) resourceLimit("string bytes", this.stringBytes, this.limits.maxStringBytes, path);
        return result;
      }
      if (code === 0x5c) {
        result += this.parseEscape(path);
        continue;
      }
      if (code < 0x20) this.invalid("unescaped control character in string", path, this.offset);
      if (isHighSurrogate(code)) {
        const low = this.source.charCodeAt(this.offset + 1);
        if (!isLowSurrogate(low)) this.invalid("lone high surrogate in string", path, this.offset);
        result += this.source.slice(this.offset, this.offset + 2);
        this.offset += 2;
        continue;
      }
      if (isLowSurrogate(code)) this.invalid("lone low surrogate in string", path, this.offset);
      result += this.source[this.offset];
      this.offset += 1;
    }
    this.invalid("unterminated string", path, this.offset);
  }

  private parseEscape(path: string): string {
    const escapeOffset = this.offset;
    this.offset += 1;
    const escape = this.source[this.offset];
    this.offset += 1;
    switch (escape) {
      case '"': return '"';
      case "\\": return "\\";
      case "/": return "/";
      case "b": return "\b";
      case "f": return "\f";
      case "n": return "\n";
      case "r": return "\r";
      case "t": return "\t";
      case "u": {
        const first = this.readUnicodeEscape(path, escapeOffset);
        if (isHighSurrogate(first)) {
          if (this.source.slice(this.offset, this.offset + 2) !== "\\u") this.invalid("escaped high surrogate must be followed by escaped low surrogate", path, escapeOffset);
          this.offset += 2;
          const second = this.readUnicodeEscape(path, this.offset - 2);
          if (!isLowSurrogate(second)) this.invalid("escaped high surrogate must be followed by escaped low surrogate", path, escapeOffset);
          return String.fromCodePoint(0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00));
        }
        if (isLowSurrogate(first)) this.invalid("lone escaped low surrogate in string", path, escapeOffset);
        return String.fromCharCode(first);
      }
      default:
        this.invalid("invalid JSON string escape", path, escapeOffset);
    }
  }

  private readUnicodeEscape(path: string, escapeOffset: number): number {
    const hex = this.source.slice(this.offset, this.offset + 4);
    if (!/^[0-9A-Fa-f]{4}$/u.test(hex)) this.invalid("invalid Unicode escape", path, escapeOffset);
    this.offset += 4;
    return Number.parseInt(hex, 16);
  }

  private parseNumber(path: string): number {
    const start = this.offset;
    if (this.source[this.offset] === "-") this.offset += 1;
    if (this.source[this.offset] === "0") {
      this.offset += 1;
      if (/[0-9]/u.test(this.source[this.offset] ?? "")) this.invalid("leading zero in JSON number", path, start);
    } else {
      if (!/[1-9]/u.test(this.source[this.offset] ?? "")) this.invalid("invalid JSON number", path, start);
      while (/[0-9]/u.test(this.source[this.offset] ?? "")) this.offset += 1;
    }
    let nonIntegerLexeme = false;
    if (this.source[this.offset] === ".") {
      nonIntegerLexeme = true;
      this.offset += 1;
      if (!/[0-9]/u.test(this.source[this.offset] ?? "")) this.invalid("fraction requires digits", path, start);
      while (/[0-9]/u.test(this.source[this.offset] ?? "")) this.offset += 1;
    }
    if (this.source[this.offset] === "e" || this.source[this.offset] === "E") {
      nonIntegerLexeme = true;
      this.offset += 1;
      if (this.source[this.offset] === "+" || this.source[this.offset] === "-") this.offset += 1;
      if (!/[0-9]/u.test(this.source[this.offset] ?? "")) this.invalid("exponent requires digits", path, start);
      while (/[0-9]/u.test(this.source[this.offset] ?? "")) this.offset += 1;
    }
    const lexeme = this.source.slice(start, this.offset);
    const value = Number(lexeme);
    if (nonIntegerLexeme || lexeme === "-0" || !Number.isSafeInteger(value)) {
      throw schemaError(
        SCHEMA_DIAGNOSTIC_CODES.unsafeNumber,
        "semantic JSON numbers must use canonical safe-integer lexical form; floats use bit-pattern records",
        { path, offset: start },
      );
    }
    return value;
  }

  private consumeLiteral(literal: string): boolean {
    if (this.source.slice(this.offset, this.offset + literal.length) !== literal) return false;
    this.offset += literal.length;
    return true;
  }

  private consumeNode(path: string, depth: number): void {
    this.nodes += 1;
    if (this.nodes > this.limits.maxNodes) resourceLimit("nodes", this.nodes, this.limits.maxNodes, path);
    if (depth > this.limits.maxDepth) resourceLimit("depth", depth, this.limits.maxDepth, path);
  }

  private skipWhitespace(): void {
    while (this.offset < this.source.length) {
      const code = this.source.charCodeAt(this.offset);
      if (code !== 0x09 && code !== 0x0a && code !== 0x0d && code !== 0x20) break;
      this.offset += 1;
    }
  }

  private invalid(message: string, path: string, offset: number): never {
    throw schemaError(SCHEMA_DIAGNOSTIC_CODES.invalidJson, message, { path, offset });
  }
}

function childPath(parent: string, key: string): string {
  return /^[A-Za-z_$][A-Za-z0-9_$]*$/u.test(key) ? `${parent}.${key}` : `${parent}[${JSON.stringify(key)}]`;
}

function checkDocumentBytes(actual: number, limits: DecodeLimits): void {
  if (actual > limits.maxDocumentBytes) {
    throw schemaError(
      SCHEMA_DIAGNOSTIC_CODES.resourceLimit,
      `document has ${actual} UTF-8 bytes; limit is ${limits.maxDocumentBytes}`,
      { path: "$" },
    );
  }
}

function assertValidUnicode(value: string, path: string): void {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (isHighSurrogate(code)) {
      if (!isLowSurrogate(value.charCodeAt(index + 1))) nonCanonicalValue(path, "lone high surrogate is not canonical Unicode");
      index += 1;
    } else if (isLowSurrogate(code)) {
      nonCanonicalValue(path, "lone low surrogate is not canonical Unicode");
    }
  }
}

function isHighSurrogate(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff;
}

function isLowSurrogate(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff;
}

function resourceLimit(kind: string, actual: number, maximum: number, path: string): never {
  throw schemaError(
    SCHEMA_DIAGNOSTIC_CODES.resourceLimit,
    `${kind} ${actual} exceeds limit ${maximum}`,
    { path },
  );
}

function nonCanonicalValue(path: string, message: string): never {
  throw schemaError(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue, message, { path });
}
