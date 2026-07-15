import { SCHEMA_DIAGNOSTIC_CODES, schemaError } from "./diagnostics.js";
import { assertJsonValue, type JsonObject, type JsonValue } from "./json.js";
import { type DecodeLimits, resolveDecodeLimits } from "./limits.js";

const UTF8 = new TextEncoder();

export function canonicalizeJson(
  value: unknown,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): string {
  const limits = resolveDecodeLimits(options.limits);
  assertJsonValue(value, { limits });
  const canonical = encodeCanonical(value);
  const bytes = UTF8.encode(canonical).byteLength;
  if (bytes > limits.maxDocumentBytes) {
    throw schemaError(
      SCHEMA_DIAGNOSTIC_CODES.resourceLimit,
      `canonical document has ${bytes} UTF-8 bytes; limit is ${limits.maxDocumentBytes}`,
      { path: "$" },
    );
  }
  return canonical;
}

export function canonicalJsonBytes(
  value: unknown,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Uint8Array {
  return UTF8.encode(canonicalizeJson(value, options));
}

function encodeCanonical(value: JsonValue): string {
  if (value === null) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((entry) => encodeCanonical(entry)).join(",")}]`;
  const objectValue = value as JsonObject;
  const keys = Object.keys(objectValue).sort();
  return `{${keys.map((key) => `${JSON.stringify(key)}:${encodeCanonical(objectValue[key] as JsonValue)}`).join(",")}}`;
}
