import { SCHEMA_DIAGNOSTIC_CODES, schemaError } from "./diagnostics.js";

declare const wireI64Brand: unique symbol;
declare const wireU64Brand: unique symbol;

export type WireI64 = string & { readonly [wireI64Brand]: "WireI64" };
export type WireU64 = string & { readonly [wireU64Brand]: "WireU64" };

export const I64_MIN = -(1n << 63n);
export const I64_MAX = (1n << 63n) - 1n;
export const U64_MAX = (1n << 64n) - 1n;

const SIGNED_DECIMAL = /^(?:0|-[1-9][0-9]*|[1-9][0-9]*)$/u;
const UNSIGNED_DECIMAL = /^(?:0|[1-9][0-9]*)$/u;

export function parseWireI64(value: unknown, path = "$"): WireI64 {
  const parsed = parseWireInteger(value, SIGNED_DECIMAL, I64_MIN, I64_MAX, path, "WireI64");
  return parsed as WireI64;
}

export function parseWireU64(value: unknown, path = "$"): WireU64 {
  const parsed = parseWireInteger(value, UNSIGNED_DECIMAL, 0n, U64_MAX, path, "WireU64");
  return parsed as WireU64;
}

export function encodeWireI64(value: bigint, path = "$"): WireI64 {
  return parseWireI64(value.toString(10), path);
}

export function encodeWireU64(value: bigint, path = "$"): WireU64 {
  return parseWireU64(value.toString(10), path);
}

export function wireIntegerToBigInt(value: WireI64 | WireU64): bigint {
  return BigInt(value);
}

function parseWireInteger(
  value: unknown,
  grammar: RegExp,
  minimum: bigint,
  maximum: bigint,
  path: string,
  name: string,
): string {
  if (typeof value !== "string" || !grammar.test(value)) {
    throw schemaError(
      SCHEMA_DIAGNOSTIC_CODES.nonCanonicalInteger,
      `${name} must be canonical ASCII base-10 without +, whitespace, leading zeroes, or -0`,
      { path },
    );
  }
  const parsed = BigInt(value);
  if (parsed < minimum || parsed > maximum) {
    throw schemaError(
      SCHEMA_DIAGNOSTIC_CODES.integerRange,
      `${name} is outside [${minimum.toString(10)}, ${maximum.toString(10)}]`,
      { path },
    );
  }
  return value;
}
