import { SCHEMA_DIAGNOSTIC_CODES, schemaError } from "./diagnostics.js";
import { assertJsonValue, isJsonObject } from "./json.js";

export type FloatBitDType = "f16" | "bf16" | "f32" | "f64";

export interface FloatBits {
  readonly kind: "float-bits";
  readonly dtype: FloatBitDType;
  /** Most-significant nibble first, lowercase, fixed-width, without 0x. */
  readonly bits: string;
}

const HEX_WIDTH: Readonly<Record<FloatBitDType, number>> = Object.freeze({
  f16: 4,
  bf16: 4,
  f32: 8,
  f64: 16,
});

export function parseFloatBits(value: unknown, path = "$"): FloatBits {
  assertJsonValue(value);
  if (!isJsonObject(value)) invalid(path, "float bit record must be an object");
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record).sort();
  if (JSON.stringify(keys) !== JSON.stringify(["bits", "dtype", "kind"])) invalid(path, "float bit record has unknown or missing fields");
  if (record.kind !== "float-bits") invalid(`${path}.kind`, "float bit record kind must be float-bits");
  if (record.dtype !== "f16" && record.dtype !== "bf16" && record.dtype !== "f32" && record.dtype !== "f64") {
    invalid(`${path}.dtype`, "unsupported float bit dtype");
  }
  const width = HEX_WIDTH[record.dtype];
  if (typeof record.bits !== "string" || !new RegExp(`^[0-9a-f]{${width}}$`, "u").test(record.bits)) {
    invalid(`${path}.bits`, `${record.dtype} bits must be ${width} lowercase hexadecimal digits without 0x`);
  }
  return Object.freeze({ kind: "float-bits", dtype: record.dtype, bits: record.bits });
}

function invalid(path: string, message: string): never {
  throw schemaError(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue, message, { path });
}
