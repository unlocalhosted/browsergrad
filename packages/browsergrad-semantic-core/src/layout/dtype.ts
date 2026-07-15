import { LAYOUT_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";

export type BuiltinDTypeId =
  | "bool"
  | "i8"
  | "u8"
  | "i16"
  | "u16"
  | "i32"
  | "u32"
  | "i64"
  | "u64"
  | "f16"
  | "bf16"
  | "f32"
  | "f64";

export interface DTypeDefinition {
  readonly id: BuiltinDTypeId;
  readonly category: "boolean" | "signed-integer" | "unsigned-integer" | "float";
  readonly storageBits: 8 | 16 | 32 | 64;
  readonly alignmentBytes: 1 | 2 | 4 | 8;
  readonly byteOrder: "little";
}

export const BUILTIN_DTYPES: Readonly<Record<BuiltinDTypeId, DTypeDefinition>> = Object.freeze({
  bool: dtype("bool", "boolean", 8, 1),
  i8: dtype("i8", "signed-integer", 8, 1),
  u8: dtype("u8", "unsigned-integer", 8, 1),
  i16: dtype("i16", "signed-integer", 16, 2),
  u16: dtype("u16", "unsigned-integer", 16, 2),
  i32: dtype("i32", "signed-integer", 32, 4),
  u32: dtype("u32", "unsigned-integer", 32, 4),
  i64: dtype("i64", "signed-integer", 64, 8),
  u64: dtype("u64", "unsigned-integer", 64, 8),
  f16: dtype("f16", "float", 16, 2),
  bf16: dtype("bf16", "float", 16, 2),
  f32: dtype("f32", "float", 32, 4),
  f64: dtype("f64", "float", 64, 8),
});

export function getBuiltinDType(id: string, path = "$.dtype"): DTypeDefinition {
  if (!Object.hasOwn(BUILTIN_DTYPES, id)) {
    throw new SemanticSchemaError({
      code: LAYOUT_DIAGNOSTIC_CODES.unknownDType,
      stage: "verification",
      severity: "error",
      message: `unknown builtin dtype ${JSON.stringify(id)}`,
      path,
    });
  }
  return BUILTIN_DTYPES[id as BuiltinDTypeId];
}

function dtype(
  id: BuiltinDTypeId,
  category: DTypeDefinition["category"],
  storageBits: DTypeDefinition["storageBits"],
  alignmentBytes: DTypeDefinition["alignmentBytes"],
): DTypeDefinition {
  return Object.freeze({ id, category, storageBits, alignmentBytes, byteOrder: "little" });
}
