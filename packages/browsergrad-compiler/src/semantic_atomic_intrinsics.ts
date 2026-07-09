import type { CudaLiteScalarType } from "./types.js";

export type SemanticAtomicOp =
  | "add"
  | "sub"
  | "min"
  | "max"
  | "and"
  | "or"
  | "xor"
  | "exchange"
  | "cas"
  | "inc"
  | "dec";

const SEMANTIC_ATOMIC_OP_ENTRIES = [
  ["atomicAdd", "add"],
  ["atomicAdd_system", "add"],
  ["atomicSub", "sub"],
  ["atomicSub_system", "sub"],
  ["atomicMin", "min"],
  ["atomicMin_system", "min"],
  ["atomicMax", "max"],
  ["atomicMax_system", "max"],
  ["atomicMaxFloat", "max"],
  ["atomicAnd", "and"],
  ["atomicAnd_system", "and"],
  ["atomicOr", "or"],
  ["atomicOr_system", "or"],
  ["atomicXor", "xor"],
  ["atomicXor_system", "xor"],
  ["atomicExch", "exchange"],
  ["atomicExch_system", "exchange"],
  ["atomicCAS", "cas"],
  ["atomicCAS_system", "cas"],
  ["atomicInc", "inc"],
  ["atomicInc_system", "inc"],
  ["atomicDec", "dec"],
  ["atomicDec_system", "dec"],
] as const satisfies readonly (readonly [string, SemanticAtomicOp])[];

export const SEMANTIC_ATOMIC_OPS: ReadonlyMap<string, SemanticAtomicOp> =
  new Map(SEMANTIC_ATOMIC_OP_ENTRIES);

export const SEMANTIC_ATOMIC_ARITIES: readonly (readonly [string, readonly [min: number, max: number]])[] =
  SEMANTIC_ATOMIC_OP_ENTRIES.map(([callee, op]) => [callee, op === "cas" ? [3, 3] : [2, 2]] as const);

const SEMANTIC_ATOMIC_CALL_NAMES_BY_OP_MUTABLE = new Map<SemanticAtomicOp, Set<string>>();
for (const [callee, op] of SEMANTIC_ATOMIC_OP_ENTRIES) {
  const names = SEMANTIC_ATOMIC_CALL_NAMES_BY_OP_MUTABLE.get(op) ?? new Set<string>();
  names.add(callee);
  SEMANTIC_ATOMIC_CALL_NAMES_BY_OP_MUTABLE.set(op, names);
}

export const SEMANTIC_ATOMIC_CALL_NAMES_BY_OP: ReadonlyMap<SemanticAtomicOp, ReadonlySet<string>> =
  SEMANTIC_ATOMIC_CALL_NAMES_BY_OP_MUTABLE;

const EMPTY_ATOMIC_CALL_NAMES: ReadonlySet<string> = new Set();

export function semanticAtomicCallNamesForOperation(op: SemanticAtomicOp): ReadonlySet<string> {
  return SEMANTIC_ATOMIC_CALL_NAMES_BY_OP.get(op) ?? EMPTY_ATOMIC_CALL_NAMES;
}

export function semanticAtomicOperation(callee: string | undefined): SemanticAtomicOp | undefined {
  return callee === undefined ? undefined : SEMANTIC_ATOMIC_OPS.get(callee);
}

export function isSemanticAtomicCallName(callee: string | undefined): boolean {
  return semanticAtomicOperation(callee) !== undefined;
}

export function semanticAtomicSupportsFloat(op: SemanticAtomicOp | undefined): boolean {
  return op === "add" || op === "sub" || op === "min" || op === "max" || op === "exchange" || op === "cas";
}

export function semanticAtomicSupportsBfloatAdd(
  callee: string | undefined,
  targetType: CudaLiteScalarType | undefined,
): boolean {
  return targetType === "bf16" && semanticAtomicOperation(callee) === "add";
}

export function semanticAtomicSupportsDevicePointer(
  callee: string | undefined,
  targetType: CudaLiteScalarType,
): boolean {
  const op = semanticAtomicOperation(callee);
  if (semanticAtomicSupportsBfloatAdd(callee, targetType)) return true;
  if (targetType !== "float" && targetType !== "double" && targetType !== "int" && targetType !== "uint") return false;
  if (semanticAtomicSupportsFloat(op)) return true;
  return (targetType === "int" || targetType === "uint") && (op === "and" || op === "or" || op === "xor" || op === "inc" || op === "dec" || op === "cas");
}
