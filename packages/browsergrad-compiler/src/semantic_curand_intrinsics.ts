import type { CudaLiteScalarType } from "./types.js";

type SemanticCurandReturnType = Exclude<CudaLiteScalarType, "void">;

export const SEMANTIC_CURAND_CALLS = new Set([
  "curand_init",
  "curand",
  "curand_uniform",
  "curand_uniform4",
  "curand_uniform_double",
  "curand_normal",
  "curand_normal2",
  "curand_normal4",
  "curand_normal_double",
  "curand_log_normal",
  "curand_log_normal2",
  "curand_log_normal4",
  "curand_log_normal_double",
  "curand_poisson",
  "curand_poisson4",
  "skipahead",
]);

export const SEMANTIC_CURAND_VECTOR_RETURN_TYPES = new Map<string, SemanticCurandReturnType>([
  ["curand_uniform4", "float4"],
  ["curand_normal2", "float2"],
  ["curand_normal4", "float4"],
  ["curand_log_normal2", "float2"],
  ["curand_log_normal4", "float4"],
  ["curand_poisson4", "uint4"],
]);

export const SEMANTIC_CURAND_VECTOR_CALLS = new Set(SEMANTIC_CURAND_VECTOR_RETURN_TYPES.keys());

export const SEMANTIC_CURAND_STATE_ONLY_CALLS = new Set([
  "curand",
  "curand_uniform",
  "curand_uniform4",
  "curand_uniform_double",
  "curand_normal",
  "curand_normal2",
  "curand_normal4",
  "curand_normal_double",
]);

export const SEMANTIC_CURAND_DISTRIBUTION_CALLS = new Set([
  "curand_log_normal",
  "curand_log_normal2",
  "curand_log_normal4",
  "curand_log_normal_double",
  "curand_poisson",
  "curand_poisson4",
]);

export const SEMANTIC_CURAND_POISSON_CALLS = new Set([
  "curand_poisson",
  "curand_poisson4",
]);

export const SEMANTIC_CURAND_ARITIES: readonly (readonly [string, readonly [min: number, max: number]])[] = [
  ["curand_init", [4, 4]],
  ["curand", [1, 1]],
  ["curand_uniform", [1, 1]],
  ["curand_uniform4", [1, 1]],
  ["curand_uniform_double", [1, 1]],
  ["curand_normal", [1, 1]],
  ["curand_normal2", [1, 1]],
  ["curand_normal4", [1, 1]],
  ["curand_normal_double", [1, 1]],
  ["curand_log_normal", [3, 3]],
  ["curand_log_normal2", [3, 3]],
  ["curand_log_normal4", [3, 3]],
  ["curand_log_normal_double", [3, 3]],
  ["curand_poisson", [2, 2]],
  ["curand_poisson4", [2, 2]],
  ["skipahead", [2, 2]],
];

const SEMANTIC_CURAND_ARITY_BY_NAME: ReadonlyMap<string, number> =
  new Map(SEMANTIC_CURAND_ARITIES.map(([name, [arity]]) => [name, arity]));

export function isSemanticCurandCallName(name: string | undefined): boolean {
  return name !== undefined && SEMANTIC_CURAND_CALLS.has(name);
}

export function isSemanticCurandInitCallName(name: string | undefined): boolean {
  return name === "curand_init";
}

export function isSemanticCurandSkipaheadCallName(name: string | undefined): boolean {
  return name === "skipahead";
}

export function isSemanticCurandStateOnlyCallName(name: string | undefined): boolean {
  return name !== undefined && SEMANTIC_CURAND_STATE_ONLY_CALLS.has(name);
}

export function isSemanticCurandDistributionCallName(name: string | undefined): boolean {
  return name !== undefined && SEMANTIC_CURAND_DISTRIBUTION_CALLS.has(name);
}

export function isSemanticCurandPoissonCallName(name: string | undefined): boolean {
  return name !== undefined && SEMANTIC_CURAND_POISSON_CALLS.has(name);
}

export function isSemanticCurandVectorCallName(name: string | undefined): boolean {
  return name !== undefined && SEMANTIC_CURAND_VECTOR_RETURN_TYPES.has(name);
}

export function semanticCurandStateArgumentIndex(name: string | undefined): 0 | 1 | 3 | undefined {
  if (isSemanticCurandInitCallName(name)) return 3;
  if (isSemanticCurandSkipaheadCallName(name)) return 1;
  if (isSemanticCurandStateOnlyCallName(name) || isSemanticCurandDistributionCallName(name)) return 0;
  return undefined;
}

export function semanticCurandScalarArgumentIndices(name: string | undefined): readonly number[] {
  if (isSemanticCurandInitCallName(name)) return [0, 1, 2];
  if (isSemanticCurandSkipaheadCallName(name)) return [0];
  if (isSemanticCurandPoissonCallName(name)) return [1];
  if (isSemanticCurandDistributionCallName(name)) return [1, 2];
  return [];
}

export function semanticCurandArity(name: string | undefined): number | undefined {
  return name === undefined ? undefined : SEMANTIC_CURAND_ARITY_BY_NAME.get(name);
}

export function semanticCurandReturnType(name: string | undefined): SemanticCurandReturnType | undefined {
  if (name === "curand_init" || name === "curand" || name === "skipahead" || name === "curand_poisson") return "uint";
  if (SEMANTIC_CURAND_VECTOR_RETURN_TYPES.has(name ?? "")) return SEMANTIC_CURAND_VECTOR_RETURN_TYPES.get(name ?? "");
  if (name === "curand_uniform" || name === "curand_uniform_double" ||
    name === "curand_normal" || name === "curand_normal_double" ||
    name === "curand_log_normal" || name === "curand_log_normal_double") {
    return "float";
  }
  return undefined;
}
