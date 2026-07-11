import type { CudaLiteScalarType } from "./types.js";

export const SEMANTIC_GENERATED_RANDOM_CALLS = new Set([
  "bg_random_uniform",
  "bg_random_normal",
  "bg_random_poisson4",
]);

export function semanticGeneratedRandomReturnType(name: string): CudaLiteScalarType | undefined {
  if (name === "bg_random_uniform" || name === "bg_random_normal") return "float";
  if (name === "bg_random_poisson4") return "int";
  return undefined;
}

export function isSemanticGeneratedRandomCall(name: string | undefined): boolean {
  return name !== undefined && SEMANTIC_GENERATED_RANDOM_CALLS.has(name);
}
