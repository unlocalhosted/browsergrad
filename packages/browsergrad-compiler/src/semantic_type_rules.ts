import type { CudaLiteScalarType } from "./types.js";
import { isCudaVectorType } from "./vector_types.js";
import { promotedCudaScalarType } from "./wgsl_value_conversion.js";

const comparisonOperators = new Set(["<", "<=", ">", ">=", "==", "!="]);
const logicalOperators = new Set(["&&", "||"]);
const bitwiseOperators = new Set(["&", "|", "^", "<<", ">>"]);

export function semanticBinaryResultType(
  operator: string,
  left: CudaLiteScalarType | undefined,
  right: CudaLiteScalarType | undefined,
): CudaLiteScalarType | undefined {
  if (comparisonOperators.has(operator) || logicalOperators.has(operator)) return "bool";
  if (bitwiseOperators.has(operator) && (left === "texture2d" || left === "surface2d")) return left;
  if (operator === "<<" || operator === ">>") return left;
  if (isCudaVectorType(left)) return left;
  if (isCudaVectorType(right)) return right;
  return promotedCudaScalarType(left, right);
}
