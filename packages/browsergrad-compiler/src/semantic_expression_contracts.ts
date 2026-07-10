import type { CudaLiteScalarType } from "./types.js";

/** Backend-neutral expression rules shared by semantic backend adapters. */
export function semanticAssignmentOperatorSupported(operator: string): boolean {
  return operator === "=" || semanticAssignmentBinaryOperator(operator) !== undefined;
}

/** Maps a scalar compound assignment to its binary operation. */
export function semanticAssignmentBinaryOperator(operator: string): string | undefined {
  switch (operator) {
    case "+=": return "+";
    case "-=": return "-";
    case "*=": return "*";
    case "/=": return "/";
    case "%=": return "%";
    case "<<=": return "<<";
    case ">>=": return ">>";
    case "&=": return "&";
    case "^=": return "^";
    case "|=": return "|";
    default: return undefined;
  }
}

/** Vector writes remain limited to operators with modeled lane semantics. */
export function semanticVectorAssignmentOperatorSupported(operator: string): boolean {
  return operator === "=" || operator === "+=" || operator === "-=";
}

export function semanticVectorBinaryOperatorSupported(operator: string): boolean {
  return operator === "+" || operator === "-" || operator === "*" || operator === "/";
}

export function semanticSurfaceReadValueType(
  valueType: CudaLiteScalarType | undefined,
): Exclude<CudaLiteScalarType, "void"> {
  return valueType === undefined || valueType === "void" ? "float" : valueType;
}
