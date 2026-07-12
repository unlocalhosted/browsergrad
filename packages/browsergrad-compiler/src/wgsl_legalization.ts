import type { WgslValueType } from "@unlocalhosted/browsergrad-kernels";
import { walkSemanticOperations, type SemanticExpression, type SemanticKernelIrModule } from "./semantic_ir.js";
import type { TypeCheckedSemanticKernelIr } from "./semantic_type_check.js";
import { CudaLiteCompilerError, type CudaLiteDiagnostic, type CudaLiteScalarType, type SourceSpan } from "./types.js";
import { createTypedWgslExpression, emitTypedWgslBinary, type WgslBinaryOperator } from "./typed_wgsl_expression.js";
import { completeWgslLegalization, type WgslLegalizedIrArtifact } from "./compiler_phases.js";

const wgslLegalizedSemanticKernelIrArtifact: unique symbol = Symbol("wgsl-legalized-semantic-kernel-ir");

export interface WgslLegalizedSemanticKernelIr<T extends SemanticKernelIrModule = SemanticKernelIrModule> extends WgslLegalizedIrArtifact<T> {
  readonly typeChecked: TypeCheckedSemanticKernelIr<T>;
  readonly [wgslLegalizedSemanticKernelIrArtifact]: true;
}

export interface WgslLegalizationIssue {
  readonly code: "internal-wgsl-legalization-invariant";
  readonly message: string;
  readonly span: SourceSpan;
}

const binaryOperators = new Set<string>([
  "+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>",
  "<", "<=", ">", ">=", "==", "!=", "&&", "||",
]);
const logicalOperators = new Set(["&&", "||"]);
const comparisonOperators = new Set(["<", "<=", ">", ">=", "==", "!="]);

export function checkSemanticKernelIrWgslLegalization(
  typeChecked: TypeCheckedSemanticKernelIr,
): readonly WgslLegalizationIssue[] {
  const ir = typeChecked.ir;
  const issues: WgslLegalizationIssue[] = [];
  const visit = (expression: SemanticExpression): void => {
    if (expression.kind !== "binary" || !binaryOperators.has(expression.operator)) return;
    if (isVectorType(expression.valueType) || isPointerComparison(expression)) return;
    try {
      const operator = expression.operator as WgslBinaryOperator;
      const leftType = logicalOperators.has(operator) ? "bool" : binaryOperandType(expression);
      const rightType = operator === "<<" || operator === ">>" ? "u32" : leftType;
      const result = emitTypedWgslBinary(
        operator,
        createTypedWgslExpression("left", leftType, expression.left.span),
        createTypedWgslExpression("right", rightType, expression.right.span),
        expression.span,
      );
      const expected = comparisonOperators.has(operator) || logicalOperators.has(operator)
        ? "bool"
        : wgslScalar(expression.valueType);
      if (result.type !== expected) {
        issues.push({
          code: "internal-wgsl-legalization-invariant",
          message: `WGSL '${operator}' produces '${result.type}', semantic IR declares '${expected}'`,
          span: expression.span,
        });
      }
    } catch (error) {
      issues.push({
        code: "internal-wgsl-legalization-invariant",
        message: error instanceof Error ? error.message : String(error),
        span: expression.span,
      });
    }
  };

  walkSemanticOperations(ir.operations, visit);
  for (const fn of ir.functions) walkSemanticOperations(fn.body, visit);
  return issues;
}

export function legalizeSemanticKernelIrForWgsl<T extends SemanticKernelIrModule>(
  typeChecked: TypeCheckedSemanticKernelIr<T>,
): WgslLegalizedSemanticKernelIr<T> {
  const issues = checkSemanticKernelIrWgslLegalization(typeChecked);
  if (issues.length === 0) {
    return completeWgslLegalization(Object.freeze({
      kind: "wgsl-legalized-semantic-kernel-ir",
      ir: typeChecked.ir,
      typeChecked,
      [wgslLegalizedSemanticKernelIrArtifact]: true as const,
    }));
  }
  const diagnostics: readonly CudaLiteDiagnostic[] = issues.map((issue) => ({
    code: issue.code,
    severity: "error",
    message: issue.message,
    span: issue.span,
  }));
  throw new CudaLiteCompilerError(`WGSL legalization failed: ${issues[0]!.message}`, diagnostics);
}

function binaryOperandType(expression: Extract<SemanticExpression, { readonly kind: "binary" }>): WgslValueType {
  const left = wgslScalar(expressionValueType(expression.left));
  const right = wgslScalar(expressionValueType(expression.right));
  const result = wgslScalar(expression.valueType);
  if (left === "f32" || right === "f32" || result === "f32") return "f32";
  if (left === "f16" || right === "f16" || result === "f16") return "f16";
  if (left === "u32" || right === "u32" || result === "u32") return "u32";
  return "i32";
}

function expressionValueType(expression: SemanticExpression): CudaLiteScalarType | undefined {
  return "valueType" in expression ? expression.valueType : undefined;
}

function wgslScalar(type: CudaLiteScalarType | undefined): WgslValueType | "bool" {
  if (type === "bool") return "bool";
  if (type === "half") return "f16";
  if (type === "float" || type === "double" || type === "bf16") return "f32";
  if (type === "uint" || type === "uchar") return "u32";
  return "i32";
}

function isVectorType(type: CudaLiteScalarType | undefined): boolean {
  return type === "half2" || type === "bf162" || type === "float2" || type === "float3" || type === "float4" ||
    type === "int2" || type === "int3" || type === "int4" || type === "uint2" || type === "uint3" || type === "uint4";
}

function isPointerComparison(expression: Extract<SemanticExpression, { readonly kind: "binary" }>): boolean {
  return (expression.operator === "==" || expression.operator === "!=") &&
    (expression.left.kind === "symbol" && expression.left.addressSpace === "storage" ||
      expression.right.kind === "symbol" && expression.right.addressSpace === "storage");
}
