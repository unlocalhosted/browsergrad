import {
  createWgslFloat16Array,
  type WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import { flattenCudaLiteInitializerExpressions } from "./ast_initializers.js";
import {
  cudaLiteTotalElements as totalElements,
  cudaLiteTruthy as truthy,
} from "./cuda_lite_values.js";
import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";
import {
  referenceFloat32ToUintBits,
  referenceUintBitsToFloat32,
} from "./reference_bitcasts.js";
import {
  CudaLiteCompilerError,
  type CudaLiteDeviceGlobal,
  type CudaLiteExpression,
  type CudaLiteGlobalConstant,
  type CudaLiteScalarType,
} from "./types.js";
import {
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
} from "./vector_types.js";

export function referenceConstantInitialValue(constant: CudaLiteGlobalConstant): number | WgslTypedArray {
  if (!constant.init) throw referenceInitializerFailure(`constant '${constant.name}' has no initializer`);
  if (constant.dimensions.length === 0 && isCudaVectorType(constant.valueType)) {
    return referenceTypedVectorConstantValues(
      constant.valueType,
      referenceConstantVectorInitializerValues(constant.init, constant.valueType),
    );
  }
  const values = flattenCudaLiteInitializerExpressions(constant.init).map(referenceEvaluateConstantNumber);
  if (constant.dimensions.length === 0) return values[0] ?? 0;
  return referenceTypedConstantArrayValue(constant.valueType, constant.dimensions, values);
}

export function referenceDeviceGlobalInitialValue(global: CudaLiteDeviceGlobal): WgslTypedArray {
  const values = global.init === undefined
    ? []
    : flattenCudaLiteInitializerExpressions(global.init).map(referenceEvaluateConstantNumber);
  return referenceTypedDeviceGlobalArrayValue(global.valueType, global.dimensions, values);
}

function referenceConstantVectorInitializerValues(
  expression: CudaLiteExpression,
  valueType: CudaLiteScalarType,
): readonly number[] {
  if (expression.kind === "call" && referenceExpressionName(expression.callee) === `make_${valueType}`) {
    return expression.args.map(referenceEvaluateConstantNumber);
  }
  return flattenCudaLiteInitializerExpressions(expression).map(referenceEvaluateConstantNumber);
}

function referenceTypedVectorConstantValues(valueType: CudaLiteScalarType, values: readonly number[]): WgslTypedArray {
  const lanes = Array.from({ length: cudaVectorLaneCount(valueType) }, (_, index) => values[index] ?? 0);
  const scalar = cudaVectorScalarType(valueType);
  if (scalar === "int") return Int32Array.from(lanes.map((value) => Math.trunc(value)));
  if (scalar === "uint") return Uint32Array.from(lanes.map((value) => Math.trunc(value) >>> 0));
  if (scalar === "half") return createWgslFloat16Array(lanes);
  return Float32Array.from(lanes);
}

function referenceTypedConstantArrayValue(
  valueType: Exclude<CudaLiteScalarType, "void">,
  dimensions: readonly number[],
  values: readonly number[],
): WgslTypedArray {
  const padded = referencePaddedInitializerValues(dimensions, values);
  if (valueType === "int") return Int32Array.from(padded.map((value) => Math.trunc(value)));
  if (valueType === "uint" || valueType === "bool" || valueType === "voidptr") {
    return Uint32Array.from(padded.map((value) => Math.trunc(value) >>> 0));
  }
  if (valueType === "half") return createWgslFloat16Array(padded);
  if (valueType === "float" || valueType === "double") return Float32Array.from(padded);
  if (valueType === "complex64") return Float32Array.from(padded);
  return Float32Array.from(padded);
}

function referenceTypedDeviceGlobalArrayValue(
  valueType: Exclude<CudaLiteScalarType, "void">,
  dimensions: readonly number[],
  values: readonly number[],
): WgslTypedArray {
  const padded = referencePaddedInitializerValues(dimensions, values);
  if (valueType === "int") return Int32Array.from(padded.map((value) => Math.trunc(value)));
  if (valueType === "uint" || valueType === "uchar" || valueType === "bool" || valueType === "voidptr") {
    return Uint32Array.from(padded.map((value) => Math.trunc(value) >>> 0));
  }
  if (valueType === "half") return createWgslFloat16Array(padded);
  if (valueType === "float" || valueType === "double") return Float32Array.from(padded);
  if (valueType === "complex64") return Float32Array.from(padded);
  return Float32Array.from(padded);
}

function referencePaddedInitializerValues(dimensions: readonly number[], values: readonly number[]): readonly number[] {
  return Array.from({ length: totalElements(dimensions) }, (_, index) => values[index] ?? 0);
}

function referenceEvaluateConstantNumber(expression: CudaLiteExpression): number {
  switch (expression.kind) {
    case "number":
      return expression.value;
    case "identifier": {
      const named = CUDA_NAMED_CONSTANTS.get(expression.name);
      if (named) return named.value;
      throw referenceInitializerFailure(`constant initializer unknown symbol '${expression.name}'`);
    }
    case "cast":
      return referenceCastConstantNumber(expression.valueType, referenceEvaluateConstantNumber(expression.expression));
    case "unary": {
      const value = referenceEvaluateConstantNumber(expression.argument);
      if (expression.operator === "-") return -value;
      if (expression.operator === "+") return value;
      return truthy(value) ? 0 : 1;
    }
    case "binary":
      return referenceEvalConstantBinary(
        expression.operator,
        referenceEvaluateConstantNumber(expression.left),
        referenceEvaluateConstantNumber(expression.right),
      );
    case "conditional":
      return truthy(referenceEvaluateConstantNumber(expression.condition))
        ? referenceEvaluateConstantNumber(expression.consequent)
        : referenceEvaluateConstantNumber(expression.alternate);
    default:
      throw referenceInitializerFailure("constant initializer must be a numeric constant expression");
  }
}

function referenceCastConstantNumber(type: Exclude<CudaLiteScalarType, "void">, value: number): number {
  if (type === "int") return Math.trunc(value) | 0;
  if (type === "uint") return Math.trunc(value) >>> 0;
  if (type === "bool") return truthy(value) ? 1 : 0;
  if (type === "bf16") return referenceRoundBfloat16(value);
  if (type === "complex64") throw referenceInitializerFailure("'constant initializer' is not a scalar");
  return value;
}

function referenceEvalConstantBinary(operator: string, left: number, right: number): number {
  switch (operator) {
    case "+":
      return left + right;
    case "-":
      return left - right;
    case "*":
      return left * right;
    case "/":
      return right === 0 ? 0 : left / right;
    case "%":
      return right === 0 ? 0 : left % right;
    case "<<":
      return Math.trunc(left) << Math.trunc(right);
    case ">>":
      return Math.trunc(left) >> Math.trunc(right);
    case "&":
      return Math.trunc(left) & Math.trunc(right);
    case "^":
      return Math.trunc(left) ^ Math.trunc(right);
    case "|":
      return Math.trunc(left) | Math.trunc(right);
    case "<":
      return left < right ? 1 : 0;
    case "<=":
      return left <= right ? 1 : 0;
    case ">":
      return left > right ? 1 : 0;
    case ">=":
      return left >= right ? 1 : 0;
    case "==":
      return left === right ? 1 : 0;
    case "!=":
      return left !== right ? 1 : 0;
    case "&&":
      return truthy(left) && truthy(right) ? 1 : 0;
    case "||":
      return truthy(left) || truthy(right) ? 1 : 0;
    default:
      throw referenceInitializerFailure(`unsupported constant initializer operator '${operator}'`);
  }
}

function referenceRoundBfloat16(value: number): number {
  const bits = referenceFloat32ToUintBits(value);
  return referenceUintBitsToFloat32((bits + 0x8000) & 0xffff0000);
}

function referenceExpressionName(expression: CudaLiteExpression): string | undefined {
  return expression.kind === "identifier" ? expression.name : undefined;
}

function referenceInitializerFailure(message: string): CudaLiteCompilerError {
  return new CudaLiteCompilerError(message, [{
    code: "reference-runtime-error",
    severity: "error",
    message,
    span: { start: 0, end: 0, line: 1, column: 1 },
  }]);
}
