import {
  createWgslFloat16Array,
  type WgslStorageBufferMetadata,
  type WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import { collectExternalDevicePoolNames } from "./ast_queries.js";
import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";
import { poolDataName, poolOffsetName } from "./pool_bindings.js";
import { isCudaVectorType } from "./vector_types.js";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CudaLiteDeviceGlobal,
  type CudaLiteDiagnostic,
  type CudaLiteExpression,
} from "./types.js";

export function surfaceBufferInputs(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = {};
  for (const surface of compiled.ir.params.filter((param) => param.valueType === "surface2d")) {
    const value = input.surfaces?.[surface.name];
    if (!value) {
      throw new CudaLiteCompilerError(`missing surface input '${surface.name}'`, [{
        code: "missing-surface",
        severity: "error",
        message: `missing surface input '${surface.name}'`,
        span: surface.span,
      }]);
    }
    out[surface.name] = value.data;
  }
  return out;
}

export function memoryPoolBufferInputs(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = {};
  for (const pool of memoryPoolDescriptors(compiled)) {
    const value = input.memoryPools?.[pool.name];
    if (!value) {
      throw new CudaLiteCompilerError(`missing memory pool input '${pool.name}'`, [{
        code: "missing-memory-pool",
        severity: "error",
        message: `missing memory pool input '${pool.name}'`,
        span: pool.span,
      }]);
    }
    if (!(value.data instanceof Uint32Array)) {
      throw new CudaLiteCompilerError(`memory pool '${pool.name}' expects Uint32Array data`, [{
        code: "invalid-memory-pool",
        severity: "error",
        message: `memory pool '${pool.name}' expects Uint32Array data`,
        span: pool.span,
      }]);
    }
    const offset = value.offset ?? new Uint32Array([0]);
    if (!(offset instanceof Uint32Array) || offset.length < 1) {
      throw new CudaLiteCompilerError(`memory pool '${pool.name}' offset expects Uint32Array length >= 1`, [{
        code: "invalid-memory-pool",
        severity: "error",
        message: `memory pool '${pool.name}' offset expects Uint32Array length >= 1`,
        span: pool.span,
      }]);
    }
    out[poolDataName(pool.name)] = value.data;
    out[poolOffsetName(pool.name)] = offset;
  }
  return out;
}

export function memoryPoolStorageMetadata(
  compiled: CompiledCudaLiteKernel,
): Record<string, WgslStorageBufferMetadata> {
  const out: Record<string, WgslStorageBufferMetadata> = {};
  for (const pool of memoryPoolDescriptors(compiled)) {
    out[poolDataName(pool.name)] = { valueType: "u32", compatibleValueTypes: ["f32", "i32"] };
    out[poolOffsetName(pool.name)] = { valueType: "u32" };
  }
  return out;
}

export function constantBufferInputs(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = {};
  for (const constant of compiled.ir.constants.filter((item) =>
    item.init === undefined &&
    (item.dimensions.length > 0 || isCudaVectorType(item.valueType))
  )) {
    const value = input.constants?.[constant.name];
    if (!value || typeof value === "number") {
      throw new CudaLiteCompilerError(`missing constant buffer '${constant.name}'`, [{
        code: "missing-constant",
        severity: "error",
        message: `missing constant buffer '${constant.name}'`,
        span: constant.span,
      }]);
    }
    out[constant.name] = value;
  }
  return out;
}

export function deviceGlobalBufferInputs(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = {};
  for (const global of compiled.ir.deviceGlobals) {
    out[global.name] = input.deviceGlobals?.[global.name] ?? deviceGlobalInitialValue(global);
  }
  return out;
}

export function isDevicePoolParam(param: { readonly pointer: boolean; readonly valueType: string }): boolean {
  return param.pointer && param.valueType === "devicepool";
}

function memoryPoolDescriptors(compiled: CompiledCudaLiteKernel): Array<{ readonly name: string; readonly span: CudaLiteDiagnostic["span"] }> {
  return [
    ...compiled.ir.params.filter(isDevicePoolParam).map((param) => ({ name: param.name, span: param.span })),
    ...collectExternalDevicePoolNames(compiled.ir.body).map((name) => ({
      name,
      span: compiled.ir.body[0]?.span ?? compiled.ir.params[0]?.span ?? { start: 0, end: 0, line: 1, column: 1 },
    })),
  ];
}

function deviceGlobalInitialValue(global: CudaLiteDeviceGlobal): WgslTypedArray {
  const total = global.dimensions.length === 0
    ? 1
    : global.dimensions.reduce((product, dimension) => product * dimension, 1);
  const values = global.init === undefined
    ? []
    : flattenInitializer(global.init).map(evaluateInitializerNumber);
  const padded = Array.from({ length: total }, (_, index) => values[index] ?? 0);
  if (global.valueType === "int") return Int32Array.from(padded.map((value) => Math.trunc(value)));
  if (global.valueType === "uint" || global.valueType === "bool" || global.valueType === "voidptr") {
    return Uint32Array.from(padded.map((value) => Math.trunc(value) >>> 0));
  }
  if (global.valueType === "half") return createWgslFloat16Array(padded);
  return Float32Array.from(padded);
}

function flattenInitializer(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  if (expression.kind !== "initializer") return [expression];
  return expression.elements.flatMap((element) => flattenInitializer(element));
}

function evaluateInitializerNumber(expression: CudaLiteExpression): number {
  switch (expression.kind) {
    case "number":
      return expression.value;
    case "identifier": {
      const named = CUDA_NAMED_CONSTANTS.get(expression.name);
      if (named) return named.value;
      throw invalidDeviceGlobalInitializer(expression, `device global initializer unknown symbol '${expression.name}'`);
    }
    case "cast":
      return evaluateInitializerNumber(expression.expression);
    case "unary": {
      const value = evaluateInitializerNumber(expression.argument);
      if (expression.operator === "-") return -value;
      if (expression.operator === "+") return value;
      if (expression.operator === "!") return value === 0 ? 1 : 0;
      if (expression.operator === "~") return ~Math.trunc(value);
      break;
    }
    case "binary":
      return evaluateInitializerBinary(expression);
  }
  throw invalidDeviceGlobalInitializer(expression, "device global initializer must be numeric");
}

function evaluateInitializerBinary(expression: Extract<CudaLiteExpression, { kind: "binary" }>): number {
  const left = evaluateInitializerNumber(expression.left);
  const right = evaluateInitializerNumber(expression.right);
  switch (expression.operator) {
    case "+": return left + right;
    case "-": return left - right;
    case "*": return left * right;
    case "/": return right === 0 ? 0 : left / right;
    case "%": return right === 0 ? 0 : left % right;
    case "<<": return Math.trunc(left) << Math.trunc(right);
    case ">>": return Math.trunc(left) >> Math.trunc(right);
    case "&": return Math.trunc(left) & Math.trunc(right);
    case "|": return Math.trunc(left) | Math.trunc(right);
    case "^": return Math.trunc(left) ^ Math.trunc(right);
    case "==": return left === right ? 1 : 0;
    case "!=": return left !== right ? 1 : 0;
    case "<": return left < right ? 1 : 0;
    case "<=": return left <= right ? 1 : 0;
    case ">": return left > right ? 1 : 0;
    case ">=": return left >= right ? 1 : 0;
    case "&&": return left !== 0 && right !== 0 ? 1 : 0;
    case "||": return left !== 0 || right !== 0 ? 1 : 0;
  }
  throw invalidDeviceGlobalInitializer(expression, "device global initializer must be numeric");
}

function invalidDeviceGlobalInitializer(expression: CudaLiteExpression, message: string): CudaLiteCompilerError {
  return new CudaLiteCompilerError(message, [{
    code: "invalid-device-global-initializer",
    severity: "error",
    message,
    span: expression.span,
  }]);
}
