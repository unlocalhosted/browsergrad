import {
  isWgslFloat16Array,
  type WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import { cudaLiteTotalElements as totalElements } from "./cuda_lite_values.js";
import {
  type CudaLiteSemanticSymbol,
  type SemanticExpression,
  type SemanticKernelIrModule,
  type SemanticKernelIrOperation,
} from "./semantic_ir.js";
import { walkSemanticOperations } from "./semantic_ir.js";
import {
  CudaLiteCompilerError,
  type CompiledKernelInput,
  type CudaLiteScalarType,
} from "./types.js";
import {
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
} from "./vector_types.js";

export function validateReferenceInputs(
  input: CompiledKernelInput,
  kernelIr: SemanticKernelIrModule,
): void {
  for (const param of kernelIr.params) {
    if (param.valueType === "texture2d") {
      const texture = input.textures?.[param.name];
      if (!texture) throw referenceInputFailure(`missing texture input '${param.name}'`);
      validateSurfaceInput(`texture ${param.name}`, texture);
    } else if (param.valueType === "surface2d") {
      const surface = input.surfaces?.[param.name];
      if (!surface) throw referenceInputFailure(`missing surface input '${param.name}'`);
      validateSurfaceInput(param.name, surface);
    } else if (param.valueType === "devicepool") {
      const pool = input.memoryPools?.[param.name];
      if (!pool) throw referenceInputFailure(`missing memory pool input '${param.name}'`);
      validateMemoryPoolInput(param.name, pool);
    } else if (param.pointer) {
      const buffer = input.buffers[param.name];
      if (!buffer) throw referenceInputFailure(`missing buffer input '${param.name}'`);
      validateBufferInput(param.name, semanticSymbolValueType(param), buffer);
    } else if (isCudaVectorType(param.valueType)) {
      const vector = input.vectors?.[param.name];
      if (vector === undefined) throw referenceInputFailure(`missing vector input '${param.name}'`);
      validateTypedConstant(param.name, param.valueType, vector);
      const expected = cudaVectorLaneCount(param.valueType);
      if (vector.length < expected) throw referenceInputFailure(`vector '${param.name}' expects at least ${expected} elements`);
    } else if (input.scalars?.[param.name] === undefined) {
      throw referenceInputFailure(`missing scalar input '${param.name}'`);
    }
  }
  for (const constant of kernelIr.memory.filter((symbol) => symbol.kind === "constant")) {
    if (constant.initialized) continue;
    const value = input.constants?.[constant.name];
    if (value === undefined) throw referenceInputFailure(`missing constant input '${constant.name}'`);
    const valueType = semanticSymbolValueType(constant);
    if (constant.dimensions.length === 0 && isCudaVectorType(valueType)) {
      if (typeof value === "number") throw referenceInputFailure(`constant '${constant.name}' expects typed array`);
      validateTypedConstant(constant.name, valueType, value);
      const expected = cudaVectorLaneCount(valueType);
      if (value.length < expected) throw referenceInputFailure(`constant '${constant.name}' expects at least ${expected} elements`);
    } else if (constant.dimensions.length === 0) {
      if (typeof value !== "number") throw referenceInputFailure(`constant '${constant.name}' expects number`);
    } else {
      if (typeof value === "number") throw referenceInputFailure(`constant '${constant.name}' expects typed array`);
      validateTypedConstant(constant.name, valueType, value);
      const expected = totalElements(constant.dimensions);
      if (value.length < expected) throw referenceInputFailure(`constant '${constant.name}' expects at least ${expected} elements`);
    }
  }
  for (const global of kernelIr.memory.filter((symbol) => symbol.kind === "device-global")) {
    const value = input.deviceGlobals?.[global.name];
    if (value === undefined) continue;
    validateTypedDeviceGlobal(global.name, semanticSymbolValueType(global), value);
    const expected = totalElements(global.dimensions);
    if (value.length < expected) throw referenceInputFailure(`device global '${global.name}' expects at least ${expected} elements`);
  }
  for (const texture of kernelIr.memory.filter((symbol) => symbol.kind === "texture")) {
    const value = input.textures?.[texture.name];
    if (!value) throw referenceInputFailure(`missing texture input '${texture.name}'`);
    validateSurfaceInput(`texture ${texture.name}`, value);
  }
  for (const poolName of collectReferenceExternalDevicePoolNames(kernelIr.operations)) {
    const pool = input.memoryPools?.[poolName];
    if (!pool) throw referenceInputFailure(`missing memory pool input '${poolName}'`);
    validateMemoryPoolInput(poolName, pool);
  }
}

function semanticSymbolValueType(symbol: CudaLiteSemanticSymbol): Exclude<CudaLiteScalarType, "void"> {
  if (symbol.valueType === undefined || symbol.valueType === "void") {
    throw referenceInputFailure(`semantic symbol '${symbol.name}' has no value type`);
  }
  return symbol.valueType;
}

export function collectReferenceExternalDevicePoolNames(operations: readonly SemanticKernelIrOperation[]): readonly string[] {
  const out = new Set<string>();
  walkSemanticOperations(operations, (expression) => {
    if (expression.kind !== "call") return;
    const callName = expressionNameForSemantic(expression.callee);
    if (callName !== "deviceAllocate" && callName !== "streamOrderedAllocate") return;
    const pool = expression.args[0];
    if (pool?.kind !== "unary" || pool.operator !== "&" || pool.argument.kind !== "symbol") return;
    out.add(pool.argument.name);
  });
  return [...out];
}

function expressionNameForSemantic(expression: SemanticExpression): string | undefined {
  if (expression.kind === "symbol") return expression.name;
  if (expression.kind === "member") {
    const objectName = expressionNameForSemantic(expression.object);
    return objectName ? `${objectName}.${expression.property}` : expression.property;
  }
  return undefined;
}

function validateBufferInput(name: string, valueType: Exclude<CudaLiteScalarType, "void">, buffer: WgslTypedArray): void {
  const scalarType = cudaVectorScalarType(valueType);
  if ((valueType === "int" || scalarType === "int") && !(buffer instanceof Int32Array)) {
    throw referenceInputFailure(`buffer '${name}' expects Int32Array`);
  }
  if ((valueType === "uint" || scalarType === "uint") && !(buffer instanceof Uint32Array)) {
    throw referenceInputFailure(`buffer '${name}' expects Uint32Array`);
  }
  if ((valueType === "float" || valueType === "double" || valueType === "bf16" || scalarType === "float" || scalarType === "bf16") && !(buffer instanceof Float32Array)) {
    throw referenceInputFailure(`buffer '${name}' expects Float32Array`);
  }
  if ((valueType === "half" || scalarType === "half") && !isWgslFloat16Array(buffer)) {
    throw referenceInputFailure(`buffer '${name}' expects Float16Array`);
  }
  if (valueType === "bool" && !(buffer instanceof Uint32Array)) {
    throw referenceInputFailure(`buffer '${name}' expects Uint32Array`);
  }
  if (valueType === "complex64" && !(buffer instanceof Float32Array)) {
    throw referenceInputFailure(`buffer '${name}' expects interleaved Float32Array`);
  }
}

function validateSurfaceInput(name: string, value: { readonly width: number; readonly height: number; readonly data: Float32Array }): void {
  if (!(value.data instanceof Float32Array)) throw referenceInputFailure(`${name} expects Float32Array data`);
  if (!Number.isInteger(value.width) || value.width <= 0) throw referenceInputFailure(`${name} width must be positive`);
  if (!Number.isInteger(value.height) || value.height <= 0) throw referenceInputFailure(`${name} height must be positive`);
  const expected = value.width * value.height;
  if (value.data.length < expected) throw referenceInputFailure(`${name} expects at least ${expected} elements`);
}

function validateMemoryPoolInput(name: string, value: { readonly data: Uint32Array; readonly offset?: Uint32Array }): void {
  if (!(value.data instanceof Uint32Array)) throw referenceInputFailure(`${name} memory pool expects Uint32Array data`);
  if (value.offset !== undefined && (!(value.offset instanceof Uint32Array) || value.offset.length < 1)) {
    throw referenceInputFailure(`${name} memory pool offset expects Uint32Array length >= 1`);
  }
}

function validateTypedConstant(name: string, valueType: string, value: WgslTypedArray): void {
  const scalarType = cudaVectorScalarType(valueType as CudaLiteScalarType);
  if ((valueType === "int" || scalarType === "int") && !(value instanceof Int32Array)) {
    throw referenceInputFailure(`constant '${name}' expects Int32Array`);
  }
  if ((valueType === "uint" || scalarType === "uint") && !(value instanceof Uint32Array)) {
    throw referenceInputFailure(`constant '${name}' expects Uint32Array`);
  }
  if ((valueType === "float" || valueType === "double" || valueType === "bf16" || scalarType === "float" || scalarType === "bf16") && !(value instanceof Float32Array)) {
    throw referenceInputFailure(`constant '${name}' expects Float32Array`);
  }
  if ((valueType === "half" || scalarType === "half") && !isWgslFloat16Array(value)) {
    throw referenceInputFailure(`constant '${name}' expects Float16Array`);
  }
  if (valueType === "bool" && !(value instanceof Uint32Array)) {
    throw referenceInputFailure(`constant '${name}' expects Uint32Array`);
  }
  if (valueType === "complex64" && !(value instanceof Float32Array)) {
    throw referenceInputFailure(`constant '${name}' expects interleaved Float32Array`);
  }
}

function validateTypedDeviceGlobal(name: string, valueType: string, value: WgslTypedArray): void {
  const scalarType = cudaVectorScalarType(valueType as CudaLiteScalarType);
  if ((valueType === "int" || scalarType === "int") && !(value instanceof Int32Array)) {
    throw referenceInputFailure(`device global '${name}' expects Int32Array`);
  }
  if ((valueType === "uint" || valueType === "uchar" || scalarType === "uint" || valueType === "bool" || valueType === "voidptr") && !(value instanceof Uint32Array)) {
    throw referenceInputFailure(`device global '${name}' expects Uint32Array`);
  }
  if ((valueType === "float" || valueType === "double" || valueType === "bf16" || scalarType === "float" || scalarType === "bf16") && !(value instanceof Float32Array)) {
    throw referenceInputFailure(`device global '${name}' expects Float32Array`);
  }
  if ((valueType === "half" || scalarType === "half") && !isWgslFloat16Array(value)) {
    throw referenceInputFailure(`device global '${name}' expects Float16Array`);
  }
  if (valueType === "complex64" && !(value instanceof Float32Array)) {
    throw referenceInputFailure(`device global '${name}' expects interleaved Float32Array`);
  }
}

function referenceInputFailure(message: string): CudaLiteCompilerError {
  return new CudaLiteCompilerError(message, [{
    code: "reference-runtime-error",
    severity: "error",
    message,
    span: { start: 0, end: 0, line: 1, column: 1 },
  }]);
}
