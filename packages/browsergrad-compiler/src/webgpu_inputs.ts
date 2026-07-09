import {
  createWgslFloat16Array,
  type WgslStorageBufferMetadata,
  type WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import { flattenCudaLiteInitializerExpressions as flattenInitializer } from "./ast_initializers.js";
import { cudaLiteTotalElements as totalElements } from "./cuda_lite_values.js";
import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";
import { pointerBaseOffsetUniformName } from "./pointer_offsets.js";
import { poolDataName, poolOffsetName } from "./pool_bindings.js";
import type {
  SemanticExpression,
  SemanticKernelIrOperation,
} from "./semantic_ir.js";
import { walkSemanticOperations } from "./semantic_ir.js";
import { isCudaVectorType } from "./vector_types.js";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CudaLiteDeviceGlobal,
  type CudaLiteDiagnostic,
  type CudaLiteExpression,
  type CudaLiteGlobalConstant,
  type SourceSpan,
} from "./types.js";

export type CudaWebGpuUniformParamDescriptor =
  | {
      readonly kind: "scalar";
      readonly name: string;
      readonly valueType: NonNullable<CudaLiteSemanticSymbolValueType>;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "constant";
      readonly name: string;
      readonly valueType: CudaLiteGlobalConstant["valueType"];
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "surface-dimension";
      readonly name: string;
      readonly valueType: "uint";
      readonly surface: string;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "pointer-base";
      readonly name: string;
      readonly valueType: "uint";
      readonly pointerBase: string;
      readonly span: SourceSpan;
    };

type CudaLiteSemanticSymbolValueType = CompiledCudaLiteKernel["kernelIr"]["params"][number]["valueType"];

export function surfaceBufferInputs(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = {};
  for (const surface of compiled.kernelIr.params.filter((param) => param.addressSpace === "surface")) {
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
  for (const constant of externalConstantDeclarations(compiled).filter((item) =>
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
  for (const global of deviceGlobalDeclarations(compiled)) {
    out[global.name] = input.deviceGlobals?.[global.name] ?? deviceGlobalInitialValue(global);
  }
  return out;
}

export function cudaWebGpuUniformParamDescriptors(
  compiled: CompiledCudaLiteKernel,
): readonly CudaWebGpuUniformParamDescriptor[] {
  return [
    ...compiled.kernelIr.params.flatMap((param): readonly CudaWebGpuUniformParamDescriptor[] =>
      param.addressSpace === "uniform" && param.valueType !== undefined
        ? [{ kind: "scalar", name: param.name, valueType: param.valueType, span: param.span }]
        : []
    ),
    ...externalConstantDeclarations(compiled).flatMap((constant): readonly CudaWebGpuUniformParamDescriptor[] =>
      constant.dimensions.length === 0 && constant.init === undefined && !isCudaVectorType(constant.valueType)
        ? [{ kind: "constant", name: constant.name, valueType: constant.valueType, span: constant.span }]
        : []
    ),
    ...compiled.kernelIr.params
      .filter((param) => param.addressSpace === "surface")
      .flatMap((param): readonly CudaWebGpuUniformParamDescriptor[] => [
        { kind: "surface-dimension", name: `${param.name}_width`, valueType: "uint", surface: param.name, span: param.span },
        { kind: "surface-dimension", name: `${param.name}_height`, valueType: "uint", surface: param.name, span: param.span },
      ]),
    ...compiled.kernelIr.params.flatMap((param): readonly CudaWebGpuUniformParamDescriptor[] =>
      param.pointer && compiled.pointerBaseOffsets?.[param.name] !== undefined
        ? [{
          kind: "pointer-base",
          name: pointerBaseOffsetUniformName(param.name),
          valueType: "uint",
          pointerBase: param.name,
          span: param.span,
        }]
        : []
    ),
  ];
}

export function cudaWebGpuDefaultReadbackNames(compiled: CompiledCudaLiteKernel): readonly string[] {
  return [
    ...compiled.kernelIr.params
      .filter((param) =>
        (param.addressSpace === "storage" && param.pointer && !param.constant) ||
        param.addressSpace === "surface" ||
        isDevicePoolParam(param)
      )
      .map((param) => isDevicePoolParam(param) ? poolDataName(param.name) : param.name),
    ...deviceGlobalDeclarations(compiled).map((global) => global.name),
    ...memoryPoolDescriptors(compiled).map((pool) => poolDataName(pool.name)),
  ].filter((name, index, names) => names.indexOf(name) === index);
}

export function cudaWebGpuMemoryPoolDataAliases(compiled: CompiledCudaLiteKernel): ReadonlyMap<string, string> {
  return new Map(memoryPoolDescriptors(compiled).map((pool) => [pool.name, poolDataName(pool.name)] as const));
}

export function isDevicePoolParam(param: { readonly pointer?: boolean; readonly valueType?: string }): boolean {
  return Boolean(param.pointer && param.valueType === "devicepool");
}

function memoryPoolDescriptors(compiled: CompiledCudaLiteKernel): Array<{ readonly name: string; readonly span: CudaLiteDiagnostic["span"] }> {
  const out = new Map<string, SourceSpan>();
  for (const param of compiled.kernelIr.params.filter(isDevicePoolParam)) out.set(param.name, param.span);
  for (const descriptor of collectExternalDevicePoolDescriptors(compiled.kernelIr.operations)) {
    if (!out.has(descriptor.name)) out.set(descriptor.name, descriptor.span);
  }
  return [...out].map(([name, span]) => ({ name, span }));
}

function externalConstantDeclarations(compiled: CompiledCudaLiteKernel): readonly CudaLiteGlobalConstant[] {
  const semanticNames = new Set(
    compiled.kernelIr.memory
      .filter((symbol) => symbol.kind === "constant")
      .map((symbol) => symbol.name),
  );
  const storageBindingNames = storageBindingNameSet(compiled);
  return compiled.analysis.constants.filter((constant) =>
    semanticNames.has(constant.name) &&
    (storageBindingNames.has(constant.name) || constant.dimensions.length === 0)
  );
}

function deviceGlobalDeclarations(compiled: CompiledCudaLiteKernel): readonly CudaLiteDeviceGlobal[] {
  const semanticNames = new Set(
    compiled.kernelIr.memory
      .filter((symbol) => symbol.kind === "device-global")
      .map((symbol) => symbol.name),
  );
  const storageBindingNames = storageBindingNameSet(compiled);
  return compiled.analysis.deviceGlobals.filter((global) =>
    semanticNames.has(global.name) &&
    storageBindingNames.has(global.name)
  );
}

function storageBindingNameSet(compiled: CompiledCudaLiteKernel): ReadonlySet<string> {
  return new Set(
    compiled.wgslProgram.bindings
      .filter((binding) => binding.kind === "storage")
      .map((binding) => binding.name),
  );
}

function collectExternalDevicePoolDescriptors(
  operations: readonly SemanticKernelIrOperation[],
): readonly { readonly name: string; readonly span: SourceSpan }[] {
  const out = new Map<string, SourceSpan>();
  walkSemanticOperations(operations, (expression) => {
    if (expression.kind !== "call") return;
    const callName = semanticExpressionName(expression.callee);
    if (callName !== "deviceAllocate" && callName !== "streamOrderedAllocate") return;
    const pool = expression.args[0];
    if (pool?.kind !== "unary" || pool.operator !== "&" || pool.argument.kind !== "symbol") return;
    if (!out.has(pool.argument.name)) out.set(pool.argument.name, pool.argument.span);
  });
  return [...out].map(([name, span]) => ({ name, span }));
}

function semanticExpressionName(expression: SemanticExpression): string | undefined {
  return expression.kind === "symbol" ? expression.name : undefined;
}

function deviceGlobalInitialValue(global: CudaLiteDeviceGlobal): WgslTypedArray {
  const total = totalElements(global.dimensions);
  const values = global.init === undefined
    ? []
    : flattenInitializer(global.init).map(evaluateInitializerNumber);
  const padded = Array.from({ length: total }, (_, index) => values[index] ?? 0);
  if (global.valueType === "int") return Int32Array.from(padded.map((value) => Math.trunc(value)));
  if (global.valueType === "uint" || global.valueType === "uchar" || global.valueType === "bool" || global.valueType === "voidptr") {
    return Uint32Array.from(padded.map((value) => Math.trunc(value) >>> 0));
  }
  if (global.valueType === "half") return createWgslFloat16Array(padded);
  return Float32Array.from(padded);
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
