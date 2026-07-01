import { poolDataName } from "./pool_bindings.js";
import {
  poolPointerForAllocationCall,
  type PoolPointerAlias,
} from "./wgsl_ir_analysis.js";
import {
  type CudaLiteExpression,
  type CudaLiteScalarType,
  type SourceSpan,
} from "./types.js";

export interface PoolAccess {
  readonly poolName: string;
  readonly pointerExpression: CudaLiteExpression;
  readonly indexExpression: CudaLiteExpression;
  readonly valueType: CudaLiteScalarType;
  readonly rawBuffer?: boolean;
}

export interface WgslPoolAccessContext {
  poolPointerFor(name: string): PoolPointerAlias | undefined;
  nameFor(name: string): string;
}

export type WgslPoolExpressionEmitter = (expression: CudaLiteExpression) => string;

export function poolAccessForIndex(expression: CudaLiteExpression, context: WgslPoolAccessContext): PoolAccess | undefined {
  if (expression.kind !== "index") return undefined;
  const target = expression.target;
  if (target.kind === "cast" && target.pointer) {
    const pointer = poolPointerExpressionInfo(target.expression, context);
    if (!pointer) return undefined;
    return {
      poolName: pointer.poolName,
      pointerExpression: target.expression,
      indexExpression: expression.index,
      valueType: target.valueType,
      ...(pointer.rawBuffer ? { rawBuffer: true } : {}),
    };
  }
  if (target.kind === "identifier") {
    const pointer = context.poolPointerFor(target.name);
    if (!pointer) return undefined;
    return {
      poolName: pointer.poolName,
      pointerExpression: target,
      indexExpression: expression.index,
      valueType: "float",
      ...(pointer.rawBuffer ? { rawBuffer: true } : {}),
    };
  }
  return undefined;
}

export function poolPointerExpressionInfo(
  expression: CudaLiteExpression,
  context: WgslPoolAccessContext,
): PoolPointerAlias | undefined {
  if (expression.kind === "identifier") return context.poolPointerFor(expression.name);
  if (expression.kind === "call") return poolPointerForAllocationCall(expression);
  if (expression.kind === "cast" && expression.pointer) return poolPointerExpressionInfo(expression.expression, context);
  return undefined;
}

export function poolZeroIndex(span: SourceSpan): CudaLiteExpression {
  return { kind: "number", value: 0, raw: "0", span };
}

export function emitPoolRead(
  access: PoolAccess,
  context: WgslPoolAccessContext,
  emitExpression: WgslPoolExpressionEmitter,
): string {
  const raw = poolRawAccess(access, context, emitExpression);
  if (access.rawBuffer) return raw;
  return decodePoolWord(access.valueType, raw);
}

export function emitPoolAssignment(
  access: PoolAccess,
  operator: string,
  right: CudaLiteExpression,
  context: WgslPoolAccessContext,
  emitExpression: WgslPoolExpressionEmitter,
): string {
  const raw = poolRawAccess(access, context, emitExpression);
  const rightValue = emitExpression(right);
  if (access.rawBuffer) {
    if (operator === "=") return `${raw} = ${rightValue}`;
    const op = operator.slice(0, -1);
    return `${raw} = (${raw} ${op} ${rightValue})`;
  }
  if (operator === "=") return `${raw} = ${encodePoolWord(access.valueType, rightValue)}`;
  const current = decodePoolWord(access.valueType, raw);
  const op = operator.slice(0, -1);
  return `${raw} = ${encodePoolWord(access.valueType, `(${current} ${op} ${rightValue})`)}`;
}

function poolRawAccess(
  access: PoolAccess,
  context: WgslPoolAccessContext,
  emitExpression: WgslPoolExpressionEmitter,
): string {
  const name = access.rawBuffer ? access.poolName : poolDataName(access.poolName);
  return `${context.nameFor(name)}[${poolWordIndex(access, emitExpression)}]`;
}

function poolWordIndex(access: PoolAccess, emitExpression: WgslPoolExpressionEmitter): string {
  const pointer = emitExpression(access.pointerExpression);
  const index = emitExpression(access.indexExpression);
  const stride = access.valueType === "complex64" ? "2u" : "1u";
  return `(((${pointer} - 1u) / 4u) + (u32(${index}) * ${stride}))`;
}

function decodePoolWord(type: CudaLiteScalarType, raw: string): string {
  if (type === "float") return `bitcast<f32>(${raw})`;
  if (type === "int") return `bitcast<i32>(${raw})`;
  if (type === "uint" || type === "voidptr" || type === "devicepool" || type === "surface2d") return raw;
  if (type === "bool") return `(${raw} != 0u)`;
  if (type === "half") return `f16(0.0)`;
  if (type === "complex64") return `vec2<f32>(bitcast<f32>(${raw}), bitcast<f32>(${raw}))`;
  return raw;
}

function encodePoolWord(type: CudaLiteScalarType, value: string): string {
  if (type === "float") return `bitcast<u32>(${value})`;
  if (type === "int") return `bitcast<u32>(${value})`;
  if (type === "uint" || type === "voidptr" || type === "devicepool" || type === "surface2d") return `u32(${value})`;
  if (type === "bool") return `select(0u, 1u, ${value})`;
  if (type === "half") return "0u";
  if (type === "complex64") return `bitcast<u32>(${value}.x)`;
  return `u32(${value})`;
}
