import { expressionName, rootIdentifier } from "./analyzer.js";
import { cudaVectorFieldIndex, cudaVectorLaneCount, cudaVectorScalarType, isCudaVectorType } from "./vector_types.js";
import { isPointerIdentityCall, type PointerAlias } from "./wgsl_ir_analysis.js";
import {
  CudaLiteCompilerError,
  type CudaLiteDeviceGlobal,
  type CudaLiteExpression,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type CudaLiteVarDecl,
  type KernelIrModule,
  type SourceSpan,
} from "./types.js";

export interface DevicePointerParts {
  readonly buffer: string;
  readonly base: string;
}

export const NULL_DEVICE_POINTER_BUFFER = "4294967295u";

export interface WgslDevicePointerContext {
  readonly ir: KernelIrModule;
  nameFor(name: string): string;
  devicePointerParamFor(name: string): CudaLiteParam | undefined;
  storagePointerIdFor(name: string): number | undefined;
  sharedPointerIdFor(name: string): number | undefined;
  constantPointerIdFor(name: string): number | undefined;
  deviceGlobalPointerIdFor(name: string): number | undefined;
  paramFor(name: string): CudaLiteParam | undefined;
  deviceGlobalFor(name: string): CudaLiteDeviceGlobal | undefined;
  pointerAliasFor(name: string, span?: SourceSpan): PointerAlias | undefined;
  localPointerHandleFor(name: string): CudaLiteVarDecl | undefined;
  localPointerArrayFor(name: string, span?: SourceSpan): CudaLiteVarDecl | undefined;
  localPointerArrayRootFor(name: string, span?: SourceSpan): CudaLiteVarDecl | undefined;
}

export interface WgslDevicePointerCallbacks {
  isNullPointerExpression(expression: CudaLiteExpression): boolean;
  emitExpression(expression: CudaLiteExpression, context: WgslDevicePointerContext): string;
  emitTruthinessExpression(expression: CudaLiteExpression, context: WgslDevicePointerContext): string;
  pointerBaseExpression(rootName: string, context: WgslDevicePointerContext): string | undefined;
  sharedDeclarationFor(rootName: string, context: WgslDevicePointerContext): CudaLiteVarDecl | undefined;
  featureError(code: string, message: string): CudaLiteCompilerError;
}

export function emitDevicePointerArgument(
  expression: CudaLiteExpression,
  context: WgslDevicePointerContext,
  callbacks: WgslDevicePointerCallbacks,
): readonly [string, string] {
  const parts = devicePointerArgumentParts(expression, context, callbacks);
  if (!parts) {
    const root = rootIdentifier(expression);
    if (root && context.localPointerArrayRootFor(root, expression.span)) {
      throw callbacks.featureError(
        "unsupported-device-pointer-param",
        `local-memory pointer array '${root}' cannot cross a storage pointer helper boundary`,
      );
    }
    throw callbacks.featureError("unsupported-device-pointer-param", "device pointer argument must be a storage pointer or derived storage address");
  }
  return [parts.buffer, parts.base];
}

export function devicePointerArgumentParts(
  expression: CudaLiteExpression,
  context: WgslDevicePointerContext,
  callbacks: WgslDevicePointerCallbacks,
): DevicePointerParts | undefined {
  if (callbacks.isNullPointerExpression(expression)) {
    return { buffer: NULL_DEVICE_POINTER_BUFFER, base: "0u" };
  }
  if (expression.kind === "conditional") {
    const consequent = devicePointerArgumentParts(expression.consequent, context, callbacks);
    const alternate = devicePointerArgumentParts(expression.alternate, context, callbacks);
    if (!consequent || !alternate) return undefined;
    const condition = callbacks.emitTruthinessExpression(expression.condition, context);
    return {
      buffer: `select(${alternate.buffer}, ${consequent.buffer}, ${condition})`,
      base: `select(${alternate.base}, ${consequent.base}, ${condition})`,
    };
  }
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    const pointer = expression.args[0];
    return pointer ? devicePointerArgumentParts(pointer, context, callbacks) : undefined;
  }
  if (expression.kind === "identifier") {
    if (context.localPointerHandleFor(expression.name)) {
      return {
        buffer: context.nameFor(`${expression.name}_buffer`),
        base: context.nameFor(`${expression.name}_base`),
      };
    }
    const pointerParam = context.devicePointerParamFor(expression.name);
    if (pointerParam) return { buffer: context.nameFor(`${expression.name}_buffer`), base: context.nameFor(`${expression.name}_base`) };
    const storageId = context.storagePointerIdFor(expression.name);
    const storageParam = context.paramFor(expression.name);
    if (storageId !== undefined && storageParam?.pointer) {
      return { buffer: `${storageId}u`, base: callbacks.pointerBaseExpression(expression.name, context) ?? "0u" };
    }
    const sharedId = context.sharedPointerIdFor(expression.name);
    const shared = callbacks.sharedDeclarationFor(expression.name, context);
    if (sharedId !== undefined && shared && shared.dimensions.length > 0) return { buffer: `${sharedId}u`, base: "0u" };
    const globalId = context.deviceGlobalPointerIdFor(expression.name);
    const global = context.deviceGlobalFor(expression.name);
    if (globalId !== undefined && global && global.dimensions.length > 0) return { buffer: `${globalId}u`, base: "0u" };
    const constantId = context.constantPointerIdFor(expression.name);
    const constant = context.ir.constants.find((item) => item.name === expression.name);
    if (constantId !== undefined && constant && constant.dimensions.length > 0) return { buffer: `${constantId}u`, base: "0u" };
    const alias = flattenedDevicePointerAlias(expression.name, expression.span, context);
    if (alias) {
      const target = devicePointerArgumentParts({
        kind: "identifier",
        name: alias.rootName,
        span: expression.span,
      }, context, callbacks);
      if (!target) return undefined;
      return {
        buffer: target.buffer,
        base: `(${target.base} + ${devicePointerAliasBaseDelta(alias, context, callbacks)})`,
      };
    }
  }
  if (expression.kind === "index" && expression.target.kind === "identifier") {
    const pointerArray = context.localPointerArrayFor(expression.target.name, expression.target.span);
    if (pointerArray) {
      if (context.localPointerArrayRootFor(pointerArray.name, expression.target.span)) return undefined;
      const index = `u32(${callbacks.emitExpression(expression.index, context)})`;
      return {
        buffer: `${context.nameFor(`${pointerArray.name}_buffer`)}[${index}]`,
        base: `${context.nameFor(`${pointerArray.name}_base`)}[${index}]`,
      };
    }
  }
  if (expression.kind === "cast" && expression.pointer) {
    return devicePointerArgumentParts(expression.expression, context, callbacks);
  }
  if (expression.kind === "unary" && expression.operator === "&" && expression.argument.kind === "member") {
    const vectorType = devicePointerValueTypeForExpression(expression.argument.object, context);
    if (isCudaVectorType(vectorType)) {
      const fieldIndex = cudaVectorFieldIndex(vectorType, expression.argument.property);
      const target = devicePointerArgumentParts({
        kind: "unary",
        operator: "&",
        argument: expression.argument.object,
        span: expression.argument.object.span,
      }, context, callbacks);
      if (fieldIndex !== undefined && target) {
        return {
          buffer: target.buffer,
          base: `(${target.base} + ${fieldIndex}u)`,
        };
      }
    }
  }
  if (expression.kind === "unary" && expression.operator === "&" && expression.argument.kind === "index") {
    const argumentRoot = rootIdentifier(expression.argument.target);
    const pointerParam = argumentRoot ? context.devicePointerParamFor(argumentRoot) : undefined;
    if (pointerParam) {
      const target = devicePointerArgumentParts(expression.argument.target, context, callbacks);
      if (!target) return undefined;
      return {
        buffer: target.buffer,
        base: `(${target.base} + ${devicePointerIndexDelta(expression.argument.index, pointerParam.valueType, context, callbacks)})`,
      };
    }
    const global = deviceGlobalPointerArgumentParts(expression.argument, context, callbacks);
    if (global) return global;
    const shared = sharedPointerArgumentParts(expression.argument, context, callbacks);
    if (shared) return shared;
    const constant = constantPointerArgumentParts(expression.argument, context, callbacks);
    if (constant) return constant;
    const target = devicePointerArgumentParts(expression.argument.target, context, callbacks);
    if (!target) return undefined;
    const valueType = devicePointerValueTypeForExpression(expression.argument.target, context);
    const addressDelta = dynamicSharedVectorAliasAddressDelta(expression.argument.target, expression.argument.index, valueType, context, callbacks) ??
      devicePointerIndexDelta(expression.argument.index, valueType, context, callbacks);
    return {
      buffer: target.buffer,
      base: `(${target.base} + ${addressDelta})`,
    };
  }
  if (expression.kind === "unary" && expression.operator === "&" && expression.argument.kind === "identifier") {
    const pointerParam = context.devicePointerParamFor(expression.argument.name);
    if (pointerParam) {
      return {
        buffer: context.nameFor(`${pointerParam.name}_buffer`),
        base: context.nameFor(`${pointerParam.name}_base`),
      };
    }
    const sharedId = context.sharedPointerIdFor(expression.argument.name);
    if (sharedId !== undefined) return { buffer: `${sharedId}u`, base: "0u" };
    const globalId = context.deviceGlobalPointerIdFor(expression.argument.name);
    if (globalId !== undefined) return { buffer: `${globalId}u`, base: "0u" };
    const constantId = context.constantPointerIdFor(expression.argument.name);
    if (constantId !== undefined) return { buffer: `${constantId}u`, base: "0u" };
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const target = devicePointerArgumentParts(expression.left, context, callbacks);
    if (!target) return undefined;
    const delta = devicePointerIndexDelta(expression.right, devicePointerValueTypeForExpression(expression.left, context), context, callbacks);
    return {
      buffer: target.buffer,
      base: expression.operator === "+"
        ? `(${target.base} + ${delta})`
        : `(${target.base} - ${delta})`,
    };
  }
  return undefined;
}

function flattenedDevicePointerAlias(
  name: string,
  span: SourceSpan | undefined,
  context: WgslDevicePointerContext,
  seen: ReadonlySet<string> = new Set(),
): PointerAlias | undefined {
  if (seen.has(name)) return undefined;
  const alias = context.pointerAliasFor(name, span);
  if (!alias) return undefined;
  const parent = flattenedDevicePointerAlias(alias.rootName, span, context, new Set([...seen, name]));
  if (!parent) return alias;
  const valueType = alias.valueType ?? parent.valueType;
  return {
    rootName: parent.rootName,
    ...(valueType === undefined ? {} : { valueType }),
    baseIndex: {
      kind: "binary",
      operator: "+",
      left: parent.baseIndex,
      right: scaleDevicePointerAliasOffset(alias.baseIndex, parent.valueType, span ?? alias.baseIndex.span),
      span: span ?? alias.baseIndex.span,
    },
    ...(alias.declarationSpan === undefined ? {} : { declarationSpan: alias.declarationSpan }),
  };
}

function scaleDevicePointerAliasOffset(
  offset: CudaLiteExpression,
  pointerValueType: CudaLiteScalarType | undefined,
  span: SourceSpan,
): CudaLiteExpression {
  const lanes = isCudaVectorType(pointerValueType) ? cudaVectorLaneCount(pointerValueType) : 1;
  if (lanes <= 1) return offset;
  return {
    kind: "binary",
    operator: "*",
    left: offset,
    right: { kind: "number", value: lanes, raw: String(lanes), span },
    span,
  };
}

function devicePointerIndexDelta(
  index: CudaLiteExpression,
  valueType: CudaLiteParam["valueType"] | PointerAlias["valueType"] | undefined,
  context: WgslDevicePointerContext,
  callbacks: WgslDevicePointerCallbacks,
): string {
  const raw = `u32(${callbacks.emitExpression(index, context)})`;
  const lanes = isCudaVectorType(valueType) ? cudaVectorLaneCount(valueType) : 1;
  return lanes <= 1 ? raw : `(${raw} * ${lanes}u)`;
}

function dynamicSharedVectorAliasAddressDelta(
  pointer: CudaLiteExpression,
  index: CudaLiteExpression,
  valueType: CudaLiteScalarType | undefined,
  context: WgslDevicePointerContext,
  callbacks: WgslDevicePointerCallbacks,
): string | undefined {
  if (!isCudaVectorType(valueType) || pointer.kind !== "identifier") return undefined;
  const alias = flattenedDevicePointerAlias(pointer.name, pointer.span, context);
  if (!alias || !isCudaVectorType(alias.valueType)) return undefined;
  const rootType = pointerAliasRootValueType(alias.rootName, context);
  if (isCudaVectorType(rootType) || context.sharedPointerIdFor(alias.rootName) === undefined) return undefined;
  return `u32(${callbacks.emitExpression(index, context)})`;
}

function devicePointerAliasBaseDelta(
  alias: PointerAlias,
  context: WgslDevicePointerContext,
  callbacks: WgslDevicePointerCallbacks,
): string {
  const raw = `u32(${callbacks.emitExpression(alias.baseIndex, context)})`;
  const rootType = pointerAliasRootValueType(alias.rootName, context) ?? alias.valueType;
  const rootBytes = pointerValueByteSize(rootType);
  const aliasBytes = pointerValueByteSize(alias.valueType);
  if (rootBytes === aliasBytes) return raw;
  if (rootBytes < aliasBytes && aliasBytes % rootBytes === 0) return `(${raw} / ${aliasBytes / rootBytes}u)`;
  if (rootBytes > aliasBytes && rootBytes % aliasBytes === 0) return `(${raw} * ${rootBytes / aliasBytes}u)`;
  return raw;
}

function pointerAliasRootValueType(
  rootName: string,
  context: WgslDevicePointerContext,
): CudaLiteScalarType | undefined {
  const param = context.paramFor(rootName);
  if (param) return param.valueType;
  if (context.sharedPointerIdFor(rootName) !== undefined) {
    const shared = context.ir.sharedDeclarations.find((item) => item.name === rootName);
    if (shared) return shared.valueType;
  }
  const global = context.deviceGlobalFor(rootName);
  if (global) return global.valueType;
  const constant = context.ir.constants.find((item) => item.name === rootName);
  if (constant) return constant.valueType;
  return context.localPointerArrayRootFor(rootName)?.valueType ??
    context.localPointerArrayFor(rootName)?.valueType ??
    context.localPointerHandleFor(rootName)?.valueType;
}

function pointerValueByteSize(valueType: CudaLiteScalarType | undefined): number {
  if (!valueType) return 4;
  if (isCudaVectorType(valueType)) return cudaVectorLaneCount(valueType) * pointerValueByteSize(cudaVectorScalarType(valueType));
  if (valueType === "double" || valueType === "complex64") return 8;
  return 4;
}

function devicePointerValueTypeForExpression(
  expression: CudaLiteExpression,
  context: WgslDevicePointerContext,
): CudaLiteScalarType | undefined {
  if (expression.kind === "unary" && expression.operator === "&" && expression.argument.kind === "member") {
    const vectorType = devicePointerValueTypeForExpression(expression.argument.object, context);
    return isCudaVectorType(vectorType) ? cudaVectorScalarType(vectorType) : undefined;
  }
  if (expression.kind === "cast" && expression.pointer) return expression.valueType;
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    return expression.args[0] ? devicePointerValueTypeForExpression(expression.args[0], context) : undefined;
  }
  const root = rootIdentifier(expression);
  if (!root) return undefined;
  return context.localPointerHandleFor(root)?.valueType ??
    context.localPointerArrayFor(root, expression.span)?.valueType ??
    context.pointerAliasFor(root, expression.span)?.valueType ??
    context.devicePointerParamFor(root)?.valueType ??
    context.paramFor(root)?.valueType ??
    context.ir.sharedDeclarations.find((shared) => shared.name === root)?.valueType ??
    context.deviceGlobalFor(root)?.valueType;
}

function constantPointerArgumentParts(
  expression: CudaLiteExpression,
  context: WgslDevicePointerContext,
  callbacks: WgslDevicePointerCallbacks,
): DevicePointerParts | undefined {
  const indexes: CudaLiteExpression[] = [];
  let cursor = expression;
  while (cursor.kind === "index") {
    indexes.unshift(cursor.index);
    cursor = cursor.target;
  }
  if (cursor.kind !== "identifier") return undefined;
  const id = context.constantPointerIdFor(cursor.name);
  if (id === undefined) return undefined;
  const constant = context.ir.constants.find((item) => item.name === cursor.name);
  if (!constant) return undefined;
  if (indexes.length === 0) return { buffer: `${id}u`, base: "0u" };
  const dimensions = constant.dimensions.length === 0 ? [1] : constant.dimensions;
  const terms = indexes.map((index, axis) => {
    const stride = dimensions.slice(axis + 1).reduce((product, dimension) => product * dimension, 1);
    const value = `u32(${callbacks.emitExpression(index, context)})`;
    return stride === 1 ? value : `(${value} * ${stride}u)`;
  });
  return { buffer: `${id}u`, base: terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})` };
}

function sharedPointerArgumentParts(
  expression: CudaLiteExpression,
  context: WgslDevicePointerContext,
  callbacks: WgslDevicePointerCallbacks,
): DevicePointerParts | undefined {
  const indexes: CudaLiteExpression[] = [];
  let cursor = expression;
  while (cursor.kind === "index") {
    indexes.unshift(cursor.index);
    cursor = cursor.target;
  }
  if (cursor.kind !== "identifier") return undefined;
  const id = context.sharedPointerIdFor(cursor.name);
  if (id === undefined) return undefined;
  const declaration = context.ir.sharedDeclarations.find((item) => item.name === cursor.name);
  if (!declaration) return undefined;
  if (indexes.length === 0) return { buffer: `${id}u`, base: "0u" };
  const terms = indexes.map((index, axis) => {
    const stride = declaration.dimensions.slice(axis + 1).reduce((product, dimension) => product * dimension, 1);
    const value = `u32(${callbacks.emitExpression(index, context)})`;
    return stride === 1 ? value : `(${value} * ${stride}u)`;
  });
  return { buffer: `${id}u`, base: terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})` };
}

function deviceGlobalPointerArgumentParts(
  expression: CudaLiteExpression,
  context: WgslDevicePointerContext,
  callbacks: WgslDevicePointerCallbacks,
): DevicePointerParts | undefined {
  const indexes: CudaLiteExpression[] = [];
  let cursor = expression;
  while (cursor.kind === "index") {
    indexes.unshift(cursor.index);
    cursor = cursor.target;
  }
  if (cursor.kind !== "identifier") return undefined;
  const id = context.deviceGlobalPointerIdFor(cursor.name);
  if (id === undefined) return undefined;
  const global = context.deviceGlobalFor(cursor.name);
  if (!global) return undefined;
  if (indexes.length === 0) return { buffer: `${id}u`, base: "0u" };
  const dimensions = global.dimensions.length === 0 ? [1] : global.dimensions;
  const terms = indexes.map((index, axis) => {
    const stride = dimensions.slice(axis + 1).reduce((product, dimension) => product * dimension, 1);
    const value = `u32(${callbacks.emitExpression(index, context)})`;
    return stride === 1 ? value : `(${value} * ${stride}u)`;
  });
  return { buffer: `${id}u`, base: terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})` };
}
