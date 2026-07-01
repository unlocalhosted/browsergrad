import { rootIdentifier } from "./analyzer.js";
import { cudaVectorScalarType, isCudaVectorType } from "./vector_types.js";
import {
  floatAtomicHelperName,
  integerAtomicLoopHelperName,
} from "./wgsl_atomic_helpers.js";
import {
  isDevicePointerAtomicAddType,
  isDevicePointerAtomicBitwiseType,
  isDevicePointerAtomicCasType,
  isDevicePointerAtomicExchangeType,
  isDevicePointerAtomicIncDecType,
  isDevicePointerAtomicMinMaxType,
  isDevicePointerAtomicSubType,
  pointerAtomicAddHelperName,
  pointerAtomicCasHelperName,
  pointerAtomicExchangeHelperName,
  pointerAtomicIncDecHelperName,
  pointerAtomicRmwHelperName,
} from "./wgsl_pointer_helpers.js";
import { emitSharedFlatAccess, wgslElementByteSize, type StorageView } from "./wgsl_storage.js";
import { isAbstractIntegerLiteral } from "./wgsl_value_conversion.js";
import type { DevicePointerParts } from "./wgsl_device_pointers.js";
import type { PointerAlias } from "./wgsl_ir_analysis.js";
import type {
  CudaLiteCallExpression,
  CudaLiteDeviceGlobal,
  CudaLiteExpression,
  CudaLiteParam,
  CudaLiteScalarType,
  CudaLiteVarDecl,
  KernelIrModule,
  SourceSpan,
} from "./types.js";

type EmitMode = "value" | "lvalue";

interface AtomicTargetInfo {
  readonly address: string;
  readonly rootName: string;
  readonly valueType: CudaLiteScalarType;
  readonly storageValueType: CudaLiteScalarType;
  readonly storageScalar: "i32" | "u32";
  readonly addressSpace: "storage" | "workgroup";
}

export interface WgslAtomicContext {
  readonly ir: KernelIrModule;
  nameFor(name: string): string;
  paramFor(name: string): CudaLiteParam | undefined;
  deviceGlobalFor(name: string): CudaLiteDeviceGlobal | undefined;
  devicePointerParamFor(name: string): CudaLiteParam | undefined;
  pointerAliasFor(name: string, span?: SourceSpan): PointerAlias | undefined;
  isAtomicShared(name: string): boolean;
  isAtomicDeviceGlobal(name: string): boolean;
}

export interface WgslAtomicCallbacks {
  emitExpression(expression: CudaLiteExpression, context: WgslAtomicContext, mode?: EmitMode): string;
  emitExpressionAsValueType(expression: CudaLiteExpression, valueType: CudaLiteScalarType, context: WgslAtomicContext): string;
  emitExpressionAsWgslScalar(expression: CudaLiteExpression, type: "u32", context: WgslAtomicContext): string;
  devicePointerArgumentParts(expression: CudaLiteExpression, context: WgslAtomicContext): DevicePointerParts | undefined;
  devicePointerValueTypeForExpression(expression: CudaLiteExpression, context: WgslAtomicContext): CudaLiteScalarType;
  emitPointerIndex(rootName: string, index: CudaLiteExpression, context: WgslAtomicContext): string;
  emitPointerAliasIndex(alias: PointerAlias, index: CudaLiteExpression, context: WgslAtomicContext): string;
  storageViewForPointerExpression(
    expression: CudaLiteExpression,
    index: CudaLiteExpression,
    context: WgslAtomicContext,
  ): StorageView | undefined;
  sharedDeclarationFor(name: string, context: WgslAtomicContext): CudaLiteVarDecl | undefined;
  flattenedPointerAlias(name: string, span: CudaLiteExpression["span"], context: WgslAtomicContext): PointerAlias | undefined;
}

export function emitAtomicCall(
  wgslName: string,
  expression: CudaLiteCallExpression,
  context: WgslAtomicContext,
  args: readonly string[],
  callbacks: WgslAtomicCallbacks,
): string {
  const target = expression.args[0];
  const value = expression.args[1];
  const atomicTarget = emitAtomicTarget(target, context, callbacks);
  if (atomicTarget && value) {
    const valueExpression = callbacks.emitExpression(value, context);
    const integerValueExpression = atomicTarget.valueType === "int" || atomicTarget.valueType === "uint"
      ? emitAtomicIntegerValueExpression(value, atomicTarget.valueType, context, callbacks)
      : valueExpression;
    if (wgslName === "atomicInc") {
      return `${integerAtomicLoopHelperName("Inc", atomicTarget)}(${atomicTarget.address}, u32(${valueExpression}))`;
    }
    if (wgslName === "atomicDec") {
      return `${integerAtomicLoopHelperName("Dec", atomicTarget)}(${atomicTarget.address}, u32(${valueExpression}))`;
    }
    if (wgslName === "atomicAdd" && (atomicTarget.valueType === "float" || atomicTarget.valueType === "double")) {
      return `${floatAtomicHelperName("Add", atomicTarget.addressSpace)}(${atomicTarget.address}, ${valueExpression})`;
    }
    if (wgslName === "atomicSub" && (atomicTarget.valueType === "float" || atomicTarget.valueType === "double")) {
      return `${floatAtomicHelperName("Sub", atomicTarget.addressSpace)}(${atomicTarget.address}, ${valueExpression})`;
    }
    if (wgslName === "atomicMin" && (atomicTarget.valueType === "float" || atomicTarget.valueType === "double")) {
      return `${floatAtomicHelperName("Min", atomicTarget.addressSpace)}(${atomicTarget.address}, ${valueExpression})`;
    }
    if (wgslName === "atomicMax" && (atomicTarget.valueType === "float" || atomicTarget.valueType === "double")) {
      return `${floatAtomicHelperName("Max", atomicTarget.addressSpace)}(${atomicTarget.address}, ${valueExpression})`;
    }
    if (wgslName === "atomicExchange" && (atomicTarget.valueType === "float" || atomicTarget.valueType === "double")) {
      return `bitcast<f32>(atomicExchange(${atomicTarget.address}, bitcast<u32>(${valueExpression})))`;
    }
    return `${wgslName}(${atomicTarget.address}, ${integerValueExpression})`;
  }
  const pointerAtomic = emitDevicePointerAtomicCall(wgslName, target, value, context, callbacks);
  if (pointerAtomic) return pointerAtomic;
  return `${wgslName}(${args.join(", ")})`;
}

export function emitAtomicCasCall(
  expression: CudaLiteCallExpression,
  context: WgslAtomicContext,
  args: readonly string[],
  callbacks: WgslAtomicCallbacks,
): string {
  const target = expression.args[0];
  const compare = expression.args[1];
  const value = expression.args[2];
  const atomicTarget = emitAtomicTarget(target, context, callbacks);
  if (atomicTarget && compare && value) {
    const compareExpression = atomicTarget.valueType === "int" || atomicTarget.valueType === "uint"
      ? emitAtomicIntegerValueExpression(compare, atomicTarget.valueType, context, callbacks)
      : callbacks.emitExpression(compare, context);
    const valueExpression = atomicTarget.valueType === "int" || atomicTarget.valueType === "uint"
      ? emitAtomicIntegerValueExpression(value, atomicTarget.valueType, context, callbacks)
      : callbacks.emitExpression(value, context);
    return `atomicCompareExchangeWeak(${atomicTarget.address}, ${compareExpression}, ${valueExpression}).old_value`;
  }
  const pointerAtomic = emitDevicePointerAtomicCasCall(target, compare, value, context, callbacks);
  if (pointerAtomic) return pointerAtomic;
  return `atomicCompareExchangeWeak(${args.join(", ")}).old_value`;
}

function emitDevicePointerAtomicCall(
  wgslName: string,
  target: CudaLiteExpression | undefined,
  value: CudaLiteExpression | undefined,
  context: WgslAtomicContext,
  callbacks: WgslAtomicCallbacks,
): string | undefined {
  if (!target || !value) return undefined;
  const parts = callbacks.devicePointerArgumentParts(target, context);
  if (!parts) return undefined;
  const valueType = callbacks.devicePointerValueTypeForExpression(target, context);
  const valueExpression = callbacks.emitExpressionAsValueType(value, valueType, context);
  if (wgslName === "atomicAdd" && isDevicePointerAtomicAddType(valueType)) {
    return `${pointerAtomicAddHelperName(valueType)}(${parts.buffer}, ${parts.base}, ${valueExpression})`;
  }
  if (wgslName === "atomicSub" && isDevicePointerAtomicSubType(valueType)) {
    return `${pointerAtomicRmwHelperName("Sub", valueType)}(${parts.buffer}, ${parts.base}, ${valueExpression})`;
  }
  if (wgslName === "atomicMin" && isDevicePointerAtomicMinMaxType(valueType)) {
    return `${pointerAtomicRmwHelperName("Min", valueType)}(${parts.buffer}, ${parts.base}, ${valueExpression})`;
  }
  if (wgslName === "atomicMax" && isDevicePointerAtomicMinMaxType(valueType)) {
    return `${pointerAtomicRmwHelperName("Max", valueType)}(${parts.buffer}, ${parts.base}, ${valueExpression})`;
  }
  if (wgslName === "atomicAnd" && isDevicePointerAtomicBitwiseType(valueType)) {
    return `${pointerAtomicRmwHelperName("And", valueType)}(${parts.buffer}, ${parts.base}, ${valueExpression})`;
  }
  if (wgslName === "atomicOr" && isDevicePointerAtomicBitwiseType(valueType)) {
    return `${pointerAtomicRmwHelperName("Or", valueType)}(${parts.buffer}, ${parts.base}, ${valueExpression})`;
  }
  if (wgslName === "atomicXor" && isDevicePointerAtomicBitwiseType(valueType)) {
    return `${pointerAtomicRmwHelperName("Xor", valueType)}(${parts.buffer}, ${parts.base}, ${valueExpression})`;
  }
  if (wgslName === "atomicInc" && isDevicePointerAtomicIncDecType(valueType)) {
    return `${pointerAtomicIncDecHelperName("Inc", valueType)}(${parts.buffer}, ${parts.base}, ${callbacks.emitExpressionAsValueType(value, "uint", context)})`;
  }
  if (wgslName === "atomicDec" && isDevicePointerAtomicIncDecType(valueType)) {
    return `${pointerAtomicIncDecHelperName("Dec", valueType)}(${parts.buffer}, ${parts.base}, ${callbacks.emitExpressionAsValueType(value, "uint", context)})`;
  }
  if (wgslName === "atomicExchange" && isDevicePointerAtomicExchangeType(valueType)) {
    return `${pointerAtomicExchangeHelperName(valueType)}(${parts.buffer}, ${parts.base}, ${valueExpression})`;
  }
  return undefined;
}

function emitAtomicIntegerValueExpression(
  expression: CudaLiteExpression,
  valueType: "int" | "uint",
  context: WgslAtomicContext,
  callbacks: WgslAtomicCallbacks,
): string {
  return isAbstractIntegerLiteral(expression)
    ? callbacks.emitExpression(expression, context)
    : callbacks.emitExpressionAsValueType(expression, valueType, context);
}

function emitDevicePointerAtomicCasCall(
  target: CudaLiteExpression | undefined,
  compare: CudaLiteExpression | undefined,
  value: CudaLiteExpression | undefined,
  context: WgslAtomicContext,
  callbacks: WgslAtomicCallbacks,
): string | undefined {
  if (!target || !compare || !value) return undefined;
  const parts = callbacks.devicePointerArgumentParts(target, context);
  if (!parts) return undefined;
  const valueType = callbacks.devicePointerValueTypeForExpression(target, context);
  if (!isDevicePointerAtomicCasType(valueType)) return undefined;
  return `${pointerAtomicCasHelperName(valueType)}(${parts.buffer}, ${parts.base}, ${callbacks.emitExpressionAsValueType(compare, valueType, context)}, ${callbacks.emitExpressionAsValueType(value, valueType, context)})`;
}

function emitAtomicTarget(
  target: CudaLiteExpression | undefined,
  context: WgslAtomicContext,
  callbacks: WgslAtomicCallbacks,
): AtomicTargetInfo | undefined {
  if (!target) return undefined;
  if (target.kind === "cast" && target.pointer) {
    const vectorStorage = emitAtomicVectorStorageViewCastTarget(target, context, callbacks);
    if (vectorStorage) return vectorStorage;
    const inner = emitAtomicTarget(target.expression, context, callbacks);
    return inner ? { ...inner, valueType: target.valueType } : undefined;
  }
  if (target.kind === "unary" && target.operator === "&") {
    const addressRoot = rootIdentifier(target.argument);
    if (addressRoot && context.devicePointerParamFor(addressRoot)) return undefined;
    return emitAtomicAddressTarget(target.argument, context, callbacks);
  }
  if (target.kind === "identifier") {
    const alias = callbacks.flattenedPointerAlias(target.name, target.span, context);
    if (alias) {
      const rootName = resolveAtomicRootName(alias.rootName, context);
      const info = atomicStorageInfo(rootName, context);
      if (!info) return undefined;
      const index = callbacks.emitPointerAliasIndex(alias, zeroExpression(target.span), context);
      return {
        address: emitAtomicRootAddress(rootName, index, context, callbacks),
        rootName,
        valueType: alias.valueType ?? info.valueType,
        storageValueType: info.valueType,
        storageScalar: info.storageScalar,
        addressSpace: info.addressSpace,
      };
    }
    const param = context.paramFor(target.name);
    if (param?.pointer) {
      const info = atomicStorageInfo(target.name, context);
      if (!info) return undefined;
      const index = callbacks.emitPointerIndex(target.name, zeroExpression(target.span), context);
      return {
        address: emitAtomicRootAddress(target.name, index, context, callbacks),
        rootName: target.name,
        valueType: param.valueType,
        storageValueType: info.valueType,
        storageScalar: info.storageScalar,
        addressSpace: info.addressSpace,
      };
    }
  }
  const pointerParts = callbacks.devicePointerArgumentParts(target, context);
  const root = rootIdentifier(target);
  if (root && context.devicePointerParamFor(root)) return undefined;
  const rootName = root ? resolveAtomicRootName(root, context) : undefined;
  const info = rootName ? atomicStorageInfo(rootName, context) : undefined;
  if (pointerParts && rootName && info) {
    return {
      address: emitAtomicRootAddress(rootName, pointerParts.base, context, callbacks),
      rootName,
      valueType: atomicExpressionValueType(target, rootName, context) ?? info.valueType,
      storageValueType: info.valueType,
      storageScalar: info.storageScalar,
      addressSpace: info.addressSpace,
    };
  }
  return undefined;
}

function emitAtomicRootAddress(
  rootName: string,
  index: string,
  context: WgslAtomicContext,
  callbacks: WgslAtomicCallbacks,
): string {
  const shared = callbacks.sharedDeclarationFor(rootName, context);
  if (shared) return `&${emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, index)}`;
  return `&${context.nameFor(rootName)}[${index}]`;
}

function emitAtomicVectorStorageViewCastTarget(
  target: Extract<CudaLiteExpression, { kind: "cast" }>,
  context: WgslAtomicContext,
  callbacks: WgslAtomicCallbacks,
): AtomicTargetInfo | undefined {
  if (!target.pointer || target.expression.kind !== "unary" || target.expression.operator !== "&") return undefined;
  if (target.valueType !== "uint" && target.valueType !== "int") return undefined;
  const address = target.expression.argument;
  if (address.kind !== "index") return undefined;
  const view = callbacks.storageViewForPointerExpression(address.target, address.index, context);
  if (!view || !isCudaVectorType(view.valueType)) return undefined;
  const param = context.paramFor(view.rootName);
  if (param?.pointer && context.ir.atomicParams.includes(param.name) && wgslElementByteSize(view.valueType) === wgslElementByteSize(param.valueType)) {
    return {
      address: `&${context.nameFor(param.name)}[${view.index}]`,
      rootName: param.name,
      valueType: target.valueType,
      storageValueType: param.valueType,
      storageScalar: atomicStorageScalar(param.valueType),
      addressSpace: "storage",
    };
  }
  const shared = callbacks.sharedDeclarationFor(view.rootName, context);
  if (shared && context.isAtomicShared(shared.name) && wgslElementByteSize(view.valueType) === wgslElementByteSize(shared.valueType)) {
    return {
      address: `&${emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, view.index)}`,
      rootName: shared.name,
      valueType: target.valueType,
      storageValueType: shared.valueType,
      storageScalar: atomicStorageScalar(shared.valueType),
      addressSpace: "workgroup",
    };
  }
  const global = context.deviceGlobalFor(view.rootName);
  if (global && context.ir.atomicDeviceGlobals.includes(global.name) && wgslElementByteSize(view.valueType) === wgslElementByteSize(global.valueType)) {
    return {
      address: `&${context.nameFor(global.name)}[${view.index}]`,
      rootName: global.name,
      valueType: target.valueType,
      storageValueType: global.valueType,
      storageScalar: atomicStorageScalar(global.valueType),
      addressSpace: "storage",
    };
  }
  return undefined;
}

function emitAtomicAddressTarget(
  expression: CudaLiteExpression,
  context: WgslAtomicContext,
  callbacks: WgslAtomicCallbacks,
): AtomicTargetInfo | undefined {
  const storageViewTarget = emitAtomicStorageViewTarget(expression, context, callbacks);
  if (storageViewTarget) return storageViewTarget;
  const rootName = atomicExpressionRootName(expression, context);
  if (!rootName) return undefined;
  const info = atomicStorageInfo(rootName, context);
  if (!info) return undefined;
  return {
    address: `&${callbacks.emitExpression(expression, context, "lvalue")}`,
    rootName,
    valueType: atomicExpressionValueType(expression, rootName, context) ?? info.valueType,
    storageValueType: info.valueType,
    storageScalar: info.storageScalar,
    addressSpace: info.addressSpace,
  };
}

function emitAtomicStorageViewTarget(
  expression: CudaLiteExpression,
  context: WgslAtomicContext,
  callbacks: WgslAtomicCallbacks,
): AtomicTargetInfo | undefined {
  if (expression.kind !== "index") return undefined;
  const view = callbacks.storageViewForPointerExpression(expression.target, expression.index, context);
  if (!view || isCudaVectorType(view.valueType)) return undefined;
  const shared = callbacks.sharedDeclarationFor(view.rootName, context);
  if (shared && context.isAtomicShared(shared.name)) {
    return {
      address: `&${emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, view.index)}`,
      rootName: shared.name,
      valueType: view.valueType,
      storageValueType: shared.valueType,
      storageScalar: atomicStorageScalar(shared.valueType),
      addressSpace: "workgroup",
    };
  }
  const param = context.paramFor(view.rootName);
  if (param?.pointer && context.ir.atomicParams.includes(param.name)) {
    return {
      address: `&${context.nameFor(param.name)}[${view.index}]`,
      rootName: param.name,
      valueType: view.valueType,
      storageValueType: param.valueType,
      storageScalar: atomicStorageScalar(param.valueType),
      addressSpace: "storage",
    };
  }
  const global = context.deviceGlobalFor(view.rootName);
  if (global && context.ir.atomicDeviceGlobals.includes(global.name)) {
    return {
      address: `&${context.nameFor(global.name)}[${view.index}]`,
      rootName: global.name,
      valueType: view.valueType,
      storageValueType: global.valueType,
      storageScalar: atomicStorageScalar(global.valueType),
      addressSpace: "storage",
    };
  }
  return undefined;
}

function atomicExpressionRootName(expression: CudaLiteExpression, context: WgslAtomicContext): string | undefined {
  const root = rootIdentifier(expression);
  return root ? resolveAtomicRootName(root, context) : undefined;
}

function atomicExpressionValueType(
  expression: CudaLiteExpression,
  rootName: string,
  context: WgslAtomicContext,
): CudaLiteScalarType | undefined {
  const root = rootIdentifier(expression);
  if (root) {
    const alias = context.pointerAliasFor(root, expression.span);
    if (alias?.valueType) return alias.valueType;
    const direct = context.paramFor(root);
    if (direct?.pointer) return direct.valueType;
  }
  return atomicStorageInfo(rootName, context)?.valueType;
}

function resolveAtomicRootName(name: string, context: WgslAtomicContext, seen: ReadonlySet<string> = new Set()): string {
  if (seen.has(name)) return name;
  const alias = context.pointerAliasFor(name);
  if (!alias) return name;
  return resolveAtomicRootName(alias.rootName, context, new Set([...seen, name]));
}

function atomicStorageInfo(
  rootName: string,
  context: WgslAtomicContext,
): { readonly valueType: CudaLiteScalarType; readonly storageScalar: "i32" | "u32"; readonly addressSpace: "storage" | "workgroup" } | undefined {
  const param = context.paramFor(rootName);
  if (param?.pointer && context.ir.atomicParams.includes(rootName)) {
    return {
      valueType: param.valueType,
      storageScalar: atomicStorageScalar(param.valueType),
      addressSpace: "storage",
    };
  }
  if (context.isAtomicShared(rootName)) {
    const shared = context.ir.sharedDeclarations.find((item) => item.name === rootName);
    if (!shared) return undefined;
    return {
      valueType: shared.valueType,
      storageScalar: atomicStorageScalar(shared.valueType),
      addressSpace: "workgroup",
    };
  }
  if (context.isAtomicDeviceGlobal(rootName)) {
    const global = context.deviceGlobalFor(rootName);
    if (!global) return undefined;
    return {
      valueType: global.valueType,
      storageScalar: atomicStorageScalar(global.valueType),
      addressSpace: "storage",
    };
  }
  return undefined;
}

function atomicStorageScalar(valueType: CudaLiteScalarType): "i32" | "u32" {
  return (cudaVectorScalarType(valueType) ?? valueType) === "int" ? "i32" : "u32";
}

function zeroExpression(span: CudaLiteExpression["span"]): CudaLiteExpression {
  return { kind: "number", raw: "0", value: 0, span };
}
