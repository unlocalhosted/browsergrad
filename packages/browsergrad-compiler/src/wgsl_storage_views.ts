import { expressionName } from "./analyzer.js";
import { matrixTileStorageDimensions } from "./matrix_tiles.js";
import { cudaVectorFieldIndex, cudaVectorLaneCount, cudaVectorScalarType, isCudaVectorType } from "./vector_types.js";
import {
  flatLocalArrayIndexExpression,
  isPointerIdentityCall,
  pointerAliasForPointerExpression,
  zeroExpression,
  type PointerAlias,
} from "./wgsl_ir_analysis.js";
import { wgslElementByteSize, type StorageView } from "./wgsl_storage.js";
import type {
  CudaLiteDeviceGlobal,
  CudaLiteExpression,
  CudaLiteParam,
  CudaLiteScalarType,
  CudaLiteVarDecl,
  KernelIrModule,
  SourceSpan,
} from "./types.js";

export interface WgslStorageViewContext {
  readonly ir: KernelIrModule;
  nameFor(name: string): string;
  paramFor(name: string): CudaLiteParam | undefined;
  deviceGlobalFor(name: string): CudaLiteDeviceGlobal | undefined;
  devicePointerParamFor(name: string): CudaLiteParam | undefined;
  pointerAliasFor(name: string, span?: SourceSpan): PointerAlias | undefined;
  localArrayFor(name: string, span?: SourceSpan): CudaLiteVarDecl | undefined;
}

export interface WgslStorageViewCallbacks {
  emitExpression(expression: CudaLiteExpression, context: WgslStorageViewContext): string;
  emitExpressionAsWgslScalar(expression: CudaLiteExpression, type: "u32", context: WgslStorageViewContext): string;
  pointerBaseExpression(rootName: string, context: WgslStorageViewContext): string | undefined;
  sharedDeclarationFor(name: string, context: WgslStorageViewContext): CudaLiteVarDecl | undefined;
  devicePointerValueTypeForExpression(expression: CudaLiteExpression, context: WgslStorageViewContext): CudaLiteScalarType;
}

export interface VectorStorageLValue {
  readonly rootName?: string;
  readonly name: string;
  readonly valueType: CudaLiteScalarType;
  readonly index: string;
  readonly subElementLane?: string;
  readonly lanes: number;
  readonly field?: string;
  readonly fieldIndex?: number;
}

export type StorageViewRoot =
  | { readonly kind: "param"; readonly name: string; readonly valueType: CudaLiteScalarType }
  | { readonly kind: "shared"; readonly name: string; readonly valueType: CudaLiteScalarType }
  | { readonly kind: "local"; readonly name: string; readonly valueType: CudaLiteScalarType }
  | { readonly kind: "global"; readonly name: string; readonly valueType: CudaLiteScalarType };

export interface ResolvedStorageView extends StorageView {
  readonly root: StorageViewRoot;
  readonly rootValueType: CudaLiteScalarType;
  readonly rootBytes: number;
  readonly viewBytes: number;
  readonly flatScalarVectorStorage: boolean;
  readonly indexUnit: "storage-element" | "flat-scalar-lane";
}

export type StorageViewResolutionError =
  | { readonly reason: "unresolved-root"; readonly rootName: string }
  | { readonly reason: "device-pointer-root"; readonly rootName: string }
  | { readonly reason: "unresolved-pointer"; readonly pointerKind: CudaLiteExpression["kind"] }
  | { readonly reason: "missing-pointer-value-type"; readonly pointerName?: string };

export type StorageViewResolution =
  | { readonly ok: true; readonly view: ResolvedStorageView }
  | { readonly ok: false; readonly error: StorageViewResolutionError };

export interface ScalarStorageViewLValue {
  readonly root: CudaLiteVarDecl;
  readonly addressSpace: "local" | "shared";
  readonly valueType: CudaLiteScalarType;
  readonly index: string;
  readonly subElementLane?: string;
}

export type ScalarParamStorageViewLValue =
  | {
    readonly root: CudaLiteParam;
    readonly addressSpace: "param";
    readonly valueType: CudaLiteScalarType;
    readonly index: string;
  }
  | {
    readonly root: CudaLiteDeviceGlobal;
    readonly addressSpace: "global";
    readonly valueType: CudaLiteScalarType;
    readonly index: string;
  };

export function emitPointerAliasIndex(
  alias: PointerAlias,
  index: CudaLiteExpression,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): string {
  const base = callbacks.pointerBaseExpression(alias.rootName, context);
  const baseIndex = callbacks.emitExpressionAsWgslScalar(alias.baseIndex, "u32", context);
  const offset = callbacks.emitExpressionAsWgslScalar(index, "u32", context);
  if (!base) return `(${baseIndex}) + (${offset})`;
  return `(${base} + ${baseIndex} + ${offset})`;
}

export function createStorageView(
  rootName: string,
  baseIndex: CudaLiteExpression,
  index: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): StorageView {
  const resolved = resolveStorageView(rootName, baseIndex, index, valueType, context, callbacks);
  if (resolved.ok) return resolved.view;
  return unresolvedStorageView(rootName, baseIndex, index, valueType, context, callbacks);
}

export function resolveStorageView(
  rootName: string,
  baseIndex: CudaLiteExpression,
  index: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): StorageViewResolution {
  const root = storageViewRoot(rootName, context, callbacks);
  if (!root) return { ok: false, error: { reason: "unresolved-root", rootName } };
  const rootBytes = wgslElementByteSize(root.valueType);
  const viewBytes = wgslElementByteSize(valueType);
  const flatScalarVectorStorage = usesFlatScalarVectorStorage(rootName, context, callbacks);
  const subElementLane = emitStorageViewSubElementLane(rootName, index, valueType, context, callbacks);
  return {
    ok: true,
    view: {
      rootName,
      root,
      rootValueType: root.valueType,
      rootBytes,
      viewBytes,
      flatScalarVectorStorage,
      indexUnit: flatScalarVectorStorage && viewBytes < rootBytes ? "flat-scalar-lane" : "storage-element",
      valueType,
      index: emitStorageViewIndex(rootName, baseIndex, index, valueType, context, callbacks),
      ...(subElementLane === undefined ? {} : { subElementLane }),
    },
  };
}

export function emitPointerIndex(
  rootName: string,
  index: CudaLiteExpression,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): string {
  const base = callbacks.pointerBaseExpression(rootName, context);
  if (!base) return callbacks.emitExpression(index, context);
  return `(${base} + u32(${callbacks.emitExpression(index, context)}))`;
}

export function flattenedPointerAlias(
  name: string,
  span: CudaLiteExpression["span"],
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
  seen: ReadonlySet<string> = new Set(),
): PointerAlias | undefined {
  if (seen.has(name)) return undefined;
  const alias = context.pointerAliasFor(name);
  if (!alias) return undefined;
  return flattenPointerAlias(alias, span, context, callbacks, new Set([...seen, name]));
}

export function flattenPointerAlias(
  alias: PointerAlias,
  span: CudaLiteExpression["span"],
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
  seen: ReadonlySet<string> = new Set(),
): PointerAlias {
  const parent = flattenedPointerAlias(alias.rootName, span, context, callbacks, seen);
  if (!parent) return normalizePointerAliasForStorage(alias, context, callbacks, span);
  const valueType = alias.valueType ?? parent.valueType;
  const baseIndex = scalePointerOffsetForStorageRoot(alias.baseIndex, parent.valueType, parent.rootName, context, callbacks, span);
  return {
    rootName: parent.rootName,
    ...(valueType === undefined ? {} : { valueType }),
    baseIndex: {
      kind: "binary",
      operator: "+",
      left: parent.baseIndex,
      right: baseIndex,
      span,
    },
  };
}

function normalizePointerAliasForStorage(
  alias: PointerAlias,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
  span: CudaLiteExpression["span"],
): PointerAlias {
  if (!usesFlatScalarVectorStorage(alias.rootName, context, callbacks)) return alias;
  const rootType = storageViewRootValueType(alias.rootName, context, callbacks);
  if (!isCudaVectorType(rootType) || alias.valueType !== cudaVectorScalarType(rootType)) return alias;
  return {
    ...alias,
    baseIndex: scalePointerAliasBaseIndex(alias.baseIndex, rootType, span),
  };
}

function scalePointerOffsetForStorageRoot(
  offset: CudaLiteExpression,
  pointerValueType: CudaLiteScalarType | undefined,
  rootName: string,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
  span: CudaLiteExpression["span"],
): CudaLiteExpression {
  if (usesFlatScalarVectorStorage(rootName, context, callbacks)) {
    return scalePointerAliasBaseIndex(offset, pointerValueType, span);
  }
  return scalePointerAliasBaseIndex(offset, pointerValueType, span);
}

function scalePointerAliasBaseIndex(
  baseIndex: CudaLiteExpression,
  parentValueType: CudaLiteScalarType | undefined,
  span: CudaLiteExpression["span"],
): CudaLiteExpression {
  const lanes = isCudaVectorType(parentValueType) ? cudaVectorLaneCount(parentValueType) : 1;
  if (lanes <= 1) return baseIndex;
  return {
    kind: "binary",
    operator: "*",
    left: baseIndex,
    right: { kind: "number", value: lanes, raw: String(lanes), span },
    span,
  };
}

export function vectorStorageLValue(
  expression: CudaLiteExpression,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): VectorStorageLValue | undefined {
  let target = expression;
  let field: string | undefined;
  if (expression.kind === "member") {
    field = expression.property;
    target = expression.object;
  }
  if (target.kind !== "index" || target.target.kind !== "identifier") return undefined;
  const param = context.paramFor(target.target.name);
  if (!param || !isCudaVectorType(param.valueType)) return undefined;
  const index = emitPointerIndex(param.name, target.index, context, callbacks);
  const base = {
    rootName: param.name,
    name: context.nameFor(param.name),
    valueType: param.valueType,
    index,
    lanes: cudaVectorLaneCount(param.valueType),
  };
  if (!field) return base;
  const fieldIndex = cudaVectorFieldIndex(param.valueType, field);
  return fieldIndex === undefined ? undefined : { ...base, field, fieldIndex };
}

export function storageViewLValue(
  expression: CudaLiteExpression,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): VectorStorageLValue | undefined {
  let target = expression;
  let field: string | undefined;
  if (expression.kind === "member") {
    field = expression.property;
    target = expression.object;
  }
  const view = target.kind === "index"
    ? storageViewForPointerExpression(target.target, target.index, context, callbacks)
    : target.kind === "unary" && target.operator === "*"
      ? storageViewForPointerExpression(target.argument, zeroExpression(target.span), context, callbacks)
      : undefined;
  if (!view || !isCudaVectorType(view.valueType)) return undefined;
  const base = {
    rootName: view.rootName,
    name: context.nameFor(view.rootName),
    valueType: view.valueType,
    index: view.index,
    ...(view.subElementLane === undefined ? {} : { subElementLane: view.subElementLane }),
    lanes: cudaVectorLaneCount(view.valueType),
  };
  if (!field) return base;
  const fieldIndex = cudaVectorFieldIndex(view.valueType, field);
  return fieldIndex === undefined ? undefined : { ...base, field, fieldIndex };
}

export function scalarStorageViewLValue(
  expression: CudaLiteExpression,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): ScalarStorageViewLValue | undefined {
  if (expression.kind !== "index") return undefined;
  const view = storageViewForPointerExpression(expression.target, expression.index, context, callbacks);
  if (!view || isCudaVectorType(view.valueType)) return undefined;
  const shared = callbacks.sharedDeclarationFor(view.rootName, context);
  if (shared) {
    return {
      root: shared,
      addressSpace: "shared",
      valueType: view.valueType,
      index: view.index,
      ...(view.subElementLane === undefined ? {} : { subElementLane: view.subElementLane }),
    };
  }
  const local = localArrayForStorageView(view.rootName, expression.span, context);
  return local ? {
    root: local,
    addressSpace: "local",
    valueType: view.valueType,
    index: view.index,
    ...(view.subElementLane === undefined ? {} : { subElementLane: view.subElementLane }),
  } : undefined;
}

export function scalarParamStorageViewLValue(
  expression: CudaLiteExpression,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): ScalarParamStorageViewLValue | undefined {
  if (expression.kind !== "index" || expression.target.kind !== "identifier") return undefined;
  const alias = flattenedPointerAlias(expression.target.name, expression.target.span, context, callbacks);
  if (!alias?.valueType || isCudaVectorType(alias.valueType)) return undefined;
  const view = createStorageView(alias.rootName, alias.baseIndex, expression.index, alias.valueType, context, callbacks);
  const param = context.paramFor(alias.rootName);
  if (param?.pointer) {
    return {
      root: param,
      addressSpace: "param",
      valueType: view.valueType,
      index: view.index,
    };
  }
  const global = context.deviceGlobalFor(alias.rootName);
  return global ? {
    root: global,
    addressSpace: "global",
    valueType: view.valueType,
    index: view.index,
  } : undefined;
}

export function storageViewForPointerExpression(
  pointer: CudaLiteExpression,
  index: CudaLiteExpression,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): StorageView | undefined {
  const resolved = resolveStorageViewForPointerExpression(pointer, index, context, callbacks);
  return resolved.ok ? resolved.view : undefined;
}

export function resolveStorageViewForPointerExpression(
  pointer: CudaLiteExpression,
  index: CudaLiteExpression,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): StorageViewResolution {
  if (pointer.kind === "cast" && pointer.pointer) {
    const rawAlias = pointerAliasForContextPointerExpression(pointer.expression, pointer.valueType, context, callbacks) ??
      pointerAliasForPointerExpression(pointer.expression, pointer.valueType);
    const alias = rawAlias ? flattenPointerAlias(rawAlias, pointer.span, context, callbacks) : undefined;
    if (!alias) return { ok: false, error: { reason: "unresolved-pointer", pointerKind: pointer.kind } };
    if (context.devicePointerParamFor(alias.rootName)) return { ok: false, error: { reason: "device-pointer-root", rootName: alias.rootName } };
    return resolveStorageView(alias.rootName, alias.baseIndex, index, pointer.valueType, context, callbacks);
  }
  if (pointer.kind === "identifier") {
    const alias = flattenedPointerAlias(pointer.name, pointer.span, context, callbacks);
    if (!alias?.valueType) return { ok: false, error: { reason: "missing-pointer-value-type", pointerName: pointer.name } };
    if (context.devicePointerParamFor(alias.rootName)) return { ok: false, error: { reason: "device-pointer-root", rootName: alias.rootName } };
    return resolveStorageView(alias.rootName, alias.baseIndex, index, alias.valueType, context, callbacks);
  }
  const valueType = callbacks.devicePointerValueTypeForExpression(pointer, context);
  const rawAlias = pointerAliasForContextPointerExpression(pointer, valueType, context, callbacks) ??
    pointerAliasForPointerExpression(pointer, valueType);
  const alias = rawAlias ? flattenPointerAlias(rawAlias, pointer.span, context, callbacks) : undefined;
  if (alias?.valueType && !context.devicePointerParamFor(alias.rootName)) {
    return resolveStorageView(alias.rootName, alias.baseIndex, index, alias.valueType, context, callbacks);
  }
  return { ok: false, error: { reason: "unresolved-pointer", pointerKind: pointer.kind } };
}

export function localArrayForStorageView(
  name: string,
  span: SourceSpan | undefined,
  context: WgslStorageViewContext,
): CudaLiteVarDecl | undefined {
  return context.localArrayFor(name, span) ?? context.localArrayFor(name);
}

function emitStorageViewIndex(
  rootName: string,
  baseIndex: CudaLiteExpression,
  index: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): string {
  const base = callbacks.pointerBaseExpression(rootName, context);
  const prefix = base
    ? `${base} + u32(${callbacks.emitExpression(baseIndex, context)})`
    : `u32(${callbacks.emitExpression(baseIndex, context)})`;
  const offset = `u32(${callbacks.emitExpression(index, context)})`;
  const rootBytes = storageViewRootByteSize(rootName, context, callbacks);
  const viewBytes = wgslElementByteSize(valueType);
  if (usesFlatScalarVectorStorage(rootName, context, callbacks) && viewBytes < rootBytes) {
    return `(${prefix} + ${offset})`;
  }
  if (viewBytes === rootBytes) return `(${prefix} + ${offset})`;
  if (viewBytes > rootBytes && viewBytes % rootBytes === 0) {
    return `(${prefix} + (${offset} * ${viewBytes / rootBytes}u))`;
  }
  if (rootBytes > viewBytes && rootBytes % viewBytes === 0) {
    return `(${prefix} + (${offset} / ${rootBytes / viewBytes}u))`;
  }
  const lanes = cudaVectorLaneCount(valueType);
  return `(${prefix} + (${offset} * ${lanes}u))`;
}

function emitStorageViewSubElementLane(
  rootName: string,
  index: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): string | undefined {
  const rootBytes = storageViewRootByteSize(rootName, context, callbacks);
  const viewBytes = wgslElementByteSize(valueType);
  if (usesFlatScalarVectorStorage(rootName, context, callbacks)) return undefined;
  if (rootBytes <= viewBytes || rootBytes % viewBytes !== 0) return undefined;
  return `(u32(${callbacks.emitExpression(index, context)}) % ${rootBytes / viewBytes}u)`;
}

function usesFlatScalarVectorStorage(
  rootName: string,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): boolean {
  const rootType = storageViewRootValueType(rootName, context, callbacks);
  if (!isCudaVectorType(rootType)) return false;
  if (context.paramFor(rootName)?.pointer) return true;
  if (context.deviceGlobalFor(rootName)) return true;
  return context.ir.atomicShared.includes(rootName);
}

function storageViewRootByteSize(
  rootName: string,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): number {
  return wgslElementByteSize(storageViewRootValueType(rootName, context, callbacks) ?? "uint");
}

function storageViewRootValueType(
  rootName: string,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): CudaLiteScalarType | undefined {
  return storageViewRoot(rootName, context, callbacks)?.valueType;
}

function storageViewRoot(
  rootName: string,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): StorageViewRoot | undefined {
  const shared = callbacks.sharedDeclarationFor(rootName, context);
  if (shared) return { kind: "shared", name: rootName, valueType: shared.valueType };
  const local = context.localArrayFor(rootName);
  if (local) return { kind: "local", name: rootName, valueType: local.valueType };
  const global = context.deviceGlobalFor(rootName);
  if (global) return { kind: "global", name: rootName, valueType: global.valueType };
  const param = context.paramFor(rootName);
  if (param) return { kind: "param", name: rootName, valueType: param.valueType };
  return undefined;
}

function unresolvedStorageView(
  rootName: string,
  baseIndex: CudaLiteExpression,
  index: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): StorageView {
  const subElementLane = emitStorageViewSubElementLane(rootName, index, valueType, context, callbacks);
  return {
    rootName,
    valueType,
    index: emitStorageViewIndex(rootName, baseIndex, index, valueType, context, callbacks),
    ...(subElementLane === undefined ? {} : { subElementLane }),
  };
}

function pointerAliasForContextPointerExpression(
  expression: CudaLiteExpression | undefined,
  valueType: CudaLiteScalarType,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): PointerAlias | undefined {
  if (!expression) return undefined;
  if (expression.kind === "cast" && expression.pointer) return pointerAliasForContextPointerExpression(expression.expression, valueType, context, callbacks);
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    return pointerAliasForContextPointerExpression(expression.args[0], valueType, context, callbacks);
  }
  if (expression.kind === "unary" && expression.operator === "&") return arrayAddressAliasForContext(expression.argument, valueType, context, callbacks);
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const base = pointerAliasForContextPointerExpression(expression.left, valueType, context, callbacks);
    if (!base) return undefined;
    return {
      ...base,
      baseIndex: expression.operator === "+"
        ? { kind: "binary", operator: "+", left: base.baseIndex, right: expression.right, span: expression.span }
        : { kind: "binary", operator: "-", left: base.baseIndex, right: expression.right, span: expression.span },
    };
  }
  return undefined;
}

function arrayAddressAliasForContext(
  expression: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: WgslStorageViewContext,
  callbacks: WgslStorageViewCallbacks,
): PointerAlias | undefined {
  const indices: CudaLiteExpression[] = [];
  let cursor = expression;
  while (cursor.kind === "index") {
    indices.unshift(cursor.index);
    cursor = cursor.target;
  }
  if (cursor.kind !== "identifier") return undefined;
  const local = localArrayForStorageView(cursor.name, expression.span, context);
  const shared = callbacks.sharedDeclarationFor(cursor.name, context);
  const global = context.deviceGlobalFor(cursor.name);
  const dimensions = local
    ? matrixTileStorageDimensions(local)
    : shared
      ? shared.dimensions
      : global
        ? global.dimensions
        : undefined;
  if (!dimensions) return undefined;
  return {
    rootName: cursor.name,
    baseIndex: flatLocalArrayIndexExpression(indices, dimensions, expression.span),
    valueType,
  };
}
