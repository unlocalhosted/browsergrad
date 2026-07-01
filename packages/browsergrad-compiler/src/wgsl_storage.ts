import {
  CUDA_VECTOR_TYPES,
  cudaVectorFieldIndex,
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
} from "./vector_types.js";
import { matrixTileStorageDimensions } from "./matrix_tiles.js";
import { emitVectorLaneSetExpression } from "./wgsl_value_conversion.js";
import type {
  CudaLiteDeviceGlobal,
  CudaLiteGlobalConstant,
  CudaLiteParam,
  CudaLiteScalarType,
  CudaLiteVarDecl,
  KernelIrModule,
} from "./types.js";

export interface WgslStorageEmitContext {
  readonly ir: KernelIrModule;
  nameFor(name: string): string;
}

export interface StorageView {
  readonly rootName: string;
  readonly valueType: CudaLiteScalarType;
  readonly index: string;
  readonly subElementLane?: string;
}

export function zeroValue(type: CudaLiteScalarType): string {
  if (isCudaVectorType(type)) {
    return `${wgslScalar(type)}(${Array.from({ length: cudaVectorLaneCount(type) }, () => zeroValue(cudaVectorScalarType(type) ?? "float")).join(", ")})`;
  }
  if (type === "float") return "0.0";
  if (type === "half") return "f16(0.0)";
  if (type === "bf16") return "0.0";
  if (type === "uint" || type === "uchar") return "0u";
  if (type === "bool") return "false";
  if (type === "complex64") return "vec2<f32>(0.0, 0.0)";
  if (type === "texture2d" || type === "surface2d" || type === "devicepool" || type === "voidptr") return "0u";
  return "0";
}

export function wgslScalar(type: CudaLiteScalarType): string {
  if (isCudaVectorType(type)) {
    const scalar = cudaVectorScalarType(type) ?? "float";
    return `vec${cudaVectorLaneCount(type)}<${wgslScalar(scalar)}>`;
  }
  switch (type) {
    case "float":
    case "double":
      return "f32";
    case "int":
      return "i32";
    case "uint":
    case "uchar":
      return "u32";
    case "half":
      return "f16";
    case "bf16":
      return "f32";
    case "bool":
      return "bool";
    case "complex64":
      return "vec2<f32>";
    case "texture2d":
    case "surface2d":
    case "devicepool":
    case "voidptr":
      return "u32";
    case "void":
      return "void";
  }
}

export function cudaScalarWgslType(type: CudaLiteScalarType): "f32" | "f16" | "i32" | "u32" | "bool" | undefined {
  if (type === "float" || type === "double" || type === "bf16") return "f32";
  if (type === "half") return "f16";
  if (type === "int") return "i32";
  if (type === "uint" || type === "uchar") return "u32";
  if (type === "bool") return "bool";
  return undefined;
}

export function bitcastStorageViewType(from: CudaLiteScalarType, to: CudaLiteScalarType): "f32" | "i32" | "u32" | undefined {
  const source = cudaScalarWgslType(from);
  const target = cudaScalarWgslType(to);
  if (!source || !target || source === "bool" || target === "bool") return undefined;
  if ((source === "f32" || source === "i32" || source === "u32") &&
    (target === "f32" || target === "i32" || target === "u32")) {
    return target;
  }
  return undefined;
}

export function emitStorageCarrierAsU32(access: string, storageType: CudaLiteScalarType): string {
  const storageScalar = cudaScalarWgslType(storageType);
  if (storageScalar === "u32") return access;
  if (storageScalar === "i32" || storageScalar === "f32") return `bitcast<u32>(${access})`;
  return `u32(${access})`;
}

export function emitU32AsStorageCarrier(value: string, storageType: CudaLiteScalarType): string {
  const storageScalar = cudaScalarWgslType(storageType);
  if (storageScalar === "u32") return value;
  if (storageScalar === "i32" || storageScalar === "f32") return `bitcast<${storageScalar}>(${value})`;
  return `${wgslScalar(storageType)}(${value})`;
}

export function emitSharedFlatAccess(name: string, dimensions: readonly number[], index: string): string {
  if (dimensions.length === 0) return name;
  if (dimensions.length <= 1) return `${name}[${index}]`;
  return dimensions.reduce((expr, dimension, axis) => {
    const stride = dimensions.slice(axis + 1).reduce((product, item) => product * item, 1);
    const rawAxisIndex = stride === 1
      ? `(${index} % ${dimension}u)`
      : `(((${index}) / ${stride}u) % ${dimension}u)`;
    const axisIndex = dimension <= 1 ? "0u" : `min(${rawAxisIndex}, ${dimension - 1}u)`;
    return `${expr}[${axisIndex}]`;
  }, name);
}

export function vectorStorageBase(index: string, lanes: number): string {
  return `(u32(${index}) * ${lanes}u)`;
}

export function vectorFieldName(index: number): string {
  return index === 0 ? "x" : index === 1 ? "y" : index === 2 ? "z" : "w";
}

export function emitVectorStorageRead(name: string, type: CudaLiteScalarType, index: string): string {
  const lanes = cudaVectorLaneCount(type);
  const base = vectorStorageBase(index, lanes);
  const values = Array.from({ length: lanes }, (_, lane) => `${name}[${base} + ${lane}u]`);
  return `${wgslScalar(type)}(${values.join(", ")})`;
}

export function emitVectorStorageWrite(name: string, type: CudaLiteScalarType, index: string, value: string): string {
  const lanes = cudaVectorLaneCount(type);
  const base = vectorStorageBase(index, lanes);
  return Array.from({ length: lanes }, (_, lane) => `${name}[${base} + ${lane}u] = ${value}.${vectorFieldName(lane)}`).join("; ");
}

export function emitVectorStorageFieldWrite(name: string, type: CudaLiteScalarType, index: string, field: string, value: string): string | undefined {
  const fieldIndex = cudaVectorFieldIndex(type, field);
  if (fieldIndex === undefined) return undefined;
  const base = vectorStorageBase(index, cudaVectorLaneCount(type));
  return `${name}[${base} + ${fieldIndex}u] = ${value}`;
}

export function emitVectorStorageReadAt(name: string, type: CudaLiteScalarType, storageIndex: string): string {
  const lanes = cudaVectorLaneCount(type);
  const scalar = cudaVectorScalarType(type) ?? "float";
  const values = Array.from({ length: lanes }, (_, lane) => `${wgslScalar(scalar)}(${name}[${storageIndex} + ${lane}u])`);
  return `${wgslScalar(type)}(${values.join(", ")})`;
}

export function emitVectorStorageWriteAt(name: string, type: CudaLiteScalarType, storageIndex: string, value: string): string {
  const lanes = cudaVectorLaneCount(type);
  return Array.from({ length: lanes }, (_, lane) => `${name}[${storageIndex} + ${lane}u] = ${value}.${vectorFieldName(lane)}`).join("; ");
}

export function emitVectorStorageFieldWriteAt(name: string, storageIndex: string, fieldIndex: number, value: string): string {
  return `${name}[${storageIndex} + ${fieldIndex}u] = ${value}`;
}

export function emitVectorConstructor(vectorType: CudaLiteScalarType, args: readonly string[]): string {
  const info = CUDA_VECTOR_TYPES.get(vectorType as never);
  if (!info) return `${wgslScalar(vectorType)}(${args.join(", ")})`;
  if (args.length === 1) return `${wgslScalar(vectorType)}(${Array(info.lanes).fill(args[0] ?? "0").join(", ")})`;
  return `${wgslScalar(vectorType)}(${args.join(", ")})`;
}

export function castExpressionToVectorScalar(value: string, vectorType: CudaLiteScalarType): string {
  return `${wgslScalar(cudaVectorScalarType(vectorType) ?? "float")}(${value})`;
}

export function emitVectorSplat(vectorType: CudaLiteScalarType, value: string): string {
  return `${wgslScalar(vectorType)}(${Array.from({ length: cudaVectorLaneCount(vectorType) }, () => value).join(", ")})`;
}

export function wgslElementByteSize(valueType: CudaLiteScalarType): number {
  if (valueType === "uchar") return 1;
  if (valueType === "half") return 2;
  if (valueType === "bf16") return 2;
  if (isCudaVectorType(valueType)) return cudaVectorLaneCount(valueType) * wgslElementByteSize(cudaVectorScalarType(valueType) ?? "float");
  return 4;
}

export function emitPointerStorageRead(
  param: CudaLiteParam,
  index: string,
  ir: KernelIrModule,
  context: WgslStorageEmitContext,
  viewType: CudaLiteScalarType = param.valueType,
): string {
  const name = context.nameFor(param.name);
  if (param.valueType === "complex64" && (viewType === "complex64" || viewType === "float2")) {
    return `${name}[u32(${index})]`;
  }
  if (isCudaVectorType(viewType)) {
    if (ir.atomicParams.includes(param.name)) {
      return emitPointerVectorFlatRead(param, vectorStorageBase(index, cudaVectorLaneCount(viewType)), viewType, ir, context);
    }
    return param.valueType === viewType
      ? emitVectorStorageRead(name, viewType, index)
      : emitVectorStorageReadAt(name, viewType, index);
  }
  const access = `${name}[${index}]`;
  if (param.valueType === "bool") return `(${access} != 0u)`;
  if (ir.atomicParams.includes(param.name)) {
    const storageScalar = atomicStorageScalarType(param.valueType);
    const bitcastType = bitcastStorageViewType(storageScalar, viewType);
    const loaded = `atomicLoad(&${access})`;
    if (storageScalar === "float" || storageScalar === "double") {
      if (viewType === "uint") return loaded;
      if (viewType === "int") return `bitcast<i32>(${loaded})`;
      return `bitcast<f32>(${loaded})`;
    }
    return storageScalar !== viewType && bitcastType ? `bitcast<${bitcastType}>(${loaded})` : loaded;
  }
  const bitcastType = bitcastStorageViewType(param.valueType, viewType);
  return param.valueType !== viewType && bitcastType ? `bitcast<${bitcastType}>(${access})` : access;
}

export function emitPointerStorageWrite(
  param: CudaLiteParam,
  index: string,
  value: string,
  ir: KernelIrModule,
  context: WgslStorageEmitContext,
  viewType: CudaLiteScalarType = param.valueType,
): string {
  const name = context.nameFor(param.name);
  if (param.valueType === "complex64" && (viewType === "complex64" || viewType === "float2")) {
    return `${name}[u32(${index})] = ${value}`;
  }
  if (isCudaVectorType(viewType)) {
    if (ir.atomicParams.includes(param.name)) {
      return emitPointerVectorFlatWrite(param, vectorStorageBase(index, cudaVectorLaneCount(viewType)), value, viewType, ir, context);
    }
    return param.valueType === viewType
      ? emitVectorStorageWrite(name, viewType, index, value)
      : emitVectorStorageWriteAt(name, viewType, index, value);
  }
  const access = `${name}[${index}]`;
  if (param.valueType === "bool") return `${access} = select(0u, 1u, ${value})`;
  const bitcastType = bitcastStorageViewType(viewType, param.valueType);
  if (!ir.atomicParams.includes(param.name)) return param.valueType !== viewType && bitcastType
    ? `${access} = bitcast<${bitcastType}>(${value})`
    : `${access} = ${value}`;
  const storageScalar = atomicStorageScalarType(param.valueType);
  if (storageScalar === "float" || storageScalar === "double") {
    return `atomicStore(&${access}, ${emitStorageCarrierAsU32(value, viewType)})`;
  }
  const atomicBitcastType = bitcastStorageViewType(viewType, storageScalar);
  return storageScalar !== viewType && atomicBitcastType
    ? `atomicStore(&${access}, bitcast<${atomicBitcastType}>(${value}))`
    : `atomicStore(&${access}, ${value})`;
}

export function emitDeviceGlobalPointerRead(
  global: CudaLiteDeviceGlobal,
  index: string,
  ir: KernelIrModule,
  context: WgslStorageEmitContext,
  viewType: CudaLiteScalarType = global.valueType,
): string {
  const name = context.nameFor(global.name);
  if (isCudaVectorType(viewType)) {
    if (ir.atomicDeviceGlobals.includes(global.name)) {
      return emitDeviceGlobalVectorFlatRead(global, vectorStorageBase(index, cudaVectorLaneCount(viewType)), viewType, ir, context);
    }
    return global.valueType === viewType
      ? emitVectorStorageRead(name, viewType, index)
      : emitVectorStorageReadAt(name, viewType, index);
  }
  const access = `${name}[${index}]`;
  if (global.valueType === "bool") return `(${access} != 0u)`;
  if (ir.atomicDeviceGlobals.includes(global.name)) {
    const storageScalar = atomicStorageScalarType(global.valueType);
    const bitcastType = bitcastStorageViewType(storageScalar, viewType);
    const loaded = `atomicLoad(&${access})`;
    if (storageScalar === "float" || storageScalar === "double") {
      if (viewType === "uint") return loaded;
      if (viewType === "int") return `bitcast<i32>(${loaded})`;
      return `bitcast<f32>(${loaded})`;
    }
    return storageScalar !== viewType && bitcastType ? `bitcast<${bitcastType}>(${loaded})` : loaded;
  }
  const bitcastType = bitcastStorageViewType(global.valueType, viewType);
  return global.valueType !== viewType && bitcastType ? `bitcast<${bitcastType}>(${access})` : access;
}

export function emitDeviceGlobalPointerWrite(
  global: CudaLiteDeviceGlobal,
  index: string,
  value: string,
  ir: KernelIrModule,
  context: WgslStorageEmitContext,
  viewType: CudaLiteScalarType = global.valueType,
): string {
  const name = context.nameFor(global.name);
  if (isCudaVectorType(viewType)) {
    if (ir.atomicDeviceGlobals.includes(global.name)) {
      return emitDeviceGlobalVectorFlatWrite(global, vectorStorageBase(index, cudaVectorLaneCount(viewType)), value, viewType, ir, context);
    }
    return global.valueType === viewType
      ? emitVectorStorageWrite(name, viewType, index, value)
      : emitVectorStorageWriteAt(name, viewType, index, value);
  }
  const access = `${name}[${index}]`;
  if (global.valueType === "bool") return `${access} = select(0u, 1u, ${value})`;
  const bitcastType = bitcastStorageViewType(viewType, global.valueType);
  if (!ir.atomicDeviceGlobals.includes(global.name)) return global.valueType !== viewType && bitcastType
    ? `${access} = bitcast<${bitcastType}>(${value})`
    : `${access} = ${value}`;
  const storageScalar = atomicStorageScalarType(global.valueType);
  if (storageScalar === "float" || storageScalar === "double") return `atomicStore(&${access}, ${emitStorageCarrierAsU32(value, viewType)})`;
  const atomicBitcastType = bitcastStorageViewType(viewType, storageScalar);
  return storageScalar !== viewType && atomicBitcastType
    ? `atomicStore(&${access}, bitcast<${atomicBitcastType}>(${value}))`
    : `atomicStore(&${access}, ${value})`;
}

function atomicStorageScalarType(valueType: CudaLiteScalarType): CudaLiteScalarType {
  return cudaVectorScalarType(valueType) ?? valueType;
}

export function emitConstantPointerRead(
  constant: CudaLiteGlobalConstant,
  index: string,
  context: WgslStorageEmitContext,
  viewType: CudaLiteScalarType = constant.valueType,
): string {
  if (isCudaVectorType(viewType)) {
    return constant.valueType === viewType
      ? emitVectorStorageRead(context.nameFor(constant.name), viewType, index)
      : emitConstantVectorFlatRead(constant, index, viewType, context);
  }
  return emitSharedFlatAccess(context.nameFor(constant.name), externalConstantDimensions(constant), index);
}

export function emitSharedPointerRead(
  shared: CudaLiteVarDecl,
  index: string,
  ir: KernelIrModule,
  context: WgslStorageEmitContext,
  viewType: CudaLiteScalarType = shared.valueType,
  subElementLane?: string,
): string {
  const packedByte = emitPackedByteSharedRead(shared, index, viewType, context);
  if (packedByte) return packedByte;
  const access = emitSharedPointerAccess(shared, index, ir, context, viewType);
  const packed = emitPackedHalfStorageRead(access, shared.valueType, viewType, subElementLane);
  if (packed) return packed;
  if (isCudaVectorType(shared.valueType) && !isCudaVectorType(viewType) && subElementLane !== undefined) {
    return `${wgslScalar(viewType)}(${access}[${subElementLane}])`;
  }
  if (isCudaVectorType(viewType) && shared.valueType !== viewType) return emitSharedVectorFlatRead(shared, index, viewType, context);
  if (ir.atomicShared.includes(shared.name)) {
    const storageScalar = atomicStorageScalarType(shared.valueType);
    const bitcastType = bitcastStorageViewType(storageScalar, viewType);
    const loaded = `atomicLoad(&${access})`;
    if (storageScalar === "float" || storageScalar === "double") {
      if (viewType === "uint") return loaded;
      if (viewType === "int") return `bitcast<i32>(${loaded})`;
      return `bitcast<f32>(${loaded})`;
    }
    return storageScalar !== viewType && bitcastType ? `bitcast<${bitcastType}>(${loaded})` : loaded;
  }
  const bitcastType = bitcastStorageViewType(shared.valueType, viewType);
  if (shared.valueType !== viewType && bitcastType) return `bitcast<${bitcastType}>(${access})`;
  return access;
}

export function emitSharedPointerWrite(
  shared: CudaLiteVarDecl,
  index: string,
  value: string,
  ir: KernelIrModule,
  context: WgslStorageEmitContext,
  viewType: CudaLiteScalarType = shared.valueType,
  subElementLane?: string,
): string {
  const packedByte = emitPackedByteSharedWrite(shared, index, value, viewType, context);
  if (packedByte) return packedByte;
  const access = emitSharedPointerAccess(shared, index, ir, context, viewType);
  const packed = emitPackedHalfStorageWrite(access, shared.valueType, viewType, value, subElementLane);
  if (packed) return packed;
  if (isCudaVectorType(shared.valueType) && !isCudaVectorType(viewType) && subElementLane !== undefined) {
    return `${access} = ${emitVectorLaneSetExpression(access, shared.valueType, subElementLane, value)}`;
  }
  if (isCudaVectorType(viewType) && shared.valueType !== viewType) return emitSharedVectorFlatWrite(shared, index, value, viewType, context);
  const bitcastType = bitcastStorageViewType(viewType, shared.valueType);
  if (!ir.atomicShared.includes(shared.name)) return shared.valueType !== viewType && bitcastType
    ? `${access} = bitcast<${bitcastType}>(${value})`
    : `${access} = ${value}`;
  const storageScalar = atomicStorageScalarType(shared.valueType);
  if (storageScalar === "float" || storageScalar === "double") {
    return `atomicStore(&${access}, ${emitStorageCarrierAsU32(value, viewType)})`;
  }
  const atomicBitcastType = bitcastStorageViewType(viewType, storageScalar);
  return storageScalar !== viewType && atomicBitcastType
    ? `atomicStore(&${access}, bitcast<${atomicBitcastType}>(${value}))`
    : `atomicStore(&${access}, ${value})`;
}

function emitSharedPointerAccess(
  shared: CudaLiteVarDecl,
  index: string,
  ir: KernelIrModule,
  context: WgslStorageEmitContext,
  viewType: CudaLiteScalarType,
): string {
  if (ir.atomicShared.includes(shared.name) && isCudaVectorType(shared.valueType) && !isCudaVectorType(viewType)) {
    return `${context.nameFor(shared.name)}[${index}]`;
  }
  return emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, index);
}

export function emitLocalPointerRead(
  local: CudaLiteVarDecl,
  index: string,
  viewType: CudaLiteScalarType,
  context: WgslStorageEmitContext,
  subElementLane?: string,
): string {
  const access = emitSharedFlatAccess(context.nameFor(local.name), matrixTileStorageDimensions(local), index);
  const packed = emitPackedHalfStorageRead(access, local.valueType, viewType, subElementLane);
  if (packed) return packed;
  if (isCudaVectorType(local.valueType) && !isCudaVectorType(viewType) && subElementLane !== undefined) {
    return `${wgslScalar(viewType)}(${access}[${subElementLane}])`;
  }
  if (isCudaVectorType(viewType) && local.valueType !== viewType) return emitLocalVectorFlatRead(local, index, viewType, context);
  const bitcastType = bitcastStorageViewType(local.valueType, viewType);
  return local.valueType !== viewType && bitcastType ? `bitcast<${bitcastType}>(${access})` : access;
}

export function emitLocalPointerWrite(
  local: CudaLiteVarDecl,
  index: string,
  value: string,
  viewType: CudaLiteScalarType,
  context: WgslStorageEmitContext,
  subElementLane?: string,
): string {
  const access = emitSharedFlatAccess(context.nameFor(local.name), matrixTileStorageDimensions(local), index);
  const packed = emitPackedHalfStorageWrite(access, local.valueType, viewType, value, subElementLane);
  if (packed) return packed;
  if (isCudaVectorType(local.valueType) && !isCudaVectorType(viewType) && subElementLane !== undefined) {
    return `${access} = ${emitVectorLaneSetExpression(access, local.valueType, subElementLane, value)}`;
  }
  if (isCudaVectorType(viewType) && local.valueType !== viewType) return emitLocalVectorFlatWrite(local, index, value, viewType, context);
  const bitcastType = bitcastStorageViewType(viewType, local.valueType);
  return local.valueType !== viewType && bitcastType ? `${access} = bitcast<${bitcastType}>(${value})` : `${access} = ${value}`;
}

function emitPackedHalfStorageRead(
  access: string,
  storageType: CudaLiteScalarType,
  viewType: CudaLiteScalarType,
  subElementLane?: string,
): string | undefined {
  if (!isPackableHalfCarrier(storageType)) return undefined;
  const bits = emitStorageCarrierAsU32(access, storageType);
  if (viewType === "half2") return `${wgslScalar("half2")}(unpack2x16float(${bits}))`;
  if (viewType !== "half" || subElementLane === undefined) return undefined;
  const unpacked = `unpack2x16float(${bits})`;
  return `${wgslScalar("half")}(select(${unpacked}.x, ${unpacked}.y, (${subElementLane}) == 1u))`;
}

function emitPackedByteSharedRead(
  shared: CudaLiteVarDecl,
  index: string,
  viewType: CudaLiteScalarType,
  context: WgslStorageEmitContext,
): string | undefined {
  if (shared.valueType !== "uchar") return undefined;
  const wordIndex = `((${index}) >> 2u)`;
  const word = `${context.nameFor(shared.name)}[${wordIndex}]`;
  if (viewType !== "uchar") return viewType === "int" ? `bitcast<i32>(${word})` : word;
  const shift = `(((${index}) & 3u) * 8u)`;
  return `((${word} >> (${shift})) & 255u)`;
}

function emitPackedByteSharedWrite(
  shared: CudaLiteVarDecl,
  index: string,
  value: string,
  viewType: CudaLiteScalarType,
  context: WgslStorageEmitContext,
): string | undefined {
  if (shared.valueType !== "uchar") return undefined;
  const wordIndex = `((${index}) >> 2u)`;
  const word = `${context.nameFor(shared.name)}[${wordIndex}]`;
  if (viewType !== "uchar") return `${word} = ${viewType === "int" ? `bitcast<u32>(${value})` : `u32(${value})`}`;
  const shift = `(((${index}) & 3u) * 8u)`;
  const mask = `(255u << ${shift})`;
  return `${word} = ((${word} & ~${mask}) | ((u32(${value}) & 255u) << ${shift}))`;
}

function emitPackedHalfStorageWrite(
  access: string,
  storageType: CudaLiteScalarType,
  viewType: CudaLiteScalarType,
  value: string,
  subElementLane?: string,
): string | undefined {
  if (!isPackableHalfCarrier(storageType)) return undefined;
  if (viewType === "half2") {
    return `${access} = ${emitU32AsStorageCarrier(`pack2x16float(vec2<f32>(${value}))`, storageType)}`;
  }
  if (viewType !== "half" || subElementLane === undefined) return undefined;
  const unpacked = `unpack2x16float(${emitStorageCarrierAsU32(access, storageType)})`;
  const packed = `pack2x16float(vec2<f32>(select(${unpacked}.x, f32(${value}), (${subElementLane}) == 0u), select(${unpacked}.y, f32(${value}), (${subElementLane}) == 1u)))`;
  return `${access} = ${emitU32AsStorageCarrier(packed, storageType)}`;
}

function isPackableHalfCarrier(storageType: CudaLiteScalarType): boolean {
  return wgslElementByteSize(storageType) === 4 && !isCudaVectorType(storageType) && storageType !== "bool";
}

export function emitSharedVectorFlatRead(
  shared: CudaLiteVarDecl,
  index: string,
  viewType: CudaLiteScalarType,
  context: WgslStorageEmitContext,
): string {
  const lanes = cudaVectorLaneCount(viewType);
  const scalar = cudaVectorScalarType(viewType) ?? "float";
  const rootLanes = isCudaVectorType(shared.valueType) ? cudaVectorLaneCount(shared.valueType) : 1;
  const values = Array.from({ length: lanes }, (_, lane) => {
    const flatIndex = `(${index} + ${lane}u)`;
    return rootLanes <= 1
      ? emitSharedPointerRead(shared, flatIndex, context.ir, context, scalar)
      : emitSharedPointerRead(
        shared,
        `(u32(${flatIndex}) / ${rootLanes}u)`,
        context.ir,
        context,
        scalar,
        `(u32(${flatIndex}) % ${rootLanes}u)`,
      );
  });
  return `${wgslScalar(viewType)}(${values.join(", ")})`;
}

export function emitPointerVectorFlatRead(
  param: CudaLiteParam,
  index: string,
  viewType: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslStorageEmitContext,
): string {
  const lanes = cudaVectorLaneCount(viewType);
  const scalar = cudaVectorScalarType(viewType) ?? "float";
  const values = Array.from({ length: lanes }, (_, lane) =>
    emitPointerStorageRead(param, `(${index} + ${lane}u)`, ir, context, scalar)
  );
  return `${wgslScalar(viewType)}(${values.join(", ")})`;
}

export function emitDeviceGlobalVectorFlatRead(
  global: CudaLiteDeviceGlobal,
  index: string,
  viewType: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslStorageEmitContext,
): string {
  const lanes = cudaVectorLaneCount(viewType);
  const scalar = cudaVectorScalarType(viewType) ?? "float";
  const values = Array.from({ length: lanes }, (_, lane) =>
    emitDeviceGlobalPointerRead(global, `(${index} + ${lane}u)`, ir, context, scalar)
  );
  return `${wgslScalar(viewType)}(${values.join(", ")})`;
}

function emitLocalVectorFlatRead(
  local: CudaLiteVarDecl,
  index: string,
  viewType: CudaLiteScalarType,
  context: WgslStorageEmitContext,
): string {
  const lanes = cudaVectorLaneCount(viewType);
  const scalar = cudaVectorScalarType(viewType) ?? "float";
  const rootLanes = isCudaVectorType(local.valueType) ? cudaVectorLaneCount(local.valueType) : 1;
  const values = Array.from({ length: lanes }, (_, lane) => {
    const flatIndex = `(${index} + ${lane}u)`;
    return rootLanes <= 1
      ? emitLocalPointerRead(local, flatIndex, scalar, context)
      : emitLocalPointerRead(
        local,
        `(u32(${flatIndex}) / ${rootLanes}u)`,
        scalar,
        context,
        `(u32(${flatIndex}) % ${rootLanes}u)`,
      );
  });
  return `${wgslScalar(viewType)}(${values.join(", ")})`;
}

export function emitSharedVectorFlatWrite(
  shared: CudaLiteVarDecl,
  index: string,
  value: string,
  viewType: CudaLiteScalarType,
  context: WgslStorageEmitContext,
): string {
  const lanes = cudaVectorLaneCount(viewType);
  const scalar = cudaVectorScalarType(viewType) ?? "float";
  const rootLanes = isCudaVectorType(shared.valueType) ? cudaVectorLaneCount(shared.valueType) : 1;
  return Array.from({ length: lanes }, (_, lane) => {
    const flatIndex = `(${index} + ${lane}u)`;
    return rootLanes <= 1
      ? emitSharedPointerWrite(shared, flatIndex, `${value}.${vectorFieldName(lane)}`, context.ir, context, scalar)
      : emitSharedPointerWrite(
        shared,
        `(u32(${flatIndex}) / ${rootLanes}u)`,
        `${value}.${vectorFieldName(lane)}`,
        context.ir,
        context,
        scalar,
        `(u32(${flatIndex}) % ${rootLanes}u)`,
      );
  }).join("; ");
}

export function emitPointerVectorFlatWrite(
  param: CudaLiteParam,
  index: string,
  value: string,
  viewType: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslStorageEmitContext,
): string {
  const lanes = cudaVectorLaneCount(viewType);
  const scalar = cudaVectorScalarType(viewType) ?? "float";
  return Array.from({ length: lanes }, (_, lane) =>
    emitPointerStorageWrite(param, `(${index} + ${lane}u)`, `${value}.${vectorFieldName(lane)}`, ir, context, scalar)
  ).join("; ");
}

export function emitDeviceGlobalVectorFlatWrite(
  global: CudaLiteDeviceGlobal,
  index: string,
  value: string,
  viewType: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslStorageEmitContext,
): string {
  const lanes = cudaVectorLaneCount(viewType);
  const scalar = cudaVectorScalarType(viewType) ?? "float";
  return Array.from({ length: lanes }, (_, lane) =>
    emitDeviceGlobalPointerWrite(global, `(${index} + ${lane}u)`, `${value}.${vectorFieldName(lane)}`, ir, context, scalar)
  ).join("; ");
}

function emitLocalVectorFlatWrite(
  local: CudaLiteVarDecl,
  index: string,
  value: string,
  viewType: CudaLiteScalarType,
  context: WgslStorageEmitContext,
): string {
  const lanes = cudaVectorLaneCount(viewType);
  const scalar = cudaVectorScalarType(viewType) ?? "float";
  const rootLanes = isCudaVectorType(local.valueType) ? cudaVectorLaneCount(local.valueType) : 1;
  return Array.from({ length: lanes }, (_, lane) => {
    const flatIndex = `(${index} + ${lane}u)`;
    return rootLanes <= 1
      ? emitLocalPointerWrite(local, flatIndex, `${value}.${vectorFieldName(lane)}`, scalar, context)
      : emitLocalPointerWrite(
        local,
        `(u32(${flatIndex}) / ${rootLanes}u)`,
        `${value}.${vectorFieldName(lane)}`,
        scalar,
        context,
        `(u32(${flatIndex}) % ${rootLanes}u)`,
      );
  }).join("; ");
}

export function emitConstantVectorFlatRead(
  constant: CudaLiteGlobalConstant,
  index: string,
  viewType: CudaLiteScalarType,
  context: WgslStorageEmitContext,
): string {
  const lanes = cudaVectorLaneCount(viewType);
  const values = Array.from({ length: lanes }, (_, lane) =>
    emitSharedFlatAccess(context.nameFor(constant.name), externalConstantDimensions(constant), `(${index} + ${lane}u)`)
  );
  return `${wgslScalar(viewType)}(${values.join(", ")})`;
}

export function externalConstantDimensions(constant: CudaLiteGlobalConstant): readonly number[] {
  if (!isCudaVectorType(constant.valueType)) return constant.dimensions;
  const elements = constant.dimensions.length === 0
    ? 1
    : constant.dimensions.reduce((product, dimension) => product * dimension, 1);
  return [elements * cudaVectorLaneCount(constant.valueType)];
}
