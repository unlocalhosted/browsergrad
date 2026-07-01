import { walkCudaLiteExpressions } from "./ast_queries.js";
import { expressionName } from "./analyzer.js";
import {
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
} from "./vector_types.js";
import {
  floatAtomicHelperName,
  integerAtomicLoopHelperName,
} from "./wgsl_atomic_helpers.js";
import {
  emitConstantPointerRead,
  emitDeviceGlobalPointerRead,
  emitDeviceGlobalPointerWrite,
  emitPointerStorageRead,
  emitPointerStorageWrite,
  emitSharedFlatAccess,
  emitSharedPointerRead,
  emitSharedPointerWrite,
  wgslScalar,
  zeroValue,
  type WgslStorageEmitContext,
} from "./wgsl_storage.js";
import {
  collectLocalArrayDeclarations,
  collectLocalPointerHandles,
  collectPointerAliases,
  isLocalPointerArrayDecl,
} from "./wgsl_ir_analysis.js";
import {
  reachableDevicePointerHelperBufferIds,
  usedDevicePointerHelperTypeNames,
} from "./wgsl_pointer_usage.js";
import {
  CudaLiteCompilerError,
  type CudaLiteDeviceGlobal,
  type CudaLiteGlobalConstant,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type CudaLiteStatement,
  type CudaLiteVarDecl,
  type KernelIrModule,
  type SourceSpan,
} from "./types.js";

export interface WgslPointerHelperContext extends WgslStorageEmitContext {
  storagePointerIdFor(name: string): number | undefined;
  sharedPointerIdFor(name: string): number | undefined;
  deviceGlobalPointerIdFor(name: string): number | undefined;
  constantPointerIdFor(name: string): number | undefined;
}

export interface EmitDevicePointerHelperOptions {
  readonly reachableBufferIds?: ReadonlySet<number>;
}

export function emitDevicePointerHelpers(ir: KernelIrModule, context: WgslPointerHelperContext, usageLines?: readonly string[]): string[] {
  const types = [...devicePointerHelperTypes(ir, usageLines)];
  const bufferIds = usageLines ? reachableDevicePointerHelperBufferIds(usageLines) : new Map<string, ReadonlySet<number>>();
  return types.flatMap((type) => {
    const reachableBufferIds = bufferIds.get(pointerHelperTypeName(type));
    return reachableBufferIds
      ? emitDevicePointerHelper(type, ir, context, { reachableBufferIds })
      : emitDevicePointerHelper(type, ir, context);
  });
}

export function emitDevicePointerHelper(
  type: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
  options: EmitDevicePointerHelperOptions = {},
): string[] {
  if (!isDevicePointerHelperType(type)) {
    throw featureError("unsupported-device-pointer-param", `device pointer helpers do not support ${type} pointers yet`);
  }
  const storageParams = ir.params.filter((param) =>
    param.pointer &&
    !isDevicePoolParam(param) &&
    isPointerHelperReadableStorage(type, param.valueType)
  );
  const sharedDeclarations = ir.sharedDeclarations.filter((shared) =>
    isPointerHelperReadableStorage(type, shared.valueType) ||
    isPointerHelperBitcastCompatibleStorage(type, shared.valueType)
  );
  const deviceGlobals = ir.deviceGlobals.filter((global) =>
    isPointerHelperReadableStorage(type, global.valueType)
  );
  const constantArrays = ir.constants.filter((constant) =>
    constant.dimensions.length > 0 &&
    constant.init === undefined &&
    isPointerHelperCompatibleStorage(type, constant.valueType)
  );
  const scalar = wgslScalar(type);
  const singletonBufferId = singletonReachableBufferId(options.reachableBufferIds);
  if (singletonBufferId !== undefined) {
    return emitSingletonDevicePointerHelper(type, ir, context, singletonBufferId, storageParams, sharedDeclarations, deviceGlobals, constantArrays);
  }
  const lines = [
    `fn ${pointerReadHelperName(type)}(buffer: u32, index: u32) -> ${scalar} {`,
    "  switch buffer {",
  ];
  for (const param of storageParams) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    if (options.reachableBufferIds && !options.reachableBufferIds.has(id)) continue;
    lines.push(`    case ${id}u: { return ${emitPointerStorageRead(param, "index", ir, context, type)}; }`);
  }
  for (const shared of sharedDeclarations) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    if (options.reachableBufferIds && !options.reachableBufferIds.has(id)) continue;
    lines.push(`    case ${id}u: { return ${emitSharedPointerRead(
      shared,
      sharedPointerHelperIndex(shared.valueType, type, "index", context.ir.atomicShared.includes(shared.name)),
      ir,
      context,
      type,
      sharedPointerHelperLane(shared.valueType, type, "index", context.ir.atomicShared.includes(shared.name)),
    )}; }`);
  }
  for (const global of deviceGlobals) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    if (options.reachableBufferIds && !options.reachableBufferIds.has(id)) continue;
    lines.push(`    case ${id}u: { return ${emitDeviceGlobalPointerRead(global, "index", ir, context, type)}; }`);
  }
  for (const constant of constantArrays) {
    const id = context.constantPointerIdFor(constant.name);
    if (id === undefined) continue;
    if (options.reachableBufferIds && !options.reachableBufferIds.has(id)) continue;
    lines.push(`    case ${id}u: { return ${emitConstantPointerRead(constant, "index", context, type)}; }`);
  }
  lines.push(`    default: { return ${zeroValue(type)}; }`);
  lines.push("  }");
  lines.push("}");
  lines.push("");
  lines.push(`fn ${pointerWriteHelperName(type)}(buffer: u32, index: u32, value: ${scalar}) {`);
  lines.push("  switch buffer {");
  for (const param of storageParams.filter((param) => !param.constant)) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    if (options.reachableBufferIds && !options.reachableBufferIds.has(id)) continue;
    lines.push(`    case ${id}u: { ${emitPointerStorageWrite(param, "index", "value", ir, context, type)}; return; }`);
  }
  for (const shared of sharedDeclarations) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    if (options.reachableBufferIds && !options.reachableBufferIds.has(id)) continue;
    lines.push(`    case ${id}u: { ${emitSharedPointerWrite(
      shared,
      sharedPointerHelperIndex(shared.valueType, type, "index", context.ir.atomicShared.includes(shared.name)),
      "value",
      ir,
      context,
      type,
      sharedPointerHelperLane(shared.valueType, type, "index", context.ir.atomicShared.includes(shared.name)),
    )}; return; }`);
  }
  for (const global of deviceGlobals) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    if (options.reachableBufferIds && !options.reachableBufferIds.has(id)) continue;
    lines.push(`    case ${id}u: { ${emitDeviceGlobalPointerWrite(global, "index", "value", ir, context, type)}; return; }`);
  }
  lines.push("    default: { return; }");
  lines.push("  }");
  lines.push("}");
  lines.push(...emitDevicePointerAtomicHelpers(type, ir, context));
  return lines;
}

function singletonReachableBufferId(ids: ReadonlySet<number> | undefined): number | undefined {
  return ids?.size === 1 ? [...ids][0] : undefined;
}

function emitSingletonDevicePointerHelper(
  type: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
  bufferId: number,
  storageParams: readonly CudaLiteParam[],
  sharedDeclarations: readonly CudaLiteVarDecl[],
  deviceGlobals: readonly CudaLiteDeviceGlobal[],
  constantArrays: readonly CudaLiteGlobalConstant[],
): string[] {
  const scalar = wgslScalar(type);
  const read = singletonDevicePointerReadExpression(type, ir, context, bufferId, storageParams, sharedDeclarations, deviceGlobals, constantArrays);
  const write = singletonDevicePointerWriteExpression(type, ir, context, bufferId, storageParams, sharedDeclarations, deviceGlobals);
  const lines = [
    `fn ${pointerReadHelperName(type)}(buffer: u32, index: u32) -> ${scalar} {`,
    `  return ${read ?? zeroValue(type)};`,
    "}",
    "",
    `fn ${pointerWriteHelperName(type)}(buffer: u32, index: u32, value: ${scalar}) {`,
  ];
  if (write) lines.push(`  ${write};`);
  lines.push("  return;");
  lines.push("}");
  lines.push(...emitDevicePointerAtomicHelpers(type, ir, context));
  return lines;
}

function singletonDevicePointerReadExpression(
  type: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
  bufferId: number,
  storageParams: readonly CudaLiteParam[],
  sharedDeclarations: readonly CudaLiteVarDecl[],
  deviceGlobals: readonly CudaLiteDeviceGlobal[],
  constantArrays: readonly CudaLiteGlobalConstant[],
): string | undefined {
  for (const param of storageParams) {
    if (context.storagePointerIdFor(param.name) === bufferId) return emitPointerStorageRead(param, "index", ir, context, type);
  }
  for (const shared of sharedDeclarations) {
    if (context.sharedPointerIdFor(shared.name) === bufferId) return emitSharedPointerRead(
      shared,
      sharedPointerHelperIndex(shared.valueType, type, "index", context.ir.atomicShared.includes(shared.name)),
      ir,
      context,
      type,
      sharedPointerHelperLane(shared.valueType, type, "index", context.ir.atomicShared.includes(shared.name)),
    );
  }
  for (const global of deviceGlobals) {
    if (context.deviceGlobalPointerIdFor(global.name) === bufferId) return emitDeviceGlobalPointerRead(global, "index", ir, context, type);
  }
  for (const constant of constantArrays) {
    if (context.constantPointerIdFor(constant.name) === bufferId) return emitConstantPointerRead(constant, "index", context, type);
  }
  return undefined;
}

function singletonDevicePointerWriteExpression(
  type: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
  bufferId: number,
  storageParams: readonly CudaLiteParam[],
  sharedDeclarations: readonly CudaLiteVarDecl[],
  deviceGlobals: readonly CudaLiteDeviceGlobal[],
): string | undefined {
  for (const param of storageParams.filter((item) => !item.constant)) {
    if (context.storagePointerIdFor(param.name) === bufferId) return emitPointerStorageWrite(param, "index", "value", ir, context, type);
  }
  for (const shared of sharedDeclarations) {
    if (context.sharedPointerIdFor(shared.name) === bufferId) return emitSharedPointerWrite(
      shared,
      sharedPointerHelperIndex(shared.valueType, type, "index", context.ir.atomicShared.includes(shared.name)),
      "value",
      ir,
      context,
      type,
      sharedPointerHelperLane(shared.valueType, type, "index", context.ir.atomicShared.includes(shared.name)),
    );
  }
  for (const global of deviceGlobals) {
    if (context.deviceGlobalPointerIdFor(global.name) === bufferId) return emitDeviceGlobalPointerWrite(global, "index", "value", ir, context, type);
  }
  return undefined;
}

function emitDevicePointerAtomicHelpers(type: CudaLiteScalarType, ir: KernelIrModule, context: WgslPointerHelperContext): string[] {
  const lines: string[] = [];
  if (isDevicePointerAtomicAddType(type) && usesDevicePointerAtomicAdd(ir)) lines.push("", ...emitDevicePointerAtomicAddHelper(type, ir, context));
  if (isDevicePointerAtomicSubType(type) && usesDevicePointerAtomicSub(ir)) lines.push("", ...emitDevicePointerAtomicSubHelper(type, ir, context));
  if (isDevicePointerAtomicMinMaxType(type) && usesDevicePointerAtomicMin(ir)) lines.push("", ...emitDevicePointerAtomicMinHelper(type, ir, context));
  if (isDevicePointerAtomicMinMaxType(type) && usesDevicePointerAtomicMax(ir)) lines.push("", ...emitDevicePointerAtomicMaxHelper(type, ir, context));
  if (isDevicePointerAtomicBitwiseType(type) && usesDevicePointerAtomicAnd(ir)) lines.push("", ...emitDevicePointerAtomicAndHelper(type, ir, context));
  if (isDevicePointerAtomicBitwiseType(type) && usesDevicePointerAtomicOr(ir)) lines.push("", ...emitDevicePointerAtomicOrHelper(type, ir, context));
  if (isDevicePointerAtomicBitwiseType(type) && usesDevicePointerAtomicXor(ir)) lines.push("", ...emitDevicePointerAtomicXorHelper(type, ir, context));
  if (isDevicePointerAtomicIncDecType(type) && usesDevicePointerAtomicInc(ir)) lines.push("", ...emitDevicePointerAtomicIncHelper(type, ir, context));
  if (isDevicePointerAtomicIncDecType(type) && usesDevicePointerAtomicDec(ir)) lines.push("", ...emitDevicePointerAtomicDecHelper(type, ir, context));
  if (isDevicePointerAtomicExchangeType(type) && usesDevicePointerAtomicExchange(ir)) lines.push("", ...emitDevicePointerAtomicExchangeHelper(type, ir, context));
  if (isDevicePointerAtomicCasType(type) && usesDevicePointerAtomicCas(ir)) lines.push("", ...emitDevicePointerAtomicCasHelper(type, ir, context));
  return lines;
}

function sharedPointerHelperIndex(
  storageType: CudaLiteScalarType,
  viewType: CudaLiteScalarType,
  index: string,
  flatScalarVectorStorage = false,
): string {
  if (isCudaVectorType(storageType) && !isCudaVectorType(viewType) && !flatScalarVectorStorage) {
    return `(u32(${index}) / ${cudaVectorLaneCount(storageType)}u)`;
  }
  if (isCudaVectorType(viewType) && !isCudaVectorType(storageType)) {
    return `(u32(${index}) * ${cudaVectorLaneCount(viewType)}u)`;
  }
  return index;
}

function sharedPointerHelperLane(
  storageType: CudaLiteScalarType,
  viewType: CudaLiteScalarType,
  index: string,
  flatScalarVectorStorage = false,
): string | undefined {
  if (!isCudaVectorType(storageType) || isCudaVectorType(viewType) || flatScalarVectorStorage) return undefined;
  return `(u32(${index}) % ${cudaVectorLaneCount(storageType)}u)`;
}

export function isDevicePointerHelperType(type: CudaLiteScalarType): boolean {
  return type === "float" || type === "double" || type === "int" || type === "uint" || type === "uchar" || type === "half" || type === "bf16" || type === "bool" || isCudaVectorType(type);
}

function devicePointerHelperTypes(ir: KernelIrModule, usageLines?: readonly string[]): ReadonlySet<CudaLiteScalarType> {
  const rawTypes = [
    ...ir.params.filter((param) => param.pointer && !isDevicePoolParam(param)).map((param) => param.valueType),
    ...ir.sharedDeclarations.map((shared) => shared.valueType),
    ...ir.deviceGlobals.map((global) => global.valueType),
    ...ir.functions.flatMap((fn) => fn.params.filter((param) => param.pointer).map((param) => param.valueType)),
    ...collectLocalArrayDeclarations(ir.body).filter(isLocalPointerArrayDecl).map((item) => item.valueType),
    ...ir.functions.flatMap((fn) => collectLocalArrayDeclarations(fn.body).filter(isLocalPointerArrayDecl).map((item) => item.valueType)),
    ...[...collectLocalPointerHandles(ir.body).values()].map((item) => item.valueType),
    ...ir.functions.flatMap((fn) => [...collectLocalPointerHandles(fn.body).values()].map((item) => item.valueType)),
    ...collectPointerAliasValueTypes(ir.body),
    ...ir.functions.flatMap((fn) => collectPointerAliasValueTypes(fn.body)),
  ];
  const types = new Set<CudaLiteScalarType>(rawTypes.map(pointerHelperCanonicalType));
  for (const statements of [ir.body, ...ir.functions.map((fn) => fn.body)]) {
    walkCudaLiteExpressions(statements, (expression) => {
      if (expression.kind === "cast" && expression.pointer && isDevicePointerHelperType(expression.valueType)) {
        types.add(pointerHelperCanonicalType(expression.valueType));
      }
    });
  }
  if (usageLines === undefined) return types;
  const usedTypeNames = usedDevicePointerHelperTypeNames(usageLines);
  for (const typeName of usedTypeNames) {
    const type = pointerHelperTypeFromName(typeName);
    if (type !== undefined) types.add(type);
  }
  return uniquePointerHelperTypes([...types].filter((type) => usedTypeNames.has(pointerHelperTypeName(type))));
}

function uniquePointerHelperTypes(types: readonly CudaLiteScalarType[]): ReadonlySet<CudaLiteScalarType> {
  const byName = new Map<string, CudaLiteScalarType>();
  for (const type of types) {
    const name = pointerHelperTypeName(type);
    if (!byName.has(name)) byName.set(name, type);
  }
  return new Set(byName.values());
}

function pointerHelperTypeFromName(name: string): CudaLiteScalarType | undefined {
  if (name === "f32") return "float";
  if (name === "f16") return "half";
  if (name === "bf16") return "bf16";
  if (name === "i32") return "int";
  if (name === "u32") return "uint";
  if (name === "u8") return "uchar";
  if (name === "bool") return "bool";
  const vector = /^(f32|f16|i32|u32)x([234])$/u.exec(name);
  if (!vector) return undefined;
  const scalar = vector[1] === "f32"
    ? "float"
    : vector[1] === "f16"
      ? "half"
      : vector[1] === "i32"
        ? "int"
        : "uint";
  return `${scalar}${vector[2]}` as CudaLiteScalarType;
}

function collectPointerAliasValueTypes(statements: readonly CudaLiteStatement[]): readonly CudaLiteScalarType[] {
  return [...collectPointerAliases(statements).values()]
    .flat()
    .map((alias) => alias.valueType)
    .filter((type): type is CudaLiteScalarType => type !== undefined);
}

function pointerHelperCanonicalType(type: CudaLiteScalarType): CudaLiteScalarType {
  return type === "double" ? "float" : type;
}

export function isDevicePointerAtomicAddType(type: CudaLiteScalarType): type is "float" | "double" | "int" | "uint" {
  return type === "float" || type === "double" || type === "int" || type === "uint";
}

export function isDevicePointerAtomicSubType(type: CudaLiteScalarType): type is "float" | "double" | "int" | "uint" {
  return type === "float" || type === "double" || type === "int" || type === "uint";
}

export function isDevicePointerAtomicMinMaxType(type: CudaLiteScalarType): type is "float" | "double" | "int" | "uint" {
  return type === "float" || type === "double" || type === "int" || type === "uint";
}

export function isDevicePointerAtomicBitwiseType(type: CudaLiteScalarType): type is "int" | "uint" {
  return type === "int" || type === "uint";
}

export function isDevicePointerAtomicIncDecType(type: CudaLiteScalarType): type is "int" | "uint" {
  return type === "int" || type === "uint";
}

export function isDevicePointerAtomicExchangeType(type: CudaLiteScalarType): type is "float" | "int" | "uint" {
  return type === "float" || type === "int" || type === "uint";
}

export function isDevicePointerAtomicCasType(type: CudaLiteScalarType): type is "int" | "uint" {
  return type === "int" || type === "uint";
}

function usesDevicePointerAtomicAdd(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["atomicAdd", "atomicAdd_system"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicAdd", "atomicAdd_system"])));
}

function usesDevicePointerAtomicSub(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["atomicSub", "atomicSub_system"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicSub", "atomicSub_system"])));
}

function usesDevicePointerAtomicMin(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["atomicMin", "atomicMin_system"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicMin", "atomicMin_system"])));
}

function usesDevicePointerAtomicMax(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["atomicMax", "atomicMax_system", "atomicMaxFloat"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicMax", "atomicMax_system", "atomicMaxFloat"])));
}

function usesDevicePointerAtomicAnd(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["atomicAnd", "atomicAnd_system"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicAnd", "atomicAnd_system"])));
}

function usesDevicePointerAtomicOr(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["atomicOr", "atomicOr_system"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicOr", "atomicOr_system"])));
}

function usesDevicePointerAtomicXor(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["atomicXor", "atomicXor_system"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicXor", "atomicXor_system"])));
}

function usesDevicePointerAtomicInc(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["atomicInc", "atomicInc_system"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicInc", "atomicInc_system"])));
}

function usesDevicePointerAtomicDec(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["atomicDec", "atomicDec_system"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicDec", "atomicDec_system"])));
}

function usesDevicePointerAtomicExchange(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["atomicExch", "atomicExch_system"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicExch", "atomicExch_system"])));
}

function usesDevicePointerAtomicCas(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["atomicCAS", "atomicCAS_system"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicCAS", "atomicCAS_system"])));
}

function emitDevicePointerAtomicAddHelper(
  type: "float" | "double" | "int" | "uint",
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
): string[] {
  const scalar = wgslScalar(type);
  const lines = [
    `fn ${pointerAtomicAddHelperName(type)}(buffer: u32, index: u32, value: ${scalar}) -> ${scalar} {`,
    "  switch buffer {",
  ];
  for (const param of ir.params.filter((param) =>
    param.pointer &&
    !param.constant &&
    isPointerHelperCompatibleStorage(type, param.valueType) &&
    ir.atomicParams.includes(param.name)
  )) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    const access = `${context.nameFor(param.name)}[index]`;
    lines.push(`    case ${id}u: { return ${emitAtomicAddAtAddress(type, "storage", `&${access}`, "value")}; }`);
  }
  for (const shared of ir.sharedDeclarations.filter((shared) =>
    isPointerHelperCompatibleStorage(type, shared.valueType) &&
    ir.atomicShared.includes(shared.name)
  )) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    const access = emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, "index");
    lines.push(`    case ${id}u: { return ${emitAtomicAddAtAddress(type, "workgroup", `&${access}`, "value")}; }`);
  }
  for (const global of ir.deviceGlobals.filter((global) =>
    isPointerHelperCompatibleStorage(type, global.valueType) &&
    ir.atomicDeviceGlobals.includes(global.name)
  )) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    const access = `${context.nameFor(global.name)}[index]`;
    lines.push(`    case ${id}u: { return ${emitAtomicAddAtAddress(type, "storage", `&${access}`, "value")}; }`);
  }
  lines.push(`    default: { return ${zeroValue(type)}; }`);
  lines.push("  }");
  lines.push("}");
  return lines;
}

function emitDevicePointerAtomicSubHelper(type: "float" | "double" | "int" | "uint", ir: KernelIrModule, context: WgslPointerHelperContext): string[] {
  return emitDevicePointerAtomicRmwHelper("Sub", type, ir, context);
}

function emitDevicePointerAtomicMinHelper(type: "float" | "double" | "int" | "uint", ir: KernelIrModule, context: WgslPointerHelperContext): string[] {
  return emitDevicePointerAtomicRmwHelper("Min", type, ir, context);
}

function emitDevicePointerAtomicMaxHelper(type: "float" | "double" | "int" | "uint", ir: KernelIrModule, context: WgslPointerHelperContext): string[] {
  return emitDevicePointerAtomicRmwHelper("Max", type, ir, context);
}

function emitDevicePointerAtomicAndHelper(type: "int" | "uint", ir: KernelIrModule, context: WgslPointerHelperContext): string[] {
  return emitDevicePointerAtomicRmwHelper("And", type, ir, context);
}

function emitDevicePointerAtomicOrHelper(type: "int" | "uint", ir: KernelIrModule, context: WgslPointerHelperContext): string[] {
  return emitDevicePointerAtomicRmwHelper("Or", type, ir, context);
}

function emitDevicePointerAtomicXorHelper(type: "int" | "uint", ir: KernelIrModule, context: WgslPointerHelperContext): string[] {
  return emitDevicePointerAtomicRmwHelper("Xor", type, ir, context);
}

function emitDevicePointerAtomicIncHelper(type: "int" | "uint", ir: KernelIrModule, context: WgslPointerHelperContext): string[] {
  return emitDevicePointerAtomicIncDecHelper("Inc", type, ir, context);
}

function emitDevicePointerAtomicDecHelper(type: "int" | "uint", ir: KernelIrModule, context: WgslPointerHelperContext): string[] {
  return emitDevicePointerAtomicIncDecHelper("Dec", type, ir, context);
}

function emitDevicePointerAtomicIncDecHelper(
  kind: "Inc" | "Dec",
  type: "int" | "uint",
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
): string[] {
  const name = pointerAtomicIncDecHelperName(kind, type);
  const lines = [
    `fn ${name}(buffer: u32, index: u32, limit: u32) -> u32 {`,
    "  switch buffer {",
  ];
  for (const param of ir.params.filter((param) =>
    param.pointer &&
    !param.constant &&
    isPointerHelperCompatibleStorage(type, param.valueType) &&
    ir.atomicParams.includes(param.name)
  )) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    const access = `${context.nameFor(param.name)}[index]`;
    const helper = integerAtomicLoopHelperName(kind, {
      valueType: type,
      storageValueType: param.valueType,
      storageScalar: param.valueType === "int" ? "i32" : "u32",
      addressSpace: "storage",
    });
    lines.push(`    case ${id}u: { return ${helper}(&${access}, limit); }`);
  }
  for (const shared of ir.sharedDeclarations.filter((shared) =>
    isPointerHelperCompatibleStorage(type, shared.valueType) &&
    ir.atomicShared.includes(shared.name)
  )) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    const access = emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, "index");
    const helper = integerAtomicLoopHelperName(kind, {
      valueType: type,
      storageValueType: shared.valueType,
      storageScalar: shared.valueType === "int" ? "i32" : "u32",
      addressSpace: "workgroup",
    });
    lines.push(`    case ${id}u: { return ${helper}(&${access}, limit); }`);
  }
  for (const global of ir.deviceGlobals.filter((global) =>
    isPointerHelperCompatibleStorage(type, global.valueType) &&
    ir.atomicDeviceGlobals.includes(global.name)
  )) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    const access = `${context.nameFor(global.name)}[index]`;
    const helper = integerAtomicLoopHelperName(kind, {
      valueType: type,
      storageValueType: global.valueType,
      storageScalar: global.valueType === "int" ? "i32" : "u32",
      addressSpace: "storage",
    });
    lines.push(`    case ${id}u: { return ${helper}(&${access}, limit); }`);
  }
  lines.push("    default: { return 0u; }");
  lines.push("  }");
  lines.push("}");
  return lines;
}

function emitDevicePointerAtomicRmwHelper(
  kind: "Sub" | "Min" | "Max" | "And" | "Or" | "Xor",
  type: "float" | "double" | "int" | "uint",
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
): string[] {
  const scalar = wgslScalar(type);
  const name = pointerAtomicRmwHelperName(kind, type);
  const lines = [
    `fn ${name}(buffer: u32, index: u32, value: ${scalar}) -> ${scalar} {`,
    "  switch buffer {",
  ];
  for (const param of ir.params.filter((param) =>
    param.pointer &&
    !param.constant &&
    isPointerHelperCompatibleStorage(type, param.valueType) &&
    ir.atomicParams.includes(param.name)
  )) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    const access = `${context.nameFor(param.name)}[index]`;
    lines.push(`    case ${id}u: { return ${emitAtomicRmwAtAddress(kind, type, "storage", `&${access}`, "value")}; }`);
  }
  for (const shared of ir.sharedDeclarations.filter((shared) =>
    isPointerHelperCompatibleStorage(type, shared.valueType) &&
    ir.atomicShared.includes(shared.name)
  )) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    const access = emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, "index");
    lines.push(`    case ${id}u: { return ${emitAtomicRmwAtAddress(kind, type, "workgroup", `&${access}`, "value")}; }`);
  }
  for (const global of ir.deviceGlobals.filter((global) =>
    isPointerHelperCompatibleStorage(type, global.valueType) &&
    ir.atomicDeviceGlobals.includes(global.name)
  )) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    const access = `${context.nameFor(global.name)}[index]`;
    lines.push(`    case ${id}u: { return ${emitAtomicRmwAtAddress(kind, type, "storage", `&${access}`, "value")}; }`);
  }
  lines.push(`    default: { return ${zeroValue(type)}; }`);
  lines.push("  }");
  lines.push("}");
  return lines;
}

function emitDevicePointerAtomicExchangeHelper(
  type: "float" | "int" | "uint",
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
): string[] {
  const scalar = wgslScalar(type);
  const lines = [
    `fn ${pointerAtomicExchangeHelperName(type)}(buffer: u32, index: u32, value: ${scalar}) -> ${scalar} {`,
    "  switch buffer {",
  ];
  for (const param of ir.params.filter((param) =>
    param.pointer &&
    !param.constant &&
    isPointerHelperCompatibleStorage(type, param.valueType) &&
    ir.atomicParams.includes(param.name)
  )) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    const access = `${context.nameFor(param.name)}[index]`;
    lines.push(`    case ${id}u: { return ${emitAtomicExchangeAtAddress(type, `&${access}`, "value")}; }`);
  }
  for (const shared of ir.sharedDeclarations.filter((shared) =>
    isPointerHelperCompatibleStorage(type, shared.valueType) &&
    ir.atomicShared.includes(shared.name)
  )) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    const access = emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, "index");
    lines.push(`    case ${id}u: { return ${emitAtomicExchangeAtAddress(type, `&${access}`, "value")}; }`);
  }
  for (const global of ir.deviceGlobals.filter((global) =>
    isPointerHelperCompatibleStorage(type, global.valueType) &&
    ir.atomicDeviceGlobals.includes(global.name)
  )) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    const access = `${context.nameFor(global.name)}[index]`;
    lines.push(`    case ${id}u: { return ${emitAtomicExchangeAtAddress(type, `&${access}`, "value")}; }`);
  }
  lines.push(`    default: { return ${zeroValue(type)}; }`);
  lines.push("  }");
  lines.push("}");
  return lines;
}

function emitDevicePointerAtomicCasHelper(type: "int" | "uint", ir: KernelIrModule, context: WgslPointerHelperContext): string[] {
  const scalar = wgslScalar(type);
  const lines = [
    `fn ${pointerAtomicCasHelperName(type)}(buffer: u32, index: u32, compare: ${scalar}, value: ${scalar}) -> ${scalar} {`,
    "  switch buffer {",
  ];
  for (const param of ir.params.filter((param) =>
    param.pointer &&
    !param.constant &&
    isPointerHelperCompatibleStorage(type, param.valueType) &&
    ir.atomicParams.includes(param.name)
  )) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    const access = `${context.nameFor(param.name)}[index]`;
    lines.push(`    case ${id}u: { return atomicCompareExchangeWeak(&${access}, compare, value).old_value; }`);
  }
  for (const shared of ir.sharedDeclarations.filter((shared) =>
    isPointerHelperCompatibleStorage(type, shared.valueType) &&
    ir.atomicShared.includes(shared.name)
  )) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    const access = emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, "index");
    lines.push(`    case ${id}u: { return atomicCompareExchangeWeak(&${access}, compare, value).old_value; }`);
  }
  for (const global of ir.deviceGlobals.filter((global) =>
    isPointerHelperCompatibleStorage(type, global.valueType) &&
    ir.atomicDeviceGlobals.includes(global.name)
  )) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    const access = `${context.nameFor(global.name)}[index]`;
    lines.push(`    case ${id}u: { return atomicCompareExchangeWeak(&${access}, compare, value).old_value; }`);
  }
  lines.push(`    default: { return ${zeroValue(type)}; }`);
  lines.push("  }");
  lines.push("}");
  return lines;
}

function emitAtomicAddAtAddress(type: "float" | "double" | "int" | "uint", addressSpace: "storage" | "workgroup", address: string, value: string): string {
  return type === "float" || type === "double"
    ? `${floatAtomicHelperName("Add", addressSpace)}(${address}, ${value})`
    : `atomicAdd(${address}, ${value})`;
}

function emitAtomicRmwAtAddress(
  kind: "Sub" | "Min" | "Max" | "And" | "Or" | "Xor",
  type: "float" | "double" | "int" | "uint",
  addressSpace: "storage" | "workgroup",
  address: string,
  value: string,
): string {
  if (type === "float" || type === "double") {
    if (kind !== "Sub" && kind !== "Min" && kind !== "Max") return zeroValue(type);
    return `${floatAtomicHelperName(kind, addressSpace)}(${address}, ${value})`;
  }
  return `atomic${kind}(${address}, ${value})`;
}

function emitAtomicExchangeAtAddress(type: "float" | "int" | "uint", address: string, value: string): string {
  return type === "float"
    ? `bitcast<f32>(atomicExchange(${address}, bitcast<u32>(${value})))`
    : `atomicExchange(${address}, ${value})`;
}

function isPointerHelperCompatibleStorage(helperType: CudaLiteScalarType, storageType: CudaLiteScalarType): boolean {
  if (helperType === storageType) return true;
  if (helperType === "float" && storageType === "double") return true;
  if (isCudaVectorType(storageType) && helperType === cudaVectorScalarType(storageType)) return true;
  return isCudaVectorType(helperType) && cudaVectorScalarType(helperType) === storageType;
}

function isPointerHelperReadableStorage(helperType: CudaLiteScalarType, storageType: CudaLiteScalarType): boolean {
  return isPointerHelperCompatibleStorage(helperType, storageType) ||
    (isCudaVectorType(storageType) && helperType === cudaVectorScalarType(storageType)) ||
    ((helperType === "uint" || helperType === "int") && (storageType === "float" || storageType === "double" || storageType === "uchar"));
}

function isPointerHelperBitcastCompatibleStorage(helperType: CudaLiteScalarType, storageType: CudaLiteScalarType): boolean {
  const helperScalar = isCudaVectorType(helperType) ? cudaVectorScalarType(helperType) : helperType;
  return (helperScalar === "float" || helperScalar === "int" || helperScalar === "uint" || helperScalar === "uchar") &&
    (storageType === "float" || storageType === "double" || storageType === "int" || storageType === "uint" || storageType === "uchar");
}

export function pointerReadHelperName(type: CudaLiteScalarType): string {
  return `bg_ptr_read_${pointerHelperTypeName(type)}`;
}

export function pointerWriteHelperName(type: CudaLiteScalarType): string {
  return `bg_ptr_write_${pointerHelperTypeName(type)}`;
}

export function pointerAtomicAddHelperName(type: "float" | "double" | "int" | "uint"): string {
  return `bg_ptr_atomicAdd_${pointerHelperTypeName(type)}`;
}

export function pointerAtomicRmwHelperName(kind: "Sub" | "Min" | "Max" | "And" | "Or" | "Xor", type: "float" | "double" | "int" | "uint"): string {
  return `bg_ptr_atomic${kind}_${pointerHelperTypeName(type)}`;
}

export function pointerAtomicIncDecHelperName(kind: "Inc" | "Dec", type: "int" | "uint"): string {
  return `bg_ptr_atomic${kind}_${pointerHelperTypeName(type)}`;
}

export function pointerAtomicExchangeHelperName(type: "float" | "int" | "uint"): string {
  return `bg_ptr_atomicExchange_${pointerHelperTypeName(type)}`;
}

export function pointerAtomicCasHelperName(type: "int" | "uint"): string {
  return `bg_ptr_atomicCompareExchange_${pointerHelperTypeName(type)}`;
}

export function pointerHelperTypeName(type: CudaLiteScalarType): string {
  if (isCudaVectorType(type)) {
    const scalar = cudaVectorScalarType(type) ?? "float";
    return `${scalar === "float" || scalar === "bf16" ? "f32" : scalar === "int" ? "i32" : scalar === "half" ? "f16" : "u32"}x${cudaVectorLaneCount(type)}`;
  }
  if (type === "float") return "f32";
  if (type === "double") return "f32";
  if (type === "half") return "f16";
  if (type === "bf16") return "bf16";
  if (type === "int") return "i32";
  if (type === "uint") return "u32";
  if (type === "uchar") return "u8";
  return type;
}

function statementsUseCall(statements: readonly CudaLiteStatement[], names: ReadonlySet<string>): boolean {
  let used = false;
  walkCudaLiteExpressions(statements, (expression) => {
    if (expression.kind === "call" && names.has(expressionName(expression.callee) ?? "")) used = true;
  });
  return used;
}

function isDevicePoolParam(param: { readonly valueType: CudaLiteScalarType }): boolean {
  return param.valueType === "devicepool";
}

function featureError(code: string, message: string): CudaLiteCompilerError {
  const span: SourceSpan = { start: 0, end: 0, line: 1, column: 1 };
  return new CudaLiteCompilerError(message, [{ code, severity: "error", message, span }]);
}
