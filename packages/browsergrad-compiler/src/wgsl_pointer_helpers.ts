import { walkCudaLiteExpressions } from "./ast_queries.js";
import { expressionName } from "./analyzer.js";
import {
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
} from "./vector_types.js";
import {
  bfloatAtomicAddHelperName,
  floatAtomicHelperName,
  intViewAtomicCasHelperName,
  intViewAtomicHelperName,
  integerAtomicLoopHelperName,
  type WgslIntViewAtomicKind,
} from "./wgsl_atomic_helpers.js";
import {
  emitConstantVectorFlatRead,
  emitConstantPointerRead,
  emitDeviceGlobalVectorFlatRead,
  emitDeviceGlobalVectorFlatWrite,
  emitDeviceGlobalPointerRead,
  emitDeviceGlobalPointerWrite,
  emitPointerVectorFlatRead,
  emitPointerVectorFlatWrite,
  emitPointerStorageRead,
  emitPointerStorageWrite,
  emitSharedFlatAccess,
  emitSharedVectorFlatRead,
  emitSharedVectorFlatWrite,
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
    lines.push(`    case ${id}u: { return ${emitPointerHelperStorageRead(param, "index", type, ir, context)}; }`);
  }
  for (const shared of sharedDeclarations) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    if (options.reachableBufferIds && !options.reachableBufferIds.has(id)) continue;
    lines.push(`    case ${id}u: { return ${emitSharedPointerHelperRead(shared, "index", type, ir, context)}; }`);
  }
  for (const global of deviceGlobals) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    if (options.reachableBufferIds && !options.reachableBufferIds.has(id)) continue;
    lines.push(`    case ${id}u: { return ${emitDevicePointerHelperGlobalRead(global, "index", type, ir, context)}; }`);
  }
  for (const constant of constantArrays) {
    const id = context.constantPointerIdFor(constant.name);
    if (id === undefined) continue;
    if (options.reachableBufferIds && !options.reachableBufferIds.has(id)) continue;
    lines.push(`    case ${id}u: { return ${emitDevicePointerHelperConstantRead(constant, "index", type, context)}; }`);
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
    lines.push(`    case ${id}u: { ${emitPointerHelperStorageWrite(param, "index", "value", type, ir, context)}; return; }`);
  }
  for (const shared of sharedDeclarations) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    if (options.reachableBufferIds && !options.reachableBufferIds.has(id)) continue;
    lines.push(`    case ${id}u: { ${emitSharedPointerHelperWrite(shared, "index", "value", type, ir, context)}; return; }`);
  }
  for (const global of deviceGlobals) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    if (options.reachableBufferIds && !options.reachableBufferIds.has(id)) continue;
    lines.push(`    case ${id}u: { ${emitDevicePointerHelperGlobalWrite(global, "index", "value", type, ir, context)}; return; }`);
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
    if (context.storagePointerIdFor(param.name) === bufferId) return emitPointerHelperStorageRead(param, "index", type, ir, context);
  }
  for (const shared of sharedDeclarations) {
    if (context.sharedPointerIdFor(shared.name) === bufferId) return emitSharedPointerHelperRead(shared, "index", type, ir, context);
  }
  for (const global of deviceGlobals) {
    if (context.deviceGlobalPointerIdFor(global.name) === bufferId) return emitDevicePointerHelperGlobalRead(global, "index", type, ir, context);
  }
  for (const constant of constantArrays) {
    if (context.constantPointerIdFor(constant.name) === bufferId) return emitDevicePointerHelperConstantRead(constant, "index", type, context);
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
    if (context.storagePointerIdFor(param.name) === bufferId) return emitPointerHelperStorageWrite(param, "index", "value", type, ir, context);
  }
  for (const shared of sharedDeclarations) {
    if (context.sharedPointerIdFor(shared.name) === bufferId) return emitSharedPointerHelperWrite(shared, "index", "value", type, ir, context);
  }
  for (const global of deviceGlobals) {
    if (context.deviceGlobalPointerIdFor(global.name) === bufferId) return emitDevicePointerHelperGlobalWrite(global, "index", "value", type, ir, context);
  }
  return undefined;
}

function emitPointerHelperStorageRead(
  param: CudaLiteParam,
  index: string,
  viewType: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
): string {
  return isCudaVectorType(viewType)
    ? emitPointerVectorFlatRead(param, index, viewType, ir, context)
    : emitPointerStorageRead(param, index, ir, context, viewType);
}

function emitPointerHelperStorageWrite(
  param: CudaLiteParam,
  index: string,
  value: string,
  viewType: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
): string {
  return isCudaVectorType(viewType)
    ? emitPointerVectorFlatWrite(param, index, value, viewType, ir, context)
    : emitPointerStorageWrite(param, index, value, ir, context, viewType);
}

function emitSharedPointerHelperRead(
  shared: CudaLiteVarDecl,
  index: string,
  viewType: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
): string {
  if (isCudaVectorType(viewType)) return emitSharedVectorFlatRead(shared, index, viewType, context);
  return emitSharedPointerRead(
    shared,
    sharedPointerHelperIndex(shared.valueType, viewType, index, context.ir.atomicShared.includes(shared.name)),
    ir,
    context,
    viewType,
    sharedPointerHelperLane(shared.valueType, viewType, index, context.ir.atomicShared.includes(shared.name)),
  );
}

function emitSharedPointerHelperWrite(
  shared: CudaLiteVarDecl,
  index: string,
  value: string,
  viewType: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
): string {
  if (isCudaVectorType(viewType)) return emitSharedVectorFlatWrite(shared, index, value, viewType, context);
  return emitSharedPointerWrite(
    shared,
    sharedPointerHelperIndex(shared.valueType, viewType, index, context.ir.atomicShared.includes(shared.name)),
    value,
    ir,
    context,
    viewType,
    sharedPointerHelperLane(shared.valueType, viewType, index, context.ir.atomicShared.includes(shared.name)),
  );
}

function emitDevicePointerHelperGlobalRead(
  global: CudaLiteDeviceGlobal,
  index: string,
  viewType: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
): string {
  return isCudaVectorType(viewType)
    ? emitDeviceGlobalVectorFlatRead(global, index, viewType, ir, context)
    : emitDeviceGlobalPointerRead(global, index, ir, context, viewType);
}

function emitDevicePointerHelperGlobalWrite(
  global: CudaLiteDeviceGlobal,
  index: string,
  value: string,
  viewType: CudaLiteScalarType,
  ir: KernelIrModule,
  context: WgslPointerHelperContext,
): string {
  return isCudaVectorType(viewType)
    ? emitDeviceGlobalVectorFlatWrite(global, index, value, viewType, ir, context)
    : emitDeviceGlobalPointerWrite(global, index, value, ir, context, viewType);
}

function emitDevicePointerHelperConstantRead(
  constant: CudaLiteGlobalConstant,
  index: string,
  viewType: CudaLiteScalarType,
  context: WgslPointerHelperContext,
): string {
  return isCudaVectorType(viewType)
    ? emitConstantVectorFlatRead(constant, index, viewType, context)
    : emitConstantPointerRead(constant, index, context, viewType);
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

export function isDevicePointerAtomicAddType(type: CudaLiteScalarType): type is "float" | "double" | "bf16" | "int" | "uint" {
  return type === "float" || type === "double" || type === "bf16" || type === "int" || type === "uint";
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

export function isDevicePointerAtomicCasType(type: CudaLiteScalarType): type is "float" | "double" | "int" | "uint" {
  return type === "float" || type === "double" || type === "int" || type === "uint";
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
  type: "float" | "double" | "bf16" | "int" | "uint",
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
    isPointerHelperAtomicStorage(type, param.valueType) &&
    ir.atomicParams.includes(param.name)
  )) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    const target = pointerHelperAtomicTarget(type, context.nameFor(param.name), param.valueType, "index", "storage");
    if (!target) continue;
    lines.push(`    case ${id}u: { return ${emitAtomicAddAtAddress(type, target, "value")}; }`);
  }
  for (const shared of ir.sharedDeclarations.filter((shared) =>
    isPointerHelperAtomicStorage(type, shared.valueType) &&
    ir.atomicShared.includes(shared.name)
  )) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    const target = pointerHelperSharedAtomicTarget(type, shared, "index", context);
    if (!target) continue;
    lines.push(`    case ${id}u: { return ${emitAtomicAddAtAddress(type, target, "value")}; }`);
  }
  for (const global of ir.deviceGlobals.filter((global) =>
    isPointerHelperAtomicStorage(type, global.valueType) &&
    ir.atomicDeviceGlobals.includes(global.name)
  )) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    const target = pointerHelperAtomicTarget(type, context.nameFor(global.name), global.valueType, "index", "storage");
    if (!target) continue;
    lines.push(`    case ${id}u: { return ${emitAtomicAddAtAddress(type, target, "value")}; }`);
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
    isPointerHelperAtomicStorage(type, param.valueType) &&
    ir.atomicParams.includes(param.name)
  )) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    const target = pointerHelperAtomicTarget(type, context.nameFor(param.name), param.valueType, "index", "storage");
    if (!target) continue;
    const helper = integerAtomicLoopHelperName(kind, {
      valueType: type,
      storageValueType: target.storageValueType,
      storageScalar: target.storageScalar,
      addressSpace: target.addressSpace,
    });
    lines.push(`    case ${id}u: { return ${helper}(${target.address}, limit); }`);
  }
  for (const shared of ir.sharedDeclarations.filter((shared) =>
    isPointerHelperAtomicStorage(type, shared.valueType) &&
    ir.atomicShared.includes(shared.name)
  )) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    const target = pointerHelperSharedAtomicTarget(type, shared, "index", context);
    if (!target) continue;
    const helper = integerAtomicLoopHelperName(kind, {
      valueType: type,
      storageValueType: target.storageValueType,
      storageScalar: target.storageScalar,
      addressSpace: target.addressSpace,
    });
    lines.push(`    case ${id}u: { return ${helper}(${target.address}, limit); }`);
  }
  for (const global of ir.deviceGlobals.filter((global) =>
    isPointerHelperAtomicStorage(type, global.valueType) &&
    ir.atomicDeviceGlobals.includes(global.name)
  )) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    const target = pointerHelperAtomicTarget(type, context.nameFor(global.name), global.valueType, "index", "storage");
    if (!target) continue;
    const helper = integerAtomicLoopHelperName(kind, {
      valueType: type,
      storageValueType: target.storageValueType,
      storageScalar: target.storageScalar,
      addressSpace: target.addressSpace,
    });
    lines.push(`    case ${id}u: { return ${helper}(${target.address}, limit); }`);
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
    isPointerHelperAtomicStorage(type, param.valueType) &&
    ir.atomicParams.includes(param.name)
  )) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    const target = pointerHelperAtomicTarget(type, context.nameFor(param.name), param.valueType, "index", "storage");
    if (!target) continue;
    lines.push(`    case ${id}u: { return ${emitAtomicRmwAtAddress(kind, type, target, "value")}; }`);
  }
  for (const shared of ir.sharedDeclarations.filter((shared) =>
    isPointerHelperAtomicStorage(type, shared.valueType) &&
    ir.atomicShared.includes(shared.name)
  )) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    const target = pointerHelperSharedAtomicTarget(type, shared, "index", context);
    if (!target) continue;
    lines.push(`    case ${id}u: { return ${emitAtomicRmwAtAddress(kind, type, target, "value")}; }`);
  }
  for (const global of ir.deviceGlobals.filter((global) =>
    isPointerHelperAtomicStorage(type, global.valueType) &&
    ir.atomicDeviceGlobals.includes(global.name)
  )) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    const target = pointerHelperAtomicTarget(type, context.nameFor(global.name), global.valueType, "index", "storage");
    if (!target) continue;
    lines.push(`    case ${id}u: { return ${emitAtomicRmwAtAddress(kind, type, target, "value")}; }`);
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
    isPointerHelperAtomicStorage(type, param.valueType) &&
    ir.atomicParams.includes(param.name)
  )) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    const target = pointerHelperAtomicTarget(type, context.nameFor(param.name), param.valueType, "index", "storage");
    if (!target) continue;
    lines.push(`    case ${id}u: { return ${emitAtomicExchangeAtAddress(type, target, "value")}; }`);
  }
  for (const shared of ir.sharedDeclarations.filter((shared) =>
    isPointerHelperAtomicStorage(type, shared.valueType) &&
    ir.atomicShared.includes(shared.name)
  )) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    const target = pointerHelperSharedAtomicTarget(type, shared, "index", context);
    if (!target) continue;
    lines.push(`    case ${id}u: { return ${emitAtomicExchangeAtAddress(type, target, "value")}; }`);
  }
  for (const global of ir.deviceGlobals.filter((global) =>
    isPointerHelperAtomicStorage(type, global.valueType) &&
    ir.atomicDeviceGlobals.includes(global.name)
  )) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    const target = pointerHelperAtomicTarget(type, context.nameFor(global.name), global.valueType, "index", "storage");
    if (!target) continue;
    lines.push(`    case ${id}u: { return ${emitAtomicExchangeAtAddress(type, target, "value")}; }`);
  }
  lines.push(`    default: { return ${zeroValue(type)}; }`);
  lines.push("  }");
  lines.push("}");
  return lines;
}

function emitDevicePointerAtomicCasHelper(type: "float" | "double" | "int" | "uint", ir: KernelIrModule, context: WgslPointerHelperContext): string[] {
  const scalar = wgslScalar(type);
  const lines = [
    `fn ${pointerAtomicCasHelperName(type)}(buffer: u32, index: u32, compare: ${scalar}, value: ${scalar}) -> ${scalar} {`,
    "  switch buffer {",
  ];
  for (const param of ir.params.filter((param) =>
    param.pointer &&
    !param.constant &&
    isPointerHelperAtomicStorage(type, param.valueType) &&
    ir.atomicParams.includes(param.name)
  )) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    const target = pointerHelperAtomicTarget(type, context.nameFor(param.name), param.valueType, "index", "storage");
    if (!target) continue;
    lines.push(`    case ${id}u: { return ${emitAtomicCasAtAddress(type, target, "compare", "value")}; }`);
  }
  for (const shared of ir.sharedDeclarations.filter((shared) =>
    isPointerHelperAtomicStorage(type, shared.valueType) &&
    ir.atomicShared.includes(shared.name)
  )) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    const target = pointerHelperSharedAtomicTarget(type, shared, "index", context);
    if (!target) continue;
    lines.push(`    case ${id}u: { return ${emitAtomicCasAtAddress(type, target, "compare", "value")}; }`);
  }
  for (const global of ir.deviceGlobals.filter((global) =>
    isPointerHelperAtomicStorage(type, global.valueType) &&
    ir.atomicDeviceGlobals.includes(global.name)
  )) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    const target = pointerHelperAtomicTarget(type, context.nameFor(global.name), global.valueType, "index", "storage");
    if (!target) continue;
    lines.push(`    case ${id}u: { return ${emitAtomicCasAtAddress(type, target, "compare", "value")}; }`);
  }
  lines.push(`    default: { return ${zeroValue(type)}; }`);
  lines.push("  }");
  lines.push("}");
  return lines;
}

function emitAtomicAddAtAddress(type: "float" | "double" | "bf16" | "int" | "uint", target: PointerHelperAtomicTarget, value: string): string {
  if (type === "bf16") return `${bfloatAtomicAddHelperName(target.addressSpace)}(${target.address}, ${value})`;
  if (type === "float" || type === "double") return `${floatAtomicHelperName("Add", target.addressSpace)}(${target.address}, ${value})`;
  if (usesIntViewAtomicHelper(type, target)) return `${intViewAtomicHelperName("Add", target.addressSpace)}(${target.address}, ${value})`;
  return `atomicAdd(${target.address}, ${value})`;
}

function emitAtomicRmwAtAddress(
  kind: "Sub" | "Min" | "Max" | "And" | "Or" | "Xor",
  type: "float" | "double" | "int" | "uint",
  target: PointerHelperAtomicTarget,
  value: string,
): string {
  if (type === "float" || type === "double") {
    if (kind !== "Sub" && kind !== "Min" && kind !== "Max") return zeroValue(type);
    return `${floatAtomicHelperName(kind, target.addressSpace)}(${target.address}, ${value})`;
  }
  if (usesIntViewAtomicHelper(type, target)) return `${intViewAtomicHelperName(kind as WgslIntViewAtomicKind, target.addressSpace)}(${target.address}, ${value})`;
  return `atomic${kind}(${target.address}, ${value})`;
}

function emitAtomicExchangeAtAddress(type: "float" | "int" | "uint", target: PointerHelperAtomicTarget, value: string): string {
  if (type === "float") return `bitcast<f32>(atomicExchange(${target.address}, bitcast<u32>(${value})))`;
  if (usesIntViewAtomicHelper(type, target)) return `${intViewAtomicHelperName("Exchange", target.addressSpace)}(${target.address}, ${value})`;
  return `atomicExchange(${target.address}, ${value})`;
}

function emitAtomicCasAtAddress(type: "float" | "double" | "int" | "uint", target: PointerHelperAtomicTarget, compare: string, value: string): string {
  if (type === "float" || type === "double") return `bitcast<f32>(atomicCompareExchangeWeak(${target.address}, bitcast<u32>(${compare}), bitcast<u32>(${value})).old_value)`;
  if (usesIntViewAtomicHelper(type, target)) return `${intViewAtomicCasHelperName(target.addressSpace)}(${target.address}, ${compare}, ${value})`;
  return `atomicCompareExchangeWeak(${target.address}, ${compare}, ${value}).old_value`;
}

function usesIntViewAtomicHelper(type: CudaLiteScalarType, target: PointerHelperAtomicTarget): boolean {
  return type === "int" && target.storageScalar === "u32";
}

function pointerHelperAtomicStorageScalar(storageType: CudaLiteScalarType): "i32" | "u32" {
  return (cudaVectorScalarType(storageType) ?? storageType) === "int" ? "i32" : "u32";
}

interface PointerHelperAtomicTarget {
  readonly address: string;
  readonly storageValueType: CudaLiteScalarType;
  readonly storageScalar: "i32" | "u32";
  readonly addressSpace: "storage" | "workgroup";
}

function pointerHelperAtomicTarget(
  helperType: CudaLiteScalarType,
  name: string,
  storageType: CudaLiteScalarType,
  index: string,
  addressSpace: "storage" | "workgroup",
): PointerHelperAtomicTarget | undefined {
  if (isPointerHelperCompatibleStorage(helperType, storageType)) {
    return {
      address: `&${name}[${index}]`,
      storageValueType: storageType,
      storageScalar: pointerHelperAtomicStorageScalar(storageType),
      addressSpace,
    };
  }
  if (isPointerHelperPackedByteAtomicStorage(helperType, storageType)) {
    return {
      address: `&${name}[(u32(${index})) >> 2u]`,
      storageValueType: storageType,
      storageScalar: "u32",
      addressSpace,
    };
  }
  return undefined;
}

function pointerHelperSharedAtomicTarget(
  helperType: CudaLiteScalarType,
  shared: CudaLiteVarDecl,
  index: string,
  context: WgslPointerHelperContext,
): PointerHelperAtomicTarget | undefined {
  if (!isPointerHelperAtomicStorage(helperType, shared.valueType)) return undefined;
  const storageIndex = isPointerHelperPackedByteAtomicStorage(helperType, shared.valueType)
    ? `(u32(${index})) >> 2u`
    : index;
  return {
    address: `&${emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, storageIndex)}`,
    storageValueType: shared.valueType,
    storageScalar: pointerHelperAtomicStorageScalar(shared.valueType),
    addressSpace: "workgroup",
  };
}

function isPointerHelperAtomicStorage(helperType: CudaLiteScalarType, storageType: CudaLiteScalarType): boolean {
  return isPointerHelperCompatibleStorage(helperType, storageType) ||
    isPointerHelperPackedByteAtomicStorage(helperType, storageType);
}

function isPointerHelperPackedByteAtomicStorage(helperType: CudaLiteScalarType, storageType: CudaLiteScalarType): boolean {
  return storageType === "uchar" && (helperType === "int" || helperType === "uint" || helperType === "float");
}

function isPointerHelperCompatibleStorage(helperType: CudaLiteScalarType, storageType: CudaLiteScalarType): boolean {
  if (helperType === storageType) return true;
  if (helperType === "bf16" && storageType === "bf16") return true;
  if (helperType === "float" && storageType === "double") return true;
  if (isCudaVectorType(storageType) && helperType === cudaVectorScalarType(storageType)) return true;
  return isCudaVectorType(helperType) && cudaVectorScalarType(helperType) === storageType;
}

function isPointerHelperReadableStorage(helperType: CudaLiteScalarType, storageType: CudaLiteScalarType): boolean {
  return isPointerHelperCompatibleStorage(helperType, storageType) ||
    (isCudaVectorType(storageType) && helperType === cudaVectorScalarType(storageType)) ||
    (storageType === "uchar" && isPackedBytePointerHelperType(helperType)) ||
    ((helperType === "uint" || helperType === "int") && (storageType === "float" || storageType === "double" || storageType === "uchar"));
}

function isPackedBytePointerHelperType(type: CudaLiteScalarType): boolean {
  const scalar = isCudaVectorType(type) ? cudaVectorScalarType(type) : type;
  return scalar === "float" ||
    scalar === "double" ||
    scalar === "int" ||
    scalar === "uint" ||
    scalar === "half" ||
    scalar === "bf16" ||
    scalar === "bool";
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

export function pointerAtomicAddHelperName(type: "float" | "double" | "bf16" | "int" | "uint"): string {
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

export function pointerAtomicCasHelperName(type: "float" | "double" | "int" | "uint"): string {
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
