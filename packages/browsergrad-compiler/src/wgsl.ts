import {
  defineWgslKernelProgram,
  type WgslKernelBindingInput,
  type WgslKernelProgram,
} from "@unlocalhosted/browsergrad-kernels";
import { collectExternalDevicePoolNames, collectKernelLaunchCallees, walkCudaLiteExpressions } from "./ast_queries.js";
import { expressionName, rootIdentifier } from "./analyzer.js";
import { CUDA_CACHE_HINT_LOADS, CUDA_CACHE_HINT_STORES, CUDA_INTRINSICS_BY_NAME } from "./intrinsics.js";
import {
  type MatrixTileLayout,
  type MatrixTileResolvedSpec,
  isMatrixTileByteValueType,
  matrixTileElementCount,
  matrixTileReference,
  matrixTileStorageDimensions,
  normalizeMatrixTileLayout,
  resolveMatrixTileSpec,
  wmmaBuiltinName,
} from "./matrix_tiles.js";
import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";
import { pointerBaseOffsetUniformName } from "./pointer_offsets.js";
import { poolDataName, poolOffsetName } from "./pool_bindings.js";
import { classifyInlineAsm, inlineAsmSupportedList } from "./ptx_tile_ops.js";
import { alignofCudaType, sizeofCudaType } from "./type_layout.js";
import {
  CUDA_VECTOR_TYPES,
  cudaVectorConstructorType,
  cudaVectorFieldIndex,
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
  type CudaLiteVectorType,
} from "./vector_types.js";
import {
  CudaLiteCompilerError,
  type CudaLiteAssignmentExpression,
  type CudaLiteCallExpression,
  type CudaLiteCooperativeGroupDecl,
  type CudaLiteDeviceFunction,
  type CudaLiteExpression,
  type CudaLiteFeatureOptions,
  type CudaLiteDeviceGlobal,
  type CudaLiteGlobalConstant,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type CudaLiteStatement,
  type CudaLiteVarDecl,
  type KernelIrModule,
  type SourceSpan,
} from "./types.js";

const NULL_DEVICE_POINTER_BUFFER = "4294967295u";

export interface EmitKernelIrWgslOptions {
  readonly features?: CudaLiteFeatureOptions;
  readonly pointerBaseOffsets?: Readonly<Record<string, number>>;
}

export interface KernelIrWgslOutput {
  readonly wgsl: string;
  readonly program: WgslKernelProgram;
}

export function emitKernelIrWgsl(
  ir: KernelIrModule,
  options: EmitKernelIrWgslOptions = {},
): KernelIrWgslOutput {
  if (ir.requiredFeatures.includes("shader-f16") && !options.features?.["shader-f16"]) {
    throw featureError("missing-feature-shader-f16", "half requires WebGPU shader-f16 support");
  }
  if (ir.requiredFeatures.includes("subgroups") && !options.features?.subgroups) {
    throw featureError("missing-feature-subgroups", "bg_subgroup_add requires WebGPU subgroups support");
  }

  const context = createEmitContext(ir, options);
  const textures = textureBindings(ir);
  const lines: string[] = [];
  if (ir.requiredFeatures.includes("shader-f16")) lines.push("enable f16;");
  if (ir.requiredFeatures.includes("subgroups")) lines.push("enable subgroups;");
  if (lines.length > 0) lines.push("");
  lines.push(`// BrowserGrad CUDA-lite kernel: ${ir.name}`);

  for (const param of ir.params.filter((param) => param.pointer && !isDevicePoolParam(param) && !isSurfaceParam(param) && !isTextureParam(param))) {
    const access = param.constant ? "read" : "read_write";
    const element = storageElementType(param, ir);
    lines.push(
      `@group(0) @binding(${context.bindingFor(param.name)}) var<storage, ${access}> ${context.nameFor(param.name)}: array<${element}>;`,
    );
  }
  for (const global of ir.deviceGlobals) {
    const element = deviceGlobalStorageElementType(global, ir);
    lines.push(
      `@group(0) @binding(${context.bindingFor(global.name)}) var<storage, read_write> ${context.nameFor(global.name)}: array<${element}>;`,
    );
  }
  for (const surface of ir.params.filter(isSurfaceParam)) {
    lines.push(
      `@group(0) @binding(${context.bindingFor(surface.name)}) var<storage, read_write> ${context.nameFor(surface.name)}: array<f32>;`,
    );
  }
  for (const pool of ir.params.filter(isDevicePoolParam)) {
    const dataName = poolDataName(pool.name);
    const offsetName = poolOffsetName(pool.name);
    lines.push(
      `@group(0) @binding(${context.bindingFor(dataName)}) var<storage, read_write> ${context.nameFor(dataName)}: array<u32>;`,
    );
    lines.push(
      `@group(0) @binding(${context.bindingFor(offsetName)}) var<storage, read_write> ${context.nameFor(offsetName)}: atomic<u32>;`,
    );
  }
  for (const poolName of context.externalPoolNames) {
    const dataName = poolDataName(poolName);
    const offsetName = poolOffsetName(poolName);
    lines.push(
      `@group(0) @binding(${context.bindingFor(dataName)}) var<storage, read_write> ${context.nameFor(dataName)}: array<u32>;`,
    );
    lines.push(
      `@group(0) @binding(${context.bindingFor(offsetName)}) var<storage, read_write> ${context.nameFor(offsetName)}: atomic<u32>;`,
    );
  }

  for (const constant of ir.constants.filter(isExternalConstantBinding)) {
    lines.push(
      `@group(0) @binding(${context.bindingFor(constant.name)}) var<storage, read> ${context.nameFor(constant.name)}: ${emitConstantArrayType(constant)};`,
    );
  }
  for (const constant of ir.constants.filter((constant) => constant.init !== undefined)) {
    lines.push(emitInitializedConstant(constant, context));
  }

  for (const texture of textures) {
    lines.push(`@group(0) @binding(${context.bindingFor(texture.name)}) var ${context.nameFor(texture.name)}: texture_2d<f32>;`);
  }

  const uniformScalars = context.uniformScalars;
  if (uniformScalars.length > 0) {
    lines.push("struct Params {");
    for (const scalar of uniformScalars) {
      const align = scalar.valueType === "half" ? "@align(4) " : "";
      lines.push(`  ${align}${context.nameFor(scalar.name)}: ${wgslUniformScalar(scalar.valueType)},`);
    }
    lines.push("};");
    lines.push(`@group(0) @binding(${context.paramsBinding}) var<uniform> params: Params;`);
  }

  for (const shared of ir.sharedDeclarations) {
    lines.push(`var<workgroup> ${context.nameFor(shared.name)}: ${emitSharedType(shared, ir)};`);
  }

  if (textures.length > 0) {
    lines.push("");
    lines.push(...emitCubeTextureAtlasHelpers());
    lines.push("");
    for (const texture of textures) lines.push(...emitTextureHelper(texture.name, context));
  }
  for (const surface of ir.params.filter(isSurfaceParam)) {
    lines.push("");
    lines.push(...emitSurfaceHelper(surface.name, context));
  }
  for (const pool of ir.params.filter(isDevicePoolParam)) {
    lines.push("");
    lines.push(...emitPoolHelper(pool.name, context));
  }
  for (const poolName of context.externalPoolNames) {
    lines.push("");
    lines.push(...emitPoolHelper(poolName, context));
  }
  for (const allocator of context.rawPoolAllocators) {
    lines.push("");
    lines.push(...emitRawPoolHelper(allocator));
  }
  if (usesFloatAtomicAdd(ir)) {
    lines.push("");
    lines.push(...emitFloatAtomicAddHelper("storage"));
  }
  if (usesSharedFloatAtomicAdd(ir)) {
    lines.push("");
    lines.push(...emitFloatAtomicAddHelper("workgroup"));
  }
  if (usesFloatAtomicSub(ir)) {
    lines.push("");
    lines.push(...emitFloatAtomicSubHelper("storage"));
  }
  if (usesSharedFloatAtomicSub(ir)) {
    lines.push("");
    lines.push(...emitFloatAtomicSubHelper("workgroup"));
  }
  if (usesFloatAtomicMin(ir)) {
    lines.push("");
    lines.push(...emitFloatAtomicMinHelper("storage"));
  }
  if (usesSharedFloatAtomicMin(ir)) {
    lines.push("");
    lines.push(...emitFloatAtomicMinHelper("workgroup"));
  }
  if (usesFloatAtomicMax(ir)) {
    lines.push("");
    lines.push(...emitFloatAtomicMaxHelper("storage"));
  }
  if (usesSharedFloatAtomicMax(ir)) {
    lines.push("");
    lines.push(...emitFloatAtomicMaxHelper("workgroup"));
  }
  if (usesAtomicIncDec(ir)) {
    lines.push("");
    lines.push(...emitIntegerAtomicLoopHelpers());
  }
  if (usesCurand(ir)) {
    lines.push("");
    lines.push(...emitCurandHelpers());
  }
  if (usesFrexp(ir)) {
    lines.push("");
    lines.push(...emitFrexpHelpers());
  }
  if (usesSpecialFloatNamedConstants(ir)) {
    lines.push("");
    lines.push(...emitSpecialFloatConstantHelpers());
  }
  if (usesFp8Intrinsics(ir)) {
    lines.push("");
    lines.push(...emitFp8Helpers());
  }
  if (usesDevicePointerParams(ir)) {
    lines.push("");
    lines.push(...emitDevicePointerHelpers(ir, context));
  }

  const emittedFunctions = functionsToEmit(ir);
  const functionLines = emittedFunctions.flatMap((fn) => ["", ...emitDeviceFunction(fn, context)]);
  const mainBodyLines = ir.body.flatMap((statement) => emitStatement(statement, context, 1));
  for (const helper of context.vectorCooperativeReduceHelpers.values()) {
    lines.push("");
    lines.push(...emitVectorCooperativeReduceHelper(helper, context));
  }
  lines.push(...functionLines);

  lines.push("");
  lines.push(`@compute @workgroup_size(${ir.workgroupSize.join(", ")})`);
  lines.push("fn main(");
  lines.push("  @builtin(global_invocation_id) global_id: vec3<u32>,");
  lines.push("  @builtin(local_invocation_id) local_id: vec3<u32>,");
  lines.push("  @builtin(workgroup_id) workgroup_id: vec3<u32>,");
  lines.push("  @builtin(num_workgroups) num_workgroups: vec3<u32>");
  lines.push(") {");
  for (const name of context.mutablePointerBases) {
    const baseField = context.pointerBaseOffsetFieldFor(name);
    const baseName = context.mutablePointerBaseFor(name);
    if (baseName) lines.push(`  var ${baseName}: u32 = ${baseField ? `params.${context.nameFor(baseField)}` : "0u"};`);
  }
  for (const param of context.mutableScalarParams) {
    lines.push(`  var ${context.nameFor(param.name)}: ${wgslScalar(param.valueType)} = ${emitUniformScalarRead(param.name, context)};`);
  }
  lines.push(...mainBodyLines);
  lines.push("}");

  return {
    wgsl: lines.join("\n"),
    program: defineWgslKernelProgram({
      name: ir.name,
      wgsl: lines.join("\n"),
      bindings: context.bindings,
      workgroupSize: ir.workgroupSize,
    }),
  };
}

interface EmitContext {
  readonly ir: KernelIrModule;
  readonly bindings: readonly WgslKernelBindingInput[];
  readonly paramsBinding?: number;
  readonly uniformScalars: readonly { readonly name: string; readonly valueType: CudaLiteScalarType }[];
  readonly mutableScalarParams: readonly CudaLiteParam[];
  readonly deviceFunctionNames: ReadonlySet<string>;
  deviceFunctionFor(name: string, argCount?: number): CudaLiteDeviceFunction | undefined;
  devicePointerParamFor(name: string): CudaLiteParam | undefined;
  storagePointerIdFor(name: string): number | undefined;
  sharedPointerIdFor(name: string): number | undefined;
  constantPointerIdFor(name: string): number | undefined;
  deviceGlobalPointerIdFor(name: string): number | undefined;
  bindingFor(name: string): number;
  paramFor(name: string): CudaLiteParam | undefined;
  deviceGlobalFor(name: string): CudaLiteDeviceGlobal | undefined;
  isUniformScalar(name: string): boolean;
  uniformScalarTypeFor(name: string): CudaLiteScalarType | undefined;
  isAtomicShared(name: string): boolean;
  isAtomicDeviceGlobal(name: string): boolean;
  pointerAliasFor(name: string): PointerAlias | undefined;
  localPointerHandleFor(name: string): CudaLiteVarDecl | undefined;
  poolPointerFor(name: string): PoolPointerAlias | undefined;
  pointerBaseOffsetFieldFor(name: string): string | undefined;
  readonly rawPoolAllocators: readonly RawPoolAllocator[];
  readonly externalPoolNames: readonly string[];
  readonly mutablePointerBases: readonly string[];
  readonly vectorCooperativeReduceHelpers: Map<string, VectorCooperativeReduceHelper>;
  readonly expressionValueTypes: WeakMap<CudaLiteExpression, CudaLiteScalarType | undefined>;
  mutablePointerBaseFor(name: string): string | undefined;
  isLocalName(name: string): boolean;
  localValueTypeFor(name: string): CudaLiteScalarType | undefined;
  localArrayFor(name: string): CudaLiteVarDecl | undefined;
  cooperativeGroupFor(name: string): CudaLiteCooperativeGroupDecl | undefined;
  surfaceWidthField(name: string): string;
  nameFor(name: string): string;
}

interface VectorCooperativeReduceHelper {
  readonly key: string;
  readonly name: string;
  readonly opName: string;
  readonly valueType: CudaLiteVectorType;
  readonly tileSize: number;
}

interface PointerAlias {
  readonly rootName: string;
  readonly baseIndex: CudaLiteExpression;
  readonly valueType?: CudaLiteScalarType;
}

interface PoolPointerAlias {
  readonly poolName: string;
  readonly offsetName?: string;
  readonly rawBuffer?: boolean;
}

interface RawPoolAllocator {
  readonly baseName: string;
  readonly offsetName: string;
}

type EmitMode = "value" | "lvalue";

const WGSL_RESERVED_IDENTIFIERS = new Set([
  "alias",
  "array",
  "atomic",
  "bitcast",
  "bool",
  "break",
  "case",
  "const",
  "continue",
  "default",
  "discard",
  "else",
  "enable",
  "false",
  "fn",
  "for",
  "f16",
  "f32",
  "i32",
  "if",
  "let",
  "loop",
  "override",
  "return",
  "struct",
  "switch",
  "true",
  "u32",
  "var",
  "while",
  "main",
  "params",
  "global_id",
  "local_id",
  "workgroup_id",
  "num_workgroups",
]);

function createEmitContext(ir: KernelIrModule, options: EmitKernelIrWgslOptions = {}): EmitContext {
  const bindings: WgslKernelBindingInput[] = [];
  const bindingByName = new Map<string, number>();
  const storagePointerParams = ir.params.filter((param) => param.pointer && !isDevicePoolParam(param) && !isSurfaceParam(param) && !isTextureParam(param));
  const textures = textureBindings(ir);
  const storagePointerIds = new Map(storagePointerParams.map((param, index) => [param.name, index] as const));
  const deviceGlobalPointerIds = new Map(ir.deviceGlobals.map((global, index) => [global.name, storagePointerParams.length + index] as const));
  const constantPointerArrays = ir.constants.filter(isExternalConstantBinding);
  const sharedPointerIds = new Map(ir.sharedDeclarations
    .map((shared, index) => [shared.name, storagePointerParams.length + ir.deviceGlobals.length + index] as const));
  const constantPointerIds = new Map(constantPointerArrays
    .map((constant, index) => [constant.name, storagePointerParams.length + ir.deviceGlobals.length + ir.sharedDeclarations.length + index] as const));
  for (const param of storagePointerParams) {
    const binding = bindings.length;
    bindingByName.set(param.name, binding);
    bindings.push({
      kind: "storage",
      name: param.name,
      valueType: wgslBindingType(param.valueType),
      access: param.constant ? "read" : "read_write",
      binding,
    });
  }
  for (const global of ir.deviceGlobals) {
    const binding = bindings.length;
    bindingByName.set(global.name, binding);
    bindings.push({
      kind: "storage",
      name: global.name,
      valueType: wgslBindingType(global.valueType),
      access: "read_write",
      binding,
    });
  }
  for (const param of ir.params.filter(isSurfaceParam)) {
    const binding = bindings.length;
    bindingByName.set(param.name, binding);
    bindings.push({
      kind: "storage",
      name: param.name,
      valueType: "f32",
      access: "read_write",
      binding,
    });
  }
  for (const param of ir.params.filter(isDevicePoolParam)) {
    const dataBinding = bindings.length;
    bindingByName.set(poolDataName(param.name), dataBinding);
    bindings.push({
      kind: "storage",
      name: poolDataName(param.name),
      valueType: "u32",
      access: "read_write",
      binding: dataBinding,
    });
    const offsetBinding = bindings.length;
    bindingByName.set(poolOffsetName(param.name), offsetBinding);
    bindings.push({
      kind: "storage",
      name: poolOffsetName(param.name),
      valueType: "u32",
      access: "read_write",
      binding: offsetBinding,
    });
  }
  const externalPoolNames = collectExternalDevicePoolNames(
    ir.body,
    new Set(ir.params.filter(isDevicePoolParam).map((param) => param.name)),
  );
  for (const poolName of externalPoolNames) {
    const dataBinding = bindings.length;
    bindingByName.set(poolDataName(poolName), dataBinding);
    bindings.push({
      kind: "storage",
      name: poolDataName(poolName),
      valueType: "u32",
      access: "read_write",
      binding: dataBinding,
    });
    const offsetBinding = bindings.length;
    bindingByName.set(poolOffsetName(poolName), offsetBinding);
    bindings.push({
      kind: "storage",
      name: poolOffsetName(poolName),
      valueType: "u32",
      access: "read_write",
      binding: offsetBinding,
    });
  }
  for (const constant of constantPointerArrays) {
    const binding = bindings.length;
    bindingByName.set(constant.name, binding);
    bindings.push({
      kind: "storage",
      name: constant.name,
      valueType: wgslBindingType(constant.valueType),
      access: "read",
      binding,
    });
  }
  for (const texture of textures) {
    const binding = bindings.length;
    bindingByName.set(texture.name, binding);
    bindings.push({
      kind: "texture2d",
      name: texture.name,
      valueType: "f32",
      binding,
    });
  }
  const uniformScalars = [
    ...ir.params.filter((param) => !param.pointer && !isSurfaceParam(param) && !isTextureParam(param)).map((param) => ({ name: param.name, valueType: param.valueType })),
    ...ir.constants
      .filter((constant) => constant.dimensions.length === 0 && constant.init === undefined && !isCudaVectorType(constant.valueType))
      .map((constant) => ({ name: constant.name, valueType: constant.valueType })),
    ...ir.params.filter(isSurfaceParam).flatMap((param) => [
      { name: surfaceWidthField(param.name), valueType: "uint" as const },
      { name: surfaceHeightField(param.name), valueType: "uint" as const },
    ]),
    ...ir.params
      .filter((param) => param.pointer && options.pointerBaseOffsets?.[param.name] !== undefined)
      .map((param) => ({ name: pointerBaseOffsetUniformName(param.name), valueType: "uint" as const })),
  ];
  const uniformScalarNames = new Set(uniformScalars.map((scalar) => scalar.name));
  const uniformScalarTypes = new Map(uniformScalars.map((scalar) => [scalar.name, scalar.valueType] as const));
  const structuredPointerRoots = structuredPointerHandleRoots(ir);
  const localPointerHandles = collectLocalPointerHandles(ir.body, undefined, structuredPointerRoots);
  const pointerAliases = collectPointerAliases(ir.body, new Set(localPointerHandles.keys()));
  const mutablePointerBases = collectMutableStoragePointerBases(
    ir.body,
    new Set(storagePointerParams.map((param) => param.name)),
  );
  const poolPointers = collectPoolPointers(ir.body);
  const rawPoolAllocators = collectRawPoolAllocators(ir.body);
  const cooperativeGroups = collectCooperativeGroups(ir.body);
  const mutableScalarParams = collectMutableScalarParams(ir.body, ir.params);
  const localNames = new Set([
    ...collectLocalNames(ir.body),
    ...mutableScalarParams.map((param) => param.name),
  ]);
  const localValueTypes = new Map(collectLocalValueTypes(ir.body));
  for (const param of mutableScalarParams) localValueTypes.set(param.name, param.valueType);
  const localArrays = collectLocalArrays(ir.body);
  const wgslNames = createWgslNameMap(collectWgslDeclaredNames(ir, localNames, externalPoolNames));
  const vectorCooperativeReduceHelpers = new Map<string, VectorCooperativeReduceHelper>();
  const expressionValueTypes = new WeakMap<CudaLiteExpression, CudaLiteScalarType | undefined>();
  const paramsBinding = uniformScalars.length > 0 ? bindings.length : undefined;
  if (paramsBinding !== undefined) {
    bindings.push({
      kind: "uniform",
      name: "params",
      byteLength: Math.max(16, uniformScalars.length * 4),
      binding: paramsBinding,
    });
  }
  return {
    ir,
    bindings,
    ...(paramsBinding === undefined ? {} : { paramsBinding }),
    uniformScalars,
    mutableScalarParams,
    deviceFunctionNames: new Set(ir.functions.map((fn) => fn.name)),
    deviceFunctionFor(name, argCount) {
      return resolveDeviceFunctionForCall(ir, name, argCount);
    },
    devicePointerParamFor() {
      return undefined;
    },
    storagePointerIdFor(name) {
      return storagePointerIds.get(name);
    },
    sharedPointerIdFor(name) {
      return sharedPointerIds.get(name);
    },
    constantPointerIdFor(name) {
      return constantPointerIds.get(name);
    },
    deviceGlobalPointerIdFor(name) {
      return deviceGlobalPointerIds.get(name);
    },
    bindingFor(name) {
      const binding = bindingByName.get(name);
      if (binding === undefined) throw featureError("missing-wgsl-binding", `missing WGSL binding for '${name}'`);
      return binding;
    },
    paramFor(name) {
      return ir.params.find((param) => param.name === name);
    },
    deviceGlobalFor(name) {
      return ir.deviceGlobals.find((global) => global.name === name);
    },
    isUniformScalar(name) {
      return uniformScalarNames.has(name);
    },
    uniformScalarTypeFor(name) {
      return uniformScalarTypes.get(name);
    },
    isAtomicShared(name) {
      return ir.atomicShared.includes(name);
    },
    isAtomicDeviceGlobal(name) {
      return ir.atomicDeviceGlobals.includes(name);
    },
    pointerAliasFor(name) {
      return pointerAliases.get(name);
    },
    localPointerHandleFor(name) {
      return localPointerHandles.get(name);
    },
    poolPointerFor(name) {
      return poolPointers.get(name);
    },
    pointerBaseOffsetFieldFor(name) {
      return options.pointerBaseOffsets?.[name] === undefined ? undefined : pointerBaseOffsetUniformName(name);
    },
    rawPoolAllocators,
    externalPoolNames,
    mutablePointerBases,
    vectorCooperativeReduceHelpers,
    expressionValueTypes,
    mutablePointerBaseFor(name) {
      if (localPointerHandles.has(name)) return wgslNames.get(`${name}_base`) ?? `${name}_base`;
      return mutablePointerBases.includes(name) ? `bg_${name}_base` : undefined;
    },
    isLocalName(name) {
      return localNames.has(name);
    },
    localValueTypeFor(name) {
      return localValueTypes.get(name);
    },
    localArrayFor(name) {
      return localArrays.get(name);
    },
    cooperativeGroupFor(name) {
      return cooperativeGroups.get(name);
    },
    surfaceWidthField(name) {
      return surfaceWidthField(name);
    },
    nameFor(name) {
      return wgslNames.get(name) ?? name;
    },
  };
}

function collectWgslDeclaredNames(
  ir: KernelIrModule,
  localNames: ReadonlySet<string>,
  externalPoolNames: readonly string[],
): readonly string[] {
  const structuredPointerRoots = structuredPointerHandleRoots(ir);
  return [
    ir.name,
    ...ir.params.map((param) => param.name),
    ...ir.constants.map((constant) => constant.name),
    ...ir.deviceGlobals.map((global) => global.name),
    ...ir.textures.map((texture) => texture.name),
    ...ir.sharedDeclarations.map((shared) => shared.name),
    ...ir.functions.flatMap((fn) => [
      deviceFunctionLinkName(fn, ir),
      ...fn.params.map((param) => param.name),
      ...collectCooperativeGroupGeneratedNames(fn.params),
      ...collectLocalNames(fn.body),
      ...collectLocalPointerHandleGeneratedNames(fn.body, structuredPointerRoots),
    ]),
    ...localNames,
    ...collectLocalPointerHandleGeneratedNames(ir.body, structuredPointerRoots),
    ...externalPoolNames,
    ...externalPoolNames.flatMap((name) => [poolDataName(name), poolOffsetName(name)]),
  ];
}

function resolveDeviceFunctionForCall(
  ir: KernelIrModule,
  name: string,
  argCount?: number,
): CudaLiteDeviceFunction | undefined {
  const overloads = ir.functions.filter((fn) => fn.name === name);
  if (overloads.length === 0) return undefined;
  if (argCount === undefined) return overloads[0];
  return overloads.find((fn) => fn.params.length === argCount) ?? overloads[0];
}

function deviceFunctionLinkName(fn: CudaLiteDeviceFunction, ir: KernelIrModule): string {
  const overloads = ir.functions.filter((candidate) => candidate.name === fn.name);
  if (overloads.length <= 1) return fn.name;
  const index = overloads.indexOf(fn);
  return `${fn.name}__bg_overload_${index < 0 ? 0 : index}`;
}

function collectLocalPointerHandleGeneratedNames(
  statements: readonly CudaLiteStatement[],
  structuredPointerRoots: ReadonlySet<string> = new Set(),
): readonly string[] {
  return [...collectLocalPointerHandles(statements, undefined, structuredPointerRoots).keys()].flatMap((name) => [`${name}_buffer`, `${name}_base`]);
}

function collectCooperativeGroupGeneratedNames(params: readonly CudaLiteParam[]): readonly string[] {
  return params
    .filter((param) => param.cooperativeGroupKind === "thread")
    .flatMap((param) => [`${param.name}_tile_size`, `${param.name}_tile_size_arg`]);
}

function createWgslNameMap(names: readonly string[]): ReadonlyMap<string, string> {
  const used = new Set([...WGSL_RESERVED_IDENTIFIERS, ...CUDA_INTRINSICS_BY_NAME.keys()]);
  const out = new Map<string, string>();
  for (const name of names) {
    if (out.has(name)) continue;
    const candidate = safeWgslIdentifier(name);
    if (!used.has(candidate)) {
      used.add(candidate);
      out.set(name, candidate);
      continue;
    }
    let index = 0;
    let renamed = `bg_${candidate}`;
    while (used.has(renamed)) renamed = `bg_${candidate}_${++index}`;
    used.add(renamed);
    out.set(name, renamed);
  }
  return out;
}

function safeWgslIdentifier(name: string): string {
  const cleaned = name.replace(/[^A-Za-z0-9_]/gu, "_");
  return /^[A-Za-z_]/u.test(cleaned) ? cleaned : `bg_${cleaned}`;
}

function emitStatement(
  statement: CudaLiteStatement,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  switch (statement.kind) {
    case "block": {
      const lines = [`${prefix}{`];
      lines.push(...statement.body.flatMap((child) => emitStatement(child, context, indentLevel + 1)));
      lines.push(`${prefix}}`);
      return lines;
    }
    case "var":
      if (statement.storage === "shared") return [];
      if (statement.pointer) {
        if (context.localPointerHandleFor(statement.name)) return emitLocalPointerHandleDecl(statement, context, indentLevel);
        if (!isEmittedPointerVar(statement, context)) return [];
        return [`${prefix}var ${context.nameFor(statement.name)}: u32${statement.init ? ` = ${emitExpression(statement.init, context)}` : " = 0u"};`];
      }
      if (statement.dimensions.length > 0 || statement.matrixTile) {
        return [
          `${prefix}var ${context.nameFor(statement.name)}: ${emitLocalArrayType(statement)};`,
          ...emitLocalArrayInitializer(statement, context, indentLevel),
        ];
      }
      return [`${prefix}var ${context.nameFor(statement.name)}: ${wgslScalar(statement.valueType)}${statement.init ? ` = ${emitLocalInit(statement, context)}` : ""};`];
    case "dim3":
      return [];
    case "cooperative-group":
      return [];
    case "kernel-launch":
      return [`${prefix}// device-side launch omitted: ${statement.callee}<<<...>>>`];
    case "asm":
      return emitInlineAsmStatement(statement, context)
        .split("\n")
        .map((line) => `${prefix}${line};`);
    case "expr":
      {
        const wmma = emitWmmaStatement(statement.expression, context, indentLevel);
        if (wmma) return wmma;
      }
      {
        const cpAsync = emitCpAsyncStatement(statement.expression, context, indentLevel);
        if (cpAsync) return cpAsync;
      }
      {
        const noopComment = noopCallComment(statement.expression);
        if (noopComment) return [`${prefix}// ${noopComment}`];
      }
      {
        const fill = emitFillRegsStatement(statement.expression, context, indentLevel);
        if (fill) return fill;
      }
      if (isBarrierCall(statement.expression)) return [`${prefix}workgroupBarrier();`];
      if (statement.expression.kind === "assignment") {
        return emitAssignmentStatement(statement.expression, context, indentLevel);
      }
      return [`${prefix}${emitExpression(statement.expression, context)};`];
    case "if": {
      const lines = [`${prefix}if (${emitExpression(statement.condition, context)}) {`];
      lines.push(...statement.consequent.flatMap((child) => emitStatement(child, context, indentLevel + 1)));
      if (statement.alternate) {
        lines.push(`${prefix}} else {`);
        lines.push(...statement.alternate.flatMap((child) => emitStatement(child, context, indentLevel + 1)));
      }
      lines.push(`${prefix}}`);
      return lines;
    }
    case "for": {
      if (statement.update?.kind === "sequence" || statement.init?.kind === "sequence") {
        return emitForLoopWithContinuing(statement, context, indentLevel);
      }
      const init = statement.init?.kind === "var"
        ? emitForVar(statement.init, context)
        : statement.init
          ? emitExpression(statement.init, context)
          : "";
      const condition = statement.condition ? emitExpression(statement.condition, context) : "true";
      const update = statement.update ? emitExpression(statement.update, context) : "";
      const lines = [`${prefix}for (${init}; ${condition}; ${update}) {`];
      lines.push(...statement.body.flatMap((child) => emitStatement(child, context, indentLevel + 1)));
      lines.push(`${prefix}}`);
      return lines;
    }
    case "while": {
      const lines = [`${prefix}while (${emitExpression(statement.condition, context)}) {`];
      lines.push(...statement.body.flatMap((child) => emitStatement(child, context, indentLevel + 1)));
      lines.push(`${prefix}}`);
      return lines;
    }
    case "do-while": {
      const lines = [`${prefix}loop {`];
      lines.push(...statement.body.flatMap((child) => emitStatement(child, context, indentLevel + 1)));
      lines.push(`${indent(indentLevel + 1)}if (!(${emitExpression(statement.condition, context)})) { break; }`);
      lines.push(`${prefix}}`);
      return lines;
    }
    case "return":
      return [`${prefix}${statement.value ? `return ${emitExpression(statement.value, context)};` : "return;"}`];
    case "continue":
      return [`${prefix}continue;`];
    case "break":
      return [`${prefix}break;`];
  }
}

function functionsToEmit(ir: KernelIrModule): readonly CudaLiteDeviceFunction[] {
  const launchCallees = new Set(collectKernelLaunchCallees(ir.body));
  return ir.functions.filter((fn) => fn.name !== ir.name && !launchCallees.has(fn.name));
}

function emitInlineAsmStatement(
  statement: Extract<CudaLiteStatement, { kind: "asm" }>,
  context: EmitContext,
): string {
  const op = classifyInlineAsm(statement.template);
  const outputs = statement.outputs ?? (statement.output === undefined ? [] : [statement.output]);
  if (op?.kind === "laneid" && statement.inputs.length === 0 && outputs.length === 1) {
    return `${emitExpression(outputs[0]!, context)} = ${emitInlineU32Output(outputs[0]!, `u32(${emitLocalLinearRank(context)} % 32)`, context)}`;
  }
  if (op?.kind === "lanemask-lt" && statement.inputs.length === 0 && outputs.length === 1) {
    const lane = `u32(${emitLocalLinearRank(context)} & 31)`;
    const mask = `select(0u, ((1u << ${lane}) - 1u), ${lane} > 0u)`;
    return `${emitExpression(outputs[0]!, context)} = ${emitInlineU32Output(outputs[0]!, mask, context)}`;
  }
  if (op?.kind === "globaltimer-u64" && statement.inputs.length === 0 && outputs.length === 1) {
    const tick = `u32((global_id.x * ${context.ir.workgroupSize[0]}u) + ${emitLocalLinearRank(context)})`;
    return `${emitExpression(outputs[0]!, context)} = ${emitInlineU32Output(outputs[0]!, tick, context)}`;
  }
  if (op?.kind === "bfind-u32" && statement.inputs.length === 1 && outputs.length === 1) {
    return `${emitExpression(outputs[0]!, context)} = (31u - countLeadingZeros(u32(${emitExpression(statement.inputs[0]!, context)})))`;
  }
  if (op?.kind === "u8x4-sad-add" && statement.inputs.length === 3 && outputs.length === 1) {
    return `${emitExpression(outputs[0]!, context)} = ${emitInlineU32Output(outputs[0]!, emitU8x4SadAddExpression(statement.inputs, context), context)}`;
  }
  if (op?.kind === "ldmatrix" && statement.inputs.length === 1 && outputs.length === op.matrices) {
    const base = `u32(${emitExpression(statement.inputs[0]!, context)})`;
    const tag = op.transposed ? "0x80000000u" : "0u";
    return outputs.map((output, index) => {
      const carrier = `(${tag} + ${base} + ${index * 2}u)`;
      return `${emitExpression(output, context)} = ${emitInlineU32Output(output, carrier, context)}`;
    }).join("\n");
  }
  if (op?.kind === "mma-m16n8k16") {
    return emitMmaM16N8K16Statement(statement, outputs, op.accumulator, context);
  }
  if (op?.kind !== "fma-rn-f32" || statement.inputs.length !== 2 || outputs.length !== 1) {
    throw featureError("unsupported-inline-asm", `only ${inlineAsmSupportedList()} inline PTX are supported in WGSL output`);
  }
  const target = emitExpression(outputs[0]!, context);
  return `${target} = fma(${emitExpression(statement.inputs[0]!, context)}, ${emitExpression(statement.inputs[1]!, context)}, ${target})`;
}

function emitMmaM16N8K16Statement(
  statement: Extract<CudaLiteStatement, { kind: "asm" }>,
  outputs: readonly CudaLiteExpression[],
  accumulator: "f16" | "f32",
  context: EmitContext,
): string {
  if (accumulator === "f16") {
    if (outputs.length !== 2 || statement.inputs.length !== 8) {
      throw featureError("invalid-inline-asm-operands", "mma.m16n8k16 f16 inline PTX operand mismatch");
    }
    return outputs.map((output, index) => {
      const a = `u32(${emitExpression(statement.inputs[index % 4]!, context)})`;
      const b = `u32(${emitExpression(statement.inputs[4 + (index % 2)]!, context)})`;
      const c = `u32(${emitExpression(statement.inputs[6 + index]!, context)})`;
      const value = `pack2x16float(unpack2x16float(${c}) + (unpack2x16float(${a}) * unpack2x16float(${b})))`;
      return `${emitExpression(output, context)} = ${emitInlineU32Output(output, value, context)}`;
    }).join("\n");
  }
  if (outputs.length !== 4 || statement.inputs.length !== 10) {
    throw featureError("invalid-inline-asm-operands", "mma.m16n8k16 f32 inline PTX operand mismatch");
  }
  return outputs.map((output, index) => {
    const a = `u32(${emitExpression(statement.inputs[index % 4]!, context)})`;
    const b = `u32(${emitExpression(statement.inputs[4 + (index % 2)]!, context)})`;
    const c = emitExpression(statement.inputs[6 + index]!, context);
    return `${emitExpression(output, context)} = (${c} + dot(unpack2x16float(${a}), unpack2x16float(${b})))`;
  }).join("\n");
}

function emitU8x4SadAddExpression(inputs: readonly CudaLiteExpression[], context: EmitContext): string {
  const a = `u32(${emitExpression(inputs[0]!, context)})`;
  const b = `u32(${emitExpression(inputs[1]!, context)})`;
  const c = `u32(${emitExpression(inputs[2]!, context)})`;
  const lanes = [0, 8, 16, 24].map((shift) => {
    const left = `((${a} >> ${shift}u) & 0xffu)`;
    const right = `((${b} >> ${shift}u) & 0xffu)`;
    return `(max(${left}, ${right}) - min(${left}, ${right}))`;
  });
  return `(${c} + ${lanes.join(" + ")})`;
}

function emitForVar(statement: CudaLiteVarDecl, context: EmitContext): string {
  if (statement.pointer && context.localPointerHandleFor(statement.name)) {
    throw featureError(
      "unsupported-local-pointer-for-init",
      "mutable local pointer declarations in for-loop initializers are not supported yet",
    );
  }
  if (statement.pointer) return `var ${context.nameFor(statement.name)}: u32${statement.init ? ` = ${emitExpression(statement.init, context)}` : " = 0u"}`;
  if (statement.dimensions.length > 0) return `var ${context.nameFor(statement.name)}: ${emitLocalArrayType(statement)}`;
  return `var ${context.nameFor(statement.name)}: ${wgslScalar(statement.valueType)}${statement.init ? ` = ${emitLocalInit(statement, context)}` : ""}`;
}

function emitLocalPointerHandleDecl(
  statement: CudaLiteVarDecl,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  const [buffer, base] = emitLocalPointerHandleInit(statement, context);
  return [
    `${prefix}var ${context.nameFor(`${statement.name}_buffer`)}: u32 = ${buffer};`,
    `${prefix}var ${context.nameFor(`${statement.name}_base`)}: u32 = ${base};`,
  ];
}

function emitLocalPointerHandleInit(statement: CudaLiteVarDecl, context: EmitContext): readonly [string, string] {
  if (!statement.init) return ["0u", "0u"];
  const parts = devicePointerArgumentParts(statement.init, context);
  if (!parts) {
    throw featureError(
      "unsupported-device-pointer-param",
      `local pointer '${statement.name}' at line ${statement.span.line} must initialize from modeled storage or shared memory`,
    );
  }
  return [parts.buffer, parts.base];
}

function emitLocalInit(statement: CudaLiteVarDecl, context: EmitContext): string {
  const value = statement.init ? emitExpression(statement.init, context) : zeroValue(statement.valueType);
  if (statement.valueType === "uint") return `u32(${value})`;
  return value;
}

function emitDeviceFunction(fn: CudaLiteDeviceFunction, context: EmitContext): string[] {
  const cooperativeParams = new Map(fn.params
    .filter((param) => param.cooperativeGroupKind !== undefined)
    .map((param) => [param.name, cooperativeGroupForParam(param, context)] as const));
  const functionPointerParams = new Set(fn.params
    .filter((param) => param.pointer && usesFunctionLocalPointerParam(fn, param, context.ir))
    .map((param) => param.name));
  const functionLocalPointerHandles = collectLocalPointerHandles(fn.body, undefined, structuredPointerHandleRoots(context.ir));
  const functionContext = withDevicePointerParams(
    {
      ...context,
      localPointerHandleFor(name) {
        return functionLocalPointerHandles.get(name) ?? context.localPointerHandleFor(name);
      },
      cooperativeGroupFor(name) {
        return cooperativeParams.get(name) ?? context.cooperativeGroupFor(name);
      },
    },
    fn.params.filter((param) => param.pointer && !functionPointerParams.has(param.name)),
    new Set([...fn.params.map((param) => param.name), ...collectLocalNames(fn.body)]),
  );
  const params = [
    ...fn.params.flatMap((param) => param.pointer
      ? functionPointerParams.has(param.name)
        ? [`${context.nameFor(param.name)}: ptr<function, ${wgslScalar(param.valueType)}>`]
        : [`${context.nameFor(`${param.name}_buffer_arg`)}: u32`, `${context.nameFor(`${param.name}_base_arg`)}: u32`]
      : param.cooperativeGroupKind !== undefined
        ? param.cooperativeGroupKind === "thread"
          ? [`${context.nameFor(`${param.name}_tile_size_arg`)}: u32`]
          : []
        : param.valueType === "texture2d"
          ? [`${context.nameFor(param.name)}: texture_2d<f32>`]
        : [`${context.nameFor(`${param.name}_arg`)}: ${wgslScalar(param.valueType)}`]),
    "local_id: vec3<u32>",
    "workgroup_id: vec3<u32>",
    "num_workgroups: vec3<u32>",
  ];
  const returnType = fn.returnType === "void" ? "" : ` -> ${wgslScalar(fn.returnType)}`;
  const lines = [`fn ${context.nameFor(deviceFunctionLinkName(fn, context.ir))}(${params.join(", ")})${returnType} {`];
  for (const param of fn.params) {
    if (param.pointer) {
      if (functionPointerParams.has(param.name)) continue;
      lines.push(`  var ${context.nameFor(`${param.name}_buffer`)}: u32 = ${context.nameFor(`${param.name}_buffer_arg`)};`);
      lines.push(`  var ${context.nameFor(`${param.name}_base`)}: u32 = ${context.nameFor(`${param.name}_base_arg`)};`);
    } else if (param.cooperativeGroupKind !== undefined) {
      if (param.cooperativeGroupKind === "thread") {
        lines.push(`  let ${context.nameFor(`${param.name}_tile_size`)}: u32 = ${context.nameFor(`${param.name}_tile_size_arg`)};`);
      }
      continue;
    } else if (param.valueType === "texture2d") {
      continue;
    } else {
      lines.push(`  var ${context.nameFor(param.name)}: ${wgslScalar(param.valueType)} = ${context.nameFor(`${param.name}_arg`)};`);
    }
  }
  lines.push(...fn.body.flatMap((statement) => emitStatement(statement, functionContext, 1)));
  if (fn.returnType !== "void" && !functionBodyHasReturn(fn.body)) {
    lines.push(`  return ${zeroValue(fn.returnType)};`);
  }
  lines.push("}");
  return lines;
}

function withDevicePointerParams(
  context: EmitContext,
  params: readonly CudaLiteParam[],
  localNames: ReadonlySet<string> = new Set(),
): EmitContext {
  const pointerParams = new Map(params.map((param) => [param.name, param] as const));
  return {
    ...context,
    isLocalName(name) {
      return localNames.has(name) || context.isLocalName(name);
    },
    devicePointerParamFor(name) {
      return pointerParams.get(name) ?? context.devicePointerParamFor(name);
    },
    mutablePointerBaseFor(name) {
      return pointerParams.has(name) ? context.nameFor(`${name}_base`) : context.mutablePointerBaseFor(name);
    },
  };
}

function usesFunctionLocalPointerParam(
  fn: CudaLiteDeviceFunction,
  param: CudaLiteParam,
  ir: KernelIrModule,
): boolean {
  if (!param.pointer) return false;
  const paramIndex = fn.params.findIndex((item) => item.name === param.name);
  if (paramIndex < 0) return false;
  const sharedNames = new Set(ir.sharedDeclarations.map((shared) => shared.name));
  let sawCall = false;
  let allCallsUseLocalAddress = true;
  for (const statements of [ir.body, ...ir.functions.map((item) => item.body)]) {
    walkCudaLiteExpressions(statements, (expression) => {
      if (!allCallsUseLocalAddress || expression.kind !== "call" || expressionName(expression.callee) !== fn.name) return;
      sawCall = true;
      const arg = expression.args[paramIndex];
      if (!arg || !isFunctionLocalPointerArgument(arg, sharedNames)) allCallsUseLocalAddress = false;
    });
  }
  return sawCall && allCallsUseLocalAddress;
}

function isFunctionLocalPointerArgument(expression: CudaLiteExpression, sharedNames: ReadonlySet<string>): boolean {
  return expression.kind === "unary" &&
    expression.operator === "&" &&
    expression.argument.kind === "identifier" &&
    !sharedNames.has(expression.argument.name);
}

function usesDevicePointerParams(ir: KernelIrModule): boolean {
  const devicePointerCalls = new Set([
    ...CUDA_CACHE_HINT_LOADS,
    ...CUDA_CACHE_HINT_STORES,
    "CP_ASYNC_CA",
    "CP_ASYNC_CG",
    "CP_ASYNC_BULK",
    "wmma::load_matrix_sync",
    "nvcuda::wmma::load_matrix_sync",
    "wmma::store_matrix_sync",
    "nvcuda::wmma::store_matrix_sync",
  ]);
  return ir.functions.some((fn) => fn.params.some((param) => param.pointer)) ||
    statementsUseCall(ir.body, devicePointerCalls) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, devicePointerCalls));
}

function emitDevicePointerHelpers(ir: KernelIrModule, context: EmitContext): string[] {
  const types = [...devicePointerHelperTypes(ir)];
  return types.flatMap((type) => emitDevicePointerHelper(type, ir, context));
}

function devicePointerHelperTypes(ir: KernelIrModule): ReadonlySet<CudaLiteScalarType> {
  const rawTypes = [
    ...ir.params.filter((param) => param.pointer && !isDevicePoolParam(param)).map((param) => param.valueType),
    ...ir.deviceGlobals.map((global) => global.valueType),
    ...ir.functions.flatMap((fn) => fn.params.filter((param) => param.pointer).map((param) => param.valueType)),
    ...[...collectLocalPointerHandles(ir.body).values()].map((item) => item.valueType),
    ...ir.functions.flatMap((fn) => [...collectLocalPointerHandles(fn.body).values()].map((item) => item.valueType)),
  ];
  const types = new Set<CudaLiteScalarType>(rawTypes.map(pointerHelperCanonicalType));
  for (const statements of [ir.body, ...ir.functions.map((fn) => fn.body)]) {
    walkCudaLiteExpressions(statements, (expression) => {
      if (expression.kind === "cast" && expression.pointer && isDevicePointerHelperType(expression.valueType)) {
        types.add(pointerHelperCanonicalType(expression.valueType));
      }
    });
  }
  return types;
}

function pointerHelperCanonicalType(type: CudaLiteScalarType): CudaLiteScalarType {
  return type === "double" ? "float" : type;
}

function emitDevicePointerHelper(type: CudaLiteScalarType, ir: KernelIrModule, context: EmitContext): string[] {
  if (!isDevicePointerHelperType(type)) {
    throw featureError("unsupported-device-pointer-param", `device pointer helpers do not support ${type} pointers yet`);
  }
  const storageParams = ir.params.filter((param) =>
    param.pointer &&
    !isDevicePoolParam(param) &&
    isPointerHelperCompatibleStorage(type, param.valueType)
  );
  const sharedDeclarations = ir.sharedDeclarations.filter((shared) =>
    isPointerHelperCompatibleStorage(type, shared.valueType)
  );
  const deviceGlobals = ir.deviceGlobals.filter((global) =>
    isPointerHelperCompatibleStorage(type, global.valueType)
  );
  const constantArrays = ir.constants.filter((constant) =>
    constant.dimensions.length > 0 &&
    constant.init === undefined &&
    isPointerHelperCompatibleStorage(type, constant.valueType)
  );
  const scalar = wgslScalar(type);
  const lines = [
    `fn ${pointerReadHelperName(type)}(buffer: u32, index: u32) -> ${scalar} {`,
    "  switch buffer {",
  ];
  for (const param of storageParams) {
    const id = context.storagePointerIdFor(param.name);
    if (id === undefined) continue;
    lines.push(`    case ${id}u: { return ${emitPointerStorageRead(param, "index", ir, context, type)}; }`);
  }
  for (const shared of sharedDeclarations) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    lines.push(`    case ${id}u: { return ${emitSharedPointerRead(shared, "index", ir, context, type)}; }`);
  }
  for (const global of deviceGlobals) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    lines.push(`    case ${id}u: { return ${emitDeviceGlobalPointerRead(global, "index", ir, context, type)}; }`);
  }
  for (const constant of constantArrays) {
    const id = context.constantPointerIdFor(constant.name);
    if (id === undefined) continue;
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
    lines.push(`    case ${id}u: { ${emitPointerStorageWrite(param, "index", "value", ir, context, type)}; return; }`);
  }
  for (const shared of sharedDeclarations) {
    const id = context.sharedPointerIdFor(shared.name);
    if (id === undefined) continue;
    lines.push(`    case ${id}u: { ${emitSharedPointerWrite(shared, "index", "value", ir, context, type)}; return; }`);
  }
  for (const global of deviceGlobals) {
    const id = context.deviceGlobalPointerIdFor(global.name);
    if (id === undefined) continue;
    lines.push(`    case ${id}u: { ${emitDeviceGlobalPointerWrite(global, "index", "value", ir, context, type)}; return; }`);
  }
  lines.push("    default: { return; }");
  lines.push("  }");
  lines.push("}");
  if (isDevicePointerAtomicAddType(type) && usesDevicePointerAtomicAdd(ir)) {
    lines.push("");
    lines.push(...emitDevicePointerAtomicAddHelper(type, ir, context));
  }
  if (isDevicePointerAtomicExchangeType(type) && usesDevicePointerAtomicExchange(ir)) {
    lines.push("");
    lines.push(...emitDevicePointerAtomicExchangeHelper(type, ir, context));
  }
  if (isDevicePointerAtomicCasType(type) && usesDevicePointerAtomicCas(ir)) {
    lines.push("");
    lines.push(...emitDevicePointerAtomicCasHelper(type, ir, context));
  }
  return lines;
}

function isDevicePointerHelperType(type: CudaLiteScalarType): boolean {
  return type === "float" || type === "double" || type === "int" || type === "uint" || type === "half" || type === "bf16" || isCudaVectorType(type);
}

function isDevicePointerAtomicAddType(type: CudaLiteScalarType): type is "float" | "double" | "int" | "uint" {
  return type === "float" || type === "double" || type === "int" || type === "uint";
}

function isDevicePointerAtomicExchangeType(type: CudaLiteScalarType): type is "float" | "int" | "uint" {
  return type === "float" || type === "int" || type === "uint";
}

function isDevicePointerAtomicCasType(type: CudaLiteScalarType): type is "int" | "uint" {
  return type === "int" || type === "uint";
}

function usesDevicePointerAtomicAdd(ir: KernelIrModule): boolean {
  const atomicAdds = new Set(["atomicAdd", "atomicAdd_system"]);
  return statementsUseCall(ir.body, atomicAdds) || ir.functions.some((fn) => statementsUseCall(fn.body, atomicAdds));
}

function usesDevicePointerAtomicExchange(ir: KernelIrModule): boolean {
  const atomicExchanges = new Set(["atomicExch", "atomicExch_system"]);
  return statementsUseCall(ir.body, atomicExchanges) || ir.functions.some((fn) => statementsUseCall(fn.body, atomicExchanges));
}

function usesDevicePointerAtomicCas(ir: KernelIrModule): boolean {
  const atomicCas = new Set(["atomicCAS", "atomicCAS_system"]);
  return statementsUseCall(ir.body, atomicCas) || ir.functions.some((fn) => statementsUseCall(fn.body, atomicCas));
}

function emitDevicePointerAtomicAddHelper(
  type: "float" | "double" | "int" | "uint",
  ir: KernelIrModule,
  context: EmitContext,
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

function emitDevicePointerAtomicExchangeHelper(
  type: "float" | "int" | "uint",
  ir: KernelIrModule,
  context: EmitContext,
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

function emitDevicePointerAtomicCasHelper(
  type: "int" | "uint",
  ir: KernelIrModule,
  context: EmitContext,
): string[] {
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

function emitAtomicAddAtAddress(
  type: "float" | "double" | "int" | "uint",
  addressSpace: "storage" | "workgroup",
  address: string,
  value: string,
): string {
  return type === "float" || type === "double"
    ? `${floatAtomicHelperName("Add", addressSpace)}(${address}, ${value})`
    : `atomicAdd(${address}, ${value})`;
}

function emitAtomicExchangeAtAddress(type: "float" | "int" | "uint", address: string, value: string): string {
  return type === "float"
    ? `bitcast<f32>(atomicExchange(${address}, bitcast<u32>(${value})))`
    : `atomicExchange(${address}, ${value})`;
}

function isPointerHelperCompatibleStorage(helperType: CudaLiteScalarType, storageType: CudaLiteScalarType): boolean {
  if (helperType === storageType) return true;
  if (helperType === "float" && storageType === "double") return true;
  return isCudaVectorType(helperType) && cudaVectorScalarType(helperType) === storageType;
}

function emitPointerStorageRead(
  param: CudaLiteParam,
  index: string,
  ir: KernelIrModule,
  context: EmitContext,
  viewType: CudaLiteScalarType = param.valueType,
): string {
  const name = context.nameFor(param.name);
  if (isCudaVectorType(viewType)) {
    return param.valueType === viewType
      ? emitVectorStorageRead(name, viewType, index)
      : emitVectorStorageReadAt(name, viewType, index);
  }
  const access = `${name}[${index}]`;
  if (param.valueType === "bool") return `(${access} != 0u)`;
  if (!ir.atomicParams.includes(param.name)) return access;
  const loaded = `atomicLoad(&${access})`;
  return param.valueType === "float" || param.valueType === "double" ? `bitcast<f32>(${loaded})` : loaded;
}

function emitPointerStorageWrite(
  param: CudaLiteParam,
  index: string,
  value: string,
  ir: KernelIrModule,
  context: EmitContext,
  viewType: CudaLiteScalarType = param.valueType,
): string {
  const name = context.nameFor(param.name);
  if (isCudaVectorType(viewType)) {
    return param.valueType === viewType
      ? emitVectorStorageWrite(name, viewType, index, value)
      : emitVectorStorageWriteAt(name, viewType, index, value);
  }
  const access = `${name}[${index}]`;
  if (param.valueType === "bool") return `${access} = select(0u, 1u, ${value})`;
  if (!ir.atomicParams.includes(param.name)) return `${access} = ${value}`;
  if (param.valueType === "float" || param.valueType === "double") return `atomicStore(&${access}, bitcast<u32>(${value}))`;
  return `atomicStore(&${access}, ${value})`;
}

function emitDeviceGlobalPointerRead(
  global: CudaLiteDeviceGlobal,
  index: string,
  ir: KernelIrModule,
  context: EmitContext,
  viewType: CudaLiteScalarType = global.valueType,
): string {
  const name = context.nameFor(global.name);
  if (isCudaVectorType(viewType)) {
    return global.valueType === viewType
      ? emitVectorStorageRead(name, viewType, index)
      : emitVectorStorageReadAt(name, viewType, index);
  }
  const access = `${name}[${index}]`;
  if (global.valueType === "bool") return `(${access} != 0u)`;
  if (!ir.atomicDeviceGlobals.includes(global.name)) return access;
  const loaded = `atomicLoad(&${access})`;
  return global.valueType === "float" ? `bitcast<f32>(${loaded})` : loaded;
}

function emitDeviceGlobalPointerWrite(
  global: CudaLiteDeviceGlobal,
  index: string,
  value: string,
  ir: KernelIrModule,
  context: EmitContext,
  viewType: CudaLiteScalarType = global.valueType,
): string {
  const name = context.nameFor(global.name);
  if (isCudaVectorType(viewType)) {
    return global.valueType === viewType
      ? emitVectorStorageWrite(name, viewType, index, value)
      : emitVectorStorageWriteAt(name, viewType, index, value);
  }
  const access = `${name}[${index}]`;
  if (global.valueType === "bool") return `${access} = select(0u, 1u, ${value})`;
  if (!ir.atomicDeviceGlobals.includes(global.name)) return `${access} = ${value}`;
  if (global.valueType === "float") return `atomicStore(&${access}, bitcast<u32>(${value}))`;
  return `atomicStore(&${access}, ${value})`;
}

function emitConstantPointerRead(
  constant: CudaLiteGlobalConstant,
  index: string,
  context: EmitContext,
  viewType: CudaLiteScalarType = constant.valueType,
): string {
  if (isCudaVectorType(viewType)) {
    return constant.valueType === viewType
      ? emitVectorStorageRead(context.nameFor(constant.name), viewType, index)
      : emitConstantVectorFlatRead(constant, index, viewType, context);
  }
  return emitSharedFlatAccess(context.nameFor(constant.name), externalConstantDimensions(constant), index);
}

function emitVectorStorageRead(name: string, type: CudaLiteScalarType, index: string): string {
  const lanes = cudaVectorLaneCount(type);
  const base = vectorStorageBase(index, lanes);
  const values = Array.from({ length: lanes }, (_, lane) => `${name}[${base} + ${lane}u]`);
  return `${wgslScalar(type)}(${values.join(", ")})`;
}

function emitVectorStorageWrite(name: string, type: CudaLiteScalarType, index: string, value: string): string {
  const lanes = cudaVectorLaneCount(type);
  const base = vectorStorageBase(index, lanes);
  return Array.from({ length: lanes }, (_, lane) => `${name}[${base} + ${lane}u] = ${value}.${vectorFieldName(lane)}`).join("; ");
}

function emitVectorStorageFieldWrite(name: string, type: CudaLiteScalarType, index: string, field: string, value: string): string | undefined {
  const fieldIndex = cudaVectorFieldIndex(type, field);
  if (fieldIndex === undefined) return undefined;
  const base = vectorStorageBase(index, cudaVectorLaneCount(type));
  return `${name}[${base} + ${fieldIndex}u] = ${value}`;
}

function vectorStorageBase(index: string, lanes: number): string {
  return `(u32(${index}) * ${lanes}u)`;
}

function vectorFieldName(index: number): string {
  return index === 0 ? "x" : index === 1 ? "y" : index === 2 ? "z" : "w";
}

function emitSharedPointerRead(
  shared: CudaLiteVarDecl,
  index: string,
  ir: KernelIrModule,
  context: EmitContext,
  viewType: CudaLiteScalarType = shared.valueType,
): string {
  if (isCudaVectorType(viewType) && shared.valueType !== viewType) return emitSharedVectorFlatRead(shared, index, viewType, context);
  const access = emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, index);
  if (!ir.atomicShared.includes(shared.name)) return access;
  const loaded = `atomicLoad(&${access})`;
  return shared.valueType === "float" ? `bitcast<f32>(${loaded})` : loaded;
}

function emitSharedPointerWrite(
  shared: CudaLiteVarDecl,
  index: string,
  value: string,
  ir: KernelIrModule,
  context: EmitContext,
  viewType: CudaLiteScalarType = shared.valueType,
): string {
  if (isCudaVectorType(viewType) && shared.valueType !== viewType) return emitSharedVectorFlatWrite(shared, index, value, viewType, context);
  const access = emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, index);
  if (!ir.atomicShared.includes(shared.name)) return `${access} = ${value}`;
  return shared.valueType === "float" ? `atomicStore(&${access}, bitcast<u32>(${value}))` : `atomicStore(&${access}, ${value})`;
}

function emitSharedFlatAccess(name: string, dimensions: readonly number[], index: string): string {
  if (dimensions.length === 0) return name;
  if (dimensions.length <= 1) return `${name}[${index}]`;
  return dimensions.reduce((expr, dimension, axis) => {
    const stride = dimensions.slice(axis + 1).reduce((product, item) => product * item, 1);
    const axisIndex = stride === 1
      ? `(${index} % ${dimension}u)`
      : `(((${index}) / ${stride}u) % ${dimension}u)`;
    return `${expr}[${axisIndex}]`;
  }, name);
}

function emitSharedVectorFlatRead(
  shared: CudaLiteVarDecl,
  index: string,
  viewType: CudaLiteScalarType,
  context: EmitContext,
): string {
  const lanes = cudaVectorLaneCount(viewType);
  const values = Array.from({ length: lanes }, (_, lane) =>
    emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, `(${index} + ${lane}u)`)
  );
  return `${wgslScalar(viewType)}(${values.join(", ")})`;
}

function emitSharedVectorFlatWrite(
  shared: CudaLiteVarDecl,
  index: string,
  value: string,
  viewType: CudaLiteScalarType,
  context: EmitContext,
): string {
  const lanes = cudaVectorLaneCount(viewType);
  return Array.from({ length: lanes }, (_, lane) =>
    `${emitSharedFlatAccess(context.nameFor(shared.name), shared.dimensions, `(${index} + ${lane}u)`)} = ${value}.${vectorFieldName(lane)}`
  ).join("; ");
}

function emitConstantVectorFlatRead(
  constant: CudaLiteGlobalConstant,
  index: string,
  viewType: CudaLiteScalarType,
  context: EmitContext,
): string {
  const lanes = cudaVectorLaneCount(viewType);
  const values = Array.from({ length: lanes }, (_, lane) =>
    emitSharedFlatAccess(context.nameFor(constant.name), externalConstantDimensions(constant), `(${index} + ${lane}u)`)
  );
  return `${wgslScalar(viewType)}(${values.join(", ")})`;
}

interface DevicePointerParts {
  readonly buffer: string;
  readonly base: string;
}

interface DevicePointerLValue {
  readonly buffer: string;
  readonly index: string;
  readonly valueType: CudaLiteScalarType;
  readonly fieldIndex?: number;
}

interface AtomicTargetInfo {
  readonly address: string;
  readonly rootName: string;
  readonly valueType: CudaLiteScalarType;
  readonly storageScalar: "i32" | "u32";
  readonly addressSpace: "storage" | "workgroup";
}

function emitDevicePointerArgument(expression: CudaLiteExpression, context: EmitContext): readonly [string, string] {
  const parts = devicePointerArgumentParts(expression, context);
  if (!parts) {
    throw featureError("unsupported-device-pointer-param", "device pointer argument must be a storage pointer or derived storage address");
  }
  return [parts.buffer, parts.base];
}

function devicePointerArgumentParts(expression: CudaLiteExpression, context: EmitContext): DevicePointerParts | undefined {
  if (isNullPointerExpression(expression)) {
    return { buffer: NULL_DEVICE_POINTER_BUFFER, base: "0u" };
  }
  if (expression.kind === "conditional") {
    const consequent = devicePointerArgumentParts(expression.consequent, context);
    const alternate = devicePointerArgumentParts(expression.alternate, context);
    if (!consequent || !alternate) return undefined;
    const condition = emitTruthinessExpression(expression.condition, context);
    return {
      buffer: `select(${alternate.buffer}, ${consequent.buffer}, ${condition})`,
      base: `select(${alternate.base}, ${consequent.base}, ${condition})`,
    };
  }
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    const pointer = expression.args[0];
    return pointer ? devicePointerArgumentParts(pointer, context) : undefined;
  }
  if (expression.kind === "identifier") {
    if (context.localPointerHandleFor(expression.name)) {
      return {
        buffer: context.nameFor(`${expression.name}_buffer`),
        base: context.nameFor(`${expression.name}_base`),
      };
    }
    const pointerParam = context.devicePointerParamFor(expression.name);
    if (pointerParam) return { buffer: `${expression.name}_buffer`, base: `${expression.name}_base` };
    const storageId = context.storagePointerIdFor(expression.name);
    const storageParam = context.paramFor(expression.name);
    if (storageId !== undefined && storageParam?.pointer) {
      return { buffer: `${storageId}u`, base: pointerBaseExpression(expression.name, context) ?? "0u" };
    }
    const sharedId = context.sharedPointerIdFor(expression.name);
    if (sharedId !== undefined) return { buffer: `${sharedId}u`, base: "0u" };
    const globalId = context.deviceGlobalPointerIdFor(expression.name);
    if (globalId !== undefined) return { buffer: `${globalId}u`, base: "0u" };
    const constantId = context.constantPointerIdFor(expression.name);
    if (constantId !== undefined) return { buffer: `${constantId}u`, base: "0u" };
    const alias = context.pointerAliasFor(expression.name);
    if (alias) {
      const target = devicePointerArgumentParts({
        kind: "identifier",
        name: alias.rootName,
        span: expression.span,
      }, context);
      if (!target) return undefined;
      return {
        buffer: target.buffer,
        base: `(${target.base} + u32(${emitExpression(alias.baseIndex, context)}))`,
      };
    }
  }
  if (expression.kind === "cast" && expression.pointer) {
    return devicePointerArgumentParts(expression.expression, context);
  }
  if (expression.kind === "unary" && expression.operator === "&" && expression.argument.kind === "index") {
    const global = deviceGlobalPointerArgumentParts(expression.argument, context);
    if (global) return global;
    const shared = sharedPointerArgumentParts(expression.argument, context);
    if (shared) return shared;
    const constant = constantPointerArgumentParts(expression.argument, context);
    if (constant) return constant;
    const target = devicePointerArgumentParts(expression.argument.target, context);
    if (!target) return undefined;
    return {
      buffer: target.buffer,
      base: `(${target.base} + u32(${emitExpression(expression.argument.index, context)}))`,
    };
  }
  if (expression.kind === "unary" && expression.operator === "&" && expression.argument.kind === "identifier") {
    const sharedId = context.sharedPointerIdFor(expression.argument.name);
    if (sharedId !== undefined) return { buffer: `${sharedId}u`, base: "0u" };
    const globalId = context.deviceGlobalPointerIdFor(expression.argument.name);
    if (globalId !== undefined) return { buffer: `${globalId}u`, base: "0u" };
    const constantId = context.constantPointerIdFor(expression.argument.name);
    if (constantId !== undefined) return { buffer: `${constantId}u`, base: "0u" };
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const target = devicePointerArgumentParts(expression.left, context);
    if (!target) return undefined;
    const delta = `u32(${emitExpression(expression.right, context)})`;
    return {
      buffer: target.buffer,
      base: expression.operator === "+"
        ? `(${target.base} + ${delta})`
        : `(${target.base} - ${delta})`,
    };
  }
  return undefined;
}

function constantPointerArgumentParts(expression: CudaLiteExpression, context: EmitContext): DevicePointerParts | undefined {
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
    const value = `u32(${emitExpression(index, context)})`;
    return stride === 1 ? value : `(${value} * ${stride}u)`;
  });
  return { buffer: `${id}u`, base: terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})` };
}

function sharedPointerArgumentParts(expression: CudaLiteExpression, context: EmitContext): DevicePointerParts | undefined {
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
    const value = `u32(${emitExpression(index, context)})`;
    return stride === 1 ? value : `(${value} * ${stride}u)`;
  });
  return { buffer: `${id}u`, base: terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})` };
}

function deviceGlobalPointerArgumentParts(expression: CudaLiteExpression, context: EmitContext): DevicePointerParts | undefined {
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
    const value = `u32(${emitExpression(index, context)})`;
    return stride === 1 ? value : `(${value} * ${stride}u)`;
  });
  return { buffer: `${id}u`, base: terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})` };
}

function emitTruthinessExpression(expression: CudaLiteExpression, context: EmitContext): string {
  const pointer = devicePointerArgumentParts(expression, context);
  if (pointer) return `(${pointer.buffer} != ${NULL_DEVICE_POINTER_BUFFER})`;
  const value = emitExpression(expression, context);
  if (expressionValueTypeForEmit(expression, context) === "bool") return value;
  return `(${value} != 0)`;
}

function devicePointerValueTypeForExpression(expression: CudaLiteExpression, context: EmitContext): CudaLiteScalarType {
  if (expression.kind === "conditional") {
    return isNullPointerExpression(expression.consequent)
      ? devicePointerValueTypeForExpression(expression.alternate, context)
      : devicePointerValueTypeForExpression(expression.consequent, context);
  }
  if (expression.kind === "cast" && expression.pointer) return expression.valueType;
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    const pointer = expression.args[0];
    return pointer ? devicePointerValueTypeForExpression(pointer, context) : "float";
  }
  const root = rootIdentifier(expression);
  if (root) {
    const handle = context.localPointerHandleFor(root);
    if (handle) return handle.valueType;
    const alias = context.pointerAliasFor(root);
    if (alias?.valueType) return alias.valueType;
    const param = context.devicePointerParamFor(root) ?? context.paramFor(root);
    if (param?.pointer) return param.valueType;
    const global = context.deviceGlobalFor(root);
    if (global) return global.valueType;
  }
  return "float";
}

function devicePointerParamForIndex(
  expression: Extract<CudaLiteExpression, { kind: "index" }>,
  context: EmitContext,
): CudaLiteParam | undefined {
  return expression.target.kind === "identifier"
    ? context.devicePointerParamFor(expression.target.name)
    : undefined;
}

function devicePointerIndexExpression(
  name: string,
  index: CudaLiteExpression,
  context: EmitContext,
): string {
  return `(${name}_base + u32(${emitExpression(index, context)}))`;
}

function devicePointerLValue(expression: CudaLiteExpression, context: EmitContext): DevicePointerLValue | undefined {
  if (expression.kind === "member") {
    const base = devicePointerLValue(expression.object, context);
    if (!base || !isCudaVectorType(base.valueType)) return undefined;
    const fieldIndex = cudaVectorFieldIndex(base.valueType, expression.property);
    return fieldIndex === undefined ? undefined : { ...base, fieldIndex };
  }
  if (expression.kind === "index") {
    if (expression.target.kind === "identifier" && !context.devicePointerParamFor(expression.target.name)) return undefined;
    const parts = devicePointerArgumentParts(expression.target, context);
    if (!parts) return undefined;
    const valueType = devicePointerValueTypeForExpression(expression.target, context);
    return {
      buffer: parts.buffer,
      index: `(${parts.base} + u32(${emitExpression(expression.index, context)}))`,
      valueType,
    };
  }
  if (expression.kind === "unary" && expression.operator === "*") {
    const parts = devicePointerArgumentParts(expression.argument, context);
    if (!parts) return undefined;
    return {
      buffer: parts.buffer,
      index: parts.base,
      valueType: devicePointerValueTypeForExpression(expression.argument, context),
    };
  }
  return undefined;
}

function pointerReadHelperName(type: CudaLiteScalarType): string {
  return `bg_ptr_read_${pointerHelperTypeName(type)}`;
}

function pointerWriteHelperName(type: CudaLiteScalarType): string {
  return `bg_ptr_write_${pointerHelperTypeName(type)}`;
}

function pointerAtomicAddHelperName(type: "float" | "double" | "int" | "uint"): string {
  return `bg_ptr_atomicAdd_${pointerHelperTypeName(type)}`;
}

function pointerAtomicExchangeHelperName(type: "float" | "int" | "uint"): string {
  return `bg_ptr_atomicExchange_${pointerHelperTypeName(type)}`;
}

function pointerAtomicCasHelperName(type: "int" | "uint"): string {
  return `bg_ptr_atomicCompareExchange_${pointerHelperTypeName(type)}`;
}

function pointerHelperTypeName(type: CudaLiteScalarType): string {
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
  return type;
}

function emitExpression(expression: CudaLiteExpression, context: EmitContext, mode: EmitMode = "value"): string {
  switch (expression.kind) {
    case "number":
      return expression.raw.includes(".") ? expression.raw : `${expression.raw}`;
    case "string":
      return expression.raw;
    case "initializer":
      throw featureError("unsupported-local-array-init", "braced initializer is only valid in a declaration");
    case "identifier":
      return emitIdentifier(expression.name, context, mode);
    case "cast":
      if (expression.pointer) return emitExpression(expression.expression, context);
      return `${wgslScalar(expression.valueType)}(${emitExpression(expression.expression, context)})`;
    case "member":
      return emitMember(expression, context);
    case "index": {
      const matrixLane = emitMatrixTileLaneAccessExpression(expression, context);
      if (matrixLane) return matrixLane;
      if (expression.target.kind === "identifier") {
        const localType = context.localValueTypeFor(expression.target.name);
        if (isCudaVectorType(localType)) {
          return `${context.nameFor(expression.target.name)}[u32(${emitExpression(expression.index, context)})]`;
        }
      }
      const storageView = storageViewLValue(expression, context);
      if (storageView && isCudaVectorType(storageView.valueType)) {
        return emitVectorStorageReadAt(context.nameFor(storageView.name), storageView.valueType, storageView.index);
      }
      const poolAccess = poolAccessForIndex(expression, context);
      if (poolAccess) return emitPoolRead(poolAccess, context);
      const pointerParam = devicePointerParamForIndex(expression, context);
      if (pointerParam) {
        return `${pointerReadHelperName(pointerParam.valueType)}(${context.nameFor(`${pointerParam.name}_buffer`)}, ${devicePointerIndexExpression(pointerParam.name, expression.index, context)})`;
      }
      if (expression.target.kind === "identifier") {
        const handle = context.localPointerHandleFor(expression.target.name);
        if (handle) {
          const base = context.nameFor(`${handle.name}_base`);
          const buffer = context.nameFor(`${handle.name}_buffer`);
          const index = `(${base} + u32(${emitExpression(expression.index, context)}))`;
          return `${pointerReadHelperName(handle.valueType)}(${buffer}, ${index})`;
        }
        const alias = context.pointerAliasFor(expression.target.name);
        if (alias) {
          if (alias.valueType && isCudaVectorType(alias.valueType)) {
            const index = emitStorageViewIndex(alias.rootName, alias.baseIndex, expression.index, alias.valueType, context);
            return emitVectorStorageReadAt(context.nameFor(alias.rootName), alias.valueType, index);
          }
          return `${context.nameFor(alias.rootName)}[${emitPointerAliasIndex(alias, expression.index, context)}]`;
        }
      }
      const root = rootIdentifier(expression);
      const index = root && expression.target.kind === "identifier"
        ? emitPointerIndex(root, expression.index, context)
        : emitExpression(expression.index, context);
      const param = root ? context.paramFor(root) : undefined;
      const global = root ? context.deviceGlobalFor(root) : undefined;
      if (mode === "value" && param && isCudaVectorType(param.valueType) && expression.target.kind === "identifier") {
        return emitVectorStorageRead(context.nameFor(root!), param.valueType, index);
      }
      if (global && expression.target.kind === "identifier") {
        return mode === "value"
          ? emitDeviceGlobalPointerRead(global, index, context.ir, context)
          : `${context.nameFor(root!)}[${index}]`;
      }
      const access = `${emitExpression(expression.target, context, "lvalue")}[${index}]`;
      if (mode === "value" && param?.valueType === "bool") return `(${access} != 0u)`;
      if (mode === "value" && param && context.ir.atomicParams.includes(param.name)) {
        const loaded = `atomicLoad(&${access})`;
        return param.valueType === "float" || param.valueType === "double" ? `bitcast<f32>(${loaded})` : loaded;
      }
      if (mode === "value" && root && context.isAtomicShared(root)) {
        const shared = sharedDeclarationFor(root, context);
        const loaded = `atomicLoad(&${access})`;
        return shared?.valueType === "float" || shared?.valueType === "double" ? `bitcast<f32>(${loaded})` : loaded;
      }
      return access;
    }
    case "call":
      return emitCall(expression, context);
    case "unary":
      if (expression.operator === "&") return `&${emitExpression(expression.argument, context, "lvalue")}`;
      if (expression.operator === "*") return emitDeref(expression.argument, context);
      return `(${expression.operator}${emitExpression(expression.argument, context)})`;
    case "binary": {
      const pointerComparison = emitPointerComparison(expression, context);
      if (pointerComparison) return pointerComparison;
      const pointerDifference = emitPointerDifference(expression, context);
      if (pointerDifference) return pointerDifference;
      const vectorArithmetic = emitVectorArithmetic(expression, context);
      if (vectorArithmetic) return vectorArithmetic;
      return emitScalarBinaryExpression(expression, context);
    }
    case "conditional":
      return `select(${emitExpression(expression.alternate, context)}, ${emitExpression(expression.consequent, context)}, ${emitExpression(expression.condition, context)})`;
    case "assignment":
      return emitAssignment(expression, context);
    case "update": {
      const pointerRebase = emitPointerRebaseUpdate(expression, context);
      if (pointerRebase) return pointerRebase;
      return expression.prefix
        ? `${expression.operator}${emitExpression(expression.argument, context, "lvalue")}`
        : `${emitExpression(expression.argument, context, "lvalue")}${expression.operator}`;
    }
    case "sequence":
      throw featureError("unsupported-sequence-expression", "comma expressions are only supported in for-loop clauses");
  }
}

function emitForLoopWithContinuing(
  statement: Extract<CudaLiteStatement, { kind: "for" }>,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  const lines: string[] = [];
  if (statement.init?.kind === "var") {
    lines.push(`${prefix}${emitForVar(statement.init, context)};`);
  } else if (statement.init) {
    for (const expression of sequenceItems(statement.init)) lines.push(`${prefix}${emitExpression(expression, context)};`);
  }
  lines.push(`${prefix}loop {`);
  if (statement.condition) lines.push(`${indent(indentLevel + 1)}if (!(${emitExpression(statement.condition, context)})) { break; }`);
  lines.push(...statement.body.flatMap((child) => emitStatement(child, context, indentLevel + 1)));
  if (statement.update) {
    lines.push(`${indent(indentLevel + 1)}continuing {`);
    for (const expression of sequenceItems(statement.update)) {
      lines.push(`${indent(indentLevel + 2)}${emitExpression(expression, context)};`);
    }
    lines.push(`${indent(indentLevel + 1)}}`);
  }
  lines.push(`${prefix}}`);
  return lines;
}

function sequenceItems(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  return expression.kind === "sequence" ? expression.expressions : [expression];
}

function emitAssignmentStatement(
  expression: CudaLiteAssignmentExpression,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  if (expression.right.kind !== "assignment") return [`${prefix}${emitAssignment(expression, context)};`];
  return [
    ...emitAssignmentStatement(expression.right, context, indentLevel),
    `${prefix}${emitAssignment({ ...expression, right: expression.right.left }, context)};`,
  ];
}

function collectLocalNames(statements: readonly CudaLiteStatement[]): ReadonlySet<string> {
  const names = new Set<string>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" || item.kind === "dim3" || item.kind === "cooperative-group") names.add(item.name);
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var") names.add(item.init.name);
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return names;
}

function collectMutableScalarParams(
  statements: readonly CudaLiteStatement[],
  params: readonly CudaLiteParam[],
): readonly CudaLiteParam[] {
  const paramByName = new Map(params.filter(isMutableKernelValueParam).map((param) => [param.name, param] as const));
  const mutated = new Set<string>();
  walkCudaLiteExpressions(statements, (expression) => {
    const name = mutatedIdentifierName(expression);
    if (name && paramByName.has(name)) mutated.add(name);
  });
  return params.filter((param) => mutated.has(param.name));
}

function mutatedIdentifierName(expression: CudaLiteExpression): string | undefined {
  if (expression.kind === "assignment" && expression.left.kind === "identifier") return expression.left.name;
  if (expression.kind === "update" && expression.argument.kind === "identifier") return expression.argument.name;
  return undefined;
}

function isMutableKernelValueParam(param: CudaLiteParam): boolean {
  return !param.pointer &&
    param.cooperativeGroupKind === undefined &&
    !isSurfaceParam(param) &&
    !isTextureParam(param) &&
    param.valueType !== "devicepool" &&
    param.valueType !== "voidptr";
}

function collectLocalArrays(statements: readonly CudaLiteStatement[]): ReadonlyMap<string, CudaLiteVarDecl> {
  const arrays = new Map<string, CudaLiteVarDecl>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.storage === "local" && (item.dimensions.length > 0 || item.matrixTile)) arrays.set(item.name, item);
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.storage === "local" && (item.init.dimensions.length > 0 || item.init.matrixTile)) {
          arrays.set(item.init.name, item.init);
        }
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return arrays;
}

function collectLocalValueTypes(statements: readonly CudaLiteStatement[]): ReadonlyMap<string, CudaLiteScalarType> {
  const types = new Map<string, CudaLiteScalarType>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.storage === "local" && !item.pointer && item.dimensions.length === 0) {
        types.set(item.name, item.valueType);
      }
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.storage === "local" && !item.init.pointer && item.init.dimensions.length === 0) {
          types.set(item.init.name, item.init.valueType);
        }
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return types;
}

function emitFillRegsStatement(
  expression: CudaLiteExpression,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (expression.kind !== "call") return undefined;
  const name = expressionName(expression.callee);
  if (name !== "fill_1D_regs" && name !== "fill_2D_regs" && name !== "fill_3D_regs") return undefined;
  const target = expression.args[0];
  const value = expression.args[1];
  if (target?.kind !== "identifier" || !value) return undefined;
  const array = context.localArrayFor(target.name);
  if (!array) return undefined;
  return emitLocalArrayFill(context.nameFor(target.name), array.dimensions, emitExpression(value, context), indentLevel);
}

interface EmittedMatrixTile {
  readonly name: string;
  readonly spec: MatrixTileResolvedSpec;
  readonly base: string;
}

function emitWmmaStatement(
  expression: CudaLiteExpression,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (expression.kind !== "call") return undefined;
  const builtin = wmmaBuiltinName(expressionName(expression.callee));
  if (!builtin) return undefined;
  switch (builtin) {
    case "fill_fragment":
      return emitWmmaFillFragment(expression, context, indentLevel);
    case "load_matrix_sync":
      return emitWmmaLoadMatrixSync(expression, context, indentLevel);
    case "mma_sync":
      return emitWmmaMmaSync(expression, context, indentLevel);
    case "store_matrix_sync":
      return emitWmmaStoreMatrixSync(expression, context, indentLevel);
  }
}

function emitWmmaFillFragment(
  expression: CudaLiteCallExpression,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const tile = emitMatrixTileRef(expression.args[0], context, "wmma::fill_fragment");
  const value = emitMatrixTileValueForStore(emitExpression(expression.args[1]!, context), tile.spec);
  const index = `bg_wmma_i_${indentLevel}`;
  const prefix = indent(indentLevel);
  return [
    `${prefix}for (var ${index}: u32 = 0u; ${index} < ${matrixTileElementCount(tile.spec)}u; ${index} = ${index} + 1u) {`,
    `${indent(indentLevel + 1)}${emitMatrixTileAccess(tile, index)} = ${value};`,
    `${prefix}}`,
  ];
}

function emitWmmaLoadMatrixSync(
  expression: CudaLiteCallExpression,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const tile = emitMatrixTileRef(expression.args[0], context, "wmma::load_matrix_sync");
  const source = devicePointerArgumentParts(expression.args[1]!, context);
  if (!source) throw featureError("unsupported-wmma-pointer-operand", "wmma::load_matrix_sync source expects storage/shared pointer");
  const stride = `u32(${emitExpression(expression.args[2]!, context)})`;
  const layout = emitMatrixTileLayoutForCall(expression.args[3], tile.spec.layout ?? "row_major");
  const [rows, cols] = matrixTileRowsCols(tile.spec);
  const row = `bg_wmma_row_${indentLevel}`;
  const col = `bg_wmma_col_${indentLevel}`;
  const prefix = indent(indentLevel);
  const memIndex = emitMatrixTileMemoryIndex(row, col, stride, layout);
  const tileIndex = `(${row} * ${cols}u + ${col})`;
  const read = `${pointerReadHelperName(tile.spec.valueType)}(${source.buffer}, (${source.base} + ${memIndex}))`;
  return [
    `${prefix}for (var ${row}: u32 = 0u; ${row} < ${rows}u; ${row} = ${row} + 1u) {`,
    `${indent(indentLevel + 1)}for (var ${col}: u32 = 0u; ${col} < ${cols}u; ${col} = ${col} + 1u) {`,
    `${indent(indentLevel + 2)}${emitMatrixTileAccess(tile, tileIndex)} = ${emitMatrixTileValueForStore(read, tile.spec)};`,
    `${indent(indentLevel + 1)}}`,
    `${prefix}}`,
  ];
}

function emitWmmaMmaSync(
  expression: CudaLiteCallExpression,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const dst = emitMatrixTileRef(expression.args[0], context, "wmma::mma_sync destination");
  const a = emitMatrixTileRef(expression.args[1], context, "wmma::mma_sync A");
  const b = emitMatrixTileRef(expression.args[2], context, "wmma::mma_sync B");
  const c = emitMatrixTileRef(expression.args[3], context, "wmma::mma_sync accumulator");
  const row = `bg_wmma_row_${indentLevel}`;
  const col = `bg_wmma_col_${indentLevel}`;
  const kk = `bg_wmma_k_${indentLevel}`;
  const sum = `bg_wmma_sum_${indentLevel}`;
  const prefix = indent(indentLevel);
  const dstIndex = `(${row} * ${dst.spec.n}u + ${col})`;
  const aIndex = `(${row} * ${dst.spec.k}u + ${kk})`;
  const bIndex = `(${kk} * ${dst.spec.n}u + ${col})`;
  const integerMma = dst.spec.tileValueType === "s32" &&
    isMatrixTileByteValueType(a.spec.tileValueType) &&
    isMatrixTileByteValueType(b.spec.tileValueType);
  const sumType = integerMma ? "i32" : "f32";
  const sumInit = integerMma
    ? emitMatrixTileIntegerValue(emitMatrixTileAccess(c, dstIndex), c.spec)
    : `f32(${emitMatrixTileAccess(c, dstIndex)})`;
  const multiply = integerMma
    ? `(${emitMatrixTileIntegerValue(emitMatrixTileAccess(a, aIndex), a.spec)} * ${emitMatrixTileIntegerValue(emitMatrixTileAccess(b, bIndex), b.spec)})`
    : `f32(${emitMatrixTileAccess(a, aIndex)}) * f32(${emitMatrixTileAccess(b, bIndex)})`;
  return [
    `${prefix}for (var ${row}: u32 = 0u; ${row} < ${dst.spec.m}u; ${row} = ${row} + 1u) {`,
    `${indent(indentLevel + 1)}for (var ${col}: u32 = 0u; ${col} < ${dst.spec.n}u; ${col} = ${col} + 1u) {`,
    `${indent(indentLevel + 2)}var ${sum}: ${sumType} = ${sumInit};`,
    `${indent(indentLevel + 2)}for (var ${kk}: u32 = 0u; ${kk} < ${dst.spec.k}u; ${kk} = ${kk} + 1u) {`,
    `${indent(indentLevel + 3)}${sum} = ${sum} + ${multiply};`,
    `${indent(indentLevel + 2)}}`,
    `${indent(indentLevel + 2)}${emitMatrixTileAccess(dst, dstIndex)} = ${emitMatrixTileValueForStore(sum, dst.spec)};`,
    `${indent(indentLevel + 1)}}`,
    `${prefix}}`,
  ];
}

function emitWmmaStoreMatrixSync(
  expression: CudaLiteCallExpression,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const target = devicePointerArgumentParts(expression.args[0]!, context);
  if (!target) throw featureError("unsupported-wmma-pointer-operand", "wmma::store_matrix_sync destination expects storage/shared pointer");
  const tile = emitMatrixTileRef(expression.args[1], context, "wmma::store_matrix_sync fragment");
  const stride = `u32(${emitExpression(expression.args[2]!, context)})`;
  const layout = emitMatrixTileLayoutForCall(expression.args[3], "mem_row_major");
  const [rows, cols] = matrixTileRowsCols(tile.spec);
  const row = `bg_wmma_row_${indentLevel}`;
  const col = `bg_wmma_col_${indentLevel}`;
  const prefix = indent(indentLevel);
  const memIndex = emitMatrixTileMemoryIndex(row, col, stride, layout);
  const tileIndex = `(${row} * ${cols}u + ${col})`;
  return [
    `${prefix}for (var ${row}: u32 = 0u; ${row} < ${rows}u; ${row} = ${row} + 1u) {`,
    `${indent(indentLevel + 1)}for (var ${col}: u32 = 0u; ${col} < ${cols}u; ${col} = ${col} + 1u) {`,
    `${indent(indentLevel + 2)}${pointerWriteHelperName(tile.spec.valueType)}(${target.buffer}, (${target.base} + ${memIndex}), ${emitMatrixTileAccess(tile, tileIndex)});`,
    `${indent(indentLevel + 1)}}`,
    `${prefix}}`,
  ];
}

function emitMatrixTileRef(
  expression: CudaLiteExpression | undefined,
  context: EmitContext,
  label: string,
): EmittedMatrixTile {
  if (!expression) throw featureError("unsupported-wmma-fragment-operand", `${label} expects WMMA fragment argument`);
  const ref = matrixTileReference(expression);
  if (!ref) throw featureError("unsupported-wmma-fragment-operand", `${label} expects WMMA fragment argument`);
  const declaration = context.localArrayFor(ref.root);
  const spec = declaration?.matrixTile ? resolveMatrixTileSpec(declaration.matrixTile) : undefined;
  if (!declaration || !spec) throw featureError("unsupported-wmma-fragment-operand", `'${ref.root}' is not a WMMA fragment`);
  const count = matrixTileElementCount(spec);
  const base = emitMatrixTileBase(ref.indices, declaration.dimensions, count, context);
  return { name: context.nameFor(ref.root), spec, base };
}

function emitMatrixTileBase(
  indices: readonly CudaLiteExpression[],
  dimensions: readonly number[],
  elementCount: number,
  context: EmitContext,
): string {
  if (indices.length !== dimensions.length) return "0u";
  if (indices.length === 0) return "0u";
  const terms = indices.map((index, axis) => {
    const stride = dimensions.slice(axis + 1).reduce((product, dimension) => product * dimension, elementCount);
    const value = `u32(${emitExpression(index, context)})`;
    return stride === 1 ? value : `(${value} * ${stride}u)`;
  });
  return terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
}

function emitMatrixTileAccess(tile: EmittedMatrixTile, index: string): string {
  return tile.base === "0u" ? `${tile.name}[${index}]` : `${tile.name}[(${tile.base} + ${index})]`;
}

function matrixTileRowsCols(tile: MatrixTileResolvedSpec): readonly [number, number] {
  if (tile.role === "matrix_a") return [tile.m, tile.k];
  if (tile.role === "matrix_b") return [tile.k, tile.n];
  return [tile.m, tile.n];
}

function emitMatrixTileMemoryIndex(row: string, col: string, stride: string, layout: MatrixTileLayout): string {
  return layout === "col_major" || layout === "mem_col_major"
    ? `(${col} * ${stride} + ${row})`
    : `(${row} * ${stride} + ${col})`;
}

function emitMatrixTileLayoutForCall(
  expression: CudaLiteExpression | undefined,
  fallback: MatrixTileLayout,
): MatrixTileLayout {
  if (!expression) return fallback;
  return normalizeMatrixTileLayout(expressionName(expression)) ?? fallback;
}

function emitMatrixTileValueForStore(value: string, tile: MatrixTileResolvedSpec): string {
  switch (tile.tileValueType) {
    case "f16":
      return `f16(${value})`;
    case "u8":
      return `(u32(${value}) & 0xffu)`;
    case "s8":
    case "s32":
      return `i32(${value})`;
    default:
      return value;
  }
}

function emitMatrixTileIntegerValue(value: string, tile: MatrixTileResolvedSpec): string {
  if (tile.tileValueType === "u8") return `i32(u32(${value}) & 0xffu)`;
  return `i32(${value})`;
}

function emitCpAsyncStatement(
  expression: CudaLiteExpression,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (expression.kind !== "call") return undefined;
  const name = expressionName(expression.callee);
  const prefix = indent(indentLevel);
  if (isCpAsyncFenceCall(name)) return [`${prefix}workgroupBarrier();`];
  if (!isCpAsyncCopyCall(name)) return undefined;
  const [dst, src, bytes] = expression.args;
  if (!dst || !src) return [`${prefix}// cp.async omitted: missing pointer operand`];
  const dstParts = devicePointerArgumentParts(dst, context);
  const srcParts = devicePointerArgumentParts(src, context);
  if (!dstParts || !srcParts) return [`${prefix}// cp.async byte-address copy omitted: ${name}`];
  const valueType = devicePointerValueTypeForExpression(src, context);
  const count = cpAsyncElementCount(bytes, valueType);
  const lines: string[] = [];
  for (let index = 0; index < count; index++) {
    const srcIndex = index === 0 ? srcParts.base : `(${srcParts.base} + ${index}u)`;
    const dstIndex = index === 0 ? dstParts.base : `(${dstParts.base} + ${index}u)`;
    lines.push(`${prefix}${pointerWriteHelperName(valueType)}(${dstParts.buffer}, ${dstIndex}, ${pointerReadHelperName(valueType)}(${srcParts.buffer}, ${srcIndex}));`);
  }
  return lines;
}

function isCpAsyncCopyCall(name: string | undefined): boolean {
  return name === "CP_ASYNC_CA" || name === "CP_ASYNC_CG" || name === "CP_ASYNC_BULK";
}

function isCpAsyncFenceCall(name: string | undefined): boolean {
  return name === "CP_ASYNC_COMMIT_GROUP" ||
    name === "CP_ASYNC_WAIT_ALL" ||
    name === "CP_ASYNC_WAIT_GROUP" ||
    name === "CP_ASYNC_BULK_COMMIT_GROUP" ||
    name === "CP_ASYNC_BULK_WAIT_ALL" ||
    name === "CP_ASYNC_BULK_WAIT_GROUP";
}

function cpAsyncElementCount(bytes: CudaLiteExpression | undefined, valueType: CudaLiteScalarType): number {
  if (bytes?.kind !== "number") return 1;
  const elementBytes = wgslElementByteSize(valueType);
  if (elementBytes <= 0) return 1;
  return Math.max(1, Math.min(16, Math.floor(bytes.value / elementBytes)));
}

function wgslElementByteSize(valueType: CudaLiteScalarType): number {
  if (valueType === "half") return 2;
  if (valueType === "bf16") return 2;
  if (isCudaVectorType(valueType)) return cudaVectorLaneCount(valueType) * wgslElementByteSize(cudaVectorScalarType(valueType) ?? "float");
  return 4;
}

function emitVectorConstructor(vectorType: CudaLiteScalarType, args: readonly string[]): string {
  const info = CUDA_VECTOR_TYPES.get(vectorType as never);
  if (!info) return `${wgslScalar(vectorType)}(${args.join(", ")})`;
  if (args.length === 1) return `${wgslScalar(vectorType)}(${Array(info.lanes).fill(args[0] ?? "0").join(", ")})`;
  return `${wgslScalar(vectorType)}(${args.join(", ")})`;
}

function emitVectorConversionConstructor(
  targetType: CudaLiteVectorType,
  expression: CudaLiteExpression,
  context: EmitContext,
): string {
  const sourceType = expressionValueTypeForEmit(expression, context);
  if (!isCudaVectorType(sourceType) || cudaVectorScalarType(sourceType) !== cudaVectorScalarType(targetType)) {
    return emitVectorSplat(targetType, castExpressionToVectorScalar(emitExpression(expression, context), targetType));
  }
  const value = emitExpression(expression, context);
  const fields = ["x", "y", "z", "w"];
  const scalar = cudaVectorScalarType(targetType) ?? "float";
  const lanes = Array.from({ length: cudaVectorLaneCount(targetType) }, (_unused, index) =>
    index < cudaVectorLaneCount(sourceType) ? `${value}.${fields[index]!}` : zeroValue(scalar));
  return `${wgslScalar(targetType)}(${lanes.join(", ")})`;
}

function emitMixedVectorConstructor(
  targetType: CudaLiteVectorType,
  expressions: readonly CudaLiteExpression[],
  context: EmitContext,
): string {
  if (!expressions.some((expression) => isCudaVectorType(expressionValueTypeForEmit(expression, context)))) {
    return emitVectorConstructor(targetType, expressions.map((expression) => emitExpression(expression, context)));
  }
  const lanes: string[] = [];
  const targetScalar = cudaVectorScalarType(targetType) ?? "float";
  for (const expression of expressions) {
    const sourceType = expressionValueTypeForEmit(expression, context);
    const value = emitExpression(expression, context);
    if (isCudaVectorType(sourceType) && cudaVectorScalarType(sourceType) === targetScalar) {
      for (let lane = 0; lane < cudaVectorLaneCount(sourceType); lane++) {
        lanes.push(castExpressionToVectorScalar(`${value}.${vectorFieldName(lane)}`, targetType));
      }
    } else {
      lanes.push(castExpressionToVectorScalar(value, targetType));
    }
  }
  while (lanes.length < cudaVectorLaneCount(targetType)) lanes.push(zeroValue(targetScalar));
  return `${wgslScalar(targetType)}(${lanes.slice(0, cudaVectorLaneCount(targetType)).join(", ")})`;
}

function emitLocalArrayFill(
  name: string,
  dimensions: readonly number[],
  value: string,
  indentLevel: number,
  indexes: readonly string[] = [],
): string[] {
  if (indexes.length === dimensions.length) {
    return [`${indent(indentLevel)}${name}${indexes.map((index) => `[${index}]`).join("")} = ${value};`];
  }
  const loopName = `fill_${name}_${indexes.length}`;
  const lines = [
    `${indent(indentLevel)}for (var ${loopName}: i32 = 0; ${loopName} < ${dimensions[indexes.length] ?? 0}; ${loopName} = ${loopName} + 1) {`,
  ];
  lines.push(...emitLocalArrayFill(name, dimensions, value, indentLevel + 1, [...indexes, loopName]));
  lines.push(`${indent(indentLevel)}}`);
  return lines;
}

function collectLocalPointerHandles(
  statements: readonly CudaLiteStatement[],
  poolPointers: ReadonlyMap<string, PoolPointerAlias> = collectPoolPointers(statements),
  structuredPointerRoots: ReadonlySet<string> = new Set(),
): ReadonlyMap<string, CudaLiteVarDecl> {
  const mutableNames = collectMutableLocalPointerNames(statements);
  const handles = new Map<string, CudaLiteVarDecl>();
  const needsHandle = (statement: CudaLiteVarDecl): boolean =>
    statement.pointer &&
    statement.storage === "local" &&
    !poolPointers.has(statement.name) &&
    (mutableNames.has(statement.name) ||
      needsStructuredAddressHandle(statement.init, structuredPointerRoots) ||
      needsDynamicPointerHandle(statement.init));
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && needsHandle(item)) {
        handles.set(item.name, item);
      }
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && needsHandle(item.init)) {
          handles.set(item.init.name, item.init);
        }
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return handles;
}

function structuredPointerHandleRoots(ir: KernelIrModule): ReadonlySet<string> {
  return new Set([
    ...ir.constants.map((constant) => constant.name),
    ...ir.deviceGlobals.map((global) => global.name),
    ...ir.sharedDeclarations.map((shared) => shared.name),
  ]);
}

function needsStructuredAddressHandle(
  expression: CudaLiteExpression | undefined,
  structuredPointerRoots: ReadonlySet<string>,
): boolean {
  if (!expression) return false;
  if (expression.kind === "cast" && expression.pointer) return needsStructuredAddressHandle(expression.expression, structuredPointerRoots);
  if (expression.kind === "conditional") {
    return needsStructuredAddressHandle(expression.consequent, structuredPointerRoots) ||
      needsStructuredAddressHandle(expression.alternate, structuredPointerRoots);
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    return needsStructuredAddressHandle(expression.left, structuredPointerRoots);
  }
  if (expression.kind !== "unary" || expression.operator !== "&") return false;
  let depth = 0;
  let cursor = expression.argument;
  while (cursor.kind === "index") {
    depth++;
    cursor = cursor.target;
  }
  return depth > 0 && cursor.kind === "identifier" && structuredPointerRoots.has(cursor.name);
}

function needsDynamicPointerHandle(expression: CudaLiteExpression | undefined): boolean {
  if (!expression) return false;
  if (expression.kind === "cast" && expression.pointer) return needsDynamicPointerHandle(expression.expression);
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    return needsDynamicPointerHandle(expression.args[0]);
  }
  if (expression.kind === "conditional") return true;
  return false;
}

function collectMutableLocalPointerNames(statements: readonly CudaLiteStatement[]): ReadonlySet<string> {
  const declared = new Set<string>();
  const mutated = new Set<string>();
  const walkStatements = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.pointer && item.storage === "local") declared.add(item.name);
      if (item.kind === "for" && item.init?.kind === "var" && item.init.pointer && item.init.storage === "local") declared.add(item.init.name);
      if (item.kind === "if") {
        walkStatements(item.consequent);
        if (item.alternate) walkStatements(item.alternate);
      }
      if (item.kind === "for") walkStatements(item.body);
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walkStatements(item.body);
    }
  };
  walkStatements(statements);
  walkCudaLiteExpressions(statements, (expression) => {
    if (expression.kind === "assignment" && expression.left.kind === "identifier" && declared.has(expression.left.name)) {
      mutated.add(expression.left.name);
    }
    if (expression.kind === "update" && expression.argument.kind === "identifier" && declared.has(expression.argument.name)) {
      mutated.add(expression.argument.name);
    }
  });
  return mutated;
}

function collectPointerAliases(
  statements: readonly CudaLiteStatement[],
  skipNames: ReadonlySet<string> = new Set(),
): ReadonlyMap<string, PointerAlias> {
  const aliases = new Map<string, PointerAlias>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.pointer && !skipNames.has(item.name)) {
        const alias = pointerAliasForVar(item);
        if (alias) aliases.set(item.name, alias);
      }
      if (item.kind === "dim3" || item.kind === "cooperative-group" || item.kind === "kernel-launch") continue;
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for" || item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return aliases;
}

function collectMutableStoragePointerBases(
  statements: readonly CudaLiteStatement[],
  pointerParamNames: ReadonlySet<string>,
): readonly string[] {
  const names = new Set<string>();
  walkCudaLiteExpressions(statements, (expression) => {
    if (expression.kind === "assignment" && expression.left.kind === "identifier") {
      if ((expression.operator === "=" || expression.operator === "+=" || expression.operator === "-=") && pointerParamNames.has(expression.left.name)) {
        names.add(expression.left.name);
      }
      return;
    }
    if (expression.kind === "update" && expression.argument.kind === "identifier") {
      if (pointerParamNames.has(expression.argument.name)) names.add(expression.argument.name);
    }
  });
  return [...names].sort();
}

function collectPoolPointers(statements: readonly CudaLiteStatement[]): ReadonlyMap<string, PoolPointerAlias> {
  const aliases = new Map<string, PoolPointerAlias>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.pointer) {
        const alias = poolPointerForInitializer(item.init, aliases);
        if (alias) aliases.set(item.name, alias);
      }
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.pointer) {
          const alias = poolPointerForInitializer(item.init.init, aliases);
          if (alias) aliases.set(item.init.name, alias);
        }
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return aliases;
}

function poolPointerForInitializer(
  init: CudaLiteExpression | undefined,
  aliases: ReadonlyMap<string, PoolPointerAlias>,
): PoolPointerAlias | undefined {
  if (!init) return undefined;
  if (init.kind === "call") return poolPointerForAllocationCall(init);
  if (init.kind === "cast" && init.pointer) return poolPointerForInitializer(init.expression, aliases);
  if (init.kind === "identifier") return aliases.get(init.name);
  return undefined;
}

function poolPointerForAllocationCall(call: CudaLiteCallExpression): PoolPointerAlias | undefined {
  const name = expressionName(call.callee);
  if (name !== "deviceAllocate" && name !== "streamOrderedAllocate") return undefined;
  if (call.args.length === 4) {
    const base = call.args[0];
    const offset = call.args[1];
    return base?.kind === "identifier" && offset?.kind === "identifier"
      ? { poolName: base.name, offsetName: offset.name, rawBuffer: true }
      : undefined;
  }
  const pool = call.args[0];
  if (pool?.kind === "unary" && pool.operator === "&" && pool.argument.kind === "identifier") {
    return { poolName: pool.argument.name };
  }
  return pool?.kind === "identifier" ? { poolName: pool.name } : undefined;
}

function collectRawPoolAllocators(statements: readonly CudaLiteStatement[]): readonly RawPoolAllocator[] {
  const allocators = new Map<string, RawPoolAllocator>();
  const visitExpression = (expression: CudaLiteExpression): void => {
    if (expression.kind === "call") {
      const alias = poolPointerForAllocationCall(expression);
      if (alias?.rawBuffer && alias.offsetName) {
        const key = `${alias.poolName}\0${alias.offsetName}`;
        allocators.set(key, { baseName: alias.poolName, offsetName: alias.offsetName });
      }
      visitExpression(expression.callee);
      for (const arg of expression.args) visitExpression(arg);
      return;
    }
    if (expression.kind === "cast") visitExpression(expression.expression);
    else if (expression.kind === "member") visitExpression(expression.object);
    else if (expression.kind === "index") {
      visitExpression(expression.target);
      visitExpression(expression.index);
    } else if (expression.kind === "unary" || expression.kind === "update") visitExpression(expression.argument);
    else if (expression.kind === "binary") {
      visitExpression(expression.left);
      visitExpression(expression.right);
    } else if (expression.kind === "conditional") {
      visitExpression(expression.condition);
      visitExpression(expression.consequent);
      visitExpression(expression.alternate);
    } else if (expression.kind === "assignment") {
      visitExpression(expression.left);
      visitExpression(expression.right);
    } else if (expression.kind === "sequence") {
      for (const item of expression.expressions) visitExpression(item);
    }
  };
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.init) visitExpression(item.init);
      if (item.kind === "expr") visitExpression(item.expression);
      if (item.kind === "if") {
        visitExpression(item.condition);
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.init) visitExpression(item.init.init);
        else if (item.init && item.init.kind !== "var") visitExpression(item.init);
        if (item.condition) visitExpression(item.condition);
        if (item.update) visitExpression(item.update);
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while") {
        walk(item.body);
        visitExpression(item.condition);
      }
      if (item.kind === "block") walk(item.body);
      if (item.kind === "return" && item.value) visitExpression(item.value);
    }
  };
  walk(statements);
  return [...allocators.values()];
}

function collectCooperativeGroups(statements: readonly CudaLiteStatement[]): ReadonlyMap<string, CudaLiteCooperativeGroupDecl> {
  const groups = new Map<string, CudaLiteCooperativeGroupDecl>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "cooperative-group") groups.set(item.name, item);
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for" || item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return groups;
}

function pointerAliasForVar(statement: CudaLiteVarDecl): PointerAlias | undefined {
  const init = statement.init;
  const view = pointerAliasForPointerExpression(init, statement.valueType);
  if (view) return view;
  if (init?.kind !== "unary" || init.operator !== "&") return undefined;
  const target = init.argument;
  if (target.kind !== "index" || target.target.kind !== "identifier") return undefined;
  return {
    rootName: target.target.name,
    baseIndex: target.index,
  };
}

function pointerAliasForPointerExpression(
  expression: CudaLiteExpression | undefined,
  valueType: CudaLiteScalarType,
): PointerAlias | undefined {
  if (!expression) return undefined;
  if (expression.kind === "cast" && expression.pointer) return pointerAliasForPointerExpression(expression.expression, valueType);
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    return pointerAliasForPointerExpression(expression.args[0], valueType);
  }
  if (expression.kind === "unary" && expression.operator === "&") {
    const target = expression.argument;
    if (target.kind === "index" && target.target.kind === "identifier") {
      return { rootName: target.target.name, baseIndex: target.index, valueType };
    }
    if (target.kind === "identifier") {
      return { rootName: target.name, baseIndex: zeroExpression(target.span), valueType };
    }
    return undefined;
  }
  if (expression.kind === "identifier") {
    return { rootName: expression.name, baseIndex: zeroExpression(expression.span), valueType };
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const base = pointerAliasForPointerExpression(expression.left, valueType);
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

function isPointerIdentityCall(name: string | undefined): boolean {
  return name === "__builtin_assume_aligned" || name === "ct::assume_aligned";
}

function emitPointerAliasIndex(alias: PointerAlias, index: CudaLiteExpression, context: EmitContext): string {
  const base = pointerBaseExpression(alias.rootName, context);
  if (!base) return `(${emitExpression(alias.baseIndex, context)}) + (${emitExpression(index, context)})`;
  return `(${base} + u32(${emitExpression(alias.baseIndex, context)}) + u32(${emitExpression(index, context)}))`;
}

function emitStorageViewIndex(
  rootName: string,
  baseIndex: CudaLiteExpression,
  index: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: EmitContext,
): string {
  const base = pointerBaseExpression(rootName, context);
  const prefix = base
    ? `${base} + u32(${emitExpression(baseIndex, context)})`
    : `u32(${emitExpression(baseIndex, context)})`;
  const lanes = cudaVectorLaneCount(valueType);
  return `(${prefix} + (u32(${emitExpression(index, context)}) * ${lanes}u))`;
}

function emitVectorStorageReadAt(name: string, type: CudaLiteScalarType, storageIndex: string): string {
  const lanes = cudaVectorLaneCount(type);
  const values = Array.from({ length: lanes }, (_, lane) => `${name}[${storageIndex} + ${lane}u]`);
  return `${wgslScalar(type)}(${values.join(", ")})`;
}

function emitVectorStorageWriteAt(name: string, type: CudaLiteScalarType, storageIndex: string, value: string): string {
  const lanes = cudaVectorLaneCount(type);
  return Array.from({ length: lanes }, (_, lane) => `${name}[${storageIndex} + ${lane}u] = ${value}.${vectorFieldName(lane)}`).join("; ");
}

function emitVectorStorageFieldWriteAt(name: string, storageIndex: string, fieldIndex: number, value: string): string {
  return `${name}[${storageIndex} + ${fieldIndex}u] = ${value}`;
}

function zeroExpression(span: CudaLiteExpression["span"]): CudaLiteExpression {
  return { kind: "number", value: 0, raw: "0", span };
}

function emitPointerIndex(rootName: string, index: CudaLiteExpression, context: EmitContext): string {
  const base = pointerBaseExpression(rootName, context);
  if (!base) return emitExpression(index, context);
  return `(${base} + u32(${emitExpression(index, context)}))`;
}

function pointerBaseExpression(rootName: string, context: EmitContext): string | undefined {
  const mutable = context.mutablePointerBaseFor(rootName);
  if (mutable) return mutable;
  const base = context.pointerBaseOffsetFieldFor(rootName);
  return base ? `params.${context.nameFor(base)}` : undefined;
}

interface PointerComparisonTerm {
  readonly kind: "static" | "token" | "address";
  readonly value?: string;
  readonly buffer?: string;
  readonly base?: string;
  readonly pointerish: boolean;
  readonly nullish: boolean;
  readonly symbol?: string;
}

function emitPointerComparison(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): string | undefined {
  if (expression.operator !== "==" && expression.operator !== "!=") return undefined;
  const left = pointerComparisonTerm(expression.left, context);
  const right = pointerComparisonTerm(expression.right, context);
  if (!left || !right) return undefined;
  if (!left.pointerish && !right.pointerish && !left.nullish && !right.nullish) return undefined;

  if (left.kind === "static" && right.kind === "static") {
    const equal = staticPointerTermEqual(left, right);
    if (equal === undefined) {
      throw featureError(
        "unsupported-pointer-pointer-comparison",
        "WGSL output supports pointer comparison only against NULL/nullptr or the same pointer symbol",
      );
    }
    return expression.operator === "==" ? String(equal) : String(!equal);
  }

  if (left.kind === "address" && right.kind === "address") {
    const equal = `((${left.buffer} == ${right.buffer}) && (${left.base} == ${right.base}))`;
    return expression.operator === "==" ? equal : `(!${equal})`;
  }

  if (left.kind === "token" && right.kind === "token") {
    return `(${left.value} ${expression.operator} ${right.value})`;
  }

  const token = left.kind === "token" ? left : right.kind === "token" ? right : undefined;
  const address = left.kind === "address" ? left : right.kind === "address" ? right : undefined;
  const stat = left.kind === "static" ? left : right.kind === "static" ? right : undefined;
  if (token && stat?.nullish) {
    return `(${token.value} ${expression.operator} 0u)`;
  }
  if (address && stat?.nullish) {
    const buffer = address.buffer ?? "0u";
    if (/^\d+u$/u.test(buffer)) {
      const equal = buffer === NULL_DEVICE_POINTER_BUFFER;
      return expression.operator === "==" ? String(equal) : String(!equal);
    }
    const equal = `(${buffer} == ${NULL_DEVICE_POINTER_BUFFER})`;
    return expression.operator === "==" ? equal : `(!${equal})`;
  }
  throw featureError(
    "unsupported-pointer-pointer-comparison",
    "WGSL output supports pointer comparison only inside the same pointer address model",
  );
}

function emitPointerDifference(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): string | undefined {
  if (expression.operator !== "-") return undefined;
  const left = devicePointerArgumentParts(expression.left, context);
  const right = devicePointerArgumentParts(expression.right, context);
  if (!left || !right) return undefined;
  const diff = `(i32(${left.base}) - i32(${right.base}))`;
  return left.buffer === right.buffer
    ? diff
    : `select(0, ${diff}, (${left.buffer} == ${right.buffer}))`;
}

function staticPointerTermEqual(left: PointerComparisonTerm, right: PointerComparisonTerm): boolean | undefined {
  if (left.nullish || right.nullish) return left.nullish === right.nullish && left.value === right.value;
  if (left.symbol && right.symbol && left.symbol === right.symbol) return true;
  return undefined;
}

function isNullPointerExpression(expression: CudaLiteExpression): boolean {
  if (expression.kind === "number") return expression.value === 0;
  if (expression.kind !== "identifier") return false;
  if (expression.name === "nullptr" || expression.name === "NULL") return true;
  const namedConstant = CUDA_NAMED_CONSTANTS.get(expression.name);
  return namedConstant?.valueType === "voidptr" && namedConstant.value === 0;
}

function pointerComparisonTerm(expression: CudaLiteExpression, context: EmitContext): PointerComparisonTerm | undefined {
  if (expression.kind === "identifier") {
    if (expression.name === "nullptr") return { kind: "static", value: "null", pointerish: false, nullish: true };
    const namedConstant = CUDA_NAMED_CONSTANTS.get(expression.name);
    if (namedConstant?.valueType === "voidptr" && namedConstant.value === 0) {
      return { kind: "static", value: "null", pointerish: false, nullish: true };
    }
    if (context.poolPointerFor(expression.name)) {
      return { kind: "token", value: emitIdentifier(expression.name, context), pointerish: true, nullish: false, symbol: expression.name };
    }
    const localType = context.localValueTypeFor(expression.name);
    if (localType === "voidptr") {
      return { kind: "token", value: emitIdentifier(expression.name, context), pointerish: true, nullish: false, symbol: expression.name };
    }
  }
  if (expression.kind === "number" && expression.value === 0) {
    return { kind: "static", value: "null", pointerish: false, nullish: true };
  }
  if (expression.kind === "cast" && expression.pointer) {
    return pointerComparisonTerm(expression.expression, context);
  }
  if (expression.kind === "call") {
    const pool = poolPointerForAllocationCall(expression);
    if (pool) return { kind: "token", value: emitCall(expression, context), pointerish: true, nullish: false };
  }
  if (expression.kind === "identifier" && context.deviceGlobalFor(expression.name)?.dimensions.length === 0) {
    return undefined;
  }
  const address = devicePointerArgumentParts(expression, context);
  if (address) {
    const symbol = expressionName(expression);
    return {
      kind: "address",
      buffer: address.buffer,
      base: address.base,
      pointerish: true,
      nullish: false,
      ...(symbol === undefined ? {} : { symbol }),
    };
  }
  return undefined;
}

function emitIdentifier(name: string, context: EmitContext, mode: EmitMode = "value"): string {
  if (name === "nullptr") return "0u";
  const namedConstant = CUDA_NAMED_CONSTANTS.get(name);
  if (namedConstant) return namedConstant.wgsl;
  if (name === "threadIdx" || name === "blockIdx" || name === "blockDim" || name === "gridDim") return name;
  if (context.isAtomicShared(name) && mode === "value") {
    const shared = sharedDeclarationFor(name, context);
    const loaded = `atomicLoad(&${context.nameFor(name)})`;
    return shared?.valueType === "float" ? `bitcast<f32>(${loaded})` : loaded;
  }
  if (context.isLocalName(name)) return context.nameFor(name);
  const global = context.deviceGlobalFor(name);
  if (global) {
    if (mode === "lvalue") return `${context.nameFor(name)}[0u]`;
    return emitDeviceGlobalPointerRead(global, "0u", context.ir, context);
  }
  const constant = context.ir.constants.find((item) => item.name === name);
  if (constant && isExternalConstantBinding(constant) && constant.dimensions.length === 0) {
    return emitConstantPointerRead(constant, "0u", context);
  }
  const param = context.paramFor(name);
  if ((param && !param.pointer && !isSurfaceParam(param)) || context.isUniformScalar(name)) {
    return emitUniformScalarRead(name, context);
  }
  return context.nameFor(name);
}

function emitUniformScalarRead(name: string, context: EmitContext): string {
  return context.uniformScalarTypeFor(name) === "bool"
    ? `(params.${context.nameFor(name)} != 0u)`
    : `params.${context.nameFor(name)}`;
}

function sharedDeclarationFor(name: string, context: EmitContext): CudaLiteVarDecl | undefined {
  return context.ir.sharedDeclarations.find((item) => item.name === name);
}

function emitDeref(expression: CudaLiteExpression, context: EmitContext): string {
  if (expression.kind === "identifier") {
    const alias = context.pointerAliasFor(expression.name);
    if (alias) {
      return `${context.nameFor(alias.rootName)}[${emitPointerAliasIndex(alias, zeroExpression(expression.span), context)}]`;
    }
    const param = context.paramFor(expression.name);
    if (param?.pointer && !context.devicePointerParamFor(expression.name)) {
      const access = `${context.nameFor(expression.name)}[${emitPointerIndex(expression.name, zeroExpression(expression.span), context)}]`;
      if (context.ir.atomicParams.includes(param.name)) {
        const loaded = `atomicLoad(&${access})`;
        return param.valueType === "float" || param.valueType === "double" ? `bitcast<f32>(${loaded})` : loaded;
      }
      return access;
    }
  }
  if (expression.kind === "cast" && expression.pointer) {
    const poolPointer = poolPointerExpressionInfo(expression.expression, context);
    if (poolPointer) {
      return emitPoolRead({
        poolName: poolPointer.poolName,
        pointerExpression: expression.expression,
        indexExpression: { kind: "number", value: 0, raw: "0", span: expression.span },
        valueType: expression.valueType,
        ...(poolPointer.rawBuffer ? { rawBuffer: true } : {}),
      }, context);
    }
  }
  const storageView = storageViewForPointerExpression(expression, zeroExpression(expression.span), context);
  if (storageView && isCudaVectorType(storageView.valueType)) {
    return emitVectorStorageReadAt(context.nameFor(storageView.rootName), storageView.valueType, storageView.index);
  }
  const parts = devicePointerArgumentParts(expression, context);
  if (parts) {
    const valueType = devicePointerValueTypeForExpression(expression, context);
    return `${pointerReadHelperName(valueType)}(${parts.buffer}, ${parts.base})`;
  }
  return `*${emitExpression(expression, context)}`;
}

function emitMember(expression: Extract<CudaLiteExpression, { kind: "member" }>, context: EmitContext): string {
  const matrixMember = emitMatrixTileMemberExpression(expression, context);
  if (matrixMember) return matrixMember;
  if (expression.property === "size") {
    const valueType = expressionValueTypeForEmit(expression.object, context);
    if (isCudaVectorType(valueType)) return String(cudaVectorLaneCount(valueType));
  }
  const storageView = storageViewLValue(expression, context);
  if (storageView?.fieldIndex !== undefined) {
    return `${context.nameFor(storageView.name)}[${storageView.index} + ${storageView.fieldIndex}u]`;
  }
  const objectName = expressionName(expression.object);
  const axisIndex = expression.property === "x" ? 0 : expression.property === "y" ? 1 : 2;
  switch (objectName) {
    case "threadIdx":
      return `i32(local_id.${expression.property})`;
    case "blockIdx":
      return `i32(workgroup_id.${expression.property})`;
    case "blockDim":
      return String(context.ir.workgroupSize[axisIndex]);
    case "gridDim":
      return `i32(num_workgroups.${expression.property})`;
    default:
      return `${emitExpression(expression.object, context)}.${expression.property}`;
  }
}

function emitMatrixTileMemberExpression(
  expression: Extract<CudaLiteExpression, { kind: "member" }>,
  context: EmitContext,
): string | undefined {
  const ref = matrixTileReference(expression.object);
  if (!ref) return undefined;
  const declaration = context.localArrayFor(ref.root);
  const spec = declaration?.matrixTile ? resolveMatrixTileSpec(declaration.matrixTile) : undefined;
  if (!declaration || !spec) return undefined;
  if (expression.property === "num_elements") return String(matrixTileElementCount(spec));
  if (expression.property === "x") throw featureError("unsupported-wmma-fragment-member", "WMMA fragment lane storage requires indexed access");
  throw featureError("unsupported-wmma-fragment-member", `unsupported WMMA fragment member '${expression.property}'`);
}

function emitMatrixTileLaneAccessExpression(
  expression: Extract<CudaLiteExpression, { kind: "index" }>,
  context: EmitContext,
): string | undefined {
  if (expression.target.kind !== "member" || expression.target.property !== "x") return undefined;
  const ref = matrixTileReference(expression.target.object);
  if (!ref) return undefined;
  const declaration = context.localArrayFor(ref.root);
  const spec = declaration?.matrixTile ? resolveMatrixTileSpec(declaration.matrixTile) : undefined;
  if (!declaration || !spec) return undefined;
  const base = emitMatrixTileBase(ref.indices, declaration.dimensions, matrixTileElementCount(spec), context);
  const lane = `u32(${emitExpression(expression.index, context)})`;
  return base === "0u"
    ? `${context.nameFor(ref.root)}[${lane}]`
    : `${context.nameFor(ref.root)}[(${base} + ${lane})]`;
}

function expressionValueTypeForEmit(expression: CudaLiteExpression, context: EmitContext): CudaLiteScalarType | undefined {
  if (context.expressionValueTypes.has(expression)) return context.expressionValueTypes.get(expression);
  const valueType = uncachedExpressionValueTypeForEmit(expression, context);
  context.expressionValueTypes.set(expression, valueType);
  return valueType;
}

function uncachedExpressionValueTypeForEmit(expression: CudaLiteExpression, context: EmitContext): CudaLiteScalarType | undefined {
  if (expression.kind === "number") {
    if (/[.eEfF]/u.test(expression.raw)) return "float";
    if (/[uU]$/u.test(expression.raw)) return "uint";
    return "int";
  }
  if (expression.kind === "identifier") {
    return context.localValueTypeFor(expression.name) ??
      context.paramFor(expression.name)?.valueType ??
      context.deviceGlobalFor(expression.name)?.valueType ??
      context.ir.constants.find((item) => item.name === expression.name)?.valueType ??
      context.uniformScalarTypeFor(expression.name);
  }
  if (expression.kind === "cast") return expression.valueType;
  if (expression.kind === "call") {
    const name = expressionName(expression.callee);
    if (name !== undefined && isTextureReadCall(name)) return expression.templateValueType ?? "float";
    if (name !== undefined) {
      const fn = context.deviceFunctionFor(name, expression.args.length);
      if (fn) return fn.returnType;
    }
    return name ? cudaVectorConstructorType(name) : undefined;
  }
  if (expression.kind === "index") {
    const storageView = storageViewLValue(expression, context);
    if (storageView) return storageView.valueType;
    return expressionValueTypeForEmit(expression.target, context);
  }
  if (expression.kind === "member") {
    if (expression.property === "size") return "int";
    const objectType = expressionValueTypeForEmit(expression.object, context);
    if (isCudaVectorType(objectType)) return cudaVectorScalarType(objectType);
    if (objectType === "complex64") return "float";
    return objectType;
  }
  if (expression.kind === "binary") {
    const vectorType = vectorArithmeticTypeForEmit(expression, context);
    if (vectorType) return vectorType;
    if (isComparisonOperator(expression.operator) || expression.operator === "&&" || expression.operator === "||") return "bool";
    return promotedCudaScalarType(
      expressionValueTypeForEmit(expression.left, context),
      expressionValueTypeForEmit(expression.right, context),
    );
  }
  return undefined;
}

function vectorArithmeticTypeForEmit(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): CudaLiteVectorType | undefined {
  if (!isVectorArithmeticOperator(expression.operator)) return undefined;
  const left = expressionValueTypeForEmit(expression.left, context);
  const right = expressionValueTypeForEmit(expression.right, context);
  if (isCudaVectorType(left) && isCudaVectorType(right)) return left === right ? left : undefined;
  if (isCudaVectorType(left)) return left;
  return isCudaVectorType(right) ? right : undefined;
}

function emitVectorArithmetic(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): string | undefined {
  const vectorType = vectorArithmeticTypeForEmit(expression, context);
  if (!vectorType) return undefined;
  const left = emitExpressionAsVectorOperand(expression.left, vectorType, context);
  const right = emitExpressionAsVectorOperand(expression.right, vectorType, context);
  return `(${left} ${expression.operator} ${right})`;
}

function emitScalarBinaryExpression(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): string {
  const target = promotedWgslScalarTypeForBinary(expression, context);
  const left = target ? emitExpressionAsWgslScalar(expression.left, target, context) : emitExpression(expression.left, context);
  const right = target ? emitExpressionAsWgslScalar(expression.right, target, context) : emitExpression(expression.right, context);
  return `(${left} ${expression.operator} ${right})`;
}

function promotedWgslScalarTypeForBinary(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): "f32" | "f16" | "i32" | "u32" | undefined {
  if (!isScalarPromotionOperator(expression.operator)) return undefined;
  const left = expressionWgslScalarType(expression.left, context);
  const right = expressionWgslScalarType(expression.right, context);
  if (left === undefined || right === undefined || left === right) return undefined;
  if (left === "f32" || right === "f32") return "f32";
  if (left === "f16" || right === "f16") return "f16";
  return undefined;
}

function emitExpressionAsWgslScalar(
  expression: CudaLiteExpression,
  target: "f32" | "f16" | "i32" | "u32",
  context: EmitContext,
): string {
  const value = emitExpression(expression, context);
  return expressionWgslScalarType(expression, context) === target ? value : `${target}(${value})`;
}

function promotedCudaScalarType(
  left: CudaLiteScalarType | undefined,
  right: CudaLiteScalarType | undefined,
): CudaLiteScalarType | undefined {
  if (left === undefined || right === undefined) return left ?? right;
  if (isCudaVectorType(left) || isCudaVectorType(right)) return undefined;
  const leftWgsl = cudaScalarWgslType(left);
  const rightWgsl = cudaScalarWgslType(right);
  if (leftWgsl === "f32" || rightWgsl === "f32") return "float";
  if (leftWgsl === "f16" || rightWgsl === "f16") return "half";
  if (leftWgsl === "u32" || rightWgsl === "u32") return "uint";
  if (leftWgsl === "i32" || rightWgsl === "i32") return "int";
  if (leftWgsl === "bool" && rightWgsl === "bool") return "bool";
  return undefined;
}

function expressionWgslScalarType(
  expression: CudaLiteExpression,
  context: EmitContext,
): "f32" | "f16" | "i32" | "u32" | "bool" | undefined {
  const valueType = expressionValueTypeForEmit(expression, context);
  if (valueType === undefined || isCudaVectorType(valueType)) return undefined;
  return cudaScalarWgslType(valueType);
}

function cudaScalarWgslType(type: CudaLiteScalarType): "f32" | "f16" | "i32" | "u32" | "bool" | undefined {
  if (type === "float" || type === "double" || type === "bf16") return "f32";
  if (type === "half") return "f16";
  if (type === "int") return "i32";
  if (type === "uint") return "u32";
  if (type === "bool") return "bool";
  return undefined;
}

function isScalarPromotionOperator(operator: string): boolean {
  return isVectorArithmeticOperator(operator) || isComparisonOperator(operator);
}

function isComparisonOperator(operator: string): boolean {
  return operator === "<" || operator === "<=" || operator === ">" || operator === ">=" || operator === "==" || operator === "!=";
}

function emitExpressionAsVectorOperand(
  expression: CudaLiteExpression,
  vectorType: CudaLiteVectorType,
  context: EmitContext,
): string {
  const value = emitExpression(expression, context);
  return expressionValueTypeForEmit(expression, context) === vectorType
    ? value
    : emitVectorSplat(vectorType, castExpressionToVectorScalar(value, vectorType));
}

function emitVectorMinMaxCall(
  expression: CudaLiteCallExpression,
  name: string | undefined,
  context: EmitContext,
): string | undefined {
  if (name !== "min" && name !== "max" && name !== "fminf" && name !== "fmaxf") return undefined;
  const vectorType = expression.args
    .map((arg) => expressionValueTypeForEmit(arg, context))
    .find(isCudaVectorType);
  if (!vectorType) return undefined;
  const op = name === "min" || name === "fminf" ? "min" : "max";
  return `${op}(${expression.args.map((arg) => emitExpressionAsVectorOperand(arg, vectorType, context)).join(", ")})`;
}

function emitFrexpCall(expression: CudaLiteCallExpression, context: EmitContext): string {
  const value = expression.args[0] ? emitExpressionAsValueType(expression.args[0], "float", context) : "0.0";
  const exponent = expression.args[1] ? emitExpression(expression.args[1], context) : "&__bg_missing_frexp_exp";
  return `bg_frexp(${value}, ${exponent})`;
}

function castExpressionToVectorScalar(value: string, vectorType: CudaLiteScalarType): string {
  return `${wgslScalar(cudaVectorScalarType(vectorType) ?? "float")}(${value})`;
}

function emitVectorSplat(vectorType: CudaLiteScalarType, value: string): string {
  return `${wgslScalar(vectorType)}(${Array.from({ length: cudaVectorLaneCount(vectorType) }, () => value).join(", ")})`;
}

function isVectorArithmeticOperator(operator: string): boolean {
  return operator === "+" || operator === "-" || operator === "*" || operator === "/";
}

function emitCall(expression: CudaLiteCallExpression, context: EmitContext): string {
  const name = expressionName(expression.callee);
  const cooperativeGroupCall = emitCooperativeGroupCall(expression, context);
  if (cooperativeGroupCall !== undefined) return cooperativeGroupCall;
  if (wmmaBuiltinName(name)) return "0";
  const deviceFunction = name ? context.deviceFunctionFor(name, expression.args.length) : undefined;
  if (deviceFunction) {
    const args = deviceFunction.params.flatMap((param, index) => {
      const arg = expression.args[index];
      if (param.cooperativeGroupKind !== undefined) {
        return param.cooperativeGroupKind === "thread"
          ? [emitCooperativeGroupTileSizeArgument(arg, context)]
          : [];
      }
      if (param.valueType === "texture2d") return [emitTextureArgument(arg, context)];
      if (!arg) return param.pointer ? ["0u", "0u"] : [zeroValue(param.valueType)];
      return param.pointer
        ? usesFunctionLocalPointerParam(deviceFunction, param, context.ir)
          ? [emitExpression(arg, context)]
          : emitDevicePointerArgument(arg, context)
        : [emitExpressionAsValueType(arg, param.valueType, context)];
    });
    return `${context.nameFor(deviceFunctionLinkName(deviceFunction, context.ir))}(${[...args, "local_id", "workgroup_id", "num_workgroups"].join(", ")})`;
  }
  const args = expression.args.map((arg) => emitExpression(arg, context));
  const vectorMinMax = emitVectorMinMaxCall(expression, name, context);
  if (vectorMinMax !== undefined) return vectorMinMax;
  if (name === "frexp" || name === "frexpf") return emitFrexpCall(expression, context);
  const intrinsic = name ? CUDA_INTRINSICS_BY_NAME.get(name) : undefined;
  if (intrinsic?.emitWgsl) return intrinsic.emitWgsl(args);
  const vectorConstructor = name ? cudaVectorConstructorType(name) : undefined;
  if (vectorConstructor) {
    return expression.args.length === 1
      ? emitVectorConversionConstructor(vectorConstructor, expression.args[0]!, context)
      : emitMixedVectorConstructor(vectorConstructor, expression.args, context);
  }
  if (name === "__halves2bfloat162") return `${wgslScalar("bf162")}(${args.join(", ")})`;
  if (isPointerIdentityCall(name)) return expression.args[0] ? emitExpression(expression.args[0], context) : "0u";
  if (name !== undefined && CUDA_CACHE_HINT_LOADS.has(name)) {
    const target = expression.args[0];
    if (!target) return "0";
    const parts = devicePointerArgumentParts(target, context);
    if (!parts) throw featureError("unsupported-device-pointer-param", `${name} expects a storage pointer or derived storage address`);
    const valueType = devicePointerValueTypeForExpression(target, context);
    return `${pointerReadHelperName(valueType)}(${parts.buffer}, ${parts.base})`;
  }
  if (name !== undefined && CUDA_CACHE_HINT_STORES.has(name)) {
    const target = expression.args[0];
    const value = expression.args[1];
    if (!target || !value) return "0";
    const parts = devicePointerArgumentParts(target, context);
    if (!parts) throw featureError("unsupported-device-pointer-param", `${name} expects a storage pointer or derived storage address`);
    const valueType = devicePointerValueTypeForExpression(target, context);
    return `${pointerWriteHelperName(valueType)}(${parts.buffer}, ${parts.base}, ${emitExpression(value, context)})`;
  }
  if (name === "__cvta_generic_to_shared") {
    const target = expression.args[0];
    if (!target) return "0u";
    const sharedIndex = emitSharedAddressIndex(target, context);
    if (sharedIndex) return `u32(${sharedIndex})`;
    const parts = devicePointerArgumentParts(target, context);
    if (!parts) throw featureError("unsupported-device-pointer-param", "__cvta_generic_to_shared expects a storage or shared pointer");
    return `u32(${parts.base})`;
  }
  if (isCpAsyncCopyCall(name) || isCpAsyncFenceCall(name)) return "0";
  switch (name) {
    case "__syncthreads":
    case "__syncwarp":
      return "workgroupBarrier()";
    case "__threadfence":
      return "storageBarrier()";
    case "__trap":
      return "0";
    case "clock":
      return "(workgroup_id.x * 104729u + workgroup_id.y * 1009u + workgroup_id.z * 97u + local_id.x + local_id.y * 31u + local_id.z * 7u)";
    case "clock64":
      return "(workgroup_id.x * 104729u + workgroup_id.y * 1009u + workgroup_id.z * 97u + local_id.x + local_id.y * 31u + local_id.z * 7u)";
    case "cudaDeviceSynchronize":
    case "cudaStreamCreate":
    case "cudaStreamCreateWithFlags":
    case "cudaStreamDestroy":
    case "cudaStreamSynchronize":
    case "cudaEventCreate":
    case "cudaEventCreateWithFlags":
    case "cudaEventDestroy":
    case "cudaEventRecord":
    case "cudaEventSynchronize":
      return "0";
    case "cudaMemcpy":
      return "0";
    case "cudaMemcpyAsync":
      return "0";
    case "cudaMemcpyPeerAsync":
      return "0";
    case "min":
    case "max":
      return `${name}(${args.join(", ")})`;
    case "div_ceil":
      return `(((${args[0] ?? "0"} + ${args[1] ?? "1"}) - 1) / ${args[1] ?? "1"})`;
    case "bg_subgroup_add":
      return `subgroupAdd(${args.join(", ")})`;
    case "warpReduceSum":
    case "warp_reduce_sum":
    case "warp_reduce_sum_f32":
    case "warp_reduce_sum_f16":
    case "warp_reduce_sum_f16_f16":
    case "warp_reduce_sum_f16_f32":
    case "warp_reduce_sum_i8_i32":
    case "warp_reduce_sum_i32_i32":
      return `subgroupAdd(${args.length === 2 ? args[1] ?? "0" : args[0] ?? "0"})`;
    case "blockReduce":
      return `subgroupAdd(${args[0] ?? "0"})`;
    case "__reduce_add_sync":
      return `subgroupAdd(${args[1] ?? "0"})`;
    case "warpReduceMax":
    case "warp_reduce_max":
    case "warp_reduce_max_f32":
      return `subgroupMax(${args.length === 2 ? args[1] ?? "0" : args[0] ?? "0"})`;
    case "warpReduceMin":
    case "warp_reduce_min":
      return `subgroupMin(${args.length === 2 ? args[1] ?? "0" : args[0] ?? "0"})`;
    case "__shfl_sync":
      return `subgroupShuffle(${args[1] ?? "0"}, u32(${args[2] ?? "0"}))`;
    case "__shfl_down_sync":
      return `subgroupShuffleDown(${args[1] ?? "0"}, u32(${args[2] ?? "0"}))`;
    case "__shfl_up_sync":
      return `subgroupShuffleUp(${args[1] ?? "0"}, u32(${args[2] ?? "0"}))`;
    case "__shfl_xor_sync":
      return `subgroupShuffleXor(${args[1] ?? "0"}, u32(${args[2] ?? "0"}))`;
    case "__any_sync":
      return `select(0u, 1u, subgroupAny((${args[1] ?? "0"}) != 0))`;
    case "__all_sync":
      return `select(0u, 1u, subgroupAll((${args[1] ?? "0"}) != 0))`;
    case "__ballot_sync":
      return `subgroupBallot((${args[1] ?? "0"}) != 0).x`;
    case "tex1D":
    case "tex1Dfetch":
    case "tex2D":
    case "tex2DLod":
    case "tex2DLayered":
    case "tex3D":
    case "texCubemap":
      if (expression.args.length >= 2 && expression.args[0]?.kind === "identifier") {
        const textureArgs = textureReadArgsForEmit(expression, context);
        if (isTextureBindingName(expression.args[0].name, context)) {
          const suffix = textureReadHelperSuffix(expression.templateValueType);
          return `bg_tex2d_${suffix}_${expression.args[0].name}(${textureArgs.join(", ")})`;
        }
        return emitTextureReadExpression(
          emitTextureArgument(expression.args[0], context),
          textureArgs,
          expression.templateValueType,
        );
      }
      return `${name}(${args.join(", ")})`;
    case "surf2Dread":
      if (expression.args.length === 3) {
        const surfaceName = rootIdentifier(expression.args[0]!);
        if (!surfaceName) return `surf2Dread(${args.join(", ")})`;
        const targetType = expression.templateValueType ?? "float";
        const read = `bg_surf2dread_${surfaceName}(${emitExpression(expression.args[1]!, context)}, ${emitExpression(expression.args[2]!, context)})`;
        return targetType === "float" ? read : `${wgslScalar(targetType)}(${read})`;
      }
      if (expression.args.length >= 4) {
        const surfaceName = rootIdentifier(expression.args[1]!);
        if (!surfaceName) return `surf2Dread(${args.join(", ")})`;
        const target = expression.args[0]!;
        const lvalue = target.kind === "unary" && target.operator === "&" ? target.argument : target;
        const targetType = expressionValueTypeForEmit(lvalue, context) ?? "float";
        const read = `bg_surf2dread_${surfaceName}(${emitExpression(expression.args[2]!, context)}, ${emitExpression(expression.args[3]!, context)})`;
        const value = targetType === "float" ? read : `${wgslScalar(targetType)}(${read})`;
        return `${emitExpression(lvalue, context, "lvalue")} = ${value}`;
      }
      return `surf2Dread(${args.join(", ")})`;
    case "surf2Dwrite":
    case "surf1Dwrite":
    case "surf2DLayeredwrite":
    case "surf3Dwrite":
      if (expression.args.length >= 3) {
        const surfaceName = rootIdentifier(expression.args[1]!);
        if (!surfaceName) return `${name}(${args.join(", ")})`;
        const y = name === "surf1Dwrite"
          ? "0"
          : name === "surf2DLayeredwrite"
            ? `(${emitExpression(expression.args[3]!, context)} + ${emitExpression(expression.args[4]!, context)})`
            : emitExpression(expression.args[3]!, context);
        const z = name === "surf3Dwrite" ? emitExpression(expression.args[4]!, context) : "0";
        return emitSurfaceWriteExpression(surfaceName, expression.args[0]!, expression.args[2]!, y, z, context);
      }
      return `${name}(${args.join(", ")})`;
    case "deviceAllocate":
    case "streamOrderedAllocate":
      if (expression.args.length === 4 && expression.args[0]?.kind === "identifier" && expression.args[1]?.kind === "identifier") {
        return `${rawPoolHelperName(expression.args[0].name, expression.args[1].name)}(u32(${emitExpression(expression.args[2]!, context)}), u32(${emitExpression(expression.args[3]!, context)}))`;
      }
      if (expression.args.length >= 2 && expression.args[0]?.kind === "identifier") {
        return `bg_pool_alloc_${expression.args[0].name}(u32(${emitExpression(expression.args[1]!, context)}))`;
      }
      if (expression.args.length >= 2 && expression.args[0]?.kind === "unary" && expression.args[0].operator === "&" && expression.args[0].argument.kind === "identifier") {
        return `bg_pool_alloc_${expression.args[0].argument.name}(u32(${emitExpression(expression.args[1]!, context)}))`;
      }
      return "0u";
    case "sizeof":
      if (expression.args[0]?.kind === "identifier") return String(sizeofCudaType(expression.args[0].name) ?? 4);
      return "4";
    case "alignof":
      if (expression.args[0]?.kind === "identifier") return String(alignofCudaType(expression.args[0].name) ?? 4);
      return "4";
    case "vec_at":
      return `(${args[0] ?? "vec4<f32>()"}[u32(${args[1] ?? "0"})])`;
    case "dot":
      return `dot(${args[0] ?? "vec2<f32>()"}, ${args[1] ?? "vec2<f32>()"})`;
    case "length":
      return `length(${args[0] ?? "vec2<f32>()"})`;
    case "normalize":
      return `normalize(${args[0] ?? "vec2<f32>()"})`;
    case "cross":
      return `cross(${args[0] ?? "vec3<f32>()"}, ${args[1] ?? "vec3<f32>()"})`;
    case "curand_init":
      return `bg_curand_init(u32(${args[0] ?? "0"}), u32(${args[1] ?? "0"}), u32(${args[2] ?? "0"}), ${args[3] ?? "&state"})`;
    case "curand_uniform":
    case "curand_uniform_double":
      return `bg_curand_uniform(${args[0] ?? "&state"})`;
    case "curand_normal":
    case "curand_normal_double":
      return `bg_curand_normal(${args[0] ?? "&state"})`;
    case "atomicAdd":
    case "atomicAdd_system":
      return emitAtomicCall("atomicAdd", expression, context, args);
    case "atomicSub":
      return emitAtomicCall("atomicSub", expression, context, args);
    case "atomicMin":
    case "atomicMin_system":
      return emitAtomicCall("atomicMin", expression, context, args);
    case "atomicMax":
    case "atomicMax_system":
      return emitAtomicCall("atomicMax", expression, context, args);
    case "atomicMaxFloat":
      return emitAtomicCall("atomicMax", expression, context, args);
    case "atomicAnd":
    case "atomicAnd_system":
      return emitAtomicCall("atomicAnd", expression, context, args);
    case "atomicOr":
    case "atomicOr_system":
      return emitAtomicCall("atomicOr", expression, context, args);
    case "atomicXor":
    case "atomicXor_system":
      return emitAtomicCall("atomicXor", expression, context, args);
    case "atomicInc":
    case "atomicInc_system":
      return emitAtomicCall("atomicInc", expression, context, args);
    case "atomicDec":
    case "atomicDec_system":
      return emitAtomicCall("atomicDec", expression, context, args);
    case "atomicExch":
    case "atomicExch_system":
      return emitAtomicCall("atomicExchange", expression, context, args);
    case "atomicCAS":
    case "atomicCAS_system": {
      const target = expression.args[0];
      const compare = expression.args[1];
      const value = expression.args[2];
      const atomicTarget = emitAtomicTarget(target, context);
      if (atomicTarget && compare && value) {
        return `atomicCompareExchangeWeak(${atomicTarget.address}, ${emitExpression(compare, context)}, ${emitExpression(value, context)}).old_value`;
      }
      const pointerAtomic = emitDevicePointerAtomicCasCall(target, compare, value, context);
      if (pointerAtomic) return pointerAtomic;
      return `atomicCompareExchangeWeak(${args.join(", ")}).old_value`;
    }
    default:
      return `${emitExpression(expression.callee, context)}(${args.join(", ")})`;
  }
}

function emitCooperativeGroupTileSizeArgument(
  arg: CudaLiteExpression | undefined,
  context: EmitContext,
): string {
  const blockSize = `${context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2]}u`;
  if (arg?.kind !== "identifier") return blockSize;
  const group = context.cooperativeGroupFor(arg.name);
  if (!group) return blockSize;
  if (group.groupKind === "tile") return `${group.tileSize ?? 32}u`;
  if (group.groupKind === "thread" && group.dynamicTileSizeName) return group.dynamicTileSizeName;
  return blockSize;
}

function emitExpressionAsValueType(
  expression: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: EmitContext,
): string {
  if (expression.kind === "initializer" && isCudaVectorType(valueType)) {
    return emitVectorConstructor(
      valueType,
      expression.elements.map((element) => emitExpressionAsValueType(element, cudaVectorScalarType(valueType) ?? "float", context)),
    );
  }
  const value = emitExpression(expression, context);
  if (valueType === "void") return value;
  if (expressionValueTypeForEmit(expression, context) === valueType) return value;
  if (isCudaVectorType(valueType)) return expressionValueTypeForEmit(expression, context) !== undefined
    ? emitVectorConversionConstructor(valueType, expression, context)
    : emitVectorSplat(valueType, castExpressionToVectorScalar(value, valueType));
  if (valueType === "bool") return `(${value} != 0)`;
  return `${wgslScalar(valueType)}(${value})`;
}

function emitCooperativeGroupCall(expression: CudaLiteCallExpression, context: EmitContext): string | undefined {
  const namespaceCall = emitCooperativeNamespaceCall(expression, context);
  if (namespaceCall !== undefined) return namespaceCall;
  const callee = expression.callee;
  if (callee.kind !== "member" || callee.object.kind !== "identifier") return undefined;
  const group = context.cooperativeGroupFor(callee.object.name);
  if (!group) return undefined;
  if (callee.property === "sync") {
    return group.groupKind === "grid" ? "0" : "workgroupBarrier()";
  }
  if (callee.property === "size") {
    if (group.groupKind === "thread" && group.dynamicTileSizeName) return `i32(${group.dynamicTileSizeName})`;
    if (group.groupKind === "tile") return String(group.tileSize ?? 32);
    return String(context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2]);
  }
  if (callee.property === "thread_rank") {
    const localRank = emitLocalLinearRank(context);
    if (group.groupKind === "thread" && group.dynamicTileSizeName) return `(${localRank} % i32(${group.dynamicTileSizeName}))`;
    if (group.groupKind === "tile") return `(${localRank} % ${group.tileSize ?? 32})`;
    return localRank;
  }
  if (callee.property === "meta_group_size") {
    if (group.groupKind === "thread" && group.dynamicTileSizeName) {
      const blockSize = context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2];
      return `i32((${blockSize}u + ${group.dynamicTileSizeName} - 1u) / ${group.dynamicTileSizeName})`;
    }
    if (group.groupKind !== "tile") return "1";
    const blockSize = context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2];
    return String(Math.ceil(blockSize / (group.tileSize ?? 32)));
  }
  if (callee.property === "meta_group_rank") {
    if (group.groupKind === "thread" && group.dynamicTileSizeName) return `(${emitLocalLinearRank(context)} / i32(${group.dynamicTileSizeName}))`;
    if (group.groupKind !== "tile") return "0";
    return `(${emitLocalLinearRank(context)} / ${group.tileSize ?? 32})`;
  }
  if (callee.property === "shfl" || callee.property === "shfl_down" || callee.property === "shfl_up" || callee.property === "shfl_xor") {
    const value = expression.args[0] ? emitExpression(expression.args[0], context) : "0";
    const offset = expression.args[1] ? emitExpression(expression.args[1], context) : "0";
    const intrinsic = callee.property === "shfl"
      ? "subgroupShuffle"
      : callee.property === "shfl_up"
      ? "subgroupShuffleUp"
      : callee.property === "shfl_xor"
        ? "subgroupShuffleXor"
        : "subgroupShuffleDown";
    return `${intrinsic}(${value}, u32(${offset}))`;
  }
  if (callee.property === "ballot") {
    const predicate = expression.args[0] ? emitExpression(expression.args[0], context) : "0";
    return `subgroupBallot((${predicate}) != 0).x`;
  }
  return undefined;
}

function emitCooperativeNamespaceCall(expression: CudaLiteCallExpression, context: EmitContext): string | undefined {
  const name = expressionName(expression.callee);
  if (!name?.endsWith("::sync") && !name?.endsWith("::reduce")) return undefined;
  const groupArg = expression.args[0];
  if (groupArg?.kind !== "identifier") return undefined;
  const group = context.cooperativeGroupFor(groupArg.name);
  if (!group) return undefined;
  if (name.endsWith("::sync")) return group.groupKind === "grid" ? "0" : "workgroupBarrier()";
  const vectorReduce = emitVectorCooperativeNamespaceReduce(expression, group, context);
  if (vectorReduce !== undefined) return vectorReduce;
  const value = expression.args[1] ? emitExpression(expression.args[1], context) : "0";
  const op = cooperativeReductionOpName(expression.args[2]);
  if (op?.endsWith("::greater")) return `subgroupMax(${value})`;
  return `subgroupAdd(${value})`;
}

function emitVectorCooperativeNamespaceReduce(
  expression: CudaLiteCallExpression,
  group: CudaLiteCooperativeGroupDecl,
  context: EmitContext,
): string | undefined {
  const valueExpression = expression.args[1];
  const opExpression = expression.args[2];
  if (!valueExpression) return undefined;
  const valueType = expressionValueTypeForEmit(valueExpression, context);
  if (!isCudaVectorType(valueType)) return undefined;
  const opName = opExpression ? expressionName(opExpression) : undefined;
  const op = opName ? context.deviceFunctionFor(opName) : undefined;
  if (!op || op.returnType !== valueType) {
    throw featureError("unsupported-cooperative-groups", "vector cg::reduce expects a device helper returning the same CUDA vector type");
  }
  const tileSize = group.groupKind === "tile" ? group.tileSize ?? 32 : 32;
  const helper = registerVectorCooperativeReduceHelper(context, op.name, valueType, tileSize);
  return `${helper.name}(${emitExpression(valueExpression, context)}, local_id, workgroup_id, num_workgroups)`;
}

function registerVectorCooperativeReduceHelper(
  context: EmitContext,
  opName: string,
  valueType: CudaLiteVectorType,
  tileSize: number,
): VectorCooperativeReduceHelper {
  const key = `${opName}:${valueType}:${tileSize}`;
  const existing = context.vectorCooperativeReduceHelpers.get(key);
  if (existing) return existing;
  const helper = {
    key,
    name: `bg_cg_reduce_${sanitizeWgslIdentifier(opName)}_${valueType}_${tileSize}`,
    opName,
    valueType,
    tileSize,
  };
  context.vectorCooperativeReduceHelpers.set(key, helper);
  return helper;
}

function emitVectorCooperativeReduceHelper(
  helper: VectorCooperativeReduceHelper,
  context: EmitContext,
): string[] {
  const type = wgslScalar(helper.valueType);
  const fields = ["x", "y", "z", "w"].slice(0, cudaVectorLaneCount(helper.valueType));
  const start = Math.max(1, Math.floor(helper.tileSize / 2));
  const shuffled = `${type}(${fields.map((field) => `subgroupShuffleXor(value.${field}, offset)`).join(", ")})`;
  return [
    `fn ${helper.name}(value_arg: ${type}, local_id: vec3<u32>, workgroup_id: vec3<u32>, num_workgroups: vec3<u32>) -> ${type} {`,
    `  var value: ${type} = value_arg;`,
    `  var offset: u32 = ${start}u;`,
    "  while (offset > 0u) {",
    `    value = ${context.nameFor(helper.opName)}(value, ${shuffled}, local_id, workgroup_id, num_workgroups);`,
    "    offset = offset / 2u;",
    "  }",
    "  return value;",
    "}",
  ];
}

function sanitizeWgslIdentifier(value: string): string {
  return value.replace(/[^A-Za-z0-9_]/gu, "_");
}

function cooperativeReductionOpName(expression: CudaLiteExpression | undefined): string | undefined {
  if (expression?.kind === "call") return expressionName(expression.callee);
  return expression === undefined ? undefined : expressionName(expression);
}

function cooperativeGroupForParam(param: CudaLiteParam, context: EmitContext): CudaLiteCooperativeGroupDecl {
  return {
    kind: "cooperative-group",
    groupKind: param.cooperativeGroupKind ?? "block",
    name: param.name,
    ...(param.tileSize === undefined ? {} : { tileSize: param.tileSize }),
    ...(param.cooperativeGroupKind === "thread" ? { dynamicTileSizeName: context.nameFor(`${param.name}_tile_size`) } : {}),
    span: param.span,
  };
}

function emitLocalLinearRank(context: EmitContext): string {
  const [, y, z] = context.ir.workgroupSize;
  if (y === 1 && z === 1) return "i32(local_id.x)";
  if (z === 1) return `i32(local_id.x + local_id.y * ${context.ir.workgroupSize[0]}u)`;
  return `i32(local_id.x + local_id.y * ${context.ir.workgroupSize[0]}u + local_id.z * ${context.ir.workgroupSize[0] * context.ir.workgroupSize[1]}u)`;
}

function emitInlineU32Output(target: CudaLiteExpression, expression: string, context: EmitContext): string {
  const valueType = expressionValueTypeForEmit(target, context);
  if (valueType === "int") return `i32(${expression})`;
  return expression;
}

function emitAtomicCall(
  wgslName: string,
  expression: CudaLiteCallExpression,
  context: EmitContext,
  args: readonly string[],
): string {
  const target = expression.args[0];
  const value = expression.args[1];
  const atomicTarget = emitAtomicTarget(target, context);
  if (atomicTarget && value) {
    const valueExpression = emitExpression(value, context);
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
    return `${wgslName}(${atomicTarget.address}, ${valueExpression})`;
  }
  const pointerAtomic = emitDevicePointerAtomicCall(wgslName, target, value, context);
  if (pointerAtomic) return pointerAtomic;
  return `${wgslName}(${args.join(", ")})`;
}

function emitDevicePointerAtomicCall(
  wgslName: string,
  target: CudaLiteExpression | undefined,
  value: CudaLiteExpression | undefined,
  context: EmitContext,
): string | undefined {
  if (!target || !value) return undefined;
  const parts = devicePointerArgumentParts(target, context);
  if (!parts) return undefined;
  const valueType = devicePointerValueTypeForExpression(target, context);
  if (wgslName === "atomicAdd" && isDevicePointerAtomicAddType(valueType)) {
    return `${pointerAtomicAddHelperName(valueType)}(${parts.buffer}, ${parts.base}, ${emitExpression(value, context)})`;
  }
  if (wgslName === "atomicExchange" && isDevicePointerAtomicExchangeType(valueType)) {
    return `${pointerAtomicExchangeHelperName(valueType)}(${parts.buffer}, ${parts.base}, ${emitExpression(value, context)})`;
  }
  return undefined;
}

function emitDevicePointerAtomicCasCall(
  target: CudaLiteExpression | undefined,
  compare: CudaLiteExpression | undefined,
  value: CudaLiteExpression | undefined,
  context: EmitContext,
): string | undefined {
  if (!target || !compare || !value) return undefined;
  const parts = devicePointerArgumentParts(target, context);
  if (!parts) return undefined;
  const valueType = devicePointerValueTypeForExpression(target, context);
  if (!isDevicePointerAtomicCasType(valueType)) return undefined;
  return `${pointerAtomicCasHelperName(valueType)}(${parts.buffer}, ${parts.base}, ${emitExpression(compare, context)}, ${emitExpression(value, context)})`;
}

function emitAtomicTarget(
  target: CudaLiteExpression | undefined,
  context: EmitContext,
): AtomicTargetInfo | undefined {
  if (!target) return undefined;
  if (target.kind === "cast" && target.pointer) {
    const inner = emitAtomicTarget(target.expression, context);
    return inner ? { ...inner, valueType: target.valueType } : undefined;
  }
  if (target.kind === "unary" && target.operator === "&") return emitAtomicAddressTarget(target.argument, context);
  if (target.kind === "identifier") {
    const alias = flattenedPointerAlias(target.name, target.span, context);
    if (alias) {
      const rootName = resolveAtomicRootName(alias.rootName, context);
      const info = atomicStorageInfo(rootName, context);
      if (!info) return undefined;
      const index = emitPointerAliasIndex(alias, zeroExpression(target.span), context);
      return {
        address: `&${rootName}[${index}]`,
        rootName,
        valueType: alias.valueType ?? info.valueType,
        storageScalar: info.storageScalar,
        addressSpace: info.addressSpace,
      };
    }
    const param = context.paramFor(target.name);
    if (param?.pointer) {
      const info = atomicStorageInfo(target.name, context);
      if (!info) return undefined;
      const index = emitPointerIndex(target.name, zeroExpression(target.span), context);
      return {
        address: `&${target.name}[${index}]`,
        rootName: target.name,
        valueType: param.valueType,
        storageScalar: info.storageScalar,
        addressSpace: info.addressSpace,
      };
    }
  }
  const pointerParts = devicePointerArgumentParts(target, context);
  const root = rootIdentifier(target);
  const rootName = root ? resolveAtomicRootName(root, context) : undefined;
  const info = rootName ? atomicStorageInfo(rootName, context) : undefined;
  if (pointerParts && rootName && info) {
    return {
      address: `&${rootName}[${pointerParts.base}]`,
      rootName,
      valueType: atomicExpressionValueType(target, rootName, context) ?? info.valueType,
      storageScalar: info.storageScalar,
      addressSpace: info.addressSpace,
    };
  }
  return undefined;
}

function emitAtomicAddressTarget(expression: CudaLiteExpression, context: EmitContext): AtomicTargetInfo | undefined {
  const rootName = atomicExpressionRootName(expression, context);
  if (!rootName) return undefined;
  const info = atomicStorageInfo(rootName, context);
  if (!info) return undefined;
  return {
    address: `&${emitExpression(expression, context, "lvalue")}`,
    rootName,
    valueType: atomicExpressionValueType(expression, rootName, context) ?? info.valueType,
    storageScalar: info.storageScalar,
    addressSpace: info.addressSpace,
  };
}

function atomicExpressionRootName(expression: CudaLiteExpression, context: EmitContext): string | undefined {
  const root = rootIdentifier(expression);
  return root ? resolveAtomicRootName(root, context) : undefined;
}

function atomicExpressionValueType(
  expression: CudaLiteExpression,
  rootName: string,
  context: EmitContext,
): CudaLiteScalarType | undefined {
  const root = rootIdentifier(expression);
  if (root) {
    const alias = context.pointerAliasFor(root);
    if (alias?.valueType) return alias.valueType;
    const direct = context.paramFor(root);
    if (direct?.pointer) return direct.valueType;
  }
  return atomicStorageInfo(rootName, context)?.valueType;
}

function resolveAtomicRootName(name: string, context: EmitContext, seen: ReadonlySet<string> = new Set()): string {
  if (seen.has(name)) return name;
  const alias = context.pointerAliasFor(name);
  if (!alias) return name;
  return resolveAtomicRootName(alias.rootName, context, new Set([...seen, name]));
}

function flattenedPointerAlias(
  name: string,
  span: CudaLiteExpression["span"],
  context: EmitContext,
  seen: ReadonlySet<string> = new Set(),
): PointerAlias | undefined {
  if (seen.has(name)) return undefined;
  const alias = context.pointerAliasFor(name);
  if (!alias) return undefined;
  const parent = flattenedPointerAlias(alias.rootName, span, context, new Set([...seen, name]));
  if (!parent) return alias;
  const valueType = alias.valueType ?? parent.valueType;
  return {
    rootName: parent.rootName,
    ...(valueType === undefined ? {} : { valueType }),
    baseIndex: {
      kind: "binary",
      operator: "+",
      left: parent.baseIndex,
      right: alias.baseIndex,
      span,
    },
  };
}

function atomicStorageInfo(
  rootName: string,
  context: EmitContext,
): { readonly valueType: CudaLiteScalarType; readonly storageScalar: "i32" | "u32"; readonly addressSpace: "storage" | "workgroup" } | undefined {
  const param = context.paramFor(rootName);
  if (param?.pointer && context.ir.atomicParams.includes(rootName)) {
    return {
      valueType: param.valueType,
      storageScalar: param.valueType === "int" ? "i32" : "u32",
      addressSpace: "storage",
    };
  }
  if (context.isAtomicShared(rootName)) {
    const shared = context.ir.sharedDeclarations.find((item) => item.name === rootName);
    if (!shared) return undefined;
    return {
      valueType: shared.valueType,
      storageScalar: shared.valueType === "int" ? "i32" : "u32",
      addressSpace: "workgroup",
    };
  }
  if (context.isAtomicDeviceGlobal(rootName)) {
    const global = context.deviceGlobalFor(rootName);
    if (!global) return undefined;
    return {
      valueType: global.valueType,
      storageScalar: global.valueType === "int" ? "i32" : "u32",
      addressSpace: "storage",
    };
  }
  return undefined;
}

function integerAtomicLoopHelperName(kind: "Inc" | "Dec", target: AtomicTargetInfo): string {
  return `bg_atomic${kind}_${target.addressSpace}_${target.storageScalar}`;
}

function emitAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string {
  const pointerRebase = emitPointerRebaseAssignment(expression, context);
  if (pointerRebase) return pointerRebase;
  const localVectorAssignment = emitLocalVectorIndexAssignment(expression, context);
  if (localVectorAssignment) return localVectorAssignment;
  const storageView = storageViewLValue(expression.left, context);
  if (storageView && isCudaVectorType(storageView.valueType)) {
    const right = emitExpression(expression.right, context);
    if (storageView.field) {
      const current = `${storageView.name}[${storageView.index} + ${storageView.fieldIndex ?? 0}u]`;
      if (expression.operator !== "=") {
        const op = expression.operator.slice(0, -1);
        return `${current} = (${current} ${op} ${right})`;
      }
      return emitVectorStorageFieldWriteAt(storageView.name, storageView.index, storageView.fieldIndex ?? 0, right);
    }
    if (expression.operator !== "=") {
      const current = emitVectorStorageReadAt(storageView.name, storageView.valueType, storageView.index);
      const op = expression.operator.slice(0, -1);
      const vectorRight = emitExpressionAsVectorOperand(expression.right, storageView.valueType as CudaLiteVectorType, context);
      return emitVectorStorageWriteAt(storageView.name, storageView.valueType, storageView.index, `(${current} ${op} ${vectorRight})`);
    }
    return emitVectorStorageWriteAt(storageView.name, storageView.valueType, storageView.index, right);
  }
  const poolAccess = poolAccessForIndex(expression.left, context);
  if (poolAccess) return emitPoolAssignment(poolAccess, expression.operator, expression.right, context);
  const localVectorWholeAssignment = emitLocalVectorAssignment(expression, context);
  if (localVectorWholeAssignment) return localVectorWholeAssignment;
  const vectorAssignment = emitVectorAssignment(expression, context);
  if (vectorAssignment) return vectorAssignment;
  const pointerLvalue = devicePointerLValue(expression.left, context);
  if (pointerLvalue) {
    const right = emitExpression(expression.right, context);
    const read = `${pointerReadHelperName(pointerLvalue.valueType)}(${pointerLvalue.buffer}, ${pointerLvalue.index})`;
    if (pointerLvalue.fieldIndex !== undefined) {
      if (!isCudaVectorType(pointerLvalue.valueType)) {
        throw featureError("unsupported-vector-assignment", "member assignment through pointer expects a CUDA vector pointer");
      }
      const field = vectorFieldName(pointerLvalue.fieldIndex);
      const currentLane = `${read}.${field}`;
      const laneValue = expression.operator === "="
        ? right
        : `(${currentLane} ${expression.operator.slice(0, -1)} ${right})`;
      return `${pointerWriteHelperName(pointerLvalue.valueType)}(${pointerLvalue.buffer}, ${pointerLvalue.index}, ${emitVectorLaneSetExpression(read, pointerLvalue.valueType, pointerLvalue.fieldIndex, laneValue)})`;
    }
    const value = expression.operator === "="
      ? right
      : expression.operator === "+="
        ? `(${read} + ${right})`
        : expression.operator === "-="
          ? `(${read} - ${right})`
          : expression.operator === "*="
            ? `(${read} * ${right})`
            : expression.operator === "/="
              ? `(${read} / ${right})`
              : expression.operator === "<<="
                ? `(${read} << ${right})`
                : expression.operator === ">>="
                  ? `(${read} >> ${right})`
                  : expression.operator === "&="
                    ? `(${read} & ${right})`
                    : expression.operator === "|="
                      ? `(${read} | ${right})`
                      : `(${read} ^ ${right})`;
    return `${pointerWriteHelperName(pointerLvalue.valueType)}(${pointerLvalue.buffer}, ${pointerLvalue.index}, ${value})`;
  }
  const root = rootIdentifier(expression.left);
  const param = root ? context.paramFor(root) : undefined;
  const global = root ? context.deviceGlobalFor(root) : undefined;
  if (root && context.isAtomicShared(root)) {
    const value = emitExpression(expression.right, context);
    const target = emitExpression(expression.left, context, "lvalue");
    const shared = sharedDeclarationFor(root, context);
    if (shared?.valueType === "float" || shared?.valueType === "double") {
      if (expression.operator === "=") return `atomicStore(&${target}, bitcast<u32>(${value}))`;
      if (expression.operator === "+=") return `bg_atomicAdd_f32_workgroup(&${target}, ${value})`;
      if (expression.operator === "-=") return `bg_atomicSub_f32_workgroup(&${target}, ${value})`;
    } else {
      if (expression.operator === "=") return `atomicStore(&${target}, ${value})`;
      if (expression.operator === "+=") return `atomicAdd(&${target}, ${value})`;
      if (expression.operator === "-=") return `atomicSub(&${target}, ${value})`;
    }
  }
  if (param && context.ir.atomicParams.includes(param.name)) {
    const value = emitExpression(expression.right, context);
    const target = emitExpression(expression.left, context, "lvalue");
    if (param.valueType === "float" || param.valueType === "double") {
      if (expression.operator === "=") return `atomicStore(&${target}, bitcast<u32>(${value}))`;
      if (expression.operator === "+=") return `bg_atomicAdd_f32(&${target}, ${value})`;
      if (expression.operator === "-=") return `bg_atomicSub_f32(&${target}, ${value})`;
    } else {
      if (expression.operator === "=") return `atomicStore(&${target}, ${value})`;
      if (expression.operator === "+=") return `atomicAdd(&${target}, ${value})`;
    }
  }
  if (global && context.ir.atomicDeviceGlobals.includes(global.name)) {
    const value = emitExpression(expression.right, context);
    const target = emitExpression(expression.left, context, "lvalue");
    if (global.valueType === "float" || global.valueType === "double") {
      if (expression.operator === "=") return `atomicStore(&${target}, bitcast<u32>(${value}))`;
      if (expression.operator === "+=") return `bg_atomicAdd_f32(&${target}, ${value})`;
      if (expression.operator === "-=") return `bg_atomicSub_f32(&${target}, ${value})`;
    } else {
      if (expression.operator === "=") return `atomicStore(&${target}, ${value})`;
      if (expression.operator === "+=") return `atomicAdd(&${target}, ${value})`;
      if (expression.operator === "-=") return `atomicSub(&${target}, ${value})`;
    }
  }
  const left = emitExpression(expression.left, context, "lvalue");
  const right = emitExpression(expression.right, context);
  const scalarParamLocal = root !== undefined && param?.pointer === false && context.isLocalName(root);
  if (((param?.valueType === "bool" && !scalarParamLocal) || global?.valueType === "bool") && (expression.left.kind === "index" || expression.left.kind === "identifier")) {
    if (expression.operator === "=") return `${left} = select(0u, 1u, ${right})`;
  }
  if (expression.operator === "<<=") return `${left} = (${left} << ${right})`;
  if (expression.operator === ">>=") return `${left} = (${left} >> ${right})`;
  if (expression.operator === "&=") return `${left} = (${left} & ${right})`;
  if (expression.operator === "|=") return `${left} = (${left} | ${right})`;
  if (expression.operator === "^=") return `${left} = (${left} ^ ${right})`;
  return `${left} ${expression.operator} ${right}`;
}

function emitLocalVectorIndexAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string | undefined {
  if (expression.left.kind !== "index" || expression.left.target.kind !== "identifier") return undefined;
  const name = expression.left.target.name;
  const type = context.localValueTypeFor(name);
  if (!isCudaVectorType(type)) return undefined;
  const index = `u32(${emitExpression(expression.left.index, context)})`;
  const right = emitExpression(expression.right, context);
  const current = `${name}[${index}]`;
  const value = expression.operator === "=" ? right : `(${current} ${expression.operator.slice(0, -1)} ${right})`;
  return `${name} = ${emitVectorLaneSet(name, type, index, value)}`;
}

function emitLocalVectorAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string | undefined {
  if (expression.left.kind !== "identifier") return undefined;
  const name = expression.left.name;
  const type = context.localValueTypeFor(name);
  if (!isCudaVectorType(type) || expression.operator === "=") return undefined;
  const op = expression.operator.slice(0, -1);
  if (!isVectorArithmeticOperator(op)) return undefined;
  const left = context.nameFor(name);
  const right = emitExpressionAsVectorOperand(expression.right, type, context);
  return `${left} = (${left} ${op} ${right})`;
}

function emitVectorLaneSet(name: string, type: CudaLiteScalarType, index: string, value: string): string {
  return emitVectorLaneSetExpression(name, type, index, value);
}

function emitVectorLaneSetExpression(base: string, type: CudaLiteScalarType, index: string | number, value: string): string {
  const scalar = wgslScalar(cudaVectorScalarType(type) ?? "float");
  const indexExpression = typeof index === "number" ? `${index}u` : index;
  const values = Array.from({ length: cudaVectorLaneCount(type) }, (_, lane) => {
    const current = `(${base}).${vectorFieldName(lane)}`;
    return `select(${current}, ${scalar}(${value}), ${indexExpression} == ${lane}u)`;
  });
  return `${wgslScalar(type)}(${values.join(", ")})`;
}

function emitPointerRebaseAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string | undefined {
  if (expression.left.kind !== "identifier") return undefined;
  const handle = context.localPointerHandleFor(expression.left.name);
  if (handle) {
    const buffer = context.nameFor(`${handle.name}_buffer`);
    const base = context.nameFor(`${handle.name}_base`);
    if (expression.operator === "=") {
      const parts = devicePointerArgumentParts(expression.right, context);
      if (!parts) throw featureError("unsupported-pointer-assignment", "CUDA pointer assignment expects a modeled storage or shared pointer");
      return `${buffer} = ${parts.buffer}; ${base} = ${parts.base}`;
    }
    if (expression.operator === "+=" || expression.operator === "-=") {
      const delta = `u32(${emitExpression(expression.right, context)})`;
      const op = expression.operator === "+=" ? "+" : "-";
      return `${base} = (${base} ${op} ${delta})`;
    }
    return undefined;
  }
  const base = context.mutablePointerBaseFor(expression.left.name);
  if (!base) return undefined;
  if (expression.operator === "=") {
    const parts = devicePointerArgumentParts(expression.right, context);
    if (!parts) throw featureError("unsupported-pointer-assignment", "CUDA pointer assignment expects a modeled storage or shared pointer");
    const expectedBuffer = context.storagePointerIdFor(expression.left.name);
    if (expectedBuffer !== undefined && parts.buffer !== `${expectedBuffer}u`) {
      throw featureError("unsupported-pointer-assignment", "storage pointer parameter assignment must stay within the same buffer");
    }
    return `${base} = ${parts.base}`;
  }
  if (expression.operator !== "+=" && expression.operator !== "-=") return undefined;
  const delta = `u32(${emitExpression(expression.right, context)})`;
  const op = expression.operator === "+=" ? "+" : "-";
  return `${base} = (${base} ${op} ${delta})`;
}

function emitPointerRebaseUpdate(expression: CudaLiteExpression, context: EmitContext): string | undefined {
  if (expression.kind !== "update" || expression.argument.kind !== "identifier") return undefined;
  const base = context.mutablePointerBaseFor(expression.argument.name);
  if (!base) return undefined;
  return `${base} = (${base} ${expression.operator === "++" ? "+" : "-"} 1u)`;
}

function emitVectorAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string | undefined {
  const direct = vectorStorageLValue(expression.left, context);
  if (!direct) return undefined;
  const right = emitExpression(expression.right, context);
  if (direct.field) {
    if (expression.operator !== "=") {
      const current = `${direct.name}[${vectorStorageBase(direct.index, direct.lanes)} + ${direct.fieldIndex ?? 0}u]`;
      const op = expression.operator.slice(0, -1);
      return `${current} = (${current} ${op} ${right})`;
    }
    return emitVectorStorageFieldWrite(direct.name, direct.valueType, direct.index, direct.field, right);
  }
  if (expression.operator !== "=") {
    const current = emitVectorStorageRead(direct.name, direct.valueType, direct.index);
    const op = expression.operator.slice(0, -1);
    const vectorRight = emitExpressionAsVectorOperand(expression.right, direct.valueType as CudaLiteVectorType, context);
    return emitVectorStorageWrite(direct.name, direct.valueType, direct.index, `(${current} ${op} ${vectorRight})`);
  }
  return emitVectorStorageWrite(direct.name, direct.valueType, direct.index, right);
}

interface VectorStorageLValue {
  readonly name: string;
  readonly valueType: CudaLiteScalarType;
  readonly index: string;
  readonly lanes: number;
  readonly field?: string;
  readonly fieldIndex?: number;
}

function vectorStorageLValue(expression: CudaLiteExpression, context: EmitContext): VectorStorageLValue | undefined {
  let target = expression;
  let field: string | undefined;
  if (expression.kind === "member") {
    field = expression.property;
    target = expression.object;
  }
  if (target.kind !== "index" || target.target.kind !== "identifier") return undefined;
  const param = context.paramFor(target.target.name);
  if (!param || !isCudaVectorType(param.valueType)) return undefined;
  const index = emitPointerIndex(param.name, target.index, context);
  const base = {
    name: param.name,
    valueType: param.valueType,
    index,
    lanes: cudaVectorLaneCount(param.valueType),
  };
  if (!field) return base;
  const fieldIndex = cudaVectorFieldIndex(param.valueType, field);
  return fieldIndex === undefined ? undefined : { ...base, field, fieldIndex };
}

function storageViewLValue(expression: CudaLiteExpression, context: EmitContext): VectorStorageLValue | undefined {
  let target = expression;
  let field: string | undefined;
  if (expression.kind === "member") {
    field = expression.property;
    target = expression.object;
  }
  const view = target.kind === "index"
    ? storageViewForPointerExpression(target.target, target.index, context)
    : target.kind === "unary" && target.operator === "*"
      ? storageViewForPointerExpression(target.argument, zeroExpression(target.span), context)
      : undefined;
  if (!view || !isCudaVectorType(view.valueType)) return undefined;
  const base = {
    name: view.rootName,
    valueType: view.valueType,
    index: view.index,
    lanes: cudaVectorLaneCount(view.valueType),
  };
  if (!field) return base;
  const fieldIndex = cudaVectorFieldIndex(view.valueType, field);
  return fieldIndex === undefined ? undefined : { ...base, field, fieldIndex };
}

function storageViewForPointerExpression(
  pointer: CudaLiteExpression,
  index: CudaLiteExpression,
  context: EmitContext,
): { readonly rootName: string; readonly valueType: CudaLiteScalarType; readonly index: string } | undefined {
  if (pointer.kind === "cast" && pointer.pointer) {
    const alias = pointerAliasForPointerExpression(pointer.expression, pointer.valueType);
    if (!alias || context.devicePointerParamFor(alias.rootName)) return undefined;
    return {
      rootName: alias.rootName,
      valueType: pointer.valueType,
      index: emitStorageViewIndex(alias.rootName, alias.baseIndex, index, pointer.valueType, context),
    };
  }
  if (pointer.kind === "identifier") {
    const alias = context.pointerAliasFor(pointer.name);
    if (!alias?.valueType || context.devicePointerParamFor(alias.rootName)) return undefined;
    return {
      rootName: alias.rootName,
      valueType: alias.valueType,
      index: emitStorageViewIndex(alias.rootName, alias.baseIndex, index, alias.valueType, context),
    };
  }
  const valueType = devicePointerValueTypeForExpression(pointer, context);
  const alias = pointerAliasForPointerExpression(pointer, valueType);
  if (alias?.valueType && !context.devicePointerParamFor(alias.rootName)) {
    return {
      rootName: alias.rootName,
      valueType: alias.valueType,
      index: emitStorageViewIndex(alias.rootName, alias.baseIndex, index, alias.valueType, context),
    };
  }
  return undefined;
}

function isBarrierCall(expression: CudaLiteExpression): boolean {
  if (expression.kind !== "call") return false;
  const name = expressionName(expression.callee);
  return name === "__syncthreads" || name === "__syncwarp";
}

function noopCallComment(expression: CudaLiteExpression): string | undefined {
  if (expression.kind !== "call") return undefined;
  switch (expressionName(expression.callee)) {
    case "printf":
      return "printf omitted: WebGPU has no device stdout";
    case "cudaDeviceSynchronize":
      return "cudaDeviceSynchronize omitted: WebGPU dispatch completion is host-managed";
    case "cudaStreamCreate":
    case "cudaStreamCreateWithFlags":
    case "cudaStreamDestroy":
    case "cudaStreamSynchronize":
    case "cudaEventCreate":
    case "cudaEventCreateWithFlags":
    case "cudaEventDestroy":
    case "cudaEventRecord":
    case "cudaEventSynchronize":
      return `${expressionName(expression.callee)} omitted: WebGPU stream/event orchestration is host-managed`;
    case "cudaMemcpy":
      return "cudaMemcpy omitted: WebGPU copy orchestration is host-managed";
    case "cudaMemcpyAsync":
      return "cudaMemcpyAsync omitted: WebGPU copy orchestration is host-managed";
    case "cudaMemcpyPeerAsync":
      return "cudaMemcpyPeerAsync omitted: WebGPU copy orchestration is host-managed";
    default:
      return undefined;
  }
}

function emitSharedType(statement: CudaLiteVarDecl, ir: KernelIrModule): string {
  let type = ir.atomicShared.includes(statement.name)
    ? `atomic<${statement.valueType === "float" ? "u32" : wgslScalar(statement.valueType)}>`
    : wgslScalar(statement.valueType);
  for (let i = statement.dimensions.length - 1; i >= 0; i--) {
    type = `array<${type}, ${statement.dimensions[i]!}>`;
  }
  return type;
}

function emitLocalArrayType(statement: CudaLiteVarDecl): string {
  let type = wgslScalar(statement.valueType);
  const dimensions = matrixTileStorageDimensions(statement);
  for (let i = dimensions.length - 1; i >= 0; i--) {
    type = `array<${type}, ${dimensions[i]!}>`;
  }
  return type;
}

function emitLocalArrayInitializer(
  statement: CudaLiteVarDecl,
  context: EmitContext,
  indentLevel: number,
): string[] {
  if (!statement.init) return [];
  const elements = flattenInitializerExpressions(statement.init);
  if (elements.length === 0) return [];
  const prefix = indent(indentLevel);
  const totalElements = statement.dimensions.reduce((product, item) => product * item, 1);
  return elements.slice(0, totalElements).map((element, flatIndex) =>
    `${prefix}${emitLocalArrayElementAccess(context.nameFor(statement.name), statement.dimensions, flatIndex)} = ${emitExpression(element, context)};`
  );
}

function flattenInitializerExpressions(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  if (expression.kind !== "initializer") return [expression];
  return expression.elements.flatMap((element) => flattenInitializerExpressions(element));
}

function emitLocalArrayElementAccess(name: string, dimensions: readonly number[], flatIndex: number): string {
  let remainder = flatIndex;
  const indices: number[] = [];
  for (let i = dimensions.length - 1; i >= 0; i--) {
    const size = dimensions[i]!;
    indices.unshift(remainder % size);
    remainder = Math.trunc(remainder / size);
  }
  return indices.reduce((expr, index) => `${expr}[${index}]`, name);
}

function emitConstantArrayType(constant: CudaLiteGlobalConstant): string {
  let type = wgslScalar(cudaVectorScalarType(constant.valueType) ?? constant.valueType);
  const dimensions = externalConstantDimensions(constant);
  for (let i = dimensions.length - 1; i >= 0; i--) {
    type = `array<${type}, ${dimensions[i]!}>`;
  }
  return type;
}

function isExternalConstantBinding(constant: CudaLiteGlobalConstant): boolean {
  return constant.init === undefined && (constant.dimensions.length > 0 || isCudaVectorType(constant.valueType));
}

function externalConstantDimensions(constant: CudaLiteGlobalConstant): readonly number[] {
  if (!isCudaVectorType(constant.valueType)) return constant.dimensions;
  const elements = constant.dimensions.length === 0
    ? 1
    : constant.dimensions.reduce((product, dimension) => product * dimension, 1);
  return [elements * cudaVectorLaneCount(constant.valueType)];
}

function emitInitializedConstant(constant: CudaLiteGlobalConstant, context: EmitContext): string {
  if (!constant.init) throw featureError("invalid-constant-initializer", `constant '${constant.name}' has no initializer`);
  if (constant.dimensions.length === 0) {
    if (isCudaVectorType(constant.valueType)) {
      if (constant.init.kind !== "initializer") {
        return `const ${context.nameFor(constant.name)}: ${wgslScalar(constant.valueType)} = ${emitConstantInitializerExpression(constant.init, context)};`;
      }
      const lanes = cudaVectorLaneCount(constant.valueType);
      const values = flattenInitializerExpressions(constant.init)
        .slice(0, lanes)
        .map((expression) => emitConstantInitializerExpression(expression, context));
      while (values.length < lanes) values.push(zeroValue(cudaVectorScalarType(constant.valueType) ?? "float"));
      return `const ${context.nameFor(constant.name)}: ${wgslScalar(constant.valueType)} = ${wgslScalar(constant.valueType)}(${values.join(", ")});`;
    }
    return `const ${context.nameFor(constant.name)}: ${wgslScalar(constant.valueType)} = ${emitConstantInitializerExpression(constant.init, context)};`;
  }
  const type = emitConstantArrayType(constant);
  const total = constant.dimensions.reduce((product, dimension) => product * dimension, 1);
  const values = flattenInitializerExpressions(constant.init)
    .slice(0, total)
    .map((expression) => emitConstantInitializerExpression(expression, context));
  while (values.length < total) values.push(zeroValue(constant.valueType));
  return `const ${context.nameFor(constant.name)}: ${type} = ${type}(${values.join(", ")});`;
}

function emitConstantInitializerExpression(expression: CudaLiteExpression, context: EmitContext): string {
  if (expression.kind === "initializer") {
    throw featureError("invalid-constant-initializer", "nested constant initializer must be flattened before WGSL emission");
  }
  return emitExpression(expression, context);
}

function emitTextureHelper(name: string, context: EmitContext): string[] {
  const safeName = context.nameFor(name);
  const lines = [
    `fn bg_tex2d_coord_${name}(x: f32, y: f32) -> vec2<i32> {`,
    `  let dims = textureDimensions(${safeName});`,
    "  let max_coord = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);",
    "  return clamp(vec2<i32>(i32(floor(x)), i32(floor(y))), vec2<i32>(0, 0), max_coord);",
    "}",
    `fn bg_tex2d_f32_${name}(x: f32, y: f32) -> f32 {`,
    `  return textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).r;`,
    "}",
    `fn bg_tex2d_float2_${name}(x: f32, y: f32) -> vec2<f32> {`,
    `  return textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).xy;`,
    "}",
    `fn bg_tex2d_float3_${name}(x: f32, y: f32) -> vec3<f32> {`,
    `  return textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).xyz;`,
    "}",
    `fn bg_tex2d_float4_${name}(x: f32, y: f32) -> vec4<f32> {`,
    `  return textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0);`,
    "}",
    `fn bg_tex2d_int_${name}(x: f32, y: f32) -> i32 {`,
    `  return i32(textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).r);`,
    "}",
    `fn bg_tex2d_uint_${name}(x: f32, y: f32) -> u32 {`,
    `  return u32(textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).r);`,
    "}",
    ...emitTextureVectorCastHelpers(name, safeName, ["int2", "int3", "int4", "uint2", "uint3", "uint4"]),
  ];
  if (context.ir.requiredFeatures.includes("shader-f16")) {
    lines.push(
      `fn bg_tex2d_half_${name}(x: f32, y: f32) -> f16 {`,
      `  return f16(textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).r);`,
      "}",
      `fn bg_tex2d_half2_${name}(x: f32, y: f32) -> vec2<f16> {`,
      `  return vec2<f16>(textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0).xy);`,
      "}",
    );
  }
  return lines;
}

function emitCubeTextureAtlasHelpers(): string[] {
  return [
    "fn bg_cube_face(x: f32, y: f32, z: f32) -> f32 {",
    "  let ax = abs(x);",
    "  let ay = abs(y);",
    "  let az = abs(z);",
    "  if (ax >= ay && ax >= az) {",
    "    return select(1.0, 0.0, x >= 0.0);",
    "  }",
    "  if (ay >= az) {",
    "    return select(3.0, 2.0, y >= 0.0);",
    "  }",
    "  return select(5.0, 4.0, z >= 0.0);",
    "}",
    "fn bg_cube_u(x: f32, y: f32, z: f32) -> f32 {",
    "  let ax = max(abs(x), 0.000001);",
    "  let ay = max(abs(y), 0.000001);",
    "  let az = max(abs(z), 0.000001);",
    "  if (ax >= ay && ax >= az) { return z / ax; }",
    "  return x / max(ay, az);",
    "}",
    "fn bg_cube_v(x: f32, y: f32, z: f32) -> f32 {",
    "  let ax = max(abs(x), 0.000001);",
    "  let ay = max(abs(y), 0.000001);",
    "  let az = max(abs(z), 0.000001);",
    "  if (ax >= ay && ax >= az) { return y / ax; }",
    "  if (ay >= az) { return z / ay; }",
    "  return y / az;",
    "}",
  ];
}

function textureReadHelperSuffix(valueType: CudaLiteScalarType | undefined): string {
  if (valueType === "int" || valueType === "uint") return valueType;
  if (valueType === "half" || valueType === "half2") return valueType;
  if (isCudaVectorType(valueType)) return valueType;
  return "f32";
}

function isTextureBindingName(name: string, context: EmitContext): boolean {
  return textureBindings(context.ir).some((texture) => texture.name === name);
}

function isTextureReadCall(name: string): boolean {
  return name === "tex1D" ||
    name === "tex1Dfetch" ||
    name === "tex2D" ||
    name === "tex2DLod" ||
    name === "tex2DLayered" ||
    name === "tex3D" ||
    name === "texCubemap";
}

function textureReadArgsForEmit(expression: CudaLiteCallExpression, context: EmitContext): readonly [string, string] {
  const name = expressionName(expression.callee);
  const x = textureCoordForEmit(expression.args[1], context);
  if (name === "tex1D" || name === "tex1Dfetch") return [x, "0.0"];
  const y = textureCoordForEmit(expression.args[2], context);
  if (name === "tex2D" || name === "tex2DLod") return [x, y];
  if (name === "tex2DLayered" || name === "tex3D") {
    return [x, `(${y} + ${textureCoordForEmit(expression.args[3], context)})`];
  }
  if (name === "texCubemap") {
    const z = textureCoordForEmit(expression.args[3], context);
    const texture = emitTextureArgument(expression.args[0], context);
    const width = `f32(textureDimensions(${texture}).x)`;
    const localX = `((bg_cube_u(${x}, ${y}, ${z}) + 1.0) * 0.5 * (${width} - 1.0))`;
    const localY = `((bg_cube_v(${x}, ${y}, ${z}) + 1.0) * 0.5 * (${width} - 1.0) + bg_cube_face(${x}, ${y}, ${z}) * ${width})`;
    return [localX, localY];
  }
  return [x, y];
}

function textureCoordForEmit(expression: CudaLiteExpression | undefined, context: EmitContext): string {
  return expression ? `f32(${emitExpression(expression, context)})` : "0.0";
}

function emitTextureArgument(expression: CudaLiteExpression | undefined, context: EmitContext): string {
  if (expression?.kind === "identifier") return context.nameFor(expression.name);
  if (expression) return emitExpression(expression, context);
  return "bg_missing_texture";
}

function emitTextureReadExpression(
  texture: string,
  coordArgs: readonly [string, string],
  valueType: CudaLiteScalarType | undefined,
): string {
  const [x, y] = coordArgs;
  const coord = `clamp(vec2<i32>(i32(floor(${x})), i32(floor(${y}))), vec2<i32>(0, 0), vec2<i32>(textureDimensions(${texture})) - vec2<i32>(1, 1))`;
  const value = `textureLoad(${texture}, ${coord}, 0)`;
  if (valueType === "float2") return `${value}.xy`;
  if (valueType === "float3") return `${value}.xyz`;
  if (valueType === "float4") return value;
  if (valueType === "int") return `i32(${value}.r)`;
  if (valueType === "uint") return `u32(${value}.r)`;
  if (valueType === "half") return `f16(${value}.r)`;
  if (valueType === "half2") return `vec2<f16>(${value}.xy)`;
  if (isCudaVectorType(valueType)) {
    const fields = ["x", "y", "z", "w"].slice(0, cudaVectorLaneCount(valueType));
    const scalarType = wgslScalar(cudaVectorScalarType(valueType) ?? "float");
    return `${wgslScalar(valueType)}(${fields.map((field) => `${scalarType}(${value}.${field})`).join(", ")})`;
  }
  return `${value}.r`;
}

function emitSurfaceWriteExpression(
  surfaceName: string,
  valueExpression: CudaLiteExpression,
  xBytesExpression: CudaLiteExpression,
  y: string,
  z: string,
  context: EmitContext,
): string {
  const valueType = expressionValueTypeForEmit(valueExpression, context);
  if (isCudaVectorType(valueType)) {
    const value = emitExpression(valueExpression, context);
    const fields = ["x", "y", "z", "w"].slice(0, cudaVectorLaneCount(valueType));
    return [
      ...fields.map((field, index) =>
        `bg_surf2dwrite_${surfaceName}(f32(${value}.${field}), (${emitExpression(xBytesExpression, context)} + ${index * 4}), ${y}, ${z})`,
      ),
      "0",
    ].join("; ");
  }
  return `bg_surf2dwrite_${surfaceName}(${emitExpressionAsValueType(valueExpression, "float", context)}, ${emitExpression(xBytesExpression, context)}, ${y}, ${z})`;
}

function emitTextureVectorCastHelpers(
  name: string,
  safeName: string,
  valueTypes: readonly CudaLiteVectorType[],
): string[] {
  return valueTypes.flatMap((valueType) => {
    const fields = ["x", "y", "z", "w"].slice(0, cudaVectorLaneCount(valueType));
    const scalarType = wgslScalar(cudaVectorScalarType(valueType) ?? "float");
    const values = fields.map((field) => `${scalarType}(value.${field})`).join(", ");
    return [
      `fn bg_tex2d_${valueType}_${name}(x: f32, y: f32) -> ${wgslScalar(valueType)} {`,
      `  let value = textureLoad(${safeName}, bg_tex2d_coord_${name}(x, y), 0);`,
      `  return ${wgslScalar(valueType)}(${values});`,
      "}",
    ];
  });
}

function emitSurfaceHelper(name: string, context: EmitContext): string[] {
  const safeName = context.nameFor(name);
  return [
    `fn bg_surf2dread_${name}(x_bytes: i32, y: i32) -> f32 {`,
    "  let x = x_bytes / 4;",
    `  let index = y * i32(params.${context.nameFor(surfaceWidthField(name))}) + x;`,
    `  if (index >= 0 && index < i32(arrayLength(&${safeName}))) {`,
    `    return ${safeName}[index];`,
    "  }",
    "  return 0.0;",
    "}",
    `fn bg_surf2dwrite_${name}(value: f32, x_bytes: i32, y: i32, z: i32) {`,
    "  let x = x_bytes / 4;",
    `  let index = ((z * i32(params.${context.nameFor(surfaceHeightField(name))})) + y) * i32(params.${context.nameFor(surfaceWidthField(name))}) + x;`,
    `  if (index >= 0 && index < i32(arrayLength(&${safeName}))) {`,
    `    ${safeName}[index] = value;`,
    "  }",
    "}",
  ];
}

function emitSharedAddressIndex(expression: CudaLiteExpression, context: EmitContext): string | undefined {
  const target = expression.kind === "unary" && expression.operator === "&" ? expression.argument : expression;
  const indexes: CudaLiteExpression[] = [];
  let cursor = target;
  while (cursor.kind === "index") {
    indexes.unshift(cursor.index);
    cursor = cursor.target;
  }
  if (cursor.kind !== "identifier") return undefined;
  const declaration = context.ir.sharedDeclarations.find((item) => item.name === cursor.name);
  if (!declaration) return undefined;
  if (indexes.length === 0) return "0u";
  const terms: string[] = [];
  for (let i = 0; i < indexes.length; i++) {
    const stride = declaration.dimensions.slice(i + 1).reduce((product, dimension) => product * dimension, 1);
    const value = `u32(${emitExpression(indexes[i]!, context)})`;
    terms.push(stride === 1 ? value : `(${value} * ${stride}u)`);
  }
  return terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
}

interface PoolAccess {
  readonly poolName: string;
  readonly pointerExpression: CudaLiteExpression;
  readonly indexExpression: CudaLiteExpression;
  readonly valueType: CudaLiteScalarType;
  readonly rawBuffer?: boolean;
}

function poolAccessForIndex(expression: CudaLiteExpression, context: EmitContext): PoolAccess | undefined {
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

function poolPointerExpressionInfo(
  expression: CudaLiteExpression,
  context: EmitContext,
): PoolPointerAlias | undefined {
  if (expression.kind === "identifier") return context.poolPointerFor(expression.name);
  if (expression.kind === "call") return poolPointerForAllocationCall(expression);
  if (expression.kind === "cast" && expression.pointer) return poolPointerExpressionInfo(expression.expression, context);
  return undefined;
}

function emitPoolRead(access: PoolAccess, context: EmitContext): string {
  const raw = poolRawAccess(access, context);
  if (access.rawBuffer) return raw;
  return decodePoolWord(access.valueType, raw);
}

function emitPoolAssignment(
  access: PoolAccess,
  operator: string,
  right: CudaLiteExpression,
  context: EmitContext,
): string {
  const raw = poolRawAccess(access, context);
  const rightValue = emitExpression(right, context);
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

function poolRawAccess(access: PoolAccess, context: EmitContext): string {
  const name = access.rawBuffer ? access.poolName : poolDataName(access.poolName);
  return `${context.nameFor(name)}[${poolWordIndex(access, context)}]`;
}

function poolWordIndex(access: PoolAccess, context: EmitContext): string {
  const pointer = emitExpression(access.pointerExpression, context);
  const index = emitExpression(access.indexExpression, context);
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

function emitPoolHelper(name: string, context: EmitContext): string[] {
  const dataName = context.nameFor(poolDataName(name));
  const offsetName = context.nameFor(poolOffsetName(name));
  return [
    `fn bg_pool_alloc_${name}(size_bytes: u32) -> u32 {`,
    `  let old = atomicAdd(&${offsetName}, size_bytes);`,
    `  let capacity = arrayLength(&${dataName}) * 4u;`,
    "  if ((old + size_bytes) > capacity) {",
    "    return 0u;",
    "  }",
    "  return old + 1u;",
    "}",
  ];
}

function emitRawPoolHelper(allocator: RawPoolAllocator): string[] {
  return [
    `fn ${rawPoolHelperName(allocator.baseName, allocator.offsetName)}(pool_size_bytes: u32, size_bytes: u32) -> u32 {`,
    `  let old = atomicAdd(&${allocator.offsetName}[0], size_bytes);`,
    "  if ((old + size_bytes) > pool_size_bytes) {",
    "    return 0u;",
    "  }",
    "  return old + 1u;",
    "}",
  ];
}

function floatAtomicHelperName(kind: "Add" | "Sub" | "Min" | "Max", addressSpace: "storage" | "workgroup"): string {
  return addressSpace === "storage" ? `bg_atomic${kind}_f32` : `bg_atomic${kind}_f32_workgroup`;
}

function emitFloatAtomicAddHelper(addressSpace: "storage" | "workgroup"): string[] {
  const name = floatAtomicHelperName("Add", addressSpace);
  return [
    `fn ${name}(ptr_value: ptr<${addressSpace}, atomic<u32>, read_write>, value: f32) -> f32 {`,
    "  var old_bits = atomicLoad(ptr_value);",
    "  loop {",
    "    let old_value = bitcast<f32>(old_bits);",
    "    let new_bits = bitcast<u32>(old_value + value);",
    "    let result = atomicCompareExchangeWeak(ptr_value, old_bits, new_bits);",
    "    if (result.exchanged) {",
    "      return old_value;",
    "    }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
  ];
}

function emitFloatAtomicSubHelper(addressSpace: "storage" | "workgroup"): string[] {
  const name = floatAtomicHelperName("Sub", addressSpace);
  return [
    `fn ${name}(ptr_value: ptr<${addressSpace}, atomic<u32>, read_write>, value: f32) -> f32 {`,
    "  var old_bits = atomicLoad(ptr_value);",
    "  loop {",
    "    let old_value = bitcast<f32>(old_bits);",
    "    let new_bits = bitcast<u32>(old_value - value);",
    "    let result = atomicCompareExchangeWeak(ptr_value, old_bits, new_bits);",
    "    if (result.exchanged) {",
    "      return old_value;",
    "    }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
  ];
}

function emitFloatAtomicMinHelper(addressSpace: "storage" | "workgroup"): string[] {
  const name = floatAtomicHelperName("Min", addressSpace);
  return [
    `fn ${name}(ptr_value: ptr<${addressSpace}, atomic<u32>, read_write>, value: f32) -> f32 {`,
    "  var old_bits = atomicLoad(ptr_value);",
    "  loop {",
    "    let old_value = bitcast<f32>(old_bits);",
    "    let new_value = min(old_value, value);",
    "    let new_bits = bitcast<u32>(new_value);",
    "    if (new_bits == old_bits) {",
    "      return old_value;",
    "    }",
    "    let result = atomicCompareExchangeWeak(ptr_value, old_bits, new_bits);",
    "    if (result.exchanged) {",
    "      return old_value;",
    "    }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
  ];
}

function emitFloatAtomicMaxHelper(addressSpace: "storage" | "workgroup"): string[] {
  const name = floatAtomicHelperName("Max", addressSpace);
  return [
    `fn ${name}(ptr_value: ptr<${addressSpace}, atomic<u32>, read_write>, value: f32) -> f32 {`,
    "  var old_bits = atomicLoad(ptr_value);",
    "  loop {",
    "    let old_value = bitcast<f32>(old_bits);",
    "    let new_value = max(old_value, value);",
    "    let new_bits = bitcast<u32>(new_value);",
    "    if (new_bits == old_bits) {",
    "      return old_value;",
    "    }",
    "    let result = atomicCompareExchangeWeak(ptr_value, old_bits, new_bits);",
    "    if (result.exchanged) {",
    "      return old_value;",
    "    }",
    "    old_bits = result.old_value;",
    "  }",
    "}",
  ];
}

function emitIntegerAtomicLoopHelpers(): string[] {
  return [
    ...emitIntegerAtomicIncHelper("storage", "u32"),
    "",
    ...emitIntegerAtomicIncHelper("storage", "i32"),
    "",
    ...emitIntegerAtomicIncHelper("workgroup", "u32"),
    "",
    ...emitIntegerAtomicIncHelper("workgroup", "i32"),
    "",
    ...emitIntegerAtomicDecHelper("storage", "u32"),
    "",
    ...emitIntegerAtomicDecHelper("storage", "i32"),
    "",
    ...emitIntegerAtomicDecHelper("workgroup", "u32"),
    "",
    ...emitIntegerAtomicDecHelper("workgroup", "i32"),
  ];
}

function emitIntegerAtomicIncHelper(addressSpace: "storage" | "workgroup", scalar: "i32" | "u32"): string[] {
  const name = `bg_atomicInc_${addressSpace}_${scalar}`;
  const load = scalar === "u32" ? "atomicLoad(ptr_value)" : "bitcast<u32>(atomicLoad(ptr_value))";
  const compare = scalar === "u32" ? "old_bits" : "bitcast<i32>(old_bits)";
  const store = scalar === "u32" ? "next_bits" : "bitcast<i32>(next_bits)";
  return [
    `fn ${name}(ptr_value: ptr<${addressSpace}, atomic<${scalar}>, read_write>, limit: u32) -> u32 {`,
    `  var old_bits = ${load};`,
    "  loop {",
    "    let next_bits = select(old_bits + 1u, 0u, old_bits >= limit);",
    `    let result = atomicCompareExchangeWeak(ptr_value, ${compare}, ${store});`,
    "    if (result.exchanged) {",
    "      return old_bits;",
    "    }",
    `    old_bits = ${scalar === "u32" ? "result.old_value" : "bitcast<u32>(result.old_value)"};`,
    "  }",
    "}",
  ];
}

function emitIntegerAtomicDecHelper(addressSpace: "storage" | "workgroup", scalar: "i32" | "u32"): string[] {
  const name = `bg_atomicDec_${addressSpace}_${scalar}`;
  const load = scalar === "u32" ? "atomicLoad(ptr_value)" : "bitcast<u32>(atomicLoad(ptr_value))";
  const compare = scalar === "u32" ? "old_bits" : "bitcast<i32>(old_bits)";
  const store = scalar === "u32" ? "next_bits" : "bitcast<i32>(next_bits)";
  return [
    `fn ${name}(ptr_value: ptr<${addressSpace}, atomic<${scalar}>, read_write>, limit: u32) -> u32 {`,
    `  var old_bits = ${load};`,
    "  loop {",
    "    let next_bits = select(old_bits - 1u, limit, old_bits == 0u || old_bits > limit);",
    `    let result = atomicCompareExchangeWeak(ptr_value, ${compare}, ${store});`,
    "    if (result.exchanged) {",
    "      return old_bits;",
    "    }",
    `    old_bits = ${scalar === "u32" ? "result.old_value" : "bitcast<u32>(result.old_value)"};`,
    "  }",
    "}",
  ];
}

function emitCurandHelpers(): string[] {
  return [
    "fn bg_curand_next(state: ptr<function, u32>) -> u32 {",
    "  var x = *state;",
    "  x = (x * 1664525u) + 1013904223u;",
    "  *state = x;",
    "  return x;",
    "}",
    "fn bg_curand_init(seed: u32, sequence: u32, offset: u32, state: ptr<function, u32>) {",
    "  *state = seed ^ (sequence * 747796405u) ^ offset ^ 2891336453u;",
    "  _ = bg_curand_next(state);",
    "}",
    "fn bg_curand_uniform(state: ptr<function, u32>) -> f32 {",
    "  let bits = bg_curand_next(state);",
    "  return (f32(bits) + 1.0) * 2.3283064365386963e-10;",
    "}",
    "fn bg_curand_normal(state: ptr<function, u32>) -> f32 {",
    "  let u1 = max(bg_curand_uniform(state), 1.1754943508222875e-38);",
    "  let u2 = bg_curand_uniform(state);",
    "  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);",
    "}",
  ];
}

function emitFrexpHelpers(): string[] {
  return [
    "fn bg_frexp(value: f32, exponent_out: ptr<function, i32>) -> f32 {",
    "  if (value == 0.0 || isNan(value) || isInf(value)) {",
    "    *exponent_out = 0;",
    "    return value;",
    "  }",
    "  let exponent = i32(floor(log2(abs(value)))) + 1;",
    "  *exponent_out = exponent;",
    "  return value / exp2(f32(exponent));",
    "}",
  ];
}

function storageElementType(param: CudaLiteParam, ir: KernelIrModule): string {
  if (isCudaVectorType(param.valueType)) return wgslScalar(cudaVectorScalarType(param.valueType) ?? "float");
  if (param.valueType === "bool") return "u32";
  if (!ir.atomicParams.includes(param.name)) return wgslScalar(param.valueType);
  if (param.valueType === "float" || param.valueType === "double") return "atomic<u32>";
  return `atomic<${wgslScalar(param.valueType)}>`;
}

function deviceGlobalStorageElementType(global: CudaLiteDeviceGlobal, ir: KernelIrModule): string {
  if (isCudaVectorType(global.valueType)) return wgslScalar(cudaVectorScalarType(global.valueType) ?? "float");
  if (global.valueType === "bool") return "u32";
  if (!ir.atomicDeviceGlobals.includes(global.name)) return wgslScalar(global.valueType);
  if (global.valueType === "float" || global.valueType === "double") return "atomic<u32>";
  return `atomic<${wgslScalar(global.valueType)}>`;
}

function usesFloatAtomicAdd(ir: KernelIrModule): boolean {
  return hasAtomicStorageFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicAdd", "atomicAdd_system"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicAdd", "atomicAdd_system"]))));
}

function usesSharedFloatAtomicAdd(ir: KernelIrModule): boolean {
  return hasAtomicSharedFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicAdd", "atomicAdd_system"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicAdd", "atomicAdd_system"]))));
}

function usesFloatAtomicSub(ir: KernelIrModule): boolean {
  return hasAtomicStorageFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicSub"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicSub"]))));
}

function usesSharedFloatAtomicSub(ir: KernelIrModule): boolean {
  return hasAtomicSharedFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicSub"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicSub"]))));
}

function usesFloatAtomicMin(ir: KernelIrModule): boolean {
  return hasAtomicStorageFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicMin", "atomicMin_system"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicMin", "atomicMin_system"]))));
}

function usesSharedFloatAtomicMin(ir: KernelIrModule): boolean {
  return hasAtomicSharedFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicMin", "atomicMin_system"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicMin", "atomicMin_system"]))));
}

function usesFloatAtomicMax(ir: KernelIrModule): boolean {
  return hasAtomicStorageFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicMax", "atomicMax_system", "atomicMaxFloat"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicMax", "atomicMax_system", "atomicMaxFloat"]))));
}

function usesSharedFloatAtomicMax(ir: KernelIrModule): boolean {
  return hasAtomicSharedFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicMax", "atomicMax_system", "atomicMaxFloat"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicMax", "atomicMax_system", "atomicMaxFloat"]))));
}

function hasAtomicSharedFloat(ir: KernelIrModule): boolean {
  return ir.sharedDeclarations.some((shared) => (shared.valueType === "float" || shared.valueType === "double") && ir.atomicShared.includes(shared.name));
}

function hasAtomicStorageFloat(ir: KernelIrModule): boolean {
  return ir.params.some((param) => param.pointer && (param.valueType === "float" || param.valueType === "double") && ir.atomicParams.includes(param.name)) ||
    ir.deviceGlobals.some((global) => (global.valueType === "float" || global.valueType === "double") && ir.atomicDeviceGlobals.includes(global.name));
}

function usesAtomicIncDec(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["atomicInc", "atomicInc_system", "atomicDec", "atomicDec_system"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicInc", "atomicInc_system", "atomicDec", "atomicDec_system"])));
}

function usesCurand(ir: KernelIrModule): boolean {
  const curandCalls = new Set(["curand_init", "curand_uniform", "curand_uniform_double", "curand_normal", "curand_normal_double"]);
  return statementsUseCall(ir.body, curandCalls) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, curandCalls));
}

function usesFrexp(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["frexp", "frexpf"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["frexp", "frexpf"])));
}

function usesSpecialFloatNamedConstants(ir: KernelIrModule): boolean {
  const names = new Set(["INFINITY", "NAN"]);
  return statementsUseIdentifier(ir.body, names) ||
    ir.functions.some((fn) => statementsUseIdentifier(fn.body, names));
}

function usesFp8Intrinsics(ir: KernelIrModule): boolean {
  const names = new Set(["__nv_cvt_fp8_to_halfraw", "__nv_cvt_float_to_fp8"]);
  return statementsUseCall(ir.body, names) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, names));
}

function statementsUseCall(statements: readonly CudaLiteStatement[], names: ReadonlySet<string>): boolean {
  for (const statement of statements) {
    if (statement.kind === "expr" && expressionUsesCall(statement.expression, names)) return true;
    if (statement.kind === "var" && statement.init && expressionUsesCall(statement.init, names)) return true;
    if (statement.kind === "if" && (
      expressionUsesCall(statement.condition, names) ||
      statementsUseCall(statement.consequent, names) ||
      (statement.alternate ? statementsUseCall(statement.alternate, names) : false)
    )) return true;
    if (statement.kind === "for" && (
      (statement.init?.kind === "var" && statement.init.init ? expressionUsesCall(statement.init.init, names) : false) ||
      (statement.init && statement.init.kind !== "var" ? expressionUsesCall(statement.init, names) : false) ||
      (statement.condition ? expressionUsesCall(statement.condition, names) : false) ||
      (statement.update ? expressionUsesCall(statement.update, names) : false) ||
      statementsUseCall(statement.body, names)
    )) return true;
    if ((statement.kind === "while" || statement.kind === "do-while") && (
      expressionUsesCall(statement.condition, names) ||
      statementsUseCall(statement.body, names)
    )) return true;
    if (statement.kind === "block" && statementsUseCall(statement.body, names)) return true;
    if (statement.kind === "return" && statement.value && expressionUsesCall(statement.value, names)) return true;
  }
  return false;
}

function statementsUseIdentifier(statements: readonly CudaLiteStatement[], names: ReadonlySet<string>): boolean {
  for (const statement of statements) {
    if (statement.kind === "expr" && expressionUsesIdentifier(statement.expression, names)) return true;
    if (statement.kind === "var" && statement.init && expressionUsesIdentifier(statement.init, names)) return true;
    if (statement.kind === "if" && (
      expressionUsesIdentifier(statement.condition, names) ||
      statementsUseIdentifier(statement.consequent, names) ||
      (statement.alternate ? statementsUseIdentifier(statement.alternate, names) : false)
    )) return true;
    if (statement.kind === "for" && (
      (statement.init?.kind === "var" && statement.init.init ? expressionUsesIdentifier(statement.init.init, names) : false) ||
      (statement.init && statement.init.kind !== "var" ? expressionUsesIdentifier(statement.init, names) : false) ||
      (statement.condition ? expressionUsesIdentifier(statement.condition, names) : false) ||
      (statement.update ? expressionUsesIdentifier(statement.update, names) : false) ||
      statementsUseIdentifier(statement.body, names)
    )) return true;
    if ((statement.kind === "while" || statement.kind === "do-while") && (
      expressionUsesIdentifier(statement.condition, names) ||
      statementsUseIdentifier(statement.body, names)
    )) return true;
    if (statement.kind === "block" && statementsUseIdentifier(statement.body, names)) return true;
    if (statement.kind === "return" && statement.value && expressionUsesIdentifier(statement.value, names)) return true;
    if (statement.kind === "kernel-launch" && (
      statement.grid.some((expression) => expressionUsesIdentifier(expression, names)) ||
      statement.block.some((expression) => expressionUsesIdentifier(expression, names)) ||
      statement.args.some((expression) => expressionUsesIdentifier(expression, names))
    )) return true;
    if (statement.kind === "dim3" && statement.args.some((expression) => expressionUsesIdentifier(expression, names))) return true;
    if (statement.kind === "asm" && (
      (statement.output ? expressionUsesIdentifier(statement.output, names) : false) ||
      statement.inputs.some((expression) => expressionUsesIdentifier(expression, names))
    )) return true;
  }
  return false;
}

function expressionUsesCall(expression: CudaLiteExpression, names: ReadonlySet<string>): boolean {
  if (expression.kind === "call") {
    const name = expressionName(expression.callee);
    if (name && names.has(name)) return true;
    return expression.args.some((arg) => expressionUsesCall(arg, names)) || expressionUsesCall(expression.callee, names);
  }
  if (expression.kind === "cast") return expressionUsesCall(expression.expression, names);
  if (expression.kind === "member") return expressionUsesCall(expression.object, names);
  if (expression.kind === "index") return expressionUsesCall(expression.target, names) || expressionUsesCall(expression.index, names);
  if (expression.kind === "unary" || expression.kind === "update") return expressionUsesCall(expression.argument, names);
  if (expression.kind === "binary") return expressionUsesCall(expression.left, names) || expressionUsesCall(expression.right, names);
  if (expression.kind === "conditional") {
    return expressionUsesCall(expression.condition, names) ||
      expressionUsesCall(expression.consequent, names) ||
      expressionUsesCall(expression.alternate, names);
  }
  if (expression.kind === "assignment") return expressionUsesCall(expression.left, names) || expressionUsesCall(expression.right, names);
  if (expression.kind === "sequence") return expression.expressions.some((item) => expressionUsesCall(item, names));
  return false;
}

function expressionUsesIdentifier(expression: CudaLiteExpression, names: ReadonlySet<string>): boolean {
  if (expression.kind === "identifier") return names.has(expression.name);
  if (expression.kind === "cast") return expressionUsesIdentifier(expression.expression, names);
  if (expression.kind === "member") return expressionUsesIdentifier(expression.object, names);
  if (expression.kind === "index") return expressionUsesIdentifier(expression.target, names) || expressionUsesIdentifier(expression.index, names);
  if (expression.kind === "unary" || expression.kind === "update") return expressionUsesIdentifier(expression.argument, names);
  if (expression.kind === "binary") return expressionUsesIdentifier(expression.left, names) || expressionUsesIdentifier(expression.right, names);
  if (expression.kind === "conditional") {
    return expressionUsesIdentifier(expression.condition, names) ||
      expressionUsesIdentifier(expression.consequent, names) ||
      expressionUsesIdentifier(expression.alternate, names);
  }
  if (expression.kind === "assignment") return expressionUsesIdentifier(expression.left, names) || expressionUsesIdentifier(expression.right, names);
  if (expression.kind === "sequence") return expression.expressions.some((item) => expressionUsesIdentifier(item, names));
  if (expression.kind === "call") {
    return expressionUsesIdentifier(expression.callee, names) ||
      expression.args.some((arg) => expressionUsesIdentifier(arg, names));
  }
  return false;
}

function emitSpecialFloatConstantHelpers(): readonly string[] {
  return [
    "fn bg_f32_inf() -> f32 {",
    "  var bits: u32 = 0x7f800000u;",
    "  return bitcast<f32>(bits);",
    "}",
    "fn bg_f32_nan() -> f32 {",
    "  var bits: u32 = 0x7fc00000u;",
    "  return bitcast<f32>(bits);",
    "}",
  ];
}

function emitFp8Helpers(): readonly string[] {
  return [
    "fn bg_fp8_to_f32(bits_raw: u32, mode: u32) -> f32 {",
    "  let bits = bits_raw & 0xffu;",
    "  let sign = select(1.0, -1.0, (bits & 0x80u) != 0u);",
    "  if (mode == 1u) {",
    "    let exp_bits = (bits >> 2u) & 0x1fu;",
    "    let mant = bits & 0x03u;",
    "    if (exp_bits == 0u && mant == 0u) { return sign * 0.0; }",
    "    if (exp_bits == 0u) { return sign * f32(mant) * exp2(-16.0); }",
    "    if (exp_bits == 0x1fu) { return select(sign * bitcast<f32>(0x7f800000u), bitcast<f32>(0x7fc00000u), mant != 0u); }",
    "    return sign * (1.0 + f32(mant) / 4.0) * exp2(f32(i32(exp_bits) - 15));",
    "  }",
    "  let exp_bits = (bits >> 3u) & 0x0fu;",
    "  let mant = bits & 0x07u;",
    "  if (exp_bits == 0u && mant == 0u) { return sign * 0.0; }",
    "  if (exp_bits == 0u) { return sign * f32(mant) * exp2(-9.0); }",
    "  if (exp_bits == 0x0fu && mant == 0x07u) { return bitcast<f32>(0x7fc00000u); }",
    "  return sign * (1.0 + f32(mant) / 8.0) * exp2(f32(i32(exp_bits) - 7));",
    "}",
    "",
    "fn bg_round_even(x: f32) -> u32 {",
    "  let base = floor(x);",
    "  let diff = x - base;",
    "  if (diff < 0.5) { return u32(base); }",
    "  if (diff > 0.5) { return u32(base + 1.0); }",
    "  let even = (u32(base) & 1u) == 0u;",
    "  return select(u32(base + 1.0), u32(base), even);",
    "}",
    "",
    "fn bg_f32_to_fp8_format(value: f32, saturate: u32, mantissa_bits: u32, bias: i32, max_exponent: u32, max_mantissa: u32, nan_bits: u32, inf_bits: u32) -> u32 {",
    "  if (value != value) { return nan_bits; }",
    "  let sign_bit = select(0u, 0x80u, bitcast<u32>(value) >> 31u != 0u);",
    "  var magnitude = abs(value);",
    "  if (magnitude == 0.0) { return sign_bit; }",
    "  let mantissa_scale = f32(1u << mantissa_bits);",
    "  let max_finite = (1.0 + f32(max_mantissa) / mantissa_scale) * exp2(f32(i32(max_exponent) - bias));",
    "  if (magnitude > max_finite) {",
    "    if (saturate == 1u) { magnitude = max_finite; }",
    "    else { return sign_bit | inf_bits; }",
    "  }",
    "  let raw_exp = i32(floor(log2(magnitude)));",
    "  var exp_bits = raw_exp + bias;",
    "  if (exp_bits <= 0) {",
    "    let mant = min(max_mantissa, bg_round_even(magnitude / exp2(f32(1 - bias)) * mantissa_scale));",
    "    return sign_bit | mant;",
    "  }",
    "  var mant = bg_round_even((magnitude / exp2(f32(raw_exp)) - 1.0) * mantissa_scale);",
    "  if (mant == (1u << mantissa_bits)) {",
    "    exp_bits = exp_bits + 1;",
    "    mant = 0u;",
    "  }",
    "  if (exp_bits > i32(max_exponent) || (exp_bits == i32(max_exponent) && mant > max_mantissa)) {",
    "    if (saturate != 1u) { return sign_bit | inf_bits; }",
    "    exp_bits = i32(max_exponent);",
    "    mant = max_mantissa;",
    "  }",
    "  return sign_bit | (u32(exp_bits) << mantissa_bits) | mant;",
    "}",
    "",
    "fn bg_f32_to_fp8(value: f32, saturate: u32, mode: u32) -> u32 {",
    "  if (mode == 1u) { return bg_f32_to_fp8_format(value, saturate, 2u, 15, 30u, 3u, 0x7fu, 0x7cu); }",
    "  return bg_f32_to_fp8_format(value, saturate, 3u, 7, 15u, 6u, 0x7fu, 0x7fu);",
    "}",
  ];
}

function functionBodyHasReturn(statements: readonly CudaLiteStatement[]): boolean {
  for (const statement of statements) {
    if (statement.kind === "return") return true;
    if (statement.kind === "if" && (functionBodyHasReturn(statement.consequent) || (statement.alternate && functionBodyHasReturn(statement.alternate)))) {
      return true;
    }
  }
  return false;
}

function zeroValue(type: CudaLiteScalarType): string {
  if (isCudaVectorType(type)) return `${wgslScalar(type)}(${Array.from({ length: cudaVectorLaneCount(type) }, () => zeroValue(cudaVectorScalarType(type) ?? "float")).join(", ")})`;
  if (type === "float") return "0.0";
  if (type === "half") return "f16(0.0)";
  if (type === "bf16") return "0.0";
  if (type === "uint") return "0u";
  if (type === "bool") return "false";
  if (type === "complex64") return "vec2<f32>(0.0, 0.0)";
  if (type === "texture2d" || type === "surface2d" || type === "devicepool" || type === "voidptr") return "0u";
  return "0";
}

function wgslScalar(type: CudaLiteScalarType): string {
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

function wgslUniformScalar(type: CudaLiteScalarType): string {
  if (isCudaVectorType(type)) return wgslScalar(type);
  if (type === "complex64") return "vec2<f32>";
  if (type === "texture2d" || type === "surface2d" || type === "devicepool" || type === "voidptr") return "u32";
  return type === "bool" ? "u32" : wgslScalar(type);
}

function wgslBindingType(type: CudaLiteScalarType): "f16" | "f32" | "i32" | "u32" {
  if (isCudaVectorType(type)) {
    const scalar = cudaVectorScalarType(type);
    return scalar === "int" ? "i32" : scalar === "uint" ? "u32" : scalar === "half" ? "f16" : "f32";
  }
  if (type === "half") return "f16";
  if (type === "bf16") return "f32";
  if (type === "double") return "f32";
  if (type === "int") return "i32";
  if (type === "uint") return "u32";
  if (type === "bool") return "u32";
  if (type === "complex64") return "f32";
  if (type === "texture2d") return "f32";
  if (type === "surface2d") return "f32";
  if (type === "devicepool" || type === "voidptr") return "u32";
  return "f32";
}

function isSurfaceParam(param: CudaLiteParam): boolean {
  return param.valueType === "surface2d";
}

function isTextureParam(param: CudaLiteParam): boolean {
  return param.valueType === "texture2d";
}

function textureBindings(ir: KernelIrModule): readonly { readonly name: string }[] {
  return [
    ...ir.textures.map((texture) => ({ name: texture.name })),
    ...ir.params.filter(isTextureParam).map((param) => ({ name: param.name })),
  ];
}

function isDevicePoolParam(param: CudaLiteParam): boolean {
  return param.pointer && param.valueType === "devicepool";
}

function isEmittedPointerVar(statement: CudaLiteVarDecl, context: EmitContext): boolean {
  return statement.valueType === "voidptr" || context.poolPointerFor(statement.name) !== undefined;
}

function rawPoolHelperName(baseName: string, offsetName: string): string {
  return `bg_raw_pool_alloc_${baseName}_${offsetName}`;
}

function surfaceWidthField(name: string): string {
  return `${name}_width`;
}

function surfaceHeightField(name: string): string {
  return `${name}_height`;
}

function indent(level: number): string {
  return "  ".repeat(level);
}

function featureError(code: string, message: string): CudaLiteCompilerError {
  const span: SourceSpan = { start: 0, end: 0, line: 1, column: 1 };
  return new CudaLiteCompilerError(message, [{ code, severity: "error", message, span }]);
}
