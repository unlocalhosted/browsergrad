import type { WgslKernelBindingInput } from "@unlocalhosted/browsergrad-kernels";
import { collectExternalDevicePoolNames } from "./ast_queries.js";
import { CUDA_INTRINSICS_BY_NAME } from "./intrinsics.js";
import { pointerBaseOffsetUniformName } from "./pointer_offsets.js";
import { poolDataName, poolOffsetName } from "./pool_bindings.js";
import { isCudaVectorType } from "./vector_types.js";
import {
  collectLocalArrayDeclarations,
  collectLocalArrays,
  collectLocalNames,
  collectLocalPointerArrayRoots,
  collectLocalPointerHandles,
  collectLocalValueTypes,
  collectMutableScalarParams,
  collectMutableStoragePointerBases,
  collectPointerAliases,
  collectPoolPointers,
  collectRawPoolAllocators,
  isLocalPointerArrayDecl,
  localArrayDeclarationFor,
  pointerAliasDeclarationFor,
  structuredPointerHandleRoots,
  type PointerAlias,
  type PoolPointerAlias,
  type RawPoolAllocator,
} from "./wgsl_ir_analysis.js";
import {
  isDevicePoolParam,
  isExternalConstantBinding,
  isSurfaceParam,
  isTextureParam,
  surfaceHeightField,
  surfaceWidthField,
  textureBindings,
} from "./wgsl_declarations.js";
import { effectiveSubgroupMode, wgslBindingType } from "./wgsl_feature_usage.js";
import { createWgslNameMap } from "./wgsl_names.js";
import type {
  CudaLiteCooperativeGroupDecl,
  CudaLiteDeviceFunction,
  CudaLiteDeviceGlobal,
  CudaLiteExpression,
  CudaLiteFeatureOptions,
  CudaLiteParam,
  CudaLiteScalarType,
  CudaLiteStatement,
  CudaLiteVarDecl,
  KernelIrModule,
  SourceSpan,
} from "./types.js";

export const UNIFORM_PARAMS_NAME = "bg_uniforms";

export interface EmitKernelIrWgslOptions {
  readonly features?: CudaLiteFeatureOptions;
  readonly pointerBaseOffsets?: Readonly<Record<string, number>>;
  readonly f16Mode?: "native" | "f32";
  readonly subgroupMode?: "native" | "scalar";
}

export interface EmitContext {
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
  pointerAliasFor(name: string, span?: SourceSpan): PointerAlias | undefined;
  localPointerHandleFor(name: string): CudaLiteVarDecl | undefined;
  poolPointerFor(name: string): PoolPointerAlias | undefined;
  pointerBaseOffsetFieldFor(name: string): string | undefined;
  readonly rawPoolAllocators: readonly RawPoolAllocator[];
  readonly subgroupMode: "native" | "scalar";
  readonly externalPoolNames: readonly string[];
  readonly mutablePointerBases: readonly string[];
  readonly scalarWarpReduceHelpers: Map<string, ScalarWarpReduceHelper>;
  readonly scalarWarpShuffleHelpers: Map<string, ScalarWarpShuffleHelper>;
  readonly vectorCooperativeReduceHelpers: Map<string, VectorCooperativeReduceHelper>;
  readonly expressionValueTypes: WeakMap<CudaLiteExpression, CudaLiteScalarType | undefined>;
  readonly currentReturnType?: CudaLiteScalarType;
  mutablePointerBaseFor(name: string): string | undefined;
  isLocalName(name: string): boolean;
  localValueTypeFor(name: string): CudaLiteScalarType | undefined;
  localArrayFor(name: string, span?: SourceSpan): CudaLiteVarDecl | undefined;
  localPointerArrayFor(name: string, span?: SourceSpan): CudaLiteVarDecl | undefined;
  localPointerArrayRootFor(name: string, span?: SourceSpan): CudaLiteVarDecl | undefined;
  cooperativeGroupFor(name: string): CudaLiteCooperativeGroupDecl | undefined;
  surfaceWidthField(name: string): string;
  surfaceHeightField(name: string): string;
  nameFor(name: string): string;
}

export interface VectorCooperativeReduceHelper {
  readonly key: string;
  readonly name: string;
  readonly opName: string;
  readonly valueType: "float2" | "float3" | "float4" | "half2" | "bf162" | "int2" | "int3" | "int4" | "uint2" | "uint3" | "uint4";
  readonly tileSize: number;
}

export interface ScalarWarpReduceHelper {
  readonly key: string;
  readonly name: string;
  readonly op: "sum" | "max" | "min";
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly tileSize: number;
  readonly partitioned?: boolean;
}

export interface ScalarWarpShuffleHelper {
  readonly key: string;
  readonly name: string;
  readonly op: "sync" | "down" | "up" | "xor";
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly tileSize: number;
}

export function createEmitContext(ir: KernelIrModule, options: EmitKernelIrWgslOptions = {}): EmitContext {
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
  const localArrayDeclarations = collectLocalArrayDeclarations(ir.body);
  const localPointerArrayRoots = collectLocalPointerArrayRoots(ir.body);
  const wgslNames = createWgslNameMap(collectWgslDeclaredNames(ir, localNames, externalPoolNames), CUDA_INTRINSICS_BY_NAME.keys());
  const scalarWarpReduceHelpers = new Map<string, ScalarWarpReduceHelper>();
  const scalarWarpShuffleHelpers = new Map<string, ScalarWarpShuffleHelper>();
  const vectorCooperativeReduceHelpers = new Map<string, VectorCooperativeReduceHelper>();
  const expressionValueTypes = new WeakMap<CudaLiteExpression, CudaLiteScalarType | undefined>();
  const paramsBinding = uniformScalars.length > 0 ? bindings.length : undefined;
  if (paramsBinding !== undefined) {
    bindings.push({
      kind: "uniform",
      name: UNIFORM_PARAMS_NAME,
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
      if (binding === undefined) throw new Error(`missing WGSL binding for '${name}'`);
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
    pointerAliasFor(name, span) {
      return pointerAliasDeclarationFor(pointerAliases, name, span);
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
    subgroupMode: effectiveSubgroupMode(ir, options),
    externalPoolNames,
    mutablePointerBases,
    scalarWarpReduceHelpers,
    scalarWarpShuffleHelpers,
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
    localArrayFor(name, span) {
      return localArrayDeclarationFor(localArrayDeclarations, name, span);
    },
    localPointerArrayFor(name, span) {
      const declaration = localArrayDeclarationFor(localArrayDeclarations, name, span);
      return declaration?.pointer && declaration.dimensions.length > 0 ? declaration : undefined;
    },
    localPointerArrayRootFor(name, span) {
      const declaration = localArrayDeclarationFor(localArrayDeclarations, name, span);
      if (!declaration?.pointer) return undefined;
      return localPointerArrayRoots.get(name);
    },
    cooperativeGroupFor(name) {
      return cooperativeGroups.get(name);
    },
    surfaceWidthField(name) {
      return surfaceWidthField(name);
    },
    surfaceHeightField(name) {
      return surfaceHeightField(name);
    },
    nameFor(name) {
      return wgslNames.get(name) ?? name;
    },
  };
}

export function resolveDeviceFunctionForCall(
  ir: KernelIrModule,
  name: string,
  argCount?: number,
): CudaLiteDeviceFunction | undefined {
  const overloads = ir.functions.filter((fn) => fn.name === name);
  if (overloads.length === 0) return undefined;
  if (argCount === undefined) return overloads[0];
  return overloads.find((fn) => fn.params.length === argCount) ?? overloads[0];
}

export function deviceFunctionLinkName(fn: CudaLiteDeviceFunction, ir: KernelIrModule): string {
  const overloads = ir.functions.filter((candidate) => candidate.name === fn.name);
  if (overloads.length <= 1) return fn.name;
  const index = overloads.indexOf(fn);
  return `${fn.name}__bg_overload_${index < 0 ? 0 : index}`;
}

function collectWgslDeclaredNames(
  ir: KernelIrModule,
  localNames: ReadonlySet<string>,
  externalPoolNames: readonly string[],
): readonly string[] {
  const structuredPointerRoots = structuredPointerHandleRoots(ir);
  return [
    ir.name,
    "bg_active_lane",
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
      ...collectLocalPointerArrayGeneratedNames(fn.body),
    ]),
    ...localNames,
    ...collectLocalPointerHandleGeneratedNames(ir.body, structuredPointerRoots),
    ...collectLocalPointerArrayGeneratedNames(ir.body),
    ...externalPoolNames,
    ...externalPoolNames.flatMap((name) => [poolDataName(name), poolOffsetName(name)]),
  ];
}

function collectLocalPointerHandleGeneratedNames(
  statements: readonly CudaLiteStatement[],
  structuredPointerRoots: ReadonlySet<string> = new Set(),
): readonly string[] {
  return [...collectLocalPointerHandles(statements, undefined, structuredPointerRoots).keys()].flatMap((name) => [`${name}_buffer`, `${name}_base`]);
}

function collectLocalPointerArrayGeneratedNames(statements: readonly CudaLiteStatement[]): readonly string[] {
  return [...collectLocalArrays(statements).values()]
    .filter(isLocalPointerArrayDecl)
    .flatMap((statement) => [`${statement.name}_buffer`, `${statement.name}_base`]);
}

function collectCooperativeGroupGeneratedNames(params: readonly CudaLiteParam[]): readonly string[] {
  return params
    .filter((param) => param.cooperativeGroupKind === "thread")
    .flatMap((param) => [`${param.name}_tile_size`, `${param.name}_tile_size_arg`]);
}

export function collectCooperativeGroups(statements: readonly CudaLiteStatement[]): ReadonlyMap<string, CudaLiteCooperativeGroupDecl> {
  const groups = new Map<string, CudaLiteCooperativeGroupDecl>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "cooperative-group") {
        const parent = item.partitionParent ? groups.get(item.partitionParent) : undefined;
        const tileSize = item.tileSize ?? parent?.tileSize;
        groups.set(item.name, {
          ...item,
          ...(tileSize === undefined ? {} : { tileSize }),
        });
      }
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
