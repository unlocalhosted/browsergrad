import { poolDataName, poolOffsetName } from "./pool_bindings.js";
import {
  emitConstantArrayType,
  emitInitializedConstant,
  emitSharedType,
  isDevicePoolParam,
  isExternalConstantBinding,
  isSurfaceParam,
  isTextureParam,
  textureBindings,
} from "./wgsl_declarations.js";
import {
  deviceGlobalStorageElementType,
  irUsesSubgroups,
  storageElementType,
  usesAtomicIncDec,
  usesCurand,
  usesFloatAtomicAdd,
  usesFloatAtomicMax,
  usesFloatAtomicMin,
  usesFloatAtomicSub,
  usesFp8Intrinsics,
  usesFrexp,
  usesSharedFloatAtomicAdd,
  usesSharedFloatAtomicMax,
  usesSharedFloatAtomicMin,
  usesSharedFloatAtomicSub,
  usesSpecialFloatNamedConstants,
  wgslUniformScalar,
} from "./wgsl_feature_usage.js";
import { emitIntegerAtomicLoopHelpers, emitFloatAtomicAddHelper, emitFloatAtomicMaxHelper, emitFloatAtomicMinHelper, emitFloatAtomicSubHelper } from "./wgsl_atomic_helpers.js";
import { UNIFORM_PARAMS_NAME, type EmitContext } from "./wgsl_context.js";
import { wgslScalar } from "./wgsl_storage.js";
import {
  emitCurandHelpers,
  emitFp8Helpers,
  emitFrexpHelpers,
  emitPoolHelper,
  emitRawPoolHelper,
  emitSpecialFloatConstantHelpers,
} from "./wgsl_support_helpers.js";
import {
  emitCubeTextureAtlasHelpers,
  emitSurfaceDispatchHelpers,
  emitSurfaceHelper,
  emitTextureHelper,
  type WgslTextureSurfaceEmitContext,
} from "./wgsl_texture_surface.js";
import type { CudaLiteExpression } from "./types.js";

export interface WgslModulePreludeOptions {
  readonly f16Mode: "native" | "f32";
  readonly subgroupMode: "native" | "scalar";
  readonly textureSurface: WgslTextureSurfaceEmitContext;
  emitExpression(expression: CudaLiteExpression): string;
}

export function emitKernelModulePrelude(context: EmitContext, options: WgslModulePreludeOptions): string[] {
  const ir = context.ir;
  const lines: string[] = [];
  const textures = textureBindings(ir);

  if (options.f16Mode === "native" && ir.requiredFeatures.includes("shader-f16")) lines.push("enable f16;");
  if (options.subgroupMode === "native" && (ir.requiredFeatures.includes("subgroups") || irUsesSubgroups(ir))) lines.push("enable subgroups;");
  if (lines.length > 0) lines.push("");
  lines.push(`// BrowserGrad CUDA-lite kernel: ${ir.name}`);

  for (const param of ir.params.filter((param) => param.pointer && !isDevicePoolParam(param) && !isSurfaceParam(param) && !isTextureParam(param))) {
    const access = param.constant ? "read" : "read_write";
    const element = storageElementType(param, ir);
    lines.push(`@group(0) @binding(${context.bindingFor(param.name)}) var<storage, ${access}> ${context.nameFor(param.name)}: array<${element}>;`);
  }
  for (const global of ir.deviceGlobals) {
    const element = deviceGlobalStorageElementType(global, ir);
    lines.push(`@group(0) @binding(${context.bindingFor(global.name)}) var<storage, read_write> ${context.nameFor(global.name)}: array<${element}>;`);
  }
  for (const surface of ir.params.filter(isSurfaceParam)) {
    lines.push(`@group(0) @binding(${context.bindingFor(surface.name)}) var<storage, read_write> ${context.nameFor(surface.name)}: array<f32>;`);
  }
  for (const pool of ir.params.filter(isDevicePoolParam)) emitPoolBindings(lines, context, pool.name);
  for (const poolName of context.externalPoolNames) emitPoolBindings(lines, context, poolName);

  for (const constant of ir.constants.filter(isExternalConstantBinding)) {
    lines.push(`@group(0) @binding(${context.bindingFor(constant.name)}) var<storage, read> ${context.nameFor(constant.name)}: ${emitConstantArrayType(constant)};`);
  }
  for (const constant of ir.constants.filter((constant) => constant.init !== undefined)) {
    lines.push(emitInitializedConstant(constant, context, options.emitExpression));
  }

  for (const texture of textures) {
    lines.push(`@group(0) @binding(${context.bindingFor(texture.name)}) var ${context.nameFor(texture.name)}: texture_2d<f32>;`);
  }

  if (context.uniformScalars.length > 0) {
    lines.push("struct Params {");
    for (const scalar of context.uniformScalars) {
      const align = scalar.valueType === "half" ? "@align(4) " : "";
      lines.push(`  ${align}${context.nameFor(scalar.name)}: ${wgslUniformScalar(scalar.valueType)},`);
    }
    lines.push("};");
    lines.push(`@group(0) @binding(${context.paramsBinding}) var<uniform> ${UNIFORM_PARAMS_NAME}: Params;`);
  }

  for (const shared of ir.sharedDeclarations) {
    lines.push(`var<workgroup> ${context.nameFor(shared.name)}: ${emitSharedType(shared, ir)};`);
  }

  appendKernelModuleSupportHelpers(lines, context, options.textureSurface);
  return lines;
}

export function emitKernelEntryPoint(
  context: EmitContext,
  mainBodyLines: readonly string[],
  emitUniformScalarRead: (name: string, context: EmitContext) => string,
): string[] {
  const lines = [
    "",
    `@compute @workgroup_size(${context.ir.workgroupSize.join(", ")})`,
    "fn main(",
    "  @builtin(global_invocation_id) global_id: vec3<u32>,",
    "  @builtin(local_invocation_id) local_id: vec3<u32>,",
    "  @builtin(workgroup_id) workgroup_id: vec3<u32>,",
    "  @builtin(num_workgroups) num_workgroups: vec3<u32>",
    ") {",
  ];
  for (const name of context.mutablePointerBases) {
    const baseField = context.pointerBaseOffsetFieldFor(name);
    const baseName = context.mutablePointerBaseFor(name);
    if (baseName) lines.push(`  var ${baseName}: u32 = ${baseField ? `${UNIFORM_PARAMS_NAME}.${context.nameFor(baseField)}` : "0u"};`);
  }
  for (const param of context.mutableScalarParams) {
    lines.push(`  var ${context.nameFor(param.name)}: ${wgslScalar(param.valueType)} = ${emitUniformScalarRead(param.name, context)};`);
  }
  lines.push(...mainBodyLines, "}");
  return lines;
}

function emitPoolBindings(lines: string[], context: EmitContext, poolName: string): void {
  const dataName = poolDataName(poolName);
  const offsetName = poolOffsetName(poolName);
  lines.push(`@group(0) @binding(${context.bindingFor(dataName)}) var<storage, read_write> ${context.nameFor(dataName)}: array<u32>;`);
  lines.push(`@group(0) @binding(${context.bindingFor(offsetName)}) var<storage, read_write> ${context.nameFor(offsetName)}: atomic<u32>;`);
}

function appendKernelModuleSupportHelpers(
  lines: string[],
  context: EmitContext,
  textureSurface: WgslTextureSurfaceEmitContext,
): void {
  const ir = context.ir;
  const textures = textureBindings(ir);
  if (textures.length > 0) {
    lines.push("", ...emitCubeTextureAtlasHelpers(), "");
    for (const texture of textures) lines.push(...emitTextureHelper(texture.name, textureSurface));
  }
  for (const surface of ir.params.filter(isSurfaceParam)) lines.push("", ...emitSurfaceHelper(surface.name, textureSurface));
  if (ir.params.some(isSurfaceParam)) lines.push("", ...emitSurfaceDispatchHelpers(textureSurface));
  for (const pool of ir.params.filter(isDevicePoolParam)) lines.push("", ...emitPoolHelper(pool.name, context));
  for (const poolName of context.externalPoolNames) lines.push("", ...emitPoolHelper(poolName, context));
  for (const allocator of context.rawPoolAllocators) lines.push("", ...emitRawPoolHelper(allocator));
  if (usesFloatAtomicAdd(ir)) lines.push("", ...emitFloatAtomicAddHelper("storage"));
  if (usesSharedFloatAtomicAdd(ir)) lines.push("", ...emitFloatAtomicAddHelper("workgroup"));
  if (usesFloatAtomicSub(ir)) lines.push("", ...emitFloatAtomicSubHelper("storage"));
  if (usesSharedFloatAtomicSub(ir)) lines.push("", ...emitFloatAtomicSubHelper("workgroup"));
  if (usesFloatAtomicMin(ir)) lines.push("", ...emitFloatAtomicMinHelper("storage"));
  if (usesSharedFloatAtomicMin(ir)) lines.push("", ...emitFloatAtomicMinHelper("workgroup"));
  if (usesFloatAtomicMax(ir)) lines.push("", ...emitFloatAtomicMaxHelper("storage"));
  if (usesSharedFloatAtomicMax(ir)) lines.push("", ...emitFloatAtomicMaxHelper("workgroup"));
  if (usesAtomicIncDec(ir)) lines.push("", ...emitIntegerAtomicLoopHelpers());
  if (usesCurand(ir)) lines.push("", ...emitCurandHelpers());
  if (usesFrexp(ir)) lines.push("", ...emitFrexpHelpers());
  if (usesSpecialFloatNamedConstants(ir)) lines.push("", ...emitSpecialFloatConstantHelpers());
  if (usesFp8Intrinsics(ir)) lines.push("", ...emitFp8Helpers());
}
