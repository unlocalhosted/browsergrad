import { statementsUseCall, statementsUseIdentifier } from "./ir_usage.js";
import {
  type CudaLiteDeviceGlobal,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type KernelIrModule,
} from "./types.js";
import { cudaVectorScalarType, isCudaVectorType } from "./vector_types.js";
import { isSubgroupCallName } from "./wgsl_control_analysis.js";
import type { WgslIntViewAtomicEmitKind } from "./wgsl_atomic_helpers.js";
import { wgslScalar } from "./wgsl_storage.js";

export function effectiveF16Mode(
  ir: KernelIrModule,
  options: { readonly f16Mode?: "native" | "f32" },
): "native" | "f32" {
  if (options.f16Mode !== undefined) return options.f16Mode;
  return !ir.requiredFeatures.includes("shader-f16") && irUsesHalf(ir) ? "f32" : "native";
}

export function effectiveSubgroupMode(
  ir: KernelIrModule,
  options: { readonly subgroupMode?: "native" | "scalar" },
): "native" | "scalar" {
  if (options.subgroupMode !== undefined) return options.subgroupMode;
  return !ir.requiredFeatures.includes("subgroups") && irUsesSubgroups(ir) ? "scalar" : "native";
}

export function rewriteF16WgslToF32(wgsl: string): string {
  return wgsl.replace(/\bf16\b/gu, "f32");
}

export function rewriteF16BindingsToF32<T extends { readonly kind: string; readonly valueType?: string }>(
  bindings: readonly T[],
): readonly T[] {
  return bindings.map((binding) => {
    if (binding.kind !== "storage" || binding.valueType !== "f16") return binding;
    return { ...binding, valueType: "f32" } as T;
  });
}

function irUsesHalf(value: unknown): boolean {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value === "half" || value === "half2";
  if (typeof value !== "object") return false;
  if (Array.isArray(value)) return value.some(irUsesHalf);
  for (const [key, child] of Object.entries(value)) {
    if ((key === "span" || key === "diagnostics") && typeof child === "object") continue;
    if (irUsesHalf(child)) return true;
  }
  return false;
}

export function irUsesSubgroups(value: unknown): boolean {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return isSubgroupCallName(value);
  if (typeof value !== "object") return false;
  if (Array.isArray(value)) return value.some(irUsesSubgroups);
  for (const [key, child] of Object.entries(value)) {
    if ((key === "span" || key === "diagnostics") && typeof child === "object") continue;
    if (irUsesSubgroups(child)) return true;
  }
  return false;
}

export function storageElementType(param: CudaLiteParam, ir: KernelIrModule): string {
  if (ir.atomicParams.includes(param.name)) return atomicStorageElementType(param.valueType);
  if (isCudaVectorType(param.valueType)) return wgslScalar(cudaVectorScalarType(param.valueType) ?? "float");
  if (param.valueType === "bool") return "u32";
  return wgslScalar(param.valueType);
}

export function deviceGlobalStorageElementType(global: CudaLiteDeviceGlobal, ir: KernelIrModule): string {
  if (ir.atomicDeviceGlobals.includes(global.name)) return atomicStorageElementType(global.valueType);
  if (isCudaVectorType(global.valueType)) return wgslScalar(cudaVectorScalarType(global.valueType) ?? "float");
  if (global.valueType === "bool") return "u32";
  return wgslScalar(global.valueType);
}

function atomicStorageElementType(valueType: CudaLiteScalarType): string {
  const scalar = cudaVectorScalarType(valueType) ?? valueType;
  if (scalar === "float" || scalar === "double") return "atomic<u32>";
  return `atomic<${wgslScalar(scalar)}>`;
}

export function usesFloatAtomicAdd(ir: KernelIrModule): boolean {
  return hasAtomicStorageFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicAdd", "atomicAdd_system"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicAdd", "atomicAdd_system"]))));
}

export function usesSharedFloatAtomicAdd(ir: KernelIrModule): boolean {
  return hasAtomicSharedFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicAdd", "atomicAdd_system"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicAdd", "atomicAdd_system"]))));
}

export function usesFloatAtomicSub(ir: KernelIrModule): boolean {
  return hasAtomicStorageFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicSub", "atomicSub_system"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicSub", "atomicSub_system"]))));
}

export function usesSharedFloatAtomicSub(ir: KernelIrModule): boolean {
  return hasAtomicSharedFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicSub", "atomicSub_system"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicSub", "atomicSub_system"]))));
}

export function usesFloatAtomicMin(ir: KernelIrModule): boolean {
  return hasAtomicStorageFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicMin", "atomicMin_system"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicMin", "atomicMin_system"]))));
}

export function usesSharedFloatAtomicMin(ir: KernelIrModule): boolean {
  return hasAtomicSharedFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicMin", "atomicMin_system"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicMin", "atomicMin_system"]))));
}

export function usesFloatAtomicMax(ir: KernelIrModule): boolean {
  return hasAtomicStorageFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicMax", "atomicMax_system", "atomicMaxFloat"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicMax", "atomicMax_system", "atomicMaxFloat"]))));
}

export function usesSharedFloatAtomicMax(ir: KernelIrModule): boolean {
  return hasAtomicSharedFloat(ir) &&
    (statementsUseCall(ir.body, new Set(["atomicMax", "atomicMax_system", "atomicMaxFloat"])) ||
      ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicMax", "atomicMax_system", "atomicMaxFloat"]))));
}

function hasAtomicSharedFloat(ir: KernelIrModule): boolean {
  return ir.sharedDeclarations.some((shared) => isFloatAtomicStorageType(shared.valueType) && ir.atomicShared.includes(shared.name));
}

function hasAtomicStorageFloat(ir: KernelIrModule): boolean {
  return ir.params.some((param) => param.pointer && isFloatAtomicStorageType(param.valueType) && ir.atomicParams.includes(param.name)) ||
    ir.deviceGlobals.some((global) => isFloatAtomicStorageType(global.valueType) && ir.atomicDeviceGlobals.includes(global.name));
}

function isFloatAtomicStorageType(valueType: CudaLiteScalarType): boolean {
  const scalar = cudaVectorScalarType(valueType) ?? valueType;
  return scalar === "float" || scalar === "double";
}

export function usesAtomicIncDec(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["atomicInc", "atomicInc_system", "atomicDec", "atomicDec_system"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["atomicInc", "atomicInc_system", "atomicDec", "atomicDec_system"])));
}

export function usesIntViewAtomicStorage(ir: KernelIrModule): boolean {
  return hasUnsignedAtomicStorage(ir) && intViewAtomicKinds(ir).size > 0;
}

export function usesSharedIntViewAtomics(ir: KernelIrModule): boolean {
  return hasUnsignedAtomicShared(ir) && intViewAtomicKinds(ir).size > 0;
}

export function intViewAtomicKinds(ir: KernelIrModule): ReadonlySet<WgslIntViewAtomicEmitKind> {
  const kinds = new Set<WgslIntViewAtomicEmitKind>();
  const calls: readonly [WgslIntViewAtomicEmitKind, readonly string[]][] = [
    ["Add", ["atomicAdd", "atomicAdd_system"]],
    ["Sub", ["atomicSub", "atomicSub_system"]],
    ["Min", ["atomicMin", "atomicMin_system"]],
    ["Max", ["atomicMax", "atomicMax_system"]],
    ["And", ["atomicAnd", "atomicAnd_system"]],
    ["Or", ["atomicOr", "atomicOr_system"]],
    ["Xor", ["atomicXor", "atomicXor_system"]],
    ["Exchange", ["atomicExch", "atomicExch_system"]],
    ["CompareExchange", ["atomicCAS", "atomicCAS_system"]],
  ];
  for (const [kind, names] of calls) {
    const callNames = new Set(names);
    if (statementsUseCall(ir.body, callNames) || ir.functions.some((fn) => statementsUseCall(fn.body, callNames))) {
      kinds.add(kind);
    }
  }
  return kinds;
}

function hasUnsignedAtomicStorage(ir: KernelIrModule): boolean {
  return ir.params.some((param) => param.pointer && atomicStorageScalarType(param.valueType) === "u32" && ir.atomicParams.includes(param.name)) ||
    ir.deviceGlobals.some((global) => atomicStorageScalarType(global.valueType) === "u32" && ir.atomicDeviceGlobals.includes(global.name));
}

function hasUnsignedAtomicShared(ir: KernelIrModule): boolean {
  return ir.sharedDeclarations.some((shared) => atomicStorageScalarType(shared.valueType) === "u32" && ir.atomicShared.includes(shared.name));
}

function atomicStorageScalarType(valueType: CudaLiteScalarType): "i32" | "u32" {
  return (cudaVectorScalarType(valueType) ?? valueType) === "int" ? "i32" : "u32";
}

export function usesCurand(ir: KernelIrModule): boolean {
  const curandCalls = new Set([
    "curand_init",
    "curand",
    "curand_uniform",
    "curand_uniform4",
    "curand_uniform_double",
    "curand_normal",
    "curand_normal2",
    "curand_normal4",
    "curand_normal_double",
    "curand_log_normal",
    "curand_log_normal2",
    "curand_log_normal4",
    "curand_log_normal_double",
    "curand_poisson",
    "curand_poisson4",
    "skipahead",
  ]);
  return statementsUseCall(ir.body, curandCalls) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, curandCalls));
}

export function usesCuComplexRobustMath(ir: KernelIrModule): boolean {
  const calls = new Set(["cuCabsf", "cuCdivf", "cuCabs", "cuCdiv"]);
  return statementsUseCall(ir.body, calls) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, calls));
}

export function usesFrexp(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["frexp", "frexpf"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["frexp", "frexpf"])));
}

export function usesModf(ir: KernelIrModule): boolean {
  return statementsUseCall(ir.body, new Set(["modf", "modff"])) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, new Set(["modf", "modff"])));
}

export function usesGammaIntrinsics(ir: KernelIrModule): boolean {
  const names = new Set(["tgamma", "tgammaf", "lgamma", "lgammaf"]);
  return statementsUseCall(ir.body, names) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, names));
}

export function usesInverseDistributionIntrinsics(ir: KernelIrModule): boolean {
  const names = new Set(["erfinv", "erfinvf", "erfcinv", "erfcinvf", "normcdfinv", "normcdfinvf"]);
  return statementsUseCall(ir.body, names) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, names));
}

export function usesRoundingMathIntrinsics(ir: KernelIrModule): boolean {
  const names = new Set([
    "round", "roundf", "rint", "rintf", "nearbyint", "nearbyintf",
    "lrint", "lrintf", "llrint", "llrintf", "lround", "lroundf", "llround", "llroundf",
    "remainder", "remainderf", "remquo", "remquof", "logb", "logbf", "ilogb", "ilogbf",
    "__float2int_rn", "__float2uint_rn", "__half2int_rn", "__half2uint_rn",
    "__bfloat162int_rn", "__bfloat162uint_rn", "__bfloat162short_rn", "__bfloat162ushort_rn",
  ]);
  return statementsUseCall(ir.body, names) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, names));
}

export function usesNextafterIntrinsics(ir: KernelIrModule): boolean {
  const names = new Set(["nextafter", "nextafterf", "nexttoward", "nexttowardf"]);
  return statementsUseCall(ir.body, names) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, names));
}

export function usesFunnelShiftIntrinsics(ir: KernelIrModule): boolean {
  const names = new Set(["__funnelshift_l", "__funnelshift_lc", "__funnelshift_r", "__funnelshift_rc"]);
  return statementsUseCall(ir.body, names) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, names));
}

export function usesSpecialFloatNamedConstants(ir: KernelIrModule): boolean {
  const names = new Set(["INFINITY", "NAN"]);
  return statementsUseIdentifier(ir.body, names) ||
    ir.functions.some((fn) => statementsUseIdentifier(fn.body, names));
}

export function usesFp8Intrinsics(ir: KernelIrModule): boolean {
  const names = new Set(["__nv_cvt_fp8_to_halfraw", "__nv_cvt_float_to_fp8"]);
  return statementsUseCall(ir.body, names) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, names));
}

export function usesHalfConversionIntrinsics(ir: KernelIrModule): boolean {
  const names = new Set([
    "__float2half", "__float2half_rn", "__float2half_rz", "__float2half_ru", "__float2half_rd",
    "__int2half_rn", "__int2half_rz", "__int2half_ru", "__int2half_rd",
    "__uint2half_rn", "__uint2half_rz", "__uint2half_ru", "__uint2half_rd",
    "__short2half_rn", "__short2half_rz", "__short2half_ru", "__short2half_rd",
    "__ushort2half_rn", "__ushort2half_rz", "__ushort2half_ru", "__ushort2half_rd",
  ]);
  return statementsUseCall(ir.body, names) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, names));
}

export function usesBfloatConversionIntrinsics(ir: KernelIrModule): boolean {
  const names = new Set([
    "__float2bfloat16", "__float2bfloat16_rn", "__float2bfloat16_rz", "__float2bfloat16_ru", "__float2bfloat16_rd", "__double2bfloat16",
    "__int2bfloat16_rn", "__int2bfloat16_rz", "__int2bfloat16_ru", "__int2bfloat16_rd",
    "__uint2bfloat16_rn", "__uint2bfloat16_rz", "__uint2bfloat16_ru", "__uint2bfloat16_rd",
    "__short2bfloat16_rn", "__short2bfloat16_rz", "__short2bfloat16_ru", "__short2bfloat16_rd",
    "__ushort2bfloat16_rn", "__ushort2bfloat16_rz", "__ushort2bfloat16_ru", "__ushort2bfloat16_rd",
    "__bfloat162short_rn", "__bfloat162short_rz", "__bfloat162short_ru", "__bfloat162short_rd",
    "__bfloat162ushort_rn", "__bfloat162ushort_rz", "__bfloat162ushort_ru", "__bfloat162ushort_rd",
    "__bfloat162char_rz", "__bfloat162uchar_rz",
  ]);
  return statementsUseCall(ir.body, names) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, names));
}

export function wgslUniformScalar(type: CudaLiteScalarType): string {
  if (isCudaVectorType(type)) return wgslScalar(type);
  if (type === "complex64") return "vec2<f32>";
  if (type === "texture2d" || type === "surface2d" || type === "devicepool" || type === "voidptr") return "u32";
  return type === "bool" ? "u32" : wgslScalar(type);
}

export function wgslBindingType(type: CudaLiteScalarType): "f16" | "f32" | "i32" | "u32" {
  if (isCudaVectorType(type)) {
    const scalar = cudaVectorScalarType(type);
    return scalar === "int" ? "i32" : scalar === "uint" ? "u32" : scalar === "half" ? "f16" : "f32";
  }
  if (type === "half") return "f16";
  if (type === "bf16") return "f32";
  if (type === "double") return "f32";
  if (type === "int") return "i32";
  if (type === "uint" || type === "uchar") return "u32";
  if (type === "bool") return "u32";
  if (type === "complex64") return "f32";
  if (type === "texture2d") return "f32";
  if (type === "surface2d") return "f32";
  if (type === "devicepool" || type === "voidptr") return "u32";
  return "f32";
}
