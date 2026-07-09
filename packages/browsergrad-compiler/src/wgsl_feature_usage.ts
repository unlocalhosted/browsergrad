import {
  kernelIrUsesAtomicOperation,
  kernelIrUsesAtomicOperations,
} from "./kernel_ir_atomic_usage.js";
import {
  kernelIrStatements,
  kernelIrUsesCall,
  kernelIrUsesIdentifier,
} from "./kernel_ir_usage.js";
import { SEMANTIC_CURAND_CALLS } from "./semantic_curand_intrinsics.js";
import {
  SEMANTIC_BFLOAT_CONVERSION_CALLS,
  SEMANTIC_FP8_CALLS,
  SEMANTIC_HALF_CONVERSION_CALLS,
} from "./semantic_math_intrinsics.js";
import type { SemanticAtomicOp } from "./semantic_atomic_intrinsics.js";
import {
  type CudaLiteStatement,
  type CudaLiteDeviceGlobal,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type KernelIrModule,
} from "./types.js";
import { cudaVectorScalarType, isCudaVectorType } from "./vector_types.js";
import { isSubgroupCallName } from "./wgsl_control_analysis.js";
import type { WgslIntViewAtomicEmitKind } from "./wgsl_atomic_helpers.js";
import { wgslScalar } from "./wgsl_storage.js";
import { classifyInlineAsm } from "./features/inline_ptx/model.js";

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
  if (scalar === "float" || scalar === "double" || scalar === "bf16") return "atomic<u32>";
  return `atomic<${wgslScalar(scalar)}>`;
}

export function usesBfloatAtomicAdd(ir: KernelIrModule): boolean {
  return hasAtomicStorageBfloat(ir) && kernelIrUsesAtomicOperation(ir, "add");
}

export function usesSharedBfloatAtomicAdd(ir: KernelIrModule): boolean {
  return hasAtomicSharedBfloat(ir) && kernelIrUsesAtomicOperation(ir, "add");
}

export function usesFloatAtomicAdd(ir: KernelIrModule): boolean {
  return hasAtomicStorageFloat(ir) && kernelIrUsesAtomicOperation(ir, "add");
}

export function usesSharedFloatAtomicAdd(ir: KernelIrModule): boolean {
  return hasAtomicSharedFloat(ir) && kernelIrUsesAtomicOperation(ir, "add");
}

export function usesFloatAtomicSub(ir: KernelIrModule): boolean {
  return hasAtomicStorageFloat(ir) && kernelIrUsesAtomicOperation(ir, "sub");
}

export function usesSharedFloatAtomicSub(ir: KernelIrModule): boolean {
  return hasAtomicSharedFloat(ir) && kernelIrUsesAtomicOperation(ir, "sub");
}

export function usesFloatAtomicMin(ir: KernelIrModule): boolean {
  return hasAtomicStorageFloat(ir) && kernelIrUsesAtomicOperation(ir, "min");
}

export function usesSharedFloatAtomicMin(ir: KernelIrModule): boolean {
  return hasAtomicSharedFloat(ir) && kernelIrUsesAtomicOperation(ir, "min");
}

export function usesFloatAtomicMax(ir: KernelIrModule): boolean {
  return hasAtomicStorageFloat(ir) && kernelIrUsesAtomicOperation(ir, "max");
}

export function usesSharedFloatAtomicMax(ir: KernelIrModule): boolean {
  return hasAtomicSharedFloat(ir) && kernelIrUsesAtomicOperation(ir, "max");
}

function hasAtomicSharedFloat(ir: KernelIrModule): boolean {
  return ir.sharedDeclarations.some((shared) => isFloatAtomicStorageType(shared.valueType) && ir.atomicShared.includes(shared.name));
}

function hasAtomicStorageFloat(ir: KernelIrModule): boolean {
  return ir.params.some((param) => param.pointer && isFloatAtomicStorageType(param.valueType) && ir.atomicParams.includes(param.name)) ||
    ir.deviceGlobals.some((global) => isFloatAtomicStorageType(global.valueType) && ir.atomicDeviceGlobals.includes(global.name));
}

function hasAtomicSharedBfloat(ir: KernelIrModule): boolean {
  return ir.sharedDeclarations.some((shared) => isBfloatAtomicStorageType(shared.valueType) && ir.atomicShared.includes(shared.name));
}

function hasAtomicStorageBfloat(ir: KernelIrModule): boolean {
  return ir.params.some((param) => param.pointer && isBfloatAtomicStorageType(param.valueType) && ir.atomicParams.includes(param.name)) ||
    ir.deviceGlobals.some((global) => isBfloatAtomicStorageType(global.valueType) && ir.atomicDeviceGlobals.includes(global.name));
}

function isBfloatAtomicStorageType(valueType: CudaLiteScalarType): boolean {
  return (cudaVectorScalarType(valueType) ?? valueType) === "bf16";
}

function isFloatAtomicStorageType(valueType: CudaLiteScalarType): boolean {
  const scalar = cudaVectorScalarType(valueType) ?? valueType;
  return scalar === "float" || scalar === "double";
}

export function usesAtomicIncDec(ir: KernelIrModule): boolean {
  return kernelIrUsesAtomicOperations(ir, ["inc", "dec"]);
}

export function usesIntViewAtomicStorage(ir: KernelIrModule): boolean {
  return hasUnsignedAtomicStorage(ir) && intViewAtomicKinds(ir).size > 0;
}

export function usesSharedIntViewAtomics(ir: KernelIrModule): boolean {
  return hasUnsignedAtomicShared(ir) && intViewAtomicKinds(ir).size > 0;
}

export function intViewAtomicKinds(ir: KernelIrModule): ReadonlySet<WgslIntViewAtomicEmitKind> {
  const kinds = new Set<WgslIntViewAtomicEmitKind>();
  const calls: readonly [WgslIntViewAtomicEmitKind, readonly SemanticAtomicOp[]][] = [
    ["Add", ["add"]],
    ["Sub", ["sub"]],
    ["Min", ["min"]],
    ["Max", ["max"]],
    ["And", ["and"]],
    ["Or", ["or"]],
    ["Xor", ["xor"]],
    ["Exchange", ["exchange"]],
    ["CompareExchange", ["cas"]],
  ];
  for (const [kind, ops] of calls) {
    if (kernelIrUsesAtomicOperations(ir, ops)) {
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
  return kernelIrUsesCall(ir, SEMANTIC_CURAND_CALLS);
}

export function usesCuComplexRobustMath(ir: KernelIrModule): boolean {
  const calls = new Set(["cuCabsf", "cuCdivf", "cuCabs", "cuCdiv"]);
  return kernelIrUsesCall(ir, calls);
}

export function usesFrexp(ir: KernelIrModule): boolean {
  return kernelIrUsesCall(ir, new Set(["frexp", "frexpf"]));
}

export function usesModf(ir: KernelIrModule): boolean {
  return kernelIrUsesCall(ir, new Set(["modf", "modff"]));
}

export function usesGammaIntrinsics(ir: KernelIrModule): boolean {
  const names = new Set(["tgamma", "tgammaf", "lgamma", "lgammaf"]);
  return kernelIrUsesCall(ir, names);
}

export function usesInverseDistributionIntrinsics(ir: KernelIrModule): boolean {
  const names = new Set(["erfinv", "erfinvf", "erfcinv", "erfcinvf", "normcdfinv", "normcdfinvf"]);
  return kernelIrUsesCall(ir, names);
}

export function usesRoundingMathIntrinsics(ir: KernelIrModule): boolean {
  const names = new Set([
    "round", "roundf", "rint", "rintf", "nearbyint", "nearbyintf",
    "lrint", "lrintf", "llrint", "llrintf", "lround", "lroundf", "llround", "llroundf",
    "remainder", "remainderf", "remquo", "remquof", "logb", "logbf", "ilogb", "ilogbf",
    "__float2int_rn", "__float2uint_rn", "__half2int_rn", "__half2uint_rn",
    "__bfloat162int_rn", "__bfloat162uint_rn", "__bfloat162ll_rn", "__bfloat162ull_rn", "__bfloat162short_rn", "__bfloat162ushort_rn",
  ]);
  return kernelIrUsesCall(ir, names) ||
    kernelIrStatements(ir).some(statementsUseRoundEvenInlineAsm);
}

function statementsUseRoundEvenInlineAsm(statements: readonly CudaLiteStatement[]): boolean {
  for (const statement of statements) {
    if (statement.kind === "asm") {
      const op = classifyInlineAsm(statement.template);
      if (op?.kind === "convert-f32-to-int" && op.rounding === "rn") return true;
    }
    if (statement.kind === "if" && (
      statementsUseRoundEvenInlineAsm(statement.consequent) ||
      statementsUseRoundEvenInlineAsm(statement.alternate ?? [])
    )) return true;
    if ((statement.kind === "for" || statement.kind === "while" || statement.kind === "do-while" || statement.kind === "block") && statementsUseRoundEvenInlineAsm(statement.body)) return true;
  }
  return false;
}

export function usesNextafterIntrinsics(ir: KernelIrModule): boolean {
  const names = new Set(["nextafter", "nextafterf", "nexttoward", "nexttowardf"]);
  return kernelIrUsesCall(ir, names);
}

export function usesFunnelShiftIntrinsics(ir: KernelIrModule): boolean {
  const names = new Set(["__funnelshift_l", "__funnelshift_lc", "__funnelshift_r", "__funnelshift_rc"]);
  return kernelIrUsesCall(ir, names);
}

export function usesSpecialFloatNamedConstants(ir: KernelIrModule): boolean {
  const names = new Set(["INFINITY", "NAN"]);
  return kernelIrUsesIdentifier(ir, names);
}

export function usesFp8Intrinsics(ir: KernelIrModule): boolean {
  return kernelIrUsesCall(ir, SEMANTIC_FP8_CALLS);
}

export function usesHalfConversionIntrinsics(ir: KernelIrModule): boolean {
  return kernelIrUsesCall(ir, SEMANTIC_HALF_CONVERSION_CALLS);
}

export function usesBfloatConversionIntrinsics(ir: KernelIrModule): boolean {
  return kernelIrUsesCall(ir, SEMANTIC_BFLOAT_CONVERSION_CALLS);
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
