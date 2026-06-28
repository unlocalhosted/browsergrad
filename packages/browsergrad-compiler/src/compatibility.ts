import type { CudaLiteDiagnostic } from "./types.js";

export type CudaCompatibilityFamily =
  | "frontend"
  | "memory"
  | "atomic"
  | "math"
  | "texture"
  | "subgroup"
  | "library"
  | "runtime"
  | "safety"
  | "feature"
  | "unknown";

export type CudaLoweringKind =
  | "native"
  | "gpu-polyfill"
  | "cpu-reference"
  | "unsupported";

export interface CudaFeatureRecord {
  readonly code: string;
  readonly family: CudaCompatibilityFamily;
  readonly label: string;
  readonly lowering: CudaLoweringKind;
  readonly gpuRuns: boolean;
  readonly referenceRuns: boolean;
  readonly notes: string;
}

export interface CudaLoweringPlan {
  readonly features: readonly CudaFeatureRecord[];
  readonly canRunOnGpu: boolean;
  readonly requiresGpuPolyfill: boolean;
  readonly referenceAvailable: boolean;
  readonly unsupported: readonly CudaFeatureRecord[];
}

const BUILTIN_FEATURES: readonly CudaFeatureRecord[] = [
  feature("parse-error", "frontend", "Unsupported CUDA/C++ syntax", "unsupported", false, false, "Parser stopped before semantic IR."),
  feature("unsupported-cpp-object-model", "frontend", "Unsupported C++ object model", "unsupported", false, false, "Requires modeled constructors, member calls, and object lifetime before Kernel IR lowering."),
  feature("unsupported-cute-object", "frontend", "Unsupported CuTe object graph", "unsupported", false, false, "Requires modeled tensor/tile object graph before Kernel IR lowering."),
  feature("unsupported-dependent-carrier-param", "frontend", "Unsupported dependent C++ carrier parameter", "unsupported", false, false, "Requires concrete source/context normalization before CUDA-lite lowering."),
  feature("unsupported-wgmma-tma", "subgroup", "Unsupported WGMMA/TMA pipeline object", "unsupported", false, false, "Requires modeled async tensor-core pipeline before reference or WGSL lowering."),
  feature("unsupported-call", "library", "Unsupported CUDA/device call", "unsupported", false, true, "Add semantic builtin or library island."),
  feature("unknown-symbol", "frontend", "Unknown symbol", "unsupported", false, false, "Missing declaration, constant memory, helper, or macro context."),
  feature("unsupported-atomic-f32", "atomic", "Unsupported float atomic", "unsupported", false, true, "Supported float atomics use CAS-backed add/sub/min/max/exch lowering; remaining float forms still need modeling."),
  feature("unsupported-atomic-target", "atomic", "Unsupported atomic target", "unsupported", false, true, "Atomic must target modeled memory."),
  feature("atomic-address-required", "atomic", "Atomic address form required", "unsupported", false, true, "CUDA address semantics not recoverable from value form."),
  feature("dynamic-shared-memory", "memory", "Dynamic shared memory", "native", true, true, "Supported when launch metadata supplies element count."),
  feature("missing-feature-shader-f16", "feature", "Missing shader-f16", "unsupported", false, true, "Requires WebGPU shader-f16."),
  feature("missing-feature-subgroups", "feature", "Missing subgroups", "unsupported", false, true, "Requires WebGPU subgroups or future shared-memory fallback."),
  feature("compatibility-mode-subgroups", "feature", "Subgroups disabled by compatibility mode", "unsupported", false, true, "Compatibility devices cannot run subgroup lowering."),
  feature("divergent-barrier", "safety", "Divergent barrier", "unsupported", false, true, "Cannot lower unsafe barrier control flow."),
  feature("const-pointer-write", "safety", "Const pointer write", "unsupported", false, false, "Rejects invalid memory mutation."),
  feature("unsupported-local-array", "memory", "Local arrays", "unsupported", false, true, "Local array address space not modeled yet."),
  feature("unsupported-local-array-init", "memory", "Local array initializers", "unsupported", false, true, "Local array initializer lowering not modeled yet."),
  feature("unsupported-local-pointer", "memory", "Local pointer aliases", "unsupported", false, true, "Future pointer alias lowering into modeled memory."),
  feature("invalid-array-dimension", "memory", "Invalid array dimension", "unsupported", false, false, "Requires positive constant dimensions."),
  feature("invalid-constant-initializer", "memory", "Invalid constant initializer", "unsupported", false, false, "Constant memory initializers must be numeric constants."),
  feature("unguarded-write", "safety", "Unguarded pointer write", "native", true, true, "Compiler warning; launch/rubric may enforce bounds discipline."),
  feature("unsupported-constant-memory", "memory", "Constant memory", "unsupported", false, true, "Future readonly binding lowering."),
  feature("unsupported-texture", "texture", "Texture/surface access", "unsupported", false, true, "Future texture binding lowering."),
  feature("unsupported-cooperative-groups", "subgroup", "Cooperative groups", "unsupported", false, true, "Future group semantic IR."),
  feature("unsupported-dynamic-parallelism", "runtime", "Dynamic parallelism", "unsupported", false, true, "Future device enqueue + host launch loop."),
  feature("unsupported-cuda-runtime", "runtime", "CUDA runtime call", "unsupported", false, true, "Future host-side runtime orchestration."),
  feature("unsupported-cuda-runtime-copy-kind", "runtime", "CUDA runtime copy kind", "unsupported", false, true, "Only modeled device-to-device runtime copies can be host-lifted."),
  feature("unsupported-cufft", "library", "cuFFT library island", "unsupported", false, true, "Future WGSL FFT library lowering."),
  feature("unsupported-curand", "library", "cuRAND library island", "unsupported", false, true, "Future counter RNG library lowering."),
  feature("unsupported-inline-asm", "subgroup", "Unsupported inline PTX", "unsupported", false, false, "Inline PTX requires modeled instruction semantics before reference or WGSL lowering."),
  feature("unsupported-f64", "feature", "CUDA f64 compatibility gap", "unsupported", false, false, "True f64 is unavailable in WebGPU; opt into f32 compatibility lowering when acceptable."),
  feature("f64-lowered-to-f32", "feature", "CUDA f64 lowered to f32", "native", true, true, "Compatibility warning: double precision/storage use f32 ABI."),
];

const FEATURE_REGISTRY = new Map(BUILTIN_FEATURES.map((entry) => [entry.code, entry]));

export function getCudaFeatureRegistry(): readonly CudaFeatureRecord[] {
  return BUILTIN_FEATURES;
}

export function describeCudaDiagnostic(
  diagnostic: Pick<CudaLiteDiagnostic, "code" | "message">,
): CudaFeatureRecord {
  if (diagnostic.code === "unsupported-call" || diagnostic.code === "parse-error" || diagnostic.code === "unknown-symbol") {
    return inferFeatureFromDiagnostic(diagnostic);
  }
  const registered = FEATURE_REGISTRY.get(diagnostic.code);
  if (registered) return registered;
  return inferFeatureFromDiagnostic(diagnostic);
}

export function createCudaLoweringPlan(
  diagnostics: readonly Pick<CudaLiteDiagnostic, "code" | "message">[],
): CudaLoweringPlan {
  const features = uniqueByCode(diagnostics.map(describeCudaDiagnostic));
  const unsupported = features.filter((featureRecord) => featureRecord.lowering === "unsupported");
  return {
    features,
    canRunOnGpu: unsupported.length === 0 && features.every((featureRecord) => featureRecord.gpuRuns),
    requiresGpuPolyfill: features.some((featureRecord) => featureRecord.lowering === "gpu-polyfill"),
    referenceAvailable: features.every((featureRecord) => featureRecord.referenceRuns),
    unsupported,
  };
}

export function classifyCudaCompatibilityFamily(
  diagnostic: Pick<CudaLiteDiagnostic, "code" | "message">,
): CudaCompatibilityFamily {
  return describeCudaDiagnostic(diagnostic).family;
}

function feature(
  code: string,
  family: CudaCompatibilityFamily,
  label: string,
  lowering: CudaLoweringKind,
  gpuRuns: boolean,
  referenceRuns: boolean,
  notes: string,
): CudaFeatureRecord {
  return { code, family, label, lowering, gpuRuns, referenceRuns, notes };
}

function inferFeatureFromDiagnostic(
  diagnostic: Pick<CudaLiteDiagnostic, "code" | "message">,
): CudaFeatureRecord {
  const message = diagnostic.message;
  if (/inline PTX|asm|mma\.|wgmma|cp\.async/u.test(message)) {
    return feature(diagnostic.code, "subgroup", "Inline PTX/MMA compatibility gap", "unsupported", false, false, message);
  }
  if (/\bdouble\b|unsupported CUDA-lite type: Real\b/u.test(message)) {
    return feature(diagnostic.code, "feature", "CUDA f64 compatibility gap", "unsupported", false, false, message);
  }
  if (/bfloat|__nv_bfloat/u.test(message)) {
    return feature(diagnostic.code, "feature", "CUDA bf16 compatibility gap", "unsupported", false, false, message);
  }
  if (/fp8|__nv_fp8/u.test(message)) {
    return feature(diagnostic.code, "feature", "CUDA fp8 compatibility gap", "unsupported", false, false, message);
  }
  if (/atomic/u.test(message)) {
    return feature(diagnostic.code, "atomic", "Atomic compatibility gap", "unsupported", false, true, message);
  }
  if (/tex2D|texture|surface|cudaSurfaceObject_t/u.test(message)) {
    return feature(diagnostic.code, "texture", "Texture/surface compatibility gap", "unsupported", false, true, message);
  }
  if (/cufft|cufftComplex/u.test(message)) {
    return feature(diagnostic.code, "library", "cuFFT compatibility gap", "unsupported", false, true, message);
  }
  if (/curand|unsigned long long/u.test(message)) {
    return feature(diagnostic.code, "library", "cuRAND/u64 compatibility gap", "unsupported", false, true, message);
  }
  if (/cooperative|thread_block|tiled_partition|shfl|warp|subgroup/u.test(message)) {
    return feature(diagnostic.code, "subgroup", "Warp/group compatibility gap", "unsupported", false, true, message);
  }
  if (/<<<|cudaDeviceSynchronize|cudaMemcpy|cudaStream|cudaEvent|dynamic parallel/u.test(message)) {
    return feature(diagnostic.code, "runtime", "CUDA runtime compatibility gap", "unsupported", false, true, message);
  }
  if (/__constant__|constant/u.test(message)) {
    return feature(diagnostic.code, "memory", "Constant memory compatibility gap", "unsupported", false, true, message);
  }
  if (/call/u.test(diagnostic.code)) {
    return feature(diagnostic.code, "library", "Unsupported call", "unsupported", false, true, message);
  }
  if (/parse|type|symbol/u.test(diagnostic.code)) {
    return feature(diagnostic.code, "frontend", "Frontend compatibility gap", "unsupported", false, false, message);
  }
  return feature(diagnostic.code, "unknown", "Unknown compatibility gap", "unsupported", false, false, message);
}

function uniqueByCode(features: readonly CudaFeatureRecord[]): readonly CudaFeatureRecord[] {
  const seen = new Set<string>();
  const out: CudaFeatureRecord[] = [];
  for (const featureRecord of features) {
    if (seen.has(featureRecord.code)) continue;
    seen.add(featureRecord.code);
    out.push(featureRecord);
  }
  return out;
}
