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
  feature("unsupported-call", "library", "Unsupported CUDA/device call", "unsupported", false, true, "Add semantic builtin or library island."),
  feature("unknown-symbol", "frontend", "Unknown symbol", "unsupported", false, false, "Missing declaration, constant memory, helper, or macro context."),
  feature("unsupported-atomic-f32", "atomic", "Unsupported float atomic", "unsupported", false, true, "Float atomicAdd/atomicExch have WGSL polyfills; other float atomics still need lowering."),
  feature("unsupported-atomic-target", "atomic", "Unsupported atomic target", "unsupported", false, true, "Atomic must target modeled memory."),
  feature("atomic-address-required", "atomic", "Atomic address form required", "unsupported", false, true, "CUDA address semantics not recoverable from value form."),
  feature("dynamic-shared-memory", "memory", "Dynamic shared memory", "native", true, true, "Supported when launch metadata supplies element count."),
  feature("missing-feature-shader-f16", "feature", "Missing shader-f16", "unsupported", false, true, "Requires WebGPU shader-f16."),
  feature("missing-feature-subgroups", "feature", "Missing subgroups", "unsupported", false, true, "Requires WebGPU subgroups or future shared-memory fallback."),
  feature("compatibility-mode-subgroups", "feature", "Subgroups disabled by compatibility mode", "unsupported", false, true, "Compatibility devices cannot run subgroup lowering."),
  feature("divergent-barrier", "safety", "Divergent barrier", "unsupported", false, true, "Cannot lower unsafe barrier control flow."),
  feature("const-pointer-write", "safety", "Const pointer write", "unsupported", false, false, "Rejects invalid memory mutation."),
  feature("unsupported-local-array", "memory", "Local arrays", "unsupported", false, true, "Local array address space not modeled yet."),
  feature("unsupported-local-pointer", "memory", "Local pointer aliases", "unsupported", false, true, "Future pointer alias lowering into modeled memory."),
  feature("invalid-array-dimension", "memory", "Invalid array dimension", "unsupported", false, false, "Requires positive constant dimensions."),
  feature("unguarded-write", "safety", "Unguarded pointer write", "native", true, true, "Compiler warning; launch/rubric may enforce bounds discipline."),
  feature("unsupported-constant-memory", "memory", "Constant memory", "unsupported", false, true, "Future readonly binding lowering."),
  feature("unsupported-texture", "texture", "Texture/surface access", "unsupported", false, true, "Future texture binding lowering."),
  feature("unsupported-cooperative-groups", "subgroup", "Cooperative groups", "unsupported", false, true, "Future group semantic IR."),
  feature("unsupported-dynamic-parallelism", "runtime", "Dynamic parallelism", "unsupported", false, true, "Future device enqueue + host launch loop."),
  feature("unsupported-cufft", "library", "cuFFT library island", "unsupported", false, true, "Future WGSL FFT library lowering."),
  feature("unsupported-curand", "library", "cuRAND library island", "unsupported", false, true, "Future counter RNG library lowering."),
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
