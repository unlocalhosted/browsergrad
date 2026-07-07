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
  readonly canDirectLowerToWgsl: boolean;
  readonly requiresGpuPolyfill: boolean;
  readonly referenceAvailable: boolean;
  readonly unsupported: readonly CudaFeatureRecord[];
}

const BUILTIN_FEATURES: readonly CudaFeatureRecord[] = [
  feature("parse-error", "frontend", "Unsupported CUDA/C++ syntax", "unsupported", false, false, "Parser stopped before semantic IR."),
  feature("semantic-reference-unsupported", "frontend", "Semantic reference unsupported", "unsupported", false, false, "The semantic CPU reference path rejected a kernel outside its migrated coverage."),
  feature("semantic-wgsl-unsupported", "frontend", "Semantic WGSL unsupported", "unsupported", false, true, "The semantic WGSL path rejected a kernel outside its migrated coverage."),
  feature("internal-lowering-invariant", "frontend", "Internal lowering invariant failed", "unsupported", false, false, "Compiler invariant failed after semantic analysis; this indicates a compiler bug, not an unsupported CUDA feature."),
  feature("duplicate-symbol", "frontend", "Duplicate CUDA-lite symbol", "unsupported", false, false, "Symbol table construction requires unique names in the active scope."),
  feature("reserved-symbol", "frontend", "Reserved CUDA-lite symbol", "unsupported", false, false, "User declarations may not shadow compiler-reserved binding names."),
  feature("missing-kernel", "frontend", "Missing CUDA-lite kernel", "unsupported", false, false, "Requested kernel was not found in the parsed module."),
  feature("invalid-call-arity", "frontend", "Invalid CUDA-lite call arity", "unsupported", false, false, "Call does not match a modeled builtin or device-function signature."),
  feature("unsupported-cpp-object-model", "frontend", "Unsupported C++ object model", "unsupported", false, false, "Requires modeled constructors, member calls, and object lifetime before Kernel IR lowering."),
  feature("unsupported-cute-object", "frontend", "Unsupported CuTe object graph", "unsupported", false, false, "Requires modeled tensor/tile object graph before Kernel IR lowering."),
  feature("unsupported-dependent-carrier-param", "frontend", "Unsupported dependent C++ carrier parameter", "unsupported", false, false, "Requires concrete source/context normalization before CUDA-lite lowering."),
  feature("unsupported-wgmma-tma", "subgroup", "Unsupported WGMMA/TMA pipeline object", "unsupported", false, false, "Requires modeled async tensor-core pipeline before reference or WGSL lowering."),
  feature("unsupported-call", "library", "Unsupported CUDA/device call", "unsupported", false, true, "Add semantic builtin or library island."),
  feature("unknown-symbol", "frontend", "Unknown symbol", "unsupported", false, false, "Missing declaration, constant memory, helper, or macro context."),
  feature("unsupported-sizeof", "frontend", "Unsupported sizeof operand", "unsupported", false, false, "sizeof/alignof support modeled types and value expressions; remaining gaps are untyped operands."),
  feature("unsupported-scalar-expression", "frontend", "Unsupported scalar expression", "unsupported", false, false, "Expression must resolve to a modeled scalar value."),
  feature("unsupported-member-target", "frontend", "Unsupported member target", "unsupported", false, false, "Member access target is not modeled by CUDA-lite."),
  feature("unsupported-index-target", "frontend", "Unsupported index target", "unsupported", false, false, "Indexing target is not modeled by CUDA-lite."),
  feature("unsupported-deref-target", "frontend", "Unsupported dereference target", "unsupported", false, false, "Dereference target is not a modeled pointer or address."),
  feature("invalid-address-target", "frontend", "Invalid address target", "unsupported", false, false, "Address-of requires an addressable CUDA-lite expression."),
  feature("invalid-assignment-target", "frontend", "Invalid assignment target", "unsupported", false, false, "Assignment target must be a modeled lvalue."),
  feature("unsupported-return-expression", "frontend", "Unsupported return expression", "unsupported", false, false, "Device-function return expression must resolve to a modeled value."),
  feature("unsupported-sequence-expression", "frontend", "Unsupported value sequence expression", "unsupported", false, false, "Standalone comma expression statements, scalar var initializers, assignment RHS, return values, and for-loop clauses are supported; remaining gaps need temp lowering in arbitrary expression positions."),
  feature("side-effect-expression", "frontend", "Unsupported side-effect expression placement", "unsupported", false, false, "Side-effecting expressions are supported only where lowering can preserve CUDA evaluation order and laziness."),
  feature("unsupported-atomic-f32", "atomic", "Unsupported float atomic", "unsupported", false, true, "Supported float atomics use CAS-backed add/sub/min/max/exch lowering; remaining float forms still need modeling."),
  feature("unsupported-atomic-target", "atomic", "Unsupported atomic target", "unsupported", false, true, "Atomic must target modeled memory."),
  feature("atomic-address-required", "atomic", "Atomic address form required", "unsupported", false, true, "CUDA address semantics not recoverable from value form."),
  feature("unsupported-cache-hint-address", "memory", "Unsupported cache-hint address", "unsupported", false, true, "Cache-hint builtins require a modeled pointer expression."),
  feature("unsupported-device-pointer-param", "memory", "Unsupported device pointer parameter", "unsupported", false, true, "Device pointer arguments must map to modeled storage, shared, constant, or device-global memory."),
  feature("unsupported-device-pool", "memory", "Unsupported DevicePool use", "unsupported", false, true, "DevicePool operations require modeled pool data and offset bindings."),
  feature("dynamic-shared-memory", "memory", "Dynamic shared memory", "native", true, true, "Supported when launch metadata supplies element count."),
  feature("missing-feature-shader-f16", "feature", "Missing shader-f16", "unsupported", false, true, "Requires WebGPU shader-f16."),
  feature("missing-feature-subgroups", "feature", "Missing subgroups", "unsupported", false, true, "Requires WebGPU subgroups or future shared-memory fallback."),
  feature("compatibility-mode-subgroups", "feature", "Subgroups disabled by compatibility mode", "unsupported", false, true, "Compatibility devices cannot run subgroup lowering."),
  feature("divergent-barrier", "safety", "Divergent barrier", "unsupported", false, true, "Cannot lower unsafe barrier control flow."),
  feature("divergent-return-before-barrier", "safety", "Divergent return before barrier", "native", true, true, "Active-lane lowering preserves safe post-return barrier execution when supported."),
  feature("divergent-break-before-barrier", "safety", "Divergent break before barrier", "native", true, true, "Active-lane lowering preserves safe post-break barrier execution when supported."),
  feature("divergent-continue-before-barrier", "safety", "Divergent continue before barrier", "unsupported", false, true, "Continue before a later barrier needs per-iteration active-lane lowering before WGSL can run safely."),
  feature("barrier-expression", "safety", "Barrier used as expression", "unsupported", false, true, "CUDA barriers must be standalone statements."),
  feature("continue-outside-loop", "safety", "Continue outside loop", "unsupported", false, false, "continue is valid only inside a loop."),
  feature("break-outside-loop", "safety", "Break outside loop", "unsupported", false, false, "break is valid only inside a loop."),
  feature("const-pointer-write", "safety", "Const pointer write", "unsupported", false, false, "Rejects invalid memory mutation."),
  feature("unsupported-local-array", "memory", "Local arrays", "unsupported", false, true, "Local array address space not modeled yet."),
  feature("unsupported-local-array-init", "memory", "Local array initializers", "unsupported", false, true, "Braced local array initializer lowering is modeled; non-braced local array initializers remain unsupported."),
  feature("unsupported-local-array-fill", "memory", "Unsupported local array fill", "unsupported", false, true, "fill_* helpers require a modeled fixed local array target."),
  feature("unsupported-local-pointer", "memory", "Local pointer aliases", "unsupported", false, true, "Fixed local-array decay, element aliases, alias copies, pointer arithmetic initializers, straight-line assignments, block/if-merged assignments, conditionals, and +/- rebases are lowered into semantic IR; remaining local pointer forms need modeled alias rules."),
  feature("unsupported-pointer-cast", "memory", "Unsupported pointer cast", "unsupported", false, true, "Pointer cast requires a modeled source and target memory view."),
  feature("unsupported-pointer-assignment", "memory", "Unsupported pointer assignment", "unsupported", false, true, "Pointer assignment must stay within modeled memory and alias rules."),
  feature("unsupported-pointer-conditional", "memory", "Unsupported pointer conditional", "unsupported", false, true, "Same-root fixed local-array pointer conditionals are lowered into semantic IR; remaining pointer conditionals require compatible modeled pointer branches."),
  feature("unsupported-pointer-pointer-comparison", "memory", "Unsupported pointer comparison", "unsupported", false, true, "Same-root fixed local-array pointer comparisons are lowered into semantic IR; remaining pointer comparison requires null checks or addresses in the same modeled address system."),
  feature("unsupported-pointer-difference", "memory", "Unsupported pointer difference", "unsupported", false, true, "Same-root fixed local-array pointer differences are lowered into semantic IR; remaining pointer difference requires modeled pointers with compatible pointee types and address roots."),
  feature("invalid-array-dimension", "memory", "Invalid array dimension", "unsupported", false, false, "Requires positive constant dimensions."),
  feature("invalid-constant-initializer", "memory", "Invalid constant initializer", "unsupported", false, false, "Constant memory initializers must be numeric constants."),
  feature("invalid-device-global-initializer", "memory", "Invalid device-global initializer", "unsupported", false, false, "Device-global initializers must fit modeled storage."),
  feature("missing-constant", "memory", "Missing constant input", "unsupported", false, false, "Compiled kernel requires a constant binding not present in input."),
  feature("missing-scalar", "memory", "Missing scalar input", "unsupported", false, false, "Compiled kernel requires a scalar uniform not present in input."),
  feature("missing-surface", "memory", "Missing surface input", "unsupported", false, false, "Compiled kernel requires a surface binding not present in input."),
  feature("missing-memory-pool", "memory", "Missing memory-pool input", "unsupported", false, false, "Compiled kernel requires a DevicePool binding not present in input."),
  feature("invalid-constant-input", "memory", "Invalid constant input", "unsupported", false, false, "Constant binding has an incompatible host representation."),
  feature("invalid-memory-pool", "memory", "Invalid memory-pool input", "unsupported", false, false, "Memory pool binding has incompatible data or offset storage."),
  feature("unguarded-write", "safety", "Unguarded pointer write", "native", true, true, "Compiler warning; launch/rubric may enforce bounds discipline."),
  feature("unsupported-constant-memory", "memory", "Constant memory", "unsupported", false, true, "Future readonly binding lowering."),
  feature("unsupported-texture", "texture", "Texture/surface access", "unsupported", false, true, "Scalar direct and texture-helper-param tex2D/tex2DLod paths lower through semantic IR; descriptor, vector, atlas, and advanced texture forms use remaining modeled paths or are unsupported."),
  feature("unsupported-surface", "texture", "Unsupported surface access", "unsupported", false, true, "Direct scalar surface reads/writes and scalar surface helper params lower through semantic IR; vector and advanced surface forms use remaining modeled paths or are unsupported."),
  feature("unsupported-cooperative-groups", "subgroup", "Cooperative groups", "unsupported", false, true, "Future group semantic IR."),
  feature("unsupported-subgroup", "subgroup", "Unsupported subgroup primitive", "unsupported", false, true, "Subgroup operation requires modeled WebGPU subgroup or scalar fallback semantics."),
  feature("unsupported-dynamic-parallelism", "runtime", "Dynamic parallelism", "unsupported", false, true, "Future device enqueue + host launch loop."),
  feature("no-device-launch", "runtime", "No device launch found", "unsupported", false, true, "Host dynamic launch planning found no device-side launch to lift."),
  feature("mixed-runtime-operations", "runtime", "Mixed runtime operations", "unsupported", false, true, "Host runtime planning requires a single compatible runtime operation family."),
  feature("no-host-liftable-launch", "runtime", "No host-liftable launch", "unsupported", false, true, "Device-side launches could not be converted into deterministic host orchestration."),
  feature("unknown-child-kernel", "runtime", "Unknown child kernel", "unsupported", false, true, "Device-side launch target is not a known CUDA-lite kernel."),
  feature("child-runtime-unsupported", "runtime", "Child runtime unsupported", "unsupported", false, true, "Child kernel requires unsupported runtime orchestration."),
  feature("child-launch-dimensions-not-host-evaluable", "runtime", "Child launch dimensions not host-evaluable", "unsupported", false, true, "Device-side launch dimensions must be host-evaluable for host lifting."),
  feature("child-arguments-not-host-evaluable", "runtime", "Child launch arguments not host-evaluable", "unsupported", false, true, "Device-side launch arguments must be host-evaluable for host lifting."),
  feature("too-many-parent-invocations", "runtime", "Too many parent invocations", "unsupported", false, true, "Host-lifted dynamic launch expansion exceeded the configured parent invocation cap."),
  feature("unsafe-parent-side-effects", "runtime", "Unsafe parent side effects", "unsupported", false, true, "Host-lifted runtime orchestration cannot replay parent side effects without changing semantics."),
  feature("pool-allocation-not-single-invocation", "runtime", "Pool allocation not single invocation", "unsupported", false, true, "Host-planned DevicePool allocation requires a single host-evaluable parent allocation."),
  feature("pool-allocation-order-sensitive", "runtime", "Pool allocation order sensitive", "unsupported", false, true, "Host-planned DevicePool allocation order cannot be proven deterministic."),
  feature("branch-not-host-evaluable", "runtime", "Branch not host-evaluable", "unsupported", false, true, "Host-lifted orchestration requires host-evaluable control flow."),
  feature("no-peer-copy", "runtime", "No peer copy found", "unsupported", false, true, "Peer-copy planning found no runtime copy operation to lift."),
  feature("no-host-liftable-peer-copy", "runtime", "No host-liftable peer copy", "unsupported", false, true, "Peer-copy runtime operations could not be converted into deterministic host orchestration."),
  feature("parent-not-single-invocation", "runtime", "Parent not single invocation", "unsupported", false, true, "Host-lifted peer-copy orchestration requires a single host-evaluable parent invocation."),
  feature("arguments-not-host-evaluable", "runtime", "Arguments not host-evaluable", "unsupported", false, true, "Host-lifted peer-copy arguments must be host-evaluable."),
  feature("dynamic-child-compile-failed", "runtime", "Dynamic child compile failed", "unsupported", false, true, "Host-lifted child kernel compilation failed; inspect attached diagnostics."),
  feature("dynamic-child-compiler-unavailable", "runtime", "Dynamic child compiler unavailable", "unsupported", false, true, "Host-lifted dynamic launch planning needs a child-kernel compiler."),
  feature("dynamic-child-runtime-unsupported", "runtime", "Dynamic child runtime unsupported", "unsupported", false, true, "Host-lifted dynamic child kernel needs unsupported WebGPU runtime orchestration."),
  feature("host-dynamic-launch-unsupported", "runtime", "Host dynamic launch unsupported", "unsupported", false, true, "Device-side launch cannot be converted into supported host orchestration."),
  feature("host-dynamic-launch-depth-exceeded", "runtime", "Host dynamic launch depth exceeded", "unsupported", false, true, "Host-lifted dynamic launch recursion exceeded the configured cap."),
  feature("host-copy-unsupported", "runtime", "Host copy unsupported", "unsupported", false, true, "Runtime copy/fill operation cannot be converted into supported host WebGPU orchestration."),
  feature("parent-side-effects-with-host-pool-allocation", "runtime", "Parent side effects with host pool allocation", "unsupported", false, true, "Host-planned DevicePool allocation cannot be mixed with parent side effects."),
  feature("unsupported-cuda-runtime", "runtime", "CUDA runtime call", "unsupported", false, true, "Stream/event create/destroy/record/sync/query/wait no-ops, stream device/id/flag/priority/capture queries, thread capture-mode exchange, stream priority range queries, event elapsed-time zero writes, profiler start/stop no-ops, function/device cache/shared-memory config no-ops, function-attribute no-ops, device/thread reset/sync/cache/limits, device selection/status/attributes/limits, L2 cache-policy reset no-op, peer-access status no-ops, cudaMemGetInfo, occupancy queries, cudaGetDeviceCount, runtime/driver version queries, cudaFree lifecycle no-ops, unified-memory advice/prefetch/stream-attach no-ops, last-error no-ops, modeled 1D/2D host/device/default runtime copies, peer copies, symbol copies, and byte-pattern cudaMemset/cudaMemsetAsync/cudaMemset2D/cudaMemset2DAsync fills are modeled; remaining runtime calls need host orchestration or reference execution."),
  feature("unsupported-cuda-runtime-copy-kind", "runtime", "CUDA runtime copy kind", "unsupported", false, true, "Only modeled cudaMemcpyHostToHost/HostToDevice/DeviceToHost/DeviceToDevice/Default 1D/2D runtime copies can be host-lifted."),
  feature("cuda-graph-conditional-host-orchestration", "runtime", "CUDA graph conditional host orchestration", "unsupported", false, true, "Conditional graph setters are host-managed and require explicit WebGPU runtime orchestration."),
  feature("grid-sync-phase-unsupported", "runtime", "Unsupported grid-sync phase plan", "unsupported", false, true, "Grid synchronization cannot be split into safe WebGPU phases for this kernel."),
  feature("launch-workgroup-mismatch", "runtime", "CUDA launch/workgroup mismatch", "unsupported", false, false, "Runtime launch block dimensions must match the compiled WebGPU workgroup size."),
  feature("prepared-scalar-update-topology-changed", "runtime", "Prepared scalar update changes topology", "unsupported", false, true, "Prepared WebGPU kernels can update scalars only when the plan topology stays fixed."),
  feature("prepared-webgpu-kernel-destroyed", "runtime", "Prepared WebGPU kernel destroyed", "unsupported", false, false, "Prepared kernel handle cannot be used after destroy()."),
  feature("reference-runtime-error", "runtime", "Reference runtime error", "unsupported", false, false, "CPU reference execution encountered a runtime input or semantic error."),
  feature("launch-grid-dim-invalid", "runtime", "Invalid CUDA grid dimension", "unsupported", false, false, "Grid dimensions must be positive finite integers."),
  feature("launch-block-dim-invalid", "runtime", "Invalid CUDA block dimension", "unsupported", false, false, "Block dimensions must be positive finite integers compatible with the compiled workgroup."),
  feature("invalid-workgroup-size", "runtime", "Invalid workgroup size", "unsupported", false, false, "Compiler workgroup size must be positive finite integers."),
  feature("unsupported-cufft", "library", "cuFFT library island", "unsupported", false, true, "Future WGSL FFT library lowering."),
  feature("unsupported-curand", "library", "cuRAND library island", "unsupported", false, true, "Future counter RNG library lowering."),
  feature("curand-state-address", "library", "Unsupported cuRAND state address", "unsupported", false, true, "cuRAND state helpers require modeled state storage addresses."),
  feature("unsupported-inline-asm", "subgroup", "Unsupported inline PTX", "unsupported", false, false, "Inline PTX requires modeled instruction semantics before reference or WGSL lowering."),
  feature("invalid-inline-asm-operands", "subgroup", "Invalid inline PTX operands", "unsupported", false, false, "Modeled inline PTX instruction received incompatible operands."),
  feature("unsupported-wmma-fragment-storage", "subgroup", "Unsupported WMMA fragment storage", "unsupported", false, true, "WMMA fragments are modeled only in supported local storage forms."),
  feature("unsupported-wmma-fragment-pointer", "subgroup", "Unsupported WMMA fragment pointer", "unsupported", false, true, "WMMA fragment pointers are not modeled by CUDA-lite."),
  feature("unsupported-wmma-fragment-init", "subgroup", "Unsupported WMMA fragment initializer", "unsupported", false, true, "WMMA fragment initialization must use modeled WMMA builtins."),
  feature("unsupported-wmma-fragment-role", "subgroup", "Unsupported WMMA fragment role", "unsupported", false, true, "WMMA fragment role must match a modeled matrix or accumulator role."),
  feature("missing-wmma-fragment-layout", "subgroup", "Missing WMMA fragment layout", "unsupported", false, true, "WMMA matrix fragments require an explicit row/column layout."),
  feature("unsupported-wmma-fragment-layout", "subgroup", "Unsupported WMMA fragment layout", "unsupported", false, true, "WMMA fragment layout is outside the CUDA-lite supported subset."),
  feature("unsupported-wmma-fragment-value-type", "subgroup", "Unsupported WMMA fragment value type", "unsupported", false, true, "WMMA fragment value type is outside the CUDA-lite supported subset."),
  feature("invalid-wmma-fragment-shape", "subgroup", "Invalid WMMA fragment shape", "unsupported", false, true, "WMMA fragment shape must be a positive constant expression."),
  feature("invalid-wmma-fragment-index", "subgroup", "Invalid WMMA fragment index", "unsupported", false, true, "WMMA fragment lane/index access does not match the fragment shape."),
  feature("unsupported-wmma-fragment-operand", "subgroup", "Unsupported WMMA fragment operand", "unsupported", false, true, "WMMA builtin requires modeled fragment operands."),
  feature("unsupported-wmma-pointer-operand", "subgroup", "Unsupported WMMA pointer operand", "unsupported", false, true, "WMMA load/store requires modeled storage or shared pointer operands."),
  feature("unsupported-wmma-layout-operand", "subgroup", "Unsupported WMMA layout operand", "unsupported", false, true, "WMMA layout operand must be a modeled row/column layout constant."),
  feature("wmma-shape-mismatch", "subgroup", "WMMA shape mismatch", "unsupported", false, true, "WMMA fragment shapes do not match the modeled matrix multiply contract."),
  feature("wmma-value-type-mismatch", "subgroup", "WMMA value type mismatch", "unsupported", false, true, "WMMA fragment value types do not match the modeled matrix multiply contract."),
  feature("unsupported-wmma-fragment-member", "subgroup", "Unsupported WMMA fragment member", "unsupported", false, true, "WMMA fragment member access must use modeled lane/index forms."),
  feature("unsupported-wmma-fragment-use", "subgroup", "Unsupported WMMA fragment use", "unsupported", false, true, "WMMA fragments must be used through modeled WMMA operations."),
  feature("unsupported-vector-argument", "frontend", "Unsupported CUDA vector argument", "unsupported", false, true, "CUDA vector operation received incompatible value types."),
  feature("unsupported-vector-assignment", "frontend", "Unsupported CUDA vector assignment", "unsupported", false, true, "CUDA vector assignment requires a modeled vector lvalue and value."),
  feature("unsupported-vector-member", "frontend", "Unsupported CUDA vector member", "unsupported", false, true, "CUDA vector member must resolve to a modeled lane; xyzw, rgba, and s0-s3 lane aliases are supported."),
  feature("unsupported-frexp-exponent", "math", "Unsupported frexp exponent target", "unsupported", false, true, "frexp exponent must target modeled integer storage; local and direct storage exponent outputs are supported in scalar var initializers and assignment RHS lowering."),
  feature("unsupported-modf-intpart", "math", "Unsupported modf integer-part target", "unsupported", false, true, "modf integer-part output must target modeled float storage; local and direct storage outputs are supported in scalar var initializers, assignment RHS, and statement lowering."),
  feature("unsupported-sincos-output", "math", "Unsupported sincos output target", "unsupported", false, true, "sincos outputs must target modeled float storage; local and direct storage outputs are supported in statement lowering."),
  feature("unsupported-remquo-quotient", "math", "Unsupported remquo quotient target", "unsupported", false, true, "remquo quotient output must target modeled integer storage; local and direct storage outputs are supported in scalar var initializers, assignment RHS, and statement lowering."),
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
    canDirectLowerToWgsl: unsupported.length === 0 && features.every((featureRecord) => featureRecord.gpuRuns),
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
