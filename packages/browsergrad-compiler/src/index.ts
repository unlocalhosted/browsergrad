export {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CompileCudaLiteOptions,
  type CudaLiteAnalysis,
  type CudaLiteAnalyzeOptions,
  type CudaLiteAssignmentExpression,
  type CudaLiteBinaryExpression,
  type CudaLiteCastExpression,
  type CudaLiteCallExpression,
  type CudaLiteConditionalExpression,
  type CudaLiteContinueStatement,
  type CudaLiteCooperativeGroupDecl,
  type CudaLiteCooperativeGroupKind,
  type CudaLiteDeviceFunction,
  type CudaLiteDeviceGlobal,
  type CudaLiteDiagnostic,
  type CudaLiteDim3Decl,
  type CudaLiteExpression,
  type CudaLiteFeatureName,
  type CudaLiteFeatureOptions,
  type CudaLiteForStatement,
  type CudaLiteGlobalConstant,
  type CudaLiteIdentifier,
  type CudaLiteIfStatement,
  type CudaLiteIndexExpression,
  type CudaLiteKernel,
  type CudaLiteKernelLaunchStatement,
  type CudaLiteMemberExpression,
  type CudaLiteModule,
  type CudaLiteNumberLiteral,
  type CudaLiteParam,
  type CudaLiteReturnStatement,
  type CudaLiteScalarType,
  type CudaLiteStatement,
  type CudaLiteStringLiteral,
  type CudaLiteTexture2D,
  type CudaLiteUnaryExpression,
  type CudaLiteUpdateExpression,
  type CudaLiteVarDecl,
  type DiagnosticSeverity,
  type KernelIrModule,
  type KernelLaunch,
  type KernelMemoryAccess,
  type KernelThreadTrace,
  type ReferenceKernelResult,
  type SourceSpan,
} from "./types.js";

export { parseCudaLite } from "./parser.js";
export type { ParsedCudaLiteModule } from "./parser.js";
export type { AnalyzedCudaLiteModule } from "./analyzer.js";
export type {
  CompilerPhase,
  Parsed,
  Analyzed,
  TypedSemantic,
  CanonicalIr,
  RuntimeLoweredIr,
  VerifiedIr,
  TypeCheckedIr,
  WgslLegalizedIr,
} from "./compiler_phases.js";
export type {
  SemanticFunctionId,
  SemanticMemoryId,
  SemanticSymbolId,
} from "./semantic_ids.js";
export {
  createSemanticEnvironment,
  type SemanticEnvironment,
} from "./semantic_environment.js";
export {
  createCudaLiteSemanticModel,
  lowerSemanticModelToKernelIr,
  type CudaLiteSemanticFunction,
  type CudaLiteSemanticLaunchableEntry,
  type CudaLiteSemanticModel,
  type CudaLiteSemanticSymbol,
  type SemanticAddressSpace,
  type SemanticDeviceLaunch,
  type SemanticExpression,
  type SemanticKernelIrModule,
  type SemanticKernelIrOperation,
  type SemanticMemoryRef,
} from "./semantic_ir.js";
export {
  assertValidSemanticKernelIr,
  verifySemanticKernelIr,
  type SemanticIrVerificationIssue,
  type VerifiedSemanticKernelIr,
} from "./semantic_ir_verifier.js";
export {
  assertTypeCheckedSemanticKernelIr,
  checkSemanticKernelIrTypes,
  type SemanticTypeIssue,
  type TypeCheckedSemanticKernelIr,
} from "./semantic_type_check.js";
export {
  assertWgslLegalizedSemanticKernelIr,
  checkSemanticKernelIrWgslLegalization,
  type WgslLegalizationIssue,
  type WgslLegalizedSemanticKernelIr,
} from "./wgsl_legalization.js";
export {
  createTypedWgslExpression,
  emitTypedWgslBinary,
  type TypedWgslExpression,
  type WgslBinaryOperator,
  type WgslExpressionType,
} from "./typed_wgsl_expression.js";
export { lowerSemanticCudaRuntime } from "./semantic_runtime_lowering.js";
export { analyzeCudaLite, lowerCudaLiteToKernelIr } from "./analyzer.js";
export {
  classifyCudaCompatibilityFamily,
  createCudaLoweringPlan,
  describeCudaDiagnostic,
  getCudaFeatureRegistry,
  type CudaCompatibilityFamily,
  type CudaFeatureRecord,
  type CudaLoweringKind,
  type CudaLoweringPlan,
} from "./compatibility.js";
export { formatCudaLiteDiagnostics } from "./diagnostics.js";
export {
  compileCudaLiteOptionsFromKernelFeatures,
  cudaLiteFeatureOptionsFromKernelFeatures,
  type CudaLiteKernelFeatureSource,
} from "./features.js";
export {
  createCudaLaunchValidationDiagnostics,
  validateCudaKernelLaunch,
} from "./launch.js";
export {
  createCudaHostDynamicLaunchPlan,
  type CudaHostDynamicLaunch,
  type CudaHostDynamicLaunchBlocker,
  type CudaHostDynamicLaunchBlockerCode,
  type CudaHostDynamicLaunchPlan,
  type CudaHostDynamicLaunchPlanOptions,
} from "./dynamic_launch.js";
export {
  createCudaPeerCopyPlan,
  createCudaRuntimeCopyPlan,
  type CudaPeerCopyBlocker,
  type CudaPeerCopyBlockerCode,
  type CudaPeerCopyOperation,
  type CudaPeerCopyPlan,
  type CudaRuntimeCopyBlocker,
  type CudaRuntimeCopyBlockerCode,
  type CudaRuntimeCopyOperation,
  type CudaRuntimeCopyPlan,
} from "./peer_copy.js";
export {
  createCudaGridSyncPhasePlan,
  createCudaRuntimePlan,
  type CudaGridSyncPhasePlan,
  type CudaRuntimeOperation,
  type CudaRuntimeOperationKind,
  type CudaRuntimePlan,
} from "./runtime_plan.js";
export {
  createCudaWebGpuExecutionPlan,
  normalizeCudaWebGpuReadbackNames,
  normalizeCudaWebGpuReadback,
  summarizeCudaWebGpuExecutionPlan,
  type CudaWebGpuExecutionBlocker,
  type CudaWebGpuExecutionBlockerKind,
  type CudaWebGpuExecutionMode,
  type CudaWebGpuExecutionPlan,
  type CudaWebGpuExecutionPlanKind,
  type CudaWebGpuExecutionPlanOptions,
  type CudaWebGpuExecutionStatus,
} from "./webgpu_orchestration.js";
export {
  canEmitSemanticKernelIrWgsl,
  emitSemanticKernelIrWgsl,
  semanticKernelIrWgslPreflightBlocker,
  semanticKernelIrWgslPreflightFailure,
  type EmitSemanticKernelIrWgslOptions,
  type SemanticKernelIrWgslOutput,
  type SemanticKernelIrWgslPreflightFailure,
} from "./semantic_wgsl.js";
export {
  createCudaLiteCompileCacheKey,
  createCudaLiteCompilerCache,
  type CudaLiteCompilerCache,
  type CudaLiteCompilerCacheOptions,
  type CudaLiteCompilerCacheStats,
} from "./cache.js";
export {
  compileCudaLiteKernelForWebGpu,
  compileCudaLiteKernel,
  cudaLiteWebGpuCompileOptions,
  prepareCompiledKernelWebGpu,
  canRunCompiledKernelSemanticReference,
  runCompiledKernelReference,
  runCompiledKernelSemanticReference,
  runCompiledKernelWebGpu,
  type CompiledKernelWebGpuExecutionOptions,
  type PreparedCompiledKernelWebGpu,
  type PreparedCompiledKernelWebGpuRunOptions,
  type PrepareCompiledKernelWebGpuOptions,
} from "./runner.js";
