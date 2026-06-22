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
  type CudaLiteDiagnostic,
  type CudaLiteDim3Decl,
  type CudaLiteExpression,
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
  type CudaPeerCopyBlocker,
  type CudaPeerCopyBlockerCode,
  type CudaPeerCopyOperation,
  type CudaPeerCopyPlan,
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
  type CudaWebGpuExecutionBlocker,
  type CudaWebGpuExecutionBlockerKind,
  type CudaWebGpuExecutionPlan,
  type CudaWebGpuExecutionPlanKind,
  type CudaWebGpuExecutionPlanOptions,
} from "./webgpu_orchestration.js";
export { emitKernelIrWgsl, type EmitKernelIrWgslOptions, type KernelIrWgslOutput } from "./wgsl.js";
export {
  compileCudaLiteKernel,
  prepareCompiledKernelWebGpu,
  runCompiledKernelReference,
  runCompiledKernelWebGpu,
  type PreparedCompiledKernelWebGpu,
  type PreparedCompiledKernelWebGpuRunOptions,
} from "./runner.js";
