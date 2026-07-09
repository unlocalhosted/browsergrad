import {
  CudaLiteCompilerError,
  type CudaLiteAnalysis,
  type CudaLiteAnalyzeOptions,
  type CudaLiteAssignmentExpression,
  type CudaLiteCooperativeGroupDecl,
  type CudaLiteCooperativeGroupKind,
  type CudaLiteDeviceFunction,
  type CudaLiteDeviceGlobal,
  type CudaLiteDiagnostic,
  type CudaLiteExpression,
  type CudaLiteGlobalConstant,
  type CudaLiteKernel,
  type CudaLiteMatrixTileMetadata,
  type CudaLiteModule,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type CudaLiteStatement,
  type CudaLiteUpdateExpression,
  type CudaLiteTexture2D,
  type CudaLiteVarDecl,
  type KernelIrModule,
  type SourceSpan,
} from "./types.js";
import { collectKernelLaunchCallees, walkCudaLiteExpressions } from "./ast_queries.js";
import { CUDA_CACHE_HINT_LOADS, CUDA_CACHE_HINT_STORES, CUDA_INTRINSICS, CUDA_INTRINSICS_BY_NAME } from "./intrinsics.js";
import { isCudaRuntimeCopyCall, isCudaRuntimeSymbolCopyCall } from "./cuda_runtime_copies.js";
import { isHostManagedRuntimeNoopCall } from "./cuda_runtime_noops.js";
import {
  type WmmaBuiltin,
  isMatrixTileByteValueType,
  isMatrixTileFloatValueType,
  matrixTileElementCount,
  matrixTileReference,
  normalizeMatrixTileLayout,
  normalizeMatrixTileRole,
  normalizeMatrixTileValueType,
  resolveMatrixTileSpec,
  wmmaBuiltinName,
} from "./matrix_tiles.js";
import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";
import { classifyInlineAsm, inlineAsmSupportedList, type InlineAsmOp } from "./features/inline_ptx/model.js";
import {
  inlineAsmInputValueContracts,
  inlineAsmInputValueTypeMatches,
  inlineAsmOperandShapeDiagnostic,
  inlineAsmOutputValueContract,
  inlineAsmOutputValueTypeMatches,
} from "./features/inline_ptx/validation.js";
import { collectCudaAllowedTrapCallSpanStarts } from "./trap_preconditions.js";
import { sizeofCudaType } from "./type_layout.js";
import {
  CUDA_VECTOR_TYPES,
  CUDA_VECTOR_CONSTRUCTORS,
  cudaVectorConstructorType,
  cudaVectorLaneCount,
  cudaVectorScalarType,
  cudaVectorSwizzleIndices,
  cudaVectorSwizzleType,
  isCudaVectorType,
  type CudaLiteVectorType,
} from "./vector_types.js";

const DEFAULT_WORKGROUP_SIZE: readonly [number, number, number] = [256, 1, 1];
const BUILTIN_VECTORS = new Set(["threadIdx", "blockIdx", "blockDim", "gridDim"]);
const BUILTIN_CALLS = new Map<string, readonly [min: number, max: number]>([
  ...CUDA_INTRINSICS.map((intrinsic) => [intrinsic.name, intrinsic.arity] as const),
  ["__vibmax_s32", [3, 3]],
  ["__vibmin_s32", [3, 3]],
  ["__vibmax_u32", [3, 3]],
  ["__vibmin_u32", [3, 3]],
  ["__vibmax_s16x2", [4, 4]],
  ["__vibmin_s16x2", [4, 4]],
  ["__vibmax_u16x2", [4, 4]],
  ["__vibmin_u16x2", [4, 4]],
  ["__syncthreads", [0, 0]],
  ["__syncthreads_count", [1, 1]],
  ["__syncthreads_and", [1, 1]],
  ["__syncthreads_or", [1, 1]],
  ["__syncwarp", [0, 1]],
  ["__threadfence", [0, 0]],
  ["__threadfence_block", [0, 0]],
  ["__threadfence_system", [0, 0]],
  ["__nanosleep", [1, 1]],
  ["__prof_trigger", [1, 1]],
  ["__trap", [0, 0]],
  ["cudaCtxResetPersistingL2Cache", [0, 0]],
  ["cudaDeviceReset", [0, 0]],
  ["cudaDeviceGetAttribute", [3, 3]],
  ["cudaDeviceGetLimit", [2, 2]],
  ["cudaDeviceSetLimit", [2, 2]],
  ["cudaThreadGetLimit", [2, 2]],
  ["cudaThreadSetLimit", [2, 2]],
  ["cudaDeviceCanAccessPeer", [3, 3]],
  ["cudaDeviceEnablePeerAccess", [2, 2]],
  ["cudaDeviceDisablePeerAccess", [1, 1]],
  ["cudaGetDeviceFlags", [1, 1]],
  ["cudaSetDeviceFlags", [1, 1]],
  ["cudaMemGetInfo", [2, 2]],
  ["cudaOccupancyMaxActiveBlocksPerMultiprocessor", [4, 4]],
  ["cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", [5, 5]],
  ["cudaOccupancyMaxPotentialBlockSize", [3, 5]],
  ["cudaOccupancyMaxPotentialBlockSizeWithFlags", [6, 6]],
  ["cudaOccupancyAvailableDynamicSMemPerBlock", [4, 4]],
  ["cudaSetDevice", [1, 1]],
  ["cudaGetDevice", [1, 1]],
  ["cudaGetDeviceCount", [1, 1]],
  ["cudaGetLastError", [0, 0]],
  ["cudaPeekAtLastError", [0, 0]],
  ["cudaProfilerStart", [0, 0]],
  ["cudaProfilerStop", [0, 0]],
  ["cudaRuntimeGetVersion", [1, 1]],
  ["cudaDriverGetVersion", [1, 1]],
  ["cudaFuncSetAttribute", [3, 3]],
  ["cudaFuncSetCacheConfig", [2, 2]],
  ["cudaFuncSetSharedMemConfig", [2, 2]],
  ["cudaDeviceGetCacheConfig", [1, 1]],
  ["cudaDeviceSetCacheConfig", [1, 1]],
  ["cudaDeviceGetSharedMemConfig", [1, 1]],
  ["cudaDeviceSetSharedMemConfig", [1, 1]],
  ["cudaDeviceGetStreamPriorityRange", [2, 2]],
  ["cudaThreadGetCacheConfig", [1, 1]],
  ["cudaThreadSetCacheConfig", [1, 1]],
  ["cudaThreadExit", [0, 0]],
  ["cudaThreadSynchronize", [0, 0]],
  ["cudaThreadExchangeStreamCaptureMode", [1, 1]],
  ["cudaFree", [1, 1]],
  ["cudaFreeAsync", [2, 2]],
  ["cudaMemAdvise", [4, 4]],
  ["cudaMemPrefetchAsync", [3, 4]],
  ["cudaStreamAttachMemAsync", [2, 4]],
  ["cudaStreamBeginCapture", [2, 2]],
  ["cudaStreamEndCapture", [2, 2]],
  ["cudaStreamUpdateCaptureDependencies", [4, 4]],
  ["cudaGraphCreate", [2, 2]],
  ["cudaGraphInstantiate", [5, 5]],
  ["cudaGraphInstantiateWithFlags", [3, 3]],
  ["cudaGraphUpload", [2, 2]],
  ["cudaGraphExecUpdate", [4, 4]],
  ["cudaGraphDestroy", [1, 1]],
  ["cudaGraphExecDestroy", [1, 1]],
  ["cudaMemset", [3, 3]],
  ["cudaMemsetAsync", [4, 4]],
  ["cudaMemset2D", [5, 5]],
  ["cudaMemset2DAsync", [6, 6]],
  ["__shfl", [2, 3]],
  ["__shfl_down", [2, 3]],
  ["__shfl_up", [2, 3]],
  ["__shfl_xor", [2, 3]],
  ["__shfl_sync", [3, 4]],
  ["__shfl_down_sync", [3, 4]],
  ["__shfl_up_sync", [3, 4]],
  ["__shfl_xor_sync", [3, 4]],
  ["__activemask", [0, 0]],
  ["__isGlobal", [1, 1]],
  ["__isShared", [1, 1]],
  ["__isConstant", [1, 1]],
  ["__isLocal", [1, 1]],
  ["__any", [1, 1]],
  ["__all", [1, 1]],
  ["__ballot", [1, 1]],
  ["__any_sync", [2, 2]],
  ["__all_sync", [2, 2]],
  ["__ballot_sync", [2, 2]],
  ["__match_any_sync", [2, 2]],
  ["__reduce_add_sync", [2, 2]],
  ["__reduce_min_sync", [2, 2]],
  ["__reduce_max_sync", [2, 2]],
  ["__reduce_and_sync", [2, 2]],
  ["__reduce_or_sync", [2, 2]],
  ["__reduce_xor_sync", [2, 2]],
  ["warpReduceSum", [1, 2]],
  ["warpReduceMax", [1, 2]],
  ["warpReduceMin", [1, 2]],
  ["warp_reduce_sum", [1, 2]],
  ["warp_reduce_max", [1, 2]],
  ["warp_reduce_min", [1, 2]],
  ["warp_reduce_sum_f32", [1, 2]],
  ["warp_reduce_max_f32", [1, 2]],
  ["warp_reduce_sum_f16", [1, 2]],
  ["warp_reduce_sum_f16_f16", [1, 2]],
  ["warp_reduce_sum_f16_f32", [1, 2]],
  ["warp_reduce_sum_i8_i32", [1, 2]],
  ["warp_reduce_sum_i32_i32", [1, 2]],
  ["blockReduce", [1, 3]],
  ["min", [2, 2]],
  ["max", [2, 2]],
  ["frexp", [2, 2]],
  ["frexpf", [2, 2]],
  ["modf", [2, 2]],
  ["modff", [2, 2]],
  ["remquo", [3, 3]],
  ["remquof", [3, 3]],
  ["sincos", [3, 3]],
  ["sincosf", [3, 3]],
  ["__sincosf", [3, 3]],
  ["sincospi", [3, 3]],
  ["sincospif", [3, 3]],
  ["nan", [1, 1]],
  ["nanf", [1, 1]],
  ["__builtin_nan", [1, 1]],
  ["__builtin_nanf", [1, 1]],
  ["div_ceil", [2, 2]],
  ["fill_1D_regs", [2, 2]],
  ["fill_2D_regs", [2, 2]],
  ["fill_3D_regs", [2, 2]],
  ["wmma::fill_fragment", [2, 2]],
  ["nvcuda::wmma::fill_fragment", [2, 2]],
  ["wmma::load_matrix_sync", [3, 4]],
  ["nvcuda::wmma::load_matrix_sync", [3, 4]],
  ["wmma::mma_sync", [4, 4]],
  ["nvcuda::wmma::mma_sync", [4, 4]],
  ["wmma::store_matrix_sync", [4, 4]],
  ["nvcuda::wmma::store_matrix_sync", [4, 4]],
  ["bg_subgroup_add", [1, 1]],
  ["atomicAdd", [2, 2]],
  ["atomicAdd_system", [2, 2]],
  ["atomicSub", [2, 2]],
  ["atomicSub_system", [2, 2]],
  ["atomicMin", [2, 2]],
  ["atomicMin_system", [2, 2]],
  ["atomicMax", [2, 2]],
  ["atomicMax_system", [2, 2]],
  ["atomicMaxFloat", [2, 2]],
  ["atomicAnd", [2, 2]],
  ["atomicAnd_system", [2, 2]],
  ["atomicOr", [2, 2]],
  ["atomicOr_system", [2, 2]],
  ["atomicXor", [2, 2]],
  ["atomicXor_system", [2, 2]],
  ["atomicInc", [2, 2]],
  ["atomicInc_system", [2, 2]],
  ["atomicDec", [2, 2]],
  ["atomicDec_system", [2, 2]],
  ["atomicExch", [2, 2]],
  ["atomicExch_system", [2, 2]],
  ["atomicCAS", [3, 3]],
  ["atomicCAS_system", [3, 3]],
  ["tex1D", [2, 2]],
  ["tex2D", [3, 3]],
  ["tex2DLod", [4, 4]],
  ["tex1Dfetch", [2, 2]],
  ["tex2DLayered", [4, 4]],
  ["tex3D", [4, 4]],
  ["texCubemap", [4, 4]],
  ["surf1Dread", [2, 4]],
  ["surf2Dread", [3, 5]],
  ["surf2DLayeredread", [4, 6]],
  ["surf3Dread", [4, 6]],
  ["surf2Dwrite", [4, 5]],
  ["surf3Dwrite", [5, 6]],
  ["surf1Dwrite", [3, 4]],
  ["surf2DLayeredwrite", [5, 6]],
  ["sizeof", [1, 1]],
  ["alignof", [1, 1]],
  ["vec_at", [2, 2]],
  ["deviceAllocate", [2, 4]],
  ["streamOrderedAllocate", [2, 4]],
  ["curand_init", [4, 4]],
  ["curand", [1, 1]],
  ["curand_uniform", [1, 1]],
  ["curand_uniform4", [1, 1]],
  ["curand_uniform_double", [1, 1]],
  ["curand_normal", [1, 1]],
  ["curand_normal2", [1, 1]],
  ["curand_normal4", [1, 1]],
  ["curand_normal_double", [1, 1]],
  ["curand_log_normal", [3, 3]],
  ["curand_log_normal2", [3, 3]],
  ["curand_log_normal4", [3, 3]],
  ["curand_log_normal_double", [3, 3]],
  ["curand_poisson", [2, 2]],
  ["curand_poisson4", [2, 2]],
  ["skipahead", [2, 2]],
  ["make_cuComplex", [2, 2]],
  ["make_cuFloatComplex", [2, 2]],
  ["make_cuDoubleComplex", [2, 2]],
  ["cuCrealf", [1, 1]],
  ["cuCimagf", [1, 1]],
  ["cuCabsf", [1, 1]],
  ["cuConjf", [1, 1]],
  ["cuCaddf", [2, 2]],
  ["cuCsubf", [2, 2]],
  ["cuCmulf", [2, 2]],
  ["cuCdivf", [2, 2]],
  ["cuCfmaf", [3, 3]],
  ["cuCreal", [1, 1]],
  ["cuCimag", [1, 1]],
  ["cuCabs", [1, 1]],
  ["cuConj", [1, 1]],
  ["cuCadd", [2, 2]],
  ["cuCsub", [2, 2]],
  ["cuCmul", [2, 2]],
  ["cuCdiv", [2, 2]],
  ["cuCfma", [3, 3]],
  ["cudaDeviceSynchronize", [0, 0]],
  ["cudaStreamCreate", [1, 1]],
  ["cudaStreamCreateWithFlags", [2, 2]],
  ["cudaStreamCreateWithPriority", [3, 3]],
  ["cudaStreamDestroy", [1, 1]],
  ["cudaStreamGetDevice", [2, 2]],
  ["cudaStreamGetFlags", [2, 2]],
  ["cudaStreamGetId", [2, 2]],
  ["cudaStreamGetPriority", [2, 2]],
  ["cudaStreamIsCapturing", [2, 2]],
  ["cudaStreamGetCaptureInfo", [2, 7]],
  ["cudaStreamGetCaptureInfo_v2", [2, 7]],
  ["cudaStreamQuery", [1, 1]],
  ["cudaStreamSynchronize", [1, 1]],
  ["cudaStreamWaitEvent", [2, 3]],
  ["cudaEventCreate", [1, 1]],
  ["cudaEventCreateWithFlags", [2, 2]],
  ["cudaEventDestroy", [1, 1]],
  ["cudaEventQuery", [1, 1]],
  ["cudaEventElapsedTime", [3, 3]],
  ["cudaEventRecord", [1, 2]],
  ["cudaEventRecordWithFlags", [1, 3]],
  ["cudaEventSynchronize", [1, 1]],
  ["cudaMemcpy", [4, 4]],
  ["cudaMemcpyAsync", [5, 5]],
  ["cudaMemcpy2D", [7, 7]],
  ["cudaMemcpy2DAsync", [8, 8]],
  ["cudaMemcpyPeer", [5, 5]],
  ["cudaMemcpyPeerAsync", [6, 6]],
  ["cudaMemcpyToSymbol", [3, 5]],
  ["cudaMemcpyToSymbolAsync", [3, 6]],
  ["cudaMemcpyFromSymbol", [3, 5]],
  ["cudaMemcpyFromSymbolAsync", [3, 6]],
  ["cudaMemsetToSymbol", [3, 4]],
  ["cudaMemsetToSymbolAsync", [3, 5]],
  ["cudaGraphSetConditional", [2, 2]],
  ...[...CUDA_CACHE_HINT_LOADS].map((name) => [name, [1, 1]] as const),
  ...[...CUDA_CACHE_HINT_STORES].map((name) => [name, [2, 2]] as const),
  ["__cvta_generic_to_shared", [1, 1]],
  ["CP_ASYNC_CA", [3, 3]],
  ["CP_ASYNC_CG", [3, 3]],
  ["CP_ASYNC_BULK", [3, 3]],
  ["CP_ASYNC_COMMIT_GROUP", [0, 0]],
  ["CP_ASYNC_WAIT_ALL", [0, 0]],
  ["CP_ASYNC_WAIT_GROUP", [1, 1]],
  ["CP_ASYNC_BULK_COMMIT_GROUP", [0, 0]],
  ["CP_ASYNC_BULK_WAIT_ALL", [0, 0]],
  ["CP_ASYNC_BULK_WAIT_GROUP", [1, 1]],
  ["clock", [0, 0]],
  ["clock64", [0, 0]],
  ["__builtin_assume_aligned", [2, 2]],
  ["ct::assume_aligned", [1, 2]],
  ["__bfloat1622float2", [1, 1]],
  ["__bfloat162bfloat162", [1, 1]],
  ["__float22bfloat162_rn", [1, 1]],
  ["__float2bfloat162_rn", [1, 1]],
  ["__floats2bfloat162_rn", [2, 2]],
  ["__halves2bfloat162", [2, 2]],
  ["__low2bfloat16", [1, 1]],
  ["__high2bfloat16", [1, 1]],
  ["__low2bfloat162", [1, 1]],
  ["__high2bfloat162", [1, 1]],
  ["__lows2bfloat162", [2, 2]],
  ["__highs2bfloat162", [2, 2]],
  ["dot", [2, 2]],
  ["length", [1, 1]],
  ["normalize", [1, 1]],
  ["cross", [2, 2]],
  ["printf", [1, Number.POSITIVE_INFINITY]],
  ...[...CUDA_VECTOR_CONSTRUCTORS].map(([name, type]) => {
    const info = CUDA_VECTOR_TYPES.get(type);
    return [name, [1, info?.lanes ?? 1]] as const;
  }),
]);
type ValueType = Exclude<CudaLiteScalarType, "void">;

interface SymbolInfo {
  readonly name: string;
  readonly kind: "param" | "local" | "shared" | "constant" | "device-global" | "texture" | "cooperative-group" | "device-function" | "builtin-vector" | "builtin-call";
  readonly valueType?: ValueType;
  readonly returnType?: CudaLiteScalarType;
  readonly params?: readonly CudaLiteParam[];
  readonly body?: readonly CudaLiteStatement[];
  readonly overloads?: readonly CudaLiteDeviceFunction[];
  readonly groupKind?: CudaLiteCooperativeGroupKind;
  readonly tileSize?: number;
  readonly partitionParent?: string;
  readonly partitionPredicate?: CudaLiteExpression;
  readonly pointer?: boolean;
  readonly constant?: boolean;
  readonly pointerRoot?: string;
  readonly dimensions?: readonly number[];
  readonly matrixTile?: CudaLiteMatrixTileMetadata;
  readonly span: SourceSpan;
}

interface Scope {
  readonly symbols: Map<string, SymbolInfo>;
  readonly parent?: Scope;
}

interface ExpressionInfo {
  readonly kind: "scalar" | "complex" | "pool-pointer" | "pointer" | "array" | "texture" | "surface" | "vector" | "function" | "address" | "string" | "matrix-tile" | "unknown";
  readonly valueType?: ValueType | undefined;
  readonly dimensions?: readonly number[] | undefined;
  readonly symbol?: SymbolInfo | undefined;
  readonly matrixTile?: CudaLiteMatrixTileMetadata | undefined;
}

export function analyzeCudaLite(
  ast: CudaLiteModule,
  options: CudaLiteAnalyzeOptions = {},
): CudaLiteAnalysis {
  const launchCallees = launchedDeviceFunctionNames(ast);
  const kernel = selectKernel(ast, options.kernelName, launchCallees);
  const selectedDeviceFunctionAsKernel = ast.functions.some((fn) => fn.name === kernel.name) &&
    !ast.kernels.some((candidate) => candidate.name === kernel.name);
  const reachableFunctionSpans = reachableDeviceFunctionSpans(ast.functions, kernel.body);
  const reachableGlobalBodies = [
    kernel.body,
    ...ast.functions
      .filter((fn) => reachableFunctionSpans.has(fn.span.start))
      .map((fn) => fn.body),
  ];
  const reachableGlobalSymbols = collectReferencedSymbolNames(reachableGlobalBodies);
  const diagnostics: CudaLiteDiagnostic[] = [];
  const requiredFeatures = new Set<string>();
  let activeRequiredFeatures = requiredFeatures;
  const atomicParams = new Set<string>();
  const atomicShared = new Set<string>();
  const atomicDeviceGlobals = new Set<string>();
  let activeAtomicParams = atomicParams;
  let activeAtomicShared = atomicShared;
  let activeAtomicDeviceGlobals = atomicDeviceGlobals;
  let activeStatementsReachable = true;
  const params = new Map(kernel.params.map((param) => [param.name, param]));
  const allowedTrapCallSpanStarts = collectCudaAllowedTrapCallSpanStarts(kernel);
  const declaredNames = new Set<string>();
  const rootScope = createScope();

  for (const constant of ast.constants) {
    const reachableGlobal = reachableGlobalSymbols.has(constant.name);
    declareConstant(constant, rootScope, declaredNames, reachableGlobal ? requiredFeatures : new Set<string>(), diagnostics, options, reachableGlobal);
  }
  for (const global of ast.deviceGlobals) {
    const reachableGlobal = reachableGlobalSymbols.has(global.name);
    declareDeviceGlobal(global, rootScope, declaredNames, reachableGlobal ? requiredFeatures : new Set<string>(), diagnostics, options, reachableGlobal);
  }
  for (const texture of ast.textures) {
    declareTexture(texture, rootScope, declaredNames, diagnostics);
  }
  for (const fn of ast.functions) {
    if (selectedDeviceFunctionAsKernel && fn.name === kernel.name) continue;
    const reachableFunction = reachableFunctionSpans.has(fn.span.start);
    const featureSink = reachableFunction ? requiredFeatures : new Set<string>();
    declareDeviceFunction(fn, rootScope, declaredNames, featureSink, diagnostics, options, reachableFunction);
  }
  const rootDeclaredNames = new Set(declaredNames);

  for (const param of kernel.params) {
    if (declaredNames.has(param.name)) {
      diagnostics.push(error("duplicate-symbol", `duplicate parameter '${param.name}'`, param.span));
    }
    validateDeclaredSymbolName(param.name, param.span, diagnostics);
    declaredNames.add(param.name);
    rootScope.symbols.set(param.name, symbolForParam(param, "param"));
    if (requiresShaderF16(param.valueType)) activeRequiredFeatures.add("shader-f16");
    validateF64Type(param.valueType, param.span, diagnostics, options);
  }

  const declareVar = (statement: CudaLiteVarDecl, scope: Scope, names: Set<string>): void => {
    const resolvedDynamicShared = resolvedSharedDimensions(statement, options);
    const dimensions = resolvedDynamicShared ?? (!activeStatementsReachable && statement.dynamicShared ? [1] : statement.dimensions);
    const pointerRoot = statement.pointer ? pointerRootForInitializer(statement.init, scope) : undefined;
    if (names.has(statement.name)) {
      diagnostics.push(error("duplicate-symbol", `duplicate CUDA-lite symbol '${statement.name}'`, statement.span));
    }
    validateDeclaredSymbolName(statement.name, statement.span, diagnostics);
    names.add(statement.name);
    scope.symbols.set(statement.name, {
      name: statement.name,
      kind: statement.storage === "shared" ? "shared" : "local",
      valueType: statement.valueType,
      pointer: statement.pointer,
      ...(pointerRoot ? { pointerRoot } : {}),
      dimensions,
      ...(statement.matrixTile === undefined ? {} : { matrixTile: statement.matrixTile }),
      span: statement.span,
    });
    if (statement.matrixTile) validateMatrixTileDeclaration(statement, activeRequiredFeatures, diagnostics);
    else validateF64Type(statement.valueType, statement.span, diagnostics, options, activeStatementsReachable);
  };

  const walkExpression = (expression: CudaLiteExpression, scope: Scope): ExpressionInfo => {
    if (expression.kind === "call") {
      return validateCallExpression(
        expression,
        scope,
        params,
        activeAtomicParams,
        activeAtomicShared,
        activeAtomicDeviceGlobals,
        activeRequiredFeatures,
        diagnostics,
        walkExpression,
        options,
        activeStatementsReachable,
        allowedTrapCallSpanStarts,
      );
    }
    return validateNonCallExpression(expression, scope, diagnostics, walkExpression, activeRequiredFeatures);
  };

  for (const constant of ast.constants) {
    const previousRequiredFeatures = activeRequiredFeatures;
    const previousReachable: boolean = activeStatementsReachable;
    activeRequiredFeatures = reachableGlobalSymbols.has(constant.name) ? requiredFeatures : new Set<string>();
    activeStatementsReachable = reachableGlobalSymbols.has(constant.name);
    validateGlobalConstantInitializer(constant, rootScope, diagnostics, walkExpression);
    activeRequiredFeatures = previousRequiredFeatures;
    activeStatementsReachable = previousReachable;
  }
  for (const global of ast.deviceGlobals) {
    const previousRequiredFeatures = activeRequiredFeatures;
    const previousReachable: boolean = activeStatementsReachable;
    activeRequiredFeatures = reachableGlobalSymbols.has(global.name) ? requiredFeatures : new Set<string>();
    activeStatementsReachable = reachableGlobalSymbols.has(global.name);
    validateDeviceGlobalInitializer(global, rootScope, diagnostics, walkExpression);
    activeRequiredFeatures = previousRequiredFeatures;
    activeStatementsReachable = previousReachable;
  }

  const walkStatements = (
    statements: readonly CudaLiteStatement[],
    scope: Scope,
    guardDepth: number,
    divergentDepth: number,
    loopDepth: number,
    names: Set<string>,
  ): void => {
    for (const statement of statements) {
      switch (statement.kind) {
        case "block": {
          const blockScope = createScope(scope);
          walkStatements(statement.body, blockScope, guardDepth, divergentDepth, loopDepth, new Set());
          break;
        }
        case "var":
          declareVar(statement, scope, names);
          if (!statement.matrixTile && requiresShaderF16(statement.valueType)) activeRequiredFeatures.add("shader-f16");
          if (!statement.matrixTile && statement.pointer && !isSupportedLocalPointer(statement, scope)) {
            diagnostics.push(error("unsupported-local-pointer", "local pointer declarations are not supported in CUDA-lite yet", statement.span));
          }
          if (!statement.matrixTile && statement.storage === "local" && statement.dimensions.length > 0 && statement.init) {
            validateArrayInitializer(statement, scope, diagnostics, walkExpression);
          }
          if (activeStatementsReachable && statement.dynamicShared && !resolvedSharedDimensions(statement, options)) {
            diagnostics.push(error("dynamic-shared-memory", "__shared__ arrays must have fixed dimensions", statement.span));
          }
          for (const dimension of statement.dimensions) {
            if (!Number.isInteger(dimension) || dimension <= 0) {
              diagnostics.push(error("invalid-array-dimension", "array dimensions must be positive integer literals", statement.span));
            }
          }
          if (!statement.matrixTile && statement.init && statement.dimensions.length === 0) {
            if (statement.pointer) validatePointerInitializerExpression(statement.init, scope, diagnostics, walkExpression);
            else walkExpression(statement.init, scope);
          }
            if (statement.init) {
              const allowPointerInitializerAssignment = statement.pointer && statement.init.kind === "assignment";
              validateSideEffectPlacement(statement.init, allowPointerInitializerAssignment, diagnostics, statement.init.kind === "sequence", statement.init.kind === "call");
            }
          break;
        case "dim3":
          if (names.has(statement.name)) {
            diagnostics.push(error("duplicate-symbol", `duplicate CUDA-lite symbol '${statement.name}'`, statement.span));
          }
          validateDeclaredSymbolName(statement.name, statement.span, diagnostics);
          names.add(statement.name);
          scope.symbols.set(statement.name, {
            name: statement.name,
            kind: "local",
            valueType: "uint",
            span: statement.span,
          });
          for (const arg of statement.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
          break;
        case "cooperative-group":
          if (names.has(statement.name)) {
            diagnostics.push(error("duplicate-symbol", `duplicate CUDA-lite symbol '${statement.name}'`, statement.span));
          }
          validateDeclaredSymbolName(statement.name, statement.span, diagnostics);
          if (statement.partitionPredicate) {
            activeRequiredFeatures.add("subgroups");
            validateScalarOperand(walkExpression(statement.partitionPredicate, scope), statement.partitionPredicate.span, diagnostics);
          }
          const parent = statement.partitionParent ? lookupSymbol(statement.partitionParent, scope, statement.span) : undefined;
          if (statement.partitionParent && parent?.kind !== "cooperative-group") {
            diagnostics.push(error("unsupported-cooperative-groups", `binary partition parent '${statement.partitionParent}' must be a cooperative group`, statement.span));
          }
          const tileSize = statement.tileSize ?? parent?.tileSize;
          names.add(statement.name);
          scope.symbols.set(statement.name, {
            name: statement.name,
            kind: "cooperative-group",
            groupKind: statement.groupKind,
            ...(tileSize === undefined ? {} : { tileSize }),
            ...(statement.partitionParent === undefined ? {} : { partitionParent: statement.partitionParent }),
            ...(statement.partitionPredicate === undefined ? {} : { partitionPredicate: statement.partitionPredicate }),
            span: statement.span,
          });
          break;
        case "kernel-launch":
          if (activeStatementsReachable) {
            diagnostics.push({
              ...error(
                "cuda-dynamic-launch-host-orchestration",
                `device-side kernel launch '${statement.callee}<<<...>>>' requires WebGPU host-orchestrated child dispatch`,
                statement.span,
              ),
              severity: "warning",
            });
          }
          for (const arg of statement.grid) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
          for (const arg of statement.block) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
          for (const arg of statement.args) walkExpression(arg, scope);
          break;
        case "asm":
          validateInlineAsmStatement(statement, scope, diagnostics, walkExpression, activeStatementsReachable);
          if (isInlineAsmBarrier(statement) && divergentDepth > 0) {
            diagnostics.push(error("divergent-barrier", "bar.sync inline PTX cannot appear in divergent control flow", statement.span));
          }
          break;
        case "expr":
          if (isBarrierCall(statement.expression)) {
            validateBarrierStatement(statement.expression, diagnostics);
            if (divergentDepth > 0) {
              diagnostics.push(error("divergent-barrier", `${expressionName(statement.expression.callee) ?? "barrier"}() cannot appear in divergent control flow`, statement.span));
            }
            break;
          } else {
            validateSideEffectPlacement(statement.expression, true, diagnostics, false, expressionIsCall(statement.expression));
            validateExpressionStatement(statement.expression, params, guardDepth, diagnostics);
          }
          walkExpression(statement.expression, scope);
          break;
        case "if": {
          validateSideEffectPlacement(statement.condition, false, diagnostics);
          walkExpression(statement.condition, scope);
          const divergent = expressionIsDivergent(statement.condition, params);
          walkStatements(statement.consequent, createScope(scope), guardDepth + 1, divergent ? divergentDepth + 1 : divergentDepth, loopDepth, new Set());
          if (statement.alternate) {
            walkStatements(statement.alternate, createScope(scope), guardDepth + 1, divergent ? divergentDepth + 1 : divergentDepth, loopDepth, new Set());
          }
          break;
        }
        case "for": {
          const loopScope = createScope(scope);
          const loopNames = new Set<string>();
          if (statement.init?.kind === "var") {
            declareVar(statement.init, loopScope, loopNames);
            if (!statement.init.matrixTile && requiresShaderF16(statement.init.valueType)) activeRequiredFeatures.add("shader-f16");
            if (!statement.init.matrixTile && statement.init.pointer && !isSupportedLocalPointer(statement.init, loopScope)) {
              diagnostics.push(error("unsupported-local-pointer", "local pointer declarations are not supported in CUDA-lite yet", statement.init.span));
            }
            if (!statement.init.matrixTile && statement.init.dimensions.length > 0 && statement.init.init) {
              validateArrayInitializer(statement.init, loopScope, diagnostics, walkExpression);
            }
            if (!statement.init.matrixTile && statement.init.init && statement.init.dimensions.length === 0) {
              if (statement.init.pointer) validatePointerInitializerExpression(statement.init.init, loopScope, diagnostics, walkExpression);
              else walkExpression(statement.init.init, loopScope);
            }
            if (statement.init.init) validateSideEffectPlacement(statement.init.init, false, diagnostics, statement.init.init.kind === "sequence", statement.init.init.kind === "call");
          } else if (statement.init) {
            validateSideEffectPlacement(statement.init, true, diagnostics, false, statement.init.kind === "call");
            walkExpression(statement.init, loopScope);
          }
          if (statement.condition) validateSideEffectPlacement(statement.condition, false, diagnostics);
          if (statement.condition) walkExpression(statement.condition, loopScope);
          if (statement.update) validateSideEffectPlacement(statement.update, true, diagnostics, false, statement.update.kind === "call");
          if (statement.update) walkExpression(statement.update, loopScope);
          const divergent = statement.condition ? expressionIsDivergent(statement.condition, params) : false;
          walkStatements(statement.body, loopScope, guardDepth, divergent ? divergentDepth + 1 : divergentDepth, loopDepth + 1, loopNames);
          break;
        }
        case "while": {
          validateSideEffectPlacement(statement.condition, false, diagnostics);
          walkExpression(statement.condition, scope);
          const divergent = expressionIsDivergent(statement.condition, params);
          walkStatements(statement.body, createScope(scope), guardDepth, divergent ? divergentDepth + 1 : divergentDepth, loopDepth + 1, new Set());
          break;
        }
        case "do-while": {
          validateSideEffectPlacement(statement.condition, false, diagnostics);
          walkExpression(statement.condition, scope);
          const divergent = expressionIsDivergent(statement.condition, params);
          walkStatements(statement.body, createScope(scope), guardDepth, divergent ? divergentDepth + 1 : divergentDepth, loopDepth + 1, new Set());
          break;
        }
        case "return":
          if (statement.value) {
            validateSideEffectPlacement(statement.value, false, diagnostics, statement.value.kind === "sequence");
            const info = walkExpression(statement.value, scope);
            if (info.kind !== "scalar" && info.kind !== "vector" && info.kind !== "complex" && info.kind !== "unknown") {
              diagnostics.push(error("unsupported-return-expression", "return expression must resolve to a scalar or CUDA vector value", statement.value.span));
            }
          }
          break;
        case "continue":
          if (loopDepth === 0) {
            diagnostics.push(error("continue-outside-loop", "continue can only appear inside a loop", statement.span));
          }
          break;
        case "break":
          if (loopDepth === 0) {
            diagnostics.push(error("break-outside-loop", "break can only appear inside a loop", statement.span));
          }
          break;
      }
    }
  };

  for (const fn of ast.functions) {
    if (selectedDeviceFunctionAsKernel && fn.name === kernel.name) continue;
    const reachableFunction = reachableFunctionSpans.has(fn.span.start);
    const featureSink = reachableFunction ? requiredFeatures : new Set<string>();
    const previousRequiredFeatures = activeRequiredFeatures;
    const previousAtomicParams = activeAtomicParams;
    const previousAtomicShared = activeAtomicShared;
    const previousAtomicDeviceGlobals = activeAtomicDeviceGlobals;
    const previousStatementsReachable: boolean = activeStatementsReachable;
    activeRequiredFeatures = featureSink;
    activeAtomicParams = reachableFunction ? atomicParams : new Set<string>();
    activeAtomicShared = reachableFunction ? atomicShared : new Set<string>();
    activeAtomicDeviceGlobals = reachableFunction ? atomicDeviceGlobals : new Set<string>();
    activeStatementsReachable = reachableFunction;
    const functionScope = createScope(rootScope);
    const functionDeclaredNames = new Set(rootDeclaredNames);
    for (const param of fn.params) {
      if (functionDeclaredNames.has(param.name)) {
        diagnostics.push(error("duplicate-symbol", `duplicate parameter '${param.name}'`, param.span));
      }
      validateDeclaredSymbolName(param.name, param.span, diagnostics);
      functionDeclaredNames.add(param.name);
      functionScope.symbols.set(param.name, symbolForParam(param, "local"));
      if (requiresShaderF16(param.valueType)) activeRequiredFeatures.add("shader-f16");
      validateF64Type(param.valueType, param.span, diagnostics, options, reachableFunction);
    }
    walkStatements(fn.body, functionScope, 0, 0, 0, functionDeclaredNames);
    if (reachableFunction) {
      validateDivergentReturnsBeforeBarriers(fn.body, new Map(fn.params.map((param) => [param.name, param])), diagnostics, options.workgroupSize ?? DEFAULT_WORKGROUP_SIZE);
    }
    activeRequiredFeatures = previousRequiredFeatures;
    activeAtomicParams = previousAtomicParams;
    activeAtomicShared = previousAtomicShared;
    activeAtomicDeviceGlobals = previousAtomicDeviceGlobals;
    activeStatementsReachable = previousStatementsReachable;
  }

  walkStatements(kernel.body, rootScope, 0, 0, 0, declaredNames);
  validateDivergentReturnsBeforeBarriers(kernel.body, params, diagnostics, options.workgroupSize ?? DEFAULT_WORKGROUP_SIZE);
  markExactAtomicPointerUsage(ast, kernel, options, atomicParams, atomicShared, atomicDeviceGlobals);

  if (options.f16Mode === "f32") {
    requiredFeatures.delete("shader-f16");
  }
  if (options.subgroupMode === "scalar") {
    requiredFeatures.delete("subgroups");
  }

  if (requiredFeatures.has("shader-f16") && !options.features?.["shader-f16"]) {
    diagnostics.push(error("missing-feature-shader-f16", "half requires WebGPU shader-f16 support", kernel.span));
  }
  if (requiredFeatures.has("subgroups") && !options.features?.subgroups) {
    diagnostics.push(error("missing-feature-subgroups", "subgroup primitive requires WebGPU subgroups support", kernel.span));
  }
  if (options.features?.compatibility && requiredFeatures.has("subgroups")) {
    diagnostics.push(error("compatibility-mode-subgroups", "subgroups are disabled in WebGPU compatibility mode", kernel.span));
  }

  return {
    kernel,
    constants: ast.constants,
    deviceGlobals: ast.deviceGlobals,
    textures: ast.textures,
    functions: ast.functions,
    diagnostics,
    requiredFeatures: [...requiredFeatures].sort(),
    atomicParams: [...atomicParams].sort(),
    atomicShared: [...atomicShared].sort(),
    atomicDeviceGlobals: [...atomicDeviceGlobals].sort(),
  };
}

export function lowerCudaLiteToKernelIr(
  ast: CudaLiteModule,
  options: CudaLiteAnalyzeOptions = {},
): KernelIrModule {
  const analysis = analyzeCudaLite(ast, options);
  return lowerAnalyzedCudaLiteToKernelIr(analysis, options);
}

export function lowerAnalyzedCudaLiteToKernelIr(
  analysis: CudaLiteAnalysis,
  options: CudaLiteAnalyzeOptions = {},
): KernelIrModule {
  const errors = analysis.diagnostics.filter((diagnostic) => diagnostic.severity === "error");
  if (errors.length > 0) {
    throw new CudaLiteCompilerError("CUDA-lite analysis failed", errors);
  }
  const reachableFunctionSpans = reachableDeviceFunctionSpans(analysis.functions, analysis.kernel.body);
  const sharedDeclarationBodies = [
    analysis.kernel.body,
    ...analysis.functions
      .filter((fn) => reachableFunctionSpans.has(fn.span.start))
      .map((fn) => fn.body),
  ];
  const reachableFunctions = analysis.functions.filter((fn) => reachableFunctionSpans.has(fn.span.start));
  const reachableSymbolNames = collectReferencedSymbolNames(sharedDeclarationBodies);
  return {
    name: analysis.kernel.name,
    span: analysis.kernel.span,
    params: analysis.kernel.params,
    constants: analysis.constants.filter((constant) => reachableSymbolNames.has(constant.name)),
    deviceGlobals: analysis.deviceGlobals.filter((global) => reachableSymbolNames.has(global.name)),
    textures: analysis.textures.filter((texture) => reachableSymbolNames.has(texture.name)),
    functions: reachableFunctions,
    body: analysis.kernel.body,
    sharedDeclarations: collectSharedDeclarationsFromBodies(sharedDeclarationBodies, options),
    requiredFeatures: analysis.requiredFeatures,
    atomicParams: analysis.atomicParams,
    atomicShared: analysis.atomicShared,
    atomicDeviceGlobals: analysis.atomicDeviceGlobals,
    workgroupSize: normalizeWorkgroupSize(options.workgroupSize ?? DEFAULT_WORKGROUP_SIZE),
  };
}

function reachableDeviceFunctionSpans(
  functions: readonly CudaLiteDeviceFunction[],
  statements: readonly CudaLiteStatement[],
): ReadonlySet<number> {
  const launchCallees = new Set(collectKernelLaunchCallees(statements));
  const reachable = new Set<number>();
  const visitFunction = (fn: CudaLiteDeviceFunction): void => {
    if (reachable.has(fn.span.start)) return;
    reachable.add(fn.span.start);
    visitStatements(fn.body);
  };
  const visitFunctionByName = (name: string | undefined, arity: number): void => {
    if (name === undefined || launchCallees.has(name)) return;
    for (const candidate of functions) {
      if (candidate.name === name && candidate.params.length === arity) visitFunction(candidate);
    }
  };
  const visitStatements = (statements: readonly CudaLiteStatement[]): void => {
    walkCudaLiteExpressions(statements, (expression) => {
      if (expression.kind !== "call") return;
      const name = expressionName(expression.callee);
      visitFunctionByName(name, expression.args.length);
      const memberName = expression.callee.kind === "member" ? expression.callee.property : undefined;
      if ((name === "reduce" || name?.endsWith("::reduce") || memberName === "reduce") && expression.args.length === 3) {
        visitFunctionByName(expressionName(expression.args[2]!), 2);
      }
    });
  };
  visitStatements(statements);
  return reachable;
}

function selectKernel(
  ast: CudaLiteModule,
  kernelName: string | undefined,
  launchCallees: ReadonlySet<string>,
): CudaLiteKernel {
  if (ast.kernels.length === 0) {
    const launchableFunction = kernelName ? ast.functions.find((fn) => fn.name === kernelName && launchCallees.has(fn.name)) : undefined;
    if (launchableFunction) return deviceFunctionAsKernel(launchableFunction);
    throw new CudaLiteCompilerError("no CUDA-lite kernels found", [{
      code: "missing-kernel",
      severity: "error",
      message: "no CUDA-lite kernels found",
      span: ast.span,
    }]);
  }
  if (!kernelName) return ast.kernels[0]!;
  const kernel = ast.kernels.find((candidate) => candidate.name === kernelName);
  if (kernel) return kernel;
  const launchableFunction = ast.functions.find((fn) => fn.name === kernelName && launchCallees.has(fn.name));
  if (launchableFunction) return deviceFunctionAsKernel(launchableFunction);
  throw new CudaLiteCompilerError(`CUDA-lite kernel '${kernelName}' not found`, [{
    code: "missing-kernel",
    severity: "error",
    message: `CUDA-lite kernel '${kernelName}' not found`,
    span: ast.span,
  }]);
}

function requiresShaderF16(type: CudaLiteScalarType | undefined): boolean {
  return type === "half" || cudaVectorScalarType(type as CudaLiteScalarType) === "half";
}

function validateF64Type(
  type: CudaLiteScalarType | undefined,
  span: SourceSpan,
  diagnostics: CudaLiteDiagnostic[],
  options: CudaLiteAnalyzeOptions,
  compatibilityDiagnosticsReachable = true,
): void {
  if (type !== "double") return;
  if (!compatibilityDiagnosticsReachable) return;
  if (options.f64Mode === "f32") {
    diagnostics.push(warning(
      "f64-lowered-to-f32",
      "double is lowered to f32 in CUDA-lite f64 compatibility mode; precision and storage ABI are f32",
      span,
    ));
    return;
  }
  diagnostics.push(error("unsupported-f64", "double requires f64Mode: \"f32\" compatibility lowering; true f64 is not available in WebGPU", span));
}

function validateMatrixTileDeclaration(
  statement: CudaLiteVarDecl,
  requiredFeatures: Set<string>,
  diagnostics: CudaLiteDiagnostic[],
): void {
  const tile = statement.matrixTile;
  if (!tile) return;
  if (statement.storage !== "local") {
    diagnostics.push(error("unsupported-wmma-fragment-storage", "WMMA fragments are supported only as local variables in CUDA-lite metadata", statement.span));
  }
  if (statement.pointer) {
    diagnostics.push(error("unsupported-wmma-fragment-pointer", "WMMA fragment pointer declarations are not supported in CUDA-lite", statement.span));
  }
  if (statement.init) {
    diagnostics.push(error("unsupported-wmma-fragment-init", "WMMA fragment initializers are not supported in CUDA-lite", statement.init.span));
  }
  validateMatrixTileExtent(statement.name, "M", tile.m, diagnostics);
  validateMatrixTileExtent(statement.name, "N", tile.n, diagnostics);
  validateMatrixTileExtent(statement.name, "K", tile.k, diagnostics);

  const role = normalizeMatrixTileRole(tile.role);
  if (role === undefined) {
    diagnostics.push(error("unsupported-wmma-fragment-role", `WMMA fragment '${statement.name}' role '${tile.role}' is unsupported; supported roles: accumulator, matrix_a, matrix_b`, tile.roleSpan));
  }

  const layout = normalizeMatrixTileLayout(tile.layout);
  if (role === "matrix_a" || role === "matrix_b") {
    if (layout === undefined) {
      diagnostics.push(error("missing-wmma-fragment-layout", `WMMA fragment '${statement.name}' role '${role}' requires row_major or col_major layout`, tile.span));
    } else if (layout !== "row_major" && layout !== "col_major") {
      diagnostics.push(error("unsupported-wmma-fragment-layout", `WMMA fragment '${statement.name}' layout '${tile.layout}' is unsupported; supported layouts: row_major, col_major`, tile.layoutSpan ?? tile.span));
    }
  } else if (role === "accumulator" && layout !== undefined) {
    diagnostics.push(error("unsupported-wmma-fragment-layout", `WMMA accumulator fragment '${statement.name}' must not declare row/col layout`, tile.layoutSpan ?? tile.span));
  } else if (layout !== undefined && layout !== "row_major" && layout !== "col_major") {
    diagnostics.push(error("unsupported-wmma-fragment-layout", `WMMA fragment '${statement.name}' layout '${tile.layout}' is unsupported; supported layouts: row_major, col_major`, tile.layoutSpan ?? tile.span));
  }

  const tileValueType = normalizeMatrixTileValueType(tile.valueTypeName, tile.valueType);
  if (tileValueType === "f16") requiredFeatures.add("shader-f16");
  if (tile.valueType === undefined || tileValueType === undefined) {
    diagnostics.push(error("unsupported-wmma-fragment-value-type", `WMMA fragment '${statement.name}' value type '${tile.valueTypeName}' is unsupported; supported value types: float, half, wmma::precision::tf32, uint8_t/int8_t matrix operands, and int accumulators`, tile.valueTypeSpan));
  } else if (
    (role === "matrix_a" || role === "matrix_b") &&
    !isMatrixTileFloatValueType(tileValueType) &&
    !isMatrixTileByteValueType(tileValueType)
  ) {
    diagnostics.push(error("unsupported-wmma-fragment-value-type", `WMMA fragment '${statement.name}' role '${role}' does not support value type '${tile.valueTypeName}'`, tile.valueTypeSpan));
  } else if (
    role === "accumulator" &&
    !isMatrixTileFloatValueType(tileValueType) &&
    tileValueType !== "s32"
  ) {
    diagnostics.push(error("unsupported-wmma-fragment-value-type", `WMMA accumulator fragment '${statement.name}' does not support value type '${tile.valueTypeName}'`, tile.valueTypeSpan));
  }
}

function validateMatrixTileExtent(
  name: string,
  label: "M" | "N" | "K",
  extent: CudaLiteMatrixTileMetadata["m"],
  diagnostics: CudaLiteDiagnostic[],
): void {
  if (extent.value === undefined || !Number.isInteger(extent.value) || extent.value <= 0) {
    diagnostics.push(error("invalid-wmma-fragment-shape", `WMMA fragment '${name}' ${label} must be a positive integer constant expression`, extent.span));
  }
}

interface MatrixTileOperandInfo {
  readonly symbol: SymbolInfo;
  readonly spec: NonNullable<ReturnType<typeof resolveMatrixTileSpec>>;
}

function validateWmmaBuiltin(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  builtin: WmmaBuiltin,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): ExpressionInfo {
  switch (builtin) {
    case "fill_fragment": {
      const fragment = validateWmmaFragmentOperand(expression.args[0], scope, diagnostics, walkExpression, "wmma::fill_fragment");
      const value = expression.args[1];
      if (value) validateScalarOperand(walkExpression(value, scope), value.span, diagnostics);
      return { kind: "scalar", valueType: fragment?.spec.valueType };
    }
    case "load_matrix_sync": {
      const fragment = validateWmmaFragmentOperand(expression.args[0], scope, diagnostics, walkExpression, "wmma::load_matrix_sync");
      if (fragment && fragment.spec.role !== "matrix_a" && fragment.spec.role !== "matrix_b" && fragment.spec.role !== "accumulator") {
        diagnostics.push(error("unsupported-wmma-fragment-role", "wmma::load_matrix_sync expects a matrix or accumulator fragment", expression.args[0]?.span ?? expression.span));
      }
      validatePointerLikeOperand(expression.args[1], scope, diagnostics, walkExpression, "wmma::load_matrix_sync source");
      validateOptionalScalarOperand(expression.args[2], scope, diagnostics, walkExpression);
      validateOptionalWmmaLayoutOperand(expression.args[3], diagnostics, "load");
      return { kind: "scalar", valueType: fragment?.spec.valueType };
    }
    case "mma_sync": {
      const dst = validateWmmaFragmentOperand(expression.args[0], scope, diagnostics, walkExpression, "wmma::mma_sync destination");
      const a = validateWmmaFragmentOperand(expression.args[1], scope, diagnostics, walkExpression, "wmma::mma_sync A");
      const b = validateWmmaFragmentOperand(expression.args[2], scope, diagnostics, walkExpression, "wmma::mma_sync B");
      const c = validateWmmaFragmentOperand(expression.args[3], scope, diagnostics, walkExpression, "wmma::mma_sync accumulator");
      if (dst && dst.spec.role !== "accumulator") {
        diagnostics.push(error("unsupported-wmma-fragment-role", "wmma::mma_sync destination must be an accumulator fragment", expression.args[0]?.span ?? expression.span));
      }
      if (a && a.spec.role !== "matrix_a") {
        diagnostics.push(error("unsupported-wmma-fragment-role", "wmma::mma_sync A operand must be a matrix_a fragment", expression.args[1]?.span ?? expression.span));
      }
      if (b && b.spec.role !== "matrix_b") {
        diagnostics.push(error("unsupported-wmma-fragment-role", "wmma::mma_sync B operand must be a matrix_b fragment", expression.args[2]?.span ?? expression.span));
      }
      if (c && c.spec.role !== "accumulator") {
        diagnostics.push(error("unsupported-wmma-fragment-role", "wmma::mma_sync C operand must be an accumulator fragment", expression.args[3]?.span ?? expression.span));
      }
      validateWmmaMmaShape(dst, a, b, c, expression.span, diagnostics);
      validateWmmaMmaValueTypes(dst, a, b, c, expression.span, diagnostics);
      return { kind: "scalar", valueType: dst?.spec.valueType };
    }
    case "store_matrix_sync": {
      validatePointerLikeOperand(expression.args[0], scope, diagnostics, walkExpression, "wmma::store_matrix_sync destination");
      const fragment = validateWmmaFragmentOperand(expression.args[1], scope, diagnostics, walkExpression, "wmma::store_matrix_sync fragment");
      validateOptionalScalarOperand(expression.args[2], scope, diagnostics, walkExpression);
      validateOptionalWmmaLayoutOperand(expression.args[3], diagnostics, "store");
      return { kind: "scalar", valueType: fragment?.spec.valueType };
    }
  }
}

function validateWmmaFragmentOperand(
  expression: CudaLiteExpression | undefined,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  label: string,
): MatrixTileOperandInfo | undefined {
  if (!expression) return undefined;
  const ref = matrixTileReference(expression);
  if (!ref) {
    diagnostics.push(error("unsupported-wmma-fragment-operand", `${label} expects a WMMA fragment variable`, expression.span));
    return undefined;
  }
  for (const index of ref.indices) validateScalarOperand(walkExpression(index, scope), index.span, diagnostics);
  const symbol = lookupSymbol(ref.root, scope, expression.span);
  if (!symbol?.matrixTile) {
    diagnostics.push(error("unsupported-wmma-fragment-operand", `${label} expects a WMMA fragment variable`, expression.span));
    return undefined;
  }
  const dimensions = symbol.dimensions ?? [];
  if (ref.indices.length !== dimensions.length) {
    diagnostics.push(error("invalid-wmma-fragment-index", `WMMA fragment '${ref.root}' expects ${dimensions.length} leading index${dimensions.length === 1 ? "" : "es"} before use`, expression.span));
  }
  const spec = resolveMatrixTileSpec(symbol.matrixTile);
  if (!spec) return undefined;
  return { symbol, spec };
}

function validatePointerLikeOperand(
  expression: CudaLiteExpression | undefined,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  label: string,
): void {
  if (!expression) return;
  const info = walkExpression(expression, scope);
  if (
    info.kind !== "pointer" &&
    info.kind !== "pool-pointer" &&
    info.kind !== "address" &&
    info.kind !== "array" &&
    info.kind !== "unknown"
  ) {
    diagnostics.push(error("unsupported-wmma-pointer-operand", `${label} expects a pointer, address, or array expression`, expression.span));
  }
}

function validateOptionalScalarOperand(
  expression: CudaLiteExpression | undefined,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  if (!expression) return;
  validateScalarOperand(walkExpression(expression, scope), expression.span, diagnostics);
}

function validateOptionalWmmaLayoutOperand(
  expression: CudaLiteExpression | undefined,
  diagnostics: CudaLiteDiagnostic[],
  mode: "load" | "store",
): void {
  if (!expression) return;
  const name = expressionName(expression);
  const layout = normalizeMatrixTileLayout(name);
  if (layout === undefined) {
    diagnostics.push(error("unsupported-wmma-layout-operand", "WMMA layout operand must be a wmma row/col layout constant", expression.span));
    return;
  }
  if (mode === "store" && layout !== "mem_row_major" && layout !== "mem_col_major" && layout !== "row_major" && layout !== "col_major") {
    diagnostics.push(error("unsupported-wmma-layout-operand", "wmma::store_matrix_sync layout must be row/col memory layout", expression.span));
  }
}

function validateWmmaMmaShape(
  dst: MatrixTileOperandInfo | undefined,
  a: MatrixTileOperandInfo | undefined,
  b: MatrixTileOperandInfo | undefined,
  c: MatrixTileOperandInfo | undefined,
  span: SourceSpan,
  diagnostics: CudaLiteDiagnostic[],
): void {
  if (!dst || !a || !b || !c) return;
  if (dst.spec.m !== c.spec.m || dst.spec.n !== c.spec.n || dst.spec.k !== c.spec.k) {
    diagnostics.push(error("wmma-shape-mismatch", "wmma::mma_sync destination and accumulator fragments must have matching tile shape", span));
  }
  if (dst.spec.m !== a.spec.m || dst.spec.k !== a.spec.k || dst.spec.n !== b.spec.n || dst.spec.k !== b.spec.k) {
    diagnostics.push(error("wmma-shape-mismatch", "wmma::mma_sync matrix fragment shapes must match accumulator M/N/K", span));
  }
}

function validateWmmaMmaValueTypes(
  dst: MatrixTileOperandInfo | undefined,
  a: MatrixTileOperandInfo | undefined,
  b: MatrixTileOperandInfo | undefined,
  c: MatrixTileOperandInfo | undefined,
  span: SourceSpan,
  diagnostics: CudaLiteDiagnostic[],
): void {
  if (!dst || !a || !b || !c) return;
  if (dst.spec.tileValueType !== c.spec.tileValueType) {
    diagnostics.push(error("wmma-value-type-mismatch", "wmma::mma_sync destination and accumulator fragments must have matching value types", span));
    return;
  }
  const integerInputs = isMatrixTileByteValueType(a.spec.tileValueType) || isMatrixTileByteValueType(b.spec.tileValueType);
  const integerAccumulator = dst.spec.tileValueType === "s32";
  if (integerInputs || integerAccumulator) {
    if (
      !integerAccumulator ||
      !isMatrixTileByteValueType(a.spec.tileValueType) ||
      !isMatrixTileByteValueType(b.spec.tileValueType)
    ) {
      diagnostics.push(error("unsupported-wmma-fragment-value-type", "wmma::mma_sync integer mode supports only u8/s8 matrix_a and matrix_b fragments with int accumulator fragments", span));
    }
    return;
  }
  if (
    !isMatrixTileFloatValueType(dst.spec.tileValueType) ||
    !isMatrixTileFloatValueType(a.spec.tileValueType) ||
    !isMatrixTileFloatValueType(b.spec.tileValueType)
  ) {
    diagnostics.push(error("unsupported-wmma-fragment-value-type", "wmma::mma_sync supports float/half/tf32 fragments or u8/s8 fragments with int accumulators", span));
  }
}

function launchedDeviceFunctionNames(ast: CudaLiteModule): ReadonlySet<string> {
  const names = new Set<string>();
  for (const kernel of ast.kernels) {
    for (const name of collectKernelLaunchCallees(kernel.body)) names.add(name);
  }
  for (const fn of ast.functions) {
    for (const name of collectKernelLaunchCallees(fn.body)) names.add(name);
  }
  return names;
}

function deviceFunctionAsKernel(fn: CudaLiteDeviceFunction): CudaLiteKernel {
  return {
    kind: "kernel",
    name: fn.name,
    params: fn.params,
    body: fn.body,
    span: fn.span,
  };
}

function declareConstant(
  constant: CudaLiteGlobalConstant,
  rootScope: Scope,
  declaredNames: Set<string>,
  requiredFeatures: Set<string>,
  diagnostics: CudaLiteDiagnostic[],
  options: CudaLiteAnalyzeOptions,
  compatibilityDiagnosticsReachable = true,
): void {
  if (declaredNames.has(constant.name)) {
    diagnostics.push(error("duplicate-symbol", `duplicate CUDA-lite symbol '${constant.name}'`, constant.span));
  }
  validateDeclaredSymbolName(constant.name, constant.span, diagnostics);
  declaredNames.add(constant.name);
  rootScope.symbols.set(constant.name, {
    name: constant.name,
    kind: "constant",
    valueType: constant.valueType,
    dimensions: constant.dimensions,
    constant: true,
    span: constant.span,
  });
  if (requiresShaderF16(constant.valueType)) requiredFeatures.add("shader-f16");
  validateF64Type(constant.valueType, constant.span, diagnostics, options, compatibilityDiagnosticsReachable);
  for (const dimension of constant.dimensions) {
    if (!Number.isInteger(dimension) || dimension <= 0) {
      diagnostics.push(error("invalid-array-dimension", "array dimensions must be positive integer literals", constant.span));
    }
  }
}

function declareDeviceGlobal(
  global: CudaLiteDeviceGlobal,
  rootScope: Scope,
  declaredNames: Set<string>,
  requiredFeatures: Set<string>,
  diagnostics: CudaLiteDiagnostic[],
  options: CudaLiteAnalyzeOptions,
  compatibilityDiagnosticsReachable = true,
): void {
  if (declaredNames.has(global.name)) {
    diagnostics.push(error("duplicate-symbol", `duplicate CUDA-lite symbol '${global.name}'`, global.span));
  }
  validateDeclaredSymbolName(global.name, global.span, diagnostics);
  declaredNames.add(global.name);
  rootScope.symbols.set(global.name, {
    name: global.name,
    kind: "device-global",
    valueType: global.valueType,
    dimensions: global.dimensions,
    span: global.span,
  });
  if (requiresShaderF16(global.valueType)) requiredFeatures.add("shader-f16");
  validateF64Type(global.valueType, global.span, diagnostics, options, compatibilityDiagnosticsReachable);
  for (const dimension of global.dimensions) {
    if (!Number.isInteger(dimension) || dimension <= 0) {
      diagnostics.push(error("invalid-array-dimension", "array dimensions must be positive integer literals", global.span));
    }
  }
}

function validateGlobalConstantInitializer(
  constant: CudaLiteGlobalConstant,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  if (!constant.init) return;
  if (constant.dimensions.length === 0 && isCudaVectorType(constant.valueType)) {
    const info = walkExpression(constant.init, scope);
    if (info.kind !== "vector" || info.valueType !== constant.valueType) {
      diagnostics.push(error("invalid-constant-initializer", `constant '${constant.name}' initializer must resolve to ${constant.valueType}`, constant.init.span));
    }
    return;
  }
  const values = flattenInitializerExpressions(constant.init);
  if (constant.dimensions.length === 0 && values.length > 1 && !isCudaVectorType(constant.valueType)) {
    diagnostics.push(error("invalid-constant-initializer", `constant '${constant.name}' scalar initializer must have one value`, constant.init.span));
  }
  if (constant.dimensions.length === 0 && isCudaVectorType(constant.valueType) && values.length > cudaVectorLaneCount(constant.valueType)) {
    diagnostics.push(error("invalid-constant-initializer", `constant '${constant.name}' vector initializer has more than ${cudaVectorLaneCount(constant.valueType)} values`, constant.init.span));
  }
  const expected = constant.dimensions.reduce((product, dimension) => product * dimension, 1);
  if (constant.dimensions.length > 0 && values.length > expected) {
    diagnostics.push(error("invalid-constant-initializer", `constant '${constant.name}' initializer has more than ${expected} values`, constant.init.span));
  }
  for (const value of values) validateScalarOperand(walkExpression(value, scope), value.span, diagnostics);
}

function validateDeviceGlobalInitializer(
  global: CudaLiteDeviceGlobal,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  if (!global.init) return;
  const values = flattenInitializerExpressions(global.init);
  if (global.dimensions.length === 0 && values.length > 1) {
    diagnostics.push(error("invalid-device-global-initializer", `device global '${global.name}' scalar initializer must have one value`, global.init.span));
  }
  const expected = global.dimensions.reduce((product, dimension) => product * dimension, 1);
  if (global.dimensions.length > 0 && values.length > expected) {
    diagnostics.push(error("invalid-device-global-initializer", `device global '${global.name}' initializer has more than ${expected} values`, global.init.span));
  }
  for (const value of values) validateScalarOperand(walkExpression(value, scope), value.span, diagnostics);
}

function declareTexture(
  texture: CudaLiteTexture2D,
  rootScope: Scope,
  declaredNames: Set<string>,
  diagnostics: CudaLiteDiagnostic[],
): void {
  if (declaredNames.has(texture.name)) {
    diagnostics.push(error("duplicate-symbol", `duplicate CUDA-lite symbol '${texture.name}'`, texture.span));
  }
  validateDeclaredSymbolName(texture.name, texture.span, diagnostics);
  declaredNames.add(texture.name);
  rootScope.symbols.set(texture.name, {
    name: texture.name,
    kind: "texture",
    valueType: texture.valueType,
    span: texture.span,
  });
}

function declareDeviceFunction(
  fn: CudaLiteDeviceFunction,
  rootScope: Scope,
  declaredNames: Set<string>,
  requiredFeatures: Set<string>,
  diagnostics: CudaLiteDiagnostic[],
  options: CudaLiteAnalyzeOptions,
  compatibilityDiagnosticsReachable: boolean,
): void {
  const existing = rootScope.symbols.get(fn.name);
  if (existing?.kind === "device-function") {
    const overloads = existing.overloads ?? [];
    if (overloads.some((candidate) => deviceFunctionSignatureKey(candidate) === deviceFunctionSignatureKey(fn))) {
      diagnostics.push(error("duplicate-symbol", `duplicate CUDA-lite function overload '${fn.name}'`, fn.span));
    }
    rootScope.symbols.set(fn.name, {
      ...existing,
      overloads: [...overloads, fn],
    });
  } else if (declaredNames.has(fn.name)) {
    diagnostics.push(error("duplicate-symbol", `duplicate CUDA-lite symbol '${fn.name}'`, fn.span));
  } else {
    declaredNames.add(fn.name);
    rootScope.symbols.set(fn.name, {
      name: fn.name,
      kind: "device-function",
      returnType: fn.returnType,
      params: fn.params,
      body: fn.body,
      overloads: [fn],
      span: fn.span,
    });
  }
  validateDeclaredSymbolName(fn.name, fn.span, diagnostics);
  if (requiresShaderF16(fn.returnType)) requiredFeatures.add("shader-f16");
  validateF64Type(fn.returnType, fn.span, diagnostics, options, compatibilityDiagnosticsReachable);
  for (const param of fn.params) {
    if (requiresShaderF16(param.valueType)) requiredFeatures.add("shader-f16");
    validateF64Type(param.valueType, param.span, diagnostics, options, compatibilityDiagnosticsReachable);
  }
}

function deviceFunctionSignatureKey(fn: CudaLiteDeviceFunction): string {
  return fn.params.map((param) =>
    [
      param.valueType,
      param.pointer ? "ptr" : param.reference ? "ref" : "value",
      param.constant ? "const" : "mut",
      param.cooperativeGroupKind ?? "",
      param.tileSize ?? "",
    ].join(":")
  ).join("|");
}

function validateArrayInitializer(
  statement: CudaLiteVarDecl,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  if (!statement.init) return;
  if (statement.init.kind !== "initializer") {
    const info = walkExpression(statement.init, scope);
    if (isCudaVectorType(statement.valueType)) {
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", "CUDA vector local array scalar-fill initializer expects a vector value", statement.init.span));
      }
    } else {
      validateScalarOperand(info, statement.init.span, diagnostics);
    }
    return;
  }
  for (const element of flattenInitializerExpressions(statement.init)) {
    const info = walkExpression(element, scope);
    if (isCudaVectorType(statement.valueType)) {
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", "CUDA vector array initializer expects vector values", element.span));
      }
    } else {
      validateScalarOperand(info, element.span, diagnostics);
    }
  }
}

function flattenInitializerExpressions(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  if (expression.kind !== "initializer") return [expression];
  return expression.elements.flatMap((element) => flattenInitializerExpressions(element));
}

function flattenSequenceExpressions(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  return expression.kind === "sequence"
    ? expression.expressions.flatMap((item) => flattenSequenceExpressions(item))
    : [expression];
}

function isSupportedSharedPointerAlias(statement: CudaLiteVarDecl, scope: Scope): boolean {
  if (!statement.pointer || statement.storage !== "local") return false;
  if (statement.init?.kind !== "unary" || statement.init.operator !== "&") return false;
  if (statement.init.argument.kind !== "index" || statement.init.argument.target.kind !== "identifier") return false;
  const root = statement.init.argument.target.name;
  if (!root) return false;
  const symbol = lookupSymbol(root, scope, statement.init.argument.span);
  return symbol?.kind === "shared" && symbol.valueType === statement.valueType;
}

function isSupportedLocalPointer(statement: CudaLiteVarDecl, scope: Scope): boolean {
  if (isSupportedLocalPointerArray(statement)) return true;
  if (isSupportedSharedPointerAlias(statement, scope)) return true;
  if (isSupportedStoragePointerInitializer(statement, scope)) return true;
  if (!statement.pointer || statement.storage !== "local") return false;
  return isSupportedPoolPointerInitializer(statement.init, scope);
}

function isSupportedLocalPointerArray(statement: CudaLiteVarDecl): boolean {
  return statement.pointer &&
    statement.storage === "local" &&
    statement.dimensions.length > 0 &&
    statement.init === undefined;
}

function validatePointerInitializerExpression(
  expression: CudaLiteExpression,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  if (expression.kind === "cast" && expression.pointer) {
    validatePointerInitializerExpression(expression.expression, scope, diagnostics, walkExpression);
    return;
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const leftSource = pointerSourceType(expression.left, scope);
    if (leftSource !== undefined) {
      validatePointerInitializerExpression(expression.left, scope, diagnostics, walkExpression);
      validateScalarOperand(walkExpression(expression.right, scope), expression.right.span, diagnostics);
      return;
    }
    if (expression.operator === "+" && pointerSourceType(expression.right, scope) !== undefined) {
      validateScalarOperand(walkExpression(expression.left, scope), expression.left.span, diagnostics);
      validatePointerInitializerExpression(expression.right, scope, diagnostics, walkExpression);
      return;
    }
    validatePointerInitializerExpression(expression.left, scope, diagnostics, walkExpression);
    validateScalarOperand(walkExpression(expression.right, scope), expression.right.span, diagnostics);
    return;
  }
  if (expression.kind === "conditional") {
    const condition = walkExpression(expression.condition, scope);
    if (!isPointerLikeInfo(condition)) validateScalarOperand(condition, expression.condition.span, diagnostics);
    validatePointerInitializerExpression(expression.consequent, scope, diagnostics, walkExpression);
    validatePointerInitializerExpression(expression.alternate, scope, diagnostics, walkExpression);
    return;
  }
  if (expression.kind === "sequence") {
    const items = flattenSequenceExpressions(expression);
    const final = items.at(-1);
    for (const item of items.slice(0, -1)) walkExpression(item, scope);
    if (final) validatePointerInitializerExpression(final, scope, diagnostics, walkExpression);
    return;
  }
  if (expression.kind === "assignment" && expression.operator === "=") {
    validatePointerInitializerExpression(expression.right, scope, diagnostics, walkExpression);
    const target = walkExpression(expression.left, scope);
    if (!isPointerLikeInfo(target) && target.kind !== "unknown") {
      diagnostics.push(error("unsupported-pointer-assignment", "pointer initializer assignment target must be a modeled pointer", expression.left.span));
    }
    return;
  }
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    const pointer = expression.args[0];
    if (pointer) validatePointerInitializerExpression(pointer, scope, diagnostics, walkExpression);
    for (const arg of expression.args.slice(1)) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return;
  }
  if (expression.kind === "unary" && expression.operator === "&") {
    validateAddressOfExpression(expression.argument, scope, diagnostics, walkExpression);
    return;
  }
  if (expression.kind === "identifier") {
    expressionInfoForIdentifier(expression.name, expression.span, scope, diagnostics);
    return;
  }
  walkExpression(expression, scope);
}

function validateReadPointerExpression(
  expression: CudaLiteExpression,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): ExpressionInfo {
  if (expression.kind === "cast" && expression.pointer) {
    return validateReadPointerExpression(expression.expression, scope, diagnostics, walkExpression);
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const left = validateReadPointerExpression(expression.left, scope, diagnostics, walkExpression);
    if (isPointerLikeInfo(left) || left.kind === "array" || left.kind === "address" || left.kind === "unknown") {
      validateScalarOperand(walkExpression(expression.right, scope), expression.right.span, diagnostics);
      return left;
    }
    if (expression.operator === "+") {
      validateScalarOperand(left, expression.left.span, diagnostics);
      return validateReadPointerExpression(expression.right, scope, diagnostics, walkExpression);
    }
    validateScalarOperand(walkExpression(expression.right, scope), expression.right.span, diagnostics);
    return left;
  }
  if (expression.kind === "conditional") {
    const condition = walkExpression(expression.condition, scope);
    if (!isPointerLikeInfo(condition)) validateScalarOperand(condition, expression.condition.span, diagnostics);
    const consequent = validateReadPointerExpression(expression.consequent, scope, diagnostics, walkExpression);
    const alternate = validateReadPointerExpression(expression.alternate, scope, diagnostics, walkExpression);
    return conditionalPointerInfo(expression, consequent, alternate, diagnostics) ?? consequent;
  }
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    const pointer = expression.args[0];
    const info = pointer === undefined
      ? { kind: "unknown" as const }
      : validateReadPointerExpression(pointer, scope, diagnostics, walkExpression);
    for (const arg of expression.args.slice(1)) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return info;
  }
  if (expression.kind === "unary" && expression.operator === "&") {
    return validateAddressOfExpression(expression.argument, scope, diagnostics, walkExpression);
  }
  return walkExpression(expression, scope);
}

function validateAddressOfExpression(
  expression: CudaLiteExpression,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): ExpressionInfo {
  const info = walkExpression(expression, scope);
  const addressable = expression.kind === "identifier" ||
    expression.kind === "index" ||
    expression.kind === "member" ||
    (expression.kind === "unary" && expression.operator === "*");
  if (!addressable && info.kind !== "unknown") {
    diagnostics.push(error("invalid-address-target", "address-of expects an addressable CUDA expression", expression.span));
  }
  return { kind: "address", valueType: info.valueType, symbol: info.symbol };
}

function isSupportedStoragePointerInitializer(statement: CudaLiteVarDecl, scope: Scope): boolean {
  if (!statement.pointer || statement.storage !== "local") return false;
  const source = pointerSourceType(statement.init, scope);
  return source !== undefined && pointerTypesCompatible(statement.valueType, source, hasExplicitPointerCast(statement.init));
}

function pointerRootForInitializer(expression: CudaLiteExpression | undefined, scope: Scope): string | undefined {
  const root = expression ? rootIdentifier(expression) : undefined;
  if (!root) return undefined;
  const symbol = lookupSymbol(root, scope, expression?.span ?? { start: 0, end: 0, line: 1, column: 1 });
  return symbol?.pointerRoot ?? root;
}

function pointerSourceType(expression: CudaLiteExpression | undefined, scope: Scope): ValueType | undefined {
  if (!expression) return undefined;
  if (expression.kind === "cast" && expression.pointer) return pointerSourceType(expression.expression, scope);
  if (expression.kind === "conditional") {
    const consequent = pointerSourceType(expression.consequent, scope);
    const alternate = pointerSourceType(expression.alternate, scope);
    if (consequent !== undefined && alternate !== undefined) return pointerTypesCompatible(consequent, alternate, true) ? consequent : undefined;
    if (consequent !== undefined && isNullPointerLiteral(expression.alternate)) return consequent;
    if (alternate !== undefined && isNullPointerLiteral(expression.consequent)) return alternate;
    return undefined;
  }
  if (expression.kind === "sequence") {
    return pointerSourceType(flattenSequenceExpressions(expression).at(-1), scope);
  }
  if (expression.kind === "assignment" && expression.operator === "=") {
    const left = pointerSourceType(expression.left, scope);
    const right = pointerSourceType(expression.right, scope);
    if (left === undefined || right === undefined) return undefined;
    return pointerTypesCompatible(left, right, true) ? right : undefined;
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const left = pointerSourceType(expression.left, scope);
    if (left !== undefined) return left;
    return expression.operator === "+" ? pointerSourceType(expression.right, scope) : undefined;
  }
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    return pointerSourceType(expression.args[0], scope);
  }
  if (expression.kind === "unary" && expression.operator === "&") {
    const root = rootIdentifier(expression.argument);
    const symbol = root ? lookupSymbol(root, scope, expression.argument.span) : undefined;
    return symbol?.valueType;
  }
  if (expression.kind !== "identifier") return undefined;
  const symbol = lookupSymbol(expression.name, scope, expression.span);
  if (symbol?.kind === "local" && (symbol.dimensions?.length ?? 0) > 0) return symbol.valueType;
  if (symbol?.kind === "shared" || symbol?.kind === "device-global") return symbol.valueType;
  return symbol?.pointer ? symbol.valueType : undefined;
}

function pointerTypesCompatible(target: ValueType, source: ValueType, allowWordReinterpret = false): boolean {
  if (target === source) return true;
  if (allowWordReinterpret && isWordAddressablePointerType(target) && isWordAddressablePointerType(source)) return true;
  const targetScalar = cudaVectorScalarType(target);
  if (targetScalar && targetScalar === source) return true;
  const sourceScalar = cudaVectorScalarType(source);
  if (sourceScalar !== undefined && sourceScalar === target) return true;
  return scalarizedStorageTypesCompatible(target, source) || scalarizedStorageTypesCompatible(source, target);
}

function isWordAddressablePointerType(type: ValueType): boolean {
  return type === "float" ||
    type === "int" ||
    type === "uint" ||
    type === "uchar" ||
    type === "half" ||
    type === "bf16" ||
    isCudaVectorType(type);
}

function scalarizedStorageTypesCompatible(target: ValueType, source: ValueType): boolean {
  return (target === "float" && source === "bf16") ||
    (target === "bf16" && source === "float") ||
    (target === "float" && source === "half") ||
    (target === "half" && source === "float");
}

function hasExplicitPointerCast(expression: CudaLiteExpression | undefined): boolean {
  if (!expression) return false;
  if (expression.kind === "cast" && expression.pointer) return true;
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    return hasExplicitPointerCast(expression.left);
  }
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    return hasExplicitPointerCast(expression.args[0]);
  }
  return false;
}

function isSupportedPoolPointerInitializer(init: CudaLiteExpression | undefined, scope: Scope): boolean {
  if (!init) return true;
  if (isNullPointerLiteral(init)) return true;
  if (init.kind === "identifier") {
    const symbol = lookupSymbol(init.name, scope, init.span);
    return symbol?.kind === "local" && symbol.pointer === true;
  }
  if (init.kind === "cast" && init.pointer) return isSupportedPoolPointerInitializer(init.expression, scope);
  if (init.kind !== "call") return false;
  const callName = expressionName(init.callee);
  if (callName !== "deviceAllocate" && callName !== "streamOrderedAllocate") return false;
  if (init.args.length === 4) {
    const base = init.args[0];
    const offset = init.args[1];
    if (base?.kind !== "identifier" || offset?.kind !== "identifier") return false;
    const baseSymbol = lookupSymbol(base.name, scope, base.span);
    const offsetSymbol = lookupSymbol(offset.name, scope, offset.span);
    return baseSymbol?.pointer === true && offsetSymbol?.pointer === true &&
      (offsetSymbol.valueType === "uint" || offsetSymbol.valueType === "int");
  }
  const pool = init.args[0];
  if (isExternalPoolAddress(pool)) return true;
  if (pool?.kind !== "identifier") return false;
  const symbol = lookupSymbol(pool.name, scope, pool.span);
  return symbol?.valueType === "devicepool" && symbol.pointer === true;
}

type ExpressionWalker = (expression: CudaLiteExpression, scope: Scope) => ExpressionInfo;

function validateInlineAsmOutputValueTypes(
  op: InlineAsmOp,
  outputInfos: readonly ExpressionInfo[],
  outputs: readonly CudaLiteExpression[],
  fallbackSpan: SourceSpan,
  diagnostics: CudaLiteDiagnostic[],
): void {
  const contract = inlineAsmOutputValueContract(op);
  if (!contract) return;
  const count = contract.allOutputs ? outputs.length : Math.min(outputs.length, 1);
  for (let index = 0; index < count; index++) {
    const valueType = outputInfos[index]?.valueType;
    if (inlineAsmOutputValueTypeMatches(contract, valueType)) continue;
    diagnostics.push(error("invalid-inline-asm-operands", `${contract.label} inline PTX writes ${contract.description}`, outputs[index]?.span ?? fallbackSpan));
  }
}

function validateInlineAsmInputValueTypes(
  op: InlineAsmOp,
  statement: Extract<CudaLiteStatement, { kind: "asm" }>,
  outputCount: number,
  scope: Scope,
  walkExpression: ExpressionWalker,
  diagnostics: CudaLiteDiagnostic[],
): void {
  for (const contract of inlineAsmInputValueContracts(op, outputCount)) {
    const input = statement.inputs[contract.inputIndex];
    if (!input) continue;
    const inputInfo = walkExpression(input, scope);
    if (inputInfo.kind === "unknown" || inlineAsmInputValueTypeMatches(contract, inputInfo.valueType)) continue;
    diagnostics.push(error("invalid-inline-asm-operands", `${contract.label} inline PTX expects ${contract.description}`, input.span));
  }
}

function validateInlineAsmStatement(
  statement: Extract<CudaLiteStatement, { kind: "asm" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  compatibilityDiagnosticsReachable: boolean,
): void {
  const op = classifyInlineAsm(statement.template);
  const outputs = statement.outputs ?? (statement.output === undefined ? [] : [statement.output]);
  const asmDiagnostics = compatibilityDiagnosticsReachable ? diagnostics : [];
  if (!op) {
    asmDiagnostics.push(error("unsupported-inline-asm", `only ${inlineAsmSupportedList()} inline PTX are supported in CUDA-lite v0`, statement.span));
  }
  const outputInfos = outputs.map((output) => walkExpression(output, scope));
  for (let index = 0; index < outputs.length; index++) {
    const output = outputs[index]!;
    validateLValueExpression(output, scope, diagnostics, walkExpression);
    validateScalarOperand(outputInfos[index]!, output.span, diagnostics);
  }
  if (op) validateInlineAsmOutputValueTypes(op, outputInfos, outputs, statement.span, asmDiagnostics);
  if (op) validateInlineAsmInputValueTypes(op, statement, outputs.length, scope, walkExpression, asmDiagnostics);
  const shapeDiagnostic = op ? inlineAsmOperandShapeDiagnostic(op, outputs.length, statement.inputs.length) : undefined;
  if (shapeDiagnostic) {
    asmDiagnostics.push(error("invalid-inline-asm-operands", shapeDiagnostic, statement.span));
  }
  if (op?.kind === "isspacep") {
    const input = statement.inputs[0];
    if (input) {
      const info = walkExpression(input, scope);
      if (info.kind !== "pointer" && info.kind !== "pool-pointer" && info.kind !== "address" && info.kind !== "array" && info.kind !== "unknown") {
        asmDiagnostics.push(error("invalid-inline-asm-operands", `isspacep.${op.space} inline PTX expects a pointer input operand`, input.span));
      }
    }
    return;
  }
  if (op?.kind === "cp-async-fence") {
    for (const input of statement.inputs) {
      validateScalarOperand(walkExpression(input, scope), input.span, asmDiagnostics);
    }
  }
  if (op?.kind === "bar-sync") {
    for (const input of statement.inputs) validateScalarOperand(walkExpression(input, scope), input.span, asmDiagnostics);
  }
  for (const input of statement.inputs) {
    validateScalarOperand(walkExpression(input, scope), input.span, diagnostics);
  }
}

function validateCallExpression(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  params: ReadonlyMap<string, CudaLiteParam>,
  atomicParams: Set<string>,
  atomicShared: Set<string>,
  atomicDeviceGlobals: Set<string>,
  requiredFeatures: Set<string>,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  options: CudaLiteAnalyzeOptions,
  compatibilityDiagnosticsReachable: boolean,
  allowedTrapCallSpanStarts: ReadonlySet<number>,
): ExpressionInfo {
  const callName = expressionName(expression.callee);
  const namespaceCooperativeCall = cooperativeNamespaceCall(expression, scope);
  if (namespaceCooperativeCall) {
    return validateCooperativeNamespaceCall(expression, namespaceCooperativeCall, diagnostics, walkExpression, scope, compatibilityDiagnosticsReachable);
  }
  const cooperativeCall = cooperativeGroupCall(expression, scope);
  if (cooperativeCall) {
    return validateCooperativeGroupCall(expression, cooperativeCall, requiredFeatures, diagnostics, walkExpression, scope, compatibilityDiagnosticsReachable);
  }
  if (!callName) {
    diagnostics.push(error("unsupported-call", "CUDA-lite v0 only supports direct builtin calls", expression.span));
    for (const arg of expression.args) walkExpression(arg, scope);
    return { kind: "unknown" };
  }

  const calleeSymbol = lookupSymbol(callName, scope, expression.callee.span);
  if (calleeSymbol?.kind === "device-function") {
    return validateDeviceFunctionCall(expression, calleeSymbol, diagnostics, walkExpression, scope);
  }
  if (calleeSymbol && calleeSymbol.kind !== "builtin-call") {
    diagnostics.push(error("unsupported-call", `CUDA-lite symbol '${callName}' is not callable`, expression.callee.span));
    for (const arg of expression.args) walkExpression(arg, scope);
    return { kind: "unknown" };
  }

  const arity = BUILTIN_CALLS.get(callName);
  if (!arity) {
    diagnostics.push(error("unsupported-call", `unsupported CUDA-lite call '${callName}'`, expression.span));
    for (const arg of expression.args) walkExpression(arg, scope);
    return { kind: "unknown" };
  }

  const [minArgs, maxArgs] = arity;
  if (expression.args.length < minArgs || expression.args.length > maxArgs) {
    diagnostics.push(error(
      "invalid-call-arity",
      `${callName} expects ${formatArity(minArgs, maxArgs)} argument${maxArgs === 1 ? "" : "s"}`,
      expression.span,
    ));
  }

  const wmma = wmmaBuiltinName(callName);
  if (wmma) return validateWmmaBuiltin(expression, wmma, scope, diagnostics, walkExpression);

  if (callName === "bg_subgroup_add") requiredFeatures.add("subgroups");
  if (callName === "__syncthreads" || callName === "__syncwarp") {
    diagnostics.push(error("barrier-expression", `${callName}() must be used as a standalone statement`, expression.span));
  }
  if (callName === "printf") {
    for (const arg of expression.args.slice(1)) {
      const info = walkExpression(arg, scope);
      if (isPrintfArgument(info)) continue;
      validateScalarOperand(info, arg.span, diagnostics);
    }
    return { kind: "scalar" };
  }
  if (callName === "cudaGetDevice" ||
    callName === "cudaGetDeviceCount" ||
    callName === "cudaDeviceGetAttribute" ||
    callName === "cudaDeviceGetLimit" ||
    callName === "cudaThreadGetLimit" ||
    callName === "cudaDeviceCanAccessPeer" ||
    callName === "cudaGetDeviceFlags" ||
    callName === "cudaMemGetInfo" ||
    callName === "cudaOccupancyMaxActiveBlocksPerMultiprocessor" ||
    callName === "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags" ||
    callName === "cudaOccupancyMaxPotentialBlockSize" ||
    callName === "cudaOccupancyMaxPotentialBlockSizeWithFlags" ||
    callName === "cudaOccupancyAvailableDynamicSMemPerBlock" ||
    callName === "cudaDeviceGetCacheConfig" ||
    callName === "cudaDeviceGetSharedMemConfig" ||
    callName === "cudaThreadGetCacheConfig" ||
    callName === "cudaThreadExchangeStreamCaptureMode" ||
    callName === "cudaDeviceGetStreamPriorityRange" ||
    callName === "cudaStreamCreate" ||
    callName === "cudaStreamCreateWithFlags" ||
    callName === "cudaStreamCreateWithPriority" ||
    callName === "cudaStreamGetDevice" ||
    callName === "cudaStreamGetFlags" ||
    callName === "cudaStreamGetId" ||
    callName === "cudaStreamGetPriority" ||
    callName === "cudaStreamIsCapturing" ||
    callName === "cudaStreamGetCaptureInfo" ||
    callName === "cudaStreamGetCaptureInfo_v2" ||
    callName === "cudaStreamEndCapture" ||
    callName === "cudaGraphCreate" ||
    callName === "cudaGraphInstantiate" ||
    callName === "cudaGraphInstantiateWithFlags" ||
    callName === "cudaGraphExecUpdate" ||
    callName === "cudaEventCreate" ||
    callName === "cudaEventCreateWithFlags" ||
    callName === "cudaRuntimeGetVersion" ||
    callName === "cudaDriverGetVersion") {
    validateCudaIntegerRuntimeQuery(expression, callName, scope, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "int" };
  }
  if (isHostManagedRuntimeNoopCall(callName)) {
    validateHostManagedRuntimeNoopCall(expression, walkExpression, scope);
    return { kind: "scalar", valueType: "int" };
  }
  if (callName === "cudaEventElapsedTime") {
    validateCudaEventElapsedTime(expression, scope, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "int" };
  }
  if (isCudaRuntimeCopyCall(callName)) {
    validateRuntimeCopyCall(expression, callName, diagnostics, walkExpression, scope, options, compatibilityDiagnosticsReachable);
    return { kind: "scalar", valueType: "int" };
  }
  if (callName === "cudaGraphSetConditional") {
    validateCudaGraphSetConditionalCall(expression, diagnostics, walkExpression, scope);
    return { kind: "scalar", valueType: "int" };
  }
  if (isAtomicBuiltin(callName)) {
    validateAtomicBuiltin(expression, scope, params, atomicParams, atomicShared, atomicDeviceGlobals, diagnostics, walkExpression);
    return { kind: "scalar" };
  }
  if (isAddressSpacePredicateCall(callName)) {
    validateAddressSpacePredicateCall(expression, callName, scope, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "int" };
  }
  if (isPointerIdentityCall(callName)) {
    return validatePointerIdentityCall(expression, callName, scope, diagnostics, walkExpression);
  }
  if (isCuComplexBuiltin(callName)) {
    validateCuComplexBuiltin(expression, callName, scope, diagnostics, walkExpression, options);
    return isCuComplexScalarBuiltin(callName)
      ? { kind: "scalar", valueType: "float" }
      : { kind: "complex", valueType: "complex64" };
  }
  if (CUDA_CACHE_HINT_LOADS.has(callName)) {
    const arg = expression.args[0];
    if (!arg) return { kind: "unknown" };
    const info = validateReadPointerOperand(arg, scope, walkExpression);
    if (info.kind !== "pointer" && info.kind !== "pool-pointer" && info.kind !== "address" && info.kind !== "unknown") {
      diagnostics.push(error("unsupported-cache-hint-address", `${callName} expects a pointer expression`, arg.span));
    }
    return isCudaVectorType(info.valueType)
      ? { kind: "vector", valueType: info.valueType }
      : { kind: "scalar", valueType: info.valueType };
  }
  if (CUDA_CACHE_HINT_STORES.has(callName)) {
    const target = expression.args[0];
    const value = expression.args[1];
    let targetInfo: ExpressionInfo | undefined;
    if (target) {
      targetInfo = validateReadPointerOperand(target, scope, walkExpression);
      if (targetInfo.kind !== "pointer" && targetInfo.kind !== "pool-pointer" && targetInfo.kind !== "address" && targetInfo.kind !== "unknown") {
        diagnostics.push(error("unsupported-cache-hint-address", `${callName} expects a pointer expression`, target.span));
      }
    }
    if (value) validateCacheHintStoreValue(callName, targetInfo, value, scope, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "voidptr" };
  }
  if (callName === "__cvta_generic_to_shared") {
    const target = expression.args[0];
    if (!target) return { kind: "unknown" };
    const info = walkExpression(target, scope);
    if (info.kind !== "pointer" && info.kind !== "address" && info.kind !== "array" && info.kind !== "unknown") {
      diagnostics.push(error("unsupported-cache-hint-address", "__cvta_generic_to_shared expects a pointer expression", target.span));
    }
    return { kind: "scalar", valueType: "uint" };
  }
  if (isCpAsyncCopyCall(callName)) {
    validateCpAsyncCopy(expression, scope, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "voidptr" };
  }
  if (isCpAsyncFenceCall(callName)) {
    for (const arg of expression.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return { kind: "scalar", valueType: "voidptr" };
  }
  if (callName === "clock") {
    return { kind: "scalar", valueType: "uint" };
  }
  if (callName === "clock64") {
    return { kind: "scalar", valueType: "uint" };
  }
  if (callName === "__trap") {
    if (compatibilityDiagnosticsReachable && !allowedTrapCallSpanStarts.has(expression.span.start)) {
      diagnostics.push(error("unsupported-device-trap", "__trap cannot be lowered to WebGPU without an explicit device abort contract", expression.span));
    }
    return { kind: "scalar", valueType: "int" };
  }
  if (callName === "__nanosleep" || callName === "__prof_trigger") {
    for (const arg of expression.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return { kind: "scalar", valueType: "int" };
  }
  if (isFillRegsBuiltin(callName)) {
    validateFillRegs(expression, scope, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "voidptr" };
  }
  const vectorConstructor = cudaVectorConstructorType(callName);
  if (vectorConstructor) {
    validateVectorConstructorArgs(vectorConstructor, expression, scope, diagnostics, walkExpression);
    return { kind: "vector", valueType: vectorConstructor };
  }
  if (callName === "__bfloat1622float2" || callName === "__float22bfloat162_rn") {
    const expectedType = callName === "__bfloat1622float2" ? "bf162" : "float2";
    const returnType = callName === "__bfloat1622float2" ? "float2" : "bf162";
    const arg = expression.args[0];
    if (arg) {
      const info = walkExpression(arg, scope);
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects ${expectedType} argument`, arg.span));
      } else if (info.kind === "vector" && info.valueType !== expectedType) {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects ${expectedType} argument`, arg.span));
      }
    }
    return { kind: "vector", valueType: returnType };
  }
  if (callName === "__halves2bfloat162" || callName === "__floats2bfloat162_rn") {
    for (const arg of expression.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return { kind: "vector", valueType: "bf162" };
  }
  if (callName === "__bfloat162bfloat162" || callName === "__float2bfloat162_rn") {
    const arg = expression.args[0];
    if (arg) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return { kind: "vector", valueType: "bf162" };
  }
  if (callName === "__low2bfloat16" || callName === "__high2bfloat16") {
    const arg = expression.args[0];
    if (arg) {
      const info = walkExpression(arg, scope);
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects bf162 argument`, arg.span));
      } else if (info.kind === "vector" && info.valueType !== "bf162") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects bf162 argument`, arg.span));
      }
    }
    return { kind: "scalar", valueType: "bf16" };
  }
  if (callName === "__low2bfloat162" || callName === "__high2bfloat162" || callName === "__lows2bfloat162" || callName === "__highs2bfloat162") {
    for (const arg of expression.args) {
      const info = walkExpression(arg, scope);
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects bf162 argument`, arg.span));
      } else if (info.kind === "vector" && info.valueType !== "bf162") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects bf162 argument`, arg.span));
      }
    }
    return { kind: "vector", valueType: "bf162" };
  }
  if (callName === "__bfloat162_as_uint" || callName === "__nv_bfloat162_as_uint") {
    const arg = expression.args[0];
    if (arg) {
      const info = walkExpression(arg, scope);
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects bf162 argument`, arg.span));
      } else if (info.kind === "vector" && info.valueType !== "bf162") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects bf162 argument`, arg.span));
      }
    }
    return { kind: "scalar", valueType: "uint" };
  }
  if (callName === "__uint_as_bfloat162" || callName === "__uint_as_nv_bfloat162") {
    const arg = expression.args[0];
    if (arg) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return { kind: "vector", valueType: "bf162" };
  }
  if (callName === "__hadd") {
    const infos = expression.args.map((arg) => walkExpression(arg, scope));
    for (const [index, info] of infos.entries()) validateScalarOperand(info, expression.args[index]!.span, diagnostics);
    if (infos.some((info) => info.valueType === "bf16")) return { kind: "scalar", valueType: "bf16" };
    if (infos.some((info) => info.valueType === "half")) {
      requiredFeatures.add("shader-f16");
      return { kind: "scalar", valueType: "half" };
    }
    return { kind: "scalar", valueType: "int" };
  }
  if (isBfloat16ScalarArithmetic(callName) || isBfloat16ScalarPredicate(callName)) {
    const infos = expression.args.map((arg) => walkExpression(arg, scope));
    for (const [index, info] of infos.entries()) validateScalarOperand(info, expression.args[index]!.span, diagnostics);
    if (infos.some((info) => info.valueType === "bf16")) {
      return { kind: "scalar", valueType: isBfloat16ScalarPredicate(callName) ? callName === "__hisinf" ? "int" : "uint" : "bf16" };
    }
  }
  if (isBf162VectorIntrinsic(callName) || isBf162ComparisonMaskIntrinsic(callName) || isBf162BooleanComparisonIntrinsic(callName) || isHalf2VectorIntrinsic(callName) || isHalf2ComparisonMaskIntrinsic(callName) || isHalf2BooleanComparisonIntrinsic(callName)) {
    const infos = expression.args.map((arg) => walkExpression(arg, scope));
    const hasBf162Operand = infos.some((info) => info.kind === "vector" && info.valueType === "bf162");
    if ((isBf162VectorIntrinsic(callName) || isBf162ComparisonMaskIntrinsic(callName) || isBf162BooleanComparisonIntrinsic(callName)) && (hasBf162Operand || isBf162OnlyVectorIntrinsic(callName))) {
      for (const [index, info] of infos.entries()) {
        const arg = expression.args[index]!;
        if (info.kind !== "vector" && info.kind !== "unknown") {
          diagnostics.push(error("unsupported-vector-argument", `${callName} expects bf162 arguments`, arg.span));
        } else if (info.kind === "vector" && info.valueType !== "bf162") {
          diagnostics.push(error("unsupported-vector-argument", `${callName} expects bf162 arguments`, arg.span));
        }
      }
      if (isBf162ComparisonMaskIntrinsic(callName)) return { kind: "scalar", valueType: "uint" };
      if (isBf162BooleanComparisonIntrinsic(callName)) return { kind: "scalar", valueType: "bool" };
      return { kind: "vector", valueType: "bf162" };
    }
    requiredFeatures.add("shader-f16");
    for (const [index, info] of infos.entries()) {
      const arg = expression.args[index]!;
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects half2 arguments`, arg.span));
      } else if (info.kind === "vector" && info.valueType !== "half2") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects half2 arguments`, arg.span));
      }
    }
    if (isHalf2ComparisonMaskIntrinsic(callName)) return { kind: "scalar", valueType: "uint" };
    if (isHalf2BooleanComparisonIntrinsic(callName)) return { kind: "scalar", valueType: "bool" };
    return { kind: "vector", valueType: "half2" };
  }
  if (callName === "__half22float2" || callName === "__float22half2_rn") {
    requiredFeatures.add("shader-f16");
    const expectedType = callName === "__half22float2" ? "half2" : "float2";
    const returnType = callName === "__half22float2" ? "float2" : "half2";
    const arg = expression.args[0];
    if (arg) {
      const info = walkExpression(arg, scope);
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects ${expectedType} argument`, arg.span));
      } else if (info.kind === "vector" && info.valueType !== expectedType) {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects ${expectedType} argument`, arg.span));
      }
    }
    return { kind: "vector", valueType: returnType };
  }
  if (callName === "__half2_as_uint") {
    requiredFeatures.add("shader-f16");
    const arg = expression.args[0];
    if (arg) {
      const info = walkExpression(arg, scope);
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", "__half2_as_uint expects half2 argument", arg.span));
      } else if (info.kind === "vector" && info.valueType !== "half2") {
        diagnostics.push(error("unsupported-vector-argument", "__half2_as_uint expects half2 argument", arg.span));
      }
    }
    return { kind: "scalar", valueType: "uint" };
  }
  if (callName === "__uint_as_half2") {
    requiredFeatures.add("shader-f16");
    const arg = expression.args[0];
    if (arg) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return { kind: "vector", valueType: "half2" };
  }
  if (callName === "__low2half" || callName === "__high2half" || callName === "__low2float" || callName === "__high2float") {
    if (callName === "__low2half" || callName === "__high2half") requiredFeatures.add("shader-f16");
    const arg = expression.args[0];
    let vectorType: CudaLiteScalarType | undefined;
    if (arg) {
      const info = walkExpression(arg, scope);
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects half2 or bf162 argument`, arg.span));
      } else if (info.kind === "vector" && info.valueType !== "half2" && info.valueType !== "bf162") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects half2 or bf162 argument`, arg.span));
      } else if (info.kind === "vector") {
        vectorType = info.valueType;
      }
    }
    if (vectorType !== "bf162") requiredFeatures.add("shader-f16");
    return { kind: "scalar", valueType: callName === "__low2float" || callName === "__high2float" ? "float" : vectorType === "bf162" ? "bf16" : "half" };
  }
  if (callName === "__halves2half2" || callName === "__half2half2") {
    requiredFeatures.add("shader-f16");
    for (const arg of expression.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return { kind: "vector", valueType: "half2" };
  }
  if (callName === "__low2half2" || callName === "__high2half2" || callName === "__lowhigh2highlow" || callName === "__lows2half2" || callName === "__highs2half2") {
    if (callName !== "__lowhigh2highlow") requiredFeatures.add("shader-f16");
    let vectorType: CudaLiteScalarType | undefined;
    for (const arg of expression.args) {
      const info = walkExpression(arg, scope);
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects half2 or bf162 argument`, arg.span));
      } else if (info.kind === "vector" && info.valueType !== "half2" && info.valueType !== "bf162") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects half2 or bf162 argument`, arg.span));
      } else if (info.kind === "vector") {
        vectorType = vectorType ?? info.valueType;
        if (vectorType !== info.valueType) diagnostics.push(error("unsupported-vector-argument", `${callName} expects matching half2 or bf162 arguments`, arg.span));
      }
    }
    if (vectorType === "bf162") return { kind: "vector", valueType: "bf162" };
    requiredFeatures.add("shader-f16");
    return { kind: "vector", valueType: "half2" };
  }
  if (callName === "__float2half2_rn" || callName === "__floats2half2_rn") {
    requiredFeatures.add("shader-f16");
    for (const arg of expression.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return { kind: "vector", valueType: "half2" };
  }
  const vectorMath = validateVectorMinMaxCall(expression, callName, scope, diagnostics, requiredFeatures, walkExpression);
  if (vectorMath) return vectorMath;
  if (callName === "frexp" || callName === "frexpf" || callName === "modf" || callName === "modff" || callName === "remquo" || callName === "remquof") {
    if (callName === "frexp" || callName === "frexpf") validateFrexp(expression, scope, diagnostics, walkExpression);
    else if (callName === "modf" || callName === "modff") validateModf(expression, scope, diagnostics, walkExpression);
    else validateRemquo(expression, scope, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "float" };
  }
  if (isSincosCallName(callName)) {
    validateSincos(expression, scope, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "float" };
  }
  if (isNanPayloadCallName(callName)) {
    const payload = expression.args[0];
    if (payload && payload.kind !== "string") validateScalarOperand(walkExpression(payload, scope), payload.span, diagnostics);
    return { kind: "scalar", valueType: "float" };
  }
  if (isVectorMathBuiltin(callName)) {
    const vectorMath = validateVectorMathBuiltin(expression, callName, diagnostics, walkExpression, scope);
    if (vectorMath) return vectorMath;
  }
  const vibMinMax = validateVibMinMaxCall(expression, callName, scope, diagnostics, walkExpression);
  if (vibMinMax) return vibMinMax;
  const intrinsic = CUDA_INTRINSICS_BY_NAME.get(callName);
  if (intrinsic) {
    for (const feature of intrinsic.requiredFeatures ?? []) requiredFeatures.add(feature);
    let argumentValueType: ValueType | undefined;
    for (const arg of expression.args) {
      const info = walkExpression(arg, scope);
      validateScalarOperand(info, arg.span, diagnostics);
      argumentValueType ??= info.valueType;
    }
    return {
      kind: "scalar",
      valueType: intrinsic.returnType === "argument1" ? argumentValueType : intrinsic.returnType,
    };
  }
  if (isSyncthreadsPredicateBuiltin(callName)) {
    const predicate = expression.args[0];
    if (predicate) validateScalarOperand(walkExpression(predicate, scope), predicate.span, diagnostics);
    return { kind: "scalar", valueType: "int" };
  }
  if (callName === "__activemask") {
    requiredFeatures.add("subgroups");
    return { kind: "scalar", valueType: "uint" };
  }
  if (isShuffleBuiltin(callName) || isVoteBuiltin(callName)) {
    requiredFeatures.add("subgroups");
    let valueType: ValueType | undefined;
    for (const [index, arg] of expression.args.entries()) {
      const info = walkExpression(arg, scope);
      validateScalarOperand(info, arg.span, diagnostics);
      if (index === 1) valueType = info.valueType;
    }
    return { kind: "scalar", valueType: isVoteBuiltin(callName) ? "uint" : valueType };
  }
  if (isMaskedWarpReductionBuiltin(callName)) {
    requiredFeatures.add("subgroups");
    let valueType: ValueType | undefined;
    for (const [index, arg] of expression.args.entries()) {
      const info = walkExpression(arg, scope);
      validateScalarOperand(info, arg.span, diagnostics);
      if (index === 1) valueType = info.valueType;
    }
    return { kind: "scalar", valueType };
  }
  if (isMaskedWarpBitwiseReductionBuiltin(callName)) {
    requiredFeatures.add("subgroups");
    let valueType: ValueType | undefined;
    for (const [index, arg] of expression.args.entries()) {
      const info = walkExpression(arg, scope);
      validateScalarOperand(info, arg.span, diagnostics);
      if (index === 1) {
        valueType = info.valueType;
        if (valueType !== undefined && valueType !== "int" && valueType !== "uint") {
          diagnostics.push(error("unsupported-subgroup", `${callName} expects an int or uint value operand`, arg.span));
        }
      }
    }
    return { kind: "scalar", valueType };
  }
  if (isWarpReductionBuiltin(callName)) {
    requiredFeatures.add("subgroups");
    const arg = expression.args.length === 2 ? expression.args[1] : expression.args[0];
    if (!arg) return { kind: "unknown" };
    if (expression.args.length === 2) {
      const mask = expression.args[0];
      if (mask) validateScalarOperand(walkExpression(mask, scope), mask.span, diagnostics);
    }
    const info = walkExpression(arg, scope);
    validateScalarOperand(info, arg.span, diagnostics);
    return { kind: "scalar", valueType: warpReductionReturnType(callName, info.valueType) };
  }
  if (isTextureReadCall(callName)) {
    validateTextureRead(expression, callName, scope, diagnostics, walkExpression, compatibilityDiagnosticsReachable);
    if (requiresShaderF16(expression.templateValueType)) requiredFeatures.add("shader-f16");
    return expressionInfoForTextureRead(expression);
  }
  if (callName === "surf1Dread" || callName === "surf2Dread" || callName === "surf2DLayeredread" || callName === "surf3Dread") {
    validateSurf2DRead(expression, callName, scope, diagnostics, walkExpression, compatibilityDiagnosticsReachable);
    if (requiresShaderF16(expression.templateValueType)) requiredFeatures.add("shader-f16");
    const returnForm = callName === "surf1Dread"
      ? expression.args.length <= 2
      : callName === "surf2DLayeredread" || callName === "surf3Dread" ? expression.args.length <= 4 : expression.args.length <= 3;
    return returnForm ? expressionInfoForTextureRead(expression) : { kind: "scalar", valueType: "voidptr" };
  }
  if (callName === "surf2Dwrite" || callName === "surf1Dwrite" || callName === "surf2DLayeredwrite" || callName === "surf3Dwrite") {
    validateSurf2DWrite(expression, callName, scope, diagnostics, walkExpression, compatibilityDiagnosticsReachable);
    return { kind: "scalar", valueType: "float" };
  }
  if (callName === "sizeof" || callName === "alignof") {
    validateSizeof(expression, scope, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "uint" };
  }
  if (callName === "vec_at") {
    const vector = expression.args[0];
    const index = expression.args[1];
    const vectorInfo = vector ? walkExpression(vector, scope) : undefined;
    const vectorType = vectorInfo?.valueType;
    if (!isCudaVectorType(vectorType)) {
      diagnostics.push(error("unsupported-vector-member", "vec_at expects a CUDA vector value", vector?.span ?? expression.span));
    }
    if (index) validateScalarOperand(walkExpression(index, scope), index.span, diagnostics);
    return {
      kind: "scalar",
      valueType: isCudaVectorType(vectorType) ? cudaVectorScalarType(vectorType) : undefined,
    };
  }
  if (isVectorMathBuiltin(callName)) return validateVectorMathBuiltin(expression, callName, diagnostics, walkExpression, scope) ?? { kind: "scalar" };
  if (callName === "deviceAllocate" || callName === "streamOrderedAllocate") {
    validatePoolAllocate(expression, scope, atomicParams, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "voidptr" };
  }
  if (callName === "curand_init") {
    validateCurandInit(expression, diagnostics, walkExpression, scope);
    return { kind: "scalar", valueType: "uint" };
  }
  if (callName === "curand") {
    validateCurandStateAddress(expression, callName, diagnostics, walkExpression, scope);
    return { kind: "scalar", valueType: "uint" };
  }
  if (callName === "curand_uniform4" || callName === "curand_normal4") {
    validateCurandStateAddress(expression, callName, diagnostics, walkExpression, scope);
    return { kind: "vector", valueType: "float4" };
  }
  if (callName === "curand_normal2") {
    validateCurandStateAddress(expression, callName, diagnostics, walkExpression, scope);
    return { kind: "vector", valueType: "float2" };
  }
  if (callName === "curand_log_normal2" || callName === "curand_log_normal4") {
    validateCurandStateAddress(expression, callName, diagnostics, walkExpression, scope);
    for (const arg of expression.args.slice(1)) {
      validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    }
    return { kind: "vector", valueType: callName === "curand_log_normal2" ? "float2" : "float4" };
  }
  if (
    callName === "curand_uniform" ||
    callName === "curand_uniform_double" ||
    callName === "curand_normal" ||
    callName === "curand_normal_double" ||
    callName === "curand_log_normal" ||
    callName === "curand_log_normal_double"
  ) {
    validateCurandStateAddress(expression, callName, diagnostics, walkExpression, scope);
    for (const arg of expression.args.slice(1)) {
      validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    }
    return { kind: "scalar", valueType: "float" };
  }
  if (callName === "curand_poisson") {
    validateCurandStateAddress(expression, callName, diagnostics, walkExpression, scope);
    for (const arg of expression.args.slice(1)) {
      validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    }
    return { kind: "scalar", valueType: "uint" };
  }
  if (callName === "curand_poisson4") {
    validateCurandStateAddress(expression, callName, diagnostics, walkExpression, scope);
    for (const arg of expression.args.slice(1)) {
      validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    }
    return { kind: "vector", valueType: "uint4" };
  }
  if (callName === "skipahead") {
    const count = expression.args[0];
    if (count) validateScalarOperand(walkExpression(count, scope), count.span, diagnostics);
    const state = expression.args[1];
    if (state) {
      const info = walkExpression(state, scope);
      if (info.kind !== "address") {
        diagnostics.push(error("curand-state-address", "skipahead expects a state address as its second argument", state.span));
      }
    }
    return { kind: "scalar", valueType: "uint" };
  }

  for (const arg of expression.args) {
    const info = walkExpression(arg, scope);
    validateScalarOperand(info, arg.span, diagnostics);
  }
  return { kind: "scalar" };
}

function isPrintfArgument(info: ExpressionInfo): boolean {
  return info.kind === "scalar" ||
    info.kind === "complex" ||
    info.kind === "vector" ||
    info.kind === "string" ||
    info.kind === "array" ||
    info.kind === "pointer" ||
    info.kind === "address" ||
    info.kind === "pool-pointer";
}

function validateVectorMinMaxCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  requiredFeatures: Set<string>,
  walkExpression: ExpressionWalker,
): ExpressionInfo | undefined {
  if (callName !== "min" && callName !== "max" && callName !== "fminf" && callName !== "fmaxf") return undefined;
  const infos = expression.args.map((arg) => walkExpression(arg, scope));
  const vectorType = infos.find((info) => info.kind === "vector" && isCudaVectorType(info.valueType))?.valueType;
  if (!isCudaVectorType(vectorType)) return undefined;
  for (const [index, info] of infos.entries()) {
    if (info.kind === "vector") {
      if (info.valueType !== vectorType) {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects matching CUDA vector types`, expression.args[index]?.span ?? expression.span));
      }
    } else {
      validateScalarOperand(info, expression.args[index]?.span ?? expression.span, diagnostics);
    }
  }
  if (cudaVectorScalarType(vectorType) === "half") requiredFeatures.add("shader-f16");
  return { kind: "vector", valueType: vectorType };
}

function validateFrexp(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const value = expression.args[0];
  const exponent = expression.args[1];
  if (value) validateScalarOperand(walkExpression(value, scope), value.span, diagnostics);
  if (!exponent) return;
  const info = validateReadPointerExpression(exponent, scope, diagnostics, walkExpression);
  if (!isMathOutPointerInfo(info)) {
    diagnostics.push(error("unsupported-frexp-exponent", "frexp exponent must be an addressable int pointer", exponent.span));
    return;
  }
  if (info.valueType !== undefined && info.valueType !== "int") {
    diagnostics.push(error("unsupported-frexp-exponent", "frexp exponent must point to int storage", exponent.span));
  }
}

function validateModf(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const value = expression.args[0];
  const intpart = expression.args[1];
  if (value) validateScalarOperand(walkExpression(value, scope), value.span, diagnostics);
  if (!intpart) return;
  const info = validateReadPointerExpression(intpart, scope, diagnostics, walkExpression);
  if (!isMathOutPointerInfo(info)) {
    diagnostics.push(error("unsupported-modf-intpart", "modf integer-part output must be an addressable float pointer", intpart.span));
    return;
  }
  if (info.valueType !== undefined && info.valueType !== "float" && info.valueType !== "double") {
    diagnostics.push(error("unsupported-modf-intpart", "modf integer-part output must point to float storage", intpart.span));
  }
}

function validateSincos(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const value = expression.args[0];
  if (value) validateScalarOperand(walkExpression(value, scope), value.span, diagnostics);
  for (const target of expression.args.slice(1)) {
    const info = validateReadPointerExpression(target, scope, diagnostics, walkExpression);
    if (!isMathOutPointerInfo(info)) {
      diagnostics.push(error("unsupported-sincos-output", "sincos output must be an addressable float pointer", target.span));
      continue;
    }
    if (info.valueType !== undefined && info.valueType !== "float" && info.valueType !== "double") {
      diagnostics.push(error("unsupported-sincos-output", "sincos output must point to float storage", target.span));
    }
  }
}

function validateRemquo(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const left = expression.args[0];
  const right = expression.args[1];
  const quotient = expression.args[2];
  if (left) validateScalarOperand(walkExpression(left, scope), left.span, diagnostics);
  if (right) validateScalarOperand(walkExpression(right, scope), right.span, diagnostics);
  if (!quotient) return;
  const info = validateReadPointerExpression(quotient, scope, diagnostics, walkExpression);
  if (!isMathOutPointerInfo(info)) {
    diagnostics.push(error("unsupported-remquo-quotient", "remquo quotient output must be an addressable int pointer", quotient.span));
    return;
  }
  if (info.valueType !== undefined && info.valueType !== "int") {
    diagnostics.push(error("unsupported-remquo-quotient", "remquo quotient output must point to int storage", quotient.span));
  }
}

function validateVibMinMaxCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string | undefined,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): ExpressionInfo | undefined {
  if (!callName || !isVibMinMaxCallName(callName)) return undefined;
  const [left, right] = expression.args;
  if (left) validateScalarOperand(walkExpression(left, scope), left.span, diagnostics);
  if (right) validateScalarOperand(walkExpression(right, scope), right.span, diagnostics);
  const predicateArgs = expression.args.slice(2);
  for (const target of predicateArgs) {
    const info = validateReadPointerExpression(target, scope, diagnostics, walkExpression);
    if (!isMathOutPointerInfo(info)) {
      diagnostics.push(error("unsupported-call", `${callName} predicate output must be an addressable bool pointer`, target.span));
      continue;
    }
    if (info.valueType !== undefined && info.valueType !== "bool") {
      diagnostics.push(error("unsupported-call", `${callName} predicate output must point to bool storage`, target.span));
    }
  }
  return { kind: "scalar", valueType: callName.includes("_s32") ? "int" : "uint" };
}

function isMathOutPointerInfo(info: ExpressionInfo): boolean {
  return info.kind === "address" || info.kind === "pointer" || info.kind === "pool-pointer" || info.kind === "unknown";
}

function isVibMinMaxCallName(name: string): boolean {
  return name === "__vibmax_s32" ||
    name === "__vibmin_s32" ||
    name === "__vibmax_u32" ||
    name === "__vibmin_u32" ||
    name === "__vibmax_s16x2" ||
    name === "__vibmin_s16x2" ||
    name === "__vibmax_u16x2" ||
    name === "__vibmin_u16x2";
}

function isSincosCallName(name: string | undefined): boolean {
  return name === "sincos" || name === "sincosf" || name === "__sincosf" || name === "sincospi" || name === "sincospif";
}

function isNanPayloadCallName(name: string | undefined): boolean {
  return name === "nan" || name === "nanf" || name === "__builtin_nan" || name === "__builtin_nanf";
}

function validateCudaIntegerRuntimeQuery(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const target = expression.args[0];
  if (!target) return;
  if (callName === "cudaThreadExchangeStreamCaptureMode") {
    validateRuntimeQueryPointerTarget(callName, target, "int", scope, diagnostics, walkExpression);
    return;
  }
  if (callName === "cudaStreamCreate" || callName === "cudaStreamCreateWithFlags" || callName === "cudaStreamCreateWithPriority" || callName === "cudaEventCreate" || callName === "cudaEventCreateWithFlags") {
    validateRuntimeQueryPointerTarget(callName, target, "uint", scope, diagnostics, walkExpression);
    for (const arg of expression.args.slice(1)) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return;
  }
  if (callName === "cudaStreamGetFlags" || callName === "cudaStreamGetPriority") {
    validateScalarOperand(walkExpression(target, scope), target.span, diagnostics);
    const streamTarget = expression.args[1];
    if (!streamTarget) return;
    const streamInfo = validateReadPointerExpression(streamTarget, scope, diagnostics, walkExpression);
    const expected = callName === "cudaStreamGetFlags" ? "uint" : "int";
    if (streamInfo.kind !== "address" && streamInfo.kind !== "pointer" && streamInfo.kind !== "unknown") {
      diagnostics.push(error("unsupported-cuda-runtime", `${callName} expects a ${expected} pointer target`, streamTarget.span));
    } else if (streamInfo.valueType !== undefined && streamInfo.valueType !== expected) {
      diagnostics.push(error("unsupported-cuda-runtime", `${callName} target must point to ${expected} storage`, streamTarget.span));
    }
    return;
  }
  if (callName === "cudaStreamGetDevice" || callName === "cudaStreamGetId" || callName === "cudaStreamIsCapturing") {
    validateScalarOperand(walkExpression(target, scope), target.span, diagnostics);
    const streamTarget = expression.args[1];
    if (!streamTarget) return;
    const expected = callName === "cudaStreamGetId" ? "uint" : "int";
    validateRuntimeQueryPointerTarget(callName, streamTarget, expected, scope, diagnostics, walkExpression);
    return;
  }
  if (callName === "cudaStreamGetCaptureInfo" || callName === "cudaStreamGetCaptureInfo_v2") {
    validateScalarOperand(walkExpression(target, scope), target.span, diagnostics);
    for (const [index, arg] of expression.args.entries()) {
      if (index === 0 || !arg || isNullPointerExpression(arg)) continue;
      validateRuntimeQueryPointerTarget(callName, arg, cudaStreamGetCaptureInfoTargetType(index), scope, diagnostics, walkExpression);
    }
    return;
  }
  if (callName === "cudaStreamEndCapture") {
    validateScalarOperand(walkExpression(target, scope), target.span, diagnostics);
    const graphTarget = expression.args[1];
    if (graphTarget && !isNullPointerExpression(graphTarget)) {
      validateRuntimeQueryPointerTarget(callName, graphTarget, "uint", scope, diagnostics, walkExpression);
    }
    return;
  }
  if (callName === "cudaGraphCreate") {
    validateRuntimeQueryPointerTarget(callName, target, "uint", scope, diagnostics, walkExpression);
    const flags = expression.args[1];
    if (flags) validateScalarOperand(walkExpression(flags, scope), flags.span, diagnostics);
    return;
  }
  if (callName === "cudaGraphInstantiate" || callName === "cudaGraphInstantiateWithFlags") {
    validateRuntimeQueryPointerTarget(callName, target, "uint", scope, diagnostics, walkExpression);
    const graph = expression.args[1];
    if (graph) validateScalarOperand(walkExpression(graph, scope), graph.span, diagnostics);
    if (callName === "cudaGraphInstantiate") {
      const errorNode = expression.args[2];
      if (errorNode && !isNullPointerExpression(errorNode)) {
        validateRuntimeQueryPointerTarget(callName, errorNode, "uint", scope, diagnostics, walkExpression);
      }
      for (const arg of expression.args.slice(3)) walkExpression(arg, scope);
    } else {
      const flags = expression.args[2];
      if (flags) validateScalarOperand(walkExpression(flags, scope), flags.span, diagnostics);
    }
    return;
  }
  if (callName === "cudaGraphExecUpdate") {
    validateScalarOperand(walkExpression(target, scope), target.span, diagnostics);
    const graph = expression.args[1];
    if (graph) validateScalarOperand(walkExpression(graph, scope), graph.span, diagnostics);
    const errorNode = expression.args[2];
    if (errorNode && !isNullPointerExpression(errorNode)) {
      validateRuntimeQueryPointerTarget(callName, errorNode, "uint", scope, diagnostics, walkExpression);
    }
    const updateResult = expression.args[3];
    if (updateResult && !isNullPointerExpression(updateResult)) {
      validateRuntimeQueryPointerTarget(callName, updateResult, "uint", scope, diagnostics, walkExpression);
    }
    return;
  }
  if (callName === "cudaOccupancyMaxActiveBlocksPerMultiprocessor" || callName === "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") {
    validateRuntimeQueryPointerTarget(callName, target, "int", scope, diagnostics, walkExpression);
    for (const arg of expression.args.slice(2)) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return;
  }
  if (callName === "cudaOccupancyMaxPotentialBlockSize" || callName === "cudaOccupancyMaxPotentialBlockSizeWithFlags") {
    validateRuntimeQueryPointerTarget(callName, target, "int", scope, diagnostics, walkExpression);
    const blockSizeTarget = expression.args[1];
    if (blockSizeTarget) validateRuntimeQueryPointerTarget(callName, blockSizeTarget, "int", scope, diagnostics, walkExpression);
    for (const arg of expression.args.slice(3)) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return;
  }
  if (callName === "cudaOccupancyAvailableDynamicSMemPerBlock") {
    validateRuntimeQueryPointerTarget(callName, target, "uint", scope, diagnostics, walkExpression);
    for (const arg of expression.args.slice(2)) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return;
  }
  const info = validateReadPointerExpression(target, scope, diagnostics, walkExpression);
  if (info.kind !== "address" && info.kind !== "pointer" && info.kind !== "unknown") {
    diagnostics.push(error("unsupported-cuda-runtime", `${callName} expects an int pointer target`, target.span));
    return;
  }
  const expectedValueType = callName === "cudaDeviceGetLimit" || callName === "cudaThreadGetLimit" || callName === "cudaMemGetInfo" || callName === "cudaGetDeviceFlags" ? "uint" : "int";
  if (info.valueType !== undefined && info.valueType !== expectedValueType) {
    diagnostics.push(error("unsupported-cuda-runtime", `${callName} target must point to ${expectedValueType} storage`, target.span));
  }
  if (callName === "cudaMemGetInfo") {
    const totalTarget = expression.args[1];
    if (totalTarget) {
      const totalInfo = validateReadPointerExpression(totalTarget, scope, diagnostics, walkExpression);
      if (totalInfo.kind !== "address" && totalInfo.kind !== "pointer" && totalInfo.kind !== "unknown") {
        diagnostics.push(error("unsupported-cuda-runtime", `${callName} expects a uint pointer target`, totalTarget.span));
      } else if (totalInfo.valueType !== undefined && totalInfo.valueType !== "uint") {
        diagnostics.push(error("unsupported-cuda-runtime", `${callName} target must point to uint storage`, totalTarget.span));
      }
    }
    return;
  }
  if (callName === "cudaDeviceGetStreamPriorityRange") {
    const greatestTarget = expression.args[1];
    if (greatestTarget) {
      const greatestInfo = validateReadPointerExpression(greatestTarget, scope, diagnostics, walkExpression);
      if (greatestInfo.kind !== "address" && greatestInfo.kind !== "pointer" && greatestInfo.kind !== "unknown") {
        diagnostics.push(error("unsupported-cuda-runtime", `${callName} expects an int pointer target`, greatestTarget.span));
      } else if (greatestInfo.valueType !== undefined && greatestInfo.valueType !== "int") {
        diagnostics.push(error("unsupported-cuda-runtime", `${callName} target must point to int storage`, greatestTarget.span));
      }
    }
    return;
  }
  for (const arg of expression.args.slice(1)) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
}

function validateRuntimeQueryPointerTarget(
  callName: string,
  target: CudaLiteExpression,
  expectedValueType: Exclude<CudaLiteScalarType, "void">,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const info = validateReadPointerExpression(target, scope, diagnostics, walkExpression);
  if (info.kind !== "address" && info.kind !== "pointer" && info.kind !== "unknown") {
    diagnostics.push(error("unsupported-cuda-runtime", `${callName} expects a ${expectedValueType} pointer target`, target.span));
  } else if (info.valueType !== undefined && info.valueType !== expectedValueType) {
    diagnostics.push(error("unsupported-cuda-runtime", `${callName} target must point to ${expectedValueType} storage`, target.span));
  }
}

function cudaStreamGetCaptureInfoTargetType(index: number): Exclude<CudaLiteScalarType, "void"> {
  return index === 1 ? "int" : "uint";
}

function isNullPointerExpression(expression: CudaLiteExpression): boolean {
  return (expression.kind === "identifier" && expression.name === "NULL") ||
    (expression.kind === "number" && expression.value === 0);
}

function validateCudaEventElapsedTime(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const target = expression.args[0];
  if (target) {
    const info = validateReadPointerExpression(target, scope, diagnostics, walkExpression);
    if (info.kind !== "address" && info.kind !== "pointer" && info.kind !== "unknown") {
      diagnostics.push(error("unsupported-cuda-runtime", "cudaEventElapsedTime expects a float pointer target", target.span));
    } else if (info.valueType !== undefined && info.valueType !== "float") {
      diagnostics.push(error("unsupported-cuda-runtime", "cudaEventElapsedTime target must point to float storage", target.span));
    }
  }
  for (const arg of expression.args.slice(1)) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
}

function validateReadPointerOperand(
  expression: CudaLiteExpression,
  scope: Scope,
  walkExpression: ExpressionWalker,
): ExpressionInfo {
  if (expression.kind === "unary" && expression.operator === "&") {
    const info = walkExpression(expression.argument, scope);
    return { kind: "address", valueType: info.valueType, symbol: info.symbol };
  }
  return walkExpression(expression, scope);
}

function validateCacheHintStoreValue(
  callName: string,
  targetInfo: ExpressionInfo | undefined,
  value: CudaLiteExpression,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const valueInfo = walkExpression(value, scope);
  const targetType = targetInfo?.valueType;
  if (!isCudaVectorType(targetType)) {
    validateScalarOperand(valueInfo, value.span, diagnostics);
    return;
  }
  if (valueInfo.kind === "unknown") return;
  if (valueInfo.kind !== "vector" || valueInfo.valueType !== targetType) {
    diagnostics.push(error("unsupported-vector-assignment", `${callName} expects a ${targetType} value`, value.span));
  }
}

function isVectorMathBuiltin(name: string): boolean {
  return name === "dot" || name === "length" || name === "normalize" || name === "cross" || name === "lerp";
}

function validateVectorMathBuiltin(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
): ExpressionInfo | undefined {
  const infos = expression.args.map((arg) => walkExpression(arg, scope));
  if (callName === "lerp") {
    const leftType = infos[0]?.kind === "vector" && isCudaVectorType(infos[0].valueType) ? infos[0].valueType : undefined;
    const rightType = infos[1]?.kind === "vector" && isCudaVectorType(infos[1].valueType) ? infos[1].valueType : undefined;
    const vectorType = leftType ?? rightType;
    if (!vectorType) return undefined;
    for (const [index, info] of infos.entries()) {
      const arg = expression.args[index]!;
      if (index < 2) {
        if (info.kind !== "vector" && info.kind !== "unknown") {
          diagnostics.push(error("unsupported-vector-argument", "lerp expects matching CUDA vector endpoints", arg.span));
        } else if (info.kind === "vector" && info.valueType !== vectorType) {
          diagnostics.push(error("unsupported-vector-argument", "lerp expects matching CUDA vector endpoints", arg.span));
        }
      } else {
        validateScalarOperand(info, arg.span, diagnostics);
      }
    }
    return { kind: "vector", valueType: vectorType };
  }
  for (const [index, info] of infos.entries()) {
    const arg = expression.args[index]!;
    if (info.kind !== "vector" && info.kind !== "unknown") {
      diagnostics.push(error("unsupported-vector-argument", `${callName} expects CUDA vector argument`, arg.span));
      continue;
    }
    if (info.kind === "vector" && (!isCudaVectorType(info.valueType) || cudaVectorScalarType(info.valueType) !== "float")) {
      diagnostics.push(error("unsupported-vector-argument", `${callName} expects float CUDA vector argument`, arg.span));
    }
  }
  const firstType = infos[0]?.valueType;
  const secondType = infos[1]?.valueType;
  if ((callName === "dot" || callName === "cross") && isCudaVectorType(firstType) && isCudaVectorType(secondType) && firstType !== secondType) {
    diagnostics.push(error("unsupported-vector-argument", `${callName} expects matching CUDA vector types`, expression.span));
  }
  if (callName === "cross" && firstType !== undefined && firstType !== "float3") {
    diagnostics.push(error("unsupported-vector-argument", "cross expects float3 arguments", expression.span));
  }
  if (callName === "normalize") {
    return { kind: "vector", valueType: isCudaVectorType(firstType) ? firstType : undefined };
  }
  return { kind: "scalar", valueType: "float" };
}

function isPointerIdentityCall(callName: string | undefined): boolean {
  return callName === "__builtin_assume_aligned" || callName === "ct::assume_aligned";
}

function isAddressSpacePredicateCall(callName: string | undefined): callName is "__isGlobal" | "__isShared" | "__isConstant" | "__isLocal" {
  return callName === "__isGlobal" || callName === "__isShared" || callName === "__isConstant" || callName === "__isLocal";
}

function validateAddressSpacePredicateCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const target = expression.args[0];
  if (!target) return;
  const info = walkExpression(target, scope);
  if (!isAddressSpacePredicateTarget(info)) {
    diagnostics.push(error("unsupported-call", `${callName} expects a modeled pointer, array, or address expression`, target.span));
  }
}

function isAddressSpacePredicateTarget(info: ExpressionInfo): boolean {
  if (info.kind === "pointer" || info.kind === "pool-pointer" || info.kind === "address" || info.kind === "array" || info.kind === "unknown") {
    return true;
  }
  if (info.kind !== "scalar") return false;
  return info.symbol?.kind === "shared" ||
    info.symbol?.kind === "constant" ||
    info.symbol?.kind === "device-global" ||
    info.symbol?.kind === "local";
}

function validatePointerIdentityCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): ExpressionInfo {
  const pointer = expression.args[0];
  if (!pointer) return { kind: "unknown" };
  const info = walkExpression(pointer, scope);
  if (info.kind !== "pointer" && info.kind !== "pool-pointer" && info.kind !== "address" && info.kind !== "unknown") {
    diagnostics.push(error("unsupported-device-pointer-param", `${callName} expects a pointer expression`, pointer.span));
  }
  for (const arg of expression.args.slice(1)) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
  return info.kind === "pointer" || info.kind === "pool-pointer"
    ? info
    : { kind: "pointer", valueType: info.valueType ?? "float", symbol: info.symbol };
}

function isHalf2VectorIntrinsic(name: string): boolean {
  return name === "__habs2" ||
    name === "__hceil2" ||
    name === "__hfloor2" ||
    name === "__hneg2" ||
    name === "__hrcp2" ||
    name === "__hrsqrt2" ||
    name === "__hsqrt2" ||
    name === "__htrunc2" ||
    name === "__hisnan2" ||
    isHalf2VectorComparisonIntrinsic(name) ||
    name === "__hadd2" ||
    name === "__hadd2_rn" ||
    name === "__hadd2_sat" ||
    name === "__hsub2" ||
    name === "__hsub2_rn" ||
    name === "__hsub2_sat" ||
    name === "__hmul2" ||
    name === "__hmul2_rn" ||
    name === "__hmul2_sat" ||
    name === "__hfma2" ||
    name === "__hfma2_rn" ||
    name === "__hfma2_sat" ||
    name === "__hmin2" ||
    name === "__hmax2" ||
    name === "__hmin2_nan" ||
    name === "__hmax2_nan";
}

function isBf162VectorIntrinsic(name: string): boolean {
  return isBf162VectorArithmeticIntrinsic(name) ||
    isBf162VectorComparisonIntrinsic(name) ||
    name === "__hisnan2" ||
    name === "__hmin2" ||
    name === "__hmax2" ||
    name === "__hmin2_nan" ||
    name === "__hmax2_nan";
}

function isBf162VectorArithmeticIntrinsic(name: string): boolean {
  return isBf162UnaryMathIntrinsic(name) ||
    name === "__habs2" ||
    name === "__hneg2" ||
    name === "__hadd2" ||
    name === "__hadd2_rn" ||
    name === "__hadd2_sat" ||
    name === "__hsub2" ||
    name === "__hsub2_rn" ||
    name === "__hsub2_sat" ||
    name === "__hmul2" ||
    name === "__hmul2_rn" ||
    name === "__hmul2_sat" ||
    name === "__h2div" ||
    name === "__hfma2" ||
    name === "__hfma2_rn" ||
    name === "__hfma2_sat" ||
    name === "__hfma2_relu" ||
    name === "__hcmadd";
}

function isBf162OnlyVectorIntrinsic(name: string): boolean {
  return isBf162UnaryMathIntrinsic(name) || name === "__h2div" || name === "__hfma2_relu" || name === "__hcmadd";
}

function isBf162UnaryMathIntrinsic(name: string): boolean {
  return name === "h2ceil" ||
    name === "h2floor" ||
    name === "h2rcp" ||
    name === "h2rsqrt" ||
    name === "h2sqrt" ||
    name === "h2trunc" ||
    name === "h2exp" ||
    name === "h2exp2" ||
    name === "h2exp10" ||
    name === "h2log" ||
    name === "h2log2" ||
    name === "h2log10" ||
    name === "h2sin" ||
    name === "h2cos" ||
    name === "h2tanh" ||
    name === "h2tanh_approx" ||
    name === "h2rint";
}

function isBf162VectorComparisonIntrinsic(name: string): boolean {
  return isHalf2VectorComparisonIntrinsic(name);
}

function isBf162ComparisonMaskIntrinsic(name: string): boolean {
  return isHalf2ComparisonMaskIntrinsic(name);
}

function isBf162BooleanComparisonIntrinsic(name: string): boolean {
  return isHalf2BooleanComparisonIntrinsic(name);
}

function isHalf2VectorComparisonIntrinsic(name: string): boolean {
  return name === "__heq2" ||
    name === "__hne2" ||
    name === "__hgt2" ||
    name === "__hge2" ||
    name === "__hlt2" ||
    name === "__hle2" ||
    name === "__hequ2" ||
    name === "__hneu2" ||
    name === "__hgtu2" ||
    name === "__hgeu2" ||
    name === "__hltu2" ||
    name === "__hleu2";
}

function isHalf2ComparisonMaskIntrinsic(name: string): boolean {
  return name === "__heq2_mask" ||
    name === "__hne2_mask" ||
    name === "__hgt2_mask" ||
    name === "__hge2_mask" ||
    name === "__hlt2_mask" ||
    name === "__hle2_mask" ||
    name === "__hequ2_mask" ||
    name === "__hneu2_mask" ||
    name === "__hgtu2_mask" ||
    name === "__hgeu2_mask" ||
    name === "__hltu2_mask" ||
    name === "__hleu2_mask";
}

function isHalf2BooleanComparisonIntrinsic(name: string): boolean {
  return name === "__hbeq2" ||
    name === "__hbne2" ||
    name === "__hbgt2" ||
    name === "__hbge2" ||
    name === "__hblt2" ||
    name === "__hble2" ||
    name === "__hbequ2" ||
    name === "__hbneu2" ||
    name === "__hbgtu2" ||
    name === "__hbgeu2" ||
    name === "__hbltu2" ||
    name === "__hbleu2";
}

function isBfloat16ScalarArithmetic(name: string): boolean {
  return name === "__habs" ||
    name === "__hceil" ||
    name === "__hfloor" ||
    name === "__hrcp" ||
    name === "__hrsqrt" ||
    name === "hrsqrt" ||
    name === "__hsqrt" ||
    name === "__htrunc" ||
    name === "__hneg" ||
    name === "hexp" ||
    name === "__hadd" ||
    name === "__hadd_rn" ||
    name === "__hadd_sat" ||
    name === "__hsub" ||
    name === "__hsub_rn" ||
    name === "__hsub_sat" ||
    name === "__hmul" ||
    name === "__hmul_rn" ||
    name === "__hmul_sat" ||
    name === "__hdiv" ||
    name === "__hdiv_rn" ||
    name === "__hfma" ||
    name === "__hfma_rn" ||
    name === "__hfma_sat" ||
    name === "__hfma_relu" ||
    name === "__hmin" ||
    name === "__hmax" ||
    name === "__hmin_nan" ||
    name === "__hmax_nan";
}

function isBfloat16ScalarPredicate(name: string): boolean {
  return name === "__hisnan" ||
    name === "__hisinf" ||
    name === "__heq" ||
    name === "__hne" ||
    name === "__hgt" ||
    name === "__hge" ||
    name === "__hlt" ||
    name === "__hle" ||
    name === "__hequ" ||
    name === "__hneu" ||
    name === "__hgtu" ||
    name === "__hgeu" ||
    name === "__hltu" ||
    name === "__hleu";
}

function isFillRegsBuiltin(name: string): boolean {
  return name === "fill_1D_regs" || name === "fill_2D_regs" || name === "fill_3D_regs";
}

function isCpAsyncCopyCall(name: string): boolean {
  return name === "CP_ASYNC_CA" || name === "CP_ASYNC_CG" || name === "CP_ASYNC_BULK";
}

function isCpAsyncFenceCall(name: string): boolean {
  return name === "CP_ASYNC_COMMIT_GROUP" ||
    name === "CP_ASYNC_WAIT_ALL" ||
    name === "CP_ASYNC_WAIT_GROUP" ||
    name === "CP_ASYNC_BULK_COMMIT_GROUP" ||
    name === "CP_ASYNC_BULK_WAIT_ALL" ||
    name === "CP_ASYNC_BULK_WAIT_GROUP";
}

function validateCpAsyncCopy(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const [dst, src, bytes] = expression.args;
  if (dst !== undefined) {
    const info = walkExpression(dst, scope);
    if (info.kind !== "pointer" && info.kind !== "address" && info.kind !== "array" && info.kind !== "scalar" && info.kind !== "unknown") {
      diagnostics.push(error("unsupported-cache-hint-address", "cp.async destination expects shared pointer or byte offset", dst.span));
    }
  }
  if (src !== undefined) {
    const info = validateReadPointerExpression(src, scope, diagnostics, walkExpression);
    if (info.kind !== "pointer" && info.kind !== "address" && info.kind !== "array" && info.kind !== "unknown") {
      diagnostics.push(error("unsupported-cache-hint-address", "cp.async source expects pointer expression", src.span));
    }
  }
  if (bytes !== undefined) validateScalarOperand(walkExpression(bytes, scope), bytes.span, diagnostics);
}

function validateFillRegs(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const target = expression.args[0];
  const value = expression.args[1];
  const info = target ? walkExpression(target, scope) : undefined;
  if (info?.kind !== "array" && info?.kind !== "unknown") {
    diagnostics.push(error("unsupported-local-array-fill", "fill_*D_regs expects a fixed local array", target?.span ?? expression.span));
  }
  if (value) {
    const info = walkExpression(value, scope);
    if (info.kind !== "vector") validateScalarOperand(info, value.span, diagnostics);
  }
}

function validateHostManagedRuntimeNoopCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  walkExpression: ExpressionWalker,
  scope: Scope,
): void {
  for (const arg of expression.args) walkExpression(arg, scope);
}

function validateRuntimeCopyCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
  options: CudaLiteAnalyzeOptions,
  compatibilityDiagnosticsReachable: boolean,
): void {
  const referenceRuntime = options.referenceCudaRuntime || options.referenceDynamicParallelism;
  if (compatibilityDiagnosticsReachable) {
    diagnostics.push({
      ...error("unsupported-cuda-runtime", `${callName}() requires CUDA runtime copy orchestration`, expression.span),
      severity: referenceRuntime ? "warning" : "error",
    });
  }
  const copy2d = callName === "cudaMemcpy2D" || callName === "cudaMemcpy2DAsync";
  const memset2d = callName === "cudaMemset2D" || callName === "cudaMemset2DAsync";
  const symbolMemset = callName === "cudaMemsetToSymbol" || callName === "cudaMemsetToSymbolAsync";
  const peerCopy = callName === "cudaMemcpyPeer" || callName === "cudaMemcpyPeerAsync";
  const symbolCopy = isCudaRuntimeSymbolCopyCall(callName);
  const dst = expression.args[0];
  const src = memset2d || symbolMemset ? undefined : expression.args[copy2d ? 2 : peerCopy ? 2 : 1];
  if (dst) walkExpression(dst, scope);
  if (src) walkExpression(src, scope);
  const scalarArgs = memset2d
    ? [expression.args[1], expression.args[2], expression.args[3], expression.args[4]]
    : symbolMemset
    ? [expression.args[1], expression.args[2], expression.args[3]]
    : symbolCopy
    ? [expression.args[2], expression.args[3]]
    : copy2d
    ? [expression.args[1], expression.args[3], expression.args[4], expression.args[5]]
    : [expression.args[peerCopy ? 4 : 2]];
  for (const arg of scalarArgs) {
    if (arg) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
  }
  const kindArg = expression.args[symbolCopy ? 4 : copy2d ? 6 : 3];
  const validatesCopyKind = callName === "cudaMemcpy" || callName === "cudaMemcpyAsync" || copy2d || symbolCopy;
  if (compatibilityDiagnosticsReachable && validatesCopyKind && !(symbolCopy && kindArg === undefined) && !supportedCudaMemcpyKind(kindArg)) {
    diagnostics.push(error(
      "unsupported-cuda-runtime-copy-kind",
      `${callName} supports modeled cudaMemcpyHostToHost/HostToDevice/DeviceToHost/DeviceToDevice/Default copies only`,
      kindArg?.span ?? expression.span,
    ));
  }
}

function validateCudaGraphSetConditionalCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
): void {
  const handle = expression.args[0];
  const value = expression.args[1];
  if (handle) validateScalarOperand(walkExpression(handle, scope), handle.span, diagnostics);
  if (value) validateScalarOperand(walkExpression(value, scope), value.span, diagnostics);
  diagnostics.push(warning(
    "cuda-graph-conditional-host-orchestration",
    "cudaGraphSetConditional updates CUDA graph scheduler state; BrowserGrad single-kernel execution validates the call and treats graph body orchestration as host-managed",
    expression.span,
  ));
}

function supportedCudaMemcpyKind(expression: CudaLiteExpression | undefined): boolean {
  if (!expression) return false;
  if (expression.kind === "identifier") {
    return expression.name === "cudaMemcpyHostToHost" ||
      expression.name === "cudaMemcpyHostToDevice" ||
      expression.name === "cudaMemcpyDeviceToHost" ||
      expression.name === "cudaMemcpyDeviceToDevice" ||
      expression.name === "cudaMemcpyDefault";
  }
  if (expression.kind === "number") {
    return Number.isInteger(expression.value) && expression.value >= 0 && expression.value <= 4;
  }
  return false;
}

function validateVectorConstructorArgs(
  vectorConstructor: CudaLiteVectorType,
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  let laneCount = 0;
  for (const arg of expression.args) {
    const info = walkExpression(arg, scope);
    if (info.kind === "vector" && isCudaVectorType(info.valueType)) {
      laneCount += cudaVectorLaneCount(info.valueType);
      continue;
    }
    validateScalarOperand(info, arg.span, diagnostics);
    laneCount++;
  }
  const targetLanes = cudaVectorLaneCount(vectorConstructor);
  if (expression.args.length > 1 && laneCount > targetLanes) {
    diagnostics.push(error("invalid-call-arity", `${expressionName(expression.callee) ?? "vector constructor"} provides ${laneCount} lanes for ${targetLanes}-lane ${vectorConstructor}`, expression.span));
  }
}

function validateDeviceFunctionCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  symbol: SymbolInfo,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
): ExpressionInfo {
  const overload = resolveDeviceFunctionOverload(symbol, expression.args.length);
  const fnParams = overload?.params ?? symbol.params ?? [];
  if (expression.args.length !== fnParams.length) {
    diagnostics.push(error(
      "invalid-call-arity",
      deviceFunctionArityMessage(symbol),
      expression.span,
    ));
  }
  for (const [index, arg] of expression.args.entries()) {
    const param = fnParams[index];
    if (param?.cooperativeGroupKind !== undefined) {
      validateCooperativeGroupArgument(arg, param, scope, diagnostics);
      if (param.cooperativeGroupKind === "thread" && deviceFunctionUsesGroupReduce(overload ?? symbol, param.name)) {
        const name = rootIdentifier(arg);
        const group = name ? lookupSymbol(name, scope, arg.span) : undefined;
        if (group?.kind === "cooperative-group" && group.groupKind !== "tile") {
          diagnostics.push(error("unsupported-cooperative-groups", `device parameter '${param.name}' is reduced and requires a tile cooperative group`, arg.span));
        }
      }
      continue;
    }
    if (param?.valueType === "texture2d" || param?.valueType === "surface2d") {
      validateDeviceResourceArgument(arg, param, scope, diagnostics, walkExpression);
      continue;
    }
    if (param?.pointer) {
      validateDevicePointerArgument(arg, param, scope, diagnostics, walkExpression);
      continue;
    }
    const info = walkExpression(arg, scope);
    if (isCudaVectorType(param?.valueType)) {
      if (arg.kind === "initializer") {
        continue;
      }
      if (isFloat2ComplexCompatible(param.valueType, info)) {
        continue;
      }
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", `device parameter '${param.name}' expects ${param.valueType}`, arg.span));
      }
      continue;
    }
    validateScalarOperand(info, arg.span, diagnostics);
  }
  const returnType = overload?.returnType ?? symbol.returnType;
  if (returnType === undefined || returnType === "void") return { kind: "unknown" };
  return isCudaVectorType(returnType)
    ? { kind: "vector", valueType: returnType }
    : { kind: "scalar", valueType: returnType };
}

function resolveDeviceFunctionOverload(
  symbol: SymbolInfo,
  argCount: number,
): CudaLiteDeviceFunction | undefined {
  const overloads = symbol.overloads ?? [];
  if (overloads.length === 0) return undefined;
  return overloads.find((fn) => fn.params.length === argCount) ?? overloads[0];
}

function deviceFunctionArityMessage(symbol: SymbolInfo): string {
  const overloads = symbol.overloads ?? [];
  if (overloads.length <= 1) {
    const count = symbol.params?.length ?? 0;
    return `${symbol.name} expects ${count} argument${count === 1 ? "" : "s"}`;
  }
  const counts = [...new Set(overloads.map((fn) => fn.params.length))].sort((a, b) => a - b);
  return `${symbol.name} expects ${counts.join(" or ")} arguments`;
}

function isFloat2ComplexCompatible(expected: CudaLiteScalarType | undefined, info: ExpressionInfo): boolean {
  return (expected === "float2" || expected === "complex64") && (info.kind === "complex" || (info.kind === "vector" && info.valueType === "float2"));
}

function deviceFunctionUsesGroupReduce(
  symbol: { readonly body?: readonly CudaLiteStatement[] },
  paramName: string,
): boolean {
  let found = false;
  walkCudaLiteExpressions(symbol.body ?? [], (expression) => {
    if (found) return;
    if (expression.kind === "call" && expressionName(expression.callee)?.endsWith("::reduce")) {
      const groupArg = expression.args[0];
      if (groupArg?.kind === "identifier" && groupArg.name === paramName) found = true;
    }
  });
  return found;
}

function symbolForParam(param: CudaLiteParam, kind: "param" | "local"): SymbolInfo {
  if (param.cooperativeGroupKind !== undefined) {
    return {
      name: param.name,
      kind: "cooperative-group",
      groupKind: param.cooperativeGroupKind,
      ...(param.tileSize === undefined ? {} : { tileSize: param.tileSize }),
      span: param.span,
    };
  }
  return {
    name: param.name,
    kind,
    valueType: param.valueType,
    pointer: param.pointer,
    constant: param.constant,
    span: param.span,
  };
}

function validateCooperativeGroupArgument(
  arg: CudaLiteExpression,
  param: CudaLiteParam,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
): void {
  const name = rootIdentifier(arg);
  const symbol = name ? lookupSymbol(name, scope, arg.span) : undefined;
  if (symbol?.kind !== "cooperative-group") {
    diagnostics.push(error("unsupported-cooperative-groups", `device parameter '${param.name}' expects a cooperative group argument`, arg.span));
    return;
  }
  if (param.cooperativeGroupKind === "thread") {
    if (symbol.groupKind === "grid") {
      diagnostics.push(error("unsupported-cooperative-groups", `device parameter '${param.name}' expects block or tile cooperative group`, arg.span));
    }
    return;
  }
  if (param.cooperativeGroupKind !== symbol.groupKind) {
    diagnostics.push(error("unsupported-cooperative-groups", `device parameter '${param.name}' expects ${param.cooperativeGroupKind} cooperative group`, arg.span));
  }
}

function validateDeviceResourceArgument(
  arg: CudaLiteExpression,
  param: CudaLiteParam,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const info = walkExpression(arg, scope);
  if (param.valueType === "texture2d") {
    if (info.kind !== "texture" && info.kind !== "unknown") {
      diagnostics.push(error("unsupported-texture", `device parameter '${param.name}' expects a texture argument`, arg.span));
    }
    return;
  }
  if (param.valueType === "surface2d" && info.kind !== "surface" && info.kind !== "unknown") {
    diagnostics.push(error("unsupported-surface", `device parameter '${param.name}' expects a surface argument`, arg.span));
  }
}

function validateDevicePointerArgument(
  arg: CudaLiteExpression,
  param: CudaLiteParam,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const info = validateReadPointerExpression(arg, scope, diagnostics, walkExpression);
  const root = rootIdentifier(arg);
  const rootSymbol = root ? lookupSymbol(root, scope, arg.span) : undefined;
  const sharedArrayDecay = rootSymbol?.kind === "shared" &&
    rootSymbol.dimensions !== undefined &&
    info.kind === "array";
  const constantArrayDecay = rootSymbol?.kind === "constant" &&
    rootSymbol.dimensions !== undefined &&
    info.kind === "array";
  const globalArrayDecay = rootSymbol?.kind === "device-global" &&
    rootSymbol.dimensions !== undefined &&
    info.kind === "array";
  const localArrayDecay = rootSymbol?.kind === "local" &&
    rootSymbol.dimensions !== undefined &&
    info.kind === "array";
  if (info.kind !== "pointer" && info.kind !== "address" && info.kind !== "unknown" && !sharedArrayDecay && !constantArrayDecay && !globalArrayDecay && !localArrayDecay) {
    diagnostics.push(error("unsupported-device-pointer-param", `device pointer parameter '${param.name}' expects a pointer argument`, arg.span));
    return;
  }
  if (rootSymbol?.kind === "constant" && !(constantArrayDecay && param.constant)) {
    diagnostics.push(error("unsupported-device-pointer-param", `device pointer parameter '${param.name}' expects storage-buffer memory`, arg.span));
  }
  if (rootSymbol?.pointer && rootSymbol.constant && !param.constant) {
    diagnostics.push(error("const-pointer-write", `cannot pass const pointer '${root}' to writable device pointer parameter '${param.name}'`, arg.span));
  }
  const actualValueType = info.valueType ?? rootSymbol?.valueType;
  if (actualValueType && !pointerTypesCompatible(param.valueType, actualValueType, hasExplicitPointerCast(arg))) {
    diagnostics.push(error("unsupported-device-pointer-param", `device pointer parameter '${param.name}' expects ${param.valueType} pointer`, arg.span));
  }
}

function validateCooperativeGroupCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  call: { readonly symbol: SymbolInfo; readonly method: string },
  requiredFeatures: Set<string>,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
  compatibilityDiagnosticsReachable: boolean,
): ExpressionInfo {
  const { symbol, method } = call;
  if (symbol.groupKind === "grid" && method === "sync") {
    if (compatibilityDiagnosticsReachable) {
      diagnostics.push({
        ...error("cuda-grid-sync-host-orchestration", "grid.sync() requires WebGPU host-orchestrated dispatch phases", expression.span),
        severity: "warning",
      });
    }
    return { kind: "scalar" };
  }
  if (method === "sync") {
    if (expression.args.length !== 0) diagnostics.push(error("invalid-call-arity", `${method} expects 0 arguments`, expression.span));
    return { kind: "scalar" };
  }
  if (method === "size") {
    if (expression.args.length !== 0) diagnostics.push(error("invalid-call-arity", `${method} expects 0 arguments`, expression.span));
    return { kind: "scalar", valueType: "int" };
  }
  if (method === "thread_rank") {
    if (expression.args.length !== 0) diagnostics.push(error("invalid-call-arity", `${method} expects 0 arguments`, expression.span));
    return { kind: "scalar", valueType: "int" };
  }
  if (method === "meta_group_size" || method === "meta_group_rank") {
    if (expression.args.length !== 0) diagnostics.push(error("invalid-call-arity", `${method} expects 0 arguments`, expression.span));
    return { kind: "scalar", valueType: "int" };
  }
  if (method === "shfl" || method === "shfl_down" || method === "shfl_up" || method === "shfl_xor") {
    requiredFeatures.add("subgroups");
    if (expression.args.length !== 2) diagnostics.push(error("invalid-call-arity", `${method} expects 2 arguments`, expression.span));
    let valueType: ValueType | undefined;
    for (const [index, arg] of expression.args.entries()) {
      const info = walkExpression(arg, scope);
      validateScalarOperand(info, arg.span, diagnostics);
      if (index === 0) valueType = info.valueType;
    }
    return { kind: "scalar", valueType };
  }
  if (method === "ballot") {
    requiredFeatures.add("subgroups");
    if (expression.args.length !== 1) diagnostics.push(error("invalid-call-arity", "ballot expects 1 argument", expression.span));
    for (const arg of expression.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return { kind: "scalar", valueType: "uint" };
  }
  if (method === "any" || method === "all") {
    requiredFeatures.add("subgroups");
    if (expression.args.length !== 1) diagnostics.push(error("invalid-call-arity", `${method} expects 1 argument`, expression.span));
    for (const arg of expression.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return { kind: "scalar", valueType: "bool" };
  }
  diagnostics.push(error("unsupported-cooperative-groups", `unsupported cooperative group method '${method}'`, expression.span));
  for (const arg of expression.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
  return { kind: "unknown" };
}

function validateCooperativeNamespaceCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  call: { readonly symbol: SymbolInfo; readonly method: string; readonly groupArg: CudaLiteExpression },
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
  compatibilityDiagnosticsReachable: boolean,
): ExpressionInfo {
  const { symbol, method, groupArg } = call;
  if (method === "sync") {
    if (expression.args.length !== 1) diagnostics.push(error("invalid-call-arity", "cg::sync expects 1 argument", expression.span));
    if (compatibilityDiagnosticsReachable && symbol.groupKind === "grid") {
      diagnostics.push({
        ...error("cuda-grid-sync-host-orchestration", "cg::sync(grid) requires WebGPU host-orchestrated dispatch phases", expression.span),
        severity: "warning",
      });
    }
    return { kind: "scalar" };
  }
  if (method === "reduce") {
    if (expression.args.length !== 3) diagnostics.push(error("invalid-call-arity", "cg::reduce expects 3 arguments", expression.span));
    if (symbol.groupKind !== "tile" && symbol.groupKind !== "thread") {
      diagnostics.push(error("unsupported-cooperative-groups", "cg::reduce currently supports tile-like cooperative groups", groupArg.span));
    }
    const value = expression.args[1];
    if (!value) return { kind: "unknown" };
    const info = walkExpression(value, scope);
    if (info.kind === "vector" && isCudaVectorType(info.valueType)) return { kind: "vector", valueType: info.valueType };
    validateScalarOperand(info, value.span, diagnostics);
    return { kind: "scalar", valueType: info.valueType };
  }
  if (method === "inclusive_scan" || method === "exclusive_scan") {
    if (expression.args.length !== 2 && expression.args.length !== 3) diagnostics.push(error("invalid-call-arity", `cg::${method} expects 2 or 3 arguments`, expression.span));
    if (symbol.groupKind !== "tile" && symbol.groupKind !== "thread" && symbol.groupKind !== "block") {
      diagnostics.push(error("unsupported-cooperative-groups", `cg::${method} currently supports tile-like cooperative groups`, groupArg.span));
    }
    if (expression.args[2] !== undefined) {
      const op = expressionName(expression.args[2]);
      if (op !== undefined && !op.endsWith("::plus")) {
        diagnostics.push(error("unsupported-cooperative-groups", `cg::${method} currently supports plus scans`, expression.args[2].span));
      }
    }
    const value = expression.args[1];
    if (!value) return { kind: "unknown" };
    const info = walkExpression(value, scope);
    validateScalarOperand(info, value.span, diagnostics);
    return { kind: "scalar", valueType: info.valueType };
  }
  diagnostics.push(error("unsupported-cooperative-groups", `unsupported cooperative group call 'cg::${method}'`, expression.span));
  return { kind: "unknown" };
}

function cooperativeGroupCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
): { readonly symbol: SymbolInfo; readonly method: string } | undefined {
  const callee = expression.callee;
  if (callee.kind !== "member" || callee.object.kind !== "identifier") return undefined;
  const symbol = lookupSymbol(callee.object.name, scope, callee.object.span);
  if (symbol?.kind !== "cooperative-group") return undefined;
  return { symbol, method: callee.property };
}

function cooperativeNamespaceCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
): { readonly symbol: SymbolInfo; readonly method: string; readonly groupArg: CudaLiteExpression } | undefined {
  const callName = expressionName(expression.callee);
  const method = callName?.endsWith("::sync")
    ? "sync"
    : callName?.endsWith("::reduce")
      ? "reduce"
      : callName?.endsWith("::inclusive_scan")
        ? "inclusive_scan"
        : callName?.endsWith("::exclusive_scan")
          ? "exclusive_scan"
      : undefined;
  if (!method) return undefined;
  const groupArg = expression.args[0];
  if (groupArg?.kind !== "identifier") return undefined;
  const symbol = lookupSymbol(groupArg.name, scope, groupArg.span);
  if (symbol?.kind !== "cooperative-group") return undefined;
  return { symbol, method, groupArg };
}

function validateTextureRead(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  compatibilityDiagnosticsReachable: boolean,
): void {
  const textureDiagnostics = compatibilityDiagnosticsReachable ? diagnostics : [];
  if (!isSupportedTextureReadType(expression.templateValueType)) {
    textureDiagnostics.push(error("unsupported-texture", `${callName} currently supports float/int/uint/uchar, half, float2/3/4, int2/3/4, uint2/3/4, and half2 reads`, expression.span));
  }
  const texture = expression.args[0];
  if (texture?.kind !== "identifier") {
    textureDiagnostics.push(error("unsupported-texture", `${callName} first argument must be a texture reference`, expression.span));
  } else {
    const symbol = lookupSymbol(texture.name, scope, texture.span);
    if (symbol?.kind !== "texture" && symbol?.valueType !== "texture2d") {
      textureDiagnostics.push(error("unsupported-texture", `${callName} target '${texture.name}' is not a texture reference`, texture.span));
    }
  }
  const coords = textureCoordinateArgs(expression, callName);
  for (const coord of coords) {
    validateScalarOperand(walkExpression(coord, scope), coord.span, diagnostics);
  }
}

function validateSurf2DRead(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  compatibilityDiagnosticsReachable: boolean,
): void {
  const surfaceDiagnostics = compatibilityDiagnosticsReachable ? diagnostics : [];
  const is1D = callName === "surf1Dread";
  const hasZ = callName === "surf2DLayeredread" || callName === "surf3Dread";
  const returnForm = is1D ? expression.args.length <= 2 : hasZ ? expression.args.length <= 4 : expression.args.length <= 3;
  const target = returnForm ? undefined : expression.args[0];
  if (target) {
    const lvalue = target.kind === "unary" && target.operator === "&" ? target.argument : target;
    validateLValueExpression(lvalue, scope, diagnostics, walkExpression);
  }
  const surface = returnForm ? expression.args[0] : expression.args[1];
  const surfaceName = surface ? rootIdentifier(surface) : undefined;
  if (!surfaceName) {
    surfaceDiagnostics.push(error("unsupported-texture", returnForm ? "surf2Dread first argument must be a surface reference" : "surf2Dread second argument must be a surface reference", expression.span));
  } else {
    const symbol = lookupSymbol(surfaceName, scope, surface?.span ?? expression.span);
    if (symbol?.valueType !== "surface2d") {
      surfaceDiagnostics.push(error("unsupported-texture", `surf2Dread target '${surfaceName}' is not a surface reference`, surface?.span ?? expression.span));
    }
  }
  const end = returnForm
    ? is1D ? 2 : hasZ ? 4 : 3
    : is1D ? 3 : hasZ ? 5 : 4;
  for (const coord of returnForm ? expression.args.slice(1, end) : expression.args.slice(2, end)) {
    validateScalarOperand(walkExpression(coord, scope), coord.span, diagnostics);
  }
}

function validateSurf2DWrite(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  compatibilityDiagnosticsReachable: boolean,
): void {
  const surfaceDiagnostics = compatibilityDiagnosticsReachable ? diagnostics : [];
  const value = expression.args[0];
  const surface = expression.args[1];
  const xBytes = expression.args[2];
  const y = expression.args[3];
  const layer = callName === "surf2DLayeredwrite" ? expression.args[4] : undefined;
  const z = callName === "surf3Dwrite" ? expression.args[4] : undefined;
  if (value) {
    const info = walkExpression(value, scope);
    if (info.kind !== "vector") validateScalarOperand(info, value.span, diagnostics);
  }
  const surfaceName = surface ? rootIdentifier(surface) : undefined;
  if (!surfaceName) {
    surfaceDiagnostics.push(error("unsupported-surface", "surf2Dwrite second argument must be a surface object", expression.span));
  } else {
    const symbol = lookupSymbol(surfaceName, scope, surface?.span ?? expression.span);
    if (symbol?.valueType !== "surface2d") {
      surfaceDiagnostics.push(error("unsupported-surface", `surf2Dwrite target '${surfaceName}' is not a cudaSurfaceObject_t parameter`, surface?.span ?? expression.span));
    }
  }
  if (xBytes) validateScalarOperand(walkExpression(xBytes, scope), xBytes.span, diagnostics);
  if (y) validateScalarOperand(walkExpression(y, scope), y.span, diagnostics);
  if (layer) validateScalarOperand(walkExpression(layer, scope), layer.span, diagnostics);
  if (z) validateScalarOperand(walkExpression(z, scope), z.span, diagnostics);
}

function validateSizeof(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const target = expression.args[0];
  if (!target) {
    diagnostics.push(error("unsupported-sizeof", "sizeof/alignof expects a CUDA-lite type or modeled value", expression.span));
    return;
  }
  if (target.kind === "identifier" && sizeofCudaType(target.name) !== undefined) return;
  const info = walkExpression(target, scope);
  if (info.valueType === undefined || sizeofCudaType(info.valueType) === undefined) {
    diagnostics.push(error("unsupported-sizeof", "sizeof/alignof only support CUDA-lite scalar types or modeled values", target.span));
  }
}

function validatePoolAllocate(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  atomicParams: Set<string>,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  if (expression.args.length === 4) {
    validateRawPoolAllocate(expression, scope, atomicParams, diagnostics, walkExpression);
    return;
  }
  const pool = expression.args[0];
  if (isExternalPoolAddress(pool)) {
    // External device pool. Runtime input must provide memoryPools[name].
  } else if (pool?.kind !== "identifier") {
    diagnostics.push(error("unsupported-device-pool", "device pool allocation expects DevicePool* as first argument", expression.span));
  } else {
    const symbol = lookupSymbol(pool.name, scope, pool.span);
    if (symbol?.valueType !== "devicepool" || !symbol.pointer) {
      diagnostics.push(error("unsupported-device-pool", `allocation target '${pool.name}' is not a DevicePool* parameter`, pool.span));
    }
  }
  const size = expression.args[1];
  if (size) validateScalarOperand(walkExpression(size, scope), size.span, diagnostics);
}

function isExternalPoolAddress(expression: CudaLiteExpression | undefined): expression is Extract<CudaLiteExpression, { kind: "unary" }> {
  return expression?.kind === "unary" &&
    expression.operator === "&" &&
    expression.argument.kind === "identifier";
}

function validateRawPoolAllocate(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  atomicParams: Set<string>,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const base = expression.args[0];
  const offset = expression.args[1];
  if (base?.kind !== "identifier") {
    diagnostics.push(error("unsupported-device-pool", "raw pool allocation expects pointer base as first argument", expression.span));
  } else {
    const symbol = lookupSymbol(base.name, scope, base.span);
    if (!symbol?.pointer) diagnostics.push(error("unsupported-device-pool", `allocation base '${base.name}' is not a pointer`, base.span));
  }
  if (offset?.kind !== "identifier") {
    diagnostics.push(error("unsupported-device-pool", "raw pool allocation expects size_t* offset as second argument", expression.span));
  } else {
    const symbol = lookupSymbol(offset.name, scope, offset.span);
    if (!symbol?.pointer || (symbol.valueType !== "uint" && symbol.valueType !== "int")) {
      diagnostics.push(error("unsupported-device-pool", `allocation offset '${offset.name}' is not an integer pointer`, offset.span));
    } else {
      atomicParams.add(offset.name);
    }
  }
  for (const arg of expression.args.slice(2)) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
}

function validateCurandInit(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
): void {
  for (const arg of expression.args.slice(0, 3)) {
    validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
  }
  const state = expression.args[3];
  if (!state) return;
  const info = walkExpression(state, scope);
  if (info.kind !== "address") {
    diagnostics.push(error("curand-state-address", "curand_init expects a state address as its fourth argument", state.span));
  }
}

function validateCurandStateAddress(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
): void {
  const state = expression.args[0];
  if (!state) return;
  const info = walkExpression(state, scope);
  if (info.kind !== "address") {
    diagnostics.push(error("curand-state-address", `${callName} expects a state address`, state.span));
  }
}

function validateCuComplexBuiltin(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  options: CudaLiteAnalyzeOptions,
): void {
  if (isCuDoubleComplexBuiltin(callName)) {
    validateF64Type("double", expression.span, diagnostics, options);
  }
  if (callName === "make_cuComplex" || callName === "make_cuFloatComplex" || callName === "make_cuDoubleComplex") {
    for (const arg of expression.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return;
  }
  for (const arg of expression.args) {
    const info = walkExpression(arg, scope);
    if (info.kind === "unknown" || info.kind === "complex" || isFloat2ComplexCompatible("complex64", info)) continue;
    diagnostics.push(error("unsupported-cufft", `${callName} expects cuComplex/cuFloatComplex/cuDoubleComplex or float2 operands`, arg.span));
  }
}

function isCuComplexBuiltin(callName: string): boolean {
  return callName === "make_cuComplex" ||
    callName === "make_cuFloatComplex" ||
    callName === "make_cuDoubleComplex" ||
    callName === "cuCrealf" ||
    callName === "cuCimagf" ||
    callName === "cuCabsf" ||
    callName === "cuConjf" ||
    callName === "cuCaddf" ||
    callName === "cuCsubf" ||
    callName === "cuCmulf" ||
    callName === "cuCdivf" ||
    callName === "cuCfmaf" ||
    callName === "cuCreal" ||
    callName === "cuCimag" ||
    callName === "cuCabs" ||
    callName === "cuConj" ||
    callName === "cuCadd" ||
    callName === "cuCsub" ||
    callName === "cuCmul" ||
    callName === "cuCdiv" ||
    callName === "cuCfma";
}

function isCuComplexScalarBuiltin(callName: string): boolean {
  return callName === "cuCrealf" || callName === "cuCimagf" || callName === "cuCabsf" ||
    callName === "cuCreal" || callName === "cuCimag" || callName === "cuCabs";
}

function isCuDoubleComplexBuiltin(callName: string): boolean {
  return callName === "make_cuDoubleComplex" ||
    callName === "cuCreal" ||
    callName === "cuCimag" ||
    callName === "cuCabs" ||
    callName === "cuConj" ||
    callName === "cuCadd" ||
    callName === "cuCsub" ||
    callName === "cuCmul" ||
    callName === "cuCdiv" ||
    callName === "cuCfma";
}

function validateAtomicBuiltin(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  params: ReadonlyMap<string, CudaLiteParam>,
  atomicParams: Set<string>,
  atomicShared: Set<string>,
  atomicDeviceGlobals: Set<string>,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const target = expression.args[0];
  const callName = expressionName(expression.callee);
  const targetExpression = atomicTargetExpression(target, scope);
  if (!targetExpression) {
    diagnostics.push(error("atomic-address-required", `${callName ?? "atomic operation"} first argument must be a pointer parameter or address like &x[i]`, expression.span));
    if (target) walkExpression(target, scope);
  } else {
    if (isPointerAddressExpression(targetExpression)) {
      validatePointerInitializerExpression(targetExpression, scope, diagnostics, walkExpression);
    } else if (targetExpression.kind !== "identifier") {
      validateLValueExpression(targetExpression, scope, diagnostics, walkExpression);
    }
    const targetName = rootIdentifier(targetExpression);
    const storageRoot = targetName ? atomicStorageRoot(targetName, scope, targetExpression.span) : undefined;
    const param = storageRoot ? params.get(storageRoot) : undefined;
    const symbol = targetName ? lookupSymbol(targetName, scope, targetExpression.span) : undefined;
    const storageSymbol = storageRoot ? lookupSymbol(storageRoot, scope, targetExpression.span) : undefined;
    const targetInfo = target ? validateReadPointerOperand(target, scope, walkExpression) : undefined;
    const targetType = targetInfo?.valueType ?? symbol?.valueType ?? storageSymbol?.valueType;
    if (storageSymbol?.kind === "shared") {
      if (((targetType === "float" || targetType === "double") && isSupportedFloatAtomic(callName)) || isSupportedBfloatAtomic(callName, targetType)) {
        atomicShared.add(storageSymbol.name);
      } else if (targetType === "half" || targetType === "bf16" || targetType === "bool" || targetType === "complex64" || targetType === "float" || targetType === "double") {
        diagnostics.push(error("unsupported-atomic-target", "shared atomics support int/uint targets, CAS-backed float add/sub/min/max/exch/cas, and CAS-backed bf16 add in CUDA-lite", targetExpression.span));
      } else {
        atomicShared.add(storageSymbol.name);
      }
    } else if (storageSymbol?.kind === "device-global") {
      if (((targetType === "float" || targetType === "double") && isSupportedFloatAtomic(callName)) || isSupportedBfloatAtomic(callName, targetType)) {
        atomicDeviceGlobals.add(storageSymbol.name);
      } else if (targetType === "half" || targetType === "bf16" || targetType === "bool" || targetType === "complex64" || targetType === "float" || targetType === "double") {
        diagnostics.push(error("unsupported-atomic-target", "device global atomics support int/uint targets, CAS-backed float add/sub/min/max/exch/cas, and CAS-backed bf16 add in CUDA-lite", targetExpression.span));
      } else {
        atomicDeviceGlobals.add(storageSymbol.name);
      }
    } else if (!param?.pointer && (symbol?.kind === "local" || symbol?.kind === "param") && symbol.pointer) {
      if (symbol.constant) {
        diagnostics.push(error("const-pointer-write", `cannot ${callName ?? "atomic operation"} through const pointer '${symbol.name}'`, expression.span));
      }
      if (targetType && isSupportedDevicePointerAtomic(callName, targetType)) {
        // Exact storage roots for helper pointer atomics are marked after validation.
      } else {
        diagnostics.push(error("unsupported-atomic-target", `${callName ?? "atomic operation"} through device pointer supports int/uint read-modify-write atomics including inc/dec, CAS-backed float add/sub/min/max/exch/cas, and CAS-backed bf16 add in CUDA-lite`, targetExpression.span));
      }
    } else if (!param?.pointer) {
      diagnostics.push(error("unsupported-atomic-target", `${callName ?? "atomic operation"} target must resolve to storage or shared memory`, targetExpression.span));
    } else if (isSupportedBfloatAtomic(callName, targetType)) {
      atomicParams.add(param.name);
      if (param.constant) {
        diagnostics.push(error("const-pointer-write", `cannot ${callName} through const pointer '${param.name}'`, expression.span));
      }
    } else if ((targetType === "float" || targetType === "double") && (
      callName === "atomicAdd" ||
      callName === "atomicAdd_system" ||
      callName === "atomicSub" ||
      callName === "atomicSub_system" ||
      callName === "atomicMin" ||
      callName === "atomicMin_system" ||
      callName === "atomicMax" ||
      callName === "atomicMax_system" ||
      callName === "atomicMaxFloat" ||
      callName === "atomicExch" ||
      callName === "atomicExch_system" ||
      callName === "atomicCAS" ||
      callName === "atomicCAS_system"
    )) {
      atomicParams.add(param.name);
      if (param.constant) {
        diagnostics.push(error("const-pointer-write", `cannot ${callName} through const pointer '${param.name}'`, expression.span));
      }
    } else if (targetType === "float" || targetType === "double" || targetType === "half" || targetType === "bf16" || targetType === "complex64") {
      diagnostics.push(error("unsupported-atomic-f32", "unsupported float atomic operation in CUDA-lite v0", expression.span));
    } else {
      atomicParams.add(param.name);
      if (param.constant) {
        diagnostics.push(error("const-pointer-write", `cannot ${callName ?? "atomic operation"} through const pointer '${param.name}'`, expression.span));
      }
    }
  }
  for (const arg of expression.args.slice(1)) {
    validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
  }
}

function atomicStorageRoot(name: string, scope: Scope, span: SourceSpan): string {
  const symbol = lookupSymbol(name, scope, span);
  return symbol?.pointerRoot ?? name;
}

function markExactAtomicPointerUsage(
  ast: CudaLiteModule,
  kernel: CudaLiteKernel,
  options: CudaLiteAnalyzeOptions,
  atomicParams: Set<string>,
  atomicShared: Set<string>,
  atomicDeviceGlobals: Set<string>,
): void {
  const reachableFunctionSpans = reachableDeviceFunctionSpans(ast.functions, kernel.body);
  const reachableFunctions = ast.functions.filter((fn) => reachableFunctionSpans.has(fn.span.start));
  const functionsByName = new Map<string, CudaLiteDeviceFunction[]>();
  for (const fn of reachableFunctions) {
    const overloads = functionsByName.get(fn.name) ?? [];
    overloads.push(fn);
    functionsByName.set(fn.name, overloads);
  }

  const sharedNames = new Set(collectSharedDeclarationsFromBodies([kernel.body, ...reachableFunctions.map((fn) => fn.body)], options).map((shared) => shared.name));
  const deviceGlobalNames = new Set(ast.deviceGlobals.map((global) => global.name));
  const kernelPointerParams = new Set(kernel.params.filter((param) => param.pointer && !param.constant).map((param) => param.name));
  const functionAtomicParams = new Map<string, Set<string>>();
  for (const fn of reachableFunctions) functionAtomicParams.set(fn.name, new Set());

  const markConcreteRoot = (root: string): boolean => {
    if (kernelPointerParams.has(root)) {
      const size = atomicParams.size;
      atomicParams.add(root);
      return atomicParams.size !== size;
    }
    if (sharedNames.has(root)) {
      const size = atomicShared.size;
      atomicShared.add(root);
      return atomicShared.size !== size;
    }
    if (deviceGlobalNames.has(root)) {
      const size = atomicDeviceGlobals.size;
      atomicDeviceGlobals.add(root);
      return atomicDeviceGlobals.size !== size;
    }
    return false;
  };

  let changed = true;
  while (changed) {
    changed = false;
    for (const fn of reachableFunctions) {
      const fnPointerParams = new Set(fn.params.filter((param) => param.pointer).map((param) => param.name));
      const fnAtomicParams = functionAtomicParams.get(fn.name)!;
      const markFunctionRoot = (root: string): void => {
        if (fnPointerParams.has(root)) {
          const size = fnAtomicParams.size;
          fnAtomicParams.add(root);
          if (fnAtomicParams.size !== size) changed = true;
          return;
        }
        if (markConcreteRoot(root)) changed = true;
      };
      const aliases = new Map<string, Set<string>>([...fnPointerParams].map((name) => [name, new Set([name])]));
      scanAtomicPointerStatements(fn.body, aliases, markFunctionRoot, functionAtomicParams, functionsByName);
    }
  }

  const kernelAliases = new Map<string, Set<string>>(kernel.params.filter((param) => param.pointer).map((param) => [param.name, new Set([param.name])]));
  scanAtomicPointerStatements(kernel.body, kernelAliases, markConcreteRoot, functionAtomicParams, functionsByName);
}

function scanAtomicPointerStatements(
  statements: readonly CudaLiteStatement[],
  aliases: Map<string, Set<string>>,
  markRoot: (root: string) => void,
  functionAtomicParams: ReadonlyMap<string, ReadonlySet<string>>,
  functionsByName: ReadonlyMap<string, readonly CudaLiteDeviceFunction[]>,
): void {
  const visitExpression = (
    expression: CudaLiteExpression,
    expressionAliases: Map<string, Set<string>> = aliases,
  ): void => {
    if (
      expression.kind === "assignment" &&
      expression.operator === "=" &&
      expression.left.kind === "identifier" &&
      expressionAliases.has(expression.left.name)
    ) {
      expressionAliases.set(expression.left.name, pointerRootsFromExpression(expression.right, expressionAliases));
    }
    if (
      expression.kind === "assignment" &&
      expression.operator === "=" &&
      expression.left.kind === "index" &&
      expression.left.target.kind === "identifier" &&
      expressionAliases.has(expression.left.target.name)
    ) {
      const roots = pointerRootsFromExpression(expression.right, expressionAliases);
      const elementKey = pointerArrayElementAliasKey(expression.left);
      if (elementKey) expressionAliases.set(elementKey, new Set(roots));
      const arrayRoots = expressionAliases.get(expression.left.target.name) ?? new Set<string>();
      for (const root of roots) arrayRoots.add(root);
      expressionAliases.set(expression.left.target.name, arrayRoots);
    }
    if (expression.kind === "call") {
      const callName = expressionName(expression.callee);
      if (callName && isAtomicBuiltin(callName)) {
        const target = atomicTargetExpression(expression.args[0]);
        const roots = target ? pointerRootsFromExpression(target, expressionAliases) : new Set<string>();
        for (const root of roots) markRoot(root);
      }
      const overloads = callName ? functionsByName.get(callName) : undefined;
      const callee = overloads?.find((candidate) => candidate.params.length === expression.args.length) ?? overloads?.[0];
      const atomicParams = callName ? functionAtomicParams.get(callName) : undefined;
      if (callee && atomicParams) {
        for (const [index, param] of callee.params.entries()) {
          if (!param.pointer || !atomicParams.has(param.name)) continue;
          const roots = pointerRootsFromExpression(expression.args[index], expressionAliases);
          for (const root of roots) markRoot(root);
        }
      }
    }
    forEachExpressionChild(expression, (child) => visitExpression(child, expressionAliases));
  };

  for (const statement of statements) {
    switch (statement.kind) {
      case "block":
        scanChildAtomicPointerStatements(statement.body, aliases, markRoot, functionAtomicParams, functionsByName);
        break;
      case "var":
        if (statement.init) visitExpression(statement.init);
        if (statement.pointer) {
          aliases.set(statement.name, pointerRootsFromExpression(statement.init, aliases));
        }
        break;
      case "dim3":
        for (const arg of statement.args) visitExpression(arg);
        break;
      case "cooperative-group":
        if (statement.partitionPredicate) visitExpression(statement.partitionPredicate);
        break;
      case "kernel-launch":
        for (const arg of [...statement.grid, ...statement.block, ...statement.args]) visitExpression(arg);
        break;
      case "asm":
        for (const arg of [...(statement.outputs ?? []), ...statement.inputs]) visitExpression(arg);
        break;
      case "expr":
        visitExpression(statement.expression);
        break;
      case "if":
        visitExpression(statement.condition);
        mergeBranchAtomicPointerAliases(aliases, statement.consequent, statement.alternate, markRoot, functionAtomicParams, functionsByName);
        break;
      case "for": {
        const loopAliases = cloneAtomicPointerAliases(aliases);
        if (statement.init?.kind === "var") {
          if (statement.init.pointer) {
            loopAliases.set(statement.init.name, pointerRootsFromExpression(statement.init.init, loopAliases));
          }
          if (statement.init.init) visitExpression(statement.init.init, loopAliases);
        } else if (statement.init) {
          visitExpression(statement.init, loopAliases);
        }
        if (statement.condition) visitExpression(statement.condition, loopAliases);
        if (statement.update) visitExpression(statement.update, loopAliases);
        scanAtomicPointerStatements(statement.body, loopAliases, markRoot, functionAtomicParams, functionsByName);
        mergeExistingAtomicPointerAliases(aliases, loopAliases);
        break;
      }
      case "while":
        visitExpression(statement.condition);
        scanChildAtomicPointerStatements(statement.body, aliases, markRoot, functionAtomicParams, functionsByName);
        break;
      case "do-while":
        scanChildAtomicPointerStatements(statement.body, aliases, markRoot, functionAtomicParams, functionsByName);
        visitExpression(statement.condition);
        break;
      case "return":
        if (statement.value) visitExpression(statement.value);
        break;
      case "continue":
      case "break":
        break;
    }
  }
}

function scanChildAtomicPointerStatements(
  statements: readonly CudaLiteStatement[],
  aliases: Map<string, Set<string>>,
  markRoot: (root: string) => void,
  functionAtomicParams: ReadonlyMap<string, ReadonlySet<string>>,
  functionsByName: ReadonlyMap<string, readonly CudaLiteDeviceFunction[]>,
): void {
  const childAliases = cloneAtomicPointerAliases(aliases);
  scanAtomicPointerStatements(statements, childAliases, markRoot, functionAtomicParams, functionsByName);
  mergeExistingAtomicPointerAliases(aliases, childAliases);
}

function mergeBranchAtomicPointerAliases(
  aliases: Map<string, Set<string>>,
  consequent: readonly CudaLiteStatement[],
  alternate: readonly CudaLiteStatement[] | undefined,
  markRoot: (root: string) => void,
  functionAtomicParams: ReadonlyMap<string, ReadonlySet<string>>,
  functionsByName: ReadonlyMap<string, readonly CudaLiteDeviceFunction[]>,
): void {
  const consequentAliases = cloneAtomicPointerAliases(aliases);
  scanAtomicPointerStatements(consequent, consequentAliases, markRoot, functionAtomicParams, functionsByName);
  const alternateAliases = cloneAtomicPointerAliases(aliases);
  if (alternate) scanAtomicPointerStatements(alternate, alternateAliases, markRoot, functionAtomicParams, functionsByName);
  mergeExistingAtomicPointerAliases(aliases, consequentAliases);
  mergeExistingAtomicPointerAliases(aliases, alternateAliases);
}

function cloneAtomicPointerAliases(aliases: ReadonlyMap<string, ReadonlySet<string>>): Map<string, Set<string>> {
  return new Map([...aliases].map(([name, roots]) => [name, new Set(roots)]));
}

function mergeExistingAtomicPointerAliases(
  target: Map<string, Set<string>>,
  source: ReadonlyMap<string, ReadonlySet<string>>,
): void {
  for (const [name, roots] of source) {
    const targetRoots = target.get(name);
    if (!targetRoots) continue;
    for (const root of roots) targetRoots.add(root);
  }
}

function pointerRootsFromExpression(
  expression: CudaLiteExpression | undefined,
  aliases: ReadonlyMap<string, ReadonlySet<string>>,
): Set<string> {
  if (!expression) return new Set();
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    return pointerRootsFromExpression(expression.args[0], aliases);
  }
  if (expression.kind === "cast" && expression.pointer) {
    return pointerRootsFromExpression(expression.expression, aliases);
  }
  if (expression.kind === "conditional") {
    return new Set([
      ...pointerRootsFromExpression(expression.consequent, aliases),
      ...pointerRootsFromExpression(expression.alternate, aliases),
    ]);
  }
  if (expression.kind === "assignment" && expression.operator === "=") {
    return pointerRootsFromExpression(expression.right, aliases);
  }
  if (expression.kind === "sequence") {
    return pointerRootsFromExpression(expression.expressions.at(-1), aliases);
  }
  if (expression.kind === "index" && expression.target.kind === "identifier") {
    const elementKey = pointerArrayElementAliasKey(expression);
    const elementAlias = elementKey ? aliases.get(elementKey) : undefined;
    if (elementAlias) return new Set(elementAlias);
  }
  const root = rootIdentifier(expression);
  if (!root) return new Set();
  const aliased = aliases.get(root);
  return aliased ? new Set(aliased) : new Set([root]);
}

function pointerArrayElementAliasKey(expression: Extract<CudaLiteExpression, { kind: "index" }>): string | undefined {
  if (expression.target.kind !== "identifier") return undefined;
  if (expression.index.kind !== "number" || !Number.isInteger(expression.index.value)) return undefined;
  return `${expression.target.name}[${expression.index.value}]`;
}

function isAtomicBuiltin(callName: string): boolean {
  return callName === "atomicAdd" ||
    callName === "atomicAdd_system" ||
    callName === "atomicSub" ||
    callName === "atomicSub_system" ||
    callName === "atomicMin" ||
    callName === "atomicMin_system" ||
    callName === "atomicMax" ||
    callName === "atomicMax_system" ||
    callName === "atomicMaxFloat" ||
    callName === "atomicAnd" ||
    callName === "atomicAnd_system" ||
    callName === "atomicOr" ||
    callName === "atomicOr_system" ||
    callName === "atomicXor" ||
    callName === "atomicXor_system" ||
    callName === "atomicInc" ||
    callName === "atomicInc_system" ||
    callName === "atomicDec" ||
    callName === "atomicDec_system" ||
    callName === "atomicExch" ||
    callName === "atomicExch_system" ||
    callName === "atomicCAS" ||
    callName === "atomicCAS_system";
}

function isSupportedFloatAtomic(callName: string | undefined): boolean {
  return callName === "atomicAdd" ||
    callName === "atomicAdd_system" ||
    callName === "atomicSub" ||
    callName === "atomicMin" ||
    callName === "atomicMin_system" ||
    callName === "atomicMax" ||
    callName === "atomicMax_system" ||
    callName === "atomicMaxFloat" ||
    callName === "atomicExch" ||
    callName === "atomicExch_system" ||
    callName === "atomicCAS" ||
    callName === "atomicCAS_system";
}

function isSupportedBfloatAtomic(callName: string | undefined, targetType: CudaLiteScalarType | undefined): boolean {
  return targetType === "bf16" && (callName === "atomicAdd" || callName === "atomicAdd_system");
}

function isSupportedDevicePointerAtomic(
  callName: string | undefined,
  targetType: CudaLiteScalarType,
): boolean {
  if (isSupportedBfloatAtomic(callName, targetType)) return true;
  if (targetType !== "float" && targetType !== "double" && targetType !== "int" && targetType !== "uint") return false;
  if (callName === "atomicAdd" || callName === "atomicAdd_system") return true;
  if (callName === "atomicSub" || callName === "atomicSub_system") return true;
  if (callName === "atomicMin" || callName === "atomicMin_system") return true;
  if (callName === "atomicMax" || callName === "atomicMax_system" || callName === "atomicMaxFloat") return true;
  if (callName === "atomicExch" || callName === "atomicExch_system") return true;
  if (callName === "atomicCAS" || callName === "atomicCAS_system") return true;
  if (targetType === "int" || targetType === "uint") {
    return callName === "atomicAnd" ||
      callName === "atomicAnd_system" ||
      callName === "atomicOr" ||
      callName === "atomicOr_system" ||
      callName === "atomicXor" ||
      callName === "atomicXor_system" ||
      callName === "atomicInc" ||
      callName === "atomicInc_system" ||
      callName === "atomicDec" ||
      callName === "atomicDec_system" ||
      callName === "atomicCAS" ||
      callName === "atomicCAS_system";
  }
  return false;
}

function isShuffleBuiltin(callName: string): boolean {
  return callName === "__shfl" ||
    callName === "__shfl_down" ||
    callName === "__shfl_up" ||
    callName === "__shfl_xor" ||
    callName === "__shfl_sync" ||
    callName === "__shfl_down_sync" ||
    callName === "__shfl_up_sync" ||
    callName === "__shfl_xor_sync";
}

function isVoteBuiltin(callName: string): boolean {
  return callName === "__any" ||
    callName === "__all" ||
    callName === "__ballot" ||
    callName === "__any_sync" ||
    callName === "__all_sync" ||
    callName === "__ballot_sync" ||
    callName === "__match_any_sync";
}

function isSyncthreadsPredicateBuiltin(callName: string): boolean {
  return callName === "__syncthreads_count" || callName === "__syncthreads_and" || callName === "__syncthreads_or";
}

function isMaskedWarpReductionBuiltin(callName: string): boolean {
  return callName === "__reduce_add_sync" || callName === "__reduce_min_sync" || callName === "__reduce_max_sync";
}

function isMaskedWarpBitwiseReductionBuiltin(callName: string): boolean {
  return callName === "__reduce_and_sync" || callName === "__reduce_or_sync" || callName === "__reduce_xor_sync";
}

function isWarpReductionBuiltin(callName: string): boolean {
  return callName === "warpReduceSum" ||
    callName === "warpReduceMax" ||
    callName === "warpReduceMin" ||
    callName === "warp_reduce_sum" ||
    callName === "warp_reduce_max" ||
    callName === "warp_reduce_min" ||
    callName === "warp_reduce_sum_f32" ||
    callName === "warp_reduce_max_f32" ||
    callName === "warp_reduce_sum_f16" ||
    callName === "warp_reduce_sum_f16_f16" ||
    callName === "warp_reduce_sum_f16_f32" ||
    callName === "warp_reduce_sum_i8_i32" ||
    callName === "warp_reduce_sum_i32_i32" ||
    callName === "blockReduce";
}

function warpReductionReturnType(callName: string, valueType: ValueType | undefined): ValueType | undefined {
  if (callName.endsWith("_i32")) return "int";
  if (callName.endsWith("_f32")) return "float";
  if (callName.endsWith("_f16")) return "half";
  if (callName.endsWith("_f16_f16")) return "half";
  if (callName.endsWith("_f16_f32")) return "float";
  return valueType;
}

function atomicTargetExpression(
  target: CudaLiteExpression | undefined,
  scope?: Scope,
): CudaLiteExpression | undefined {
  if (!target) return undefined;
  if (target.kind === "cast" && target.pointer) return atomicTargetExpression(target.expression, scope);
  if (target.kind === "unary" && target.operator === "&") return target.argument;
  if (target.kind === "identifier") return target;
  if (target.kind === "index" && rootIdentifier(target.target)) {
    if (!scope) return target;
    const root = rootIdentifier(target.target);
    const symbol = root ? lookupSymbol(root, scope, target.span) : undefined;
    if (symbol?.kind === "local" && symbol.pointer && (symbol.dimensions?.length ?? 0) > 0) return target;
  }
  if (target.kind === "binary" && (target.operator === "+" || target.operator === "-")) return target;
  return undefined;
}

function isPointerAddressExpression(expression: CudaLiteExpression): boolean {
  if (expression.kind === "cast" && expression.pointer) return true;
  return expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-");
}

function validateNonCallExpression(
  expression: CudaLiteExpression,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  requiredFeatures: Set<string>,
): ExpressionInfo {
  switch (expression.kind) {
    case "number":
      return { kind: "scalar" };
    case "string":
      return { kind: "string" };
    case "initializer":
      for (const element of flattenInitializerExpressions(expression)) {
        validateScalarOperand(walkExpression(element, scope), element.span, diagnostics);
      }
      return { kind: "unknown" };
    case "identifier":
      return expressionInfoForIdentifier(expression.name, expression.span, scope, diagnostics);
    case "cast": {
      const info = walkExpression(expression.expression, scope);
      if (expression.pointer) {
        if (info.kind !== "scalar" && info.kind !== "pointer" && info.kind !== "pool-pointer" && info.kind !== "address" && info.kind !== "array" && info.kind !== "unknown") {
          diagnostics.push(error("unsupported-pointer-cast", "pointer cast expects scalar or pointer expression", expression.expression.span));
        }
        return info.kind === "pool-pointer"
          ? { kind: "pool-pointer", valueType: expression.valueType }
          : { kind: "pointer", valueType: expression.valueType, symbol: info.symbol };
      }
      validateScalarOperand(info, expression.expression.span, diagnostics);
      return { kind: "scalar", valueType: expression.valueType };
    }
    case "member": {
      const matrixMember = validateMatrixTileMemberExpression(expression, scope, diagnostics, walkExpression);
      if (matrixMember) return matrixMember;
      const object = walkExpression(expression.object, scope);
      if (object.kind === "unknown") return { kind: "unknown" };
      if (object.kind === "complex") {
        if (expression.property !== "x" && expression.property !== "y") {
          diagnostics.push(error("unsupported-member-target", `unsupported complex member '${expression.property}'`, expression.span));
        }
        return { kind: "scalar", valueType: "float" };
      }
      if (isCudaVectorType(object.valueType)) {
        if (expression.property === "size") return { kind: "scalar", valueType: "int" };
        const swizzleType = cudaVectorSwizzleType(object.valueType, expression.property);
        if (swizzleType === undefined) {
          diagnostics.push(error("unsupported-vector-member", `unsupported ${object.valueType} member '${expression.property}'`, expression.span));
        }
        const scalarType = swizzleType === undefined || isCudaVectorType(swizzleType)
          ? cudaVectorScalarType(object.valueType)
          : swizzleType as ValueType;
        return isCudaVectorType(swizzleType)
          ? { kind: "vector", valueType: swizzleType }
          : { kind: "scalar", valueType: scalarType };
      }
      if (object.kind !== "vector") {
        diagnostics.push(error("unsupported-member-target", "member access is only supported on CUDA-lite builtin vectors", expression.span));
      }
      if (expression.property !== "x" && expression.property !== "y" && expression.property !== "z") {
        diagnostics.push(error("unsupported-member-target", `unsupported vector member '${expression.property}'`, expression.span));
      }
      return { kind: "scalar", valueType: "int" };
    }
    case "index": {
      const target = walkExpression(expression.target, scope);
      validateScalarOperand(walkExpression(expression.index, scope), expression.index.span, diagnostics);
      if (target.kind === "vector") {
        const scalar = isCudaVectorType(target.valueType) ? cudaVectorScalarType(target.valueType) : "int";
        return { kind: "scalar", valueType: scalar };
      }
      if (target.kind === "pointer") {
        if (isCudaVectorType(target.valueType)) {
          return { kind: "vector", valueType: target.valueType, symbol: target.symbol };
        }
        return target.valueType === "complex64"
          ? { kind: "complex", valueType: target.valueType, symbol: target.symbol }
          : { kind: "scalar", valueType: target.valueType, symbol: target.symbol };
      }
      if (target.kind === "pool-pointer") {
        return { kind: "scalar", valueType: target.valueType };
      }
      if (target.kind === "array") {
        const dimensions = target.dimensions ?? [];
        if (dimensions.length > 1) {
          return {
            kind: "array",
            valueType: target.valueType,
            dimensions: dimensions.slice(1),
            symbol: target.symbol,
          };
        }
        if (target.symbol?.pointer) {
          return { kind: "pointer", valueType: target.valueType, symbol: target.symbol };
        }
        if (isCudaVectorType(target.valueType)) {
          return { kind: "vector", valueType: target.valueType, symbol: target.symbol };
        }
        return target.valueType === "complex64"
          ? { kind: "complex", valueType: target.valueType, symbol: target.symbol }
          : { kind: "scalar", valueType: target.valueType, symbol: target.symbol };
      }
      diagnostics.push(error("unsupported-index-target", "only pointer parameters, local arrays, fixed __shared__ arrays, constants, and device globals can be indexed", expression.span));
      return { kind: "unknown" };
    }
    case "unary": {
      if (expression.operator === "&") {
        return validateAddressOfExpression(expression.argument, scope, diagnostics, walkExpression);
      }
      if (expression.operator === "*") {
        const info = walkExpression(expression.argument, scope);
        if (info.kind !== "pointer" && info.kind !== "pool-pointer") {
          diagnostics.push(error("unsupported-deref-target", "unary * expects a pointer expression", expression.argument.span));
          return { kind: "unknown" };
        }
        if (isCudaVectorType(info.valueType)) return { kind: "vector", valueType: info.valueType, symbol: info.symbol };
        return info.valueType === "complex64"
          ? { kind: "complex", valueType: info.valueType, symbol: info.symbol }
          : { kind: "scalar", valueType: info.valueType, symbol: info.symbol };
      }
      const info = walkExpression(expression.argument, scope);
      validateScalarOperand(info, expression.argument.span, diagnostics);
      return { kind: "scalar", valueType: info.valueType };
    }
    case "binary": {
      const left = walkExpression(expression.left, scope);
      const right = walkExpression(expression.right, scope);
      if (expression.operator === "-" && isPointerLikeInfo(left) && isPointerLikeInfo(right)) {
        if (
          left.valueType !== undefined &&
          right.valueType !== undefined &&
          left.valueType !== "voidptr" &&
          right.valueType !== "voidptr" &&
          left.valueType !== right.valueType
        ) {
          diagnostics.push(error(
            "unsupported-pointer-difference",
            "pointer difference expects matching pointee types",
            expression.span,
          ));
        }
        return { kind: "scalar", valueType: "int" };
      }
      if ((expression.operator === "+" || expression.operator === "-") && isPointerLikeInfo(left)) {
        validateScalarOperand(right, expression.right.span, diagnostics);
        return left;
      }
      if (expression.operator === "+" && isPointerLikeInfo(right)) {
        validateScalarOperand(left, expression.left.span, diagnostics);
        return right;
      }
      if ((expression.operator === "+" || expression.operator === "-") && left.kind === "array") {
        validateScalarOperand(right, expression.right.span, diagnostics);
        return { kind: "pointer", valueType: left.valueType, symbol: left.symbol };
      }
      if (expression.operator === "+" && right.kind === "array") {
        validateScalarOperand(left, expression.left.span, diagnostics);
        return { kind: "pointer", valueType: right.valueType, symbol: right.symbol };
      }
      if (isVectorArithmeticOperator(expression.operator) && left.kind === "vector" && right.kind === "vector") {
        if (!left.valueType || !right.valueType || left.valueType !== right.valueType) {
          diagnostics.push(error("unsupported-vector-argument", "vector arithmetic expects matching CUDA vector types", expression.span));
          return { kind: "unknown" };
        }
        if (cudaVectorScalarType(left.valueType) === "half") requiredFeatures.add("shader-f16");
        return { kind: "vector", valueType: left.valueType };
      }
      const vectorArithmetic = vectorArithmeticInfo(
        expression.operator,
        left,
        right,
        expression.left,
        expression.right,
        diagnostics,
        requiredFeatures,
      );
      if (vectorArithmetic) return vectorArithmetic;
      if ((expression.operator === "==" || expression.operator === "!=") && pointerComparable(left, right, expression.left, expression.right)) {
        return { kind: "scalar", valueType: "bool" };
      }
      validateScalarOperand(left, expression.left.span, diagnostics);
      validateScalarOperand(right, expression.right.span, diagnostics);
      return { kind: "scalar" };
    }
    case "conditional": {
      const condition = walkExpression(expression.condition, scope);
      if (!isPointerLikeInfo(condition)) validateScalarOperand(condition, expression.condition.span, diagnostics);
      const consequent = walkExpression(expression.consequent, scope);
      const alternate = walkExpression(expression.alternate, scope);
      const pointer = conditionalPointerInfo(expression, consequent, alternate, diagnostics);
      if (pointer) return pointer;
      if (consequent.kind === "vector" || alternate.kind === "vector") {
        if (consequent.kind !== "vector" || alternate.kind !== "vector" || consequent.valueType !== alternate.valueType) {
          diagnostics.push(error("unsupported-vector-argument", "conditional CUDA vector expressions require matching vector branches", expression.span));
        }
        return { kind: "vector", valueType: consequent.valueType ?? alternate.valueType };
      }
      validateScalarOperand(consequent, expression.consequent.span, diagnostics);
      validateScalarOperand(alternate, expression.alternate.span, diagnostics);
      return { kind: "scalar" };
    }
    case "sequence": {
      let info: ExpressionInfo = { kind: "scalar" };
      for (const item of expression.expressions) info = walkExpression(item, scope);
      return info;
    }
    case "assignment": {
      validateLValueExpression(expression.left, scope, diagnostics, walkExpression, expression.operator);
      const left = walkExpression(expression.left, scope);
      const right = walkExpression(expression.right, scope);
      if (left.kind === "complex") {
        if (right.kind !== "complex" && right.kind !== "unknown" && !isFloat2ComplexCompatible("float2", right)) {
          diagnostics.push(error("unsupported-scalar-expression", "complex assignment expects a complex value", expression.right.span));
        }
      } else if (left.kind === "vector") {
        if (expression.operator === "=" && right.kind !== "vector" && right.kind !== "unknown" && !isFloat2ComplexCompatible(left.valueType, right)) {
          diagnostics.push(error("unsupported-vector-assignment", "CUDA vector assignment expects a CUDA vector value", expression.right.span));
        } else if (expression.operator !== "=") {
          const op = assignmentArithmeticOperator(expression.operator);
          if (!op || !vectorArithmeticInfo(op, left, right, expression.left, expression.right, diagnostics, requiredFeatures)) {
            diagnostics.push(error("unsupported-vector-assignment", "CUDA vector compound assignment expects a scalar or matching CUDA vector value", expression.right.span));
          }
        }
      } else if (left.kind === "pointer") {
        if (expression.operator === "=") {
          const localArrayDecay = right.kind === "array" &&
            right.symbol?.kind === "local" &&
            (right.symbol.dimensions?.length ?? 0) > 0 &&
            pointerTypesCompatible(left.valueType ?? "float", right.valueType ?? "float", hasExplicitPointerCast(expression.right));
          if (right.kind !== "pointer" && right.kind !== "pool-pointer" && right.kind !== "address" && right.kind !== "unknown" && !localArrayDecay) {
            diagnostics.push(error("unsupported-pointer-assignment", "CUDA pointer assignment expects a pointer value", expression.right.span));
          }
        } else if (isPointerRebaseOperator(expression.operator)) {
          validateScalarOperand(right, expression.right.span, diagnostics);
        } else {
          diagnostics.push(error("unsupported-pointer-assignment", "CUDA pointer compound assignment supports =, +=, and -=", expression.right.span));
        }
      } else {
        validateScalarOperand(right, expression.right.span, diagnostics);
      }
      if (left.kind === "vector" || left.kind === "complex" || left.kind === "pointer" || left.kind === "pool-pointer") return left;
      return { kind: "scalar", valueType: left.valueType };
    }
    case "update": {
      validateLValueExpression(expression.argument, scope, diagnostics, walkExpression, expression.operator);
      return { kind: "scalar" };
    }
    case "call":
      return { kind: "unknown" };
  }
}

function validateMatrixTileMemberExpression(
  expression: Extract<CudaLiteExpression, { kind: "member" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): ExpressionInfo | undefined {
  const ref = matrixTileReference(expression.object);
  if (!ref) return undefined;
  const symbol = lookupSymbol(ref.root, scope, expression.span);
  if (!symbol?.matrixTile) return undefined;
  for (const index of ref.indices) validateScalarOperand(walkExpression(index, scope), index.span, diagnostics);
  const dimensions = symbol.dimensions ?? [];
  if (ref.indices.length !== dimensions.length) {
    diagnostics.push(error("invalid-wmma-fragment-index", `WMMA fragment '${ref.root}' expects ${dimensions.length} leading index${dimensions.length === 1 ? "" : "es"} before member access`, expression.span));
  }
  const spec = resolveMatrixTileSpec(symbol.matrixTile);
  if (!spec) return { kind: "unknown" };
  if (expression.property === "num_elements") return { kind: "scalar", valueType: "int" };
  if (expression.property === "x") {
    return {
      kind: "array",
      valueType: spec.valueType,
      dimensions: [matrixTileElementCount(spec)],
      symbol,
    };
  }
  diagnostics.push(error("unsupported-wmma-fragment-member", `unsupported WMMA fragment member '${expression.property}'`, expression.span));
  return { kind: "unknown" };
}

function validateLValueExpression(
  expression: CudaLiteExpression,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  operator?: CudaLiteAssignmentExpression["operator"] | CudaLiteUpdateExpression["operator"],
): void {
  if (expression.kind === "identifier") {
    const symbol = lookupSymbol(expression.name, scope, expression.span);
    if (!symbol) {
      diagnostics.push(unknownSymbolDiagnostic(expression.name, scope, expression.span));
      return;
    }
    if (symbol.matrixTile) {
      diagnostics.push(error("unsupported-wmma-fragment-use", "WMMA fragments must be used through supported wmma::* operations", expression.span));
      return;
    }
    if (symbol.kind === "local" || symbol.kind === "shared" || symbol.kind === "device-global") return;
    if (symbol.kind === "param" && !symbol.pointer) return;
    if (symbol.kind === "param" && symbol.pointer && (operator === "=" || isPointerRebaseOperator(operator))) return;
    diagnostics.push(error("invalid-assignment-target", "assignment target must be a local variable, pointer element, shared element, or device global", expression.span));
    return;
  }
  if (expression.kind === "index") {
    const info = walkExpression(expression, scope);
    const root = rootIdentifier(expression);
    const symbol = root ? lookupSymbol(root, scope, expression.span) : undefined;
    const rootTarget = symbol?.pointerRoot ? lookupSymbol(symbol.pointerRoot, scope, expression.span) : symbol;
    if (rootTarget?.kind === "constant") {
      diagnostics.push(error("const-pointer-write", `cannot write to constant memory '${symbol?.pointerRoot ?? root}'`, expression.span));
      return;
    }
    if (symbol?.pointer && symbol.constant) {
      diagnostics.push(error("const-pointer-write", `cannot write through const pointer '${root}'`, expression.span));
      return;
    }
    if (info.kind !== "scalar" && info.kind !== "complex" && info.kind !== "vector" && info.kind !== "pointer") {
      diagnostics.push(error("invalid-assignment-target", "assignment target must resolve to a scalar or complex element", expression.span));
    }
    return;
  }
  if (expression.kind === "member") {
    const info = walkExpression(expression.object, scope);
    if (info.kind === "complex") {
      if (expression.property !== "x" && expression.property !== "y") {
        diagnostics.push(error("invalid-assignment-target", "complex assignment target must be .x or .y", expression.span));
      }
      return;
    }
    if (isCudaVectorType(info.valueType)) {
      const fields = cudaVectorSwizzleIndices(info.valueType, expression.property);
      if (fields === undefined) {
        diagnostics.push(error("invalid-assignment-target", `vector assignment target must be one of .${CUDA_VECTOR_TYPES.get(info.valueType)!.fields.join("/.")}`, expression.span));
      } else if (new Set(fields).size !== fields.length) {
        diagnostics.push(error("invalid-assignment-target", "vector swizzle assignment target cannot repeat lanes", expression.span));
      }
      return;
    }
  }
  if (expression.kind === "unary" && expression.operator === "*") {
    const info = walkExpression(expression.argument, scope);
    if (info.kind !== "pointer" && info.kind !== "pool-pointer" && info.kind !== "address" && info.kind !== "unknown") {
      diagnostics.push(error("unsupported-deref-target", "unary * expects a pointer expression", expression.argument.span));
      return;
    }
    const root = rootIdentifier(expression);
    const symbol = root ? lookupSymbol(root, scope, expression.span) : undefined;
    const rootTarget = symbol?.pointerRoot ? lookupSymbol(symbol.pointerRoot, scope, expression.span) : symbol;
    if (rootTarget?.kind === "constant") {
      diagnostics.push(error("const-pointer-write", `cannot write to constant memory '${symbol?.pointerRoot ?? root}'`, expression.span));
      return;
    }
    if (symbol?.pointer && symbol.constant) {
      diagnostics.push(error("const-pointer-write", `cannot write through const pointer '${root}'`, expression.span));
    }
    return;
  }
  diagnostics.push(error("invalid-assignment-target", "assignment target must be a local variable, pointer element, shared element, or device global", expression.span));
}

function isPointerRebaseOperator(
  operator: CudaLiteAssignmentExpression["operator"] | CudaLiteUpdateExpression["operator"] | undefined,
): boolean {
  return operator === "+=" || operator === "-=" || operator === "++" || operator === "--";
}

function expressionInfoForIdentifier(
  name: string,
  span: SourceSpan,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
): ExpressionInfo {
  const symbol = lookupSymbol(name, scope, span);
  if (!symbol && name === "nullptr") return { kind: "scalar", valueType: "voidptr" };
  const namedConstant = !symbol ? CUDA_NAMED_CONSTANTS.get(name) : undefined;
  if (namedConstant) return { kind: "scalar", valueType: namedConstant.valueType };
  if (!symbol) {
    diagnostics.push(unknownSymbolDiagnostic(name, scope, span));
    return { kind: "unknown" };
  }
  if (symbol.kind === "builtin-vector") return { kind: "vector", symbol };
  if (symbol.kind === "builtin-call") return { kind: "function", symbol };
  if (symbol.kind === "device-function") return { kind: "function", symbol };
  if (symbol.kind === "cooperative-group") return { kind: "unknown", symbol };
  if (symbol.kind === "texture") return { kind: "texture", valueType: symbol.valueType, symbol };
  if (symbol.valueType === "texture2d") return { kind: "texture", valueType: symbol.valueType, symbol };
  if (symbol.valueType === "surface2d") return { kind: "surface", valueType: symbol.valueType, symbol };
  if (symbol.matrixTile) return { kind: "matrix-tile", valueType: symbol.valueType, symbol, matrixTile: symbol.matrixTile };
  if (symbol.kind === "local" && symbol.dimensions && symbol.dimensions.length > 0) {
    return {
      kind: "array",
      valueType: symbol.valueType,
      dimensions: symbol.dimensions,
      symbol,
    };
  }
  if (symbol.kind === "shared" || symbol.kind === "constant" || symbol.kind === "device-global") {
    if (symbol.valueType === "complex64" && (!symbol.dimensions || symbol.dimensions.length === 0)) {
      return { kind: "complex", valueType: symbol.valueType, symbol };
    }
    if (isCudaVectorType(symbol.valueType) && (!symbol.dimensions || symbol.dimensions.length === 0)) {
      return { kind: "vector", valueType: symbol.valueType, symbol };
    }
    return {
      kind: symbol.dimensions && symbol.dimensions.length > 0 ? "array" : "scalar",
      valueType: symbol.valueType,
      dimensions: symbol.dimensions,
      symbol,
    };
  }
  if (symbol.kind === "param" && symbol.pointer) {
    if (symbol.valueType === "devicepool") return { kind: "pool-pointer", valueType: "devicepool", symbol };
    return { kind: "pointer", valueType: symbol.valueType, symbol };
  }
  if (symbol.kind === "local" && symbol.pointer) {
    if (symbol.valueType === "voidptr") return { kind: "scalar", valueType: "voidptr", symbol };
    return { kind: "pointer", valueType: symbol.valueType, symbol };
  }
  if (symbol.kind === "local" && symbol.valueType === "complex64") {
    return { kind: "complex", valueType: symbol.valueType, symbol };
  }
  if (isCudaVectorType(symbol.valueType)) {
    return { kind: "vector", valueType: symbol.valueType, symbol };
  }
  return { kind: "scalar", valueType: symbol.valueType, symbol };
}

function validateScalarOperand(
  info: ExpressionInfo,
  span: SourceSpan,
  diagnostics: CudaLiteDiagnostic[],
): void {
  if (info.kind === "scalar" || info.kind === "unknown") return;
  if (info.kind === "matrix-tile") {
    diagnostics.push(error("unsupported-wmma-fragment-use", "WMMA fragments must be used through supported wmma::* operations", span));
    return;
  }
  if (info.kind === "pool-pointer") return;
  if (info.kind === "pointer" && info.symbol?.kind === "local") return;
  diagnostics.push(error("unsupported-scalar-expression", "expression must resolve to a scalar value", span));
}

function pointerComparable(
  left: ExpressionInfo,
  right: ExpressionInfo,
  leftExpression: CudaLiteExpression,
  rightExpression: CudaLiteExpression,
): boolean {
  if (isPointerLikeInfo(left) && isPointerLikeInfo(right)) return true;
  if (isPointerLikeInfo(left) && isNullPointerLiteral(rightExpression)) return true;
  return isPointerLikeInfo(right) && isNullPointerLiteral(leftExpression);
}

function conditionalPointerInfo(
  expression: Extract<CudaLiteExpression, { kind: "conditional" }>,
  consequent: ExpressionInfo,
  alternate: ExpressionInfo,
  diagnostics: CudaLiteDiagnostic[],
): ExpressionInfo | undefined {
  const consequentPointer = isPointerLikeInfo(consequent);
  const alternatePointer = isPointerLikeInfo(alternate);
  const consequentNull = isNullPointerLiteral(expression.consequent);
  const alternateNull = isNullPointerLiteral(expression.alternate);
  if (!consequentPointer && !alternatePointer) return undefined;
  if (!consequentPointer && !consequentNull) return undefined;
  if (!alternatePointer && !alternateNull) return undefined;
  const valueType = consequent.valueType ?? alternate.valueType;
  if (
    consequentPointer &&
    alternatePointer &&
    consequent.valueType !== undefined &&
    alternate.valueType !== undefined &&
    consequent.valueType !== alternate.valueType
  ) {
    diagnostics.push(error("unsupported-pointer-conditional", "conditional pointer expressions require matching pointer value types", expression.span));
  }
  return {
    kind: consequentPointer ? consequent.kind : alternate.kind,
    ...(valueType === undefined ? {} : { valueType }),
  };
}

function isPointerLikeInfo(info: ExpressionInfo): boolean {
  if (info.kind === "pointer" || info.kind === "pool-pointer" || info.kind === "address") return true;
  return info.kind === "scalar" && info.valueType === "voidptr";
}

function isNullPointerLiteral(expression: CudaLiteExpression): boolean {
  if (expression.kind === "number") return expression.value === 0;
  return expression.kind === "identifier" && (expression.name === "nullptr" || expression.name === "NULL");
}

function isVectorArithmeticOperator(operator: string): boolean {
  return operator === "+" || operator === "-" || operator === "*" || operator === "/";
}

function assignmentArithmeticOperator(operator: CudaLiteAssignmentExpression["operator"]): "+" | "-" | "*" | "/" | undefined {
  if (operator === "+=") return "+";
  if (operator === "-=") return "-";
  if (operator === "*=") return "*";
  if (operator === "/=") return "/";
  return undefined;
}

function expressionInfoForTextureRead(expression: Extract<CudaLiteExpression, { kind: "call" }>): ExpressionInfo {
  const valueType = expression.templateValueType ?? "float";
  return isCudaVectorType(valueType)
    ? { kind: "vector", valueType }
    : { kind: "scalar", valueType };
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

function textureCoordinateArgs(expression: Extract<CudaLiteExpression, { kind: "call" }>, callName: string): readonly CudaLiteExpression[] {
  if (callName === "tex1D" || callName === "tex1Dfetch") return expression.args.slice(1, 2);
  if (callName === "tex2DLod") return expression.args.slice(1, 3);
  return expression.args.slice(1);
}

function vectorArithmeticInfo(
  operator: string,
  left: ExpressionInfo,
  right: ExpressionInfo,
  leftExpression: CudaLiteExpression,
  rightExpression: CudaLiteExpression,
  diagnostics: CudaLiteDiagnostic[],
  requiredFeatures: Set<string>,
): ExpressionInfo | undefined {
  if (!isVectorArithmeticOperator(operator)) return undefined;
  const leftVectorType = left.kind === "vector" && isCudaVectorType(left.valueType) ? left.valueType : undefined;
  const rightVectorType = right.kind === "vector" && isCudaVectorType(right.valueType) ? right.valueType : undefined;
  if (!leftVectorType && !rightVectorType) return undefined;
  if (leftVectorType && rightVectorType) {
    if (leftVectorType !== rightVectorType) {
      diagnostics.push(error("unsupported-vector-argument", "vector arithmetic expects matching CUDA vector types", leftExpression.span));
      return { kind: "unknown" };
    }
    if (cudaVectorScalarType(leftVectorType) === "half") requiredFeatures.add("shader-f16");
    return { kind: "vector", valueType: leftVectorType };
  }
  const vectorType = leftVectorType ?? rightVectorType!;
  const scalarInfo = leftVectorType ? right : left;
  const scalarExpression = leftVectorType ? rightExpression : leftExpression;
  validateScalarOperand(scalarInfo, scalarExpression.span, diagnostics);
  if (cudaVectorScalarType(vectorType) === "half") requiredFeatures.add("shader-f16");
  return { kind: "vector", valueType: vectorType };
}

function isSupportedTextureReadType(type: CudaLiteScalarType | undefined): boolean {
  return type === undefined ||
    type === "float" ||
    type === "int" ||
    type === "uint" ||
    type === "uchar" ||
    type === "half" ||
    type === "half2" ||
    type === "bf16" ||
    type === "bf162" ||
    type === "float2" ||
    type === "float3" ||
    type === "float4" ||
    type === "int2" ||
    type === "int3" ||
    type === "int4" ||
    type === "uint2" ||
    type === "uint3" ||
    type === "uint4";
}

function lookupSymbol(name: string, scope: Scope, span: SourceSpan): SymbolInfo | undefined {
  let cursor: Scope | undefined = scope;
  while (cursor) {
    const symbol = cursor.symbols.get(name);
    if (symbol) return symbol;
    cursor = cursor.parent;
  }
  if (BUILTIN_VECTORS.has(name)) return { name, kind: "builtin-vector", span };
  if (BUILTIN_CALLS.has(name)) return { name, kind: "builtin-call", span };
  return undefined;
}

function unknownSymbolDiagnostic(name: string, scope: Scope, span: SourceSpan): CudaLiteDiagnostic {
  const nearest = nearestSymbolName(name, scope);
  const hint = nearest === undefined ? "" : `; nearest visible symbol '${nearest}'`;
  return error("unknown-symbol", `unknown CUDA-lite symbol '${name}'${hint}`, span);
}

function nearestSymbolName(name: string, scope: Scope): string | undefined {
  let best: { readonly name: string; readonly score: number } | undefined;
  const seen = new Set<string>();
  let cursor: Scope | undefined = scope;
  while (cursor) {
    for (const candidate of cursor.symbols.keys()) {
      if (seen.has(candidate)) continue;
      seen.add(candidate);
      const score = symbolSimilarityScore(name, candidate);
      if (score <= 0) continue;
      if (!best || score > best.score || (score === best.score && candidate.length < best.name.length)) {
        best = { name: candidate, score };
      }
    }
    cursor = cursor.parent;
  }
  return best?.score !== undefined && best.score >= 3 ? best.name : undefined;
}

function symbolSimilarityScore(left: string, right: string): number {
  if (left === right) return 100;
  let score = Math.max(0, Math.min(left.length, right.length) - levenshteinDistance(left, right));
  const leftParts = left.split("_").filter(Boolean);
  const rightParts = new Set(right.split("_").filter(Boolean));
  for (const part of leftParts) {
    if (rightParts.has(part)) score += Math.min(4, part.length);
  }
  if (left.at(0) === right.at(0)) score += 1;
  return score;
}

function levenshteinDistance(left: string, right: string): number {
  const previous = Array.from({ length: right.length + 1 }, () => 0);
  const current = Array.from({ length: right.length + 1 }, () => 0);
  for (let index = 0; index <= right.length; index++) previous[index] = index;
  for (let row = 1; row <= left.length; row++) {
    current[0] = row;
    for (let column = 1; column <= right.length; column++) {
      const substitution = previous[column - 1]! + (left[row - 1] === right[column - 1] ? 0 : 1);
      current[column] = Math.min(previous[column]! + 1, current[column - 1]! + 1, substitution);
    }
    for (let index = 0; index <= right.length; index++) previous[index] = current[index]!;
  }
  return previous[right.length]!;
}

function createScope(parent?: Scope): Scope {
  return parent === undefined ? { symbols: new Map() } : { symbols: new Map(), parent };
}

function formatArity(minArgs: number, maxArgs: number): string {
  return minArgs === maxArgs ? String(minArgs) : `${minArgs}-${maxArgs}`;
}

function validateExpressionStatement(
  expression: CudaLiteExpression,
  params: ReadonlyMap<string, CudaLiteParam>,
  guardDepth: number,
  diagnostics: CudaLiteDiagnostic[],
): void {
  if (expression.kind !== "assignment") return;
  const root = rootIdentifier(expression.left);
  if (!root) return;
  const param = params.get(root);
  if (!param?.pointer) return;
  if (expression.left.kind === "identifier" && isPointerRebaseOperator(expression.operator)) return;
  if (param.constant) {
    diagnostics.push(error("const-pointer-write", `cannot write through const pointer '${root}'`, expression.span));
  }
  if (guardDepth === 0) {
    diagnostics.push(warning("unguarded-write", `write to pointer '${root}' has no syntactic bounds guard`, expression.span));
  }
}

function validateDeclaredSymbolName(
  name: string,
  span: SourceSpan,
  diagnostics: CudaLiteDiagnostic[],
): void {
  if (BUILTIN_VECTORS.has(name)) {
    diagnostics.push(error("reserved-symbol", `symbol '${name}' conflicts with a CUDA-lite builtin`, span));
  }
}

function validateSideEffectPlacement(
  expression: CudaLiteExpression,
  allowRootSideEffect: boolean,
  diagnostics: CudaLiteDiagnostic[],
  allowRootSequenceSideEffects = false,
  allowRootCallSideEffects = false,
): void {
  if (allowRootSequenceSideEffects && expression.kind === "sequence") {
    validateSideEffectPlacement(expression, true, diagnostics, false, allowRootCallSideEffects);
    return;
  }
  if (allowRootSideEffect && expression.kind === "assignment") {
    validateAssignmentStatementSideEffects(expression, diagnostics);
    return;
  }
  const visit = (node: CudaLiteExpression, root: boolean): void => {
    if ((node.kind === "assignment" || node.kind === "update") && !(allowRootSideEffect && root)) {
      diagnostics.push(error(
        "side-effect-expression",
        "assignments and ++/-- must be standalone statements or for-loop clauses",
        node.span,
      ));
    }
    if (node.kind === "call" && isSideEffectingCudaRuntimeCallName(expressionName(node.callee)) && !(allowRootCallSideEffects && root)) {
      diagnostics.push(error(
        "side-effect-expression",
        "side-effecting CUDA runtime calls must be standalone statements, variable initializers, assignment RHS, or for-loop clauses",
        node.span,
      ));
    }
    forEachExpressionChild(node, (child) => {
      visit(child, allowRootSideEffect && root && node.kind === "sequence");
    });
  };
  visit(expression, true);
}

function validateAssignmentStatementSideEffects(
  expression: CudaLiteAssignmentExpression,
  diagnostics: CudaLiteDiagnostic[],
): void {
  const visitNoSideEffect = (node: CudaLiteExpression): void => {
    if (node.kind === "assignment" || node.kind === "update") {
      diagnostics.push(error(
        "side-effect-expression",
        "assignments and ++/-- must be standalone statements or for-loop clauses",
        node.span,
      ));
      return;
    }
    forEachExpressionChild(node, visitNoSideEffect);
  };
  let cursor: CudaLiteAssignmentExpression = expression;
  while (true) {
    visitNoSideEffect(cursor.left);
    if (cursor.right.kind !== "assignment") {
      if (cursor.right.kind === "sequence") validateSideEffectPlacement(cursor.right, true, diagnostics);
      else validateSideEffectPlacement(cursor.right, false, diagnostics, false, cursor.right.kind === "call");
      return;
    }
    cursor = cursor.right;
  }
}

function expressionIsCall(expression: CudaLiteExpression): boolean {
  return expression.kind === "call";
}

function isSideEffectingCudaRuntimeCallName(name: string | undefined): boolean {
  return name !== undefined && (
    isCudaIntegerRuntimeQueryCall(name) ||
    name === "cudaEventElapsedTime" ||
    name === "cudaGraphSetConditional" ||
    isCudaRuntimeCopyCall(name) ||
    isHostManagedRuntimeNoopCall(name)
  );
}

function isCudaIntegerRuntimeQueryCall(callName: string): boolean {
  return callName === "cudaGetDevice" ||
    callName === "cudaGetDeviceCount" ||
    callName === "cudaDeviceGetAttribute" ||
    callName === "cudaDeviceGetLimit" ||
    callName === "cudaThreadGetLimit" ||
    callName === "cudaDeviceCanAccessPeer" ||
    callName === "cudaGetDeviceFlags" ||
    callName === "cudaMemGetInfo" ||
    callName === "cudaOccupancyMaxActiveBlocksPerMultiprocessor" ||
    callName === "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags" ||
    callName === "cudaOccupancyMaxPotentialBlockSize" ||
    callName === "cudaOccupancyMaxPotentialBlockSizeWithFlags" ||
    callName === "cudaOccupancyAvailableDynamicSMemPerBlock" ||
    callName === "cudaDeviceGetCacheConfig" ||
    callName === "cudaDeviceGetSharedMemConfig" ||
    callName === "cudaThreadGetCacheConfig" ||
    callName === "cudaThreadExchangeStreamCaptureMode" ||
    callName === "cudaDeviceGetStreamPriorityRange" ||
    callName === "cudaStreamCreate" ||
    callName === "cudaStreamCreateWithFlags" ||
    callName === "cudaStreamCreateWithPriority" ||
    callName === "cudaStreamGetDevice" ||
    callName === "cudaStreamGetFlags" ||
    callName === "cudaStreamGetId" ||
    callName === "cudaStreamGetPriority" ||
    callName === "cudaStreamIsCapturing" ||
    callName === "cudaStreamGetCaptureInfo" ||
    callName === "cudaStreamGetCaptureInfo_v2" ||
    callName === "cudaStreamEndCapture" ||
    callName === "cudaGraphCreate" ||
    callName === "cudaGraphInstantiate" ||
    callName === "cudaGraphInstantiateWithFlags" ||
    callName === "cudaGraphExecUpdate" ||
    callName === "cudaEventCreate" ||
    callName === "cudaEventCreateWithFlags" ||
    callName === "cudaRuntimeGetVersion" ||
    callName === "cudaDriverGetVersion";
}

function validateBarrierStatement(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  diagnostics: CudaLiteDiagnostic[],
): void {
  const name = expressionName(expression.callee) ?? "barrier";
  const maxArgs = name === "__syncwarp" ? 1 : 0;
  if (expression.args.length > maxArgs) {
    diagnostics.push(error("invalid-call-arity", `${name} expects ${maxArgs === 0 ? "0" : "0-1"} arguments`, expression.span));
  }
}

function collectSharedDeclarations(
  statements: readonly CudaLiteStatement[],
  options: CudaLiteAnalyzeOptions,
): readonly CudaLiteVarDecl[] {
  const declarations: CudaLiteVarDecl[] = [];
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.storage === "shared") {
        const dimensions = resolvedSharedDimensions(item, options);
        declarations.push(dimensions ? { ...item, dimensions } : item);
      }
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for" || item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return declarations;
}

function collectSharedDeclarationsFromBodies(
  bodies: readonly (readonly CudaLiteStatement[])[],
  options: CudaLiteAnalyzeOptions,
): readonly CudaLiteVarDecl[] {
  const declarations = new Map<string, CudaLiteVarDecl>();
  for (const body of bodies) {
    for (const declaration of collectSharedDeclarations(body, options)) {
      if (!declarations.has(declaration.name)) declarations.set(declaration.name, declaration);
    }
  }
  return [...declarations.values()];
}

function collectReferencedSymbolNames(
  bodies: readonly (readonly CudaLiteStatement[])[],
): ReadonlySet<string> {
  const names = new Set<string>();
  const walkExpression = (expression: CudaLiteExpression): void => {
    if (expression.kind === "identifier") names.add(expression.name);
    forEachExpressionChild(expression, walkExpression);
  };
  const walkStatements = (statements: readonly CudaLiteStatement[]): void => {
    for (const statement of statements) {
      switch (statement.kind) {
        case "block":
          walkStatements(statement.body);
          break;
        case "var":
          if (statement.init) walkExpression(statement.init);
          break;
        case "dim3":
          for (const arg of statement.args) walkExpression(arg);
          break;
        case "cooperative-group":
          if (statement.partitionPredicate) walkExpression(statement.partitionPredicate);
          break;
        case "kernel-launch":
          for (const arg of statement.grid) walkExpression(arg);
          for (const arg of statement.block) walkExpression(arg);
          for (const arg of statement.args) walkExpression(arg);
          break;
        case "asm":
          for (const output of statement.outputs ?? (statement.output === undefined ? [] : [statement.output])) walkExpression(output);
          for (const input of statement.inputs) walkExpression(input);
          break;
        case "expr":
          walkExpression(statement.expression);
          break;
        case "if":
          walkExpression(statement.condition);
          walkStatements(statement.consequent);
          if (statement.alternate) walkStatements(statement.alternate);
          break;
        case "for":
          if (statement.init?.kind === "var") {
            if (statement.init.init) walkExpression(statement.init.init);
          } else if (statement.init) {
            walkExpression(statement.init);
          }
          if (statement.condition) walkExpression(statement.condition);
          if (statement.update) walkExpression(statement.update);
          walkStatements(statement.body);
          break;
        case "while":
          walkExpression(statement.condition);
          walkStatements(statement.body);
          break;
        case "do-while":
          walkStatements(statement.body);
          walkExpression(statement.condition);
          break;
        case "return":
          if (statement.value) walkExpression(statement.value);
          break;
        case "continue":
        case "break":
          break;
      }
    }
  };
  for (const body of bodies) walkStatements(body);
  return names;
}

function resolvedSharedDimensions(
  statement: CudaLiteVarDecl,
  options: CudaLiteAnalyzeOptions,
): readonly number[] | undefined {
  if (statement.storage !== "shared" || !statement.dynamicShared) return undefined;
  const elements = options.dynamicSharedMemory?.[statement.name];
  if (elements === undefined) return undefined;
  const leading = positiveInteger(elements, `dynamicSharedMemory.${statement.name}`);
  return statement.dimensions.length === 0 ? [leading] : [leading, ...statement.dimensions];
}

function expressionIsDivergent(
  expression: CudaLiteExpression,
  params: ReadonlyMap<string, CudaLiteParam>,
): boolean {
  let divergent = false;
  const walk = (item: CudaLiteExpression): void => {
    const name = expressionName(item);
    if (name === "threadIdx") divergent = true;
    if (item.kind === "index") {
      const root = rootIdentifier(item.target);
      if (root && params.get(root)?.pointer) divergent = true;
    }
    forEachExpressionChild(item, walk);
  };
  walk(expression);
  return divergent;
}

function isBarrierCall(expression: CudaLiteExpression): expression is Extract<CudaLiteExpression, { kind: "call" }> {
  if (expression.kind !== "call") return false;
  const name = expressionName(expression.callee);
  return name === "__syncthreads" || name === "__syncwarp";
}

function isInlineAsmBarrier(statement: CudaLiteStatement): statement is Extract<CudaLiteStatement, { kind: "asm" }> {
  return statement.kind === "asm" && classifyInlineAsm(statement.template)?.kind === "bar-sync";
}

function validateDivergentReturnsBeforeBarriers(
  statements: readonly CudaLiteStatement[],
  params: ReadonlyMap<string, CudaLiteParam>,
  diagnostics: CudaLiteDiagnostic[],
  workgroupSize: readonly [number, number, number],
): void {
  const uniformity = collectBarrierUniformity(statements, params, workgroupSize);
  const visitBlock = (
    body: readonly CudaLiteStatement[],
    divergentDepth: number,
    initialBarrierLater = false,
    initialContinueBarrierLater = false,
  ): boolean => {
    let barrierLater = initialBarrierLater;
    let localBarrierLater = false;
    let continueBarrierLater = initialContinueBarrierLater;
    let containsBarrier = false;
    for (let index = body.length - 1; index >= 0; index--) {
      const statement = body[index]!;
      const info = visitStatement(statement, divergentDepth, barrierLater, continueBarrierLater || localBarrierLater);
      containsBarrier = info.containsBarrier || containsBarrier;
      barrierLater = barrierLater || info.containsBarrier;
      localBarrierLater = localBarrierLater || info.containsBarrier;
    }
    return containsBarrier;
  };

  const visitStatement = (
    statement: CudaLiteStatement,
    divergentDepth: number,
    barrierLater: boolean,
    continueBarrierLater: boolean,
  ): { readonly containsBarrier: boolean } => {
    switch (statement.kind) {
      case "block":
        return { containsBarrier: visitBlock(statement.body, divergentDepth, barrierLater, continueBarrierLater) };
      case "expr":
        return { containsBarrier: isBarrierCall(statement.expression) };
      case "asm":
        return { containsBarrier: isInlineAsmBarrier(statement) };
      case "return":
        if (divergentDepth > 0 && barrierLater) {
          diagnostics.push(warning(
            "divergent-return-before-barrier",
            "thread-dependent return before a later barrier would make WGSL barrier control flow non-uniform",
            statement.span,
          ));
        }
        return { containsBarrier: false };
      case "break":
        if (divergentDepth > 0 && barrierLater) {
          diagnostics.push(warning(
            "divergent-break-before-barrier",
            "thread-dependent break before a later barrier would make WGSL barrier control flow non-uniform",
            statement.span,
          ));
        }
        return { containsBarrier: false };
      case "continue":
        if (divergentDepth > 0 && continueBarrierLater) {
          diagnostics.push(error(
            "divergent-continue-before-barrier",
            "thread-dependent continue before a later barrier would make WGSL barrier control flow non-uniform",
            statement.span,
          ));
        }
        return { containsBarrier: false };
      case "if": {
        const nestedDivergentDepth = divergentDepth + (expressionMayBeNonUniformBeforeBarrier(statement.condition, uniformity) ? 1 : 0);
        const consequentHasBarrier = visitBlock(statement.consequent, nestedDivergentDepth, barrierLater, continueBarrierLater);
        const alternateHasBarrier = statement.alternate ? visitBlock(statement.alternate, nestedDivergentDepth, barrierLater, continueBarrierLater) : false;
        return { containsBarrier: consequentHasBarrier || alternateHasBarrier };
      }
      case "for": {
        const nestedDivergentDepth = divergentDepth + (statement.condition && expressionMayBeNonUniformBeforeBarrier(statement.condition, uniformity) ? 1 : 0);
        return { containsBarrier: visitBlock(statement.body, nestedDivergentDepth, barrierLater, false) };
      }
      case "while": {
        const nestedDivergentDepth = divergentDepth + (expressionMayBeNonUniformBeforeBarrier(statement.condition, uniformity) ? 1 : 0);
        return { containsBarrier: visitBlock(statement.body, nestedDivergentDepth, barrierLater, false) };
      }
      case "do-while": {
        const nestedDivergentDepth = divergentDepth + (expressionMayBeNonUniformBeforeBarrier(statement.condition, uniformity) ? 1 : 0);
        return { containsBarrier: visitBlock(statement.body, nestedDivergentDepth, barrierLater, false) };
      }
      default:
        return { containsBarrier: false };
    }
  };

  visitBlock(statements, 0);
}

interface BarrierUniformityContext {
  readonly params: ReadonlyMap<string, CudaLiteParam>;
  readonly locals: ReadonlyMap<string, boolean>;
  readonly cooperativeGroups: ReadonlyMap<string, CudaLiteCooperativeGroupDecl>;
  readonly workgroupSize: readonly [number, number, number];
}

function collectBarrierUniformity(
  statements: readonly CudaLiteStatement[],
  params: ReadonlyMap<string, CudaLiteParam>,
  workgroupSize: readonly [number, number, number],
): BarrierUniformityContext {
  const locals = new Map<string, boolean>();
  const cooperativeGroups = new Map<string, CudaLiteCooperativeGroupDecl>();
  const context: BarrierUniformityContext = { params, locals, cooperativeGroups, workgroupSize };
  const visitStatements = (body: readonly CudaLiteStatement[]): void => {
    for (const statement of body) {
      if (statement.kind === "var") {
        locals.set(statement.name, statement.init ? expressionMayBeNonUniformBeforeBarrier(statement.init, context) : true);
      } else if (statement.kind === "dim3") {
        locals.set(statement.name, statement.args.some((arg) => expressionMayBeNonUniformBeforeBarrier(arg, context)));
      } else if (statement.kind === "cooperative-group") {
        cooperativeGroups.set(statement.name, statement);
      } else if (statement.kind === "expr" && statement.expression.kind === "assignment" && statement.expression.left.kind === "identifier") {
        locals.set(statement.expression.left.name, expressionMayBeNonUniformBeforeBarrier(statement.expression.right, context));
      }
      if (statement.kind === "block" || statement.kind === "for" || statement.kind === "while" || statement.kind === "do-while") visitStatements(statement.body);
      if (statement.kind === "if") {
        visitStatements(statement.consequent);
        if (statement.alternate) visitStatements(statement.alternate);
      }
    }
  };
  visitStatements(statements);
  return context;
}

function expressionMayBeNonUniformBeforeBarrier(
  expression: CudaLiteExpression,
  context: BarrierUniformityContext,
): boolean {
  if (expressionIsDivergent(expression, context.params)) return true;
  if (expression.kind === "identifier") {
    const param = context.params.get(expression.name);
    if (param) return param.pointer;
    if (context.locals.has(expression.name)) return context.locals.get(expression.name) ?? true;
    return !BUILTIN_VECTORS.has(expression.name);
  }
  if (expression.kind === "call") {
    const callee = expression.callee;
    if (callee.kind === "member" && callee.object.kind === "identifier") {
      const group = context.cooperativeGroups.get(callee.object.name);
      if (group) {
        const argsNonUniform = expression.args.some((arg) => expressionMayBeNonUniformBeforeBarrier(arg, context));
        if (argsNonUniform) return true;
        if (callee.property === "meta_group_size" || callee.property === "size") return false;
        if (callee.property === "meta_group_rank") return cooperativeGroupMetaRankMayBeNonUniform(group, context.workgroupSize);
        if (callee.property === "thread_rank") return true;
      }
    }
  }
  if (expression.kind === "member" && expression.object.kind === "identifier") {
    const name = expression.object.name;
    return name !== "blockIdx" && name !== "blockDim" && name !== "gridDim";
  }
  let nonUniform = false;
  forEachExpressionChild(expression, (child) => {
    if (expressionMayBeNonUniformBeforeBarrier(child, context)) nonUniform = true;
  });
  return nonUniform;
}

function cooperativeGroupMetaRankMayBeNonUniform(
  group: CudaLiteCooperativeGroupDecl,
  workgroupSize: readonly [number, number, number],
): boolean {
  if (group.groupKind !== "tile") return false;
  const tileSize = group.tileSize ?? 32;
  const blockSize = workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  return blockSize > tileSize;
}

function forEachExpressionChild(
  expression: CudaLiteExpression,
  visit: (child: CudaLiteExpression) => void,
): void {
  switch (expression.kind) {
    case "number":
    case "string":
    case "identifier":
      return;
    case "initializer":
      for (const element of expression.elements) visit(element);
      return;
    case "cast":
      visit(expression.expression);
      return;
    case "member":
      visit(expression.object);
      return;
    case "index":
      visit(expression.target);
      visit(expression.index);
      return;
    case "call":
      visit(expression.callee);
      for (const arg of expression.args) visit(arg);
      return;
    case "unary":
    case "update":
      visit(expression.argument);
      return;
    case "binary":
    case "assignment":
      visit(expression.left);
      visit(expression.right);
      return;
    case "conditional":
      visit(expression.condition);
      visit(expression.consequent);
      visit(expression.alternate);
      return;
    case "sequence":
      for (const item of expression.expressions) visit(item);
      return;
  }
}

export function expressionName(expression: CudaLiteExpression): string | undefined {
  if (expression.kind === "identifier") return expression.name;
  return undefined;
}

export function rootIdentifier(expression: CudaLiteExpression): string | undefined {
  if (expression.kind === "identifier") return expression.name;
  if (expression.kind === "cast") return rootIdentifier(expression.expression);
  if (expression.kind === "index") return rootIdentifier(expression.target);
  if (expression.kind === "member") return rootIdentifier(expression.object);
  if (expression.kind === "unary" && (expression.operator === "&" || expression.operator === "*")) return rootIdentifier(expression.argument);
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) return rootIdentifier(expression.left);
  return undefined;
}

export function normalizeWorkgroupSize(
  value: readonly [number, number, number],
): readonly [number, number, number] {
  return [
    positiveInteger(value[0], "workgroupSize[0]"),
    positiveInteger(value[1], "workgroupSize[1]"),
    positiveInteger(value[2], "workgroupSize[2]"),
  ];
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new CudaLiteCompilerError(`${name} must be a positive integer`, [{
      code: "invalid-workgroup-size",
      severity: "error",
      message: `${name} must be a positive integer`,
      span: { start: 0, end: 0, line: 1, column: 1 },
    }]);
  }
  return value;
}

function error(code: string, message: string, span: SourceSpan): CudaLiteDiagnostic {
  return { code, severity: "error", message, span };
}

function warning(code: string, message: string, span: SourceSpan): CudaLiteDiagnostic {
  return { code, severity: "warning", message, span };
}
