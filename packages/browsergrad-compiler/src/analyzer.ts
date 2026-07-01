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
import { classifyInlineAsm, inlineAsmSupportedList } from "./ptx_tile_ops.js";
import { sizeofCudaType } from "./type_layout.js";
import {
  CUDA_VECTOR_TYPES,
  CUDA_VECTOR_CONSTRUCTORS,
  cudaVectorConstructorType,
  cudaVectorFieldIndex,
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
  type CudaLiteVectorType,
} from "./vector_types.js";

const DEFAULT_WORKGROUP_SIZE: readonly [number, number, number] = [256, 1, 1];
const BUILTIN_VECTORS = new Set(["threadIdx", "blockIdx", "blockDim", "gridDim"]);
const BUILTIN_CALLS = new Map<string, readonly [min: number, max: number]>([
  ...CUDA_INTRINSICS.map((intrinsic) => [intrinsic.name, intrinsic.arity] as const),
  ["__syncthreads", [0, 0]],
  ["__syncwarp", [0, 1]],
  ["__threadfence", [0, 0]],
  ["__trap", [0, 0]],
  ["__shfl_sync", [3, 4]],
  ["__shfl_down_sync", [3, 4]],
  ["__shfl_up_sync", [3, 4]],
  ["__shfl_xor_sync", [3, 4]],
  ["__any_sync", [2, 2]],
  ["__all_sync", [2, 2]],
  ["__ballot_sync", [2, 2]],
  ["__reduce_add_sync", [2, 2]],
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
  ["curand_uniform", [1, 1]],
  ["curand_uniform_double", [1, 1]],
  ["curand_normal", [1, 1]],
  ["curand_normal_double", [1, 1]],
  ["cudaDeviceSynchronize", [0, 0]],
  ["cudaStreamCreate", [1, 1]],
  ["cudaStreamCreateWithFlags", [2, 2]],
  ["cudaStreamDestroy", [1, 1]],
  ["cudaStreamSynchronize", [1, 1]],
  ["cudaEventCreate", [1, 1]],
  ["cudaEventCreateWithFlags", [2, 2]],
  ["cudaEventDestroy", [1, 1]],
  ["cudaEventRecord", [1, 2]],
  ["cudaEventSynchronize", [1, 1]],
  ["cudaMemcpy", [4, 4]],
  ["cudaMemcpyAsync", [5, 5]],
  ["cudaMemcpyPeerAsync", [6, 6]],
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
  ["__halves2bfloat162", [2, 2]],
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
  const diagnostics: CudaLiteDiagnostic[] = [];
  const requiredFeatures = new Set<string>();
  const atomicParams = new Set<string>();
  const atomicShared = new Set<string>();
  const atomicDeviceGlobals = new Set<string>();
  const params = new Map(kernel.params.map((param) => [param.name, param]));
  const declaredNames = new Set<string>();
  const rootScope = createScope();

  for (const constant of ast.constants) {
    declareConstant(constant, rootScope, declaredNames, requiredFeatures, diagnostics, options);
  }
  for (const global of ast.deviceGlobals) {
    declareDeviceGlobal(global, rootScope, declaredNames, requiredFeatures, diagnostics, options);
  }
  for (const texture of ast.textures) {
    declareTexture(texture, rootScope, declaredNames, diagnostics);
  }
  for (const fn of ast.functions) {
    if (selectedDeviceFunctionAsKernel && fn.name === kernel.name) continue;
    declareDeviceFunction(fn, rootScope, declaredNames, requiredFeatures, diagnostics, options);
  }
  const rootDeclaredNames = new Set(declaredNames);

  for (const param of kernel.params) {
    if (declaredNames.has(param.name)) {
      diagnostics.push(error("duplicate-symbol", `duplicate parameter '${param.name}'`, param.span));
    }
    validateDeclaredSymbolName(param.name, param.span, diagnostics);
    declaredNames.add(param.name);
    rootScope.symbols.set(param.name, symbolForParam(param, "param"));
    if (requiresShaderF16(param.valueType)) requiredFeatures.add("shader-f16");
    validateF64Type(param.valueType, param.span, diagnostics, options);
  }

  const declareVar = (statement: CudaLiteVarDecl, scope: Scope, names: Set<string>): void => {
    const dimensions = resolvedSharedDimensions(statement, options) ?? statement.dimensions;
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
    if (statement.matrixTile) validateMatrixTileDeclaration(statement, requiredFeatures, diagnostics);
    else validateF64Type(statement.valueType, statement.span, diagnostics, options);
  };

  const walkExpression = (expression: CudaLiteExpression, scope: Scope): ExpressionInfo => {
    if (expression.kind === "call") {
      return validateCallExpression(expression, scope, params, atomicParams, atomicShared, atomicDeviceGlobals, requiredFeatures, diagnostics, walkExpression, options);
    }
    return validateNonCallExpression(expression, scope, diagnostics, walkExpression, requiredFeatures);
  };

  for (const constant of ast.constants) {
    validateGlobalConstantInitializer(constant, rootScope, diagnostics, walkExpression);
  }
  for (const global of ast.deviceGlobals) {
    validateDeviceGlobalInitializer(global, rootScope, diagnostics, walkExpression);
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
          if (!statement.matrixTile && requiresShaderF16(statement.valueType)) requiredFeatures.add("shader-f16");
          if (!statement.matrixTile && statement.pointer && !isSupportedLocalPointer(statement, scope)) {
            diagnostics.push(error("unsupported-local-pointer", "local pointer declarations are not supported in CUDA-lite yet", statement.span));
          }
          if (!statement.matrixTile && statement.storage === "local" && statement.dimensions.length > 0 && statement.init) {
            validateArrayInitializer(statement, scope, diagnostics, walkExpression);
          }
          if (statement.dynamicShared && !resolvedSharedDimensions(statement, options)) {
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
            if (statement.init) validateSideEffectPlacement(statement.init, false, diagnostics);
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
            requiredFeatures.add("subgroups");
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
          diagnostics.push({
            ...error(
              "unsupported-dynamic-parallelism",
              `device-side kernel launch '${statement.callee}<<<...>>>' requires explicit runtime orchestration in CUDA-lite`,
              statement.span,
            ),
            severity: options.referenceDynamicParallelism ? "warning" : "error",
          });
          for (const arg of statement.grid) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
          for (const arg of statement.block) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
          for (const arg of statement.args) walkExpression(arg, scope);
          break;
        case "asm":
          validateInlineAsmStatement(statement, scope, diagnostics, walkExpression);
          break;
        case "expr":
          if (isBarrierCall(statement.expression)) {
            validateBarrierStatement(statement.expression, diagnostics);
            if (divergentDepth > 0) {
              diagnostics.push(error("divergent-barrier", `${expressionName(statement.expression.callee) ?? "barrier"}() cannot appear in divergent control flow`, statement.span));
            }
            break;
          } else {
            validateSideEffectPlacement(statement.expression, true, diagnostics);
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
            if (!statement.init.matrixTile && requiresShaderF16(statement.init.valueType)) requiredFeatures.add("shader-f16");
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
            if (statement.init.init) validateSideEffectPlacement(statement.init.init, false, diagnostics);
          } else if (statement.init) {
            validateSideEffectPlacement(statement.init, true, diagnostics);
            walkExpression(statement.init, loopScope);
          }
          if (statement.condition) validateSideEffectPlacement(statement.condition, false, diagnostics);
          if (statement.condition) walkExpression(statement.condition, loopScope);
          if (statement.update) validateSideEffectPlacement(statement.update, true, diagnostics);
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
          if (hasContinueTargetingDoWhile(statement.body)) {
            diagnostics.push(error("unsupported-do-while-continue", "continue inside do-while is not supported until WGSL continuing-condition lowering is explicit", statement.span));
          }
          const divergent = expressionIsDivergent(statement.condition, params);
          walkStatements(statement.body, createScope(scope), guardDepth, divergent ? divergentDepth + 1 : divergentDepth, loopDepth + 1, new Set());
          break;
        }
        case "return":
          if (statement.value) {
            validateSideEffectPlacement(statement.value, false, diagnostics);
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
    const functionScope = createScope(rootScope);
    const functionDeclaredNames = new Set(rootDeclaredNames);
    for (const param of fn.params) {
      if (functionDeclaredNames.has(param.name)) {
        diagnostics.push(error("duplicate-symbol", `duplicate parameter '${param.name}'`, param.span));
      }
      validateDeclaredSymbolName(param.name, param.span, diagnostics);
      functionDeclaredNames.add(param.name);
      functionScope.symbols.set(param.name, symbolForParam(param, "local"));
      if (requiresShaderF16(param.valueType)) requiredFeatures.add("shader-f16");
      validateF64Type(param.valueType, param.span, diagnostics, options);
    }
    walkStatements(fn.body, functionScope, 0, 0, 0, functionDeclaredNames);
    validateDivergentReturnsBeforeBarriers(fn.body, new Map(fn.params.map((param) => [param.name, param])), diagnostics, options.workgroupSize ?? DEFAULT_WORKGROUP_SIZE);
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
    diagnostics.push(error("missing-feature-subgroups", "bg_subgroup_add requires WebGPU subgroups support", kernel.span));
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
  return {
    name: analysis.kernel.name,
    params: analysis.kernel.params,
    constants: analysis.constants,
    deviceGlobals: analysis.deviceGlobals,
    textures: analysis.textures,
    functions: analysis.functions,
    body: analysis.kernel.body,
    sharedDeclarations: collectSharedDeclarationsFromBodies([analysis.kernel.body, ...analysis.functions.map((fn) => fn.body)], options),
    requiredFeatures: analysis.requiredFeatures,
    atomicParams: analysis.atomicParams,
    atomicShared: analysis.atomicShared,
    atomicDeviceGlobals: analysis.atomicDeviceGlobals,
    workgroupSize: normalizeWorkgroupSize(options.workgroupSize ?? DEFAULT_WORKGROUP_SIZE),
  };
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
): void {
  if (type !== "double") return;
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
  validateF64Type(constant.valueType, constant.span, diagnostics, options);
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
  validateF64Type(global.valueType, global.span, diagnostics, options);
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
  validateF64Type(fn.returnType, fn.span, diagnostics, options);
  for (const param of fn.params) {
    if (requiresShaderF16(param.valueType)) requiredFeatures.add("shader-f16");
    validateF64Type(param.valueType, param.span, diagnostics, options);
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
    diagnostics.push(error("unsupported-local-array-init", "local array initializers must use CUDA/C braced initializer syntax", statement.init.span));
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
    const info = validateReadPointerExpression(expression.left, scope, diagnostics, walkExpression);
    validateScalarOperand(walkExpression(expression.right, scope), expression.right.span, diagnostics);
    return info;
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
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    return pointerSourceType(expression.left, scope);
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
    type === "half" ||
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

function validateInlineAsmStatement(
  statement: Extract<CudaLiteStatement, { kind: "asm" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const op = classifyInlineAsm(statement.template);
  const outputs = statement.outputs ?? (statement.output === undefined ? [] : [statement.output]);
  if (!op) {
    diagnostics.push(error("unsupported-inline-asm", `only ${inlineAsmSupportedList()} inline PTX are supported in CUDA-lite v0`, statement.span));
  }
  const outputInfos = outputs.map((output) => walkExpression(output, scope));
  for (let index = 0; index < outputs.length; index++) {
    const output = outputs[index]!;
    validateLValueExpression(output, scope, diagnostics, walkExpression);
    validateScalarOperand(outputInfos[index]!, output.span, diagnostics);
  }
  if (op?.kind === "fma-rn-f32" && (outputs.length !== 1 || statement.inputs.length !== 2)) {
    diagnostics.push(error("invalid-inline-asm-operands", "fma.rn.f32 inline PTX expects one output operand and exactly two input operands", statement.span));
  }
  if (op?.kind === "laneid" && (outputs.length !== 1 || statement.inputs.length !== 0)) {
    diagnostics.push(error("invalid-inline-asm-operands", "laneid inline PTX expects one output operand and no input operands", statement.span));
  }
  if (op?.kind === "laneid" && outputInfos[0]?.valueType !== undefined && outputInfos[0]?.valueType !== "uint" && outputInfos[0]?.valueType !== "int") {
    diagnostics.push(error("invalid-inline-asm-operands", "laneid inline PTX writes an integer output operand", outputs[0]?.span ?? statement.span));
  }
  if (op?.kind === "lanemask-lt" && (outputs.length !== 1 || statement.inputs.length !== 0)) {
    diagnostics.push(error("invalid-inline-asm-operands", "lanemask_lt inline PTX expects one output operand and no input operands", statement.span));
  }
  if (op?.kind === "lanemask-lt" && outputInfos[0]?.valueType !== undefined && outputInfos[0]?.valueType !== "uint" && outputInfos[0]?.valueType !== "int") {
    diagnostics.push(error("invalid-inline-asm-operands", "lanemask_lt inline PTX writes an integer output operand", outputs[0]?.span ?? statement.span));
  }
  if (op?.kind === "globaltimer-u64" && (outputs.length !== 1 || statement.inputs.length !== 0)) {
    diagnostics.push(error("invalid-inline-asm-operands", "globaltimer inline PTX expects one output operand and no input operands", statement.span));
  }
  if (op?.kind === "globaltimer-u64" && outputInfos[0]?.valueType !== undefined && outputInfos[0]?.valueType !== "uint" && outputInfos[0]?.valueType !== "int") {
    diagnostics.push(error("invalid-inline-asm-operands", "globaltimer inline PTX writes an integer output operand", outputs[0]?.span ?? statement.span));
  }
  if (op?.kind === "bfind-u32" && (outputs.length !== 1 || statement.inputs.length !== 1)) {
    diagnostics.push(error("invalid-inline-asm-operands", "bfind.u32 inline PTX expects one output operand and one input operand", statement.span));
  }
  if (op?.kind === "bfind-u32" && outputInfos[0]?.valueType !== undefined && outputInfos[0]?.valueType !== "uint") {
    diagnostics.push(error("invalid-inline-asm-operands", "bfind.u32 inline PTX writes a uint output operand", outputs[0]?.span ?? statement.span));
  }
  if (op?.kind === "u8x4-sad-add" && (outputs.length !== 1 || statement.inputs.length !== 3)) {
    diagnostics.push(error("invalid-inline-asm-operands", "vabsdiff4.u32.u32.u32.add inline PTX expects one output operand and three input operands", statement.span));
  }
  if (op?.kind === "u8x4-sad-add" && outputInfos[0]?.valueType !== undefined && outputInfos[0]?.valueType !== "uint" && outputInfos[0]?.valueType !== "int") {
    diagnostics.push(error("invalid-inline-asm-operands", "vabsdiff4.u32.u32.u32.add inline PTX writes an integer output operand", outputs[0]?.span ?? statement.span));
  }
  if (op?.kind === "ldmatrix") {
    if (outputs.length !== op.matrices || statement.inputs.length !== 1) {
      diagnostics.push(error("invalid-inline-asm-operands", `ldmatrix.x${op.matrices} inline PTX expects ${op.matrices} output operand(s) and one shared-address input operand`, statement.span));
    }
    for (let index = 0; index < outputs.length; index++) {
      const type = outputInfos[index]?.valueType;
      if (type !== undefined && type !== "uint" && type !== "int") {
        diagnostics.push(error("invalid-inline-asm-operands", "ldmatrix inline PTX writes integer register carrier operands", outputs[index]?.span ?? statement.span));
      }
    }
  }
  if (op?.kind === "mma-m16n8k16") {
    const outputCount = op.accumulator === "f32" ? 4 : 2;
    const inputCount = op.accumulator === "f32" ? 10 : 8;
    if (outputs.length !== outputCount || statement.inputs.length !== inputCount) {
      diagnostics.push(error("invalid-inline-asm-operands", `mma.m16n8k16 ${op.accumulator} inline PTX expects ${outputCount} output operand(s) and ${inputCount} input operands`, statement.span));
    }
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
): ExpressionInfo {
  const callName = expressionName(expression.callee);
  const namespaceCooperativeCall = cooperativeNamespaceCall(expression, scope);
  if (namespaceCooperativeCall) {
    return validateCooperativeNamespaceCall(expression, namespaceCooperativeCall, requiredFeatures, diagnostics, walkExpression, scope, options);
  }
  const cooperativeCall = cooperativeGroupCall(expression, scope);
  if (cooperativeCall) {
    return validateCooperativeGroupCall(expression, cooperativeCall, requiredFeatures, diagnostics, walkExpression, scope, options);
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
  if (isHostManagedRuntimeNoopCall(callName)) {
    validateRuntimeCall(expression, `${callName}() is host-managed in CUDA-lite WebGPU execution`, diagnostics, walkExpression, scope, options);
    return { kind: "scalar", valueType: "int" };
  }
  if (isCudaRuntimeCopyCall(callName)) {
    validateRuntimeCopyCall(expression, callName, diagnostics, walkExpression, scope, options);
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
  if (isPointerIdentityCall(callName)) {
    return validatePointerIdentityCall(expression, callName, scope, diagnostics, walkExpression);
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
  if (callName === "__halves2bfloat162") {
    for (const arg of expression.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return { kind: "vector", valueType: "bf162" };
  }
  if (isBfloat16ScalarArithmetic(callName)) {
    const infos = expression.args.map((arg) => walkExpression(arg, scope));
    for (const [index, info] of infos.entries()) validateScalarOperand(info, expression.args[index]!.span, diagnostics);
    if (infos.some((info) => info.valueType === "bf16")) return { kind: "scalar", valueType: "bf16" };
  }
  if (isHalf2Intrinsic(callName)) {
    requiredFeatures.add("shader-f16");
    for (const arg of expression.args) {
      const info = walkExpression(arg, scope);
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects half2 arguments`, arg.span));
      }
    }
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
  if (callName === "__low2float" || callName === "__high2float") {
    requiredFeatures.add("shader-f16");
    const arg = expression.args[0];
    if (arg) {
      const info = walkExpression(arg, scope);
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects half2 argument`, arg.span));
      } else if (info.kind === "vector" && info.valueType !== "half2") {
        diagnostics.push(error("unsupported-vector-argument", `${callName} expects half2 argument`, arg.span));
      }
    }
    return { kind: "scalar", valueType: "float" };
  }
  if (callName === "__float2half2_rn" || callName === "__floats2half2_rn") {
    requiredFeatures.add("shader-f16");
    for (const arg of expression.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return { kind: "vector", valueType: "half2" };
  }
  const vectorMath = validateVectorMinMaxCall(expression, callName, scope, diagnostics, requiredFeatures, walkExpression);
  if (vectorMath) return vectorMath;
  if (callName === "frexp" || callName === "frexpf") {
    validateFrexp(expression, scope, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "float" };
  }
  if (isVectorMathBuiltin(callName)) {
    const vectorMath = validateVectorMathBuiltin(expression, callName, diagnostics, walkExpression, scope);
    if (vectorMath) return vectorMath;
  }
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
    validateTextureRead(expression, callName, scope, diagnostics, walkExpression);
    if (requiresShaderF16(expression.templateValueType)) requiredFeatures.add("shader-f16");
    return expressionInfoForTextureRead(expression);
  }
  if (callName === "surf2Dread" || callName === "surf2DLayeredread" || callName === "surf3Dread") {
    validateSurf2DRead(expression, callName, scope, diagnostics, walkExpression);
    const returnForm = callName === "surf2DLayeredread" || callName === "surf3Dread" ? expression.args.length <= 4 : expression.args.length <= 3;
    return returnForm ? expressionInfoForTextureRead(expression) : { kind: "scalar", valueType: "voidptr" };
  }
  if (callName === "surf2Dwrite" || callName === "surf1Dwrite" || callName === "surf2DLayeredwrite" || callName === "surf3Dwrite") {
    validateSurf2DWrite(expression, callName, scope, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "float" };
  }
  if (callName === "sizeof" || callName === "alignof") {
    validateSizeof(expression, diagnostics);
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
  if (
    callName === "curand_uniform" ||
    callName === "curand_uniform_double" ||
    callName === "curand_normal" ||
    callName === "curand_normal_double"
  ) {
    validateCurandStateAddress(expression, callName, diagnostics, walkExpression, scope);
    return { kind: "scalar", valueType: "float" };
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
  if (info.kind !== "address" && info.kind !== "unknown") {
    diagnostics.push(error("unsupported-frexp-exponent", "frexp exponent must be an addressable local int", exponent.span));
    return;
  }
  if (info.valueType !== undefined && info.valueType !== "int") {
    diagnostics.push(error("unsupported-frexp-exponent", "frexp exponent must point to int storage", exponent.span));
  }
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

function isHalf2Intrinsic(name: string): boolean {
  return name === "__hadd2" ||
    name === "__hsub2" ||
    name === "__hmul2" ||
    name === "__hfma2" ||
    name === "__hmin2" ||
    name === "__hmax2";
}

function isBfloat16ScalarArithmetic(name: string): boolean {
  return name === "__hadd" ||
    name === "__hsub" ||
    name === "__hmul" ||
    name === "__hdiv" ||
    name === "__hfma" ||
    name === "__hmin" ||
    name === "__hmax";
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

function validateRuntimeCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  message: string,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
  options: CudaLiteAnalyzeOptions,
): void {
  const referenceRuntime = options.referenceCudaRuntime || options.referenceDynamicParallelism;
  diagnostics.push({
    ...error("unsupported-cuda-runtime", message, expression.span),
    severity: referenceRuntime ? "warning" : "error",
  });
  for (const arg of expression.args) walkExpression(arg, scope);
}

function validateRuntimeCopyCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
  options: CudaLiteAnalyzeOptions,
): void {
  const referenceRuntime = options.referenceCudaRuntime || options.referenceDynamicParallelism;
  diagnostics.push({
    ...error("unsupported-cuda-runtime", `${callName}() requires CUDA runtime copy orchestration`, expression.span),
    severity: referenceRuntime ? "warning" : "error",
  });
  const dst = expression.args[0];
  const src = expression.args[callName === "cudaMemcpyPeerAsync" ? 2 : 1];
  const byteCount = expression.args[callName === "cudaMemcpyPeerAsync" ? 4 : 2];
  if (dst) walkExpression(dst, scope);
  if (src) walkExpression(src, scope);
  if (byteCount) validateScalarOperand(walkExpression(byteCount, scope), byteCount.span, diagnostics);
  if ((callName === "cudaMemcpy" || callName === "cudaMemcpyAsync") && !supportedCudaMemcpyKind(expression.args[3])) {
    diagnostics.push(error(
      "unsupported-cuda-runtime-copy-kind",
      `${callName} supports cudaMemcpyDeviceToDevice/cudaMemcpyDefault only`,
      expression.args[3]?.span ?? expression.span,
    ));
  }
}

function isCudaRuntimeCopyCall(callName: string): boolean {
  return callName === "cudaMemcpy" || callName === "cudaMemcpyAsync" || callName === "cudaMemcpyPeerAsync";
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

function isHostManagedRuntimeNoopCall(callName: string): boolean {
  return callName === "cudaDeviceSynchronize" ||
    callName === "cudaStreamCreate" ||
    callName === "cudaStreamCreateWithFlags" ||
    callName === "cudaStreamDestroy" ||
    callName === "cudaStreamSynchronize" ||
    callName === "cudaEventCreate" ||
    callName === "cudaEventCreateWithFlags" ||
    callName === "cudaEventDestroy" ||
    callName === "cudaEventRecord" ||
    callName === "cudaEventSynchronize";
}

function supportedCudaMemcpyKind(expression: CudaLiteExpression | undefined): boolean {
  if (!expression) return false;
  if (expression.kind === "identifier") {
    return expression.name === "cudaMemcpyDeviceToDevice" || expression.name === "cudaMemcpyDefault";
  }
  if (expression.kind === "number") {
    return expression.value === 3 || expression.value === 4;
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
  return expected === "float2" && (info.kind === "complex" || (info.kind === "vector" && info.valueType === "float2"));
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
  options: CudaLiteAnalyzeOptions,
): ExpressionInfo {
  const { symbol, method } = call;
  if (symbol.groupKind === "grid" && method === "sync") {
    diagnostics.push({
      ...error("unsupported-cooperative-groups", "grid.sync() requires explicit runtime orchestration", expression.span),
      severity: options.referenceGridSync ? "warning" : "error",
    });
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
  requiredFeatures: Set<string>,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
  options: CudaLiteAnalyzeOptions,
): ExpressionInfo {
  const { symbol, method, groupArg } = call;
  if (method === "sync") {
    if (expression.args.length !== 1) diagnostics.push(error("invalid-call-arity", "cg::sync expects 1 argument", expression.span));
    if (symbol.groupKind === "grid") {
      diagnostics.push({
        ...error("unsupported-cooperative-groups", "cg::sync(grid) requires explicit runtime orchestration", expression.span),
        severity: options.referenceGridSync ? "warning" : "error",
      });
    }
    return { kind: "scalar" };
  }
  if (method === "reduce") {
    requiredFeatures.add("subgroups");
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
): void {
  if (!isSupportedTextureReadType(expression.templateValueType)) {
    diagnostics.push(error("unsupported-texture", `${callName} currently supports float/int/uint/uchar, half, float2/3/4, int2/3/4, uint2/3/4, and half2 reads`, expression.span));
  }
  const texture = expression.args[0];
  if (texture?.kind !== "identifier") {
    diagnostics.push(error("unsupported-texture", `${callName} first argument must be a texture reference`, expression.span));
  } else {
    const symbol = lookupSymbol(texture.name, scope, texture.span);
    if (symbol?.kind !== "texture" && symbol?.valueType !== "texture2d") {
      diagnostics.push(error("unsupported-texture", `${callName} target '${texture.name}' is not a texture reference`, texture.span));
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
): void {
  const hasZ = callName === "surf2DLayeredread" || callName === "surf3Dread";
  const returnForm = hasZ ? expression.args.length <= 4 : expression.args.length <= 3;
  const target = returnForm ? undefined : expression.args[0];
  if (target) {
    const lvalue = target.kind === "unary" && target.operator === "&" ? target.argument : target;
    validateLValueExpression(lvalue, scope, diagnostics, walkExpression);
  }
  const surface = returnForm ? expression.args[0] : expression.args[1];
  const surfaceName = surface ? rootIdentifier(surface) : undefined;
  if (!surfaceName) {
    diagnostics.push(error("unsupported-texture", returnForm ? "surf2Dread first argument must be a surface reference" : "surf2Dread second argument must be a surface reference", expression.span));
  } else {
    const symbol = lookupSymbol(surfaceName, scope, surface?.span ?? expression.span);
    if (symbol?.valueType !== "surface2d") {
      diagnostics.push(error("unsupported-texture", `surf2Dread target '${surfaceName}' is not a surface reference`, surface?.span ?? expression.span));
    }
  }
  const end = returnForm
    ? hasZ ? 4 : 3
    : hasZ ? 5 : 4;
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
): void {
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
    diagnostics.push(error("unsupported-surface", "surf2Dwrite second argument must be a surface object", expression.span));
  } else {
    const symbol = lookupSymbol(surfaceName, scope, surface?.span ?? expression.span);
    if (symbol?.valueType !== "surface2d") {
      diagnostics.push(error("unsupported-surface", `surf2Dwrite target '${surfaceName}' is not a cudaSurfaceObject_t parameter`, surface?.span ?? expression.span));
    }
  }
  if (xBytes) validateScalarOperand(walkExpression(xBytes, scope), xBytes.span, diagnostics);
  if (y) validateScalarOperand(walkExpression(y, scope), y.span, diagnostics);
  if (layer) validateScalarOperand(walkExpression(layer, scope), layer.span, diagnostics);
  if (z) validateScalarOperand(walkExpression(z, scope), z.span, diagnostics);
}

function validateSizeof(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  diagnostics: CudaLiteDiagnostic[],
): void {
  const target = expression.args[0];
  if (target?.kind !== "identifier" || sizeofCudaType(target.name) === undefined) {
    diagnostics.push(error("unsupported-sizeof", "sizeof/alignof only support CUDA-lite scalar types", target?.span ?? expression.span));
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
      if ((targetType === "float" || targetType === "double") && isSupportedFloatAtomic(callName)) {
        atomicShared.add(storageSymbol.name);
      } else if (targetType === "half" || targetType === "bool" || targetType === "complex64" || targetType === "float" || targetType === "double") {
        diagnostics.push(error("unsupported-atomic-target", "shared atomics support int/uint targets and CAS-backed float add/sub/min/max/exch in CUDA-lite", targetExpression.span));
      } else {
        atomicShared.add(storageSymbol.name);
      }
    } else if (storageSymbol?.kind === "device-global") {
      if ((targetType === "float" || targetType === "double") && isSupportedFloatAtomic(callName)) {
        atomicDeviceGlobals.add(storageSymbol.name);
      } else if (targetType === "half" || targetType === "bool" || targetType === "complex64" || targetType === "float" || targetType === "double") {
        diagnostics.push(error("unsupported-atomic-target", "device global atomics support int/uint targets and CAS-backed float add/sub/min/max/exch in CUDA-lite", targetExpression.span));
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
        diagnostics.push(error("unsupported-atomic-target", `${callName ?? "atomic operation"} through device pointer supports int/uint read-modify-write atomics including inc/dec and CAS-backed float add/sub/min/max/exch in CUDA-lite`, targetExpression.span));
      }
    } else if (!param?.pointer) {
      diagnostics.push(error("unsupported-atomic-target", `${callName ?? "atomic operation"} target must resolve to storage or shared memory`, targetExpression.span));
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
      callName === "atomicExch_system"
    )) {
      atomicParams.add(param.name);
      if (param.constant) {
        diagnostics.push(error("const-pointer-write", `cannot ${callName} through const pointer '${param.name}'`, expression.span));
      }
    } else if (targetType === "float" || targetType === "double" || targetType === "half" || targetType === "complex64") {
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
  const functionsByName = new Map<string, CudaLiteDeviceFunction[]>();
  for (const fn of ast.functions) {
    const overloads = functionsByName.get(fn.name) ?? [];
    overloads.push(fn);
    functionsByName.set(fn.name, overloads);
  }

  const sharedNames = new Set(collectSharedDeclarationsFromBodies([kernel.body, ...ast.functions.map((fn) => fn.body)], options).map((shared) => shared.name));
  const deviceGlobalNames = new Set(ast.deviceGlobals.map((global) => global.name));
  const kernelPointerParams = new Set(kernel.params.filter((param) => param.pointer && !param.constant).map((param) => param.name));
  const functionAtomicParams = new Map<string, Set<string>>();
  for (const fn of ast.functions) functionAtomicParams.set(fn.name, new Set());

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
    for (const fn of ast.functions) {
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
        if (statement.pointer) {
          aliases.set(statement.name, pointerRootsFromExpression(statement.init, aliases));
        }
        if (statement.init) visitExpression(statement.init);
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
    callName === "atomicExch_system";
}

function isSupportedDevicePointerAtomic(
  callName: string | undefined,
  targetType: CudaLiteScalarType,
): boolean {
  if (targetType !== "float" && targetType !== "double" && targetType !== "int" && targetType !== "uint") return false;
  if (callName === "atomicAdd" || callName === "atomicAdd_system") return true;
  if (callName === "atomicSub" || callName === "atomicSub_system") return true;
  if (callName === "atomicMin" || callName === "atomicMin_system") return true;
  if (callName === "atomicMax" || callName === "atomicMax_system" || callName === "atomicMaxFloat") return true;
  if (callName === "atomicExch" || callName === "atomicExch_system") return true;
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
  return callName === "__shfl_sync" ||
    callName === "__shfl_down_sync" ||
    callName === "__shfl_up_sync" ||
    callName === "__shfl_xor_sync";
}

function isVoteBuiltin(callName: string): boolean {
  return callName === "__any_sync" || callName === "__all_sync" || callName === "__ballot_sync";
}

function isMaskedWarpReductionBuiltin(callName: string): boolean {
  return callName === "__reduce_add_sync";
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
        const field = cudaVectorFieldIndex(object.valueType, expression.property);
        if (field === undefined) {
          diagnostics.push(error("unsupported-vector-member", `unsupported ${object.valueType} member '${expression.property}'`, expression.span));
        }
        return { kind: "scalar", valueType: cudaVectorScalarType(object.valueType) };
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
      if ((expression.operator === "+" || expression.operator === "-") && left.kind === "array") {
        validateScalarOperand(right, expression.right.span, diagnostics);
        return { kind: "pointer", valueType: left.valueType, symbol: left.symbol };
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
          if (right.kind !== "pointer" && right.kind !== "pool-pointer" && right.kind !== "address" && right.kind !== "unknown") {
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
      if (cudaVectorFieldIndex(info.valueType, expression.property) === undefined) {
        diagnostics.push(error("invalid-assignment-target", `vector assignment target must be one of .${CUDA_VECTOR_TYPES.get(info.valueType)!.fields.join("/.")}`, expression.span));
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
): void {
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
      visitNoSideEffect(cursor.right);
      return;
    }
    cursor = cursor.right;
  }
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

function hasContinueTargetingDoWhile(statements: readonly CudaLiteStatement[]): boolean {
  for (const statement of statements) {
    if (statement.kind === "continue") return true;
    if (statement.kind === "block" && hasContinueTargetingDoWhile(statement.body)) return true;
    if (statement.kind === "if") {
      if (hasContinueTargetingDoWhile(statement.consequent)) return true;
      if (statement.alternate && hasContinueTargetingDoWhile(statement.alternate)) return true;
    }
    if (statement.kind === "do-while" && hasContinueTargetingDoWhile(statement.body)) return true;
    // A continue inside these bodies targets the nested loop, not the enclosing do-while.
    if (statement.kind === "for" || statement.kind === "while") continue;
  }
  return false;
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
  ): boolean => {
    let barrierLater = initialBarrierLater;
    let containsBarrier = false;
    for (let index = body.length - 1; index >= 0; index--) {
      const statement = body[index]!;
      const info = visitStatement(statement, divergentDepth, barrierLater);
      containsBarrier = info.containsBarrier || containsBarrier;
      barrierLater = barrierLater || info.containsBarrier;
    }
    return containsBarrier;
  };

  const visitStatement = (
    statement: CudaLiteStatement,
    divergentDepth: number,
    barrierLater: boolean,
  ): { readonly containsBarrier: boolean } => {
    switch (statement.kind) {
      case "block":
        return { containsBarrier: visitBlock(statement.body, divergentDepth, barrierLater) };
      case "expr":
        return { containsBarrier: isBarrierCall(statement.expression) };
      case "return":
        if (divergentDepth > 0 && barrierLater) {
          diagnostics.push(warning(
            "divergent-return-before-barrier",
            "thread-dependent return before a later barrier would make WGSL barrier control flow non-uniform",
            statement.span,
          ));
        }
        return { containsBarrier: false };
      case "if": {
        const nestedDivergentDepth = divergentDepth + (expressionMayBeNonUniformBeforeBarrier(statement.condition, uniformity) ? 1 : 0);
        const consequentHasBarrier = visitBlock(statement.consequent, nestedDivergentDepth, barrierLater);
        const alternateHasBarrier = statement.alternate ? visitBlock(statement.alternate, nestedDivergentDepth, barrierLater) : false;
        return { containsBarrier: consequentHasBarrier || alternateHasBarrier };
      }
      case "for": {
        const nestedDivergentDepth = divergentDepth + (statement.condition && expressionMayBeNonUniformBeforeBarrier(statement.condition, uniformity) ? 1 : 0);
        return { containsBarrier: visitBlock(statement.body, nestedDivergentDepth, barrierLater) };
      }
      case "while": {
        const nestedDivergentDepth = divergentDepth + (expressionMayBeNonUniformBeforeBarrier(statement.condition, uniformity) ? 1 : 0);
        return { containsBarrier: visitBlock(statement.body, nestedDivergentDepth, barrierLater) };
      }
      case "do-while": {
        const nestedDivergentDepth = divergentDepth + (expressionMayBeNonUniformBeforeBarrier(statement.condition, uniformity) ? 1 : 0);
        return { containsBarrier: visitBlock(statement.body, nestedDivergentDepth, barrierLater) };
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
