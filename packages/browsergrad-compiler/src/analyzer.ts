import {
  CudaLiteCompilerError,
  type CudaLiteAnalysis,
  type CudaLiteAnalyzeOptions,
  type CudaLiteAssignmentExpression,
  type CudaLiteCooperativeGroupKind,
  type CudaLiteDeviceFunction,
  type CudaLiteDiagnostic,
  type CudaLiteExpression,
  type CudaLiteGlobalConstant,
  type CudaLiteKernel,
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
import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";
import { sizeofCudaType } from "./type_layout.js";
import {
  CUDA_VECTOR_TYPES,
  CUDA_VECTOR_CONSTRUCTORS,
  cudaVectorConstructorType,
  cudaVectorFieldIndex,
  cudaVectorScalarType,
  isCudaVectorType,
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
  ["warpReduceSum", [1, 1]],
  ["warpReduceMax", [1, 1]],
  ["warpReduceMin", [1, 1]],
  ["warp_reduce_sum", [1, 1]],
  ["warp_reduce_max", [1, 1]],
  ["warp_reduce_min", [1, 1]],
  ["warp_reduce_sum_f32", [1, 1]],
  ["warp_reduce_max_f32", [1, 1]],
  ["warp_reduce_sum_f16", [1, 1]],
  ["warp_reduce_sum_f16_f16", [1, 1]],
  ["warp_reduce_sum_f16_f32", [1, 1]],
  ["warp_reduce_sum_i8_i32", [1, 1]],
  ["warp_reduce_sum_i32_i32", [1, 1]],
  ["blockReduce", [1, 3]],
  ["min", [2, 2]],
  ["max", [2, 2]],
  ["frexp", [2, 2]],
  ["frexpf", [2, 2]],
  ["div_ceil", [2, 2]],
  ["fill_1D_regs", [2, 2]],
  ["fill_2D_regs", [2, 2]],
  ["fill_3D_regs", [2, 2]],
  ["bg_subgroup_add", [1, 1]],
  ["atomicAdd", [2, 2]],
  ["atomicAdd_system", [2, 2]],
  ["atomicSub", [2, 2]],
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
  ["surf2Dwrite", [4, 5]],
  ["surf1Dwrite", [3, 4]],
  ["surf2DLayeredwrite", [5, 6]],
  ["sizeof", [1, 1]],
  ["alignof", [1, 1]],
  ["vec_at", [2, 2]],
  ["deviceAllocate", [2, 4]],
  ["streamOrderedAllocate", [2, 4]],
  ["curand_init", [4, 4]],
  ["curand_uniform", [1, 1]],
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
  readonly kind: "param" | "local" | "shared" | "constant" | "texture" | "cooperative-group" | "device-function" | "builtin-vector" | "builtin-call";
  readonly valueType?: ValueType;
  readonly returnType?: CudaLiteScalarType;
  readonly params?: readonly CudaLiteParam[];
  readonly body?: readonly CudaLiteStatement[];
  readonly groupKind?: CudaLiteCooperativeGroupKind;
  readonly tileSize?: number;
  readonly pointer?: boolean;
  readonly constant?: boolean;
  readonly pointerRoot?: string;
  readonly dimensions?: readonly number[];
  readonly span: SourceSpan;
}

interface Scope {
  readonly symbols: Map<string, SymbolInfo>;
  readonly parent?: Scope;
}

interface ExpressionInfo {
  readonly kind: "scalar" | "complex" | "pool-pointer" | "pointer" | "array" | "texture" | "surface" | "vector" | "function" | "address" | "string" | "unknown";
  readonly valueType?: ValueType | undefined;
  readonly dimensions?: readonly number[] | undefined;
  readonly symbol?: SymbolInfo | undefined;
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
  const atomicDevicePointerTypes = new Set<CudaLiteScalarType>();
  const params = new Map(kernel.params.map((param) => [param.name, param]));
  const declaredNames = new Set<string>();
  const rootScope = createScope();

  for (const constant of ast.constants) {
    declareConstant(constant, rootScope, declaredNames, requiredFeatures, diagnostics, options);
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
      span: statement.span,
    });
    validateF64Type(statement.valueType, statement.span, diagnostics, options);
  };

  const walkExpression = (expression: CudaLiteExpression, scope: Scope): ExpressionInfo => {
    if (expression.kind === "call") {
      return validateCallExpression(expression, scope, params, atomicParams, atomicShared, atomicDevicePointerTypes, requiredFeatures, diagnostics, walkExpression, options);
    }
    return validateNonCallExpression(expression, scope, diagnostics, walkExpression, requiredFeatures);
  };

  for (const constant of ast.constants) {
    validateGlobalConstantInitializer(constant, rootScope, diagnostics, walkExpression);
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
          if (requiresShaderF16(statement.valueType)) requiredFeatures.add("shader-f16");
          if (statement.pointer && !isSupportedLocalPointer(statement, scope)) {
            diagnostics.push(error("unsupported-local-pointer", "local pointer declarations are not supported in CUDA-lite yet", statement.span));
          }
          if (statement.storage === "local" && statement.dimensions.length > 0 && statement.init) {
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
          if (statement.init && statement.dimensions.length === 0) {
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
          names.add(statement.name);
          scope.symbols.set(statement.name, {
            name: statement.name,
            kind: "cooperative-group",
            groupKind: statement.groupKind,
            ...(statement.tileSize === undefined ? {} : { tileSize: statement.tileSize }),
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
            if (requiresShaderF16(statement.init.valueType)) requiredFeatures.add("shader-f16");
            if (statement.init.pointer && !isSupportedLocalPointer(statement.init, loopScope)) {
              diagnostics.push(error("unsupported-local-pointer", "local pointer declarations are not supported in CUDA-lite yet", statement.init.span));
            }
            if (statement.init.dimensions.length > 0 && statement.init.init) {
              validateArrayInitializer(statement.init, loopScope, diagnostics, walkExpression);
            }
            if (statement.init.init && statement.init.dimensions.length === 0) {
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
  }

  walkStatements(kernel.body, rootScope, 0, 0, 0, declaredNames);
  const sharedDeclarations = collectSharedDeclarationsFromBodies([kernel.body, ...ast.functions.map((fn) => fn.body)], options);
  for (const type of atomicDevicePointerTypes) {
    for (const param of kernel.params) {
      if (param.pointer && !param.constant && isDevicePointerAtomicMemoryCompatible(type, param.valueType)) {
        atomicParams.add(param.name);
      }
    }
    for (const shared of sharedDeclarations) {
      if (isDevicePointerAtomicMemoryCompatible(type, shared.valueType)) {
        atomicShared.add(shared.name);
      }
    }
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
    textures: ast.textures,
    functions: ast.functions,
    diagnostics,
    requiredFeatures: [...requiredFeatures].sort(),
    atomicParams: [...atomicParams].sort(),
    atomicShared: [...atomicShared].sort(),
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
    textures: analysis.textures,
    functions: analysis.functions,
    body: analysis.kernel.body,
    sharedDeclarations: collectSharedDeclarationsFromBodies([analysis.kernel.body, ...analysis.functions.map((fn) => fn.body)], options),
    requiredFeatures: analysis.requiredFeatures,
    atomicParams: analysis.atomicParams,
    atomicShared: analysis.atomicShared,
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

function validateGlobalConstantInitializer(
  constant: CudaLiteGlobalConstant,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  if (!constant.init) return;
  const values = flattenInitializerExpressions(constant.init);
  if (constant.dimensions.length === 0 && values.length > 1) {
    diagnostics.push(error("invalid-constant-initializer", `constant '${constant.name}' scalar initializer must have one value`, constant.init.span));
  }
  const expected = constant.dimensions.reduce((product, dimension) => product * dimension, 1);
  if (constant.dimensions.length > 0 && values.length > expected) {
    diagnostics.push(error("invalid-constant-initializer", `constant '${constant.name}' initializer has more than ${expected} values`, constant.init.span));
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
  if (declaredNames.has(fn.name)) {
    diagnostics.push(error("duplicate-symbol", `duplicate CUDA-lite symbol '${fn.name}'`, fn.span));
  }
  validateDeclaredSymbolName(fn.name, fn.span, diagnostics);
  declaredNames.add(fn.name);
  rootScope.symbols.set(fn.name, {
    name: fn.name,
    kind: "device-function",
    returnType: fn.returnType,
    params: fn.params,
    body: fn.body,
    span: fn.span,
  });
  if (requiresShaderF16(fn.returnType)) requiredFeatures.add("shader-f16");
  validateF64Type(fn.returnType, fn.span, diagnostics, options);
  for (const param of fn.params) {
    if (requiresShaderF16(param.valueType)) requiredFeatures.add("shader-f16");
    validateF64Type(param.valueType, param.span, diagnostics, options);
  }
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
  if (isSupportedSharedPointerAlias(statement, scope)) return true;
  if (isSupportedStoragePointerInitializer(statement, scope)) return true;
  if (!statement.pointer || statement.storage !== "local") return false;
  return isSupportedPoolPointerInitializer(statement.init, scope);
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
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    const pointer = expression.args[0];
    if (pointer) validatePointerInitializerExpression(pointer, scope, diagnostics, walkExpression);
    for (const arg of expression.args.slice(1)) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return;
  }
  if (expression.kind === "unary" && expression.operator === "&") {
    validateLValueExpression(expression.argument, scope, diagnostics, walkExpression);
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
  if (symbol?.kind === "shared") return symbol.valueType;
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
  if (init.kind === "identifier") {
    if (init.name === "nullptr") return true;
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
  const fma = isInlineAsmFma(statement.template);
  const laneId = isInlineAsmLaneId(statement.template);
  const laneMaskLt = isInlineAsmLaneMaskLt(statement.template);
  const bfindU32 = isInlineAsmBfindU32(statement.template);
  const sadAdd = isInlineAsmU8x4SadAdd(statement.template);
  if (!fma && !laneId && !laneMaskLt && !bfindU32 && !sadAdd) {
    diagnostics.push(error("unsupported-inline-asm", "only fma.rn.f32, laneid, lanemask_lt, bfind.u32, and vabsdiff4.u32.u32.u32.add inline PTX are supported in CUDA-lite v0", statement.span));
  }
  const outputInfo = statement.output === undefined ? undefined : walkExpression(statement.output, scope);
  if (statement.output !== undefined) {
    validateLValueExpression(statement.output, scope, diagnostics, walkExpression);
    validateScalarOperand(outputInfo!, statement.output.span, diagnostics);
  }
  if (fma && (statement.output === undefined || statement.inputs.length !== 2)) {
    diagnostics.push(error("invalid-inline-asm-operands", "fma.rn.f32 inline PTX expects exactly two input operands", statement.span));
  }
  if (laneId && (statement.output === undefined || statement.inputs.length !== 0)) {
    diagnostics.push(error("invalid-inline-asm-operands", "laneid inline PTX expects no input operands", statement.span));
  }
  if (laneId && outputInfo?.valueType !== undefined && outputInfo.valueType !== "uint" && outputInfo.valueType !== "int") {
    diagnostics.push(error("invalid-inline-asm-operands", "laneid inline PTX writes an integer output operand", statement.output?.span ?? statement.span));
  }
  if (laneMaskLt && (statement.output === undefined || statement.inputs.length !== 0)) {
    diagnostics.push(error("invalid-inline-asm-operands", "lanemask_lt inline PTX expects no input operands", statement.span));
  }
  if (laneMaskLt && outputInfo?.valueType !== undefined && outputInfo.valueType !== "uint" && outputInfo.valueType !== "int") {
    diagnostics.push(error("invalid-inline-asm-operands", "lanemask_lt inline PTX writes an integer output operand", statement.output?.span ?? statement.span));
  }
  if (bfindU32 && (statement.output === undefined || statement.inputs.length !== 1)) {
    diagnostics.push(error("invalid-inline-asm-operands", "bfind.u32 inline PTX expects one input operand", statement.span));
  }
  if (bfindU32 && outputInfo?.valueType !== undefined && outputInfo.valueType !== "uint") {
    diagnostics.push(error("invalid-inline-asm-operands", "bfind.u32 inline PTX writes a uint output operand", statement.output?.span ?? statement.span));
  }
  if (sadAdd && (statement.output === undefined || statement.inputs.length !== 3)) {
    diagnostics.push(error("invalid-inline-asm-operands", "vabsdiff4.u32.u32.u32.add inline PTX expects three input operands", statement.span));
  }
  if (sadAdd && outputInfo?.valueType !== undefined && outputInfo.valueType !== "uint" && outputInfo.valueType !== "int") {
    diagnostics.push(error("invalid-inline-asm-operands", "vabsdiff4.u32.u32.u32.add inline PTX writes an integer output operand", statement.output?.span ?? statement.span));
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
  atomicDevicePointerTypes: Set<CudaLiteScalarType>,
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

  if (callName === "bg_subgroup_add") requiredFeatures.add("subgroups");
  if (callName === "__syncthreads" || callName === "__syncwarp") {
    diagnostics.push(error("barrier-expression", `${callName}() must be used as a standalone statement`, expression.span));
  }
  if (callName === "printf") {
    for (const arg of expression.args.slice(1)) {
      const info = walkExpression(arg, scope);
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
  if (isAtomicBuiltin(callName)) {
    validateAtomicBuiltin(expression, scope, params, atomicParams, atomicShared, atomicDevicePointerTypes, diagnostics, walkExpression);
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
    if (target) {
      const info = walkExpression(target, scope);
      if (info.kind !== "pointer" && info.kind !== "pool-pointer" && info.kind !== "address" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-cache-hint-address", `${callName} expects a pointer expression`, target.span));
      }
    }
    if (value) validateScalarOperand(walkExpression(value, scope), value.span, diagnostics);
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
    const targetScalar = cudaVectorScalarType(vectorConstructor);
    for (const arg of expression.args) {
      const info = walkExpression(arg, scope);
      if (
        expression.args.length === 1 &&
        info.kind === "vector" &&
        isCudaVectorType(info.valueType) &&
        cudaVectorScalarType(info.valueType) === targetScalar
      ) {
        continue;
      }
      validateScalarOperand(info, arg.span, diagnostics);
    }
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
    const arg = expression.args[0];
    if (!arg) return { kind: "unknown" };
    const info = walkExpression(arg, scope);
    validateScalarOperand(info, arg.span, diagnostics);
    return { kind: "scalar", valueType: warpReductionReturnType(callName, info.valueType) };
  }
  if (isTextureReadCall(callName)) {
    validateTextureRead(expression, callName, scope, diagnostics, walkExpression);
    if (requiresShaderF16(expression.templateValueType)) requiredFeatures.add("shader-f16");
    return expressionInfoForTextureRead(expression);
  }
  if (callName === "surf2Dread") {
    validateSurf2DRead(expression, scope, diagnostics, walkExpression);
    return expression.args.length <= 3 ? expressionInfoForTextureRead(expression) : { kind: "scalar", valueType: "voidptr" };
  }
  if (callName === "surf2Dwrite" || callName === "surf1Dwrite" || callName === "surf2DLayeredwrite") {
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
  if (isVectorMathBuiltin(callName)) {
    return validateVectorMathBuiltin(expression, callName, diagnostics, walkExpression, scope);
  }
  if (callName === "deviceAllocate" || callName === "streamOrderedAllocate") {
    validatePoolAllocate(expression, scope, atomicParams, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "voidptr" };
  }
  if (callName === "curand_init") {
    validateCurandInit(expression, diagnostics, walkExpression, scope);
    return { kind: "scalar", valueType: "uint" };
  }
  if (callName === "curand_uniform") {
    validateCurandUniform(expression, diagnostics, walkExpression, scope);
    return { kind: "scalar", valueType: "float" };
  }

  for (const arg of expression.args) {
    const info = walkExpression(arg, scope);
    validateScalarOperand(info, arg.span, diagnostics);
  }
  return { kind: "scalar" };
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

function isVectorMathBuiltin(name: string): boolean {
  return name === "dot" || name === "length" || name === "normalize" || name === "cross";
}

function validateVectorMathBuiltin(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  callName: string,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
): ExpressionInfo {
  const infos = expression.args.map((arg) => walkExpression(arg, scope));
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

function validateDeviceFunctionCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  symbol: SymbolInfo,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
): ExpressionInfo {
  const fnParams = symbol.params ?? [];
  if (expression.args.length !== fnParams.length) {
    diagnostics.push(error(
      "invalid-call-arity",
      `${symbol.name} expects ${fnParams.length} argument${fnParams.length === 1 ? "" : "s"}`,
      expression.span,
    ));
  }
  for (const [index, arg] of expression.args.entries()) {
    const param = fnParams[index];
    if (param?.cooperativeGroupKind !== undefined) {
      validateCooperativeGroupArgument(arg, param, scope, diagnostics);
      if (param.cooperativeGroupKind === "thread" && deviceFunctionUsesGroupReduce(symbol, param.name)) {
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
      if (info.kind !== "vector" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-vector-argument", `device parameter '${param.name}' expects ${param.valueType}`, arg.span));
      }
      continue;
    }
    validateScalarOperand(info, arg.span, diagnostics);
  }
  if (symbol.returnType === undefined || symbol.returnType === "void") return { kind: "unknown" };
  return isCudaVectorType(symbol.returnType)
    ? { kind: "vector", valueType: symbol.returnType }
    : { kind: "scalar", valueType: symbol.returnType };
}

function deviceFunctionUsesGroupReduce(symbol: SymbolInfo, paramName: string): boolean {
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
  const info = walkExpression(arg, scope);
  const root = rootIdentifier(arg);
  const rootSymbol = root ? lookupSymbol(root, scope, arg.span) : undefined;
  const sharedArrayDecay = rootSymbol?.kind === "shared" &&
    rootSymbol.dimensions !== undefined &&
    info.kind === "array";
  const constantArrayDecay = rootSymbol?.kind === "constant" &&
    rootSymbol.dimensions !== undefined &&
    info.kind === "array";
  if (info.kind !== "pointer" && info.kind !== "address" && info.kind !== "unknown" && !sharedArrayDecay && !constantArrayDecay) {
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
    diagnostics.push(error("unsupported-texture", `${callName} currently supports float/int/uint, half, float2/3/4, int2/3/4, uint2/3/4, and half2 reads`, expression.span));
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
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const returnForm = expression.args.length <= 3;
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
  for (const coord of returnForm ? expression.args.slice(1, 3) : expression.args.slice(2, 4)) {
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

function validateCurandUniform(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  scope: Scope,
): void {
  const state = expression.args[0];
  if (!state) return;
  const info = walkExpression(state, scope);
  if (info.kind !== "address") {
    diagnostics.push(error("curand-state-address", "curand_uniform expects a state address", state.span));
  }
}

function validateAtomicBuiltin(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  params: ReadonlyMap<string, CudaLiteParam>,
  atomicParams: Set<string>,
  atomicShared: Set<string>,
  atomicDevicePointerTypes: Set<CudaLiteScalarType>,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const target = expression.args[0];
  const callName = expressionName(expression.callee);
  const targetExpression = atomicTargetExpression(target);
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
    const targetType = symbol?.valueType ?? storageSymbol?.valueType;
    if (storageSymbol?.kind === "shared") {
      if (targetType === "float" && isSupportedFloatAtomic(callName)) {
        atomicShared.add(storageSymbol.name);
      } else if (targetType === "half" || targetType === "bool" || targetType === "complex64" || targetType === "float") {
        diagnostics.push(error("unsupported-atomic-target", "shared atomics support int/uint targets and CAS-backed float add/sub/min/max/exch in CUDA-lite", targetExpression.span));
      } else {
        atomicShared.add(storageSymbol.name);
      }
    } else if (!param?.pointer && symbol?.kind === "local" && symbol.pointer) {
      if (symbol.constant) {
        diagnostics.push(error("const-pointer-write", `cannot ${callName ?? "atomic operation"} through const pointer '${symbol.name}'`, expression.span));
      }
      if (targetType && isSupportedDevicePointerAtomic(callName, targetType)) {
        atomicDevicePointerTypes.add(targetType);
      } else {
        diagnostics.push(error("unsupported-atomic-target", `${callName ?? "atomic operation"} through device pointer supports int/uint add and CAS-backed float add in CUDA-lite`, targetExpression.span));
      }
    } else if (!param?.pointer) {
      diagnostics.push(error("unsupported-atomic-target", `${callName ?? "atomic operation"} target must resolve to storage or shared memory`, targetExpression.span));
    } else if (targetType === "float" && (
      callName === "atomicAdd" ||
      callName === "atomicAdd_system" ||
      callName === "atomicSub" ||
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
    } else if (targetType === "float" || targetType === "half" || targetType === "complex64") {
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

function isAtomicBuiltin(callName: string): boolean {
  return callName === "atomicAdd" ||
    callName === "atomicAdd_system" ||
    callName === "atomicSub" ||
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
  return (callName === "atomicAdd" || callName === "atomicAdd_system") &&
    (targetType === "float" || targetType === "int" || targetType === "uint");
}

function isDevicePointerAtomicMemoryCompatible(
  pointerType: CudaLiteScalarType,
  memoryType: CudaLiteScalarType,
): boolean {
  return pointerType === memoryType && (memoryType === "float" || memoryType === "int" || memoryType === "uint");
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
): CudaLiteExpression | undefined {
  if (!target) return undefined;
  if (target.kind === "cast" && target.pointer) return atomicTargetExpression(target.expression);
  if (target.kind === "unary" && target.operator === "&") return target.argument;
  if (target.kind === "identifier") return target;
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
        if (info.kind !== "scalar" && info.kind !== "pointer" && info.kind !== "pool-pointer" && info.kind !== "address" && info.kind !== "unknown") {
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
        if (isCudaVectorType(target.valueType)) {
          return { kind: "vector", valueType: target.valueType, symbol: target.symbol };
        }
        return target.valueType === "complex64"
          ? { kind: "complex", valueType: target.valueType, symbol: target.symbol }
          : { kind: "scalar", valueType: target.valueType, symbol: target.symbol };
      }
      diagnostics.push(error("unsupported-index-target", "only pointer parameters, local arrays, fixed __shared__ arrays, and constants can be indexed", expression.span));
      return { kind: "unknown" };
    }
    case "unary": {
      if (expression.operator === "&") {
        validateLValueExpression(expression.argument, scope, diagnostics, walkExpression);
        return { kind: "address" };
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
      if ((expression.operator === "+" || expression.operator === "-") && (left.kind === "pointer" || left.kind === "pool-pointer")) {
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
        if (right.kind !== "complex" && right.kind !== "unknown") {
          diagnostics.push(error("unsupported-scalar-expression", "complex assignment expects a complex value", expression.right.span));
        }
      } else if (left.kind === "vector") {
        if (expression.operator === "=" && right.kind !== "vector" && right.kind !== "unknown") {
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
      diagnostics.push(error("unknown-symbol", `unknown CUDA-lite symbol '${expression.name}'`, expression.span));
      return;
    }
    if (symbol.kind === "local" || symbol.kind === "shared") return;
    if (symbol.kind === "param" && !symbol.pointer) {
      diagnostics.push(error("parameter-assignment", `cannot assign to scalar parameter '${expression.name}'`, expression.span));
      return;
    }
    if (symbol.kind === "param" && symbol.pointer && (operator === "=" || isPointerRebaseOperator(operator))) return;
    diagnostics.push(error("invalid-assignment-target", "assignment target must be a local variable, pointer element, or shared element", expression.span));
    return;
  }
  if (expression.kind === "index") {
    const info = walkExpression(expression, scope);
    const root = rootIdentifier(expression);
    const symbol = root ? lookupSymbol(root, scope, expression.span) : undefined;
    if (symbol?.kind === "constant") {
      diagnostics.push(error("const-pointer-write", `cannot write to constant memory '${root}'`, expression.span));
      return;
    }
    if (symbol?.pointer && symbol.constant) {
      diagnostics.push(error("const-pointer-write", `cannot write through const pointer '${root}'`, expression.span));
      return;
    }
    if (info.kind !== "scalar" && info.kind !== "complex" && info.kind !== "vector") {
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
    if (symbol?.pointer && symbol.constant) {
      diagnostics.push(error("const-pointer-write", `cannot write through const pointer '${root}'`, expression.span));
    }
    return;
  }
  diagnostics.push(error("invalid-assignment-target", "assignment target must be a local variable, pointer element, or shared element", expression.span));
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
    diagnostics.push(error("unknown-symbol", `unknown CUDA-lite symbol '${name}'`, span));
    return { kind: "unknown" };
  }
  if (symbol.kind === "builtin-vector") return { kind: "vector", symbol };
  if (symbol.kind === "builtin-call") return { kind: "function", symbol };
  if (symbol.kind === "device-function") return { kind: "function", symbol };
  if (symbol.kind === "cooperative-group") return { kind: "unknown", symbol };
  if (symbol.kind === "texture") return { kind: "texture", valueType: symbol.valueType, symbol };
  if (symbol.valueType === "texture2d") return { kind: "texture", valueType: symbol.valueType, symbol };
  if (symbol.valueType === "surface2d") return { kind: "surface", valueType: symbol.valueType, symbol };
  if (symbol.kind === "local" && symbol.dimensions && symbol.dimensions.length > 0) {
    return {
      kind: "array",
      valueType: symbol.valueType,
      dimensions: symbol.dimensions,
      symbol,
    };
  }
  if (symbol.kind === "shared" || symbol.kind === "constant") {
    if (symbol.valueType === "complex64" && (!symbol.dimensions || symbol.dimensions.length === 0)) {
      return { kind: "complex", valueType: symbol.valueType, symbol };
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

function isInlineAsmFma(template: string): boolean {
  return /\bfma\.rn\.f32\b/u.test(template);
}

function isInlineAsmLaneId(template: string): boolean {
  return /\bmov\.u32\b/u.test(template) && /%%laneid\b/u.test(template);
}

function isInlineAsmLaneMaskLt(template: string): boolean {
  return /\bmov\.u32\b/u.test(template) && /%%lanemask_lt\b/u.test(template);
}

function isInlineAsmBfindU32(template: string): boolean {
  return /\bbfind\.u32\b/u.test(template);
}

function isInlineAsmU8x4SadAdd(template: string): boolean {
  return /\bvabsdiff4\.u32\.u32\.u32\.add\b/u.test(template);
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
      if (item.kind === "for") walk(item.body);
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
  if (statement.storage !== "shared" || !statement.dynamicShared || statement.dimensions.length > 0) return undefined;
  const elements = options.dynamicSharedMemory?.[statement.name];
  if (elements === undefined) return undefined;
  return [positiveInteger(elements, `dynamicSharedMemory.${statement.name}`)];
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
