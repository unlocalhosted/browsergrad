import {
  CudaLiteCompilerError,
  type CudaLiteAnalysis,
  type CudaLiteAnalyzeOptions,
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
  type CudaLiteTexture2D,
  type CudaLiteVarDecl,
  type KernelIrModule,
  type SourceSpan,
} from "./types.js";
import { collectKernelLaunchCallees } from "./ast_queries.js";
import { CUDA_INTRINSICS, CUDA_INTRINSICS_BY_NAME } from "./intrinsics.js";
import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";
import {
  CUDA_VECTOR_TYPES,
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
  ["__shfl_down_sync", [3, 4]],
  ["__shfl_up_sync", [3, 4]],
  ["__shfl_xor_sync", [3, 4]],
  ["min", [2, 2]],
  ["max", [2, 2]],
  ["bg_subgroup_add", [1, 1]],
  ["atomicAdd", [2, 2]],
  ["atomicSub", [2, 2]],
  ["atomicMin", [2, 2]],
  ["atomicMax", [2, 2]],
  ["atomicMaxFloat", [2, 2]],
  ["atomicExch", [2, 2]],
  ["atomicCAS", [3, 3]],
  ["tex2D", [3, 3]],
  ["surf2Dwrite", [4, 4]],
  ["sizeof", [1, 1]],
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
  ["__ldcs", [1, 1]],
  ["__stcs", [2, 2]],
  ["printf", [1, Number.POSITIVE_INFINITY]],
  ...[...CUDA_VECTOR_TYPES].map(([type, info]) => [`make_${type}`, [info.lanes, info.lanes]] as const),
]);
const WGSL_RESERVED_WORDS = new Set([
  "alias",
  "array",
  "atomic",
  "bitcast",
  "bool",
  "break",
  "case",
  "const",
  "continue",
  "default",
  "discard",
  "else",
  "enable",
  "false",
  "fn",
  "for",
  "f16",
  "f32",
  "i32",
  "if",
  "let",
  "loop",
  "override",
  "return",
  "struct",
  "switch",
  "true",
  "u32",
  "var",
  "while",
]);

type ValueType = Exclude<CudaLiteScalarType, "void">;

interface SymbolInfo {
  readonly name: string;
  readonly kind: "param" | "local" | "shared" | "constant" | "texture" | "cooperative-group" | "device-function" | "builtin-vector" | "builtin-call";
  readonly valueType?: ValueType;
  readonly returnType?: CudaLiteScalarType;
  readonly params?: readonly CudaLiteParam[];
  readonly groupKind?: CudaLiteCooperativeGroupKind;
  readonly tileSize?: number;
  readonly pointer?: boolean;
  readonly constant?: boolean;
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
  const params = new Map(kernel.params.map((param) => [param.name, param]));
  const declaredNames = new Set<string>();
  const rootScope = createScope();

  for (const constant of ast.constants) {
    declareConstant(constant, rootScope, declaredNames, requiredFeatures, diagnostics);
  }
  for (const texture of ast.textures) {
    declareTexture(texture, rootScope, declaredNames, diagnostics);
  }
  for (const fn of ast.functions) {
    if (selectedDeviceFunctionAsKernel && fn.name === kernel.name) continue;
    declareDeviceFunction(fn, rootScope, declaredNames, requiredFeatures, diagnostics);
  }
  const rootDeclaredNames = new Set(declaredNames);

  for (const param of kernel.params) {
    if (declaredNames.has(param.name)) {
      diagnostics.push(error("duplicate-symbol", `duplicate parameter '${param.name}'`, param.span));
    }
    validateDeclaredSymbolName(param.name, param.span, diagnostics);
    declaredNames.add(param.name);
    rootScope.symbols.set(param.name, {
      name: param.name,
      kind: "param",
      valueType: param.valueType,
      pointer: param.pointer,
      constant: param.constant,
      span: param.span,
    });
    if (param.valueType === "half") requiredFeatures.add("shader-f16");
    if (param.valueType === "bool" && param.pointer) {
      diagnostics.push(error("unsupported-bool-pointer", "bool pointer parameters are not supported in CUDA-lite v0", param.span));
    }
  }

  const declareVar = (statement: CudaLiteVarDecl, scope: Scope, names: Set<string>): void => {
    const dimensions = resolvedSharedDimensions(statement, options) ?? statement.dimensions;
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
      dimensions,
      span: statement.span,
    });
  };

  const walkExpression = (expression: CudaLiteExpression, scope: Scope): ExpressionInfo => {
    if (expression.kind === "call") {
      return validateCallExpression(expression, scope, params, atomicParams, atomicShared, requiredFeatures, diagnostics, walkExpression, options);
    }
    return validateNonCallExpression(expression, scope, diagnostics, walkExpression);
  };

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
        case "var":
          declareVar(statement, scope, names);
          if (statement.valueType === "half") requiredFeatures.add("shader-f16");
          if (statement.pointer && !isSupportedLocalPointer(statement, scope)) {
            diagnostics.push(error("unsupported-local-pointer", "local pointer declarations are not supported in CUDA-lite yet", statement.span));
          }
          if (statement.storage === "local" && statement.dimensions.length > 0 && statement.init) {
            diagnostics.push(error("unsupported-local-array-init", "local array initializers are not supported in CUDA-lite yet", statement.span));
          }
          if (statement.dynamicShared && !resolvedSharedDimensions(statement, options)) {
            diagnostics.push(error("dynamic-shared-memory", "__shared__ arrays must have fixed dimensions", statement.span));
          }
          for (const dimension of statement.dimensions) {
            if (!Number.isInteger(dimension) || dimension <= 0) {
              diagnostics.push(error("invalid-array-dimension", "array dimensions must be positive integer literals", statement.span));
            }
          }
            if (statement.init) walkExpression(statement.init, scope);
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
              diagnostics.push(error("divergent-barrier", "__syncthreads() cannot appear in divergent control flow", statement.span));
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
          walkStatements(statement.consequent, createScope(scope), guardDepth + 1, divergent ? divergentDepth + 1 : divergentDepth, loopDepth, new Set(names));
          if (statement.alternate) {
            walkStatements(statement.alternate, createScope(scope), guardDepth + 1, divergent ? divergentDepth + 1 : divergentDepth, loopDepth, new Set(names));
          }
          break;
        }
        case "for": {
          const loopScope = createScope(scope);
          const loopNames = new Set(names);
          if (statement.init?.kind === "var") {
            declareVar(statement.init, loopScope, loopNames);
            if (statement.init.valueType === "half") requiredFeatures.add("shader-f16");
            if (statement.init.pointer && !isSupportedLocalPointer(statement.init, loopScope)) {
              diagnostics.push(error("unsupported-local-pointer", "local pointer declarations are not supported in CUDA-lite yet", statement.init.span));
            }
            if (statement.init.init) walkExpression(statement.init.init, loopScope);
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
            diagnostics.push(error("continue-outside-loop", "continue can only appear inside a for-loop", statement.span));
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
      functionScope.symbols.set(param.name, {
        name: param.name,
        kind: "local",
        valueType: param.valueType,
        pointer: param.pointer,
        constant: param.constant,
        span: param.span,
      });
      if (param.valueType === "half") requiredFeatures.add("shader-f16");
    }
    walkStatements(fn.body, functionScope, 0, 0, 0, functionDeclaredNames);
  }

  walkStatements(kernel.body, rootScope, 0, 0, 0, declaredNames);

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
    sharedDeclarations: collectSharedDeclarations(analysis.kernel.body, options),
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
  if (constant.valueType === "half") requiredFeatures.add("shader-f16");
  for (const dimension of constant.dimensions) {
    if (!Number.isInteger(dimension) || dimension <= 0) {
      diagnostics.push(error("invalid-array-dimension", "array dimensions must be positive integer literals", constant.span));
    }
  }
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
    span: fn.span,
  });
  if (fn.returnType === "half") requiredFeatures.add("shader-f16");
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
  if (!statement.pointer || statement.storage !== "local") return false;
  return isSupportedPoolPointerInitializer(statement.init, scope);
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
  if (!isInlineAsmFma(statement.template)) {
    diagnostics.push(error("unsupported-inline-asm", "only fma.rn.f32 inline PTX is supported in CUDA-lite v0", statement.span));
  }
  validateLValueExpression(statement.output, scope, diagnostics, walkExpression);
  validateScalarOperand(walkExpression(statement.output, scope), statement.output.span, diagnostics);
  if (statement.inputs.length !== 2) {
    diagnostics.push(error("invalid-inline-asm-operands", "fma.rn.f32 inline PTX expects exactly two input operands", statement.span));
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
  requiredFeatures: Set<string>,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
  options: CudaLiteAnalyzeOptions,
): ExpressionInfo {
  const callName = expressionName(expression.callee);
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
  if (callName === "__syncthreads") {
    diagnostics.push(error("barrier-expression", "__syncthreads() must be used as a standalone statement", expression.span));
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
    validateAtomicBuiltin(expression, scope, params, atomicParams, atomicShared, diagnostics, walkExpression);
    return { kind: "scalar" };
  }
  if (callName === "__ldcs") {
    const arg = expression.args[0];
    if (!arg) return { kind: "unknown" };
    const info = walkExpression(arg, scope);
    if (info.kind !== "pointer" && info.kind !== "pool-pointer" && info.kind !== "address" && info.kind !== "unknown") {
      diagnostics.push(error("unsupported-cache-hint-address", "__ldcs expects a pointer expression", arg.span));
    }
    return isCudaVectorType(info.valueType)
      ? { kind: "vector", valueType: info.valueType }
      : { kind: "scalar", valueType: info.valueType };
  }
  if (callName === "__stcs") {
    const target = expression.args[0];
    const value = expression.args[1];
    if (target) {
      const info = walkExpression(target, scope);
      if (info.kind !== "pointer" && info.kind !== "pool-pointer" && info.kind !== "address" && info.kind !== "unknown") {
        diagnostics.push(error("unsupported-cache-hint-address", "__stcs expects a pointer expression", target.span));
      }
    }
    if (value) validateScalarOperand(walkExpression(value, scope), value.span, diagnostics);
    return { kind: "scalar", valueType: "voidptr" };
  }
  const vectorConstructor = cudaVectorConstructorType(callName);
  if (vectorConstructor) {
    for (const arg of expression.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    return { kind: "vector", valueType: vectorConstructor };
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
  if (isShuffleBuiltin(callName)) {
    requiredFeatures.add("subgroups");
    let valueType: ValueType | undefined;
    for (const [index, arg] of expression.args.entries()) {
      const info = walkExpression(arg, scope);
      validateScalarOperand(info, arg.span, diagnostics);
      if (index === 1) valueType = info.valueType;
    }
    return { kind: "scalar", valueType };
  }
  if (callName === "tex2D") {
    validateTex2D(expression, scope, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "float" };
  }
  if (callName === "surf2Dwrite") {
    validateSurf2DWrite(expression, scope, diagnostics, walkExpression);
    return { kind: "scalar", valueType: "float" };
  }
  if (callName === "sizeof") {
    validateSizeof(expression, diagnostics);
    return { kind: "scalar", valueType: "uint" };
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
    if (param?.pointer) {
      validateDevicePointerArgument(arg, param, scope, diagnostics, walkExpression);
      continue;
    }
    const info = walkExpression(arg, scope);
    if (isCudaVectorType(param?.valueType)) {
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
    rootSymbol.dimensions.length === 1 &&
    info.kind === "array";
  if (info.kind !== "pointer" && info.kind !== "address" && info.kind !== "unknown" && !sharedArrayDecay) {
    diagnostics.push(error("unsupported-device-pointer-param", `device pointer parameter '${param.name}' expects a pointer argument`, arg.span));
    return;
  }
  if (rootSymbol?.kind === "shared" && rootSymbol.dimensions && rootSymbol.dimensions.length > 1) {
    diagnostics.push(error("unsupported-device-pointer-param", `device pointer parameter '${param.name}' only supports one-dimensional shared arrays`, arg.span));
  }
  if (rootSymbol?.kind === "constant") {
    diagnostics.push(error("unsupported-device-pointer-param", `device pointer parameter '${param.name}' expects storage-buffer memory`, arg.span));
  }
  if (rootSymbol?.pointer && rootSymbol.constant && !param.constant) {
    diagnostics.push(error("const-pointer-write", `cannot pass const pointer '${root}' to writable device pointer parameter '${param.name}'`, arg.span));
  }
  const actualValueType = info.valueType ?? rootSymbol?.valueType;
  if (actualValueType && actualValueType !== param.valueType) {
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
  if (method === "shfl_down" || method === "shfl_up" || method === "shfl_xor") {
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
  diagnostics.push(error("unsupported-cooperative-groups", `unsupported cooperative group method '${method}'`, expression.span));
  for (const arg of expression.args) validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
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

function validateTex2D(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const texture = expression.args[0];
  if (texture?.kind !== "identifier") {
    diagnostics.push(error("unsupported-texture", "tex2D first argument must be a texture reference", expression.span));
  } else {
    const symbol = lookupSymbol(texture.name, scope, texture.span);
    if (symbol?.kind !== "texture") {
      diagnostics.push(error("unsupported-texture", `tex2D target '${texture.name}' is not a texture reference`, texture.span));
    }
  }
  for (const coord of expression.args.slice(1)) {
    validateScalarOperand(walkExpression(coord, scope), coord.span, diagnostics);
  }
}

function validateSurf2DWrite(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const value = expression.args[0];
  const surface = expression.args[1];
  const xBytes = expression.args[2];
  const y = expression.args[3];
  if (value) validateScalarOperand(walkExpression(value, scope), value.span, diagnostics);
  if (surface?.kind !== "identifier") {
    diagnostics.push(error("unsupported-surface", "surf2Dwrite second argument must be a surface object", expression.span));
  } else {
    const symbol = lookupSymbol(surface.name, scope, surface.span);
    if (symbol?.valueType !== "surface2d") {
      diagnostics.push(error("unsupported-surface", `surf2Dwrite target '${surface.name}' is not a cudaSurfaceObject_t parameter`, surface.span));
    }
  }
  if (xBytes) validateScalarOperand(walkExpression(xBytes, scope), xBytes.span, diagnostics);
  if (y) validateScalarOperand(walkExpression(y, scope), y.span, diagnostics);
}

function validateSizeof(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  diagnostics: CudaLiteDiagnostic[],
): void {
  const target = expression.args[0];
  if (target?.kind !== "identifier" || sizeofType(target.name) === undefined) {
    diagnostics.push(error("unsupported-sizeof", "sizeof only supports CUDA-lite scalar types", target?.span ?? expression.span));
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
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): void {
  const target = expression.args[0];
  const callName = expressionName(expression.callee);
  const targetExpression = atomicTargetExpression(target);
  if (!targetExpression) {
    diagnostics.push(error("atomic-address-required", "atomicAdd first argument must be a pointer parameter or address like &x[i]", expression.span));
    if (target) walkExpression(target, scope);
  } else {
    if (targetExpression.kind !== "identifier") {
      validateLValueExpression(targetExpression, scope, diagnostics, walkExpression);
    }
    const targetName = rootIdentifier(targetExpression);
    const param = targetName ? params.get(targetName) : undefined;
    const symbol = targetName ? lookupSymbol(targetName, scope, targetExpression.span) : undefined;
    if (symbol?.kind === "shared") {
      if (symbol.valueType === "float" || symbol.valueType === "half" || symbol.valueType === "bool") {
        diagnostics.push(error("unsupported-atomic-target", "shared atomics support int/uint targets in CUDA-lite v0", targetExpression.span));
      } else {
        atomicShared.add(symbol.name);
      }
    } else if (!param?.pointer) {
      diagnostics.push(error("unsupported-atomic-target", "atomicAdd target must be a pointer parameter element", targetExpression.span));
    } else if (param.valueType === "float" && (
      callName === "atomicAdd" ||
      callName === "atomicSub" ||
      callName === "atomicMin" ||
      callName === "atomicMax" ||
      callName === "atomicMaxFloat" ||
      callName === "atomicExch"
    )) {
      atomicParams.add(param.name);
      if (param.constant) {
        diagnostics.push(error("const-pointer-write", `cannot ${callName} through const pointer '${param.name}'`, expression.span));
      }
    } else if (param.valueType === "float" || param.valueType === "half" || param.valueType === "complex64") {
      diagnostics.push(error("unsupported-atomic-f32", "unsupported float atomic operation in CUDA-lite v0", expression.span));
    } else {
      atomicParams.add(param.name);
      if (param.constant) {
        diagnostics.push(error("const-pointer-write", `cannot atomicAdd through const pointer '${param.name}'`, expression.span));
      }
    }
  }
  for (const arg of expression.args.slice(1)) {
    validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
  }
}

function isAtomicBuiltin(callName: string): boolean {
  return callName === "atomicAdd" ||
    callName === "atomicSub" ||
    callName === "atomicMin" ||
    callName === "atomicMax" ||
    callName === "atomicMaxFloat" ||
    callName === "atomicExch" ||
    callName === "atomicCAS";
}

function isShuffleBuiltin(callName: string): boolean {
  return callName === "__shfl_down_sync" ||
    callName === "__shfl_up_sync" ||
    callName === "__shfl_xor_sync";
}

function atomicTargetExpression(
  target: CudaLiteExpression | undefined,
): CudaLiteExpression | undefined {
  if (!target) return undefined;
  if (target.kind === "unary" && target.operator === "&") return target.argument;
  if (target.kind === "identifier") return target;
  return undefined;
}

function validateNonCallExpression(
  expression: CudaLiteExpression,
  scope: Scope,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): ExpressionInfo {
  switch (expression.kind) {
    case "number":
      return { kind: "scalar" };
    case "string":
      return { kind: "string" };
    case "identifier":
      return expressionInfoForIdentifier(expression.name, expression.span, scope, diagnostics);
    case "cast": {
      const info = walkExpression(expression.expression, scope);
      if (expression.pointer) {
        if (info.kind !== "scalar" && info.kind !== "pointer" && info.kind !== "pool-pointer" && info.kind !== "unknown") {
          diagnostics.push(error("unsupported-pointer-cast", "pointer cast expects scalar or pointer expression", expression.expression.span));
        }
        return { kind: "pool-pointer", valueType: expression.valueType };
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
        return { kind: "scalar", valueType: info.valueType };
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
      validateScalarOperand(left, expression.left.span, diagnostics);
      validateScalarOperand(right, expression.right.span, diagnostics);
      return { kind: "scalar" };
    }
    case "conditional": {
      validateScalarOperand(walkExpression(expression.condition, scope), expression.condition.span, diagnostics);
      validateScalarOperand(walkExpression(expression.consequent, scope), expression.consequent.span, diagnostics);
      validateScalarOperand(walkExpression(expression.alternate, scope), expression.alternate.span, diagnostics);
      return { kind: "scalar" };
    }
    case "assignment": {
      validateLValueExpression(expression.left, scope, diagnostics, walkExpression);
      const left = walkExpression(expression.left, scope);
      const right = walkExpression(expression.right, scope);
      if (left.kind === "complex") {
        if (right.kind !== "complex" && right.kind !== "unknown") {
          diagnostics.push(error("unsupported-scalar-expression", "complex assignment expects a complex value", expression.right.span));
        }
      } else if (left.kind === "vector") {
        if (right.kind !== "vector" && right.kind !== "unknown") {
          diagnostics.push(error("unsupported-vector-assignment", "CUDA vector assignment expects a CUDA vector value", expression.right.span));
        }
      } else {
        validateScalarOperand(right, expression.right.span, diagnostics);
      }
      return { kind: "scalar" };
    }
    case "update": {
      validateLValueExpression(expression.argument, scope, diagnostics, walkExpression);
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
  diagnostics.push(error("invalid-assignment-target", "assignment target must be a local variable, pointer element, or shared element", expression.span));
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
  if (BUILTIN_VECTORS.has(name) || BUILTIN_CALLS.has(name)) {
    diagnostics.push(error("reserved-symbol", `symbol '${name}' conflicts with a CUDA-lite builtin`, span));
    return;
  }
  if (WGSL_RESERVED_WORDS.has(name)) {
    diagnostics.push(error("reserved-symbol", `symbol '${name}' is reserved by WGSL output`, span));
  }
}

function isInlineAsmFma(template: string): boolean {
  return /\bfma\.rn\.f32\b/u.test(template);
}

function sizeofType(typeName: string): number | undefined {
  switch (typeName) {
    case "float":
    case "int":
    case "uint":
    case "unsigned":
    case "signed":
    case "long":
    case "short":
    case "size_t":
    case "int32_t":
    case "uint32_t":
    case "int64_t":
    case "uint64_t":
    case "uintptr_t":
      return 4;
    case "half":
    case "__half":
      return 2;
    case "bool":
      return 4;
    case "cufftComplex":
      return 8;
    case "float2":
    case "int2":
    case "uint2":
      return 8;
    case "float3":
    case "int3":
    case "uint3":
      return 12;
    case "float4":
    case "int4":
    case "uint4":
      return 16;
    default:
      return undefined;
  }
}

function validateSideEffectPlacement(
  expression: CudaLiteExpression,
  allowRootSideEffect: boolean,
  diagnostics: CudaLiteDiagnostic[],
): void {
  const visit = (node: CudaLiteExpression, root: boolean): void => {
    if ((node.kind === "assignment" || node.kind === "update") && !(allowRootSideEffect && root)) {
      diagnostics.push(error(
        "side-effect-expression",
        "assignments and ++/-- must be standalone statements or for-loop clauses",
        node.span,
      ));
    }
    forEachExpressionChild(node, (child) => {
      visit(child, false);
    });
  };
  visit(expression, true);
}

function validateBarrierStatement(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  diagnostics: CudaLiteDiagnostic[],
): void {
  if (expression.args.length !== 0) {
    diagnostics.push(error("invalid-call-arity", "__syncthreads expects 0 arguments", expression.span));
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
  return expression.kind === "call" && expressionName(expression.callee) === "__syncthreads";
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
  }
}

export function expressionName(expression: CudaLiteExpression): string | undefined {
  if (expression.kind === "identifier") return expression.name;
  return undefined;
}

export function rootIdentifier(expression: CudaLiteExpression): string | undefined {
  if (expression.kind === "identifier") return expression.name;
  if (expression.kind === "index") return rootIdentifier(expression.target);
  if (expression.kind === "member") return rootIdentifier(expression.object);
  if (expression.kind === "unary" && expression.operator === "&") return rootIdentifier(expression.argument);
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
