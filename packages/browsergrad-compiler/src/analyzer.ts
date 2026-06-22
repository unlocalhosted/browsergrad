import {
  CudaLiteCompilerError,
  type CudaLiteAnalysis,
  type CudaLiteAnalyzeOptions,
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

const DEFAULT_WORKGROUP_SIZE: readonly [number, number, number] = [256, 1, 1];
const BUILTIN_VECTORS = new Set(["threadIdx", "blockIdx", "blockDim", "gridDim"]);
const BUILTIN_CALLS = new Map<string, readonly [min: number, max: number]>([
  ["__syncthreads", [0, 0]],
  ["sqrtf", [1, 1]],
  ["expf", [1, 1]],
  ["logf", [1, 1]],
  ["__half2float", [1, 1]],
  ["__float2half", [1, 1]],
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
  ["atomicExch", [2, 2]],
  ["atomicCAS", [3, 3]],
  ["tex2D", [3, 3]],
  ["printf", [1, Number.POSITIVE_INFINITY]],
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
  readonly kind: "param" | "local" | "shared" | "constant" | "texture" | "device-function" | "builtin-vector" | "builtin-call";
  readonly valueType?: ValueType;
  readonly returnType?: CudaLiteScalarType;
  readonly params?: readonly CudaLiteParam[];
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
  readonly kind: "scalar" | "pointer" | "array" | "texture" | "vector" | "function" | "address" | "string" | "unknown";
  readonly valueType?: ValueType | undefined;
  readonly dimensions?: readonly number[] | undefined;
  readonly symbol?: SymbolInfo | undefined;
}

export function analyzeCudaLite(
  ast: CudaLiteModule,
  options: CudaLiteAnalyzeOptions = {},
): CudaLiteAnalysis {
  const kernel = selectKernel(ast, options.kernelName);
  const diagnostics: CudaLiteDiagnostic[] = [];
  const requiredFeatures = new Set<string>();
  const atomicParams = new Set<string>();
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
    declareDeviceFunction(fn, rootScope, declaredNames, requiredFeatures, diagnostics);
  }

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
      return validateCallExpression(expression, scope, params, atomicParams, requiredFeatures, diagnostics, walkExpression);
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
          if (statement.pointer && !isSupportedSharedPointerAlias(statement, scope)) {
            diagnostics.push(error("unsupported-local-pointer", "local pointer declarations are not supported in CUDA-lite yet", statement.span));
          }
          if (statement.storage === "local" && statement.dimensions.length > 0) {
            diagnostics.push(error("unsupported-local-array", "local arrays are not supported in CUDA-lite v0; use fixed __shared__ arrays or scalar locals", statement.span));
          }
          if (statement.storage === "shared" && statement.dimensions.length === 0 && !resolvedSharedDimensions(statement, options)) {
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
          break;
        case "kernel-launch":
          diagnostics.push(error(
            "unsupported-dynamic-parallelism",
            `device-side kernel launch '${statement.callee}<<<...>>>' is not lowered in CUDA-lite v0`,
            statement.span,
          ));
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
          walkStatements(statement.consequent, createScope(scope), guardDepth + 1, divergent ? divergentDepth + 1 : divergentDepth, loopDepth, names);
          if (statement.alternate) {
            walkStatements(statement.alternate, createScope(scope), guardDepth + 1, divergent ? divergentDepth + 1 : divergentDepth, loopDepth, names);
          }
          break;
        }
        case "for": {
          const loopScope = createScope(scope);
          if (statement.init?.kind === "var") {
            declareVar(statement.init, loopScope, names);
            if (statement.init.valueType === "half") requiredFeatures.add("shader-f16");
            if (statement.init.pointer && !isSupportedSharedPointerAlias(statement.init, loopScope)) {
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
          walkStatements(statement.body, loopScope, guardDepth, divergent ? divergentDepth + 1 : divergentDepth, loopDepth + 1, names);
          break;
        }
        case "return":
          if (statement.value) {
            validateSideEffectPlacement(statement.value, false, diagnostics);
            validateScalarOperand(walkExpression(statement.value, scope), statement.value.span, diagnostics);
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
    const functionScope = createScope(rootScope);
    const functionDeclaredNames = new Set(declaredNames);
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
      if (param.pointer) {
        diagnostics.push(error("unsupported-device-pointer-param", "CUDA-lite device functions only support scalar params in v0", param.span));
      }
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
    workgroupSize: normalizeWorkgroupSize(options.workgroupSize ?? DEFAULT_WORKGROUP_SIZE),
  };
}

function selectKernel(ast: CudaLiteModule, kernelName: string | undefined): CudaLiteKernel {
  if (ast.kernels.length === 0) {
    throw new CudaLiteCompilerError("no CUDA-lite kernels found", [{
      code: "missing-kernel",
      severity: "error",
      message: "no CUDA-lite kernels found",
      span: ast.span,
    }]);
  }
  if (!kernelName) return ast.kernels[0]!;
  const kernel = ast.kernels.find((candidate) => candidate.name === kernelName);
  if (!kernel) {
    throw new CudaLiteCompilerError(`CUDA-lite kernel '${kernelName}' not found`, [{
      code: "missing-kernel",
      severity: "error",
      message: `CUDA-lite kernel '${kernelName}' not found`,
      span: ast.span,
    }]);
  }
  return kernel;
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

type ExpressionWalker = (expression: CudaLiteExpression, scope: Scope) => ExpressionInfo;

function validateCallExpression(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  params: ReadonlyMap<string, CudaLiteParam>,
  atomicParams: Set<string>,
  requiredFeatures: Set<string>,
  diagnostics: CudaLiteDiagnostic[],
  walkExpression: ExpressionWalker,
): ExpressionInfo {
  const callName = expressionName(expression.callee);
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
  if (isAtomicBuiltin(callName)) {
    validateAtomicBuiltin(expression, scope, params, atomicParams, diagnostics, walkExpression);
    return { kind: "scalar" };
  }
  if (callName === "__half2float" || callName === "__float2half") {
    requiredFeatures.add("shader-f16");
    for (const arg of expression.args) {
      validateScalarOperand(walkExpression(arg, scope), arg.span, diagnostics);
    }
    return {
      kind: "scalar",
      valueType: callName === "__half2float" ? "float" : "half",
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

  for (const arg of expression.args) {
    const info = walkExpression(arg, scope);
    validateScalarOperand(info, arg.span, diagnostics);
  }
  return { kind: "scalar" };
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
    const info = walkExpression(arg, scope);
    validateScalarOperand(info, arg.span, diagnostics);
    const param = fnParams[index];
    if (param?.pointer) {
      diagnostics.push(error("unsupported-device-pointer-param", "CUDA-lite device function calls only support scalar params in v0", arg.span));
    }
  }
  if (symbol.returnType === undefined || symbol.returnType === "void") return { kind: "unknown" };
  return { kind: "scalar", valueType: symbol.returnType };
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

function validateAtomicBuiltin(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  scope: Scope,
  params: ReadonlyMap<string, CudaLiteParam>,
  atomicParams: Set<string>,
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
    if (!param?.pointer) {
      diagnostics.push(error("unsupported-atomic-target", "atomicAdd target must be a pointer parameter element", targetExpression.span));
    } else if (param.valueType === "float" && (callName === "atomicAdd" || callName === "atomicExch")) {
      atomicParams.add(param.name);
      if (param.constant) {
        diagnostics.push(error("const-pointer-write", `cannot ${callName} through const pointer '${param.name}'`, expression.span));
      }
    } else if (param.valueType === "float" || param.valueType === "half") {
      diagnostics.push(error("unsupported-atomic-f32", "only atomicAdd/atomicExch are supported for float pointers in CUDA-lite v0", expression.span));
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
      validateScalarOperand(info, expression.expression.span, diagnostics);
      return { kind: "scalar", valueType: expression.valueType };
    }
    case "member": {
      const object = walkExpression(expression.object, scope);
      if (object.kind !== "vector") {
        diagnostics.push(error("unsupported-member-target", "member access is only supported on CUDA-lite builtin vectors", expression.span));
      }
      return { kind: "scalar", valueType: "int" };
    }
    case "index": {
      const target = walkExpression(expression.target, scope);
      validateScalarOperand(walkExpression(expression.index, scope), expression.index.span, diagnostics);
      if (target.kind === "pointer") return { kind: "scalar", valueType: target.valueType, symbol: target.symbol };
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
        return { kind: "scalar", valueType: target.valueType, symbol: target.symbol };
      }
      diagnostics.push(error("unsupported-index-target", "only pointer parameters and fixed __shared__ arrays can be indexed", expression.span));
      return { kind: "unknown" };
    }
    case "unary": {
      if (expression.operator === "&") {
        validateLValueExpression(expression.argument, scope, diagnostics, walkExpression);
        return { kind: "address" };
      }
      const info = walkExpression(expression.argument, scope);
      validateScalarOperand(info, expression.argument.span, diagnostics);
      return { kind: "scalar", valueType: info.valueType };
    }
    case "binary": {
      const left = walkExpression(expression.left, scope);
      const right = walkExpression(expression.right, scope);
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
      validateScalarOperand(walkExpression(expression.right, scope), expression.right.span, diagnostics);
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
    if (symbol.kind === "local") return;
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
    if (info.kind !== "scalar") {
      diagnostics.push(error("invalid-assignment-target", "assignment target must resolve to a scalar element", expression.span));
    }
    return;
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
  if (!symbol) {
    diagnostics.push(error("unknown-symbol", `unknown CUDA-lite symbol '${name}'`, span));
    return { kind: "unknown" };
  }
  if (symbol.kind === "builtin-vector") return { kind: "vector", symbol };
  if (symbol.kind === "builtin-call") return { kind: "function", symbol };
  if (symbol.kind === "device-function") return { kind: "function", symbol };
  if (symbol.kind === "texture") return { kind: "texture", valueType: symbol.valueType, symbol };
  if (symbol.kind === "shared" || symbol.kind === "constant") {
    return {
      kind: symbol.dimensions && symbol.dimensions.length > 0 ? "array" : "scalar",
      valueType: symbol.valueType,
      dimensions: symbol.dimensions,
      symbol,
    };
  }
  if (symbol.kind === "param" && symbol.pointer) {
    return { kind: "pointer", valueType: symbol.valueType, symbol };
  }
  if (symbol.kind === "local" && symbol.pointer) {
    return { kind: "pointer", valueType: symbol.valueType, symbol };
  }
  return { kind: "scalar", valueType: symbol.valueType, symbol };
}

function validateScalarOperand(
  info: ExpressionInfo,
  span: SourceSpan,
  diagnostics: CudaLiteDiagnostic[],
): void {
  if (info.kind === "scalar" || info.kind === "unknown") return;
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
  if (statement.storage !== "shared" || statement.dimensions.length > 0) return undefined;
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
