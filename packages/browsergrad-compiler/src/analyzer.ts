import {
  CudaLiteCompilerError,
  type CudaLiteAnalysis,
  type CudaLiteAnalyzeOptions,
  type CudaLiteDiagnostic,
  type CudaLiteExpression,
  type CudaLiteKernel,
  type CudaLiteModule,
  type CudaLiteParam,
  type CudaLiteStatement,
  type CudaLiteVarDecl,
  type KernelIrModule,
  type SourceSpan,
} from "./types.js";

const DEFAULT_WORKGROUP_SIZE: readonly [number, number, number] = [256, 1, 1];

export function analyzeCudaLite(
  ast: CudaLiteModule,
  options: CudaLiteAnalyzeOptions = {},
): CudaLiteAnalysis {
  const kernel = selectKernel(ast, options.kernelName);
  const diagnostics: CudaLiteDiagnostic[] = [];
  const requiredFeatures = new Set<string>();
  const atomicParams = new Set<string>();
  const params = new Map(kernel.params.map((param) => [param.name, param]));

  for (const param of kernel.params) {
    if (param.valueType === "half") requiredFeatures.add("shader-f16");
  }

  const walkExpression = (expression: CudaLiteExpression): void => {
    if (expression.kind === "call") {
      const callName = expressionName(expression.callee);
      if (callName === "bg_subgroup_add") requiredFeatures.add("subgroups");
      if (callName === "atomicAdd") {
        const target = expression.args[0];
        const targetName = target ? rootIdentifier(target) : undefined;
        const param = targetName ? params.get(targetName) : undefined;
        if (!param) {
          diagnostics.push(error("unsupported-atomic-target", "atomicAdd target must be a pointer parameter", expression.span));
        } else if (param.valueType === "float" || param.valueType === "half") {
          diagnostics.push(error("unsupported-atomic-f32", "atomicAdd is only supported for int/uint pointers in CUDA-lite v0", expression.span));
        } else {
          atomicParams.add(param.name);
          if (param.constant) {
            diagnostics.push(error("const-pointer-write", `cannot atomicAdd through const pointer '${param.name}'`, expression.span));
          }
        }
      }
    }
    forEachExpressionChild(expression, walkExpression);
  };

  const walkStatements = (
    statements: readonly CudaLiteStatement[],
    guardDepth: number,
    divergentDepth: number,
  ): void => {
    for (const statement of statements) {
      switch (statement.kind) {
        case "var":
          if (statement.valueType === "half") requiredFeatures.add("shader-f16");
          if (statement.storage === "shared" && statement.dimensions.length === 0) {
            diagnostics.push(error("dynamic-shared-memory", "__shared__ arrays must have fixed dimensions", statement.span));
          }
          if (statement.init) walkExpression(statement.init);
          break;
        case "expr":
          if (isBarrierCall(statement.expression)) {
            if (divergentDepth > 0) {
              diagnostics.push(error("divergent-barrier", "__syncthreads() cannot appear in divergent control flow", statement.span));
            }
          } else {
            validateExpressionStatement(statement.expression, params, guardDepth, diagnostics);
          }
          walkExpression(statement.expression);
          break;
        case "if": {
          walkExpression(statement.condition);
          const divergent = expressionIsDivergent(statement.condition, params);
          walkStatements(statement.consequent, guardDepth + 1, divergent ? divergentDepth + 1 : divergentDepth);
          if (statement.alternate) {
            walkStatements(statement.alternate, guardDepth + 1, divergent ? divergentDepth + 1 : divergentDepth);
          }
          break;
        }
        case "for": {
          if (statement.init?.kind === "var") {
            if (statement.init.init) walkExpression(statement.init.init);
          } else if (statement.init) {
            walkExpression(statement.init);
          }
          if (statement.condition) walkExpression(statement.condition);
          if (statement.update) walkExpression(statement.update);
          const divergent = statement.condition ? expressionIsDivergent(statement.condition, params) : false;
          walkStatements(statement.body, guardDepth, divergent ? divergentDepth + 1 : divergentDepth);
          break;
        }
      }
    }
  };

  walkStatements(kernel.body, 0, 0);

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
  const errors = analysis.diagnostics.filter((diagnostic) => diagnostic.severity === "error");
  if (errors.length > 0) {
    throw new CudaLiteCompilerError("CUDA-lite analysis failed", errors);
  }
  return {
    name: analysis.kernel.name,
    params: analysis.kernel.params,
    body: analysis.kernel.body,
    sharedDeclarations: collectSharedDeclarations(analysis.kernel.body),
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
    diagnostics.push(error("unguarded-write", `write to pointer '${root}' must be guarded in CUDA-lite v0`, expression.span));
  }
}

function collectSharedDeclarations(statements: readonly CudaLiteStatement[]): readonly CudaLiteVarDecl[] {
  const declarations: CudaLiteVarDecl[] = [];
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.storage === "shared") declarations.push(item);
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

function isBarrierCall(expression: CudaLiteExpression): boolean {
  return expression.kind === "call" && expressionName(expression.callee) === "__syncthreads";
}

function forEachExpressionChild(
  expression: CudaLiteExpression,
  visit: (child: CudaLiteExpression) => void,
): void {
  switch (expression.kind) {
    case "number":
    case "identifier":
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
