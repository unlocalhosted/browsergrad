import type {
  CudaLiteSemanticFunction,
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import { CudaLiteCompilerError, type CudaLiteDiagnostic, type SourceSpan } from "./types.js";

export interface SemanticIrVerificationIssue {
  readonly code: "internal-lowering-invariant";
  readonly message: string;
  readonly span: SourceSpan;
}

export function verifySemanticKernelIr(
  ir: SemanticKernelIrModule,
): readonly SemanticIrVerificationIssue[] {
  const issues: SemanticIrVerificationIssue[] = [];
  const report = (message: string, span: SourceSpan): void => {
    issues.push({ code: "internal-lowering-invariant", message, span });
  };

  verifySpan(ir.span, "module", report);
  if (!ir.name) report("IR module name must not be empty", ir.span);
  if (ir.workgroupSize.length !== 3 || ir.workgroupSize.some((value) => !Number.isInteger(value) || value <= 0)) {
    report("IR workgroup size must contain three positive integers", ir.span);
  }

  for (const symbol of [...ir.params, ...ir.memory]) {
    verifySpan(symbol.span, `symbol '${symbol.name}'`, report);
    if (!symbol.name) report("IR symbol name must not be empty", symbol.span);
    if (symbol.dimensions.some((value) => !Number.isInteger(value) || value < 0)) {
      report(`IR symbol '${symbol.name}' has an invalid dimension`, symbol.span);
    }
  }

  const signatures = new Set<string>();
  for (const fn of ir.functions) {
    const signature = `${fn.name}(${fn.params.map((param) => `${param.pointer ? "*" : ""}${param.valueType ?? "?"}`).join(",")})`;
    if (signatures.has(signature)) report(`IR contains duplicate function signature '${signature}'`, fn.span);
    signatures.add(signature);
    verifyFunction(fn, report);
  }
  verifyOperations(ir.operations, undefined, 0, report);
  return issues;
}

export function assertValidSemanticKernelIr(
  ir: SemanticKernelIrModule,
): void {
  const issues = verifySemanticKernelIr(ir);
  if (issues.length === 0) return;
  const diagnostics: readonly CudaLiteDiagnostic[] = issues.map((issue) => ({
    code: issue.code,
    severity: "error",
    message: issue.message,
    span: issue.span,
  }));
  throw new CudaLiteCompilerError(`invalid semantic Kernel IR: ${issues[0]!.message}`, diagnostics);
}

function verifyFunction(
  fn: CudaLiteSemanticFunction,
  report: (message: string, span: SourceSpan) => void,
): void {
  verifySpan(fn.span, `function '${fn.name}'`, report);
  if (!fn.name) report("IR function name must not be empty", fn.span);
  verifyOperations(fn.body, fn, 0, report);
}

function verifyOperations(
  operations: readonly SemanticKernelIrOperation[],
  fn: CudaLiteSemanticFunction | undefined,
  loopDepth: number,
  report: (message: string, span: SourceSpan) => void,
): void {
  for (const operation of operations) {
    verifySpan(operation.span, `operation '${operation.kind}'`, report);
    for (const expression of operationExpressions(operation)) verifyExpression(expression, report);
    for (const ref of operationMemoryRefs(operation)) verifyMemoryRef(ref, operation, report);

    if (operation.kind === "store" && operation.target.addressSpace === "constant") {
      report("IR store cannot target constant memory", operation.span);
    }
    if (operation.kind === "copy" && (!Number.isInteger(operation.bytes) || operation.bytes <= 0)) {
      report("IR copy byte count must be a positive integer", operation.span);
    }
    if ((operation.kind === "break" || operation.kind === "continue") && loopDepth === 0) {
      report(`IR ${operation.kind} must be nested in a loop`, operation.span);
    }
    if (operation.kind === "return") {
      if (!fn && operation.value !== undefined) report("Kernel IR return cannot carry a value", operation.span);
      if (fn?.returnType === "void" && operation.value !== undefined) report(`Void function '${fn.name}' cannot return a value`, operation.span);
      if (fn?.returnType !== undefined && fn.returnType !== "void" && operation.value === undefined) {
        report(`Non-void function '${fn.name}' must return a value`, operation.span);
      }
    }

    if (operation.kind === "branch") {
      verifyOperations(operation.consequent, fn, loopDepth, report);
      verifyOperations(operation.alternate, fn, loopDepth, report);
    } else if (operation.kind === "block") {
      verifyOperations(operation.body, fn, loopDepth, report);
    } else if (operation.kind === "loop") {
      if (operation.init && isOperation(operation.init)) verifyOperations([operation.init], fn, loopDepth, report);
      verifyOperations(operation.body, fn, loopDepth + 1, report);
      if (operation.continuing) verifyOperations(operation.continuing, fn, loopDepth + 1, report);
    }
  }
}

function verifyExpression(
  expression: SemanticExpression,
  report: (message: string, span: SourceSpan) => void,
): void {
  verifySpan(expression.span, `expression '${expression.kind}'`, report);
  if (expression.kind === "symbol" && !expression.name) report("IR symbol expression name must not be empty", expression.span);
  for (const child of expressionChildren(expression)) verifyExpression(child, report);
}

function verifyMemoryRef(
  ref: SemanticMemoryRef,
  operation: SemanticKernelIrOperation,
  report: (message: string, span: SourceSpan) => void,
): void {
  verifySpan(ref.span, `memory reference '${ref.base}'`, report);
  if (!ref.base) report(`IR ${operation.kind} has an empty memory root`, ref.span);
  if (ref.addressSpace === "unknown" || ref.addressSpace === "builtin" || ref.addressSpace === "function" || ref.addressSpace === "uniform") {
    report(`IR ${operation.kind} uses invalid memory address space '${ref.addressSpace}'`, ref.span);
  }
  for (const index of ref.indices) verifyExpression(index, report);
}

function verifySpan(
  span: SourceSpan,
  owner: string,
  report: (message: string, span: SourceSpan) => void,
): void {
  if (
    !Number.isInteger(span.start) ||
    !Number.isInteger(span.end) ||
    span.start < 0 ||
    span.end < span.start ||
    !Number.isInteger(span.line) ||
    span.line < 1 ||
    !Number.isInteger(span.column) ||
    span.column < 1
  ) report(`IR ${owner} has an invalid source span`, span);
}

function operationExpressions(operation: SemanticKernelIrOperation): readonly SemanticExpression[] {
  switch (operation.kind) {
    case "declare": return operation.init ? [operation.init] : [];
    case "dim3-declare": return operation.args;
    case "cooperative-group-declare": return operation.declaration.partitionPredicate ? [operation.declaration.partitionPredicate] : [];
    case "store": return [operation.value];
    case "matrix-fill": return [operation.value];
    case "matrix-load":
    case "matrix-store": return [operation.stride];
    case "surface-write": return [operation.surface, operation.value, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : [])];
    case "surface-read-store": return [operation.target, operation.surface, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : [])];
    case "atomic": return operation.args;
    case "call": return [...operation.args, ...(operation.result ? [operation.result] : [])];
    case "expression": return [operation.expression];
    case "branch": return [operation.condition];
    case "loop": return [
      ...(operation.init && !isOperation(operation.init) ? [operation.init] : []),
      ...(operation.condition ? [operation.condition] : []),
      ...(operation.update ? [operation.update] : []),
    ];
    case "device-launch": return [...operation.launch.grid, ...operation.launch.block, ...operation.launch.args];
    case "inline-asm": return [...operation.outputs, ...operation.inputs];
    case "return": return operation.value ? [operation.value] : [];
    default: return [];
  }
}

function operationMemoryRefs(operation: SemanticKernelIrOperation): readonly SemanticMemoryRef[] {
  switch (operation.kind) {
    case "load": return [operation.source];
    case "store": return [operation.target, ...operation.reads];
    case "copy": return [operation.source, operation.target];
    case "matrix-load": return [operation.source];
    case "matrix-store": return [operation.target];
    case "atomic": return operation.target ? [operation.target] : [];
    case "call": return operation.reads;
    default: return [];
  }
}

function expressionChildren(expression: SemanticExpression): readonly SemanticExpression[] {
  switch (expression.kind) {
    case "member": return [expression.object];
    case "index": return [expression.target, expression.index];
    case "call": return [expression.callee, ...expression.args];
    case "texture-read": return [expression.texture, expression.x, expression.y, ...(expression.z ? [expression.z] : [])];
    case "surface-read": return [expression.surface, expression.xBytes, expression.y, ...(expression.z ? [expression.z] : [])];
    case "cast": return [expression.expression];
    case "unary": return [expression.argument];
    case "binary": return [expression.left, expression.right];
    case "conditional": return [expression.condition, expression.consequent, expression.alternate];
    case "assignment": return [expression.target, expression.value];
    case "update": return [expression.argument];
    case "initializer": return expression.elements;
    case "sequence": return expression.expressions;
    default: return [];
  }
}

function isOperation(value: SemanticKernelIrOperation | SemanticExpression): value is SemanticKernelIrOperation {
  if (value.kind === "call") return "reads" in value;
  return value.kind === "declare" || value.kind === "dim3-declare" || value.kind === "cooperative-group-declare" ||
    value.kind === "load" || value.kind === "store" || value.kind === "copy" || value.kind === "copy-fence" ||
    value.kind === "matrix-fill" || value.kind === "matrix-load" || value.kind === "matrix-mma" || value.kind === "matrix-store" ||
    value.kind === "surface-write" || value.kind === "surface-read-store" || value.kind === "atomic" || value.kind === "expression" ||
    value.kind === "branch" || value.kind === "loop" || value.kind === "barrier" || value.kind === "fence" ||
    value.kind === "device-launch" || value.kind === "inline-asm" || value.kind === "return" || value.kind === "continue" ||
    value.kind === "break" || value.kind === "block";
}
