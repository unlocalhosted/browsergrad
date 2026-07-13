import type {
  CudaLiteSemanticFunction,
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import type { VerifiedSemanticKernelIr } from "./semantic_ir_verifier.js";
import { CudaLiteCompilerError, type CudaLiteDiagnostic, type CudaLiteScalarType, type SourceSpan } from "./types.js";
import { semanticBinaryResultType } from "./semantic_type_rules.js";
import { isSemanticValueType } from "./semantic_value_type.js";
import { completeIrTypeChecking, type TypeCheckedIrArtifact } from "./compiler_phases.js";
import { semanticFunctionIdFromSymbol, semanticIdsEqual } from "./semantic_ids.js";

const typeCheckedSemanticKernelIrArtifact: unique symbol = Symbol("type-checked-semantic-kernel-ir");

export interface TypeCheckedSemanticKernelIr<T extends SemanticKernelIrModule = SemanticKernelIrModule> extends TypeCheckedIrArtifact<T> {
  readonly verified: VerifiedSemanticKernelIr<T>;
  readonly [typeCheckedSemanticKernelIrArtifact]: true;
}

export interface SemanticTypeIssue {
  readonly code: "internal-semantic-type-invariant";
  readonly message: string;
  readonly span: SourceSpan;
}

type ExpressionUse = "value" | "discard" | "callee";

export function checkSemanticKernelIrTypes(
  verified: VerifiedSemanticKernelIr,
): readonly SemanticTypeIssue[] {
  const ir = verified.ir;
  const issues: SemanticTypeIssue[] = [];
  const report = (message: string, span: SourceSpan): void => {
    issues.push({ code: "internal-semantic-type-invariant", message, span });
  };

  checkOperations(ir.operations, undefined, ir, report);
  for (const fn of ir.functions) checkOperations(fn.body, fn, ir, report);
  return issues;
}

export function typeCheckSemanticKernelIr<T extends SemanticKernelIrModule>(
  verified: VerifiedSemanticKernelIr<T>,
): TypeCheckedSemanticKernelIr<T> {
  const issues = checkSemanticKernelIrTypes(verified);
  if (issues.length === 0) {
    return completeIrTypeChecking(Object.freeze({
      kind: "type-checked-semantic-kernel-ir",
      ir: verified.ir,
      verified,
      [typeCheckedSemanticKernelIrArtifact]: true as const,
    }));
  }
  const diagnostics: readonly CudaLiteDiagnostic[] = issues.map((issue) => ({
    code: issue.code,
    severity: "error",
    message: issue.message,
    span: issue.span,
  }));
  throw new CudaLiteCompilerError(`invalid semantic types: ${issues[0]!.message}`, diagnostics);
}

function checkOperations(
  operations: readonly SemanticKernelIrOperation[],
  fn: CudaLiteSemanticFunction | undefined,
  ir: SemanticKernelIrModule,
  report: (message: string, span: SourceSpan) => void,
): void {
  for (const operation of operations) {
    switch (operation.kind) {
      case "declare":
        if (operation.init) {
          checkExpression(operation.init, "value", ir, report);
          checkAssignable(operation.target.valueType, expressionValueType(operation.init), `initializer for '${operation.target.name}'`, operation.span, report);
        }
        break;
      case "dim3-declare":
        operation.args.forEach((arg) => checkExpression(arg, "value", ir, report));
        break;
      case "cooperative-group-declare":
        if (operation.declaration.partitionPredicate) checkExpression(operation.declaration.partitionPredicate, "value", ir, report);
        break;
      case "load":
        checkMemoryRef(operation.source, ir, report);
        break;
      case "store":
        checkMemoryRef(operation.target, ir, report);
        operation.reads.forEach((ref) => checkMemoryRef(ref, ir, report));
        checkExpression(operation.value, "value", ir, report);
        checkAssignable(operation.target.valueType, expressionValueType(operation.value), `store to '${operation.target.base}'`, operation.span, report, operation.operator !== "=");
        break;
      case "copy":
        checkMemoryRef(operation.source, ir, report);
        checkMemoryRef(operation.target, ir, report);
        break;
      case "matrix-fill":
        checkExpression(operation.value, "value", ir, report);
        break;
      case "matrix-load":
        checkMemoryRef(operation.source, ir, report);
        checkExpression(operation.stride, "value", ir, report);
        break;
      case "matrix-store":
        checkMemoryRef(operation.target, ir, report);
        checkExpression(operation.stride, "value", ir, report);
        break;
      case "surface-write":
        [operation.surface, operation.value, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : [])]
          .forEach((expression) => checkExpression(expression, "value", ir, report));
        break;
      case "surface-read-store":
        [operation.target, operation.surface, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : [])]
          .forEach((expression) => checkExpression(expression, "value", ir, report));
        break;
      case "atomic":
        if (operation.target) checkMemoryRef(operation.target, ir, report);
        operation.args.forEach((arg) => checkExpression(arg, "value", ir, report));
        break;
      case "call":
        operation.args.forEach((arg) => checkExpression(arg, arg.kind === "literal" && arg.literalKind === "string" ? "discard" : "value", ir, report));
        if (operation.result) checkExpression(operation.result, "value", ir, report);
        break;
      case "runtime-copy":
        operation.args.forEach((arg) => checkExpression(arg, "value", ir, report));
        break;
      case "pool-allocate":
        checkExpression(operation.sizeBytes, "value", ir, report);
        if (!operation.target.pointer || operation.target.addressSpace !== "local") {
          report(`pool allocation target '${operation.target.name}' must be a local pointer`, operation.span);
        }
        if (operation.pool.kind === "raw-pool") {
          checkMemoryRef(operation.pool.data, ir, report);
          checkMemoryRef(operation.pool.offset, ir, report);
          checkExpression(operation.pool.capacityBytes, "value", ir, report);
        }
        break;
      case "pointer-rebind":
        checkMemoryRef(operation.source, ir, report);
        if (!operation.target.pointer || operation.target.addressSpace !== "local") {
          report(`pointer rebind target '${operation.target.name}' must be a local pointer`, operation.span);
        }
        if (operation.source.addressSpace !== "storage") {
          report(`pointer rebind source '${operation.source.base}' must use storage memory`, operation.source.span);
        }
        checkAssignable(operation.target.valueType, operation.source.valueType, `pointer rebind '${operation.target.name}'`, operation.span, report);
        break;
      case "expression":
        checkExpression(operation.expression, "discard", ir, report);
        break;
      case "branch":
        checkCondition(operation.condition, ir, report);
        checkOperations(operation.consequent, fn, ir, report);
        checkOperations(operation.alternate, fn, ir, report);
        break;
      case "loop":
        if (operation.init) {
          if (isOperation(operation.init)) checkOperations([operation.init], fn, ir, report);
          else checkExpression(operation.init, "discard", ir, report);
        }
        if (operation.condition) checkCondition(operation.condition, ir, report);
        if (operation.update) checkExpression(operation.update, "discard", ir, report);
        checkOperations(operation.body, fn, ir, report);
        if (operation.continuing) checkOperations(operation.continuing, fn, ir, report);
        break;
      case "device-launch":
        [...operation.launch.grid, ...operation.launch.block, ...operation.launch.args]
          .forEach((expression) => checkExpression(expression, "value", ir, report));
        break;
      case "inline-asm":
        [...operation.outputs, ...operation.inputs].forEach((expression) => checkExpression(expression, "value", ir, report));
        break;
      case "return":
        if (operation.value) {
          checkExpression(operation.value, "value", ir, report);
          checkAssignable(fn?.returnType, expressionValueType(operation.value), `return from '${fn?.name ?? ir.name}'`, operation.span, report);
        }
        break;
      case "block":
        checkOperations(operation.body, fn, ir, report);
        break;
      default:
        break;
    }
  }
}

function checkCondition(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  report: (message: string, span: SourceSpan) => void,
): void {
  checkExpression(expression, "value", ir, report);
  const valueType = expressionValueType(expression);
  if (valueType !== undefined && (valueType === "void" || isVectorType(valueType))) {
    report(`condition has non-scalar type '${valueType}'`, expression.span);
  }
}

function checkExpression(
  expression: SemanticExpression,
  use: ExpressionUse,
  ir: SemanticKernelIrModule,
  report: (message: string, span: SourceSpan) => void,
): void {
  const nonScalarBuiltinValue = expression.kind === "symbol" && expression.addressSpace === "builtin" && expression.name === "cg::plus";
  if (use === "value" && !nonScalarBuiltinValue && expression.kind !== "initializer" && !(expression.kind === "literal" && expression.literalKind === "string")) {
    const valueType = expressionValueType(expression);
    if (valueType === undefined || valueType === "void") report(`${describeExpression(expression)} has no value type`, expression.span);
  }

  switch (expression.kind) {
    case "literal":
    case "symbol":
    case "pointer-valid":
      return;
    case "member":
      checkExpression(expression.object, "value", ir, report);
      return;
    case "index":
      checkExpression(expression.target, "value", ir, report);
      checkExpression(expression.index, "value", ir, report);
      return;
    case "call":
      checkExpression(expression.callee, "callee", ir, report);
      expression.args.forEach((arg) => checkExpression(arg, arg.kind === "literal" && arg.literalKind === "string" ? "discard" : "value", ir, report));
      checkResolvedFunctionCall(expression, ir, report);
      return;
    case "texture-read":
      [expression.texture, expression.x, expression.y, ...(expression.z ? [expression.z] : [])]
        .forEach((child) => checkExpression(child, "value", ir, report));
      return;
    case "surface-read":
      [expression.surface, expression.xBytes, expression.y, ...(expression.z ? [expression.z] : [])]
        .forEach((child) => checkExpression(child, "value", ir, report));
      return;
    case "cast":
      checkExpression(expression.expression, "value", ir, report);
      return;
    case "unary":
      checkExpression(expression.argument, "value", ir, report);
      return;
    case "binary":
      checkExpression(expression.left, "value", ir, report);
      checkExpression(expression.right, "value", ir, report);
      checkBinaryResultType(expression, report);
      return;
    case "conditional":
      checkCondition(expression.condition, ir, report);
      checkExpression(expression.consequent, "value", ir, report);
      checkExpression(expression.alternate, "value", ir, report);
      checkAssignable(expressionValueType(expression.consequent), expressionValueType(expression.alternate), "conditional branches", expression.span, report);
      return;
    case "assignment":
      checkExpression(expression.target, "value", ir, report);
      checkExpression(expression.value, "value", ir, report);
      checkAssignable(expressionValueType(expression.target), expressionValueType(expression.value), "assignment", expression.span, report, expression.operator !== "=");
      return;
    case "update":
      checkExpression(expression.argument, "value", ir, report);
      return;
    case "initializer":
      expression.elements.forEach((element) => checkExpression(element, "value", ir, report));
      return;
    case "sequence":
      expression.expressions.forEach((item, index) => checkExpression(item, index === expression.expressions.length - 1 ? use : "discard", ir, report));
  }
}

function checkResolvedFunctionCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  report: (message: string, span: SourceSpan) => void,
): void {
  if (expression.callee.kind !== "symbol" || expression.callee.addressSpace !== "function") return;
  const calleeId = semanticFunctionIdFromSymbol(expression.callee.id);
  const fn = ir.functions.find((candidate) => semanticIdsEqual(candidate.id, calleeId));
  if (fn === undefined) return;
  if (expression.args.length !== fn.params.length) {
    report(`call '${fn.name}' expects ${fn.params.length} arguments but received ${expression.args.length}`, expression.span);
    return;
  }
  checkAssignable(fn.returnType, expression.valueType, `return type of call '${fn.name}'`, expression.span, report);
  for (const [index, param] of fn.params.entries()) {
    if (param.pointer || param.cooperativeGroupKind !== undefined) continue;
    const arg = expression.args[index]!;
    checkAssignable(param.valueType, expressionValueType(arg), `argument ${index + 1} of call '${fn.name}'`, arg.span, report);
  }
}

function checkBinaryResultType(
  expression: Extract<SemanticExpression, { readonly kind: "binary" }>,
  report: (message: string, span: SourceSpan) => void,
): void {
  const actual = expressionValueType(expression);
  const expected = semanticBinaryResultType(
    expression.operator,
    expressionValueType(expression.left),
    expressionValueType(expression.right),
  );
  if (actual !== undefined && expected !== undefined && actual !== expected) {
    report(`binary '${expression.operator}' declares '${actual}', expected '${expected}' from its operands`, expression.span);
  }
}

function checkMemoryRef(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
  report: (message: string, span: SourceSpan) => void,
): void {
  if (!isSemanticValueType(ref.valueType)) report(`memory reference '${ref.base}' has no value type`, ref.span);
  ref.indices.forEach((index) => checkExpression(index, "value", ir, report));
}

function checkAssignable(
  target: CudaLiteScalarType | undefined,
  source: CudaLiteScalarType | undefined,
  owner: string,
  span: SourceSpan,
  report: (message: string, span: SourceSpan) => void,
  allowScalarSplat = false,
): void {
  if (target === undefined || source === undefined || target === "void" || source === "void") return;
  if (target === "voidptr" || source === "voidptr") return;
  if (isVectorType(target) !== isVectorType(source) && !(allowScalarSplat && isVectorType(target) && !isVectorType(source))) {
    report(`${owner} mixes scalar '${source}' and vector '${target}'`, span);
  }
}

function expressionValueType(expression: SemanticExpression): CudaLiteScalarType | undefined {
  return "valueType" in expression ? expression.valueType : undefined;
}

function describeExpression(expression: SemanticExpression): string {
  if (expression.kind === "call" && expression.callee.kind === "symbol") return `call '${expression.callee.name}'`;
  if (expression.kind === "unary") return `unary '${expression.operator}' expression`;
  if (expression.kind === "binary") return `binary '${expression.operator}' expression`;
  if (expression.kind === "symbol") return `symbol '${expression.name}'`;
  return `expression '${expression.kind}'`;
}

function isVectorType(valueType: CudaLiteScalarType): boolean {
  return valueType === "complex64" || /^(?:float|int|uint)(?:2|3|4)$/.test(valueType) || valueType === "half2" || valueType === "bf162";
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
