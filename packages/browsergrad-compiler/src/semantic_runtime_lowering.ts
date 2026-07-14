import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";
import {
  cudaIntegerRuntimeQueryTargetValueType,
  cudaIntegerRuntimeQueryValue,
  cudaRuntimeAvailableDynamicSmemBytes,
  cudaRuntimeMemInfoBytes,
  isCudaIntegerRuntimeQueryCall,
} from "./cuda_runtime_queries.js";
import { isHostManagedRuntimeNoopCall } from "./cuda_runtime_noops.js";
import { isCudaRuntimeCopyCall } from "./cuda_runtime_copies.js";
import type { SemanticExpression, SemanticKernelIrOperation, SemanticMemoryRef, CanonicalSemanticKernelIr } from "./semantic_ir_types.js";
import { semanticMemoryIdFromSymbol } from "./semantic_ids.js";
import type { CudaLiteScalarType, SourceSpan } from "./types.js";
import { requireSemanticValueType } from "./semantic_value_type.js";
import { completeRuntimeLowering, type RuntimeLoweredIr } from "./compiler_phases.js";

export type RuntimeLoweredSemanticKernelIr = RuntimeLoweredIr<CanonicalSemanticKernelIr>;

interface LoweredRuntimeExpression {
  readonly operations: readonly SemanticKernelIrOperation[];
  readonly expression: SemanticExpression;
}

export function lowerSemanticCudaRuntime(
  ir: CanonicalSemanticKernelIr,
): RuntimeLoweredSemanticKernelIr {
  const operations = lowerRuntimeOperations(ir.operations);
  const functions = ir.functions.map((fn) => {
    const body = lowerRuntimeOperations(fn.body);
    return body === fn.body ? fn : { ...fn, body };
  });
  const lowered = operations === ir.operations && functions.every((fn, index) => fn === ir.functions[index])
    ? ir
    : { ...ir, operations, functions };
  return completeRuntimeLowering(lowered);
}

function lowerRuntimeOperations(
  operations: readonly SemanticKernelIrOperation[],
): readonly SemanticKernelIrOperation[] {
  let changed = false;
  const lowered = operations.flatMap((operation): readonly SemanticKernelIrOperation[] => {
    const next = lowerRuntimeOperation(operation);
    if (next.length !== 1 || next[0] !== operation) changed = true;
    return next;
  });
  return changed ? lowered : operations;
}

function lowerRuntimeOperation(
  operation: SemanticKernelIrOperation,
): readonly SemanticKernelIrOperation[] {
  if (operation.kind === "call" && isCudaRuntimeCopyCall(operation.callee)) {
    return [{ kind: "runtime-copy", callee: operation.callee, args: operation.args, span: operation.span }];
  }
  if (operation.kind === "call" && isModeledRuntimeCall(operation.callee)) {
    const writes = runtimeCallWriteOperations(operation.callee, operation.args, operation.span);
    const result = operation.result === undefined ? [] : [assignmentOperation(operation.result, zeroLiteral("int", operation.span), operation.span)];
    return [...writes, ...result];
  }
  if (operation.kind === "declare" && operation.init) {
    const init = lowerRuntimeExpression(operation.init);
    return init.operations.length === 0 && init.expression === operation.init
      ? [operation]
      : [...init.operations, { ...operation, init: init.expression }];
  }
  if (operation.kind === "expression") {
    const expression = lowerRuntimeExpression(operation.expression);
    return expression.operations.length === 0 && expression.expression === operation.expression
      ? [operation]
      : [...expression.operations, { ...operation, expression: expression.expression }];
  }
  if (operation.kind === "store") {
    const value = lowerRuntimeExpression(operation.value);
    const target = lowerRuntimeMemoryRef(operation.target);
    return value.operations.length === 0 && target.operations.length === 0
      ? [operation]
      : [...target.operations, ...value.operations, { ...operation, target: target.ref, value: value.expression }];
  }
  if (operation.kind === "branch") {
    const condition = lowerRuntimeExpression(operation.condition);
    const consequent = lowerRuntimeOperations(operation.consequent);
    const alternate = lowerRuntimeOperations(operation.alternate);
    return condition.operations.length === 0 && consequent === operation.consequent && alternate === operation.alternate
      ? [operation]
      : [...condition.operations, { ...operation, condition: condition.expression, consequent, alternate }];
  }
  if (operation.kind === "block") {
    const body = lowerRuntimeOperations(operation.body);
    return body === operation.body ? [operation] : [{ ...operation, body }];
  }
  if (operation.kind === "loop") {
    const init = operation.init && !isOperation(operation.init) ? lowerRuntimeExpression(operation.init) : undefined;
    const condition = operation.condition ? lowerRuntimeExpression(operation.condition) : undefined;
    const update = operation.update ? lowerRuntimeExpression(operation.update) : undefined;
    const body = lowerRuntimeOperations(operation.body);
    const continuing = operation.continuing ? lowerRuntimeOperations(operation.continuing) : undefined;
    if ((condition?.operations.length ?? 0) > 0 || (update?.operations.length ?? 0) > 0) return [operation];
    const loweredInitOperation = operation.init && isOperation(operation.init) ? lowerRuntimeOperations([operation.init]) : undefined;
    const initPrefix = init?.operations ?? [];
    const loopInit = loweredInitOperation?.length === 1 ? loweredInitOperation[0] : init?.expression ?? operation.init;
    const next = {
      ...operation,
      ...(loopInit === undefined ? {} : { init: loopInit }),
      ...(condition === undefined ? {} : { condition: condition.expression }),
      ...(update === undefined ? {} : { update: update.expression }),
      body,
      ...(continuing === undefined ? {} : { continuing }),
    };
    return initPrefix.length === 0 && body === operation.body && continuing === operation.continuing && loopInit === operation.init
      ? [operation]
      : [...initPrefix, next];
  }
  return [operation];
}

function lowerRuntimeExpression(expression: SemanticExpression): LoweredRuntimeExpression {
  if (expression.kind === "call" && expression.callee.kind === "symbol" && isModeledRuntimeCall(expression.callee.name)) {
    return {
      operations: runtimeCallWriteOperations(expression.callee.name, expression.args, expression.span),
      expression: zeroLiteral(expression.valueType === "float" ? "float" : "int", expression.span),
    };
  }
  if (expression.kind === "member") {
    const object = lowerRuntimeExpression(expression.object);
    return { operations: object.operations, expression: object.expression === expression.object ? expression : { ...expression, object: object.expression } };
  }
  if (expression.kind === "cast") {
    const nested = lowerRuntimeExpression(expression.expression);
    return { operations: nested.operations, expression: nested.expression === expression.expression ? expression : { ...expression, expression: nested.expression } };
  }
  if (expression.kind === "unary" || expression.kind === "update") {
    const argument = lowerRuntimeExpression(expression.argument);
    return { operations: argument.operations, expression: argument.expression === expression.argument ? expression : { ...expression, argument: argument.expression } };
  }
  if (expression.kind === "index") {
    const target = lowerRuntimeExpression(expression.target);
    const index = lowerRuntimeExpression(expression.index);
    return {
      operations: [...target.operations, ...index.operations],
      expression: target.expression === expression.target && index.expression === expression.index
        ? expression
        : { ...expression, target: target.expression, index: index.expression },
    };
  }
  if (expression.kind === "binary") {
    const left = lowerRuntimeExpression(expression.left);
    const right = lowerRuntimeExpression(expression.right);
    return {
      operations: [...left.operations, ...right.operations],
      expression: left.expression === expression.left && right.expression === expression.right
        ? expression
        : { ...expression, left: left.expression, right: right.expression },
    };
  }
  if (expression.kind === "assignment") {
    const target = lowerRuntimeExpression(expression.target);
    const value = lowerRuntimeExpression(expression.value);
    return {
      operations: [...target.operations, ...value.operations],
      expression: target.expression === expression.target && value.expression === expression.value
        ? expression
        : { ...expression, target: target.expression, value: value.expression },
    };
  }
  if (expression.kind === "conditional") {
    const condition = lowerRuntimeExpression(expression.condition);
    const consequent = lowerRuntimeExpression(expression.consequent);
    const alternate = lowerRuntimeExpression(expression.alternate);
    if (consequent.operations.length > 0 || alternate.operations.length > 0) return { operations: [], expression };
    return {
      operations: condition.operations,
      expression: condition.expression === expression.condition && consequent.expression === expression.consequent && alternate.expression === expression.alternate
        ? expression
        : { ...expression, condition: condition.expression, consequent: consequent.expression, alternate: alternate.expression },
    };
  }
  if (expression.kind === "sequence" || expression.kind === "initializer") {
    const source = expression.kind === "sequence" ? expression.expressions : expression.elements;
    const parts = source.map(lowerRuntimeExpression);
    const values = parts.map((part) => part.expression);
    return {
      operations: parts.flatMap((part) => part.operations),
      expression: parts.every((part, index) => part.expression === source[index])
        ? expression
        : expression.kind === "sequence" ? { ...expression, expressions: values } : { ...expression, elements: values },
    };
  }
  return { operations: [], expression };
}

function runtimeCallWriteOperations(
  callee: string,
  args: readonly SemanticExpression[],
  span: SourceSpan,
): readonly SemanticKernelIrOperation[] {
  if (!isCudaIntegerRuntimeQueryCall(callee) && callee !== "cudaEventElapsedTime") return [];
  const writes: Array<{ readonly target: SemanticExpression | undefined; readonly value: number; readonly valueType: Exclude<CudaLiteScalarType, "void"> }> = [];
  const add = (target: SemanticExpression | undefined, value: number, valueType: Exclude<CudaLiteScalarType, "void"> = "int"): void => {
    if (!semanticRuntimeNullPointer(target)) writes.push({ target, value, valueType });
  };
  if (callee === "cudaEventElapsedTime") add(args[0], 0, "float");
  else if (callee === "cudaMemGetInfo") {
    add(args[0], cudaRuntimeMemInfoBytes(), "uint");
    add(args[1], cudaRuntimeMemInfoBytes(), "uint");
  } else if (callee === "cudaOccupancyMaxPotentialBlockSize" || callee === "cudaOccupancyMaxPotentialBlockSizeWithFlags") {
    add(args[0], 1);
    add(args[1], 256);
  } else if (callee === "cudaOccupancyAvailableDynamicSMemPerBlock") add(args[0], cudaRuntimeAvailableDynamicSmemBytes(), "uint");
  else if (callee === "cudaDeviceGetStreamPriorityRange") {
    add(args[0], 0);
    add(args[1], 0);
  } else if (callee === "cudaStreamGetCaptureInfo" || callee === "cudaStreamGetCaptureInfo_v2") {
    args.slice(1).forEach((target, index) => add(target, 0, index === 0 ? "int" : "uint"));
  } else if (callee === "cudaGraphInstantiate") {
    add(args[0], 0, "uint");
    add(args[2], 0, "uint");
  } else if (callee === "cudaGraphExecUpdate") {
    add(args[2], 0, "uint");
    add(args[3], 0);
  } else {
    const targetIndex = callee === "cudaStreamGetFlags" || callee === "cudaStreamGetPriority" || callee === "cudaStreamGetDevice" ||
        callee === "cudaStreamGetId" || callee === "cudaStreamIsCapturing" || callee === "cudaStreamEndCapture"
      ? 1
      : 0;
    add(args[targetIndex], cudaIntegerRuntimeQueryValue(callee, (index) => semanticConstantInteger(args[index])), cudaIntegerRuntimeQueryTargetValueType(callee));
  }
  return writes.flatMap((write) => semanticRuntimeWrite(write.target!, write.value, write.valueType, span));
}

function semanticRuntimeWrite(
  pointer: SemanticExpression,
  value: number,
  valueType: Exclude<CudaLiteScalarType, "void">,
  span: SourceSpan,
): readonly SemanticKernelIrOperation[] {
  const target = semanticRuntimePointerTarget(pointer);
  const literal = zeroLiteral(valueType, span, value);
  if (!target) return [];
  if (target.kind === "symbol" && target.addressSpace === "local") return [assignmentOperation(target, literal, span)];
  const ref = semanticRuntimeMemoryRef(target);
  return ref ? [{ kind: "store", target: ref, value: literal, operator: "=", reads: [], span }] : [];
}

function semanticRuntimePointerTarget(pointer: SemanticExpression): SemanticExpression | undefined {
  if (pointer.kind === "unary" && pointer.operator === "&") return pointer.argument;
  if (pointer.kind === "cast" && pointer.pointer) return semanticRuntimePointerTarget(pointer.expression);
  if (pointer.kind === "index" || pointer.kind === "symbol") return pointer;
  return undefined;
}

function semanticRuntimeMemoryRef(expression: SemanticExpression): SemanticMemoryRef | undefined {
  const indices: SemanticExpression[] = [];
  let target = expression;
  while (target.kind === "index") {
    indices.unshift(target.index);
    target = target.target;
  }
  if (target.kind !== "symbol" || target.addressSpace === "local" && indices.length === 0) return undefined;
  const valueType = semanticRuntimeExpressionValueType(expression);
  return {
    baseId: semanticMemoryIdFromSymbol(target.id),
    base: target.name,
    addressSpace: target.addressSpace,
    valueType: requireSemanticValueType(valueType, `runtime memory '${target.name}'`, expression.span),
    indices,
    fields: [],
    span: expression.span,
  };
}

function assignmentOperation(
  target: Extract<SemanticExpression, { readonly kind: "symbol" }>,
  value: SemanticExpression,
  span: SourceSpan,
): SemanticKernelIrOperation {
  return {
    kind: "expression",
    expression: { kind: "assignment", operator: "=", target, value, ...(target.valueType === undefined ? {} : { valueType: target.valueType }), span },
    span,
  };
}

function semanticRuntimeExpressionValueType(expression: SemanticExpression): CudaLiteScalarType | undefined {
  return "valueType" in expression ? expression.valueType : undefined;
}

function zeroLiteral(
  valueType: Exclude<CudaLiteScalarType, "void">,
  span: SourceSpan,
  value = 0,
): SemanticExpression {
  return { kind: "literal", literalKind: "number", value, valueType, span };
}

function semanticConstantInteger(expression: SemanticExpression | undefined): number | undefined {
  if (!expression) return undefined;
  if (expression.kind === "literal" && typeof expression.value === "number") return Math.trunc(expression.value);
  if (expression.kind === "symbol") return CUDA_NAMED_CONSTANTS.get(expression.name)?.value;
  if (expression.kind === "cast") return semanticConstantInteger(expression.expression);
  return undefined;
}

function semanticRuntimeNullPointer(expression: SemanticExpression | undefined): boolean {
  return expression === undefined || expression.kind === "literal" && expression.value === 0 ||
    expression.kind === "symbol" && (expression.name === "NULL" || expression.name === "nullptr");
}

function isModeledRuntimeCall(name: string): boolean {
  return isHostManagedRuntimeNoopCall(name) || isCudaIntegerRuntimeQueryCall(name) || name === "cudaEventElapsedTime";
}

function lowerRuntimeMemoryRef(ref: SemanticMemoryRef): { readonly operations: readonly SemanticKernelIrOperation[]; readonly ref: SemanticMemoryRef } {
  const indices = ref.indices.map(lowerRuntimeExpression);
  return {
    operations: indices.flatMap((index) => index.operations),
    ref: indices.every((index, position) => index.expression === ref.indices[position])
      ? ref
      : { ...ref, indices: indices.map((index) => index.expression) },
  };
}

function isOperation(value: SemanticKernelIrOperation | SemanticExpression): value is SemanticKernelIrOperation {
  return value.kind === "call" ? "reads" in value :
    value.kind === "declare" || value.kind === "dim3-declare" || value.kind === "cooperative-group-declare" ||
    value.kind === "load" || value.kind === "store" || value.kind === "copy" || value.kind === "copy-fence" ||
    value.kind === "matrix-fill" || value.kind === "matrix-load" || value.kind === "matrix-mma" || value.kind === "matrix-store" ||
    value.kind === "surface-write" || value.kind === "surface-read-store" || value.kind === "atomic" || value.kind === "expression" ||
    value.kind === "branch" || value.kind === "loop" || value.kind === "barrier" || value.kind === "fence" ||
    value.kind === "device-launch" || value.kind === "inline-asm" || value.kind === "return" || value.kind === "continue" ||
    value.kind === "break" || value.kind === "block";
}
