import type {
  CudaLiteCallExpression,
  CudaLiteDeviceGlobal,
  CudaLiteExpression,
  CudaLiteScalarType,
  CudaLiteStatement,
} from "./types.js";
import type {
  CudaLiteSemanticSymbol,
  SemanticAddressSpace,
  SemanticExpression,
  SemanticKernelIrOperation,
} from "./semantic_ir_types.js";

/**
 * Adapts the small host-orchestration language from semantic IR to the
 * established host-copy evaluator. This deliberately does not read parsed or
 * analyzed statements: runtime planning can therefore survive a detached
 * public compile result whose semantic Kernel IR is intact.
 */
export function semanticRuntimeOperationsToHostStatements(
  operations: readonly SemanticKernelIrOperation[],
): readonly CudaLiteStatement[] {
  return operations.flatMap(semanticRuntimeOperationToHostStatements);
}

/**
 * Host-copy planning needs default device-global storage shape when a caller
 * has not supplied a backing typed array. Those facts belong to semantic
 * symbols, not the analyzer's private AST-shaped summary.
 */
export function semanticRuntimeDeviceGlobals(
  symbols: readonly CudaLiteSemanticSymbol[],
): readonly CudaLiteDeviceGlobal[] {
  return symbols
    .filter((symbol): symbol is CudaLiteSemanticSymbol & { readonly valueType: Exclude<CudaLiteScalarType, "void"> } =>
      symbol.kind === "device-global" && isConcreteCudaValueType(symbol.valueType),
    )
    .map((symbol) => ({
      kind: "device-global" as const,
      valueType: symbol.valueType!,
      name: symbol.name,
      dimensions: symbol.dimensions,
      span: symbol.span,
    }));
}

function semanticRuntimeOperationToHostStatements(
  operation: SemanticKernelIrOperation,
): readonly CudaLiteStatement[] {
  switch (operation.kind) {
    case "dim3-declare":
      return [{
        kind: "dim3",
        name: operation.target.name,
        args: operation.args.map(semanticExpressionToHostExpression),
        span: operation.span,
      }];
    case "declare":
      if (operation.target.addressSpace !== "local" || !isConcreteCudaValueType(operation.target.valueType)) return [hostSideEffect(operation.span)];
      return [{
        kind: "var",
        storage: "local",
        valueType: operation.target.valueType,
        pointer: operation.target.pointer === true,
        name: operation.target.name,
        dimensions: operation.target.dimensions,
        ...(operation.init === undefined ? {} : { init: semanticExpressionToHostExpression(operation.init) }),
        span: operation.span,
      }];
    case "runtime-copy":
      return [{ kind: "expr", expression: semanticRuntimeCall(operation.callee, operation.args, operation.span), span: operation.span }];
    case "call":
      return [{ kind: "expr", expression: semanticRuntimeCall(operation.callee, operation.args, operation.span), span: operation.span }];
    case "expression":
      return [{ kind: "expr", expression: semanticExpressionToHostExpression(operation.expression), span: operation.span }];
    case "branch":
      return [{
        kind: "if",
        condition: semanticExpressionToHostExpression(operation.condition),
        consequent: semanticRuntimeOperationsToHostStatements(operation.consequent),
        ...(operation.alternate.length === 0 ? {} : { alternate: semanticRuntimeOperationsToHostStatements(operation.alternate) }),
        span: operation.span,
      }];
    case "block":
      return semanticRuntimeOperationsToHostStatements(operation.body);
    case "loop":
    case "load":
    case "store":
    case "copy":
    case "matrix-fill":
    case "matrix-load":
    case "matrix-mma":
    case "matrix-store":
    case "surface-write":
    case "surface-read-store":
    case "atomic":
    case "pool-allocate":
    case "pointer-rebind":
    case "pointer-array-rebind":
    case "barrier":
    case "fence":
    case "inline-asm":
      return [hostSideEffect(operation.span)];
    case "copy-fence":
    case "cooperative-group-declare":
      return [];
    case "device-launch":
      return [{
        kind: "kernel-launch",
        callee: operation.launch.callee,
        grid: operation.launch.grid.map(semanticExpressionToHostExpression),
        block: operation.launch.block.map(semanticExpressionToHostExpression),
        args: operation.launch.args.map(semanticExpressionToHostExpression),
        span: operation.span,
      }];
    case "return":
      return [{
        kind: "return",
        ...(operation.value === undefined ? {} : { value: semanticExpressionToHostExpression(operation.value) }),
        span: operation.span,
      }];
    case "continue":
    case "break":
      return [{ kind: operation.kind, span: operation.span }];
  }
}

function semanticRuntimeCall(
  callee: string,
  args: readonly SemanticExpression[],
  span: SemanticExpression["span"],
): CudaLiteCallExpression {
  return {
    kind: "call",
    callee: { kind: "identifier", name: callee, span },
    args: args.map(semanticExpressionToHostExpression),
    span,
  };
}

function semanticExpressionToHostExpression(expression: SemanticExpression): CudaLiteExpression {
  switch (expression.kind) {
    case "literal":
      return expression.literalKind === "number"
        ? { kind: "number", value: expression.value, raw: String(expression.value), span: expression.span }
        : { kind: "string", value: expression.value, raw: JSON.stringify(expression.value), span: expression.span };
    case "symbol":
      return { kind: "identifier", name: expression.name, span: expression.span };
    case "pointer-valid":
      return { kind: "identifier", name: expression.pointer, span: expression.span };
    case "member":
      return {
        kind: "member",
        object: semanticExpressionToHostExpression(expression.object),
        property: expression.property,
        span: expression.span,
      };
    case "index":
      if (isHostStorageAddressSpace(expression.addressSpace) && expression.target.kind === "symbol") {
        return {
          kind: "binary",
          operator: "+",
          left: semanticExpressionToHostExpression(expression.target),
          right: semanticExpressionToHostExpression(expression.index),
          span: expression.span,
        };
      }
      return {
        kind: "index",
        target: semanticExpressionToHostExpression(expression.target),
        index: semanticExpressionToHostExpression(expression.index),
        span: expression.span,
      };
    case "call":
      return {
        kind: "call",
        callee: semanticExpressionToHostExpression(expression.callee),
        args: expression.args.map(semanticExpressionToHostExpression),
        ...(expression.templateValueType === undefined ? {} : { templateValueType: expression.templateValueType }),
        span: expression.span,
      };
    case "cast":
      return {
        kind: "cast",
        valueType: expression.valueType,
        pointer: expression.pointer,
        ...(expression.packedByteLanes === undefined ? {} : { packedByteLanes: expression.packedByteLanes }),
        expression: semanticExpressionToHostExpression(expression.expression),
        span: expression.span,
      };
    case "unary":
      return {
        kind: "unary",
        operator: expression.operator as Extract<CudaLiteExpression, { readonly kind: "unary" }>["operator"],
        argument: semanticExpressionToHostExpression(expression.argument),
        span: expression.span,
      };
    case "binary":
      return {
        kind: "binary",
        operator: expression.operator as Extract<CudaLiteExpression, { readonly kind: "binary" }>["operator"],
        left: semanticExpressionToHostExpression(expression.left),
        right: semanticExpressionToHostExpression(expression.right),
        span: expression.span,
      };
    case "conditional":
      return {
        kind: "conditional",
        condition: semanticExpressionToHostExpression(expression.condition),
        consequent: semanticExpressionToHostExpression(expression.consequent),
        alternate: semanticExpressionToHostExpression(expression.alternate),
        span: expression.span,
      };
    case "initializer":
      return { kind: "initializer", elements: expression.elements.map(semanticExpressionToHostExpression), span: expression.span };
    case "sequence":
      return { kind: "sequence", expressions: expression.expressions.map(semanticExpressionToHostExpression), span: expression.span };
    case "assignment":
      return {
        kind: "assignment",
        operator: expression.operator as Extract<CudaLiteExpression, { readonly kind: "assignment" }>["operator"],
        left: semanticExpressionToHostExpression(expression.target),
        right: semanticExpressionToHostExpression(expression.value),
        span: expression.span,
      };
    case "update":
      return {
        kind: "update",
        operator: expression.operator as Extract<CudaLiteExpression, { readonly kind: "update" }>["operator"],
        argument: semanticExpressionToHostExpression(expression.argument),
        prefix: expression.prefix,
        span: expression.span,
      };
    case "texture-read":
    case "surface-read":
      return { kind: "identifier", name: "__bg_runtime_value", span: expression.span };
  }
}

function hostSideEffect(span: SemanticExpression["span"]): CudaLiteStatement {
  return { kind: "asm", template: "", inputs: [], span };
}

function isConcreteCudaValueType(
  valueType: CudaLiteScalarType | undefined,
): valueType is Exclude<CudaLiteScalarType, "void"> {
  return valueType !== undefined && valueType !== "void";
}

function isHostStorageAddressSpace(addressSpace: SemanticAddressSpace): boolean {
  return addressSpace === "storage" || addressSpace === "constant" || addressSpace === "device-global";
}
