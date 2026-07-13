import type { WgslValueType } from "@unlocalhosted/browsergrad-kernels";
import type {
  CudaLiteSemanticSymbol,
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
} from "./semantic_ir.js";
import { semanticIdsEqual } from "./semantic_ids.js";
import { isSemanticKernelIrOperation, semanticExpressionChildren } from "./semantic_ir_walk.js";
import { SEMANTIC_SUBGROUP_CALLS } from "./semantic_builtin_calls.js";
import { semanticAssignmentBinaryOperator } from "./semantic_expression_contracts.js";
import { wgslValueType } from "./semantic_wgsl_types.js";
import {
  createTypedWgslIdentifier,
  emitTypedWgslSelect,
  type TypedWgslExpression,
} from "./typed_wgsl_expression.js";
import { createTypedWgslLocalAssignmentStatement } from "./typed_wgsl_statement.js";
import type { SourceSpan } from "./types.js";

type SemanticLoopOperation = Extract<SemanticKernelIrOperation, { readonly kind: "loop" }>;
type SemanticLocalMutation =
  | Extract<SemanticExpression, { readonly kind: "assignment" }>
  | Extract<SemanticExpression, { readonly kind: "update" }>;

interface SemanticSubgroupControlOptions {
  readonly activeCollectivePredicate?: SemanticExpression;
}

interface SemanticSubgroupControlDependencies<
  Options extends SemanticSubgroupControlOptions,
  TextureSpecializations,
> {
  readonly createGeneratedSymbolId: (name: string, span: SourceSpan) => CudaLiteSemanticSymbol["id"];
  readonly emitExpressionAs: (
    expression: SemanticExpression,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    targetType: WgslValueType,
    options: Options,
    textureSpecializations: TextureSpecializations,
  ) => TypedWgslExpression;
  readonly emitLoopInit: (
    init: SemanticKernelIrOperation | SemanticExpression,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: Options,
    textureSpecializations: TextureSpecializations,
  ) => string;
  readonly emitOperation: (
    operation: SemanticKernelIrOperation,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    indentLevel: number,
    allowReturnValue: boolean,
    options: Options,
    textureSpecializations: TextureSpecializations,
  ) => readonly string[];
  readonly emitTruthiness: (
    expression: SemanticExpression,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: Options,
  ) => string;
  readonly nameFor: (name: string, names: ReadonlyMap<string, string>) => string;
  readonly semanticError: (message: string, span: SourceSpan) => Error;
}

export interface SemanticSubgroupControlEmitter<
  Options extends SemanticSubgroupControlOptions,
  TextureSpecializations,
> {
  readonly emitUniformForLoop: (
    operation: SemanticLoopOperation,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    indentLevel: number,
    allowReturnValue: boolean,
    options: Options,
    textureSpecializations: TextureSpecializations,
  ) => readonly string[];
  readonly operationsContainNativeCollective: (operations: readonly SemanticKernelIrOperation[]) => boolean;
}

export function createSemanticSubgroupControlEmitter<
  Options extends SemanticSubgroupControlOptions,
  TextureSpecializations,
>(
  dependencies: SemanticSubgroupControlDependencies<Options, TextureSpecializations>,
): SemanticSubgroupControlEmitter<Options, TextureSpecializations> {
  const {
    createGeneratedSymbolId,
    emitExpressionAs,
    emitLoopInit,
    emitOperation,
    emitTruthiness,
    nameFor,
    semanticError,
  } = dependencies;

  const expressionContainsNativeCollective = (expression: SemanticExpression): boolean => {
    if (expression.kind === "call" && expression.callee.kind === "symbol" &&
      expression.callee.addressSpace !== "function" && SEMANTIC_SUBGROUP_CALLS.has(expression.callee.name)) return true;
    return semanticExpressionChildren(expression).some(expressionContainsNativeCollective);
  };

  const operationsContainNativeCollective = (operations: readonly SemanticKernelIrOperation[]): boolean =>
    operations.some((operation) => {
      if (operation.kind === "declare") return operation.init !== undefined && expressionContainsNativeCollective(operation.init);
      if (operation.kind === "store") return expressionContainsNativeCollective(operation.value) || operation.target.indices.some(expressionContainsNativeCollective);
      if (operation.kind === "atomic" || operation.kind === "call") return operation.args.some(expressionContainsNativeCollective);
      if (operation.kind === "expression") return expressionContainsNativeCollective(operation.expression);
      if (operation.kind === "branch") return expressionContainsNativeCollective(operation.condition) ||
        operationsContainNativeCollective(operation.consequent) || operationsContainNativeCollective(operation.alternate);
      if (operation.kind === "loop") return (operation.init !== undefined && !isSemanticKernelIrOperation(operation.init) && expressionContainsNativeCollective(operation.init)) ||
        (operation.condition !== undefined && expressionContainsNativeCollective(operation.condition)) ||
        (operation.update !== undefined && expressionContainsNativeCollective(operation.update)) ||
        operationsContainNativeCollective(operation.body) ||
        (operation.continuing !== undefined && operationsContainNativeCollective(operation.continuing));
      if (operation.kind === "block") return operationsContainNativeCollective(operation.body);
      if (operation.kind === "return") return operation.value !== undefined && expressionContainsNativeCollective(operation.value);
      return false;
    });

  const emitBranchlessLocalMutation = (
    expression: SemanticLocalMutation,
    predicate: string,
    predicateExpression: SemanticExpression,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: Options,
    textureSpecializations: TextureSpecializations,
  ): string => {
    const target = expression.kind === "assignment" ? expression.target : expression.argument;
    if (target.kind !== "symbol" || target.addressSpace !== "local" || !target.valueType || target.valueType === "void") {
      throw semanticError("native subgroup loops require typed local mutation targets", expression.span);
    }
    const semanticTargetType = wgslValueType(target.valueType);
    if (semanticTargetType === "bool" || semanticTargetType.startsWith("vec") && semanticTargetType.endsWith("<bool>")) {
      throw semanticError("native subgroup loops require numeric local mutation targets", expression.span);
    }
    const targetType = semanticTargetType as WgslValueType;
    const current = createTypedWgslIdentifier(nameFor(target.name, names), targetType, target.span);
    let nextExpression: SemanticExpression;
    if (expression.kind === "update") {
      nextExpression = {
        kind: "binary",
        operator: expression.operator === "++" ? "+" : "-",
        left: target,
        right: { kind: "literal", literalKind: "number", value: 1, valueType: target.valueType, span: expression.span },
        valueType: target.valueType,
        span: expression.span,
      };
    } else {
      const binaryOperator = semanticAssignmentBinaryOperator(expression.operator);
      nextExpression = binaryOperator === undefined
        ? expression.value
        : {
            kind: "binary",
            operator: binaryOperator,
            left: target,
            right: expression.value,
            valueType: target.valueType,
            span: expression.span,
          };
    }
    const collectiveOptions = expressionContainsNativeCollective(nextExpression)
      ? { ...options, activeCollectivePredicate: predicateExpression }
      : options;
    const next = emitExpressionAs(nextExpression, ir, names, targetType, collectiveOptions, textureSpecializations);
    const condition = createTypedWgslIdentifier(predicate, "bool", expression.span);
    const selected = emitTypedWgslSelect(current, next, condition, expression.span);
    return createTypedWgslLocalAssignmentStatement(
      nameFor(target.name, names),
      targetType,
      "=",
      selected,
      expression.span,
    ).code.slice(0, -1);
  };

  const emitBranchlessOperations = (
    operations: readonly SemanticKernelIrOperation[],
    predicate: string,
    predicateExpression: SemanticExpression,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    indentLevel: number,
    allowReturnValue: boolean,
    options: Options,
    textureSpecializations: TextureSpecializations,
  ): readonly string[] => {
    const lines: string[] = [];
    const prefix = "  ".repeat(indentLevel);
    for (const operation of operations) {
      if (operation.kind === "block") {
        lines.push(`${prefix}{`);
        lines.push(...emitBranchlessOperations(operation.body, predicate, predicateExpression, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
        lines.push(`${prefix}}`);
        continue;
      }
      if (operation.kind === "branch") {
        const condition = emitTruthiness(operation.condition, ir, names, options);
        lines.push(`${prefix}{`);
        lines.push(...emitBranchlessOperations(operation.consequent, `(${predicate}) && (${condition})`, collectiveAnd(predicateExpression, operation.condition), ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
        lines.push(`${prefix}}`);
        lines.push(`${prefix}{`);
        lines.push(...emitBranchlessOperations(operation.alternate, `(${predicate}) && !(${condition})`, collectiveAnd(predicateExpression, collectiveNot(operation.condition)), ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
        lines.push(`${prefix}}`);
        continue;
      }
      if (operation.kind === "declare" && operation.target.addressSpace === "local" &&
        !operation.target.pointer && operation.target.dimensions.length === 0) {
        lines.push(...emitOperation(
          operation,
          ir,
          names,
          indentLevel,
          allowReturnValue,
          operation.init && expressionContainsNativeCollective(operation.init)
            ? { ...options, activeCollectivePredicate: predicateExpression }
            : options,
          textureSpecializations,
        ));
        continue;
      }
      if (operation.kind === "expression" &&
        (operation.expression.kind === "assignment" || operation.expression.kind === "update")) {
        lines.push(`${prefix}${emitBranchlessLocalMutation(operation.expression, predicate, predicateExpression, ir, names, options, textureSpecializations)};`);
        continue;
      }
      throw semanticError("native subgroup loops require branchless local-state operations", operation.span);
    }
    return lines;
  };

  const emitUniformForLoop = (
    operation: SemanticLoopOperation,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    indentLevel: number,
    allowReturnValue: boolean,
    options: Options,
    textureSpecializations: TextureSpecializations,
  ): readonly string[] => {
    const prefix = "  ".repeat(indentLevel);
    const bodyPrefix = "  ".repeat(indentLevel + 1);
    const loopPrefix = "  ".repeat(indentLevel + 2);
    const activeSourceName = `bg_subgroup_loop_active_${operation.span.start}`;
    const activeName = nameFor(activeSourceName, names);
    const activeExpression: SemanticExpression = {
      kind: "symbol",
      id: createGeneratedSymbolId(activeSourceName, operation.span),
      name: activeSourceName,
      valueType: "bool",
      addressSpace: "local",
      span: operation.span,
    };
    const init = operation.init ? emitLoopInit(operation.init, ir, names, options, textureSpecializations) : "";
    const condition = operation.condition ? emitTruthiness(operation.condition, ir, names, options) : "true";
    const update = operation.update;
    if (update !== undefined && update.kind !== "assignment" && update.kind !== "update") {
      throw semanticError("native subgroup loops require local assignment or increment updates", update.span);
    }
    const iterationBudget = emitUniformIterationBudget(operation, ir, names, options, textureSpecializations, dependencies);
    const iterationName = nameFor(`bg_subgroup_loop_iteration_${operation.span.start}`, names);
    return [
      `${prefix}{`,
      ...(init === "" ? [] : [`${bodyPrefix}${init};`]),
      `${bodyPrefix}var ${activeName}: bool = ${condition};`,
      `${bodyPrefix}for (var ${iterationName}: u32 = 0u; ${iterationName} < ${iterationBudget}; ${iterationName} += 1u) {`,
      ...emitBranchlessOperations(operation.body, activeName, activeExpression, ir, names, indentLevel + 2, allowReturnValue, options, textureSpecializations),
      ...(update === undefined ? [] : [
        `${loopPrefix}${emitBranchlessLocalMutation(update, activeName, activeExpression, ir, names, options, textureSpecializations)};`,
      ]),
      `${loopPrefix}${activeName} = ${activeName} && (${condition});`,
      `${bodyPrefix}}`,
      `${prefix}}`,
    ];
  };

  return { emitUniformForLoop, operationsContainNativeCollective };
}

function emitUniformIterationBudget<
  Options extends SemanticSubgroupControlOptions,
  TextureSpecializations,
>(
  operation: SemanticLoopOperation,
  ir: SemanticKernelIrModule,
  names: ReadonlyMap<string, string>,
  options: Options,
  textureSpecializations: TextureSpecializations,
  dependencies: SemanticSubgroupControlDependencies<Options, TextureSpecializations>,
): string {
  if (operation.init?.kind !== "declare" || operation.init.target.addressSpace !== "local" ||
    operation.condition?.kind !== "binary" || (operation.condition.operator !== "<" && operation.condition.operator !== "<=") ||
    operation.condition.left.kind !== "symbol" || !semanticIdsEqual(operation.condition.left.id, operation.init.target.id)) {
    throw dependencies.semanticError("native subgroup loops require a canonical increasing integer bound", operation.span);
  }
  const update = operation.update;
  const progresses = update?.kind === "update"
    ? update.operator === "++" && update.argument.kind === "symbol" && semanticIdsEqual(update.argument.id, operation.init.target.id)
    : update?.kind === "assignment" && update.operator === "+=" && update.target.kind === "symbol" &&
      semanticIdsEqual(update.target.id, operation.init.target.id) && update.value.kind === "literal" &&
      update.value.literalKind === "number" && update.value.value > 0;
  if (!progresses) {
    throw dependencies.semanticError("native subgroup loops require a compile-time positive counter step", update?.span ?? operation.span);
  }
  const initialRange = operation.init.init ? staticIntegerRange(operation.init.init, ir) : { min: 0, max: 0 };
  if (!initialRange) throw dependencies.semanticError("native subgroup loop initializer has no provable integer range", operation.init.span);
  const inclusive = operation.condition.operator === "<=" ? 1 : 0;
  const bound = operation.condition.right;
  const staticBound = staticIntegerRange(bound, ir);
  if (staticBound) return `${Math.max(0, Math.trunc(staticBound.max - initialRange.min + inclusive))}u`;
  if (!expressionIsWorkgroupUniform(bound)) {
    throw dependencies.semanticError("native subgroup loop bound must be uniform or statically bounded", bound.span);
  }
  const emittedBound = dependencies.emitExpressionAs(bound, ir, names, "i32", options, textureSpecializations).code;
  const adjustment = Math.trunc(-initialRange.min + inclusive);
  const adjusted = adjustment === 0 ? emittedBound : `(${emittedBound} + ${adjustment})`;
  return `u32(max(${adjusted}, 0))`;
}

interface SemanticIntegerRange {
  readonly min: number;
  readonly max: number;
}

function staticIntegerRange(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  visiting: ReadonlySet<string> = new Set(),
): SemanticIntegerRange | undefined {
  if (expression.kind === "literal" && expression.literalKind === "number" && Number.isFinite(expression.value)) {
    const value = Math.trunc(expression.value);
    return { min: value, max: value };
  }
  if (expression.kind === "cast") return staticIntegerRange(expression.expression, ir, visiting);
  if (expression.kind === "member" && expression.object.kind === "symbol" && expression.object.name === "threadIdx") {
    const axis = expression.property === "x" ? 0 : expression.property === "y" ? 1 : expression.property === "z" ? 2 : undefined;
    if (axis !== undefined) return { min: 0, max: Math.max(0, ir.workgroupSize[axis] - 1) };
  }
  if (expression.kind === "symbol" && expression.addressSpace === "local") {
    if (visiting.has(expression.id.key)) return undefined;
    const declaration = findLocalDeclaration(ir.operations, expression.id);
    if (!declaration?.init) return undefined;
    return staticIntegerRange(declaration.init, ir, new Set([...visiting, expression.id.key]));
  }
  if (expression.kind === "unary" && (expression.operator === "+" || expression.operator === "-")) {
    const value = staticIntegerRange(expression.argument, ir, visiting);
    if (!value) return undefined;
    return expression.operator === "+" ? value : { min: -value.max, max: -value.min };
  }
  if (expression.kind !== "binary") return undefined;
  const left = staticIntegerRange(expression.left, ir, visiting);
  const right = staticIntegerRange(expression.right, ir, visiting);
  if (!left || !right) return undefined;
  if (expression.operator === "+") return { min: left.min + right.min, max: left.max + right.max };
  if (expression.operator === "-") return { min: left.min - right.max, max: left.max - right.min };
  if (expression.operator === "*") {
    const values = [left.min * right.min, left.min * right.max, left.max * right.min, left.max * right.max];
    return { min: Math.min(...values), max: Math.max(...values) };
  }
  if (expression.operator === "/" && right.min === right.max && right.min > 0) {
    return { min: Math.trunc(left.min / right.min), max: Math.trunc(left.max / right.min) };
  }
  if (expression.operator === "%" && right.min === right.max && right.min > 0 && left.min >= 0) {
    return { min: 0, max: right.min - 1 };
  }
  return undefined;
}

function findLocalDeclaration(
  operations: readonly SemanticKernelIrOperation[],
  id: CudaLiteSemanticSymbol["id"],
): Extract<SemanticKernelIrOperation, { readonly kind: "declare" }> | undefined {
  for (const operation of operations) {
    if (operation.kind === "declare" && semanticIdsEqual(operation.target.id, id)) return operation;
    if (operation.kind === "branch") {
      const nested = findLocalDeclaration([...operation.consequent, ...operation.alternate], id);
      if (nested) return nested;
    }
    if (operation.kind === "loop") {
      const nested = findLocalDeclaration([
        ...(operation.init && isSemanticKernelIrOperation(operation.init) ? [operation.init] : []),
        ...operation.body,
        ...(operation.continuing ?? []),
      ], id);
      if (nested) return nested;
    }
    if (operation.kind === "block") {
      const nested = findLocalDeclaration(operation.body, id);
      if (nested) return nested;
    }
  }
  return undefined;
}

function expressionIsWorkgroupUniform(expression: SemanticExpression): boolean {
  if (expression.kind === "literal") return true;
  if (expression.kind === "symbol") return expression.addressSpace === "uniform" || expression.addressSpace === "constant";
  if (expression.kind === "cast") return expressionIsWorkgroupUniform(expression.expression);
  if (expression.kind === "unary") return expressionIsWorkgroupUniform(expression.argument);
  if (expression.kind === "binary") return expressionIsWorkgroupUniform(expression.left) && expressionIsWorkgroupUniform(expression.right);
  if (expression.kind === "conditional") return expressionIsWorkgroupUniform(expression.condition) &&
    expressionIsWorkgroupUniform(expression.consequent) && expressionIsWorkgroupUniform(expression.alternate);
  return false;
}

function collectiveNot(expression: SemanticExpression): SemanticExpression {
  return { kind: "unary", operator: "!", argument: expression, valueType: "bool", span: expression.span };
}

function collectiveAnd(left: SemanticExpression, right: SemanticExpression): SemanticExpression {
  return { kind: "binary", operator: "&&", left, right, valueType: "bool", span: { ...left.span, end: right.span.end } };
}
