import type { WgslValueType } from "@unlocalhosted/browsergrad-kernels";
import type { CudaLiteSemanticSymbol, SemanticExpression, SemanticKernelIrModule, SemanticKernelIrOperation } from "./semantic_ir_types.js";
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
const SEMANTIC_SUBGROUP_LOOP_BUDGET_SCRATCH = "bg_semantic_subgroup_loop_budget_scratch";

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
  readonly emitBoolExpression: (
    expression: SemanticExpression,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
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

export function semanticSubgroupControlDeclarations(ir: SemanticKernelIrModule): readonly string[] {
  return ir.subgroupMode !== "scalar" && operationsNeedUniformSubgroupLoop(ir, ir.operations)
    ? [`var<workgroup> ${SEMANTIC_SUBGROUP_LOOP_BUDGET_SCRATCH}: array<u32, ${semanticWorkgroupSize(ir)}>;`]
    : [];
}

export function semanticSubgroupLoopControlIsWorkgroupUniform(
  operation: SemanticLoopOperation,
  ir: SemanticKernelIrModule,
): boolean {
  if (operation.init?.kind !== "declare" || operation.init.target.addressSpace !== "local" || !operation.init.init) return false;
  const counter = operation.init.target.id;
  if (!expressionIsWorkgroupUniform(operation.init.init, ir) ||
    !operation.condition || !expressionIsWorkgroupUniformWithCounter(operation.condition, counter, ir)) return false;
  const update = operation.update;
  if (update?.kind === "update") return update.argument.kind === "symbol" && semanticIdsEqual(update.argument.id, counter);
  return update?.kind === "assignment" && update.target.kind === "symbol" && semanticIdsEqual(update.target.id, counter) &&
    expressionIsWorkgroupUniformWithCounter(update.value, counter, ir);
}

function expressionContainsNativeCollective(expression: SemanticExpression): boolean {
  if (expression.kind === "call" && expression.callee.kind === "symbol" &&
    expression.callee.addressSpace !== "function" && SEMANTIC_SUBGROUP_CALLS.has(expression.callee.name)) return true;
  if (expression.kind === "call" && expression.callee.kind === "member" &&
    ["all", "any", "ballot", "shfl"].includes(expression.callee.property)) return true;
  return semanticExpressionChildren(expression).some(expressionContainsNativeCollective);
}

function operationsContainNativeCollective(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.some((operation) => {
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
}

function operationsNeedUniformSubgroupLoop(
  ir: SemanticKernelIrModule,
  operations: readonly SemanticKernelIrOperation[],
): boolean {
  return operations.some((operation) => {
    if (operation.kind === "loop" && operation.loopKind === "for" && operation.continuing === undefined &&
      operation.update?.kind !== "sequence" && !semanticSubgroupLoopControlIsWorkgroupUniform(operation, ir) &&
      operationsContainNativeCollective(operation.body)) return true;
    if (operation.kind === "branch") return operationsNeedUniformSubgroupLoop(ir, operation.consequent) || operationsNeedUniformSubgroupLoop(ir, operation.alternate);
    if (operation.kind === "loop" || operation.kind === "block") return operationsNeedUniformSubgroupLoop(ir, operation.body);
    return false;
  });
}

export function createSemanticSubgroupControlEmitter<
  Options extends SemanticSubgroupControlOptions,
  TextureSpecializations,
>(
  dependencies: SemanticSubgroupControlDependencies<Options, TextureSpecializations>,
): SemanticSubgroupControlEmitter<Options, TextureSpecializations> {
  const {
    createGeneratedSymbolId,
    emitBoolExpression,
    emitExpressionAs,
    emitLoopInit,
    emitOperation,
    emitTruthiness,
    nameFor,
    semanticError,
  } = dependencies;

  const emitBranchlessLocalMutation = (
    expression: SemanticLocalMutation,
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
    const condition = emitBoolExpression(predicateExpression, ir, names, options, textureSpecializations);
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
        lines.push(...emitBranchlessOperations(operation.body, predicateExpression, ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
        lines.push(`${prefix}}`);
        continue;
      }
      // CUDA cp.async commit/wait fences have no standalone WGSL operation: the
      // semantic WGSL backend intentionally emits them as comments.  Keeping
      // that no-op in the uniformized loop is therefore sound, while copies
      // themselves remain subject to the guarded branchless lowering below.
      if (operation.kind === "copy-fence") {
        lines.push(...emitOperation(operation, ir, names, indentLevel, allowReturnValue, options, textureSpecializations));
        continue;
      }
      // The whole point of this transform is to retain a workgroup-uniform
      // control path around native subgroup work.  Barriers must therefore
      // remain unconditional; predicating them on the per-lane activity bit
      // would make the generated WGSL invalid and could deadlock a workgroup.
      if (operation.kind === "barrier") {
        lines.push(...emitOperation(operation, ir, names, indentLevel, allowReturnValue, options, textureSpecializations));
        continue;
      }
      if (operation.kind === "branch") {
        lines.push(`${prefix}{`);
        lines.push(...emitBranchlessOperations(operation.consequent, collectiveAnd(predicateExpression, operation.condition), ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
        lines.push(`${prefix}}`);
        lines.push(`${prefix}{`);
        lines.push(...emitBranchlessOperations(operation.alternate, collectiveAnd(predicateExpression, collectiveNot(operation.condition)), ir, names, indentLevel + 1, allowReturnValue, options, textureSpecializations));
        lines.push(`${prefix}}`);
        continue;
      }
      if (operation.kind === "declare" && operation.target.addressSpace === "local" && operation.target.dimensions.length === 0 &&
        (!operation.target.pointer || operation.init !== undefined && !expressionContainsNativeCollective(operation.init))) {
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
        lines.push(`${prefix}${emitBranchlessLocalMutation(operation.expression, predicateExpression, ir, names, options, textureSpecializations)};`);
        continue;
      }
      if (operation.kind === "store" && operation.target.addressSpace === "local") {
        const neutral = semanticCompoundStoreNeutral(operation.operator, operation.target.valueType, operation.span);
        if (neutral) {
          lines.push(...emitOperation(
            {
              ...operation,
              value: {
                kind: "conditional",
                condition: predicateExpression,
                consequent: operation.value,
                alternate: neutral,
                valueType: operation.target.valueType,
                span: operation.span,
              },
            },
            ir,
            names,
            indentLevel,
            allowReturnValue,
            expressionContainsNativeCollective(operation.value)
              ? { ...options, activeCollectivePredicate: predicateExpression }
              : options,
            textureSpecializations,
          ));
          continue;
        }
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
    const laneCondition = semanticSubgroupLoopLaneCondition(operation.condition);
    const condition = laneCondition ? emitTruthiness(laneCondition, ir, names, options) : "true";
    const update = operation.update;
    if (update !== undefined && update.kind !== "assignment" && update.kind !== "update") {
      throw semanticError("native subgroup loops require local assignment or increment updates", update.span);
    }
    const iterationBudget = emitUniformIterationBudget(operation, ir, names, options, textureSpecializations, dependencies);
    const iterationName = nameFor(`bg_subgroup_loop_iteration_${operation.span.start}`, names);
    const budgetName = nameFor(`bg_subgroup_loop_budget_${operation.span.start}`, names);
    const budgetLaneName = nameFor(`bg_subgroup_loop_budget_lane_${operation.span.start}`, names);
    const linearRank = semanticLocalLinearRank(ir);
    return [
      `${prefix}{`,
      ...(init === "" ? [] : [`${bodyPrefix}${init};`]),
      ...(iterationBudget.workgroupMaximum ? [
        `${bodyPrefix}${SEMANTIC_SUBGROUP_LOOP_BUDGET_SCRATCH}[${linearRank}] = ${iterationBudget.expression};`,
        `${bodyPrefix}workgroupBarrier();`,
        `${bodyPrefix}if (${linearRank} == 0u) {`,
        `${loopPrefix}var ${budgetName}: u32 = 0u;`,
        `${loopPrefix}for (var ${budgetLaneName}: u32 = 0u; ${budgetLaneName} < ${semanticWorkgroupSize(ir)}u; ${budgetLaneName} += 1u) {`,
        `${loopPrefix}  ${budgetName} = max(${budgetName}, ${SEMANTIC_SUBGROUP_LOOP_BUDGET_SCRATCH}[${budgetLaneName}]);`,
        `${loopPrefix}}`,
        `${loopPrefix}${SEMANTIC_SUBGROUP_LOOP_BUDGET_SCRATCH}[0u] = ${budgetName};`,
        `${bodyPrefix}}`,
        `${bodyPrefix}workgroupBarrier();`,
        `${bodyPrefix}let ${budgetName}: u32 = workgroupUniformLoad(&${SEMANTIC_SUBGROUP_LOOP_BUDGET_SCRATCH}[0u]);`,
      ] : []),
      `${bodyPrefix}var ${activeName}: bool = ${condition};`,
      `${bodyPrefix}for (var ${iterationName}: u32 = 0u; ${iterationName} < ${iterationBudget.workgroupMaximum ? budgetName : iterationBudget.expression}; ${iterationName} += 1u) {`,
      ...emitBranchlessOperations(operation.body, activeExpression, ir, names, indentLevel + 2, allowReturnValue, options, textureSpecializations),
      ...(update === undefined ? [] : [
        `${loopPrefix}${emitBranchlessLocalMutation(update, activeExpression, ir, names, options, textureSpecializations)};`,
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
): { readonly expression: string; readonly workgroupMaximum: boolean } {
  const condition = semanticSubgroupLoopLaneCondition(operation.condition);
  if (operation.init?.kind !== "declare" || operation.init.target.addressSpace !== "local" ||
    condition?.kind !== "binary" || !["<", "<=", ">", ">="].includes(condition.operator) ||
    condition.left.kind !== "symbol" || !semanticIdsEqual(condition.left.id, operation.init.target.id)) {
    throw dependencies.semanticError("native subgroup loops require a canonical directional integer bound", operation.span);
  }
  const update = operation.update;
  const additiveProgress = update?.kind === "update"
    ? update.operator === "++" && update.argument.kind === "symbol" && semanticIdsEqual(update.argument.id, operation.init.target.id)
    : update?.kind === "assignment" && update.operator === "+=" && update.target.kind === "symbol" &&
      semanticIdsEqual(update.target.id, operation.init.target.id) && update.value.kind === "literal" &&
      update.value.literalKind === "number" && update.value.value > 0;
  const shift = update?.kind === "assignment" && (update.operator === "<<=" || update.operator === ">>=") &&
    update.target.kind === "symbol" && semanticIdsEqual(update.target.id, operation.init.target.id) &&
    update.value.kind === "literal" && update.value.literalKind === "number" &&
    Number.isInteger(update.value.value) && update.value.value > 0 && update.value.value < 32
    ? { operator: update.operator, amount: update.value.value }
    : undefined;
  const geometric = update?.kind === "assignment" && (update.operator === "*=" || update.operator === "/=") &&
    update.target.kind === "symbol" && semanticIdsEqual(update.target.id, operation.init.target.id) &&
    update.value.kind === "literal" && update.value.literalKind === "number" &&
    typeof update.value.value === "number" && Number.isInteger(update.value.value) && update.value.value > 1
    ? { operator: update.operator, factor: update.value.value }
    : undefined;
  const conditionIncreases = condition.operator === "<" || condition.operator === "<=";
  const shiftIncreases = shift?.operator === "<<=";
  const geometricIncreases = geometric?.operator === "*=";
  if (!additiveProgress &&
    (shift === undefined || shiftIncreases !== conditionIncreases) &&
    (geometric === undefined || geometricIncreases !== conditionIncreases)) {
    throw dependencies.semanticError("native subgroup loops require a compile-time positive counter step", update?.span ?? operation.span);
  }
  const startExpression = operation.init.init;
  const initialRange = startExpression ? staticIntegerRange(startExpression, ir) : { min: 0, max: 0 };
  if (shift !== undefined) {
    if (shift.operator === "<<=" && (!initialRange || initialRange.min <= 0)) {
      throw dependencies.semanticError("native subgroup left-shift loops require a provably positive initializer", operation.init.span);
    }
    return { expression: `${Math.ceil(32 / shift.amount)}u`, workgroupMaximum: false };
  }
  if (geometric !== undefined) {
    return { expression: `${Math.ceil(32 / Math.log2(geometric.factor))}u`, workgroupMaximum: false };
  }
  const inclusive = condition.operator === "<=" || condition.operator === ">=" ? 1 : 0;
  const bound = condition.right;
  const staticBound = staticIntegerRange(bound, ir);
  if (initialRange && staticBound) {
    const distance = conditionIncreases
      ? staticBound.max - initialRange.min + inclusive
      : initialRange.max - staticBound.min + inclusive;
    return { expression: `${Math.max(0, Math.trunc(distance))}u`, workgroupMaximum: false };
  }
  const start = startExpression
    ? dependencies.emitExpressionAs(startExpression, ir, names, "i32", options, textureSpecializations).code
    : "0";
  const emittedBound = dependencies.emitExpressionAs(bound, ir, names, "i32", options, textureSpecializations).code;
  const step = update?.kind === "update"
    ? 1
    : update?.kind === "assignment" && update.value.kind === "literal" && typeof update.value.value === "number"
      ? update.value.value
      : 0;
  const distance = conditionIncreases ? `(${emittedBound} - ${start} + ${inclusive})` : `(${start} - ${emittedBound} + ${inclusive})`;
  return {
    expression: `u32((max(${distance}, 0) + ${step - 1}) / ${step})`,
    workgroupMaximum: (startExpression !== undefined && !expressionIsWorkgroupUniform(startExpression, ir)) ||
      !expressionIsWorkgroupUniform(bound, ir),
  };
}

function semanticSubgroupLoopLaneCondition(condition: SemanticExpression | undefined): SemanticExpression | undefined {
  if (condition?.kind === "call" && condition.callee.kind === "member" && condition.callee.property === "any") {
    return condition.args[0];
  }
  return condition;
}

function semanticCompoundStoreNeutral(
  operator: string,
  valueType: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>["target"]["valueType"],
  span: SourceSpan,
): SemanticExpression | undefined {
  if (!["float", "double", "half", "int", "uint", "uchar"].includes(valueType)) return undefined;
  const value = operator === "*=" || operator === "/=" ? 1 : ["+=", "-=", "|=", "^="].includes(operator) ? 0 : undefined;
  return value === undefined
    ? undefined
    : { kind: "literal", literalKind: "number", value, valueType, span };
}

function semanticWorkgroupSize(ir: SemanticKernelIrModule): number {
  return ir.workgroupSize[0] * ir.workgroupSize[1] * ir.workgroupSize[2];
}

function semanticLocalLinearRank(ir: SemanticKernelIrModule): string {
  return `(local_id.x + local_id.y * ${ir.workgroupSize[0]}u + local_id.z * ${ir.workgroupSize[0] * ir.workgroupSize[1]}u)`;
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

const WORKGROUP_UNIFORM_BUILTIN_CALLS = new Set([
  "div_ceil",
  "min",
  "max",
]);

function expressionIsWorkgroupUniform(
  expression: SemanticExpression,
  ir: SemanticKernelIrModule,
  visiting: ReadonlySet<string> = new Set(),
): boolean {
  if (expression.kind === "literal") return true;
  if (expression.kind === "symbol") {
    if (expression.addressSpace === "uniform" || expression.addressSpace === "constant") return true;
    if (expression.addressSpace !== "local" || visiting.has(expression.id.key)) return false;
    const declaration = findLocalDeclaration(ir.operations, expression.id);
    if (!declaration?.target.constant || !declaration.init) return false;
    return expressionIsWorkgroupUniform(declaration.init, ir, new Set([...visiting, expression.id.key]));
  }
  if (expression.kind === "cast") return expressionIsWorkgroupUniform(expression.expression, ir, visiting);
  if (expression.kind === "unary") return expressionIsWorkgroupUniform(expression.argument, ir, visiting);
  if (expression.kind === "binary") return expressionIsWorkgroupUniform(expression.left, ir, visiting) && expressionIsWorkgroupUniform(expression.right, ir, visiting);
  if (expression.kind === "conditional") return expressionIsWorkgroupUniform(expression.condition, ir, visiting) &&
    expressionIsWorkgroupUniform(expression.consequent, ir, visiting) && expressionIsWorkgroupUniform(expression.alternate, ir, visiting);
  if (expression.kind === "call" && expression.callee.kind === "symbol" &&
    expression.callee.addressSpace === "builtin" && WORKGROUP_UNIFORM_BUILTIN_CALLS.has(expression.callee.name)) {
    return expression.args.every((arg) => expressionIsWorkgroupUniform(arg, ir, visiting));
  }
  if (expression.kind === "call" && expression.callee.kind === "member" && expression.args.length === 0) {
    return expression.callee.property === "size" || expression.callee.property === "meta_group_size";
  }
  return false;
}

function expressionIsWorkgroupUniformWithCounter(
  expression: SemanticExpression,
  counter: CudaLiteSemanticSymbol["id"],
  ir: SemanticKernelIrModule,
): boolean {
  if (expression.kind === "symbol" && semanticIdsEqual(expression.id, counter)) return true;
  if (expressionIsWorkgroupUniform(expression, ir)) return true;
  const children = semanticExpressionChildren(expression);
  return children.length > 0 && children.every((child) => expressionIsWorkgroupUniformWithCounter(child, counter, ir));
}

function collectiveNot(expression: SemanticExpression): SemanticExpression {
  return { kind: "unary", operator: "!", argument: expression, valueType: "bool", span: expression.span };
}

function collectiveAnd(left: SemanticExpression, right: SemanticExpression): SemanticExpression {
  return { kind: "binary", operator: "&&", left, right, valueType: "bool", span: { ...left.span, end: right.span.end } };
}
