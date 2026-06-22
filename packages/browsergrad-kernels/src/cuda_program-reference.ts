import { runThreadGrid, type Cuda1DGridResult, type Cuda1DThreadContext } from "./cuda_concepts.js";
import type { Cuda1DCondition, Cuda1DExpression, Cuda1DProgram, Cuda1DProgramRunInput, Cuda1DStatement, Kernel1DProgram, Kernel1DProgramReferenceResult, Kernel1DProgramRunInput } from "./cuda_program-types.js";
import { readParameter, validateFiniteNumber, validateInteger } from "./cuda_program-validate.js";

export function simulateCuda1DProgram(
  program: Cuda1DProgram,
  input: Cuda1DProgramRunInput = {},
): Cuda1DGridResult {
  return runKernel1DProgramReference(program, input);
}

export function runKernel1DProgramReference(
  program: Kernel1DProgram,
  input: Kernel1DProgramRunInput = {},
): Kernel1DProgramReferenceResult {
  return runThreadGrid({
    inputLength: program.inputLength,
    outputLength: program.outputLength,
    blocks: program.launch.blocks,
    threadsPerBlock: program.launch.threadsPerBlock,
    ...(input.initialInput !== undefined ? { initialInput: input.initialInput } : {}),
    ...(input.initialOutput !== undefined ? { initialOutput: input.initialOutput } : {}),
    kernel(context) {
      executeStatements(program.body, context, program.parameters ?? {});
    },
  });
}

function executeStatements(
  statements: readonly Cuda1DStatement[],
  context: Cuda1DThreadContext,
  parameters: Readonly<Record<string, number>>,
): void {
  for (const statement of statements) {
    switch (statement.op) {
      case "write":
        context.write(
          evaluateIndexExpression(statement.index, context, parameters),
          evaluateExpression(statement.value, context, parameters),
        );
        break;
      case "if":
        if (evaluateCondition(statement.condition, context, parameters)) {
          executeStatements(statement.body, context, parameters);
        }
        break;
    }
  }
}

function evaluateCondition(
  condition: Cuda1DCondition,
  context: Cuda1DThreadContext,
  parameters: Readonly<Record<string, number>>,
): boolean {
  const left = evaluateExpression(condition.left, context, parameters);
  const right = evaluateExpression(condition.right, context, parameters);
  switch (condition.op) {
    case "lt":
      return left < right;
    case "lte":
      return left <= right;
    case "eq":
      return left === right;
  }
}

function evaluateIndexExpression(
  expression: Cuda1DExpression,
  context: Cuda1DThreadContext,
  parameters: Readonly<Record<string, number>>,
): number {
  return validateInteger(
    evaluateExpression(expression, context, parameters),
    "memory index",
  );
}

function evaluateExpression(
  expression: Cuda1DExpression,
  context: Cuda1DThreadContext,
  parameters: Readonly<Record<string, number>>,
): number {
  switch (expression.op) {
    case "literal":
      return validateFiniteNumber(expression.value, "literal");
    case "threadId":
      return context.globalThreadId;
    case "inputLength":
      return context.inputLength;
    case "outputLength":
      return context.outputLength;
    case "param":
      return readParameter(parameters, expression.name);
    case "read":
      return context.read(evaluateIndexExpression(expression.index, context, parameters));
    case "outputRead":
      return context.readOutput(
        evaluateIndexExpression(expression.index, context, parameters),
      );
    case "add":
      return evaluateExpression(expression.left, context, parameters) +
        evaluateExpression(expression.right, context, parameters);
    case "sub":
      return evaluateExpression(expression.left, context, parameters) -
        evaluateExpression(expression.right, context, parameters);
    case "mul":
      return evaluateExpression(expression.left, context, parameters) *
        evaluateExpression(expression.right, context, parameters);
    case "div":
      return evaluateExpression(expression.left, context, parameters) /
        evaluateExpression(expression.right, context, parameters);
  }
}
