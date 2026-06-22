import type { Cuda1DCondition, Cuda1DExpression, Cuda1DProgram, Cuda1DStatement, Kernel1DProgram } from "./cuda_program-types.js";
import { validateFiniteNumber, validateNonNegativeInteger } from "./cuda_program-validate.js";

export function emitCuda1DProgramWgsl(program: Cuda1DProgram): string {
  return emitKernel1DProgramWgsl(program);
}

export function emitKernel1DProgramWgsl(program: Kernel1DProgram): string {
  const body = emitStatements(program.body, 1);
  const parameterNames = Object.keys(program.parameters ?? {}).sort();
  const parameterLines =
    parameterNames.length === 0
      ? []
      : [
          "struct Params {",
          ...parameterNames.map((name) => `  ${name}: f32,`),
          "};",
          "@group(0) @binding(2) var<uniform> params: Params;",
          "",
        ];
  return [
    `// BrowserGrad Kernel1D program: ${program.name}`,
    "@group(0) @binding(0) var<storage, read> inputBuffer: array<f32>;",
    "@group(0) @binding(1) var<storage, read_write> outputBuffer: array<f32>;",
    ...parameterLines,
    "",
    `@compute @workgroup_size(${program.launch.threadsPerBlock})`,
    "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {",
    "  let i: u32 = global_id.x;",
    "  let inputLength: u32 = arrayLength(&inputBuffer);",
    "  let outputLength: u32 = arrayLength(&outputBuffer);",
    ...body,
    "}",
    "",
  ].join("\n");
}

function emitStatements(
  statements: readonly Cuda1DStatement[],
  indentLevel: number,
): string[] {
  return statements.flatMap((statement) => emitStatement(statement, indentLevel));
}

function emitStatement(
  statement: Cuda1DStatement,
  indentLevel: number,
): string[] {
  const indent = "  ".repeat(indentLevel);
  switch (statement.op) {
    case "write":
      return [
        `${indent}outputBuffer[${emitIndexExpression(statement.index)}] = ${emitExpression(statement.value, "value")};`,
      ];
    case "if":
      return [
        `${indent}if (${emitCondition(statement.condition)}) {`,
        ...emitStatements(statement.body, indentLevel + 1),
        `${indent}}`,
      ];
  }
}

function emitCondition(condition: Cuda1DCondition): string {
  const left = emitExpression(condition.left, "index");
  const right = emitExpression(condition.right, "index");
  switch (condition.op) {
    case "lt":
      return `${left} < ${right}`;
    case "lte":
      return `${left} <= ${right}`;
    case "eq":
      return `${left} == ${right}`;
  }
}

function emitIndexExpression(expression: Cuda1DExpression): string {
  return emitExpression(expression, "index");
}

function emitExpression(
  expression: Cuda1DExpression,
  usage: "index" | "value",
): string {
  switch (expression.op) {
    case "literal":
      return usage === "index"
        ? emitU32Literal(expression.value)
        : emitF32Literal(expression.value);
    case "threadId":
      return "i";
    case "inputLength":
      return "inputLength";
    case "outputLength":
      return "outputLength";
    case "param":
      return `params.${expression.name}`;
    case "read":
      return `inputBuffer[${emitIndexExpression(expression.index)}]`;
    case "outputRead":
      return `outputBuffer[${emitIndexExpression(expression.index)}]`;
    case "add":
      return `(${emitExpression(expression.left, usage)} + ${emitExpression(expression.right, usage)})`;
    case "sub":
      return `(${emitExpression(expression.left, usage)} - ${emitExpression(expression.right, usage)})`;
    case "mul":
      return `(${emitExpression(expression.left, usage)} * ${emitExpression(expression.right, usage)})`;
    case "div":
      return `(${emitExpression(expression.left, usage)} / ${emitExpression(expression.right, usage)})`;
  }
}

function emitF32Literal(value: number): string {
  const finite = validateFiniteNumber(value, "literal");
  if (Number.isInteger(finite)) return `${finite}.0`;
  return String(finite);
}

function emitU32Literal(value: number): string {
  const integer = validateNonNegativeInteger(value, "literal index");
  return `${integer}u`;
}
