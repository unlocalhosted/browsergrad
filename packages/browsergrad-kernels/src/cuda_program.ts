import {
  runThreadGrid,
  type Cuda1DGridResult,
  type Cuda1DThreadContext,
} from "./cuda_concepts.js";
import { dispatch } from "./runner.js";
import { KernelError, type KernelDevice } from "./types.js";

export interface Cuda1DProgramInput {
  readonly name: string;
  readonly inputLength: number;
  readonly outputLength: number;
  readonly parameters?: Readonly<Record<string, number>>;
  readonly launch: Cuda1DLaunch;
  readonly body: readonly Cuda1DStatement[];
}

export interface Cuda1DProgram extends Cuda1DProgramInput {
  readonly body: readonly Cuda1DStatement[];
}

export interface Cuda1DLaunch {
  readonly blocks: number;
  readonly threadsPerBlock: number;
}

export type Cuda1DStatement =
  | Cuda1DWriteStatement
  | Cuda1DIfStatement;

export interface Cuda1DWriteStatement {
  readonly op: "write";
  readonly index: Cuda1DExpression;
  readonly value: Cuda1DExpression;
}

export interface Cuda1DIfStatement {
  readonly op: "if";
  readonly condition: Cuda1DCondition;
  readonly body: readonly Cuda1DStatement[];
}

export type Cuda1DExpression =
  | { readonly op: "literal"; readonly value: number }
  | { readonly op: "threadId" }
  | { readonly op: "inputLength" }
  | { readonly op: "outputLength" }
  | { readonly op: "param"; readonly name: string }
  | { readonly op: "read"; readonly index: Cuda1DExpression }
  | { readonly op: "outputRead"; readonly index: Cuda1DExpression }
  | { readonly op: "add"; readonly left: Cuda1DExpression; readonly right: Cuda1DExpression }
  | { readonly op: "sub"; readonly left: Cuda1DExpression; readonly right: Cuda1DExpression }
  | { readonly op: "mul"; readonly left: Cuda1DExpression; readonly right: Cuda1DExpression }
  | { readonly op: "div"; readonly left: Cuda1DExpression; readonly right: Cuda1DExpression };

export type Cuda1DCondition =
  | { readonly op: "lt"; readonly left: Cuda1DExpression; readonly right: Cuda1DExpression }
  | { readonly op: "lte"; readonly left: Cuda1DExpression; readonly right: Cuda1DExpression }
  | { readonly op: "eq"; readonly left: Cuda1DExpression; readonly right: Cuda1DExpression };

export interface Cuda1DProgramRunInput {
  readonly initialInput?: readonly number[];
  readonly initialOutput?: readonly number[];
}

export type Kernel1DProgramInput = Cuda1DProgramInput;
export type Kernel1DProgram = Cuda1DProgram;
export type Kernel1DLaunch = Cuda1DLaunch;
export type Kernel1DStatement = Cuda1DStatement;
export type Kernel1DWriteStatement = Cuda1DWriteStatement;
export type Kernel1DIfStatement = Cuda1DIfStatement;
export type Kernel1DExpression = Cuda1DExpression;
export type Kernel1DCondition = Cuda1DCondition;
export type Kernel1DProgramRunInput = Cuda1DProgramRunInput;
export type Kernel1DProgramReferenceResult = Cuda1DGridResult;

export function defineKernel1DProgram(
  input: Kernel1DProgramInput,
): Kernel1DProgram {
  return defineCuda1DProgram(input);
}

export function defineCuda1DProgram(input: Cuda1DProgramInput): Cuda1DProgram {
  validateIdentifier(input.name, "name");
  validateNonNegativeInteger(input.inputLength, "inputLength");
  validateNonNegativeInteger(input.outputLength, "outputLength");
  validatePositiveInteger(input.launch.blocks, "launch.blocks");
  validatePositiveInteger(
    input.launch.threadsPerBlock,
    "launch.threadsPerBlock",
  );
  if (input.body.length === 0) {
    throw new KernelError("Cuda1DProgram body must contain at least one statement");
  }
  const parameters = validateParameters(input.parameters ?? {});
  return {
    name: input.name,
    inputLength: input.inputLength,
    outputLength: input.outputLength,
    parameters,
    launch: {
      blocks: input.launch.blocks,
      threadsPerBlock: input.launch.threadsPerBlock,
    },
    body: input.body.map(cloneStatement),
  };
}

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

export async function runCuda1DProgramWebGpu(
  device: KernelDevice,
  program: Cuda1DProgram,
  input: Cuda1DProgramRunInput = {},
): Promise<Float32Array> {
  return runKernel1DProgramWebGpu(device, program, input);
}

export async function runKernel1DProgramWebGpu(
  device: KernelDevice,
  program: Kernel1DProgram,
  input: Kernel1DProgramRunInput = {},
): Promise<Float32Array> {
  const initialInput = Float32Array.from(
    materializeVector(input.initialInput, program.inputLength, "initialInput"),
  );
  const initialOutput =
    input.initialOutput === undefined
      ? undefined
      : Float32Array.from(
          materializeVector(input.initialOutput, program.outputLength, "initialOutput"),
        );
  const parameterNames = Object.keys(program.parameters ?? {}).sort();
  const params = Float32Array.from(
    parameterNames.map((name) => readParameter(program.parameters ?? {}, name)),
  );

  return dispatch(
    device,
    {
      name: `kernel-1d-program-${program.name}`,
      wgsl: emitKernel1DProgramWgsl(program),
      workgroupSize: [program.launch.threadsPerBlock, 1, 1],
    },
    {
      inputs: [initialInput],
      outputLength: program.outputLength,
      params,
      ...(initialOutput ? { initialOutput } : {}),
      dispatchCount: [
        program.launch.blocks * program.launch.threadsPerBlock,
        1,
        1,
      ],
      cacheKeySuffix: [
        program.name,
        program.inputLength,
        program.outputLength,
        program.launch.blocks,
        program.launch.threadsPerBlock,
        parameterNames.join(","),
      ].join(":"),
    },
  );
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

function cloneStatement(statement: Cuda1DStatement): Cuda1DStatement {
  switch (statement.op) {
    case "write":
      return {
        op: "write",
        index: cloneExpression(statement.index),
        value: cloneExpression(statement.value),
      };
    case "if":
      return {
        op: "if",
        condition: cloneCondition(statement.condition),
        body: statement.body.map(cloneStatement),
      };
  }
}

function cloneCondition(condition: Cuda1DCondition): Cuda1DCondition {
  return {
    op: condition.op,
    left: cloneExpression(condition.left),
    right: cloneExpression(condition.right),
  };
}

function cloneExpression(expression: Cuda1DExpression): Cuda1DExpression {
  switch (expression.op) {
    case "literal":
      return { op: "literal", value: expression.value };
    case "threadId":
    case "inputLength":
    case "outputLength":
      return { op: expression.op };
    case "param":
      return { op: "param", name: expression.name };
    case "read":
      return { op: "read", index: cloneExpression(expression.index) };
    case "outputRead":
      return { op: "outputRead", index: cloneExpression(expression.index) };
    case "add":
    case "sub":
    case "mul":
    case "div":
      return {
        op: expression.op,
        left: cloneExpression(expression.left),
        right: cloneExpression(expression.right),
      };
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

function validateParameters(
  parameters: Readonly<Record<string, number>>,
): Readonly<Record<string, number>> {
  const out: Record<string, number> = {};
  for (const [name, value] of Object.entries(parameters).sort(([a], [b]) =>
    a.localeCompare(b),
  )) {
    validateIdentifier(name, "parameter name");
    out[name] = validateFiniteNumber(value, `parameters.${name}`);
  }
  return out;
}

function readParameter(
  parameters: Readonly<Record<string, number>>,
  name: string,
): number {
  validateIdentifier(name, "parameter name");
  const value = parameters[name];
  if (value === undefined) {
    throw new KernelError(`missing Cuda1DProgram parameter: ${name}`);
  }
  return value;
}

function materializeVector(
  values: readonly number[] | undefined,
  length: number,
  name: string,
): number[] {
  if (values === undefined) return Array.from({ length }, () => 0);
  if (values.length !== length) {
    throw new KernelError(`${name} length must be ${length}`);
  }
  return values.map((value, index) =>
    validateFiniteNumber(value, `${name}[${index}]`),
  );
}

function validateIdentifier(value: string, name: string): string {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new KernelError(`${name} must be a valid WGSL identifier`);
  }
  return value;
}

function validatePositiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new KernelError(`${name} must be a positive integer`);
  }
  return value;
}

function validateNonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new KernelError(`${name} must be a non-negative integer`);
  }
  return value;
}

function validateInteger(value: number, name: string): number {
  if (!Number.isInteger(value)) {
    throw new KernelError(`${name} must be an integer`);
  }
  return value;
}

function validateFiniteNumber(value: number, name: string): number {
  if (!Number.isFinite(value)) {
    throw new KernelError(`${name} must be finite`);
  }
  return value;
}
