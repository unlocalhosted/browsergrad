import type { Cuda1DGridResult } from "./cuda_concepts.js";

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
