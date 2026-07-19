export interface VerifyCompilerCommand {
  readonly id: string;
  readonly executable: string;
  readonly arguments: readonly string[];
}

export interface VerifyCompilerLane {
  readonly id: string;
  readonly commands: readonly VerifyCompilerCommand[];
}

export interface VerifyCompilerPlan {
  readonly schema: "browsergrad.compiler.verify-plan";
  readonly version: 1;
  readonly maximumConcurrentLanes: number;
  readonly prerequisites: readonly VerifyCompilerCommand[];
  readonly lanes: readonly VerifyCompilerLane[];
}

export type VerifyCompilerEvent = Readonly<{
  type: "command-start" | "command-pass" | "command-fail" | "lane-start" | "lane-pass";
  phase?: "prerequisite" | "lane";
  laneId?: string;
  commandId?: string;
}>;

export class VerifyCompilerCommandError extends Error {
  readonly commandId: string;
  constructor(command: VerifyCompilerCommand, message: string, options?: ErrorOptions);
}

export class VerifyCompilerSignalError extends Error {
  readonly signal: NodeJS.Signals;
  constructor(signal: NodeJS.Signals);
}

export function createVerifyCompilerPlan(): VerifyCompilerPlan;
export function validateVerifyCompilerPlan(value: unknown): VerifyCompilerPlan;
export function executeVerifyCompilerPlan(
  plan: VerifyCompilerPlan,
  options?: Readonly<{
    executeCommand?: typeof executeVerifyCompilerCommand;
    signal?: AbortSignal;
    onEvent?: (event: VerifyCompilerEvent) => void;
  }>,
): Promise<void>;
export function executeVerifyCompilerCommand(
  command: VerifyCompilerCommand,
  options: Readonly<{ signal: AbortSignal }>,
): Promise<void>;
