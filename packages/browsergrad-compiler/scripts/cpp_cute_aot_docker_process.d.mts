export type CppCuteAotDockerProcessErrorCode =
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-INVALID"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-SPAWN"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-TIMEOUT"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-STDOUT-LIMIT"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-STDERR-LIMIT"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-KILL";

export class CppCuteAotDockerProcessError extends Error {
  readonly code: CppCuteAotDockerProcessErrorCode;
  readonly path: string;
}

export interface BoundedChildProcessRequest {
  readonly executable: string;
  readonly arguments: readonly string[];
  readonly cwd: string;
  readonly environment: Readonly<Record<string, string>>;
  readonly timeoutMs: number;
  readonly killGraceMs: number;
  readonly stdoutByteLimit: number;
  readonly stderrByteLimit: number;
  readonly signal?: AbortSignal;
}

export interface BoundedChildProcessResult {
  readonly exitCode: number | null;
  readonly signal: string | null;
  readonly stdout: Uint8Array;
  readonly stderr: Uint8Array;
}

export interface CppCuteAotPrivateDockerRequestInput {
  readonly runRoot: string;
  readonly configDirectory: string;
  readonly homeDirectory: string;
  readonly signal?: AbortSignal;
}

export function buildCppCuteAotDockerVersionRequest(
  input: CppCuteAotPrivateDockerRequestInput,
): BoundedChildProcessRequest;

export function buildCppCuteAotDockerInfoRequest(
  input: CppCuteAotPrivateDockerRequestInput,
): BoundedChildProcessRequest;

export function buildCppCuteAotDockerImageInspectRequest(input: Readonly<{
  runRoot: string;
  configDirectory: string;
  homeDirectory: string;
  imageReference: string;
  signal?: AbortSignal;
}>): BoundedChildProcessRequest;

export function buildCppCuteAotDockerCreateRequest(input: Readonly<{
  runRoot: string;
  configDirectory: string;
  homeDirectory: string;
  controlDirectory: string;
  sourceDirectory: string;
  containerIdFile: string;
  containerName: string;
  sessionNonce: string;
  imageReference: string;
  runMetadataId: string;
  requestId: string;
  executionPlanSha256: string;
  memoryBytes: number;
  maxProcesses: number;
  seccompProfilePath: string;
  signal?: AbortSignal;
}>): BoundedChildProcessRequest;

export function buildCppCuteAotDockerContainerInspectRequest(
  input: Readonly<CppCuteAotPrivateDockerRequestInput & { containerId: string }>,
): BoundedChildProcessRequest;

export function buildCppCuteAotDockerContainerRecoveryRequest(
  input: Readonly<CppCuteAotPrivateDockerRequestInput & { containerName: string }>,
): BoundedChildProcessRequest;

export function buildCppCuteAotDockerStartAttachedRequest(input: Readonly<
  CppCuteAotPrivateDockerRequestInput & {
    containerId: string;
    timeoutMs: number;
    stdoutByteLimit: number;
  }
>): BoundedChildProcessRequest;

export function buildCppCuteAotDockerRemoveRequest(input: Readonly<{
  runRoot: string;
  configDirectory: string;
  homeDirectory: string;
  containerId: string;
  force: boolean;
}>): BoundedChildProcessRequest;

export function buildCppCuteAotDockerAbsenceRequest(
  input: Readonly<CppCuteAotPrivateDockerRequestInput & { containerId: string }>,
): BoundedChildProcessRequest;

export function runBoundedChildProcess(
  request: BoundedChildProcessRequest,
): Promise<BoundedChildProcessResult>;
