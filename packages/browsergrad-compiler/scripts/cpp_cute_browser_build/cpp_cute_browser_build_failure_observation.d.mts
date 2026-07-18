export interface CppCuteBrowserBuildFailureObservationResult {
  readonly outputPath: string;
  readonly sha256: string;
  readonly byteLength: number;
  readonly partialLogCount: number;
  readonly successfulBuildReceiptWritten: false;
}

export interface CppCuteBrowserBuildFailureCauseProjection {
  readonly name: string;
  readonly code: string;
  readonly path: string;
  readonly message: string;
}

export interface CppCuteBrowserBuildFailureProjection
  extends CppCuteBrowserBuildFailureCauseProjection {
  readonly causes: readonly CppCuteBrowserBuildFailureCauseProjection[];
  readonly causeChainComplete: boolean;
}

export class CppCuteBrowserBuildFailureObservationError extends Error {
  readonly code:
    | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-FAILURE-OBSERVATION-INVALID"
    | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-FAILURE-OBSERVATION-CONFLICT"
    | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-FAILURE-OBSERVATION-IO";
  readonly path: string;
}

export function persistCppCuteBrowserBuildFailureObservation(input: Readonly<{
  outputRoot: string;
  stateRoot: string;
  lockId: string;
  sourceSetSha256: string;
  cause: unknown;
}>): Promise<CppCuteBrowserBuildFailureObservationResult>;

export function projectCppCuteBrowserBuildFailure(
  cause: unknown,
): CppCuteBrowserBuildFailureProjection;
