export interface CppCuteBrowserBuildRuntimeClosureFileV1 {
  readonly kind: "runtime" | "extractor";
  readonly path: string;
  readonly sha256: string;
  readonly byteLength: number;
}

export interface CppCuteBrowserBuildRuntimeClosureObservationV1 {
  readonly schema: "browsergrad.compiler.cpp-cute.build-runtime-closure";
  readonly version: 1;
  readonly authority: "staged-build-runtime-closure-observation-only";
  readonly lockId: string;
  readonly extractorSourceSetSha256: string;
  readonly closureSha256: string;
  readonly fileCount: number;
  readonly files: readonly CppCuteBrowserBuildRuntimeClosureFileV1[];
  readonly claims: Readonly<{
    exactReadableWorkspaceClosureVerified: true;
    buildExecuted: false;
    outputIdentityAuthorized: false;
    reproducibilityVerified: false;
    releaseReady: false;
  }>;
}

export interface StagedCppCuteBrowserBuildRuntimeClosure {
  readonly workspaceRoot: string;
  readonly observationPath: string;
  readonly observationSha256: string;
  readonly observationByteLength: number;
  readonly closureSha256: string;
  readonly fileCount: number;
}

export interface VerifiedCppCuteBrowserBuildRuntimeClosure {
  readonly observationPath: string;
  readonly observationSha256: string;
  readonly observationByteLength: number;
  readonly observation: CppCuteBrowserBuildRuntimeClosureObservationV1;
}

export const CPP_CUTE_BROWSER_BUILD_RUNTIME_CLOSURE_OBSERVATION_NAME: string;
export const CPP_CUTE_BROWSER_BUILD_RUNTIME_SOURCE_PATHS: readonly string[];

export function parseCppCuteBrowserBuildRuntimeClosureArguments(
  argv: readonly string[],
): Readonly<{ outputRoot: string; sourceRoot: string }>;

export function stageCppCuteBrowserBuildRuntimeClosure(input: Readonly<{
  outputRoot: string;
  sourceRoot: string;
}>): Promise<StagedCppCuteBrowserBuildRuntimeClosure>;

export function verifyCppCuteBrowserBuildRuntimeClosure(input: Readonly<{
  workspaceRoot: string;
}>): Promise<VerifiedCppCuteBrowserBuildRuntimeClosure>;

export class CppCuteBrowserBuildRuntimeClosureError extends Error {
  readonly code: string;
  readonly path: string;
}
