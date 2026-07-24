export interface CppCuteBrowserStrictCompileObservationAuthoringArguments {
  readonly check: boolean;
  readonly inputPath: string | undefined;
}

export interface CppCuteBrowserStrictCompileObservationAuthoringProjection {
  readonly schema:
    "browsergrad.compiler.cpp-cute.strict-compile-observation-authoring-projection";
  readonly version: 1;
  readonly authority: "package-authoring-projection-only";
  readonly matrix: Readonly<Record<string, unknown>>;
  readonly resourceSha256: string;
  readonly resourceByteLength: number;
  readonly sourceRevision: string;
  readonly workerBundleSha256: string;
  readonly verifierWorkerBundleSha256: string;
  readonly wasmSha256: string;
  readonly wasmByteLength: number;
}

export class CppCuteBrowserStrictCompileObservationAuthoringError
  extends Error {
  readonly code:
    "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-OBSERVATION-AUTHORING";
  readonly path: string;
}

export function parseCppCuteBrowserStrictCompileObservationAuthoringArguments(
  argv: readonly string[],
): Readonly<CppCuteBrowserStrictCompileObservationAuthoringArguments>;

export function projectCppCuteBrowserStrictCompileObservation(
  value: unknown,
): Promise<Readonly<CppCuteBrowserStrictCompileObservationAuthoringProjection>>;

export function renderCppCuteBrowserStrictCompileObservationResource(
  projection: CppCuteBrowserStrictCompileObservationAuthoringProjection,
): string;

export function renderCppCuteBrowserStrictCompileObservationIdentity(
  projection: CppCuteBrowserStrictCompileObservationAuthoringProjection,
): string;
