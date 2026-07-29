export interface CppCuteBrowserReproducibilityAuthoringArguments {
  readonly check: boolean;
  readonly inputPath: string | undefined;
  readonly runId: string | undefined;
  readonly sourceRevision: string | undefined;
}

export interface CppCuteBrowserReproducibilityAuthoringMetadata {
  readonly runId: string;
  readonly sourceRevision: string;
}

export interface CppCuteBrowserReproducibilityAuthoringProjection {
  readonly schema:
    "browsergrad.compiler.cpp-cute.reproducibility-authoring-projection";
  readonly version: 1;
  readonly authority: "package-authoring-projection-only";
  readonly evidence: Readonly<Record<string, unknown>>;
  readonly resourceSha256: string;
  readonly resourceByteLength: number;
  readonly runId: string;
  readonly sourceRevision: string;
  readonly wasmSha256: string;
  readonly wasmByteLength: number;
}

export class CppCuteBrowserReproducibilityAuthoringError extends Error {
  readonly code:
    "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-AUTHORING";
  readonly path: string;
}

export function parseCppCuteBrowserReproducibilityAuthoringArguments(
  argv: readonly string[],
): Readonly<CppCuteBrowserReproducibilityAuthoringArguments>;

export function projectCppCuteBrowserReproducibility(
  value: unknown,
  metadata: CppCuteBrowserReproducibilityAuthoringMetadata,
): Promise<Readonly<CppCuteBrowserReproducibilityAuthoringProjection>>;

export function renderCppCuteBrowserReproducibilityResource(
  projection: CppCuteBrowserReproducibilityAuthoringProjection,
): string;

export function renderCppCuteBrowserReproducibilityIdentity(
  projection: CppCuteBrowserReproducibilityAuthoringProjection,
): string;
