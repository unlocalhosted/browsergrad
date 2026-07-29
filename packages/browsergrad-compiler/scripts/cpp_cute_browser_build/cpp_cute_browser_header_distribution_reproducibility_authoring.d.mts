export interface CppCuteBrowserHeaderDistributionReproducibilityAuthoringArguments {
  readonly check: boolean;
  readonly inputPath: string | undefined;
  readonly sourceRevision: string | undefined;
}

export interface CppCuteBrowserHeaderDistributionReproducibilityAuthoringProjection {
  readonly schema:
    "browsergrad.compiler.cpp-cute.browser-header-distribution-reproducibility-authoring-projection";
  readonly version: 1;
  readonly authority: "package-authoring-projection-only";
  readonly resource: Readonly<Record<string, unknown>>;
  readonly resourceSha256: string;
  readonly resourceByteLength: number;
  readonly sourceRevision: string;
  readonly headerInputProjectionId: string;
}

export class CppCuteBrowserHeaderDistributionReproducibilityAuthoringError
  extends Error {
  readonly code:
    "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-DISTRIBUTION-REPRODUCIBILITY-AUTHORING";
  readonly path: string;
}

export function parseCppCuteBrowserHeaderDistributionReproducibilityAuthoringArguments(
  argv: readonly string[],
): Readonly<CppCuteBrowserHeaderDistributionReproducibilityAuthoringArguments>;

export function projectCppCuteBrowserHeaderDistributionReproducibility(
  value: unknown,
  sourceRevision: string,
): Promise<
  Readonly<CppCuteBrowserHeaderDistributionReproducibilityAuthoringProjection>
>;

export function renderCppCuteBrowserHeaderDistributionReproducibilityResource(
  projection: CppCuteBrowserHeaderDistributionReproducibilityAuthoringProjection,
): string;

export function renderCppCuteBrowserHeaderDistributionReproducibilityIdentity(
  projection: CppCuteBrowserHeaderDistributionReproducibilityAuthoringProjection,
): string;
