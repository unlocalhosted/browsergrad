export interface CppCuteBrowserFullDistributionReproducibilityAuthoringArguments {
  readonly check: boolean;
  readonly deterministicRoot: string | undefined;
  readonly evidencePath: string | undefined;
  readonly producerPolicyPath: string | undefined;
  readonly profilePath: string | undefined;
  readonly sourceRevision: string | undefined;
}

export interface CppCuteBrowserFullDistributionReproducibilityAuthoringProjection {
  readonly schema:
    "browsergrad.compiler.cpp-cute.browser-full-distribution-reproducibility-authoring-projection";
  readonly version: 1;
  readonly authority: "package-authoring-projection-only";
  readonly resource: Readonly<Record<string, unknown>>;
  readonly resourceSha256: string;
  readonly resourceByteLength: number;
  readonly sourceRevision: string;
}

export class CppCuteBrowserFullDistributionReproducibilityAuthoringError
  extends Error {
  readonly code:
    "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-REPRODUCIBILITY-AUTHORING";
  readonly path: string;
}

export function parseCppCuteBrowserFullDistributionReproducibilityAuthoringArguments(
  argv: readonly string[],
): Readonly<CppCuteBrowserFullDistributionReproducibilityAuthoringArguments>;

export function projectCppCuteBrowserFullDistributionReproducibility(
  evidence: unknown,
  inputs: Readonly<{
    deterministicRoot: string;
    producerPolicyBytes: Uint8Array;
    profileBytes: Uint8Array;
    sourceRevision: string;
  }>,
): Promise<
  Readonly<CppCuteBrowserFullDistributionReproducibilityAuthoringProjection>
>;

export function renderCppCuteBrowserFullDistributionReproducibilityResource(
  value: CppCuteBrowserFullDistributionReproducibilityAuthoringProjection,
): string;

export function renderCppCuteBrowserFullDistributionReproducibilityIdentity(
  value: CppCuteBrowserFullDistributionReproducibilityAuthoringProjection,
): string;
