export interface CppCuteBrowserExactDistributionConvergenceAuthoringArguments {
  readonly check: boolean;
  readonly inputPath: string | undefined;
}

export interface CppCuteBrowserExactDistributionConvergenceAuthoringProjection {
  readonly schema:
    "browsergrad.compiler.cpp-cute.exact-distribution-convergence-authoring-projection";
  readonly version: 1;
  readonly authority: "package-authoring-projection-only";
  readonly resource: Readonly<Record<string, unknown>>;
  readonly resourceSha256: string;
  readonly resourceByteLength: number;
  readonly sourceRevision: string;
  readonly matrixId: string;
  readonly caseCount: 9;
}

export class CppCuteBrowserExactDistributionConvergenceAuthoringError
  extends Error {
  readonly code:
    "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-CONVERGENCE-AUTHORING";
  readonly path: string;
}

export function
parseCppCuteBrowserExactDistributionConvergenceAuthoringArguments(
  argv: readonly string[],
): Readonly<
  CppCuteBrowserExactDistributionConvergenceAuthoringArguments
>;

export function projectCppCuteBrowserExactDistributionConvergence(
  value: unknown,
): Promise<Readonly<
  CppCuteBrowserExactDistributionConvergenceAuthoringProjection
>>;

export function
renderCppCuteBrowserExactDistributionConvergenceResource(
  projection:
    CppCuteBrowserExactDistributionConvergenceAuthoringProjection,
): string;

export function
renderCppCuteBrowserExactDistributionConvergenceIdentity(
  projection:
    CppCuteBrowserExactDistributionConvergenceAuthoringProjection,
): string;
