export interface CppCuteBrowserExactDistributionConvergenceArguments {
  readonly distributionRoot: string;
  readonly profilePath: string;
  readonly producerPolicyPath: string;
  readonly producerTrustStorePath: string;
  readonly preflightOnly: boolean;
  readonly checkpointDirectory?: string;
  readonly evidenceOutput?: string;
  readonly sourceRevision?: string;
}

export interface CppCuteBrowserExactDistributionConvergencePreflight {
  readonly schema: string;
  readonly version: 1;
  readonly authority: string;
  readonly controls: Readonly<Record<string, Readonly<{
    readonly route: string;
    readonly path: string;
    readonly mediaType: string;
    readonly sha256: string;
    readonly byteLength: number;
  }>>>;
  readonly assets: readonly Readonly<{
    readonly assetId: string;
    readonly route: string;
    readonly path: string;
    readonly mediaType: string;
    readonly sha256: string;
    readonly byteLength: number;
  }>[];
  readonly distribution: Readonly<Record<string, string | number>>;
  readonly producer: Readonly<Record<string, string>>;
  readonly claims: Readonly<Record<string, boolean>>;
}

export interface CppCuteBrowserExactDistributionConvergenceMatrix {
  readonly schema: string;
  readonly version: 1;
  readonly authority: string;
  readonly matrixId: string;
  readonly sourceRevision: string;
  readonly caseCount: number;
  readonly webgpu: Readonly<Record<string, boolean | number>>;
  readonly claims: Readonly<Record<string, boolean>>;
  readonly cases: readonly Readonly<Record<string, unknown>>[];
}

export class CppCuteBrowserExactDistributionConvergenceError extends Error {
  readonly code:
    "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-DISTRIBUTION-CONVERGENCE";
  readonly path: string;
}

export const
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_INPUT_SCHEMA: string;
export const
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_OBSERVATION_SCHEMA: string;
export const
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_MATRIX_SCHEMA: string;

export function parseCppCuteBrowserExactDistributionConvergenceArguments(
  argv: readonly string[],
): CppCuteBrowserExactDistributionConvergenceArguments;

export function isRetryableCppCuteBrowserExactDistributionFailure(
  output: unknown,
): boolean;

export function preflightCppCuteBrowserExactDistributionConvergence(
  input: Readonly<{
    readonly distributionRoot: string;
    readonly profilePath: string;
    readonly producerPolicyPath: string;
    readonly producerTrustStorePath: string;
  }>,
): Promise<CppCuteBrowserExactDistributionConvergencePreflight>;

export function prepareCppCuteBrowserExactDistributionConvergenceMatrix(
  observations: readonly Readonly<Record<string, unknown>>[],
  preflight: CppCuteBrowserExactDistributionConvergencePreflight,
  sourceRevision: string,
): Readonly<CppCuteBrowserExactDistributionConvergenceMatrix>;

export function runCppCuteBrowserExactDistributionConvergence(
  argv?: readonly string[],
): Promise<Readonly<CppCuteBrowserExactDistributionConvergenceMatrix>>;
