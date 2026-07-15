export const SEMANTIC_PERMUTE_EVIDENCE_PREFIX: string;

export interface SemanticPermuteEvidenceLogOptions {
  readonly expectedSourceRevision: string;
  readonly gitHead: string;
  readonly relevantStatus: string;
  readonly producerVersions: Readonly<Record<string, string>>;
}

export function verifySemanticPermuteEvidenceLog(
  log: string,
  options: SemanticPermuteEvidenceLogOptions,
): Promise<Record<string, unknown>>;

export function loadSemanticPermuteProducerVersions(
  repositoryRoot: URL,
): Readonly<Record<string, string>>;

export function readBoundedSemanticPermuteEvidenceLog(logPath: string): string;
