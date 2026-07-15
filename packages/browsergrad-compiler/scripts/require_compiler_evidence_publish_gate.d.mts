export interface CompilerEvidencePublishGateInput {
  readonly layoutEvidenceCommit?: string | undefined;
  readonly viewCopyEvidenceCommit?: string | undefined;
  readonly githubSha?: string | undefined;
  readonly head: string;
  readonly relevantStatus: string;
}

export function validateCompilerEvidencePublishGate(input: CompilerEvidencePublishGateInput): void;
