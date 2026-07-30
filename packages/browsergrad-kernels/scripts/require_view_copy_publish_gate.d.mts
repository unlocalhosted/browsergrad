export interface ViewCopyPublishGateInput {
  readonly evidenceCommit: string | undefined;
  readonly semanticHostGraphEvidenceCommit: string | undefined;
  readonly semanticHostGraphWorkerEvidenceCommit: string | undefined;
  readonly semanticHostGraphPerformanceEvidenceCommit: string | undefined;
  readonly semanticGemmEvidenceCommit: string | undefined;
  readonly semanticAttentionEvidenceCommit: string | undefined;
  readonly semanticAttentionPerformanceEvidenceCommit: string | undefined;
  readonly jitEvidenceCommit: string | undefined;
  readonly githubSha: string | undefined;
  readonly head: string;
  readonly relevantStatus: string;
}

export function validateViewCopyPublishGate(input: ViewCopyPublishGateInput): void;
