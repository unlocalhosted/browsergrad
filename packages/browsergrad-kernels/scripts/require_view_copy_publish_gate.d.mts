export interface ViewCopyPublishGateInput {
  readonly evidenceCommit: string | undefined;
  readonly jitEvidenceCommit: string | undefined;
  readonly githubSha: string | undefined;
  readonly head: string;
  readonly relevantStatus: string;
}

export function validateViewCopyPublishGate(input: ViewCopyPublishGateInput): void;
