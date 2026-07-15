export interface SemanticPermutePublishGateInput {
  readonly evidenceCommit: string | undefined;
  readonly githubSha: string | undefined;
  readonly head: string;
  readonly relevantStatus: string;
}

export function validateSemanticPermutePublishGate(input: SemanticPermutePublishGateInput): void;
