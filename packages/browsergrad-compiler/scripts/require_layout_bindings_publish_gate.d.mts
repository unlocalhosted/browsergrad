export interface LayoutBindingsPublishGateInput {
  readonly evidenceCommit?: string | undefined;
  readonly githubSha?: string | undefined;
  readonly head: string;
  readonly relevantStatus: string;
}

export function validateLayoutBindingsPublishGate(input: LayoutBindingsPublishGateInput): void;
