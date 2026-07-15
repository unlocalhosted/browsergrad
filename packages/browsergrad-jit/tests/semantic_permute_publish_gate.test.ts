import { describe, expect, it } from "vitest";

import { validateSemanticPermutePublishGate } from "../scripts/require_semantic_permute_publish_gate.mjs";

const HEAD = "1111111111111111111111111111111111111111";

describe("JIT semantic-permute publish evidence gate", () => {
  it("rejects missing or stale evidence commits", () => {
    expect(() => validate(undefined)).toThrow(/test:browser:semantic-permute:required/u);
    expect(() => validate("0000000000000000000000000000000000000000"))
      .toThrow(/does not match HEAD/u);
  });

  it("accepts only the exact clean evidenced commit", () => {
    expect(() => validate(HEAD)).not.toThrow();
    expect(() => validate(HEAD, " M packages/browsergrad-jit/src/python/_gpu_plan.py"))
      .toThrow(/differ from evidenced HEAD/u);
    expect(() => validate(HEAD, "", "2222222222222222222222222222222222222222"))
      .toThrow(/GitHub SHA/u);
  });
});

function validate(evidenceCommit: string | undefined, relevantStatus = "", githubSha?: string) {
  return validateSemanticPermutePublishGate({
    evidenceCommit,
    githubSha,
    head: HEAD,
    relevantStatus,
  });
}
