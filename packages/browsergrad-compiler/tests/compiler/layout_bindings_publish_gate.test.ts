import { describe, expect, it } from "vitest";
import { validateLayoutBindingsPublishGate } from "../../scripts/require_layout_bindings_publish_gate.mjs";

const head = "1111111111111111111111111111111111111111";

describe("compiler layout-binding publish evidence gate", () => {
  it("rejects missing or stale evidence commits", () => {
    expect(() => validate(undefined)).toThrow(/test:browser:layout-bindings:required/u);
    expect(() => validate("0000000000000000000000000000000000000000")).toThrow(/does not match HEAD/u);
  });

  it("accepts only the exact clean evidenced commit", () => {
    expect(() => validate(head)).not.toThrow();
    expect(() => validate(head, " M packages/browsergrad-compiler/src/runner.ts")).toThrow(/differ from evidenced HEAD/u);
    expect(() => validate(head, "", "2222222222222222222222222222222222222222")).toThrow(/GitHub SHA/u);
  });
});

function validate(evidenceCommit: string | undefined, relevantStatus = "", githubSha?: string) {
  return validateLayoutBindingsPublishGate({ evidenceCommit, githubSha, head, relevantStatus });
}
