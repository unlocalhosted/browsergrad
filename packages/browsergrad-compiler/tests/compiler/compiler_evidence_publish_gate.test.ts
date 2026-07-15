import { describe, expect, it } from "vitest";
import { validateCompilerEvidencePublishGate } from "../../scripts/require_compiler_evidence_publish_gate.mjs";

const head = "1111111111111111111111111111111111111111";

describe("combined compiler publish evidence gate", () => {
  it("rejects missing or stale evidence commits", () => {
    expect(() => validate(undefined, head)).toThrow(/test:browser:layout-bindings:required/u);
    expect(() => validate("0000000000000000000000000000000000000000", head))
      .toThrow(/layout-binding evidence commit .* does not match HEAD/u);
    expect(() => validate(head, undefined)).toThrow(/test:browser:view-copy-bindings:required/u);
    expect(() => validate(head, "0000000000000000000000000000000000000000"))
      .toThrow(/view-copy-binding evidence commit .* does not match HEAD/u);
  });

  it("accepts only the exact clean evidenced commit", () => {
    expect(() => validate(head, head)).not.toThrow();
    expect(() => validate(head, head, " M packages/browsergrad-compiler/src/runner.ts"))
      .toThrow(/differ from evidenced HEAD/u);
    expect(() => validate(head, head, "", "2222222222222222222222222222222222222222"))
      .toThrow(/GitHub SHA/u);
  });
});

function validate(
  layoutEvidenceCommit: string | undefined,
  viewCopyEvidenceCommit: string | undefined,
  relevantStatus = "",
  githubSha?: string,
) {
  return validateCompilerEvidencePublishGate({
    layoutEvidenceCommit,
    viewCopyEvidenceCommit,
    githubSha,
    head,
    relevantStatus,
  });
}
