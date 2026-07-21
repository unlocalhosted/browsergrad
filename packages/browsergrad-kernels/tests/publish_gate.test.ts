import { describe, expect, it } from "vitest";
import { validateViewCopyPublishGate } from "../scripts/require_view_copy_publish_gate.mjs";

const head = "1111111111111111111111111111111111111111";

describe("kernels publish evidence gate", () => {
  it("rejects missing or stale evidence commits", () => {
    expect(() => validate(undefined)).toThrow(/run test:browser:view-copy:required/u);
    expect(() => validate("0000000000000000000000000000000000000000")).toThrow(/does not match HEAD/u);
    expect(() => validateViewCopyPublishGate({
      evidenceCommit: head,
      semanticGemmEvidenceCommit: head,
      jitEvidenceCommit: undefined,
      githubSha: undefined,
      head,
      relevantStatus: "",
    })).toThrow(/semantic-permute:required/u);
    expect(() => validateViewCopyPublishGate({
      evidenceCommit: head,
      semanticGemmEvidenceCommit: undefined,
      jitEvidenceCommit: head,
      githubSha: undefined,
      head,
      relevantStatus: "",
    })).toThrow(/semantic-gemm:required/u);
    expect(() => validate(head, "", undefined, head, "0000000000000000000000000000000000000000"))
      .toThrow(/semantic GEMM evidence commit/u);
    expect(() => validate(head, "", undefined, "0000000000000000000000000000000000000000"))
      .toThrow(/JIT semantic-permute evidence commit/u);
  });

  it("accepts only the exact clean evidenced commit", () => {
    expect(() => validate(head)).not.toThrow();
    expect(() => validate(head, " M packages/browsergrad-kernels/src/index.ts")).toThrow(/differ from evidenced HEAD/u);
    expect(() => validate(head, "", "2222222222222222222222222222222222222222")).toThrow(/GitHub SHA/u);
  });
});

function validate(
  evidenceCommit: string | undefined,
  relevantStatus = "",
  githubSha?: string,
  jitEvidenceCommit: string | undefined = evidenceCommit,
  semanticGemmEvidenceCommit: string | undefined = evidenceCommit,
) {
  return validateViewCopyPublishGate({
    evidenceCommit,
    semanticGemmEvidenceCommit,
    jitEvidenceCommit,
    githubSha,
    head,
    relevantStatus,
  });
}
