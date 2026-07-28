import { describe, expect, it } from "vitest";
import { validateViewCopyPublishGate } from "../scripts/require_view_copy_publish_gate.mjs";

const head = "1111111111111111111111111111111111111111";

describe("kernels publish evidence gate", () => {
  it("rejects missing or stale evidence commits", () => {
    expect(() => validate(undefined)).toThrow(/run test:browser:view-copy:required/u);
    expect(() => validate("0000000000000000000000000000000000000000")).toThrow(/does not match HEAD/u);
    expect(() => validateViewCopyPublishGate({
      evidenceCommit: head,
      semanticHostGraphEvidenceCommit: head,
      semanticHostGraphPerformanceEvidenceCommit: head,
      semanticGemmEvidenceCommit: head,
      semanticAttentionEvidenceCommit: head,
      semanticAttentionPerformanceEvidenceCommit: head,
      jitEvidenceCommit: undefined,
      githubSha: undefined,
      head,
      relevantStatus: "",
    })).toThrow(/semantic-permute:required/u);
    expect(() => validateViewCopyPublishGate({
      evidenceCommit: head,
      semanticHostGraphEvidenceCommit: head,
      semanticHostGraphPerformanceEvidenceCommit: head,
      semanticGemmEvidenceCommit: undefined,
      semanticAttentionEvidenceCommit: head,
      semanticAttentionPerformanceEvidenceCommit: head,
      jitEvidenceCommit: head,
      githubSha: undefined,
      head,
      relevantStatus: "",
    })).toThrow(/semantic-gemm:required/u);
    expect(() => validateViewCopyPublishGate({
      evidenceCommit: head,
      semanticHostGraphEvidenceCommit: head,
      semanticHostGraphPerformanceEvidenceCommit: head,
      semanticGemmEvidenceCommit: head,
      semanticAttentionEvidenceCommit: undefined,
      semanticAttentionPerformanceEvidenceCommit: head,
      jitEvidenceCommit: head,
      githubSha: undefined,
      head,
      relevantStatus: "",
    })).toThrow(/semantic-attention:required/u);
    expect(() => validateViewCopyPublishGate({
      evidenceCommit: head,
      semanticHostGraphEvidenceCommit: head,
      semanticHostGraphPerformanceEvidenceCommit: undefined,
      semanticGemmEvidenceCommit: head,
      semanticAttentionEvidenceCommit: head,
      semanticAttentionPerformanceEvidenceCommit: head,
      jitEvidenceCommit: head,
      githubSha: undefined,
      head,
      relevantStatus: "",
    })).toThrow(/semantic-host-graph:performance:required/u);
    expect(() => validateViewCopyPublishGate({
      evidenceCommit: head,
      semanticHostGraphEvidenceCommit: head,
      semanticHostGraphPerformanceEvidenceCommit: head,
      semanticGemmEvidenceCommit: head,
      semanticAttentionEvidenceCommit: head,
      semanticAttentionPerformanceEvidenceCommit: undefined,
      jitEvidenceCommit: head,
      githubSha: undefined,
      head,
      relevantStatus: "",
    })).toThrow(/semantic-attention:performance:required/u);
    expect(() => validateViewCopyPublishGate({
      evidenceCommit: head,
      semanticHostGraphEvidenceCommit: undefined,
      semanticHostGraphPerformanceEvidenceCommit: head,
      semanticGemmEvidenceCommit: head,
      semanticAttentionEvidenceCommit: head,
      semanticAttentionPerformanceEvidenceCommit: head,
      jitEvidenceCommit: head,
      githubSha: undefined,
      head,
      relevantStatus: "",
    })).toThrow(/semantic-host-graph:required/u);
    expect(() => validate(head, "", undefined, head, "0000000000000000000000000000000000000000"))
      .toThrow(/semantic GEMM evidence commit/u);
    expect(() => validate(
      head,
      "",
      undefined,
      head,
      head,
      "0000000000000000000000000000000000000000",
    )).toThrow(/semantic attention evidence commit/u);
    expect(() => validate(
      head,
      "",
      undefined,
      head,
      head,
      head,
      "0000000000000000000000000000000000000000",
    )).toThrow(/semantic attention performance evidence commit/u);
    expect(() => validate(head, "", undefined, "0000000000000000000000000000000000000000"))
      .toThrow(/JIT semantic-permute evidence commit/u);
    expect(() => validate(
      head,
      "",
      undefined,
      head,
      head,
      head,
      head,
      "0000000000000000000000000000000000000000",
    )).toThrow(/semantic host-graph evidence commit/u);
    expect(() => validate(
      head,
      "",
      undefined,
      head,
      head,
      head,
      head,
      head,
      "0000000000000000000000000000000000000000",
    )).toThrow(/semantic host-graph performance evidence commit/u);
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
  semanticAttentionEvidenceCommit: string | undefined = evidenceCommit,
  semanticAttentionPerformanceEvidenceCommit: string | undefined = evidenceCommit,
  semanticHostGraphEvidenceCommit: string | undefined = evidenceCommit,
  semanticHostGraphPerformanceEvidenceCommit: string | undefined = evidenceCommit,
) {
  return validateViewCopyPublishGate({
    evidenceCommit,
    semanticHostGraphEvidenceCommit,
    semanticHostGraphPerformanceEvidenceCommit,
    semanticGemmEvidenceCommit,
    semanticAttentionEvidenceCommit,
    semanticAttentionPerformanceEvidenceCommit,
    jitEvidenceCommit,
    githubSha,
    head,
    relevantStatus,
  });
}
