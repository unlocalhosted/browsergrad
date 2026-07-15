import { describe, expect, it, vi } from "vitest";
import {
  EXECUTION_EVIDENCE_SCHEMA,
  createTerminalEvidenceEmitter,
  validateTerminalExecutionEvidence,
} from "../../../test-support/webgpu-evidence";

const EXPECTED = Object.freeze({
  suiteId: "browsergrad.test.webgpu-evidence@1",
  capabilityId: "browsergrad.test.capability",
  backendId: "browsergrad.backend.webgpu.core",
  comparisonPolicyId: "browsergrad.comparison.test.v1",
  requireDeviceProfile: true,
});

function terminal(overrides: Readonly<Record<string, unknown>> = {}) {
  return {
    schema: EXECUTION_EVIDENCE_SCHEMA,
    kind: "terminal",
    suiteId: EXPECTED.suiteId,
    required: true,
    evidence: {
      capabilityId: EXPECTED.capabilityId,
      artifactHash: "a".repeat(64),
      backendId: EXPECTED.backendId,
      environmentId: "b".repeat(64),
      producerVersions: { package: "1.0.0" },
      deviceProfileHash: "c".repeat(64),
      recordedAt: "2026-07-15T00:00:00.000Z",
      outcome: "passed",
      comparisonPolicyId: EXPECTED.comparisonPolicyId,
      diagnosticCodes: [],
      ...overrides,
    },
  };
}

describe("shared WebGPU terminal evidence contract", () => {
  it("accepts a complete required-device pass", () => {
    expect(() => validateTerminalExecutionEvidence(terminal(), EXPECTED)).not.toThrow();
  });

  it("rejects false-green required and passed records", () => {
    expect(() => validateTerminalExecutionEvidence(terminal({
      outcome: "not-run",
      diagnosticCodes: ["BG-WEBGPU-EVIDENCE-DEVICE-UNAVAILABLE"],
    }), EXPECTED)).toThrow(/required evidence cannot report not-run/u);
    expect(() => validateTerminalExecutionEvidence(terminal({ deviceProfileHash: undefined }), EXPECTED))
      .toThrow(/requires a profile hash/u);
    expect(() => validateTerminalExecutionEvidence(terminal({ diagnosticCodes: ["unexpected"] }), EXPECTED))
      .toThrow(/passed evidence cannot contain diagnostics/u);
  });

  it("allows exactly one validated terminal emission", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    try {
      const emitter = createTerminalEvidenceEmitter("[evidence]", EXPECTED);
      emitter.emit(terminal());
      expect(emitter.emitted).toBe(true);
      expect(warn).toHaveBeenCalledTimes(1);
      expect(() => emitter.emit(terminal())).toThrow(/more than once/u);
    } finally {
      warn.mockRestore();
    }
  });
});
