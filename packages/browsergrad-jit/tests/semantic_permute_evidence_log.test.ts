import { describe, expect, it } from "vitest";
import { execFileSync } from "node:child_process";
import {
  closeSync,
  ftruncateSync,
  mkdtempSync,
  openSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  SEMANTIC_PERMUTE_EVIDENCE_PREFIX,
  loadSemanticPermuteProducerVersions,
  readBoundedSemanticPermuteEvidenceLog,
  verifySemanticPermuteEvidenceLog,
} from "../scripts/verify_semantic_permute_evidence_log.mjs";
import {
  terminalManifestHashFor,
  type TerminalEvidenceRecord,
  type UnsignedTerminalEvidenceRecord,
} from "../tests-browser/semantic_permute_evidence";

const SOURCE_REVISION = "1".repeat(40);
const PRODUCER_VERSIONS = loadSemanticPermuteProducerVersions(
  new URL("../../../", import.meta.url),
);
const OPTIONS = Object.freeze({
  expectedSourceRevision: SOURCE_REVISION,
  gitHead: SOURCE_REVISION,
  relevantStatus: "",
  producerVersions: PRODUCER_VERSIONS,
});

describe("retained JIT semantic-permute evidence log", () => {
  it("accepts exactly one compact required pass bound to source and whole-record hash", async () => {
    const record = await passedRecord();
    const retainedReporterLine = `20:00:00 [vite] (client) [console.warn] ${log(record)}\r\n`;
    await expect(verifySemanticPermuteEvidenceLog(retainedReporterLine, OPTIONS))
      .resolves.toMatchObject({ required: true, stage: "terminal-summary" });
  });

  it("rejects duplicate prefixes and any trailing record text", async () => {
    const record = await passedRecord();
    await expect(verifySemanticPermuteEvidenceLog(`${log(record)}\n${log(record)}`, OPTIONS))
      .rejects.toThrow(/exactly one literal prefixed terminal record/u);
    await expect(verifySemanticPermuteEvidenceLog(`${log(record)} trailing`, OPTIONS))
      .rejects.toThrow(/exact JSON|compact JSON/u);
    await expect(verifySemanticPermuteEvidenceLog("browser test passed", OPTIONS))
      .rejects.toThrow(/found 0/u);
    await expect(verifySemanticPermuteEvidenceLog(
      `${SEMANTIC_PERMUTE_EVIDENCE_PREFIX}{not-json}`,
      OPTIONS,
    )).rejects.toThrow(/not exact JSON/u);
    await expect(verifySemanticPermuteEvidenceLog("x".repeat((8 * 1024 * 1024) + 1), OPTIONS))
      .rejects.toThrow(/exceeds/u);
  });

  it("rejects an oversized retained file before reading its contents", () => {
    const directory = mkdtempSync(join(tmpdir(), "browsergrad-jit-evidence-"));
    const path = join(directory, "oversized.log");
    const descriptor = openSync(path, "w");
    try {
      ftruncateSync(descriptor, (8 * 1024 * 1024) + 1);
    } finally {
      closeSync(descriptor);
    }
    try {
      expect(() => readBoundedSemanticPermuteEvidenceLog(path)).toThrow(/exceeds/u);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("rejects a non-regular retained log without reading it", () => {
    const directory = mkdtempSync(join(tmpdir(), "browsergrad-jit-evidence-"));
    const path = join(directory, "evidence.fifo");
    try {
      execFileSync("mkfifo", [path]);
      expect(() => readBoundedSemanticPermuteEvidenceLog(path)).toThrow(/regular file/u);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("rejects resealed advisory, source, and producer-version drift", async () => {
    const record = await passedRecord();
    const advisory = await reseal({ ...record, required: false });
    await expect(verifySemanticPermuteEvidenceLog(log(advisory), OPTIONS))
      .rejects.toThrow(/required must equal true/u);

    const foreignSource = await reseal({
      ...record,
      evidence: { ...record.evidence, sourceRevision: "2".repeat(40) },
    });
    await expect(verifySemanticPermuteEvidenceLog(log(foreignSource), OPTIONS))
      .rejects.toThrow(/evidence.sourceRevision/u);

    const foreignProducer = await reseal({
      ...record,
      evidence: {
        ...record.evidence,
        producerVersions: {
          ...record.evidence.producerVersions,
          "@unlocalhosted/browsergrad-jit": "9.9.9",
        },
      },
    });
    await expect(verifySemanticPermuteEvidenceLog(log(foreignProducer), OPTIONS))
      .rejects.toThrow(/producerVersions differ/u);

    const extraProducer = await reseal({
      ...record,
      evidence: {
        ...record.evidence,
        producerVersions: { ...record.evidence.producerVersions, extra: "1.0.0" },
      },
    });
    await expect(verifySemanticPermuteEvidenceLog(log(extraProducer), OPTIONS))
      .rejects.toThrow(/producerVersions differ/u);
  });

  it("rejects resealed wrong IDs and non-passed outcomes", async () => {
    const record = await passedRecord();
    const wrongSuite = await reseal({
      ...record,
      suiteId: "wrong-suite",
    } as unknown as TerminalEvidenceRecord);
    await expect(verifySemanticPermuteEvidenceLog(log(wrongSuite), OPTIONS))
      .rejects.toThrow(/suiteId/u);
    const failed = await reseal({
      ...record,
      evidence: { ...record.evidence, outcome: "failed" },
    });
    await expect(verifySemanticPermuteEvidenceLog(log(failed), OPTIONS))
      .rejects.toThrow(/evidence.outcome/u);
  });

  it("rejects a mutated whole-record digest and expected SHA not equal to git HEAD", async () => {
    const record = await passedRecord();
    await expect(verifySemanticPermuteEvidenceLog(log({
      ...record,
      terminalManifestHash: "f".repeat(64),
    }), OPTIONS)).rejects.toThrow(/does not bind/u);
    await expect(verifySemanticPermuteEvidenceLog(log(record), {
      ...OPTIONS,
      gitHead: "2".repeat(40),
    })).rejects.toThrow(/differs from git HEAD/u);
    await expect(verifySemanticPermuteEvidenceLog(log(record), {
      ...OPTIONS,
      relevantStatus: " M packages/browsergrad-jit/src/index.ts",
    })).rejects.toThrow(/relevant source differs/u);
  });
});

async function passedRecord(): Promise<TerminalEvidenceRecord> {
  const unsigned = {
    schema: "browsergrad.execution-evidence@1",
    kind: "terminal",
    suiteId: "browsergrad.jit.semantic-permute.webgpu-conformance@1",
    required: true,
    evidence: {
      capabilityId: "browsergrad.jit.tensor-plan.semantic-permute",
      artifactHash: "a".repeat(64),
      backendId: "browsergrad.backend.webgpu.core",
      environmentId: "b".repeat(64),
      producerVersions: PRODUCER_VERSIONS,
      sourceRevision: SOURCE_REVISION,
      deviceProfileHash: "c".repeat(64),
      recordedAt: "2026-07-15T00:00:00.000Z",
      outcome: "passed",
      comparisonPolicyId: "browsergrad.comparison.bit-exact-u32-complete-root.v1",
      diagnosticCodes: [],
    },
    environment: {},
    artifactHashKind: "prepared-case-set",
    plannedCaseIds: ["rank2-transpose", "rank3-permutation"],
    completedCases: [
      { caseId: "rank2-transpose" },
      { caseId: "rank3-permutation" },
    ],
    stage: "terminal-summary",
    uncapturedErrors: [],
  } as unknown as UnsignedTerminalEvidenceRecord;
  return {
    ...unsigned,
    terminalManifestHash: await terminalManifestHashFor(unsigned),
  } as TerminalEvidenceRecord;
}

async function reseal(
  record: TerminalEvidenceRecord | UnsignedTerminalEvidenceRecord,
): Promise<TerminalEvidenceRecord> {
  const { terminalManifestHash: _excluded, ...unsigned } = record as TerminalEvidenceRecord;
  return {
    ...unsigned,
    terminalManifestHash: await terminalManifestHashFor(
      unsigned as UnsignedTerminalEvidenceRecord,
    ),
  } as TerminalEvidenceRecord;
}

function log(record: unknown): string {
  return `${SEMANTIC_PERMUTE_EVIDENCE_PREFIX}${JSON.stringify(record)}`;
}
