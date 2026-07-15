import { execFileSync } from "node:child_process";
import {
  closeSync,
  ftruncateSync,
  mkdtempSync,
  openSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  hashCanonicalJson,
  hashNamedComponents,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  COMPILER_VIEW_COPY_EVIDENCE_SOURCE_PATHS,
} from "../../../../scripts/compiler-view-copy-evidence-source.mjs";
import {
  COMPILER_VIEW_COPY_EVIDENCE_PREFIX,
  deriveCompilerViewCopyBindingsExpectedEvidence,
  loadCompilerViewCopyProducerVersions,
  readBoundedCompilerViewCopyEvidenceLog,
  verifyCompilerViewCopyBindingsEvidenceLog,
  type CompilerViewCopyPreparedCaseManifest,
} from "../../scripts/verify_view_copy_bindings_evidence_log.mjs";

const SOURCE_REVISION = "1".repeat(40);
const TERMINAL_HASH_DOMAIN = "browsergrad.compiler.view-copy-bindings.terminal-manifest.v1";
const SUITE_ID = "browsergrad.compiler.view-copy-bindings.webgpu-conformance@1";
const BACKEND_ID = "browsergrad.backend.webgpu.core";
const COMPARISON_POLICY_ID = "browsergrad.comparison.bit-exact-u32-complete-root.v1";
const CASE_IDS = Object.freeze([
  "rank2-transpose-control",
  "rank2-padding-exact-nan",
  "rank3-padding-exact-nan",
]);
const PRODUCER_VERSIONS = loadCompilerViewCopyProducerVersions(
  new URL("../../../../", import.meta.url),
);
const OPTIONS = Object.freeze({
  expectedSourceRevision: SOURCE_REVISION,
  gitHead: SOURCE_REVISION,
  relevantStatus: "",
  producerVersions: PRODUCER_VERSIONS,
});

interface TestTerminalEvidence extends JsonObject {
  readonly capabilityId: string;
  readonly artifactHash: string;
  readonly backendId: string;
  readonly environmentId: string;
  readonly producerVersions: Readonly<Record<string, string>>;
  readonly sourceRevision: string;
  readonly deviceProfileHash: string;
  readonly recordedAt: string;
  readonly outcome: string;
  readonly comparisonPolicyId: string;
  readonly diagnosticCodes: readonly string[];
}

interface TestEnvironment extends JsonObject {
  readonly schema: string;
  readonly acquisition: string;
  readonly userAgent: string;
  readonly platform: string;
  readonly adapter: JsonObject;
  readonly adapterSupportedFeatures: readonly string[];
  readonly negotiatedDeviceFeatures: readonly string[];
  readonly negotiatedDeviceLimits: JsonObject;
}

interface TestPreparedCase extends CompilerViewCopyPreparedCaseManifest, JsonObject {}

interface TestObservation extends TestPreparedCase {
  readonly actualSourceHash: string;
  readonly actualDestinationHash: string;
  readonly planKind: string;
  readonly stepCount: number;
  readonly plannedPipelineCount: number;
  readonly comparisonPolicyId: string;
}

interface TestUnsignedTerminalRecord extends JsonObject {
  readonly schema: string;
  readonly kind: string;
  readonly suiteId: string;
  readonly required: boolean;
  readonly evidence: TestTerminalEvidence;
  readonly environment: TestEnvironment;
  readonly artifactHashKind: string;
  readonly preparedBackendArtifactHash: string;
  readonly caseSetHash: string;
  readonly preparedCases: readonly TestPreparedCase[];
  readonly plannedCaseIds: readonly string[];
  readonly completedCases: readonly TestObservation[];
  readonly stage: string;
  readonly uncapturedErrors: readonly string[];
  readonly currentCaseId?: string;
  readonly error?: JsonObject;
}

interface TestTerminalRecord extends TestUnsignedTerminalRecord {
  readonly terminalManifestHash: string;
}

describe("retained compiler view-copy-bindings evidence log", () => {
  it("covers the complete source authority surface and exact producer packages", () => {
    expect(COMPILER_VIEW_COPY_EVIDENCE_SOURCE_PATHS).toEqual([
      "packages/browsergrad-compiler",
      "packages/browsergrad-kernels",
      "packages/browsergrad-semantic-core",
      "architecture/semantic-fixture-contracts.json",
      "test-support/view-copy-conformance-fixtures.ts",
      "test-support/webgpu-evidence.ts",
      "scripts/compiler-view-copy-evidence-source.mjs",
      "scripts/compiler-view-copy-evidence-source.d.mts",
      "package.json",
      "pnpm-lock.yaml",
      "pnpm-workspace.yaml",
      "tsconfig.base.json",
    ]);
    expect(Object.keys(PRODUCER_VERSIONS)).toEqual([
      "@unlocalhosted/browsergrad-compiler",
      "@unlocalhosted/browsergrad-kernels",
      "@unlocalhosted/browsergrad-semantic-core",
    ]);
    expect(Object.values(PRODUCER_VERSIONS).every((version) => version.length > 0)).toBe(true);
  });

  it("accepts exactly one compact required pass bound to source and whole-record hash", async () => {
    const record = await passedRecord();
    const retainedReporterLine = `20:00:00 [vite] (client) [console.warn] ${log(record)}\r\n`;
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(retainedReporterLine, OPTIONS))
      .resolves.toMatchObject({ required: true, stage: "terminal-summary" });
  });

  it("rejects duplicate prefixes, trailing record text, malformed JSON, and oversized text", async () => {
    const record = await passedRecord();
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(`${log(record)}\n${log(record)}`, OPTIONS))
      .rejects.toThrow(/exactly one literal prefixed terminal record/u);
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(`${log(record)} trailing`, OPTIONS))
      .rejects.toThrow(/exact JSON|compact JSON/u);
    await expect(verifyCompilerViewCopyBindingsEvidenceLog("browser test passed", OPTIONS))
      .rejects.toThrow(/found 0/u);
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(
      `${COMPILER_VIEW_COPY_EVIDENCE_PREFIX}{not-json}`,
      OPTIONS,
    )).rejects.toThrow(/not exact JSON/u);
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(
      "x".repeat((8 * 1024 * 1024) + 1),
      OPTIONS,
    )).rejects.toThrow(/exceeds/u);
  });

  it("rejects oversized and non-regular retained files before reading their contents", () => {
    const directory = mkdtempSync(join(tmpdir(), "browsergrad-compiler-view-copy-evidence-"));
    const stablePath = join(directory, "stable.log");
    writeFileSync(stablePath, "stable retained evidence", "utf8");
    const oversizedPath = join(directory, "oversized.log");
    const descriptor = openSync(oversizedPath, "w");
    try {
      ftruncateSync(descriptor, (8 * 1024 * 1024) + 1);
    } finally {
      closeSync(descriptor);
    }
    const fifoPath = join(directory, "evidence.fifo");
    try {
      execFileSync("mkfifo", [fifoPath]);
      expect(readBoundedCompilerViewCopyEvidenceLog(stablePath)).toBe("stable retained evidence");
      expect(() => readBoundedCompilerViewCopyEvidenceLog(oversizedPath)).toThrow(/exceeds/u);
      expect(() => readBoundedCompilerViewCopyEvidenceLog(fifoPath)).toThrow(/regular file/u);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("rejects resealed advisory, source, and producer-version drift", async () => {
    const record = await passedRecord();
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(
      log(await reseal({ ...record, required: false })),
      OPTIONS,
    )).rejects.toThrow(/required must equal true/u);

    const foreignSource = await reseal({
      ...record,
      evidence: { ...record.evidence, sourceRevision: "2".repeat(40) },
    });
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(foreignSource), OPTIONS))
      .rejects.toThrow(/evidence.sourceRevision/u);

    const foreignProducer = await reseal({
      ...record,
      evidence: {
        ...record.evidence,
        producerVersions: {
          ...record.evidence.producerVersions,
          "@unlocalhosted/browsergrad-compiler": "9.9.9",
        },
      },
    });
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(foreignProducer), OPTIONS))
      .rejects.toThrow(/producerVersions differ/u);

    const extraProducer = await reseal({
      ...record,
      evidence: {
        ...record.evidence,
        producerVersions: { ...record.evidence.producerVersions, extra: "1.0.0" },
      },
    });
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(extraProducer), OPTIONS))
      .rejects.toThrow(/producerVersions differ/u);
  });

  it("rejects resealed wrong identity, non-passed outcomes, and incomplete cases", async () => {
    const record = await passedRecord();
    const mutations: readonly [TestTerminalRecord, RegExp][] = [
      [await reseal({ ...record, suiteId: "wrong-suite" }), /suiteId/u],
      [await reseal({ ...record, stage: "case-execution" }), /stage/u],
      [await reseal({ ...record, artifactHashKind: "planned-suite-manifest" }), /artifactHashKind/u],
      [await reseal({
        ...record,
        evidence: { ...record.evidence, capabilityId: "wrong-capability" },
      }), /capabilityId/u],
      [await reseal({
        ...record,
        evidence: { ...record.evidence, backendId: "wrong-backend" },
      }), /backendId/u],
      [await reseal({
        ...record,
        evidence: { ...record.evidence, comparisonPolicyId: "wrong-policy" },
      }), /comparisonPolicyId/u],
      [await reseal({
        ...record,
        evidence: { ...record.evidence, outcome: "failed" },
      }), /evidence.outcome/u],
      [await reseal({ ...record, plannedCaseIds: [...CASE_IDS].reverse() }), /plannedCaseIds/u],
      [await reseal({ ...record, completedCases: record.completedCases.slice(0, 2) }), /completedCases/u],
    ];
    for (const [mutation, expected] of mutations) {
      await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(mutation), OPTIONS))
        .rejects.toThrow(expected);
    }
  });

  it("rejects resealed diagnostics/errors plus digest, HEAD, and relevant-source drift", async () => {
    const record = await passedRecord();
    const diagnostics = await reseal({
      ...record,
      evidence: { ...record.evidence, diagnosticCodes: ["BG-COMPILER-FAILED"] },
    });
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(diagnostics), OPTIONS))
      .rejects.toThrow(/diagnosticCodes must be empty/u);
    const uncaptured = await reseal({ ...record, uncapturedErrors: ["validation error"] });
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(uncaptured), OPTIONS))
      .rejects.toThrow(/uncapturedErrors must be empty/u);
    const error = await reseal({ ...record, error: { message: "failed" } });
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(error), OPTIONS))
      .rejects.toThrow(/cannot carry currentCaseId or error/u);

    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log({
      ...record,
      terminalManifestHash: "f".repeat(64),
    }), OPTIONS)).rejects.toThrow(/does not bind/u);
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(record), {
      ...OPTIONS,
      gitHead: "2".repeat(40),
    })).rejects.toThrow(/differs from git HEAD/u);
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(record), {
      ...OPTIONS,
      relevantStatus: " M packages/browsergrad-compiler/src/index.ts",
    })).rejects.toThrow(/relevant source differs/u);
  });

  it("rejects the former compact synthetic pass and incomplete passed schemas", async () => {
    const record = await passedRecord();
    const compactForgery = await resealUnchecked({
      schema: record.schema,
      kind: record.kind,
      suiteId: record.suiteId,
      required: record.required,
      evidence: {
        ...record.evidence,
        artifactHash: "a".repeat(64),
        environmentId: "b".repeat(64),
        deviceProfileHash: "c".repeat(64),
      },
      environment: {},
      artifactHashKind: record.artifactHashKind,
      plannedCaseIds: record.plannedCaseIds,
      completedCases: CASE_IDS.map((caseId) => ({ caseId })),
      stage: record.stage,
      uncapturedErrors: [],
    });
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(compactForgery), OPTIONS))
      .rejects.toThrow(/exact keys/u);

    for (const key of [
      "preparedBackendArtifactHash",
      "caseSetHash",
      "preparedCases",
    ] as const) {
      const unsigned = omitKeys(record, "terminalManifestHash", key);
      await expect(verifyCompilerViewCopyBindingsEvidenceLog(
        log(await resealUnchecked(unsigned)),
        OPTIONS,
      )).rejects.toThrow(/exact keys/u);
    }
    const missingObservationFacts = {
      ...record,
      completedCases: CASE_IDS.map((caseId) => ({ caseId })),
    } as unknown as TestUnsignedTerminalRecord;
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(
      log(await reseal(missingObservationFacts)),
      OPTIONS,
    )).rejects.toThrow(/completedCases\[0\].*exact keys/u);
  });

  it("rejects fully resealed source-independent manifest and output forgeries", async () => {
    const record = await passedRecord();
    const topologyForgery = await resealPreparedForgery(record, 0, {
      logicalShape: [1, 6],
    });
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(topologyForgery), OPTIONS))
      .rejects.toThrow(/preparedBackendArtifactHash|source-derived manifest/u);

    const wgslForgery = await resealPreparedForgery(record, 1, {
      wgslModuleHash: "d".repeat(64),
    });
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(wgslForgery), OPTIONS))
      .rejects.toThrow(/preparedBackendArtifactHash|source-derived manifest/u);

    const outputForgery = await resealPreparedForgery(record, 2, {
      expectedDestinationHash: "e".repeat(64),
    });
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(outputForgery), OPTIONS))
      .rejects.toThrow(/preparedBackendArtifactHash|source-derived manifest/u);
  });

  it("rejects noncanonical timestamps, incomplete environments, and unknown passed fields", async () => {
    const record = await passedRecord();
    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(await reseal({
      ...record,
      evidence: { ...record.evidence, recordedAt: "2026-07-16" },
    })), OPTIONS)).rejects.toThrow(/canonical ISO timestamp/u);

    await expect(verifyCompilerViewCopyBindingsEvidenceLog(log(await reseal({
      ...record,
      environment: {
        ...record.environment,
        negotiatedDeviceFeatures: ["not-supported-by-adapter"],
      },
    })), OPTIONS)).rejects.toThrow(/subset/u);

    const unknownMutations = [
      { ...record, assertedByRetainedLine: true },
      { ...record, evidence: { ...record.evidence, attested: true } },
      { ...record, environment: { ...record.environment, gpuWasReal: true } },
      {
        ...record,
        preparedCases: [
          { ...record.preparedCases[0]!, verified: true },
          ...record.preparedCases.slice(1),
        ],
      },
      {
        ...record,
        completedCases: [
          { ...record.completedCases[0]!, submittedWorkgroupCount: 1 },
          ...record.completedCases.slice(1),
        ],
      },
    ];
    for (const mutation of unknownMutations) {
      await expect(verifyCompilerViewCopyBindingsEvidenceLog(
        log(await resealUnchecked(mutation)),
        OPTIONS,
      )).rejects.toThrow(/exact keys/u);
    }
  });
});

async function passedRecord(): Promise<TestTerminalRecord> {
  const expected = await deriveCompilerViewCopyBindingsExpectedEvidence(
    SOURCE_REVISION,
    PRODUCER_VERSIONS,
  );
  const environment: TestEnvironment = {
    schema: "browsergrad.execution-environment@1",
    acquisition: "navigator.gpu.requestAdapter/requestDevice",
    userAgent: "browsergrad-evidence-log-test",
    platform: "test-platform",
    adapter: {
      vendor: "test-vendor",
      architecture: "test-architecture",
      device: "test-device",
      description: "test-adapter",
    },
    adapterSupportedFeatures: ["shader-f16", "timestamp-query"],
    negotiatedDeviceFeatures: ["shader-f16"],
    negotiatedDeviceLimits: {
      maxBufferSize: 268_435_456,
      maxStorageBufferBindingSize: 134_217_728,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxBindingsPerBindGroup: 1_000,
      maxStorageBuffersPerShaderStage: 8,
    },
  };
  const [environmentId, deviceProfileHash] = await Promise.all([
    hashNamedComponents({ environment }),
    hashNamedComponents({
      backendId: BACKEND_ID,
      adapter: environment.adapter,
      selectedFeatures: [],
      adapterSupportedFeatures: environment.adapterSupportedFeatures,
      negotiatedDeviceFeatures: environment.negotiatedDeviceFeatures,
      negotiatedDeviceLimits: environment.negotiatedDeviceLimits,
    }),
  ]);
  const preparedCases = expected.preparedCases as readonly TestPreparedCase[];
  const completedCases: readonly TestObservation[] = preparedCases.map((preparedCase) => ({
    ...preparedCase,
    actualSourceHash: preparedCase.expectedSourceHash,
    actualDestinationHash: preparedCase.expectedDestinationHash,
    planKind: "single-dispatch",
    stepCount: 1,
    plannedPipelineCount: 1,
    comparisonPolicyId: COMPARISON_POLICY_ID,
  }));
  const unsigned: TestUnsignedTerminalRecord = {
    schema: "browsergrad.execution-evidence@1",
    kind: "terminal",
    suiteId: SUITE_ID,
    required: true,
    evidence: {
      capabilityId: "browsergrad.compiler.verified-view-copy-binding",
      artifactHash: expected.artifactHash,
      backendId: BACKEND_ID,
      environmentId,
      producerVersions: PRODUCER_VERSIONS,
      sourceRevision: SOURCE_REVISION,
      deviceProfileHash,
      recordedAt: "2026-07-16T00:00:00.000Z",
      outcome: "passed",
      comparisonPolicyId: COMPARISON_POLICY_ID,
      diagnosticCodes: [],
    },
    environment,
    artifactHashKind: "prepared-case-set",
    preparedBackendArtifactHash: expected.preparedBackendArtifactHash,
    caseSetHash: expected.caseSetHash,
    preparedCases,
    plannedCaseIds: CASE_IDS,
    completedCases,
    stage: "terminal-summary",
    uncapturedErrors: [],
  };
  const detachedUnsigned = jsonTreeClone(unsigned);
  return {
    ...detachedUnsigned,
    terminalManifestHash: await terminalManifestHashFor(detachedUnsigned),
  };
}

async function reseal(
  record: TestTerminalRecord | TestUnsignedTerminalRecord,
): Promise<TestTerminalRecord> {
  const { terminalManifestHash: _excluded, ...unsigned } = record as TestTerminalRecord;
  const detachedUnsigned = jsonTreeClone(unsigned as TestUnsignedTerminalRecord);
  return {
    ...detachedUnsigned,
    terminalManifestHash: await terminalManifestHashFor(detachedUnsigned),
  };
}

async function resealUnchecked(record: JsonObject): Promise<JsonObject> {
  const { terminalManifestHash: _excluded, ...unsigned } = record;
  const detachedUnsigned = jsonTreeClone(unsigned as JsonObject);
  const evidence = detachedUnsigned.evidence as JsonObject;
  if (typeof evidence.sourceRevision !== "string") throw new Error("test record has no source revision");
  return {
    ...detachedUnsigned,
    terminalManifestHash: await hashCanonicalJson({
      domain: TERMINAL_HASH_DOMAIN,
      sourceRevision: evidence.sourceRevision,
      terminalRecord: detachedUnsigned,
    }),
  };
}

async function resealPreparedForgery(
  record: TestTerminalRecord,
  caseIndex: number,
  updates: Partial<TestPreparedCase>,
): Promise<TestTerminalRecord> {
  const current = record.preparedCases[caseIndex];
  if (current === undefined) throw new Error(`no prepared case at ${caseIndex}`);
  const { caseArtifactHash: _excluded, ...manifest } = { ...current, ...updates };
  const forgedCase: TestPreparedCase = {
    ...manifest,
    caseArtifactHash: await hashNamedComponents({
      suiteId: SUITE_ID,
      case: manifest as unknown as JsonObject,
    }),
  };
  const preparedCases = record.preparedCases.map((entry, index) => (
    index === caseIndex ? forgedCase : entry
  ));
  const preparedBackendArtifactHash = await preparedBackendHashFor(preparedCases);
  const caseSetHash = await hashNamedComponents({
    suiteId: SUITE_ID,
    comparisonPolicyId: COMPARISON_POLICY_ID,
    sourceRevision: record.evidence.sourceRevision,
    producerVersions: record.evidence.producerVersions,
    cases: preparedCases,
  });
  const artifactHash = await hashNamedComponents({
    suiteId: SUITE_ID,
    artifactHashKind: "prepared-case-set",
    preparedBackendArtifactHash,
    caseSetHash,
    comparisonPolicyId: COMPARISON_POLICY_ID,
    sourceRevision: record.evidence.sourceRevision,
    producerVersions: record.evidence.producerVersions,
  });
  const completedCases = record.completedCases.map((entry, index) => (
    index === caseIndex
      ? {
        ...forgedCase,
        actualSourceHash: forgedCase.expectedSourceHash,
        actualDestinationHash: forgedCase.expectedDestinationHash,
        planKind: "single-dispatch",
        stepCount: 1,
        plannedPipelineCount: 1,
        comparisonPolicyId: COMPARISON_POLICY_ID,
      }
      : entry
  ));
  return reseal({
    ...record,
    evidence: { ...record.evidence, artifactHash },
    preparedBackendArtifactHash,
    caseSetHash,
    preparedCases,
    completedCases,
  });
}

function preparedBackendHashFor(cases: readonly TestPreparedCase[]): Promise<string> {
  return hashNamedComponents({
    suiteId: SUITE_ID,
    backendId: BACKEND_ID,
    artifacts: cases.map(({ caseId, caseArtifactHash, wgslModuleHash, programName }) => ({
      caseId,
      caseArtifactHash,
      wgslModuleHash,
      programName,
    })),
  });
}

async function terminalManifestHashFor(record: TestUnsignedTerminalRecord): Promise<string> {
  return hashCanonicalJson({
    domain: TERMINAL_HASH_DOMAIN,
    sourceRevision: record.evidence.sourceRevision,
    terminalRecord: record as unknown as JsonObject,
  });
}

function omitKeys(
  record: TestTerminalRecord,
  ...keys: readonly (keyof TestTerminalRecord)[]
): JsonObject {
  return Object.fromEntries(
    Object.entries(record).filter(([key]) => !keys.includes(key as keyof TestTerminalRecord)),
  ) as JsonObject;
}

function jsonTreeClone<T extends JsonObject>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function log(record: unknown): string {
  return `${COMPILER_VIEW_COPY_EVIDENCE_PREFIX}${JSON.stringify(record)}`;
}
