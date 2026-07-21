import { it } from "vitest";
import {
  EXECUTION_EVIDENCE_SCHEMA,
  acquireWebGpuEvidenceDevice,
  createWebGpuExecutionEnvironmentRecord,
  createTerminalEvidenceEmitter,
  nextWebGpuEvidenceTask,
  requiredEvidenceFailure,
  requiresWebGpuEvidence,
  webGpuSemanticDeviceLimits,
  withWebGpuEvidenceTimeout,
} from "../../../test-support/webgpu-evidence";

import {
  copyCertifiedLogicalGemmExactF32Inputs,
  createVerifiedDenseLogicalGemmTileArtifacts,
  createVerifiedLogicalGemmExactF32InputCertificate,
  prepareLogicalGemmTileCpu,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  createVerifiedLogicalGemmTileSchedule,
} from "@unlocalhosted/browsergrad-semantic-core/schedule";
import {
  hashNamedComponents,
  hashSemanticArtifact,
  parseWireU64,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createDevice } from "../src/device";
import {
  SEMANTIC_GEMM_WEBGPU_BACKEND_VERSION,
  SemanticGemmWebGpuError,
  prepareSemanticGemmWgsl,
  runSemanticGemmWebGpu,
  type PreparedSemanticGemmWgsl,
} from "../src/semantic_gemm";

declare const __BG_KERNELS_VERSION__: string;
declare const __BG_SEMANTIC_CORE_VERSION__: string;

const wire = (value: number) => parseWireU64(String(value));
const EVIDENCE_PREFIX = "[browsergrad-semantic-gemm-webgpu-evidence]";
const SUITE_ID = "browsergrad.kernels.semantic-gemm.webgpu-conformance@1";
const CAPABILITY_ID = "browsergrad.kernel.logical-gemm-tile";
const BACKEND_ID = "browsergrad.backend.webgpu.core";
const COMPARISON_POLICY_ID = "browsergrad.comparison.certified-f32-bit-exact-complete-destination.v1";
const CASE_IDS = Object.freeze(["physical-8x8x8", "physical-16x16x16"]);
const PRODUCER_VERSIONS = Object.freeze({
  "@unlocalhosted/browsergrad-kernels": __BG_KERNELS_VERSION__,
  "@unlocalhosted/browsergrad-semantic-core": __BG_SEMANTIC_CORE_VERSION__,
  "browsergrad.backend.webgpu.semantic-gemm": SEMANTIC_GEMM_WEBGPU_BACKEND_VERSION,
});
const TERMINAL_EMITTER = createTerminalEvidenceEmitter(EVIDENCE_PREFIX, {
  suiteId: SUITE_ID,
  capabilityId: CAPABILITY_ID,
  backendId: BACKEND_ID,
  comparisonPolicyId: COMPARISON_POLICY_ID,
  requireDeviceProfile: true,
});

interface PreparedCase {
  readonly caseId: string;
  readonly prepared: PreparedSemanticGemmWgsl;
  readonly artifactHash: string;
}

interface CaseObservation extends JsonObject {
  readonly caseId: string;
  readonly artifactHash: string;
  readonly semanticSpecializationHash: string;
  readonly scheduleSemanticHash: string;
  readonly scheduleSpecializationHash: string;
  readonly backendPreparationHash: string;
  readonly backendSpecializationHash: string;
  readonly inputCertificateSemanticHash: string;
  readonly wgslModuleHash: string;
  readonly executionTier: "portable-webgpu-core";
  readonly preservationLevel: "portable-relegalized";
  readonly physicalTile: JsonObject;
  readonly dispatchWorkgroups: JsonObject;
  readonly cpuComparison: "bit-exact-complete-destination";
  readonly crossScheduleComparison: "baseline" | "bit-exact-complete-destination";
}

it("emits required evidence for irregular semantic GEMM under two schedules", async (context) => {
  const required = requiresWebGpuEvidence();
  let stage = "suite-manifest";
  let currentCaseId: string | undefined;
  let artifactHash = await hashNamedComponents({ suiteId: SUITE_ID, plannedCaseIds: CASE_IDS });
  let caseSetHash: string | undefined;
  let environment = createWebGpuExecutionEnvironmentRecord({ acquisition: "not-attempted" });
  let environmentId = await hashNamedComponents({ environment });
  let deviceProfileHash: string | undefined;
  let terminalEmitted = false;
  const completedCases: CaseObservation[] = [];
  const uncapturedErrors: string[] = [];
  let device: GPUDevice | undefined;
  let kernelDevice: Awaited<ReturnType<typeof createDevice>> | undefined;
  const uncapturedHandler = (event: GPUUncapturedErrorEvent) => {
    uncapturedErrors.push(event.error.message);
  };

  try {
    stage = "fixture-construction";
    const m = 17;
    const n = 19;
    const k = 23;
    const semantics = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire(m),
      n: wire(n),
      k: wire(k),
      logicalTile: { m: wire(16), n: wire(16), k: wire(16) },
    });
    const lhs = patternedBytes(m * k, 5);
    const rhs = patternedBytes(k * n, 3);
    const certificate = await createVerifiedLogicalGemmExactF32InputCertificate(
      semantics.layout,
      semantics.kernel,
      { operationId: semantics.operationId, inputs: { lhs, rhs } },
    );
    lhs.fill(0xff);
    rhs.fill(0xff);
    const certifiedCpuInputs = copyCertifiedLogicalGemmExactF32Inputs(certificate.certificate);
    const cpu = await prepareLogicalGemmTileCpu(
      semantics.layout,
      semantics.kernel,
      { operationId: semantics.operationId },
    );
    const expected = new Uint8Array(m * n * 4);
    cpu.execute({
      lhs: certifiedCpuInputs.lhs,
      rhs: certifiedCpuInputs.rhs,
      destination: expected,
    });

    stage = "semantic-and-wgsl-preparation";
    const schedules = await Promise.all([
      createVerifiedLogicalGemmTileSchedule(semantics.kernel, {
        physicalTile: { m: wire(8), n: wire(8), k: wire(8) },
      }),
      createVerifiedLogicalGemmTileSchedule(semantics.kernel, {
        physicalTile: { m: wire(16), n: wire(16), k: wire(16) },
      }),
    ]);
    const preparedCases = await Promise.all(schedules.map(async ({ artifact }, index): Promise<PreparedCase> => {
      const prepared = await prepareSemanticGemmWgsl(
        semantics.layout,
        semantics.kernel,
        artifact,
        { operationId: semantics.operationId },
      );
      const caseId = CASE_IDS[index];
      if (caseId === undefined) throw new EvidenceLaneError("BG-WEBGPU-GEMM-EVIDENCE-CASE-SET", "missing planned schedule case");
      return Object.freeze({
        caseId,
        prepared,
        artifactHash: await hashNamedComponents({
          caseId,
          logicalSpecialization: prepared.semantic.specializationHash,
          schedule: prepared.scheduled.scheduleSemanticHash,
          scheduleSpecialization: prepared.scheduled.scheduleSpecializationHash,
          backendPreparation: prepared.backendPreparationHash,
          wgsl: prepared.wgslModuleHash,
        }),
      });
    }));
    const [layoutSemanticHash, kernelSemanticHash, inputCertificateSemanticHash] = await Promise.all([
      hashSemanticArtifact(semantics.layout),
      hashSemanticArtifact(semantics.kernel),
      hashSemanticArtifact(certificate.certificate),
    ]);
    artifactHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      layoutSemanticHash,
      kernelSemanticHash,
      inputCertificateSemanticHash,
      cases: preparedCases.map(({ caseId, artifactHash: caseArtifactHash }) => ({
        caseId,
        artifactHash: caseArtifactHash,
      })),
    });
    caseSetHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      artifactHash,
      plannedCaseIds: CASE_IDS,
      comparisonPolicyId: COMPARISON_POLICY_ID,
    });

    stage = "device-acquisition";
    const acquisition = await acquireWebGpuEvidenceDevice();
    if (acquisition.kind === "unavailable") {
      environment = createWebGpuExecutionEnvironmentRecord({
        acquisition: "navigator.gpu.requestAdapter/requestDevice",
        unavailableReason: acquisition.reason,
      });
      environmentId = await hashNamedComponents({ environment });
      emitTerminal({
        required,
        artifactHash,
        caseSetHash,
        environment,
        environmentId,
        outcome: required ? "failed" : "not-run",
        diagnosticCodes: ["BG-WEBGPU-GEMM-EVIDENCE-DEVICE-UNAVAILABLE"],
        completedCases,
        stage,
        uncapturedErrors,
        error: { name: "WebGpuEvidenceUnavailable", message: acquisition.reason },
      });
      terminalEmitted = true;
      if (required) throw requiredEvidenceFailure(acquisition.reason);
      context.skip(acquisition.reason);
      return;
    }

    const acquired = acquisition.value;
    device = acquired.device;
    const adapterFeatures = Object.freeze([...acquired.adapter.features].map(String).sort());
    const deviceFeatures = Object.freeze([...device.features].map(String).sort());
    const negotiatedLimits = webGpuSemanticDeviceLimits(device);
    environment = createWebGpuExecutionEnvironmentRecord({
      acquisition: "navigator.gpu.requestAdapter/requestDevice",
      adapter: acquired.adapterInfo as unknown as JsonObject,
      adapterSupportedFeatures: adapterFeatures,
      negotiatedDeviceFeatures: deviceFeatures,
      negotiatedDeviceLimits: negotiatedLimits,
    });
    environmentId = await hashNamedComponents({ environment });
    deviceProfileHash = await hashNamedComponents({
      backendId: BACKEND_ID,
      adapter: acquired.adapterInfo as unknown as JsonObject,
      adapterSupportedFeatures: adapterFeatures,
      negotiatedDeviceFeatures: deviceFeatures,
      negotiatedDeviceLimits: negotiatedLimits,
    });
    stage = "kernel-device-construction";
    kernelDevice = await createDevice({ device });
    device.addEventListener("uncapturederror", uncapturedHandler);

    let firstOutput: Uint8Array | undefined;
    for (const preparedCase of preparedCases) {
      stage = "case-execution";
      currentCaseId = preparedCase.caseId;
      const result = await runSemanticGemmWebGpu(
        kernelDevice,
        preparedCase.prepared,
        certificate.certificate,
      );
      await withGemmEvidenceTimeout(device.queue.onSubmittedWorkDone(), 10_000, "queue-drain");
      assertEqualBytes(result.destination, expected, `${preparedCase.caseId} CPU`);
      if (firstOutput === undefined) firstOutput = new Uint8Array(result.destination);
      else assertEqualBytes(result.destination, firstOutput, `${preparedCase.caseId} cross-schedule`);
      assertTrace(preparedCase, inputCertificateSemanticHash, result.trace);
      completedCases.push(Object.freeze({
        caseId: preparedCase.caseId,
        artifactHash: preparedCase.artifactHash,
        semanticSpecializationHash: result.trace.semanticSpecializationHash,
        scheduleSemanticHash: result.trace.scheduleSemanticHash,
        scheduleSpecializationHash: result.trace.scheduleSpecializationHash,
        backendPreparationHash: result.trace.backendPreparationHash,
        backendSpecializationHash: result.trace.backendSpecializationHash,
        inputCertificateSemanticHash: result.trace.inputCertificateSemanticHash,
        wgslModuleHash: result.trace.wgslModuleHash,
        executionTier: result.trace.executionTier,
        preservationLevel: result.trace.preservationLevel,
        physicalTile: result.trace.physicalTile as JsonObject,
        dispatchWorkgroups: result.trace.dispatchWorkgroups as JsonObject,
        cpuComparison: "bit-exact-complete-destination",
        crossScheduleComparison: completedCases.length === 0
          ? "baseline"
          : "bit-exact-complete-destination",
      }));
    }

    stage = "late-error-drain";
    currentCaseId = undefined;
    await withGemmEvidenceTimeout(device.queue.onSubmittedWorkDone(), 10_000, "final-queue-drain");
    await nextWebGpuEvidenceTask();
    if (uncapturedErrors.length > 0) {
      throw new EvidenceLaneError(
        "BG-WEBGPU-GEMM-EVIDENCE-UNCAUGHT-GPU-ERROR",
        uncapturedErrors.join("; "),
      );
    }
    stage = "terminal-summary";
    emitTerminal({
      required,
      artifactHash,
      caseSetHash,
      environment,
      environmentId,
      deviceProfileHash,
      outcome: "passed",
      diagnosticCodes: [],
      completedCases,
      stage,
      uncapturedErrors,
    });
    terminalEmitted = true;
  } catch (error) {
    if (!terminalEmitted) {
      emitTerminal({
        required,
        artifactHash,
        ...(caseSetHash === undefined ? {} : { caseSetHash }),
        environment,
        environmentId,
        ...(deviceProfileHash === undefined ? {} : { deviceProfileHash }),
        outcome: "failed",
        diagnosticCodes: [diagnosticCode(error, stage)],
        completedCases,
        stage,
        ...(currentCaseId === undefined ? {} : { currentCaseId }),
        uncapturedErrors,
        error: errorRecord(error),
      });
    }
    throw error;
  } finally {
    device?.removeEventListener("uncapturederror", uncapturedHandler);
    kernelDevice?.clearCache();
    device?.destroy();
  }
});

function assertTrace(
  preparedCase: PreparedCase,
  certificateHash: string,
  trace: Awaited<ReturnType<typeof runSemanticGemmWebGpu>>["trace"],
): void {
  if (
    trace.semanticSpecializationHash !== preparedCase.prepared.semantic.specializationHash
    || trace.scheduleSemanticHash !== preparedCase.prepared.scheduled.scheduleSemanticHash
    || trace.scheduleSpecializationHash !== preparedCase.prepared.scheduled.scheduleSpecializationHash
    || trace.backendPreparationHash !== preparedCase.prepared.backendPreparationHash
    || trace.wgslModuleHash !== preparedCase.prepared.wgslModuleHash
    || trace.inputCertificateSemanticHash !== certificateHash
    || trace.executionTier !== "portable-webgpu-core"
    || trace.preservationLevel !== "portable-relegalized"
    || trace.numericalPreservation !== "bit-exact-on-certified-inputs"
  ) {
    throw new EvidenceLaneError(
      "BG-WEBGPU-GEMM-EVIDENCE-IDENTITY",
      `${preparedCase.caseId} execution trace differs from prepared/certified identities`,
    );
  }
}

function patternedBytes(length: number, modulus: number): Uint8Array {
  const values = new Float32Array(length);
  for (let index = 0; index < values.length; index++) values[index] = index % modulus;
  return new Uint8Array(values.buffer);
}

function assertEqualBytes(actual: Uint8Array, expected: Uint8Array, label: string): void {
  if (actual.byteLength !== expected.byteLength) {
    throw new EvidenceLaneError(
      "BG-WEBGPU-GEMM-EVIDENCE-COMPARISON",
      `${label} destination length ${actual.byteLength} does not equal ${expected.byteLength}`,
    );
  }
  for (let index = 0; index < expected.byteLength; index++) {
    if (actual[index] !== expected[index]) {
      throw new EvidenceLaneError(
        "BG-WEBGPU-GEMM-EVIDENCE-COMPARISON",
        `${label} destination differs at byte ${index}: ${actual[index]} != ${expected[index]}`,
      );
    }
  }
}

function emitTerminal(input: Readonly<{
  required: boolean;
  artifactHash: string;
  caseSetHash?: string;
  environment: JsonObject;
  environmentId: string;
  deviceProfileHash?: string;
  outcome: "not-run" | "passed" | "failed";
  diagnosticCodes: readonly string[];
  completedCases: readonly CaseObservation[];
  stage: string;
  currentCaseId?: string;
  uncapturedErrors: readonly string[];
  error?: JsonObject;
}>): void {
  const record = Object.freeze({
    schema: EXECUTION_EVIDENCE_SCHEMA,
    kind: "terminal",
    suiteId: SUITE_ID,
    required: input.required,
    evidence: Object.freeze({
      capabilityId: CAPABILITY_ID,
      artifactHash: input.artifactHash,
      backendId: BACKEND_ID,
      environmentId: input.environmentId,
      producerVersions: PRODUCER_VERSIONS,
      ...(input.deviceProfileHash === undefined ? {} : { deviceProfileHash: input.deviceProfileHash }),
      recordedAt: new Date().toISOString(),
      outcome: input.outcome,
      comparisonPolicyId: COMPARISON_POLICY_ID,
      diagnosticCodes: Object.freeze([...input.diagnosticCodes]),
    }),
    environment: input.environment,
    artifactHashKind: input.caseSetHash === undefined ? "planned-suite-manifest" : "prepared-suite",
    ...(input.caseSetHash === undefined ? {} : { caseSetHash: input.caseSetHash }),
    plannedCaseIds: CASE_IDS,
    completedCases: Object.freeze([...input.completedCases]),
    stage: input.stage,
    ...(input.currentCaseId === undefined ? {} : { currentCaseId: input.currentCaseId }),
    uncapturedErrors: Object.freeze([...input.uncapturedErrors]),
    ...(input.error === undefined ? {} : { error: input.error }),
  });
  if (input.outcome === "passed") {
    const ids = input.completedCases.map(({ caseId }) => caseId);
    if (ids.length !== CASE_IDS.length || ids.some((id, index) => id !== CASE_IDS[index])) {
      throw new EvidenceLaneError(
        "BG-WEBGPU-GEMM-EVIDENCE-CASE-SET",
        "passed terminal evidence requires both physical schedules in order",
      );
    }
  }
  TERMINAL_EMITTER.emit(record);
}

function diagnosticCode(error: unknown, stage: string): string {
  if (error instanceof SemanticGemmWebGpuError) return error.code;
  if (error instanceof EvidenceLaneError) return error.code;
  if (stage === "fixture-construction") return "BG-WEBGPU-GEMM-EVIDENCE-FIXTURE";
  if (stage === "semantic-and-wgsl-preparation") return "BG-WEBGPU-GEMM-EVIDENCE-PREPARATION";
  if (stage === "device-acquisition") return "BG-WEBGPU-GEMM-EVIDENCE-DEVICE-UNAVAILABLE";
  if (stage === "kernel-device-construction") return "BG-WEBGPU-GEMM-EVIDENCE-DEVICE-WRAP";
  return "BG-WEBGPU-GEMM-EVIDENCE-INTERNAL";
}

function errorRecord(error: unknown): JsonObject {
  if (error instanceof SemanticGemmWebGpuError) {
    return { name: error.name, message: error.message, code: error.code, path: error.path };
  }
  if (error instanceof EvidenceLaneError) {
    return { name: error.name, message: error.message, code: error.code };
  }
  if (error instanceof Error) return { name: error.name, message: error.message };
  return { name: "UnknownError", message: String(error) };
}

function withGemmEvidenceTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  label: string,
): Promise<T> {
  return withWebGpuEvidenceTimeout(
    promise,
    timeoutMs,
    label,
    (message) => new EvidenceLaneError("BG-WEBGPU-GEMM-EVIDENCE-TIMEOUT", message),
  );
}

class EvidenceLaneError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "EvidenceLaneError";
  }
}
