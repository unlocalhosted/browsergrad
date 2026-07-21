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
  createVerifiedDenseAttentionForwardArtifacts,
  prepareAttentionForwardCpu,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  createVerifiedAttentionOnlineKvTileSchedule,
} from "@unlocalhosted/browsergrad-semantic-core/schedule";
import {
  hashNamedComponents,
  hashSemanticArtifact,
  parseWireU64,
  sha256Hex,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createDevice } from "../src/device";
import {
  SEMANTIC_ATTENTION_WEBGPU_BACKEND_VERSION,
  SemanticAttentionWebGpuError,
  prepareSemanticAttentionWgsl,
  runSemanticAttentionWebGpu,
  type PreparedSemanticAttentionWgsl,
} from "../src/semantic_attention";

declare const __BG_KERNELS_VERSION__: string;
declare const __BG_SEMANTIC_CORE_VERSION__: string;

const wire = (value: number) => parseWireU64(String(value));
const EVIDENCE_PREFIX = "[browsergrad-semantic-attention-webgpu-evidence]";
const SUITE_ID = "browsergrad.kernels.semantic-attention.webgpu-conformance@1";
const CAPABILITY_ID = "browsergrad.kernel.attention-forward.block-tiled-online-kv";
const BACKEND_ID = "browsergrad.backend.webgpu.core";
const COMPARISON_POLICY_ID = "browsergrad.attention-forward.f32-abs-relative@1";
const CASE_IDS = Object.freeze([
  "noncausal-q8-k8",
  "noncausal-q8-k16",
  "causal-q8-k8",
  "causal-q8-k16",
]);
const PRODUCER_VERSIONS = Object.freeze({
  "@unlocalhosted/browsergrad-kernels": __BG_KERNELS_VERSION__,
  "@unlocalhosted/browsergrad-semantic-core": __BG_SEMANTIC_CORE_VERSION__,
  "browsergrad.backend.webgpu.semantic-attention": SEMANTIC_ATTENTION_WEBGPU_BACKEND_VERSION,
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
  readonly prepared: PreparedSemanticAttentionWgsl;
  readonly artifactHash: string;
  readonly cpu: Awaited<ReturnType<typeof prepareAttentionForwardCpu>>;
  readonly expected: Uint8Array;
  readonly inputs: Readonly<{
    query: Uint8Array;
    key: Uint8Array;
    value: Uint8Array;
  }>;
}

interface CaseObservation extends JsonObject {
  readonly caseId: string;
  readonly artifactHash: string;
  readonly semanticSpecializationHash: string;
  readonly scheduleSemanticHash: string;
  readonly scheduleSpecializationHash: string;
  readonly backendPreparationHash: string;
  readonly backendSpecializationHash: string;
  readonly wgslModuleHash: string;
  readonly mask: "none" | "causal-upper-left";
  readonly physicalTile: JsonObject;
  readonly cpuComparison: "declared-policy-complete-destination";
  readonly crossScheduleComparison: "baseline" | "declared-policy-complete-destination";
}

it("emits required causal/non-causal two-schedule semantic attention evidence", async (context) => {
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
    const preparedCases = await prepareCases();
    artifactHash = await hashNamedComponents({
      suiteId: SUITE_ID,
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
        diagnosticCodes: ["BG-WEBGPU-ATTENTION-EVIDENCE-DEVICE-UNAVAILABLE"],
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

    const firstByMask = new Map<string, Uint8Array>();
    for (const preparedCase of preparedCases) {
      stage = "case-execution";
      currentCaseId = preparedCase.caseId;
      const result = await runSemanticAttentionWebGpu(
        kernelDevice,
        preparedCase.prepared,
        preparedCase.inputs,
      );
      await withAttentionEvidenceTimeout(device.queue.onSubmittedWorkDone(), 10_000, "queue-drain");
      const cpuComparison = preparedCase.cpu.compare(result.destination, preparedCase.expected);
      if (!cpuComparison.passed) {
        throw new EvidenceLaneError(
          "BG-WEBGPU-ATTENTION-EVIDENCE-COMPARISON",
          `${preparedCase.caseId} differs from CPU at ${cpuComparison.firstMismatchIndex}`,
        );
      }
      const prior = firstByMask.get(result.trace.mask);
      let crossScheduleComparison: CaseObservation["crossScheduleComparison"] = "baseline";
      if (prior === undefined) firstByMask.set(result.trace.mask, new Uint8Array(result.destination));
      else {
        const comparison = preparedCase.cpu.compare(result.destination, prior);
        if (!comparison.passed) {
          throw new EvidenceLaneError(
            "BG-WEBGPU-ATTENTION-EVIDENCE-CROSS-SCHEDULE",
            `${preparedCase.caseId} differs from its same-mask schedule baseline`,
          );
        }
        crossScheduleComparison = "declared-policy-complete-destination";
      }
      assertTrace(preparedCase, result.trace);
      completedCases.push(Object.freeze({
        caseId: preparedCase.caseId,
        artifactHash: preparedCase.artifactHash,
        semanticSpecializationHash: result.trace.semanticSpecializationHash,
        scheduleSemanticHash: result.trace.scheduleSemanticHash,
        scheduleSpecializationHash: result.trace.scheduleSpecializationHash,
        backendPreparationHash: result.trace.backendPreparationHash,
        backendSpecializationHash: result.trace.backendSpecializationHash,
        wgslModuleHash: result.trace.wgslModuleHash,
        mask: result.trace.mask,
        physicalTile: result.trace.physicalTile as JsonObject,
        cpuComparison: "declared-policy-complete-destination",
        crossScheduleComparison,
      }));
    }

    stage = "late-error-drain";
    currentCaseId = undefined;
    await withAttentionEvidenceTimeout(device.queue.onSubmittedWorkDone(), 10_000, "final-queue-drain");
    await nextWebGpuEvidenceTask();
    if (uncapturedErrors.length > 0) {
      throw new EvidenceLaneError(
        "BG-WEBGPU-ATTENTION-EVIDENCE-UNCAUGHT-GPU-ERROR",
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

async function prepareCases(): Promise<readonly PreparedCase[]> {
  const cases: PreparedCase[] = [];
  for (const causal of [false, true]) {
    const semantics = await createVerifiedDenseAttentionForwardArtifacts({
      batch: wire(1),
      heads: wire(2),
      queryLength: wire(9),
      keyLength: wire(11),
      queryDepth: wire(4),
      valueDepth: wire(6),
      causal,
    });
    const query = patternedBytes(1 * 2 * 9 * 4, 7);
    const key = patternedBytes(1 * 2 * 11 * 4, 11);
    const value = patternedBytes(1 * 2 * 11 * 6, 13);
    const destination = new Uint8Array(1 * 2 * 9 * 6 * 4);
    const cpu = await prepareAttentionForwardCpu(
      semantics.layout,
      semantics.kernel,
      { operationId: semantics.operationId },
    );
    await cpu.execute({ query, key, value, destination });
    const inputHash = await hashNamedComponents({
      query: await sha256Hex(query),
      key: await sha256Hex(key),
      value: await sha256Hex(value),
    });
    const [layoutHash, kernelHash] = await Promise.all([
      hashSemanticArtifact(semantics.layout),
      hashSemanticArtifact(semantics.kernel),
    ]);
    for (const keyRows of [8, 16]) {
      const schedule = await createVerifiedAttentionOnlineKvTileSchedule(semantics.kernel, {
        physicalTile: { queryRows: wire(8), keyRows: wire(keyRows) },
      });
      const prepared = await prepareSemanticAttentionWgsl(
        semantics.layout,
        semantics.kernel,
        schedule.artifact,
        { operationId: semantics.operationId },
      );
      const caseId = `${causal ? "causal" : "noncausal"}-q8-k${keyRows}`;
      cases.push(Object.freeze({
        caseId,
        prepared,
        artifactHash: await hashNamedComponents({
          caseId,
          inputHash,
          layoutHash,
          kernelHash,
          schedule: prepared.scheduled.scheduleSemanticHash,
          backendPreparation: prepared.backendPreparationHash,
          wgsl: prepared.wgslModuleHash,
        }),
        cpu,
        expected: new Uint8Array(destination),
        inputs: Object.freeze({ query, key, value }),
      }));
    }
  }
  const ids = cases.map(({ caseId }) => caseId);
  if (ids.some((id, index) => id !== CASE_IDS[index])) {
    throw new EvidenceLaneError(
      "BG-WEBGPU-ATTENTION-EVIDENCE-CASE-SET",
      `prepared case order ${ids.join(",")} differs from manifest`,
    );
  }
  return Object.freeze(cases);
}

function assertTrace(
  preparedCase: PreparedCase,
  trace: Awaited<ReturnType<typeof runSemanticAttentionWebGpu>>["trace"],
): void {
  if (
    trace.semanticSpecializationHash !== preparedCase.prepared.semantic.specializationHash
    || trace.scheduleSemanticHash !== preparedCase.prepared.scheduled.scheduleSemanticHash
    || trace.scheduleSpecializationHash
      !== preparedCase.prepared.scheduled.scheduleSpecializationHash
    || trace.backendPreparationHash !== preparedCase.prepared.backendPreparationHash
    || trace.wgslModuleHash !== preparedCase.prepared.wgslModuleHash
    || trace.algorithmProfile !== "block-tiled-kv-online-softmax-forward"
    || trace.executionTier !== "portable-webgpu-core"
    || trace.preservationLevel !== "portable-relegalized"
    || trace.numericalPreservation !== "requires-declared-policy-comparison"
  ) {
    throw new EvidenceLaneError(
      "BG-WEBGPU-ATTENTION-EVIDENCE-IDENTITY",
      `${preparedCase.caseId} execution trace differs from prepared identities`,
    );
  }
}

function patternedBytes(length: number, modulus: number): Uint8Array {
  const values = new Float32Array(length);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = (((index * 7) % modulus) - Math.floor(modulus / 2)) / modulus;
  }
  return new Uint8Array(values.buffer);
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
      ...(input.deviceProfileHash === undefined
        ? {}
        : { deviceProfileHash: input.deviceProfileHash }),
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
        "BG-WEBGPU-ATTENTION-EVIDENCE-CASE-SET",
        "passed terminal evidence requires all four cases in order",
      );
    }
  }
  TERMINAL_EMITTER.emit(record);
}

function diagnosticCode(error: unknown, stage: string): string {
  if (error instanceof SemanticAttentionWebGpuError) return error.code;
  if (error instanceof EvidenceLaneError) return error.code;
  if (stage === "fixture-construction") return "BG-WEBGPU-ATTENTION-EVIDENCE-FIXTURE";
  if (stage === "device-acquisition") return "BG-WEBGPU-ATTENTION-EVIDENCE-DEVICE-UNAVAILABLE";
  if (stage === "kernel-device-construction") return "BG-WEBGPU-ATTENTION-EVIDENCE-DEVICE-WRAP";
  return "BG-WEBGPU-ATTENTION-EVIDENCE-INTERNAL";
}

function errorRecord(error: unknown): JsonObject {
  if (error instanceof SemanticAttentionWebGpuError) {
    return { name: error.name, message: error.message, code: error.code, path: error.path };
  }
  if (error instanceof EvidenceLaneError) {
    return { name: error.name, message: error.message, code: error.code };
  }
  if (error instanceof Error) return { name: error.name, message: error.message };
  return { name: "UnknownError", message: String(error) };
}

function withAttentionEvidenceTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  label: string,
): Promise<T> {
  return withWebGpuEvidenceTimeout(
    promise,
    timeoutMs,
    label,
    (message) => new EvidenceLaneError("BG-WEBGPU-ATTENTION-EVIDENCE-TIMEOUT", message),
  );
}

class EvidenceLaneError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "EvidenceLaneError";
  }
}
