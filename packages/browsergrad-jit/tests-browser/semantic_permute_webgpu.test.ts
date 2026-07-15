import { it } from "vitest";

import {
  createDevice,
  createWebGpuRealizerBridge,
  prepareTensorPlanSemanticRequests,
  SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION,
  type DirectDispatchProfile,
  type KernelDevice,
  type PreparedTensorPlanSemanticRequest,
  type SemanticTensorPlanExecutionTrace,
} from "@unlocalhosted/browsergrad-kernels";
import {
  hashNamedComponents,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  DENSE_PERMUTATION_VIEW_COPY_FIXTURES,
  fixtureExtentNumbers,
  fixtureWords,
  type DensePermutationFixtureCase,
} from "../../../test-support/dense-permutation-view-copy-fixtures";
import {
  EXECUTION_ENVIRONMENT_SCHEMA,
  EXECUTION_EVIDENCE_SCHEMA,
  acquireWebGpuEvidenceDevice,
  createTerminalEvidenceEmitter,
  requiredEvidenceFailure,
  requiresWebGpuEvidence,
} from "../../../test-support/webgpu-evidence";
import {
  BACKEND_ID,
  CAPABILITY_ID,
  COMPARISON_POLICY_ID,
  DEVICE_UNAVAILABLE_DIAGNOSTIC,
  PLANNED_CASE_IDS,
  SUITE_ID,
  TERMINAL_EXPECTATION,
  UNCAPTURED_GPU_ERROR_DIAGNOSTIC,
  finalizeTerminalEvidence,
  validateObservation,
  validatePreparedCaseManifest,
  type CaseObservation,
  type EvidenceEnvironment,
  type PreparedCaseManifest,
  type TerminalEvidenceRecord,
  type TerminalStage,
  type UnsignedTerminalEvidenceRecord,
} from "./semantic_permute_evidence";

declare const __BG_JIT_VERSION__: string;
declare const __BG_KERNELS_VERSION__: string;
declare const __BG_SEMANTIC_CORE_VERSION__: string;
declare const __BG_SOURCE_REVISION__: string;
declare const __BG_JIT_SEMANTIC_PERMUTE_CAPTURE_JSON__: string;

const EVIDENCE_PREFIX = "[browsergrad-webgpu-evidence]";
const PRODUCER_VERSIONS = Object.freeze({
  "@unlocalhosted/browsergrad-jit": __BG_JIT_VERSION__,
  "@unlocalhosted/browsergrad-kernels": __BG_KERNELS_VERSION__,
  "@unlocalhosted/browsergrad-semantic-core": __BG_SEMANTIC_CORE_VERSION__,
  "browsergrad.backend.webgpu.view-copy": SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION,
});
const TERMINAL_EMITTER = createTerminalEvidenceEmitter(EVIDENCE_PREFIX, TERMINAL_EXPECTATION);

interface PreparedEvidenceCase {
  readonly fixture: DensePermutationFixtureCase;
  readonly plan: JsonObject;
  readonly requestEnvelope: JsonObject;
  readonly semanticRequestsJson: string;
  readonly inputValueId: number;
  readonly valueId: number;
  readonly requestCount: number;
  readonly legacyArgErased: boolean;
  readonly backendArtifactHash: string;
  readonly inputHash: string;
  readonly expectedOutputHash: string;
  readonly requestEnvelopeHash: string;
  readonly planProjectionHash: string;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly semanticSpecializationHash: string;
  readonly wgslModuleHash: string;
  readonly backendProfile: string;
  readonly backendVersion: string;
  readonly workgroupSize: number;
  readonly logicalInvocationCount: readonly [number, number, number];
  readonly plannedWorkgroupCount: readonly [number, number, number];
}

interface JitEmissionCaptureCase {
  readonly caseId: string;
  readonly plan: JsonObject;
  readonly semanticRequestsJson: string;
}

it("executes JIT semantic PERMUTE requests resident on a required real GPUDevice", async (context) => {
  const required = requiresWebGpuEvidence();
  let stage: TerminalStage = "suite-manifest";
  let currentCaseId: string | undefined;
  let artifactHash = await hashNamedComponents({ suiteId: SUITE_ID, plannedCaseIds: PLANNED_CASE_IDS });
  let artifactHashKind: TerminalEvidenceRecord["artifactHashKind"] = "planned-suite-manifest";
  let preparedBackendArtifactHash: string | undefined;
  let caseSetHash: string | undefined;
  let preparedCaseManifest: readonly PreparedCaseManifest[] | undefined;
  let environment = freezeEnvironment({ acquisition: "not-attempted" });
  let environmentId = await hashNamedComponents({ environment });
  let deviceProfileHash: string | undefined;
  let terminalEmitted = false;
  const completedCases: CaseObservation[] = [];
  const uncapturedErrors: string[] = [];
  let device: GPUDevice | undefined;
  let kernelDevice: KernelDevice | undefined;
  let deviceLostBeforeTerminal: GPUDeviceLostInfo | undefined;
  let deviceLoss: Promise<GPUDeviceLostInfo> | undefined;
  const uncapturedHandler = (event: GPUUncapturedErrorEvent) => uncapturedErrors.push(event.error.message);
  try {
    stage = "jit-submission-emission";
    const capturedCases = decodeJitEmissionCapture(
      __BG_JIT_SEMANTIC_PERMUTE_CAPTURE_JSON__,
    );
    stage = "fixture-and-semantic-preparation";
    const preparedCases = await Promise.all(
      DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases.map((fixture, index) =>
        prepareEvidenceCase(fixture, capturedCases[index]),
      ),
    );
    assertPreparedCases(preparedCases);
    const nextPreparedCaseManifest = Object.freeze(
      preparedCases.map(createPreparedCaseManifest),
    );
    const nextPreparedBackendArtifactHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      artifacts: preparedCases.map(({ fixture, backendArtifactHash }) => ({
        caseId: fixture.id,
        backendArtifactHash,
      })),
    });
    const nextCaseSetHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      comparisonPolicyId: COMPARISON_POLICY_ID,
      producerVersions: PRODUCER_VERSIONS,
      cases: nextPreparedCaseManifest,
    });
    const nextArtifactHashKind = "prepared-case-set" as const;
    const nextArtifactHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      artifactHashKind: nextArtifactHashKind,
      preparedBackendArtifactHash: nextPreparedBackendArtifactHash,
      caseSetHash: nextCaseSetHash,
      comparisonPolicyId: COMPARISON_POLICY_ID,
      producerVersions: PRODUCER_VERSIONS,
    });
    // Publish the prepared-state tuple only after every dependent digest exists.
    // A hash failure therefore leaves the prior planned-suite state intact.
    preparedCaseManifest = nextPreparedCaseManifest;
    preparedBackendArtifactHash = nextPreparedBackendArtifactHash;
    caseSetHash = nextCaseSetHash;
    artifactHashKind = nextArtifactHashKind;
    artifactHash = nextArtifactHash;

    stage = "device-acquisition";
    const acquisition = await acquireWebGpuEvidenceDevice();
    if (acquisition.kind === "unavailable") {
      const nextEnvironment = freezeEnvironment({
        acquisition: "navigator.gpu.requestAdapter/requestDevice",
        unavailableReason: acquisition.reason,
      });
      const nextEnvironmentId = await hashNamedComponents({
        environment: nextEnvironment,
      });
      environment = nextEnvironment;
      environmentId = nextEnvironmentId;
      await emitTerminalEvidence({
        required,
        artifactHash,
        artifactHashKind,
        preparedBackendArtifactHash,
        caseSetHash,
        preparedCases: preparedCaseManifest,
        environment,
        environmentId,
        outcome: required ? "failed" : "not-run",
        diagnosticCodes: [DEVICE_UNAVAILABLE_DIAGNOSTIC],
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

    const { adapter, device: acquiredDevice, adapterInfo } = acquisition.value;
    device = acquiredDevice;
    device.addEventListener("uncapturederror", uncapturedHandler);
    deviceLoss = device.lost.then((info) => {
      deviceLostBeforeTerminal = info;
      return info;
    });
    const adapterSupportedFeatures = Object.freeze([...adapter.features].map(String).sort());
    const negotiatedDeviceFeatures = Object.freeze([...device.features].map(String).sort());
    const negotiatedDeviceLimits = deviceLimits(device);
    const nextEnvironment = freezeEnvironment({
      acquisition: "navigator.gpu.requestAdapter/requestDevice",
      adapter: adapterInfo as unknown as JsonObject,
      adapterSupportedFeatures,
      negotiatedDeviceFeatures,
      negotiatedDeviceLimits,
    });
    const nextEnvironmentId = await hashNamedComponents({
      environment: nextEnvironment,
    });
    const nextDeviceProfileHash = await hashNamedComponents({
      backendId: BACKEND_ID,
      adapter: adapterInfo as unknown as JsonObject,
      selectedFeatures: [],
      adapterSupportedFeatures,
      negotiatedDeviceFeatures,
      negotiatedDeviceLimits,
    });
    // Publish environment provenance as one coherent tuple only after both
    // hashes have been computed from the same immutable facts.
    environment = nextEnvironment;
    environmentId = nextEnvironmentId;
    deviceProfileHash = nextDeviceProfileHash;
    stage = "kernel-device-construction";
    kernelDevice = await createDevice({ device });

    for (const preparedCase of preparedCases) {
      currentCaseId = preparedCase.fixture.id;
      stage = "resident-semantic-execution";
      const bridge = createWebGpuRealizerBridge(kernelDevice, { profiling: true });
      const statsBefore = kernelDevice.getStats();
      let root: number | undefined;
      let materializationBoundaryCount = 0;
      let observation: CaseObservation;
      try {
        const sourceWords = fixtureWords(preparedCase.fixture.sourceWords);
        root = await raceDeviceLoss(
          withEvidenceTimeout(
            bridge.run_tensor_plan_resident_semantic(
              preparedCase.plan,
              preparedCase.semanticRequestsJson,
              [{ value_id: preparedCase.inputValueId, data: wordsAsBytes(sourceWords) }],
              "float32",
            ),
            10_000,
            "run_tensor_plan_resident_semantic",
          ),
          deviceLoss,
        );
        const executionTrace = await raceDeviceLoss(
          withEvidenceTimeout(
            bridge.semanticTensorPlanExecutionTrace(root),
            10_000,
            "semanticTensorPlanExecutionTrace",
          ),
          deviceLoss,
        );
        const { request: actualRequest, profile: dispatchProfile } =
          assertExecutionTrace(preparedCase, executionTrace);
        const profileSnapshot = await bridge.flushProfiles();
        if (
          profileSnapshot.pendingProfileCount !== 0
          || profileSnapshot.passProfiles.length !== 1
        ) {
          throw new EvidenceLaneError(
            "BG-JIT-SEMANTIC-PERMUTE-TOPOLOGY",
            `${preparedCase.fixture.id} expected one settled dispatch profile`,
          );
        }
        const beforeReadback = bridge.resourceSnapshot();
        const expectedBytes = preparedCase.fixture.expectedOutputWords.length * 4;
        if (
          bridge.aliveHandleCount() !== 1
          || beforeReadback.aliveHandleCount !== 1
          || beforeReadback.currentOwnedGpuBytes !== expectedBytes
        ) {
          throw new EvidenceLaneError(
            "BG-JIT-SEMANTIC-PERMUTE-RESIDENCY",
            `${preparedCase.fixture.id} did not retain exactly one complete resident root before readback`,
          );
        }

        stage = "explicit-materialization-boundary";
        materializationBoundaryCount += 1;
        const outputBytes = await raceDeviceLoss(
          withEvidenceTimeout(
            bridge.materialize(
              root,
              fixtureExtentNumbers(preparedCase.fixture.outputShape),
              "float32",
            ),
            10_000,
            "materialize",
          ),
          deviceLoss,
        );
        const outputWords = bytesAsWords(outputBytes);
        const expectedWords = fixtureWords(preparedCase.fixture.expectedOutputWords);
        assertExactWords(preparedCase.fixture.id, outputWords, expectedWords);
        const actualOutputHash = await hashNamedComponents({
          caseId: preparedCase.fixture.id,
          outputWords: wordsToHex(outputWords),
        });
        if (actualOutputHash !== preparedCase.expectedOutputHash) {
          throw new EvidenceLaneError(
            "BG-JIT-SEMANTIC-PERMUTE-OUTPUT-HASH",
            `${preparedCase.fixture.id} complete output hash differs from the fixture`,
          );
        }
        const releasedRoot = root;
        bridge.release(releasedRoot);
        root = undefined;
        await assertTraceUnavailableAfterRelease(bridge, releasedRoot);
        if (
          bridge.aliveHandleCount() !== 0
          || bridge.resourceSnapshot().currentOwnedGpuBytes !== 0
        ) {
          throw new EvidenceLaneError(
            "BG-JIT-SEMANTIC-PERMUTE-RESIDENCY",
            `${preparedCase.fixture.id} retained bridge-owned buffers after release`,
          );
        }
        const statsAfter = kernelDevice.getStats();
        const pipelineCount =
          statsAfter.pipelineCacheMisses - statsBefore.pipelineCacheMisses;
        const kernelInvocationCount =
          statsAfter.kernelInvocations - statsBefore.kernelInvocations;
        if (pipelineCount !== 1 || kernelInvocationCount !== 1) {
          throw new EvidenceLaneError(
            "BG-JIT-SEMANTIC-PERMUTE-TOPOLOGY",
            `${preparedCase.fixture.id} expected one pipeline/invocation, got ${pipelineCount}/${kernelInvocationCount}`,
          );
        }
        if (materializationBoundaryCount !== 1) {
          throw new EvidenceLaneError(
            "BG-JIT-SEMANTIC-PERMUTE-READBACK",
            "expected one explicit materialization boundary",
          );
        }
        stage = "case-queue-drain";
        await raceDeviceLoss(
          withEvidenceTimeout(
            device.queue.onSubmittedWorkDone(),
            10_000,
            "case-queue-drain",
          ),
          deviceLoss,
        );
        observation = Object.freeze({
          caseId: preparedCase.fixture.id,
          backendArtifactHash: preparedCase.backendArtifactHash,
          inputHash: preparedCase.inputHash,
          expectedOutputHash: preparedCase.expectedOutputHash,
          actualOutputHash,
          requestEnvelopeHash: preparedCase.requestEnvelopeHash,
          planProjectionHash: preparedCase.planProjectionHash,
          layoutSemanticHash: actualRequest.layoutSemanticHash,
          kernelSemanticHash: actualRequest.kernelSemanticHash,
          semanticSpecializationHash: actualRequest.semanticSpecializationHash,
          wgslModuleHash: actualRequest.wgslModuleHash,
          backendProfile: actualRequest.backendProfile,
          backendVersion: actualRequest.backendVersion,
          requestSchema: "browsergrad.jit.tensor-plan-semantic-requests",
          requestVersion: "1.0",
          requestCount: executionTrace.preparation.requests.length,
          permuteValueId: actualRequest.valueId,
          legacyArgErased: preparedCase.legacyArgErased,
          executionEntrypoint: "run_tensor_plan_resident_semantic",
          rootResidentBeforeReadback: true,
          materializationBoundaryCount,
          pipelineCount,
          kernelInvocationCount,
          workgroupSize: actualRequest.workgroupSize,
          logicalInvocationCount: actualRequest.logicalInvocationCount,
          plannedWorkgroupCount: actualRequest.plannedWorkgroupCount,
          submittedWorkgroupCount: dispatchProfile.dispatchCount,
          submittedWorkgroupSize: dispatchProfile.workgroupSize,
          dispatchProfileLabel: dispatchProfile.label,
          dispatchTimingMode: dispatchProfile.timingMode,
          dispatchTimingConfidence: dispatchProfile.confidence,
          dispatchProfileCount: executionTrace.dispatchProfiles.length,
          actualPreparationMatchesManifest: true,
          comparisonPolicyId: COMPARISON_POLICY_ID,
        });
      } finally {
        if (root !== undefined) bridge.release(root);
      }
      // A case is complete only after production-scoped resident execution and
      // materialization, profile settlement, and the explicit queue drain.
      completedCases.push(observation);
    }

    stage = "late-error-drain";
    currentCaseId = undefined;
    await raceDeviceLoss(
      withEvidenceTimeout(device.queue.onSubmittedWorkDone(), 10_000, "final-queue-drain"),
      deviceLoss,
    );
    await nextTask();
    if (deviceLostBeforeTerminal !== undefined) {
      throw new EvidenceLaneError(
        "BG-JIT-SEMANTIC-PERMUTE-DEVICE-LOST",
        `${deviceLostBeforeTerminal.reason}: ${deviceLostBeforeTerminal.message}`,
      );
    }
    if (uncapturedErrors.length > 0) {
      throw new EvidenceLaneError("BG-JIT-SEMANTIC-PERMUTE-UNCAUGHT-GPU-ERROR", uncapturedErrors.join("; "));
    }
    stage = "terminal-summary";
    await emitTerminalEvidence({
      required,
      artifactHash,
      artifactHashKind,
      preparedBackendArtifactHash,
      caseSetHash,
      preparedCases: preparedCaseManifest,
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
    if (!terminalEmitted && !TERMINAL_EMITTER.emitted) {
      const failure = terminalFailure(error, stage, uncapturedErrors);
      await emitTerminalEvidence({
        required,
        artifactHash,
        artifactHashKind,
        ...(preparedBackendArtifactHash === undefined
          ? {}
          : { preparedBackendArtifactHash }),
        ...(caseSetHash === undefined ? {} : { caseSetHash }),
        ...(preparedCaseManifest === undefined
          ? {}
          : { preparedCases: preparedCaseManifest }),
        environment,
        environmentId,
        ...(deviceProfileHash === undefined ? {} : { deviceProfileHash }),
        outcome: "failed",
        diagnosticCodes: [failure.diagnosticCode],
        completedCases,
        stage,
        ...(currentCaseId === undefined ? {} : { currentCaseId }),
        uncapturedErrors,
        error: failure.error,
      });
    }
    throw error;
  } finally {
    device?.removeEventListener("uncapturederror", uncapturedHandler);
    kernelDevice?.clearCache();
    device?.destroy();
  }
});

async function prepareEvidenceCase(
  fixture: DensePermutationFixtureCase,
  capture: JitEmissionCaptureCase | undefined,
): Promise<PreparedEvidenceCase> {
  if (capture === undefined || capture.caseId !== fixture.id) {
    throw new EvidenceLaneError(
      "BG-JIT-SEMANTIC-PERMUTE-JIT-EMISSION",
      `${fixture.id} has no ordered JIT emission capture`,
    );
  }
  const plan = capture.plan;
  const requestEnvelope = parseJsonObject(
    capture.semanticRequestsJson,
    `${fixture.id} semantic request wire`,
  );
  if (JSON.stringify(requestEnvelope) !== capture.semanticRequestsJson) {
    throw new EvidenceLaneError(
      "BG-JIT-SEMANTIC-PERMUTE-JIT-EMISSION",
      `${fixture.id} semantic request wire is not canonical compact JSON`,
    );
  }
  const prepared = await prepareTensorPlanSemanticRequests(plan, requestEnvelope);
  const request = prepared.requests[0];
  if (request === undefined || prepared.requests.length !== 1) {
    throw new EvidenceLaneError("BG-JIT-SEMANTIC-PERMUTE-PREPARATION", `${fixture.id} expected one request`);
  }
  if (request.layoutSemanticHash !== fixture.layoutSemanticHash || request.kernelSemanticHash !== fixture.kernelSemanticHash) {
    throw new EvidenceLaneError("BG-JIT-SEMANTIC-PERMUTE-HASH", `${fixture.id} semantic hashes differ from fixture`);
  }
  assertRequestMatchesFixture(request, fixture);
  const { inputValueId, legacyArgErased } = derivePlanRouting(
    plan,
    request.valueId,
    fixture,
  );
  const [inputHash, expectedOutputHash, requestEnvelopeHash, planProjectionHash, backendArtifactHash] = await Promise.all([
    hashNamedComponents({ caseId: fixture.id, sourceWords: fixture.sourceWords }),
    hashNamedComponents({ caseId: fixture.id, outputWords: fixture.expectedOutputWords }),
    hashNamedComponents({ semanticRequestsJson: capture.semanticRequestsJson }),
    hashNamedComponents({ plan }),
    hashNamedComponents({
      caseId: fixture.id,
      layoutSemanticHash: request.layoutSemanticHash,
      kernelSemanticHash: request.kernelSemanticHash,
      semanticSpecializationHash: request.semanticSpecializationHash,
      wgslModuleHash: request.wgslModuleHash,
      backendProfile: request.backendProfile,
      backendVersion: request.backendVersion,
      workgroupSize: request.workgroupSize,
      logicalInvocationCount: request.logicalInvocationCount,
      plannedWorkgroupCount: request.plannedWorkgroupCount,
    }),
  ]);
  return Object.freeze({
    fixture,
    plan,
    requestEnvelope,
    semanticRequestsJson: capture.semanticRequestsJson,
    inputValueId,
    valueId: request.valueId,
    requestCount: prepared.requests.length,
    legacyArgErased,
    backendArtifactHash,
    inputHash,
    expectedOutputHash,
    requestEnvelopeHash,
    planProjectionHash,
    layoutSemanticHash: request.layoutSemanticHash,
    kernelSemanticHash: request.kernelSemanticHash,
    semanticSpecializationHash: request.semanticSpecializationHash,
    wgslModuleHash: request.wgslModuleHash,
    backendProfile: request.backendProfile,
    backendVersion: request.backendVersion,
    workgroupSize: request.workgroupSize,
    logicalInvocationCount: request.logicalInvocationCount,
    plannedWorkgroupCount: request.plannedWorkgroupCount,
  });
}

function createPreparedCaseManifest(
  prepared: PreparedEvidenceCase,
): PreparedCaseManifest {
  return Object.freeze({
    caseId: prepared.fixture.id,
    plan: prepared.plan,
    semanticRequestsJson: prepared.semanticRequestsJson,
    backendArtifactHash: prepared.backendArtifactHash,
    inputHash: prepared.inputHash,
    expectedOutputHash: prepared.expectedOutputHash,
    requestEnvelopeHash: prepared.requestEnvelopeHash,
    planProjectionHash: prepared.planProjectionHash,
    layoutSemanticHash: prepared.layoutSemanticHash,
    kernelSemanticHash: prepared.kernelSemanticHash,
    semanticSpecializationHash: prepared.semanticSpecializationHash,
    wgslModuleHash: prepared.wgslModuleHash,
    backendProfile: prepared.backendProfile,
    backendVersion: prepared.backendVersion,
    workgroupSize: prepared.workgroupSize,
    logicalInvocationCount: prepared.logicalInvocationCount,
    plannedWorkgroupCount: prepared.plannedWorkgroupCount,
    inputValueId: prepared.inputValueId,
    permuteValueId: prepared.valueId,
    requestCount: prepared.requestCount,
    legacyArgErased: prepared.legacyArgErased,
  });
}

function assertPreparedCases(cases: readonly PreparedEvidenceCase[]): void {
  const ids = cases.map(({ fixture }) => fixture.id);
  if (ids.length !== PLANNED_CASE_IDS.length || ids.some((id, index) => id !== PLANNED_CASE_IDS[index])) {
    throw new EvidenceLaneError("BG-JIT-SEMANTIC-PERMUTE-CASE-SET", "prepared case order/coverage differs from the shared fixture");
  }
}

function decodeJitEmissionCapture(raw: string): readonly JitEmissionCaptureCase[] {
  const envelope = parseJsonObject(raw, "JIT emission capture");
  if (envelope.schema === "browsergrad.jit.semantic-permute-emission-capture-failure") {
    const error = jsonRecord(envelope.error, "JIT emission capture failure");
    throw new EvidenceLaneError(
      "BG-JIT-SEMANTIC-PERMUTE-JIT-EMISSION",
      `${String(error.name)}: ${String(error.message)}`,
    );
  }
  if (envelope.schema !== "browsergrad.jit.semantic-permute-emission-capture") {
    throw new EvidenceLaneError(
      "BG-JIT-SEMANTIC-PERMUTE-JIT-EMISSION",
      "unknown JIT emission capture schema",
    );
  }
  const version = jsonRecord(envelope.version, "JIT emission capture version");
  if (version.major !== 1 || version.minor !== 0 || !Array.isArray(envelope.cases)) {
    throw new EvidenceLaneError(
      "BG-JIT-SEMANTIC-PERMUTE-JIT-EMISSION",
      "unsupported JIT emission capture version or case set",
    );
  }
  const captures = envelope.cases.map((input, index) => {
    const record = jsonRecord(input, `JIT emission capture case ${index}`);
    if (
      typeof record.caseId !== "string"
      || typeof record.semanticRequestsJson !== "string"
      || record.semanticRequestsJson.length > 512 * 1_024
    ) {
      throw new EvidenceLaneError(
        "BG-JIT-SEMANTIC-PERMUTE-JIT-EMISSION",
        `capture case ${index} has invalid routing fields`,
      );
    }
    return Object.freeze({
      caseId: record.caseId,
      plan: jsonRecord(record.plan, `JIT emission capture case ${index} plan`) as JsonObject,
      semanticRequestsJson: record.semanticRequestsJson,
    });
  });
  const ids = captures.map(({ caseId }) => caseId);
  if (
    ids.length !== PLANNED_CASE_IDS.length
    || ids.some((id, index) => id !== PLANNED_CASE_IDS[index])
  ) {
    throw new EvidenceLaneError(
      "BG-JIT-SEMANTIC-PERMUTE-JIT-EMISSION",
      "JIT emission capture differs from the complete ordered fixture case set",
    );
  }
  return Object.freeze(captures);
}

function derivePlanRouting(
  plan: JsonObject,
  expectedValueId: number,
  fixture: DensePermutationFixtureCase,
): Readonly<{ inputValueId: number; legacyArgErased: true }> {
  if (!Array.isArray(plan.steps) || plan.steps.length !== 3) {
    throw new EvidenceLaneError(
      "BG-JIT-SEMANTIC-PERMUTE-PLAN",
      `initial conformance plan must contain exactly BUFFER, LOAD, PERMUTE; got ${
        Array.isArray(plan.steps)
          ? plan.steps.map((step) => jsonRecord(step, "plan step").op).join(",")
          : "missing steps"
      }`,
    );
  }
  const steps = plan.steps.map((step, index) => jsonRecord(step, `plan step ${index}`));
  const input = steps[0]!;
  const load = steps[1]!;
  const permute = steps[2]!;
  const loadInputIds = load.input_ids;
  const inputIds = permute.input_ids;
  if (
    input.op !== "BUFFER"
    || load.op !== "LOAD"
    || permute.op !== "PERMUTE"
    || !Number.isSafeInteger(input.value_id)
    || !Number.isSafeInteger(load.value_id)
    || !Array.isArray(loadInputIds)
    || loadInputIds.length !== 1
    || loadInputIds[0] !== input.value_id
    || !Array.isArray(inputIds)
    || inputIds.length !== 1
    || inputIds[0] !== load.value_id
    || permute.value_id !== expectedValueId
    || permute.arg !== null
    || plan.root_id !== expectedValueId
    || plan.materialization_boundary !== "root"
    || plan.has_custom_ops !== false
    || !arrayEqual(permute.shape, fixtureExtentNumbers(fixture.outputShape))
  ) {
    throw new EvidenceLaneError(
      "BG-JIT-SEMANTIC-PERMUTE-PLAN",
      `${fixture.id} JIT plan routing/projection differs from the strict profile`,
    );
  }
  return Object.freeze({
    inputValueId: input.value_id as number,
    legacyArgErased: true,
  });
}

function assertRequestMatchesFixture(
  request: PreparedTensorPlanSemanticRequest,
  fixture: DensePermutationFixtureCase,
): void {
  if (
    request.kind !== fixture.request.kind
    || request.dtype !== fixture.request.dtype
    || !arrayEqual(request.inputShape, fixture.request.inputShape)
    || !arrayEqual(request.axes, fixture.request.axes)
  ) {
    throw new EvidenceLaneError(
      "BG-JIT-SEMANTIC-PERMUTE-JIT-EMISSION",
      `${fixture.id} emitted request differs from semantic-core fixture input`,
    );
  }
}

function assertExecutionTrace(
  preparedCase: PreparedEvidenceCase,
  trace: SemanticTensorPlanExecutionTrace,
): Readonly<{
  request: PreparedTensorPlanSemanticRequest;
  profile: DirectDispatchProfile;
}> {
  const request = trace.preparation.requests[0];
  const profile = trace.dispatchProfiles[0];
  if (
    request === undefined
    || trace.preparation.requests.length !== 1
    || profile === undefined
    || trace.dispatchProfiles.length !== 1
  ) {
    throw new EvidenceLaneError(
      "BG-JIT-SEMANTIC-PERMUTE-EXECUTION-TRACE",
      `${preparedCase.fixture.id} requires exactly one actual preparation and dispatch profile`,
    );
  }
  if (
    request.valueId !== preparedCase.valueId
    || request.layoutSemanticHash !== preparedCase.layoutSemanticHash
    || request.kernelSemanticHash !== preparedCase.kernelSemanticHash
    || request.semanticSpecializationHash !== preparedCase.semanticSpecializationHash
    || request.wgslModuleHash !== preparedCase.wgslModuleHash
    || request.backendProfile !== preparedCase.backendProfile
    || request.backendVersion !== preparedCase.backendVersion
    || request.workgroupSize !== preparedCase.workgroupSize
    || !arrayEqual(request.logicalInvocationCount, preparedCase.logicalInvocationCount)
    || !arrayEqual(request.plannedWorkgroupCount, preparedCase.plannedWorkgroupCount)
    || !arrayEqual(profile.dispatchCount, request.plannedWorkgroupCount)
    || !arrayEqual(profile.workgroupSize, [request.workgroupSize, 1, 1])
    || profile.unavailableReason !== undefined
    || !isAvailableTimingPair(profile.timingMode, profile.confidence)
  ) {
    throw new EvidenceLaneError(
      "BG-JIT-SEMANTIC-PERMUTE-EXECUTION-TRACE",
      `${preparedCase.fixture.id} actual preparation/dispatch differs from its bound manifest`,
    );
  }
  return Object.freeze({ request, profile });
}

function isAvailableTimingPair(mode: string, confidence: string): boolean {
  return (mode === "timestamp-query" && confidence === "exact")
    || (mode === "queue-completion" && confidence === "coarse");
}

async function assertTraceUnavailableAfterRelease(
  bridge: ReturnType<typeof createWebGpuRealizerBridge>,
  handle: number,
): Promise<void> {
  try {
    await bridge.semanticTensorPlanExecutionTrace(handle);
  } catch (error) {
    if (error instanceof Error && /unknown handle/u.test(error.message)) return;
    throw error;
  }
  throw new EvidenceLaneError(
    "BG-JIT-SEMANTIC-PERMUTE-RESIDENCY",
    "released semantic handle retained execution trace reachability",
  );
}

function assertExactWords(caseId: string, actual: Uint32Array, expected: Uint32Array): void {
  if (actual.length !== expected.length) {
    throw new EvidenceLaneError("BG-JIT-SEMANTIC-PERMUTE-BUFFER", `${caseId} output length differs`);
  }
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] !== expected[index]) {
      throw new EvidenceLaneError(
        "BG-JIT-SEMANTIC-PERMUTE-BITS",
        `${caseId} output word ${index} expected ${hex(expected[index]!)}, got ${hex(actual[index]!)}`,
      );
    }
  }
}

function wordsAsBytes(words: Uint32Array): Uint8Array {
  return new Uint8Array(words.buffer, words.byteOffset, words.byteLength);
}

function bytesAsWords(bytes: Uint8Array): Uint32Array {
  if (bytes.byteLength % 4 !== 0) throw new EvidenceLaneError("BG-JIT-SEMANTIC-PERMUTE-BUFFER", "output byte length is not u32-aligned");
  if ((bytes.byteOffset & 3) === 0) return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Uint32Array(copy.buffer);
}

function wordsToHex(words: Uint32Array): readonly string[] {
  return Object.freeze(Array.from(words, hex));
}

function hex(word: number): string {
  return word.toString(16).padStart(8, "0");
}

function freezeEnvironment(input: Readonly<{
  acquisition: string;
  adapter?: JsonObject;
  adapterSupportedFeatures?: readonly string[];
  negotiatedDeviceFeatures?: readonly string[];
  negotiatedDeviceLimits?: JsonObject;
  unavailableReason?: string;
}>): EvidenceEnvironment {
  return Object.freeze({
    schema: EXECUTION_ENVIRONMENT_SCHEMA,
    acquisition: input.acquisition,
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    ...(input.adapter === undefined ? {} : { adapter: Object.freeze({ ...input.adapter }) }),
    ...(input.adapterSupportedFeatures === undefined ? {} : { adapterSupportedFeatures: Object.freeze([...input.adapterSupportedFeatures]) }),
    ...(input.negotiatedDeviceFeatures === undefined ? {} : { negotiatedDeviceFeatures: Object.freeze([...input.negotiatedDeviceFeatures]) }),
    ...(input.negotiatedDeviceLimits === undefined ? {} : { negotiatedDeviceLimits: Object.freeze({ ...input.negotiatedDeviceLimits }) }),
    ...(input.unavailableReason === undefined ? {} : { unavailableReason: input.unavailableReason }),
  }) as EvidenceEnvironment;
}

function deviceLimits(device: GPUDevice): JsonObject {
  return Object.freeze({
    maxBufferSize: device.limits.maxBufferSize,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
    maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
    maxBindingsPerBindGroup: device.limits.maxBindingsPerBindGroup,
    maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
  });
}

async function emitTerminalEvidence(input: Readonly<{
  required: boolean;
  artifactHash: string;
  artifactHashKind: TerminalEvidenceRecord["artifactHashKind"];
  preparedBackendArtifactHash?: string;
  caseSetHash?: string;
  preparedCases?: readonly PreparedCaseManifest[];
  environment: EvidenceEnvironment;
  environmentId: string;
  deviceProfileHash?: string;
  outcome: "not-run" | "passed" | "failed";
  diagnosticCodes: readonly string[];
  completedCases: readonly CaseObservation[];
  stage: TerminalStage;
  currentCaseId?: string;
  uncapturedErrors: readonly string[];
  error?: JsonObject;
}>): Promise<void> {
  const unsignedRecord = Object.freeze({
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
      sourceRevision: __BG_SOURCE_REVISION__,
      ...(input.deviceProfileHash === undefined ? {} : { deviceProfileHash: input.deviceProfileHash }),
      recordedAt: new Date().toISOString(),
      outcome: input.outcome,
      comparisonPolicyId: COMPARISON_POLICY_ID,
      diagnosticCodes: Object.freeze([...input.diagnosticCodes]),
    }),
    environment: input.environment,
    artifactHashKind: input.artifactHashKind,
    ...(input.preparedBackendArtifactHash === undefined
      ? {}
      : { preparedBackendArtifactHash: input.preparedBackendArtifactHash }),
    ...(input.caseSetHash === undefined ? {} : { caseSetHash: input.caseSetHash }),
    ...(input.preparedCases === undefined
      ? {}
      : { preparedCases: Object.freeze([...input.preparedCases]) }),
    plannedCaseIds: PLANNED_CASE_IDS,
    completedCases: Object.freeze([...input.completedCases]),
    stage: input.stage,
    ...(input.currentCaseId === undefined ? {} : { currentCaseId: input.currentCaseId }),
    uncapturedErrors: Object.freeze([...input.uncapturedErrors]),
    ...(input.error === undefined ? {} : { error: input.error }),
  }) as UnsignedTerminalEvidenceRecord;
  const record = await finalizeTerminalEvidence(unsignedRecord, {
    expectedRequired: requiresWebGpuEvidence(),
    expectedSourceRevision: __BG_SOURCE_REVISION__,
    producerVersions: PRODUCER_VERSIONS,
    validatePreparedCaseManifest,
    validateObservation,
  });
  TERMINAL_EMITTER.emit(record);
}

function parseJsonObject(raw: string, name: string): JsonObject {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw) as unknown;
  } catch (error) {
    throw new EvidenceLaneError(
      "BG-JIT-SEMANTIC-PERMUTE-JIT-EMISSION",
      `${name} is invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  return deepFreezeJson(jsonRecord(parsed, name)) as JsonObject;
}

function jsonRecord(value: unknown, name: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new EvidenceLaneError(
      "BG-JIT-SEMANTIC-PERMUTE-JIT-EMISSION",
      `${name} must be an object`,
    );
  }
  return value as Record<string, unknown>;
}

function deepFreezeJson(value: unknown): unknown {
  if (Array.isArray(value)) {
    return Object.freeze(value.map(deepFreezeJson));
  }
  if (value !== null && typeof value === "object") {
    const frozen: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(value)) {
      frozen[key] = deepFreezeJson(child);
    }
    return Object.freeze(frozen);
  }
  if (
    value === null
    || typeof value === "string"
    || typeof value === "boolean"
    || (typeof value === "number" && Number.isFinite(value))
  ) {
    return value;
  }
  throw new EvidenceLaneError(
    "BG-JIT-SEMANTIC-PERMUTE-JIT-EMISSION",
    "captured JSON contains an unsupported value",
  );
}

function arrayEqual(left: unknown, right: readonly unknown[]): boolean {
  return Array.isArray(left)
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

async function raceDeviceLoss<T>(promise: Promise<T>, loss: Promise<GPUDeviceLostInfo>): Promise<T> {
  const result = await Promise.race([
    promise.then((value) => ({ kind: "value" as const, value })),
    loss.then((info) => ({ kind: "lost" as const, info })),
  ]);
  if (result.kind === "lost") {
    throw new EvidenceLaneError("BG-JIT-SEMANTIC-PERMUTE-DEVICE-LOST", `${result.info.reason}: ${result.info.message}`);
  }
  return result.value;
}

async function withEvidenceTimeout<T>(promise: Promise<T>, timeoutMs: number, name: string): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_resolve, reject) => {
        timeout = setTimeout(() => reject(new EvidenceLaneError(
          "BG-JIT-SEMANTIC-PERMUTE-TIMEOUT",
          `${name} did not settle within ${timeoutMs}ms`,
        )), timeoutMs);
      }),
    ]);
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
  }
}

function nextTask(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function terminalFailure(
  error: unknown,
  stage: TerminalStage,
  uncapturedErrors: readonly string[],
): Readonly<{ diagnosticCode: string; error: JsonObject }> {
  if (uncapturedErrors.length === 0) {
    const code = diagnosticCode(error, stage);
    return Object.freeze({
      diagnosticCode: code,
      error: { ...errorRecord(error), code },
    });
  }
  return Object.freeze({
    diagnosticCode: UNCAPTURED_GPU_ERROR_DIAGNOSTIC,
    error: {
      name: "GPUUncapturedError",
      message: uncapturedErrors.join("; "),
      code: UNCAPTURED_GPU_ERROR_DIAGNOSTIC,
      cause: errorRecord(error),
    },
  });
}

function diagnosticCode(error: unknown, stage: TerminalStage): string {
  if (error instanceof EvidenceLaneError) return error.code;
  if (
    error instanceof Error
    && error.name === "ScopedWebGpuIssueError"
    && "kind" in error
  ) {
    const kind = String(error.kind);
    if (kind === "validation") return "BG-JIT-SEMANTIC-PERMUTE-GPU-VALIDATION";
    if (kind === "out-of-memory") return "BG-JIT-SEMANTIC-PERMUTE-GPU-OUT-OF-MEMORY";
    if (kind === "device-lost") return "BG-JIT-SEMANTIC-PERMUTE-DEVICE-LOST";
    if (kind === "error-scope") return "BG-JIT-SEMANTIC-PERMUTE-ERROR-SCOPE";
    if (kind === "internal") return "BG-JIT-SEMANTIC-PERMUTE-GPU-INTERNAL";
  }
  if (stage === "jit-submission-emission") return "BG-JIT-SEMANTIC-PERMUTE-JIT-EMISSION";
  if (stage === "fixture-and-semantic-preparation") return "BG-JIT-SEMANTIC-PERMUTE-PREPARATION";
  if (stage === "device-acquisition") return "BG-JIT-SEMANTIC-PERMUTE-DEVICE-PROVENANCE";
  if (stage === "kernel-device-construction") return "BG-JIT-SEMANTIC-PERMUTE-DEVICE-WRAP";
  if (stage === "resident-semantic-execution" || stage === "explicit-materialization-boundary") {
    return "BG-JIT-SEMANTIC-PERMUTE-EXECUTION";
  }
  return "BG-JIT-SEMANTIC-PERMUTE-INTERNAL";
}

function errorRecord(error: unknown): JsonObject {
  if (error instanceof EvidenceLaneError) return { name: error.name, message: error.message, code: error.code };
  if (error instanceof Error) return { name: error.name || "Error", message: error.message || error.name || "Error" };
  return { name: "UnknownError", message: String(error) || "Unknown error" };
}

class EvidenceLaneError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "EvidenceLaneError";
  }
}
