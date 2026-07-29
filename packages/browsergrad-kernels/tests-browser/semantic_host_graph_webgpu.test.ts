import { expect, it } from "vitest";
import {
  EXECUTION_EVIDENCE_SCHEMA,
  acquireWebGpuEvidenceDevice,
  createTerminalEvidenceEmitter,
  createWebGpuExecutionEnvironmentRecord,
  nextWebGpuEvidenceTask,
  requiredEvidenceFailure,
  requiresWebGpuEvidence,
  webGpuSemanticDeviceLimits,
  withWebGpuEvidenceTimeout,
} from "../../../test-support/webgpu-evidence";

import {
  createVerifiedHostGraphArtifact,
  prepareHostGraphCpu,
  type HostGraphProgram,
  type VerifiedHostGraphArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/graph";
import {
  createVerifiedDensePermutationViewCopyArtifacts,
  type VerifiedViewCopyArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  hashNamedComponents,
  parseWireI64,
  parseWireU64,
  type JsonObject,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createDevice } from "../src/device";
import {
  SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
  SemanticHostGraphWebGpuError,
  destroySemanticHostGraphWebGpuPipeline,
  prepareSemanticHostGraphWebGpuPipeline,
  prepareSemanticHostGraphWebGpu,
  runSemanticHostGraphWebGpuPipeline,
  runSemanticHostGraphWebGpu,
  type PreparedSemanticHostGraphWebGpu,
  type SemanticHostGraphWebGpuControlBinding,
  type SemanticHostGraphWebGpuInputBinding,
} from "../src/semantic_host_graph";

declare const __BG_KERNELS_VERSION__: string;
declare const __BG_SEMANTIC_CORE_VERSION__: string;

const EVIDENCE_PREFIX = "[browsergrad-semantic-host-graph-webgpu-evidence]";
const SUITE_ID = "browsergrad.kernels.semantic-host-graph.webgpu-conformance@1";
const CAPABILITY_ID = "browsergrad.host-graph";
const BACKEND_ID = "browsergrad.backend.webgpu.core";
const COMPARISON_POLICY_ID =
  "browsergrad.comparison.cpu-reference-bit-exact-complete-outputs.v1";
const CASE_IDS = Object.freeze([
  "f32-rank-order-sum",
  "f32-signed-zero-min",
  "i32-wrapping-sum",
  "u8-whole-allocation-copy",
  "u32-exact-max",
  "f32-fixed-repeat-sum",
  "f32-runtime-repeat-zero",
  "f32-runtime-repeat-two",
  "f32-resource-repeat-zero",
  "f32-resource-repeat-two",
  "f32-dynamic-dispatch-one",
  "f32-dynamic-dispatch-two",
  "f32-resource-dynamic-dispatch-one",
  "f32-resource-dynamic-dispatch-two",
  "f32-aligned-dynamic-dispatch-64",
  "f32-aligned-dynamic-dispatch-128",
  "f32-aligned-resource-dynamic-dispatch-64",
  "f32-aligned-resource-dynamic-dispatch-128",
  "f32-unaligned-dynamic-dispatch-65",
  "f32-unaligned-dynamic-dispatch-127",
  "f32-unaligned-resource-dynamic-dispatch-65",
  "f32-unaligned-resource-dynamic-dispatch-127",
  "f32-rectangular-dynamic-rank2-small",
  "f32-rectangular-dynamic-rank2-large",
  "f32-rectangular-dynamic-rank3-small",
  "f32-rectangular-dynamic-rank3-large",
  "f32-resource-rectangular-dynamic-rank2-small",
  "f32-resource-rectangular-dynamic-rank2-large",
  "f32-resource-rectangular-dynamic-rank3-small",
  "f32-resource-rectangular-dynamic-rank3-large",
  "u8-input-conditional-then",
  "u8-input-conditional-else",
  "u8-runtime-conditional-then",
  "u8-runtime-conditional-else",
  "u8-resource-conditional-then",
  "u8-resource-conditional-else",
]);
const PRODUCER_VERSIONS = Object.freeze({
  "@unlocalhosted/browsergrad-kernels": __BG_KERNELS_VERSION__,
  "@unlocalhosted/browsergrad-semantic-core": __BG_SEMANTIC_CORE_VERSION__,
  "browsergrad.backend.webgpu.semantic-host-graph":
    SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
});
const TERMINAL_EMITTER = createTerminalEvidenceEmitter(EVIDENCE_PREFIX, {
  suiteId: SUITE_ID,
  capabilityId: CAPABILITY_ID,
  backendId: BACKEND_ID,
  comparisonPolicyId: COMPARISON_POLICY_ID,
  requireDeviceProfile: true,
});
const wire = (value: number): WireU64 => parseWireU64(String(value));

interface PreparedCase {
  readonly caseId: string;
  readonly artifacts?: VerifiedViewCopyArtifacts;
  readonly graph: VerifiedHostGraphArtifact;
  readonly prepared: PreparedSemanticHostGraphWebGpu;
  readonly inputs: readonly SemanticHostGraphWebGpuInputBinding[];
  readonly controls?: readonly SemanticHostGraphWebGpuControlBinding[];
  readonly artifactHash: string;
}

interface CaseObservation extends JsonObject {
  readonly caseId: string;
  readonly artifactHash: string;
  readonly graphSemanticHash: string;
  readonly pipelineIdentityHash: string;
  readonly backendSpecializationHash: string;
  readonly expandedStepCount: number;
  readonly dispatchStepCount: number;
  readonly copyStepCount: number;
  readonly materializationCount: number;
  readonly completedEventIds: readonly string[];
  readonly completedRepeats: readonly JsonObject[];
  readonly completedDynamicDispatches: readonly JsonObject[];
  readonly completedConditionals: readonly JsonObject[];
  readonly midGraphFeedbackCount: number;
  readonly collectiveReductionStepCount: number;
  readonly collectiveReplicationStepCount: number;
  readonly wgslModuleHashes: readonly string[];
  readonly submitted: boolean;
  readonly cpuComparison: "bit-exact-complete-outputs";
  readonly inputSnapshot:
    "caller-bindings-mutated-after-admission-bit-exact";
  readonly runtimeControlSnapshot?:
    "caller-controls-mutated-after-admission-bit-exact";
}

it("executes multi-rank host graphs on a required real GPUDevice", async (context) => {
  const required = requiresWebGpuEvidence();
  let stage = "fixture-construction";
  let artifactHash = await hashNamedComponents({
    suiteId: SUITE_ID,
    plannedCaseIds: CASE_IDS,
  });
  let environment = createWebGpuExecutionEnvironmentRecord({
    acquisition: "not-attempted",
  });
  let environmentId = await hashNamedComponents({ environment });
  let deviceProfileHash: string | undefined;
  let terminalEmitted = false;
  let deviceLossRefusalObserved = false;
  const completedCases: CaseObservation[] = [];
  const uncapturedErrors: string[] = [];
  let device: GPUDevice | undefined;
  let lossProbeDevice: GPUDevice | undefined;
  const uncapturedHandler = (event: GPUUncapturedErrorEvent) => {
    uncapturedErrors.push(event.error.message);
  };

  try {
    const cases = await Promise.all([
      prepareCase(
        "f32-rank-order-sum",
        "f32",
        "sum",
        [f32Bytes([1.5, -2]), f32Bytes([2.25, 5])],
      ),
      prepareCase(
        "f32-signed-zero-min",
        "f32",
        "min",
        [f32Bytes([0, -0]), f32Bytes([-0, 0])],
      ),
      prepareCase(
        "i32-wrapping-sum",
        "i32",
        "sum",
        [
          i32Bytes([2_147_483_647, -2]),
          i32Bytes([1, -3]),
        ],
      ),
      prepareRawCopyCase(),
      prepareCase(
        "u32-exact-max",
        "u32",
        "max",
        [
          u32Bytes([1, 0xffff_ffff]),
          u32Bytes([2, 5]),
        ],
      ),
      prepareRepeatedCollectiveCase(),
      prepareRuntimeRepeatedCollectiveCase(
        "f32-runtime-repeat-zero",
        0,
      ),
      prepareRuntimeRepeatedCollectiveCase(
        "f32-runtime-repeat-two",
        2,
      ),
      prepareResourceRepeatedCollectiveCase(
        "f32-resource-repeat-zero",
        0,
      ),
      prepareResourceRepeatedCollectiveCase(
        "f32-resource-repeat-two",
        2,
      ),
      prepareDynamicDispatchCase(
        "f32-dynamic-dispatch-one",
        1,
      ),
      prepareDynamicDispatchCase(
        "f32-dynamic-dispatch-two",
        2,
      ),
      prepareResourceDynamicDispatchCase(
        "f32-resource-dynamic-dispatch-one",
        1,
      ),
      prepareResourceDynamicDispatchCase(
        "f32-resource-dynamic-dispatch-two",
        2,
      ),
      prepareWideDynamicDispatchCase(
        "f32-aligned-dynamic-dispatch-64",
        64,
        false,
      ),
      prepareWideDynamicDispatchCase(
        "f32-aligned-dynamic-dispatch-128",
        128,
        false,
      ),
      prepareWideDynamicDispatchCase(
        "f32-aligned-resource-dynamic-dispatch-64",
        64,
        true,
      ),
      prepareWideDynamicDispatchCase(
        "f32-aligned-resource-dynamic-dispatch-128",
        128,
        true,
      ),
      prepareWideDynamicDispatchCase(
        "f32-unaligned-dynamic-dispatch-65",
        65,
        false,
      ),
      prepareWideDynamicDispatchCase(
        "f32-unaligned-dynamic-dispatch-127",
        127,
        false,
      ),
      prepareWideDynamicDispatchCase(
        "f32-unaligned-resource-dynamic-dispatch-65",
        65,
        true,
      ),
      prepareWideDynamicDispatchCase(
        "f32-unaligned-resource-dynamic-dispatch-127",
        127,
        true,
      ),
      prepareRectangularDynamicDispatchCase(
        "f32-rectangular-dynamic-rank2-small",
        [3, 4],
        [2, 3],
      ),
      prepareRectangularDynamicDispatchCase(
        "f32-rectangular-dynamic-rank2-large",
        [3, 4],
        [3, 4],
      ),
      prepareRectangularDynamicDispatchCase(
        "f32-rectangular-dynamic-rank3-small",
        [2, 3, 4],
        [1, 2, 3],
      ),
      prepareRectangularDynamicDispatchCase(
        "f32-rectangular-dynamic-rank3-large",
        [2, 3, 4],
        [2, 3, 4],
      ),
      prepareResourceRectangularDynamicDispatchCase(
        "f32-resource-rectangular-dynamic-rank2-small",
        [3, 4],
        [2, 3],
      ),
      prepareResourceRectangularDynamicDispatchCase(
        "f32-resource-rectangular-dynamic-rank2-large",
        [3, 4],
        [3, 4],
      ),
      prepareResourceRectangularDynamicDispatchCase(
        "f32-resource-rectangular-dynamic-rank3-small",
        [2, 3, 4],
        [1, 2, 3],
      ),
      prepareResourceRectangularDynamicDispatchCase(
        "f32-resource-rectangular-dynamic-rank3-large",
        [2, 3, 4],
        [2, 3, 4],
      ),
      prepareConditionalRawCopyCase(
        "u8-input-conditional-then",
        1,
      ),
      prepareConditionalRawCopyCase(
        "u8-input-conditional-else",
        0,
      ),
      prepareRuntimeConditionalRawCopyCase(
        "u8-runtime-conditional-then",
        1,
      ),
      prepareRuntimeConditionalRawCopyCase(
        "u8-runtime-conditional-else",
        0,
      ),
      prepareResourceConditionalRawCopyCase(
        "u8-resource-conditional-then",
        1,
      ),
      prepareResourceConditionalRawCopyCase(
        "u8-resource-conditional-else",
        0,
      ),
    ]);
    artifactHash = await hashNamedComponents({
      suiteId: SUITE_ID,
      cases: cases.map(({ caseId, artifactHash: caseHash }) => ({
        caseId,
        artifactHash: caseHash,
      })),
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
        environment,
        environmentId,
        outcome: required ? "failed" : "not-run",
        diagnosticCodes: [
          "BG-WEBGPU-GRAPH-EVIDENCE-DEVICE-UNAVAILABLE",
        ],
        completedCases,
        stage,
        uncapturedErrors,
      });
      terminalEmitted = true;
      if (required) throw requiredEvidenceFailure(acquisition.reason);
      context.skip(acquisition.reason);
      return;
    }

    const acquired = acquisition.value;
    device = acquired.device;
    device.addEventListener("uncapturederror", uncapturedHandler);
    const adapterFeatures = Object.freeze(
      [...acquired.adapter.features].map(String).sort(),
    );
    const deviceFeatures = Object.freeze(
      [...device.features].map(String).sort(),
    );
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
    const kernelDevice = await createDevice({ device });

    stage = "pipeline-budget-refusal";
    await expect(prepareSemanticHostGraphWebGpuPipeline(
      kernelDevice,
      cases[0]!.prepared,
      { maxPipelineCount: 1 },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
      path: "$.options.maxPipelineCount",
    });

    for (const preparedCase of cases) {
      stage = `pipeline-authority:${preparedCase.caseId}`;
      const preparedPipeline =
        await prepareSemanticHostGraphWebGpuPipeline(
          kernelDevice,
          preparedCase.prepared,
        );
      expect(preparedPipeline).toMatchObject({
        profile: "browsergrad.host-graph.webgpu-pipeline@1",
        backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
        graphSemanticHash: preparedCase.prepared.graphSemanticHash,
      });
      expect(preparedPipeline.stepCount)
        .toBe(preparedCase.prepared.expandedStepCount);
      expect(preparedPipeline.pipelineCount).toBeGreaterThan(0);
      stage = `execute:${preparedCase.caseId}`;
      const cpu = await prepareHostGraphCpu(
        preparedCase.graph,
        preparedCase.artifacts === undefined
          ? { kernelArtifacts: [], layoutArtifacts: [] }
          : artifactOptions(preparedCase.artifacts),
      );
      const mutableInputs = preparedCase.inputs.map((binding) => ({
        ...binding,
        bytes: new Uint8Array(binding.bytes),
      }));
      const mutableControls = preparedCase.controls?.map((binding) => ({
        ...binding,
      }));
      const expectedPromise = cpu.execute({
        inputs: preparedCase.inputs,
        ...(preparedCase.controls === undefined
          ? {}
          : { controls: preparedCase.controls }),
      });
      const actualPromise = runSemanticHostGraphWebGpuPipeline(
        preparedPipeline,
        {
          inputs: mutableInputs,
          ...(mutableControls === undefined
            ? {}
            : { controls: mutableControls }),
        },
      );
      for (const binding of mutableInputs) binding.bytes.fill(0);
      for (const binding of mutableControls ?? []) {
        binding.value = binding.value === wire(0) ? wire(1) : wire(0);
      }
      let expected: Awaited<typeof expectedPromise>;
      let actual: Awaited<typeof actualPromise>;
      try {
        [expected, actual] = await Promise.all([
          expectedPromise,
          actualPromise,
        ]);
      } finally {
        destroySemanticHostGraphWebGpuPipeline(preparedPipeline);
      }
      assertOutputEquality(actual.outputs, expected.outputs);
      expect(actual.trace.submitted).toBe(true);
      expect(actual.trace.executedNodeIds)
        .toEqual(expected.executedNodeIds);
      expect(actual.trace.completedEventIds)
        .toEqual(expected.completedEventIds);
      expect(actual.trace.completedRepeats)
        .toEqual(expected.completedRepeats);
      expect(actual.trace.completedDynamicDispatches)
        .toEqual(expected.completedDynamicDispatches);
      expect(actual.trace.completedConditionals)
        .toEqual(expected.completedConditionals);
      completedCases.push(Object.freeze({
        caseId: preparedCase.caseId,
        artifactHash: preparedCase.artifactHash,
        graphSemanticHash: actual.trace.graphSemanticHash,
        pipelineIdentityHash: actual.trace.pipelineIdentityHash,
        backendSpecializationHash:
          actual.trace.backendSpecializationHash,
        expandedStepCount: actual.trace.expandedStepCount,
        dispatchStepCount: actual.trace.dispatchStepCount,
        copyStepCount: actual.trace.copyStepCount,
        materializationCount: actual.trace.materializationCount,
        completedEventIds: actual.trace.completedEventIds,
        completedRepeats: actual.trace.completedRepeats,
        completedDynamicDispatches:
          actual.trace.completedDynamicDispatches,
        completedConditionals: actual.trace.completedConditionals,
        midGraphFeedbackCount: actual.trace.midGraphFeedbackCount,
        collectiveReductionStepCount:
          actual.trace.collectiveReductionStepCount,
        collectiveReplicationStepCount:
          actual.trace.collectiveReplicationStepCount,
        wgslModuleHashes: actual.trace.wgslModuleHashes,
        submitted: actual.trace.submitted,
        cpuComparison: "bit-exact-complete-outputs",
        inputSnapshot:
          "caller-bindings-mutated-after-admission-bit-exact",
        ...(mutableControls === undefined
          ? {}
          : {
              runtimeControlSnapshot:
                "caller-controls-mutated-after-admission-bit-exact" as const,
            }),
      }));
    }
    const thenConditional = completedCases.find(({ caseId }) =>
      caseId === "u8-input-conditional-then");
    const elseConditional = completedCases.find(({ caseId }) =>
      caseId === "u8-input-conditional-else");
    expect(thenConditional?.pipelineIdentityHash)
      .toBe(elseConditional?.pipelineIdentityHash);
    expect(thenConditional?.backendSpecializationHash)
      .not.toBe(elseConditional?.backendSpecializationHash);
    const thenRuntimeConditional = completedCases.find(({ caseId }) =>
      caseId === "u8-runtime-conditional-then");
    const elseRuntimeConditional = completedCases.find(({ caseId }) =>
      caseId === "u8-runtime-conditional-else");
    expect(thenRuntimeConditional?.pipelineIdentityHash)
      .toBe(elseRuntimeConditional?.pipelineIdentityHash);
    expect(thenRuntimeConditional?.backendSpecializationHash)
      .not.toBe(elseRuntimeConditional?.backendSpecializationHash);
    const thenResourceConditional = completedCases.find(({ caseId }) =>
      caseId === "u8-resource-conditional-then");
    const elseResourceConditional = completedCases.find(({ caseId }) =>
      caseId === "u8-resource-conditional-else");
    expect(thenResourceConditional?.pipelineIdentityHash)
      .toBe(elseResourceConditional?.pipelineIdentityHash);
    expect(thenResourceConditional?.backendSpecializationHash)
      .not.toBe(elseResourceConditional?.backendSpecializationHash);
    expect(thenResourceConditional?.midGraphFeedbackCount).toBe(1);
    expect(elseResourceConditional?.midGraphFeedbackCount).toBe(1);
    const zeroRuntimeRepeat = completedCases.find(({ caseId }) =>
      caseId === "f32-runtime-repeat-zero");
    const twoRuntimeRepeat = completedCases.find(({ caseId }) =>
      caseId === "f32-runtime-repeat-two");
    expect(zeroRuntimeRepeat?.pipelineIdentityHash)
      .toBe(twoRuntimeRepeat?.pipelineIdentityHash);
    expect(zeroRuntimeRepeat?.backendSpecializationHash)
      .not.toBe(twoRuntimeRepeat?.backendSpecializationHash);
    expect(zeroRuntimeRepeat?.expandedStepCount).toBe(2);
    expect(twoRuntimeRepeat?.expandedStepCount).toBe(6);
    const zeroResourceRepeat = completedCases.find(({ caseId }) =>
      caseId === "f32-resource-repeat-zero");
    const twoResourceRepeat = completedCases.find(({ caseId }) =>
      caseId === "f32-resource-repeat-two");
    expect(zeroResourceRepeat?.pipelineIdentityHash)
      .toBe(twoResourceRepeat?.pipelineIdentityHash);
    expect(zeroResourceRepeat?.backendSpecializationHash)
      .not.toBe(twoResourceRepeat?.backendSpecializationHash);
    expect(zeroResourceRepeat?.expandedStepCount).toBe(4);
    expect(twoResourceRepeat?.expandedStepCount).toBe(8);
    expect(zeroResourceRepeat?.midGraphFeedbackCount).toBe(1);
    expect(twoResourceRepeat?.midGraphFeedbackCount).toBe(1);
    const oneDynamicDispatch = completedCases.find(({ caseId }) =>
      caseId === "f32-dynamic-dispatch-one");
    const twoDynamicDispatch = completedCases.find(({ caseId }) =>
      caseId === "f32-dynamic-dispatch-two");
    expect(oneDynamicDispatch?.pipelineIdentityHash)
      .toBe(twoDynamicDispatch?.pipelineIdentityHash);
    expect(oneDynamicDispatch?.backendSpecializationHash)
      .not.toBe(twoDynamicDispatch?.backendSpecializationHash);
    expect(oneDynamicDispatch?.expandedStepCount).toBe(2);
    expect(twoDynamicDispatch?.expandedStepCount).toBe(2);
    const oneResourceDynamicDispatch = completedCases.find(({ caseId }) =>
      caseId === "f32-resource-dynamic-dispatch-one");
    const twoResourceDynamicDispatch = completedCases.find(({ caseId }) =>
      caseId === "f32-resource-dynamic-dispatch-two");
    expect(oneResourceDynamicDispatch?.pipelineIdentityHash)
      .toBe(twoResourceDynamicDispatch?.pipelineIdentityHash);
    expect(oneResourceDynamicDispatch?.backendSpecializationHash)
      .not.toBe(twoResourceDynamicDispatch?.backendSpecializationHash);
    expect(oneResourceDynamicDispatch?.expandedStepCount).toBe(4);
    expect(twoResourceDynamicDispatch?.expandedStepCount).toBe(4);
    expect(oneResourceDynamicDispatch?.midGraphFeedbackCount).toBe(1);
    expect(twoResourceDynamicDispatch?.midGraphFeedbackCount).toBe(1);
    const alignedDynamic64 = completedCases.find(({ caseId }) =>
      caseId === "f32-aligned-dynamic-dispatch-64");
    const alignedDynamic128 = completedCases.find(({ caseId }) =>
      caseId === "f32-aligned-dynamic-dispatch-128");
    expect(alignedDynamic64?.pipelineIdentityHash)
      .toBe(alignedDynamic128?.pipelineIdentityHash);
    expect(alignedDynamic64?.backendSpecializationHash)
      .not.toBe(alignedDynamic128?.backendSpecializationHash);
    expect(alignedDynamic64?.expandedStepCount).toBe(2);
    expect(alignedDynamic128?.expandedStepCount).toBe(2);
    const alignedResourceDynamic64 = completedCases.find(({ caseId }) =>
      caseId === "f32-aligned-resource-dynamic-dispatch-64");
    const alignedResourceDynamic128 = completedCases.find(({ caseId }) =>
      caseId === "f32-aligned-resource-dynamic-dispatch-128");
    expect(alignedResourceDynamic64?.pipelineIdentityHash)
      .toBe(alignedResourceDynamic128?.pipelineIdentityHash);
    expect(alignedResourceDynamic64?.backendSpecializationHash)
      .not.toBe(alignedResourceDynamic128?.backendSpecializationHash);
    expect(alignedResourceDynamic64?.expandedStepCount).toBe(4);
    expect(alignedResourceDynamic128?.expandedStepCount).toBe(4);
    expect(alignedResourceDynamic64?.midGraphFeedbackCount).toBe(1);
    expect(alignedResourceDynamic128?.midGraphFeedbackCount).toBe(1);
    const unalignedDynamic65 = completedCases.find(({ caseId }) =>
      caseId === "f32-unaligned-dynamic-dispatch-65");
    const unalignedDynamic127 = completedCases.find(({ caseId }) =>
      caseId === "f32-unaligned-dynamic-dispatch-127");
    expect(unalignedDynamic65?.pipelineIdentityHash)
      .toBe(alignedDynamic64?.pipelineIdentityHash);
    expect(unalignedDynamic127?.pipelineIdentityHash)
      .toBe(alignedDynamic64?.pipelineIdentityHash);
    expect(unalignedDynamic65?.backendSpecializationHash)
      .not.toBe(unalignedDynamic127?.backendSpecializationHash);
    expect(unalignedDynamic65?.expandedStepCount).toBe(2);
    expect(unalignedDynamic127?.expandedStepCount).toBe(2);
    const unalignedResourceDynamic65 = completedCases.find(({ caseId }) =>
      caseId === "f32-unaligned-resource-dynamic-dispatch-65");
    const unalignedResourceDynamic127 = completedCases.find(({ caseId }) =>
      caseId === "f32-unaligned-resource-dynamic-dispatch-127");
    expect(unalignedResourceDynamic65?.pipelineIdentityHash)
      .toBe(alignedResourceDynamic64?.pipelineIdentityHash);
    expect(unalignedResourceDynamic127?.pipelineIdentityHash)
      .toBe(alignedResourceDynamic64?.pipelineIdentityHash);
    expect(unalignedResourceDynamic65?.backendSpecializationHash)
      .not.toBe(unalignedResourceDynamic127?.backendSpecializationHash);
    expect(unalignedResourceDynamic65?.expandedStepCount).toBe(4);
    expect(unalignedResourceDynamic127?.expandedStepCount).toBe(4);
    expect(unalignedResourceDynamic65?.midGraphFeedbackCount).toBe(1);
    expect(unalignedResourceDynamic127?.midGraphFeedbackCount).toBe(1);
    for (const rank of [2, 3] as const) {
      const small = completedCases.find(({ caseId }) =>
        caseId === `f32-rectangular-dynamic-rank${rank}-small`);
      const large = completedCases.find(({ caseId }) =>
        caseId === `f32-rectangular-dynamic-rank${rank}-large`);
      expect(small?.pipelineIdentityHash).toBe(large?.pipelineIdentityHash);
      expect(small?.backendSpecializationHash)
        .not.toBe(large?.backendSpecializationHash);
      expect(small?.expandedStepCount).toBe(2);
      expect(large?.expandedStepCount).toBe(2);
      expect(small?.completedDynamicDispatches).toHaveLength(1);
      expect(large?.completedDynamicDispatches).toHaveLength(1);
      const resourceSmall = completedCases.find(({ caseId }) =>
        caseId ===
          `f32-resource-rectangular-dynamic-rank${rank}-small`);
      const resourceLarge = completedCases.find(({ caseId }) =>
        caseId ===
          `f32-resource-rectangular-dynamic-rank${rank}-large`);
      expect(resourceSmall?.pipelineIdentityHash)
        .toBe(resourceLarge?.pipelineIdentityHash);
      expect(resourceSmall?.backendSpecializationHash)
        .not.toBe(resourceLarge?.backendSpecializationHash);
      expect(resourceSmall?.expandedStepCount).toBe(rank * 2 + 2);
      expect(resourceLarge?.expandedStepCount).toBe(rank * 2 + 2);
      expect(resourceSmall?.completedDynamicDispatches).toHaveLength(1);
      expect(resourceLarge?.completedDynamicDispatches).toHaveLength(1);
      expect(resourceSmall?.midGraphFeedbackCount).toBe(1);
      expect(resourceLarge?.midGraphFeedbackCount).toBe(1);
    }

    stage = "resource-repeat-bound-refusal";
    const resourceRepeatCase = cases.find(({ caseId }) =>
      caseId === "f32-resource-repeat-two");
    if (resourceRepeatCase === undefined) {
      throw new Error("missing resource repeat case");
    }
    await expect(runSemanticHostGraphWebGpu(
      kernelDevice,
      resourceRepeatCase.prepared,
      {
        inputs: resourceRepeatCase.inputs.map((binding) => ({
          ...binding,
          bytes:
            binding.resourceId === "iteration-input" &&
              binding.rank === wire(0)
              ? u32Bytes([4])
              : new Uint8Array(binding.bytes),
        })),
      },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
      path: "$.nodes.repeat-reduction.iterationSource",
    });

    stage = "resource-dynamic-dispatch-bound-refusal";
    const resourceDynamicDispatchCase = cases.find(({ caseId }) =>
      caseId === "f32-resource-dynamic-dispatch-two");
    if (resourceDynamicDispatchCase === undefined) {
      throw new Error("missing resource dynamic dispatch case");
    }
    for (const elementCount of [0, 3]) {
      await expect(runSemanticHostGraphWebGpu(
        kernelDevice,
        resourceDynamicDispatchCase.prepared,
        {
          inputs: resourceDynamicDispatchCase.inputs.map((binding) => ({
            ...binding,
            bytes:
              binding.resourceId === "launch-input" &&
                binding.rank === wire(0)
                ? u32Bytes([elementCount])
                : new Uint8Array(binding.bytes),
          })),
        },
      )).rejects.toMatchObject({
        code: "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
        path: "$.nodes.copy.launchSource",
      });
    }

    stage = "resource-rectangular-dynamic-dispatch-bound-refusal";
    const resourceRectangularCase = cases.find(({ caseId }) =>
      caseId === "f32-resource-rectangular-dynamic-rank2-small");
    if (resourceRectangularCase === undefined) {
      throw new Error("missing resource rectangular dynamic dispatch case");
    }
    for (const [axis, extent] of [[0, 0], [1, 5]] as const) {
      await expect(runSemanticHostGraphWebGpu(
        kernelDevice,
        resourceRectangularCase.prepared,
        {
          inputs: resourceRectangularCase.inputs.map((binding) => ({
            ...binding,
            bytes:
              binding.resourceId === `extent-input-${axis}` &&
                binding.rank === wire(0)
                ? u32Bytes([extent])
                : new Uint8Array(binding.bytes),
          })),
        },
      )).rejects.toMatchObject({
        code: "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
        path: `$.nodes.copy.launchSources[${axis}]`,
      });
    }

    stage = "non-finite-fail-stop";
    const finiteCase = cases[0] as PreparedCase;
    const invalidInputs = [
      input(0, f32Bytes([Number.NaN, 1])),
      input(1, f32Bytes([2, 3])),
    ];
    await expect(runSemanticHostGraphWebGpu(
      kernelDevice,
      finiteCase.prepared,
      { inputs: invalidInputs },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-NUMERICAL-DOMAIN",
    });

    stage = "device-loss-refusal";
    const lossProbeAcquisition = await acquireWebGpuEvidenceDevice();
    if (lossProbeAcquisition.kind === "unavailable") {
      throw new EvidenceLaneError(
        "BG-WEBGPU-GRAPH-EVIDENCE-DEVICE-LOSS-PROBE-UNAVAILABLE",
        lossProbeAcquisition.reason,
      );
    }
    const disposableDevice = lossProbeAcquisition.value.device;
    lossProbeDevice = disposableDevice;
    const disposableKernelDevice = await createDevice({
      device: disposableDevice,
    });
    const disposablePipeline =
      await prepareSemanticHostGraphWebGpuPipeline(
        disposableKernelDevice,
        finiteCase.prepared,
      );
    await runSemanticHostGraphWebGpuPipeline(
      disposablePipeline,
      { inputs: finiteCase.inputs },
    );
    disposableDevice.destroy();
    await disposableDevice.lost;
    await nextWebGpuEvidenceTask();
    await expect(runSemanticHostGraphWebGpuPipeline(
      disposablePipeline,
      { inputs: finiteCase.inputs },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-DEVICE-LOST",
    });
    destroySemanticHostGraphWebGpuPipeline(disposablePipeline);
    await expect(runSemanticHostGraphWebGpuPipeline(
      disposablePipeline,
      { inputs: finiteCase.inputs },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-UNVERIFIED-PIPELINE",
    });
    await expect(runSemanticHostGraphWebGpuPipeline(
      { ...disposablePipeline },
      { inputs: finiteCase.inputs },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-UNVERIFIED-PIPELINE",
    });
    await expect(runSemanticHostGraphWebGpu(
      disposableKernelDevice,
      finiteCase.prepared,
      { inputs: finiteCase.inputs },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-DEVICE-LOST",
    });
    deviceLossRefusalObserved = true;

    stage = "late-error-drain";
    await withGraphTimeout(
      device.queue.onSubmittedWorkDone(),
      10_000,
      "queue-drain",
    );
    await nextWebGpuEvidenceTask();
    if (uncapturedErrors.length > 0) {
      throw new EvidenceLaneError(
        "BG-WEBGPU-GRAPH-EVIDENCE-UNCAUGHT-GPU-ERROR",
        uncapturedErrors.join("; "),
      );
    }
    stage = "terminal-summary";
    emitTerminal({
      required,
      artifactHash,
      environment,
      environmentId,
      deviceProfileHash,
      outcome: "passed",
      diagnosticCodes: [],
      completedCases,
      deviceLossRefusalObserved,
      stage,
      uncapturedErrors,
    });
    terminalEmitted = true;
  } catch (error) {
    if (!terminalEmitted) {
      emitTerminal({
        required,
        artifactHash,
        environment,
        environmentId,
        ...(deviceProfileHash === undefined
          ? {}
          : { deviceProfileHash }),
        outcome: "failed",
        diagnosticCodes: [diagnosticCode(error)],
        completedCases,
        deviceLossRefusalObserved,
        stage,
        uncapturedErrors,
        error: errorRecord(error),
      });
      terminalEmitted = true;
    }
    throw error;
  } finally {
    if (device !== undefined) {
      device.removeEventListener("uncapturederror", uncapturedHandler);
      device.destroy();
    }
    lossProbeDevice?.destroy();
  }
});

async function prepareCase(
  caseId: string,
  dtype: "f32" | "i32" | "u32",
  reduction: "sum" | "min" | "max",
  values: readonly Uint8Array[],
): Promise<PreparedCase> {
  const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: [parseWireI64("2")],
    axes: [0],
    dtype,
  });
  const graph = (await createVerifiedHostGraphArtifact(
    collectiveProgram(artifacts, dtype, reduction),
    artifactOptions(artifacts),
  )).artifact;
  const prepared = await prepareSemanticHostGraphWebGpu(
    graph,
    artifactOptions(artifacts),
  );
  const inputs = Object.freeze(values.map((bytes, rank) =>
    input(rank, bytes)));
  return Object.freeze({
    caseId,
    artifacts,
    graph,
    prepared,
    inputs,
    artifactHash: await hashNamedComponents({
      caseId,
      graph: prepared.graphSemanticHash,
      modules: prepared.wgslModuleHashes,
      inputs: values.map((bytes) => Array.from(bytes)),
    }),
  });
}

async function prepareRawCopyCase(): Promise<PreparedCase> {
  const caseId = "u8-whole-allocation-copy";
  const values = [
    new Uint8Array([0, 1, 2, 3, 4, 5, 6, 255]),
    new Uint8Array([255, 6, 5, 4, 3, 2, 1, 0]),
  ];
  const graph = (await createVerifiedHostGraphArtifact(
    rawCopyProgram(),
    { kernelArtifacts: [], layoutArtifacts: [] },
  )).artifact;
  const prepared = await prepareSemanticHostGraphWebGpu(graph, {
    kernelArtifacts: [],
    layoutArtifacts: [],
  });
  const inputs = Object.freeze(values.map((bytes, rank) =>
    input(rank, bytes)));
  return Object.freeze({
    caseId,
    graph,
    prepared,
    inputs,
    artifactHash: await hashNamedComponents({
      caseId,
      graph: prepared.graphSemanticHash,
      modules: prepared.wgslModuleHashes,
      inputs: values.map((bytes) => Array.from(bytes)),
    }),
  });
}

async function prepareRepeatedCollectiveCase(): Promise<PreparedCase> {
  const caseId = "f32-fixed-repeat-sum";
  const values = [f32Bytes([1, 2]), f32Bytes([3, 4])];
  const graph = (await createVerifiedHostGraphArtifact(
    repeatedCollectiveProgram(),
    { kernelArtifacts: [], layoutArtifacts: [] },
  )).artifact;
  const prepared = await prepareSemanticHostGraphWebGpu(graph, {
    kernelArtifacts: [],
    layoutArtifacts: [],
  });
  const inputs = Object.freeze(values.map((bytes, rank) =>
    input(rank, bytes)));
  return Object.freeze({
    caseId,
    graph,
    prepared,
    inputs,
    artifactHash: await hashNamedComponents({
      caseId,
      graph: prepared.graphSemanticHash,
      modules: prepared.wgslModuleHashes,
      inputs: values.map((bytes) => Array.from(bytes)),
    }),
  });
}

async function prepareRuntimeRepeatedCollectiveCase(
  caseId: "f32-runtime-repeat-zero" | "f32-runtime-repeat-two",
  iterationCount: 0 | 2,
): Promise<PreparedCase> {
  const values = [f32Bytes([1, 2]), f32Bytes([3, 4])];
  const graph = (await createVerifiedHostGraphArtifact(
    runtimeRepeatedCollectiveProgram(),
    { kernelArtifacts: [], layoutArtifacts: [] },
  )).artifact;
  const prepared = await prepareSemanticHostGraphWebGpu(graph, {
    kernelArtifacts: [],
    layoutArtifacts: [],
  });
  const inputs = Object.freeze(values.map((bytes, rank) =>
    input(rank, bytes)));
  const controls = Object.freeze([{
    controlId: "iterations",
    value: wire(iterationCount),
  }]);
  return Object.freeze({
    caseId,
    graph,
    prepared,
    inputs,
    controls,
    artifactHash: await hashNamedComponents({
      caseId,
      graph: prepared.graphSemanticHash,
      modules: prepared.wgslModuleHashes,
      inputs: values.map((bytes) => Array.from(bytes)),
      controls,
    }),
  });
}

async function prepareResourceRepeatedCollectiveCase(
  caseId:
    | "f32-resource-repeat-zero"
    | "f32-resource-repeat-two",
  iterationCount: 0 | 2,
): Promise<PreparedCase> {
  const values = [f32Bytes([1, 2]), f32Bytes([3, 4])];
  const graph = (await createVerifiedHostGraphArtifact(
    resourceRepeatedCollectiveProgram(),
    { kernelArtifacts: [], layoutArtifacts: [] },
  )).artifact;
  const prepared = await prepareSemanticHostGraphWebGpu(graph, {
    kernelArtifacts: [],
    layoutArtifacts: [],
  });
  const inputs = Object.freeze([
    ...values.map((bytes, rank) => input(rank, bytes)),
    namedInput(
      0,
      "iteration-input",
      u32Bytes([iterationCount]),
    ),
    namedInput(1, "iteration-input", u32Bytes([0])),
  ]);
  return Object.freeze({
    caseId,
    graph,
    prepared,
    inputs,
    artifactHash: await hashNamedComponents({
      caseId,
      graph: prepared.graphSemanticHash,
      modules: prepared.wgslModuleHashes,
      inputs: inputs.map(({ rank, resourceId, bytes }) => ({
        rank,
        resourceId,
        bytes: Array.from(bytes),
      })),
    }),
  });
}

async function prepareDynamicDispatchCase(
  caseId:
    | "f32-dynamic-dispatch-one"
    | "f32-dynamic-dispatch-two",
  elementCount: 1 | 2,
): Promise<PreparedCase> {
  const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: [parseWireI64("2")],
    axes: [0],
    dtype: "f32",
  });
  const values = [f32Bytes([1.25, -2.5]), f32Bytes([3.5, 4.75])];
  const graph = (await createVerifiedHostGraphArtifact(
    dynamicDispatchProgram(artifacts),
    artifactOptions(artifacts),
  )).artifact;
  const prepared = await prepareSemanticHostGraphWebGpu(
    graph,
    { ...artifactOptions(artifacts), workgroupSize: 64 },
  );
  const inputs = Object.freeze(values.map((bytes, rank) =>
    input(rank, bytes)));
  const controls = Object.freeze([{
    controlId: "prefix-elements",
    value: wire(elementCount),
  }]);
  return Object.freeze({
    caseId,
    artifacts,
    graph,
    prepared,
    inputs,
    controls,
    artifactHash: await hashNamedComponents({
      caseId,
      graph: prepared.graphSemanticHash,
      modules: prepared.wgslModuleHashes,
      inputs: values.map((bytes) => Array.from(bytes)),
      controls,
    }),
  });
}

async function prepareResourceDynamicDispatchCase(
  caseId:
    | "f32-resource-dynamic-dispatch-one"
    | "f32-resource-dynamic-dispatch-two",
  elementCount: 1 | 2,
): Promise<PreparedCase> {
  const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: [parseWireI64("2")],
    axes: [0],
    dtype: "f32",
  });
  const values = [f32Bytes([1.25, -2.5]), f32Bytes([3.5, 4.75])];
  const graph = (await createVerifiedHostGraphArtifact(
    resourceDynamicDispatchProgram(artifacts),
    artifactOptions(artifacts),
  )).artifact;
  const prepared = await prepareSemanticHostGraphWebGpu(
    graph,
    { ...artifactOptions(artifacts), workgroupSize: 64 },
  );
  const inputs = Object.freeze([
    ...values.map((bytes, rank) => input(rank, bytes)),
    namedInput(0, "launch-input", u32Bytes([elementCount])),
    namedInput(1, "launch-input", u32Bytes([0])),
  ]);
  return Object.freeze({
    caseId,
    artifacts,
    graph,
    prepared,
    inputs,
    artifactHash: await hashNamedComponents({
      caseId,
      graph: prepared.graphSemanticHash,
      modules: prepared.wgslModuleHashes,
      inputs: inputs.map(({ rank, resourceId, bytes }) => ({
        rank,
        resourceId,
        bytes: Array.from(bytes),
      })),
    }),
  });
}

async function prepareWideDynamicDispatchCase(
  caseId:
    | "f32-aligned-dynamic-dispatch-64"
    | "f32-aligned-dynamic-dispatch-128"
    | "f32-aligned-resource-dynamic-dispatch-64"
    | "f32-aligned-resource-dynamic-dispatch-128"
    | "f32-unaligned-dynamic-dispatch-65"
    | "f32-unaligned-dynamic-dispatch-127"
    | "f32-unaligned-resource-dynamic-dispatch-65"
    | "f32-unaligned-resource-dynamic-dispatch-127",
  elementCount: 64 | 65 | 127 | 128,
  resourceControlled: boolean,
): Promise<PreparedCase> {
  const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: [parseWireI64("128")],
    axes: [0],
    dtype: "f32",
  });
  const values = [
    f32Bytes(Array.from({ length: 128 }, (_, index) => index + 0.25)),
    f32Bytes(Array.from({ length: 128 }, (_, index) => -index - 0.5)),
  ];
  const graph = (await createVerifiedHostGraphArtifact(
    resourceControlled
      ? resourceDynamicDispatchProgram(artifacts, 128)
      : dynamicDispatchProgram(artifacts, 128),
    artifactOptions(artifacts),
  )).artifact;
  const prepared = await prepareSemanticHostGraphWebGpu(
    graph,
    { ...artifactOptions(artifacts), workgroupSize: 64 },
  );
  const inputs = Object.freeze([
    ...values.map((bytes, rank) => input(rank, bytes)),
    ...(resourceControlled
      ? [
          namedInput(0, "launch-input", u32Bytes([elementCount])),
          namedInput(1, "launch-input", u32Bytes([0])),
        ]
      : []),
  ]);
  const controls = resourceControlled
    ? undefined
    : Object.freeze([{
        controlId: "prefix-elements",
        value: wire(elementCount),
      }]);
  return Object.freeze({
    caseId,
    artifacts,
    graph,
    prepared,
    inputs,
    ...(controls === undefined ? {} : { controls }),
    artifactHash: await hashNamedComponents({
      caseId,
      graph: prepared.graphSemanticHash,
      modules: prepared.wgslModuleHashes,
      inputs: inputs.map(({ rank, resourceId, bytes }) => ({
        rank,
        resourceId,
        bytes: Array.from(bytes),
      })),
      ...(controls === undefined ? {} : { controls }),
    }),
  });
}

async function prepareRectangularDynamicDispatchCase(
  caseId:
    | "f32-rectangular-dynamic-rank2-small"
    | "f32-rectangular-dynamic-rank2-large"
    | "f32-rectangular-dynamic-rank3-small"
    | "f32-rectangular-dynamic-rank3-large",
  shape: readonly [number, number] | readonly [number, number, number],
  logicalExtents:
    readonly [number, number] | readonly [number, number, number],
): Promise<PreparedCase> {
  const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: shape.map((extent) => parseWireI64(String(extent))),
    axes: shape.map((_, axis) => axis),
    dtype: "f32",
  });
  const elementCount = shape.reduce(
    (product, extent) => product * extent,
    1,
  );
  const values = [
    f32Bytes(Array.from(
      { length: elementCount },
      (_, index) => index + 0.25,
    )),
    f32Bytes(Array.from(
      { length: elementCount },
      (_, index) => -index - 0.5,
    )),
  ];
  const graph = (await createVerifiedHostGraphArtifact(
    rectangularDynamicDispatchProgram(artifacts, shape),
    artifactOptions(artifacts),
  )).artifact;
  const prepared = await prepareSemanticHostGraphWebGpu(
    graph,
    { ...artifactOptions(artifacts), workgroupSize: 64 },
  );
  const inputs = Object.freeze(values.map((bytes, rank) =>
    input(rank, bytes)));
  const controls = Object.freeze(logicalExtents.map((extent, axis) => ({
    controlId: `prefix-axis-${axis}`,
    value: wire(extent),
  })));
  return Object.freeze({
    caseId,
    artifacts,
    graph,
    prepared,
    inputs,
    controls,
    artifactHash: await hashNamedComponents({
      caseId,
      graph: prepared.graphSemanticHash,
      modules: prepared.wgslModuleHashes,
      inputs: values.map((bytes) => Array.from(bytes)),
      controls,
    }),
  });
}

async function prepareResourceRectangularDynamicDispatchCase(
  caseId:
    | "f32-resource-rectangular-dynamic-rank2-small"
    | "f32-resource-rectangular-dynamic-rank2-large"
    | "f32-resource-rectangular-dynamic-rank3-small"
    | "f32-resource-rectangular-dynamic-rank3-large",
  shape: readonly [number, number] | readonly [number, number, number],
  logicalExtents:
    readonly [number, number] | readonly [number, number, number],
): Promise<PreparedCase> {
  const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: shape.map((extent) => parseWireI64(String(extent))),
    axes: shape.map((_, axis) => axis),
    dtype: "f32",
  });
  const elementCount = shape.reduce(
    (product, extent) => product * extent,
    1,
  );
  const values = [
    f32Bytes(Array.from(
      { length: elementCount },
      (_, index) => index + 0.25,
    )),
    f32Bytes(Array.from(
      { length: elementCount },
      (_, index) => -index - 0.5,
    )),
  ];
  const graph = (await createVerifiedHostGraphArtifact(
    resourceRectangularDynamicDispatchProgram(artifacts, shape),
    artifactOptions(artifacts),
  )).artifact;
  const prepared = await prepareSemanticHostGraphWebGpu(
    graph,
    { ...artifactOptions(artifacts), workgroupSize: 64 },
  );
  const inputs = Object.freeze([
    ...values.map((bytes, rank) => input(rank, bytes)),
    ...logicalExtents.flatMap((extent, axis) => [
      namedInput(0, `extent-input-${axis}`, u32Bytes([extent])),
      namedInput(1, `extent-input-${axis}`, u32Bytes([0])),
    ]),
  ]);
  return Object.freeze({
    caseId,
    artifacts,
    graph,
    prepared,
    inputs,
    artifactHash: await hashNamedComponents({
      caseId,
      graph: prepared.graphSemanticHash,
      modules: prepared.wgslModuleHashes,
      inputs: inputs.map(({ rank, resourceId, bytes }) => ({
        rank,
        resourceId,
        bytes: Array.from(bytes),
      })),
    }),
  });
}

async function prepareConditionalRawCopyCase(
  caseId: "u8-input-conditional-then" | "u8-input-conditional-else",
  predicate: 0 | 1,
): Promise<PreparedCase> {
  const thenBytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const elseBytes = new Uint8Array([8, 7, 6, 5, 4, 3, 2, 1]);
  const graph = (await createVerifiedHostGraphArtifact(
    conditionalRawCopyProgram(),
    { kernelArtifacts: [], layoutArtifacts: [] },
  )).artifact;
  const prepared = await prepareSemanticHostGraphWebGpu(graph, {
    kernelArtifacts: [],
    layoutArtifacts: [],
  });
  const inputs = Object.freeze([
    namedInput(0, "predicate", u32Bytes([predicate])),
    namedInput(0, "then-input", thenBytes),
    namedInput(0, "else-input", elseBytes),
  ]);
  return Object.freeze({
    caseId,
    graph,
    prepared,
    inputs,
    artifactHash: await hashNamedComponents({
      caseId,
      graph: prepared.graphSemanticHash,
      modules: prepared.wgslModuleHashes,
      inputs: inputs.map(({ rank, resourceId, bytes }) => ({
        rank,
        resourceId,
        bytes: Array.from(bytes),
      })),
    }),
  });
}

async function prepareRuntimeConditionalRawCopyCase(
  caseId:
    | "u8-runtime-conditional-then"
    | "u8-runtime-conditional-else",
  predicate: 0 | 1,
): Promise<PreparedCase> {
  const thenBytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const elseBytes = new Uint8Array([8, 7, 6, 5, 4, 3, 2, 1]);
  const graph = (await createVerifiedHostGraphArtifact(
    runtimeConditionalRawCopyProgram(),
    { kernelArtifacts: [], layoutArtifacts: [] },
  )).artifact;
  const prepared = await prepareSemanticHostGraphWebGpu(graph, {
    kernelArtifacts: [],
    layoutArtifacts: [],
  });
  const inputs = Object.freeze([
    namedInput(0, "then-input", thenBytes),
    namedInput(0, "else-input", elseBytes),
  ]);
  const controls = Object.freeze([{
    controlId: "choose",
    value: wire(predicate),
  }]);
  return Object.freeze({
    caseId,
    graph,
    prepared,
    inputs,
    controls,
    artifactHash: await hashNamedComponents({
      caseId,
      graph: prepared.graphSemanticHash,
      modules: prepared.wgslModuleHashes,
      inputs: inputs.map(({ rank, resourceId, bytes }) => ({
        rank,
        resourceId,
        bytes: Array.from(bytes),
      })),
      controls,
    }),
  });
}

async function prepareResourceConditionalRawCopyCase(
  caseId:
    | "u8-resource-conditional-then"
    | "u8-resource-conditional-else",
  predicate: 0 | 1,
): Promise<PreparedCase> {
  const thenBytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const elseBytes = new Uint8Array([8, 7, 6, 5, 4, 3, 2, 1]);
  const graph = (await createVerifiedHostGraphArtifact(
    resourceConditionalRawCopyProgram(),
    { kernelArtifacts: [], layoutArtifacts: [] },
  )).artifact;
  const prepared = await prepareSemanticHostGraphWebGpu(graph, {
    kernelArtifacts: [],
    layoutArtifacts: [],
  });
  const inputs = Object.freeze([
    namedInput(0, "predicate-source", u32Bytes([predicate])),
    namedInput(0, "then-input", thenBytes),
    namedInput(0, "else-input", elseBytes),
  ]);
  return Object.freeze({
    caseId,
    graph,
    prepared,
    inputs,
    artifactHash: await hashNamedComponents({
      caseId,
      graph: prepared.graphSemanticHash,
      modules: prepared.wgslModuleHashes,
      inputs: inputs.map(({ rank, resourceId, bytes }) => ({
        rank,
        resourceId,
        bytes: Array.from(bytes),
      })),
    }),
  });
}

function collectiveProgram(
  artifacts: VerifiedViewCopyArtifacts,
  dtype: "f32" | "i32" | "u32",
  reduction: "sum" | "min" | "max",
): HostGraphProgram {
  const numericalPolicy = dtype === "f32"
    ? "rank-order-f32" as const
    : reduction === "sum"
      ? "rank-order-wrapping-32" as const
      : "exact-32-bit" as const;
  return {
    kind: "host-graph",
    version: { major: 1, minor: 0 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire(2),
    resources: [
      resource("input", "input", dtype),
      resource("output", "output", dtype),
    ],
    nodes: [
      dispatch(artifacts),
      {
        nodeId: "reduce",
        kind: "all-reduce",
        dependsOn: ["copy"],
        resourceId: "output",
        reduction,
        dtype,
        numericalPolicy,
        participants: [wire(0), wire(1)],
        result: "replicated-to-all-participants",
      },
    ],
  };
}

function dynamicDispatchProgram(
  artifacts: VerifiedViewCopyArtifacts,
  maxElementCount = 2,
): HostGraphProgram {
  const staticDispatch = dispatch(artifacts);
  return {
    kind: "host-graph",
    version: { major: 1, minor: 9 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire(2),
    resources: [
      resource("input", "input", "f32", maxElementCount * 4),
      resource("output", "output", "f32", maxElementCount * 4),
    ],
    nodes: [
      {
        ...staticDispatch,
        kind: "dynamic-dispatch",
        launchControl: {
          controlId: "prefix-elements",
          mode: "u32-prefix-element-count",
        },
        maxElementCount: wire(maxElementCount),
        mode: "runtime-u32-prefix-elements",
      },
      {
        nodeId: "materialize-output",
        kind: "materialize",
        dependsOn: ["copy"],
        resourceId: "output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function rectangularDynamicDispatchProgram(
  artifacts: VerifiedViewCopyArtifacts,
  shape: readonly [number, number] | readonly [number, number, number],
): HostGraphProgram {
  const staticDispatch = dispatch(artifacts);
  const elementCount = shape.reduce(
    (product, extent) => product * extent,
    1,
  );
  return {
    kind: "host-graph",
    version: { major: 1, minor: 12 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire(2),
    resources: [
      resource("input", "input", "f32", elementCount * 4),
      resource("output", "output", "f32", elementCount * 4),
    ],
    nodes: [
      {
        ...staticDispatch,
        kind: "dynamic-dispatch",
        launchControls: shape.map((_, axis) => ({
          axis,
          controlId: `prefix-axis-${axis}`,
          mode: "u32-prefix-extent" as const,
        })),
        maxExtents: shape.map((extent) => wire(extent)),
        mode: "runtime-u32-rectangular-prefix",
      },
      {
        nodeId: "materialize-output",
        kind: "materialize",
        dependsOn: ["copy"],
        resourceId: "output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function resourceRectangularDynamicDispatchProgram(
  artifacts: VerifiedViewCopyArtifacts,
  shape: readonly [number, number] | readonly [number, number, number],
): HostGraphProgram {
  const base = rectangularDynamicDispatchProgram(artifacts, shape);
  const producerIds = shape.map((_, axis) => `produce-extent-${axis}`);
  return {
    ...base,
    version: { major: 1, minor: 13 },
    resources: [
      ...base.resources,
      ...shape.flatMap((_, axis) => [
        resource(`extent-input-${axis}`, "input", "u32", 4),
        resource(`extent-${axis}`, "temporary", "u32", 4),
      ]),
    ],
    nodes: [
      ...shape.map((_, axis) => ({
        nodeId: producerIds[axis] as string,
        kind: "copy" as const,
        dependsOn: [],
        sourceResourceId: `extent-input-${axis}`,
        destinationResourceId: `extent-${axis}`,
        mode: "whole-allocation-bytes-per-rank" as const,
      })),
      ...base.nodes.map((node) => {
        if (
          node.kind !== "dynamic-dispatch" ||
          node.mode !== "runtime-u32-rectangular-prefix"
        ) {
          return node;
        }
        const { launchControls: _launchControls, ...common } = node;
        return {
          ...common,
          dependsOn: producerIds,
          launchSources: shape.map((_, axis) => ({
            axis,
            resourceId: `extent-${axis}`,
            rank: wire(0),
            mode: "u32-prefix-extent" as const,
          })),
          mode: "resource-u32-rectangular-prefix" as const,
        };
      }),
    ],
  };
}

function resourceDynamicDispatchProgram(
  artifacts: VerifiedViewCopyArtifacts,
  maxElementCount = 2,
): HostGraphProgram {
  const base = dynamicDispatchProgram(artifacts, maxElementCount);
  return {
    ...base,
    version: { major: 1, minor: 11 },
    resources: [
      ...base.resources,
      {
        resourceId: "launch-input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "u32",
        byteLength: wire(4),
        alignmentBytes: 4,
      },
      {
        resourceId: "launch-count",
        role: "temporary",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "u32",
        byteLength: wire(4),
        alignmentBytes: 4,
      },
    ],
    nodes: [
      {
        nodeId: "produce-launch-count",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "launch-input",
        destinationResourceId: "launch-count",
        mode: "whole-allocation-bytes-per-rank",
      },
      ...base.nodes.map((node) => {
        if (
          node.kind !== "dynamic-dispatch" ||
          node.mode !== "runtime-u32-prefix-elements"
        ) {
          return node;
        }
        const { launchControl: _launchControl, ...common } = node;
        return {
          ...common,
          dependsOn: ["produce-launch-count"],
          launchSource: {
            resourceId: "launch-count",
            rank: wire(0),
            mode: "u32-prefix-element-count" as const,
          },
          mode: "resource-u32-prefix-elements" as const,
        };
      }),
    ],
  };
}

function rawCopyProgram(): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 3 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire(2),
    resources: [
      {
        resourceId: "input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "u8",
        byteLength: wire(8),
        alignmentBytes: 1,
      },
      {
        resourceId: "output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "u8",
        byteLength: wire(8),
        alignmentBytes: 1,
      },
    ],
    nodes: [
      {
        nodeId: "raw-copy",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "input",
        destinationResourceId: "output",
        mode: "whole-allocation-bytes-per-rank",
      },
      {
        nodeId: "copy-complete-event",
        kind: "event",
        dependsOn: ["raw-copy"],
        eventId: "copy-complete",
        mode: "completion-after-dependencies",
      },
      {
        nodeId: "materialize-output",
        kind: "materialize",
        dependsOn: ["copy-complete-event"],
        resourceId: "output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function repeatedCollectiveProgram(): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 4 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire(2),
    resources: [
      resource("input", "input", "f32"),
      resource("output", "output", "f32"),
    ],
    nodes: [
      {
        nodeId: "initialize",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "input",
        destinationResourceId: "output",
        mode: "whole-allocation-bytes-per-rank",
      },
      {
        nodeId: "repeat-reduction",
        kind: "repeat",
        dependsOn: ["initialize"],
        iterationCount: wire(3),
        body: [{
          nodeId: "reduce-body",
          kind: "all-reduce",
          dependsOn: [],
          resourceId: "output",
          reduction: "sum",
          dtype: "f32",
          numericalPolicy: "rank-order-f32",
          participants: [wire(0), wire(1)],
          result: "replicated-to-all-participants",
        }],
        mode: "fixed-count-sequential",
      },
      {
        nodeId: "materialize-output",
        kind: "materialize",
        dependsOn: ["repeat-reduction"],
        resourceId: "output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function runtimeRepeatedCollectiveProgram(): HostGraphProgram {
  const base = repeatedCollectiveProgram();
  return {
    ...base,
    version: { major: 1, minor: 8 },
    nodes: base.nodes.map((node) => {
      if (
        node.kind !== "repeat" ||
        node.mode !== "fixed-count-sequential"
      ) {
        return node;
      }
      const { iterationCount: _iterationCount, ...common } = node;
      return {
        ...common,
        iterationControl: {
          controlId: "iterations",
          mode: "u32-count" as const,
        },
        maxIterationCount: wire(3),
        mode: "runtime-u32-count-sequential" as const,
      };
    }),
  };
}

function resourceRepeatedCollectiveProgram(): HostGraphProgram {
  const base = repeatedCollectiveProgram();
  return {
    ...base,
    version: { major: 1, minor: 10 },
    resources: [
      ...base.resources,
      {
        resourceId: "iteration-input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "u32",
        byteLength: wire(4),
        alignmentBytes: 4,
      },
      {
        resourceId: "iteration-count",
        role: "temporary",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "u32",
        byteLength: wire(4),
        alignmentBytes: 4,
      },
    ],
    nodes: [
      {
        nodeId: "produce-iteration-count",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "iteration-input",
        destinationResourceId: "iteration-count",
        mode: "whole-allocation-bytes-per-rank",
      },
      ...base.nodes.map((node) => {
        if (
          node.kind !== "repeat" ||
          node.mode !== "fixed-count-sequential"
        ) {
          return node;
        }
        const { iterationCount: _iterationCount, ...common } = node;
        return {
          ...common,
          dependsOn: ["initialize", "produce-iteration-count"],
          iterationSource: {
            resourceId: "iteration-count",
            rank: wire(0),
            mode: "u32-count" as const,
          },
          maxIterationCount: wire(3),
          mode: "resource-u32-count-sequential" as const,
        };
      }),
    ],
  };
}

function conditionalRawCopyProgram(): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 5 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire(1),
    resources: [
      {
        resourceId: "predicate",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "u32",
        byteLength: wire(4),
        alignmentBytes: 4,
      },
      ...["then-input", "else-input"].map((resourceId) => ({
        resourceId,
        role: "input" as const,
        multiplicity: "per-rank" as const,
        initialization: "external-input" as const,
        dtype: "u8" as const,
        byteLength: wire(8),
        alignmentBytes: 1,
      })),
      {
        resourceId: "output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "u8",
        byteLength: wire(8),
        alignmentBytes: 1,
      },
    ],
    nodes: [
      {
        nodeId: "choose-output",
        kind: "conditional",
        dependsOn: [],
        predicate: {
          resourceId: "predicate",
          rank: wire(0),
          mode: "u32-nonzero",
        },
        thenBody: [{
          nodeId: "copy-then",
          kind: "copy",
          dependsOn: [],
          sourceResourceId: "then-input",
          destinationResourceId: "output",
          mode: "whole-allocation-bytes-per-rank",
        }],
        elseBody: [{
          nodeId: "copy-else",
          kind: "copy",
          dependsOn: [],
          sourceResourceId: "else-input",
          destinationResourceId: "output",
          mode: "whole-allocation-bytes-per-rank",
        }],
        mode: "input-u32-branch-sequential",
      },
      {
        nodeId: "materialize-output",
        kind: "materialize",
        dependsOn: ["choose-output"],
        resourceId: "output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function runtimeConditionalRawCopyProgram(): HostGraphProgram {
  const base = conditionalRawCopyProgram();
  return {
    ...base,
    version: { major: 1, minor: 6 },
    resources: base.resources.filter((resource) =>
      resource.resourceId !== "predicate"),
    nodes: base.nodes.map((node) =>
      node.kind === "conditional"
        ? {
            ...node,
            predicate: {
              controlId: "choose",
              mode: "u32-nonzero" as const,
            },
            mode: "runtime-u32-branch-sequential" as const,
          }
        : node),
  };
}

function resourceConditionalRawCopyProgram(): HostGraphProgram {
  const base = conditionalRawCopyProgram();
  return {
    ...base,
    version: { major: 1, minor: 7 },
    resources: [
      ...base.resources.map((item) =>
        item.resourceId === "predicate"
          ? {
              ...item,
              role: "temporary" as const,
              initialization: "zero-fill" as const,
            }
          : item),
      {
        resourceId: "predicate-source",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "u32",
        byteLength: wire(4),
        alignmentBytes: 4,
      },
    ],
    nodes: [
      {
        nodeId: "produce-predicate",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "predicate-source",
        destinationResourceId: "predicate",
        mode: "whole-allocation-bytes-per-rank",
      },
      ...base.nodes.map((node) =>
        node.kind === "conditional" &&
          node.mode === "input-u32-branch-sequential"
          ? {
              ...node,
              dependsOn: ["produce-predicate"],
              mode: "resource-u32-branch-sequential" as const,
            }
          : node),
    ],
  };
}

function resource(
  resourceId: string,
  role: "input" | "temporary" | "output",
  dtype: "f32" | "i32" | "u32",
  byteLength = 8,
) {
  return {
    resourceId,
    role,
    multiplicity: "per-rank" as const,
    initialization: role === "input"
      ? "external-input" as const
      : "zero-fill" as const,
    dtype,
    byteLength: wire(byteLength),
    alignmentBytes: 4,
  };
}

function dispatch(artifacts: VerifiedViewCopyArtifacts) {
  return {
    nodeId: "copy",
    kind: "dispatch" as const,
    dependsOn: [],
    semanticArtifactHash: artifacts.kernelSemanticHash,
    entrypointId: artifacts.operationId,
    dimensionBindings: {},
    bindings: [
      {
        semanticResourceId: artifacts.source.viewId,
        graphResourceId: "input",
      },
      {
        semanticResourceId: artifacts.destination.viewId,
        graphResourceId: "output",
      },
    ],
  };
}

function artifactOptions(artifacts: VerifiedViewCopyArtifacts) {
  return {
    kernelArtifacts: [artifacts.kernel],
    layoutArtifacts: [artifacts.layout],
  };
}

function input(
  rank: number,
  bytes: Uint8Array,
): SemanticHostGraphWebGpuInputBinding {
  return namedInput(rank, "input", bytes);
}

function namedInput(
  rank: number,
  resourceId: string,
  bytes: Uint8Array,
): SemanticHostGraphWebGpuInputBinding {
  return { rank: wire(rank), resourceId, bytes };
}

function f32Bytes(values: readonly number[]): Uint8Array {
  const output = new Uint8Array(values.length * 4);
  const view = new DataView(output.buffer);
  values.forEach((value, index) =>
    view.setFloat32(index * 4, value, true));
  return output;
}

function i32Bytes(values: readonly number[]): Uint8Array {
  const output = new Uint8Array(values.length * 4);
  const view = new DataView(output.buffer);
  values.forEach((value, index) =>
    view.setInt32(index * 4, value, true));
  return output;
}

function u32Bytes(values: readonly number[]): Uint8Array {
  const output = new Uint8Array(values.length * 4);
  const view = new DataView(output.buffer);
  values.forEach((value, index) =>
    view.setUint32(index * 4, value, true));
  return output;
}

function assertOutputEquality(
  actual: readonly { readonly rank: WireU64; readonly resourceId: string; readonly bytes: Uint8Array }[],
  expected: readonly { readonly rank: WireU64; readonly resourceId: string; readonly bytes: Uint8Array }[],
): void {
  expect(actual.map(({ rank, resourceId }) => ({ rank, resourceId })))
    .toEqual(expected.map(({ rank, resourceId }) => ({ rank, resourceId })));
  expect(actual.map(({ bytes }) => Array.from(bytes)))
    .toEqual(expected.map(({ bytes }) => Array.from(bytes)));
}

function emitTerminal(input: {
  readonly required: boolean;
  readonly artifactHash: string;
  readonly environment: JsonObject;
  readonly environmentId: string;
  readonly deviceProfileHash?: string;
  readonly outcome: "not-run" | "passed" | "failed";
  readonly diagnosticCodes: readonly string[];
  readonly completedCases: readonly CaseObservation[];
  readonly deviceLossRefusalObserved?: boolean;
  readonly stage: string;
  readonly uncapturedErrors: readonly string[];
  readonly error?: JsonObject;
}): void {
  TERMINAL_EMITTER.emit({
    schema: EXECUTION_EVIDENCE_SCHEMA,
    kind: "terminal",
    suiteId: SUITE_ID,
    required: input.required,
    evidence: {
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
      diagnosticCodes: input.diagnosticCodes,
    },
    environment: input.environment,
    plannedCaseIds: CASE_IDS,
    completedCases: input.completedCases,
    ...(input.deviceLossRefusalObserved === undefined
      ? {}
      : {
          deviceLossRefusalObserved:
            input.deviceLossRefusalObserved,
        }),
    stage: input.stage,
    uncapturedErrors: input.uncapturedErrors,
    ...(input.error === undefined ? {} : { error: input.error }),
  });
}

function diagnosticCode(error: unknown): string {
  if (error instanceof SemanticHostGraphWebGpuError) return error.code;
  if (error instanceof EvidenceLaneError) return error.code;
  return "BG-WEBGPU-GRAPH-EVIDENCE-UNEXPECTED";
}

function errorRecord(error: unknown): JsonObject {
  return {
    name: error instanceof Error ? error.name : "UnknownError",
    message: error instanceof Error ? error.message : String(error),
    diagnosticCode: diagnosticCode(error),
  };
}

function withGraphTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  label: string,
): Promise<T> {
  return withWebGpuEvidenceTimeout(
    promise,
    timeoutMs,
    label,
    (message) => new EvidenceLaneError(
      "BG-WEBGPU-GRAPH-EVIDENCE-TIMEOUT",
      message,
    ),
  );
}

class EvidenceLaneError extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "EvidenceLaneError";
  }
}
