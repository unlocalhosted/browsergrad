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
  prepareSemanticHostGraphWebGpu,
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
  "u8-input-conditional-then",
  "u8-input-conditional-else",
  "u8-runtime-conditional-then",
  "u8-runtime-conditional-else",
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
  readonly backendSpecializationHash: string;
  readonly expandedStepCount: number;
  readonly dispatchStepCount: number;
  readonly copyStepCount: number;
  readonly materializationCount: number;
  readonly completedEventIds: readonly string[];
  readonly completedRepeats: readonly JsonObject[];
  readonly completedConditionals: readonly JsonObject[];
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

    for (const preparedCase of cases) {
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
      const actualPromise = runSemanticHostGraphWebGpu(
        kernelDevice,
        preparedCase.prepared,
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
      const [expected, actual] = await Promise.all([
        expectedPromise,
        actualPromise,
      ]);
      assertOutputEquality(actual.outputs, expected.outputs);
      expect(actual.trace.submitted).toBe(true);
      expect(actual.trace.executedNodeIds)
        .toEqual(expected.executedNodeIds);
      expect(actual.trace.completedEventIds)
        .toEqual(expected.completedEventIds);
      expect(actual.trace.completedRepeats)
        .toEqual(expected.completedRepeats);
      expect(actual.trace.completedConditionals)
        .toEqual(expected.completedConditionals);
      completedCases.push(Object.freeze({
        caseId: preparedCase.caseId,
        artifactHash: preparedCase.artifactHash,
        graphSemanticHash: actual.trace.graphSemanticHash,
        backendSpecializationHash:
          actual.trace.backendSpecializationHash,
        expandedStepCount: actual.trace.expandedStepCount,
        dispatchStepCount: actual.trace.dispatchStepCount,
        copyStepCount: actual.trace.copyStepCount,
        materializationCount: actual.trace.materializationCount,
        completedEventIds: actual.trace.completedEventIds,
        completedRepeats: actual.trace.completedRepeats,
        completedConditionals: actual.trace.completedConditionals,
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
    expect(thenConditional?.backendSpecializationHash)
      .not.toBe(elseConditional?.backendSpecializationHash);
    const thenRuntimeConditional = completedCases.find(({ caseId }) =>
      caseId === "u8-runtime-conditional-then");
    const elseRuntimeConditional = completedCases.find(({ caseId }) =>
      caseId === "u8-runtime-conditional-else");
    expect(thenRuntimeConditional?.backendSpecializationHash)
      .not.toBe(elseRuntimeConditional?.backendSpecializationHash);

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
    await runSemanticHostGraphWebGpu(
      disposableKernelDevice,
      finiteCase.prepared,
      { inputs: finiteCase.inputs },
    );
    disposableDevice.destroy();
    await disposableDevice.lost;
    await nextWebGpuEvidenceTask();
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

function resource(
  resourceId: string,
  role: "input" | "output",
  dtype: "f32" | "i32" | "u32",
) {
  return {
    resourceId,
    role,
    multiplicity: "per-rank" as const,
    initialization: role === "input"
      ? "external-input" as const
      : "zero-fill" as const,
    dtype,
    byteLength: wire(8),
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
