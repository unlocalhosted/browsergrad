import { expect, it } from "vitest";
import {
  createTerminalEvidenceEmitter,
  createWebGpuExecutionEnvironmentRecord,
  requiredEvidenceFailure,
  requiresWebGpuEvidence,
} from "../../../test-support/webgpu-evidence";

import {
  createVerifiedHostGraphArtifact,
  type HostGraphProgram,
} from "@unlocalhosted/browsergrad-semantic-core/graph";
import {
  createVerifiedDensePermutationViewCopyArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  BUILTIN_DTYPES,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  parseWireI64,
  parseWireU64,
  hashNamedComponents,
  sha256Hex,
  type JsonObject,
  type JsonValue,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
} from "../src/semantic_host_graph";
import {
  SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
  SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION,
  SemanticHostGraphBrowserWorkerReportedError,
  executeSemanticHostGraphBrowserWorker,
  type SemanticHostGraphBrowserWorkerExecutionResult,
} from "../src/semantic_host_graph_worker";
import {
  createVerifiedSignedReverseViewCopyArtifacts,
  patternedStorageBytes,
  reverseStorageElements,
  singleViewCopyGraphProgram,
} from "../tests/semantic_host_graph_fixtures";
import {
  SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_BACKEND_ID,
  SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CAPABILITY_ID,
  SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS,
  SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_COMPARISON_POLICY_ID,
  SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_PREFIX,
  SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_SUITE_ID,
  createSemanticHostGraphWorkerTerminalRecord,
  type SemanticHostGraphWorkerCaseObservation,
} from "../tests/semantic_host_graph_worker_evidence";

declare const __BG_KERNELS_VERSION__: string;
declare const __BG_SEMANTIC_CORE_VERSION__: string;

const wire = (value: number): WireU64 => parseWireU64(String(value));
const PRODUCER_VERSIONS = Object.freeze({
  "@unlocalhosted/browsergrad-kernels": __BG_KERNELS_VERSION__,
  "@unlocalhosted/browsergrad-semantic-core":
    __BG_SEMANTIC_CORE_VERSION__,
  "browsergrad.backend.webgpu.semantic-host-graph":
    SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
  "browsergrad.transport.semantic-host-graph-worker":
    String(SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION),
});
const TERMINAL_EMITTER = createTerminalEvidenceEmitter(
  SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_PREFIX,
  {
    suiteId: SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_SUITE_ID,
    capabilityId: SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CAPABILITY_ID,
    backendId: SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_BACKEND_ID,
    comparisonPolicyId:
      SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_COMPARISON_POLICY_ID,
    requireDeviceProfile: true,
  },
);

it("re-verifies and executes host graphs in one-shot WebGPU Workers", async (context) => {
  const required = requiresWebGpuEvidence();
  let stage = "suite-manifest";
  let currentCaseId: string | undefined;
  let artifactHash = await hashNamedComponents({
    suiteId: SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_SUITE_ID,
    plannedCaseIds: SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS,
  });
  let environment = createWebGpuExecutionEnvironmentRecord({
    acquisition: "not-attempted",
  }) as JsonObject;
  let environmentId = await hashNamedComponents({ environment });
  let deviceProfileHash: string | undefined;
  let terminalEmitted = false;
  const completedCases: SemanticHostGraphWorkerCaseObservation[] = [];

  try {
    if (navigator.gpu === undefined) {
      const message = "navigator.gpu is unavailable";
      environment = createWebGpuExecutionEnvironmentRecord({
        acquisition:
          "Worker-owned navigator.gpu.requestAdapter/requestDevice",
        unavailableReason: message,
      }) as JsonObject;
      environmentId = await hashNamedComponents({ environment });
      stage = "device-capability";
      emitTerminal({
        required,
        artifactHash,
        environment,
        environmentId,
        outcome: required ? "failed" : "not-run",
        diagnosticCodes: [
          "BG-WEBGPU-GRAPH-WORKER-EVIDENCE-DEVICE-UNAVAILABLE",
        ],
        completedCases,
        stage,
        error: { name: "WebGpuEvidenceUnavailable", message },
      });
      terminalEmitted = true;
      if (required) throw requiredEvidenceFailure(message);
      context.skip(message);
      return;
    }

    stage = "case-execution";
    currentCaseId = SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS[0]!;
    const rawInput = new Uint8Array([0, 1, 2, 3, 252, 253, 254, 255]);
    const rawGraph = (await createVerifiedHostGraphArtifact(
      rawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const rawPending = executeSemanticHostGraphBrowserWorker({
      graphArtifact: rawGraph,
      kernelArtifacts: [],
      layoutArtifacts: [],
      request: {
        inputs: [{ rank: wire(0), resourceId: "input", bytes: rawInput }],
      },
    });
    rawInput.fill(77);
    const rawResult = await rawPending;
    expect(rawResult.outputs).toHaveLength(1);
    expect(rawResult.outputs[0]?.bytes).toEqual(
      new Uint8Array([0, 1, 2, 3, 252, 253, 254, 255]),
    );
    expect(rawResult.backendTrace).toMatchObject({
      submitted: true,
      copyStepCount: 1,
      materializationCount: 1,
    });
    deviceProfileHash = appendWorkerObservation(
      completedCases,
      await observeWorkerCase(currentCaseId, rawResult),
    );

    currentCaseId = SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS[1]!;
    const artifacts =
      await createVerifiedDensePermutationViewCopyArtifacts({
        inputShape: [parseWireI64("4")],
        axes: [0],
        dtype: "f32",
      });
    const semanticGraph = (await createVerifiedHostGraphArtifact(
      singleViewCopyGraphProgram(artifacts, "f32", 16, 1),
      {
        kernelArtifacts: [artifacts.kernel],
        layoutArtifacts: [artifacts.layout],
      },
    )).artifact;
    const semanticInput = f32Bytes([1.5, -2, 0, 99.25]);
    const semanticPending = executeSemanticHostGraphBrowserWorker({
      graphArtifact: semanticGraph,
      kernelArtifacts: [artifacts.kernel],
      layoutArtifacts: [artifacts.layout],
      request: {
        inputs: [{
          rank: wire(0),
          resourceId: "input",
          bytes: semanticInput,
        }],
      },
    });
    semanticInput.fill(0);
    const semanticResult = await semanticPending;
    expect(semanticResult.outputs[0]?.bytes).toEqual(
      f32Bytes([1.5, -2, 0, 99.25]),
    );
    expect(semanticResult.backendTrace).toMatchObject({
      submitted: true,
      dispatchStepCount: 1,
      materializationCount: 1,
    });
    expect(semanticResult.transportTrace).toMatchObject({
      profile: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
      topology: "single-dedicated-browser-worker",
      acceptedTerminalMessages: 1,
      workerExecutionObserved: true,
      workerLifecycle: "one-shot-terminated",
      inputByteLength: 16,
      outputByteLength: 16,
    });
    expect(semanticResult.transportTrace.artifactByteLength)
      .toBeGreaterThan(rawResult.transportTrace.artifactByteLength);
    deviceProfileHash = appendWorkerObservation(
      completedCases,
      await observeWorkerCase(currentCaseId, semanticResult),
    );

    currentCaseId = SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS[2]!;
    const i8SignedResult = await executeSignedStorageWorkerCase(
      "i8",
      0x21,
    );
    deviceProfileHash = appendWorkerObservation(
      completedCases,
      await observeWorkerCase(currentCaseId, i8SignedResult),
    );

    currentCaseId = SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS[3]!;
    const f64SignedResult = await executeSignedStorageWorkerCase(
      "f64",
      0x43,
    );
    deviceProfileHash = appendWorkerObservation(
      completedCases,
      await observeWorkerCase(currentCaseId, f64SignedResult),
    );

    stage = "environment-identity";
    currentCaseId = undefined;
    if (deviceProfileHash === undefined) {
      throw new EvidenceLaneError(
        "BG-WEBGPU-GRAPH-WORKER-EVIDENCE-DEVICE-PROFILE",
        "completed Worker suite has no device profile",
      );
    }
    const workerDevice = rawResult.backendTrace.device;
    environment = createWebGpuExecutionEnvironmentRecord({
      acquisition:
        "four independent Worker-owned navigator.gpu.requestAdapter/requestDevice calls",
      negotiatedDeviceFeatures: workerDevice.features,
      negotiatedDeviceLimits: workerDevice.limits as JsonObject,
    }) as JsonObject;
    environmentId = await hashNamedComponents({ environment });

    stage = "correctness-artifact";
    artifactHash = await hashNamedComponents({
      profile: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
      caseIds: SEMANTIC_HOST_GRAPH_WORKER_EVIDENCE_CASE_IDS,
      graphSemanticHashes: [
        rawResult.backendTrace.graphSemanticHash,
        semanticResult.backendTrace.graphSemanticHash,
        i8SignedResult.backendTrace.graphSemanticHash,
        f64SignedResult.backendTrace.graphSemanticHash,
      ],
      backendSpecializationHashes: [
        rawResult.backendTrace.backendSpecializationHash,
        semanticResult.backendTrace.backendSpecializationHash,
        i8SignedResult.backendTrace.backendSpecializationHash,
        f64SignedResult.backendTrace.backendSpecializationHash,
      ],
      outputs: [
        Array.from(rawResult.outputs[0]!.bytes),
        Array.from(semanticResult.outputs[0]!.bytes),
        Array.from(i8SignedResult.outputs[0]!.bytes),
        Array.from(f64SignedResult.outputs[0]!.bytes),
      ],
      deviceProfileHash,
    });
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
      stage,
    });
    terminalEmitted = true;
  } catch (cause) {
    if (
      !required &&
      cause instanceof SemanticHostGraphBrowserWorkerReportedError &&
      cause.failure.phase === "device-acquisition"
    ) {
      environment = createWebGpuExecutionEnvironmentRecord({
        acquisition:
          "Worker-owned navigator.gpu.requestAdapter/requestDevice",
        unavailableReason: cause.message,
      }) as JsonObject;
      environmentId = await hashNamedComponents({ environment });
      emitTerminal({
        required,
        artifactHash,
        environment,
        environmentId,
        ...(deviceProfileHash === undefined
          ? {}
          : { deviceProfileHash }),
        outcome: "not-run",
        diagnosticCodes: [
          "BG-WEBGPU-GRAPH-WORKER-EVIDENCE-DEVICE-UNAVAILABLE",
        ],
        completedCases,
        stage: "device-acquisition",
        ...(currentCaseId === undefined ? {} : { currentCaseId }),
        error: errorRecord(cause),
      });
      terminalEmitted = true;
      context.skip(cause.message);
      return;
    }
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
        diagnosticCodes: [diagnosticCode(cause, stage)],
        completedCases,
        stage,
        ...(currentCaseId === undefined ? {} : { currentCaseId }),
        error: errorRecord(cause),
      });
    }
    throw cause;
  }
});

function appendWorkerObservation(
  completedCases: SemanticHostGraphWorkerCaseObservation[],
  observation: SemanticHostGraphWorkerCaseObservation,
): string {
  const existingDeviceProfileHash =
    completedCases[0]?.deviceProfileHash;
  if (
    existingDeviceProfileHash !== undefined &&
    existingDeviceProfileHash !== observation.deviceProfileHash
  ) {
    throw new EvidenceLaneError(
      "BG-WEBGPU-GRAPH-WORKER-EVIDENCE-DEVICE-PROFILE",
      "independent Workers reported different device profiles",
    );
  }
  completedCases.push(observation);
  return observation.deviceProfileHash;
}

async function observeWorkerCase(
  caseId: string,
  result: SemanticHostGraphBrowserWorkerExecutionResult,
): Promise<SemanticHostGraphWorkerCaseObservation> {
  if (result.outputs.length !== 1 || result.outputs[0] === undefined) {
    throw new EvidenceLaneError(
      "BG-WEBGPU-GRAPH-WORKER-EVIDENCE-OUTPUT",
      `${caseId} did not materialize exactly one complete output`,
    );
  }
  return Object.freeze({
    caseId,
    graphSemanticHash: result.backendTrace.graphSemanticHash,
    backendSpecializationHash:
      result.backendTrace.backendSpecializationHash,
    outputHash: await sha256Hex(result.outputs[0].bytes),
    deviceProfileHash: await hashNamedComponents({
      device: result.backendTrace.device as unknown as JsonValue,
    }),
    artifactByteLength: result.transportTrace.artifactByteLength,
    inputByteLength: result.transportTrace.inputByteLength,
    outputByteLength: result.transportTrace.outputByteLength,
    acceptedTerminalMessages:
      result.transportTrace.acceptedTerminalMessages,
    workerExecutionObserved:
      result.transportTrace.workerExecutionObserved,
    workerLifecycle: result.transportTrace.workerLifecycle,
    comparison: "bit-exact-complete-output",
    inputSnapshot:
      "caller-input-mutated-after-admission-bit-exact",
  });
}

function emitTerminal(input: Readonly<{
  required: boolean;
  artifactHash: string;
  environment: JsonObject;
  environmentId: string;
  deviceProfileHash?: string;
  outcome: "not-run" | "passed" | "failed";
  diagnosticCodes: readonly string[];
  completedCases:
    readonly SemanticHostGraphWorkerCaseObservation[];
  stage: string;
  currentCaseId?: string;
  error?: JsonObject;
}>): void {
  TERMINAL_EMITTER.emit(
    createSemanticHostGraphWorkerTerminalRecord({
      ...input,
      producerVersions: PRODUCER_VERSIONS,
    }),
  );
}

function diagnosticCode(error: unknown, stage: string): string {
  if (
    error instanceof SemanticHostGraphBrowserWorkerReportedError
  ) {
    return error.failure.failureCode;
  }
  if (error instanceof EvidenceLaneError) return error.code;
  if (stage === "suite-manifest") {
    return "BG-WEBGPU-GRAPH-WORKER-EVIDENCE-MANIFEST";
  }
  if (stage === "case-execution") {
    return "BG-WEBGPU-GRAPH-WORKER-EVIDENCE-EXECUTION";
  }
  return "BG-WEBGPU-GRAPH-WORKER-EVIDENCE-INTERNAL";
}

function errorRecord(error: unknown): JsonObject {
  if (error instanceof SemanticHostGraphBrowserWorkerReportedError) {
    return {
      name: error.name,
      message: error.message,
      code: error.code,
      path: error.path,
      workerPhase: error.failure.phase,
      workerFailureCode: error.failure.failureCode,
      workerFailurePath: error.failure.failurePath,
    };
  }
  if (error instanceof EvidenceLaneError) {
    return {
      name: error.name,
      message: error.message,
      code: error.code,
    };
  }
  if (error instanceof Error) {
    return { name: error.name, message: error.message };
  }
  return { name: "UnknownError", message: String(error) };
}

class EvidenceLaneError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "EvidenceLaneError";
  }
}

async function executeSignedStorageWorkerCase(
  dtype: "i8" | "f64",
  seed: number,
): Promise<SemanticHostGraphBrowserWorkerExecutionResult> {
  const rank = 8;
  const byteLength = (2 ** rank) *
    (BUILTIN_DTYPES[dtype].storageBits / 8);
  const artifacts = await createVerifiedSignedReverseViewCopyArtifacts(
    dtype,
    rank,
  );
  const graph = (await createVerifiedHostGraphArtifact(
    singleViewCopyGraphProgram(artifacts, dtype, byteLength, 1),
    {
      kernelArtifacts: [artifacts.kernel],
      layoutArtifacts: [artifacts.layout],
    },
  )).artifact;
  const bytes = patternedStorageBytes(byteLength, seed);
  const expected = reverseStorageElements(
    bytes,
    BUILTIN_DTYPES[dtype].storageBits / 8,
  );
  const pending = executeSemanticHostGraphBrowserWorker({
    graphArtifact: graph,
    kernelArtifacts: [artifacts.kernel],
    layoutArtifacts: [artifacts.layout],
    request: {
      inputs: [{ rank: wire(0), resourceId: "input", bytes }],
    },
  });
  bytes.fill(seed ^ 0xff);
  const result = await pending;
  expect(result.outputs).toHaveLength(1);
  expect(result.outputs[0]?.bytes).toEqual(expected);
  expect(result.backendTrace).toMatchObject({
    submitted: true,
    dispatchStepCount: 1,
    materializationCount: 1,
  });
  expect(result.transportTrace).toMatchObject({
    profile: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
    topology: "single-dedicated-browser-worker",
    acceptedTerminalMessages: 1,
    workerExecutionObserved: true,
    workerLifecycle: "one-shot-terminated",
    inputByteLength: byteLength,
    outputByteLength: byteLength,
  });
  return result;
}

function rawCopyProgram(): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 3 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire(1),
    resources: [
      resource("input", "input", "u8", 8, 1),
      resource("output", "output", "u8", 8, 1),
    ],
    nodes: [
      {
        nodeId: "copy",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "input",
        destinationResourceId: "output",
        mode: "whole-allocation-bytes-per-rank",
      },
      {
        nodeId: "materialize",
        kind: "materialize",
        dependsOn: ["copy"],
        resourceId: "output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function resource(
  resourceId: string,
  role: "input" | "output",
  dtype: "u8",
  byteLength: number,
  alignmentBytes: number,
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
    alignmentBytes,
  };
}

function f32Bytes(values: readonly number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => {
    view.setFloat32(index * 4, value, true);
  });
  return bytes;
}
