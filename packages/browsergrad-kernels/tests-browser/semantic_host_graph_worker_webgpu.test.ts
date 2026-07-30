import { expect, it } from "vitest";
import {
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
  type JsonValue,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
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

const wire = (value: number): WireU64 => parseWireU64(String(value));

it("re-verifies and executes host graphs in one-shot WebGPU Workers", async (context) => {
  const required = requiresWebGpuEvidence();
  if (navigator.gpu === undefined) {
    const message = "navigator.gpu is unavailable";
    if (required) throw requiredEvidenceFailure(message);
    context.skip(message);
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) {
    const message = "WebGPU adapter is unavailable";
    if (required) throw requiredEvidenceFailure(message);
    context.skip(message);
    return;
  }

  try {
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
    const i8SignedResult = await executeSignedStorageWorkerCase(
      "i8",
      0x21,
    );
    const f64SignedResult = await executeSignedStorageWorkerCase(
      "f64",
      0x43,
    );
    const results = [
      rawResult,
      semanticResult,
      i8SignedResult,
      f64SignedResult,
    ] as const;
    const deviceProfileHashes = await Promise.all(results.map((result) =>
      hashNamedComponents({
        device: result.backendTrace.device as unknown as JsonValue,
      })));
    expect(new Set(deviceProfileHashes).size).toBe(1);
    const deviceProfileHash = deviceProfileHashes[0]!;
    const correctnessArtifact = await hashNamedComponents({
      profile: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
      caseIds: [
        "raw-u8-whole-allocation",
        "f32-identity-view-copy",
        "i8-signed-rank8-view-copy",
        "f64-signed-rank8-view-copy",
      ],
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
    console.warn(
      "[browsergrad-semantic-host-graph-worker-evidence]",
      JSON.stringify({
        suiteId: "browsergrad.kernels.semantic-host-graph.worker-conformance@2",
        profile: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
        topology: "single-dedicated-browser-worker",
        caseCount: 4,
        correctnessArtifact,
        deviceProfileHash,
        outcome: "passed",
      }),
    );
  } catch (cause) {
    if (
      !required &&
      cause instanceof SemanticHostGraphBrowserWorkerReportedError &&
      cause.failure.phase === "device-acquisition"
    ) {
      context.skip(cause.message);
      return;
    }
    throw cause;
  }
});

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
