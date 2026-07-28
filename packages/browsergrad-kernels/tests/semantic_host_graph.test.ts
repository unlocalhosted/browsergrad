import { describe, expect, it } from "vitest";

import {
  createVerifiedHostGraphArtifact,
  type HostGraphProgram,
  type VerifiedHostGraphArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/graph";
import {
  createVerifiedDensePermutationViewCopyArtifacts,
  type VerifiedViewCopyArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  parseWireI64,
  parseWireU64,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  SEMANTIC_HOST_GRAPH_WEBGPU_PROFILE,
  SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
  SemanticHostGraphWebGpuError,
  prepareSemanticHostGraphWebGpu,
  runSemanticHostGraphWebGpu,
  type SemanticHostGraphWebGpuInputBinding,
} from "../src/semantic_host_graph";
import type { KernelDevice } from "../src/types";

const wire = (value: string): WireU64 => parseWireU64(value);
const NO_DEVICE = {} as KernelDevice;

async function identityArtifacts(
  dtype: "f32" | "i32" | "u32" = "f32",
  elements = 2,
): Promise<VerifiedViewCopyArtifacts> {
  return createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: [parseWireI64(String(elements))],
    axes: [0],
    dtype,
  });
}

function artifactOptions(artifacts: VerifiedViewCopyArtifacts) {
  return {
    kernelArtifacts: [artifacts.kernel],
    layoutArtifacts: [artifacts.layout],
  };
}

function resource(
  resourceId: string,
  role: "input" | "temporary" | "output",
  dtype: "f32" | "i32" | "u32",
  byteLength = "8",
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

function dispatch(
  artifacts: VerifiedViewCopyArtifacts,
  nodeId: string,
  source: string,
  destination: string,
  dependsOn: readonly string[],
) {
  return {
    nodeId,
    kind: "dispatch" as const,
    dependsOn,
    semanticArtifactHash: artifacts.kernelSemanticHash,
    entrypointId: artifacts.operationId,
    dimensionBindings: {},
    bindings: [
      {
        semanticResourceId: artifacts.source.viewId,
        graphResourceId: source,
      },
      {
        semanticResourceId: artifacts.destination.viewId,
        graphResourceId: destination,
      },
    ],
  };
}

function pipelineProgram(
  artifacts: VerifiedViewCopyArtifacts,
): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 0 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire("2"),
    resources: [
      resource("input", "input", "f32"),
      resource("temporary", "temporary", "f32"),
      resource("output", "output", "f32"),
    ],
    nodes: [
      dispatch(artifacts, "first", "input", "temporary", []),
      dispatch(
        artifacts,
        "second",
        "temporary",
        "output",
        ["first"],
      ),
    ],
  };
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
    rankCount: wire("3"),
    resources: [
      resource("input", "input", dtype),
      resource("output", "output", dtype),
    ],
    nodes: [
      dispatch(artifacts, "copy", "input", "output", []),
      {
        nodeId: "reduce",
        kind: "all-reduce",
        dependsOn: ["copy"],
        resourceId: "output",
        reduction,
        dtype,
        numericalPolicy,
        participants: [wire("0"), wire("1"), wire("2")],
        result: "replicated-to-all-participants",
      },
    ],
  };
}

function rawCopyProgram(byteLength = "8"): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 1 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire("2"),
    resources: [
      {
        resourceId: "input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "u8",
        byteLength: wire(byteLength),
        alignmentBytes: 1,
      },
      {
        resourceId: "output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "u8",
        byteLength: wire(byteLength),
        alignmentBytes: 1,
      },
    ],
    nodes: [{
      nodeId: "raw-copy",
      kind: "copy",
      dependsOn: [],
      sourceResourceId: "input",
      destinationResourceId: "output",
      mode: "whole-allocation-bytes-per-rank",
    }],
  };
}

function materializedRawCopyProgram(byteLength = "8"): HostGraphProgram {
  const copy = rawCopyProgram(byteLength);
  return {
    ...copy,
    version: { major: 1, minor: 2 },
    nodes: [
      ...copy.nodes,
      {
        nodeId: "materialize-output",
        kind: "materialize",
        dependsOn: ["raw-copy"],
        resourceId: "output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function eventfulRawCopyProgram(byteLength = "8"): HostGraphProgram {
  const program = materializedRawCopyProgram(byteLength);
  return {
    ...program,
    version: { major: 1, minor: 3 },
    nodes: [
      ...program.nodes.filter((node) => node.kind !== "materialize"),
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
    rankCount: wire("2"),
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
        iterationCount: wire("3"),
        body: [{
          nodeId: "reduce-body",
          kind: "all-reduce",
          dependsOn: [],
          resourceId: "output",
          reduction: "sum",
          dtype: "f32",
          numericalPolicy: "rank-order-f32",
          participants: [wire("0"), wire("1")],
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

async function verified(
  program: HostGraphProgram,
  artifacts: VerifiedViewCopyArtifacts,
): Promise<VerifiedHostGraphArtifact> {
  return (await createVerifiedHostGraphArtifact(
    program,
    artifactOptions(artifacts),
  )).artifact;
}

function input(
  rank: number,
  bytes: Uint8Array,
): SemanticHostGraphWebGpuInputBinding {
  return {
    rank: wire(String(rank)),
    resourceId: "input",
    bytes,
  };
}

describe("semantic host-graph WebGPU preparation", () => {
  it("statically expands bounded version-1.4 repetition", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      repeatedCollectiveProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareSemanticHostGraphWebGpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });

    expect(prepared).toMatchObject({
      backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
      nodeCount: 3,
      expandedNodeCount: 5,
      expandedStepCount: 8,
      dispatchStepCount: 0,
      copyStepCount: 2,
      materializationCount: 1,
      eventCount: 0,
      eventIds: [],
      repeatCount: 1,
      repeatIterationCount: 3,
      collectiveReductionStepCount: 3,
      collectiveReplicationStepCount: 3,
    });
    expect(prepared.wgslModuleHashes).toHaveLength(2);
    await expect(prepareSemanticHostGraphWebGpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
      maxExpandedSteps: 7,
    })).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
      path: "$.maxExpandedSteps",
    });
  });

  it("retains version-1.3 completion events without an extra GPU step", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      eventfulRawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareSemanticHostGraphWebGpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });

    expect(prepared).toMatchObject({
      backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
      nodeCount: 3,
      expandedStepCount: 2,
      copyStepCount: 2,
      materializationCount: 1,
      eventCount: 1,
      eventIds: ["copy-complete"],
    });
  });

  it("prepares explicit version-1.2 materialization without an extra GPU step", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      materializedRawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareSemanticHostGraphWebGpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });

    expect(prepared).toMatchObject({
      backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
      nodeCount: 2,
      outputResourceIds: ["output"],
      expandedStepCount: 2,
      dispatchStepCount: 0,
      copyStepCount: 2,
      materializationCount: 1,
      eventCount: 0,
      eventIds: [],
      collectiveReductionStepCount: 0,
      collectiveReplicationStepCount: 0,
    });
  });

  it("lowers version-1.1 raw copy nodes per rank without kernel artifacts", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      rawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareSemanticHostGraphWebGpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });

    expect(prepared).toMatchObject({
      rankCount: "2",
      nodeCount: 1,
      expandedStepCount: 2,
      dispatchStepCount: 0,
      copyStepCount: 2,
      materializationCount: 0,
      eventCount: 0,
      eventIds: [],
      collectiveReductionStepCount: 0,
      collectiveReplicationStepCount: 0,
      plannedTransientGpuBytes: "48",
      plannedTransientHostBytes: "80",
      plannedTransientWorkingSetBytes: "128",
    });
    expect(prepared.wgslModuleHashes).toHaveLength(1);

    const oddGraph = (await createVerifiedHostGraphArtifact(
      rawCopyProgram("7"),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    await expect(prepareSemanticHostGraphWebGpu(oddGraph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    })).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-UNSUPPORTED-PROFILE",
      path: "$.nodes.raw-copy",
    });
  });

  it("expands a multi-rank pipeline at the canonical graph seam", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts), artifacts);
    const prepared = await prepareSemanticHostGraphWebGpu(
      graph,
      artifactOptions(artifacts),
    );

    expect(prepared).toMatchObject({
      profile: SEMANTIC_HOST_GRAPH_WEBGPU_PROFILE,
      backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
      rankCount: "2",
      nodeCount: 2,
      inputResourceIds: ["input"],
      outputResourceIds: ["output"],
      expandedStepCount: 4,
      dispatchStepCount: 4,
      copyStepCount: 0,
      materializationCount: 0,
      eventCount: 0,
      eventIds: [],
      collectiveReductionStepCount: 0,
      collectiveReplicationStepCount: 0,
      plannedTransientGpuBytes: "64",
      plannedTransientHostBytes: "96",
      plannedTransientWorkingSetBytes: "160",
    });
    expect(prepared.wgslModuleHashes).toHaveLength(1);
    expect(prepared.wgslModuleHashes[0]).toMatch(/^[0-9a-f]{64}$/u);
    expect(Object.isFrozen(prepared)).toBe(true);
    expect(Object.isFrozen(prepared.wgslModuleHashes)).toBe(true);
  });

  it("lowers all-reduce to bounded pairwise reduction and replication", async () => {
    for (const [dtype, reduction] of [
      ["f32", "sum"],
      ["i32", "sum"],
      ["u32", "max"],
    ] as const) {
      const artifacts = await identityArtifacts(dtype);
      const graph = await verified(
        collectiveProgram(artifacts, dtype, reduction),
        artifacts,
      );
      const prepared = await prepareSemanticHostGraphWebGpu(
        graph,
        artifactOptions(artifacts),
      );

      expect(prepared).toMatchObject({
        rankCount: "3",
        expandedStepCount: 7,
        dispatchStepCount: 3,
        collectiveReductionStepCount: 2,
        collectiveReplicationStepCount: 2,
      });
      expect(prepared.wgslModuleHashes).toHaveLength(3);
    }
  });

  it("enforces whole-graph step, memory, time, and cancellation budgets", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts), artifacts);
    await expect(prepareSemanticHostGraphWebGpu(graph, {
      ...artifactOptions(artifacts),
      maxExpandedSteps: 3,
    })).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
      path: "$.maxExpandedSteps",
    });
    await expect(prepareSemanticHostGraphWebGpu(graph, {
      ...artifactOptions(artifacts),
      maxTransientWorkingSetBytes: 1,
    })).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
    });
    await expect(prepareSemanticHostGraphWebGpu(graph, {
      ...artifactOptions(artifacts),
      maxPreparationMs: -1,
    })).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
    });
    const controller = new AbortController();
    controller.abort();
    await expect(prepareSemanticHostGraphWebGpu(graph, {
      ...artifactOptions(artifacts),
      signal: controller.signal,
    })).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-CANCELLED",
    });
  });

  it("requires exact graph and semantic authority without invoking accessors", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts), artifacts);
    await expect(prepareSemanticHostGraphWebGpu(
      JSON.parse(JSON.stringify(graph)) as VerifiedHostGraphArtifact,
      artifactOptions(artifacts),
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-INVALID-AUTHORITY",
    });
    await expect(prepareSemanticHostGraphWebGpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [artifacts.layout],
    })).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
    });

    let reads = 0;
    const hostile = artifactOptions(artifacts);
    Object.defineProperty(hostile, "kernelArtifacts", {
      enumerable: true,
      get() {
        reads += 1;
        return [artifacts.kernel];
      },
    });
    await expect(prepareSemanticHostGraphWebGpu(graph, hostile))
      .rejects.toBeInstanceOf(SemanticHostGraphWebGpuError);
    expect(reads).toBe(0);
  });
});

describe("semantic host-graph WebGPU execution admission", () => {
  it("rejects copied preparation authority before observing a device", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts), artifacts);
    const prepared = await prepareSemanticHostGraphWebGpu(
      graph,
      artifactOptions(artifacts),
    );

    await expect(runSemanticHostGraphWebGpu(
      NO_DEVICE,
      { ...prepared },
      { inputs: [] },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-UNVERIFIED-PREPARED",
    });
  });

  it("fails closed for incomplete, duplicate, shared, and misaligned inputs", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts), artifacts);
    const prepared = await prepareSemanticHostGraphWebGpu(
      graph,
      artifactOptions(artifacts),
    );
    const first = new Uint8Array(8);
    const second = new Uint8Array(8);

    await expect(runSemanticHostGraphWebGpu(
      NO_DEVICE,
      prepared,
      { inputs: [input(0, first)] },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
    });
    await expect(runSemanticHostGraphWebGpu(
      NO_DEVICE,
      prepared,
      { inputs: [input(0, first), input(0, second)] },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
    });
    if (typeof SharedArrayBuffer !== "undefined") {
      await expect(runSemanticHostGraphWebGpu(
        NO_DEVICE,
        prepared,
        {
          inputs: [
            input(0, new Uint8Array(new SharedArrayBuffer(8))),
            input(1, second),
          ],
        },
      )).rejects.toMatchObject({
        code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
      });
    }
    const misaligned = new Uint8Array(new ArrayBuffer(9), 1, 8);
    await expect(runSemanticHostGraphWebGpu(
      NO_DEVICE,
      prepared,
      { inputs: [input(0, misaligned), input(1, second)] },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
    });
  });

  it("rejects request accessors and pre-aborted runs without device access", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts), artifacts);
    const prepared = await prepareSemanticHostGraphWebGpu(
      graph,
      artifactOptions(artifacts),
    );
    let reads = 0;
    const hostile = {} as { inputs: readonly SemanticHostGraphWebGpuInputBinding[] };
    Object.defineProperty(hostile, "inputs", {
      enumerable: true,
      get() {
        reads += 1;
        return [];
      },
    });
    await expect(runSemanticHostGraphWebGpu(
      NO_DEVICE,
      prepared,
      hostile,
    )).rejects.toBeInstanceOf(SemanticHostGraphWebGpuError);
    expect(reads).toBe(0);

    const controller = new AbortController();
    controller.abort();
    await expect(runSemanticHostGraphWebGpu(
      NO_DEVICE,
      prepared,
      { inputs: [] },
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-CANCELLED",
    });
  });

  it("classifies revoked request, byte, and signal proxies without device access", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts), artifacts);
    const prepared = await prepareSemanticHostGraphWebGpu(
      graph,
      artifactOptions(artifacts),
    );
    const requestProxy = Proxy.revocable({ inputs: [] }, {});
    requestProxy.revoke();
    await expect(runSemanticHostGraphWebGpu(
      NO_DEVICE,
      prepared,
      requestProxy.proxy,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path: "$.request",
    });

    const bytesProxy = Proxy.revocable(new Uint8Array(8), {});
    bytesProxy.revoke();
    await expect(runSemanticHostGraphWebGpu(
      NO_DEVICE,
      prepared,
      {
        inputs: [
          input(0, bytesProxy.proxy),
          input(1, new Uint8Array(8)),
        ],
      },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path: "$.request.inputs[0].bytes",
    });

    const signalProxy = Proxy.revocable(new AbortController().signal, {});
    signalProxy.revoke();
    await expect(runSemanticHostGraphWebGpu(
      NO_DEVICE,
      prepared,
      { inputs: [] },
      { signal: signalProxy.proxy },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path: "$.options.signal",
    });
  });

  it("rejects device allocation limits before creating GPU resources", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts), artifacts);
    const prepared = await prepareSemanticHostGraphWebGpu(
      graph,
      artifactOptions(artifacts),
    );
    const gpu = {
      features: new Set<string>(),
      limits: {
        maxBufferSize: 4,
        maxStorageBufferBindingSize: 4,
        maxComputeWorkgroupsPerDimension: 65_535,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxBindingsPerBindGroup: 8,
        maxStorageBuffersPerShaderStage: 8,
      },
      lost: new Promise<GPUDeviceLostInfo>(() => undefined),
    } as unknown as GPUDevice;
    const device = {
      gpu,
      clearCache() {},
      getStats() {
        throw new Error("device stats must not be observed");
      },
    } as KernelDevice;

    await expect(runSemanticHostGraphWebGpu(
      device,
      prepared,
      {
        inputs: [
          input(0, new Uint8Array(8)),
          input(1, new Uint8Array(8)),
        ],
      },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-DEVICE-LIMIT",
      path: "$.device.limits.maxBufferSize",
    });
  });

  it("classifies a hostile kernel-device getter before GPU work", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts), artifacts);
    const prepared = await prepareSemanticHostGraphWebGpu(
      graph,
      artifactOptions(artifacts),
    );
    let reads = 0;
    const hostile = {
      get gpu(): GPUDevice {
        reads += 1;
        throw new Error("hostile gpu getter");
      },
      clearCache() {},
      getStats() {
        throw new Error("device stats must not be observed");
      },
    } as KernelDevice;

    await expect(runSemanticHostGraphWebGpu(
      hostile,
      prepared,
      {
        inputs: [
          input(0, new Uint8Array(8)),
          input(1, new Uint8Array(8)),
        ],
      },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path: "$.device",
    });
    expect(reads).toBe(1);
  });
});
