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

async function rectangularIdentityArtifacts(
  shape: readonly number[],
): Promise<VerifiedViewCopyArtifacts> {
  return createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: shape.map((extent) => parseWireI64(String(extent))),
    axes: shape.map((_, axis) => axis),
    dtype: "f32",
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

function dynamicPipelineProgram(
  artifacts: VerifiedViewCopyArtifacts,
): HostGraphProgram {
  const base = pipelineProgram(artifacts);
  return {
    ...base,
    version: { major: 1, minor: 9 },
    nodes: [
      ...base.nodes.map((node) =>
        node.kind === "dispatch" && node.nodeId === "first"
          ? {
              ...node,
              kind: "dynamic-dispatch" as const,
              launchControl: {
                controlId: "prefix-elements",
                mode: "u32-prefix-element-count" as const,
              },
              maxElementCount: wire("2"),
              mode: "runtime-u32-prefix-elements" as const,
            }
          : node),
      {
        nodeId: "materialize-output",
        kind: "materialize",
        dependsOn: ["second"],
        resourceId: "output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function rectangularDynamicProgram(
  artifacts: VerifiedViewCopyArtifacts,
  shape: readonly number[],
): HostGraphProgram {
  const byteLength = String(
    shape.reduce((product, extent) => product * extent, 1) * 4,
  );
  return {
    kind: "host-graph",
    version: {
      major: 1,
      minor: shape.length === 5 ? 16 : shape.length === 4 ? 14 : 12,
    },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire("1"),
    resources: [
      resource("input", "input", "f32", byteLength),
      resource("output", "output", "f32", byteLength),
    ],
    nodes: [
      {
        ...dispatch(artifacts, "copy-region", "input", "output", []),
        kind: "dynamic-dispatch",
        launchControls: shape.map((_, axis) => ({
          axis,
          controlId: `prefix-axis-${axis}`,
          mode: "u32-prefix-extent" as const,
        })),
        maxExtents: shape.map((extent) => wire(String(extent))),
        mode: "runtime-u32-rectangular-prefix",
      },
      {
        nodeId: "materialize-output",
        kind: "materialize",
        dependsOn: ["copy-region"],
        resourceId: "output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function resourceRectangularDynamicProgram(
  artifacts: VerifiedViewCopyArtifacts,
  shape: readonly number[],
): HostGraphProgram {
  const base = rectangularDynamicProgram(artifacts, shape);
  const producerIds = shape.map((_, axis) => `produce-extent-${axis}`);
  return {
    ...base,
    version: { major: 1, minor: shape.length === 4 ? 15 : 13 },
    resources: [
      ...base.resources,
      ...shape.flatMap((_, axis) => [
        resource(`extent-input-${axis}`, "input", "u32", "4"),
        resource(`extent-${axis}`, "temporary", "u32", "4"),
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
            rank: wire("0"),
            mode: "u32-prefix-extent" as const,
          })),
          mode: "resource-u32-rectangular-prefix" as const,
        };
      }),
    ],
  };
}

function resourceDynamicPipelineProgram(
  artifacts: VerifiedViewCopyArtifacts,
): HostGraphProgram {
  const base = dynamicPipelineProgram(artifacts);
  return {
    ...base,
    version: { major: 1, minor: 11 },
    resources: [
      ...base.resources,
      resource("launch-input", "input", "u32", "4"),
      resource("launch-count", "temporary", "u32", "4"),
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
            rank: wire("0"),
            mode: "u32-prefix-element-count" as const,
          },
          mode: "resource-u32-prefix-elements" as const,
        };
      }),
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
        maxIterationCount: wire("3"),
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
      resource("iteration-input", "input", "u32", "4"),
      resource("iteration-count", "temporary", "u32", "4"),
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
            rank: wire("0"),
            mode: "u32-count" as const,
          },
          maxIterationCount: wire("3"),
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
    rankCount: wire("1"),
    resources: [
      resource("predicate", "input", "u32", "4"),
      {
        ...resource("then-input", "input", "u32"),
        dtype: "u8",
        alignmentBytes: 1,
      },
      {
        ...resource("else-input", "input", "u32"),
        dtype: "u8",
        alignmentBytes: 1,
      },
      {
        ...resource("output", "output", "u32"),
        dtype: "u8",
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
          rank: wire("0"),
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
      resource("predicate-source", "input", "u32", "4"),
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
  it("pre-lowers both bounded conditional branches with exact public counts", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      conditionalRawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareSemanticHostGraphWebGpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });

    expect(prepared).toMatchObject({
      backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
      nodeCount: 2,
      expandedNodeCount: 2,
      expandedStepCount: 1,
      dispatchStepCount: 0,
      copyStepCount: 1,
      materializationCount: 1,
      conditionalCount: 1,
      conditionalNodeIds: ["choose-output"],
      collectiveReductionStepCount: 0,
      collectiveReplicationStepCount: 0,
      plannedTransientGpuBytes: "32",
      plannedTransientHostBytes: "60",
      plannedTransientWorkingSetBytes: "92",
    });
    expect(prepared.wgslModuleHashes).toHaveLength(1);
  });

  it("pre-lowers version-1.6 runtime control through the same branch path", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      runtimeConditionalRawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareSemanticHostGraphWebGpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });

    expect(prepared).toMatchObject({
      backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
      nodeCount: 2,
      expandedNodeCount: 2,
      expandedStepCount: 1,
      copyStepCount: 1,
      conditionalCount: 1,
      conditionalNodeIds: ["choose-output"],
      runtimeControlIds: ["choose"],
    });
    expect(prepared.wgslModuleHashes).toHaveLength(1);
  });

  it("pre-lowers version-1.7 resource control with one bounded feedback stage", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      resourceConditionalRawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareSemanticHostGraphWebGpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });

    expect(prepared).toMatchObject({
      backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
      nodeCount: 3,
      expandedNodeCount: 3,
      expandedStepCount: 2,
      copyStepCount: 2,
      conditionalCount: 1,
      conditionalNodeIds: ["choose-output"],
      resourceConditionalCount: 1,
      midGraphFeedbackCount: 1,
      runtimeControlIds: [],
      plannedTransientGpuBytes: "44",
      plannedTransientHostBytes: "72",
      plannedTransientWorkingSetBytes: "116",
    });
    expect(prepared.wgslModuleHashes).toHaveLength(1);
  });

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

  it("prewarms the maximum version-1.8 runtime repeat schedule", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      runtimeRepeatedCollectiveProgram(),
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
      copyStepCount: 2,
      repeatCount: 1,
      repeatIterationCount: 3,
      runtimeRepeatCount: 1,
      runtimeControlIds: ["iterations"],
      collectiveReductionStepCount: 3,
      collectiveReplicationStepCount: 3,
    });
  });

  it("prewarms one version-1.10 produced-resource repeat schedule", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      resourceRepeatedCollectiveProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareSemanticHostGraphWebGpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });

    expect(prepared).toMatchObject({
      backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
      nodeCount: 4,
      expandedNodeCount: 6,
      expandedStepCount: 10,
      copyStepCount: 4,
      repeatCount: 1,
      repeatIterationCount: 3,
      runtimeRepeatCount: 0,
      resourceRepeatCount: 1,
      resourceConditionalCount: 0,
      midGraphFeedbackCount: 1,
      runtimeControlIds: [],
      collectiveReductionStepCount: 3,
      collectiveReplicationStepCount: 3,
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

  it("prewarms a bounded version-1.9 dynamic dispatch maximum", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(
      dynamicPipelineProgram(artifacts),
      artifacts,
    );
    const prepared = await prepareSemanticHostGraphWebGpu(
      graph,
      artifactOptions(artifacts),
    );

    expect(prepared).toMatchObject({
      backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
      nodeCount: 3,
      expandedStepCount: 4,
      dispatchStepCount: 4,
      materializationCount: 1,
      dynamicDispatchCount: 1,
      runtimeControlIds: ["prefix-elements"],
      plannedTransientGpuBytes: "96",
      plannedTransientHostBytes: "104",
      plannedTransientWorkingSetBytes: "200",
    });
    const oneLane = await prepareSemanticHostGraphWebGpu(
      graph,
      { ...artifactOptions(artifacts), workgroupSize: 1 },
    );
    expect(oneLane).toMatchObject({
      dynamicDispatchCount: 1,
      expandedStepCount: 4,
      dispatchStepCount: 4,
      runtimeControlIds: ["prefix-elements"],
    });
    const aligned = await prepareSemanticHostGraphWebGpu(
      graph,
      { ...artifactOptions(artifacts), workgroupSize: 2 },
    );
    expect(aligned).toMatchObject({
      dynamicDispatchCount: 1,
      expandedStepCount: 4,
      dispatchStepCount: 4,
      runtimeControlIds: ["prefix-elements"],
    });
  });

  it("prewarms bounded rank-2 through rank-5 rectangular dynamic dispatch", async () => {
    for (const shape of [
      [3, 4],
      [2, 3, 4],
      [2, 2, 3, 4],
      [2, 2, 2, 3, 4],
    ] as const) {
      const artifacts = await rectangularIdentityArtifacts(shape);
      const graph = await verified(
        rectangularDynamicProgram(artifacts, shape),
        artifacts,
      );
      const prepared = await prepareSemanticHostGraphWebGpu(
        graph,
        { ...artifactOptions(artifacts), workgroupSize: 64 },
      );

      expect(prepared).toMatchObject({
        backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
        nodeCount: 2,
        expandedStepCount: 1,
        dispatchStepCount: 1,
        materializationCount: 1,
        dynamicDispatchCount: 1,
        runtimeControlIds: shape.map((_, axis) => `prefix-axis-${axis}`),
      });
      expect(Number(prepared.plannedTransientGpuBytes)).toBe(
        shape.reduce((product, extent) => product * extent, 1) * 12 +
          (shape.length === 5 ? 32 : 16),
      );
      expect(Number(prepared.plannedTransientHostBytes)).toBe(
        shape.reduce((product, extent) => product * extent, 1) * 20 +
          (shape.length === 5 ? 32 : 16),
      );
      expect(prepared.wgslModuleHashes).toHaveLength(1);
    }
  });

  it("prewarms one produced rank-2 through rank-4 rectangle with exact feedback budgets", async () => {
    for (const shape of [
      [3, 4],
      [2, 3, 4],
      [2, 2, 3, 4],
    ] as const) {
      const artifacts = await rectangularIdentityArtifacts(shape);
      const graph = await verified(
        resourceRectangularDynamicProgram(artifacts, shape),
        artifacts,
      );
      const prepared = await prepareSemanticHostGraphWebGpu(
        graph,
        { ...artifactOptions(artifacts), workgroupSize: 64 },
      );
      const elementCount = shape.reduce(
        (product, extent) => product * extent,
        1,
      );

      expect(prepared).toMatchObject({
        backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
        nodeCount: shape.length + 2,
        expandedStepCount: shape.length + 1,
        dispatchStepCount: 1,
        copyStepCount: shape.length,
        materializationCount: 1,
        dynamicDispatchCount: 1,
        resourceDynamicDispatchCount: 1,
        midGraphFeedbackCount: 1,
        runtimeControlIds: [],
        plannedTransientGpuBytes:
          String(elementCount * 12 + 16 + shape.length * 12),
        plannedTransientHostBytes:
          String(elementCount * 20 + 16 + shape.length * 16),
      });
      expect(prepared.wgslModuleHashes).toHaveLength(2);
    }
  });

  it("admits every rectangular workgroup dimension before GPU allocation", async () => {
    const shape = [3, 4] as const;
    const artifacts = await rectangularIdentityArtifacts(shape);
    const graph = await verified(
      rectangularDynamicProgram(artifacts, shape),
      artifacts,
    );
    const prepared = await prepareSemanticHostGraphWebGpu(
      graph,
      { ...artifactOptions(artifacts), workgroupSize: 64 },
    );
    const gpu = {
      features: new Set<string>(),
      limits: {
        maxBufferSize: 1024,
        maxStorageBufferBindingSize: 1024,
        maxComputeWorkgroupsPerDimension: 2,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxBindingsPerBindGroup: 8,
        maxStorageBuffersPerShaderStage: 8,
        maxUniformBuffersPerShaderStage: 8,
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
        inputs: [input(0, new Uint8Array(48))],
        controls: [
          { controlId: "prefix-axis-0", value: wire("3") },
          { controlId: "prefix-axis-1", value: wire("4") },
        ],
      },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-DEVICE-LIMIT",
      path: "$.steps[0].launch[1]",
    });

    const rankFourShape = [2, 2, 2, 2] as const;
    const rankFourArtifacts = await rectangularIdentityArtifacts(
      rankFourShape,
    );
    const rankFourGraph = await verified(
      rectangularDynamicProgram(rankFourArtifacts, rankFourShape),
      rankFourArtifacts,
    );
    const rankFourPrepared = await prepareSemanticHostGraphWebGpu(
      rankFourGraph,
      { ...artifactOptions(rankFourArtifacts), workgroupSize: 64 },
    );
    await expect(runSemanticHostGraphWebGpu(
      device,
      rankFourPrepared,
      {
        inputs: [input(0, new Uint8Array(64))],
        controls: rankFourShape.map((extent, axis) => ({
          controlId: `prefix-axis-${axis}`,
          value: wire(String(extent)),
        })),
      },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-DEVICE-LIMIT",
      path: "$.steps[0].launch[2]",
    });
  });

  it("rejects rectangular extents before observing input bytes", async () => {
    const shape = [3, 4] as const;
    const artifacts = await rectangularIdentityArtifacts(shape);
    const graph = await verified(
      rectangularDynamicProgram(artifacts, shape),
      artifacts,
    );
    const prepared = await prepareSemanticHostGraphWebGpu(
      graph,
      artifactOptions(artifacts),
    );
    let inputReads = 0;
    const unreadInput = {
      rank: wire("0"),
      resourceId: "input",
    } as SemanticHostGraphWebGpuInputBinding;
    Object.defineProperty(unreadInput, "bytes", {
      enumerable: true,
      get() {
        inputReads += 1;
        return new Uint8Array(48);
      },
    });

    for (const extents of [[0, 4], [3, 5]] as const) {
      await expect(runSemanticHostGraphWebGpu(
        NO_DEVICE,
        prepared,
        {
          inputs: [unreadInput],
          controls: extents.map((value, axis) => ({
            controlId: `prefix-axis-${axis}`,
            value: wire(String(value)),
          })),
        },
      )).rejects.toMatchObject({
        code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
      });
    }
    expect(inputReads).toBe(0);
  });

  it("prewarms one bounded version-1.11 produced-resource dynamic dispatch", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(
      resourceDynamicPipelineProgram(artifacts),
      artifacts,
    );
    const prepared = await prepareSemanticHostGraphWebGpu(
      graph,
      artifactOptions(artifacts),
    );

    expect(prepared).toMatchObject({
      backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
      nodeCount: 4,
      expandedStepCount: 6,
      dispatchStepCount: 4,
      copyStepCount: 2,
      materializationCount: 1,
      dynamicDispatchCount: 1,
      resourceDynamicDispatchCount: 1,
      midGraphFeedbackCount: 1,
      runtimeControlIds: [],
    });
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
  it("rejects a runtime repeat count above its artifact bound before device access", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      runtimeRepeatedCollectiveProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareSemanticHostGraphWebGpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });
    let inputReads = 0;
    const unreadInput = {
      rank: wire("0"),
      resourceId: "input",
    } as SemanticHostGraphWebGpuInputBinding;
    Object.defineProperty(unreadInput, "bytes", {
      enumerable: true,
      get() {
        inputReads += 1;
        return new Uint8Array(8);
      },
    });
    await expect(runSemanticHostGraphWebGpu(
      NO_DEVICE,
      prepared,
      {
        inputs: [
          unreadInput,
          input(1, new Uint8Array(8)),
        ],
        controls: [{
          controlId: "iterations",
          value: wire("4"),
        }],
      },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path: "$.request.controls[0].value",
    });
    expect(inputReads).toBe(0);
  });

  it("rejects dynamic dispatch values outside the artifact bound before device access", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(
      dynamicPipelineProgram(artifacts),
      artifacts,
    );
    const prepared = await prepareSemanticHostGraphWebGpu(
      graph,
      { ...artifactOptions(artifacts), workgroupSize: 1 },
    );
    let inputReads = 0;
    const unreadInput = {
      rank: wire("0"),
      resourceId: "input",
    } as SemanticHostGraphWebGpuInputBinding;
    Object.defineProperty(unreadInput, "bytes", {
      enumerable: true,
      get() {
        inputReads += 1;
        return new Uint8Array(8);
      },
    });

    for (const value of ["0", "3"]) {
      await expect(runSemanticHostGraphWebGpu(
        NO_DEVICE,
        prepared,
        {
          inputs: [unreadInput, input(1, new Uint8Array(8))],
          controls: [{
            controlId: "prefix-elements",
            value: wire(value),
          }],
        },
      )).rejects.toMatchObject({
        code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
        path: "$.request.controls[0].value",
      });
    }
    expect(inputReads).toBe(0);
  });

  it("rejects missing, duplicate, unknown, and out-of-range controls before device access", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      runtimeConditionalRawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareSemanticHostGraphWebGpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });
    const inputs = [
      {
        rank: wire("0"),
        resourceId: "then-input",
        bytes: new Uint8Array(8),
      },
      {
        rank: wire("0"),
        resourceId: "else-input",
        bytes: new Uint8Array(8),
      },
    ];
    await expect(runSemanticHostGraphWebGpu(
      NO_DEVICE,
      prepared,
      { inputs: null as unknown as typeof inputs },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path: "$.request.controls",
    });
    for (const controls of [
      undefined,
      [
        { controlId: "choose", value: wire("0") },
        { controlId: "choose", value: wire("1") },
      ],
      [{ controlId: "other", value: wire("0") }],
      [{ controlId: "choose", value: wire("4294967296") }],
    ]) {
      await expect(runSemanticHostGraphWebGpu(
        NO_DEVICE,
        prepared,
        {
          inputs,
          ...(controls === undefined ? {} : { controls }),
        },
      )).rejects.toMatchObject({
        code: "BG-WEBGPU-GRAPH-INVALID-BINDING",
      });
    }
  });

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
        maxUniformBuffersPerShaderStage: 8,
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

  it("rejects missing dynamic-prefix uniform capacity before GPU work", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(
      dynamicPipelineProgram(artifacts),
      artifacts,
    );
    const prepared = await prepareSemanticHostGraphWebGpu(
      graph,
      artifactOptions(artifacts),
    );
    const gpu = {
      features: new Set<string>(),
      limits: {
        maxBufferSize: 1024,
        maxStorageBufferBindingSize: 1024,
        maxComputeWorkgroupsPerDimension: 65_535,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxBindingsPerBindGroup: 8,
        maxStorageBuffersPerShaderStage: 8,
        maxUniformBuffersPerShaderStage: 0,
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
        controls: [{
          controlId: "prefix-elements",
          value: wire("1"),
        }],
      },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-DEVICE-LIMIT",
      path: "$.device.limits.maxUniformBuffersPerShaderStage",
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
