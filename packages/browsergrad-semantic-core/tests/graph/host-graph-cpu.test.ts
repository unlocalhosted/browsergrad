import { describe, expect, it } from "vitest";

import {
  HOST_GRAPH_CPU_PROFILE,
  HostGraphCpuError,
  createVerifiedHostGraphArtifact,
  prepareHostGraphCpu,
  type HostGraphCpuInputBinding,
  type HostGraphProgram,
  type VerifiedHostGraphArtifact,
} from "../../src/graph";
import {
  createVerifiedDensePermutationViewCopyArtifacts,
  type VerifiedViewCopyArtifacts,
} from "../../src/kernel";
import {
  parseWireI64,
  parseWireU64,
  type WireU64,
} from "../../src/schema";

const wire = (value: string): WireU64 => parseWireU64(value);

async function identityArtifacts(
  dtype: "f32" | "i32" | "u32" = "f32",
): Promise<VerifiedViewCopyArtifacts> {
  return createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: [parseWireI64("2")],
    axes: [0],
    dtype,
  });
}

async function rectangularArtifacts(
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
  dtype: "f32" | "i32" | "u32",
): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 0 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire("1"),
    resources: [
      resource("input", "input", dtype),
      resource("temporary", "temporary", dtype),
      resource("output", "output", dtype),
    ],
    nodes: [
      dispatch(artifacts, "first", "input", "temporary", []),
      dispatch(artifacts, "second", "temporary", "output", ["first"]),
    ],
  };
}

function dynamicPipelineProgram(
  artifacts: VerifiedViewCopyArtifacts,
): HostGraphProgram {
  const base = pipelineProgram(artifacts, "f32");
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

function resourceDynamicPipelineProgram(
  artifacts: VerifiedViewCopyArtifacts,
): HostGraphProgram {
  const base = dynamicPipelineProgram(artifacts);
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
        byteLength: wire("4"),
        alignmentBytes: 4,
      },
      {
        resourceId: "launch-count",
        role: "temporary",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "u32",
        byteLength: wire("4"),
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
            rank: wire("0"),
            mode: "u32-prefix-element-count" as const,
          },
          mode: "resource-u32-prefix-elements" as const,
        };
      }),
    ],
  };
}

function resourceDynamicFanoutProgram(
  artifacts: VerifiedViewCopyArtifacts,
): HostGraphProgram {
  const base = resourceDynamicPipelineProgram(artifacts);
  const dynamic = base.nodes.find((node) =>
    node.kind === "dynamic-dispatch");
  if (
    dynamic?.kind !== "dynamic-dispatch" ||
    dynamic.mode !== "resource-u32-prefix-elements"
  ) {
    throw new Error("missing resource dynamic dispatch");
  }
  return {
    ...base,
    version: { major: 1, minor: 24 },
    resources: [
      ...base.resources,
      {
        resourceId: "fanout-output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "f32",
        byteLength: wire("8"),
        alignmentBytes: 4,
      },
    ],
    nodes: [
      ...base.nodes.filter((node) => node.kind !== "materialize"),
      {
        ...dynamic,
        nodeId: "fanout",
        dependsOn: [...dynamic.dependsOn],
        dimensionBindings: { ...dynamic.dimensionBindings },
        launchSource: { ...dynamic.launchSource },
        bindings: dynamic.bindings.map((binding) => ({
          ...binding,
          graphResourceId: binding.graphResourceId === "temporary"
            ? "fanout-output"
            : binding.graphResourceId,
        })),
      },
      ...base.nodes.filter((node) => node.kind === "materialize"),
      {
        nodeId: "materialize-fanout-output",
        kind: "materialize",
        dependsOn: ["fanout"],
        resourceId: "fanout-output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

type SequentialFeedbackStageCount = 2 | 3 | 4 | 5 | 6 | 7 | 8;

function sequentialFeedbackStages(stageCount: SequentialFeedbackStageCount) {
  const legacy = [
    {
      inputResourceId: "next-count-input",
      outputResourceId: "next-count",
      nodeId: "produce-next-count",
    },
    {
      inputResourceId: "final-count-input",
      outputResourceId: "final-count",
      nodeId: "produce-final-count",
    },
    {
      inputResourceId: "terminal-count-input",
      outputResourceId: "terminal-count",
      nodeId: "produce-terminal-count",
    },
  ] as const;
  return Array.from({ length: stageCount - 1 }, (_, index) =>
    legacy[index] ?? {
      inputResourceId: `feedback-count-input-${index + 1}`,
      outputResourceId: `feedback-count-${index + 1}`,
      nodeId: `produce-feedback-count-${index + 1}`,
    });
}

function sequentialResourceDynamicProgram(
  countArtifacts: VerifiedViewCopyArtifacts,
  dataArtifacts: VerifiedViewCopyArtifacts,
  stageCount: SequentialFeedbackStageCount = 2,
): HostGraphProgram {
  const stages = sequentialFeedbackStages(stageCount);
  const countResources = stages.flatMap((stage) => [
    resource(stage.inputResourceId, "input", "u32", "4"),
    resource(stage.outputResourceId, "temporary", "u32", "4"),
  ]);
  const countDispatches = stages.map((stage, index) => {
    const predecessor = stages[index - 1];
    return {
      nodeId: stage.nodeId,
      kind: "dynamic-dispatch" as const,
      dependsOn: [
        predecessor?.nodeId ?? "produce-launch-count",
      ],
      semanticArtifactHash: countArtifacts.kernelSemanticHash,
      entrypointId: countArtifacts.operationId,
      dimensionBindings: {},
      bindings: [
        {
          semanticResourceId: countArtifacts.source.viewId,
          graphResourceId: stage.inputResourceId,
        },
        {
          semanticResourceId: countArtifacts.destination.viewId,
          graphResourceId: stage.outputResourceId,
        },
      ],
      launchSource: {
        resourceId:
          predecessor?.outputResourceId ?? "launch-count",
        rank: wire("0"),
        mode: "u32-prefix-element-count" as const,
      },
      maxElementCount: wire("1"),
      mode: "resource-u32-prefix-elements" as const,
    };
  });
  const finalStage = stages.at(-1)!;
  return {
    kind: "host-graph",
    version: {
      major: 1,
      minor: stageCount === 2
        ? 26
        : stageCount === 3
          ? 27
          : stageCount === 4
            ? 28
            : 34,
    },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire("1"),
    resources: [
      {
        resourceId: "launch-input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "u32",
        byteLength: wire("4"),
        alignmentBytes: 4,
      },
      {
        resourceId: "launch-count",
        role: "temporary",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "u32",
        byteLength: wire("4"),
        alignmentBytes: 4,
      },
      ...countResources,
      {
        resourceId: "input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "f32",
        byteLength: wire("8"),
        alignmentBytes: 4,
      },
      {
        resourceId: "output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "f32",
        byteLength: wire("8"),
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
      ...countDispatches,
      {
        nodeId: "copy-selected-prefix",
        kind: "dynamic-dispatch",
        dependsOn: [finalStage.nodeId],
        semanticArtifactHash: dataArtifacts.kernelSemanticHash,
        entrypointId: dataArtifacts.operationId,
        dimensionBindings: {},
        bindings: [
          {
            semanticResourceId: dataArtifacts.source.viewId,
            graphResourceId: "input",
          },
          {
            semanticResourceId: dataArtifacts.destination.viewId,
            graphResourceId: "output",
          },
        ],
        launchSource: {
          resourceId: finalStage.outputResourceId,
          rank: wire("0"),
          mode: "u32-prefix-element-count",
        },
        maxElementCount: wire("2"),
        mode: "resource-u32-prefix-elements",
      },
      {
        nodeId: "materialize-output",
        kind: "materialize",
        dependsOn: ["copy-selected-prefix"],
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
  const byteLength = shape.reduce(
    (product, extent) => product * extent,
    1,
  ) * 4;
  return {
    kind: "host-graph",
    version: {
      major: 1,
      minor: shape.length === 8
        ? 22
        : shape.length === 7
          ? 20
          : shape.length === 6
            ? 18
            : shape.length === 5
              ? 16
              : shape.length === 4
                ? 14
                : 12,
    },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire("1"),
    resources: [
      {
        resourceId: "input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "f32",
        byteLength: wire(String(byteLength)),
        alignmentBytes: 4,
      },
      {
        resourceId: "output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "f32",
        byteLength: wire(String(byteLength)),
        alignmentBytes: 4,
      },
    ],
    nodes: [
      {
        nodeId: "copy-region",
        kind: "dynamic-dispatch",
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
    version: {
      major: 1,
      minor: shape.length === 8
        ? 23
        : shape.length === 7
          ? 21
          : shape.length === 6
            ? 19
            : shape.length === 5
              ? 17
              : shape.length === 4
                ? 15
                : 13,
    },
    resources: [
      ...base.resources,
      ...shape.flatMap((_, axis) => [
        {
          resourceId: `extent-input-${axis}`,
          role: "input" as const,
          multiplicity: "per-rank" as const,
          initialization: "external-input" as const,
          dtype: "u32" as const,
          byteLength: wire("4"),
          alignmentBytes: 4,
        },
        {
          resourceId: `extent-${axis}`,
          role: "temporary" as const,
          multiplicity: "per-rank" as const,
          initialization: "zero-fill" as const,
          dtype: "u32" as const,
          byteLength: wire("4"),
          alignmentBytes: 4,
        },
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

function resourceRectangularDynamicFanoutProgram(
  artifacts: VerifiedViewCopyArtifacts,
  shape: readonly number[],
): HostGraphProgram {
  const base = resourceRectangularDynamicProgram(artifacts, shape);
  const dynamic = base.nodes.find((node) =>
    node.kind === "dynamic-dispatch");
  if (
    dynamic?.kind !== "dynamic-dispatch" ||
    dynamic.mode !== "resource-u32-rectangular-prefix"
  ) {
    throw new Error("missing resource rectangular dynamic dispatch");
  }
  const byteLength = shape.reduce(
    (product, extent) => product * extent,
    1,
  ) * 4;
  return {
    ...base,
    version: { major: 1, minor: 25 },
    resources: [
      ...base.resources,
      {
        resourceId: "fanout-output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "f32",
        byteLength: wire(String(byteLength)),
        alignmentBytes: 4,
      },
    ],
    nodes: [
      ...base.nodes.filter((node) => node.kind !== "materialize"),
      {
        ...dynamic,
        nodeId: "fanout-region",
        dependsOn: [...dynamic.dependsOn],
        dimensionBindings: { ...dynamic.dimensionBindings },
        launchSources: dynamic.launchSources.map((source) => ({
          ...source,
        })),
        maxExtents: [...dynamic.maxExtents],
        bindings: dynamic.bindings.map((binding) => ({
          ...binding,
          graphResourceId: binding.graphResourceId === "output"
            ? "fanout-output"
            : binding.graphResourceId,
        })),
      },
      ...base.nodes.filter((node) => node.kind === "materialize"),
      {
        nodeId: "materialize-fanout-output",
        kind: "materialize",
        dependsOn: ["fanout-region"],
        resourceId: "fanout-output",
        mode: "host-readback-after-graph-success",
      },
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
    rankCount: wire("2"),
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
        participants: [wire("0"), wire("1")],
        result: "replicated-to-all-participants",
      },
    ],
  };
}

function rawCopyProgram(): HostGraphProgram {
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
        byteLength: wire("7"),
        alignmentBytes: 1,
      },
      {
        resourceId: "output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "u8",
        byteLength: wire("7"),
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

function materializedRawCopyProgram(): HostGraphProgram {
  const copy = rawCopyProgram();
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

function eventfulRawCopyProgram(): HostGraphProgram {
  const program = materializedRawCopyProgram();
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
          nodeId: "reduce-step",
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
      {
        resourceId: "iteration-input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "u32",
        byteLength: wire("4"),
        alignmentBytes: 4,
      },
      {
        resourceId: "iteration-count",
        role: "temporary",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "u32",
        byteLength: wire("4"),
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

function sharedConditionalRepeatFeedbackProgram(): HostGraphProgram {
  const base = resourceRepeatedCollectiveProgram();
  return {
    ...base,
    version: { major: 1, minor: 29 },
    resources: [
      ...base.resources,
      resource("then-input", "input", "u32"),
      resource("else-input", "input", "u32"),
      resource("branch-output", "output", "u32"),
    ],
    nodes: [
      ...base.nodes,
      {
        nodeId: "choose-branch",
        kind: "conditional",
        dependsOn: ["produce-iteration-count"],
        predicate: {
          resourceId: "iteration-count",
          rank: wire("0"),
          mode: "u32-nonzero",
        },
        thenBody: [{
          nodeId: "copy-then",
          kind: "copy",
          dependsOn: [],
          sourceResourceId: "then-input",
          destinationResourceId: "branch-output",
          mode: "whole-allocation-bytes-per-rank",
        }],
        elseBody: [{
          nodeId: "copy-else",
          kind: "copy",
          dependsOn: [],
          sourceResourceId: "else-input",
          destinationResourceId: "branch-output",
          mode: "whole-allocation-bytes-per-rank",
        }],
        mode: "resource-u32-branch-sequential",
      },
      {
        nodeId: "materialize-branch-output",
        kind: "materialize",
        dependsOn: ["choose-branch"],
        resourceId: "branch-output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function sequentialConditionalRepeatFeedbackProgram(): HostGraphProgram {
  const base = repeatedCollectiveProgram();
  return {
    ...base,
    version: { major: 1, minor: 30 },
    resources: [
      ...base.resources,
      resource("predicate-input", "input", "u32", "4"),
      resource("predicate", "temporary", "u32", "4"),
      resource("then-count-input", "input", "u32", "4"),
      resource("else-count-input", "input", "u32", "4"),
      resource("iteration-count", "temporary", "u32", "4"),
    ],
    nodes: [
      {
        nodeId: "produce-predicate",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "predicate-input",
        destinationResourceId: "predicate",
        mode: "whole-allocation-bytes-per-rank",
      },
      base.nodes[0]!,
      {
        nodeId: "choose-iteration-count",
        kind: "conditional",
        dependsOn: ["produce-predicate"],
        predicate: {
          resourceId: "predicate",
          rank: wire("0"),
          mode: "u32-nonzero",
        },
        thenBody: [{
          nodeId: "copy-then-count",
          kind: "copy",
          dependsOn: [],
          sourceResourceId: "then-count-input",
          destinationResourceId: "iteration-count",
          mode: "whole-allocation-bytes-per-rank",
        }],
        elseBody: [{
          nodeId: "copy-else-count",
          kind: "copy",
          dependsOn: [],
          sourceResourceId: "else-count-input",
          destinationResourceId: "iteration-count",
          mode: "whole-allocation-bytes-per-rank",
        }],
        mode: "resource-u32-branch-sequential",
      },
      {
        nodeId: "repeat-reduction",
        kind: "repeat",
        dependsOn: ["initialize", "choose-iteration-count"],
        iterationSource: {
          resourceId: "iteration-count",
          rank: wire("0"),
          mode: "u32-count",
        },
        maxIterationCount: wire("2"),
        body: [{
          nodeId: "reduce-step",
          kind: "all-reduce",
          dependsOn: [],
          resourceId: "output",
          reduction: "sum",
          dtype: "f32",
          numericalPolicy: "rank-order-f32",
          participants: [wire("0"), wire("1")],
          result: "replicated-to-all-participants",
        }],
        mode: "resource-u32-count-sequential",
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

function sequentialConditionalDynamicDispatchProgram(
  artifacts: VerifiedViewCopyArtifacts,
): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 31 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire("1"),
    resources: [
      resource("predicate-input", "input", "u32", "4"),
      resource("predicate", "temporary", "u32", "4"),
      resource("then-count-input", "input", "u32", "4"),
      resource("else-count-input", "input", "u32", "4"),
      resource("launch-count", "temporary", "u32", "4"),
      resource("input", "input", "f32"),
      resource("output", "output", "f32"),
    ],
    nodes: [
      {
        nodeId: "produce-predicate",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "predicate-input",
        destinationResourceId: "predicate",
        mode: "whole-allocation-bytes-per-rank",
      },
      {
        nodeId: "choose-launch-count",
        kind: "conditional",
        dependsOn: ["produce-predicate"],
        predicate: {
          resourceId: "predicate",
          rank: wire("0"),
          mode: "u32-nonzero",
        },
        thenBody: [{
          nodeId: "copy-then-count",
          kind: "copy",
          dependsOn: [],
          sourceResourceId: "then-count-input",
          destinationResourceId: "launch-count",
          mode: "whole-allocation-bytes-per-rank",
        }],
        elseBody: [{
          nodeId: "copy-else-count",
          kind: "copy",
          dependsOn: [],
          sourceResourceId: "else-count-input",
          destinationResourceId: "launch-count",
          mode: "whole-allocation-bytes-per-rank",
        }],
        mode: "resource-u32-branch-sequential",
      },
      {
        nodeId: "copy-selected-prefix",
        kind: "dynamic-dispatch",
        dependsOn: ["choose-launch-count"],
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
        launchSource: {
          resourceId: "launch-count",
          rank: wire("0"),
          mode: "u32-prefix-element-count",
        },
        maxElementCount: wire("2"),
        mode: "resource-u32-prefix-elements",
      },
      {
        nodeId: "materialize-output",
        kind: "materialize",
        dependsOn: ["copy-selected-prefix"],
        resourceId: "output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function sequentialConditionalRectangularDispatchProgram(
  artifacts: VerifiedViewCopyArtifacts,
  shape: readonly number[] = [2, 3],
): HostGraphProgram {
  const byteLength = shape.reduce(
    (product, extent) => product * extent,
    1,
  ) * 4;
  const branchCopies = (
    branch: "then" | "else",
  ) => shape.map((_, axis) => ({
    nodeId: `copy-${branch}-extent-${axis}`,
    kind: "copy" as const,
    dependsOn: axis === 0 ? [] : [`copy-${branch}-extent-${axis - 1}`],
    sourceResourceId: `${branch}-extent-input-${axis}`,
    destinationResourceId: `extent-${axis}`,
    mode: "whole-allocation-bytes-per-rank" as const,
  }));
  return {
    kind: "host-graph",
    version: { major: 1, minor: shape.length === 2 ? 32 : 33 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire("1"),
    resources: [
      resource("predicate-input", "input", "u32", "4"),
      resource("predicate", "temporary", "u32", "4"),
      ...shape.flatMap((_, axis) => [
        resource(`then-extent-input-${axis}`, "input", "u32", "4"),
        resource(`else-extent-input-${axis}`, "input", "u32", "4"),
        resource(`extent-${axis}`, "temporary", "u32", "4"),
      ]),
      resource("input", "input", "f32", String(byteLength)),
      resource("output", "output", "f32", String(byteLength)),
    ],
    nodes: [
      {
        nodeId: "produce-predicate",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "predicate-input",
        destinationResourceId: "predicate",
        mode: "whole-allocation-bytes-per-rank",
      },
      {
        nodeId: "choose-launch-extents",
        kind: "conditional",
        dependsOn: ["produce-predicate"],
        predicate: {
          resourceId: "predicate",
          rank: wire("0"),
          mode: "u32-nonzero",
        },
        thenBody: branchCopies("then"),
        elseBody: branchCopies("else"),
        mode: "resource-u32-branch-sequential",
      },
      {
        nodeId: "copy-selected-rectangle",
        kind: "dynamic-dispatch",
        dependsOn: ["choose-launch-extents"],
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
        launchSources: shape.map((_, axis) => ({
          axis,
          resourceId: `extent-${axis}`,
          rank: wire("0"),
          mode: "u32-prefix-extent" as const,
        })),
        maxExtents: shape.map((extent) => wire(String(extent))),
        mode: "resource-u32-rectangular-prefix",
      },
      {
        nodeId: "materialize-output",
        kind: "materialize",
        dependsOn: ["copy-selected-rectangle"],
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
    rankCount: wire("1"),
    resources: [
      {
        resourceId: "predicate",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "u32",
        byteLength: wire("4"),
        alignmentBytes: 4,
      },
      {
        resourceId: "then-input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "u8",
        byteLength: wire("8"),
        alignmentBytes: 1,
      },
      {
        resourceId: "else-input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "u8",
        byteLength: wire("8"),
        alignmentBytes: 1,
      },
      {
        resourceId: "output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "u8",
        byteLength: wire("8"),
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
      ...base.resources.map((resource) =>
        resource.resourceId === "predicate"
          ? {
              ...resource,
              role: "temporary" as const,
              initialization: "zero-fill" as const,
            }
          : resource),
      {
        resourceId: "predicate-source",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "u32",
        byteLength: wire("4"),
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
): HostGraphCpuInputBinding {
  return {
    rank: wire(String(rank)),
    resourceId: "input",
    bytes,
  };
}

function f32Bytes(values: readonly number[]): Uint8Array {
  const result = new Uint8Array(values.length * 4);
  const view = new DataView(result.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return result;
}

function i32Bytes(values: readonly number[]): Uint8Array {
  const result = new Uint8Array(values.length * 4);
  const view = new DataView(result.buffer);
  values.forEach((value, index) => view.setInt32(index * 4, value, true));
  return result;
}

function u32Bytes(values: readonly number[]): Uint8Array {
  const result = new Uint8Array(values.length * 4);
  const view = new DataView(result.buffer);
  values.forEach((value, index) => view.setUint32(index * 4, value, true));
  return result;
}

function readF32(bytes: Uint8Array): number[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from(
    { length: bytes.byteLength / 4 },
    (_, index) => view.getFloat32(index * 4, true),
  );
}

function rectangularExpected(
  values: readonly number[],
  shape: readonly number[],
  extents: readonly number[],
): number[] {
  return values.map((value, linearIndex) => {
    let remainder = linearIndex;
    for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
      const shapeExtent = shape[axis] as number;
      const coordinate = remainder % shapeExtent;
      remainder = Math.floor(remainder / shapeExtent);
      if (coordinate >= (extents[axis] as number)) return 0;
    }
    return value;
  });
}

function readI32(bytes: Uint8Array): number[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from(
    { length: bytes.byteLength / 4 },
    (_, index) => view.getInt32(index * 4, true),
  );
}

function readU32(bytes: Uint8Array): number[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from(
    { length: bytes.byteLength / 4 },
    (_, index) => view.getUint32(index * 4, true),
  );
}

describe("host graph CPU reference", () => {
  it("selects bounded input conditionals from captured u32 input", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      conditionalRawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });
    const thenBytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const elseBytes = new Uint8Array([8, 7, 6, 5, 4, 3, 2, 1]);
    const predicate = u32Bytes([1]);
    const request = {
      inputs: [
        {
          rank: wire("0"),
          resourceId: "predicate",
          bytes: predicate,
        },
        {
          rank: wire("0"),
          resourceId: "then-input",
          bytes: thenBytes,
        },
        {
          rank: wire("0"),
          resourceId: "else-input",
          bytes: elseBytes,
        },
      ],
    };
    const pending = prepared.execute(request);
    predicate.fill(0);
    thenBytes.fill(0);
    elseBytes.fill(0);
    const thenResult = await pending;

    expect(prepared.conditionalNodeIds).toEqual(["choose-output"]);
    expect(prepared.expandedNodeCount).toBe(2);
    expect(prepared.elementOperations).toBe(8n);
    expect(thenResult.executedNodeIds).toEqual([
      "choose-output",
      "materialize-output",
    ]);
    expect(thenResult.completedConditionals).toEqual([{
      nodeId: "choose-output",
      selectedBranch: "then",
      bodyNodeIds: ["copy-then"],
    }]);
    expect(thenResult.outputs[0]?.bytes).toEqual(
      new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]),
    );

    const elseResult = await prepared.execute({
      inputs: [
        {
          rank: wire("0"),
          resourceId: "predicate",
          bytes: u32Bytes([0]),
        },
        {
          rank: wire("0"),
          resourceId: "then-input",
          bytes: new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]),
        },
        {
          rank: wire("0"),
          resourceId: "else-input",
          bytes: new Uint8Array([8, 7, 6, 5, 4, 3, 2, 1]),
        },
      ],
    });
    expect(elseResult.completedConditionals).toEqual([{
      nodeId: "choose-output",
      selectedBranch: "else",
      bodyNodeIds: ["copy-else"],
    }]);
    expect(elseResult.outputs[0]?.bytes).toEqual(
      new Uint8Array([8, 7, 6, 5, 4, 3, 2, 1]),
    );

    await expect(prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
      maxElementOperations: 7,
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-RESOURCE-LIMIT",
      path: "$.maxElementOperations",
    });
  });

  it("selects runtime conditionals from captured execution controls", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      runtimeConditionalRawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });
    const thenBytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const elseBytes = new Uint8Array([8, 7, 6, 5, 4, 3, 2, 1]);
    const control = { controlId: "choose", value: wire("7") };
    const request = {
      inputs: [
        {
          rank: wire("0"),
          resourceId: "then-input",
          bytes: thenBytes,
        },
        {
          rank: wire("0"),
          resourceId: "else-input",
          bytes: elseBytes,
        },
      ],
      controls: [control],
    };
    const pending = prepared.execute(request);
    control.value = wire("0");
    thenBytes.fill(0);
    elseBytes.fill(0);
    const thenResult = await pending;

    expect(prepared.runtimeControlIds).toEqual(["choose"]);
    expect(thenResult.completedConditionals).toEqual([{
      nodeId: "choose-output",
      selectedBranch: "then",
      bodyNodeIds: ["copy-then"],
    }]);
    expect(thenResult.outputs[0]?.bytes).toEqual(
      new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]),
    );

    const elseResult = await prepared.execute({
      inputs: [
        {
          rank: wire("0"),
          resourceId: "then-input",
          bytes: new Uint8Array(8).fill(1),
        },
        {
          rank: wire("0"),
          resourceId: "else-input",
          bytes: new Uint8Array(8).fill(2),
        },
      ],
      controls: [{ controlId: "choose", value: wire("0") }],
    });
    expect(elseResult.completedConditionals[0]?.selectedBranch).toBe("else");
    expect(elseResult.outputs[0]?.bytes).toEqual(new Uint8Array(8).fill(2));
  });

  it("selects an ordered resource conditional after its producer executes", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      resourceConditionalRawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });
    const inputs = (predicate: number) => [
      {
        rank: wire("0"),
        resourceId: "predicate-source",
        bytes: u32Bytes([predicate]),
      },
      {
        rank: wire("0"),
        resourceId: "then-input",
        bytes: new Uint8Array(8).fill(3),
      },
      {
        rank: wire("0"),
        resourceId: "else-input",
        bytes: new Uint8Array(8).fill(9),
      },
    ];

    const thenResult = await prepared.execute({ inputs: inputs(5) });
    const elseResult = await prepared.execute({ inputs: inputs(0) });

    expect(prepared.resourceConditionalCount).toBe(1);
    expect(prepared.elementOperations).toBe(12n);
    expect(thenResult.completedConditionals).toEqual([{
      nodeId: "choose-output",
      selectedBranch: "then",
      bodyNodeIds: ["copy-then"],
    }]);
    expect(thenResult.outputs[0]?.bytes).toEqual(new Uint8Array(8).fill(3));
    expect(elseResult.completedConditionals[0]?.selectedBranch).toBe("else");
    expect(elseResult.outputs[0]?.bytes).toEqual(new Uint8Array(8).fill(9));
  });

  it("rejects missing, duplicate, unknown, and out-of-range runtime controls", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      runtimeConditionalRawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, {
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
    await expect(prepared.execute({
      inputs: null as unknown as typeof inputs,
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-INVALID-BINDING",
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
      await expect(prepared.execute({
        inputs,
        ...(controls === undefined ? {} : { controls }),
      })).rejects.toMatchObject({
        code: "BG-GRAPH-CPU-INVALID-BINDING",
      });
    }
  });

  it("executes bounded repetition with cancellation points per iteration", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      repeatedCollectiveProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });
    const result = await prepared.execute({
      inputs: [
        input(0, f32Bytes([1, 2])),
        input(1, f32Bytes([3, 4])),
      ],
    });

    expect(prepared.expandedNodeCount).toBe(5);
    expect(prepared.elementOperations).toBe(28n);
    expect(prepared.repeats).toEqual([{
      nodeId: "repeat-reduction",
      iterationCount: "3",
      bodyNodeIds: ["reduce-step"],
    }]);
    expect(result.executedNodeIds).toEqual([
      "initialize",
      "repeat-reduction",
      "materialize-output",
    ]);
    expect(result.completedRepeats).toEqual(prepared.repeats);
    expect(result.outputs.map(({ bytes }) => readF32(bytes))).toEqual([
      [16, 24],
      [16, 24],
    ]);

    await expect(prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
      maxElementOperations: 27,
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-RESOURCE-LIMIT",
      path: "$.maxElementOperations",
    });
  });

  it("executes the exact captured runtime repeat count within its artifact bound", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      runtimeRepeatedCollectiveProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });
    const inputs = [
      input(0, f32Bytes([1, 2])),
      input(1, f32Bytes([3, 4])),
    ];

    expect(prepared.runtimeRepeatCount).toBe(1);
    expect(prepared.repeatNodeIds).toEqual(["repeat-reduction"]);
    expect(prepared.repeats).toEqual([]);
    expect(prepared.elementOperations).toBe(28n);

    const zero = await prepared.execute({
      inputs,
      controls: [{ controlId: "iterations", value: wire("0") }],
    });
    expect(zero.completedRepeats).toEqual([{
      nodeId: "repeat-reduction",
      iterationCount: "0",
      bodyNodeIds: ["reduce-step"],
    }]);
    expect(zero.elementOperations).toBe("16");
    expect(zero.outputs.map(({ bytes }) => readF32(bytes))).toEqual([
      [1, 2],
      [3, 4],
    ]);

    const two = await prepared.execute({
      inputs,
      controls: [{ controlId: "iterations", value: wire("2") }],
    });
    expect(two.completedRepeats[0]?.iterationCount).toBe("2");
    expect(two.elementOperations).toBe("24");
    expect(two.outputs.map(({ bytes }) => readF32(bytes))).toEqual([
      [8, 12],
      [8, 12],
    ]);

    await expect(prepared.execute({
      inputs,
      controls: [{ controlId: "iterations", value: wire("4") }],
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-INVALID-BINDING",
      path: "$.request.controls[0].value",
    });

    let inputReads = 0;
    const unreadInput = {
      rank: wire("0"),
      resourceId: "input",
    } as HostGraphCpuInputBinding;
    Object.defineProperty(unreadInput, "bytes", {
      enumerable: true,
      get() {
        inputReads += 1;
        return f32Bytes([1, 2]);
      },
    });
    await expect(prepared.execute({
      inputs: [unreadInput, input(1, f32Bytes([3, 4]))],
      controls: [{ controlId: "iterations", value: wire("4") }],
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-INVALID-BINDING",
      path: "$.request.controls[0].value",
    });
    expect(inputReads).toBe(0);
  });

  it("executes one produced-resource repeat count within its artifact bound", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      resourceRepeatedCollectiveProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });
    const executionInputs = (count: number): HostGraphCpuInputBinding[] => [
      input(0, f32Bytes([1, 2])),
      input(1, f32Bytes([3, 4])),
      {
        rank: wire("0"),
        resourceId: "iteration-input",
        bytes: u32Bytes([count]),
      },
      {
        rank: wire("1"),
        resourceId: "iteration-input",
        bytes: u32Bytes([0]),
      },
    ];

    expect(prepared).toMatchObject({
      runtimeRepeatCount: 0,
      resourceRepeatCount: 1,
      repeatNodeIds: ["repeat-reduction"],
      repeats: [],
      elementOperations: 36n,
    });

    const zero = await prepared.execute({ inputs: executionInputs(0) });
    expect(zero).toMatchObject({
      completedRepeats: [{
        nodeId: "repeat-reduction",
        iterationCount: "0",
        bodyNodeIds: ["reduce-step"],
      }],
      elementOperations: "24",
    });
    expect(zero.outputs.map(({ bytes }) => readF32(bytes))).toEqual([
      [1, 2],
      [3, 4],
    ]);

    const two = await prepared.execute({ inputs: executionInputs(2) });
    expect(two).toMatchObject({
      completedRepeats: [{
        nodeId: "repeat-reduction",
        iterationCount: "2",
        bodyNodeIds: ["reduce-step"],
      }],
      elementOperations: "32",
    });
    expect(two.outputs.map(({ bytes }) => readF32(bytes))).toEqual([
      [8, 12],
      [8, 12],
    ]);

    await expect(prepared.execute({
      inputs: executionInputs(4),
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-RESOURCE-LIMIT",
      path: "$.nodes.repeat-reduction.iterationSource",
    });
  });

  it("shares one produced u32 across a conditional and repeat", async () => {
    const program = sharedConditionalRepeatFeedbackProgram();
    await expect(createVerifiedHostGraphArtifact(
      {
        ...program,
        version: { major: 1, minor: 28 },
      },
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-UNSUPPORTED-PROFILE" },
    });
    await expect(createVerifiedHostGraphArtifact(
      {
        ...program,
        nodes: program.nodes.map((node) =>
          node.kind === "conditional" &&
            node.mode === "resource-u32-branch-sequential"
            ? {
              ...node,
              predicate: {
                ...node.predicate,
                rank: wire("1"),
              },
            }
            : node),
      },
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-INVALID-BINDING" },
    });
    const graph = (await createVerifiedHostGraphArtifact(
      program,
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });
    const executionInputs = (
      selection: 0 | 2,
    ): HostGraphCpuInputBinding[] => [
      input(0, f32Bytes([1, 2])),
      input(1, f32Bytes([3, 4])),
      {
        rank: wire("0"),
        resourceId: "iteration-input",
        bytes: u32Bytes([selection]),
      },
      {
        rank: wire("1"),
        resourceId: "iteration-input",
        bytes: u32Bytes([0]),
      },
      ...[0, 1].flatMap((rank) => [
        {
          rank: wire(String(rank)),
          resourceId: "then-input",
          bytes: u32Bytes([9 + rank, 19 + rank]),
        },
        {
          rank: wire(String(rank)),
          resourceId: "else-input",
          bytes: u32Bytes([7 + rank, 17 + rank]),
        },
      ]),
    ];

    expect(prepared).toMatchObject({
      resourceRepeatCount: 1,
      resourceConditionalCount: 1,
    });
    for (const selection of [0, 2] as const) {
      const result = await prepared.execute({
        inputs: executionInputs(selection),
      });
      expect(result.completedRepeats).toEqual([{
        nodeId: "repeat-reduction",
        iterationCount: String(selection),
        bodyNodeIds: ["reduce-step"],
      }]);
      expect(result.completedConditionals).toEqual([{
        nodeId: "choose-branch",
        selectedBranch: selection === 0 ? "else" : "then",
        bodyNodeIds: [
          selection === 0 ? "copy-else" : "copy-then",
        ],
      }]);
      const valueOutputs = result.outputs
        .filter(({ resourceId }) => resourceId === "output")
        .map(({ bytes }) => readF32(bytes));
      expect(valueOutputs).toEqual(
        selection === 0
          ? [[1, 2], [3, 4]]
          : [[8, 12], [8, 12]],
      );
      const branchOutputs = result.outputs
        .filter(({ resourceId }) => resourceId === "branch-output")
        .map(({ bytes }) => readU32(bytes));
      expect(branchOutputs).toEqual(
        selection === 0
          ? [[7, 17], [8, 18]]
          : [[9, 19], [10, 20]],
      );
    }
  });

  it("feeds a conditional-produced u32 into a later repeat", async () => {
    const program = sequentialConditionalRepeatFeedbackProgram();
    await expect(createVerifiedHostGraphArtifact(
      {
        ...program,
        version: { major: 1, minor: 29 },
      },
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-UNSUPPORTED-PROFILE" },
    });
    await expect(createVerifiedHostGraphArtifact(
      {
        ...program,
        nodes: program.nodes.map((node) =>
          node.kind === "conditional" &&
            node.mode === "resource-u32-branch-sequential"
            ? {
              ...node,
              elseBody: node.elseBody.map((bodyNode) =>
                bodyNode.kind === "copy"
                  ? { ...bodyNode, destinationResourceId: "predicate" }
                  : bodyNode),
            }
            : node),
      },
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-INVALID-BINDING" },
    });
    const graph = (await createVerifiedHostGraphArtifact(
      program,
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });
    const executionInputs = (
      predicate: 0 | 1,
    ): HostGraphCpuInputBinding[] => [
      input(0, f32Bytes([1, 2])),
      input(1, f32Bytes([3, 4])),
      ...[0, 1].flatMap((rank) => [
        {
          rank: wire(String(rank)),
          resourceId: "predicate-input",
          bytes: u32Bytes([rank === 0 ? predicate : 0]),
        },
        {
          rank: wire(String(rank)),
          resourceId: "then-count-input",
          bytes: u32Bytes([rank === 0 ? 2 : 0]),
        },
        {
          rank: wire(String(rank)),
          resourceId: "else-count-input",
          bytes: u32Bytes([0]),
        },
      ]),
    ];

    expect(prepared).toMatchObject({
      resourceRepeatCount: 1,
      resourceConditionalCount: 1,
    });
    for (const predicate of [0, 1] as const) {
      const result = await prepared.execute({
        inputs: executionInputs(predicate),
      });
      expect(result.completedConditionals).toEqual([{
        nodeId: "choose-iteration-count",
        selectedBranch: predicate === 0 ? "else" : "then",
        bodyNodeIds: [
          predicate === 0 ? "copy-else-count" : "copy-then-count",
        ],
      }]);
      expect(result.completedRepeats).toEqual([{
        nodeId: "repeat-reduction",
        iterationCount: predicate === 0 ? "0" : "2",
        bodyNodeIds: ["reduce-step"],
      }]);
      expect(result.outputs.map(({ bytes }) => readF32(bytes))).toEqual(
        predicate === 0
          ? [[1, 2], [3, 4]]
          : [[8, 12], [8, 12]],
      );
    }
  });

  it("feeds a conditional-produced u32 into a linear dispatch", async () => {
    const artifacts = await identityArtifacts();
    const options = artifactOptions(artifacts);
    const program = sequentialConditionalDynamicDispatchProgram(artifacts);
    await expect(createVerifiedHostGraphArtifact(
      {
        ...program,
        version: { major: 1, minor: 30 },
      },
      options,
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-UNSUPPORTED-PROFILE" },
    });
    await expect(createVerifiedHostGraphArtifact(
      {
        ...program,
        nodes: program.nodes.map((node) =>
          node.kind === "conditional" &&
            node.mode === "resource-u32-branch-sequential"
            ? {
              ...node,
              elseBody: node.elseBody.map((bodyNode) =>
                bodyNode.kind === "copy"
                  ? { ...bodyNode, destinationResourceId: "predicate" }
                  : bodyNode),
            }
            : node),
      },
      options,
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-INVALID-BINDING" },
    });
    await expect(createVerifiedHostGraphArtifact(
      {
        ...program,
        nodes: program.nodes.map((node) => {
          if (
            node.kind === "conditional" &&
            node.mode === "resource-u32-branch-sequential"
          ) {
            return {
              ...node,
              thenBody: node.thenBody.map((bodyNode) =>
                bodyNode.kind === "copy"
                  ? { ...bodyNode, destinationResourceId: "predicate" }
                  : bodyNode),
              elseBody: node.elseBody.map((bodyNode) =>
                bodyNode.kind === "copy"
                  ? { ...bodyNode, destinationResourceId: "predicate" }
                  : bodyNode),
            };
          }
          return node.kind === "dynamic-dispatch" &&
              node.mode === "resource-u32-prefix-elements"
            ? {
              ...node,
              launchSource: {
                ...node.launchSource,
                resourceId: "predicate",
              },
            }
            : node;
        }),
      },
      options,
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-INVALID-BINDING" },
    });
    const graph = (await createVerifiedHostGraphArtifact(
      program,
      options,
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, options);
    const executionInputs = (
      predicate: 0 | 1,
    ): HostGraphCpuInputBinding[] => [
      {
        rank: wire("0"),
        resourceId: "predicate-input",
        bytes: u32Bytes([predicate]),
      },
      {
        rank: wire("0"),
        resourceId: "then-count-input",
        bytes: u32Bytes([2]),
      },
      {
        rank: wire("0"),
        resourceId: "else-count-input",
        bytes: u32Bytes([1]),
      },
      input(0, f32Bytes([5, 6])),
    ];

    expect(prepared).toMatchObject({
      resourceConditionalCount: 1,
      resourceDynamicDispatchCount: 1,
    });
    for (const predicate of [0, 1] as const) {
      const result = await prepared.execute({
        inputs: executionInputs(predicate),
      });
      expect(result.completedConditionals).toEqual([{
        nodeId: "choose-launch-count",
        selectedBranch: predicate === 0 ? "else" : "then",
        bodyNodeIds: [
          predicate === 0 ? "copy-else-count" : "copy-then-count",
        ],
      }]);
      expect(result.completedDynamicDispatches).toEqual([{
        nodeId: "copy-selected-prefix",
        elementCount: predicate === 0 ? "1" : "2",
      }]);
      expect(result.outputs.map(({ bytes }) => readF32(bytes))).toEqual([
        predicate === 0 ? [5, 0] : [5, 6],
      ]);
    }
  });

  it("feeds conditional-produced extents into a rectangular dispatch", async () => {
    const shape = [2, 3] as const;
    const artifacts = await rectangularArtifacts(shape);
    const options = artifactOptions(artifacts);
    const program =
      sequentialConditionalRectangularDispatchProgram(artifacts);
    await expect(createVerifiedHostGraphArtifact(
      {
        ...program,
        version: { major: 1, minor: 31 },
      },
      options,
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-UNSUPPORTED-PROFILE" },
    });
    const rankThreeShape = [2, 2, 3] as const;
    const rankThreeArtifacts = await rectangularArtifacts(rankThreeShape);
    await expect(createVerifiedHostGraphArtifact(
      {
        ...sequentialConditionalRectangularDispatchProgram(
          rankThreeArtifacts,
          rankThreeShape,
        ),
        version: { major: 1, minor: 32 },
      },
      artifactOptions(rankThreeArtifacts),
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-UNSUPPORTED-PROFILE" },
    });
    await expect(createVerifiedHostGraphArtifact(
      {
        ...program,
        nodes: program.nodes.map((node) =>
          node.kind === "conditional" &&
            node.mode === "resource-u32-branch-sequential"
            ? {
              ...node,
              elseBody: node.elseBody.map((bodyNode) =>
                bodyNode.nodeId === "copy-else-extent-1" &&
                  bodyNode.kind === "copy"
                  ? { ...bodyNode, destinationResourceId: "extent-0" }
                  : bodyNode),
            }
            : node),
      },
      options,
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-INVALID-BINDING" },
    });
    await expect(createVerifiedHostGraphArtifact(
      {
        ...program,
        nodes: program.nodes.map((node) =>
          node.kind === "dynamic-dispatch" &&
            node.mode === "resource-u32-rectangular-prefix"
            ? {
              ...node,
              launchSources: node.launchSources.map((source) =>
                source.axis === 0
                  ? { ...source, resourceId: "predicate" }
                  : source),
            }
            : node),
      },
      options,
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-INVALID-BINDING" },
    });
    const graph = (await createVerifiedHostGraphArtifact(
      program,
      options,
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, options);
    const values = [1, 2, 3, 4, 5, 6];
    const executionInputs = (
      predicate: 0 | 1,
      thenExtents: readonly [number, number] = [2, 3],
    ): HostGraphCpuInputBinding[] => [
      {
        rank: wire("0"),
        resourceId: "predicate-input",
        bytes: u32Bytes([predicate]),
      },
      ...thenExtents.map((extent, axis) => ({
        rank: wire("0"),
        resourceId: `then-extent-input-${axis}`,
        bytes: u32Bytes([extent]),
      })),
      ...[1, 2].map((extent, axis) => ({
        rank: wire("0"),
        resourceId: `else-extent-input-${axis}`,
        bytes: u32Bytes([extent]),
      })),
      input(0, f32Bytes(values)),
    ];

    expect(prepared).toMatchObject({
      resourceConditionalCount: 1,
      resourceDynamicDispatchCount: 1,
    });
    for (const predicate of [0, 1] as const) {
      const extents = predicate === 0 ? [1, 2] as const : shape;
      const branch = predicate === 0 ? "else" : "then";
      const result = await prepared.execute({
        inputs: executionInputs(predicate),
      });
      expect(result.completedConditionals).toEqual([{
        nodeId: "choose-launch-extents",
        selectedBranch: branch,
        bodyNodeIds: [
          `copy-${branch}-extent-0`,
          `copy-${branch}-extent-1`,
        ],
      }]);
      expect(result.completedDynamicDispatches).toEqual([{
        nodeId: "copy-selected-rectangle",
        logicalExtents: extents.map(String),
        elementCount: String(extents[0] * extents[1]),
      }]);
      expect(readF32(result.outputs[0]!.bytes)).toEqual(
        rectangularExpected(values, shape, extents),
      );
    }
    await expect(prepared.execute({
      inputs: executionInputs(1, [3, 3]),
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-RESOURCE-LIMIT",
      path: "$.nodes.copy-selected-rectangle.launchSources[0]",
    });
  });

  it("feeds conditional-produced rank-8 extents into one generic dispatch", async () => {
    const shape = [2, 2, 2, 2, 2, 2, 3, 4] as const;
    const smallExtents = [1, 2, 1, 2, 1, 2, 2, 3] as const;
    const artifacts = await rectangularArtifacts(shape);
    const options = artifactOptions(artifacts);
    const program = sequentialConditionalRectangularDispatchProgram(
      artifacts,
      shape,
    );
    const graph = (await createVerifiedHostGraphArtifact(
      program,
      options,
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, options);
    const elementCount = shape.reduce<number>(
      (product, extent) => product * extent,
      1,
    );
    const values = Array.from(
      { length: elementCount },
      (_, index) => index + 0.25,
    );
    const executionInputs = (
      predicate: 0 | 1,
    ): HostGraphCpuInputBinding[] => [
      {
        rank: wire("0"),
        resourceId: "predicate-input",
        bytes: u32Bytes([predicate]),
      },
      ...shape.map((extent, axis) => ({
        rank: wire("0"),
        resourceId: `then-extent-input-${axis}`,
        bytes: u32Bytes([extent]),
      })),
      ...smallExtents.map((extent, axis) => ({
        rank: wire("0"),
        resourceId: `else-extent-input-${axis}`,
        bytes: u32Bytes([extent]),
      })),
      input(0, f32Bytes(values)),
    ];

    for (const predicate of [0, 1] as const) {
      const extents = predicate === 0 ? smallExtents : shape;
      const branch = predicate === 0 ? "else" : "then";
      const result = await prepared.execute({
        inputs: executionInputs(predicate),
      });
      expect(result.completedConditionals).toEqual([{
        nodeId: "choose-launch-extents",
        selectedBranch: branch,
        bodyNodeIds: shape.map((_, axis) =>
          `copy-${branch}-extent-${axis}`),
      }]);
      expect(result.completedDynamicDispatches).toEqual([{
        nodeId: "copy-selected-rectangle",
        logicalExtents: extents.map(String),
        elementCount: String(extents.reduce<number>(
          (product, extent) => product * extent,
          1,
        )),
      }]);
      expect(readF32(result.outputs[0]!.bytes)).toEqual(
        rectangularExpected(values, shape, extents),
      );
    }
  });

  it("reports completion events only with a successful whole-graph result", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      eventfulRawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });
    const source = new Uint8Array([0, 1, 2, 3, 4, 5, 255]);
    const result = await prepared.execute({
      inputs: [input(0, source), input(1, source)],
    });

    expect(prepared.eventIds).toEqual(["copy-complete"]);
    expect(prepared.elementOperations).toBe(14n);
    expect(result.executedNodeIds).toEqual([
      "raw-copy",
      "copy-complete-event",
      "materialize-output",
    ]);
    expect(result.completedEventIds).toEqual(["copy-complete"]);
    expect(result.outputs.map(({ bytes }) => bytes)).toEqual([
      source,
      source,
    ]);
  });

  it("publishes only explicitly materialized version-1.2 outputs", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      materializedRawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });
    const source = new Uint8Array([0, 1, 2, 3, 4, 5, 255]);
    const result = await prepared.execute({
      inputs: [input(0, source), input(1, source)],
    });

    expect(prepared.outputResourceIds).toEqual(["output"]);
    expect(prepared.elementOperations).toBe(14n);
    expect(result.executedNodeIds).toEqual([
      "raw-copy",
      "materialize-output",
    ]);
    expect(result.completedEventIds).toEqual([]);
    expect(result.outputs.map(({ bytes }) => bytes)).toEqual([
      source,
      source,
    ]);
  });

  it("copies complete odd-sized allocations per rank without semantic artifacts", async () => {
    const graph = (await createVerifiedHostGraphArtifact(
      rawCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    )).artifact;
    const prepared = await prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
    });
    const first = new Uint8Array([0, 1, 2, 3, 4, 5, 255]);
    const second = new Uint8Array([255, 5, 4, 3, 2, 1, 0]);
    const expected = [new Uint8Array(first), new Uint8Array(second)];
    const pending = prepared.execute({
      inputs: [input(0, first), input(1, second)],
    });
    first.fill(9);
    second.fill(9);
    const result = await pending;

    expect(prepared.elementOperations).toBe(14n);
    expect(result.executedNodeIds).toEqual(["raw-copy"]);
    expect(result.outputs.map(({ bytes }) => bytes)).toEqual(expected);

    await expect(prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [],
      maxElementOperations: 13,
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-RESOURCE-LIMIT",
      path: "$.maxElementOperations",
    });
  });

  it("executes an authority-bound multi-dispatch pipeline", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts, "f32"), artifacts);
    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    const source = f32Bytes([1.25, -2.5]);
    const result = await prepared.execute({ inputs: [input(0, source)] });

    expect(prepared).toMatchObject({
      profile: HOST_GRAPH_CPU_PROFILE,
      rankCount: 1n,
      inputResourceIds: ["input"],
      outputResourceIds: ["output"],
      elementOperations: 4n,
    });
    expect(result).toMatchObject({
      profile: HOST_GRAPH_CPU_PROFILE,
      failureModel: "fail-stop-no-partial-output-commit",
      executedNodeIds: ["first", "second"],
      elementOperations: "4",
    });
    expect(readF32(result.outputs[0]!.bytes)).toEqual([1.25, -2.5]);
  });

  it("executes and reports an admitted dynamic dispatch prefix", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(
      dynamicPipelineProgram(artifacts),
      artifacts,
    );
    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    const source = f32Bytes([1.25, -2.5]);
    const result = await prepared.execute({
      inputs: [input(0, source)],
      controls: [{
        controlId: "prefix-elements",
        value: wire("1"),
      }],
    });

    expect(prepared).toMatchObject({
      dynamicDispatchCount: 1,
      runtimeControlIds: ["prefix-elements"],
      elementOperations: 4n,
    });
    expect(result).toMatchObject({
      executedNodeIds: ["first", "second", "materialize-output"],
      completedDynamicDispatches: [{
        nodeId: "first",
        elementCount: "1",
      }],
      elementOperations: "3",
    });
    expect(readF32(result.outputs[0]!.bytes)).toEqual([1.25, 0]);
  });

  it("executes and reports exact rank-2 through rank-8 request-time rectangular prefixes", async () => {
    const cases = [
      {
        shape: [3, 4],
        extents: [2, 3],
      },
      {
        shape: [2, 3, 4],
        extents: [2, 2, 3],
      },
      {
        shape: [2, 2, 3, 4],
        extents: [1, 2, 2, 3],
      },
      {
        shape: [2, 2, 2, 3, 4],
        extents: [1, 2, 1, 2, 3],
      },
      {
        shape: [2, 2, 2, 2, 3, 4],
        extents: [1, 2, 1, 2, 2, 3],
      },
      {
        shape: [2, 2, 2, 2, 2, 3, 4],
        extents: [1, 2, 1, 2, 1, 2, 3],
      },
      {
        shape: [2, 2, 2, 2, 2, 2, 3, 4],
        extents: [1, 2, 1, 2, 1, 2, 2, 3],
      },
    ] as const;
    for (const testCase of cases) {
      const artifacts = await rectangularArtifacts(testCase.shape);
      const graph = await verified(
        rectangularDynamicProgram(artifacts, testCase.shape),
        artifacts,
      );
      const prepared = await prepareHostGraphCpu(
        graph,
        artifactOptions(artifacts),
      );
      const elementCount = testCase.shape.reduce(
        (product, extent) => product * extent,
        1,
      );
      const selectedElementCount = testCase.extents.reduce<number>(
        (product, extent) => product * extent,
        1,
      );
      const values = Array.from(
        { length: elementCount },
        (_, index) => index + 0.25,
      );
      const result = await prepared.execute({
        inputs: [input(0, f32Bytes(values))],
        controls: testCase.extents.map((extent, axis) => ({
          controlId: `prefix-axis-${axis}`,
          value: wire(String(extent)),
        })),
      });

      expect(prepared).toMatchObject({
        dynamicDispatchCount: 1,
        runtimeControlIds: testCase.shape.map(
          (_, axis) => `prefix-axis-${axis}`,
        ),
        elementOperations: BigInt(elementCount),
      });
      expect(result).toMatchObject({
        executedNodeIds: ["copy-region", "materialize-output"],
        completedDynamicDispatches: [{
          nodeId: "copy-region",
          logicalExtents: testCase.extents.map(String),
          elementCount: String(selectedElementCount),
        }],
        elementOperations: String(selectedElementCount),
      });
      expect(readF32(result.outputs[0]!.bytes)).toEqual(
        rectangularExpected(values, testCase.shape, testCase.extents),
      );
    }
  });

  it("keeps rank 8 request and produced-resource versioning separate", async () => {
    const shape = [2, 2, 2, 2, 2, 2, 3, 4] as const;
    const artifacts = await rectangularArtifacts(shape);
    const current = rectangularDynamicProgram(artifacts, shape);
    const legacy = {
      ...current,
      version: { major: 1 as const, minor: 21 as const },
    };

    await expect(createVerifiedHostGraphArtifact(
      legacy,
      artifactOptions(artifacts),
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-UNSUPPORTED-PROFILE" },
    });

    const resource = {
      ...resourceRectangularDynamicProgram(artifacts, shape),
      version: { major: 1 as const, minor: 22 as const },
    };
    await expect(createVerifiedHostGraphArtifact(
      resource,
      artifactOptions(artifacts),
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-UNSUPPORTED-PROFILE" },
    });
  });

  it("rejects rectangular extents before reading caller inputs", async () => {
    const shape = [3, 4] as const;
    const artifacts = await rectangularArtifacts(shape);
    const graph = await verified(
      rectangularDynamicProgram(artifacts, shape),
      artifacts,
    );
    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    let inputReads = 0;
    const unreadInput = {
      rank: wire("0"),
      resourceId: "input",
    } as HostGraphCpuInputBinding;
    Object.defineProperty(unreadInput, "bytes", {
      enumerable: true,
      get() {
        inputReads += 1;
        return f32Bytes(Array.from({ length: 12 }, (_, index) => index));
      },
    });

    for (const [axis, value] of [[0, "0"], [1, "5"]] as const) {
      const controls = [
        { controlId: "prefix-axis-0", value: wire("2") },
        { controlId: "prefix-axis-1", value: wire("3") },
      ];
      controls[axis] = {
        controlId: `prefix-axis-${axis}`,
        value: wire(value),
      };
      await expect(prepared.execute({
        inputs: [unreadInput],
        controls,
      })).rejects.toMatchObject({
        code: "BG-GRAPH-CPU-INVALID-BINDING",
        path: `$.request.controls[${axis}].value`,
      });
    }
    expect(inputReads).toBe(0);
  });

  it("executes and reports produced rank-2 through rank-8 rectangular prefixes", async () => {
    const cases = [
      { shape: [3, 4], extents: [2, 3] },
      { shape: [2, 3, 4], extents: [1, 2, 3] },
      { shape: [2, 2, 3, 4], extents: [1, 2, 2, 3] },
      { shape: [2, 2, 2, 3, 4], extents: [1, 2, 1, 2, 3] },
      { shape: [2, 2, 2, 2, 3, 4], extents: [1, 2, 1, 2, 2, 3] },
      {
        shape: [2, 2, 2, 2, 2, 3, 4],
        extents: [1, 2, 1, 2, 1, 2, 3],
      },
      {
        shape: [2, 2, 2, 2, 2, 2, 3, 4],
        extents: [1, 2, 1, 2, 1, 2, 2, 3],
      },
    ] as const;
    for (const testCase of cases) {
      const artifacts = await rectangularArtifacts(testCase.shape);
      const graph = await verified(
        resourceRectangularDynamicProgram(artifacts, testCase.shape),
        artifacts,
      );
      const prepared = await prepareHostGraphCpu(
        graph,
        artifactOptions(artifacts),
      );
      const elementCount = testCase.shape.reduce(
        (product, extent) => product * extent,
        1,
      );
      const selectedElementCount = testCase.extents.reduce<number>(
        (product, extent) => product * extent,
        1,
      );
      const values = Array.from(
        { length: elementCount },
        (_, index) => index + 0.5,
      );
      const result = await prepared.execute({
        inputs: [
          input(0, f32Bytes(values)),
          ...testCase.extents.map((extent, axis) => ({
            rank: wire("0"),
            resourceId: `extent-input-${axis}`,
            bytes: u32Bytes([extent]),
          })),
        ],
      });

      expect(prepared).toMatchObject({
        dynamicDispatchCount: 1,
        resourceDynamicDispatchCount: 1,
        runtimeControlIds: [],
      });
      expect(result).toMatchObject({
        completedDynamicDispatches: [{
          nodeId: "copy-region",
          logicalExtents: testCase.extents.map(String),
          elementCount: String(selectedElementCount),
        }],
      });
      expect(readF32(result.outputs[0]!.bytes)).toEqual(
        rectangularExpected(values, testCase.shape, testCase.extents),
      );
    }

    const shape = [2, 2, 2, 2, 2, 2, 3, 4] as const;
    const extents = [1, 2, 1, 2, 1, 2, 2, 3] as const;
    const artifacts = await rectangularArtifacts(shape);
    const fanoutProgram = resourceRectangularDynamicFanoutProgram(
      artifacts,
      shape,
    );
    await expect(createVerifiedHostGraphArtifact(
      {
        ...fanoutProgram,
        version: { major: 1, minor: 24 },
      },
      artifactOptions(artifacts),
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-UNSUPPORTED-PROFILE" },
    });
    const mismatchedMaximum = {
      ...fanoutProgram,
      nodes: fanoutProgram.nodes.map((node) =>
        node.kind === "dynamic-dispatch" &&
          node.mode === "resource-u32-rectangular-prefix" &&
          node.nodeId === "fanout-region"
          ? {
            ...node,
            maxExtents: node.maxExtents.map((extent, axis) =>
              axis === 0 ? wire("1") : extent),
          }
          : node),
    } satisfies HostGraphProgram;
    await expect(createVerifiedHostGraphArtifact(
      mismatchedMaximum,
      artifactOptions(artifacts),
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-INVALID-BINDING" },
    });
    const mismatchedSources = {
      ...fanoutProgram,
      nodes: fanoutProgram.nodes.map((node) =>
        node.kind === "dynamic-dispatch" &&
          node.mode === "resource-u32-rectangular-prefix" &&
          node.nodeId === "fanout-region"
          ? {
            ...node,
            launchSources: node.launchSources.map((source, axis) =>
              axis < 2
                ? {
                  ...source,
                  resourceId: `extent-${1 - axis}`,
                }
                : source),
          }
          : node),
    } satisfies HostGraphProgram;
    await expect(createVerifiedHostGraphArtifact(
      mismatchedSources,
      artifactOptions(artifacts),
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-INVALID-BINDING" },
    });
    const fanout = await prepareHostGraphCpu(
      await verified(fanoutProgram, artifacts),
      artifactOptions(artifacts),
    );
    const elementCount = shape.reduce(
      (product, extent) => product * extent,
      1,
    );
    const values = Array.from(
      { length: elementCount },
      (_, index) => index + 0.5,
    );
    const fanoutResult = await fanout.execute({
      inputs: [
        input(0, f32Bytes(values)),
        ...extents.map((extent, axis) => ({
          rank: wire("0"),
          resourceId: `extent-input-${axis}`,
          bytes: u32Bytes([extent]),
        })),
      ],
    });
    expect(fanout).toMatchObject({
      dynamicDispatchCount: 2,
      resourceDynamicDispatchCount: 2,
    });
    expect(fanoutResult.completedDynamicDispatches).toHaveLength(2);
    expect(fanoutResult.outputs.map(({ bytes }) => readF32(bytes))).toEqual([
      rectangularExpected(values, shape, extents),
      rectangularExpected(values, shape, extents),
    ]);
  });

  it("rejects produced rectangular extents before output publication", async () => {
    const shape = [3, 4] as const;
    const artifacts = await rectangularArtifacts(shape);
    const graph = await verified(
      resourceRectangularDynamicProgram(artifacts, shape),
      artifacts,
    );
    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    const executionInputs = (
      extents: readonly number[],
    ): HostGraphCpuInputBinding[] => [
      input(0, f32Bytes(Array.from({ length: 12 }, (_, index) => index))),
      ...extents.map((extent, axis) => ({
        rank: wire("0"),
        resourceId: `extent-input-${axis}`,
        bytes: u32Bytes([extent]),
      })),
    ];

    for (const [axis, extents] of [
      [0, [0, 3]],
      [1, [2, 5]],
    ] as const) {
      await expect(prepared.execute({
        inputs: executionInputs(extents),
      })).rejects.toMatchObject({
        code: "BG-GRAPH-CPU-RESOURCE-LIMIT",
        path: `$.nodes.copy-region.launchSources[${axis}]`,
      });
    }
  });

  it("executes produced-resource dynamic prefix, shared fanout, and sequential feedback", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(
      resourceDynamicPipelineProgram(artifacts),
      artifacts,
    );
    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    const executionInputs = (
      elementCount: number,
    ): HostGraphCpuInputBinding[] => [
      input(0, f32Bytes([1.25, -2.5])),
      {
        rank: wire("0"),
        resourceId: "launch-input",
        bytes: u32Bytes([elementCount]),
      },
    ];

    expect(prepared).toMatchObject({
      dynamicDispatchCount: 1,
      resourceDynamicDispatchCount: 1,
      runtimeControlIds: [],
      elementOperations: 8n,
    });

    const one = await prepared.execute({ inputs: executionInputs(1) });
    expect(one).toMatchObject({
      executedNodeIds: [
        "produce-launch-count",
        "first",
        "second",
        "materialize-output",
      ],
      completedDynamicDispatches: [{
        nodeId: "first",
        elementCount: "1",
      }],
      elementOperations: "7",
    });
    expect(readF32(one.outputs[0]!.bytes)).toEqual([1.25, 0]);

    const two = await prepared.execute({ inputs: executionInputs(2) });
    expect(two).toMatchObject({
      completedDynamicDispatches: [{
        nodeId: "first",
        elementCount: "2",
      }],
      elementOperations: "8",
    });
    expect(readF32(two.outputs[0]!.bytes)).toEqual([1.25, -2.5]);

    for (const elementCount of [0, 3]) {
      await expect(prepared.execute({
        inputs: executionInputs(elementCount),
      })).rejects.toMatchObject({
        code: "BG-GRAPH-CPU-RESOURCE-LIMIT",
        path: "$.nodes.first.launchSource",
      });
    }

    const fanoutProgram = resourceDynamicFanoutProgram(artifacts);
    await expect(createVerifiedHostGraphArtifact(
      {
        ...fanoutProgram,
        version: { major: 1, minor: 23 },
      },
      artifactOptions(artifacts),
    )).rejects.toThrow("resource feedback node count exceeds 1");
    await expect(createVerifiedHostGraphArtifact(
      {
        ...fanoutProgram,
        nodes: fanoutProgram.nodes.map((node) =>
          node.kind === "dynamic-dispatch" && node.nodeId === "fanout"
            ? { ...node, maxElementCount: wire("1") }
            : node),
      },
      artifactOptions(artifacts),
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-INVALID-BINDING" },
    });
    const fanout = await prepareHostGraphCpu(
      await verified(fanoutProgram, artifacts),
      artifactOptions(artifacts),
    );
    expect(fanout).toMatchObject({
      dynamicDispatchCount: 2,
      resourceDynamicDispatchCount: 2,
      runtimeControlIds: [],
    });
    const fanoutResult = await fanout.execute({
      inputs: executionInputs(1),
    });
    expect(fanoutResult.completedDynamicDispatches).toEqual([
      { nodeId: "fanout", elementCount: "1" },
      { nodeId: "first", elementCount: "1" },
    ]);
    expect(fanoutResult.outputs.map(({ bytes }) => readF32(bytes))).toEqual([
      [1.25, 0],
      [1.25, 0],
    ]);

    const countArtifacts =
      await createVerifiedDensePermutationViewCopyArtifacts({
        inputShape: [parseWireI64("1")],
        axes: [0],
        dtype: "u32",
      });
    const sequentialOptions = {
      kernelArtifacts: [countArtifacts.kernel, artifacts.kernel],
      layoutArtifacts: [countArtifacts.layout, artifacts.layout],
    };
    const sequentialProgram = sequentialResourceDynamicProgram(
      countArtifacts,
      artifacts,
    );
    await expect(createVerifiedHostGraphArtifact(
      {
        ...sequentialProgram,
        version: { major: 1, minor: 25 },
      },
      sequentialOptions,
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-INVALID-BINDING" },
    });
    await expect(createVerifiedHostGraphArtifact(
      {
        ...sequentialProgram,
        resources: [
          ...sequentialProgram.resources,
          {
            resourceId: "orphan-count",
            role: "temporary",
            multiplicity: "per-rank",
            initialization: "zero-fill",
            dtype: "u32",
            byteLength: wire("4"),
            alignmentBytes: 4,
          },
        ],
        nodes: sequentialProgram.nodes.map((node) =>
          node.kind === "dynamic-dispatch" &&
            node.nodeId === "copy-selected-prefix" &&
            node.mode === "resource-u32-prefix-elements"
            ? {
              ...node,
              launchSource: {
                ...node.launchSource,
                resourceId: "orphan-count",
              },
            }
            : node),
      },
      sequentialOptions,
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-INVALID-BINDING" },
    });
    const sequentialArtifact = (await createVerifiedHostGraphArtifact(
      sequentialProgram,
      sequentialOptions,
    )).artifact;
    const sequential = await prepareHostGraphCpu(
      sequentialArtifact,
      sequentialOptions,
    );
    expect(sequential).toMatchObject({
      dynamicDispatchCount: 2,
      resourceDynamicDispatchCount: 2,
    });
    for (const selectedCount of [1, 2]) {
      const result = await sequential.execute({
        inputs: [
          {
            rank: wire("0"),
            resourceId: "launch-input",
            bytes: u32Bytes([1]),
          },
          {
            rank: wire("0"),
            resourceId: "next-count-input",
            bytes: u32Bytes([selectedCount]),
          },
          input(0, f32Bytes([1.25, -2.5])),
        ],
      });
      expect(result.completedDynamicDispatches).toEqual([
        { nodeId: "produce-next-count", elementCount: "1" },
        {
          nodeId: "copy-selected-prefix",
          elementCount: String(selectedCount),
        },
      ]);
      expect(readF32(result.outputs[0]!.bytes)).toEqual(
        selectedCount === 1 ? [1.25, 0] : [1.25, -2.5],
      );
    }

    const threeStageProgram = sequentialResourceDynamicProgram(
      countArtifacts,
      artifacts,
      3,
    );
    await expect(createVerifiedHostGraphArtifact(
      {
        ...threeStageProgram,
        version: { major: 1, minor: 26 },
      },
      sequentialOptions,
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-RESOURCE-LIMIT" },
    });
    await expect(createVerifiedHostGraphArtifact(
      {
        ...threeStageProgram,
        nodes: threeStageProgram.nodes.map((node) =>
          node.kind === "dynamic-dispatch" &&
            node.nodeId === "copy-selected-prefix" &&
            node.mode === "resource-u32-prefix-elements"
            ? {
              ...node,
              launchSource: {
                ...node.launchSource,
                resourceId: "next-count",
              },
            }
            : node),
      },
      sequentialOptions,
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-INVALID-BINDING" },
    });
    const threeStage = await prepareHostGraphCpu(
      (await createVerifiedHostGraphArtifact(
        threeStageProgram,
        sequentialOptions,
      )).artifact,
      sequentialOptions,
    );
    expect(threeStage).toMatchObject({
      dynamicDispatchCount: 3,
      resourceDynamicDispatchCount: 3,
    });
    for (const selectedCount of [1, 2]) {
      const result = await threeStage.execute({
        inputs: [
          {
            rank: wire("0"),
            resourceId: "launch-input",
            bytes: u32Bytes([1]),
          },
          {
            rank: wire("0"),
            resourceId: "next-count-input",
            bytes: u32Bytes([1]),
          },
          {
            rank: wire("0"),
            resourceId: "final-count-input",
            bytes: u32Bytes([selectedCount]),
          },
          input(0, f32Bytes([1.25, -2.5])),
        ],
      });
      expect(result.completedDynamicDispatches).toEqual([
        { nodeId: "produce-next-count", elementCount: "1" },
        { nodeId: "produce-final-count", elementCount: "1" },
        {
          nodeId: "copy-selected-prefix",
          elementCount: String(selectedCount),
        },
      ]);
      expect(readF32(result.outputs[0]!.bytes)).toEqual(
        selectedCount === 1 ? [1.25, 0] : [1.25, -2.5],
      );
    }

    const fourStageProgram = sequentialResourceDynamicProgram(
      countArtifacts,
      artifacts,
      4,
    );
    await expect(createVerifiedHostGraphArtifact(
      {
        ...fourStageProgram,
        version: { major: 1, minor: 27 },
      },
      sequentialOptions,
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-RESOURCE-LIMIT" },
    });
    await expect(createVerifiedHostGraphArtifact(
      {
        ...fourStageProgram,
        nodes: fourStageProgram.nodes.map((node) => {
          if (
            node.kind !== "dynamic-dispatch" ||
            node.mode !== "resource-u32-prefix-elements" ||
            (
              node.nodeId !== "produce-next-count" &&
              node.nodeId !== "produce-terminal-count"
            )
          ) {
            return node;
          }
          return {
            ...node,
            bindings: node.bindings.map((binding) =>
              binding.semanticResourceId ===
                  countArtifacts.destination.viewId
                ? {
                    ...binding,
                    graphResourceId:
                      node.nodeId === "produce-next-count"
                        ? "terminal-count"
                        : "next-count",
                  }
                : binding),
          };
        }),
      },
      sequentialOptions,
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-INVALID-BINDING" },
    });
    const fourStage = await prepareHostGraphCpu(
      (await createVerifiedHostGraphArtifact(
        fourStageProgram,
        sequentialOptions,
      )).artifact,
      sequentialOptions,
    );
    expect(fourStage).toMatchObject({
      dynamicDispatchCount: 4,
      resourceDynamicDispatchCount: 4,
    });
    for (const selectedCount of [1, 2]) {
      const result = await fourStage.execute({
        inputs: [
          {
            rank: wire("0"),
            resourceId: "launch-input",
            bytes: u32Bytes([1]),
          },
          {
            rank: wire("0"),
            resourceId: "next-count-input",
            bytes: u32Bytes([1]),
          },
          {
            rank: wire("0"),
            resourceId: "final-count-input",
            bytes: u32Bytes([1]),
          },
          {
            rank: wire("0"),
            resourceId: "terminal-count-input",
            bytes: u32Bytes([selectedCount]),
          },
          input(0, f32Bytes([1.25, -2.5])),
        ],
      });
      expect(result.completedDynamicDispatches).toEqual([
        { nodeId: "produce-next-count", elementCount: "1" },
        { nodeId: "produce-final-count", elementCount: "1" },
        { nodeId: "produce-terminal-count", elementCount: "1" },
        {
          nodeId: "copy-selected-prefix",
          elementCount: String(selectedCount),
        },
      ]);
      expect(readF32(result.outputs[0]!.bytes)).toEqual(
        selectedCount === 1 ? [1.25, 0] : [1.25, -2.5],
      );
    }

    const eightStageProgram = sequentialResourceDynamicProgram(
      countArtifacts,
      artifacts,
      8,
    );
    await expect(createVerifiedHostGraphArtifact(
      {
        ...eightStageProgram,
        version: { major: 1, minor: 33 },
      },
      sequentialOptions,
    )).rejects.toMatchObject({
      diagnostic: { code: "BG-GRAPH-RESOURCE-LIMIT" },
    });
    const eightStage = await prepareHostGraphCpu(
      (await createVerifiedHostGraphArtifact(
        eightStageProgram,
        sequentialOptions,
      )).artifact,
      sequentialOptions,
    );
    expect(eightStage).toMatchObject({
      dynamicDispatchCount: 8,
      resourceDynamicDispatchCount: 8,
    });
    const eightStageSources = sequentialFeedbackStages(8);
    for (const selectedCount of [1, 2]) {
      const result = await eightStage.execute({
        inputs: [
          {
            rank: wire("0"),
            resourceId: "launch-input",
            bytes: u32Bytes([1]),
          },
          ...eightStageSources.map((stage, index) => ({
            rank: wire("0"),
            resourceId: stage.inputResourceId,
            bytes: u32Bytes([
              index === eightStageSources.length - 1
                ? selectedCount
                : 1,
            ]),
          })),
          input(0, f32Bytes([1.25, -2.5])),
        ],
      });
      expect(result.completedDynamicDispatches).toEqual([
        ...eightStageSources.map((stage) => ({
          nodeId: stage.nodeId,
          elementCount: "1",
        })),
        {
          nodeId: "copy-selected-prefix",
          elementCount: String(selectedCount),
        },
      ]);
      expect(readF32(result.outputs[0]!.bytes)).toEqual(
        selectedCount === 1 ? [1.25, 0] : [1.25, -2.5],
      );
    }
  });

  it("rejects dynamic dispatch values before reading caller inputs", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(
      dynamicPipelineProgram(artifacts),
      artifacts,
    );
    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    let inputReads = 0;
    const unreadInput = {
      rank: wire("0"),
      resourceId: "input",
    } as HostGraphCpuInputBinding;
    Object.defineProperty(unreadInput, "bytes", {
      enumerable: true,
      get() {
        inputReads += 1;
        return f32Bytes([1, 2]);
      },
    });

    for (const value of ["0", "3"]) {
      await expect(prepared.execute({
        inputs: [unreadInput],
        controls: [{
          controlId: "prefix-elements",
          value: wire(value),
        }],
      })).rejects.toMatchObject({
        code: "BG-GRAPH-CPU-INVALID-BINDING",
        path: "$.request.controls[0].value",
      });
    }
    expect(inputReads).toBe(0);
  });

  it("reduces finite f32 in ascending participant-rank order", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(
      collectiveProgram(artifacts, "f32", "sum"),
      artifacts,
    );
    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    const result = await prepared.execute({
      inputs: [
        input(1, f32Bytes([2.25, 5])),
        input(0, f32Bytes([1.5, -2])),
      ],
    });

    expect(result.outputs.map((output) => output.rank)).toEqual(["0", "1"]);
    expect(result.outputs.map((output) => readF32(output.bytes))).toEqual([
      [3.75, 3],
      [3.75, 3],
    ]);
  });

  it("implements wrapping i32 sum and exact u32 max", async () => {
    const signed = await identityArtifacts("i32");
    const signedGraph = await verified(
      collectiveProgram(signed, "i32", "sum"),
      signed,
    );
    const signedCpu = await prepareHostGraphCpu(
      signedGraph,
      artifactOptions(signed),
    );
    const signedResult = await signedCpu.execute({
      inputs: [
        input(0, i32Bytes([2_147_483_647, -2])),
        input(1, i32Bytes([1, -3])),
      ],
    });
    expect(signedResult.outputs.map((output) => readI32(output.bytes))).toEqual([
      [-2_147_483_648, -5],
      [-2_147_483_648, -5],
    ]);

    const unsigned = await identityArtifacts("u32");
    const unsignedGraph = await verified(
      collectiveProgram(unsigned, "u32", "max"),
      unsigned,
    );
    const unsignedCpu = await prepareHostGraphCpu(
      unsignedGraph,
      artifactOptions(unsigned),
    );
    const unsignedResult = await unsignedCpu.execute({
      inputs: [
        input(0, u32Bytes([1, 0xffff_ffff])),
        input(1, u32Bytes([2, 5])),
      ],
    });
    expect(
      unsignedResult.outputs.map((output) => readU32(output.bytes)),
    ).toEqual([
      [2, 0xffff_ffff],
      [2, 0xffff_ffff],
    ]);
  });

  it("snapshots inputs and rejects non-finite collective values before commit", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(
      collectiveProgram(artifacts, "f32", "sum"),
      artifacts,
    );
    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    const first = f32Bytes([1, 2]);
    const second = f32Bytes([3, 4]);
    const pending = prepared.execute({
      inputs: [input(0, first), input(1, second)],
    });
    first.fill(0);
    second.fill(0);
    const result = await pending;
    expect(result.outputs.map((output) => readF32(output.bytes))).toEqual([
      [4, 6],
      [4, 6],
    ]);

    const nonFinite = f32Bytes([Number.NaN, 1]);
    const original = new Uint8Array(nonFinite);
    await expect(prepared.execute({
      inputs: [input(0, nonFinite), input(1, f32Bytes([1, 2]))],
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-NUMERICAL-DOMAIN",
    });
    expect(nonFinite).toEqual(original);
  });

  it("fails closed for incomplete, duplicate, shared, and misaligned inputs", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(
      collectiveProgram(artifacts, "f32", "sum"),
      artifacts,
    );
    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    await expect(prepared.execute({
      inputs: [input(0, f32Bytes([1, 2]))],
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-INVALID-BINDING",
    });
    await expect(prepared.execute({
      inputs: [
        input(0, f32Bytes([1, 2])),
        input(0, f32Bytes([3, 4])),
      ],
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-INVALID-BINDING",
    });

    if (typeof SharedArrayBuffer !== "undefined") {
      await expect(prepared.execute({
        inputs: [
          input(0, new Uint8Array(new SharedArrayBuffer(8))),
          input(1, f32Bytes([3, 4])),
        ],
      })).rejects.toMatchObject({
        code: "BG-GRAPH-CPU-UNSUPPORTED-PROFILE",
      });
    }
    const misaligned = new Uint8Array(new ArrayBuffer(9), 1, 8);
    await expect(prepared.execute({
      inputs: [
        input(0, misaligned),
        input(1, f32Bytes([3, 4])),
      ],
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-INVALID-BINDING",
    });
  });

  it("requires exact graph and semantic authority without invoking accessors", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts, "f32"), artifacts);
    const copied = JSON.parse(JSON.stringify(graph)) as
      VerifiedHostGraphArtifact;
    await expect(prepareHostGraphCpu(
      copied,
      artifactOptions(artifacts),
    )).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-INVALID-AUTHORITY",
    });
    await expect(prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [artifacts.layout],
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-INVALID-BINDING",
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
    await expect(prepareHostGraphCpu(graph, hostile))
      .rejects.toBeInstanceOf(HostGraphCpuError);
    expect(reads).toBe(0);
  });

  it("enforces preparation, execution, and cancellation budgets", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts, "f32"), artifacts);
    await expect(prepareHostGraphCpu(graph, {
      ...artifactOptions(artifacts),
      maxWorkingBytes: 1,
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-RESOURCE-LIMIT",
    });
    await expect(prepareHostGraphCpu(graph, {
      ...artifactOptions(artifacts),
      maxElementOperations: 1,
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-RESOURCE-LIMIT",
    });

    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    const controller = new AbortController();
    controller.abort();
    await expect(prepared.execute({
      inputs: [input(0, f32Bytes([1, 2]))],
      signal: controller.signal,
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-ABORTED",
    });
  });
});
