import { describe, expect, it } from "vitest";

import {
  GRAPH_DIAGNOSTIC_CODES,
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  canonicalJsonBytes,
  hashSemanticArtifact,
  parseWireI64,
  parseWireU64,
  type WireU64,
} from "../../src/schema";
import {
  HOST_GRAPH_ARTIFACT_SCHEMA,
  HOST_GRAPH_FAILURE_MODEL,
  HOST_GRAPH_MAX_EXPANDED_NODES,
  HOST_GRAPH_MAX_RESOURCES,
  HOST_GRAPH_MAX_REPEAT_ITERATIONS,
  HOST_GRAPH_MAX_TOTAL_RESOURCE_BYTES,
  createVerifiedHostGraphArtifact,
  decodeHostGraphArtifact,
  hostGraphArtifactPayload,
  prepareHostGraphProgram,
  requirePreparedHostGraphProgram,
  verifyHostGraphArtifact,
  type HostGraphProgram,
} from "../../src/graph";
import {
  createVerifiedDensePermutationViewCopyArtifacts,
  type VerifiedViewCopyArtifacts,
} from "../../src/kernel";

async function semantic(): Promise<VerifiedViewCopyArtifacts> {
  return createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: [parseWireI64("2"), parseWireI64("3")],
    axes: [1, 0],
    dtype: "f32",
  });
}

const wire = (value: string): WireU64 => parseWireU64(value);

function program(artifacts: VerifiedViewCopyArtifacts): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 0 },
    failureModel: HOST_GRAPH_FAILURE_MODEL,
    rankCount: wire("2"),
    resources: [
      {
        resourceId: "output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "f32",
        byteLength: wire("24"),
        alignmentBytes: 4,
      },
      {
        resourceId: "input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "f32",
        byteLength: wire("24"),
        alignmentBytes: 4,
      },
      {
        resourceId: "bucket",
        role: "temporary",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "f32",
        byteLength: wire("24"),
        alignmentBytes: 4,
      },
    ],
    nodes: [
      {
        nodeId: "store",
        kind: "dispatch",
        dependsOn: ["synchronize"],
        semanticArtifactHash: artifacts.kernelSemanticHash,
        entrypointId: artifacts.operationId,
        dimensionBindings: {},
        bindings: [
          {
            semanticResourceId: artifacts.source.viewId,
            graphResourceId: "bucket",
          },
          {
            semanticResourceId: artifacts.destination.viewId,
            graphResourceId: "output",
          },
        ],
      },
      {
        nodeId: "transform",
        kind: "dispatch",
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
            graphResourceId: "bucket",
          },
        ],
      },
      {
        nodeId: "synchronize",
        kind: "all-reduce",
        dependsOn: ["transform"],
        resourceId: "bucket",
        reduction: "sum",
        dtype: "f32",
        numericalPolicy: "rank-order-f32",
        participants: [wire("1"), wire("0")],
        result: "replicated-to-all-participants",
      },
    ],
  };
}

function copyProgram(): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 1 },
    failureModel: HOST_GRAPH_FAILURE_MODEL,
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
        resourceId: "temporary",
        role: "temporary",
        multiplicity: "per-rank",
        initialization: "zero-fill",
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
    nodes: [
      {
        nodeId: "copy-input",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "input",
        destinationResourceId: "temporary",
        mode: "whole-allocation-bytes-per-rank",
      },
      {
        nodeId: "copy-output",
        kind: "copy",
        dependsOn: ["copy-input"],
        sourceResourceId: "temporary",
        destinationResourceId: "output",
        mode: "whole-allocation-bytes-per-rank",
      },
    ],
  };
}

function materializedCopyProgram(): HostGraphProgram {
  const copy = copyProgram();
  return {
    ...copy,
    version: { major: 1, minor: 2 },
    nodes: [
      ...copy.nodes,
      {
        nodeId: "materialize-output",
        kind: "materialize",
        dependsOn: ["copy-output"],
        resourceId: "output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function eventfulCopyProgram(): HostGraphProgram {
  const program = materializedCopyProgram();
  const materialization = program.nodes.find((node) =>
    node.kind === "materialize");
  if (materialization?.kind !== "materialize") {
    throw new Error("missing materialize node");
  }
  return {
    ...program,
    version: { major: 1, minor: 3 },
    nodes: [
      ...program.nodes.filter((node) => node.kind !== "materialize"),
      {
        nodeId: "copy-complete-event",
        kind: "event",
        dependsOn: ["copy-output"],
        eventId: "copy-complete",
        mode: "completion-after-dependencies",
      },
      {
        ...materialization,
        dependsOn: ["copy-complete-event"],
      },
    ],
  };
}

function repeatedCollectiveProgram(): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 4 },
    failureModel: HOST_GRAPH_FAILURE_MODEL,
    rankCount: wire("2"),
    resources: [
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
      if (node.kind !== "repeat") return node;
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

function dynamicDispatchProgram(
  artifacts: VerifiedViewCopyArtifacts,
): HostGraphProgram {
  const base = program(artifacts);
  return {
    ...base,
    version: { major: 1, minor: 9 },
    nodes: [
      ...base.nodes.map((node) =>
        node.kind === "dispatch" && node.nodeId === "transform"
          ? {
              ...node,
              kind: "dynamic-dispatch" as const,
              launchControl: {
                controlId: "prefix-elements",
                mode: "u32-prefix-element-count" as const,
              },
              maxElementCount: wire("4"),
              mode: "runtime-u32-prefix-elements" as const,
            }
          : node),
      {
        nodeId: "materialize-output",
        kind: "materialize",
        dependsOn: ["store"],
        resourceId: "output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function rectangularDynamicDispatchProgram(
  artifacts: VerifiedViewCopyArtifacts,
): HostGraphProgram {
  const base = dynamicDispatchProgram(artifacts);
  return {
    ...base,
    version: { major: 1, minor: 12 },
    nodes: base.nodes.map((node) => {
      if (
        node.kind !== "dynamic-dispatch" ||
        node.mode !== "runtime-u32-prefix-elements"
      ) {
        return node;
      }
      const {
        launchControl: _launchControl,
        maxElementCount: _maxElementCount,
        ...common
      } = node;
      return {
        ...common,
        launchControls: [
          {
            axis: 1,
            controlId: "prefix-columns",
            mode: "u32-prefix-extent" as const,
          },
          {
            axis: 0,
            controlId: "prefix-rows",
            mode: "u32-prefix-extent" as const,
          },
        ],
        maxExtents: [wire("2"), wire("2")],
        mode: "runtime-u32-rectangular-prefix" as const,
      };
    }),
  };
}

function resourceDynamicDispatchProgram(
  artifacts: VerifiedViewCopyArtifacts,
): HostGraphProgram {
  const base = dynamicDispatchProgram(artifacts);
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

function resourceRectangularDynamicDispatchProgram(
  artifacts: VerifiedViewCopyArtifacts,
): HostGraphProgram {
  const base = rectangularDynamicDispatchProgram(artifacts);
  const sourceResources = ["rows", "columns"].flatMap((axisName) => [
    {
      resourceId: `${axisName}-input`,
      role: "input" as const,
      multiplicity: "per-rank" as const,
      initialization: "external-input" as const,
      dtype: "u32" as const,
      byteLength: wire("4"),
      alignmentBytes: 4,
    },
    {
      resourceId: `${axisName}-extent`,
      role: "temporary" as const,
      multiplicity: "per-rank" as const,
      initialization: "zero-fill" as const,
      dtype: "u32" as const,
      byteLength: wire("4"),
      alignmentBytes: 4,
    },
  ]);
  const producerIds = ["rows", "columns"].map((axisName) =>
    `produce-${axisName}-extent`);
  return {
    ...base,
    version: { major: 1, minor: 13 },
    resources: [...base.resources, ...sourceResources],
    nodes: [
      ...["rows", "columns"].map((axisName) => ({
        nodeId: `produce-${axisName}-extent`,
        kind: "copy" as const,
        dependsOn: [],
        sourceResourceId: `${axisName}-input`,
        destinationResourceId: `${axisName}-extent`,
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
          dependsOn: [...node.dependsOn, ...producerIds],
          launchSources: [
            {
              axis: 1,
              resourceId: "columns-extent",
              rank: wire("0"),
              mode: "u32-prefix-extent" as const,
            },
            {
              axis: 0,
              resourceId: "rows-extent",
              rank: wire("0"),
              mode: "u32-prefix-extent" as const,
            },
          ],
          mode: "resource-u32-rectangular-prefix" as const,
        };
      }),
    ],
  };
}

function conditionalCopyProgram(): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 5 },
    failureModel: HOST_GRAPH_FAILURE_MODEL,
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

function runtimeConditionalCopyProgram(): HostGraphProgram {
  const program = clone(conditionalCopyProgram());
  program.version.minor = 6;
  program.resources = program.resources.filter((resource) =>
    resource.resourceId !== "predicate");
  const node = program.nodes.find((candidate) =>
    candidate.kind === "conditional");
  if (node?.kind !== "conditional") throw new Error("missing conditional");
  node.mode = "runtime-u32-branch-sequential";
  node.predicate = {
    controlId: "choose",
    mode: "u32-nonzero",
  };
  return program;
}

function resourceConditionalCopyProgram(): HostGraphProgram {
  const program = clone(conditionalCopyProgram());
  program.version.minor = 7;
  const predicate = program.resources.find((resource) =>
    resource.resourceId === "predicate");
  if (predicate === undefined) throw new Error("missing predicate resource");
  predicate.role = "temporary";
  predicate.initialization = "zero-fill";
  program.resources.push({
    resourceId: "predicate-source",
    role: "input",
    multiplicity: "per-rank",
    initialization: "external-input",
    dtype: "u32",
    byteLength: wire("4"),
    alignmentBytes: 4,
  });
  const conditional = program.nodes.find((node) =>
    node.kind === "conditional");
  if (conditional?.kind !== "conditional") {
    throw new Error("missing conditional");
  }
  conditional.mode = "resource-u32-branch-sequential";
  conditional.dependsOn = ["produce-predicate"];
  program.nodes.push({
    nodeId: "produce-predicate",
    kind: "copy",
    dependsOn: [],
    sourceResourceId: "predicate-source",
    destinationResourceId: "predicate",
    mode: "whole-allocation-bytes-per-rank",
  });
  return program;
}

async function diagnostic(
  run: () => Promise<unknown> | unknown,
): Promise<SemanticSchemaError> {
  try {
    await run();
    throw new Error("expected semantic failure");
  } catch (error) {
    expect(error).toBeInstanceOf(SemanticSchemaError);
    return error as SemanticSchemaError;
  }
}

type Mutable<T> = T extends string | number | boolean | bigint | symbol |
  null | undefined
  ? T
  : T extends readonly (infer Item)[]
    ? Mutable<Item>[]
    : T extends object
      ? { -readonly [Key in keyof T]: Mutable<T[Key]> }
      : T;

function clone<T>(value: T): Mutable<T> {
  return JSON.parse(JSON.stringify(value)) as Mutable<T>;
}

function artifactsFor(artifacts: VerifiedViewCopyArtifacts) {
  return {
    kernelArtifacts: [artifacts.kernel],
    layoutArtifacts: [artifacts.layout],
  };
}

describe("host graph artifact", () => {
  it("normalizes bounded input conditionals in version 1.5", async () => {
    const constructed = await createVerifiedHostGraphArtifact(
      conditionalCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.version).toEqual({ major: 1, minor: 5 });
    expect(payload.program.nodes.find((node) => node.kind === "conditional"))
      .toEqual({
        nodeId: "choose-output",
        kind: "conditional",
        dependsOn: [],
        predicate: {
          resourceId: "predicate",
          rank: "0",
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
      });
    expect(prepared).toMatchObject({
      nodeCount: 2,
      expandedNodeCount: 2,
      copyCount: 1,
      materializationCount: 1,
      conditionalCount: 1,
      topologicalNodeIds: ["choose-output", "materialize-output"],
    });
  });

  it("normalizes runtime u32 conditionals in version 1.6", async () => {
    const constructed = await createVerifiedHostGraphArtifact(
      runtimeConditionalCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.version).toEqual({ major: 1, minor: 6 });
    expect(payload.program.nodes.find((node) => node.kind === "conditional"))
      .toMatchObject({
        nodeId: "choose-output",
        mode: "runtime-u32-branch-sequential",
        predicate: {
          controlId: "choose",
          mode: "u32-nonzero",
        },
      });
    expect(prepared).toMatchObject({
      conditionalCount: 1,
      runtimeControlIds: ["choose"],
      expandedNodeCount: 2,
    });
  });

  it("normalizes one ordered resource u32 conditional in version 1.7", async () => {
    const constructed = await createVerifiedHostGraphArtifact(
      resourceConditionalCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.version).toEqual({ major: 1, minor: 7 });
    expect(payload.program.nodes.find((node) => node.kind === "conditional"))
      .toMatchObject({
        nodeId: "choose-output",
        dependsOn: ["produce-predicate"],
        mode: "resource-u32-branch-sequential",
        predicate: {
          resourceId: "predicate",
          rank: "0",
          mode: "u32-nonzero",
        },
      });
    expect(prepared).toMatchObject({
      conditionalCount: 1,
      resourceConditionalCount: 1,
      runtimeControlIds: [],
      copyCount: 2,
      expandedNodeCount: 3,
      topologicalNodeIds: [
        "produce-predicate",
        "choose-output",
        "materialize-output",
      ],
    });
  });

  it("rejects resource conditional version, authority, ordering, and count drift", async () => {
    const oldVersion = clone(resourceConditionalCopyProgram());
    oldVersion.version.minor = 6;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      oldVersion,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const externalPredicate = clone(resourceConditionalCopyProgram());
    const externalResource = externalPredicate.resources.find((resource) =>
      resource.resourceId === "predicate");
    if (externalResource === undefined) {
      throw new Error("missing predicate resource");
    }
    externalResource.role = "input";
    externalResource.initialization = "external-input";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      externalPredicate,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);

    const unordered = clone(resourceConditionalCopyProgram());
    const unorderedConditional = unordered.nodes.find((node) =>
      node.kind === "conditional");
    if (unorderedConditional?.kind !== "conditional") {
      throw new Error("missing conditional");
    }
    unorderedConditional.dependsOn = [];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      unordered,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.readBeforeWrite);

    const multiple = clone(resourceConditionalCopyProgram());
    const first = multiple.nodes.find((node) => node.kind === "conditional");
    if (first?.kind !== "conditional") throw new Error("missing conditional");
    const second = clone(first);
    second.nodeId = "choose-output-second";
    second.thenBody[0]!.nodeId = "copy-then-second";
    second.elseBody[0]!.nodeId = "copy-else-second";
    multiple.nodes.push(second);
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      multiple,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.resourceLimit);
  });

  it("rejects runtime conditional version and control drift", async () => {
    const oldVersion = clone(runtimeConditionalCopyProgram());
    oldVersion.version.minor = 5;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      oldVersion,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const missingControl = clone(runtimeConditionalCopyProgram());
    const missingControlNode = missingControl.nodes.find((node) =>
      node.kind === "conditional");
    if (
      missingControlNode?.kind !== "conditional" ||
      missingControlNode.mode !== "runtime-u32-branch-sequential"
    ) {
      throw new Error("missing runtime conditional");
    }
    missingControlNode.predicate.controlId = "";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      missingControl,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidArtifact);

    const predicateMode = clone(runtimeConditionalCopyProgram());
    const predicateModeNode = predicateMode.nodes.find((node) =>
      node.kind === "conditional");
    if (
      predicateModeNode?.kind !== "conditional" ||
      predicateModeNode.mode !== "runtime-u32-branch-sequential"
    ) {
      throw new Error("missing runtime conditional");
    }
    predicateModeNode.predicate.mode =
      "i32-positive" as typeof predicateModeNode.predicate.mode;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      predicateMode,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);
  });

  it("rejects forged conditional versions, predicates, shapes, and control", async () => {
    const oldVersion = clone(conditionalCopyProgram());
    oldVersion.version.minor = 4;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      oldVersion,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const mode = clone(conditionalCopyProgram());
    const modeNode = mode.nodes.find((node) => node.kind === "conditional");
    if (modeNode?.kind !== "conditional") {
      throw new Error("missing conditional node");
    }
    modeNode.mode = "device-branch" as typeof modeNode.mode;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      mode,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const predicateMode = clone(conditionalCopyProgram());
    const predicateModeNode = predicateMode.nodes.find((node) =>
      node.kind === "conditional");
    if (predicateModeNode?.kind !== "conditional") {
      throw new Error("missing conditional node");
    }
    predicateModeNode.predicate.mode =
      "i32-positive" as typeof predicateModeNode.predicate.mode;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      predicateMode,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    for (const mutate of [
      (candidate: Mutable<HostGraphProgram>) => {
        const predicate = candidate.resources.find((item) =>
          item.resourceId === "predicate");
        if (predicate === undefined) throw new Error("missing predicate");
        predicate.dtype = "i32";
      },
      (candidate: Mutable<HostGraphProgram>) => {
        const predicate = candidate.resources.find((item) =>
          item.resourceId === "predicate");
        if (predicate === undefined) throw new Error("missing predicate");
        predicate.byteLength = wire("8");
      },
      (candidate: Mutable<HostGraphProgram>) => {
        const predicate = candidate.resources.find((item) =>
          item.resourceId === "predicate");
        if (predicate === undefined) throw new Error("missing predicate");
        predicate.alignmentBytes = 2;
      },
      (candidate: Mutable<HostGraphProgram>) => {
        const predicate = candidate.resources.find((item) =>
          item.resourceId === "predicate");
        if (predicate === undefined) throw new Error("missing predicate");
        predicate.role = "temporary";
        predicate.initialization = "zero-fill";
      },
    ]) {
      const candidate = clone(conditionalCopyProgram());
      mutate(candidate);
      expect((await diagnostic(() => createVerifiedHostGraphArtifact(
        candidate,
        { kernelArtifacts: [], layoutArtifacts: [] },
      ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);
    }

    const missingPredicate = clone(conditionalCopyProgram());
    const missingPredicateNode = missingPredicate.nodes.find((node) =>
      node.kind === "conditional");
    if (missingPredicateNode?.kind !== "conditional") {
      throw new Error("missing conditional node");
    }
    missingPredicateNode.predicate.resourceId = "missing";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      missingPredicate,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.danglingReference);

    const rank = clone(conditionalCopyProgram());
    const rankNode = rank.nodes.find((node) => node.kind === "conditional");
    if (rankNode?.kind !== "conditional") {
      throw new Error("missing conditional node");
    }
    rankNode.predicate.rank = wire("1");
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      rank,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);

    const mismatchedShape = clone(conditionalCopyProgram());
    const mismatchedShapeNode = mismatchedShape.nodes.find((node) =>
      node.kind === "conditional");
    if (mismatchedShapeNode?.kind !== "conditional") {
      throw new Error("missing conditional node");
    }
    mismatchedShapeNode.elseBody.push({
      nodeId: "copy-else-again",
      kind: "copy",
      dependsOn: ["copy-else"],
      sourceResourceId: "else-input",
      destinationResourceId: "output",
      mode: "whole-allocation-bytes-per-rank",
    });
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      mismatchedShape,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const nestedControl = clone(conditionalCopyProgram());
    const nestedNode = nestedControl.nodes.find((node) =>
      node.kind === "conditional");
    if (nestedNode?.kind !== "conditional") {
      throw new Error("missing conditional node");
    }
    nestedNode.thenBody[0] = {
      nodeId: "nested-event",
      kind: "event",
      dependsOn: [],
      eventId: "nested",
      mode: "completion-after-dependencies",
    } as unknown as typeof nestedNode.thenBody[number];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      nestedControl,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const duplicateNodeId = clone(conditionalCopyProgram());
    const duplicateNode = duplicateNodeId.nodes.find((node) =>
      node.kind === "conditional");
    if (duplicateNode?.kind !== "conditional") {
      throw new Error("missing conditional node");
    }
    duplicateNode.elseBody[0]!.nodeId = "copy-then";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      duplicateNodeId,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.duplicateId);
  });

  it("requires conditional writes to be guaranteed across both branches", async () => {
    const optionalWrite = clone(conditionalCopyProgram());
    optionalWrite.resources.push({
      resourceId: "temporary",
      role: "temporary",
      multiplicity: "per-rank",
      initialization: "zero-fill",
      dtype: "u8",
      byteLength: wire("8"),
      alignmentBytes: 1,
    });
    const conditional = optionalWrite.nodes.find((node) =>
      node.kind === "conditional");
    if (conditional?.kind !== "conditional") {
      throw new Error("missing conditional node");
    }
    conditional.elseBody[0]!.destinationResourceId = "temporary";

    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      optionalWrite,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.readBeforeWrite);
  });

  it("normalizes bounded fixed-count sequential repetition in version 1.4", async () => {
    const constructed = await createVerifiedHostGraphArtifact(
      repeatedCollectiveProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.version).toEqual({ major: 1, minor: 4 });
    expect(payload.program.nodes.find((node) => node.kind === "repeat"))
      .toEqual({
        nodeId: "repeat-reduction",
        kind: "repeat",
        dependsOn: ["initialize"],
        iterationCount: "3",
        body: [{
          nodeId: "reduce-step",
          kind: "all-reduce",
          dependsOn: [],
          resourceId: "output",
          reduction: "sum",
          dtype: "f32",
          numericalPolicy: "rank-order-f32",
          participants: ["0", "1"],
          result: "replicated-to-all-participants",
        }],
        mode: "fixed-count-sequential",
      });
    expect(prepared).toMatchObject({
      nodeCount: 3,
      expandedNodeCount: 5,
      copyCount: 1,
      collectiveCount: 3,
      materializationCount: 1,
      repeatCount: 1,
      repeatIterationCount: 3,
      topologicalNodeIds: [
        "initialize",
        "repeat-reduction",
        "materialize-output",
      ],
    });
  });

  it("normalizes zero-through-bound runtime repetition in version 1.8", async () => {
    const constructed = await createVerifiedHostGraphArtifact(
      runtimeRepeatedCollectiveProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.version).toEqual({ major: 1, minor: 8 });
    expect(payload.program.nodes.find((node) => node.kind === "repeat"))
      .toEqual({
        nodeId: "repeat-reduction",
        kind: "repeat",
        dependsOn: ["initialize"],
        iterationControl: {
          controlId: "iterations",
          mode: "u32-count",
        },
        maxIterationCount: "3",
        body: [{
          nodeId: "reduce-step",
          kind: "all-reduce",
          dependsOn: [],
          resourceId: "output",
          reduction: "sum",
          dtype: "f32",
          numericalPolicy: "rank-order-f32",
          participants: ["0", "1"],
          result: "replicated-to-all-participants",
        }],
        mode: "runtime-u32-count-sequential",
      });
    expect(prepared).toMatchObject({
      repeatCount: 1,
      runtimeRepeatCount: 1,
      repeatIterationCount: 3,
      runtimeControlIds: ["iterations"],
      expandedNodeCount: 5,
    });
  });

  it("rejects unbounded, malformed, and pre-version runtime repetition", async () => {
    const oldVersion = clone(runtimeRepeatedCollectiveProgram());
    oldVersion.version.minor = 7;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      oldVersion,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const zeroBound = clone(runtimeRepeatedCollectiveProgram());
    const zeroRepeat = zeroBound.nodes.find((node) =>
      node.kind === "repeat");
    if (
      zeroRepeat?.kind !== "repeat" ||
      zeroRepeat.mode !== "runtime-u32-count-sequential"
    ) {
      throw new Error("missing runtime repeat");
    }
    zeroRepeat.maxIterationCount = wire("0");
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      zeroBound,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidArtifact);

    const excessive = clone(runtimeRepeatedCollectiveProgram());
    const excessiveRepeat = excessive.nodes.find((node) =>
      node.kind === "repeat");
    if (
      excessiveRepeat?.kind !== "repeat" ||
      excessiveRepeat.mode !== "runtime-u32-count-sequential"
    ) {
      throw new Error("missing runtime repeat");
    }
    excessiveRepeat.maxIterationCount = wire(String(
      HOST_GRAPH_MAX_REPEAT_ITERATIONS + 1,
    ));
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      excessive,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.resourceLimit);

    const wrongControlMode = clone(runtimeRepeatedCollectiveProgram());
    const wrongRepeat = wrongControlMode.nodes.find((node) =>
      node.kind === "repeat");
    if (
      wrongRepeat?.kind !== "repeat" ||
      wrongRepeat.mode !== "runtime-u32-count-sequential"
    ) {
      throw new Error("missing runtime repeat");
    }
    wrongRepeat.iterationControl.mode =
      "i32-count" as typeof wrongRepeat.iterationControl.mode;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      wrongControlMode,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);
  });

  it("does not treat a zero-iteration runtime repeat as a guaranteed writer", async () => {
    const candidate = clone(runtimeRepeatedCollectiveProgram());
    const repeat = candidate.nodes.find((node) => node.kind === "repeat");
    if (
      repeat?.kind !== "repeat" ||
      repeat.mode !== "runtime-u32-count-sequential"
    ) {
      throw new Error("missing runtime repeat");
    }
    repeat.dependsOn = [];
    repeat.body = [{
      nodeId: "write-output",
      kind: "copy",
      dependsOn: [],
      sourceResourceId: "input",
      destinationResourceId: "output",
      mode: "whole-allocation-bytes-per-rank",
    }];
    candidate.nodes = candidate.nodes.filter((node) =>
      node.nodeId !== "initialize");

    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      candidate,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.readBeforeWrite);
  });

  it("normalizes one bounded produced-resource repeat in version 1.10", async () => {
    const constructed = await createVerifiedHostGraphArtifact(
      resourceRepeatedCollectiveProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.version).toEqual({ major: 1, minor: 10 });
    expect(payload.program.nodes.find((node) =>
      node.kind === "repeat")).toMatchObject({
        nodeId: "repeat-reduction",
        iterationSource: {
          resourceId: "iteration-count",
          rank: "0",
          mode: "u32-count",
        },
        maxIterationCount: "3",
        mode: "resource-u32-count-sequential",
      });
    expect(prepared).toMatchObject({
      repeatCount: 1,
      runtimeRepeatCount: 0,
      resourceRepeatCount: 1,
      repeatIterationCount: 3,
      runtimeControlIds: [],
      expandedNodeCount: 6,
    });
  });

  it("rejects malformed, unordered, body-mutated, and pre-version resource repeats", async () => {
    const oldVersion = clone(resourceRepeatedCollectiveProgram());
    oldVersion.version.minor = 9;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      oldVersion,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const wrongRole = clone(resourceRepeatedCollectiveProgram());
    const countResource = wrongRole.resources.find((resource) =>
      resource.resourceId === "iteration-count");
    if (countResource === undefined) throw new Error("missing count resource");
    countResource.role = "input";
    countResource.initialization = "external-input";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      wrongRole,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);

    const unordered = clone(resourceRepeatedCollectiveProgram());
    const unorderedRepeat = unordered.nodes.find((node) =>
      node.kind === "repeat");
    if (
      unorderedRepeat?.kind !== "repeat" ||
      unorderedRepeat.mode !== "resource-u32-count-sequential"
    ) {
      throw new Error("missing resource repeat");
    }
    unorderedRepeat.dependsOn = ["initialize"];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      unordered,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.readBeforeWrite);

    const bodyMutation = clone(resourceRepeatedCollectiveProgram());
    const mutatedRepeat = bodyMutation.nodes.find((node) =>
      node.kind === "repeat");
    if (
      mutatedRepeat?.kind !== "repeat" ||
      mutatedRepeat.mode !== "resource-u32-count-sequential"
    ) {
      throw new Error("missing resource repeat");
    }
    mutatedRepeat.body = [{
      nodeId: "mutate-count",
      kind: "copy",
      dependsOn: [],
      sourceResourceId: "iteration-input",
      destinationResourceId: "iteration-count",
      mode: "whole-allocation-bytes-per-rank",
    }];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      bodyMutation,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);
  });

  it("normalizes bounded request-time dynamic dispatch in version 1.9", async () => {
    const artifacts = await semantic();
    const constructed = await createVerifiedHostGraphArtifact(
      dynamicDispatchProgram(artifacts),
      {
        kernelArtifacts: [artifacts.kernel],
        layoutArtifacts: [artifacts.layout],
      },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.version).toEqual({ major: 1, minor: 9 });
    expect(payload.program.nodes.find((node) =>
      node.kind === "dynamic-dispatch")).toMatchObject({
        nodeId: "transform",
        kind: "dynamic-dispatch",
        launchControl: {
          controlId: "prefix-elements",
          mode: "u32-prefix-element-count",
        },
        maxElementCount: "4",
        mode: "runtime-u32-prefix-elements",
      });
    expect(prepared).toMatchObject({
      dispatchCount: 2,
      dynamicDispatchCount: 1,
      runtimeControlIds: ["prefix-elements"],
    });
  });

  it("rejects malformed, excessive, nested, and pre-version dynamic dispatch", async () => {
    const artifacts = await semantic();
    const options = {
      kernelArtifacts: [artifacts.kernel],
      layoutArtifacts: [artifacts.layout],
    };
    const oldVersion = clone(dynamicDispatchProgram(artifacts));
    oldVersion.version.minor = 8;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      oldVersion,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const zeroMaximum = clone(dynamicDispatchProgram(artifacts));
    const zeroNode = zeroMaximum.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (zeroNode?.kind !== "dynamic-dispatch") {
      throw new Error("missing dynamic dispatch");
    }
    zeroNode.maxElementCount = wire("0");
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      zeroMaximum,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidArtifact);

    const excessive = clone(dynamicDispatchProgram(artifacts));
    const excessiveNode = excessive.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (excessiveNode?.kind !== "dynamic-dispatch") {
      throw new Error("missing dynamic dispatch");
    }
    excessiveNode.maxElementCount = wire("7");
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      excessive,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);

    const wrongMode = clone(dynamicDispatchProgram(artifacts));
    const wrongNode = wrongMode.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (
      wrongNode?.kind !== "dynamic-dispatch" ||
      wrongNode.mode !== "runtime-u32-prefix-elements"
    ) {
      throw new Error("missing dynamic dispatch");
    }
    wrongNode.launchControl.mode =
      "workgroups" as typeof wrongNode.launchControl.mode;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      wrongMode,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const nested = clone(runtimeRepeatedCollectiveProgram());
    const repeat = nested.nodes.find((node) => node.kind === "repeat");
    if (repeat?.kind !== "repeat") throw new Error("missing repeat");
    repeat.body = [clone(
      dynamicDispatchProgram(artifacts).nodes.find((node) =>
        node.kind === "dynamic-dispatch")!,
    ) as unknown as typeof repeat.body[number]];
    nested.version.minor = 9;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      nested,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);
  });

  it("normalizes bounded rank-2 rectangular dynamic dispatch in version 1.12", async () => {
    const artifacts = await semantic();
    const constructed = await createVerifiedHostGraphArtifact(
      rectangularDynamicDispatchProgram(artifacts),
      {
        kernelArtifacts: [artifacts.kernel],
        layoutArtifacts: [artifacts.layout],
      },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.version).toEqual({ major: 1, minor: 12 });
    expect(payload.program.nodes.find((node) =>
      node.kind === "dynamic-dispatch")).toMatchObject({
        nodeId: "transform",
        launchControls: [
          {
            axis: 0,
            controlId: "prefix-rows",
            mode: "u32-prefix-extent",
          },
          {
            axis: 1,
            controlId: "prefix-columns",
            mode: "u32-prefix-extent",
          },
        ],
        maxExtents: ["2", "2"],
        mode: "runtime-u32-rectangular-prefix",
      });
    expect(prepared).toMatchObject({
      dynamicDispatchCount: 1,
      resourceDynamicDispatchCount: 0,
      runtimeControlIds: ["prefix-columns", "prefix-rows"],
    });
  });

  it("rejects malformed, excessive, and pre-version rectangular dispatch", async () => {
    const artifacts = await semantic();
    const options = {
      kernelArtifacts: [artifacts.kernel],
      layoutArtifacts: [artifacts.layout],
    };
    const oldVersion = clone(rectangularDynamicDispatchProgram(artifacts));
    oldVersion.version.minor = 11;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      oldVersion,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const excessive = clone(rectangularDynamicDispatchProgram(artifacts));
    const excessiveNode = excessive.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (
      excessiveNode?.kind !== "dynamic-dispatch" ||
      excessiveNode.mode !== "runtime-u32-rectangular-prefix"
    ) {
      throw new Error("missing rectangular dynamic dispatch");
    }
    excessiveNode.maxExtents = [wire("4"), wire("2")];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      excessive,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);

    const rankDrift = clone(rectangularDynamicDispatchProgram(artifacts));
    const rankNode = rankDrift.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (
      rankNode?.kind !== "dynamic-dispatch" ||
      rankNode.mode !== "runtime-u32-rectangular-prefix"
    ) {
      throw new Error("missing rectangular dynamic dispatch");
    }
    rankNode.maxExtents = [wire("2"), wire("2"), wire("1")];
    rankNode.launchControls = [
      ...rankNode.launchControls,
      {
        axis: 2,
        controlId: "prefix-depth",
        mode: "u32-prefix-extent",
      },
    ];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      rankDrift,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);

    const duplicateAxis = clone(rectangularDynamicDispatchProgram(artifacts));
    const duplicateAxisNode = duplicateAxis.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (
      duplicateAxisNode?.kind !== "dynamic-dispatch" ||
      duplicateAxisNode.mode !== "runtime-u32-rectangular-prefix"
    ) {
      throw new Error("missing rectangular dynamic dispatch");
    }
    duplicateAxisNode.launchControls[0]!.axis = 0;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      duplicateAxis,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.duplicateId);

    const duplicateControl = clone(
      rectangularDynamicDispatchProgram(artifacts),
    );
    const duplicateControlNode = duplicateControl.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (
      duplicateControlNode?.kind !== "dynamic-dispatch" ||
      duplicateControlNode.mode !== "runtime-u32-rectangular-prefix"
    ) {
      throw new Error("missing rectangular dynamic dispatch");
    }
    duplicateControlNode.launchControls[1]!.controlId =
      duplicateControlNode.launchControls[0]!.controlId;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      duplicateControl,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.duplicateId);

    for (const maxExtents of [
      [wire("2")],
      [wire("1"), wire("1"), wire("1"), wire("1")],
    ]) {
      const unsupportedRank = clone(
        rectangularDynamicDispatchProgram(artifacts),
      );
      const unsupportedNode = unsupportedRank.nodes.find((node) =>
        node.kind === "dynamic-dispatch");
      if (
        unsupportedNode?.kind !== "dynamic-dispatch" ||
        unsupportedNode.mode !== "runtime-u32-rectangular-prefix"
      ) {
        throw new Error("missing rectangular dynamic dispatch");
      }
      unsupportedNode.maxExtents = maxExtents;
      expect((await diagnostic(() => createVerifiedHostGraphArtifact(
        unsupportedRank,
        options,
      ))).diagnostic.code).toBe(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      );
    }
  });

  it("normalizes one bounded produced-resource rectangle in version 1.13", async () => {
    const artifacts = await semantic();
    const constructed = await createVerifiedHostGraphArtifact(
      resourceRectangularDynamicDispatchProgram(artifacts),
      {
        kernelArtifacts: [artifacts.kernel],
        layoutArtifacts: [artifacts.layout],
      },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.version).toEqual({ major: 1, minor: 13 });
    expect(payload.program.nodes.find((node) =>
      node.kind === "dynamic-dispatch")).toMatchObject({
        nodeId: "transform",
        launchSources: [
          {
            axis: 0,
            resourceId: "rows-extent",
            rank: "0",
            mode: "u32-prefix-extent",
          },
          {
            axis: 1,
            resourceId: "columns-extent",
            rank: "0",
            mode: "u32-prefix-extent",
          },
        ],
        maxExtents: ["2", "2"],
        mode: "resource-u32-rectangular-prefix",
      });
    expect(prepared).toMatchObject({
      dynamicDispatchCount: 1,
      resourceDynamicDispatchCount: 1,
      runtimeControlIds: [],
    });
  });

  it("rejects malformed, unordered, and pre-version produced-resource rectangles", async () => {
    const artifacts = await semantic();
    const options = {
      kernelArtifacts: [artifacts.kernel],
      layoutArtifacts: [artifacts.layout],
    };
    const oldVersion = clone(
      resourceRectangularDynamicDispatchProgram(artifacts),
    );
    oldVersion.version.minor = 12;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      oldVersion,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const unordered = clone(
      resourceRectangularDynamicDispatchProgram(artifacts),
    );
    const unorderedNode = unordered.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (
      unorderedNode?.kind !== "dynamic-dispatch" ||
      unorderedNode.mode !== "resource-u32-rectangular-prefix"
    ) {
      throw new Error("missing resource rectangular dynamic dispatch");
    }
    unorderedNode.dependsOn = [];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      unordered,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.readBeforeWrite);

    const duplicateAxis = clone(
      resourceRectangularDynamicDispatchProgram(artifacts),
    );
    const duplicateAxisNode = duplicateAxis.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (
      duplicateAxisNode?.kind !== "dynamic-dispatch" ||
      duplicateAxisNode.mode !== "resource-u32-rectangular-prefix"
    ) {
      throw new Error("missing resource rectangular dynamic dispatch");
    }
    duplicateAxisNode.launchSources[1]!.axis = 1;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      duplicateAxis,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.duplicateId);

    const duplicateResource = clone(
      resourceRectangularDynamicDispatchProgram(artifacts),
    );
    const duplicateResourceNode = duplicateResource.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (
      duplicateResourceNode?.kind !== "dynamic-dispatch" ||
      duplicateResourceNode.mode !== "resource-u32-rectangular-prefix"
    ) {
      throw new Error("missing resource rectangular dynamic dispatch");
    }
    duplicateResourceNode.launchSources[1]!.resourceId =
      duplicateResourceNode.launchSources[0]!.resourceId;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      duplicateResource,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.duplicateId);

    const unsupportedRank = clone(
      resourceRectangularDynamicDispatchProgram(artifacts),
    );
    unsupportedRank.version.minor = 14;
    const unsupportedRankNode = unsupportedRank.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (
      unsupportedRankNode?.kind !== "dynamic-dispatch" ||
      unsupportedRankNode.mode !== "resource-u32-rectangular-prefix"
    ) {
      throw new Error("missing resource rectangular dynamic dispatch");
    }
    unsupportedRankNode.maxExtents = [
      wire("2"),
      wire("2"),
      wire("1"),
      wire("1"),
    ];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      unsupportedRank,
      options,
    ))).diagnostic.code).toBe(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
    );

    const unsupportedRankFive = clone(unsupportedRank);
    unsupportedRankFive.version.minor = 16;
    const unsupportedRankFiveNode = unsupportedRankFive.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (
      unsupportedRankFiveNode?.kind !== "dynamic-dispatch" ||
      unsupportedRankFiveNode.mode !== "resource-u32-rectangular-prefix"
    ) {
      throw new Error("missing resource rectangular dynamic dispatch");
    }
    unsupportedRankFiveNode.maxExtents = [
      wire("1"),
      wire("1"),
      wire("1"),
      wire("1"),
      wire("1"),
    ];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      unsupportedRankFive,
      options,
    ))).diagnostic.code).toBe(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
    );
  });

  it("normalizes one bounded produced-resource dynamic dispatch in version 1.11", async () => {
    const artifacts = await semantic();
    const constructed = await createVerifiedHostGraphArtifact(
      resourceDynamicDispatchProgram(artifacts),
      {
        kernelArtifacts: [artifacts.kernel],
        layoutArtifacts: [artifacts.layout],
      },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.version).toEqual({ major: 1, minor: 11 });
    expect(payload.program.nodes.find((node) =>
      node.kind === "dynamic-dispatch")).toMatchObject({
        nodeId: "transform",
        launchSource: {
          resourceId: "launch-count",
          rank: "0",
          mode: "u32-prefix-element-count",
        },
        maxElementCount: "4",
        mode: "resource-u32-prefix-elements",
      });
    expect(prepared).toMatchObject({
      dynamicDispatchCount: 1,
      resourceDynamicDispatchCount: 1,
      runtimeControlIds: [],
    });
  });

  it("rejects malformed, unordered, duplicate, and pre-version resource dynamic dispatch", async () => {
    const artifacts = await semantic();
    const options = {
      kernelArtifacts: [artifacts.kernel],
      layoutArtifacts: [artifacts.layout],
    };
    const oldVersion = clone(resourceDynamicDispatchProgram(artifacts));
    oldVersion.version.minor = 10;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      oldVersion,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const wrongRole = clone(resourceDynamicDispatchProgram(artifacts));
    const launchResource = wrongRole.resources.find((resource) =>
      resource.resourceId === "launch-count");
    if (launchResource === undefined) {
      throw new Error("missing launch resource");
    }
    launchResource.role = "input";
    launchResource.initialization = "external-input";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      wrongRole,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);

    const unordered = clone(resourceDynamicDispatchProgram(artifacts));
    const unorderedDispatch = unordered.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (
      unorderedDispatch?.kind !== "dynamic-dispatch" ||
      unorderedDispatch.mode !== "resource-u32-prefix-elements"
    ) {
      throw new Error("missing resource dynamic dispatch");
    }
    unorderedDispatch.dependsOn = [];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      unordered,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.readBeforeWrite);

    const duplicate = clone(resourceDynamicDispatchProgram(artifacts));
    const dynamic = duplicate.nodes.find((node) =>
      node.kind === "dynamic-dispatch");
    if (dynamic?.kind !== "dynamic-dispatch") {
      throw new Error("missing resource dynamic dispatch");
    }
    duplicate.nodes.push({
      ...clone(dynamic),
      nodeId: "second-resource-dynamic-dispatch",
    });
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      duplicate,
      options,
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.resourceLimit);
  });

  it("rejects forged repeat versions, modes, bounds, and body control", async () => {
    const oldVersion = clone(repeatedCollectiveProgram());
    oldVersion.version.minor = 3;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      oldVersion,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const mode = clone(repeatedCollectiveProgram());
    const modeNode = mode.nodes.find((node) => node.kind === "repeat");
    if (modeNode?.kind !== "repeat") throw new Error("missing repeat node");
    modeNode.mode = "while" as typeof modeNode.mode;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      mode,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const zero = clone(repeatedCollectiveProgram());
    const zeroNode = zero.nodes.find((node) => node.kind === "repeat");
    if (zeroNode?.kind !== "repeat") throw new Error("missing repeat node");
    zeroNode.iterationCount = wire("0");
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      zero,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidArtifact);

    const excessive = clone(repeatedCollectiveProgram());
    const excessiveNode = excessive.nodes.find((node) =>
      node.kind === "repeat");
    if (excessiveNode?.kind !== "repeat") {
      throw new Error("missing repeat node");
    }
    excessiveNode.iterationCount = wire(String(
      HOST_GRAPH_MAX_REPEAT_ITERATIONS + 1,
    ));
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      excessive,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.resourceLimit);

    const nonlinear = clone(repeatedCollectiveProgram());
    const nonlinearNode = nonlinear.nodes.find((node) =>
      node.kind === "repeat");
    if (nonlinearNode?.kind !== "repeat") {
      throw new Error("missing repeat node");
    }
    nonlinearNode.body.push({
      nodeId: "copy-step",
      kind: "copy",
      dependsOn: [],
      sourceResourceId: "input",
      destinationResourceId: "output",
      mode: "whole-allocation-bytes-per-rank",
    });
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      nonlinear,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidAccess);

    const nestedControl = clone(repeatedCollectiveProgram());
    const nestedNode = nestedControl.nodes.find((node) =>
      node.kind === "repeat");
    if (nestedNode?.kind !== "repeat") throw new Error("missing repeat node");
    nestedNode.body[0] = {
      nodeId: "nested-event",
      kind: "event",
      dependsOn: [],
      eventId: "nested",
      mode: "completion-after-dependencies",
    } as unknown as typeof nestedNode.body[number];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      nestedControl,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const duplicateNodeId = clone(repeatedCollectiveProgram());
    const duplicateNode = duplicateNodeId.nodes.find((node) =>
      node.kind === "repeat");
    if (duplicateNode?.kind !== "repeat") {
      throw new Error("missing repeat node");
    }
    duplicateNode.body[0]!.nodeId = "initialize";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      duplicateNodeId,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.duplicateId);

    const expansion = clone(repeatedCollectiveProgram());
    const expansionNode = expansion.nodes.find((node) =>
      node.kind === "repeat");
    if (expansionNode?.kind !== "repeat") {
      throw new Error("missing repeat node");
    }
    expansionNode.iterationCount = wire(String(
      HOST_GRAPH_MAX_REPEAT_ITERATIONS,
    ));
    expansionNode.body = Array.from({ length: 16 }, (_, index) => ({
      nodeId: `copy-step-${index}`,
      kind: "copy" as const,
      dependsOn: index === 0 ? [] : [`copy-step-${index - 1}`],
      sourceResourceId: "input",
      destinationResourceId: "output",
      mode: "whole-allocation-bytes-per-rank" as const,
    }));
    expect(HOST_GRAPH_MAX_REPEAT_ITERATIONS * expansionNode.body.length)
      .toBe(HOST_GRAPH_MAX_EXPANDED_NODES);
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      expansion,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.resourceLimit);
  });

  it("normalizes dependency-ordered completion events in version 1.3", async () => {
    const constructed = await createVerifiedHostGraphArtifact(
      eventfulCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.version).toEqual({ major: 1, minor: 3 });
    expect(payload.program.nodes.find((node) => node.kind === "event"))
      .toEqual({
        nodeId: "copy-complete-event",
        kind: "event",
        dependsOn: ["copy-output"],
        eventId: "copy-complete",
        mode: "completion-after-dependencies",
      });
    expect(prepared).toMatchObject({
      nodeCount: 4,
      copyCount: 2,
      materializationCount: 1,
      eventCount: 1,
      eventIds: ["copy-complete"],
      topologicalNodeIds: [
        "copy-input",
        "copy-output",
        "copy-complete-event",
        "materialize-output",
      ],
    });
  });

  it("rejects forged event versions, modes, and duplicate identities", async () => {
    const version = clone(eventfulCopyProgram());
    version.version.minor = 2;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      version,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const mode = clone(eventfulCopyProgram());
    const modeNode = mode.nodes.find((node) => node.kind === "event");
    if (modeNode?.kind !== "event") throw new Error("missing event node");
    modeNode.mode = "timestamp" as typeof modeNode.mode;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      mode,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const duplicate = clone(eventfulCopyProgram());
    duplicate.nodes.push({
      nodeId: "copy-complete-event-again",
      kind: "event",
      dependsOn: ["copy-output"],
      eventId: "copy-complete",
      mode: "completion-after-dependencies",
    });
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      duplicate,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.duplicateId);

    const unordered = clone(eventfulCopyProgram());
    const unorderedEvent = unordered.nodes.find((node) =>
      node.kind === "event");
    if (unorderedEvent?.kind !== "event") {
      throw new Error("missing event node");
    }
    unorderedEvent.dependsOn = [];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      unordered,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.readBeforeWrite);
  });

  it("normalizes explicit terminal materialization in version 1.2", async () => {
    const constructed = await createVerifiedHostGraphArtifact(
      materializedCopyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.version).toEqual({ major: 1, minor: 2 });
    expect(payload.program.nodes.at(-1)).toEqual({
      nodeId: "materialize-output",
      kind: "materialize",
      dependsOn: ["copy-output"],
      resourceId: "output",
      mode: "host-readback-after-graph-success",
    });
    expect(prepared).toMatchObject({
      rankCount: 2n,
      nodeCount: 3,
      dispatchCount: 0,
      collectiveCount: 0,
      copyCount: 2,
      materializationCount: 1,
      eventCount: 0,
      eventIds: [],
      topologicalNodeIds: [
        "copy-input",
        "copy-output",
        "materialize-output",
      ],
      outputResourceIds: ["output"],
    });
  });

  it("rejects incomplete, duplicated, nonterminal, or forged materialization", async () => {
    const missing = clone(materializedCopyProgram());
    missing.nodes = missing.nodes.filter((node) =>
      node.kind !== "materialize");
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      missing,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidAccess);

    const duplicate = clone(materializedCopyProgram());
    duplicate.nodes.push({
      nodeId: "materialize-output-again",
      kind: "materialize",
      dependsOn: ["copy-output"],
      resourceId: "output",
      mode: "host-readback-after-graph-success",
    });
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      duplicate,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidAccess);

    const wrongRole = clone(materializedCopyProgram());
    const wrongRoleNode = wrongRole.nodes.find((node) =>
      node.kind === "materialize");
    if (wrongRoleNode?.kind !== "materialize") {
      throw new Error("missing materialize node");
    }
    wrongRoleNode.resourceId = "temporary";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      wrongRole,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidAccess);

    const missingResource = clone(materializedCopyProgram());
    const missingResourceNode = missingResource.nodes.find((node) =>
      node.kind === "materialize");
    if (missingResourceNode?.kind !== "materialize") {
      throw new Error("missing materialize node");
    }
    missingResourceNode.resourceId = "missing";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      missingResource,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.danglingReference);

    const unordered = clone(materializedCopyProgram());
    const unorderedNode = unordered.nodes.find((node) =>
      node.kind === "materialize");
    if (unorderedNode?.kind !== "materialize") {
      throw new Error("missing materialize node");
    }
    unorderedNode.dependsOn = [];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      unordered,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.readBeforeWrite);

    const nonterminal = clone(materializedCopyProgram());
    nonterminal.nodes.push({
      nodeId: "copy-after-materialize",
      kind: "copy",
      dependsOn: ["materialize-output"],
      sourceResourceId: "input",
      destinationResourceId: "temporary",
      mode: "whole-allocation-bytes-per-rank",
    });
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      nonterminal,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidAccess);

    const mode = clone(materializedCopyProgram());
    const modeNode = mode.nodes.find((node) =>
      node.kind === "materialize");
    if (modeNode?.kind !== "materialize") {
      throw new Error("missing materialize node");
    }
    modeNode.mode = "eager-readback" as typeof modeNode.mode;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      mode,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const version = clone(materializedCopyProgram());
    version.version.minor = 1;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      version,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);
  });

  it("normalizes version-1.1 whole-allocation per-rank copies", async () => {
    const constructed = await createVerifiedHostGraphArtifact(
      copyProgram(),
      { kernelArtifacts: [], layoutArtifacts: [] },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.version).toEqual({ major: 1, minor: 1 });
    expect(payload.program.nodes).toEqual([
      {
        nodeId: "copy-input",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "input",
        destinationResourceId: "temporary",
        mode: "whole-allocation-bytes-per-rank",
      },
      {
        nodeId: "copy-output",
        kind: "copy",
        dependsOn: ["copy-input"],
        sourceResourceId: "temporary",
        destinationResourceId: "output",
        mode: "whole-allocation-bytes-per-rank",
      },
    ]);
    expect(prepared).toMatchObject({
      rankCount: 2n,
      nodeCount: 2,
      dispatchCount: 0,
      collectiveCount: 0,
      copyCount: 2,
      materializationCount: 0,
      eventCount: 0,
      eventIds: [],
      topologicalNodeIds: ["copy-input", "copy-output"],
    });
  });

  it("rejects copy version, mode, resource, dtype, and hazard drift", async () => {
    const future = clone(copyProgram());
    future.version.minor = 33 as unknown as typeof future.version.minor;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      future,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const version = clone(copyProgram());
    version.version.minor = 0;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      version,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const mismatch = {
      schema: HOST_GRAPH_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 1 },
      producer: { id: "test", version: "1" },
      artifactId: "minor-mismatch",
      requiredExtensions: [],
      payload: { program: { ...copyProgram(), version: { major: 1, minor: 0 } } },
    };
    expect((await diagnostic(() => verifyHostGraphArtifact(
      mismatch,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidArtifact);

    const mode = clone(copyProgram());
    const modeNode = mode.nodes[0]!;
    if (modeNode.kind !== "copy") throw new Error("missing copy node");
    modeNode.mode = "partial" as typeof modeNode.mode;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      mode,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const missing = clone(copyProgram());
    const missingNode = missing.nodes[0]!;
    if (missingNode.kind !== "copy") throw new Error("missing copy node");
    missingNode.sourceResourceId = "missing";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      missing,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.danglingReference);

    const dtype = clone(copyProgram());
    dtype.resources.find((item) => item.resourceId === "temporary")!.dtype =
      "i8";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      dtype,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);

    const size = clone(copyProgram());
    size.resources.find((item) => item.resourceId === "temporary")!
      .byteLength = wire("8");
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      size,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);

    const self = clone(copyProgram());
    const selfNode = self.nodes[0]!;
    if (selfNode.kind !== "copy") throw new Error("missing copy node");
    selfNode.destinationResourceId = "input";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      self,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidAccess);

    const inputWrite = clone(copyProgram());
    const inputWriter = inputWrite.nodes[1]!;
    if (inputWriter.kind !== "copy") throw new Error("missing copy node");
    inputWriter.destinationResourceId = "input";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      inputWrite,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidAccess);

    const readBeforeWrite = clone(copyProgram());
    const first = readBeforeWrite.nodes[0]!;
    if (first.kind !== "copy") throw new Error("missing copy node");
    first.sourceResourceId = "temporary";
    first.destinationResourceId = "output";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      readBeforeWrite,
      { kernelArtifacts: [], layoutArtifacts: [] },
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.readBeforeWrite);
  });

  it("normalizes a multi-dispatch DAG with explicit collective meaning", async () => {
    const artifacts = await semantic();
    const constructed = await createVerifiedHostGraphArtifact(
      program(artifacts),
      {
        ...artifactsFor(artifacts),
        producer: { id: "graph-builder", version: "7" },
        artifactId: "training-step",
      },
    );
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(payload.program.resources.map((item) => item.resourceId)).toEqual([
      "bucket",
      "input",
      "output",
    ]);
    expect(payload.program.nodes.map((item) => item.nodeId)).toEqual([
      "store",
      "synchronize",
      "transform",
    ]);
    expect(payload.program.nodes[1]).toMatchObject({
      kind: "all-reduce",
      participants: ["0", "1"],
      numericalPolicy: "rank-order-f32",
      result: "replicated-to-all-participants",
    });
    expect(prepared).toMatchObject({
      artifact: constructed.artifact,
      graphSemanticHash: constructed.graphSemanticHash,
      failureModel: HOST_GRAPH_FAILURE_MODEL,
      rankCount: 2n,
      resourceCount: 3,
      nodeCount: 3,
      edgeCount: 2,
      dispatchCount: 2,
      collectiveCount: 1,
      copyCount: 0,
      materializationCount: 0,
      eventCount: 0,
      eventIds: [],
      topologicalNodeIds: ["transform", "synchronize", "store"],
      outputResourceIds: ["output"],
    });
    expect(() => requirePreparedHostGraphProgram(prepared)).not.toThrow();
  });

  it("round-trips canonical bytes and excludes transport metadata from identity", async () => {
    const artifacts = await semantic();
    const first = await createVerifiedHostGraphArtifact(
      program(artifacts),
      {
        ...artifactsFor(artifacts),
        producer: { id: "producer-a", version: "1" },
        artifactId: "transport-a",
      },
    );
    const envelope = {
      schema: HOST_GRAPH_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 0 },
      producer: { id: "producer-b", version: "99" },
      artifactId: "transport-b",
      requiredExtensions: [],
      payload: hostGraphArtifactPayload(first.artifact),
    };
    const second = await decodeHostGraphArtifact(
      canonicalJsonBytes(envelope),
      artifactsFor(artifacts),
    );

    expect(await hashSemanticArtifact(second)).toBe(first.graphSemanticHash);
    expect(
      JSON.stringify(hostGraphArtifactPayload(second)).toLowerCase(),
    ).not.toContain("transport");
  });

  it("rejects copied authority and semantic artifacts absent from the verifier set", async () => {
    const artifacts = await semantic();
    const constructed = await createVerifiedHostGraphArtifact(
      program(artifacts),
      artifactsFor(artifacts),
    );
    const prepared = await prepareHostGraphProgram(constructed.artifact);
    expect((await diagnostic(() =>
      hostGraphArtifactPayload(
        hostGraphArtifactPayload(constructed.artifact) as never,
      ))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.unverifiedArtifact);
    expect((await diagnostic(() =>
      requirePreparedHostGraphProgram({ ...prepared })
    )).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);

    const envelope = {
      schema: HOST_GRAPH_ARTIFACT_SCHEMA,
      version: { major: 1, minor: 0 },
      producer: { id: "test", version: "1" },
      artifactId: "missing-semantic",
      requiredExtensions: [],
      payload: { program: program(artifacts) },
    };
    expect((await diagnostic(() => verifyHostGraphArtifact(envelope, {
      kernelArtifacts: [],
      layoutArtifacts: [artifacts.layout],
    }))).diagnostic.code).toBe(
      GRAPH_DIAGNOSTIC_CODES.semanticArtifactMismatch,
    );
    expect((await diagnostic(() => verifyHostGraphArtifact(envelope, {
      kernelArtifacts: [artifacts.kernel],
      layoutArtifacts: [],
    }))).diagnostic.code).toBe(
      GRAPH_DIAGNOSTIC_CODES.semanticArtifactMismatch,
    );
  });

  it("rejects dangling dependencies, cycles, and unordered resource hazards", async () => {
    const artifacts = await semantic();
    const dangling = clone(program(artifacts));
    dangling.nodes[0]!.dependsOn = ["missing"];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      dangling,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.danglingReference);

    const cycle = clone(program(artifacts));
    cycle.nodes.find((node) => node.nodeId === "transform")!.dependsOn = [
      "store",
    ];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      cycle,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.cycle);

    const hazard = clone(program(artifacts));
    hazard.nodes.find((node) => node.nodeId === "store")!.dependsOn = [
      "transform",
    ];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      hazard,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.effectConflict);
  });

  it("rejects read-before-write, input mutation, and unwritten outputs", async () => {
    const artifacts = await semantic();
    const beforeWrite = clone(program(artifacts));
    const transform = beforeWrite.nodes.find(
      (node) => node.nodeId === "transform",
    );
    if (transform?.kind !== "dispatch") throw new Error("missing transform");
    transform.bindings[0]!.graphResourceId = "bucket";
    transform.bindings[1]!.graphResourceId = "output";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      beforeWrite,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.readBeforeWrite);

    const inputWrite = clone(program(artifacts));
    const inputWriter = inputWrite.nodes.find(
      (node) => node.nodeId === "transform",
    );
    if (inputWriter?.kind !== "dispatch") throw new Error("missing transform");
    inputWriter.bindings[1]!.graphResourceId = "input";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      inputWrite,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidAccess);

    const unwritten = clone(program(artifacts));
    const store = unwritten.nodes.find((node) => node.nodeId === "store");
    if (store?.kind !== "dispatch") throw new Error("missing store");
    unwritten.resources.push({
      resourceId: "sink",
      role: "temporary",
      multiplicity: "per-rank",
      initialization: "zero-fill",
      dtype: "f32",
      byteLength: wire("24"),
      alignmentBytes: 4,
    });
    store.bindings[1]!.graphResourceId = "sink";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      unwritten,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.readBeforeWrite);
  });

  it("derives effects and resource geometry from verified kernel meaning", async () => {
    const artifacts = await semantic();
    const forgedBinding = clone(program(artifacts));
    const forgedDispatch = forgedBinding.nodes.find(
      (node) => node.nodeId === "transform",
    );
    if (forgedDispatch?.kind !== "dispatch") {
      throw new Error("missing transform");
    }
    forgedDispatch.bindings[0]!.semanticResourceId = "forgedView";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      forgedBinding,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(
      GRAPH_DIAGNOSTIC_CODES.semanticArtifactMismatch,
    );

    const callerEffects = clone(program(artifacts)) as unknown as
      HostGraphProgram;
    const callerDispatch = callerEffects.nodes[0] as unknown as {
      effects?: unknown;
    };
    callerDispatch.effects = [{ resourceId: "input", access: "read" }];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      callerEffects,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unknownField);

    const dispatchDtype = clone(program(artifacts));
    dispatchDtype.resources.find(
      (resource) => resource.resourceId === "bucket",
    )!.dtype = "i32";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      dispatchDtype,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);

    const dispatchLength = clone(program(artifacts));
    dispatchLength.resources.find(
      (resource) => resource.resourceId === "bucket",
    )!.byteLength = wire("28");
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      dispatchLength,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);

    const dispatchAlignment = clone(program(artifacts));
    dispatchAlignment.resources.find(
      (resource) => resource.resourceId === "bucket",
    )!.alignmentBytes = 2;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      dispatchAlignment,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);

    const collectiveDtype = clone(program(artifacts));
    const collective = collectiveDtype.nodes.find(
      (node) => node.nodeId === "synchronize",
    );
    if (collective?.kind !== "all-reduce") {
      throw new Error("missing collective");
    }
    collective.dtype = "i32";
    collective.numericalPolicy = "rank-order-wrapping-32";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      collectiveDtype,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidBinding);
  });

  it("rejects invalid collective ranks and numerical policies", async () => {
    const artifacts = await semantic();
    const rank = clone(program(artifacts));
    const rankCollective = rank.nodes.find(
      (node) => node.nodeId === "synchronize",
    );
    if (rankCollective?.kind !== "all-reduce") {
      throw new Error("missing collective");
    }
    rankCollective.participants = [wire("0"), wire("2")];
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      rank,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidCollective);

    const policy = clone(program(artifacts));
    const policyCollective = policy.nodes.find(
      (node) => node.nodeId === "synchronize",
    );
    if (policyCollective?.kind !== "all-reduce") {
      throw new Error("missing collective");
    }
    policyCollective.numericalPolicy = "exact-32-bit";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      policy,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const partialElement = clone(program(artifacts));
    partialElement.resources.find(
      (resource) => resource.resourceId === "bucket",
    )!.byteLength = wire("23");
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      partialElement,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.invalidCollective);
  });

  it("enforces graph resource ceilings and closed records", async () => {
    const artifacts = await semantic();
    const oversized = clone(program(artifacts));
    oversized.resources = Array.from(
      { length: HOST_GRAPH_MAX_RESOURCES + 1 },
      (_, index) => ({
        resourceId: `r${index}`,
        role: index === 0 ? "output" as const : "temporary" as const,
        multiplicity: "per-rank" as const,
        initialization: "zero-fill" as const,
        dtype: "f32" as const,
        byteLength: wire("24"),
        alignmentBytes: 4,
      }),
    );
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      oversized,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.resourceLimit);

    const excessiveRankLocalBytes = clone(program(artifacts));
    excessiveRankLocalBytes.resources[0]!.byteLength = parseWireU64(
      HOST_GRAPH_MAX_TOTAL_RESOURCE_BYTES.toString(),
    );
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      excessiveRankLocalBytes,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.resourceLimit);

    const sharedResource = clone(program(artifacts));
    sharedResource.resources[0]!.multiplicity = "shared" as "per-rank";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      sharedResource,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const wrongInitialization = clone(program(artifacts));
    wrongInitialization.resources.find(
      (resource) => resource.resourceId === "input",
    )!.initialization = "zero-fill";
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      wrongInitialization,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unsupportedProfile);

    const unknown = clone(program(artifacts)) as unknown as
      HostGraphProgram & { surprise?: boolean };
    unknown.surprise = true;
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      unknown,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(GRAPH_DIAGNOSTIC_CODES.unknownField);
  });

  it("snapshots caller input and rejects accessors without invoking them", async () => {
    const artifacts = await semantic();
    const mutable = clone(program(artifacts));
    const pending = createVerifiedHostGraphArtifact(mutable, {
      ...artifactsFor(artifacts),
    });
    mutable.nodes[0]!.dependsOn = ["mutated"];
    expect((await pending).graphSemanticHash).toMatch(/^[0-9a-f]{64}$/u);

    let reads = 0;
    const hostile = clone(program(artifacts));
    Object.defineProperty(hostile, "rankCount", {
      enumerable: true,
      get() {
        reads += 1;
        return "2";
      },
    });
    expect((await diagnostic(() => createVerifiedHostGraphArtifact(
      hostile,
      artifactsFor(artifacts),
    ))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);
    expect(reads).toBe(0);

    const hostileOptions = {
      ...artifactsFor(artifacts),
    };
    Object.defineProperty(hostileOptions, "artifactId", {
      enumerable: true,
      get() {
        reads += 1;
        return "hostile";
      },
    });
    await expect(createVerifiedHostGraphArtifact(
      program(artifacts),
      hostileOptions,
    )).rejects.toThrow(/enumerable data property/u);
    expect(reads).toBe(0);
  });
});
