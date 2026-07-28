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
) {
  return {
    resourceId,
    role,
    multiplicity: "per-rank" as const,
    initialization: role === "input"
      ? "external-input" as const
      : "zero-fill" as const,
    dtype,
    byteLength: wire("8"),
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
