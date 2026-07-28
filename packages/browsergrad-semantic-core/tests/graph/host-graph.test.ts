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
    future.version.minor = 5 as typeof future.version.minor;
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
