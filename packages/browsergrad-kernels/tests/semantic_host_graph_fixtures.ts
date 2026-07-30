import type {
  HostGraphProgram,
} from "@unlocalhosted/browsergrad-semantic-core/graph";
import {
  createVerifiedViewCopyArtifacts,
  type VerifiedViewCopyArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  BUILTIN_DTYPES,
  evaluateDimExpr,
  layoutArtifactPayload,
  type BuiltinDTypeId,
  type DimExpr,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  parseWireI64,
  parseWireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

const MAX_FIXTURE_BYTES = 1_048_576;

export function sharedConditionalRepeatFeedbackProgram(): HostGraphProgram {
  const wire = (value: number) => parseWireU64(String(value));
  const resource = (
    resourceId: string,
    role: "input" | "temporary" | "output",
    dtype: "f32" | "u32" | "u8",
  ) => ({
    resourceId,
    role,
    multiplicity: "per-rank" as const,
    initialization: role === "input"
      ? "external-input" as const
      : "zero-fill" as const,
    dtype,
    byteLength: wire(4),
    alignmentBytes: dtype === "u8" ? 1 : 4,
  });
  return {
    kind: "host-graph",
    version: { major: 1, minor: 29 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire(2),
    resources: [
      resource("selection-input", "input", "u32"),
      resource("selection", "temporary", "u32"),
      resource("value-input", "input", "f32"),
      resource("value-output", "output", "f32"),
      resource("then-input", "input", "u8"),
      resource("else-input", "input", "u8"),
      resource("branch-output", "output", "u8"),
    ],
    nodes: [
      {
        nodeId: "produce-selection",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "selection-input",
        destinationResourceId: "selection",
        mode: "whole-allocation-bytes-per-rank",
      },
      {
        nodeId: "initialize-value",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "value-input",
        destinationResourceId: "value-output",
        mode: "whole-allocation-bytes-per-rank",
      },
      {
        nodeId: "repeat-value",
        kind: "repeat",
        dependsOn: ["initialize-value", "produce-selection"],
        iterationSource: {
          resourceId: "selection",
          rank: wire(0),
          mode: "u32-count",
        },
        maxIterationCount: wire(2),
        body: [{
          nodeId: "sum-value",
          kind: "all-reduce",
          dependsOn: [],
          resourceId: "value-output",
          reduction: "sum",
          dtype: "f32",
          participants: [wire(0), wire(1)],
          numericalPolicy: "rank-order-f32",
          result: "replicated-to-all-participants",
        }],
        mode: "resource-u32-count-sequential",
      },
      {
        nodeId: "choose-branch",
        kind: "conditional",
        dependsOn: ["produce-selection"],
        predicate: {
          resourceId: "selection",
          rank: wire(0),
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
        nodeId: "materialize-value",
        kind: "materialize",
        dependsOn: ["repeat-value"],
        resourceId: "value-output",
        mode: "host-readback-after-graph-success",
      },
      {
        nodeId: "materialize-branch",
        kind: "materialize",
        dependsOn: ["choose-branch"],
        resourceId: "branch-output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

export function sequentialConditionalRepeatFeedbackProgram(): HostGraphProgram {
  const wire = (value: number) => parseWireU64(String(value));
  const resource = (
    resourceId: string,
    role: "input" | "temporary" | "output",
    dtype: "f32" | "u32",
  ) => ({
    resourceId,
    role,
    multiplicity: "per-rank" as const,
    initialization: role === "input"
      ? "external-input" as const
      : "zero-fill" as const,
    dtype,
    byteLength: wire(4),
    alignmentBytes: 4,
  });
  return {
    kind: "host-graph",
    version: { major: 1, minor: 30 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire(2),
    resources: [
      resource("predicate-input", "input", "u32"),
      resource("predicate", "temporary", "u32"),
      resource("then-count-input", "input", "u32"),
      resource("else-count-input", "input", "u32"),
      resource("repeat-count", "temporary", "u32"),
      resource("value-input", "input", "f32"),
      resource("value-output", "output", "f32"),
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
        nodeId: "initialize-value",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "value-input",
        destinationResourceId: "value-output",
        mode: "whole-allocation-bytes-per-rank",
      },
      {
        nodeId: "choose-repeat-count",
        kind: "conditional",
        dependsOn: ["produce-predicate"],
        predicate: {
          resourceId: "predicate",
          rank: wire(0),
          mode: "u32-nonzero",
        },
        thenBody: [{
          nodeId: "copy-then-count",
          kind: "copy",
          dependsOn: [],
          sourceResourceId: "then-count-input",
          destinationResourceId: "repeat-count",
          mode: "whole-allocation-bytes-per-rank",
        }],
        elseBody: [{
          nodeId: "copy-else-count",
          kind: "copy",
          dependsOn: [],
          sourceResourceId: "else-count-input",
          destinationResourceId: "repeat-count",
          mode: "whole-allocation-bytes-per-rank",
        }],
        mode: "resource-u32-branch-sequential",
      },
      {
        nodeId: "repeat-value",
        kind: "repeat",
        dependsOn: ["initialize-value", "choose-repeat-count"],
        iterationSource: {
          resourceId: "repeat-count",
          rank: wire(0),
          mode: "u32-count",
        },
        maxIterationCount: wire(2),
        body: [{
          nodeId: "sum-value",
          kind: "all-reduce",
          dependsOn: [],
          resourceId: "value-output",
          reduction: "sum",
          dtype: "f32",
          participants: [wire(0), wire(1)],
          numericalPolicy: "rank-order-f32",
          result: "replicated-to-all-participants",
        }],
        mode: "resource-u32-count-sequential",
      },
      {
        nodeId: "materialize-value",
        kind: "materialize",
        dependsOn: ["repeat-value"],
        resourceId: "value-output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

export function sequentialConditionalDynamicDispatchProgram(
  artifacts: VerifiedViewCopyArtifacts,
): HostGraphProgram {
  const wire = (value: number) => parseWireU64(String(value));
  const resource = (
    resourceId: string,
    role: "input" | "temporary" | "output",
    dtype: "f32" | "u32",
    byteLength = 4,
  ) => ({
    resourceId,
    role,
    multiplicity: "per-rank" as const,
    initialization: role === "input"
      ? "external-input" as const
      : "zero-fill" as const,
    dtype,
    byteLength: wire(byteLength),
    alignmentBytes: 4,
  });
  return {
    kind: "host-graph",
    version: { major: 1, minor: 31 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire(1),
    resources: [
      resource("predicate-input", "input", "u32"),
      resource("predicate", "temporary", "u32"),
      resource("then-count-input", "input", "u32"),
      resource("else-count-input", "input", "u32"),
      resource("launch-count", "temporary", "u32"),
      resource("value-input", "input", "f32", 8),
      resource("value-output", "output", "f32", 8),
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
          rank: wire(0),
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
            graphResourceId: "value-input",
          },
          {
            semanticResourceId: artifacts.destination.viewId,
            graphResourceId: "value-output",
          },
        ],
        launchSource: {
          resourceId: "launch-count",
          rank: wire(0),
          mode: "u32-prefix-element-count",
        },
        maxElementCount: wire(2),
        mode: "resource-u32-prefix-elements",
      },
      {
        nodeId: "materialize-value",
        kind: "materialize",
        dependsOn: ["copy-selected-prefix"],
        resourceId: "value-output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

export function sequentialConditionalRectangularDispatchProgram(
  artifacts: VerifiedViewCopyArtifacts,
): HostGraphProgram {
  const wire = (value: number) => parseWireU64(String(value));
  const shape = [2, 3] as const;
  const resource = (
    resourceId: string,
    role: "input" | "temporary" | "output",
    dtype: "f32" | "u32",
    byteLength = 4,
  ) => ({
    resourceId,
    role,
    multiplicity: "per-rank" as const,
    initialization: role === "input"
      ? "external-input" as const
      : "zero-fill" as const,
    dtype,
    byteLength: wire(byteLength),
    alignmentBytes: 4,
  });
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
    version: { major: 1, minor: 32 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire(1),
    resources: [
      resource("predicate-input", "input", "u32"),
      resource("predicate", "temporary", "u32"),
      ...shape.flatMap((_, axis) => [
        resource(`then-extent-input-${axis}`, "input", "u32"),
        resource(`else-extent-input-${axis}`, "input", "u32"),
        resource(`extent-${axis}`, "temporary", "u32"),
      ]),
      resource("value-input", "input", "f32", 24),
      resource("value-output", "output", "f32", 24),
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
          rank: wire(0),
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
            graphResourceId: "value-input",
          },
          {
            semanticResourceId: artifacts.destination.viewId,
            graphResourceId: "value-output",
          },
        ],
        launchSources: shape.map((_, axis) => ({
          axis,
          resourceId: `extent-${axis}`,
          rank: wire(0),
          mode: "u32-prefix-extent" as const,
        })),
        maxExtents: shape.map((extent) => wire(extent)),
        mode: "resource-u32-rectangular-prefix",
      },
      {
        nodeId: "materialize-value",
        kind: "materialize",
        dependsOn: ["copy-selected-rectangle"],
        resourceId: "value-output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

export async function createVerifiedSignedReverseViewCopyArtifacts(
  dtype: BuiltinDTypeId,
  rank: number,
): Promise<VerifiedViewCopyArtifacts> {
  if (!Number.isInteger(rank) || rank < 1 || rank > 8) {
    throw new Error("signed reverse fixture rank must be an integer from 1 to 8");
  }
  const definition = BUILTIN_DTYPES[dtype];
  const elementBytes = definition.storageBits / 8;
  const elementCount = 2 ** rank;
  const byteLength = elementCount * elementBytes;
  const shape = () => Array.from(
    { length: rank },
    () => dimension(2),
  );
  const strides = (sign: 1 | -1) => Array.from(
    { length: rank },
    (_, axis) => dimension(sign * (2 ** (rank - axis - 1))),
  );
  const allocation = () => ({
    byteLength: dimension(byteLength),
    memorySpace: { kind: "global" as const },
    alignmentBytes: definition.alignmentBytes,
  });
  return createVerifiedViewCopyArtifacts({
    dtype,
    symbols: [],
    constraints: [],
    source: {
      layout: {
        kind: "strided",
        shape: shape(),
        strides: strides(-1),
      },
      allocation: allocation(),
      byteOffset: dimension(byteLength - elementBytes),
      requiredAlignmentBytes: definition.alignmentBytes,
    },
    destination: {
      layout: {
        kind: "strided",
        shape: shape(),
        strides: strides(1),
      },
      allocation: allocation(),
      byteOffset: dimension(0),
      requiredAlignmentBytes: definition.alignmentBytes,
    },
    invalidSource: { kind: "reject" },
  });
}

export function singleViewCopyGraphProgram(
  artifacts: VerifiedViewCopyArtifacts,
  dtype: BuiltinDTypeId,
  byteLength: number,
  rankCount: number,
): HostGraphProgram {
  if (
    !Number.isSafeInteger(byteLength) ||
    byteLength <= 0 ||
    byteLength > MAX_FIXTURE_BYTES ||
    byteLength % 4 !== 0
  ) {
    throw new Error(
      `view-copy graph fixture byte length must be a positive whole-word value no larger than ${MAX_FIXTURE_BYTES}`,
    );
  }
  if (!Number.isSafeInteger(rankCount) || rankCount < 1 || rankCount > 64) {
    throw new Error("view-copy graph fixture rank count must be from 1 to 64");
  }
  const layout = layoutArtifactPayload(artifacts.layout);
  const sourceView = layout.views.find((view) =>
    view.viewId === artifacts.source.viewId);
  const destinationView = layout.views.find((view) =>
    view.viewId === artifacts.destination.viewId);
  const allocations = new Map(
    layout.allocations.map((allocation) => [
      allocation.allocationId,
      allocation,
    ]),
  );
  const sourceAllocation = sourceView === undefined
    ? undefined
    : allocations.get(sourceView.allocationId);
  const destinationAllocation = destinationView === undefined
    ? undefined
    : allocations.get(destinationView.allocationId);
  if (
    sourceView?.dtype !== dtype ||
    destinationView?.dtype !== dtype ||
    !hasExactByteLength(sourceAllocation?.byteLength, byteLength) ||
    !hasExactByteLength(destinationAllocation?.byteLength, byteLength)
  ) {
    throw new Error(
      "view-copy graph fixture dtype and byte length must match both verified artifact roles",
    );
  }
  const alignmentBytes = BUILTIN_DTYPES[dtype].alignmentBytes;
  return {
    kind: "host-graph",
    version: { major: 1, minor: 2 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: parseWireU64(String(rankCount)),
    resources: [
      {
        resourceId: "input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype,
        byteLength: parseWireU64(String(byteLength)),
        alignmentBytes,
      },
      {
        resourceId: "output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype,
        byteLength: parseWireU64(String(byteLength)),
        alignmentBytes,
      },
    ],
    nodes: [
      {
        nodeId: "copy",
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
            graphResourceId: "output",
          },
        ],
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

export function patternedStorageBytes(
  byteLength: number,
  seed: number,
): Uint8Array {
  if (
    !Number.isSafeInteger(byteLength) ||
    byteLength <= 0 ||
    byteLength > MAX_FIXTURE_BYTES
  ) {
    throw new Error(
      `storage fixture byte length must be from 1 to ${MAX_FIXTURE_BYTES}`,
    );
  }
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xff) {
    throw new Error("storage fixture seed must be one byte");
  }
  return Uint8Array.from(
    { length: byteLength },
    (_, index) => (seed + (index * 37)) & 0xff,
  );
}

export function reverseStorageElements(
  bytes: Uint8Array,
  elementBytes: number,
): Uint8Array {
  if (
    !Number.isSafeInteger(elementBytes) ||
    elementBytes <= 0 ||
    bytes.byteLength === 0 ||
    bytes.byteLength > MAX_FIXTURE_BYTES ||
    bytes.byteLength % elementBytes !== 0
  ) {
    throw new Error("reverse storage fixture requires complete bounded elements");
  }
  const output = new Uint8Array(bytes.byteLength);
  const elementCount = bytes.byteLength / elementBytes;
  for (let destination = 0; destination < elementCount; destination += 1) {
    const source = elementCount - destination - 1;
    output.set(
      bytes.subarray(
        source * elementBytes,
        (source + 1) * elementBytes,
      ),
      destination * elementBytes,
    );
  }
  return output;
}

function dimension(value: number): DimExpr {
  return { kind: "const", value: parseWireI64(String(value)) };
}

function hasExactByteLength(
  expression: DimExpr | undefined,
  byteLength: number,
): boolean {
  if (expression === undefined) return false;
  const evaluated = evaluateDimExpr(expression);
  return evaluated.kind === "resolved" &&
    evaluated.value === BigInt(byteLength);
}
