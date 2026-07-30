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
