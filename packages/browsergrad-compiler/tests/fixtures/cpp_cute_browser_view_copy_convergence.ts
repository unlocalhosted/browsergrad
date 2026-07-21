/*
 * Realm-neutral convergence fixture for the first browser-authorized CuTe
 * view-copy. Keep this module limited to immutable JavaScript data so Node and
 * browser tests reconstruct authority-independent semantic artifacts from the
 * same exact contract.
 */

const LAYOUT_SCOPE = "5361780f48f83d4ee2954aaa99251c2bdeb727c2fdc5a3afb676163ef709d509";
const KERNEL_SCOPE = "c6bf7e791036a479a05dfa6df2b22122f9552af397d610fe0a56f8811e59867d";
const entityId = (kind: string, ordinal: number) =>
  `bg.entity.${kind}.scope-sha256.${LAYOUT_SCOPE}.ordinal.${ordinal}`;
const constant = (value: string) => Object.freeze({ kind: "const" as const, value });
const coordinate = (axis: number) => Object.freeze({ kind: "coordinate" as const, axis });
const add = (...terms: readonly object[]) => Object.freeze({
  kind: "add" as const,
  terms: Object.freeze(terms),
});
const multiply = (lhs: object, rhs: object) => Object.freeze({
  kind: "mul" as const,
  lhs,
  rhs,
});
const lessEqual = (lhs: object, rhs: object) => Object.freeze({
  kind: "lessEqual" as const,
  lhs,
  rhs,
});

const sourceAllocationId = entityId("allocation", 0);
const destinationAllocationId = entityId("allocation", 1);
const sourceAliasSetId = entityId("alias-set", 0);
const destinationAliasSetId = entityId("alias-set", 1);
const sourceIndexMapId = entityId("index-map", 0);
const destinationIndexMapId = entityId("index-map", 1);
const sourceViewId = entityId("view", 0);
const destinationViewId = entityId("view", 1);
const operationId =
  `bg.entity.kernel-operation.scope-sha256.${KERNEL_SCOPE}.ordinal.0`;

const logicalShape = Object.freeze([constant("3"), constant("2")]);
const inBounds = Object.freeze({
  kind: "and" as const,
  values: Object.freeze([
    lessEqual(constant("0"), coordinate(0)),
    lessEqual(add(coordinate(0), constant("1")), constant("3")),
    lessEqual(constant("0"), coordinate(1)),
    lessEqual(add(coordinate(1), constant("1")), constant("2")),
  ]),
});

const construction = Object.freeze({
  dtype: "f32" as const,
  symbols: Object.freeze([]),
  constraints: Object.freeze([]),
  source: Object.freeze({
    layout: Object.freeze({
      kind: "strided" as const,
      shape: Object.freeze([constant("3"), constant("2")]),
      strides: Object.freeze([constant("1"), constant("3")]),
    }),
    allocation: Object.freeze({
      byteLength: constant("32"),
      memorySpace: Object.freeze({ kind: "global" as const }),
      alignmentBytes: 4,
    }),
    byteOffset: constant("4"),
    requiredAlignmentBytes: 4,
  }),
  destination: Object.freeze({
    layout: Object.freeze({
      kind: "strided" as const,
      shape: Object.freeze([constant("3"), constant("2")]),
      strides: Object.freeze([constant("2"), constant("1")]),
    }),
    allocation: Object.freeze({
      byteLength: constant("32"),
      memorySpace: Object.freeze({ kind: "global" as const }),
      alignmentBytes: 4,
    }),
    byteOffset: constant("4"),
    requiredAlignmentBytes: 4,
  }),
  invalidSource: Object.freeze({ kind: "reject" as const }),
});

const layoutPayload = Object.freeze({
  symbols: Object.freeze([]),
  constraints: Object.freeze([]),
  allocations: Object.freeze([
    Object.freeze({
      allocationId: sourceAllocationId,
      byteLength: constant("32"),
      memorySpace: Object.freeze({ kind: "global" as const }),
      alignmentBytes: 4,
      aliasSetId: sourceAliasSetId,
    }),
    Object.freeze({
      allocationId: destinationAllocationId,
      byteLength: constant("32"),
      memorySpace: Object.freeze({ kind: "global" as const }),
      alignmentBytes: 4,
      aliasSetId: destinationAliasSetId,
    }),
  ]),
  indexMaps: Object.freeze([
    Object.freeze({
      indexMapId: sourceIndexMapId,
      coordinateRank: 2,
      locationUnit: "element" as const,
      location: add(
        coordinate(0),
        multiply(coordinate(1), constant("3")),
      ),
      inBounds,
    }),
    Object.freeze({
      indexMapId: destinationIndexMapId,
      coordinateRank: 2,
      locationUnit: "element" as const,
      location: add(
        multiply(coordinate(0), constant("2")),
        coordinate(1),
      ),
      inBounds,
    }),
  ]),
  views: Object.freeze([
    Object.freeze({
      viewId: sourceViewId,
      allocationId: sourceAllocationId,
      dtype: "f32" as const,
      byteOffset: constant("4"),
      shape: logicalShape,
      indexMapId: sourceIndexMapId,
      requiredAlignmentBytes: 4,
    }),
    Object.freeze({
      viewId: destinationViewId,
      allocationId: destinationAllocationId,
      dtype: "f32" as const,
      byteOffset: constant("4"),
      shape: logicalShape,
      indexMapId: destinationIndexMapId,
      requiredAlignmentBytes: 4,
    }),
  ]),
});

const kernelPayload = Object.freeze({
  layoutSemanticHash:
    "5ade6e063773ba40a1046423e76776cf963544a26c7f17b301565d54a86ecdfe",
  operations: Object.freeze([
    Object.freeze({
      operationId,
      kind: "view-copy" as const,
      version: Object.freeze({ major: 1 as const, minor: 0 as const }),
      dtype: "f32" as const,
      source: Object.freeze({
        viewId: sourceViewId,
        access: "read" as const,
        invalidSource: Object.freeze({ kind: "reject" as const }),
      }),
      destination: Object.freeze({
        viewId: destinationViewId,
        access: "write" as const,
      }),
      overlap: Object.freeze({ kind: "forbid" as const }),
    }),
  ]),
});

export const CPP_CUTE_BROWSER_VIEW_COPY_CONVERGENCE_FIXTURE = Object.freeze({
  schema: "browsergrad.compiler.cpp-cute-browser-view-copy-convergence@1" as const,
  entryId: `bg.cpp.entry.sha256.${"b".repeat(64)}`,
  claims: Object.freeze({
    productionBrowserCompileObserved: false as const,
    backendExecutionAuthorizationMinted: false as const,
    cudaLiteRunnerUsed: false as const,
  }),
  storage: Object.freeze({
    sourceAllocationByteLength: "32",
    destinationAllocationByteLength: "32",
    sourceByteOffset: "4",
    destinationByteOffset: "4",
  }),
  construction,
  constructionOptions: Object.freeze({
    producer: Object.freeze({
      id: "browsergrad.compiler.cpp-cute-view-copy-lowering",
      version: "1",
    }),
    layoutArtifactId: "authorized-cpp-cute-view-copy-layout",
    kernelArtifactId: "authorized-cpp-cute-view-copy-kernel",
  }),
  expected: Object.freeze({
    layoutSemanticHash:
      "5ade6e063773ba40a1046423e76776cf963544a26c7f17b301565d54a86ecdfe",
    kernelSemanticHash:
      "64dc9d67e4f0de9c1f7b68fa369957c9521d8bbb9aa9725ac82f0dfaa573f409",
    source: Object.freeze({ allocationId: sourceAllocationId, indexMapId: sourceIndexMapId, viewId: sourceViewId }),
    destination: Object.freeze({
      allocationId: destinationAllocationId,
      indexMapId: destinationIndexMapId,
      viewId: destinationViewId,
    }),
    operationId,
    layoutPayload,
    kernelPayload,
    sourceWords: Object.freeze([
      0xa1b2c3d4,
      0x3f800000,
      0x40000000,
      0x40400000,
      0x40800000,
      0x40a00000,
      0x40c00000,
      0xd4c3b2a1,
    ]),
    initialDestinationWords: Object.freeze([
      0x13579bdf,
      0xdeadbeef,
      0xdeadbeef,
      0xdeadbeef,
      0xdeadbeef,
      0xdeadbeef,
      0xdeadbeef,
      0x2468ace0,
    ]),
    destinationWords: Object.freeze([
      0x13579bdf,
      0x3f800000,
      0x40800000,
      0x40000000,
      0x40a00000,
      0x40400000,
      0x40c00000,
      0x2468ace0,
    ]),
    cpuTrace: Object.freeze({
      logicalShape: Object.freeze(["3", "2"]),
      elementCount: "6",
      readElements: "6",
      filledElements: "0",
      bytesRead: "24",
      bytesWritten: "24",
    }),
  }),
});
