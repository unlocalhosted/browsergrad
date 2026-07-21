/*
 * Realm-neutral convergence fixtures for browser-authorized CuTe view-copy
 * payloads. Keep this module limited to immutable JavaScript data so Node and
 * browser tests reconstruct authority-independent semantic artifacts from the
 * same exact contracts.
 */

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

interface ConvergenceFixtureSpec {
  readonly caseId: string;
  readonly layoutScope: string;
  readonly kernelScope: string;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly shape: readonly string[];
  readonly sourceStrides: readonly string[];
  readonly destinationStrides: readonly string[];
  readonly allocationByteLength: string;
  readonly sourceWords: readonly number[];
  readonly destinationWords: readonly number[];
}

function createConvergenceFixture(spec: ConvergenceFixtureSpec) {
  if ((spec.shape.length !== 2 && spec.shape.length !== 3) ||
      spec.sourceStrides.length !== spec.shape.length ||
      spec.destinationStrides.length !== spec.shape.length ||
      [...spec.shape, ...spec.sourceStrides, ...spec.destinationStrides]
        .some((value) => !/^[1-9][0-9]*$/u.test(value))) {
    throw new Error(`invalid positive flat fixture geometry for ${spec.caseId}`);
  }
  const elementCount = spec.shape.reduce((product, extent) => product * Number(extent), 1);
  if (!Number.isSafeInteger(elementCount) ||
      spec.sourceWords.length !== elementCount ||
      spec.destinationWords.length !== elementCount ||
      Number(spec.allocationByteLength) !== (elementCount + 2) * Uint32Array.BYTES_PER_ELEMENT) {
    throw new Error(`invalid storage/case cardinality for ${spec.caseId}`);
  }
  if ([
    spec.layoutScope,
    spec.kernelScope,
    spec.layoutSemanticHash,
    spec.kernelSemanticHash,
  ].some((value) => !/^[0-9a-f]{64}$/u.test(value))) {
    throw new Error(`invalid pinned semantic identity for ${spec.caseId}`);
  }
  const entityId = (kind: string, ordinal: number) =>
    `bg.entity.${kind}.scope-sha256.${spec.layoutScope}.ordinal.${ordinal}`;
  const sourceAllocationId = entityId("allocation", 0);
  const destinationAllocationId = entityId("allocation", 1);
  const sourceAliasSetId = entityId("alias-set", 0);
  const destinationAliasSetId = entityId("alias-set", 1);
  const sourceIndexMapId = entityId("index-map", 0);
  const destinationIndexMapId = entityId("index-map", 1);
  const sourceViewId = entityId("view", 0);
  const destinationViewId = entityId("view", 1);
  const operationId =
    `bg.entity.kernel-operation.scope-sha256.${spec.kernelScope}.ordinal.0`;
  const freshLogicalShape = () => Object.freeze(spec.shape.map(constant));
  const freshInBounds = () => Object.freeze({
    kind: "and" as const,
    values: Object.freeze(spec.shape.flatMap((extent, axis) => [
      lessEqual(constant("0"), coordinate(axis)),
      lessEqual(add(coordinate(axis), constant("1")), constant(extent)),
    ])),
  });
  const location = (strides: readonly string[]) => add(...strides.map((stride, axis) => (
    stride === "1"
      ? coordinate(axis)
      : multiply(coordinate(axis), constant(stride))
  )));
  const construction = Object.freeze({
    dtype: "f32" as const,
    symbols: Object.freeze([]),
    constraints: Object.freeze([]),
    source: Object.freeze({
      layout: Object.freeze({
        kind: "strided" as const,
        shape: freshLogicalShape(),
        strides: Object.freeze(spec.sourceStrides.map(constant)),
      }),
      allocation: Object.freeze({
        byteLength: constant(spec.allocationByteLength),
        memorySpace: Object.freeze({ kind: "global" as const }),
        alignmentBytes: 4,
      }),
      byteOffset: constant("4"),
      requiredAlignmentBytes: 4,
    }),
    destination: Object.freeze({
      layout: Object.freeze({
        kind: "strided" as const,
        shape: freshLogicalShape(),
        strides: Object.freeze(spec.destinationStrides.map(constant)),
      }),
      allocation: Object.freeze({
        byteLength: constant(spec.allocationByteLength),
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
        byteLength: constant(spec.allocationByteLength),
        memorySpace: Object.freeze({ kind: "global" as const }),
        alignmentBytes: 4,
        aliasSetId: sourceAliasSetId,
      }),
      Object.freeze({
        allocationId: destinationAllocationId,
        byteLength: constant(spec.allocationByteLength),
        memorySpace: Object.freeze({ kind: "global" as const }),
        alignmentBytes: 4,
        aliasSetId: destinationAliasSetId,
      }),
    ]),
    indexMaps: Object.freeze([
      Object.freeze({
        indexMapId: sourceIndexMapId,
        coordinateRank: spec.shape.length,
        locationUnit: "element" as const,
        location: location(spec.sourceStrides),
        inBounds: freshInBounds(),
      }),
      Object.freeze({
        indexMapId: destinationIndexMapId,
        coordinateRank: spec.shape.length,
        locationUnit: "element" as const,
        location: location(spec.destinationStrides),
        inBounds: freshInBounds(),
      }),
    ]),
    views: Object.freeze([
      Object.freeze({
        viewId: sourceViewId,
        allocationId: sourceAllocationId,
        dtype: "f32" as const,
        byteOffset: constant("4"),
        shape: freshLogicalShape(),
        indexMapId: sourceIndexMapId,
        requiredAlignmentBytes: 4,
      }),
      Object.freeze({
        viewId: destinationViewId,
        allocationId: destinationAllocationId,
        dtype: "f32" as const,
        byteOffset: constant("4"),
        shape: freshLogicalShape(),
        indexMapId: destinationIndexMapId,
        requiredAlignmentBytes: 4,
      }),
    ]),
  });
  const kernelPayload = Object.freeze({
    layoutSemanticHash: spec.layoutSemanticHash,
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
  const initialDestinationWords = Object.freeze([
    0x13579bdf,
    ...Array.from({ length: spec.destinationWords.length }, () => 0xdeadbeef),
    0x2468ace0,
  ]);
  return Object.freeze({
    schema: "browsergrad.compiler.cpp-cute-browser-view-copy-convergence@1" as const,
    caseId: spec.caseId,
    entryId: `bg.cpp.entry.sha256.${"b".repeat(64)}`,
    claims: Object.freeze({
      productionBrowserCompileObserved: false as const,
      backendExecutionAuthorizationMinted: false as const,
      cudaLiteRunnerUsed: false as const,
    }),
    storage: Object.freeze({
      sourceAllocationByteLength: spec.allocationByteLength,
      destinationAllocationByteLength: spec.allocationByteLength,
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
      layoutSemanticHash: spec.layoutSemanticHash,
      kernelSemanticHash: spec.kernelSemanticHash,
      source: Object.freeze({
        allocationId: sourceAllocationId,
        indexMapId: sourceIndexMapId,
        viewId: sourceViewId,
      }),
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
        ...spec.sourceWords,
        0xd4c3b2a1,
      ]),
      initialDestinationWords,
      destinationWords: Object.freeze([
        initialDestinationWords[0]!,
        ...spec.destinationWords,
        initialDestinationWords.at(-1)!,
      ]),
      cpuTrace: Object.freeze({
        logicalShape: Object.freeze([...spec.shape]),
        elementCount: String(elementCount),
        readElements: String(elementCount),
        filledElements: "0",
        bytesRead: String(elementCount * 4),
        bytesWritten: String(elementCount * 4),
      }),
    }),
  });
}

const FLOAT_1_TO_24_WORDS = Object.freeze([
  0x3f800000, 0x40000000, 0x40400000, 0x40800000,
  0x40a00000, 0x40c00000, 0x40e00000, 0x41000000,
  0x41100000, 0x41200000, 0x41300000, 0x41400000,
  0x41500000, 0x41600000, 0x41700000, 0x41800000,
  0x41880000, 0x41900000, 0x41980000, 0x41a00000,
  0x41a80000, 0x41b00000, 0x41b80000, 0x41c00000,
]);

export const CPP_CUTE_BROWSER_VIEW_COPY_RANK2_CONVERGENCE_FIXTURE =
  createConvergenceFixture({
    caseId: "canonical-rank2-cute-view-copy-payload",
    layoutScope: "5361780f48f83d4ee2954aaa99251c2bdeb727c2fdc5a3afb676163ef709d509",
    kernelScope: "c6bf7e791036a479a05dfa6df2b22122f9552af397d610fe0a56f8811e59867d",
    layoutSemanticHash: "5ade6e063773ba40a1046423e76776cf963544a26c7f17b301565d54a86ecdfe",
    kernelSemanticHash: "64dc9d67e4f0de9c1f7b68fa369957c9521d8bbb9aa9725ac82f0dfaa573f409",
    shape: ["3", "2"],
    sourceStrides: ["1", "3"],
    destinationStrides: ["2", "1"],
    allocationByteLength: "32",
    sourceWords: FLOAT_1_TO_24_WORDS.slice(0, 6),
    destinationWords: [
      FLOAT_1_TO_24_WORDS[0]!, FLOAT_1_TO_24_WORDS[3]!,
      FLOAT_1_TO_24_WORDS[1]!, FLOAT_1_TO_24_WORDS[4]!,
      FLOAT_1_TO_24_WORDS[2]!, FLOAT_1_TO_24_WORDS[5]!,
    ],
  });

export const CPP_CUTE_BROWSER_VIEW_COPY_RANK3_CONVERGENCE_FIXTURE =
  createConvergenceFixture({
    caseId: "canonical-rank3-cute-view-copy-payload",
    layoutScope: "959fb68682f5117b82e99a7ac87137a32d75da8225eef6152f4cba44be9c445c",
    kernelScope: "0c25c36228f02f1db997da07f28f32c3bf28f13273f769fbffedb11d91f052b6",
    layoutSemanticHash: "c2b5e8a0489bd2ee5a54d15399af95b91d9fe102aab63e450361500ffa946a6f",
    kernelSemanticHash: "e335ea9d9e9a38f591c80c737b8a33401578739e02d6892e5f1907e6b76e6ff2",
    shape: ["2", "3", "4"],
    sourceStrides: ["1", "2", "6"],
    destinationStrides: ["12", "4", "1"],
    allocationByteLength: "104",
    sourceWords: FLOAT_1_TO_24_WORDS,
    destinationWords: [
      0, 6, 12, 18, 2, 8, 14, 20, 4, 10, 16, 22,
      1, 7, 13, 19, 3, 9, 15, 21, 5, 11, 17, 23,
    ].map((index) => FLOAT_1_TO_24_WORDS[index]!),
  });

export const CPP_CUTE_BROWSER_VIEW_COPY_CONVERGENCE_FIXTURE =
  CPP_CUTE_BROWSER_VIEW_COPY_RANK2_CONVERGENCE_FIXTURE;

export const CPP_CUTE_BROWSER_VIEW_COPY_CONVERGENCE_FIXTURES = Object.freeze([
  CPP_CUTE_BROWSER_VIEW_COPY_RANK2_CONVERGENCE_FIXTURE,
  CPP_CUTE_BROWSER_VIEW_COPY_RANK3_CONVERGENCE_FIXTURE,
]);
