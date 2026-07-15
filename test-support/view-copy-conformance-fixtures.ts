import {
  createVerifiedViewCopyArtifacts,
  type VerifiedViewCopyArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import { parseWireI64 } from "@unlocalhosted/browsergrad-semantic-core/schema";

export const VIEW_COPY_CONFORMANCE_CASE_IDS = Object.freeze([
  "rank2-transpose-control",
  "rank2-padding-exact-nan",
  "rank3-padding-exact-nan",
] as const);

export type ViewCopyConformanceCaseId = typeof VIEW_COPY_CONFORMANCE_CASE_IDS[number];

export interface ViewCopyConformanceCase {
  readonly id: ViewCopyConformanceCaseId;
  readonly artifacts: VerifiedViewCopyArtifacts;
  readonly logicalShape: readonly number[];
  /** Complete source allocation, including prefix/suffix canaries. */
  readonly sourceWords: Uint32Array;
  /** Expected complete source allocation after the read-only copy. */
  readonly expectedSourceWords: Uint32Array;
  /** Complete destination allocation before execution. */
  readonly initialDestinationWords: Uint32Array;
  /** Expected complete destination allocation after execution. */
  readonly expectedDestinationWords: Uint32Array;
  /** Root-relative word indices, in logical traversal/read order. */
  readonly expectedSourcePhysicalIndices: readonly number[];
  /** Root-relative word indices, in logical traversal/write order. */
  readonly expectedDestinationPhysicalIndices: readonly number[];
  readonly expectedReadElements: number;
  readonly expectedFilledElements: number;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
}

const EXACT_NAN_FILL = 0x7fc01234;
const SOURCE_PREFIX_CANARY = 0xa1b2c3d4;
const SOURCE_SUFFIX_CANARY = 0xd4c3b2a1;
const DESTINATION_PREFIX_CANARY = 0x13579bdf;
const DESTINATION_SUFFIX_CANARY = 0x2468ace0;
const DESTINATION_INITIAL_WORD = 0xdeadbeef;

interface CaseDefinition {
  readonly id: ViewCopyConformanceCaseId;
  readonly logicalShape: readonly number[];
  readonly sourceWords: readonly number[];
  readonly initialDestinationWords: readonly number[];
  readonly expectedDestinationWords: readonly number[];
  readonly expectedSourcePhysicalIndices: readonly number[];
  readonly expectedDestinationPhysicalIndices: readonly number[];
  readonly expectedReadElements: number;
  readonly expectedFilledElements: number;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  createArtifacts(): Promise<VerifiedViewCopyArtifacts>;
}

const TRANSPOSE_SOURCE_DATA = Object.freeze([
  0x3f800000,
  0x40000000,
  0x40400000,
  0x40800000,
  0x40a00000,
  0x40c00000,
]);
const RANK3_SOURCE_DATA = Object.freeze([
  0x41000000,
  0x41100000,
  0x41200000,
  0x41300000,
  0x41400000,
  0x41500000,
  0x41600000,
  0x41700000,
]);

const CASE_DEFINITIONS: readonly CaseDefinition[] = Object.freeze([
  Object.freeze({
    id: "rank2-transpose-control",
    logicalShape: Object.freeze([3, 2]),
    sourceWords: completeRoot(TRANSPOSE_SOURCE_DATA, SOURCE_PREFIX_CANARY, SOURCE_SUFFIX_CANARY),
    initialDestinationWords: initializedDestination(6),
    expectedDestinationWords: completeRoot(
      [
        TRANSPOSE_SOURCE_DATA[0]!,
        TRANSPOSE_SOURCE_DATA[3]!,
        TRANSPOSE_SOURCE_DATA[1]!,
        TRANSPOSE_SOURCE_DATA[4]!,
        TRANSPOSE_SOURCE_DATA[2]!,
        TRANSPOSE_SOURCE_DATA[5]!,
      ],
      DESTINATION_PREFIX_CANARY,
      DESTINATION_SUFFIX_CANARY,
    ),
    expectedSourcePhysicalIndices: frozenRange([1, 4, 2, 5, 3, 6]),
    expectedDestinationPhysicalIndices: frozenRange([1, 2, 3, 4, 5, 6]),
    expectedReadElements: 6,
    expectedFilledElements: 0,
    layoutSemanticHash: "58a894aad82f9e4e201f84342b942c23d4efc9d40b754ef873d49d8d012548f5",
    kernelSemanticHash: "a4a27d833b618da7c105280eb5ee99627ef0f254b0a95b8cf1d25c04e73244bc",
    createArtifacts: createRank2TransposeArtifacts,
  }),
  Object.freeze({
    id: "rank2-padding-exact-nan",
    logicalShape: Object.freeze([4, 5]),
    sourceWords: completeRoot(TRANSPOSE_SOURCE_DATA, SOURCE_PREFIX_CANARY, SOURCE_SUFFIX_CANARY),
    initialDestinationWords: initializedDestination(20),
    expectedDestinationWords: completeRoot(
      paddedWords2d(TRANSPOSE_SOURCE_DATA, 4, 5, 1, 1, 2, 3, EXACT_NAN_FILL),
      DESTINATION_PREFIX_CANARY,
      DESTINATION_SUFFIX_CANARY,
    ),
    expectedSourcePhysicalIndices: frozenRange([1, 2, 3, 4, 5, 6]),
    expectedDestinationPhysicalIndices: frozenRange(sequence(1, 20)),
    expectedReadElements: 6,
    expectedFilledElements: 14,
    layoutSemanticHash: "8254649ed00708241a5380c0533da769bdd3e716e9584024230729ce1e062080",
    kernelSemanticHash: "9482913f65442c162f1960ea4ee4037d0ace2c8679fb052d42b8fcb6986e92ca",
    createArtifacts: () => createPaddingArtifacts([2, 3]),
  }),
  Object.freeze({
    id: "rank3-padding-exact-nan",
    logicalShape: Object.freeze([4, 4, 4]),
    sourceWords: completeRoot(RANK3_SOURCE_DATA, SOURCE_PREFIX_CANARY, SOURCE_SUFFIX_CANARY),
    initialDestinationWords: initializedDestination(64),
    expectedDestinationWords: completeRoot(
      paddedWords3d(RANK3_SOURCE_DATA, 4, 4, 4, 1, 1, 1, 2, 2, 2, EXACT_NAN_FILL),
      DESTINATION_PREFIX_CANARY,
      DESTINATION_SUFFIX_CANARY,
    ),
    expectedSourcePhysicalIndices: frozenRange(sequence(1, 8)),
    expectedDestinationPhysicalIndices: frozenRange(sequence(1, 64)),
    expectedReadElements: 8,
    expectedFilledElements: 56,
    layoutSemanticHash: "5a527266d1ddaf4bb623d974086d25025e384de7070025e7b0885298e7d3dcc9",
    kernelSemanticHash: "c14d5567262b77d074d8c0776fb978562f46ea0335b3f28320d6cc9b7b155966",
    createArtifacts: () => createPaddingArtifacts([2, 2, 2]),
  }),
]);

export async function createViewCopyConformanceCases(): Promise<readonly ViewCopyConformanceCase[]> {
  const cases = await Promise.all(CASE_DEFINITIONS.map(async (definition) => {
    const artifacts = await definition.createArtifacts();
    requirePinnedHash(definition.id, "layout", definition.layoutSemanticHash, artifacts.layoutSemanticHash);
    requirePinnedHash(definition.id, "kernel", definition.kernelSemanticHash, artifacts.kernelSemanticHash);
    const sourceWords = Uint32Array.from(definition.sourceWords);
    return Object.freeze({
      id: definition.id,
      artifacts,
      logicalShape: definition.logicalShape,
      sourceWords,
      expectedSourceWords: cloneViewCopyConformanceWords(sourceWords),
      initialDestinationWords: Uint32Array.from(definition.initialDestinationWords),
      expectedDestinationWords: Uint32Array.from(definition.expectedDestinationWords),
      expectedSourcePhysicalIndices: definition.expectedSourcePhysicalIndices,
      expectedDestinationPhysicalIndices: definition.expectedDestinationPhysicalIndices,
      expectedReadElements: definition.expectedReadElements,
      expectedFilledElements: definition.expectedFilledElements,
      layoutSemanticHash: definition.layoutSemanticHash,
      kernelSemanticHash: definition.kernelSemanticHash,
    });
  }));
  return Object.freeze(cases);
}

export function cloneViewCopyConformanceWords(words: Uint32Array): Uint32Array {
  return new Uint32Array(words);
}

async function createRank2TransposeArtifacts(): Promise<VerifiedViewCopyArtifacts> {
  return createVerifiedViewCopyArtifacts({
    dtype: "f32",
    symbols: [],
    constraints: [],
    source: {
      layout: {
        kind: "permute",
        source: strided([2, 3]),
        axes: [1, 0],
      },
      allocation: globalAllocation(8),
      byteOffset: constant(4),
      requiredAlignmentBytes: 4,
    },
    destination: {
      layout: strided([3, 2]),
      allocation: globalAllocation(8),
      byteOffset: constant(4),
      requiredAlignmentBytes: 4,
    },
    invalidSource: { kind: "reject" },
  }, fixtureConstructionOptions("rank2-transpose-control"));
}

async function createPaddingArtifacts(sourceShape: readonly number[]): Promise<VerifiedViewCopyArtifacts> {
  const logicalShape = sourceShape.map((extent) => extent + 2);
  return createVerifiedViewCopyArtifacts({
    dtype: "f32",
    symbols: [],
    constraints: [],
    source: {
      layout: {
        kind: "pad",
        source: strided(sourceShape),
        low: sourceShape.map(() => constant(1)),
        high: sourceShape.map(() => constant(1)),
      },
      allocation: globalAllocation(elementCount(sourceShape) + 2),
      byteOffset: constant(4),
      requiredAlignmentBytes: 4,
    },
    destination: {
      layout: paddedDestination(sourceShape),
      allocation: globalAllocation(elementCount(logicalShape) + 2),
      byteOffset: constant(4),
      requiredAlignmentBytes: 4,
    },
    invalidSource: {
      kind: "fill",
      value: { kind: "float-bits", dtype: "f32", bits: "7fc01234" },
    },
  }, fixtureConstructionOptions(sourceShape.length === 2
    ? "rank2-padding-exact-nan"
    : "rank3-padding-exact-nan"));
}

function paddedDestination(sourceShape: readonly number[]) {
  const logicalShape = sourceShape.map((extent) => ({
    kind: "add" as const,
    terms: [constant(1), constant(extent), constant(1)],
  }));
  const resolvedShape = sourceShape.map((extent) => extent + 2);
  return {
    kind: "strided" as const,
    shape: logicalShape,
    strides: resolvedShape.map((_, axis) => constant(
      resolvedShape.slice(axis + 1).reduce((product, extent) => product * extent, 1),
    )),
  };
}

function fixtureConstructionOptions(id: ViewCopyConformanceCaseId) {
  return {
    producer: { id: "browsergrad-view-copy-conformance-fixtures", version: "1" },
    layoutArtifactId: `${id}-layout`,
    kernelArtifactId: `${id}-kernel`,
  } as const;
}

function strided(shape: readonly number[]) {
  return {
    kind: "strided" as const,
    shape: shape.map(constant),
    strides: shape.map((_, axis) => constant(
      shape.slice(axis + 1).reduce((product, extent) => product * extent, 1),
    )),
  };
}

function constant(value: number) {
  return { kind: "const" as const, value: parseWireI64(String(value)) };
}

function globalAllocation(wordLength: number) {
  return {
    byteLength: constant(wordLength * 4),
    memorySpace: { kind: "global" as const },
    alignmentBytes: 4,
  };
}

function completeRoot(
  interior: readonly number[],
  prefix: number,
  suffix: number,
): readonly number[] {
  return Object.freeze([prefix, ...interior, suffix]);
}

function initializedDestination(interiorLength: number): readonly number[] {
  return completeRoot(
    Array.from({ length: interiorLength }, () => DESTINATION_INITIAL_WORD),
    DESTINATION_PREFIX_CANARY,
    DESTINATION_SUFFIX_CANARY,
  );
}

function frozenRange(values: readonly number[]): readonly number[] {
  return Object.freeze([...values]);
}

function sequence(first: number, last: number): readonly number[] {
  return Array.from({ length: last - first + 1 }, (_, index) => first + index);
}

function elementCount(shape: readonly number[]): number {
  return shape.reduce((product, extent) => product * extent, 1);
}

function paddedWords2d(
  source: readonly number[],
  rows: number,
  columns: number,
  lowRow: number,
  lowColumn: number,
  sourceRows: number,
  sourceColumns: number,
  fill: number,
): readonly number[] {
  return Array.from({ length: rows * columns }, (_, flat) => {
    const row = Math.floor(flat / columns);
    const column = flat % columns;
    const sourceRow = row - lowRow;
    const sourceColumn = column - lowColumn;
    return sourceRow >= 0 && sourceRow < sourceRows && sourceColumn >= 0 && sourceColumn < sourceColumns
      ? source[sourceRow * sourceColumns + sourceColumn]!
      : fill;
  });
}

function paddedWords3d(
  source: readonly number[],
  depth: number,
  rows: number,
  columns: number,
  lowDepth: number,
  lowRow: number,
  lowColumn: number,
  sourceDepth: number,
  sourceRows: number,
  sourceColumns: number,
  fill: number,
): readonly number[] {
  return Array.from({ length: depth * rows * columns }, (_, flat) => {
    const plane = rows * columns;
    const z = Math.floor(flat / plane);
    const remainder = flat % plane;
    const row = Math.floor(remainder / columns);
    const column = remainder % columns;
    const sourceZ = z - lowDepth;
    const sourceRow = row - lowRow;
    const sourceColumn = column - lowColumn;
    return sourceZ >= 0 && sourceZ < sourceDepth &&
      sourceRow >= 0 && sourceRow < sourceRows &&
      sourceColumn >= 0 && sourceColumn < sourceColumns
      ? source[(sourceZ * sourceRows + sourceRow) * sourceColumns + sourceColumn]!
      : fill;
  });
}

function requirePinnedHash(
  caseId: ViewCopyConformanceCaseId,
  kind: "layout" | "kernel",
  expected: string,
  actual: string,
): void {
  if (actual !== expected) {
    throw new Error(`${caseId} ${kind} semantic hash drifted: expected ${expected}, received ${actual}`);
  }
}
