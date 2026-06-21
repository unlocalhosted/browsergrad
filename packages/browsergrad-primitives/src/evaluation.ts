import {
  compareSnapshot,
  createSnapshotOracle,
  SnapshotError,
  type SnapshotCompareOptions,
  type SnapshotComparison,
  type SnapshotMismatch,
  type SnapshotMismatchKind,
  type SnapshotOracle,
} from "@unlocalhosted/browsergrad-snapshots";

export {
  compareSnapshot,
  SnapshotError,
  type SnapshotCompareOptions,
  type SnapshotComparison,
  type SnapshotMismatch,
  type SnapshotMismatchKind,
};

export type SnapshotComparator = SnapshotOracle;

export function createSnapshotComparator(
  expected: unknown,
  options: SnapshotCompareOptions = {},
): SnapshotComparator {
  return createSnapshotOracle(expected, options);
}

