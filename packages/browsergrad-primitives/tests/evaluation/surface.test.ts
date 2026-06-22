import { describe, expect, it } from "vitest";
import {
  compareSnapshot,
  createSnapshotComparator,
  SnapshotError,
  type SnapshotComparison,
  type SnapshotMismatch,
  type SnapshotComparator,
} from "../../src/evaluation";

describe("public surface", () => {
  it("exports snapshot comparator primitives", () => {
    expect(typeof compareSnapshot).toBe("function");
    expect(typeof createSnapshotComparator).toBe("function");
    expect(typeof SnapshotError).toBe("function");
  });

  it("types snapshot comparator results for compile-time consumers", () => {
    const comparator: SnapshotComparator = createSnapshotComparator({ loss: 1 });
    const comparison: SnapshotComparison = comparator.compare({ loss: 1 });
    const mismatch: SnapshotMismatch = {
      path: "$.loss",
      kind: "value",
      expected: "1",
      actual: "2",
    };
    expect(comparison.ok).toBe(true);
    expect(mismatch.path).toBe("$.loss");
  });
});
