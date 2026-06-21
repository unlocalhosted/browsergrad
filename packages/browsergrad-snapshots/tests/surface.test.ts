import { describe, expect, it } from "vitest";
import {
  compareSnapshot,
  createSnapshotOracle,
  SnapshotError,
  type SnapshotComparison,
  type SnapshotMismatch,
  type SnapshotOracle,
} from "../src/index";

describe("public surface", () => {
  it("exports snapshot oracle primitives", () => {
    expect(typeof compareSnapshot).toBe("function");
    expect(typeof createSnapshotOracle).toBe("function");
    expect(typeof SnapshotError).toBe("function");
  });

  it("types snapshot oracle results for compile-time consumers", () => {
    const oracle: SnapshotOracle = createSnapshotOracle({ loss: 1 });
    const comparison: SnapshotComparison = oracle.compare({ loss: 1 });
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
