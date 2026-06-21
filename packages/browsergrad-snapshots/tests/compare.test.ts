import { describe, expect, it } from "vitest";
import { compareSnapshot, createSnapshotOracle, SnapshotError } from "../src/index";

describe("compareSnapshot", () => {
  it("passes nested numeric snapshots within absolute and relative tolerance", () => {
    const oracle = createSnapshotOracle(
      {
        loss: 1.25,
        logits: [0.1, 0.2, 0.3],
        metrics: { reward: 1000, split: "sft" },
      },
      { absoluteTolerance: 1e-4, relativeTolerance: 1e-3 },
    );

    expect(
      oracle.compare({
        loss: 1.25005,
        logits: [0.10001, 0.19995, 0.30002],
        metrics: { reward: 1000.5, split: "sft" },
      }),
    ).toEqual({ ok: true, mismatches: [] });
  });

  it("reports deterministic missing, unexpected, shape, and value mismatches", () => {
    const result = compareSnapshot(
      {
        a: [1, 4, 5],
        b: {},
        extra: "nope",
      },
      {
        a: [1, 2],
        b: { c: 3 },
        d: true,
      },
      { absoluteTolerance: 0.1 },
    );

    expect(result.ok).toBe(false);
    expect(result.mismatches).toEqual([
      {
        path: "$.a.length",
        kind: "shape",
        expected: "2",
        actual: "3",
      },
      {
        path: "$.a[1]",
        kind: "value",
        expected: "2",
        actual: "4",
        delta: 2,
        tolerance: 0.1,
      },
      { path: "$.b.c", kind: "missing", expected: "3" },
      { path: "$.d", kind: "missing", expected: "true" },
      { path: "$.extra", kind: "unexpected", actual: "\"nope\"" },
    ]);
  });

  it("rejects non-finite values and invalid tolerance options", () => {
    expect(
      compareSnapshot(Number.NaN, 1, { absoluteTolerance: 0 }).mismatches,
    ).toEqual([
      {
        path: "$",
        kind: "non-finite",
        expected: "1",
        actual: "NaN",
      },
    ]);

    expect(() =>
      compareSnapshot(1, 1, { absoluteTolerance: -1 }),
    ).toThrow(SnapshotError);
    expect(() =>
      compareSnapshot(1, 1, { relativeTolerance: Number.NaN }),
    ).toThrow(SnapshotError);
  });
});
