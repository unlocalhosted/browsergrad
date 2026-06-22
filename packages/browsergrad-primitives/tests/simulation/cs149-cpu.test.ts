import { describe, expect, it } from "vitest";
import {
  partitionStaticWork,
  SimulatorError,
  simulateVectorizedArraySum,
  simulateVectorizedClampedExp,
} from "../../src/simulation";

function expectCloseArray(actual: readonly number[], expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    expect(actual[index]).toBeCloseTo(expected[index]!, 6);
  }
}

describe("CPU/SIMD teaching simulators", () => {
  it("simulates clampedExp with tail masks and divergent lane utilization", () => {
    const result = simulateVectorizedClampedExp({
      values: [2, 3, -2, 4, 2],
      exponents: [0, 2, 3, 4, 5],
      vectorWidth: 4,
    });

    expectCloseArray(result.output, [1, 9, -8, 9.999999, 9.999999]);
    expect(result.stats).toMatchObject({
      vectorWidth: 4,
      chunks: 2,
      inputLanes: 5,
      laneSlots: 8,
      tailInactiveLanes: 3,
      vectorInstructions: result.trace.length,
    });
    expect(result.stats.utilization).toBeGreaterThan(0);
    expect(result.stats.utilization).toBeLessThan(1);
    expect(result.trace.some((event) => event.op === "mul" && event.activeLanes === 1))
      .toBe(true);

    result.trace[0]!.activeLanes = 0;
    expect(
      simulateVectorizedClampedExp({
        values: [2],
        exponents: [1],
        vectorWidth: 4,
      }).trace[0]?.activeLanes,
    ).toBe(1);
  });

  it("rejects invalid SIMD fixtures before rubrics trust their stats", () => {
    expect(() =>
      simulateVectorizedClampedExp({
        values: [2],
        exponents: [1, 2],
        vectorWidth: 4,
      }),
    ).toThrow(SimulatorError);
    expect(() =>
      simulateVectorizedClampedExp({
        values: [2],
        exponents: [Number.NaN],
        vectorWidth: 4,
      }),
    ).toThrow(SimulatorError);
    expect(() =>
      simulateVectorizedClampedExp({
        values: [2],
        exponents: [1],
        vectorWidth: 0,
      }),
    ).toThrow(SimulatorError);
  });

  it("simulates vector array sum with deterministic horizontal reduction stats", () => {
    const result = simulateVectorizedArraySum({
      values: [1, 2, 3, 4, 5, 6, 7, 8],
      vectorWidth: 4,
    });

    expect(result.sum).toBe(36);
    expect(result.partialLaneSums).toEqual([6, 8, 10, 12]);
    expect(result.horizontalReductionRounds).toBe(2);
    expect(result.stats).toMatchObject({
      vectorWidth: 4,
      chunks: 2,
      inputLanes: 8,
      laneSlots: 8,
      tailInactiveLanes: 0,
      vectorInstructions: result.trace.length,
    });
    expect(result.trace.map((event) => event.op)).toContain("horizontal-add");
  });

  it("partitions static work into balanced contiguous or cyclic chunks", () => {
    expect(partitionStaticWork({ items: 10, workers: 3 })).toEqual([
      { worker: 0, ranges: [{ start: 0, end: 4 }] },
      { worker: 1, ranges: [{ start: 4, end: 7 }] },
      { worker: 2, ranges: [{ start: 7, end: 10 }] },
    ]);

    expect(partitionStaticWork({ items: 10, workers: 3, chunkSize: 2 })).toEqual([
      {
        worker: 0,
        ranges: [
          { start: 0, end: 2 },
          { start: 6, end: 8 },
        ],
      },
      {
        worker: 1,
        ranges: [
          { start: 2, end: 4 },
          { start: 8, end: 10 },
        ],
      },
      { worker: 2, ranges: [{ start: 4, end: 6 }] },
    ]);
  });
});
