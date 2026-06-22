import { describe, expect, it } from "vitest";
import {
  referenceExclusiveScan,
  referenceFindRepeats,
  referenceOrderedCircleRender,
  referenceSaxpy,
  simulateCuda1DGrid,
} from "../src/index";

describe("CUDA-shaped concept oracles", () => {
  it("runs GPU Puzzles-style map kernels with thread/block traces", () => {
    const result = simulateCuda1DGrid({
      inputLength: 4,
      outputLength: 4,
      threadsPerBlock: 2,
      blocks: 2,
      initialInput: [0, 1, 2, 3],
      kernel(ctx) {
        const i = ctx.globalThreadId;
        ctx.write(i, ctx.read(i) + 10);
      },
    });

    expect(result.output).toEqual([10, 11, 12, 13]);
    expect(result.stats).toEqual({
      launchedThreads: 4,
      activeThreads: 4,
      globalReads: 4,
      globalWrites: 4,
      outOfBoundsReads: 0,
      outOfBoundsWrites: 0,
    });
    expect(result.trace.map((event) => [event.globalThreadId, event.blockIdxX, event.threadIdxX])).toEqual([
      [0, 0, 0],
      [1, 0, 1],
      [2, 1, 0],
      [3, 1, 1],
    ]);
  });

  it("reports guard failures instead of hiding out-of-bounds memory access", () => {
    const result = simulateCuda1DGrid({
      inputLength: 4,
      outputLength: 4,
      threadsPerBlock: 8,
      blocks: 1,
      initialInput: [0, 1, 2, 3],
      kernel(ctx) {
        const i = ctx.globalThreadId;
        ctx.write(i, ctx.read(i) + 10);
      },
    });

    expect(result.output).toEqual([10, 11, 12, 13]);
    expect(result.stats).toMatchObject({
      launchedThreads: 8,
      activeThreads: 8,
      globalReads: 4,
      globalWrites: 4,
      outOfBoundsReads: 4,
      outOfBoundsWrites: 4,
    });
    expect(result.violations.map((violation) => violation.kind)).toEqual([
      "read-oob",
      "write-oob",
      "read-oob",
      "write-oob",
      "read-oob",
      "write-oob",
      "read-oob",
      "write-oob",
    ]);
  });

  it("provides CS149 A3-style SAXPY and exclusive scan references", () => {
    expect(referenceSaxpy({ a: 2, x: [1, 2, 3], y: [10, 20, 30] })).toEqual([
      12,
      24,
      36,
    ]);
    expect(referenceExclusiveScan([3, 1, 4, 1, 5])).toEqual([0, 3, 4, 8, 9]);
    expect(referenceFindRepeats([3, 3, 1, 4, 4, 4, 2])).toEqual([0, 3, 4]);
  });

  it("renders overlapping translucent circles in input order", () => {
    const redGreenBlue = referenceOrderedCircleRender({
      width: 1,
      height: 1,
      background: [0, 0, 0],
      circles: [
        { center: [0.5, 0.5], radius: 1, color: [1, 0, 0], alpha: 0.5 },
        { center: [0.5, 0.5], radius: 1, color: [0, 1, 0], alpha: 0.5 },
        { center: [0.5, 0.5], radius: 1, color: [0, 0, 1], alpha: 0.5 },
      ],
    });
    const blueGreenRed = referenceOrderedCircleRender({
      width: 1,
      height: 1,
      background: [0, 0, 0],
      circles: [
        { center: [0.5, 0.5], radius: 1, color: [0, 0, 1], alpha: 0.5 },
        { center: [0.5, 0.5], radius: 1, color: [0, 1, 0], alpha: 0.5 },
        { center: [0.5, 0.5], radius: 1, color: [1, 0, 0], alpha: 0.5 },
      ],
    });

    expect(redGreenBlue.pixels).toEqual([0.125, 0.25, 0.5]);
    expect(blueGreenRed.pixels).toEqual([0.5, 0.25, 0.125]);
  });
});
