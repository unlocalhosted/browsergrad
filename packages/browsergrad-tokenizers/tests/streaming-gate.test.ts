import { describe, expect, it } from "vitest";
import { createStreamingGate } from "../src/index";

describe("createStreamingGate", () => {
  it("fails when too many chunks are consumed before first yield", () => {
    const gate = createStreamingGate({
      maxChunksBeforeFirstYield: 2,
      chunkCount: 10,
    });

    gate.noteChunkConsumed();
    gate.noteChunkConsumed();

    expect(() => gate.noteChunkConsumed()).toThrow(
      "consumed 3 chunks before first yield",
    );
  });

  it("allows additional chunks after first yield", () => {
    const gate = createStreamingGate({
      maxChunksBeforeFirstYield: 2,
      chunkCount: 4,
    });

    gate.noteChunkConsumed();
    gate.noteFirstYield();
    gate.noteChunkConsumed();
    gate.noteChunkConsumed();
    gate.noteChunkConsumed();

    expect(gate.chunksConsumed).toBe(4);
    expect(gate.firstYieldSeen).toBe(true);
  });

  it("fails when consumption exceeds declared chunk count", () => {
    const gate = createStreamingGate({
      maxChunksBeforeFirstYield: 3,
      chunkCount: 1,
    });

    gate.noteFirstYield();
    gate.noteChunkConsumed();

    expect(() => gate.noteChunkConsumed()).toThrow(
      "consumed more than declared 1 chunks",
    );
  });
});
