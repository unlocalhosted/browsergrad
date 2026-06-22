import { describe, expect, it } from "vitest";
import { createStreamingGate } from "../../src/text";

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

  it("wraps input and output iterables for incremental JS rubrics", () => {
    const gate = createStreamingGate({
      maxChunksBeforeFirstYield: 1,
      chunkCount: 3,
    });
    function* streamingUpper(chunks: Iterable<string>): Iterable<string> {
      for (const chunk of chunks) {
        yield chunk.toUpperCase();
      }
    }

    const output = gate.wrapOutput(
      streamingUpper(gate.wrapInput(["a", "b", "c"])),
    );
    const iterator = output[Symbol.iterator]();

    expect(iterator.next()).toEqual({ value: "A", done: false });
    expect(gate.chunksConsumed).toBe(1);
    expect(gate.firstYieldSeen).toBe(true);
    const rest: string[] = [];
    for (let next = iterator.next(); !next.done; next = iterator.next()) {
      rest.push(next.value);
    }
    expect(rest).toEqual(["B", "C"]);
  });

  it("fails when wrapped JS input is consumed eagerly before first output", () => {
    const gate = createStreamingGate({
      maxChunksBeforeFirstYield: 1,
      chunkCount: 3,
    });
    function* eagerJoin(chunks: Iterable<string>): Iterable<string> {
      yield [...chunks].join("");
    }

    const output = gate.wrapOutput(eagerJoin(gate.wrapInput(["a", "b", "c"])));
    const iterator = output[Symbol.iterator]();

    expect(() => iterator.next()).toThrow("consumed 2 chunks before first yield");
  });
});
