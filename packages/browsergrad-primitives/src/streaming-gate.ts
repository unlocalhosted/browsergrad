export interface StreamingGateOptions {
  readonly maxChunksBeforeFirstYield: number;
  readonly chunkCount: number;
}

export interface StreamingGate {
  readonly maxChunksBeforeFirstYield: number;
  readonly chunkCount: number;
  readonly chunksConsumed: number;
  readonly firstYieldSeen: boolean;
  wrapInput<T>(iterable: Iterable<T>): Iterable<T>;
  wrapOutput<T>(iterable: Iterable<T>): Iterable<T>;
  noteChunkConsumed(): void;
  noteFirstYield(): void;
  assertAllowed(): void;
}

export function createStreamingGate(
  options: StreamingGateOptions,
): StreamingGate {
  if (
    !Number.isInteger(options.maxChunksBeforeFirstYield) ||
    options.maxChunksBeforeFirstYield < 0
  ) {
    throw new Error("maxChunksBeforeFirstYield must be a non-negative integer");
  }
  if (!Number.isInteger(options.chunkCount) || options.chunkCount < 0) {
    throw new Error("chunkCount must be a non-negative integer");
  }
  return new BasicStreamingGate(options);
}

class BasicStreamingGate implements StreamingGate {
  readonly maxChunksBeforeFirstYield: number;
  readonly chunkCount: number;
  private consumed = 0;
  private yielded = false;

  constructor(options: StreamingGateOptions) {
    this.maxChunksBeforeFirstYield = options.maxChunksBeforeFirstYield;
    this.chunkCount = options.chunkCount;
  }

  get chunksConsumed(): number {
    return this.consumed;
  }

  get firstYieldSeen(): boolean {
    return this.yielded;
  }

  wrapInput<T>(iterable: Iterable<T>): Iterable<T> {
    const noteChunkConsumed = this.noteChunkConsumed.bind(this);
    return {
      *[Symbol.iterator](): Iterator<T> {
        for (const item of iterable) {
          noteChunkConsumed();
          yield item;
        }
      },
    };
  }

  wrapOutput<T>(iterable: Iterable<T>): Iterable<T> {
    const noteFirstYield = this.noteFirstYield.bind(this);
    return {
      *[Symbol.iterator](): Iterator<T> {
        for (const item of iterable) {
          noteFirstYield();
          yield item;
        }
      },
    };
  }

  noteChunkConsumed(): void {
    if (this.consumed >= this.chunkCount) {
      throw new Error(
        `streaming gate failed: consumed more than declared ${this.chunkCount} chunks`,
      );
    }
    this.consumed += 1;
    this.assertAllowed();
  }

  noteFirstYield(): void {
    this.yielded = true;
  }

  assertAllowed(): void {
    if (
      !this.yielded &&
      this.consumed > this.maxChunksBeforeFirstYield
    ) {
      throw new Error(
        `streaming gate failed: consumed ${this.consumed} chunks before first yield ` +
          `(limit ${this.maxChunksBeforeFirstYield})`,
      );
    }
  }
}
