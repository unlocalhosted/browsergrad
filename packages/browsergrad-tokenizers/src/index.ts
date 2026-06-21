export const CS336_PRETOKENIZER_PATTERN =
  String.raw`'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`;

export const CS336_DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"] as const;

export interface TrainByteBpeOptions {
  readonly vocabSize: number;
  readonly specialTokens?: readonly string[];
  readonly pretokenizerPattern?: string;
}

export type ByteToken = Uint8Array;
export type BytePair = readonly [ByteToken, ByteToken];

export interface ByteBpeModel {
  readonly vocab: ReadonlyMap<number, ByteToken>;
  readonly merges: readonly BytePair[];
  readonly specialTokens: readonly string[];
  readonly pretokenizerPattern: string;
}

export interface SerializedByteBpeModel {
  readonly vocab: Record<string, readonly number[]>;
  readonly merges: readonly (readonly [readonly number[], readonly number[]])[];
  readonly specialTokens: readonly string[];
  readonly pretokenizerPattern: string;
}

export interface TokenizerOracle {
  trainByteBpe(input: string, options: TrainByteBpeOptions): ByteBpeModel;
  encodeByteBpe(text: string, model: ByteBpeModel): number[];
  decodeByteBpe(ids: readonly number[], model: ByteBpeModel): string;
}

export interface StreamingGateOptions {
  readonly maxChunksBeforeFirstYield: number;
  readonly chunkCount: number;
}

export interface StreamingGate {
  readonly maxChunksBeforeFirstYield: number;
  readonly chunkCount: number;
  readonly chunksConsumed: number;
  noteChunkConsumed(): void;
  noteFirstYield(): void;
  assertAllowed(): void;
}

export const CS336_BPE_EXAMPLE = {
  corpus: "low low low low low lower lower widest widest widest newest newest newest newest newest newest",
  vocabSize: 259,
  specialTokens: CS336_DEFAULT_SPECIAL_TOKENS,
} as const;

export function trainByteBpe(
  _input: string,
  _options: TrainByteBpeOptions,
): ByteBpeModel {
  throw new Error("trainByteBpe is not implemented yet");
}

export function encodeByteBpe(
  _text: string,
  _model: ByteBpeModel,
): number[] {
  throw new Error("encodeByteBpe is not implemented yet");
}

export function decodeByteBpe(
  _ids: readonly number[],
  _model: ByteBpeModel,
): string {
  throw new Error("decodeByteBpe is not implemented yet");
}

export function serializeByteBpeModel(
  _model: ByteBpeModel,
): SerializedByteBpeModel {
  throw new Error("serializeByteBpeModel is not implemented yet");
}

export function deserializeByteBpeModel(
  _model: SerializedByteBpeModel,
): ByteBpeModel {
  throw new Error("deserializeByteBpeModel is not implemented yet");
}

export function createCs336TokenizerOracle(): TokenizerOracle {
  return {
    trainByteBpe,
    encodeByteBpe,
    decodeByteBpe,
  };
}

export function createStreamingGate(
  _options: StreamingGateOptions,
): StreamingGate {
  throw new Error("createStreamingGate is not implemented yet");
}
