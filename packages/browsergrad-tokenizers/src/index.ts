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

export interface Cs336TokenizerOracleModule {
  train_cs336_bpe(
    input: string,
    vocabSize: number,
    specialTokens?: readonly string[],
  ): SerializedByteBpeModel;
  encode_cs336(text: string, model: SerializedByteBpeModel): number[];
  decode_cs336(ids: readonly number[], model: SerializedByteBpeModel): string;
}

export interface StreamingGateOptions {
  readonly maxChunksBeforeFirstYield: number;
  readonly chunkCount: number;
}

export interface StreamingGate {
  readonly maxChunksBeforeFirstYield: number;
  readonly chunkCount: number;
  readonly chunksConsumed: number;
  readonly firstYieldSeen: boolean;
  noteChunkConsumed(): void;
  noteFirstYield(): void;
  assertAllowed(): void;
}

export const CS336_BPE_EXAMPLE = {
  corpus: "low low low low low lower lower widest widest widest newest newest newest newest newest newest",
  vocabSize: 259,
  specialTokens: CS336_DEFAULT_SPECIAL_TOKENS,
} as const;

const UTF8_ENCODER = new TextEncoder();
const UTF8_DECODER = new TextDecoder();

interface WordEntry {
  tokens: number[];
  frequency: number;
}

interface PairCount {
  readonly left: number;
  readonly right: number;
  count: number;
}

export function trainByteBpe(
  input: string,
  options: TrainByteBpeOptions,
): ByteBpeModel {
  if (!Number.isInteger(options.vocabSize) || options.vocabSize < 0) {
    throw new Error("vocabSize must be a non-negative integer");
  }

  const specialTokens = [...(options.specialTokens ?? [])];
  const pretokenizerPattern =
    options.pretokenizerPattern ?? CS336_PRETOKENIZER_PATTERN;
  const vocab = createInitialVocabulary(specialTokens);
  const targetSize = options.vocabSize;
  if (targetSize <= vocab.size) {
    return {
      vocab: limitVocabulary(vocab, targetSize),
      merges: [],
      specialTokens,
      pretokenizerPattern,
    };
  }

  const words = buildWordEntries(input, specialTokens, pretokenizerPattern);
  const merges: BytePair[] = [];

  while (vocab.size < targetSize) {
    const pairCounts = countPairs(words);
    const best = chooseBestPair(pairCounts, vocab);
    if (!best) break;

    const leftBytes = vocab.get(best.left);
    const rightBytes = vocab.get(best.right);
    if (!leftBytes || !rightBytes) {
      throw new Error("internal BPE error: selected pair is missing from vocabulary");
    }

    const merged = concatBytes(leftBytes, rightBytes);
    const mergedId = vocab.size;
    vocab.set(mergedId, merged);
    merges.push([copyBytes(leftBytes), copyBytes(rightBytes)]);
    mergePairInWords(words, best.left, best.right, mergedId);
  }

  return {
    vocab,
    merges,
    specialTokens,
    pretokenizerPattern,
  };
}

export function encodeByteBpe(
  text: string,
  model: ByteBpeModel,
): number[] {
  const out: number[] = [];
  const byteToId = buildByteToId(model.vocab);
  const specialTokenIds = buildSpecialTokenIds(model);
  const pattern = new RegExp(model.pretokenizerPattern, "gu");

  for (const segment of splitPreservingSpecialTokens(text, model.specialTokens)) {
    if (segment.kind === "special") {
      const id = specialTokenIds.get(segment.text);
      if (id === undefined) {
        throw new Error(`special token is missing from vocabulary: ${segment.text}`);
      }
      out.push(id);
      continue;
    }

    pattern.lastIndex = 0;
    for (const match of segment.text.matchAll(pattern)) {
      const pretoken = match[0];
      if (pretoken.length === 0) continue;
      const bytes = UTF8_ENCODER.encode(pretoken);
      let ids = [...bytes];
      ids = applyMerges(ids, model, byteToId);
      out.push(...ids);
    }
  }

  return out;
}

export function decodeByteBpe(
  ids: readonly number[],
  model: ByteBpeModel,
): string {
  const chunks: Uint8Array[] = [];
  let totalLength = 0;
  for (const id of ids) {
    const bytes = model.vocab.get(id);
    if (!bytes) {
      throw new Error(`unknown token id: ${id}`);
    }
    chunks.push(bytes);
    totalLength += bytes.length;
  }

  const joined = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.length;
  }
  return UTF8_DECODER.decode(joined);
}

export function serializeByteBpeModel(
  model: ByteBpeModel,
): SerializedByteBpeModel {
  const vocab: Record<string, readonly number[]> = {};
  for (const [id, bytes] of [...model.vocab.entries()].sort((a, b) => a[0] - b[0])) {
    vocab[String(id)] = [...bytes];
  }
  return {
    vocab,
    merges: model.merges.map(([left, right]) => [[...left], [...right]]),
    specialTokens: [...model.specialTokens],
    pretokenizerPattern: model.pretokenizerPattern,
  };
}

export function deserializeByteBpeModel(
  model: SerializedByteBpeModel,
): ByteBpeModel {
  const vocab = new Map<number, Uint8Array>();
  for (const [idText, bytes] of Object.entries(model.vocab)) {
    const id = Number(idText);
    if (!Number.isInteger(id) || id < 0) {
      throw new Error(`invalid serialized token id: ${idText}`);
    }
    vocab.set(id, bytesToUint8Array(bytes, `vocab[${idText}]`));
  }
  return {
    vocab,
    merges: model.merges.map(([left, right]) => [
      bytesToUint8Array(left, "merge left token"),
      bytesToUint8Array(right, "merge right token"),
    ]),
    specialTokens: [...model.specialTokens],
    pretokenizerPattern: model.pretokenizerPattern,
  };
}

export function createCs336TokenizerOracle(): TokenizerOracle {
  return {
    trainByteBpe: (input, options) =>
      trainByteBpe(input, {
        ...options,
        specialTokens: options.specialTokens ?? CS336_DEFAULT_SPECIAL_TOKENS,
        pretokenizerPattern:
          options.pretokenizerPattern ?? CS336_PRETOKENIZER_PATTERN,
      }),
    encodeByteBpe,
    decodeByteBpe,
  };
}

export function createCs336TokenizerOracleModule(): Cs336TokenizerOracleModule {
  return {
    train_cs336_bpe: (input, vocabSize, specialTokens) =>
      serializeByteBpeModel(
        trainByteBpe(input, {
          vocabSize,
          specialTokens: specialTokens ?? CS336_DEFAULT_SPECIAL_TOKENS,
          pretokenizerPattern: CS336_PRETOKENIZER_PATTERN,
        }),
      ),
    encode_cs336: (text, model) =>
      encodeByteBpe(text, deserializeByteBpeModel(model)),
    decode_cs336: (ids, model) =>
      decodeByteBpe(ids, deserializeByteBpeModel(model)),
  };
}

export const cs336TokenizerOracleModule = createCs336TokenizerOracleModule();

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

function createInitialVocabulary(
  specialTokens: readonly string[],
): Map<number, Uint8Array> {
  const vocab = new Map<number, Uint8Array>();
  for (let i = 0; i < 256; i++) {
    vocab.set(i, Uint8Array.of(i));
  }
  for (const token of specialTokens) {
    vocab.set(vocab.size, UTF8_ENCODER.encode(token));
  }
  return vocab;
}

function limitVocabulary(
  vocab: ReadonlyMap<number, Uint8Array>,
  size: number,
): Map<number, Uint8Array> {
  const out = new Map<number, Uint8Array>();
  for (const [id, bytes] of vocab) {
    if (id >= size) break;
    out.set(id, copyBytes(bytes));
  }
  return out;
}

function buildWordEntries(
  input: string,
  specialTokens: readonly string[],
  pretokenizerPattern: string,
): WordEntry[] {
  const pattern = new RegExp(pretokenizerPattern, "gu");
  const byBytes = new Map<string, WordEntry>();

  for (const segment of splitOnSpecialTokens(input, specialTokens)) {
    pattern.lastIndex = 0;
    for (const match of segment.matchAll(pattern)) {
      const text = match[0];
      if (text.length === 0) continue;
      const bytes = UTF8_ENCODER.encode(text);
      const key = bytesKey(bytes);
      const existing = byBytes.get(key);
      if (existing) {
        existing.frequency += 1;
      } else {
        byBytes.set(key, {
          tokens: [...bytes],
          frequency: 1,
        });
      }
    }
  }

  return [...byBytes.values()];
}

function splitOnSpecialTokens(
  input: string,
  specialTokens: readonly string[],
): string[] {
  const activeTokens = [...specialTokens]
    .filter((token) => token.length > 0)
    .sort((a, b) => b.length - a.length || a.localeCompare(b));
  if (activeTokens.length === 0) return [input];

  const segments: string[] = [];
  let cursor = 0;
  while (cursor < input.length) {
    let nextIndex = -1;
    let nextToken = "";
    for (const token of activeTokens) {
      const idx = input.indexOf(token, cursor);
      if (idx === -1) continue;
      if (
        nextIndex === -1 ||
        idx < nextIndex ||
        (idx === nextIndex && token.length > nextToken.length)
      ) {
        nextIndex = idx;
        nextToken = token;
      }
    }

    if (nextIndex === -1) {
      segments.push(input.slice(cursor));
      break;
    }
    if (nextIndex > cursor) {
      segments.push(input.slice(cursor, nextIndex));
    }
    cursor = nextIndex + nextToken.length;
  }

  return segments;
}

type PreservedSegment =
  | { readonly kind: "text"; readonly text: string }
  | { readonly kind: "special"; readonly text: string };

function splitPreservingSpecialTokens(
  input: string,
  specialTokens: readonly string[],
): PreservedSegment[] {
  const activeTokens = [...specialTokens]
    .filter((token) => token.length > 0)
    .sort((a, b) => b.length - a.length || a.localeCompare(b));
  if (activeTokens.length === 0) return [{ kind: "text", text: input }];

  const segments: PreservedSegment[] = [];
  let cursor = 0;
  while (cursor < input.length) {
    let nextIndex = -1;
    let nextToken = "";
    for (const token of activeTokens) {
      const idx = input.indexOf(token, cursor);
      if (idx === -1) continue;
      if (
        nextIndex === -1 ||
        idx < nextIndex ||
        (idx === nextIndex && token.length > nextToken.length)
      ) {
        nextIndex = idx;
        nextToken = token;
      }
    }

    if (nextIndex === -1) {
      segments.push({ kind: "text", text: input.slice(cursor) });
      break;
    }
    if (nextIndex > cursor) {
      segments.push({ kind: "text", text: input.slice(cursor, nextIndex) });
    }
    segments.push({ kind: "special", text: nextToken });
    cursor = nextIndex + nextToken.length;
  }

  return segments;
}

function countPairs(words: readonly WordEntry[]): Map<string, PairCount> {
  const pairCounts = new Map<string, PairCount>();
  for (const word of words) {
    const tokens = word.tokens;
    for (let i = 0; i < tokens.length - 1; i++) {
      const left = tokens[i];
      const right = tokens[i + 1];
      if (left === undefined || right === undefined) continue;
      const key = pairKey(left, right);
      const existing = pairCounts.get(key);
      if (existing) {
        existing.count += word.frequency;
      } else {
        pairCounts.set(key, { left, right, count: word.frequency });
      }
    }
  }
  return pairCounts;
}

function chooseBestPair(
  pairCounts: ReadonlyMap<string, PairCount>,
  vocab: ReadonlyMap<number, Uint8Array>,
): PairCount | null {
  let best: PairCount | null = null;
  for (const pair of pairCounts.values()) {
    if (!best) {
      best = pair;
      continue;
    }
    if (pair.count > best.count) {
      best = pair;
      continue;
    }
    if (pair.count === best.count && comparePair(pair, best, vocab) > 0) {
      best = pair;
    }
  }
  return best;
}

function comparePair(
  a: PairCount,
  b: PairCount,
  vocab: ReadonlyMap<number, Uint8Array>,
): number {
  const aLeft = vocab.get(a.left);
  const aRight = vocab.get(a.right);
  const bLeft = vocab.get(b.left);
  const bRight = vocab.get(b.right);
  if (!aLeft || !aRight || !bLeft || !bRight) {
    throw new Error("internal BPE error: pair refers to missing vocabulary entry");
  }
  const leftCmp = compareBytes(aLeft, bLeft);
  if (leftCmp !== 0) return leftCmp;
  return compareBytes(aRight, bRight);
}

function mergePairInWords(
  words: readonly WordEntry[],
  left: number,
  right: number,
  merged: number,
): void {
  for (const word of words) {
    const next: number[] = [];
    for (let i = 0; i < word.tokens.length; i++) {
      const current = word.tokens[i];
      const following = word.tokens[i + 1];
      if (current === left && following === right) {
        next.push(merged);
        i += 1;
      } else if (current !== undefined) {
        next.push(current);
      }
    }
    word.tokens = next;
  }
}

function compareBytes(a: Uint8Array, b: Uint8Array): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const ai = a[i];
    const bi = b[i];
    if (ai === undefined || bi === undefined) break;
    if (ai !== bi) return ai - bi;
  }
  return a.length - b.length;
}

function applyMerges(
  ids: readonly number[],
  model: ByteBpeModel,
  byteToId: ReadonlyMap<string, number>,
): number[] {
  let current = [...ids];
  const firstMergeId = 256 + model.specialTokens.length;
  for (let mergeIndex = 0; mergeIndex < model.merges.length; mergeIndex++) {
    const pair = model.merges[mergeIndex];
    if (!pair) continue;
    const [leftBytes, rightBytes] = pair;
    const leftId = byteToId.get(bytesKey(leftBytes));
    const rightId = byteToId.get(bytesKey(rightBytes));
    const mergedId = firstMergeId + mergeIndex;
    if (leftId === undefined || rightId === undefined) {
      throw new Error("model merge references a token missing from vocabulary");
    }
    if (!model.vocab.has(mergedId)) {
      throw new Error(`model merge id ${mergedId} is missing from vocabulary`);
    }

    const next: number[] = [];
    for (let i = 0; i < current.length; i++) {
      const token = current[i];
      const following = current[i + 1];
      if (token === leftId && following === rightId) {
        next.push(mergedId);
        i += 1;
      } else if (token !== undefined) {
        next.push(token);
      }
    }
    current = next;
  }
  return current;
}

function buildByteToId(vocab: ReadonlyMap<number, Uint8Array>): Map<string, number> {
  const out = new Map<string, number>();
  for (const [id, bytes] of [...vocab.entries()].sort((a, b) => a[0] - b[0])) {
    const key = bytesKey(bytes);
    if (!out.has(key)) out.set(key, id);
  }
  return out;
}

function buildSpecialTokenIds(model: ByteBpeModel): Map<string, number> {
  const out = new Map<string, number>();
  for (let i = 0; i < model.specialTokens.length; i++) {
    const token = model.specialTokens[i];
    if (token !== undefined) out.set(token, 256 + i);
  }
  return out;
}

function concatBytes(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function copyBytes(bytes: Uint8Array): Uint8Array {
  return new Uint8Array(bytes);
}

function pairKey(left: number, right: number): string {
  return `${left},${right}`;
}

function bytesKey(bytes: Uint8Array): string {
  return [...bytes].join(",");
}

function bytesToUint8Array(
  bytes: readonly number[],
  label: string,
): Uint8Array {
  for (const byte of bytes) {
    if (!Number.isInteger(byte) || byte < 0 || byte > 255) {
      throw new Error(`${label} contains invalid byte: ${byte}`);
    }
  }
  return Uint8Array.from(bytes);
}
