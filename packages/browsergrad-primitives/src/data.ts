export type PiiKind = "email" | "phone" | "ip";

export interface PiiSpan {
  readonly kind: PiiKind;
  readonly original: string;
  readonly replacement: string;
  readonly start: number;
  readonly end: number;
}

export interface PiiMaskResult {
  readonly text: string;
  readonly spans: readonly PiiSpan[];
}

export interface DeduplicationOptions {
  readonly trim?: boolean;
  readonly collapseWhitespace?: boolean;
  readonly caseSensitive?: boolean;
}

export interface DuplicateLine {
  readonly line: string;
  readonly index: number;
  readonly firstIndex: number;
  readonly key: string;
}

export interface DeduplicationResult {
  readonly keptLines: readonly string[];
  readonly duplicates: readonly DuplicateLine[];
}

export interface DocumentInput {
  readonly id: string;
  readonly text: string;
}

export interface MinHashDeduplicationOptions {
  readonly ngrams?: number;
  readonly numHashes?: number;
  readonly numBands?: number;
  readonly jaccardThreshold?: number;
  readonly caseSensitive?: boolean;
}

export interface NearDuplicateDocument {
  readonly id: string;
  readonly duplicateOf: string;
  readonly jaccard: number;
  readonly candidateReason: "lsh-band" | "exact-jaccard";
}

export interface MinHashDeduplicationResult {
  readonly keptDocuments: readonly DocumentInput[];
  readonly duplicates: readonly NearDuplicateDocument[];
}

export type GopherQualityRule =
  | "non_symbol_word_count"
  | "average_word_length"
  | "ellipsis_line_ratio"
  | "alphabetic_word_ratio";

export interface GopherQualityOptions {
  readonly minNonSymbolWords?: number;
  readonly maxNonSymbolWords?: number;
  readonly minAverageWordLength?: number;
  readonly maxAverageWordLength?: number;
  readonly maxEllipsisLineRatio?: number;
  readonly minAlphabeticWordRatio?: number;
}

export interface GopherQualityReport {
  readonly passed: boolean;
  readonly failedRules: readonly GopherQualityRule[];
  readonly metrics: {
    readonly nonSymbolWords: number;
    readonly averageWordLength: number;
    readonly ellipsisLineRatio: number;
    readonly alphabeticWordRatio: number;
  };
}

const EMAIL_RE = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/giu;
const IP_RE =
  /\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b/gu;
const PHONE_RE =
  /(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)/gu;

export function maskPii(input: string): PiiMaskResult {
  const matches = [
    ...findPiiMatches(input, "email", EMAIL_RE, "<EMAIL>"),
    ...findPiiMatches(input, "ip", IP_RE, "<IP>"),
    ...findPiiMatches(input, "phone", PHONE_RE, "<PHONE>"),
  ].sort((a, b) => a.start - b.start || b.end - a.end);
  const selected: PiiSpan[] = [];
  let lastEnd = -1;
  for (const match of matches) {
    if (match.start < lastEnd) continue;
    selected.push(match);
    lastEnd = match.end;
  }

  let cursor = 0;
  let text = "";
  for (const span of selected) {
    text += input.slice(cursor, span.start);
    text += span.replacement;
    cursor = span.end;
  }
  text += input.slice(cursor);

  return { text, spans: selected };
}

export function exactLineDeduplicate(
  lines: readonly string[],
  options: DeduplicationOptions = {},
): DeduplicationResult {
  const seen = new Map<string, number>();
  const keptLines: string[] = [];
  const duplicates: DuplicateLine[] = [];

  for (let index = 0; index < lines.length; index++) {
    const line = lines[index] ?? "";
    const key = normalizeLineKey(line, options);
    const firstIndex = seen.get(key);
    if (firstIndex !== undefined) {
      duplicates.push({ line, index, firstIndex, key });
      continue;
    }
    seen.set(key, index);
    keptLines.push(line);
  }

  return { keptLines, duplicates };
}

export function extractVisibleTextFromHtml(html: string): string {
  const withoutBlocks = html
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<script\b[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[\s\S]*?<\/style>/gi, " ");
  const withBreaks = withoutBlocks
    .replace(/<\s*br\s*\/?>/gi, " ")
    .replace(/<\/(?:p|div|section|article|h[1-6]|li|tr|table|body|html)>/gi, " ");
  const withoutTags = withBreaks.replace(/<[^>]+>/g, " ");
  return decodeBasicHtmlEntities(withoutTags)
    .replace(/\s+/g, " ")
    .trim();
}

export function minhashDeduplicateDocuments(
  documents: readonly DocumentInput[],
  options: MinHashDeduplicationOptions = {},
): MinHashDeduplicationResult {
  const ngrams = validatePositiveInteger(options.ngrams ?? 5, "ngrams");
  const numHashes = validatePositiveInteger(options.numHashes ?? 100, "numHashes");
  const numBands = validatePositiveInteger(options.numBands ?? 10, "numBands");
  const jaccardThreshold = validateThreshold(options.jaccardThreshold ?? 0.8);
  const bandSize = Math.ceil(numHashes / numBands);
  const keptDocuments: DocumentInput[] = [];
  const keptShingles: Array<Set<string>> = [];
  const lshBuckets = new Map<string, number[]>();
  const duplicates: NearDuplicateDocument[] = [];

  for (const document of documents) {
    const shingles = createWordShingles(document.text, ngrams, options);
    const signature = createMinHashSignature(shingles, numHashes);
    const candidateIndexes = collectCandidateIndexes(signature, bandSize, lshBuckets);
    const duplicate = findDuplicate(
      document.id,
      shingles,
      keptDocuments,
      keptShingles,
      candidateIndexes,
      jaccardThreshold,
    );

    if (duplicate) {
      duplicates.push(duplicate);
      continue;
    }

    const keptIndex = keptDocuments.length;
    keptDocuments.push(document);
    keptShingles.push(shingles);
    indexSignatureBands(signature, keptIndex, bandSize, lshBuckets);
  }

  return { keptDocuments, duplicates };
}

export function evaluateGopherQuality(
  text: string,
  options: GopherQualityOptions = {},
): GopherQualityReport {
  const minNonSymbolWords = options.minNonSymbolWords ?? 50;
  const maxNonSymbolWords = options.maxNonSymbolWords ?? 100_000;
  const minAverageWordLength = options.minAverageWordLength ?? 3;
  const maxAverageWordLength = options.maxAverageWordLength ?? 10;
  const maxEllipsisLineRatio = options.maxEllipsisLineRatio ?? 0.3;
  const minAlphabeticWordRatio = options.minAlphabeticWordRatio ?? 0.8;
  const words = text.match(/\S+/gu) ?? [];
  const nonSymbolWords = words
    .map((word) => word.match(/[\p{L}\p{N}]+/gu)?.join("") ?? "")
    .filter((word) => word.length > 0);
  const alphabeticWords = nonSymbolWords.filter((word) => /\p{L}/u.test(word));
  const lines = text.split(/\r?\n/u).filter((line) => line.length > 0);
  const ellipsisLines = lines.filter((line) => {
    const trimmed = line.trimEnd();
    return trimmed.endsWith("...") || trimmed.endsWith("…");
  });
  const averageWordLength =
    nonSymbolWords.length === 0
      ? 0
      : nonSymbolWords.reduce((sum, word) => sum + word.length, 0) /
        nonSymbolWords.length;
  const ellipsisLineRatio = lines.length === 0 ? 0 : ellipsisLines.length / lines.length;
  const alphabeticWordRatio =
    nonSymbolWords.length === 0 ? 0 : alphabeticWords.length / nonSymbolWords.length;
  const failedRules: GopherQualityRule[] = [];

  if (
    nonSymbolWords.length < minNonSymbolWords ||
    nonSymbolWords.length > maxNonSymbolWords
  ) {
    failedRules.push("non_symbol_word_count");
  }
  if (
    averageWordLength < minAverageWordLength ||
    averageWordLength > maxAverageWordLength
  ) {
    failedRules.push("average_word_length");
  }
  if (ellipsisLineRatio > maxEllipsisLineRatio) {
    failedRules.push("ellipsis_line_ratio");
  }
  if (alphabeticWordRatio < minAlphabeticWordRatio) {
    failedRules.push("alphabetic_word_ratio");
  }

  return {
    passed: failedRules.length === 0,
    failedRules,
    metrics: {
      nonSymbolWords: nonSymbolWords.length,
      averageWordLength,
      ellipsisLineRatio,
      alphabeticWordRatio,
    },
  };
}

export function gopherQualityFilter(
  text: string,
  options: GopherQualityOptions = {},
): boolean {
  return evaluateGopherQuality(text, options).passed;
}

function findPiiMatches(
  input: string,
  kind: PiiKind,
  pattern: RegExp,
  replacement: string,
): PiiSpan[] {
  pattern.lastIndex = 0;
  const spans: PiiSpan[] = [];
  for (const match of input.matchAll(pattern)) {
    const original = match[0];
    const start = match.index;
    if (start === undefined) continue;
    spans.push({
      kind,
      original,
      replacement,
      start,
      end: start + original.length,
    });
  }
  return spans;
}

function normalizeLineKey(line: string, options: DeduplicationOptions): string {
  let key = options.trim ? line.trim() : line;
  if (options.collapseWhitespace) {
    key = key.replace(/\s+/g, " ");
  }
  if (options.caseSensitive === false) {
    key = key.toLowerCase();
  }
  return key;
}

function decodeBasicHtmlEntities(text: string): string {
  return text.replace(
    /&(#x[0-9a-f]+|#\d+|amp|lt|gt|quot|apos|nbsp);/gi,
    (entity, body: string) => {
      const normalized = body.toLowerCase();
      if (normalized === "amp") return "&";
      if (normalized === "lt") return "<";
      if (normalized === "gt") return ">";
      if (normalized === "quot") return '"';
      if (normalized === "apos") return "'";
      if (normalized === "nbsp") return " ";
      if (normalized.startsWith("#x")) {
        return codePointToString(Number.parseInt(normalized.slice(2), 16), entity);
      }
      if (normalized.startsWith("#")) {
        return codePointToString(Number.parseInt(normalized.slice(1), 10), entity);
      }
      return entity;
    },
  );
}

function codePointToString(codePoint: number, fallback: string): string {
  if (!Number.isInteger(codePoint) || codePoint < 0) return fallback;
  try {
    return String.fromCodePoint(codePoint);
  } catch {
    return fallback;
  }
}

function validatePositiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

function validateThreshold(value: number): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error("jaccardThreshold must be between 0 and 1");
  }
  return value;
}

function createWordShingles(
  text: string,
  ngrams: number,
  options: MinHashDeduplicationOptions,
): Set<string> {
  const normalized = options.caseSensitive === true ? text : text.toLowerCase();
  const tokens = normalized.normalize("NFKC").match(/[\p{L}\p{N}]+/gu) ?? [];
  if (tokens.length === 0) return new Set([""]);
  if (tokens.length <= ngrams) return new Set([tokens.join(" ")]);

  const shingles = new Set<string>();
  for (let index = 0; index <= tokens.length - ngrams; index++) {
    shingles.add(tokens.slice(index, index + ngrams).join(" "));
  }
  return shingles;
}

function createMinHashSignature(shingles: Set<string>, numHashes: number): number[] {
  const signature: number[] = [];
  for (let seed = 0; seed < numHashes; seed++) {
    let minHash = 0xffffffff;
    for (const shingle of shingles) {
      minHash = Math.min(minHash, hashStringToUint32(shingle, seed));
    }
    signature.push(minHash);
  }
  return signature;
}

function hashStringToUint32(value: string, seed: number): number {
  let hash = (0x811c9dc5 ^ Math.imul(seed + 1, 0x9e3779b1)) >>> 0;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x85ebca6b) >>> 0;
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35) >>> 0;
  return (hash ^ (hash >>> 16)) >>> 0;
}

function collectCandidateIndexes(
  signature: readonly number[],
  bandSize: number,
  lshBuckets: ReadonlyMap<string, readonly number[]>,
): Set<number> {
  const candidates = new Set<number>();
  for (const key of signatureBandKeys(signature, bandSize)) {
    for (const index of lshBuckets.get(key) ?? []) {
      candidates.add(index);
    }
  }
  return candidates;
}

function indexSignatureBands(
  signature: readonly number[],
  keptIndex: number,
  bandSize: number,
  lshBuckets: Map<string, number[]>,
): void {
  for (const key of signatureBandKeys(signature, bandSize)) {
    const bucket = lshBuckets.get(key);
    if (bucket) {
      bucket.push(keptIndex);
    } else {
      lshBuckets.set(key, [keptIndex]);
    }
  }
}

function signatureBandKeys(signature: readonly number[], bandSize: number): string[] {
  const keys: string[] = [];
  for (let start = 0; start < signature.length; start += bandSize) {
    keys.push(`${start}:${signature.slice(start, start + bandSize).join(",")}`);
  }
  return keys;
}

function findDuplicate(
  documentId: string,
  shingles: Set<string>,
  keptDocuments: readonly DocumentInput[],
  keptShingles: readonly Set<string>[],
  lshCandidateIndexes: ReadonlySet<number>,
  jaccardThreshold: number,
): NearDuplicateDocument | undefined {
  let best: NearDuplicateDocument | undefined;

  for (const index of lshCandidateIndexes) {
    const candidate = candidateDuplicate(
      documentId,
      shingles,
      keptDocuments[index],
      keptShingles[index],
      jaccardThreshold,
      "lsh-band",
    );
    best = chooseBetterDuplicate(best, candidate);
  }

  for (let index = 0; index < keptDocuments.length; index++) {
    if (lshCandidateIndexes.has(index)) continue;
    const candidate = candidateDuplicate(
      documentId,
      shingles,
      keptDocuments[index],
      keptShingles[index],
      jaccardThreshold,
      "exact-jaccard",
    );
    best = chooseBetterDuplicate(best, candidate);
  }

  return best;
}

function candidateDuplicate(
  documentId: string,
  shingles: Set<string>,
  keptDocument: DocumentInput | undefined,
  keptShingles: Set<string> | undefined,
  jaccardThreshold: number,
  candidateReason: NearDuplicateDocument["candidateReason"],
): NearDuplicateDocument | undefined {
  if (!keptDocument || !keptShingles) return undefined;
  const jaccard = jaccardSimilarity(shingles, keptShingles);
  if (jaccard < jaccardThreshold) return undefined;
  return {
    id: documentId,
    duplicateOf: keptDocument.id,
    jaccard,
    candidateReason,
  };
}

function chooseBetterDuplicate(
  current: NearDuplicateDocument | undefined,
  candidate: NearDuplicateDocument | undefined,
): NearDuplicateDocument | undefined {
  if (!candidate) return current;
  if (!current) return candidate;
  if (candidate.jaccard > current.jaccard) return candidate;
  return current;
}

function jaccardSimilarity(left: Set<string>, right: Set<string>): number {
  let intersection = 0;
  for (const value of left) {
    if (right.has(value)) intersection++;
  }
  const union = left.size + right.size - intersection;
  if (union === 0) return 1;
  return intersection / union;
}

export interface DataCleaningReference {
  readonly extractVisibleTextFromHtml: typeof extractVisibleTextFromHtml;
  readonly exactLineDeduplicate: typeof exactLineDeduplicate;
  readonly minhashDeduplicateDocuments: typeof minhashDeduplicateDocuments;
  readonly maskPii: typeof maskPii;
  readonly evaluateGopherQuality: typeof evaluateGopherQuality;
  readonly gopherQualityFilter: typeof gopherQualityFilter;
}

const defaultDataCleaningReference: DataCleaningReference = Object.freeze({
  extractVisibleTextFromHtml,
  exactLineDeduplicate,
  minhashDeduplicateDocuments,
  maskPii,
  evaluateGopherQuality,
  gopherQualityFilter,
});

export function createDataCleaningReference(): DataCleaningReference {
  return defaultDataCleaningReference;
}
