import type {
  DeduplicationOptions,
  DeduplicationResult,
  DocumentInput,
  DuplicateLine,
  MinHashDeduplicationOptions,
  MinHashDeduplicationResult,
  NearDuplicateDocument,
} from "./data-types.js";

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
