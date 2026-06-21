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
export type GopherQualityRule = "non_symbol_word_count" | "average_word_length" | "ellipsis_line_ratio" | "alphabetic_word_ratio";
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
export declare function maskPii(input: string): PiiMaskResult;
export declare function exactLineDeduplicate(lines: readonly string[], options?: DeduplicationOptions): DeduplicationResult;
export declare function extractVisibleTextFromHtml(html: string): string;
export declare function minhashDeduplicateDocuments(documents: readonly DocumentInput[], options?: MinHashDeduplicationOptions): MinHashDeduplicationResult;
export declare function evaluateGopherQuality(text: string, options?: GopherQualityOptions): GopherQualityReport;
export declare function gopherQualityFilter(text: string, options?: GopherQualityOptions): boolean;
//# sourceMappingURL=index.d.ts.map