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
export declare function maskPii(input: string): PiiMaskResult;
export declare function exactLineDeduplicate(lines: readonly string[], options?: DeduplicationOptions): DeduplicationResult;
export declare function extractVisibleTextFromHtml(html: string): string;
//# sourceMappingURL=index.d.ts.map