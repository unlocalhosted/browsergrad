export type SnapshotMismatchKind = "missing" | "unexpected" | "type" | "shape" | "value" | "non-finite";
export interface SnapshotMismatch {
    readonly path: string;
    readonly kind: SnapshotMismatchKind;
    readonly expected?: string;
    readonly actual?: string;
    readonly delta?: number;
    readonly tolerance?: number;
}
export interface SnapshotComparison {
    readonly ok: boolean;
    readonly mismatches: readonly SnapshotMismatch[];
}
export interface SnapshotCompareOptions {
    readonly absoluteTolerance?: number;
    readonly relativeTolerance?: number;
    readonly allowExtraKeys?: boolean;
}
export interface SnapshotOracle {
    compare(actual: unknown): SnapshotComparison;
}
export declare class SnapshotError extends Error {
    constructor(message: string);
}
export declare function createSnapshotOracle(expected: unknown, options?: SnapshotCompareOptions): SnapshotOracle;
export declare function compareSnapshot(actual: unknown, expected: unknown, options?: SnapshotCompareOptions): SnapshotComparison;
//# sourceMappingURL=index.d.ts.map