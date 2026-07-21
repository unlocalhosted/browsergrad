import type { WireU64 } from "../schema/integers.js";

export interface LogicalGemmTileExtent {
  readonly m: WireU64;
  readonly n: WireU64;
  readonly k: WireU64;
}

export interface LogicalGemmTileReadEffect {
  readonly viewId: string;
  readonly access: "read";
}

export interface LogicalGemmTileWriteEffect {
  readonly viewId: string;
  readonly access: "write";
}

/**
 * Backend-neutral logical GEMM tile meaning. Physical invocation mapping,
 * staging, vectorization, workgroups, subgroups, and target instructions are
 * intentionally absent and belong to a later schedule artifact.
 */
export interface LogicalGemmTileOperation {
  readonly operationId: string;
  readonly kind: "logical-gemm-tile";
  readonly version: { readonly major: 1; readonly minor: 0 };
  readonly lhs: LogicalGemmTileReadEffect;
  readonly rhs: LogicalGemmTileReadEffect;
  readonly destination: LogicalGemmTileWriteEffect;
  readonly logicalTile: LogicalGemmTileExtent;
  readonly boundary: {
    readonly lhs: "zero-fill";
    readonly rhs: "zero-fill";
    readonly destination: "mask-outside-logical-shape";
  };
  readonly accumulation: {
    readonly inputDType: "f32";
    readonly accumulatorDType: "f32";
    readonly outputDType: "f32";
    readonly product: "multiply";
    readonly reduction: "sum";
    readonly reductionOrder: "increasing-k";
    readonly rounding: "toward-nearest-ties-even";
    readonly contraction: "forbid";
    readonly reassociation: "forbid";
  };
  readonly phases: {
    readonly order: readonly ["load", "accumulate", "store"];
    readonly participation: "masked-full-logical-tile";
  };
  readonly overlap: { readonly kind: "forbid-all" };
}
