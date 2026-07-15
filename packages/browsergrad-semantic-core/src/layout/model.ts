import type { DimExpr } from "./dim-expr.js";
import type { BuiltinDTypeId } from "./dtype.js";
import type { WireI64 } from "../schema/integers.js";

export type MemorySpace =
  | { readonly kind: "host" }
  | { readonly kind: "global" }
  | { readonly kind: "shared"; readonly scope: "subgroup" | "workgroup" | "cluster" }
  | { readonly kind: "local"; readonly scope: "invocation" }
  | { readonly kind: "constant" }
  | { readonly kind: "target"; readonly targetId: string; readonly spaceId: string };

export interface AllocationSpec {
  readonly allocationId: string;
  readonly byteLength: DimExpr;
  readonly memorySpace: MemorySpace;
  readonly alignmentBytes: number;
  readonly aliasSetId: string;
}

export interface TensorView {
  readonly viewId: string;
  readonly allocationId: string;
  readonly dtype: BuiltinDTypeId | `${string}:${string}`;
  readonly byteOffset: DimExpr;
  readonly shape: readonly DimExpr[];
  readonly indexMapId: string;
  readonly requiredAlignmentBytes: number;
}

export type IndexExpr =
  | { readonly kind: "const"; readonly value: WireI64 }
  | { readonly kind: "coordinate"; readonly axis: number }
  | { readonly kind: "dimension"; readonly symbolId: string }
  | { readonly kind: "add"; readonly terms: readonly IndexExpr[] }
  | { readonly kind: "mul"; readonly lhs: IndexExpr; readonly rhs: IndexExpr }
  | { readonly kind: "floorDiv" | "ceilDiv" | "mod"; readonly value: IndexExpr; readonly divisor: IndexExpr }
  | { readonly kind: "min" | "max"; readonly values: readonly IndexExpr[] };

export type PredicateExpr =
  | { readonly kind: "bool"; readonly value: boolean }
  | { readonly kind: "equal" | "lessEqual"; readonly lhs: IndexExpr; readonly rhs: IndexExpr }
  | { readonly kind: "and" | "or"; readonly values: readonly PredicateExpr[] }
  | { readonly kind: "not"; readonly value: PredicateExpr };

export interface IndexMap {
  readonly indexMapId: string;
  readonly coordinateRank: number;
  readonly locationUnit: "element" | "byte";
  readonly location: IndexExpr;
  readonly inBounds: PredicateExpr;
}

export type LayoutExpr =
  | { readonly kind: "strided"; readonly shape: readonly DimExpr[]; readonly strides: readonly DimExpr[] }
  | { readonly kind: "compose"; readonly outer: LayoutExpr; readonly inner: LayoutExpr }
  | { readonly kind: "permute"; readonly source: LayoutExpr; readonly axes: readonly number[] }
  | { readonly kind: "slice"; readonly source: LayoutExpr; readonly offsets: readonly DimExpr[]; readonly sizes: readonly DimExpr[]; readonly steps: readonly DimExpr[] }
  | { readonly kind: "broadcast"; readonly source: LayoutExpr; readonly shape: readonly DimExpr[] }
  | { readonly kind: "pad"; readonly source: LayoutExpr; readonly low: readonly DimExpr[]; readonly high: readonly DimExpr[] };
