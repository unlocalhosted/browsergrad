import type { BuiltinDTypeId } from "../layout/dtype.js";
import type { FloatBits } from "../schema/float-bits.js";

export type InvalidSourcePolicy =
  | { readonly kind: "reject" }
  | { readonly kind: "fill"; readonly value: FloatBits };

export interface ViewCopySourceEffect {
  readonly viewId: string;
  readonly access: "read";
  readonly invalidSource: InvalidSourcePolicy;
}

export interface ViewCopyDestinationEffect {
  readonly viewId: string;
  readonly access: "write";
}

export interface ViewCopyOperation {
  readonly operationId: string;
  readonly kind: "view-copy";
  readonly version: { readonly major: 1; readonly minor: 0 };
  readonly dtype: BuiltinDTypeId;
  readonly source: ViewCopySourceEffect;
  readonly destination: ViewCopyDestinationEffect;
  readonly overlap: { readonly kind: "forbid" };
}
