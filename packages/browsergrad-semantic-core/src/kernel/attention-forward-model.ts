import type { NumericalPolicy } from "../layout/numerical-policy.js";
import type { FloatBits } from "../schema/float-bits.js";

export const INITIAL_ATTENTION_FORWARD_MAX_DIMENSION = 0xffff_ffffn;
export const INITIAL_ATTENTION_FORWARD_MAX_DEPTH = 256n;

export const INITIAL_ATTENTION_FORWARD_COMPARISON_POLICY = Object.freeze({
  policyId: "browsergrad.attention-forward.f32-abs-relative@1",
  rule: "absolute-or-relative" as const,
  absoluteTolerance: 0.0001,
  relativeTolerance: 0.0001,
  nonFinite: "reject" as const,
  signedZero: "ignore-sign" as const,
});

export const INITIAL_ATTENTION_FORWARD_NUMERICAL_POLICY = Object.freeze({
  policyId: "browsergrad.attention-forward.f32-stable-softmax@1",
  inputDTypes: Object.freeze(["f32", "f32", "f32"] as const),
  computeDType: "f32" as const,
  accumulatorDType: "f32" as const,
  outputDType: "f32" as const,
  rounding: "toward-nearest-ties-even" as const,
  integerOverflow: "error" as const,
  denormals: "backend-declared" as const,
  contraction: "allow" as const,
  reassociation: "allow" as const,
  reductionOrder: "backend-declared" as const,
  determinism: "order-deterministic" as const,
  nan: "backend-declared" as const,
  infinity: "backend-declared" as const,
  signedZero: "ignore-sign" as const,
  comparisonPolicyId: INITIAL_ATTENTION_FORWARD_COMPARISON_POLICY.policyId,
}) satisfies NumericalPolicy;

/** Derives the exact default f32 scale bound by attention-forward@1. */
export function attentionForwardDefaultScaleBits(queryDepth: bigint): FloatBits {
  if (queryDepth <= 0n || queryDepth > INITIAL_ATTENTION_FORWARD_MAX_DEPTH) {
    throw new RangeError("attention-forward query depth is outside the initial profile");
  }
  const value = Math.fround(1 / Math.sqrt(Number(queryDepth)));
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError("attention-forward query depth produced an invalid default scale");
  }
  const float = new Float32Array(1);
  float[0] = value;
  const word = new Uint32Array(float.buffer)[0];
  if (word === undefined) throw new Error("internal: failed to encode attention-forward scale");
  return Object.freeze({
    kind: "float-bits",
    dtype: "f32",
    bits: word.toString(16).padStart(8, "0"),
  });
}

export interface AttentionForwardReadEffect {
  readonly viewId: string;
  readonly access: "read";
}

export interface AttentionForwardWriteEffect {
  readonly viewId: string;
  readonly access: "write";
}

export type AttentionForwardMask =
  | { readonly kind: "none" }
  | {
      readonly kind: "causal";
      readonly orientation: "upper-left";
      readonly predicate: "key-index-less-equal-query-index";
    };

/**
 * Backend-neutral scaled-dot-product attention forward meaning. Query/key/value
 * blocking, workgroup mapping, staged storage, barriers, and backend facilities
 * are deliberately absent and belong to a later schedule artifact.
 */
export interface AttentionForwardOperation {
  readonly operationId: string;
  readonly kind: "scaled-dot-product-attention-forward";
  readonly version: { readonly major: 1; readonly minor: 0 };
  readonly query: AttentionForwardReadEffect;
  readonly key: AttentionForwardReadEffect;
  readonly value: AttentionForwardReadEffect;
  readonly destination: AttentionForwardWriteEffect;
  readonly mask: AttentionForwardMask;
  readonly scale: {
    readonly source: "inverse-square-root-query-depth-rounded-to-f32";
    readonly value: FloatBits;
  };
  readonly inputDomain: {
    readonly query: "finite-f32";
    readonly key: "finite-f32";
    readonly value: "finite-f32";
    readonly scaledScores: "finite-f32-required";
    readonly onlineState: "finite-f32-required";
  };
  readonly score: {
    readonly product: "multiply";
    readonly reduction: "sum";
    readonly reductionAxis: "query-key-depth";
    readonly reductionOrder: "increasing-depth";
    readonly scaleApplication: "after-reduction";
  };
  readonly softmax: {
    readonly kind: "stable-max-subtracted";
    readonly scope: "complete-logical-key-range";
    readonly maximumOrder: "increasing-key";
    readonly exponential: "natural-exp";
    readonly sumOrder: "increasing-key";
    readonly normalization: "divide-by-sum";
    readonly fullyMaskedRows: "forbidden";
  };
  readonly weightedValue: {
    readonly product: "multiply";
    readonly reduction: "sum";
    readonly reductionAxis: "key";
    readonly reductionOrder: "increasing-key";
  };
  readonly numerical: NumericalPolicy;
  readonly autodiff: {
    readonly vjp: "not-defined";
    readonly diagnosticId: "browsergrad.attention-forward-vjp-unavailable";
  };
  readonly phases: {
    readonly order: readonly ["load", "score", "softmax", "weighted-value", "store"];
  };
  readonly overlap: { readonly kind: "forbid-all" };
}
