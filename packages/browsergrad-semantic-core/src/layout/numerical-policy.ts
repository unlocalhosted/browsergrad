import type { BuiltinDTypeId } from "./dtype.js";

export interface NumericalPolicy {
  readonly policyId: string;
  readonly inputDTypes: readonly BuiltinDTypeId[];
  readonly computeDType: BuiltinDTypeId;
  readonly accumulatorDType: BuiltinDTypeId;
  readonly outputDType: BuiltinDTypeId;
  readonly rounding: "toward-nearest-ties-even" | "toward-zero" | "toward-positive" | "toward-negative";
  readonly integerOverflow: "wrap" | "saturate" | "error";
  readonly denormals: "preserve" | "flush-to-zero" | "backend-declared";
  readonly contraction: "forbid" | "allow";
  readonly reassociation: "forbid" | "allow";
  readonly reductionOrder: "source-order" | "fixed-tree" | "backend-declared";
  readonly determinism: "deterministic" | "order-deterministic" | "nondeterministic";
  readonly nan: "preserve-payload" | "preserve-class" | "canonicalize" | "backend-declared";
  readonly infinity: "ieee" | "saturate" | "backend-declared";
  readonly signedZero: "preserve" | "ignore-sign" | "backend-declared";
  readonly comparisonPolicyId: string;
}
