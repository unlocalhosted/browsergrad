# ADR-0007: Migrate JIT Flip to Typed Semantics

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT

## Context

`TensorProxy.flip` was a frozen `CUSTOM` NumPy callback. It coerced arbitrary
objects through `int()`, treated bool as an axis, and wrapped every integer
with modulo rank, so an invalid axis could silently target an unrelated
dimension. Its materializing reverse, involutive derivative, transform rule,
and negative-stride backend refusal were invisible to typed IR.

## Decision

`TensorProxy.flip` emits typed `FLIP` with one normalized axis. The initial
surface remains the existing single-axis API. It accepts only exact built-in
or supported NumPy integer scalar types, rejects bool and arbitrary
`__index__`/`__int__` hooks without invoking them, normalizes one negative axis,
and rejects scalar rank or any out-of-range value instead of wrapping modulo
rank.

The executable contract requires one input, a plain closed `{axis}` record,
exact shape/dtype preservation, and a normalized nonnegative axis inside the
input rank. It revalidates those facts at construction and CPU, VJP, vmap, and
ONNX boundaries.

CPU realization returns an owning copy of NumPy's reversed view for every IR
dtype. Closure and symbolic VJP apply the same flip because reversal is
involutive. Functional `grad` therefore contains typed `FLIP`, not `CUSTOM`.
Vmap shifts the normalized logical axis by one to keep the leading batch axis
outside the operation.

ONNX opset 17 admits float32, int32, int64, and bool graphs and emits `Slice`
with exact int64 one-element inputs: start `-1`, end `INT64_MIN`, the normalized
axis, and step `-1`. Other graph dtypes fail explicitly. Tests structurally
decode the protobuf and verify input order, tensor dtype/size, and signed values.

Tensor-plan and WebGPU explicitly refuse the negative-stride profile. This
capability does not widen the canonical Gate 2 positive-stride view profile or
claim portable device indexing. Residency is host-materialized only.

The opaque baseline narrows from 30 constructor calls and 33 operations to 29
constructor calls and 32 operations. `tensor.flip`, the `flip` label, and
`jit.custom.flip.v0` are retired from the current inventory. The executable
registry and frozen original-ID list retain the exact partition of all 39
original opaque operation identities.

## Compatibility and removal

Valid single-axis calls retain values, dtype, owning CPU materialization, and
closure gradients while gaining typed symbolic VJP, functional transforms,
and ONNX export. Bool, float, string, hostile-conversion, scalar-rank, and
formerly modulo-wrapped axes now fail before execution. No actual-device
capability is claimed.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
