# ADR-0014: Migrate JIT Tril to Typed Triangular Selection

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`tril` was a frozen `CUSTOM` NumPy callback. It accepted `diagonal` through
unchecked integer coercion, admitted rank-zero and rank-one inputs until
realization, hid its selection derivative from functional autograd, and left
vmap, export, residency, and device behavior behind the generic opaque
refusal. Grad implemented a separate forward-only function and therefore
discarded the otherwise exact triangular-selection gradient.

## Decision

JIT now owns `TRIL` as a typed unary operation over the last two axes of a
matrix or batch of matrices. Construction requires rank at least two, a
supported real-numeric or boolean dtype, and an exact built-in or fixed-width
NumPy integer diagonal. Boolean, floating, string, and user-defined conversion
objects fail before graph construction.

For nonempty matrices the diagonal is normalized to the unique closed semantic
range `[-rows, columns - 1]` of the final two dimensions; empty matrices use
zero because every diagonal has the same empty result. Values outside that
range are equivalent to the all-zero or all-input selection and are saturated
before entering IR. This keeps arbitrarily large Python integers away from
NumPy and ONNX while preserving their exact triangular meaning. The executable
validator requires one input, exact `diagonal` plus optional VJP provenance,
rank at least two, source/output shape and dtype identity, and the canonical
diagonal range.

CPU realization validates the node and returns an owning array in the declared
dtype. Closure and symbolic autograd apply the same lower-triangular selection
to the incoming cotangent; triangular selection is idempotent. Vmap preserves
the final matrix axes when it inserts one leading mapped axis and refuses a
captured-only source instead of manufacturing an implicit batch.

ONNX opset 17 emits `Trilu` with `upper=0` and an exact scalar int64 diagonal
initializer for the float32, int32, int64, and bool profile. Other dtypes fail
explicitly. Tensor-plan and WebGPU execution refuse until a portable
triangular-selection lowering exists; no host callback or structural plan
admission substitutes for that missing backend operation.

Grad exposes the same instance and top-level spellings and consumes the shared
value, dtype, diagonal, gradient, saturation, and refusal fixture. It preserves
source shape and dtype and returns an owning result.

The opaque baseline narrows from 23 constructor calls and 26 operations to 22
constructor calls and 25 operations. `tensor.tril`, the `tril` label, and
`jit.custom.tril.v0` are retired from the current inventory. The frozen
original operation-ID list retains the retired ID, preserving the exact
partition of all 39 original opaque identities.

## Compatibility and removal

Rank-two matrices, higher-rank matrix batches, positive and negative
diagonals, empty matrix dimensions, and dtype-preserving outputs remain
available. Invalid rank, implicit diagonal coercion, unsupported dtypes, and
malformed or mutated IR now fail at the true contract boundary. The new
`Tensor.tril` instance spelling is equivalent to the retained top-level
`tril(input, diagonal)` surface.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
