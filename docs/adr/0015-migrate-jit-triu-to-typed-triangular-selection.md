# ADR-0015: Migrate JIT Triu to Typed Triangular Selection

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`triu` was the final triangular-selection surface implemented as a frozen
`CUSTOM` NumPy callback. It coerced `diagonal` during realization, admitted
invalid ranks and dtypes until NumPy, hid its selection derivative from
functional autograd, and left vmap, export, residency, and device behavior
behind generic opaque refusals. Grad exposed only a disconnected top-level
function with a separate dtype path.

## Decision

JIT now owns `TRIU` as a typed unary operation over the final two axes of a
matrix or batch of matrices. `TRIL` and `TRIU` share one construction-time
normalizer and one executable triangular validator. Construction requires rank
at least two, a supported real-numeric or boolean dtype, and an exact built-in
or fixed-width NumPy integer diagonal. Boolean, floating, string, and
user-defined conversion objects fail before graph construction.

For a nonempty upper-triangular matrix the diagonal is normalized to the
unique closed semantic range `[1 - rows, columns]`; empty matrices use zero.
The lower endpoint is the all-input representative and the upper endpoint is
the all-zero representative. Saturating before IR keeps arbitrarily large
Python integers away from NumPy and ONNX without changing selection meaning.
The validator requires one input, exact `diagonal` plus optional VJP
provenance, rank at least two, source/output shape and dtype identity, and the
upper-specific canonical diagonal range.

CPU realization validates the node and returns an owning array in the declared
dtype. Closure and symbolic autograd apply the same upper-triangular selection
to the incoming cotangent. Vmap preserves the final matrix axes when inserting
one leading mapped axis and refuses a captured-only source.

ONNX opset 17 emits `Trilu` with `upper=1` and an exact scalar int64 diagonal
initializer for the float32, int32, int64, and bool exporter profile. Other
dtypes fail explicitly. Tensor-plan and WebGPU execution refuse until a
portable triangular-selection lowering exists; no callback or structural plan
admission substitutes for that backend operation.

Grad now exposes equivalent instance and top-level spellings. Both packages
consume a shared two-variant triangular conformance harness instead of
duplicating lower/upper fixtures. The harness covers batched values, exact
diagonal saturation, empty dimensions, owning dtype preservation, closure and
symbolic gradients, vmap, exact ONNX protobuf fields, hostile inputs, and
backend refusals.

The opaque baseline narrows from 22 constructor calls and 25 operations to 21
constructor calls and 24 operations. `tensor.triu`, the `triu` label, and
`jit.custom.triu.v0` are retired from the current inventory. The frozen
original operation-ID list retains the retired ID, preserving the exact
partition of all 39 original opaque identities.

## Compatibility and removal

Rank-two matrices, higher-rank matrix batches, positive and negative
diagonals, empty matrix dimensions, dtype-preserving outputs, and the top-level
`triu(input, diagonal)` spelling remain available. `Tensor.triu` now provides
the equivalent instance spelling. Invalid rank, implicit diagonal coercion,
unsupported dtypes, and malformed or mutated IR fail at the true boundary.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
