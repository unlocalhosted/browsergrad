# ADR-0002: Migrate JIT Expand to Typed Broadcast

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad, kernels

## Context

`TensorProxy.expand` was one of the opaque `CUSTOM` callbacks frozen by
ADR-0001. The JIT already had a `BROADCAST_TO` opcode, NumPy handler, vmap
rule, ONNX `Expand` mapping, tensor-plan lowering, and rank-at-most-four f32
WebGPU kernel, but the advertised tensor surface did not use them. It instead
captured a NumPy closure, which blocked symbolic VJP, vmap, ONNX, tensor-plan
execution, and GPU residency.

## Decision

`TensorProxy.expand` emits `BROADCAST_TO` with one input, exact output shape,
preserved dtype, and a closed shape argument. One shared validator enforces
arity, argument fields, shape identity, dtype preservation, rank direction,
and broadcast compatibility at construction and again at transform, export,
CPU realization, and tensor-plan boundaries. Revalidation is required because
legacy UOp argument dictionaries are not deeply immutable.

The symbolic VJP sums every expanded axis back to the input shape. Vmap keeps
the batch axis outside the declared broadcast. ONNX emits `Expand`. The CPU
handler materializes an owning copy. The tensor-plan route admits the typed op
without `CUSTOM`; the current production WebGPU backend supports only its
existing non-empty f32, rank-at-most-four profile and otherwise refuses at the
backend contract. Materializing and resident tensor-plan routes are both
explicitly tested.

The frozen opaque baseline is narrowed from 36 constructor calls and 39
operations to 35 constructor calls and 38 operations. `tensor.expand`, the
`expand` label, and `jit.custom.expand.v0` are retired from that inventory.
The validator derives the exact constructor total from the per-file frozen
counts instead of embedding the historical total, while still comparing every
site, operation, label, field, definition, fixture, and content hash.
The frozen planner decision definition is re-pinned because it now revalidates
typed `BROADCAST_TO` nodes before scheduling; its `CUSTOM` refusal is unchanged.

Grad's eager `Tensor.expand` adopts the same dimension-validation and dtype-
contiguous materialization at this decision's acceptance. ADR-0046
subsequently supersedes only that eager ownership choice with a storage-sharing
zero-stride view. Its former float16-to-f32 substitution remains removed, and
invalid shapes remain rejected before NumPy execution.

## Compatibility and removal

Valid `expand` calls retain the public surface, values, dtype, and
closure-autograd behavior. ADR-0046 changes eager ownership from materialized
copy to zero-stride view. Shape capture is intentionally stricter:
booleans, non-index integers such as floats, negative dimensions other than
`-1`, leading `-1`, rank reduction, and incompatible non-singleton expansion
now fail before execution instead of relying on NumPy or integer truncation.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This migration narrows that debt and
does not widen any remaining caller, label, or policy.
