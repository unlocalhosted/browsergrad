# ADR-0020: Migrate JIT Sort to Typed Axis Ordering

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`torch.sort` returned values and indices through two independent frozen
`CUSTOM` NumPy callbacks. The callbacks hid the relationship between the two
outputs and disconnected both from symbolic autograd, vmap, ONNX, and portable
backend selection. They also accepted coercive arguments, exposed no resource
limits, did not validate realized shape or dtype, and used dtype-changing
negation for descending order. Grad maintained a separate eager path without a
shared validation or tie-order contract.

## Decision

JIT now represents sort as a paired typed operation. `SORT_INDICES(source)`
owns the canonical permutation; `SORT_VALUES(source, indices)` gathers through
that exact paired index node. Both nodes carry the same normalized axis,
descending, and stable fields, have the same source shape, and are validated
again at every consumer boundary. The public surface accepts an exact tensor,
an exact built-in or fixed-width NumPy integer dimension, and exact bool flags.
The `out` mutation form remains an explicit refusal.

The supported dtype domain is bool, uint8, signed int8/16/32/64, and
float16/32/64. Values preserve source dtype and indices are owning int64.
Scalar dimensions `-1` and `0` are accepted. Rank is capped at 32, every extent
at 268,435,456 elements, the selected axis at 1,048,576 elements, and combined
values-plus-indices output storage at 256 MiB before allocation. These checks
also reject zero-sized tensors whose nonselected extents carry oversized host
integers.

CPU realization computes one stable permutation and gathers values through it.
Ascending order uses stable `argsort`. Descending order stably sorts a reversed
source, reverses the resulting permutation, and remaps indices; this preserves
the original order of equal elements without negating unsigned integers,
minimum signed integers, or floating NaNs. `stable=False` uses the same
deterministic stable implementation, which is compatible with the weaker
unspecified tie-order contract. Both results are fresh owning arrays.

Indices are discrete and nondifferentiable. Closure and symbolic autograd for
values scatter the cotangent through the captured permutation;
symbolic VJP emits typed `SCATTER_ADD`, so no opaque callback enters the
derivative graph. Mutation of the user-visible eager indices cannot change the
captured Grad backward permutation. Vmap shifts the selected axis past its
leading batch dimension; scalar values remain the mapped source and scalar
indices become broadcast int64 zero.

ONNX opset 17 lowers indices to full-axis `TopK` and values to
`GatherElements` for float32, int32, and int64. The exporter fixes `sorted=1`,
sets `largest` from descending order, and refuses scalar or empty selected axes
because ONNX `TopK` requires a positive `K`. The tensor planner validates and
then refuses both operations until canonical portable ordering and gather
lowerings exist. Direct WebGPU support remains absent and explicit.

Grad consumes the same cross-package conformance fixture and applies the same
ordering, tie, dtype, ownership, gradient, hostile-input, empty/scalar, and
resource-bound rules.

The opaque baseline narrows from 17 constructor calls and 20 operations to 15
constructor calls and 18 operations. `tensor.sort-indices`,
`tensor.sort-values`, `sort_indices`, `sort_values`,
`jit.custom.sort-indices.v0`, and `jit.custom.sort-values.v0` are retired from
the current inventory. The frozen original operation-ID list retains both
retired IDs, preserving the exact partition of all 39 original opaque
identities. Gate 0 now retains only `einsum`, `scatter`, `topk_indices`, and
`topk_values` as representative forward-only callbacks.

## Compatibility and removal

Valid sort calls retain their two-result surface with deterministic stable
ties, exact dtype preservation, owning results, and connected floating value
autograd. Coercive dimensions or flags, unsupported dtypes, invalid axes,
oversized requests, output mutation, malformed or mismatched paired IR, and
unavailable portable backends fail at their semantic boundary.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
