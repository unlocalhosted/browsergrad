# ADR-0021: Migrate JIT Top-K to Typed Partial Selection

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`Tensor.topk` returned values and indices through two independent frozen
`CUSTOM` callbacks. Each callback performed its own full `argsort`, so reading
both public outputs duplicated ordering work and temporary allocation. The
callbacks used dtype-changing negation for largest selection, accepted
coercive arguments, declared no allocation or work limits, silently
disconnected autograd, and hid the paired-output relationship from vmap, ONNX,
and backend selection. Grad separately returned float32 values and float32
indices regardless of source dtype.

## Decision

JIT now represents top-k as paired typed operations.
`TOPK_INDICES(source)` owns the canonical selection and
`TOPK_VALUES(source, indices)` gathers through that exact paired index node.
Both carry the same canonical `axis`, `k`, `largest`, and `sorted` fields and
derive the selected-axis output shape together. The public top-level and tensor
surfaces accept an exact tensor, an exact built-in or fixed-width NumPy integer
`k`, `None` or an exact integer dimension, and exact bool flags. `out=` remains
an explicit refusal. Scalar inputs remain outside the first typed profile.

The closed dtype domain is bool, uint8, signed int8/16/32/64, and
float16/32/64. Values preserve source dtype and indices are owning int64.
`k` may range from zero through the selected-axis extent. Rank is capped at 32,
every extent at 268,435,456, and the selected axis at 1,048,576 elements.
Paired output storage and a conservative NumPy selection-workspace projection
are each capped at 256 MiB. The workspace projection accounts for the full
int64 `argpartition` permutation plus selected values and ordering/remap
buffers. The same change adds a 256 MiB conservative workspace ceiling to
typed full sort, closing a temporary-allocation gap in ADR-0020.

CPU realization computes the indices exactly once. It uses `argpartition` to
select in linear expected work without negating source values. When
`sorted=True`, it stably orders only the selected values and remaps the selected
indices; the dominant work is selection plus `k log k`, not two full
`n log n` sorts. When `sorted=False`, it preserves the partial-selection order.
PyTorch does not promise stable indices for ties, so the typed contract does
not claim stable tie identity. The pinned NumPy implementation remains
deterministic for one exact runtime and request. Unsigned values, minimum signed
integers, NaNs, ties, empty `k=0`, and the common `(4, 50257) -> (4, 10)` shape
are covered explicitly.

Indices are discrete and nondifferentiable. Closure and symbolic autograd for
values scatter the cotangent through the immutable selected permutation;
symbolic VJP emits typed `SCATTER_ADD`. Grad captures a private index copy so
mutating the returned index tensor cannot alter backward. Vmap shifts the
selected axis past its leading mapped dimension.

ONNX opset 17 emits selected-`k` `TopK` for indices and `GatherElements` for
values on float32, int32, and int64. `largest` and `sorted` remain exact node
attributes. `k=0` export fails explicitly because ONNX requires positive K.
Tensor-plan and direct WebGPU execution validate and refuse until canonical
portable partial-selection semantics and kernels exist.

Grad consumes the same cross-package conformance fixture and applies the same
shape, dtype, partial-selection, ownership, gradient, hostile-input, and
resource rules. The torch alias exposes both method and top-level spellings.

The opaque baseline narrows from 15 constructor calls and 18 operations to 13
constructor calls and 16 operations. `tensor.topk-indices`,
`tensor.topk-values`, `topk_indices`, `topk_values`,
`jit.custom.topk-indices.v0`, and `jit.custom.topk-values.v0` are retired from
the current inventory. The frozen original operation-ID list retains both
retired IDs, preserving the exact partition of all 39 original opaque
identities. Gate 0 now retains `einsum` and `scatter` as its forward-only
callback representatives.

## Compatibility and removal

Valid ranked top-k calls retain their two-result surface with exact source
dtype, int64 indices, optional sorted output, `k=0`, connected floating value
autograd, and substantially less duplicated work. Coercive arguments, scalar
inputs, unsupported dtypes, invalid axes or `k`, oversized work/output,
mutation, malformed or mismatched paired IR, unavailable exporters, and
portable backends fail at their semantic boundary.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
