# ADR-0038 — Fix Grad Contiguous Materialization

- **Status:** Accepted
- **Date:** 2026-07-23
- **Owners:** `@unlocalhosted/browsergrad-grad`
- **Retires:** `grad.materialization.contiguous-noop.v0`

## Context

`Tensor.contiguous()` returned `self` for every input, including transposed or
permuted NumPy views whose storage was not C-contiguous. Downstream
`contiguous().view(...)` patterns happened to work only because a later NumPy
reshape could materialize implicitly. The method name therefore made a storage
promise that the implementation did not keep.

## Decision

Implement the exact default-memory-format profile:

- return the same tensor when its backing array is already C-contiguous;
- otherwise allocate one independent owning C-order copy with identical shape
  and storage dtype;
- attach an identity-gradient edge when the source participates in autograd;
  and
- preserve integer, boolean, and floating storage without float32 coercion.

The compatibility fixture separately proves the identity branch, a
non-contiguous float32 copy with bidirectional mutation isolation and backward,
and a dtype-preserving float16 copy.

## Consequences

`contiguous().view(...)` no longer depends on an implicit later reshape copy.
Non-contiguous calls allocate by contract; already-contiguous calls retain
identity and their existing graph. No alternate memory format is accepted in
this profile.

The frozen Grad adapter remains open for detach, cross-dtype conversion,
invalid dtype/device disambiguation, and NumPy interop behavior.
