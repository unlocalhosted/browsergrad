# ADR-0006: Migrate JIT Clamp to Typed Semantics

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT

## Context

`TensorProxy.clamp` was a frozen `CUSTOM` NumPy callback. It accepted arbitrary
objects through `float()`, including strings and user-controlled conversion
hooks, and stored neither bound in typed IR. Portable transforms could not
represent its piecewise derivative or ONNX `Clip` mapping. Integer tensors
could also realize a promoted NumPy dtype while the UOp still declared the
integer input dtype.

## Decision

`TensorProxy.clamp` emits typed `CLAMP` with exactly one input and a closed
argument record containing normalized optional `min` and `max` floats. At
least one bound is required; both must be finite and ordered. The public
builder accepts only exact built-in or supported NumPy real scalar types,
rejects bool and arbitrary `__float__` hooks without invoking them, and
normalizes admitted bounds once before UOp construction.

The initial typed profile accepts float16, float32, and float64 inputs and
preserves exact shape and dtype. The executable contract revalidates arity,
plain/closed arguments, finite normalized bounds, ordering, shape, and dtype at
construction and at CPU, VJP, vmap, and ONNX boundaries.

CPU realization uses NumPy `clip` and returns an owning array cast to the
declared dtype. Closure and symbolic VJP pass the upstream gradient where the
input is inclusively within every supplied bound and zero it outside. The
symbolic rule emits typed comparisons, mask composition, cast, and multiply;
functional `grad` contains no `CUSTOM`. Vmap preserves a leading batch axis.
ONNX opset 17 emits `Clip` with typed one-element initializers and correct
optional-input positions for min-only, max-only, and two-bound calls.

There is no portable tensor-plan lowering or production WebGPU kernel in this
capability slice. Both paths explicitly refuse `CLAMP`, and residency remains
host-materialized only. `clip` and `clamp_min` reuse the same public builder and
typed contract rather than creating separate operation identities.

The opaque baseline narrows from 31 constructor calls and 34 operations to 30
constructor calls and 33 operations. `tensor.clamp`, the `clamp` label, and
`jit.custom.clamp.v0` are retired from the current inventory. The executable
registry and frozen original-ID list retain the exact partition of all 39
original opaque operation identities.

## Compatibility and removal

Valid finite floating calls retain values, owning CPU materialization, and the
existing inclusive-bound closure gradient while gaining typed symbolic VJP,
functional transforms, and ONNX export. Bool, complex, string, non-finite,
hostile-conversion, and integer-tensor cases now fail before execution instead
of inheriting host coercion or dtype drift. No actual-device capability is
claimed.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
