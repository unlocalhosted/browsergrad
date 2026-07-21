# ADR-0008: Migrate JIT Repeat to Typed Semantics

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`TensorProxy.repeat` was a frozen `CUSTOM` NumPy callback. It coerced arbitrary
objects through `int()`, admitted negative or oversized multipliers until a
later shape/NumPy failure, hid tile-block gradients from functional transforms,
and could not export or report a truthful backend decision. Grad's eager
surface independently cast every repeated result to float32, so integer,
boolean, and float16 behavior diverged across framework packages.

## Decision

JIT `Tensor.repeat` emits typed `REPEAT` with one canonical tuple of tile
multipliers. JIT and Grad accept only exact built-in or fixed-width NumPy
integer scalars, reject bool and arbitrary `__index__`/`__int__` hooks without
invoking them, require at least the input rank and at most 32 axes, and admit
each factor only in `[0, 2^30]`. Zero repeats remain valid. These independent
rank/factor ceilings bound empty-tensor requests whose output element count
alone would not expose hostile multipliers.

The executable JIT contract requires one input, a plain closed `{repeats}`
record plus optional VJP provenance, exact input dtype preservation, and an
output shape derived by left-padding the input shape with ones before
axis-wise multiplication. Construction and CPU, VJP, vmap, and ONNX consumers
revalidate those facts.

CPU realization returns an owning dtype-preserving NumPy tile. Closure and
symbolic VJP reshape the upstream gradient into interleaved
`(repeat, source-extent)` axes, sum every repeat axis, and remove any leading
rank padding. Vmap prepends a unit repeat so the leading batch axis is never
tiled.

ONNX opset 17 admits float32, int32, int64, and bool graphs and emits `Tile`
with one exact signed-int64 repeat vector. Other graph dtypes fail explicitly.
Tensor-plan and WebGPU refuse this operation until a canonical tile/index
layout profile exists; a backend-shaped modulo handler is not introduced.
Residency remains host-materialized.

Grad consumes the same cross-package conformance fixture, returns owning
results in the input dtype, and rejects the same malformed multipliers before
NumPy execution.

The opaque baseline narrows from 29 constructor calls and 32 operations to 28
constructor calls and 31 operations. `tensor.repeat`, the `repeat` label, and
`jit.custom.repeat.v0` are retired from the current inventory. The executable
registry and frozen original-ID list retain the exact partition of all 39
original opaque operation identities.

## Compatibility and removal

Valid repeat values, output shapes, and closure gradients remain compatible.
Integer, boolean, and float16 Grad results now preserve dtype instead of
silently becoming float32. Empty, non-integral, boolean, negative, hostile,
rank-short, over-rank, and oversized factors fail before execution. No WebGPU
execution or canonical tile-layout capability is claimed.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
