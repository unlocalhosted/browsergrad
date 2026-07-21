# ADR-0009: Migrate JIT Repeat Interleave to Typed Semantics

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`TensorProxy.repeat_interleave` was a frozen `CUSTOM` NumPy callback. It
coerced arbitrary repeat and axis objects through `int()`, normalized every
axis through modulo, admitted negative repeat counts until NumPy rejected
them, hid its reduction gradient from functional transforms, and had no
portable export or backend contract. Grad's eager implementation also cast
all results and gradients to float32, diverging for float16, integer, and
boolean inputs.

## Decision

JIT `Tensor.repeat_interleave` emits typed `REPEAT_INTERLEAVE` with a canonical
non-negative repeat count and normalized selected axis. JIT and Grad accept
only exact built-in or fixed-width NumPy integer scalars, reject bool and
arbitrary `__index__`/`__int__` hooks without invoking them, require a real
input axis, and admit repeat counts only in `[0, 2^30]`. Zero repeats remain
valid. The repeat ceiling applies independently of output size, so empty
inputs cannot conceal resource-hostile counts.

The executable JIT contract requires one input, a plain closed
`{axis, repeats}` record plus optional VJP provenance, exact input dtype
preservation, and an output shape derived by multiplying only the selected
axis. Construction and CPU, VJP, vmap, and ONNX consumers revalidate those
facts.

CPU realization returns an owning dtype-preserving NumPy result. Closure and
symbolic VJP split the expanded axis into `(source-extent, repeats)` and sum
the repeat block. Vmap shifts the selected axis past the leading batch axis,
which is never repeated.

ONNX opset 17 admits float32, int32, int64, and bool graphs and lowers the
operation exactly through `Unsqueeze`, `Tile`, and `Reshape` with signed-int64
axis, repeat-vector, and output-shape initializers. Other graph dtypes fail
explicitly. Tensor-plan and WebGPU refuse the operation until a canonical
selected-axis replication layout profile exists; no source-shaped backend
handler is introduced. Residency remains host-materialized.

Grad consumes the same cross-package conformance fixture, returns owning
results in the input dtype, preserves gradient dtype, and rejects malformed
repeat or axis values before NumPy execution.

The opaque baseline narrows from 28 constructor calls and 31 operations to 27
constructor calls and 30 operations. `tensor.repeat-interleave`, the
`repeat_interleave` label, and `jit.custom.repeat-interleave.v0` are retired
from the current inventory. The executable registry and frozen original-ID
list retain the exact partition of all 39 original opaque operation
identities.

## Compatibility and removal

Valid repeat-interleave values, output shapes, negative-axis normalization,
and closure gradients remain compatible. Grad now preserves float16, integer,
and boolean output dtype instead of silently returning float32. Boolean,
non-integral, hostile, negative, oversized, scalar-axis, and out-of-range
inputs fail before execution. No WebGPU execution or canonical replication
layout capability is claimed.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
