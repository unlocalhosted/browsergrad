# ADR-0010: Migrate JIT Product Reduction to Typed Semantics

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`TensorProxy.prod` was a frozen `CUSTOM` NumPy callback. It accepted arbitrary
iterables and scalar conversion hooks as axes, silently wrapped duplicate or
out-of-range dimensions, could build a declared scalar whose NumPy callback
returned a rejected scalar object, and hid reduction meaning from functional
transforms and export. Its derivative returned zero for every zero input,
which is wrong when a reduction group contains exactly one zero. Grad's eager
implementation separately cast every result and gradient to float32.

## Decision

JIT `Tensor.prod` emits typed `PROD` with a canonical strictly increasing tuple
of normalized axes and an exact boolean `keepdims`. JIT and Grad accept only
`None`, exact built-in or fixed-width NumPy integer scalars, or plain nonempty
tuple/list axis collections. They reject bool, arbitrary iterables and
conversion hooks, duplicate normalized axes, scalar explicit axes,
out-of-range axes, simultaneous `axis`/`dim`, and non-boolean keep flags before
NumPy execution. `axis=None` reduces every input axis; scalar full reduction
uses the canonical empty-axis tuple.

The executable JIT contract requires one input, a plain closed
`{axes, keepdims}` record plus optional VJP provenance, exact input dtype
preservation, canonical axes, and an output shape derived only from those axes
and the keep-dimension flag. Construction and CPU, VJP, vmap, and ONNX
consumers revalidate those facts.

CPU realization passes the exact input dtype to NumPy product and always
returns an owning ndarray, including scalar outputs. Closure and symbolic VJP
count zeros per reduction group, multiply nonzero values, use the quotient rule
only for zero-free groups, place the nonzero product at the sole zero when
exactly one exists, and return zero when two or more exist. Vmap shifts every
logical reduction axis past the leading batch axis.

ONNX opset 17 admits float32, int32, and int64 graphs and emits `ReduceProd`
with exact static axes and keepdims attributes. Bool, float16, and other graph
dtypes fail explicitly. Tensor-plan and WebGPU refuse the operation until a
portable product-reduction lowering exists. Residency remains host-materialized.

Grad consumes the same cross-package conformance fixture and now returns
owning scalar or tensor results in the input dtype while preserving backward
dtype and the same zero-aware derivative.

The opaque baseline narrows from 27 constructor calls and 30 operations to 26
constructor calls and 29 operations. `tensor.prod`, the `prod` label, and
`jit.custom.prod.v0` are retired from the current inventory. The executable
registry and frozen original-ID list retain the exact partition of all 39
original opaque operation identities.

## Compatibility and removal

Valid single-axis, multi-axis, negative-axis, keepdims, full-reduction, scalar,
and empty-dimension products remain available. Scalar JIT products now realize
as owning zero-dimensional arrays instead of failing. Grad preserves float16,
integer, and boolean output dtype instead of returning float32. The derivative
for one-zero groups is corrected. Ambiguous, coercive, duplicate, empty-axis,
and malformed keep-flag requests now fail early. No WebGPU execution capability
is claimed.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
