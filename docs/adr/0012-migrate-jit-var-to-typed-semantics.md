# ADR-0012: Migrate JIT Variance to Typed Reduction Semantics

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`TensorProxy.var` was a frozen `CUSTOM` NumPy callback. It accepted coercive or
wrapped axes, exposed only the legacy boolean `unbiased` switch, admitted
integer inputs that drifted to float64 at realization, failed full scalar
realization, and hid reduction meaning from symbolic autograd, batching,
export, and backend policy. Grad implemented a separate callback that cast
every result and gradient to float32.

## Decision

JIT `Tensor.var` now emits the typed `VAR` opcode. Its executable contract
requires one floating input, a canonical sorted tuple of static axes, an exact
signed 32-bit correction, an exact keepdims flag, a derived output shape, and
input/output dtype identity. The public surface defaults to correction 1 and
retains `unbiased=True` and `unbiased=False` as compatibility aliases only when
`correction` is absent. It accepts built-in and fixed-width NumPy integer
scalars, rejects bool and conversion hooks, normalizes negative axes once, and
rejects empty, duplicate, or out-of-range axis sets.

CPU realization computes NumPy variance in the declared float16, float32, or
float64 dtype and always returns an owning array, including scalar results.
Closure and symbolic autograd use the centered derivative
`2 * (x - mean) / max(0, N - correction)`. The zero-denominator profile retains
the underlying floating `NaN`/infinity behavior instead of silently replacing
it. Grad consumes the same values, dtype, correction, and refusal fixture and
uses the same derivative without float32 substitution.

Vmap shifts every reduction axis past a leading mapped dimension while keeping
the correction and keepdims contract closed. ONNX opset 17 exports float32
variance as `ReduceMean`, `Sub`, `Mul`, `ReduceSum`, and `Div` with an exact
typed denominator initializer; float16 and float64 export fail explicitly.
Tensor-plan and WebGPU execution refuse until a portable reduction lowering
with the same correction behavior exists. Structural admission is not reported
as device capability.

Every CPU, VJP, vmap, ONNX, and tensor-plan boundary revalidates the typed
record before consuming it. Post-construction mutation and malformed UOps
therefore fail at the first admitted boundary rather than changing semantics.

The opaque baseline narrows from 25 constructor calls and 28 operations to 24
constructor calls and 27 operations. `tensor.var`, the `var` label, and
`jit.custom.var.v0` are retired from the current inventory. The frozen original
operation-ID list retains `jit.custom.var.v0`, preserving the exact partition
of all 39 original opaque identities.

## Compatibility and removal

Default sample variance, population variance through `correction=0` or
`unbiased=False`, normalized negative axes, keepdims, scalar reductions, and
float16/float32/float64 dtype preservation remain available. Integer and bool
inputs, coercive axes or corrections, duplicate/empty axis lists, ambiguous
aliases, and out-of-range corrections now fail explicitly. These are
intentional corrections to the former NumPy-convenience behavior.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
