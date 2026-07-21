# ADR-0004: Migrate JIT Abs and Sign to Typed Unary Operations

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT

## Context

`TensorProxy.abs` and `TensorProxy.sign` were NumPy closures in the frozen
`CUSTOM` inventory. Their public methods executed on CPU and carried closure
gradients, but symbolic VJP, functional transforms, ONNX export, and portable
planning could not distinguish their semantics from an arbitrary host
callback. `abs` also depends on `sign` for its derivative, so migrating only
one operation would leave its symbolic derivative opaque.

## Decision

`TensorProxy.abs` and `TensorProxy.sign` emit the typed `ABS` and `SIGN`
opcodes. Both operations accept one real numeric tensor, preserve its exact
shape and declared dtype, and reject `bool` at construction. The executable
framework-operation registry owns these contracts and revalidates nodes at
CPU realization, symbolic VJP, vmap, and ONNX boundaries because UOp argument
dictionaries are not deeply immutable.

CPU realization uses NumPy and returns an owning array cast to the declared
dtype. Closure autograd and symbolic VJP define `d(abs(x))` as `sign(x)` with
the zero subgradient at `x == 0`; `d(sign(x))` is zero. Functional `grad` uses
the symbolic rules. Vmap keeps a leading batch axis while preserving the
unary contract. ONNX emits direct opset-17 `Abs` and `Sign` nodes for dtypes
already supported by the exporter.

There is no portable tensor-plan lowering or production WebGPU kernel for
these opcodes in this capability slice. Tensor-plan and WebGPU paths therefore
fail explicitly instead of materializing through `CUSTOM` or claiming device
execution. The supported residency profile is host-materialized only.

The opaque baseline narrows from 35 constructor calls and 38 operations to 33
constructor calls and 36 operations. `tensor.abs`, `tensor.sign`, their labels,
and `jit.custom.abs.v0`/`jit.custom.sign.v0` are retired from the current
inventory. The registry and frozen `originalOperationIds` retain an exact
partition of all 39 original opaque operation identities.

## Compatibility and removal

Valid real-numeric calls retain values, shape, dtype, owning CPU
materialization, and closure-gradient behavior. Symbolic differentiation,
functional transforms, and ONNX export become available. Boolean inputs now
fail at the semantic boundary rather than inheriting NumPy-specific behavior.
No actual-device capability is claimed.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
