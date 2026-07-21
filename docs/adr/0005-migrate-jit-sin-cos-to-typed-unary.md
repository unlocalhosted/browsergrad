# ADR-0005: Migrate JIT Sin and Cos to Typed Unary Operations

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT

## Context

`TensorProxy.sin` and `TensorProxy.cos` were frozen `CUSTOM` NumPy callbacks.
Floating inputs executed on CPU with closure gradients, but symbolic VJP,
functional transforms, ONNX export, and portable planning could not represent
their meaning. Integer inputs were worse: the UOp declared the integer input
dtype while NumPy realized float64, violating the graph's dtype contract.
Their derivatives are mutually dependent, so migrating either operation alone
would retain an opaque derivative.

## Decision

`TensorProxy.sin` and `TensorProxy.cos` emit typed `SIN` and `COS` opcodes.
Both require one float16, float32, or float64 input and preserve exact shape and
dtype. Bool and integer inputs fail during construction. The shared typed-unary
validator enforces plain closed arguments, arity, shape, dtype, and the
registry-selected dtype profile at construction and again at every admitted
CPU, VJP, vmap, and ONNX boundary.

CPU realization uses NumPy and returns an owning array cast to the declared
floating dtype. Closure and symbolic VJP define `d(sin(x)) = cos(x)` and
`d(cos(x)) = -sin(x)`. Functional `grad` consumes those typed symbolic rules;
neither derivative introduces `CUSTOM`. Vmap preserves a leading batch axis.
ONNX emits direct opset-17 `Sin` and `Cos` nodes for exporter-supported dtypes.

There is no portable tensor-plan lowering or production WebGPU kernel for
these opcodes in this capability slice. Tensor-plan and WebGPU paths remain
explicit refusals, and residency is host-materialized only.

The opaque baseline narrows from 33 constructor calls and 36 operations to 31
constructor calls and 34 operations. `tensor.sin`, `tensor.cos`, their labels,
and `jit.custom.sin.v0`/`jit.custom.cos.v0` are retired from the current
inventory. The executable registry and frozen original-ID list retain the
exact partition of all 39 original opaque operation identities.

## Compatibility and removal

Valid floating calls retain values, shape, dtype, owning CPU materialization,
and closure gradients while gaining symbolic VJP, functional transforms, and
ONNX export. Integer and bool inputs now fail before execution instead of
declaring an output dtype that CPU realization violates. No actual-device
capability is claimed.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
