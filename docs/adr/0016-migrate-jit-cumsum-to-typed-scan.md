# ADR-0016: Migrate JIT Cumsum to Typed Scan

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`cumsum` was implemented as a frozen `CUSTOM` NumPy callback. It preserved the
declared input dtype even when NumPy promoted the realized value, coerced the
axis only during callback execution, hid the scan derivative from functional
autograd, and left vmap, export, residency, and device behavior behind generic
opaque refusals. Grad had a separate eager implementation without the same
validated dtype and failure contract.

## Decision

JIT now owns `CUMSUM` as a typed inclusive scan over one exact axis.
Construction rejects scalar inputs and requires a built-in or fixed-width
NumPy integer axis; negative axes are normalized once. Boolean, floating,
string, and user-defined conversion objects fail before graph construction.
The executable validator requires one input, exact normalized axis and reverse
fields, source/output shape identity, and a supported source/output dtype.

The supported dtype domain is float16, float32, float64, int8, int16, int32,
int64, uint8, and bool. A default scan preserves a floating dtype and promotes
an integral or boolean input to int64. An explicit supported dtype casts before
accumulation. CPU realization returns an owning NumPy array in that exact
declared dtype, including empty tensors, so metadata cannot drift from storage.

Closure and symbolic autograd use an inclusive scan in the opposite direction
and cast the cotangent result back to the source dtype when necessary. Gradient
edges are created only when both source and output are floating. Vmap inserts
one leading mapped axis, shifts the scan axis by one, and refuses a
captured-only source.

ONNX opset 17 emits an exact scalar int64 axis initializer and `CumSum` with
`exclusive=0` and the typed direction. It emits `Cast` before the scan when
the output dtype differs from the input. The exporter profile admits float32,
int32, and int64 outputs and explicitly rejects other dtypes. Tensor-plan and
WebGPU execution refuse until a portable scan lowering exists; neither a host
callback nor structural plan admission substitutes for device execution.

Grad exposes the same instance and top-level spellings and shares one
conformance fixture with JIT. The fixture covers axis normalization, empty
dimensions, owning dtype behavior, default promotion, explicit dtype aliases,
closure and symbolic gradients, vmap, exact ONNX protobuf fields, hostile
inputs, `out=` refusal, and backend boundaries. Mutable `out=` remains refused
until the packages share a typed mutation/effect contract.

The opaque baseline narrows from 21 constructor calls and 24 operations to 20
constructor calls and 23 operations. `tensor.cumsum`, the `cumsum` label, and
`jit.custom.cumsum.v0` are retired from the current inventory. The frozen
original operation-ID list retains the retired ID, preserving the exact
partition of all 39 original opaque identities.

## Compatibility and removal

Positive and negative axes, empty dimensions, floating dtype preservation,
integral promotion, explicit dtype selection, and the top-level
`cumsum(input, dim, dtype=...)` spelling remain available. `Tensor.cumsum`
provides the equivalent instance spelling. Scalar input, implicit axis or dtype
coercion, unsupported dtypes, mutation through `out=`, and malformed or
mutated IR fail at their semantic boundary.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
