# ADR-0017: Migrate JIT Cat to Typed Concatenation

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`cat` was implemented as a frozen `CUSTOM` NumPy callback. Shape and axis
checks happened before callback construction, but the node still hid
concatenation, dtype promotion, and gradient splitting from symbolic autograd,
vmap, export, and backend selection. The callback also had no bounded variadic
input or output-allocation contract. Grad maintained a separate eager path and
silently accepted non-tensor values through its generic wrapping helper.

## Decision

JIT now owns `CONCAT` as typed variadic concatenation over one normalized
existing axis. The public boundary accepts only a nonempty plain tuple or list
of exact tensor values from one execution session, rejects more than 1,024
inputs, and requires a built-in or fixed-width NumPy integer axis without
invoking arbitrary conversion hooks. All substantive inputs have equal rank
and equal extents outside the selected axis. The PyTorch compatibility form of
a one-dimensional empty tensor with shape `(0,)` may accompany any valid
substantive input and contributes zero elements; an all-`(0,)` request remains
rank one. Scalar inputs are rejected.

The closed dtype domain is bool, uint8, signed int8/16/32/64, and
float16/32/64. Promotion follows dimensioned-tensor category precedence:
floating dominates integral and bool without being widened by a lower
category, the widest floating dtype wins among floating inputs, the widest
signed dtype wins among signed inputs, and uint8 plus int8 widens to int16.
The output is capped at 256 MiB before allocation. Construction and the CPU,
VJP, vmap, ONNX, and tensor-plan boundaries all rederive axis, shape, promotion,
segment sizes, legacy-empty positions, and the resource bound from the input
nodes. CPU realization casts inputs to the declared output dtype and returns an
owning copy.

Closure and symbolic autograd split the cotangent at the static input segment
boundaries. Symbolic VJP uses the typed internal `NARROW` primitive, reshapes a
rank-mismatched legacy empty gradient back to `(0,)`, and casts each floating
gradient to its source dtype. Gradient edges exist only for floating outputs
and floating tracked inputs. Vmap shifts the concatenation axis past one
leading mapped axis and broadcasts captured inputs across that batch; it omits
rank-mismatched legacy empties without changing their promotion contribution.

ONNX opset 17 emits `Cast` for each mismatched input and one exact-axis
`Concat` for the float32, int32, int64, and bool output profile. A
rank-mismatched `(0,)` compatibility input is omitted from the ONNX node after
its dtype has participated in promotion. Other output dtypes and gradient-only
`NARROW` export fail explicitly. Tensor-plan and WebGPU execution refuse until
there is a canonical variadic owning-copy lowering; a host concatenation is not
reported as device execution. Mutable `out=` remains refused until the
framework packages share a typed mutation/effect contract.

Grad consumes the same cross-package conformance fixture and applies the same
input, axis, shape, promotion, legacy-empty, allocation, ownership, mutation,
and gradient rules. The fixture also covers mixed-category promotion,
zero-width substantive inputs, all-empty inputs, captured-input vmap, exact
ONNX protobuf structure, hostile inputs, post-construction mutation, and
portable-backend refusal.

The opaque baseline narrows from 20 constructor calls and 23 operations to 19
constructor calls and 22 operations. `tensor.cat`, the `cat` callback label,
and `jit.custom.cat.v0` are retired from the current inventory. The frozen
original operation-ID list retains the retired ID, preserving the exact
partition of all 39 original opaque identities.

## Compatibility and removal

Valid positive and negative axes, mixed supported dtypes, zero-width inputs,
the legacy `(0,)` empty form, closure autograd, and the top-level
`cat(tensors, dim=...)` spelling remain available. Non-tensor inputs,
coercive axis objects, unsupported dtypes, incompatible ranks or extents,
oversized requests, mutation through `out=`, and malformed or mutated IR fail
at their semantic boundary.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
