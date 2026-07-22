# ADR-0018: Migrate JIT Stack to Typed New-Axis Concatenation

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`stack` was implemented as a frozen `CUSTOM` NumPy callback. It checked equal
input shapes before callback construction, but hid new-axis insertion, dtype
promotion, and gradient selection from symbolic autograd, vmap, export, and
backend selection. The callback also accepted an unbounded number of inputs
and had no output-allocation ceiling. Grad maintained a separate eager path
whose type, dtype, ownership, and mutation behavior could drift independently.

## Decision

JIT now owns `STACK` as typed variadic new-axis concatenation. The public
boundary accepts only a nonempty plain tuple or list of exact tensor values
from one execution session, rejects more than 1,024 inputs, requires identical
input shapes, and accepts only a built-in or fixed-width NumPy integer axis
without invoking arbitrary conversion hooks. The axis is normalized over the
output rank, so both endpoints of `[-input_rank - 1, input_rank]` are valid.
Scalar and empty tensors remain valid when every input has the same shape.

The closed dtype domain and promotion lattice are shared with typed `cat`:
bool, uint8, signed int8/16/32/64, and float16/32/64 use dimensioned-tensor
category precedence; floating dominates integral and bool without widening
from a lower category, the widest dtype wins within a category, and uint8 plus
int8 widens to int16. The output is capped at 256 MiB before allocation.
Construction and the CPU, VJP, vmap, ONNX, and tensor-plan boundaries all
rederive axis, shape, promotion, and resource facts from the input nodes. CPU
realization casts inputs to the declared output dtype and returns an owning
copy.

Closure and symbolic autograd select each source cotangent at its static stack
index. Symbolic VJP uses typed internal `NARROW`, removes the inserted
single-element axis with `RESHAPE`, and casts floating gradients to their
source dtype. Gradient edges exist only for floating outputs and floating
tracked inputs. Vmap shifts the stack axis past one leading mapped axis and
broadcasts captured inputs across that batch.

ONNX opset 17 emits any required per-input `Cast`, then `Unsqueeze` for every
input using one exact int64 axes initializer, followed by `Concat` for the
float32, int32, int64, and bool output profile. Other output dtypes fail
explicitly. Tensor-plan and WebGPU execution refuse until there is a canonical
variadic owning-copy lowering; a host stack is not reported as device
execution. Mutable `out=` remains refused until the framework packages share a
typed mutation/effect contract.

Grad consumes the same cross-package conformance fixture and applies the same
input, axis, shape, promotion, allocation, ownership, mutation, and gradient
rules. The fixture covers three-way and negative axes, mixed-category
promotion, scalar and empty inputs, mixed differentiability, captured-input
vmap, exact ONNX protobuf structure, hostile inputs, post-construction
mutation, resource bounds, and portable-backend refusal.

The opaque baseline narrows from 19 constructor calls and 22 operations to 18
constructor calls and 21 operations. `tensor.stack`, the `stack` callback
label, and `jit.custom.stack.v0` are retired from the current inventory. The
frozen original operation-ID list retains the retired ID, preserving the exact
partition of all 39 original opaque identities. The Gate 0 representative
callback test now exercises still-opaque `pad` rather than a retired operation.

## Compatibility and removal

Valid positive and negative axes, mixed supported dtypes, scalar and empty
inputs, closure autograd, and top-level `stack(tensors, dim=...)` remain
available. Non-tensor inputs, coercive axis objects, unsupported dtypes,
mismatched shapes, oversized requests, mutation through `out=`, and malformed
or mutated IR fail at their semantic boundary.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
