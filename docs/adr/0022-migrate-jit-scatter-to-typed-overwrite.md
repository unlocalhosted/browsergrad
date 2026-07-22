# ADR-0022: Migrate JIT Scatter to Typed Overwrite Semantics

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`Tensor.scatter` lowered through a frozen `CUSTOM` callback. It coerced the
axis and index, broadcast arbitrary source shapes, cast updates and eager Grad
outputs to float32, inherited NumPy negative-index behavior, declared no
allocation or work ceiling, silently disconnected both differentiable inputs,
and was unavailable to vmap, ONNX, and portable backend selection. Duplicate
overwrite destinations inherited an implementation-dependent assignment order
without an explicit semantic or gradient contract.

PyTorch's overwrite scatter requires a LongTensor index and same-rank target,
index, and tensor source. Tensor-source backward is defined only when source and
index shapes match. PyTorch also warns that duplicate indices make overwrite
selection nondeterministic and its gradient incorrect. ONNX `ScatterElements`
likewise requires unique indices for reduction `none`. BrowserGrad therefore
does not silently bless one accidental duplicate-write order as portable
meaning.

## Decision

JIT now emits typed `SCATTER(target, index, source)` with one canonical exact
axis. The target and index have the same nonzero rank, capped at 32. The index
is int64. A tensor source must have exactly the index shape and target dtype;
the scalar overload accepts only a built-in or fixed-width NumPy real scalar
that is exactly normalized into the target dtype. The closed target/source
domain is bool, uint8, signed int8/16/32/64, and float16/32/64. Deprecated
`reduce=` spellings fail explicitly and remain a separate future
`scatter_reduce` capability.

Each index extent is no greater than the corresponding target extent. This is
required on the selected axis as well because the typed overwrite profile
requires unique destinations within every index fiber. Runtime validation
rejects negative, out-of-range, or duplicate destinations before assignment.
Empty indices return an owning copy of the target. Output storage is capped at
256 MiB. A separate 256 MiB conservative workspace ceiling covers the owning
output plus a full sorted int64 index copy and adjacent-equality temporary used
for deterministic duplicate rejection. Every target extent is bounded even
when another zero extent makes total storage zero.

CPU realization copies the target exactly once and performs one
`put_along_axis` overwrite after validation. The scalar source stays scalar in
IR and is broadcast by the indexed assignment, avoiding an index-shaped
materialization. Valid tensor updates are never broadcast or cast.

For floating tensors, closure and symbolic autograd return the incoming
cotangent with overwritten target positions zeroed and gather the source
cotangent from those same unique destinations. Symbolic VJP uses typed
`SCATTER` and `INDEX`; the index is discrete. Grad snapshots the index for
backward so caller mutation cannot change the derivative. Vmap shifts the axis
past the leading batch and explicitly broadcasts captured target, index, or
tensor source inputs; a scalar source remains scalar.

ONNX opset 17 emits `ScatterElements` for float32, int32, int64, and bool. A
scalar source receives an explicit `Expand` to the index shape in the exported
graph. The unique, nonnegative in-range index condition remains the admitted
program contract. Tensor-plan and direct WebGPU execution validate and refuse
until canonical deterministic overwrite lowering exists.

Grad consumes the same cross-package conformance fixture and now exposes both
method and top-level/torch-alias spellings with dtype-preserving owning output,
target and tensor-source gradients, scalar updates, duplicate refusal, and the
same resource ceilings.

The opaque baseline narrows from 13 constructor calls and 16 operations to 12
constructor calls and 15 operations. `tensor.scatter`, `scatter`, and
`jit.custom.scatter.v0` are retired from the current inventory. The frozen
original operation-ID list retains the retired identity, preserving the exact
partition of all 39 original opaque operations. Gate 0 retains `einsum` as its
sole forward-only callback representative.

## Compatibility and removal

Valid unique-destination ranked scatter calls retain owning out-of-place
overwrite behavior while gaining exact dtype, scalar and tensor-source
semantics, connected autograd, transforms, export, and bounded work. Coercive
arguments, broadcasting tensor updates, implicit dtype conversion, duplicate
or invalid destinations, scalar targets, reduction spellings, oversized work,
mutation, malformed IR, and unavailable portable backends fail at their true
semantic boundary.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
