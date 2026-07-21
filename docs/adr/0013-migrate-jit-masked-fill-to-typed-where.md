# ADR-0013: Migrate JIT Masked Fill to Typed Selection Semantics

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`TensorProxy.masked_fill` was a frozen `CUSTOM` NumPy callback even though the
JIT already had the canonical elementwise `WHERE` operation. The callback
accepted arbitrary mask coercion, allowed a broadcast to enlarge the source
shape, captured an unchecked host scalar, hid its selection derivative from
functional autograd, and left transform, export, and device behavior behind
the generic opaque refusal. Grad independently coerced every mask to bool and
cast every result and gradient to float32.

## Decision

JIT `Tensor.masked_fill` now constructs `WHERE(mask, fill, source)`. The mask
must be an actual bool tensor whose shape broadcasts to, but never enlarges,
the source shape. The fill is an exact built-in or fixed-width NumPy scalar,
normalized once into a scalar `CONST` of the source dtype without invoking
user conversion hooks. Floating sources retain infinities and NaNs; integer
sources require exact finite integral values within dtype bounds; boolean
sources require a boolean fill.

The executable masked-fill contract is identified by an immutable contract ID
on the `WHERE` node. It requires bool condition dtype, three exact semantic
inputs, a scalar source-dtype fill constant, source-shaped output, supported
real-numeric or boolean source dtype, and source/output dtype identity. The
shared `WHERE` validator also closes ordinary selection arity, bool-condition,
broadcast-shape, and provenance structure so masked-fill VJP nodes and existing
generic `where` graphs reuse one selection seam.

CPU realization validates `WHERE`, returns an owning declared-dtype result,
and never relies on NumPy promotion to repair the graph. Closure and symbolic
autograd route source cotangents only through the complement of the mask; the
mask and constant fill remain nondifferentiable. Registering the generic
`WHERE` VJP also makes functional differentiation of ordinary typed selection
composable. Vmap supports a leading mapped source with either a captured
broadcast mask or a correspondingly mapped mask.

ONNX opset 17 emits `Where` for the exact float32, int32, int64, and bool
profile; other dtypes fail explicitly. Tensor-plan and WebGPU execution refuse
until a portable masked-selection lowering exists. The planner now validates
and refuses `WHERE` instead of structurally admitting an operation absent from
the production kernels tensor-plan schema.

Grad consumes the same value, mask, dtype, scalar, and refusal fixture. Both
out-of-place and compatibility in-place forms preserve source dtype and shape;
the out-of-place derivative uses the same mask-complement rule.

The opaque baseline narrows from 24 constructor calls and 27 operations to 23
constructor calls and 26 operations. `tensor.masked-fill`, the `masked_fill`
label, and `jit.custom.masked-fill.v0` are retired from the current inventory.
The frozen original operation-ID list retains the retired ID, preserving the
exact partition of all 39 original opaque identities.

## Compatibility and removal

Valid boolean masks, trailing-rank broadcasting into the source, scalar masks,
float infinities, dtype-preserving integer/boolean fills, and source gradients
remain available. Lists/arrays used as masks, numeric masks, masks that enlarge
the source, hostile scalar hooks, fractional or overflowing integer fills, and
implicit float32 output substitution now fail explicitly. Callers that relied
on Grad's default float32 constructor for a Python boolean list must request
`dtype="bool"` or construct the mask through a comparison.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
