# ADR-0024: Migrate JIT L1 Loss to a Typed Reduction

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`torch.nn.functional.l1_loss` shared the frozen `_custom_elementwise_loss`
callback with unrelated loss functions. JIT accepted the call into `CUSTOM`,
declared the first input dtype without validating the realized result, retained
only an input-side closure derivative, and left functional gradients, vmap,
ONNX, tensor-plan, and WebGPU behavior to generic opaque-operation refusals.
The callback also deferred shape, dtype, work, and allocation decisions until
NumPy execution. Eager Grad separately cast inputs and results through its
default float32 tensor constructor and disconnected the target derivative.

L1 loss is a same-shape elementwise absolute difference followed by one of
three exact reductions. Treating it as a generic source-shaped callback hides
small but important semantics: both operands are differentiable, equality has
the zero subgradient, float16 needs a declared accumulation policy, empty mean
is NaN while empty sum is zero, and a mapped reduction must reduce per example
rather than across the transform-owned batch prefix.

## Decision

JIT now emits typed `L1_LOSS(input, target)` with the exact canonical argument
`{reduction, batch_rank}`. `reduction` is one of `none`, `sum`, or `mean` and
`batch_rank` is transform-owned, zero at the public boundary, and included in
contract identity. Inputs must be TensorProxy-compatible same-session tensors
(including package Parameters) backed at execution by exact NumPy ndarrays,
with identical shapes and floating dtypes
from float16/32/64. Array subclasses are refused before numerical hooks can
run. Mixed floating inputs use the
existing dimensioned-tensor promotion lattice. Float16 output computes the
absolute difference and any reduction in float32 before an owning float16
store; float32 and float64 compute in their output dtype.

Rank is capped at 32, each extent at 256 Mi elements, output storage at
256 MiB, conservative output/cast/intermediate/both-cotangent workspace at
256 MiB, and element visits at `2^28`. Resource validation treats a zero
extent as one while checking sibling-domain capacity so an empty dimension
cannot conceal a hostile extent. All bounds are checked from metadata before
NumPy allocation or execution. `none` preserves the complete shape, while
`sum` and `mean` preserve only the leading transform batch prefix. Empty
`none`, zero-valued empty `sum`, NaN empty `mean`, and empty gradients are
explicit contract cases.

CPU realization validates the closed record again and returns an owning array.
Closure autograd derives the signed difference from the BufferTable's immutable
registered inputs; eager Grad snapshots it during forward construction.
Symbolic autograd emits internal typed `L1_LOSS_VJP(dy, input, target)` with
the same reduction/batch rank and an exact operand selector. Both paths apply
`sign(input - target)`, use zero at equality, scale mean by the per-example
element count, produce the target cotangent as the exact negation with
normalized positive zero, and cast each cotangent to its source dtype.

Vmap increments `batch_rank`, broadcasts captured operands across the leading
mapped prefix, and retains `none`, `sum`, and `mean` as per-example operations.
This representation composes under nested transforms without rewriting the
public reduction spelling. ONNX opset 17 emits `Sub` then `Abs`, followed by
`ReduceSum` or `ReduceMean` across only non-batch axes; float16 inputs are cast
to float32 for difference/reduction and the result is cast back to float16,
while other mixed inputs are cast to their promoted output dtype. Tensor-plan and direct WebGPU paths
validate and refuse until a canonical loss-reduction lowering exists. Host CPU
support is not reported as portable device execution.

Grad consumes the same package-independent conformance fixture. It preserves
the promoted output dtype, returns owning arrays, snapshots the signed
difference for backward, propagates both input and target gradients in source
dtype, rejects malformed or over-budget requests before NumPy, and preserves
the explicit empty/scalar behavior.

The opaque baseline remains at 11 constructor calls because the shared helper
still serves three loss operations, and narrows from 14 to 13 operations.
`jit.custom.l1-loss.v0` is retired from the current inventory while the frozen
original 39-operation list retains it. The executable framework-operation
registry advances from 25 to 26 typed retirement records.

## Compatibility and removal

Valid same-shape floating tensor calls retain their public spelling and NumPy-backed
results while gaining bounded construction, explicit promotion, both
derivatives, functional transforms, export, and backend decisions. Calls that
previously depended on implicit broadcasting, non-floating inputs, hostile
conversion hooks, unchecked resource consumption, float32 drift, or a
disconnected target now receive deterministic errors or corrected semantics.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen the shared
callback helper, any opaque label, constructor site, operation, or policy.
