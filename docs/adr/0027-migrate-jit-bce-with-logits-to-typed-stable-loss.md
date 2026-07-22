# ADR-0027: Migrate JIT BCE With Logits to Typed Stable Loss

- **Status:** accepted
- **Date:** 2026-07-22
- **Owners:** JIT and Grad
- **Gate:** 6 — framework convergence

## Context

`torch.nn.functional.binary_cross_entropy_with_logits` constructed a dedicated
JIT `CUSTOM` callback. It converted both operands to float32 while declaring
the logits dtype, differentiated only logits, admitted coercive target values,
and left functional differentiation, vmap, export, tensor-plan, and WebGPU
behavior behind opaque handling. Grad independently exposed only mean
reduction, converted targets to float32, returned float32, used an
overflow-prone direct sigmoid, and disconnected the target.

BCE with logits is not probability-domain BCE with a hidden sigmoid. Its
stable forward is based on `log_sigmoid(input)`, its input derivative is a
stable sigmoid minus target, and its target derivative is exactly negative
input for the unweighted profile. Those semantics must remain inspectable in
typed IR and must work for finite logits whose magnitude would overflow a
direct exponential.

## Decision

JIT emits binary `BINARY_CROSS_ENTROPY_WITH_LOGITS(logits, target)` with
canonical `{reduction, batch_rank}` arguments. Internal
`BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP(dy, logits, target)` carries those
arguments plus the selected operand. Both operations revalidate their complete
structural contract at every CPU, VJP, vmap, ONNX, and tensor-plan boundary.

The first profile is closed as follows:

- Both operands are same-shape `float16`, `float32`, or `float64` tensors from
  the same session. Dimensioned tensor promotion determines the output dtype;
  a half-only result computes and accumulates in float32. Mixed floating dtype
  support is an explicit BrowserGrad extension of the existing surface.
- Forward evaluates `(1 - target) * logits + softplus(-logits)` using
  `max(-logits, 0) + log1p(exp(-abs(logits)))`. It never evaluates
  `exp(logits)` and remains finite for the admitted finite `+/-1000` fixtures.
  Exact zero results are normalized to positive zero.
- The logits derivative computes a branch-stable sigmoid from
  `exp(-abs(logits))` and subtracts target. The target derivative is
  `-logits`. Closure and symbolic autograd preserve both derivatives and return
  cotangents in each source dtype. Grad snapshots both derivatives at forward
  time.
- Reductions are exactly `none`, `sum`, and `mean`. Empty sum is zero, empty
  mean is NaN, and vmap-owned leading dimensions remain outside each
  per-example reduction. CPU outputs are owning arrays.
- Rank is bounded at 32; individual extents, output bytes, aggregate visits,
  and conservative peak workspace are bounded before allocation. The 36-visit
  factor covers stable forward, both derivative directions, and reductions.
  Workspace counts input casts, output, both retained source-dtype cotangents,
  four compute buffers, and one mask. Zero extents do not hide hostile
  capacity in another axis.
- ONNX opset 17 emits promoted `Neg`/`Softplus`, `Sub`/`Mul`/`Add`, the exact
  non-batch reduction, and output cast. Tensor-plan and direct WebGPU execution
  remain explicit refusals until a canonical stable loss-reduction lowering
  and kernel exist.

PyTorch documents targets in `[0, 1]`, but its current native logits loss does
not perform the runtime domain rejection used by probability BCE. This profile
therefore preserves the native algebra rather than claiming a validation that
the source operation does not perform. Non-finite propagation is not promoted
to a finite-result claim.

The existing JIT surface does not accept optional `weight`, `pos_weight`, or
the deprecated `size_average` and `reduce` aliases. This migration keeps those
arguments as explicit Python signature failures rather than silently ignoring
them or pretending the unweighted derivative/export contract covers them.
Weighted/broadcast semantics require a separately versioned extension.

The numerical policy follows PyTorch's current native implementation and
autograd formula definitions:

- <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp>
- <https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml>
- <https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/FunctionsManual.cpp>

## Architecture-Freeze Transition

The opaque baseline narrows from 11 constructor calls and 11 operations to 10
of each. The `binary_cross_entropy_with_logits` label,
`functional.binary-cross-entropy-with-logits` constructor site, and
`jit.custom.binary-cross-entropy-with-logits.v0` identity are retired from the
current inventory. The frozen original operation-ID list retains the identity,
preserving the exact partition of all 39 original opaque operations. The
executable typed registry advances from 28 to 29 retirement records.

Gate 0 continues to use still-opaque `kl_div` for representative callback
refusals. The accepted freeze changes only remove the retired constructor,
label, and identity, update affected decision digests, and bind the new
inventory and fixture hashes. No remaining caller, label, operation, or policy
is widened.

## Consequences

BCE with logits now has inspectable numerical, derivative, transform, export,
resource, and backend semantics instead of an opaque host callback. Large
finite logits avoid overflow, both operands participate in eager and
functional differentiation, Grad exposes all three reductions and the exact
PyTorch functional alias, and ONNX receives the same stable decomposition.
This migration does not claim optional weighting, target-domain enforcement,
tensor-plan/WebGPU execution, or probability-domain BCE semantics.
