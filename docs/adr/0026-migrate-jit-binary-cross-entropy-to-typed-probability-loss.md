# ADR-0026: Migrate JIT Binary Cross Entropy to Typed Probability Loss

- **Status:** accepted
- **Date:** 2026-07-22
- **Owners:** JIT and Grad
- **Gate:** 6 — framework convergence

## Context

`torch.nn.functional.binary_cross_entropy` used the shared JIT
`_custom_elementwise_loss` callback. That callback clipped probabilities to
`[1e-12, 1 - 1e-12]`, so endpoint losses were about 27.63 instead of the
PyTorch-compatible value 100, accepted non-finite and out-of-range inputs,
differentiated only the probability input, and left functional
differentiation, vmap, export, tensor-plan, and WebGPU behavior behind generic
`CUSTOM` handling. Grad duplicated the same clipping drift and disconnected
the target.

The upstream contract is not ordinary logarithmic algebra. PyTorch validates
both probability domains, clamps each forward log result to a minimum of -100,
uses an independently epsilon-bounded probability derivative, and
differentiates the target through the unclamped logit. These choices must be
represented explicitly rather than recovered from a source label or replaced
with a superficially stable formula.

## Decision

JIT emits binary `BINARY_CROSS_ENTROPY(input, target)` with canonical
`{reduction, batch_rank}` arguments. Internal
`BINARY_CROSS_ENTROPY_VJP(dy, input, target)` carries those arguments plus the
selected operand. Both operations revalidate their complete structural
contract at every CPU, VJP, vmap, ONNX, and tensor-plan boundary.

The first profile is closed as follows:

- Both operands are same-shape `float16`, `float32`, or `float64` tensors from
  the same session. Dimensioned tensor promotion determines the output dtype;
  a half-only result computes and accumulates in float32. Mixed floating dtype
  support is an explicit BrowserGrad extension of the existing surface.
- Exact runtime arrays for both operands must contain only finite values in the
  closed interval `[0, 1]`. Graph construction validates metadata; CPU forward
  and VJP realization validate values before logarithms or result allocation.
- Forward computes `-(t * log(p) + (1 - t) * log(1 - p))` after clamping each
  log result, not `p`, to a minimum of `-100`. Exact `p=0,t=1` and `p=1,t=0`
  therefore produce 100. Exact zero results are normalized to positive zero.
- The probability derivative is `(p - t) / max((1 - p) * p, 1e-12)`, with the
  epsilon represented in the promoted compute dtype. The target derivative is
  `log(1 - p) - log(p)` without the forward log clamp, so valid endpoints
  produce the corresponding signed infinity. Closure and symbolic autograd
  preserve both derivatives and return cotangents in each source dtype. Grad
  snapshots both derivatives at forward time.
- Reductions are exactly `none`, `sum`, and `mean`. Empty sum is zero, empty
  mean is NaN, and vmap-owned leading dimensions remain outside each
  per-example reduction. CPU outputs are owning arrays.
- Rank is bounded at 32; individual extents, output bytes, aggregate visits,
  and conservative peak workspace are bounded before allocation. The 48-visit
  factor covers domain scans, forward logarithms, both derivative directions,
  and reductions. Workspace counts input casts, output, both retained
  source-dtype cotangents, four compute buffers, and one mask. Zero extents do
  not hide hostile capacity in another axis.
- ONNX export explicitly refuses this first profile. Opset 17 can decompose the
  arithmetic but cannot preserve the required fail-closed runtime rejection of
  non-finite or out-of-range probabilities. Clipping invalid values or
  returning NaN would be a semantic substitution.
- Tensor-plan and direct WebGPU execution remain explicit refusals until a
  canonical probability-domain loss-reduction lowering and kernel exist.
  Typed CPU realization is not reported as portable device execution.

The existing surface still does not accept an optional element weight or the
deprecated `size_average` and `reduce` aliases. This migration does not widen
that surface silently; those features require separately versioned contracts.

The numerical policy follows PyTorch's current native BCE implementation and
autograd formula definitions:

- <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp>
- <https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml>
- <https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/FunctionsManual.cpp>

## Architecture-Freeze Transition

The opaque baseline narrows from 12 operations to 11 while remaining at 11
constructor calls because KL divergence still owns the shared callback-helper
constructor. The `binary_cross_entropy` label and
`jit.custom.binary-cross-entropy.v0` are retired from the current inventory.
The frozen original operation-ID list retains the identity, preserving the
exact partition of all 39 original opaque operations. The executable typed
registry advances from 27 to 28 retirement records.

Gate 0 now uses still-opaque `kl_div` for the representative callback-refusal
fixture. The accepted freeze changes only remove the retired label and
identity, reduce the helper token count, update the affected plan digest, and
bind the new inventory and fixture hashes. No remaining caller, label,
operation, or policy is widened.

## Consequences

BCE now has inspectable probability-domain, numerical, derivative, transform,
resource, and backend semantics instead of an opaque host callback. Endpoint
behavior matches the declared upstream profile, both operands participate in
eager and functional differentiation, and invalid runtime values fail before
numerical work. The honest ONNX refusal avoids exporting a graph that cannot
enforce the source contract. This migration does not claim weighting,
broadcasting, integer losses, a tensor-plan/WebGPU kernel, or the distinct
logits-domain BCE operation.
