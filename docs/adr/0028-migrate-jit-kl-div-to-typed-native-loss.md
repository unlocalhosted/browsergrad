# ADR-0028: Migrate JIT KL Divergence to Typed Native Loss

- **Status:** accepted
- **Date:** 2026-07-22
- **Owners:** JIT and Grad
- **Gate:** 6 — framework convergence

## Context

`torch.nn.functional.kl_div` was the last caller of the shared JIT
elementwise-loss `CUSTOM` helper. The callback converted operands to float32,
declared only the input dtype, differentiated only the input, and hid
`log_target`, reduction, transform, export, device, and resource behavior from
the typed compiler. Grad maintained a separate float32 implementation with the
same disconnected target and without the native `batchmean` reduction.

KL divergence has two distinct target representations. Probability targets use
the native `xlogy(target, target) - target * input` algebra; logarithmic targets
use `exp(target) * (target - input)`. Substituting a clipped logarithm changes
the zero-target value and derivative contract. Treating `batchmean` as ordinary
mean also changes the result because it divides a non-scalar input's total loss
by its first user dimension rather than by its element count.

## Decision

JIT emits binary `KL_DIV(input, target)` with canonical
`{reduction, batch_rank, log_target}` arguments. Internal
`KL_DIV_VJP(dy, input, target)` carries the same arguments plus the selected
operand. Construction and every CPU, VJP, vmap, ONNX, and tensor-plan boundary
revalidate the complete typed contract.

The first profile is closed as follows:

- Both operands are exact same-shape `float16`, `float32`, or `float64`
  tensors from the same session. Dimensioned tensor promotion selects the
  output dtype; a half-only result computes and accumulates in float32. Each
  operand's cotangent is returned in that operand's source dtype.
- `log_target=False` evaluates `xlogy(target, target) - target * input` and
  explicitly returns positive zero for the `xlogy(0, 0)` contribution. The
  input derivative is `-target`; the target derivative follows the current
  native composite `log(target) + 1 - input`, including NaN at an exact zero
  target. BrowserGrad does not replace that derivative with a clipped or
  hand-selected finite subgradient.
- `log_target=True` evaluates `exp(target) * (target - input)`. Its input
  derivative is `-exp(target)` and its target derivative is
  `exp(target) * (target - input + 1)`. Grad snapshots both derivatives during
  forward construction; JIT exposes them through closure and symbolic VJP.
- Reductions are exactly `none`, `sum`, `mean`, and `batchmean`. `mean` divides
  by the complete per-example element count. `batchmean` sums and divides by
  the first user dimension for non-scalar input; scalar `batchmean` equals
  scalar sum. A zero first dimension produces NaN, while a nonzero batch with
  empty support produces zero. Vmap-owned leading dimensions remain outside
  both denominators, including nested vmap.
- Rank is bounded at 32. Individual extents, output bytes, aggregate visits,
  and conservative peak workspace are bounded before numerical work or
  allocation. The 48-visit factor covers both forward target modes, both
  derivative directions, masks, and reductions. Zero extents cannot conceal a
  hostile capacity in another dimension.
- CPU realization returns an owning array. ONNX opset 17 emits the exact
  promoted probability-target or log-target decomposition, preserves the
  zero-target `xlogy` value convention with `Equal`/`Where`, and implements
  exact scalar, ordinary, and batchmean reductions. Float16 computation is
  promoted to float32 and cast back. Tensor-plan and WebGPU execution remain
  explicit refusals until canonical loss-reduction lowering and a portable
  kernel exist.

`log_target` must be an exact boolean. The previous JIT surface did not accept
the deprecated `size_average` or `reduce` aliases, so the typed API and Grad
conformance surface reject them instead of adding ambiguous reduction
precedence. Their compatibility behavior requires a separately versioned
extension.

The numerical and reduction policy follows PyTorch's current functional and
native implementations and its current `xlogy` derivative definition:

- <https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html>
- <https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py>
- <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp>
- <https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml>

## Architecture-Freeze Transition

The opaque baseline narrows from 10 constructor calls and 10 operations to 9
of each. The shared `_custom_elementwise_loss` helper is deleted because it has
no remaining callers. The `kl_div` label,
`functional.elementwise-loss-helper` constructor site, and
`jit.custom.kl-div.v0` identity are retired from the current inventory. The
frozen original operation-ID list retains the identity, preserving the exact
partition of all 39 original opaque operations. The executable typed registry
advances from 29 to 30 retirement records.

Gate 0 now uses still-opaque `nll_loss` for representative callback refusals.
The accepted freeze changes only remove the retired constructor, label, helper,
and identity, update the affected device-decision digest, and bind the new
inventory and fixture hashes. No remaining caller, label, operation, or policy
is widened.

## Consequences

KL divergence now has inspectable target representation, reduction, numerical,
derivative, transform, export, resource, and backend semantics instead of an
opaque host callback. JIT and Grad share one conformance fixture for both target
modes, both derivatives, mixed dtypes, scalar and empty cases, nested mapping,
hostile resources, and public aliases. This migration does not claim clipped
zero-target gradients, deprecated reduction aliases, tensor-plan/WebGPU
execution, or a broader broadcasting profile.
