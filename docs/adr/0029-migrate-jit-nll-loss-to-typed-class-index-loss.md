# ADR-0029: Migrate JIT NLL Loss to Typed Class-Index Loss

- **Status:** accepted
- **Date:** 2026-07-23
- **Owners:** JIT and Grad
- **Gate:** 6 — framework convergence

## Context

`torch.nn.functional.nll_loss` constructed a JIT `CUSTOM` callback that
accepted only a two-dimensional input and one-dimensional target, converted
the input to float32, implemented only mean reduction, and differentiated only
through a captured NumPy closure. Class weights, ignored targets, unbatched and
spatial inputs, reduction aliases, symbolic differentiation, vmap, export,
resource limits, and backend decisions were absent from the compiler-visible
contract. Grad carried a separate implementation with the same narrow
semantics.

Negative-log-likelihood is a class-axis indexed reduction, not an ordinary
elementwise loss. For input `(N, C, d1, ..., dK)`, target coordinates select
one value from class axis `C`; ignored coordinates contribute neither a value
nor denominator weight. Weighted mean divides by the sum of selected class
weights rather than the number of target elements. An unbatched `(C)` input
uses a scalar target, while transform-owned leading batch axes must remain
outside each mapped example's reduction.

## Decision

JIT emits variadic `NLL_LOSS(input, target[, weight])` with canonical
`{reduction, batch_rank, ignore_index, has_weight}` arguments. Internal
`NLL_LOSS_VJP(dy, input, target[, weight])` carries the same arguments and
returns only the input cotangent. Construction and every CPU, VJP, vmap, ONNX,
and tensor-plan boundary revalidate the complete typed contract.

The first profile is closed as follows:

- Input is exact `float16`, `float32`, or `float64`. Target is exact `int64`.
  Optional weight has the input dtype, contains exactly one value per class,
  and is non-differentiable. All JIT operands belong to one session.
- User input is either unbatched `(C)` with scalar target, or
  `(N, C, d1, ..., dK)` with target `(N, d1, ..., dK)`. Vmap-owned leading
  axes shift the class axis and permit either captured or mapped class
  weights. Reductions remain per mapped example, including nested transforms.
- Every non-ignored target is range checked before indexed access. Ignored
  targets use a safe zero index only behind an explicit validity mask, so
  negative NumPy wrapping and out-of-range memory access cannot substitute for
  the public contract.
- `none` returns the weighted selected losses in target shape. `sum` adds all
  non-ignored selected losses. `mean` divides by the selected weight sum when
  weight is present and by the non-ignored count otherwise. Empty or
  all-ignored mean returns NaN; its input gradient is zero.
- Closure and symbolic autograd write `-upstream * selected_weight` at each
  valid selected class coordinate, apply the mean denominator where required,
  and return the cotangent in the input dtype. Target and weight are explicitly
  non-differentiable. Grad snapshots target validity, safe indices, and
  selected weights at forward construction so later mutation cannot change
  backward meaning.
- Rank is bounded at 32. Individual extents, output bytes, aggregate visits,
  and conservative peak workspace are bounded before numerical work or
  allocation. Capacity uses `max(1, extent)` so a zero dimension cannot conceal
  hostile work in another dimension. Float16 compute and accumulation use
  float32 before an explicit cast back.
- CPU realization returns an owning array. ONNX opset 17 emits
  `NegativeLogLikelihoodLoss` for the unmapped profile with exact reduction,
  ignore-index, and optional weight inputs. The unbatched rank-one form is
  normalized with `Unsqueeze` and, for `none`, `Squeeze`. Export after vmap is
  refused because ONNX fixes the class axis. Tensor-plan and WebGPU execution
  remain explicit refusals until canonical indexed-loss lowering exists.

The full functional and module signatures admit `weight`, `ignore_index`,
`size_average`, `reduce`, and `reduction`. Deprecated reduction flags follow
the current PyTorch precedence: `reduce=False` selects `none`; otherwise
`size_average=False` selects `sum`, and the default is `mean`. Exact boolean
validation prevents truthy objects from silently changing the contract.

The shape, reduction, weighting, ignore-index, and exporter policy follows the
current public specifications:

- <https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html>
- <https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py>
- <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossNLL.cpp>
- <https://onnx.ai/onnx/operators/onnx__NegativeLogLikelihoodLoss.html>

## Architecture-Freeze Transition

The opaque baseline narrows from 9 constructor calls and 9 operations to 8 of
each. The `nll_loss` label, `functional.nll-loss` constructor site, and
`jit.custom.nll-loss.v0` identity are retired from the current inventory. The
frozen original operation-ID list retains the identity, preserving the exact
partition of all 39 original opaque operations. The executable typed registry
advances from 30 to 31 retirement records.

Gate 0 now uses still-opaque `cross_entropy` for representative callback
refusals. The accepted freeze changes only remove the retired constructor,
label, and identity, update the exact callback fixture, and bind the new
inventory and test hashes. No remaining caller, label, operation, or policy is
widened.

## Consequences

NLL loss now has inspectable class-axis, index-range, weighting, ignored-target,
reduction, derivative, transform, export, resource, residency, and backend
semantics instead of an opaque host callback. JIT and Grad consume one
conformance fixture spanning unbatched and spatial shapes, all reductions,
weighted denominators, ignored and empty targets, mixed floating dtypes,
legacy aliases, mutation snapshots, malformed runtime state, and hostile
resource shapes.

This migration does not claim differentiation of target or weight, ONNX export
after vmap, tensor-plan/WebGPU execution, probabilistic input validation, or
cross-entropy/log-softmax fusion.
