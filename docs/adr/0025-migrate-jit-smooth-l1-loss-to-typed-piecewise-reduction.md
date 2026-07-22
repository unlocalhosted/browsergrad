# ADR-0025: Migrate JIT Smooth L1 Loss to Typed Piecewise Reduction

- **Status:** accepted
- **Date:** 2026-07-22
- **Owners:** JIT and Grad
- **Gate:** 6 — framework convergence

## Context

`torch.nn.functional.smooth_l1_loss` used the shared JIT
`_custom_elementwise_loss` callback. The callback exposed a declared first-input
dtype without validating the realized result, differentiated only the input,
and left functional differentiation, vmap, ONNX, tensor-plan, and WebGPU
behavior behind generic `CUSTOM` refusals. It also allowed NumPy to determine
shape, dtype, allocation, and work behavior after graph construction. Grad had
a separate float32-only derivative path, disconnected the target, rejected the
zero-beta L1 limit, and did not share the JIT resource contract.

Smooth L1 is piecewise rather than a spelling-specific special case. Its
semantic contract therefore belongs beside the typed loss-reduction seam
introduced by ADR-0024, with the branch point and both derivative directions
represented explicitly.

## Decision

JIT emits binary `SMOOTH_L1_LOSS(input, target)` with canonical
`{reduction, batch_rank, beta}` arguments. An internal
`SMOOTH_L1_LOSS_VJP(dy, input, target)` carries the same arguments plus the
selected operand. Both operations revalidate their complete contract at every
CPU, VJP, vmap, ONNX, and tensor-plan boundary.

The first profile is closed as follows:

- Both operands are same-shape `float16`, `float32`, or `float64` tensors from
  the same session. Dimensioned tensor promotion determines the output dtype;
  a half-only result computes and accumulates in float32.
- `beta` accepts only exact built-in or fixed-width NumPy real scalar types.
  It is finite, non-negative, normalized to positive zero, and rounded once to
  the promoted compute dtype. A positive value that underflows to zero or
  overflows to infinity in that dtype is rejected before UOp construction.
- For positive beta, the per-element result is
  `0.5 * difference**2 / beta` when `abs(difference) < beta`, and
  `abs(difference) - 0.5 * beta` otherwise. The exact boundary uses the linear
  branch. Zero beta is the exact L1 limit and never evaluates a division.
- Reductions are exactly `none`, `sum`, and `mean`. Empty sum is zero, empty
  mean is NaN, and vmap-owned leading dimensions remain outside the per-example
  reduction. CPU outputs are owning arrays.
- Closure and symbolic autograd propagate the piecewise derivative to both
  operands. The target derivative is its negative, and zero derivatives are
  normalized to positive zero. Grad snapshots the derivative at forward time
  so later input mutation cannot change backward semantics.
- Rank is bounded at 32; individual extents, output bytes, aggregate element
  visits, and conservative peak workspace are bounded at 256 MiB-equivalent
  ceilings before allocation. The 32-visit work factor covers the piecewise
  forward and both derivatives. Workspace counts input casts, the output, both
  retained source-dtype cotangents, three compute buffers, and a mask. Zero
  extents do not hide hostile capacity in other axes.
- ONNX opset 17 emits `Sub`, `Abs`, `Less`, `Mul`, `Div`, `Where`, and the exact
  reduction for positive beta. Zero beta emits the smaller L1 decomposition.
  Float16 operands compute through explicit float32 casts and cast back.
- Tensor-plan and direct WebGPU execution remain explicit refusals until a
  canonical piecewise loss-reduction lowering and kernel exist. Typed CPU
  realization is not reported as portable device execution.

JIT and Grad consume one cross-package conformance fixture. Their resource
validation shares the same constants and formulas within each package; JIT's
L1 and Smooth L1 contracts additionally share one internal geometry,
reduction, runtime-array, and upstream-cotangent seam. ADR-0024's L1 workspace
bound is tightened to count the retained upstream compute buffer explicitly.

## Architecture-Freeze Transition

The opaque baseline narrows from 13 operations to 12 while remaining at 11
constructor calls because two operations still share the one callback-helper
constructor. The `smooth_l1_loss` label and
`jit.custom.smooth-l1-loss.v0` are retired from the current inventory. The
frozen original operation-ID list retains the retired identity, preserving the
exact partition of all 39 original opaque operations. The executable typed
registry advances from 26 to 27 retirement records.

Gate 0 now uses still-opaque `binary_cross_entropy` for the representative
callback-refusal fixture. The accepted freeze changes only remove the retired
label and identity, reduce the helper token count, update the affected
decision-source digest, and bind the new inventory and fixture hashes. No
remaining caller, label, operation, or policy is widened.

## Consequences

Smooth L1 now has inspectable semantics and portable transform/export decisions
instead of an opaque host callback. Both operands participate in eager and
functional differentiation, zero beta is safe and exact, mixed floating dtypes
are explicit, and hostile scalar/array/resource inputs fail before numerical
hooks or allocation. The migration does not claim a tensor-plan or WebGPU
kernel, broadcasting, integer losses, deprecated reduction aliases, or a
general Huber-loss frontend.
