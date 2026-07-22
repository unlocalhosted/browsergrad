# ADR-0023: Migrate JIT Einsum to a Typed Contraction

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

Top-level `torch.einsum` lowered through a frozen `CUSTOM` callback. JIT
executed `np.einsum` once during graph construction solely to discover output
shape, then executed it again during realization. Construction could therefore
allocate and compute in proportion to an unchecked output or contraction
domain. The callback result dtype was not revalidated, closure autograd was
absent, functional gradients silently returned zero, and vmap, ONNX,
tensor-plan, and WebGPU decisions were implicit refusals. Eager Grad separately
required an explicit arrow, supported no more than two operands, cast every
result and derivative to float32, and built backward equations that could not
represent diagonals, broadcasting, ellipses, or labels absent from the other
derivative inputs.

PyTorch string equations admit ASCII upper- and lower-case labels, optional
explicit output, repeated-label diagonals, broadcast-compatible shared labels,
and ellipses of different rank. Unlike NumPy, PyTorch also permits ellipsis
dimensions to be reduced when an explicit output omits the ellipsis. The
public BrowserGrad surface is the established string-equation overload; the
integer-sublist overload is a separate future API capability.

## Decision

JIT now emits variadic typed `EINSUM` with the exact canonical argument
`{equation, batch_rank}`. Parsing is allocation-free and never invokes user
conversion hooks. Spaces are removed, implicit output is resolved into an
explicit term using PyTorch's uppercase-before-lowercase label order, and base
ellipses are expanded into resolved internal labels. Repeated labels within one
operand require equal extents and represent a diagonal. Shared labels and
right-aligned ellipsis dimensions follow broadcast rules. An omitted explicit
output ellipsis reduces those dimensions rather than silently changing to
NumPy's string behavior.

The closed input domain contains 1 through 64 operands; equations at most
4,096 UTF-8 bytes; rank at most 32; at most 52 resolved NumPy labels; bool,
uint8, signed int8/16/32/64, and float16/32/64; and nonnegative extents no
larger than 256 Mi. Output storage and conservative
output/cast/contraction/largest-gradient
workspace are each capped at 256 MiB. The product of the complete resolved
label domain and operand count is capped at `2^28` elements. These bounds are
validated from metadata before NumPy execution or allocation, including when a
zero extent would otherwise hide a hostile sibling extent.

Dtype selection uses BrowserGrad's dimensioned-tensor promotion lattice.
Float16 contractions cast operands to float32 for contraction and accumulation,
then store an owning float16 result. Other admitted dtypes compute in the
promoted output dtype. CPU realization performs one `np.einsum` using numeric
subscripts and `optimize="greedy"`; graph construction performs none.

Closure and symbolic autograd share the same general derivative evaluator.
Symbolic VJP emits internal `EINSUM_VJP(dy, *operands)` with the canonical
equation and exact target operand. It contracts the output cotangent with every
other operand, broadcasts target labels absent from those inputs, reduces
operand broadcast axes, and scatters repeated-label derivatives back onto the
diagonal. Only floating operands receive cotangents. Grad snapshots all input
arrays before forward execution so later caller mutation cannot alter closure
derivatives.

Vmap records a leading `batch_rank` outside the user equation and broadcasts
captured inputs to that prefix. The CPU evaluator uses NumPy's list-subscript
ellipsis only for this transform-owned batch prefix; user ellipses have already
been resolved. This keeps mapped dimensions even when the source equation
reduces its own ellipsis and composes under nested vmap.

ONNX opset 17 emits `Einsum` after resolving every mapped and base label to an
explicit lower-case equation. Mixed inputs are cast to the promoted output
dtype. Export refuses bool and more than 26 resolved ONNX labels rather than
truncating or aliasing labels. Tensor-plan and direct WebGPU paths validate and
refuse until a canonical contraction schedule and lowering exist; typed host
execution is not reported as portable device support.

Grad consumes the same cross-package conformance fixture and now implements
implicit/explicit equations, arbitrary admitted arity, diagonals, broadcasting,
different-rank ellipses, PyTorch ellipsis reduction, scalars, uppercase labels,
promotion, float16 accumulation, owning output, immutable backward snapshots,
and exact hostile/resource refusals.

The opaque baseline narrows from 12 constructor calls and 15 operations to 11
constructor calls and 14 operations. `tensor.einsum` and
`jit.custom.einsum.v0` are retired from the current inventory. The frozen
original operation-ID list retains the retired identity, preserving the exact
partition of all 39 original opaque operations. No forward-only NumPy callback
remains in the public tensor surface.

## Compatibility and removal

Valid string-equation calls retain NumPy-backed results while gaining bounded
construction, canonical metadata, exact dtype behavior, connected closure and
functional gradients, transforms, and export. Previously accepted equations
that depended on unchecked allocation, malformed/coercive arguments,
float32-only output drift, or mathematically incomplete backward reconstruction
now receive deterministic errors or corrected results.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
