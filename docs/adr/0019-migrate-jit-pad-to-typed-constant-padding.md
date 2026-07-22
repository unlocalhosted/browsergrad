# ADR-0019: Migrate JIT Pad to Typed Constant Padding

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`torch.nn.functional.pad` was implemented as a frozen `CUSTOM` NumPy callback.
The callback hid padding geometry, fill-value conversion, gradient extraction,
and export meaning from symbolic autograd, vmap, ONNX, and backend selection.
It accepted coercive padding components and fill values, delegated resource
limits to NumPy allocation, and allowed declared tensor dtype to disagree with
the callback result. Grad maintained an independent eager path with different
validation, conversion, ownership, and differentiation behavior.

## Decision

JIT now owns `PAD` as typed constant padding over trailing dimensions. The
public boundary accepts an exact tensor input and a plain tuple or list with an
even number of padding components. The number of padding pairs cannot
exceed the source rank. Components must be built-in or fixed-width NumPy
integers, must be nonnegative, and cannot invoke arbitrary conversion hooks.
The PyTorch last-dimension-first sequence is normalized into one canonical
first-dimension-first `pad_width` tuple. This first profile supports
`mode="constant"` only; reflection, replication, circular padding, and
negative cropping remain explicit refusals.

The closed dtype domain is bool, uint8, signed int8/16/32/64, and
float16/32/64. `value=None` becomes the dtype's exact zero. Bool padding
requires a bool fill; integral padding requires an exact in-range integral
fill; floating padding requires an exact real scalar whose conversion remains
finite in the destination dtype. Rank is capped at 32, each output extent at
268,435,456 elements, and output storage at 256 MiB before allocation. The
per-axis ceiling also closes zero-sized tensor requests whose other dimensions
could otherwise carry unbounded host integers past the byte-product check.
Construction and the CPU, VJP, vmap, ONNX, and
tensor-plan boundaries all rederive padding geometry, dtype, exact fill, and
resource facts from the input node. CPU realization returns an owning array in
the exact declared dtype.

Closure and symbolic autograd extract the unpadded static interior. Symbolic
VJP emits typed `SLICE`, so no opaque callback enters the derivative graph.
Gradient edges exist only for floating sources. Vmap prepends one zero-padding
pair for its leading batch axis and otherwise preserves the normalized
trailing-dimension request.

ONNX opset 17 emits `Pad` with one exact rank-sized int64 pads initializer and
one scalar constant-value initializer for float32, int32, and int64. Other
exporter dtypes fail explicitly. Tensor-plan and WebGPU execution refuse until
there is a canonical padding/layout lowering and corresponding kernel; host
NumPy padding is not reported as device execution.

Grad consumes the same cross-package conformance fixture and applies the same
shape, dtype, fill, allocation, ownership, and gradient rules. The fixture
covers asymmetric multi-axis padding, exact dtype preservation, scalar and
empty dimensions, gradients, leading-axis vmap, exact ONNX protobuf structure,
hostile values, malformed requests, post-construction mutation, resource
bounds, and portable-backend refusal.

The opaque baseline narrows from 18 constructor calls and 21 operations to 17
constructor calls and 20 operations. `functional.pad`, the `pad` callback
label, and `jit.custom.pad.v0` are retired from the current inventory. The
frozen original operation-ID list retains the retired ID, preserving the exact
partition of all 39 original opaque identities. The Gate 0 representative
callback test now exercises still-opaque `l1_loss`.

## Compatibility and removal

Valid nonnegative constant padding remains available with stricter early
validation, exact dtype preservation, owning results, and connected floating
autograd. Coercive components, unsupported dtypes and modes, negative padding,
non-finite or out-of-range fills, oversized requests, malformed or mutated IR,
and unavailable portable backends fail at their semantic boundary.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
