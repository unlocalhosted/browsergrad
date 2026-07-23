# ADR-0039: Fix Grad detach aliasing

## Status

Accepted.

## Context

Grad's `Tensor.detach()` copied its NumPy storage and allowed the constructor
default to convert non-float32 inputs to float32. That behavior contradicted
the PyTorch-shaped contract: detaching should sever autograd history without
materializing or changing the underlying storage representation.

The old behavior was frozen as compatibility debt by
`grad.materialization.detach-copy.v0` and two executable fixture cases. Keeping
that baseline would make storage aliasing, dtype, and layout unreliable at a
core eager-runtime boundary.

## Decision

`Tensor.detach()` returns a distinct `Tensor` object that:

- shares the exact NumPy storage with the source;
- preserves dtype, shape, strides, and contiguity;
- has `requires_grad=False`, no `_ctx`, and leaf status;
- exposes mutations bidirectionally through the shared storage.

It performs no allocation or copy. This is an out-of-place metadata operation,
not an implementation of in-place `detach_()`.

The Grad compatibility inventory replaces
`grad.materialization.detach-copy.v0` with `grad.view.detach.v1`. Its executable
evidence covers float32 mutation aliasing, float16 dtype preservation, and
non-contiguous stride preservation.

## Consequences

The previous copying and float16-to-float32 fixture cases are retired. Any
future change to detach storage sharing, dtype/layout preservation, or
autograd severance must update this ADR-backed frozen baseline and its
executable fixtures.

Remaining Grad convergence work includes cross-dtype `Tensor.to`, invalid
dtype/device disambiguation, and NumPy interop behavior.
