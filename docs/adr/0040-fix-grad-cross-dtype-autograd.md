# ADR-0040: Fix Grad cross-dtype autograd

## Status

Accepted.

## Context

Grad's effective `Tensor.to()` implementation already returned self for a
same-dtype request and materialized independent NumPy storage for a
cross-dtype request. Every cross-dtype result nevertheless lost
`requires_grad` and its source graph, including float16/float32/float64 casts
that should be differentiable.

The frozen compatibility behavior
`grad.conversion.to-cross-dtype-detaches.v0` made that disconnection visible,
but retaining it blocked dtype convergence and mixed-precision teaching
workloads.

## Decision

Cross-dtype casts among float16, float32, and float64:

- materialize independent target-dtype storage using NumPy's order-preserving
  conversion;
- record the source tensor as the sole autograd parent when gradients are
  enabled and required;
- cast the incoming VJP back to the source storage dtype;
- remain detached inside `no_grad()`.

Casts with a bool or integer source or target remain nondifferentiable and
produce detached target-dtype storage. Same-dtype requests continue to return
the original tensor.

The compatibility inventory replaces the old detached cross-dtype behavior
with separate `grad.conversion.to-cross-floating.v1` and
`grad.conversion.to-cross-nonfloating.v1` records. Executable evidence covers
float32-to-float64 VJP, non-contiguous float16-to-float64 layout and VJP,
`no_grad()`, and float-to-int detachment.

## Consequences

Float-to-float `Tensor.to()` no longer silently severs training graphs.
Integer and bool casts remain explicit nondifferentiable boundaries.

Invalid dtype/device disambiguation remains a separate compatibility debt:
unrecognized positional strings are still treated as device no-ops. It must
be retired without conflating device placement with dtype conversion.
