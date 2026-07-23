# ADR-0046: Converge Grad expand as a zero-stride view

## Status

Accepted.

## Context

ADR-0002 moved lazy JIT `Tensor.expand` to typed `BROADCAST_TO`, but eager
Grad retained an owning NumPy copy. Values, dtype, validation, and VJP agreed,
yet the eager surface still contradicted the PyTorch-shaped view contract:
source/result mutations were isolated, expanded axes were contiguous rather
than zero-stride, and large logical broadcasts allocated full output storage.

This was the final behavior marked `compatibility-debt` in the Grad inventory.

## Decision

After the existing exact rank, integer, `-1`, non-negative, and broadcast
validation, eager `Tensor.expand` returns a NumPy-backed view:

- existing source strides are preserved on non-expanded axes;
- every expanded singleton axis receives stride zero;
- the result shares storage and source writeability;
- mutations are visible through source and result aliases;
- physical dtype and non-contiguous source layout are preserved;
- no output-sized allocation occurs;
- the existing unbroadcast VJP reduces every expanded axis and reshapes to the
  original input shape.

The implementation uses `numpy.lib.stride_tricks.as_strided` only after proving
that each target axis either retains its source extent or broadcasts a
singleton. Zero strides therefore refer only to an existing source element and
cannot introduce an out-of-bounds address.

## Consequences

Grad and the PyTorch-shaped eager contract now agree on expand aliasing,
zero-stride layout, validation, dtype, and autograd. The lazy JIT CPU realizer
may still materialize its backend result; backend realization ownership is a
separate contract from the eager public view.

No Grad behavior remains classified as `compatibility-debt`. Remaining Gate 6
work is generated runtime/profile consumption of executable requirement,
capability, and lowering records.
