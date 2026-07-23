# ADR-0045: Converge Grad constructor semantics

## Status

Accepted.

## Context

The Grad inventory combined two different constructor surfaces into one
compatibility-debt record:

- direct `Tensor(data)` is the small educational library's legacy constructor;
- `torch.tensor(data)` is a compatibility surface that promises PyTorch-shaped
  inference and ownership.

The combined implementation coerced every non-integer `torch.tensor` input to
float32. It therefore turned booleans into floats, discarded supported NumPy
dtypes, changed existing Tensor dtypes, and sometimes aliased caller-owned
arrays despite `torch.tensor` being a copying constructor. It also allowed
integer tensors to request gradients.

## Decision

The direct BrowserGrad `Tensor(data, dtype=None)` contract remains explicit and
separate: it defaults to float32 and may alias an existing float32 ndarray.
Callers that require another admitted storage dtype pass it explicitly or use
`from_numpy`.

`torch.tensor` is an owning leaf constructor:

- Python boolean data infers bool, integer data infers int64, and floating or
  mixed integer/floating data infers float32;
- supported NumPy arrays/scalars and existing Grad Tensor inputs preserve their
  storage dtype unless an explicit supported dtype is requested;
- every result owns a copy, preserves NumPy layout order where representable,
  and has no inherited autograd history;
- `requires_grad` must be boolean and may be true only for
  float16/float32/float64 storage;
- complex, object, string, structured, datetime, and other unsupported input
  kinds reject before tensor allocation;
- device handling remains the CPU-only fail-closed contract from ADR-0042.

## Consequences

Constructor defaults, dtype inference, ownership, and autograd admission are
now executable, separately classified contracts. The torch compatibility
surface no longer aliases its input or silently changes supported NumPy/Tensor
dtypes.

This closes the constructor compatibility-debt record. ADR-0046 subsequently
closes the final owning-`Tensor.expand()` materialization record.
