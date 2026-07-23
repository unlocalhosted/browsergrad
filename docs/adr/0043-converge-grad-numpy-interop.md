# ADR-0043: Converge Grad NumPy interop

## Status

Accepted.

## Context

Grad exposed three incompatible NumPy ownership rules:

- `from_numpy` aliased float32 arrays but silently converted every other dtype
  to independent float32 storage;
- `Tensor.numpy()` returned an owning copy;
- the NumPy array protocol exposed the live mutable tensor storage.

The same data could therefore alias, copy, change dtype, or bypass the export
snapshot boundary depending on surface spelling.

## Decision

NumPy input is an explicit zero-copy wrapping boundary. `from_numpy` accepts
writable `numpy.ndarray` storage with the closed eager dtype set: bool,
float16/32/64, int8/16/32/64, and uint8. ADR-0044 subsequently extends the
package-owned eager storage registry and this zero-copy boundary to uint16,
uint32, and uint64. It preserves exact ndarray identity, dtype, shape, strides,
contiguity, and bidirectional mutation. Non-arrays, read-only arrays, and
unsupported dtypes reject before wrapping.

NumPy output is an explicit owning-snapshot boundary. `Tensor.numpy()` and
`Tensor.__array__` share one `_numpy_snapshot` implementation:

- source dtype is preserved unless the array protocol requests another dtype;
- NumPy `order="K"` preserves C/F layout order where representable;
- mutations never propagate in either direction;
- `copy=False` rejects rather than exposing mutable tensor storage;
- malformed copy flags reject.

This output policy is intentionally BrowserGrad-defined rather than a claim of
PyTorch `Tensor.numpy()` alias compatibility. It permits safe inspection of
gradient-bearing educational tensors without creating a hidden mutation path.

## Consequences

All supported NumPy input dtypes now retain zero-copy storage and
non-contiguous layouts. Every public NumPy output spelling produces the same
owning snapshot policy.

This closes the NumPy interop sub-slice. ADR-0044 subsequently closes the
NumPy-delegated dtype fallback. Constructor-default classification and owning
`Tensor.expand()` materialization remain, followed by generated
runtime/profile support consumption.
