# ADR-0011: Migrate JIT Gather to Typed Index Semantics

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, Grad

## Context

`TensorProxy.gather` was a frozen `CUSTOM` NumPy callback. It coerced arbitrary
axis values through `int`, wrapped out-of-range axes with modulo, converted
every index dtype to int64, delegated negative-index behavior to NumPy, and hid
indexing meaning from symbolic autograd, batching, export, and backend policy.
Grad independently accepted arbitrary array coercion and converted every
result and gradient to float32. The existing `INDEX` and `SCATTER_ADD` IR names
had incomplete, mutually inconsistent NumPy handlers and no construction path.

## Decision

JIT `Tensor.gather` now emits `INDEX` with two inputs and one normalized static
axis. The executable contract requires a non-scalar source, an int64 index of
the same rank, an output shaped exactly like the index, source-dtype
preservation, and index extents no larger than source extents on every
non-gather dimension. JIT and Grad accept only an actual tensor index and an
exact built-in or fixed-width NumPy integer axis. They reject arbitrary
conversion hooks, bool/floating axes, non-int64 indices, rank mismatch, and
invalid non-gather extents before execution.

Index values are data rather than graph metadata. CPU realization and both
closure and symbolic backward therefore check every nonempty index tensor
against the closed interval `[0, source_extent)` before indexing. Negative
indices are rejected rather than inheriting NumPy's wrapping extension. Empty
indices remain valid. CPU realization uses exact coordinate tensors so the
PyTorch rule allowing smaller non-gather dimensions is preserved and always
returns an owning source-dtype array.

Closure backward and the symbolic VJP deterministically scatter-add upstream
values into a zero source-shaped tensor. Duplicate indices accumulate. The
`SCATTER_ADD` handler now consumes the same exact coordinate semantics and has
a separate closed structural validator. The index input is discrete and has no
gradient. Grad uses the same rule and preserves source/output/gradient dtype;
the shared eager backward accumulator now stores gradients in each parent's
declared dtype. The compatibility freeze also advances Grad indexing, reshape,
transpose, and permute to dtype-preserving view semantics. This is required for
a valid int64 token/index tensor to remain int64 after the ordinary
slice-and-unsqueeze path that feeds gather; compatible NumPy views continue to
alias while fancy or incompatible reshape paths retain their existing copy
policy.

Vmap admits only the exact paired profile in which source and index both share
the leading mapped axis; it shifts the gather axis past that batch dimension.
ONNX opset 17 emits `GatherElements` for float32, int32, int64, and bool source
graphs with int64 indices. Tensor-plan and WebGPU execution refuse until a
deterministic bounds-checked index/scatter lowering exists. No structural plan
admission or legacy bridge registration is reported as device capability.

The opaque baseline narrows from 26 constructor calls and 29 operations to 25
constructor calls and 28 operations. `tensor.gather`, the `gather` label, and
`jit.custom.gather.v0` are retired from the current inventory. The executable
registry and frozen original-ID list retain the exact partition of all 39
original opaque operation identities.

## Compatibility and removal

Valid positive/negative-axis gathers, repeated indices, smaller non-gather
dimensions, empty indices, and float/integer/bool sources remain available.
Outputs and gradients now preserve dtype. Float/bool indices, negative or
out-of-range index values, implicit list/array indices, scalar sources, and
coercive axes now fail early or at the first value-observing boundary. This is
an intentional move from NumPy convenience behavior to the documented tensor
gather contract. Grad view results no longer silently cast float16, bool, or
integer storage to float32; that compatibility debt is intentionally retired
under this ADR.

The remaining `CUSTOM` compatibility baseline still targets removal in
`@unlocalhosted/browsergrad-jit@1.0.0`. This decision does not widen any
remaining opaque caller, label, operation, or policy.
