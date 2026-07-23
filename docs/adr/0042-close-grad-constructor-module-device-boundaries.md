# ADR-0042: Close Grad constructor and module device boundaries

## Status

Accepted.

## Context

ADR-0041 made eager tensor placement CPU-only and fail-closed, but two
torch-compatibility seams still silently ignored placement:

- `torch.tensor(..., device=...)` accepted any device while allocating a
  CPU/Pyodide-backed tensor;
- `nn.Module.to(...)` returned the same module for every device and dtype
  request without moving or converting parameters.

Those paths could still make CUDA placement or module dtype conversion appear
successful.

## Decision

`torch.tensor` accepts no device or the exact string `"cpu"`. It then follows
the frozen torch constructor dtype/default contract. Non-string or non-CPU
device requests fail before tensor allocation.

The compatibility `nn.Module.to` shim:

- returns the same module for no request or the exact CPU device;
- rejects CUDA, MPS, XPU, Meta, indexed CPU, and other non-CPU device
  requests before parameter access or execution;
- rejects every non-null dtype conversion request because recursive module
  parameter conversion is not implemented;
- rejects unsupported keywords, duplicate requests, excessive positional
  arguments, and non-string device values.

The Grad architecture freeze now includes
`_torch_compat_limited.py:install_limited._module_to_shim` and the changed
`_torch_compat_real.py:install_real._tensor_factory`, so these compatibility
boundaries cannot drift outside the executable inventory.

## Consequences

All currently exposed eager tensor construction, tensor movement, and module
movement entrypoints are CPU-only and fail-closed. No path may report or imply
CUDA storage or module dtype conversion without implementing the underlying
parameter/storage semantics.

NumPy interop remains the next Grad compatibility-convergence slice.
