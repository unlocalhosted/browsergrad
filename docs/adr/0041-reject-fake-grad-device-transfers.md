# ADR-0041: Reject fake Grad tensor device transfers

## Status

Accepted.

## Context

Eager Grad stores tensors in NumPy arrays inside Pyodide/Wasm. Its
`Tensor.to()` previously caught every invalid dtype string as though it were a
device request, while `to("cuda")` and `cuda()` returned the same CPU tensor.
That made malformed dtype requests and unavailable device transfers appear to
succeed.

Silent CPU substitution is incompatible with BrowserGrad's production
contract: a CPU reference or CPU-backed eager tensor is not CUDA execution.

## Decision

The tensor request boundary is closed and explicit:

- `to("cpu")`, `to(device="cpu")`, and `cpu()` return the same tensor when no
  dtype conversion is requested;
- CPU device and supported dtype requests may be composed, preserving the
  frozen dtype-conversion and autograd behavior;
- CUDA, MPS, XPU, Meta, indexed CPU, and other non-CPU storage requests fail
  before execution and state that no transfer occurred;
- `cuda()` fails with the same honest contract;
- invalid dtype strings, unsupported keywords, duplicate dtype/device
  requests, excessive positional arguments, and ambiguous signatures fail
  before allocation.

`Tensor.to(other)` adopts the other tensor's CPU device and dtype through the
same validation and conversion path.

## Consequences

Tensor code can no longer mistake CPU identity for GPU movement, and invalid
dtype requests cannot become silent no-ops. The Grad compatibility freeze now
includes `Tensor.cpu`, `Tensor.cuda`, and the complete `Tensor.to` parser.

The separate `torch.tensor(device=...)` factory and compatibility
`nn.Module.to(...)` shim still need the same CPU-only refusal contract. They
remain the next device-convergence slice and are not covered by this decision.
