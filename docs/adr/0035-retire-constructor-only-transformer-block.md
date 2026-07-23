# ADR-0035 — Retire the Constructor-Only Transformer Block Surface

- **Status:** Accepted
- **Date:** 2026-07-23
- **Owners:** `@unlocalhosted/browsergrad-jit`
- **Supersedes behavior:** `jit.custom.transformer-block.v0`

## Context

`bg.kernels.transformer_block(...)` constructed
`CUSTOM(op="transformer_block")`, but no CPU, portable tensor-plan, legacy
WebGPU, or other backend executed that operation. The constructor did not
validate the promised transformer-block shapes, dtypes, head geometry,
weights, or backward contract. Its result was disconnected from autograd and
failed only when a realizer was invoked.

PRD-012c remains a draft design for pattern recognition and cross-block
megakernel code generation. A callable constructor that produces an
unexecutable opaque node is not evidence that any part of that design ships.

## Decision

Remove `bg.kernels.transformer_block` and its `OP_CUSTOM` constructor. Do not
silently replace it with an ordinary sequence of primitive operations: that
would execute different work while retaining a namespace and name that promise
megakernel recognition and lowering.

Keep `jit.custom.transformer-block.v0` only in the frozen original-operation
partition as a removed unsupported surface. The original 39 opaque operation
identities now partition exactly into:

- 35 typed retirements;
- 2 current opaque operations: the legacy `flash_attention` accelerator route
  and the intentional `user` kernel extension; and
- 2 removed unsupported surfaces: `transformer_block` and `webnn_matmul`.

## Consequences

Callers now fail at attribute lookup instead of constructing a graph that
cannot execute. PRD-012c remains explicitly unimplemented.

A future transformer-block capability must either recognize a typed graph
pattern or introduce an exact typed semantic operation. It must define shape,
dtype, resource, CPU/reference, VJP, transform, export, backend, residency, and
materialization decisions and provide actual execution evidence before the
surface can return.
