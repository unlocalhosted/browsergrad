# ADR-0036 — Migrate JIT Attention Forward to Typed IR

- **Status:** Accepted
- **Date:** 2026-07-23
- **Owners:** `@unlocalhosted/browsergrad-jit`
- **Retires:** `jit.custom.flash-attention.v0`

## Context

`bg.kernels.flash_attention` constructed `CUSTOM(op="flash_attention")`.
That opaque node bypassed CPU semantics and all framework-operation
validation, silently disconnected autograd, admitted malformed shape/dtype
relationships, and routed only through the legacy row-wise online-softmax
WebGPU bridge. Its name incorrectly suggested the independently proved Gate 5
block-tiled algorithm.

BrowserGrad already owns a frontend-neutral Gate 5 attention-forward semantic
contract and separately proves a block-tiled WebGPU realization. The JIT
compatibility route must not reuse that proof for a different algorithm.

## Decision

Introduce typed `ATTENTION_FORWARD` IR and expose it as
`bg.kernels.attention_forward`. Keep `bg.kernels.flash_attention` only as a
compatibility alias that constructs the same typed node.

The initial JIT profile is deliberately exact and narrow:

- dense positive rank-4 `(B,H,S,D)` query/key/value tensors;
- exact `float32`, matching batch/head dimensions, matching key/value sequence
  length, and identical head depth;
- head depth at most 64, bounded output/work/workspace projections, canonical
  float32 inverse-square-root scale, and no mask;
- finite runtime Q/K/V, scaled scores, softmax state, and output; and
- owning stable NumPy CPU output.

The old additive-mask/custom-scale alias profile is rejected with a pointer to
`torch.nn.functional.scaled_dot_product_attention`, whose primitive typed graph
owns those semantics. Inputs requiring gradients fail at construction because
the shared Gate 5 v1 contract has no VJP. Vmap, ONNX, and the portable tensor
plan validate and then refuse until their own attention contracts exist.

The legacy direct WebGPU realizer may execute the typed node through the
accurately named row-wise online-softmax compatibility kernel and materialize
the root to host. This is not block-tiled execution, Gate 5 semantic-side-table
integration, FlashAttention-v2, or a portable tensor-plan claim.

## Consequences

The original 39 opaque identities now partition exactly into:

- 36 typed retirements;
- 1 current intentional opaque identity, the user-authored WGSL extension; and
- 2 removed unsupported surfaces, transformer block and WebNN matmul.

`_custom_kernel.py` is the only remaining `OP_CUSTOM` constructor caller.
Framework support reporting is generated from the executable attention
validator and records every supported or refused boundary explicitly.
