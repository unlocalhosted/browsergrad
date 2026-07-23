# ADR-0034: Retire the constructor-only WebNN matmul surface

- Status: Accepted
- Date: 2026-07-23
- Scope: `@unlocalhosted/browsergrad-jit`

## Context

`bg.experimental.webnn.matmul` constructed
`CUSTOM(op="webnn_matmul")`, but no CPU, WebGPU, or WebNN realizer could
execute the node. The constructor did not validate dtype agreement, did not
provide closure or symbolic gradients, disconnected autograd, and failed only
after graph construction. `navigator.ml` presence was the only implemented
WebNN fact.

This was not a WebNN backend or a useful compatibility surface. Delegating the
call to ordinary `MATMUL` would also be incorrect because a namespace-specific
WebNN request must not silently execute through NumPy or WebGPU.

## Decision

Remove `bg.experimental.webnn.matmul` and its `OP_CUSTOM` constructor. Retain
only `bg.experimental.webnn.is_available`, whose narrow contract is presence
detection and not operation support.

The future PRD-011 backend remains a graph-level execution tier. It must
consume existing typed IR through explicit partitioning, lowering, capability
records, device/context negotiation, fallback policy, resource limits, and
real-browser execution evidence. It must not reintroduce a source-facing
backend-named opaque operation.

Record `jit.custom.webnn-matmul.v0` as an unsupported surface removed from the
original opaque-ID partition. The architecture gate keeps three disjoint
classes: current opaque identities, typed retirements, and removed unsupported
surfaces. It rejects missing, duplicate, or overlapping classifications.

## Consequences

The current opaque inventory narrows from four constructor calls and four
operations to three of each: `flash_attention`, `transformer_block`, and the
intentional `user` kernel extension.

The original 39-ID partition now contains thirty-five typed retirements, three
current opaque identities, and one removed unsupported surface. No execution
capability is claimed for WebNN. Callers use ordinary tensor matmul until a
real graph-level WebNN backend exists.

Gate 6 retains two advertised opaque compatibility surfaces:
`flash_attention` and `transformer_block`. The user-authored kernel boundary is
an intentional extension rather than framework compatibility debt.
