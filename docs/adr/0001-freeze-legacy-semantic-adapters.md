# ADR-0001: Freeze Legacy Semantic Adapters

- **Status:** accepted
- **Date:** 2026-07-15
- **Decision owners:** compiler, kernels, JIT, Grad, runtime

## Context

BrowserGrad has useful shipping paths whose schemas also carry accidental or
incomplete semantics: CUDA-lite pointer fields, rank-one CuTe parser sugar, the
shape-only/f32 tensor GPU plan, opaque JIT `CUSTOM` nodes, Grad dtype/view
compatibility behavior, a misleading attention symbol, and generic runtime
capability labels.

Extending these paths would make migration to the multi-level semantic model
more expensive and could turn compatibility behavior into a permanent public
contract.

## Decision

The baselines in `architecture/semantic-freeze.json` are frozen. No new feature
may widen a frozen surface, caller set, operation set, label set, dtype/view
substitution, or unique semantic field.

The JIT baseline is the exact 36-constructor-call/39-operation inventory, not
the former token-derived allowlist. Its label field, constructor definition,
conditional reachability/effects, declared-versus-realized dtype behavior,
autograd and transform decisions, default/inspection/execution plan decisions,
realization route, and materialization boundary are part of the reviewed
baseline. Token strings from non-`CUSTOM` operations do not acquire
compatibility status.

An intentional baseline change requires all of:

1. A new accepted ADR naming the exact invariant changed.
2. Updated machine-readable baseline and focused negative tests.
3. A migration effect: delete/narrow an adapter or measurably advance its
   retirement gate.
4. Compatibility and removal-version analysis.

Raising a count or allowlist solely to make the architecture check pass is not
an accepted exception.

## Consequences

- Existing behavior remains available only to its recorded callers.
- New shared semantics enter through `browsergrad-semantic-core` artifacts.
- Architecture-check changes are review-sensitive even when product tests pass.
- The implementation ledger records every accepted exception and verification.
