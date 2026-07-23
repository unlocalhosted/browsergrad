# ADR-0047: Introduce provider-bound assignment requirement resolutions

## Status

Accepted.

## Context

Assignment profiles name 53 registered legacy requirements, but runtime
preflight represented an environment as only a string list plus a mode map.
That was sufficient for the frozen compatibility adapter, but it could not say
which provider established availability, which evidence supported it, or
whether a known definition was merely present versus actually resolved.

Static vocabulary entries also lived only in architecture JSON. Runtime
consumers could therefore duplicate the registry or infer support from method
presence and feature booleans.

## Decision

Semantic-core exports a concrete `/requirement` protocol with immutable,
versioned requirement definitions and provider-bound resolution records.
Available resolutions require exactly one provider ID, one closed provider
mode, and a deterministic evidence-ID set. Unavailable resolutions carry no
invented provider or evidence.

Runtime generates its complete sorted definition registry from
`architecture/platform-vocabulary.json`. Architecture validation compares the
generated source byte-for-byte with that vocabulary and rejects stale,
duplicated, or malformed definitions.

`createAssignmentRequirementResolutionEnvironment()` resolves every registered
definition exactly once. Only explicitly supplied providers produce
`available` records. The legacy `AssignmentCapabilityEnvironment` is derived
from those available resolutions through a compatibility bridge; definition
presence alone never satisfies a profile gate.

## Consequences

Platforms can now retain the provider and evidence behind every readiness
decision without changing existing assignment profile IDs. The generated
registry is a release-packed runtime dependency on semantic-core `0.x`, not a
hand-maintained support table.

The existing run-plan, report, handoff, and matrix APIs still accept the legacy
capability environment. Their migration to consume the resolution environment
directly is the next Gate 6 slice. Requirement availability also does not imply
semantic lowering, backend execution, or program-specific support; those
decisions require separate artifact/program-scoped capability records.
