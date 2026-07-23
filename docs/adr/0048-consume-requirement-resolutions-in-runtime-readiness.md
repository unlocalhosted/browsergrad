# ADR-0048: Consume requirement resolutions in runtime readiness

## Status

Accepted.

## Context

ADR-0047 introduced provider-bound requirement environments, but public
run-plan and platform APIs still accepted only
`AssignmentCapabilityEnvironment`. A caller had to erase provider/evidence
records through the compatibility bridge before planning. Reports, handoffs,
external-runner requests, and benchmark matrices consequently exposed only
selected or missing strings.

That made the new protocol an optional side channel instead of the source of
truth for readiness UI.

## Decision

The existing run-plan, preflight, matrix, and JavaScript profile-runner
entrypoints now accept `AssignmentReadinessEnvironment`, a backward-compatible
union of the provider-bound resolution environment and the frozen legacy
capability environment.

Resolution inputs are validated, canonically reconstructed, frozen, and then
evaluated through the existing compatibility semantics. Each plan retains the
available or unavailable resolution records for every requirement referenced
by that profile. Preflight reports, platform handoffs, external-runner
requests, and benchmark rows carry those exact relevant records forward.

Architecture checks pin every migrated entrypoint to
`AssignmentReadinessEnvironment` and require every public carrier to retain an
optional readonly resolution array.

## Consequences

New platforms pass provider-bound records directly from detection through
planning and UI. They can display why a requirement is available, which mode
will execute it, and which evidence established it without joining a second
table or inferring support from method presence.

Existing callers may continue to pass `AssignmentCapabilityEnvironment`
through the compatibility window. Those legacy inputs cannot invent provider
or evidence records, so corresponding output fields remain absent.

This completes environment-level requirement consumption, not Gate 6.
Artifact/program-specific semantic capability and lowering records, followed
by generated cross-framework/platform support views, remain required.
