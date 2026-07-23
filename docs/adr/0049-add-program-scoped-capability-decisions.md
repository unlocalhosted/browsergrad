# ADR-0049: Add program-scoped capability decisions

## Status

Accepted.

## Context

The architecture vocabulary already registered one semantic capability and
three backends, but runtime had no canonical carrier for the lowering result of
one concrete program or semantic artifact. Requirement resolutions describe
environment providers and device facts; framework-operation support records
describe typed API contracts. Neither can establish that a particular program
lowered for a particular backend.

A hand-written platform support table would duplicate those registries and
could turn definition presence, a test path, or an available requirement into
an unsupported execution claim.

## Decision

Semantic-core exports a concrete `/capability` protocol with immutable,
versioned semantic capability definitions, backend definitions, and lowering
decisions. Every lowering decision is bound to an exact canonical program ID
or 64-hex semantic artifact hash, one registered capability, one registered
backend, and one backend-owned execution tier.

Supported and conditional decisions require a capability-owned preservation
level. Conditional decisions retain at least one exact feature, limit, or
runtime guard. Unsupported, unknown, and not-applicable decisions require a
reason and cannot claim preservation.

Runtime generates the static capability/backend registry byte-for-byte from
the architecture vocabulary. `createProgramCapabilitySupportView()` accepts
only actual decisions for one subject, rejects empty, unknown, or duplicate
capability/backend pairs, canonically reconstructs each decision through
semantic-core, and includes only the definitions referenced by those
decisions.

Architecture checking regenerates the registry in memory and rejects drift or
malformed vocabulary before package tests.

## Consequences

Platforms can carry a deterministic program support view without inferring
support from method presence, environment availability, backend registration,
or test-file references. Static definitions still contain no support state or
evidence outcome.

This does not generate cross-framework/platform views from the JIT executable
operation registry, and it does not replace terminal execution evidence.
Those remain separate Gate 6 steps.
