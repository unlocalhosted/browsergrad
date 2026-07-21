# ADR-0003: Executable Framework-Operation Contract Registry

- **Status:** accepted
- **Date:** 2026-07-22
- **Decision owners:** JIT, runtime, platform

## Context

Gate 6 requires public framework support reporting to come from the contracts
used to compile and execute operations. Method presence, opcode allowlists,
hand-written documentation, and the frozen opaque-operation inventory each
answer a different question. None can safely claim end-to-end operation
support.

`Tensor.expand` is the first operation migrated to typed IR. Its validation and
backend decisions were executable, but their public description was still
duplicated in prose. Continuing that pattern would let support tables drift as
the remaining opaque operation families migrate.

## Decision

JIT owns one versioned package asset,
`framework-operation-contracts.v1.json`. Each admitted record names one public
surface, typed opcode, shape and dtype contract, retired opaque-operation ID,
and explicit CPU, closure-autograd, symbolic-VJP, functional-grad, vmap, ONNX,
tensor-plan, WebGPU-profile, residency, and materialization decisions.

The registry is bounded and decoded as strict UTF-8 JSON with duplicate-key,
unknown-field, unknown-enum, invalid-version, and duplicate-identity rejection.
Import binds every record one-to-one with a package-owned executable validator;
an unbound record or validator fails package import. UOp construction and every
CPU, transform, export, and plan boundary invoke that binding.

`browsergrad_jit.framework_operation_support()` returns a detached,
deterministically ordered projection of the admitted executable records. A
caller cannot mutate the registry through the returned table. The WebGPU field
states an eligible backend profile, not current device availability or an
execution-evidence claim.

The semantic architecture check independently validates the registry schema,
closed decisions, declared non-`CUSTOM` opcodes, validator bindings, and the
exact partition between the original 39 opaque operation IDs and their current
opaque or typed state. This prevents retirement history from disappearing as
the mutable current inventory narrows.

## Compatibility and evolution

The schema starts at version 1.0. New decision values or fields require a
versioned contract change. Adding a migrated operation requires the accepted
opaque-baseline exception, removal of its frozen ID, an executable typed
validator, full behavior evidence, and a registry record in the same capability
commit.

The table reports only admitted typed migrations. Absence means no typed
framework-operation contract; it must not be reinterpreted as unsupported in
all compatibility tiers or supported through method presence.
