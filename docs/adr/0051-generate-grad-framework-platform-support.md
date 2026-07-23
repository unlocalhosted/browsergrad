# ADR-0051: Generate Grad framework platform support

## Status

Accepted.

## Context

ADR-0050 gives runtime a framework-neutral composition seam and JIT a generated
36-operation source. Grad still had no equivalent source, even though its
schema-v2 compatibility inventory already freezes 22 verified eager
dtype/view/materialization/device contracts, their exact source definitions,
and their executable fixture cases.

A hand-maintained Grad support table would immediately create another truth.
Treating eager method presence as support would also overclaim symbolic
autograd, functional transforms, export, tensor planning, or WebGPU execution
that Grad does not provide.

## Decision

Generate Grad's platform source byte-for-byte from
`architecture/grad-compatibility-inventory.json`. The generator accepts only
the exact schema-v2 inventory identity and verified target contracts, validates
the closed behavior shape, rejects duplicate or unknown mappings, and sorts
the 22 records by canonical operation ID.

Each record preserves the inventory's public surface, shape/condition, dtype,
CPU/refusal, eager autograd, residency, and materialization facts. Symbolic
VJP, functional grad, vmap, and ONNX remain explicitly
`not-applicable-eager`; tensor plans are not applicable to the eager framework;
WebGPU is refused for the NumPy-reference execution context. A rejected dtype
or device contract cannot imply residency or allocation.

Grad exports a detached `frameworkPlatformSupportSource()` from its JavaScript
root. The semantic architecture gate regenerates and exact-compares the
checked-in source, and mutation tests reject stale output and duplicate
inventory behavior. A packed fresh consumer composes Grad's 22 records and
JIT's 36 records through runtime without adding framework dependencies to
runtime.

## Consequences

The initial Gate 6 framework-convergence exit is met: both BrowserGrad
framework surfaces now generate platform records from their executable
contracts, and runtime consumes them alongside provider-bound requirements
and subject-bound lowering decisions.

This does not make an eager Grad contract a JIT contract, turn a framework
decision into device availability, or replace terminal execution evidence.
Broader Grad dtype/layout coverage, new JIT operation families, and new backend
profiles require new executable contracts and regenerated records.
