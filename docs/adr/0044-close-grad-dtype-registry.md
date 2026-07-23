# ADR-0044: Close the Grad eager dtype registry

## Status

Accepted.

## Context

Grad advertised a closed BrowserGrad/PyTorch-shaped dtype alias table, but
`_resolve_dtype` delegated every unknown string and non-string specification
to NumPy. That admitted storage such as complex, object, structured, and
datetime dtypes even though Grad operations, autograd, serialization, and
backend contracts do not support them.

The result depended on the installed NumPy parser rather than a versioned Grad
capability. A spelling such as `complex64` could therefore allocate an eager
tensor despite being outside the documented storage set.

## Decision

String dtype requests are resolved only through the frozen BrowserGrad alias
table. Unknown strings reject before allocation, including NumPy abbreviation
spellings that are not public BrowserGrad aliases.

NumPy dtype objects and scalar types remain valid dtype specifications only
when they resolve to the closed eager storage set: bool, float16/32/64,
int8/16/32/64, or uint8/16/32/64. Complex, object, structured, datetime, and
every other unregistered storage type reject before allocation.

The same twelve-dtype set governs `from_numpy`, so unsigned 16/32/64-bit arrays
join the existing zero-copy input boundary. Individual operations retain
narrower dtype contracts and reject an admitted storage dtype at the operation
boundary when that operation has no semantics for it.

`bf16` and `bfloat16` retain their distinct unsupported-storage diagnostic from
ADR-0037.

## Consequences

Grad dtype acceptance is now package-owned, versioned, and independent of
NumPy parser expansion. The twelve admitted physical storage dtypes remain
available through both explicit string aliases and NumPy dtype objects/scalar
types.

This closes the dtype-registry compatibility-debt record. Constructor-default
classification and owning `Tensor.expand()` materialization remain as the two
eager Grad compatibility records.
