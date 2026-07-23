# ADR-0050: Compose framework platform support views

## Status

Accepted.

## Context

Runtime now owns provider-bound requirement environments and program-scoped
lowering support, while JIT owns an executable 36-operation framework
registry. Platform UI still had to fetch those records independently, and the
JIT registry was exposed only through Python after installing the package into
Pyodide.

Making runtime import JIT would violate the tensor-agnostic runtime boundary.
Copying JIT's support table into runtime would create a second hand-maintained
truth. Flattening requirement availability, framework eligibility, and a
program lowering result into booleans would erase their different scopes.

## Decision

JIT exports `frameworkOperationSupport()` and
`frameworkPlatformSupportSource()` from its JavaScript root. Both parse and
project the same generated JSON registry loaded by Python's executable
validators, preserve the exact ten decision categories, and return detached
records. The platform source carries package and contract identity but makes no
runtime availability or execution claim.

Runtime exports `createFrameworkPlatformSupportView()`. It accepts one complete
provider-bound requirement environment, one raw program-support input, and
one to sixteen framework-neutral support sources. Runtime canonically
reconstructs the requirement and program records, bounds and validates every
framework source, rejects open fields and duplicate framework/operation IDs,
sorts all identities, and freezes the result.

The output retains requirement resolutions, program lowering decisions, and
framework contracts in separate fields. It does not synthesize a common
`supported` boolean.

The two packages remain dependency-independent. A packed fresh-consumer test
installs both packages and proves that JIT's structural output is accepted by
runtime without a runtime-to-JIT or JIT-to-runtime dependency.

## Consequences

Platform code can render one deterministic payload while preserving the
authority and scope of each fact. Additional eager or lazy frameworks can
provide the same structural source without changing runtime.

JIT now has a real generated platform source. Grad still needs a generated
source derived from its frozen executable compatibility contracts before the
cross-framework view represents both BrowserGrad framework surfaces.
ADR-0051 subsequently supplies that source and proves the combined packed
consumer.
