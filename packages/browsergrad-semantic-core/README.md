# `@unlocalhosted/browsergrad-semantic-core`

Private Gate 1 incubation package for BrowserGrad's canonical semantic wire
format and value/layout model.

Only explicit subpaths exist:

- `@unlocalhosted/browsergrad-semantic-core/schema`
- `@unlocalhosted/browsergrad-semantic-core/layout`

There is no root barrel. The package must remain browser-safe and cannot import
compiler frontends, framework packages, runtimes, or device APIs. It is not yet
a public runtime dependency. Before a published BrowserGrad package imports it
at runtime, this package must become a packed and release-tested `0.x` package.

Current status and evidence live in
[`docs/internal/package-requirements-implementation-ledger.md`](../../docs/internal/package-requirements-implementation-ledger.md).
