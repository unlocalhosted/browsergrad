<!--
Before you open this PR, please confirm:

- [ ] You read CONTRIBUTING.md
- [ ] One feature per PR — if this touches >1 concern, split it
- [ ] Your branch is up to date with `main`
-->

### What this changes
<!-- A short summary in plain language. Skip implementation details — those are in the diff. -->

### Why
<!--
What problem does this solve? Link to the issue if there is one
(e.g. "Closes #42"). If it's a new feature, explain the use case.
If it's a bug fix, describe how to reproduce the old behavior.
-->

### Tests
<!--
Every PR has tests. Describe what oracle you used:
- Hand-derived math identity
- NumPy reference
- Finite-difference gradient check
- scipy / statistical property
- New end-to-end integration check

Never compare an implementation against itself.
-->

### Quality gates I ran locally

- [ ] `pnpm typecheck`
- [ ] `pnpm test` (all surface tests)
- [ ] `pnpm -F @unlocalhosted/browsergrad-grad test:integration` (if grad changed)
- [ ] `pnpm -F @unlocalhosted/browsergrad-runtime test:integration` (if runtime changed)
- [ ] `pnpm -r run lint`
- [ ] `pnpm -r run build`

### Versioning + changelog
<!-- Required if user-visible API changed. Skip for internal refactors. -->

- [ ] Bumped package version
- [ ] Updated package CHANGELOG
- [ ] Updated `__version__` in the package's `__init__.py` (grad only)
- [ ] Updated installer smoke check pin (grad only)
- [ ] Updated STATUS.md surface inventory (if a public API was added/removed)

### Breaking changes
<!--
List any breaking change. If there are any, the PR must bump a major
version. Otherwise leave empty.
-->
