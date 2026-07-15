# Releasing

How to publish a package to npm. Independent versions per package.

## One-time setup

1. **npm org**: `@unlocalhosted` org must exist on npm with publishers added.
   You're already set up — verified `dprophecguy` as owner.
2. **Preferred auth**: configure each package's npm trusted publisher for the
   `release.yml` GitHub Actions workflow and `npm-production` environment.
   Trusted publishing uses short-lived OIDC credentials and generates
   provenance automatically; this workflow receives no long-lived npm token.
   npm permits one trusted publisher per package.
   See [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/).
3. **Manual-workflow fallback**: `publish-npm.yml` uses `NPM_TOKEN`. Only
   granular access tokens are supported by npm; scope a short-lived read/write
   token to the required packages and set its expiry. Remove it after fallback
   publication. See [npm access-token guidance](https://docs.npmjs.com/about-access-tokens/).
4. **Protected environment**: create `npm-production` under GitHub Settings →
   Environments, require maintainer review, and allow only `main` plus the seven
   documented release-tag patterns. Both workflows independently reject SHAs
   not reachable from `origin/main`.
5. **Repo permissions**: Settings → Actions → General → Workflow permissions
   → "Read and write permissions" (needed for `gh release create`).

That's it. The two workflows then handle every release.

## Releasing a package

Pick the package + bump the version:

```sh
# Example: bump jit from 0.8.2 → 0.9.0
cd packages/browsergrad-jit
```

1. **Update `package.json` version**: `"version": "0.9.0"`.
2. **Update `CHANGELOG.md`** for the package. Add a section at the top:
   ```md
   ## [0.9.0] — 2026-07-15

   ### Fixed
   - Brief bullet description of what changed.
   ```
3. **Optionally update the top-level `CHANGELOG.md`** for visibility.
4. **Commit and push to `main`** — CI runs.
5. **Tag and push**:
   ```sh
   git tag jit-v0.9.0                    # tag = <shortname>-v<semver>
   git push origin jit-v0.9.0
   ```

The Release workflow then:
- Verifies the tag version matches `package.json`
- Builds all packages and verifies architecture plus packed-package contracts
- Runs surface tests, integration tests for JIT/Grad, and required device gates
- Generically verifies the selected package's complete workspace dependency closure
- Packs once after validation, records SHA-512 integrity, and uploads that immutable artifact
- Publishes the exact tarball with lifecycle scripts disabled from a separate protected job
- Installs pinned npm `11.12.1` with scripts disabled and requires npm
  `>=11.12.0`, the first line used here that returns cryptographically verified
  attestation bundles from `npm audit signatures --include-attestations`
- Rechecks registry integrity/tree equality and validates the SLSA statement
  decoded from npm's exact verified bundle against subject, workflow, ref,
  repository, and commit before creating the GitHub Release
- Creates a GitHub Release with the changelog excerpt in a third job that has
  repository write access but no npm/OIDC publication authority

Single-package tags do not publish dependencies. Publish in dependency order:

1. `browsergrad-semantic-core` (runtime and primitives are independent)
2. `browsergrad-kernels`
3. `browsergrad-compiler`, `browsergrad-grad`, and `browsergrad-jit` in any order

JIT keeps kernels and Pyodide optional peers so its core/source/Node-adapter
imports work without either package installed. A JIT tag still requires the
exact workspace semantic-core and kernels versions to be present on npm before
the required integrated WebGPU evidence lane runs and publication begins.

Tag format per package:

| Package | Tag prefix | Example |
|---|---|---|
| `@unlocalhosted/browsergrad-jit` | `jit-v` | `jit-v0.9.0` |
| `@unlocalhosted/browsergrad-runtime` | `runtime-v` | `runtime-v0.1.2` |
| `@unlocalhosted/browsergrad-primitives` | `primitives-v` | `primitives-v0.1.1` |
| `@unlocalhosted/browsergrad-kernels` | `kernels-v` | `kernels-v0.2.0` |
| `@unlocalhosted/browsergrad-grad` | `grad-v` | `grad-v0.5.2` |
| `@unlocalhosted/browsergrad-compiler` | `compiler-v` | `compiler-v0.2.0` |
| `@unlocalhosted/browsergrad-semantic-core` | `semantic-core-v` | `semantic-core-v0.2.0` |

## Dry run locally

```sh
# Build and validate packed metadata, exports, declarations, and fresh consumers
pnpm -r build
pnpm test:release-packages

# Create the same workspace-rewritten tarball shape used by publication
mkdir -p /tmp/bg-pack
pnpm --filter @unlocalhosted/browsergrad-jit pack --config.ignore-scripts=true --pack-destination /tmp/bg-pack

# Test the tarball in a throwaway project before tagging
mkdir /tmp/bg-test && cd /tmp/bg-test
npm init -y
npm install /tmp/bg-pack/unlocalhosted-browsergrad-jit-0.9.0.tgz
node -e "import('@unlocalhosted/browsergrad-jit').then(m => console.log(Object.keys(m)))"
```

If the imports resolve and types load, the published version will too.

## Versioning policy

- **0.x line**: API stable within a minor version; breaking changes allowed in
  minor bumps (`0.8.x` → `0.9.0`). Patch bumps (`0.8.0` → `0.8.1`) are bugfix
  and additive only.
- **1.x line**: not yet. Bumping to `1.0.0` implies long-term API stability —
  flip only when ready to support 12+ months of compat.
- Each package has its own version. They don't have to move together.

## If something goes wrong

- **`npm publish` fails with 403**: verify trusted-publisher repository,
  workflow, and environment settings. For manual fallback, create a short-lived
  granular read/write token scoped only to required packages; npm no longer
  supports legacy token types.
- **`npm publish` fails with "cannot publish over existing version"**: bump
  the version and re-tag. npm doesn't allow republishing the same `name@version`
  even if you `npm unpublish` first (the name is reserved for 72 hours).
- **Tag pushed but workflow didn't run**: GitHub limits Actions on tags
  pushed by other Actions (loop protection). Use a personal PAT in the secret
  to bypass, or push tags from your local machine.
- **Provenance signing fails**: workflow needs `id-token: write` permission.
  Already configured in `release.yml`.

## Manual publish (if CI is unavailable)

Use `.github/workflows/publish-npm.yml` through GitHub's manual workflow
dispatch from protected `main`. It builds, validates, captures required
exact-commit WebGPU evidence, and stages all candidate tarballs without npm
credentials or OIDC authority. A separate protected job receives only those
artifacts and publish credentials. `scripts/publish-missing-npm.mjs` derives a
deterministic dependency-first order from public workspace runtime, optional,
and peer dependencies. A batch stages and audits every current public package,
then mutates only versions that are missing. An existing batch target resumes
only after exact integrity/tree equality, npm/Sigstore verification, approved
workflow/ref provenance, and protected-main reachability. A selected tag resume
additionally requires the exact current workflow, tag, and commit identity.
Already-published dependencies from earlier releases must match the staged
artifact, carry provenance from an approved BrowserGrad release workflow, and
attest a commit reachable from protected `main`; same-version drift fails before
any dependent publication.

Do not run package-local `npm publish` as a fallback. The workflows freeze and
transfer one validated tarball, enforce dependency order, and verify its final
registry identity. Package-local publication breaks that chain.

## Adding a new package to the release pipeline

1. Add the package under `packages/<name>/`. Standard layout: `dist/`, `src/`,
   `package.json` with exact repository metadata, `publishConfig.access =
   "public"`, a complete `prepublishOnly` gate, README, LICENSE, and CHANGELOG.
   Do not add artifact-mutating `prepare`, `prepack`, `postpack`, `publish`, or
   `postpublish` lifecycle scripts.
2. Add a tag-prefix entry in `.github/workflows/release.yml` under `on.push.tags`.
3. Add a row to the tag-format table above.

The release workflow detects the prefix → derives `browsergrad-<shortname>`
as the package directory.
