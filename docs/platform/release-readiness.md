# Release Readiness

BrowserGrad packages are independently versioned npm packages. A pushed commit is
not a release, and a green workspace build is not proof that the npm tarball is
usable.

## Required Pre-Publish Gates

Run these before tagging or dispatching a publish workflow:

```sh
git status --short --branch
pnpm -r build
pnpm test:release-packages
node scripts/publish-missing-npm.mjs --dry-run
```

For compiler releases, also run the scoped compiler gates appropriate to the
change. Release CI runs the real-world CUDA gate before npm publish.

For `@unlocalhosted/browsergrad-kernels` 0.2.0 and later, publishing also
requires actual-device view-copy evidence:

```sh
# Local driver (headed Chromium)
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:view-copy:required

# CI/headless environment with a WebGPU adapter (for example SwiftShader)
BG_BROWSER_HEADLESS=1 pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:view-copy:required
```

The required lane emits one terminal `browsergrad.execution-evidence@1`
record and fails on adapter/device absence, validation, out-of-memory, device
loss, comparison failure, timeout, or a skipped case. Publish workflows retain
the complete lane log as a commit-addressed artifact. An advisory `not-run`
record is useful local evidence but is not a release pass.

## Workspace Dependency Rule

Use `pnpm publish`, not `npm publish`, for workspace packages. `pnpm publish`
rewrites `workspace:*` dependency ranges to concrete package versions in the
packed tarball. `npm publish` does not, and can publish an unusable package with
raw `workspace:*` metadata. Tag-triggered CI runs on a detached HEAD, so release
workflows pass `--no-git-checks` while still relying on GitHub tag/version
validation.

`scripts/publish-missing-npm.mjs` intentionally uses `pnpm publish` and
topologically publishes workspace dependencies before dependents. The semantic
view-copy release chain is semantic-core, then kernels, then compiler. Verify
each dependency version on npm before publishing the next package.

Do not run the non-dry-run `publish-missing-npm.mjs` path manually when it
would publish kernels unless the required WebGPU lane has passed for the exact
commit and its terminal evidence log will be retained. Prefer the guarded
workflow dispatch. The script's dry run is metadata evidence only; it does not
prove device execution.

Kernels' `prepublishOnly` hook also requires
`BG_REQUIRED_WEBGPU_EVIDENCE_COMMIT` to equal the full current `HEAD`. Official
workflows set it only after the strict lane succeeds. The hook also rejects
tracked or untracked kernels, semantic-core, or lockfile changes relative to
that commit. This marker prevents an accidental direct/dirty publish; it is not
a substitute for retaining the terminal evidence log.

## Packed Tarball Contract

`pnpm test:release-packages` verifies the artifact that npm will receive:

- `@unlocalhosted/browsergrad-kernels` tarball has rebuilt `dist/` and an exact
  dependency on the packed semantic-core version.
- Kernels root export includes WGSL program, float16, CUDA concept, CUDA
  program, and rubric helpers.
- Kernels subpath exports exist for `./wgsl_program`, `./float16`,
  `./cuda_concepts`, `./cuda_program`, `./rubric`, and
  `./semantic_view_copy`.
- A fresh temporary consumer resolves semantic-core and kernels through bare
  package specifiers, typechecks them, and prepares the same verified
  CPU/WebGPU view-copy specialization.
- `@unlocalhosted/browsergrad-compiler` tarball depends on kernels via a
  concrete npm version, never `workspace:*`.

## Tag Releases

Single-package release workflow tags use:

```sh
git tag semantic-core-v0.2.0
git push origin semantic-core-v0.2.0
git tag kernels-v0.2.0
git push origin kernels-v0.2.0
git tag compiler-v0.1.2
git push origin compiler-v0.1.2
```

For dependent releases, publish dependency tags first and verify npm before
pushing the dependent tag.

## Post-Publish Verification

Verify npm, not local package files:

```sh
npm view @unlocalhosted/browsergrad-kernels version exports --json
npm view @unlocalhosted/browsergrad-compiler version dependencies --json
npm view @unlocalhosted/browsergrad-semantic-core version exports --json
```

The expected kernels dependency is the just-published semantic-core version;
the expected compiler dependency is the just-published kernels version.
