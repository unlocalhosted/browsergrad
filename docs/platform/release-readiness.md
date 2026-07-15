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

Do not invoke package-local `npm publish` or `pnpm publish`. Publication is only
available through an explicit staged mode in the protected workflows. The
validation job runs `pnpm pack` after `prepublishOnly`; pnpm rewrites workspace
ranges in that tarball. The protected job receives that exact tarball, verifies
its recorded SHA-512 and identity, then invokes `npm publish <tarball>` with
lifecycle scripts disabled. No source tree is rebuilt or repacked beside npm
credentials.

`scripts/publish-missing-npm.mjs --dry-run` is read-only metadata evidence.
`--stage-dir` requires a clean tracked worktree and has no publication
authority. `--publish-staged` requires the protected workflow, explicit
provenance, exact source revision, and the immutable staged manifest. There is
no direct non-staged publication mode.

Protected publication pins npm `11.12.1` and rejects npm older than `11.12.0`.
That floor is required because provenance identity is decoded only from the
exact attestation bundle returned as cryptographically verified by
`npm audit signatures --include-attestations`; a separately fetched registry
statement is not accepted as proof.

The script derives a deterministic dependency-first order over runtime,
optional, and peer workspace edges. The semantic view-copy release chain is
semantic-core, then kernels, then compiler. Every already-published dependency
must match the staged tarball before the first registry mutation.
Manual batch mode stages and audits all current public package versions and
publishes only missing ones, so rerunning after a partial publication cannot
filter a failed target out of the proof path. Selected tag mode requires exact
current tag/workflow/commit provenance for an existing target; batch resumes
accept prior provenance only from the approved tag or protected-main workflow,
with the attested commit reachable from `origin/main`.

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
git tag compiler-v0.2.0
git push origin compiler-v0.2.0
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
