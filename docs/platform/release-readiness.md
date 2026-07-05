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

## Workspace Dependency Rule

Use `pnpm publish`, not `npm publish`, for workspace packages. `pnpm publish`
rewrites `workspace:*` dependency ranges to concrete package versions in the
packed tarball. `npm publish` does not, and can publish an unusable package with
raw `workspace:*` metadata.

`scripts/publish-missing-npm.mjs` intentionally uses `pnpm publish` and
topologically publishes workspace dependencies before dependents. For the
compiler/kernels pair, kernels must publish before compiler.

## Packed Tarball Contract

`pnpm test:release-packages` verifies the artifact that npm will receive:

- `@unlocalhosted/browsergrad-kernels` tarball has rebuilt `dist/`.
- Kernels root export includes WGSL program, float16, CUDA concept, CUDA
  program, and rubric helpers.
- Kernels subpath exports exist for `./wgsl_program`, `./float16`,
  `./cuda_concepts`, `./cuda_program`, and `./rubric`.
- `@unlocalhosted/browsergrad-compiler` tarball depends on kernels via a
  concrete npm version, never `workspace:*`.

## Tag Releases

Single-package release workflow tags use:

```sh
git tag kernels-v0.1.2
git push origin kernels-v0.1.2
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
```

The expected compiler dependency is the just-published kernels version.

