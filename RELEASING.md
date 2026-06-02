# Releasing

How to publish a package to npm. Independent versions per package.

## One-time setup

1. **npm org**: `@unlocalhosted` org must exist on npm with publishers added.
   You're already set up — verified `dprophecguy` as owner.
2. **Repo secret**: in GitHub repo Settings → Secrets → Actions, add
   `NPM_TOKEN`. Create the token at
   <https://www.npmjs.com/settings/dprophecguy/tokens> with the **Automation**
   type (not "Publish") so it skips 2FA. Restrict scope to packages under
   `@unlocalhosted` if desired.
3. **Repo permissions**: Settings → Actions → General → Workflow permissions
   → "Read and write permissions" (needed for `gh release create`).

That's it. The two workflows then handle every release.

## Releasing a package

Pick the package + bump the version:

```sh
# Example: bump jit from 0.8.0 → 0.8.1
cd packages/browsergrad-jit
```

1. **Update `package.json` version**: `"version": "0.8.1"`.
2. **Update `CHANGELOG.md`** for the package. Add a section at the top:
   ```md
   ## [0.8.1] — 2026-06-02

   ### Fixed
   - Brief bullet description of what changed.
   ```
3. **Optionally update the top-level `CHANGELOG.md`** for visibility.
4. **Commit and push to `main`** — CI runs.
5. **Tag and push**:
   ```sh
   git tag jit-v0.8.1                    # tag = <shortname>-v<semver>
   git push --tags
   ```

The Release workflow then:
- Verifies the tag version matches `package.json`
- Runs full build + surface tests + integration tests (jit/grad)
- Publishes to npm with `--provenance`
- Creates a GitHub Release with the changelog excerpt

Tag format per package:

| Package | Tag prefix | Example |
|---|---|---|
| `@unlocalhosted/browsergrad-jit` | `jit-v` | `jit-v0.8.1` |
| `@unlocalhosted/browsergrad-runtime` | `runtime-v` | `runtime-v0.1.2` |
| `@unlocalhosted/browsergrad-kernels` | `kernels-v` | `kernels-v0.1.1` |
| `@unlocalhosted/browsergrad-grad` | `grad-v` | `grad-v0.5.1` |

## Dry run locally

```sh
# Build everything
pnpm -r build

# See exactly what would be in the npm tarball — no surprises
cd packages/browsergrad-jit
npm pack --dry-run

# Test the tarball in a throwaway project before tagging
npm pack
mkdir /tmp/bg-test && cd /tmp/bg-test
npm init -y
npm install /path/to/unlocalhosted-browsergrad-jit-0.8.1.tgz
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

- **`npm publish` fails with 403**: token expired or wrong scope. Regenerate
  with the **Automation** type at <https://www.npmjs.com/settings/dprophecguy/tokens>.
- **`npm publish` fails with "cannot publish over existing version"**: bump
  the version and re-tag. npm doesn't allow republishing the same `name@version`
  even if you `npm unpublish` first (the name is reserved for 72 hours).
- **Tag pushed but workflow didn't run**: GitHub limits Actions on tags
  pushed by other Actions (loop protection). Use a personal PAT in the secret
  to bypass, or push tags from your local machine.
- **Provenance signing fails**: workflow needs `id-token: write` permission.
  Already configured in `release.yml`.

## Manual publish (if CI is unavailable)

```sh
pnpm -r build
pnpm -r test
cd packages/browsergrad-jit
npm login
npm publish --access public
git tag jit-v0.8.1
git push --tags
```

Skip this path when possible — the CI workflow has provenance + uniform test
gates that catch dirty-tree publishes.

## Adding a new package to the release pipeline

1. Add the package under `packages/<name>/`. Standard layout: `dist/`, `src/`,
   `package.json` with `publishConfig.access = "public"`, README, LICENSE,
   CHANGELOG.
2. Add a tag-prefix entry in `.github/workflows/release.yml` under `on.push.tags`.
3. Add a row to the tag-format table above.

The release workflow detects the prefix → derives `browsergrad-<shortname>`
as the package directory.
