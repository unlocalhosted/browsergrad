# Contributing to browsergrad

Thanks for considering a contribution. The library is small enough that almost every PR will touch test, source, and documentation — that's the expected shape.

Read [DEVELOPMENT.md](./DEVELOPMENT.md) for setup and methodology before opening a PR.

## Ground rules

- Be kind. Disagreements about technical direction are welcome; disrespect isn't.
- One feature per PR. If you find yourself touching three concerns, split into three PRs.
- Tests are not optional. PRs without tests will be asked for tests.
- Tests must use **independent oracles** (NumPy, finite differences, hand-derived math) — never compare an impl against itself.

## What to work on

Good first contributions:

- **A new test for an existing behavior** that's currently uncovered. (See [STATUS.md](./STATUS.md) for what's tested.)
- **A new kernel** in `browsergrad-kernels` with its JS reference impl.
- **A new `browsergrad-grad.functional` op** with finite-difference gradient check.
- **A bug fix.** File an issue first describing the bug, then send a PR with a regression test.

Things that need design discussion before a PR:

- New public API on any of the three packages
- Breaking changes (anything that bumps a major version)
- Cross-package coordination changes (e.g. plumbing GPU dispatch from grad to kernels)

If you're not sure, open an issue first and ask.

## Workflow

1. Fork the repo, create a branch off `main`.
2. Install workspace deps: `pnpm install`.
3. Make your change. Add a test first (RED), then write the code to pass it (GREEN).
4. Run the full gates locally:
   ```sh
   pnpm typecheck
   pnpm test
   pnpm -F @unlocalhosted/browsergrad-grad test:integration
   pnpm -F @unlocalhosted/browsergrad-runtime test:integration
   pnpm -r run lint
   pnpm -r run build
   ```
   All of these must be green.
5. Update the relevant package's `CHANGELOG.md` under an `[Unreleased]` section. Describe the user-visible change, not the implementation detail.
6. Update [STATUS.md](./STATUS.md) if you added or removed an API surface item.
7. Open a PR. Reference the issue if there is one. Use the PR template.

## Commit messages

Conventional-commit prefixes preferred:

- `feat(grad): ...` — new feature in browsergrad-grad
- `fix(runtime): ...` — bug fix in browsergrad-runtime
- `perf(kernels): ...` — performance improvement (no behavior change)
- `test(grad): ...` — test-only change
- `docs: ...` — documentation only
- `refactor(grad): ...` — internal refactor (no behavior change)

The body should explain *why*, not just *what*. The diff already shows what.

## Code style

- TypeScript strict mode. Run `pnpm typecheck` to verify.
- Oxlint for lint. Run `pnpm -r run lint` to verify.
- No emojis in source code unless explicitly asked by the user.
- Comments explain non-obvious *why*, not *what*. Don't comment well-named code.

## Asking questions

Open a discussion or issue with the `question` label. We don't have a Discord.

## License

By contributing, you agree your contribution will be licensed under the MIT License (see [LICENSE](./LICENSE)).
