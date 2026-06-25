# Development

How this library family is developed and validated. Read this before sending a PR.

## Setup

```sh
pnpm install
```

That installs every package's deps in the workspace.

## Layout

```
browsergrad/
├── packages/
│   ├── browsergrad-runtime/    Pyodide-in-Worker host
│   ├── browsergrad-kernels/    WGSL kernel catalog
│   ├── browsergrad-compiler/   CUDA-lite compiler + WebGPU runner
│   ├── browsergrad-grad/       tensor + autograd library
│   └── browsergrad-primitives/ browser-safe lab primitives
├── package.json                pnpm workspaces
├── README.md
├── STATUS.md                   current state, supported APIs
├── DEVELOPMENT.md              this file
├── CONTRIBUTING.md             how to contribute
├── CODE_OF_CONDUCT.md
├── SECURITY.md
└── LICENSE
```

## Methodology

### Test-driven development

Every layer is built one test at a time:

1. Write a behavior test against an **independent oracle** — NumPy result, hand-derived math identity, or finite-difference numerical check. Never compare an implementation against itself.
2. Run it; watch it fail (`RED`).
3. Write the minimum implementation to make that one test pass (`GREEN`).
4. Never refactor while red.
5. Repeat until the feature is complete.

Refactors (like the Conv2d im2col rewrite) introduce no new tests. The existing safety net is the verification.

### Quality gates

- `pnpm typecheck` — TypeScript strict mode across all packages
- `pnpm test` — fast surface tests (~1s)
- `pnpm -F @unlocalhosted/browsergrad-grad test:integration` — full Pyodide-in-node suite (~25s; 115 tests)
- `pnpm -F @unlocalhosted/browsergrad-runtime test:integration` — runtime Python-bridge + client-routing tests (~3s; 23 tests)
- `pnpm -r run lint` — oxlint across all packages
- `pnpm -r run build` — TypeScript declaration emission for all packages

All of these should be green before opening a PR.

### Conformance and oracles

Three independent reference points are used to validate behavior:

| Oracle | Used for | Where |
|---|---|---|
| **NumPy** | Forward correctness | Compute the same op via raw NumPy in the test, compare results. |
| **Finite differences** | Gradient correctness | `(f(x+ε) − f(x−ε)) / (2ε)` per parameter, compare to analytic `.grad`. |
| **Hand-derived math** | Known closed-form values | E.g. `d(x²)/dx = 2x`, `softmax` rows sum to 1, etc. |

For cross-package, `browsergrad-kernels` JS reference impls and `browsergrad-grad` NumPy ops are tested against each other in `tests-integration/cross-package-conformance.test.ts` — two independent implementations agreeing within `1e-4`.

### Integration testing

`browsergrad-grad` and `browsergrad-runtime` integration tests boot real Pyodide in node (via the `pyodide` npm package) and execute Python. This is the only way to catch real bugs in the Python source code we ship — and it has caught real ones (e.g. closure-deletion bug in `PY_PREAMBLE`).

## Workspace commands

```sh
pnpm typecheck             # type-check all packages
pnpm test                  # all surface tests (fast)
pnpm -r run build          # build all packages
pnpm -r run lint           # lint all packages
pnpm -r run clean          # remove dist + tsbuildinfo
pnpm -F <pkg> test:integration   # integration tests for one package
```

## Adding a feature

1. **Plan** — what's the interface, what behaviors define it correct? Write them down.
2. **Tracer bullet** — write the first test against a hand-derived oracle, watch it fail, write the smallest impl, watch it pass.
3. **Loop** — one behavior at a time. Independence properties (multi-channel, batch) often pass without code change — they're regression checks that document invariants.
4. **Backward** — implement gradients last, verify with finite differences.
5. **Integration** — add an end-to-end test if the feature combines with the rest of the library in a non-trivial way.

## Reference: the architecture

- **`browsergrad-runtime`** owns the Pyodide worker. It exposes a stable JS API (`createSession`) and a stable wire protocol between the main thread and the worker. The Python side installs a `browsergrad` module via `PY_PREAMBLE` that lets user Python emit structured assertions and artifacts back to JS.

- **`browsergrad-kernels`** is the WGSL catalog. It depends on nothing. Each kernel ships the WGSL string, a JS dispatcher, AND a pure-JS reference implementation. The reference impls are testable in node; the WGSL impls require WebGPU.

- **`browsergrad-compiler`** owns CUDA-lite semantics: source/context normalization, parser/analyzer, Kernel IR, lockstep CPU reference, WGSL emission, WebGPU runner orchestration, and pinned real-world corpus gates. It consumes `browsergrad-kernels` dispatch helpers; platform code should consume compiler capability reports instead of copying CUDA heuristics.

- **`browsergrad-grad`** is the tensor + autograd library, in Python, embedded as TypeScript string constants. It's installed into a Pyodide target via `installGrad(target)` — duck-typed so it works with the runtime's `Session` OR any other Pyodide setup. Optional `install_torch_alias()` registers a `torch` namespace shim.

## Releasing

Each package is versioned independently. Bump the `package.json` version, the `__version__` constant in the package's `__init__.py` (for grad), and the install-time smoke check pin. Update the package's `CHANGELOG.md`. Tag the release. `npm publish` from the package directory (currently a manual step until CI publishes).
