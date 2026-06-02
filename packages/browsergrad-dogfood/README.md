# `@unlocalhosted/browsergrad-dogfood`

**Post-publish verification suite. Not published to npm.**

Consumes the **npm-published** `@unlocalhosted/browsergrad-*` tarballs (not
the local source tree) and runs adversarial tests against them in real
Chromium with WebGPU. Designed to catch the failure modes that
`file:`-linked workspace tests miss:

- Type contract violations between `.d.ts` and runtime
- Files stripped by `.npmignore` / `files[]` / `prepublishOnly`
- API drift between mocks (in upstream tests) and real impls
- Numerical bugs only visible on real hardware
- Bridge lifecycle, handle leaks, codegen poisoning

## Why a separate workspace package

The other packages depend on each other via the npm name. `pnpm` will resolve
`@unlocalhosted/browsergrad-kernels` inside this package's `node_modules` to
the **real npm tarball** (because we pin an exact version like `"0.1.0"`), not
to the local workspace `packages/browsergrad-kernels/` (which would require a
`workspace:*` protocol). That gap is the whole point — we want to test the
published artifact.

## Running

```sh
pnpm --filter @unlocalhosted/browsergrad-dogfood test
```

Headed Chromium is required on macOS for real WebGPU (Metal). On Linux CI,
set `BG_BROWSER_HEADLESS=1` and rely on SwiftShader Vulkan.

## Structure — coverage spans all 4 published packages

```
tests/                              # browser mode (Chromium + WebGPU)
  surface.test.ts                   — kernels public exports + types
  runtime/
    manifest.test.ts                — parseManifest adversarial (R1-R5)
    semver.test.ts                  — isSemverCompatible / assertCompatibleRuntime (R6-R11)
  numerical/                        — kernels numerical attacks
    softmax.test.ts                 — H1-H5 + H0c (known GPU bug)
    layernorm.test.ts               — H6-H7, H18 (zero-variance)
    matmul.test.ts                  — H10-H12, H15 (accumulator, NaN, tile-edge)
    attention.test.ts               — H0b cross-attention, H19 (decoder single-token)
    relu_gelu.test.ts               — H8-H9 (-0, Inf, NaN)
  shape/                            — kernels shape boundaries
    boundaries.test.ts              — H13-H15, H19 (empty, 1×1, tile-edge)
    errors.test.ts                  — H16-H17, H40 (rank, inner-dim, shape lying)
  bridge/                           — WebGpuRealizerBridge contract (PRD-011.5)
    lifecycle.test.ts               — H20-H25 (handle leaks, double-release)
    methods.test.ts                 — each of 9 bridge methods
    concurrency.test.ts             — H23 (two bridges, same device)
  codegen/                          — fused_elementwise codegen + pipeline cache
    determinism.test.ts             — H30 (same ops → same WGSL)
    cache.test.ts                   — H31, H34 (hash collision)
    errors.test.ts                  — H27-H29, H32-H33, H35

tests-node/                         # node mode (Pyodide-in-Node)
  helpers.ts                        — shared Pyodide bootstrap
  grad/
    install_and_tensor.test.ts      — G1-G15 (autograd, nn, optim, state_dict, torch alias)
  jit/
    jit_specific.test.ts            — J1-J11 (lazy IR, cache, fusion, custom kernel, ONNX)
  cross-package/
    integration.test.ts             — CP1-CP3 (grad+jit coexist, manifest vs jit version)
```

Plus `hypotheses.md` — 77 adversarial hypotheses every test file derives from.

## Coverage by package

| Package | Surface | Adversarial |
|---|---|---|
| `@unlocalhosted/browsergrad-kernels@0.1.0` | ✅ | 40 hypotheses (H1-H40) |
| `@unlocalhosted/browsergrad-runtime@0.1.1` | ✅ | 11 hypotheses (R1-R11) |
| `@unlocalhosted/browsergrad-grad@0.5.0` | ✅ | 15 hypotheses (G1-G15) |
| `@unlocalhosted/browsergrad-jit@0.8.0` | ✅ | 11 hypotheses (J1-J11) — beyond craftingattention's 61-test suite |
| Cross-package (grad+jit+runtime+kernels coexistence) | ✅ | 3 hypotheses (CP1-CP3) |

## Output convention

Tests categorize each finding:
- 🟢 **SAFE** — library survived the attack as expected
- 🔴 **BUG** — attack succeeded; file an issue (or test should fail in CI)
- 🟡 **PARTIAL** — degraded but not broken (e.g. correct but slow)
- ⚪ **N/A** — attack didn't apply

Known-failing tests use `it.fails()` so the suite stays green while
documenting the bug. Remove `.fails` when a fix lands.
