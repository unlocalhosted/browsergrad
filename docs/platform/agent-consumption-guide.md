# Agent Consumption Guide

This guide is for coding agents that need to use BrowserGrad packages correctly
from npm or inside this monorepo.
For full package requirements, low-level design, production gates, and research
basis, read [`package-requirements-lld.md`](./package-requirements-lld.md).

## First Decision

| User intent | Use |
| --- | --- |
| "Run Python in browser", "Pyodide worker", "lab runtime" | `@unlocalhosted/browsergrad-runtime` |
| "PyTorch-like training", "lazy tensor", "fusion", "ONNX", "custom WGSL" | `@unlocalhosted/browsergrad-jit` |
| "Simple autograd", "teaching tensor", "eager backward" | `@unlocalhosted/browsergrad-grad` |
| "Eager autograd plus WebGPU forward matmul/softmax/layernorm/attention" | `@unlocalhosted/browsergrad-grad` + `@unlocalhosted/browsergrad-kernels` |
| "WebGPU kernels", "WGSL", "FlashAttention reference", "rubric tensor compare" | `@unlocalhosted/browsergrad-kernels` |
| "CUDA-like kernel source", "compile CUDA to WGSL", "real CUDA corpus" | `@unlocalhosted/browsergrad-compiler` |
| "Tokenization/data/eval/simulation/RL helper" | `@unlocalhosted/browsergrad-primitives` |

## Import Rules

- Prefer documented top-level exports.
- Use `@unlocalhosted/browsergrad-grad/kernel-device` only for the explicit
  eager `device=` bridge; regular eager autograd does not need it.
- Use kernels subpaths only when the caller benefits from a smaller domain entry:
  `reference`, `wgsl_program`, `float16`, `cuda_concepts`, `cuda_program`, or
  `rubric`.
- Do not import private files under `dist/` or `src/` from consumer code.
- Do not assume a workspace package is publishable just because local imports
  work. Verify the packed tarball.

## CUDA-Lite Agent Flow

1. Use `detectKernelFeatures(device)` from `browsergrad-kernels`.
2. Convert features with `compileCudaLiteOptionsFromKernelFeatures()`.
3. Compile with `compileCudaLiteKernelForWebGpu()` when host-orchestrated plans
   such as grid-sync, runtime copy, or dynamic launch may be allowed.
4. Inspect `createCudaWebGpuExecutionPlan()` and
   `summarizeCudaWebGpuExecutionPlan()` before dispatch.
5. Run `runCompiledKernelReference()` as the correctness oracle.
6. Run `runCompiledKernelWebGpu()` only when the plan is supported.

## Release Safety For Agents

Before telling a user "published" or "release-ready":

```sh
git status --short --branch
pnpm -r build
pnpm test:release-packages
node scripts/publish-missing-npm.mjs --dry-run
```

After CI publishes, verify npm directly:

```sh
npm view @unlocalhosted/browsergrad-kernels version exports --json
npm view @unlocalhosted/browsergrad-compiler version dependencies --json
```

The compiler package must never publish a raw `workspace:*` dependency. If npm
shows `workspace:*`, the release is broken even if CI was green.

## What Not To Do

- Do not use `npm publish` for workspace packages with `workspace:*`
  dependencies. Use `pnpm publish` so workspace ranges are rewritten.
- Do not trust stale local `dist/` files. Always rebuild before packing.
- Do not claim WebGPU coverage from unit tests. Use browser/WebGPU gates when
  shader behavior matters.
- Do not skip tests to make release green.
