# Security

## Reporting a vulnerability

If you've found a security issue in browsergrad, **do not open a public GitHub issue**. Email the maintainers at `security@unlocalhosted` (or open a private security advisory via the GitHub UI under the repo's Security tab).

Please include:

- A description of the issue and the impact
- Steps to reproduce (a minimal failing snippet is ideal)
- Affected version(s) — package name + version from `pnpm pack`
- Your contact info for follow-up

We'll acknowledge within 3 business days and aim to ship a fix within 30 days for high-severity issues. We'll credit you in the release notes unless you'd rather stay anonymous.

## Threat model

browsergrad is a library that runs **untrusted Python code inside a Pyodide Web Worker**, plus dispatches **WGSL compute shaders to the GPU**. The threat surface is roughly:

| Component | What it touches | Threat |
|---|---|---|
| `browsergrad-runtime` | Pyodide WASM sandbox, virtual FS, structured messages | Code-injection via the message protocol; sandbox escape (Pyodide's responsibility, not ours) |
| `browsergrad-kernels` | `GPUDevice`, `GPUBuffer`, WGSL shaders | Malformed WGSL crashing the device; OOB reads via incorrect bind groups |
| `browsergrad-grad` | Python autograd in Pyodide | Untrusted user Python — already sandboxed by Pyodide; we don't add a layer |

We do **not** consider the following in scope:

- Pyodide WASM sandbox escapes (upstream)
- WebGPU driver bugs (browser / driver vendor)
- DoS via expensive computations (the runtime exposes `AbortSignal` + `timeoutMs` for callers to handle this)
- Information disclosure via timing channels in GPU compute (not modeled at this layer)

## Hardening guidance for consumers

If you embed browsergrad in a product:

1. **Always set `timeoutMs`** on `session.exec()` for any user-provided code. Without it, an infinite loop runs forever.
2. **Set `AbortSignal`** for UX-driven cancellation; the runtime cooperatively cancels via SIGINT when SharedArrayBuffer is available and falls back to terminating the worker.
3. **Serve Pyodide assets same-origin.** ADR-style: do not load from a third-party CDN. The runtime accepts `pyodideIndexURL` precisely so you control this.
4. **Run hidden grading server-side.** The browser-tier runtime is for visible feedback; never for credit-bearing evaluation.
5. **Enable cross-origin isolation** (`COOP: same-origin`, `COEP: require-corp`) if you need cooperative cancellation via SharedArrayBuffer. The library auto-detects and silently falls back if not isolated.

## Supported versions

Currently the latest release of each package is the only supported version. We may backport critical security fixes to a previous minor on request, but the default is "upgrade to latest."

| Package | Latest |
|---|---|
| `@unlocalhosted/browsergrad-runtime` | `0.1.1` |
| `@unlocalhosted/browsergrad-kernels`  | `0.1.0` |
| `@unlocalhosted/browsergrad-grad`     | `0.4.7` |
