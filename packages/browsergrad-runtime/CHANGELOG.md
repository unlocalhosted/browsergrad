# Changelog

All notable changes to `@unlocalhosted/browsergrad-runtime`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-25

Initial functional release. The public API in `src/types.ts` is stable; bumping
to `0.2.0` only on breaking changes.

### Added

- `createSession(options)` — boots Pyodide in a Web Worker.
- `Session.fs.write(path, content)` / `Session.fs.read(path)` — virtual FS access.
- `Session.exec({ code, timeoutMs, signal, onStdout, onStderr, onAssertion, onArtifact })`
  — execute Python with streaming output and structured event handlers.
- `Session.interrupt()` — cooperative cancel via SharedArrayBuffer + SIGINT.
- `Session.clearNamespace()` — reset user globals while keeping preloaded modules.
- `Session.dispose()` — idempotent shutdown, rejects all in-flight requests.
- `Session.canInterrupt` — runtime feature-detection flag (true iff
  `SharedArrayBuffer` and `crossOriginIsolated` are available).
- Cooperative cancel: `AbortSignal` and `timeoutMs` first try `setInterruptBuffer`,
  then fall back to `worker.terminate()` after a 500 ms grace period.
- Worker override: `SessionOptions.worker` lets consumers supply their own
  pre-constructed `Worker` for bundlers that don't support
  `new URL("./worker/index.js", import.meta.url)`.
- Python bridge module `browsergrad`, registered via Pyodide's `registerJsModule`:
  ```python
  import browsergrad as bg
  bg.assert_pass("test_name", duration_ms=12.3)
  bg.assert_fail("test_name", "msg", expected=3, actual=2)
  bg.assert_error("test_name", "msg", exc=caught_exception)
  bg.log("status", "training complete", level="info")
  bg.emit_json("loss_curve", {"steps": [...], "loss": [...]})
  bg.emit_image("filter_0", "image/png", base64_str)
  ```
- TypeScript declarations + source maps for every emitted module.
- Vitest test suite covering the type surface and argument-validation paths.

### Deferred

These are not in 0.1.0; they're additive, planned for 0.2.0+:

- Browser integration tests via `@vitest/browser`. Node-mode tests cover the
  type surface, but no test actually boots Pyodide in this release.
- Binary artifact transport (e.g. raw `ArrayBuffer` tensors). v0.1 supports
  text, JSON, and base64-encoded images.
- A "concurrent exec" mode. All current operations serialize through one worker;
  this is documented behavior, not a bug.

### Known limitations

- `Session.canInterrupt === false` in non-COI documents. Browsers throw on
  `new SharedArrayBuffer()` without cross-origin isolation (`COOP: same-origin`
  + `COEP: require-corp`). The library auto-detects and silently falls back
  to terminate-only cancel — no error thrown.
- Pyodide's `FS` is typed `any` by the upstream Pyodide types. The methods we
  call (`writeFile`, `readFile`, `mkdirTree`, `analyzePath`) are real Emscripten
  FS bindings, documented at
  <https://emscripten.org/docs/api_reference/Filesystem-API.html>.

[0.1.0]: https://github.com/unlocalhosted/browsergrad/releases/tag/runtime%40v0.1.0
