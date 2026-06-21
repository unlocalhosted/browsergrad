# Changelog

All notable changes to `@unlocalhosted/browsergrad-runtime`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `browsergrad.oracle(name)` for Python rubrics. It imports a JS module
  registered through `createSession({ jsModules })`, giving assignment rubrics
  a stable BrowserGrad API for profile-declared TS/JS oracles instead of raw
  Pyodide import mechanics.
- Assignment rubric launchers now expose profile execution context in
  environment variables, including `BROWSERGRAD_ASSIGNMENT_ROOT`,
  `BROWSERGRAD_FIXTURES_PATH`, and `BROWSERGRAD_BEHAVIORAL_GATES_JSON`.
- `browsergrad.assignment_context()` parses the launcher-provided assignment
  environment into a plain Python dict for rubrics.
- `runAssignmentRubric(session, plan, contents, options)` orchestrates the
  common assignment path: derive mounts, materialize contents, create the rubric
  exec request, run it, and return both mount and exec results.
- `assignmentRubricKind(plan)` classifies resolved rubric paths as `python`,
  `javascript`, or `unknown` so platforms can route non-Pyodide labs before
  launch.
- `assignmentRunReadiness(plan)` summarizes selected capability modes as
  `runnable`, `simulated`, `external-only`, or `blocked` for platform preflight
  UIs.
- `createAssignmentPreflightReport(profile, environment)` bundles run plan,
  rubric kind, readiness, required capabilities, and mount plan for platform
  preflight panels.
- Capability `any_of` evaluation now prefers the strongest satisfied path by
  mode: browser, then simulated, then external.
- Capability gate evaluations now expose gate-level `status`, `selectedAnyOf`,
  and `selectedCapabilities` for platform preflight routing UI.
- `evaluateAssignmentMountContents(mountPlan, contents)` dry-runs file/dataset
  content readiness before writing to `Session.fs`.
- Assignment mount contents and `Session.fs.write` now accept `Uint8Array`
  bytes for small binary fixtures such as `.pt`, `.npz`, and snapshots.
- `runAssignmentJavascriptRubric(plan, contents, rubric, options)` runs
  browser-native JS rubrics with assignment context, mounted text reads,
  declared oracles, and structured assertion/artifact helpers.
- JavaScript rubric contexts now expose `ctx.readBytes(path)` for binary
  mounted fixtures.
- JavaScript rubric contexts now expose `ctx.substrate(name)` for browser
  resources such as WebGPU adapters/devices supplied by the platform.

## [0.1.1] — 2026-05-25

Bug fix + comprehensive integration tests.

### Fixed

- **`bg.emit_json`, `bg.log`, `bg.emit_image`, and `bg.assert_error` would
  raise `NameError` at call time.** The PY_PREAMBLE deleted internal
  helpers (`_bg_post_artifact`, `_bg_post_assertion`, `_bg_traceback`) from
  module globals at the end of installation, but the user-facing helpers
  resolved those names lazily at call time. After deletion, any call to an
  artifact-emitting function failed.
  Fixed by capturing all needed references in keyword-only default args
  evaluated at function-definition time — the helpers now own their
  references and are immune to module-globals cleanup.
  **This was caught by the new integration test suite. Previously
  undetected.**

### Added

- `tests-integration/python-bridge.test.ts` — 11 tests boot real Pyodide
  in node, install the PY_PREAMBLE, and verify the entire `browsergrad`
  module surface: import, assert_pass/fail/error with full payload shapes,
  log/emit_json/emit_image, plus a hygiene check that no `_bg_*` loader
  locals leak into user globals.
- `tests-integration/client-routing.test.ts` — 12 tests use a FakeWorker
  (implements postMessage / addEventListener / terminate) to exercise
  client.ts's message routing without booting Pyodide:
    - createSession → init:done correlation
    - exec request/response, fs.write + fs.read round-trip
    - onStdout / onAssertion / onArtifact streaming routes correctly
    - Error paths: runtime errors, timeout, AbortSignal, dispose
    - Idempotent dispose, post-dispose operations throw
- `createSession({ worker })` now skips the `typeof Worker === "undefined"`
  check when a custom worker is supplied — non-browser test environments
  can drive the runtime via their own Worker substitute.
- `src/worker/python-preamble.ts` — PY_PREAMBLE extracted from the worker
  entry so it can be imported by tests and any future tooling.

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
