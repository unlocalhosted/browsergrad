/**
 * Public types for @unlocalhosted/browsergrad-runtime.
 *
 * Stability contract: every field here is part of the public API.
 * - Adding a new optional field is non-breaking (minor bump).
 * - Removing a field, narrowing a type, or making a field required is breaking (major bump).
 * - Anything not exported from `./index.ts` is private and may change at any time.
 */

/* ────────────────────────────────────────────────────────────
 * Session construction
 * ──────────────────────────────────────────────────────────── */

export interface SessionOptions {
  /**
   * URL prefix where Pyodide runtime assets are served from.
   * Must point at a directory containing `pyodide.mjs`, `pyodide.asm.wasm`,
   * `python_stdlib.zip`, and `pyodide-lock.json` (all served from the same origin
   * as the calling page — required by COEP and recommended by privacy policy).
   *
   * Example: `"/pyodide/v0.26.4/"`
   *
   * No default. Set to a same-origin path you control; we do not bake in a CDN.
   */
  pyodideIndexURL: string;

  /**
   * Python packages to load eagerly during `createSession`.
   * These are loaded via Pyodide's `loadPackage()` before the session resolves.
   * Anything not listed here can still be loaded later via `micropip` from user code.
   */
  packages?: readonly string[];

  /**
   * JS modules to import inside the Pyodide worker and register via
   * `pyodide.registerJsModule()`. Use this for small browser-safe oracle
   * helpers that Python rubrics call through Pyodide's JS bridge.
   *
   * `importURL` must be a URL the worker can dynamically import. If
   * `exportName` is omitted, the full module namespace object is registered.
   * If it is set, that named export must be an object.
   */
  jsModules?: readonly PyodideJsModule[];

  /** Called as each preload package transitions states. Useful for boot UI. */
  onPackageProgress?: (event: PackageProgressEvent) => void;

  /**
   * Disable cooperative cancellation via SharedArrayBuffer.
   * When disabled or unavailable, `AbortSignal` and `timeoutMs` terminate the
   * worker (hard cancel). When enabled (default if available), they first try
   * to raise a Python `KeyboardInterrupt` and only terminate as a safety net
   * after a short grace period.
   *
   * Requires the document to be cross-origin isolated (`COOP: same-origin`,
   * `COEP: require-corp`). The library auto-detects via `crossOriginIsolated`
   * and silently falls back to terminate-only mode if not available.
   */
  disableInterruptBuffer?: boolean;

  /**
   * Override the Worker construction. Defaults to a bundler-detected URL
   * (`new URL("./worker/index.js", import.meta.url)`), which works in Vite,
   * Rollup ≥ 3, webpack ≥ 5, Parcel ≥ 2, and esbuild.
   *
   * Provide your own Worker (constructed with `{ type: "module" }`) if your
   * bundler doesn't support that pattern. The library will postMessage to it
   * using the wire protocol; you don't need to know the protocol details.
   */
  worker?: Worker;
}

export interface PackageProgressEvent {
  readonly package: string;
  readonly status: "loading" | "loaded" | "failed";
  /** Error message if `status === "failed"`. */
  readonly message?: string;
}

export interface PyodideJsModule {
  /** Python import name exposed through Pyodide, e.g. `_bg_tokenizers`. */
  readonly name: string;
  /** Worker-importable module URL. */
  readonly importURL: string;
  /** Optional named export to register instead of the full module namespace. */
  readonly exportName?: string;
}

/* ────────────────────────────────────────────────────────────
 * Session — the running Pyodide worker
 * ──────────────────────────────────────────────────────────── */

export interface Session {
  /** Virtual filesystem mounted inside the Pyodide worker. */
  readonly fs: SessionFS;

  /**
   * `true` if cooperative cancellation is available for this session
   * (SharedArrayBuffer + crossOriginIsolated, not disabled by options).
   * When `false`, `interrupt()` is a no-op and `signal`/`timeoutMs` use hard termination.
   */
  readonly canInterrupt: boolean;

  /**
   * Execute Python code in this session's persistent global namespace.
   * Variables defined by one `exec` are visible to the next — like a Jupyter kernel.
   * Call {@link clearNamespace} to reset.
   */
  exec(options: ExecOptions): Promise<ExecResult>;

  /**
   * Request cooperative cancellation of the in-flight `exec`, if any.
   * Writes SIGINT to the interrupt buffer so Python raises `KeyboardInterrupt`.
   * No-op if `canInterrupt` is `false` or no exec is running.
   *
   * For hard cancel, use AbortSignal on `exec` or call `dispose()`.
   */
  interrupt(): void;

  /**
   * Reset the Python global namespace.
   * Preloaded packages stay loaded; only user-defined names are cleared.
   */
  clearNamespace(): Promise<void>;

  /**
   * Terminate the worker and release resources.
   * After dispose, all methods on this Session reject with an error.
   * Idempotent.
   */
  dispose(): Promise<void>;
}

/* ────────────────────────────────────────────────────────────
 * Filesystem (mounted inside the Pyodide worker via MEMFS)
 * ──────────────────────────────────────────────────────────── */

export interface SessionFS {
  /**
   * Write a file to the worker's virtual FS.
   * Parent directories are created automatically.
   * Strings are written as UTF-8; Uint8Array values are written as bytes.
   */
  write(path: string, content: string | Uint8Array): Promise<void>;

  /** Read a UTF-8 file from the worker's virtual FS. */
  read(path: string): Promise<string>;

  /** Read raw bytes from the worker's virtual FS. */
  readBytes(path: string): Promise<Uint8Array>;
}

/* ────────────────────────────────────────────────────────────
 * Exec API
 * ──────────────────────────────────────────────────────────── */

export interface ExecOptions {
  /** Python source to execute. */
  code: string;

  /**
   * Wall-clock timeout in milliseconds.
   * Uses cooperative cancel if available, falls back to worker termination.
   * Default: no timeout. Set this for any user-provided code.
   */
  timeoutMs?: number;

  /**
   * Standard `AbortSignal`. Aborting attempts a cooperative cancel via
   * the interrupt buffer (if available), then terminates the worker as
   * a safety net.
   */
  signal?: AbortSignal;

  /** Called with each chunk of stdout as it arrives (line-buffered by Pyodide). */
  onStdout?: (chunk: string) => void;

  /** Called with each chunk of stderr as it arrives (line-buffered by Pyodide). */
  onStderr?: (chunk: string) => void;

  /**
   * Called for each `browsergrad.assert_*` call inside the executing code.
   * The library does not interpret assertions — it just relays them.
   */
  onAssertion?: (assertion: Assertion) => void;

  /**
   * Called for each `browsergrad.log` / `emit_*` call inside the executing code.
   * The library does not interpret artifacts — it just relays them.
   */
  onArtifact?: (artifact: Artifact) => void;
}

export interface ExecResult {
  /** `true` iff the code ran to completion without an uncaught exception. */
  readonly ok: boolean;

  /** Concatenated stdout from this exec (also streamed via `onStdout`). */
  readonly stdout: string;

  /** Concatenated stderr from this exec (also streamed via `onStderr`). */
  readonly stderr: string;

  /**
   * All assertions emitted during this exec, in arrival order.
   * Also streamed via `onAssertion`. Empty if `browsergrad` wasn't used.
   */
  readonly assertions: readonly Assertion[];

  /**
   * All artifacts emitted during this exec, in arrival order.
   * Also streamed via `onArtifact`. Empty if `browsergrad` wasn't used.
   */
  readonly artifacts: readonly Artifact[];

  /** Non-null when `ok === false`. */
  readonly error: ExecError | null;

  /** Wall-clock time spent in the worker for this exec, in milliseconds. */
  readonly durationMs: number;
}

export type ExecErrorKind =
  | "syntax"        // SyntaxError before execution
  | "runtime"       // Uncaught Python exception
  | "timeout"       // timeoutMs exceeded
  | "aborted"       // AbortSignal triggered
  | "interrupted"   // KeyboardInterrupt from cooperative cancel
  | "worker-crash"; // Worker died unexpectedly

export interface ExecError {
  readonly kind: ExecErrorKind;
  readonly message: string;
  /** Python traceback if available. */
  readonly traceback?: string;
}

/* ────────────────────────────────────────────────────────────
 * Structured assertion + artifact protocols
 *
 * Emitted from Python via:
 *   import browsergrad as bg
 *   bg.assert_pass("test_name")
 *   bg.assert_fail("test_name", "msg", expected=3, actual=2)
 *   bg.log("status", "training complete", level="info")
 *   bg.emit_json("loss_curve", {"steps": [...], "loss": [...]})
 *   bg.emit_image("filter_0", "image/png", base64_encoded_png)
 *
 * Adding a new variant to either union is non-breaking.
 * ──────────────────────────────────────────────────────────── */

export type Assertion =
  | AssertionPass
  | AssertionFail
  | AssertionError;

export interface AssertionPass {
  readonly kind: "pass";
  readonly name: string;
  readonly durationMs?: number;
}

export interface AssertionFail {
  readonly kind: "fail";
  readonly name: string;
  readonly message: string;
  /** `repr(expected_value)` from Python — string form for safe transport. */
  readonly expectedRepr?: string;
  /** `repr(actual_value)` from Python. */
  readonly actualRepr?: string;
  readonly durationMs?: number;
}

/**
 * Trailing underscore avoids shadowing the global `Error` constructor inside
 * this module. Public consumers import this as `AssertionError`.
 */
export interface AssertionError_ {
  readonly kind: "error";
  readonly name: string;
  readonly message: string;
  readonly traceback?: string;
  readonly durationMs?: number;
}

export type AssertionError = AssertionError_;

export type Artifact =
  | ArtifactLog
  | ArtifactJson
  | ArtifactImage;

export interface ArtifactLog {
  readonly kind: "log";
  readonly name: string;
  readonly level?: "info" | "warn" | "error";
  readonly data: string;
}

export interface ArtifactJson {
  readonly kind: "json";
  readonly name: string;
  readonly data: unknown;
}

export interface ArtifactImage {
  readonly kind: "image";
  readonly name: string;
  readonly mime: string;
  readonly dataBase64: string;
}

/* ────────────────────────────────────────────────────────────
 * Errors thrown by the library itself (not Python-level errors).
 * Python-level errors arrive via ExecResult.error.
 * ──────────────────────────────────────────────────────────── */

/**
 * Thrown when the library is used incorrectly (bad arguments, post-dispose usage,
 * etc). Not thrown for Python-level failures — those come back as `ExecResult.error`.
 */
export class BrowsergradError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BrowsergradError";
  }
}
