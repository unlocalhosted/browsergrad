/**
 * Main-thread client for the browsergrad worker.
 *
 * Spawns (or accepts) the worker, owns the request bookkeeping (id → resolver),
 * and implements the public Session interface from `./types.ts`.
 *
 * Worker URL construction uses `new URL("./worker/index.js", import.meta.url)`
 * which modern bundlers (Vite, Rollup ≥ 3, webpack ≥ 5, Parcel ≥ 2) detect and
 * emit as a separate worker chunk. Consumers using older bundlers can pass
 * their own `Worker` via `SessionOptions.worker`.
 *
 * Cooperative cancellation: when `SharedArrayBuffer` is available and the
 * document is cross-origin isolated, the client allocates a 1-byte SAB-backed
 * Uint8Array and sends it to the worker during `init`. Writing `2` (SIGINT)
 * into byte 0 triggers `KeyboardInterrupt` inside Python on the next bytecode
 * boundary. The client tries this before falling back to `worker.terminate()`.
 */

import type {
  ClientToWorker,
  ExecDoneResponse,
  WorkerToClient,
} from "./protocol.js";
import type {
  Artifact,
  Assertion,
  ExecOptions,
  ExecResult,
  PackageProgressEvent,
  Session,
  SessionFS,
  SessionOptions,
} from "./types.js";
import { BrowsergradError } from "./types.js";

/**
 * Distributive Omit — preserves the discriminated-union shape.
 * Plain `Omit<A | B, K>` collapses to the keys common to all variants, which
 * is wrong for tagged unions like ClientToWorker.
 */
type ClientRequestPayload = ClientToWorker extends infer T
  ? T extends ClientToWorker
    ? Omit<T, "id">
    : never
  : never;

interface PendingRequest {
  readonly kind: string;
  readonly resolve: (value: WorkerToClient) => void;
  readonly reject: (err: Error) => void;
  /** Per-exec handlers, routing streaming events to the right caller. */
  readonly onStdout?: (chunk: string) => void;
  readonly onStderr?: (chunk: string) => void;
  readonly onAssertion?: (assertion: Assertion) => void;
  readonly onArtifact?: (artifact: Artifact) => void;
  /** Accumulators for the per-exec result, populated as events arrive. */
  readonly assertions?: Assertion[];
  readonly artifacts?: Artifact[];
}

/** Time we wait for cooperative cancel to land before terminating. */
const COOPERATIVE_CANCEL_GRACE_MS = 500;

function isInterruptBufferSupported(): boolean {
  // SharedArrayBuffer requires cross-origin isolation. In non-isolated contexts
  // SharedArrayBuffer is either undefined or its constructor throws on use.
  if (typeof SharedArrayBuffer === "undefined") return false;
  if (
    typeof globalThis !== "undefined" &&
    "crossOriginIsolated" in globalThis &&
    globalThis.crossOriginIsolated === false
  ) {
    return false;
  }
  return true;
}

class SessionImpl implements Session {
  readonly fs: SessionFS;
  readonly canInterrupt: boolean;

  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, PendingRequest>();
  private disposed = false;
  private interruptBuffer: Uint8Array | null;
  private onInitProgress: ((event: PackageProgressEvent) => void) | undefined;

  constructor(worker: Worker, interruptBuffer: Uint8Array | null) {
    this.worker = worker;
    this.interruptBuffer = interruptBuffer;
    this.canInterrupt = interruptBuffer !== null;
    this.fs = {
      write: (path, content) => this.fsWrite(path, content),
      read: (path) => this.fsRead(path),
    };
    this.worker.addEventListener("message", this.onMessage);
    this.worker.addEventListener("error", this.onWorkerError);
  }

  /* ── Lifecycle ─────────────────────────────────────────── */

  async init(opts: SessionOptions): Promise<void> {
    this.onInitProgress = opts.onPackageProgress;
    const initPayload: ClientRequestPayload = {
      kind: "init",
      pyodideIndexURL: opts.pyodideIndexURL,
      packages: opts.packages ?? [],
      jsModules: opts.jsModules ?? [],
      ...(this.interruptBuffer ? { interruptBuffer: this.interruptBuffer } : {}),
    };
    await this.request(initPayload);
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    this.worker.removeEventListener("message", this.onMessage);
    this.worker.removeEventListener("error", this.onWorkerError);
    this.worker.terminate();
    const inflight = [...this.pending.values()];
    this.pending.clear();
    for (const p of inflight) {
      p.reject(new BrowsergradError("Session disposed"));
    }
  }

  /* ── Public Session methods ────────────────────────────── */

  interrupt(): void {
    if (this.disposed) return;
    if (!this.interruptBuffer) return;
    this.interruptBuffer[0] = 2; // SIGINT — Pyodide raises KeyboardInterrupt
  }

  async exec(opts: ExecOptions): Promise<ExecResult> {
    this.assertLive();
    const id = this.nextId++;
    const t0 = performance.now();
    const assertions: Assertion[] = [];
    const artifacts: Artifact[] = [];

    const donePromise = new Promise<ExecDoneResponse>((resolve, reject) => {
      this.pending.set(id, {
        kind: "exec",
        resolve: (msg) => {
          if (msg.kind === "exec:done") resolve(msg);
          else reject(new BrowsergradError(`unexpected reply: ${msg.kind}`));
        },
        reject,
        ...(opts.onStdout ? { onStdout: opts.onStdout } : {}),
        ...(opts.onStderr ? { onStderr: opts.onStderr } : {}),
        ...(opts.onAssertion ? { onAssertion: opts.onAssertion } : {}),
        ...(opts.onArtifact ? { onArtifact: opts.onArtifact } : {}),
        assertions,
        artifacts,
      });
    });

    /* AbortSignal + timeout handling.
     * Both first attempt cooperative cancel via the interrupt buffer, then
     * fall back to disposing the worker after a grace period. The exec:done
     * (with KeyboardInterrupt error) arrives via the normal channel.
     */
    let cancelReason: "aborted" | "timeout" | null = null;
    const cancelHandle = (reason: "aborted" | "timeout"): void => {
      if (cancelReason !== null) return;
      cancelReason = reason;
      if (this.interruptBuffer) {
        this.interruptBuffer[0] = 2;
        // Safety net — if Python ignores SIGINT, terminate
        setTimeout(() => {
          if (this.pending.has(id)) void this.dispose();
        }, COOPERATIVE_CANCEL_GRACE_MS);
      } else {
        void this.dispose();
      }
    };

    let removeAbortListener: (() => void) | null = null;
    if (opts.signal) {
      if (opts.signal.aborted) {
        cancelHandle("aborted");
      } else {
        const handler = (): void => cancelHandle("aborted");
        opts.signal.addEventListener("abort", handler, { once: true });
        removeAbortListener = () =>
          opts.signal?.removeEventListener("abort", handler);
      }
    }

    let timeoutHandle: ReturnType<typeof setTimeout> | null = null;
    if (typeof opts.timeoutMs === "number" && opts.timeoutMs > 0) {
      timeoutHandle = setTimeout(() => cancelHandle("timeout"), opts.timeoutMs);
    }

    this.post({ id, kind: "exec", code: opts.code });

    let done: ExecDoneResponse;
    try {
      done = await donePromise;
    } catch (err) {
      // Either disposed mid-exec (cooperative cancel grace expired, or signal
      // aborted with no interrupt buffer) — synthesize an ExecResult so the
      // caller has a uniform shape.
      const kind =
        cancelReason === "timeout"
          ? "timeout"
          : cancelReason === "aborted"
            ? "aborted"
            : "worker-crash";
      return {
        ok: false,
        stdout: "",
        stderr: "",
        assertions,
        artifacts,
        error: {
          kind,
          message:
            err instanceof Error ? err.message : "exec failed before completion",
        },
        durationMs: performance.now() - t0,
      };
    } finally {
      if (timeoutHandle !== null) clearTimeout(timeoutHandle);
      if (removeAbortListener) removeAbortListener();
    }

    // If we initiated a cancel and Python honored it, the error came back as
    // "interrupted". Re-label according to the cancel reason so the API stays clean.
    let error = done.error;
    if (error && error.kind === "interrupted") {
      if (cancelReason === "aborted") {
        error = { ...error, kind: "aborted" };
      } else if (cancelReason === "timeout") {
        error = { ...error, kind: "timeout" };
      }
    }

    return {
      ok: done.ok,
      stdout: done.stdout,
      stderr: done.stderr,
      assertions,
      artifacts,
      error,
      durationMs: done.durationMs,
    };
  }

  async clearNamespace(): Promise<void> {
    this.assertLive();
    await this.request({ kind: "clearNamespace" });
  }

  /* ── FS impl ───────────────────────────────────────────── */

  private async fsWrite(path: string, content: string): Promise<void> {
    this.assertLive();
    await this.request({ kind: "fs.write", path, content });
  }

  private async fsRead(path: string): Promise<string> {
    this.assertLive();
    const reply = await this.request({ kind: "fs.read", path });
    if (reply.kind !== "fs.read:done") {
      throw new BrowsergradError(`unexpected fs.read reply: ${reply.kind}`);
    }
    return reply.content;
  }

  /* ── RPC primitives ────────────────────────────────────── */

  private request(payload: ClientRequestPayload): Promise<WorkerToClient> {
    const id = this.nextId++;
    return new Promise<WorkerToClient>((resolve, reject) => {
      this.pending.set(id, { kind: payload.kind, resolve, reject });
      this.post({ ...payload, id } as ClientToWorker);
    });
  }

  private post(msg: ClientToWorker): void {
    // Worker.postMessage is unary; targetOrigin doesn't apply.
    // oxlint-disable-next-line require-post-message-target-origin
    this.worker.postMessage(msg);
  }

  private onMessage = (e: MessageEvent<WorkerToClient>): void => {
    const msg = e.data;
    if (msg.id === 0 && msg.kind === "init:progress") {
      this.onInitProgress?.(msg.event);
      return;
    }
    // Streaming events for an exec — route to that exec's handlers, don't resolve.
    if (
      msg.kind === "exec:stdout" ||
      msg.kind === "exec:stderr" ||
      msg.kind === "exec:assertion" ||
      msg.kind === "exec:artifact"
    ) {
      const slot = this.pending.get(msg.id);
      if (!slot) return;
      switch (msg.kind) {
        case "exec:stdout":
          slot.onStdout?.(msg.chunk);
          return;
        case "exec:stderr":
          slot.onStderr?.(msg.chunk);
          return;
        case "exec:assertion":
          slot.assertions?.push(msg.assertion);
          slot.onAssertion?.(msg.assertion);
          return;
        case "exec:artifact":
          slot.artifacts?.push(msg.artifact);
          slot.onArtifact?.(msg.artifact);
          return;
      }
    }
    // Terminal response — resolve and remove.
    const slot = this.pending.get(msg.id);
    if (!slot) return;
    this.pending.delete(msg.id);
    if (msg.kind === "error") {
      slot.reject(new BrowsergradError(msg.message));
      return;
    }
    slot.resolve(msg);
  };

  private onWorkerError = (e: ErrorEvent): void => {
    const err = new BrowsergradError(e.message || "worker crashed");
    const inflight = [...this.pending.values()];
    this.pending.clear();
    for (const slot of inflight) slot.reject(err);
  };

  private assertLive(): void {
    if (this.disposed) {
      throw new BrowsergradError("Session has been disposed");
    }
  }
}

/* ────────────────────────────────────────────────────────────
 * Factory
 * ──────────────────────────────────────────────────────────── */

function createDefaultWorker(): Worker {
  // Bundler-detected URL — Vite, Rollup ≥ 3, webpack ≥ 5, Parcel ≥ 2, esbuild
  // all emit a separate worker chunk for this pattern.
  const workerUrl = new URL("./worker/index.js", import.meta.url);
  return new Worker(workerUrl, { type: "module" });
}

export async function createSession(
  options: SessionOptions,
): Promise<Session> {
  if (!options.pyodideIndexURL) {
    throw new BrowsergradError("createSession: pyodideIndexURL is required");
  }
  // Only require the global Worker constructor when we have to build a default
  // worker. Callers providing their own (testing, custom bundlers) bypass.
  if (!options.worker && typeof Worker === "undefined") {
    throw new BrowsergradError(
      "createSession: Worker is not available in this environment",
    );
  }

  let interruptBuffer: Uint8Array | null = null;
  if (
    !options.disableInterruptBuffer &&
    isInterruptBufferSupported()
  ) {
    interruptBuffer = new Uint8Array(new SharedArrayBuffer(1));
  }

  const worker = options.worker ?? createDefaultWorker();
  const session = new SessionImpl(worker, interruptBuffer);
  try {
    await session.init(options);
  } catch (err) {
    await session.dispose();
    throw err;
  }
  return session;
}
