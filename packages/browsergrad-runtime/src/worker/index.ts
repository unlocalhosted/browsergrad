/// <reference lib="webworker" />

/**
 * browsergrad worker entry.
 *
 * Boots Pyodide on first `init`, then services `fs.*`, `exec`, and
 * `clearNamespace` requests over the message protocol defined in
 * `../protocol.ts`.
 *
 * Stdout/stderr stream as `exec:stdout` / `exec:stderr` events with
 * `id` correlated to the originating ExecRequest. Structured assertions
 * and artifacts emitted from Python (via `import browsergrad`) stream as
 * `exec:assertion` / `exec:artifact` events. The final `exec:done`
 * includes the full concatenated stdout/stderr for callers that ignore streaming.
 *
 * Cancellation: if the InitRequest carries an `interruptBuffer`, Pyodide
 * is configured for cooperative cancel — the main thread writes `2` (SIGINT)
 * and Python raises `KeyboardInterrupt` on the next bytecode boundary.
 * If no buffer is provided, cancellation is by worker termination only.
 */

import type {
  ClearNamespaceRequest,
  ClientToWorker,
  ExecRequest,
  FsReadRequest,
  FsWriteRequest,
  InitRequest,
  WorkerToClient,
} from "../protocol.js";
import type {
  Artifact,
  Assertion,
  ExecError,
  PackageProgressEvent,
  PyodideJsModule,
} from "../types.js";
import { PY_PREAMBLE } from "./python-preamble.js";

const ctx = self as unknown as DedicatedWorkerGlobalScope;

/* ────────────────────────────────────────────────────────────
 * Pyodide loading
 * ──────────────────────────────────────────────────────────── */

interface PyProxy {
  destroy: () => void;
  (...args: unknown[]): unknown;
}

interface PyodideAPI {
  runPythonAsync: (code: string) => Promise<unknown>;
  loadPackage: (
    names: readonly string[],
    options?: { messageCallback?: (msg: string) => void },
  ) => Promise<void>;
  setStdout: (handlers: { batched?: (s: string) => void }) => void;
  setStderr: (handlers: { batched?: (s: string) => void }) => void;
  setInterruptBuffer: (buffer: Uint8Array) => void;
  registerJsModule: (name: string, module: object) => void;
  globals: { get: (name: string) => PyProxy };
  FS: {
    writeFile: (path: string, data: string, opts?: { encoding?: string }) => void;
    readFile: (path: string, opts?: { encoding?: string }) => string;
    mkdirTree: (path: string) => void;
    analyzePath: (path: string) => { exists: boolean };
  };
}

let pyodidePromise: Promise<PyodideAPI> | null = null;
let currentExecId: number | null = null;

async function bootPyodide(
  indexURL: string,
  packages: readonly string[],
  jsModules: readonly PyodideJsModule[],
  interruptBuffer: Uint8Array | undefined,
  emit: (event: PackageProgressEvent) => void,
): Promise<PyodideAPI> {
  if (pyodidePromise) return pyodidePromise;
  pyodidePromise = (async () => {
    const mod = (await import(/* @vite-ignore */ `${indexURL}pyodide.mjs`)) as {
      loadPyodide: (opts: { indexURL: string }) => Promise<PyodideAPI>;
    };
    const py = await mod.loadPyodide({ indexURL });

    if (interruptBuffer) {
      py.setInterruptBuffer(interruptBuffer);
    }

    py.registerJsModule("_bg_native", {
      postAssertion: (json: string) => {
        if (currentExecId === null) return;
        let assertion: Assertion;
        try {
          assertion = JSON.parse(json) as Assertion;
        } catch {
          return;
        }
        reply({ id: currentExecId, kind: "exec:assertion", assertion });
      },
      postArtifact: (json: string) => {
        if (currentExecId === null) return;
        let artifact: Artifact;
        try {
          artifact = JSON.parse(json) as Artifact;
        } catch {
          return;
        }
        reply({ id: currentExecId, kind: "exec:artifact", artifact });
      },
    });

    await registerJsModules(py, jsModules);

    for (const pkg of packages) {
      emit({ package: pkg, status: "loading" });
      try {
        await py.loadPackage([pkg]);
        emit({ package: pkg, status: "loaded" });
      } catch (err) {
        emit({
          package: pkg,
          status: "failed",
          message: err instanceof Error ? err.message : String(err),
        });
        throw err;
      }
    }

    await py.runPythonAsync(PY_PREAMBLE);
    return py;
  })();
  return pyodidePromise;
}

async function registerJsModules(
  py: PyodideAPI,
  jsModules: readonly PyodideJsModule[],
): Promise<void> {
  for (const spec of jsModules) {
    const imported = (await import(/* @vite-ignore */ spec.importURL)) as Record<
      string,
      unknown
    >;
    const value =
      spec.exportName === undefined ? imported : imported[spec.exportName];
    if (typeof value !== "object" || value === null) {
      const suffix =
        spec.exportName === undefined ? "" : ` export ${spec.exportName}`;
      throw new Error(
        `JS module ${spec.name}${suffix} must resolve to an object for Pyodide registration`,
      );
    }
    py.registerJsModule(spec.name, value);
  }
}

/* ────────────────────────────────────────────────────────────
 * Message reply helper
 * ──────────────────────────────────────────────────────────── */

function reply(message: WorkerToClient): void {
  // Worker.postMessage is unary; targetOrigin doesn't apply here.
  // oxlint-disable-next-line require-post-message-target-origin
  ctx.postMessage(message);
}

function replyError(id: number, err: unknown): void {
  reply({
    id,
    kind: "error",
    message: err instanceof Error ? err.message : String(err),
  });
}

/* ────────────────────────────────────────────────────────────
 * Request handlers
 * ──────────────────────────────────────────────────────────── */

async function handleInit(req: InitRequest): Promise<void> {
  try {
    await bootPyodide(
      req.pyodideIndexURL,
      req.packages,
      req.jsModules,
      req.interruptBuffer,
      (event) => {
        reply({ id: 0, kind: "init:progress", event });
      },
    );
    reply({ id: req.id, kind: "init:done" });
  } catch (err) {
    replyError(req.id, err);
  }
}

async function handleFsWrite(req: FsWriteRequest): Promise<void> {
  try {
    const py = await ensureReady(req.id);
    if (!py) return;
    const parent = req.path.substring(0, req.path.lastIndexOf("/"));
    if (parent && !py.FS.analyzePath(parent).exists) {
      py.FS.mkdirTree(parent);
    }
    py.FS.writeFile(req.path, req.content, { encoding: "utf8" });
    reply({ id: req.id, kind: "fs.write:done" });
  } catch (err) {
    replyError(req.id, err);
  }
}

async function handleFsRead(req: FsReadRequest): Promise<void> {
  try {
    const py = await ensureReady(req.id);
    if (!py) return;
    const content = py.FS.readFile(req.path, { encoding: "utf8" });
    reply({ id: req.id, kind: "fs.read:done", content });
  } catch (err) {
    replyError(req.id, err);
  }
}

async function handleClearNamespace(req: ClearNamespaceRequest): Promise<void> {
  try {
    const py = await ensureReady(req.id);
    if (!py) return;
    // Drop everything that isn't a dunder, a preloaded module, or `browsergrad`.
    await py.runPythonAsync(`
import sys as _sys
_preserve = {"__name__", "__doc__", "__package__", "__loader__", "__spec__",
             "__annotations__", "__builtins__"}
for _k in list(globals().keys()):
    if _k in _preserve or _k.startswith("__"):
        continue
    _v = globals().get(_k)
    if isinstance(_v, type(_sys)):  # module
        continue
    del globals()[_k]
del _sys
`);
    reply({ id: req.id, kind: "clearNamespace:done" });
  } catch (err) {
    replyError(req.id, err);
  }
}

async function handleExec(req: ExecRequest): Promise<void> {
  const t0 = performance.now();
  let stdoutAll = "";
  let stderrAll = "";

  try {
    const py = await ensureReady(req.id);
    if (!py) return;

    currentExecId = req.id;

    py.setStdout({
      batched: (chunk) => {
        stdoutAll += chunk;
        reply({ id: req.id, kind: "exec:stdout", chunk });
      },
    });
    py.setStderr({
      batched: (chunk) => {
        stderrAll += chunk;
        reply({ id: req.id, kind: "exec:stderr", chunk });
      },
    });

    let error: ExecError | null = null;
    try {
      await py.runPythonAsync(req.code);
    } catch (err) {
      error = classifyPythonError(err);
    }

    reply({
      id: req.id,
      kind: "exec:done",
      ok: error === null,
      durationMs: performance.now() - t0,
      stdout: stdoutAll,
      stderr: stderrAll,
      error,
    });
  } catch (err) {
    // Worker-internal failure (not a Python error). Treat as crash.
    reply({
      id: req.id,
      kind: "exec:done",
      ok: false,
      durationMs: performance.now() - t0,
      stdout: stdoutAll,
      stderr: stderrAll,
      error: {
        kind: "worker-crash",
        message: err instanceof Error ? err.message : String(err),
      },
    });
  } finally {
    currentExecId = null;
    // Reset stdout/stderr so subsequent execs without listeners don't leak.
    pyodidePromise
      ?.then((py) => {
        py.setStdout({});
        py.setStderr({});
      })
      .catch(() => {});
  }
}

function classifyPythonError(err: unknown): ExecError {
  const message = err instanceof Error ? err.message : String(err);
  const isSyntax = /^\s*SyntaxError|^\s*IndentationError/m.test(message);
  const isKeyboardInterrupt = /KeyboardInterrupt/.test(message);
  let kind: ExecError["kind"] = "runtime";
  if (isSyntax) kind = "syntax";
  else if (isKeyboardInterrupt) kind = "interrupted";
  return {
    kind,
    message: extractFinalErrorLine(message),
    traceback: message,
  };
}

function extractFinalErrorLine(traceback: string): string {
  const lines = traceback.trimEnd().split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i];
    if (line && line.trim() && !line.startsWith(" ")) return line.trim();
  }
  return traceback;
}

async function ensureReady(id: number): Promise<PyodideAPI | null> {
  if (!pyodidePromise) {
    replyError(id, new Error("Session not initialized — call init first"));
    return null;
  }
  try {
    return await pyodidePromise;
  } catch (err) {
    replyError(id, err);
    return null;
  }
}

/* ────────────────────────────────────────────────────────────
 * Dispatcher
 * ──────────────────────────────────────────────────────────── */

ctx.addEventListener("message", (event: MessageEvent<ClientToWorker>) => {
  const req = event.data;
  switch (req.kind) {
    case "init":
      void handleInit(req);
      return;
    case "fs.write":
      void handleFsWrite(req);
      return;
    case "fs.read":
      void handleFsRead(req);
      return;
    case "exec":
      void handleExec(req);
      return;
    case "clearNamespace":
      void handleClearNamespace(req);
      return;
  }
});
