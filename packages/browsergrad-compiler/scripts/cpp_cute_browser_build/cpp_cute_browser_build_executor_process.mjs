import { spawn } from "node:child_process";

const KILL_GRACE_MS = 5_000;
const PROCESS_ERROR = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROCESS";

export class CppCuteBrowserBuildProcessError extends Error {
  /**
   * @param {"cancelled" | "output-limit" | "spawn" | "timeout"} reason
   * @param {string} message
   * @param {ErrorOptions} [options]
   */
  constructor(reason, message, options) {
    super(`${PROCESS_ERROR}-${reason.toUpperCase()}: ${message}`, options);
    this.name = "CppCuteBrowserBuildProcessError";
    this.reason = reason;
  }
}

/**
 * Internal process effect boundary. Production callers cannot replace it;
 * executor tests mock this module, while process-boundary tests exercise the
 * real no-shell implementation.
 */
export const CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS = Object.freeze({
  run: runProcessStep,
});

/**
 * @param {Readonly<{
 *   executable: string;
 *   arguments: readonly string[];
 *   cwd: string;
 *   environment: Readonly<Record<string, string>>;
 *   maximumOutputByteLength: number;
 *   maximumDurationMs: number;
 *   signal?: AbortSignal;
 * }>} input
 * @returns {Promise<Readonly<{
 *   exitCode: number | null;
 *   terminationSignal: NodeJS.Signals | null;
 *   stdout: Uint8Array;
 *   stderr: Uint8Array;
 * }>>}
 */
async function runProcessStep(input) {
  if (input.signal?.aborted === true) {
    throw new CppCuteBrowserBuildProcessError("cancelled", "build step was cancelled before spawn");
  }
  if (!Number.isSafeInteger(input.maximumOutputByteLength) ||
      input.maximumOutputByteLength <= 0) {
    throw new TypeError("maximumOutputByteLength must be a positive safe integer");
  }
  if (!Number.isSafeInteger(input.maximumDurationMs) || input.maximumDurationMs <= 0) {
    throw new TypeError("maximumDurationMs must be a positive safe integer");
  }

  return await new Promise((resolve, reject) => {
    /** @type {Buffer[]} */
    const stdoutChunks = [];
    /** @type {Buffer[]} */
    const stderrChunks = [];
    let stdoutByteLength = 0;
    let stderrByteLength = 0;
    /** @type {"cancelled" | "output-limit" | "timeout" | undefined} */
    let terminationReason;
    /** @type {Error | undefined} */
    let spawnFailure;
    /** @type {ReturnType<typeof setTimeout> | undefined} */
    let killTimer;

    const child = spawn(input.executable, [...input.arguments], {
      cwd: input.cwd,
      env: { ...input.environment },
      detached: true,
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });

    /** @param {NodeJS.Signals} signal */
    const killGroup = (signal) => {
      if (child.pid === undefined) return;
      try {
        process.kill(-child.pid, signal);
      } catch (cause) {
        if (!isNodeError(cause, "ESRCH")) {
          try {
            child.kill(signal);
          } catch {
            // The close/error event remains the single settlement boundary.
          }
        }
      }
    };

    /** @param {"cancelled" | "output-limit" | "timeout"} reason */
    const terminate = (reason) => {
      if (terminationReason !== undefined) return;
      terminationReason = reason;
      killGroup("SIGTERM");
      killTimer = setTimeout(() => killGroup("SIGKILL"), KILL_GRACE_MS);
      killTimer.unref();
    };

    const onAbort = () => terminate("cancelled");
    input.signal?.addEventListener("abort", onAbort, { once: true });

    const timeout = setTimeout(
      () => terminate("timeout"),
      input.maximumDurationMs,
    );
    timeout.unref();

    /** @param {Buffer[]} chunks @param {number} currentByteLength @param {Uint8Array} chunk */
    const capture = (chunks, currentByteLength, chunk) => {
      if (terminationReason !== undefined) return currentByteLength;
      const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      const nextByteLength = currentByteLength + bytes.byteLength;
      if (nextByteLength > input.maximumOutputByteLength) {
        terminate("output-limit");
        return nextByteLength;
      }
      chunks.push(bytes);
      return nextByteLength;
    };

    child.stdout.on("data", (chunk) => {
      stdoutByteLength = capture(stdoutChunks, stdoutByteLength, chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderrByteLength = capture(stderrChunks, stderrByteLength, chunk);
    });
    child.once("error", (cause) => {
      spawnFailure = cause;
    });
    child.once("close", (exitCode, terminationSignal) => {
      clearTimeout(timeout);
      if (killTimer !== undefined) clearTimeout(killTimer);
      input.signal?.removeEventListener("abort", onAbort);

      if (spawnFailure !== undefined) {
        reject(new CppCuteBrowserBuildProcessError(
          "spawn",
          "failed to spawn the exact build step",
          { cause: spawnFailure },
        ));
        return;
      }
      if (terminationReason !== undefined) {
        const messages = {
          cancelled: "build step was cancelled",
          "output-limit": `build step output exceeded ${input.maximumOutputByteLength} bytes per stream`,
          timeout: `build step exceeded ${input.maximumDurationMs} milliseconds`,
        };
        reject(new CppCuteBrowserBuildProcessError(
          terminationReason,
          messages[terminationReason],
        ));
        return;
      }
      resolve(Object.freeze({
        exitCode,
        terminationSignal,
        stdout: new Uint8Array(Buffer.concat(stdoutChunks, stdoutByteLength)),
        stderr: new Uint8Array(Buffer.concat(stderrChunks, stderrByteLength)),
      }));
    });
  });
}

/** @param {unknown} value @param {string} code */
function isNodeError(value, code) {
  return value instanceof Error &&
    "code" in value &&
    value.code === code;
}
