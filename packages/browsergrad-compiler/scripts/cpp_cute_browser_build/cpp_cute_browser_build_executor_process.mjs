import { spawn } from "node:child_process";

const KILL_GRACE_MS = 5_000;
const PROCESS_ERROR = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROCESS";

export class CppCuteBrowserBuildProcessError extends Error {
  /**
   * @param {"cancelled" | "output-limit" | "output-sink" | "spawn" | "timeout"} reason
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
 *   onOutputChunk?: (stream: "stdout" | "stderr", chunk: Uint8Array) => Promise<void>;
 * }>} input
 * @returns {Promise<Readonly<{
 *   exitCode: number | null;
 *   terminationSignal: NodeJS.Signals | null;
 *   stdout: Uint8Array;
 *   stderr: Uint8Array;
 *   stdoutByteLength: number;
 *   stderrByteLength: number;
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
  if (input.onOutputChunk !== undefined && typeof input.onOutputChunk !== "function") {
    throw new TypeError("onOutputChunk must be an async output handler");
  }

  return await new Promise((resolve, reject) => {
    /** @type {Buffer[]} */
    const stdoutChunks = [];
    /** @type {Buffer[]} */
    const stderrChunks = [];
    let stdoutByteLength = 0;
    let stderrByteLength = 0;
    /** @type {"cancelled" | "output-limit" | "output-sink" | "timeout" | undefined} */
    let terminationReason;
    /** @type {unknown} */
    let outputSinkFailure;
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

    /** @param {"cancelled" | "output-limit" | "output-sink" | "timeout"} reason */
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

    let stdoutDelivery = Promise.resolve();
    let stderrDelivery = Promise.resolve();

    /**
     * @param {"stdout" | "stderr"} stream
     * @param {Buffer[]} chunks
     * @param {number} currentByteLength
     * @param {Uint8Array} chunk
     */
    const capture = (stream, chunks, currentByteLength, chunk) => {
      if (terminationReason !== undefined) return currentByteLength;
      const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      const remaining = input.maximumOutputByteLength - currentByteLength;
      const accepted = bytes.subarray(0, Math.max(0, remaining));
      const nextByteLength = currentByteLength + accepted.byteLength;
      if (input.onOutputChunk === undefined) {
        if (accepted.byteLength > 0) chunks.push(Buffer.from(accepted));
      } else if (accepted.byteLength > 0) {
        const readable = stream === "stdout" ? child.stdout : child.stderr;
        readable.pause();
        const previous = stream === "stdout" ? stdoutDelivery : stderrDelivery;
        const delivery = previous.then(async () => {
          await input.onOutputChunk?.(stream, new Uint8Array(accepted));
          readable.resume();
        }).catch((cause) => {
          outputSinkFailure = cause;
          terminate("output-sink");
          readable.resume();
        });
        if (stream === "stdout") stdoutDelivery = delivery;
        else stderrDelivery = delivery;
      }
      if (bytes.byteLength > remaining) {
        terminate("output-limit");
      }
      return nextByteLength;
    };

    child.stdout.on("data", (chunk) => {
      stdoutByteLength = capture("stdout", stdoutChunks, stdoutByteLength, chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderrByteLength = capture("stderr", stderrChunks, stderrByteLength, chunk);
    });
    child.once("error", (cause) => {
      spawnFailure = cause;
    });
    child.once("close", async (exitCode, terminationSignal) => {
      clearTimeout(timeout);
      if (killTimer !== undefined) clearTimeout(killTimer);
      input.signal?.removeEventListener("abort", onAbort);
      await Promise.all([stdoutDelivery, stderrDelivery]);

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
          "output-sink": "build step output could not be persisted",
          timeout: `build step exceeded ${input.maximumDurationMs} milliseconds`,
        };
        reject(new CppCuteBrowserBuildProcessError(
          terminationReason,
          messages[terminationReason],
          outputSinkFailure === undefined ? undefined : { cause: outputSinkFailure },
        ));
        return;
      }
      resolve(Object.freeze({
        exitCode,
        terminationSignal,
        stdout: input.onOutputChunk === undefined
          ? new Uint8Array(Buffer.concat(stdoutChunks, stdoutByteLength))
          : new Uint8Array(),
        stderr: input.onOutputChunk === undefined
          ? new Uint8Array(Buffer.concat(stderrChunks, stderrByteLength))
          : new Uint8Array(),
        stdoutByteLength,
        stderrByteLength,
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
