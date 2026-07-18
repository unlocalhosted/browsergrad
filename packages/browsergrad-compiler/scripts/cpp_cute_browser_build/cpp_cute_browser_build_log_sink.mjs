import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { open } from "node:fs/promises";

const INVALID = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOG-SINK-INVALID";

/**
 * Opens one exclusive, private build-log sink. Bytes are hashed and persisted
 * as they arrive, so a long-running or failed build does not retain its log in
 * memory or lose the completed prefix. The caller must seal the sink exactly
 * once, including after process failure.
 *
 * @param {Readonly<{
 *   path: string;
 *   maximumByteLength: number;
 *   mirror: "stdout" | "stderr" | null;
 * }>} input
 */
export async function createCppCuteBrowserBuildLogSink(input) {
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    throw new TypeError(`${INVALID}: expected an input object`);
  }
  if (typeof input.path !== "string" || input.path.length === 0) {
    throw new TypeError(`${INVALID}: expected a nonempty path`);
  }
  if (!Number.isSafeInteger(input.maximumByteLength) || input.maximumByteLength <= 0) {
    throw new TypeError(`${INVALID}: maximumByteLength must be a positive safe integer`);
  }
  if (input.mirror !== null && input.mirror !== "stdout" && input.mirror !== "stderr") {
    throw new TypeError(`${INVALID}: mirror must be stdout, stderr, or null`);
  }

  const handle = await open(
    input.path,
    constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
    0o600,
  );
  const hash = createHash("sha256");
  let byteLength = 0;
  let sealed = false;

  return Object.freeze({
    /** @param {Uint8Array} bytes */
    write: async (bytes) => {
      if (sealed) throw new Error(`${INVALID}: cannot write a sealed build log`);
      if (!(bytes instanceof Uint8Array)) {
        throw new TypeError(`${INVALID}: build-log chunks must be Uint8Array values`);
      }
      if (byteLength + bytes.byteLength > input.maximumByteLength) {
        throw new Error(`${INVALID}: build-log bytes exceeded the admitted maximum`);
      }
      let offset = 0;
      while (offset < bytes.byteLength) {
        const { bytesWritten } = await handle.write(
          bytes,
          offset,
          bytes.byteLength - offset,
          null,
        );
        if (bytesWritten === 0) {
          throw new Error(`${INVALID}: build-log write made no progress`);
        }
        offset += bytesWritten;
      }
      hash.update(bytes);
      byteLength += bytes.byteLength;
      if (input.mirror !== null && bytes.byteLength > 0) {
        await writeMirroredChunk(input.mirror, bytes);
      }
    },
    seal: async () => {
      if (sealed) throw new Error(`${INVALID}: build-log sink was already sealed`);
      sealed = true;
      let stat;
      try {
        await handle.sync();
        await handle.chmod(0o444);
        stat = await handle.stat({ bigint: true });
      } finally {
        await handle.close();
      }
      if (!stat.isFile() || stat.size !== BigInt(byteLength) || stat.nlink !== 1n ||
          (stat.mode & 0o222n) !== 0n) {
        throw new Error(`${INVALID}: sealed build-log metadata is invalid`);
      }
      return Object.freeze({
        path: input.path,
        sha256: hash.digest("hex"),
        byteLength,
        identity: Object.freeze({
          dev: stat.dev,
          ino: stat.ino,
          ctimeNs: stat.ctimeNs,
          birthtimeNs: stat.birthtimeNs,
        }),
      });
    },
  });
}

/** @param {"stdout" | "stderr"} name @param {Uint8Array} bytes */
async function writeMirroredChunk(name, bytes) {
  const stream = name === "stdout" ? process.stdout : process.stderr;
  if (stream.write(bytes)) return;
  await new Promise((resolve, reject) => {
    const cleanup = () => {
      stream.removeListener("drain", onDrain);
      stream.removeListener("error", onError);
    };
    const onDrain = () => {
      cleanup();
      resolve(undefined);
    };
    /** @param {Error} cause */
    const onError = (cause) => {
      cleanup();
      reject(cause);
    };
    stream.once("drain", onDrain);
    stream.once("error", onError);
  });
}
