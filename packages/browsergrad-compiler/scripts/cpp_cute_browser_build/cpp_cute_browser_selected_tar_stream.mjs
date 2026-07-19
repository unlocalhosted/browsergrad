import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdir, open, realpath, rm } from "node:fs/promises";
import { dirname, isAbsolute, join } from "node:path/posix";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

export const CPP_CUTE_BROWSER_SELECTED_TAR_MATERIALIZATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.selected-tar-materialization";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-SELECTED-TAR-STREAM";
const MATERIALIZATION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.selected-tar-materialization.v2";
const BLOCK_BYTES = 512;
const MAX_PAX_BYTES = 256 * 1024;
const MAX_FILE_BYTES = 64 * 1024 * 1024;
const MAX_TOTAL_FILE_BYTES = 512 * 1024 * 1024;
const MAX_FILES = 100_000;
const MAX_PATH_BYTES = 4_096;
const MAX_STREAM_BYTES = 768 * 1024 * 1024;
const MAX_SELECTIONS = 16;
const PORTABLE_SEGMENT = /^[A-Za-z0-9._+@=-]+$/u;
const SELECTION_ID = /^[a-z][a-z0-9-]*$/u;
const PAX_TIMESTAMP = /^-?[0-9]+(?:\.[0-9]+)?$/u;
const IGNORED_PAX_TIMESTAMP_KEYS = new Set([
  "LIBARCHIVE.creationtime",
  "atime",
  "ctime",
  "mtime",
]);
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });
const MATERIALIZED_TREES = new WeakMap();

export class CppCuteBrowserSelectedTarStreamError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserSelectedTarStreamError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Parses one normalized uncompressed POSIX pax/ustar stream and writes only
 * complete caller-selected subtrees or exact files into a new private output
 * root. This seam
 * grants no archive identity, decompressor, package-plan, license, or release
 * authority; those remain responsibilities of a higher-level wrapper.
 */
export async function materializeCppCuteBrowserSelectedTarStream(input) {
  const object = exactObject(input, ["chunks", "outputRoot", "selections"], "$.input");
  const outputRoot = absolutePath(object.outputRoot, "$.input.outputRoot");
  const selections = parseSelections(object.selections, "$.input.selections");
  const chunks = requireAsyncOrSyncIterable(object.chunks, "$.input.chunks");
  const parent = dirname(outputRoot);
  await admitPrivateCanonicalDirectory(parent, "$.input.outputRoot.parent");
  let rootIdentity;
  let parser;
  try {
    await mkdir(outputRoot, { mode: 0o700 });
    const root = await lstat(outputRoot, { bigint: true });
    requirePrivateDirectory(root, "$.input.outputRoot");
    rootIdentity = Object.freeze({ dev: root.dev, ino: root.ino });
    for (const selection of selections) {
      await createPrivateDirectory(join(outputRoot, selection.outputSubdirectory), "$.outputRoot");
    }
    parser = createParserState(outputRoot, selections);
    let streamBytes = 0;
    for await (const value of chunks) {
      const chunk = snapshotChunk(value, "$.input.chunks");
      streamBytes += chunk.byteLength;
      if (streamBytes > MAX_STREAM_BYTES) resource("$.input.chunks", "tar stream exceeds byte ceiling");
      await consumeChunk(parser, chunk);
    }
    await finishParser(parser);
    const selectionResults = selections.map((selection) => {
      const stored = parser.results.get(selection.selectionId);
      if (stored === undefined || stored.files.length === 0) {
        invalid("$.input.selections", `selected path ${JSON.stringify(selection.selectionId)} has no files`);
      }
      const files = [...stored.files].sort((left, right) => compareUtf8(left.relativePath, right.relativePath));
      const contentHash = sha256(canonicalJsonBytes({
        domain: "browsergrad.compiler.cpp-cute.selected-source-tree-content.v1",
        files,
      }));
      return Object.freeze({
        selectionId: selection.selectionId,
        selectionKind: selection.selectionKind,
        archiveSubtree: selection.archiveSubtree,
        outputSubdirectory: selection.outputSubdirectory,
        sourceTreeId: `bg.cpp.selected-source-tree.sha256.${contentHash}`,
        fileCount: files.length,
        fileContentByteLength: files.reduce(
          (total, file) => total + BigInt(file.byteLength),
          0n,
        ).toString(),
        files: Object.freeze(files),
      });
    });
    const manifestHash = sha256(canonicalJsonBytes({
      domain: MATERIALIZATION_HASH_DOMAIN,
      selections: selectionResults,
    }));
    const manifest = Object.freeze({
      schema: CPP_CUTE_BROWSER_SELECTED_TAR_MATERIALIZATION_SCHEMA,
      version: 2,
      materializationId: `bg.cpp.selected-tar-materialization.sha256.${manifestHash}`,
      authority: "caller-selected-normalized-tar-materialization-only",
      selections: Object.freeze(selectionResults),
      totals: Object.freeze({
        selectionCount: selectionResults.length,
        fileCount: parser.fileCount,
        fileContentByteLength: String(parser.fileBytes),
        consumedTarByteLength: String(streamBytes),
      }),
      claims: Object.freeze({
        strictNormalizedTarParsed: true,
        onlyRegularFileContentsMaterialized: true,
        collisionFreePortableStorageMaterialized: true,
        hierarchicalSourceTreesMaterialized: false,
        allSelectedStreamFilesMaterialized: true,
        callerSelectedPathsComplete: false,
        archiveIdentityVerified: false,
        decompressorVerified: false,
        headerSourcePlanBound: false,
        generatedClangResourceHeadersComplete: false,
        licenseReviewComplete: false,
        headerPacksAssembled: false,
        releaseReady: false,
      }),
    });
    MATERIALIZED_TREES.set(manifest, Object.freeze({
      outputRoot,
      rootIdentity,
      selectionFiles: new Map([...parser.results].map(([selectionId, result]) => [
        selectionId,
        new Map(result.sourceFiles),
      ])),
    }));
    return manifest;
  } catch (cause) {
    await closeParserOutput(parser);
    if (rootIdentity !== undefined) await removeOwnedRoot(outputRoot, rootIdentity);
    if (cause instanceof CppCuteBrowserSelectedTarStreamError) throw cause;
    invalid("$.input", "failed to materialize selected tar stream", { cause });
  }
}

export function requireCppCuteBrowserSelectedTarMaterializationAuthority(manifest) {
  if (typeof manifest !== "object" || manifest === null ||
      MATERIALIZED_TREES.get(manifest) === undefined) {
    invalid("$.manifest", "expected parser-issued live selected-tree authority");
  }
}

export function cppCuteBrowserSelectedTarMaterializationRoots(manifest) {
  const stored = MATERIALIZED_TREES.get(manifest);
  if (stored === undefined) invalid("$.manifest", "expected parser-issued live selected-tree authority");
  return Object.freeze(manifest.selections.map((selection) => Object.freeze({
    selectionId: selection.selectionId,
    storageRoot: join(stored.outputRoot, selection.outputSubdirectory),
    sourceTreeId: selection.sourceTreeId,
  })));
}

export async function copyCppCuteBrowserSelectedTarMaterializationFile(
  manifest,
  selectionId,
  relativePath,
) {
  const stored = MATERIALIZED_TREES.get(manifest);
  if (stored === undefined) invalid("$.manifest", "expected parser-issued selected-file authority");
  if (typeof selectionId !== "string" || !SELECTION_ID.test(selectionId)) {
    invalid("$.selectionId", "expected one portable selection ID");
  }
  portableRelativePath(relativePath, "$.relativePath");
  const source = stored.selectionFiles.get(selectionId)?.get(relativePath);
  if (source === undefined) invalid("$.relativePath", "selected file is absent from the materialization");
  let handle;
  try {
    const discovered = await lstat(source.path, { bigint: true });
    if (!sameOwnedFile(source, discovered)) invalid("$.relativePath", "selected file identity changed");
    handle = await open(source.path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameOwnedFile(source, before)) invalid("$.relativePath", "selected file changed before read");
    const bytes = new Uint8Array(Number(before.size));
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesRead <= 0) invalid("$.relativePath", "selected file became shorter while read");
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameOwnedFile(source, after) || sha256(bytes) !== source.contentSha256) {
      invalid("$.relativePath", "selected file bytes changed while read");
    }
    return bytes;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserSelectedTarStreamError) throw cause;
    invalid("$.relativePath", "failed to read exact selected file", { cause });
  } finally {
    await handle?.close();
  }
}

function createParserState(outputRoot, selections) {
  return {
    outputRoot,
    selections,
    buffer: Buffer.alloc(0),
    mode: "header",
    current: undefined,
    pendingPax: undefined,
    zeroBlockCount: 0,
    ended: false,
    fileCount: 0,
    fileBytes: 0,
    entryCount: 0,
    results: new Map(selections.map((selection) => [selection.selectionId, {
      files: [],
      sourceFiles: new Map(),
    }])),
    seenArchivePaths: new Set(),
  };
}

async function consumeChunk(state, chunk) {
  if (state.ended && chunk.some((byte) => byte !== 0)) {
    invalid("$.input.chunks", "nonzero bytes follow the tar end marker");
  }
  state.buffer = state.buffer.byteLength === 0
    ? Buffer.from(chunk)
    : Buffer.concat([state.buffer, chunk]);
  while (true) {
    if (state.mode === "header") {
      if (state.buffer.byteLength < BLOCK_BYTES) return;
      const header = state.buffer.subarray(0, BLOCK_BYTES);
      state.buffer = state.buffer.subarray(BLOCK_BYTES);
      if (header.every((byte) => byte === 0)) {
        state.zeroBlockCount += 1;
        if (state.zeroBlockCount >= 2) state.ended = true;
        continue;
      }
      if (state.ended || state.zeroBlockCount !== 0) {
        invalid("$.input.chunks", "nonzero bytes follow a zero tar end block");
      }
      state.entryCount += 1;
      if (state.entryCount > MAX_FILES * 4) resource("$.input.chunks", "tar entry count exceeds ceiling");
      state.current = parseHeader(header, state.pendingPax, state.entryCount);
      state.pendingPax = undefined;
      state.mode = "body";
      await prepareEntry(state);
      if (state.current.remaining === 0) await finishEntryBody(state, state.current);
      continue;
    }
    const current = state.current;
    if (current === undefined) invalid("$.input.chunks", "tar parser lost current entry");
    if (current.remaining > 0) {
      if (state.buffer.byteLength === 0) return;
      const length = Math.min(current.remaining, state.buffer.byteLength);
      const bytes = state.buffer.subarray(0, length);
      state.buffer = state.buffer.subarray(length);
      await consumeEntryBytes(current, bytes);
      current.remaining -= length;
      if (current.remaining > 0) continue;
      await finishEntryBody(state, current);
    }
    if (current.paddingRemaining > 0) {
      if (state.buffer.byteLength === 0) return;
      const length = Math.min(current.paddingRemaining, state.buffer.byteLength);
      const padding = state.buffer.subarray(0, length);
      if (padding.some((byte) => byte !== 0)) invalid(current.path, "tar entry has nonzero padding");
      state.buffer = state.buffer.subarray(length);
      current.paddingRemaining -= length;
      if (current.paddingRemaining > 0) continue;
    }
    state.current = undefined;
    state.mode = "header";
  }
}

async function prepareEntry(state) {
  const current = state.current;
  if (current === undefined) invalid("$.input.chunks", "tar parser lost current entry");
  if (current.type === "pax") {
    if (current.size > MAX_PAX_BYTES) resource(current.path, "PAX header exceeds byte ceiling");
    current.paxChunks = [];
    return;
  }
  if (current.type === "directory") return;
  const selected = selectArchivePath(state.selections, current.path);
  if (selected === undefined) {
    invalid(current.path, "normalized tar contains a regular file outside selected paths");
  }
  portableRelativePath(selected.relativePath, current.path);
  const result = state.results.get(selected.selection.selectionId);
  if (result === undefined) invalid(current.path, "selected tar result is unavailable");
  const duplicateKey = `${selected.selection.selectionId}\0${selected.relativePath}`;
  if (state.seenArchivePaths.has(duplicateKey)) invalid(current.path, "duplicate selected file path");
  state.seenArchivePaths.add(duplicateKey);
  if (current.size > MAX_FILE_BYTES) resource(current.path, "selected file exceeds byte ceiling");
  state.fileCount += 1;
  state.fileBytes += current.size;
  if (state.fileCount > MAX_FILES) resource(current.path, "selected file count exceeds ceiling");
  if (state.fileBytes > MAX_TOTAL_FILE_BYTES) resource(current.path, "selected file bytes exceed ceiling");
  const storageKey = sha256(Buffer.from(
    `${selected.selection.selectionId}\0${selected.relativePath}`,
    "utf8",
  ));
  const outputPath = join(
    state.outputRoot,
    selected.selection.outputSubdirectory,
    ".browsergrad-files",
    storageKey.slice(0, 2),
    storageKey,
  );
  await ensurePrivateDirectory(dirname(outputPath), state.outputRoot, current.path);
  let handle;
  try {
    handle = await open(
      outputPath,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o400,
    );
    const opened = await handle.stat({ bigint: true });
    if (!opened.isFile() || opened.nlink !== 1n || opened.size !== 0n) {
      invalid(current.path, "new selected output is not one empty regular file");
    }
    current.output = {
      handle,
      path: outputPath,
      identity: Object.freeze({ dev: opened.dev, ino: opened.ino }),
      offset: 0,
      digest: createHash("sha256"),
      result,
      relativePath: selected.relativePath,
    };
  } catch (cause) {
    await handle?.close().catch(() => {});
    if (cause instanceof CppCuteBrowserSelectedTarStreamError) throw cause;
    invalid(current.path, "failed to create selected output file", { cause });
  }
}

async function consumeEntryBytes(current, bytes) {
  if (current.type === "pax") {
    current.paxChunks.push(Buffer.from(bytes));
    return;
  }
  const output = current.output;
  if (output === undefined) return;
  output.digest.update(bytes);
  let written = 0;
  while (written < bytes.byteLength) {
    const result = await output.handle.write(
      bytes,
      written,
      bytes.byteLength - written,
      output.offset + written,
    );
    if (result.bytesWritten <= 0) invalid(current.path, "selected output stopped accepting bytes");
    written += result.bytesWritten;
  }
  output.offset += bytes.byteLength;
}

async function finishEntryBody(state, current) {
  if (current.type === "pax") {
    state.pendingPax = parsePax(Buffer.concat(current.paxChunks), current.path);
    return;
  }
  const output = current.output;
  if (output === undefined) return;
  const written = await output.handle.stat({ bigint: true });
  if (written.dev !== output.identity.dev || written.ino !== output.identity.ino ||
      written.nlink !== 1n || written.size !== BigInt(current.size)) {
    invalid(current.path, "selected output identity changed while written");
  }
  await output.handle.close();
  output.handle = undefined;
  const persisted = await lstat(output.path, { bigint: true });
  if (!persisted.isFile() || persisted.isSymbolicLink() || persisted.nlink !== 1n ||
      persisted.dev !== output.identity.dev || persisted.ino !== output.identity.ino ||
      persisted.size !== BigInt(current.size)) {
    invalid(current.path, "selected output path no longer names the owned file");
  }
  output.result.files.push(Object.freeze({
    relativePath: output.relativePath,
    contentSha256: output.digest.digest("hex"),
    byteLength: String(current.size),
  }));
  const expected = output.result.files.at(-1);
  output.result.sourceFiles.set(output.relativePath, Object.freeze({
    path: output.path,
    identity: output.identity,
    contentSha256: expected.contentSha256,
    byteLength: expected.byteLength,
  }));
}

async function finishParser(state) {
  await closeParserOutput(state);
  if (state.mode !== "header" || state.current !== undefined) {
    invalid("$.input.chunks", "tar stream ended inside an entry");
  }
  if (!state.ended || state.zeroBlockCount < 2) {
    invalid("$.input.chunks", "tar stream lacks two zero end blocks");
  }
  if (state.pendingPax !== undefined) invalid("$.input.chunks", "orphan PAX header at end of stream");
  if (state.buffer.some((byte) => byte !== 0)) {
    invalid("$.input.chunks", "nonzero trailing bytes follow tar end marker");
  }
}

async function closeParserOutput(state) {
  if (state?.current?.output?.handle !== undefined) {
    await state.current.output.handle.close().catch(() => {});
    state.current.output.handle = undefined;
  }
}

function parseHeader(header, pax, ordinal) {
  verifyHeaderChecksum(header, ordinal);
  const rawType = header[156];
  const type = rawType === 0 || rawType === 0x30
    ? "regular"
    : rawType === 0x35
      ? "directory"
      : rawType === 0x78
        ? "pax"
        : undefined;
  if (type === undefined) {
    invalid(`$.entries[${ordinal}]`, `tar entry type ${JSON.stringify(String.fromCharCode(rawType))} is forbidden`);
  }
  if (type === "pax" && pax !== undefined) {
    invalid(`$.entries[${ordinal}]`, "consecutive PAX headers are forbidden");
  }
  const magic = header.subarray(257, 263);
  if (!magic.equals(Buffer.from("ustar\0", "ascii")) &&
      !magic.equals(Buffer.from("ustar ", "ascii"))) {
    invalid(`$.entries[${ordinal}]`, "expected POSIX ustar/pax header magic");
  }
  const name = decodeNulTerminated(header.subarray(0, 100), `$.entries[${ordinal}].name`);
  const prefix = decodeNulTerminated(header.subarray(345, 500), `$.entries[${ordinal}].prefix`);
  const headerPath = prefix === "" ? name : `${prefix}/${name}`;
  const selectedPath = pax?.path ?? headerPath;
  const path = canonicalArchivePath(selectedPath, `$.entries[${ordinal}].path`, type === "directory");
  const size = parseOctal(header.subarray(124, 136), `$.entries[${ordinal}].size`);
  if (type === "directory" && size !== 0) invalid(path, "directory entry must have zero size");
  if (type !== "pax" && pax !== undefined && Object.hasOwn(pax, "size")) {
    invalid(path, "PAX size overrides are forbidden");
  }
  return {
    path,
    type,
    size,
    remaining: size,
    paddingRemaining: (BLOCK_BYTES - (size % BLOCK_BYTES)) % BLOCK_BYTES,
    output: undefined,
    paxChunks: undefined,
  };
}

function verifyHeaderChecksum(header, ordinal) {
  const expected = parseOctal(header.subarray(148, 156), `$.entries[${ordinal}].checksum`);
  let sum = 0;
  for (let index = 0; index < header.byteLength; index += 1) {
    sum += index >= 148 && index < 156 ? 0x20 : header[index];
  }
  if (sum !== expected) invalid(`$.entries[${ordinal}].checksum`, "tar header checksum mismatch");
}

function parsePax(bytes, diagnosticPath) {
  const result = {};
  let offset = 0;
  while (offset < bytes.byteLength) {
    const space = bytes.indexOf(0x20, offset);
    if (space <= offset) invalid(diagnosticPath, "malformed PAX record length");
    const lengthText = Buffer.from(bytes.subarray(offset, space)).toString("ascii");
    if (!/^[1-9][0-9]*$/u.test(lengthText)) invalid(diagnosticPath, "malformed PAX record length");
    const byteLength = Number(lengthText);
    if (!Number.isSafeInteger(byteLength) || byteLength <= space - offset + 3) {
      invalid(diagnosticPath, "malformed PAX record length");
    }
    const recordEnd = offset + byteLength;
    if (recordEnd > bytes.byteLength || bytes[recordEnd - 1] !== 0x0a) {
      invalid(diagnosticPath, "PAX record length does not match UTF-8 bytes");
    }
    let body;
    try {
      body = UTF8_DECODER.decode(bytes.subarray(space + 1, recordEnd - 1));
    } catch (cause) {
      invalid(diagnosticPath, "PAX records must be strict UTF-8", { cause });
    }
    const separator = body.indexOf("=");
    if (separator < 0) invalid(diagnosticPath, "PAX record lacks key/value separator");
    const key = body.slice(0, separator);
    const value = body.slice(separator + 1);
    if (!/^[A-Za-z][A-Za-z0-9.]*$/u.test(key) || Object.hasOwn(result, key)) {
      invalid(diagnosticPath, "PAX keys must be unique portable names");
    }
    if (key === "linkpath" || key === "size") invalid(diagnosticPath, `PAX key ${key} is forbidden`);
    if (key === "path") {
      result.path = value;
    } else if (IGNORED_PAX_TIMESTAMP_KEYS.has(key)) {
      if (!PAX_TIMESTAMP.test(value)) invalid(diagnosticPath, `PAX timestamp ${key} is malformed`);
      result[key] = value;
    } else {
      invalid(diagnosticPath, `PAX key ${key} is not admitted`);
    }
    offset += byteLength;
  }
  return Object.freeze(result);
}

function parseSelections(value, diagnosticPath) {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype ||
      value.length < 1 || value.length > MAX_SELECTIONS) {
    invalid(diagnosticPath, `expected a dense array with 1 to ${MAX_SELECTIONS} selections`);
  }
  const selections = [];
  for (let index = 0; index < value.length; index += 1) {
    if (!Object.hasOwn(value, index)) invalid(`${diagnosticPath}[${index}]`, "sparse arrays are forbidden");
    const path = `${diagnosticPath}[${index}]`;
    const object = exactObject(
      value[index],
      ["selectionId", "selectionKind", "archiveSubtree", "outputSubdirectory"],
      path,
    );
    if (typeof object.selectionId !== "string" || !SELECTION_ID.test(object.selectionId)) {
      invalid(`${path}.selectionId`, "expected one portable selection ID");
    }
    const archiveSubtree = canonicalArchivePath(object.archiveSubtree, `${path}.archiveSubtree`, false);
    if (object.selectionKind !== "subtree" && object.selectionKind !== "file") {
      invalid(`${path}.selectionKind`, "expected subtree or file");
    }
    const outputSubdirectory = portableSingleSegment(
      object.outputSubdirectory,
      `${path}.outputSubdirectory`,
    );
    selections.push(Object.freeze({
      selectionId: object.selectionId,
      selectionKind: object.selectionKind,
      archiveSubtree,
      outputSubdirectory,
    }));
  }
  selections.sort((left, right) => compareUtf8(left.selectionId, right.selectionId));
  if (new Set(selections.map((selection) => selection.selectionId)).size !== selections.length ||
      new Set(selections.map((selection) => selection.outputSubdirectory)).size !== selections.length ||
      new Set(selections.map((selection) => selection.archiveSubtree)).size !== selections.length) {
    invalid(diagnosticPath, "selection IDs, output directories, and archive subtrees must be unique");
  }
  for (const [index, left] of selections.entries()) {
    for (const right of selections.slice(index + 1)) {
      if ((left.selectionKind === "subtree" &&
            right.archiveSubtree.startsWith(`${left.archiveSubtree}/`)) ||
          (right.selectionKind === "subtree" &&
            left.archiveSubtree.startsWith(`${right.archiveSubtree}/`))) {
        invalid(diagnosticPath, "selected archive paths must not overlap");
      }
    }
  }
  return Object.freeze(selections);
}

function selectArchivePath(selections, path) {
  for (const selection of selections) {
    if (selection.selectionKind === "file" && path === selection.archiveSubtree) {
      return Object.freeze({
        selection,
        relativePath: selection.archiveSubtree.slice(selection.archiveSubtree.lastIndexOf("/") + 1),
      });
    }
    if (selection.selectionKind === "subtree" && path.startsWith(`${selection.archiveSubtree}/`)) {
      return Object.freeze({
        selection,
        relativePath: path.slice(selection.archiveSubtree.length + 1),
      });
    }
  }
  return undefined;
}

function canonicalArchivePath(value, diagnosticPath, allowTrailingSlash) {
  if (typeof value !== "string" || value === "" || value.includes("\0") ||
      value.includes("\\") || isAbsolute(value)) {
    invalid(diagnosticPath, "expected one nonempty relative NUL-free POSIX archive path");
  }
  let path = value.startsWith("./") ? value.slice(2) : value;
  if (allowTrailingSlash && path.endsWith("/")) path = path.slice(0, -1);
  const segments = path.split("/");
  if (segments.some((segment) => segment === "" || segment === "." || segment === "..")) {
    invalid(diagnosticPath, "archive paths must not contain empty, dot, or parent segments");
  }
  if (Buffer.byteLength(path, "utf8") > MAX_PATH_BYTES) resource(diagnosticPath, "archive path is too long");
  return path;
}

function portableRelativePath(value, diagnosticPath) {
  const path = canonicalArchivePath(value, diagnosticPath, false);
  if (path.split("/").some((segment) => !PORTABLE_SEGMENT.test(segment))) {
    invalid(diagnosticPath, "selected relative path contains a non-portable segment");
  }
  return path;
}

function portableSingleSegment(value, diagnosticPath) {
  if (typeof value !== "string" || !PORTABLE_SEGMENT.test(value)) {
    invalid(diagnosticPath, "expected one portable output-directory segment");
  }
  return value;
}

function parseOctal(bytes, diagnosticPath) {
  if ((bytes[0] & 0x80) !== 0) invalid(diagnosticPath, "base-256 tar integers are forbidden");
  const raw = Buffer.from(bytes).toString("ascii");
  const nul = raw.indexOf("\0");
  if (nul >= 0) {
    for (let index = nul; index < raw.length; index += 1) {
      if (raw[index] !== "\0" && raw[index] !== " ") {
        invalid(diagnosticPath, "tar octal field has non-padding bytes after NUL");
      }
    }
  }
  const text = (nul < 0 ? raw : raw.slice(0, nul)).trim();
  if (text === "") return 0;
  if (!/^[0-7]+$/u.test(text)) invalid(diagnosticPath, "expected canonical tar octal integer");
  const value = Number.parseInt(text, 8);
  if (!Number.isSafeInteger(value) || value < 0) resource(diagnosticPath, "tar integer exceeds safe range");
  return value;
}

function decodeNulTerminated(bytes, diagnosticPath) {
  const nul = bytes.indexOf(0);
  const content = nul < 0 ? bytes : bytes.subarray(0, nul);
  if (nul >= 0 && bytes.subarray(nul).some((byte) => byte !== 0)) {
    invalid(diagnosticPath, "tar string has nonzero bytes after NUL");
  }
  try {
    return UTF8_DECODER.decode(content);
  } catch (cause) {
    invalid(diagnosticPath, "tar paths must be strict UTF-8", { cause });
  }
}

async function createPrivateDirectory(path, diagnosticPath) {
  try {
    await mkdir(path, { mode: 0o700 });
  } catch (cause) {
    invalid(diagnosticPath, "failed to create private selected-tree directory", { cause });
  }
  const entry = await lstat(path, { bigint: true });
  requirePrivateDirectory(entry, diagnosticPath);
}

async function ensurePrivateDirectory(path, outputRoot, diagnosticPath) {
  if (path === outputRoot) return;
  const relative = path.slice(outputRoot.length + 1);
  if (relative.startsWith("../") || relative === ".." || isAbsolute(relative)) {
    invalid(diagnosticPath, "selected output escaped its owned root");
  }
  let current = outputRoot;
  for (const segment of relative.split("/")) {
    current = join(current, segment);
    try {
      await mkdir(current, { mode: 0o700 });
    } catch (cause) {
      if (!(typeof cause === "object" && cause !== null && "code" in cause && cause.code === "EEXIST")) {
        invalid(diagnosticPath, "failed to create selected output directory", { cause });
      }
    }
    const entry = await lstat(current, { bigint: true });
    requirePrivateDirectory(entry, diagnosticPath);
  }
}

async function admitPrivateCanonicalDirectory(path, diagnosticPath) {
  let before;
  try {
    before = await lstat(path, { bigint: true });
  } catch (cause) {
    invalid(diagnosticPath, "selected-tree parent is unavailable", { cause });
  }
  requirePrivateDirectory(before, diagnosticPath);
  if (await realpath(path) !== path) invalid(diagnosticPath, "selected-tree parent must be canonical");
  const after = await lstat(path, { bigint: true });
  if (before.dev !== after.dev || before.ino !== after.ino) {
    invalid(diagnosticPath, "selected-tree parent identity changed");
  }
}

function requirePrivateDirectory(entry, diagnosticPath) {
  if (!entry.isDirectory() || entry.isSymbolicLink() || entry.nlink < 1n) {
    invalid(diagnosticPath, "expected one non-symlink directory");
  }
  if (typeof process.getuid !== "function" || entry.uid !== BigInt(process.getuid())) {
    invalid(diagnosticPath, "selected-tree directory must be owned by the current user");
  }
  if ((Number(entry.mode) & 0o077) !== 0) {
    invalid(diagnosticPath, "selected-tree directories must be private to the current user");
  }
}

async function removeOwnedRoot(path, identity) {
  try {
    const current = await lstat(path, { bigint: true });
    if (current.dev === identity.dev && current.ino === identity.ino) {
      await rm(path, { recursive: true });
    }
  } catch {
    // Never remove a path that no longer names the directory created here.
  }
}

function requireAsyncOrSyncIterable(value, diagnosticPath) {
  if ((typeof value !== "object" && typeof value !== "function") || value === null) {
    invalid(diagnosticPath, "expected one iterable byte-stream source");
  }
  let asyncIterator;
  let iterator;
  try {
    asyncIterator = value[Symbol.asyncIterator];
    iterator = value[Symbol.iterator];
  } catch (cause) {
    invalid(diagnosticPath, "byte-stream iterator access failed", { cause });
  }
  if (typeof asyncIterator !== "function" && typeof iterator !== "function") {
    invalid(diagnosticPath, "expected one iterable byte-stream source");
  }
  if (typeof asyncIterator === "function") {
    return Object.freeze({
      [Symbol.asyncIterator]() {
        return Reflect.apply(asyncIterator, value, []);
      },
    });
  }
  return Object.freeze({
    [Symbol.iterator]() {
      return Reflect.apply(iterator, value, []);
    },
  });
}

function snapshotChunk(value, diagnosticPath) {
  if (!(value instanceof Uint8Array) || value.buffer instanceof SharedArrayBuffer) {
    invalid(diagnosticPath, "tar chunks must be unshared Uint8Array values");
  }
  return new Uint8Array(value);
}

function exactObject(value, keys, diagnosticPath) {
  if (typeof value !== "object" || value === null || Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(diagnosticPath, "expected one plain data record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const actual = Reflect.ownKeys(descriptors);
  if (actual.length !== keys.length ||
      actual.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(diagnosticPath, `expected only ${keys.join(", ")}`);
  }
  const result = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor)) {
      invalid(`${diagnosticPath}.${key}`, "expected data property");
    }
    result[key] = descriptor.value;
  }
  return result;
}

function absolutePath(value, diagnosticPath) {
  if (typeof value !== "string" || !isAbsolute(value) || value.includes("\0")) {
    invalid(diagnosticPath, "expected one absolute NUL-free POSIX path");
  }
  return value;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function sameOwnedFile(source, stat) {
  return stat.isFile() && !stat.isSymbolicLink() && stat.nlink === 1n &&
    stat.dev === source.identity.dev && stat.ino === source.identity.ino &&
    stat.size === BigInt(source.byteLength);
}

function invalid(path, message, options) {
  throw new CppCuteBrowserSelectedTarStreamError(path, message, options);
}

function resource(path, message) {
  invalid(path, `resource limit: ${message}`);
}
