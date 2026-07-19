import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdtemp, open, realpath, rm, unlink } from "node:fs/promises";
import { dirname, isAbsolute, join } from "node:path/posix";
import { pipeline } from "node:stream/promises";
import { pathToFileURL } from "node:url";
import { createZstdDecompress } from "node:zlib";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  copyCppCuteBrowserSelectedTarMaterializationFile,
  cppCuteBrowserSelectedTarMaterializationRoots,
  materializeCppCuteBrowserSelectedTarStream,
} from "./cpp_cute_browser_selected_tar_stream.mjs";

export const CPP_CUTE_BROWSER_BSDTAR_TOOL_ADMISSION_SCHEMA =
  "browsergrad.compiler.cpp-cute.bsdtar-tool-admission";
export const CPP_CUTE_BROWSER_ARCHIVE_NORMALIZATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.archive-normalization";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-ARCHIVE-NORMALIZATION";
const TOOL_HASH_DOMAIN = "browsergrad.compiler.cpp-cute.bsdtar-tool-admission.v1";
const PINNED_TOOL_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.pinned-archive-normalization-environment.v1";
const NORMALIZATION_HASH_DOMAIN = "browsergrad.compiler.cpp-cute.archive-normalization.v1";
const ARCHIVE_FORMATS = new Set(["deb-data-tar-zstd", "tar.gz", "tar.xz"]);
const PORTABLE_PATH = /^[A-Za-z0-9._+@=-]+(?:\/[A-Za-z0-9._+@=-]+)*$/u;
const SELECTION_ID = /^[a-z][a-z0-9-]*$/u;
const MAX_SELECTIONS = 16;
const MAX_TOOL_BYTES = 64 * 1024 * 1024;
const MAX_ARCHIVE_BYTES = 512 * 1024 * 1024;
const MAX_INNER_ARCHIVE_BYTES = 512 * 1024 * 1024;
const MAX_INNER_TAR_BYTES = 512 * 1024 * 1024;
const MAX_STDERR_BYTES = 64 * 1024;
const MAX_VERSION_BYTES = 4 * 1024;
const PROCESS_TIMEOUT_MS = 5 * 60 * 1000;
const KILL_GRACE_MS = 5_000;
const READ_BUFFER_BYTES = 1024 * 1024;
const CLOSED_ENVIRONMENT = Object.freeze({ LC_ALL: "C", LANG: "C", TZ: "UTC" });
const TOOL_AUTHORITIES = new WeakMap();
const NORMALIZATION_AUTHORITIES = new WeakMap();
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });
const PINNED_BSDTAR = Object.freeze({
  platform: "darwin",
  architecture: "arm64",
  executablePath: "/usr/bin/bsdtar",
  executableSha256: "2806c6e01f077f360f4046e597ef1a62d96c772eb937b5c35852ad97c9d0a625",
  executableByteLength: "195680",
  observedVersion: "bsdtar 3.5.3 - libarchive 3.5.3 zlib/1.2.12 liblzma/5.4.3 bz2lib/1.0.8",
});
const PINNED_NODE_ZSTD_RUNTIME = Object.freeze({
  platform: "darwin",
  architecture: "arm64",
  runtimeVersion: "v25.9.0",
  executableSha256: "4b3fe8b384e30ee917e28a9f5b79a3ca64b72b13b70d9ab2273e6e9a823f4cbf",
  executableByteLength: "133274256",
  zstdVersion: "1.5.7",
  execArgv: Object.freeze([]),
  nodeOptions: "absent",
});

export class CppCuteBrowserArchiveNormalizationError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserArchiveNormalizationError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Hashes and executes one caller-selected bsdtar executable under a closed
 * environment. The result observes exact local bytes and a bounded --version
 * response; it does not attest that those bytes implement libarchive.
 */
export async function admitCppCuteBrowserBsdtarTool(input) {
  const object = exactObject(input, ["executablePath"], "$.input");
  const executablePath = absolutePath(object.executablePath, "$.input.executablePath");
  const binding = await inspectTool(executablePath, "$.input.executablePath");
  const versionRun = await runBoundToolProcess({
    arguments: ["--version"],
    binding,
    consumeStdout: (chunks) => readBoundedChunks(chunks, MAX_VERSION_BYTES, "$.tool.version"),
    diagnosticPath: "$.tool.version",
  });
  const version = strictUtf8(versionRun.consumerResult, "$.tool.version").trimEnd();
  if (!/^bsdtar [0-9]+\.[0-9]+(?:\.[0-9]+)? - libarchive [^\r\n]{1,256}$/u.test(version)) {
    invalid("$.tool.version", "expected one bounded bsdtar/libarchive version line");
  }
  const toolHash = sha256(canonicalJsonBytes({
    domain: TOOL_HASH_DOMAIN,
    executableSha256: binding.sha256,
    executableByteLength: String(binding.snapshot.size),
    version,
  }));
  const admission = Object.freeze({
    schema: CPP_CUTE_BROWSER_BSDTAR_TOOL_ADMISSION_SCHEMA,
    version: 1,
    toolAdmissionId: `bg.cpp.bsdtar-tool-admission.sha256.${toolHash}`,
    authority: "caller-selected-host-bsdtar-observation-only",
    executableSha256: binding.sha256,
    executableByteLength: String(binding.snapshot.size),
    observedVersion: version,
    claims: Object.freeze({
      executableRegularFileObserved: true,
      executableBytesHashed: true,
      closedEnvironmentVersionObserved: true,
      toolImplementationAttested: false,
      packageToolIdentityPinned: false,
      releaseReady: false,
    }),
  });
  TOOL_AUTHORITIES.set(admission, binding);
  return admission;
}

/**
 * Narrows the generic observed-tool seam to the exact package-reviewed Darwin
 * archive-normalization environment used by the current header-pack builder.
 * Other platforms may add an independently reviewed identity; they must not
 * inherit authority from version strings alone.
 */
export async function admitPinnedCppCuteBrowserArchiveNormalizationEnvironment(input) {
  const object = exactObject(input, ["executablePath"], "$.input");
  const executablePath = absolutePath(object.executablePath, "$.input.executablePath");
  const observed = await admitCppCuteBrowserBsdtarTool({ executablePath });
  if (process.platform !== PINNED_BSDTAR.platform || process.arch !== PINNED_BSDTAR.architecture ||
      executablePath !== PINNED_BSDTAR.executablePath ||
      observed.executableSha256 !== PINNED_BSDTAR.executableSha256 ||
      observed.executableByteLength !== PINNED_BSDTAR.executableByteLength ||
      observed.observedVersion !== PINNED_BSDTAR.observedVersion) {
    invalid("$.input.executablePath", "bsdtar differs from the package-reviewed builder identity");
  }
  const nodeRuntime = await inspectPinnedNodeZstdRuntime();
  const pinnedHash = sha256(canonicalJsonBytes({
    domain: PINNED_TOOL_HASH_DOMAIN,
    bsdtar: PINNED_BSDTAR,
    nodeZstdRuntime: nodeRuntime,
  }));
  const pinned = Object.freeze({
    schema: CPP_CUTE_BROWSER_BSDTAR_TOOL_ADMISSION_SCHEMA,
    version: 1,
    toolAdmissionId: `bg.cpp.pinned-archive-normalization-environment.sha256.${pinnedHash}`,
    authority: "package-pinned-archive-normalization-environment",
    executableSha256: observed.executableSha256,
    executableByteLength: observed.executableByteLength,
    observedVersion: observed.observedVersion,
    nodeZstdRuntime: nodeRuntime,
    claims: Object.freeze({
      executableRegularFileObserved: true,
      executableBytesHashed: true,
      closedEnvironmentVersionObserved: true,
      toolImplementationAttested: false,
      packageToolIdentityPinned: true,
      nodeZstdRuntimeIdentityPinned: true,
      releaseReady: false,
    }),
  });
  const authority = TOOL_AUTHORITIES.get(observed);
  if (authority === undefined) {
    invalid("$.input.executablePath", "observed normalization-tool authority was lost");
  }
  TOOL_AUTHORITIES.set(pinned, authority);
  return pinned;
}

export function requireCppCuteBrowserBsdtarToolAuthority(admission) {
  if (typeof admission !== "object" || admission === null ||
      TOOL_AUTHORITIES.get(admission) === undefined) {
    invalid("$.tool", "expected verifier-issued bsdtar tool authority");
  }
}

/**
 * Repackages caller-selected subtrees into a normalized pax stream, then
 * delegates all path and file materialization to the strict streaming parser.
 * The input archive must already live in a private current-user directory.
 */
export async function materializeCppCuteBrowserNormalizedArchive(input) {
  const object = exactObject(
    input,
    ["archiveFormat", "archivePath", "outputRoot", "selections", "tool"],
    "$.input",
  );
  const binding = TOOL_AUTHORITIES.get(object.tool);
  if (binding === undefined) invalid("$.input.tool", "expected verifier-issued bsdtar tool authority");
  const archiveFormat = archiveFormatValue(object.archiveFormat, "$.input.archiveFormat");
  const archivePath = absolutePath(object.archivePath, "$.input.archivePath");
  const outputRoot = absolutePath(object.outputRoot, "$.input.outputRoot");
  const selections = parseSelections(object.selections, "$.input.selections");
  await admitPrivateCanonicalDirectory(dirname(outputRoot), "$.input.outputRoot.parent");
  const archive = await inspectPrivateArchive(archivePath, "$.input.archivePath");
  let scratchRoot;
  let scratchIdentity;
  let materialization;
  let outputIdentity;
  try {
    let normalizedArchivePath = archivePath;
    const processReports = [];
    let intermediate;
    if (archiveFormat === "deb-data-tar-zstd") {
      scratchRoot = await realpath(await mkdtemp(join(dirname(outputRoot), ".browsergrad-deb-data-")));
      const scratch = await lstat(scratchRoot, { bigint: true });
      scratchIdentity = Object.freeze({ dev: scratch.dev, ino: scratch.ino });
      const innerPath = join(scratchRoot, "data.tar.zst");
      const innerTarPath = join(scratchRoot, "data.tar");
      const innerRun = await runBoundToolProcess({
        arguments: ["-xOf", archivePath, "data.tar.zst"],
        binding,
        consumeStdout: (chunks) => materializeRawStream(
          chunks,
          innerPath,
          MAX_INNER_ARCHIVE_BYTES,
          "$.archive.dataTarZstd",
        ),
        diagnosticPath: "$.archive.dataTarZstd",
      });
      const decompressed = await decompressZstdFile(
        innerPath,
        innerTarPath,
        "$.archive.dataTar",
      );
      normalizedArchivePath = innerTarPath;
      intermediate = Object.freeze({
        memberName: "data.tar.zst",
        memberSha256: innerRun.consumerResult.sha256,
        memberByteLength: innerRun.consumerResult.byteLength,
        decompressedTarSha256: decompressed.sha256,
        decompressedTarByteLength: decompressed.byteLength,
        decompressor: "node:zlib.createZstdDecompress",
        runtimeVersion: process.version,
        ...(object.tool.nodeZstdRuntime === undefined
          ? {}
          : { pinnedRuntime: object.tool.nodeZstdRuntime }),
      });
      processReports.push(processReport("deb-data-member-read", innerRun));
    }
    const normalizeRun = await runBoundToolProcess({
      arguments: [
        "-cf",
        "-",
        "--format=pax",
        ...selections.map((selection) => `--include=${selection.archiveSubtree}/*`),
        `@${normalizedArchivePath}`,
      ],
      binding,
      consumeStdout: async (chunks) => {
        const parsed = await materializeCppCuteBrowserSelectedTarStream({
          chunks,
          outputRoot,
          selections,
        });
        const root = await lstat(outputRoot, { bigint: true });
        outputIdentity = Object.freeze({ dev: root.dev, ino: root.ino });
        materialization = parsed;
        return parsed;
      },
      diagnosticPath: "$.archive.normalization",
    });
    processReports.push(processReport("selected-pax-normalization", normalizeRun));
    await verifyPrivateArchive(archive, "$.input.archivePath");
    await verifyTool(binding, "$.input.tool");
    const normalizationHash = sha256(canonicalJsonBytes({
      domain: NORMALIZATION_HASH_DOMAIN,
      archiveFormat,
      observedArchiveSha256: archive.sha256,
      observedArchiveByteLength: String(archive.snapshot.size),
      toolAdmissionId: object.tool.toolAdmissionId,
      materializationId: materialization.materializationId,
      ...(intermediate === undefined ? {} : { intermediate }),
      processes: processReports,
    }));
    const result = Object.freeze({
      schema: CPP_CUTE_BROWSER_ARCHIVE_NORMALIZATION_SCHEMA,
      version: 1,
      normalizationId: `bg.cpp.archive-normalization.sha256.${normalizationHash}`,
      authority: "caller-expected-host-tool-archive-normalization-only",
      archiveFormat,
      observedArchiveSha256: archive.sha256,
      observedArchiveByteLength: String(archive.snapshot.size),
      tool: Object.freeze({
        toolAdmissionId: object.tool.toolAdmissionId,
        executableSha256: object.tool.executableSha256,
        executableByteLength: object.tool.executableByteLength,
        observedVersion: object.tool.observedVersion,
      }),
      selections: materialization.selections,
      totals: materialization.totals,
      ...(intermediate === undefined ? {} : { intermediate }),
      processes: Object.freeze(processReports),
      claims: Object.freeze({
        observedArchiveBytesHashed: true,
        expectedArchiveIdentityBound: false,
        hostToolExecutableBytesHashed: true,
        hostToolImplementationAttested: false,
        hostToolPackageIdentityPinned: object.tool.claims.packageToolIdentityPinned,
        nodeZstdDecompressorObserved: archiveFormat === "deb-data-tar-zstd",
        decompressorImplementationAttested: false,
        nodeZstdDecompressorPackageIdentityPinned:
          archiveFormat === "deb-data-tar-zstd" && object.tool.nodeZstdRuntime !== undefined,
        strictNormalizedTarParsed: true,
        collisionFreePortableStorageMaterialized: true,
        hierarchicalSourceTreesMaterialized: false,
        allSelectedStreamFilesMaterialized: true,
        callerSelectedSubtreesComplete: false,
        headerSourcePlanBound: false,
        licenseReviewComplete: false,
        releaseReady: false,
      }),
    });
    NORMALIZATION_AUTHORITIES.set(result, Object.freeze({ materialization }));
    return result;
  } catch (cause) {
    if (outputIdentity !== undefined) await removeOwnedRoot(outputRoot, outputIdentity);
    if (cause instanceof CppCuteBrowserArchiveNormalizationError) throw cause;
    invalid("$.input", "failed to normalize selected archive subtrees", { cause });
  } finally {
    if (scratchRoot !== undefined && scratchIdentity !== undefined) {
      await removeOwnedRoot(scratchRoot, scratchIdentity);
    }
  }
}

async function inspectPinnedNodeZstdRuntime() {
  if (process.platform !== PINNED_NODE_ZSTD_RUNTIME.platform ||
      process.arch !== PINNED_NODE_ZSTD_RUNTIME.architecture ||
      process.version !== PINNED_NODE_ZSTD_RUNTIME.runtimeVersion ||
      process.versions.zstd !== PINNED_NODE_ZSTD_RUNTIME.zstdVersion ||
      process.execArgv.length !== 0 || process.env.NODE_OPTIONS !== undefined) {
    invalid(
      "$.runtime",
      `Node/Zstd runtime differs from the package-reviewed builder identity; execArgv=${JSON.stringify(process.execArgv).slice(0, 512)}`,
    );
  }
  const binding = await inspectExecutable(
    process.execPath,
    "$.runtime.executablePath",
    "Node runtime",
    256 * 1024 * 1024,
  );
  if (binding.sha256 !== PINNED_NODE_ZSTD_RUNTIME.executableSha256 ||
      String(binding.snapshot.size) !== PINNED_NODE_ZSTD_RUNTIME.executableByteLength) {
    invalid("$.runtime.executablePath", "Node executable differs from the package-reviewed builder identity");
  }
  return Object.freeze({ ...PINNED_NODE_ZSTD_RUNTIME });
}

export function requireCppCuteBrowserArchiveNormalizationAuthority(normalization) {
  if (typeof normalization !== "object" || normalization === null ||
      NORMALIZATION_AUTHORITIES.get(normalization) === undefined) {
    invalid("$.normalization", "expected normalizer-issued archive authority");
  }
}

export function cppCuteBrowserArchiveNormalizationRoots(normalization) {
  const stored = NORMALIZATION_AUTHORITIES.get(normalization);
  if (stored === undefined) invalid("$.normalization", "expected normalizer-issued archive authority");
  return cppCuteBrowserSelectedTarMaterializationRoots(stored.materialization);
}

export async function copyCppCuteBrowserArchiveNormalizationFile(
  normalization,
  selectionId,
  relativePath,
) {
  const stored = NORMALIZATION_AUTHORITIES.get(normalization);
  if (stored === undefined) invalid("$.normalization", "expected normalizer-issued archive authority");
  return copyCppCuteBrowserSelectedTarMaterializationFile(
    stored.materialization,
    selectionId,
    relativePath,
  );
}

async function inspectTool(path, diagnosticPath) {
  return inspectExecutable(path, diagnosticPath, "bsdtar", MAX_TOOL_BYTES);
}

async function inspectExecutable(path, diagnosticPath, label, maximumBytes) {
  if (await realpath(path).catch(() => undefined) !== path) {
    invalid(diagnosticPath, `${label} executable path must be canonical and non-symlinked`);
  }
  await admitTrustedToolDirectory(dirname(path), diagnosticPath, label);
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    requireToolStat(before, diagnosticPath, label);
    if (before.size > BigInt(maximumBytes)) resource(diagnosticPath, `${label} executable exceeds byte ceiling`);
    const sha = await hashOpenFile(handle, Number(before.size), diagnosticPath);
    const after = await handle.stat({ bigint: true });
    if (!sameSnapshot(before, after)) invalid(diagnosticPath, "bsdtar executable changed while hashed");
    return Object.freeze({ path, sha256: sha, snapshot: fileSnapshot(after) });
  } catch (cause) {
    if (cause instanceof CppCuteBrowserArchiveNormalizationError) throw cause;
    invalid(diagnosticPath, `failed to inspect ${label} executable`, { cause });
  } finally {
    await handle?.close();
  }
}

async function inspectPrivateArchive(path, diagnosticPath) {
  if (await realpath(path).catch(() => undefined) !== path) {
    invalid(diagnosticPath, "archive path must be canonical and non-symlinked");
  }
  await admitPrivateCanonicalDirectory(dirname(path), `${diagnosticPath}.parent`);
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    requirePrivateArchiveStat(before, diagnosticPath);
    if (before.size > BigInt(MAX_ARCHIVE_BYTES)) resource(diagnosticPath, "archive exceeds byte ceiling");
    const sha = await hashOpenFile(handle, Number(before.size), diagnosticPath);
    const after = await handle.stat({ bigint: true });
    if (!sameSnapshot(before, after)) invalid(diagnosticPath, "archive changed while hashed");
    return Object.freeze({ path, sha256: sha, snapshot: fileSnapshot(after) });
  } catch (cause) {
    if (cause instanceof CppCuteBrowserArchiveNormalizationError) throw cause;
    invalid(diagnosticPath, "failed to inspect private archive", { cause });
  } finally {
    await handle?.close();
  }
}

async function verifyPrivateArchive(binding, diagnosticPath) {
  const current = await lstat(binding.path, { bigint: true });
  requirePrivateArchiveStat(current, diagnosticPath);
  if (!sameFileSnapshot(binding.snapshot, current)) invalid(diagnosticPath, "archive changed during normalization");
}

async function verifyTool(binding, diagnosticPath) {
  const current = await lstat(binding.path, { bigint: true });
  requireToolStat(current, diagnosticPath);
  if (!sameFileSnapshot(binding.snapshot, current)) invalid(diagnosticPath, "bsdtar executable changed during execution");
}

function requireToolStat(stat, diagnosticPath, label = "bsdtar") {
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size <= 0n || stat.nlink < 1n) {
    invalid(diagnosticPath, `expected one nonempty regular ${label} executable`);
  }
  if ((stat.mode & 0o111n) === 0n || (stat.mode & 0o6022n) !== 0n) {
    invalid(diagnosticPath, `${label} must be executable, non-setid, and not group/other writable`);
  }
}

function requirePrivateArchiveStat(stat, diagnosticPath) {
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size <= 0n || stat.nlink !== 1n) {
    invalid(diagnosticPath, "expected one nonempty singly-linked regular archive");
  }
  if (typeof process.getuid !== "function" || stat.uid !== BigInt(process.getuid()) ||
      (stat.mode & 0o077n) !== 0n) {
    invalid(diagnosticPath, "archive must be current-user owned and private");
  }
}

async function admitTrustedToolDirectory(path, diagnosticPath, label = "bsdtar") {
  const canonical = await realpath(path).catch(() => undefined);
  if (canonical !== path) invalid(diagnosticPath, `${label} parent must be canonical`);
  const stat = await lstat(path, { bigint: true });
  if (!stat.isDirectory() || stat.isSymbolicLink() || (stat.mode & 0o022n) !== 0n) {
    invalid(diagnosticPath, `${label} parent must be a non-writable trusted directory`);
  }
  if (typeof process.getuid !== "function" || (stat.uid !== 0n && stat.uid !== BigInt(process.getuid()))) {
    invalid(diagnosticPath, `${label} parent must be owned by root or the current user`);
  }
}

async function admitPrivateCanonicalDirectory(path, diagnosticPath) {
  const canonical = await realpath(path).catch(() => undefined);
  if (canonical !== path) invalid(diagnosticPath, "expected one canonical private directory");
  const stat = await lstat(path, { bigint: true });
  if (!stat.isDirectory() || stat.isSymbolicLink() ||
      typeof process.getuid !== "function" || stat.uid !== BigInt(process.getuid()) ||
      (stat.mode & 0o077n) !== 0n) {
    invalid(diagnosticPath, "expected one current-user-owned private directory");
  }
}

async function runBoundToolProcess(input) {
  await verifyTool(input.binding, `${input.diagnosticPath}.tool`);
  let child;
  try {
    child = spawn(input.binding.path, [...input.arguments], {
      cwd: dirname(input.binding.path),
      env: { ...CLOSED_ENVIRONMENT },
      detached: true,
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
  } catch (cause) {
    invalid(input.diagnosticPath, "failed to spawn observed bsdtar executable", { cause });
  }
  let spawnFailure;
  let terminationReason;
  let killTimer;
  child.once("error", (cause) => { spawnFailure = cause; });
  const closed = new Promise((resolve) => {
    child.once("close", (exitCode, signal) => resolve(Object.freeze({ exitCode, signal })));
  });
  const killGroup = (signal) => {
    if (child.pid === undefined) return;
    try {
      process.kill(-child.pid, signal);
    } catch {
      try { child.kill(signal); } catch { /* close remains the settlement boundary */ }
    }
  };
  const terminate = (reason) => {
    if (terminationReason !== undefined) return;
    terminationReason = reason;
    killGroup("SIGTERM");
    killTimer = setTimeout(() => killGroup("SIGKILL"), KILL_GRACE_MS);
    killTimer.unref();
  };
  const timeout = setTimeout(() => terminate("timeout"), PROCESS_TIMEOUT_MS);
  timeout.unref();
  const consumer = Promise.resolve()
    .then(() => input.consumeStdout(child.stdout))
    .catch((cause) => {
      terminate("stdout-consumer");
      throw cause;
    });
  const stderr = readBoundedChunks(
    child.stderr,
    MAX_STDERR_BYTES,
    `${input.diagnosticPath}.stderr`,
  ).catch((cause) => {
    terminate("stderr-limit");
    throw cause;
  });
  const [closedResult, consumerResult, stderrResult] = await Promise.all([
    closed,
    Promise.allSettled([consumer]),
    Promise.allSettled([stderr]),
  ]);
  clearTimeout(timeout);
  if (killTimer !== undefined) clearTimeout(killTimer);
  await verifyTool(input.binding, `${input.diagnosticPath}.tool`);
  const consumed = consumerResult[0];
  const capturedStderr = stderrResult[0];
  if (spawnFailure !== undefined) {
    invalid(input.diagnosticPath, "observed bsdtar executable failed to spawn", { cause: spawnFailure });
  }
  if (terminationReason === "timeout") resource(input.diagnosticPath, "bsdtar process exceeded timeout");
  if (terminationReason === "stderr-limit" || capturedStderr.status === "rejected") {
    invalid(input.diagnosticPath, "bsdtar stderr exceeded its bounded capture", {
      cause: capturedStderr.status === "rejected" ? capturedStderr.reason : undefined,
    });
  }
  const stderrBytes = capturedStderr.value;
  if (consumed.status === "rejected" && terminationReason === "stdout-consumer") {
    const processStatus = `exit=${closedResult.exitCode ?? "null"} signal=${closedResult.signal ?? "none"}`;
    const processDetail = stderrBytes.byteLength === 0
      ? processStatus
      : `${processStatus} stderr=${strictUtf8(stderrBytes, `${input.diagnosticPath}.stderr`).trimEnd()}`;
    const consumerDetail = consumed.reason instanceof Error ? consumed.reason.message : "unknown stream error";
    invalid(
      input.diagnosticPath,
      `bsdtar stdout failed strict materialization: ${consumerDetail}; process: ${processDetail}`,
      { cause: consumed.reason },
    );
  }
  if (closedResult.exitCode !== 0 || closedResult.signal !== null) {
    const detail = stderrBytes.byteLength === 0
      ? ""
      : `: ${strictUtf8(stderrBytes, `${input.diagnosticPath}.stderr`).trimEnd()}`;
    invalid(
      input.diagnosticPath,
      `bsdtar failed with exit=${closedResult.exitCode ?? "null"} signal=${closedResult.signal ?? "none"}${detail}`,
    );
  }
  if (consumed.status === "rejected") {
    invalid(input.diagnosticPath, "bsdtar stdout failed strict materialization", { cause: consumed.reason });
  }
  if (stderrBytes.byteLength !== 0) {
    invalid(input.diagnosticPath, "successful bsdtar execution produced unexpected stderr");
  }
  return Object.freeze({
    consumerResult: consumed.value,
    stderrSha256: sha256(stderrBytes),
    stderrByteLength: String(stderrBytes.byteLength),
  });
}

async function materializeRawStream(chunks, outputPath, maximumBytes, diagnosticPath) {
  let handle;
  let identity;
  try {
    handle = await open(
      outputPath,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o400,
    );
    const before = await handle.stat({ bigint: true });
    if (!before.isFile() || before.nlink !== 1n || before.size !== 0n) {
      invalid(diagnosticPath, "new raw stream output is not one empty regular file");
    }
    identity = Object.freeze({ dev: before.dev, ino: before.ino });
    const digest = createHash("sha256");
    let byteLength = 0;
    for await (const chunk of chunks) {
      const bytes = snapshotChunk(chunk, diagnosticPath);
      byteLength += bytes.byteLength;
      if (byteLength > maximumBytes) resource(diagnosticPath, "raw stream exceeds byte ceiling");
      digest.update(bytes);
      await writeAll(handle, bytes, byteLength - bytes.byteLength, diagnosticPath);
    }
    if (byteLength === 0) invalid(diagnosticPath, "raw stream is empty");
    await handle.sync();
    const after = await handle.stat({ bigint: true });
    if (after.dev !== before.dev || after.ino !== before.ino ||
        after.nlink !== 1n || after.size !== BigInt(byteLength)) {
      invalid(diagnosticPath, "raw stream output identity changed while written");
    }
    await handle.close();
    handle = undefined;
    const persisted = await lstat(outputPath, { bigint: true });
    if (!persisted.isFile() || persisted.isSymbolicLink() || persisted.nlink !== 1n ||
        persisted.dev !== before.dev || persisted.ino !== before.ino ||
        persisted.size !== BigInt(byteLength)) {
      invalid(diagnosticPath, "raw stream path no longer names the owned output");
    }
    return Object.freeze({ sha256: digest.digest("hex"), byteLength: String(byteLength) });
  } catch (cause) {
    await handle?.close().catch(() => {});
    if (identity !== undefined) await unlinkOwnedFile(outputPath, identity);
    if (cause instanceof CppCuteBrowserArchiveNormalizationError) throw cause;
    invalid(diagnosticPath, "failed to materialize raw process stream", { cause });
  }
}

async function decompressZstdFile(inputPath, outputPath, diagnosticPath) {
  let handle;
  let source;
  let decompressor;
  let piping;
  try {
    handle = await open(inputPath, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    requirePrivateArchiveStat(before, `${diagnosticPath}.input`);
    source = handle.createReadStream({ autoClose: false });
    decompressor = createZstdDecompress();
    piping = pipeline(source, decompressor);
    void piping.catch(() => {});
    const output = await materializeRawStream(
      decompressor,
      outputPath,
      MAX_INNER_TAR_BYTES,
      diagnosticPath,
    );
    await piping;
    const after = await handle.stat({ bigint: true });
    if (!sameSnapshot(before, after)) invalid(diagnosticPath, "compressed data member changed while read");
    return output;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserArchiveNormalizationError) throw cause;
    invalid(diagnosticPath, "failed to decompress the exact data.tar.zst member", { cause });
  } finally {
    source?.destroy();
    decompressor?.destroy();
    await piping?.catch(() => {});
    await handle?.close().catch(() => {});
  }
}

async function readBoundedChunks(chunks, maximumBytes, diagnosticPath) {
  const output = [];
  let byteLength = 0;
  for await (const chunk of chunks) {
    const bytes = snapshotChunk(chunk, diagnosticPath);
    byteLength += bytes.byteLength;
    if (byteLength > maximumBytes) resource(diagnosticPath, "process stream exceeds byte ceiling");
    output.push(Buffer.from(bytes));
  }
  return new Uint8Array(Buffer.concat(output, byteLength));
}

async function hashOpenFile(handle, byteLength, diagnosticPath) {
  const hash = createHash("sha256");
  const buffer = Buffer.allocUnsafe(Math.min(READ_BUFFER_BYTES, byteLength));
  let offset = 0;
  while (offset < byteLength) {
    const length = Math.min(buffer.byteLength, byteLength - offset);
    const { bytesRead } = await handle.read(buffer, 0, length, offset);
    if (bytesRead <= 0) invalid(diagnosticPath, "file became shorter while hashed");
    hash.update(buffer.subarray(0, bytesRead));
    offset += bytesRead;
  }
  return hash.digest("hex");
}

async function writeAll(handle, bytes, position, diagnosticPath) {
  let written = 0;
  while (written < bytes.byteLength) {
    const result = await handle.write(bytes, written, bytes.byteLength - written, position + written);
    if (result.bytesWritten <= 0) invalid(diagnosticPath, "output stopped accepting bytes");
    written += result.bytesWritten;
  }
}

function processReport(stageId, run) {
  return Object.freeze({
    stageId,
    stderrSha256: run.stderrSha256,
    stderrByteLength: run.stderrByteLength,
  });
}

function parseSelections(value, diagnosticPath) {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype ||
      value.length < 1 || value.length > MAX_SELECTIONS) {
    invalid(diagnosticPath, `expected a dense array with 1 to ${MAX_SELECTIONS} selections`);
  }
  const parsed = [];
  for (let index = 0; index < value.length; index += 1) {
    if (!Object.hasOwn(value, index)) invalid(`${diagnosticPath}[${index}]`, "sparse arrays are forbidden");
    const path = `${diagnosticPath}[${index}]`;
    const selection = exactObject(
      value[index],
      ["archiveSubtree", "outputSubdirectory", "selectionId"],
      path,
    );
    if (typeof selection.selectionId !== "string" || !SELECTION_ID.test(selection.selectionId)) {
      invalid(`${path}.selectionId`, "expected one portable selection ID");
    }
    if (typeof selection.archiveSubtree !== "string" ||
        !PORTABLE_PATH.test(selection.archiveSubtree)) {
      invalid(`${path}.archiveSubtree`, "expected one portable non-glob archive subtree");
    }
    if (typeof selection.outputSubdirectory !== "string" ||
        !/^[A-Za-z0-9._+@=-]+$/u.test(selection.outputSubdirectory)) {
      invalid(`${path}.outputSubdirectory`, "expected one portable output directory");
    }
    parsed.push(Object.freeze({
      selectionId: selection.selectionId,
      archiveSubtree: selection.archiveSubtree,
      outputSubdirectory: selection.outputSubdirectory,
    }));
  }
  parsed.sort((left, right) => compareUtf8(left.selectionId, right.selectionId));
  if (new Set(parsed.map(({ selectionId }) => selectionId)).size !== parsed.length ||
      new Set(parsed.map(({ archiveSubtree }) => archiveSubtree)).size !== parsed.length ||
      new Set(parsed.map(({ outputSubdirectory }) => outputSubdirectory)).size !== parsed.length) {
    invalid(diagnosticPath, "selection IDs, archive subtrees, and output directories must be unique");
  }
  for (const [index, left] of parsed.entries()) {
    for (const right of parsed.slice(index + 1)) {
      if (left.archiveSubtree.startsWith(`${right.archiveSubtree}/`) ||
          right.archiveSubtree.startsWith(`${left.archiveSubtree}/`)) {
        invalid(diagnosticPath, "archive selection subtrees must not overlap");
      }
    }
  }
  return Object.freeze(parsed);
}

function archiveFormatValue(value, diagnosticPath) {
  if (typeof value !== "string" || !ARCHIVE_FORMATS.has(value)) {
    invalid(diagnosticPath, "expected tar.gz, tar.xz, or deb-data-tar-zstd");
  }
  return value;
}

function fileSnapshot(stat) {
  return Object.freeze({
    dev: stat.dev,
    ino: stat.ino,
    size: stat.size,
    mode: stat.mode,
    uid: stat.uid,
    gid: stat.gid,
    nlink: stat.nlink,
    mtimeNs: stat.mtimeNs,
    ctimeNs: stat.ctimeNs,
  });
}

function sameSnapshot(left, right) {
  return sameFileSnapshot(fileSnapshot(left), right);
}

function sameFileSnapshot(left, right) {
  const current = "isFile" in right ? fileSnapshot(right) : right;
  return left.dev === current.dev && left.ino === current.ino && left.size === current.size &&
    left.mode === current.mode && left.uid === current.uid && left.gid === current.gid &&
    left.nlink === current.nlink && left.mtimeNs === current.mtimeNs && left.ctimeNs === current.ctimeNs;
}

async function unlinkOwnedFile(path, identity) {
  try {
    const current = await lstat(path, { bigint: true });
    if (current.dev === identity.dev && current.ino === identity.ino) await unlink(path);
  } catch { /* never unlink a path that no longer names the created inode */ }
}

async function removeOwnedRoot(path, identity) {
  try {
    const current = await lstat(path, { bigint: true });
    if (current.dev === identity.dev && current.ino === identity.ino) {
      await rm(path, { recursive: true });
    }
  } catch { /* never remove a path that no longer names the created directory */ }
}

function snapshotChunk(value, diagnosticPath) {
  if (!(value instanceof Uint8Array) || value.buffer instanceof SharedArrayBuffer) {
    invalid(diagnosticPath, "process chunks must be unshared Uint8Array values");
  }
  return new Uint8Array(value);
}

function strictUtf8(bytes, diagnosticPath) {
  try {
    return UTF8_DECODER.decode(bytes);
  } catch (cause) {
    invalid(diagnosticPath, "process output must be strict UTF-8", { cause });
  }
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

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function invalid(path, message, options) {
  throw new CppCuteBrowserArchiveNormalizationError(path, message, options);
}

async function main() {
  try {
    const argument = process.argv[2];
    const match = typeof argument === "string" ? /^--verify-pinned=(.+)$/u.exec(argument) : null;
    if (process.argv.length !== 3 || match === null) {
      invalid("$arguments", "expected exactly --verify-pinned=/absolute/bsdtar/path");
    }
    const environment = await admitPinnedCppCuteBrowserArchiveNormalizationEnvironment({
      executablePath: match[1],
    });
    process.stdout.write(`${JSON.stringify(environment)}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown normalization pin failure");
    const path = typeof cause === "object" && cause !== null && "path" in cause &&
      typeof cause.path === "string" ? ` at ${cause.path}` : "";
    process.stderr.write(`${error.name}${path}: ${error.message.replace(/[\r\n]+/gu, " ").slice(0, 2_048)}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}

function resource(path, message) {
  invalid(path, `resource limit: ${message}`);
}
