import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import { constants } from "node:fs";
import {
  chmod,
  lstat,
  mkdir,
  open,
  opendir,
  readFile,
  realpath,
  rm,
} from "node:fs/promises";
import { dirname, isAbsolute, join, normalize } from "node:path/posix";
import { TextDecoder } from "node:util";

import { hashCanonicalJson } from "@unlocalhosted/browsergrad-semantic-core/schema";

import { CppCuteBrowserBuildCacheReuseError, invalidateCachedCppCuteExtractorObjects, refreshCachedCppCuteToolchainOutputs } from "./cpp_cute_browser_build_cache_reuse.mjs";
import { planCppCuteClangWasmBuild } from "./cpp_cute_browser_build_plan.mjs";
import { reviewCppCuteBrowserConfiguredTarget } from "./cpp_cute_browser_configured_target_review.mjs";
import { CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS } from "./cpp_cute_browser_build_executor_fs.mjs";
import { createCppCuteBrowserBuildLogSink } from "./cpp_cute_browser_build_log_sink.mjs";
import {
  CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS,
  CppCuteBrowserBuildProcessError,
} from "./cpp_cute_browser_build_executor_process.mjs";
import {
  normalizeCppCuteBrowserBuildExecutorOptions,
} from "./cpp_cute_browser_build_executor_options.mjs";
import { CppCuteBrowserBuildExecutorError } from "./cpp_cute_browser_build_executor_error.mjs";
import { createCppCuteClangWasmBuildTimingRegistry } from "./cpp_cute_browser_build_timing.mjs";

export { CppCuteBrowserBuildExecutorError };

const CANCELLED = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CANCELLED";
const INVALID = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID";
const RESOURCE_LIMIT = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-RESOURCE-LIMIT";
const HASH_MISMATCH = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-HASH-MISMATCH";
const CONFLICT = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT";
const IO = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-IO";
const CLEANUP = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CLEANUP";
const UNVERIFIED = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-UNVERIFIED";
const BUILD_FAILED = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-BUILD-FAILED";
const SOURCE_SET_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-extractor-source-set.v1";
const MAX_SOURCE_FILE_COUNT = 64;
const MAX_SOURCE_TOTAL_BYTE_LENGTH = 1024 * 1024;
const MAX_WASM_SIDECAR_BYTE_LENGTH = 256 * 1024 * 1024;
const MAX_FACTORY_MODULE_BYTE_LENGTH = 32 * 1024 * 1024;
const MAX_LINK_MAP_BYTE_LENGTH = 128 * 1024 * 1024;
const MAX_NATIVE_TABLEGEN_BYTE_LENGTH = 256 * 1024 * 1024;
const MAX_STEP_OUTPUT_BYTE_LENGTH = 16 * 1024 * 1024;
const MAX_STEP_DURATION_MS = 4 * 60 * 60 * 1_000;
const HASH_BUFFER_BYTE_LENGTH = 1024 * 1024;
const PORTABLE_ABSOLUTE_PATH = /^\/[A-Za-z0-9._+/-]+$/u;
const WASM_HEADER = Uint8Array.of(
  0x00, 0x61, 0x73, 0x6d,
  0x01, 0x00, 0x00, 0x00,
);
const PREPARED_SOURCES = new WeakMap();
const EXECUTED_BUILDS = new WeakMap();
const EXECUTION_TIMINGS = createCppCuteClangWasmBuildTimingRegistry(unverified);
const MATERIALIZED_SIDECARS = new WeakMap();
const ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

/** @typedef {Readonly<{ dev: bigint; ino: bigint }>} FileIdentity */
/** @typedef {Readonly<{ dev: bigint; ino: bigint; ctimeNs: bigint; birthtimeNs: bigint }>} FileVersionIdentity */
/** @typedef {Readonly<{ path: string; identity: FileVersionIdentity }>} StagedFileIdentity */
/** @typedef {Readonly<{ root: FileIdentity; files: readonly StagedFileIdentity[] }>} StagedSourceIdentity */
/** @typedef {Readonly<{ path: string; identity: FileVersionIdentity; executable: boolean; diagnosticPath: string }>} RegularFileBinding */
/** @typedef {Readonly<{ path: string; identity: FileVersionIdentity; diagnosticPath: string }>} DirectoryBinding */

/**
 * Verifies the exact lock-owned extractor source closure, snapshots its bytes,
 * and stages only those snapshots. The returned opaque value grants no process,
 * output, reproducibility, provenance, or release authority.
 * The staging parent must be owned by the current build user and not writable
 * by group/other users. Same-uid actors must honor a single-writer boundary.
 *
 * @param {import("./cpp_cute_browser_build_executor.mjs").PrepareCppCuteClangWasmBuildSourceInput} input
 * @param {import("./cpp_cute_browser_build_executor.mjs").CppCuteBrowserBuildExecutorOptions} [options]
 * @returns {Promise<import("./cpp_cute_browser_build_executor.mjs").PreparedCppCuteClangWasmBuildSource>}
 */
export async function prepareCppCuteClangWasmBuildSource(input, options = {}) {
  const { signal } = normalizeCppCuteBrowserBuildExecutorOptions(options, false, invalid);
  throwIfAborted(signal);
  const selected = snapshotInput(input);
  let plan;
  try {
    plan = planCppCuteClangWasmBuild({
      lock: selected.lock,
      tools: selected.tools,
      roots: selected.roots,
    });
  } catch (cause) {
    invalid("$input", "build input could not produce the exact lock-derived plan", { cause });
  }
  throwIfAborted(signal);
  if (plan.extractorSource.files.length > MAX_SOURCE_FILE_COUNT) {
    resource(
      "$.extractorSource.files",
      `extractor source file count exceeds ${MAX_SOURCE_FILE_COUNT}`,
    );
  }

  let snapshots;
  try {
    snapshots = await snapshotSourceClosure(
      selected.extractorSourceInputRoot,
      plan.extractorSource.files,
      signal,
    );
  } catch (cause) {
    rethrowExecutorOrIo(cause, "$.extractorSourceInputRoot", "failed to snapshot extractor source closure");
  }
  throwIfAborted(signal);

  let observedSourceSetSha256;
  try {
    observedSourceSetSha256 = await hashCanonicalJson({
      domain: SOURCE_SET_HASH_DOMAIN,
      files: snapshots.map(({ path, sha256, byteLength }) => ({
        path,
        sha256,
        byteLength,
      })),
    });
  } catch (cause) {
    io("$.extractorSource.sourceSetSha256", "source-set hashing failed", { cause });
  }
  throwIfAborted(signal);
  if (observedSourceSetSha256 !== plan.extractorSource.sourceSetSha256) {
    hashMismatch(
      "$.extractorSource.sourceSetSha256",
      "observed extractor source set differs from the build lock",
    );
  }

  let stagedSourceIdentity;
  try {
    stagedSourceIdentity = await stageSourceSnapshots(
      selected.roots.extractorSourceRoot,
      snapshots,
      sourceDateEpochSeconds(plan),
      signal,
    );
  } catch (cause) {
    rethrowExecutorOrIo(cause, "$.roots.extractorSourceRoot", "failed to stage verified extractor snapshots");
  }
  await throwIfAbortedOrRemove(
    signal,
    selected.roots.extractorSourceRoot,
    stagedSourceIdentity.root,
  );

  const totalByteLength = snapshots.reduce(
    (total, snapshot) => total + snapshot.bytes.byteLength,
    0,
  );
  const prepared = Object.freeze({
    authority: "build-source-snapshot-only",
    lockId: plan.lockId,
    sourceSetSha256: observedSourceSetSha256,
    fileCount: snapshots.length,
    totalByteLength,
    stagedSourceRoot: selected.roots.extractorSourceRoot,
    sourceVerified: true,
    buildExecuted: false,
    outputIdentityAuthorized: false,
    reproducibilityVerified: false,
    releaseReady: false,
  });
  PREPARED_SOURCES.set(prepared, Object.freeze({
    plan,
    roots: Object.freeze({ ...selected.roots }),
    tools: Object.freeze({
      ...selected.tools,
      searchPath: Object.freeze([...selected.tools.searchPath]),
    }),
    stagedSourceIdentity,
    snapshots: Object.freeze(snapshots.map((snapshot) => Object.freeze({
      path: snapshot.path,
      sha256: snapshot.sha256,
      byteLength: snapshot.byteLength,
      bytes: new Uint8Array(snapshot.bytes),
    }))),
  }));
  return /** @type {import("./cpp_cute_browser_build_executor.mjs").PreparedCppCuteClangWasmBuildSource} */ (prepared);
}

/**
 * Executes the four exact lock-derived CMake steps without a shell, persists
 * bounded immutable stdout/stderr evidence, and validates the generated
 * factory, link map, and WebAssembly binary. This proves one build execution;
 * it does not prove reproducibility, ABI conformance, provenance, or release
 * readiness.
 *
 * @param {import("./cpp_cute_browser_build_executor.mjs").PreparedCppCuteClangWasmBuildSource} prepared
 * @param {import("./cpp_cute_browser_build_executor.mjs").CppCuteBrowserBuildExecutionOptions} [options]
 * @returns {Promise<import("./cpp_cute_browser_build_executor.mjs").ExecutedCppCuteClangWasmBuild>}
 */
export async function executeCppCuteClangWasmBuild(prepared, options = {}) {
  const buildTiming = EXECUTION_TIMINGS.beginBuild();
  const { signal, mirrorOutput, buildDirectoryPolicy } =
    normalizeCppCuteBrowserBuildExecutorOptions(options, true, invalid);
  throwIfAborted(signal);
  const stored = storedPreparedSource(prepared);
  const { plan, roots, tools } = stored;

  try {
    await verifyStagedSourceClosure(
      prepared.stagedSourceRoot,
      stored.snapshots,
      signal,
      stored.stagedSourceIdentity.root,
      stored.stagedSourceIdentity.files,
    );
  } catch (cause) {
    rethrowExecutorOrIo(cause, "$.stagedSourceRoot", "staged extractor source no longer matches its authority");
  }
  throwIfAborted(signal);

  let inputBindings;
  try {
    inputBindings = await snapshotBuildInputBindings(roots, tools);
    await prepareBuildExecutionDirectories(
      roots,
      plan,
      stored.snapshots,
      buildDirectoryPolicy,
    );
  } catch (cause) {
    rethrowExecutorOrIo(cause, "$.buildInputs", "failed to admit the materialized build inputs");
  }

  const evidenceRoot = join(roots.stateRoot, "evidence");
  const logRoot = join(evidenceRoot, "build-logs");
  const receipts = [];
  let nativeToolBindings;
  let configuredTargetReview;
  for (const [index, step] of plan.steps.entries()) {
    const phaseStartedAt = EXECUTION_TIMINGS.beginPhase();
    throwIfAborted(signal);
    await verifyBuildInputBindings(inputBindings);
    await verifyStagedSourceClosure(
      prepared.stagedSourceRoot,
      stored.snapshots,
      signal,
      stored.stagedSourceIdentity.root,
      stored.stagedSourceIdentity.files,
    );
    if (step.stageId === "clang-extractor-wasm") {
      if (nativeToolBindings === undefined) {
        unverified(`$.steps[${index}]`, "native TableGen tools were not admitted before the Wasm stage");
      }
      await verifyRegularFileBindings(nativeToolBindings);
    }
    if (step.id === "clang-extractor-wasm-build" && buildDirectoryPolicy === "reuse-untrusted-diagnostic")
      await refreshCachedCppCuteToolchainOutputs({ wasmBuildRoot: roots.wasmBuildRoot });
    if (step.id === "clang-extractor-wasm-build") {
      try {
        configuredTargetReview = await reviewCppCuteBrowserConfiguredTarget({
          wasmBuildRoot: roots.wasmBuildRoot,
          llvmProjectSourceRoot: roots.llvmProjectSourceRoot,
          factoryModulePath: plan.generatedExtractor.factoryModulePath,
        });
      } catch (cause) {
        invalid(
          "$.configuredTarget",
          "configured BrowserGrad target flags failed pre-build review",
          { cause },
        );
      }
    }

    const outputSinks = await openBuildLogSinks(
      logRoot,
      step.id,
      mirrorOutput,
      `$.steps[${index}]`,
    );
    let processResult;
    let processFailure;
    try {
      processResult = await CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS.run({
        executable: step.executable,
        arguments: step.arguments,
        cwd: step.cwd,
        environment: step.environment,
        maximumOutputByteLength: MAX_STEP_OUTPUT_BYTE_LENGTH,
        maximumDurationMs: MAX_STEP_DURATION_MS,
        onOutputChunk: async (stream, chunk) => {
          await outputSinks[stream].write(chunk);
        },
        ...(signal === undefined ? {} : { signal }),
      });
    } catch (cause) {
      processFailure = cause;
    }
    const { stdout: stdoutEvidence, stderr: stderrEvidence } = await sealBuildLogSinks(
      outputSinks,
      `$.steps[${index}]`,
    );
    if (processFailure !== undefined) {
      rethrowBuildProcessFailure(processFailure, `$.steps[${index}]`);
    }
    if (processResult.stdoutByteLength !== stdoutEvidence.byteLength ||
        processResult.stderrByteLength !== stderrEvidence.byteLength) {
      conflict(`$.steps[${index}].output`, "persisted build-log lengths differ from the process boundary");
    }
    const receipt = Object.freeze({
      id: step.id,
      stageId: step.stageId,
      kind: step.kind,
      executable: step.executable,
      arguments: Object.freeze([...step.arguments]),
      cwd: step.cwd,
      environment: Object.freeze({ ...step.environment }),
      exitCode: processResult.exitCode,
      terminationSignal: processResult.terminationSignal,
      stdoutPath: stdoutEvidence.path,
      stdoutSha256: stdoutEvidence.sha256,
      stdoutByteLength: stdoutEvidence.byteLength,
      stderrPath: stderrEvidence.path,
      stderrSha256: stderrEvidence.sha256,
      stderrByteLength: stderrEvidence.byteLength,
    });
    receipts.push(receipt);
    if (processResult.exitCode !== 0 || processResult.terminationSignal !== null) {
      buildFailed(
        `$.steps[${index}]`,
        `exact build step ${step.id} failed with exit ${String(processResult.exitCode)} and signal ${String(processResult.terminationSignal)}`,
      );
    }
    if (step.id === "native-tablegen-build") {
      nativeToolBindings = await snapshotRegularFileBindings([
        { path: plan.nativeTools.clangTablegen, executable: true, diagnosticPath: "$.nativeTools.clangTablegen" },
        { path: plan.nativeTools.llvmTablegen, executable: true, diagnosticPath: "$.nativeTools.llvmTablegen" },
      ]);
    }
    EXECUTION_TIMINGS.finishPhase(buildTiming, step, phaseStartedAt);
  }
  if (receipts.length !== 4 || nativeToolBindings === undefined ||
      configuredTargetReview === undefined) {
    unverified("$.steps", "exact four-step build did not complete");
  }

  let completedTargetReview;
  try {
    completedTargetReview = await reviewCppCuteBrowserConfiguredTarget({
      wasmBuildRoot: roots.wasmBuildRoot,
      llvmProjectSourceRoot: roots.llvmProjectSourceRoot,
      factoryModulePath: plan.generatedExtractor.factoryModulePath,
    });
  } catch (cause) {
    invalid(
      "$.configuredTarget",
      "configured BrowserGrad target flags failed post-build review",
      { cause },
    );
  }
  if (configuredTargetReview.compileFlagsSha256 !== completedTargetReview.compileFlagsSha256 ||
      configuredTargetReview.linkCommandSha256 !== completedTargetReview.linkCommandSha256) {
    conflict(
      "$.configuredTarget",
      "configured target flags changed between pre-build and post-build review",
    );
  }

  await verifyBuildInputBindings(inputBindings);
  await verifyRegularFileBindings(nativeToolBindings);
  const nativeTools = Object.freeze({
    clangTablegen: await observeBoundRegularFile(
      nativeToolBindings[0],
      MAX_NATIVE_TABLEGEN_BYTE_LENGTH,
      signal,
    ),
    llvmTablegen: await observeBoundRegularFile(
      nativeToolBindings[1],
      MAX_NATIVE_TABLEGEN_BYTE_LENGTH,
      signal,
    ),
  });
  await verifyStagedSourceClosure(
    prepared.stagedSourceRoot,
    stored.snapshots,
    signal,
    stored.stagedSourceIdentity.root,
    stored.stagedSourceIdentity.files,
  );
  throwIfAborted(signal);

  const generatedRoot = join(evidenceRoot, "generated");
  await assertExactSourceTree(
    generatedRoot,
    ["clang-extractor.mjs", "clang-extractor.wasm"],
    "$.generatedExtractor",
  );
  const factory = await sealGeneratedFile(
    plan.generatedExtractor.factoryModulePath,
    MAX_FACTORY_MODULE_BYTE_LENGTH,
    "$.generatedExtractor.factoryModulePath",
    signal,
  );
  validateFactoryModule(factory.bytes);
  const wasm = await sealGeneratedFile(
    plan.generatedExtractor.wasmSidecarPath,
    MAX_WASM_SIDECAR_BYTE_LENGTH,
    "$.generatedExtractor.wasmSidecarPath",
    signal,
  );
  if (!startsWithBytes(wasm.bytes, WASM_HEADER) || !WebAssembly.validate(wasm.bytes)) {
    invalid("$.generatedExtractor.wasmSidecarPath", "generated sidecar is not valid WebAssembly v1");
  }
  const linkMap = await sealGeneratedFile(
    join(evidenceRoot, "clang-extractor.link.map"),
    MAX_LINK_MAP_BYTE_LENGTH,
    "$.buildEvidence.linkMap",
    signal,
  );
  throwIfAborted(signal);

  const executed = Object.freeze({
    authority: "clang-wasm-build-execution-observation-only",
    lockId: plan.lockId,
    sourceSetSha256: prepared.sourceSetSha256,
    paths: Object.freeze({
      llvmProjectSourceRoot: roots.llvmProjectSourceRoot,
      extractorSourceRoot: roots.extractorSourceRoot,
      nativeBuildRoot: roots.nativeBuildRoot,
      wasmBuildRoot: roots.wasmBuildRoot,
      outputRoot: roots.outputRoot,
      stateRoot: roots.stateRoot,
    }),
    nativeTools,
    stepCount: receipts.length,
    steps: Object.freeze(receipts),
    factoryModulePath: plan.generatedExtractor.factoryModulePath,
    factoryModuleSha256: factory.sha256,
    factoryModuleByteLength: factory.byteLength,
    wasmSidecarPath: plan.generatedExtractor.wasmSidecarPath,
    wasmSha256: wasm.sha256,
    wasmByteLength: wasm.byteLength,
    linkMapPath: linkMap.path,
    linkMapSha256: linkMap.sha256,
    linkMapByteLength: linkMap.byteLength,
    sourceVerified: true,
    buildExecuted: true,
    factoryModuleUtf8Validated: true,
    webAssemblyValidated: true,
    abiConformanceVerified: false,
    outputIdentityAuthorized: false,
    reproducibilityVerified: false,
    releaseReady: false,
    factoryModuleDistributed: false,
  });
  EXECUTED_BUILDS.set(executed, Object.freeze({
    prepared,
    factoryIdentity: factory.identity,
    wasmIdentity: wasm.identity,
    linkMapIdentity: linkMap.identity,
  }));
  EXECUTION_TIMINGS.attach(buildTiming, executed);
  return /** @type {import("./cpp_cute_browser_build_executor.mjs").ExecutedCppCuteClangWasmBuild} */ (executed);
}

export function observeCppCuteClangWasmBuildTiming(executed) {
  return EXECUTION_TIMINGS.observe(executed);
}

/**
 * @param {RegularFileBinding} binding
 * @param {number} maximumByteLength
 * @param {AbortSignal | undefined} signal
 */
async function observeBoundRegularFile(binding, maximumByteLength, signal) {
  let handle;
  try {
    handle = await open(binding.path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!before.isFile() || before.size <= 0n ||
        !sameFileVersionIdentity(fileVersionIdentity(before), binding.identity)) {
      conflict(binding.diagnosticPath, "bound build tool changed before its identity was observed");
    }
    if (before.size > BigInt(maximumByteLength)) {
      resource(binding.diagnosticPath, `build tool exceeds ${maximumByteLength} bytes`);
    }
    const hash = createHash("sha256");
    const buffer = new Uint8Array(HASH_BUFFER_BYTE_LENGTH);
    let offset = 0;
    const byteLength = Number(before.size);
    while (offset < byteLength) {
      throwIfAborted(signal);
      const maximum = Math.min(buffer.byteLength, byteLength - offset);
      const { bytesRead } = await handle.read(buffer, 0, maximum, offset);
      if (bytesRead === 0) {
        conflict(binding.diagnosticPath, "bound build tool became shorter while hashing");
      }
      hash.update(buffer.subarray(0, bytesRead));
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileSnapshot(before, after) ||
        !sameFileVersionIdentity(fileVersionIdentity(after), binding.identity)) {
      conflict(binding.diagnosticPath, "bound build tool changed while its identity was observed");
    }
    return Object.freeze({
      path: binding.path,
      sha256: hash.digest("hex"),
      byteLength,
    });
  } catch (cause) {
    rethrowExecutorOrIo(
      cause,
      binding.diagnosticPath,
      "failed to observe the bound build-tool identity",
    );
  } finally {
    await handle?.close();
  }
}

/**
 * Copies the exact generated Wasm sidecar to the lock-derived distributed path.
 * The operation is atomic, no-clobber, and idempotent for identical bytes. It
 * does not execute the build or authorize output identity, reproducibility, or
 * release.
 * Output roots must be owned by the current build user and not writable by
 * group/other users. The atomic hard-link is the cancellation commit point;
 * cancellation observed before it leaves no destination, while a committed
 * identical destination is reported as success. Same-uid actors must honor a
 * single-writer boundary because Node exposes no openat/linkat authority.
 *
 * @param {import("./cpp_cute_browser_build_executor.mjs").PreparedCppCuteClangWasmBuildSource} prepared
 * @param {import("./cpp_cute_browser_build_executor.mjs").CppCuteBrowserBuildExecutorOptions} [options]
 * @returns {Promise<import("./cpp_cute_browser_build_executor.mjs").MaterializedCppCuteClangWasmSidecar>}
 */
export async function materializeCppCuteClangWasmSidecar(prepared, options = {}) {
  const { signal } = normalizeCppCuteBrowserBuildExecutorOptions(options, false, invalid);
  throwIfAborted(signal);
  const stored = storedPreparedSource(prepared);
  const { plan } = stored;
  if (plan.generatedExtractor.distributedWasmPath === plan.generatedExtractor.factoryModulePath ||
      !plan.generatedExtractor.wasmSidecarPath.endsWith("/clang-extractor.wasm") ||
      !plan.generatedExtractor.distributedWasmPath.endsWith("/browsergrad-cpp-cute/clang-extractor.wasm")) {
    invalid("$.generatedExtractor", "lock-derived generated/distributed sidecar paths are inconsistent");
  }

  try {
    await verifyStagedSourceClosure(
      prepared.stagedSourceRoot,
      stored.snapshots,
      signal,
      stored.stagedSourceIdentity.root,
      stored.stagedSourceIdentity.files,
    );
  } catch (cause) {
    rethrowExecutorOrIo(cause, "$.stagedSourceRoot", "staged extractor source no longer matches its authority");
  }
  throwIfAborted(signal);

  let generatedBytes;
  try {
    generatedBytes = await readBoundedRegularFile(
      plan.generatedExtractor.wasmSidecarPath,
      MAX_WASM_SIDECAR_BYTE_LENGTH,
      "$.generatedWasmPath",
      signal,
    );
  } catch (cause) {
    rethrowExecutorOrIo(cause, "$.generatedWasmPath", "failed to read generated Wasm sidecar");
  }
  if (!startsWithBytes(generatedBytes, WASM_HEADER)) {
    invalid(
      "$.generatedWasmPath",
      "generated sidecar is not a WebAssembly v1 binary",
    );
  }
  throwIfAborted(signal);
  const generatedWasmSha256 = sha256(generatedBytes);

  try {
    await installSidecarNoClobber(
      plan.outputRoot,
      plan.generatedExtractor.distributedWasmPath,
      generatedBytes,
      generatedWasmSha256,
      signal,
    );
  } catch (cause) {
    rethrowExecutorOrIo(cause, "$.distributedWasmPath", "failed to materialize Wasm sidecar");
  }

  const materialized = Object.freeze({
    authority: "wasm-sidecar-byte-materialization-observation-only",
    lockId: plan.lockId,
    sourceSetSha256: prepared.sourceSetSha256,
    generatedWasmSha256,
    distributedWasmSha256: generatedWasmSha256,
    wasmByteLength: generatedBytes.byteLength,
    distributedWasmPath: plan.generatedExtractor.distributedWasmPath,
    sidecarBytesMaterialized: true,
    webAssemblyValidated: false,
    abiConformanceVerified: false,
    sourceVerified: true,
    buildExecuted: false,
    outputIdentityAuthorized: false,
    reproducibilityVerified: false,
    releaseReady: false,
    factoryModuleDistributed: false,
  });
  MATERIALIZED_SIDECARS.set(materialized, Object.freeze({ prepared }));
  return /** @type {import("./cpp_cute_browser_build_executor.mjs").MaterializedCppCuteClangWasmSidecar} */ (materialized);
}

/**
 * @param {import("./cpp_cute_browser_build_plan.mjs").CppCuteClangWasmBuildRoots} roots
 * @param {import("./cpp_cute_browser_build_plan.mjs").CppCuteClangWasmMaterializedTools} tools
 */
async function snapshotBuildInputBindings(roots, tools) {
  const files = await snapshotRegularFileBindings([
    { path: tools.cmakeExecutable, executable: true, diagnosticPath: "$.tools.cmakeExecutable" },
    { path: tools.buildToolExecutable, executable: true, diagnosticPath: "$.tools.buildToolExecutable" },
    { path: tools.emscriptenToolchainFile, executable: false, diagnosticPath: "$.tools.emscriptenToolchainFile" },
    { path: tools.emscriptenConfigFile, executable: false, diagnosticPath: "$.tools.emscriptenConfigFile" },
  ]);
  const directorySpecs = [
    { path: roots.llvmProjectSourceRoot, diagnosticPath: "$.roots.llvmProjectSourceRoot" },
    { path: join(roots.llvmProjectSourceRoot, "llvm"), diagnosticPath: "$.roots.llvmProjectSourceRoot/llvm" },
    { path: tools.emsdkRoot, diagnosticPath: "$.tools.emsdkRoot" },
    ...tools.searchPath.map((path, index) => ({
      path,
      diagnosticPath: `$.tools.searchPath[${index}]`,
    })),
  ];
  /** @type {Map<string, { path: string; diagnosticPath: string }>} */
  const uniqueDirectorySpecs = new Map();
  for (const spec of directorySpecs) uniqueDirectorySpecs.set(spec.path, spec);
  const directories = await snapshotDirectoryBindings([...uniqueDirectorySpecs.values()]);
  return Object.freeze({ files, directories });
}

/** @param {Readonly<{ files: readonly RegularFileBinding[]; directories: readonly DirectoryBinding[] }>} bindings */
async function verifyBuildInputBindings(bindings) {
  await verifyRegularFileBindings(bindings.files);
  for (const binding of bindings.directories) {
    const stat = await lstat(binding.path, { bigint: true });
    if (!stat.isDirectory() || stat.isSymbolicLink() ||
        !sameFileVersionIdentity(fileVersionIdentity(stat), binding.identity)) {
      conflict(binding.diagnosticPath, "build-input directory identity changed during execution");
    }
    await assertTrustedOwnerAndMode(stat, binding.path, binding.diagnosticPath);
  }
}

/**
 * @param {readonly { readonly path: string; readonly executable: boolean; readonly diagnosticPath: string }[]} specs
 * @returns {Promise<readonly RegularFileBinding[]>}
 */
async function snapshotRegularFileBindings(specs) {
  const bindings = [];
  for (const spec of specs) {
    const stat = await lstat(spec.path, { bigint: true });
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size <= 0n) {
      invalid(spec.diagnosticPath, "expected a nonempty non-symlink regular file");
    }
    await assertTrustedOwnerAndMode(stat, spec.path, spec.diagnosticPath);
    if (spec.executable && (stat.mode & 0o111n) === 0n) {
      invalid(spec.diagnosticPath, "build tool must have an executable mode bit");
    }
    if ((stat.mode & 0o6000n) !== 0n) {
      invalid(spec.diagnosticPath, "build input must not carry setuid or setgid mode bits");
    }
    await assertTrustedBuildInputAncestry(dirname(spec.path), spec.diagnosticPath);
    bindings.push(Object.freeze({
      ...spec,
      identity: fileVersionIdentity(stat),
    }));
  }
  return Object.freeze(bindings);
}

/** @param {readonly RegularFileBinding[]} bindings */
async function verifyRegularFileBindings(bindings) {
  for (const binding of bindings) {
    const stat = await lstat(binding.path, { bigint: true });
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size <= 0n ||
        !sameFileVersionIdentity(fileVersionIdentity(stat), binding.identity)) {
      conflict(binding.diagnosticPath, "build-input file identity changed during execution");
    }
    await assertTrustedOwnerAndMode(stat, binding.path, binding.diagnosticPath);
    if (binding.executable && (stat.mode & 0o111n) === 0n) {
      conflict(binding.diagnosticPath, "build tool lost its executable mode during execution");
    }
    if ((stat.mode & 0o6000n) !== 0n) {
      conflict(binding.diagnosticPath, "build input acquired setuid or setgid mode bits");
    }
  }
}

/**
 * @param {readonly { readonly path: string; readonly diagnosticPath: string }[]} specs
 * @returns {Promise<readonly DirectoryBinding[]>}
 */
async function snapshotDirectoryBindings(specs) {
  const bindings = [];
  for (const spec of specs) {
    const stat = await lstat(spec.path, { bigint: true });
    if (!stat.isDirectory() || stat.isSymbolicLink()) {
      invalid(spec.diagnosticPath, "expected a non-symlink directory");
    }
    await assertTrustedOwnerAndMode(stat, spec.path, spec.diagnosticPath);
    await assertTrustedBuildInputAncestry(spec.path, spec.diagnosticPath);
    bindings.push(Object.freeze({
      ...spec,
      identity: fileVersionIdentity(stat),
    }));
  }
  return Object.freeze(bindings);
}

/** @param {import("node:fs").BigIntStats} stat @param {string} path @param {string} diagnosticPath */
async function assertTrustedOwnerAndMode(stat, path, diagnosticPath) {
  if (typeof process.getuid !== "function") {
    invalid(diagnosticPath, "build-input ownership requires POSIX uid support");
  }
  const currentUid = BigInt(process.getuid());
  if (stat.uid !== 0n && stat.uid !== currentUid) {
    await assertPathOnReadOnlyLinuxMount(path, diagnosticPath);
  }
  if ((stat.mode & 0o022n) !== 0n) {
    await assertPathOnReadOnlyLinuxMount(path, diagnosticPath);
  }
}

/**
 * Pinned OCI toolchains commonly retain their image-build uid. Such inputs are
 * admitted only when the kernel reports their effective mount read-only; a
 * mode-bit-only or caller-declared immutability claim is insufficient.
 *
 * @param {string} path
 * @param {string} diagnosticPath
 */
async function assertPathOnReadOnlyLinuxMount(path, diagnosticPath) {
  let mountInfo;
  try {
    mountInfo = await readFile("/proc/self/mountinfo");
  } catch (cause) {
    invalid(
      diagnosticPath,
      "foreign-owned build input requires a kernel-observed read-only Linux mount",
      { cause },
    );
  }
  if (mountInfo.byteLength === 0 || mountInfo.byteLength > 4 * 1024 * 1024) {
    resource(`${diagnosticPath}.mountInfo`, "Linux mountinfo length is outside the admitted range");
  }
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(mountInfo);
  } catch (cause) {
    invalid(`${diagnosticPath}.mountInfo`, "Linux mountinfo is not valid UTF-8", { cause });
  }
  let selectedMountPoint;
  let selectedOptions;
  for (const [index, line] of text.trim().split("\n").entries()) {
    const fields = line.split(" ");
    const separator = fields.indexOf("-");
    if (separator < 6 || fields[4] === undefined || fields[5] === undefined) {
      invalid(`${diagnosticPath}.mountInfo[${index}]`, "malformed Linux mountinfo record");
    }
    const mountPoint = decodeMountInfoPath(fields[4]);
    if ((path === mountPoint ||
         path.startsWith(`${mountPoint === "/" ? "" : mountPoint}/`)) &&
        (selectedMountPoint === undefined || mountPoint.length > selectedMountPoint.length)) {
      selectedMountPoint = mountPoint;
      selectedOptions = fields[5].split(",");
    }
  }
  if (selectedMountPoint === undefined || selectedOptions === undefined ||
      !selectedOptions.includes("ro")) {
    invalid(
      diagnosticPath,
      "foreign-owned build input is not protected by a kernel-observed read-only mount",
    );
  }
}

/** @param {string} path @param {string} diagnosticPath */
async function assertTrustedBuildInputAncestry(path, diagnosticPath) {
  try {
    await assertTrustedPosixAncestry(path, diagnosticPath);
  } catch (cause) {
    if (!(cause instanceof CppCuteBrowserBuildExecutorError) || cause.code !== INVALID) {
      throw cause;
    }
    await assertPathOnReadOnlyLinuxMount(path, diagnosticPath);
  }
}

/** @param {string} value */
function decodeMountInfoPath(value) {
  return value
    .replaceAll("\\040", " ")
    .replaceAll("\\011", "\t")
    .replaceAll("\\012", "\n")
    .replaceAll("\\134", "\\");
}

/**
 * @param {import("./cpp_cute_browser_build_plan.mjs").CppCuteClangWasmBuildRoots} roots
 * @param {import("./cpp_cute_browser_build_plan.mjs").CppCuteClangWasmBuildPlan} plan
 */
async function prepareBuildExecutionDirectories(
  roots,
  plan,
  snapshots,
  buildDirectoryPolicy,
) {
  const evidenceRoot = join(roots.stateRoot, "evidence");
  const generatedRoot = join(evidenceRoot, "generated");
  const logRoot = join(evidenceRoot, "build-logs");
  if (plan.generatedExtractor.factoryModulePath !== join(generatedRoot, "clang-extractor.mjs") ||
      plan.generatedExtractor.wasmSidecarPath !== join(generatedRoot, "clang-extractor.wasm")) {
    invalid("$.generatedExtractor", "generated output paths escaped the exact evidence root");
  }
  await assertPrivateOwnedDirectory(roots.outputRoot, "$.roots.outputRoot");
  for (const [path, diagnosticPath] of [
    [roots.nativeBuildRoot, "$.roots.nativeBuildRoot"],
    [roots.wasmBuildRoot, "$.roots.wasmBuildRoot"],
  ]) {
    if (buildDirectoryPolicy === "reuse-untrusted-diagnostic") {
      await createOrReusePrivateBuildDirectory(path, diagnosticPath);
    } else {
      await createPrivateEmptyDirectory(path, diagnosticPath);
    }
  }
  if (buildDirectoryPolicy === "reuse-untrusted-diagnostic") {
    try {
      await invalidateCachedCppCuteExtractorObjects({
        wasmBuildRoot: roots.wasmBuildRoot,
        sourcePaths: snapshots.map((snapshot) => snapshot.path),
      });
    } catch (cause) {
      if (cause instanceof CppCuteBrowserBuildCacheReuseError) {
        conflict(`$.cachedExtractorObjects.${cause.path}`, cause.message);
      }
      throw cause;
    }
  }
  await createPrivateEmptyDirectory(roots.stateRoot, "$.roots.stateRoot");
  await createPrivateEmptyDirectory(evidenceRoot, "$.buildEvidence");
  await createPrivateEmptyDirectory(generatedRoot, "$.generatedExtractor");
  await createPrivateEmptyDirectory(logRoot, "$.buildEvidence.logs");
  await createPrivateEmptyDirectory(join(roots.stateRoot, "home"), "$.buildState.home");
  await createPrivateEmptyDirectory(join(roots.stateRoot, "tmp"), "$.buildState.tmp");
  await createPrivateEmptyDirectory(
    join(roots.stateRoot, "empty-pkgconfig"),
    "$.buildState.emptyPkgConfig",
  );
  await createPrivateEmptyDirectory(join(roots.stateRoot, "em-cache"), "$.buildState.emCache");
}

/** @param {string} path @param {string} diagnosticPath */
async function createOrReusePrivateBuildDirectory(path, diagnosticPath) {
  await assertPrivateOwnedDirectory(dirname(path), `${diagnosticPath}.parent`);
  if (await lstatIfExists(path, diagnosticPath) === undefined) {
    await createPrivateEmptyDirectory(path, diagnosticPath);
    return;
  }
  await assertPrivateOwnedDirectory(path, diagnosticPath);
}

/** @param {string} path @param {string} diagnosticPath */
async function createPrivateEmptyDirectory(path, diagnosticPath) {
  await assertPrivateOwnedDirectory(dirname(path), `${diagnosticPath}.parent`);
  if (await lstatIfExists(path, diagnosticPath) !== undefined) {
    conflict(diagnosticPath, "clean build directory must not already exist");
  }
  try {
    await mkdir(path, { mode: 0o700 });
  } catch (cause) {
    if (isNodeError(cause, "EEXIST")) {
      conflict(diagnosticPath, "build directory was concurrently created");
    }
    throw cause;
  }
  await assertPrivateOwnedDirectory(path, diagnosticPath);
  await syncDirectory(dirname(path));
}

/** @param {string} logRoot @param {string} stepId @param {boolean} mirrorOutput @param {string} path */
async function openBuildLogSinks(logRoot, stepId, mirrorOutput, path) {
  let stdout;
  let stderr;
  try {
    stdout = await createCppCuteBrowserBuildLogSink({
      path: join(logRoot, `${stepId}.stdout.log`),
      maximumByteLength: MAX_STEP_OUTPUT_BYTE_LENGTH,
      mirror: mirrorOutput ? "stdout" : null,
    });
    stderr = await createCppCuteBrowserBuildLogSink({
      path: join(logRoot, `${stepId}.stderr.log`),
      maximumByteLength: MAX_STEP_OUTPUT_BYTE_LENGTH,
      mirror: mirrorOutput ? "stderr" : null,
    });
    await syncDirectory(logRoot);
    return Object.freeze({ stdout, stderr });
  } catch (cause) {
    const cleanupResults = await Promise.allSettled([
      ...(stdout === undefined ? [] : [stdout.seal()]),
      ...(stderr === undefined ? [] : [stderr.seal()]),
      ...((stdout === undefined && stderr === undefined) ? [] : [syncDirectory(logRoot)]),
    ]);
    const cleanupFailures = cleanupResults
      .filter((result) => result.status === "rejected")
      .map((result) => result.reason);
    if (cleanupFailures.length > 0) {
      compositeCleanup(
        `${path}.output`,
        "failed to settle partially opened build-log evidence",
        cause,
        cleanupFailures,
      );
    }
    if (isNodeError(cause, "EEXIST") || isNodeError(cause, "ELOOP")) {
      conflict(`${path}.output`, "build-log evidence path must not already exist");
    }
    io(`${path}.output`, "failed to open exclusive build-log evidence", { cause });
  }
}

/** @param {Awaited<ReturnType<typeof openBuildLogSinks>>} sinks @param {string} path */
async function sealBuildLogSinks(sinks, path) {
  const settled = await Promise.allSettled([sinks.stdout.seal(), sinks.stderr.seal()]);
  const failures = settled
    .filter((result) => result.status === "rejected")
    .map((result) => result.reason);
  if (failures.length > 0) {
    io(`${path}.output`, "failed to seal streamed build-log evidence", {
      cause: new AggregateError(failures, "build-log sealing failures"),
    });
  }
  const stdout = settled[0].value;
  const stderr = settled[1].value;
  await assertReadOnlyRegularPathIdentity(
    stdout.path,
    stdout.identity,
    stdout.byteLength,
    `${path}.stdout`,
  );
  await assertReadOnlyRegularPathIdentity(
    stderr.path,
    stderr.identity,
    stderr.byteLength,
    `${path}.stderr`,
  );
  await syncDirectory(dirname(stdout.path));
  return Object.freeze({ stdout, stderr });
}

/**
 * @param {string} path
 * @param {number} maximumByteLength
 * @param {string} diagnosticPath
 * @param {AbortSignal | undefined} signal
 */
async function sealGeneratedFile(path, maximumByteLength, diagnosticPath, signal) {
  const before = await readBoundedRegularFileSnapshot(
    path,
    maximumByteLength,
    diagnosticPath,
    signal,
  );
  const beforeIdentity = fileIdentity(before.stat);
  if (before.stat.nlink !== 1n) {
    conflict(diagnosticPath, "generated output must have exactly one hard link");
  }
  await chmod(path, 0o444);
  const after = await readBoundedRegularFileSnapshot(
    path,
    maximumByteLength,
    diagnosticPath,
    signal,
  );
  if (!sameFileIdentity(fileIdentity(after.stat), beforeIdentity) ||
      after.stat.nlink !== 1n || !equalBytes(before.bytes, after.bytes)) {
    conflict(diagnosticPath, "generated output identity changed while it was being sealed");
  }
  const identity = fileVersionIdentity(after.stat);
  await assertReadOnlyRegularPathIdentity(
    path,
    identity,
    after.bytes.byteLength,
    diagnosticPath,
  );
  await syncDirectory(dirname(path));
  return Object.freeze({
    path,
    bytes: after.bytes,
    sha256: sha256(after.bytes),
    byteLength: after.bytes.byteLength,
    identity,
  });
}

/** @param {Uint8Array} bytes */
function validateFactoryModule(bytes) {
  let source;
  try {
    source = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (cause) {
    invalid("$.generatedExtractor.factoryModulePath", "factory module is not valid UTF-8", { cause });
  }
  if (containsForbiddenTextControl(source) ||
      !source.includes("createBrowserGradCppCuteExtractor")) {
    invalid(
      "$.generatedExtractor.factoryModulePath",
      "factory module lacks the exact Emscripten export or contains forbidden control bytes",
    );
  }
}

/** @param {string} value */
function containsForbiddenTextControl(value) {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if ((code >= 0 && code <= 8) || code === 11 || code === 12 ||
        (code >= 14 && code <= 31) || code === 127) return true;
  }
  return false;
}

/** @param {unknown} cause @param {string} path */
function rethrowBuildProcessFailure(cause, path) {
  if (cause instanceof CppCuteBrowserBuildProcessError) {
    if (cause.reason === "cancelled") cancelled();
    if (cause.reason === "output-limit") {
      resource(`${path}.output`, "build step exceeded its bounded output budget");
    }
    if (cause.reason === "timeout") {
      resource(`${path}.duration`, "build step exceeded its bounded execution duration");
    }
    if (cause.reason === "output-sink") {
      io(`${path}.output`, "failed to persist streamed build-step output", { cause });
    }
    io(path, "failed to spawn the exact build step", { cause });
  }
  rethrowExecutorOrIo(cause, path, "build process boundary failed");
}

/**
 * @param {unknown} input
 * @returns {import("./cpp_cute_browser_build_executor.mjs").PrepareCppCuteClangWasmBuildSourceInput}
 */
function snapshotInput(input) {
  const descriptors = exactDataObject(
    input,
    ["lock", "tools", "roots", "extractorSourceInputRoot"],
    "$input",
  );
  const tools = snapshotStringRecord(
    dataValue(descriptors, "tools", "$input"),
    [
      "cmakeExecutable",
      "buildToolExecutable",
      "emsdkRoot",
      "emscriptenToolchainFile",
      "emscriptenConfigFile",
      "searchPath",
    ],
    "$input.tools",
  );
  const toolDescriptors = exactDataObject(
    dataValue(descriptors, "tools", "$input"),
    [
      "cmakeExecutable",
      "buildToolExecutable",
      "emsdkRoot",
      "emscriptenToolchainFile",
      "emscriptenConfigFile",
      "searchPath",
    ],
    "$input.tools",
  );
  const searchPathValue = dataValue(toolDescriptors, "searchPath", "$input.tools");
  if (!Array.isArray(searchPathValue)) invalid("$input.tools.searchPath", "expected an array");
  const searchPath = searchPathValue.map((entry, index) => (
    pathString(entry, `$input.tools.searchPath[${index}]`)
  ));
  const rootValue = dataValue(descriptors, "roots", "$input");
  const roots = snapshotStringRecord(
    rootValue,
    [
      "llvmProjectSourceRoot",
      "extractorSourceRoot",
      "nativeBuildRoot",
      "wasmBuildRoot",
      "outputRoot",
      "stateRoot",
    ],
    "$input.roots",
  );
  return {
    lock: /** @type {import("../../dist/cpp_cute_browser_build_lock.js").PreparedCppCuteBrowserBuildInputLock} */ (
      dataValue(descriptors, "lock", "$input")
    ),
    tools: {
      cmakeExecutable: tools.cmakeExecutable,
      buildToolExecutable: tools.buildToolExecutable,
      emsdkRoot: tools.emsdkRoot,
      emscriptenToolchainFile: tools.emscriptenToolchainFile,
      emscriptenConfigFile: tools.emscriptenConfigFile,
      searchPath: Object.freeze(searchPath),
    },
    roots: {
      llvmProjectSourceRoot: roots.llvmProjectSourceRoot,
      extractorSourceRoot: roots.extractorSourceRoot,
      nativeBuildRoot: roots.nativeBuildRoot,
      wasmBuildRoot: roots.wasmBuildRoot,
      outputRoot: roots.outputRoot,
      stateRoot: roots.stateRoot,
    },
    extractorSourceInputRoot: pathString(
      dataValue(descriptors, "extractorSourceInputRoot", "$input"),
      "$input.extractorSourceInputRoot",
    ),
  };
}

/**
 * @param {string} sourceRoot
 * @param {readonly import("./cpp_cute_browser_build_plan.mjs").CppCuteClangWasmExtractorSourceFile[]} expectedFiles
 * @param {AbortSignal | undefined} signal
 */
async function snapshotSourceClosure(sourceRoot, expectedFiles, signal) {
  await assertExactSourceTree(sourceRoot, expectedFiles.map((file) => file.path), "$.extractorSourceInputRoot");
  let totalByteLength = 0;
  const snapshots = [];
  for (const [index, expected] of expectedFiles.entries()) {
    throwIfAborted(signal);
    const expectedByteLength = Number(BigInt(expected.byteLength));
    if (!Number.isSafeInteger(expectedByteLength) || expectedByteLength <= 0) {
      resource(
        `$.extractorSource.files[${index}].byteLength`,
        "locked source length is outside the executor range",
      );
    }
    totalByteLength += expectedByteLength;
    if (totalByteLength > MAX_SOURCE_TOTAL_BYTE_LENGTH) {
      resource(
        "$.extractorSource.files",
        `extractor source bytes exceed ${MAX_SOURCE_TOTAL_BYTE_LENGTH}`,
      );
    }
    const path = join(sourceRoot, expected.path);
    const bytes = await readBoundedRegularFile(
      path,
      expectedByteLength,
      `$.extractorSourceInputRoot/${expected.path}`,
      signal,
      expectedByteLength,
    );
    if (bytes.byteLength !== expectedByteLength) {
      hashMismatch(
        `$.extractorSource.files[${index}].byteLength`,
        `observed ${expected.path} length differs from the build lock`,
      );
    }
    const observedSha256 = sha256(bytes);
    if (observedSha256 !== expected.sha256) {
      hashMismatch(
        `$.extractorSource.files[${index}].sha256`,
        `observed ${expected.path} digest differs from the build lock`,
      );
    }
    snapshots.push({
      path: expected.path,
      sha256: observedSha256,
      byteLength: String(bytes.byteLength),
      bytes,
    });
  }
  await assertExactSourceTree(sourceRoot, expectedFiles.map((file) => file.path), "$.extractorSourceInputRoot");
  return snapshots;
}

/**
 * @param {string} stagedRoot
 * @param {readonly { readonly path: string; readonly sha256: string; readonly byteLength: string; readonly bytes: Uint8Array }[]} snapshots
 * @param {number} sourceDateEpoch
 * @param {AbortSignal | undefined} signal
 * @returns {Promise<StagedSourceIdentity>}
 */
async function stageSourceSnapshots(stagedRoot, snapshots, sourceDateEpoch, signal) {
  const parentPath = dirname(stagedRoot);
  const parentIdentity = await assertPrivateOwnedDirectory(
    parentPath,
    "$.roots.extractorSourceRoot.parent",
  );
  if (await lstatIfExists(stagedRoot, "$.roots.extractorSourceRoot") !== undefined) {
    conflict("$.roots.extractorSourceRoot", "staging root must not already exist");
  }
  let created = false;
  let stagedRootIdentity;
  try {
    try {
      await mkdir(stagedRoot, { mode: 0o700 });
    } catch (cause) {
      if (isNodeError(cause, "EEXIST")) {
        conflict("$.roots.extractorSourceRoot", "staging root was concurrently created");
      }
      throw cause;
    }
    created = true;
    stagedRootIdentity = await directoryIdentity(
      stagedRoot,
      "$.roots.extractorSourceRoot",
    );
    const directories = sourceDirectories(snapshots.map((snapshot) => snapshot.path));
    for (const directory of directories) {
      await mkdir(join(stagedRoot, directory), { mode: 0o700 });
    }
    for (const snapshot of snapshots) {
      throwIfAborted(signal);
      const filePath = join(stagedRoot, snapshot.path);
      const handle = await open(
        filePath,
        constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
        0o400,
      );
      try {
        await handle.writeFile(snapshot.bytes);
        await handle.sync();
        await handle.utimes(sourceDateEpoch, sourceDateEpoch);
        await handle.chmod(0o444);
      } finally {
        await handle.close();
      }
    }
    for (const directory of [...directories].reverse()) {
      const absoluteDirectory = join(stagedRoot, directory);
      await chmod(absoluteDirectory, 0o555);
      await syncDirectory(absoluteDirectory);
    }
    await chmod(stagedRoot, 0o555);
    await syncDirectory(stagedRoot);
    const stagedFileIdentities = await verifyStagedSourceClosure(
      stagedRoot,
      snapshots,
      signal,
      stagedRootIdentity,
    );
    await assertPathIdentity(stagedRoot, stagedRootIdentity, "$.roots.extractorSourceRoot");
    await assertPathIdentity(parentPath, parentIdentity, "$.roots.extractorSourceRoot.parent");
    await syncDirectory(parentPath);
    return Object.freeze({
      root: stagedRootIdentity,
      files: Object.freeze(stagedFileIdentities),
    });
  } catch (cause) {
    if (created && stagedRootIdentity !== undefined) {
      await removeStagingRoot(stagedRoot, stagedRootIdentity, cause);
    }
    throw cause;
  }
}

/** @param {import("./cpp_cute_browser_build_plan.mjs").CppCuteClangWasmBuildPlan} plan */
function sourceDateEpochSeconds(plan) {
  const value = plan.steps[0]?.environment.SOURCE_DATE_EPOCH;
  const seconds = Number(value);
  if (value === undefined || !Number.isSafeInteger(seconds) || seconds <= 0) {
    invalid("$.buildPlan.environment.SOURCE_DATE_EPOCH", "expected a positive safe epoch");
  }
  return seconds;
}

/**
 * @param {string} stagedRoot
 * @param {readonly { readonly path: string; readonly sha256: string; readonly byteLength: string; readonly bytes: Uint8Array }[]} snapshots
 * @param {AbortSignal | undefined} signal
 * @param {FileIdentity} stagedRootIdentity
 * @param {readonly StagedFileIdentity[]} [expectedFileIdentities]
 * @returns {Promise<readonly StagedFileIdentity[]>}
 */
async function verifyStagedSourceClosure(
  stagedRoot,
  snapshots,
  signal,
  stagedRootIdentity,
  expectedFileIdentities,
) {
  const expectedPaths = snapshots.map((snapshot) => snapshot.path);
  const expectedIdentityByPath = new Map(
    expectedFileIdentities?.map(({ path, identity }) => [path, identity]) ?? [],
  );
  if (expectedFileIdentities !== undefined &&
      expectedIdentityByPath.size !== snapshots.length) {
    unverified("$.stagedSourceRoot", "stored staged-file authority is incomplete");
  }
  await assertExactSourceTree(
    stagedRoot,
    expectedPaths,
    "$.stagedSourceRoot",
    stagedRootIdentity,
  );
  const observedFileIdentities = [];
  for (const [index, snapshot] of snapshots.entries()) {
    throwIfAborted(signal);
    const filePath = join(stagedRoot, snapshot.path);
    const observed = await readBoundedRegularFileSnapshot(
      filePath,
      snapshot.bytes.byteLength,
      `$.stagedSourceRoot/${snapshot.path}`,
      signal,
      snapshot.bytes.byteLength,
    );
    const identity = fileVersionIdentity(observed.stat);
    const expectedIdentity = expectedIdentityByPath.get(snapshot.path);
    if (expectedFileIdentities !== undefined &&
        (expectedIdentity === undefined || !sameFileVersionIdentity(identity, expectedIdentity))) {
      conflict(
        `$.stagedSourceRoot/${snapshot.path}`,
        "staged source file identity changed after preparation",
      );
    }
    await assertReadOnlyRegularPathIdentity(
      filePath,
      identity,
      snapshot.bytes.byteLength,
      `$.stagedSourceRoot/${snapshot.path}`,
    );
    if (sha256(observed.bytes) !== snapshot.sha256 ||
        !equalBytes(observed.bytes, snapshot.bytes)) {
      hashMismatch(
        `$.stagedSource.files[${index}]`,
        `staged ${snapshot.path} differs from the verified snapshot`,
      );
    }
    observedFileIdentities.push(Object.freeze({ path: snapshot.path, identity }));
  }
  throwIfAborted(signal);
  await assertExactSourceTree(
    stagedRoot,
    expectedPaths,
    "$.stagedSourceRoot",
    stagedRootIdentity,
  );
  for (const { path, identity } of observedFileIdentities) {
    const snapshot = snapshots.find((candidate) => candidate.path === path);
    if (snapshot === undefined) {
      unverified("$.stagedSourceRoot", "observed staged-file authority escaped its closure");
    }
    await assertReadOnlyRegularPathIdentity(
      join(stagedRoot, path),
      identity,
      snapshot.bytes.byteLength,
      `$.stagedSourceRoot/${path}`,
    );
  }
  return Object.freeze(observedFileIdentities);
}

/**
 * @param {string} outputRoot
 * @param {string} destinationPath
 * @param {Uint8Array} bytes
 * @param {string} expectedSha256
 * @param {AbortSignal | undefined} signal
 */
async function installSidecarNoClobber(
  outputRoot,
  destinationPath,
  bytes,
  expectedSha256,
  signal,
) {
  const outputRootIdentity = await assertPrivateOwnedDirectory(outputRoot, "$.outputRoot");
  const outputDirectory = dirname(destinationPath);
  if (outputDirectory !== join(outputRoot, "browsergrad-cpp-cute")) {
    invalid("$.distributedWasmPath", "distributed sidecar escaped its exact output directory");
  }
  const existingOutputDirectory = await lstatIfExists(outputDirectory, "$.distributedWasmPath.parent");
  if (existingOutputDirectory === undefined) {
    try {
      await mkdir(outputDirectory, { mode: 0o700 });
      await syncDirectory(outputRoot);
    } catch (cause) {
      if (!isNodeError(cause, "EEXIST")) throw cause;
    }
  }
  const outputDirectoryIdentity = await assertPrivateOwnedDirectory(
    outputDirectory,
    "$.distributedWasmPath.parent",
  );
  await assertPathIdentity(outputRoot, outputRootIdentity, "$.outputRoot");
  if (await destinationMatches(destinationPath, bytes, expectedSha256, signal)) {
    await assertPathIdentity(outputDirectory, outputDirectoryIdentity, "$.distributedWasmPath.parent");
    await assertPathIdentity(outputRoot, outputRootIdentity, "$.outputRoot");
    return;
  }
  throwIfAborted(signal);

  const temporaryPath = join(
    outputDirectory,
    `.clang-extractor.wasm.tmp-${process.pid}-${randomBytes(12).toString("hex")}`,
  );
  let temporaryCreated = false;
  /** @type {import("node:fs/promises").FileHandle | undefined} */
  let temporaryHandle;
  /** @type {FileIdentity | undefined} */
  let temporaryIdentity;
  let destinationLinked = false;
  /** @type {unknown} */
  let operationFailure;
  try {
    temporaryHandle = await open(
      temporaryPath,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o400,
    );
    temporaryCreated = true;
    temporaryIdentity = fileIdentity(await temporaryHandle.stat({ bigint: true }));
    await temporaryHandle.writeFile(bytes);
    await temporaryHandle.sync();
    await temporaryHandle.chmod(0o444);
    const temporaryStat = await temporaryHandle.stat({ bigint: true });
    if (!temporaryStat.isFile() || (temporaryStat.mode & 0o222n) !== 0n ||
        temporaryStat.size !== BigInt(bytes.byteLength)) {
      conflict("$.distributedWasmPath", "temporary sidecar metadata changed before commit");
    }
    if (!sameFileIdentity(fileIdentity(temporaryStat), temporaryIdentity)) {
      conflict("$.distributedWasmPath", "temporary sidecar inode changed while open");
    }
    await assertPathIdentity(temporaryPath, temporaryIdentity, "$.distributedWasmPath.temporary");
    await assertPathIdentity(outputDirectory, outputDirectoryIdentity, "$.distributedWasmPath.parent");
    throwIfAborted(signal);
    try {
      await CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS.link(temporaryPath, destinationPath);
      destinationLinked = true;
    } catch (cause) {
      if (!isNodeError(cause, "EEXIST")) throw cause;
      if (await destinationMatches(destinationPath, bytes, expectedSha256, signal)) {
        await assertPathIdentity(outputDirectory, outputDirectoryIdentity, "$.distributedWasmPath.parent");
        await assertPathIdentity(outputRoot, outputRootIdentity, "$.outputRoot");
        // A concurrently committed byte-identical destination is idempotent.
        // Cleanup still runs before this operation returns.
        destinationLinked = false;
      } else {
        conflict("$.distributedWasmPath", "distributed sidecar already exists with different bytes or unsafe metadata");
      }
    }
    if (destinationLinked) {
      try {
        const installedMatches = await destinationMatches(
          destinationPath,
          bytes,
          expectedSha256,
          undefined,
          temporaryIdentity,
        );
        if (!installedMatches) {
          conflict("$.distributedWasmPath", "installed sidecar did not preserve generated bytes");
        }
        await assertPathIdentity(outputDirectory, outputDirectoryIdentity, "$.distributedWasmPath.parent");
        await syncDirectory(outputDirectory);
        await assertPathIdentity(outputRoot, outputRootIdentity, "$.outputRoot");
        await assertReadOnlyRegularPathIdentity(
          destinationPath,
          temporaryIdentity,
          bytes.byteLength,
          "$.distributedWasmPath",
        );
      } catch (cause) {
        if (destinationLinked) {
          await rollbackLinkedDestination(destinationPath, temporaryIdentity, outputDirectory, cause);
          destinationLinked = false;
        }
        throw cause;
      }
    }
  } catch (cause) {
    operationFailure = cause;
  }

  const cleanupFailures = [];
  if (temporaryHandle !== undefined) {
    try {
      await CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS.closeFileHandle(
        temporaryHandle,
        "temporary-sidecar",
      );
    } catch (cause) {
      cleanupFailures.push(cause);
    }
  }
  if (temporaryCreated) {
    if (temporaryIdentity === undefined) {
      cleanupFailures.push(new Error("temporary sidecar identity was not captured"));
    } else {
      try {
        await removeTemporaryFile(temporaryPath, temporaryIdentity);
      } catch (cause) {
        cleanupFailures.push(cause);
      }
    }
  }
  if (cleanupFailures.length > 0) {
    compositeCleanup(
      "$.distributedWasmPath",
      "temporary sidecar cleanup did not complete",
      operationFailure,
      cleanupFailures,
    );
  }
  if (operationFailure !== undefined) throw operationFailure;
}

/**
 * @param {string} path
 * @param {Uint8Array} expectedBytes
 * @param {string} expectedSha256
 * @param {AbortSignal | undefined} signal
 */
async function destinationMatches(
  path,
  expectedBytes,
  expectedSha256,
  signal,
  expectedIdentity,
) {
  let observed;
  try {
    observed = await readBoundedRegularFileSnapshot(
      path,
      expectedBytes.byteLength,
      "$.distributedWasmPath",
      signal,
      expectedBytes.byteLength,
    );
  } catch (cause) {
    if (isNodeError(cause, "ENOENT")) return false;
    if (isNodeError(cause, "ELOOP")) {
      conflict("$.distributedWasmPath", "distributed sidecar path must not be a symbolic link");
    }
    if (cause instanceof CppCuteBrowserBuildExecutorError &&
        (cause.code === HASH_MISMATCH || cause.code === INVALID || cause.code === RESOURCE_LIMIT)) {
      conflict("$.distributedWasmPath", "distributed sidecar changed during admission");
    }
    throw cause;
  }
  if ((observed.stat.mode & 0o222n) !== 0n) {
    conflict("$.distributedWasmPath", "existing distributed sidecar must be read-only");
  }
  const observedIdentity = fileIdentity(observed.stat);
  if (expectedIdentity !== undefined && !sameFileIdentity(observedIdentity, expectedIdentity)) {
    conflict("$.distributedWasmPath", "installed sidecar differs from the committed temporary inode");
  }
  if (sha256(observed.bytes) !== expectedSha256 || !equalBytes(observed.bytes, expectedBytes)) {
    conflict("$.distributedWasmPath", "distributed sidecar already exists with different bytes");
  }
  await assertReadOnlyRegularPathIdentity(
    path,
    observedIdentity,
    expectedBytes.byteLength,
    "$.distributedWasmPath",
  );
  return true;
}

/**
 * @param {string} path
 * @param {number} maximumByteLength
 * @param {string} diagnosticPath
 * @param {AbortSignal | undefined} signal
 * @param {number} [exactByteLength]
 */
async function readBoundedRegularFile(
  path,
  maximumByteLength,
  diagnosticPath,
  signal,
  exactByteLength,
) {
  return (await readBoundedRegularFileSnapshot(
    path,
    maximumByteLength,
    diagnosticPath,
    signal,
    exactByteLength,
  )).bytes;
}

/**
 * @param {string} path
 * @param {number} maximumByteLength
 * @param {string} diagnosticPath
 * @param {AbortSignal | undefined} signal
 * @param {number} [exactByteLength]
 */
async function readBoundedRegularFileSnapshot(
  path,
  maximumByteLength,
  diagnosticPath,
  signal,
  exactByteLength,
) {
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  } catch (cause) {
    if (isNodeError(cause, "ELOOP")) {
      invalid(diagnosticPath, "file must not be a symbolic link", { cause });
    }
    throw cause;
  }
  try {
    const before = await handle.stat({ bigint: true });
    if (!before.isFile()) invalid(diagnosticPath, "expected a regular file");
    if (before.size <= 0n) invalid(diagnosticPath, "file must be nonempty");
    if (before.size > BigInt(maximumByteLength)) {
      if (exactByteLength !== undefined) {
        hashMismatch(`${diagnosticPath}.byteLength`, "file length differs from the build lock");
      }
      resource(diagnosticPath, `file exceeds ${maximumByteLength} bytes`);
    }
    if (exactByteLength !== undefined && before.size !== BigInt(exactByteLength)) {
      hashMismatch(`${diagnosticPath}.byteLength`, "file length differs from the build lock");
    }
    const expectedReadByteLength = Number(before.size);
    const bytes = new Uint8Array(expectedReadByteLength);
    let offset = 0;
    while (offset < expectedReadByteLength) {
      throwIfAborted(signal);
      const { bytesRead } = await handle.read(
        bytes,
        offset,
        expectedReadByteLength - offset,
        offset,
      );
      if (bytesRead === 0) {
        invalid(diagnosticPath, "file became shorter while it was being snapshotted");
      }
      offset += bytesRead;
    }
    const trailing = new Uint8Array(1);
    const { bytesRead: trailingByteLength } = await handle.read(
      trailing,
      0,
      1,
      expectedReadByteLength,
    );
    if (trailingByteLength !== 0) {
      invalid(diagnosticPath, "file became longer while it was being snapshotted");
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileSnapshot(before, after) || bytes.byteLength !== Number(after.size)) {
      invalid(diagnosticPath, "file changed while it was being snapshotted");
    }
    return { bytes, stat: after };
  } finally {
    await handle.close();
  }
}

/** @param {import("node:fs").BigIntStats} before @param {import("node:fs").BigIntStats} after */
function sameFileSnapshot(before, after) {
  return before.dev === after.dev &&
    before.ino === after.ino &&
    before.size === after.size &&
    before.mtimeNs === after.mtimeNs &&
    before.ctimeNs === after.ctimeNs;
}

/**
 * @param {string} path
 * @param {readonly string[]} expectedFiles
 * @param {string} diagnosticPath
 * @param {FileIdentity} [expectedRootIdentity]
 */
async function assertExactSourceTree(path, expectedFiles, diagnosticPath, expectedRootIdentity) {
  const rootIdentity = expectedRootIdentity ?? await directoryIdentity(path, diagnosticPath);
  await assertPathIdentity(path, rootIdentity, diagnosticPath);
  const expectedFileSet = new Set(expectedFiles);
  const expectedDirectorySet = new Set(sourceDirectories(expectedFiles));
  const observedFiles = new Set();
  let observedNodeCount = 0;
  const maximumNodeCount = expectedFileSet.size + expectedDirectorySet.size;

  /** @param {string} absoluteDirectory @param {string} relativeDirectory */
  async function visit(absoluteDirectory, relativeDirectory) {
    const traversalIdentity = await directoryIdentity(absoluteDirectory, diagnosticPath);
    const directory = await opendir(absoluteDirectory);
    try {
      while (true) {
        const entry = await directory.read();
        if (entry === null) break;
        observedNodeCount += 1;
        if (observedNodeCount > maximumNodeCount) {
          invalid(diagnosticPath, "source tree contains entries outside the exact locked closure");
        }
        const relativePath = relativeDirectory === ""
          ? entry.name
          : `${relativeDirectory}/${entry.name}`;
        const entryPath = `${diagnosticPath}/${relativePath}`;
        if (entry.isSymbolicLink()) {
          invalid(entryPath, "source tree must not contain symbolic links");
        }
        if (entry.isDirectory()) {
          if (!expectedDirectorySet.has(relativePath)) {
            invalid(entryPath, "source tree contains an undeclared directory");
          }
          await visit(join(absoluteDirectory, entry.name), relativePath);
          continue;
        }
        if (!entry.isFile() || !expectedFileSet.has(relativePath)) {
          invalid(entryPath, "source tree contains an undeclared or non-regular file");
        }
        observedFiles.add(relativePath);
      }
    } finally {
      await directory.close().catch((cause) => {
        if (!isNodeError(cause, "ERR_DIR_CLOSED")) throw cause;
      });
    }
    await assertPathIdentity(absoluteDirectory, traversalIdentity, diagnosticPath);
  }

  await visit(path, "");
  if (observedFiles.size !== expectedFileSet.size ||
      [...expectedFileSet].some((expected) => !observedFiles.has(expected))) {
    invalid(diagnosticPath, "source tree is missing one or more locked files");
  }
  await assertPathIdentity(path, rootIdentity, diagnosticPath);
}

/** @param {readonly string[]} paths */
function sourceDirectories(paths) {
  const directories = new Set();
  for (const path of paths) {
    const segments = path.split("/");
    for (let length = 1; length < segments.length; length += 1) {
      directories.add(segments.slice(0, length).join("/"));
    }
  }
  return [...directories].sort((left, right) => {
    const depth = left.split("/").length - right.split("/").length;
    return depth === 0 ? compareStrings(left, right) : depth;
  });
}

/**
 * Build staging and output roots are trusted only against other OS users.
 * Same-uid replacement cannot be made race-free in Node without openat-style
 * directory-relative operations, so every commit and cleanup rebinds dev/ino
 * and refuses destructive work after identity drift.
 *
 * @param {string} path
 * @param {string} diagnosticPath
 * @returns {Promise<FileIdentity>}
 */
async function assertPrivateOwnedDirectory(path, diagnosticPath) {
  const stat = await lstat(path, { bigint: true });
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    invalid(diagnosticPath, "expected a non-symlink directory");
  }
  if (typeof process.getuid !== "function" || stat.uid !== BigInt(process.getuid())) {
    invalid(diagnosticPath, "directory must be owned by the current build user");
  }
  if ((stat.mode & 0o022n) !== 0n) {
    invalid(diagnosticPath, "directory must not be writable by group or other users");
  }
  await assertTrustedPosixAncestry(path, diagnosticPath);
  return fileIdentity(stat);
}

/**
 * A private leaf is insufficient when another user can rename one of its
 * ancestors. Validate both the lexical path (including root-owned compatibility
 * symlinks such as macOS /var) and the fully resolved path. Same-uid mutation is
 * deliberately outside this executor's threat model and is covered by the
 * documented single-writer contract.
 *
 * @param {string} path
 * @param {string} diagnosticPath
 */
async function assertTrustedPosixAncestry(path, diagnosticPath) {
  let resolved;
  try {
    resolved = await realpath(path);
  } catch (cause) {
    io(diagnosticPath, "failed to resolve directory ancestry", { cause });
  }
  await assertTrustedAncestryPath(path, diagnosticPath, true);
  if (resolved !== path) {
    await assertTrustedAncestryPath(resolved, diagnosticPath, false);
  }
}

/** @param {string} path @param {string} diagnosticPath @param {boolean} allowRootOwnedSymlinks */
async function assertTrustedAncestryPath(path, diagnosticPath, allowRootOwnedSymlinks) {
  if (!isAbsolute(path) || normalize(path) !== path) {
    invalid(diagnosticPath, "directory ancestry must resolve to a normalized absolute path");
  }
  if (typeof process.getuid !== "function") {
    invalid(diagnosticPath, "directory ancestry ownership requires POSIX uid support");
  }
  const currentUid = BigInt(process.getuid());
  for (const ancestor of absolutePathPrefixes(path)) {
    let stat;
    try {
      stat = await lstat(ancestor, { bigint: true });
    } catch (cause) {
      io(diagnosticPath, `failed to inspect directory ancestor ${ancestor}`, { cause });
    }
    if (stat.isSymbolicLink()) {
      if (!allowRootOwnedSymlinks || stat.uid !== 0n) {
        invalid(
          diagnosticPath,
          `directory ancestor ${ancestor} must not be a non-root-owned symbolic link`,
        );
      }
      continue;
    }
    if (!stat.isDirectory()) {
      invalid(diagnosticPath, `directory ancestor ${ancestor} must be a directory`);
    }
    if (stat.uid !== 0n && stat.uid !== currentUid) {
      invalid(
        diagnosticPath,
        `directory ancestor ${ancestor} must be owned by root or the current build user`,
      );
    }
    const writableByOtherUsers = (stat.mode & 0o022n) !== 0n;
    const rootOwnedStickyDirectory = stat.uid === 0n && (stat.mode & 0o1000n) !== 0n;
    if (writableByOtherUsers && !rootOwnedStickyDirectory) {
      invalid(
        diagnosticPath,
        `directory ancestor ${ancestor} is writable by other users without root-owned sticky protection`,
      );
    }
  }
}

/** @param {string} path */
function absolutePathPrefixes(path) {
  const prefixes = ["/"];
  let current = "/";
  for (const segment of path.split("/").filter((entry) => entry.length > 0)) {
    current = join(current, segment);
    prefixes.push(current);
  }
  return prefixes;
}

/** @param {string} path @param {string} diagnosticPath @returns {Promise<FileIdentity>} */
async function directoryIdentity(path, diagnosticPath) {
  const stat = await lstat(path, { bigint: true });
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    invalid(diagnosticPath, "expected a non-symlink directory");
  }
  return fileIdentity(stat);
}

/** @param {string} path @param {FileIdentity} expected @param {string} diagnosticPath */
async function assertPathIdentity(path, expected, diagnosticPath) {
  let stat;
  try {
    stat = await lstat(path, { bigint: true });
  } catch (cause) {
    if (isNodeError(cause, "ENOENT")) {
      conflict(diagnosticPath, "filesystem path disappeared during the operation");
    }
    throw cause;
  }
  if (stat.isSymbolicLink() || !sameFileIdentity(fileIdentity(stat), expected)) {
    conflict(diagnosticPath, "filesystem path identity changed during the operation");
  }
}

/**
 * @param {string} path
 * @param {FileIdentity} expected
 * @param {number} expectedByteLength
 * @param {string} diagnosticPath
 */
async function assertReadOnlyRegularPathIdentity(
  path,
  expected,
  expectedByteLength,
  diagnosticPath,
) {
  let stat;
  try {
    stat = await lstat(path, { bigint: true });
  } catch (cause) {
    if (isNodeError(cause, "ENOENT")) {
      conflict(diagnosticPath, "filesystem path disappeared during the operation");
    }
    throw cause;
  }
  if (!stat.isFile() || stat.isSymbolicLink() ||
      !sameFileIdentity(fileIdentity(stat), expected) ||
      (stat.mode & 0o222n) !== 0n || stat.size !== BigInt(expectedByteLength)) {
    conflict(diagnosticPath, "file path identity or read-only metadata changed during admission");
  }
}

/** @param {import("node:fs").BigIntStats} stat @returns {FileIdentity} */
function fileIdentity(stat) {
  return Object.freeze({ dev: stat.dev, ino: stat.ino });
}

/** @param {import("node:fs").BigIntStats} stat @returns {FileVersionIdentity} */
function fileVersionIdentity(stat) {
  return Object.freeze({
    dev: stat.dev,
    ino: stat.ino,
    ctimeNs: stat.ctimeNs,
    birthtimeNs: stat.birthtimeNs,
  });
}

/** @param {FileIdentity} left @param {FileIdentity} right */
function sameFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino;
}

/** @param {FileVersionIdentity} left @param {FileVersionIdentity} right */
function sameFileVersionIdentity(left, right) {
  return sameFileIdentity(left, right) &&
    left.ctimeNs === right.ctimeNs &&
    left.birthtimeNs === right.birthtimeNs;
}

/** @param {string} path @param {string} diagnosticPath */
async function lstatIfExists(path, diagnosticPath) {
  try {
    return await lstat(path);
  } catch (cause) {
    if (isNodeError(cause, "ENOENT")) return undefined;
    io(diagnosticPath, "failed to inspect filesystem path", { cause });
  }
}

/** @param {string} path */
async function syncDirectory(path) {
  await CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS.syncDirectory(path);
}

/** @param {string} path @param {FileIdentity} identity @param {unknown} primary */
async function removeStagingRoot(path, identity, primary) {
  try {
    await assertPathIdentity(path, identity, "$.roots.extractorSourceRoot");
    await makeOwnedDirectoryTreeWritable(path, identity);
    await assertPathIdentity(path, identity, "$.roots.extractorSourceRoot");
    await rm(path, { recursive: true, force: true, maxRetries: 0 });
  } catch (cause) {
    cleanup("$.roots.extractorSourceRoot", "failed to remove partial staging root", primary, cause);
  }
}

/** @param {string} path @param {FileIdentity} [expectedIdentity] */
async function makeOwnedDirectoryTreeWritable(path, expectedIdentity) {
  const stat = await lstatIfExists(path, "$.roots.extractorSourceRoot");
  if (stat === undefined) return;
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    cleanup(
      "$.roots.extractorSourceRoot",
      "partial staging root changed into an unsafe filesystem object",
    );
  }
  if (expectedIdentity !== undefined) {
    await assertPathIdentity(path, expectedIdentity, "$.roots.extractorSourceRoot");
  }
  await chmod(path, 0o700);
  const identity = expectedIdentity ?? await directoryIdentity(path, "$.roots.extractorSourceRoot");
  const directory = await opendir(path);
  try {
    while (true) {
      const entry = await directory.read();
      if (entry === null) break;
      if (entry.isDirectory() && !entry.isSymbolicLink()) {
        const childPath = join(path, entry.name);
        const childIdentity = await directoryIdentity(childPath, "$.roots.extractorSourceRoot");
        await makeOwnedDirectoryTreeWritable(childPath, childIdentity);
      }
    }
  } finally {
    await directory.close().catch((cause) => {
      if (!isNodeError(cause, "ERR_DIR_CLOSED")) throw cause;
    });
  }
  await assertPathIdentity(path, identity, "$.roots.extractorSourceRoot");
}

/**
 * @param {string} destinationPath
 * @param {FileIdentity} installedIdentity
 * @param {string} outputDirectory
 * @param {unknown} primary
 */
async function rollbackLinkedDestination(
  destinationPath,
  installedIdentity,
  outputDirectory,
  primary,
) {
  try {
    const stat = await CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS.lstat(
      destinationPath,
      { bigint: true },
    );
    if (stat.isSymbolicLink() || !sameFileIdentity(fileIdentity(stat), installedIdentity)) {
      cleanup(
        "$.distributedWasmPath",
        "destination identity changed before safe rollback; refusing to remove it",
        primary,
      );
    }
    await CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS.rm(
      destinationPath,
      { force: false, maxRetries: 0 },
    );
    await syncDirectory(outputDirectory);
  } catch (cause) {
    if (isNodeError(cause, "ENOENT")) return;
    if (cause instanceof CppCuteBrowserBuildExecutorError && cause.code === CLEANUP) throw cause;
    cleanup("$.distributedWasmPath", "failed to roll back linked sidecar", primary, cause);
  }
}

/** @param {string} path @param {FileIdentity} identity */
async function removeTemporaryFile(path, identity) {
  try {
    const stat = await lstat(path, { bigint: true });
    if (stat.isSymbolicLink() || !sameFileIdentity(fileIdentity(stat), identity)) {
      cleanup(
        "$.distributedWasmPath",
        "temporary sidecar identity changed before cleanup; refusing to remove it",
      );
    }
    await rm(path, { force: true, maxRetries: 0 });
  } catch (cause) {
    if (isNodeError(cause, "ENOENT")) return;
    if (cause instanceof CppCuteBrowserBuildExecutorError && cause.code === CLEANUP) throw cause;
    cleanup("$.distributedWasmPath", "failed to remove temporary sidecar", undefined, cause);
  }
}

/** @param {AbortSignal | undefined} signal @param {string} stagedRoot @param {FileIdentity} identity */
async function throwIfAbortedOrRemove(signal, stagedRoot, identity) {
  try {
    throwIfAborted(signal);
  } catch (cause) {
    await removeStagingRoot(stagedRoot, identity, cause);
    throw cause;
  }
}

/** @param {import("./cpp_cute_browser_build_executor.mjs").PreparedCppCuteClangWasmBuildSource} prepared */
function storedPreparedSource(prepared) {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const stored = PREPARED_SOURCES.get(prepared);
  if (stored === undefined || prepared.authority !== "build-source-snapshot-only" ||
      prepared.sourceVerified !== true || prepared.buildExecuted !== false ||
      prepared.outputIdentityAuthorized !== false ||
      prepared.reproducibilityVerified !== false || prepared.releaseReady !== false) {
    unverified();
  }
  return stored;
}

/** @param {unknown} value @param {readonly string[]} keys @param {string} path */
function exactDataObject(value, keys, path) {
  let prototype;
  let descriptors;
  try {
    prototype = Object.getPrototypeOf(value);
    descriptors = Object.getOwnPropertyDescriptors(value);
  } catch (cause) {
    invalid(path, "expected an inspectable plain object", { cause });
  }
  if (typeof value !== "object" || value === null || prototype !== Object.prototype) {
    invalid(path, "expected a plain object");
  }
  const ownKeys = Reflect.ownKeys(descriptors);
  if (ownKeys.length !== keys.length || ownKeys.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(path, `expected exactly fields ${keys.join(", ")}`);
  }
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor) || descriptor.enumerable !== true) {
      invalid(`${path}.${key}`, "field must be an enumerable data property");
    }
  }
  return descriptors;
}

/** @param {unknown} value @param {readonly string[]} keys @param {string} path */
function snapshotStringRecord(value, keys, path) {
  const descriptors = exactDataObject(value, keys, path);
  return Object.fromEntries(keys.filter((key) => key !== "searchPath").map((key) => [
    key,
    pathString(dataValue(descriptors, key, path), `${path}.${key}`),
  ]));
}

/** @param {PropertyDescriptorMap} descriptors @param {string} key @param {string} path */
function dataValue(descriptors, key, path) {
  const descriptor = descriptors[key];
  if (descriptor === undefined || !("value" in descriptor)) invalid(`${path}.${key}`, "missing field");
  return descriptor.value;
}

/** @param {unknown} value @param {string} path */
function pathString(value, path) {
  if (typeof value !== "string" || value.length === 0 || value.length > 4_096 ||
      !isAbsolute(value) || normalize(value) !== value || value === "/" ||
      value.endsWith("/") || !PORTABLE_ABSOLUTE_PATH.test(value)) {
    invalid(path, "expected a normalized portable absolute POSIX path");
  }
  return value;
}

/** @param {AbortSignal | undefined} signal */
function throwIfAborted(signal) {
  if (signal !== undefined && ABORTED_GETTER?.call(signal) === true) cancelled();
}

/** @param {Uint8Array} bytes */
function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

/** @param {Uint8Array} value @param {Uint8Array} prefix */
function startsWithBytes(value, prefix) {
  if (value.byteLength < prefix.byteLength) return false;
  return timingSafeEqual(value.subarray(0, prefix.byteLength), prefix);
}

/** @param {Uint8Array} left @param {Uint8Array} right */
function equalBytes(left, right) {
  return left.byteLength === right.byteLength && timingSafeEqual(left, right);
}

/** @param {string} left @param {string} right */
function compareStrings(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

/** @param {unknown} value @param {string} code */
function isNodeError(value, code) {
  return typeof value === "object" && value !== null && "code" in value && value.code === code;
}

/** @param {unknown} cause @param {string} path @param {string} message */
function rethrowExecutorOrIo(cause, path, message) {
  if (cause instanceof CppCuteBrowserBuildExecutorError) throw cause;
  if (isNodeError(cause, "ABORT_ERR")) cancelled();
  io(path, message, { cause });
}

function cancelled() {
  fail(CANCELLED, "$options.signal", "operation was aborted");
}

/** @param {string} path @param {string} message @param {ErrorOptions} [options] */
function invalid(path, message, options) {
  fail(INVALID, path, message, options);
}

/** @param {string} path @param {string} message */
function resource(path, message) {
  fail(RESOURCE_LIMIT, path, message);
}

/** @param {string} path @param {string} message */
function hashMismatch(path, message) {
  fail(HASH_MISMATCH, path, message);
}

/** @param {string} path @param {string} message */
function conflict(path, message) {
  fail(CONFLICT, path, message);
}

/** @param {string} path @param {string} message @param {ErrorOptions} [options] */
function io(path, message, options) {
  fail(IO, path, message, options);
}

/** @param {string} path @param {string} message */
function buildFailed(path, message) {
  fail(BUILD_FAILED, path, message);
}

/** @param {string} path @param {string} message @param {unknown} primary @param {unknown} cause */
function cleanup(path, message, primary, cause) {
  fail(CLEANUP, path, message, { cause: cause ?? primary });
}

/** @param {string} path @param {string} message @param {unknown} primary @param {readonly unknown[]} failures */
function compositeCleanup(path, message, primary, failures) {
  const causes = primary === undefined ? [...failures] : [primary, ...failures];
  cleanup(
    path,
    message,
    primary,
    new AggregateError(causes, "build executor operation and cleanup failures"),
  );
}

/** @param {string} [path] @param {string} [message] */
function unverified(
  path = "$prepared",
  message = "expected opaque verified source-snapshot authority",
) {
  fail(UNVERIFIED, path, message);
}

/**
 * @param {import("./cpp_cute_browser_build_executor.mjs").CppCuteBrowserBuildExecutorErrorCode} code
 * @param {string} path
 * @param {string} message
 * @param {ErrorOptions} [options]
 * @returns {never}
 */
function fail(code, path, message, options) {
  throw new CppCuteBrowserBuildExecutorError(code, path, message, options);
}
