import { createHash, timingSafeEqual } from "node:crypto";
import { constants } from "node:fs";
import {
  lstat,
  open,
  opendir,
  realpath,
} from "node:fs/promises";
import { dirname, isAbsolute, join, normalize, resolve } from "node:path/posix";
import { pathToFileURL } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";

const INVALID = "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-INVALID";
const MISMATCH = "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-MISMATCH";
const CONFLICT = "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-CONFLICT";
const IO = "BG-COMPILER-CPP-CUTE-BROWSER-REPRODUCIBILITY-IO";
const BUILD_EXECUTION_SCHEMA =
  "browsergrad.compiler.cpp-cute.clang-wasm-build-execution-observation";
const RUNTIME_CLOSURE_HASH_DOMAIN = "browsergrad.compiler.cpp-cute.build-runtime-closure.v1";
const REPRODUCIBILITY_SCHEMA =
  "browsergrad.compiler.cpp-cute.clang-wasm-reproducibility";
const PORTABLE_ABSOLUTE_PATH = /^\/[A-Za-z0-9._+/-]+$/u;
const SHA256_HEX = /^[0-9a-f]{64}$/u;
const MAX_EVIDENCE_BYTE_LENGTH = 1024 * 1024;
const MAX_RUNTIME_ABI_REVIEW_BYTE_LENGTH = 1024 * 1024;
const MAX_RUNTIME_CLOSURE_OBSERVATION_BYTE_LENGTH = 256 * 1024;
const MAX_RUNTIME_CLOSURE_FILE_COUNT = 256;
const MAX_RUNTIME_CLOSURE_FILE_BYTE_LENGTH = 16 * 1024 * 1024;
const MAX_FACTORY_MODULE_BYTE_LENGTH = 32 * 1024 * 1024;
const MAX_WASM_BYTE_LENGTH = 256 * 1024 * 1024;
const MAX_LINK_MAP_BYTE_LENGTH = 128 * 1024 * 1024;
const MAX_LOG_BYTE_LENGTH = 16 * 1024 * 1024;
const MAX_NATIVE_TOOL_BYTE_LENGTH = 256 * 1024 * 1024;
const HASH_BUFFER_BYTE_LENGTH = 1024 * 1024;
const ARGUMENT_NAMES = Object.freeze(["first-root", "output", "second-root"]);
const STEP_PROFILES = Object.freeze([
  Object.freeze({ id: "native-tablegen-configure", stageId: "native-tablegen", kind: "configure" }),
  Object.freeze({ id: "native-tablegen-build", stageId: "native-tablegen", kind: "build" }),
  Object.freeze({ id: "clang-extractor-wasm-configure", stageId: "clang-extractor-wasm", kind: "configure" }),
  Object.freeze({ id: "clang-extractor-wasm-build", stageId: "clang-extractor-wasm", kind: "build" }),
]);
const VERIFIED_REPRODUCIBILITY = new WeakSet();

export class CppCuteClangWasmReproducibilityError extends Error {
  /** @param {string} code @param {string} path @param {string} message @param {ErrorOptions} [options] */
  constructor(code, path, message, options) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteClangWasmReproducibilityError";
    this.code = code;
    this.path = path;
  }
}

/** @param {readonly string[]} argv */
export function parseCppCuteClangWasmReproducibilityArguments(argv) {
  if (argv.length !== ARGUMENT_NAMES.length) invalid("$argv", "expected exactly three named arguments");
  const values = new Map();
  for (const [index, argument] of argv.entries()) {
    if (typeof argument !== "string" || !argument.startsWith("--")) {
      invalid(`$argv[${index}]`, "expected --name=/absolute/path");
    }
    const equals = argument.indexOf("=");
    if (equals <= 2) invalid(`$argv[${index}]`, "expected --name=/absolute/path");
    const name = argument.slice(2, equals);
    if (!ARGUMENT_NAMES.includes(name)) invalid(`$argv[${index}]`, `unknown argument ${name}`);
    if (values.has(name)) invalid(`$argv[${index}]`, `duplicate argument ${name}`);
    values.set(name, portableAbsolutePath(argument.slice(equals + 1), `$argv.${name}`));
  }
  return Object.freeze(Object.fromEntries(ARGUMENT_NAMES.map((name) => [name, values.get(name)])));
}

/**
 * @param {import("./cpp_cute_browser_build_reproducibility.mjs").VerifyCppCuteClangWasmReproducibilityInput} input
 * @returns {Promise<import("./cpp_cute_browser_build_reproducibility.mjs").VerifiedCppCuteClangWasmReproducibility>}
 */
export async function verifyCppCuteClangWasmReproducibility(input) {
  const object = exactObject(input, ["firstRoot", "secondRoot"], "$input");
  const firstRoot = portableAbsolutePath(exactString(object.firstRoot, "$.firstRoot"), "$.firstRoot");
  const secondRoot = portableAbsolutePath(exactString(object.secondRoot, "$.secondRoot"), "$.secondRoot");
  const [firstRealRoot, secondRealRoot] = await Promise.all([
    admitBuildRoot(firstRoot, "$.firstRoot"),
    admitBuildRoot(secondRoot, "$.secondRoot"),
  ]);
  if (pathsOverlap(firstRealRoot, secondRealRoot)) {
    conflict("$input", "clean-build artifact roots must be distinct and non-overlapping");
  }

  const lock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const body = unwrapPreparedCppCuteBrowserBuildInputLock(lock).lock.body;
  const llvm = body.sources.find((source) => source.sourceId === "llvm-project");
  if (llvm === undefined) invalid("$buildLock.sources", "LLVM source selection is missing");

  const [first, second] = await Promise.all([
    admitBuild(firstRealRoot, "$builds[0]", lock.lockId, lock.extractorSourceSetSha256, body, llvm),
    admitBuild(secondRealRoot, "$builds[1]", lock.lockId, lock.extractorSourceSetSha256, body, llvm),
  ]);
  assertDistinctBuildPaths(first.execution.paths, second.execution.paths);
  assertCanonicalCommandParity(first, second);
  assertIdentityParity(first, second);

  const evidence = Object.freeze({
    schema: REPRODUCIBILITY_SCHEMA,
    version: 2,
    authority: "clang-wasm-extractor-reproducibility-observation-only",
    lockId: lock.lockId,
    sourceSetSha256: lock.extractorSourceSetSha256,
    cleanBuildCount: 2,
    builds: Object.freeze([
      buildIdentity(1, first),
      buildIdentity(2, second),
    ]),
    comparison: Object.freeze({
      sourceAndBuildPathsDistinct: true,
      runtimeClosureMatched: true,
      canonicalCommandsAndEnvironmentMatched: true,
      nativeTablegenIdentitiesMatched: true,
      factoryModuleBytesMatched: true,
      wasmBytesMatched: true,
      runtimeAbiReviewBytesMatched: true,
      linkMapBytesMatched: true,
    }),
    claims: Object.freeze({
      extractorOutputsReproducible: true,
      fullDistributedOutputSetReproducible: false,
      abiConformanceVerified: false,
      outputIdentityAuthorized: false,
      producerAttested: false,
      releaseReady: false,
    }),
  });
  VERIFIED_REPRODUCIBILITY.add(evidence);
  return /** @type {import("./cpp_cute_browser_build_reproducibility.mjs").VerifiedCppCuteClangWasmReproducibility} */ (evidence);
}

/**
 * @param {string} outputPath
 * @param {import("./cpp_cute_browser_build_reproducibility.mjs").VerifiedCppCuteClangWasmReproducibility} evidence
 */
export async function writeCppCuteClangWasmReproducibilityEvidence(outputPath, evidence) {
  if (typeof evidence !== "object" || evidence === null || !VERIFIED_REPRODUCIBILITY.has(evidence)) {
    invalid("$evidence", "expected verifier-issued reproducibility evidence");
  }
  const path = portableAbsolutePath(outputPath, "$outputPath");
  await admitPrivateDirectory(dirname(path), "$outputPath.parent");
  const bytes = canonicalJsonBytes(evidence);
  let handle;
  try {
    handle = await open(
      path,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o400,
    );
    await handle.writeFile(bytes);
    await handle.sync();
    await handle.chmod(0o444);
  } catch (cause) {
    if (isNodeError(cause, "EEXIST") || isNodeError(cause, "ELOOP")) {
      conflict("$outputPath", "reproducibility evidence output must not already exist", { cause });
    }
    io("$outputPath", "failed to persist reproducibility evidence", { cause });
  } finally {
    await handle?.close();
  }
  const directory = await open(dirname(path), constants.O_RDONLY | constants.O_DIRECTORY);
  try {
    await directory.sync();
  } finally {
    await directory.close();
  }
  return Object.freeze({
    outputPath: path,
    sha256: sha256(bytes),
    byteLength: bytes.byteLength,
  });
}

async function admitBuild(root, diagnosticPath, lockId, sourceSetSha256, body, llvm) {
  const expectedOutputFiles = [
    "browsergrad-cpp-cute/clang-extractor.wasm",
    "build-execution-observation.v2.json",
    "clang-wasm-runtime-abi-review.v1.json",
  ];
  const expectedEvidenceFiles = [
    "build-logs/clang-extractor-wasm-build.stderr.log",
    "build-logs/clang-extractor-wasm-build.stdout.log",
    "build-logs/clang-extractor-wasm-configure.stderr.log",
    "build-logs/clang-extractor-wasm-configure.stdout.log",
    "build-logs/native-tablegen-build.stderr.log",
    "build-logs/native-tablegen-build.stdout.log",
    "build-logs/native-tablegen-configure.stderr.log",
    "build-logs/native-tablegen-configure.stdout.log",
    "clang-extractor.link.map",
    "generated/clang-extractor.mjs",
    "generated/clang-extractor.wasm",
  ];
  await Promise.all([
    assertExactTree(join(root, "output"), expectedOutputFiles, `${diagnosticPath}.output`),
    assertExactTree(join(root, "state", "evidence"), expectedEvidenceFiles, `${diagnosticPath}.evidence`),
  ]);
  const evidencePath = join(root, "output", "build-execution-observation.v2.json");
  const evidenceSnapshot = await readSmallFile(
    evidencePath,
    MAX_EVIDENCE_BYTE_LENGTH,
    `${diagnosticPath}.buildExecutionEvidence`,
  );
  const evidence = decodeBuildExecutionEvidence(
    evidenceSnapshot.bytes,
    diagnosticPath,
    lockId,
    sourceSetSha256,
    body,
    llvm,
  );
  await verifyObservedFiles(root, evidence, diagnosticPath);
  const runtimeAbiReview = await admitRuntimeAbiReview(
    join(root, "output", "clang-wasm-runtime-abi-review.v1.json"),
    evidence.execution,
    `${diagnosticPath}.runtimeAbiReview`,
  );
  return Object.freeze({
    evidenceSha256: sha256(evidenceSnapshot.bytes),
    evidenceByteLength: evidenceSnapshot.bytes.byteLength,
    runtimeAbiReview,
    ...evidence,
  });
}

async function admitRuntimeAbiReview(path, execution, diagnosticPath) {
  const snapshot = await readSmallFile(
    path,
    MAX_RUNTIME_ABI_REVIEW_BYTE_LENGTH,
    diagnosticPath,
  );
  let value;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(snapshot.bytes));
  } catch (cause) {
    invalid(diagnosticPath, "runtime-ABI review must be strict UTF-8 JSON", { cause });
  }
  const canonical = canonicalJsonBytes(value);
  if (!equalBytes(snapshot.bytes, canonical)) {
    invalid(diagnosticPath, "runtime-ABI review must use canonical JSON bytes");
  }
  const report = exactObject(value, [
    "authority", "wasmSha256", "wasmByteLength", "observedProjectionSha256",
    "runtimeAbiManifestId", "runtimeAbiContractSha256", "exactInterfaceConformance",
    "mismatches", "projection", "rawWasmVerified", "workerExecutionReady", "releaseReady",
  ], diagnosticPath);
  exactValue(report.authority, "review-observation-only", `${diagnosticPath}.authority`);
  exactValue(report.wasmSha256, execution.wasmSha256, `${diagnosticPath}.wasmSha256`);
  exactValue(report.wasmByteLength, execution.wasmByteLength, `${diagnosticPath}.wasmByteLength`);
  exactSha(report.observedProjectionSha256, `${diagnosticPath}.observedProjectionSha256`);
  exactSha(report.runtimeAbiContractSha256, `${diagnosticPath}.runtimeAbiContractSha256`);
  exactString(report.runtimeAbiManifestId, `${diagnosticPath}.runtimeAbiManifestId`);
  exactValue(report.rawWasmVerified, true, `${diagnosticPath}.rawWasmVerified`);
  exactValue(report.workerExecutionReady, false, `${diagnosticPath}.workerExecutionReady`);
  exactValue(report.releaseReady, false, `${diagnosticPath}.releaseReady`);
  const mismatches = exactArray(report.mismatches, `${diagnosticPath}.mismatches`);
  for (const [index, mismatch_] of mismatches.entries()) {
    exactString(mismatch_, `${diagnosticPath}.mismatches[${index}]`);
  }
  exactValue(
    report.exactInterfaceConformance,
    mismatches.length === 0,
    `${diagnosticPath}.exactInterfaceConformance`,
  );
  if (typeof report.projection !== "object" || report.projection === null ||
      Array.isArray(report.projection)) {
    invalid(`${diagnosticPath}.projection`, "expected a projection object");
  }
  return Object.freeze({
    sha256: sha256(snapshot.bytes),
    byteLength: snapshot.bytes.byteLength,
    exactInterfaceConformance: report.exactInterfaceConformance,
  });
}

function decodeBuildExecutionEvidence(bytes, diagnosticPath, lockId, sourceSetSha256, body, llvm) {
  let value;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch (cause) {
    invalid(`${diagnosticPath}.buildExecutionEvidence`, "evidence must be strict UTF-8 JSON", { cause });
  }
  let canonical;
  try {
    canonical = canonicalJsonBytes(value);
  } catch (cause) {
    invalid(`${diagnosticPath}.buildExecutionEvidence`, "evidence is outside canonical JSON", { cause });
  }
  if (!equalBytes(bytes, canonical)) {
    invalid(`${diagnosticPath}.buildExecutionEvidence`, "evidence must use canonical JSON bytes");
  }
  const top = exactObject(value, [
    "schema", "version", "authority", "lockId", "builder", "runtimeClosure", "isolation",
    "llvmSourceArchive", "execution", "sidecarMaterialization", "claims",
  ], diagnosticPath);
  exactValue(top.schema, BUILD_EXECUTION_SCHEMA, `${diagnosticPath}.schema`);
  exactValue(top.version, 2, `${diagnosticPath}.version`);
  exactValue(top.authority, "build-execution-observation-only", `${diagnosticPath}.authority`);
  exactValue(top.lockId, lockId, `${diagnosticPath}.lockId`);
  const builder = validateBuilder(top.builder, `${diagnosticPath}.builder`, body.builder);
  const runtimeClosure = validateRuntimeClosure(
    top.runtimeClosure,
    `${diagnosticPath}.runtimeClosure`,
    lockId,
    sourceSetSha256,
  );
  const isolation = validateIsolation(top.isolation, `${diagnosticPath}.isolation`);
  const llvmSourceArchive = validateLlvm(top.llvmSourceArchive, `${diagnosticPath}.llvmSourceArchive`, llvm);
  const execution = validateExecution(top.execution, `${diagnosticPath}.execution`, lockId, sourceSetSha256);
  const sidecarMaterialization = validateSidecar(
    top.sidecarMaterialization,
    `${diagnosticPath}.sidecarMaterialization`,
    execution,
  );
  validateClaims(top.claims, `${diagnosticPath}.claims`);
  return Object.freeze({
    builder,
    runtimeClosure,
    isolation,
    llvmSourceArchive,
    execution,
    sidecarMaterialization,
  });
}

function validateRuntimeClosure(value, path, lockId, sourceSetSha256) {
  const object = exactObject(value, [
    "observationSha256", "observationByteLength", "observation",
  ], path);
  const observation = exactObject(object.observation, [
    "schema", "version", "authority", "lockId", "extractorSourceSetSha256",
    "closureSha256", "fileCount", "files", "claims",
  ], `${path}.observation`);
  exactValue(
    observation.schema,
    "browsergrad.compiler.cpp-cute.build-runtime-closure",
    `${path}.observation.schema`,
  );
  exactValue(observation.version, 1, `${path}.observation.version`);
  exactValue(
    observation.authority,
    "staged-build-runtime-closure-observation-only",
    `${path}.observation.authority`,
  );
  exactValue(observation.lockId, lockId, `${path}.observation.lockId`);
  exactValue(
    observation.extractorSourceSetSha256,
    sourceSetSha256,
    `${path}.observation.extractorSourceSetSha256`,
  );
  exactSha(observation.closureSha256, `${path}.observation.closureSha256`);
  const fileCount = boundedInteger(
    observation.fileCount,
    1,
    MAX_RUNTIME_CLOSURE_FILE_COUNT,
    `${path}.observation.fileCount`,
  );
  const files = exactArray(observation.files, `${path}.observation.files`);
  if (files.length !== fileCount) invalid(`${path}.observation.files`, "files must match fileCount");
  let previousPath;
  for (const [index, value_] of files.entries()) {
    const filePath = `${path}.observation.files[${index}]`;
    const file = exactObject(value_, ["kind", "path", "sha256", "byteLength"], filePath);
    if (file.kind !== "runtime" && file.kind !== "extractor") {
      invalid(`${filePath}.kind`, "kind must be runtime or extractor");
    }
    const relativePath = exactString(file.path, `${filePath}.path`);
    if (relativePath.length > 4_096 || normalize(relativePath) !== relativePath ||
        !/^(?!\/)(?!.*\\)(?!.*(?:^|\/)\.\.(?:\/|$))[A-Za-z0-9._@/-]+$/u.test(relativePath)) {
      invalid(`${filePath}.path`, "expected a safe relative path");
    }
    if (previousPath !== undefined && previousPath >= relativePath) {
      invalid(`${path}.observation.files`, "files must be strictly sorted and unique");
    }
    previousPath = relativePath;
    exactSha(file.sha256, `${filePath}.sha256`);
    boundedInteger(
      file.byteLength,
      0,
      MAX_RUNTIME_CLOSURE_FILE_BYTE_LENGTH,
      `${filePath}.byteLength`,
    );
  }
  exactValue(
    sha256(canonicalJsonBytes({ domain: RUNTIME_CLOSURE_HASH_DOMAIN, files })),
    observation.closureSha256,
    `${path}.observation.closureSha256`,
  );
  const claims = exactObject(observation.claims, [
    "exactReadableWorkspaceClosureVerified", "buildExecuted", "outputIdentityAuthorized",
    "reproducibilityVerified", "releaseReady",
  ], `${path}.observation.claims`);
  exactValue(
    claims.exactReadableWorkspaceClosureVerified,
    true,
    `${path}.observation.claims.exactReadableWorkspaceClosureVerified`,
  );
  for (const name of ["buildExecuted", "outputIdentityAuthorized", "reproducibilityVerified", "releaseReady"]) {
    exactValue(claims[name], false, `${path}.observation.claims.${name}`);
  }
  const observationBytes = canonicalJsonBytes(observation);
  exactValue(
    observationBytes.byteLength,
    boundedInteger(
      object.observationByteLength,
      1,
      MAX_RUNTIME_CLOSURE_OBSERVATION_BYTE_LENGTH,
      `${path}.observationByteLength`,
    ),
    `${path}.observationByteLength`,
  );
  exactValue(
    sha256(observationBytes),
    exactSha(object.observationSha256, `${path}.observationSha256`),
    `${path}.observationSha256`,
  );
  return Object.freeze({
    observationSha256: object.observationSha256,
    observationByteLength: object.observationByteLength,
    observation: Object.freeze({ ...observation, files: Object.freeze([...files]) }),
  });
}

function validateBuilder(value, path, expected) {
  const object = exactObject(value, [
    "schema", "version", "platform", "platformManifestDigest", "imageConfigDigest",
  ], path);
  exactValue(object.schema, "browsergrad.compiler.cpp-cute.builder-container-observation", `${path}.schema`);
  exactValue(object.version, 1, `${path}.version`);
  exactValue(object.platform, "linux/amd64", `${path}.platform`);
  exactValue(object.platformManifestDigest, expected.platformManifestDigest, `${path}.platformManifestDigest`);
  exactValue(object.imageConfigDigest, expected.imageConfigDigest, `${path}.imageConfigDigest`);
  return Object.freeze({
    schema: object.schema,
    version: object.version,
    platform: object.platform,
    platformManifestDigest: object.platformManifestDigest,
    imageConfigDigest: object.imageConfigDigest,
  });
}

function validateIsolation(value, path) {
  const object = exactObject(value, [
    "networkInterfaces", "effectiveCapabilities", "noNewPrivileges",
    "rootFilesystemReadOnly", "inputMountsReadOnly", "workMountReadWrite",
  ], path);
  const interfaces = exactArray(object.networkInterfaces, `${path}.networkInterfaces`);
  if (interfaces.length !== 1) invalid(`${path}.networkInterfaces`, "expected only loopback");
  exactValue(interfaces[0], "lo", `${path}.networkInterfaces[0]`);
  const capability = exactString(object.effectiveCapabilities, `${path}.effectiveCapabilities`);
  if (!/^0+$/u.test(capability)) invalid(`${path}.effectiveCapabilities`, "expected zero capabilities");
  for (const name of [
    "noNewPrivileges", "rootFilesystemReadOnly", "inputMountsReadOnly", "workMountReadWrite",
  ]) exactValue(object[name], true, `${path}.${name}`);
  return Object.freeze({
    networkInterfaces: Object.freeze(["lo"]),
    effectiveCapabilities: capability,
    noNewPrivileges: true,
    rootFilesystemReadOnly: true,
    inputMountsReadOnly: true,
    workMountReadWrite: true,
  });
}

function validateLlvm(value, path, expected) {
  const object = exactObject(value, ["sourceId", "sha256", "byteLength", "verified"], path);
  exactValue(object.sourceId, "llvm-project", `${path}.sourceId`);
  exactValue(object.sha256, expected.archiveSha256, `${path}.sha256`);
  exactValue(object.byteLength, expected.archiveByteLength, `${path}.byteLength`);
  exactValue(object.verified, true, `${path}.verified`);
  return Object.freeze({ ...object });
}

function validateExecution(value, path, lockId, sourceSetSha256) {
  const object = exactObject(value, [
    "authority", "lockId", "sourceSetSha256", "paths", "nativeTools", "stepCount", "steps",
    "factoryModulePath", "factoryModuleSha256", "factoryModuleByteLength",
    "wasmSidecarPath", "wasmSha256", "wasmByteLength",
    "linkMapPath", "linkMapSha256", "linkMapByteLength", "sourceVerified", "buildExecuted",
    "factoryModuleUtf8Validated", "webAssemblyValidated", "abiConformanceVerified",
    "outputIdentityAuthorized", "reproducibilityVerified", "releaseReady", "factoryModuleDistributed",
  ], path);
  exactValue(object.authority, "clang-wasm-build-execution-observation-only", `${path}.authority`);
  exactValue(object.lockId, lockId, `${path}.lockId`);
  exactValue(object.sourceSetSha256, sourceSetSha256, `${path}.sourceSetSha256`);
  const paths = validateExecutionPaths(object.paths, `${path}.paths`);
  const nativeTools = validateNativeTools(object.nativeTools, `${path}.nativeTools`, paths);
  exactValue(object.stepCount, 4, `${path}.stepCount`);
  const steps = validateSteps(object.steps, `${path}.steps`, paths);
  const factoryModulePath = exactPath(
    object.factoryModulePath,
    join(paths.stateRoot, "evidence", "generated", "clang-extractor.mjs"),
    `${path}.factoryModulePath`,
  );
  const wasmSidecarPath = exactPath(
    object.wasmSidecarPath,
    join(paths.stateRoot, "evidence", "generated", "clang-extractor.wasm"),
    `${path}.wasmSidecarPath`,
  );
  const linkMapPath = exactPath(
    object.linkMapPath,
    join(paths.stateRoot, "evidence", "clang-extractor.link.map"),
    `${path}.linkMapPath`,
  );
  const factoryModuleSha256 = exactSha(object.factoryModuleSha256, `${path}.factoryModuleSha256`);
  const wasmSha256 = exactSha(object.wasmSha256, `${path}.wasmSha256`);
  const linkMapSha256 = exactSha(object.linkMapSha256, `${path}.linkMapSha256`);
  const factoryModuleByteLength = boundedInteger(
    object.factoryModuleByteLength, 1, MAX_FACTORY_MODULE_BYTE_LENGTH, `${path}.factoryModuleByteLength`,
  );
  const wasmByteLength = boundedInteger(object.wasmByteLength, 8, MAX_WASM_BYTE_LENGTH, `${path}.wasmByteLength`);
  const linkMapByteLength = boundedInteger(object.linkMapByteLength, 1, MAX_LINK_MAP_BYTE_LENGTH, `${path}.linkMapByteLength`);
  for (const name of ["sourceVerified", "buildExecuted", "factoryModuleUtf8Validated", "webAssemblyValidated"]) {
    exactValue(object[name], true, `${path}.${name}`);
  }
  for (const name of [
    "abiConformanceVerified", "outputIdentityAuthorized", "reproducibilityVerified",
    "releaseReady", "factoryModuleDistributed",
  ]) exactValue(object[name], false, `${path}.${name}`);
  return Object.freeze({
    ...object,
    paths,
    nativeTools,
    steps,
    factoryModulePath,
    factoryModuleSha256,
    factoryModuleByteLength,
    wasmSidecarPath,
    wasmSha256,
    wasmByteLength,
    linkMapPath,
    linkMapSha256,
    linkMapByteLength,
  });
}

function validateExecutionPaths(value, path) {
  const object = exactObject(value, [
    "llvmProjectSourceRoot", "extractorSourceRoot", "nativeBuildRoot",
    "wasmBuildRoot", "outputRoot", "stateRoot",
  ], path);
  const paths = Object.freeze(Object.fromEntries(Object.keys(object).map((name) => [
    name,
    portableAbsolutePath(exactString(object[name], `${path}.${name}`), `${path}.${name}`),
  ])));
  const workRoot = dirname(paths.nativeBuildRoot);
  const exactChildren = {
    extractorSourceRoot: "staged-extractor-source",
    nativeBuildRoot: "native-tablegen",
    wasmBuildRoot: "clang-extractor-wasm",
    outputRoot: "output",
    stateRoot: "state",
  };
  for (const [name, child] of Object.entries(exactChildren)) {
    exactValue(paths[name], join(workRoot, child), `${path}.${name}`);
  }
  const values = Object.values(paths);
  for (let left = 0; left < values.length; left += 1) {
    for (let right = left + 1; right < values.length; right += 1) {
      if (pathsOverlap(values[left], values[right])) invalid(path, "execution paths must be non-overlapping");
    }
  }
  return paths;
}

function validateNativeTools(value, path, paths) {
  const object = exactObject(value, ["clangTablegen", "llvmTablegen"], path);
  return Object.freeze({
    clangTablegen: validateNativeTool(
      object.clangTablegen, `${path}.clangTablegen`, join(paths.nativeBuildRoot, "bin", "clang-tblgen"),
    ),
    llvmTablegen: validateNativeTool(
      object.llvmTablegen, `${path}.llvmTablegen`, join(paths.nativeBuildRoot, "bin", "llvm-tblgen"),
    ),
  });
}

function validateNativeTool(value, path, expectedPath) {
  const object = exactObject(value, ["path", "sha256", "byteLength"], path);
  return Object.freeze({
    path: exactPath(object.path, expectedPath, `${path}.path`),
    sha256: exactSha(object.sha256, `${path}.sha256`),
    byteLength: boundedInteger(object.byteLength, 1, MAX_NATIVE_TOOL_BYTE_LENGTH, `${path}.byteLength`),
  });
}

function validateSteps(value, path, paths) {
  const values = exactArray(value, path);
  if (values.length !== STEP_PROFILES.length) invalid(path, "expected exactly four build steps");
  return Object.freeze(values.map((step, index) => {
    const stepPath = `${path}[${index}]`;
    const object = exactObject(step, [
      "id", "stageId", "kind", "executable", "arguments", "cwd", "environment",
      "exitCode", "terminationSignal", "stdoutPath", "stdoutSha256", "stdoutByteLength",
      "stderrPath", "stderrSha256", "stderrByteLength",
    ], stepPath);
    const profile = STEP_PROFILES[index];
    exactValue(object.id, profile.id, `${stepPath}.id`);
    exactValue(object.stageId, profile.stageId, `${stepPath}.stageId`);
    exactValue(object.kind, profile.kind, `${stepPath}.kind`);
    const executable = portableAbsolutePath(exactString(object.executable, `${stepPath}.executable`), `${stepPath}.executable`);
    const arguments_ = exactStringArray(object.arguments, `${stepPath}.arguments`, 256, 16_384);
    const cwd = portableAbsolutePath(exactString(object.cwd, `${stepPath}.cwd`), `${stepPath}.cwd`);
    const expectedCwd = profile.stageId === "native-tablegen" ? paths.nativeBuildRoot : paths.wasmBuildRoot;
    exactValue(cwd, expectedCwd, `${stepPath}.cwd`);
    const environment = exactStringRecord(object.environment, `${stepPath}.environment`, 64, 32_768);
    exactValue(object.exitCode, 0, `${stepPath}.exitCode`);
    exactValue(object.terminationSignal, null, `${stepPath}.terminationSignal`);
    const logRoot = join(paths.stateRoot, "evidence", "build-logs");
    const stdoutPath = exactPath(object.stdoutPath, join(logRoot, `${profile.id}.stdout.log`), `${stepPath}.stdoutPath`);
    const stderrPath = exactPath(object.stderrPath, join(logRoot, `${profile.id}.stderr.log`), `${stepPath}.stderrPath`);
    return Object.freeze({
      ...object,
      executable,
      arguments: arguments_,
      cwd,
      environment,
      stdoutPath,
      stdoutSha256: exactSha(object.stdoutSha256, `${stepPath}.stdoutSha256`),
      stdoutByteLength: boundedInteger(object.stdoutByteLength, 0, MAX_LOG_BYTE_LENGTH, `${stepPath}.stdoutByteLength`),
      stderrPath,
      stderrSha256: exactSha(object.stderrSha256, `${stepPath}.stderrSha256`),
      stderrByteLength: boundedInteger(object.stderrByteLength, 0, MAX_LOG_BYTE_LENGTH, `${stepPath}.stderrByteLength`),
    });
  }));
}

function validateSidecar(value, path, execution) {
  const object = exactObject(value, [
    "authority", "lockId", "sourceSetSha256", "generatedWasmSha256", "distributedWasmSha256",
    "wasmByteLength", "distributedWasmPath", "sidecarBytesMaterialized", "webAssemblyValidated",
    "abiConformanceVerified", "sourceVerified", "buildExecuted", "outputIdentityAuthorized",
    "reproducibilityVerified", "releaseReady", "factoryModuleDistributed",
  ], path);
  exactValue(object.authority, "wasm-sidecar-byte-materialization-observation-only", `${path}.authority`);
  exactValue(object.lockId, execution.lockId, `${path}.lockId`);
  exactValue(object.sourceSetSha256, execution.sourceSetSha256, `${path}.sourceSetSha256`);
  exactValue(object.generatedWasmSha256, execution.wasmSha256, `${path}.generatedWasmSha256`);
  exactValue(object.distributedWasmSha256, execution.wasmSha256, `${path}.distributedWasmSha256`);
  exactValue(object.wasmByteLength, execution.wasmByteLength, `${path}.wasmByteLength`);
  const distributedWasmPath = exactPath(
    object.distributedWasmPath,
    join(execution.paths.outputRoot, "browsergrad-cpp-cute", "clang-extractor.wasm"),
    `${path}.distributedWasmPath`,
  );
  for (const name of ["sidecarBytesMaterialized", "sourceVerified"]) exactValue(object[name], true, `${path}.${name}`);
  for (const name of [
    "webAssemblyValidated", "abiConformanceVerified", "buildExecuted", "outputIdentityAuthorized",
    "reproducibilityVerified", "releaseReady", "factoryModuleDistributed",
  ]) exactValue(object[name], false, `${path}.${name}`);
  return Object.freeze({ ...object, distributedWasmPath });
}

function validateClaims(value, path) {
  const object = exactObject(value, [
    "sourceArchiveVerified", "buildExecuted", "networkDuringBuildObservedDisabled",
    "exactReadableWorkspaceClosureVerified",
    "outputIdentityAuthorized", "reproducibilityVerified", "producerAttested", "releaseReady",
  ], path);
  for (const name of [
    "sourceArchiveVerified", "buildExecuted", "networkDuringBuildObservedDisabled",
    "exactReadableWorkspaceClosureVerified",
  ]) {
    exactValue(object[name], true, `${path}.${name}`);
  }
  for (const name of ["outputIdentityAuthorized", "reproducibilityVerified", "producerAttested", "releaseReady"]) {
    exactValue(object[name], false, `${path}.${name}`);
  }
}

async function verifyObservedFiles(root, evidence, path) {
  const execution = evidence.execution;
  const files = [
    [join(root, "state", "evidence", "generated", "clang-extractor.mjs"), execution.factoryModuleByteLength, execution.factoryModuleSha256, MAX_FACTORY_MODULE_BYTE_LENGTH, `${path}.factoryModule`],
    [join(root, "state", "evidence", "generated", "clang-extractor.wasm"), execution.wasmByteLength, execution.wasmSha256, MAX_WASM_BYTE_LENGTH, `${path}.generatedWasm`],
    [join(root, "output", "browsergrad-cpp-cute", "clang-extractor.wasm"), execution.wasmByteLength, execution.wasmSha256, MAX_WASM_BYTE_LENGTH, `${path}.distributedWasm`],
    [join(root, "state", "evidence", "clang-extractor.link.map"), execution.linkMapByteLength, execution.linkMapSha256, MAX_LINK_MAP_BYTE_LENGTH, `${path}.linkMap`],
  ];
  for (const [index, step] of execution.steps.entries()) {
    files.push(
      [join(root, "state", "evidence", "build-logs", `${step.id}.stdout.log`), step.stdoutByteLength, step.stdoutSha256, MAX_LOG_BYTE_LENGTH, `${path}.steps[${index}].stdout`],
      [join(root, "state", "evidence", "build-logs", `${step.id}.stderr.log`), step.stderrByteLength, step.stderrSha256, MAX_LOG_BYTE_LENGTH, `${path}.steps[${index}].stderr`],
    );
  }
  for (const [filePath, byteLength, expectedSha256, maximum, diagnosticPath] of files) {
    await hashRegularFile(filePath, maximum, byteLength, expectedSha256, diagnosticPath);
  }
}

function assertDistinctBuildPaths(first, second) {
  for (const [firstName, firstPath] of Object.entries(first)) {
    for (const [secondName, secondPath] of Object.entries(second)) {
      if (pathsOverlap(firstPath, secondPath)) {
        mismatch(
          `$comparison.paths.${firstName}.${secondName}`,
          "all clean-build source and build paths must be distinct and non-overlapping",
        );
      }
    }
  }
}

function assertCanonicalCommandParity(first, second) {
  const firstBytes = canonicalJsonBytes(normalizedCommandRecords(first));
  const secondBytes = canonicalJsonBytes(normalizedCommandRecords(second));
  if (!equalBytes(firstBytes, secondBytes)) {
    mismatch("$comparison.commands", "canonical commands or environments differ across clean builds");
  }
}

function normalizedCommandRecords(build) {
  const replacements = Object.entries({
    llvmProjectSourceRoot: "@LLVM_SOURCE@",
    extractorSourceRoot: "@EXTRACTOR_SOURCE@",
    nativeBuildRoot: "@NATIVE_BUILD@",
    wasmBuildRoot: "@WASM_BUILD@",
    outputRoot: "@OUTPUT@",
    stateRoot: "@STATE@",
  }).map(([name, replacement]) => [build.execution.paths[name], replacement])
    .sort((left, right) => right[0].length - left[0].length);
  const replace = (value) => replacements.reduce(
    (result, [path, replacement]) => result.split(path).join(replacement),
    value,
  );
  return build.execution.steps.map((step) => ({
    id: step.id,
    stageId: step.stageId,
    kind: step.kind,
    executable: replace(step.executable),
    arguments: step.arguments.map(replace),
    cwd: replace(step.cwd),
    environment: Object.fromEntries(Object.entries(step.environment).map(([name, value]) => [name, replace(value)])),
  }));
}

function assertIdentityParity(first, second) {
  const pairs = [
    [
      {
        sha256: first.runtimeClosure.observationSha256,
        byteLength: first.runtimeClosure.observationByteLength,
      },
      {
        sha256: second.runtimeClosure.observationSha256,
        byteLength: second.runtimeClosure.observationByteLength,
      },
      "$comparison.runtimeClosure",
    ],
    [first.execution.nativeTools.clangTablegen, second.execution.nativeTools.clangTablegen, "$comparison.nativeTools.clangTablegen"],
    [first.execution.nativeTools.llvmTablegen, second.execution.nativeTools.llvmTablegen, "$comparison.nativeTools.llvmTablegen"],
    [identity(first.execution, "factoryModule"), identity(second.execution, "factoryModule"), "$comparison.factoryModule"],
    [identity(first.execution, "wasm"), identity(second.execution, "wasm"), "$comparison.wasm"],
    [first.runtimeAbiReview, second.runtimeAbiReview, "$comparison.runtimeAbiReview"],
    [identity(first.execution, "linkMap"), identity(second.execution, "linkMap"), "$comparison.linkMap"],
  ];
  for (const [left, right, path] of pairs) {
    if (left.sha256 !== right.sha256 || left.byteLength !== right.byteLength) {
      mismatch(path, "clean-build byte identities differ");
    }
  }
}

function identity(execution, prefix) {
  return {
    sha256: execution[`${prefix}Sha256`],
    byteLength: execution[`${prefix}ByteLength`],
  };
}

function buildIdentity(ordinal, build) {
  return Object.freeze({
    ordinal,
    buildExecutionEvidenceSha256: build.evidenceSha256,
    buildExecutionEvidenceByteLength: build.evidenceByteLength,
    runtimeClosureSha256: build.runtimeClosure.observation.closureSha256,
    runtimeClosureObservationSha256: build.runtimeClosure.observationSha256,
    runtimeClosureObservationByteLength: build.runtimeClosure.observationByteLength,
    nativeTools: Object.freeze({
      clangTablegenSha256: build.execution.nativeTools.clangTablegen.sha256,
      clangTablegenByteLength: build.execution.nativeTools.clangTablegen.byteLength,
      llvmTablegenSha256: build.execution.nativeTools.llvmTablegen.sha256,
      llvmTablegenByteLength: build.execution.nativeTools.llvmTablegen.byteLength,
    }),
    factoryModuleSha256: build.execution.factoryModuleSha256,
    factoryModuleByteLength: build.execution.factoryModuleByteLength,
    wasmSha256: build.execution.wasmSha256,
    wasmByteLength: build.execution.wasmByteLength,
    runtimeAbiReviewSha256: build.runtimeAbiReview.sha256,
    runtimeAbiReviewByteLength: build.runtimeAbiReview.byteLength,
    runtimeAbiReviewExactInterfaceConformance:
      build.runtimeAbiReview.exactInterfaceConformance,
    linkMapSha256: build.execution.linkMapSha256,
    linkMapByteLength: build.execution.linkMapByteLength,
  });
}

async function admitBuildRoot(path, diagnosticPath) {
  await admitPrivateDirectory(path, diagnosticPath);
  try {
    return await realpath(path);
  } catch (cause) {
    io(diagnosticPath, "failed to resolve clean-build root", { cause });
  }
}

async function admitPrivateDirectory(path, diagnosticPath) {
  let stat;
  try {
    stat = await lstat(path, { bigint: true });
  } catch (cause) {
    io(diagnosticPath, "failed to inspect private directory", { cause });
  }
  if (!stat.isDirectory() || stat.isSymbolicLink()) invalid(diagnosticPath, "expected a non-symlink directory");
  if (typeof process.getuid !== "function" || stat.uid !== BigInt(process.getuid())) {
    invalid(diagnosticPath, "directory must be owned by the current user");
  }
  if ((stat.mode & 0o022n) !== 0n) invalid(diagnosticPath, "directory must not be writable by group or other users");
}

async function assertExactTree(root, expectedFiles, diagnosticPath) {
  const expected = new Set(expectedFiles);
  const observed = new Set();
  const maximumNodes = expectedFiles.length + new Set(expectedFiles.flatMap((path) => {
    const parts = path.split("/");
    return parts.slice(0, -1).map((_, index) => parts.slice(0, index + 1).join("/"));
  })).size;
  let nodes = 0;
  async function visit(directoryPath, relativeDirectory) {
    const directory = await opendir(directoryPath);
    try {
      while (true) {
        const entry = await directory.read();
        if (entry === null) break;
        nodes += 1;
        if (nodes > maximumNodes) invalid(diagnosticPath, "artifact tree contains undeclared entries");
        const relativePath = relativeDirectory === "" ? entry.name : `${relativeDirectory}/${entry.name}`;
        if (entry.isSymbolicLink()) invalid(`${diagnosticPath}/${relativePath}`, "artifact tree must not contain symbolic links");
        if (entry.isDirectory()) {
          if (![...expected].some((path) => path.startsWith(`${relativePath}/`))) {
            invalid(`${diagnosticPath}/${relativePath}`, "artifact tree contains an undeclared directory");
          }
          await visit(join(directoryPath, entry.name), relativePath);
        } else if (entry.isFile() && expected.has(relativePath)) {
          observed.add(relativePath);
        } else {
          invalid(`${diagnosticPath}/${relativePath}`, "artifact tree contains an undeclared node");
        }
      }
    } finally {
      await directory.close().catch((cause) => {
        if (!isNodeError(cause, "ERR_DIR_CLOSED")) throw cause;
      });
    }
  }
  try {
    await visit(root, "");
  } catch (cause) {
    if (cause instanceof CppCuteClangWasmReproducibilityError) throw cause;
    io(diagnosticPath, "failed to inspect exact artifact tree", { cause });
  }
  if (observed.size !== expected.size) invalid(diagnosticPath, "artifact tree is missing declared files");
}

async function readSmallFile(path, maximumByteLength, diagnosticPath) {
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!before.isFile() || before.size <= 0n || before.size > BigInt(maximumByteLength)) {
      invalid(diagnosticPath, "file type or length is outside the admitted range");
    }
    const bytes = new Uint8Array(Number(before.size));
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesRead === 0) conflict(diagnosticPath, "file became shorter while reading");
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileSnapshot(before, after)) conflict(diagnosticPath, "file changed while reading");
    return Object.freeze({ bytes });
  } catch (cause) {
    if (cause instanceof CppCuteClangWasmReproducibilityError) throw cause;
    io(diagnosticPath, "failed to read bounded evidence", { cause });
  } finally {
    await handle?.close();
  }
}

async function hashRegularFile(path, maximumByteLength, exactByteLength, expectedSha256, diagnosticPath) {
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!before.isFile() || before.size !== BigInt(exactByteLength) || before.size > BigInt(maximumByteLength)) {
      mismatch(`${diagnosticPath}.byteLength`, "observed file length differs from build evidence");
    }
    const buffer = new Uint8Array(HASH_BUFFER_BYTE_LENGTH);
    const hash = createHash("sha256");
    let offset = 0;
    while (offset < exactByteLength) {
      const maximum = Math.min(buffer.byteLength, exactByteLength - offset);
      const { bytesRead } = await handle.read(buffer, 0, maximum, offset);
      if (bytesRead === 0) conflict(diagnosticPath, "file became shorter while hashing");
      hash.update(buffer.subarray(0, bytesRead));
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileSnapshot(before, after)) conflict(diagnosticPath, "file changed while hashing");
    if (hash.digest("hex") !== expectedSha256) mismatch(`${diagnosticPath}.sha256`, "observed file digest differs from build evidence");
  } catch (cause) {
    if (cause instanceof CppCuteClangWasmReproducibilityError) throw cause;
    io(diagnosticPath, "failed to hash observed build file", { cause });
  } finally {
    await handle?.close();
  }
}

function exactObject(value, keys, path) {
  if (typeof value !== "object" || value === null || Array.isArray(value) || Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(path, "expected a plain object");
  }
  const object = /** @type {Record<string, unknown>} */ (value);
  const observed = Object.keys(object);
  if (observed.length !== keys.length || observed.some((key) => !keys.includes(key))) {
    invalid(path, `expected exactly fields ${keys.join(", ")}`);
  }
  return object;
}

function exactArray(value, path) {
  if (!Array.isArray(value)) invalid(path, "expected an array");
  return value;
}

function exactString(value, path) {
  if (typeof value !== "string") invalid(path, "expected a string");
  return value;
}

function exactStringArray(value, path, maximumCount, maximumStringLength) {
  const array = exactArray(value, path);
  if (array.length > maximumCount) invalid(path, "array exceeds its admitted item count");
  return Object.freeze(array.map((item, index) => {
    const string = exactString(item, `${path}[${index}]`);
    if (string.length > maximumStringLength) invalid(`${path}[${index}]`, "string exceeds its admitted length");
    return string;
  }));
}

function exactStringRecord(value, path, maximumCount, maximumStringLength) {
  if (typeof value !== "object" || value === null || Array.isArray(value) ||
      Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(path, "expected a plain string record");
  }
  const object = exactObject(value, Object.keys(value), path);
  const entries = Object.entries(object);
  if (entries.length > maximumCount) invalid(path, "record exceeds its admitted field count");
  return Object.freeze(Object.fromEntries(entries.map(([name, item]) => {
    if (name.length === 0 || name.length > 256) invalid(`${path}.${name}`, "environment name length is invalid");
    const string = exactString(item, `${path}.${name}`);
    if (string.length > maximumStringLength) invalid(`${path}.${name}`, "environment value exceeds its admitted length");
    return [name, string];
  })));
}

function exactSha(value, path) {
  const string = exactString(value, path);
  if (!SHA256_HEX.test(string)) invalid(path, "expected lowercase SHA-256 hex");
  return string;
}

function exactPath(value, expected, path) {
  const observed = portableAbsolutePath(exactString(value, path), path);
  exactValue(observed, expected, path);
  return observed;
}

function boundedInteger(value, minimum, maximum, path) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    invalid(path, `expected an integer from ${minimum} through ${maximum}`);
  }
  return value;
}

function portableAbsolutePath(value, path) {
  if (!isAbsolute(value) || normalize(value) !== value || value === "/" || value.endsWith("/") ||
      value.length > 4_096 || !PORTABLE_ABSOLUTE_PATH.test(value)) {
    invalid(path, "expected a normalized portable absolute POSIX path");
  }
  return value;
}

function exactValue(observed, expected, path) {
  if (observed !== expected) invalid(path, `expected ${String(expected)}`);
}

function pathsOverlap(left, right) {
  return left === right || left.startsWith(`${right}/`) || right.startsWith(`${left}/`);
}

function sameFileSnapshot(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.size === right.size &&
    left.mtimeNs === right.mtimeNs && left.ctimeNs === right.ctimeNs;
}

function equalBytes(left, right) {
  return left.byteLength === right.byteLength && timingSafeEqual(left, right);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function isNodeError(value, code) {
  return typeof value === "object" && value !== null && "code" in value && value.code === code;
}

function invalid(path, message, options) {
  throw new CppCuteClangWasmReproducibilityError(INVALID, path, message, options);
}

function mismatch(path, message, options) {
  throw new CppCuteClangWasmReproducibilityError(MISMATCH, path, message, options);
}

function conflict(path, message, options) {
  throw new CppCuteClangWasmReproducibilityError(CONFLICT, path, message, options);
}

function io(path, message, options) {
  throw new CppCuteClangWasmReproducibilityError(IO, path, message, options);
}

const mainUrl = process.argv[1] === undefined ? undefined : pathToFileURL(resolve(process.argv[1])).href;
if (mainUrl === import.meta.url) {
  try {
    const arguments_ = parseCppCuteClangWasmReproducibilityArguments(process.argv.slice(2));
    const evidence = await verifyCppCuteClangWasmReproducibility({
      firstRoot: arguments_["first-root"],
      secondRoot: arguments_["second-root"],
    });
    const result = await writeCppCuteClangWasmReproducibilityEvidence(arguments_.output, evidence);
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown reproducibility-verifier failure");
    process.stderr.write(`${JSON.stringify({
      name: error.name,
      message: error.message,
      ...ownErrorString(error, "code"),
      ...ownErrorString(error, "path"),
    })}\n`);
    process.exitCode = 1;
  }
}

function ownErrorString(error, name) {
  const descriptor = Object.getOwnPropertyDescriptor(error, name);
  return descriptor !== undefined && "value" in descriptor && typeof descriptor.value === "string"
    ? { [name]: descriptor.value }
    : {};
}
