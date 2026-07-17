import {
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  canonicalJsonBytes,
  deepFreezeJson,
  encodeWireU64,
  hashCanonicalJson,
  sha256Hex,
  wireIntegerToBigInt,
  type DecodeLimits,
  type JsonObject,
  type JsonValue,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  unwrapPreparedCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
} from "./cpp_cute_browser_build_lock.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE,
} from "./resources/cpp_cute_browser_runtime_abi_v1.js";
import {
  CPP_CUTE_BROWSER_VFS_CONTENT_SET_HASH_DOMAIN,
  CPP_CUTE_BROWSER_VFS_CONTENT_SET_MAX_STRING_BYTES,
  CPP_CUTE_BROWSER_VFS_PACK_HEADER_BYTES,
  CppCuteBrowserVfsPackError,
  deriveCppCuteBrowserVfsContentSetSha256,
  encodeCppCuteBrowserVfsPack,
  inspectCppCuteBrowserVfsPack,
  unwrapInspectedCppCuteBrowserVfsPack,
  type CppCuteBrowserVfsPackEntry,
} from "./cpp_cute_browser_vfs_pack.js";
import {
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type CppCuteFrontendDependencyKind,
  type CppCuteFrontendIncludeRoot,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";

export const CPP_CUTE_BROWSER_HEADER_PACK_SELECTION_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-header-pack-selection";
export const CPP_CUTE_BROWSER_HEADER_PACK_SELECTION_MAJOR = 1;
export const CPP_CUTE_BROWSER_HEADER_PACK_SELECTION_MINOR = 0;
export const CPP_CUTE_BROWSER_HEADER_PACK_SELECTION_BYTE_LIMIT = 32 * 1024 * 1024;

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const WIRE_U64 = /^(?:0|[1-9][0-9]{0,19})$/u;
const TEXT_ENCODER = new TextEncoder();
const MAX_SELECTION_FILES = 100_000;
const MAX_LICENSE_COMPONENTS_PER_FILE = 16;
const MAX_SELECTION_FILE_INVENTORY_LOGICAL_BYTES = 24 * 1024 * 1024;
const SELECTION_CANONICAL_LIMITS = Object.freeze({
  maxDocumentBytes: CPP_CUTE_BROWSER_HEADER_PACK_SELECTION_BYTE_LIMIT,
  maxDepth: 16,
  maxNodes: 1_000_000,
  maxStringBytes: CPP_CUTE_BROWSER_VFS_CONTENT_SET_MAX_STRING_BYTES,
  maxArrayLength: MAX_SELECTION_FILES,
  maxObjectProperties: 64,
  maxRank: 64,
  maxIntegerBits: 256,
  maxArithmeticOperations: 200_000,
} satisfies DecodeLimits);
const VFS_PACK_ENTRY_FIXED_BYTES = 2 + 8 + 32;
const CONTENT_SET_JSON_ROOT_FIXED_BYTES = TEXT_ENCODER.encode(
  `{"domain":"${CPP_CUTE_BROWSER_VFS_CONTENT_SET_HASH_DOMAIN}","files":[]}`,
).byteLength;
const CONTENT_SET_JSON_ENTRY_FIXED_BYTES = TEXT_ENCODER.encode(
  `{"byteLength":"","contentSha256":"${"0".repeat(64)}","virtualPath":""}`,
).byteLength;
const CONTENT_SET_ROOT_STRING_BYTES = TEXT_ENCODER.encode(
  `domainfiles${CPP_CUTE_BROWSER_VFS_CONTENT_SET_HASH_DOMAIN}`,
).byteLength;
const CONTENT_SET_ENTRY_FIXED_STRING_BYTES = TEXT_ENCODER.encode(
  `byteLengthcontentSha256virtualPath${"0".repeat(64)}`,
).byteLength;
const REQUIRED_NON_PACK_ASSET_COUNT = 3;
const HEADER_PACK_INTEGRATION_BLOCKER =
  "header-pack-acquisition-materialization-and-build-integration";
const HEADER_PACK_NOTICE_BYTES_BLOCKER =
  "header-pack-exact-notice-bytes-verification";
const HEADER_PACK_LICENSE_MAP_BLOCKER =
  "header-pack-externally-reviewed-distributed-file-license-map";
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export type CppCuteBrowserHeaderPackRole =
  | "compiler-resource"
  | "cxx-standard-library"
  | "cuda"
  | "cute"
  | "linux-sysroot";

export interface CppCuteBrowserHeaderPackSelectionFileV1 extends JsonObject {
  readonly virtualPath: string;
  readonly contentSha256: string;
  readonly byteLength: WireU64;
  readonly licenseComponentIds: readonly string[];
}

export interface CppCuteBrowserHeaderPackSelectionPackV1 extends JsonObject {
  readonly ordinal: number;
  readonly role: CppCuteBrowserHeaderPackRole;
  readonly assetKind: "compiler-resource-pack" | "dependency-header-pack";
  readonly outputPath: string;
  readonly includeRootId: string;
  readonly mountedVirtualRoot: string;
  readonly dependencyId: string | null;
  readonly dependencyKind: CppCuteFrontendDependencyKind | null;
  readonly version: string;
  readonly revision: string;
  readonly expectedContentSetSha256: string;
  readonly fileCount: WireU64;
  readonly fileContentByteLength: WireU64;
  readonly files: readonly CppCuteBrowserHeaderPackSelectionFileV1[];
}

export type CppCuteBrowserHeaderPackNoticeV1 =
  | (JsonObject & {
      readonly componentId: string;
      readonly reviewStatus: "reviewed";
      readonly licenseExpression: string;
      readonly sourcePath: string;
      readonly noticeOutputPath: string;
      readonly noticeSha256: string;
      readonly noticeByteLength: WireU64;
      readonly appliesTo: readonly string[];
    })
  | (JsonObject & {
      readonly componentId: string;
      readonly reviewStatus: "unresolved";
      readonly intendedAsset: string;
      readonly reasonCode: string;
      readonly disposition: "blocks-release";
    });

export interface CppCuteBrowserHeaderPackSelectionBodyV1 extends JsonObject {
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly profileId: string;
  readonly profileHash: string;
  readonly compilationContractHash: string;
  readonly policy: JsonObject & {
    readonly scope: "complete-profile-header-sets-not-corpus-minimal-closures";
    readonly network: "forbidden";
    readonly buildSysrootReuse: "forbidden";
    readonly inputBytes: "offline-caller-supplied-exact-inventory-match-required";
    readonly noticeBytes: "exact-bytes-unverified";
    readonly fileLicenseMapping: "derived-notice-policy-only-external-review-required";
    readonly outputAuthority: "not-authorized";
    readonly releaseAuthority: "not-authorized";
  };
  readonly packs: readonly CppCuteBrowserHeaderPackSelectionPackV1[];
  readonly notices: readonly CppCuteBrowserHeaderPackNoticeV1[];
  readonly licenseReviewComplete: false;
  readonly releaseBlockerIds: readonly string[];
}

export interface CppCuteBrowserHeaderPackSelectionManifestV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_BROWSER_HEADER_PACK_SELECTION_SCHEMA;
  readonly version: JsonObject & {
    readonly major: typeof CPP_CUTE_BROWSER_HEADER_PACK_SELECTION_MAJOR;
    readonly minor: typeof CPP_CUTE_BROWSER_HEADER_PACK_SELECTION_MINOR;
  };
  readonly selectionId: string;
  readonly body: CppCuteBrowserHeaderPackSelectionBodyV1;
}

export interface CppCuteBrowserHeaderPackSelectionFileInput {
  readonly virtualPath: string;
  readonly contentSha256: string;
  readonly byteLength: WireU64;
  readonly licenseComponentIds: readonly string[];
}

export interface CppCuteBrowserHeaderPackSelectionPackInput {
  readonly includeRootId: string;
  readonly files: readonly CppCuteBrowserHeaderPackSelectionFileInput[];
}

export interface PrepareCppCuteBrowserHeaderPackSelectionInput {
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly packs: readonly CppCuteBrowserHeaderPackSelectionPackInput[];
}

export interface CppCuteBrowserHeaderPackOperationOptions {
  readonly signal?: AbortSignal;
}

declare const preparedHeaderPackSelectionBrand: unique symbol;

/**
 * Opaque authority over one deterministic, profile-complete header selection.
 * It proves no file acquisition, pack output, license approval, or release.
 */
export interface PreparedCppCuteBrowserHeaderPackSelection {
  readonly [preparedHeaderPackSelectionBrand]: true;
  readonly selectionId: string;
  readonly selectionSha256: string;
  readonly selectionByteLength: number;
  readonly buildInputLockId: string;
  readonly profileHash: string;
  readonly packCount: number;
  readonly fileCount: number;
  readonly licenseReviewComplete: false;
  readonly outputIdentityAuthorized: false;
  readonly releaseReady: false;
  readonly releaseBlockerIds: readonly string[];
}

export interface PreparedCppCuteBrowserHeaderPackSelectionRecord {
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly manifest: CppCuteBrowserHeaderPackSelectionManifestV1;
}

interface StoredHeaderPackSelection extends PreparedCppCuteBrowserHeaderPackSelectionRecord {
  readonly canonicalBytes: Uint8Array;
}

export interface CppCuteBrowserHeaderPackSourceFileInput {
  readonly includeRootId: string;
  readonly virtualPath: string;
  readonly bytes: Uint8Array;
}

export interface AssembleCppCuteBrowserHeaderPacksInput {
  readonly files: readonly CppCuteBrowserHeaderPackSourceFileInput[];
}

export interface CppCuteBrowserAssembledHeaderPackV1 {
  readonly ordinal: number;
  readonly role: CppCuteBrowserHeaderPackRole;
  readonly outputPath: string;
  readonly includeRootId: string;
  readonly mountedVirtualRoot: string;
  readonly packSha256: string;
  readonly packByteLength: WireU64;
  readonly fileContentByteLength: WireU64;
  readonly contentSetSha256: string;
  readonly fileCount: number;
}

declare const assembledHeaderPacksBrand: unique symbol;

/** Offline deterministic pack bytes. This is not asset, build, or release authority. */
export interface AssembledCppCuteBrowserHeaderPacks {
  readonly [assembledHeaderPacksBrand]: true;
  readonly selectionId: string;
  readonly buildInputLockId: string;
  readonly profileHash: string;
  readonly outputs: readonly CppCuteBrowserAssembledHeaderPackV1[];
  /** Conservative byte-copy upper bound enforced before each pack encode. */
  readonly peakWorkingSetUpperBoundByteLength: WireU64;
  readonly peakWorkingSetByteLimit: WireU64;
  readonly networkAccessed: false;
  readonly outputIdentityAuthorized: false;
  readonly buildExecutionObserved: false;
  readonly reproducibilityObserved: false;
  readonly releaseReady: false;
  readonly releaseBlockerIds: readonly string[];
}

export interface AssembledCppCuteBrowserHeaderPacksRecord {
  readonly selection: PreparedCppCuteBrowserHeaderPackSelection;
  readonly outputs: readonly CppCuteBrowserAssembledHeaderPackV1[];
}

interface StoredAssembledHeaderPacks extends AssembledCppCuteBrowserHeaderPacksRecord {
  readonly bytesByIncludeRootId: ReadonlyMap<string, Uint8Array>;
}

export type CppCuteBrowserHeaderPackAssemblyErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-UNVERIFIED";

export class CppCuteBrowserHeaderPackAssemblyError extends Error {
  constructor(
    readonly code: CppCuteBrowserHeaderPackAssemblyErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserHeaderPackAssemblyError";
  }
}

const PREPARED_SELECTIONS = new WeakMap<object, StoredHeaderPackSelection>();
const ASSEMBLED_PACKS = new WeakMap<object, StoredAssembledHeaderPacks>();

export async function prepareCppCuteBrowserHeaderPackSelection(
  input: PrepareCppCuteBrowserHeaderPackSelectionInput,
  options: CppCuteBrowserHeaderPackOperationOptions = {},
): Promise<PreparedCppCuteBrowserHeaderPackSelection> {
  const signal = operationSignal(options);
  throwIfAborted(signal);
  const values = exactDataRecord(input, "$.input", ["buildInputLock", "profile", "packs"]);
  const buildInputLock = values["buildInputLock"] as PreparedCppCuteBrowserBuildInputLock;
  const profile = values["profile"] as PreparedCppCuteFrontendProfile;
  const rawPacks = values["packs"];
  const lockRecord = unwrapPreparedCppCuteBrowserBuildInputLock(buildInputLock);
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(profile);
  const browserProfile = profileRecord.profile;
  if (browserProfile.deployment.buildProvenanceLockSha256 !== buildInputLock.resourceSha256) {
    mismatch(
      "$.input.profile",
      "browser profile does not bind the exact prepared build-input lock bytes",
    );
  }
  const packSpecs = derivePackSpecs(lockRecord.lock.body, browserProfile);
  const inputs = denseArray(rawPacks, "$.input.packs", packSpecs.length);
  if (inputs.length !== packSpecs.length) {
    mismatch("$.input.packs", "selection must cover every non-source profile include root exactly once");
  }
  const inputByRoot = new Map<string, {
    readonly files: unknown;
    readonly path: string;
  }>();
  for (const [index, rawPack] of inputs.entries()) {
    throwIfAborted(signal);
    const path = `$.input.packs[${index}]`;
    const fields = exactDataRecord(rawPack, path, ["includeRootId", "files"]);
    const includeRootId = boundedToken(fields["includeRootId"], `${path}.includeRootId`, 128);
    if (inputByRoot.has(includeRootId)) invalid(`${path}.includeRootId`, "duplicate include-root selection");
    inputByRoot.set(includeRootId, { files: fields["files"], path });
  }
  const notices = deriveNoticeInventory(lockRecord.lock.body, packSpecs);
  const releaseBlockerIds = selectionReleaseBlockers(buildInputLock);
  const noticeByComponent = new Map(notices.map((notice) => [notice.componentId, notice]));
  const packs: CppCuteBrowserHeaderPackSelectionPackV1[] = [];
  const parseBudget: SelectionParseBudget = {
    remainingFiles: MAX_SELECTION_FILES,
    remainingLogicalBytes: MAX_SELECTION_FILE_INVENTORY_LOGICAL_BYTES,
  };
  let declaredTotalFiles = 0;
  for (const spec of packSpecs) {
    const rawPack = inputByRoot.get(spec.root.includeRootId);
    if (rawPack === undefined) {
      mismatch(
        "$.input.packs",
        `selection is missing include root ${JSON.stringify(spec.root.includeRootId)}`,
      );
    }
    const declaredFileCount = denseArrayLength(rawPack.files, `${rawPack.path}.files`);
    declaredTotalFiles += declaredFileCount;
    if (declaredTotalFiles > MAX_SELECTION_FILES) {
      resource("$.input.packs", `selection exceeds ${MAX_SELECTION_FILES} files globally`);
    }
  }
  let totalSelectedFileContentBytes = 0n;
  for (const spec of packSpecs) {
    throwIfAborted(signal);
    const rawPack = inputByRoot.get(spec.root.includeRootId)!;
    const files = parseSelectionFiles(
      rawPack.files,
      `${rawPack.path}.files`,
      spec,
      noticeByComponent,
      parseBudget,
      signal,
    );
    const contentSetStringBytes = exactContentSetCanonicalStringByteLength(files);
    if (contentSetStringBytes >
        BigInt(CPP_CUTE_BROWSER_VFS_CONTENT_SET_MAX_STRING_BYTES)) {
      resource(
        `${rawPack.path}.files`,
        "content-set projection exceeds the explicit cumulative-string hash budget",
      );
    }
    const contentSetSha256 = await deriveSelectionContentSetSha256(
      files,
      `${rawPack.path}.files`,
    );
    throwIfAborted(signal);
    if (contentSetSha256 !== spec.expectedContentSetSha256) {
      hashMismatch(
        `${rawPack.path}.files`,
        "file inventory content-set hash differs from the exact profile header-set identity",
      );
    }
    const fileContentByteLength = files.reduce(
      (total, file) => total + wireIntegerToBigInt(file.byteLength),
      0n,
    );
    totalSelectedFileContentBytes += fileContentByteLength;
    if (fileContentByteLength > BigInt(browserProfile.deployment.assetLimits.maxAssetFileContentByteLength)) {
      resource(
        `${rawPack.path}.files`,
        "file inventory exceeds the profile per-asset file-content ceiling",
      );
    }
    packs.push({
      ordinal: spec.ordinal,
      role: spec.role,
      assetKind: spec.assetKind,
      outputPath: spec.outputPath,
      includeRootId: spec.root.includeRootId,
      mountedVirtualRoot: spec.root.virtualPath,
      dependencyId: spec.dependencyId,
      dependencyKind: spec.dependencyKind,
      version: spec.version,
      revision: spec.revision,
      expectedContentSetSha256: spec.expectedContentSetSha256,
      fileCount: encodeWireU64(BigInt(files.length)),
      fileContentByteLength: encodeWireU64(fileContentByteLength),
      files,
    });
  }
  if (totalSelectedFileContentBytes >
      BigInt(browserProfile.deployment.assetLimits.maxTotalFileContentByteLength)) {
    resource("$.input.packs", "selected header content exceeds the profile aggregate ceiling");
  }
  const body = deepFreezeJson({
    buildInputLockId: buildInputLock.lockId,
    buildInputLockResourceSha256: buildInputLock.resourceSha256,
    profileId: profile.profileId,
    profileHash: profile.profileHash,
    compilationContractHash: profile.compilationContractHash,
    policy: {
      scope: "complete-profile-header-sets-not-corpus-minimal-closures",
      network: "forbidden",
      buildSysrootReuse: "forbidden",
      inputBytes: "offline-caller-supplied-exact-inventory-match-required",
      noticeBytes: "exact-bytes-unverified",
      fileLicenseMapping: "derived-notice-policy-only-external-review-required",
      outputAuthority: "not-authorized",
      releaseAuthority: "not-authorized",
    },
    packs,
    notices,
    licenseReviewComplete: false,
    releaseBlockerIds,
  }) as CppCuteBrowserHeaderPackSelectionBodyV1;
  const selectionHashProjection = {
    domain: "browsergrad.compiler.cpp-cute.browser-header-pack-selection.v1",
    body,
  } as const satisfies JsonObject;
  preflightSelectionCanonicalJson(selectionHashProjection, "$.selectionHash");
  const selectionHash = await hashSelectionCanonicalJson(
    selectionHashProjection,
    "$.selectionHash",
  );
  throwIfAborted(signal);
  const selectionId = `bg.cpp.browser-header-pack-selection.sha256.${selectionHash}`;
  const manifest = deepFreezeJson({
    schema: CPP_CUTE_BROWSER_HEADER_PACK_SELECTION_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_HEADER_PACK_SELECTION_MAJOR,
      minor: CPP_CUTE_BROWSER_HEADER_PACK_SELECTION_MINOR,
    },
    selectionId,
    body,
  }) as CppCuteBrowserHeaderPackSelectionManifestV1;
  preflightSelectionCanonicalJson(manifest, "$.selection");
  const canonicalBytes = canonicalSelectionJsonBytes(manifest, "$.selection");
  if (canonicalBytes.byteLength > CPP_CUTE_BROWSER_HEADER_PACK_SELECTION_BYTE_LIMIT) {
    resource("$.input.packs", "canonical selection manifest exceeds its fixed byte ceiling");
  }
  const selectionSha256 = await hashBytes(canonicalBytes, "$.selection");
  throwIfAborted(signal);
  const prepared = Object.freeze({
    selectionId,
    selectionSha256,
    selectionByteLength: canonicalBytes.byteLength,
    buildInputLockId: buildInputLock.lockId,
    profileHash: profile.profileHash,
    packCount: packs.length,
    fileCount: declaredTotalFiles,
    licenseReviewComplete: false,
    outputIdentityAuthorized: false,
    releaseReady: false,
    releaseBlockerIds,
  }) as PreparedCppCuteBrowserHeaderPackSelection;
  PREPARED_SELECTIONS.set(prepared, Object.freeze({
    buildInputLock,
    profile,
    manifest,
    canonicalBytes: new Uint8Array(canonicalBytes),
  }));
  return prepared;
}

export function canonicalCppCuteBrowserHeaderPackSelectionBytes(
  selection: PreparedCppCuteBrowserHeaderPackSelection,
): Uint8Array {
  return new Uint8Array(storedSelection(selection).canonicalBytes);
}

export function unwrapPreparedCppCuteBrowserHeaderPackSelection(
  selection: PreparedCppCuteBrowserHeaderPackSelection,
): PreparedCppCuteBrowserHeaderPackSelectionRecord {
  const stored = storedSelection(selection);
  return Object.freeze({
    buildInputLock: stored.buildInputLock,
    profile: stored.profile,
    manifest: stored.manifest,
  });
}

export async function assembleCppCuteBrowserHeaderPacks(
  selection: PreparedCppCuteBrowserHeaderPackSelection,
  input: AssembleCppCuteBrowserHeaderPacksInput,
  options: CppCuteBrowserHeaderPackOperationOptions = {},
): Promise<AssembledCppCuteBrowserHeaderPacks> {
  const stored = storedSelection(selection);
  const signal = operationSignal(options);
  throwIfAborted(signal);
  const inputFields = exactDataRecord(input, "$.input", ["files"]);
  const expectedFiles = new Map<string, CppCuteBrowserHeaderPackSelectionFileV1>();
  for (const pack of stored.manifest.body.packs) {
    for (const file of pack.files) {
      expectedFiles.set(fileKey(pack.includeRootId, file.virtualPath), file);
    }
  }
  const rawFiles = inputFields["files"];
  if (denseArrayLength(rawFiles, "$.input.files") !== expectedFiles.size) {
    mismatch("$.input.files", "offline file set must equal the prepared selection exactly");
  }
  const grouped = new Map<string, IndexedSourceFile[]>();
  const seen = new Set<string>();
  for (let index = 0; index < expectedFiles.size; index += 1) {
    throwIfAborted(signal);
    const path = `$.input.files[${index}]`;
    const rawFile = denseArrayEntry(rawFiles, index, "$.input.files");
    const fields = exactDataRecord(rawFile, path, ["includeRootId", "virtualPath", "bytes"]);
    const includeRootId = boundedToken(fields["includeRootId"], `${path}.includeRootId`, 128);
    const virtualPath = portableRelativePathWithBytes(
      fields["virtualPath"],
      `${path}.virtualPath`,
    ).value;
    const key = fileKey(includeRootId, virtualPath);
    const expected = expectedFiles.get(key);
    if (expected === undefined) mismatch(path, "offline file is outside the prepared exact inventory");
    if (seen.has(key)) invalid(path, "duplicate offline file");
    seen.add(key);
    let inspection: ReturnType<typeof inspectUnsharedPlainUint8Array>;
    try {
      inspection = inspectUnsharedPlainUint8Array(fields["bytes"]);
    } catch (cause) {
      invalid(`${path}.bytes`, "bytes must be an unshared plain Uint8Array", { cause });
    }
    if (BigInt(inspection.byteLength) !== wireIntegerToBigInt(expected.byteLength)) {
      hashMismatch(`${path}.bytes`, "offline file length differs from the prepared inventory");
    }
    const files = grouped.get(includeRootId) ?? [];
    files.push(Object.freeze({
      inputIndex: index,
      virtualPath,
      rawBytes: fields["bytes"],
      expected,
    }));
    grouped.set(includeRootId, files);
  }
  if (seen.size !== expectedFiles.size) {
    mismatch("$.input.files", "offline file set is missing prepared inventory entries");
  }

  const profile = unwrapPreparedCppCuteBrowserFrontendProfile(stored.profile).profile;
  const outputs: CppCuteBrowserAssembledHeaderPackV1[] = [];
  const bytesByIncludeRootId = new Map<string, Uint8Array>();
  let totalPackBytes = 0n;
  let totalFileContentBytes = 0n;
  let peakWorkingSetUpperBoundBytes = 0n;
  const peakWorkingSetByteLimit = BigInt(
    profile.deployment.compilerRuntime.virtualFileSystem.maxRetainedHostPackByteLength,
  );
  for (const pack of stored.manifest.body.packs) {
    throwIfAborted(signal);
    const workingSet = conservativePackAssemblyByteCopyProjection(pack, totalPackBytes);
    peakWorkingSetUpperBoundBytes = maxBigInt(
      peakWorkingSetUpperBoundBytes,
      workingSet.peakBytes,
    );
    if (workingSet.peakBytes > peakWorkingSetByteLimit) {
      resource(
        `$.packs[${pack.ordinal}].workingSet`,
        `pack assembly peak upper bound exceeds ${peakWorkingSetByteLimit} bytes before encode`,
      );
    }
    const result = await assembleOneHeaderPack(
      pack,
      grouped.get(pack.includeRootId) ?? [],
      profile,
      signal,
    );
    if (BigInt(result.bytes.byteLength) !== workingSet.packByteLength) {
      hashMismatch(
        `$.packs[${pack.ordinal}]`,
        "encoded pack length differs from the preflight working-set projection",
      );
    }
    const prospectiveTotalPackBytes = totalPackBytes + BigInt(result.bytes.byteLength);
    const prospectiveTotalFileContentBytes = totalFileContentBytes +
      wireIntegerToBigInt(result.output.fileContentByteLength);
    if (prospectiveTotalPackBytes > BigInt(profile.deployment.assetLimits.maxTotalCompressedByteLength) ||
        prospectiveTotalPackBytes > BigInt(profile.deployment.assetLimits.maxTotalUnpackedByteLength) ||
        prospectiveTotalPackBytes >
          BigInt(profile.deployment.compilerRuntime.virtualFileSystem.maxRetainedHostPackByteLength)) {
      resource("$.packs", "aggregate identity packs exceed the profile retained/asset byte ceilings");
    }
    if (prospectiveTotalFileContentBytes >
        BigInt(profile.deployment.assetLimits.maxTotalFileContentByteLength)) {
      resource("$.packs", "aggregate header content exceeds the profile total file-content ceiling");
    }
    totalPackBytes = prospectiveTotalPackBytes;
    totalFileContentBytes = prospectiveTotalFileContentBytes;
    outputs.push(result.output);
    bytesByIncludeRootId.set(pack.includeRootId, result.bytes);
  }
  if (outputs.length + REQUIRED_NON_PACK_ASSET_COUNT > profile.deployment.assetLimits.maxAssets) {
    resource("$.packs", "header packs leave no profile capacity for required non-pack assets");
  }
  const assembled = Object.freeze({
    selectionId: selection.selectionId,
    buildInputLockId: selection.buildInputLockId,
    profileHash: selection.profileHash,
    outputs: Object.freeze(outputs),
    peakWorkingSetUpperBoundByteLength: encodeWireU64(peakWorkingSetUpperBoundBytes),
    peakWorkingSetByteLimit: encodeWireU64(peakWorkingSetByteLimit),
    networkAccessed: false,
    outputIdentityAuthorized: false,
    buildExecutionObserved: false,
    reproducibilityObserved: false,
    releaseReady: false,
    releaseBlockerIds: Object.freeze([...selection.releaseBlockerIds]),
  }) as AssembledCppCuteBrowserHeaderPacks;
  ASSEMBLED_PACKS.set(assembled, Object.freeze({
    selection,
    outputs: assembled.outputs,
    bytesByIncludeRootId,
  }));
  return assembled;
}

export function copyAssembledCppCuteBrowserHeaderPackBytes(
  assembled: AssembledCppCuteBrowserHeaderPacks,
  includeRootId: string,
): Uint8Array {
  const stored = storedAssembly(assembled);
  if (typeof includeRootId !== "string") invalid("$.includeRootId", "expected string");
  const bytes = stored.bytesByIncludeRootId.get(includeRootId);
  if (bytes === undefined) invalid("$.includeRootId", "assembled pack is absent");
  return new Uint8Array(bytes);
}

export function unwrapAssembledCppCuteBrowserHeaderPacks(
  assembled: AssembledCppCuteBrowserHeaderPacks,
): AssembledCppCuteBrowserHeaderPacksRecord {
  const stored = storedAssembly(assembled);
  return Object.freeze({ selection: stored.selection, outputs: stored.outputs });
}

interface PackSpec {
  readonly ordinal: number;
  readonly role: CppCuteBrowserHeaderPackRole;
  readonly assetKind: "compiler-resource-pack" | "dependency-header-pack";
  readonly outputPath: string;
  readonly intendedAsset: string;
  readonly noticeComponentId: string;
  readonly root: CppCuteFrontendIncludeRoot;
  readonly dependencyId: string | null;
  readonly dependencyKind: CppCuteFrontendDependencyKind | null;
  readonly version: string;
  readonly revision: string;
  readonly expectedContentSetSha256: string;
}

type BuildLockBody = ReturnType<typeof unwrapPreparedCppCuteBrowserBuildInputLock>["lock"]["body"];
type BrowserProfile = ReturnType<typeof unwrapPreparedCppCuteBrowserFrontendProfile>["profile"];

interface SelectionParseBudget {
  remainingFiles: number;
  remainingLogicalBytes: number;
}

interface IndexedSourceFile {
  readonly inputIndex: number;
  readonly virtualPath: string;
  readonly rawBytes: unknown;
  readonly expected: CppCuteBrowserHeaderPackSelectionFileV1;
}

interface ConservativePackAssemblyByteCopyProjection {
  readonly packByteLength: bigint;
  readonly contentSetCanonicalJsonByteLength: bigint;
  readonly peakBytes: bigint;
}

async function assembleOneHeaderPack(
  pack: CppCuteBrowserHeaderPackSelectionPackV1,
  indexedFiles: readonly IndexedSourceFile[],
  profile: BrowserProfile,
  signal: AbortSignal | undefined,
): Promise<{
  readonly output: CppCuteBrowserAssembledHeaderPackV1;
  readonly bytes: Uint8Array;
}> {
  if (indexedFiles.length !== pack.files.length) {
    mismatch(`$.packs[${pack.ordinal}]`, "offline pack is missing prepared inventory entries");
  }
  const files: Array<{ readonly virtualPath: string; readonly bytes: Uint8Array }> = [];
  for (const source of indexedFiles) {
    throwIfAborted(signal);
    const path = `$.input.files[${source.inputIndex}].bytes`;
    let inspection: ReturnType<typeof inspectUnsharedPlainUint8Array>;
    try {
      inspection = inspectUnsharedPlainUint8Array(source.rawBytes);
    } catch (cause) {
      invalid(path, "bytes must remain an unshared plain Uint8Array", { cause });
    }
    if (BigInt(inspection.byteLength) !== wireIntegerToBigInt(source.expected.byteLength)) {
      hashMismatch(path, "offline file length differs from the prepared inventory");
    }
    let bytes: Uint8Array;
    try {
      bytes = copyInspectedUnsharedUint8Array(source.rawBytes, inspection);
    } catch (cause) {
      invalid(path, "bytes changed after exact inspection", { cause });
    }
    const actualSha256 = await hashBytes(bytes, path);
    throwIfAborted(signal);
    if (actualSha256 !== source.expected.contentSha256) {
      hashMismatch(path, "offline file bytes differ from the prepared content digest");
    }
    files.push(Object.freeze({ virtualPath: source.virtualPath, bytes }));
  }

  let packBytes: Uint8Array;
  try {
    const writerOptions = {
      limits: {
        maxPackBytes: Math.min(
          profile.deployment.assetLimits.maxAssetCompressedByteLength,
          profile.deployment.assetLimits.maxAssetUnpackedByteLength,
        ),
        maxFileContentBytes: profile.deployment.assetLimits.maxAssetFileContentByteLength,
      },
      ...(signal === undefined ? {} : { signal }),
    };
    packBytes = await encodeCppCuteBrowserVfsPack(files, writerOptions);
  } catch (cause) {
    translateVfsError(cause, `$.packs[${pack.ordinal}]`);
  }
  // No later phase needs the owned source snapshots. The inspector receives
  // only the canonical pack and therefore cannot keep a second source set live.
  files.length = 0;

  let inspected: Awaited<ReturnType<typeof inspectCppCuteBrowserVfsPack>>;
  try {
    inspected = await inspectCppCuteBrowserVfsPack(
      packBytes!,
      signal === undefined ? {} : { signal },
    );
  } catch (cause) {
    translateVfsError(cause, `$.packs[${pack.ordinal}]`);
  }
  const actualEntries = unwrapInspectedCppCuteBrowserVfsPack(inspected!).entries;
  if (!sameEntries(actualEntries, pack.files) ||
      inspected!.contentSetSha256 !== pack.expectedContentSetSha256 ||
      inspected!.fileContentByteLength !== pack.fileContentByteLength) {
    hashMismatch(
      `$.packs[${pack.ordinal}]`,
      "closed VFS-pack output differs from the prepared exact inventory",
    );
  }
  return {
    output: Object.freeze({
      ordinal: pack.ordinal,
      role: pack.role,
      outputPath: pack.outputPath,
      includeRootId: pack.includeRootId,
      mountedVirtualRoot: pack.mountedVirtualRoot,
      packSha256: inspected!.packSha256,
      packByteLength: inspected!.packByteLength,
      fileContentByteLength: inspected!.fileContentByteLength,
      contentSetSha256: inspected!.contentSetSha256,
      fileCount: inspected!.fileCount,
    }),
    // encode owns these bytes and no caller has observed them; the inspector
    // validated an independent snapshot, so retaining this exact array avoids
    // a third full-pack copy.
    bytes: packBytes!,
  };
}

function conservativePackAssemblyByteCopyProjection(
  pack: CppCuteBrowserHeaderPackSelectionPackV1,
  retainedOutputBytes: bigint,
): ConservativePackAssemblyByteCopyProjection {
  const fileContentBytes = wireIntegerToBigInt(pack.fileContentByteLength);
  let pathBytes = 0n;
  let maximumFileBytes = 0n;
  for (const file of pack.files) {
    pathBytes += BigInt(TEXT_ENCODER.encode(file.virtualPath).byteLength);
    maximumFileBytes = maxBigInt(maximumFileBytes, wireIntegerToBigInt(file.byteLength));
  }
  const indexByteLength = BigInt(pack.files.length * VFS_PACK_ENTRY_FIXED_BYTES) + pathBytes;
  const packByteLength = BigInt(CPP_CUTE_BROWSER_VFS_PACK_HEADER_BYTES) +
    indexByteLength + fileContentBytes;
  const contentSetCanonicalJsonByteLength =
    exactContentSetCanonicalJsonByteLength(pack.files);

  // Live byte-copy projection, excluding caller-owned inputs and JS object
  // metadata. Hashing is sequential, so one maximum file-sized SHA input copy
  // is live at a time. Canonical content-set hashing keeps its encoded JSON and
  // SHA input copy live together. The inspector retains the writer's original
  // pack, its own snapshot, and a full-pack SHA input copy at minimum.
  const outerSourceHashPeak = retainedOutputBytes + fileContentBytes + maximumFileBytes;
  const writerSourceHashPeak = retainedOutputBytes + (2n * fileContentBytes) +
    pathBytes + maximumFileBytes;
  const writerContentSetHashPeak = retainedOutputBytes + (2n * fileContentBytes) +
    pathBytes + (2n * contentSetCanonicalJsonByteLength);
  const writerIndexHashPeak = retainedOutputBytes + (2n * fileContentBytes) +
    pathBytes + packByteLength + indexByteLength;
  const inspectorPackHashPeak = retainedOutputBytes + (3n * packByteLength);
  const inspectorIndexHashPeak = retainedOutputBytes + (2n * packByteLength) +
    indexByteLength;
  const inspectorFileHashPeak = retainedOutputBytes + (2n * packByteLength) +
    maximumFileBytes;
  const inspectorContentSetHashPeak = retainedOutputBytes + (2n * packByteLength) +
    (2n * contentSetCanonicalJsonByteLength);
  return Object.freeze({
    packByteLength,
    contentSetCanonicalJsonByteLength,
    peakBytes: maxBigInt(
      outerSourceHashPeak,
      writerSourceHashPeak,
      writerContentSetHashPeak,
      writerIndexHashPeak,
      inspectorPackHashPeak,
      inspectorIndexHashPeak,
      inspectorFileHashPeak,
      inspectorContentSetHashPeak,
    ),
  });
}

function exactContentSetCanonicalJsonByteLength(
  files: readonly CppCuteBrowserHeaderPackSelectionFileV1[],
): bigint {
  let byteLength = BigInt(CONTENT_SET_JSON_ROOT_FIXED_BYTES);
  for (const file of files) {
    byteLength += BigInt(CONTENT_SET_JSON_ENTRY_FIXED_BYTES) +
      BigInt(TEXT_ENCODER.encode(file.virtualPath).byteLength) +
      BigInt(file.byteLength.length);
  }
  if (files.length > 1) byteLength += BigInt(files.length - 1);
  return byteLength;
}

function exactContentSetCanonicalStringByteLength(
  files: readonly CppCuteBrowserHeaderPackSelectionFileV1[],
): bigint {
  let byteLength = BigInt(CONTENT_SET_ROOT_STRING_BYTES);
  for (const file of files) {
    byteLength += BigInt(CONTENT_SET_ENTRY_FIXED_STRING_BYTES) +
      BigInt(TEXT_ENCODER.encode(file.virtualPath).byteLength) +
      BigInt(file.byteLength.length);
  }
  return byteLength;
}

function maxBigInt(...values: readonly bigint[]): bigint {
  let maximum = 0n;
  for (const value of values) {
    if (value > maximum) maximum = value;
  }
  return maximum;
}

function derivePackSpecs(body: BuildLockBody, profile: BrowserProfile): readonly PackSpec[] {
  const dependencyById = new Map(
    profile.toolchain.dependencies.map((dependency) => [dependency.dependencyId, dependency]),
  );
  const nonSourceRoots = profile.virtualFileSystem.includeRoots.filter(
    (root) => root.owner.kind !== "source",
  );
  const specs: PackSpec[] = [];
  for (const [ordinal, root] of nonSourceRoots.entries()) {
    if (root.owner.kind === "compiler-resource-directory") {
      const output = exactOutput(body, "clang-resource-header-vfs", "assets/browsergrad-cpp-cute/clang-resource.headers.bgvfs");
      const llvm = body.sources.find((source) => source.sourceId === "llvm-project");
      if (llvm === undefined || profile.toolchain.compiler.version !== stripPrefix(llvm.tag, "llvmorg-")) {
        mismatch("$.input.profile.toolchain.compiler.version", "compiler version differs from the selected LLVM source");
      }
      specs.push({
        ordinal,
        role: "compiler-resource",
        assetKind: "compiler-resource-pack",
        outputPath: output.path,
        intendedAsset: "compiler-resource-pack",
        noticeComponentId: "clang",
        root,
        dependencyId: null,
        dependencyKind: null,
        version: profile.toolchain.compiler.version,
        revision: llvm.commit,
        expectedContentSetSha256: profile.toolchain.compiler.resourceDirectorySha256,
      });
      continue;
    }
    if (root.owner.kind !== "dependency") {
      mismatch("$.input.profile", "non-source include root has unsupported ownership");
    }
    const dependency = dependencyById.get(root.owner.dependencyId);
    if (dependency === undefined) mismatch("$.input.profile", "dependency include root lost its profile dependency");
    const mapping = dependencyMapping(dependency.dependencyId, dependency.kind);
    const output = exactOutput(body, mapping.outputRole, mapping.outputPath);
    if (dependency.kind === "cutlass") {
      const cutlass = body.sources.find((source) => source.sourceId === "cutlass");
      if (cutlass === undefined || dependency.version !== stripPrefix(cutlass.tag, "v") ||
          dependency.revision !== cutlass.commit) {
        mismatch("$.input.profile.toolchain.dependencies", "CuTe/CUTLASS dependency differs from the selected source");
      }
    }
    if (dependency.kind === "cxx-standard-library") {
      const llvm = body.sources.find((source) => source.sourceId === "llvm-project");
      if (llvm === undefined || dependency.version !== stripPrefix(llvm.tag, "llvmorg-") ||
          dependency.revision !== llvm.tag) {
        mismatch("$.input.profile.toolchain.dependencies", "C++ standard-library dependency differs from the selected LLVM source");
      }
    }
    if (dependency.kind === "cuda-toolkit" && dependency.version !== "12.6.3") {
      mismatch("$.input.profile.toolchain.dependencies", "CUDA header version differs from the selected output plan");
    }
    specs.push({
      ordinal,
      role: mapping.role,
      assetKind: "dependency-header-pack",
      outputPath: output.path,
      intendedAsset: `dependency-header-pack:${dependency.dependencyId}`,
      noticeComponentId: mapping.noticeComponentId,
      root,
      dependencyId: dependency.dependencyId,
      dependencyKind: dependency.kind,
      version: dependency.version,
      revision: dependency.revision,
      expectedContentSetSha256: dependency.headerSetSha256,
    });
  }
  if (specs.length !== profile.toolchain.dependencies.length + 1) {
    mismatch("$.input.profile.virtualFileSystem.includeRoots", "profile must mount exactly one pack for compiler resources and every dependency");
  }
  return Object.freeze(specs);
}

function dependencyMapping(
  dependencyId: string,
  kind: CppCuteFrontendDependencyKind,
): {
  readonly role: Exclude<CppCuteBrowserHeaderPackRole, "compiler-resource">;
  readonly outputRole: string;
  readonly outputPath: string;
  readonly noticeComponentId: string;
} {
  if (dependencyId === "cxx-stdlib" && kind === "cxx-standard-library") {
    return {
      role: "cxx-standard-library",
      outputRole: "libcxx-header-vfs",
      outputPath: "assets/browsergrad-cpp-cute/libcxx-22.1.8.headers.bgvfs",
      noticeComponentId: "libcxx",
    };
  }
  if (dependencyId === "cuda" && kind === "cuda-toolkit") {
    return {
      role: "cuda",
      outputRole: "cuda-header-vfs",
      outputPath: "assets/browsergrad-cpp-cute/cuda-12.6.3.headers.bgvfs",
      noticeComponentId: "cuda-toolkit-12.6.3-headers",
    };
  }
  if (dependencyId === "cutlass" && kind === "cutlass") {
    return {
      role: "cute",
      outputRole: "cutlass-header-vfs",
      outputPath: "assets/browsergrad-cpp-cute/cutlass-3.7.0.headers.bgvfs",
      noticeComponentId: "cutlass",
    };
  }
  if (dependencyId === "linux-sysroot" && kind === "linux-sysroot") {
    return {
      role: "linux-sysroot",
      outputRole: "linux-sysroot-header-vfs",
      outputPath: "assets/browsergrad-cpp-cute/linux-sysroot.headers.bgvfs",
      noticeComponentId: "linux-sysroot",
    };
  }
  mismatch("$.input.profile.toolchain.dependencies", "dependency has no selected header-pack output in the current build lock");
}

function exactOutput(body: BuildLockBody, role: string, path: string) {
  const matches = body.recipe.distributedOutputPlan.outputs.filter((output) => output.role === role);
  if (matches.length !== 1 || matches[0]?.path !== path ||
      matches[0].mediaType !== "application/octet-stream" ||
      matches[0].reproducibilityClass !== "deterministic-subject") {
    mismatch("$.input.buildInputLock", `build lock lost exact deterministic output ${JSON.stringify(role)}`);
  }
  return matches[0];
}

function deriveNoticeInventory(
  body: BuildLockBody,
  specs: readonly PackSpec[],
): readonly CppCuteBrowserHeaderPackNoticeV1[] {
  const notices: CppCuteBrowserHeaderPackNoticeV1[] = [];
  const seen = new Set<string>();
  const intendedAssets = new Set(specs.map((spec) => spec.intendedAsset));
  for (const approved of body.notices.approvedComponents) {
    if (!(approved.appliesTo as readonly string[]).some((asset) => intendedAssets.has(asset))) {
      continue;
    }
    if (seen.has(approved.componentId)) {
      mismatch("$.input.buildInputLock.notices", "notice component identity is ambiguous");
    }
    seen.add(approved.componentId);
    notices.push({
      componentId: approved.componentId,
      reviewStatus: "reviewed",
      licenseExpression: approved.licenseExpression,
      sourcePath: approved.sourcePath,
      noticeOutputPath: approved.noticeOutputPath,
      noticeSha256: approved.noticeSha256,
      noticeByteLength: wireU64(approved.noticeByteLength, "$.input.buildInputLock.notices"),
      appliesTo: [...approved.appliesTo],
    });
  }
  for (const unresolved of body.notices.unresolvedComponents) {
    if (!intendedAssets.has(unresolved.intendedAsset)) continue;
    if (unresolved.disposition !== "blocks-release" || seen.has(unresolved.componentId)) {
      mismatch("$.input.buildInputLock.notices", "unresolved notice binding is ambiguous or non-blocking");
    }
    seen.add(unresolved.componentId);
    notices.push({
      componentId: unresolved.componentId,
      reviewStatus: "unresolved",
      intendedAsset: unresolved.intendedAsset,
      reasonCode: unresolved.reasonCode,
      disposition: "blocks-release",
    });
  }
  for (const spec of specs) {
    if (!notices.some((notice) =>
      notice.componentId === spec.noticeComponentId && noticeAppliesTo(notice, spec.intendedAsset))) {
      mismatch(
        "$.input.buildInputLock.notices",
        "selected header pack has no exact reviewed or unresolved notice-policy binding",
      );
    }
  }
  notices.sort((left, right) => compareUtf8(left.componentId, right.componentId));
  return Object.freeze(notices.map((notice) => deepFreezeJson(notice)));
}

function parseSelectionFiles(
  value: unknown,
  inputPath: string,
  spec: PackSpec,
  noticeByComponent: ReadonlyMap<string, CppCuteBrowserHeaderPackNoticeV1>,
  budget: SelectionParseBudget,
  signal: AbortSignal | undefined,
): readonly CppCuteBrowserHeaderPackSelectionFileV1[] {
  const fileCount = denseArrayLength(value, inputPath);
  if (fileCount === 0) invalid(inputPath, "header pack must contain at least one file");
  if (fileCount > budget.remainingFiles) {
    resource("$.input.packs", `selection exceeds ${MAX_SELECTION_FILES} files globally`);
  }
  const expectedLicenseComponentIds = noticeComponentIdsForAsset(
    spec.intendedAsset,
    noticeByComponent,
  );
  if (!expectedLicenseComponentIds.includes(spec.noticeComponentId)) {
    mismatch("$.input.buildInputLock.notices", "pack lost its required notice-policy component");
  }
  const files: Array<{
    readonly value: CppCuteBrowserHeaderPackSelectionFileV1;
    readonly pathBytes: Uint8Array;
  }> = [];
  const seen = new Set<string>();
  for (let index = 0; index < fileCount; index += 1) {
    throwIfAborted(signal);
    const path = `${inputPath}[${index}]`;
    const rawFile = denseArrayEntry(value, index, inputPath);
    const fields = exactDataRecord(rawFile, path, [
      "virtualPath", "contentSha256", "byteLength", "licenseComponentIds",
    ]);
    const parsedPath = portableRelativePathWithBytes(
      fields["virtualPath"],
      `${path}.virtualPath`,
    );
    const virtualPath = parsedPath.value;
    if (seen.has(virtualPath)) invalid(`${path}.virtualPath`, "duplicate pack file path");
    seen.add(virtualPath);
    validateMountedPath(spec.root.virtualPath, parsedPath.bytes, `${path}.virtualPath`);
    const contentSha256 = sha256(fields["contentSha256"], `${path}.contentSha256`);
    const byteLength = wireU64(fields["byteLength"], `${path}.byteLength`);
    const licenseComponentIds = stringArray(
      fields["licenseComponentIds"],
      `${path}.licenseComponentIds`,
      MAX_LICENSE_COMPONENTS_PER_FILE,
    );
    if (!sameStrings(licenseComponentIds, expectedLicenseComponentIds)) {
      mismatch(
        `${path}.licenseComponentIds`,
        "file license mapping must equal every build-lock notice-policy component for this asset",
      );
    }
    const logicalBytes = 160 + parsedPath.bytes.byteLength + licenseComponentIds.reduce(
      (total, componentId) => total + TEXT_ENCODER.encode(componentId).byteLength,
      0,
    );
    if (logicalBytes > budget.remainingLogicalBytes) {
      resource(
        "$.input.packs",
        `selection file inventory exceeds ${MAX_SELECTION_FILE_INVENTORY_LOGICAL_BYTES} logical bytes globally`,
      );
    }
    budget.remainingFiles -= 1;
    budget.remainingLogicalBytes -= logicalBytes;
    files.push({
      value: { virtualPath, contentSha256, byteLength, licenseComponentIds },
      pathBytes: parsedPath.bytes,
    });
  }
  files.sort((left, right) => compareBytes(left.pathBytes, right.pathBytes));
  for (let index = 1; index < files.length; index += 1) {
    const previous = files[index - 1]!.value.virtualPath;
    const current = files[index]!.value.virtualPath;
    if (current.startsWith(`${previous}/`)) {
      invalid(inputPath, "a regular file cannot be an implicit parent directory");
    }
  }
  return Object.freeze(files.map((file) => deepFreezeJson(file.value)));
}

function noticeAppliesTo(
  notice: CppCuteBrowserHeaderPackNoticeV1,
  intendedAsset: string,
): boolean {
  return notice.reviewStatus === "reviewed"
    ? notice.appliesTo.includes(intendedAsset)
    : notice.intendedAsset === intendedAsset;
}

function noticeComponentIdsForAsset(
  intendedAsset: string,
  noticeByComponent: ReadonlyMap<string, CppCuteBrowserHeaderPackNoticeV1>,
): readonly string[] {
  return Object.freeze([...noticeByComponent.values()]
    .filter((notice) => noticeAppliesTo(notice, intendedAsset))
    .map((notice) => notice.componentId)
    .sort(compareUtf8));
}

function validateMountedPath(root: string, relativeBytes: Uint8Array, path: string): void {
  const rootBytes = TEXT_ENCODER.encode(root);
  const mountedByteLength = rootBytes.byteLength + relativeBytes.byteLength +
    (root === "/" ? 0 : 1);
  if (mountedByteLength >
      CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.vfs.maxPathByteLength) {
    resource(path, "final mounted UTF-8 path exceeds the runtime-ABI path ceiling");
  }
}

function portableRelativePathWithBytes(
  value: unknown,
  path: string,
): { readonly value: string; readonly bytes: Uint8Array } {
  if (typeof value !== "string" || value.length === 0) invalid(path, "expected nonempty relative path");
  const bytes = TEXT_ENCODER.encode(value);
  if (bytes.byteLength > CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.vfs.maxPathByteLength) {
    resource(path, "relative UTF-8 path exceeds the runtime-ABI path ceiling");
  }
  if (value !== value.normalize("NFC") || value.startsWith("/") || value.endsWith("/") || value.includes("\\")) {
    invalid(path, "path must be NFC-normalized relative POSIX syntax");
  }
  const segments = value.split("/");
  if (segments.some((segment) => segment === "" || segment === "." || segment === ".." ||
      !/^[A-Za-z0-9._+@=-]+$/u.test(segment))) {
    invalid(path, "path contains a non-portable segment");
  }
  return Object.freeze({ value, bytes });
}

function sameEntries(
  actual: readonly CppCuteBrowserVfsPackEntry[],
  expected: readonly CppCuteBrowserHeaderPackSelectionFileV1[],
): boolean {
  return actual.length === expected.length && actual.every((entry, index) => {
    const wanted = expected[index];
    return wanted !== undefined && entry.virtualPath === wanted.virtualPath &&
      entry.contentSha256 === wanted.contentSha256 && entry.byteLength === wanted.byteLength;
  });
}

interface SelectionCanonicalMeasureState {
  nodes: number;
  stringBytes: number;
}

function preflightSelectionCanonicalJson(value: JsonValue, path: string): void {
  const state: SelectionCanonicalMeasureState = { nodes: 0, stringBytes: 0 };
  const documentBytes = measureSelectionCanonicalJson(value, 1, state, path);
  if (documentBytes > SELECTION_CANONICAL_LIMITS.maxDocumentBytes) {
    resource(
      path,
      `canonical selection requires ${documentBytes} bytes; limit is ${SELECTION_CANONICAL_LIMITS.maxDocumentBytes}`,
    );
  }
}

function measureSelectionCanonicalJson(
  value: JsonValue,
  depth: number,
  state: SelectionCanonicalMeasureState,
  path: string,
): number {
  state.nodes += 1;
  if (state.nodes > SELECTION_CANONICAL_LIMITS.maxNodes) {
    resource(path, `canonical selection exceeds ${SELECTION_CANONICAL_LIMITS.maxNodes} nodes`);
  }
  if (depth > SELECTION_CANONICAL_LIMITS.maxDepth) {
    resource(path, `canonical selection exceeds depth ${SELECTION_CANONICAL_LIMITS.maxDepth}`);
  }
  if (value === null) return 4;
  if (value === true) return 4;
  if (value === false) return 5;
  if (typeof value === "number") return String(value).length;
  if (typeof value === "string") {
    consumeSelectionCanonicalStringBytes(value, state, path);
    return encodedJsonStringByteLength(value);
  }
  if (Array.isArray(value)) {
    if (value.length > SELECTION_CANONICAL_LIMITS.maxArrayLength) {
      resource(path, `canonical selection array exceeds ${SELECTION_CANONICAL_LIMITS.maxArrayLength} entries`);
    }
    let bytes = 2 + Math.max(0, value.length - 1);
    for (const child of value) {
      bytes += measureSelectionCanonicalJson(child, depth + 1, state, path);
      if (bytes > SELECTION_CANONICAL_LIMITS.maxDocumentBytes) {
        resource(path, "canonical selection exceeds its document-byte limit");
      }
    }
    return bytes;
  }
  const objectValue = value as JsonObject;
  const keys = Object.keys(objectValue);
  if (keys.length > SELECTION_CANONICAL_LIMITS.maxObjectProperties) {
    resource(
      path,
      `canonical selection object exceeds ${SELECTION_CANONICAL_LIMITS.maxObjectProperties} properties`,
    );
  }
  let bytes = 2 + Math.max(0, keys.length - 1);
  for (const key of keys) {
    consumeSelectionCanonicalStringBytes(key, state, path);
    bytes += encodedJsonStringByteLength(key) + 1;
    bytes += measureSelectionCanonicalJson(objectValue[key]!, depth + 1, state, path);
    if (bytes > SELECTION_CANONICAL_LIMITS.maxDocumentBytes) {
      resource(path, "canonical selection exceeds its document-byte limit");
    }
  }
  return bytes;
}

function consumeSelectionCanonicalStringBytes(
  value: string,
  state: SelectionCanonicalMeasureState,
  path: string,
): void {
  state.stringBytes += TEXT_ENCODER.encode(value).byteLength;
  if (state.stringBytes > SELECTION_CANONICAL_LIMITS.maxStringBytes) {
    resource(
      path,
      `canonical selection exceeds ${SELECTION_CANONICAL_LIMITS.maxStringBytes} cumulative string bytes`,
    );
  }
}

function encodedJsonStringByteLength(value: string): number {
  return TEXT_ENCODER.encode(JSON.stringify(value)).byteLength;
}

async function hashSelectionCanonicalJson(value: JsonValue, path: string): Promise<string> {
  try {
    return await hashCanonicalJson(value, { limits: SELECTION_CANONICAL_LIMITS });
  } catch (cause) {
    translateSelectionCanonicalError(cause, path, true);
  }
}

function canonicalSelectionJsonBytes(value: JsonValue, path: string): Uint8Array {
  try {
    return canonicalJsonBytes(value, { limits: SELECTION_CANONICAL_LIMITS });
  } catch (cause) {
    translateSelectionCanonicalError(cause, path, false);
  }
}

function translateSelectionCanonicalError(
  cause: unknown,
  path: string,
  hashing: boolean,
): never {
  if (cause instanceof SemanticSchemaError) {
    if (cause.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit) {
      resource(path, "canonical selection projection exceeds its explicit limits", { cause });
    }
    if (cause.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.hashUnavailable) {
      fail(
        "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-HASH-UNAVAILABLE",
        path,
        "SHA-256 is unavailable for the canonical selection projection",
        { cause },
      );
    }
    invalid(path, "canonical selection projection is invalid", { cause });
  }
  if (hashing) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-HASH-UNAVAILABLE",
      path,
      "SHA-256 failed for the canonical selection projection",
      { cause },
    );
  }
  invalid(path, "canonical selection encoding failed", { cause });
}

async function deriveSelectionContentSetSha256(
  files: readonly CppCuteBrowserHeaderPackSelectionFileV1[],
  path: string,
): Promise<string> {
  try {
    return await deriveCppCuteBrowserVfsContentSetSha256(files);
  } catch (cause) {
    if (cause instanceof CppCuteBrowserVfsPackError) {
      if (cause.code === "BG-COMPILER-CPP-CUTE-BROWSER-VFS-RESOURCE-LIMIT") {
        resource(path, "content-set hash projection exceeds the closed selection budget", { cause });
      }
      if (cause.code === "BG-COMPILER-CPP-CUTE-BROWSER-VFS-HASH-UNAVAILABLE") {
        fail(
          "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-HASH-UNAVAILABLE",
          path,
          "SHA-256 is unavailable for the content-set projection",
          { cause },
        );
      }
      invalid(path, "content-set hash projection rejected the selected inventory", { cause });
    }
    invalid(path, "content-set hash projection failed", { cause });
  }
}

function selectionReleaseBlockers(
  buildInputLock: PreparedCppCuteBrowserBuildInputLock,
): readonly string[] {
  const blockers = new Set([
    ...buildInputLock.releaseBlockerIds,
    HEADER_PACK_INTEGRATION_BLOCKER,
    HEADER_PACK_NOTICE_BYTES_BLOCKER,
    HEADER_PACK_LICENSE_MAP_BLOCKER,
  ]);
  return Object.freeze([...blockers].sort(compareUtf8));
}

function translateVfsError(cause: unknown, path: string): never {
  if (cause instanceof CppCuteBrowserVfsPackError) {
    if (cause.code === "BG-COMPILER-CPP-CUTE-BROWSER-VFS-CANCELLED") {
      cancelled("$.signal", { cause });
    }
    if (cause.code === "BG-COMPILER-CPP-CUTE-BROWSER-VFS-RESOURCE-LIMIT") {
      resource(path, "closed VFS-pack writer exceeded a bounded assembly limit", { cause });
    }
    if (cause.code === "BG-COMPILER-CPP-CUTE-BROWSER-VFS-HASH-UNAVAILABLE") {
      fail(
        "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-HASH-UNAVAILABLE",
        path,
        "SHA-256 is unavailable during closed VFS-pack assembly",
        { cause },
      );
    }
    if (cause.code === "BG-COMPILER-CPP-CUTE-BROWSER-VFS-HASH-MISMATCH") {
      hashMismatch(path, "closed VFS-pack verification found a digest mismatch");
    }
    invalid(path, "closed VFS-pack writer rejected the selected file bytes", { cause });
  }
  invalid(path, "closed VFS-pack writer failed", { cause });
}

function storedSelection(
  selection: PreparedCppCuteBrowserHeaderPackSelection,
): StoredHeaderPackSelection {
  if (typeof selection !== "object" || selection === null) unverified("$.selection");
  const stored = PREPARED_SELECTIONS.get(selection as object);
  if (stored === undefined) unverified("$.selection");
  return stored;
}

function storedAssembly(assembled: AssembledCppCuteBrowserHeaderPacks): StoredAssembledHeaderPacks {
  if (typeof assembled !== "object" || assembled === null) unverified("$.assembled");
  const stored = ASSEMBLED_PACKS.get(assembled as object);
  if (stored === undefined) unverified("$.assembled");
  return stored;
}

function operationSignal(options: CppCuteBrowserHeaderPackOperationOptions): AbortSignal | undefined {
  const fields = exactDataRecord(options, "$.options", ["signal"], true);
  const signal = fields["signal"];
  if (signal !== undefined && !isAbortSignal(signal)) invalid("$.options.signal", "expected AbortSignal");
  return signal as AbortSignal | undefined;
}

function isAbortSignal(value: unknown): value is AbortSignal {
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined || typeof value !== "object" || value === null) return false;
  try {
    ABORT_SIGNAL_ABORTED_GETTER.call(value);
    return true;
  } catch {
    return false;
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal === undefined || ABORT_SIGNAL_ABORTED_GETTER === undefined) return;
  let aborted: unknown;
  try {
    aborted = ABORT_SIGNAL_ABORTED_GETTER.call(signal);
  } catch (cause) {
    invalid("$.signal", "abort signal became unreadable", { cause });
  }
  if (aborted === true) cancelled("$.signal");
}

function denseArray(value: unknown, path: string, maximumLength: number): readonly unknown[] {
  const length = denseArrayLength(value, path, maximumLength);
  const result: unknown[] = [];
  for (let index = 0; index < length; index += 1) {
    result.push(denseArrayEntry(value, index, path));
  }
  return result;
}

function denseArrayLength(value: unknown, path: string, maximumLength = MAX_SELECTION_FILES): number {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype) {
    invalid(path, "expected dense plain array");
  }
  const descriptor = Object.getOwnPropertyDescriptor(value, "length");
  if (descriptor === undefined || !("value" in descriptor) ||
      !Number.isSafeInteger(descriptor.value) || descriptor.value < 0) {
    invalid(path, "array must have one ordinary length data property");
  }
  const length = descriptor.value as number;
  if (length > maximumLength) resource(path, `array exceeds maximum length ${maximumLength}`);
  let ownKeys: readonly PropertyKey[];
  try {
    ownKeys = Reflect.ownKeys(value);
  } catch (cause) {
    invalid(path, "array own keys could not be inspected", { cause });
  }
  if (ownKeys.length !== length + 1) {
    invalid(path, "array must contain only its length and dense indexed data properties");
  }
  for (const key of ownKeys) {
    if (key === "length") continue;
    if (typeof key !== "string") {
      invalid(path, "array must not contain symbol properties");
    }
    const index = Number(key);
    if (!Number.isInteger(index) || index < 0 || index >= length || String(index) !== key) {
      invalid(path, "array may contain only canonical index properties");
    }
  }
  return length;
}

function denseArrayEntry(value: unknown, index: number, path: string): unknown {
  const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
  if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
    invalid(`${path}[${index}]`, "array entry must be one enumerable data property");
  }
  return descriptor.value;
}

function exactDataRecord(
  value: unknown,
  path: string,
  keys: readonly string[],
  allowMissing = false,
): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null ||
      (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null)) {
    invalid(path, "expected plain data object");
  }
  const result: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
  for (const key of keys) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (descriptor === undefined) {
      if (allowMissing) continue;
      invalid(`${path}.${key}`, "missing field");
    }
    if (descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "field must be one enumerable data property");
    }
    result[key] = descriptor.value;
  }
  let ownKeys: readonly PropertyKey[];
  try {
    ownKeys = Reflect.ownKeys(value);
  } catch (cause) {
    invalid(path, "record own keys could not be inspected", { cause });
  }
  const expectedOwnKeyCount = allowMissing
    ? Object.keys(result).length
    : keys.length;
  if (ownKeys.length !== expectedOwnKeyCount ||
      ownKeys.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(path, `expected exactly data fields ${keys.join(", ")}`);
  }
  return Object.freeze(result);
}

function stringArray(value: unknown, path: string, maximumLength: number): readonly string[] {
  const values = denseArray(value, path, maximumLength).map((entry, index) =>
    boundedToken(entry, `${path}[${index}]`, 128));
  for (let index = 1; index < values.length; index += 1) {
    if (compareUtf8(values[index - 1]!, values[index]!) >= 0) {
      invalid(path, "values must be strictly sorted and unique by UTF-8 bytes");
    }
  }
  return Object.freeze(values);
}

function boundedToken(value: unknown, path: string, maximumLength: number): string {
  if (typeof value !== "string" || value.length === 0 || value.length > maximumLength ||
      !/^[A-Za-z0-9._:@+-]+$/u.test(value)) {
    invalid(path, "expected bounded portable token");
  }
  return value;
}

function sha256(value: unknown, path: string): string {
  if (typeof value !== "string" || !SHA256_HEX.test(value)) invalid(path, "expected lowercase SHA-256 hex");
  return value;
}

function wireU64(value: unknown, path: string): WireU64 {
  if (typeof value !== "string" || !WIRE_U64.test(value)) invalid(path, "expected canonical WireU64 decimal string");
  try {
    return encodeWireU64(BigInt(value));
  } catch (cause) {
    invalid(path, "WireU64 exceeds unsigned 64-bit range", { cause });
  }
}

function fileKey(includeRootId: string, virtualPath: string): string {
  return `${includeRootId.length}:${includeRootId}${virtualPath}`;
}

function stripPrefix(value: string, prefix: string): string {
  return value.startsWith(prefix) ? value.slice(prefix.length) : value;
}

function compareUtf8(left: string, right: string): number {
  return compareBytes(TEXT_ENCODER.encode(left), TEXT_ENCODER.encode(right));
}

function compareBytes(leftBytes: Uint8Array, rightBytes: Uint8Array): number {
  const length = Math.min(leftBytes.byteLength, rightBytes.byteLength);
  for (let index = 0; index < length; index += 1) {
    const difference = leftBytes[index]! - rightBytes[index]!;
    if (difference !== 0) return difference;
  }
  return leftBytes.byteLength - rightBytes.byteLength;
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

async function hashBytes(bytes: Uint8Array, path: string): Promise<string> {
  try {
    return await sha256Hex(bytes);
  } catch (cause) {
    fail("BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-HASH-UNAVAILABLE", path, "SHA-256 is unavailable", { cause });
  }
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-INVALID", path, message, options);
}

function mismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-MISMATCH", path, message);
}

function resource(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-RESOURCE-LIMIT", path, message, options);
}

function hashMismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-HASH-MISMATCH", path, message);
}

function cancelled(path: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-CANCELLED", path, "operation was cancelled", options);
}

function unverified(path: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-UNVERIFIED", path, "value is not an opaque header-pack authority");
}

function fail(
  code: CppCuteBrowserHeaderPackAssemblyErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserHeaderPackAssemblyError(code, path, message, options);
}
