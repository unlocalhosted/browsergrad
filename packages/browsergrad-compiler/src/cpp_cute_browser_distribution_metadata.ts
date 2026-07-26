import {
  canonicalJsonBytes,
  hashCanonicalJson,
  sha256Hex,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR,
  CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR,
  CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA,
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
  canonicalCppCuteBrowserAssetManifestBytes,
  cppCuteBrowserSourceAbi,
  deriveCppCuteBrowserAssetManifestId,
  deriveCppCuteBrowserAssetSetSha256,
  prepareCppCuteBrowserAssetManifest,
  type CppCuteBrowserAssetManifestBodyV1,
  type CppCuteBrowserAssetManifestV1,
  type CppCuteBrowserAssetV1,
  type PreparedCppCuteBrowserAssetManifest,
} from "./cpp_cute_browser_assets.js";
import {
  unwrapPreparedCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
} from "./cpp_cute_browser_build_lock.js";
import {
  deriveCppCuteBrowserBuildSubjectIdentity,
  type CppCuteBrowserBuildSubjectIdentity,
} from "./cpp_cute_browser_build_provenance_syntax.js";
import {
  createCppCuteBrowserCompileProfileInput,
  deriveCppCuteBrowserEmptySourceIncludeRootManifestSha256,
} from "./cpp_cute_browser_compile_profile.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
} from "./cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
  cppCuteDiagnosticNormalizationResourceBytes,
} from "./cpp_cute_diagnostic_normalization.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import {
  unwrapAdmittedCppCuteBrowserProducerTrustPolicy,
  type AdmittedCppCuteBrowserProducerTrustPolicy,
} from "./cpp_cute_browser_producer_trust_policy.js";
import {
  requireVerifiedCppCuteBrowserReproducibility,
  type VerifiedCppCuteBrowserReproducibility,
} from "./cpp_cute_browser_reproducibility.js";
import {
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
  cppCuteSemanticAdapterManifestResourceBytes,
} from "./cpp_cute_semantic_adapter_manifest.js";
import {
  unwrapInspectedCppCuteBrowserVfsPack,
  type InspectedCppCuteBrowserVfsPack,
} from "./cpp_cute_browser_vfs_pack.js";
import {
  inspectVerifiedCppCuteBrowserWorkerBundle,
  type VerifiedCppCuteBrowserWorkerBundle,
} from "./cpp_cute_browser_worker_bundle.js";

export const CPP_CUTE_BROWSER_DISTRIBUTION_METADATA_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-distribution-metadata";

const METADATA_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-distribution-metadata.v1";
const PROVISIONAL_BUILD_SUBJECT_ID =
  `bg.cpp.browser-build-subject.sha256.${"0".repeat(64)}`;
const PACK_BINDINGS = Object.freeze([
  Object.freeze({
    includeRootId: "clang-resource",
    outputPath:
      "assets/browsergrad-cpp-cute/clang-resource.headers.bgvfs",
  }),
  Object.freeze({
    includeRootId: "cuda",
    outputPath: "assets/browsergrad-cpp-cute/cuda-12.6.3.headers.bgvfs",
  }),
  Object.freeze({
    includeRootId: "cutlass",
    outputPath: "assets/browsergrad-cpp-cute/cutlass-3.7.0.headers.bgvfs",
  }),
  Object.freeze({
    includeRootId: "cxx-stdlib",
    outputPath:
      "assets/browsergrad-cpp-cute/libcxx-22.1.8.headers.bgvfs",
  }),
  Object.freeze({
    includeRootId: "linux-sysroot",
    outputPath:
      "assets/browsergrad-cpp-cute/linux-sysroot.headers.bgvfs",
  }),
] as const);
const PREPARED_METADATA = new WeakMap<
  object,
  PreparedCppCuteBrowserDistributionMetadataRecord
>();

export type CppCuteBrowserDistributionPackIncludeRootId =
  typeof PACK_BINDINGS[number]["includeRootId"];

export interface CppCuteBrowserDistributionPackInput {
  readonly includeRootId: CppCuteBrowserDistributionPackIncludeRootId;
  readonly outputPath: string;
  readonly pack: InspectedCppCuteBrowserVfsPack;
}

export interface PrepareCppCuteBrowserDistributionMetadataInput {
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  readonly packs: readonly CppCuteBrowserDistributionPackInput[];
  readonly producerTrustPolicy: AdmittedCppCuteBrowserProducerTrustPolicy;
  readonly wasmReproducibility: VerifiedCppCuteBrowserReproducibility;
  readonly workerBundle: VerifiedCppCuteBrowserWorkerBundle;
}

declare const preparedCppCuteBrowserDistributionMetadataBrand: unique symbol;

/**
 * Exact profile, manifest, and build-subject metadata for one current browser
 * distribution. This is deterministic content preparation only. It does not
 * prove that any output file exists or grant producer/distribution authority.
 */
export interface PreparedCppCuteBrowserDistributionMetadata {
  readonly [preparedCppCuteBrowserDistributionMetadataBrand]: true;
  readonly schema: typeof CPP_CUTE_BROWSER_DISTRIBUTION_METADATA_SCHEMA;
  readonly version: 1;
  readonly metadataId: string;
  readonly authority: "deterministic-browser-distribution-metadata-only";
  readonly profileId: string;
  readonly profileHash: string;
  readonly compilationContractHash: string;
  readonly profileSha256: string;
  readonly profileByteLength: number;
  readonly assetManifestId: string;
  readonly assetManifestSha256: string;
  readonly assetManifestByteLength: WireU64;
  readonly assetSetSha256: string;
  readonly buildSubjectId: string;
  readonly buildSubjectSha256: string;
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly workerBundleSha256: string;
  readonly wasmSha256: string;
  readonly wasmByteLength: number;
  readonly producerPolicyId: string;
  readonly producerPolicySha256: string;
  readonly packCount: 5;
  readonly packByteLength: string;
  readonly profilePrepared: true;
  readonly assetManifestPrepared: true;
  readonly cycleFreeBuildSubjectVerified: true;
  readonly exactOutputFilesVerified: false;
  readonly producerTrusted: false;
  readonly distributionAuthorized: false;
  readonly releaseReady: false;
}

export interface PreparedCppCuteBrowserDistributionMetadataRecord {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
  readonly buildSubject: CppCuteBrowserBuildSubjectIdentity;
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  readonly workerBundle: VerifiedCppCuteBrowserWorkerBundle;
  readonly producerTrustPolicy: AdmittedCppCuteBrowserProducerTrustPolicy;
  readonly wasmReproducibility: VerifiedCppCuteBrowserReproducibility;
  readonly packs: ReadonlyMap<
    CppCuteBrowserDistributionPackIncludeRootId,
    CppCuteBrowserDistributionPackInput
  >;
}

export type CppCuteBrowserDistributionMetadataErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-METADATA-BINDING"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-METADATA-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-METADATA-UNVERIFIED";

export class CppCuteBrowserDistributionMetadataError extends Error {
  constructor(
    readonly code: CppCuteBrowserDistributionMetadataErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserDistributionMetadataError";
  }
}

export async function prepareCppCuteBrowserDistributionMetadata(
  input: PrepareCppCuteBrowserDistributionMetadataInput,
): Promise<PreparedCppCuteBrowserDistributionMetadata> {
  const normalized = normalizeInput(input);
  const lock = unwrapPreparedCppCuteBrowserBuildInputLock(
    normalized.buildInputLock,
  );
  const worker = inspectVerifiedCppCuteBrowserWorkerBundle(
    normalized.workerBundle,
  );
  const policy = unwrapAdmittedCppCuteBrowserProducerTrustPolicy(
    normalized.producerTrustPolicy,
  ).policy;
  try {
    requireVerifiedCppCuteBrowserReproducibility(
      normalized.wasmReproducibility,
    );
  } catch (cause) {
    unverified(
      "$.input.wasmReproducibility",
      "expected package-verified extractor reproducibility",
      { cause },
    );
  }
  if (normalized.wasmReproducibility.sourceSetSha256 !==
        normalized.buildInputLock.extractorSourceSetSha256) {
    binding(
      "$.input.wasmReproducibility.sourceSetSha256",
      "extractor evidence differs from the current build-input source set",
    );
  }
  if (policy.predicateType !==
        CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE) {
    binding(
      "$.input.producerTrustPolicy.predicateType",
      "producer policy does not admit the browser build predicate",
    );
  }

  const sourceRootManifestSha256 =
    await deriveCppCuteBrowserEmptySourceIncludeRootManifestSha256();
  const baseProfileInput = {
    buildProvenanceLockSha256: normalized.buildInputLock.resourceSha256,
    extractorWasmSha256: normalized.wasmReproducibility.wasmSha256,
    runtimeAbiManifestSha256:
      CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
    semanticAdapterManifestSha256:
      CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
    sourceRootManifestSha256,
    workerModuleSha256: worker.sha256,
    workerModuleByteLength: worker.byteLength,
    headerContentSets: {
      clangResource: requiredPack(
        normalized.packs,
        "clang-resource",
      ).pack.contentSetSha256,
      cuda: requiredPack(normalized.packs, "cuda").pack.contentSetSha256,
      cutlass:
        requiredPack(normalized.packs, "cutlass").pack.contentSetSha256,
      cxxStdlib:
        requiredPack(normalized.packs, "cxx-stdlib").pack.contentSetSha256,
      linuxSysroot:
        requiredPack(
          normalized.packs,
          "linux-sysroot",
        ).pack.contentSetSha256,
    },
  } as const;

  const provisional = await prepareMetadata(
    baseProfileInput,
    PROVISIONAL_BUILD_SUBJECT_ID,
    normalized.packs,
    policy,
    normalized.wasmReproducibility.wasmByteLength,
  );
  const provisionalSubject = await deriveCppCuteBrowserBuildSubjectIdentity({
    assetManifest: provisional.assetManifest,
    buildInputLock: normalized.buildInputLock,
    workerBundle: normalized.workerBundle,
  });
  const final = await prepareMetadata(
    baseProfileInput,
    provisionalSubject.buildSubjectId,
    normalized.packs,
    policy,
    normalized.wasmReproducibility.wasmByteLength,
  );
  const buildSubject = await deriveCppCuteBrowserBuildSubjectIdentity({
    assetManifest: final.assetManifest,
    buildInputLock: normalized.buildInputLock,
    workerBundle: normalized.workerBundle,
  });
  if (buildSubject.buildSubjectId !== provisionalSubject.buildSubjectId ||
      buildSubject.buildSubjectSha256 !==
        provisionalSubject.buildSubjectSha256) {
    binding(
      "$.buildSubject",
      "cycle-free build-subject identity changed after final manifest binding",
    );
  }

  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(
    final.profile,
  );
  const profileBytes = canonicalJsonBytes(profileRecord.profile);
  const profileSha256 = await sha256Hex(profileBytes);
  const packByteLength = [...normalized.packs.values()].reduce(
    (total, pack) => total + BigInt(pack.pack.packByteLength),
    0n,
  ).toString();
  const metadataHash = await hashCanonicalJson({
    domain: METADATA_HASH_DOMAIN,
    profileHash: profileRecord.profileHash,
    profileSha256,
    assetManifestId: final.assetManifest.manifestId,
    assetManifestSha256: final.assetManifest.manifestSha256,
    buildSubjectId: buildSubject.buildSubjectId,
    buildSubjectSha256: buildSubject.buildSubjectSha256,
    buildInputLockId: normalized.buildInputLock.lockId,
    buildInputLockResourceSha256:
      normalized.buildInputLock.resourceSha256,
    workerBundleSha256: worker.sha256,
    wasmSha256: normalized.wasmReproducibility.wasmSha256,
    producerPolicyId: normalized.producerTrustPolicy.policyId,
    producerPolicySha256: normalized.producerTrustPolicy.policySha256,
    packs: PACK_BINDINGS.map((binding_) => {
      const pack = requiredPack(normalized.packs, binding_.includeRootId);
      return {
        includeRootId: pack.includeRootId,
        outputPath: pack.outputPath,
        packSha256: pack.pack.packSha256,
        packByteLength: pack.pack.packByteLength,
        contentSetSha256: pack.pack.contentSetSha256,
      };
    }),
  });
  const prepared = Object.freeze({
    schema: CPP_CUTE_BROWSER_DISTRIBUTION_METADATA_SCHEMA,
    version: 1,
    metadataId:
      `bg.cpp.browser-distribution-metadata.sha256.${metadataHash}`,
    authority: "deterministic-browser-distribution-metadata-only",
    profileId: final.profile.profileId,
    profileHash: profileRecord.profileHash,
    compilationContractHash: profileRecord.compilationContractHash,
    profileSha256,
    profileByteLength: profileBytes.byteLength,
    assetManifestId: final.assetManifest.manifestId,
    assetManifestSha256: final.assetManifest.manifestSha256,
    assetManifestByteLength: final.assetManifest.manifestByteLength,
    assetSetSha256: final.assetManifest.assetSetSha256,
    buildSubjectId: buildSubject.buildSubjectId,
    buildSubjectSha256: buildSubject.buildSubjectSha256,
    buildInputLockId: normalized.buildInputLock.lockId,
    buildInputLockResourceSha256:
      normalized.buildInputLock.resourceSha256,
    workerBundleSha256: worker.sha256,
    wasmSha256: normalized.wasmReproducibility.wasmSha256,
    wasmByteLength: normalized.wasmReproducibility.wasmByteLength,
    producerPolicyId: normalized.producerTrustPolicy.policyId,
    producerPolicySha256: normalized.producerTrustPolicy.policySha256,
    packCount: 5,
    packByteLength,
    profilePrepared: true,
    assetManifestPrepared: true,
    cycleFreeBuildSubjectVerified: true,
    exactOutputFilesVerified: false,
    producerTrusted: false,
    distributionAuthorized: false,
    releaseReady: false,
  }) as PreparedCppCuteBrowserDistributionMetadata;
  PREPARED_METADATA.set(prepared, Object.freeze({
    profile: final.profile,
    assetManifest: final.assetManifest,
    buildSubject,
    buildInputLock: normalized.buildInputLock,
    workerBundle: normalized.workerBundle,
    producerTrustPolicy: normalized.producerTrustPolicy,
    wasmReproducibility: normalized.wasmReproducibility,
    packs: normalized.packs,
  }));
  void lock;
  return prepared;
}

export function unwrapPreparedCppCuteBrowserDistributionMetadata(
  prepared: PreparedCppCuteBrowserDistributionMetadata,
): PreparedCppCuteBrowserDistributionMetadataRecord {
  const record = PREPARED_METADATA.get(prepared as object);
  if (record === undefined) {
    unverified(
      "$.metadata",
      "expected verifier-issued browser distribution metadata",
    );
  }
  return record;
}

export function copyPreparedCppCuteBrowserDistributionProfileBytes(
  prepared: PreparedCppCuteBrowserDistributionMetadata,
): Uint8Array {
  const record = unwrapPreparedCppCuteBrowserDistributionMetadata(prepared);
  return canonicalJsonBytes(
    unwrapPreparedCppCuteBrowserFrontendProfile(record.profile).profile,
  );
}

export function copyPreparedCppCuteBrowserDistributionAssetManifestBytes(
  prepared: PreparedCppCuteBrowserDistributionMetadata,
): Uint8Array {
  return canonicalCppCuteBrowserAssetManifestBytes(
    unwrapPreparedCppCuteBrowserDistributionMetadata(prepared).assetManifest,
  );
}

interface PreparedMetadata {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
}

async function prepareMetadata(
  baseProfileInput: Omit<
    Parameters<typeof createCppCuteBrowserCompileProfileInput>[0],
    "assetSetSha256"
  >,
  buildSubjectId: string,
  packs: ReadonlyMap<
    CppCuteBrowserDistributionPackIncludeRootId,
    CppCuteBrowserDistributionPackInput
  >,
  policy: ReturnType<
    typeof unwrapAdmittedCppCuteBrowserProducerTrustPolicy
  >["policy"],
  wasmByteLength: number,
): Promise<PreparedMetadata> {
  const provisionalProfile = await prepareCppCuteFrontendProfile(
    createCppCuteBrowserCompileProfileInput({
      ...baseProfileInput,
      assetSetSha256: "0".repeat(64),
    }),
  );
  const sourceAbi = cppCuteBrowserSourceAbi(provisionalProfile);
  const sourceAbiSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-source-abi.v1",
    sourceAbi,
  });
  const assets = distributionAssets(
    provisionalProfile,
    buildSubjectId,
    sourceAbiSha256,
    packs,
    wasmByteLength,
  );
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(
    provisionalProfile,
  ).profile;
  const dependencyIds = profileRecord.toolchain.dependencies.map(
    (dependency) => dependency.dependencyId,
  );
  const mountedVirtualRoots = assets.flatMap((asset): string[] =>
    asset.kind === "compiler-resource-pack" ||
      asset.kind === "dependency-header-pack"
      ? [asset.mountedVirtualRoot]
      : []).sort(compareText);
  const buildProvenancePolicy = {
    predicateType: CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
    trustStoreSha256: policy.trustStoreSha256,
    builderIds: [...policy.builderIds],
  } as const;
  const assetSetSha256 = await deriveCppCuteBrowserAssetSetSha256({
    sourceAbiSha256,
    dependencyIds,
    buildSubjectIds: [buildSubjectId],
    buildProvenancePolicy,
    mountedVirtualRoots,
    assets,
  });
  const profile = await prepareCppCuteFrontendProfile(
    createCppCuteBrowserCompileProfileInput({
      ...baseProfileInput,
      assetSetSha256,
    }),
  );
  const finalSourceAbi = cppCuteBrowserSourceAbi(profile);
  if (!sameBytes(
    canonicalJsonBytes(sourceAbi),
    canonicalJsonBytes(finalSourceAbi),
  )) {
    binding(
      "$.profile.sourceAbi",
      "asset-set binding changed the source ABI",
    );
  }
  const finalRecord = unwrapPreparedCppCuteBrowserFrontendProfile(profile);
  const totals = assetTotals(assets);
  const body: CppCuteBrowserAssetManifestBodyV1 = {
    profileHash: finalRecord.profileHash,
    sourceAbi: finalSourceAbi,
    sourceAbiSha256,
    assetSetSha256,
    dependencyIds,
    buildSubjectIds: [buildSubjectId],
    buildProvenancePolicy,
    mountedVirtualRoots,
    assets,
    totals,
  };
  const manifest: CppCuteBrowserAssetManifestV1 = {
    schema: CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR,
      minor: CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR,
    },
    manifestId: await deriveCppCuteBrowserAssetManifestId(body),
    body,
  };
  const assetManifest = await prepareCppCuteBrowserAssetManifest(
    manifest,
    profile,
  );
  return Object.freeze({ profile, assetManifest });
}

function distributionAssets(
  profile: PreparedCppCuteFrontendProfile,
  buildSubjectId: string,
  sourceAbiSha256: string,
  packs: ReadonlyMap<
    CppCuteBrowserDistributionPackIncludeRootId,
    CppCuteBrowserDistributionPackInput
  >,
  wasmByteLength: number,
): readonly CppCuteBrowserAssetV1[] {
  const profileRecord =
    unwrapPreparedCppCuteBrowserFrontendProfile(profile).profile;
  const adapterBytes = cppCuteSemanticAdapterManifestResourceBytes();
  const diagnosticBytes = cppCuteDiagnosticNormalizationResourceBytes();
  const runtimeBytes = cppCuteBrowserRuntimeAbiManifestResourceBytes();
  const assets: CppCuteBrowserAssetV1[] = [
    {
      assetId: "adapter",
      kind: "semantic-adapter-manifest",
      url: "/assets/browsergrad-cpp-cute/semantic-adapter-manifest.json",
      urlPolicy: "same-origin-root-relative",
      sha256: CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
      byteLength: wire(adapterBytes.byteLength),
      unpackedByteLength: wire(adapterBytes.byteLength),
      mediaType:
        "application/vnd.browsergrad.cpp-cute.semantic-adapter.v1+json",
      compression: "identity",
      buildSubjectId,
    },
    {
      assetId: "clang-wasm",
      kind: "clang-extractor-wasm",
      url: "/assets/browsergrad-cpp-cute/clang-extractor.wasm",
      urlPolicy: "same-origin-root-relative",
      sha256: profileRecord.deployment.extractor.binarySha256,
      byteLength: wireFromNumber(
        wasmByteLength,
        "$.input.wasmReproducibility.wasmByteLength",
      ),
      unpackedByteLength: wireFromNumber(
        wasmByteLength,
        "$.input.wasmReproducibility.wasmByteLength",
      ),
      mediaType: "application/wasm",
      compression: "identity",
      buildSubjectId,
      sourceAbiSha256,
    },
    {
      assetId: "diagnostic-normalization",
      kind: "diagnostic-normalization-manifest",
      url: "/assets/browsergrad-cpp-cute/diagnostic-normalization.json",
      urlPolicy: "same-origin-root-relative",
      sha256: CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
      byteLength: wire(diagnosticBytes.byteLength),
      unpackedByteLength: wire(diagnosticBytes.byteLength),
      mediaType:
        "application/vnd.browsergrad.cpp-cute.diagnostic-normalization.v1+json",
      compression: "identity",
      buildSubjectId,
    },
    {
      assetId: "runtime-abi",
      kind: "runtime-abi-manifest",
      url: "/assets/browsergrad-cpp-cute/runtime-abi-manifest.json",
      urlPolicy: "same-origin-root-relative",
      sha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
      byteLength: wire(runtimeBytes.byteLength),
      unpackedByteLength: wire(runtimeBytes.byteLength),
      mediaType:
        "application/vnd.browsergrad.cpp-cute.runtime-abi-manifest.v1+json",
      compression: "identity",
      buildSubjectId,
      runtimeAbiId:
        "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1",
      runtimeAbiManifestId:
        CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
    },
  ];
  for (const root of profileRecord.virtualFileSystem.includeRoots) {
    if (root.owner.kind === "source") continue;
    const pack = requiredPack(
      packs,
      root.includeRootId as CppCuteBrowserDistributionPackIncludeRootId,
    );
    const common = {
      assetId: root.includeRootId,
      url: `/${pack.outputPath}`,
      urlPolicy: "same-origin-root-relative" as const,
      sha256: pack.pack.packSha256,
      byteLength: pack.pack.packByteLength,
      unpackedByteLength: pack.pack.packByteLength,
      fileContentByteLength: pack.pack.fileContentByteLength,
      mediaType:
        "application/vnd.browsergrad.vfs-pack.v1" as const,
      compression: "identity" as const,
      buildSubjectId,
      includeRootId: root.includeRootId,
      mountedVirtualRoot: root.virtualPath,
      contentSetSha256: pack.pack.contentSetSha256,
    };
    assets.push(root.owner.kind === "compiler-resource-directory"
      ? { ...common, kind: "compiler-resource-pack" }
      : {
          ...common,
          kind: "dependency-header-pack",
          dependencyId: root.owner.dependencyId,
        });
  }
  assets.sort((left, right) => compareText(left.assetId, right.assetId));
  return Object.freeze(assets);
}

function assetTotals(
  assets: readonly CppCuteBrowserAssetV1[],
): CppCuteBrowserAssetManifestBodyV1["totals"] {
  let compressed = 0n;
  let unpacked = 0n;
  let fileContent = 0n;
  for (const asset of assets) {
    compressed += BigInt(asset.byteLength);
    unpacked += BigInt(asset.unpackedByteLength);
    if (asset.kind === "compiler-resource-pack" ||
        asset.kind === "dependency-header-pack") {
      fileContent += BigInt(asset.fileContentByteLength);
    }
  }
  return {
    compressedByteLength: compressed.toString() as WireU64,
    unpackedByteLength: unpacked.toString() as WireU64,
    fileContentByteLength: fileContent.toString() as WireU64,
  };
}

function normalizeInput(
  input: PrepareCppCuteBrowserDistributionMetadataInput,
): Readonly<{
  buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  packs: ReadonlyMap<
    CppCuteBrowserDistributionPackIncludeRootId,
    CppCuteBrowserDistributionPackInput
  >;
  producerTrustPolicy: AdmittedCppCuteBrowserProducerTrustPolicy;
  wasmReproducibility: VerifiedCppCuteBrowserReproducibility;
  workerBundle: VerifiedCppCuteBrowserWorkerBundle;
}> {
  if (typeof input !== "object" || input === null ||
      Object.getPrototypeOf(input) !== Object.prototype) {
    invalid("$.input", "expected one plain input record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(input);
  const keys = [
    "buildInputLock",
    "packs",
    "producerTrustPolicy",
    "wasmReproducibility",
    "workerBundle",
  ];
  const actual = Reflect.ownKeys(descriptors);
  if (actual.length !== keys.length ||
      actual.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid("$.input", `expected only ${keys.join(", ")}`);
  }
  const value = <K extends keyof PrepareCppCuteBrowserDistributionMetadataInput>(
    key: K,
  ): PrepareCppCuteBrowserDistributionMetadataInput[K] => {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor) ||
        descriptor.enumerable !== true) {
      invalid(`$.input.${key}`, "expected one enumerable data property");
    }
    return descriptor.value as
      PrepareCppCuteBrowserDistributionMetadataInput[K];
  };
  const packs = normalizePacks(value("packs"));
  return Object.freeze({
    buildInputLock: value("buildInputLock"),
    packs,
    producerTrustPolicy: value("producerTrustPolicy"),
    wasmReproducibility: value("wasmReproducibility"),
    workerBundle: value("workerBundle"),
  });
}

function normalizePacks(
  value: readonly CppCuteBrowserDistributionPackInput[],
): ReadonlyMap<
  CppCuteBrowserDistributionPackIncludeRootId,
  CppCuteBrowserDistributionPackInput
> {
  if (!Array.isArray(value) || value.length !== PACK_BINDINGS.length) {
    invalid("$.input.packs", "expected exactly five distribution packs");
  }
  const packs = new Map<
    CppCuteBrowserDistributionPackIncludeRootId,
    CppCuteBrowserDistributionPackInput
  >();
  for (const [index, raw] of value.entries()) {
    if (typeof raw !== "object" || raw === null ||
        Object.getPrototypeOf(raw) !== Object.prototype) {
      invalid(`$.input.packs[${index}]`, "expected one plain pack record");
    }
    const descriptors = Object.getOwnPropertyDescriptors(raw);
    const keys = ["includeRootId", "outputPath", "pack"];
    const actual = Reflect.ownKeys(descriptors);
    if (actual.length !== keys.length ||
        actual.some((key) =>
          typeof key !== "string" || !keys.includes(key))) {
      invalid(
        `$.input.packs[${index}]`,
        `expected only ${keys.join(", ")}`,
      );
    }
    const includeRootId =
      dataValue(descriptors, "includeRootId", `$.input.packs[${index}]`);
    const outputPath =
      dataValue(descriptors, "outputPath", `$.input.packs[${index}]`);
    const pack = dataValue(
      descriptors,
      "pack",
      `$.input.packs[${index}]`,
    );
    const binding_ = PACK_BINDINGS.find(
      (candidate) => candidate.includeRootId === includeRootId,
    );
    if (binding_ === undefined || outputPath !== binding_.outputPath) {
      binding(
        `$.input.packs[${index}]`,
        "pack include-root/output-path binding differs from the distribution contract",
      );
    }
    try {
      unwrapInspectedCppCuteBrowserVfsPack(
        pack as InspectedCppCuteBrowserVfsPack,
      );
    } catch (cause) {
      unverified(
        `$.input.packs[${index}].pack`,
        "expected one opaque inspected VFS pack",
        { cause },
      );
    }
    if (packs.has(binding_.includeRootId)) {
      invalid("$.input.packs", "pack include roots must be unique");
    }
    packs.set(binding_.includeRootId, Object.freeze({
      includeRootId: binding_.includeRootId,
      outputPath: binding_.outputPath,
      pack: pack as InspectedCppCuteBrowserVfsPack,
    }));
  }
  for (const binding_ of PACK_BINDINGS) {
    if (!packs.has(binding_.includeRootId)) {
      invalid(
        "$.input.packs",
        `missing ${binding_.includeRootId} distribution pack`,
      );
    }
  }
  return packs;
}

function dataValue(
  descriptors: PropertyDescriptorMap,
  key: string,
  path: string,
): unknown {
  const descriptor = descriptors[key];
  if (descriptor === undefined || !("value" in descriptor) ||
      descriptor.enumerable !== true) {
    invalid(`${path}.${key}`, "expected one enumerable data property");
  }
  return descriptor.value;
}

function requiredPack(
  packs: ReadonlyMap<
    CppCuteBrowserDistributionPackIncludeRootId,
    CppCuteBrowserDistributionPackInput
  >,
  includeRootId: CppCuteBrowserDistributionPackIncludeRootId,
): CppCuteBrowserDistributionPackInput {
  const pack = packs.get(includeRootId);
  if (pack === undefined) {
    invalid("$.input.packs", `missing ${includeRootId} pack`);
  }
  return pack;
}

function wire(value: number): WireU64 {
  if (!Number.isSafeInteger(value) || value <= 0) {
    invalid("$.byteLength", "expected one positive safe byte length");
  }
  return value.toString() as WireU64;
}

function wireFromNumber(
  value: number,
  path: string,
): WireU64 {
  if (!Number.isSafeInteger(value) || value <= 0) {
    invalid(path, "expected one positive safe byte length");
  }
  return value.toString() as WireU64;
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function sameBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function invalid(path: string, message: string): never {
  throw new CppCuteBrowserDistributionMetadataError(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-METADATA-INVALID",
    path,
    message,
  );
}

function binding(path: string, message: string): never {
  throw new CppCuteBrowserDistributionMetadataError(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-METADATA-BINDING",
    path,
    message,
  );
}

function unverified(
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserDistributionMetadataError(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-METADATA-UNVERIFIED",
    path,
    message,
    options,
  );
}
