import {
  canonicalJsonBytes,
  hashCanonicalJson,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR,
  CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR,
  CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA,
  cppCuteBrowserSourceAbi,
  deriveCppCuteBrowserAssetSetSha256,
  deriveCppCuteBrowserAssetManifestId,
  type CppCuteBrowserAssetManifestBodyV1,
  type CppCuteBrowserAssetManifestV1,
  type CppCuteBrowserAssetV1,
} from "../../../src/cpp_cute_browser_assets.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
} from "../../../src/cpp_cute_browser_runtime_abi.js";
import {
  cppCuteDiagnosticNormalizationResourceBytes,
} from "../../../src/cpp_cute_diagnostic_normalization.js";
import {
  cppCuteSemanticAdapterManifestResourceBytes,
} from "../../../src/cpp_cute_semantic_adapter_manifest.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type CppCuteFrontendBrowserAssetLimits,
  type PreparedCppCuteFrontendProfile,
} from "../../../src/cpp_cute_frontend_profile.js";
import {
  createCppCuteBrowserProfileInput,
  type CppCuteBrowserProfileFixtureOptions,
} from "./cpp_cute_frontend_fixtures.js";

const BUILD_SUBJECT_ID = `bg.cpp.browser-build-subject.sha256.${"9".repeat(64)}`;
const SEMANTIC_ADAPTER_RESOURCE_BYTE_LENGTH =
  cppCuteSemanticAdapterManifestResourceBytes().byteLength;
const DIAGNOSTIC_NORMALIZATION_RESOURCE_BYTE_LENGTH =
  cppCuteDiagnosticNormalizationResourceBytes().byteLength;

export interface CppCuteBrowserAssetFixture {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly input: CppCuteBrowserAssetManifestV1;
  readonly bytes: Uint8Array;
}

export interface CppCuteBrowserAssetFixtureOptions {
  readonly profile?: CppCuteBrowserProfileFixtureOptions;
  readonly assetLimits?: Partial<CppCuteFrontendBrowserAssetLimits>;
  readonly packOverrides?: Readonly<Record<string, CppCuteBrowserPackAssetFixtureOverride>>;
}

export interface CppCuteBrowserPackAssetFixtureOverride {
  readonly sha256: string;
  readonly byteLength: number;
  readonly fileContentByteLength: number;
  readonly contentSetSha256: string;
}

export async function createCppCuteBrowserAssetFixture(
  options: CppCuteBrowserAssetFixtureOptions = {},
): Promise<CppCuteBrowserAssetFixture> {
  const provisionalProfile = await prepareCppCuteFrontendProfile(
    browserProfileInput(options, "0".repeat(64)),
  );
  const provisionalRecord = unwrapPreparedCppCuteBrowserFrontendProfile(provisionalProfile).profile;
  const sourceAbi = cppCuteBrowserSourceAbi(provisionalProfile);
  const sourceAbiSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-source-abi.v1",
    sourceAbi,
  });
  const assets: CppCuteBrowserAssetV1[] = [
    {
      assetId: "adapter",
      kind: "semantic-adapter-manifest",
      url: "/browsergrad/cpp-cute/semantic-adapter.json",
      urlPolicy: "same-origin-root-relative",
      sha256: provisionalRecord.deployment.extractor.semanticAdapterManifestSha256,
      byteLength: wire(SEMANTIC_ADAPTER_RESOURCE_BYTE_LENGTH),
      unpackedByteLength: wire(SEMANTIC_ADAPTER_RESOURCE_BYTE_LENGTH),
      mediaType: "application/vnd.browsergrad.cpp-cute.semantic-adapter.v1+json",
      compression: "identity",
      buildSubjectId: BUILD_SUBJECT_ID,
    },
    {
      assetId: "clang-wasm",
      kind: "clang-extractor-wasm",
      url: "/browsergrad/cpp-cute/clang-extractor.wasm",
      urlPolicy: "same-origin-root-relative",
      sha256: provisionalRecord.deployment.extractor.binarySha256,
      byteLength: wire(4_096),
      unpackedByteLength: wire(4_096),
      mediaType: "application/wasm",
      compression: "identity",
      buildSubjectId: BUILD_SUBJECT_ID,
      sourceAbiSha256,
    },
    {
      assetId: "diagnostic-normalization",
      kind: "diagnostic-normalization-manifest",
      url: "/browsergrad/cpp-cute/diagnostic-normalization.json",
      urlPolicy: "same-origin-root-relative",
      sha256: provisionalRecord.language.diagnostics.normalizationManifestSha256,
      byteLength: wire(DIAGNOSTIC_NORMALIZATION_RESOURCE_BYTE_LENGTH),
      unpackedByteLength: wire(DIAGNOSTIC_NORMALIZATION_RESOURCE_BYTE_LENGTH),
      mediaType: "application/vnd.browsergrad.cpp-cute.diagnostic-normalization.v1+json",
      compression: "identity",
      buildSubjectId: BUILD_SUBJECT_ID,
    },
    {
      assetId: "runtime-abi",
      kind: "runtime-abi-manifest",
      url: "/browsergrad/cpp-cute/runtime-abi-manifest.json",
      urlPolicy: "same-origin-root-relative",
      sha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
      byteLength: wire(cppCuteBrowserRuntimeAbiManifestResourceBytes().byteLength),
      unpackedByteLength: wire(cppCuteBrowserRuntimeAbiManifestResourceBytes().byteLength),
      mediaType: "application/vnd.browsergrad.cpp-cute.runtime-abi-manifest.v1+json",
      compression: "identity",
      buildSubjectId: BUILD_SUBJECT_ID,
      runtimeAbiId: "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1",
      runtimeAbiManifestId: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
    },
    compilerResourceAsset(provisionalRecord, BUILD_SUBJECT_ID, options.packOverrides),
    ...provisionalRecord.virtualFileSystem.includeRoots
      .filter((root) => root.owner.kind === "dependency")
      .map((root, index): CppCuteBrowserAssetV1 => {
        if (root.owner.kind !== "dependency") throw new Error("fixture dependency root narrowing failed");
        const override = options.packOverrides?.[root.includeRootId];
        const byteLength = override?.byteLength ?? 16_500 + index;
        return {
          assetId: `dependency.${root.owner.dependencyId}`,
          kind: "dependency-header-pack",
          url: `/browsergrad/cpp-cute/${root.owner.dependencyId}.headers.bgvfs`,
          urlPolicy: "same-origin-root-relative",
          sha256: override?.sha256 ?? (index + 3).toString(16).repeat(64),
          byteLength: wire(byteLength),
          unpackedByteLength: wire(byteLength),
          fileContentByteLength: wire(override?.fileContentByteLength ?? 16_000 + index),
          mediaType: "application/vnd.browsergrad.vfs-pack.v1",
          compression: "identity",
          buildSubjectId: BUILD_SUBJECT_ID,
          dependencyId: root.owner.dependencyId,
          includeRootId: root.includeRootId,
          mountedVirtualRoot: root.virtualPath,
          contentSetSha256: override?.contentSetSha256 ?? root.manifestSha256,
        };
      }),
  ];
  assets.sort((left, right) => left.assetId < right.assetId ? -1 : left.assetId > right.assetId ? 1 : 0);
  const compressed = assets.reduce((total, asset) => total + BigInt(asset.byteLength), 0n);
  const unpacked = assets.reduce((total, asset) => total + BigInt(asset.unpackedByteLength), 0n);
  const fileContent = assets.reduce((total, asset) => total + (
    asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack"
      ? BigInt(asset.fileContentByteLength)
      : 0n
  ), 0n);
  const mountedVirtualRoots = assets.flatMap((asset): string[] =>
    asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack"
      ? [asset.mountedVirtualRoot]
      : []).sort();
  const assetSetSha256 = await deriveCppCuteBrowserAssetSetSha256({
    sourceAbiSha256,
    dependencyIds: provisionalRecord.toolchain.dependencies.map((dependency) => dependency.dependencyId),
    buildSubjectIds: [BUILD_SUBJECT_ID],
    mountedVirtualRoots,
    assets,
  });
  const finalProfileInput = browserProfileInput(options, assetSetSha256);
  if (options.assetLimits !== undefined) {
    Object.assign(finalProfileInput.deployment.assetLimits, options.assetLimits);
    const runtimeVfs = finalProfileInput.deployment.compilerRuntime.virtualFileSystem as {
      maxRetainedHostPackByteLength: number;
    };
    runtimeVfs.maxRetainedHostPackByteLength = Math.min(
      runtimeVfs.maxRetainedHostPackByteLength,
      finalProfileInput.deployment.assetLimits.maxTotalUnpackedByteLength,
    );
  }
  const profile = await prepareCppCuteFrontendProfile(finalProfileInput);
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(profile).profile;
  const body: CppCuteBrowserAssetManifestBodyV1 = {
    profileHash: profile.profileHash,
    sourceAbi,
    sourceAbiSha256,
    assetSetSha256,
    dependencyIds: profileRecord.toolchain.dependencies.map((dependency) => dependency.dependencyId),
    buildSubjectIds: [BUILD_SUBJECT_ID],
    mountedVirtualRoots,
    assets,
    totals: {
      compressedByteLength: compressed.toString() as WireU64,
      unpackedByteLength: unpacked.toString() as WireU64,
      fileContentByteLength: fileContent.toString() as WireU64,
    },
  };
  const input: CppCuteBrowserAssetManifestV1 = {
    schema: CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR,
      minor: CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR,
    },
    manifestId: await deriveCppCuteBrowserAssetManifestId(body),
    body,
  };
  return { profile, input, bytes: canonicalJsonBytes(input) };
}

export function cloneCppCuteBrowserAssetInput(
  input: CppCuteBrowserAssetManifestV1,
): Record<string, unknown> {
  return structuredClone(input) as unknown as Record<string, unknown>;
}

function compilerResourceAsset(
  profile: ReturnType<typeof unwrapPreparedCppCuteBrowserFrontendProfile>["profile"],
  buildSubjectId: string,
  overrides: CppCuteBrowserAssetFixtureOptions["packOverrides"],
): CppCuteBrowserAssetV1 {
  const root = profile.virtualFileSystem.includeRoots.find((entry) =>
    entry.owner.kind === "compiler-resource-directory");
  if (root === undefined) throw new Error("fixture profile lost compiler resource root");
  const override = overrides?.[root.includeRootId];
  const byteLength = override?.byteLength ?? 8_500;
  return {
    assetId: "compiler-resource",
    kind: "compiler-resource-pack",
    url: "/browsergrad/cpp-cute/clang-resource.bgvfs",
    urlPolicy: "same-origin-root-relative",
    sha256: override?.sha256 ?? "2".repeat(64),
    byteLength: wire(byteLength),
    unpackedByteLength: wire(byteLength),
    fileContentByteLength: wire(override?.fileContentByteLength ?? 8_000),
    mediaType: "application/vnd.browsergrad.vfs-pack.v1",
    compression: "identity",
    buildSubjectId,
    includeRootId: root.includeRootId,
    mountedVirtualRoot: root.virtualPath,
    contentSetSha256: override?.contentSetSha256 ?? root.manifestSha256,
  };
}

function browserProfileInput(
  options: CppCuteBrowserAssetFixtureOptions,
  assetSetSha256: string,
): ReturnType<typeof createCppCuteBrowserProfileInput> {
  const input = structuredClone(createCppCuteBrowserProfileInput({
    ...options.profile,
    assetSetSha256,
  }));
  for (const root of input.virtualFileSystem.includeRoots) {
    const override = options.packOverrides?.[root.includeRootId];
    if (override === undefined) continue;
    (root as { manifestSha256: string }).manifestSha256 = override.contentSetSha256;
    if (root.owner.kind === "compiler-resource-directory") {
      (input.toolchain.compiler as { resourceDirectorySha256: string }).resourceDirectorySha256 =
        override.contentSetSha256;
      continue;
    }
    if (root.owner.kind === "dependency") {
      const dependency = input.toolchain.dependencies.find((entry) =>
        entry.dependencyId === root.owner.dependencyId);
      if (dependency === undefined) throw new Error("fixture pack override lost dependency");
      (dependency as { headerSetSha256: string }).headerSetSha256 = override.contentSetSha256;
    }
  }
  return input;
}

function wire(value: number): WireU64 {
  return String(value) as WireU64;
}
