import {
  canonicalJsonBytes,
  hashCanonicalJson,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA,
  cppCuteBrowserSourceAbi,
  deriveCppCuteBrowserAssetSetSha256,
  deriveCppCuteBrowserAssetManifestId,
  type CppCuteBrowserAssetManifestBodyV1,
  type CppCuteBrowserAssetManifestV1,
  type CppCuteBrowserAssetV1,
} from "../../../src/cpp_cute_browser_assets.js";
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

const PROVENANCE_ID = `bg.build-provenance.sha256.${"9".repeat(64)}`;

export interface CppCuteBrowserAssetFixture {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly input: CppCuteBrowserAssetManifestV1;
  readonly bytes: Uint8Array;
}

export interface CppCuteBrowserAssetFixtureOptions {
  readonly profile?: CppCuteBrowserProfileFixtureOptions;
  readonly assetLimits?: Partial<CppCuteFrontendBrowserAssetLimits>;
}

export async function createCppCuteBrowserAssetFixture(
  options: CppCuteBrowserAssetFixtureOptions = {},
): Promise<CppCuteBrowserAssetFixture> {
  const provisionalProfile = await prepareCppCuteFrontendProfile(createCppCuteBrowserProfileInput({
    ...options.profile,
    assetSetSha256: "0".repeat(64),
  }));
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
      byteLength: wire(128),
      unpackedByteLength: wire(128),
      mediaType: "application/vnd.browsergrad.cpp-cute.semantic-adapter.v1+json",
      compression: "identity",
      buildProvenanceId: PROVENANCE_ID,
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
      buildProvenanceId: PROVENANCE_ID,
      sourceAbiSha256,
    },
    compilerResourceAsset(provisionalRecord, PROVENANCE_ID),
    ...provisionalRecord.virtualFileSystem.includeRoots
      .filter((root) => root.owner.kind === "dependency")
      .map((root, index): CppCuteBrowserAssetV1 => {
        if (root.owner.kind !== "dependency") throw new Error("fixture dependency root narrowing failed");
        return {
          assetId: `dependency.${root.owner.dependencyId}`,
          kind: "dependency-header-pack",
          url: `/browsergrad/cpp-cute/${root.owner.dependencyId}.headers.tar.gz`,
          urlPolicy: "same-origin-root-relative",
          sha256: (index + 3).toString(16).repeat(64),
          byteLength: wire(2_000 + index),
          unpackedByteLength: wire(16_000 + index),
          mediaType: "application/vnd.browsergrad.vfs-pack.v1+tar",
          compression: "gzip",
          buildProvenanceId: PROVENANCE_ID,
          dependencyId: root.owner.dependencyId,
          includeRootId: root.includeRootId,
          mountedVirtualRoot: root.virtualPath,
          contentSetSha256: root.manifestSha256,
        };
      }),
  ];
  assets.sort((left, right) => left.assetId < right.assetId ? -1 : left.assetId > right.assetId ? 1 : 0);
  const compressed = assets.reduce((total, asset) => total + BigInt(asset.byteLength), 0n);
  const unpacked = assets.reduce((total, asset) => total + BigInt(asset.unpackedByteLength), 0n);
  const mountedVirtualRoots = assets.flatMap((asset): string[] =>
    asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack"
      ? [asset.mountedVirtualRoot]
      : []).sort();
  const assetSetSha256 = await deriveCppCuteBrowserAssetSetSha256({
    sourceAbiSha256,
    dependencyIds: provisionalRecord.toolchain.dependencies.map((dependency) => dependency.dependencyId),
    buildProvenanceIds: [PROVENANCE_ID],
    mountedVirtualRoots,
    assets,
  });
  const finalProfileInput = structuredClone(createCppCuteBrowserProfileInput({
    ...options.profile,
    assetSetSha256,
  }));
  if (options.assetLimits !== undefined) {
    Object.assign(finalProfileInput.deployment.assetLimits, options.assetLimits);
  }
  const profile = await prepareCppCuteFrontendProfile(finalProfileInput);
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(profile).profile;
  const body: CppCuteBrowserAssetManifestBodyV1 = {
    profileHash: profile.profileHash,
    sourceAbi,
    sourceAbiSha256,
    assetSetSha256,
    dependencyIds: profileRecord.toolchain.dependencies.map((dependency) => dependency.dependencyId),
    buildProvenanceIds: [PROVENANCE_ID],
    mountedVirtualRoots,
    assets,
    totals: {
      compressedByteLength: compressed.toString() as WireU64,
      unpackedByteLength: unpacked.toString() as WireU64,
    },
  };
  const input: CppCuteBrowserAssetManifestV1 = {
    schema: CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA,
    version: { major: 1, minor: 0 },
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
  buildProvenanceId: string,
): CppCuteBrowserAssetV1 {
  const root = profile.virtualFileSystem.includeRoots.find((entry) =>
    entry.owner.kind === "compiler-resource-directory");
  if (root === undefined) throw new Error("fixture profile lost compiler resource root");
  return {
    assetId: "compiler-resource",
    kind: "compiler-resource-pack",
    url: "/browsergrad/cpp-cute/clang-resource.tar.gz",
    urlPolicy: "same-origin-root-relative",
    sha256: "2".repeat(64),
    byteLength: wire(1_000),
    unpackedByteLength: wire(8_000),
    mediaType: "application/vnd.browsergrad.vfs-pack.v1+tar",
    compression: "gzip",
    buildProvenanceId,
    includeRootId: root.includeRootId,
    mountedVirtualRoot: root.virtualPath,
    contentSetSha256: root.manifestSha256,
  };
}

function wire(value: number): WireU64 {
  return String(value) as WireU64;
}
