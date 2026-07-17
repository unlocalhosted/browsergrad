import {
  hashCanonicalJson,
  sha256Hex,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  acquireCppCuteBrowserAssetSet,
  admitCppCuteBrowserAssetSetToCache,
  copyVerifiedCppCuteBrowserAssetBytes,
  CppCuteBrowserAssetInstallationError,
  decodeAcquiredCppCuteBrowserRuntimeAbiAsset,
  installCppCuteBrowserVfs,
  loadCppCuteBrowserAssetSetFromCache,
  unwrapVerifiedCppCuteBrowserRuntimeAbiAsset,
  unwrapVerifiedCppCuteBrowserVfsInstallation,
  verifyTransferredCppCuteBrowserAssetSet,
  type CppCuteBrowserTransferredAssetInput,
  type VerifiedCppCuteBrowserRuntimeAbiAsset,
  type CppCuteBrowserContentCache,
  type CppCuteBrowserHostFetch,
} from "../../src/cpp_cute_browser_asset_installation.js";
import {
  CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR,
  CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR,
  CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA,
  cppCuteBrowserSourceAbi,
  deriveCppCuteBrowserAssetManifestId,
  deriveCppCuteBrowserAssetSetSha256,
  prepareCppCuteBrowserAssetManifest,
  unwrapPreparedCppCuteBrowserAssetManifest,
  type CppCuteBrowserAssetManifestBodyV1,
  type CppCuteBrowserAssetManifestV1,
  type CppCuteBrowserAssetV1,
  type PreparedCppCuteBrowserAssetManifest,
} from "../../src/cpp_cute_browser_assets.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
} from "../../src/cpp_cute_browser_runtime_abi.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteBrowserFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  encodeCppCuteBrowserVfsPack,
  inspectCppCuteBrowserVfsPack,
} from "../../src/cpp_cute_browser_vfs_pack.js";
import { createCppCuteBrowserProfileInput } from "./support/cpp_cute_frontend_fixtures.js";

const ORIGIN = "https://assets.example.test";
const PROVENANCE_ID = `bg.build-provenance.sha256.${"9".repeat(64)}`;

interface Environment {
  readonly manifest: PreparedCppCuteBrowserAssetManifest;
  readonly bytesByUrl: ReadonlyMap<string, Uint8Array>;
}

interface EnvironmentOptions {
  readonly collision?: boolean;
  readonly retainedPackCopies?: 1 | 2;
}

class MemoryCache implements CppCuteBrowserContentCache {
  readonly entries = new Map<string, Uint8Array>();

  async get(contentSha256: string): Promise<unknown | undefined> {
    return this.entries.get(contentSha256);
  }

  async put(contentSha256: string, bytes: Uint8Array): Promise<void> {
    this.entries.set(contentSha256, bytes);
  }
}

async function expectIoError(
  operation: Promise<unknown>,
  code: CppCuteBrowserAssetInstallationError["code"],
  path?: string,
): Promise<void> {
  await expect(operation).rejects.toMatchObject(path === undefined ? { code } : { code, path });
}

function hostFetch(
  bytesByUrl: ReadonlyMap<string, Uint8Array>,
  observe?: (url: string, init: RequestInit) => void,
): CppCuteBrowserHostFetch {
  return async (url, init) => {
    observe?.(url, init);
    const bytes = bytesByUrl.get(url);
    if (bytes === undefined) return response(url, Uint8Array.of(), { status: 404 });
    return response(url, bytes);
  };
}

function response(
  url: string,
  bytes: Uint8Array,
  options: { readonly status?: number; readonly contentLength?: string | null; readonly redirected?: boolean } = {},
): Response {
  const headers = new Headers();
  if (options.contentLength !== null) headers.set("content-length", options.contentLength ?? String(bytes.byteLength));
  const value = new Response(bytes.slice().buffer, { status: options.status ?? 200, headers });
  Object.defineProperty(value, "url", { configurable: true, value: url });
  Object.defineProperty(value, "redirected", { configurable: true, value: options.redirected ?? false });
  return value;
}

function streamResponse(
  url: string,
  body: ReadableStream<Uint8Array>,
  options: { readonly status?: number; readonly contentLength?: string | null; readonly redirected?: boolean } = {},
): Response {
  const headers = new Headers();
  if (options.contentLength !== null) headers.set("content-length", options.contentLength ?? "0");
  const value = new Response(body, { status: options.status ?? 200, headers });
  Object.defineProperty(value, "url", { configurable: true, value: url });
  Object.defineProperty(value, "redirected", { configurable: true, value: options.redirected ?? false });
  return value;
}

async function createEnvironment(options: EnvironmentOptions = {}): Promise<Environment> {
  const collision = options.collision ?? false;
  const input = structuredClone(createCppCuteBrowserProfileInput());
  if (collision) {
    const cuda = input.virtualFileSystem.includeRoots.find((root) => root.includeRootId === "cuda");
    const cutlass = input.virtualFileSystem.includeRoots.find((root) => root.includeRootId === "cutlass");
    if (cuda === undefined || cutlass === undefined) throw new Error("fixture roots missing");
    (cutlass as { virtualPath: string }).virtualPath = `${cuda.virtualPath}/nested`;
  }

  const adapterBytes = Uint8Array.of(1, 2, 3);
  const wasmBytes = Uint8Array.of(4, 5, 6, 7);
  const runtimeAbiBytes = cppCuteBrowserRuntimeAbiManifestResourceBytes();
  const adapterSha256 = await sha256Hex(adapterBytes);
  const wasmSha256 = await sha256Hex(wasmBytes);
  (input.toolchain.compiler as { binarySha256: string }).binarySha256 = wasmSha256;
  (input.deployment.extractor as { binarySha256: string; semanticAdapterManifestSha256: string }).binarySha256 =
    wasmSha256;
  (input.deployment.extractor as { binarySha256: string; semanticAdapterManifestSha256: string })
    .semanticAdapterManifestSha256 = adapterSha256;

  const packByRoot = new Map<string, {
    readonly bytes: Uint8Array;
    readonly sha256: string;
    readonly contentSetSha256: string;
    readonly fileContentByteLength: WireU64;
  }>();
  let ordinal = 0;
  for (const root of input.virtualFileSystem.includeRoots) {
    if (root.owner.kind === "source") continue;
    let relative = "header.h";
    if (collision && root.includeRootId === "cuda") relative = "nested/collision.h";
    if (collision && root.includeRootId === "cutlass") relative = "collision.h";
    const bytes = await encodeCppCuteBrowserVfsPack([{
      virtualPath: relative,
      bytes: Uint8Array.of(ordinal + 1),
    }]);
    const inspected = await inspectCppCuteBrowserVfsPack(bytes);
    packByRoot.set(root.includeRootId, {
      bytes,
      sha256: inspected.packSha256,
      contentSetSha256: inspected.contentSetSha256,
      fileContentByteLength: inspected.fileContentByteLength,
    });
    (root as { manifestSha256: string }).manifestSha256 = inspected.contentSetSha256;
    if (root.owner.kind === "compiler-resource-directory") {
      (input.toolchain.compiler as { resourceDirectorySha256: string }).resourceDirectorySha256 =
        inspected.contentSetSha256;
    } else {
      const dependency = input.toolchain.dependencies.find((entry) => entry.dependencyId === root.owner.dependencyId);
      if (dependency === undefined) throw new Error("fixture dependency missing");
      (dependency as { headerSetSha256: string }).headerSetSha256 = inspected.contentSetSha256;
    }
    ordinal += 1;
  }

  if (options.retainedPackCopies !== undefined) {
    const sourcePackBytes = [...packByRoot.values()].reduce((total, pack) => total + BigInt(pack.bytes.byteLength), 0n);
    (input.deployment.compilerRuntime.virtualFileSystem as { maxRetainedHostPackByteLength: number })
      .maxRetainedHostPackByteLength = Number(sourcePackBytes * BigInt(options.retainedPackCopies));
  }

  const provisional = await prepareCppCuteFrontendProfile(input);
  const provisionalProfile = unwrapPreparedCppCuteBrowserFrontendProfile(provisional).profile;
  const sourceAbi = cppCuteBrowserSourceAbi(provisional);
  const sourceAbiSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-source-abi.v1",
    sourceAbi,
  });
  const assets: CppCuteBrowserAssetV1[] = [
    {
      assetId: "adapter",
      kind: "semantic-adapter-manifest",
      url: "/assets/adapter.json",
      urlPolicy: "same-origin-root-relative",
      sha256: adapterSha256,
      byteLength: wire(adapterBytes.byteLength),
      unpackedByteLength: wire(adapterBytes.byteLength),
      mediaType: "application/vnd.browsergrad.cpp-cute.semantic-adapter.v1+json",
      compression: "identity",
      buildProvenanceId: PROVENANCE_ID,
    },
    {
      assetId: "clang-wasm",
      kind: "clang-extractor-wasm",
      url: "/assets/clang.wasm",
      urlPolicy: "same-origin-root-relative",
      sha256: wasmSha256,
      byteLength: wire(wasmBytes.byteLength),
      unpackedByteLength: wire(wasmBytes.byteLength),
      mediaType: "application/wasm",
      compression: "identity",
      buildProvenanceId: PROVENANCE_ID,
      sourceAbiSha256,
    },
    {
      assetId: "runtime-abi",
      kind: "runtime-abi-manifest",
      url: "/assets/runtime-abi-manifest.json",
      urlPolicy: "same-origin-root-relative",
      sha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
      byteLength: wire(runtimeAbiBytes.byteLength),
      unpackedByteLength: wire(runtimeAbiBytes.byteLength),
      mediaType: "application/vnd.browsergrad.cpp-cute.runtime-abi-manifest.v1+json",
      compression: "identity",
      buildProvenanceId: PROVENANCE_ID,
      runtimeAbiId: "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1",
      runtimeAbiManifestId: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
    },
  ];
  for (const root of provisionalProfile.virtualFileSystem.includeRoots) {
    if (root.owner.kind === "source") continue;
    const pack = packByRoot.get(root.includeRootId);
    if (pack === undefined) throw new Error("fixture pack missing");
    const common = {
      assetId: root.owner.kind === "compiler-resource-directory"
        ? "compiler-resource"
        : `dependency.${root.owner.dependencyId}`,
      url: `/assets/${root.includeRootId}.bgvfs`,
      urlPolicy: "same-origin-root-relative" as const,
      sha256: pack.sha256,
      byteLength: wire(pack.bytes.byteLength),
      unpackedByteLength: wire(pack.bytes.byteLength),
      fileContentByteLength: pack.fileContentByteLength,
      mediaType: "application/vnd.browsergrad.vfs-pack.v1" as const,
      compression: "identity" as const,
      buildProvenanceId: PROVENANCE_ID,
      includeRootId: root.includeRootId,
      mountedVirtualRoot: root.virtualPath,
      contentSetSha256: pack.contentSetSha256,
    };
    assets.push(root.owner.kind === "compiler-resource-directory"
      ? { ...common, kind: "compiler-resource-pack" }
      : { ...common, kind: "dependency-header-pack", dependencyId: root.owner.dependencyId });
  }
  assets.sort((left, right) => left.assetId < right.assetId ? -1 : left.assetId > right.assetId ? 1 : 0);
  const mountedVirtualRoots = assets.flatMap((asset): string[] =>
    asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack"
      ? [asset.mountedVirtualRoot]
      : []).sort();
  const dependencyIds = provisionalProfile.toolchain.dependencies.map((entry) => entry.dependencyId);
  const assetSetSha256 = await deriveCppCuteBrowserAssetSetSha256({
    sourceAbiSha256,
    dependencyIds,
    buildProvenanceIds: [PROVENANCE_ID],
    mountedVirtualRoots,
    assets,
  });
  (input.deployment as { assetSetSha256: string }).assetSetSha256 = assetSetSha256;
  const profile = await prepareCppCuteFrontendProfile(input);
  const compressed = assets.reduce((sum, asset) => sum + BigInt(asset.byteLength), 0n);
  const fileContent = assets.reduce((sum, asset) => sum + (
    asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack"
      ? BigInt(asset.fileContentByteLength)
      : 0n
  ), 0n);
  const body: CppCuteBrowserAssetManifestBodyV1 = {
    profileHash: profile.profileHash,
    sourceAbi,
    sourceAbiSha256,
    assetSetSha256,
    dependencyIds,
    buildProvenanceIds: [PROVENANCE_ID],
    mountedVirtualRoots,
    assets,
    totals: {
      compressedByteLength: compressed.toString() as WireU64,
      unpackedByteLength: compressed.toString() as WireU64,
      fileContentByteLength: fileContent.toString() as WireU64,
    },
  };
  const manifestInput: CppCuteBrowserAssetManifestV1 = {
    schema: CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR,
      minor: CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR,
    },
    manifestId: await deriveCppCuteBrowserAssetManifestId(body),
    body,
  };
  const manifest = await prepareCppCuteBrowserAssetManifest(manifestInput, profile);
  const bytesByUrl = new Map<string, Uint8Array>([
    [`${ORIGIN}/assets/adapter.json`, adapterBytes],
    [`${ORIGIN}/assets/clang.wasm`, wasmBytes],
    [`${ORIGIN}/assets/runtime-abi-manifest.json`, runtimeAbiBytes],
  ]);
  for (const root of provisionalProfile.virtualFileSystem.includeRoots) {
    if (root.owner.kind === "source") continue;
    const pack = packByRoot.get(root.includeRootId);
    if (pack === undefined) throw new Error("fixture pack missing");
    bytesByUrl.set(`${ORIGIN}/assets/${root.includeRootId}.bgvfs`, pack.bytes);
  }
  return { manifest, bytesByUrl };
}

function transferredAssets(environment: Environment): CppCuteBrowserTransferredAssetInput[] {
  const manifest = unwrapPreparedCppCuteBrowserAssetManifest(environment.manifest).manifest;
  return manifest.body.assets.map((asset) => {
    const bytes = environment.bytesByUrl.get(`${ORIGIN}${asset.url}`);
    if (bytes === undefined) throw new Error(`fixture asset bytes missing for ${asset.assetId}`);
    return { assetId: asset.assetId, bytes: new Uint8Array(bytes) };
  });
}

describe("C++/CuTe browser asset acquisition and VFS installation", () => {
  it("reconstructs isolated local authority from exact ordered transferred buffers", async () => {
    const environment = await createEnvironment();
    const transferred = transferredAssets(environment);
    const expectedClang = new Uint8Array(
      transferred.find((entry) => entry.assetId === "clang-wasm")?.bytes ?? Uint8Array.of(),
    );
    const pending = verifyTransferredCppCuteBrowserAssetSet(environment.manifest, transferred);
    for (const entry of transferred) entry.bytes.fill(0);
    const assetSet = await pending;

    expect(assetSet).toMatchObject({
      source: "worker-transfer",
      assetCount: transferred.length,
      manifestId: environment.manifest.manifestId,
      assetSetSha256: environment.manifest.assetSetSha256,
    });
    expect(assetSet).not.toHaveProperty("workerExecutionReady");
    expect(assetSet).not.toHaveProperty("releaseReady");
    expect(copyVerifiedCppCuteBrowserAssetBytes(assetSet, "clang-wasm")).toEqual(expectedClang);
    const runtimeAbi = await decodeAcquiredCppCuteBrowserRuntimeAbiAsset(assetSet);
    expect(runtimeAbi).toMatchObject({
      observedWasmVerified: false,
      workerExecutionReady: false,
      releaseReady: false,
    });
    await expect(installCppCuteBrowserVfs(assetSet)).resolves.toMatchObject({
      manifestId: environment.manifest.manifestId,
    });
  });

  it("requires exact manifest cardinality, order, IDs, fields, lengths, and hashes", async () => {
    const environment = await createEnvironment();
    const exact = transferredAssets(environment);

    await expectIoError(
      verifyTransferredCppCuteBrowserAssetSet(environment.manifest, exact.slice(0, -1)),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-INVALID",
      "$.assets",
    );
    await expectIoError(
      verifyTransferredCppCuteBrowserAssetSet(environment.manifest, [...exact, exact[0]!]),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-INVALID",
      "$.assets",
    );
    await expectIoError(
      verifyTransferredCppCuteBrowserAssetSet(environment.manifest, [...exact].reverse()),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-INVALID",
      "$.assets[0].assetId",
    );

    const wrongId = transferredAssets(environment);
    wrongId[0] = { assetId: `${wrongId[0]!.assetId}.wrong`, bytes: wrongId[0]!.bytes };
    await expectIoError(
      verifyTransferredCppCuteBrowserAssetSet(environment.manifest, wrongId),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-INVALID",
      "$.assets[0].assetId",
    );

    const wrongLength = transferredAssets(environment);
    wrongLength[0] = { assetId: wrongLength[0]!.assetId, bytes: new Uint8Array(1) };
    await expectIoError(
      verifyTransferredCppCuteBrowserAssetSet(environment.manifest, wrongLength),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-LENGTH-MISMATCH",
      "$.assets[0].bytes",
    );

    const wrongHash = transferredAssets(environment);
    wrongHash[0]!.bytes[0] = (wrongHash[0]!.bytes[0] ?? 0) ^ 0xff;
    await expectIoError(
      verifyTransferredCppCuteBrowserAssetSet(environment.manifest, wrongHash),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-HASH-MISMATCH",
      "$.assets[0].bytes",
    );

    const extraField = transferredAssets(environment);
    Object.assign(extraField[0]!, { url: "/must-not-be-used" });
    await expectIoError(
      verifyTransferredCppCuteBrowserAssetSet(environment.manifest, extraField),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-INVALID",
      "$.assets[0]",
    );

    let accessorRead = false;
    const hostile = transferredAssets(environment);
    hostile[0] = Object.defineProperty({ assetId: hostile[0]!.assetId }, "bytes", {
      enumerable: true,
      get: () => {
        accessorRead = true;
        throw new Error("transferred accessor must not run");
      },
    }) as CppCuteBrowserTransferredAssetInput;
    await expectIoError(
      verifyTransferredCppCuteBrowserAssetSet(environment.manifest, hostile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-INVALID",
      "$.assets[0].bytes",
    );
    expect(accessorRead).toBe(false);
  });

  it("requires unique standalone unshared readable transfer buffers", async () => {
    const environment = await createEnvironment();

    const duplicate = transferredAssets(environment);
    duplicate[1] = { assetId: duplicate[1]!.assetId, bytes: duplicate[0]!.bytes };
    await expectIoError(
      verifyTransferredCppCuteBrowserAssetSet(environment.manifest, duplicate),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-INVALID",
      "$.assets[1].bytes",
    );

    const partial = transferredAssets(environment);
    const first = partial[0]!.bytes;
    const padded = new Uint8Array(first.byteLength + 1);
    padded.set(first, 1);
    partial[0] = {
      assetId: partial[0]!.assetId,
      bytes: new Uint8Array(padded.buffer, 1, first.byteLength),
    };
    await expectIoError(
      verifyTransferredCppCuteBrowserAssetSet(environment.manifest, partial),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-INVALID",
      "$.assets[0].bytes",
    );

    if (typeof SharedArrayBuffer !== "undefined") {
      const shared = transferredAssets(environment);
      const sharedBytes = new Uint8Array(new SharedArrayBuffer(shared[0]!.bytes.byteLength));
      sharedBytes.set(shared[0]!.bytes);
      shared[0] = { assetId: shared[0]!.assetId, bytes: sharedBytes };
      await expectIoError(
        verifyTransferredCppCuteBrowserAssetSet(environment.manifest, shared),
        "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-INVALID",
        "$.assets[0].bytes",
      );
    }

    const detached = transferredAssets(environment);
    structuredClone(detached[0]!.bytes.buffer, { transfer: [detached[0]!.bytes.buffer] });
    await expect(
      verifyTransferredCppCuteBrowserAssetSet(environment.manifest, detached),
    ).rejects.toMatchObject({ path: "$.assets[0].bytes" });
  });

  it("fetches exact same-origin bytes, verifies them, and installs every collision-free pack", async () => {
    const environment = await createEnvironment();
    const requests: Array<{ readonly url: string; readonly init: RequestInit }> = [];
    const assetSet = await acquireCppCuteBrowserAssetSet(
      environment.manifest,
      ORIGIN,
      hostFetch(environment.bytesByUrl, (url, init) => requests.push({ url, init })),
    );
    expect(assetSet).toMatchObject({ source: "host-fetch", assetCount: environment.bytesByUrl.size });
    expect(requests).toHaveLength(environment.bytesByUrl.size);
    expect(requests.every(({ url, init }) =>
      url.startsWith(`${ORIGIN}/`) && init.redirect === "error" && init.credentials === "same-origin" &&
      init.cache === "no-store")).toBe(true);

    const installation = await installCppCuteBrowserVfs(assetSet);
    const record = unwrapVerifiedCppCuteBrowserVfsInstallation(installation);
    expect(installation.packCount).toBe(record.mounts.length);
    expect(installation.fileCount).toBe(record.files.length);
    expect(record.mounts).toHaveLength(environment.bytesByUrl.size - 3);
    expect(new Set(record.files.map((entry) => entry.virtualPath)).size).toBe(record.files.length);
    const onePackCopy = record.mounts.reduce((total, mount) => total + BigInt(mount.pack.packByteLength), 0n);
    expect(BigInt(installation.sourcePackByteLength)).toBe(onePackCopy);
    expect(BigInt(installation.verifiedPackByteLength)).toBe(onePackCopy);
    expect(BigInt(installation.retainedPackByteLength)).toBe(2n * onePackCopy);
    expect(() => unwrapVerifiedCppCuteBrowserVfsInstallation({ ...installation })).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-UNVERIFIED" }),
    );

    const copy = copyVerifiedCppCuteBrowserAssetBytes(assetSet, "clang-wasm");
    copy.fill(0);
    expect(copyVerifiedCppCuteBrowserAssetBytes(assetSet, "clang-wasm")).toEqual(
      environment.bytesByUrl.get(`${ORIGIN}/assets/clang.wasm`),
    );
    const runtimeAbi = await decodeAcquiredCppCuteBrowserRuntimeAbiAsset(assetSet);
    expect(runtimeAbi).toMatchObject({
      assetManifestId: assetSet.manifestId,
      assetSetSha256: assetSet.assetSetSha256,
      assetId: "runtime-abi",
      runtimeAbiManifestId: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
      runtimeAbiResourceSha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
      runtimeAbiResourceByteLength: String(cppCuteBrowserRuntimeAbiManifestResourceBytes().byteLength),
      designAuthority: true,
      observedWasmVerified: false,
      workerExecutionReady: false,
      releaseReady: false,
    });
    expect(unwrapVerifiedCppCuteBrowserRuntimeAbiAsset(runtimeAbi).assetSet).toBe(assetSet);
    expect(() => unwrapVerifiedCppCuteBrowserRuntimeAbiAsset(
      { ...runtimeAbi } as VerifiedCppCuteBrowserRuntimeAbiAsset,
    )).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-UNVERIFIED",
      path: "$.runtimeAbi",
    }));
  });

  it("admits isolated copies and rehashes every cache hit before authority", async () => {
    const environment = await createEnvironment();
    const fetched = await acquireCppCuteBrowserAssetSet(
      environment.manifest,
      ORIGIN,
      hostFetch(environment.bytesByUrl),
    );
    const cache = new MemoryCache();
    await admitCppCuteBrowserAssetSetToCache(fetched, cache);
    const cached = await loadCppCuteBrowserAssetSetFromCache(environment.manifest, cache);
    expect(cached).toMatchObject({ source: "content-cache", assetCount: fetched.assetCount });

    const first = cache.entries.values().next().value as Uint8Array | undefined;
    if (first === undefined) throw new Error("cache admission missing");
    first[0] = (first[0] ?? 0) ^ 0xff;
    await expectIoError(
      loadCppCuteBrowserAssetSetFromCache(environment.manifest, cache),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-HASH-MISMATCH",
    );
  });

  it("rejects redirects, response length drift, hash drift, and cancellation", async () => {
    const environment = await createEnvironment();
    const firstBytes = environment.bytesByUrl.values().next().value as Uint8Array | undefined;
    if (firstBytes === undefined) throw new Error("fixture assets missing");
    await expectIoError(
      acquireCppCuteBrowserAssetSet(
        environment.manifest,
        ORIGIN,
        async (url) => response(`${url}.redirected`, firstBytes, { redirected: true }),
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-REDIRECT",
    );
    await expectIoError(
      acquireCppCuteBrowserAssetSet(
        environment.manifest,
        ORIGIN,
        async (url) => response(url, firstBytes, { contentLength: String(firstBytes.byteLength + 1) }),
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-LENGTH-MISMATCH",
    );
    const corrupted = new Map(environment.bytesByUrl);
    const firstUrl = corrupted.keys().next().value as string | undefined;
    if (firstUrl === undefined) throw new Error("fixture assets missing");
    const firstAsset = corrupted.get(firstUrl);
    if (firstAsset === undefined) throw new Error("fixture asset bytes missing");
    const wrong = new Uint8Array(firstAsset);
    wrong[0] = (wrong[0] ?? 0) ^ 0xff;
    corrupted.set(firstUrl, wrong);
    await expectIoError(
      acquireCppCuteBrowserAssetSet(environment.manifest, ORIGIN, hostFetch(corrupted)),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-HASH-MISMATCH",
    );

    const controller = new AbortController();
    controller.abort();
    await expectIoError(
      acquireCppCuteBrowserAssetSet(environment.manifest, ORIGIN, hostFetch(environment.bytesByUrl), {
        signal: controller.signal,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CANCELLED",
      "$.signal",
    );
  });

  it("cancels hung fetch and cache adapters without minting authority", async () => {
    const environment = await createEnvironment();

    const fetchController = new AbortController();
    const fetchPending = acquireCppCuteBrowserAssetSet(
      environment.manifest,
      ORIGIN,
      async () => new Promise<Response>(() => undefined),
      { signal: fetchController.signal },
    );
    fetchController.abort();
    await expectIoError(fetchPending, "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CANCELLED", "$.signal");

    const cacheReadController = new AbortController();
    const cacheReadPending = loadCppCuteBrowserAssetSetFromCache(environment.manifest, {
      get: async () => new Promise<never>(() => undefined),
      put: async () => undefined,
    }, { signal: cacheReadController.signal });
    cacheReadController.abort();
    await expectIoError(cacheReadPending, "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CANCELLED", "$.signal");

    const fetched = await acquireCppCuteBrowserAssetSet(
      environment.manifest,
      ORIGIN,
      hostFetch(environment.bytesByUrl),
    );
    const cacheWriteController = new AbortController();
    const cacheWritePending = admitCppCuteBrowserAssetSetToCache(fetched, {
      get: async () => undefined,
      put: async () => new Promise<never>(() => undefined),
    }, { signal: cacheWriteController.signal });
    cacheWriteController.abort();
    await expectIoError(cacheWritePending, "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CANCELLED", "$.signal");
  });

  it("returns cancellation without awaiting hostile stream cleanup or shadowed signal methods", async () => {
    const environment = await createEnvironment();
    const firstBytes = environment.bytesByUrl.values().next().value as Uint8Array | undefined;
    if (firstBytes === undefined) throw new Error("fixture assets missing");
    let cancelCalls = 0;
    let markReadStarted: (() => void) | undefined;
    const readStarted = new Promise<void>((resolve) => { markReadStarted = resolve; });
    const body = new ReadableStream<Uint8Array>({
      pull: async () => {
        markReadStarted?.();
        return new Promise<never>(() => undefined);
      },
      cancel: async () => {
        cancelCalls += 1;
        return new Promise<never>(() => undefined);
      },
    }, { highWaterMark: 0 });
    const controller = new AbortController();
    Object.defineProperty(controller.signal, "addEventListener", {
      configurable: true,
      value: (): never => { throw new Error("shadowed addEventListener must not run"); },
    });
    Object.defineProperty(controller.signal, "removeEventListener", {
      configurable: true,
      value: (): never => { throw new Error("shadowed removeEventListener must not run"); },
    });
    const pending = acquireCppCuteBrowserAssetSet(
      environment.manifest,
      ORIGIN,
      async (url) => streamResponse(url, body, { contentLength: String(firstBytes.byteLength) }),
      { signal: controller.signal },
    );
    await readStarted;
    controller.abort();
    await expectIoError(pending, "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CANCELLED", "$.signal");
    expect(cancelCalls).toBe(1);
  });

  it("cancels every terminal response body and snapshots hostile adapter properties once", async () => {
    const environment = await createEnvironment();
    const firstUrl = environment.bytesByUrl.keys().next().value as string | undefined;
    const firstBytes = environment.bytesByUrl.values().next().value as Uint8Array | undefined;
    if (firstUrl === undefined || firstBytes === undefined) throw new Error("fixture assets missing");

    let overflowCancels = 0;
    const overflowBody = new ReadableStream<Uint8Array>({
      start: (controller) => controller.enqueue(new Uint8Array(firstBytes.byteLength + 1)),
      cancel: () => { overflowCancels += 1; },
    });
    await expectIoError(
      acquireCppCuteBrowserAssetSet(
        environment.manifest,
        ORIGIN,
        async (url) => streamResponse(url, overflowBody, { contentLength: String(firstBytes.byteLength) }),
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-LENGTH-MISMATCH",
    );
    expect(overflowCancels).toBe(1);

    let statusCancels = 0;
    const statusBody = new ReadableStream<Uint8Array>({ cancel: () => { statusCancels += 1; } });
    await expectIoError(
      acquireCppCuteBrowserAssetSet(
        environment.manifest,
        ORIGIN,
        async (url) => streamResponse(url, statusBody, { status: 503 }),
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-FETCH-FAILED",
    );
    expect(statusCancels).toBe(1);

    const hostile = response(firstUrl, firstBytes);
    const stableBody = hostile.body;
    if (stableBody === null) throw new Error("fixture response body missing");
    let bodyReads = 0;
    Object.defineProperty(hostile, "body", {
      configurable: true,
      get: () => {
        bodyReads += 1;
        if (bodyReads > 1) throw new Error("response body read twice");
        return stableBody;
      },
    });
    Object.defineProperty(stableBody, "getReader", {
      configurable: true,
      value: (): never => { throw new Error("shadowed getReader must not run"); },
    });
    const assetSet = await acquireCppCuteBrowserAssetSet(
      environment.manifest,
      ORIGIN,
      async (url) => url === firstUrl
        ? hostile
        : response(url, environment.bytesByUrl.get(url) ?? Uint8Array.of(), {
          status: environment.bytesByUrl.has(url) ? 200 : 404,
        }),
    );
    expect(assetSet.assetCount).toBe(environment.bytesByUrl.size);
    expect(bodyReads).toBe(1);

    const hostileCache = Object.create(null) as CppCuteBrowserContentCache;
    Object.defineProperty(hostileCache, "get", {
      get: (): never => { throw new Error("hostile cache getter"); },
    });
    Object.defineProperty(hostileCache, "put", { value: async () => undefined });
    await expectIoError(
      loadCppCuteBrowserAssetSetFromCache(environment.manifest, hostileCache),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-INVALID",
      "$.cache",
    );
  });

  it("preserves pack-verifier cancellation and accounts every resident pack copy", async () => {
    const environment = await createEnvironment();
    const assetSet = await acquireCppCuteBrowserAssetSet(
      environment.manifest,
      ORIGIN,
      hostFetch(environment.bytesByUrl),
    );
    const controller = new AbortController();
    const pending = installCppCuteBrowserVfs(assetSet, { signal: controller.signal });
    controller.abort();
    await expectIoError(pending, "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CANCELLED", "$.signal");

    const oneCopyEnvironment = await createEnvironment({ retainedPackCopies: 1 });
    const oneCopyAssetSet = await acquireCppCuteBrowserAssetSet(
      oneCopyEnvironment.manifest,
      ORIGIN,
      hostFetch(oneCopyEnvironment.bytesByUrl),
    );
    await expectIoError(
      installCppCuteBrowserVfs(oneCopyAssetSet),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-RESOURCE-LIMIT",
      "$.packs",
    );
  });

  it("disposes late fetch responses and preserves cancellation through hostile response inspection", async () => {
    const environment = await createEnvironment();
    const firstBytes = environment.bytesByUrl.values().next().value as Uint8Array | undefined;
    if (firstBytes === undefined) throw new Error("fixture assets missing");

    let resolveLate: ((value: Response) => void) | undefined;
    const lateController = new AbortController();
    const latePending = acquireCppCuteBrowserAssetSet(
      environment.manifest,
      ORIGIN,
      async () => new Promise<Response>((resolve) => { resolveLate = resolve; }),
      { signal: lateController.signal },
    );
    lateController.abort();
    await expectIoError(latePending, "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CANCELLED", "$.signal");
    let lateCancels = 0;
    const lateBody = new ReadableStream<Uint8Array>({ cancel: () => { lateCancels += 1; } });
    resolveLate?.(streamResponse(`${ORIGIN}/assets/late`, lateBody, {
      contentLength: String(firstBytes.byteLength),
    }));
    await Promise.resolve();
    await Promise.resolve();
    expect(lateCancels).toBe(1);

    const base = response(`${ORIGIN}/assets/proxy`, firstBytes);
    const hostileProxy = new Proxy(base, {
      get: (target, property, receiver) => property === "then"
        ? undefined
        : Reflect.get(target, property, receiver),
      getPrototypeOf: (): never => { throw new Error("hostile response prototype"); },
    });
    await expectIoError(
      acquireCppCuteBrowserAssetSet(
        environment.manifest,
        ORIGIN,
        () => Promise.resolve(hostileProxy),
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-FETCH-FAILED",
    );

    const inspectionController = new AbortController();
    const abortingResponse = response(`${ORIGIN}/assets/aborting`, firstBytes);
    Object.defineProperty(abortingResponse, "redirected", {
      configurable: true,
      get: () => {
        inspectionController.abort();
        return true;
      },
    });
    await expectIoError(
      acquireCppCuteBrowserAssetSet(
        environment.manifest,
        ORIGIN,
        () => Promise.resolve(abortingResponse),
        { signal: inspectionController.signal },
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CANCELLED",
      "$.signal",
    );

    const responseBrandController = new AbortController();
    const nonmatchingResponse = new Proxy(Object.create(null) as Response, {
      getPrototypeOf: () => {
        responseBrandController.abort();
        return null;
      },
    });
    await expectIoError(
      acquireCppCuteBrowserAssetSet(
        environment.manifest,
        ORIGIN,
        () => Promise.resolve(nonmatchingResponse),
        { signal: responseBrandController.signal },
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CANCELLED",
      "$.signal",
    );

    let fieldBrandCancels = 0;
    const fieldBrandBody = new ReadableStream<Uint8Array>({
      cancel: () => { fieldBrandCancels += 1; },
    });
    const fieldBrandController = new AbortController();
    const fieldBrandResponse = streamResponse(`${ORIGIN}/assets/field-brand`, fieldBrandBody, {
      contentLength: String(firstBytes.byteLength),
    });
    const nonmatchingHeaders = new Proxy(new Headers(), {
      getPrototypeOf: () => {
        fieldBrandController.abort();
        return null;
      },
    });
    Object.defineProperty(fieldBrandResponse, "headers", { configurable: true, value: nonmatchingHeaders });
    await expectIoError(
      acquireCppCuteBrowserAssetSet(
        environment.manifest,
        ORIGIN,
        () => Promise.resolve(fieldBrandResponse),
        { signal: fieldBrandController.signal },
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-CANCELLED",
      "$.signal",
    );
    expect(fieldBrandCancels).toBe(1);
  });

  it("rejects cross-pack regular-file collisions after exact mount roots are applied", async () => {
    const environment = await createEnvironment({ collision: true });
    const assetSet = await acquireCppCuteBrowserAssetSet(
      environment.manifest,
      ORIGIN,
      hostFetch(environment.bytesByUrl),
    );
    await expectIoError(
      installCppCuteBrowserVfs(assetSet),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSET-IO-MOUNT-COLLISION",
    );
  });
});

function wire(value: number): WireU64 {
  return String(value) as WireU64;
}
