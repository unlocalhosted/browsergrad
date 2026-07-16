import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT,
  CppCuteBrowserAssetManifestError,
  canonicalCppCuteBrowserAssetManifestBytes,
  decodeCppCuteBrowserAssetManifest,
  deriveCppCuteBrowserAssetManifestId,
  deriveCppCuteBrowserAssetSetSha256,
  prepareCppCuteBrowserAssetManifest,
  unwrapPreparedCppCuteBrowserAssetManifest,
  type CppCuteBrowserAssetManifestBodyV1,
  type PreparedCppCuteBrowserAssetManifest,
} from "../../src/cpp_cute_browser_assets.js";
import { prepareCppCuteFrontendProfile } from "../../src/cpp_cute_frontend_profile.js";
import {
  cloneCppCuteBrowserAssetInput,
  createCppCuteBrowserAssetFixture,
} from "./support/cpp_cute_browser_asset_fixtures.js";
import {
  createCppCuteBrowserProfileInput,
  createCppCuteProfileInput,
} from "./support/cpp_cute_frontend_fixtures.js";

function expectAssetError(
  operation: Promise<unknown>,
  code: CppCuteBrowserAssetManifestError["code"],
  path: string,
): Promise<void> {
  return expect(operation).rejects.toMatchObject({ code, path });
}

function body(input: Record<string, unknown>): Record<string, unknown> {
  return input["body"] as Record<string, unknown>;
}

function assets(input: Record<string, unknown>): Array<Record<string, unknown>> {
  return body(input)["assets"] as Array<Record<string, unknown>>;
}

async function resign(input: Record<string, unknown>): Promise<void> {
  input["manifestId"] = await deriveCppCuteBrowserAssetManifestId(
    body(input) as unknown as CppCuteBrowserAssetManifestBodyV1,
  );
}

async function rederiveAssetSetAndResign(input: Record<string, unknown>): Promise<void> {
  body(input)["assetSetSha256"] = await deriveCppCuteBrowserAssetSetSha256(
    body(input) as unknown as CppCuteBrowserAssetManifestBodyV1,
  );
  await resign(input);
}

async function rebindProfileToMutatedAssetSet(input: Record<string, unknown>) {
  await rederiveAssetSetAndResign(input);
  const profile = await prepareCppCuteFrontendProfile(createCppCuteBrowserProfileInput({
    assetSetSha256: String(body(input)["assetSetSha256"]),
  }));
  body(input)["profileHash"] = profile.profileHash;
  await resign(input);
  return profile;
}

describe("C++/CuTe browser-local asset manifest", () => {
  it("prepares deterministic immutable profile-bound authority", async () => {
    const fixture = await createCppCuteBrowserAssetFixture();
    const first = await prepareCppCuteBrowserAssetManifest(fixture.input, fixture.profile);
    const second = await prepareCppCuteBrowserAssetManifest(structuredClone(fixture.input), fixture.profile);

    expect(first).toEqual(second);
    expect(first.manifestId).toBe(fixture.input.manifestId);
    expect(fixture.input.body.assetSetSha256).toBe(first.assetSetSha256);
    expect(fixture.input.manifestId).toBe(
      "bg.cpp.browser-assets.sha256.dc840fff1f86760f52e845223987f71a484343cbe54b34af21c85ed79bea6953",
    );
    expect(first.assetSetSha256).toBe("44520f6d9f9ef9aa403a3c281142b463a66adc874d3b914d97c04f9b02af2829");
    expect(first.assetCount).toBe(7);
    expect(first.manifestSha256).toBe("b3fe988d02bb05221efbe0ac8631e2299ea4b7eae7157c2ea07e536a8d4e6ccb");
    expect(first.manifestByteLength).toBe("9315");
    expect(Object.isFrozen(first)).toBe(true);
    const record = unwrapPreparedCppCuteBrowserAssetManifest(first);
    expect(record.profile).toBe(fixture.profile);
    expect(Object.isFrozen(record.manifest)).toBe(true);
    expect(Object.isFrozen(record.manifest.body.assets)).toBe(true);
    expect(first).not.toHaveProperty("fetched");
    expect(first).not.toHaveProperty("executed");

    const bytes = canonicalCppCuteBrowserAssetManifestBytes(first);
    const expectedFirstByte = bytes[0];
    bytes[0] = 0;
    expect(canonicalCppCuteBrowserAssetManifestBytes(first)[0]).toBe(expectedFirstByte);
  });

  it("rejects ahead-of-time profiles", async () => {
    const fixture = await createCppCuteBrowserAssetFixture();
    const aot = await prepareCppCuteFrontendProfile(createCppCuteProfileInput());
    await expect(prepareCppCuteBrowserAssetManifest(fixture.input, aot)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      path: "$.deployment.mode",
    });
  });

  it("decodes only exact canonical JSON bytes", async () => {
    const fixture = await createCppCuteBrowserAssetFixture();
    const prepared = await decodeCppCuteBrowserAssetManifest(fixture.bytes, fixture.profile);
    expect(canonicalCppCuteBrowserAssetManifestBytes(prepared)).toEqual(fixture.bytes);

    const trailingWhitespace = new Uint8Array(fixture.bytes.byteLength + 1);
    trailingWhitespace.set(fixture.bytes);
    trailingWhitespace[trailingWhitespace.length - 1] = 0x0a;
    await expectAssetError(
      decodeCppCuteBrowserAssetManifest(trailingWhitespace, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-NONCANONICAL-BYTES",
      "$bytes",
    );
    await expectAssetError(
      decodeCppCuteBrowserAssetManifest(new TextEncoder().encode('{"schema":1,"schema":2}'), fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$bytes",
    );
    await expectAssetError(
      decodeCppCuteBrowserAssetManifest(Uint8Array.of(0xff), fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$bytes",
    );
  });

  it("requires exact closed asset kinds and cardinalities", async () => {
    const fixture = await createCppCuteBrowserAssetFixture();
    const duplicateWasm = cloneCppCuteBrowserAssetInput(fixture.input);
    const wasm = structuredClone(assets(duplicateWasm).find((asset) => asset["kind"] === "clang-extractor-wasm"));
    if (wasm === undefined) throw new Error("fixture lost wasm");
    wasm["assetId"] = "clang-wasm.second";
    wasm["url"] = "/browsergrad/cpp-cute/clang-extractor.second.wasm";
    assets(duplicateWasm).push(wasm);
    assets(duplicateWasm).sort((left, right) => String(left["assetId"]).localeCompare(String(right["assetId"])));
    await resign(duplicateWasm);
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(duplicateWasm, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$.body.assets",
    );

    const packageOwnedWorker = cloneCppCuteBrowserAssetInput(fixture.input);
    const first = assets(packageOwnedWorker)[0]!;
    first["kind"] = "loader-javascript";
    first["mediaType"] = "text/javascript";
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(packageOwnedWorker, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$.body.assets[0].kind",
    );

    const wrongMedia = cloneCppCuteBrowserAssetInput(fixture.input);
    const wrongWasm = assets(wrongMedia).find((asset) => asset["kind"] === "clang-extractor-wasm");
    if (wrongWasm === undefined) throw new Error("fixture lost wasm");
    wrongWasm["mediaType"] = "application/octet-stream";
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(wrongMedia, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$.body.assets[1].mediaType",
    );

    const extraField = cloneCppCuteBrowserAssetInput(fixture.input);
    assets(extraField)[0]!["integrity"] = "sha256-anything";
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(extraField, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$.body.assets[0]",
    );

    for (const [field, value] of [
      ["mediaType", "application/vnd.browsergrad.vfs-pack.v1+tar"],
      ["compression", "gzip"],
    ] as const) {
      const legacyArchive = cloneCppCuteBrowserAssetInput(fixture.input);
      const pack = assets(legacyArchive).find((asset) => asset["kind"] === "dependency-header-pack");
      if (pack === undefined) throw new Error("fixture lost dependency pack");
      pack[field] = value;
      await expectAssetError(
        prepareCppCuteBrowserAssetManifest(legacyArchive, fixture.profile),
        "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
        `$.body.assets[3].${field}`,
      );
    }
  });

  it("accepts only unique normalized root-relative URLs", async () => {
    const fixture = await createCppCuteBrowserAssetFixture();
    for (const badUrl of [
      "https://example.test/browsergrad/asset",
      "//evil.test/asset",
      "browsergrad/asset",
      "/browsergrad/../private/asset",
      "/browsergrad/asset?next=https://evil.test",
      "/browsergrad/%2e%2e/private",
    ]) {
      const input = cloneCppCuteBrowserAssetInput(fixture.input);
      assets(input)[0]!["url"] = badUrl;
      await expectAssetError(
        prepareCppCuteBrowserAssetManifest(input, fixture.profile),
        "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
        "$.body.assets[0].url",
      );
    }

    const duplicate = cloneCppCuteBrowserAssetInput(fixture.input);
    assets(duplicate)[1]!["url"] = assets(duplicate)[0]!["url"];
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(duplicate, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$.body.assets",
    );
  });

  it("requires sorted identities and accepts root virtual mount", async () => {
    const fixture = await createCppCuteBrowserAssetFixture();
    const unsortedAssets = cloneCppCuteBrowserAssetInput(fixture.input);
    assets(unsortedAssets).reverse();
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(unsortedAssets, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$.body.assets",
    );

    const dependencies = cloneCppCuteBrowserAssetInput(fixture.input);
    (body(dependencies)["dependencyIds"] as string[]).reverse();
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(dependencies, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$.body.dependencyIds",
    );

    const base = createCppCuteBrowserProfileInput();
    const roots = structuredClone(base.virtualFileSystem.includeRoots);
    const compilerRoot = roots.find((root) => root.owner.kind === "compiler-resource-directory");
    if (compilerRoot === undefined) throw new Error("fixture lost compiler root");
    (compilerRoot as { virtualPath: string }).virtualPath = "/";
    const rootFixture = await createCppCuteBrowserAssetFixture({ profile: { includeRoots: roots } });
    await expect(prepareCppCuteBrowserAssetManifest(rootFixture.input, rootFixture.profile)).resolves.toMatchObject({
      assetCount: 7,
    });
  });

  it("binds exact source ABI, extractor, adapter, resource, and dependency content", async () => {
    const fixture = await createCppCuteBrowserAssetFixture();
    const profileHash = cloneCppCuteBrowserAssetInput(fixture.input);
    body(profileHash)["profileHash"] = "0".repeat(64);
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(profileHash, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$.body.profileHash",
    );

    const sourceAbi = cloneCppCuteBrowserAssetInput(fixture.input);
    const sourceAbiValue = body(sourceAbi)["sourceAbi"] as Record<string, unknown>;
    const sourceAbiTarget = sourceAbiValue["target"] as Record<string, unknown>;
    (sourceAbiTarget["device"] as Record<string, unknown>)["pointerBits"] = 32;
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(sourceAbi, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$.body.sourceAbi",
    );

    for (const field of ["sourceAbiSha256", "sha256"] as const) {
      const wasmDrift = cloneCppCuteBrowserAssetInput(fixture.input);
      const wasm = assets(wasmDrift).find((asset) => asset["kind"] === "clang-extractor-wasm");
      if (wasm === undefined) throw new Error("fixture lost wasm");
      wasm[field] = "0".repeat(64);
      const reboundProfile = await rebindProfileToMutatedAssetSet(wasmDrift);
      await expectAssetError(
        prepareCppCuteBrowserAssetManifest(wasmDrift, reboundProfile),
        "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-HASH-MISMATCH",
        "$.body.assets",
      );
    }

    const adapterHash = cloneCppCuteBrowserAssetInput(fixture.input);
    const adapter = assets(adapterHash).find((asset) => asset["kind"] === "semantic-adapter-manifest");
    if (adapter === undefined) throw new Error("fixture lost adapter");
    adapter["sha256"] = "0".repeat(64);
    const adapterProfile = await rebindProfileToMutatedAssetSet(adapterHash);
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(adapterHash, adapterProfile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-HASH-MISMATCH",
      "$.body.assets",
    );

    for (const kind of ["compiler-resource-pack", "dependency-header-pack"]) {
      const packHash = cloneCppCuteBrowserAssetInput(fixture.input);
      const pack = assets(packHash).find((asset) => asset["kind"] === kind);
      if (pack === undefined) throw new Error(`fixture lost ${kind}`);
      pack["contentSetSha256"] = "0".repeat(64);
      const packProfile = await rebindProfileToMutatedAssetSet(packHash);
      await expectAssetError(
        prepareCppCuteBrowserAssetManifest(packHash, packProfile),
        "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-HASH-MISMATCH",
        "$.body.assets",
      );
    }
  });

  it("rejects self-resigned URL and provenance drift against profile asset-set lock", async () => {
    const fixture = await createCppCuteBrowserAssetFixture();
    const wasm = cloneCppCuteBrowserAssetInput(fixture.input);
    const wasmAsset = assets(wasm).find((asset) => asset["kind"] === "clang-extractor-wasm");
    if (wasmAsset === undefined) throw new Error("fixture lost wasm");
    wasmAsset["sha256"] = "0".repeat(64);
    await rederiveAssetSetAndResign(wasm);
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(wasm, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-HASH-MISMATCH",
      "$.body.assetSetSha256",
    );

    const url = cloneCppCuteBrowserAssetInput(fixture.input);
    assets(url)[0]!["url"] = "/browsergrad/cpp-cute/adapter-renamed.json";
    await rederiveAssetSetAndResign(url);
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(url, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-HASH-MISMATCH",
      "$.body.assetSetSha256",
    );

    const provenance = cloneCppCuteBrowserAssetInput(fixture.input);
    const replacement = `bg.build-provenance.sha256.${"a".repeat(64)}`;
    for (const asset of assets(provenance)) asset["buildProvenanceId"] = replacement;
    body(provenance)["buildProvenanceIds"] = [replacement];
    await rederiveAssetSetAndResign(provenance);
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(provenance, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-HASH-MISMATCH",
      "$.body.assetSetSha256",
    );
  });

  it("checks source ABI, asset-set, and manifest hashes", async () => {
    const fixture = await createCppCuteBrowserAssetFixture();
    const sourceAbiHash = cloneCppCuteBrowserAssetInput(fixture.input);
    body(sourceAbiHash)["sourceAbiSha256"] = "0".repeat(64);
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(sourceAbiHash, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-HASH-MISMATCH",
      "$.body.sourceAbiSha256",
    );

    const assetSetHash = cloneCppCuteBrowserAssetInput(fixture.input);
    body(assetSetHash)["assetSetSha256"] = "0".repeat(64);
    await resign(assetSetHash);
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(assetSetHash, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-HASH-MISMATCH",
      "$.body.assetSetSha256",
    );

    const manifestId = cloneCppCuteBrowserAssetInput(fixture.input);
    manifestId["manifestId"] = `bg.cpp.browser-assets.sha256.${"0".repeat(64)}`;
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(manifestId, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-HASH-MISMATCH",
      "$.manifestId",
    );
  });

  it("enforces profile-owned ceilings and exact totals", async () => {
    const perAsset = await createCppCuteBrowserAssetFixture({
      assetLimits: { maxAssetCompressedByteLength: 100 },
    });
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(perAsset.input, perAsset.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-RESOURCE-LIMIT",
      "$.body.assets[0].byteLength",
    );

    const count = await createCppCuteBrowserAssetFixture({ assetLimits: { maxAssets: 6 } });
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(count.input, count.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-RESOURCE-LIMIT",
      "$.body.assets",
    );

    const total = await createCppCuteBrowserAssetFixture({
      assetLimits: {
        maxAssetUnpackedByteLength: 20_000,
        maxTotalUnpackedByteLength: 30_000,
      },
    });
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(total.input, total.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-RESOURCE-LIMIT",
      "$.body.totals.unpackedByteLength",
    );

    const fileContent = await createCppCuteBrowserAssetFixture({
      assetLimits: { maxAssetFileContentByteLength: 100 },
    });
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(fileContent.input, fileContent.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-RESOURCE-LIMIT",
      "$.body.assets[2].fileContentByteLength",
    );

    const totalDrift = cloneCppCuteBrowserAssetInput((await createCppCuteBrowserAssetFixture()).input);
    (body(totalDrift)["totals"] as Record<string, unknown>)["compressedByteLength"] = "1";
    const fixture = await createCppCuteBrowserAssetFixture();
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(totalDrift, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$.body.totals.compressedByteLength",
    );

    const manifestCeilings = cloneCppCuteBrowserAssetInput(fixture.input);
    body(manifestCeilings)["resourceLimits"] = { maxAssets: 1_000_000 };
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(manifestCeilings, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$.body",
    );
  });

  it("rejects accessors, polluted prototypes, and forged authorities without invoking getters", async () => {
    const fixture = await createCppCuteBrowserAssetFixture();
    let calls = 0;
    const accessor = cloneCppCuteBrowserAssetInput(fixture.input);
    Object.defineProperty(accessor, "manifestId", {
      enumerable: true,
      get() {
        calls += 1;
        return fixture.input.manifestId;
      },
    });
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(accessor, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$",
    );
    expect(calls).toBe(0);

    const polluted = cloneCppCuteBrowserAssetInput(fixture.input);
    Object.setPrototypeOf(polluted, { injected: true });
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(polluted, fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$",
    );

    const hostileOptions = Object.create(null) as Record<string, unknown>;
    Object.defineProperty(hostileOptions, "signal", {
      enumerable: true,
      get() {
        calls += 1;
        return undefined;
      },
    });
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(
        fixture.input,
        fixture.profile,
        hostileOptions as unknown as { signal?: AbortSignal },
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$options",
    );
    expect(calls).toBe(0);

    const forged = {
      manifestId: fixture.input.manifestId,
      manifestSha256: "0".repeat(64),
    } as unknown as PreparedCppCuteBrowserAssetManifest;
    expect(() => unwrapPreparedCppCuteBrowserAssetManifest(forged)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-UNVERIFIED", path: "$" }),
    );
    expect(() => canonicalCppCuteBrowserAssetManifestBytes(forged)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-UNVERIFIED", path: "$" }),
    );
  });

  it("rejects non-plain, shared, and oversized byte inputs before decoding", async () => {
    const fixture = await createCppCuteBrowserAssetFixture();
    class BytesSubclass extends Uint8Array {}
    await expectAssetError(
      decodeCppCuteBrowserAssetManifest(new BytesSubclass(fixture.bytes), fixture.profile),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
      "$bytes",
    );
    await expectAssetError(
      decodeCppCuteBrowserAssetManifest(
        new Uint8Array(CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT + 1),
        fixture.profile,
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-RESOURCE-LIMIT",
      "$bytes",
    );
    if (typeof SharedArrayBuffer !== "undefined") {
      await expectAssetError(
        decodeCppCuteBrowserAssetManifest(new Uint8Array(new SharedArrayBuffer(16)), fixture.profile),
        "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID",
        "$bytes",
      );
    }
  });

  it("honors cancellation before parsing and hashing", async () => {
    const fixture = await createCppCuteBrowserAssetFixture();
    const controller = new AbortController();
    controller.abort();
    await expectAssetError(
      prepareCppCuteBrowserAssetManifest(fixture.input, fixture.profile, { signal: controller.signal }),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-CANCELLED",
      "$options.signal",
    );
    await expectAssetError(
      decodeCppCuteBrowserAssetManifest(fixture.bytes, fixture.profile, { signal: controller.signal }),
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-CANCELLED",
      "$options.signal",
    );
  });
});
