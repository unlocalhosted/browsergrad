import { canonicalJsonBytes, sha256Hex, type JsonValue } from
  "@unlocalhosted/browsergrad-semantic-core/schema";
import { beforeEach, describe, expect, it, vi } from "vitest";

const authorities = vi.hoisted(() => ({
  manifests: new WeakMap<object, unknown>(),
  assetSets: new WeakMap<object, unknown>(),
  runtimeAbiAssets: new WeakMap<object, unknown>(),
  runtimeAbis: new WeakMap<object, unknown>(),
  observed: new WeakMap<object, { readonly inspection: object; readonly record: object }>(),
}));

function required<T>(map: WeakMap<object, T>, value: object, label: string): T {
  const record = map.get(value);
  if (record === undefined) throw new Error(`unregistered ${label}`);
  return record;
}

vi.mock("../../src/cpp_cute_browser_asset_installation.js", () => ({
  unwrapVerifiedCppCuteBrowserAssetSet: (value: object) =>
    required(authorities.assetSets, value, "asset set"),
  unwrapVerifiedCppCuteBrowserRuntimeAbiAsset: (value: object) =>
    required(authorities.runtimeAbiAssets, value, "runtime ABI asset"),
}));

vi.mock("../../src/cpp_cute_browser_assets.js", () => ({
  unwrapPreparedCppCuteBrowserAssetManifest: (value: object) =>
    required(authorities.manifests, value, "asset manifest"),
}));

vi.mock("../../src/cpp_cute_browser_runtime_abi.js", () => ({
  unwrapPreparedCppCuteBrowserRuntimeAbiManifest: (value: object) =>
    required(authorities.runtimeAbis, value, "runtime ABI"),
}));

vi.mock("../../src/cpp_cute_browser_wasm_verifier_controller.js", () => ({
  inspectObservedCppCuteBrowserPackageWasmConformance: (value: object) =>
    required(authorities.observed, value, "observed verifier authority").inspection,
  unwrapObservedCppCuteBrowserPackageWasmConformance: (value: object) =>
    required(authorities.observed, value, "observed verifier authority").record,
}));

import { CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_ID } from
  "../../src/cpp_cute_browser_wasm_verifier_bundle.js";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_BYTE_LENGTH,
  CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256,
} from "../../src/resources/cpp_cute_browser_wasm_verifier_bundle_v1.js";
import {
  canonicalCppCuteBrowserWasmVerifierEvidenceBytes,
  decodeCppCuteBrowserWasmVerifierEvidence,
  prepareCppCuteBrowserWasmVerifierEvidence,
  unwrapPreparedCppCuteBrowserWasmVerifierEvidence,
  type CppCuteBrowserWasmVerifierEvidenceBindingInput,
} from "../../src/cpp_cute_browser_wasm_verifier_evidence.js";

const HASH = "a".repeat(64);

interface Fixture {
  readonly binding: CppCuteBrowserWasmVerifierEvidenceBindingInput;
  readonly observed: object;
}

beforeEach(() => {
  authorities.manifests = new WeakMap();
  authorities.assetSets = new WeakMap();
  authorities.runtimeAbiAssets = new WeakMap();
  authorities.runtimeAbis = new WeakMap();
  authorities.observed = new WeakMap();
});

describe("canonical package Wasm verifier evidence", () => {
  it("binds exact host authority and reconstructs only derivative Worker evidence", async () => {
    const fixture = createFixture();
    const host = await prepareCppCuteBrowserWasmVerifierEvidence(
      fixture.observed as never,
      fixture.binding,
    );
    expect(host).toMatchObject({
      authority: "host-observed-verifier-evidence-region-binding",
      sourceHostVerifierExecutionReported: true,
      hostVerifierExecutionLocallyObserved: false,
      workerLocalVerifierExecutionObserved: false,
      productionConformanceAuthorityMinted: false,
      releaseReady: false,
    });
    const bytes = canonicalCppCuteBrowserWasmVerifierEvidenceBytes(host);
    expect(bytes.byteLength).toBe(host.regionByteLength);
    expect(await sha256Hex(bytes)).toBe(host.regionSha256);

    const worker = await decodeCppCuteBrowserWasmVerifierEvidence(
      bytes,
      fixture.binding,
      host.regionSha256,
    );
    expect(worker).toMatchObject({
      authority: "worker-reconstructed-verifier-evidence-region-binding",
      sourceEvidenceId: host.sourceEvidenceId,
      regionSha256: host.regionSha256,
      sourceHostVerifierExecutionReported: true,
      hostVerifierExecutionLocallyObserved: false,
      workerLocalVerifierExecutionObserved: false,
      productionConformanceAuthorityMinted: false,
      releaseReady: false,
    });
    const record = unwrapPreparedCppCuteBrowserWasmVerifierEvidence(worker);
    expect(record.sourceObservedConformance).toBeNull();
    expect(record.workerReconstructed).toBe(true);
    expect(record.productionAuthority).toBe(false);
    expect(() => unwrapPreparedCppCuteBrowserWasmVerifierEvidence({ ...worker } as never))
      .toThrow(/opaque prepared verifier evidence/u);
  });

  it("rejects observed copies, forgeries, and exact-authority cross bindings", async () => {
    const first = createFixture();
    await expect(prepareCppCuteBrowserWasmVerifierEvidence(
      { ...first.observed } as never,
      first.binding,
    )).rejects.toThrow(/unregistered observed verifier authority/u);

    const second = createFixture();
    await expect(prepareCppCuteBrowserWasmVerifierEvidence(
      first.observed as never,
      second.binding,
    )).rejects.toThrow(/different exact asset authorities/u);
  });

  it("rejects every canonical evidence-field mutation against the host region binding", async () => {
    const fixture = createFixture();
    const host = await prepareCppCuteBrowserWasmVerifierEvidence(
      fixture.observed as never,
      fixture.binding,
    );
    const value = JSON.parse(
      new TextDecoder().decode(canonicalCppCuteBrowserWasmVerifierEvidenceBytes(host)),
    ) as Record<string, JsonValue>;
    const mutations: readonly [string, (copy: Record<string, JsonValue>) => void][] = [
      ["schema", (copy) => { copy["schema"] = "browsergrad.invalid"; }],
      ["version.major", (copy) => { (copy["version"] as Record<string, JsonValue>)["major"] = 2; }],
      ["version.minor", (copy) => { (copy["version"] as Record<string, JsonValue>)["minor"] = 1; }],
      ...[
        "sourceEvidenceId", "verifierBundleId", "verifierRequestId",
        "verifierInvocationNonceSha256", "verifierModuleSha256", "assetManifestId",
        "assetManifestSha256", "assetSetSha256", "wasmAssetId", "wasmSha256",
        "runtimeAbiManifestId", "runtimeAbiContractSha256", "runtimeAbiResourceSha256",
        "observedProjectionSha256", "reportSha256", "acceptedTerminalMessages",
      ].map((key) => [key, (copy: Record<string, JsonValue>) => {
        copy[key] = key.endsWith("Sha256") ? "0".repeat(64) : `${String(copy[key])}-mutated`;
      }] as [string, (copy: Record<string, JsonValue>) => void]),
      ...["verifierModuleByteLength", "wasmByteLength", "reportByteLength"].map(
        (key) => [key, (copy: Record<string, JsonValue>) => { copy[key] = "1"; }] as
          [string, (copy: Record<string, JsonValue>) => void],
      ),
      ...[
        "verifierWorkerExecutionObserved", "rawWasmVerified",
        "exactInterfaceConformanceObserved", "packageOwnedVerifier",
        "sourceProductionConformanceAuthorityMinted", "compilerWorkerExecutionObserved",
        "loweringAuthorityMinted", "releaseReady",
      ].map((key) => [key, (copy: Record<string, JsonValue>) => {
        copy[key] = !(copy[key] as boolean);
      }] as [string, (copy: Record<string, JsonValue>) => void]),
    ];

    for (const [label, mutate] of mutations) {
      const copy = JSON.parse(JSON.stringify(value)) as Record<string, JsonValue>;
      mutate(copy);
      const bytes = canonicalJsonBytes(copy as JsonValue);
      await expect(decodeCppCuteBrowserWasmVerifierEvidence(
        bytes,
        fixture.binding,
        host.regionSha256,
      ), label).rejects.toThrow();
    }
  });

  it("uses captured constructors and WeakMap intrinsics after import", async () => {
    const fixture = createFixture();
    const host = await prepareCppCuteBrowserWasmVerifierEvidence(
      fixture.observed as never,
      fixture.binding,
    );
    const NativeUint8Array = globalThis.Uint8Array;
    const nativeWeakMapGet = WeakMap.prototype.get;
    let copied: Uint8Array | undefined;
    let workerReconstructed: boolean | undefined;
    try {
      globalThis.Uint8Array = class PoisonedUint8Array {
        constructor() { throw new Error("ambient Uint8Array construction"); }
      } as unknown as Uint8ArrayConstructor;
      WeakMap.prototype.get = function () {
        throw new Error("ambient WeakMap.get");
      };
      copied = canonicalCppCuteBrowserWasmVerifierEvidenceBytes(host);
      workerReconstructed =
        unwrapPreparedCppCuteBrowserWasmVerifierEvidence(host).workerReconstructed;
    } finally {
      globalThis.Uint8Array = NativeUint8Array;
      WeakMap.prototype.get = nativeWeakMapGet;
    }
    expect(copied).toBeInstanceOf(NativeUint8Array);
    expect(workerReconstructed).toBe(false);
  });
});

function createFixture(): Fixture {
  const assetManifest = Object.freeze({
    manifestId: `bg.cpp.browser-assets.sha256.${HASH}`,
    manifestSha256: "b".repeat(64),
    assetSetSha256: "c".repeat(64),
  });
  const assetSet = Object.freeze({ assetSetSha256: assetManifest.assetSetSha256 });
  const runtimeAbi = Object.freeze({
    manifestId: `bg.cpp.browser-runtime-abi.sha256.${"d".repeat(64)}`,
    contractSha256: "e".repeat(64),
    resourceSha256: "f".repeat(64),
  });
  const runtimeAbiAsset = Object.freeze({ runtimeAbiManifestId: runtimeAbi.manifestId });
  const observed = Object.freeze({ evidenceId: "opaque-host-verifier" });
  const clangAsset = Object.freeze({
    assetId: "clang-wasm",
    kind: "clang-extractor-wasm",
    sha256: "1".repeat(64),
    byteLength: "8",
  });
  authorities.manifests.set(assetManifest, {
    manifest: { body: { assets: [clangAsset] } },
  });
  authorities.assetSets.set(assetSet, { manifest: assetManifest });
  authorities.runtimeAbiAssets.set(runtimeAbiAsset, { assetSet, runtimeAbi });
  authorities.runtimeAbis.set(runtimeAbi, { manifest: {} });
  const inspection = Object.freeze({
    evidenceId: `bg.cpp.browser-wasm-verifier-conformance.sha256.${"2".repeat(64)}`,
    requestId: `bg.cpp.browser-wasm-verifier-request.sha256.${"3".repeat(64)}`,
    invocationNonceSha256: "4".repeat(64),
    verifierBundleId: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_ID,
    verifierModuleSha256: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256,
    verifierModuleByteLength: CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_BYTE_LENGTH,
    assetManifestId: assetManifest.manifestId,
    assetSetSha256: assetManifest.assetSetSha256,
    wasmAssetId: clangAsset.assetId,
    wasmSha256: clangAsset.sha256,
    wasmByteLength: 8,
    runtimeAbiManifestId: runtimeAbi.manifestId,
    runtimeAbiContractSha256: runtimeAbi.contractSha256,
    runtimeAbiResourceSha256: runtimeAbi.resourceSha256,
    observedProjectionSha256: "5".repeat(64),
    reportSha256: "6".repeat(64),
    reportByteLength: 512,
    releaseReady: false,
  });
  authorities.observed.set(observed, {
    inspection,
    record: { assetSet, assetManifest, runtimeAbiAsset, runtimeAbi },
  });
  return {
    observed,
    binding: {
      assetSet: assetSet as never,
      assetManifest: assetManifest as never,
      runtimeAbiAsset: runtimeAbiAsset as never,
    },
  };
}
