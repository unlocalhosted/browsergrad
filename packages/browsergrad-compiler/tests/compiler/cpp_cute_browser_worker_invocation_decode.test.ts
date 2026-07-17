import {
  canonicalJsonBytes,
  encodeWireU64,
  sha256Hex,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it, vi } from "vitest";

const authorities = vi.hoisted(() => ({
  manifests: new WeakMap<object, unknown>(),
  assetSets: new WeakMap<object, unknown>(),
  installations: new WeakMap<object, unknown>(),
  runtimeAbiAssets: new WeakMap<object, unknown>(),
  wasmConformance: new WeakMap<object, unknown>(),
  assetBytes: new WeakMap<object, ReadonlyMap<string, Uint8Array>>(),
}));

vi.mock("../../src/cpp_cute_browser_assets.js", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("../../src/cpp_cute_browser_assets.js")
  >();
  return {
    ...actual,
    unwrapPreparedCppCuteBrowserAssetManifest: (value: object) => {
      const record = authorities.manifests.get(value);
      if (record === undefined) throw new Error("unregistered test manifest authority");
      return record;
    },
  };
});

vi.mock("../../src/cpp_cute_browser_asset_installation.js", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("../../src/cpp_cute_browser_asset_installation.js")
  >();
  return {
    ...actual,
    unwrapVerifiedCppCuteBrowserAssetSet: (value: object) => {
      const record = authorities.assetSets.get(value);
      if (record === undefined) throw new Error("unregistered test asset-set authority");
      return record;
    },
    unwrapVerifiedCppCuteBrowserVfsInstallation: (value: object) => {
      const record = authorities.installations.get(value);
      if (record === undefined) throw new Error("unregistered test VFS authority");
      return record;
    },
    unwrapVerifiedCppCuteBrowserRuntimeAbiAsset: (value: object) => {
      const record = authorities.runtimeAbiAssets.get(value);
      if (record === undefined) throw new Error("unregistered test runtime-ABI authority");
      return record;
    },
    copyVerifiedCppCuteBrowserAssetBytes: (value: object, assetId: string) => {
      const bytes = authorities.assetBytes.get(value)?.get(assetId);
      if (bytes === undefined) throw new Error("unregistered test asset bytes");
      return new Uint8Array(bytes);
    },
  };
});

vi.mock("../../src/cpp_cute_browser_wasm_inspection.js", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("../../src/cpp_cute_browser_wasm_inspection.js")
  >();
  return {
    ...actual,
    unwrapPreparedCppCuteBrowserWasmConformance: (value: object) => {
      const record = authorities.wasmConformance.get(value);
      if (record === undefined) throw new Error("unregistered test Wasm-conformance authority");
      return record;
    },
  };
});

import {
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
  decodeCppCuteBrowserRuntimeAbiManifest,
} from "../../src/cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_BROWSER_WORKER_INVOCATION_BYTE_LIMIT,
  canonicalCppCuteBrowserWorkerInvocationBytes,
  copyCppCuteBrowserWorkerModuleBytes,
  decodeCppCuteBrowserWorkerInvocation,
  prepareCppCuteBrowserWorkerInvocation,
  unwrapPreparedCppCuteBrowserWorkerInvocation,
  type DecodeCppCuteBrowserWorkerInvocationInput,
  type PreparedCppCuteBrowserWorkerInvocation,
} from "../../src/cpp_cute_browser_worker_protocol.js";
import {
  deriveCppCuteFrontendEntryRequestId,
  deriveCppCuteFrontendRequestHash,
  deriveCppCuteFrontendSourceFileId,
  prepareCppCuteFrontendRequest,
  type CppCuteFrontendRequestBodyV1,
  type CppCuteFrontendRequestLimitsV1,
  type CppCuteFrontendRequestSourceFileV1,
  type CppCuteFrontendRequestV1,
} from "../../src/cpp_cute_frontend_request.js";
import {
  prepareCppCuteFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
} from "../../src/cpp_cute_frontend_types.js";
import { createCppCuteBrowserProfileInput } from "./support/cpp_cute_frontend_fixtures.js";

const SOURCE_PATH = "/src/layout.cu";
const SOURCE_BYTES = new TextEncoder().encode("auto layout = make_layout(Int<2>{});");

interface InvocationEnvironment {
  readonly hostInvocation: PreparedCppCuteBrowserWorkerInvocation;
  readonly decodeInput: DecodeCppCuteBrowserWorkerInvocationInput;
}

describe("C++/CuTe Worker invocation byte reconstruction", () => {
  it("strict-decodes canonical bytes through local authorities without Worker claims or module bytes", async () => {
    const environment = await createEnvironment();
    const bytes = canonicalCppCuteBrowserWorkerInvocationBytes(environment.hostInvocation);
    const hostRecord = unwrapPreparedCppCuteBrowserWorkerInvocation(
      environment.hostInvocation,
    );

    const decoded = await decodeCppCuteBrowserWorkerInvocation(
      bytes,
      environment.decodeInput,
    );
    const decodedRecord = unwrapPreparedCppCuteBrowserWorkerInvocation(decoded);

    expect(decoded).not.toBe(environment.hostInvocation);
    expect(decoded).toEqual(environment.hostInvocation);
    expect(decodedRecord.invocation).toEqual(hostRecord.invocation);
    expect(decodedRecord.invocation.invocationNonceSha256)
      .toBe(hostRecord.invocation.invocationNonceSha256);
    expect(canonicalCppCuteBrowserWorkerInvocationBytes(decoded)).toEqual(bytes);
    expect(Object.isFrozen(decoded)).toBe(true);
    expect(decoded).not.toHaveProperty("workerExecutionObserved");
    expect(decoded).not.toHaveProperty("loweringAuthorityMinted");
    expect(() => copyCppCuteBrowserWorkerModuleBytes(decoded)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-UNVERIFIED",
        path: "$.invocation.workerModuleBytes",
      }),
    );
  });

  it("rejects every mutated deterministic identity against reconstructed authorities", async () => {
    const environment = await createEnvironment();
    const canonical = invocationObject(environment.hostInvocation);
    const mutations: Array<(value: Record<string, unknown>) => void> = [
      (value) => { value["profileHash"] = "c".repeat(64); },
      (value) => { value["requestId"] = `bg.cpp.frontend-request.sha256.${"c".repeat(64)}`; },
      (value) => { value["invocationNonceSha256"] = "c".repeat(64); },
      (value) => {
        (value["worker"] as Record<string, unknown>)["moduleSha256"] = "c".repeat(64);
      },
      (value) => {
        value["invocationId"] =
          `bg.cpp.browser-worker-invocation.sha256.${"c".repeat(64)}`;
      },
    ];

    for (const mutate of mutations) {
      const mutated = structuredClone(canonical);
      mutate(mutated);
      await expect(decodeCppCuteBrowserWorkerInvocation(
        canonicalJsonBytes(mutated),
        environment.decodeInput,
      )).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVOCATION-MISMATCH",
        path: "$.invocationBytes",
      });
    }
  });

  it("rejects trailing, unknown, duplicate-key, and noncanonical invocation bytes", async () => {
    const environment = await createEnvironment();
    const canonical = canonicalCppCuteBrowserWorkerInvocationBytes(
      environment.hostInvocation,
    );
    const trailing = new Uint8Array(canonical.byteLength + 1);
    trailing.set(canonical);
    trailing[canonical.byteLength] = 0x20;
    await expect(decodeCppCuteBrowserWorkerInvocation(
      trailing,
      environment.decodeInput,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-NONCANONICAL-BYTES",
      path: "$.invocationBytes",
    });

    const unknown = invocationObject(environment.hostInvocation);
    unknown["workerExecutionObserved"] = true;
    await expect(decodeCppCuteBrowserWorkerInvocation(
      canonicalJsonBytes(unknown),
      environment.decodeInput,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      path: "$.invocation",
    });

    const text = new TextDecoder().decode(canonical);
    const duplicate = new TextEncoder().encode(text.replace(
      '"schema":',
      '"schema":"browsergrad.compiler.cpp-cute.browser-worker-invocation","schema":',
    ));
    await expect(decodeCppCuteBrowserWorkerInvocation(
      duplicate,
      environment.decodeInput,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      path: "$.invocationBytes",
    });

    const firstColon = text.indexOf(":");
    const noncanonical = new TextEncoder().encode(
      `${text.slice(0, firstColon + 1)} ${text.slice(firstColon + 1)}`,
    );
    await expect(decodeCppCuteBrowserWorkerInvocation(
      noncanonical,
      environment.decodeInput,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-NONCANONICAL-BYTES",
      path: "$.invocationBytes",
    });
  });

  it("rejects oversized, shared, proxied, subclassed, detached, and malformed byte views", async () => {
    const environment = await createEnvironment();
    const canonical = canonicalCppCuteBrowserWorkerInvocationBytes(
      environment.hostInvocation,
    );
    await expect(decodeCppCuteBrowserWorkerInvocation(
      new Uint8Array(CPP_CUTE_BROWSER_WORKER_INVOCATION_BYTE_LIMIT + 1),
      environment.decodeInput,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RESOURCE-LIMIT",
      path: "$.invocationBytes",
    });
    if (typeof SharedArrayBuffer !== "undefined") {
      await expect(decodeCppCuteBrowserWorkerInvocation(
        new Uint8Array(new SharedArrayBuffer(canonical.byteLength)),
        environment.decodeInput,
      )).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
        path: "$.invocationBytes",
      });
    }

    await expect(decodeCppCuteBrowserWorkerInvocation(
      new Proxy(canonical, {}),
      environment.decodeInput,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      path: "$.invocationBytes",
    });

    class ByteSubclass extends Uint8Array {}
    await expect(decodeCppCuteBrowserWorkerInvocation(
      new ByteSubclass(canonical),
      environment.decodeInput,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      path: "$.invocationBytes",
    });

    const detached = new Uint8Array(canonical);
    structuredClone(detached.buffer, { transfer: [detached.buffer] });
    await expect(decodeCppCuteBrowserWorkerInvocation(
      detached,
      environment.decodeInput,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      path: "$.invocationBytes",
    });
    await expect(decodeCppCuteBrowserWorkerInvocation(
      Uint8Array.of(0xff, 0xfe, 0xfd),
      environment.decodeInput,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-INVALID",
      path: "$.invocationBytes",
    });
  });
});

async function createEnvironment(): Promise<InvocationEnvironment> {
  const clangBytes = Uint8Array.of(0, 97, 115, 109, 1, 0, 0, 0);
  const workerBytes = new Uint8Array(65_536);
  workerBytes.set(new TextEncoder().encode("browsergrad-worker-invocation-fixture"));
  const clangSha256 = await sha256Hex(clangBytes);
  const workerSha256 = await sha256Hex(workerBytes);
  const profileInput = structuredClone(createCppCuteBrowserProfileInput({
    sourceRoots: ["/src"],
  }));
  (profileInput.toolchain.compiler as { binarySha256: string }).binarySha256 = clangSha256;
  (profileInput.deployment.extractor as { binarySha256: string }).binarySha256 = clangSha256;
  const worker = profileInput.deployment.worker as {
    moduleSha256: string;
    moduleByteLength: number;
  };
  worker.moduleSha256 = workerSha256;
  worker.moduleByteLength = workerBytes.byteLength;
  const profile = await prepareCppCuteFrontendProfile(profileInput);
  const request = await prepareCppCuteFrontendRequest(
    profile,
    await createRequestInput(profile),
    [{ virtualPath: SOURCE_PATH, bytes: SOURCE_BYTES }],
  );
  const runtimeAbi = await decodeCppCuteBrowserRuntimeAbiManifest(
    cppCuteBrowserRuntimeAbiManifestResourceBytes(),
  );
  const hash = "a".repeat(64);
  const assetManifest = Object.freeze({
    manifestId: `bg.cpp.browser-assets.sha256.${hash}`,
    manifestSha256: hash,
    assetSetSha256: profileInput.deployment.assetSetSha256,
  });
  const assetSet = Object.freeze({
    assetSetSha256: profileInput.deployment.assetSetSha256,
  });
  const installation = Object.freeze({
    installationId: `bg.cpp.browser-vfs-installation.sha256.${hash}`,
  });
  const runtimeAbiAsset = Object.freeze({
    runtimeAbiManifestId: runtimeAbi.manifestId,
  });
  const rawWasmConformance = Object.freeze({
    wasmSha256: clangSha256,
    wasmByteLength: clangBytes.byteLength,
    observedProjectionSha256: "b".repeat(64),
    runtimeAbiManifestId: runtimeAbi.manifestId,
    runtimeAbiContractSha256: runtimeAbi.contractSha256,
  });
  authorities.manifests.set(assetManifest, {
    profile,
    manifest: {
      body: {
        assets: [{
          assetId: "clang-wasm",
          kind: "clang-extractor-wasm",
          sha256: clangSha256,
          byteLength: String(clangBytes.byteLength),
        }],
      },
    },
  });
  authorities.assetSets.set(assetSet, { manifest: assetManifest });
  authorities.installations.set(installation, { assetSet, files: [] });
  authorities.runtimeAbiAssets.set(runtimeAbiAsset, { assetSet, runtimeAbi });
  authorities.wasmConformance.set(rawWasmConformance, {});
  authorities.assetBytes.set(assetSet, new Map([["clang-wasm", clangBytes]]));
  const decodeInput = Object.freeze({
    profile,
    assetManifest: assetManifest as never,
    vfsInstallation: installation as never,
    request,
    runtimeAbiAsset: runtimeAbiAsset as never,
    rawWasmConformance: rawWasmConformance as never,
  });
  const hostInvocation = await prepareCppCuteBrowserWorkerInvocation({
    ...decodeInput,
    workerModuleBytes: workerBytes,
  });
  return { hostInvocation, decodeInput };
}

async function createRequestInput(
  profile: PreparedCppCuteFrontendProfile,
): Promise<CppCuteFrontendRequestV1> {
  const descriptor = {
    role: "main-source" as const,
    virtualPath: SOURCE_PATH,
    contentSha256: await sha256Hex(SOURCE_BYTES),
    byteLength: encodeWireU64(BigInt(SOURCE_BYTES.byteLength)),
    includeRootId: null,
  };
  const source: CppCuteFrontendRequestSourceFileV1 = {
    fileId: await deriveCppCuteFrontendSourceFileId(descriptor),
    ...descriptor,
  };
  const entryBody = {
    requestId: `bg.cpp.entry-request.sha256.${"0".repeat(64)}`,
    kind: "layout" as const,
    declarationKind: "variable" as const,
    anchor: {
      virtualPath: SOURCE_PATH,
      beginByte: "0" as WireU64,
      endByte: String(SOURCE_BYTES.byteLength) as WireU64,
      tokenSha256: await sha256Hex(SOURCE_BYTES),
    },
  };
  const entry = {
    ...entryBody,
    requestId: await deriveCppCuteFrontendEntryRequestId(entryBody),
  };
  const body: CppCuteFrontendRequestBodyV1 = {
    schema: "browsergrad.compiler.cpp-cute.frontend-request",
    version: { major: 1, minor: 0 },
    compilationContractHash: profile.compilationContractHash,
    mainVirtualPath: SOURCE_PATH,
    files: [source],
    entryRequests: [entry],
    expectedArtifact: {
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
      version: {
        major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
        minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
      },
    },
    limits: representableRequestLimits(profile),
  };
  return {
    ...body,
    requestId: `bg.cpp.frontend-request.sha256.${await deriveCppCuteFrontendRequestHash(body)}`,
  };
}

function representableRequestLimits(
  profile: PreparedCppCuteFrontendProfile,
): CppCuteFrontendRequestLimitsV1 {
  const limits = profile.extractionLimits;
  return {
    maxSourceFiles: limits.maxSourceFiles,
    maxSourceBytes: limits.maxSourceBytes,
    maxHeaderFiles: 100,
    maxHeaderBytes: limits.maxHeaderBytes,
    maxIncludeDepth: limits.maxIncludeDepth,
    maxMacroExpansions: 8_192,
    maxPreprocessedTokens: limits.maxPreprocessedTokens,
    maxAstNodes: limits.maxAstNodes,
    maxConstexprSteps: limits.maxConstexprSteps,
    maxTemplateInstantiations: 8_192,
    maxTemplateDepth: limits.maxTemplateDepth,
    maxDeclarations: 16_384,
    maxTypes: 16_384,
    maxConstants: 16_384,
    maxLayouts: 1_024,
    maxTensors: 1_024,
    maxOperations: 4_096,
    maxTargetIntrinsics: 1_024,
    maxDiagnostics: 4_096,
    maxOutputBytes: limits.maxOutputBytes,
  };
}

function invocationObject(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
): Record<string, unknown> {
  return JSON.parse(new TextDecoder().decode(
    canonicalCppCuteBrowserWorkerInvocationBytes(invocation),
  )) as Record<string, unknown>;
}
