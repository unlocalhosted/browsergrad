import {
  DEFAULT_DECODE_LIMITS,
  MAXIMUM_DECODE_LIMITS,
  canonicalJsonBytes,
  encodeWireU64,
  hashCanonicalJson,
  sha256Hex,
  wireIntegerToBigInt,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it, vi } from "vitest";

const authorities = vi.hoisted(() => ({
  manifests: new WeakMap<object, unknown>(),
  assetSets: new WeakMap<object, unknown>(),
  installations: new WeakMap<object, unknown>(),
  runtimeAbiAssets: new WeakMap<object, unknown>(),
  verifierEvidence: new WeakMap<object, unknown>(),
  assetBytes: new WeakMap<object, ReadonlyMap<string, Uint8Array>>(),
  assetCopyCalls: 0,
}));
const hashControl = vi.hoisted(() => ({
  beforeNextSha256: null as null | (() => Promise<void>),
  sha256Calls: 0,
  lastInvocationSha256Calls: 0,
}));

vi.mock("@unlocalhosted/browsergrad-semantic-core/schema", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@unlocalhosted/browsergrad-semantic-core/schema")>();
  return {
    ...actual,
    sha256Hex: async (bytes: Uint8Array) => {
      hashControl.sha256Calls += 1;
      const beforeHash = hashControl.beforeNextSha256;
      if (beforeHash !== null) {
        hashControl.beforeNextSha256 = null;
        await beforeHash();
      }
      return actual.sha256Hex(bytes);
    },
  };
});

vi.mock("../../src/cpp_cute_browser_assets.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_browser_assets.js")>();
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
  const actual = await importOriginal<typeof import("../../src/cpp_cute_browser_asset_installation.js")>();
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
      authorities.assetCopyCalls += 1;
      const bytes = authorities.assetBytes.get(value)?.get(assetId);
      if (bytes === undefined) throw new Error("unregistered test asset bytes");
      return new Uint8Array(bytes);
    },
  };
});

vi.mock("../../src/cpp_cute_browser_wasm_verifier_evidence.js", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("../../src/cpp_cute_browser_wasm_verifier_evidence.js")
  >();
  return {
    ...actual,
    unwrapPreparedCppCuteBrowserWasmVerifierEvidence: (value: object) => {
      const record = authorities.verifierEvidence.get(value);
      if (record === undefined) throw new Error("unregistered test verifier-evidence authority");
      return record;
    },
  };
});

import {
  canonicalCppCuteFrontendArtifactBytes,
  deriveCppCuteFrontendArtifactId,
  unwrapVerifiedCppCuteFrontendArtifact,
  verifyCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
} from "../../src/cpp_cute_frontend_artifact.js";
import {
  assembleCppCuteBrowserInputFrameRegions,
  copyPreparedCppCuteBrowserInputFrameBytes,
  CppCuteBrowserInputFrameError,
  prepareCppCuteBrowserInputFrame,
} from "../../src/cpp_cute_browser_input_frame.js";
import {
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
  decodeCppCuteBrowserRuntimeAbiManifest,
  unwrapPreparedCppCuteBrowserRuntimeAbiManifest,
} from "../../src/cpp_cute_browser_runtime_abi.js";
import {
  buildCanonicalCppCuteBrowserWorkerResultControl,
  canonicalCppCuteBrowserWorkerProfileRegionBytes,
  canonicalCppCuteBrowserWorkerRequestRegionBytes,
  copyCppCuteBrowserWorkerSourceSnapshots,
  discardCppCuteBrowserWorkerInvocation,
  prepareCppCuteBrowserWorkerInvocation,
  unwrapPreparedCppCuteBrowserWorkerInvocation,
  unwrapValidatedCppCuteBrowserWorkerResultFrame,
  validateCppCuteBrowserWorkerResultFrame,
  type CppCuteBrowserWorkerResultV1,
  type PreparedCppCuteBrowserWorkerInvocation,
} from "../../src/cpp_cute_browser_worker_protocol.js";
import {
  CPP_CUTE_BROWSER_WASM_COMPILER_PROTOCOL,
  type CppCuteBrowserWasmCompilerExecution,
} from "../../src/cpp_cute_browser_wasm_compiler.js";
import {
  CPP_CUTE_BROWSER_FRONTEND_WORK_METRICS_PROTOCOL,
} from "../../src/cpp_cute_browser_frontend_work_metrics.js";
import {
  CPP_CUTE_FRONTEND_REQUEST_LOGICAL_GEMM_TILE_MINOR,
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
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  unwrapPreparedCppCuteFrontendRequestBinding,
} from "../../src/cpp_cute_frontend_request_binding.js";
import type {
  CppCuteFrontendArtifactV3,
  CppCuteFrontendPayloadV3,
} from "../../src/cpp_cute_frontend_types.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_LOGICAL_GEMM_TILE_MINOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
} from "../../src/cpp_cute_frontend_types.js";
import {
  computeCppCuteInputHashes,
  computeCppCuteSemanticPassInputClosureHash,
  computeCppCuteSharedSurfaceHash,
} from "../../src/cpp_cute_frontend_verify.js";
import {
  CPP_CUTE_FIXTURE_MAIN_FILE_ID,
  createCppCuteBrowserProfileInput,
  createCppCutePayloadInput,
  rebindCppCuteFixtureSourceEntityIds,
} from "./support/cpp_cute_frontend_fixtures.js";

const SOURCE_PATH = "/src/layout.cu";
const SOURCE_BYTES = new TextEncoder().encode(
  "auto layout = make_layout(Int<2>{});".padEnd(100, " "),
);
const ARTIFACT_TEST_DECODE_LIMITS = Object.freeze({
  ...MAXIMUM_DECODE_LIMITS,
  maxDocumentBytes: 8 * 1024 * 1024,
  maxStringBytes: 8 * 1024 * 1024,
});

interface PositiveEnvironment {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly invocation: PreparedCppCuteBrowserWorkerInvocation;
  readonly accepted: ArtifactFixture;
  readonly rejected: ArtifactFixture;
}

interface ArtifactFixture {
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly bytes: Uint8Array;
  readonly payload: CppCuteFrontendPayloadV3;
}

describe("C++/CuTe browser Worker positive framing", () => {
  it("builds canonical control from the exact local execution projection", async () => {
    const environment = await createEnvironment("accepted");
    const expected = await createResult(
      environment.invocation,
      environment.accepted,
      environment.profile,
    );
    const execution = await createExecution(
      environment.invocation,
      environment.accepted,
      expected,
    );

    const controlBytes = await buildCanonicalCppCuteBrowserWorkerResultControl(
      environment.invocation,
      execution,
    );
    expect(controlBytes).toEqual(canonicalJsonBytes(expected));
    await expect(validateCppCuteBrowserWorkerResultFrame(
      environment.invocation,
      controlBytes,
      environment.accepted.bytes,
    )).resolves.toMatchObject({
      outcome: "accepted",
      artifactId: environment.accepted.artifact.artifactId,
    });
  });

  it("validates canonical accepted and rejected caller-frame protocol consistency", async () => {
    const acceptedEnvironment = await createEnvironment("accepted");
    const acceptedResult = await createResult(
      acceptedEnvironment.invocation,
      acceptedEnvironment.accepted,
      acceptedEnvironment.profile,
    );
    const accepted = await validateCppCuteBrowserWorkerResultFrame(
      acceptedEnvironment.invocation,
      canonicalJsonBytes(acceptedResult),
      acceptedEnvironment.accepted.bytes,
    );
    expect(accepted).toMatchObject({
      outcome: "accepted",
      artifactId: acceptedEnvironment.accepted.artifact.artifactId,
      workerExecutionObserved: false,
      workerTerminationObserved: false,
      loweringAuthorityMinted: false,
    });
    const acceptedRecord = unwrapValidatedCppCuteBrowserWorkerResultFrame(accepted);
    const acceptedBinding = unwrapPreparedCppCuteFrontendRequestBinding(
      acceptedRecord.requestBinding,
    );
    expect(acceptedRecord).toMatchObject({
      artifact: acceptedEnvironment.accepted.artifact,
      profile: acceptedEnvironment.profile,
    });
    expect(acceptedBinding).toMatchObject({
      artifact: acceptedEnvironment.accepted.artifact,
    });
    expect(acceptedBinding.request.profileHash).toBe(acceptedEnvironment.profile.profileHash);
    expect(() => unwrapValidatedCppCuteBrowserWorkerResultFrame({ ...accepted } as never))
      .toThrowError(expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-UNVERIFIED",
        path: "$.validatedResultFrame",
      }));

    const rejectedEnvironment = await createEnvironment("rejected");
    const rejectedResult = await createResult(
      rejectedEnvironment.invocation,
      rejectedEnvironment.rejected,
      rejectedEnvironment.profile,
    );
    const rejected = await validateCppCuteBrowserWorkerResultFrame(
      rejectedEnvironment.invocation,
      canonicalJsonBytes(rejectedResult),
      rejectedEnvironment.rejected.bytes,
    );
    expect(rejected).toMatchObject({
      outcome: "rejected",
      artifactId: rejectedEnvironment.rejected.artifact.artifactId,
      loweringAuthorityMinted: false,
    });
  });

  it("accepts the exact diagnostic ceiling above generic schema decode budgets", {
    timeout: 15_000,
  }, async () => {
    const environment = await createEnvironment("accepted");
    const largeRejected = await createArtifact(environment.profile, "rejected", {
      diagnosticCount: 4_095,
      renderedMessageBytes: 513,
    });
    const cumulativeDiagnosticBytes = largeRejected.payload.diagnostics.reduce(
      (total, diagnostic) => total + new TextEncoder().encode(diagnostic.renderedMessage).byteLength,
      0,
    );
    expect(largeRejected.payload.diagnostics).toHaveLength(4_096);
    expect(cumulativeDiagnosticBytes).toBeGreaterThan(DEFAULT_DECODE_LIMITS.maxStringBytes);
    const result = await createResult(
      environment.invocation,
      largeRejected,
      environment.profile,
    );

    await expect(validateCppCuteBrowserWorkerResultFrame(
      environment.invocation,
      canonicalJsonBytes(result),
      largeRejected.bytes,
    )).resolves.toMatchObject({ outcome: "rejected" });
  });

  it("encodes the exact 64-byte little-endian ABI header and zero padding", async () => {
    const environment = await createEnvironment("accepted");
    const prepared = await prepareCppCuteBrowserInputFrame(environment.invocation);
    const bytes = copyPreparedCppCuteBrowserInputFrameBytes(prepared);
    const invocationRecord = unwrapPreparedCppCuteBrowserWorkerInvocation(environment.invocation);
    const abi = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(invocationRecord.runtimeAbi).manifest.body.inputFrame;
    const profileBytes = canonicalCppCuteBrowserWorkerProfileRegionBytes(environment.invocation);
    const requestBytes = canonicalCppCuteBrowserWorkerRequestRegionBytes(environment.invocation);
    const header = new DataView(bytes.buffer, bytes.byteOffset, 64);

    expect(new TextDecoder().decode(bytes.subarray(0, 8))).toBe("BGCCABI1");
    expect(hex(bytes.subarray(0, 16))).toBe("42474343414249310100000040000000");
    expect([
      header.getUint16(8, true),
      header.getUint16(10, true),
      header.getUint32(12, true),
      header.getUint32(16, true),
      header.getUint32(20, true),
      header.getUint32(24, true),
      header.getUint32(28, true),
      header.getUint32(32, true),
      header.getUint32(36, true),
    ]).toEqual([
      1, 0, 64, prepared.frameByteLength, 0, 64, profileBytes.byteLength,
      prepared.requestOffset, requestBytes.byteLength,
    ]);
    expect([...bytes.subarray(40, 64)]).toEqual(Array.from({ length: 24 }, () => 0));
    expect(bytes.subarray(prepared.profileOffset, prepared.profileOffset + prepared.profileByteLength))
      .toEqual(profileBytes);
    expect(bytes.subarray(prepared.requestOffset, prepared.requestOffset + prepared.requestByteLength))
      .toEqual(requestBytes);
    expect([...bytes.subarray(
      prepared.profileOffset + prepared.profileByteLength,
      prepared.requestOffset,
    )]).toEqual(Array.from({
      length: prepared.requestOffset - prepared.profileOffset - prepared.profileByteLength,
    }, () => 0));
    expect([...bytes.subarray(prepared.requestOffset + prepared.requestByteLength)])
      .toEqual(Array.from({
        length: prepared.frameByteLength - prepared.requestOffset - prepared.requestByteLength,
      }, () => 0));
    expect(prepared.frameByteLength).toBeLessThanOrEqual(abi.maxFrameByteLength);
    expect(prepared.frameSha256).toBe(await sha256Hex(bytes));
  });

  it("delegates to the pure region assembler with byte-identical framing", async () => {
    const environment = await createEnvironment("accepted");
    const invocationRecord = unwrapPreparedCppCuteBrowserWorkerInvocation(environment.invocation);
    const profileRegionBytes = canonicalCppCuteBrowserWorkerProfileRegionBytes(
      environment.invocation,
    );
    const requestRegionBytes = canonicalCppCuteBrowserWorkerRequestRegionBytes(
      environment.invocation,
    );
    const inputFrame = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(
      invocationRecord.runtimeAbi,
    ).manifest.body.inputFrame;
    const sourceSnapshots = copyCppCuteBrowserWorkerSourceSnapshots(environment.invocation);
    const assembled = assembleCppCuteBrowserInputFrameRegions({
      profileRegionBytes,
      requestRegionBytes,
      sourceSnapshots,
      limits: {
        maxFrameByteLength: inputFrame.maxFrameByteLength,
        maxSourceSnapshotCount: invocationRecord.request.sourceFileCount,
        maxSourceSnapshotByteLength: Number(invocationRecord.request.sourceByteLength),
      },
    });
    const prepared = await prepareCppCuteBrowserInputFrame(environment.invocation);

    expect(assembled.frameBytes).toEqual(
      copyPreparedCppCuteBrowserInputFrameBytes(prepared),
    );
    expect(assembled).toMatchObject({
      frameByteLength: prepared.frameByteLength,
      profileOffset: prepared.profileOffset,
      profileByteLength: prepared.profileByteLength,
      requestOffset: prepared.requestOffset,
      requestByteLength: prepared.requestByteLength,
      sourceSnapshotCount: sourceSnapshots.length,
      sourceSnapshotByteLength: SOURCE_BYTES.byteLength,
    });
    expect(Reflect.ownKeys(assembled)).toEqual([
      "frameBytes",
      "frameByteLength",
      "profileOffset",
      "profileByteLength",
      "requestOffset",
      "requestByteLength",
      "sourceSnapshotCount",
      "sourceSnapshotByteLength",
    ]);
    expect(assembled).not.toHaveProperty("invocationId");
    expect(assembled).not.toHaveProperty("workerExecutionObserved");
    expect(assembled).not.toHaveProperty("loweringAuthorityMinted");

    profileRegionBytes.fill(0);
    requestRegionBytes.fill(0);
    sourceSnapshots[0]?.bytes.fill(0);
    expect(new TextDecoder().decode(assembled.frameBytes.subarray(0, 8))).toBe("BGCCABI1");
    expect(assembled.frameBytes.subarray(
      assembled.profileOffset,
      assembled.profileOffset + assembled.profileByteLength,
    )).not.toEqual(profileRegionBytes);
  });

  it("rejects hostile region/source buffers and accessor-bearing records", () => {
    class DerivedBytes extends Uint8Array {}
    const limits = {
      maxFrameByteLength: 1024,
      maxSourceSnapshotCount: 1,
      maxSourceSnapshotByteLength: 4,
    };
    expect(() => assembleCppCuteBrowserInputFrameRegions({
      profileRegionBytes: new DerivedBytes([1]),
      requestRegionBytes: Uint8Array.of(2),
      sourceSnapshots: [],
      limits,
    })).toThrowError(expect.objectContaining<Partial<CppCuteBrowserInputFrameError>>({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-INVALID",
      path: "$.input.profileRegionBytes",
    }));

    const accessorSource = Object.defineProperty({}, "bytes", {
      enumerable: true,
      get: () => Uint8Array.of(3),
    });
    Object.defineProperty(accessorSource, "virtualPath", {
      enumerable: true,
      value: "/src/main.cu",
    });
    expect(() => assembleCppCuteBrowserInputFrameRegions({
      profileRegionBytes: Uint8Array.of(1),
      requestRegionBytes: Uint8Array.of(2),
      sourceSnapshots: [accessorSource as never],
      limits,
    })).toThrowError(expect.objectContaining<Partial<CppCuteBrowserInputFrameError>>({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-INVALID",
      path: "$.input.sourceSnapshots[0].bytes",
    }));

    if (typeof SharedArrayBuffer !== "undefined") {
      expect(() => assembleCppCuteBrowserInputFrameRegions({
        profileRegionBytes: Uint8Array.of(1),
        requestRegionBytes: new Uint8Array(new SharedArrayBuffer(1)),
        sourceSnapshots: [],
        limits,
      })).toThrowError(expect.objectContaining<Partial<CppCuteBrowserInputFrameError>>({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-INVALID",
        path: "$.input.requestRegionBytes",
      }));
    }
  });

  it("fails closed on frame, source-count, source-byte, and manifest hard-limit overflow", async () => {
    const runtimeAbi = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
    const maxFrameByteLength = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(runtimeAbi)
      .manifest.body.inputFrame.maxFrameByteLength;
    const base = {
      profileRegionBytes: Uint8Array.of(1),
      requestRegionBytes: Uint8Array.of(2),
      sourceSnapshots: [{ virtualPath: "/src/main.cu", bytes: Uint8Array.of(3, 4) }],
    };
    expect(() => assembleCppCuteBrowserInputFrameRegions({
      ...base,
      limits: {
        maxFrameByteLength: 64,
        maxSourceSnapshotCount: 1,
        maxSourceSnapshotByteLength: 2,
      },
    })).toThrowError(expect.objectContaining<Partial<CppCuteBrowserInputFrameError>>({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-RESOURCE-LIMIT",
      path: "$.frameByteLength",
    }));
    expect(() => assembleCppCuteBrowserInputFrameRegions({
      ...base,
      limits: {
        maxFrameByteLength: 1024,
        maxSourceSnapshotCount: 0,
        maxSourceSnapshotByteLength: 2,
      },
    })).toThrowError(expect.objectContaining<Partial<CppCuteBrowserInputFrameError>>({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-RESOURCE-LIMIT",
      path: "$.input.sourceSnapshots",
    }));
    expect(() => assembleCppCuteBrowserInputFrameRegions({
      ...base,
      limits: {
        maxFrameByteLength: 1024,
        maxSourceSnapshotCount: 1,
        maxSourceSnapshotByteLength: 1,
      },
    })).toThrowError(expect.objectContaining<Partial<CppCuteBrowserInputFrameError>>({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-RESOURCE-LIMIT",
      path: "$.input.sourceSnapshots",
    }));
    expect(() => assembleCppCuteBrowserInputFrameRegions({
      ...base,
      limits: {
        maxFrameByteLength: maxFrameByteLength + 1,
        maxSourceSnapshotCount: 1,
        maxSourceSnapshotByteLength: 2,
      },
    })).toThrowError(expect.objectContaining<Partial<CppCuteBrowserInputFrameError>>({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-INVALID",
      path: "$.input.limits.maxFrameByteLength",
    }));
  });

  it("keeps frame bytes isolated and rejects forged frame authorities", async () => {
    const environment = await createEnvironment("accepted");
    const prepared = await prepareCppCuteBrowserInputFrame(environment.invocation);
    const first = copyPreparedCppCuteBrowserInputFrameBytes(prepared);
    first.fill(0);
    expect(copyPreparedCppCuteBrowserInputFrameBytes(prepared).subarray(0, 8))
      .toEqual(new TextEncoder().encode("BGCCABI1"));
    expect(() => copyPreparedCppCuteBrowserInputFrameBytes({ ...prepared } as never)).toThrowError(
      expect.objectContaining<Partial<CppCuteBrowserInputFrameError>>({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-UNVERIFIED",
        path: "$.prepared",
      }),
    );
  });

  it("memoizes in-flight and prepared frames per invocation", async () => {
    const environment = await createEnvironment("accepted");
    const first = prepareCppCuteBrowserInputFrame(environment.invocation);
    const second = prepareCppCuteBrowserInputFrame(environment.invocation);
    expect(second).toBe(first);

    const prepared = await first;
    await expect(prepareCppCuteBrowserInputFrame(environment.invocation)).resolves.toBe(prepared);
  });

  it("revokes prepared frame bytes when the invocation terminalizes", async () => {
    const environment = await createEnvironment("accepted");
    const prepared = await prepareCppCuteBrowserInputFrame(environment.invocation);

    discardCppCuteBrowserWorkerInvocation(environment.invocation, "caller-cancelled");

    expect(() => copyPreparedCppCuteBrowserInputFrameBytes(prepared)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-DUPLICATE-OR-LATE",
        path: "$.invocation",
      }),
    );
    await expect(prepareCppCuteBrowserInputFrame(environment.invocation)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-DUPLICATE-OR-LATE",
      path: "$.invocation",
    });
  });

  it("does not mint a frame if hashing races with invocation terminalization", async () => {
    const environment = await createEnvironment("accepted");
    const hashStarted = deferred();
    const releaseHash = deferred();
    hashControl.beforeNextSha256 = async () => {
      hashStarted.resolve();
      await releaseHash.promise;
    };

    const preparation = prepareCppCuteBrowserInputFrame(environment.invocation);
    await hashStarted.promise;
    discardCppCuteBrowserWorkerInvocation(environment.invocation, "caller-cancelled");
    releaseHash.resolve();

    await expect(preparation).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-DUPLICATE-OR-LATE",
      path: "$.invocation",
    });
  });

  it("rejects request-derived ceilings before copying executable assets", async () => {
    const assetCopyCalls = authorities.assetCopyCalls;
    await expect(createEnvironment("accepted", "profile-wide")).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RESOURCE-LIMIT",
      path: "$.request.limits",
    });
    expect(authorities.assetCopyCalls).toBe(assetCopyCalls);
    expect(hashControl.lastInvocationSha256Calls).toBe(0);
  });

  it("rejects profile include roots above the artifact verifier ceiling before asset copies", async () => {
    const assetCopyCalls = authorities.assetCopyCalls;
    await expect(createEnvironment(
      "accepted",
      "representable",
      "include-root-overflow",
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RESOURCE-LIMIT",
      path: "$.profile.virtualFileSystem.includeRoots",
    });
    expect(authorities.assetCopyCalls).toBe(assetCopyCalls);
    expect(hashControl.lastInvocationSha256Calls).toBe(0);
  });

  it("rejects typed-artifact logical GEMM requests before Worker assets or compilation", async () => {
    const assetCopyCalls = authorities.assetCopyCalls;
    await expect(createEnvironment(
      "accepted",
      "representable",
      "default",
      "logical-gemm-tile",
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-UNSUPPORTED-ENTRY",
      path: "$.request.entryRequests[0].kind",
    });
    expect(authorities.assetCopyCalls).toBe(assetCopyCalls);
    expect(hashControl.lastInvocationSha256Calls).toBe(0);
  });
});

async function createEnvironment(
  outcome: "accepted" | "rejected",
  requestLimitMode: "representable" | "profile-wide" = "representable",
  profileMode: "default" | "include-root-overflow" = "default",
  entryKind: "layout" | "logical-gemm-tile" = "layout",
): Promise<PositiveEnvironment> {
  const clangBytes = Uint8Array.of(0, 97, 115, 109, 1, 0, 0, 0);
  const workerBytes = new Uint8Array(65_536);
  workerBytes.set(new TextEncoder().encode("browsergrad-worker-test-fixture"));
  const profileInput = structuredClone(createCppCuteBrowserProfileInput({ sourceRoots: ["/src"] }));
  if (profileMode === "include-root-overflow") {
    const includeRoots = profileInput.virtualFileSystem.includeRoots as Array<
      (typeof profileInput.virtualFileSystem.includeRoots)[number]
    >;
    const sourceRoot = includeRoots.find((root) => root.owner.kind === "source");
    if (sourceRoot === undefined) throw new Error("fixture lost source include root");
    const firstSystemRoot = includeRoots.findIndex((root) => root.owner.kind !== "source");
    includeRoots.splice(firstSystemRoot, 0, ...Array.from({ length: 59 }, (_, index) => ({
      includeRootId: `extra-source-${index}`,
      mode: "quote" as const,
      virtualPath: `/src/extra-${index}`,
      manifestSha256: sourceRoot.manifestSha256,
      owner: { kind: "source" as const },
    })));
  }
  const clangSha256 = await sha256Hex(clangBytes);
  const workerSha256 = await sha256Hex(workerBytes);
  (profileInput.toolchain.compiler as { binarySha256: string }).binarySha256 = clangSha256;
  (profileInput.deployment.extractor as { binarySha256: string }).binarySha256 = clangSha256;
  (profileInput.deployment.worker as { moduleSha256: string; moduleByteLength: number }).moduleSha256 = workerSha256;
  (profileInput.deployment.worker as { moduleSha256: string; moduleByteLength: number }).moduleByteLength =
    workerBytes.byteLength;
  const profile = await prepareCppCuteFrontendProfile(profileInput);
  const requestInput = await createRequestInput(profile, requestLimitMode, entryKind);
  const request = await prepareCppCuteFrontendRequest(
    profile,
    requestInput,
    [{ virtualPath: SOURCE_PATH, bytes: SOURCE_BYTES }],
  );
  const accepted = await createArtifact(profile, "accepted");
  const rejected = await createArtifact(profile, "rejected");
  const runtimeAbi = await decodeCppCuteBrowserRuntimeAbiManifest(
    cppCuteBrowserRuntimeAbiManifestResourceBytes(),
  );
  const hash = "a".repeat(64);
  const assetManifest = Object.freeze({
    manifestId: `bg.cpp.browser-assets.sha256.${hash}`,
    manifestSha256: hash,
    assetSetSha256: profileInput.deployment.assetSetSha256,
  });
  const assetSet = Object.freeze({ assetSetSha256: profileInput.deployment.assetSetSha256 });
  const installation = Object.freeze({ installationId: `bg.cpp.browser-vfs.sha256.${hash}` });
  const runtimeAbiAsset = Object.freeze({ runtimeAbiManifestId: runtimeAbi.manifestId });
  const verifierEvidence = Object.freeze({
    sourceEvidenceId: `bg.cpp.browser-wasm-verifier-conformance.sha256.${"c".repeat(64)}`,
    regionSha256: "e".repeat(64),
    wasmAssetId: "clang-wasm",
    wasmSha256: clangSha256,
    wasmByteLength: clangBytes.byteLength,
    observedProjectionSha256: "b".repeat(64),
    runtimeAbiManifestId: runtimeAbi.manifestId,
    runtimeAbiContractSha256: runtimeAbi.contractSha256,
    runtimeAbiResourceSha256: runtimeAbi.resourceSha256,
  });
  const payload = outcome === "accepted" ? accepted.payload : rejected.payload;
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
  authorities.installations.set(installation, {
    assetSet,
    files: payload.inputs.files.filter((file) => file.owner.kind !== "source").map((file) => ({
      virtualPath: file.virtualPath,
      contentSha256: file.contentSha256,
      byteLength: file.byteLength,
      includeRootId: file.includeRootId,
    })),
  });
  authorities.runtimeAbiAssets.set(runtimeAbiAsset, { assetSet, runtimeAbi });
  authorities.verifierEvidence.set(verifierEvidence, {
    assetSet,
    assetManifest,
    runtimeAbiAsset,
    runtimeAbi,
  });
  authorities.assetBytes.set(assetSet, new Map([["clang-wasm", clangBytes]]));
  const sha256Calls = hashControl.sha256Calls;
  const invocation = await prepareCppCuteBrowserWorkerInvocation({
    profile,
    assetManifest: assetManifest as never,
    vfsInstallation: installation as never,
    request,
    runtimeAbiAsset: runtimeAbiAsset as never,
    verifierEvidence: verifierEvidence as never,
    workerModuleBytes: workerBytes,
  }).finally(() => {
    hashControl.lastInvocationSha256Calls = hashControl.sha256Calls - sha256Calls;
  });
  return { profile, invocation, accepted, rejected };
}

async function createRequestInput(
  profile: PreparedCppCuteFrontendProfile,
  limitMode: "representable" | "profile-wide",
  entryKind: "layout" | "logical-gemm-tile" = "layout",
): Promise<CppCuteFrontendRequestV1> {
  const descriptorBody = {
    role: "main-source" as const,
    virtualPath: SOURCE_PATH,
    contentSha256: await sha256Hex(SOURCE_BYTES),
    byteLength: encodeWireU64(BigInt(SOURCE_BYTES.byteLength)),
    includeRootId: null,
  };
  const file: CppCuteFrontendRequestSourceFileV1 = {
    fileId: await deriveCppCuteFrontendSourceFileId(descriptorBody),
    ...descriptorBody,
  };
  const entrySelection = entryKind === "layout"
    ? { kind: "layout" as const, declarationKind: "variable" as const }
    : { kind: "logical-gemm-tile" as const, declarationKind: "function" as const };
  const entryBody = {
    requestId: `bg.cpp.entry-request.sha256.${"0".repeat(64)}`,
    ...entrySelection,
    anchor: {
      virtualPath: SOURCE_PATH,
      beginByte: "0" as WireU64,
      endByte: String(SOURCE_BYTES.byteLength) as WireU64,
      tokenSha256: await sha256Hex(SOURCE_BYTES),
    },
  };
  const entry = { ...entryBody, requestId: await deriveCppCuteFrontendEntryRequestId(entryBody) };
  const body: CppCuteFrontendRequestBodyV1 = {
    schema: "browsergrad.compiler.cpp-cute.frontend-request",
    version: {
      major: 1,
      minor: entryKind === "logical-gemm-tile"
        ? CPP_CUTE_FRONTEND_REQUEST_LOGICAL_GEMM_TILE_MINOR
        : 0,
    },
    compilationContractHash: profile.compilationContractHash,
    mainVirtualPath: SOURCE_PATH,
    files: [file],
    entryRequests: [entry],
    expectedArtifact: {
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
      version: {
        major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
        minor: entryKind === "logical-gemm-tile"
          ? CPP_CUTE_FRONTEND_ARTIFACT_LOGICAL_GEMM_TILE_MINOR
          : CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
      },
    },
    limits: limitMode === "representable"
      ? representableRequestLimits(profile)
      : profileWideRequestLimits(profile),
  };
  return {
    ...body,
    requestId: `bg.cpp.frontend-request.sha256.${await deriveCppCuteFrontendRequestHash(body)}`,
  };
}

function profileWideRequestLimits(profile: PreparedCppCuteFrontendProfile): CppCuteFrontendRequestLimitsV1 {
  const limits = profile.extractionLimits;
  return {
    maxSourceFiles: limits.maxSourceFiles,
    maxSourceBytes: limits.maxSourceBytes,
    maxHeaderFiles: limits.maxHeaderFiles,
    maxHeaderBytes: limits.maxHeaderBytes,
    maxIncludeDepth: limits.maxIncludeDepth,
    maxMacroExpansions: limits.maxMacroExpansions,
    maxPreprocessedTokens: limits.maxPreprocessedTokens,
    maxAstNodes: limits.maxAstNodes,
    maxConstexprSteps: limits.maxConstexprSteps,
    maxTemplateInstantiations: limits.maxTemplateInstantiations,
    maxTemplateDepth: limits.maxTemplateDepth,
    maxDeclarations: limits.maxDeclarations,
    maxTypes: limits.maxTypes,
    maxConstants: limits.maxConstants,
    maxLayouts: limits.maxLayouts,
    maxTensors: limits.maxTensors,
    maxOperations: limits.maxOperations,
    maxTargetIntrinsics: limits.maxTargetIntrinsics,
    maxDiagnostics: limits.maxDiagnostics,
    maxOutputBytes: limits.maxOutputBytes,
  };
}

function representableRequestLimits(profile: PreparedCppCuteFrontendProfile): CppCuteFrontendRequestLimitsV1 {
  const limits = profile.extractionLimits;
  return {
    maxSourceFiles: limits.maxSourceFiles,
    maxSourceBytes: limits.maxSourceBytes,
    maxHeaderFiles: 100,
    maxHeaderBytes: limits.maxHeaderBytes,
    maxIncludeDepth: limits.maxIncludeDepth,
    maxMacroExpansions: limits.maxMacroExpansions,
    maxPreprocessedTokens: limits.maxPreprocessedTokens,
    maxAstNodes: limits.maxAstNodes,
    maxConstexprSteps: limits.maxConstexprSteps,
    maxTemplateInstantiations: limits.maxTemplateInstantiations,
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

async function createArtifact(
  profile: PreparedCppCuteFrontendProfile,
  outcome: "accepted" | "rejected",
  options: {
    readonly diagnosticCount?: number;
    readonly renderedMessageBytes?: number;
  } = {},
): Promise<ArtifactFixture> {
  const payload = structuredClone(await createCppCutePayloadInput(profile.compilationContractHash));
  const fileBody = {
    role: "main-source" as const,
    virtualPath: SOURCE_PATH,
    contentSha256: await sha256Hex(SOURCE_BYTES),
    byteLength: String(SOURCE_BYTES.byteLength) as WireU64,
    includeRootId: null,
  };
  const fileId = await deriveCppCuteFrontendSourceFileId(fileBody);
  replaceString(payload, CPP_CUTE_FIXTURE_MAIN_FILE_ID, fileId);
  const mainFile = payload.inputs.files.find((file) => file.fileId === fileId);
  if (mainFile === undefined) throw new Error("fixture lost main source");
  (mainFile as { contentSha256: string }).contentSha256 = fileBody.contentSha256;
  (mainFile as { byteLength: WireU64 }).byteLength = fileBody.byteLength;
  await rebindCppCuteFixtureSourceEntityIds(payload);
  const inputHashes = await computeCppCuteInputHashes(payload, {
    limits: ARTIFACT_TEST_DECODE_LIMITS,
  });
  (payload.inputs as { sourceSetSha256: string }).sourceSetSha256 = inputHashes.sourceSetSha256;
  (payload.inputs as { headerSetSha256: string }).headerSetSha256 = inputHashes.headerSetSha256;
  (payload.inputs as { closureSha256: string }).closureSha256 = inputHashes.closureSha256;
  (payload.extraction as { inputClosureSha256: string }).inputClosureSha256 = inputHashes.closureSha256;
  for (const [index, pass] of payload.semanticPasses.entries()) {
    (pass as { observedInputClosureSha256: string }).observedInputClosureSha256 =
      await computeCppCuteSemanticPassInputClosureHash(payload, index, {
        limits: ARTIFACT_TEST_DECODE_LIMITS,
      });
    (pass as { sharedSurfaceSha256: string }).sharedSurfaceSha256 =
      await computeCppCuteSharedSurfaceHash(payload, pass.domain, {
        limits: ARTIFACT_TEST_DECODE_LIMITS,
      });
  }
  if (outcome === "rejected") {
    rejectPayload(
      payload,
      options.diagnosticCount ?? 1,
      options.renderedMessageBytes ?? 48,
    );
  }
  const artifactInput: CppCuteFrontendArtifactV3 = {
    schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
    version: {
      major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
      minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
    },
    producer: { id: "browsergrad-tools/cpp-cute-frontend", version: "0.1.0" },
    artifactId: await deriveCppCuteFrontendArtifactId(payload, {
      limits: ARTIFACT_TEST_DECODE_LIMITS,
    }),
    payload,
    requiredExtensions: [],
  };
  const artifact = await verifyCppCuteFrontendArtifact(artifactInput, {
    limits: ARTIFACT_TEST_DECODE_LIMITS,
  });
  return {
    artifact,
    bytes: canonicalCppCuteFrontendArtifactBytes(artifact, {
      limits: ARTIFACT_TEST_DECODE_LIMITS,
    }),
    payload,
  };
}

function rejectPayload(
  payload: CppCuteFrontendPayloadV3,
  diagnosticCount: number,
  renderedMessageBytes: number,
): void {
  const blockingDiagnosticIds = Array.from({ length: diagnosticCount }, (_, index) =>
    `bg.cpp.diagnostic.sha256.${(index + 1).toString(16).padStart(64, "0")}`);
  (payload.diagnostics as unknown as Array<unknown>).push(...blockingDiagnosticIds.map(
    (diagnosticId, index) => ({
      diagnosticId,
      phase: "artifact-extraction",
      severity: "error",
      code: `browsergrad.cpp-cute:fixture-rejected-${index}`,
      renderedMessage: "x".repeat(renderedMessageBytes),
      location: { kind: "none" },
      subject: { kind: "compiler" },
      parentDiagnosticId: null,
    }),
  ));
  (payload.diagnostics as unknown as Array<{ diagnosticId: string }>).sort((left, right) =>
    left.diagnosticId.localeCompare(right.diagnosticId));
  const hostPass = payload.semanticPasses[1];
  if (hostPass === undefined) throw new Error("fixture lost host pass");
  (hostPass as { status: string }).status = "failed";
  (hostPass as { diagnosticIds: readonly string[] }).diagnosticIds = [...blockingDiagnosticIds];
  (payload as { outcome: unknown }).outcome = {
    kind: "rejected",
    blockingDiagnosticIds: [...blockingDiagnosticIds],
  };
}

async function createResult(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
  fixture: ArtifactFixture,
  profile: PreparedCppCuteFrontendProfile,
): Promise<CppCuteBrowserWorkerResultV1> {
  const invocationRecord = unwrapPreparedCppCuteBrowserWorkerInvocation(invocation);
  const payload = unwrapVerifiedCppCuteFrontendArtifact(fixture.artifact).envelope.payload;
  const sources = payload.inputs.files.filter((file) => file.owner.kind === "source");
  const headers = payload.inputs.files.filter((file) => file.owner.kind !== "source");
  const sourceBytes = sumBytes(sources);
  const headerBytes = sumBytes(headers);
  const diagnosticsSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-worker-diagnostics.v1",
    diagnostics: payload.diagnostics,
  }, { limits: ARTIFACT_TEST_DECODE_LIMITS });
  const countSeverity = (severity: string): WireU64 =>
    String(payload.diagnostics.filter((entry) => entry.severity === severity).length) as WireU64;
  const runtime = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(invocationRecord.runtimeAbi).manifest.body;
  const deployment = unwrapPreparedCppCuteBrowserFrontendProfile(profile).profile.deployment;
  return {
    schema: "browsergrad.compiler.cpp-cute.browser-worker-result",
    version: { major: 1, minor: 0 },
    invocationId: invocation.invocationId,
    invocationNonceSha256: invocationRecord.invocation.invocationNonceSha256,
    terminal: "completed",
    compileStatus: { code: 0, name: "artifact-ready" },
    artifact: {
      artifactId: fixture.artifact.artifactId,
      artifactHash: fixture.artifact.artifactHash,
      transportHash: fixture.artifact.transportHash,
      artifactBytesSha256: fixture.artifact.artifactBytesSha256,
      artifactByteLength: fixture.artifact.artifactByteLength,
    },
    openedInputs: {
      sourceSetSha256: fixture.artifact.sourceSetSha256,
      headerSetSha256: fixture.artifact.headerSetSha256,
      inputClosureSha256: fixture.artifact.inputClosureSha256,
      openedSourceFiles: wire(sources.length),
      openedSourceBytes: wire(sourceBytes),
      openedHeaderFiles: wire(headers.length),
      openedHeaderBytes: wire(headerBytes),
    },
    diagnostics: {
      diagnosticsSha256,
      count: wire(payload.diagnostics.length),
      remarks: countSeverity("remark"),
      notes: countSeverity("note"),
      warnings: countSeverity("warning"),
      errors: countSeverity("error"),
      fatals: countSeverity("fatal"),
    },
    resources: {
      wasmMemory: {
        initialPages: wire(deployment.compilerRuntime.memory.initialPages),
        peakPages: wire(deployment.compilerRuntime.memory.initialPages),
        finalPages: wire(deployment.compilerRuntime.memory.initialPages),
      },
      frontendWork: {
        includeDepth: wire(0),
        macroExpansions: wire(0),
        preprocessedTokens: wire(0),
        astNodes: wire(0),
        constexprSteps: wire(0),
        templateInstantiations: wire(0),
        templateDepth: wire(0),
      },
      emittedArtifact: {
        declarations: wire(payload.declarations.length),
        types: wire(payload.types.length),
        constants: wire(payload.constants.length),
        layouts: wire(payload.facts.filter((fact) => fact.kind === "affine-layout").length),
        tensors: wire(payload.facts.filter((fact) => fact.kind === "tensor").length),
        operations: wire(payload.facts.filter((fact) =>
          fact.kind !== "affine-layout" && fact.kind !== "tensor" && fact.kind !== "target-intrinsic").length),
        targetIntrinsics: wire(payload.facts.filter((fact) => fact.kind === "target-intrinsic").length),
        diagnostics: wire(payload.diagnostics.length),
      },
      vfs: {
        ceilingStatus: "enforced-runtime-abi-and-profile-ceilings",
        maxLiveFileHandles: wire(runtime.vfs.maxLiveFileHandles),
        maxSessionCalls: wire(runtime.vfs.maxSessionCalls),
        maxIndexedNodes: wire(deployment.compilerRuntime.virtualFileSystem.maxIndexedNodes),
        maxIndexLogicalByteLength: wire(deployment.compilerRuntime.virtualFileSystem.maxIndexLogicalByteLength),
        indexedNodes: wire(payload.inputs.files.length),
        indexLogicalByteLength: wire(0),
        totalSessionCalls: wire(payload.inputs.files.length * 3),
        statusCalls: wire(0),
        openCalls: wire(payload.inputs.files.length),
        readCalls: wire(payload.inputs.files.length),
        closeCalls: wire(payload.inputs.files.length),
        directoryCountCalls: wire(0),
        directoryEntryCalls: wire(0),
        peakLiveHandles: wire(1),
        logicalOpenedSourceByteLength: wire(sourceBytes),
        logicalOpenedInstalledVfsByteLength: wire(headerBytes),
        logicalOpenedTotalByteLength: wire(sourceBytes + headerBytes),
        peakLiveLogicalReservationByteLength: wire(Math.max(...payload.inputs.files.map((file) =>
          Number(wireIntegerToBigInt(file.byteLength))))),
      },
      resultBytesCopied: fixture.artifact.artifactByteLength,
    },
    outcome: fixture.artifact.outcome,
  };
}

async function createExecution(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
  fixture: ArtifactFixture,
  claimed: CppCuteBrowserWorkerResultV1,
): Promise<CppCuteBrowserWasmCompilerExecution> {
  const invocationRecord = unwrapPreparedCppCuteBrowserWorkerInvocation(invocation).invocation;
  const vfs = claimed.resources.vfs;
  const zeroAllocator = {
    currentLiveGlobalRequestedByteLength: wire(0),
    peakLiveGlobalRequestedByteLength: wire(0),
    cumulativeGlobalAllocatedRequestedByteLength: wire(0),
    cumulativeGlobalFreedRequestedByteLength: wire(0),
    successfulAllocationCount: wire(0),
    freeCount: wire(0),
    failedAllocationCount: wire(0),
  };
  const runtimeSample = (pages: WireU64) => ({
    wasmMemory: {
      source: "webassembly-memory-buffer-byte-length" as const,
      confidence: "exact" as const,
      pageByteLength: 65_536 as const,
      pages,
      linearMemoryCapacityByteLength: wire(wireIntegerToBigInt(pages) * 65_536n),
    },
    allocator: {
      source: "wasm-memory-allocator-metrics-record-v1" as const,
      confidence: "record-exact-unverified-producer" as const,
      values: zeroAllocator,
    },
  });
  return {
    authority: "wasm-c-abi-local-execution-only",
    protocol: CPP_CUTE_BROWSER_WASM_COMPILER_PROTOCOL,
    profileHash: invocation.profileHash,
    wasmSha256: invocationRecord.clangWasmSha256,
    wasmByteLength: Number(wireIntegerToBigInt(invocationRecord.clangWasmByteLength)),
    inputFrameByteLength: 0,
    resultByteLength: fixture.bytes.byteLength,
    compileStatus: { code: 0, name: "artifact-ready" },
    artifactBytes: fixture.bytes,
    runtime: {
      authority: "wasm-runtime-local-observation-only",
      profileHash: invocation.profileHash,
      initial: runtimeSample(claimed.resources.wasmMemory.initialPages),
      current: runtimeSample(claimed.resources.wasmMemory.finalPages),
      peakWasmMemoryPages: claimed.resources.wasmMemory.peakPages,
      phases: [],
      workerExecutionObserved: false,
      loweringAuthorityReady: false,
    },
    frontendWork: {
      authority: "wasm-frontend-work-local-observation-only",
      protocol: CPP_CUTE_BROWSER_FRONTEND_WORK_METRICS_PROTOCOL,
      profileHash: invocation.profileHash,
      source: "wasm-memory-frontend-work-metrics-record-v1",
      confidence: "record-exact-unverified-producer",
      generation: wire(1),
      values: {
        ...claimed.resources.frontendWork,
        completedSemanticPasses: wire(2),
      },
      resetConfirmed: true,
      workerExecutionObserved: false,
      loweringAuthorityReady: false,
    },
    vfs: {
      installationId: invocationRecord.vfsInstallationId,
      requestId: invocation.requestId,
      profileHash: invocation.profileHash,
      state: "disposed",
      counters: {
        totalSessionCalls: vfs.totalSessionCalls,
        statusCalls: vfs.statusCalls,
        openCalls: vfs.openCalls,
        readCalls: vfs.readCalls,
        closeCalls: vfs.closeCalls,
        directoryCountCalls: vfs.directoryCountCalls,
        directoryEntryCalls: vfs.directoryEntryCalls,
        currentLiveHandles: wire(0),
        peakLiveHandles: vfs.peakLiveHandles,
        currentLiveSourceLogicalReservationByteLength: wire(0),
        currentLiveInstalledVfsLogicalReservationByteLength: wire(0),
        currentLiveLogicalReservationByteLength: wire(0),
        peakLiveLogicalReservationByteLength: vfs.peakLiveLogicalReservationByteLength,
        indexedNodes: vfs.indexedNodes,
        indexLogicalByteLength: vfs.indexLogicalByteLength,
        logicalOpenedSourceByteLength: vfs.logicalOpenedSourceByteLength,
        logicalOpenedInstalledVfsByteLength: vfs.logicalOpenedInstalledVfsByteLength,
        logicalOpenedTotalByteLength: vfs.logicalOpenedTotalByteLength,
      },
      openedFiles: fixture.payload.inputs.files.map((file) => ({
        virtualPath: file.virtualPath,
        source: file.owner.kind === "source" ? "request-source" : "installed-pack",
        contentSha256: file.contentSha256,
        byteLength: file.byteLength,
      })),
      lookupMisses: {
        total: wire(0),
        uniquePaths: [],
        truncated: false,
      },
    },
    cAbiExecutionObserved: true,
    artifactVerificationObserved: false,
    workerExecutionObserved: false,
    loweringAuthorityMinted: false,
  };
}

function replaceString(value: unknown, before: string, after: string): void {
  if (Array.isArray(value)) {
    for (const [index, entry] of value.entries()) {
      if (entry === before) value[index] = after;
      else replaceString(entry, before, after);
    }
    return;
  }
  if (typeof value !== "object" || value === null) return;
  for (const [key, entry] of Object.entries(value)) {
    if (entry === before) (value as Record<string, unknown>)[key] = after;
    else replaceString(entry, before, after);
  }
}

function sumBytes(files: readonly { readonly byteLength: WireU64 }[]): number {
  return files.reduce((total, file) => total + Number(wireIntegerToBigInt(file.byteLength)), 0);
}

function wire(value: number | bigint): WireU64 {
  return String(value) as WireU64;
}

function hex(bytes: Uint8Array): string {
  return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function deferred(): { readonly promise: Promise<void>; readonly resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}
