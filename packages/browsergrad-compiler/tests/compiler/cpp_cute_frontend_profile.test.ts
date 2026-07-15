import { describe, expect, it } from "vitest";
import {
  CppCuteFrontendProfileError,
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteFrontendProfile,
  type CppCuteFrontendProfileV1,
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";

const HEADER_SET_HASH = "2".repeat(64);
const COMPILER_HASH = "3".repeat(64);
const CONTAINER_DIGEST = `sha256:${"4".repeat(64)}`;
const CUDA_HEADER_HASH = "5".repeat(64);
const CUTLASS_HEADER_HASH = "6".repeat(64);
const CUTLASS_REVISION = "7".repeat(40);

function profileFixture(): CppCuteFrontendProfileV1 {
  return {
    schema: "browsergrad.compiler.cpp-cute.frontend-profile",
    version: { major: 1, minor: 0 },
    profileId: "browsergrad.compiler.cpp-cute.layout-tracer@1",
    deployment: {
      mode: "ahead-of-time",
      extractor: {
        id: "browsergrad-tools/cpp-cute-frontend",
        version: "0.1.0",
        buildId: "browsergrad-cpp-cute-extractor-0.1.0",
        binarySha256: "a".repeat(64),
      },
      provenance: {
        kind: "external-attestation",
        predicateType: "https://slsa.dev/provenance/v1",
        builderIds: ["https://github.com/unlocalhosted/browsergrad/.github/workflows/cpp-cute-aot.yml"],
      },
    },
    language: {
      cxxStandard: "c++17",
      cudaCompatibility: "12.6",
      options: [
        { kind: "define", name: "CUTE_SM80_ENABLED", value: "1" },
        { kind: "frontend-option", id: "cuda-host-only", value: null },
        { kind: "frontend-option", id: "syntax-only", value: null },
      ],
    },
    target: {
      hostTriple: "x86_64-unknown-linux-gnu",
      deviceArchitecture: "sm_80",
      endianness: "little",
      pointerBits: 64,
    },
    toolchain: {
      compiler: {
        id: "clang",
        version: "20.1.8",
        buildId: "llvmorg-20.1.8",
        binarySha256: COMPILER_HASH,
        containerImageDigest: CONTAINER_DIGEST,
        resourceDirectorySha256: "b".repeat(64),
      },
      dependencies: [
        {
          dependencyId: "cuda",
          kind: "cuda-toolkit",
          version: "12.6.3",
          revision: "12.6.3",
          headerSetSha256: CUDA_HEADER_HASH,
        },
        {
          dependencyId: "cutlass",
          kind: "cutlass",
          version: "3.7.0",
          revision: CUTLASS_REVISION,
          headerSetSha256: CUTLASS_HEADER_HASH,
        },
      ],
    },
    virtualFileSystem: {
      sourceRoots: ["/workspace/src"],
      includeRoots: [
        {
          includeRootId: "cuda",
          mode: "system",
          virtualPath: "/toolchain/cuda/include",
          manifestSha256: CUDA_HEADER_HASH,
        },
        {
          includeRootId: "cutlass",
          mode: "system",
          virtualPath: "/toolchain/cutlass/include",
          manifestSha256: CUTLASS_HEADER_HASH,
        },
      ],
    },
    compatibility: {
      expectedHeaderSetSha256: HEADER_SET_HASH,
      supportedSourceFeatures: [
        "cuda:language@1",
        "cute:layout-algebra@1",
        "cxx:templates@1",
      ],
      unsupportedIntrinsicFamilies: [
        "nvidia:tma@1",
        "nvidia:wgmma@1",
      ],
    },
    extractionLimits: {
      maxSourceFiles: 8,
      maxSourceBytes: 1_048_576,
      maxHeaderFiles: 20_000,
      maxHeaderBytes: 268_435_456,
      maxIncludeDepth: 256,
      maxMacroExpansions: 1_000_000,
      maxAstNodes: 5_000_000,
      maxTemplateInstantiations: 1_000_000,
      maxTemplateDepth: 1_024,
      maxDeclarations: 1_000_000,
      maxTypes: 1_000_000,
      maxConstants: 1_000_000,
      maxLayouts: 100_000,
      maxTensors: 100_000,
      maxOperations: 1_000_000,
      maxTargetIntrinsics: 100_000,
      maxDiagnostics: 100_000,
      maxOutputBytes: 8_388_608,
      maxWallTimeMs: 120_000,
      maxCpuTimeMs: 120_000,
      maxMemoryBytes: 4_294_967_296,
      maxProcesses: 32,
    },
  };
}

function cloneFixture(): Record<string, unknown> {
  return structuredClone(profileFixture()) as unknown as Record<string, unknown>;
}

function expectProfileError(
  value: unknown,
  code: CppCuteFrontendProfileError["code"],
  path: string,
): Promise<void> {
  return expect(prepareCppCuteFrontendProfile(value)).rejects.toMatchObject({ code, path });
}

describe("C++/CuTe frontend profile", () => {
  it("prepares one deterministic opaque profile", async () => {
    const first = await prepareCppCuteFrontendProfile(profileFixture());
    const second = await prepareCppCuteFrontendProfile(profileFixture());

    expect(first).toEqual(second);
    expect(first.profileHash).toBe("5585fc471477bd96710a27a13aa80b16be2e0bc23bd8e429b9dd826febd024e5");
    expect(first.profileId).toBe("browsergrad.compiler.cpp-cute.layout-tracer@1");
    expect(first.deploymentMode).toBe("ahead-of-time");
    expect(first.expectedHeaderSetSha256).toBe(HEADER_SET_HASH);
    expect(Object.isFrozen(first)).toBe(true);
    expect(Object.isFrozen(first.extractionLimits)).toBe(true);
    expect(Object.isFrozen(unwrapPreparedCppCuteFrontendProfile(first).profile)).toBe(true);
  });

  it("rejects structural copies without profile authority", () => {
    const forged = {
      profileId: "browsergrad.compiler.cpp-cute.layout-tracer@1",
      profileHash: "0".repeat(64),
      deploymentMode: "ahead-of-time",
      expectedHeaderSetSha256: HEADER_SET_HASH,
      extractionLimits: profileFixture().extractionLimits,
    } as unknown as PreparedCppCuteFrontendProfile;

    expect(() => unwrapPreparedCppCuteFrontendProfile(forged)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-PROFILE-UNVERIFIED", path: "$" }),
    );
  });

  it("fails closed on unknown profile fields", async () => {
    const value = cloneFixture();
    value["hostPath"] = "/Users/example/toolchain";
    await expectProfileError(value, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$");
  });

  it("rejects unsupported profile versions", async () => {
    const value = cloneFixture();
    value["version"] = { major: 2, minor: 0 };
    await expectProfileError(value, "BG-COMPILER-CPP-CUTE-PROFILE-UNSUPPORTED-VERSION", "$.version.major");
  });

  it("requires sorted set-like profile fields", async () => {
    const value = cloneFixture();
    const compatibility = value["compatibility"] as Record<string, unknown>;
    compatibility["supportedSourceFeatures"] = ["cxx:templates@1", "cuda:language@1"];
    await expectProfileError(value, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.compatibility.supportedSourceFeatures");
  });

  it("preserves order-sensitive compiler options and include roots", async () => {
    const value = cloneFixture();
    const language = value["language"] as Record<string, unknown>;
    language["options"] = [
      { kind: "frontend-option", id: "syntax-only", value: null },
      { kind: "define", name: "CUTE_SM80_ENABLED", value: "1" },
    ];
    const vfs = value["virtualFileSystem"] as Record<string, unknown>;
    vfs["includeRoots"] = [
      {
        includeRootId: "cutlass",
        mode: "system",
        virtualPath: "/toolchain/cutlass/include",
        manifestSha256: CUTLASS_HEADER_HASH,
      },
      {
        includeRootId: "cuda",
        mode: "system",
        virtualPath: "/toolchain/cuda/include",
        manifestSha256: CUDA_HEADER_HASH,
      },
    ];

    const prepared = await prepareCppCuteFrontendProfile(value);
    const profile = unwrapPreparedCppCuteFrontendProfile(prepared).profile;
    expect(profile.language.options).toEqual([
      { kind: "frontend-option", id: "syntax-only", value: null },
      { kind: "define", name: "CUTE_SM80_ENABLED", value: "1" },
    ]);
    expect(profile.virtualFileSystem.includeRoots).toEqual([
      {
        includeRootId: "cutlass",
        mode: "system",
        virtualPath: "/toolchain/cutlass/include",
        manifestSha256: CUTLASS_HEADER_HASH,
      },
      {
        includeRootId: "cuda",
        mode: "system",
        virtualPath: "/toolchain/cuda/include",
        manifestSha256: CUDA_HEADER_HASH,
      },
    ]);
  });

  it("rejects path traversal and host-path syntax", async () => {
    const traversal = cloneFixture();
    (traversal["virtualFileSystem"] as Record<string, unknown>)["sourceRoots"] = ["/workspace/../private"];
    await expectProfileError(traversal, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.virtualFileSystem.sourceRoots[0]");

    const windows = cloneFixture();
    const includeRoots = (windows["virtualFileSystem"] as Record<string, unknown>)["includeRoots"] as Record<string, unknown>[];
    if (includeRoots[0] === undefined) throw new Error("fixture lost include root");
    includeRoots[0]["virtualPath"] = "C:\\cuda\\include";
    await expectProfileError(windows, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.virtualFileSystem.includeRoots[0].virtualPath");
  });

  it("requires exact compiler, container, dependency, and header identities", async () => {
    const compilerDigest = cloneFixture();
    const toolchain = compilerDigest["toolchain"] as Record<string, unknown>;
    (toolchain["compiler"] as Record<string, unknown>)["binarySha256"] = "ABC";
    await expectProfileError(compilerDigest, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.toolchain.compiler.binarySha256");

    const cutlassRevision = cloneFixture();
    const dependencies = (cutlassRevision["toolchain"] as Record<string, unknown>)["dependencies"] as Record<string, unknown>[];
    const cutlass = dependencies.find((dependency) => dependency["kind"] === "cutlass");
    if (cutlass === undefined) throw new Error("fixture lost CUTLASS dependency");
    cutlass["revision"] = "v3.7.0";
    await expectProfileError(cutlassRevision, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.toolchain.dependencies[1].revision");
  });

  it("requires exactly one CUDA toolkit and CUTLASS dependency", async () => {
    const value = cloneFixture();
    const toolchain = value["toolchain"] as Record<string, unknown>;
    toolchain["dependencies"] = [
      ...toolchain["dependencies"] as unknown[],
      {
        dependencyId: "cutlass_shadow",
        kind: "cutlass",
        version: "3.7.0",
        revision: "8".repeat(40),
        headerSetSha256: "9".repeat(64),
      },
    ];
    await expectProfileError(value, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.toolchain.dependencies");
  });

  it("caps every extraction resource budget", async () => {
    const value = cloneFixture();
    const limits = value["extractionLimits"] as Record<string, unknown>;
    limits["maxOutputBytes"] = 64 * 1024 * 1024 + 1;
    await expectProfileError(value, "BG-COMPILER-CPP-CUTE-PROFILE-RESOURCE-LIMIT", "$.extractionLimits.maxOutputBytes");
  });

  it("rejects arbitrary, duplicate, or conflicting raw compiler options", async () => {
    const arbitrary = cloneFixture();
    (arbitrary["language"] as Record<string, unknown>)["options"] = [
      { kind: "frontend-option", id: "load-plugin", value: "/host/plugin.so" },
    ];
    await expectProfileError(arbitrary, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.language.options[0].id");

    const conflict = cloneFixture();
    (conflict["language"] as Record<string, unknown>)["options"] = [
      { kind: "define", name: "MODE", value: "1" },
      { kind: "undefine", name: "MODE" },
    ];
    await expectProfileError(conflict, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.language.options[1]");
  });

  it("honors cancellation before and after hashing", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(prepareCppCuteFrontendProfile(profileFixture(), { signal: controller.signal })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROFILE-CANCELLED",
      path: "$.signal",
    });
  });

  it("rejects accessor-bearing in-memory input before reading fields", async () => {
    const value = cloneFixture();
    Object.defineProperty(value, "profileId", {
      enumerable: true,
      get: () => "browsergrad.compiler.cpp-cute.evil@1",
    });
    await expect(prepareCppCuteFrontendProfile(value)).rejects.toMatchObject({
      diagnostic: {
        code: "BG-SCHEMA-NONCANONICAL-VALUE",
        path: "$.profileId",
      },
    });
  });
});
