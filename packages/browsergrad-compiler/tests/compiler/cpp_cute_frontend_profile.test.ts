import { describe, expect, it } from "vitest";
import {
  CppCuteFrontendProfileError,
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  cloneCppCuteProfileInput,
  CPP_CUTE_FIXTURE_CUDA_HEADER_HASH,
  CPP_CUTE_FIXTURE_CUTLASS_HEADER_HASH,
  CPP_CUTE_FIXTURE_HEADER_SET_HASH,
  createCppCuteProfileInput,
} from "./support/cpp_cute_frontend_fixtures.js";

function expectProfileError(
  value: unknown,
  code: CppCuteFrontendProfileError["code"],
  path: string,
): Promise<void> {
  return expect(prepareCppCuteFrontendProfile(value)).rejects.toMatchObject({ code, path });
}

describe("C++/CuTe frontend profile", () => {
  it("prepares one deterministic opaque profile", async () => {
    const first = await prepareCppCuteFrontendProfile(createCppCuteProfileInput());
    const second = await prepareCppCuteFrontendProfile(createCppCuteProfileInput());

    expect(first).toEqual(second);
    expect(first.profileHash).toBe("428edf03d9bc0005651fbcafdbad952fb7830b7ab73a15f0861c2900bd066717");
    expect(first.profileId).toBe("browsergrad.compiler.cpp-cute.layout-tracer@1");
    expect(first.deploymentMode).toBe("ahead-of-time");
    expect(first.expectedHeaderSetSha256).toBe(CPP_CUTE_FIXTURE_HEADER_SET_HASH);
    expect(Object.isFrozen(first)).toBe(true);
    expect(Object.isFrozen(first.extractionLimits)).toBe(true);
    expect(Object.isFrozen(unwrapPreparedCppCuteFrontendProfile(first).profile)).toBe(true);
  });

  it("rejects structural copies without profile authority", () => {
    const forged = {
      profileId: "browsergrad.compiler.cpp-cute.layout-tracer@1",
      profileHash: "0".repeat(64),
      deploymentMode: "ahead-of-time",
      expectedHeaderSetSha256: CPP_CUTE_FIXTURE_HEADER_SET_HASH,
      extractionLimits: createCppCuteProfileInput().extractionLimits,
    } as unknown as PreparedCppCuteFrontendProfile;

    expect(() => unwrapPreparedCppCuteFrontendProfile(forged)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-PROFILE-UNVERIFIED", path: "$" }),
    );
  });

  it("fails closed on unknown profile fields", async () => {
    const value = cloneCppCuteProfileInput();
    value["hostPath"] = "/Users/example/toolchain";
    await expectProfileError(value, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$");
  });

  it("rejects unsupported profile versions", async () => {
    const value = cloneCppCuteProfileInput();
    value["version"] = { major: 2, minor: 0 };
    await expectProfileError(value, "BG-COMPILER-CPP-CUTE-PROFILE-UNSUPPORTED-VERSION", "$.version.major");
  });

  it("requires sorted set-like profile fields", async () => {
    const value = cloneCppCuteProfileInput();
    const compatibility = value["compatibility"] as Record<string, unknown>;
    compatibility["supportedSourceFeatures"] = ["cxx:templates@1", "cuda:language@1"];
    await expectProfileError(value, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.compatibility.supportedSourceFeatures");
  });

  it("preserves order-sensitive compiler options and include roots", async () => {
    const value = cloneCppCuteProfileInput();
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
        manifestSha256: CPP_CUTE_FIXTURE_CUTLASS_HEADER_HASH,
      },
      {
        includeRootId: "cuda",
        mode: "system",
        virtualPath: "/toolchain/cuda/include",
        manifestSha256: CPP_CUTE_FIXTURE_CUDA_HEADER_HASH,
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
        manifestSha256: CPP_CUTE_FIXTURE_CUTLASS_HEADER_HASH,
      },
      {
        includeRootId: "cuda",
        mode: "system",
        virtualPath: "/toolchain/cuda/include",
        manifestSha256: CPP_CUTE_FIXTURE_CUDA_HEADER_HASH,
      },
    ]);
  });

  it("rejects path traversal and host-path syntax", async () => {
    const traversal = cloneCppCuteProfileInput();
    (traversal["virtualFileSystem"] as Record<string, unknown>)["sourceRoots"] = ["/workspace/../private"];
    await expectProfileError(traversal, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.virtualFileSystem.sourceRoots[0]");

    const windows = cloneCppCuteProfileInput();
    const includeRoots = (windows["virtualFileSystem"] as Record<string, unknown>)["includeRoots"] as Record<string, unknown>[];
    if (includeRoots[0] === undefined) throw new Error("fixture lost include root");
    includeRoots[0]["virtualPath"] = "C:\\cuda\\include";
    await expectProfileError(windows, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.virtualFileSystem.includeRoots[0].virtualPath");
  });

  it("requires canonical runner, container, trust, compiler, dependency, and header identities", async () => {
    const runner = cloneCppCuteProfileInput();
    const runnerDeployment = runner["deployment"] as Record<string, unknown>;
    (runnerDeployment["runner"] as Record<string, unknown>)["binarySha256"] = "not-a-digest";
    await expectProfileError(runner, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.deployment.runner.binarySha256");

    const container = cloneCppCuteProfileInput();
    const deployment = container["deployment"] as Record<string, unknown>;
    (deployment["container"] as Record<string, unknown>)["repository"] = "GHCR.io/example/image:latest";
    await expectProfileError(container, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.deployment.container.repository");

    const configDigest = cloneCppCuteProfileInput();
    const configDeployment = configDigest["deployment"] as Record<string, unknown>;
    (configDeployment["container"] as Record<string, unknown>)["configDigest"] = "sha256:ABC";
    await expectProfileError(
      configDigest,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.container.configDigest",
    );

    const executionEnvironment = cloneCppCuteProfileInput();
    const environmentDeployment = executionEnvironment["deployment"] as Record<string, unknown>;
    environmentDeployment["executionEnvironmentManifestSha256"] = "not-a-digest";
    await expectProfileError(
      executionEnvironment,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.executionEnvironmentManifestSha256",
    );

    const trust = cloneCppCuteProfileInput();
    const trustDeployment = trust["deployment"] as Record<string, unknown>;
    (trustDeployment["provenance"] as Record<string, unknown>)["trustStoreSha256"] = "not-a-digest";
    await expectProfileError(trust, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.deployment.provenance.trustStoreSha256");

    const builder = cloneCppCuteProfileInput();
    const builderDeployment = builder["deployment"] as Record<string, unknown>;
    (builderDeployment["provenance"] as Record<string, unknown>)["builderIds"] = ["github:workflow"];
    await expectProfileError(builder, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.deployment.provenance.builderIds[0]");

    const compilerDigest = cloneCppCuteProfileInput();
    const toolchain = compilerDigest["toolchain"] as Record<string, unknown>;
    (toolchain["compiler"] as Record<string, unknown>)["binarySha256"] = "ABC";
    await expectProfileError(compilerDigest, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.toolchain.compiler.binarySha256");

    const cutlassRevision = cloneCppCuteProfileInput();
    const dependencies = (cutlassRevision["toolchain"] as Record<string, unknown>)["dependencies"] as Record<string, unknown>[];
    const cutlass = dependencies.find((dependency) => dependency["kind"] === "cutlass");
    if (cutlass === undefined) throw new Error("fixture lost CUTLASS dependency");
    cutlass["revision"] = "v3.7.0";
    await expectProfileError(cutlassRevision, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.toolchain.dependencies[1].revision");
  });

  it("requires exactly one CUDA toolkit and CUTLASS dependency", async () => {
    const value = cloneCppCuteProfileInput();
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
    const value = cloneCppCuteProfileInput();
    const limits = value["extractionLimits"] as Record<string, unknown>;
    limits["maxOutputBytes"] = 64 * 1024 * 1024 + 1;
    await expectProfileError(value, "BG-COMPILER-CPP-CUTE-PROFILE-RESOURCE-LIMIT", "$.extractionLimits.maxOutputBytes");
  });

  it("rejects arbitrary, duplicate, or conflicting raw compiler options", async () => {
    const arbitrary = cloneCppCuteProfileInput();
    (arbitrary["language"] as Record<string, unknown>)["options"] = [
      { kind: "frontend-option", id: "load-plugin", value: "/host/plugin.so" },
    ];
    await expectProfileError(arbitrary, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.language.options[0].id");

    const conflict = cloneCppCuteProfileInput();
    (conflict["language"] as Record<string, unknown>)["options"] = [
      { kind: "define", name: "MODE", value: "1" },
      { kind: "undefine", name: "MODE" },
    ];
    await expectProfileError(conflict, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", "$.language.options[1]");
  });

  it("honors cancellation before and after hashing", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(prepareCppCuteFrontendProfile(createCppCuteProfileInput(), { signal: controller.signal })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROFILE-CANCELLED",
      path: "$.signal",
    });
  });

  it("rejects accessor-bearing in-memory input before reading fields", async () => {
    const value = cloneCppCuteProfileInput();
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
