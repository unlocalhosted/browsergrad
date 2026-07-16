import {
  canonicalJsonBytes,
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_BYTE_LIMIT,
  CppCuteAotExecutionEnvironmentError,
  computeCppCuteAotExecutionEnvironmentClosureHashes,
  copyPreparedCppCuteAotExecutionEnvironmentBytes,
  prepareCppCuteAotExecutionEnvironment,
  unwrapPreparedCppCuteAotExecutionEnvironment,
} from "../../src/cpp_cute_aot_environment.js";
import { prepareCppCuteFrontendProfile } from "../../src/cpp_cute_frontend_profile.js";
import {
  asMutableObject,
  canonicalEnvironmentBytes,
  cloneEnvironmentInput,
  createCppCuteAotExecutionEnvironmentFixture,
} from "./support/cpp_cute_aot_environment_fixtures.js";
import { createCppCuteProfileInput } from "./support/cpp_cute_frontend_fixtures.js";

describe("C++/CuTe execution-environment authority", () => {
  it("strictly authorizes one immutable canonical environment for one profile", async () => {
    const fixture = await createCppCuteAotExecutionEnvironmentFixture();
    expect(fixture.environment).toMatchObject({
      manifestId: "bg.cpp.execution-environment.sha256.770ae1cb99dc7802695a1e312b170ac7bded5828f77e986c86ba20cf94665f00",
      manifestSha256: "c4b4a24ad2baab471945c0133766036837e12a18743a86dc9406e9df723b6d42",
      manifestByteLength: "6588",
      bodySha256: "770ae1cb99dc7802695a1e312b170ac7bded5828f77e986c86ba20cf94665f00",
      profileHash: fixture.profile.profileHash,
    });
    const record = unwrapPreparedCppCuteAotExecutionEnvironment(fixture.environment);
    expect(record.profile).toBe(fixture.profile);
    expect(record.manifest.body.scope).toEqual({
      contractId: "browsergrad.compiler.cpp-cute.aot@1",
      sandboxPolicySha256: fixture.input.body.scope.sandboxPolicySha256,
      identity: "environment-only",
      runEvidence: "detached",
      isolation: "single-job-disposable-vm",
    });
    expect(Object.isFrozen(record.manifest)).toBe(true);
    expect(Object.isFrozen(record.manifest.body.toolchain.binaries)).toBe(true);
    expect(Object.isFrozen(record.manifest.body.image.layers[0])).toBe(true);
    expect(copyPreparedCppCuteAotExecutionEnvironmentBytes(fixture.environment)).toEqual(fixture.bytes);
  });

  it("returns only disposable byte copies and rejects structural authority copies", async () => {
    const fixture = await createCppCuteAotExecutionEnvironmentFixture();
    const first = copyPreparedCppCuteAotExecutionEnvironmentBytes(fixture.environment);
    first[0] = 0;
    expect(copyPreparedCppCuteAotExecutionEnvironmentBytes(fixture.environment)).toEqual(fixture.bytes);
    expect(() => unwrapPreparedCppCuteAotExecutionEnvironment({
      ...fixture.environment,
    })).toThrowError(CppCuteAotExecutionEnvironmentError);
  });

  it("rejects noncanonical, duplicate-key, unknown-field, and trailing JSON", async () => {
    const fixture = await createCppCuteAotExecutionEnvironmentFixture();
    const text = new TextDecoder().decode(fixture.bytes);
    const cases = [
      new TextEncoder().encode(`${text}\n`),
      new TextEncoder().encode(text.replace("{", "{\"schema\":\"duplicate\",")),
      new TextEncoder().encode(`${text}null`),
      canonicalJsonBytes({ ...fixture.input, unknown: true }),
    ];
    for (const bytes of cases) {
      await expect(prepareCppCuteAotExecutionEnvironment(fixture.profile, bytes)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-INVALID",
      });
    }
  });

  it("rejects manifest-ID drift before profile authorization", async () => {
    const fixture = await createCppCuteAotExecutionEnvironmentFixture();
    const input = cloneEnvironmentInput(fixture.input);
    (input as { manifestId: string }).manifestId = `bg.cpp.execution-environment.sha256.${"0".repeat(64)}`;
    await expect(prepareCppCuteAotExecutionEnvironment(
      fixture.profile,
      canonicalJsonBytes(input),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-HASH-MISMATCH",
      path: "$.manifestId",
    });
  });

  it("requires the profile to name the exact canonical resource", async () => {
    const first = await createCppCuteAotExecutionEnvironmentFixture();
    const second = await createCppCuteAotExecutionEnvironmentFixture({
      mutateBody(body) {
        asMutableObject(body.platform.kernel).buildId = "browsergrad-linux-amd64-2";
      },
    });
    await expect(prepareCppCuteAotExecutionEnvironment(
      first.profile,
      second.bytes,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-PROFILE-MISMATCH",
      path: "$.profile.deployment.executionEnvironmentManifestSha256",
    });
  });

  it("cross-binds policy, image, toolchain, headers, roots, and attestors", async () => {
    const base = await createCppCuteAotExecutionEnvironmentFixture();
    const mutations: Array<(input: typeof base.input) => void> = [
      (input) => { asMutableObject(input.body.scope).sandboxPolicySha256 = "f".repeat(64); },
      (input) => { asMutableObject(input.body.image).manifestDigest = `sha256:${"0".repeat(64)}`; },
      (input) => { asMutableObject(input.body.toolchain.binaries[0]).sha256 = "f".repeat(64); },
      (input) => {
        const headerSet = input.body.toolchain.headerSets[0];
        if (headerSet === undefined) throw new Error("fixture lost its first header set");
        asMutableObject(headerSet).headerSetSha256 = "f".repeat(64);
        const root = input.body.toolchain.includeRoots.find((entry) =>
          entry.owner.kind === "dependency" && entry.owner.dependencyId === headerSet.dependencyId);
        if (root === undefined) throw new Error("fixture lost its first dependency-owned include root");
        asMutableObject(root).manifestSha256 = "f".repeat(64);
      },
      (input) => { asMutableObject(input.body.toolchain.includeRoots[0]).manifestSha256 = "f".repeat(64); },
      (input) => { asMutableObject(input.body.attestation).trustStoreSha256 = "f".repeat(64); },
    ];
    for (const mutate of mutations) {
      const input = cloneEnvironmentInput(base.input);
      mutate(input);
      const closureHashes = await computeCppCuteAotExecutionEnvironmentClosureHashes(input.body);
      asMutableObject(input.body.image).rootfsManifestSha256 = closureHashes.rootfsManifestSha256;
      asMutableObject(input.body.toolchain).binariesManifestSha256 = closureHashes.binariesManifestSha256;
      asMutableObject(input.body.toolchain).dynamicLibrariesManifestSha256 = closureHashes.dynamicLibrariesManifestSha256;
      asMutableObject(input.body.toolchain).headersManifestSha256 = closureHashes.headersManifestSha256;
      const bytes = await canonicalEnvironmentBytes(input);
      const profile = await prepareCppCuteFrontendProfile(createCppCuteProfileInput({
        executionEnvironmentManifestSha256: await sha256Hex(bytes),
      }));
      await expect(prepareCppCuteAotExecutionEnvironment(profile, bytes)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-PROFILE-MISMATCH",
      });
    }
  });

  it("rejects invalid or ambiguous include-root ownership before profile authorization", async () => {
    const base = await createCppCuteAotExecutionEnvironmentFixture();

    const unknownOwner = cloneEnvironmentInput(base.input);
    asMutableObject(unknownOwner.body.toolchain.includeRoots[0]).owner = { kind: "host" };
    await expect(prepareCppCuteAotExecutionEnvironment(
      base.profile,
      await canonicalEnvironmentBytes(unknownOwner),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-INVALID",
      path: "$.body.toolchain.includeRoots[0].owner.kind",
    });

    const missingDependency = cloneEnvironmentInput(base.input);
    asMutableObject(missingDependency.body.toolchain.includeRoots[2]).owner = {
      kind: "dependency",
      dependencyId: "missing",
    };
    await expect(prepareCppCuteAotExecutionEnvironment(
      base.profile,
      await canonicalEnvironmentBytes(missingDependency),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-INVALID",
      path: "$.body.toolchain.includeRoots[2].owner.dependencyId",
    });

    const duplicatePath = cloneEnvironmentInput(base.input);
    const firstRoot = duplicatePath.body.toolchain.includeRoots[0];
    const secondRoot = duplicatePath.body.toolchain.includeRoots[1];
    if (firstRoot === undefined || secondRoot === undefined) throw new Error("fixture lost include roots");
    asMutableObject(secondRoot).virtualPath = firstRoot.virtualPath;
    await expect(prepareCppCuteAotExecutionEnvironment(
      base.profile,
      await canonicalEnvironmentBytes(duplicatePath),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-INVALID",
      path: "$.body.toolchain.includeRoots[1].virtualPath",
    });
  });

  it("rejects valid owner or semantic-adapter drift from the exact prepared profile", async () => {
    const base = await createCppCuteAotExecutionEnvironmentFixture();
    const cases: Array<{
      readonly input: typeof base.input;
      readonly path: string;
    }> = [];

    const ownerDrift = cloneEnvironmentInput(base.input);
    const cudaRoot = ownerDrift.body.toolchain.includeRoots.find((root) =>
      root.owner.kind === "dependency" && root.owner.dependencyId === "cuda");
    const cutlassRoot = ownerDrift.body.toolchain.includeRoots.find((root) =>
      root.owner.kind === "dependency" && root.owner.dependencyId === "cutlass");
    const cudaHeaders = ownerDrift.body.toolchain.headerSets.find((headerSet) => headerSet.dependencyId === "cuda");
    const cutlassHeaders = ownerDrift.body.toolchain.headerSets.find((headerSet) => headerSet.dependencyId === "cutlass");
    if (cudaRoot === undefined || cutlassRoot === undefined || cudaHeaders === undefined || cutlassHeaders === undefined) {
      throw new Error("fixture lost CUDA/CUTLASS ownership records");
    }
    asMutableObject(cudaRoot).owner = { kind: "dependency", dependencyId: "cutlass" };
    asMutableObject(cudaRoot).manifestSha256 = cutlassHeaders.headerSetSha256;
    asMutableObject(cutlassRoot).owner = { kind: "dependency", dependencyId: "cuda" };
    asMutableObject(cutlassRoot).manifestSha256 = cudaHeaders.headerSetSha256;
    cases.push({ input: ownerDrift, path: "$.body.toolchain.includeRoots" });

    const adapterDrift = cloneEnvironmentInput(base.input);
    asMutableObject(adapterDrift.body.toolchain).semanticAdapterManifestSha256 = "f".repeat(64);
    cases.push({
      input: adapterDrift,
      path: "$.body.toolchain.semanticAdapterManifestSha256",
    });

    for (const { input, path } of cases) {
      const closureHashes = await computeCppCuteAotExecutionEnvironmentClosureHashes(input.body);
      asMutableObject(input.body.toolchain).headersManifestSha256 = closureHashes.headersManifestSha256;
      const bytes = await canonicalEnvironmentBytes(input);
      const profile = await prepareCppCuteFrontendProfile(createCppCuteProfileInput({
        executionEnvironmentManifestSha256: await sha256Hex(bytes),
      }));
      await expect(prepareCppCuteAotExecutionEnvironment(profile, bytes)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-PROFILE-MISMATCH",
        path,
      });
    }
  });

  it("requires canonical closed inventories and real enforcement declarations", async () => {
    const base = await createCppCuteAotExecutionEnvironmentFixture();
    const mutations: Array<(input: typeof base.input) => void> = [
      (input) => { asMutableObject(input.body.scope).isolation = "shared-host"; },
      (input) => { asMutableObject(input.body.platform.runnerIdentity).coreDumps = "enabled"; },
      (input) => { asMutableObject(input.body.platform.cgroup).controllers = ["memory", "cpu", "pids"]; },
      (input) => { asMutableObject(input.body.platform.lsm[0]).enforcing = false; },
      (input) => { asMutableObject(input.body.runtime.seccomp).mode = "disabled"; },
      (input) => {
        asMutableObject(input.body.toolchain).binaries = [
          input.body.toolchain.binaries[1],
          input.body.toolchain.binaries[0],
          input.body.toolchain.binaries[2],
        ];
      },
      (input) => {
        asMutableObject(input.body.toolchain).dynamicLibraries = [
          input.body.toolchain.dynamicLibraries[0],
          structuredClone(input.body.toolchain.dynamicLibraries[0]),
        ];
      },
    ];
    for (const mutate of mutations) {
      const input = cloneEnvironmentInput(base.input);
      mutate(input);
      await expect(prepareCppCuteAotExecutionEnvironment(
        base.profile,
        await canonicalEnvironmentBytes(input),
      )).rejects.toBeInstanceOf(CppCuteAotExecutionEnvironmentError);
    }
  });

  it("recomputes every inline closure identity", async () => {
    const base = await createCppCuteAotExecutionEnvironmentFixture();
    for (const path of [
      ["image", "rootfsManifestSha256"],
      ["toolchain", "binariesManifestSha256"],
      ["toolchain", "dynamicLibrariesManifestSha256"],
      ["toolchain", "headersManifestSha256"],
    ] as const) {
      const input = cloneEnvironmentInput(base.input);
      const parent = path[0] === "image" ? input.body.image : input.body.toolchain;
      asMutableObject(parent)[path[1]] = "f".repeat(64);
      const bytes = await canonicalEnvironmentBytes(input);
      const profile = await prepareCppCuteFrontendProfile(createCppCuteProfileInput({
        executionEnvironmentManifestSha256: await sha256Hex(bytes),
      }));
      await expect(prepareCppCuteAotExecutionEnvironment(profile, bytes)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-HASH-MISMATCH",
      });
    }
  });

  it("rejects unsupported versions and resource amplification", async () => {
    const fixture = await createCppCuteAotExecutionEnvironmentFixture();
    const version = cloneEnvironmentInput(fixture.input);
    asMutableObject(version).version = { major: 1, minor: 0 };
    await expect(prepareCppCuteAotExecutionEnvironment(
      fixture.profile,
      canonicalJsonBytes(version),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-UNSUPPORTED-VERSION",
    });
    await expect(prepareCppCuteAotExecutionEnvironment(
      fixture.profile,
      new Uint8Array(CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_BYTE_LIMIT + 1),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-RESOURCE-LIMIT",
    });
  });

  it("snapshots plain unshared bytes and rejects hostile byte containers", async () => {
    const fixture = await createCppCuteAotExecutionEnvironmentFixture();
    class DerivedBytes extends Uint8Array {}
    const shared = new Uint8Array(new SharedArrayBuffer(fixture.bytes.byteLength));
    shared.set(fixture.bytes);
    for (const hostile of [
      new DerivedBytes(fixture.bytes),
      shared,
      new Proxy(fixture.bytes, {}),
      new Uint16Array(new ArrayBuffer(2)),
    ]) {
      await expect(prepareCppCuteAotExecutionEnvironment(
        fixture.profile,
        hostile as never,
      )).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-INVALID",
        path: "$bytes",
      });
    }
  });

  it("checks cancellation and closed options without granting authority", async () => {
    const fixture = await createCppCuteAotExecutionEnvironmentFixture();
    const controller = new AbortController();
    controller.abort();
    await expect(prepareCppCuteAotExecutionEnvironment(
      fixture.profile,
      fixture.bytes,
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-CANCELLED",
    });
    await expect(prepareCppCuteAotExecutionEnvironment(
      fixture.profile,
      fixture.bytes,
      { extra: true } as never,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-INVALID",
      path: "$options",
    });
  });
});
