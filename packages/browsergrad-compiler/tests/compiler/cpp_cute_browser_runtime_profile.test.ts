import { describe, expect, it } from "vitest";
import {
  CppCuteFrontendProfileError,
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteBrowserFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import { createCppCuteBrowserProfileInput } from "./support/cpp_cute_frontend_fixtures.js";

const MIB = 1024 * 1024;

function browserProfileInput(): Record<string, unknown> {
  return structuredClone(createCppCuteBrowserProfileInput()) as unknown as Record<string, unknown>;
}

function deployment(value: Record<string, unknown>): Record<string, unknown> {
  return value["deployment"] as Record<string, unknown>;
}

function runtime(value: Record<string, unknown>): Record<string, unknown> {
  return deployment(value)["compilerRuntime"] as Record<string, unknown>;
}

function memory(value: Record<string, unknown>): Record<string, unknown> {
  return runtime(value)["memory"] as Record<string, unknown>;
}

function virtualFileSystem(value: Record<string, unknown>): Record<string, unknown> {
  return runtime(value)["virtualFileSystem"] as Record<string, unknown>;
}

function worker(value: Record<string, unknown>): Record<string, unknown> {
  return deployment(value)["worker"] as Record<string, unknown>;
}

function assetLimits(value: Record<string, unknown>): Record<string, unknown> {
  return deployment(value)["assetLimits"] as Record<string, unknown>;
}

function extractionLimits(value: Record<string, unknown>): Record<string, unknown> {
  return value["extractionLimits"] as Record<string, unknown>;
}

function expectProfileError(
  value: unknown,
  code: CppCuteFrontendProfileError["code"],
  path: string,
): Promise<void> {
  return expect(prepareCppCuteFrontendProfile(value)).rejects.toMatchObject({ code, path });
}

describe("browser C++/CuTe compiler-runtime profile", () => {
  it("pins the canonical runtime-ABI manifest and fixed module shape", async () => {
    await expect(prepareCppCuteFrontendProfile(browserProfileInput())).resolves.toMatchObject({
      deploymentMode: "browser-local",
    });

    const missingManifest = browserProfileInput();
    delete runtime(missingManifest)["runtimeAbiManifestSha256"];
    await expectProfileError(
      missingManifest,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime",
    );

    for (const [mutate, path] of [
      [(value: Record<string, unknown>) => { runtime(value)["runtimeAbiId"] = "other-runtime@1"; },
        "$.deployment.compilerRuntime.runtimeAbiId"],
      [(value: Record<string, unknown>) => { runtime(value)["runtimeAbiManifestSha256"] = "0".repeat(64); },
        "$.deployment.compilerRuntime.runtimeAbiManifestSha256"],
      [(value: Record<string, unknown>) => { runtime(value)["wasmAddressBits"] = 64; },
        "$.deployment.compilerRuntime.wasmAddressBits"],
      [(value: Record<string, unknown>) => { memory(value)["initialPages"] = 2_048; },
        "$.deployment.compilerRuntime.memory.initialPages"],
      [(value: Record<string, unknown>) => { memory(value)["maximumPages"] = 8_192; },
        "$.deployment.compilerRuntime.memory.maximumPages"],
      [(value: Record<string, unknown>) => { memory(value)["stackByteLength"] = 8 * MIB; },
        "$.deployment.compilerRuntime.memory.stackByteLength"],
    ] as const) {
      const value = browserProfileInput();
      mutate(value);
      await expectProfileError(value, "BG-COMPILER-CPP-CUTE-PROFILE-INVALID", path);
    }
  });

  it("forbids worker-side module fetch and alternate handoff authority", async () => {
    const fetch = browserProfileInput();
    runtime(fetch)["workerSideFetch"] = "same-origin";
    await expectProfileError(
      fetch,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.workerSideFetch",
    );

    const handoff = browserProfileInput();
    runtime(handoff)["moduleHandoff"] = "worker-fetched-by-url";
    await expectProfileError(
      handoff,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.moduleHandoff",
    );
  });

  it("requires a sorted closed set of runtime-v1 WASM features", async () => {
    const unsorted = browserProfileInput();
    runtime(unsorted)["requiredWasmFeatures"] = [
      "mutable-globals",
      "bulk-memory",
      "nontrapping-fptoint",
      "sign-extension",
    ];
    await expectProfileError(
      unsorted,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.requiredWasmFeatures",
    );

    const unknown = browserProfileInput();
    runtime(unknown)["requiredWasmFeatures"] = [
      "bulk-memory",
      "mutable-globals",
      "nontrapping-fptoint",
      "reference-types",
      "sign-extension",
    ];
    await expectProfileError(
      unknown,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.requiredWasmFeatures",
    );

    const missing = browserProfileInput();
    runtime(missing)["requiredWasmFeatures"] = [
      "bulk-memory",
      "mutable-globals",
      "nontrapping-fptoint",
    ];
    await expectProfileError(
      missing,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.requiredWasmFeatures",
    );
  });

  it("requires worker-owned unshared linear memory", async () => {
    const shared = browserProfileInput();
    memory(shared)["sharing"] = "shared";
    await expectProfileError(
      shared,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.memory.sharing",
    );

    const hostOwned = browserProfileInput();
    memory(hostOwned)["ownership"] = "host";
    await expectProfileError(
      hostOwned,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.memory.ownership",
    );
  });

  it("rejects impossible initial and aggregate linear-memory reservations", async () => {
    const initial = browserProfileInput();
    memory(initial)["initialPages"] = 128;
    await expectProfileError(
      initial,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.memory.initialPages",
    );

    const aggregate = browserProfileInput();
    memory(aggregate)["maxCompilerWorkingByteLength"] = 700 * MIB;
    await expectProfileError(
      aggregate,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.memory.maxCompilerWorkingByteLength",
    );
  });

  it("reserves the live ABI input frame alongside stack, compiler, and result bytes", async () => {
    const belowFourTermSum = browserProfileInput();
    extractionLimits(belowFourTermSum)["maxMemoryBytes"] = 563 * MIB;
    await expectProfileError(
      belowFourTermSum,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.extractionLimits.maxMemoryBytes",
    );

    const exactFourTermSum = browserProfileInput();
    extractionLimits(exactFourTermSum)["maxMemoryBytes"] = 564 * MIB;
    await expect(prepareCppCuteFrontendProfile(exactFourTermSum)).resolves.toMatchObject({
      deploymentMode: "browser-local",
    });
  });

  it("allows profile-owned working, live-open, and logical-index ceilings to narrow ABI maxima", async () => {
    const narrowed = browserProfileInput();
    memory(narrowed)["maxCompilerWorkingByteLength"] = 400 * MIB;
    virtualFileSystem(narrowed)["maxAggregateLiveOpenByteLength"] = 300 * MIB;
    virtualFileSystem(narrowed)["maxIndexedNodes"] = 32_768;
    virtualFileSystem(narrowed)["maxIndexLogicalByteLength"] = 16 * MIB;

    const prepared = await prepareCppCuteFrontendProfile(narrowed);
    const profile = unwrapPreparedCppCuteBrowserFrontendProfile(prepared).profile;
    expect(prepared.deploymentMode).toBe("browser-local");
    expect(profile.deployment.compilerRuntime.memory.maxCompilerWorkingByteLength).toBe(400 * MIB);
    expect(profile.deployment.compilerRuntime.virtualFileSystem.maxAggregateLiveOpenByteLength).toBe(300 * MIB);
    expect(profile.deployment.compilerRuntime.virtualFileSystem.maxIndexedNodes).toBe(32_768);
    expect(profile.deployment.compilerRuntime.virtualFileSystem.maxIndexLogicalByteLength).toBe(16 * MIB);
    expect(profile.deployment.compilerRuntime.virtualFileSystem.maxRetainedHostPackByteLength).toBe(512 * MIB);
  });

  it("rejects compiler, live-open, logical-index, and result ceilings above ABI maxima", async () => {
    const working = browserProfileInput();
    memory(working)["maxCompilerWorkingByteLength"] = 512 * MIB + 1;
    await expectProfileError(
      working,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.memory.maxCompilerWorkingByteLength",
    );

    const opened = browserProfileInput();
    virtualFileSystem(opened)["maxAggregateLiveOpenByteLength"] = 384 * MIB + 1;
    await expectProfileError(
      opened,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.virtualFileSystem.maxAggregateLiveOpenByteLength",
    );

    const nodes = browserProfileInput();
    virtualFileSystem(nodes)["maxIndexedNodes"] = 262_145;
    await expectProfileError(
      nodes,
      "BG-COMPILER-CPP-CUTE-PROFILE-RESOURCE-LIMIT",
      "$.deployment.compilerRuntime.virtualFileSystem.maxIndexedNodes",
    );

    const indexBytes = browserProfileInput();
    virtualFileSystem(indexBytes)["maxIndexLogicalByteLength"] = 128 * MIB + 1;
    await expectProfileError(
      indexBytes,
      "BG-COMPILER-CPP-CUTE-PROFILE-RESOURCE-LIMIT",
      "$.deployment.compilerRuntime.virtualFileSystem.maxIndexLogicalByteLength",
    );

    const output = browserProfileInput();
    (output["extractionLimits"] as Record<string, unknown>)["maxOutputBytes"] = 32 * MIB + 1;
    await expectProfileError(
      output,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.extractionLimits.maxOutputBytes",
    );
  });

  it("bounds retained host packs and aggregate live-open bytes by verified asset ceilings", async () => {
    const retained = browserProfileInput();
    virtualFileSystem(retained)["maxRetainedHostPackByteLength"] =
      Number(assetLimits(retained)["maxTotalUnpackedByteLength"]) + 1;
    await expectProfileError(
      retained,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.virtualFileSystem.maxRetainedHostPackByteLength",
    );

    const opened = browserProfileInput();
    virtualFileSystem(opened)["maxAggregateLiveOpenByteLength"] =
      Number(assetLimits(opened)["maxTotalFileContentByteLength"]) + 1;
    await expectProfileError(
      opened,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.virtualFileSystem.maxAggregateLiveOpenByteLength",
    );
  });

  it("bounds exact worker module byte length", async () => {
    for (const moduleByteLength of [0, 64 * MIB + 1]) {
      const value = browserProfileInput();
      worker(value)["moduleByteLength"] = moduleByteLength;
      await expectProfileError(
        value,
        "BG-COMPILER-CPP-CUTE-PROFILE-RESOURCE-LIMIT",
        "$.deployment.worker.moduleByteLength",
      );
    }
  });
});
