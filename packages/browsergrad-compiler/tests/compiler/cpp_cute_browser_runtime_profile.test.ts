import { describe, expect, it } from "vitest";
import {
  CppCuteFrontendProfileError,
  prepareCppCuteFrontendProfile,
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

function expectProfileError(
  value: unknown,
  code: CppCuteFrontendProfileError["code"],
  path: string,
): Promise<void> {
  return expect(prepareCppCuteFrontendProfile(value)).rejects.toMatchObject({ code, path });
}

describe("browser C++/CuTe compiler-runtime profile", () => {
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
      "$.deployment.compilerRuntime.requiredWasmFeatures[3]",
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
      "$.deployment.compilerRuntime.memory.stackByteLength",
    );

    const aggregate = browserProfileInput();
    memory(aggregate)["maxCompilerWorkingByteLength"] = 700 * MIB;
    await expectProfileError(
      aggregate,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.extractionLimits.maxMemoryBytes",
    );
  });

  it("bounds retained host packs and opened WASM files by verified asset ceilings", async () => {
    const retained = browserProfileInput();
    virtualFileSystem(retained)["maxRetainedHostPackByteLength"] =
      Number(assetLimits(retained)["maxTotalUnpackedByteLength"]) + 1;
    await expectProfileError(
      retained,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.virtualFileSystem.maxRetainedHostPackByteLength",
    );

    const opened = browserProfileInput();
    virtualFileSystem(opened)["maxOpenedWasmFileByteLength"] =
      Number(assetLimits(opened)["maxTotalFileContentByteLength"]) + 1;
    await expectProfileError(
      opened,
      "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      "$.deployment.compilerRuntime.virtualFileSystem.maxOpenedWasmFileByteLength",
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
