import { describe, expect, it } from "vitest";
import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_CONTRACT_SHA256,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
} from "../../src/cpp_cute_browser_runtime_abi.js";
import {
  CppCuteBrowserWasmVerifierEntryError,
  handleCppCuteBrowserWasmVerifierLaunch,
  installCppCuteBrowserWasmVerifierEntry,
  type CppCuteBrowserWasmVerifierEntryMessageListener,
  type CppCuteBrowserWasmVerifierEntryScope,
} from "../../src/cpp_cute_browser_wasm_verifier_entry.js";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
  CPP_CUTE_BROWSER_WASM_VERIFIER_MINOR,
  CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
  type CppCuteBrowserWasmVerifierLaunchMessage,
  type CppCuteBrowserWasmVerifierTerminalMessage,
} from "../../src/cpp_cute_browser_wasm_verifier_messages.js";

const EMPTY_WASM = new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
const REQUEST_ID = `bg.cpp.browser-wasm-verifier-request.sha256.${"a".repeat(64)}`;
const NONCE = "b".repeat(64);

async function launch(): Promise<CppCuteBrowserWasmVerifierLaunchMessage> {
  return {
    kind: "browsergrad-cpp-cute-wasm-verifier-launch",
    version: {
      major: CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
      minor: CPP_CUTE_BROWSER_WASM_VERIFIER_MINOR,
    },
    protocol: CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
    requestId: REQUEST_ID,
    invocationNonceSha256: NONCE,
    wasmAssetId: "clang-wasm",
    expectedWasmSha256: await sha256Hex(EMPTY_WASM),
    expectedWasmByteLength: EMPTY_WASM.byteLength,
    expectedRuntimeAbiManifestId: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
    expectedRuntimeAbiContractSha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_CONTRACT_SHA256,
    expectedRuntimeAbiResourceSha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
    maxOperations: 8_000_000,
    runtimeAbiManifestBytes: cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    wasmBytes: EMPTY_WASM,
  };
}

describe("disposable raw-Wasm verifier Worker entry", () => {
  it("reports ABI mismatch as a typed non-authoritative terminal failure", async () => {
    const terminal: CppCuteBrowserWasmVerifierTerminalMessage[] = [];
    const transfers: Array<readonly ArrayBuffer[]> = [];
    await handleCppCuteBrowserWasmVerifierLaunch(
      await launch(),
      (message, transfer) => {
        terminal.push(message);
        transfers.push(transfer);
      },
    );
    expect(terminal).toHaveLength(1);
    expect(terminal[0]).toMatchObject({
      kind: "browsergrad-cpp-cute-wasm-verifier-failure",
      requestId: REQUEST_ID,
      invocationNonceSha256: NONCE,
      phase: "raw-wasm",
      failureCode: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-ABI-MISMATCH",
      rawWasmVerified: false,
      verifierWorkerSelfAttested: false,
      productionConformanceAuthorityMinted: false,
      releaseReady: false,
    });
    expect(transfers).toEqual([[]]);
  });

  it("fails before a trusted terminal identity on malformed launch envelopes", async () => {
    const valid = await launch();
    await expect(handleCppCuteBrowserWasmVerifierLaunch(
      { ...valid, protocol: "caller-protocol" },
      () => { throw new Error("must not emit"); },
    )).rejects.toBeInstanceOf(CppCuteBrowserWasmVerifierEntryError);
    await expect(handleCppCuteBrowserWasmVerifierLaunch(
      { ...valid, expectedWasmByteLength: valid.expectedWasmByteLength + 1 },
      () => { throw new Error("must not emit"); },
    )).rejects.toBeInstanceOf(CppCuteBrowserWasmVerifierEntryError);
  });

  it("binds the exact runtime-ABI resource identity inside the Worker", async () => {
    const value = await launch();
    const terminals: CppCuteBrowserWasmVerifierTerminalMessage[] = [];
    await handleCppCuteBrowserWasmVerifierLaunch(
      { ...value, expectedRuntimeAbiResourceSha256: "f".repeat(64) },
      (message) => terminals.push(message),
    );
    expect(terminals[0]).toMatchObject({
      kind: "browsergrad-cpp-cute-wasm-verifier-failure",
      phase: "runtime-abi",
      rawWasmVerified: false,
    });
  });

  it("rejects callable getters without invoking them", () => {
    let getterReads = 0;
    const scope = Object.create(null) as Record<string, unknown>;
    Object.defineProperties(scope, {
      addEventListener: { enumerable: true, value: () => undefined },
      removeEventListener: { enumerable: true, value: () => undefined },
      postMessage: {
        enumerable: true,
        get: () => {
          getterReads += 1;
          return () => undefined;
        },
      },
      queueMicrotask: { enumerable: true, value: () => undefined },
    });
    expect(() => installCppCuteBrowserWasmVerifierEntry(
      scope as unknown as CppCuteBrowserWasmVerifierEntryScope,
    )).toThrow(/getter-free callable data property/u);
    expect(getterReads).toBe(0);
  });

  it("uses one entry-scope effect snapshot after the scope is changed", async () => {
    let listener: CppCuteBrowserWasmVerifierEntryMessageListener | undefined;
    const terminals: CppCuteBrowserWasmVerifierTerminalMessage[] = [];
    const scope: CppCuteBrowserWasmVerifierEntryScope = {
      addEventListener: (_type, nextListener) => {
        listener = nextListener;
        Object.defineProperty(scope, "postMessage", {
          configurable: true,
          enumerable: true,
          value: () => { throw new Error("changed postMessage must not run"); },
        });
      },
      removeEventListener: () => undefined,
      postMessage: (message) => { terminals.push(message); },
      queueMicrotask: (callback) => callback(),
    };
    installCppCuteBrowserWasmVerifierEntry(scope);
    listener?.({ data: await launch() });
    for (let attempt = 0; attempt < 20 && terminals.length === 0; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    expect(terminals).toHaveLength(1);
    expect(terminals[0]?.kind).toBe("browsergrad-cpp-cute-wasm-verifier-failure");
  });
});
