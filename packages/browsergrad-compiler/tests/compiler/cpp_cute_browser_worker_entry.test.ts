import { beforeEach, describe, expect, it, vi } from "vitest";

const runtime = vi.hoisted(() => ({
  reconstruct: vi.fn(),
  inspect: vi.fn(),
  discard: vi.fn(),
  prepare: vi.fn(),
  start: vi.fn(),
}));

vi.mock("../../src/cpp_cute_browser_worker_transfer.js", () => ({
  reconstructCppCuteBrowserWorkerTransfer: runtime.reconstruct,
  inspectCppCuteBrowserWorkerRealmInput: runtime.inspect,
  discardCppCuteBrowserWorkerRealmInput: runtime.discard,
}));

vi.mock("../../src/cpp_cute_browser_worker_runtime.js", () => ({
  prepareCppCuteBrowserWorkerRuntimeBinding: runtime.prepare,
  startCppCuteBrowserWorkerRuntime: runtime.start,
}));

import {
  CppCuteBrowserWorkerEntryError,
  handleCppCuteBrowserWorkerTransfer,
} from "../../src/cpp_cute_browser_worker_entry.js";
import type {
  CppCuteBrowserWorkerControllerInboundMessage,
} from "../../src/cpp_cute_browser_worker_messages.js";

const INVOCATION_ID = `bg.cpp.browser-worker-invocation.sha256.${"a".repeat(64)}`;
const NONCE = "b".repeat(64);
const REALM_INPUT = Object.freeze({
  invocationId: INVOCATION_ID,
  invocationNonceSha256: NONCE,
});

beforeEach(() => {
  runtime.reconstruct.mockReset();
  runtime.inspect.mockReset();
  runtime.discard.mockReset();
  runtime.prepare.mockReset();
  runtime.start.mockReset();
  runtime.reconstruct.mockResolvedValue(REALM_INPUT);
  runtime.inspect.mockReturnValue(Object.freeze({ state: "adopted" }));
  runtime.prepare.mockResolvedValue(Object.freeze({ binding: true }));
});

describe("C++/CuTe browser Worker entry", () => {
  it("reports a known start failure without adding execution claims", async () => {
    runtime.start.mockRejectedValue(Object.assign(new Error("blocked"), {
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CAPABILITY",
      path: "$.bundle",
    }));
    const terminal: CppCuteBrowserWorkerControllerInboundMessage[] = [];

    await handleCppCuteBrowserWorkerTransfer({} as never, (message) => terminal.push(message));

    expect(terminal).toEqual([{
      kind: "browsergrad-cpp-cute-worker-failure",
      version: 1,
      controllerProtocol: "browsergrad.compiler.cpp-cute.browser-worker-controller@1",
      invocationId: INVOCATION_ID,
      invocationNonceSha256: NONCE,
      phase: "runtime-start",
      failureCode: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CAPABILITY",
      failurePath: "$.bundle",
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    }]);
  });

  it("reports the deepest authenticated runtime failure across the Worker boundary", async () => {
    const compileFailure = Object.assign(new Error("producer failed"), {
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-COMPILE-STATUS",
      path: "$.runtime.compile",
    });
    runtime.start.mockRejectedValue(Object.assign(
      new Error("runtime execution failed", { cause: compileFailure }),
      {
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-EXECUTION",
        path: "$.runtime.execution",
      },
    ));
    const terminal: CppCuteBrowserWorkerControllerInboundMessage[] = [];

    await handleCppCuteBrowserWorkerTransfer({} as never, (message) => terminal.push(message));

    expect(terminal[0]).toMatchObject({
      phase: "runtime-start",
      failureCode: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-COMPILE-STATUS",
      failurePath: "$.runtime.compile",
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
  });

  it("sanitizes unknown adoption failures instead of leaking attacker-controlled fields", async () => {
    runtime.inspect.mockReturnValue(Object.freeze({ state: "prepared" }));
    runtime.prepare.mockRejectedValue(Object.freeze({
      code: "attacker-code",
      path: "$\u0000secret",
      message: "must not cross Worker boundary",
    }));
    const terminal: CppCuteBrowserWorkerControllerInboundMessage[] = [];

    await handleCppCuteBrowserWorkerTransfer({} as never, (message) => terminal.push(message));

    expect(terminal[0]).toMatchObject({
      phase: "runtime-adoption",
      failureCode: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-ENTRY-INTERNAL",
      failurePath: "$.runtime.adoption",
    });
    expect(terminal[0]).not.toHaveProperty("message");
    expect(runtime.discard).toHaveBeenCalledTimes(1);
    expect(runtime.discard).toHaveBeenCalledWith(REALM_INPUT);
    expect(runtime.start).not.toHaveBeenCalled();
  });

  it("does not double-discard when runtime adoption already took ownership before failing", async () => {
    runtime.prepare.mockRejectedValue(Object.assign(new Error("cleaned after take"), {
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH",
      path: "$.input.realmInput",
    }));
    runtime.inspect.mockReturnValue(Object.freeze({ state: "adopted" }));
    const terminal: CppCuteBrowserWorkerControllerInboundMessage[] = [];

    await handleCppCuteBrowserWorkerTransfer({} as never, (message) => terminal.push(message));

    expect(runtime.discard).not.toHaveBeenCalled();
    expect(terminal[0]).toMatchObject({
      phase: "runtime-adoption",
      failureCode: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH",
    });
  });

  it("emits one transferable success terminal through the same one-shot sink", async () => {
    runtime.start.mockResolvedValue(Object.freeze({
      kind: "browsergrad-cpp-cute-runtime-result",
      controlBytes: Uint8Array.of(1, 2),
      artifactBytes: Uint8Array.of(3, 4, 5),
    }));
    const terminal: Array<{
      readonly message: CppCuteBrowserWorkerControllerInboundMessage;
      readonly transfer: readonly ArrayBuffer[];
    }> = [];

    await handleCppCuteBrowserWorkerTransfer(
      {} as never,
      (message, transfer) => terminal.push({ message, transfer }),
    );

    expect(terminal).toHaveLength(1);
    const emitted = terminal[0]!;
    expect(emitted.message).toMatchObject({
      kind: "browsergrad-cpp-cute-worker-terminal",
      invocationId: INVOCATION_ID,
      invocationNonceSha256: NONCE,
      controlBytes: Uint8Array.of(1, 2),
      artifactBytes: Uint8Array.of(3, 4, 5),
    });
    expect(emitted.transfer).toEqual([
      (emitted.message as { controlBytes: Uint8Array }).controlBytes.buffer,
      (emitted.message as { artifactBytes: Uint8Array }).artifactBytes.buffer,
    ]);
  });

  it("does not self-assert a terminal identity when transfer reconstruction fails", async () => {
    const failure = new Error("untrusted transfer");
    runtime.reconstruct.mockRejectedValue(failure);
    const emit = vi.fn();

    await expect(handleCppCuteBrowserWorkerTransfer({} as never, emit)).rejects.toBe(failure);
    expect(emit).not.toHaveBeenCalled();
    expect(runtime.prepare).not.toHaveBeenCalled();
  });

  it("propagates terminal transport failure after runtime authority is already terminal", async () => {
    runtime.start.mockRejectedValue(Object.assign(new Error("blocked"), {
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CAPABILITY",
      path: "$.bundle",
    }));
    const postFailure = new Error("postMessage failed");

    await expect(handleCppCuteBrowserWorkerTransfer(
      {} as never,
      () => { throw postFailure; },
    )).rejects.toBe(postFailure);
  });

  it("identifies entry errors nominally", () => {
    const error = new CppCuteBrowserWorkerEntryError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-ENTRY-INTERNAL",
      "$.scope",
      "missing",
    );
    expect(error.name).toBe("CppCuteBrowserWorkerEntryError");
    expect(error.message).toContain("WORKER-ENTRY-INTERNAL");
  });
});
