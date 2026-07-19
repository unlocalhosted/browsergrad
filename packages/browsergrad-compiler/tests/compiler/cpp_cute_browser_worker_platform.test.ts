import { afterEach, describe, expect, it, vi } from "vitest";
import type { CppCuteBrowserWasmVerifierPlatformWorker } from
  "../../src/cpp_cute_browser_wasm_verifier_controller.js";
import type { CppCuteBrowserWorkerTransferMessage } from
  "../../src/cpp_cute_browser_worker_transfer.js";

interface BrowserHarnessOptions {
  readonly getRandomValues?: (bytes: Uint8Array) => unknown;
}

function installBrowserHarness(options: BrowserHarnessOptions = {}) {
  const NativeUrl = globalThis.URL;
  const observations = {
    blobs: [] as Array<{ readonly parts: readonly unknown[]; readonly options: unknown }>,
    workers: [] as Array<{ readonly url: string; readonly options: unknown }>,
    messages: [] as Array<{
      readonly message: unknown;
      readonly transfer: readonly Transferable[];
    }>,
    addedListeners: [] as Array<{ readonly type: string; readonly listener: unknown }>,
    removedListeners: [] as Array<{ readonly type: string; readonly listener: unknown }>,
    revokedUrls: [] as string[],
    clearedHandles: [] as unknown[],
    randomLengths: [] as number[],
    terminateCalls: 0,
    timeoutCalls: 0,
  };

  class HarnessBlob {
    constructor(parts: readonly unknown[], blobOptions: unknown) {
      observations.blobs.push({ parts, options: blobOptions });
    }
  }
  class HarnessEventTarget {
    addEventListener(type: string, listener: unknown): void {
      observations.addedListeners.push({ type, listener });
    }
    removeEventListener(type: string, listener: unknown): void {
      observations.removedListeners.push({ type, listener });
    }
  }
  class HarnessWorker extends HarnessEventTarget {
    constructor(url: string, workerOptions: unknown) {
      super();
      observations.workers.push({ url, options: workerOptions });
    }
    postMessage(message: unknown, transfer: readonly Transferable[]): void {
      observations.messages.push({ message, transfer });
    }
    terminate(): void {
      observations.terminateCalls += 1;
    }
  }
  class HarnessUrl extends NativeUrl {
    static override createObjectURL(_blob: unknown): string {
      return "blob:captured-platform";
    }
    static override revokeObjectURL(url: string): void {
      observations.revokedUrls.push(url);
    }
  }
  const performanceObject = {
    now: (): number => 42.5,
  };
  const cryptoObject = {
    getRandomValues: (bytes: Uint8Array): unknown => {
      observations.randomLengths.push(bytes.byteLength);
      if (options.getRandomValues !== undefined) {
        return options.getRandomValues(bytes);
      }
      for (let index = 0; index < bytes.byteLength; index += 1) {
        bytes[index] = 0xa0 + index;
      }
      return bytes;
    },
  };
  const setTimeoutFunction = (_callback: () => void, _delay: number): object => {
    observations.timeoutCalls += 1;
    return Object.freeze({ capturedTimer: true });
  };
  const clearTimeoutFunction = (handle: unknown): void => {
    observations.clearedHandles.push(handle);
  };

  vi.stubGlobal("Blob", HarnessBlob);
  vi.stubGlobal("Worker", HarnessWorker);
  vi.stubGlobal("EventTarget", HarnessEventTarget);
  vi.stubGlobal("URL", HarnessUrl);
  vi.stubGlobal("performance", performanceObject);
  vi.stubGlobal("crypto", cryptoObject);
  vi.stubGlobal("setTimeout", setTimeoutFunction);
  vi.stubGlobal("clearTimeout", clearTimeoutFunction);

  return {
    observations,
    HarnessWorker,
    HarnessEventTarget,
    HarnessUrl,
    performanceObject,
    cryptoObject,
  };
}

async function importFreshPlatform() {
  vi.resetModules();
  return import("../../src/cpp_cute_browser_worker_platform.js");
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

describe("C++/CuTe module-captured browser platform", () => {
  it("keeps generic Worker and entropy effects captured after global poisoning", async () => {
    const harness = installBrowserHarness();
    const module = await importFreshPlatform();
    const platform = module.getCppCuteBrowserCapturedPlatform();
    expect(platform).toBeDefined();
    if (platform === undefined) throw new Error("expected captured browser platform");
    const moduleBytes = Uint8Array.of(1, 2, 3);
    const message = Object.freeze({ kind: "generic-verifier-launch", bytes: Uint8Array.of(9) });
    const transfer = Object.freeze([new ArrayBuffer(8)]);

    harness.HarnessWorker.prototype.postMessage = () => {
      throw new Error("poisoned Worker.postMessage");
    };
    harness.HarnessWorker.prototype.terminate = () => {
      throw new Error("poisoned Worker.terminate");
    };
    harness.HarnessEventTarget.prototype.addEventListener = () => {
      throw new Error("poisoned EventTarget.addEventListener");
    };
    harness.HarnessEventTarget.prototype.removeEventListener = () => {
      throw new Error("poisoned EventTarget.removeEventListener");
    };
    harness.HarnessUrl.createObjectURL = () => {
      throw new Error("poisoned URL.createObjectURL");
    };
    harness.HarnessUrl.revokeObjectURL = () => {
      throw new Error("poisoned URL.revokeObjectURL");
    };
    harness.performanceObject.now = () => {
      throw new Error("poisoned performance.now");
    };
    harness.cryptoObject.getRandomValues = () => {
      throw new Error("poisoned crypto.getRandomValues");
    };
    vi.stubGlobal("Blob", class PoisonedBlob {
      constructor() { throw new Error("poisoned Blob"); }
    });
    vi.stubGlobal("Worker", class PoisonedWorker {
      constructor() { throw new Error("poisoned Worker"); }
    });
    vi.stubGlobal("Uint8Array", class PoisonedUint8Array {
      constructor() { throw new Error("poisoned Uint8Array"); }
    });
    vi.stubGlobal("setTimeout", () => { throw new Error("poisoned setTimeout"); });
    vi.stubGlobal("clearTimeout", () => { throw new Error("poisoned clearTimeout"); });

    const originalIsSafeInteger = Number.isSafeInteger;
    Number.isSafeInteger = () => { throw new Error("poisoned Number.isSafeInteger"); };
    let firstRandom: Uint8Array;
    let secondRandom: Uint8Array;
    try {
      firstRandom = platform.randomBytes(4);
      secondRandom = platform.randomBytes(4);
    } finally {
      Number.isSafeInteger = originalIsSafeInteger;
    }
    expect(Array.from(firstRandom)).toEqual([0xa0, 0xa1, 0xa2, 0xa3]);
    expect(secondRandom).not.toBe(firstRandom);
    expect(secondRandom.byteLength).toBe(4);
    expect(platform.createModuleBlobUrl(moduleBytes)).toBe(
      "blob:captured-platform",
    );
    const originalFreeze = Object.freeze;
    Object.freeze = () => { throw new Error("poisoned Object.freeze"); };
    let worker: ReturnType<typeof platform.createModuleWorker>;
    try {
      worker = platform.createModuleWorker("blob:captured-platform", "verifier");
    } finally {
      Object.freeze = originalFreeze;
    }
    const compilerCompatible: {
      postMessage(
        message: CppCuteBrowserWorkerTransferMessage,
        transfer: readonly ArrayBuffer[],
      ): void;
    } = worker;
    const verifierCompatible: CppCuteBrowserWasmVerifierPlatformWorker = worker;
    expect(compilerCompatible).toBe(worker);
    expect(verifierCompatible).toBe(worker);
    worker.postMessage(message, transfer);
    const listener = (): void => undefined;
    worker.addEventListener("message", listener);
    worker.removeEventListener("message", listener);
    worker.terminate();
    expect(platform.monotonicNowMilliseconds()).toBe(42.5);
    const timer = platform.setHostTimeout(() => undefined, 25);
    platform.clearHostTimeout(timer);
    platform.revokeModuleBlobUrl("blob:captured-platform");

    expect(harness.observations.randomLengths).toEqual([4, 4]);
    expect(harness.observations.blobs).toHaveLength(1);
    expect(harness.observations.blobs[0]?.parts[0]).toEqual(moduleBytes);
    expect(harness.observations.blobs[0]?.parts[0]).not.toBe(moduleBytes);
    expect(harness.observations.workers).toEqual([{
      url: "blob:captured-platform",
      options: { type: "module", name: "verifier" },
    }]);
    expect(harness.observations.messages).toEqual([{ message, transfer }]);
    expect(harness.observations.addedListeners).toEqual([{ type: "message", listener }]);
    expect(harness.observations.removedListeners).toEqual([{ type: "message", listener }]);
    expect(harness.observations.terminateCalls).toBe(1);
    expect(harness.observations.timeoutCalls).toBe(1);
    expect(harness.observations.clearedHandles).toEqual([timer]);
    expect(harness.observations.revokedUrls).toEqual(["blob:captured-platform"]);
  });

  it("rejects invalid random lengths before invoking captured entropy", async () => {
    const harness = installBrowserHarness();
    const module = await importFreshPlatform();
    const platform = module.getCppCuteBrowserCapturedPlatform();
    expect(platform).toBeDefined();
    if (platform === undefined) throw new Error("expected captured browser platform");

    for (const byteLength of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, 65_537]) {
      expect(() => platform.randomBytes(byteLength)).toThrowError(
        expect.objectContaining({
          code: "BG-COMPILER-CPP-CUTE-BROWSER-PLATFORM-RANDOM-LENGTH",
          path: "$.randomBytes",
          message: expect.stringContaining("byte length must be an integer"),
        }),
      );
    }
    expect(harness.observations.randomLengths).toEqual([]);
  });

  it("reports captured entropy failures and rejects a non-identical result", async () => {
    installBrowserHarness({
      getRandomValues: () => { throw new Error("entropy offline"); },
    });
    const failedModule = await importFreshPlatform();
    const failedPlatform = failedModule.getCppCuteBrowserCapturedPlatform();
    expect(failedPlatform).toBeDefined();
    if (failedPlatform === undefined) throw new Error("expected captured browser platform");
    expect(() => failedPlatform.randomBytes(32)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-PLATFORM-RANDOM-FAILED",
        path: "$.randomBytes",
        message: expect.stringContaining("captured crypto.getRandomValues failed"),
      }),
    );

    vi.unstubAllGlobals();
    installBrowserHarness({
      getRandomValues: () => new Uint8Array(32),
    });
    const invalidModule = await importFreshPlatform();
    const invalidPlatform = invalidModule.getCppCuteBrowserCapturedPlatform();
    expect(invalidPlatform).toBeDefined();
    if (invalidPlatform === undefined) throw new Error("expected captured browser platform");
    expect(() => invalidPlatform.randomBytes(32)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-PLATFORM-RANDOM-RESULT",
        path: "$.randomBytes",
        message: expect.stringContaining("did not return the exact requested byte array"),
      }),
    );
  });

  it("never accepts caller effects or labels captured effects as production authority", async () => {
    installBrowserHarness();
    const module = await importFreshPlatform();
    const forgedEffects = Object.create(null, {
      randomBytes: {
        enumerable: true,
        get: () => { throw new Error("caller effect was read"); },
      },
      productionAuthority: { enumerable: true, value: true },
    });
    const getWithIgnoredInput = module.getCppCuteBrowserCapturedPlatform as unknown as
      (effects: unknown) => ReturnType<typeof module.getCppCuteBrowserCapturedPlatform>;
    const platform = getWithIgnoredInput(forgedEffects);
    expect(platform).toBe(module.getCppCuteBrowserCapturedPlatform());
    expect(platform).toMatchObject({
      authority: "module-captured-browser-platform-effects",
      capturedBeforeInvocation: true,
      callerEffectsAccepted: false,
    });
    expect(platform).not.toHaveProperty("productionAuthority");
    expect(Object.isFrozen(platform)).toBe(true);
  });
});
