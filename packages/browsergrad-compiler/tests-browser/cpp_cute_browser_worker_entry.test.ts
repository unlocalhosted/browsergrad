import { expect, it } from "vitest";
import {
  copyVerifiedCppCuteBrowserWorkerBundleBytes,
  verifyCppCuteBrowserWorkerBundle,
} from "../src/cpp_cute_browser_worker_bundle.js";
import { getCppCuteBrowserCapturedPlatform } from "../src/cpp_cute_browser_worker_platform.js";

it("loads the package entry in a real one-shot module Worker", async () => {
  const worker = new Worker(
    new URL("../src/cpp_cute_browser_worker_module.ts", import.meta.url),
    { type: "module", name: "browsergrad-cpp-cute-entry-test" },
  );
  const errors: ErrorEvent[] = [];
  try {
    const firstError = new Promise<ErrorEvent>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error("module Worker did not report its invalid launch")),
        10_000,
      );
      worker.addEventListener("error", (event) => {
        event.preventDefault();
        errors.push(event);
        clearTimeout(timeout);
        resolve(event);
      });
    });

    worker.postMessage(Object.freeze({ kind: "untrusted-test-launch" }));
    const error = await firstError;
    expect(error.message).toContain("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-INVALID");

    worker.postMessage(Object.freeze({ kind: "second-launch-must-be-ignored" }));
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(errors).toHaveLength(1);
  } finally {
    worker.terminate();
  }
});

it("loads the exact package-owned zero-import bundle bytes in a real module Worker", async () => {
  const bundle = await verifyCppCuteBrowserWorkerBundle();
  const bytes = copyVerifiedCppCuteBrowserWorkerBundleBytes(bundle);
  const blobUrl = URL.createObjectURL(new Blob([
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
  ], { type: "text/javascript" }));
  const worker = new Worker(blobUrl, {
    type: "module",
    name: "browsergrad-cpp-cute-pinned-bundle-test",
  });
  const errors: ErrorEvent[] = [];
  try {
    const firstError = new Promise<ErrorEvent>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error("pinned module Worker did not report its invalid launch")),
        10_000,
      );
      worker.addEventListener("error", (event) => {
        event.preventDefault();
        errors.push(event);
        clearTimeout(timeout);
        resolve(event);
      });
    });

    worker.postMessage(Object.freeze({ kind: "untrusted-pinned-bundle-test-launch" }));
    const error = await firstError;
    expect(error.message).toContain("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-INVALID");

    worker.postMessage(Object.freeze({ kind: "second-launch-must-be-ignored" }));
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(errors).toHaveLength(1);
  } finally {
    worker.terminate();
    URL.revokeObjectURL(blobUrl);
  }
});

it("uses only the module-captured browser effects for an exact package Worker", async () => {
  const platform = getCppCuteBrowserCapturedPlatform();
  expect(platform).toMatchObject({
    authority: "module-captured-browser-platform-effects",
    capturedBeforeInvocation: true,
    callerEffectsAccepted: false,
  });
  if (platform === undefined) throw new Error("captured browser platform is unavailable");
  const bundle = await verifyCppCuteBrowserWorkerBundle();
  const blobUrl = platform.createModuleBlobUrl(
    copyVerifiedCppCuteBrowserWorkerBundleBytes(bundle),
  );
  const worker = platform.createModuleWorker(
    blobUrl,
    "browsergrad-cpp-cute-captured-platform-test",
  );
  const errors: ErrorEvent[] = [];
  try {
    const firstError = new Promise<ErrorEvent>((resolve, reject) => {
      const timeout = platform.setHostTimeout(
        () => reject(new Error("captured platform Worker did not report its invalid launch")),
        10_000,
      );
      worker.addEventListener("error", (value) => {
        const event = value as ErrorEvent;
        event.preventDefault();
        errors.push(event);
        platform.clearHostTimeout(timeout);
        resolve(event);
      });
    });

    worker.postMessage(Object.freeze({ kind: "untrusted-captured-platform-launch" }) as never, []);
    const error = await firstError;
    expect(error.message).toContain("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-INVALID");

    worker.postMessage(Object.freeze({ kind: "second-launch-must-be-ignored" }) as never, []);
    await new Promise<void>((resolve) => platform.setHostTimeout(() => resolve(), 100));
    expect(errors).toHaveLength(1);
  } finally {
    worker.terminate();
    platform.revokeModuleBlobUrl(blobUrl);
  }
});
