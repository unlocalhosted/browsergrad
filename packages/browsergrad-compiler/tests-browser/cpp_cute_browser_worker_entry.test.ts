import { expect, it } from "vitest";

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
