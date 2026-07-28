import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    globals: false,
    include: [
      "scripts/cpp_cute_browser_build/*_native.test.ts",
      // Process-tree behavior belongs beside the native child-process lane.
      "scripts/cpp_cute_browser_build/cpp_cute_browser_native_test_harness.test.ts",
    ],
    fileParallelism: false,
    maxWorkers: 1,
  },
});
