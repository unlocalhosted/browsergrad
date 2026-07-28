import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    globals: false,
    include: ["scripts/cpp_cute_browser_build/*.test.ts"],
    exclude: ["scripts/cpp_cute_browser_build/*_native.test.ts"],
    fileParallelism: true,
    maxWorkers: 4,
  },
});
