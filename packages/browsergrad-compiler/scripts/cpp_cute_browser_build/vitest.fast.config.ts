import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    globals: false,
    include: [
      "scripts/cpp_cute_browser_build/*.test.ts",
      "tests/compiler/cpp_cute_browser_build_lock.test.ts",
      "tests/compiler/cpp_cute_browser_runtime_abi.test.ts",
      "tests/compiler/cpp_cute_browser_runtime_profile.test.ts",
      "tests/compiler/cpp_cute_browser_assets.test.ts",
      "tests/compiler/cpp_cute_frontend_profile.test.ts",
    ],
    exclude: ["scripts/cpp_cute_browser_build/*_native.test.ts"],
    fileParallelism: false,
  },
});
