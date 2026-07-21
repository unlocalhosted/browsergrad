import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    globals: false,
    include: [
      "scripts/cpp_cute_browser_build/*.test.ts",
      "tests/compiler/cpp_cute_browser_*.test.ts",
      "tests/compiler/cpp_cute_frontend_artifact.test.ts",
      "tests/compiler/cpp_cute_frontend_profile.test.ts",
      "tests/compiler/verify_compiler_runner.test.ts",
    ],
    exclude: ["scripts/cpp_cute_browser_build/*_native.test.ts"],
    fileParallelism: true,
    maxWorkers: 4,
  },
});
