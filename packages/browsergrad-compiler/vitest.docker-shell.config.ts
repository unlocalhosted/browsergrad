import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    globals: false,
    include: ["scripts/cpp_cute_aot_docker_*.test.ts"],
    fileParallelism: false,
  },
});
