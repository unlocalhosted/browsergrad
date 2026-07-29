import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    globals: false,
    include: [
      "scripts/cpp_cute_browser_build/cpp_cute_browser_build_cache_reuse.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_build_executor.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_build_executor_process.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_build_failure_observation.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_build_lock_authoring.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_build_plan.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_build_reproducibility.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_build_runner.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_build_runtime_closure.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_build_workflow.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_configured_target_review.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_extractor_source.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_exact_distribution_convergence.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_exact_distribution_convergence_authoring.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_full_distribution_reproducibility_authoring.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_header_distribution_reproducibility_authoring.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_reproducibility_authoring.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_toolchain_cache.test.ts",
      "scripts/cpp_cute_browser_build/cpp_cute_browser_wasm_review.test.ts",
    ],
    fileParallelism: true,
    maxWorkers: 4,
  },
});
