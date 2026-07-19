import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { beforeAll, describe, expect, it } from "vitest";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";

let workflow: string;
let existingArtifactsWorkflow: string;
let ciWorkflow: string;
let body: ReturnType<typeof unwrapPreparedCppCuteBrowserBuildInputLock>["lock"]["body"];

beforeAll(async () => {
  const repositoryRoot = join(dirname(fileURLToPath(import.meta.url)), "../../../..");
  workflow = await readFile(
    join(repositoryRoot, ".github", "workflows", "build-clang-wasm.yml"),
    "utf8",
  );
  existingArtifactsWorkflow = await readFile(
    join(repositoryRoot, ".github", "workflows", "verify-clang-wasm-reproducibility.yml"),
    "utf8",
  );
  ciWorkflow = await readFile(
    join(repositoryRoot, ".github", "workflows", "ci.yml"),
    "utf8",
  );
  const lock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  body = unwrapPreparedCppCuteBrowserBuildInputLock(lock).lock.body;
});

describe("Clang-Wasm evidence workflow", () => {
  it("binds acquisition and builder bytes to the checked-in build lock", () => {
    const llvm = body.sources.find((source) => source.sourceId === "llvm-project");
    expect(llvm).toBeDefined();
    expect(workflow).toContain(llvm?.acquisitionUrl);
    expect(workflow).toContain(llvm?.archiveSha256);
    expect(workflow).toContain(llvm?.archiveByteLength);
    expect(workflow).toContain(body.builder.platformManifestDigest);
    expect(workflow).toContain(body.builder.imageConfigDigest);
    expect(workflow).not.toContain(`${body.builder.repository}:${body.builder.tag}`);
  });

  it("keeps the expensive build manual, networkless, read-only, and unprivileged", () => {
    expect(workflow).toContain("workflow_dispatch:");
    expect(workflow).not.toMatch(/^\s+push:/mu);
    expect(workflow).not.toMatch(/^\s+pull_request:/mu);
    expect(workflow).toContain("--network none");
    expect(workflow).toContain("--read-only");
    expect(workflow).toContain("--cap-drop ALL");
    expect(workflow).toContain("--security-opt no-new-privileges:true");
    expect(workflow).toContain("--user \"$(id -u):$(id -g)\"");
    expect(workflow).not.toContain("src=${GITHUB_WORKSPACE},dst=/workspace,readonly");
    expect(workflow).not.toContain("src=${GITHUB_WORKSPACE}/packages/");
    expect(workflow).toContain("cpp_cute_browser_build_runtime_closure.mjs");
    expect(workflow).toContain(
      "--output-root=\"${BG_CLANG_BUILD_ROOT}/inputs/runtime-workspace\"",
    );
    expect(workflow).toContain(
      "src=${BG_CLANG_BUILD_ROOT}/inputs/runtime-workspace,dst=/workspace,readonly",
    );
    expect(workflow).toContain("dst=/browsergrad/inputs/build-${BG_CLANG_BUILD_ORDINAL},readonly");
    expect(workflow).toContain("build-execution-observation.v2.json");
    expect(workflow).not.toContain("build-execution-observation.v1.json");
  });

  it("separates cached feedback, clean validation, and reproducibility", () => {
    expect(workflow).toContain("default: fast-validation");
    expect(workflow).toContain("- fast-validation");
    expect(workflow).toContain("- clean-validation");
    expect(workflow).toContain("- reproducibility");
    expect(workflow).toContain(
      "buildOrdinal: ${{ fromJSON(inputs.mode == 'reproducibility' && '[1,2]' || '[1]') }}",
    );
    expect(workflow).toContain("work/build-${BG_CLANG_BUILD_ORDINAL}");
    expect(workflow).toContain("inputs/build-${BG_CLANG_BUILD_ORDINAL}");
    expect(workflow).toContain("needs: [verification-boundary, build]");
    expect(workflow).toContain("if: ${{ inputs.mode == 'reproducibility' }}");
    expect(workflow).toContain("cpp_cute_browser_build_reproducibility.mjs");
    expect(workflow).toContain("clang-wasm-reproducibility.v3.json");
    expect(workflow).not.toContain("clang-wasm-reproducibility.v1.json");
    expect(workflow.match(/cpp_cute_browser_wasm_review\.mjs/gu)).toHaveLength(2);
    expect(workflow).toContain("Inspect built Wasm against the pinned runtime ABI");
    expect(workflow).toContain(
      "--wasm=\"${bg_output_root}/browsergrad-cpp-cute/clang-extractor.wasm\"",
    );
    expect(workflow).toContain(
      "--output=\"${bg_output_root}/clang-wasm-runtime-abi-review.v1.json\"",
    );
    expect(workflow).toContain("--first-root=\"${BG_CLANG_REPRO_ROOT}/first\"");
    expect(workflow).toContain("--second-root=\"${BG_CLANG_REPRO_ROOT}/second\"");
    expect(workflow).toContain("clang-wasm-runtime-abi-review.v1.json");
  });

  it("keeps cached feedback content-addressed and non-authoritative", () => {
    expect(workflow.match(/pnpm --filter @unlocalhosted\/browsergrad-compiler\.\.\. build/gu))
      .toHaveLength(3);
    expect(workflow).not.toContain("pnpm -r build");
    expect(workflow).toContain("cpp_cute_browser_toolchain_cache.mjs");
    expect(workflow).toContain(
      "actions/cache/restore@55cc8345863c7cc4c66a329aec7e433d2d1c52a9 # v6.1.0",
    );
    expect(workflow).toContain(
      "actions/cache/save@55cc8345863c7cc4c66a329aec7e433d2d1c52a9 # v6.1.0",
    );
    expect(workflow).toContain(
      "${{ runner.temp }}/browsergrad-clang-wasm-cache-v2/build-${{ matrix.buildOrdinal }}/native-tablegen",
    );
    expect(workflow).toContain("Admit the restored diagnostic cache into the private build root");
    expect(workflow).toContain(
      "BG_MATCHED_DIAGNOSTIC_CACHE_KEY: ${{ steps.toolchain-cache.outputs.cache-matched-key }}",
    );
    expect(workflow).toContain('if [[ -n "${BG_MATCHED_DIAGNOSTIC_CACHE_KEY}" ]]');
    expect(workflow).toContain("Stage a complete reusable diagnostic toolchain cache");
    expect(workflow).not.toContain("diagnostic cache for migration");
    expect(workflow).not.toContain("previous-broad-toolchain-cache");
    expect(workflow).toContain(
      "if: ${{ success() && inputs.mode == 'fast-validation' && steps.toolchain-cache.outputs.cache-hit != 'true' }}",
    );
    expect(workflow).toContain("native-tablegen/bin/clang-tblgen");
    expect(workflow).toContain("clang-extractor-wasm/lib/libclangTooling.a");
    expect(workflow).toContain("--execution-mode=\"${{ inputs.mode == 'fast-validation' && 'cached-diagnostic' || 'clean' }}\"");
    expect(workflow).toContain("fast-validation-observation.v1.json");
    expect(workflow).toContain("test:browser-clang-wasm-build-plan:fast");
    expect(workflow).toContain("test:browser-clang-wasm-build-plan:run");
    expect(workflow.match(/restore-keys:/gu)).toHaveLength(1);
    expect(workflow).toContain(
      "linux-amd64-${{ steps.toolchain-cache-key.outputs.compatible-legacy-cache-key }}",
    );
  });

  it("runs JavaScript verification once in parallel with every expensive build", () => {
    expect(workflow).toContain("verification-boundary:");
    expect(workflow).toContain("name: Verify JavaScript build boundary");
    expect(workflow).toContain(
      "name: Verify the JavaScript boundary outside the compiler critical path",
    );
    expect(workflow.match(/test:browser-clang-wasm-build-plan:fast/gu)).toHaveLength(1);
    expect(workflow.match(/test:browser-clang-wasm-build-plan:run/gu)).toHaveLength(1);

    const buildStart = workflow.indexOf("\n  build:\n");
    const reproducibilityStart = workflow.indexOf("\n  reproducibility:\n");
    expect(buildStart).toBeGreaterThan(0);
    expect(reproducibilityStart).toBeGreaterThan(buildStart);
    const buildJob = workflow.slice(buildStart, reproducibilityStart);
    expect(buildJob).toContain("Materialize the exact JavaScript runtime closure");
    expect(buildJob).not.toContain("test:browser-clang-wasm-build-plan:");
    expect(buildJob).not.toContain("needs: verification-boundary");
  });

  it("does not queue fast feedback behind independent clean evidence modes", () => {
    expect(workflow).toContain(
      "group: clang-wasm-build-${{ github.ref }}-${{ inputs.mode }}",
    );
    expect(workflow).toContain(
      "cancel-in-progress: ${{ inputs.mode == 'fast-validation' }}",
    );
    expect(workflow).not.toContain("group: clang-wasm-build-${{ github.ref }}\n");
    expect(workflow).not.toContain("cancel-in-progress: false");
  });

  it("pins every third-party workflow action to a full commit", () => {
    const actionLines = workflow.split("\n").filter((line) => line.includes("uses:"));
    expect(actionLines.length).toBeGreaterThan(0);
    for (const line of actionLines) {
      expect(line).toMatch(
        /uses:\s+[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)*@[0-9a-f]{40}(?:\s+#.*)?$/u,
      );
    }
  });

  it("uses the Node-24-capable artifact transfer actions", () => {
    expect(workflow).toContain(
      "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1",
    );
    expect(workflow).toContain(
      "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8.0.1",
    );
    expect(workflow).not.toContain("actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02");
    expect(workflow).not.toContain("actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093");
  });

  it("can reverify existing exact artifacts without rebuilding Clang", () => {
    expect(existingArtifactsWorkflow).toContain("workflow_dispatch:");
    expect(existingArtifactsWorkflow).toContain("actions: read");
    expect(existingArtifactsWorkflow).toContain("contents: read");
    expect(existingArtifactsWorkflow).toContain(".path == \".github/workflows/build-clang-wasm.yml\"");
    expect(existingArtifactsWorkflow).toContain(".event == \"workflow_dispatch\"");
    expect(existingArtifactsWorkflow).toContain("Locked Linux amd64 build 1");
    expect(existingArtifactsWorkflow).toContain("Locked Linux amd64 build 2");
    expect(existingArtifactsWorkflow).toContain("run-id: ${{ inputs.source_run_id }}");
    expect(existingArtifactsWorkflow).toContain("clang-wasm-reproducibility.v3.json");
    expect(existingArtifactsWorkflow).toContain("cpp_cute_browser_wasm_review.mjs");
    expect(existingArtifactsWorkflow).not.toContain("cpp_cute_browser_build_runner.mjs");
    expect(existingArtifactsWorkflow).not.toContain("docker run");
  });

  it("pins every existing-artifact verifier action to a full commit", () => {
    const actionLines = existingArtifactsWorkflow.split("\n")
      .filter((line) => line.includes("uses:"));
    expect(actionLines.length).toBeGreaterThan(0);
    for (const line of actionLines) {
      expect(line).toMatch(
        /uses:\s+[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)*@[0-9a-f]{40}(?:\s+#.*)?$/u,
      );
    }
  });

  it("requires the exact native Clang version used by the locked source", () => {
    expect(ciWorkflow).toContain("llvm-toolchain-noble-22");
    expect(ciWorkflow).toContain("libclang-rt-22-dev");
    expect(ciWorkflow).toContain("/usr/lib/llvm-22/bin/llvm-config");
    expect(ciWorkflow).toContain("--version)\" = \"22.1.8\"");
    expect(ciWorkflow).toContain(
      "8b2a587ffd672c4687e7581dad4b2f6c1bb2ad6b480cd9771ba2ff48e0b8c75d",
    );
    expect(ciWorkflow).not.toContain("clang-18");
  });

  it("pins every main-CI workflow action to a full commit", () => {
    const actionLines = ciWorkflow.split("\n").filter((line) => line.includes("uses:"));
    expect(actionLines.length).toBeGreaterThan(0);
    for (const line of actionLines) {
      expect(line).toMatch(
        /uses:\s+[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)*@[0-9a-f]{40}(?:\s+#.*)?$/u,
      );
    }
  });

  it("keeps Grad integration and real WebGPU suites blocking", () => {
    expect(ciWorkflow).toContain("Run grad integration suite");
    expect(ciWorkflow).toContain("Run kernels browser suite");
    expect(ciWorkflow).not.toContain("continue-on-error");
  });

  it("keeps oldest, LTS native-harness, and Node 25 default compatibility lanes", () => {
    expect(ciWorkflow).toContain("node: [20, 24, 25]");
    expect(ciWorkflow).toContain("if: matrix.node == 24");
  });
});
