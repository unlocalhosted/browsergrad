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
let ciWorkflow: string;
let body: ReturnType<typeof unwrapPreparedCppCuteBrowserBuildInputLock>["lock"]["body"];

beforeAll(async () => {
  const repositoryRoot = join(dirname(fileURLToPath(import.meta.url)), "../../../..");
  workflow = await readFile(
    join(repositoryRoot, ".github", "workflows", "build-clang-wasm.yml"),
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
    expect(workflow).toContain("dst=/workspace,readonly");
    expect(workflow).toContain("dst=/browsergrad/inputs/build-${BG_CLANG_BUILD_ORDINAL},readonly");
  });

  it("separates one-build validation from two-build reproducibility", () => {
    expect(workflow).toContain("default: validation");
    expect(workflow).toContain("- validation");
    expect(workflow).toContain("- reproducibility");
    expect(workflow).toContain(
      "buildOrdinal: ${{ fromJSON(inputs.mode == 'reproducibility' && '[1,2]' || '[1]') }}",
    );
    expect(workflow).toContain("work/build-${BG_CLANG_BUILD_ORDINAL}");
    expect(workflow).toContain("inputs/build-${BG_CLANG_BUILD_ORDINAL}");
    expect(workflow).toContain("needs: build");
    expect(workflow).toContain("if: ${{ inputs.mode == 'reproducibility' }}");
    expect(workflow).toContain("cpp_cute_browser_build_reproducibility.mjs");
    expect(workflow).toContain("cpp_cute_browser_wasm_review.mjs");
    expect(workflow).toContain("--first-root=\"${BG_CLANG_REPRO_ROOT}/first\"");
    expect(workflow).toContain("--second-root=\"${BG_CLANG_REPRO_ROOT}/second\"");
    expect(workflow).toContain("clang-wasm-runtime-abi-review.v1.json");
  });

  it("pins every third-party workflow action to a full commit", () => {
    const actionLines = workflow.split("\n").filter((line) => line.includes("uses:"));
    expect(actionLines.length).toBeGreaterThan(0);
    for (const line of actionLines) {
      expect(line).toMatch(/uses:\s+[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+@[0-9a-f]{40}(?:\s+#.*)?$/u);
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
      expect(line).toMatch(/uses:\s+[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+@[0-9a-f]{40}(?:\s+#.*)?$/u);
    }
  });

  it("keeps Grad integration and real WebGPU suites blocking", () => {
    expect(ciWorkflow).toContain("Run grad integration suite");
    expect(ciWorkflow).toContain("Run kernels browser suite");
    expect(ciWorkflow).not.toContain("continue-on-error");
  });
});
