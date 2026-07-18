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
let body: ReturnType<typeof unwrapPreparedCppCuteBrowserBuildInputLock>["lock"]["body"];

beforeAll(async () => {
  const repositoryRoot = join(dirname(fileURLToPath(import.meta.url)), "../../../..");
  workflow = await readFile(
    join(repositoryRoot, ".github", "workflows", "build-clang-wasm.yml"),
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
    expect(workflow).toContain("dst=/browsergrad/inputs,readonly");
  });

  it("pins every third-party workflow action to a full commit", () => {
    const actionLines = workflow.split("\n").filter((line) => line.includes("uses:"));
    expect(actionLines.length).toBeGreaterThan(0);
    for (const line of actionLines) {
      expect(line).toMatch(/uses:\s+[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+@[0-9a-f]{40}(?:\s+#.*)?$/u);
    }
  });
});
