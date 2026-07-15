import { execFileSync } from "node:child_process";

/**
 * Complete tracked/untracked source surface capable of changing compiler
 * verified-view-copy preparation, execution, evidence, or authorization.
 */
export const COMPILER_VIEW_COPY_EVIDENCE_SOURCE_PATHS = Object.freeze([
  "packages/browsergrad-compiler",
  "packages/browsergrad-kernels",
  "packages/browsergrad-semantic-core",
  "architecture/semantic-fixture-contracts.json",
  "test-support/view-copy-conformance-fixtures.ts",
  "test-support/webgpu-evidence.ts",
  "scripts/compiler-view-copy-evidence-source.mjs",
  "scripts/compiler-view-copy-evidence-source.d.mts",
  "package.json",
  "pnpm-lock.yaml",
  "pnpm-workspace.yaml",
  "tsconfig.base.json",
]);

export function readCompilerViewCopyEvidenceSourceStatus(repositoryRoot) {
  return execFileSync(
    "git",
    [
      "status",
      "--porcelain",
      "--untracked-files=all",
      "--",
      ...COMPILER_VIEW_COPY_EVIDENCE_SOURCE_PATHS,
    ],
    { cwd: repositoryRoot, encoding: "utf8" },
  ).trim();
}
