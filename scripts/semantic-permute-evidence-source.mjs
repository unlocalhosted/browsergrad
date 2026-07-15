import { execFileSync } from "node:child_process";

/**
 * Complete tracked/untracked source surface capable of changing JIT semantic
 * permutation preparation, execution, evidence, or release authorization.
 */
export const SEMANTIC_PERMUTE_EVIDENCE_SOURCE_PATHS = Object.freeze([
  "packages/browsergrad-jit",
  "packages/browsergrad-kernels",
  "packages/browsergrad-semantic-core",
  "architecture/semantic-fixture-contracts.json",
  "test-support/dense-permutation-view-copy-fixtures.ts",
  "test-support/webgpu-evidence.ts",
  "scripts/semantic-permute-evidence-source.mjs",
  "scripts/semantic-permute-evidence-source.d.mts",
  "package.json",
  "pnpm-lock.yaml",
  "pnpm-workspace.yaml",
  "tsconfig.base.json",
]);

export function readSemanticPermuteEvidenceSourceStatus(repositoryRoot) {
  return execFileSync(
    "git",
    [
      "status",
      "--porcelain",
      "--untracked-files=all",
      "--",
      ...SEMANTIC_PERMUTE_EVIDENCE_SOURCE_PATHS,
    ],
    { cwd: repositoryRoot, encoding: "utf8" },
  ).trim();
}
