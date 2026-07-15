import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import { COMPILER_VIEW_COPY_EVIDENCE_SOURCE_PATHS } from "../../../scripts/compiler-view-copy-evidence-source.mjs";

export function validateCompilerEvidencePublishGate({
  layoutEvidenceCommit,
  viewCopyEvidenceCommit,
  githubSha,
  head,
  relevantStatus,
}) {
  if (!layoutEvidenceCommit) {
    throw new Error(
      "compiler publish blocked: run test:browser:layout-bindings:required for this commit and set BG_REQUIRED_COMPILER_LAYOUT_WEBGPU_EVIDENCE_COMMIT to its full commit SHA",
    );
  }
  if (!/^[0-9a-f]{40}$/u.test(layoutEvidenceCommit) || layoutEvidenceCommit !== head) {
    throw new Error(
      `compiler publish blocked: layout-binding evidence commit ${layoutEvidenceCommit} does not match HEAD ${head}`,
    );
  }
  if (!viewCopyEvidenceCommit) {
    throw new Error(
      "compiler publish blocked: run test:browser:view-copy-bindings:required for this commit and set BG_REQUIRED_COMPILER_VIEW_COPY_BINDINGS_WEBGPU_EVIDENCE_COMMIT to its full commit SHA",
    );
  }
  if (!/^[0-9a-f]{40}$/u.test(viewCopyEvidenceCommit) || viewCopyEvidenceCommit !== head) {
    throw new Error(
      `compiler publish blocked: view-copy-binding evidence commit ${viewCopyEvidenceCommit} does not match HEAD ${head}`,
    );
  }
  if (githubSha && githubSha !== head) {
    throw new Error(`compiler publish blocked: GitHub SHA ${githubSha} does not match HEAD ${head}`);
  }
  if (relevantStatus.trim().length > 0) {
    throw new Error(
      `compiler publish blocked: package/dependency/evidence files differ from evidenced HEAD\n${relevantStatus.trim()}`,
    );
  }
}

function main() {
  const repositoryRoot = new URL("../../..", import.meta.url);
  const git = (...args) => execFileSync("git", args, {
    cwd: repositoryRoot,
    encoding: "utf8",
  }).trim();
  validateCompilerEvidencePublishGate({
    layoutEvidenceCommit: process.env.BG_REQUIRED_COMPILER_LAYOUT_WEBGPU_EVIDENCE_COMMIT?.trim(),
    viewCopyEvidenceCommit:
      process.env.BG_REQUIRED_COMPILER_VIEW_COPY_BINDINGS_WEBGPU_EVIDENCE_COMMIT?.trim(),
    githubSha: process.env.GITHUB_SHA?.trim(),
    head: git("rev-parse", "HEAD"),
    relevantStatus: git(
      "status",
      "--porcelain",
      "--untracked-files=all",
      "--",
      ...COMPILER_VIEW_COPY_EVIDENCE_SOURCE_PATHS,
    ),
  });
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) main();
