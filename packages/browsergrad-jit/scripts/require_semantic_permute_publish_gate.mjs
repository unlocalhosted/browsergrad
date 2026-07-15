import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import { SEMANTIC_PERMUTE_EVIDENCE_SOURCE_PATHS } from "../../../scripts/semantic-permute-evidence-source.mjs";

export function validateSemanticPermutePublishGate({ evidenceCommit, githubSha, head, relevantStatus }) {
  if (!evidenceCommit) {
    throw new Error(
      "JIT publish blocked: run test:browser:semantic-permute:required for this commit and set BG_REQUIRED_JIT_SEMANTIC_PERMUTE_WEBGPU_EVIDENCE_COMMIT to its full commit SHA",
    );
  }
  if (!/^[0-9a-f]{40}$/u.test(evidenceCommit) || evidenceCommit !== head) {
    throw new Error(
      `JIT publish blocked: evidence commit ${evidenceCommit || "<missing>"} does not match HEAD ${head}`,
    );
  }
  if (githubSha && githubSha !== head) {
    throw new Error(`JIT publish blocked: GitHub SHA ${githubSha} does not match HEAD ${head}`);
  }
  if (relevantStatus.trim().length > 0) {
    throw new Error(
      `JIT publish blocked: package/dependency/evidence files differ from evidenced HEAD\n${relevantStatus.trim()}`,
    );
  }
}

function main() {
  const repositoryRoot = new URL("../../..", import.meta.url);
  const git = (...args) => execFileSync("git", args, {
    cwd: repositoryRoot,
    encoding: "utf8",
  }).trim();
  validateSemanticPermutePublishGate({
    evidenceCommit: process.env.BG_REQUIRED_JIT_SEMANTIC_PERMUTE_WEBGPU_EVIDENCE_COMMIT?.trim(),
    githubSha: process.env.GITHUB_SHA?.trim(),
    head: git("rev-parse", "HEAD"),
    relevantStatus: git(
      "status",
      "--porcelain",
      "--untracked-files=all",
      "--",
      ...SEMANTIC_PERMUTE_EVIDENCE_SOURCE_PATHS,
    ),
  });
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) main();
