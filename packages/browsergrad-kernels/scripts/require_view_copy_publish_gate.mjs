import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import { SEMANTIC_PERMUTE_EVIDENCE_SOURCE_PATHS } from "../../../scripts/semantic-permute-evidence-source.mjs";

export function validateViewCopyPublishGate({
  evidenceCommit,
  semanticHostGraphEvidenceCommit,
  semanticHostGraphWorkerEvidenceCommit,
  semanticHostGraphPerformanceEvidenceCommit,
  semanticGemmEvidenceCommit,
  semanticAttentionEvidenceCommit,
  semanticAttentionPerformanceEvidenceCommit,
  jitEvidenceCommit,
  githubSha,
  head,
  relevantStatus,
}) {
  if (!evidenceCommit) {
    throw new Error(
      "kernels publish blocked: run test:browser:view-copy:required for this commit and set BG_REQUIRED_WEBGPU_EVIDENCE_COMMIT to its full commit SHA",
    );
  }
  if (!/^[0-9a-f]{40}$/u.test(evidenceCommit) || evidenceCommit !== head) {
    throw new Error(
      `kernels publish blocked: evidence commit ${evidenceCommit || "<missing>"} does not match HEAD ${head}`,
    );
  }
  if (!semanticHostGraphEvidenceCommit) {
    throw new Error(
      "kernels publish blocked: run test:browser:semantic-host-graph:required for this commit and set BG_REQUIRED_SEMANTIC_HOST_GRAPH_WEBGPU_EVIDENCE_COMMIT to its full commit SHA",
    );
  }
  if (!/^[0-9a-f]{40}$/u.test(semanticHostGraphEvidenceCommit)
    || semanticHostGraphEvidenceCommit !== head) {
    throw new Error(
      `kernels publish blocked: semantic host-graph evidence commit ${semanticHostGraphEvidenceCommit || "<missing>"} does not match HEAD ${head}`,
    );
  }
  if (!semanticHostGraphWorkerEvidenceCommit) {
    throw new Error(
      "kernels publish blocked: run test:browser:semantic-host-graph-worker:required for this commit and set BG_REQUIRED_SEMANTIC_HOST_GRAPH_WORKER_WEBGPU_EVIDENCE_COMMIT to its full commit SHA",
    );
  }
  if (
    !/^[0-9a-f]{40}$/u.test(semanticHostGraphWorkerEvidenceCommit) ||
    semanticHostGraphWorkerEvidenceCommit !== head
  ) {
    throw new Error(
      `kernels publish blocked: semantic host-graph Worker evidence commit ${semanticHostGraphWorkerEvidenceCommit || "<missing>"} does not match HEAD ${head}`,
    );
  }
  if (!semanticHostGraphPerformanceEvidenceCommit) {
    throw new Error(
      "kernels publish blocked: run test:browser:semantic-host-graph:performance:required for this commit and set BG_REQUIRED_SEMANTIC_HOST_GRAPH_WEBGPU_PERFORMANCE_EVIDENCE_COMMIT to its full commit SHA",
    );
  }
  if (
    !/^[0-9a-f]{40}$/u.test(semanticHostGraphPerformanceEvidenceCommit) ||
    semanticHostGraphPerformanceEvidenceCommit !== head
  ) {
    throw new Error(
      `kernels publish blocked: semantic host-graph performance evidence commit ${semanticHostGraphPerformanceEvidenceCommit || "<missing>"} does not match HEAD ${head}`,
    );
  }
  if (!semanticGemmEvidenceCommit) {
    throw new Error(
      "kernels publish blocked: run test:browser:semantic-gemm:required for this commit and set BG_REQUIRED_SEMANTIC_GEMM_WEBGPU_EVIDENCE_COMMIT to its full commit SHA",
    );
  }
  if (!/^[0-9a-f]{40}$/u.test(semanticGemmEvidenceCommit) || semanticGemmEvidenceCommit !== head) {
    throw new Error(
      `kernels publish blocked: semantic GEMM evidence commit ${semanticGemmEvidenceCommit || "<missing>"} does not match HEAD ${head}`,
    );
  }
  if (!semanticAttentionEvidenceCommit) {
    throw new Error(
      "kernels publish blocked: run test:browser:semantic-attention:required for this commit and set BG_REQUIRED_SEMANTIC_ATTENTION_WEBGPU_EVIDENCE_COMMIT to its full commit SHA",
    );
  }
  if (!/^[0-9a-f]{40}$/u.test(semanticAttentionEvidenceCommit)
    || semanticAttentionEvidenceCommit !== head) {
    throw new Error(
      `kernels publish blocked: semantic attention evidence commit ${semanticAttentionEvidenceCommit || "<missing>"} does not match HEAD ${head}`,
    );
  }
  if (!semanticAttentionPerformanceEvidenceCommit) {
    throw new Error(
      "kernels publish blocked: run test:browser:semantic-attention:performance:required for this commit and set BG_REQUIRED_SEMANTIC_ATTENTION_WEBGPU_PERFORMANCE_EVIDENCE_COMMIT to its full commit SHA",
    );
  }
  if (!/^[0-9a-f]{40}$/u.test(semanticAttentionPerformanceEvidenceCommit)
    || semanticAttentionPerformanceEvidenceCommit !== head) {
    throw new Error(
      `kernels publish blocked: semantic attention performance evidence commit ${semanticAttentionPerformanceEvidenceCommit || "<missing>"} does not match HEAD ${head}`,
    );
  }
  if (!jitEvidenceCommit) {
    throw new Error(
      "kernels publish blocked: run test:browser:semantic-permute:required for this commit and set BG_REQUIRED_JIT_SEMANTIC_PERMUTE_WEBGPU_EVIDENCE_COMMIT to its full commit SHA",
    );
  }
  if (!/^[0-9a-f]{40}$/u.test(jitEvidenceCommit) || jitEvidenceCommit !== head) {
    throw new Error(
      `kernels publish blocked: JIT semantic-permute evidence commit ${jitEvidenceCommit || "<missing>"} does not match HEAD ${head}`,
    );
  }
  if (githubSha && githubSha !== head) {
    throw new Error(
      `kernels publish blocked: GitHub SHA ${githubSha} does not match HEAD ${head}`,
    );
  }
  if (relevantStatus.trim().length > 0) {
    throw new Error(
      `kernels publish blocked: package/dependency files differ from evidenced HEAD\n${relevantStatus.trim()}`,
    );
  }
}

function main() {
  const repositoryRoot = new URL("../../..", import.meta.url);
  const git = (...args) => execFileSync("git", args, {
    cwd: repositoryRoot,
    encoding: "utf8",
  }).trim();
  validateViewCopyPublishGate({
    evidenceCommit: process.env.BG_REQUIRED_WEBGPU_EVIDENCE_COMMIT?.trim(),
    semanticHostGraphEvidenceCommit:
      process.env.BG_REQUIRED_SEMANTIC_HOST_GRAPH_WEBGPU_EVIDENCE_COMMIT
        ?.trim(),
    semanticHostGraphWorkerEvidenceCommit:
      process.env
        .BG_REQUIRED_SEMANTIC_HOST_GRAPH_WORKER_WEBGPU_EVIDENCE_COMMIT
        ?.trim(),
    semanticHostGraphPerformanceEvidenceCommit:
      process.env
        .BG_REQUIRED_SEMANTIC_HOST_GRAPH_WEBGPU_PERFORMANCE_EVIDENCE_COMMIT
        ?.trim(),
    semanticGemmEvidenceCommit: process.env.BG_REQUIRED_SEMANTIC_GEMM_WEBGPU_EVIDENCE_COMMIT?.trim(),
    semanticAttentionEvidenceCommit: process.env.BG_REQUIRED_SEMANTIC_ATTENTION_WEBGPU_EVIDENCE_COMMIT?.trim(),
    semanticAttentionPerformanceEvidenceCommit: process.env.BG_REQUIRED_SEMANTIC_ATTENTION_WEBGPU_PERFORMANCE_EVIDENCE_COMMIT?.trim(),
    jitEvidenceCommit: process.env.BG_REQUIRED_JIT_SEMANTIC_PERMUTE_WEBGPU_EVIDENCE_COMMIT?.trim(),
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
