import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";

export const CPP_CUTE_BROWSER_HEADER_SOURCE_PLAN_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-header-source-plan";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-SOURCE-PLAN";
const PLAN_HASH_DOMAIN = "browsergrad.compiler.cpp-cute.browser-header-source-plan.v1";
const SHA256 = /^[0-9a-f]{64}$/u;
const SOURCE_ID = /^[a-z][a-z0-9-]*$/u;
const INCLUDE_ROOT_IDS = Object.freeze([
  "clang-resource",
  "cuda",
  "cutlass",
  "cxx-stdlib",
  "linux-sysroot",
]);
const INCLUDE_ROOT_ASSETS = Object.freeze(new Map([
  ["clang-resource", "compiler-resource-pack"],
  ["cuda", "dependency-header-pack:cuda"],
  ["cutlass", "dependency-header-pack:cutlass"],
  ["cxx-stdlib", "dependency-header-pack:cxx-stdlib"],
  ["linux-sysroot", "dependency-header-pack:linux-sysroot"],
]));
const PLAN_AUTHORITIES = new WeakSet();

const SUPPLEMENTAL_ARCHIVES = Object.freeze([
  Object.freeze({
    sourceId: "cuda-cccl-linux-x86-64",
    sourceKind: "nvidia-cuda-redist-component",
    provider: "NVIDIA",
    version: "12.6.77",
    acquisitionUrl:
      "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.6.77-archive.tar.xz",
    archiveFormat: "tar.xz",
    archiveSha256: "9c3145ef01f73e50c0f5fcf923f0899c847f487c529817daa8f8b1a3ecf20925",
    archiveByteLength: "934952",
    index: Object.freeze({
      url: "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json",
      sha256: "9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf",
      byteLength: "49142",
      releaseLabel: "12.6.3",
    }),
    licenseComponentId: "cuda-toolkit-12.6.3-headers",
    licensePolicy: "external-exact-file-redistribution-review-required",
    selections: Object.freeze([
      Object.freeze({
        includeRootId: "cuda",
        archiveSubtree: "cuda_cccl-linux-x86_64-12.6.77-archive/include",
        virtualPrefix: "",
        contribution: "complete-selected-header-subtree",
      }),
    ]),
  }),
  Object.freeze({
    sourceId: "cuda-cudart-linux-x86-64",
    sourceKind: "nvidia-cuda-redist-component",
    provider: "NVIDIA",
    version: "12.6.77",
    acquisitionUrl:
      "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.6.77-archive.tar.xz",
    archiveFormat: "tar.xz",
    archiveSha256: "f74689258a60fd9c5bdfa7679458527a55e22442691ba678dcfaeffbf4391ef9",
    archiveByteLength: "1126072",
    index: Object.freeze({
      url: "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json",
      sha256: "9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf",
      byteLength: "49142",
      releaseLabel: "12.6.3",
    }),
    licenseComponentId: "cuda-toolkit-12.6.3-headers",
    licensePolicy: "external-exact-file-redistribution-review-required",
    selections: Object.freeze([
      Object.freeze({
        includeRootId: "cuda",
        archiveSubtree: "cuda_cudart-linux-x86_64-12.6.77-archive/include",
        virtualPrefix: "",
        contribution: "complete-selected-header-subtree",
      }),
    ]),
  }),
  Object.freeze({
    sourceId: "cuda-nvcc-linux-x86-64",
    sourceKind: "nvidia-cuda-redist-component",
    provider: "NVIDIA",
    version: "12.6.85",
    acquisitionUrl:
      "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.6.85-archive.tar.xz",
    archiveFormat: "tar.xz",
    archiveSha256: "840deff234d9bef20d6856439c49881cb4f29423b214f9ecd2fa59b7ac323817",
    archiveByteLength: "49996208",
    index: Object.freeze({
      url: "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json",
      sha256: "9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf",
      byteLength: "49142",
      releaseLabel: "12.6.3",
    }),
    licenseComponentId: "cuda-toolkit-12.6.3-headers",
    licensePolicy: "external-exact-file-redistribution-review-required",
    selections: Object.freeze([
      Object.freeze({
        includeRootId: "cuda",
        archiveSubtree: "cuda_nvcc-linux-x86_64-12.6.85-archive/include",
        virtualPrefix: "",
        contribution: "complete-selected-header-subtree",
      }),
    ]),
  }),
  Object.freeze({
    sourceId: "ubuntu-noble-libc6-dev-amd64-cross",
    sourceKind: "ubuntu-debian-binary-package",
    provider: "Ubuntu",
    version: "2.39-0ubuntu8cross1",
    acquisitionUrl:
      "https://archive.ubuntu.com/ubuntu/pool/main/c/cross-toolchain-base/libc6-dev-amd64-cross_2.39-0ubuntu8cross1_all.deb",
    archiveFormat: "deb-data-tar-zstd",
    archiveSha256: "ceec73b7dbee49022fa52b8ce21961a118902f3ad1ff51e8f83d9dfc0270962d",
    archiveByteLength: "2158644",
    index: Object.freeze({
      suite: "ubuntu-24.04-noble",
      component: "main",
      package: "libc6-dev-amd64-cross",
      architecture: "all",
    }),
    licenseComponentId: "linux-sysroot",
    licensePolicy: "external-source-package-license-map-review-required",
    selections: Object.freeze([
      Object.freeze({
        includeRootId: "linux-sysroot",
        archiveSubtree: "data.tar.zst:./usr/x86_64-linux-gnu/include",
        virtualPrefix: "",
        contribution: "complete-selected-header-subtree",
      }),
    ]),
  }),
  Object.freeze({
    sourceId: "ubuntu-noble-linux-libc-dev-amd64-cross",
    sourceKind: "ubuntu-debian-binary-package",
    provider: "Ubuntu",
    version: "6.8.0-25.25cross1",
    acquisitionUrl:
      "https://archive.ubuntu.com/ubuntu/pool/main/c/cross-toolchain-base/linux-libc-dev-amd64-cross_6.8.0-25.25cross1_all.deb",
    archiveFormat: "deb-data-tar-zstd",
    archiveSha256: "bc504dcc35c15ff606df44ca081d0abaa613b2f3b7a56896d6211eced1368af3",
    archiveByteLength: "1400892",
    index: Object.freeze({
      suite: "ubuntu-24.04-noble",
      component: "main",
      package: "linux-libc-dev-amd64-cross",
      architecture: "all",
    }),
    licenseComponentId: "linux-sysroot",
    licensePolicy: "external-source-package-license-map-review-required",
    selections: Object.freeze([
      Object.freeze({
        includeRootId: "linux-sysroot",
        archiveSubtree: "data.tar.zst:./usr/x86_64-linux-gnu/include",
        virtualPrefix: "",
        contribution: "complete-selected-header-subtree",
      }),
    ]),
  }),
]);

export class CppCuteBrowserHeaderSourcePlanError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserHeaderSourcePlanError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Selects the exact upstream archives and complete archive subtrees intended
 * to feed all five browser header packs. This is source-selection policy only:
 * it reads no archive, does no extraction, and deliberately leaves generated
 * Clang resource headers and external license review unresolved.
 */
export async function prepareCppCuteBrowserHeaderSourcePlan() {
  const buildInputLock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const body = unwrapPreparedCppCuteBrowserBuildInputLock(buildInputLock).lock.body;
  const gitArchives = currentGitSourceArchives(body.sources);
  const archives = bindSelectionLicensePolicies(
    [...gitArchives, ...SUPPLEMENTAL_ARCHIVES],
    body,
  ).sort((left, right) => compareUtf8(left.sourceId, right.sourceId));
  validateArchiveSet(archives);
  const includeRoots = summarizeIncludeRoots(archives);
  const planBody = Object.freeze({
    buildInputLockId: buildInputLock.lockId,
    buildInputLockResourceSha256: buildInputLock.resourceSha256,
    archives: Object.freeze(archives),
    includeRoots,
    unresolvedBlockers: Object.freeze([
      Object.freeze({
        blockerId: "clang-resource-generated-headers",
        requirement: "pin-and-admit-the-complete-generated-clang-22-resource-header-output",
      }),
      Object.freeze({
        blockerId: "cuda-header-redistribution",
        requirement: "external-exact-file-review-of-the-three-selected-cuda-header-subtrees",
      }),
      Object.freeze({
        blockerId: "distributed-file-license-manifest",
        requirement: "external-per-file-license-map-for-every-selected-and-generated-header",
      }),
      Object.freeze({
        blockerId: "linux-sysroot-redistribution",
        requirement: "external-source-package-copyright-and-redistribution-closure",
      }),
    ]),
  });
  const planHash = sha256(canonicalJsonBytes({ domain: PLAN_HASH_DOMAIN, body: planBody }));
  const plan = Object.freeze({
    schema: CPP_CUTE_BROWSER_HEADER_SOURCE_PLAN_SCHEMA,
    version: 1,
    planId: `bg.cpp.browser-header-source-plan.sha256.${planHash}`,
    authority: "exact-header-source-selection-policy-only",
    body: planBody,
    totals: Object.freeze({
      archiveCount: archives.length,
      archiveByteLength: archives.reduce(
        (total, archive) => total + BigInt(archive.archiveByteLength),
        0n,
      ).toString(),
      includeRootCount: includeRoots.length,
      selectedSubtreeCount: archives.reduce(
        (total, archive) => total + archive.selections.length,
        0,
      ),
    }),
    claims: Object.freeze({
      exactBuildInputLockBound: true,
      exactArchiveSelectionPinned: true,
      exactSourceSubtreesPinned: true,
      exactHeaderPackLicensePolicyBound: true,
      allFiveIncludeRootsSelected: true,
      archiveBytesVerified: false,
      archiveAttestationsVerified: false,
      sourceSubtreesExtracted: false,
      generatedClangResourceHeadersComplete: false,
      externalDistributedFileLicenseMapReviewed: false,
      licenseReviewComplete: false,
      headerUniverseComplete: false,
      headerPacksAssembled: false,
      buildExecuted: false,
      releaseReady: false,
    }),
  });
  PLAN_AUTHORITIES.add(plan);
  return plan;
}

function bindSelectionLicensePolicies(archives, buildLockBody) {
  const approved = buildLockBody.notices.approvedComponents;
  const unresolved = buildLockBody.notices.unresolvedComponents;
  return archives.map((archive) => Object.freeze({
    ...archive,
    selections: Object.freeze(archive.selections.map((selection) => {
      const intendedAsset = INCLUDE_ROOT_ASSETS.get(selection.includeRootId);
      if (intendedAsset === undefined) {
        invalid("$.archives.selections", "selection has no header-pack asset policy");
      }
      const licenseComponentIds = [
        ...approved.filter((notice) => notice.appliesTo.includes(intendedAsset))
          .map((notice) => notice.componentId),
        ...unresolved.filter((notice) => notice.intendedAsset === intendedAsset)
          .map((notice) => notice.componentId),
      ].sort(compareUtf8);
      if (licenseComponentIds.length === 0 ||
          new Set(licenseComponentIds).size !== licenseComponentIds.length) {
        invalid("$.buildInputLock.notices", "header selection license policy is incomplete");
      }
      return Object.freeze({
        ...selection,
        intendedAsset,
        licenseComponentIds: Object.freeze(licenseComponentIds),
      });
    })),
  }));
}

export function requireCppCuteBrowserHeaderSourcePlanAuthority(plan) {
  if (typeof plan !== "object" || plan === null || !PLAN_AUTHORITIES.has(plan)) {
    invalid("$.plan", "expected preparer-issued exact header-source selection authority");
  }
}

export function canonicalCppCuteBrowserHeaderSourcePlanBytes(plan) {
  requireCppCuteBrowserHeaderSourcePlanAuthority(plan);
  return canonicalJsonBytes(plan);
}

function currentGitSourceArchives(sources) {
  if (!Array.isArray(sources) || sources.length !== 2) {
    invalid("$.buildInputLock.body.sources", "current build lock lost its exact Git source set");
  }
  const byId = new Map(sources.map((source) => [source.sourceId, source]));
  const cutlass = byId.get("cutlass");
  const llvm = byId.get("llvm-project");
  if (cutlass === undefined || llvm === undefined) {
    invalid("$.buildInputLock.body.sources", "current build lock lost LLVM or CUTLASS");
  }
  return Object.freeze([
    Object.freeze({
      sourceId: "cutlass",
      sourceKind: "git-source-archive",
      provider: "NVIDIA",
      version: cutlass.tag,
      repository: cutlass.repository,
      acquisitionUrl: cutlass.acquisitionUrl,
      archiveFormat: "tar.gz",
      archiveSha256: cutlass.archiveSha256,
      archiveByteLength: cutlass.archiveByteLength,
      commit: cutlass.commit,
      treeSha1: cutlass.treeSha1,
      licenseComponentId: "cutlass",
      licensePolicy: "current-build-lock-approved-notice-plus-external-file-map-required",
      selections: Object.freeze([
        Object.freeze({
          includeRootId: "cutlass",
          archiveSubtree: "cutlass-3.7.0/include",
          virtualPrefix: "",
          contribution: "complete-selected-header-subtree",
        }),
      ]),
    }),
    Object.freeze({
      sourceId: "llvm-project",
      sourceKind: "git-source-archive",
      provider: "LLVM",
      version: llvm.tag,
      repository: llvm.repository,
      acquisitionUrl: llvm.acquisitionUrl,
      archiveFormat: "tar.xz",
      archiveSha256: llvm.archiveSha256,
      archiveByteLength: llvm.archiveByteLength,
      commit: llvm.commit,
      treeSha1: llvm.treeSha1,
      attestationUrl: llvm.attestationUrl,
      attestationSha256: llvm.attestationSha256,
      attestationByteLength: llvm.attestationByteLength,
      licenseComponentId: "clang-and-libcxx",
      licensePolicy: "current-build-lock-approved-notices-plus-external-file-map-required",
      selections: Object.freeze([
        Object.freeze({
          includeRootId: "clang-resource",
          archiveSubtree: "llvm-project-22.1.8.src/clang/lib/Headers",
          virtualPrefix: "",
          contribution: "source-templates-requires-generated-resource-headers",
        }),
        Object.freeze({
          includeRootId: "cxx-stdlib",
          archiveSubtree: "llvm-project-22.1.8.src/libcxx/include",
          virtualPrefix: "",
          contribution: "complete-selected-header-subtree",
        }),
      ]),
    }),
  ]);
}

function validateArchiveSet(archives) {
  if (archives.length !== 7) invalid("$.archives", "header source plan must select exactly seven archives");
  for (const [index, archive] of archives.entries()) {
    const path = `$.archives[${index}]`;
    if (!SOURCE_ID.test(archive.sourceId) || !SHA256.test(archive.archiveSha256) ||
        !/^[1-9][0-9]*$/u.test(archive.archiveByteLength) ||
        BigInt(archive.archiveByteLength) > 512n * 1024n * 1024n ||
        !archive.acquisitionUrl.startsWith("https://") || archive.selections.length === 0) {
      invalid(path, "archive record is outside the exact source-plan contract");
    }
    if (index > 0 && archives[index - 1].sourceId === archive.sourceId) {
      invalid(path, "archive source IDs must be unique");
    }
    for (const selection of archive.selections) {
      if (!INCLUDE_ROOT_IDS.includes(selection.includeRootId) ||
          typeof selection.archiveSubtree !== "string" || selection.archiveSubtree === "" ||
          selection.archiveSubtree.startsWith("/") || selection.archiveSubtree.includes("..") ||
          selection.archiveSubtree.includes("\\") || selection.archiveSubtree.includes("\0") ||
          selection.virtualPrefix !== "") {
        invalid(path, "archive selection is outside the closed subtree contract");
      }
    }
  }
}

function summarizeIncludeRoots(archives) {
  const includeRoots = INCLUDE_ROOT_IDS.map((includeRootId) => {
    const contributors = archives.flatMap((archive) => archive.selections
      .filter((selection) => selection.includeRootId === includeRootId)
      .map((selection) => Object.freeze({
        sourceId: archive.sourceId,
        archiveSubtree: selection.archiveSubtree,
        virtualPrefix: selection.virtualPrefix,
        contribution: selection.contribution,
      })));
    if (contributors.length === 0) {
      invalid("$.archives", `header source plan does not cover ${JSON.stringify(includeRootId)}`);
    }
    return Object.freeze({
      includeRootId,
      contributors: Object.freeze(contributors),
      generatedInputsComplete: includeRootId !== "clang-resource",
      licenseReviewComplete: false,
      readyForPackAssembly: false,
    });
  });
  return Object.freeze(includeRoots);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function invalid(path, message, options) {
  throw new CppCuteBrowserHeaderSourcePlanError(path, message, options);
}

async function main() {
  try {
    if (process.argv.length !== 2) invalid("$arguments", "this command accepts no arguments");
    const plan = await prepareCppCuteBrowserHeaderSourcePlan();
    process.stdout.write(`${new TextDecoder().decode(canonicalCppCuteBrowserHeaderSourcePlanBytes(plan))}\n`);
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error("unknown header-source plan failure");
    const path = typeof cause === "object" && cause !== null && "path" in cause &&
      typeof cause.path === "string"
      ? ` at ${cause.path}`
      : "";
    process.stderr.write(`${error.name}${path}: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
