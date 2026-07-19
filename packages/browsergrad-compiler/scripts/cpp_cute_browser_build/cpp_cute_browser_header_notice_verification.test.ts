import {
  chmod,
  mkdtemp,
  readFile,
  realpath,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  CppCuteBrowserHeaderNoticeVerificationError,
  canonicalCppCuteBrowserHeaderNoticeVerificationBytes,
  requireCppCuteBrowserHeaderNoticeVerificationAuthority,
  verifyCppCuteBrowserHeaderPackNotices,
} from "./cpp_cute_browser_header_notice_verification.mjs";

const TEST_ROOTS: string[] = [];
const SCRIPT_ROOT = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = resolve(SCRIPT_ROOT, "..", "..");
const RESOURCE_ROOT = join(PACKAGE_ROOT, "src", "resources", "licenses");
const NOTICE_NAMES = [
  "browsergrad-compiler.LICENSE",
  "clang.LICENSE.txt",
  "compiler-rt.LICENSE.txt",
  "cutlass.LICENSE.txt",
  "emscripten-musl.COPYRIGHT",
  "emscripten.LICENSE",
  "libcxx.LICENSE.txt",
  "libcxxabi.LICENSE.txt",
  "libunwind.LICENSE.txt",
  "llvm.LICENSE.txt",
];

afterEach(async () => {
  await Promise.all(TEST_ROOTS.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("browser header-pack approved notice bytes", () => {
  it("binds the exact package resources to the current build lock", async () => {
    const evidence = await verifyCppCuteBrowserHeaderPackNotices();
    expect(evidence).toMatchObject({
      schema: "browsergrad.compiler.cpp-cute.browser-header-notice-verification",
      version: 1,
      authority: "approved-header-notice-byte-verification-only",
      claims: {
        exactApprovedDistributionNoticeBytesVerified: true,
        unresolvedHeaderNoticeComponentCount: 2,
        allHeaderNoticesResolved: false,
        externalDistributedFileLicenseMapReviewed: false,
        licenseReviewComplete: false,
        distributionAuthorized: false,
        releaseReady: false,
      },
    });
    expect(evidence.notices.map((notice) => notice.componentId)).toEqual([
      "browsergrad-compiler",
      "clang",
      "compiler-rt",
      "cutlass",
      "emscripten",
      "emscripten-musl",
      "libcxx",
      "libcxxabi",
      "libunwind",
      "llvm",
    ]);
    expect(evidence.notices.find((notice) => notice.componentId === "clang")).toMatchObject({
      noticeSha256: "ebcd9bbf783a73d05c53ba4d586b8d5813dcdf3bbec50265860ccc885e606f47",
      noticeByteLength: "15140",
    });
    expect(evidence.unresolvedNotices.map((notice) => notice.componentId)).toEqual([
      "cuda-toolkit-12.6.3-headers",
      "linux-sysroot",
    ]);
    expect(canonicalCppCuteBrowserHeaderNoticeVerificationBytes(evidence))
      .toEqual(canonicalJsonBytes(evidence));
    expect(() => requireCppCuteBrowserHeaderNoticeVerificationAuthority({ ...evidence }))
      .toThrow(CppCuteBrowserHeaderNoticeVerificationError);
  });

  it("re-verifies an independently copied exact resource set", async () => {
    const root = await resourceFixture("copy");
    const evidence = await verifyCppCuteBrowserHeaderPackNotices({ resourceRoot: root });
    expect(evidence.notices).toHaveLength(10);
  });

  it("rejects modified, linked, or additional resources", async () => {
    const modified = await resourceFixture("modified");
    await chmod(join(modified, "cutlass.LICENSE.txt"), 0o600);
    await writeFile(join(modified, "cutlass.LICENSE.txt"), "modified\n");
    await expect(verifyCppCuteBrowserHeaderPackNotices({ resourceRoot: modified }))
      .rejects.toBeInstanceOf(CppCuteBrowserHeaderNoticeVerificationError);

    const linked = await resourceFixture("linked");
    await rm(join(linked, "clang.LICENSE.txt"));
    await symlink(join(RESOURCE_ROOT, "clang.LICENSE.txt"), join(linked, "clang.LICENSE.txt"));
    await expect(verifyCppCuteBrowserHeaderPackNotices({ resourceRoot: linked }))
      .rejects.toBeInstanceOf(CppCuteBrowserHeaderNoticeVerificationError);

    const additional = await resourceFixture("additional");
    await writeFile(join(additional, "unexpected.txt"), "unexpected\n");
    await expect(verifyCppCuteBrowserHeaderPackNotices({ resourceRoot: additional }))
      .rejects.toSatisfy(expectNoticePath("$.input.resourceRoot"));
  });

  it("rejects ambient object behavior and noncanonical paths", async () => {
    const hostile = Object.defineProperty({}, "resourceRoot", { get: () => RESOURCE_ROOT });
    await expect(verifyCppCuteBrowserHeaderPackNotices(hostile as never))
      .rejects.toSatisfy(expectNoticePath("$.input.resourceRoot"));
    await expect(verifyCppCuteBrowserHeaderPackNotices({ resourceRoot: "relative" }))
      .rejects.toSatisfy(expectNoticePath("$.input.resourceRoot"));
  });
});

async function resourceFixture(name: string): Promise<string> {
  const created = await mkdtemp(join(tmpdir(), `browsergrad-header-notices-${name}-`));
  const root = await realpath(created);
  TEST_ROOTS.push(root);
  await Promise.all(NOTICE_NAMES.map(async (fileName) => {
    const bytes = await readFile(join(RESOURCE_ROOT, fileName));
    await writeFile(join(root, fileName), bytes, { mode: 0o400 });
  }));
  return root;
}

function expectNoticePath(path: string): (error: unknown) => boolean {
  return (error) => error instanceof CppCuteBrowserHeaderNoticeVerificationError && error.path === path;
}
