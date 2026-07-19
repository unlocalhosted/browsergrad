import { createHash } from "node:crypto";
import {
  chmod,
  link,
  mkdir,
  mkdtemp,
  readFile,
  realpath,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  CppCuteBrowserSourceArchiveAdmissionError,
  admitCppCuteBrowserCurrentSourceArchives,
  canonicalCppCuteBrowserCurrentSourceArchiveAdmissionBytes,
  cppCuteBrowserCurrentSourceArchiveExpectations,
  inspectCppCuteBrowserSourceArchives,
  parseCppCuteBrowserSourceArchiveArguments,
  requireCppCuteBrowserCurrentSourceArchiveAuthority,
  type CppCuteBrowserSourceArchiveExpectation,
} from "./cpp_cute_browser_source_archive_admission.mjs";

const TEST_ROOTS: string[] = [];

afterEach(async () => {
  await Promise.all(TEST_ROOTS.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("browser source-archive admission", () => {
  it("inspects caller-expected archives without claiming the current build lock", async () => {
    const first = await fixture("first", [
      ["z-source", "archive-z"],
      ["a-source", "archive-a"],
    ]);
    const second = await fixture("second", [
      ["a-source", "archive-a"],
      ["z-source", "archive-z"],
    ]);
    const one = await inspectCppCuteBrowserSourceArchives({
      archives: [...first.archives].reverse(),
      expectedSources: [...first.expectations].reverse(),
    });
    const two = await inspectCppCuteBrowserSourceArchives({
      archives: second.archives,
      expectedSources: second.expectations,
    });

    expect(one).toEqual(two);
    expect(one.inspectionId).toMatch(/^bg\.cpp\.source-archive-inspection\.sha256\.[0-9a-f]{64}$/u);
    expect(one.archives.map((archive) => archive.sourceId)).toEqual(["a-source", "z-source"]);
    expect(one.totals).toEqual({ archiveCount: 2, archiveByteLength: "18" });
    expect(one.claims).toEqual({
      exactCallerExpectedArchiveBytesVerified: true,
      currentBuildInputLockBound: false,
      networkAccessed: false,
      archivesExtracted: false,
      sourceTreesVerified: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      buildExecuted: false,
      releaseReady: false,
    });
    expect(one).not.toHaveProperty("archivePath");
    expect(JSON.stringify(one)).not.toContain(first.root);
  });

  it("projects exactly the two package-lock source archives", async () => {
    const current = await cppCuteBrowserCurrentSourceArchiveExpectations();

    expect(current).toMatchObject({
      buildInputLockId:
        "bg.cpp.browser-build-input-lock.sha256.bf62353c9421b955cd1a07e14e13c5e3417b5431e2be4555283acdacc0ee7def",
      sources: [
        {
          sourceId: "cutlass",
          tag: "v3.7.0",
          archiveSha256: "dfcafb7435a1b114ce32faee4f3257e276caf08f55fea04fa8bf3efa3a83c814",
          archiveByteLength: "29728321",
        },
        {
          sourceId: "llvm-project",
          tag: "llvmorg-22.1.8",
          archiveSha256: "922f1817a0df7b1489272d18134ee0087a8b068828f87ac63b9861b1a9965888",
          archiveByteLength: "167061596",
          attestationSha256: "dd4aa06bd73706743090631300c02a6d8a3df43d41d85c627ec438d5a13b3739",
          attestationByteLength: "11234",
        },
      ],
    });
    expect(current.sources).toHaveLength(2);
  });

  it("rejects wrong bytes, lengths, duplicate inodes, and changed archives", async () => {
    const wrongHash = await fixture("wrong-hash", [["source", "archive"]]);
    const changedHash = [{ ...wrongHash.expectations[0]!, archiveSha256: "0".repeat(64) }];
    await expect(inspectCppCuteBrowserSourceArchives({
      archives: wrongHash.archives,
      expectedSources: changedHash,
    })).rejects.toSatisfy(expectMessage("SHA-256 differs"));

    const wrongLength = await fixture("wrong-length", [["source", "archive"]]);
    const changedLength = [{ ...wrongLength.expectations[0]!, archiveByteLength: "999" }];
    await expect(inspectCppCuteBrowserSourceArchives({
      archives: wrongLength.archives,
      expectedSources: changedLength,
    })).rejects.toSatisfy(expectMessage("byte length differs"));

    const duplicate = await fixture("duplicate", [["source-a", "same"]]);
    const duplicatePath = join(duplicate.root, "duplicate.bin");
    await link(duplicate.archives[0]!.archivePath, duplicatePath);
    const bytes = await readFile(duplicate.archives[0]!.archivePath);
    const duplicateExpectation = expectation("source-b", bytes);
    await expect(inspectCppCuteBrowserSourceArchives({
      archives: [
        duplicate.archives[0]!,
        { sourceId: "source-b", archivePath: duplicatePath },
      ],
      expectedSources: [duplicate.expectations[0]!, duplicateExpectation],
    })).rejects.toSatisfy(expectMessage("one hard link"));

    const changed = await fixture("changed", [["source", "before"]]);
    await inspectCppCuteBrowserSourceArchives({
      archives: changed.archives,
      expectedSources: changed.expectations,
    });
    await chmod(changed.archives[0]!.archivePath, 0o600);
    await writeFile(changed.archives[0]!.archivePath, "after!", { mode: 0o600 });
    await chmod(changed.archives[0]!.archivePath, 0o400);
    await expect(inspectCppCuteBrowserSourceArchives({
      archives: changed.archives,
      expectedSources: changed.expectations,
    })).rejects.toSatisfy(expectMessage("SHA-256 differs"));
  });

  it("rejects symlinks, hard links, writable paths, and non-files", async () => {
    const linked = await fixture("linked", [["source", "archive"]]);
    const linkPath = join(linked.root, "link.bin");
    await symlink(linked.archives[0]!.archivePath, linkPath);
    await expect(inspectCppCuteBrowserSourceArchives({
      archives: [{ sourceId: "source", archivePath: linkPath }],
      expectedSources: linked.expectations,
    })).rejects.toSatisfy(expectMessage("non-symlink regular file"));

    const hardLinked = await fixture("hard-linked", [["source", "archive"]]);
    await link(hardLinked.archives[0]!.archivePath, join(hardLinked.root, "alias.bin"));
    await expect(inspectCppCuteBrowserSourceArchives({
      archives: hardLinked.archives,
      expectedSources: hardLinked.expectations,
    })).rejects.toSatisfy(expectMessage("one hard link"));

    const writable = await fixture("writable", [["source", "archive"]]);
    await chmod(writable.archives[0]!.archivePath, 0o622);
    await expect(inspectCppCuteBrowserSourceArchives({
      archives: writable.archives,
      expectedSources: writable.expectations,
    })).rejects.toSatisfy(expectMessage("group- or world-writable"));

    const directory = await fixture("directory", [["source", "archive"]]);
    const directoryPath = join(directory.root, "not-an-archive");
    await mkdir(directoryPath, { mode: 0o700 });
    await expect(inspectCppCuteBrowserSourceArchives({
      archives: [{ sourceId: "source", archivePath: directoryPath }],
      expectedSources: directory.expectations,
    })).rejects.toSatisfy(expectMessage("regular file"));
  });

  it("keeps current-lock authority opaque and rejects incomplete specifications", async () => {
    const forged = Object.freeze({
      schema: "browsergrad.compiler.cpp-cute.current-source-archive-admission",
      claims: { exactCurrentBuildInputLockArchiveBytesVerified: true },
    });
    expect(() => requireCppCuteBrowserCurrentSourceArchiveAuthority(forged as never))
      .toThrow(CppCuteBrowserSourceArchiveAdmissionError);
    expect(() => canonicalCppCuteBrowserCurrentSourceArchiveAdmissionBytes(forged as never))
      .toThrow(CppCuteBrowserSourceArchiveAdmissionError);

    const tiny = await fixture("current-mismatch", [
      ["cutlass", "tiny-cutlass"],
      ["llvm-project", "tiny-llvm"],
    ]);
    await expect(admitCppCuteBrowserCurrentSourceArchives({ archives: tiny.archives }))
      .rejects.toSatisfy(expectMessage("byte length differs"));

    const hostile = Object.defineProperty({}, "archives", { get: () => [] });
    await expect(admitCppCuteBrowserCurrentSourceArchives(hostile as never))
      .rejects.toSatisfy(expectPath("$.input.archives"));
    await expect(inspectCppCuteBrowserSourceArchives({
      archives: [tiny.archives[0]!, tiny.archives[0]!],
      expectedSources: tiny.expectations,
    })).rejects.toSatisfy(expectMessage("duplicate source ID"));
  });

  it("parses only the two exact absolute-path command arguments", () => {
    expect(parseCppCuteBrowserSourceArchiveArguments([
      "--llvm-project=/tmp/llvm.tar.xz",
      "--cutlass=/tmp/cutlass.tar.gz",
    ])).toEqual({
      archives: [
        { sourceId: "cutlass", archivePath: "/tmp/cutlass.tar.gz" },
        { sourceId: "llvm-project", archivePath: "/tmp/llvm.tar.xz" },
      ],
    });
    expect(() => parseCppCuteBrowserSourceArchiveArguments([
      "--cutlass=/tmp/cutlass.tar.gz",
    ])).toThrow(CppCuteBrowserSourceArchiveAdmissionError);
    expect(() => parseCppCuteBrowserSourceArchiveArguments([
      "--cutlass=relative",
      "--llvm-project=/tmp/llvm.tar.xz",
    ])).toThrow(CppCuteBrowserSourceArchiveAdmissionError);
    expect(() => parseCppCuteBrowserSourceArchiveArguments([
      "--cutlass=/tmp/one",
      "--cutlass=/tmp/two",
    ])).toThrow(CppCuteBrowserSourceArchiveAdmissionError);
  });
});

async function fixture(
  name: string,
  entries: readonly (readonly [string, string])[],
): Promise<{
  readonly root: string;
  readonly archives: readonly { readonly sourceId: string; readonly archivePath: string }[];
  readonly expectations: readonly CppCuteBrowserSourceArchiveExpectation[];
}> {
  const root = await realpath(await mkdtemp(join(tmpdir(), `browsergrad-source-archive-${name}-`)));
  TEST_ROOTS.push(root);
  const archives = [];
  const expectations = [];
  for (const [sourceId, content] of entries) {
    const archivePath = join(root, `${sourceId}.archive`);
    const bytes = Buffer.from(content, "utf8");
    await writeFile(archivePath, bytes, { mode: 0o400 });
    archives.push(Object.freeze({ sourceId, archivePath }));
    expectations.push(expectation(sourceId, bytes));
  }
  return Object.freeze({ root, archives: Object.freeze(archives), expectations: Object.freeze(expectations) });
}

function expectation(sourceId: string, bytes: Uint8Array): CppCuteBrowserSourceArchiveExpectation {
  return Object.freeze({
    sourceId,
    repository: `https://example.com/${sourceId}`,
    acquisitionUrl: `https://example.com/${sourceId}/archive`,
    tag: "v1",
    commit: "1".repeat(40),
    treeSha1: "2".repeat(40),
    archiveSha256: createHash("sha256").update(bytes).digest("hex"),
    archiveByteLength: String(bytes.byteLength),
  });
}

function expectMessage(message: string) {
  return (error: unknown) => error instanceof CppCuteBrowserSourceArchiveAdmissionError &&
    error.message.includes(message);
}

function expectPath(path: string) {
  return (error: unknown) => error instanceof CppCuteBrowserSourceArchiveAdmissionError &&
    error.path === path;
}
