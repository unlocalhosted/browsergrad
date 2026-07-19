import { createHash } from "node:crypto";
import { chmod, lstat, mkdir, mkdtemp, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  CppCuteBrowserDistributionOutputFilesError,
  materializeCppCuteBrowserDistributionOutputFiles,
} from "./cpp_cute_browser_distribution_output_files.mjs";

const ROOTS: string[] = [];

afterEach(async () => {
  await Promise.all(ROOTS.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("private distribution output files", () => {
  it("reverifies the exact initial tree and materializes a bounded no-clobber batch", async () => {
    const root = await fixtureRoot("success");
    await mkdir(join(root, "assets"), { mode: 0o700 });
    const existingBytes = new TextEncoder().encode("existing\n");
    await writeFile(join(root, "assets", "pack.bin"), existingBytes, { mode: 0o400 });
    const report = await materializeCppCuteBrowserDistributionOutputFiles({
      outputRoot: root,
      existingOutputs: [{
        outputPath: "assets/pack.bin",
        sha256: sha256(existingBytes),
        byteLength: String(existingBytes.byteLength),
      }],
      outputs: [
        { outputPath: "assets/review.json", bytes: new TextEncoder().encode("{\"ok\":true}\n") },
        { outputPath: "licenses/component.txt", bytes: new TextEncoder().encode("license\n") },
      ],
    });

    expect(report).toMatchObject({
      totals: {
        existingFileCount: 1,
        materializedFileCount: 2,
      },
      claims: {
        exactInitialTreeVerified: true,
        exactExistingFileBytesReverified: true,
        newFilesWrittenWithoutClobber: true,
        newFilesIndependentlyReread: true,
        exactFinalTreeVerified: true,
        callerPolicyBound: false,
        distributionAuthorized: false,
      },
    });
    expect(new TextDecoder().decode(await readFile(join(root, "licenses", "component.txt"))))
      .toBe("license\n");
    expect(Number((await lstat(join(root, "licenses"), { bigint: true })).mode & 0o077n)).toBe(0);
  });

  it("rejects unexpected files, unsafe entries, byte drift, traversal, and clobber", async () => {
    const root = await fixtureRoot("reject");
    const existing = new TextEncoder().encode("existing\n");
    await writeFile(join(root, "pack.bin"), existing, { mode: 0o400 });
    const base = {
      outputRoot: root,
      existingOutputs: [{
        outputPath: "pack.bin",
        sha256: sha256(existing),
        byteLength: String(existing.byteLength),
      }],
      outputs: [{ outputPath: "notice.txt", bytes: new TextEncoder().encode("notice\n") }],
    } as const;

    await writeFile(join(root, "unexpected"), "x", { mode: 0o400 });
    await expect(materializeCppCuteBrowserDistributionOutputFiles(base))
      .rejects.toBeInstanceOf(CppCuteBrowserDistributionOutputFilesError);
    await rm(join(root, "unexpected"));
    await chmod(join(root, "pack.bin"), 0o600);
    await expect(materializeCppCuteBrowserDistributionOutputFiles(base))
      .rejects.toBeInstanceOf(CppCuteBrowserDistributionOutputFilesError);
    await chmod(join(root, "pack.bin"), 0o400);
    await expect(materializeCppCuteBrowserDistributionOutputFiles({
      ...base,
      outputs: [{ outputPath: "../escape", bytes: new Uint8Array([1]) }],
    })).rejects.toBeInstanceOf(CppCuteBrowserDistributionOutputFilesError);
    await symlink(join(root, "pack.bin"), join(root, "link"));
    await expect(materializeCppCuteBrowserDistributionOutputFiles(base))
      .rejects.toBeInstanceOf(CppCuteBrowserDistributionOutputFilesError);
    await rm(join(root, "link"));
    await mkdir(join(root, "unexpected-empty"), { mode: 0o700 });
    await expect(materializeCppCuteBrowserDistributionOutputFiles(base))
      .rejects.toBeInstanceOf(CppCuteBrowserDistributionOutputFilesError);
  });
});

async function fixtureRoot(name: string): Promise<string> {
  const root = await realpath(await mkdtemp(join(tmpdir(), `browsergrad-output-${name}-`)));
  ROOTS.push(root);
  return root;
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}
