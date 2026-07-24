import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, readdir, realpath, rm, stat, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_PACKAGE_FACTORY_BYTE_LENGTH,
  CPP_CUTE_BROWSER_PACKAGE_FACTORY_SHA256,
  CppCuteBrowserPackageFactoryError,
  materializeCppCuteBrowserPackageFactory,
} from "./cpp_cute_browser_package_factory.mjs";
import {
  CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE,
} from "../../dist/resources/cpp_cute_browser_reproducibility_v3.js";

const TEST_ROOTS: string[] = [];
const SCRIPT_ROOT = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = resolve(SCRIPT_ROOT, "..", "..");
const EXACT_SOURCE = join(PACKAGE_ROOT, "src", "resources", "clang-extractor.mjs");

afterEach(async () => {
  await Promise.all(TEST_ROOTS.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("package-owned generated factory materialization", () => {
  it("derives its exact factory identity from the reviewed reproducibility resource", () => {
    const reproducible = CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.builds[0];
    expect(CPP_CUTE_BROWSER_PACKAGE_FACTORY_SHA256).toBe(
      reproducible.factoryModuleSha256,
    );
    expect(CPP_CUTE_BROWSER_PACKAGE_FACTORY_BYTE_LENGTH).toBe(
      reproducible.factoryModuleByteLength,
    );
  });

  it("copies the exact reviewed source once into an immutable fresh dist tree", async () => {
    const { destinationRoot } = await fixture();
    const report = await materializeCppCuteBrowserPackageFactory({ destinationRoot });
    const bytes = await readFile(report.destinationPath);
    const output = await stat(report.destinationPath);

    expect(report).toMatchObject({
      schema: "browsergrad.compiler.cpp-cute.package-factory-materialization",
      version: 1,
      authority: "package-materialization-only",
      factorySha256: CPP_CUTE_BROWSER_PACKAGE_FACTORY_SHA256,
      factoryByteLength: CPP_CUTE_BROWSER_PACKAGE_FACTORY_BYTE_LENGTH,
      exactSourceVerified: true,
      packageOwned: true,
      cleanBuildVerified: false,
      reproducibilityVerified: false,
      workerBundleVerified: false,
      workerExecutionObserved: false,
      releaseReady: false,
    });
    expect(bytes.byteLength).toBe(CPP_CUTE_BROWSER_PACKAGE_FACTORY_BYTE_LENGTH);
    expect(createHash("sha256").update(bytes).digest("hex"))
      .toBe(CPP_CUTE_BROWSER_PACKAGE_FACTORY_SHA256);
    expect(output.mode & 0o222).toBe(0);
  });

  it("never overwrites an existing package resource", async () => {
    const { destinationRoot } = await fixture();
    const first = await materializeCppCuteBrowserPackageFactory({ destinationRoot });
    const before = await readFile(first.destinationPath);
    await expect(materializeCppCuteBrowserPackageFactory({ destinationRoot }))
      .rejects.toSatisfy(expectPath("$.destinationPath"));
    expect(await readFile(first.destinationPath)).toEqual(before);
  });

  it("rejects modified source bytes before creating a destination", async () => {
    const { root, destinationRoot } = await fixture();
    const sourcePath = join(root, "modified-factory.mjs");
    await writeFile(sourcePath, "export default async function modified() {}\n", { mode: 0o444 });
    await expect(materializeCppCuteBrowserPackageFactory({ sourcePath, destinationRoot }))
      .rejects.toSatisfy(expectPath("$.sourcePath"));
    expect(await readdir(destinationRoot)).toEqual([]);
  });

  it("rejects symlinked source and destination authority", async () => {
    const first = await fixture();
    const sourceLink = join(first.root, "factory-link.mjs");
    await symlink(EXACT_SOURCE, sourceLink);
    await expect(materializeCppCuteBrowserPackageFactory({
      sourcePath: sourceLink,
      destinationRoot: first.destinationRoot,
    })).rejects.toSatisfy(expectPath("$.sourcePath"));

    const second = await fixture();
    const destinationLink = join(second.root, "destination-link");
    await symlink(second.destinationRoot, destinationLink);
    await expect(materializeCppCuteBrowserPackageFactory({
      destinationRoot: destinationLink,
    })).rejects.toSatisfy(expectPath("$.destinationRoot"));
  });

  it("rejects accessors, extra keys, relative paths, and non-record inputs", async () => {
    const cases: unknown[] = [
      Object.defineProperty({}, "destinationRoot", { get: () => "/tmp" }),
      { destinationRoot: "/tmp", extra: true },
      { destinationRoot: "relative" },
      Object.create(null),
    ];
    for (const value of cases) {
      await expect(materializeCppCuteBrowserPackageFactory(value as never))
        .rejects.toBeInstanceOf(CppCuteBrowserPackageFactoryError);
    }
  });
});

async function fixture(): Promise<{ readonly root: string; readonly destinationRoot: string }> {
  const created = await mkdtemp(join(tmpdir(), "browsergrad-package-factory-"));
  const root = await realpath(created);
  TEST_ROOTS.push(root);
  const destinationRoot = join(root, "resources");
  await mkdir(destinationRoot, { mode: 0o700 });
  return { root, destinationRoot };
}

function expectPath(path: string): (error: unknown) => boolean {
  return (error) => error instanceof CppCuteBrowserPackageFactoryError && error.path === path;
}
