import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { linkSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";

const root = resolve(new URL("..", import.meta.url).pathname);
const publisher = join(root, "scripts/publish-missing-npm.mjs");
const semanticCore = "@unlocalhosted/browsergrad-semantic-core";
const semanticCoreVersion = "0.3.0";
const compiler = "@unlocalhosted/browsergrad-compiler";
const compilerVersion = "0.2.0";
const manifestName = "browsergrad-release-manifest.json";

test("selected staged release must target exactly the selected package", () => {
  withStagedDirectory((directory) => {
    writeManifest(directory, {
      schema: "browsergrad.staged-npm-release@1",
      sourceRevision: "0".repeat(40),
      selectedPackageName: semanticCore,
      targetSpecs: [`${compiler}@${compilerVersion}`],
      artifacts: [],
    });
    assertPublisherFails(
      directory,
      ["--package", semanticCore],
      "Selected-package staged release must target exactly the selected package",
    );
  });
});

test("staged manifest has a strict size bound", () => {
  withStagedDirectory((directory) => {
    writeFileSync(join(directory, manifestName), "x".repeat(1024 * 1024 + 1));
    assertPublisherFails(
      directory,
      [],
      "Staged release manifest must be single-link and at most 1048576 bytes",
    );
  });
});

test("staged manifest cannot be replaced by a hardlink", () => {
  const sourceDirectory = mkdtempSync(join(tmpdir(), "browsergrad-manifest-hardlink-source-"));
  try {
    withStagedDirectory((directory) => {
      const source = join(sourceDirectory, "manifest.json");
      writeFileSync(source, `${JSON.stringify({ schema: "foreign" }, null, 2)}\n`);
      linkSync(source, join(directory, manifestName));
      assertPublisherFails(
        directory,
        [],
        "Staged release manifest must be single-link and at most 1048576 bytes",
      );
    });
  } finally {
    rmSync(sourceDirectory, { recursive: true, force: true });
  }
});

test("staged manifest is closed and canonically encoded", () => {
  const baseManifest = {
    schema: "browsergrad.staged-npm-release@1",
    sourceRevision: "0".repeat(40),
    selectedPackageName: semanticCore,
    targetSpecs: [`${semanticCore}@${semanticCoreVersion}`],
    artifacts: [],
  };
  withStagedDirectory((directory) => {
    writeFileSync(join(directory, manifestName), JSON.stringify(baseManifest));
    assertPublisherFails(
      directory,
      ["--package", semanticCore],
      "Staged release manifest must use the canonical JSON encoding",
    );
  });
  withStagedDirectory((directory) => {
    writeManifest(directory, { ...baseManifest, ignoredAuthority: true });
    assertPublisherFails(
      directory,
      ["--package", semanticCore],
      "Invalid staged release manifest",
    );
  });
});

test("staged directory rejects undeclared files", () => {
  withStagedDirectory((directory) => {
    writeManifest(directory, {
      schema: "browsergrad.staged-npm-release@1",
      sourceRevision: "0".repeat(40),
      selectedPackageName: semanticCore,
      targetSpecs: [`${semanticCore}@${semanticCoreVersion}`],
      artifacts: [],
    });
    writeFileSync(join(directory, "undeclared.bin"), "unexpected");
    assertPublisherFails(
      directory,
      ["--package", semanticCore],
      "Staged release directory must contain only its manifest and declared artifacts",
    );
  });
});

test("staged tarballs cannot be hardlinks", () => {
  const sourceDirectory = mkdtempSync(join(tmpdir(), "browsergrad-hardlink-source-"));
  try {
    withStagedDirectory((directory) => {
      const source = join(sourceDirectory, "source.tgz");
      const artifact = join(directory, "semantic-core.tgz");
      const bytes = Buffer.from("not-a-tarball");
      writeFileSync(source, bytes);
      linkSync(source, artifact);
      writeManifest(directory, {
        schema: "browsergrad.staged-npm-release@1",
        sourceRevision: "0".repeat(40),
        selectedPackageName: semanticCore,
        targetSpecs: [`${semanticCore}@${semanticCoreVersion}`],
        artifacts: [{
          spec: `${semanticCore}@${semanticCoreVersion}`,
          file: "semantic-core.tgz",
          integrity: `sha512-${createHash("sha512").update(bytes).digest("base64")}`,
        }],
      });
      assertPublisherFails(
        directory,
        ["--package", semanticCore],
        "Staged artifact must be a bounded single-link regular file",
      );
    });
  } finally {
    rmSync(sourceDirectory, { recursive: true, force: true });
  }
});

function withStagedDirectory(operation) {
  const directory = mkdtempSync(join(tmpdir(), "browsergrad-publisher-test-"));
  try {
    operation(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

function writeManifest(directory, manifest) {
  writeFileSync(join(directory, manifestName), `${JSON.stringify(manifest, null, 2)}\n`);
}

function assertPublisherFails(directory, extraArguments, expectedMessage) {
  const result = spawnSync(
    process.execPath,
    [publisher, "--publish-staged", directory, "--provenance", ...extraArguments],
    { cwd: root, encoding: "utf8" },
  );
  assert.notEqual(result.status, 0, "publisher unexpectedly accepted an adversarial staged release");
  assert.match(`${result.stdout}${result.stderr}`, new RegExp(escapeRegExp(expectedMessage), "u"));
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}
