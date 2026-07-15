import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { comparePackageTrees } from "./verify-published-package-equivalence.mjs";

test("compares packed trees by path, mode, and bytes", () => {
  const temp = mkdtempSync(join(tmpdir(), "browsergrad-package-tree-test-"));
  try {
    const left = join(temp, "left");
    const right = join(temp, "right");
    mkdirSync(join(left, "dist"), { recursive: true });
    mkdirSync(join(right, "dist"), { recursive: true });
    writeFileSync(join(left, "package.json"), "{\"name\":\"x\"}\n");
    writeFileSync(join(right, "package.json"), "{\"name\":\"x\"}\n");
    writeFileSync(join(left, "dist/index.js"), "export const x = 1;\n");
    writeFileSync(join(right, "dist/index.js"), "export const x = 1;\n");

    const equal = comparePackageTrees(left, right);
    assert.equal(equal.equal, true);
    assert.equal(equal.fileCount, 2);
    assert.match(equal.treeHash, /^[0-9a-f]{64}$/u);

    writeFileSync(join(right, "dist/index.js"), "export const x = 2;\n");
    const changed = comparePackageTrees(left, right);
    assert.equal(changed.equal, false);
    assert.deepEqual(changed.differences, ["content-or-mode dist/index.js"]);

    writeFileSync(join(right, "registry-only.txt"), "extra\n");
    const added = comparePackageTrees(left, right);
    assert.equal(added.equal, false);
    assert.ok(added.differences.includes("registry-only registry-only.txt"));
  } finally {
    rmSync(temp, { recursive: true, force: true });
  }
});
