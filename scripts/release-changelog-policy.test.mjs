import assert from "node:assert/strict";
import { test } from "node:test";

import { validateReleaseChangelog } from "./release-changelog-policy.mjs";

test("accepts bracketed and bare current-version release headings", () => {
  assert.deepEqual(validateReleaseChangelog({
    source: "# Changes\n\n## [0.3.0] - 2026-08-25\n\n- Shipped.\n",
    expectedVersion: "0.3.0",
  }), []);
  assert.deepEqual(validateReleaseChangelog({
    source: "# Changes\n\n## 1.2.3 - 2026-08-25\n",
    expectedVersion: "1.2.3",
  }), []);
});

test("rejects unreleased, mismatched, undated, and impossible release headings", () => {
  for (const [source, expectedFailure] of [
    ["# Changes\n\n## [Unreleased]\n", "current version and canonical release date"],
    ["# Changes\n\n## [0.2.0] - 2026-08-25\n", "does not match package"],
    ["# Changes\n\n## 0.3.0\n", "current version and canonical release date"],
    ["# Changes\n\n## [0.3.0] - 2026-02-30\n", "real canonical"],
  ]) {
    assert.ok(validateReleaseChangelog({
      source,
      expectedVersion: "0.3.0",
    }).some((failure) => failure.includes(expectedFailure)));
  }
});

test("bounds changelog input and validates expected stable semver", () => {
  assert.ok(validateReleaseChangelog({
    source: `# Changes\n\n## 0.3.0 - 2026-08-25\n${"x".repeat(1024 * 1024)}`,
    expectedVersion: "0.3.0",
  }).some((failure) => failure.includes("exceeds")));
  assert.ok(validateReleaseChangelog({
    source: "# Changes\n\n## 0.3.0 - 2026-08-25\n",
    expectedVersion: "v0.3.0",
  }).some((failure) => failure.includes("stable semver")));
});
