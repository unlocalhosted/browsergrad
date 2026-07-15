import assert from "node:assert/strict";
import test from "node:test";

import {
  assertMinimumNpmVersion,
  MINIMUM_ATTESTATION_AUDIT_NPM_VERSION,
} from "./require-npm-version.mjs";

test("compares all three npm semver components", () => {
  assert.deepEqual(
    assertMinimumNpmVersion("11.12.0\n"),
    [11, 12, 0],
  );
  assert.deepEqual(assertMinimumNpmVersion("11.12.1"), [11, 12, 1]);
  assert.deepEqual(assertMinimumNpmVersion("11.13.0"), [11, 13, 0]);
  assert.deepEqual(assertMinimumNpmVersion("12.0.0"), [12, 0, 0]);
  assert.equal(MINIMUM_ATTESTATION_AUDIT_NPM_VERSION, "11.12.0");

  for (const version of ["11.11.99", "11.5.10", "10.99.99"]) {
    assert.throws(
      () => assertMinimumNpmVersion(version),
      /npm >=11\.12\.0 is required/u,
      version,
    );
  }
});

test("rejects ambiguous or non-release npm version output", () => {
  for (const version of ["v11.12.0", "11.12", "11.12.0-beta.1", "011.12.0", "11.12.0 extra"]) {
    assert.throws(
      () => assertMinimumNpmVersion(version),
      /exact three-component semver/u,
      version,
    );
  }
});
