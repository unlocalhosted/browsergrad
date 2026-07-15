import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { createReadOnlyNpmEnvironment } from "./npm-read-only-environment.mjs";

export const MINIMUM_ATTESTATION_AUDIT_NPM_VERSION = "11.12.0";

export function assertMinimumNpmVersion(
  actualVersion,
  minimumVersion = MINIMUM_ATTESTATION_AUDIT_NPM_VERSION,
) {
  const actual = parseExactSemver(actualVersion, "npm version");
  const minimum = parseExactSemver(minimumVersion, "minimum npm version");
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] > minimum[index]) return Object.freeze(actual);
    if (actual[index] < minimum[index]) {
      throw new Error(`npm >=${minimumVersion} is required, got ${actualVersion}`);
    }
  }
  return Object.freeze(actual);
}

function parseExactSemver(value, label) {
  if (typeof value !== "string") {
    throw new TypeError(`${label} must be a string`);
  }
  const normalized = value.trim();
  const match = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/u.exec(normalized);
  if (match === null) {
    throw new Error(`${label} must be exact three-component semver, got ${normalized}`);
  }
  const parts = match.slice(1).map(Number);
  if (parts.some((part) => !Number.isSafeInteger(part))) {
    throw new Error(`${label} components must be safe integers, got ${normalized}`);
  }
  return parts;
}

const invokedPath = process.argv[1] === undefined ? undefined : resolve(process.argv[1]);
if (invokedPath === fileURLToPath(import.meta.url)) {
  const minimum = process.argv[2] ?? MINIMUM_ATTESTATION_AUDIT_NPM_VERSION;
  if (process.argv.length > 3) {
    throw new Error("Usage: node scripts/require-npm-version.mjs [minimum-version]");
  }
  const result = spawnSync("npm", ["--version"], {
    encoding: "utf8",
    env: createReadOnlyNpmEnvironment(),
    shell: false,
    stdio: "pipe",
  });
  if (result.error !== undefined) {
    throw new Error("npm --version failed to execute", { cause: result.error });
  }
  if (result.status !== 0) {
    throw new Error(`npm --version failed with exit ${String(result.status)}\n${result.stderr ?? ""}`);
  }
  const actual = (result.stdout ?? "").trim();
  assertMinimumNpmVersion(actual, minimum);
  console.log(`npm ${actual} satisfies >=${minimum}`);
}
