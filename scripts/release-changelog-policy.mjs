import { Buffer } from "node:buffer";

const MAX_CHANGELOG_BYTES = 1024 * 1024;
const STABLE_SEMVER = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/u;
const RELEASE_HEADING = /^## (?:\[([0-9]+\.[0-9]+\.[0-9]+)\]|([0-9]+\.[0-9]+\.[0-9]+)) - ([0-9]{4}-[0-9]{2}-[0-9]{2})$/u;

export function validateReleaseChangelog({ source, expectedVersion, path = "CHANGELOG.md" }) {
  const failures = [];
  if (typeof source !== "string") {
    return [`${path} must be UTF-8 text`];
  }
  if (Buffer.byteLength(source, "utf8") > MAX_CHANGELOG_BYTES) {
    failures.push(`${path} exceeds ${MAX_CHANGELOG_BYTES} bytes`);
  }
  if (source.includes("\u0000")) failures.push(`${path} contains a NUL byte`);
  if (typeof expectedVersion !== "string" || !STABLE_SEMVER.test(expectedVersion)) {
    failures.push(`${path} expected version must be stable semver`);
    return failures;
  }

  const firstHeading = source.split(/\r?\n/u).find((line) => line.startsWith("## "));
  if (firstHeading === undefined) {
    failures.push(`${path} must start release history with an H2 release heading`);
    return failures;
  }
  const match = RELEASE_HEADING.exec(firstHeading);
  if (match === null) {
    failures.push(
      `${path} first release heading must be the current version and canonical release date; got ${firstHeading}`,
    );
    return failures;
  }
  const headingVersion = match[1] ?? match[2];
  const releaseDate = match[3];
  if (headingVersion !== expectedVersion) {
    failures.push(
      `${path} first release heading version ${headingVersion} does not match package ${expectedVersion}`,
    );
  }
  if (releaseDate === undefined || !isCanonicalIsoDate(releaseDate)) {
    failures.push(`${path} release date must be a real canonical YYYY-MM-DD date`);
  }
  return failures;
}

function isCanonicalIsoDate(value) {
  const parsed = new Date(`${value}T00:00:00.000Z`);
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}
