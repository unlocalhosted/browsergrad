import { execFileSync } from "node:child_process";
import {
  closeSync,
  constants,
  fstatSync,
  openSync,
  readFileSync,
  readSync,
} from "node:fs";
import { pathToFileURL } from "node:url";
import {
  canonicalizeJson,
  hashCanonicalJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION } from "@unlocalhosted/browsergrad-kernels";
import { readSemanticPermuteEvidenceSourceStatus } from "../../../scripts/semantic-permute-evidence-source.mjs";

const EVIDENCE_TOKEN = "[browsergrad-webgpu-evidence]";
export const SEMANTIC_PERMUTE_EVIDENCE_PREFIX = `${EVIDENCE_TOKEN} `;
const MAX_LOG_BYTES = 8 * 1024 * 1024;
const TERMINAL_HASH_DOMAIN = "browsergrad.execution-evidence.terminal-manifest.v1";
const EXPECTED_IDS = Object.freeze({
  schema: "browsergrad.execution-evidence@1",
  suiteId: "browsergrad.jit.semantic-permute.webgpu-conformance@1",
  capabilityId: "browsergrad.jit.tensor-plan.semantic-permute",
  backendId: "browsergrad.backend.webgpu.core",
  comparisonPolicyId: "browsergrad.comparison.bit-exact-u32-complete-root.v1",
});
const EXPECTED_CASE_IDS = Object.freeze(["rank2-transpose", "rank3-permutation"]);

export async function verifySemanticPermuteEvidenceLog(log, options) {
  if (typeof log !== "string") fail("retained log must be text");
  if (Buffer.byteLength(log, "utf8") > MAX_LOG_BYTES) {
    fail(`retained log exceeds ${MAX_LOG_BYTES} bytes`);
  }
  requireRevision(options?.expectedSourceRevision, "expected source revision");
  requireRevision(options?.gitHead, "git HEAD");
  if (options.expectedSourceRevision !== options.gitHead) {
    fail(`expected source revision ${options.expectedSourceRevision} differs from git HEAD ${options.gitHead}`);
  }
  if (typeof options.relevantStatus !== "string" || options.relevantStatus.trim().length > 0) {
    fail(`relevant source differs from expected git HEAD${
      typeof options.relevantStatus === "string" && options.relevantStatus.trim().length > 0
        ? `\n${options.relevantStatus.trim()}`
        : ""
    }`);
  }
  const producerVersions = requireRecord(options?.producerVersions, "expected producer versions");
  const literalCount = log.split(EVIDENCE_TOKEN).length - 1;
  if (literalCount !== 1) {
    fail(`expected exactly one literal prefixed terminal record, found ${literalCount}`);
  }
  const tokenIndex = log.indexOf(EVIDENCE_TOKEN);
  const jsonStart = tokenIndex + EVIDENCE_TOKEN.length;
  if (log[jsonStart] !== " ") {
    fail("literal terminal prefix must be followed by exactly one space");
  }
  const lineEnd = log.indexOf("\n", jsonStart + 1);
  const rawLineTail = log.slice(jsonStart + 1, lineEnd === -1 ? log.length : lineEnd);
  const json = rawLineTail.endsWith("\r") ? rawLineTail.slice(0, -1) : rawLineTail;
  if (json.length === 0 || json.trim() !== json) {
    fail("prefixed terminal record must contain compact JSON with no surrounding text");
  }
  let record;
  try {
    record = JSON.parse(json);
  } catch (error) {
    fail(`prefixed terminal record is not exact JSON: ${message(error)}`);
  }
  if (JSON.stringify(record) !== json) {
    fail("prefixed terminal record must be compact JSON with no trailing characters");
  }
  requireRecord(record, "terminal record");
  requireEqual(record.schema, EXPECTED_IDS.schema, "schema");
  requireEqual(record.kind, "terminal", "kind");
  requireEqual(record.suiteId, EXPECTED_IDS.suiteId, "suiteId");
  requireEqual(record.required, true, "required");
  requireEqual(record.stage, "terminal-summary", "stage");
  requireEqual(record.artifactHashKind, "prepared-case-set", "artifactHashKind");
  if (Object.hasOwn(record, "currentCaseId") || Object.hasOwn(record, "error")) {
    fail("passed terminal record cannot carry currentCaseId or error");
  }
  requireStringArray(record.uncapturedErrors, "uncapturedErrors", true);
  const plannedCaseIds = requireStringArray(record.plannedCaseIds, "plannedCaseIds");
  if (!equalArray(plannedCaseIds, EXPECTED_CASE_IDS)) fail("plannedCaseIds differ from the exact suite");
  if (!Array.isArray(record.completedCases)) fail("completedCases must be an array");
  if (!equalArray(record.completedCases.map((entry) => requireRecord(entry, "completed case").caseId), EXPECTED_CASE_IDS)) {
    fail("completedCases must cover the exact ordered suite");
  }

  const evidence = requireRecord(record.evidence, "evidence");
  requireEqual(evidence.capabilityId, EXPECTED_IDS.capabilityId, "evidence.capabilityId");
  requireEqual(evidence.backendId, EXPECTED_IDS.backendId, "evidence.backendId");
  requireEqual(
    evidence.comparisonPolicyId,
    EXPECTED_IDS.comparisonPolicyId,
    "evidence.comparisonPolicyId",
  );
  requireEqual(evidence.outcome, "passed", "evidence.outcome");
  requireEqual(
    evidence.sourceRevision,
    options.expectedSourceRevision,
    "evidence.sourceRevision",
  );
  requireStringArray(evidence.diagnosticCodes, "evidence.diagnosticCodes", true);
  requireDigest(evidence.artifactHash, "evidence.artifactHash");
  requireDigest(evidence.environmentId, "evidence.environmentId");
  requireDigest(evidence.deviceProfileHash, "evidence.deviceProfileHash");
  const actualProducerVersions = requireRecord(
    evidence.producerVersions,
    "evidence.producerVersions",
  );
  if (canonicalizeJson(actualProducerVersions) !== canonicalizeJson(producerVersions)) {
    fail("evidence.producerVersions differ from exact workspace package/backend versions");
  }

  requireDigest(record.terminalManifestHash, "terminalManifestHash");
  const { terminalManifestHash: _excluded, ...unsigned } = record;
  const expectedTerminalManifestHash = await hashCanonicalJson({
    domain: TERMINAL_HASH_DOMAIN,
    terminalRecord: unsigned,
  });
  if (record.terminalManifestHash !== expectedTerminalManifestHash) {
    fail("terminalManifestHash does not bind the complete terminal record");
  }
  return record;
}

export function loadSemanticPermuteProducerVersions(repositoryRoot) {
  const version = (relativePath) => {
    const parsed = JSON.parse(readFileSync(new URL(relativePath, repositoryRoot), "utf8"));
    if (typeof parsed.version !== "string" || parsed.version.length === 0) {
      fail(`${relativePath} has no package version`);
    }
    return parsed.version;
  };
  return Object.freeze({
    "@unlocalhosted/browsergrad-jit": version("packages/browsergrad-jit/package.json"),
    "@unlocalhosted/browsergrad-kernels": version("packages/browsergrad-kernels/package.json"),
    "@unlocalhosted/browsergrad-semantic-core": version("packages/browsergrad-semantic-core/package.json"),
    "browsergrad.backend.webgpu.view-copy": SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION,
  });
}

async function main() {
  const repositoryRoot = new URL("../../..", import.meta.url);
  const [logPath, expectedSourceRevision] = process.argv.slice(2);
  if (!logPath || !expectedSourceRevision) {
    fail("usage: verify_semantic_permute_evidence_log.mjs <log> <expected-source-revision>");
  }
  const gitHead = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repositoryRoot,
    encoding: "utf8",
  }).trim();
  await verifySemanticPermuteEvidenceLog(readBoundedSemanticPermuteEvidenceLog(logPath), {
    expectedSourceRevision,
    gitHead,
    relevantStatus: readSemanticPermuteEvidenceSourceStatus(repositoryRoot),
    producerVersions: loadSemanticPermuteProducerVersions(repositoryRoot),
  });
}

export function readBoundedSemanticPermuteEvidenceLog(logPath) {
  const descriptor = openSync(logPath, constants.O_RDONLY | constants.O_NONBLOCK);
  try {
    const initial = fstatSync(descriptor);
    if (!initial.isFile()) fail("retained log must be a regular file");
    const size = initial.size;
    if (size > MAX_LOG_BYTES) {
      fail(`retained log exceeds ${MAX_LOG_BYTES} bytes`);
    }
    const bytes = Buffer.allocUnsafe(size);
    let offset = 0;
    while (offset < size) {
      const read = readSync(descriptor, bytes, offset, size - offset, offset);
      if (read === 0) fail("retained log changed while being read");
      offset += read;
    }
    const final = fstatSync(descriptor);
    if (final.size !== size) fail("retained log changed while being read");
    const extra = Buffer.allocUnsafe(1);
    if (readSync(descriptor, extra, 0, 1, offset) !== 0) {
      fail("retained log exceeds its validated size");
    }
    return bytes.toString("utf8");
  } finally {
    closeSync(descriptor);
  }
}

function requireRecord(value, name) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(`${name} must be an object`);
  }
  return value;
}

function requireStringArray(value, name, requireEmpty = false) {
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== "string" || entry.length === 0)) {
    fail(`${name} must be an array of nonempty strings`);
  }
  if (requireEmpty && value.length !== 0) fail(`${name} must be empty for passed evidence`);
  return value;
}

function requireRevision(value, name) {
  if (typeof value !== "string" || !/^[0-9a-f]{40}$/u.test(value)) {
    fail(`${name} must be a full lowercase git SHA`);
  }
}

function requireDigest(value, name) {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/u.test(value)) {
    fail(`${name} must be a full lowercase SHA-256 digest`);
  }
}

function requireEqual(actual, expected, name) {
  if (actual !== expected) fail(`${name} must equal ${String(expected)}`);
}

function equalArray(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function message(error) {
  return error instanceof Error ? error.message : String(error);
}

function fail(message) {
  throw new Error(`JIT semantic-permute evidence log rejected: ${message}`);
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  await main();
}
