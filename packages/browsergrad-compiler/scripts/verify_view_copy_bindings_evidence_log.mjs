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
  hashNamedComponents,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  compileCudaLiteKernelWithViewCopyBinding,
  prepareCudaLiteViewCopyBinding,
} from "../dist/index.js";
import {
  createViewCopyConformanceCases,
} from "../../../test-support/view-copy-conformance-fixtures.ts";
import { readCompilerViewCopyEvidenceSourceStatus } from "../../../scripts/compiler-view-copy-evidence-source.mjs";

const EVIDENCE_TOKEN = "[browsergrad-webgpu-evidence]";
export const COMPILER_VIEW_COPY_EVIDENCE_PREFIX = `${EVIDENCE_TOKEN} `;
const MAX_LOG_BYTES = 8 * 1024 * 1024;
const TERMINAL_HASH_DOMAIN = "browsergrad.compiler.view-copy-bindings.terminal-manifest.v1";
const EXPECTED_IDS = Object.freeze({
  schema: "browsergrad.execution-evidence@1",
  suiteId: "browsergrad.compiler.view-copy-bindings.webgpu-conformance@1",
  capabilityId: "browsergrad.compiler.verified-view-copy-binding",
  backendId: "browsergrad.backend.webgpu.core",
  comparisonPolicyId: "browsergrad.comparison.bit-exact-u32-complete-root.v1",
});
const EXPECTED_CASE_IDS = Object.freeze([
  "rank2-transpose-control",
  "rank2-padding-exact-nan",
  "rank3-padding-exact-nan",
]);
const TOP_LEVEL_KEYS = Object.freeze([
  "schema",
  "kind",
  "suiteId",
  "required",
  "evidence",
  "environment",
  "artifactHashKind",
  "preparedBackendArtifactHash",
  "caseSetHash",
  "preparedCases",
  "plannedCaseIds",
  "completedCases",
  "stage",
  "uncapturedErrors",
  "terminalManifestHash",
]);
const EVIDENCE_KEYS = Object.freeze([
  "capabilityId",
  "artifactHash",
  "backendId",
  "environmentId",
  "producerVersions",
  "sourceRevision",
  "deviceProfileHash",
  "recordedAt",
  "outcome",
  "comparisonPolicyId",
  "diagnosticCodes",
]);
const ENVIRONMENT_KEYS = Object.freeze([
  "schema",
  "acquisition",
  "userAgent",
  "platform",
  "adapter",
  "adapterSupportedFeatures",
  "negotiatedDeviceFeatures",
  "negotiatedDeviceLimits",
]);
const ADAPTER_KEYS = Object.freeze(["vendor", "architecture", "device", "description"]);
const DEVICE_LIMIT_KEYS = Object.freeze([
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxComputeWorkgroupsPerDimension",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxBindingsPerBindGroup",
  "maxStorageBuffersPerShaderStage",
]);
const PREPARED_CASE_KEYS = Object.freeze([
  "caseId",
  "layoutSemanticHash",
  "kernelSemanticHash",
  "specializationHash",
  "bindingProjectionHash",
  "compileIdentityHash",
  "wgslModuleHash",
  "programName",
  "sourceHash",
  "initialDestinationHash",
  "expectedSourceHash",
  "expectedDestinationHash",
  "logicalShape",
  "logicalInvocationCount",
  "plannedWorkgroupCount",
  "expectedReadElements",
  "expectedFilledElements",
  "caseArtifactHash",
]);
const OBSERVATION_KEYS = Object.freeze([
  ...PREPARED_CASE_KEYS,
  "actualSourceHash",
  "actualDestinationHash",
  "planKind",
  "stepCount",
  "plannedPipelineCount",
  "comparisonPolicyId",
]);
let expectedPreparedCasesPromise;

export async function verifyCompilerViewCopyBindingsEvidenceLog(log, options) {
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
  if (Object.hasOwn(record, "currentCaseId") || Object.hasOwn(record, "error")) {
    fail("passed terminal record cannot carry currentCaseId or error");
  }
  requireExactKeys(record, TOP_LEVEL_KEYS, "terminal record");
  requireEqual(record.schema, EXPECTED_IDS.schema, "schema");
  requireEqual(record.kind, "terminal", "kind");
  requireEqual(record.suiteId, EXPECTED_IDS.suiteId, "suiteId");
  requireEqual(record.required, true, "required");
  requireEqual(record.stage, "terminal-summary", "stage");
  requireEqual(record.artifactHashKind, "prepared-case-set", "artifactHashKind");
  requireStringArray(record.uncapturedErrors, "uncapturedErrors", true);
  const plannedCaseIds = requireStringArray(record.plannedCaseIds, "plannedCaseIds");
  if (!equalArray(plannedCaseIds, EXPECTED_CASE_IDS)) {
    fail("plannedCaseIds differ from the exact suite");
  }
  const evidence = requireRecord(record.evidence, "evidence");
  requireExactKeys(evidence, EVIDENCE_KEYS, "evidence");
  requireEqual(evidence.capabilityId, EXPECTED_IDS.capabilityId, "evidence.capabilityId");
  requireEqual(evidence.backendId, EXPECTED_IDS.backendId, "evidence.backendId");
  requireEqual(
    evidence.comparisonPolicyId,
    EXPECTED_IDS.comparisonPolicyId,
    "evidence.comparisonPolicyId",
  );
  requireEqual(evidence.outcome, "passed", "evidence.outcome");
  requireEqual(evidence.sourceRevision, options.expectedSourceRevision, "evidence.sourceRevision");
  requireStringArray(evidence.diagnosticCodes, "evidence.diagnosticCodes", true);
  requireDigest(evidence.artifactHash, "evidence.artifactHash");
  requireDigest(evidence.environmentId, "evidence.environmentId");
  requireDigest(evidence.deviceProfileHash, "evidence.deviceProfileHash");
  requireIsoTimestamp(evidence.recordedAt, "evidence.recordedAt");
  const actualProducerVersions = requireRecord(
    evidence.producerVersions,
    "evidence.producerVersions",
  );
  if (canonicalizeJson(actualProducerVersions) !== canonicalizeJson(producerVersions)) {
    fail("evidence.producerVersions differ from exact workspace package versions");
  }

  const environment = requireAvailableEnvironment(record.environment);
  const [expectedEnvironmentId, expectedDeviceProfileHash] = await Promise.all([
    hashNamedComponents({ environment }),
    hashNamedComponents({
      backendId: EXPECTED_IDS.backendId,
      adapter: environment.adapter,
      selectedFeatures: [],
      adapterSupportedFeatures: environment.adapterSupportedFeatures,
      negotiatedDeviceFeatures: environment.negotiatedDeviceFeatures,
      negotiatedDeviceLimits: environment.negotiatedDeviceLimits,
    }),
  ]);
  requireEqual(evidence.environmentId, expectedEnvironmentId, "evidence.environmentId");
  requireEqual(
    evidence.deviceProfileHash,
    expectedDeviceProfileHash,
    "evidence.deviceProfileHash",
  );

  const expected = await deriveCompilerViewCopyBindingsExpectedEvidence(
    options.expectedSourceRevision,
    producerVersions,
  );
  requireDigest(record.preparedBackendArtifactHash, "preparedBackendArtifactHash");
  requireDigest(record.caseSetHash, "caseSetHash");
  requireEqual(
    record.preparedBackendArtifactHash,
    expected.preparedBackendArtifactHash,
    "preparedBackendArtifactHash",
  );
  requireEqual(record.caseSetHash, expected.caseSetHash, "caseSetHash");
  requireEqual(evidence.artifactHash, expected.artifactHash, "evidence.artifactHash");

  if (!Array.isArray(record.preparedCases)) fail("preparedCases must be an array");
  if (record.preparedCases.length !== expected.preparedCases.length) {
    fail("preparedCases must cover the exact ordered suite");
  }
  record.preparedCases.forEach((entry, index) => {
    const preparedCase = requireRecord(entry, `preparedCases[${index}]`);
    requireExactKeys(preparedCase, PREPARED_CASE_KEYS, `preparedCases[${index}]`);
    if (canonicalizeJson(preparedCase) !== canonicalizeJson(expected.preparedCases[index])) {
      fail(`preparedCases[${index}] differs from the checked-out source-derived manifest`);
    }
  });

  if (!Array.isArray(record.completedCases)) fail("completedCases must be an array");
  if (record.completedCases.length !== expected.preparedCases.length) {
    fail("completedCases must cover the exact ordered suite");
  }
  record.completedCases.forEach((entry, index) => {
    const observation = requireRecord(entry, `completedCases[${index}]`);
    requireExactKeys(observation, OBSERVATION_KEYS, `completedCases[${index}]`);
    const preparedCase = expected.preparedCases[index];
    const expectedObservation = {
      ...preparedCase,
      actualSourceHash: preparedCase.expectedSourceHash,
      actualDestinationHash: preparedCase.expectedDestinationHash,
      planKind: "single-dispatch",
      stepCount: 1,
      plannedPipelineCount: 1,
      comparisonPolicyId: EXPECTED_IDS.comparisonPolicyId,
    };
    if (canonicalizeJson(observation) !== canonicalizeJson(expectedObservation)) {
      fail(`completedCases[${index}] differs from the exact prepared execution observation`);
    }
  });

  requireDigest(record.terminalManifestHash, "terminalManifestHash");
  const { terminalManifestHash: _excluded, ...unsigned } = record;
  const expectedTerminalManifestHash = await hashCanonicalJson({
    domain: TERMINAL_HASH_DOMAIN,
    sourceRevision: evidence.sourceRevision,
    terminalRecord: unsigned,
  });
  if (record.terminalManifestHash !== expectedTerminalManifestHash) {
    fail("terminalManifestHash does not bind the complete terminal record");
  }
  // This self-hash detects retained-line corruption. Execution provenance still
  // comes from the trusted CI step that owns the required browser execution.
  return record;
}

export async function deriveCompilerViewCopyBindingsExpectedEvidence(
  sourceRevision,
  producerVersions,
) {
  requireRevision(sourceRevision, "source revision");
  requireRecord(producerVersions, "producer versions");
  const preparedCases = await expectedPreparedCasesForCheckedOutSource();
  const preparedBackendArtifactHash = await hashNamedComponents({
    suiteId: EXPECTED_IDS.suiteId,
    backendId: EXPECTED_IDS.backendId,
    artifacts: preparedCases.map(({ caseId, caseArtifactHash, wgslModuleHash, programName }) => ({
      caseId,
      caseArtifactHash,
      wgslModuleHash,
      programName,
    })),
  });
  const caseSetHash = await hashNamedComponents({
    suiteId: EXPECTED_IDS.suiteId,
    comparisonPolicyId: EXPECTED_IDS.comparisonPolicyId,
    sourceRevision,
    producerVersions,
    cases: preparedCases,
  });
  const artifactHash = await hashNamedComponents({
    suiteId: EXPECTED_IDS.suiteId,
    artifactHashKind: "prepared-case-set",
    preparedBackendArtifactHash,
    caseSetHash,
    comparisonPolicyId: EXPECTED_IDS.comparisonPolicyId,
    sourceRevision,
    producerVersions,
  });
  return Object.freeze({
    preparedCases,
    preparedBackendArtifactHash,
    caseSetHash,
    artifactHash,
  });
}

function expectedPreparedCasesForCheckedOutSource() {
  expectedPreparedCasesPromise ??= createViewCopyConformanceCases().then((fixtures) => {
    if (!equalArray(fixtures.map(({ id }) => id), EXPECTED_CASE_IDS)) {
      fail("checked-out source fixture differs from the exact ordered suite");
    }
    return Promise.all(fixtures.map(deriveExpectedPreparedCase));
  }).then((cases) => Object.freeze(cases));
  return expectedPreparedCasesPromise;
}

async function deriveExpectedPreparedCase(fixture) {
  const binding = await prepareCudaLiteViewCopyBinding(
    fixture.artifacts.layout,
    fixture.artifacts.kernel,
    {
      operationId: fixture.artifacts.operationId,
      sourceParameter: "input",
      destinationParameter: "output",
      indexing: "row-major-flat",
    },
  );
  const elementCount = fixture.logicalShape.reduce((product, extent) => product * extent, 1);
  const compiled = compileCudaLiteKernelWithViewCopyBinding(
    directViewCopySource(elementCount),
    binding,
    { workgroupSize: [elementCount, 1, 1] },
  );
  if (compiled.wgsl === undefined || compiled.wgslProgram === undefined) {
    fail(`${fixture.id} source-derived preparation produced no WGSL program`);
  }
  const [
    compileIdentityHash,
    wgslModuleHash,
    sourceHash,
    initialDestinationHash,
    expectedSourceHash,
    expectedDestinationHash,
  ] = await Promise.all([
    hashNamedComponents({ compileCacheKey: compiled.viewCopyBindingCompileCacheKey }),
    hashNamedComponents({ wgsl: compiled.wgsl }),
    hashWords(fixture.sourceWords),
    hashWords(fixture.initialDestinationWords),
    hashWords(fixture.expectedSourceWords),
    hashWords(fixture.expectedDestinationWords),
  ]);
  const manifest = {
    caseId: fixture.id,
    layoutSemanticHash: binding.layoutSemanticHash,
    kernelSemanticHash: binding.kernelSemanticHash,
    specializationHash: binding.specializationHash,
    bindingProjectionHash: binding.bindingProjectionHash,
    compileIdentityHash,
    wgslModuleHash,
    programName: compiled.wgslProgram.name,
    sourceHash,
    initialDestinationHash,
    expectedSourceHash,
    expectedDestinationHash,
    logicalShape: Object.freeze([...fixture.logicalShape]),
    logicalInvocationCount: Object.freeze([elementCount, 1, 1]),
    plannedWorkgroupCount: Object.freeze([1, 1, 1]),
    expectedReadElements: fixture.expectedReadElements,
    expectedFilledElements: fixture.expectedFilledElements,
  };
  return Object.freeze({
    ...manifest,
    caseArtifactHash: await hashNamedComponents({
      suiteId: EXPECTED_IDS.suiteId,
      case: manifest,
    }),
  });
}

function directViewCopySource(elementCount) {
  return `
__global__ void copy_view(const float* input, float* output) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${elementCount}u) output[i] = input[i];
}`;
}

function hashWords(words) {
  return hashNamedComponents({
    byteLength: words.byteLength,
    words: [...words],
  });
}

export function loadCompilerViewCopyProducerVersions(repositoryRoot) {
  const version = (relativePath) => {
    const parsed = JSON.parse(readFileSync(new URL(relativePath, repositoryRoot), "utf8"));
    if (typeof parsed.version !== "string" || parsed.version.length === 0) {
      fail(`${relativePath} has no package version`);
    }
    return parsed.version;
  };
  return Object.freeze({
    "@unlocalhosted/browsergrad-compiler": version("packages/browsergrad-compiler/package.json"),
    "@unlocalhosted/browsergrad-kernels": version("packages/browsergrad-kernels/package.json"),
    "@unlocalhosted/browsergrad-semantic-core": version("packages/browsergrad-semantic-core/package.json"),
  });
}

async function main() {
  const repositoryRoot = new URL("../../..", import.meta.url);
  const [logPath, expectedSourceRevision] = process.argv.slice(2);
  if (!logPath || !expectedSourceRevision) {
    fail("usage: verify_view_copy_bindings_evidence_log.mjs <log> <expected-source-revision>");
  }
  const gitHead = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repositoryRoot,
    encoding: "utf8",
  }).trim();
  await verifyCompilerViewCopyBindingsEvidenceLog(
    readBoundedCompilerViewCopyEvidenceLog(logPath),
    {
      expectedSourceRevision,
      gitHead,
      relevantStatus: readCompilerViewCopyEvidenceSourceStatus(repositoryRoot),
      producerVersions: loadCompilerViewCopyProducerVersions(repositoryRoot),
    },
  );
}

export function readBoundedCompilerViewCopyEvidenceLog(logPath) {
  const descriptor = openSync(logPath, constants.O_RDONLY | constants.O_NONBLOCK);
  try {
    const initial = fstatSync(descriptor, { bigint: true });
    if (!initial.isFile()) fail("retained log must be a regular file");
    const size = Number(initial.size);
    if (!Number.isSafeInteger(size) || size > MAX_LOG_BYTES) {
      fail(`retained log exceeds ${MAX_LOG_BYTES} bytes`);
    }
    const bytes = Buffer.allocUnsafe(size);
    let offset = 0;
    while (offset < size) {
      const read = readSync(descriptor, bytes, offset, size - offset, offset);
      if (read === 0) fail("retained log changed while being read");
      offset += read;
    }
    const final = fstatSync(descriptor, { bigint: true });
    if (
      final.dev !== initial.dev
      || final.ino !== initial.ino
      || final.size !== initial.size
      || final.mtimeNs !== initial.mtimeNs
      || final.ctimeNs !== initial.ctimeNs
    ) {
      fail("retained log changed while being read");
    }
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

function requireExactKeys(value, expectedKeys, name) {
  const actualKeys = Object.keys(value);
  const missing = expectedKeys.filter((key) => !Object.hasOwn(value, key));
  const unknown = actualKeys.filter((key) => !expectedKeys.includes(key));
  if (missing.length > 0 || unknown.length > 0) {
    fail(`${name} must have exact keys; missing=[${missing.join(",")}], unknown=[${unknown.join(",")}]`);
  }
}

function requireAvailableEnvironment(value) {
  const environment = requireRecord(value, "environment");
  requireExactKeys(environment, ENVIRONMENT_KEYS, "environment");
  requireEqual(environment.schema, "browsergrad.execution-environment@1", "environment.schema");
  requireEqual(
    environment.acquisition,
    "navigator.gpu.requestAdapter/requestDevice",
    "environment.acquisition",
  );
  requireString(environment.userAgent, "environment.userAgent");
  requireString(environment.platform, "environment.platform");

  const adapter = requireRecord(environment.adapter, "environment.adapter");
  requireExactKeys(adapter, ADAPTER_KEYS, "environment.adapter");
  for (const key of ADAPTER_KEYS) requireString(adapter[key], `environment.adapter.${key}`);

  const adapterFeatures = requireSortedUniqueStringArray(
    environment.adapterSupportedFeatures,
    "environment.adapterSupportedFeatures",
  );
  const negotiatedFeatures = requireSortedUniqueStringArray(
    environment.negotiatedDeviceFeatures,
    "environment.negotiatedDeviceFeatures",
  );
  if (negotiatedFeatures.some((feature) => !adapterFeatures.includes(feature))) {
    fail("environment.negotiatedDeviceFeatures must be a subset of adapterSupportedFeatures");
  }

  const limits = requireRecord(
    environment.negotiatedDeviceLimits,
    "environment.negotiatedDeviceLimits",
  );
  requireExactKeys(limits, DEVICE_LIMIT_KEYS, "environment.negotiatedDeviceLimits");
  for (const key of DEVICE_LIMIT_KEYS) {
    if (!Number.isSafeInteger(limits[key]) || limits[key] <= 0) {
      fail(`environment.negotiatedDeviceLimits.${key} must be a positive safe integer`);
    }
  }
  return environment;
}

function requireStringArray(value, name, requireEmpty = false) {
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== "string" || entry.length === 0)) {
    fail(`${name} must be an array of nonempty strings`);
  }
  if (requireEmpty && value.length !== 0) fail(`${name} must be empty for passed evidence`);
  return value;
}

function requireSortedUniqueStringArray(value, name) {
  const array = requireStringArray(value, name);
  const sortedUnique = [...new Set(array)].sort();
  if (!equalArray(array, sortedUnique)) {
    fail(`${name} must be sorted and contain no duplicates`);
  }
  return array;
}

function requireString(value, name) {
  if (typeof value !== "string") fail(`${name} must be a string`);
}

function requireIsoTimestamp(value, name) {
  if (typeof value !== "string") fail(`${name} must be a canonical ISO timestamp`);
  const milliseconds = Date.parse(value);
  if (!Number.isFinite(milliseconds) || new Date(milliseconds).toISOString() !== value) {
    fail(`${name} must be a canonical ISO timestamp`);
  }
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
  throw new Error(`compiler view-copy evidence log rejected: ${message}`);
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  await main();
}
