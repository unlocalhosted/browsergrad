import { spawnSync } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { createReadOnlyNpmEnvironment } from "./npm-read-only-environment.mjs";
import {
  assertMinimumNpmVersion,
  MINIMUM_ATTESTATION_AUDIT_NPM_VERSION,
} from "./require-npm-version.mjs";

export const NPM_REGISTRY_URL = "https://registry.npmjs.org/";
export const INTOTO_STATEMENT_V1 = "https://in-toto.io/Statement/v1";
export const SLSA_PROVENANCE_V1 = "https://slsa.dev/provenance/v1";
export const INTOTO_DSSE_PAYLOAD_TYPE = "application/vnd.in-toto+json";
export const GITHUB_ACTIONS_BUILD_TYPE =
  "https://slsa-framework.github.io/github-actions-buildtypes/workflow/v1";
export const GITHUB_HOSTED_BUILDER_ID =
  "https://github.com/actions/runner/github-hosted";

const MAX_REGISTRY_RESPONSE_BYTES = 8 * 1024 * 1024;
const DEFAULT_FETCH_ATTEMPTS = 3;
const DEFAULT_FETCH_TIMEOUT_MS = 10_000;
const DEFAULT_COMMAND_TIMEOUT_MS = 120_000;
const DEFAULT_RETRY_DELAY_MS = 200;
const MAX_COMMAND_OUTPUT_BYTES = 16 * 1024 * 1024;

/**
 * Validate an already decoded SLSA v1 statement without performing I/O.
 *
 * The expected values are deliberately explicit. The validator never reads
 * process.env, git state, package manifests, or the network, so callers cannot
 * accidentally prove a statement against facts supplied by that statement.
 */
export function validateSlsaV1GithubActionsStatement(statement, expected) {
  const identity = validateExpectedIdentity(expected);
  return validateSlsaV1GithubActionsStatementWithPolicy(
    statement,
    identity,
    {
      validateWorkflow({ workflowPath, ref }) {
        requireEqual(workflowPath, identity.workflowPath, "statement workflow path");
        requireEqual(ref, identity.ref, "statement workflow ref");
      },
      validateCommit(commit) {
        requireEqual(commit, identity.commit, "statement source gitCommit");
      },
    },
  );
}

function validateDependencySlsaV1GithubActionsStatement(statement, expected) {
  const identity = validateExpectedDependencyIdentity(expected);
  return validateSlsaV1GithubActionsStatementWithPolicy(
    statement,
    identity,
    {
      validateWorkflow({ workflowPath, ref }) {
        if (!identity.allowedWorkflowPaths.includes(workflowPath)) {
          fail(
            `statement workflow path must be in allowedWorkflowPaths, got ${workflowPath}`,
          );
        }
        if (
          workflowPath === ".github/workflows/publish-npm.yml"
          && ref !== "refs/heads/main"
        ) {
          fail("publish-npm.yml provenance must use refs/heads/main");
        }
        if (
          workflowPath === ".github/workflows/release.yml"
          && !/^refs\/tags\/(?:jit|runtime|primitives|kernels|grad|compiler|semantic-core)-v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/u.test(ref)
        ) {
          fail("release.yml provenance must use an approved BrowserGrad release tag");
        }
      },
    },
  );
}

function validateSlsaV1GithubActionsStatementWithPolicy(
  statement,
  identity,
  policy,
) {
  const root = requireRecord(statement, "provenance statement");

  requireEqual(root._type, INTOTO_STATEMENT_V1, "statement._type");
  requireEqual(root.predicateType, SLSA_PROVENANCE_V1, "statement.predicateType");

  const subjects = requireArray(root.subject, "statement.subject");
  if (subjects.length !== 1) {
    fail(`statement.subject must contain exactly one npm package, got ${subjects.length}`);
  }
  const subject = requireRecord(subjects[0], "statement.subject[0]");
  requireEqual(subject.name, identity.purl, "statement.subject[0].name");
  const digest = requireRecord(subject.digest, "statement.subject[0].digest");
  requireEqual(
    digest.sha512,
    identity.sha512Hex,
    "statement.subject[0].digest.sha512",
  );

  const predicate = requireRecord(root.predicate, "statement.predicate");
  const buildDefinition = requireRecord(
    predicate.buildDefinition,
    "statement.predicate.buildDefinition",
  );
  requireEqual(
    buildDefinition.buildType,
    GITHUB_ACTIONS_BUILD_TYPE,
    "statement.predicate.buildDefinition.buildType",
  );

  const externalParameters = requireRecord(
    buildDefinition.externalParameters,
    "statement.predicate.buildDefinition.externalParameters",
  );
  const workflow = requireRecord(
    externalParameters.workflow,
    "statement.predicate.buildDefinition.externalParameters.workflow",
  );
  requireEqual(
    workflow.repository,
    identity.repository,
    "statement workflow repository",
  );
  const workflowPath = validateWorkflowPath(workflow.path, "statement workflow path");
  const ref = validateGithubRef(workflow.ref, "statement workflow ref");
  policy.validateWorkflow({ workflowPath, ref });

  const dependencies = requireArray(
    buildDefinition.resolvedDependencies,
    "statement.predicate.buildDefinition.resolvedDependencies",
  );
  if (dependencies.length !== 1) {
    fail(`statement resolvedDependencies must contain exactly one source, got ${dependencies.length}`);
  }
  const source = requireRecord(
    dependencies[0],
    "statement.predicate.buildDefinition.resolvedDependencies[0]",
  );
  requireEqual(
    source.uri,
    `git+${identity.repository}@${ref}`,
    "statement source uri",
  );
  const sourceDigest = requireRecord(source.digest, "statement source digest");
  const commit = validateGitCommit(
    sourceDigest.gitCommit,
    "statement source gitCommit",
  );

  policy.validateCommit?.(commit);

  const runDetails = requireRecord(predicate.runDetails, "statement.predicate.runDetails");
  const builder = requireRecord(runDetails.builder, "statement.predicate.runDetails.builder");
  requireEqual(builder.id, GITHUB_HOSTED_BUILDER_ID, "statement builder id");
  const metadata = requireRecord(runDetails.metadata, "statement.predicate.runDetails.metadata");
  const invocationId = requireString(
    metadata.invocationId,
    "statement.predicate.runDetails.metadata.invocationId",
  );
  validateGithubInvocationId(invocationId, identity.repository);

  return Object.freeze({
    spec: identity.spec,
    name: identity.name,
    version: identity.version,
    integrity: identity.expectedIntegrity,
    purl: identity.purl,
    sha512: identity.sha512Hex,
    repository: identity.repository,
    workflowPath,
    ref,
    commit,
    builderId: GITHUB_HOSTED_BUILDER_ID,
    invocationId,
  });
}

/** Decode exactly one SLSA v1 DSSE payload from an npm attestation response. */
export function decodeSingleSlsaV1DsseStatement(attestationDocument) {
  const document = requireRecord(attestationDocument, "npm attestation document");
  const attestations = requireArray(document.attestations, "npm attestation document.attestations");
  const provenanceAttestations = attestations.filter((candidate) =>
    isRecord(candidate) && candidate.predicateType === SLSA_PROVENANCE_V1);
  if (provenanceAttestations.length !== 1) {
    fail(
      `npm attestation document must contain exactly one ${SLSA_PROVENANCE_V1} attestation, got ${provenanceAttestations.length}`,
    );
  }

  const attestation = provenanceAttestations[0];
  const bundle = requireRecord(attestation.bundle, "SLSA attestation.bundle");
  const envelope = requireRecord(bundle.dsseEnvelope, "SLSA attestation.bundle.dsseEnvelope");
  requireEqual(envelope.payloadType, INTOTO_DSSE_PAYLOAD_TYPE, "SLSA DSSE payloadType");
  const signatures = requireArray(envelope.signatures, "SLSA DSSE signatures");
  if (signatures.length === 0) {
    fail("SLSA DSSE envelope must carry at least one signature");
  }
  for (const [index, signature] of signatures.entries()) {
    const entry = requireRecord(signature, `SLSA DSSE signatures[${index}]`);
    requireString(entry.sig, `SLSA DSSE signatures[${index}].sig`);
  }

  const payload = decodeCanonicalBase64(
    requireString(envelope.payload, "SLSA DSSE payload"),
    "SLSA DSSE payload",
  );
  let statement;
  try {
    statement = JSON.parse(payload.toString("utf8"));
  } catch (error) {
    throw new Error("SLSA DSSE payload is not valid JSON", { cause: error });
  }
  const root = requireRecord(statement, "decoded SLSA DSSE statement");
  requireEqual(root.predicateType, attestation.predicateType, "DSSE/attestation predicateType");
  return root;
}

/**
 * Verify registry metadata, SLSA provenance identity, and npm/Sigstore proofs
 * for one immutable package version.
 */
export async function verifyPublishedNpmProvenance(spec, expected, runtime = {}) {
  const identity = validateExpectedIdentity({ ...expected, spec });
  return verifyPublishedNpmProvenanceWithPolicy(
    identity,
    runtime,
    (statement) => validateSlsaV1GithubActionsStatement(statement, identity),
  );
}

/**
 * Verify immutable dependency provenance while trusting only approved release
 * workflows. Unlike the strict release verifier, ref and commit are outputs
 * attested by npm/Sigstore rather than inputs from the current checkout.
 */
export async function verifyPublishedNpmDependencyProvenance(
  spec,
  expected,
  runtime = {},
) {
  const identity = validateExpectedDependencyIdentity({ ...expected, spec });
  return verifyPublishedNpmProvenanceWithPolicy(
    identity,
    runtime,
    (statement) => validateDependencySlsaV1GithubActionsStatement(statement, identity),
  );
}

async function verifyPublishedNpmProvenanceWithPolicy(
  identity,
  runtime,
  validateStatement,
) {
  const boundary = validateRuntime(runtime);
  const metadataUrl = new URL(
    `${encodeURIComponent(identity.name)}/${encodeURIComponent(identity.version)}`,
    NPM_REGISTRY_URL,
  );
  const metadata = requireRecord(
    await fetchJsonWithRetry(metadataUrl, "npm package metadata", boundary),
    "npm package metadata",
  );
  requireEqual(metadata.name, identity.name, "registry package name");
  requireEqual(metadata.version, identity.version, "registry package version");
  const dist = requireRecord(metadata.dist, "registry package dist");
  requireEqual(dist.integrity, identity.expectedIntegrity, "registry dist.integrity");

  const registrySignatures = requireArray(dist.signatures, "registry dist.signatures");
  if (registrySignatures.length === 0) {
    fail("registry dist.signatures must contain at least one registry signature");
  }
  for (const [index, signature] of registrySignatures.entries()) {
    const entry = requireRecord(signature, `registry dist.signatures[${index}]`);
    requireString(entry.keyid, `registry dist.signatures[${index}].keyid`);
    requireString(entry.sig, `registry dist.signatures[${index}].sig`);
  }

  const descriptor = requireRecord(dist.attestations, "registry dist.attestations");
  const provenanceDescriptor = requireRecord(
    descriptor.provenance,
    "registry dist.attestations.provenance",
  );
  requireEqual(
    provenanceDescriptor.predicateType,
    SLSA_PROVENANCE_V1,
    "registry provenance predicateType",
  );
  const attestationUrl = validateRegistryAttestationUrl(
    descriptor.url,
    identity.name,
    identity.version,
  );
  const temp = await mkdtemp(join(tmpdir(), "browsergrad-npm-provenance-"));
  let validated;
  try {
    const cache = join(temp, "npm-cache");
    await writeFile(
      join(temp, "package.json"),
      `${JSON.stringify({
        name: "browsergrad-npm-provenance-verification",
        version: "0.0.0",
        private: true,
      }, null, 2)}\n`,
      { encoding: "utf8", mode: 0o600 },
    );
    const readOnlyUserConfig = join(temp, ".npmrc");
    await writeFile(
      readOnlyUserConfig,
      `registry=${NPM_REGISTRY_URL}\n`,
      { encoding: "utf8", mode: 0o600 },
    );
    const npmVersion = await runNpm(["--version"], temp, boundary);
    assertMinimumNpmVersion(
      npmVersion.stdout,
      MINIMUM_ATTESTATION_AUDIT_NPM_VERSION,
    );
    await runNpm([
      "install",
      "--ignore-scripts",
      "--omit=optional",
      "--omit=peer",
      "--save-exact",
      "--package-lock=true",
      "--audit=false",
      "--fund=false",
      "--prefer-online",
      `--registry=${NPM_REGISTRY_URL}`,
      `--cache=${cache}`,
      identity.spec,
    ], temp, boundary);
    const audit = await runNpm([
      "audit",
      "signatures",
      "--json",
      "--include-attestations",
      "--prefer-online",
      `--registry=${NPM_REGISTRY_URL}`,
      `--cache=${cache}`,
    ], temp, boundary);
    const verifiedAttestations = validateSignatureAuditOutput(
      audit.stdout,
      identity,
      attestationUrl,
    );
    const statement = decodeSingleSlsaV1DsseStatement(verifiedAttestations);
    validated = validateStatement(statement);
  } finally {
    await rm(temp, { recursive: true, force: true });
  }

  return Object.freeze({
    ...validated,
    attestationUrl: attestationUrl.href,
    registrySignatures: registrySignatures.length,
    provenanceAttestations: 1,
    cryptographicallyVerified: true,
  });
}

function validateExpectedIdentity(expected) {
  const identity = validateExpectedPackageIdentity(expected);
  const input = requireRecord(expected, "expected provenance identity");
  return Object.freeze({
    ...identity,
    workflowPath: validateWorkflowPath(input.workflowPath, "workflowPath"),
    ref: validateGithubRef(input.ref, "ref"),
    commit: validateGitCommit(input.commit, "commit"),
  });
}

function validateExpectedDependencyIdentity(expected) {
  const identity = validateExpectedPackageIdentity(expected);
  const input = requireRecord(expected, "expected dependency provenance identity");
  const values = requireArray(input.allowedWorkflowPaths, "allowedWorkflowPaths");
  if (values.length === 0) {
    fail("allowedWorkflowPaths must contain at least one approved workflow");
  }
  const allowedWorkflowPaths = values.map((value, index) =>
    validateWorkflowPath(value, `allowedWorkflowPaths[${index}]`));
  if (new Set(allowedWorkflowPaths).size !== allowedWorkflowPaths.length) {
    fail("allowedWorkflowPaths must not contain duplicates");
  }
  return Object.freeze({
    ...identity,
    allowedWorkflowPaths: Object.freeze(allowedWorkflowPaths),
  });
}

function validateExpectedPackageIdentity(expected) {
  const input = requireRecord(expected, "expected provenance identity");
  const { name, version, spec } = parseExactNpmSpec(
    requireString(input.spec, "expected spec"),
  );
  const expectedIntegrity = requireString(input.expectedIntegrity, "expectedIntegrity");
  const sha512Hex = sha512HexFromIntegrity(expectedIntegrity);
  const repository = validateGithubRepository(input.repository);
  return Object.freeze({
    spec,
    name,
    version,
    expectedIntegrity,
    sha512Hex,
    purl: npmPurl(name, version),
    repository,
  });
}

function validateWorkflowPath(value, label) {
  const workflowPath = requireString(value, label);
  if (
    !/^\.github\/workflows\/[A-Za-z0-9._/-]+\.ya?ml$/u.test(workflowPath)
    || workflowPath.includes("//")
    || workflowPath.split("/").includes("..")
  ) {
    fail(`${label} must identify a repository workflow YAML file, got ${workflowPath}`);
  }
  return workflowPath;
}

function validateGithubRef(value, label) {
  const ref = requireString(value, label);
  if (!/^refs\/(?:heads|tags)\/[A-Za-z0-9._/-]+$/u.test(ref) || ref.includes("..")) {
    fail(`${label} must be an exact GitHub heads/tags ref, got ${ref}`);
  }
  return ref;
}

function validateGitCommit(value, label) {
  const commit = requireString(value, label);
  if (!/^[0-9a-f]{40}$/u.test(commit)) {
    fail(`${label} must be an exact lowercase 40-character git SHA, got ${commit}`);
  }
  return commit;
}

function parseExactNpmSpec(spec) {
  const separator = spec.lastIndexOf("@");
  if (separator <= 0) {
    fail(`Expected an exact npm name@version spec, got ${spec}`);
  }
  const name = spec.slice(0, separator);
  const version = spec.slice(separator + 1);
  if (!/^(?:@[a-z0-9][a-z0-9._-]*\/)?[a-z0-9][a-z0-9._-]*$/u.test(name)) {
    fail(`Invalid npm package name in exact spec: ${name}`);
  }
  if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/u.test(version)) {
    fail(`Invalid exact npm package version: ${version}`);
  }
  return Object.freeze({ name, version, spec: `${name}@${version}` });
}

function sha512HexFromIntegrity(integrity) {
  const match = /^sha512-([A-Za-z0-9+/]+={0,2})$/u.exec(integrity);
  if (match === null) {
    fail("expectedIntegrity must be one canonical sha512 SRI value");
  }
  const digest = Buffer.from(match[1], "base64");
  if (digest.length !== 64 || digest.toString("base64") !== match[1]) {
    fail("expectedIntegrity must contain a canonical 64-byte sha512 digest");
  }
  return digest.toString("hex");
}

function npmPurl(name, version) {
  let encodedName;
  if (name.startsWith("@")) {
    const separator = name.indexOf("/");
    encodedName = `%40${encodeURIComponent(name.slice(1, separator))}/${encodeURIComponent(name.slice(separator + 1))}`;
  } else {
    encodedName = encodeURIComponent(name);
  }
  return `pkg:npm/${encodedName}@${encodeURIComponent(version)}`;
}

function validateGithubRepository(value) {
  const repository = requireString(value, "repository");
  let url;
  try {
    url = new URL(repository);
  } catch (error) {
    throw new Error(`repository must be an absolute GitHub URL, got ${repository}`, { cause: error });
  }
  const parts = url.pathname.split("/").filter(Boolean);
  if (
    url.protocol !== "https:"
    || url.hostname !== "github.com"
    || url.port !== ""
    || url.username !== ""
    || url.password !== ""
    || url.search !== ""
    || url.hash !== ""
    || parts.length !== 2
    || parts[1].endsWith(".git")
    || repository !== `https://github.com/${parts[0]}/${parts[1]}`
  ) {
    fail(`repository must be canonical https://github.com/<owner>/<repo>, got ${repository}`);
  }
  return repository;
}

function validateGithubInvocationId(invocationId, repository) {
  const escapedRepository = repository.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
  const pattern = new RegExp(
    `^${escapedRepository}/actions/runs/[1-9][0-9]*/attempts/[1-9][0-9]*$`,
    "u",
  );
  if (!pattern.test(invocationId)) {
    fail(
      `statement invocationId must identify a GitHub Actions run attempt under ${repository}, got ${invocationId}`,
    );
  }
}

function validateRegistryAttestationUrl(value, name, version) {
  const raw = requireString(value, "registry dist.attestations.url");
  let url;
  try {
    url = new URL(raw);
  } catch (error) {
    throw new Error(`registry attestation URL is invalid: ${raw}`, { cause: error });
  }
  if (
    url.origin !== new URL(NPM_REGISTRY_URL).origin
    || url.username !== ""
    || url.password !== ""
    || url.search !== ""
    || url.hash !== ""
  ) {
    fail(`registry attestation URL must stay on ${NPM_REGISTRY_URL}, got ${raw}`);
  }
  let decodedPath;
  try {
    decodedPath = decodeURIComponent(url.pathname);
  } catch (error) {
    throw new Error(`registry attestation URL path is not valid encoding: ${raw}`, { cause: error });
  }
  const expectedPath = `/-/npm/v1/attestations/${name}@${version}`;
  if (decodedPath !== expectedPath) {
    fail(`registry attestation URL must identify ${name}@${version}, got ${raw}`);
  }
  return url;
}

function validateRuntime(runtime) {
  const input = requireRecord(runtime, "provenance verifier runtime");
  const fetchImpl = input.fetchImpl ?? globalThis.fetch;
  if (typeof fetchImpl !== "function") {
    fail("provenance verifier requires a fetch implementation");
  }
  const runCommand = input.runCommand ?? defaultRunCommand;
  if (typeof runCommand !== "function") {
    fail("provenance verifier runCommand must be a function");
  }
  const sleep = input.sleep ?? ((delayMs) => new Promise((resolve) => setTimeout(resolve, delayMs)));
  if (typeof sleep !== "function") {
    fail("provenance verifier sleep must be a function");
  }
  return Object.freeze({
    fetchImpl,
    runCommand,
    sleep,
    fetchAttempts: boundedInteger(
      input.fetchAttempts ?? DEFAULT_FETCH_ATTEMPTS,
      "fetchAttempts",
      1,
      5,
    ),
    fetchTimeoutMs: boundedInteger(
      input.fetchTimeoutMs ?? DEFAULT_FETCH_TIMEOUT_MS,
      "fetchTimeoutMs",
      100,
      60_000,
    ),
    commandTimeoutMs: boundedInteger(
      input.commandTimeoutMs ?? DEFAULT_COMMAND_TIMEOUT_MS,
      "commandTimeoutMs",
      1_000,
      300_000,
    ),
    retryDelayMs: boundedInteger(
      input.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS,
      "retryDelayMs",
      0,
      5_000,
    ),
  });
}

async function fetchJsonWithRetry(url, label, runtime) {
  let lastError;
  for (let attempt = 1; attempt <= runtime.fetchAttempts; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), runtime.fetchTimeoutMs);
    try {
      const response = await runtime.fetchImpl(url, {
        method: "GET",
        redirect: "error",
        signal: controller.signal,
        headers: {
          accept: "application/json",
          "user-agent": "browsergrad-npm-provenance-verifier/1",
        },
      });
      if (!isResponseLike(response)) {
        fail(`${label} fetch returned a non-Response value`);
      }
      if (!response.ok) {
        const detail = await readBoundedResponseText(response);
        const error = new Error(
          `${label} request failed with HTTP ${response.status}${detail === "" ? "" : `: ${detail}`}`,
        );
        error.retryable = isRetryableStatus(response.status);
        if (!isRetryableStatus(response.status) || attempt === runtime.fetchAttempts) {
          throw error;
        }
        lastError = error;
      } else {
        const body = await readBoundedResponseText(response);
        try {
          return JSON.parse(body);
        } catch (error) {
          throw new Error(`${label} response is not valid JSON`, { cause: error });
        }
      }
    } catch (error) {
      lastError = error;
      if (attempt === runtime.fetchAttempts || !isRetryableFetchError(error)) {
        throw new Error(`${label} request failed after ${attempt} attempt(s)`, { cause: error });
      }
    } finally {
      clearTimeout(timeout);
    }
    const delay = Math.min(runtime.retryDelayMs * (2 ** (attempt - 1)), 5_000);
    await runtime.sleep(delay);
  }
  throw new Error(`${label} request exhausted retries`, { cause: lastError });
}

async function readBoundedResponseText(response) {
  const contentLength = response.headers?.get?.("content-length");
  if (contentLength !== null && contentLength !== undefined) {
    const length = Number(contentLength);
    if (!Number.isSafeInteger(length) || length < 0 || length > MAX_REGISTRY_RESPONSE_BYTES) {
      fail(`registry response Content-Length is invalid or exceeds ${MAX_REGISTRY_RESPONSE_BYTES} bytes`);
    }
  }
  if (response.body === null || response.body === undefined) {
    return "";
  }
  const reader = response.body.getReader();
  const chunks = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      total += value.byteLength;
      if (total > MAX_REGISTRY_RESPONSE_BYTES) {
        fail(`registry response exceeds ${MAX_REGISTRY_RESPONSE_BYTES} bytes`);
      }
      chunks.push(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
  }
  return Buffer.concat(chunks, total).toString("utf8");
}

function isResponseLike(value) {
  return isRecord(value)
    && typeof value.ok === "boolean"
    && Number.isInteger(value.status)
    && value.body !== undefined;
}

function isRetryableStatus(status) {
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

function isRetryableFetchError(error) {
  if (error instanceof Error && Object.hasOwn(error, "retryable")) {
    return error.retryable === true;
  }
  return error instanceof TypeError || error?.name === "AbortError";
}

async function runNpm(args, cwd, runtime) {
  return runtime.runCommand("npm", args, {
    cwd,
    encoding: "utf8",
    env: createReadOnlyNpmEnvironment({ userConfig: join(cwd, ".npmrc") }),
    shell: false,
    timeoutMs: runtime.commandTimeoutMs,
    maxBufferBytes: MAX_COMMAND_OUTPUT_BYTES,
  });
}

function defaultRunCommand(command, args, options) {
  const result = spawnSync(command, args, {
    cwd: options.cwd,
    encoding: options.encoding,
    env: options.env,
    shell: false,
    stdio: "pipe",
    timeout: options.timeoutMs,
    maxBuffer: options.maxBufferBytes,
  });
  if (result.error !== undefined) {
    throw new Error(`${command} ${args.slice(0, 2).join(" ")} failed to execute`, {
      cause: result.error,
    });
  }
  if (result.status !== 0) {
    throw new Error(
      `${command} ${args.slice(0, 2).join(" ")} failed with exit ${result.status}\n${result.stdout ?? ""}\n${result.stderr ?? ""}`,
    );
  }
  return Object.freeze({ stdout: result.stdout ?? "", stderr: result.stderr ?? "" });
}

function validateSignatureAuditOutput(stdout, identity, attestationUrl) {
  let audit;
  try {
    audit = JSON.parse(stdout);
  } catch (error) {
    throw new Error("npm audit signatures did not emit valid JSON", { cause: error });
  }
  const root = requireRecord(audit, "npm audit signatures output");
  const invalid = requireArray(root.invalid, "npm audit signatures invalid");
  const missing = requireArray(root.missing, "npm audit signatures missing");
  if (invalid.length !== 0 || missing.length !== 0) {
    fail(
      `npm audit signatures reported ${invalid.length} invalid and ${missing.length} missing proofs`,
    );
  }
  const verified = requireArray(root.verified, "npm audit signatures verified");
  const expectedLocation = `node_modules/${identity.name}`;
  const targetEntries = verified.filter((candidate) => {
    if (!isRecord(candidate)) return false;
    return candidate.name === identity.name
      && candidate.version === identity.version
      && candidate.location === expectedLocation;
  });
  if (targetEntries.length !== 1) {
    fail(`npm audit signatures must return exactly one verified root entry for ${identity.spec}`);
  }
  const target = targetEntries[0];
  requireEqual(target.registry, NPM_REGISTRY_URL, "verified npm registry");
  const attestations = requireRecord(target.attestations, "verified npm attestations");
  const verifiedAttestationUrl = validateRegistryAttestationUrl(
    attestations.url,
    identity.name,
    identity.version,
  );
  requireEqual(
    verifiedAttestationUrl.href,
    attestationUrl.href,
    "verified npm attestation URL",
  );
  const provenance = requireRecord(
    attestations.provenance,
    "verified npm provenance descriptor",
  );
  requireEqual(
    provenance.predicateType,
    SLSA_PROVENANCE_V1,
    "verified npm provenance predicateType",
  );
  const attestationBundles = requireArray(
    target.attestationBundles,
    "verified npm attestation bundles",
  );
  return Object.freeze({ attestations: attestationBundles });
}

function decodeCanonicalBase64(value, label) {
  if (!/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/u.test(value)) {
    fail(`${label} must be canonical base64`);
  }
  const decoded = Buffer.from(value, "base64");
  if (decoded.length === 0 || decoded.toString("base64") !== value) {
    fail(`${label} must be non-empty canonical base64`);
  }
  return decoded;
}

function boundedInteger(value, label, minimum, maximum) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    fail(`${label} must be an integer between ${minimum} and ${maximum}`);
  }
  return value;
}

function requireRecord(value, label) {
  if (!isRecord(value)) {
    fail(`${label} must be an object`);
  }
  return value;
}

function requireArray(value, label) {
  if (!Array.isArray(value)) {
    fail(`${label} must be an array`);
  }
  return value;
}

function requireString(value, label) {
  if (typeof value !== "string" || value.length === 0) {
    fail(`${label} must be a non-empty string`);
  }
  return value;
}

function requireEqual(actual, expected, label) {
  if (actual !== expected) {
    fail(`${label} must equal ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function fail(message) {
  throw new Error(message);
}
