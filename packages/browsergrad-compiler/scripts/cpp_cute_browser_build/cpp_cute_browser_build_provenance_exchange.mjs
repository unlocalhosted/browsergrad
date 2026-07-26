import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, realpath, unlink } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, normalize } from "node:path/posix";
import { pathToFileURL } from "node:url";

import {
  canonicalJsonBytes,
  decodeWireJson,
  deepFreezeJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT,
  canonicalCppCuteBrowserAssetManifestBytes,
  decodeCppCuteBrowserAssetManifest,
} from "../../dist/cpp_cute_browser_assets.js";
import {
  CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_BYTE_LIMIT,
  canonicalCppCuteBrowserBuildInputLockBytes,
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  createCppCuteBrowserBuildProvenanceSigningRequest,
  verifyCppCuteBrowserBuildSignatureBinding,
} from "../../dist/cpp_cute_browser_build_provenance.js";
import {
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT,
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS,
} from "../../dist/cpp_cute_browser_build_provenance_syntax.js";
import {
  verifyCppCuteBrowserBuildProducer,
} from "../../dist/cpp_cute_browser_producer_trust.js";
import {
  CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT,
  admitCppCuteBrowserProducerTrustPolicy,
  copyAdmittedCppCuteBrowserProducerTrustPolicyBytes,
} from "../../dist/cpp_cute_browser_producer_trust_policy.js";
import {
  prepareCppCuteAttestationTrustStore,
} from "../../dist/cpp_cute_frontend_provenance.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteBrowserFrontendProfile,
} from "../../dist/cpp_cute_frontend_profile.js";
import {
  copyVerifiedCppCuteBrowserWorkerBundleBytes,
  verifyCppCuteBrowserWorkerBundle,
} from "../../dist/cpp_cute_browser_worker_bundle.js";

export const CPP_CUTE_BROWSER_BUILD_PROVENANCE_SIGNING_REQUEST_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-build-provenance-signing-request";
export const CPP_CUTE_BROWSER_BUILD_PRODUCER_OBSERVATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-build-producer-verification-observation";

const ERROR_PREFIX = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-EXCHANGE";
const SIGNING_REQUEST_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-build-provenance-signing-request.v1";
const PRODUCER_OBSERVATION_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-build-producer-verification-observation.v1";
const PROFILE_BYTE_LIMIT = 256 * 1024;
const TRUST_STORE_BYTE_LIMIT = 256 * 1024;
const MAX_ARGUMENT_COUNT = 16;
const MAX_ARGUMENT_BYTE_LENGTH = 16 * 1024;
const MAX_OUTPUT_BYTE_LENGTH = 1024 * 1024;
const SHA256 = /^[0-9a-f]{64}$/u;
const KEY_ID = /^sha256:[0-9a-f]{64}$/u;
const REQUEST_ID =
  /^bg\.cpp\.browser-build-provenance-signing-request\.sha256\.[0-9a-f]{64}$/u;
const PROFILE_DECODE_LIMITS = Object.freeze({
  maxDocumentBytes: PROFILE_BYTE_LIMIT,
  maxDepth: 32,
  maxNodes: 32_768,
  maxStringBytes: 192 * 1024,
  maxArrayLength: 8_192,
  maxObjectProperties: 1_024,
  maxRank: 8,
  maxIntegerBits: 64,
  maxArithmeticOperations: 131_072,
});
const TRUST_STORE_DECODE_LIMITS = Object.freeze({
  maxDocumentBytes: TRUST_STORE_BYTE_LIMIT,
  maxDepth: 8,
  maxNodes: 2_048,
  maxStringBytes: 192 * 1024,
  maxArrayLength: 256,
  maxObjectProperties: 16,
  maxRank: 1,
  maxIntegerBits: 32,
  maxArithmeticOperations: 4_096,
});
const ARGUMENTS = Object.freeze({
  "signing-request": Object.freeze([
    "asset-manifest",
    "build-input-lock",
    "builder-id",
    "key-id",
    "operation",
    "output",
    "producer-policy",
    "profile",
    "trust-store",
    "worker-module",
  ]),
  "verify-envelope": Object.freeze([
    "asset-manifest",
    "build-input-lock",
    "envelope",
    "operation",
    "output",
    "producer-policy",
    "profile",
    "signing-request",
    "trust-store",
    "worker-module",
  ]),
});
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export class CppCuteBrowserBuildProvenanceExchangeError extends Error {
  constructor(code, path, message, options) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserBuildProvenanceExchangeError";
    this.code = code;
    this.path = path;
  }
}

/**
 * Executes one complete host exchange operation through a deliberately narrow
 * interface: closed CLI arguments in, one immutable canonical artifact out.
 * Private keys are never accepted. Serialized verification observations are
 * never reusable producer authority.
 */
export async function runCppCuteBrowserBuildProvenanceExchange(
  argv,
  options = {},
) {
  const signal = normalizeOptions(options);
  const args = parseArguments(snapshotArguments(argv));
  throwIfAborted(signal);
  const paths = exchangePaths(args);
  assertDistinctPaths(paths);
  const packageWorker = await verifyCppCuteBrowserWorkerBundle();
  const packageWorkerBytes =
    copyVerifiedCppCuteBrowserWorkerBundleBytes(packageWorker);
  const inputSpecifications = commonInputSpecifications(
    args,
    packageWorkerBytes.byteLength,
  );
  if (args.operation === "verify-envelope") {
    inputSpecifications.push(
      fileSpecification(
        "envelope",
        args["envelope"],
        CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT,
      ),
      fileSpecification(
        "signingRequest",
        args["signing-request"],
        MAX_OUTPUT_BYTE_LENGTH,
      ),
    );
  }
  const files = await readInputFiles(inputSpecifications, signal);
  assertDistinctInputFiles(files);
  throwIfAborted(signal);
  const authorities = await prepareAuthorities(
    files,
    packageWorker,
    packageWorkerBytes,
    signal,
  );
  const inputIdentities = commonInputIdentities(files);

  let record;
  if (args.operation === "signing-request") {
    record = await createSigningRequestRecord({
      authorities,
      builderId: args["builder-id"],
      inputIdentities,
      keyId: args["key-id"],
      signal,
    });
  } else {
    record = await createProducerObservationRecord({
      authorities,
      files,
      inputIdentities,
      signal,
    });
  }
  throwIfAborted(signal);
  const bytes = canonicalJsonBytes(record);
  if (bytes.byteLength === 0 || bytes.byteLength > MAX_OUTPUT_BYTE_LENGTH) {
    resource("$.output", "canonical exchange output exceeds its fixed byte limit");
  }
  const persisted = await persistCanonicalOutput(args.output, bytes, signal);
  return Object.freeze({
    operation: args.operation,
    outputPath: persisted.outputPath,
    outputSha256: persisted.sha256,
    outputByteLength: String(persisted.byteLength),
    record,
  });
}

async function createSigningRequestRecord(input) {
  const request = await createCppCuteBrowserBuildProvenanceSigningRequest(
    {
      assetManifest: input.authorities.assetManifest,
      buildInputLock: input.authorities.buildInputLock,
      workerBundle: input.authorities.workerBundle,
      trustPolicy: input.authorities.trustPolicy,
      trustStore: input.authorities.trustStore,
      builderId: nonemptyString(input.builderId, "$.arguments.builder-id"),
      keyId: pattern(input.keyId, KEY_ID, "$.arguments.key-id"),
    },
    input.signal === undefined ? {} : { signal: input.signal },
  );
  const body = {
    schema: CPP_CUTE_BROWSER_BUILD_PROVENANCE_SIGNING_REQUEST_SCHEMA,
    version: 1,
    authority: "format-only-external-signing-request",
    inputs: input.inputIdentities,
    policyId: request.policyId,
    policySha256: request.policySha256,
    builderId: request.builderId,
    keyId: request.keyId,
    statement: request.statement,
    payloadType: request.payloadType,
    payload: request.payload,
    signingBytesBase64: encodeBase64(request.signingBytes),
    claims: {
      signatureVerified: false,
      producerTrusted: false,
      exactAssetBytesVerified: false,
      fullDistributedOutputSetReproducible: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    },
  };
  const requestId = `bg.cpp.browser-build-provenance-signing-request.sha256.${
    hashProjection(SIGNING_REQUEST_DOMAIN, body)
  }`;
  return deepFreezeJson({
    ...body,
    requestId,
  });
}

async function createProducerObservationRecord(input) {
  const signingRequestFile = requiredFile(input.files, "signingRequest");
  const envelopeFile = requiredFile(input.files, "envelope");
  const signingRequestValue = decodeCanonicalJson(
    signingRequestFile.bytes,
    CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS,
    "$.inputs.signingRequest",
  );
  const requested = signingRequestCoordinates(signingRequestValue);
  const expectedRequest = await createSigningRequestRecord({
    authorities: input.authorities,
    builderId: requested.builderId,
    inputIdentities: input.inputIdentities,
    keyId: requested.keyId,
    signal: input.signal,
  });
  const expectedRequestBytes = canonicalJsonBytes(expectedRequest);
  if (!sameBytes(signingRequestFile.bytes, expectedRequestBytes)) {
    mismatch(
      "$.inputs.signingRequest",
      "signing request differs from the exact current package inputs and policy",
    );
  }
  const envelope = decodeCanonicalJson(
    envelopeFile.bytes,
    CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS,
    "$.inputs.envelope",
  );
  const envelopeCoordinates = dsseCoordinates(envelope);
  if (envelopeCoordinates.payloadType !== expectedRequest.payloadType ||
      envelopeCoordinates.payload !== expectedRequest.payload ||
      envelopeCoordinates.keyId !== expectedRequest.keyId) {
    mismatch(
      "$.inputs.envelope",
      "external envelope differs from the exact issued signing request",
    );
  }
  const signatureBinding = await verifyCppCuteBrowserBuildSignatureBinding(
    envelope,
    {
      assetManifest: input.authorities.assetManifest,
      buildInputLock: input.authorities.buildInputLock,
      workerBundle: input.authorities.workerBundle,
      trustStore: input.authorities.trustStore,
      ...(input.signal === undefined ? {} : { signal: input.signal }),
    },
  );
  const producer = await verifyCppCuteBrowserBuildProducer(
    signatureBinding,
    input.authorities.trustPolicy,
    input.signal === undefined ? {} : { signal: input.signal },
  );
  const inputs = deepFreezeJson({
    ...input.inputIdentities,
    signingRequest: fileIdentity(signingRequestFile),
    envelope: fileIdentity(envelopeFile),
  });
  const body = {
    schema: CPP_CUTE_BROWSER_BUILD_PRODUCER_OBSERVATION_SCHEMA,
    version: 1,
    authority: "host-verification-observation-only",
    signingRequestId: expectedRequest.requestId,
    inputs,
    producer: producerProjection(producer),
    observed: {
      signatureVerified: true,
      manifestSignaturePolicyMatched: true,
      independentTrustPolicyMatched: true,
      producerTrustedInThisProcess: true,
      buildSubjectBound: true,
    },
    claims: {
      reusableProducerAuthority: false,
      producerAuthoritySerialized: false,
      exactAssetBytesVerified: false,
      fullDistributedOutputSetReproducible: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    },
  };
  const observationId = `bg.cpp.browser-build-producer-observation.sha256.${
    hashProjection(PRODUCER_OBSERVATION_DOMAIN, body)
  }`;
  return deepFreezeJson({
    ...body,
    observationId,
  });
}

async function prepareAuthorities(files, workerBundle, packageWorkerBytes, signal) {
  const profileFile = requiredFile(files, "profile");
  const profileValue = decodeCanonicalJson(
    profileFile.bytes,
    PROFILE_DECODE_LIMITS,
    "$.inputs.profile",
  );
  const profile = await prepareCppCuteFrontendProfile(
    profileValue,
    signal === undefined ? {} : { signal },
  );
  unwrapPreparedCppCuteBrowserFrontendProfile(profile);
  throwIfAborted(signal);

  const assetManifestFile = requiredFile(files, "assetManifest");
  const assetManifest = await decodeCppCuteBrowserAssetManifest(
    assetManifestFile.bytes,
    profile,
    signal === undefined ? {} : { signal },
  );
  if (!sameBytes(
    assetManifestFile.bytes,
    canonicalCppCuteBrowserAssetManifestBytes(assetManifest),
  )) {
    mismatch("$.inputs.assetManifest", "asset manifest canonical identity changed");
  }

  const buildInputLockFile = requiredFile(files, "buildInputLock");
  const buildInputLock = await decodeCppCuteBrowserBuildInputLock(
    buildInputLockFile.bytes,
    signal === undefined ? {} : { signal },
  );
  if (!sameBytes(buildInputLockFile.bytes, cppCuteBrowserBuildInputLockResourceBytes()) ||
      !sameBytes(
        buildInputLockFile.bytes,
        canonicalCppCuteBrowserBuildInputLockBytes(buildInputLock),
      )) {
    mismatch(
      "$.inputs.buildInputLock",
      "build-input lock differs from the exact current package resource",
    );
  }

  const workerFile = requiredFile(files, "workerModule");
  if (!sameBytes(workerFile.bytes, packageWorkerBytes)) {
    mismatch(
      "$.inputs.workerModule",
      "Worker module differs from the exact current package bundle",
    );
  }

  const policyFile = requiredFile(files, "producerPolicy");
  const trustPolicy = await admitCppCuteBrowserProducerTrustPolicy(
    policyFile.bytes,
    signal === undefined ? {} : { signal },
  );
  if (!sameBytes(
    policyFile.bytes,
    copyAdmittedCppCuteBrowserProducerTrustPolicyBytes(trustPolicy),
  )) {
    mismatch("$.inputs.producerPolicy", "producer policy canonical identity changed");
  }

  const trustStoreFile = requiredFile(files, "trustStore");
  const trustStoreValue = decodeCanonicalJson(
    trustStoreFile.bytes,
    TRUST_STORE_DECODE_LIMITS,
    "$.inputs.trustStore",
  );
  const trustStore = await prepareCppCuteAttestationTrustStore(
    trustStoreValue,
    signal === undefined
      ? { limits: TRUST_STORE_DECODE_LIMITS }
      : { limits: TRUST_STORE_DECODE_LIMITS, signal },
  );
  throwIfAborted(signal);
  return Object.freeze({
    profile,
    assetManifest,
    buildInputLock,
    workerBundle,
    trustPolicy,
    trustStore,
  });
}

function producerProjection(producer) {
  return deepFreezeJson({
    producerEvidenceId: producer.producerEvidenceId,
    policyId: producer.policyId,
    policySha256: producer.policySha256,
    policyVersion: producer.policyVersion,
    buildSubjectId: producer.buildSubjectId,
    buildSubjectSha256: producer.buildSubjectSha256,
    statementSha256: producer.statementSha256,
    signatureEvidenceSha256: producer.signatureEvidenceSha256,
    predicateType: producer.predicateType,
    builderId: producer.builderId,
    keyId: producer.keyId,
    trustStoreSha256: producer.trustStoreSha256,
    profileHash: producer.profileHash,
    manifestId: producer.manifestId,
    assetSetSha256: producer.assetSetSha256,
    buildInputLockResourceSha256: producer.buildInputLockResourceSha256,
    workerBundleSha256: producer.workerBundleSha256,
  });
}

function commonInputSpecifications(args, workerByteLength) {
  return [
    fileSpecification("profile", args.profile, PROFILE_BYTE_LIMIT),
    fileSpecification(
      "assetManifest",
      args["asset-manifest"],
      CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT,
    ),
    fileSpecification(
      "buildInputLock",
      args["build-input-lock"],
      CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_BYTE_LIMIT,
    ),
    fileSpecification("workerModule", args["worker-module"], workerByteLength),
    fileSpecification(
      "producerPolicy",
      args["producer-policy"],
      CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT,
    ),
    fileSpecification("trustStore", args["trust-store"], TRUST_STORE_BYTE_LIMIT),
  ];
}

function commonInputIdentities(files) {
  return deepFreezeJson({
    profile: fileIdentity(requiredFile(files, "profile")),
    assetManifest: fileIdentity(requiredFile(files, "assetManifest")),
    buildInputLock: fileIdentity(requiredFile(files, "buildInputLock")),
    workerModule: fileIdentity(requiredFile(files, "workerModule")),
    producerPolicy: fileIdentity(requiredFile(files, "producerPolicy")),
    trustStore: fileIdentity(requiredFile(files, "trustStore")),
  });
}

function fileSpecification(name, path, maxByteLength) {
  return Object.freeze({
    name,
    path: absolutePath(path, `$.arguments.${argumentName(name)}`),
    maxByteLength,
  });
}

function argumentName(name) {
  return {
    assetManifest: "asset-manifest",
    buildInputLock: "build-input-lock",
    envelope: "envelope",
    producerPolicy: "producer-policy",
    profile: "profile",
    signingRequest: "signing-request",
    trustStore: "trust-store",
    workerModule: "worker-module",
  }[name];
}

async function readInputFiles(specifications, signal) {
  return new Map(await Promise.all(specifications.map(async (specification) => {
    const file = await readImmutableFile(specification, signal);
    return [specification.name, file];
  })));
}

async function readImmutableFile(specification, signal) {
  const parentPath = dirname(specification.path);
  const parentBefore = await inspectPrivateDirectory(
    parentPath,
    `$.arguments.${argumentName(specification.name)}.parent`,
  );
  let handle;
  try {
    handle = await open(
      specification.path,
      constants.O_RDONLY | constants.O_NOFOLLOW,
    );
    const before = await handle.stat({ bigint: true });
    const uid = currentUid();
    if (!before.isFile() || before.nlink !== 1n || before.uid !== uid ||
        before.size <= 0n ||
        before.size > BigInt(specification.maxByteLength) ||
        (before.mode & 0o222n) !== 0n) {
      invalid(
        `$.arguments.${argumentName(specification.name)}`,
        "input must be one bounded current-user-owned immutable regular file",
      );
    }
    const discovered = await lstat(specification.path, { bigint: true });
    if (!sameFileIdentity(before, discovered) ||
        await realpath(specification.path) !== specification.path) {
      conflict(
        `$.arguments.${argumentName(specification.name)}`,
        "input path identity changed before read",
      );
    }
    const byteLength = Number(before.size);
    const bytes = new Uint8Array(byteLength);
    let offset = 0;
    while (offset < byteLength) {
      throwIfAborted(signal);
      const read = await handle.read(
        bytes,
        offset,
        byteLength - offset,
        offset,
      );
      if (read.bytesRead <= 0) {
        conflict(
          `$.arguments.${argumentName(specification.name)}`,
          "input became shorter while read",
        );
      }
      offset += read.bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    const pathAfter = await lstat(specification.path, { bigint: true });
    const parentAfter = await inspectPrivateDirectory(
      parentPath,
      `$.arguments.${argumentName(specification.name)}.parent`,
    );
    if (!sameFileIdentity(before, after) ||
        !sameFileIdentity(after, pathAfter) ||
        !sameDirectoryIdentity(parentBefore, parentAfter) ||
        await realpath(specification.path) !== specification.path) {
      conflict(
        `$.arguments.${argumentName(specification.name)}`,
        "input identity changed while read",
      );
    }
    return Object.freeze({
      name: specification.name,
      path: specification.path,
      bytes,
      sha256: sha256(bytes),
      byteLength,
      identity: after,
    });
  } catch (cause) {
    if (cause instanceof CppCuteBrowserBuildProvenanceExchangeError) throw cause;
    io(
      `$.arguments.${argumentName(specification.name)}`,
      "failed to read exact immutable exchange input",
      { cause },
    );
  } finally {
    await handle?.close();
  }
}

async function persistCanonicalOutput(pathValue, bytes, signal) {
  const path = absolutePath(pathValue, "$.arguments.output");
  const parentPath = dirname(path);
  const parentBefore = await inspectPrivateDirectory(
    parentPath,
    "$.arguments.output.parent",
  );
  if (join(parentPath, basename(path)) !== path) {
    invalid("$.arguments.output", "output path must name one direct canonical child");
  }
  throwIfAborted(signal);
  let handle;
  let createdIdentity;
  let completed = false;
  try {
    handle = await open(
      path,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL |
        constants.O_NOFOLLOW,
      0o400,
    );
    createdIdentity = await handle.stat({ bigint: true });
    if (!createdIdentity.isFile() || createdIdentity.nlink !== 1n ||
        createdIdentity.uid !== currentUid()) {
      conflict("$.arguments.output", "created output has an unsafe identity");
    }
    let offset = 0;
    while (offset < bytes.byteLength) {
      throwIfAborted(signal);
      const written = await handle.write(
        bytes,
        offset,
        bytes.byteLength - offset,
        offset,
      );
      if (written.bytesWritten <= 0) {
        conflict("$.arguments.output", "output write made no progress");
      }
      offset += written.bytesWritten;
    }
    await handle.sync();
    await handle.chmod(0o444);
    const afterWrite = await handle.stat({ bigint: true });
    if (!sameStableFileIdentity(createdIdentity, afterWrite) ||
        afterWrite.size !== BigInt(bytes.byteLength) ||
        (afterWrite.mode & 0o222n) !== 0n) {
      conflict("$.arguments.output", "output identity changed while written");
    }
    await handle.close();
    handle = undefined;
    const parentAfter = await inspectPrivateDirectory(
      parentPath,
      "$.arguments.output.parent",
    );
    if (!sameDirectoryIdentity(parentBefore, parentAfter)) {
      conflict("$.arguments.output.parent", "output parent identity changed");
    }
    const directory = await open(
      parentPath,
      constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW,
    );
    try {
      await directory.sync();
    } finally {
      await directory.close();
    }
    const persisted = await readPersistedOutput(
      path,
      bytes.byteLength,
      afterWrite,
    );
    if (!sameBytes(bytes, persisted.bytes)) {
      conflict("$.arguments.output", "persisted canonical bytes changed");
    }
    completed = true;
    return Object.freeze({
      outputPath: path,
      sha256: persisted.sha256,
      byteLength: bytes.byteLength,
    });
  } catch (cause) {
    if (cause instanceof CppCuteBrowserBuildProvenanceExchangeError) throw cause;
    if (isNodeError(cause, "EEXIST") || isNodeError(cause, "ELOOP")) {
      conflict(
        "$.arguments.output",
        "output must not already exist or be a symbolic link",
        { cause },
      );
    }
    io("$.arguments.output", "failed to persist canonical exchange output", {
      cause,
    });
  } finally {
    await handle?.close();
    if (!completed && createdIdentity !== undefined) {
      await unlinkOwnedOutput(path, createdIdentity);
    }
  }
}

async function readPersistedOutput(path, expectedByteLength, expectedIdentity) {
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameFileIdentity(expectedIdentity, before) ||
        before.size !== BigInt(expectedByteLength) ||
        (before.mode & 0o222n) !== 0n) {
      conflict("$.arguments.output", "persisted output identity differs");
    }
    const bytes = new Uint8Array(expectedByteLength);
    let offset = 0;
    while (offset < expectedByteLength) {
      const read = await handle.read(
        bytes,
        offset,
        expectedByteLength - offset,
        offset,
      );
      if (read.bytesRead <= 0) {
        conflict("$.arguments.output", "persisted output became shorter");
      }
      offset += read.bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    const discovered = await lstat(path, { bigint: true });
    if (!sameFileIdentity(before, after) ||
        !sameFileIdentity(after, discovered) ||
        await realpath(path) !== path) {
      conflict("$.arguments.output", "persisted output changed while reread");
    }
    return Object.freeze({ bytes, sha256: sha256(bytes) });
  } finally {
    await handle?.close();
  }
}

async function unlinkOwnedOutput(path, expected) {
  try {
    const observed = await lstat(path, { bigint: true });
    if (observed.dev === expected.dev && observed.ino === expected.ino) {
      await unlink(path);
    }
  } catch {
    // Preserve the original failure. Never unlink an identity we did not create.
  }
}

async function inspectPrivateDirectory(path, diagnosticPath) {
  let stat;
  let canonical;
  try {
    [stat, canonical] = await Promise.all([
      lstat(path, { bigint: true }),
      realpath(path),
    ]);
  } catch (cause) {
    io(diagnosticPath, "failed to inspect exchange directory", { cause });
  }
  if (canonical !== path || !stat.isDirectory() || stat.isSymbolicLink() ||
      stat.uid !== currentUid() || (stat.mode & 0o022n) !== 0n) {
    invalid(
      diagnosticPath,
      "directory must be canonical, current-user-owned, and not group/world writable",
    );
  }
  return stat;
}

function decodeCanonicalJson(bytes, limits, path) {
  let value;
  let canonical;
  try {
    value = decodeWireJson(bytes, { limits });
    canonical = canonicalJsonBytes(value, { limits });
  } catch (cause) {
    invalid(path, "input must be bounded strict UTF-8 JSON", { cause });
  }
  if (!sameBytes(bytes, canonical)) {
    mismatch(path, "input bytes must exactly equal canonical JSON");
  }
  return value;
}

function signingRequestCoordinates(value) {
  const object = jsonObject(value, "$.inputs.signingRequest");
  if (object.schema !== CPP_CUTE_BROWSER_BUILD_PROVENANCE_SIGNING_REQUEST_SCHEMA ||
      object.version !== 1 ||
      object.authority !== "format-only-external-signing-request" ||
      typeof object.requestId !== "string" || !REQUEST_ID.test(object.requestId)) {
    invalid(
      "$.inputs.signingRequest",
      "input is not one BrowserGrad build-provenance signing request",
    );
  }
  return Object.freeze({
    builderId: nonemptyString(
      object.builderId,
      "$.inputs.signingRequest.builderId",
    ),
    keyId: pattern(
      object.keyId,
      KEY_ID,
      "$.inputs.signingRequest.keyId",
    ),
  });
}

function dsseCoordinates(value) {
  const object = jsonObject(value, "$.inputs.envelope");
  const signatures = object.signatures;
  if (typeof object.payloadType !== "string" ||
      typeof object.payload !== "string" ||
      !Array.isArray(signatures) ||
      signatures.length !== 1 ||
      signatures[0] === undefined) {
    invalid("$.inputs.envelope", "input is not one single-signature DSSE envelope");
  }
  const signature = jsonObject(signatures[0], "$.inputs.envelope.signatures[0]");
  return Object.freeze({
    payloadType: object.payloadType,
    payload: object.payload,
    keyId: pattern(
      signature.keyid,
      KEY_ID,
      "$.inputs.envelope.signatures[0].keyid",
    ),
  });
}

function jsonObject(value, path) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    invalid(path, "expected one plain JSON object");
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    invalid(path, "expected one plain JSON object");
  }
  return value;
}

function parseArguments(argv) {
  const values = {};
  for (const [index, argument] of argv.entries()) {
    const equals = argument.indexOf("=");
    if (!argument.startsWith("--") || equals <= 2 ||
        equals === argument.length - 1) {
      invalid(
        `$.argv[${index}]`,
        "arguments must use the exact --name=value form",
      );
    }
    const key = argument.slice(2, equals);
    const value = argument.slice(equals + 1);
    if (!/^[a-z][a-z0-9-]*$/u.test(key)) {
      invalid(`$.argv[${index}]`, "argument name is outside the closed grammar");
    }
    if (Object.hasOwn(values, key)) {
      invalid(`$.argv[${index}]`, `duplicate --${key} argument`);
    }
    values[key] = value;
  }
  const operation = values.operation;
  if (operation !== "signing-request" && operation !== "verify-envelope") {
    invalid(
      "$.arguments.operation",
      "operation must be signing-request or verify-envelope",
    );
  }
  const expected = ARGUMENTS[operation];
  const actual = Object.keys(values).sort();
  if (actual.length !== expected.length ||
      actual.some((key, index) => key !== expected[index])) {
    invalid(
      "$.arguments",
      `operation ${operation} requires exactly ${expected.map((key) => `--${key}`).join(", ")}`,
    );
  }
  values.operation = operation;
  values.output = absolutePath(values.output, "$.arguments.output");
  return Object.freeze(values);
}

function snapshotArguments(argv) {
  try {
    if (!Array.isArray(argv) || Object.getPrototypeOf(argv) !== Array.prototype) {
      invalid("$.argv", "expected one plain argument array");
    }
    const descriptors = Object.getOwnPropertyDescriptors(argv);
    const lengthDescriptor = descriptors.length;
    if (lengthDescriptor === undefined || !("value" in lengthDescriptor) ||
        !Number.isSafeInteger(lengthDescriptor.value) ||
        lengthDescriptor.value < 1 ||
        lengthDescriptor.value > MAX_ARGUMENT_COUNT) {
      invalid("$.argv", "argument array exceeds its closed count bound");
    }
    const expectedKeys = new Set([
      "length",
      ...Array.from(
        { length: lengthDescriptor.value },
        (_, index) => String(index),
      ),
    ]);
    const actualKeys = Reflect.ownKeys(descriptors);
    if (actualKeys.length !== expectedKeys.size ||
        actualKeys.some((key) =>
          typeof key !== "string" || !expectedKeys.has(key))) {
      invalid("$.argv", "argument array must be dense and accessor-free");
    }
    return Object.freeze(Array.from(
      { length: lengthDescriptor.value },
      (_, index) => {
        const descriptor = descriptors[String(index)];
        if (descriptor === undefined || !("value" in descriptor) ||
            typeof descriptor.value !== "string" ||
            descriptor.value.length === 0 ||
            Buffer.byteLength(descriptor.value, "utf8") >
              MAX_ARGUMENT_BYTE_LENGTH ||
            descriptor.value.includes("\0")) {
          invalid(`$.argv[${index}]`, "argument must be one bounded data string");
        }
        return descriptor.value;
      },
    ));
  } catch (cause) {
    if (cause instanceof CppCuteBrowserBuildProvenanceExchangeError) throw cause;
    invalid("$.argv", "argument array could not be inspected", { cause });
  }
}

function normalizeOptions(options) {
  try {
    if (typeof options !== "object" || options === null ||
        Object.getPrototypeOf(options) !== Object.prototype) {
      invalid("$.options", "options must be one plain data record");
    }
    const descriptors = Object.getOwnPropertyDescriptors(options);
    const keys = Reflect.ownKeys(descriptors);
    if (keys.some((key) => typeof key !== "string" || key !== "signal")) {
      invalid("$.options", "options contains unknown fields");
    }
    const signal = descriptors.signal;
    if (signal === undefined) return undefined;
    if (!("value" in signal) || signal.enumerable !== true ||
        typeof AbortSignal === "undefined" ||
        !(signal.value instanceof AbortSignal)) {
      invalid("$.options.signal", "signal must be an enumerable AbortSignal data property");
    }
    return signal.value;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserBuildProvenanceExchangeError) throw cause;
    invalid("$.options", "options could not be inspected", { cause });
  }
}

function exchangePaths(args) {
  const paths = [
    ["profile", args.profile],
    ["asset-manifest", args["asset-manifest"]],
    ["build-input-lock", args["build-input-lock"]],
    ["worker-module", args["worker-module"]],
    ["producer-policy", args["producer-policy"]],
    ["trust-store", args["trust-store"]],
    ["output", args.output],
  ];
  if (args.operation === "verify-envelope") {
    paths.push(
      ["envelope", args.envelope],
      ["signing-request", args["signing-request"]],
    );
  }
  return paths.map(([name, path]) =>
    Object.freeze({ name, path: absolutePath(path, `$.arguments.${name}`) }));
}

function assertDistinctPaths(paths) {
  const observed = new Map();
  for (const entry of paths) {
    const prior = observed.get(entry.path);
    if (prior !== undefined) {
      invalid(
        `$.arguments.${entry.name}`,
        `path must differ from --${prior}`,
      );
    }
    observed.set(entry.path, entry.name);
  }
}

function assertDistinctInputFiles(files) {
  const observed = new Map();
  for (const file of files.values()) {
    const key = `${file.identity.dev}:${file.identity.ino}`;
    const prior = observed.get(key);
    if (prior !== undefined) {
      invalid(
        `$.arguments.${argumentName(file.name)}`,
        `input inode must differ from --${argumentName(prior)}`,
      );
    }
    observed.set(key, file.name);
  }
}

function requiredFile(files, name) {
  const file = files.get(name);
  if (file === undefined) {
    invalid("$.inputs", `required ${name} input was not read`);
  }
  return file;
}

function fileIdentity(file) {
  return deepFreezeJson({
    sha256: pattern(file.sha256, SHA256, "$.inputs.sha256"),
    byteLength: String(file.byteLength),
  });
}

function absolutePath(value, path) {
  if (typeof value !== "string" || value.includes("\0") ||
      !isAbsolute(value) || normalize(value) !== value ||
      basename(value) === "." || basename(value) === "/") {
    invalid(path, "expected one normalized absolute POSIX file path");
  }
  return value;
}

function nonemptyString(value, path) {
  if (typeof value !== "string" || value.length === 0 ||
      Buffer.byteLength(value, "utf8") > 4_096 || value.includes("\0")) {
    invalid(path, "expected one bounded nonempty string");
  }
  return value;
}

function pattern(value, expression, path) {
  if (typeof value !== "string" || !expression.test(value)) {
    invalid(path, "value differs from the closed format");
  }
  return value;
}

function hashProjection(domain, body) {
  return sha256(canonicalJsonBytes({ domain, body }));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function encodeBase64(bytes) {
  return Buffer.from(bytes).toString("base64");
}

function sameBytes(left, right) {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= left[index] ^ right[index];
  }
  return difference === 0;
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev &&
    left.ino === right.ino &&
    left.mode === right.mode &&
    left.nlink === right.nlink &&
    left.uid === right.uid &&
    left.gid === right.gid &&
    left.size === right.size &&
    left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs;
}

function sameStableFileIdentity(left, right) {
  return left.dev === right.dev &&
    left.ino === right.ino &&
    left.nlink === right.nlink &&
    left.uid === right.uid &&
    left.gid === right.gid;
}

function sameDirectoryIdentity(left, right) {
  return left.dev === right.dev &&
    left.ino === right.ino &&
    left.mode === right.mode &&
    left.uid === right.uid &&
    left.gid === right.gid;
}

function currentUid() {
  if (typeof process.getuid !== "function") {
    invalid("$", "current-user ownership checks are unavailable");
  }
  return BigInt(process.getuid());
}

function throwIfAborted(signal) {
  if (signal === undefined) return;
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined ||
      Reflect.apply(ABORT_SIGNAL_ABORTED_GETTER, signal, []) === true) {
    throw new CppCuteBrowserBuildProvenanceExchangeError(
      `${ERROR_PREFIX}-CANCELLED`,
      "$.options.signal",
      "browser build provenance exchange was cancelled",
    );
  }
}

function isNodeError(value, code) {
  return typeof value === "object" && value !== null &&
    "code" in value && value.code === code;
}

function invalid(path, message, options) {
  throw new CppCuteBrowserBuildProvenanceExchangeError(
    `${ERROR_PREFIX}-INVALID`,
    path,
    message,
    options,
  );
}

function mismatch(path, message, options) {
  throw new CppCuteBrowserBuildProvenanceExchangeError(
    `${ERROR_PREFIX}-MISMATCH`,
    path,
    message,
    options,
  );
}

function resource(path, message, options) {
  throw new CppCuteBrowserBuildProvenanceExchangeError(
    `${ERROR_PREFIX}-RESOURCE-LIMIT`,
    path,
    message,
    options,
  );
}

function conflict(path, message, options) {
  throw new CppCuteBrowserBuildProvenanceExchangeError(
    `${ERROR_PREFIX}-CONFLICT`,
    path,
    message,
    options,
  );
}

function io(path, message, options) {
  throw new CppCuteBrowserBuildProvenanceExchangeError(
    `${ERROR_PREFIX}-IO`,
    path,
    message,
    options,
  );
}

if (process.argv[1] !== undefined &&
    pathToFileURL(process.argv[1]).href === import.meta.url) {
  const result = await runCppCuteBrowserBuildProvenanceExchange(
    process.argv.slice(2),
  );
  process.stdout.write(`${JSON.stringify({
    operation: result.operation,
    outputPath: result.outputPath,
    outputSha256: result.outputSha256,
    outputByteLength: result.outputByteLength,
  })}\n`);
}
