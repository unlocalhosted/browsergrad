import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  lstat,
  open,
  realpath,
} from "node:fs/promises";
import {
  isAbsolute,
  join,
  resolve,
} from "node:path";
import { pathToFileURL } from "node:url";

import {
  canonicalJsonBytes,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
} from "../../dist/cpp_cute_browser_assets.js";
import {
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MAJOR,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MINOR,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_SCHEMA,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE,
  admitCppCuteBrowserDistributionApprovalPolicy,
  copyAdmittedCppCuteBrowserDistributionApprovalPolicyBytes,
  deriveCppCuteBrowserDistributionApprovalPolicyId,
} from "../../dist/cpp_cute_browser_distribution_approval_policy.js";
import {
  CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MAJOR,
  CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MINOR,
  CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_SCHEMA,
  admitCppCuteBrowserProducerTrustPolicy,
  copyAdmittedCppCuteBrowserProducerTrustPolicyBytes,
  deriveCppCuteBrowserProducerTrustPolicyId,
} from "../../dist/cpp_cute_browser_producer_trust_policy.js";
import {
  CPP_CUTE_ATTESTATION_TRUST_STORE_MAJOR,
  CPP_CUTE_ATTESTATION_TRUST_STORE_MINOR,
  CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA,
  prepareCppCuteAttestationTrustStore,
} from "../../dist/cpp_cute_frontend_provenance.js";
import {
  materializeCppCuteBrowserDistributionOutputFiles,
} from "./cpp_cute_browser_distribution_output_files.mjs";

export const CPP_CUTE_BROWSER_PRODUCTION_POLICY_HANDOFF_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-production-policy-handoff";

const ERROR_CODE =
  "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-POLICY-AUTHORING";
const HANDOFF_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-production-policy-handoff.v1";
const PRODUCER_PUBLIC_KEY_PATH = "producer-public-key.spki.der";
const REVIEWER_PUBLIC_KEY_PATH = "reviewer-public-key.spki.der";
const PRODUCER_TRUST_STORE_PATH = "producer-trust-store.json";
const PRODUCER_POLICY_PATH = "producer-trust-policy.json";
const REVIEWER_TRUST_STORE_PATH = "reviewer-trust-store.json";
const APPROVAL_POLICY_PATH = "distribution-approval-policy.json";
const HANDOFF_PATH = "production-policy-handoff.json";
const MAX_PUBLIC_KEY_BYTES = 4 * 1024;
const MAX_ARGUMENT_BYTES = 4 * 1024;

export class CppCuteBrowserProductionPolicyAuthoringError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserProductionPolicyAuthoringError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Converts two preprovisioned public P-256 keys into exact trust stores and
 * host-admitted policies. This boundary has no private-key or signing input.
 */
export async function authorCppCuteBrowserProductionPolicies(input) {
  const object = exactObject(
    input,
    ["outputRoot", "producerId", "reviewerId"],
    "$.input",
  );
  const outputRoot = absolutePath(
    object.outputRoot,
    "$.input.outputRoot",
  );
  const producerId = boundedString(
    object.producerId,
    "$.input.producerId",
  );
  const reviewerId = boundedString(
    object.reviewerId,
    "$.input.reviewerId",
  );
  if (producerId === reviewerId) {
    invalid(
      "$.input.reviewerId",
      "producer and reviewer identities must differ",
    );
  }

  const [producerKey, reviewerKey] = await Promise.all([
    readImmutablePublicKey(
      outputRoot,
      PRODUCER_PUBLIC_KEY_PATH,
      "$.input.producerPublicKey",
    ),
    readImmutablePublicKey(
      outputRoot,
      REVIEWER_PUBLIC_KEY_PATH,
      "$.input.reviewerPublicKey",
    ),
  ]);
  if (producerKey.keyId === reviewerKey.keyId) {
    invalid(
      "$.input.reviewerPublicKey",
      "producer and reviewer public-key hashes must differ",
    );
  }

  const [producerTrust, reviewerTrust] = await Promise.all([
    prepareTrustStore(producerId, producerKey, "producer"),
    prepareTrustStore(reviewerId, reviewerKey, "reviewer"),
  ]);
  const [producerPolicy, approvalPolicy] = await Promise.all([
    prepareProducerPolicy(producerId, producerKey.keyId, producerTrust),
    prepareApprovalPolicy(reviewerId, reviewerKey.keyId, reviewerTrust),
  ]);

  const authoredOutputs = Object.freeze([
    outputIdentity(PRODUCER_POLICY_PATH, producerPolicy.bytes),
    outputIdentity(PRODUCER_TRUST_STORE_PATH, producerTrust.bytes),
    outputIdentity(APPROVAL_POLICY_PATH, approvalPolicy.bytes),
    outputIdentity(REVIEWER_TRUST_STORE_PATH, reviewerTrust.bytes),
  ].sort((left, right) => compareUtf8(left.outputPath, right.outputPath)));
  const projection = Object.freeze({
    schema: CPP_CUTE_BROWSER_PRODUCTION_POLICY_HANDOFF_SCHEMA,
    version: 1,
    authority: "package-authored-public-policy-handoff-only",
    producer: Object.freeze({
      identity: producerId,
      keyId: producerKey.keyId,
      publicKey: publicKeyRecord(PRODUCER_PUBLIC_KEY_PATH, producerKey),
      trustStore: authorityRecord(
        PRODUCER_TRUST_STORE_PATH,
        producerTrust.bytes,
        "trustStoreHash",
        producerTrust.prepared.trustStoreHash,
      ),
      policy: authorityRecord(
        PRODUCER_POLICY_PATH,
        producerPolicy.bytes,
        "policyId",
        producerPolicy.admitted.policyId,
      ),
    }),
    reviewer: Object.freeze({
      identity: reviewerId,
      keyId: reviewerKey.keyId,
      publicKey: publicKeyRecord(REVIEWER_PUBLIC_KEY_PATH, reviewerKey),
      trustStore: authorityRecord(
        REVIEWER_TRUST_STORE_PATH,
        reviewerTrust.bytes,
        "trustStoreHash",
        reviewerTrust.prepared.trustStoreHash,
      ),
      policy: authorityRecord(
        APPROVAL_POLICY_PATH,
        approvalPolicy.bytes,
        "policyId",
        approvalPolicy.admitted.policyId,
      ),
    }),
    separation: Object.freeze({
      producerReviewerIdentitySeparated: true,
      producerReviewerKeySeparated: true,
    }),
    authoredOutputs,
    claims: Object.freeze({
      exactPublicKeysReverified: true,
      canonicalTrustStoresPrepared: true,
      canonicalPoliciesAdmitted: true,
      privateKeyAccepted: false,
      signatureCreated: false,
      externalKeyControlVerified: false,
      producerTrusted: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      releaseReady: false,
    }),
  });
  const record = Object.freeze({
    ...projection,
    handoffId: `bg.cpp.browser-production-policy-handoff.sha256.${sha256(
      canonicalJsonBytes({ domain: HANDOFF_HASH_DOMAIN, handoff: projection }),
    )}`,
  });
  const recordBytes = canonicalJsonBytes(record);

  let materialization;
  try {
    materialization =
      await materializeCppCuteBrowserDistributionOutputFiles({
        outputRoot,
        existingOutputs: [
          publicKeyRecord(PRODUCER_PUBLIC_KEY_PATH, producerKey),
          publicKeyRecord(REVIEWER_PUBLIC_KEY_PATH, reviewerKey),
        ],
        outputs: [
          { outputPath: PRODUCER_TRUST_STORE_PATH, bytes: producerTrust.bytes },
          { outputPath: PRODUCER_POLICY_PATH, bytes: producerPolicy.bytes },
          { outputPath: REVIEWER_TRUST_STORE_PATH, bytes: reviewerTrust.bytes },
          { outputPath: APPROVAL_POLICY_PATH, bytes: approvalPolicy.bytes },
          { outputPath: HANDOFF_PATH, bytes: recordBytes },
        ],
      });
  } catch (cause) {
    invalid(
      "$.input.outputRoot",
      "failed to materialize exact production-policy handoff",
      { cause },
    );
  }
  return Object.freeze({
    operation: "author-production-policies",
    record,
    recordSha256: sha256(recordBytes),
    recordByteLength: String(recordBytes.byteLength),
    materialization,
  });
}

export function parseCppCuteBrowserProductionPolicyAuthoringArguments(argv) {
  if (!Array.isArray(argv)) {
    invalid("$arguments", "expected one argument array");
  }
  const arguments_ = argv[0] === "--" ? argv.slice(1) : argv;
  if (arguments_.length !== 3) {
    invalid(
      "$arguments",
      "expected exactly --output-root, --producer-id, and --reviewer-id",
    );
  }
  const values = {};
  for (const [index, argument] of arguments_.entries()) {
    if (typeof argument !== "string" || argument.length === 0 ||
        Buffer.byteLength(argument) > MAX_ARGUMENT_BYTES) {
      invalid(`$arguments[${index}]`, "expected one bounded argument string");
    }
    const match = /^--(output-root|producer-id|reviewer-id)=(.+)$/u.exec(
      argument,
    );
    if (match === null || match[1] === undefined || match[2] === undefined) {
      invalid(
        `$arguments[${index}]`,
        "unsupported argument; private-key and signing inputs are forbidden",
      );
    }
    if (Object.hasOwn(values, match[1])) {
      invalid(`$arguments[${index}]`, `duplicate --${match[1]}`);
    }
    values[match[1]] = match[2];
  }
  const actual = Object.keys(values).sort(compareText);
  const expected = ["output-root", "producer-id", "reviewer-id"];
  if (actual.some((key, index) => key !== expected[index])) {
    invalid(
      "$arguments",
      "expected exactly --output-root, --producer-id, and --reviewer-id",
    );
  }
  return Object.freeze({
    outputRoot: absolutePath(values["output-root"], "$arguments.output-root"),
    producerId: boundedString(values["producer-id"], "$arguments.producer-id"),
    reviewerId: boundedString(values["reviewer-id"], "$arguments.reviewer-id"),
  });
}

export async function runCppCuteBrowserProductionPolicyAuthoring(argv) {
  return await authorCppCuteBrowserProductionPolicies(
    parseCppCuteBrowserProductionPolicyAuthoringArguments(argv),
  );
}

async function prepareTrustStore(identity, publicKey, role) {
  const input = {
    schema: CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA,
    version: {
      major: CPP_CUTE_ATTESTATION_TRUST_STORE_MAJOR,
      minor: CPP_CUTE_ATTESTATION_TRUST_STORE_MINOR,
    },
    keys: [{
      keyId: publicKey.keyId,
      builderId: identity,
      algorithm: "ecdsa-p256-sha256",
      spkiDerBase64: Buffer.from(publicKey.bytes).toString("base64"),
    }],
  };
  try {
    const prepared = await prepareCppCuteAttestationTrustStore(input);
    return Object.freeze({
      prepared,
      bytes: canonicalJsonBytes(input),
    });
  } catch (cause) {
    invalid(
      `$.input.${role}PublicKey`,
      `${role} public key or identity is invalid`,
      { cause },
    );
  }
}

async function prepareProducerPolicy(identity, keyId, trustStore) {
  const projection = {
    schema: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MAJOR,
      minor: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MINOR,
    },
    predicateType: CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
    trustStoreSha256: trustStore.prepared.trustStoreHash,
    builderIds: [identity],
    keyIds: [keyId],
  };
  try {
    const policyId = await deriveCppCuteBrowserProducerTrustPolicyId(
      projection,
    );
    const admitted = await admitCppCuteBrowserProducerTrustPolicy(
      canonicalJsonBytes({ ...projection, policyId }),
    );
    return Object.freeze({
      admitted,
      bytes: copyAdmittedCppCuteBrowserProducerTrustPolicyBytes(admitted),
    });
  } catch (cause) {
    invalid("$.input.producerId", "producer policy admission failed", {
      cause,
    });
  }
}

async function prepareApprovalPolicy(identity, keyId, trustStore) {
  const projection = {
    schema: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MAJOR,
      minor: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MINOR,
    },
    predicateType: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE,
    trustStoreSha256: trustStore.prepared.trustStoreHash,
    reviewerIds: [identity],
    keyIds: [keyId],
  };
  try {
    const policyId =
      await deriveCppCuteBrowserDistributionApprovalPolicyId(projection);
    const admitted = await admitCppCuteBrowserDistributionApprovalPolicy(
      canonicalJsonBytes({ ...projection, policyId }),
    );
    return Object.freeze({
      admitted,
      bytes:
        copyAdmittedCppCuteBrowserDistributionApprovalPolicyBytes(admitted),
    });
  } catch (cause) {
    invalid("$.input.reviewerId", "approval policy admission failed", {
      cause,
    });
  }
}

async function readImmutablePublicKey(outputRoot, outputPath, path) {
  const filePath = join(outputRoot, outputPath);
  let handle;
  try {
    const discovered = await lstat(filePath, { bigint: true });
    const uid = typeof process.getuid === "function"
      ? BigInt(process.getuid())
      : undefined;
    if (!discovered.isFile() || discovered.isSymbolicLink() ||
        discovered.nlink !== 1n || discovered.size <= 0n ||
        discovered.size > BigInt(MAX_PUBLIC_KEY_BYTES) ||
        Number(discovered.mode & 0o222n) !== 0 ||
        (uid !== undefined && discovered.uid !== uid) ||
        await realpath(filePath) !== filePath) {
      invalid(
        path,
        "expected one owned canonical non-writable public-key file",
      );
    }
    handle = await open(filePath, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameFileIdentity(discovered, before)) {
      invalid(path, "public-key file identity changed before read");
    }
    const bytes = new Uint8Array(Number(before.size));
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesRead } = await handle.read(
        bytes,
        offset,
        bytes.byteLength - offset,
        offset,
      );
      if (bytesRead <= 0) {
        invalid(path, "public-key file changed while read");
      }
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, after)) {
      invalid(path, "public-key file identity changed while read");
    }
    const digest = sha256(bytes);
    return Object.freeze({
      bytes,
      keyId: `sha256:${digest}`,
      sha256: digest,
      byteLength: String(bytes.byteLength),
    });
  } catch (cause) {
    if (cause instanceof CppCuteBrowserProductionPolicyAuthoringError) {
      throw cause;
    }
    invalid(path, "failed to read immutable public-key file", { cause });
  } finally {
    await handle?.close();
  }
}

function publicKeyRecord(outputPath, key) {
  return Object.freeze({
    outputPath,
    sha256: key.sha256,
    byteLength: key.byteLength,
  });
}

function authorityRecord(outputPath, bytes, idName, id) {
  return Object.freeze({
    outputPath,
    [idName]: id,
    sha256: sha256(bytes),
    byteLength: String(bytes.byteLength),
  });
}

function outputIdentity(outputPath, bytes) {
  return Object.freeze({
    outputPath,
    sha256: sha256(bytes),
    byteLength: String(bytes.byteLength),
  });
}

function exactObject(value, keys, path) {
  if (typeof value !== "object" || value === null ||
      Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(path, "expected one plain data record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const actual = Reflect.ownKeys(descriptors);
  if (actual.length !== keys.length ||
      actual.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(path, `expected only ${keys.join(", ")}`);
  }
  const result = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "expected data property");
    }
    result[key] = descriptor.value;
  }
  return result;
}

function absolutePath(value, path) {
  if (typeof value !== "string" || !isAbsolute(value) ||
      value.includes("\0")) {
    invalid(path, "expected one absolute NUL-free path");
  }
  return value;
}

function boundedString(value, path) {
  if (typeof value !== "string" || value.length === 0 ||
      Buffer.byteLength(value) > MAX_ARGUMENT_BYTES) {
    invalid(path, "expected one bounded nonempty string");
  }
  return value;
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino &&
    left.mode === right.mode && left.nlink === right.nlink &&
    left.size === right.size && left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs;
}

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function compareText(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function invalid(path, message, options) {
  throw new CppCuteBrowserProductionPolicyAuthoringError(
    path,
    message,
    options,
  );
}

if (process.argv[1] !== undefined &&
    import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  runCppCuteBrowserProductionPolicyAuthoring(process.argv.slice(2)).then(
    (result) => {
      process.stdout.write(`${JSON.stringify(result.record)}\n`);
    },
    (cause) => {
      const error = cause instanceof Error
        ? cause
        : new Error("unknown production-policy authoring failure");
      process.stderr.write(`${error.stack ?? error.message}\n`);
      process.exitCode = 1;
    },
  );
}
