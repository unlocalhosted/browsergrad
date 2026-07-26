import {
  hashCanonicalJson,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapPreparedCppCuteBrowserAssetManifest,
  type PreparedCppCuteBrowserAssetManifest,
} from "./cpp_cute_browser_assets.js";
import {
  unwrapPreparedCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
} from "./cpp_cute_browser_build_lock.js";
import {
  CPP_CUTE_BROWSER_BUILD_DSSE_PAYLOAD_TYPE,
  CPP_CUTE_BROWSER_BUILD_IN_TOTO_STATEMENT_TYPE,
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
  CPP_CUTE_BROWSER_BUILD_TYPE,
  cppCuteBrowserBuildProvenanceDsseSigningBytes,
  cppCuteBrowserBuildProvenancePayloadBase64,
  decodeUntrustedCppCuteBrowserBuildProvenanceSyntax,
  deriveCppCuteBrowserBuildSubjectIdentity,
  type CppCuteBrowserBuildProvenanceEnvelopeV1,
  type CppCuteBrowserBuildProvenanceStatementV1,
  type CppCuteBrowserBuildSubjectIdentity,
} from "./cpp_cute_browser_build_provenance_syntax.js";
import {
  CppCuteFrontendProvenanceError,
  verifyCppCutePreparedAttestationSignature,
  type PreparedCppCuteAttestationTrustStore,
} from "./cpp_cute_frontend_provenance.js";
import {
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import {
  unwrapAdmittedCppCuteBrowserProducerTrustPolicy,
  type AdmittedCppCuteBrowserProducerTrustPolicy,
} from "./cpp_cute_browser_producer_trust_policy.js";
import {
  inspectVerifiedCppCuteBrowserWorkerBundle,
  type VerifiedCppCuteBrowserWorkerBundle,
} from "./cpp_cute_browser_worker_bundle.js";

const CAPTURED_OBJECT = Object;
const CAPTURED_REFLECT = Reflect;
const NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTORS =
  Object.getOwnPropertyDescriptors;
const NATIVE_OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

const VERIFIED_BUILD_SIGNATURE_BINDINGS = new WeakMap<object, VerifiedCppCuteBrowserBuildSignatureBindingRecord>();

declare const verifiedCppCuteBrowserBuildSignatureBindingBrand: unique symbol;

/**
 * Manifest-policy signature/build-subject binding. This is deliberately named
 * as a signature binding rather than provenance authority. Until a separate
 * package-owned trust-root authority admits the manifest policy, this does not
 * prove producer trust, acquired asset bytes, legal approval, complete
 * reproducibility, Worker execution, or release readiness.
 */
export interface VerifiedCppCuteBrowserBuildSignatureBinding {
  readonly [verifiedCppCuteBrowserBuildSignatureBindingBrand]: true;
  readonly buildSubjectId: string;
  readonly buildSubjectSha256: string;
  readonly statementSha256: string;
  readonly evidenceSha256: string;
  readonly builderId: string;
  readonly keyId: string;
  readonly trustStoreSha256: string;
  readonly profileHash: string;
  readonly manifestId: string;
  readonly manifestSha256: string;
  readonly manifestByteLength: WireU64;
  readonly assetSetSha256: string;
  readonly buildInputLockResourceSha256: string;
  readonly workerBundleSha256: string;
  readonly signatureVerified: true;
  readonly manifestSignaturePolicyMatched: true;
  readonly producerTrusted: false;
  readonly buildSubjectBound: true;
  readonly exactAssetBytesVerified: false;
  readonly fullDistributedOutputSetReproducible: false;
  readonly licenseReviewComplete: false;
  readonly distributionAuthorized: false;
  readonly workerExecutionObserved: false;
  readonly releaseReady: false;
}

export interface VerifiedCppCuteBrowserBuildSignatureBindingRecord {
  readonly envelope: CppCuteBrowserBuildProvenanceEnvelopeV1;
  readonly statement: CppCuteBrowserBuildProvenanceStatementV1;
  readonly buildSubject: CppCuteBrowserBuildSubjectIdentity;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  readonly workerBundle: VerifiedCppCuteBrowserWorkerBundle;
  readonly trustStore: PreparedCppCuteAttestationTrustStore;
}

export interface VerifyCppCuteBrowserBuildProvenanceRequest {
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  readonly workerBundle: VerifiedCppCuteBrowserWorkerBundle;
  readonly trustStore: PreparedCppCuteAttestationTrustStore;
  readonly signal?: AbortSignal;
}

export interface CreateCppCuteBrowserBuildProvenanceSigningRequestInput {
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  readonly workerBundle: VerifiedCppCuteBrowserWorkerBundle;
  readonly trustPolicy: AdmittedCppCuteBrowserProducerTrustPolicy;
  readonly trustStore: PreparedCppCuteAttestationTrustStore;
  readonly builderId: string;
  readonly keyId: string;
}

export interface CreateCppCuteBrowserBuildProvenanceSigningRequestOptions {
  readonly signal?: AbortSignal;
}

/**
 * Exact format-only material for an external builder signature. This does not
 * verify a signature or mint producer, asset, execution, distribution, or
 * release authority.
 */
export interface CppCuteBrowserBuildProvenanceSigningRequest {
  readonly formatOnly: true;
  readonly policyId: string;
  readonly policySha256: string;
  readonly builderId: string;
  readonly keyId: string;
  readonly statement: CppCuteBrowserBuildProvenanceStatementV1;
  readonly payloadType: typeof CPP_CUTE_BROWSER_BUILD_DSSE_PAYLOAD_TYPE;
  readonly payload: string;
  readonly signingBytes: Uint8Array;
  readonly signatureVerified: false;
  readonly producerTrusted: false;
  readonly exactAssetBytesVerified: false;
  readonly fullDistributedOutputSetReproducible: false;
  readonly licenseReviewComplete: false;
  readonly distributionAuthorized: false;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
  readonly backendExecutionObserved: false;
  readonly releaseReady: false;
}

export type CppCuteBrowserBuildProvenanceErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-BINDING"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-POLICY"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SIGNATURE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-UNVERIFIED";

export class CppCuteBrowserBuildProvenanceError extends Error {
  constructor(
    readonly code: CppCuteBrowserBuildProvenanceErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserBuildProvenanceError";
  }
}

export async function createCppCuteBrowserBuildProvenanceSigningRequest(
  input: CreateCppCuteBrowserBuildProvenanceSigningRequestInput,
  options: CreateCppCuteBrowserBuildProvenanceSigningRequestOptions = {},
): Promise<CppCuteBrowserBuildProvenanceSigningRequest> {
  const request = normalizeSigningRequestInput(input);
  const signal = normalizeSigningRequestOptions(options);
  throwIfAborted(signal);

  let manifestRecord: ReturnType<
    typeof unwrapPreparedCppCuteBrowserAssetManifest
  >;
  let policyRecord: ReturnType<
    typeof unwrapAdmittedCppCuteBrowserProducerTrustPolicy
  >;
  let worker: ReturnType<typeof inspectVerifiedCppCuteBrowserWorkerBundle>;
  try {
    manifestRecord = unwrapPreparedCppCuteBrowserAssetManifest(
      request.assetManifest,
    );
  } catch (cause) {
    unverifiedAt(
      "$.input.assetManifest",
      "asset manifest is not an opaque prepared authority",
      cause,
    );
  }
  try {
    unwrapPreparedCppCuteBrowserBuildInputLock(request.buildInputLock);
  } catch (cause) {
    unverifiedAt(
      "$.input.buildInputLock",
      "build-input lock is not an opaque prepared authority",
      cause,
    );
  }
  try {
    worker = inspectVerifiedCppCuteBrowserWorkerBundle(request.workerBundle);
  } catch (cause) {
    unverifiedAt(
      "$.input.workerBundle",
      "Worker bundle is not an opaque verified package authority",
      cause,
    );
  }
  try {
    policyRecord = unwrapAdmittedCppCuteBrowserProducerTrustPolicy(
      request.trustPolicy,
    );
  } catch (cause) {
    unverifiedAt(
      "$.input.trustPolicy",
      "producer policy is not an opaque host admission authority",
      cause,
    );
  }

  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(
    manifestRecord.profile,
  );
  const profile = profileRecord.profile;
  const manifest = manifestRecord.manifest;
  const manifestPolicy = manifest.body.buildProvenancePolicy;
  const policy = policyRecord.policy;
  if (request.trustPolicy.policyId !== policy.policyId ||
      request.trustPolicy.policySha256.length !== 64 ||
      request.trustPolicy.policyVersion !== "1.0" ||
      request.trustPolicy.predicateType !== policy.predicateType ||
      request.trustPolicy.trustStoreSha256 !== policy.trustStoreSha256 ||
      !sameStrings(request.trustPolicy.builderIds, policy.builderIds) ||
      !sameStrings(request.trustPolicy.keyIds, policy.keyIds) ||
      request.trustPolicy.hostOnly !== true ||
      request.trustPolicy.workerTransferable !== false ||
      request.trustPolicy.producerTrusted !== false ||
      request.trustPolicy.releaseReady !== false) {
    binding(
      "$.input.trustPolicy",
      "producer policy projection differs from its retained canonical authority",
    );
  }
  if (!policy.builderIds.includes(request.builderId)) {
    policyMismatch(
      "$.input.builderId",
      "builder is not admitted by the exact producer trust policy",
    );
  }
  if (!policy.keyIds.includes(request.keyId)) {
    policyMismatch(
      "$.input.keyId",
      "key is not admitted by the exact producer trust policy",
    );
  }
  if (request.trustStore.trustStoreHash !== policy.trustStoreSha256) {
    policyMismatch(
      "$.input.trustStore",
      "prepared trust store differs from the exact producer trust policy",
    );
  }
  if (!request.trustStore.keyIds.includes(request.keyId)) {
    policyMismatch(
      "$.input.keyId",
      "key is not present in the prepared policy-pinned trust store",
    );
  }
  if (policy.predicateType !==
        CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE ||
      policy.predicateType !== manifestPolicy.predicateType) {
    policyMismatch(
      "$.input.trustPolicy.predicateType",
      "producer policy does not admit the manifest build predicate",
    );
  }
  if (policy.trustStoreSha256 !== manifestPolicy.trustStoreSha256) {
    policyMismatch(
      "$.input.trustPolicy.trustStoreSha256",
      "producer policy does not admit the manifest trust store",
    );
  }
  if (!manifestPolicy.builderIds.includes(request.builderId)) {
    policyMismatch(
      "$.input.builderId",
      "builder is not admitted by the exact asset-manifest policy",
    );
  }
  if (profile.deployment.buildProvenanceLockSha256 !==
        request.buildInputLock.resourceSha256) {
    binding(
      "$.input.buildInputLock",
      "profile build-input lock differs from the signing request",
    );
  }
  if (profile.deployment.worker.moduleSha256 !== worker.sha256 ||
      profile.deployment.worker.moduleByteLength !== worker.byteLength) {
    binding(
      "$.input.workerBundle",
      "profile Worker module differs from the signing request",
    );
  }

  const buildSubject = await deriveCppCuteBrowserBuildSubjectIdentity({
    assetManifest: request.assetManifest,
    buildInputLock: request.buildInputLock,
    workerBundle: request.workerBundle,
  });
  throwIfAborted(signal);
  if (manifest.body.buildSubjectIds.length !== 1 ||
      manifest.body.buildSubjectIds[0] !== buildSubject.buildSubjectId) {
    binding(
      "$.input.assetManifest.body.buildSubjectIds",
      "asset manifest does not bind exactly the derived build subject",
    );
  }

  const statement = NATIVE_OBJECT_FREEZE({
    _type: CPP_CUTE_BROWSER_BUILD_IN_TOTO_STATEMENT_TYPE,
    subject: NATIVE_OBJECT_FREEZE([NATIVE_OBJECT_FREEZE({
      name: buildSubject.buildSubjectId,
      digest: NATIVE_OBJECT_FREEZE({
        sha256: buildSubject.buildSubjectSha256,
      }),
    })]),
    predicateType: CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
    predicate: NATIVE_OBJECT_FREEZE({
      builderId: request.builderId,
      buildType: CPP_CUTE_BROWSER_BUILD_TYPE,
      buildSubject: NATIVE_OBJECT_FREEZE({
        buildSubjectId: buildSubject.buildSubjectId,
        buildSubjectSha256: buildSubject.buildSubjectSha256,
      }),
      profile: NATIVE_OBJECT_FREEZE({
        profileId: profile.profileId,
        profileHash: profileRecord.profileHash,
        compilationContractHash: profileRecord.compilationContractHash,
      }),
      assetManifest: NATIVE_OBJECT_FREEZE({
        manifestId: request.assetManifest.manifestId,
        manifestSha256: request.assetManifest.manifestSha256,
        manifestByteLength: request.assetManifest.manifestByteLength,
        assetSetSha256: request.assetManifest.assetSetSha256,
      }),
      buildInputLock: NATIVE_OBJECT_FREEZE({
        lockId: request.buildInputLock.lockId,
        resourceSha256: request.buildInputLock.resourceSha256,
        recipeSha256: request.buildInputLock.recipeSha256,
      }),
      workerBundle: NATIVE_OBJECT_FREEZE({
        bundleId: worker.bundleId,
        sha256: worker.sha256,
        byteLength: worker.byteLength,
        factorySha256: worker.factorySha256,
        factoryByteLength: worker.factoryByteLength,
      }),
      authorityLimits: NATIVE_OBJECT_FREEZE({
        fullDistributedOutputSetReproducible: false,
        licenseReviewComplete: false,
        distributionAuthorized: false,
        releaseReady: false,
      }),
    }),
  }) as CppCuteBrowserBuildProvenanceStatementV1;
  const signingBytes = cppCuteBrowserBuildProvenanceDsseSigningBytes(statement);
  throwIfAborted(signal);
  return NATIVE_OBJECT_FREEZE({
    formatOnly: true,
    policyId: request.trustPolicy.policyId,
    policySha256: request.trustPolicy.policySha256,
    builderId: request.builderId,
    keyId: request.keyId,
    statement,
    payloadType: CPP_CUTE_BROWSER_BUILD_DSSE_PAYLOAD_TYPE,
    payload: cppCuteBrowserBuildProvenancePayloadBase64(statement),
    signingBytes,
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
  });
}

export async function verifyCppCuteBrowserBuildSignatureBinding(
  value: unknown,
  request: VerifyCppCuteBrowserBuildProvenanceRequest,
): Promise<VerifiedCppCuteBrowserBuildSignatureBinding> {
  const assetManifest = request.assetManifest;
  const buildInputLock = request.buildInputLock;
  const workerBundle = request.workerBundle;
  const trustStore = request.trustStore;
  const signal = request.signal;
  throwIfAborted(signal);
  const decoded = decodeUntrustedCppCuteBrowserBuildProvenanceSyntax(
    value,
    signal === undefined ? {} : { signal },
  );
  const manifestRecord = unwrapPreparedCppCuteBrowserAssetManifest(assetManifest);
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(manifestRecord.profile);
  unwrapPreparedCppCuteBrowserBuildInputLock(buildInputLock);
  const lock = buildInputLock;
  const worker = inspectVerifiedCppCuteBrowserWorkerBundle(workerBundle);
  const buildSubject = await deriveCppCuteBrowserBuildSubjectIdentity({
    assetManifest,
    buildInputLock,
    workerBundle,
    ...(signal === undefined ? {} : { signal }),
  });
  throwIfAborted(signal);

  const profile = profileRecord.profile;
  const deployment = profile.deployment;
  const manifest = manifestRecord.manifest;
  const provenancePolicy = manifest.body.buildProvenancePolicy;
  const predicate = decoded.statement.predicate;
  const signature = decoded.envelope.signatures[0];

  if (deployment.buildProvenanceLockSha256 !== lock.resourceSha256) {
    binding("$.payload.predicate.buildInputLock.resourceSha256", "profile build-input lock differs from provenance input");
  }
  if (deployment.worker.moduleSha256 !== worker.sha256 ||
      deployment.worker.moduleByteLength !== worker.byteLength) {
    binding("$.payload.predicate.workerBundle", "profile Worker module differs from the verified package bundle");
  }
  if (manifest.body.buildSubjectIds.length !== 1 ||
      manifest.body.buildSubjectIds[0] !== buildSubject.buildSubjectId) {
    binding("$.assetManifest.body.buildSubjectIds", "asset manifest must reference exactly the derived build subject");
  }

  equal(decoded.statement.subject[0].name, buildSubject.buildSubjectId, "$.payload.subject[0].name");
  equal(decoded.statement.subject[0].digest.sha256, buildSubject.buildSubjectSha256, "$.payload.subject[0].digest.sha256");
  equal(predicate.buildSubject.buildSubjectId, buildSubject.buildSubjectId, "$.payload.predicate.buildSubject.buildSubjectId");
  equal(predicate.buildSubject.buildSubjectSha256, buildSubject.buildSubjectSha256, "$.payload.predicate.buildSubject.buildSubjectSha256");
  equal(predicate.profile.profileId, profile.profileId, "$.payload.predicate.profile.profileId");
  equal(predicate.profile.profileHash, profileRecord.profileHash, "$.payload.predicate.profile.profileHash");
  equal(
    predicate.profile.compilationContractHash,
    profileRecord.compilationContractHash,
    "$.payload.predicate.profile.compilationContractHash",
  );
  equal(predicate.assetManifest.manifestId, assetManifest.manifestId, "$.payload.predicate.assetManifest.manifestId");
  equal(predicate.assetManifest.manifestSha256, assetManifest.manifestSha256, "$.payload.predicate.assetManifest.manifestSha256");
  equal(predicate.assetManifest.manifestByteLength, assetManifest.manifestByteLength, "$.payload.predicate.assetManifest.manifestByteLength");
  equal(predicate.assetManifest.assetSetSha256, assetManifest.assetSetSha256, "$.payload.predicate.assetManifest.assetSetSha256");
  equal(predicate.buildInputLock.lockId, lock.lockId, "$.payload.predicate.buildInputLock.lockId");
  equal(predicate.buildInputLock.resourceSha256, lock.resourceSha256, "$.payload.predicate.buildInputLock.resourceSha256");
  equal(predicate.buildInputLock.recipeSha256, lock.recipeSha256, "$.payload.predicate.buildInputLock.recipeSha256");
  equal(predicate.workerBundle.bundleId, worker.bundleId, "$.payload.predicate.workerBundle.bundleId");
  equal(predicate.workerBundle.sha256, worker.sha256, "$.payload.predicate.workerBundle.sha256");
  equal(predicate.workerBundle.byteLength, worker.byteLength, "$.payload.predicate.workerBundle.byteLength");
  equal(predicate.workerBundle.factorySha256, worker.factorySha256, "$.payload.predicate.workerBundle.factorySha256");
  equal(predicate.workerBundle.factoryByteLength, worker.factoryByteLength, "$.payload.predicate.workerBundle.factoryByteLength");

  try {
    await verifyCppCutePreparedAttestationSignature({
      trustStore,
      expectedTrustStoreHash: provenancePolicy.trustStoreSha256,
      allowlistedBuilderIds: provenancePolicy.builderIds,
      builderId: predicate.builderId,
      keyId: signature.keyid,
      signatureBase64: signature.sig,
      signingBytes: cppCuteBrowserBuildProvenanceDsseSigningBytes(decoded.statement),
      ...(signal === undefined ? {} : { signal }),
    });
  } catch (cause) {
    if (cause instanceof CppCuteFrontendProvenanceError) {
      const code = cause.code === "BG-COMPILER-CPP-CUTE-PROVENANCE-CANCELLED"
        ? "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-CANCELLED"
        : cause.code === "BG-COMPILER-CPP-CUTE-PROVENANCE-POLICY-MISMATCH"
          ? "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-POLICY"
          : "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SIGNATURE";
      throw new CppCuteBrowserBuildProvenanceError(
        code,
        cause.path,
        "browser build attestation failed profile-pinned signature verification",
        { cause },
      );
    }
    throw cause;
  }
  throwIfAborted(signal);
  const statementSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-build-provenance-statement.v1",
    statement: decoded.statement,
  });
  const evidenceSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-build-provenance-evidence.v1",
    statementSha256,
    buildSubjectId: buildSubject.buildSubjectId,
    profileHash: profileRecord.profileHash,
    manifestId: assetManifest.manifestId,
    manifestSha256: assetManifest.manifestSha256,
    buildInputLockResourceSha256: lock.resourceSha256,
    workerBundleSha256: worker.sha256,
    trustStoreSha256: provenancePolicy.trustStoreSha256,
    keyId: signature.keyid,
    builderId: predicate.builderId,
  });
  throwIfAborted(signal);
  const verified = Object.freeze({
    buildSubjectId: buildSubject.buildSubjectId,
    buildSubjectSha256: buildSubject.buildSubjectSha256,
    statementSha256,
    evidenceSha256,
    builderId: predicate.builderId,
    keyId: signature.keyid,
    trustStoreSha256: provenancePolicy.trustStoreSha256,
    profileHash: profileRecord.profileHash,
    manifestId: assetManifest.manifestId,
    manifestSha256: assetManifest.manifestSha256,
    manifestByteLength: assetManifest.manifestByteLength,
    assetSetSha256: assetManifest.assetSetSha256,
    buildInputLockResourceSha256: lock.resourceSha256,
    workerBundleSha256: worker.sha256,
    signatureVerified: true,
    manifestSignaturePolicyMatched: true,
    producerTrusted: false,
    buildSubjectBound: true,
    exactAssetBytesVerified: false,
    fullDistributedOutputSetReproducible: false,
    licenseReviewComplete: false,
    distributionAuthorized: false,
    workerExecutionObserved: false,
    releaseReady: false,
  }) as VerifiedCppCuteBrowserBuildSignatureBinding;
  VERIFIED_BUILD_SIGNATURE_BINDINGS.set(verified, Object.freeze({
    envelope: decoded.envelope,
    statement: decoded.statement,
    buildSubject,
    profile: manifestRecord.profile,
    assetManifest,
    buildInputLock,
    workerBundle,
    trustStore,
  }));
  return verified;
}

export function unwrapVerifiedCppCuteBrowserBuildSignatureBinding(
  verified: VerifiedCppCuteBrowserBuildSignatureBinding,
): VerifiedCppCuteBrowserBuildSignatureBindingRecord {
  if (typeof verified !== "object" || verified === null) unverified();
  const record = VERIFIED_BUILD_SIGNATURE_BINDINGS.get(verified as object);
  if (record === undefined) unverified();
  return record;
}

function equal(actual: string | number, expected: string | number, path: string): void {
  if (actual !== expected) binding(path, "provenance field differs from the exact opaque input authority");
}

function normalizeSigningRequestInput(
  input: CreateCppCuteBrowserBuildProvenanceSigningRequestInput,
): CreateCppCuteBrowserBuildProvenanceSigningRequestInput {
  const descriptors = inspectPlainRecord(
    input,
    [
      "assetManifest",
      "buildInputLock",
      "workerBundle",
      "trustPolicy",
      "trustStore",
      "builderId",
      "keyId",
    ],
    "$.input",
  );
  const builderId = dataProperty(descriptors, "builderId", "$.input");
  const keyId = dataProperty(descriptors, "keyId", "$.input");
  if (typeof builderId !== "string" || builderId.length === 0) {
    binding("$.input.builderId", "builderId must be one nonempty string");
  }
  if (typeof keyId !== "string" || keyId.length === 0) {
    binding("$.input.keyId", "keyId must be one nonempty string");
  }
  return {
    assetManifest: dataProperty(
      descriptors,
      "assetManifest",
      "$.input",
    ) as PreparedCppCuteBrowserAssetManifest,
    buildInputLock: dataProperty(
      descriptors,
      "buildInputLock",
      "$.input",
    ) as PreparedCppCuteBrowserBuildInputLock,
    workerBundle: dataProperty(
      descriptors,
      "workerBundle",
      "$.input",
    ) as VerifiedCppCuteBrowserWorkerBundle,
    trustPolicy: dataProperty(
      descriptors,
      "trustPolicy",
      "$.input",
    ) as AdmittedCppCuteBrowserProducerTrustPolicy,
    trustStore: dataProperty(
      descriptors,
      "trustStore",
      "$.input",
    ) as PreparedCppCuteAttestationTrustStore,
    builderId,
    keyId,
  };
}

function normalizeSigningRequestOptions(
  options: CreateCppCuteBrowserBuildProvenanceSigningRequestOptions,
): AbortSignal | undefined {
  const descriptors = inspectPlainRecord(
    options,
    ["signal"],
    "$.options",
    false,
  );
  if (descriptors.signal === undefined) return undefined;
  const signal = dataProperty(descriptors, "signal", "$.options");
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined ||
      typeof signal !== "object" || signal === null) {
    binding("$.options.signal", "signal must be one native AbortSignal");
  }
  try {
    NATIVE_REFLECT_APPLY(ABORT_SIGNAL_ABORTED_GETTER, signal, []);
  } catch {
    binding("$.options.signal", "signal must be one native AbortSignal");
  }
  return signal as AbortSignal;
}

function inspectPlainRecord(
  value: unknown,
  allowedKeys: readonly string[],
  path: string,
  requireAll = true,
): PropertyDescriptorMap {
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  let keys: readonly PropertyKey[];
  try {
    prototype = NATIVE_REFLECT_APPLY(
      NATIVE_OBJECT_GET_PROTOTYPE_OF,
      CAPTURED_OBJECT,
      [value],
    ) as object | null;
    descriptors = NATIVE_REFLECT_APPLY(
      NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTORS,
      CAPTURED_OBJECT,
      [value],
    ) as PropertyDescriptorMap;
    keys = NATIVE_REFLECT_APPLY(
      NATIVE_REFLECT_OWN_KEYS,
      CAPTURED_REFLECT,
      [descriptors],
    ) as readonly PropertyKey[];
  } catch (cause) {
    unverifiedAt(
      path,
      "record could not be inspected without invoking accessors",
      cause,
    );
  }
  if (prototype !== CAPTURED_OBJECT.prototype) {
    binding(path, "record must have the plain Object prototype");
  }
  if ((requireAll && keys.length !== allowedKeys.length) ||
      keys.some((key) =>
        typeof key !== "string" || !allowedKeys.includes(key))) {
    binding(path, "record fields differ from the exact closed interface");
  }
  return descriptors;
}

function dataProperty(
  descriptors: PropertyDescriptorMap,
  key: string,
  path: string,
): unknown {
  const descriptor = descriptors[key];
  if (descriptor === undefined || !("value" in descriptor) ||
      descriptor.enumerable !== true) {
    binding(`${path}.${key}`, "field must be an enumerable plain data property");
  }
  return descriptor.value;
}

function sameStrings(
  left: readonly string[],
  right: readonly string[],
): boolean {
  return left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal !== undefined &&
      (ABORT_SIGNAL_ABORTED_GETTER === undefined ||
       NATIVE_REFLECT_APPLY(ABORT_SIGNAL_ABORTED_GETTER, signal, []) === true)) {
    throw new CppCuteBrowserBuildProvenanceError(
      "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-CANCELLED",
      "$.signal",
      "browser build provenance verification was cancelled",
    );
  }
}

function binding(path: string, message: string): never {
  throw new CppCuteBrowserBuildProvenanceError(
    "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-BINDING",
    path,
    message,
  );
}

function policyMismatch(path: string, message: string): never {
  throw new CppCuteBrowserBuildProvenanceError(
    "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-POLICY",
    path,
    message,
  );
}

function unverifiedAt(path: string, message: string, cause?: unknown): never {
  throw new CppCuteBrowserBuildProvenanceError(
    "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-UNVERIFIED",
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}

function unverified(): never {
  throw new CppCuteBrowserBuildProvenanceError(
    "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-UNVERIFIED",
    "$",
    "browser build provenance must come from the verified opaque authority",
  );
}
