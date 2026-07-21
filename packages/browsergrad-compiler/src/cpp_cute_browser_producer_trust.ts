import {
  hashCanonicalJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapPreparedCppCuteBrowserAssetManifest,
} from "./cpp_cute_browser_assets.js";
import {
  unwrapVerifiedCppCuteBrowserBuildSignatureBinding,
  type VerifiedCppCuteBrowserBuildSignatureBinding,
} from "./cpp_cute_browser_build_provenance.js";
import {
  unwrapAdmittedCppCuteBrowserProducerTrustPolicy,
  type AdmittedCppCuteBrowserProducerTrustPolicy,
} from "./cpp_cute_browser_producer_trust_policy.js";

const VERIFIED_PRODUCERS = new WeakMap<object, VerifiedCppCuteBrowserBuildProducerRecord>();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

declare const verifiedProducerBrand: unique symbol;

/**
 * Producer trust relative to one explicit host-admitted policy. This is not
 * asset installation, legal approval, Worker execution, lowering, backend, or
 * release authority.
 */
export interface VerifiedCppCuteBrowserBuildProducer {
  readonly [verifiedProducerBrand]: true;
  readonly authority: "independently-admitted-browser-build-producer";
  readonly producerEvidenceId: string;
  readonly policyId: string;
  readonly policySha256: string;
  readonly policyVersion: "1.0";
  readonly buildSubjectId: string;
  readonly buildSubjectSha256: string;
  readonly statementSha256: string;
  readonly signatureEvidenceSha256: string;
  readonly predicateType: string;
  readonly builderId: string;
  readonly keyId: string;
  readonly trustStoreSha256: string;
  readonly profileHash: string;
  readonly manifestId: string;
  readonly assetSetSha256: string;
  readonly buildInputLockResourceSha256: string;
  readonly workerBundleSha256: string;
  readonly signatureVerified: true;
  readonly manifestSignaturePolicyMatched: true;
  readonly independentTrustPolicyMatched: true;
  readonly producerTrusted: true;
  readonly buildSubjectBound: true;
  readonly exactAssetBytesVerified: false;
  readonly fullDistributedOutputSetReproducible: false;
  readonly licenseReviewComplete: false;
  readonly distributionAuthorized: false;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
  readonly backendExecutionObserved: false;
  readonly releaseReady: false;
}

export interface VerifiedCppCuteBrowserBuildProducerRecord {
  readonly signatureBinding: VerifiedCppCuteBrowserBuildSignatureBinding;
  readonly trustPolicy: AdmittedCppCuteBrowserProducerTrustPolicy;
}

export interface VerifyCppCuteBrowserBuildProducerOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserProducerTrustErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-BINDING"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-UNVERIFIED";

export class CppCuteBrowserProducerTrustError extends Error {
  constructor(
    readonly code: CppCuteBrowserProducerTrustErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserProducerTrustError";
  }
}

export async function verifyCppCuteBrowserBuildProducer(
  signatureBinding: VerifiedCppCuteBrowserBuildSignatureBinding,
  trustPolicy: AdmittedCppCuteBrowserProducerTrustPolicy,
  options: VerifyCppCuteBrowserBuildProducerOptions = {},
): Promise<VerifiedCppCuteBrowserBuildProducer> {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  let signatureRecord: ReturnType<
    typeof unwrapVerifiedCppCuteBrowserBuildSignatureBinding
  >;
  let policyRecord: ReturnType<
    typeof unwrapAdmittedCppCuteBrowserProducerTrustPolicy
  >;
  try {
    signatureRecord = unwrapVerifiedCppCuteBrowserBuildSignatureBinding(signatureBinding);
  } catch (cause) {
    unverified("$.signatureBinding", "signature binding is not an opaque verified authority", cause);
  }
  try {
    policyRecord = unwrapAdmittedCppCuteBrowserProducerTrustPolicy(trustPolicy);
  } catch (cause) {
    unverified("$.trustPolicy", "trust policy is not an opaque host admission authority", cause);
  }
  const manifest = unwrapPreparedCppCuteBrowserAssetManifest(
    signatureRecord.assetManifest,
  ).manifest;
  const manifestPolicy = manifest.body.buildProvenancePolicy;
  const statement = signatureRecord.statement;
  const predicate = statement.predicate;
  const signature = signatureRecord.envelope.signatures[0];
  const policy = policyRecord.policy;

  if (signatureBinding.signatureVerified !== true ||
      signatureBinding.manifestSignaturePolicyMatched !== true ||
      signatureBinding.producerTrusted !== false ||
      signatureBinding.buildSubjectBound !== true ||
      signatureBinding.buildSubjectId !== signatureRecord.buildSubject.buildSubjectId ||
      signatureBinding.buildSubjectSha256 !== signatureRecord.buildSubject.buildSubjectSha256 ||
      signatureBinding.builderId !== predicate.builderId ||
      signatureBinding.keyId !== signature.keyid ||
      signatureBinding.trustStoreSha256 !== manifestPolicy.trustStoreSha256 ||
      signatureBinding.profileHash !== statement.predicate.profile.profileHash ||
      signatureBinding.manifestId !== signatureRecord.assetManifest.manifestId ||
      signatureBinding.manifestSha256 !== statement.predicate.assetManifest.manifestSha256 ||
      signatureBinding.assetSetSha256 !== signatureRecord.assetManifest.assetSetSha256 ||
      signatureBinding.buildInputLockResourceSha256 !==
        signatureRecord.buildInputLock.resourceSha256 ||
      signatureBinding.workerBundleSha256 !== statement.predicate.workerBundle.sha256) {
    binding(
      "$.signatureBinding",
      "signature binding fields differ from their retained opaque build authorities",
    );
  }
  if (trustPolicy.hostOnly !== true || trustPolicy.workerTransferable !== false ||
      trustPolicy.producerTrusted !== false || trustPolicy.releaseReady !== false ||
      trustPolicy.policyId !== policy.policyId ||
      trustPolicy.predicateType !== policy.predicateType ||
      trustPolicy.trustStoreSha256 !== policy.trustStoreSha256) {
    binding(
      "$.trustPolicy",
      "host trust policy projection differs from its retained canonical authority",
    );
  }
  if (policy.predicateType !== statement.predicateType ||
      policy.predicateType !== manifestPolicy.predicateType) {
    policyMismatch(
      "$.trustPolicy.predicateType",
      "host policy does not admit the authenticated build predicate",
    );
  }
  if (policy.trustStoreSha256 !== signatureBinding.trustStoreSha256) {
    policyMismatch(
      "$.trustPolicy.trustStoreSha256",
      "host policy does not admit the authenticated trust store",
    );
  }
  if (!policy.builderIds.includes(predicate.builderId)) {
    policyMismatch(
      "$.trustPolicy.builderIds",
      "host policy does not admit the authenticated builder",
    );
  }
  if (!policy.keyIds.includes(signature.keyid)) {
    policyMismatch(
      "$.trustPolicy.keyIds",
      "host policy does not admit the authenticated signing key",
    );
  }
  if (!manifestPolicy.builderIds.includes(predicate.builderId)) {
    binding(
      "$.signatureBinding.builderId",
      "authenticated builder is no longer admitted by the retained manifest policy",
    );
  }
  throwIfAborted(signal);
  const evidenceHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-build-producer-trust.v1",
    policyId: trustPolicy.policyId,
    policySha256: trustPolicy.policySha256,
    buildSubjectId: signatureBinding.buildSubjectId,
    buildSubjectSha256: signatureBinding.buildSubjectSha256,
    statementSha256: signatureBinding.statementSha256,
    signatureEvidenceSha256: signatureBinding.evidenceSha256,
    predicateType: policy.predicateType,
    builderId: signatureBinding.builderId,
    keyId: signatureBinding.keyId,
    trustStoreSha256: signatureBinding.trustStoreSha256,
    profileHash: signatureBinding.profileHash,
    manifestId: signatureBinding.manifestId,
    assetSetSha256: signatureBinding.assetSetSha256,
    buildInputLockResourceSha256: signatureBinding.buildInputLockResourceSha256,
    workerBundleSha256: signatureBinding.workerBundleSha256,
  });
  throwIfAborted(signal);
  const verified = Object.freeze({
    authority: "independently-admitted-browser-build-producer",
    producerEvidenceId: `bg.cpp.browser-build-producer.sha256.${evidenceHash}`,
    policyId: trustPolicy.policyId,
    policySha256: trustPolicy.policySha256,
    policyVersion: "1.0",
    buildSubjectId: signatureBinding.buildSubjectId,
    buildSubjectSha256: signatureBinding.buildSubjectSha256,
    statementSha256: signatureBinding.statementSha256,
    signatureEvidenceSha256: signatureBinding.evidenceSha256,
    predicateType: policy.predicateType,
    builderId: signatureBinding.builderId,
    keyId: signatureBinding.keyId,
    trustStoreSha256: signatureBinding.trustStoreSha256,
    profileHash: signatureBinding.profileHash,
    manifestId: signatureBinding.manifestId,
    assetSetSha256: signatureBinding.assetSetSha256,
    buildInputLockResourceSha256: signatureBinding.buildInputLockResourceSha256,
    workerBundleSha256: signatureBinding.workerBundleSha256,
    signatureVerified: true,
    manifestSignaturePolicyMatched: true,
    independentTrustPolicyMatched: true,
    producerTrusted: true,
    buildSubjectBound: true,
    exactAssetBytesVerified: false,
    fullDistributedOutputSetReproducible: false,
    licenseReviewComplete: false,
    distributionAuthorized: false,
    workerExecutionObserved: false,
    loweringAuthorityMinted: false,
    backendExecutionObserved: false,
    releaseReady: false,
  }) as VerifiedCppCuteBrowserBuildProducer;
  VERIFIED_PRODUCERS.set(verified, Object.freeze({ signatureBinding, trustPolicy }));
  return verified;
}

export function unwrapVerifiedCppCuteBrowserBuildProducer(
  verified: VerifiedCppCuteBrowserBuildProducer,
): VerifiedCppCuteBrowserBuildProducerRecord {
  if (typeof verified !== "object" || verified === null) {
    unverified("$", "producer trust must come from the opaque verifier authority");
  }
  const record = VERIFIED_PRODUCERS.get(verified as object);
  if (record === undefined) {
    unverified("$", "producer trust must come from the opaque verifier authority");
  }
  return record;
}

function normalizeOptions(options: VerifyCppCuteBrowserBuildProducerOptions): AbortSignal | undefined {
  try {
    if (typeof options !== "object" || options === null ||
        Object.getPrototypeOf(options) !== Object.prototype) {
      binding("$.options", "options must be a plain data record");
    }
    const descriptors = Object.getOwnPropertyDescriptors(options);
    const keys = Reflect.ownKeys(descriptors);
    if (keys.some((key) => typeof key !== "string" || key !== "signal")) {
      binding("$.options", "options contains unknown fields");
    }
    const signalDescriptor = descriptors.signal;
    if (signalDescriptor === undefined) return undefined;
    if (!("value" in signalDescriptor) || signalDescriptor.enumerable !== true ||
        typeof AbortSignal === "undefined" ||
        signalDescriptor.value instanceof AbortSignal === false) {
      binding("$.options.signal", "signal must be an enumerable AbortSignal data property");
    }
    return signalDescriptor.value as AbortSignal;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserProducerTrustError) throw cause;
    binding("$.options", "options could not be inspected as a plain data record");
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal === undefined) return;
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined ||
      Reflect.apply(ABORT_SIGNAL_ABORTED_GETTER, signal, []) === true) {
    throw new CppCuteBrowserProducerTrustError(
      "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-CANCELLED",
      "$.options.signal",
      "browser build producer trust verification was cancelled",
    );
  }
}

function binding(path: string, message: string): never {
  throw new CppCuteBrowserProducerTrustError(
    "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-BINDING",
    path,
    message,
  );
}

function policyMismatch(path: string, message: string): never {
  throw new CppCuteBrowserProducerTrustError(
    "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY",
    path,
    message,
  );
}

function unverified(path: string, message: string, cause?: unknown): never {
  throw new CppCuteBrowserProducerTrustError(
    "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-UNVERIFIED",
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}
