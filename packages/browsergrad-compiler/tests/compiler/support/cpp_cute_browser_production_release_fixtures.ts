import {
  canonicalJsonBytes,
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  createCppCuteBrowserDistributionApprovalSigningRequest,
  verifyCppCuteBrowserDistributionApproval,
  type CppCuteBrowserDistributionApprovalEnvelopeV1,
  type VerifiedCppCuteBrowserDistributionApproval,
} from "../../../src/cpp_cute_browser_distribution_approval.js";
import {
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MAJOR,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MINOR,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_SCHEMA,
  CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE,
  admitCppCuteBrowserDistributionApprovalPolicy,
  deriveCppCuteBrowserDistributionApprovalPolicyId,
  type CppCuteBrowserDistributionApprovalPolicyProjectionV1,
  type CppCuteBrowserDistributionApprovalPolicyV1,
} from "../../../src/cpp_cute_browser_distribution_approval_policy.js";
import {
  CPP_CUTE_ATTESTATION_TRUST_STORE_MAJOR,
  CPP_CUTE_ATTESTATION_TRUST_STORE_MINOR,
  CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA,
  prepareCppCuteAttestationTrustStore,
} from "../../../src/cpp_cute_frontend_provenance.js";

const REVIEWER_ID =
  "https://reviewers.browsergrad.dev/production-composition-test";

export async function createExternallyRootedDistributionApproval():
Promise<VerifiedCppCuteBrowserDistributionApproval> {
  const keyPair = await crypto.subtle.generateKey(
    { name: "ECDSA", namedCurve: "P-256" },
    true,
    ["sign", "verify"],
  );
  const spki = new Uint8Array(
    await crypto.subtle.exportKey("spki", keyPair.publicKey),
  );
  const keyId = `sha256:${await sha256Hex(spki)}`;
  const trustStore = await prepareCppCuteAttestationTrustStore({
    schema: CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA,
    version: {
      major: CPP_CUTE_ATTESTATION_TRUST_STORE_MAJOR,
      minor: CPP_CUTE_ATTESTATION_TRUST_STORE_MINOR,
    },
    keys: [{
      keyId,
      builderId: REVIEWER_ID,
      algorithm: "ecdsa-p256-sha256",
      spkiDerBase64: encodeBase64(spki),
    }],
  });
  const projection: CppCuteBrowserDistributionApprovalPolicyProjectionV1 = {
    schema: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MAJOR,
      minor: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MINOR,
    },
    predicateType:
      CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE,
    trustStoreSha256: trustStore.trustStoreHash,
    reviewerIds: [REVIEWER_ID],
    keyIds: [keyId],
  };
  const policyDocument: CppCuteBrowserDistributionApprovalPolicyV1 = {
    ...projection,
    policyId:
      await deriveCppCuteBrowserDistributionApprovalPolicyId(projection),
  };
  const policy = await admitCppCuteBrowserDistributionApprovalPolicy(
    canonicalJsonBytes(policyDocument),
  );
  const request =
    await createCppCuteBrowserDistributionApprovalSigningRequest(
      policy,
      REVIEWER_ID,
    );
  const signature = new Uint8Array(await crypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    keyPair.privateKey,
    copyToArrayBuffer(request.signingBytes),
  ));
  if (signature.byteLength !== 64) {
    throw new Error("test runtime did not emit P-256 P1363 bytes");
  }
  const envelope: CppCuteBrowserDistributionApprovalEnvelopeV1 = {
    payloadType: request.payloadType,
    payload: request.payload,
    signatures: [{ keyid: keyId, sig: encodeBase64(signature) }],
  };
  return await verifyCppCuteBrowserDistributionApproval(
    envelope,
    policy,
    trustStore,
  );
}

function encodeBase64(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes));
}

function copyToArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return Uint8Array.from(bytes).buffer;
}
