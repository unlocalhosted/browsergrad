import { canonicalJsonBytes } from "./canonical-json.js";
import { SCHEMA_DIAGNOSTIC_CODES, schemaError } from "./diagnostics.js";
import { unwrapVerifiedArtifact, type VerifiedArtifact } from "./envelope.js";
import type { DecodeLimits } from "./limits.js";
import type { JsonObject, JsonValue } from "./json.js";

export const SEMANTIC_ARTIFACT_HASH_DOMAIN = "browsergrad.semantic-artifact.v1";
export const CACHE_KEY_HASH_DOMAIN = "browsergrad.cache-key.v1";

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (subtle === undefined) {
    throw schemaError(
      SCHEMA_DIAGNOSTIC_CODES.hashUnavailable,
      "SHA-256 requires Web Crypto SubtleCrypto in this environment",
      { path: "$" },
    );
  }
  const input = new Uint8Array(bytes);
  const digest = await subtle.digest("SHA-256", input);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

export async function hashCanonicalJson(
  value: unknown,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  return sha256Hex(canonicalJsonBytes(value, options));
}

export async function hashSemanticArtifact(
  artifact: VerifiedArtifact<JsonValue>,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  const envelope = unwrapVerifiedArtifact(artifact);
  const projection: JsonObject = {
    domain: SEMANTIC_ARTIFACT_HASH_DOMAIN,
    schema: envelope.schema,
    version: envelope.version,
    requiredExtensions: [...envelope.requiredExtensions].sort(),
    payload: envelope.payload,
  };
  return hashCanonicalJson(projection, options);
}

export async function hashNamedComponents(
  components: Readonly<Record<string, JsonValue>>,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  const names = Object.keys(components);
  if (names.length === 0 || names.some((name) => name.length === 0)) {
    throw schemaError(
      SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue,
      "cache keys require at least one non-empty named component",
      { path: "$.components" },
    );
  }
  const projection: JsonObject = {
    domain: CACHE_KEY_HASH_DOMAIN,
    components,
  };
  return hashCanonicalJson(projection, options);
}

export async function derivePureValueId(
  kind: string,
  value: JsonValue,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  validateIdKind(kind);
  const digest = await hashCanonicalJson({
    domain: `browsergrad.pure-value.${kind}.v1`,
    value,
  }, options);
  return `bg.pure.${kind}.sha256.${digest}`;
}

export async function deriveScopedEntityId(
  artifactScope: string,
  kind: string,
  canonicalPosition: string,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  validateIdKind(kind);
  if (artifactScope.length === 0 || canonicalPosition.length === 0) {
    throw schemaError(
      SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue,
      "scoped entity IDs require non-empty artifact scope and canonical position",
      { path: "$" },
    );
  }
  const digest = await hashCanonicalJson({
    domain: `browsergrad.entity-id.${kind}.v1`,
    artifactScope,
    canonicalPosition,
  }, options);
  return `bg.entity.${kind}.sha256.${digest}`;
}

function validateIdKind(kind: string): void {
  if (!/^[a-z][a-z0-9-]*$/u.test(kind)) {
    throw schemaError(
      SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue,
      "ID kind must match ^[a-z][a-z0-9-]*$",
      { path: "$.kind" },
    );
  }
}
