export const SCHEMA_DIAGNOSTIC_CODES = {
  invalidJson: "BG-SCHEMA-INVALID-JSON",
  duplicateKey: "BG-SCHEMA-DUPLICATE-KEY",
  resourceLimit: "BG-SCHEMA-RESOURCE-LIMIT",
  unsafeNumber: "BG-SCHEMA-UNSAFE-NUMBER",
  nonCanonicalInteger: "BG-SCHEMA-NONCANONICAL-INTEGER",
  integerRange: "BG-SCHEMA-INTEGER-RANGE",
  invalidEnvelope: "BG-SCHEMA-INVALID-ENVELOPE",
  unsupportedVersion: "BG-SCHEMA-UNSUPPORTED-VERSION",
  unknownRequiredExtension: "BG-SCHEMA-UNKNOWN-REQUIRED-EXTENSION",
  unverifiedArtifact: "BG-SCHEMA-UNVERIFIED-ARTIFACT",
  nonCanonicalValue: "BG-SCHEMA-NONCANONICAL-VALUE",
  hashUnavailable: "BG-SCHEMA-HASH-UNAVAILABLE",
} as const;

export type SchemaDiagnosticCode =
  (typeof SCHEMA_DIAGNOSTIC_CODES)[keyof typeof SCHEMA_DIAGNOSTIC_CODES];

export interface SemanticDiagnostic {
  readonly code: SchemaDiagnosticCode | `BG-LAYOUT-${string}`;
  readonly stage: "verification";
  readonly severity: "error" | "warning" | "note";
  readonly message: string;
  readonly path?: string;
  readonly offset?: number;
  readonly remediation?: string;
}

export type ValidationResult<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly diagnostics: readonly SemanticDiagnostic[] };

export class SemanticSchemaError extends Error {
  readonly diagnostic: SemanticDiagnostic;

  constructor(diagnostic: SemanticDiagnostic) {
    super(`${diagnostic.code}: ${diagnostic.message}`);
    this.name = "SemanticSchemaError";
    this.diagnostic = diagnostic;
  }
}

export function schemaError(
  code: SchemaDiagnosticCode,
  message: string,
  details: { readonly path?: string; readonly offset?: number; readonly remediation?: string } = {},
): SemanticSchemaError {
  return new SemanticSchemaError({
    code,
    stage: "verification",
    severity: "error",
    message,
    ...(details.path === undefined ? {} : { path: details.path }),
    ...(details.offset === undefined ? {} : { offset: details.offset }),
    ...(details.remediation === undefined ? {} : { remediation: details.remediation }),
  });
}
