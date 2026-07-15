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

export const LAYOUT_DIAGNOSTIC_CODES = {
  invalidDimExpr: "BG-LAYOUT-INVALID-DIM-EXPR",
  invalidIndexExpr: "BG-LAYOUT-INVALID-INDEX-EXPR",
  invalidLayoutExpr: "BG-LAYOUT-INVALID-LAYOUT-EXPR",
  resourceLimit: "BG-LAYOUT-RESOURCE-LIMIT",
  nonpositiveDivisor: "BG-LAYOUT-NONPOSITIVE-DIVISOR",
  undeclaredSymbol: "BG-LAYOUT-UNDECLARED-SYMBOL",
  duplicateSymbol: "BG-LAYOUT-DUPLICATE-SYMBOL",
  invalidSymbolDomain: "BG-LAYOUT-INVALID-SYMBOL-DOMAIN",
  symbolDomain: "BG-LAYOUT-SYMBOL-DOMAIN",
  invalidBindings: "BG-LAYOUT-INVALID-BINDINGS",
  undeclaredBinding: "BG-LAYOUT-UNDECLARED-BINDING",
  unknownDType: "BG-LAYOUT-UNKNOWN-DTYPE",
  invalidArtifact: "BG-LAYOUT-INVALID-ARTIFACT",
  unknownField: "BG-LAYOUT-UNKNOWN-FIELD",
  duplicateId: "BG-LAYOUT-DUPLICATE-ID",
  danglingReference: "BG-LAYOUT-DANGLING-REFERENCE",
  rankMismatch: "BG-LAYOUT-RANK-MISMATCH",
  invalidAlignment: "BG-LAYOUT-INVALID-ALIGNMENT",
  fieldRange: "BG-LAYOUT-FIELD-RANGE",
  constraintViolation: "BG-LAYOUT-CONSTRAINT-VIOLATION",
  unresolvedSymbol: "BG-LAYOUT-UNRESOLVED-SYMBOL",
  invalidCoordinate: "BG-LAYOUT-INVALID-COORDINATE",
} as const;

export type LayoutDiagnosticCode =
  (typeof LAYOUT_DIAGNOSTIC_CODES)[keyof typeof LAYOUT_DIAGNOSTIC_CODES];

export interface SemanticDiagnostic {
  readonly code: SchemaDiagnosticCode | LayoutDiagnosticCode | `BG-LAYOUT-${string}`;
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
