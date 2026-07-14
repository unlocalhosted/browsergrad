import {
  CudaLiteCompilerError,
  type CudaLiteDiagnostic,
} from "./types.js";

export function formatCudaLiteDiagnostics(
  source: string,
  diagnostics: readonly CudaLiteDiagnostic[],
): string {
  const lines = source.split(/\r?\n/);
  return diagnostics
    .map((diagnostic) => {
      const sourceLine = lines[diagnostic.span.line - 1] ?? "";
      const caretColumn = Math.max(diagnostic.span.column, 1);
      const width = Math.max(diagnostic.span.end - diagnostic.span.start, 1);
      const caret = `${" ".repeat(caretColumn - 1)}${"^".repeat(Math.min(width, Math.max(sourceLine.length - caretColumn + 1, 1)))}`;
      return [
        `${diagnostic.severity.toUpperCase()} ${diagnostic.code} ${diagnostic.span.line}:${diagnostic.span.column} ${diagnostic.message}`,
        sourceLine,
        caret,
      ].join("\n");
    })
    .join("\n\n");
}

/**
 * Creates the user-facing compiler error at a phase boundary. Keeping source
 * formatting here means parser, analyzer, and semantic-pass failures all show
 * the same location context without changing diagnostic codes or spans.
 */
export function createCudaLiteCompilerError(
  message: string,
  diagnostics: readonly CudaLiteDiagnostic[],
  source?: string,
): CudaLiteCompilerError {
  const formatted = source === undefined || diagnostics.length === 0
    ? message
    : `${message}\n${formatCudaLiteDiagnostics(source, diagnostics)}`;
  return new CudaLiteCompilerError(formatted, diagnostics, source);
}

/**
 * Adds source context to a compiler error produced by a lower compiler phase.
 * Errors that already carry their source are returned unchanged so their
 * diagnostic block is never formatted twice.
 */
export function withCudaLiteDiagnosticSource(
  error: CudaLiteCompilerError,
  source: string,
): CudaLiteCompilerError {
  if (error.source !== undefined) return error;
  return createCudaLiteCompilerError(error.message, error.diagnostics, source);
}
