import type { CudaLiteDiagnostic } from "./types.js";

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
