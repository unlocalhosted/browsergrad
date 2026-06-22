import type { SourceSpan } from "./types.js";

export type TokenKind = "identifier" | "number" | "string" | "punctuator" | "eof";

export interface Token {
  readonly kind: TokenKind;
  readonly value: string;
  readonly span: SourceSpan;
}

export function tokenizeCudaLite(source: string): readonly Token[] {
  return scanCudaLiteTokens(source, collectObjectLikeDefines(source));
}

function scanCudaLiteTokens(
  source: string,
  macros: ReadonlyMap<string, string>,
  expansionStack: readonly string[] = [],
  expansionSpan?: SourceSpan,
): readonly Token[] {
  const tokens: Token[] = [];
  let index = 0;
  let line = 1;
  let column = 1;

  const span = (start: number, end: number, startLine: number, startColumn: number): SourceSpan => ({
    start,
    end,
    line: startLine,
    column: startColumn,
  });

  const tokenSpan = (start: number, end: number, startLine: number, startColumn: number): SourceSpan =>
    expansionSpan ?? span(start, end, startLine, startColumn);

  const advance = (): string => {
    const char = source[index] ?? "";
    index++;
    if (char === "\n") {
      line++;
      column = 1;
    } else {
      column++;
    }
    return char;
  };

  while (index < source.length) {
    const char = source[index]!;
    if (/\s/.test(char)) {
      advance();
      continue;
    }
    if (char === "/" && source[index + 1] === "/") {
      while (index < source.length && source[index] !== "\n") advance();
      continue;
    }
    if (char === "/" && source[index + 1] === "*") {
      advance();
      advance();
      while (index < source.length && !(source[index] === "*" && source[index + 1] === "/")) {
        advance();
      }
      if (index < source.length) {
        advance();
        advance();
      }
      continue;
    }
    if (char === "#") {
      while (index < source.length && source[index] !== "\n") advance();
      continue;
    }

    const start = index;
    const startLine = line;
    const startColumn = column;
    if (/[A-Za-z_]/.test(char)) {
      let value = "";
      while (index < source.length && /[A-Za-z0-9_]/.test(source[index]!)) {
        value += advance();
      }
      const macroValue = macros.get(value);
      if (macroValue !== undefined && !expansionStack.includes(value)) {
        const expanded = scanCudaLiteTokens(
          macroValue,
          macros,
          [...expansionStack, value],
          tokenSpan(start, index, startLine, startColumn),
        ).filter((token) => token.kind !== "eof");
        tokens.push(...expanded);
      } else {
        tokens.push({ kind: "identifier", value, span: tokenSpan(start, index, startLine, startColumn) });
      }
      continue;
    }
    if (char === "\"") {
      let value = advance();
      let escaped = false;
      while (index < source.length) {
        const next = advance();
        value += next;
        if (escaped) {
          escaped = false;
          continue;
        }
        if (next === "\\") {
          escaped = true;
          continue;
        }
        if (next === "\"") break;
      }
      tokens.push({ kind: "string", value, span: tokenSpan(start, index, startLine, startColumn) });
      continue;
    }
    if (/[0-9]/.test(char) || (char === "." && /[0-9]/.test(source[index + 1] ?? ""))) {
      let value = "";
      while (index < source.length && /[0-9.]/.test(source[index]!)) {
        value += advance();
      }
      while (index < source.length && /[A-Za-z]/.test(source[index]!)) {
        value += advance();
      }
      tokens.push({ kind: "number", value, span: tokenSpan(start, index, startLine, startColumn) });
      continue;
    }

    const three = source.slice(index, index + 3);
    const two = source.slice(index, index + 2);
    if (["<<=", ">>="].includes(three)) {
      advance();
      advance();
      advance();
      tokens.push({ kind: "punctuator", value: three, span: tokenSpan(start, index, startLine, startColumn) });
      continue;
    }
    if (["<=", ">=", "==", "!=", "&&", "||", "+=", "-=", "*=", "/=", "++", "--", "<<", ">>"].includes(two)) {
      advance();
      advance();
      tokens.push({ kind: "punctuator", value: two, span: tokenSpan(start, index, startLine, startColumn) });
      continue;
    }
    advance();
    tokens.push({ kind: "punctuator", value: char, span: tokenSpan(start, index, startLine, startColumn) });
  }

  tokens.push({
    kind: "eof",
    value: "<eof>",
    span: expansionSpan ?? { start: source.length, end: source.length, line, column },
  });
  return tokens;
}

function collectObjectLikeDefines(source: string): ReadonlyMap<string, string> {
  const macros = new Map<string, string>();
  for (const line of source.split(/\r?\n/u)) {
    const stripped = stripLineComment(line);
    const match = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)(?!\()\s+(.+?)\s*$/u.exec(stripped);
    if (match === null) continue;
    const [, name, replacement] = match;
    if (name === undefined || replacement === undefined) continue;
    const trimmed = replacement.trim();
    if (trimmed.length > 0) macros.set(name, trimmed);
  }
  return macros;
}

function stripLineComment(line: string): string {
  let escaped = false;
  let inString = false;
  for (let index = 0; index < line.length - 1; index++) {
    const char = line[index]!;
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }
    if (char === "\"") {
      inString = true;
      continue;
    }
    if (char === "/" && line[index + 1] === "/") return line.slice(0, index);
  }
  return line;
}
