import type { SourceSpan } from "./types.js";

export type TokenKind = "identifier" | "number" | "punctuator" | "eof";

export interface Token {
  readonly kind: TokenKind;
  readonly value: string;
  readonly span: SourceSpan;
}

export function tokenizeCudaLite(source: string): readonly Token[] {
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

    const start = index;
    const startLine = line;
    const startColumn = column;
    if (/[A-Za-z_]/.test(char)) {
      let value = "";
      while (index < source.length && /[A-Za-z0-9_]/.test(source[index]!)) {
        value += advance();
      }
      tokens.push({ kind: "identifier", value, span: span(start, index, startLine, startColumn) });
      continue;
    }
    if (/[0-9]/.test(char) || (char === "." && /[0-9]/.test(source[index + 1] ?? ""))) {
      let value = "";
      while (index < source.length && /[0-9.]/.test(source[index]!)) {
        value += advance();
      }
      tokens.push({ kind: "number", value, span: span(start, index, startLine, startColumn) });
      continue;
    }

    const three = source.slice(index, index + 3);
    const two = source.slice(index, index + 2);
    if (["<<=", ">>="].includes(three)) {
      advance();
      advance();
      advance();
      tokens.push({ kind: "punctuator", value: three, span: span(start, index, startLine, startColumn) });
      continue;
    }
    if (["<=", ">=", "==", "!=", "&&", "||", "+=", "-=", "*=", "/=", "++", "--"].includes(two)) {
      advance();
      advance();
      tokens.push({ kind: "punctuator", value: two, span: span(start, index, startLine, startColumn) });
      continue;
    }
    advance();
    tokens.push({ kind: "punctuator", value: char, span: span(start, index, startLine, startColumn) });
  }

  tokens.push({
    kind: "eof",
    value: "<eof>",
    span: { start: source.length, end: source.length, line, column },
  });
  return tokens;
}
