import { tokenizeCudaLite, type Token } from "./lexer.js";
import {
  CudaLiteCompilerError,
  type CudaLiteAssignmentExpression,
  type CudaLiteBinaryExpression,
  type CudaLiteExpression,
  type CudaLiteForStatement,
  type CudaLiteIfStatement,
  type CudaLiteKernel,
  type CudaLiteMemberExpression,
  type CudaLiteModule,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type CudaLiteStatement,
  type CudaLiteUnaryExpression,
  type CudaLiteUpdateExpression,
  type CudaLiteVarDecl,
  type SourceSpan,
} from "./types.js";

const TYPE_KEYWORDS = new Set(["float", "int", "uint", "half"]);
const ASSIGNMENT = new Set(["=", "+=", "-=", "*=", "/="]);
const BINARY_PRECEDENCE = new Map<string, number>([
  ["||", 2],
  ["&&", 3],
  ["==", 4],
  ["!=", 4],
  ["<", 5],
  ["<=", 5],
  [">", 5],
  [">=", 5],
  ["+", 6],
  ["-", 6],
  ["*", 7],
  ["/", 7],
  ["%", 7],
]);

export function parseCudaLite(source: string): CudaLiteModule {
  return new Parser(source).parseModule();
}

class Parser {
  private readonly source: string;
  private readonly tokens: readonly Token[];
  private index = 0;

  constructor(source: string) {
    this.source = source;
    this.tokens = tokenizeCudaLite(source);
  }

  parseModule(): CudaLiteModule {
    const kernels: CudaLiteKernel[] = [];
    while (!this.match("<eof>")) {
      kernels.push(this.parseKernel());
    }
    return {
      kind: "module",
      source: this.source,
      kernels,
      span: { start: 0, end: this.source.length, line: 1, column: 1 },
    };
  }

  private parseKernel(): CudaLiteKernel {
    const start = this.expect("__global__").span;
    this.expect("void");
    const name = this.expectIdentifier("kernel name");
    this.expect("(");
    const params = this.parseParams();
    this.expect(")");
    const body = this.parseBlock();
    return {
      kind: "kernel",
      name: name.value,
      params,
      body,
      span: mergeSpans(start, body.at(-1)?.span ?? name.span),
    };
  }

  private parseParams(): readonly CudaLiteParam[] {
    const params: CudaLiteParam[] = [];
    if (this.match(")")) return params;
    do {
      const startToken = this.peek();
      const constant = this.consumeIf("const") !== undefined;
      const type = this.parseType();
      const pointer = this.consumeIf("*") !== undefined;
      const name = this.expectIdentifier("parameter name");
      params.push({
        name: name.value,
        valueType: type,
        pointer,
        constant,
        span: mergeSpans(startToken.span, name.span),
      });
    } while (this.consumeIf(","));
    return params;
  }

  private parseBlock(): readonly CudaLiteStatement[] {
    this.expect("{");
    const statements: CudaLiteStatement[] = [];
    while (!this.match("}")) statements.push(this.parseStatement());
    this.expect("}");
    return statements;
  }

  private parseStatement(): CudaLiteStatement {
    if (this.match("{")) {
      const block = this.parseBlock();
      return {
        kind: "expr",
        expression: { kind: "number", value: 0, raw: "0", span: block[0]?.span ?? this.previous().span },
        span: block[0]?.span ?? this.previous().span,
      };
    }
    if (this.match("if")) return this.parseIf();
    if (this.match("for")) return this.parseFor();
    if (this.startsVarDecl()) return this.parseVarDecl(true);
    const expression = this.parseExpression();
    const end = this.expect(";");
    return { kind: "expr", expression, span: mergeSpans(expression.span, end.span) };
  }

  private parseIf(): CudaLiteIfStatement {
    const start = this.expect("if").span;
    this.expect("(");
    const condition = this.parseExpression();
    this.expect(")");
    const consequent = this.parseBodyAsStatements();
    const alternate = this.consumeIf("else") ? this.parseBodyAsStatements() : undefined;
    return {
      kind: "if",
      condition,
      consequent,
      ...(alternate === undefined ? {} : { alternate }),
      span: mergeSpans(start, (alternate ?? consequent).at(-1)?.span ?? condition.span),
    };
  }

  private parseFor(): CudaLiteForStatement {
    const start = this.expect("for").span;
    this.expect("(");
    let init: CudaLiteVarDecl | CudaLiteExpression | undefined;
    if (!this.match(";")) {
      init = this.startsVarDecl() ? this.parseVarDecl(false) : this.parseExpression();
    }
    this.expect(";");
    const condition = this.match(";") ? undefined : this.parseExpression();
    this.expect(";");
    const update = this.match(")") ? undefined : this.parseExpression();
    this.expect(")");
    const body = this.parseBodyAsStatements();
    return {
      kind: "for",
      ...(init === undefined ? {} : { init }),
      ...(condition === undefined ? {} : { condition }),
      ...(update === undefined ? {} : { update }),
      body,
      span: mergeSpans(start, body.at(-1)?.span ?? update?.span ?? condition?.span ?? start),
    };
  }

  private parseBodyAsStatements(): readonly CudaLiteStatement[] {
    if (this.match("{")) return this.parseBlock();
    return [this.parseStatement()];
  }

  private parseVarDecl(expectSemicolon: boolean): CudaLiteVarDecl {
    const start = this.peek().span;
    const storage = this.consumeIf("__shared__") ? "shared" : "local";
    const valueType = this.parseType();
    const name = this.expectIdentifier("variable name");
    const dimensions: number[] = [];
    while (this.consumeIf("[")) {
      const size = this.expectNumber("array size");
      dimensions.push(Number(size.value));
      this.expect("]");
    }
    const init = this.consumeIf("=") ? this.parseExpression() : undefined;
    const end = expectSemicolon ? this.expect(";").span : (init?.span ?? name.span);
    return {
      kind: "var",
      storage,
      valueType,
      name: name.value,
      dimensions,
      ...(init === undefined ? {} : { init }),
      span: mergeSpans(start, end),
    };
  }

  private parseExpression(minPrecedence = 1): CudaLiteExpression {
    let left = this.parsePrefix();
    while (true) {
      if (ASSIGNMENT.has(this.peek().value) && minPrecedence <= 1) {
        const op = this.advance().value as CudaLiteAssignmentExpression["operator"];
        const right = this.parseExpression(1);
        left = { kind: "assignment", operator: op, left, right, span: mergeSpans(left.span, right.span) };
        continue;
      }
      const precedence = BINARY_PRECEDENCE.get(this.peek().value);
      if (precedence === undefined || precedence < minPrecedence) break;
      const op = this.advance().value as CudaLiteBinaryExpression["operator"];
      const right = this.parseExpression(precedence + 1);
      left = { kind: "binary", operator: op, left, right, span: mergeSpans(left.span, right.span) };
    }
    return left;
  }

  private parsePrefix(): CudaLiteExpression {
    const token = this.peek();
    if (["-", "+", "!", "&"].includes(token.value)) {
      const op = this.advance().value as CudaLiteUnaryExpression["operator"];
      const argument = this.parsePrefix();
      return { kind: "unary", operator: op, argument, span: mergeSpans(token.span, argument.span) };
    }
    if (["++", "--"].includes(token.value)) {
      const op = this.advance().value as CudaLiteUpdateExpression["operator"];
      const argument = this.parsePrefix();
      return { kind: "update", operator: op, argument, prefix: true, span: mergeSpans(token.span, argument.span) };
    }
    let expression = this.parsePrimary();
    while (true) {
      if (this.consumeIf(".")) {
        const property = this.expectIdentifier("member property");
        if (property.value !== "x" && property.value !== "y" && property.value !== "z") {
          this.fail(`unsupported vector member: ${property.value}`, property.span);
        }
        expression = {
          kind: "member",
          object: expression,
          property: property.value as CudaLiteMemberExpression["property"],
          span: mergeSpans(expression.span, property.span),
        };
        continue;
      }
      if (this.consumeIf("[")) {
        const index = this.parseExpression();
        const end = this.expect("]").span;
        expression = { kind: "index", target: expression, index, span: mergeSpans(expression.span, end) };
        continue;
      }
      if (this.consumeIf("(")) {
        const args: CudaLiteExpression[] = [];
        if (!this.match(")")) {
          do args.push(this.parseExpression());
          while (this.consumeIf(","));
        }
        const end = this.expect(")").span;
        expression = { kind: "call", callee: expression, args, span: mergeSpans(expression.span, end) };
        continue;
      }
      if (["++", "--"].includes(this.peek().value)) {
        const op = this.advance().value as CudaLiteUpdateExpression["operator"];
        expression = { kind: "update", operator: op, argument: expression, prefix: false, span: mergeSpans(expression.span, this.previous().span) };
        continue;
      }
      break;
    }
    return expression;
  }

  private parsePrimary(): CudaLiteExpression {
    if (this.match("(")) {
      this.expect("(");
      const expression = this.parseExpression();
      this.expect(")");
      return expression;
    }
    if (this.peek().kind === "number") {
      const token = this.advance();
      return { kind: "number", value: Number(token.value), raw: token.value, span: token.span };
    }
    const ident = this.expectIdentifier("expression");
    return { kind: "identifier", name: ident.value, span: ident.span };
  }

  private parseType(): Exclude<CudaLiteScalarType, "void"> {
    const token = this.expectIdentifier("type");
    if (!TYPE_KEYWORDS.has(token.value)) this.fail(`unsupported CUDA-lite type: ${token.value}`, token.span);
    return token.value as Exclude<CudaLiteScalarType, "void">;
  }

  private startsVarDecl(): boolean {
    if (this.match("__shared__")) return true;
    return TYPE_KEYWORDS.has(this.peek().value);
  }

  private expectIdentifier(label: string): Token {
    const token = this.peek();
    if (token.kind !== "identifier") this.fail(`expected ${label}`, token.span);
    return this.advance();
  }

  private expectNumber(label: string): Token {
    const token = this.peek();
    if (token.kind !== "number") this.fail(`expected ${label}`, token.span);
    return this.advance();
  }

  private expect(value: string): Token {
    const token = this.peek();
    if (token.value !== value) this.fail(`expected '${value}'`, token.span);
    return this.advance();
  }

  private consumeIf(value: string): Token | undefined {
    if (!this.match(value)) return undefined;
    return this.advance();
  }

  private match(value: string): boolean {
    return this.peek().value === value;
  }

  private peek(): Token {
    return this.tokens[this.index]!;
  }

  private previous(): Token {
    return this.tokens[Math.max(this.index - 1, 0)]!;
  }

  private advance(): Token {
    return this.tokens[this.index++]!;
  }

  private fail(message: string, span: SourceSpan): never {
    throw new CudaLiteCompilerError(message, [{
      code: "parse-error",
      severity: "error",
      message,
      span,
    }]);
  }
}

export function mergeSpans(left: SourceSpan, right: SourceSpan): SourceSpan {
  return {
    start: left.start,
    end: right.end,
    line: left.line,
    column: left.column,
  };
}
