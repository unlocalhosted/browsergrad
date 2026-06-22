import { tokenizeCudaLite, type Token } from "./lexer.js";
import {
  CudaLiteCompilerError,
  type CudaLiteAssignmentExpression,
  type CudaLiteAsmStatement,
  type CudaLiteBinaryExpression,
  type CudaLiteCastExpression,
  type CudaLiteCooperativeGroupDecl,
  type CudaLiteCooperativeGroupKind,
  type CudaLiteDeviceFunction,
  type CudaLiteDim3Decl,
  type CudaLiteExpression,
  type CudaLiteForStatement,
  type CudaLiteGlobalConstant,
  type CudaLiteIfStatement,
  type CudaLiteKernel,
  type CudaLiteKernelLaunchStatement,
  type CudaLiteModule,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type CudaLiteStatement,
  type CudaLiteTexture2D,
  type CudaLiteUnaryExpression,
  type CudaLiteUpdateExpression,
  type CudaLiteVarDecl,
  type SourceSpan,
} from "./types.js";

const TYPE_KEYWORDS = new Set(["float", "int", "uint", "half", "bool"]);
const ASSIGNMENT = new Set(["=", "+=", "-=", "*=", "/=", "<<=", ">>="]);
const BINARY_PRECEDENCE = new Map<string, number>([
  ["||", 2],
  ["&&", 3],
  ["|", 4],
  ["^", 5],
  ["&", 6],
  ["==", 7],
  ["!=", 7],
  ["<", 8],
  ["<=", 8],
  [">", 8],
  [">=", 8],
  ["<<", 9],
  [">>", 9],
  ["+", 10],
  ["-", 10],
  ["*", 11],
  ["/", 11],
  ["%", 11],
]);

export function parseCudaLite(source: string): CudaLiteModule {
  return new Parser(source).parseModule();
}

class Parser {
  private readonly source: string;
  private readonly tokens: readonly Token[];
  private index = 0;
  private anonymousKernelIndex = 0;

  constructor(source: string) {
    this.source = source;
    this.tokens = tokenizeCudaLite(source);
  }

  parseModule(): CudaLiteModule {
    const constants: CudaLiteGlobalConstant[] = [];
    const textures: CudaLiteTexture2D[] = [];
    const functions: CudaLiteDeviceFunction[] = [];
    const kernels: CudaLiteKernel[] = [];
    while (!this.match("<eof>")) {
      if (this.match("__constant__")) constants.push(this.parseGlobalConstant());
      else if (this.match("texture")) textures.push(this.parseTexture2D());
      else if (this.match("namespace")) this.skipNamespaceAliasDecl();
      else if (this.startsDeviceFunction()) functions.push(this.parseDeviceFunction());
      else kernels.push(this.parseKernel());
    }
    return {
      kind: "module",
      source: this.source,
      constants,
      textures,
      functions,
      kernels,
      span: { start: 0, end: this.source.length, line: 1, column: 1 },
    };
  }

  private parseTexture2D(): CudaLiteTexture2D {
    const start = this.expect("texture").span;
    this.expect("<");
    const valueType = this.parseType();
    this.expect(",");
    const textureType = this.expectTextureDimension();
    this.expect(",");
    const readMode = this.expectIdentifier("texture read mode");
    this.expect(">");
    const name = this.expectIdentifier("texture name");
    const end = this.expect(";").span;
    if (valueType !== "float") this.fail("only texture<float, cudaTextureType2D, ...> is supported in CUDA-lite", start);
    if (textureType.value !== "cudaTextureType2D" && textureType.value !== "2") {
      this.fail("only cudaTextureType2D / 2D textures are supported in CUDA-lite", textureType.span);
    }
    if (readMode.value !== "cudaReadModeElementType") {
      this.fail("only cudaReadModeElementType texture reads are supported in CUDA-lite", readMode.span);
    }
    return {
      kind: "texture2d",
      valueType: "float",
      name: name.value,
      span: mergeSpans(start, end),
    };
  }

  private parseGlobalConstant(): CudaLiteGlobalConstant {
    const start = this.expect("__constant__").span;
    const valueType = this.parseType();
    const name = this.expectIdentifier("constant name");
    const dimensions: number[] = [];
    while (this.consumeIf("[")) {
      if (!this.match("]")) dimensions.push(this.parseArrayDimension());
      this.expect("]");
    }
    const end = this.expect(";").span;
    return {
      kind: "constant",
      valueType,
      name: name.value,
      dimensions,
      span: mergeSpans(start, end),
    };
  }

  private parseDeviceFunction(): CudaLiteDeviceFunction {
    const start = this.consumeDeviceFunctionAttributes();
    const returnType = this.parseReturnType();
    const name = this.expectIdentifier("device function name");
    this.expect("(");
    const params = this.parseParams();
    this.expect(")");
    const body = this.parseBlock();
    return {
      kind: "device-function",
      name: name.value,
      returnType,
      params,
      body,
      span: mergeSpans(start, body.at(-1)?.span ?? name.span),
    };
  }

  private parseKernel(): CudaLiteKernel {
    this.consumeKernelAttributes();
    const start = this.expect("__global__").span;
    this.consumeKernelAttributes();
    let name: string;
    let nameSpan = start;
    if (this.match("(")) {
      this.anonymousKernelIndex++;
      name = `anonymous_kernel_${this.anonymousKernelIndex}`;
    } else {
      this.expect("void");
      const nameToken = this.expectIdentifier("kernel name");
      name = nameToken.value;
      nameSpan = nameToken.span;
    }
    this.expect("(");
    const params = this.parseParams();
    this.expect(")");
    const body = this.parseBlock();
    return {
      kind: "kernel",
      name,
      params,
      body,
      span: mergeSpans(start, body.at(-1)?.span ?? nameSpan),
    };
  }

  private consumeKernelAttributes(): void {
    while (this.consumeIf("__launch_bounds__")) {
      this.expect("(");
      if (!this.match(")")) {
        do {
          this.parseExpression();
        } while (this.consumeIf(","));
      }
      this.expect(")");
    }
  }

  private consumeDeviceFunctionAttributes(): SourceSpan {
    let start: SourceSpan | undefined;
    let sawDevice = false;
    while (this.match("__inline__") || this.match("inline") || this.match("__device__") || this.match("__forceinline__")) {
      const token = this.advance();
      start ??= token.span;
      if (token.value === "__device__") sawDevice = true;
    }
    if (!sawDevice) this.fail("expected __device__ function attribute", start ?? this.peek().span);
    return start ?? this.peek().span;
  }

  private parseParams(): readonly CudaLiteParam[] {
    const params: CudaLiteParam[] = [];
    if (this.match(")")) return params;
    do {
      if (this.match(")")) break;
      const startToken = this.peek();
      const constant = this.consumeIf("const") !== undefined;
      const type = this.parseType();
      const pointer = this.consumeIf("*") !== undefined;
      this.consumeTypeQualifiers();
      const name = this.expectIdentifier("parameter name");
      this.consumeTypeQualifiers();
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
    while (!this.match("}")) statements.push(...this.parseStatementEntry());
    this.expect("}");
    return statements;
  }

  private parseStatementEntry(): readonly CudaLiteStatement[] {
    if (this.match("{")) {
      this.fail("standalone blocks are not supported in CUDA-lite v0", this.peek().span);
    }
    if (this.match("if")) return [this.parseIf()];
    if (this.match("for")) return [this.parseFor()];
    if (this.match("return")) return [this.parseReturn()];
    if (this.match("continue")) return [this.parseContinue()];
    if (this.match("dim3")) return [this.parseDim3Decl()];
    if (this.startsCooperativeGroupDecl()) return [this.parseCooperativeGroupDecl()];
    if (this.startsKernelLaunch()) return [this.parseKernelLaunch()];
    if (this.match("asm")) return [this.parseAsmStatement()];
    if (this.startsVarDecl()) return this.parseVarDeclList(true);
    const expression = this.parseExpression();
    const end = this.expect(";");
    return [{ kind: "expr", expression, span: mergeSpans(expression.span, end.span) }];
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

  private parseReturn(): CudaLiteStatement {
    const start = this.expect("return").span;
    const value = this.match(";") ? undefined : this.parseExpression();
    const end = this.expect(";");
    return {
      kind: "return",
      ...(value === undefined ? {} : { value }),
      span: mergeSpans(start, end.span),
    };
  }

  private parseContinue(): CudaLiteStatement {
    const start = this.expect("continue").span;
    const end = this.expect(";");
    return { kind: "continue", span: mergeSpans(start, end.span) };
  }

  private parseDim3Decl(): CudaLiteDim3Decl {
    const start = this.expect("dim3").span;
    const name = this.expectIdentifier("dim3 variable name");
    if (this.match("(")) this.skipBalanced("(", ")");
    const end = this.expect(";").span;
    return {
      kind: "dim3",
      name: name.value,
      span: mergeSpans(start, end),
    };
  }

  private parseCooperativeGroupDecl(): CudaLiteCooperativeGroupDecl {
    const start = this.peek().span;
    let name: Token;
    let groupKind: CudaLiteCooperativeGroupKind;
    let tileSize: number | undefined;
    if (this.consumeIf("auto")) {
      name = this.expectIdentifier("cooperative group variable name");
      this.expect("=");
      this.consumeNamespaceQualifier();
      this.expect("tiled_partition");
      this.expect("<");
      tileSize = this.parseTemplateIntegerArgument();
      this.expect(">");
      this.skipBalanced("(", ")");
      groupKind = "tile";
    } else {
      this.consumeNamespaceQualifier();
      const type = this.expectIdentifier("cooperative group type");
      groupKind = type.value === "thread_block" ? "block" : type.value === "grid_group" ? "grid" : "tile";
      name = this.expectIdentifier("cooperative group variable name");
      if (this.consumeIf("=")) {
        while (!this.match(";")) {
          if (this.match("<eof>")) this.fail("expected ';'", this.peek().span);
          this.advance();
        }
      }
    }
    const end = this.expect(";").span;
    return {
      kind: "cooperative-group",
      groupKind,
      name: name.value,
      ...(tileSize === undefined ? {} : { tileSize }),
      span: mergeSpans(start, end),
    };
  }

  private parseKernelLaunch(): CudaLiteKernelLaunchStatement {
    const name = this.expectIdentifier("kernel launch callee");
    this.expect("<<");
    this.expect("<");
    this.skipUntilTripleChevronClose();
    this.skipBalanced("(", ")");
    const end = this.expect(";").span;
    return {
      kind: "kernel-launch",
      callee: name.value,
      span: mergeSpans(name.span, end),
    };
  }

  private parseAsmStatement(): CudaLiteAsmStatement {
    const start = this.expect("asm").span;
    this.consumeIf("volatile");
    this.expect("(");
    const template = this.expectString("inline assembly template");
    this.expect(":");
    const output = this.parseAsmOperand();
    this.expect(":");
    const inputs: CudaLiteExpression[] = [];
    if (!this.match(")")) {
      do inputs.push(this.parseAsmOperand());
      while (this.consumeIf(","));
    }
    this.expect(")");
    const end = this.expect(";").span;
    return {
      kind: "asm",
      template: template.value.slice(1, -1),
      output,
      inputs,
      span: mergeSpans(start, end),
    };
  }

  private parseAsmOperand(): CudaLiteExpression {
    this.expectString("inline assembly constraint");
    this.expect("(");
    const expression = this.parseExpression();
    this.expect(")");
    return expression;
  }

  private parseBodyAsStatements(): readonly CudaLiteStatement[] {
    if (this.match("{")) return this.parseBlock();
    return this.parseStatementEntry();
  }

  private parseVarDecl(expectSemicolon: boolean): CudaLiteVarDecl {
    const declarations = this.parseVarDeclList(expectSemicolon);
    if (declarations.length !== 1) {
      this.fail("multiple for-init declarations are not supported in CUDA-lite v0", declarations[1]?.span ?? declarations[0]!.span);
    }
    return declarations[0]!;
  }

  private parseVarDeclList(expectSemicolon: boolean): readonly CudaLiteVarDecl[] {
    const start = this.peek().span;
    const storageInfo = this.consumeStorageQualifier();
    const valueType = this.parseType();
    const declarations: CudaLiteVarDecl[] = [];
    do {
      const pointer = this.consumeIf("*") !== undefined;
      const name = this.expectIdentifier("variable name");
      const dimensions: number[] = [];
      while (this.consumeIf("[")) {
        if (!this.match("]")) {
          const size = this.parseArrayDimension();
          dimensions.push(size);
        }
        this.expect("]");
      }
      const init = this.consumeIf("=") ? this.parseExpression() : undefined;
      declarations.push({
        kind: "var",
        storage: storageInfo.storage,
        valueType,
        pointer,
        name: name.value,
        dimensions,
        ...(storageInfo.dynamicShared ? { dynamicShared: true } : {}),
        ...(init === undefined ? {} : { init }),
        span: mergeSpans(start, init?.span ?? name.span),
      });
    } while (this.consumeIf(","));
    const end = expectSemicolon ? this.expect(";").span : (declarations.at(-1)?.span ?? start);
    return declarations.map((declaration) => ({
      ...declaration,
      span: mergeSpans(declaration.span, end),
    }));
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
      if (this.match("?") && minPrecedence <= 1) {
        this.expect("?");
        const consequent = this.parseExpression();
        this.expect(":");
        const alternate = this.parseExpression(1);
        left = { kind: "conditional", condition: left, consequent, alternate, span: mergeSpans(left.span, alternate.span) };
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
    if (this.startsScalarCast()) {
      const start = this.expect("(").span;
      const valueType = this.parseType();
      const pointer = this.consumeIf("*") !== undefined;
      this.expect(")");
      const expression = this.parsePrefix();
      return {
        kind: "cast",
        valueType,
        ...(pointer ? { pointer: true } : {}),
        expression,
        span: mergeSpans(start, expression.span),
      } satisfies CudaLiteCastExpression;
    }
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
        expression = {
          kind: "member",
          object: expression,
          property: property.value,
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
      const token = this.parseNumberLiteral("numeric literal");
      return { kind: "number", value: token.value, raw: token.raw, span: token.span };
    }
    if (this.peek().kind === "string") {
      const token = this.advance();
      return { kind: "string", value: token.value.slice(1, -1), raw: token.value, span: token.span };
    }
    const ident = this.expectIdentifier("expression");
    return { kind: "identifier", name: ident.value, span: ident.span };
  }

  private startsScalarCast(): boolean {
    if (!this.match("(")) return false;
    const typeToken = this.tokens[this.index + 1];
    const maybeStar = this.tokens[this.index + 2];
    const closeToken = maybeStar?.value === "*" ? this.tokens[this.index + 3] : maybeStar;
    if (typeToken === undefined || closeToken === undefined || closeToken.value !== ")") return false;
    if (typeToken.value === "unsigned") return true;
    if (typeToken.value === "size_t") return true;
    if (typeToken.value === "void") return true;
    if (typeToken.value === "DevicePool") return true;
    if (typeToken.value === "cufftComplex") return true;
    return TYPE_KEYWORDS.has(typeToken.value);
  }

  private startsDeviceFunction(): boolean {
    let index = this.index;
    let sawDevice = false;
    while (true) {
      const token = this.tokens[index];
      if (!token) return false;
      if (token.value === "__device__") sawDevice = true;
      if (token.value === "__inline__" || token.value === "inline" || token.value === "__device__" || token.value === "__forceinline__") {
        index++;
        continue;
      }
      break;
    }
    return sawDevice;
  }

  private startsKernelLaunch(): boolean {
    return this.peek().kind === "identifier" &&
      this.tokens[this.index + 1]?.value === "<<" &&
      this.tokens[this.index + 2]?.value === "<";
  }

  private startsCooperativeGroupDecl(): boolean {
    if (this.match("auto")) {
      return this.tokens[this.index + 2]?.value === "=" &&
        this.tokens.slice(this.index + 3, this.index + 8).some((token) => token.value === "tiled_partition");
    }
    return this.peek().kind === "identifier" &&
      this.tokens[this.index + 1]?.value === "::" &&
      (this.tokens[this.index + 2]?.value === "thread_block" || this.tokens[this.index + 2]?.value === "grid_group");
  }

  private expectTextureDimension(): Token {
    if (this.peek().kind === "number") {
      const token = this.advance();
      if (token.value !== "2") this.fail("only 2D textures are supported in CUDA-lite", token.span);
      return token;
    }
    return this.expectIdentifier("texture type");
  }

  private parseType(): Exclude<CudaLiteScalarType, "void"> {
    const token = this.expectIdentifier("type");
    if (token.value === "unsigned") {
      if (this.consumeIf("int")) return "uint";
      if (this.consumeIf("long")) {
        if (this.consumeIf("long")) {
          return "uint";
        }
        return "uint";
      }
      return "uint";
    }
    if (token.value === "size_t") return "uint";
    if (token.value === "curandState_t") return "uint";
    if (token.value === "cufftComplex") return "complex64";
    if (token.value === "cudaSurfaceObject_t") return "surface2d";
    if (token.value === "DevicePool") return "devicepool";
    if (token.value === "void") return "voidptr";
    if (!TYPE_KEYWORDS.has(token.value)) this.fail(`unsupported CUDA-lite type: ${token.value}`, token.span);
    return token.value as Exclude<CudaLiteScalarType, "void">;
  }

  private parseReturnType(): CudaLiteScalarType {
    if (this.consumeIf("void")) return "void";
    return this.parseType();
  }

  private startsVarDecl(): boolean {
    if (this.match("__shared__") || this.match("extern")) return true;
    if (this.match("unsigned")) return true;
    if (this.match("curandState_t")) return true;
    if (this.match("cufftComplex")) return true;
    if (this.match("DevicePool")) return true;
    if (this.match("void")) return true;
    return TYPE_KEYWORDS.has(this.peek().value);
  }

  private consumeStorageQualifier(): { readonly storage: "local" | "shared"; readonly dynamicShared: boolean } {
    const external = this.consumeIf("extern") !== undefined;
    if (this.consumeIf("__shared__")) return { storage: "shared", dynamicShared: external };
    if (external) this.fail("extern is only supported with __shared__ declarations", this.previous().span);
    return { storage: "local", dynamicShared: false };
  }

  private consumeTypeQualifiers(): void {
    while (
      this.consumeIf("__restrict__") ||
      this.consumeIf("__restrict") ||
      this.consumeIf("restrict")
    ) {
      // accepted CUDA pointer qualifiers; no CUDA-lite semantic effect
    }
  }

  private expectIdentifier(label: string): Token {
    const token = this.peek();
    if (token.kind !== "identifier") this.fail(`expected ${label}`, token.span);
    return this.advance();
  }

  private expectString(label: string): Token {
    const token = this.peek();
    if (token.kind !== "string") this.fail(`expected ${label}`, token.span);
    return this.advance();
  }

  private expectNumber(label: string): Token {
    const token = this.peek();
    if (token.kind !== "number") this.fail(`expected ${label}`, token.span);
    return this.advance();
  }

  private parseNumberLiteral(label: string): { readonly value: number; readonly raw: string; readonly span: SourceSpan } {
    const token = this.expectNumber(label);
    const rawNumber = /^0[xX][0-9A-Fa-f]+/u.exec(token.value)?.[0] ?? token.value.replace(/[fFuUlL]+$/u, "");
    const value = Number(rawNumber);
    if (!Number.isFinite(value)) this.fail(`invalid ${label}: ${token.value}`, token.span);
    return { value, raw: rawNumber, span: token.span };
  }

  private parseArrayDimension(): number {
    const expression = this.parseExpression();
    const value = evaluateIntegerConstantExpression(expression);
    if (value === undefined) {
      this.fail("array size must be an integer constant expression", expression.span);
    }
    return value;
  }

  private parseTemplateIntegerArgument(): number {
    const token = this.parseNumberLiteral("template integer argument");
    if (!Number.isInteger(token.value) || token.value <= 0) {
      this.fail("template argument must be a positive integer", token.span);
    }
    return token.value;
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

  private skipBalanced(open: string, close: string): void {
    this.expect(open);
    let depth = 1;
    while (depth > 0) {
      if (this.match("<eof>")) this.fail(`expected '${close}'`, this.peek().span);
      const token = this.advance();
      if (token.value === open) depth++;
      else if (token.value === close) depth--;
    }
  }

  private skipUntilTripleChevronClose(): void {
    while (!(this.peek().value === ">>" && this.tokens[this.index + 1]?.value === ">")) {
      if (this.match("<eof>")) this.fail("expected '>>>'", this.peek().span);
      this.advance();
    }
    this.expect(">>");
    this.expect(">");
  }

  private consumeNamespaceQualifier(): void {
    this.expectIdentifier("namespace alias");
    this.expect("::");
  }

  private skipNamespaceAliasDecl(): void {
    this.expect("namespace");
    this.expectIdentifier("namespace alias");
    this.expect("=");
    this.expectIdentifier("namespace target");
    while (!this.match(";")) {
      if (this.match("<eof>")) this.fail("expected ';'", this.peek().span);
      this.advance();
    }
    this.expect(";");
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

function evaluateIntegerConstantExpression(expression: CudaLiteExpression): number | undefined {
  switch (expression.kind) {
    case "number":
      return Number.isInteger(expression.value) ? expression.value : undefined;
    case "unary": {
      const value = evaluateIntegerConstantExpression(expression.argument);
      if (value === undefined) return undefined;
      if (expression.operator === "+") return value;
      if (expression.operator === "-") return -value;
      return undefined;
    }
    case "binary": {
      const left = evaluateIntegerConstantExpression(expression.left);
      const right = evaluateIntegerConstantExpression(expression.right);
      if (left === undefined || right === undefined) return undefined;
      switch (expression.operator) {
        case "+":
          return left + right;
        case "-":
          return left - right;
        case "*":
          return left * right;
        case "/":
          return right === 0 ? undefined : Math.trunc(left / right);
        case "%":
          return right === 0 ? undefined : left % right;
        case "<<":
          return left << right;
        case ">>":
          return left >> right;
        default:
          return undefined;
      }
    }
    default:
      return undefined;
  }
}
