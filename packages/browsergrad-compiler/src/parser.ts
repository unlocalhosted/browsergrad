import { tokenizeCudaLite, type Token } from "./lexer.js";
import {
  CudaLiteCompilerError,
  type CudaLiteAssignmentExpression,
  type CudaLiteAsmStatement,
  type CudaLiteBinaryExpression,
  type CudaLiteBlockStatement,
  type CudaLiteCastExpression,
  type CudaLiteCooperativeGroupDecl,
  type CudaLiteCooperativeGroupKind,
  type CudaLiteDeviceFunction,
  type CudaLiteDim3Decl,
  type CudaLiteExpression,
  type CudaLiteForStatement,
  type CudaLiteGlobalConstant,
  type CudaLiteIfStatement,
  type CudaLiteInitializerExpression,
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
import { CUDA_VECTOR_TYPES, cudaVectorTypeAlias, isCudaVectorType } from "./vector_types.js";
import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";

const CUDA_SCALAR_TYPE_ALIASES = new Map<string, Exclude<CudaLiteScalarType, "void">>([
  ["char", "int"],
  ["int8_t", "int"],
  ["uchar", "uint"],
  ["uint8_t", "uint"],
  ["clock_t", "uint"],
]);
const TYPE_KEYWORDS = new Set([
  "float",
  "int",
  "uint",
  "half",
  "__half",
  "bool",
  ...CUDA_SCALAR_TYPE_ALIASES.keys(),
  ...CUDA_VECTOR_TYPES.keys(),
]);
const TYPE_START_KEYWORDS = new Set([
  ...TYPE_KEYWORDS,
  "char2",
  "char3",
  "char4",
  "uchar2",
  "uchar3",
  "uchar4",
  "uint8_t2",
  "uint8_t3",
  "uint8_t4",
  "int8_t2",
  "int8_t3",
  "int8_t4",
  "unsigned",
  "signed",
  "long",
  "short",
  "size_t",
  "int32_t",
  "uint32_t",
  "int64_t",
  "uint64_t",
  "uintptr_t",
  "curandState_t",
  "cufftComplex",
  "cudaTextureObject_t",
  "cudaSurfaceObject_t",
  "cudaEvent_t",
  "cudaStream_t",
  "DevicePool",
  "void",
]);
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
  private readonly integerConstantScopes: Map<string, number>[] = [new Map()];
  private templateArgumentDepth = 0;

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
      else if (this.match("typedef") || this.match("using")) this.skipSimpleDeclaration();
      else if (this.match("template")) {
        const templateConstants = this.parseTemplateIntegerDefaults();
        this.integerConstantScopes.push(templateConstants);
        try {
          if (this.startsDeviceFunction()) functions.push(this.parseDeviceFunction());
          else kernels.push(this.parseKernel());
        } finally {
          this.integerConstantScopes.pop();
        }
      }
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
    let inferredArrayDimension = false;
    while (this.consumeIf("[")) {
      if (!this.match("]")) dimensions.push(this.parseArrayDimension());
      else inferredArrayDimension = true;
      this.expect("]");
    }
    const init = this.consumeIf("=")
      ? this.parseInitializerExpression(valueType, inferredArrayDimension && dimensions.length === 0 ? [1] : dimensions)
      : undefined;
    if (inferredArrayDimension) {
      if (!init) this.fail("unsized constant arrays require an initializer", name.span);
      dimensions.push(flattenInitializerExpressions(init).length);
    }
    const end = this.expect(";").span;
    return {
      kind: "constant",
      valueType,
      name: name.value,
      dimensions,
      ...(init === undefined ? {} : { init }),
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
    this.consumeKernelQualifiers();
    let name: string;
    let nameSpan = start;
    if (this.match("(")) {
      this.anonymousKernelIndex++;
      name = `anonymous_kernel_${this.anonymousKernelIndex}`;
    } else {
      this.expect("void");
      this.consumeKernelAttributes();
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

  private consumeKernelQualifiers(): void {
    while (this.consumeIf("static") || this.consumeIf("extern")) {
      // CUDA samples sometimes spell kernels as `__global__ static void`.
    }
  }

  private consumeDeviceFunctionAttributes(): SourceSpan {
    let start: SourceSpan | undefined;
    let sawDevice = false;
    while (
      this.match("static") ||
      this.match("__inline__") ||
      this.match("inline") ||
      this.match("__device__") ||
      this.match("__host__") ||
      this.match("__forceinline__")
    ) {
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
      this.consumeIf("&");
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

  private parseTemplateIntegerDefaults(): Map<string, number> {
    const constants = new Map<string, number>();
    this.expect("template");
    this.expect("<");
    if (!this.match(">")) {
      do {
        const parsed = this.parseTemplateParameterDefault();
        if (parsed) constants.set(parsed.name, parsed.value);
      } while (this.consumeIf(","));
    }
    this.expect(">");
    return constants;
  }

  private parseTemplateParameterDefault(): { readonly name: string; readonly value: number } | undefined {
    const start = this.peek().span;
    this.consumeIf("const");
    const typeToken = this.peek();
    if (
      typeToken.value === "int" ||
      typeToken.value === "uint" ||
      typeToken.value === "unsigned" ||
      typeToken.value === "size_t" ||
      typeToken.value === "bool"
    ) {
      this.parseType();
      const name = this.expectIdentifier("template parameter name");
      if (!this.consumeIf("=")) return undefined;
      this.templateArgumentDepth++;
      let expression: CudaLiteExpression;
      try {
        expression = this.parseExpression();
      } finally {
        this.templateArgumentDepth--;
      }
      const value = this.evaluateIntegerConstantExpression(expression);
      if (value === undefined || !Number.isInteger(value)) {
        this.fail("template default must be an integer constant expression", expression.span);
      }
      return { name: name.value, value };
    }
    this.skipTemplateParameter(start);
    return undefined;
  }

  private skipTemplateParameter(start: SourceSpan): void {
    let depth = 0;
    while (!this.match("<eof>")) {
      if (depth === 0 && (this.match(",") || this.match(">"))) return;
      const value = this.advance().value;
      if (value === "<") depth++;
      else if (value === ">") {
        if (depth === 0) {
          this.index--;
          return;
        }
        depth--;
      } else if (value === ">>") {
        depth = Math.max(0, depth - 2);
      }
    }
    this.fail("unterminated template parameter list", start);
  }

  private parseBlock(): readonly CudaLiteStatement[] {
    this.expect("{");
    this.integerConstantScopes.push(new Map());
    const statements: CudaLiteStatement[] = [];
    while (!this.match("}")) statements.push(...this.parseStatementEntry());
    this.expect("}");
    this.integerConstantScopes.pop();
    return statements;
  }

  private parseStatementEntry(): readonly CudaLiteStatement[] {
    if (this.match("{")) return [this.parseStandaloneBlock()];
    if (this.match("if")) return [this.parseIf()];
    if (this.match("for")) return [this.parseFor()];
    if (this.match("return")) return [this.parseReturn()];
    if (this.match("continue")) return [this.parseContinue()];
    if (this.match("namespace")) {
      this.skipNamespaceAliasDecl();
      return [];
    }
    if (this.match("using") && this.tokens[this.index + 1]?.value === "namespace") {
      this.skipSimpleDeclaration();
      return [];
    }
    if (this.match("dim3")) return [this.parseDim3Decl()];
    if (this.startsCooperativeGroupDecl()) return [this.parseCooperativeGroupDecl()];
    if (this.startsKernelLaunch()) return [this.parseKernelLaunch()];
    if (this.match("asm")) return [this.parseAsmStatement()];
    if (this.match("static_assert")) return this.parseStaticAssert();
    if (this.startsVarDecl()) return this.parseVarDeclList(true);
    const expression = this.parseExpression();
    const end = this.expect(";");
    return [{ kind: "expr", expression, span: mergeSpans(expression.span, end.span) }];
  }

  private parseStandaloneBlock(): CudaLiteBlockStatement {
    const start = this.peek().span;
    const body = this.parseBlock();
    return {
      kind: "block",
      body,
      span: mergeSpans(start, body.at(-1)?.span ?? start),
    };
  }

  private parseIf(): CudaLiteIfStatement {
    const start = this.expect("if").span;
    this.consumeIf("constexpr");
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

  private parseFor(): CudaLiteForStatement | CudaLiteBlockStatement {
    const start = this.expect("for").span;
    this.expect("(");
    let init: CudaLiteVarDecl | CudaLiteExpression | undefined;
    let initDeclarations: readonly CudaLiteVarDecl[] | undefined;
    if (!this.match(";")) {
      if (this.startsVarDecl()) {
        initDeclarations = this.parseVarDeclList(false);
        init = initDeclarations.length === 1 ? initDeclarations[0] : undefined;
      } else {
        init = this.parseExpressionSequence();
      }
    }
    this.expect(";");
    const condition = this.match(";") ? undefined : this.parseExpression();
    this.expect(";");
    const update = this.match(")") ? undefined : this.parseExpressionSequence();
    this.expect(")");
    const body = this.parseBodyAsStatements();
    const statement: CudaLiteForStatement = {
      kind: "for",
      ...(init === undefined ? {} : { init }),
      ...(condition === undefined ? {} : { condition }),
      ...(update === undefined ? {} : { update }),
      body,
      span: mergeSpans(start, body.at(-1)?.span ?? update?.span ?? condition?.span ?? start),
    };
    if (initDeclarations !== undefined && initDeclarations.length > 1) {
      return {
        kind: "block",
        body: [
          ...initDeclarations,
          statement,
        ],
        span: mergeSpans(start, statement.span),
      };
    }
    return statement;
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
    const args = this.match("(") ? this.parseArgumentList() : [];
    const end = this.expect(";").span;
    return {
      kind: "dim3",
      name: name.value,
      args,
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
      if (type.value === "thread_block_tile") {
        this.expect("<");
        tileSize = this.parseTemplateIntegerArgument();
        this.expect(">");
      }
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
    const launchArgs: CudaLiteExpression[][] = [];
    if (!(this.peek().value === ">>" && this.tokens[this.index + 1]?.value === ">")) {
      do {
        launchArgs.push([this.parseExpression()]);
      } while (this.consumeIf(","));
    }
    this.expect(">>");
    this.expect(">");
    const args = this.parseArgumentList();
    const end = this.expect(";").span;
    return {
      kind: "kernel-launch",
      callee: name.value,
      grid: launchArgs[0] ?? [],
      block: launchArgs[1] ?? [],
      args,
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

  private parseStaticAssert(): readonly CudaLiteStatement[] {
    this.expect("static_assert");
    this.skipBalanced("(", ")");
    this.expect(";");
    return [];
  }

  private parseBodyAsStatements(): readonly CudaLiteStatement[] {
    if (this.match("{")) return this.parseBlock();
    return this.parseStatementEntry();
  }

  private parseVarDeclList(expectSemicolon: boolean): readonly CudaLiteVarDecl[] {
    const start = this.peek().span;
    this.consumeIf("static");
    const storageInfo = this.consumeStorageQualifier();
    this.consumeIf("static");
    const constexpr = this.consumeIf("constexpr") !== undefined;
    this.consumeIf("const");
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
      const init = this.consumeIf("=") ? this.parseInitializerExpression(valueType, dimensions) : undefined;
      if (constexpr && init !== undefined) {
        const value = this.evaluateIntegerConstantExpression(init);
        if (value !== undefined) this.recordIntegerConstant(name.value, value);
      }
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

  private parseInitializerExpression(valueType: Exclude<CudaLiteScalarType, "void">, dimensions: readonly number[]): CudaLiteExpression {
    if (!this.match("{")) return this.parseExpression();
    const initializer = this.parseBracedInitializer();
    if (dimensions.length > 0) return initializer;
    if (isCudaVectorType(valueType)) {
      return {
        kind: "call",
        callee: { kind: "identifier", name: `make_${valueType}`, span: initializer.span },
        args: initializer.elements,
        span: initializer.span,
      };
    }
    if (initializer.elements.length === 0) {
      return { kind: "number", value: 0, raw: "0", span: initializer.span };
    }
    if (initializer.elements.length === 1) return initializer.elements[0]!;
    this.fail("scalar braced initializer expects at most one value", initializer.span);
  }

  private parseBracedInitializer(): CudaLiteInitializerExpression {
    const start = this.expect("{").span;
    const args: CudaLiteExpression[] = [];
    if (!this.match("}")) {
      do args.push(this.match("{") ? this.parseBracedInitializer() : this.parseExpression());
      while (this.consumeIf(","));
    }
    const end = this.expect("}").span;
    return {
      kind: "initializer",
      elements: args,
      span: mergeSpans(start, end),
    };
  }

  private parseExpression(minPrecedence = 1): CudaLiteExpression {
    let left = this.parsePrefix();
    while (true) {
      if (this.peek().value === ">>" && this.tokens[this.index + 1]?.value === ">") break;
      if (this.templateArgumentDepth > 0 && this.peek().value === ">") break;
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

  private parseExpressionSequence(): CudaLiteExpression {
    const expressions = [this.parseExpression()];
    while (this.consumeIf(",")) expressions.push(this.parseExpression());
    if (expressions.length === 1) return expressions[0]!;
    return {
      kind: "sequence",
      expressions,
      span: mergeSpans(expressions[0]!.span, expressions.at(-1)!.span),
    };
  }

  private parsePrefix(): CudaLiteExpression {
    const token = this.peek();
    if (this.startsScalarCast()) {
      const start = this.expect("(").span;
      this.consumeIf("const");
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
    if (["-", "+", "!", "&", "*"].includes(token.value)) {
      const op = this.advance().value as CudaLiteUnaryExpression["operator"];
      const argument = this.parsePrefix();
      return { kind: "unary", operator: op, argument, span: mergeSpans(token.span, argument.span) };
    }
    if (["++", "--"].includes(token.value)) {
      const op = this.advance().value as CudaLiteUpdateExpression["operator"];
      const argument = this.parsePrefix();
      return { kind: "update", operator: op, argument, prefix: true, span: mergeSpans(token.span, argument.span) };
    }
    if (this.startsFunctionalCast()) return this.parseFunctionalCastExpression();
    let expression = this.startsCxxCast()
      ? this.parseCxxCastExpression()
      : this.parsePrimary();
    while (true) {
      this.consumeTemplateArgumentsBeforeCall();
      if (this.match("{")) {
        this.skipBalanced("{", "}");
        continue;
      }
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
        const { args, end } = this.parseArgumentListAfterOpen();
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

  private parseCxxCastExpression(): CudaLiteExpression {
    const start = this.advance().span;
    this.expect("<");
    this.consumeIf("const");
    const valueType = this.parseType();
    this.consumeIf("const");
    const pointer = this.consumeIf("*") !== undefined;
    this.consumeIf("const");
    this.expect(">");
    this.expect("(");
    const expression = this.parseExpression();
    const end = this.expect(")").span;
    return {
      kind: "cast",
      valueType,
      ...(pointer ? { pointer: true } : {}),
      expression,
      span: mergeSpans(start, end),
    } satisfies CudaLiteCastExpression;
  }

  private parseFunctionalCastExpression(): CudaLiteExpression {
    const start = this.peek().span;
    const valueType = this.parseType();
    this.expect("(");
    const expression = this.parseExpression();
    const end = this.expect(")").span;
    return {
      kind: "cast",
      valueType,
      expression,
      span: mergeSpans(start, end),
    } satisfies CudaLiteCastExpression;
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
    let name = ident.value;
    let end = ident.span;
    while (this.consumeIf("::")) {
      const part = this.expectIdentifier("qualified expression");
      name += `::${part.value}`;
      end = part.span;
    }
    const constant = this.lookupIntegerConstant(name);
    if (constant !== undefined) {
      return { kind: "number", value: constant, raw: String(constant), span: mergeSpans(ident.span, end) };
    }
    return { kind: "identifier", name, span: mergeSpans(ident.span, end) };
  }

  private parseArgumentList(): readonly CudaLiteExpression[] {
    this.expect("(");
    return this.parseArgumentListAfterOpen().args;
  }

  private parseArgumentListAfterOpen(): { readonly args: readonly CudaLiteExpression[]; readonly end: SourceSpan } {
    const args: CudaLiteExpression[] = [];
    if (!this.match(")")) {
      do args.push(this.parseExpression());
      while (this.consumeIf(","));
    }
    const end = this.expect(")").span;
    return { args, end };
  }

  private startsScalarCast(): boolean {
    if (!this.match("(")) return false;
    const typeEndIndex = this.typeEndIndex(this.index + 1);
    if (typeEndIndex === undefined) return false;
    const closeIndex = this.tokens[typeEndIndex]?.value === "*" ? typeEndIndex + 1 : typeEndIndex;
    return this.tokens[closeIndex]?.value === ")";
  }

  private startsCxxCast(): boolean {
    const token = this.peek();
    return token.kind === "identifier" &&
      (token.value === "reinterpret_cast" || token.value === "static_cast" || token.value === "const_cast") &&
      this.tokens[this.index + 1]?.value === "<";
  }

  private startsFunctionalCast(): boolean {
    return this.peek().kind === "identifier" &&
      TYPE_KEYWORDS.has(this.peek().value) &&
      this.tokens[this.index + 1]?.value === "(";
  }

  private startsDeviceFunction(): boolean {
    let index = this.index;
    let sawDevice = false;
    while (true) {
      const token = this.tokens[index];
      if (!token) return false;
      if (token.value === "__device__") sawDevice = true;
      if (
        token.value === "static" ||
        token.value === "__inline__" ||
        token.value === "inline" ||
        token.value === "__device__" ||
        token.value === "__host__" ||
        token.value === "__forceinline__"
      ) {
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
      (this.tokens[this.index + 2]?.value === "thread_block" ||
        this.tokens[this.index + 2]?.value === "grid_group" ||
        this.tokens[this.index + 2]?.value === "thread_block_tile");
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
      if (this.consumeIf("char")) return "uint";
      this.consumeIntegerWidthSuffix();
      return "uint";
    }
    if (token.value === "signed") {
      if (this.consumeIf("char")) return "int";
      this.consumeIntegerWidthSuffix();
      return "int";
    }
    if (token.value === "long") {
      this.consumeIf("long");
      return "int";
    }
    if (token.value === "short") {
      this.consumeIf("int");
      return "int";
    }
    if (token.value === "size_t") return "uint";
    if (token.value === "int32_t" || token.value === "int64_t") return "int";
    if (token.value === "uint32_t" || token.value === "uint64_t" || token.value === "uintptr_t") return "uint";
    if (token.value === "curandState_t") return "uint";
    if (token.value === "cufftComplex") return "complex64";
    if (token.value === "cudaTextureObject_t") return "texture2d";
    if (token.value === "cudaSurfaceObject_t") return "surface2d";
    if (token.value === "cudaEvent_t" || token.value === "cudaStream_t") return "uint";
    if (token.value === "DevicePool") return "devicepool";
    if (token.value === "void") return "voidptr";
    if (token.value === "__half") return "half";
    const scalarAlias = CUDA_SCALAR_TYPE_ALIASES.get(token.value);
    if (scalarAlias !== undefined) return scalarAlias;
    const vectorAlias = cudaVectorTypeAlias(token.value);
    if (vectorAlias !== undefined) return vectorAlias;
    if (!TYPE_KEYWORDS.has(token.value)) this.fail(`unsupported CUDA-lite type: ${token.value}`, token.span);
    return token.value as Exclude<CudaLiteScalarType, "void">;
  }

  private parseReturnType(): CudaLiteScalarType {
    if (this.consumeIf("void")) return "void";
    return this.parseType();
  }

  private startsVarDecl(): boolean {
    if (this.match("__shared__") || this.match("extern")) return true;
    if (this.match("static")) return this.tokens[this.index + 1]?.value === "__shared__" ||
      this.tokens[this.index + 1]?.value === "constexpr" ||
      TYPE_START_KEYWORDS.has(this.tokens[this.index + 1]?.value ?? "");
    if (this.match("constexpr")) return TYPE_START_KEYWORDS.has(this.tokens[this.index + 1]?.value ?? "") ||
      (this.tokens[this.index + 1]?.value === "const" && TYPE_START_KEYWORDS.has(this.tokens[this.index + 2]?.value ?? ""));
    if (this.match("const")) return TYPE_START_KEYWORDS.has(this.tokens[this.index + 1]?.value ?? "");
    return TYPE_START_KEYWORDS.has(this.peek().value);
  }

  private consumeIntegerWidthSuffix(): void {
    if (this.consumeIf("int")) return;
    if (this.consumeIf("short")) {
      this.consumeIf("int");
      return;
    }
    if (this.consumeIf("long")) {
      this.consumeIf("long");
      return;
    }
  }

  private typeEndIndex(startIndex: number): number | undefined {
    let index = startIndex;
    if (this.tokens[index]?.value === "const") index++;
    const value = this.tokens[index]?.value;
    if (value === "unsigned" || value === "signed") {
      index++;
      if (this.tokens[index]?.value === "int") return index + 1;
      if (this.tokens[index]?.value === "char") return index + 1;
      if (this.tokens[index]?.value === "short") return this.tokens[index + 1]?.value === "int" ? index + 2 : index + 1;
      if (this.tokens[index]?.value === "long") return this.tokens[index + 1]?.value === "long" ? index + 2 : index + 1;
      return index;
    }
    if (value === "long") return this.tokens[index + 1]?.value === "long" ? index + 2 : index + 1;
    if (value === "short") return this.tokens[index + 1]?.value === "int" ? index + 2 : index + 1;
    if (value !== undefined && TYPE_START_KEYWORDS.has(value)) return index + 1;
    return undefined;
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
    const value = this.evaluateIntegerConstantExpression(expression);
    if (value === undefined) {
      this.fail("array size must be an integer constant expression", expression.span);
    }
    return value;
  }

  private parseTemplateIntegerArgument(): number {
    this.templateArgumentDepth++;
    let expression: CudaLiteExpression;
    try {
      expression = this.parseExpression();
    } finally {
      this.templateArgumentDepth--;
    }
    const value = this.evaluateIntegerConstantExpression(expression);
    if (value === undefined || !Number.isInteger(value) || value <= 0) {
      this.fail("template argument must be a positive integer", expression.span);
    }
    return value;
  }

  private evaluateIntegerConstantExpression(expression: CudaLiteExpression): number | undefined {
    return evaluateIntegerConstantExpression(expression, (name) => this.lookupIntegerConstant(name));
  }

  private recordIntegerConstant(name: string, value: number): void {
    this.integerConstantScopes.at(-1)?.set(name, value);
  }

  private lookupIntegerConstant(name: string): number | undefined {
    if (name === "true") return 1;
    if (name === "false") return 0;
    for (let index = this.integerConstantScopes.length - 1; index >= 0; index--) {
      const value = this.integerConstantScopes[index]?.get(name);
      if (value !== undefined) return value;
    }
    const named = CUDA_NAMED_CONSTANTS.get(name);
    if (named && Number.isSafeInteger(named.value)) return named.value;
    return undefined;
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

  private consumeTemplateArgumentsBeforeCall(): void {
    if (!this.match("<")) return;
    const endIndex = this.findTemplateArgumentEnd(this.index);
    if (endIndex === undefined) return;
    const next = this.tokens[endIndex + 1]?.value;
    if (next !== "(" && next !== "{") return;
    this.index = endIndex + 1;
  }

  private findTemplateArgumentEnd(startIndex: number): number | undefined {
    let depth = 0;
    for (let index = startIndex; index < this.tokens.length; index++) {
      const value = this.tokens[index]?.value;
      if (value === "<") {
        depth++;
        continue;
      }
      if (value === ">") {
        depth--;
      } else if (value === ">>") {
        depth -= 2;
      } else if (value === "<eof>" || value === ";" || value === "{") {
        return undefined;
      }
      if (depth === 0) return index;
      if (depth < 0) return undefined;
    }
    return undefined;
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

  private skipSimpleDeclaration(): void {
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

function evaluateIntegerConstantExpression(
  expression: CudaLiteExpression,
  lookupIdentifier: (name: string) => number | undefined,
): number | undefined {
  switch (expression.kind) {
    case "number":
      return Number.isInteger(expression.value) ? expression.value : undefined;
    case "identifier":
      return lookupIdentifier(expression.name);
    case "unary": {
      const value = evaluateIntegerConstantExpression(expression.argument, lookupIdentifier);
      if (value === undefined) return undefined;
      if (expression.operator === "+") return value;
      if (expression.operator === "-") return -value;
      return undefined;
    }
    case "binary": {
      const left = evaluateIntegerConstantExpression(expression.left, lookupIdentifier);
      const right = evaluateIntegerConstantExpression(expression.right, lookupIdentifier);
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
        case "<":
          return left < right ? 1 : 0;
        case "<=":
          return left <= right ? 1 : 0;
        case ">":
          return left > right ? 1 : 0;
        case ">=":
          return left >= right ? 1 : 0;
        case "==":
          return left === right ? 1 : 0;
        case "!=":
          return left !== right ? 1 : 0;
        case "&&":
          return left !== 0 && right !== 0 ? 1 : 0;
        case "||":
          return left !== 0 || right !== 0 ? 1 : 0;
        case "&":
          return left & right;
        case "|":
          return left | right;
        case "^":
          return left ^ right;
        default:
          return undefined;
      }
    }
    case "conditional": {
      const condition = evaluateIntegerConstantExpression(expression.condition, lookupIdentifier);
      if (condition === undefined) return undefined;
      return evaluateIntegerConstantExpression(condition !== 0 ? expression.consequent : expression.alternate, lookupIdentifier);
    }
    default:
      return undefined;
  }
}

function flattenInitializerExpressions(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  if (expression.kind !== "initializer") return [expression];
  return expression.elements.flatMap((element) => flattenInitializerExpressions(element));
}
