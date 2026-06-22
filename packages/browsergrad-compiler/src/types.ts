import type {
  WgslTexture2DInput,
  WgslKernelProgram,
  WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import type { CudaLoweringPlan } from "./compatibility.js";

export interface SourceSpan {
  readonly start: number;
  readonly end: number;
  readonly line: number;
  readonly column: number;
}

export type CudaLiteScalarType = "float" | "int" | "uint" | "half" | "bool" | "complex64" | "surface2d" | "void";
export type DiagnosticSeverity = "error" | "warning";

export interface CudaLiteDiagnostic {
  readonly code: string;
  readonly severity: DiagnosticSeverity;
  readonly message: string;
  readonly span: SourceSpan;
}

export class CudaLiteCompilerError extends Error {
  readonly diagnostics: readonly CudaLiteDiagnostic[];

  constructor(message: string, diagnostics: readonly CudaLiteDiagnostic[]) {
    super(message);
    this.name = "CudaLiteCompilerError";
    this.diagnostics = diagnostics;
  }
}

export interface CudaLiteModule {
  readonly kind: "module";
  readonly source: string;
  readonly constants: readonly CudaLiteGlobalConstant[];
  readonly textures: readonly CudaLiteTexture2D[];
  readonly functions: readonly CudaLiteDeviceFunction[];
  readonly kernels: readonly CudaLiteKernel[];
  readonly span: SourceSpan;
}

export interface CudaLiteGlobalConstant {
  readonly kind: "constant";
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly name: string;
  readonly dimensions: readonly number[];
  readonly span: SourceSpan;
}

export interface CudaLiteTexture2D {
  readonly kind: "texture2d";
  readonly valueType: "float";
  readonly name: string;
  readonly span: SourceSpan;
}

export interface CudaLiteKernel {
  readonly kind: "kernel";
  readonly name: string;
  readonly params: readonly CudaLiteParam[];
  readonly body: readonly CudaLiteStatement[];
  readonly span: SourceSpan;
}

export interface CudaLiteDeviceFunction {
  readonly kind: "device-function";
  readonly name: string;
  readonly returnType: CudaLiteScalarType;
  readonly params: readonly CudaLiteParam[];
  readonly body: readonly CudaLiteStatement[];
  readonly span: SourceSpan;
}

export interface CudaLiteParam {
  readonly name: string;
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly pointer: boolean;
  readonly constant: boolean;
  readonly span: SourceSpan;
}

export type CudaLiteStatement =
  | CudaLiteVarDecl
  | CudaLiteDim3Decl
  | CudaLiteCooperativeGroupDecl
  | CudaLiteKernelLaunchStatement
  | CudaLiteAsmStatement
  | CudaLiteIfStatement
  | CudaLiteForStatement
  | CudaLiteExprStatement
  | CudaLiteReturnStatement
  | CudaLiteContinueStatement;

export interface CudaLiteVarDecl {
  readonly kind: "var";
  readonly storage: "local" | "shared";
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly pointer: boolean;
  readonly name: string;
  readonly dimensions: readonly number[];
  readonly dynamicShared?: boolean;
  readonly init?: CudaLiteExpression;
  readonly span: SourceSpan;
}

export interface CudaLiteDim3Decl {
  readonly kind: "dim3";
  readonly name: string;
  readonly span: SourceSpan;
}

export type CudaLiteCooperativeGroupKind = "block" | "grid" | "tile";

export interface CudaLiteCooperativeGroupDecl {
  readonly kind: "cooperative-group";
  readonly groupKind: CudaLiteCooperativeGroupKind;
  readonly name: string;
  readonly tileSize?: number;
  readonly span: SourceSpan;
}

export interface CudaLiteKernelLaunchStatement {
  readonly kind: "kernel-launch";
  readonly callee: string;
  readonly span: SourceSpan;
}

export interface CudaLiteAsmStatement {
  readonly kind: "asm";
  readonly template: string;
  readonly output: CudaLiteExpression;
  readonly inputs: readonly CudaLiteExpression[];
  readonly span: SourceSpan;
}

export interface CudaLiteIfStatement {
  readonly kind: "if";
  readonly condition: CudaLiteExpression;
  readonly consequent: readonly CudaLiteStatement[];
  readonly alternate?: readonly CudaLiteStatement[];
  readonly span: SourceSpan;
}

export interface CudaLiteForStatement {
  readonly kind: "for";
  readonly init?: CudaLiteVarDecl | CudaLiteExpression;
  readonly condition?: CudaLiteExpression;
  readonly update?: CudaLiteExpression;
  readonly body: readonly CudaLiteStatement[];
  readonly span: SourceSpan;
}

export interface CudaLiteExprStatement {
  readonly kind: "expr";
  readonly expression: CudaLiteExpression;
  readonly span: SourceSpan;
}

export interface CudaLiteReturnStatement {
  readonly kind: "return";
  readonly value?: CudaLiteExpression;
  readonly span: SourceSpan;
}

export interface CudaLiteContinueStatement {
  readonly kind: "continue";
  readonly span: SourceSpan;
}

export type CudaLiteExpression =
  | CudaLiteNumberLiteral
  | CudaLiteStringLiteral
  | CudaLiteIdentifier
  | CudaLiteCastExpression
  | CudaLiteMemberExpression
  | CudaLiteIndexExpression
  | CudaLiteCallExpression
  | CudaLiteUnaryExpression
  | CudaLiteBinaryExpression
  | CudaLiteConditionalExpression
  | CudaLiteAssignmentExpression
  | CudaLiteUpdateExpression;

export interface CudaLiteNumberLiteral {
  readonly kind: "number";
  readonly value: number;
  readonly raw: string;
  readonly span: SourceSpan;
}

export interface CudaLiteStringLiteral {
  readonly kind: "string";
  readonly value: string;
  readonly raw: string;
  readonly span: SourceSpan;
}

export interface CudaLiteIdentifier {
  readonly kind: "identifier";
  readonly name: string;
  readonly span: SourceSpan;
}

export interface CudaLiteCastExpression {
  readonly kind: "cast";
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly expression: CudaLiteExpression;
  readonly span: SourceSpan;
}

export interface CudaLiteMemberExpression {
  readonly kind: "member";
  readonly object: CudaLiteExpression;
  readonly property: string;
  readonly span: SourceSpan;
}

export interface CudaLiteIndexExpression {
  readonly kind: "index";
  readonly target: CudaLiteExpression;
  readonly index: CudaLiteExpression;
  readonly span: SourceSpan;
}

export interface CudaLiteCallExpression {
  readonly kind: "call";
  readonly callee: CudaLiteExpression;
  readonly args: readonly CudaLiteExpression[];
  readonly span: SourceSpan;
}

export interface CudaLiteUnaryExpression {
  readonly kind: "unary";
  readonly operator: "-" | "+" | "!" | "&";
  readonly argument: CudaLiteExpression;
  readonly span: SourceSpan;
}

export interface CudaLiteBinaryExpression {
  readonly kind: "binary";
  readonly operator:
    | "+"
    | "-"
    | "*"
    | "/"
    | "%"
    | "<<"
    | ">>"
    | "&"
    | "^"
    | "|"
    | "<"
    | "<="
    | ">"
    | ">="
    | "=="
    | "!="
    | "&&"
    | "||";
  readonly left: CudaLiteExpression;
  readonly right: CudaLiteExpression;
  readonly span: SourceSpan;
}

export interface CudaLiteConditionalExpression {
  readonly kind: "conditional";
  readonly condition: CudaLiteExpression;
  readonly consequent: CudaLiteExpression;
  readonly alternate: CudaLiteExpression;
  readonly span: SourceSpan;
}

export interface CudaLiteAssignmentExpression {
  readonly kind: "assignment";
  readonly operator: "=" | "+=" | "-=" | "*=" | "/=" | "<<=" | ">>=";
  readonly left: CudaLiteExpression;
  readonly right: CudaLiteExpression;
  readonly span: SourceSpan;
}

export interface CudaLiteUpdateExpression {
  readonly kind: "update";
  readonly operator: "++" | "--";
  readonly argument: CudaLiteExpression;
  readonly prefix: boolean;
  readonly span: SourceSpan;
}

export interface CudaLiteAnalyzeOptions {
  readonly kernelName?: string;
  readonly features?: Partial<Record<"shader-f16" | "subgroups" | "compatibility", boolean>>;
  readonly workgroupSize?: readonly [number, number, number];
  readonly dynamicSharedMemory?: Readonly<Record<string, number>>;
}

export interface CudaLiteAnalysis {
  readonly kernel: CudaLiteKernel;
  readonly constants: readonly CudaLiteGlobalConstant[];
  readonly textures: readonly CudaLiteTexture2D[];
  readonly functions: readonly CudaLiteDeviceFunction[];
  readonly diagnostics: readonly CudaLiteDiagnostic[];
  readonly requiredFeatures: readonly string[];
  readonly atomicParams: readonly string[];
  readonly atomicShared: readonly string[];
}

export interface KernelIrModule {
  readonly name: string;
  readonly params: readonly CudaLiteParam[];
  readonly constants: readonly CudaLiteGlobalConstant[];
  readonly textures: readonly CudaLiteTexture2D[];
  readonly functions: readonly CudaLiteDeviceFunction[];
  readonly body: readonly CudaLiteStatement[];
  readonly sharedDeclarations: readonly CudaLiteVarDecl[];
  readonly requiredFeatures: readonly string[];
  readonly atomicParams: readonly string[];
  readonly atomicShared: readonly string[];
  readonly workgroupSize: readonly [number, number, number];
}

export interface CompileCudaLiteOptions extends CudaLiteAnalyzeOptions {}

export interface CompiledCudaLiteKernel {
  readonly ast: CudaLiteModule;
  readonly analysis: CudaLiteAnalysis;
  readonly ir: KernelIrModule;
  readonly wgsl: string;
  readonly wgslProgram: WgslKernelProgram;
  readonly diagnostics: readonly CudaLiteDiagnostic[];
  readonly loweringPlan: CudaLoweringPlan;
}

export interface KernelLaunch {
  readonly gridDim: readonly [number, number, number];
  readonly blockDim: readonly [number, number, number];
}

export interface CompiledKernelInput {
  readonly buffers: Readonly<Record<string, WgslTypedArray>>;
  readonly constants?: Readonly<Record<string, number | WgslTypedArray>>;
  readonly textures?: Readonly<Record<string, WgslTexture2DInput>>;
  readonly surfaces?: Readonly<Record<string, WgslTexture2DInput>>;
  readonly scalars?: Readonly<Record<string, number>>;
  readonly readback?: readonly string[];
}

export interface KernelMemoryAccess {
  readonly name: string;
  readonly index: number;
  readonly value: number;
  readonly ok: boolean;
}

export interface KernelThreadTrace {
  readonly blockIdx: readonly [number, number, number];
  readonly threadIdx: readonly [number, number, number];
  readonly reads: readonly KernelMemoryAccess[];
  readonly writes: readonly KernelMemoryAccess[];
  readonly sharedReads: readonly KernelMemoryAccess[];
  readonly sharedWrites: readonly KernelMemoryAccess[];
}

export interface ReferenceKernelResult {
  readonly buffers: Readonly<Record<string, WgslTypedArray>>;
  readonly trace: readonly KernelThreadTrace[];
}
