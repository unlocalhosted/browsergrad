import type {
  WgslTexture2DInput,
  WgslKernelProgram,
  WgslResidentBuffer,
  WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import type { CudaLoweringPlan } from "./compatibility.js";

export interface SourceSpan {
  readonly start: number;
  readonly end: number;
  readonly line: number;
  readonly column: number;
}

export type CudaLiteScalarType =
  | "float"
  | "double"
  | "int"
  | "uint"
  | "half"
  | "half2"
  | "bf16"
  | "bf162"
  | "bool"
  | "complex64"
  | "float2"
  | "float3"
  | "float4"
  | "int2"
  | "int3"
  | "int4"
  | "uint2"
  | "uint3"
  | "uint4"
  | "texture2d"
  | "surface2d"
  | "devicepool"
  | "voidptr"
  | "void";
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
  readonly deviceGlobals: readonly CudaLiteDeviceGlobal[];
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
  readonly init?: CudaLiteExpression;
  readonly span: SourceSpan;
}

export interface CudaLiteDeviceGlobal {
  readonly kind: "device-global";
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly name: string;
  readonly dimensions: readonly number[];
  readonly init?: CudaLiteExpression;
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
  readonly cooperativeGroupKind?: CudaLiteCooperativeGroupKind;
  readonly tileSize?: number;
  readonly span: SourceSpan;
}

export type CudaLiteStatement =
  | CudaLiteVarDecl
  | CudaLiteBlockStatement
  | CudaLiteDim3Decl
  | CudaLiteCooperativeGroupDecl
  | CudaLiteKernelLaunchStatement
  | CudaLiteAsmStatement
  | CudaLiteIfStatement
  | CudaLiteForStatement
  | CudaLiteWhileStatement
  | CudaLiteExprStatement
  | CudaLiteReturnStatement
  | CudaLiteContinueStatement
  | CudaLiteBreakStatement;

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

export interface CudaLiteBlockStatement {
  readonly kind: "block";
  readonly body: readonly CudaLiteStatement[];
  readonly span: SourceSpan;
}

export interface CudaLiteDim3Decl {
  readonly kind: "dim3";
  readonly name: string;
  readonly args: readonly CudaLiteExpression[];
  readonly span: SourceSpan;
}

export type CudaLiteCooperativeGroupKind = "block" | "grid" | "tile" | "thread";

export interface CudaLiteCooperativeGroupDecl {
  readonly kind: "cooperative-group";
  readonly groupKind: CudaLiteCooperativeGroupKind;
  readonly name: string;
  readonly tileSize?: number;
  readonly dynamicTileSizeName?: string;
  readonly span: SourceSpan;
}

export interface CudaLiteKernelLaunchStatement {
  readonly kind: "kernel-launch";
  readonly callee: string;
  readonly grid: readonly CudaLiteExpression[];
  readonly block: readonly CudaLiteExpression[];
  readonly args: readonly CudaLiteExpression[];
  readonly span: SourceSpan;
}

export interface CudaLiteAsmStatement {
  readonly kind: "asm";
  readonly template: string;
  readonly output?: CudaLiteExpression;
  readonly outputs?: readonly CudaLiteExpression[];
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

export interface CudaLiteWhileStatement {
  readonly kind: "while";
  readonly condition: CudaLiteExpression;
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

export interface CudaLiteBreakStatement {
  readonly kind: "break";
  readonly span: SourceSpan;
}

export type CudaLiteExpression =
  | CudaLiteNumberLiteral
  | CudaLiteStringLiteral
  | CudaLiteInitializerExpression
  | CudaLiteIdentifier
  | CudaLiteCastExpression
  | CudaLiteMemberExpression
  | CudaLiteIndexExpression
  | CudaLiteCallExpression
  | CudaLiteUnaryExpression
  | CudaLiteBinaryExpression
  | CudaLiteConditionalExpression
  | CudaLiteSequenceExpression
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

export interface CudaLiteInitializerExpression {
  readonly kind: "initializer";
  readonly elements: readonly CudaLiteExpression[];
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
  readonly pointer?: boolean;
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
  readonly templateValueType?: Exclude<CudaLiteScalarType, "void">;
  readonly span: SourceSpan;
}

export interface CudaLiteUnaryExpression {
  readonly kind: "unary";
  readonly operator: "-" | "+" | "!" | "~" | "&" | "*";
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

export interface CudaLiteSequenceExpression {
  readonly kind: "sequence";
  readonly expressions: readonly CudaLiteExpression[];
  readonly span: SourceSpan;
}

export interface CudaLiteAssignmentExpression {
  readonly kind: "assignment";
  readonly operator: "=" | "+=" | "-=" | "*=" | "/=" | "<<=" | ">>=" | "&=" | "|=" | "^=";
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

export type CudaLiteFeatureName = "shader-f16" | "subgroups" | "compatibility";
export type CudaLiteFeatureOptions = Partial<Record<CudaLiteFeatureName, boolean>>;

export interface CudaLiteAnalyzeOptions {
  readonly kernelName?: string;
  readonly features?: CudaLiteFeatureOptions;
  readonly workgroupSize?: readonly [number, number, number];
  readonly dynamicSharedMemory?: Readonly<Record<string, number>>;
  readonly referenceDynamicParallelism?: boolean;
  readonly referenceGridSync?: boolean;
  readonly referenceCudaRuntime?: boolean;
  readonly f64Mode?: "reject" | "f32";
}

export interface CudaLiteAnalysis {
  readonly kernel: CudaLiteKernel;
  readonly constants: readonly CudaLiteGlobalConstant[];
  readonly deviceGlobals: readonly CudaLiteDeviceGlobal[];
  readonly textures: readonly CudaLiteTexture2D[];
  readonly functions: readonly CudaLiteDeviceFunction[];
  readonly diagnostics: readonly CudaLiteDiagnostic[];
  readonly requiredFeatures: readonly string[];
  readonly atomicParams: readonly string[];
  readonly atomicShared: readonly string[];
  readonly atomicDeviceGlobals: readonly string[];
}

export interface KernelIrModule {
  readonly name: string;
  readonly params: readonly CudaLiteParam[];
  readonly constants: readonly CudaLiteGlobalConstant[];
  readonly deviceGlobals: readonly CudaLiteDeviceGlobal[];
  readonly textures: readonly CudaLiteTexture2D[];
  readonly functions: readonly CudaLiteDeviceFunction[];
  readonly body: readonly CudaLiteStatement[];
  readonly sharedDeclarations: readonly CudaLiteVarDecl[];
  readonly requiredFeatures: readonly string[];
  readonly atomicParams: readonly string[];
  readonly atomicShared: readonly string[];
  readonly atomicDeviceGlobals: readonly string[];
  readonly workgroupSize: readonly [number, number, number];
}

export interface CompileCudaLiteOptions extends CudaLiteAnalyzeOptions {
  readonly pointerBaseOffsets?: Readonly<Record<string, number>>;
}

export interface CompiledCudaLiteKernel {
  readonly ast: CudaLiteModule;
  readonly analysis: CudaLiteAnalysis;
  readonly ir: KernelIrModule;
  readonly wgsl: string;
  readonly wgslProgram: WgslKernelProgram;
  readonly diagnostics: readonly CudaLiteDiagnostic[];
  readonly loweringPlan: CudaLoweringPlan;
  readonly pointerBaseOffsets?: Readonly<Record<string, number>>;
}

export interface KernelLaunch {
  readonly gridDim: readonly [number, number, number];
  readonly blockDim: readonly [number, number, number];
}

export interface CompiledKernelInput {
  readonly buffers: Readonly<Record<string, WgslTypedArray>>;
  readonly residentBuffers?: Readonly<Record<string, WgslResidentBuffer>>;
  readonly constants?: Readonly<Record<string, number | WgslTypedArray>>;
  readonly deviceGlobals?: Readonly<Record<string, WgslTypedArray>>;
  readonly textures?: Readonly<Record<string, WgslTexture2DInput>>;
  readonly surfaces?: Readonly<Record<string, WgslTexture2DInput>>;
  readonly memoryPools?: Readonly<Record<string, CudaLiteMemoryPoolInput>>;
  readonly scalars?: Readonly<Record<string, number>>;
  readonly readback?: readonly string[];
}

export interface CudaLiteMemoryPoolInput {
  readonly data: Uint32Array;
  readonly offset?: Uint32Array;
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
