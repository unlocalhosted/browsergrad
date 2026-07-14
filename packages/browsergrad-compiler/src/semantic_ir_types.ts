import type { CudaLiteAnalysis, CudaLiteParam, CudaLiteScalarType, KernelLaunch, SourceSpan } from "./types.js";
import type { SemanticValueType } from "./semantic_value_type.js";
import type { SemanticTextureReadCall } from "./semantic_texture_surface.js";
import type { InlineAsmOp } from "./features/inline_ptx/model.js";
import type { CanonicalIr, TypedSemantic } from "./compiler_phases.js";
import type { SemanticFunctionId, SemanticMemoryId, SemanticSymbolId } from "./semantic_ids.js";
import type { SemanticEnvironment } from "./semantic_environment.js";
import type { MatrixTileLayout, MatrixTileResolvedSpec } from "./matrix_tiles.js";

export type SemanticAddressSpace =
  | "uniform"
  | "storage"
  | "constant"
  | "device-global"
  | "texture"
  | "surface"
  | "shared"
  | "local"
  | "pool"
  | "function"
  | "builtin"
  | "unknown";

export interface CudaLiteSemanticSymbol {
  readonly id: SemanticSymbolId;
  readonly name: string;
  readonly kind:
    | "param"
    | "local"
    | "shared"
    | "constant"
    | "device-global"
    | "external-pool"
    | "texture"
    | "function"
    | "builtin";
  readonly valueType?: CudaLiteScalarType;
  readonly pointer?: boolean;
  readonly pointerRuntimeState?: boolean;
  readonly pointerMayBeNull?: boolean;
  readonly pointerRoot?: SemanticMemoryId;
  readonly pointerMemoryAlias?: SemanticMemoryId;
  readonly pointerParamAlias?: SemanticSymbolId;
  readonly pointerAddressSpace?: SemanticAddressSpace;
  readonly pointerBaseIndices?: readonly SemanticExpression[];
  readonly pointerBaseIsScalarLane?: boolean;
  readonly pointerBaseUnitBytes?: number;
  readonly pointerValid?: SemanticExpression;
  readonly pointerSelection?: SemanticPointerSelection;
  readonly pointerArrayAliases?: readonly (SemanticPointerAlias | undefined)[];
  readonly pointerCarrierValueType?: CudaLiteScalarType;
  readonly packedByteLanes?: 2 | 3 | 4;
  readonly cooperativeGroupKind?: CudaLiteParam["cooperativeGroupKind"];
  readonly tileSize?: number;
  readonly constant?: boolean;
  readonly initialized?: boolean;
  readonly init?: SemanticExpression;
  readonly dimensions: readonly number[];
  readonly dynamicShared?: boolean;
  readonly matrixTile?: MatrixTileResolvedSpec;
  readonly matrixTileArrayDimensions?: readonly number[];
  readonly addressSpace: SemanticAddressSpace;
  readonly span: SourceSpan;
}

export interface SemanticPointerAlias {
  readonly pointerRoot?: SemanticMemoryId;
  readonly pointerAddressSpace?: SemanticAddressSpace;
  readonly pointerBaseIndices?: readonly SemanticExpression[];
  readonly pointerBaseIsScalarLane?: boolean;
  readonly pointerBaseUnitBytes?: number;
  readonly pointerValid?: SemanticExpression;
  readonly pointerSelection?: SemanticPointerSelection;
}

export interface SemanticPointerSelection {
  readonly condition: SemanticExpression;
  readonly consequent: SemanticPointerAlias;
  readonly alternate: SemanticPointerAlias;
}

export interface CudaLiteSemanticFunction {
  readonly id: SemanticFunctionId;
  readonly name: string;
  readonly returnType: CudaLiteScalarType;
  readonly params: readonly CudaLiteSemanticSymbol[];
  readonly body: readonly SemanticKernelIrOperation[];
  readonly span: SourceSpan;
}

export interface SemanticCooperativeGroupDeclaration {
  readonly kind: "cooperative-group";
  readonly id: SemanticSymbolId;
  readonly groupKind: "thread" | "block" | "grid" | "tile" | "coalesced" | "binary";
  readonly name: string;
  readonly tileSize?: number;
  readonly partitionParent?: string;
  readonly partitionPredicate?: SemanticExpression;
  readonly span: SourceSpan;
}

export interface CudaLiteSemanticLaunchableEntry {
  readonly id: SemanticFunctionId;
  readonly kind: "kernel" | "device-function";
  readonly name: string;
  readonly params: readonly CudaLiteSemanticSymbol[];
  readonly span: SourceSpan;
}

export interface CudaLiteSemanticModel {
  readonly kind: "cuda-lite-semantic-model";
  readonly kernelName: string;
  readonly span: SourceSpan;
  readonly params: readonly CudaLiteSemanticSymbol[];
  readonly symbols: readonly CudaLiteSemanticSymbol[];
  readonly functions: readonly CudaLiteSemanticFunction[];
  readonly launchableEntries: readonly CudaLiteSemanticLaunchableEntry[];
  readonly requiredFeatures: readonly string[];
  readonly environment: SemanticEnvironment;
}

export type TypedCudaLiteSemanticModel = TypedSemantic<CudaLiteSemanticModel>;

export interface SemanticMemoryRef {
  readonly baseId: SemanticMemoryId;
  readonly base: string;
  readonly addressSpace: SemanticAddressSpace;
  readonly valueType: SemanticValueType;
  readonly containerValueType?: CudaLiteScalarType;
  readonly pointerBaseIsScalarLane?: boolean;
  readonly pointerBaseUnitBytes?: number;
  readonly packedByteLanes?: 2 | 3 | 4;
  readonly indices: readonly SemanticExpression[];
  readonly fields: readonly string[];
  readonly span: SourceSpan;
}

export type SemanticPoolRef =
  | {
      readonly kind: "device-pool";
      readonly id: SemanticMemoryId;
      readonly name: string;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "raw-pool";
      readonly data: SemanticMemoryRef;
      readonly offset: SemanticMemoryRef;
      readonly capacityBytes: SemanticExpression;
      readonly span: SourceSpan;
    };

export interface SemanticMatrixTileRef {
  readonly baseId: SemanticMemoryId;
  readonly base: string;
  readonly spec: MatrixTileResolvedSpec;
  readonly arrayDimensions: readonly number[];
  readonly indices: readonly SemanticExpression[];
  readonly span: SourceSpan;
}

export type SemanticExpression =
  | {
      readonly kind: "literal";
      readonly literalKind: "number";
      readonly value: number;
      readonly valueType: SemanticValueType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "literal";
      readonly literalKind: "string";
      readonly value: string;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "symbol";
      readonly id: SemanticSymbolId;
      readonly name: string;
      readonly valueType?: CudaLiteScalarType;
      readonly addressSpace: SemanticAddressSpace;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "pointer-valid";
      readonly pointerId: SemanticSymbolId;
      readonly pointer: string;
      readonly valueType: "bool";
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "member";
      readonly object: SemanticExpression;
      readonly property: string;
      readonly valueType: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "index";
      readonly target: SemanticExpression;
      readonly index: SemanticExpression;
      readonly valueType: SemanticValueType;
      readonly addressSpace: SemanticAddressSpace;
      readonly pointerBaseIsScalarLane?: boolean;
      readonly pointerBaseUnitBytes?: number;
      readonly packedByteLanes?: 2 | 3 | 4;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "call";
      readonly callee: SemanticExpression;
      readonly args: readonly SemanticExpression[];
      readonly templateValueType?: Exclude<CudaLiteScalarType, "void">;
      readonly valueType: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "texture-read";
      readonly callee: SemanticTextureReadCall;
      readonly texture: SemanticExpression;
      readonly x: SemanticExpression;
      readonly y: SemanticExpression;
      readonly z?: SemanticExpression;
  readonly valueType: SemanticValueType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "surface-read";
      readonly callee: "surf1Dread" | "surf2Dread" | "surf2DLayeredread" | "surf3Dread";
      readonly surface: SemanticExpression;
      readonly xBytes: SemanticExpression;
      readonly y: SemanticExpression;
      readonly z?: SemanticExpression;
      readonly valueType: Exclude<CudaLiteScalarType, "void">;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "cast";
      readonly valueType: Exclude<CudaLiteScalarType, "void">;
      readonly pointer: boolean;
      readonly packedByteLanes?: 2 | 3 | 4;
      readonly expression: SemanticExpression;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "unary";
      readonly operator: string;
      readonly argument: SemanticExpression;
      readonly valueType: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "binary";
      readonly operator: string;
      readonly left: SemanticExpression;
      readonly right: SemanticExpression;
      readonly valueType: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "conditional";
      readonly condition: SemanticExpression;
      readonly consequent: SemanticExpression;
      readonly alternate: SemanticExpression;
      readonly valueType: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "assignment";
      readonly operator: string;
      readonly target: SemanticExpression;
      readonly value: SemanticExpression;
      readonly valueType: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "update";
      readonly operator: string;
      readonly argument: SemanticExpression;
      readonly prefix: boolean;
      readonly valueType: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "initializer";
      readonly elements: readonly SemanticExpression[];
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "sequence";
      readonly expressions: readonly SemanticExpression[];
      readonly valueType: CudaLiteScalarType;
      readonly span: SourceSpan;
    };

export type SemanticKernelIrOperation =
  | { readonly kind: "declare"; readonly target: CudaLiteSemanticSymbol; readonly init?: SemanticExpression; readonly span: SourceSpan }
  | { readonly kind: "dim3-declare"; readonly target: CudaLiteSemanticSymbol; readonly args: readonly SemanticExpression[]; readonly span: SourceSpan }
  | { readonly kind: "cooperative-group-declare"; readonly declaration: SemanticCooperativeGroupDeclaration; readonly span: SourceSpan }
  | { readonly kind: "load"; readonly source: SemanticMemoryRef; readonly span: SourceSpan }
  | { readonly kind: "store"; readonly target: SemanticMemoryRef; readonly value: SemanticExpression; readonly operator: string; readonly reads: readonly SemanticMemoryRef[]; readonly span: SourceSpan }
  | { readonly kind: "copy"; readonly source: SemanticMemoryRef; readonly target: SemanticMemoryRef; readonly bytes: number; readonly span: SourceSpan }
  | { readonly kind: "copy-fence"; readonly callee: string; readonly span: SourceSpan }
  | { readonly kind: "matrix-fill"; readonly fragment: SemanticMatrixTileRef; readonly value: SemanticExpression; readonly span: SourceSpan }
  | { readonly kind: "matrix-load"; readonly fragment: SemanticMatrixTileRef; readonly source: SemanticMemoryRef; readonly stride: SemanticExpression; readonly layout: MatrixTileLayout; readonly span: SourceSpan }
  | { readonly kind: "matrix-mma"; readonly destination: SemanticMatrixTileRef; readonly a: SemanticMatrixTileRef; readonly b: SemanticMatrixTileRef; readonly accumulator: SemanticMatrixTileRef; readonly span: SourceSpan }
  | { readonly kind: "matrix-store"; readonly target: SemanticMemoryRef; readonly fragment: SemanticMatrixTileRef; readonly stride: SemanticExpression; readonly layout: MatrixTileLayout; readonly span: SourceSpan }
  | { readonly kind: "surface-write"; readonly surface: SemanticExpression; readonly value: SemanticExpression; readonly xBytes: SemanticExpression; readonly y: SemanticExpression; readonly z?: SemanticExpression; readonly span: SourceSpan }
  | { readonly kind: "surface-read-store"; readonly target: SemanticExpression; readonly surface: SemanticExpression; readonly xBytes: SemanticExpression; readonly y: SemanticExpression; readonly z?: SemanticExpression; readonly valueType?: CudaLiteScalarType; readonly span: SourceSpan }
  | { readonly kind: "atomic"; readonly callee: string; readonly target?: SemanticMemoryRef; readonly args: readonly SemanticExpression[]; readonly span: SourceSpan }
  | { readonly kind: "call"; readonly calleeId: SemanticSymbolId; readonly callee: string; readonly args: readonly SemanticExpression[]; readonly reads: readonly SemanticMemoryRef[]; readonly result?: Extract<SemanticExpression, { readonly kind: "symbol" }>; readonly span: SourceSpan }
  | { readonly kind: "runtime-copy"; readonly callee: string; readonly args: readonly SemanticExpression[]; readonly span: SourceSpan }
  | { readonly kind: "pool-allocate"; readonly allocator: "deviceAllocate" | "streamOrderedAllocate"; readonly target: CudaLiteSemanticSymbol; readonly pool: SemanticPoolRef; readonly sizeBytes: SemanticExpression; readonly span: SourceSpan }
  | { readonly kind: "pointer-rebind"; readonly target: CudaLiteSemanticSymbol; readonly source: SemanticMemoryRef; readonly span: SourceSpan }
  | { readonly kind: "expression"; readonly expression: SemanticExpression; readonly span: SourceSpan }
  | { readonly kind: "branch"; readonly condition: SemanticExpression; readonly consequent: readonly SemanticKernelIrOperation[]; readonly alternate: readonly SemanticKernelIrOperation[]; readonly conditionUniformity?: "workgroup"; readonly span: SourceSpan }
  | { readonly kind: "loop"; readonly loopKind: "for" | "while" | "do-while"; readonly init?: SemanticKernelIrOperation | SemanticExpression; readonly condition?: SemanticExpression; readonly update?: SemanticExpression; readonly body: readonly SemanticKernelIrOperation[]; readonly continuing?: readonly SemanticKernelIrOperation[]; readonly span: SourceSpan }
  | { readonly kind: "barrier"; readonly callee: string; readonly scope: "subgroup" | "workgroup" | "grid"; readonly groupName?: string; readonly span: SourceSpan }
  | { readonly kind: "fence"; readonly callee: string; readonly span: SourceSpan }
  | { readonly kind: "device-launch"; readonly launch: SemanticDeviceLaunch; readonly span: SourceSpan }
  | {
      readonly kind: "inline-asm";
      readonly op?: InlineAsmOp;
      readonly outputs: readonly SemanticExpression[];
      readonly inputs: readonly SemanticExpression[];
      readonly span: SourceSpan;
    }
  | { readonly kind: "return"; readonly value?: SemanticExpression; readonly span: SourceSpan }
  | { readonly kind: "continue"; readonly span: SourceSpan }
  | { readonly kind: "break"; readonly span: SourceSpan }
  | { readonly kind: "block"; readonly body: readonly SemanticKernelIrOperation[]; readonly span: SourceSpan };

export interface SemanticDeviceLaunch {
  readonly calleeId: SemanticFunctionId;
  readonly callee: string;
  readonly grid: readonly SemanticExpression[];
  readonly block: readonly SemanticExpression[];
  readonly args: readonly SemanticExpression[];
}

export interface SemanticKernelIrModule {
  readonly kind: "semantic-kernel-ir";
  readonly name: string;
  readonly span: SourceSpan;
  readonly symbols: readonly CudaLiteSemanticSymbol[];
  readonly params: readonly CudaLiteSemanticSymbol[];
  readonly memory: readonly CudaLiteSemanticSymbol[];
  readonly functions: readonly CudaLiteSemanticFunction[];
  readonly launchableEntries: readonly CudaLiteSemanticLaunchableEntry[];
  readonly operations: readonly SemanticKernelIrOperation[];
  readonly requiredFeatures: readonly string[];
  readonly barrierUniformity: CudaLiteAnalysis["barrierUniformity"];
  readonly workgroupSize: KernelLaunch["blockDim"];
  readonly subgroupMode?: "native" | "scalar";
  readonly bindlessTextures?: readonly string[];
}

export type CanonicalSemanticKernelIr = CanonicalIr<SemanticKernelIrModule>;
