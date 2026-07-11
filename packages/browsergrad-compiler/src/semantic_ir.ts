import type {
  CudaLiteAnalysis,
  CudaLiteAsmStatement,
  CudaLiteAssignmentExpression,
  CudaLiteCallExpression,
  CudaLiteCooperativeGroupDecl,
  CudaLiteExpression,
  CudaLiteKernelLaunchStatement,
  CudaLiteParam,
  CudaLiteScalarType,
  CudaLiteStatement,
  CudaLiteTexture2D,
  CudaLiteVarDecl,
  CudaLiteDeviceFunction,
  CudaLiteGlobalConstant,
  CudaLiteDeviceGlobal,
  KernelLaunch,
  SourceSpan,
} from "./types.js";
import { walkCudaLiteExpressions } from "./ast_queries.js";
import { cudaBuiltinVectorMemberValueType } from "./cuda_builtin_symbols.js";
import {
  cudaLiteDimensionStride as dimensionStride,
  cudaLiteTotalElements as totalElements,
} from "./cuda_lite_values.js";
import { cudaBfloat16IntrinsicReturnType } from "./cuda_bfloat16_intrinsics.js";
import {
  isCudaSemanticSurfaceWriteCallName,
} from "./cuda_texture_surface_calls.js";
import {
  isSemanticTextureReadCall,
  semanticTextureReadCoordinateCount,
  type SemanticTextureReadCall,
} from "./semantic_texture_surface.js";
import {
  cudaVibMinMaxInfo,
  isCudaFrexpCallName as isFrexpCallName,
  isCudaModfCallName as isModfCallName,
  isCudaRemquoCallName as isRemquoCallName,
  isCudaSincosCallName as isSincosCallName,
  isCudaSincosPiCallName as isSincosPiCallName,
  isCudaVibMinMaxCallName as isVibMinMaxCallName,
} from "./cuda_math_calls.js";
import {
  isCudaAddressSpacePredicateCallName as isAddressSpacePredicateName,
  isCudaPointerIdentityCallName,
} from "./cuda_pointer_calls.js";
import {
  isCudaCpAsyncCopyCall,
  isCudaCpAsyncFenceCall,
} from "./cuda_cp_async.js";
import {
  CUDA_BARRIER_CALL_NAMES,
  CUDA_COOPERATIVE_BARRIER_CALL_NAMES,
  CUDA_FENCE_CALL_NAMES,
} from "./cuda_sync_calls.js";
import {
  isCudaArithmeticReduceCallName,
  isCudaBitwiseReduceCallName,
  isCudaLegacyShuffleCallName as legacyShuffleCall,
  isCudaShuffleCallName,
  isCudaVoteCallName,
} from "./cuda_subgroup_calls.js";
import { CUDA_CACHE_HINT_LOADS, CUDA_CACHE_HINT_STORES } from "./intrinsics.js";
import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";
import {
  classifyInlineAsm,
  expectedInlineAsmF32SourceInputs,
  type InlineAsmF32Source,
  type PtxSpecialU32Register,
} from "./features/inline_ptx/model.js";
import { alignofCudaType, sizeofCudaType } from "./type_layout.js";
import { cudaVectorConstructorType, cudaVectorLaneCount, cudaVectorScalarType, cudaVectorSwizzleType, isCudaVectorType } from "./vector_types.js";
import { SEMANTIC_LOCAL_ARRAY_FILL_CALLS } from "./semantic_builtin_calls.js";
import { SEMANTIC_CURAND_CALLS } from "./semantic_curand_intrinsics.js";
import {
  isSemanticGeneratedRandomCall,
  semanticGeneratedRandomReturnType,
} from "./semantic_generated_random_intrinsics.js";
import { semanticPointerArgumentMemoryRef as semanticIrPointerArgumentMemoryRef } from "./semantic_pointer_arguments.js";
import { resolveSemanticFunctionOverloads } from "./semantic_function_overloads.js";
import { semanticVectorMathReturnType } from "./semantic_vector_math.js";
import { semanticStorageVectorFieldIndices } from "./semantic_value_types.js";

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
  readonly name: string;
  readonly kind:
    | "param"
    | "local"
    | "shared"
    | "constant"
    | "device-global"
    | "texture"
    | "function"
    | "builtin";
  readonly valueType?: CudaLiteScalarType;
  readonly pointer?: boolean;
  readonly pointerRoot?: string;
  readonly pointerAliasOf?: string;
  readonly pointerAddressSpace?: SemanticAddressSpace;
  readonly pointerBaseIndices?: readonly SemanticExpression[];
  readonly pointerBaseIsScalarLane?: boolean;
  readonly pointerBaseUnitBytes?: number;
  readonly pointerValid?: SemanticExpression;
  readonly pointerArrayAliases?: readonly (SemanticPointerAlias | undefined)[];
  readonly pointerCarrierValueType?: CudaLiteScalarType;
  readonly packedByteLanes?: 2 | 3 | 4;
  readonly cooperativeGroupKind?: CudaLiteParam["cooperativeGroupKind"];
  readonly tileSize?: number;
  readonly constant?: boolean;
  readonly initialized?: boolean;
  readonly init?: SemanticExpression;
  readonly dimensions: readonly number[];
  readonly addressSpace: SemanticAddressSpace;
  readonly span: SourceSpan;
}

interface SemanticPointerAlias {
  readonly pointerRoot?: string;
  readonly pointerAddressSpace?: SemanticAddressSpace;
  readonly pointerBaseIndices?: readonly SemanticExpression[];
  readonly pointerBaseIsScalarLane?: boolean;
  readonly pointerBaseUnitBytes?: number;
  readonly pointerValid?: SemanticExpression;
}

export interface CudaLiteSemanticFunction {
  readonly name: string;
  readonly returnType: CudaLiteScalarType;
  readonly params: readonly CudaLiteSemanticSymbol[];
  readonly body: readonly SemanticKernelIrOperation[];
  readonly span: SourceSpan;
}

export interface SemanticCooperativeGroupDeclaration {
  readonly kind: "cooperative-group";
  readonly groupKind: "thread" | "block" | "grid" | "tile" | "coalesced" | "binary";
  readonly name: string;
  readonly tileSize?: number;
  readonly partitionParent?: string;
  readonly partitionPredicate?: SemanticExpression;
  readonly span: SourceSpan;
}

export interface CudaLiteSemanticLaunchableEntry {
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
}

export interface SemanticMemoryRef {
  readonly base: string;
  readonly addressSpace: SemanticAddressSpace;
  readonly valueType?: CudaLiteScalarType;
  readonly containerValueType?: CudaLiteScalarType;
  readonly pointerBaseIsScalarLane?: boolean;
  readonly pointerBaseUnitBytes?: number;
  readonly packedByteLanes?: 2 | 3 | 4;
  readonly indices: readonly SemanticExpression[];
  readonly fields: readonly string[];
  readonly span: SourceSpan;
}

export type SemanticExpression =
  | {
      readonly kind: "literal";
      readonly literalKind: "number" | "string";
      readonly value: string | number;
      readonly valueType?: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "symbol";
      readonly name: string;
      readonly valueType?: CudaLiteScalarType;
      readonly addressSpace: SemanticAddressSpace;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "member";
      readonly object: SemanticExpression;
      readonly property: string;
      readonly valueType?: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "index";
      readonly target: SemanticExpression;
      readonly index: SemanticExpression;
      readonly valueType?: CudaLiteScalarType;
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
      readonly valueType?: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "texture-read";
      readonly callee: SemanticTextureReadCall;
      readonly texture: SemanticExpression;
      readonly x: SemanticExpression;
      readonly y: SemanticExpression;
      readonly z?: SemanticExpression;
      readonly valueType: Exclude<CudaLiteScalarType, "void">;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "surface-read";
      readonly callee: "surf2Dread" | "surf2DLayeredread" | "surf3Dread";
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
      readonly valueType?: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "binary";
      readonly operator: string;
      readonly left: SemanticExpression;
      readonly right: SemanticExpression;
      readonly valueType?: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "conditional";
      readonly condition: SemanticExpression;
      readonly consequent: SemanticExpression;
      readonly alternate: SemanticExpression;
      readonly valueType?: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "assignment";
      readonly operator: string;
      readonly target: SemanticExpression;
      readonly value: SemanticExpression;
      readonly valueType?: CudaLiteScalarType;
      readonly span: SourceSpan;
    }
  | {
      readonly kind: "update";
      readonly operator: string;
      readonly argument: SemanticExpression;
      readonly prefix: boolean;
      readonly valueType?: CudaLiteScalarType;
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
      readonly valueType?: CudaLiteScalarType;
      readonly span: SourceSpan;
    };

export type SemanticKernelIrOperation =
  | { readonly kind: "declare"; readonly target: CudaLiteSemanticSymbol; readonly init?: SemanticExpression; readonly span: SourceSpan }
  | { readonly kind: "dim3-declare"; readonly name: string; readonly args: readonly SemanticExpression[]; readonly span: SourceSpan }
  | { readonly kind: "cooperative-group-declare"; readonly declaration: SemanticCooperativeGroupDeclaration; readonly span: SourceSpan }
  | { readonly kind: "load"; readonly source: SemanticMemoryRef; readonly span: SourceSpan }
  | { readonly kind: "store"; readonly target: SemanticMemoryRef; readonly value: SemanticExpression; readonly operator: string; readonly reads: readonly SemanticMemoryRef[]; readonly span: SourceSpan }
  | { readonly kind: "copy"; readonly source: SemanticMemoryRef; readonly target: SemanticMemoryRef; readonly elements: number; readonly span: SourceSpan }
  | { readonly kind: "copy-fence"; readonly callee: string; readonly span: SourceSpan }
  | { readonly kind: "surface-write"; readonly surface: SemanticExpression; readonly value: SemanticExpression; readonly xBytes: SemanticExpression; readonly y: SemanticExpression; readonly z?: SemanticExpression; readonly span: SourceSpan }
  | { readonly kind: "surface-read-store"; readonly target: SemanticExpression; readonly surface: SemanticExpression; readonly xBytes: SemanticExpression; readonly y: SemanticExpression; readonly z?: SemanticExpression; readonly valueType?: CudaLiteScalarType; readonly span: SourceSpan }
  | { readonly kind: "atomic"; readonly callee: string; readonly target?: SemanticMemoryRef; readonly args: readonly SemanticExpression[]; readonly span: SourceSpan }
  | { readonly kind: "call"; readonly callee: string; readonly args: readonly SemanticExpression[]; readonly reads: readonly SemanticMemoryRef[]; readonly result?: Extract<SemanticExpression, { readonly kind: "symbol" }>; readonly span: SourceSpan }
  | { readonly kind: "expression"; readonly expression: SemanticExpression; readonly span: SourceSpan }
  | { readonly kind: "branch"; readonly condition: SemanticExpression; readonly consequent: readonly SemanticKernelIrOperation[]; readonly alternate: readonly SemanticKernelIrOperation[]; readonly span: SourceSpan }
  | { readonly kind: "loop"; readonly loopKind: "for" | "while" | "do-while"; readonly init?: SemanticKernelIrOperation | SemanticExpression; readonly condition?: SemanticExpression; readonly update?: SemanticExpression; readonly body: readonly SemanticKernelIrOperation[]; readonly span: SourceSpan }
  | { readonly kind: "barrier"; readonly callee: string; readonly scope: "subgroup" | "workgroup" | "grid"; readonly groupName?: string; readonly span: SourceSpan }
  | { readonly kind: "fence"; readonly callee: string; readonly span: SourceSpan }
  | { readonly kind: "device-launch"; readonly launch: SemanticDeviceLaunch; readonly span: SourceSpan }
  | { readonly kind: "inline-asm"; readonly statement: CudaLiteAsmStatement; readonly span: SourceSpan }
  | { readonly kind: "return"; readonly value?: SemanticExpression; readonly span: SourceSpan }
  | { readonly kind: "continue"; readonly span: SourceSpan }
  | { readonly kind: "break"; readonly span: SourceSpan }
  | { readonly kind: "block"; readonly body: readonly SemanticKernelIrOperation[]; readonly span: SourceSpan };

export interface SemanticDeviceLaunch {
  readonly callee: string;
  readonly grid: readonly SemanticExpression[];
  readonly block: readonly SemanticExpression[];
  readonly args: readonly SemanticExpression[];
}

export interface SemanticKernelIrModule {
  readonly kind: "semantic-kernel-ir";
  readonly name: string;
  readonly span: SourceSpan;
  readonly params: readonly CudaLiteSemanticSymbol[];
  readonly memory: readonly CudaLiteSemanticSymbol[];
  readonly functions: readonly CudaLiteSemanticFunction[];
  readonly operations: readonly SemanticKernelIrOperation[];
  readonly requiredFeatures: readonly string[];
  readonly barrierUniformity: CudaLiteAnalysis["barrierUniformity"];
  readonly workgroupSize: KernelLaunch["blockDim"];
}

export function walkSemanticOperations(
  operations: readonly SemanticKernelIrOperation[],
  visitExpression: (expression: SemanticExpression) => void,
): void {
  for (const operation of operations) walkSemanticOperation(operation, visitExpression);
}

export function walkSemanticOperation(
  operation: SemanticKernelIrOperation,
  visitExpression: (expression: SemanticExpression) => void,
): void {
  switch (operation.kind) {
    case "declare":
      if (operation.init) walkSemanticExpression(operation.init, visitExpression);
      return;
    case "dim3-declare":
      for (const arg of operation.args) walkSemanticExpression(arg, visitExpression);
      return;
    case "cooperative-group-declare":
      if (operation.declaration.partitionPredicate) {
        walkSemanticExpression(operation.declaration.partitionPredicate, visitExpression);
      }
      return;
    case "load":
      walkSemanticMemoryRef(operation.source, visitExpression);
      return;
    case "store":
      walkSemanticMemoryRef(operation.target, visitExpression);
      walkSemanticExpression(operation.value, visitExpression);
      for (const read of operation.reads) walkSemanticMemoryRef(read, visitExpression);
      return;
    case "copy":
      walkSemanticMemoryRef(operation.source, visitExpression);
      walkSemanticMemoryRef(operation.target, visitExpression);
      return;
    case "surface-write":
      walkSemanticExpression(operation.surface, visitExpression);
      walkSemanticExpression(operation.value, visitExpression);
      walkSemanticExpression(operation.xBytes, visitExpression);
      walkSemanticExpression(operation.y, visitExpression);
      if (operation.z) walkSemanticExpression(operation.z, visitExpression);
      return;
    case "surface-read-store":
      walkSemanticExpression(operation.target, visitExpression);
      walkSemanticExpression(operation.surface, visitExpression);
      walkSemanticExpression(operation.xBytes, visitExpression);
      walkSemanticExpression(operation.y, visitExpression);
      if (operation.z) walkSemanticExpression(operation.z, visitExpression);
      return;
    case "atomic":
      if (operation.target) walkSemanticMemoryRef(operation.target, visitExpression);
      for (const arg of operation.args) walkSemanticExpression(arg, visitExpression);
      return;
    case "call":
      for (const arg of operation.args) walkSemanticExpression(arg, visitExpression);
      for (const read of operation.reads) walkSemanticMemoryRef(read, visitExpression);
      return;
    case "expression":
      walkSemanticExpression(operation.expression, visitExpression);
      return;
    case "branch":
      walkSemanticExpression(operation.condition, visitExpression);
      walkSemanticOperations(operation.consequent, visitExpression);
      walkSemanticOperations(operation.alternate, visitExpression);
      return;
    case "loop":
      if (operation.init) {
        if (isSemanticKernelIrOperation(operation.init)) walkSemanticOperation(operation.init, visitExpression);
        else walkSemanticExpression(operation.init, visitExpression);
      }
      if (operation.condition) walkSemanticExpression(operation.condition, visitExpression);
      if (operation.update) walkSemanticExpression(operation.update, visitExpression);
      walkSemanticOperations(operation.body, visitExpression);
      return;
    case "device-launch":
      for (const expression of [...operation.launch.grid, ...operation.launch.block, ...operation.launch.args]) {
        walkSemanticExpression(expression, visitExpression);
      }
      return;
    case "return":
      if (operation.value) walkSemanticExpression(operation.value, visitExpression);
      return;
    case "block":
      walkSemanticOperations(operation.body, visitExpression);
      return;
    case "barrier":
    case "fence":
    case "inline-asm":
    case "continue":
    case "break":
      return;
  }
}

export function walkSemanticMemoryRef(
  ref: SemanticMemoryRef,
  visitExpression: (expression: SemanticExpression) => void,
): void {
  for (const index of ref.indices) walkSemanticExpression(index, visitExpression);
}

export function walkSemanticExpression(
  expression: SemanticExpression,
  visitExpression: (expression: SemanticExpression) => void,
): void {
  visitExpression(expression);
  switch (expression.kind) {
    case "literal":
    case "symbol":
      return;
    case "member":
      walkSemanticExpression(expression.object, visitExpression);
      return;
    case "index":
      walkSemanticExpression(expression.target, visitExpression);
      walkSemanticExpression(expression.index, visitExpression);
      return;
    case "call":
      walkSemanticExpression(expression.callee, visitExpression);
      for (const arg of expression.args) walkSemanticExpression(arg, visitExpression);
      return;
    case "texture-read":
      walkSemanticExpression(expression.texture, visitExpression);
      walkSemanticExpression(expression.x, visitExpression);
      walkSemanticExpression(expression.y, visitExpression);
      if (expression.z) walkSemanticExpression(expression.z, visitExpression);
      return;
    case "surface-read":
      walkSemanticExpression(expression.surface, visitExpression);
      walkSemanticExpression(expression.xBytes, visitExpression);
      walkSemanticExpression(expression.y, visitExpression);
      if (expression.z) walkSemanticExpression(expression.z, visitExpression);
      return;
    case "cast":
      walkSemanticExpression(expression.expression, visitExpression);
      return;
    case "unary":
    case "update":
      walkSemanticExpression(expression.argument, visitExpression);
      return;
    case "binary":
      walkSemanticExpression(expression.left, visitExpression);
      walkSemanticExpression(expression.right, visitExpression);
      return;
    case "conditional":
      walkSemanticExpression(expression.condition, visitExpression);
      walkSemanticExpression(expression.consequent, visitExpression);
      walkSemanticExpression(expression.alternate, visitExpression);
      return;
    case "assignment":
      walkSemanticExpression(expression.target, visitExpression);
      walkSemanticExpression(expression.value, visitExpression);
      return;
    case "initializer":
      for (const element of expression.elements) walkSemanticExpression(element, visitExpression);
      return;
    case "sequence":
      for (const item of expression.expressions) walkSemanticExpression(item, visitExpression);
      return;
  }
}

export function isSemanticKernelIrOperation(
  value: SemanticKernelIrOperation | SemanticExpression,
): value is SemanticKernelIrOperation {
  switch (value.kind) {
    case "declare":
    case "dim3-declare":
    case "cooperative-group-declare":
    case "load":
    case "store":
    case "copy":
    case "copy-fence":
    case "surface-write":
    case "surface-read-store":
    case "atomic":
    case "expression":
    case "branch":
    case "loop":
    case "barrier":
    case "fence":
    case "device-launch":
    case "inline-asm":
    case "return":
    case "continue":
    case "break":
    case "block":
      return true;
    case "call":
      return typeof value.callee === "string";
    default:
      return false;
  }
}

const DEFAULT_WORKGROUP_SIZE: KernelLaunch["blockDim"] = [256, 1, 1];
const COMPARISON_OPERATORS = new Set(["<", "<=", ">", ">=", "==", "!=", "&&", "||"]);
const POINTER_ORDER_OPERATORS = new Set(["<", "<=", ">", ">=", "==", "!="]);
const BARRIER_CALLS: ReadonlySet<string> = new Set([...CUDA_BARRIER_CALL_NAMES, ...CUDA_COOPERATIVE_BARRIER_CALL_NAMES, "grid.sync"]);
const FENCE_CALLS: ReadonlySet<string> = new Set(CUDA_FENCE_CALL_NAMES);
const ATOMIC_CALL_PREFIX = "atomic";
export function createCudaLiteSemanticModel(analysis: CudaLiteAnalysis): CudaLiteSemanticModel {
  const params = analysis.kernel.params.map(symbolForParam);
  const constants = analysis.constants.map(symbolForConstant);
  const deviceGlobals = analysis.deviceGlobals.map(symbolForDeviceGlobal);
  const textures = analysis.textures.map(symbolForTexture);
  const functionSymbols = analysis.functions.map(symbolForFunctionDeclaration);
  const globalScope = new Map([...params, ...constants, ...deviceGlobals, ...textures, ...functionSymbols].map((symbol) => [symbol.name, symbol]));
  const functions = analysis.functions.map((fn) => symbolForFunction(fn, globalScope));
  const launchableEntries: CudaLiteSemanticLaunchableEntry[] = [
    ...analysis.kernels.map((kernel) => ({
      kind: "kernel" as const,
      name: kernel.name,
      params: kernel.params.map(symbolForParam),
      span: kernel.span,
    })),
    ...analysis.functions.map((fn) => ({
      kind: "device-function" as const,
      name: fn.name,
      params: fn.params.map(symbolForParam),
      span: fn.span,
    })),
  ];
  return {
    kind: "cuda-lite-semantic-model",
    kernelName: analysis.kernel.name,
    span: analysis.kernel.span,
    params,
    symbols: [...params, ...constants, ...deviceGlobals, ...textures],
    functions,
    launchableEntries,
    requiredFeatures: analysis.requiredFeatures,
  };
}

export function lowerSemanticModelToKernelIr(
  analysis: CudaLiteAnalysis,
  semantic: CudaLiteSemanticModel,
  options: {
    readonly workgroupSize?: readonly [number, number, number];
    readonly dynamicSharedMemory?: Readonly<Record<string, number>>;
  } = {},
): SemanticKernelIrModule {
  const functionSymbols = semantic.functions.map(symbolForSemanticFunctionDeclaration);
  const scope = new Map([...semantic.symbols, ...functionSymbols].map((symbol) => [symbol.name, symbol]));
  const mutableParams = mutableKernelParamShadows(analysis, semantic.params);
  for (const shadow of mutableParams) scope.set(shadow.sourceName, shadow.symbol);
  const rawOperations = [
    ...mutableParams.map((shadow): SemanticKernelIrOperation => ({
      kind: "declare",
      target: shadow.symbol,
      init: semanticSymbolExpression(shadow.param, shadow.param.span),
      span: shadow.param.span,
    })),
    ...lowerStatements(analysis.kernel.body, scope),
  ];
  const loweredOperations = lowerSemanticEarlyReturnsBeforeDirectBarriers(rawOperations, semantic.functions, analysis.kernel.span);
  const localMemory = collectDeclaredMemory(loweredOperations);
  const reachable = collectReachableAnalysisNames(analysis);
  const sharedMemorySymbols = [...semantic.symbols, ...localMemory]
    .filter((symbol) => symbol.addressSpace === "shared")
    .map((symbol) => semanticMemorySymbolWithDynamicSharedExtent(symbol, options.dynamicSharedMemory));
  const resolved = resolveSemanticFunctionOverloads(
    loweredOperations,
    semantic.functions.filter((fn) => reachable.functionNames.has(fn.name) && !isSemanticGeneratedRandomCall(fn.name)),
  );
  const resolvedFunctionSharedMemory = resolved.functions.flatMap((fn) =>
    collectDeclaredMemory(fn.body)
      .filter((symbol) => symbol.addressSpace === "shared")
      .map((symbol) => semanticMemorySymbolWithDynamicSharedExtent(symbol, options.dynamicSharedMemory))
  );
  const sharedMemoryDimensions = new Map(
    [...sharedMemorySymbols, ...resolvedFunctionSharedMemory].map((symbol) => [symbol.name, symbol.dimensions] as const),
  );
  const sharedMemoryValueTypes = new Map(
    [...sharedMemorySymbols, ...resolvedFunctionSharedMemory].map((symbol) => [symbol.name, symbol.valueType] as const),
  );
  const specializedFunctions = specializeSharedPointerFunctions(
    resolved.operations,
    resolved.functions,
    sharedMemoryDimensions,
    sharedMemoryValueTypes,
  );
  const localSpecializedFunctions = specializeLocalPointerFunctions(resolved.operations, specializedFunctions);
  const constantSpecializedFunctions = specializeConstantPointerFunctions(resolved.operations, localSpecializedFunctions);
  const barrierFunctions = semanticIrBarrierFunctionNames(constantSpecializedFunctions);
  const operations = promoteSemanticBarrierResultCalls(resolved.operations, barrierFunctions);
  const functions = constantSpecializedFunctions.map((fn) => ({
    ...fn,
    body: promoteSemanticBarrierResultCalls(fn.body, barrierFunctions),
  }));
  const functionSharedMemory = functions.flatMap((fn) =>
    collectDeclaredMemory(fn.body).filter((symbol) => symbol.addressSpace === "shared")
  );
  return {
    kind: "semantic-kernel-ir",
    name: analysis.kernel.name,
    span: analysis.kernel.span,
    params: semantic.params,
    memory: [
      ...semantic.symbols.filter((symbol) =>
        symbol.kind !== "param" &&
        symbol.kind !== "function" &&
        reachable.symbolNames.has(symbol.name)
      ).map((symbol) => semanticMemorySymbolWithDynamicSharedExtent(symbol, options.dynamicSharedMemory)),
      ...localMemory.map((symbol) => semanticMemorySymbolWithDynamicSharedExtent(symbol, options.dynamicSharedMemory)),
      ...functionSharedMemory.map((symbol) => semanticMemorySymbolWithDynamicSharedExtent(symbol, options.dynamicSharedMemory)),
    ],
    functions,
    operations,
    requiredFeatures: semantic.requiredFeatures,
    barrierUniformity: analysis.barrierUniformity,
    workgroupSize: normalizeWorkgroupSize(options.workgroupSize ?? DEFAULT_WORKGROUP_SIZE),
  };
}

function lowerSemanticEarlyReturnsBeforeDirectBarriers(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
  kernelSpan: SourceSpan,
): readonly SemanticKernelIrOperation[] {
  const firstReturnBeforeBarrier = operations.findIndex((operation, index) =>
    semanticOperationContainsVoidReturn(operation) &&
    operations.slice(index + 1).some((later) => later.kind === "barrier")
  );
  if (firstReturnBeforeBarrier < 0) return operations;
  const affected = operations.slice(firstReturnBeforeBarrier);
  const pointerFunctions = new Set(functions.filter((fn) => fn.params.some((param) => param.pointer)).map((fn) => fn.name));
  if (!semanticVoidReturnsAreTerminal(affected)) return operations;
  if (affected.some((operation) => semanticActiveLaneTransformUnsafe(operation, pointerFunctions))) return operations;

  const active: CudaLiteSemanticSymbol = {
    name: "bg_active_lane",
    kind: "local",
    valueType: "bool",
    dimensions: [],
    addressSpace: "local",
    span: kernelSpan,
  };
  const activeExpression = semanticSymbolExpression(active, kernelSpan);
  const declare: SemanticKernelIrOperation = {
    kind: "declare",
    target: active,
    init: booleanExpression(true, kernelSpan),
    span: kernelSpan,
  };
  const lowered = affected.map((operation): SemanticKernelIrOperation => {
    const rewritten = rewriteSemanticVoidReturnsAsInactive(operation, active);
    if (operation.kind === "barrier") return rewritten;
    return {
      kind: "branch",
      condition: activeExpression,
      consequent: [rewritten],
      alternate: [],
      span: operation.span,
    };
  });
  return [...operations.slice(0, firstReturnBeforeBarrier), declare, ...lowered];
}

function semanticVoidReturnsAreTerminal(operations: readonly SemanticKernelIrOperation[]): boolean {
  for (const [index, operation] of operations.entries()) {
    if (semanticOperationContainsVoidReturn(operation) && index < operations.length - 1 && operation.kind === "return") return false;
    if (operation.kind === "branch") {
      if (!semanticVoidReturnsAreTerminal(operation.consequent) || !semanticVoidReturnsAreTerminal(operation.alternate)) return false;
    } else if (operation.kind === "block" || operation.kind === "loop") {
      if (!semanticVoidReturnsAreTerminal(operation.body)) return false;
    }
  }
  return true;
}

function semanticOperationContainsVoidReturn(operation: SemanticKernelIrOperation): boolean {
  if (operation.kind === "return") return operation.value === undefined;
  if (operation.kind === "branch") {
    return operation.consequent.some(semanticOperationContainsVoidReturn) ||
      operation.alternate.some(semanticOperationContainsVoidReturn);
  }
  if (operation.kind === "block" || operation.kind === "loop") {
    return operation.body.some(semanticOperationContainsVoidReturn);
  }
  return false;
}

function semanticActiveLaneTransformUnsafe(
  operation: SemanticKernelIrOperation,
  pointerFunctions: ReadonlySet<string>,
): boolean {
  if (collectSemanticFunctionCalls([operation]).some((call) => pointerFunctions.has(call.callee))) return true;
  if (operation.kind === "loop") return semanticOperationContainsVoidReturn(operation);
  if (operation.kind === "declare") return true;
  if (operation.kind === "branch") {
    return operation.consequent.some((item) => semanticActiveLaneTransformUnsafe(item, pointerFunctions)) ||
      operation.alternate.some((item) => semanticActiveLaneTransformUnsafe(item, pointerFunctions));
  }
  if (operation.kind === "block") return operation.body.some((item) => semanticActiveLaneTransformUnsafe(item, pointerFunctions));
  return false;
}

function rewriteSemanticVoidReturnsAsInactive(
  operation: SemanticKernelIrOperation,
  active: CudaLiteSemanticSymbol,
): SemanticKernelIrOperation {
  if (operation.kind === "return" && operation.value === undefined) {
    return {
      kind: "expression",
      expression: {
        kind: "assignment",
        operator: "=",
        target: semanticSymbolExpression(active, operation.span),
        value: booleanExpression(false, operation.span),
        valueType: "bool",
        span: operation.span,
      },
      span: operation.span,
    };
  }
  if (operation.kind === "branch") {
    return {
      ...operation,
      consequent: operation.consequent.map((item) => rewriteSemanticVoidReturnsAsInactive(item, active)),
      alternate: operation.alternate.map((item) => rewriteSemanticVoidReturnsAsInactive(item, active)),
    };
  }
  if (operation.kind === "block") {
    return { ...operation, body: operation.body.map((item) => rewriteSemanticVoidReturnsAsInactive(item, active)) };
  }
  return operation;
}

function specializeLocalPointerFunctions(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
): readonly CudaLiteSemanticFunction[] {
  let current = functions;
  for (let pass = 0; pass <= functions.length; pass++) {
    const next = specializeLocalPointerFunctionsOnce(operations, current);
    if (next.every((fn, index) => fn === current[index])) return next;
    current = next;
  }
  return current;
}

function specializeLocalPointerFunctionsOnce(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
): readonly CudaLiteSemanticFunction[] {
  const calls = [
    ...collectSemanticFunctionCalls(operations),
    ...functions.flatMap((fn) => collectSemanticFunctionCalls(
      fn.body,
      new Set(fn.params.filter((param) => param.pointer && param.addressSpace === "local").map((param) => param.name)),
    )),
  ];
  return functions.map((fn) => {
    const fnCalls = calls.filter((call) => call.callee === fn.name);
    const localPointers = new Map<string, string>();
    for (const [index, param] of fn.params.entries()) {
      if (!param.pointer || param.addressSpace !== "storage" || param.dimensions.length !== 0) continue;
      const refs = fnCalls.map((call) => semanticIrPointerArgumentMemoryRef(call.args[index]!));
      if (refs.length > 0 && refs.every((ref, callIndex) =>
        ref?.addressSpace === "local" &&
        (ref.indices.length === 0 ||
          ref.indices.length === 1 && isSemanticZeroLiteral(ref.indices[0]) && fnCalls[callIndex]!.ownerLocalPointerNames.has(ref.base)))) {
        localPointers.set(param.name, param.name);
      }
    }
    if (localPointers.size === 0) return fn;
    return {
      ...fn,
      params: fn.params.map((param) => localPointers.has(param.name) ? { ...param, addressSpace: "local" as const } : param),
      body: rewriteSemanticPointerAddressSpace(fn.body, localPointers, "local"),
    };
  });
}

function specializeConstantPointerFunctions(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
): readonly CudaLiteSemanticFunction[] {
  const calls = [
    ...collectSemanticFunctionCalls(operations),
    ...functions.flatMap((fn) => collectSemanticFunctionCalls(fn.body)),
  ];
  return functions.map((fn) => {
    const fnCalls = calls.filter((call) => call.callee === fn.name);
    const roots = new Map<string, string>();
    for (const [index, param] of fn.params.entries()) {
      if (!param.pointer || !param.constant || param.addressSpace !== "storage") continue;
      const refs = fnCalls.map((call) => semanticIrPointerArgumentMemoryRef(call.args[index]!));
      const root = refs[0]?.base;
      if (root && refs.length > 0 && refs.every((ref) => ref?.addressSpace === "constant" && ref.base === root && ref.indices.length === 0)) {
        roots.set(param.name, root);
      }
    }
    if (roots.size === 0) return fn;
    return {
      ...fn,
      params: fn.params.map((param) => roots.has(param.name) ? {
        ...param,
        addressSpace: "constant" as const,
        pointerAliasOf: roots.get(param.name)!,
      } : param),
      body: rewriteSemanticPointerAddressSpace(fn.body, roots, "constant"),
    };
  });
}

function mutableKernelParamShadows(
  analysis: CudaLiteAnalysis,
  params: readonly CudaLiteSemanticSymbol[],
): readonly {
  readonly sourceName: string;
  readonly param: CudaLiteSemanticSymbol;
  readonly symbol: CudaLiteSemanticSymbol;
}[] {
  const paramByName = new Map(params.map((param) => [param.name, param]));
  const mutable = new Set<string>();
  walkCudaLiteExpressions(analysis.kernel.body, (expression) => {
    const target = expression.kind === "assignment"
      ? expression.left
      : expression.kind === "update"
        ? expression.argument
        : undefined;
    const targetName = target === undefined ? undefined : mutatedParamRootName(target);
    if (targetName !== undefined && paramByName.has(targetName)) mutable.add(targetName);
  });
  return [...mutable].flatMap((sourceName) => {
    const param = paramByName.get(sourceName);
    if (!param || param.pointer || param.addressSpace !== "uniform" || param.valueType === undefined) return [];
    return [{
      sourceName,
      param,
      symbol: {
        ...param,
        name: `bg_param_local_${sourceName}_${param.span.start}`,
        kind: "local" as const,
        addressSpace: "local" as const,
      },
    }];
  });
}

function mutatedParamRootName(expression: CudaLiteExpression): string | undefined {
  if (expression.kind === "identifier") return expression.name;
  if (expression.kind === "member") return mutatedParamRootName(expression.object);
  return undefined;
}

function specializeSharedPointerFunctions(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
  sharedMemoryDimensions: ReadonlyMap<string, readonly number[]>,
  sharedMemoryValueTypes: ReadonlyMap<string, CudaLiteScalarType | undefined>,
): readonly CudaLiteSemanticFunction[] {
  let current = functions;
  for (let pass = 0; pass <= functions.length; pass++) {
    const dimensions = new Map(sharedMemoryDimensions);
    const valueTypes = new Map(sharedMemoryValueTypes);
    for (const param of current.flatMap((fn) => fn.params).filter((param) => param.pointer && param.addressSpace === "shared")) {
      dimensions.set(param.name, param.dimensions);
      valueTypes.set(param.name, param.valueType);
    }
    const calls = [
      ...collectSemanticFunctionCalls(operations),
      ...current.flatMap((fn) => collectSemanticFunctionCalls(fn.body)),
    ];
    const next = specializeSharedPointerFunctionsOnce(current, calls, dimensions, valueTypes);
    if (next.every((fn, index) => fn === current[index])) return next;
    current = next;
  }
  return current;
}

function specializeSharedPointerFunctionsOnce(
  functions: readonly CudaLiteSemanticFunction[],
  calls: readonly SemanticFunctionCallSite[],
  sharedMemoryDimensions: ReadonlyMap<string, readonly number[]>,
  sharedMemoryValueTypes: ReadonlyMap<string, CudaLiteScalarType | undefined>,
): readonly CudaLiteSemanticFunction[] {
  return functions.map((fn) => {
    const fnCalls = calls.filter((call) => call.callee === fn.name);
    const sharedPointerNames = new Map<string, string>();
    const sharedPointerDimensions = new Map<string, readonly number[]>();
    const sharedPointerRoots = new Map<string, readonly (string | undefined)[]>();
    for (const [index, param] of fn.params.entries()) {
      if (!param.pointer || param.addressSpace !== "storage") continue;
      const args = fnCalls
        .map((call) => call.args[index])
        .flatMap((arg) => arg === undefined ? [] : [sharedPointerRoot(arg)]);
      const dimensions = args.map((root) => root === undefined ? undefined : sharedMemoryDimensions.get(root));
      const carrierTypes = args.map((root) => root === undefined ? undefined : sharedMemoryValueTypes.get(root));
      const matchingValueTypes = param.valueType !== undefined && carrierTypes.every((valueType) =>
        valueType !== undefined && (valueType === param.valueType || sizeofCudaType(valueType) === sizeofCudaType(param.valueType!)),
      );
      if (args.length > 0 && args.every((root) => root !== undefined) && matchingValueTypes && dimensions.every((item) => item !== undefined && item.length <= 1 && (item.length === 0 || item[0] !== undefined)) && sameSemanticDimensions(dimensions as readonly (readonly number[])[])) {
        sharedPointerNames.set(param.name, `${param.name}__bg_shared_ptr`);
        sharedPointerDimensions.set(param.name, dimensions[0]!);
        sharedPointerRoots.set(param.name, args);
      }
    }
    if (sharedPointerNames.size === 0) return fn;
    const sharedPointerAliases = new Map<string, string>();
    const specializedParams = fn.params.filter((param) => sharedPointerNames.has(param.name));
    for (const [index, param] of specializedParams.entries()) {
      const roots = sharedPointerRoots.get(param.name)!;
      const canonical = specializedParams.slice(0, index).find((candidate) =>
        sameSemanticPointerRoots(roots, sharedPointerRoots.get(candidate.name)!),
      );
      if (canonical) sharedPointerAliases.set(param.name, sharedPointerNames.get(canonical.name)!);
    }
    return {
      ...fn,
      params: fn.params.map((param) => sharedPointerNames.has(param.name) ? {
        ...param,
        name: sharedPointerNames.get(param.name)!,
        addressSpace: "shared" as const,
        dimensions: sharedPointerDimensions.get(param.name)!,
        ...optionalPointerCarrierValueType(
          sharedMemoryValueTypes.get(sharedPointerRoots.get(param.name)![0]!) ?? param.valueType,
        ),
        ...(sharedPointerAliases.has(param.name) ? { pointerAliasOf: sharedPointerAliases.get(param.name)! } : {}),
      } : param),
      body: rewriteSemanticPointerAddressSpace(fn.body, sharedPointerNames),
    };
  });
}

function optionalPointerCarrierValueType(
  valueType: CudaLiteScalarType | undefined,
): { readonly pointerCarrierValueType?: CudaLiteScalarType } {
  return valueType === undefined ? {} : { pointerCarrierValueType: valueType };
}

function sameSemanticPointerRoots(
  left: readonly (string | undefined)[],
  right: readonly (string | undefined)[],
): boolean {
  return left.length === right.length && left.every((root, index) => root === right[index]);
}

interface SemanticFunctionCallSite {
  readonly callee: string;
  readonly args: readonly SemanticExpression[];
  readonly ownerLocalPointerNames: ReadonlySet<string>;
}

function collectSemanticFunctionCalls(
  operations: readonly SemanticKernelIrOperation[],
  ownerLocalPointerNames: ReadonlySet<string> = new Set(),
): readonly SemanticFunctionCallSite[] {
  const out: SemanticFunctionCallSite[] = [];
  collectSemanticOperationFunctionCalls(operations, out, ownerLocalPointerNames);
  walkSemanticOperations(operations, (expression) => {
    if (expression.kind === "call" && expression.callee.kind === "symbol") {
      out.push({ callee: expression.callee.name, args: expression.args, ownerLocalPointerNames });
    }
  });
  return out;
}

function collectSemanticOperationFunctionCalls(
  operations: readonly SemanticKernelIrOperation[],
  out: SemanticFunctionCallSite[],
  ownerLocalPointerNames: ReadonlySet<string>,
): void {
  for (const operation of operations) {
    if (operation.kind === "call") out.push({ callee: operation.callee, args: operation.args, ownerLocalPointerNames });
    if (operation.kind === "branch") {
      collectSemanticOperationFunctionCalls(operation.consequent, out, ownerLocalPointerNames);
      collectSemanticOperationFunctionCalls(operation.alternate, out, ownerLocalPointerNames);
    }
    if (operation.kind === "loop" || operation.kind === "block") collectSemanticOperationFunctionCalls(operation.body, out, ownerLocalPointerNames);
  }
}

function isSemanticZeroLiteral(expression: SemanticExpression | undefined): boolean {
  return expression?.kind === "literal" && expression.literalKind === "number" && expression.value === 0;
}

function sharedPointerRoot(expression: SemanticExpression): string | undefined {
  const ref = semanticIrPointerArgumentMemoryRef(expression);
  return ref?.addressSpace === "shared" ? ref.base : undefined;
}

function sameSemanticDimensions(dimensions: readonly (readonly number[])[]): boolean {
  return dimensions.every((item) => item.length === dimensions[0]!.length && item.every((value, index) => value === dimensions[0]![index]));
}

function rewriteSemanticPointerAddressSpace(
  operations: readonly SemanticKernelIrOperation[],
  names: ReadonlyMap<string, string>,
  addressSpace: "shared" | "constant" | "local" = "shared",
): readonly SemanticKernelIrOperation[] {
  return operations.map((operation) => {
    if (operation.kind === "store") return { ...operation, target: rewriteSemanticMemoryRef(operation.target, names, addressSpace), value: rewriteSemanticExpressionAddressSpace(operation.value, names, addressSpace), reads: operation.reads.map((ref) => rewriteSemanticMemoryRef(ref, names, addressSpace)) };
    if (operation.kind === "load") return { ...operation, source: rewriteSemanticMemoryRef(operation.source, names, addressSpace) };
    if (operation.kind === "atomic") return { ...operation, ...(operation.target === undefined ? {} : { target: rewriteSemanticMemoryRef(operation.target, names, addressSpace) }), args: operation.args.map((arg) => rewriteSemanticExpressionAddressSpace(arg, names, addressSpace)) };
    if (operation.kind === "call") return { ...operation, args: operation.args.map((arg) => rewriteSemanticExpressionAddressSpace(arg, names, addressSpace)), reads: operation.reads.map((ref) => rewriteSemanticMemoryRef(ref, names, addressSpace)) };
    if (operation.kind === "declare") return operation.init === undefined ? operation : { ...operation, init: rewriteSemanticExpressionAddressSpace(operation.init, names, addressSpace) };
    if (operation.kind === "expression") return { ...operation, expression: rewriteSemanticExpressionAddressSpace(operation.expression, names, addressSpace) };
    if (operation.kind === "branch") return { ...operation, condition: rewriteSemanticExpressionAddressSpace(operation.condition, names, addressSpace), consequent: rewriteSemanticPointerAddressSpace(operation.consequent, names, addressSpace), alternate: rewriteSemanticPointerAddressSpace(operation.alternate, names, addressSpace) };
    if (operation.kind === "loop") return { ...operation, ...(operation.init === undefined ? {} : { init: isSemanticKernelIrOperation(operation.init) ? rewriteSemanticPointerAddressSpace([operation.init], names, addressSpace)[0]! : rewriteSemanticExpressionAddressSpace(operation.init, names, addressSpace) }), ...(operation.condition === undefined ? {} : { condition: rewriteSemanticExpressionAddressSpace(operation.condition, names, addressSpace) }), ...(operation.update === undefined ? {} : { update: rewriteSemanticExpressionAddressSpace(operation.update, names, addressSpace) }), body: rewriteSemanticPointerAddressSpace(operation.body, names, addressSpace) };
    if (operation.kind === "block") return { ...operation, body: rewriteSemanticPointerAddressSpace(operation.body, names, addressSpace) };
    if (operation.kind === "return" && operation.value) return { ...operation, value: rewriteSemanticExpressionAddressSpace(operation.value, names, addressSpace) };
    return operation;
  });
}

function rewriteSemanticMemoryRef(ref: SemanticMemoryRef, names: ReadonlyMap<string, string>, addressSpace: "shared" | "constant" | "local"): SemanticMemoryRef {
  return {
    ...ref,
    ...(names.has(ref.base) && ref.addressSpace === "storage" ? { base: names.get(ref.base)!, addressSpace } : {}),
    indices: ref.indices.map((index) => rewriteSemanticExpressionAddressSpace(index, names, addressSpace)),
  };
}

function rewriteSemanticExpressionAddressSpace(expression: SemanticExpression, names: ReadonlyMap<string, string>, addressSpace: "shared" | "constant" | "local"): SemanticExpression {
  switch (expression.kind) {
    case "symbol": return names.has(expression.name) && expression.addressSpace === "storage" ? { ...expression, name: names.get(expression.name)!, addressSpace } : expression;
    case "member": return { ...expression, object: rewriteSemanticExpressionAddressSpace(expression.object, names, addressSpace) };
    case "index": return { ...expression, target: rewriteSemanticExpressionAddressSpace(expression.target, names, addressSpace), index: rewriteSemanticExpressionAddressSpace(expression.index, names, addressSpace), ...(expression.addressSpace === "storage" && expression.target.kind === "symbol" && names.has(expression.target.name) ? { addressSpace } : {}) };
    case "call": return { ...expression, callee: rewriteSemanticExpressionAddressSpace(expression.callee, names, addressSpace), args: expression.args.map((arg) => rewriteSemanticExpressionAddressSpace(arg, names, addressSpace)) };
    case "texture-read": return { ...expression, texture: rewriteSemanticExpressionAddressSpace(expression.texture, names, addressSpace), x: rewriteSemanticExpressionAddressSpace(expression.x, names, addressSpace), y: rewriteSemanticExpressionAddressSpace(expression.y, names, addressSpace), ...(expression.z === undefined ? {} : { z: rewriteSemanticExpressionAddressSpace(expression.z, names, addressSpace) }) };
    case "surface-read": return { ...expression, surface: rewriteSemanticExpressionAddressSpace(expression.surface, names, addressSpace), xBytes: rewriteSemanticExpressionAddressSpace(expression.xBytes, names, addressSpace), y: rewriteSemanticExpressionAddressSpace(expression.y, names, addressSpace), ...(expression.z === undefined ? {} : { z: rewriteSemanticExpressionAddressSpace(expression.z, names, addressSpace) }) };
    case "cast": return { ...expression, expression: rewriteSemanticExpressionAddressSpace(expression.expression, names, addressSpace) };
    case "unary": return { ...expression, argument: rewriteSemanticExpressionAddressSpace(expression.argument, names, addressSpace) };
    case "binary": return { ...expression, left: rewriteSemanticExpressionAddressSpace(expression.left, names, addressSpace), right: rewriteSemanticExpressionAddressSpace(expression.right, names, addressSpace) };
    case "conditional": return { ...expression, condition: rewriteSemanticExpressionAddressSpace(expression.condition, names, addressSpace), consequent: rewriteSemanticExpressionAddressSpace(expression.consequent, names, addressSpace), alternate: rewriteSemanticExpressionAddressSpace(expression.alternate, names, addressSpace) };
    case "assignment": return { ...expression, target: rewriteSemanticExpressionAddressSpace(expression.target, names, addressSpace), value: rewriteSemanticExpressionAddressSpace(expression.value, names, addressSpace) };
    case "update": return { ...expression, argument: rewriteSemanticExpressionAddressSpace(expression.argument, names, addressSpace) };
    case "initializer": return { ...expression, elements: expression.elements.map((item) => rewriteSemanticExpressionAddressSpace(item, names, addressSpace)) };
    case "sequence": return { ...expression, expressions: expression.expressions.map((item) => rewriteSemanticExpressionAddressSpace(item, names, addressSpace)) };
    case "literal": return expression;
  }
}

function semanticMemorySymbolWithDynamicSharedExtent(
  symbol: CudaLiteSemanticSymbol,
  dynamicSharedMemory: Readonly<Record<string, number>> | undefined,
): CudaLiteSemanticSymbol {
  const extent = dynamicSharedMemory?.[symbol.name];
  if (symbol.addressSpace !== "shared" || extent === undefined) return symbol;
  return { ...symbol, dimensions: [extent, ...symbol.dimensions] };
}

function collectReachableAnalysisNames(analysis: CudaLiteAnalysis): {
  readonly symbolNames: ReadonlySet<string>;
  readonly functionNames: ReadonlySet<string>;
} {
  const symbolNames = new Set<string>();
  const functionNames = new Set<string>();
  const functionsByName = new Map(analysis.functions.map((fn) => [fn.name, fn]));
  const pending = [analysis.kernel.body];

  for (let index = 0; index < pending.length; index++) {
    const body = pending[index]!;
    walkCudaLiteExpressions(body, (expression) => {
      if (expression.kind === "identifier") {
        symbolNames.add(expression.name);
        return;
      }
      if (expression.kind !== "call" || expression.callee.kind !== "identifier") return;
      const callee = expression.callee.name;
      symbolNames.add(callee);
      const fn = functionsByName.get(callee);
      if (!fn || functionNames.has(callee)) return;
      functionNames.add(callee);
      pending.push(fn.body);
    });
  }

  return { symbolNames, functionNames };
}

function lowerStatements(
  statements: readonly CudaLiteStatement[],
  parentScope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] {
  const scope = new Map(parentScope);
  return lowerStatementsWithScope(statements, scope);
}

function lowerStatementsWithScope(
  statements: readonly CudaLiteStatement[],
  scope: Map<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] {
  const out: SemanticKernelIrOperation[] = [];
  for (let index = 0; index < statements.length; index++) {
    const statement = statements[index]!;
    if (isLocalPointerAliasPlaceholder(statement) && hasLaterLocalPointerAliasAssignment(statement.name, statements.slice(index + 1), scope)) {
      const target = symbolForVar(statement, scope);
      scope.set(target.name, target);
      out.push({ kind: "expression", expression: zeroExpression(statement.span), span: statement.span });
      continue;
    }
    out.push(...lowerStatementOperations(statement, scope));
  }
  return out.map((operation) => {
    if (operation.kind !== "declare" || !operation.target.pointer || operation.target.dimensions.length !== 1) return operation;
    const target = scope.get(operation.target.name);
    if (!semanticPointerArrayAliasesComplete(target)) return operation;
    return { kind: "expression", expression: zeroExpression(operation.span), span: operation.span };
  });
}

function lowerStatementOperations(
  statement: CudaLiteStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] {
  const chainedStores = semanticMemoryAssignmentChainOperations(statement, scope);
  if (chainedStores) return chainedStores;
  const mathOutVarDecl = semanticMathOutVarDeclOperations(statement, scope);
  const mathOutAssignment = semanticMathOutAssignmentOperations(statement, scope);
  const mathOutCall = semanticMathOutCallStatementOperations(statement, scope);
  return mathOutVarDecl ?? mathOutAssignment ?? mathOutCall ?? [lowerStatement(statement, scope)];
}

function semanticMemoryAssignmentChainOperations(
  statement: CudaLiteStatement,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (statement.kind !== "expr" || statement.expression.kind !== "assignment") return undefined;
  let expression = lowerExpression(statement.expression, scope);
  const targets: SemanticMemoryRef[] = [];
  while (expression.kind === "assignment" && expression.operator === "=") {
    const target = memoryRefFromExpression(expression.target);
    if (!target || !semanticExpressionSideEffectFree(expression.target) || semanticExpressionContainsCall(expression.target)) return undefined;
    targets.push(target);
    expression = expression.value;
  }
  if (targets.length < 2 || (expression.kind !== "literal" && expression.kind !== "symbol")) return undefined;
  return targets.reverse().map((target) => storeOperation(target, expression, statement.span));
}

function semanticExpressionContainsCall(expression: SemanticExpression): boolean {
  let found = false;
  walkSemanticExpression(expression, (item) => {
    if (item.kind === "call") found = true;
  });
  return found;
}

function lowerInlineAsmBuiltinRegisterAssignment(
  statement: CudaLiteAsmStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
): SemanticKernelIrOperation | undefined {
  const op = classifyInlineAsm(statement.template);
  const outputs = statement.outputs ?? (statement.output === undefined ? [] : [statement.output]);
  if (outputs.length !== 1) return undefined;
  const target = lowerExpression(outputs[0]!, scope);
  const fmaSources = op?.kind === "fma-rn-f32"
    ? op.sources ?? [
        { kind: "operand", index: outputs.length },
        { kind: "operand", index: outputs.length + 1 },
        { kind: "operand", index: 0 },
      ] satisfies readonly [InlineAsmF32Source, InlineAsmF32Source, InlineAsmF32Source]
    : undefined;
  const fmaInputs = fmaSources === undefined ? undefined : expectedInlineAsmF32SourceInputs(fmaSources, outputs.length);
  const fmaArgs = fmaSources !== undefined && fmaInputs === statement.inputs.length
    ? fmaSources.map((source) => semanticInlineAsmF32Source(source, statement, outputs, scope))
    : undefined;
  const floatBinarySources = op?.kind === "float-binary-rn-f32"
    ? op.sources ?? [
        { kind: "operand", index: outputs.length },
        { kind: "operand", index: outputs.length + 1 },
      ] satisfies readonly [InlineAsmF32Source, InlineAsmF32Source]
    : undefined;
  const floatBinaryInputs = floatBinarySources === undefined ? undefined : expectedInlineAsmF32SourceInputs(floatBinarySources, outputs.length);
  const floatBinaryArgs = floatBinarySources !== undefined && floatBinaryInputs === statement.inputs.length
    ? floatBinarySources.map((source) => semanticInlineAsmF32Source(source, statement, outputs, scope))
    : undefined;
  const bitInput = op && (op.kind === "bfind-u32" || op.kind === "ffs-b32" || op.kind === "popc-b32" || op.kind === "clz-b32" || op.kind === "brev-b32")
    ? op.immediate === undefined
      ? statement.inputs.length === 1 ? lowerExpression(statement.inputs[0]!, scope) : undefined
      : statement.inputs.length === 0 ? semanticUintLiteralExpression(op.immediate, statement.span) : undefined
    : undefined;
  const value = op?.kind === "fma-rn-f32" && fmaArgs?.length === 3 && fmaArgs.every((arg) => arg !== undefined)
    ? semanticCallExpression("fma", fmaArgs as readonly SemanticExpression[], "float", statement.span)
    : op?.kind === "float-binary-rn-f32" && floatBinaryArgs?.length === 2 && floatBinaryArgs.every((arg) => arg !== undefined)
      ? semanticFloatBinaryExpression(op.op, floatBinaryArgs[0]!, floatBinaryArgs[1]!, statement.span)
    : op?.kind === "bfind-u32" && bitInput !== undefined
    ? semanticUintBinaryExpression(
        "-",
        semanticUintLiteralExpression(31, statement.span),
        castScalarExpression(
          semanticCallExpression("__clz", [bitInput], "int", statement.span),
          "uint",
          statement.span,
        ),
        statement.span,
      )
    : op?.kind === "ffs-b32" && bitInput !== undefined
      ? semanticCallExpression("__ffs", [bitInput], "int", statement.span)
    : op?.kind === "popc-b32" && bitInput !== undefined
      ? castScalarExpression(semanticCallExpression("__popc", [bitInput], "int", statement.span), "uint", statement.span)
    : op?.kind === "clz-b32" && bitInput !== undefined
      ? castScalarExpression(semanticCallExpression("__clz", [bitInput], "int", statement.span), "uint", statement.span)
    : op?.kind === "brev-b32" && bitInput !== undefined
      ? semanticCallExpression("__brev", [bitInput], "uint", statement.span)
    : statement.inputs.length !== 0
      ? undefined
      : op?.kind === "special-register-u32"
    ? semanticSpecialRegisterExpression(op.register, statement.span)
    : op?.kind === "laneid"
      ? semanticLaneIdExpression(statement.span)
      : op?.kind === "warpid"
        ? semanticWarpIdExpression(statement.span)
        : op?.kind === "lanemask-lt"
          ? semanticLaneMaskLtExpression(statement.span)
      : undefined;
  if (!value) return undefined;
  return {
    kind: "expression",
    expression: {
      kind: "assignment",
      operator: "=",
      target,
      value,
      valueType: expressionValueType(target) ?? "uint",
      span: statement.span,
    },
    span: statement.span,
  };
}

function semanticFloatBinaryExpression(
  op: "add" | "sub" | "mul" | "div",
  left: SemanticExpression,
  right: SemanticExpression,
  span: SourceSpan,
): SemanticExpression {
  return {
    kind: "binary",
    operator: op === "add" ? "+" : op === "sub" ? "-" : op === "mul" ? "*" : "/",
    left,
    right,
    valueType: "float",
    span,
  };
}

function semanticInlineAsmF32Source(
  source: InlineAsmF32Source,
  statement: CudaLiteAsmStatement,
  outputs: readonly CudaLiteExpression[],
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticExpression | undefined {
  if (source.kind === "immediate") return numberExpression(source.value, statement.span);
  const expression = source.index < outputs.length
    ? outputs[source.index]
    : statement.inputs[source.index - outputs.length];
  return expression === undefined ? undefined : lowerExpression(expression, scope);
}

function semanticLaneIdExpression(span: SourceSpan): SemanticExpression {
  return semanticUintBinaryExpression("&", semanticThreadLinearIndexExpression(span), semanticUintLiteralExpression(31, span), span);
}

function semanticWarpIdExpression(span: SourceSpan): SemanticExpression {
  return semanticUintBinaryExpression("/", semanticThreadLinearIndexExpression(span), semanticUintLiteralExpression(32, span), span);
}

function semanticLaneMaskLtExpression(span: SourceSpan): SemanticExpression {
  const one = semanticUintLiteralExpression(1, span);
  return semanticUintBinaryExpression("-", semanticUintBinaryExpression("<<", one, semanticLaneIdExpression(span), span), one, span);
}

function semanticUintLiteralExpression(value: number, span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value, valueType: "uint", span };
}

function semanticThreadLinearIndexExpression(span: SourceSpan): SemanticExpression {
  const threadX = semanticSpecialRegisterExpression("tid.x", span);
  const threadY = semanticSpecialRegisterExpression("tid.y", span);
  const threadZ = semanticSpecialRegisterExpression("tid.z", span);
  const blockX = semanticSpecialRegisterExpression("ntid.x", span);
  const blockY = semanticSpecialRegisterExpression("ntid.y", span);
  const row = semanticUintBinaryExpression("+", threadY, semanticUintBinaryExpression("*", blockY, threadZ, span), span);
  return semanticUintBinaryExpression("+", threadX, semanticUintBinaryExpression("*", blockX, row, span), span);
}

function semanticUintBinaryExpression(
  operator: "+" | "-" | "*" | "/" | "<<" | "&",
  left: SemanticExpression,
  right: SemanticExpression,
  span: SourceSpan,
): SemanticExpression {
  return { kind: "binary", operator, left, right, valueType: "uint", span };
}

function semanticSpecialRegisterExpression(register: PtxSpecialU32Register, span: SourceSpan): SemanticExpression {
  const [root, property] = register.split(".") as [string, "x" | "y" | "z"];
  const objectName = root === "tid" ? "threadIdx" : root === "ctaid" ? "blockIdx" : root === "ntid" ? "blockDim" : "gridDim";
  return {
    kind: "member",
    object: {
      kind: "symbol",
      name: objectName,
      valueType: "uint",
      addressSpace: "builtin",
      span,
    },
    property,
    valueType: "uint",
    span,
  };
}

function lowerStatement(
  statement: CudaLiteStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
): SemanticKernelIrOperation {
  switch (statement.kind) {
    case "block": {
      const childScope = new Map(scope);
      const body = lowerStatementsWithScope(statement.body, childScope);
      mergeBlockLocalPointerAliases(scope, childScope);
      return { kind: "block", body, span: statement.span };
    }
    case "var": {
      const target = symbolForVar(statement, scope);
      scope.set(target.name, target);
      if (target.pointerRoot && semanticPointerAliasAddressSpaceSupported(target.pointerAddressSpace) && target.pointerBaseIndices) {
        return { kind: "expression", expression: zeroExpression(statement.span), span: statement.span };
      }
      return {
        kind: "declare",
        target,
        ...(statement.init === undefined ? {} : { init: lowerExpression(statement.init, scope) }),
        span: statement.span,
      };
    }
    case "dim3":
      return { kind: "dim3-declare", name: statement.name, args: statement.args.map((arg) => lowerExpression(arg, scope)), span: statement.span };
    case "cooperative-group":
      scope.set(statement.name, semanticSymbolForCooperativeGroup(statement));
      return {
        kind: "cooperative-group-declare",
        declaration: {
          kind: "cooperative-group",
          groupKind: statement.groupKind,
          name: statement.name,
          ...(statement.tileSize === undefined ? {} : { tileSize: statement.tileSize }),
          ...(statement.partitionParent === undefined ? {} : { partitionParent: statement.partitionParent }),
          ...(statement.partitionPredicate === undefined ? {} : { partitionPredicate: lowerExpression(statement.partitionPredicate, scope) }),
          span: statement.span,
        },
        span: statement.span,
      };
    case "kernel-launch":
      return { kind: "device-launch", launch: lowerDeviceLaunch(statement, scope), span: statement.span };
    case "asm": {
      const registerAssignment = lowerInlineAsmBuiltinRegisterAssignment(statement, scope);
      if (registerAssignment) return registerAssignment;
      return { kind: "inline-asm", statement, span: statement.span };
    }
    case "expr": {
      const cpAsync = semanticCpAsyncOperation(statement.expression, scope, statement.span);
      if (cpAsync) return cpAsync;
      const pointerRebase = semanticStoragePointerRebaseOperation(statement.expression, scope, statement.span);
      if (pointerRebase) return pointerRebase;
      const aliasAssignment = localPointerAliasUpdate(statement.expression, scope);
      if (aliasAssignment) return { kind: "expression", expression: zeroExpression(statement.span), span: statement.span };
      const pointerArrayAssignment = localPointerArrayAliasUpdate(statement.expression, scope);
      if (pointerArrayAssignment) return { kind: "expression", expression: zeroExpression(statement.span), span: statement.span };
      const expression = lowerExpression(statement.expression, scope);
      const callName = expression.kind === "call" ? semanticCallName(expression.callee) : undefined;
      const memberBarrier = expression.kind === "call" ? semanticCooperativeMemberBarrier(expression, scope) : undefined;
      if (memberBarrier) {
        return {
          kind: "barrier",
          callee: memberBarrier.callee,
          scope: memberBarrier.scope,
          groupName: memberBarrier.groupName,
          span: statement.span,
        };
      }
      if (expression.kind === "call" && callName !== undefined && BARRIER_CALLS.has(callName)) {
        const groupName = barrierGroupName(expression);
        const barrierScope = semanticBarrierScope(callName, groupName, scope);
        return {
          kind: "barrier",
          callee: callName,
          scope: barrierScope,
          ...(groupName === undefined ? {} : { groupName }),
          span: statement.span,
        };
      }
      if (expression.kind === "call" && callName !== undefined && FENCE_CALLS.has(callName)) {
        return { kind: "fence", callee: callName, span: statement.span };
      }
      if (expression.kind === "assignment") {
        if (statement.expression.kind === "assignment") {
          const mathOutAssignment = semanticMathOutAssignmentBlock(statement.expression, expression, scope, statement.span);
          if (mathOutAssignment) return mathOutAssignment;
        }
        const target = memoryRefFromExpression(expression.target);
        if (target) {
          const value = semanticSequencedAssignmentValue(expression.value);
          const reinterpretedCopy = semanticReinterpretedScalarCopy(target, value, expression.operator, scope, statement.span);
          if (reinterpretedCopy) return reinterpretedCopy;
          return {
            kind: "store",
            target,
            value,
            operator: expression.operator,
            reads: collectMemoryRefs(value),
            span: statement.span,
          };
        }
      }
      if (expression.kind === "call" && expression.callee.kind === "symbol") {
        if (
          isCudaSemanticSurfaceWriteCallName(expression.callee.name) &&
          expression.args.length >= (expression.callee.name === "surf1Dwrite" ? 3 : 4)
        ) {
          return {
            kind: "surface-write",
            value: expression.args[0]!,
            surface: expression.args[1]!,
            xBytes: expression.args[2]!,
            y: expression.callee.name === "surf1Dwrite" ? zeroExpression(expression.span) : expression.args[3]!,
            ...(semanticSurfaceWriteUsesZ(expression.callee.name) && expression.args[4] !== undefined ? { z: expression.args[4]! } : {}),
            span: statement.span,
          };
        }
        if (
          (expression.callee.name === "surf2Dread" && expression.args.length === 4) ||
          ((expression.callee.name === "surf2DLayeredread" || expression.callee.name === "surf3Dread") && expression.args.length === 5)
        ) {
          const target = expression.args[0]!;
          return {
            kind: "surface-read-store",
            target,
            surface: expression.args[1]!,
            xBytes: expression.args[2]!,
            y: expression.args[3]!,
            ...(expression.args[4] === undefined ? {} : { z: expression.args[4]! }),
            ...optionalValueType(target.kind === "unary" && target.operator === "&" ? expressionValueType(target.argument) : expressionValueType(target)),
            span: statement.span,
          };
        }
        const target = atomicTargetFromCall(expression);
        if (expression.callee.name.startsWith(ATOMIC_CALL_PREFIX)) {
          return {
            kind: "atomic",
            callee: expression.callee.name,
            ...(target === undefined ? {} : { target }),
            args: expression.args,
            span: statement.span,
          };
        }
        if (statement.expression.kind === "call") {
          const sincos = semanticSincosStores(statement.expression, expression, scope, statement.span);
          if (sincos) return sincos;
          const modf = semanticModfStore(statement.expression, expression, scope, statement.span);
          if (modf) return modf;
          const remquo = semanticRemquoStore(statement.expression, expression, scope, statement.span);
          if (remquo) return remquo;
          const frexp = semanticFrexpStore(statement.expression, expression, scope, statement.span);
          if (frexp) return frexp;
        }
        if (statement.expression.kind === "call" && CUDA_CACHE_HINT_STORES.has(expression.callee.name)) {
          const cacheTarget = cacheHintStoreTarget(statement.expression, scope);
          const value = expression.args[1];
          if (cacheTarget && value) {
            return {
              kind: "store",
              target: cacheTarget,
              value,
              operator: "=",
              reads: collectMemoryRefs(value),
              span: statement.span,
            };
          }
        }
        return {
          kind: "call",
          callee: expression.callee.name,
          args: expression.args,
          reads: expression.args.flatMap((arg) => collectMemoryRefs(arg)),
          span: statement.span,
        };
      }
      return { kind: "expression", expression, span: statement.span };
    }
    case "if": {
      const condition = lowerExpression(statement.condition, scope);
      const consequentScope = new Map(scope);
      const alternateScope = new Map(scope);
      const consequent = lowerStatementsWithScope(statement.consequent, consequentScope);
      const alternate = lowerStatementsWithScope(statement.alternate ?? [], alternateScope);
      mergeBranchLocalPointerAliases(scope, consequentScope, alternateScope, condition, statement.span);
      return {
        kind: "branch",
        condition,
        consequent,
        alternate,
        span: statement.span,
      };
    }
    case "for":
      {
        const loopScope = new Map(scope);
        const init = statement.init?.kind === "var"
          ? lowerForInitStatement(statement.init, loopScope)
          : statement.init
          ? lowerExpression(statement.init, loopScope)
          : undefined;
        return {
          kind: "loop",
          loopKind: "for",
          ...(init === undefined ? {} : { init }),
          ...(statement.condition === undefined ? {} : { condition: lowerExpression(statement.condition, loopScope) }),
          ...(statement.update === undefined ? {} : { update: lowerExpression(statement.update, loopScope) }),
          body: lowerStatements(statement.body, loopScope),
          span: statement.span,
        };
      }
    case "while":
      return {
        kind: "loop",
        loopKind: "while",
        condition: lowerExpression(statement.condition, scope),
        body: lowerStatements(statement.body, scope),
        span: statement.span,
      };
    case "do-while":
      return {
        kind: "loop",
        loopKind: "do-while",
        condition: lowerExpression(statement.condition, scope),
        body: lowerStatements(statement.body, scope),
        span: statement.span,
      };
    case "return":
      return {
        kind: "return",
        ...(statement.value === undefined ? {} : { value: lowerExpression(statement.value, scope) }),
        span: statement.span,
      };
    case "continue":
      return { kind: "continue", span: statement.span };
    case "break":
      return { kind: "break", span: statement.span };
  }
}

function semanticReinterpretedScalarCopy(
  target: SemanticMemoryRef,
  value: SemanticExpression,
  operator: string,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticKernelIrOperation | undefined {
  if (operator !== "=" || target.fields.length > 0 || target.indices.length === 0) return undefined;
  const source = memoryRefFromExpression(value);
  if (!source || source.fields.length > 0 || source.indices.length === 0 || source.valueType !== target.valueType) return undefined;
  const viewType = target.valueType;
  if (!isCudaVectorType(viewType)) return undefined;
  const targetRoot = scope.get(target.base);
  const sourceRoot = scope.get(source.base);
  const scalarType = targetRoot?.valueType;
  if (!scalarType || scalarType !== sourceRoot?.valueType || isCudaVectorType(scalarType)) return undefined;
  const viewBytes = sizeofCudaType(viewType);
  const scalarBytes = sizeofCudaType(scalarType);
  if (!viewBytes || !scalarBytes || viewBytes % scalarBytes !== 0) return undefined;
  const elements = viewBytes / scalarBytes;
  if (elements < 1 || elements > 16) return undefined;
  return {
    kind: "copy",
    source: { ...source, valueType: scalarType },
    target: { ...target, valueType: scalarType },
    elements,
    span,
  };
}

function semanticSequencedAssignmentValue(expression: SemanticExpression): SemanticExpression {
  if (expression.kind !== "assignment") return expression;
  return {
    kind: "sequence",
    expressions: [expression, expression.target],
    ...optionalValueType(expression.valueType ?? expressionValueType(expression.target)),
    span: expression.span,
  };
}

function semanticResultCallOperation(
  expression: Extract<SemanticExpression, { readonly kind: "assignment" }>,
  span: SourceSpan,
): Extract<SemanticKernelIrOperation, { readonly kind: "call" }> | undefined {
  if (
    expression.operator !== "=" ||
    expression.target.kind !== "symbol" ||
    expression.target.addressSpace !== "local" ||
    expression.value.kind !== "call" ||
    expression.value.callee.kind !== "symbol" ||
    expression.value.callee.addressSpace !== "function"
  ) return undefined;
  return {
    kind: "call",
    callee: expression.value.callee.name,
    args: expression.value.args,
    reads: collectMemoryRefs(expression.value),
    result: expression.target,
    span,
  };
}

function semanticIrBarrierFunctionNames(functions: readonly CudaLiteSemanticFunction[]): ReadonlySet<string> {
  const names = new Set<string>();
  let changed = true;
  while (changed) {
    changed = false;
    for (const fn of functions) {
      if (names.has(fn.name) || !semanticIrOperationsContainBarrier(fn.body, names)) continue;
      names.add(fn.name);
      changed = true;
    }
  }
  return names;
}

function semanticIrOperationsContainBarrier(
  operations: readonly SemanticKernelIrOperation[],
  barrierFunctions: ReadonlySet<string>,
): boolean {
  return operations.some((operation) =>
    operation.kind === "barrier" ||
    operation.kind === "call" && barrierFunctions.has(operation.callee) ||
    operation.kind === "branch" && (semanticIrOperationsContainBarrier(operation.consequent, barrierFunctions) || semanticIrOperationsContainBarrier(operation.alternate, barrierFunctions)) ||
    (operation.kind === "loop" || operation.kind === "block") && semanticIrOperationsContainBarrier(operation.body, barrierFunctions)
  );
}

function promoteSemanticBarrierResultCalls(
  operations: readonly SemanticKernelIrOperation[],
  barrierFunctions: ReadonlySet<string>,
): readonly SemanticKernelIrOperation[] {
  return operations.map((operation) => {
    if (operation.kind === "expression" && operation.expression.kind === "assignment") {
      const promoted = semanticResultCallOperation(operation.expression, operation.span);
      if (promoted && barrierFunctions.has(promoted.callee)) return promoted;
    }
    if (operation.kind === "branch") {
      return {
        ...operation,
        consequent: promoteSemanticBarrierResultCalls(operation.consequent, barrierFunctions),
        alternate: promoteSemanticBarrierResultCalls(operation.alternate, barrierFunctions),
      };
    }
    if (operation.kind === "loop" || operation.kind === "block") {
      return { ...operation, body: promoteSemanticBarrierResultCalls(operation.body, barrierFunctions) };
    }
    return operation;
  });
}

function semanticSymbolForCooperativeGroup(statement: CudaLiteCooperativeGroupDecl): CudaLiteSemanticSymbol {
  return {
    name: statement.name,
    kind: "local",
    valueType: "uint",
    pointer: false,
    cooperativeGroupKind: statement.groupKind,
    ...(statement.tileSize === undefined ? {} : { tileSize: statement.tileSize }),
    dimensions: [],
    addressSpace: "local",
    span: statement.span,
  };
}

function semanticCooperativeMemberBarrier(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): { readonly callee: "cg::sync" | "grid.sync"; readonly scope: "subgroup" | "workgroup" | "grid"; readonly groupName: string } | undefined {
  if (expression.callee.kind !== "member" || expression.callee.property !== "sync" || expression.callee.object.kind !== "symbol") return undefined;
  const group = scope.get(expression.callee.object.name);
  if (group?.cooperativeGroupKind === undefined) return undefined;
  return {
    callee: group.cooperativeGroupKind === "grid" ? "grid.sync" : "cg::sync",
    scope: group.cooperativeGroupKind === "grid"
      ? "grid"
      : group.cooperativeGroupKind === "tile" || group.cooperativeGroupKind === "thread"
        ? "subgroup"
        : "workgroup",
    groupName: group.name,
  };
}

function semanticBarrierScope(
  callee: string,
  groupName: string | undefined,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): "subgroup" | "workgroup" | "grid" {
  if (callee === "__syncwarp") return "subgroup";
  if (callee === "__syncthreads") return "workgroup";
  const groupKind = groupName === undefined ? undefined : scope.get(groupName)?.cooperativeGroupKind;
  if (groupKind === "grid") return "grid";
  if (groupKind === "tile" || groupKind === "thread") return "subgroup";
  return "workgroup";
}

function semanticStoragePointerRebaseOperation(
  source: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticKernelIrOperation | undefined {
  if (source.kind !== "assignment" || source.operator !== "=") return undefined;
  const targetExpression = lowerExpression(source.left, scope);
  const target = memoryRefFromExpression(targetExpression);
  if (!target || target.addressSpace !== "storage" || target.indices.length !== 0 || target.fields.length !== 0) return undefined;
  const value = lowerExpression(source.right, scope);
  const rebase = semanticStoragePointerRebaseValue(target, value);
  if (!rebase) return undefined;
  return {
    kind: "store",
    target,
    value: rebase.value,
    operator: rebase.operator,
    reads: collectMemoryRefs(rebase.value),
    span,
  };
}

function semanticStoragePointerRebaseValue(
  target: SemanticMemoryRef,
  value: SemanticExpression,
): { readonly operator: "+=" | "-="; readonly value: SemanticExpression } | undefined {
  if (value.kind === "binary" && (value.operator === "+" || value.operator === "-")) {
    const base = memoryRefFromExpression(value.left);
    if (base?.base === target.base && base.addressSpace === "storage" && base.indices.length === 0 && base.fields.length === 0) {
      return { operator: value.operator === "+" ? "+=" : "-=", value: value.right };
    }
  }
  if (value.kind !== "unary" || value.operator !== "&" || value.argument.kind !== "index") return undefined;
  const base = memoryRefFromExpression(value.argument.target);
  if (base?.base !== target.base || base.addressSpace !== "storage" || base.indices.length !== 0 || base.fields.length !== 0) return undefined;
  return { operator: "+=", value: value.argument.index };
}

function semanticSurfaceWriteUsesZ(callee: string): boolean {
  return callee === "surf2DLayeredwrite" || callee === "surf3Dwrite";
}

function lowerExpression(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticExpression {
  switch (expression.kind) {
    case "number":
      return { kind: "literal", literalKind: "number", value: expression.value, valueType: numberLiteralType(expression.raw), span: expression.span };
    case "string":
      return { kind: "literal", literalKind: "string", value: expression.value, span: expression.span };
    case "identifier": {
      const symbol = scope.get(expression.name);
      const namedConstant = symbol === undefined ? CUDA_NAMED_CONSTANTS.get(expression.name) : undefined;
      if (namedConstant) {
        return {
          kind: "literal",
          literalKind: "number",
          value: namedConstant.value,
          valueType: namedConstant.valueType,
          span: expression.span,
        };
      }
      if (symbol === undefined && expression.name === "nullptr") {
        return { kind: "literal", literalKind: "number", value: 0, valueType: "voidptr", span: expression.span };
      }
      return {
        kind: "symbol",
        name: symbol?.name ?? expression.name,
        ...(symbol?.valueType === undefined ? {} : { valueType: symbol.valueType }),
        addressSpace: symbol?.addressSpace ?? "unknown",
        span: expression.span,
      };
    }
    case "member": {
      const object = lowerExpression(expression.object, scope);
      return {
        kind: "member",
        object,
        property: expression.property,
        ...optionalValueType(memberValueType(object, expression.property)),
        span: expression.span,
      };
    }
    case "index": {
      const aliased = localPointerAliasIndexExpression(expression, scope);
      if (aliased) return aliased;
      const target = lowerExpression(expression.target, scope);
      return {
        kind: "index",
        target,
        index: lowerExpression(expression.index, scope),
        ...optionalValueType(indexedExpressionValueType(expression, target, scope)),
        addressSpace: expressionAddressSpace(target),
        span: expression.span,
      };
    }
    case "call": {
      if (expression.callee.kind === "identifier" && (expression.callee.name === "sizeof" || expression.callee.name === "alignof")) {
        const value = semanticSizeofAlignofValue(expression.callee.name, expression.args[0], scope);
        if (value !== undefined) return intNumberExpression(value, expression.span);
      }
      const generatedRandom = expression.callee.kind === "identifier" && isSemanticGeneratedRandomCall(expression.callee.name)
        ? semanticGeneratedRandomReturnType(expression.callee.name)
        : undefined;
      const preservePointerArgs = expression.callee.kind === "identifier" &&
        (SEMANTIC_LOCAL_ARRAY_FILL_CALLS.has(expression.callee.name) || SEMANTIC_CURAND_CALLS.has(expression.callee.name) || generatedRandom !== undefined);
      const args = expression.args.map((arg) => preservePointerArgs
        ? lowerExpression(arg, scope)
        : pointerAliasValueExpression(arg, scope, arg.span) ?? lowerExpression(arg, scope));
      if (generatedRandom !== undefined && expression.callee.kind === "identifier") {
        return {
          kind: "call",
          callee: { kind: "symbol", name: expression.callee.name, valueType: generatedRandom, addressSpace: "builtin", span: expression.callee.span },
          args,
          valueType: generatedRandom,
          span: expression.span,
        };
      }
      const cooperativeShuffle = semanticCooperativeShuffleCall(expression, args, scope);
      if (cooperativeShuffle) return cooperativeShuffle;
      if (expression.callee.kind === "identifier" && CUDA_CACHE_HINT_LOADS.has(expression.callee.name)) {
        const load = cacheHintLoadExpression(expression, scope);
        if (load) return load;
      }
      const semanticTextureCall = expression.callee.kind === "identifier" && isSemanticTextureReadCall(expression.callee.name)
        ? expression.callee.name
        : undefined;
      if (
        semanticTextureCall !== undefined &&
        args.length >= semanticTextureReadCoordinateCount(semanticTextureCall) + 1
      ) {
        const coordinateCount = semanticTextureReadCoordinateCount(semanticTextureCall);
        return {
          kind: "texture-read",
          callee: semanticTextureCall,
          texture: args[0]!,
          x: args[1]!,
          y: coordinateCount === 1 ? numberExpression(0, expression.span) : args[2]!,
          ...(coordinateCount === 3 ? { z: args[3]! } : {}),
          valueType: expression.templateValueType ?? "float",
          span: expression.span,
        };
      }
      if (
        expression.callee.kind === "identifier" &&
        (expression.callee.name === "surf2Dread" && args.length === 3 ||
          (expression.callee.name === "surf2DLayeredread" || expression.callee.name === "surf3Dread") && args.length === 4)
      ) {
        return {
          kind: "surface-read",
          callee: expression.callee.name as "surf2Dread" | "surf2DLayeredread" | "surf3Dread",
          surface: args[0]!,
          xBytes: args[1]!,
          y: args[2]!,
          ...(args[3] === undefined ? {} : { z: args[3]! }),
          valueType: expression.templateValueType ?? "float",
          span: expression.span,
        };
      }
      const callee = lowerExpression(expression.callee, scope);
      const cooperativeGroupValueType = semanticCooperativeGroupCallValueType(expression);
      return {
        kind: "call",
        callee,
        args,
        ...(expression.templateValueType === undefined ? {} : { templateValueType: expression.templateValueType }),
        ...optionalValueType(expression.callee.kind === "identifier" && expression.callee.name === "__activemask"
          ? "uint"
          : expression.callee.kind === "identifier" && isCudaVoteCallName(expression.callee.name)
            ? "uint"
          : expression.callee.kind === "identifier" && (
              isCudaArithmeticReduceCallName(expression.callee.name) ||
              isCudaBitwiseReduceCallName(expression.callee.name) ||
              isCudaShuffleCallName(expression.callee.name)
            )
            ? expressionValueType(legacyShuffleCall(expression.callee.name) ? args[0] : args[1]) ?? "uint"
          : expression.callee.kind === "identifier" && (expression.callee.name === "clock" || expression.callee.name === "clock64")
            ? "uint"
          : expression.callee.kind === "identifier" && isAddressSpacePredicateName(expression.callee.name)
            ? "int"
            : cooperativeGroupValueType ?? expression.templateValueType ?? semanticIntrinsicReturnType(expression.callee.kind === "identifier" ? expression.callee.name : undefined, args) ?? expressionValueType(callee) ?? expressionValueType(args[0])),
        span: expression.span,
      };
    }
    case "cast":
      return {
        kind: "cast",
        valueType: expression.valueType,
        pointer: expression.pointer ?? false,
        ...(expression.packedByteLanes === undefined ? {} : { packedByteLanes: expression.packedByteLanes }),
        expression: lowerExpression(expression.expression, scope),
        span: expression.span,
      };
    case "unary": {
      const aliased = expression.operator === "*" ? localPointerAliasDerefExpression(expression.argument, scope, expression.span) : undefined;
      if (aliased) return aliased;
      const argument = lowerExpression(expression.argument, scope);
      return {
        kind: "unary",
        operator: expression.operator,
        argument,
        ...optionalValueType(expression.operator === "&" ? "voidptr" : expression.operator === "!" ? "bool" : expressionValueType(argument)),
        span: expression.span,
      };
    }
    case "binary": {
      const pointerDifference = localPointerAliasDifferenceExpression(expression, scope);
      if (pointerDifference) return pointerDifference;
      const pointerComparison = localPointerAliasComparisonExpression(expression, scope);
      if (pointerComparison) return pointerComparison;
      const left = lowerExpression(expression.left, scope);
      const right = lowerExpression(expression.right, scope);
      return {
        kind: "binary",
        operator: expression.operator,
        left,
        right,
        ...optionalValueType(semanticBinaryResultValueType(expression.operator, left, right)),
        span: expression.span,
      };
    }
    case "conditional": {
      const consequent = lowerExpression(expression.consequent, scope);
      const alternate = lowerExpression(expression.alternate, scope);
      return {
        kind: "conditional",
        condition: lowerExpression(expression.condition, scope),
        consequent,
        alternate,
        ...optionalValueType(expressionValueType(consequent) ?? expressionValueType(alternate)),
        span: expression.span,
      };
    }
    case "assignment": {
      const value = lowerExpression(expression.right, scope);
      return {
        kind: "assignment",
        operator: expression.operator,
        target: lowerExpression(expression.left, scope),
        value,
        ...optionalValueType(expressionValueType(value)),
        span: expression.span,
      };
    }
    case "update": {
      const argument = lowerExpression(expression.argument, scope);
      return { kind: "update", operator: expression.operator, argument, prefix: expression.prefix, ...optionalValueType(expressionValueType(argument)), span: expression.span };
    }
    case "initializer":
      return { kind: "initializer", elements: expression.elements.map((element) => lowerExpression(element, scope)), span: expression.span };
    case "sequence": {
      const expressions = expression.expressions.map((item) => lowerExpression(item, scope));
      return { kind: "sequence", expressions, ...optionalValueType(expressionValueType(expressions.at(-1))), span: expression.span };
    }
  }
}

function semanticCooperativeShuffleCall(
  expression: Extract<CudaLiteExpression, { readonly kind: "call" }>,
  args: readonly SemanticExpression[],
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): Extract<SemanticExpression, { readonly kind: "call" }> | undefined {
  if (expression.callee.kind !== "member" || expression.callee.object.kind !== "identifier") return undefined;
  const op = expression.callee.property;
  if (op !== "shfl" && op !== "shfl_down" && op !== "shfl_up" && op !== "shfl_xor") return undefined;
  const group = scope.get(expression.callee.object.name);
  if (group?.cooperativeGroupKind !== "tile" || args.length !== 2) return undefined;
  const value = args[0]!;
  const index = args[1]!;
  const width = intNumberExpression(group.tileSize ?? 32, expression.span);
  return {
    kind: "call",
    callee: {
      kind: "symbol",
      name: `__${op}`,
      addressSpace: "builtin",
      span: expression.callee.span,
    },
    args: [value, index, width],
    ...optionalValueType(expressionValueType(value)),
    span: expression.span,
  };
}

function semanticCooperativeGroupCallValueType(
  expression: CudaLiteCallExpression,
): CudaLiteScalarType | undefined {
  if (
    expression.callee.kind !== "member" ||
    expression.callee.property !== "thread_rank" &&
    expression.callee.property !== "size" &&
    expression.callee.property !== "meta_group_rank" &&
    expression.callee.property !== "meta_group_size"
  ) return undefined;
  return "int";
}

function lowerForInitStatement(
  statement: Extract<CudaLiteStatement, { readonly kind: "var" }>,
  scope: Map<string, CudaLiteSemanticSymbol>,
): SemanticKernelIrOperation {
  const target = symbolForVar(statement, scope);
  if (target.pointerRoot && target.pointerAddressSpace === "storage") {
    const legacyTarget = semanticSymbolWithoutPointerAlias(target);
    scope.set(legacyTarget.name, legacyTarget);
    return {
      kind: "declare",
      target: legacyTarget,
      ...(statement.init === undefined ? {} : { init: lowerExpression(statement.init, scope) }),
      span: statement.span,
    };
  }
  return lowerStatement(statement, scope);
}

function semanticSymbolWithoutPointerAlias(symbol: CudaLiteSemanticSymbol): CudaLiteSemanticSymbol {
  const {
    pointerRoot: _pointerRoot,
    pointerAddressSpace: _pointerAddressSpace,
    pointerBaseIndices: _pointerBaseIndices,
    pointerValid: _pointerValid,
    ...rest
  } = symbol;
  return rest;
}

function lowerDeviceLaunch(
  statement: CudaLiteKernelLaunchStatement,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticDeviceLaunch {
  return {
    callee: statement.callee,
    grid: statement.grid.map((arg) => lowerExpression(arg, scope)),
    block: statement.block.map((arg) => lowerExpression(arg, scope)),
    args: statement.args.map((arg) => lowerExpression(arg, scope)),
  };
}

function barrierGroupName(expression: Extract<SemanticExpression, { readonly kind: "call" }>): string | undefined {
  if (
    expression.callee.kind === "member" &&
    expression.callee.property === "sync" &&
    expression.callee.object.kind === "symbol"
  ) {
    return expression.callee.object.name;
  }
  if (expression.callee.kind === "symbol" && expression.callee.name.endsWith("::sync")) {
    const group = expression.args[0];
    return group?.kind === "symbol" ? group.name : undefined;
  }
  return undefined;
}

function semanticCallName(callee: SemanticExpression): string | undefined {
  if (callee.kind === "symbol") return callee.name;
  if (callee.kind === "member") {
    const objectName = semanticCallName(callee.object);
    return objectName ? `${objectName}.${callee.property}` : undefined;
  }
  return undefined;
}

function collectDeclaredMemory(operations: readonly SemanticKernelIrOperation[]): readonly CudaLiteSemanticSymbol[] {
  const out: CudaLiteSemanticSymbol[] = [];
  for (const operation of operations) {
    if (operation.kind === "declare" && (operation.target.pointer || operation.target.dimensions.length > 0 || operation.target.addressSpace === "shared")) out.push(operation.target);
    if (operation.kind === "block") out.push(...collectDeclaredMemory(operation.body));
    else if (operation.kind === "branch") out.push(...collectDeclaredMemory(operation.consequent), ...collectDeclaredMemory(operation.alternate));
    else if (operation.kind === "loop") out.push(...collectDeclaredMemory(operation.body));
  }
  return out;
}

function cacheHintLoadExpression(
  expression: Extract<CudaLiteExpression, { readonly kind: "call" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticExpression | undefined {
  const pointer = expression.args[0];
  return pointer === undefined ? undefined : pointerAliasValueExpression(pointer, scope, expression.span);
}

function cacheHintStoreTarget(
  expression: Extract<CudaLiteExpression, { readonly kind: "call" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticMemoryRef | undefined {
  const pointer = expression.args[0];
  if (pointer === undefined) return undefined;
  const target = pointerAliasValueExpression(pointer, scope, pointer.span);
  return target === undefined ? undefined : memoryRefFromExpression(target);
}

function semanticCpAsyncOperation(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticKernelIrOperation | undefined {
  if (expression.kind !== "call" || expression.callee.kind !== "identifier") return undefined;
  const callee = expression.callee.name;
  if (isCudaCpAsyncFenceCall(callee)) return { kind: "copy-fence", callee, span };
  if (!isCudaCpAsyncCopyCall(callee)) return undefined;
  const [targetSource, sourceSource, byteCountSource] = expression.args;
  if (!targetSource || !sourceSource) return undefined;
  const target = semanticPointerArgumentMemoryRef(targetSource, scope);
  const source = semanticPointerArgumentMemoryRef(sourceSource, scope);
  const byteCount = byteCountSource === undefined ? undefined : staticNumberValue(lowerExpression(byteCountSource, scope));
  if (
    !target ||
    !source ||
    target.valueType === undefined ||
    target.valueType !== source.valueType ||
    target.fields.length > 0 ||
    source.fields.length > 0 ||
    byteCount === undefined ||
    !Number.isInteger(byteCount) ||
    byteCount <= 0
  ) return undefined;
  const elementBytes = sizeofCudaType(source.valueType) ?? 0;
  if (elementBytes <= 0 || byteCount % elementBytes !== 0) return undefined;
  const elements = byteCount / elementBytes;
  if (elements < 1 || elements > 16) return undefined;
  return { kind: "copy", source, target, elements, span };
}

function semanticPointerArgumentMemoryRef(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticMemoryRef | undefined {
  const lowered = pointerAliasValueExpression(expression, scope, expression.span) ?? lowerExpression(expression, scope);
  const value = lowered.kind === "unary" && lowered.operator === "&" ? lowered.argument : lowered;
  return memoryRefFromExpression(value);
}

function semanticMathOutVarDeclOperations(
  statement: CudaLiteStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (
    statement.kind !== "var" ||
    statement.storage === "shared" ||
    statement.pointer ||
    statement.dimensions.length > 0 ||
    statement.matrixTile ||
    statement.init?.kind !== "call"
  ) return undefined;

  const target = symbolForVar(statement, scope);
  if (target.pointerRoot) return undefined;
  scope.set(target.name, target);
  const expression = lowerExpression(statement.init, scope);
  if (expression.kind !== "call") return undefined;
  const result = semanticMathOutCallResult(statement.init, expression, scope, statement.span);
  if (!result) return undefined;
  return [
    ...result.sideEffects,
    {
      kind: "declare",
      target,
      init: result.value,
      span: statement.span,
    },
  ];
}

function semanticMathOutAssignmentOperations(
  statement: CudaLiteStatement,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (statement.kind !== "expr" || statement.expression.kind !== "assignment" || statement.expression.operator !== "=" || statement.expression.right.kind !== "call") return undefined;
  const expression = lowerExpression(statement.expression, scope);
  if (expression.kind !== "assignment" || expression.value.kind !== "call") return undefined;
  const target = memoryRefFromExpression(expression.target);
  if (target) return semanticMathOutAssignmentStores(statement.expression.right, expression.value, target, scope, statement.span);
  const result = semanticMathOutCallResult(statement.expression.right, expression.value, scope, statement.span);
  if (!result) return undefined;
  return [
    ...result.sideEffects,
    {
      kind: "expression",
      expression: {
        ...expression,
        value: result.value,
      },
      span: statement.span,
    },
  ];
}

function semanticMathOutCallStatementOperations(
  statement: CudaLiteStatement,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (statement.kind !== "expr" || statement.expression.kind !== "call") return undefined;
  const expression = lowerExpression(statement.expression, scope);
  if (expression.kind !== "call") return undefined;
  const result = semanticMathOutCallResult(statement.expression, expression, scope, statement.span);
  return result?.sideEffects;
}

function semanticMathOutAssignmentBlock(
  source: CudaLiteAssignmentExpression,
  expression: Extract<SemanticExpression, { readonly kind: "assignment" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticKernelIrOperation | undefined {
  if (source.operator !== "=" || source.right.kind !== "call" || expression.value.kind !== "call") return undefined;
  const target = memoryRefFromExpression(expression.target);
  if (!target) return undefined;
  const stores = semanticMathOutAssignmentStores(source.right, expression.value, target, scope, span);
  return stores === undefined ? undefined : { kind: "block", body: stores, span };
}

function semanticMathOutAssignmentStores(
  source: CudaLiteCallExpression,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  target: SemanticMemoryRef,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): readonly SemanticKernelIrOperation[] | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const result = semanticMathOutCallResult(source, expression, scope, span);
  return result === undefined ? undefined : [
    ...result.sideEffects,
    storeOperation(target, result.value, span),
  ];
}

function semanticMathOutCallResult(
  source: CudaLiteCallExpression,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): { readonly sideEffects: readonly SemanticKernelIrOperation[]; readonly value: SemanticExpression } | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  if (isVibMinMaxCallName(expression.callee.name)) return semanticVibMinMaxCallResult(source, expression, scope, span);
  if (isModfCallName(expression.callee.name)) return semanticModfCallResult(source, expression, scope, span);
  if (isFrexpCallName(expression.callee.name)) return semanticFrexpCallResult(source, expression, scope, span);
  if (isRemquoCallName(expression.callee.name)) return semanticRemquoCallResult(source, expression, scope, span);
  return undefined;
}

function semanticModfCallResult(
  source: CudaLiteCallExpression,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): { readonly sideEffects: readonly SemanticKernelIrOperation[]; readonly value: SemanticExpression } | undefined {
  const value = expression.args[0];
  const intpartTarget = source.args[1] === undefined ? undefined : pointerAliasValueExpression(source.args[1], scope, source.args[1].span);
  if (value === undefined || !intpartTarget || !semanticExpressionSideEffectFree(value)) return undefined;
  const intpartRef = memoryRefFromExpression(intpartTarget);
  if (!intpartRef) return undefined;
  const temp = tempScalarSymbol("__bg.modf.value", span, "float");
  const tempValue = semanticSymbolExpression(temp, value.span);
  return {
    sideEffects: [
      { kind: "declare", target: temp, init: value, span },
      storeOperation(intpartRef, unaryFloatCallExpression("__bg_modf_intpart", tempValue, expression.span), span),
    ],
    value: unaryFloatCallExpression("__bg_modf_fraction", tempValue, expression.span),
  };
}

function semanticFrexpCallResult(
  source: CudaLiteCallExpression,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): { readonly sideEffects: readonly SemanticKernelIrOperation[]; readonly value: SemanticExpression } | undefined {
  const value = expression.args[0];
  const exponentTarget = source.args[1] === undefined ? undefined : mathOutTargetExpressionFromSource(source.args[1], scope);
  if (value === undefined || !exponentTarget || !semanticExpressionSideEffectFree(value)) return undefined;
  const temp = tempScalarSymbol("__bg.frexp.value", span, "float");
  const tempValue = semanticSymbolExpression(temp, value.span);
  return {
    sideEffects: [
      { kind: "declare", target: temp, init: value, span },
      mathOutStoreOrAssignOperation(exponentTarget, unaryIntCallExpression("__bg_frexp_exponent", tempValue, expression.span), span),
    ],
    value: unaryFloatCallExpression("__bg_frexp_mantissa", tempValue, expression.span),
  };
}

function semanticRemquoCallResult(
  source: CudaLiteCallExpression,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): { readonly sideEffects: readonly SemanticKernelIrOperation[]; readonly value: SemanticExpression } | undefined {
  const dividend = expression.args[0];
  const divisor = expression.args[1] ? staticNumberValue(expression.args[1]) : undefined;
  const quotientTarget = source.args[2] === undefined ? undefined : pointerAliasValueExpression(source.args[2], scope, source.args[2].span);
  if (dividend === undefined || divisor === undefined || divisor === 0 || !quotientTarget || !semanticExpressionSideEffectFree(dividend)) return undefined;
  const quotientRef = memoryRefFromExpression(quotientTarget);
  if (!quotientRef) return undefined;
  const temp = tempScalarSymbol("__bg.remquo.dividend", span, "float");
  const tempValue = semanticSymbolExpression(temp, dividend.span);
  const divisorValue = numberExpression(divisor, expression.span);
  return {
    sideEffects: [
      { kind: "declare", target: temp, init: dividend, span },
      storeOperation(quotientRef, binaryIntCallExpression("__bg_remquo_quotient", tempValue, divisorValue, expression.span), span),
    ],
    value: binaryFloatCallExpression("__bg_remquo_remainder", tempValue, divisorValue, expression.span),
  };
}

function semanticVibMinMaxCallResult(
  source: CudaLiteCallExpression,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): { readonly sideEffects: readonly SemanticKernelIrOperation[]; readonly value: SemanticExpression } | undefined {
  if (expression.callee.kind !== "symbol") return undefined;
  const left = expression.args[0];
  const right = expression.args[1];
  if (!left || !right || !semanticExpressionSideEffectFree(left) || !semanticExpressionSideEffectFree(right)) return undefined;
  const name = expression.callee.name;
  const vib = cudaVibMinMaxInfo(name);
  if (!vib) return undefined;
  const predicateArgs = source.args.slice(2);
  const predicateTargets = predicateArgs.map((arg) => mathOutTargetExpressionFromSource(arg, scope));
  if (predicateTargets.some((target) => target === undefined)) return undefined;
  const value = semanticCallExpression(name, [left, right], vib.valueType, expression.span);
  if (vib.packed) {
    const lo = vibLanePredicateExpression(left, right, vib.signed, vib.choose, 0, expression.span);
    const hi = vibLanePredicateExpression(left, right, vib.signed, vib.choose, 16, expression.span);
    const [hiTarget, loTarget] = predicateTargets;
    if (!hiTarget || !loTarget) return undefined;
    return {
      sideEffects: [
        mathOutStoreOrAssignOperation(hiTarget, hi, span),
        mathOutStoreOrAssignOperation(loTarget, lo, span),
      ],
      value,
    };
  }
  const [predicateTarget] = predicateTargets;
  if (!predicateTarget) return undefined;
  return {
    sideEffects: [mathOutStoreOrAssignOperation(predicateTarget, vibScalarPredicateExpression(left, right, vib.signed, vib.choose, expression.span), span)],
    value,
  };
}

function semanticModfStore(
  source: Extract<CudaLiteExpression, { readonly kind: "call" }>,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticKernelIrOperation | undefined {
  if (expression.callee.kind !== "symbol" || !isModfCallName(expression.callee.name)) return undefined;
  const value = expression.args[0];
  const target = source.args[1] === undefined ? undefined : pointerAliasValueExpression(source.args[1], scope, source.args[1].span);
  if (!value || !target) return undefined;
  if (!isFiniteStaticNumberExpression(value)) return undefined;
  const targetRef = memoryRefFromExpression(target);
  if (!targetRef) return undefined;
  return {
    kind: "store",
    target: targetRef,
    value: mathCallExpression("trunc", value, value.span),
    operator: "=",
    reads: collectMemoryRefs(value),
    span,
  };
}

function semanticRemquoStore(
  source: Extract<CudaLiteExpression, { readonly kind: "call" }>,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticKernelIrOperation | undefined {
  if (expression.callee.kind !== "symbol" || !isRemquoCallName(expression.callee.name)) return undefined;
  const dividend = expression.args[0] ? staticNumberValue(expression.args[0]) : undefined;
  const divisor = expression.args[1] ? staticNumberValue(expression.args[1]) : undefined;
  const target = source.args[2] === undefined ? undefined : pointerAliasValueExpression(source.args[2], scope, source.args[2].span);
  if (dividend === undefined || divisor === undefined || divisor === 0 || !target) return undefined;
  const targetRef = memoryRefFromExpression(target);
  if (!targetRef) return undefined;
  const quotient = roundTiesToEvenNumber(dividend / divisor);
  return {
    kind: "store",
    target: targetRef,
    value: intNumberExpression(quotient, expression.span),
    operator: "=",
    reads: [],
    span,
  };
}

function semanticFrexpStore(
  source: Extract<CudaLiteExpression, { readonly kind: "call" }>,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticKernelIrOperation | undefined {
  if (expression.callee.kind !== "symbol" || !isFrexpCallName(expression.callee.name)) return undefined;
  const value = expression.args[0] ? staticNumberValue(expression.args[0]) : undefined;
  const target = source.args[1] === undefined ? undefined : mathOutTargetExpressionFromSource(source.args[1], scope);
  if (value === undefined) {
    const dynamic = semanticFrexpCallResult(source, expression, scope, span);
    return dynamic === undefined ? undefined : { kind: "block", body: dynamic.sideEffects, span };
  }
  if (!target) return undefined;
  const exponent = frexpExponentForFiniteNumber(value);
  return mathOutStoreOrAssignOperation(target, intNumberExpression(exponent, expression.span), span);
}

function semanticSincosStores(
  source: Extract<CudaLiteExpression, { readonly kind: "call" }>,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticKernelIrOperation | undefined {
  if (expression.callee.kind !== "symbol" || !isSincosCallName(expression.callee.name)) return undefined;
  const value = expression.args[0];
  const sinTarget = source.args[1] === undefined ? undefined : pointerAliasValueExpression(source.args[1], scope, source.args[1].span);
  const cosTarget = source.args[2] === undefined ? undefined : pointerAliasValueExpression(source.args[2], scope, source.args[2].span);
  if (!value || !sinTarget || !cosTarget) return undefined;
  const sinRef = memoryRefFromExpression(sinTarget);
  const cosRef = memoryRefFromExpression(cosTarget);
  if (!sinRef || !cosRef) return undefined;
  const angle = isSincosPiCallName(expression.callee.name)
    ? multiplyFloatExpressions(numberExpression(Math.PI, value.span), value, value.span)
    : value;
  return {
    kind: "block",
    body: [
      {
        kind: "store",
        target: sinRef,
        value: mathCallExpression("sin", angle, value.span),
        operator: "=",
        reads: collectMemoryRefs(angle),
        span,
      },
      {
        kind: "store",
        target: cosRef,
        value: mathCallExpression("cos", angle, value.span),
        operator: "=",
        reads: collectMemoryRefs(angle),
        span,
      },
    ],
    span,
  };
}

function mathCallExpression(name: string, value: SemanticExpression, span: SourceSpan): SemanticExpression {
  return {
    kind: "call",
    callee: { kind: "symbol", name, valueType: "float", addressSpace: "builtin", span },
    args: [value],
    valueType: "float",
    span,
  };
}

function semanticCallExpression(name: string, args: readonly SemanticExpression[], valueType: Exclude<CudaLiteScalarType, "void">, span: SourceSpan): SemanticExpression {
  return {
    kind: "call",
    callee: { kind: "symbol", name, valueType, addressSpace: "builtin", span },
    args,
    valueType,
    span,
  };
}

function vibScalarPredicateExpression(left: SemanticExpression, right: SemanticExpression, signed: boolean, choose: "max" | "min", span: SourceSpan): SemanticExpression {
  const valueType: CudaLiteScalarType = signed ? "int" : "uint";
  return {
    kind: "binary",
    operator: choose === "max" ? ">=" : "<=",
    left: castScalarExpression(left, valueType, span),
    right: castScalarExpression(right, valueType, span),
    valueType: "bool",
    span,
  };
}

function vibLanePredicateExpression(left: SemanticExpression, right: SemanticExpression, signed: boolean, choose: "max" | "min", shift: 0 | 16, span: SourceSpan): SemanticExpression {
  return {
    kind: "binary",
    operator: choose === "max" ? ">=" : "<=",
    left: castScalarExpression(semanticCallExpression(signed ? "__bg_i16_lane" : "__bg_u16_lane", [left, intNumberExpression(shift, span)], signed ? "int" : "uint", span), signed ? "int" : "uint", span),
    right: castScalarExpression(semanticCallExpression(signed ? "__bg_i16_lane" : "__bg_u16_lane", [right, intNumberExpression(shift, span)], signed ? "int" : "uint", span), signed ? "int" : "uint", span),
    valueType: "bool",
    span,
  };
}

function castScalarExpression(expression: SemanticExpression, valueType: Exclude<CudaLiteScalarType, "void">, span: SourceSpan): SemanticExpression {
  return {
    kind: "cast",
    valueType,
    pointer: false,
    expression,
    span,
  };
}

function unaryFloatCallExpression(name: string, value: SemanticExpression, span: SourceSpan): SemanticExpression {
  return mathCallExpression(name, value, span);
}

function unaryIntCallExpression(name: string, value: SemanticExpression, span: SourceSpan): SemanticExpression {
  return {
    kind: "call",
    callee: { kind: "symbol", name, valueType: "int", addressSpace: "builtin", span },
    args: [value],
    valueType: "int",
    span,
  };
}

function binaryFloatCallExpression(name: string, left: SemanticExpression, right: SemanticExpression, span: SourceSpan): SemanticExpression {
  return {
    kind: "call",
    callee: { kind: "symbol", name, valueType: "float", addressSpace: "builtin", span },
    args: [left, right],
    valueType: "float",
    span,
  };
}

function binaryIntCallExpression(name: string, left: SemanticExpression, right: SemanticExpression, span: SourceSpan): SemanticExpression {
  return {
    kind: "call",
    callee: { kind: "symbol", name, valueType: "int", addressSpace: "builtin", span },
    args: [left, right],
    valueType: "int",
    span,
  };
}

function multiplyFloatExpressions(left: SemanticExpression, right: SemanticExpression, span: SourceSpan): SemanticExpression {
  return {
    kind: "binary",
    operator: "*",
    left,
    right,
    valueType: "float",
    span,
  };
}

function numberExpression(value: number, span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value, valueType: "float", span };
}

function intNumberExpression(value: number, span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value, valueType: "int", span };
}

function semanticSizeofAlignofValue(
  kind: "sizeof" | "alignof",
  expression: CudaLiteExpression | undefined,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): number | undefined {
  if (!expression) return undefined;
  const layout = kind === "sizeof" ? sizeofCudaType : alignofCudaType;
  if (expression.kind === "identifier") {
    const typeLayout = layout(expression.name);
    if (typeLayout !== undefined) return typeLayout;
    const symbol = scope.get(expression.name);
    if (!symbol) return undefined;
    const elementLayout = layout(symbol.valueType ?? "");
    if (elementLayout === undefined) return undefined;
    if (kind === "alignof") return elementLayout;
    if (symbol.pointer && symbol.dimensions.length === 0) return sizeofCudaType("voidptr") ?? 4;
    const elements = totalElements(symbol.dimensions);
    return elementLayout * Math.max(1, elements);
  }
  const valueType = expressionValueType(lowerExpression(expression, scope));
  return valueType === undefined ? undefined : layout(valueType);
}

function tempScalarSymbol(prefix: string, span: SourceSpan, valueType: CudaLiteScalarType): CudaLiteSemanticSymbol {
  return {
    name: `${prefix}.${span.start}.${span.end}`,
    kind: "local",
    valueType,
    addressSpace: "local",
    dimensions: [],
    span,
  };
}

function isFiniteStaticNumberExpression(expression: SemanticExpression): boolean {
  return staticNumberValue(expression) !== undefined;
}

function staticNumberValue(expression: SemanticExpression): number | undefined {
  if (
    expression.kind === "literal" &&
    expression.literalKind === "number" &&
    typeof expression.value === "number" &&
    Number.isFinite(expression.value)
  ) {
    return expression.value;
  }
  if (expression.kind === "unary" && expression.operator === "-") {
    const value = staticNumberValue(expression.argument);
    return value === undefined ? undefined : -value;
  }
  return undefined;
}

function semanticExpressionSideEffectFree(expression: SemanticExpression): boolean {
  switch (expression.kind) {
    case "assignment":
    case "update":
    case "sequence":
      return false;
    case "literal":
    case "symbol":
      return true;
    case "member":
      return semanticExpressionSideEffectFree(expression.object);
    case "index":
      return semanticExpressionSideEffectFree(expression.target) && semanticExpressionSideEffectFree(expression.index);
    case "call":
      return semanticExpressionSideEffectFree(expression.callee) && expression.args.every(semanticExpressionSideEffectFree);
    case "texture-read":
      return semanticExpressionSideEffectFree(expression.texture) &&
        semanticExpressionSideEffectFree(expression.x) &&
        semanticExpressionSideEffectFree(expression.y) &&
        (expression.z === undefined || semanticExpressionSideEffectFree(expression.z));
    case "surface-read":
      return semanticExpressionSideEffectFree(expression.surface) &&
        semanticExpressionSideEffectFree(expression.xBytes) &&
        semanticExpressionSideEffectFree(expression.y) &&
        (expression.z === undefined || semanticExpressionSideEffectFree(expression.z));
    case "cast":
      return semanticExpressionSideEffectFree(expression.expression);
    case "unary":
      return semanticExpressionSideEffectFree(expression.argument);
    case "binary":
      return semanticExpressionSideEffectFree(expression.left) && semanticExpressionSideEffectFree(expression.right);
    case "conditional":
      return semanticExpressionSideEffectFree(expression.condition) &&
        semanticExpressionSideEffectFree(expression.consequent) &&
        semanticExpressionSideEffectFree(expression.alternate);
    case "initializer":
      return expression.elements.every(semanticExpressionSideEffectFree);
  }
}

function storeOperation(target: SemanticMemoryRef, value: SemanticExpression, span: SourceSpan): SemanticKernelIrOperation {
  return {
    kind: "store",
    target,
    value,
    operator: "=",
    reads: collectMemoryRefs(value),
    span,
  };
}

function frexpExponentForFiniteNumber(value: number): number {
  return value === 0 ? 0 : Math.floor(Math.log2(Math.abs(value))) + 1;
}

function roundTiesToEvenNumber(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

function pointerAliasValueExpression(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticExpression | undefined {
  if (expression.kind === "identifier") {
    const symbol = scope.get(expression.name);
    if (symbol?.kind === "shared" && !symbol.pointer && symbol.dimensions.length > 0) return undefined;
  }
  if (isDirectSharedPointerAddress(expression, scope)) return undefined;
  const alias = localPointerAliasForInitializer(expression, scope);
  if (!alias?.pointerRoot || !semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace) || alias.pointerBaseIndices?.length !== 1) return undefined;
  const root = scope.get(alias.pointerRoot);
  if (!root || !semanticPointerAliasAddressSpaceSupported(root.addressSpace)) return undefined;
  const aliasValueType = pointerAliasTargetValueType(expression, scope) ?? root.valueType;
  return {
    kind: "index",
    target: semanticSymbolExpression(root, span),
    index: alias.pointerBaseIndices[0]!,
    ...optionalValueType(aliasValueType),
    addressSpace: root.addressSpace,
    ...(alias.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
    span,
  };
}

function pointerAliasOffsetForBaseUnit(
  alias: SemanticPointerAlias,
  valueType: CudaLiteScalarType | undefined,
  offset: SemanticExpression,
  span: SourceSpan,
): SemanticExpression {
  if (alias.pointerBaseUnitBytes !== undefined) return multiplyIndexExpression(offset, alias.pointerBaseUnitBytes, span);
  return alias.pointerBaseIsScalarLane === true && isCudaVectorType(valueType)
    ? multiplyIndexExpression(offset, cudaVectorLaneCount(valueType), span)
    : offset;
}

function isDirectSharedPointerAddress(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): boolean {
  if (expression.kind === "cast" && expression.pointer) return isDirectSharedPointerAddress(expression.expression, scope);
  if (expression.kind === "call" && localPointerIdentityCallName(expression.callee)) {
    const argument = expression.args[0];
    return argument !== undefined && isDirectSharedPointerAddress(argument, scope);
  }
  if (expression.kind !== "unary" || expression.operator !== "&") return false;
  return localPointerAliasForInitializer(expression, scope)?.pointerAddressSpace === "shared";
}

function mathOutTargetExpressionFromSource(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticExpression | undefined {
  const lowered = lowerExpression(expression, scope);
  if (lowered.kind === "unary" && lowered.operator === "&") return lowered.argument;
  return pointerAliasValueExpression(expression, scope, expression.span);
}

function mathOutStoreOrAssignOperation(target: SemanticExpression, value: SemanticExpression, span: SourceSpan): SemanticKernelIrOperation {
  const ref = memoryRefFromExpression(target);
  if (ref) return storeOperation(ref, value, span);
  return {
    kind: "expression",
    expression: {
      kind: "assignment",
      operator: "=",
      target,
      value,
      ...optionalValueType(expressionValueType(target)),
      span,
    },
    span,
  };
}

function atomicTargetFromCall(expression: Extract<SemanticExpression, { readonly kind: "call" }>): SemanticMemoryRef | undefined {
  const firstArg = expression.args[0];
  if (firstArg === undefined) return undefined;
  if (firstArg.kind === "unary" && firstArg.operator === "&") return memoryRefFromExpression(firstArg.argument);
  return memoryRefFromExpression(firstArg);
}

function collectMemoryRefs(expression: SemanticExpression): readonly SemanticMemoryRef[] {
  const refs: SemanticMemoryRef[] = [];
  collectMemoryRefsInto(expression, refs);
  return dedupeMemoryRefs(refs);
}

function collectMemoryRefsInto(expression: SemanticExpression, refs: SemanticMemoryRef[]): void {
  const ref = memoryRefFromExpression(expression);
  if (ref) {
    refs.push(ref);
    for (const index of ref.indices) collectMemoryRefsInto(index, refs);
    return;
  }
  switch (expression.kind) {
    case "literal":
    case "symbol":
      return;
    case "member":
      collectMemoryRefsInto(expression.object, refs);
      return;
    case "index":
      collectMemoryRefsInto(expression.target, refs);
      collectMemoryRefsInto(expression.index, refs);
      return;
    case "call":
      collectMemoryRefsInto(expression.callee, refs);
      for (const arg of expression.args) collectMemoryRefsInto(arg, refs);
      return;
    case "texture-read":
      collectMemoryRefsInto(expression.texture, refs);
      collectMemoryRefsInto(expression.x, refs);
      collectMemoryRefsInto(expression.y, refs);
      if (expression.z) collectMemoryRefsInto(expression.z, refs);
      return;
    case "surface-read":
      collectMemoryRefsInto(expression.surface, refs);
      collectMemoryRefsInto(expression.xBytes, refs);
      collectMemoryRefsInto(expression.y, refs);
      if (expression.z) collectMemoryRefsInto(expression.z, refs);
      return;
    case "cast":
      collectMemoryRefsInto(expression.expression, refs);
      return;
    case "unary":
    case "update":
      collectMemoryRefsInto(expression.argument, refs);
      return;
    case "binary":
      collectMemoryRefsInto(expression.left, refs);
      collectMemoryRefsInto(expression.right, refs);
      return;
    case "conditional":
      collectMemoryRefsInto(expression.condition, refs);
      collectMemoryRefsInto(expression.consequent, refs);
      collectMemoryRefsInto(expression.alternate, refs);
      return;
    case "assignment":
      collectMemoryRefsInto(expression.target, refs);
      collectMemoryRefsInto(expression.value, refs);
      return;
    case "initializer":
      for (const element of expression.elements) collectMemoryRefsInto(element, refs);
      return;
    case "sequence":
      for (const item of expression.expressions) collectMemoryRefsInto(item, refs);
      return;
  }
}

function memoryRefFromExpression(expression: SemanticExpression): SemanticMemoryRef | undefined {
  const parts = flattenMemoryRef(expression);
  if (!parts || !isMemoryAddressSpace(parts.base.addressSpace, parts.indices.length)) return undefined;
  const valueType = expressionValueType(expression);
  return {
    base: parts.base.name,
    addressSpace: parts.base.addressSpace,
    ...(valueType === undefined ? {} : { valueType }),
    ...(expression.kind === "member" ? optionalContainerValueType(expressionValueType(expression.object)) : {}),
    ...(expression.kind === "index" && expression.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
    ...(expression.kind === "index" && expression.pointerBaseUnitBytes !== undefined ? { pointerBaseUnitBytes: expression.pointerBaseUnitBytes } : {}),
    ...(expression.kind === "index" ? optionalPackedByteLanes(expression.packedByteLanes) : {}),
    indices: parts.indices,
    fields: parts.fields,
    span: expression.span,
  };
}

function optionalPackedByteLanes(packedByteLanes: 2 | 3 | 4 | undefined): { readonly packedByteLanes?: 2 | 3 | 4 } {
  return packedByteLanes === undefined ? {} : { packedByteLanes };
}

function optionalContainerValueType(valueType: CudaLiteScalarType | undefined): { readonly containerValueType: CudaLiteScalarType } | Record<string, never> {
  return valueType === undefined ? {} : { containerValueType: valueType };
}

function flattenMemoryRef(expression: SemanticExpression): {
  readonly base: Extract<SemanticExpression, { readonly kind: "symbol" }>;
  readonly indices: readonly SemanticExpression[];
  readonly fields: readonly string[];
} | undefined {
  if (expression.kind === "symbol") return { base: expression, indices: [], fields: [] };
  if (expression.kind === "index") {
    const target = flattenMemoryRef(expression.target);
    if (!target) return undefined;
    return { ...target, indices: [...target.indices, expression.index] };
  }
  if (expression.kind === "member") {
    const object = flattenMemoryRef(expression.object);
    if (!object) return undefined;
    return { ...object, fields: [...object.fields, expression.property] };
  }
  return undefined;
}

function isMemoryAddressSpace(addressSpace: SemanticAddressSpace, indexCount: number): boolean {
  return addressSpace === "storage"
    || addressSpace === "constant"
    || addressSpace === "device-global"
    || addressSpace === "shared"
    || (addressSpace === "local" && indexCount > 0)
    || addressSpace === "pool"
    || addressSpace === "texture"
    || addressSpace === "surface";
}

function dedupeMemoryRefs(refs: readonly SemanticMemoryRef[]): readonly SemanticMemoryRef[] {
  const seen = new Set<string>();
  const out: SemanticMemoryRef[] = [];
  for (const ref of refs) {
    const key = `${ref.base}:${ref.addressSpace}:${ref.span.start}:${ref.span.end}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(ref);
  }
  return out;
}

function symbolForParam(param: CudaLiteParam): CudaLiteSemanticSymbol {
  return {
    name: param.name,
    kind: "param",
    valueType: param.valueType,
    pointer: param.pointer,
    constant: param.constant,
    ...(param.cooperativeGroupKind === undefined ? {} : { cooperativeGroupKind: param.cooperativeGroupKind }),
    ...(param.tileSize === undefined ? {} : { tileSize: param.tileSize }),
    dimensions: [],
    addressSpace: paramAddressSpace(param),
    span: param.span,
  };
}

function symbolForConstant(constant: CudaLiteGlobalConstant): CudaLiteSemanticSymbol {
  return {
    name: constant.name,
    kind: "constant",
    valueType: constant.valueType,
    pointer: false,
    constant: true,
    initialized: constant.init !== undefined,
    ...(constant.init === undefined ? {} : { init: lowerExpression(constant.init, new Map()) }),
    dimensions: constant.dimensions,
    addressSpace: "constant",
    span: constant.span,
  };
}

function symbolForDeviceGlobal(global: CudaLiteDeviceGlobal): CudaLiteSemanticSymbol {
  return {
    name: global.name,
    kind: "device-global",
    valueType: global.valueType,
    pointer: global.dimensions.length > 0,
    constant: false,
    initialized: global.init !== undefined,
    ...(global.init === undefined ? {} : { init: lowerExpression(global.init, new Map()) }),
    dimensions: global.dimensions,
    addressSpace: "device-global",
    span: global.span,
  };
}

function symbolForTexture(texture: CudaLiteTexture2D): CudaLiteSemanticSymbol {
  return {
    name: texture.name,
    kind: "texture",
    valueType: "texture2d",
    pointer: false,
    constant: true,
    dimensions: [],
    addressSpace: "texture",
    span: texture.span,
  };
}

function symbolForFunction(
  fn: CudaLiteDeviceFunction,
  globalScope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): CudaLiteSemanticFunction {
  const scope = new Map(globalScope);
  const params = fn.params.map((param) => symbolForFunctionParam(param, fn.name, globalScope.has(param.name)));
  for (const [index, param] of params.entries()) scope.set(fn.params[index]!.name, param);
  return {
    name: fn.name,
    returnType: fn.returnType,
    params,
    body: lowerStatements(fn.body, scope),
    span: fn.span,
  };
}

function symbolForFunctionDeclaration(fn: CudaLiteDeviceFunction): CudaLiteSemanticSymbol {
  return {
    name: fn.name,
    kind: "function",
    valueType: fn.returnType,
    pointer: false,
    constant: true,
    dimensions: [],
    addressSpace: "function",
    span: fn.span,
  };
}

function symbolForSemanticFunctionDeclaration(fn: CudaLiteSemanticFunction): CudaLiteSemanticSymbol {
  return {
    name: fn.name,
    kind: "function",
    valueType: fn.returnType,
    pointer: false,
    constant: true,
    dimensions: [],
    addressSpace: "function",
    span: fn.span,
  };
}

function symbolForFunctionParam(param: CudaLiteParam, functionName: string, collidesWithGlobal: boolean): CudaLiteSemanticSymbol {
  const symbol = symbolForParam(param);
  if (symbol.addressSpace === "texture" || symbol.addressSpace === "surface") return symbol;
  if (symbol.pointer && symbol.addressSpace === "storage" && symbol.valueType === "uchar" && collidesWithGlobal) {
    return { ...symbol, name: `__bg_param_${functionName}_${param.name}_${param.span.start}` };
  }
  if (symbol.pointer) return symbol;
  return { ...symbol, addressSpace: "local" };
}

function symbolForVar(
  statement: CudaLiteVarDecl,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): CudaLiteSemanticSymbol {
  const pointerAlias = statement.pointer ? localPointerAliasForInitializer(statement.init, scope) : undefined;
  return {
    name: statement.name,
    kind: statement.storage === "shared" ? "shared" : "local",
    valueType: statement.valueType,
    pointer: statement.pointer,
    ...(statement.packedByteLanes === undefined ? {} : { packedByteLanes: statement.packedByteLanes }),
    ...(pointerAlias === undefined ? {} : pointerAlias),
    constant: false,
    dimensions: statement.dimensions,
    addressSpace: statement.storage,
    span: statement.span,
  };
}

function localPointerAliasForInitializer(
  expression: CudaLiteExpression | undefined,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): Pick<CudaLiteSemanticSymbol, "pointerRoot" | "pointerAddressSpace" | "pointerBaseIndices" | "pointerBaseIsScalarLane" | "pointerBaseUnitBytes" | "pointerValid"> | undefined {
  if (!expression) return undefined;
  if (expression.kind === "cast" && expression.pointer) {
    const alias = localPointerAliasForInitializer(expression.expression, scope);
    const sourceType = pointerAliasTargetValueType(expression.expression, scope);
    const targetBytes = expression.pointerElementBytes ?? sizeofCudaType(expression.valueType);
    if (alias && sourceType === "uchar" && targetBytes !== undefined && targetBytes > 1) {
      return { ...alias, pointerBaseUnitBytes: targetBytes };
    }
    if (
      alias?.pointerBaseIndices?.length === 1 &&
      isCudaVectorType(sourceType) &&
      !isCudaVectorType(expression.valueType)
    ) {
      return {
        ...alias,
        pointerBaseIndices: [multiplyIndexExpression(alias.pointerBaseIndices[0]!, cudaVectorLaneCount(sourceType), expression.span)],
        pointerBaseIsScalarLane: true,
      };
    }
    return alias;
  }
  if (expression.kind === "call" && localPointerIdentityCallName(expression.callee)) {
    return localPointerAliasForInitializer(expression.args[0], scope);
  }
  if (expression.kind === "conditional") {
    const consequent = localPointerAliasForInitializer(expression.consequent, scope);
    const alternate = localPointerAliasForInitializer(expression.alternate, scope);
    const condition = lowerExpression(expression.condition, scope);
    const consequentNull = isNullPointerLiteral(expression.consequent);
    const alternateNull = isNullPointerLiteral(expression.alternate);
    const nonNull = consequent ?? alternate;
    if ((consequentNull || alternateNull) && nonNull?.pointerRoot && semanticPointerAliasAddressSpaceSupported(nonNull.pointerAddressSpace) && nonNull.pointerBaseIndices?.length === 1) {
      return {
        pointerRoot: nonNull.pointerRoot,
        pointerAddressSpace: nonNull.pointerAddressSpace,
        pointerBaseIndices: nonNull.pointerBaseIndices,
        ...(nonNull.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
        ...(nonNull.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: nonNull.pointerBaseUnitBytes }),
        pointerValid: consequentNull ? negateExpression(condition, expression.span) : condition,
      };
    }
    if (!consequent || !alternate) return undefined;
    const root = consequent.pointerRoot;
    const addressSpace = consequent.pointerAddressSpace;
    if (
      root === undefined ||
      addressSpace === undefined ||
      root !== alternate.pointerRoot ||
      addressSpace !== alternate.pointerAddressSpace ||
      consequent.pointerBaseIndices?.length !== 1 ||
      alternate.pointerBaseIndices?.length !== 1 ||
      consequent.pointerBaseIsScalarLane !== alternate.pointerBaseIsScalarLane
      || consequent.pointerBaseUnitBytes !== alternate.pointerBaseUnitBytes
    ) {
      return undefined;
    }
    return {
      pointerRoot: root,
      pointerAddressSpace: addressSpace,
      pointerBaseIndices: [{
        kind: "conditional",
        condition,
        consequent: consequent.pointerBaseIndices[0]!,
        alternate: alternate.pointerBaseIndices[0]!,
        valueType: "int",
        span: expression.span,
      }],
      ...(consequent.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
      ...(consequent.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: consequent.pointerBaseUnitBytes }),
    };
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const left = localPointerAliasForInitializer(expression.left, scope);
    if (left?.pointerRoot && semanticPointerAliasAddressSpaceSupported(left.pointerAddressSpace) && left.pointerBaseIndices?.length === 1) {
      const right = lowerExpression(expression.right, scope);
      const offset = pointerAliasOffsetForBaseUnit(left, pointerAliasTargetValueType(expression.left, scope), right, expression.span);
      return {
        pointerRoot: left.pointerRoot,
        pointerAddressSpace: left.pointerAddressSpace,
        pointerBaseIndices: [expression.operator === "+"
          ? addIndexExpressions(left.pointerBaseIndices[0]!, offset, expression.span)
          : subtractIndexExpressions(left.pointerBaseIndices[0]!, offset, expression.span)],
        ...(left.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
        ...(left.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: left.pointerBaseUnitBytes }),
      };
    }
    if (expression.operator === "+") {
      const right = localPointerAliasForInitializer(expression.right, scope);
      if (right?.pointerRoot && semanticPointerAliasAddressSpaceSupported(right.pointerAddressSpace) && right.pointerBaseIndices?.length === 1) {
        const leftOffset = lowerExpression(expression.left, scope);
        const offset = pointerAliasOffsetForBaseUnit(right, pointerAliasTargetValueType(expression.right, scope), leftOffset, expression.span);
        return {
          pointerRoot: right.pointerRoot,
          pointerAddressSpace: right.pointerAddressSpace,
          pointerBaseIndices: [addIndexExpressions(right.pointerBaseIndices[0]!, offset, expression.span)],
          ...(right.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
          ...(right.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: right.pointerBaseUnitBytes }),
        };
      }
    }
  }
  if (expression.kind === "identifier") {
    const root = scope.get(expression.name);
    if (root?.pointerRoot && semanticPointerAliasAddressSpaceSupported(root.pointerAddressSpace) && root.pointerBaseIndices?.length === 1) {
      return {
        pointerRoot: root.pointerRoot,
        pointerAddressSpace: root.pointerAddressSpace,
        pointerBaseIndices: root.pointerBaseIndices,
        ...(root.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
        ...(root.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: root.pointerBaseUnitBytes }),
      };
    }
    if (root?.kind === "param" && root.pointer && root.addressSpace === "storage") {
      return {
        pointerRoot: root.name,
        pointerAddressSpace: root.addressSpace,
        pointerBaseIndices: [zeroExpression(expression.span)],
      };
    }
    if (!root || (root.kind !== "local" && root.kind !== "shared") || root.dimensions.length !== 1 || root.pointer) return undefined;
    return {
      pointerRoot: root.name,
      pointerAddressSpace: root.addressSpace,
      pointerBaseIndices: [zeroExpression(expression.span)],
    };
  }
  if (expression.kind === "index" && expression.target.kind === "identifier") {
    const target = scope.get(expression.target.name);
    const slot = staticPointerArrayIndex(expression.index);
    const alias = slot === undefined ? undefined : target?.pointerArrayAliases?.[slot];
    if (alias?.pointerRoot && semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace) && alias.pointerBaseIndices?.length === 1) {
      return alias;
    }
  }
  if (expression.kind !== "unary" || expression.operator !== "&") return undefined;
  if (expression.argument.kind === "unary" && expression.argument.operator === "*") {
    const alias = localPointerAliasForInitializer(expression.argument.argument, scope);
    if (semanticPointerAliasAddressSpaceSupported(alias?.pointerAddressSpace)) return alias;
  }
  const ref = localPointerAliasRoot(expression.argument, scope);
  if (!ref || !semanticPointerAliasAddressSpaceSupported(ref.root.addressSpace)) return undefined;
  return {
    pointerRoot: ref.root.name,
    pointerAddressSpace: ref.root.addressSpace,
    pointerBaseIndices: ref.indices,
  };
}

function localPointerIdentityCallName(expression: CudaLiteExpression): string | undefined {
  if (expression.kind === "identifier") {
    return isCudaPointerIdentityCallName(expression.name) ? expression.name : undefined;
  }
  return undefined;
}

function isLocalPointerAliasPlaceholder(statement: CudaLiteStatement): statement is Extract<CudaLiteStatement, { readonly kind: "var" }> {
  return statement.kind === "var" &&
    statement.storage === "local" &&
    statement.pointer &&
    statement.dimensions.length === 0 &&
    (statement.init === undefined || isNullPointerLiteral(statement.init));
}

function semanticPointerAliasAddressSpaceSupported(addressSpace: SemanticAddressSpace | undefined): addressSpace is "local" | "shared" | "storage" | "constant" {
  return addressSpace === "local" || addressSpace === "shared" || addressSpace === "storage" || addressSpace === "constant";
}

function isNullPointerLiteral(expression: CudaLiteExpression): boolean {
  if (expression.kind === "number") return expression.value === 0;
  if (expression.kind === "identifier") return expression.name === "NULL" || expression.name === "nullptr";
  return expression.kind === "cast" && expression.pointer === true && isNullPointerLiteral(expression.expression);
}

function hasLaterLocalPointerAliasAssignment(
  name: string,
  statements: readonly CudaLiteStatement[],
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): boolean {
  for (const statement of statements) {
    if (statement.kind === "expr") {
      const expression = statement.expression;
      if (expression.kind !== "assignment" || expression.operator !== "=" || expression.left.kind !== "identifier" || expression.left.name !== name) continue;
      const alias = localPointerAliasForInitializer(expression.right, scope);
      return alias !== undefined && semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace);
    }
    if (statement.kind === "block" && hasLaterLocalPointerAliasAssignment(name, statement.body, scope)) return true;
    if (
      (statement.kind === "for" || statement.kind === "while" || statement.kind === "do-while") &&
      hasLaterLocalPointerAliasAssignment(name, statement.body, scope)
    ) {
      return true;
    }
  }
  return false;
}

function localPointerAliasUpdate(
  expression: CudaLiteExpression,
  scope: Map<string, CudaLiteSemanticSymbol>,
): boolean {
  if (expression.kind === "update" && expression.argument.kind === "identifier") {
    const target = scope.get(expression.argument.name);
    if (!target?.pointerRoot || !semanticPointerAliasAddressSpaceSupported(target.pointerAddressSpace) || target.pointerBaseIndices?.length !== 1) return false;
    const one = pointerAliasOffsetForBaseUnit(
      target,
      target.valueType,
      { kind: "literal", literalKind: "number", value: 1, valueType: "int", span: expression.span },
      expression.span,
    );
    const index = expression.operator === "++"
      ? addIndexExpressions(target.pointerBaseIndices[0]!, one, expression.span)
      : subtractIndexExpressions(target.pointerBaseIndices[0]!, one, expression.span);
    scope.set(target.name, { ...target, pointerBaseIndices: [index] });
    return true;
  }
  if (expression.kind !== "assignment" || expression.left.kind !== "identifier") return false;
  const target = scope.get(expression.left.name);
  if (!target || target.kind !== "local" || !target.pointer || target.dimensions.length > 0) return false;
  if (expression.operator === "=") {
    const alias = localPointerAliasForInitializer(expression.right, scope);
    if (!alias || !semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace)) return false;
    scope.set(target.name, { ...target, ...alias });
    return true;
  }
  if ((expression.operator === "+=" || expression.operator === "-=") && target.pointerRoot && semanticPointerAliasAddressSpaceSupported(target.pointerAddressSpace) && target.pointerBaseIndices?.length === 1) {
    const delta = pointerAliasOffsetForBaseUnit(target, target.valueType, lowerExpression(expression.right, scope), expression.span);
    const index = expression.operator === "+="
      ? addIndexExpressions(target.pointerBaseIndices[0]!, delta, expression.span)
      : subtractIndexExpressions(target.pointerBaseIndices[0]!, delta, expression.span);
    scope.set(target.name, { ...target, pointerBaseIndices: [index] });
    return true;
  }
  return false;
}

function localPointerArrayAliasUpdate(
  expression: CudaLiteExpression,
  scope: Map<string, CudaLiteSemanticSymbol>,
): boolean {
  if (
    expression.kind !== "assignment" ||
    expression.operator !== "=" ||
    expression.left.kind !== "index" ||
    expression.left.target.kind !== "identifier"
  ) {
    return false;
  }
  const target = scope.get(expression.left.target.name);
  if (!target || target.kind !== "local" || !target.pointer || target.dimensions.length !== 1) return false;
  const slot = staticPointerArrayIndex(expression.left.index);
  const extent = target.dimensions[0];
  if (slot === undefined || extent === undefined || slot >= extent) return false;
  const alias = localPointerAliasForInitializer(expression.right, scope);
  if (!alias?.pointerRoot || !semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace) || alias.pointerBaseIndices?.length !== 1) return false;
  const aliases = Array.from({ length: extent }, (_, index) => target.pointerArrayAliases?.[index]);
  aliases[slot] = alias;
  scope.set(target.name, { ...target, pointerArrayAliases: aliases });
  return true;
}

function staticPointerArrayIndex(expression: CudaLiteExpression): number | undefined {
  if (expression.kind !== "number" || !Number.isInteger(expression.value) || expression.value < 0) return undefined;
  return expression.value;
}

function semanticPointerArrayAliasesComplete(symbol: CudaLiteSemanticSymbol | undefined): boolean {
  const extent = symbol?.dimensions.length === 1 ? symbol.dimensions[0] : undefined;
  return extent !== undefined && extent > 0 && symbol?.pointerArrayAliases?.length === extent &&
    symbol.pointerArrayAliases.every((alias) =>
      alias?.pointerRoot !== undefined &&
      semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace) &&
      alias.pointerBaseIndices?.length === 1
    );
}

function mergeBlockLocalPointerAliases(
  parent: Map<string, CudaLiteSemanticSymbol>,
  child: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): void {
  for (const [name, current] of parent) {
    if (!current.pointer || current.dimensions.length > 0) continue;
    const next = child.get(name);
    if (
      !next ||
      !sameSymbolDeclaration(current, next) ||
      !next.pointerRoot ||
      !semanticPointerAliasAddressSpaceSupported(next.pointerAddressSpace) ||
      next.pointerBaseIndices?.length !== 1
    ) {
      continue;
    }
    parent.set(name, {
      ...current,
      pointerRoot: next.pointerRoot,
      pointerAddressSpace: next.pointerAddressSpace,
      pointerBaseIndices: next.pointerBaseIndices,
      ...(next.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
    });
  }
}

function mergeBranchLocalPointerAliases(
  parent: Map<string, CudaLiteSemanticSymbol>,
  consequent: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  alternate: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  condition: SemanticExpression,
  span: SourceSpan,
): void {
  for (const [name, current] of parent) {
    if (!current.pointer || current.dimensions.length > 0) continue;
    const left = consequent.get(name);
    const right = alternate.get(name);
    if (
      !left?.pointerRoot ||
      !right?.pointerRoot ||
      !sameSymbolDeclaration(current, left) ||
      !sameSymbolDeclaration(current, right) ||
      left.pointerRoot !== right.pointerRoot ||
      left.pointerAddressSpace !== right.pointerAddressSpace ||
      left.pointerBaseIsScalarLane !== right.pointerBaseIsScalarLane ||
      !semanticPointerAliasAddressSpaceSupported(left.pointerAddressSpace) ||
      left.pointerBaseIndices?.length !== 1 ||
      right.pointerBaseIndices?.length !== 1
    ) {
      continue;
    }
    parent.set(name, {
      ...current,
      pointerRoot: left.pointerRoot,
      pointerAddressSpace: left.pointerAddressSpace,
      pointerBaseIndices: [{
        kind: "conditional",
        condition,
        consequent: left.pointerBaseIndices[0]!,
        alternate: right.pointerBaseIndices[0]!,
        valueType: "int",
        span,
      }],
      ...(left.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
    });
  }
}

function sameSymbolDeclaration(left: CudaLiteSemanticSymbol, right: CudaLiteSemanticSymbol): boolean {
  return left.name === right.name &&
    left.kind === right.kind &&
    left.span.start === right.span.start &&
    left.span.end === right.span.end;
}

function localPointerAliasRoot(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): { readonly root: CudaLiteSemanticSymbol; readonly indices: readonly SemanticExpression[] } | undefined {
  const ref = localArrayRefFromExpression(expression, scope, true);
  if (ref) {
    if (ref.root.dimensions.length === 0) return undefined;
    return { root: ref.root, indices: [flatIndexExpressionForDimensions(ref.root.dimensions, ref.indices, expression.span)] };
  }
  if (expression.kind !== "index" || expression.target.kind !== "identifier") return undefined;
  const target = scope.get(expression.target.name);
  if (target?.kind === "param" && target.pointer && target.addressSpace === "storage") {
    return {
      root: target,
      indices: [pointerAliasElementOffset(target.valueType, target.valueType, lowerExpression(expression.index, scope), expression.index.span)],
    };
  }
  if (target?.pointerRoot && semanticPointerAliasAddressSpaceSupported(target.pointerAddressSpace) && target.pointerBaseIndices?.length === 1) {
    const root = scope.get(target.pointerRoot);
    if (!root || !semanticPointerAliasAddressSpaceSupported(root.addressSpace)) return undefined;
    return {
      root,
      indices: [addIndexExpressions(
        target.pointerBaseIndices[0]!,
        pointerAliasElementOffset(target.valueType, root.valueType, lowerExpression(expression.index, scope), expression.index.span),
        expression.span,
      )],
    };
  }
  return undefined;
}

function localArrayRefFromExpression(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  allowShared = false,
): { readonly root: CudaLiteSemanticSymbol; readonly indices: readonly SemanticExpression[] } | undefined {
  if (expression.kind !== "index") return undefined;
  if (expression.target.kind === "identifier") {
    const root = scope.get(expression.target.name);
    if (!root || (root.kind !== "local" && (!allowShared || root.kind !== "shared" && root.kind !== "constant")) || root.dimensions.length === 0 || root.pointer) return undefined;
    return { root, indices: [lowerExpression(expression.index, scope)] };
  }
  const target = localArrayRefFromExpression(expression.target, scope, allowShared);
  if (!target) return undefined;
  return { root: target.root, indices: [...target.indices, lowerExpression(expression.index, scope)] };
}

function localPointerAliasIndexExpression(
  expression: Extract<CudaLiteExpression, { readonly kind: "index" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticExpression | undefined {
  const alias = localPointerAliasForInitializer(expression.target, scope);
  if (!alias?.pointerRoot || !semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace) || !alias.pointerBaseIndices || alias.pointerBaseIndices.length !== 1) return undefined;
  const root = scope.get(alias.pointerRoot);
  if (!root) return undefined;
  const aliasValueType = pointerAliasTargetValueType(expression.target, scope) ?? root.valueType;
  const target = semanticSymbolExpression(root, expression.target.span);
  const index = addIndexExpressions(
    alias.pointerBaseIndices[0]!,
    pointerAliasElementOffset(aliasValueType, root.valueType, lowerExpression(expression.index, scope), expression.index.span, alias.pointerBaseUnitBytes),
    expression.index.span,
  );
  return {
    kind: "index",
    target,
    index,
    ...optionalValueType(aliasValueType),
    addressSpace: root.addressSpace,
    ...(alias.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
    ...(alias.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: alias.pointerBaseUnitBytes }),
    ...optionalPackedByteLanes(pointerAliasPackedByteLanes(expression.target, root, scope)),
    span: expression.span,
  };
}

function pointerAliasTargetValueType(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): CudaLiteScalarType | undefined {
  if (expression.kind === "cast" && expression.pointer) return expression.valueType;
  if (expression.kind === "identifier") return scope.get(expression.name)?.valueType;
  if (expression.kind === "index") return pointerAliasTargetValueType(expression.target, scope);
  if (expression.kind === "unary" && (expression.operator === "&" || expression.operator === "*")) {
    return pointerAliasTargetValueType(expression.argument, scope);
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    return pointerAliasTargetValueType(expression.left, scope) ?? pointerAliasTargetValueType(expression.right, scope);
  }
  if (expression.kind === "call" && localPointerIdentityCallName(expression.callee)) {
    return expression.args[0] === undefined ? undefined : pointerAliasTargetValueType(expression.args[0], scope);
  }
  if (expression.kind === "conditional") {
    const consequent = pointerAliasTargetValueType(expression.consequent, scope);
    const alternate = pointerAliasTargetValueType(expression.alternate, scope);
    return consequent === alternate ? consequent : undefined;
  }
  return undefined;
}

function pointerAliasElementOffset(
  aliasType: CudaLiteScalarType | undefined,
  rootType: CudaLiteScalarType | undefined,
  index: SemanticExpression,
  span: SourceSpan,
  baseUnitBytes?: number,
): SemanticExpression {
  if (baseUnitBytes !== undefined) return multiplyIndexExpression(index, baseUnitBytes, span);
  return aliasType !== undefined &&
      rootType !== undefined &&
      isCudaVectorType(aliasType) &&
      !isCudaVectorType(rootType)
    ? multiplyIndexExpression(index, cudaVectorLaneCount(aliasType), span)
    : index;
}

function localPointerAliasDerefExpression(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticExpression | undefined {
  const directPackedLocal = expression.kind === "cast" && expression.pointer &&
    expression.expression.kind === "unary" && expression.expression.operator === "&" &&
    expression.expression.argument.kind === "identifier"
    ? scope.get(expression.expression.argument.name)
    : undefined;
  const alias = directPackedLocal?.kind === "local" && directPackedLocal.dimensions.length === 0 && directPackedLocal.packedByteLanes !== undefined
    ? {
        pointerRoot: directPackedLocal.name,
        pointerAddressSpace: directPackedLocal.addressSpace,
        pointerBaseIndices: [zeroExpression(expression.span)],
      }
    : localPointerAliasForInitializer(expression, scope);
  if (!alias?.pointerRoot || !semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace) || !alias.pointerBaseIndices || alias.pointerBaseIndices.length !== 1) return undefined;
  const root = scope.get(alias.pointerRoot);
  if (!root) return undefined;
  return {
    kind: "index",
    target: semanticSymbolExpression(root, expression.span),
    index: alias.pointerBaseIndices[0]!,
    ...optionalValueType(pointerAliasTargetValueType(expression, scope) ?? root.valueType),
    addressSpace: root.addressSpace,
    ...(alias.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
    ...optionalPackedByteLanes(pointerAliasPackedByteLanes(expression, root, scope)),
    span,
  };
}

function pointerAliasPackedByteLanes(
  expression: CudaLiteExpression,
  root: CudaLiteSemanticSymbol,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): 2 | 3 | 4 | undefined {
  if (expression.kind === "cast" && expression.pointer && expression.packedByteLanes !== undefined) return expression.packedByteLanes;
  const aliasType = pointerAliasTargetValueType(expression, scope);
  if (
    (root.valueType === "uchar" || root.packedByteLanes !== undefined) &&
    aliasType !== undefined &&
    !isCudaVectorType(aliasType) &&
    sizeofCudaType(aliasType) === 4
  ) return 4;
  if (
    expression.kind === "cast" && expression.pointer &&
    (root.valueType === "uchar" || root.packedByteLanes !== undefined) &&
    !isCudaVectorType(expression.valueType) && sizeofCudaType(expression.valueType) === 4
  ) return 4;
  if (expression.kind === "unary" && (expression.operator === "&" || expression.operator === "*")) return pointerAliasPackedByteLanes(expression.argument, root, scope);
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    return pointerAliasPackedByteLanes(expression.left, root, scope) ?? pointerAliasPackedByteLanes(expression.right, root, scope);
  }
  return undefined;
}

function localPointerAliasDifferenceExpression(
  expression: Extract<CudaLiteExpression, { readonly kind: "binary" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticExpression | undefined {
  if (expression.operator !== "-") return undefined;
  const left = localPointerAliasScalarIndex(expression.left, scope);
  const right = localPointerAliasScalarIndex(expression.right, scope);
  if (!left || !right || left.root !== right.root) return undefined;
  return subtractIndexExpressions(left.index, right.index, expression.span);
}

function localPointerAliasComparisonExpression(
  expression: Extract<CudaLiteExpression, { readonly kind: "binary" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticExpression | undefined {
  if (!POINTER_ORDER_OPERATORS.has(expression.operator)) return undefined;
  if (expression.operator === "==" || expression.operator === "!=") {
    const leftAlias = localPointerAliasScalarIndex(expression.left, scope);
    const rightAlias = localPointerAliasScalarIndex(expression.right, scope);
    if (leftAlias && isNullPointerLiteral(expression.right)) return pointerNullComparisonExpression(leftAlias.valid, expression.operator, expression.span);
    if (rightAlias && isNullPointerLiteral(expression.left)) return pointerNullComparisonExpression(rightAlias.valid, expression.operator, expression.span);
  }
  const left = localPointerAliasScalarIndex(expression.left, scope);
  const right = localPointerAliasScalarIndex(expression.right, scope);
  if (!left || !right || left.root !== right.root) return undefined;
  const indexComparisonOperator = left.valid || right.valid ? "==" : expression.operator;
  const indexComparison: SemanticExpression = {
    kind: "binary",
    operator: indexComparisonOperator,
    left: left.index,
    right: right.index,
    valueType: "bool",
    span: expression.span,
  };
  if (!left.valid && !right.valid) return indexComparison;
  if (expression.operator !== "==" && expression.operator !== "!=") return undefined;
  return nullablePointerAliasEqualityExpression(left, right, indexComparison, expression.operator, expression.span);
}

function localPointerAliasScalarIndex(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): { readonly root: string; readonly index: SemanticExpression; readonly valid?: SemanticExpression } | undefined {
  if (expression.kind !== "identifier") return undefined;
  const symbol = scope.get(expression.name);
  if (!symbol?.pointerRoot || !semanticPointerAliasAddressSpaceSupported(symbol.pointerAddressSpace) || !symbol.pointerBaseIndices || symbol.pointerBaseIndices.length !== 1) return undefined;
  return { root: symbol.pointerRoot, index: symbol.pointerBaseIndices[0]!, ...(symbol.pointerValid === undefined ? {} : { valid: symbol.pointerValid }) };
}

function semanticSymbolExpression(symbol: CudaLiteSemanticSymbol, span: SourceSpan): Extract<SemanticExpression, { readonly kind: "symbol" }> {
  return {
    kind: "symbol",
    name: symbol.name,
    ...(symbol.valueType === undefined ? {} : { valueType: symbol.valueType }),
    addressSpace: symbol.addressSpace,
    span,
  };
}

function addIndexExpressions(left: SemanticExpression, right: SemanticExpression, span: SourceSpan): SemanticExpression {
  if (isZeroLiteral(right)) return left;
  if (isZeroLiteral(left)) return right;
  return {
    kind: "binary",
    operator: "+",
    left,
    right,
    ...optionalValueType(expressionValueType(left) ?? expressionValueType(right)),
    span,
  };
}

function subtractIndexExpressions(left: SemanticExpression, right: SemanticExpression, span: SourceSpan): SemanticExpression {
  if (isZeroLiteral(right)) return left;
  return {
    kind: "binary",
    operator: "-",
    left,
    right,
    ...optionalValueType(expressionValueType(left) ?? expressionValueType(right)),
    span,
  };
}

function flatIndexExpressionForDimensions(
  dimensions: readonly number[],
  indices: readonly SemanticExpression[],
  span: SourceSpan,
): SemanticExpression {
  let flat = zeroExpression(span);
  for (const [offset, index] of indices.entries()) {
    const stride = dimensionStride(dimensions, offset);
    const term = stride === 1 ? index : multiplyIndexExpression(index, stride, span);
    flat = addIndexExpressions(flat, term, span);
  }
  return flat;
}

function multiplyIndexExpression(left: SemanticExpression, right: number, span: SourceSpan): SemanticExpression {
  if (right === 1) return left;
  return {
    kind: "binary",
    operator: "*",
    left,
    right: { kind: "literal", literalKind: "number", value: right, valueType: "int", span },
    ...optionalValueType(expressionValueType(left)),
    span,
  };
}

function isZeroLiteral(expression: SemanticExpression): boolean {
  return expression.kind === "literal" && expression.literalKind === "number" && expression.value === 0;
}

function zeroExpression(span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value: 0, valueType: "int", span };
}

function booleanExpression(value: boolean, span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value: value ? 1 : 0, valueType: "bool", span };
}

function pointerNullComparisonExpression(valid: SemanticExpression | undefined, operator: string, span: SourceSpan): SemanticExpression {
  if (!valid) return booleanExpression(operator === "!=", span);
  return operator === "!=" ? valid : negateExpression(valid, span);
}

function nullablePointerAliasEqualityExpression(
  left: { readonly valid?: SemanticExpression },
  right: { readonly valid?: SemanticExpression },
  indexEqual: SemanticExpression,
  operator: string,
  span: SourceSpan,
): SemanticExpression {
  const leftValid = left.valid ?? booleanExpression(true, span);
  const rightValid = right.valid ?? booleanExpression(true, span);
  const bothValidAndSame = andExpression(andExpression(leftValid, rightValid, span), indexEqual, span);
  const bothInvalid = andExpression(
    left.valid ? negateExpression(left.valid, span) : booleanExpression(false, span),
    right.valid ? negateExpression(right.valid, span) : booleanExpression(false, span),
    span,
  );
  const equal = orExpression(bothInvalid, bothValidAndSame, span);
  return operator === "==" ? equal : negateExpression(equal, span);
}

function andExpression(left: SemanticExpression, right: SemanticExpression, span: SourceSpan): SemanticExpression {
  return logicalExpression("&&", left, right, span);
}

function orExpression(left: SemanticExpression, right: SemanticExpression, span: SourceSpan): SemanticExpression {
  return logicalExpression("||", left, right, span);
}

function logicalExpression(operator: "&&" | "||", left: SemanticExpression, right: SemanticExpression, span: SourceSpan): SemanticExpression {
  return { kind: "binary", operator, left, right, valueType: "bool", span };
}

function negateExpression(expression: SemanticExpression, span: SourceSpan): SemanticExpression {
  return {
    kind: "binary",
    operator: "==",
    left: expression,
    right: zeroExpression(span),
    valueType: "bool",
    span,
  };
}

function paramAddressSpace(param: CudaLiteParam): SemanticAddressSpace {
  if (param.valueType === "texture2d") return "texture";
  if (param.valueType === "surface2d") return "surface";
  if (param.valueType === "devicepool") return "pool";
  if (param.pointer) return "storage";
  return "uniform";
}

function memberValueType(object: SemanticExpression, property: string): CudaLiteScalarType | undefined {
  if (object.kind === "symbol") {
    const builtinType = cudaBuiltinVectorMemberValueType(object.name, property);
    if (builtinType) return builtinType;
  }
  const objectType = expressionValueType(object);
  const storageFields = semanticStorageVectorFieldIndices(objectType, property);
  if (objectType === "complex64" && storageFields !== undefined) {
    if (storageFields.length === 1) return "float";
    return "float2";
  }
  const swizzleType = cudaVectorSwizzleType(objectType, property);
  if (swizzleType !== undefined) {
    return swizzleType;
  }
  return expressionValueType(object);
}

function expressionAddressSpace(expression: SemanticExpression): SemanticAddressSpace {
  if (expression.kind === "symbol") return expression.addressSpace;
  if (expression.kind === "index") return expression.addressSpace;
  if (expression.kind === "member") return expressionAddressSpace(expression.object);
  return "unknown";
}

function semanticBinaryResultValueType(
  operator: string,
  left: SemanticExpression,
  right: SemanticExpression,
): CudaLiteScalarType | undefined {
  if (COMPARISON_OPERATORS.has(operator)) return "bool";
  const leftType = expressionValueType(left);
  const rightType = expressionValueType(right);
  if (isCudaVectorType(leftType)) return leftType;
  if (isCudaVectorType(rightType)) return rightType;
  return leftType ?? rightType;
}

function expressionValueType(expression: SemanticExpression | undefined): CudaLiteScalarType | undefined {
  if (!expression || expression.kind === "initializer") return undefined;
  return "valueType" in expression ? expression.valueType : undefined;
}

function semanticIntrinsicReturnType(name: string | undefined, args: readonly SemanticExpression[]): CudaLiteScalarType | undefined {
  if (name === undefined) return undefined;
  const vectorConstructorType = cudaVectorConstructorType(name);
  if (vectorConstructorType) return vectorConstructorType;
  const vectorMathReturnType = semanticVectorMathReturnType(name, args);
  if (vectorMathReturnType) return vectorMathReturnType;
  const bfloat16ReturnType = cudaBfloat16IntrinsicReturnType(name, args.some((arg) => expressionValueType(arg) === "bf16"));
  if (bfloat16ReturnType) return bfloat16ReturnType;
  if (isBfloat162VectorName(name)) return "bf162";
  if (name === "__hisnan2" ||
    name === "__heq2" || name === "__hne2" || name === "__hgt2" || name === "__hge2" || name === "__hlt2" || name === "__hle2" ||
    name === "__hequ2" || name === "__hneu2" || name === "__hgtu2" || name === "__hgeu2" || name === "__hltu2" || name === "__hleu2") return "half2";
  if (name === "__heq2_mask" || name === "__hne2_mask" || name === "__hgt2_mask" || name === "__hge2_mask" || name === "__hlt2_mask" || name === "__hle2_mask" ||
    name === "__hequ2_mask" || name === "__hneu2_mask" || name === "__hgtu2_mask" || name === "__hgeu2_mask" || name === "__hltu2_mask" || name === "__hleu2_mask") return "uint";
  if (name === "__hbeq2" || name === "__hbne2" || name === "__hbgt2" || name === "__hbge2" || name === "__hblt2" || name === "__hble2" ||
    name === "__hbequ2" || name === "__hbneu2" || name === "__hbgtu2" || name === "__hbgeu2" || name === "__hbltu2" || name === "__hbleu2") return "bool";
  if (name === "__low2half" || name === "__high2half") return "half";
  if (name === "__halves2half2" || name === "__half2half2" || name === "__low2half2" || name === "__high2half2" || name === "__lows2half2" || name === "__highs2half2" || name === "__lowhigh2highlow") return "half2";
  void args;
  return undefined;
}

function isBfloat162VectorName(name: string): boolean {
  return name === "h2ceil" ||
    name === "h2floor" ||
    name === "h2rcp" ||
    name === "h2rsqrt" ||
    name === "h2sqrt" ||
    name === "h2trunc" ||
    name === "h2exp" ||
    name === "h2exp2" ||
    name === "h2exp10" ||
    name === "h2log" ||
    name === "h2log2" ||
    name === "h2log10" ||
    name === "h2sin" ||
    name === "h2cos" ||
    name === "h2tanh" ||
    name === "h2tanh_approx" ||
    name === "h2rint";
}

function indexedValueType(target: SemanticExpression): CudaLiteScalarType | undefined {
  const targetType = expressionValueType(target);
  return expressionAddressSpace(target) === "local" && targetType !== undefined && isCudaVectorType(targetType)
    ? cudaVectorScalarType(targetType)
    : targetType;
}

function indexedExpressionValueType(
  expression: Extract<CudaLiteExpression, { readonly kind: "index" }>,
  target: SemanticExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): CudaLiteScalarType | undefined {
  const vectorArray = vectorArrayIndexInfo(expression, scope);
  if (vectorArray) {
    if (vectorArray.indexDepth === vectorArray.dimensions.length) return vectorArray.valueType;
    if (vectorArray.indexDepth > vectorArray.dimensions.length) return cudaVectorScalarType(vectorArray.valueType);
  }
  return indexedValueType(target);
}

function vectorArrayIndexInfo(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): { readonly valueType: CudaLiteScalarType; readonly dimensions: readonly number[]; readonly indexDepth: number } | undefined {
  let current = expression;
  let indexDepth = 0;
  while (current.kind === "index") {
    indexDepth += 1;
    current = current.target;
  }
  if (current.kind !== "identifier") return undefined;
  const symbol = scope.get(current.name);
  if (!symbol || symbol.dimensions.length === 0 || !isCudaVectorType(symbol.valueType)) return undefined;
  return { valueType: symbol.valueType, dimensions: symbol.dimensions, indexDepth };
}

function optionalValueType(valueType: CudaLiteScalarType | undefined): { readonly valueType?: CudaLiteScalarType } {
  return valueType === undefined ? {} : { valueType };
}

function numberLiteralType(raw: string): CudaLiteScalarType {
  if (/^0x/iu.test(raw)) {
    if (/(?:[uU][lL]*|[lL]+[uU][lL]*)$/u.test(raw)) return "uint";
    const digits = raw.replace(/^0x/iu, "").replace(/[lL]+$/u, "");
    try {
      return BigInt(`0x${digits}`) > 0x7fffffffn ? "uint" : "int";
    } catch {
      return "int";
    }
  }
  return /[.eE]|[fF]$/u.test(raw) ? "float" : /(?:[uU][lL]*|[lL]+[uU][lL]*)$/u.test(raw) ? "uint" : "int";
}

function normalizeWorkgroupSize(value: readonly [number, number, number]): [number, number, number] {
  return [normalizeDimension(value[0]), normalizeDimension(value[1]), normalizeDimension(value[2])];
}

function normalizeDimension(value: number): number {
  return Number.isFinite(value) && Number.isInteger(value) && value > 0 ? value : 1;
}
