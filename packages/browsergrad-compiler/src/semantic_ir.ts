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
import { requireSemanticValueType, type SemanticValueType } from "./semantic_value_type.js";
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
} from "./semantic_texture_surface.js";
import {
  cudaVibMinMaxInfo,
  isCudaFrexpCallName as isFrexpCallName,
  isCudaModfCallName as isModfCallName,
  isCudaNanPayloadCallName as isNanPayloadCallName,
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
  type InlineAsmIntSource,
  type InlineAsmOp,
  type PtxSpecialU32Register,
} from "./features/inline_ptx/model.js";
import { alignofCudaType, sizeofCudaType } from "./type_layout.js";
import { cudaVectorConstructorType, cudaVectorLaneCount, cudaVectorScalarType, cudaVectorSwizzleType, isCudaVectorType } from "./vector_types.js";
import { SEMANTIC_LOCAL_ARRAY_FILL_CALLS, SEMANTIC_NOOP_CALLS, SEMANTIC_SUBGROUP_CALLS } from "./semantic_builtin_calls.js";
import { isHostManagedRuntimeNoopCall } from "./cuda_runtime_noops.js";
import { isCooperativeReductionObjectName } from "./cooperative_reduction.js";
import { SEMANTIC_CURAND_CALLS, SEMANTIC_CURAND_VECTOR_RETURN_TYPES } from "./semantic_curand_intrinsics.js";
import {
  isSemanticGeneratedRandomCall,
  semanticGeneratedRandomReturnType,
} from "./semantic_generated_random_intrinsics.js";
import { semanticPointerArgumentMemoryRef as semanticIrPointerArgumentMemoryRef } from "./semantic_pointer_arguments.js";
import { resolveSemanticFunctionOverloads } from "./semantic_function_overloads.js";
import { semanticVectorMathReturnType } from "./semantic_vector_math.js";
import { semanticStorageVectorFieldIndices } from "./semantic_value_types.js";
import {
  isSemanticBf162OverloadedVectorCall,
  semanticBf162VectorReturnType,
  semanticHalf2VectorReturnType,
} from "./semantic_vector_intrinsics.js";
import { isSemanticMathCallName, semanticMathCallReturnType } from "./semantic_math_intrinsics.js";
import { semanticAtomicOperation } from "./semantic_atomic_intrinsics.js";
import { isCudaBuiltinVectorSymbolName } from "./cuda_builtin_symbols.js";
import { collectExternalDevicePoolNames } from "./ast_queries.js";
import {
  completeCanonicalLowering,
  completeSemanticTyping,
} from "./compiler_phases.js";
import type { AnalyzedCudaLiteModule } from "./analyzer.js";
import {
  createBuiltinSemanticSymbolId,
  createGeneratedSemanticSymbolId,
  createUnresolvedSemanticFunctionId,
  createUnresolvedSemanticSymbolId,
  createSemanticFunctionId,
  createSemanticSymbolId,
  semanticIdKey,
  semanticIdsEqual,
  semanticFunctionIdFromSymbol,
  semanticMemoryIdFromSymbol,
  semanticSymbolIdFromFunction,
  semanticSymbolIdFromMemory,
  type SemanticFunctionId,
  type SemanticMemoryId,
  type SemanticSymbolId,
} from "./semantic_ids.js";
import { createSemanticEnvironment, type SemanticEnvironment } from "./semantic_environment.js";
import { semanticBinaryResultType } from "./semantic_type_rules.js";
import {
  binaryFloatCallExpression,
  binaryIntCallExpression,
  castScalarExpression,
  frexpExponentForFiniteNumber,
  intNumberExpression,
  mathCallExpression,
  multiplyFloatExpressions,
  numberExpression,
  roundTiesToEvenNumber,
  semanticCallExpression,
  semanticExpressionSideEffectFree,
  staticNumberValue,
  uintNumberExpression,
  unaryFloatCallExpression,
  unaryIntCallExpression,
} from "./semantic_expression_builders.js";
import {
  matrixTileElementCount,
  normalizeMatrixTileLayout,
  resolveMatrixTileSpec,
  wmmaBuiltinName,
  type MatrixTileLayout,
} from "./matrix_tiles.js";
import type {
  SemanticAddressSpace,
  CudaLiteSemanticSymbol,
  SemanticPointerAlias,
  SemanticPointerSelection,
  CudaLiteSemanticFunction,
  CudaLiteSemanticLaunchableEntry,
  TypedCudaLiteSemanticModel,
  SemanticMemoryRef,
  SemanticPoolRef,
  SemanticMatrixTileRef,
  SemanticExpression,
  SemanticKernelIrOperation,
  SemanticDeviceLaunch,
  CanonicalSemanticKernelIr,
} from "./semantic_ir_types.js";
import { isSemanticKernelIrOperation, walkSemanticExpression, walkSemanticOperations } from "./semantic_ir_walk.js";

export type {
  SemanticAddressSpace,
  CudaLiteSemanticSymbol,
  SemanticPointerAlias,
  SemanticPointerSelection,
  CudaLiteSemanticFunction,
  SemanticCooperativeGroupDeclaration,
  CudaLiteSemanticLaunchableEntry,
  CudaLiteSemanticModel,
  TypedCudaLiteSemanticModel,
  SemanticMemoryRef,
  SemanticPoolRef,
  SemanticMatrixTileRef,
  SemanticExpression,
  SemanticKernelIrOperation,
  SemanticDeviceLaunch,
  SemanticKernelIrModule,
  CanonicalSemanticKernelIr,
} from "./semantic_ir_types.js";
export {
  collectSemanticPoolAllocations,
  isSemanticKernelIrOperation,
  walkSemanticExpression,
  walkSemanticMemoryRef,
  walkSemanticOperation,
  walkSemanticOperations,
} from "./semantic_ir_walk.js";

export function semanticPointerSymbolNeedsRuntimeState(symbol: CudaLiteSemanticSymbol): boolean {
  return Boolean(symbol.pointer) && (
    symbol.pointerRuntimeState === true ||
    symbol.pointerRoot === undefined &&
      symbol.pointerSelection === undefined &&
      symbol.pointerArrayAliases === undefined
  );
}

export function semanticInlineAsmLdmatrixAssignments(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "inline-asm" }>,
): readonly Extract<SemanticExpression, { readonly kind: "assignment" }>[] | undefined {
  const op = operation.op;
  if (op?.kind !== "ldmatrix" || operation.inputs.length !== 1 || operation.outputs.length !== op.matrices) return undefined;
  const base = operation.inputs[0]!;
  return operation.outputs.map((target, index) => {
    const tag = op.transposed ? 0x80000000 : 0;
    const value = binaryIndexExpression(
      "+",
      binaryIndexExpression("+", uintNumberExpression(tag, operation.span), base, operation.span),
      uintNumberExpression(index * 2, operation.span),
      operation.span,
    );
    return {
      kind: "assignment",
      operator: "=",
      target,
      value,
      valueType: "uint",
      span: operation.span,
    };
  });
}

const DEFAULT_WORKGROUP_SIZE: KernelLaunch["blockDim"] = [256, 1, 1];
const POINTER_ORDER_OPERATORS = new Set(["<", "<=", ">", ">=", "==", "!="]);
const BARRIER_CALLS: ReadonlySet<string> = new Set([...CUDA_BARRIER_CALL_NAMES, ...CUDA_COOPERATIVE_BARRIER_CALL_NAMES, "grid.sync"]);
const FENCE_CALLS: ReadonlySet<string> = new Set(CUDA_FENCE_CALL_NAMES);

/**
 * Keeps mutable locals separate from the immutable module environment.
 *
 * A lowering scope intentionally supports assignment and shadowing, but it must
 * not materialize a second name table for parameters, globals, and functions.
 */
class SemanticLexicalScope extends Map<string, CudaLiteSemanticSymbol> {
  constructor(
    private readonly environment: SemanticEnvironment,
    locals?: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  ) {
    super(locals);
  }

  override get(name: string): CudaLiteSemanticSymbol | undefined {
    return super.get(name) ?? this.environment.resolveSymbol(name);
  }

  override has(name: string): boolean {
    return super.has(name) || this.environment.resolveSymbol(name) !== undefined;
  }

  clone(): SemanticLexicalScope {
    return new SemanticLexicalScope(this.environment, this);
  }

  resolveMemorySymbol(id: SemanticMemoryId): CudaLiteSemanticSymbol | undefined {
    for (const symbol of this.values()) {
      if (semanticIdsEqual(semanticMemoryIdFromSymbol(symbol.id), id)) return symbol;
    }
    return this.environment.resolveMemorySymbol(id);
  }
}

function createSemanticLexicalScope(environment: SemanticEnvironment): SemanticLexicalScope {
  return new SemanticLexicalScope(environment);
}

function cloneSemanticScope(
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): Map<string, CudaLiteSemanticSymbol> {
  return scope instanceof SemanticLexicalScope ? scope.clone() : new Map(scope);
}

export function createCudaLiteSemanticModel(analysis: AnalyzedCudaLiteModule): TypedCudaLiteSemanticModel {
  const params = analysis.kernel.params.map(symbolForParam);
  const constants = analysis.constants.map(symbolForConstant);
  const deviceGlobals = analysis.deviceGlobals.map(symbolForDeviceGlobal);
  const textures = analysis.textures.map(symbolForTexture);
  const helperFunctions = analysis.functions.filter((fn) => fn.span.start !== analysis.kernel.span.start);
  const externalPoolNames = new Set([
    ...collectExternalDevicePoolNames(analysis.kernel.body),
    ...helperFunctions.flatMap((fn) => collectExternalDevicePoolNames(fn.body)),
  ]);
  const externalPools = [...externalPoolNames].map((name) => symbolForExternalPool(name, analysis.kernel.span));
  const functionSymbols = helperFunctions.map(symbolForFunctionDeclaration);
  const launchableEntries: CudaLiteSemanticLaunchableEntry[] = [
    ...analysis.kernels.map((kernel) => ({
      id: createSemanticFunctionId(kernel.name, kernel.span),
      kind: "kernel" as const,
      name: kernel.name,
      params: kernel.params.map(symbolForParam),
      span: kernel.span,
    })),
    ...analysis.functions.map((fn) => ({
      id: createSemanticFunctionId(fn.name, fn.span),
      kind: "device-function" as const,
      name: fn.name,
      params: fn.params.map(symbolForParam),
      span: fn.span,
    })),
  ];
  const symbols = [...params, ...constants, ...deviceGlobals, ...textures, ...externalPools];
  const kernelSymbols = launchableEntries
    .filter((entry) => entry.kind === "kernel")
    .map(symbolForLaunchableEntry);
  const declarationEnvironment = createSemanticEnvironment(
    [...symbols, ...functionSymbols, ...kernelSymbols],
    [],
  );
  const functionSignatures = helperFunctions.map((fn) => semanticFunctionSignature(fn, declarationEnvironment));
  const loweringEnvironment = createSemanticEnvironment(
    [...symbols, ...functionSymbols, ...kernelSymbols],
    functionSignatures,
  );
  const functions = helperFunctions.map((fn, index) =>
    symbolForFunction(fn, loweringEnvironment, functionSignatures[index]!.params),
  );
  const environment = createSemanticEnvironment(
    [...symbols, ...functionSymbols, ...kernelSymbols],
    functions,
  );
  return completeSemanticTyping({
    kind: "cuda-lite-semantic-model",
    kernelName: analysis.kernel.name,
    span: analysis.kernel.span,
    params,
    symbols,
    functions,
    launchableEntries,
    requiredFeatures: analysis.requiredFeatures,
    environment,
  }, analysis);
}

export function lowerSemanticModelToKernelIr(
  analysis: AnalyzedCudaLiteModule,
  semantic: TypedCudaLiteSemanticModel,
  options: {
    readonly workgroupSize?: readonly [number, number, number];
    readonly dynamicSharedMemory?: Readonly<Record<string, number>>;
    readonly subgroupMode?: "native" | "scalar";
    readonly bindlessTextures?: readonly string[];
  } = {},
): CanonicalSemanticKernelIr {
  // These declarations remain part of the emitted module metadata. Resolution
  // itself comes from semantic.environment through the lexical scope below.
  const functionSymbols = semantic.functions.map(symbolForSemanticFunctionDeclaration);
  const scope = createSemanticLexicalScope(semantic.environment);
  const mutableParams = mutableKernelParamShadows(analysis, semantic.params);
  for (const shadow of mutableParams) scope.set(shadow.sourceName, shadow.symbol);
  const sourceBarrierFunctions = semanticIrBarrierFunctionNames(semantic.functions);
  const guardedBarrierFunctionNames = new Map(
    [...sourceBarrierFunctions].map((name) => [name, semanticGuardedBarrierFunctionName(name)] as const),
  );
  const loweredOriginalFunctions = semantic.functions.map((fn): CudaLiteSemanticFunction => {
    const proof = analysis.barrierUniformity.functions[fn.name];
    if (proof === undefined || !sourceBarrierFunctions.has(fn.name)) return fn;
    const promoted = promoteSemanticBarrierResultCalls(fn.body, sourceBarrierFunctions);
    const loweredBranches = lowerSemanticDivergentBarrierBranches(
      promoted,
      semantic.functions,
      fn.span,
      proof,
      guardedBarrierFunctionNames,
    );
    const loweredBreaks = lowerSemanticDivergentBreaksBeforeBarriers(
      loweredBranches,
      fn.span,
      proof,
      sourceBarrierFunctions,
    );
    const loweredContinues = lowerSemanticDivergentContinuesBeforeBarriers(
      loweredBreaks,
      fn.span,
      proof,
      sourceBarrierFunctions,
    );
    const body = lowerSemanticEarlyReturnsBeforeCollectives(loweredContinues, semantic.functions, fn.span, proof);
    return {
      ...fn,
      body: markSemanticWorkgroupUniformControl(body, proof.workgroupUniformControlStatementStarts),
    };
  });
  const guardedBarrierFunctions = createSemanticGuardedBarrierFunctions(
    semantic.functions,
    sourceBarrierFunctions,
    guardedBarrierFunctionNames,
  );
  const loweredSourceFunctions = [...loweredOriginalFunctions, ...guardedBarrierFunctions];
  const rawOperations = [
    ...mutableParams.map((shadow): SemanticKernelIrOperation => ({
      kind: "declare",
      target: shadow.symbol,
      init: semanticSymbolExpression(shadow.param, shadow.param.span),
      span: shadow.param.span,
    })),
    ...lowerStatements(analysis.kernel.body, scope),
  ];
  const promotedRawOperations = promoteSemanticBarrierResultCalls(rawOperations, sourceBarrierFunctions);
  const barrierBranchOperations = lowerSemanticDivergentBarrierBranches(
    promotedRawOperations,
    loweredSourceFunctions,
    analysis.kernel.span,
    analysis.barrierUniformity.kernel,
    guardedBarrierFunctionNames,
  );
  const breakOperations = lowerSemanticDivergentBreaksBeforeBarriers(
    barrierBranchOperations,
    analysis.kernel.span,
    analysis.barrierUniformity.kernel,
    sourceBarrierFunctions,
  );
  const continueOperations = lowerSemanticDivergentContinuesBeforeBarriers(
    breakOperations,
    analysis.kernel.span,
    analysis.barrierUniformity.kernel,
    sourceBarrierFunctions,
  );
  const activeLaneOperations = lowerSemanticEarlyReturnsBeforeCollectives(
    continueOperations,
    loweredSourceFunctions,
    analysis.kernel.span,
    analysis.barrierUniformity.kernel,
  );
  const controlledOperations = markSemanticWorkgroupUniformControl(
    activeLaneOperations,
    analysis.barrierUniformity.kernel.workgroupUniformControlStatementStarts,
  );
  const loweredOperations = lowerSemanticBindlessTextureAliases(
    controlledOperations,
    loweredSourceFunctions,
    options.bindlessTextures ?? [],
  );
  const localMemory = collectDeclaredMemory(loweredOperations);
  const reachable = collectReachableAnalysisNames(analysis);
  const reachableSemanticFunctions = collectReachableSemanticFunctionIds(loweredOperations, loweredSourceFunctions);
  const sharedMemorySymbols = [...semantic.symbols, ...localMemory]
    .filter((symbol) => symbol.addressSpace === "shared")
    .map((symbol) => semanticMemorySymbolWithDynamicSharedExtent(symbol, options.dynamicSharedMemory));
  const reachableSourceFunctions = loweredSourceFunctions.filter((fn) =>
    reachableSemanticFunctions.has(fn.id.key) && !isSemanticGeneratedRandomCall(fn.name)
  );
  const sourceSharedMemoryDimensions = new Map(
    sharedMemorySymbols.map((symbol) => [symbol.name, symbol.dimensions] as const),
  );
  const sourceSharedMemoryValueTypes = new Map(
    sharedMemorySymbols.map((symbol) => [symbol.name, symbol.valueType] as const),
  );
  const localAddressSpaceOverloads = cloneMixedLocalPointerFunctionOverloads(
    loweredOperations,
    reachableSourceFunctions,
  );
  const addressSpaceOverloads = cloneMixedSharedPointerFunctionOverloads(
    loweredOperations,
    localAddressSpaceOverloads,
    sourceSharedMemoryDimensions,
    sourceSharedMemoryValueTypes,
  );
  const resolved = resolveSemanticFunctionOverloads(loweredOperations, addressSpaceOverloads);
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
  const functionIdsByName = new Map(functions.map((fn) => [fn.name, fn.id]));
  const identifiedOperations = linkSemanticOperationCallIdentities(operations, functionIdsByName);
  const identifiedFunctions = functions.map((fn) => ({
    ...fn,
    body: linkSemanticOperationCallIdentities(fn.body, functionIdsByName),
  }));
  const functionSharedMemory = identifiedFunctions.flatMap((fn) =>
    collectDeclaredMemory(fn.body).filter((symbol) => symbol.addressSpace === "shared")
  );
  return completeCanonicalLowering({
    kind: "semantic-kernel-ir",
    name: analysis.kernel.name,
    span: analysis.kernel.span,
    symbols: [
      ...semantic.symbols,
      ...functionSymbols,
      ...identifiedFunctions.map(symbolForSemanticFunctionDeclaration),
    ],
    params: semantic.params,
    memory: [
      ...semantic.symbols.filter((symbol) =>
        symbol.kind !== "param" &&
        symbol.kind !== "function" &&
        reachable.symbolNames.has(symbol.name)
      ).map((symbol) => semanticMemorySymbolWithDynamicSharedExtent(symbol, options.dynamicSharedMemory)),
      ...localMemory.map((symbol) => semanticMemorySymbolWithDynamicSharedExtent(symbol, options.dynamicSharedMemory)),
      ...functionSharedMemory.map((symbol) => semanticMemorySymbolWithDynamicSharedExtent(symbol, options.dynamicSharedMemory)),
      ...(options.bindlessTextures ?? []).map((name): CudaLiteSemanticSymbol => ({
        id: createSemanticSymbolId("texture", name, analysis.kernel.span),
        name,
        kind: "texture",
        valueType: "texture2d",
        pointer: false,
        constant: true,
        dimensions: [],
        addressSpace: "texture",
        span: analysis.kernel.span,
      })),
    ],
    functions: identifiedFunctions,
    launchableEntries: semantic.launchableEntries,
    operations: identifiedOperations,
    requiredFeatures: semantic.requiredFeatures,
    barrierUniformity: analysis.barrierUniformity,
    workgroupSize: normalizeWorkgroupSize(options.workgroupSize ?? DEFAULT_WORKGROUP_SIZE),
    ...(options.subgroupMode === undefined ? {} : { subgroupMode: options.subgroupMode }),
    ...(options.bindlessTextures === undefined ? {} : { bindlessTextures: options.bindlessTextures }),
  }, semantic);
}

function lowerSemanticDivergentBreaksBeforeBarriers(
  operations: readonly SemanticKernelIrOperation[],
  scopeSpan: SourceSpan,
  proof: CudaLiteAnalysis["barrierUniformity"]["kernel"],
  barrierFunctions: ReadonlySet<string>,
): readonly SemanticKernelIrOperation[] {
  const unverified = new Set(proof.unverifiedControlStatementStarts);
  const lower = (items: readonly SemanticKernelIrOperation[]): readonly SemanticKernelIrOperation[] =>
    items.flatMap((operation): readonly SemanticKernelIrOperation[] => {
      if (operation.kind === "branch") {
        return [{
          ...operation,
          consequent: lower(operation.consequent),
          alternate: lower(operation.alternate),
        }];
      }
      if (operation.kind === "block") return [{ ...operation, body: lower(operation.body) }];
      if (operation.kind !== "loop") return [operation];

      const body = lower(operation.body);
      const continuing = operation.continuing === undefined ? undefined : lower(operation.continuing);
      const breakStarts = semanticCurrentLoopBreakStarts(body);
      if (![...breakStarts].some((start) => unverified.has(start)) || unverified.has(operation.span.start)) {
        return [{
          ...operation,
          body,
          ...(continuing === undefined ? {} : { continuing }),
        }];
      }

      const activeName = `bg_loop_active_${operation.span.start}`;
      const active: CudaLiteSemanticSymbol = {
        id: createSemanticSymbolId("generated-local", activeName, scopeSpan),
        name: activeName,
        kind: "local",
        valueType: "bool",
        dimensions: [],
        addressSpace: "local",
        span: operation.span,
      };
      const activeExpression = semanticSymbolExpression(active, operation.span);
      const loopControlSymbols = semanticExpressionSymbolIds(operation.condition);
      const declaration: SemanticKernelIrOperation = {
        kind: "declare",
        target: active,
        init: booleanExpression(true, operation.span),
        span: operation.span,
      };
      return [declaration, {
        ...operation,
        body: lowerSemanticLoopActiveOperations(body, active, activeExpression, loopControlSymbols, barrierFunctions),
        ...(continuing === undefined ? {} : {
          continuing: lowerSemanticLoopActiveOperations(continuing, active, activeExpression, loopControlSymbols, barrierFunctions),
        }),
      }];
    });
  return lower(operations);
}

/**
 * A canonical divergent continue can skip a lane's useful work while every
 * lane still reaches the workgroup barrier. The boolean is declared inside
 * the loop body, so it resets for every CUDA iteration; the for-loop update
 * remains outside the guard and therefore still runs after a continue.
 */
function lowerSemanticDivergentContinuesBeforeBarriers(
  operations: readonly SemanticKernelIrOperation[],
  scopeSpan: SourceSpan,
  proof: CudaLiteAnalysis["barrierUniformity"]["kernel"],
  barrierFunctions: ReadonlySet<string>,
): readonly SemanticKernelIrOperation[] {
  const unverified = new Set(proof.unverifiedControlStatementStarts);
  const lower = (items: readonly SemanticKernelIrOperation[]): readonly SemanticKernelIrOperation[] =>
    items.flatMap((operation): readonly SemanticKernelIrOperation[] => {
      if (operation.kind === "branch") {
        return [{
          ...operation,
          consequent: lower(operation.consequent),
          alternate: lower(operation.alternate),
        }];
      }
      if (operation.kind === "block") return [{ ...operation, body: lower(operation.body) }];
      if (operation.kind !== "loop") return [operation];

      const body = lower(operation.body);
      const continuing = operation.continuing === undefined ? undefined : lower(operation.continuing);
      const continueStarts = semanticBarrierSafeCurrentLoopContinueStarts(body, unverified, barrierFunctions);
      if (
        operation.loopKind !== "for" ||
        operation.condition === undefined ||
        operation.update === undefined ||
        unverified.has(operation.span.start) ||
        continueStarts === undefined
      ) {
        return [{
          ...operation,
          body,
          ...(continuing === undefined ? {} : { continuing }),
        }];
      }

      const activeName = `bg_continue_active_${operation.span.start}`;
      const active: CudaLiteSemanticSymbol = {
        id: createSemanticSymbolId("generated-local", activeName, scopeSpan),
        name: activeName,
        kind: "local",
        valueType: "bool",
        dimensions: [],
        addressSpace: "local",
        span: operation.span,
      };
      const activeExpression = semanticSymbolExpression(active, operation.span);
      const declaration: SemanticKernelIrOperation = {
        kind: "declare",
        target: active,
        init: booleanExpression(true, operation.span),
        span: operation.span,
      };
      return [{
        ...operation,
        body: [
          declaration,
          ...lowerSemanticLoopContinueOperations(body, active, activeExpression, continueStarts, barrierFunctions),
        ],
        ...(continuing === undefined ? {} : { continuing }),
      }];
    });
  return lower(operations);
}

function semanticBarrierSafeCurrentLoopContinueStarts(
  operations: readonly SemanticKernelIrOperation[],
  unverified: ReadonlySet<number>,
  barrierFunctions: ReadonlySet<string>,
): ReadonlySet<number> | undefined {
  const starts = new Set<number>();
  for (const [index, operation] of operations.entries()) {
    const continueStart = semanticDirectCurrentLoopContinueStart(operation);
    if (continueStart !== undefined && unverified.has(continueStart)) {
      if (!operations.slice(index + 1).some((item) => semanticDirectBarrierOperation(item, barrierFunctions))) return undefined;
      starts.add(continueStart);
      continue;
    }
    if (
      operation.kind === "loop" ||
      operation.kind === "block" ||
      operation.kind === "declare" ||
      semanticOperationContainsCurrentLoopExitOrContinue(operation) ||
      semanticOperationContainsNestedBarrier(operation, barrierFunctions)
    ) return undefined;
  }
  return starts.size === 0 ? undefined : starts;
}

function semanticDirectCurrentLoopContinueStart(operation: SemanticKernelIrOperation): number | undefined {
  if (operation.kind !== "branch" || operation.alternate.length !== 0 || operation.consequent.length !== 1) return undefined;
  const consequent = operation.consequent[0]!;
  return consequent.kind === "continue" ? consequent.span.start : undefined;
}

function semanticDirectBarrierOperation(
  operation: SemanticKernelIrOperation,
  barrierFunctions: ReadonlySet<string>,
): boolean {
  return operation.kind === "barrier" || operation.kind === "call" && barrierFunctions.has(operation.callee);
}

function semanticOperationContainsCurrentLoopExitOrContinue(operation: SemanticKernelIrOperation): boolean {
  if (operation.kind === "break" || operation.kind === "continue" || operation.kind === "return") return true;
  if (operation.kind === "branch") {
    return operation.consequent.some(semanticOperationContainsCurrentLoopExitOrContinue) ||
      operation.alternate.some(semanticOperationContainsCurrentLoopExitOrContinue);
  }
  return operation.kind === "loop" || operation.kind === "block"
    ? operation.body.some(semanticOperationContainsCurrentLoopExitOrContinue)
    : false;
}

function semanticOperationContainsNestedBarrier(
  operation: SemanticKernelIrOperation,
  barrierFunctions: ReadonlySet<string>,
): boolean {
  if (semanticDirectBarrierOperation(operation, barrierFunctions)) return false;
  if (operation.kind === "branch") {
    return operation.consequent.some((item) => semanticOperationContainsBarrier(item, barrierFunctions)) ||
      operation.alternate.some((item) => semanticOperationContainsBarrier(item, barrierFunctions));
  }
  return operation.kind === "loop" || operation.kind === "block"
    ? operation.body.some((item) => semanticOperationContainsBarrier(item, barrierFunctions))
    : false;
}

function semanticOperationContainsBarrier(
  operation: SemanticKernelIrOperation,
  barrierFunctions: ReadonlySet<string>,
): boolean {
  if (semanticDirectBarrierOperation(operation, barrierFunctions)) return true;
  if (operation.kind === "branch") {
    return operation.consequent.some((item) => semanticOperationContainsBarrier(item, barrierFunctions)) ||
      operation.alternate.some((item) => semanticOperationContainsBarrier(item, barrierFunctions));
  }
  return operation.kind === "loop" || operation.kind === "block"
    ? operation.body.some((item) => semanticOperationContainsBarrier(item, barrierFunctions))
    : false;
}

function lowerSemanticLoopContinueOperations(
  operations: readonly SemanticKernelIrOperation[],
  active: CudaLiteSemanticSymbol,
  activeExpression: SemanticExpression,
  continueStarts: ReadonlySet<number>,
  barrierFunctions: ReadonlySet<string>,
): readonly SemanticKernelIrOperation[] {
  return operations.flatMap((operation): readonly SemanticKernelIrOperation[] => {
    const rewritten = rewriteSemanticCurrentLoopContinuesAsInactive(operation, active, continueStarts);
    if (semanticDirectBarrierOperation(operation, barrierFunctions)) return [rewritten];
    return [semanticActiveLaneBranch(activeExpression, [rewritten], operation.span)];
  });
}

function rewriteSemanticCurrentLoopContinuesAsInactive(
  operation: SemanticKernelIrOperation,
  active: CudaLiteSemanticSymbol,
  continueStarts: ReadonlySet<number>,
): SemanticKernelIrOperation {
  if (operation.kind === "continue" && continueStarts.has(operation.span.start)) {
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
      consequent: operation.consequent.map((item) => rewriteSemanticCurrentLoopContinuesAsInactive(item, active, continueStarts)),
      alternate: operation.alternate.map((item) => rewriteSemanticCurrentLoopContinuesAsInactive(item, active, continueStarts)),
    };
  }
  return operation;
}

function linkSemanticOperationCallIdentities(
  operations: readonly SemanticKernelIrOperation[],
  functionIdsByName: ReadonlyMap<string, SemanticFunctionId>,
): readonly SemanticKernelIrOperation[] {
  return operations.map((operation): SemanticKernelIrOperation => {
    if (operation.kind === "call") {
      const functionId = functionIdsByName.get(operation.callee);
      return functionId === undefined
        ? operation
        : { ...operation, calleeId: semanticSymbolIdFromFunction(functionId) };
    }
    if (operation.kind === "branch") {
      return {
        ...operation,
        consequent: linkSemanticOperationCallIdentities(operation.consequent, functionIdsByName),
        alternate: linkSemanticOperationCallIdentities(operation.alternate, functionIdsByName),
      };
    }
    if (operation.kind === "block") {
      return { ...operation, body: linkSemanticOperationCallIdentities(operation.body, functionIdsByName) };
    }
    if (operation.kind === "loop") {
      return {
        ...operation,
        ...(operation.init !== undefined && isSemanticKernelIrOperation(operation.init)
          ? { init: linkSemanticOperationCallIdentities([operation.init], functionIdsByName)[0]! }
          : {}),
        body: linkSemanticOperationCallIdentities(operation.body, functionIdsByName),
        ...(operation.continuing === undefined
          ? {}
          : { continuing: linkSemanticOperationCallIdentities(operation.continuing, functionIdsByName) }),
      };
    }
    return operation;
  });
}

function semanticCurrentLoopBreakStarts(
  operations: readonly SemanticKernelIrOperation[],
): ReadonlySet<number> {
  const starts = new Set<number>();
  const visit = (items: readonly SemanticKernelIrOperation[]): void => {
    for (const operation of items) {
      if (operation.kind === "break") starts.add(operation.span.start);
      else if (operation.kind === "branch") {
        visit(operation.consequent);
        visit(operation.alternate);
      } else if (operation.kind === "block") visit(operation.body);
    }
  };
  visit(operations);
  return starts;
}

function lowerSemanticLoopActiveOperations(
  operations: readonly SemanticKernelIrOperation[],
  active: CudaLiteSemanticSymbol,
  activeExpression: SemanticExpression,
  loopControlSymbols: ReadonlySet<SemanticSymbolId>,
  barrierFunctions: ReadonlySet<string>,
): readonly SemanticKernelIrOperation[] {
  return operations.flatMap((operation): readonly SemanticKernelIrOperation[] => {
    const rewritten = rewriteSemanticCurrentLoopBreaksAsInactive(operation, active);
    if (semanticIrOperationsContainBarrier([operation], barrierFunctions)) {
      if (operation.kind === "loop" || operation.kind === "block") {
        return [{
          ...operation,
          body: lowerSemanticLoopActiveOperations(operation.body, active, activeExpression, loopControlSymbols, barrierFunctions),
          ...(operation.kind === "loop" && operation.continuing !== undefined ? {
            continuing: lowerSemanticLoopActiveOperations(operation.continuing, active, activeExpression, loopControlSymbols, barrierFunctions),
          } : {}),
        }];
      }
      if (operation.kind === "branch") {
        return [{
          ...operation,
          consequent: lowerSemanticLoopActiveOperations(operation.consequent, active, activeExpression, loopControlSymbols, barrierFunctions),
          alternate: lowerSemanticLoopActiveOperations(operation.alternate, active, activeExpression, loopControlSymbols, barrierFunctions),
        }];
      }
      return [operation];
    }
    if (semanticOperationAdvancesLoopControl(operation, loopControlSymbols)) return [rewritten];
    if (operation.kind === "branch" && semanticCurrentLoopBreakStarts([operation]).size > 0) {
      return [semanticActiveLaneBranch(activeExpression, [rewritten], operation.span)];
    }
    if (operation.kind === "declare") {
      if (operation.target.pointer && !semanticPointerSymbolNeedsRuntimeState(operation.target)) return [rewritten];
      if (operation.target.addressSpace === "shared" || operation.init === undefined || operation.target.dimensions.length > 0) return [rewritten];
      const { init: _init, ...declaration } = operation;
      return [declaration, semanticActiveLaneBranch(activeExpression, [{
        kind: "expression",
        expression: {
          kind: "assignment",
          operator: "=",
          target: semanticSymbolExpression(operation.target, operation.span),
          value: operation.init,
          valueType: requiredSemanticValueType(operation.target.valueType, `active-lane assignment '${operation.target.name}'`, operation.span),
          span: operation.span,
        },
        span: operation.span,
      }], operation.span)];
    }
    return [semanticActiveLaneBranch(activeExpression, [rewritten], operation.span)];
  });
}

function semanticExpressionSymbolIds(
  expression: SemanticExpression | undefined,
): ReadonlySet<SemanticSymbolId> {
  const ids = new Set<SemanticSymbolId>();
  if (expression !== undefined) {
    walkSemanticExpression(expression, (item) => {
      if (item.kind === "symbol") ids.add(item.id);
    });
  }
  return ids;
}

function semanticOperationAdvancesLoopControl(
  operation: SemanticKernelIrOperation,
  loopControlSymbols: ReadonlySet<SemanticSymbolId>,
): boolean {
  if (operation.kind !== "expression") return false;
  const expression = operation.expression;
  if (expression.kind === "update" && expression.argument.kind === "symbol") {
    return loopControlSymbols.has(expression.argument.id);
  }
  return expression.kind === "assignment" && expression.target.kind === "symbol" &&
    loopControlSymbols.has(expression.target.id);
}

function rewriteSemanticCurrentLoopBreaksAsInactive(
  operation: SemanticKernelIrOperation,
  active: CudaLiteSemanticSymbol,
): SemanticKernelIrOperation {
  if (operation.kind === "break") {
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
      consequent: operation.consequent.map((item) => rewriteSemanticCurrentLoopBreaksAsInactive(item, active)),
      alternate: operation.alternate.map((item) => rewriteSemanticCurrentLoopBreaksAsInactive(item, active)),
    };
  }
  if (operation.kind === "block") {
    return { ...operation, body: operation.body.map((item) => rewriteSemanticCurrentLoopBreaksAsInactive(item, active)) };
  }
  return operation;
}

function markSemanticWorkgroupUniformControl(
  operations: readonly SemanticKernelIrOperation[],
  statementStarts: readonly number[],
): readonly SemanticKernelIrOperation[] {
  if (statementStarts.length === 0) return operations;
  const starts = new Set(statementStarts);
  return operations.map((operation): SemanticKernelIrOperation => {
    if (operation.kind === "branch") {
      return {
        ...operation,
        ...(starts.has(operation.span.start) ? { conditionUniformity: "workgroup" as const } : {}),
        consequent: markSemanticWorkgroupUniformControl(operation.consequent, statementStarts),
        alternate: markSemanticWorkgroupUniformControl(operation.alternate, statementStarts),
      };
    }
    if (operation.kind === "loop" || operation.kind === "block") {
      return {
        ...operation,
        body: markSemanticWorkgroupUniformControl(operation.body, statementStarts),
        ...(operation.kind === "loop" && operation.continuing !== undefined
          ? { continuing: markSemanticWorkgroupUniformControl(operation.continuing, statementStarts) }
          : {}),
      };
    }
    return operation;
  });
}

function lowerSemanticBindlessTextureAliases(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
  bindlessTextures: readonly string[],
  inherited: ReadonlyMap<string, SemanticExpression> = new Map(),
): readonly SemanticKernelIrOperation[] {
  if (bindlessTextures.length === 0) return operations;
  const aliases = new Map(inherited);
  const out: SemanticKernelIrOperation[] = [];
  for (const operation of operations) {
    if (operation.kind === "declare" && operation.target.valueType === "texture2d" && operation.init?.kind === "call") {
      const handle = semanticBindlessHandleFromCall(operation.init, functions);
      if (handle !== undefined) {
        aliases.set(operation.target.name, handle);
        continue;
      }
    }
    out.push(rewriteSemanticBindlessOperation(operation, aliases, functions, bindlessTextures));
  }
  return out;
}

function semanticBindlessHandleFromCall(
  call: Extract<SemanticExpression, { readonly kind: "call" }>,
  functions: readonly CudaLiteSemanticFunction[],
): SemanticExpression | undefined {
  if (call.callee.kind !== "symbol" || call.args.length !== 1) return undefined;
  const calleeName = call.callee.name;
  const fn = functions.find((candidate) => candidate.name === calleeName && candidate.returnType === "texture2d" && candidate.params.length === 1);
  const param = fn?.params[0];
  const arg = call.args[0];
  if (!fn || !param || !arg || fn.body.length !== 1 || fn.body[0]?.kind !== "return" || fn.body[0].value === undefined) return undefined;
  if (!semanticTextureHandleReturnUsesLowLane(fn.body[0].value, param.name)) return undefined;
  if (param.valueType === "uint2") {
    return { kind: "member", object: arg, property: "x", valueType: "uint", span: call.span };
  }
  return param.valueType === "uint" ? arg : undefined;
}

function semanticTextureHandleReturnUsesLowLane(expression: SemanticExpression, paramName: string): boolean {
  if (expression.kind === "cast") return semanticTextureHandleReturnUsesLowLane(expression.expression, paramName);
  if (expression.kind === "binary" && expression.operator === "|") return semanticTextureHandleReturnUsesLowLane(expression.left, paramName);
  return expression.kind === "member" && expression.property === "x" && expression.object.kind === "symbol" && expression.object.name === paramName;
}

function rewriteSemanticBindlessOperation(
  operation: SemanticKernelIrOperation,
  aliases: ReadonlyMap<string, SemanticExpression>,
  functions: readonly CudaLiteSemanticFunction[],
  bindlessTextures: readonly string[],
): SemanticKernelIrOperation {
  const rewrite = (expression: SemanticExpression): SemanticExpression => rewriteSemanticBindlessExpression(expression, aliases);
  if (operation.kind === "declare") return operation.init === undefined ? operation : { ...operation, init: rewrite(operation.init) };
  if (operation.kind === "store") return { ...operation, value: rewrite(operation.value), reads: collectMemoryRefs(rewrite(operation.value)) };
  if (operation.kind === "expression") return { ...operation, expression: rewrite(operation.expression) };
  if (operation.kind === "call") return { ...operation, args: operation.args.map(rewrite) };
  if (operation.kind === "atomic") return { ...operation, args: operation.args.map(rewrite) };
  if (operation.kind === "branch") return {
    ...operation,
    condition: rewrite(operation.condition),
    consequent: lowerSemanticBindlessTextureAliases(operation.consequent, functions, bindlessTextures, aliases),
    alternate: lowerSemanticBindlessTextureAliases(operation.alternate, functions, bindlessTextures, aliases),
  };
  if (operation.kind === "loop") return {
    ...operation,
    ...(operation.init === undefined ? {} : { init: isSemanticKernelIrOperation(operation.init) ? rewriteSemanticBindlessOperation(operation.init, aliases, functions, bindlessTextures) : rewrite(operation.init) }),
    ...(operation.condition === undefined ? {} : { condition: rewrite(operation.condition) }),
    ...(operation.update === undefined ? {} : { update: rewrite(operation.update) }),
    body: lowerSemanticBindlessTextureAliases(operation.body, functions, bindlessTextures, aliases),
    ...(operation.continuing === undefined ? {} : { continuing: lowerSemanticBindlessTextureAliases(operation.continuing, functions, bindlessTextures, aliases) }),
  };
  if (operation.kind === "block") return { ...operation, body: lowerSemanticBindlessTextureAliases(operation.body, functions, bindlessTextures, aliases) };
  if (operation.kind === "return" && operation.value !== undefined) return { ...operation, value: rewrite(operation.value) };
  return operation;
}

function rewriteSemanticBindlessExpression(
  expression: SemanticExpression,
  aliases: ReadonlyMap<string, SemanticExpression>,
): SemanticExpression {
  switch (expression.kind) {
    case "member": return { ...expression, object: rewriteSemanticBindlessExpression(expression.object, aliases) };
    case "index": return { ...expression, target: rewriteSemanticBindlessExpression(expression.target, aliases), index: rewriteSemanticBindlessExpression(expression.index, aliases) };
    case "call": return { ...expression, callee: rewriteSemanticBindlessExpression(expression.callee, aliases), args: expression.args.map((arg) => rewriteSemanticBindlessExpression(arg, aliases)) };
    case "texture-read": {
      const texture = expression.texture.kind === "symbol" ? aliases.get(expression.texture.name) ?? expression.texture : rewriteSemanticBindlessExpression(expression.texture, aliases);
      return { ...expression, texture, x: rewriteSemanticBindlessExpression(expression.x, aliases), y: rewriteSemanticBindlessExpression(expression.y, aliases), ...(expression.z === undefined ? {} : { z: rewriteSemanticBindlessExpression(expression.z, aliases) }) };
    }
    case "surface-read": return { ...expression, surface: rewriteSemanticBindlessExpression(expression.surface, aliases), xBytes: rewriteSemanticBindlessExpression(expression.xBytes, aliases), y: rewriteSemanticBindlessExpression(expression.y, aliases), ...(expression.z === undefined ? {} : { z: rewriteSemanticBindlessExpression(expression.z, aliases) }) };
    case "cast": return { ...expression, expression: rewriteSemanticBindlessExpression(expression.expression, aliases) };
    case "unary": return { ...expression, argument: rewriteSemanticBindlessExpression(expression.argument, aliases) };
    case "binary": return { ...expression, left: rewriteSemanticBindlessExpression(expression.left, aliases), right: rewriteSemanticBindlessExpression(expression.right, aliases) };
    case "conditional": return { ...expression, condition: rewriteSemanticBindlessExpression(expression.condition, aliases), consequent: rewriteSemanticBindlessExpression(expression.consequent, aliases), alternate: rewriteSemanticBindlessExpression(expression.alternate, aliases) };
    case "assignment": return { ...expression, target: rewriteSemanticBindlessExpression(expression.target, aliases), value: rewriteSemanticBindlessExpression(expression.value, aliases) };
    case "update": return { ...expression, argument: rewriteSemanticBindlessExpression(expression.argument, aliases) };
    case "initializer": return { ...expression, elements: expression.elements.map((item) => rewriteSemanticBindlessExpression(item, aliases)) };
    case "sequence": return { ...expression, expressions: expression.expressions.map((item) => rewriteSemanticBindlessExpression(item, aliases)) };
    default: return expression;
  }
}

const SEMANTIC_GUARDED_BARRIER_FUNCTION_SUFFIX = "__bg_guarded_barrier";

function semanticGuardedBarrierFunctionName(name: string): string {
  return `${name}${SEMANTIC_GUARDED_BARRIER_FUNCTION_SUFFIX}`;
}

function createSemanticGuardedBarrierFunctions(
  functions: readonly CudaLiteSemanticFunction[],
  barrierFunctions: ReadonlySet<string>,
  guardedBarrierFunctionNames: ReadonlyMap<string, string>,
): readonly CudaLiteSemanticFunction[] {
  return functions.filter((fn) => barrierFunctions.has(fn.name)).map((fn): CudaLiteSemanticFunction => {
    const active: CudaLiteSemanticSymbol = {
      id: createSemanticSymbolId("guard-param", `bg_call_active:${fn.name}`, fn.span),
      name: "bg_call_active",
      kind: "param",
      valueType: "bool",
      pointer: false,
      constant: true,
      dimensions: [],
      addressSpace: "local",
      span: fn.span,
    };
    return {
      ...fn,
      name: guardedBarrierFunctionNames.get(fn.name)!,
      id: createSemanticFunctionId(guardedBarrierFunctionNames.get(fn.name)!, fn.span),
      params: [...fn.params, active],
      body: lowerSemanticPredicatedBarrierOperations(
        promoteSemanticBarrierResultCalls(fn.body, barrierFunctions),
        semanticSymbolExpression(active, fn.span),
        barrierFunctions,
        guardedBarrierFunctionNames,
      ),
    };
  });
}

function lowerSemanticDivergentBarrierBranches(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
  kernelSpan: SourceSpan,
  barrierProof: CudaLiteAnalysis["barrierUniformity"]["kernel"],
  guardedBarrierFunctionNames: ReadonlyMap<string, string> = new Map(),
): readonly SemanticKernelIrOperation[] {
  const branchStarts = barrierProof.unverifiedControlStatementStarts.length > 0
    ? barrierProof.unverifiedControlStatementStarts
    : barrierProof.workgroupUniformControlStatementStarts;
  if (branchStarts.length === 0) return operations;
  const branchStartSet = new Set(branchStarts);
  const barrierFunctions = semanticIrBarrierFunctionNames(functions);
  return operations.flatMap((operation): readonly SemanticKernelIrOperation[] => {
    if (operation.kind !== "branch" || !branchStartSet.has(operation.span.start) || operation.alternate.length > 0 ||
      !semanticIrOperationsContainBarrier(operation.consequent, barrierFunctions) || semanticOperationContainsVoidReturn(operation) ||
      semanticPredicatedBarrierTransformUnsafe(operation.consequent, barrierFunctions, guardedBarrierFunctionNames)) {
      return [operation];
    }
    const predicateName = branchStarts.length === 1 ? "bg_active_lane" : `bg_active_lane_${operation.span.start}`;
    const predicate: CudaLiteSemanticSymbol = {
      id: createSemanticSymbolId("generated-local", predicateName, kernelSpan),
      name: predicateName,
      kind: "local",
      valueType: "bool",
      dimensions: [],
      addressSpace: "local",
      span: kernelSpan,
    };
    const predicateExpression = semanticSymbolExpression(predicate, kernelSpan);
    const declaration: SemanticKernelIrOperation = {
      kind: "declare",
      target: predicate,
      init: operation.condition,
      span: operation.span,
    };
    return [{
      kind: "block",
      body: [
        declaration,
        ...lowerSemanticPredicatedBarrierOperations(
          operation.consequent,
          predicateExpression,
          barrierFunctions,
          guardedBarrierFunctionNames,
        ),
      ],
      span: operation.span,
    }];
  });
}

function semanticPredicatedBarrierTransformUnsafe(
  operations: readonly SemanticKernelIrOperation[],
  barrierFunctions: ReadonlySet<string>,
  guardedBarrierFunctionNames: ReadonlyMap<string, string>,
): boolean {
  return operations.some((operation) => {
    if (operation.kind === "call" && barrierFunctions.has(operation.callee)) {
      return !guardedBarrierFunctionNames.has(operation.callee);
    }
    if (operation.kind === "declare") {
      return operation.target.addressSpace !== "local" || operation.target.pointer ||
        operation.target.dimensions.length > 0 && operation.init !== undefined;
    }
    if (operation.kind === "branch" && semanticIrOperationsContainBarrier([operation], barrierFunctions)) return true;
    if (operation.kind === "loop" || operation.kind === "block") {
      return semanticPredicatedBarrierTransformUnsafe(operation.body, barrierFunctions, guardedBarrierFunctionNames) ||
        (operation.kind === "loop" && operation.continuing !== undefined && semanticPredicatedBarrierTransformUnsafe(operation.continuing, barrierFunctions, guardedBarrierFunctionNames));
    }
    return false;
  });
}

function lowerSemanticPredicatedBarrierOperations(
  operations: readonly SemanticKernelIrOperation[],
  predicate: SemanticExpression,
  barrierFunctions: ReadonlySet<string>,
  guardedBarrierFunctionNames: ReadonlyMap<string, string> = new Map(),
): readonly SemanticKernelIrOperation[] {
  return operations.flatMap((operation): readonly SemanticKernelIrOperation[] => {
    if (semanticIrOperationsContainBarrier([operation], barrierFunctions)) {
      if (operation.kind === "call" && barrierFunctions.has(operation.callee)) {
        const guarded = guardedBarrierFunctionNames.get(operation.callee);
        return guarded === undefined ? [operation] : [{ ...operation, callee: guarded, args: [...operation.args, predicate] }];
      }
      if (operation.kind === "loop" || operation.kind === "block") {
        return [{ ...operation, body: lowerSemanticPredicatedBarrierOperations(operation.body, predicate, barrierFunctions, guardedBarrierFunctionNames),
          ...(operation.kind === "loop" && operation.continuing !== undefined ? { continuing: lowerSemanticPredicatedBarrierOperations(operation.continuing, predicate, barrierFunctions, guardedBarrierFunctionNames) } : {}) }];
      }
      if (operation.kind === "branch") {
        return lowerSemanticPredicatedBarrierOperations(operation.consequent, predicate, barrierFunctions, guardedBarrierFunctionNames);
      }
      return [operation];
    }
    if (operation.kind === "declare") {
      if (operation.target.addressSpace !== "local" || operation.target.pointer || operation.target.dimensions.length > 0 && operation.init !== undefined) return [operation];
      if (operation.init === undefined) return [operation];
      const { init: _init, ...declaration } = operation;
      return [declaration, semanticActiveLaneBranch(predicate, [{
        kind: "expression",
        expression: {
          kind: "assignment",
          operator: "=",
          target: semanticSymbolExpression(operation.target, operation.span),
          value: operation.init,
          valueType: requiredSemanticValueType(operation.target.valueType, `predicated assignment '${operation.target.name}'`, operation.span),
          span: operation.span,
        },
        span: operation.span,
      }], operation.span)];
    }
    return [semanticActiveLaneBranch(predicate, [operation], operation.span)];
  });
}

function lowerSemanticEarlyReturnsBeforeCollectives(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
  kernelSpan: SourceSpan,
  barrierProof: CudaLiteAnalysis["barrierUniformity"]["kernel"],
): readonly SemanticKernelIrOperation[] {
  const barrierFunctions = semanticIrBarrierFunctionNames(functions);
  const firstReturnBeforeCollective = operations.findIndex((operation, index) =>
    semanticOperationContainsVoidReturn(operation) &&
    (semanticOperationContainsVoidReturnBeforeCollective(operation, barrierFunctions) ||
      semanticIrOperationsContainSchedulingCollective(operations.slice(index + 1), barrierFunctions))
  );
  if (firstReturnBeforeCollective < 0) return operations;
  const affected = operations.slice(firstReturnBeforeCollective);
  const returnStarts = semanticVoidReturnStarts(affected);
  if (barrierProof.unverifiedControlStatementStarts.some((start) => !returnStarts.has(start))) return operations;
  const pointerFunctions = new Map(functions.filter((fn) => fn.params.some((param) => param.pointer)).map((fn) => [fn.name, fn] as const));
  const pointerWrites = semanticFunctionPointerWrites(functions);
  if (!semanticVoidReturnsAreTerminal(affected)) return operations;
  if (affected.some((operation) => semanticActiveLaneTransformUnsafe(operation, pointerFunctions, pointerWrites, barrierFunctions))) return operations;

  const firstAffected = operations[firstReturnBeforeCollective]!;
  const activeName = firstAffected.kind === "loop"
    ? `bg_barrier_loop_active_${firstAffected.span.start}`
    : "bg_active_lane";
  const active: CudaLiteSemanticSymbol = {
    id: createSemanticSymbolId("generated-local", activeName, kernelSpan),
    name: activeName,
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
  const lowered = lowerSemanticActiveLaneOperations(affected, active, activeExpression, barrierFunctions);
  return [...operations.slice(0, firstReturnBeforeCollective), declare, ...lowered];
}

function semanticOperationContainsVoidReturnBeforeCollective(
  operation: SemanticKernelIrOperation,
  barrierFunctions: ReadonlySet<string>,
): boolean {
  if (operation.kind === "branch") {
    return semanticOperationsContainVoidReturnBeforeCollective(operation.consequent, barrierFunctions) ||
      semanticOperationsContainVoidReturnBeforeCollective(operation.alternate, barrierFunctions);
  }
  if (operation.kind === "loop" || operation.kind === "block") {
    return semanticOperationsContainVoidReturnBeforeCollective(operation.body, barrierFunctions) ||
      (operation.kind === "loop" && operation.continuing !== undefined &&
        semanticOperationsContainVoidReturnBeforeCollective(operation.continuing, barrierFunctions));
  }
  return false;
}

function semanticOperationsContainVoidReturnBeforeCollective(
  operations: readonly SemanticKernelIrOperation[],
  barrierFunctions: ReadonlySet<string>,
): boolean {
  return operations.some((operation, index) =>
    semanticOperationContainsVoidReturn(operation) &&
    (semanticOperationContainsVoidReturnBeforeCollective(operation, barrierFunctions) ||
      semanticIrOperationsContainSchedulingCollective(operations.slice(index + 1), barrierFunctions))
  );
}

function semanticVoidReturnStarts(operations: readonly SemanticKernelIrOperation[]): ReadonlySet<number> {
  const starts = new Set<number>();
  const visit = (items: readonly SemanticKernelIrOperation[]): void => {
    for (const operation of items) {
      if (operation.kind === "return" && operation.value === undefined) starts.add(operation.span.start);
      else if (operation.kind === "branch") {
        visit(operation.consequent);
        visit(operation.alternate);
      } else if (operation.kind === "loop" || operation.kind === "block") {
        visit(operation.body);
        if (operation.kind === "loop" && operation.continuing) visit(operation.continuing);
      }
    }
  };
  visit(operations);
  return starts;
}

function lowerSemanticActiveLaneOperations(
  operations: readonly SemanticKernelIrOperation[],
  active: CudaLiteSemanticSymbol,
  activeExpression: SemanticExpression,
  barrierFunctions: ReadonlySet<string>,
): readonly SemanticKernelIrOperation[] {
  return operations.flatMap((operation): readonly SemanticKernelIrOperation[] => {
    const rewritten = rewriteSemanticVoidReturnsAsInactive(operation, active);
    if (semanticIrOperationsContainBarrier([operation], barrierFunctions)) {
      if (operation.kind === "loop" || operation.kind === "block") {
        return [{ ...operation, body: lowerSemanticActiveLaneOperations(operation.body, active, activeExpression, barrierFunctions),
          ...(operation.kind === "loop" && operation.continuing !== undefined ? { continuing: lowerSemanticActiveLaneOperations(operation.continuing, active, activeExpression, barrierFunctions) } : {}) }];
      }
      if (operation.kind === "branch") {
        return [{
          ...operation,
          consequent: lowerSemanticActiveLaneOperations(operation.consequent, active, activeExpression, barrierFunctions),
          alternate: lowerSemanticActiveLaneOperations(operation.alternate, active, activeExpression, barrierFunctions),
        }];
      }
      return [rewritten];
    }
    if (operation.kind === "declare") {
      if (operation.target.addressSpace === "shared" || operation.init === undefined || operation.target.dimensions.length > 0) return [rewritten];
      if (operation.target.pointer) {
        const source = semanticIrPointerArgumentMemoryRef(operation.init);
        if (source?.addressSpace !== "storage") return [rewritten];
        const target: CudaLiteSemanticSymbol = {
          ...semanticSymbolWithoutPointerAlias(operation.target),
          pointerRuntimeState: true,
        };
        const { init: _init, ...declaration } = operation;
        return [
          { ...declaration, target },
          semanticActiveLaneBranch(activeExpression, [{ kind: "pointer-rebind", target, source, span: operation.span }], operation.span),
        ];
      }
      const { init: _init, ...declaration } = operation;
      return [declaration, semanticActiveLaneBranch(activeExpression, [{
        kind: "expression",
        expression: {
          kind: "assignment",
          operator: "=",
          target: semanticSymbolExpression(operation.target, operation.span),
          value: operation.init,
          valueType: requiredSemanticValueType(operation.target.valueType, `active-lane assignment '${operation.target.name}'`, operation.span),
          span: operation.span,
        },
        span: operation.span,
      }], operation.span)];
    }
    return [semanticActiveLaneBranch(activeExpression, [rewritten], operation.span)];
  });
}

function semanticActiveLaneBranch(
  activeExpression: SemanticExpression,
  consequent: readonly SemanticKernelIrOperation[],
  span: SourceSpan,
): SemanticKernelIrOperation {
  return { kind: "branch", condition: activeExpression, consequent, alternate: [], span };
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
  pointerFunctions: ReadonlyMap<string, CudaLiteSemanticFunction>,
  pointerWrites: ReadonlyMap<string, ReadonlySet<number>>,
  barrierFunctions: ReadonlySet<string>,
): boolean {
  if (collectSemanticFunctionCalls([operation]).some((call) => {
    const fn = pointerFunctions.get(call.callee);
    return fn?.params.some((param, index) => {
      if (!param.pointer) return false;
      const ref = call.args[index] === undefined ? undefined : semanticIrPointerArgumentMemoryRef(call.args[index]!);
      const writesPointer = pointerWrites.get(call.callee)?.has(index) === true;
      return writesPointer && (ref === undefined || barrierFunctions.has(call.callee) && ref.addressSpace !== "local" && ref.addressSpace !== "shared");
    }) ?? false;
  })) return true;
  if (operation.kind === "loop") {
    return operation.body.some((item) => semanticActiveLaneTransformUnsafe(item, pointerFunctions, pointerWrites, barrierFunctions)) ||
      (operation.continuing?.some((item) => semanticActiveLaneTransformUnsafe(item, pointerFunctions, pointerWrites, barrierFunctions)) ?? false);
  }
  if (operation.kind === "declare") {
    if (operation.target.pointer && !semanticPointerSymbolNeedsRuntimeState(operation.target)) return false;
    if (operation.target.addressSpace === "shared") return operation.target.pointer || operation.init !== undefined;
    return operation.target.addressSpace !== "local" || operation.target.pointer ||
      operation.target.dimensions.length > 0 && operation.init !== undefined && (
        !semanticExpressionSideEffectFree(operation.init) ||
        semanticExpressionContainsUnsafeActiveLaneDeclarationCall(operation.init) ||
        collectMemoryRefs(operation.init).length > 0
      );
  }
  if (operation.kind === "branch") {
    return operation.consequent.some((item) => semanticActiveLaneTransformUnsafe(item, pointerFunctions, pointerWrites, barrierFunctions)) ||
      operation.alternate.some((item) => semanticActiveLaneTransformUnsafe(item, pointerFunctions, pointerWrites, barrierFunctions));
  }
  if (operation.kind === "block") return operation.body.some((item) => semanticActiveLaneTransformUnsafe(item, pointerFunctions, pointerWrites, barrierFunctions));
  return false;
}

function semanticFunctionPointerWrites(
  functions: readonly CudaLiteSemanticFunction[],
): ReadonlyMap<string, ReadonlySet<number>> {
  const byName = new Map<string, readonly CudaLiteSemanticFunction[]>();
  for (const fn of functions) byName.set(fn.name, [...byName.get(fn.name) ?? [], fn]);
  const writes = new Map<string, Set<number>>();
  for (const fn of functions) {
    const written = writes.get(fn.name) ?? new Set<number>();
    const pointerParams = new Map(fn.params.map((param, index) => [param.name, index] as const).filter(([_, index]) => fn.params[index]!.pointer));
    for (const base of semanticOperationWriteBases(fn.body)) {
      const index = pointerParams.get(base);
      if (index !== undefined) written.add(index);
    }
    writes.set(fn.name, written);
  }
  let changed = true;
  while (changed) {
    changed = false;
    for (const fn of functions) {
      const ownWrites = writes.get(fn.name)!;
      const ownPointerParams = new Map(fn.params.map((param, index) => [param.name, index] as const).filter(([_, index]) => fn.params[index]!.pointer));
      for (const call of collectSemanticFunctionCalls(fn.body)) {
        if (!byName.has(call.callee)) continue;
        for (const calleeIndex of writes.get(call.callee) ?? []) {
          const arg = call.args[calleeIndex];
          const ref = arg === undefined ? undefined : semanticIrPointerArgumentMemoryRef(arg);
          const ownIndex = ref === undefined ? undefined : ownPointerParams.get(ref.base);
          if (ownIndex !== undefined && !ownWrites.has(ownIndex)) {
            ownWrites.add(ownIndex);
            changed = true;
          }
        }
      }
    }
  }
  return writes;
}

function semanticOperationWriteBases(
  operations: readonly SemanticKernelIrOperation[],
): ReadonlySet<string> {
  const bases = new Set<string>();
  for (const operation of operations) {
    if (operation.kind === "store" || operation.kind === "copy" || operation.kind === "matrix-store") bases.add(operation.target.base);
    else if (operation.kind === "atomic" && operation.target) bases.add(operation.target.base);
    else if (operation.kind === "surface-read-store") {
      for (const ref of collectMemoryRefs(operation.target)) bases.add(ref.base);
    } else if (operation.kind === "inline-asm") {
      for (const output of operation.outputs) for (const ref of collectMemoryRefs(output)) bases.add(ref.base);
    } else if (operation.kind === "branch") {
      for (const base of semanticOperationWriteBases(operation.consequent)) bases.add(base);
      for (const base of semanticOperationWriteBases(operation.alternate)) bases.add(base);
    } else if (operation.kind === "loop" || operation.kind === "block") {
      for (const base of semanticOperationWriteBases(operation.body)) bases.add(base);
      if (operation.kind === "loop" && operation.continuing) {
        for (const base of semanticOperationWriteBases(operation.continuing)) bases.add(base);
      }
    }
  }
  return bases;
}

function semanticExpressionContainsUnsafeActiveLaneDeclarationCall(expression: SemanticExpression): boolean {
  let unsafe = false;
  walkSemanticExpression(expression, (item) => {
    if (item.kind !== "call") return;
    const name = semanticCallName(item.callee);
    if (name === undefined || !isSemanticMathCallName(name) && semanticIntrinsicReturnType(name, item.args) === undefined) unsafe = true;
  });
  return unsafe;
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
  const localDimensions = semanticLocalAddressDimensions(operations, functions);
  const calls = [
    ...collectSemanticFunctionCalls(operations),
    ...functions.flatMap((fn) => collectSemanticFunctionCalls(
      fn.body,
      new Set(fn.params.filter((param) => param.pointer && param.addressSpace === "local").map((param) => param.name)),
    )),
  ];
  return functions.map((fn) => {
    const fnCalls = calls.filter((call) => call.callee === fn.name);
    const localPointers = new Map<string, readonly number[]>();
    for (const [index, param] of fn.params.entries()) {
      if (!param.pointer || param.addressSpace !== "storage" || param.dimensions.length !== 0) continue;
      const refs = fnCalls.map((call) => semanticIrPointerArgumentMemoryRef(call.args[index]!));
      if (refs.length > 0 && refs.every((ref, callIndex) =>
        ref?.addressSpace === "local" &&
        ref.indices.length <= 1 &&
        fnCalls[callIndex]!.ownerLocalPointerNames.has(ref.base))) {
        const dimensions = refs.map((ref) => ref === undefined ? undefined : localDimensions.get(ref.baseId));
        const first = dimensions[0] ?? [];
        if (dimensions.every((candidate) => candidate !== undefined && sameDimensions(candidate, first))) {
          localPointers.set(param.name, first);
        }
      }
    }
    if (localPointers.size === 0) return fn;
    return {
      ...fn,
      params: fn.params.map((param) => {
        const dimensions = localPointers.get(param.name);
        return dimensions === undefined ? param : { ...param, addressSpace: "local" as const, dimensions };
      }),
      body: rewriteSemanticPointerAddressSpace(
        fn.body,
        new Map(fn.params
          .filter((param) => localPointers.has(param.name))
          .map((param) => [param.name, { name: param.name, id: semanticMemoryIdFromSymbol(param.id) }] as const)),
        "local",
      ),
    };
  });
}

function semanticLocalAddressDimensions(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
): ReadonlyMap<SemanticMemoryId, readonly number[]> {
  const dimensions = new Map<SemanticMemoryId, readonly number[]>();
  const collect = (items: readonly SemanticKernelIrOperation[]): void => {
    for (const operation of items) {
      if (operation.kind === "declare" && operation.target.addressSpace === "local") {
        dimensions.set(semanticMemoryIdFromSymbol(operation.target.id), operation.target.dimensions);
      }
      if (operation.kind === "block") collect(operation.body);
      if (operation.kind === "branch") {
        collect(operation.consequent);
        collect(operation.alternate);
      }
      if (operation.kind === "loop") {
        if (operation.init && isSemanticKernelIrOperation(operation.init)) collect([operation.init]);
        collect(operation.body);
        if (operation.continuing) collect(operation.continuing);
      }
    }
  };
  collect(operations);
  for (const fn of functions) {
    for (const param of fn.params) {
      if (param.pointer && param.addressSpace === "local") dimensions.set(semanticMemoryIdFromSymbol(param.id), param.dimensions);
    }
    collect(fn.body);
  }
  return dimensions;
}

function sameDimensions(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
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
    const roots = new Map<string, SemanticPointerRewriteTarget>();
    for (const [index, param] of fn.params.entries()) {
      if (!param.pointer || !param.constant || param.addressSpace !== "storage") continue;
      const refs = fnCalls.map((call) => semanticIrPointerArgumentMemoryRef(call.args[index]!));
      const root = refs[0];
      if (root && refs.length > 0 && refs.every((ref) =>
        ref?.addressSpace === "constant" &&
        semanticIdsEqual(ref.baseId, root.baseId) &&
        (ref.indices.length === 0 || ref.indices.length === 1 && isSemanticZeroLiteral(ref.indices[0]))
      )) {
        roots.set(param.name, { name: root.base, id: root.baseId });
      }
    }
    if (roots.size === 0) return fn;
    return {
      ...fn,
      params: fn.params.map((param) => roots.has(param.name) ? {
        ...param,
        addressSpace: "constant" as const,
        pointerMemoryAlias: roots.get(param.name)!.id,
      } : param),
      body: rewriteSemanticPointerAddressSpace(fn.body, roots, "constant"),
    };
  });
}

function cloneMixedLocalPointerFunctionOverloads(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
): readonly CudaLiteSemanticFunction[] {
  const localDimensions = semanticLocalAddressDimensions(operations, functions);
  const calls = [
    ...collectSemanticFunctionCalls(operations),
    ...functions.flatMap((fn) => collectSemanticFunctionCalls(
      fn.body,
      new Set(fn.params.filter((param) => param.pointer && param.addressSpace === "local").map((param) => param.name)),
    )),
  ];
  return functions.flatMap((fn) => {
    const pointerIndexes = fn.params.flatMap((param, index) =>
      param.pointer && param.addressSpace === "storage" ? [index] : []
    );
    if (pointerIndexes.length === 0) return [fn];
    const signatures = new Map<string, SemanticFunctionCallSite[]>();
    for (const call of calls.filter((candidate) => candidate.callee === fn.name)) {
      const refs = pointerIndexes.map((index) => semanticIrPointerArgumentMemoryRef(call.args[index]!));
      if (refs.some((ref) => ref === undefined ||
        ref.addressSpace !== "storage" && ref.addressSpace !== "device-global" && ref.addressSpace !== "local")) continue;
      const signature = refs.map((ref) => ref!.addressSpace === "local" ? "local" : "storage").join(",");
      const group = signatures.get(signature) ?? [];
      group.push(call);
      signatures.set(signature, group);
    }
    if (signatures.size <= 1 || ![...signatures].some(([signature]) => signature.includes("local"))) return [fn];
    return [...signatures].flatMap(([signature, signatureCalls]) => {
      const addressSpaces = signature.split(",");
      if (!addressSpaces.includes("local")) return [fn];
      const rewrites = new Map<string, SemanticPointerRewriteTarget>();
      const params = [...fn.params];
      for (const [pointerPosition, paramIndex] of pointerIndexes.entries()) {
        if (addressSpaces[pointerPosition] !== "local") continue;
        const param = fn.params[paramIndex]!;
        const refs = signatureCalls.map((call) => semanticIrPointerArgumentMemoryRef(call.args[paramIndex]!)!);
        const dimensions = refs.map((ref) => localDimensions.get(ref.baseId));
        if (!dimensions.every((value) => value !== undefined && value.length <= 1) ||
          !sameSemanticDimensions(dimensions as readonly (readonly number[])[])) return [];
        const name = `${param.name}__bg_local_ptr`;
        const id = createGeneratedSemanticSymbolId(
          `bg_local_overload_${fn.name}_${signature}_${param.name}_${param.span.start}`,
          param.span,
        );
        params[paramIndex] = { ...param, id, name, addressSpace: "local", dimensions: dimensions[0]! };
        rewrites.set(param.name, { name, id: semanticMemoryIdFromSymbol(id) });
      }
      return [{ ...fn, params, body: rewriteSemanticPointerAddressSpace(fn.body, rewrites, "local") }];
    });
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
        id: createGeneratedSemanticSymbolId(`bg_param_local_${sourceName}_${param.span.start}`, param.span),
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

function cloneMixedSharedPointerFunctionOverloads(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
  sharedMemoryDimensions: ReadonlyMap<string, readonly number[]>,
  sharedMemoryValueTypes: ReadonlyMap<string, CudaLiteScalarType | undefined>,
): readonly CudaLiteSemanticFunction[] {
  const calls = [
    ...collectSemanticFunctionCalls(operations),
    ...functions.flatMap((fn) => collectSemanticFunctionCalls(fn.body)),
  ];
  return functions.flatMap((fn) => {
    const pointerIndexes = fn.params.flatMap((param, index) =>
      param.pointer && param.addressSpace === "storage" ? [index] : []
    );
    if (pointerIndexes.length === 0) return [fn];
    const signatures = new Map<string, SemanticFunctionCallSite[]>();
    for (const call of calls.filter((candidate) => candidate.callee === fn.name)) {
      const refs = pointerIndexes.map((index) => semanticIrPointerArgumentMemoryRef(call.args[index]!));
      if (refs.some((ref) => ref === undefined || ref.addressSpace !== "storage" && ref.addressSpace !== "device-global" && ref.addressSpace !== "shared")) continue;
      const signature = refs.map((ref) => ref!.addressSpace === "shared" ? "shared" : "storage").join(",");
      const group = signatures.get(signature) ?? [];
      group.push(call);
      signatures.set(signature, group);
    }
    if (signatures.size <= 1 || ![...signatures].some(([signature]) => signature.includes("shared"))) return [fn];
    return [...signatures].flatMap(([signature, signatureCalls]) => {
      const addressSpaces = signature.split(",");
      if (!addressSpaces.includes("shared")) return [fn];
      const rewrites = new Map<string, SemanticPointerRewriteTarget>();
      const params = [...fn.params];
      for (const [pointerPosition, paramIndex] of pointerIndexes.entries()) {
        if (addressSpaces[pointerPosition] !== "shared") continue;
        const param = fn.params[paramIndex]!;
        const refs = signatureCalls.map((call) => semanticIrPointerArgumentMemoryRef(call.args[paramIndex]!)!);
        const roots = refs.map((ref) => ref.base);
        const dimensions = roots.map((root) => sharedMemoryDimensions.get(root));
        const carriers = roots.map((root) => sharedMemoryValueTypes.get(root));
        const compatible = dimensions.every((value) => value !== undefined && value.length <= 1) &&
          sameSemanticDimensions(dimensions as readonly (readonly number[])[]) &&
          carriers.every((value) => value !== undefined && param.valueType !== undefined && sizeofCudaType(value) === sizeofCudaType(param.valueType));
        if (!compatible) return [];
        const name = `${param.name}__bg_shared_ptr`;
        const id = createGeneratedSemanticSymbolId(
          `bg_shared_overload_${fn.name}_${signature}_${param.name}_${param.span.start}`,
          param.span,
        );
        params[paramIndex] = {
          ...param,
          id,
          name,
          addressSpace: "shared",
          dimensions: dimensions[0]!,
          ...optionalPointerCarrierValueType(carriers[0]),
        };
        rewrites.set(param.name, { name, id: semanticMemoryIdFromSymbol(id) });
      }
      return [{
        ...fn,
        params,
        body: rewriteSemanticPointerAddressSpace(fn.body, rewrites),
      }];
    });
  });
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
      const callArgs = fnCalls.map((call) => call.args[index]).filter((arg): arg is SemanticExpression => arg !== undefined);
      const refs = callArgs.map(semanticIrPointerArgumentMemoryRef);
      const args = callArgs.map(sharedPointerRoot);
      const dimensions = args.map((root) => root === undefined ? undefined : sharedMemoryDimensions.get(root));
      const carrierTypes = args.map((root) => root === undefined ? undefined : sharedMemoryValueTypes.get(root));
      const matchingValueTypes = param.valueType !== undefined && carrierTypes.every((valueType, argIndex) =>
        valueType !== undefined && (
          valueType === param.valueType ||
          sizeofCudaType(valueType) === sizeofCudaType(param.valueType!) ||
          refs[argIndex]?.pointerBaseIsScalarLane === true && cudaVectorScalarType(valueType) === param.valueType ||
          isCudaVectorType(param.valueType) && cudaVectorScalarType(param.valueType) === valueType
        ),
      );
      const effectiveDimensions = dimensions.map((item, argIndex) => {
        const carrierType = carrierTypes[argIndex];
        if (item === undefined) return undefined;
        if (item.length === 0) return [];
        const flatExtent = totalElements(item);
        return refs[argIndex]?.pointerBaseIsScalarLane === true && isCudaVectorType(carrierType)
          ? [flatExtent * cudaVectorLaneCount(carrierType)]
          : [flatExtent];
      });
      if (args.length > 0 && args.every((root) => root !== undefined) && matchingValueTypes && effectiveDimensions.every((item) => item !== undefined) && sameSemanticDimensions(effectiveDimensions as readonly (readonly number[])[])) {
        sharedPointerNames.set(param.name, `${param.name}__bg_shared_ptr`);
        sharedPointerDimensions.set(param.name, effectiveDimensions[0]!);
        sharedPointerRoots.set(param.name, args);
      }
    }
    if (sharedPointerNames.size === 0) return fn;
    const sharedPointerAliases = new Map<string, SemanticSymbolId>();
    const specializedParams = fn.params.filter((param) => sharedPointerNames.has(param.name));
    for (const [index, param] of specializedParams.entries()) {
      const roots = sharedPointerRoots.get(param.name)!;
      const canonical = specializedParams.slice(0, index).find((candidate) =>
        sameSemanticPointerRoots(roots, sharedPointerRoots.get(candidate.name)!),
      );
      if (canonical) sharedPointerAliases.set(param.name, canonical.id);
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
        ...(sharedPointerAliases.has(param.name) ? { pointerParamAlias: sharedPointerAliases.get(param.name)! } : {}),
      } : param),
      body: rewriteSemanticPointerAddressSpace(
        fn.body,
        new Map(fn.params
          .filter((param) => sharedPointerNames.has(param.name))
          .map((param) => [param.name, {
            name: sharedPointerNames.get(param.name)!,
            id: semanticMemoryIdFromSymbol(param.id),
          }] as const)),
      ),
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
  const localAddressNames = new Set(ownerLocalPointerNames);
  collectSemanticLocalAddressNames(operations, localAddressNames);
  const out: SemanticFunctionCallSite[] = [];
  collectSemanticOperationFunctionCalls(operations, out, localAddressNames);
  walkSemanticOperations(operations, (expression) => {
    if (expression.kind === "call" && expression.callee.kind === "symbol") {
      out.push({ callee: expression.callee.name, args: expression.args, ownerLocalPointerNames: localAddressNames });
    }
  });
  return out;
}

function collectSemanticLocalAddressNames(
  operations: readonly SemanticKernelIrOperation[],
  names: Set<string>,
): void {
  for (const operation of operations) {
    if (operation.kind === "declare" && operation.target.addressSpace === "local" &&
      (!operation.target.pointer || operation.target.pointerAddressSpace === "local")) names.add(operation.target.name);
    if (operation.kind === "block") collectSemanticLocalAddressNames(operation.body, names);
    if (operation.kind === "branch") {
      collectSemanticLocalAddressNames(operation.consequent, names);
      collectSemanticLocalAddressNames(operation.alternate, names);
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) collectSemanticLocalAddressNames([operation.init], names);
      collectSemanticLocalAddressNames(operation.body, names);
      if (operation.continuing) collectSemanticLocalAddressNames(operation.continuing, names);
    }
  }
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
    if (operation.kind === "loop" || operation.kind === "block") {
      collectSemanticOperationFunctionCalls(operation.body, out, ownerLocalPointerNames);
      if (operation.kind === "loop" && operation.continuing) collectSemanticOperationFunctionCalls(operation.continuing, out, ownerLocalPointerNames);
    }
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

interface SemanticPointerRewriteTarget {
  readonly name: string;
  readonly id: SemanticMemoryId;
}

function rewriteSemanticPointerAddressSpace(
  operations: readonly SemanticKernelIrOperation[],
  names: ReadonlyMap<string, SemanticPointerRewriteTarget>,
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
    if (operation.kind === "loop") return { ...operation, ...(operation.init === undefined ? {} : { init: isSemanticKernelIrOperation(operation.init) ? rewriteSemanticPointerAddressSpace([operation.init], names, addressSpace)[0]! : rewriteSemanticExpressionAddressSpace(operation.init, names, addressSpace) }), ...(operation.condition === undefined ? {} : { condition: rewriteSemanticExpressionAddressSpace(operation.condition, names, addressSpace) }), ...(operation.update === undefined ? {} : { update: rewriteSemanticExpressionAddressSpace(operation.update, names, addressSpace) }), body: rewriteSemanticPointerAddressSpace(operation.body, names, addressSpace), ...(operation.continuing === undefined ? {} : { continuing: rewriteSemanticPointerAddressSpace(operation.continuing, names, addressSpace) }) };
    if (operation.kind === "block") return { ...operation, body: rewriteSemanticPointerAddressSpace(operation.body, names, addressSpace) };
    if (operation.kind === "return" && operation.value) return { ...operation, value: rewriteSemanticExpressionAddressSpace(operation.value, names, addressSpace) };
    return operation;
  });
}

function rewriteSemanticMemoryRef(ref: SemanticMemoryRef, names: ReadonlyMap<string, SemanticPointerRewriteTarget>, addressSpace: "shared" | "constant" | "local"): SemanticMemoryRef {
  const target = names.get(ref.base);
  return {
    ...ref,
    ...(target !== undefined && ref.addressSpace === "storage" ? { baseId: target.id, base: target.name, addressSpace } : {}),
    indices: ref.indices.map((index) => rewriteSemanticExpressionAddressSpace(index, names, addressSpace)),
  };
}

function rewriteSemanticExpressionAddressSpace(expression: SemanticExpression, names: ReadonlyMap<string, SemanticPointerRewriteTarget>, addressSpace: "shared" | "constant" | "local"): SemanticExpression {
  switch (expression.kind) {
    case "pointer-valid": {
      const target = names.get(expression.pointer);
      return target === undefined
        ? expression
        : { ...expression, pointerId: semanticSymbolIdFromMemory(target.id), pointer: target.name };
    }
    case "symbol": {
      const target = names.get(expression.name);
      return target !== undefined && expression.addressSpace === "storage"
        ? { ...expression, id: semanticSymbolIdFromMemory(target.id), name: target.name, addressSpace }
        : expression;
    }
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

function collectReachableSemanticFunctionIds(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
): ReadonlySet<string> {
  const ids = new Set<string>();
  const pending = semanticReferencedFunctions(operations, functions);
  for (let index = 0; index < pending.length; index++) {
    const fn = pending[index]!;
    if (ids.has(fn.id.key)) continue;
    ids.add(fn.id.key);
    pending.push(...semanticReferencedFunctions(fn.body, functions));
  }
  return ids;
}

function semanticReferencedFunctions(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
): CudaLiteSemanticFunction[] {
  const names = new Set(collectSemanticFunctionCalls(operations).map((call) => call.callee));
  const ids = new Set<string>();
  walkSemanticOperations(operations, (expression) => {
    if (expression.kind === "symbol" && expression.addressSpace === "function") ids.add(expression.id.key);
  });
  return functions.filter((fn) => names.has(fn.name) || ids.has(fn.id.key));
}

function lowerStatements(
  statements: readonly CudaLiteStatement[],
  parentScope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] {
  const scope = cloneSemanticScope(parentScope);
  return lowerStatementsWithScope(statements, scope);
}

function lowerStatementsWithScope(
  statements: readonly CudaLiteStatement[],
  scope: Map<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] {
  const out: SemanticKernelIrOperation[] = [];
  for (let index = 0; index < statements.length; index++) {
    const statement = statements[index]!;
    if (isLocalPointerArray(statement) && hasLaterDynamicStoragePointerArrayAssignment(statement.name, statements.slice(index + 1), scope)) {
      const original = symbolForVar(statement, scope);
      const target: CudaLiteSemanticSymbol = {
        ...semanticSymbolWithoutPointerAlias(original),
        pointerRuntimeState: true,
      };
      scope.set(target.name, target);
      out.push({ kind: "declare", target, span: statement.span });
      continue;
    }
    if (isLocalPointerAliasPlaceholder(statement) && hasLaterLocalPointerAliasAssignment(statement.name, statements.slice(index + 1), scope)) {
      const original = symbolForVar(statement, scope);
      const assignmentProfile = localPointerAliasAssignmentProfile(statement.name, statements.slice(index + 1), scope);
      const target = assignmentProfile.total > 1 && assignmentProfile.controlDependent
        ? { ...semanticSymbolWithoutPointerAlias(original), pointerRuntimeState: true }
        : original;
      scope.set(target.name, target);
      out.push({ kind: "declare", target, span: statement.span });
      continue;
    }
    out.push(...lowerStatementOperations(statement, scope, statements.slice(index + 1)));
  }
  return out.map((operation) => {
    if (operation.kind !== "declare" || !operation.target.pointer) return operation;
    const target = scope.get(operation.target.name);
    if (target?.pointerRuntimeState === true && sameSymbolDeclaration(operation.target, target)) {
      return { ...operation, target };
    }
    if (operation.target.dimensions.length !== 1) return operation;
    if (!semanticPointerArrayAliasesComplete(target)) return operation;
    return { ...operation, target: target! };
  });
}

function bindSemanticSymbol(
  scope: Map<string, CudaLiteSemanticSymbol>,
  sourceName: string,
  symbol: CudaLiteSemanticSymbol,
): void {
  const shadowed = scope.get(sourceName);
  if (shadowed !== undefined && shadowed.id.key !== symbol.id.key) {
    scope.set(`__bg_identity:${shadowed.id.key}`, shadowed);
  }
  scope.set(sourceName, symbol);
}

function lowerStatementOperations(
  statement: CudaLiteStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
  followingStatements: readonly CudaLiteStatement[],
): readonly SemanticKernelIrOperation[] {
  const poolAllocation = semanticPoolAllocationVarInitOperation(statement, scope);
  if (poolAllocation) return poolAllocation;
  const pointerSideEffectInit = semanticPointerSideEffectVarInitOperations(statement, scope);
  if (pointerSideEffectInit) return pointerSideEffectInit;
  const pointerAssignmentChain = semanticLocalPointerAssignmentChainOperations(statement, scope);
  if (pointerAssignmentChain) return pointerAssignmentChain;
  const dynamicPointerArrayAssignment = semanticDynamicPointerArrayAliasAssignmentOperations(statement, scope);
  if (dynamicPointerArrayAssignment) return dynamicPointerArrayAssignment;
  const chainedStores = semanticMemoryAssignmentChainOperations(statement, scope);
  if (chainedStores) return chainedStores;
  const conditionalAssignmentTarget = semanticConditionalAssignmentTargetOperations(statement, scope);
  if (conditionalAssignmentTarget) return conditionalAssignmentTarget;
  const conditionalAssignment = semanticConditionalLocalAssignmentOperations(statement, scope);
  if (conditionalAssignment) return conditionalAssignment;
  const conditionalPointerInit = semanticConditionalPointerVarInitOperations(statement, scope, followingStatements);
  if (conditionalPointerInit) return conditionalPointerInit;
  const conditionalPointerOffsetInit = semanticConditionalPointerOffsetVarInitOperations(statement, scope);
  if (conditionalPointerOffsetInit) return conditionalPointerOffsetInit;
  const conditionalVarInit = semanticConditionalVarInitOperations(statement, scope);
  if (conditionalVarInit) return conditionalVarInit;
  const conditionalReturn = semanticConditionalReturnOperations(statement, scope);
  if (conditionalReturn) return conditionalReturn;
  const conditionalCallArgs = semanticConditionalCallArgumentOperations(statement, scope);
  if (conditionalCallArgs) return conditionalCallArgs;
  const mathOutVarDecl = semanticMathOutVarDeclOperations(statement, scope);
  const mathOutAssignment = semanticMathOutAssignmentOperations(statement, scope);
  const mathOutCall = semanticMathOutCallStatementOperations(statement, scope);
  return mathOutVarDecl ?? mathOutAssignment ?? mathOutCall ?? [lowerStatement(statement, scope)];
}

function semanticPoolAllocationVarInitOperation(
  statement: CudaLiteStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (statement.kind !== "var" || statement.storage !== "local" || !statement.pointer || statement.dimensions.length !== 0 || !statement.init) {
    return undefined;
  }
  const call = cudaPoolAllocationCall(statement.init);
  if (!call || call.callee.kind !== "identifier" || (call.callee.name !== "deviceAllocate" && call.callee.name !== "streamOrderedAllocate")) {
    return undefined;
  }
  const sizeArg = call.args.at(-1);
  if (!sizeArg) return undefined;
  const pool = semanticPoolRefForAllocation(call, scope);
  if (!pool) return undefined;
  const original = symbolForVar(statement, scope);
  const { init: _init, ...withoutInit } = semanticSymbolWithoutPointerAlias(original);
  const target: CudaLiteSemanticSymbol = {
    ...withoutInit,
    pointerRuntimeState: true,
  };
  scope.set(target.name, target);
  return [
    { kind: "declare", target, span: statement.span },
    {
      kind: "pool-allocate",
      allocator: call.callee.name,
      target,
      pool,
      sizeBytes: lowerExpression(sizeArg, scope),
      span: statement.span,
    },
  ];
}

function cudaPoolAllocationCall(expression: CudaLiteExpression): CudaLiteCallExpression | undefined {
  if (expression.kind === "call") return expression;
  if (expression.kind === "cast" && expression.pointer) return cudaPoolAllocationCall(expression.expression);
  return undefined;
}

function semanticPoolRefForAllocation(
  call: CudaLiteCallExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticPoolRef | undefined {
  if (call.args.length === 2) {
    const expression = call.args[0];
    const poolName = expression?.kind === "identifier"
      ? expression.name
      : expression?.kind === "unary" && expression.operator === "&" && expression.argument.kind === "identifier"
        ? expression.argument.name
        : undefined;
    const pool = poolName === undefined ? undefined : scope.get(poolName);
    if (!pool || pool.addressSpace !== "pool") return undefined;
    return {
      kind: "device-pool",
      id: semanticMemoryIdFromSymbol(pool.id),
      name: pool.name,
      span: expression!.span,
    };
  }
  if (call.args.length !== 4) return undefined;
  const dataArg = call.args[0];
  const offsetArg = call.args[1];
  const capacityArg = call.args[2];
  if (!dataArg || !offsetArg || !capacityArg) return undefined;
  const data = semanticPointerArgumentMemoryRef(dataArg, scope);
  const offset = semanticPointerArgumentMemoryRef(offsetArg, scope);
  if (!data || !offset) return undefined;
  return {
    kind: "raw-pool",
    data,
    offset,
    capacityBytes: lowerExpression(capacityArg, scope),
    span: call.span,
  };
}

function semanticPointerSideEffectVarInitOperations(
  statement: CudaLiteStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (
    statement.kind !== "var" ||
    statement.storage !== "local" ||
    !statement.pointer ||
    statement.dimensions.length !== 0 ||
    (statement.init?.kind !== "assignment" && statement.init?.kind !== "sequence")
  ) return undefined;
  const alias = applyUnconditionalPointerAliasInitializer(statement.init, scope);
  if (!alias || !semanticPointerAliasComplete(alias)) return undefined;
  const original = symbolForVar(statement, scope);
  const { init: _init, ...withoutInit } = semanticSymbolWithoutPointerAlias(original);
  const target: CudaLiteSemanticSymbol = { ...withoutInit, ...alias };
  scope.set(target.name, target);
  return [{ kind: "declare", target, span: statement.span }];
}

function applyUnconditionalPointerAliasInitializer(
  expression: CudaLiteExpression,
  scope: Map<string, CudaLiteSemanticSymbol>,
): SemanticPointerAlias | undefined {
  if (expression.kind === "sequence") {
    let alias: SemanticPointerAlias | undefined;
    for (const item of expression.expressions) {
      alias = applyUnconditionalPointerAliasInitializer(item, scope);
      if (!alias) return undefined;
    }
    return alias;
  }
  if (expression.kind === "assignment" && expression.operator === "=" && expression.left.kind === "identifier") {
    const target = scope.get(expression.left.name);
    if (!target || target.kind !== "local" || !target.pointer || target.dimensions.length !== 0) return undefined;
    const alias = applyUnconditionalPointerAliasInitializer(expression.right, scope);
    if (!alias || !semanticPointerAliasComplete(alias)) return undefined;
    scope.set(target.name, { ...semanticSymbolWithoutPointerAlias(target), ...alias });
    return alias;
  }
  return localPointerAliasForInitializer(expression, scope);
}

function semanticLocalPointerAssignmentChainOperations(
  statement: CudaLiteStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (statement.kind !== "expr") return undefined;
  const targets: CudaLiteSemanticSymbol[] = [];
  let source = statement.expression;
  while (source.kind === "assignment" && source.operator === "=" && source.left.kind === "identifier") {
    const target = scope.get(source.left.name);
    if (!target || target.kind !== "local" || !target.pointer || target.dimensions.length !== 0) return undefined;
    targets.push(target);
    source = source.right;
  }
  if (targets.length < 2) return undefined;
  const alias = localPointerAliasForInitializer(source, scope);
  if (!alias?.pointerRoot || alias.pointerAddressSpace !== "storage" || alias.pointerBaseIndices?.length !== 1) return undefined;
  const operations: SemanticKernelIrOperation[] = [];
  for (let index = targets.length - 1; index >= 0; index--) {
    const original = targets[index]!;
    const target: CudaLiteSemanticSymbol = {
      ...semanticSymbolWithoutPointerAlias(original),
      pointerRuntimeState: true,
    };
    scope.set(target.name, target);
    const rebind = semanticLocalPointerRebindFromSource(target, source, scope, statement.span);
    if (!rebind) return undefined;
    operations.push(rebind);
  }
  return operations;
}

function semanticDynamicPointerArrayAliasAssignmentOperations(
  statement: CudaLiteStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (
    statement.kind !== "expr" ||
    statement.expression.kind !== "assignment" ||
    statement.expression.operator !== "=" ||
    statement.expression.left.kind !== "index" ||
    statement.expression.left.target.kind !== "identifier" ||
    staticPointerArrayIndex(statement.expression.left.index) !== undefined
  ) return undefined;
  const assignment = statement.expression as CudaLiteAssignmentExpression;
  const left = assignment.left as Extract<CudaLiteExpression, { readonly kind: "index" }>;
  const targetName = (left.target as Extract<CudaLiteExpression, { readonly kind: "identifier" }>).name;
  const target = scope.get(targetName);
  if (!semanticPointerArrayAliasesComplete(target)) return undefined;
  const replacement = localPointerAliasForInitializer(assignment.right, scope);
  if (!replacement || !semanticStoragePointerAlias(replacement)) return undefined;
  const loweredIndex = lowerExpression(left.index, scope);
  const indexType = expressionValueType(loweredIndex);
  const materialized = semanticExpressionContainsCall(loweredIndex) && indexType && indexType !== "void" && !isCudaVectorType(indexType)
    ? materializeSemanticExpressionOnce({ operations: [], expression: loweredIndex }, indexType, "__bg.pointer.array.index", left.index.span)
    : { operations: [] as readonly SemanticKernelIrOperation[], expression: loweredIndex };
  const aliases = target!.pointerArrayAliases!.map((current, slot): SemanticPointerAlias => ({
    pointerSelection: {
      condition: {
        kind: "binary",
        operator: "==",
        left: materialized.expression,
        right: intNumberExpression(slot, left.index.span),
        valueType: "bool",
        span: left.index.span,
      },
      consequent: replacement,
      alternate: current!,
    },
  }));
  scope.set(target!.name, { ...target!, pointerArrayAliases: aliases });
  const rebinds = semanticPointerArrayRebindOperationsFromAlias(target!, materialized.expression, replacement, scope, statement.span);
  return rebinds === undefined
    ? undefined
    : [...materialized.operations, ...rebinds];
}

function semanticConditionalReturnOperations(
  statement: CudaLiteStatement,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (statement.kind !== "return" || statement.value === undefined) return undefined;
  const materialized = materializeConditionalCalls(lowerExpression(statement.value, scope));
  if (materialized.operations.length === 0) return undefined;
  return [
    ...materialized.operations,
    { kind: "return", value: materialized.expression, span: statement.span },
  ];
}

function semanticConditionalAssignmentTargetOperations(
  statement: CudaLiteStatement,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (statement.kind !== "expr" || statement.expression.kind !== "assignment") return undefined;
  const expression = lowerExpression(statement.expression, scope);
  if (expression.kind !== "assignment") return undefined;
  const target = materializeConditionalCalls(expression.target);
  if (target.operations.length === 0) return undefined;
  const ref = semanticMatrixLaneMemoryRef(target.expression, scope) ?? memoryRefFromExpression(target.expression);
  if (!ref) return undefined;
  const value = semanticSequencedAssignmentValue(expression.value);
  return [
    ...target.operations,
    {
      kind: "store",
      target: ref,
      value,
      operator: expression.operator,
      reads: collectMemoryRefs(value),
      span: statement.span,
    },
  ];
}

function semanticConditionalCallArgumentOperations(
  statement: CudaLiteStatement,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (statement.kind !== "expr" || statement.expression.kind !== "call") return undefined;
  const call = lowerExpression(statement.expression, scope);
  if (call.kind !== "call" || call.callee.kind !== "symbol") return undefined;
  const operations: SemanticKernelIrOperation[] = [];
  const args = call.args.map((arg) => {
    const materialized = materializeConditionalCalls(arg);
    operations.push(...materialized.operations);
    return materialized.expression;
  });
  if (operations.length === 0) return undefined;
  const materializedCall: Extract<SemanticExpression, { readonly kind: "call" }> = { ...call, args };
  if (semanticAtomicOperation(call.callee.name) !== undefined) {
    const target = atomicTargetFromCall(materializedCall);
    return [
      ...operations,
      {
        kind: "atomic",
        callee: call.callee.name,
        ...(target === undefined ? {} : { target }),
        args,
        span: statement.span,
      },
    ];
  }
  return [
    ...operations,
    {
      kind: "call",
      calleeId: call.callee.id,
      callee: call.callee.name,
      args,
      reads: args.flatMap((arg) => collectMemoryRefs(arg)),
      span: statement.span,
    },
  ];
}

function semanticConditionalVarInitOperations(
  statement: CudaLiteStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (statement.kind !== "var" || statement.init === undefined) return undefined;
  const target = symbolForVar(statement, scope);
  if (target.pointer || target.dimensions.length > 0) return undefined;
  const initScope = cloneSemanticScope(scope).set(target.name, target);
  const materialized = materializeConditionalCalls(lowerExpression(statement.init, initScope));
  if (materialized.operations.length === 0) return undefined;
  scope.set(target.name, target);
  const targetExpression = semanticSymbolExpression(target, statement.span);
  return [
    { kind: "declare", target, span: statement.span },
    ...materialized.operations,
    {
      kind: "expression",
      expression: {
        kind: "assignment",
        operator: "=",
        target: targetExpression,
        value: materialized.expression,
        valueType: requiredSemanticValueType(target.valueType, `conditional initializer '${target.name}'`, statement.span),
        span: statement.span,
      },
      span: statement.span,
    },
  ];
}

function semanticConditionalPointerOffsetVarInitOperations(
  statement: CudaLiteStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (statement.kind !== "var" || statement.storage !== "local" || !statement.pointer ||
    statement.dimensions.length !== 0 || statement.init === undefined) return undefined;
  const original = symbolForVar(statement, scope);
  if (original.pointerRoot === undefined || original.pointerAddressSpace !== "storage" || original.pointerBaseIndices?.length !== 1) return undefined;
  const materialized = materializeConditionalCalls(original.pointerBaseIndices[0]!);
  if (materialized.operations.length === 0) return undefined;
  const root = semanticSymbolForMemoryId(scope, original.pointerRoot);
  const valueType = original.valueType;
  if (!root || valueType === undefined || valueType === "void") return undefined;
  const init: SemanticExpression = {
    kind: "index",
    target: semanticSymbolExpression(root, statement.init.span),
    index: materialized.expression,
    valueType,
    addressSpace: "storage",
    span: statement.init.span,
  };
  const target: CudaLiteSemanticSymbol = {
    ...original,
    init,
    pointerBaseIndices: [materialized.expression],
  };
  scope.set(target.name, target);
  return [
    ...materialized.operations,
    { kind: "declare", target, init, span: statement.span },
  ];
}

function semanticConditionalLocalAssignmentOperations(
  statement: CudaLiteStatement,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] | undefined {
  if (
    statement.kind !== "expr" ||
    statement.expression.kind !== "assignment" ||
    statement.expression.right.kind !== "conditional"
  ) return undefined;
  const target = lowerExpression(statement.expression.left, scope);
  if (target.kind !== "symbol" || target.addressSpace !== "local") return undefined;
  const conditional = statement.expression.right;
  const assignment = (value: CudaLiteExpression): SemanticKernelIrOperation => ({
    kind: "expression",
    expression: {
      kind: "assignment",
      operator: statement.expression.kind === "assignment" ? statement.expression.operator : "=",
      target,
      value: lowerExpression(value, scope),
      valueType: requiredSemanticValueType(expressionValueType(target), "conditional assignment", statement.span),
      span: statement.span,
    },
    span: statement.span,
  });
  return [{
    kind: "branch",
    condition: lowerConditionExpression(conditional.condition, scope),
    consequent: [assignment(conditional.consequent)],
    alternate: [assignment(conditional.alternate)],
    span: statement.span,
  }];
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
  const conversionSource = op?.kind === "convert-f32-to-int"
    ? semanticInlineAsmF32Source(op.source ?? { kind: "operand", index: outputs.length }, statement, outputs, scope)
    : op?.kind === "convert-int-to-f32"
      ? semanticInlineAsmIntSource(op.source ?? { kind: "operand", index: outputs.length }, statement, outputs, scope, op.fromSigned)
      : undefined;
  const bitInput = op && (op.kind === "bfind-u32" || op.kind === "ffs-b32" || op.kind === "popc-b32" || op.kind === "clz-b32" || op.kind === "brev-b32")
    ? op.immediate === undefined
      ? statement.inputs.length === 1 ? lowerExpression(statement.inputs[0]!, scope) : undefined
      : statement.inputs.length === 0 ? semanticUintLiteralExpression(op.immediate, statement.span) : undefined
    : undefined;
  const integerValue = op === undefined ? undefined : semanticInlineAsmIntegerExpression(op, statement, scope, expressionValueType(target));
  const addressPredicate = op?.kind === "isspacep" && statement.inputs.length === 1
    ? semanticInlineAsmAddressPredicate(op.space, statement.inputs[0]!, scope, statement.span)
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
    : integerValue !== undefined
      ? integerValue
    : addressPredicate !== undefined
      ? addressPredicate
    : op?.kind === "convert-f32-to-int" && conversionSource !== undefined
      ? semanticCallExpression(
          `__float2${op.toSigned ? "int" : "uint"}_${op.rounding === "rm" ? "rd" : op.rounding === "rp" ? "ru" : op.rounding}`,
          [conversionSource],
          op.toSigned ? "int" : "uint",
          statement.span,
        )
    : op?.kind === "convert-int-to-f32" && conversionSource !== undefined
      ? semanticCallExpression(op.fromSigned ? "__int2float_rn" : "__uint2float_rn", [conversionSource], "float", statement.span)
    : op?.kind === "u8x4-sad-add" && statement.inputs.length === 3
      ? semanticCallExpression(
          "__usad4",
          statement.inputs.map((input) => castScalarExpression(lowerExpression(input, scope), "uint", statement.span)),
          "uint",
          statement.span,
        )
    : statement.inputs.length !== 0
      ? undefined
      : op?.kind === "globaltimer-u64"
        ? semanticCallExpression("clock64", [], "uint", statement.span)
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

function semanticInlineAsmAddressPredicate(
  expected: "global" | "shared" | "const" | "local",
  source: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticExpression {
  const actual = expressionAddressSpace(lowerExpression(source, scope));
  const matches = expected === "global"
    ? actual === "storage" || actual === "device-global"
    : expected === "shared"
      ? actual === "shared"
      : expected === "const"
        ? actual === "constant"
        : actual === "local";
  return intNumberExpression(matches ? 1 : 0, span);
}

function semanticInlineAsmIntegerExpression(
  op: InlineAsmOp,
  statement: CudaLiteAsmStatement,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  targetValueType: CudaLiteScalarType | undefined,
): SemanticExpression | undefined {
  const signed = op.kind === "convert-b32" ? op.toSigned : "signed" in op ? op.signed : false;
  const valueType: CudaLiteScalarType = signed ? "int" : "uint";
  const inputs = statement.inputs.map((input) => castScalarExpression(lowerExpression(input, scope), valueType, statement.span));
  const immediate = "immediate" in op && op.immediate !== undefined
    ? castScalarExpression(semanticUintLiteralExpression(op.immediate, statement.span), valueType, statement.span)
    : undefined;
  const operands = immediate === undefined ? inputs : [...inputs, immediate];
  const binary = (operator: string, left: SemanticExpression | undefined, right: SemanticExpression | undefined): SemanticExpression | undefined =>
    left === undefined || right === undefined ? undefined : { kind: "binary", operator, left, right, valueType, span: statement.span };
  switch (op.kind) {
    case "arithmetic-b32":
      return semanticInlineAsmIntegerCall(`__bg_ptx_arithmetic_${op.op.replace("-", "_")}`, operands, targetValueType, statement.span);
    case "bitwise-b32":
      if (op.op === "not") {
        const argument = operands[0];
        return argument === undefined ? undefined : { kind: "unary", operator: "~", argument, valueType, span: statement.span };
      }
      return binary(op.op === "and" ? "&" : op.op === "or" ? "|" : "^", operands[0], operands[1]);
    case "move-b32":
    case "convert-b32":
      return operands[0];
    case "shift-b32":
      return semanticInlineAsmIntegerCall(`__bg_ptx_shift_${op.op}_${op.signed ? "s" : "u"}`, operands, targetValueType, statement.span);
    case "minmax-b32":
      return semanticInlineAsmIntegerCall(`__bg_ptx_${op.op}_${op.signed ? "s" : "u"}`, operands, targetValueType, statement.span);
    case "unary-int-b32":
      return semanticInlineAsmIntegerCall(`__bg_ptx_${op.op}`, operands, targetValueType, statement.span);
    case "prmt-b32":
      return semanticInlineAsmIntegerCall(
        "__bg_ptx_prmt",
        op.selectorImmediate === undefined ? inputs : [...inputs, semanticUintLiteralExpression(op.selectorImmediate, statement.span)],
        targetValueType,
        statement.span,
      );
    case "lop3-b32": {
      let inputIndex = 0;
      const data = (op.dataImmediates ?? [undefined, undefined, undefined]).map((value) =>
        value === undefined ? inputs[inputIndex++] : semanticUintLiteralExpression(value, statement.span)
      );
      const lut = op.immLut === undefined ? inputs[inputIndex] : semanticUintLiteralExpression(op.immLut, statement.span);
      return lut === undefined || data.some((value) => value === undefined)
        ? undefined
        : semanticInlineAsmIntegerCall("__bg_ptx_lop3", [...data as SemanticExpression[], lut], targetValueType, statement.span);
    }
    case "select-b32": {
      let inputIndex = 0;
      const trueValue = op.trueImmediate === undefined ? inputs[inputIndex++] : semanticUintLiteralExpression(op.trueImmediate, statement.span);
      const falseValue = op.falseImmediate === undefined ? inputs[inputIndex++] : semanticUintLiteralExpression(op.falseImmediate, statement.span);
      const predicate = inputs[inputIndex];
      return trueValue === undefined || falseValue === undefined || predicate === undefined
        ? undefined
        : semanticInlineAsmIntegerCall("__bg_ptx_select", [trueValue, falseValue, predicate], targetValueType, statement.span);
    }
    case "compare-b32":
      return semanticInlineAsmIntegerCall(`__bg_ptx_compare_${op.op}_${op.signed ? "s" : "u"}`, operands, targetValueType, statement.span);
    default:
      return undefined;
  }
}

function semanticInlineAsmIntegerCall(
  name: string,
  args: readonly SemanticExpression[],
  valueType: CudaLiteScalarType | undefined,
  span: SourceSpan,
): SemanticExpression | undefined {
  if (args.length === 0 || valueType === undefined || valueType === "void") return undefined;
  return semanticCallExpression(name, args, valueType, span);
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

function semanticInlineAsmIntSource(
  source: InlineAsmIntSource,
  statement: CudaLiteAsmStatement,
  outputs: readonly CudaLiteExpression[],
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  signed: boolean,
): SemanticExpression | undefined {
  const valueType = signed ? "int" : "uint";
  if (source.kind === "immediate") {
    return castScalarExpression(semanticUintLiteralExpression(source.value, statement.span), valueType, statement.span);
  }
  const expression = source.index < outputs.length
    ? outputs[source.index]
    : statement.inputs[source.index - outputs.length];
  return expression === undefined
    ? undefined
    : castScalarExpression(lowerExpression(expression, scope), valueType, statement.span);
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
      id: createBuiltinSemanticSymbolId(objectName),
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
      const childScope = cloneSemanticScope(scope);
      const body = lowerStatementsWithScope(statement.body, childScope);
      mergeBlockLocalPointerAliases(scope, childScope);
      return { kind: "block", body, span: statement.span };
    }
    case "var": {
      const target = symbolForVar(statement, scope);
      bindSemanticSymbol(scope, target.name, target);
      return {
        kind: "declare",
        target,
        ...(statement.init === undefined ? {} : { init: lowerExpression(statement.init, scope) }),
        span: statement.span,
      };
    }
    case "dim3": {
      const target = semanticSymbolForDim3(statement.name, statement.span);
      scope.set(target.name, target);
      return { kind: "dim3-declare", target, args: statement.args.map((arg) => lowerExpression(arg, scope)), span: statement.span };
    }
    case "cooperative-group": {
      const target = semanticSymbolForCooperativeGroup(statement);
      scope.set(statement.name, target);
      return {
        kind: "cooperative-group-declare",
        declaration: {
          kind: "cooperative-group",
          id: target.id,
          groupKind: statement.groupKind,
          name: statement.name,
          ...(statement.tileSize === undefined ? {} : { tileSize: statement.tileSize }),
          ...(statement.partitionParent === undefined ? {} : { partitionParent: statement.partitionParent }),
          ...(statement.partitionPredicate === undefined ? {} : { partitionPredicate: lowerExpression(statement.partitionPredicate, scope) }),
          span: statement.span,
        },
        span: statement.span,
      };
    }
    case "kernel-launch":
      return { kind: "device-launch", launch: lowerDeviceLaunch(statement, scope), span: statement.span };
    case "asm": {
      const registerAssignment = lowerInlineAsmBuiltinRegisterAssignment(statement, scope);
      if (registerAssignment) return registerAssignment;
      const outputs = statement.outputs ?? (statement.output === undefined ? [] : [statement.output]);
      const op = classifyInlineAsm(statement.template);
      return {
        kind: "inline-asm",
        ...(op === undefined ? {} : { op }),
        outputs: outputs.map((output) => lowerExpression(output, scope)),
        inputs: statement.inputs.map((input) => lowerExpression(input, scope)),
        span: statement.span,
      };
    }
    case "expr": {
      const cpAsync = semanticCpAsyncOperation(statement.expression, scope, statement.span);
      if (cpAsync) return cpAsync;
      const matrixOperation = semanticMatrixOperation(statement.expression, scope, statement.span);
      if (matrixOperation) return matrixOperation;
      const localPointerRebind = semanticLocalPointerRebindOperation(statement.expression, scope);
      if (localPointerRebind) return localPointerRebind;
      const pointerRebase = semanticStoragePointerRebaseOperation(statement.expression, scope, statement.span);
      if (pointerRebase) return pointerRebase;
      const aliasAssignment = localPointerAliasUpdate(statement.expression, scope);
      if (aliasAssignment) return { kind: "expression", expression: zeroExpression(statement.span), span: statement.span };
      const pointerArrayAssignment = localPointerArrayAliasUpdate(statement.expression, scope);
      if (pointerArrayAssignment) return pointerArrayAssignment;
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
        const target = semanticMatrixLaneMemoryRef(expression.target, scope) ?? memoryRefFromExpression(expression.target);
        if (target) {
          const value = semanticSequencedAssignmentValue(expression.value);
          const reinterpretedCopy = semanticReinterpretedScalarCopy(target, value, expression.operator, scope, statement.span);
          if (reinterpretedCopy) return reinterpretedCopy;
          const conditionalCopy = semanticConditionalReinterpretedCopy(target, value, expression.operator, scope, statement.span);
          if (conditionalCopy) return conditionalCopy;
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
          const surface = semanticIndexedSurfaceLayer(
            expression.args[1]!,
            semanticSurfaceWriteUsesZ(expression.callee.name) ? expression.args[4] : undefined,
          );
          return {
            kind: "surface-write",
            value: expression.args[0]!,
            surface: surface.surface,
            xBytes: expression.args[2]!,
            y: expression.callee.name === "surf1Dwrite" ? zeroExpression(expression.span) : expression.args[3]!,
            ...(surface.z === undefined ? {} : { z: surface.z }),
            span: statement.span,
          };
        }
        if (
          (expression.callee.name === "surf1Dread" && expression.args.length === 3) ||
          (expression.callee.name === "surf2Dread" && expression.args.length === 4) ||
          ((expression.callee.name === "surf2DLayeredread" || expression.callee.name === "surf3Dread") && expression.args.length === 5)
        ) {
          const target = expression.args[0]!;
          const surface = semanticIndexedSurfaceLayer(expression.args[1]!, expression.args[4]);
          return {
            kind: "surface-read-store",
            target,
            surface: surface.surface,
            xBytes: expression.args[2]!,
            y: expression.callee.name === "surf1Dread" ? zeroExpression(expression.span) : expression.args[3]!,
            ...(surface.z === undefined ? {} : { z: surface.z }),
            ...optionalValueType(target.kind === "unary" && target.operator === "&" ? expressionValueType(target.argument) : expressionValueType(target)),
            span: statement.span,
          };
        }
        const target = atomicTargetFromCall(expression);
        if (semanticAtomicOperation(expression.callee.name) !== undefined) {
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
          calleeId: expression.callee.id,
          callee: expression.callee.name,
          args: expression.args,
          reads: expression.args.flatMap((arg) => collectMemoryRefs(arg)),
          span: statement.span,
        };
      }
      return { kind: "expression", expression, span: statement.span };
    }
    case "if": {
      markBranchPointerRuntimeState(statement.consequent, statement.alternate ?? [], scope);
      const loweredCondition = lowerConditionExpression(statement.condition, scope);
      const materializedCondition = materializeConditionalCalls(loweredCondition);
      const condition = materializedCondition.expression;
      const constantCondition = staticNumberValue(condition);
      if (constantCondition !== undefined && materializedCondition.operations.length === 0) {
        const selectedScope = cloneSemanticScope(scope);
        const body = lowerStatementsWithScope(constantCondition !== 0 ? statement.consequent : statement.alternate ?? [], selectedScope);
        mergeBlockLocalPointerAliases(scope, selectedScope);
        return { kind: "block", body, span: statement.span };
      }
      const consequentScope = cloneSemanticScope(scope);
      const alternateScope = cloneSemanticScope(scope);
      const consequent = lowerStatementsWithScope(statement.consequent, consequentScope);
      const alternate = lowerStatementsWithScope(statement.alternate ?? [], alternateScope);
      mergeBranchLocalPointerAliases(scope, consequentScope, alternateScope, condition, statement.span);
      const branch: SemanticKernelIrOperation = {
        kind: "branch",
        condition,
        consequent,
        alternate,
        span: statement.span,
      };
      return materializedCondition.operations.length === 0
        ? branch
        : { kind: "block", body: [...materializedCondition.operations, branch], span: statement.span };
    }
    case "for":
      {
        markLoopUpdatePointerRuntimeState(statement.update, scope);
        const loopScope = cloneSemanticScope(scope);
        const init = statement.init?.kind === "var"
          ? lowerForInitStatement(statement.init, loopScope)
          : statement.init
          ? lowerExpression(statement.init, loopScope)
          : undefined;
        const loweredCondition = statement.condition === undefined
          ? undefined
          : materializeConditionalCalls(lowerConditionExpression(statement.condition, loopScope));
        const body = lowerStatements(statement.body, loopScope);
        const continuing = statement.update !== undefined && loopUpdateRequiresOperations(statement.update, loopScope)
          ? lowerLoopUpdateOperations(statement.update, loopScope)
          : undefined;
        const materializedBody = loweredCondition && loweredCondition.operations.length > 0
          ? semanticMaterializedLoopBody(loweredCondition, body, statement.span)
          : body;
        return {
          kind: "loop",
          loopKind: "for",
          ...(init === undefined ? {} : { init }),
          ...(loweredCondition === undefined || loweredCondition.operations.length > 0 ? {} : { condition: loweredCondition.expression }),
          ...(statement.update === undefined || continuing !== undefined ? {} : { update: lowerExpression(statement.update, loopScope) }),
          ...(continuing === undefined || continuing.length === 0 ? {} : { continuing }),
          body: materializedBody,
          span: statement.span,
        };
      }
    case "while": {
      const loweredCondition = materializeConditionalCalls(lowerConditionExpression(statement.condition, scope));
      const body = lowerStatements(statement.body, scope);
      return {
        kind: "loop",
        loopKind: "while",
        ...(loweredCondition.operations.length === 0 ? { condition: loweredCondition.expression } : {}),
        body: loweredCondition.operations.length === 0 ? body : semanticMaterializedLoopBody(loweredCondition, body, statement.span),
        span: statement.span,
      };
    }
    case "do-while": {
      const loweredCondition = materializeConditionalCalls(lowerConditionExpression(statement.condition, scope));
      return {
        kind: "loop",
        loopKind: "do-while",
        ...(loweredCondition.operations.length === 0 ? { condition: loweredCondition.expression } : {}),
        body: lowerStatements(statement.body, scope),
        ...(loweredCondition.operations.length === 0
          ? {}
          : { continuing: semanticMaterializedLoopCondition(loweredCondition, statement.span) }),
        span: statement.span,
      };
    }
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

function semanticMaterializedLoopBody(
  condition: { readonly operations: readonly SemanticKernelIrOperation[]; readonly expression: SemanticExpression },
  body: readonly SemanticKernelIrOperation[],
  span: SourceSpan,
): readonly SemanticKernelIrOperation[] {
  return [
    ...condition.operations,
    {
      kind: "branch",
      condition: condition.expression,
      consequent: body,
      alternate: [{ kind: "break", span }],
      span,
    },
  ];
}

function semanticMaterializedLoopCondition(
  condition: { readonly operations: readonly SemanticKernelIrOperation[]; readonly expression: SemanticExpression },
  span: SourceSpan,
): readonly SemanticKernelIrOperation[] {
  return [
    ...condition.operations,
    {
      kind: "branch",
      condition: condition.expression,
      consequent: [],
      alternate: [{ kind: "break", span }],
      span,
    },
  ];
}

function materializeConditionalCalls(
  expression: SemanticExpression,
): { readonly operations: readonly SemanticKernelIrOperation[]; readonly expression: SemanticExpression } {
  if (expression.kind === "conditional" &&
    (semanticExpressionContainsCall(expression.condition) || semanticExpressionContainsCall(expression.consequent) || semanticExpressionContainsCall(expression.alternate))) {
    const loweredCondition = materializeConditionalCalls(expression.condition);
    const condition = semanticExpressionContainsCall(loweredCondition.expression)
      ? materializeSemanticExpressionOnce(loweredCondition, "bool", "__bg.condition.test", expression.condition.span)
      : loweredCondition;
    const consequent = materializeConditionalCalls(expression.consequent);
    const alternate = materializeConditionalCalls(expression.alternate);
    const valueType = expressionValueType(expression);
    if (!valueType || valueType === "void" || isCudaVectorType(valueType)) return { operations: [], expression };
    const temp = tempScalarSymbol("__bg.condition.value", expression.span, valueType);
    const target = semanticSymbolExpression(temp, expression.span);
    const assign = (value: SemanticExpression): SemanticKernelIrOperation => ({
      kind: "expression",
      expression: { kind: "assignment", operator: "=", target, value, valueType, span: expression.span },
      span: expression.span,
    });
    return {
      operations: [
        ...condition.operations,
        { kind: "declare", target: temp, span: expression.span },
        {
          kind: "branch",
          condition: condition.expression,
          consequent: [...consequent.operations, assign(consequent.expression)],
          alternate: [...alternate.operations, assign(alternate.expression)],
          span: expression.span,
        },
      ],
      expression: target,
    };
  }
  if (expression.kind === "binary" && expression.operator !== "&&" && expression.operator !== "||") {
    const left = materializeConditionalCalls(expression.left);
    const right = materializeConditionalCalls(expression.right);
    return { operations: [...left.operations, ...right.operations], expression: { ...expression, left: left.expression, right: right.expression } };
  }
  if (expression.kind === "binary" && (expression.operator === "&&" || expression.operator === "||")) {
    const left = materializeConditionalCalls(expression.left);
    const right = materializeConditionalCalls(expression.right);
    if (left.operations.length === 0 && right.operations.length === 0) {
      return { operations: [], expression: { ...expression, left: left.expression, right: right.expression } };
    }
    const temp = tempScalarSymbol("__bg.short.circuit", expression.span, "bool");
    const target = semanticSymbolExpression(temp, expression.span);
    const assign = (value: SemanticExpression): SemanticKernelIrOperation => ({
      kind: "expression",
      expression: { kind: "assignment", operator: "=", target, value, valueType: "bool", span: expression.span },
      span: expression.span,
    });
    const constant = (value: boolean): SemanticExpression => ({
      kind: "literal",
      literalKind: "number",
      value: value ? 1 : 0,
      valueType: "bool",
      span: expression.span,
    });
    return {
      operations: [
        ...left.operations,
        { kind: "declare", target: temp, span: expression.span },
        expression.operator === "&&"
          ? { kind: "branch", condition: left.expression, consequent: [...right.operations, assign(right.expression)], alternate: [assign(constant(false))], span: expression.span }
          : { kind: "branch", condition: left.expression, consequent: [assign(constant(true))], alternate: [...right.operations, assign(right.expression)], span: expression.span },
      ],
      expression: target,
    };
  }
  if (expression.kind === "cast") {
    const nested = materializeConditionalCalls(expression.expression);
    return { operations: nested.operations, expression: { ...expression, expression: nested.expression } };
  }
  if (expression.kind === "unary") {
    const nested = materializeConditionalCalls(expression.argument);
    return { operations: nested.operations, expression: { ...expression, argument: nested.expression } };
  }
  if (expression.kind === "member") {
    const object = materializeConditionalCalls(expression.object);
    return { operations: object.operations, expression: { ...expression, object: object.expression } };
  }
  if (expression.kind === "index") {
    const target = materializeConditionalCalls(expression.target);
    const index = materializeConditionalCalls(expression.index);
    return {
      operations: [...target.operations, ...index.operations],
      expression: { ...expression, target: target.expression, index: index.expression },
    };
  }
  if (expression.kind === "call") {
    const callee = materializeConditionalCalls(expression.callee);
    const args = expression.args.map(materializeConditionalCalls);
    return {
      operations: [...callee.operations, ...args.flatMap((arg) => arg.operations)],
      expression: { ...expression, callee: callee.expression, args: args.map((arg) => arg.expression) },
    };
  }
  if (expression.kind === "assignment") {
    const target = materializeConditionalCalls(expression.target);
    const value = materializeConditionalCalls(expression.value);
    return {
      operations: [...target.operations, ...value.operations],
      expression: { ...expression, target: target.expression, value: value.expression },
    };
  }
  if (expression.kind === "update") {
    const argument = materializeConditionalCalls(expression.argument);
    return { operations: argument.operations, expression: { ...expression, argument: argument.expression } };
  }
  if (expression.kind === "sequence") {
    const parts = expression.expressions.map(materializeConditionalCalls);
    return {
      operations: parts.flatMap((part) => part.operations),
      expression: { ...expression, expressions: parts.map((part) => part.expression) },
    };
  }
  if (expression.kind === "initializer") {
    const elements = expression.elements.map(materializeConditionalCalls);
    return {
      operations: elements.flatMap((element) => element.operations),
      expression: { ...expression, elements: elements.map((element) => element.expression) },
    };
  }
  if (expression.kind === "texture-read") {
    const values = [expression.texture, expression.x, expression.y, ...(expression.z ? [expression.z] : [])].map(materializeConditionalCalls);
    return {
      operations: values.flatMap((value) => value.operations),
      expression: {
        ...expression,
        texture: values[0]!.expression,
        x: values[1]!.expression,
        y: values[2]!.expression,
        ...(expression.z ? { z: values[3]!.expression } : {}),
      },
    };
  }
  if (expression.kind === "surface-read") {
    const values = [expression.surface, expression.xBytes, expression.y, ...(expression.z ? [expression.z] : [])].map(materializeConditionalCalls);
    return {
      operations: values.flatMap((value) => value.operations),
      expression: {
        ...expression,
        surface: values[0]!.expression,
        xBytes: values[1]!.expression,
        y: values[2]!.expression,
        ...(expression.z ? { z: values[3]!.expression } : {}),
      },
    };
  }
  return { operations: [], expression };
}

function materializeSemanticExpressionOnce(
  lowered: { readonly operations: readonly SemanticKernelIrOperation[]; readonly expression: SemanticExpression },
  valueType: Exclude<CudaLiteScalarType, "void">,
  name: string,
  span: SourceSpan,
): { readonly operations: readonly SemanticKernelIrOperation[]; readonly expression: SemanticExpression } {
  const deduplicated = materializeRepeatedSemanticCalls(lowered.expression);
  const temp = tempScalarSymbol(name, span, valueType);
  return {
    operations: [...lowered.operations, ...deduplicated.operations, { kind: "declare", target: temp, init: deduplicated.expression, span }],
    expression: semanticSymbolExpression(temp, span),
  };
}

function materializeRepeatedSemanticCalls(
  expression: SemanticExpression,
): { readonly operations: readonly SemanticKernelIrOperation[]; readonly expression: SemanticExpression } {
  const calls = new Map<string, { readonly expression: Extract<SemanticExpression, { readonly kind: "call" }>; count: number }>();
  walkSemanticExpression(expression, (item) => {
    if (item.kind !== "call" || item.callee.kind !== "symbol") return;
    const valueType = expressionValueType(item);
    if (!valueType || valueType === "void" || isCudaVectorType(valueType)) return;
    const key = semanticCallIdentity(item);
    const existing = calls.get(key);
    if (existing) existing.count++;
    else calls.set(key, { expression: item, count: 1 });
  });
  const repeated = [...calls].filter(([, call]) => call.count > 1);
  if (repeated.length === 0) return { operations: [], expression };
  const replacements = new Map<string, Extract<SemanticExpression, { readonly kind: "symbol" }>>();
  const operations = repeated.map(([key, call]): SemanticKernelIrOperation => {
    const valueType = expressionValueType(call.expression)! as Exclude<CudaLiteScalarType, "void">;
    const temp = tempScalarSymbol("__bg.call.once", call.expression.span, valueType);
    replacements.set(key, semanticSymbolExpression(temp, call.expression.span));
    return { kind: "declare", target: temp, init: call.expression, span: call.expression.span };
  });
  return { operations, expression: rewriteMaterializedSemanticCalls(expression, replacements) };
}

function semanticCallIdentity(expression: Extract<SemanticExpression, { readonly kind: "call" }>): string {
  const name = expression.callee.kind === "symbol" ? expression.callee.name : expression.callee.kind;
  return `${name}:${expression.span.start}:${expression.span.end}`;
}

function rewriteMaterializedSemanticCalls(
  expression: SemanticExpression,
  replacements: ReadonlyMap<string, Extract<SemanticExpression, { readonly kind: "symbol" }>>,
): SemanticExpression {
  if (expression.kind === "call") {
    const replacement = replacements.get(semanticCallIdentity(expression));
    if (replacement) return replacement;
    return {
      ...expression,
      callee: rewriteMaterializedSemanticCalls(expression.callee, replacements),
      args: expression.args.map((arg) => rewriteMaterializedSemanticCalls(arg, replacements)),
    };
  }
  if (expression.kind === "member") return { ...expression, object: rewriteMaterializedSemanticCalls(expression.object, replacements) };
  if (expression.kind === "index") return {
    ...expression,
    target: rewriteMaterializedSemanticCalls(expression.target, replacements),
    index: rewriteMaterializedSemanticCalls(expression.index, replacements),
  };
  if (expression.kind === "cast") return { ...expression, expression: rewriteMaterializedSemanticCalls(expression.expression, replacements) };
  if (expression.kind === "unary") return { ...expression, argument: rewriteMaterializedSemanticCalls(expression.argument, replacements) };
  if (expression.kind === "binary") return {
    ...expression,
    left: rewriteMaterializedSemanticCalls(expression.left, replacements),
    right: rewriteMaterializedSemanticCalls(expression.right, replacements),
  };
  if (expression.kind === "conditional") return {
    ...expression,
    condition: rewriteMaterializedSemanticCalls(expression.condition, replacements),
    consequent: rewriteMaterializedSemanticCalls(expression.consequent, replacements),
    alternate: rewriteMaterializedSemanticCalls(expression.alternate, replacements),
  };
  if (expression.kind === "assignment") return {
    ...expression,
    target: rewriteMaterializedSemanticCalls(expression.target, replacements),
    value: rewriteMaterializedSemanticCalls(expression.value, replacements),
  };
  if (expression.kind === "update") return { ...expression, argument: rewriteMaterializedSemanticCalls(expression.argument, replacements) };
  if (expression.kind === "initializer") return { ...expression, elements: expression.elements.map((item) => rewriteMaterializedSemanticCalls(item, replacements)) };
  if (expression.kind === "sequence") return { ...expression, expressions: expression.expressions.map((item) => rewriteMaterializedSemanticCalls(item, replacements)) };
  if (expression.kind === "texture-read") return {
    ...expression,
    texture: rewriteMaterializedSemanticCalls(expression.texture, replacements),
    x: rewriteMaterializedSemanticCalls(expression.x, replacements),
    y: rewriteMaterializedSemanticCalls(expression.y, replacements),
    ...(expression.z ? { z: rewriteMaterializedSemanticCalls(expression.z, replacements) } : {}),
  };
  if (expression.kind === "surface-read") return {
    ...expression,
    surface: rewriteMaterializedSemanticCalls(expression.surface, replacements),
    xBytes: rewriteMaterializedSemanticCalls(expression.xBytes, replacements),
    y: rewriteMaterializedSemanticCalls(expression.y, replacements),
    ...(expression.z ? { z: rewriteMaterializedSemanticCalls(expression.z, replacements) } : {}),
  };
  return expression;
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
  const targetScalarType = targetRoot?.valueType;
  const sourceScalarType = sourceRoot?.valueType;
  if (!targetScalarType || targetScalarType === "void" || !sourceScalarType || sourceScalarType === "void" || isCudaVectorType(targetScalarType) || isCudaVectorType(sourceScalarType)) return undefined;
  const viewBytes = sizeofCudaType(viewType);
  const targetScalarBytes = sizeofCudaType(targetScalarType);
  const sourceScalarBytes = sizeofCudaType(sourceScalarType);
  if (!viewBytes || !targetScalarBytes || !sourceScalarBytes || viewBytes % targetScalarBytes !== 0 || viewBytes % sourceScalarBytes !== 0) return undefined;
  if (viewBytes < 1 || viewBytes > 64) return undefined;
  return {
    kind: "copy",
    source: { ...source, valueType: sourceScalarType },
    target: { ...target, valueType: targetScalarType },
    bytes: viewBytes,
    span,
  };
}

function semanticConditionalReinterpretedCopy(
  target: SemanticMemoryRef,
  value: SemanticExpression,
  operator: string,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticKernelIrOperation | undefined {
  if (value.kind !== "conditional") return undefined;
  const consequent = semanticReinterpretedScalarCopy(target, value.consequent, operator, scope, span);
  const alternate = semanticReinterpretedScalarCopy(target, value.alternate, operator, scope, span);
  if (!consequent || !alternate) return undefined;
  return { kind: "branch", condition: value.condition, consequent: [consequent], alternate: [alternate], span };
}

function semanticSequencedAssignmentValue(expression: SemanticExpression): SemanticExpression {
  if (expression.kind !== "assignment") return expression;
  return {
    kind: "sequence",
    expressions: [expression, expression.target],
    valueType: expression.valueType,
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
    calleeId: expression.value.callee.id,
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
    (operation.kind === "loop" || operation.kind === "block") && semanticIrOperationsContainBarrier(operation.body, barrierFunctions) ||
    operation.kind === "loop" && operation.continuing !== undefined && semanticIrOperationsContainBarrier(operation.continuing, barrierFunctions)
  );
}

function semanticIrOperationsContainSchedulingCollective(
  operations: readonly SemanticKernelIrOperation[],
  barrierFunctions: ReadonlySet<string>,
): boolean {
  if (semanticIrOperationsContainBarrier(operations, barrierFunctions)) return true;
  return collectSemanticFunctionCalls(operations).some(({ callee }) =>
    SEMANTIC_SUBGROUP_CALLS.has(callee) ||
    callee === "cg::reduce" ||
    callee === "cooperative_groups::reduce" ||
    callee === "cg::inclusive_scan" ||
    callee === "cooperative_groups::inclusive_scan" ||
    callee === "cg::exclusive_scan" ||
    callee === "cooperative_groups::exclusive_scan"
  );
}

function promoteSemanticBarrierResultCalls(
  operations: readonly SemanticKernelIrOperation[],
  barrierFunctions: ReadonlySet<string>,
): readonly SemanticKernelIrOperation[] {
  return operations.flatMap((operation): readonly SemanticKernelIrOperation[] => {
    if (
      operation.kind === "declare" &&
      operation.init?.kind === "call" &&
      operation.init.callee.kind === "symbol" &&
      barrierFunctions.has(operation.init.callee.name)
    ) {
      const { init: _init, ...declaration } = operation;
      return [declaration, {
        kind: "call",
        calleeId: operation.init.callee.id,
        callee: operation.init.callee.name,
        args: operation.init.args,
        reads: operation.init.args.flatMap((arg) => collectMemoryRefs(arg)),
        result: semanticSymbolExpression(operation.target, operation.span),
        span: operation.span,
      }];
    }
    if (operation.kind === "expression" && operation.expression.kind === "assignment") {
      const promoted = semanticResultCallOperation(operation.expression, operation.span);
      if (promoted && barrierFunctions.has(promoted.callee)) return [promoted];
    }
    if (operation.kind === "branch") {
      return [{
        ...operation,
        consequent: promoteSemanticBarrierResultCalls(operation.consequent, barrierFunctions),
        alternate: promoteSemanticBarrierResultCalls(operation.alternate, barrierFunctions),
      }];
    }
    if (operation.kind === "loop" || operation.kind === "block") {
      return [{ ...operation, body: promoteSemanticBarrierResultCalls(operation.body, barrierFunctions),
        ...(operation.kind === "loop" && operation.continuing !== undefined ? { continuing: promoteSemanticBarrierResultCalls(operation.continuing, barrierFunctions) } : {}) }];
    }
    return [operation];
  });
}

function semanticSymbolForCooperativeGroup(statement: CudaLiteCooperativeGroupDecl): CudaLiteSemanticSymbol {
  return {
    id: createSemanticSymbolId("cooperative-group", statement.name, statement.span),
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
  const targetSource = source.kind === "assignment" && (source.operator === "=" || source.operator === "+=" || source.operator === "-=")
    ? source.left
    : source.kind === "update" && (source.operator === "++" || source.operator === "--")
      ? source.argument
      : undefined;
  if (!targetSource) return undefined;
  const targetExpression = lowerExpression(targetSource, scope);
  const target = memoryRefFromExpression(targetExpression);
  if (!target || target.addressSpace !== "storage" || target.indices.length !== 0 || target.fields.length !== 0) return undefined;
  if (source.kind === "assignment" && (source.operator === "+=" || source.operator === "-=")) {
    const value = lowerExpression(source.right, scope);
    return { kind: "store", target, value, operator: source.operator, reads: collectMemoryRefs(value), span };
  }
  if (source.kind === "update") {
    const value = intNumberExpression(1, span);
    return { kind: "store", target, value, operator: source.operator === "++" ? "+=" : "-=", reads: [], span };
  }
  if (source.kind !== "assignment") return undefined;
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

function markLoopUpdatePointerRuntimeState(
  expression: CudaLiteExpression | undefined,
  scope: Map<string, CudaLiteSemanticSymbol>,
): void {
  if (!expression) return;
  if (expression.kind === "sequence") {
    for (const item of expression.expressions) markLoopUpdatePointerRuntimeState(item, scope);
    return;
  }
  if (expression.kind !== "assignment" || expression.operator !== "=" || expression.left.kind !== "identifier") return;
  const target = scope.get(expression.left.name);
  if (!target || target.kind !== "local" || !target.pointer || target.dimensions.length !== 0) return;
  const source = localPointerAliasForInitializer(expression.right, scope);
  if (!source?.pointerRoot || source.pointerAddressSpace !== "storage" || source.pointerBaseIndices?.length !== 1) return;
  scope.set(target.name, {
    ...semanticSymbolWithoutPointerAlias(target),
    pointerRuntimeState: true,
  });
}

function lowerLoopUpdateOperations(
  expression: CudaLiteExpression,
  scope: Map<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] {
  if (expression.kind === "sequence") {
    return expression.expressions.flatMap((item) => lowerLoopUpdateOperations(item, scope));
  }
  const pointerRebind = semanticLocalPointerRebindOperation(expression, scope);
  if (pointerRebind) return [pointerRebind];
  return [{ kind: "expression", expression: lowerExpression(expression, scope), span: expression.span }];
}

function loopUpdateRequiresOperations(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): boolean {
  if (expression.kind === "sequence") return expression.expressions.some((item) => loopUpdateRequiresOperations(item, scope));
  return semanticLocalPointerRebindOperation(expression, scope) !== undefined;
}

function semanticLocalPointerRebindOperation(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): Extract<SemanticKernelIrOperation, { readonly kind: "pointer-rebind" }> | undefined {
  if (expression.kind !== "assignment" || expression.operator !== "=" || expression.left.kind !== "identifier") return undefined;
  const target = scope.get(expression.left.name);
  if (!target?.pointerRuntimeState || target.kind !== "local" || !target.pointer) return undefined;
  return semanticLocalPointerRebindFromSource(target, expression.right, scope, expression.span);
}

function semanticLocalPointerRebindFromSource(
  target: CudaLiteSemanticSymbol,
  source: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): Extract<SemanticKernelIrOperation, { readonly kind: "pointer-rebind" }> | undefined {
  const alias = localPointerAliasForInitializer(source, scope);
  if (!alias?.pointerRoot || alias.pointerAddressSpace !== "storage" || alias.pointerBaseIndices?.length !== 1) return undefined;
  const root = semanticSymbolForMemoryId(scope, alias.pointerRoot);
  const valueType = target.valueType;
  if (!root || valueType === undefined || valueType === "void") return undefined;
  return {
    kind: "pointer-rebind",
    target,
    source: {
      baseId: alias.pointerRoot,
      base: root.name,
      addressSpace: alias.pointerAddressSpace,
      valueType,
      ...(alias.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
      ...(alias.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: alias.pointerBaseUnitBytes }),
      indices: alias.pointerBaseIndices,
      fields: [],
      span: source.span,
    },
    span,
  };
}

function semanticPointerArrayRebindFromAlias(
  target: CudaLiteSemanticSymbol,
  slot: SemanticExpression,
  alias: SemanticPointerAlias,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): Extract<SemanticKernelIrOperation, { readonly kind: "pointer-array-rebind" }> | undefined {
  if (target.dimensions.length !== 1 || !alias.pointerRoot || alias.pointerAddressSpace !== "storage" || alias.pointerBaseIndices?.length !== 1) {
    return undefined;
  }
  const root = semanticSymbolForMemoryId(scope, alias.pointerRoot);
  const valueType = target.valueType;
  if (!root || valueType === undefined || valueType === "void") return undefined;
  return {
    kind: "pointer-array-rebind",
    target,
    slot,
    source: {
      baseId: alias.pointerRoot,
      base: root.name,
      addressSpace: alias.pointerAddressSpace,
      valueType,
      ...(alias.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
      ...(alias.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: alias.pointerBaseUnitBytes }),
      indices: alias.pointerBaseIndices,
      fields: [],
      span,
    },
    span,
  };
}

function semanticPointerArrayRebindOperationsFromAlias(
  target: CudaLiteSemanticSymbol,
  slot: SemanticExpression,
  alias: SemanticPointerAlias,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): readonly SemanticKernelIrOperation[] | undefined {
  if (alias.pointerSelection) {
    const consequent = semanticPointerArrayRebindOperationsFromAlias(target, slot, alias.pointerSelection.consequent, scope, span);
    const alternate = semanticPointerArrayRebindOperationsFromAlias(target, slot, alias.pointerSelection.alternate, scope, span);
    if (!consequent || !alternate) return undefined;
    return [{
      kind: "branch",
      condition: alias.pointerSelection.condition,
      consequent,
      alternate,
      span,
    }];
  }
  const rebind = semanticPointerArrayRebindFromAlias(target, slot, alias, scope, span);
  return rebind === undefined ? undefined : [rebind];
}

function markBranchPointerRuntimeState(
  consequent: readonly CudaLiteStatement[],
  alternate: readonly CudaLiteStatement[],
  scope: Map<string, CudaLiteSemanticSymbol>,
): void {
  const collect = (statements: readonly CudaLiteStatement[]): ReadonlySet<string> => {
    const assigned = new Set<string>();
    for (const statement of statements) {
      if (statement.kind === "expr" && statement.expression.kind === "assignment" &&
        statement.expression.operator === "=" && statement.expression.left.kind === "identifier") {
        const target = scope.get(statement.expression.left.name);
        const source = localPointerAliasForInitializer(statement.expression.right, scope);
        if (target?.kind === "local" && target.pointer && target.dimensions.length === 0 &&
          source?.pointerRoot && source.pointerAddressSpace === "storage" && source.pointerBaseIndices?.length === 1) {
          assigned.add(target.name);
        }
      }
    }
    return assigned;
  };
  const consequentAssignments = collect(consequent);
  const alternateAssignments = collect(alternate);
  for (const name of new Set([...consequentAssignments, ...alternateAssignments])) {
    const target = scope.get(name)!;
    const hasInitialStorageRoot = target.pointerRoot !== undefined && target.pointerAddressSpace === "storage";
    const assignedOnEveryPath = consequentAssignments.has(name) && alternateAssignments.has(name);
    if (!hasInitialStorageRoot && !assignedOnEveryPath) continue;
    scope.set(name, { ...semanticSymbolWithoutPointerAlias(target), pointerRuntimeState: true });
  }
}

function semanticConditionalPointerVarInitOperations(
  statement: CudaLiteStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
  followingStatements: readonly CudaLiteStatement[],
): readonly SemanticKernelIrOperation[] | undefined {
  if (statement.kind !== "var" || statement.storage !== "local" || !statement.pointer ||
    statement.dimensions.length !== 0 || statement.init?.kind !== "conditional" ||
    !pointerRequiresRuntimeRootIdentity(statement.name, followingStatements)) return undefined;
  const consequentAlias = localPointerAliasForInitializer(statement.init.consequent, scope);
  const alternateAlias = localPointerAliasForInitializer(statement.init.alternate, scope);
  if (!consequentAlias?.pointerRoot || consequentAlias.pointerAddressSpace !== "storage" || consequentAlias.pointerBaseIndices?.length !== 1 ||
    !alternateAlias?.pointerRoot || alternateAlias.pointerAddressSpace !== "storage" || alternateAlias.pointerBaseIndices?.length !== 1 ||
    semanticIdsEqual(consequentAlias.pointerRoot, alternateAlias.pointerRoot)) return undefined;
  const original = symbolForVar(statement, scope);
  const target: CudaLiteSemanticSymbol = {
    ...semanticSymbolWithoutPointerAlias(original),
    pointerRuntimeState: true,
  };
  scope.set(target.name, target);
  const consequent = semanticLocalPointerRebindFromSource(target, statement.init.consequent, scope, statement.init.consequent.span);
  const alternate = semanticLocalPointerRebindFromSource(target, statement.init.alternate, scope, statement.init.alternate.span);
  if (!consequent || !alternate) return undefined;
  return [
    { kind: "declare", target, span: statement.span },
    {
      kind: "branch",
      condition: lowerConditionExpression(statement.init.condition, scope),
      consequent: [consequent],
      alternate: [alternate],
      span: statement.init.span,
    },
  ];
}

function pointerRequiresRuntimeRootIdentity(
  pointerName: string,
  statements: readonly CudaLiteStatement[],
): boolean {
  let required = false;
  walkCudaLiteExpressions(statements, (expression) => {
    if (required || expression.kind !== "call" || expression.callee.kind !== "identifier" ||
      semanticAtomicOperation(expression.callee.name) === undefined) return;
    const target = expression.args[0];
    if (target?.kind === "identifier" && target.name === pointerName) required = true;
  });
  return required;
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
      const constantValue = symbol?.constant && symbol.init ? staticNumberValue(symbol.init) : undefined;
      if (constantValue !== undefined) {
        return {
          kind: "literal",
          literalKind: "number",
          value: constantValue,
          valueType: requiredSemanticValueType(symbol?.valueType, `constant '${expression.name}'`, expression.span),
          span: expression.span,
        };
      }
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
      if (symbol === undefined && isCudaBuiltinVectorSymbolName(expression.name)) {
        return {
          kind: "symbol",
          id: createBuiltinSemanticSymbolId(expression.name),
          name: expression.name,
          valueType: "uint3",
          addressSpace: "builtin",
          span: expression.span,
        };
      }
      if (symbol === undefined && isCooperativeReductionObjectName(expression.name)) {
        return {
          kind: "symbol",
          id: createBuiltinSemanticSymbolId(expression.name),
          name: expression.name,
          addressSpace: "builtin",
          span: expression.span,
        };
      }
      return {
        kind: "symbol",
        id: symbol?.id ?? createUnresolvedSemanticSymbolId(expression.name, expression.span),
        name: symbol?.name ?? expression.name,
        ...(symbol?.valueType === undefined ? {} : { valueType: symbol.valueType }),
        addressSpace: symbol?.addressSpace ?? "unknown",
        span: expression.span,
      };
    }
    case "member": {
      const object = lowerExpression(expression.object, scope);
      const matrixTile = expression.property === "num_elements" ? semanticMatrixTileRef(object, scope) : undefined;
      if (matrixTile) return intNumberExpression(matrixTileElementCount(matrixTile.spec), expression.span);
      return {
        kind: "member",
        object,
        property: expression.property,
        valueType: requiredSemanticExpressionType(memberValueType(object, expression.property), `member '${expression.property}'`, expression.span),
        span: expression.span,
      };
    }
    case "index": {
      const localScalar = directLocalScalarPointerIndexExpression(expression, scope);
      if (localScalar) return localScalar;
      const localVectorLane = directLocalVectorPointerIndexExpression(expression, scope);
      if (localVectorLane) return localVectorLane;
      const aliased = localPointerAliasIndexExpression(expression, scope);
      if (aliased) return aliased;
      const target = lowerExpression(expression.target, scope);
      const lowered: SemanticExpression = {
        kind: "index",
        target,
        index: lowerExpression(expression.index, scope),
        valueType: requiredSemanticValueType(indexedExpressionValueType(expression, target, scope), "index expression", expression.span),
        addressSpace: expressionAddressSpace(target),
        span: expression.span,
      };
      const matrixLane = semanticMatrixLaneMemoryRef(lowered, scope);
      if (!matrixLane || matrixLane.indices.length !== 1) return lowered;
      const root = scope.get(matrixLane.base);
      if (!root) return lowered;
      return {
        kind: "index",
        target: semanticSymbolExpression(root, expression.target.span),
        index: matrixLane.indices[0]!,
        valueType: requiredSemanticValueType(matrixLane.valueType, "matrix lane index", expression.span),
        addressSpace: "local",
        span: expression.span,
      };
    }
    case "call": {
      if (expression.callee.kind === "identifier" && (expression.callee.name === "sizeof" || expression.callee.name === "alignof")) {
        const value = semanticSizeofAlignofValue(expression.callee.name, expression.args[0], scope);
        if (value !== undefined) return intNumberExpression(value, expression.span);
      }
      if (expression.callee.kind === "identifier" && isNanPayloadCallName(expression.callee.name)) {
        return semanticCallExpression(
          "__uint_as_float",
          [semanticUintLiteralExpression(0x7fc00000, expression.span)],
          "float",
          expression.span,
        );
      }
      if (expression.callee.kind === "identifier" && expression.callee.name === "cudaSetDeviceFlags") {
        return intNumberExpression(0, expression.span);
      }
      const generatedRandom = expression.callee.kind === "identifier" && isSemanticGeneratedRandomCall(expression.callee.name)
        ? semanticGeneratedRandomReturnType(expression.callee.name)
        : undefined;
      const preservePointerArgs = expression.callee.kind === "identifier" &&
        (SEMANTIC_LOCAL_ARRAY_FILL_CALLS.has(expression.callee.name) || SEMANTIC_CURAND_CALLS.has(expression.callee.name) || generatedRandom !== undefined);
      const encodeNullablePointerArgs = expression.callee.kind === "identifier" && scope.get(expression.callee.name)?.kind === "function";
      const args = expression.args.map((arg) => preservePointerArgs
        ? lowerExpression(arg, scope)
        : pointerAliasValueExpression(arg, scope, arg.span, encodeNullablePointerArgs) ?? lowerExpression(arg, scope));
      if (generatedRandom !== undefined && expression.callee.kind === "identifier") {
        return {
          kind: "call",
          callee: { kind: "symbol", id: createBuiltinSemanticSymbolId(expression.callee.name), name: expression.callee.name, valueType: generatedRandom, addressSpace: "builtin", span: expression.callee.span },
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
        (expression.callee.name === "surf1Dread" && args.length === 2 ||
          expression.callee.name === "surf2Dread" && args.length === 3 ||
          (expression.callee.name === "surf2DLayeredread" || expression.callee.name === "surf3Dread") && args.length === 4)
      ) {
        const surface = semanticIndexedSurfaceLayer(args[0]!, args[3]);
        return {
          kind: "surface-read",
          callee: expression.callee.name as "surf1Dread" | "surf2Dread" | "surf2DLayeredread" | "surf3Dread",
          surface: surface.surface,
          xBytes: args[1]!,
          y: expression.callee.name === "surf1Dread" ? numberExpression(0, expression.span) : args[2]!,
          ...(surface.z === undefined ? {} : { z: surface.z }),
          valueType: expression.templateValueType ?? "float",
          span: expression.span,
        };
      }
      const callee = expression.callee.kind === "identifier" && !scope.has(expression.callee.name)
        ? {
            kind: "symbol" as const,
            id: createBuiltinSemanticSymbolId(expression.callee.name),
            name: expression.callee.name,
            addressSpace: "builtin" as const,
            span: expression.callee.span,
          }
        : lowerExpression(expression.callee, scope);
      const cooperativeGroupValueType = semanticCooperativeGroupCallValueType(expression);
      const valueType = expression.callee.kind === "identifier" && expression.callee.name === "__activemask"
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
          : cooperativeGroupValueType ?? expression.templateValueType ?? semanticIntrinsicReturnType(expression.callee.kind === "identifier" ? expression.callee.name : undefined, args) ?? expressionValueType(callee) ?? expressionValueType(args[0]);
      return {
        kind: "call",
        callee,
        args,
        ...(expression.templateValueType === undefined ? {} : { templateValueType: expression.templateValueType }),
        valueType: requiredSemanticExpressionType(
          valueType,
          `call '${semanticCallName(callee) ?? "<expression>"}'`,
          expression.span,
        ),
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
        valueType: requiredSemanticExpressionType(expression.operator === "&" ? "voidptr" : expression.operator === "!" ? "bool" : expressionValueType(argument), `unary '${expression.operator}'`, expression.span),
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
        valueType: requiredSemanticValueType(semanticBinaryResultValueType(expression.operator, left, right), `binary '${expression.operator}'`, expression.span),
        span: expression.span,
      };
    }
    case "conditional": {
      const consequent = lowerExpression(expression.consequent, scope);
      const alternate = lowerExpression(expression.alternate, scope);
      return {
        kind: "conditional",
        condition: lowerConditionExpression(expression.condition, scope),
        consequent,
        alternate,
        valueType: requiredSemanticExpressionType(expressionValueType(consequent) ?? expressionValueType(alternate), "conditional expression", expression.span),
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
        valueType: requiredSemanticValueType(expressionValueType(value), `assignment '${expression.operator}'`, expression.span),
        span: expression.span,
      };
    }
    case "update": {
      const argument = lowerExpression(expression.argument, scope);
      return { kind: "update", operator: expression.operator, argument, prefix: expression.prefix, valueType: requiredSemanticValueType(expressionValueType(argument), `update '${expression.operator}'`, expression.span), span: expression.span };
    }
    case "initializer":
      return { kind: "initializer", elements: expression.elements.map((element) => lowerExpression(element, scope)), span: expression.span };
    case "sequence": {
      const expressions = expression.expressions.map((item) => lowerExpression(item, scope));
      return { kind: "sequence", expressions, valueType: requiredSemanticExpressionType(expressionValueType(expressions.at(-1)), "sequence expression", expression.span), span: expression.span };
    }
  }
}

function lowerConditionExpression(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticExpression {
  const alias = localPointerAliasForInitializer(expression, scope);
  const validity = semanticPointerAliasValidity(alias, expression.span);
  return validity ?? lowerExpression(expression, scope);
}

function semanticPointerAliasValidity(
  alias: SemanticPointerAlias | undefined,
  span: SourceSpan,
): SemanticExpression | undefined {
  if (!alias) return undefined;
  if (alias.pointerRoot && semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace)) {
    return alias.pointerValid ?? booleanExpression(true, span);
  }
  if (!alias.pointerSelection) return undefined;
  const consequent = semanticPointerAliasValidity(alias.pointerSelection.consequent, span);
  const alternate = semanticPointerAliasValidity(alias.pointerSelection.alternate, span);
  if (!consequent || !alternate) return undefined;
  return {
    kind: "conditional",
    condition: alias.pointerSelection.condition,
    consequent,
    alternate,
    valueType: "bool",
    span,
  };
}

function semanticIndexedSurfaceLayer(
  surface: SemanticExpression,
  explicitZ: SemanticExpression | undefined,
): { readonly surface: SemanticExpression; readonly z?: SemanticExpression } {
  if (
    explicitZ === undefined &&
    surface.kind === "index" &&
    surface.target.kind === "symbol" &&
    surface.target.addressSpace === "surface"
  ) {
    return { surface: surface.target, z: surface.index };
  }
  return { surface, ...(explicitZ === undefined ? {} : { z: explicitZ }) };
}

function directLocalVectorPointerIndexExpression(
  expression: Extract<CudaLiteExpression, { readonly kind: "index" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticExpression | undefined {
  if (
    expression.target.kind !== "cast" || !expression.target.pointer ||
    expression.target.expression.kind !== "unary" || expression.target.expression.operator !== "&" ||
    expression.target.expression.argument.kind !== "identifier" ||
    isCudaVectorType(expression.target.valueType)
  ) return undefined;
  const symbol = scope.get(expression.target.expression.argument.name);
  if (!symbol || symbol.kind !== "local" || symbol.pointer || symbol.dimensions.length !== 0 || !isCudaVectorType(symbol.valueType)) return undefined;
  const rootScalar = cudaVectorScalarType(symbol.valueType);
  if (sizeofCudaType(rootScalar ?? "") !== sizeofCudaType(expression.target.valueType)) return undefined;
  return {
    kind: "index",
    target: semanticSymbolExpression(symbol, expression.target.expression.argument.span),
    index: lowerExpression(expression.index, scope),
    valueType: expression.target.valueType,
    addressSpace: "local",
    pointerBaseIsScalarLane: true,
    span: expression.span,
  };
}

function directLocalScalarPointerIndexExpression(
  expression: Extract<CudaLiteExpression, { readonly kind: "index" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticExpression | undefined {
  if (
    staticNumberValue(lowerExpression(expression.index, scope)) !== 0 ||
    expression.target.kind !== "cast" || !expression.target.pointer ||
    expression.target.expression.kind !== "unary" || expression.target.expression.operator !== "&" ||
    expression.target.expression.argument.kind !== "identifier"
  ) return undefined;
  const symbol = scope.get(expression.target.expression.argument.name);
  if (
    !symbol || symbol.kind !== "local" || symbol.pointer || symbol.dimensions.length !== 0 ||
    symbol.valueType !== expression.target.valueType
  ) return undefined;
  return semanticSymbolExpression(symbol, expression.span);
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
      id: createBuiltinSemanticSymbolId(`__${op}`),
      name: `__${op}`,
      addressSpace: "builtin",
      span: expression.callee.span,
    },
    args: [value, index, width],
    valueType: requiredSemanticExpressionType(expressionValueType(value), "cooperative shuffle", expression.span),
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
    pointerBaseIsScalarLane: _pointerBaseIsScalarLane,
    pointerBaseUnitBytes: _pointerBaseUnitBytes,
    pointerValid: _pointerValid,
    pointerSelection: _pointerSelection,
    ...rest
  } = symbol;
  return rest;
}

function lowerDeviceLaunch(
  statement: CudaLiteKernelLaunchStatement,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticDeviceLaunch {
  const callee = scope.get(statement.callee);
  return {
    calleeId: callee?.kind === "function"
      ? semanticFunctionIdFromSymbol(callee.id)
      : createUnresolvedSemanticFunctionId(statement.callee, statement.span),
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
    if (operation.kind === "declare" && (
      semanticPointerSymbolNeedsRuntimeState(operation.target) ||
      !operation.target.pointer && operation.target.dimensions.length > 0 ||
      operation.target.addressSpace === "shared"
    )) out.push(operation.target);
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
  const source = semanticPointerArgumentMemoryRef(sourceSource, scope);
  const byteCount = byteCountSource === undefined ? undefined : staticNumberValue(lowerExpression(byteCountSource, scope));
  if (
    !source ||
    source.valueType === undefined ||
    source.fields.length > 0 ||
    byteCount === undefined ||
    !Number.isInteger(byteCount) ||
    byteCount <= 0
  ) return undefined;
  const elementBytes = sizeofCudaType(source.valueType) ?? 0;
  if (elementBytes <= 0 || byteCount % elementBytes !== 0) return undefined;
  const elements = byteCount / elementBytes;
  if (elements < 1 || elements > 16) return undefined;
  const pointerTarget = semanticPointerArgumentMemoryRef(targetSource, scope);
  const target = pointerTarget ?? semanticCpAsyncSharedByteTarget(targetSource, source.valueType, scope);
  if (
    !target ||
    target.valueType !== source.valueType ||
    target.fields.length > 0
  ) return undefined;
  return { kind: "copy", source, target, bytes: byteCount, span };
}

function semanticMatrixOperation(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticKernelIrOperation | undefined {
  if (expression.kind !== "call" || expression.callee.kind !== "identifier") return undefined;
  const builtin = wmmaBuiltinName(expression.callee.name);
  if (!builtin) return undefined;
  const args = expression.args;
  if (builtin === "fill_fragment" && args[0] && args[1]) {
    const fragment = semanticMatrixTileRef(lowerExpression(args[0], scope), scope);
    return fragment ? { kind: "matrix-fill", fragment, value: lowerExpression(args[1], scope), span } : undefined;
  }
  if (builtin === "load_matrix_sync" && args[0] && args[1] && args[2]) {
    const fragment = semanticMatrixTileRef(lowerExpression(args[0], scope), scope);
    const source = semanticPointerArgumentMemoryRef(args[1], scope);
    const layout = semanticMatrixLayout(args[3]) ?? fragment?.spec.layout;
    return fragment && source && layout
      ? { kind: "matrix-load", fragment, source, stride: lowerExpression(args[2], scope), layout, span }
      : undefined;
  }
  if (builtin === "mma_sync" && args[0] && args[1] && args[2] && args[3]) {
    const destination = semanticMatrixTileRef(lowerExpression(args[0], scope), scope);
    const a = semanticMatrixTileRef(lowerExpression(args[1], scope), scope);
    const b = semanticMatrixTileRef(lowerExpression(args[2], scope), scope);
    const accumulator = semanticMatrixTileRef(lowerExpression(args[3], scope), scope);
    return destination && a && b && accumulator
      ? { kind: "matrix-mma", destination, a, b, accumulator, span }
      : undefined;
  }
  if (builtin === "store_matrix_sync" && args[0] && args[1] && args[2]) {
    const target = semanticPointerArgumentMemoryRef(args[0], scope);
    const fragment = semanticMatrixTileRef(lowerExpression(args[1], scope), scope);
    const layout = semanticMatrixLayout(args[3]);
    return target && fragment && layout
      ? { kind: "matrix-store", target, fragment, stride: lowerExpression(args[2], scope), layout, span }
      : undefined;
  }
  return undefined;
}

function semanticMatrixTileRef(
  expression: SemanticExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticMatrixTileRef | undefined {
  const indices: SemanticExpression[] = [];
  let cursor = expression;
  while (cursor.kind === "index") {
    indices.unshift(cursor.index);
    cursor = cursor.target;
  }
  if (cursor.kind !== "symbol") return undefined;
  const symbol = scope.get(cursor.name);
  if (!symbol?.matrixTile || indices.length !== (symbol.matrixTileArrayDimensions?.length ?? 0)) return undefined;
  return {
    baseId: semanticMemoryIdFromSymbol(symbol.id),
    base: symbol.name,
    spec: symbol.matrixTile,
    arrayDimensions: symbol.matrixTileArrayDimensions ?? [],
    indices,
    span: expression.span,
  };
}

function semanticMatrixLaneMemoryRef(
  expression: SemanticExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticMemoryRef | undefined {
  const ref = memoryRefFromExpression(expression);
  if (!ref || ref.addressSpace !== "local" || ref.fields.length !== 1 || ref.fields[0] !== "x") return undefined;
  const symbol = scope.get(ref.base);
  const dimensions = symbol?.matrixTileArrayDimensions;
  const tile = symbol?.matrixTile;
  if (!tile || !dimensions || ref.indices.length !== dimensions.length + 1) return undefined;
  const arrayIndices = ref.indices.slice(0, -1);
  const lane = ref.indices.at(-1)!;
  if (arrayIndices.length === 0) {
    return { ...ref, valueType: tile.valueType, containerValueType: tile.valueType, indices: [lane], fields: [] };
  }
  let flat = arrayIndices[0]!;
  for (let axis = 1; axis < arrayIndices.length; axis++) {
    flat = addIndexExpressions(
      multiplyIndexExpression(flat, dimensions[axis]!, expression.span),
      arrayIndices[axis]!,
      expression.span,
    );
  }
  flat = addIndexExpressions(
    multiplyIndexExpression(flat, matrixTileElementCount(tile), expression.span),
    lane,
    expression.span,
  );
  return { ...ref, valueType: tile.valueType, containerValueType: tile.valueType, indices: [flat], fields: [] };
}

function semanticMatrixLayout(expression: CudaLiteExpression | undefined): MatrixTileLayout | undefined {
  return expression?.kind === "identifier" ? normalizeMatrixTileLayout(expression.name) : undefined;
}

interface SemanticSharedByteAddress {
  readonly root: SemanticMemoryRef;
  readonly byteOffset: SemanticExpression;
}

function semanticCpAsyncSharedByteTarget(
  expression: CudaLiteExpression,
  valueType: SemanticValueType,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticMemoryRef | undefined {
  const address = semanticSharedByteAddress(lowerExpression(expression, scope), scope, new Set());
  const elementBytes = sizeofCudaType(valueType);
  if (
    !address ||
    !elementBytes ||
    !semanticExpressionKnownMultipleOf(address.byteOffset, elementBytes) ||
    address.root.fields.length > 0 ||
    address.root.addressSpace !== "shared"
  ) return undefined;
  return {
    ...address.root,
    valueType,
    pointerBaseIsScalarLane: true,
    indices: [divideIndexExpression(address.byteOffset, elementBytes, expression.span)],
    fields: [],
    span: expression.span,
  };
}

function semanticSharedByteAddress(
  expression: SemanticExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  seen: Set<string>,
): SemanticSharedByteAddress | undefined {
  if (expression.kind === "cast") return semanticSharedByteAddress(expression.expression, scope, seen);
  if (expression.kind === "symbol") {
    if (seen.has(expression.name)) return undefined;
    const init = scope.get(expression.name)?.init;
    if (!init) return undefined;
    const nextSeen = new Set(seen);
    nextSeen.add(expression.name);
    return semanticSharedByteAddress(init, scope, nextSeen);
  }
  if (expression.kind === "call" && expression.callee.kind === "symbol" && expression.callee.name === "__cvta_generic_to_shared") {
    const target = expression.args[0];
    if (!target) return undefined;
    const root = memoryRefFromExpression(target.kind === "unary" && target.operator === "&" ? target.argument : target);
    if (!root || root.addressSpace !== "shared" || root.fields.length > 0) return undefined;
    const rootElementBytes = sizeofCudaType(root.valueType ?? "uchar");
    if (!rootElementBytes) return undefined;
    const symbol = scope.get(root.base);
    if (root.indices.length > 1 && (!symbol || symbol.dimensions.length < root.indices.length)) return undefined;
    const index = root.indices.length <= 1
      ? root.indices[0] ?? zeroExpression(expression.span)
      : flatIndexExpressionForDimensions(symbol!.dimensions, root.indices, expression.span);
    return {
      root: { ...root, indices: [], fields: [] },
      byteOffset: multiplyIndexExpression(index, rootElementBytes, expression.span),
    };
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const left = semanticSharedByteAddress(expression.left, scope, seen);
    const right = semanticSharedByteAddress(expression.right, scope, seen);
    if (left && !right) return { ...left, byteOffset: binaryIndexExpression(expression.operator, left.byteOffset, expression.right, expression.span) };
    if (right && !left && expression.operator === "+") return { ...right, byteOffset: binaryIndexExpression("+", right.byteOffset, expression.left, expression.span) };
  }
  return undefined;
}

function divideIndexExpression(expression: SemanticExpression, divisor: number, span: SourceSpan): SemanticExpression {
  if (divisor === 1) return expression;
  return binaryIndexExpression("/", expression, intNumberExpression(divisor, span), span);
}

function semanticExpressionKnownMultipleOf(expression: SemanticExpression, divisor: number): boolean {
  if (divisor === 1) return true;
  const staticValue = staticNumberValue(expression);
  if (staticValue !== undefined) return Number.isInteger(staticValue) && staticValue % divisor === 0;
  if (expression.kind === "cast") return semanticExpressionKnownMultipleOf(expression.expression, divisor);
  if (expression.kind !== "binary") return false;
  if (expression.operator === "+" || expression.operator === "-") {
    return semanticExpressionKnownMultipleOf(expression.left, divisor) && semanticExpressionKnownMultipleOf(expression.right, divisor);
  }
  if (expression.operator === "*") {
    return semanticExpressionKnownMultipleOf(expression.left, divisor) || semanticExpressionKnownMultipleOf(expression.right, divisor);
  }
  return false;
}

function binaryIndexExpression(
  operator: string,
  left: SemanticExpression,
  right: SemanticExpression,
  span: SourceSpan,
): SemanticExpression {
  return {
    kind: "binary",
    operator,
    left,
    right,
    valueType: requiredSemanticValueType(semanticBinaryResultType(operator, expressionValueType(left) ?? "uint", expressionValueType(right)), `binary '${operator}'`, span),
    span,
  };
}

function semanticPointerArgumentMemoryRef(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticMemoryRef | undefined {
  const dereference = localPointerAliasDerefExpression(expression, scope, expression.span);
  if (dereference) return memoryRefFromExpression(dereference);
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
  if (expression.callee.name === "cudaGetDeviceFlags") return semanticDeviceFlagsCallResult(source, scope, span);
  if (isVibMinMaxCallName(expression.callee.name)) return semanticVibMinMaxCallResult(source, expression, scope, span);
  if (isModfCallName(expression.callee.name)) return semanticModfCallResult(source, expression, scope, span);
  if (isFrexpCallName(expression.callee.name)) return semanticFrexpCallResult(source, expression, scope, span);
  if (isRemquoCallName(expression.callee.name)) return semanticRemquoCallResult(source, expression, scope, span);
  return undefined;
}

function semanticDeviceFlagsCallResult(
  source: CudaLiteCallExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): { readonly sideEffects: readonly SemanticKernelIrOperation[]; readonly value: SemanticExpression } | undefined {
  const target = source.args[0] === undefined ? undefined : mathOutTargetExpressionFromSource(source.args[0], scope);
  if (!target || source.args.length !== 1) return undefined;
  return {
    sideEffects: [mathOutStoreOrAssignOperation(target, semanticUintLiteralExpression(0, span), span)],
    value: intNumberExpression(0, span),
  };
}

function semanticModfCallResult(
  source: CudaLiteCallExpression,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): { readonly sideEffects: readonly SemanticKernelIrOperation[]; readonly value: SemanticExpression } | undefined {
  const value = expression.args[0];
  const intpartTarget = source.args[1] === undefined ? undefined : mathOutTargetExpressionFromSource(source.args[1], scope);
  if (value === undefined || !intpartTarget || !semanticExpressionSideEffectFree(value)) return undefined;
  const temp = tempScalarSymbol("__bg.modf.value", span, "float");
  const tempValue = semanticSymbolExpression(temp, value.span);
  return {
    sideEffects: [
      { kind: "declare", target: temp, init: value, span },
      mathOutStoreOrAssignOperation(intpartTarget, unaryFloatCallExpression("__bg_modf_intpart", tempValue, expression.span), span),
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
  const quotientTarget = source.args[2] === undefined ? undefined : mathOutTargetExpressionFromSource(source.args[2], scope);
  if (dividend === undefined || divisor === undefined || divisor === 0 || !quotientTarget || !semanticExpressionSideEffectFree(dividend)) return undefined;
  const temp = tempScalarSymbol("__bg.remquo.dividend", span, "float");
  const tempValue = semanticSymbolExpression(temp, dividend.span);
  const divisorValue = numberExpression(divisor, expression.span);
  return {
    sideEffects: [
      { kind: "declare", target: temp, init: dividend, span },
      mathOutStoreOrAssignOperation(quotientTarget, binaryIntCallExpression("__bg_remquo_quotient", tempValue, divisorValue, expression.span), span),
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
  const sinTarget = source.args[1] === undefined ? undefined : mathOutTargetExpressionFromSource(source.args[1], scope);
  const cosTarget = source.args[2] === undefined ? undefined : mathOutTargetExpressionFromSource(source.args[2], scope);
  if (!value || !sinTarget || !cosTarget) return undefined;
  const angle = isSincosPiCallName(expression.callee.name)
    ? multiplyFloatExpressions(numberExpression(Math.PI, value.span), value, value.span)
    : value;
  const temp = tempScalarSymbol("__bg.sincos.angle", span, "float");
  const tempValue = semanticSymbolExpression(temp, angle.span);
  return {
    kind: "block",
    body: [
      { kind: "declare", target: temp, init: angle, span },
      mathOutStoreOrAssignOperation(sinTarget, mathCallExpression("sin", tempValue, value.span), span),
      mathOutStoreOrAssignOperation(cosTarget, mathCallExpression("cos", tempValue, value.span), span),
    ],
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
  const name = `${prefix}.${span.start}.${span.end}`;
  return {
    id: createSemanticSymbolId("temporary", name, span),
    name,
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

function pointerAliasValueExpression(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
  encodeNull = false,
): SemanticExpression | undefined {
  if (expression.kind === "identifier") {
    const symbol = scope.get(expression.name);
    if (symbol?.kind === "shared" && !symbol.pointer && symbol.dimensions.length > 0) return undefined;
  }
  if (isDirectSharedPointerAddress(expression, scope)) return undefined;
  const alias = localPointerAliasForInitializer(expression, scope);
  const scalar = semanticPointerAliasScalarIndex(alias, span);
  if (!scalar) return undefined;
  const root = semanticSymbolForMemoryId(scope, scalar.root);
  if (!root || !semanticPointerAliasAddressSpaceSupported(root.addressSpace)) return undefined;
  const aliasValueType = pointerAliasTargetValueType(expression, scope) ?? root.valueType;
  return {
    kind: "index",
    target: semanticSymbolExpression(root, span),
    index: !encodeNull || scalar.valid === undefined
      ? scalar.index
      : {
          kind: "conditional",
          condition: scalar.valid,
          consequent: scalar.index,
          alternate: { kind: "literal", literalKind: "number", value: 0xffffffff, valueType: "uint", span },
          valueType: "uint",
          span,
        },
    valueType: requiredSemanticValueType(aliasValueType, "pointer alias value", span),
    addressSpace: root.addressSpace,
    ...(scalar.scalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
    ...(scalar.unitBytes === undefined ? {} : { pointerBaseUnitBytes: scalar.unitBytes }),
    ...optionalPackedByteLanes(pointerAliasPackedByteLanes(expression, root, scope)),
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
      valueType: requiredSemanticValueType(expressionValueType(target), "math output assignment", span),
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
  if (valueType === undefined || valueType === "void") return undefined;
  return {
    baseId: semanticMemoryIdFromSymbol(parts.base.id),
    base: parts.base.name,
    addressSpace: parts.base.addressSpace,
    valueType,
    ...(expression.kind === "member" ? optionalContainerValueType(expressionValueType(expression.object)) : {}),
    ...(expression.kind === "index" ? optionalContainerValueType(expressionValueType(expression.target)) : {}),
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
    const key = `${semanticIdKey(ref.baseId)}:${ref.addressSpace}:${ref.span.start}:${ref.span.end}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(ref);
  }
  return out;
}

function symbolForParam(param: CudaLiteParam): CudaLiteSemanticSymbol {
  return {
    id: createSemanticSymbolId("param", param.name, param.span),
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
    id: createSemanticSymbolId("constant", constant.name, constant.span),
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
    id: createSemanticSymbolId("device-global", global.name, global.span),
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
    id: createSemanticSymbolId("texture", texture.name, texture.span),
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

function symbolForExternalPool(name: string, span: SourceSpan): CudaLiteSemanticSymbol {
  return {
    id: createSemanticSymbolId("external-pool", name, span),
    name,
    kind: "external-pool",
    valueType: "devicepool",
    pointer: true,
    constant: false,
    dimensions: [],
    addressSpace: "pool",
    span,
  };
}

function semanticSymbolForDim3(name: string, span: SourceSpan): CudaLiteSemanticSymbol {
  return {
    id: createSemanticSymbolId("dim3", name, span),
    name,
    kind: "local",
    valueType: "uint3",
    pointer: false,
    constant: false,
    dimensions: [],
    addressSpace: "local",
    span,
  };
}

function symbolForFunction(
  fn: CudaLiteDeviceFunction,
  environment: SemanticEnvironment,
  params: readonly CudaLiteSemanticSymbol[],
): CudaLiteSemanticFunction {
  const scope = createSemanticLexicalScope(environment);
  for (const [index, param] of params.entries()) scope.set(fn.params[index]!.name, param);
  return {
    id: createSemanticFunctionId(fn.name, fn.span),
    name: fn.name,
    returnType: fn.returnType,
    params,
    body: lowerStatements(fn.body, scope),
    span: fn.span,
  };
}

function semanticFunctionSignature(
  fn: CudaLiteDeviceFunction,
  environment: SemanticEnvironment,
): CudaLiteSemanticFunction {
  return {
    id: createSemanticFunctionId(fn.name, fn.span),
    name: fn.name,
    returnType: fn.returnType,
    params: fn.params.map((param) => symbolForFunctionParam(param, fn.name, environment.resolveSymbol(param.name) !== undefined)),
    body: [],
    span: fn.span,
  };
}

function symbolForFunctionDeclaration(fn: CudaLiteDeviceFunction): CudaLiteSemanticSymbol {
  return {
    id: createSemanticSymbolId("function", fn.name, fn.span),
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
    id: createSemanticSymbolId("function", fn.name, fn.span),
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

function symbolForLaunchableEntry(entry: CudaLiteSemanticLaunchableEntry): CudaLiteSemanticSymbol {
  return {
    id: semanticSymbolIdFromFunction(entry.id),
    name: entry.name,
    kind: "function",
    valueType: "void",
    pointer: false,
    constant: true,
    dimensions: [],
    addressSpace: "function",
    span: entry.span,
  };
}

function symbolForFunctionParam(param: CudaLiteParam, functionName: string, collidesWithGlobal: boolean): CudaLiteSemanticSymbol {
  const symbol = symbolForParam(param);
  if (symbol.addressSpace === "texture" || symbol.addressSpace === "surface") return symbol;
  if (symbol.pointer && symbol.addressSpace === "storage" && symbol.valueType === "uchar" && collidesWithGlobal) {
    const name = `__bg_param_${functionName}_${param.name}_${param.span.start}`;
    return { ...symbol, id: createSemanticSymbolId("function-param", name, param.span), name };
  }
  if (symbol.pointer) return { ...symbol, pointerMayBeNull: true };
  return { ...symbol, addressSpace: "local" };
}

function symbolForVar(
  statement: CudaLiteVarDecl,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): CudaLiteSemanticSymbol {
  const pointerAlias = statement.pointer ? localPointerAliasForInitializer(statement.init, scope) : undefined;
  const matrixTile = statement.matrixTile ? resolveMatrixTileSpec(statement.matrixTile) : undefined;
  return {
    id: createSemanticSymbolId(statement.storage === "shared" ? "shared" : "local", statement.name, statement.span),
    name: statement.name,
    kind: statement.storage === "shared" ? "shared" : "local",
    valueType: statement.valueType,
    pointer: statement.pointer,
    ...(statement.packedByteLanes === undefined ? {} : { packedByteLanes: statement.packedByteLanes }),
    ...(statement.dynamicShared === true ? { dynamicShared: true } : {}),
    ...(pointerAlias === undefined ? {} : pointerAlias),
    ...(statement.init === undefined ? {} : { init: lowerExpression(statement.init, scope) }),
    constant: statement.constant ?? false,
    dimensions: matrixTile ? [Math.max(1, totalElements(statement.dimensions)) * matrixTileElementCount(matrixTile)] : statement.dimensions,
    ...(matrixTile === undefined ? {} : { matrixTile, matrixTileArrayDimensions: statement.dimensions }),
    addressSpace: statement.storage,
    span: statement.span,
  };
}

function localPointerAliasForInitializer(
  expression: CudaLiteExpression | undefined,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): Pick<CudaLiteSemanticSymbol, "pointerRoot" | "pointerAddressSpace" | "pointerBaseIndices" | "pointerBaseIsScalarLane" | "pointerBaseUnitBytes" | "pointerValid" | "pointerSelection"> | undefined {
  if (!expression) return undefined;
  if (expression.kind === "cast" && expression.pointer) {
    const alias = localPointerAliasForInitializer(expression.expression, scope);
    const sourceType = pointerAliasTargetValueType(expression.expression, scope);
    const targetBytes = expression.pointerElementBytes ?? sizeofCudaType(expression.valueType);
    const rootType = alias?.pointerRoot === undefined ? undefined : semanticSymbolForMemoryId(scope, alias.pointerRoot)?.valueType;
    const rootBytes = rootType === undefined ? undefined : sizeofCudaType(rootType);
    if (alias && targetBytes !== undefined && rootBytes !== undefined && targetBytes >= rootBytes && targetBytes % rootBytes === 0) {
      const { pointerBaseUnitBytes: _pointerBaseUnitBytes, ...rest } = alias;
      const scale = targetBytes / rootBytes;
      const scalarLane = isCudaVectorType(expression.valueType) && cudaVectorScalarType(expression.valueType) === rootType;
      return {
        ...rest,
        ...(scale === 1 ? {} : { pointerBaseUnitBytes: scale }),
        ...(scalarLane ? { pointerBaseIsScalarLane: true } : {}),
      };
    }
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
    const sourceBytes = sourceType === undefined ? undefined : sizeofCudaType(sourceType);
    if (
      alias?.pointerBaseIndices?.length === 1 &&
      alias.pointerAddressSpace === "local" &&
      sourceBytes !== undefined && targetBytes !== undefined &&
      sourceBytes > targetBytes && sourceBytes % targetBytes === 0
    ) {
      return {
        ...alias,
        pointerBaseIndices: [multiplyIndexExpression(alias.pointerBaseIndices[0]!, sourceBytes / targetBytes, expression.span)],
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
      alternate.pointerRoot === undefined ||
      !semanticIdsEqual(root, alternate.pointerRoot) ||
      addressSpace !== alternate.pointerAddressSpace ||
      consequent.pointerBaseIndices?.length !== 1 ||
      alternate.pointerBaseIndices?.length !== 1 ||
      consequent.pointerBaseIsScalarLane !== alternate.pointerBaseIsScalarLane
      || consequent.pointerBaseUnitBytes !== alternate.pointerBaseUnitBytes
    ) {
      if (
        semanticPointerAliasAddressSpaceSupported(consequent.pointerAddressSpace) &&
        consequent.pointerAddressSpace === alternate.pointerAddressSpace &&
        consequent.pointerBaseIsScalarLane === alternate.pointerBaseIsScalarLane &&
        consequent.pointerBaseUnitBytes === alternate.pointerBaseUnitBytes
      ) {
        return { pointerSelection: { condition, consequent, alternate } };
      }
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
    if (left?.pointerSelection) {
      const offset = lowerExpression(expression.right, scope);
      return offsetSemanticPointerAliasSelection(left, offset, expression.operator, pointerAliasTargetValueType(expression.left, scope), expression.span);
    }
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
        ...(left.pointerValid === undefined ? {} : { pointerValid: left.pointerValid }),
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
          ...(right.pointerValid === undefined ? {} : { pointerValid: right.pointerValid }),
        };
      }
    }
  }
  if (expression.kind === "identifier") {
    const root = scope.get(expression.name);
    if (root?.pointerSelection) return { pointerSelection: root.pointerSelection };
    if (root?.pointerRoot && semanticPointerAliasAddressSpaceSupported(root.pointerAddressSpace) && root.pointerBaseIndices?.length === 1) {
      return {
        pointerRoot: root.pointerRoot,
        pointerAddressSpace: root.pointerAddressSpace,
        pointerBaseIndices: root.pointerBaseIndices,
        ...(root.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
        ...(root.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: root.pointerBaseUnitBytes }),
        ...(root.pointerValid === undefined ? {} : { pointerValid: root.pointerValid }),
      };
    }
    if (root?.kind === "param" && root.pointer && root.addressSpace === "storage") {
      return {
        pointerRoot: semanticMemoryIdFromSymbol(root.id),
        pointerAddressSpace: root.addressSpace,
        pointerBaseIndices: [zeroExpression(expression.span)],
        ...(root.pointerMayBeNull === true
          ? { pointerValid: { kind: "pointer-valid" as const, pointerId: root.id, pointer: root.name, valueType: "bool" as const, span: expression.span } }
          : {}),
      };
    }
    if ((root?.kind === "device-global" || root?.kind === "constant") && root.dimensions.length > 0) {
      return {
        pointerRoot: semanticMemoryIdFromSymbol(root.id),
        pointerAddressSpace: root.addressSpace,
        pointerBaseIndices: [zeroExpression(expression.span)],
      };
    }
    if (!root || (root.kind !== "local" && root.kind !== "shared") || root.pointer ||
      root.kind === "local" && root.dimensions.length !== 1 ||
      root.kind === "shared" && root.dimensions.length !== 1 && root.dynamicShared !== true) return undefined;
    return {
      pointerRoot: semanticMemoryIdFromSymbol(root.id),
      pointerAddressSpace: root.addressSpace,
      pointerBaseIndices: [zeroExpression(expression.span)],
    };
  }
  if (expression.kind === "index" && expression.target.kind === "identifier") {
    const target = scope.get(expression.target.name);
    const slot = staticPointerArrayIndex(expression.index);
    const alias = slot === undefined ? undefined : target?.pointerArrayAliases?.[slot];
    if (semanticPointerAliasComplete(alias)) return alias;
    if (slot === undefined && semanticPointerArrayAliasesComplete(target)) {
      const aliases = target!.pointerArrayAliases as readonly SemanticPointerAlias[];
      const index = lowerExpression(expression.index, scope);
      let selected = aliases.at(-1)!;
      for (let candidate = aliases.length - 2; candidate >= 0; candidate--) {
        selected = {
          pointerSelection: {
            condition: {
              kind: "binary",
              operator: "==",
              left: index,
              right: intNumberExpression(candidate, expression.index.span),
              valueType: "bool",
              span: expression.index.span,
            },
            consequent: aliases[candidate]!,
            alternate: selected,
          },
        };
      }
      return selected;
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
    pointerRoot: semanticMemoryIdFromSymbol(ref.root.id),
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

function isLocalPointerArray(statement: CudaLiteStatement): statement is Extract<CudaLiteStatement, { readonly kind: "var" }> {
  return statement.kind === "var" &&
    statement.storage === "local" &&
    statement.pointer &&
    statement.dimensions.length === 1 &&
    statement.init === undefined;
}

/**
 * A dynamic slot cannot be represented by a final alias snapshot: the slot
 * expression may execute after a helper call or an atomic side effect. Mark
 * the array as runtime state only when every visible assignment can become a
 * source-ordered storage rebind; static-only arrays keep the smaller alias
 * representation.
 */
function hasLaterDynamicStoragePointerArrayAssignment(
  name: string,
  statements: readonly CudaLiteStatement[],
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): boolean {
  let sawDynamic = false;
  const scan = (items: readonly CudaLiteStatement[]): boolean => {
    for (const statement of items) {
      if (statement.kind === "expr" && statement.expression.kind === "assignment" && statement.expression.operator === "=" &&
        statement.expression.left.kind === "index" && statement.expression.left.target.kind === "identifier" &&
        statement.expression.left.target.name === name) {
        const alias = localPointerAliasForInitializer(statement.expression.right, scope);
        const dynamicSlot = staticPointerArrayIndex(statement.expression.left.index) === undefined;
        if (
          !alias ||
          dynamicSlot && !semanticStoragePointerAlias(alias) ||
          !dynamicSlot && (!alias.pointerRoot || alias.pointerAddressSpace !== "storage" || alias.pointerBaseIndices?.length !== 1)
        ) return false;
        if (dynamicSlot) sawDynamic = true;
        continue;
      }
      if (statement.kind === "block" || statement.kind === "if") {
        if (!scan(statement.kind === "block" ? statement.body : [...statement.consequent, ...(statement.alternate ?? [])])) return false;
        continue;
      }
      if (statement.kind === "for" || statement.kind === "while" || statement.kind === "do-while") {
        if (!scan(statement.body)) return false;
      }
    }
    return true;
  };
  return scan(statements) && sawDynamic;
}

function semanticPointerAliasAddressSpaceSupported(addressSpace: SemanticAddressSpace | undefined): addressSpace is "local" | "shared" | "storage" | "constant" | "device-global" {
  return addressSpace === "local" || addressSpace === "shared" || addressSpace === "storage" || addressSpace === "constant" || addressSpace === "device-global";
}

function semanticStoragePointerAlias(alias: SemanticPointerAlias): boolean {
  if (alias.pointerSelection) {
    return semanticStoragePointerAlias(alias.pointerSelection.consequent) && semanticStoragePointerAlias(alias.pointerSelection.alternate);
  }
  return alias.pointerRoot !== undefined && alias.pointerAddressSpace === "storage" && alias.pointerBaseIndices?.length === 1;
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
    if (statement.kind === "for") {
      const loopScope = cloneSemanticScope(scope);
      if (statement.init?.kind === "var") {
        const init = symbolForVar(statement.init, loopScope);
        loopScope.set(init.name, init);
      }
      if (hasLaterLocalPointerAliasAssignment(name, statement.body, loopScope)) return true;
    }
    if ((statement.kind === "while" || statement.kind === "do-while") && hasLaterLocalPointerAliasAssignment(name, statement.body, scope)) return true;
  }
  return false;
}

function localPointerAliasAssignmentProfile(
  name: string,
  statements: readonly CudaLiteStatement[],
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  controlDependent = false,
): { readonly total: number; readonly controlDependent: boolean } {
  let total = 0;
  let hasControlDependentAssignment = false;
  const add = (profile: { readonly total: number; readonly controlDependent: boolean }): void => {
    total += profile.total;
    hasControlDependentAssignment ||= profile.controlDependent;
  };
  for (const statement of statements) {
    if (statement.kind === "expr") {
      const expression = statement.expression;
      if (expression.kind !== "assignment" || expression.operator !== "=" ||
        expression.left.kind !== "identifier" || expression.left.name !== name) continue;
      const alias = localPointerAliasForInitializer(expression.right, scope);
      if (alias !== undefined && semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace)) {
        total += 1;
        hasControlDependentAssignment ||= controlDependent;
      }
      continue;
    }
    if (statement.kind === "block") {
      add(localPointerAliasAssignmentProfile(name, statement.body, scope, controlDependent));
      continue;
    }
    if (statement.kind === "if") {
      add(localPointerAliasAssignmentProfile(name, statement.consequent, scope, true));
      add(localPointerAliasAssignmentProfile(name, statement.alternate ?? [], scope, true));
      continue;
    }
    if (statement.kind === "for") {
      const loopScope = cloneSemanticScope(scope);
      if (statement.init?.kind === "var") {
        const init = symbolForVar(statement.init, loopScope);
        loopScope.set(init.name, init);
      }
      add(localPointerAliasAssignmentProfile(name, statement.body, loopScope, controlDependent));
      continue;
    }
    if (statement.kind === "while" || statement.kind === "do-while") {
      add(localPointerAliasAssignmentProfile(name, statement.body, scope, controlDependent));
    }
  }
  return { total, controlDependent: hasControlDependentAssignment };
}

function offsetSemanticPointerAliasSelection(
  alias: SemanticPointerAlias,
  offset: SemanticExpression,
  operator: "+" | "-",
  aliasValueType: CudaLiteScalarType | undefined,
  span: SourceSpan,
): SemanticPointerAlias | undefined {
  if (alias.pointerSelection) {
    const consequent = offsetSemanticPointerAliasSelection(alias.pointerSelection.consequent, offset, operator, aliasValueType, span);
    const alternate = offsetSemanticPointerAliasSelection(alias.pointerSelection.alternate, offset, operator, aliasValueType, span);
    return consequent && alternate
      ? { pointerSelection: { condition: alias.pointerSelection.condition, consequent, alternate } }
      : undefined;
  }
  if (!alias.pointerRoot || !semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace) || alias.pointerBaseIndices?.length !== 1) return undefined;
  const scaled = pointerAliasOffsetForBaseUnit(alias, aliasValueType, offset, span);
  return {
    ...alias,
    pointerBaseIndices: [operator === "+"
      ? addIndexExpressions(alias.pointerBaseIndices[0]!, scaled, span)
      : subtractIndexExpressions(alias.pointerBaseIndices[0]!, scaled, span)],
  };
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
    if (!alias || !alias.pointerSelection && !semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace)) return false;
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
): SemanticKernelIrOperation | undefined {
  if (
    expression.kind !== "assignment" ||
    expression.operator !== "=" ||
    expression.left.kind !== "index" ||
    expression.left.target.kind !== "identifier"
  ) {
    return undefined;
  }
  const target = scope.get(expression.left.target.name);
  if (!target || target.kind !== "local" || !target.pointer || target.dimensions.length !== 1) return undefined;
  const slot = staticPointerArrayIndex(expression.left.index);
  const extent = target.dimensions[0];
  if (slot === undefined || extent === undefined || slot >= extent) return undefined;
  const alias = localPointerAliasForInitializer(expression.right, scope);
  if (!alias?.pointerRoot || !semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace) || alias.pointerBaseIndices?.length !== 1) return undefined;
  const aliases = Array.from({ length: extent }, (_, index) => target.pointerArrayAliases?.[index]);
  aliases[slot] = alias;
  scope.set(target.name, { ...target, pointerArrayAliases: aliases });
  if (target.pointerRuntimeState !== true) {
    return { kind: "expression", expression: zeroExpression(expression.span), span: expression.span };
  }
  return semanticPointerArrayRebindFromAlias(
    target,
    intNumberExpression(slot, expression.left.index.span),
    alias,
    scope,
    expression.span,
  );
}

function staticPointerArrayIndex(expression: CudaLiteExpression): number | undefined {
  if (expression.kind !== "number" || !Number.isInteger(expression.value) || expression.value < 0) return undefined;
  return expression.value;
}

function semanticPointerArrayAliasesComplete(symbol: CudaLiteSemanticSymbol | undefined): boolean {
  const extent = symbol?.dimensions.length === 1 ? symbol.dimensions[0] : undefined;
  return extent !== undefined && extent > 0 && symbol?.pointerArrayAliases?.length === extent &&
    symbol.pointerArrayAliases.every(semanticPointerAliasComplete);
}

function semanticPointerAliasComplete(alias: SemanticPointerAlias | undefined): boolean {
  if (alias?.pointerRoot !== undefined && semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace) && alias.pointerBaseIndices?.length === 1) return true;
  return alias?.pointerSelection !== undefined &&
    semanticPointerAliasComplete(alias.pointerSelection.consequent) &&
    semanticPointerAliasComplete(alias.pointerSelection.alternate);
}

function mergeBlockLocalPointerAliases(
  parent: Map<string, CudaLiteSemanticSymbol>,
  child: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): void {
  for (const [name, current] of parent) {
    if (!current.pointer || current.dimensions.length > 0) continue;
    const next = child.get(name);
    if (next?.pointerRuntimeState === true && sameSymbolDeclaration(current, next)) {
      parent.set(name, next);
      continue;
    }
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
    if ((left?.pointerRuntimeState === true || right?.pointerRuntimeState === true) &&
      left && right && sameSymbolDeclaration(current, left) && sameSymbolDeclaration(current, right)) {
      parent.set(name, {
        ...semanticSymbolWithoutPointerAlias(current),
        pointerRuntimeState: true,
      });
      continue;
    }
    if (
      !left?.pointerRoot ||
      !right?.pointerRoot ||
      !sameSymbolDeclaration(current, left) ||
      !sameSymbolDeclaration(current, right) ||
      !semanticIdsEqual(left.pointerRoot, right.pointerRoot) ||
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

function semanticSymbolForMemoryId(
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  memoryId: SemanticMemoryId,
): CudaLiteSemanticSymbol | undefined {
  if (scope instanceof SemanticLexicalScope) return scope.resolveMemorySymbol(memoryId);
  for (const symbol of scope.values()) {
    if (semanticIdsEqual(semanticMemoryIdFromSymbol(symbol.id), memoryId)) return symbol;
  }
  return undefined;
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
    const root = semanticSymbolForMemoryId(scope, target.pointerRoot);
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
    if (!root || (root.kind !== "local" && (!allowShared || root.kind !== "shared" && root.kind !== "constant" && root.kind !== "device-global")) ||
      root.dimensions.length === 0 || root.pointer && root.kind !== "device-global") return undefined;
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
  if (alias?.pointerSelection) {
    const index = lowerExpression(expression.index, scope);
    return semanticPointerSelectionValue(
      alias.pointerSelection,
      (selected) => semanticPointerAliasIndexedValue(selected, index, expression.target, scope, expression.span),
      expression.span,
    );
  }
  if (!alias?.pointerRoot || !semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace) || !alias.pointerBaseIndices || alias.pointerBaseIndices.length !== 1) return undefined;
  const root = semanticSymbolForMemoryId(scope, alias.pointerRoot);
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
    valueType: requiredSemanticValueType(aliasValueType, "pointer alias index", expression.span),
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
  const runtimePointerArray = semanticRuntimePointerArrayDerefExpression(expression, scope, span);
  if (runtimePointerArray) return runtimePointerArray;
  const directPackedLocal = expression.kind === "cast" && expression.pointer &&
    expression.expression.kind === "unary" && expression.expression.operator === "&" &&
    expression.expression.argument.kind === "identifier"
    ? scope.get(expression.expression.argument.name)
    : undefined;
  const alias = directPackedLocal?.kind === "local" && directPackedLocal.dimensions.length === 0 && directPackedLocal.packedByteLanes !== undefined
    ? {
        pointerRoot: semanticMemoryIdFromSymbol(directPackedLocal.id),
        pointerAddressSpace: directPackedLocal.addressSpace,
        pointerBaseIndices: [zeroExpression(expression.span)],
      }
    : localPointerAliasForInitializer(expression, scope);
  if (alias?.pointerSelection) {
    return semanticPointerSelectionValue(
      alias.pointerSelection,
      (selected) => semanticPointerAliasIndexedValue(selected, zeroExpression(span), expression, scope, span),
      span,
    );
  }
  if (!alias?.pointerRoot || !semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace) || !alias.pointerBaseIndices || alias.pointerBaseIndices.length !== 1) return undefined;
  const root = semanticSymbolForMemoryId(scope, alias.pointerRoot);
  if (!root) return undefined;
  return {
    kind: "index",
    target: semanticSymbolExpression(root, expression.span),
    index: alias.pointerBaseIndices[0]!,
    valueType: requiredSemanticValueType(pointerAliasTargetValueType(expression, scope) ?? root.valueType, "pointer dereference", span),
    addressSpace: root.addressSpace,
    ...(alias.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
    ...(alias.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: alias.pointerBaseUnitBytes }),
    ...optionalPackedByteLanes(pointerAliasPackedByteLanes(expression, root, scope)),
    span,
  };
}

/**
 * Dynamic pointer-array slots keep their storage root and base index in
 * runtime state. Represent their dereference as a local memory reference so
 * the reference evaluator and WGSL lowering can route through that state,
 * rather than snapshotting the aliases as a conditional expression.
 */
function semanticRuntimePointerArrayDerefExpression(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticExpression | undefined {
  if (expression.kind !== "index" || expression.target.kind !== "identifier") return undefined;
  const target = scope.get(expression.target.name);
  if (
    !target?.pointerRuntimeState ||
    target.kind !== "local" ||
    !target.pointer ||
    target.dimensions.length !== 1 ||
    target.valueType === undefined ||
    target.valueType === "void"
  ) return undefined;
  return {
    kind: "index",
    target: semanticSymbolExpression(target, expression.target.span),
    index: lowerExpression(expression.index, scope),
    valueType: target.valueType,
    addressSpace: "local",
    span,
  };
}

function semanticPointerSelectionValue(
  selection: SemanticPointerSelection,
  value: (alias: SemanticPointerAlias) => SemanticExpression | undefined,
  span: SourceSpan,
): SemanticExpression | undefined {
  const consequent = selection.consequent.pointerSelection
    ? semanticPointerSelectionValue(selection.consequent.pointerSelection, value, span)
    : value(selection.consequent);
  const alternate = selection.alternate.pointerSelection
    ? semanticPointerSelectionValue(selection.alternate.pointerSelection, value, span)
    : value(selection.alternate);
  if (!consequent || !alternate || expressionValueType(consequent) !== expressionValueType(alternate)) return undefined;
  return {
    kind: "conditional",
    condition: selection.condition,
    consequent,
    alternate,
    valueType: requiredSemanticExpressionType(expressionValueType(consequent), "pointer selection", span),
    span,
  };
}

function semanticPointerAliasIndexedValue(
  alias: SemanticPointerAlias,
  index: SemanticExpression,
  source: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
  span: SourceSpan,
): SemanticExpression | undefined {
  if (!alias.pointerRoot || !semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace) || alias.pointerBaseIndices?.length !== 1) return undefined;
  const root = semanticSymbolForMemoryId(scope, alias.pointerRoot);
  if (!root) return undefined;
  const aliasValueType = pointerAliasTargetValueType(source, scope) ?? root.valueType;
  return {
    kind: "index",
    target: semanticSymbolExpression(root, span),
    index: addIndexExpressions(
      alias.pointerBaseIndices[0]!,
      pointerAliasElementOffset(aliasValueType, root.valueType, index, span, alias.pointerBaseUnitBytes),
      span,
    ),
    valueType: requiredSemanticValueType(aliasValueType, "indexed pointer alias", span),
    addressSpace: root.addressSpace,
    ...(alias.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
    ...(alias.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: alias.pointerBaseUnitBytes }),
    ...optionalPackedByteLanes(pointerAliasPackedByteLanes(source, root, scope)),
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
  if (!left || !right || !semanticIdsEqual(left.root, right.root) || left.unitBytes !== right.unitBytes) return undefined;
  const difference = semanticCastExpression(
    subtractIndexExpressions(left.index, right.index, expression.span),
    "int",
    expression.span,
  );
  return left.unitBytes === undefined || left.unitBytes === 1
    ? difference
    : {
        kind: "binary",
        operator: "/",
        left: difference,
        right: intNumberExpression(left.unitBytes, expression.span),
        valueType: "int",
        span: expression.span,
      };
}

function semanticCastExpression(
  expression: SemanticExpression,
  valueType: Exclude<CudaLiteScalarType, "void">,
  span: SourceSpan,
): SemanticExpression {
  return expressionValueType(expression) === valueType
    ? expression
    : { kind: "cast", valueType, pointer: false, expression, span };
}

function localPointerAliasComparisonExpression(
  expression: Extract<CudaLiteExpression, { readonly kind: "binary" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticExpression | undefined {
  if (!POINTER_ORDER_OPERATORS.has(expression.operator)) return undefined;
  if (expression.operator === "==" || expression.operator === "!=") {
    const leftRuntime = localRuntimePointerSymbol(expression.left, scope);
    const rightRuntime = localRuntimePointerSymbol(expression.right, scope);
    if (leftRuntime && isNullPointerLiteral(expression.right)) {
      return runtimePointerNullComparisonExpression(leftRuntime, expression.operator, expression.span);
    }
    if (rightRuntime && isNullPointerLiteral(expression.left)) {
      return runtimePointerNullComparisonExpression(rightRuntime, expression.operator, expression.span);
    }
    const leftAlias = localPointerAliasScalarIndex(expression.left, scope);
    const rightAlias = localPointerAliasScalarIndex(expression.right, scope);
    if (leftAlias && isNullPointerLiteral(expression.right)) return pointerNullComparisonExpression(leftAlias.valid, expression.operator, expression.span);
    if (rightAlias && isNullPointerLiteral(expression.left)) return pointerNullComparisonExpression(rightAlias.valid, expression.operator, expression.span);
  }
  const left = localPointerAliasScalarIndex(expression.left, scope);
  const right = localPointerAliasScalarIndex(expression.right, scope);
  if (!left || !right || !semanticIdsEqual(left.root, right.root)) return undefined;
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

function localRuntimePointerSymbol(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): CudaLiteSemanticSymbol | undefined {
  if (expression.kind === "cast" && expression.pointer) return localRuntimePointerSymbol(expression.expression, scope);
  if (expression.kind !== "identifier") return undefined;
  const symbol = scope.get(expression.name);
  return symbol?.kind === "local" && symbol.pointerRuntimeState === true ? symbol : undefined;
}

function runtimePointerNullComparisonExpression(
  pointer: CudaLiteSemanticSymbol,
  operator: "==" | "!=",
  span: SourceSpan,
): SemanticExpression {
  const valid: SemanticExpression = {
    kind: "pointer-valid",
    pointerId: pointer.id,
    pointer: pointer.name,
    valueType: "bool",
    span,
  };
  return operator === "!=" ? valid : negateExpression(valid, span);
}

function localPointerAliasScalarIndex(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): { readonly root: SemanticMemoryId; readonly index: SemanticExpression; readonly valid?: SemanticExpression; readonly unitBytes?: number } | undefined {
  return semanticPointerAliasScalarIndex(localPointerAliasForInitializer(expression, scope), expression.span);
}

function semanticPointerAliasScalarIndex(
  alias: SemanticPointerAlias | undefined,
  span: SourceSpan,
): { readonly root: SemanticMemoryId; readonly index: SemanticExpression; readonly valid?: SemanticExpression; readonly unitBytes?: number; readonly scalarLane?: boolean } | undefined {
  if (alias?.pointerRoot && semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace) && alias.pointerBaseIndices?.length === 1) {
    return {
      root: alias.pointerRoot,
      index: alias.pointerBaseIndices[0]!,
      ...(alias.pointerValid === undefined ? {} : { valid: alias.pointerValid }),
      ...(alias.pointerBaseUnitBytes === undefined ? {} : { unitBytes: alias.pointerBaseUnitBytes }),
      ...(alias.pointerBaseIsScalarLane === true ? { scalarLane: true } : {}),
    };
  }
  if (!alias?.pointerSelection) return undefined;
  const consequent = semanticPointerAliasScalarIndex(alias.pointerSelection.consequent, span);
  const alternate = semanticPointerAliasScalarIndex(alias.pointerSelection.alternate, span);
  if (!consequent || !alternate || !semanticIdsEqual(consequent.root, alternate.root) || consequent.valid || alternate.valid ||
    consequent.unitBytes !== alternate.unitBytes || consequent.scalarLane !== alternate.scalarLane) return undefined;
  return {
    root: consequent.root,
    index: {
      kind: "conditional",
      condition: alias.pointerSelection.condition,
      consequent: consequent.index,
      alternate: alternate.index,
      valueType: expressionValueType(consequent.index) ?? expressionValueType(alternate.index) ?? "int",
      span,
    },
    ...(consequent.unitBytes === undefined ? {} : { unitBytes: consequent.unitBytes }),
    ...(consequent.scalarLane === true ? { scalarLane: true } : {}),
  };
}

function semanticSymbolExpression(symbol: CudaLiteSemanticSymbol, span: SourceSpan): Extract<SemanticExpression, { readonly kind: "symbol" }> {
  return {
    kind: "symbol",
    id: symbol.id,
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
    valueType: requiredSemanticValueType(semanticBinaryResultType("+", expressionValueType(left), expressionValueType(right)), "index addition", span),
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
    valueType: requiredSemanticValueType(semanticBinaryResultType("-", expressionValueType(left), expressionValueType(right)), "index subtraction", span),
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
    valueType: requiredSemanticValueType(semanticBinaryResultType("*", expressionValueType(left), "int"), "index multiplication", span),
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
  if (property === "size" && isCudaVectorType(objectType)) return "int";
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
  if (expression.kind === "cast" && expression.pointer) return expressionAddressSpace(expression.expression);
  return "unknown";
}

function semanticBinaryResultValueType(
  operator: string,
  left: SemanticExpression,
  right: SemanticExpression,
): CudaLiteScalarType | undefined {
  return semanticBinaryResultType(operator, expressionValueType(left), expressionValueType(right));
}

function expressionValueType(expression: SemanticExpression | undefined): CudaLiteScalarType | undefined {
  if (!expression || expression.kind === "initializer") return undefined;
  return "valueType" in expression ? expression.valueType : undefined;
}

function semanticIntrinsicReturnType(name: string | undefined, args: readonly SemanticExpression[]): CudaLiteScalarType | undefined {
  if (name === undefined) return undefined;
  if (
    name === "printf" ||
    name === "assert" ||
    SEMANTIC_NOOP_CALLS.has(name) ||
    BARRIER_CALLS.has(name) ||
    FENCE_CALLS.has(name) ||
    isCudaCpAsyncFenceCall(name)
  ) return "void";
  if (isHostManagedRuntimeNoopCall(name)) return "int";
  if (name === "__half2_as_uint" || name === "__bfloat162_as_uint" || name === "__nv_bfloat162_as_uint") return "uint";
  const curandVectorReturnType = SEMANTIC_CURAND_VECTOR_RETURN_TYPES.get(name);
  if (curandVectorReturnType) return curandVectorReturnType;
  if (name === "curand" || name === "curand_poisson") return "uint";
  if (name === "curand_uniform" || name === "curand_uniform_double" || name === "curand_normal" || name === "curand_normal_double" ||
    name === "curand_log_normal" || name === "curand_log_normal_double") return "float";
  if (name === "__low2float" || name === "__high2float") return "float";
  if (name === "__low2bfloat16" || name === "__high2bfloat16") return "bf16";
  if (name === "vec_at") {
    const vectorType = expressionValueType(args[0]);
    return isCudaVectorType(vectorType) ? cudaVectorScalarType(vectorType) : undefined;
  }
  if (name === "cg::reduce" || name === "cg::inclusive_scan" || name === "cg::exclusive_scan") return expressionValueType(args[1]);
  if (name === "make_cuComplex" || name === "make_cuFloatComplex" || name === "make_cuDoubleComplex" ||
    name === "cuCaddf" || name === "cuCsubf" || name === "cuCmulf" || name === "cuCdivf" || name === "cuConjf" ||
    name === "cuCadd" || name === "cuCsub" || name === "cuCmul" || name === "cuCdiv" || name === "cuConj") return "complex64";
  if (name === "cuCabsf" || name === "cuCrealf" || name === "cuCimagf" || name === "cuCabs" || name === "cuCreal" || name === "cuCimag") return "float";
  if (isSemanticBf162OverloadedVectorCall(name) && args.some((arg) => expressionValueType(arg) === "bf162")) {
    const overloaded = semanticBf162VectorReturnType(name);
    if (overloaded) return overloaded;
  }
  const half2VectorReturnType = semanticHalf2VectorReturnType(name);
  if (half2VectorReturnType) return half2VectorReturnType;
  const bf162VectorReturnType = semanticBf162VectorReturnType(name);
  if (bf162VectorReturnType) return bf162VectorReturnType;
  const vectorConstructorType = cudaVectorConstructorType(name);
  if (vectorConstructorType) return vectorConstructorType;
  const vectorMathReturnType = semanticVectorMathReturnType(name, args);
  if (vectorMathReturnType) return vectorMathReturnType;
  const bfloat16ReturnType = cudaBfloat16IntrinsicReturnType(name, args.some((arg) => expressionValueType(arg) === "bf16"));
  if (bfloat16ReturnType) return bfloat16ReturnType;
  if (name === "__half2float") return "float";
  if (
    name === "__float2half" || name.startsWith("__float2half_") ||
    name.startsWith("__int2half_") || name.startsWith("__uint2half_") ||
    name.startsWith("__short2half_") || name.startsWith("__ushort2half_") ||
    name === "__short_as_half" || name === "__ushort_as_half"
  ) return "half";
  if (name === "__half_as_short") return "int";
  if (name === "__half_as_ushort") return "uint";
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
  const mathReturnType = semanticMathCallReturnType(name, args);
  if (mathReturnType) return mathReturnType;
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
  const indexesVectorValue = target.kind === "index" ||
    expressionAddressSpace(target) === "local" && !(target.kind === "cast" && target.pointer);
  return indexesVectorValue && targetType !== undefined && isCudaVectorType(targetType)
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

function numberLiteralType(raw: string): Exclude<CudaLiteScalarType, "void"> {
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

function requiredSemanticValueType(
  valueType: CudaLiteScalarType | undefined,
  owner: string,
  span: SourceSpan,
): Exclude<CudaLiteScalarType, "void"> {
  return requireSemanticValueType(valueType, owner, span);
}

function requiredSemanticExpressionType(
  valueType: CudaLiteScalarType | undefined,
  owner: string,
  span: SourceSpan,
): CudaLiteScalarType {
  if (valueType !== undefined) return valueType;
  return requireSemanticValueType(valueType, owner, span);
}

function normalizeWorkgroupSize(value: readonly [number, number, number]): [number, number, number] {
  return [normalizeDimension(value[0]), normalizeDimension(value[1]), normalizeDimension(value[2])];
}

function normalizeDimension(value: number): number {
  return Number.isFinite(value) && Number.isInteger(value) && value > 0 ? value : 1;
}
