import {
  defineWgslKernelProgram,
  type WgslKernelProgram,
} from "@unlocalhosted/browsergrad-kernels";
import { collectKernelLaunchCallees, walkCudaLiteExpressions } from "./ast_queries.js";
import { expressionName, rootIdentifier } from "./analyzer.js";
import { CUDA_CACHE_HINT_LOADS, CUDA_CACHE_HINT_STORES, CUDA_INTRINSICS_BY_NAME } from "./intrinsics.js";
import {
  type MatrixTileLayout,
  type MatrixTileResolvedSpec,
  isMatrixTileByteValueType,
  matrixTileElementCount,
  matrixTileReference,
  normalizeMatrixTileLayout,
  resolveMatrixTileSpec,
  wmmaBuiltinName,
} from "./matrix_tiles.js";
import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";
import { classifyInlineAsm, inlineAsmSupportedList } from "./ptx_tile_ops.js";
import { alignofCudaType, sizeofCudaType } from "./type_layout.js";
import {
  cudaVectorConstructorType,
  cudaVectorFieldIndex,
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
  type CudaLiteVectorType,
} from "./vector_types.js";
import {
  collectLocalArrays,
  collectLocalNames,
  collectLocalPointerArrayRoots,
  collectLocalPointerHandles,
  collectLocalValueTypes,
  collectPointerAliases,
  expressionChildren,
  isLocalPointerArrayDecl,
  isPointerIdentityCall,
  pointerAliasDeclarationFor,
  poolPointerForAllocationCall,
  structuredPointerHandleRoots,
  zeroExpression,
  type PointerAlias,
} from "./wgsl_ir_analysis.js";
import {
  emitLocalArrayInitializer,
  emitLocalArrayType,
  emitSharedAddressIndex,
  functionBodyHasReturn,
  isExternalConstantBinding,
  isSurfaceParam,
  textureBindings,
} from "./wgsl_declarations.js";
import {
  NULL_DEVICE_POINTER_BUFFER,
  devicePointerArgumentParts as resolveDevicePointerArgumentParts,
  emitDevicePointerArgument as resolveDevicePointerArgument,
  type DevicePointerParts,
  type WgslDevicePointerCallbacks,
} from "./wgsl_device_pointers.js";
import {
  collectAssignedNames,
  constantBooleanExpression,
  countReturns,
  expressionContainsSubgroupCall,
  isBarrierCall,
  replaceExpressionNode,
  splitIfTrailingBreak,
  splitIfTrailingVoidReturn,
  statementContainsBarrier,
  statementContainsVoidReturn,
  statementContainsSubgroupCall,
  type IfTrailingBreak,
  type IfTrailingReturn,
} from "./wgsl_control_analysis.js";
import {
  effectiveF16Mode,
  effectiveSubgroupMode,
  rewriteF16BindingsToF32,
  rewriteF16WgslToF32,
} from "./wgsl_feature_usage.js";
import {
  UNIFORM_PARAMS_NAME,
  collectCooperativeGroups,
  createEmitContext,
  deviceFunctionLinkName,
  resolveDeviceFunctionForCall,
  type EmitContext,
  type EmitKernelIrWgslOptions,
} from "./wgsl_context.js";
import {
  cooperativeGroupForParam,
  cooperativeReduceDeviceFunctionName,
  emitCooperativeGroupCall,
  emitLocalLinearRank,
  emitScalarWarpReduceCall,
  emitScalarWarpReduceHelper,
  emitScalarWarpReduceWorkgroupStorage,
  emitScalarWarpShuffleCall,
  emitScalarWarpShuffleHelper,
  emitScalarWarpShuffleWorkgroupStorage,
  emitVectorCooperativeReduceHelper,
  emitVectorCooperativeReduceWorkgroupStorage,
  type WgslCooperativeCallbacks,
} from "./wgsl_cooperative.js";
import { emitKernelEntryPoint, emitKernelModulePrelude } from "./wgsl_module.js";
import { safeWgslIdentifier } from "./wgsl_names.js";

export type { EmitKernelIrWgslOptions } from "./wgsl_context.js";
import {
  isAtomicCasCallName,
  isAtomicExchangeCallName,
  isAtomicReturnCallName,
} from "./wgsl_atomic_helpers.js";
import {
  emitAtomicCall as emitAtomicCallImpl,
  emitAtomicCasCall as emitAtomicCasCallImpl,
  type WgslAtomicCallbacks,
} from "./wgsl_atomics.js";
import {
  emitDevicePointerHelpers,
  pointerReadHelperName,
  pointerWriteHelperName,
} from "./wgsl_pointer_helpers.js";
import {
  emitPoolAssignment,
  emitPoolRead,
  poolAccessForIndex,
  poolPointerExpressionInfo,
  poolZeroIndex,
} from "./wgsl_pool_access.js";
import {
  castExpressionToVectorScalar,
  cudaScalarWgslType,
  emitDeviceGlobalVectorFlatRead,
  emitStorageCarrierAsU32,
  emitConstantPointerRead,
  emitDeviceGlobalPointerRead,
  emitDeviceGlobalPointerWrite,
  emitLocalPointerRead,
  emitLocalPointerWrite,
  emitPointerVectorFlatRead,
  emitPointerStorageRead,
  emitPointerStorageWrite,
  emitSharedPointerRead,
  emitSharedPointerWrite,
  emitVectorConstructor,
  emitVectorSplat,
  emitVectorStorageFieldWrite,
  emitVectorStorageFieldWriteAt,
  emitVectorStorageRead,
  emitVectorStorageReadAt,
  emitVectorStorageWrite,
  emitVectorStorageWriteAt,
  vectorFieldName,
  vectorStorageBase,
  wgslElementByteSize,
  wgslScalar,
  zeroValue,
  type StorageView,
} from "./wgsl_storage.js";
import { rawPoolHelperName } from "./wgsl_support_helpers.js";
import {
  emitSurfaceArgument,
  emitSurfaceReadExpression,
  emitSurfaceWriteExpression,
  emitTextureArgument,
  emitTextureReadExpression,
  isTextureBindingName,
  isTextureReadCall,
  textureReadArgsForEmit,
  textureReadHelperSuffix,
  type WgslTextureSurfaceEmitContext,
} from "./wgsl_texture_surface.js";
import {
  emitExpressionAsWgslScalarText,
  emitNumberLiteral,
  emitNumberLiteralAsU32,
  emitVectorLaneSetExpression,
  isAbstractIntegerLiteral,
  isBitwiseOperator,
  isComparisonOperator,
  isIntegerNumberLiteral,
  isShiftOperator,
  isVectorArithmeticOperator,
  numberLiteralHasFloatSyntax,
  numberLiteralHasUnsignedSuffix,
  promotedCudaScalarType,
  updateDeltaForValueType,
} from "./wgsl_value_conversion.js";
import {
  createStorageView as createStorageViewImpl,
  emitPointerAliasIndex as emitPointerAliasIndexImpl,
  emitPointerIndex as emitPointerIndexImpl,
  flattenedPointerAlias as flattenedPointerAliasImpl,
  localArrayForStorageView as localArrayForStorageViewImpl,
  scalarParamStorageViewLValue as scalarParamStorageViewLValueImpl,
  scalarStorageViewLValue as scalarStorageViewLValueImpl,
  storageViewForPointerExpression as storageViewForPointerExpressionImpl,
  storageViewLValue as storageViewLValueImpl,
  vectorStorageLValue as vectorStorageLValueImpl,
  type ScalarParamStorageViewLValue,
  type ScalarStorageViewLValue,
  type VectorStorageLValue,
  type WgslStorageViewCallbacks,
} from "./wgsl_storage_views.js";
import {
  CudaLiteCompilerError,
  type CudaLiteAssignmentExpression,
  type CudaLiteCallExpression,
  type CudaLiteConditionalExpression,
  type CudaLiteDeviceGlobal,
  type CudaLiteDeviceFunction,
  type CudaLiteExpression,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type CudaLiteStatement,
  type CudaLiteVarDecl,
  type KernelIrModule,
  type SourceSpan,
} from "./types.js";

export interface KernelIrWgslOutput {
  readonly wgsl: string;
  readonly program: WgslKernelProgram;
}

export function emitKernelIrWgsl(
  ir: KernelIrModule,
  options: EmitKernelIrWgslOptions = {},
): KernelIrWgslOutput {
  const f16Mode = effectiveF16Mode(ir, options);
  const subgroupMode = effectiveSubgroupMode(ir, options);
  if (f16Mode === "native" && ir.requiredFeatures.includes("shader-f16") && !options.features?.["shader-f16"]) {
    throw featureError("missing-feature-shader-f16", "half requires WebGPU shader-f16 support");
  }
  if (subgroupMode === "native" && ir.requiredFeatures.includes("subgroups") && !options.features?.subgroups) {
    throw featureError("missing-feature-subgroups", "bg_subgroup_add requires WebGPU subgroups support");
  }

  const context = createEmitContext(ir, options);
  const textureSurface = textureSurfaceContext(context);
  const lines = emitKernelModulePrelude(context, {
    f16Mode,
    subgroupMode,
    textureSurface,
    emitExpression: (expression) => emitExpression(expression, context),
  });

  const emittedFunctions = functionsToEmit(ir);
  const functionLines = emittedFunctions.flatMap((fn) => [
    "",
    ...emitDeviceFunction(fn, context),
    ...(deviceFunctionNeedsGuardedBarrierClone(fn) ? ["", ...emitDeviceFunction(fn, context, { guardedBarrierClone: true })] : []),
  ]);
  const mainBodyLines = emitStatementSequence(ir.body, context, 1, { lowerEarlyReturnsBeforeBarriers: true });
  const pointerHelperLines = emitDevicePointerHelpers(ir, context, [...functionLines, ...mainBodyLines]);
  if (pointerHelperLines.length > 0) {
    lines.push("");
    lines.push(...pointerHelperLines);
  }
  for (const helper of context.scalarWarpReduceHelpers.values()) {
    lines.push("");
    lines.push(...emitScalarWarpReduceWorkgroupStorage(helper, context));
    lines.push("");
    lines.push(...emitScalarWarpReduceHelper(helper, context));
  }
  for (const helper of context.scalarWarpShuffleHelpers.values()) {
    lines.push("");
    lines.push(...emitScalarWarpShuffleWorkgroupStorage(helper, context));
    lines.push("");
    lines.push(...emitScalarWarpShuffleHelper(helper, context));
  }
  for (const helper of context.vectorCooperativeReduceHelpers.values()) {
    lines.push("");
    lines.push(...emitVectorCooperativeReduceWorkgroupStorage(helper, context));
  }
  lines.push(...functionLines);
  for (const helper of context.vectorCooperativeReduceHelpers.values()) {
    lines.push("");
    lines.push(...emitVectorCooperativeReduceHelper(helper, context));
  }

  lines.push(...emitKernelEntryPoint(context, mainBodyLines, emitUniformScalarRead));

  const rawWgsl = normalizeIntegerOrAssignments(lines.join("\n"));
  const wgsl = f16Mode === "f32" ? rewriteF16WgslToF32(rawWgsl) : rawWgsl;
  const bindings = f16Mode === "f32" ? rewriteF16BindingsToF32(context.bindings) : context.bindings;
  return {
    wgsl,
    program: defineWgslKernelProgram({
      name: ir.name,
      wgsl,
      bindings,
      workgroupSize: ir.workgroupSize,
    }),
  };
}

type EmitMode = "value" | "lvalue";

function normalizeIntegerOrAssignments(wgsl: string): string {
  return wgsl.replace(
    /^(\s*)([A-Za-z_][A-Za-z0-9_]*) = \(i32\(\2\) \| i32\((.+)\)\);$/gmu,
    "$1$2 = ($2 | u32($3));",
  );
}

function textureSurfaceContext(context: EmitContext): WgslTextureSurfaceEmitContext {
  return {
    requiredFeatures: context.ir.requiredFeatures,
    textureNames: textureBindings(context.ir).map((texture) => texture.name),
    surfaceNames: context.ir.params.filter(isSurfaceParam).map((surface) => surface.name),
    uniformParamsName: UNIFORM_PARAMS_NAME,
    nameFor: (name) => context.nameFor(name),
    surfaceWidthField: (name) => context.surfaceWidthField(name),
    surfaceHeightField: (name) => context.surfaceHeightField(name),
    emitExpression: (expression, mode = "value") => emitExpression(expression, context, mode),
    emitExpressionAsValueType: (expression, valueType) => emitExpressionAsValueType(expression, valueType, context),
    expressionValueType: (expression) => expressionValueTypeForEmit(expression, context),
  };
}

function cooperativeCallbacks(context: EmitContext): WgslCooperativeCallbacks {
  return {
    emitExpression: (expression) => emitExpression(expression, context),
    emitTruthinessExpression: (expression) => emitTruthinessExpression(expression, context),
    emitExpressionAsValueType: (expression, valueType) => emitExpressionAsValueType(expression, valueType, context),
    emitExpressionAsWgslScalar: (expression, wgslType) => emitExpressionAsWgslScalar(expression, wgslType, context),
    expressionValueType: (expression) => expressionValueTypeForEmit(expression, context),
  };
}

function emitStatement(
  statement: CudaLiteStatement,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  switch (statement.kind) {
    case "block": {
      const lines = [`${prefix}{`];
      lines.push(...emitStatementSequence(statement.body, context, indentLevel + 1));
      lines.push(`${prefix}}`);
      return lines;
    }
    case "var":
      if (statement.storage === "shared") return [];
      if (statement.pointer) {
        if (isLocalPointerArrayDecl(statement)) return emitLocalPointerArrayDecl(statement, context, indentLevel);
        if (context.localPointerHandleFor(statement.name)) return emitLocalPointerHandleDecl(statement, context, indentLevel);
        if (!isEmittedPointerVar(statement, context)) return [];
        return [`${prefix}var ${context.nameFor(statement.name)}: u32${statement.init ? ` = ${emitExpression(statement.init, context)}` : " = 0u"};`];
      }
      if (statement.dimensions.length > 0 || statement.matrixTile) {
        return [
          `${prefix}var ${context.nameFor(statement.name)}: ${emitLocalArrayType(statement)};`,
          ...emitLocalArrayInitializer(statement, context, indentLevel, indent, (expression) => emitExpression(expression, context)),
        ];
      }
      return [`${prefix}var ${context.nameFor(statement.name)}: ${wgslScalar(statement.valueType)}${statement.init ? ` = ${emitLocalInit(statement, context)}` : ""};`];
    case "dim3":
      return [];
    case "cooperative-group":
      return [];
    case "kernel-launch":
      return [`${prefix}// device-side launch omitted: ${statement.callee}<<<...>>>`];
    case "asm":
      return emitInlineAsmStatement(statement, context)
        .split("\n")
        .map((line) => `${prefix}${line};`);
    case "expr":
      {
        const wmma = emitWmmaStatement(statement.expression, context, indentLevel);
        if (wmma) return wmma;
      }
      {
        const cpAsync = emitCpAsyncStatement(statement.expression, context, indentLevel);
        if (cpAsync) return cpAsync;
      }
      {
        const noopComment = noopCallComment(statement.expression);
        if (noopComment) return [`${prefix}// ${noopComment}`];
      }
      {
        const fill = emitFillRegsStatement(statement.expression, context, indentLevel);
        if (fill) return fill;
      }
      if (isBarrierCall(statement.expression)) return [`${prefix}workgroupBarrier();`];
      {
        const inlined = emitInlineBarrierDeviceFunctionExprStatement(statement.expression, context, indentLevel);
        if (inlined) return inlined;
      }
      if (statement.expression.kind === "assignment") {
        const inlined = emitInlineBarrierDeviceFunctionAssignment(statement.expression, context, indentLevel);
        if (inlined) return inlined;
        return emitAssignmentStatement(statement.expression, context, indentLevel);
      }
      {
        const emitted = emitExpressionStatement(statement.expression, context);
        return emitted.length === 0 ? [] : [`${prefix}${emitted};`];
      }
    case "if": {
      const constantCondition = constantBooleanExpression(statement.condition, context);
      if (constantCondition === false) {
        return statement.alternate ? emitStatementSequence(statement.alternate, context, indentLevel) : [];
      }
      if (constantCondition === true) return emitStatementSequence(statement.consequent, context, indentLevel);
      if (!statement.alternate && statement.consequent.some((child) => statementContainsBarrierLike(child, context))) {
        const guardSource = emitTruthinessExpression(statement.condition, context);
        return emitGuardedBranch(statement.consequent, guardSource, context, indentLevel);
      }
      if (statement.alternate && (
        statement.consequent.some((child) => statementContainsBarrierLike(child, context)) ||
        statement.alternate.some((child) => statementContainsBarrierLike(child, context))
      )) {
        const guardSource = emitTruthinessExpression(statement.condition, context);
        return [
          ...emitGuardedBranch(statement.consequent, guardSource, context, indentLevel),
          ...emitGuardedBranch(statement.alternate, `!(${guardSource})`, context, indentLevel),
        ];
      }
      const subgroupAssignment = emitPredicatedSubgroupAssignment(statement, context, indentLevel);
      if (subgroupAssignment) return subgroupAssignment;
      if (!statement.alternate && statement.consequent.some(statementContainsSubgroupCall)) {
        return emitPredicatedSubgroupBranch(statement, context, indentLevel);
      }
      const lines = [`${prefix}if (${emitTruthinessExpression(statement.condition, context)}) {`];
      lines.push(...emitStatementSequence(statement.consequent, context, indentLevel + 1));
      if (statement.alternate) {
        lines.push(`${prefix}} else {`);
        lines.push(...emitStatementSequence(statement.alternate, context, indentLevel + 1));
      }
      lines.push(`${prefix}}`);
      return lines;
    }
    case "for": {
      if (statement.update?.kind === "sequence" || statement.init?.kind === "sequence") {
        return emitForLoopWithContinuing(statement, context, indentLevel);
      }
      const loopContext = scopedForLoopContext(statement, context);
      const init = statement.init?.kind === "var"
        ? emitForVar(statement.init, loopContext)
        : statement.init
          ? emitExpression(statement.init, loopContext)
          : "";
      const condition = statement.condition ? emitTruthinessExpression(statement.condition, loopContext) : "true";
      const update = statement.update ? emitExpression(statement.update, loopContext) : "";
      const breakFlag = statementHasEarlyBreakBeforeBarrier(statement, context) ? `bg_loop_active_${statement.span.start}` : undefined;
      if (breakFlag) {
        const lines = [`${prefix}var ${breakFlag}: bool = true;`, `${prefix}for (${init}; ${condition}; ${update}) {`];
        lines.push(...emitStatementSequence(statement.body, loopContext, indentLevel + 1, { activeFlag: breakFlag }));
        lines.push(`${prefix}}`);
        return lines;
      }
      if (
        statement.body.some(statementContainsBarrier) ||
        statement.body.some(statementContainsSubgroupCall) ||
        statement.body.some((child) => statementContainsBarrierLike(child, context))
      ) {
        return emitBoundedBarrierForLoop(statement, loopContext, indentLevel);
      }
      const lines = [`${prefix}for (${init}; ${condition}; ${update}) {`];
      lines.push(...emitStatementSequence(statement.body, loopContext, indentLevel + 1));
      lines.push(`${prefix}}`);
      return lines;
    }
    case "while": {
      if (statementHasEarlyBreakBeforeBarrier(statement, context)) {
        return emitWhileLoopWithEarlyBreakBeforeBarrier(statement, context, indentLevel);
      }
      const lines = [`${prefix}while (${emitTruthinessExpression(statement.condition, context)}) {`];
      lines.push(...emitStatementSequence(statement.body, context, indentLevel + 1));
      lines.push(`${prefix}}`);
      return lines;
    }
    case "do-while": {
      const lines = [`${prefix}loop {`];
      lines.push(...emitStatementSequence(statement.body, context, indentLevel + 1));
      lines.push(`${indent(indentLevel + 1)}if (!(${emitTruthinessExpression(statement.condition, context)})) { break; }`);
      lines.push(`${prefix}}`);
      return lines;
    }
    case "return":
      return [`${prefix}${statement.value ? `return ${emitReturnValue(statement.value, context)};` : "return;"}`];
    case "continue":
      return [`${prefix}continue;`];
    case "break":
      return [`${prefix}break;`];
  }
}

interface StatementSequenceOptions {
  readonly activeFlag?: string;
  readonly lowerEarlyReturnsBeforeBarriers?: boolean;
  readonly localValueScopeApplied?: boolean;
}

function emitStatementSequence(
  statements: readonly CudaLiteStatement[],
  context: EmitContext,
  indentLevel: number,
  options: StatementSequenceOptions = {},
): string[] {
  if (!options.localValueScopeApplied) {
    const scoped = contextWithImmediateLocalValueTypes(statements, context);
    if (scoped !== context) {
      return emitStatementSequence(statements, scoped, indentLevel, { ...options, localValueScopeApplied: true });
    }
  }
  const activeFlag = options.activeFlag;
  if (activeFlag) {
    return statements.flatMap((statement) => emitStatementWithActiveFlag(statement, activeFlag, context, indentLevel));
  }
  if (options.lowerEarlyReturnsBeforeBarriers) {
    const earlyReturnIndex = statements.findIndex((statement, index) =>
      (splitIfTrailingVoidReturn(statement) !== undefined || statementContainsVoidReturn(statement)) &&
      statements.slice(index + 1).some((item) => statementContainsBarrierLike(item, context) || statementContainsSubgroupCall(item))
    );
    if (earlyReturnIndex >= 0) {
      const flag = context.nameFor("bg_active_lane");
      const prefix = indent(indentLevel);
      const lines = statements.slice(0, earlyReturnIndex).flatMap((statement) => emitStatement(statement, context, indentLevel));
      lines.push(`${prefix}var ${flag}: bool = true;`);
      for (let index = earlyReturnIndex; index < statements.length; index++) {
        const statement = statements[index]!;
        const split = splitIfTrailingVoidReturn(statement);
        if (split) {
          lines.push(...emitIfTrailingReturnAsActiveFlag(split, flag, context, indentLevel));
        } else {
          lines.push(...emitStatementWithActiveFlag(statement, flag, context, indentLevel));
        }
      }
      return lines;
    }
    const earlyBreakLoopIndex = statements.findIndex((statement, index) =>
      statement.kind === "for" &&
      statement.body.some((child) => splitIfTrailingBreak(child) !== undefined) &&
      statements.slice(index + 1).some((item) => statementContainsBarrierLike(item, context) || statementContainsSubgroupCall(item))
    );
    if (earlyBreakLoopIndex >= 0) {
      const flag = context.nameFor("bg_active_lane");
      const prefix = indent(indentLevel);
      const lines = statements.slice(0, earlyBreakLoopIndex).flatMap((statement) => emitStatement(statement, context, indentLevel));
      lines.push(`${prefix}var ${flag}: bool = true;`);
      for (let index = earlyBreakLoopIndex; index < statements.length; index++) {
        lines.push(...emitStatementWithActiveFlag(statements[index]!, flag, context, indentLevel));
      }
      return lines;
    }
  }
  const sharedAtomicBreakIndex = statements.findIndex((statement, index) => {
    const split = splitIfTrailingBreak(statement);
    const next = statements[index + 1];
    return split !== undefined &&
      split.beforeBreak.length === 0 &&
      isSharedAtomicUniformCondition(split.condition, context) &&
      next !== undefined &&
      isBarrierStatement(next);
  });
  if (sharedAtomicBreakIndex >= 0) {
    const split = splitIfTrailingBreak(statements[sharedAtomicBreakIndex]!);
    const barrier = statements[sharedAtomicBreakIndex + 1]!;
    if (split !== undefined) {
      return [
        ...statements.slice(0, sharedAtomicBreakIndex).flatMap((statement) => emitStatement(statement, context, indentLevel)),
        ...emitStatement(barrier, context, indentLevel),
        ...emitStatement(statements[sharedAtomicBreakIndex]!, context, indentLevel),
        ...emitStatementSequence(statements.slice(sharedAtomicBreakIndex + 2), context, indentLevel),
      ];
    }
  }
  return statements.flatMap((statement) => emitStatement(statement, context, indentLevel));
}

function contextWithImmediateLocalValueTypes(
  statements: readonly CudaLiteStatement[],
  context: EmitContext,
): EmitContext {
  const scopedTypes = new Map<string, CudaLiteScalarType>();
  for (const statement of statements) {
    if (statement.kind === "var" && statement.storage === "local" && !statement.pointer && statement.dimensions.length === 0) {
      scopedTypes.set(statement.name, statement.valueType);
    }
  }
  if (scopedTypes.size === 0) return context;
  return {
    ...context,
    localValueTypeFor(name) {
      return scopedTypes.get(name) ?? context.localValueTypeFor(name);
    },
  };
}

function emitStatementWithActiveFlag(
  statement: CudaLiteStatement,
  activeFlag: string,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  if (statement.kind === "if") {
    const constantCondition = constantBooleanExpression(statement.condition, context);
    if (constantCondition === false) {
      return statement.alternate
        ? emitStatementSequence(statement.alternate, context, indentLevel, { activeFlag })
        : [];
    }
    if (constantCondition === true) return emitStatementSequence(statement.consequent, context, indentLevel, { activeFlag });
  }
  if (isBarrierStatement(statement)) return [`${prefix}workgroupBarrier();`];
  const split = splitIfTrailingVoidReturn(statement);
  if (split) return emitIfTrailingReturnAsActiveFlag(split, activeFlag, context, indentLevel);
  if (statement.kind === "if" && statementContainsVoidReturn(statement)) {
    return emitIfWithNestedReturnAsActiveFlag(statement, activeFlag, context, indentLevel);
  }
  const breakSplit = splitIfTrailingBreak(statement);
  if (breakSplit) return emitIfTrailingBreakAsActiveFlag(breakSplit, activeFlag, context, indentLevel);
  if (statement.kind === "expr" && statement.expression.kind === "assignment") {
    const hoisted = emitAssignmentWithHoistedBarrierDeviceFunction(statement.expression, context, indentLevel, activeFlag);
    if (hoisted) return hoisted;
    const inlined = emitInlineBarrierDeviceFunctionAssignment(statement.expression, context, indentLevel, activeFlag);
    if (inlined) return inlined;
  }
  if (statement.kind === "var") return emitVarStatementWithActiveFlag(statement, activeFlag, context, indentLevel);
  const localConstantAssignment = emitLocalConstantAssignmentStatement(statement, context, indentLevel);
  if (localConstantAssignment) return localConstantAssignment;
  const predicatedAssignment = emitPredicatedScalarAssignmentStatement(statement, activeFlag, context, indentLevel);
  if (predicatedAssignment) return predicatedAssignment;
  const guardedAssignment = emitGuardedAssignmentWithHoistedUniformRhs(statement, activeFlag, context, indentLevel);
  if (guardedAssignment) return guardedAssignment;
  if (statement.kind === "for" && (
    statement.body.some(statementContainsSubgroupCall) ||
    statement.body.some((child) => statementContainsBarrierLike(child, context))
  )) {
    return emitBoundedBarrierForLoop(statement, scopedForLoopContext(statement, context), indentLevel, activeFlag);
  }
  if (statement.kind === "for" && statement.body.some((child) => splitIfTrailingBreak(child) !== undefined)) {
    return emitForLoopWithActiveBreak(statement, scopedForLoopContext(statement, context), indentLevel, activeFlag);
  }
  if (statementContainsBarrierLike(statement, context) || statementContainsSubgroupCall(statement)) {
    return emitStatementWithGuard(statement, activeFlag, context, indentLevel);
  }
  const lines = [`${prefix}if (${activeFlag}) {`];
  lines.push(...emitStatement(statement, context, indentLevel + 1));
  lines.push(`${prefix}}`);
  return lines;
}

function emitStatementWithGuard(
  statement: CudaLiteStatement,
  guardSource: string,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  if (statement.kind === "if") {
    const constantCondition = constantBooleanExpression(statement.condition, context);
    if (constantCondition === false) {
      return statement.alternate
        ? emitStatementSequence(statement.alternate, context, indentLevel, { activeFlag: guardSource })
        : [];
    }
    if (constantCondition === true) return emitStatementSequence(statement.consequent, context, indentLevel, { activeFlag: guardSource });
  }
  if (isBarrierStatement(statement)) return [`${prefix}workgroupBarrier();`];
  const guardedDeviceFunctionCall = emitInlineGuardedBarrierDeviceFunctionStatement(statement, guardSource, context, indentLevel);
  if (guardedDeviceFunctionCall) return guardedDeviceFunctionCall;
  const guardedCallWithHoistedArgs = emitGuardedCallWithHoistedSubgroupArgs(statement, guardSource, context, indentLevel);
  if (guardedCallWithHoistedArgs) return guardedCallWithHoistedArgs;
  if (statement.kind === "expr" && statement.expression.kind === "assignment") {
    const hoisted = emitAssignmentWithHoistedBarrierDeviceFunction(statement.expression, context, indentLevel, guardSource);
    if (hoisted) return hoisted;
    const inlined = emitInlineBarrierDeviceFunctionAssignment(statement.expression, context, indentLevel, guardSource);
    if (inlined) return inlined;
  }
  if (statement.kind === "var") return emitVarStatementWithActiveFlag(statement, guardSource, context, indentLevel);
  const localConstantAssignment = emitLocalConstantAssignmentStatement(statement, context, indentLevel);
  if (localConstantAssignment) return localConstantAssignment;
  const predicatedAssignment = emitPredicatedScalarAssignmentStatement(statement, guardSource, context, indentLevel);
  if (predicatedAssignment) return predicatedAssignment;
  const guardedAssignment = emitGuardedAssignmentWithHoistedUniformRhs(statement, guardSource, context, indentLevel);
  if (guardedAssignment) return guardedAssignment;
  if (statement.kind === "for" && (
    statement.body.some(statementContainsSubgroupCall) ||
    statement.body.some((child) => statementContainsBarrierLike(child, context))
  )) {
    return emitBoundedBarrierForLoop(statement, scopedForLoopContext(statement, context), indentLevel, guardSource);
  }
  if (statementContainsBarrierLike(statement, context) || statementContainsSubgroupCall(statement)) {
    switch (statement.kind) {
      case "block": {
        const lines = [`${prefix}{`];
        lines.push(...statement.body.flatMap((child) => emitStatementWithGuard(child, guardSource, context, indentLevel + 1)));
        lines.push(`${prefix}}`);
        return lines;
      }
      case "if": {
        if (expressionContainsSubgroupCall(statement.condition)) {
          const conditionName = context.nameFor(`bg_guard_cond_${statement.span.start}`);
          const condition = emitTruthinessExpression(statement.condition, context);
          return [
            `${prefix}let ${conditionName}: bool = ${condition};`,
            ...emitGuardedBranch(statement.consequent, `${guardSource} && ${conditionName}`, context, indentLevel),
            ...(statement.alternate
              ? emitGuardedBranch(statement.alternate, `${guardSource} && !(${conditionName})`, context, indentLevel)
              : []),
          ];
        }
        if (!statement.alternate) {
          const nestedGuard = `${guardSource} && (${emitTruthinessExpression(statement.condition, context)})`;
          return emitGuardedBranch(statement.consequent, nestedGuard, context, indentLevel);
        }
        const condition = emitTruthinessExpression(statement.condition, context);
        return [
          ...emitGuardedBranch(statement.consequent, `${guardSource} && (${condition})`, context, indentLevel),
          ...emitGuardedBranch(statement.alternate, `${guardSource} && !(${condition})`, context, indentLevel),
        ];
      }
      case "for": {
        if (statement.update?.kind === "sequence" || statement.init?.kind === "sequence") {
          return emitForLoopWithContinuing(statement, context, indentLevel, guardSource);
        }
        const loopContext = scopedForLoopContext(statement, context);
        const init = statement.init?.kind === "var"
          ? emitForVar(statement.init, loopContext)
          : statement.init
            ? emitExpression(statement.init, loopContext)
            : "";
        const condition = statement.condition ? emitTruthinessExpression(statement.condition, loopContext) : "true";
        const update = statement.update ? emitExpression(statement.update, loopContext) : "";
        if (
          statement.body.some(statementContainsBarrier) ||
          statement.body.some(statementContainsSubgroupCall) ||
          statement.body.some((child) => statementContainsBarrierLike(child, context))
        ) {
          return emitBoundedBarrierForLoop(statement, loopContext, indentLevel, guardSource);
        }
        const lines = [`${prefix}for (${init}; ${condition}; ${update}) {`];
        lines.push(...statement.body.flatMap((child) => emitStatementWithGuard(child, guardSource, loopContext, indentLevel + 1)));
        lines.push(`${prefix}}`);
        return lines;
      }
      case "while": {
        const lines = [`${prefix}while (${emitTruthinessExpression(statement.condition, context)}) {`];
        lines.push(...statement.body.flatMap((child) => emitStatementWithGuard(child, guardSource, context, indentLevel + 1)));
        lines.push(`${prefix}}`);
        return lines;
      }
      case "do-while": {
        const lines = [`${prefix}loop {`];
        lines.push(...statement.body.flatMap((child) => emitStatementWithGuard(child, guardSource, context, indentLevel + 1)));
        lines.push(`${indent(indentLevel + 1)}if (!(${emitTruthinessExpression(statement.condition, context)})) { break; }`);
        lines.push(`${prefix}}`);
        return lines;
      }
    }
  }
  const lines = [`${prefix}if (${guardSource}) {`];
  lines.push(...emitStatement(statement, context, indentLevel + 1));
  lines.push(`${prefix}}`);
  return lines;
}

function emitInlineGuardedBarrierDeviceFunctionStatement(
  statement: CudaLiteStatement,
  guardSource: string,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (statement.kind !== "expr" || statement.expression.kind !== "call") return undefined;
  const name = expressionName(statement.expression.callee);
  const deviceFunction = name ? context.deviceFunctionFor(name, statement.expression.args.length) : undefined;
  if (!deviceFunction || !deviceFunctionNeedsGuardedBarrierClone(deviceFunction)) return undefined;
  const args = emitDeviceFunctionCallArgs(statement.expression, deviceFunction, context);
  const prefix = indent(indentLevel);
  return [`${prefix}${context.nameFor(guardedBarrierDeviceFunctionLinkName(deviceFunction, context.ir))}(${[...args, guardSource, "local_id", "workgroup_id", "num_workgroups"].join(", ")});`];
}

function emitGuardedCallWithHoistedSubgroupArgs(
  statement: CudaLiteStatement,
  guardSource: string,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (statement.kind !== "expr" || statement.expression.kind !== "call") return undefined;
  const call = statement.expression;
  if (!expressionContainsSubgroupCall(call) || !call.args.some(expressionContainsSubgroupCall)) return undefined;
  const name = expressionName(call.callee);
  if (!name?.startsWith("atomic")) return undefined;
  const prefix = indent(indentLevel);
  const replacements = new Map<CudaLiteExpression, CudaLiteExpression>();
  const lines: string[] = [];
  for (const [index, arg] of call.args.entries()) {
    if (!expressionContainsSubgroupCall(arg)) continue;
    const valueType = expressionValueTypeForEmit(arg, context) ?? "int";
    if (valueType === "void" || valueType === "complex64" || isCudaVectorType(valueType)) return undefined;
    const tempName = context.nameFor(`bg_subgroup_arg_${call.span.start}_${index}`);
    lines.push(`${prefix}let ${tempName}: ${wgslScalar(valueType)} = ${emitExpressionAsValueType(arg, valueType, context)};`);
    replacements.set(arg, { kind: "identifier", name: tempName, span: arg.span });
  }
  if (lines.length === 0) return undefined;
  const hoistedCall: CudaLiteCallExpression = {
    ...call,
    args: call.args.map((arg) => replacements.get(arg) ?? arg),
  };
  lines.push(`${prefix}if (${guardSource}) {`);
  lines.push(`${indent(indentLevel + 1)}${emitExpressionStatement(hoistedCall, context)};`);
  lines.push(`${prefix}}`);
  return lines;
}

function emitGuardedBranch(
  body: readonly CudaLiteStatement[],
  guardSource: string,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  const lines = [`${prefix}{`];
  lines.push(...body.flatMap((child) => emitStatementWithGuard(child, guardSource, context, indentLevel + 1)));
  lines.push(`${prefix}}`);
  return lines;
}

function emitInlineBarrierDeviceFunctionExprStatement(
  expression: CudaLiteExpression,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (expression.kind !== "call") return undefined;
  const name = expressionName(expression.callee);
  const fn = name ? context.deviceFunctionFor(name, expression.args.length) : undefined;
  if (!fn || !deviceFunctionNeedsInlineBarrierLowering(fn) || fn.returnType !== "void") return undefined;
  if (deviceFunctionHasSharedDeclarations(fn)) return undefined;
  return emitInlineBarrierDeviceFunctionCall(expression, fn, context, indentLevel);
}

function emitInlineBarrierDeviceFunctionAssignment(
  expression: CudaLiteAssignmentExpression,
  context: EmitContext,
  indentLevel: number,
  activeFlag?: string,
): string[] | undefined {
  if (expression.operator !== "=" || expression.right.kind !== "call") return undefined;
  const name = expressionName(expression.right.callee);
  const fn = name ? context.deviceFunctionFor(name, expression.right.args.length) : undefined;
  if (!fn || !deviceFunctionNeedsInlineBarrierLowering(fn) || fn.returnType === "void") return undefined;
  if (!activeFlag && deviceFunctionHasSharedDeclarations(fn)) return undefined;
  return emitInlineBarrierDeviceFunctionCall(
    expression.right,
    fn,
    context,
    indentLevel,
    activeFlag ? { activeFlag, assignTarget: expression.left } : { assignTarget: expression.left },
  );
}

function emitAssignmentWithHoistedBarrierDeviceFunction(
  expression: CudaLiteAssignmentExpression,
  context: EmitContext,
  indentLevel: number,
  activeFlag: string,
): string[] | undefined {
  const call = findBarrierDeviceFunctionCall(expression, context);
  if (!call || call.fn.returnType === "void") return undefined;
  const tempName = `bg_device_fn_value_${call.expression.span.start}`;
  const scopedContext = contextWithHoistedLocalValue(context, tempName, call.fn.returnType);
  const replacement: CudaLiteExpression = { kind: "identifier", name: tempName, span: call.expression.span };
  const replaced = replaceExpressionNode(expression, call.expression, replacement);
  if (replaced.kind !== "assignment") return undefined;
  const prefix = indent(indentLevel);
  return [
    `${prefix}var ${scopedContext.nameFor(tempName)}: ${wgslScalar(call.fn.returnType)} = ${zeroValue(call.fn.returnType)};`,
    ...emitInlineBarrierDeviceFunctionCall(call.expression, call.fn, scopedContext, indentLevel, { activeFlag, assignTarget: replacement }) ?? [],
    `${prefix}if (${activeFlag}) {`,
    `${indent(indentLevel + 1)}${emitAssignment(replaced, scopedContext)};`,
    `${prefix}}`,
  ];
}

function contextWithHoistedLocalValue(
  context: EmitContext,
  name: string,
  valueType: CudaLiteScalarType,
): EmitContext {
  return {
    ...context,
    isLocalName(candidate) {
      return candidate === name || context.isLocalName(candidate);
    },
    localValueTypeFor(candidate) {
      return candidate === name ? valueType : context.localValueTypeFor(candidate);
    },
  };
}

function findBarrierDeviceFunctionCall(
  expression: CudaLiteExpression,
  context: EmitContext,
): { readonly expression: CudaLiteCallExpression; readonly fn: CudaLiteDeviceFunction } | undefined {
  if (expression.kind === "call") {
    const name = expressionName(expression.callee);
    const fn = name ? context.deviceFunctionFor(name, expression.args.length) : undefined;
    if (fn && deviceFunctionCanInlineForBarrierLowering(fn, fn.returnType !== "void")) {
      return { expression, fn };
    }
  }
  for (const child of expressionChildren(expression)) {
    const found = findBarrierDeviceFunctionCall(child, context);
    if (found) return found;
  }
  return undefined;
}

interface InlineBarrierDeviceFunctionOptions {
  readonly activeFlag?: string;
  readonly assignTarget?: CudaLiteExpression;
}

function emitInlineBarrierDeviceFunctionCall(
  expression: CudaLiteCallExpression,
  fn: CudaLiteDeviceFunction,
  context: EmitContext,
  indentLevel: number,
  options: InlineBarrierDeviceFunctionOptions = {},
): string[] | undefined {
  if (!deviceFunctionCanInlineForBarrierLowering(fn, options.assignTarget !== undefined)) return undefined;
  const prefix = indent(indentLevel);
  const uniquePrefix = `bg_inline_${safeWgslIdentifier(fn.name)}_${expression.span.start}`;
  const remapped = new Map<string, string>();
  const remap = (name: string): string => {
    let mapped = remapped.get(name);
    if (!mapped) {
      mapped = context.nameFor(`${uniquePrefix}_${name}`);
      remapped.set(name, mapped);
    }
    return mapped;
  };
  for (const param of fn.params) {
    remap(param.name);
    if (param.pointer) {
      remap(`${param.name}_buffer`);
      remap(`${param.name}_base`);
    }
    if (param.cooperativeGroupKind === "thread") remap(`${param.name}_tile_size`);
  }
  const functionSharedNames = collectFunctionSharedDeclarationNames(fn.body);
  const remappableLocalNames = new Set([...collectLocalNames(fn.body)].filter((name) => !functionSharedNames.has(name)));
  for (const name of remappableLocalNames) remap(name);

  const cooperativeParams = new Map(fn.params
    .filter((param) => param.cooperativeGroupKind !== undefined)
    .map((param) => {
      const group = cooperativeGroupForParam(param, context);
      return [
        param.name,
        {
          ...group,
          ...(param.cooperativeGroupKind === "thread" ? { dynamicTileSizeName: remap(`${param.name}_tile_size`) } : {}),
        },
      ] as const;
    }));
  const functionCooperativeGroups = collectCooperativeGroups(fn.body);
  const functionLocalPointerHandles = collectLocalPointerHandles(fn.body, undefined, structuredPointerHandleRoots(context.ir));
  const functionPointerAliases = collectPointerAliases(fn.body, new Set(functionLocalPointerHandles.keys()));
  const functionLocalValueTypes = new Map(collectLocalValueTypes(fn.body));
  const functionParamNames = new Set(fn.params.map((param) => param.name));
  const functionLocalNames = new Set([...fn.params.map((param) => param.name), ...remappableLocalNames]);
  const assignedNames = collectAssignedNames(fn.body);
  const functionExpressionValueTypes = new WeakMap<CudaLiteExpression, CudaLiteScalarType | undefined>();
  const functionPointerParams = new Set(fn.params
    .filter((param) => param.pointer && usesFunctionLocalPointerParam(fn, param, context.ir))
    .map((param) => param.name));
  if (functionPointerParams.size > 0) return undefined;

  const inlineBaseContext: EmitContext = {
    ...context,
    currentReturnType: fn.returnType,
    expressionValueTypes: functionExpressionValueTypes,
    nameFor(name) {
      return remapped.get(name) ?? context.nameFor(name);
    },
    isLocalName(name) {
      return remapped.has(name) || functionLocalNames.has(name) || context.isLocalName(name);
    },
    localValueTypeFor(name) {
      const param = fn.params.find((item) => item.name === name && !item.pointer);
      return functionLocalValueTypes.get(name) ?? param?.valueType ?? context.localValueTypeFor(name);
    },
    localPointerHandleFor(name) {
      return functionLocalPointerHandles.get(name) ?? context.localPointerHandleFor(name);
    },
    pointerAliasFor(name, span) {
      return pointerAliasDeclarationFor(functionPointerAliases, name, span) ?? context.pointerAliasFor(name, span);
    },
    paramFor(name) {
      return functionParamNames.has(name) ? undefined : context.paramFor(name);
    },
    cooperativeGroupFor(name) {
      return functionCooperativeGroups.get(name) ?? cooperativeParams.get(name) ?? context.cooperativeGroupFor(name);
    },
    isAtomicShared(name) {
      return functionLocalNames.has(name) ? false : context.isAtomicShared(name);
    },
  };
  const functionContext = withDevicePointerParams(
    inlineBaseContext,
    fn.params.filter((param) => param.pointer),
    functionLocalNames,
  );
  const lines = [`${prefix}{`];
  for (const [index, param] of fn.params.entries()) {
    const arg = expression.args[index];
    if (param.pointer) {
      const [buffer, base] = arg ? emitDevicePointerArgument(arg, context) : ["0u", "0u"];
      lines.push(`${indent(indentLevel + 1)}let ${functionContext.nameFor(`${param.name}_buffer`)}: u32 = ${buffer};`);
      lines.push(`${indent(indentLevel + 1)}let ${functionContext.nameFor(`${param.name}_base`)}: u32 = ${base};`);
      continue;
    }
    if (param.cooperativeGroupKind !== undefined) {
      if (param.cooperativeGroupKind === "thread") {
        lines.push(`${indent(indentLevel + 1)}let ${functionContext.nameFor(`${param.name}_tile_size`)}: u32 = ${emitCooperativeGroupTileSizeArgument(arg, context)};`);
      }
      continue;
    }
    if (param.valueType === "texture2d") continue;
    if (param.valueType === "surface2d") {
      const value = emitSurfaceArgument(arg, textureSurfaceContext(context));
      const binding = deviceFunctionParamNeedsMutableBinding(fn, param.name, assignedNames) ? "var" : "let";
      lines.push(`${indent(indentLevel + 1)}${binding} ${functionContext.nameFor(param.name)}: u32 = ${value};`);
      continue;
    }
    const value = arg
      ? emitInlineBarrierScalarArgument(arg, param.valueType, context)
      : zeroValue(param.valueType);
    const binding = deviceFunctionParamNeedsMutableBinding(fn, param.name, assignedNames) ? "var" : "let";
    lines.push(`${indent(indentLevel + 1)}${binding} ${functionContext.nameFor(param.name)}: ${wgslScalar(param.valueType)} = ${value};`);
  }

  const final = fn.body[fn.body.length - 1];
  const body = options.assignTarget && final?.kind === "return" ? fn.body.slice(0, -1) : fn.body;
  lines.push(...emitStatementSequence(
    body,
    functionContext,
    indentLevel + 1,
    options.activeFlag ? { activeFlag: options.activeFlag } : { lowerEarlyReturnsBeforeBarriers: true },
  ));
  if (options.assignTarget && final?.kind === "return" && final.value) {
    const target = emitExpression(options.assignTarget, context, "lvalue");
    lines.push(`${indent(indentLevel + 1)}${target} = ${emitReturnValue(final.value, functionContext)};`);
  }
  lines.push(`${prefix}}`);
  return lines;
}

function collectFunctionSharedDeclarationNames(statements: readonly CudaLiteStatement[]): ReadonlySet<string> {
  const names = new Set<string>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.storage === "shared") names.add(item.name);
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.storage === "shared") names.add(item.init.name);
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return names;
}

function emitInlineBarrierScalarArgument(
  expression: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: EmitContext,
): string {
  if (expression.kind === "identifier" && sharedDeclarationFor(expression.name, context) && !context.isAtomicShared(expression.name)) {
    return emitExpressionAsValueType(expression, valueType, context);
  }
  return emitExpressionAsValueType(expression, valueType, context);
}

function deviceFunctionNeedsInlineBarrierLowering(fn: CudaLiteDeviceFunction): boolean {
  return fn.body.some(statementContainsBarrier) || fn.body.some(statementContainsSubgroupCall);
}

function deviceFunctionHasSharedDeclarations(fn: CudaLiteDeviceFunction): boolean {
  return fn.body.some(statementContainsSharedDeclaration);
}

function statementContainsSharedDeclaration(statement: CudaLiteStatement): boolean {
  if (statement.kind === "var") return statement.storage === "shared";
  if (statement.kind === "block") return statement.body.some(statementContainsSharedDeclaration);
  if (statement.kind === "if") return statement.consequent.some(statementContainsSharedDeclaration) ||
    (statement.alternate?.some(statementContainsSharedDeclaration) ?? false);
  if (statement.kind === "for") return (statement.init?.kind === "var" && statement.init.storage === "shared") ||
    statement.body.some(statementContainsSharedDeclaration);
  if (statement.kind === "while" || statement.kind === "do-while") return statement.body.some(statementContainsSharedDeclaration);
  return false;
}

function deviceFunctionCanInlineForBarrierLowering(fn: CudaLiteDeviceFunction, needsReturnValue: boolean): boolean {
  if (!deviceFunctionNeedsInlineBarrierLowering(fn)) return false;
  if (!needsReturnValue) return fn.returnType === "void";
  const final = fn.body[fn.body.length - 1];
  return fn.returnType !== "void" && final?.kind === "return" && countReturns(fn.body) === 1;
}

function emitVarStatementWithActiveFlag(
  statement: CudaLiteVarDecl,
  activeFlag: string,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  if (statement.storage === "shared") return [];
  if (statement.pointer || statement.dimensions.length > 0 || statement.matrixTile || !statement.init) {
    return emitStatement(statement, context, indentLevel);
  }
  if (statement.init.kind === "call") {
    const name = expressionName(statement.init.callee);
    const fn = name ? context.deviceFunctionFor(name, statement.init.args.length) : undefined;
    if (fn && deviceFunctionNeedsInlineBarrierLowering(fn) && fn.returnType !== "void") {
      const target: CudaLiteExpression = { kind: "identifier", name: statement.name, span: statement.span };
      return [
        `${prefix}var ${context.nameFor(statement.name)}: ${wgslScalar(statement.valueType)} = ${zeroValue(statement.valueType)};`,
        ...emitInlineBarrierDeviceFunctionCall(statement.init, fn, context, indentLevel, { activeFlag, assignTarget: target }) ?? [],
      ];
    }
  }
  if (statementContainsSubgroupCall(statement)) {
    const name = context.nameFor(statement.name);
    const init = emitActiveFlagLocalInit(statement, context);
    return [
      `${prefix}var ${name}: ${wgslScalar(statement.valueType)} = ${zeroValue(statement.valueType)};`,
      `${prefix}${name} = select(${name}, ${init}, ${activeFlag});`,
    ];
  }
  if (expressionIsLocalConstantSafe(statement.init, context)) {
    return [
      `${prefix}var ${context.nameFor(statement.name)}: ${wgslScalar(statement.valueType)} = ${zeroValue(statement.valueType)};`,
      `${prefix}${context.nameFor(statement.name)} = ${emitActiveFlagLocalInit(statement, context)};`,
    ];
  }
  return [
    `${prefix}var ${context.nameFor(statement.name)}: ${wgslScalar(statement.valueType)} = ${zeroValue(statement.valueType)};`,
    `${prefix}if (${activeFlag}) {`,
    `${indent(indentLevel + 1)}${context.nameFor(statement.name)} = ${emitActiveFlagLocalInit(statement, context)};`,
    `${prefix}}`,
  ];
}

function emitActiveFlagLocalInit(statement: CudaLiteVarDecl, context: EmitContext): string {
  if ((isCudaVectorType(statement.valueType) || statement.valueType === "complex64") && statement.init?.kind === "identifier") {
    return emitExpression(statement.init, context);
  }
  return emitLocalInit(statement, context);
}

function emitIfTrailingReturnAsActiveFlag(
  split: IfTrailingReturn,
  activeFlag: string,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  const lines = [`${prefix}if (${activeFlag} && (${emitTruthinessExpression(split.condition, context)})) {`];
  lines.push(...split.beforeReturn.flatMap((child) => emitStatement(child, context, indentLevel + 1)));
  lines.push(`${indent(indentLevel + 1)}${activeFlag} = false;`);
  lines.push(`${prefix}}`);
  if (split.activeBranch && split.activeBranch.length > 0) {
    lines.push(...emitStatementSequence(split.activeBranch, context, indentLevel, { activeFlag }));
  }
  return lines;
}

function emitIfWithNestedReturnAsActiveFlag(
  statement: Extract<CudaLiteStatement, { kind: "if" }>,
  activeFlag: string,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  const condition = emitTruthinessExpression(statement.condition, context);
  if (!statement.alternate) {
    const lines = [`${prefix}if (${activeFlag} && (${condition})) {`];
    lines.push(...emitStatementSequence(statement.consequent, context, indentLevel + 1, { activeFlag }));
    lines.push(`${prefix}}`);
    return lines;
  }
  const lines = [`${prefix}if (${activeFlag}) {`, `${indent(indentLevel + 1)}if (${condition}) {`];
  lines.push(...emitStatementSequence(statement.consequent, context, indentLevel + 2, { activeFlag }));
  lines.push(`${indent(indentLevel + 1)}} else {`);
  lines.push(...emitStatementSequence(statement.alternate, context, indentLevel + 2, { activeFlag }));
  lines.push(`${indent(indentLevel + 1)}}`);
  lines.push(`${prefix}}`);
  return lines;
}

function emitIfTrailingBreakAsActiveFlag(
  split: IfTrailingBreak,
  activeFlag: string,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  if (split.beforeBreak.length === 0) {
    return [`${prefix}${activeFlag} = (${activeFlag} && !(${emitTruthinessExpression(split.condition, context)}));`];
  }
  const lines = [`${prefix}if (${activeFlag} && (${emitTruthinessExpression(split.condition, context)})) {`];
  lines.push(...split.beforeBreak.flatMap((child) => emitStatement(child, context, indentLevel + 1)));
  lines.push(`${indent(indentLevel + 1)}${activeFlag} = false;`);
  lines.push(`${prefix}}`);
  return lines;
}

function statementHasEarlyBreakBeforeBarrier(statement: CudaLiteStatement, context: EmitContext): boolean {
  if (statement.kind !== "for" && statement.kind !== "while" && statement.kind !== "do-while") return false;
  return statement.body.some((child, index) =>
    splitIfTrailingBreak(child) !== undefined && statement.body.slice(index + 1).some((item) => statementContainsBarrierLike(item, context))
  );
}

function statementContainsBarrierLike(statement: CudaLiteStatement, context: EmitContext): boolean {
  if (statementContainsBarrier(statement)) return true;
  switch (statement.kind) {
    case "block":
      return statement.body.some((child) => statementContainsBarrierLike(child, context));
    case "var":
      if (!statement.init) return false;
      if (expressionContainsBarrierDeviceFunctionCall(statement.init, context)) return true;
      if (statement.init.kind !== "call") return false;
      {
        const name = expressionName(statement.init.callee);
        const fn = name ? context.deviceFunctionFor(name, statement.init.args.length) : undefined;
        return fn !== undefined && deviceFunctionNeedsInlineBarrierLowering(fn);
      }
    case "expr":
      if (expressionContainsBarrierDeviceFunctionCall(statement.expression, context)) return true;
      if (statement.expression.kind === "assignment" && statement.expression.right.kind === "call") {
        const name = expressionName(statement.expression.right.callee);
        const fn = name ? context.deviceFunctionFor(name, statement.expression.right.args.length) : undefined;
        return fn !== undefined && deviceFunctionNeedsInlineBarrierLowering(fn);
      }
      if (statement.expression.kind !== "call") return false;
      {
        const name = expressionName(statement.expression.callee);
        const fn = name ? context.deviceFunctionFor(name, statement.expression.args.length) : undefined;
        return fn !== undefined && deviceFunctionNeedsGuardedBarrierClone(fn);
      }
    case "if":
      return statement.consequent.some((child) => statementContainsBarrierLike(child, context)) ||
        (statement.alternate?.some((child) => statementContainsBarrierLike(child, context)) ?? false);
    case "for":
    case "while":
    case "do-while":
      return statement.body.some((child) => statementContainsBarrierLike(child, context));
    default:
      return false;
  }
}

function expressionContainsBarrierDeviceFunctionCall(expression: CudaLiteExpression, context: EmitContext): boolean {
  return findBarrierDeviceFunctionCall(expression, context) !== undefined;
}

function isSharedAtomicUniformCondition(expression: CudaLiteExpression, context: EmitContext): boolean {
  const info = sharedAtomicUniformExpressionInfo(expression, context);
  return info.uniform && info.hasSharedAtomicLoad;
}

function sharedAtomicUniformExpressionInfo(
  expression: CudaLiteExpression,
  context: EmitContext,
): { readonly uniform: boolean; readonly hasSharedAtomicLoad: boolean } {
  switch (expression.kind) {
    case "number":
      return { uniform: true, hasSharedAtomicLoad: false };
    case "identifier":
      return {
        uniform: context.isUniformScalar(expression.name) || context.isAtomicShared(expression.name),
        hasSharedAtomicLoad: context.isAtomicShared(expression.name),
      };
    case "cast":
      return sharedAtomicUniformExpressionInfo(expression.expression, context);
    case "unary":
      if (expression.operator !== "!" && expression.operator !== "+" && expression.operator !== "-" && expression.operator !== "~") {
        return { uniform: false, hasSharedAtomicLoad: false };
      }
      return sharedAtomicUniformExpressionInfo(expression.argument, context);
    case "binary": {
      const left = sharedAtomicUniformExpressionInfo(expression.left, context);
      const right = sharedAtomicUniformExpressionInfo(expression.right, context);
      return {
        uniform: left.uniform && right.uniform,
        hasSharedAtomicLoad: left.hasSharedAtomicLoad || right.hasSharedAtomicLoad,
      };
    }
    case "conditional": {
      const condition = sharedAtomicUniformExpressionInfo(expression.condition, context);
      const consequent = sharedAtomicUniformExpressionInfo(expression.consequent, context);
      const alternate = sharedAtomicUniformExpressionInfo(expression.alternate, context);
      return {
        uniform: condition.uniform && consequent.uniform && alternate.uniform,
        hasSharedAtomicLoad: condition.hasSharedAtomicLoad || consequent.hasSharedAtomicLoad || alternate.hasSharedAtomicLoad,
      };
    }
    case "call":
      if (expressionName(expression.callee) === "atomicLoad" && expression.args[0] && isSharedAtomicAddress(expression.args[0], context)) {
        return { uniform: true, hasSharedAtomicLoad: true };
      }
      return { uniform: false, hasSharedAtomicLoad: false };
    default:
      return { uniform: false, hasSharedAtomicLoad: false };
  }
}

function isSharedAtomicAddress(expression: CudaLiteExpression, context: EmitContext): boolean {
  if (expression.kind !== "unary" || expression.operator !== "&") return false;
  const target = expression.argument;
  if (target.kind === "identifier") return context.isAtomicShared(target.name);
  if (target.kind === "index" && target.target.kind === "identifier") return context.isAtomicShared(target.target.name);
  return false;
}

function scopedForLoopContext(
  statement: Extract<CudaLiteStatement, { kind: "for" }>,
  context: EmitContext,
): EmitContext {
  const init = statement.init;
  if (init?.kind !== "var" || init.storage !== "local" || init.pointer || init.dimensions.length > 0) return context;
  return {
    ...context,
    localValueTypeFor(name) {
      return name === init.name ? init.valueType : context.localValueTypeFor(name);
    },
  };
}

function emitPredicatedSubgroupAssignment(
  statement: Extract<CudaLiteStatement, { kind: "if" }>,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (statement.alternate || statement.consequent.length !== 1) return undefined;
  const only = statement.consequent[0];
  if (only?.kind !== "expr" || only.expression.kind !== "assignment" || only.expression.operator !== "=") return undefined;
  if (expressionContainsAssignment(only.expression.right)) return undefined;
  if (!expressionContainsSubgroupCall(only.expression.right)) return undefined;
  const left = emitExpression(only.expression.left, context);
  const right = emitExpressionAsValueType(
    only.expression.right,
    expressionValueTypeForEmit(only.expression.left, context) ?? expressionValueTypeForEmit(only.expression.right, context) ?? "float",
    context,
  );
  return [`${indent(indentLevel)}${left} = select(${left}, ${right}, ${emitTruthinessExpression(statement.condition, context)});`];
}

function emitPredicatedSubgroupBranch(
  statement: Extract<CudaLiteStatement, { kind: "if" }>,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  const active = context.nameFor(`bg_subgroup_if_active_${statement.span.start}`);
  return [
    `${prefix}{`,
    `${indent(indentLevel + 1)}let ${active}: bool = ${emitTruthinessExpression(statement.condition, context)};`,
    ...emitStatementSequence(statement.consequent, context, indentLevel + 1, { activeFlag: active }),
    `${prefix}}`,
  ];
}

function isBarrierStatement(statement: CudaLiteStatement): boolean {
  return statement.kind === "expr" && isBarrierCall(statement.expression);
}

function functionsToEmit(ir: KernelIrModule): readonly CudaLiteDeviceFunction[] {
  const launchCallees = new Set(collectKernelLaunchCallees(ir.body));
  const reachable = new Set<string>();
  const markReachable = (fn: CudaLiteDeviceFunction): void => {
    const linkName = deviceFunctionLinkName(fn, ir);
    if (reachable.has(linkName)) return;
    reachable.add(linkName);
    visitBody(fn.body);
  };
  const visitBody = (statements: readonly CudaLiteStatement[]): void => {
    walkCudaLiteExpressions(statements, (expression) => {
      const reduceOpName = cooperativeReduceDeviceFunctionName(expression);
      if (reduceOpName !== undefined) {
        const reduceOp = resolveDeviceFunctionForCall(ir, reduceOpName, 2);
        if (reduceOp) markReachable(reduceOp);
      }
      if (expression.kind !== "call") return;
      const name = expressionName(expression.callee);
      if (name === undefined || launchCallees.has(name)) return;
      const fn = resolveDeviceFunctionForCall(ir, name, expression.args.length);
      if (fn) markReachable(fn);
    });
  };
  visitBody(ir.body);
  return ir.functions.filter((fn) => fn.name !== ir.name && reachable.has(deviceFunctionLinkName(fn, ir)));
}

function deviceFunctionParamNeedsMutableBinding(
  fn: CudaLiteDeviceFunction,
  paramName: string,
  assignedNames: ReadonlySet<string>,
): boolean {
  if (assignedNames.has(paramName)) return true;
  let addressTaken = false;
  walkCudaLiteExpressions(fn.body, (expression) => {
    if (
      expression.kind === "unary" &&
      expression.operator === "&" &&
      expression.argument.kind === "identifier" &&
      expression.argument.name === paramName
    ) {
      addressTaken = true;
    }
  });
  return addressTaken;
}

function emitInlineAsmStatement(
  statement: Extract<CudaLiteStatement, { kind: "asm" }>,
  context: EmitContext,
): string {
  const op = classifyInlineAsm(statement.template);
  const outputs = statement.outputs ?? (statement.output === undefined ? [] : [statement.output]);
  if (op?.kind === "laneid" && statement.inputs.length === 0 && outputs.length === 1) {
    return `${emitExpression(outputs[0]!, context)} = ${emitInlineU32Output(outputs[0]!, `u32(${emitLocalLinearRank(context)} % 32)`, context)}`;
  }
  if (op?.kind === "lanemask-lt" && statement.inputs.length === 0 && outputs.length === 1) {
    const lane = `u32(${emitLocalLinearRank(context)} & 31)`;
    const mask = `select(0u, ((1u << ${lane}) - 1u), ${lane} > 0u)`;
    return `${emitExpression(outputs[0]!, context)} = ${emitInlineU32Output(outputs[0]!, mask, context)}`;
  }
  if (op?.kind === "globaltimer-u64" && statement.inputs.length === 0 && outputs.length === 1) {
    const tick = `((workgroup_id.x * ${context.ir.workgroupSize[0]}u) + u32(${emitLocalLinearRank(context)}))`;
    return `${emitExpression(outputs[0]!, context)} = ${emitInlineU32Output(outputs[0]!, tick, context)}`;
  }
  if (op?.kind === "bfind-u32" && statement.inputs.length === 1 && outputs.length === 1) {
    return `${emitExpression(outputs[0]!, context)} = (31u - countLeadingZeros(u32(${emitExpression(statement.inputs[0]!, context)})))`;
  }
  if (op?.kind === "u8x4-sad-add" && statement.inputs.length === 3 && outputs.length === 1) {
    return `${emitExpression(outputs[0]!, context)} = ${emitInlineU32Output(outputs[0]!, emitU8x4SadAddExpression(statement.inputs, context), context)}`;
  }
  if (op?.kind === "ldmatrix" && statement.inputs.length === 1 && outputs.length === op.matrices) {
    const base = `u32(${emitExpression(statement.inputs[0]!, context)})`;
    const tag = op.transposed ? "0x80000000u" : "0u";
    return outputs.map((output, index) => {
      const carrier = `(${tag} + ${base} + ${index * 2}u)`;
      return `${emitExpression(output, context)} = ${emitInlineU32Output(output, carrier, context)}`;
    }).join("\n");
  }
  if (op?.kind === "mma-m16n8k16") {
    return emitMmaM16N8K16Statement(statement, outputs, op.accumulator, context);
  }
  if (op?.kind !== "fma-rn-f32" || statement.inputs.length !== 2 || outputs.length !== 1) {
    throw featureError("unsupported-inline-asm", `only ${inlineAsmSupportedList()} inline PTX are supported in WGSL output`);
  }
  const target = emitExpression(outputs[0]!, context);
  return `${target} = fma(${emitExpression(statement.inputs[0]!, context)}, ${emitExpression(statement.inputs[1]!, context)}, ${target})`;
}

function emitMmaM16N8K16Statement(
  statement: Extract<CudaLiteStatement, { kind: "asm" }>,
  outputs: readonly CudaLiteExpression[],
  accumulator: "f16" | "f32",
  context: EmitContext,
): string {
  if (accumulator === "f16") {
    if (outputs.length !== 2 || statement.inputs.length !== 8) {
      throw featureError("invalid-inline-asm-operands", "mma.m16n8k16 f16 inline PTX operand mismatch");
    }
    return outputs.map((output, index) => {
      const a = `u32(${emitExpression(statement.inputs[index % 4]!, context)})`;
      const b = `u32(${emitExpression(statement.inputs[4 + (index % 2)]!, context)})`;
      const c = `u32(${emitExpression(statement.inputs[6 + index]!, context)})`;
      const value = `pack2x16float(unpack2x16float(${c}) + (unpack2x16float(${a}) * unpack2x16float(${b})))`;
      return `${emitExpression(output, context)} = ${emitInlineU32Output(output, value, context)}`;
    }).join("\n");
  }
  if (outputs.length !== 4 || statement.inputs.length !== 10) {
    throw featureError("invalid-inline-asm-operands", "mma.m16n8k16 f32 inline PTX operand mismatch");
  }
  return outputs.map((output, index) => {
    const a = `u32(${emitExpression(statement.inputs[index % 4]!, context)})`;
    const b = `u32(${emitExpression(statement.inputs[4 + (index % 2)]!, context)})`;
    const c = emitMmaF32AccumulatorInput(statement.inputs[6 + index]!, context);
    const value = `(${c} + dot(unpack2x16float(${a}), unpack2x16float(${b})))`;
    return `${emitExpression(output, context)} = ${emitMmaF32AccumulatorOutput(output, value, context)}`;
  }).join("\n");
}

function emitMmaF32AccumulatorInput(expression: CudaLiteExpression, context: EmitContext): string {
  const value = emitExpression(expression, context);
  const type = expressionValueTypeForEmit(expression, context);
  const scalar = type === undefined || isCudaVectorType(type) ? undefined : cudaScalarWgslType(type);
  if (scalar === "u32") return `bitcast<f32>(${value})`;
  if (scalar === "i32") return `bitcast<f32>(u32(${value}))`;
  if (scalar === "f16") return `f32(${value})`;
  return value;
}

function emitMmaF32AccumulatorOutput(target: CudaLiteExpression, value: string, context: EmitContext): string {
  const type = expressionValueTypeForEmit(target, context);
  const scalar = type === undefined || isCudaVectorType(type) ? undefined : cudaScalarWgslType(type);
  if (scalar === "u32") return `bitcast<u32>(${value})`;
  if (scalar === "i32") return `bitcast<i32>(${value})`;
  if (scalar === "f16") return `f16(${value})`;
  return value;
}

function emitU8x4SadAddExpression(inputs: readonly CudaLiteExpression[], context: EmitContext): string {
  const a = `u32(${emitExpression(inputs[0]!, context)})`;
  const b = `u32(${emitExpression(inputs[1]!, context)})`;
  const c = `u32(${emitExpression(inputs[2]!, context)})`;
  const lanes = [0, 8, 16, 24].map((shift) => {
    const left = `((${a} >> ${shift}u) & 0xffu)`;
    const right = `((${b} >> ${shift}u) & 0xffu)`;
    return `(max(${left}, ${right}) - min(${left}, ${right}))`;
  });
  return `(${c} + ${lanes.join(" + ")})`;
}

function emitForVar(statement: CudaLiteVarDecl, context: EmitContext): string {
  if (statement.pointer && context.localPointerHandleFor(statement.name)) {
    throw featureError(
      "unsupported-local-pointer-for-init",
      "mutable local pointer declarations in for-loop initializers are not supported yet",
    );
  }
  if (statement.pointer) return `var ${context.nameFor(statement.name)}: u32${statement.init ? ` = ${emitExpression(statement.init, context)}` : " = 0u"}`;
  if (statement.dimensions.length > 0) return `var ${context.nameFor(statement.name)}: ${emitLocalArrayType(statement)}`;
  return `var ${context.nameFor(statement.name)}: ${wgslScalar(statement.valueType)}${statement.init ? ` = ${emitLocalInit(statement, context)}` : ""}`;
}

function emitLocalPointerHandleDecl(
  statement: CudaLiteVarDecl,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  const [buffer, base] = emitLocalPointerHandleInit(statement, context);
  return [
    `${prefix}var ${context.nameFor(`${statement.name}_buffer`)}: u32 = ${buffer};`,
    `${prefix}var ${context.nameFor(`${statement.name}_base`)}: u32 = ${base};`,
  ];
}

function emitLocalPointerArrayDecl(
  statement: CudaLiteVarDecl,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  const length = localPointerArrayLength(statement);
  if (context.localPointerArrayRootFor(statement.name, statement.span)) {
    return [`${prefix}var ${context.nameFor(`${statement.name}_base`)}: array<u32, ${length}>;`];
  }
  return [
    `${prefix}var ${context.nameFor(`${statement.name}_buffer`)}: array<u32, ${length}>;`,
    `${prefix}var ${context.nameFor(`${statement.name}_base`)}: array<u32, ${length}>;`,
  ];
}

function localPointerArrayLength(statement: CudaLiteVarDecl): number {
  return statement.dimensions.reduce((product, dimension) => product * dimension, 1);
}

function emitLocalPointerHandleInit(statement: CudaLiteVarDecl, context: EmitContext): readonly [string, string] {
  if (!statement.init) return ["0u", "0u"];
  const parts = devicePointerArgumentParts(statement.init, context);
  if (!parts) {
    throw featureError(
      "unsupported-device-pointer-param",
      `local pointer '${statement.name}' at line ${statement.span.line} must initialize from modeled storage or shared memory`,
    );
  }
  return [parts.buffer, parts.base];
}

function emitLocalInit(statement: CudaLiteVarDecl, context: EmitContext): string {
  const value = statement.init ? emitExpression(statement.init, context) : zeroValue(statement.valueType);
  if (!statement.init) return value;
  if (statement.valueType === "bool") return emitTruthinessExpression(statement.init, context);
  if (isCudaVectorType(statement.valueType) || statement.valueType === "complex64") {
    if (
      statement.init.kind === "identifier" &&
      context.paramFor(statement.init.name) === undefined &&
      context.deviceGlobalFor(statement.init.name) === undefined &&
      !context.isUniformScalar(statement.init.name)
    ) {
      return emitExpression(statement.init, context);
    }
    return emitExpressionAsValueType(statement.init, statement.valueType, context);
  }
  if (statement.valueType === "uint") return emitExpressionAsWgslScalar(statement.init, "u32", context);
  if (statement.valueType === "int") return emitExpressionAsWgslScalar(statement.init, "i32", context);
  const sourceType = expressionValueTypeForEmit(statement.init, context);
  if (sourceType === undefined || sourceType === statement.valueType) return value;
  if ((statement.valueType === "float" || statement.valueType === "double" || statement.valueType === "bf16") && (sourceType === "int" || sourceType === "uint")) return `f32(${value})`;
  if (statement.valueType === "half" && sourceType !== "half") return `f16(${value})`;
  return value;
}

function emitDeviceFunction(
  fn: CudaLiteDeviceFunction,
  context: EmitContext,
  options: { readonly guardedBarrierClone?: boolean } = {},
): string[] {
  const cooperativeParams = new Map(fn.params
    .filter((param) => param.cooperativeGroupKind !== undefined)
    .map((param) => [param.name, cooperativeGroupForParam(param, context)] as const));
  const functionCooperativeGroups = collectCooperativeGroups(fn.body);
  const functionPointerParams = new Set(fn.params
    .filter((param) => param.pointer && usesFunctionLocalPointerParam(fn, param, context.ir))
    .map((param) => param.name));
  const functionLocalPointerHandles = collectLocalPointerHandles(fn.body, undefined, structuredPointerHandleRoots(context.ir));
  const functionPointerAliases = collectPointerAliases(fn.body, new Set(functionLocalPointerHandles.keys()));
  const functionLocalValueTypes = new Map(collectLocalValueTypes(fn.body));
  const functionParamNames = new Set(fn.params.map((param) => param.name));
  const functionLocalNames = new Set([...fn.params.map((param) => param.name), ...collectLocalNames(fn.body)]);
  const functionExpressionValueTypes = new WeakMap<CudaLiteExpression, CudaLiteScalarType | undefined>();
  const functionContext = withDevicePointerParams(
    {
      ...context,
      currentReturnType: fn.returnType,
      expressionValueTypes: functionExpressionValueTypes,
      localValueTypeFor(name) {
        return functionLocalValueTypes.get(name) ?? (functionParamNames.has(name) ? undefined : context.localValueTypeFor(name));
      },
      localPointerHandleFor(name) {
        return functionLocalPointerHandles.get(name) ?? context.localPointerHandleFor(name);
      },
      pointerAliasFor(name, span) {
        return pointerAliasDeclarationFor(functionPointerAliases, name, span) ?? context.pointerAliasFor(name, span);
      },
      paramFor(name) {
        return fn.params.find((param) => param.name === name) ?? context.paramFor(name);
      },
      cooperativeGroupFor(name) {
        return functionCooperativeGroups.get(name) ?? cooperativeParams.get(name) ?? context.cooperativeGroupFor(name);
      },
      isAtomicShared(name) {
        return functionLocalNames.has(name) ? false : context.isAtomicShared(name);
      },
    },
    fn.params.filter((param) => param.pointer && !functionPointerParams.has(param.name)),
    new Set([...fn.params.map((param) => param.name), ...collectLocalNames(fn.body)]),
  );
  const params = [
    ...fn.params.flatMap((param) => param.pointer
      ? functionPointerParams.has(param.name)
        ? [`${context.nameFor(param.name)}: ptr<function, ${wgslScalar(param.valueType)}>`]
        : [`${context.nameFor(`${param.name}_buffer_arg`)}: u32`, `${context.nameFor(`${param.name}_base_arg`)}: u32`]
      : param.cooperativeGroupKind !== undefined
        ? param.cooperativeGroupKind === "thread"
          ? [`${context.nameFor(`${param.name}_tile_size_arg`)}: u32`]
          : []
        : param.valueType === "texture2d"
          ? [`${context.nameFor(param.name)}: texture_2d<f32>`]
        : [`${context.nameFor(`${param.name}_arg`)}: ${wgslScalar(param.valueType)}`]),
    ...(options.guardedBarrierClone ? ["bg_call_active_arg: bool"] : []),
    "local_id: vec3<u32>",
    "workgroup_id: vec3<u32>",
    "num_workgroups: vec3<u32>",
  ];
  const returnType = fn.returnType === "void" ? "" : ` -> ${wgslScalar(fn.returnType)}`;
  const functionName = options.guardedBarrierClone
    ? guardedBarrierDeviceFunctionLinkName(fn, context.ir)
    : deviceFunctionLinkName(fn, context.ir);
  const lines = [`fn ${context.nameFor(functionName)}(${params.join(", ")})${returnType} {`];
  const assignedNames = collectAssignedNames(fn.body);
  if (options.guardedBarrierClone) {
    lines.push("  var bg_call_active: bool = bg_call_active_arg;");
  }
  for (const param of fn.params) {
    if (param.pointer) {
      if (functionPointerParams.has(param.name)) continue;
      const binding = deviceFunctionParamNeedsMutableBinding(fn, param.name, assignedNames) ? "var" : "let";
      lines.push(`  ${binding} ${context.nameFor(`${param.name}_buffer`)}: u32 = ${context.nameFor(`${param.name}_buffer_arg`)};`);
      lines.push(`  ${binding} ${context.nameFor(`${param.name}_base`)}: u32 = ${context.nameFor(`${param.name}_base_arg`)};`);
    } else if (param.cooperativeGroupKind !== undefined) {
      if (param.cooperativeGroupKind === "thread") {
        lines.push(`  let ${context.nameFor(`${param.name}_tile_size`)}: u32 = ${context.nameFor(`${param.name}_tile_size_arg`)};`);
      }
      continue;
    } else if (param.valueType === "texture2d") {
      continue;
    } else {
      const binding = deviceFunctionParamNeedsMutableBinding(fn, param.name, assignedNames) ? "var" : "let";
      lines.push(`  ${binding} ${context.nameFor(param.name)}: ${wgslScalar(param.valueType)} = ${context.nameFor(`${param.name}_arg`)};`);
    }
  }
  lines.push(...(options.guardedBarrierClone
    ? emitStatementSequence(fn.body, functionContext, 1, { activeFlag: "bg_call_active" })
    : fn.body.flatMap((statement) => emitStatement(statement, functionContext, 1))));
  if (fn.returnType !== "void" && !functionBodyHasReturn(fn.body)) {
    lines.push(`  return ${zeroValue(fn.returnType)};`);
  }
  lines.push("}");
  return lines;
}

function deviceFunctionNeedsGuardedBarrierClone(fn: CudaLiteDeviceFunction): boolean {
  return fn.returnType === "void" && fn.body.some(statementContainsBarrier);
}

function guardedBarrierDeviceFunctionLinkName(fn: CudaLiteDeviceFunction, ir: KernelIrModule): string {
  return `${deviceFunctionLinkName(fn, ir)}__bg_guarded_barrier`;
}

function withDevicePointerParams(
  context: EmitContext,
  params: readonly CudaLiteParam[],
  localNames: ReadonlySet<string> = new Set(),
): EmitContext {
  const pointerParams = new Map(params.map((param) => [param.name, param] as const));
  return {
    ...context,
    isLocalName(name) {
      return localNames.has(name) || context.isLocalName(name);
    },
    devicePointerParamFor(name) {
      return pointerParams.get(name);
    },
    mutablePointerBaseFor(name) {
      return pointerParams.has(name) ? context.nameFor(`${name}_base`) : context.mutablePointerBaseFor(name);
    },
  };
}

function usesFunctionLocalPointerParam(
  fn: CudaLiteDeviceFunction,
  param: CudaLiteParam,
  ir: KernelIrModule,
  memo: Map<string, boolean | "visiting"> = new Map(),
): boolean {
  if (!param.pointer) return false;
  const paramIndex = fn.params.findIndex((item) => item.name === param.name);
  if (paramIndex < 0) return false;
  const memoKey = `${fn.name}/${paramIndex}/${param.name}`;
  const cached = memo.get(memoKey);
  if (cached === "visiting") return false;
  if (cached !== undefined) return cached;
  memo.set(memoKey, "visiting");
  const sharedNames = new Set(ir.sharedDeclarations.map((shared) => shared.name));
  let sawCall = false;
  let allCallsUseLocalAddress = true;
  const roots: Array<{
    readonly statements: readonly CudaLiteStatement[];
    readonly localPointerParams: ReadonlySet<string>;
  }> = [
    { statements: ir.body, localPointerParams: new Set() },
    ...ir.functions.map((caller) => ({
      statements: caller.body,
      localPointerParams: new Set(caller.params
        .filter((callerParam) => callerParam.pointer && usesFunctionLocalPointerParam(caller, callerParam, ir, memo))
        .map((callerParam) => callerParam.name)),
    })),
  ];
  for (const { statements, localPointerParams } of roots) {
    const localArrayNames = new Set(collectLocalArrays(statements).keys());
    const localPointerArrayNames = new Set(collectLocalPointerArrayRoots(statements).keys());
    walkCudaLiteExpressions(statements, (expression) => {
      if (!allCallsUseLocalAddress || expression.kind !== "call" || expressionName(expression.callee) !== fn.name) return;
      sawCall = true;
      const arg = expression.args[paramIndex];
      if (!arg || !isFunctionLocalPointerArgument(arg, sharedNames, localArrayNames, localPointerArrayNames, localPointerParams)) allCallsUseLocalAddress = false;
    });
  }
  const result = sawCall && allCallsUseLocalAddress;
  memo.set(memoKey, result);
  return result;
}

function emitFunctionLocalPointerArgument(expression: CudaLiteExpression, context: EmitContext): string {
  if (expression.kind === "identifier" && context.localArrayFor(expression.name, expression.span)) {
    return `&${context.nameFor(expression.name)}[0]`;
  }
  if (expression.kind === "index" && expression.target.kind === "identifier") {
    const root = context.localPointerArrayRootFor(expression.target.name, expression.target.span);
    if (root) {
      return `&${context.nameFor(root.name)}[${context.nameFor(`${expression.target.name}_base`)}[u32(${emitExpression(expression.index, context)})]]`;
    }
  }
  return emitExpression(expression, context);
}

function isFunctionLocalPointerArgument(
  expression: CudaLiteExpression,
  sharedNames: ReadonlySet<string>,
  localArrayNames: ReadonlySet<string>,
  localPointerArrayNames: ReadonlySet<string>,
  localPointerParamNames: ReadonlySet<string>,
): boolean {
  if (expression.kind === "identifier") {
    return localArrayNames.has(expression.name) || localPointerParamNames.has(expression.name);
  }
  if (expression.kind === "index" && expression.target.kind === "identifier") {
    return localPointerArrayNames.has(expression.target.name);
  }
  return expression.kind === "unary" &&
    expression.operator === "&" &&
    expression.argument.kind === "identifier" &&
    !sharedNames.has(expression.argument.name);
}

function emitBf162PackExpression(value: string): string {
  return `((bitcast<u32>(${value}.x) >> 16u) | (bitcast<u32>(${value}.y) & 0xffff0000u))`;
}

function emitBf162UnpackExpression(bits: string): string {
  return `vec2<f32>(bitcast<f32>((${bits} & 0x0000ffffu) << 16u), bitcast<f32>(${bits} & 0xffff0000u))`;
}

interface DevicePointerLValue {
  readonly buffer: string;
  readonly index: string;
  readonly valueType: CudaLiteScalarType;
  readonly fieldIndex?: number;
}

const devicePointerCallbacks: WgslDevicePointerCallbacks = {
  isNullPointerExpression,
  emitExpression: (expression, context) => emitExpression(expression, context as EmitContext),
  emitTruthinessExpression: (expression, context) => emitTruthinessExpression(expression, context as EmitContext),
  pointerBaseExpression: (rootName, context) => pointerBaseExpression(rootName, context as EmitContext),
  sharedDeclarationFor: (rootName, context) => sharedDeclarationFor(rootName, context as EmitContext),
  featureError,
};

const storageViewCallbacks: WgslStorageViewCallbacks = {
  emitExpression: (expression, context) => emitExpression(expression, context as EmitContext),
  emitExpressionAsWgslScalar: (expression, type, context) => emitExpressionAsWgslScalar(expression, type, context as EmitContext),
  pointerBaseExpression: (rootName, context) => pointerBaseExpression(rootName, context as EmitContext),
  sharedDeclarationFor: (rootName, context) => sharedDeclarationFor(rootName, context as EmitContext),
  devicePointerValueTypeForExpression: (expression, context) => devicePointerValueTypeForExpression(expression, context as EmitContext),
};

const atomicCallbacks: WgslAtomicCallbacks = {
  emitExpression: (expression, context, mode) => emitExpression(expression, context as EmitContext, mode),
  emitExpressionAsValueType: (expression, valueType, context) => emitExpressionAsValueType(expression, valueType, context as EmitContext),
  emitExpressionAsWgslScalar: (expression, type, context) => emitExpressionAsWgslScalar(expression, type, context as EmitContext),
  devicePointerArgumentParts: (expression, context) => devicePointerArgumentParts(expression, context as EmitContext),
  devicePointerValueTypeForExpression: (expression, context) => devicePointerValueTypeForExpression(expression, context as EmitContext),
  emitPointerIndex: (rootName, index, context) => emitPointerIndex(rootName, index, context as EmitContext),
  emitPointerAliasIndex: (alias, index, context) => emitPointerAliasIndex(alias, index, context as EmitContext),
  storageViewForPointerExpression: (expression, index, context) => storageViewForPointerExpression(expression, index, context as EmitContext),
  sharedDeclarationFor: (name, context) => sharedDeclarationFor(name, context as EmitContext),
  flattenedPointerAlias: (name, span, context) => flattenedPointerAlias(name, span, context as EmitContext),
};

function emitDevicePointerArgument(expression: CudaLiteExpression, context: EmitContext): readonly [string, string] {
  return resolveDevicePointerArgument(expression, context, devicePointerCallbacks);
}

function devicePointerArgumentParts(expression: CudaLiteExpression, context: EmitContext): DevicePointerParts | undefined {
  return resolveDevicePointerArgumentParts(expression, context, devicePointerCallbacks);
}

function emitTruthinessExpression(expression: CudaLiteExpression, context: EmitContext): string {
  const pointer = devicePointerArgumentParts(expression, context);
  if (pointer) return `(${pointer.buffer} != ${NULL_DEVICE_POINTER_BUFFER})`;
  if (
    expression.kind === "call" &&
    expression.callee.kind === "member" &&
    (expression.callee.property === "any" || expression.callee.property === "all")
  ) {
    return emitExpression(expression, context);
  }
  const value = emitExpression(expression, context);
  const type = expressionWgslScalarType(expression, context);
  if (type === "bool") return value;
  if (type === "u32") return `(${value} != 0u)`;
  if (type === "f32" || type === "f16") return `(${value} != ${type}(0))`;
  return `(${value} != 0)`;
}

function emitReturnValue(expression: CudaLiteExpression, context: EmitContext): string {
  return context.currentReturnType === undefined || context.currentReturnType === "void"
    ? emitExpression(expression, context)
    : emitExpressionAsValueType(expression, context.currentReturnType, context);
}

function devicePointerValueTypeForExpression(expression: CudaLiteExpression, context: EmitContext): CudaLiteScalarType {
  if (expression.kind === "unary" && expression.operator === "&" && expression.argument.kind === "member") {
    const objectType = devicePointerValueTypeForExpression(expression.argument.object, context);
    return isCudaVectorType(objectType) ? cudaVectorScalarType(objectType) ?? "float" : objectType;
  }
  if (expression.kind === "conditional") {
    return isNullPointerExpression(expression.consequent)
      ? devicePointerValueTypeForExpression(expression.alternate, context)
      : devicePointerValueTypeForExpression(expression.consequent, context);
  }
  if (expression.kind === "cast" && expression.pointer) return expression.valueType;
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    const pointer = expression.args[0];
    return pointer ? devicePointerValueTypeForExpression(pointer, context) : "float";
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    return devicePointerValueTypeForExpression(expression.left, context);
  }
  const root = rootIdentifier(expression);
  if (root) {
    const pointerArray = context.localPointerArrayFor(root, expression.span);
    if (pointerArray) return pointerArray.valueType;
    const handle = context.localPointerHandleFor(root);
    if (handle) return handle.valueType;
    const alias = context.pointerAliasFor(root, expression.span);
    if (alias?.valueType) return alias.valueType;
    const param = context.devicePointerParamFor(root) ?? context.paramFor(root);
    if (param?.pointer) return param.valueType;
    const shared = sharedDeclarationFor(root, context);
    if (shared) return shared.valueType;
    const local = localArrayForStorageView(root, expression.span, context);
    if (local) return local.valueType;
    const global = context.deviceGlobalFor(root);
    if (global) return global.valueType;
  }
  return "float";
}

function devicePointerParamForIndex(
  expression: Extract<CudaLiteExpression, { kind: "index" }>,
  context: EmitContext,
): CudaLiteParam | undefined {
  return expression.target.kind === "identifier"
    ? context.mutablePointerBaseFor(expression.target.name) === context.nameFor(`${expression.target.name}_base`)
      ? context.devicePointerParamFor(expression.target.name)
      : undefined
    : undefined;
}

function devicePointerIndexExpression(
  name: string,
  index: CudaLiteExpression,
  context: EmitContext,
): string {
  return `(${context.nameFor(`${name}_base`)} + ${emitDevicePointerIndexDelta(index, context.devicePointerParamFor(name)?.valueType, context)})`;
}

function emitDevicePointerIndexDelta(
  index: CudaLiteExpression,
  valueType: CudaLiteScalarType | undefined,
  context: EmitContext,
): string {
  const raw = `u32(${emitExpression(index, context)})`;
  const lanes = isCudaVectorType(valueType) ? cudaVectorLaneCount(valueType) : 1;
  return lanes <= 1 ? raw : `(${raw} * ${lanes}u)`;
}

function devicePointerLValue(expression: CudaLiteExpression, context: EmitContext): DevicePointerLValue | undefined {
  if (expression.kind === "unary" && expression.operator === "*") {
    const parts = devicePointerArgumentParts(expression.argument, context);
    if (!parts) return undefined;
    return {
      buffer: parts.buffer,
      index: parts.base,
      valueType: devicePointerValueTypeForExpression(expression.argument, context),
    };
  }
  if (expression.kind === "member") {
    const base = devicePointerLValue(expression.object, context);
    if (!base || !isCudaVectorType(base.valueType)) return undefined;
    const fieldIndex = cudaVectorFieldIndex(base.valueType, expression.property);
    return fieldIndex === undefined ? undefined : { ...base, fieldIndex };
  }
  if (expression.kind === "index") {
    const parts = devicePointerArgumentParts(expression.target, context);
    if (!parts) return undefined;
    const valueType = devicePointerValueTypeForExpression(expression.target, context);
    return {
      buffer: parts.buffer,
      index: `(${parts.base} + ${emitDevicePointerIndexDelta(expression.index, valueType, context)})`,
      valueType,
    };
  }
  if (expression.kind === "identifier" && context.devicePointerParamFor(expression.name)) {
    const parts = devicePointerArgumentParts(expression, context);
    if (!parts) return undefined;
    return {
      buffer: parts.buffer,
      index: parts.base,
      valueType: devicePointerValueTypeForExpression(expression, context),
    };
  }
  if (expression.kind === "unary" && expression.operator === "*") {
    const parts = devicePointerArgumentParts(expression.argument, context);
    if (!parts) return undefined;
    return {
      buffer: parts.buffer,
      index: parts.base,
      valueType: devicePointerValueTypeForExpression(expression.argument, context),
    };
  }
  return undefined;
}

function isWritableDevicePointerExpression(expression: CudaLiteExpression, context: EmitContext): boolean {
  if (expression.kind === "conditional") {
    return isWritableDevicePointerExpression(expression.consequent, context) &&
      isWritableDevicePointerExpression(expression.alternate, context);
  }
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    const pointer = expression.args[0];
    return pointer !== undefined && isWritableDevicePointerExpression(pointer, context);
  }
  if (expression.kind === "cast" && expression.pointer) return isWritableDevicePointerExpression(expression.expression, context);
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    return isWritableDevicePointerExpression(expression.left, context);
  }
  if (expression.kind === "index" && expression.target.kind === "identifier") {
    return context.localPointerArrayFor(expression.target.name, expression.target.span) !== undefined;
  }
  if (expression.kind !== "identifier") return false;
  return context.localPointerHandleFor(expression.name) !== undefined ||
    context.devicePointerParamFor(expression.name) !== undefined ||
    context.localPointerArrayFor(expression.name, expression.span) !== undefined ||
    context.pointerAliasFor(expression.name, expression.span) !== undefined;
}

function emitDirectPointerIndexAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string | undefined {
  if (expression.left.kind !== "index") return undefined;
  const target = expression.left.target;
  if (
    target.kind === "identifier" &&
    context.devicePointerParamFor(target.name) === undefined &&
    context.ir.params.some((param) => param.pointer && param.name === target.name)
  ) {
    return undefined;
  }
  if (!isWritableDevicePointerExpression(target, context)) return undefined;
  const parts = devicePointerArgumentParts(target, context);
  if (!parts) return undefined;
  const valueType = devicePointerValueTypeForExpression(expression.left.target, context);
  const index = `(${parts.base} + ${emitDevicePointerIndexDelta(expression.left.index, valueType, context)})`;
  const right = emitPointerAssignmentValue(expression.right, valueType, context);
  const value = expression.operator === "="
    ? right
    : `(${pointerReadHelperName(valueType)}(${parts.buffer}, ${index}) ${expression.operator.slice(0, -1)} ${right})`;
  return `${pointerWriteHelperName(valueType)}(${parts.buffer}, ${index}, ${value})`;
}

function emitPointerAssignmentValue(
  expression: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: EmitContext,
): string {
  if (!isCudaVectorType(valueType)) return emitExpressionAsValueType(expression, valueType, context);
  if (expression.kind === "initializer") return emitExpressionAsValueType(expression, valueType, context);
  if (expression.kind === "unary" && expression.operator === "*") {
    const parts = devicePointerArgumentParts(expression.argument, context);
    if (parts) return `${pointerReadHelperName(valueType)}(${parts.buffer}, ${parts.base})`;
  }
  const expressionType = expressionValueTypeForEmit(expression, context);
  if (expressionType === valueType) return emitExpression(expression, context);
  const value = emitExpression(expression, context);
  if (isCudaVectorType(expressionType)) return emitVectorConversionConstructor(valueType, expression, context);
  if (
    expression.kind === "identifier" &&
    context.paramFor(expression.name) === undefined &&
    context.deviceGlobalFor(expression.name) === undefined &&
    !context.isUniformScalar(expression.name)
  ) {
    return value;
  }
  return emitVectorSplat(valueType, castExpressionToVectorScalar(value, valueType));
}

function emitExpression(expression: CudaLiteExpression, context: EmitContext, mode: EmitMode = "value"): string {
  switch (expression.kind) {
    case "number":
      return emitNumberLiteral(expression.raw);
    case "string":
      return expression.raw;
    case "initializer":
      throw featureError("unsupported-local-array-init", "braced initializer is only valid in a declaration");
    case "identifier":
      return emitIdentifier(expression.name, context, mode);
    case "cast": {
      if (expression.pointer) return emitExpression(expression.expression, context);
      const sourceType = expressionValueTypeForEmit(expression.expression, context);
      if (expression.valueType === "complex64") return emitExpressionAsValueType(expression.expression, "complex64", context);
      if (isCudaVectorType(expression.valueType)) return emitVectorConversionConstructor(expression.valueType, expression.expression, context);
      const value = emitExpression(expression.expression, context);
      if (sourceType === "complex64") return `${wgslScalar(expression.valueType)}(${value}.x)`;
      if (isCudaVectorType(sourceType)) return `${wgslScalar(expression.valueType)}(${value}.x)`;
      return `${wgslScalar(expression.valueType)}(${value})`;
    }
    case "member":
      return emitMember(expression, context);
    case "index": {
      const matrixLane = emitMatrixTileLaneAccessExpression(expression, context);
      if (matrixLane) return matrixLane;
      const localVectorAddressView = localVectorAddressViewForIndex(expression, context);
      if (localVectorAddressView) return localVectorAddressView;
      if (expression.target.kind === "identifier" && context.localPointerArrayFor(expression.target.name, expression.target.span)) {
        const localRoot = context.localPointerArrayRootFor(expression.target.name, expression.target.span);
        if (localRoot) {
          return `${context.nameFor(`${expression.target.name}_base`)}[u32(${emitExpression(expression.index, context)})]`;
        }
        const parts = devicePointerArgumentParts(expression, context);
        return mode === "lvalue"
          ? `${context.nameFor(`${expression.target.name}_base`)}[u32(${emitExpression(expression.index, context)})]`
          : parts?.base ?? "0u";
      }
      if (expression.target.kind === "identifier") {
        const localType = context.localValueTypeFor(expression.target.name);
        if (isCudaVectorType(localType)) {
          return `${context.nameFor(expression.target.name)}[u32(${emitExpression(expression.index, context)})]`;
        }
      }
      const storageView = storageViewLValue(expression, context);
      if (storageView && isCudaVectorType(storageView.valueType)) {
        const shared = storageView.rootName ? sharedDeclarationFor(storageView.rootName, context) : undefined;
        if (shared) return emitSharedPointerRead(shared, storageView.index, context.ir, context, storageView.valueType, storageView.subElementLane);
        const local = storageView.rootName ? localArrayForStorageView(storageView.rootName, expression.span, context) : undefined;
        if (local) return emitLocalPointerRead(local, storageView.index, storageView.valueType, context, storageView.subElementLane);
        const param = storageView.rootName ? context.paramFor(storageView.rootName) : undefined;
        if (param) return emitPointerStorageRead(param, storageView.index, context.ir, context, storageView.valueType);
        const global = storageView.rootName ? context.deviceGlobalFor(storageView.rootName) : undefined;
        if (global && context.ir.atomicDeviceGlobals.includes(global.name)) {
          return emitDeviceGlobalPointerRead(global, storageView.index, context.ir, context, storageView.valueType);
        }
        return emitVectorStorageReadAt(storageView.name, storageView.valueType, storageView.index);
      }
      const poolAccess = poolAccessForIndex(expression, context);
      if (poolAccess) return emitPoolRead(poolAccess, context, (item) => emitExpression(item, context));
      const pointerParam = devicePointerParamForIndex(expression, context);
      if (pointerParam) {
        return `${pointerReadHelperName(pointerParam.valueType)}(${context.nameFor(`${pointerParam.name}_buffer`)}, ${devicePointerIndexExpression(pointerParam.name, expression.index, context)})`;
      }
      if (mode === "value" && expression.target.kind !== "identifier") {
        const pointerParts = devicePointerArgumentParts(expression.target, context);
        if (pointerParts) {
          const valueType = devicePointerValueTypeForExpression(expression.target, context);
          return `${pointerReadHelperName(valueType)}(${pointerParts.buffer}, (${pointerParts.base} + ${emitDevicePointerIndexDelta(expression.index, valueType, context)}))`;
        }
      }
      if (expression.target.kind === "identifier") {
        const handle = context.localPointerHandleFor(expression.target.name);
        if (handle) {
          const base = context.nameFor(`${handle.name}_base`);
          const buffer = context.nameFor(`${handle.name}_buffer`);
          const index = `(${base} + ${emitDevicePointerIndexDelta(expression.index, handle.valueType, context)})`;
          return `${pointerReadHelperName(handle.valueType)}(${buffer}, ${index})`;
        }
        const alias = flattenedPointerAlias(expression.target.name, expression.target.span, context);
        if (alias) {
          const pointerParam = context.devicePointerParamFor(alias.rootName);
          if (pointerParam) {
            const valueType = alias.valueType ?? pointerParam.valueType;
            return `${pointerReadHelperName(valueType)}(${context.nameFor(`${alias.rootName}_buffer`)}, ${emitPointerAliasIndex(alias, expression.index, context)})`;
          }
          if (alias.valueType && isCudaVectorType(alias.valueType)) {
            const view = createStorageView(alias.rootName, alias.baseIndex, expression.index, alias.valueType, context);
            const shared = sharedDeclarationFor(alias.rootName, context);
            if (shared) return emitSharedPointerRead(shared, view.index, context.ir, context, alias.valueType, view.subElementLane);
            const local = localArrayForStorageView(alias.rootName, expression.span, context);
            if (local) return emitLocalPointerRead(local, view.index, alias.valueType, context, view.subElementLane);
            const param = context.paramFor(alias.rootName);
            if (param) return emitPointerStorageRead(param, view.index, context.ir, context, alias.valueType);
            const global = context.deviceGlobalFor(alias.rootName);
            if (global && context.ir.atomicDeviceGlobals.includes(global.name)) {
              return emitDeviceGlobalVectorFlatRead(global, view.index, alias.valueType, context.ir, context);
            }
            return emitVectorStorageReadAt(context.nameFor(alias.rootName), alias.valueType, view.index);
          }
          if (alias.valueType) {
            const shared = sharedDeclarationFor(alias.rootName, context);
            if (shared) {
              const view = createStorageView(alias.rootName, alias.baseIndex, expression.index, alias.valueType, context);
              return emitSharedPointerRead(shared, view.index, context.ir, context, alias.valueType, view.subElementLane);
            }
            const local = localArrayForStorageView(alias.rootName, expression.span, context);
            if (local) {
              const view = createStorageView(alias.rootName, alias.baseIndex, expression.index, alias.valueType, context);
              return emitLocalPointerRead(local, view.index, alias.valueType, context, view.subElementLane);
            }
            const param = context.paramFor(alias.rootName);
            if (param) {
              const view = createStorageView(alias.rootName, alias.baseIndex, expression.index, alias.valueType, context);
              return emitPointerStorageRead(param, view.index, context.ir, context, alias.valueType);
            }
            const global = context.deviceGlobalFor(alias.rootName);
            if (global) {
              const view = createStorageView(alias.rootName, alias.baseIndex, expression.index, alias.valueType, context);
              return emitDeviceGlobalPointerRead(global, view.index, context.ir, context, alias.valueType);
            }
          }
          return `${context.nameFor(alias.rootName)}[${emitPointerAliasIndex(alias, expression.index, context)}]`;
        }
      }
      const root = rootIdentifier(expression);
      const index = root && expression.target.kind === "identifier"
        ? emitPointerIndex(root, expression.index, context)
        : emitExpression(expression.index, context);
      const param = root ? context.paramFor(root) : undefined;
      const global = root ? context.deviceGlobalFor(root) : undefined;
      const pointerParamForValue = root ? context.devicePointerParamFor(root) : undefined;
      if (
        mode === "value" &&
        pointerParamForValue &&
        context.mutablePointerBaseFor(pointerParamForValue.name) === context.nameFor(`${pointerParamForValue.name}_base`) &&
        isCudaVectorType(pointerParamForValue.valueType) &&
        expression.target.kind === "identifier"
      ) {
        return `${pointerReadHelperName(pointerParamForValue.valueType)}(${context.nameFor(`${pointerParamForValue.name}_buffer`)}, (${context.nameFor(`${pointerParamForValue.name}_base`)} + ${emitDevicePointerIndexDelta(expression.index, pointerParamForValue.valueType, context)}))`;
      }
      if (mode === "value" && param && isCudaVectorType(param.valueType) && expression.target.kind === "identifier") {
        if (context.ir.atomicParams.includes(param.name)) {
          return emitPointerVectorFlatRead(
            param,
            vectorStorageBase(index, cudaVectorLaneCount(param.valueType)),
            param.valueType,
            context.ir,
            context,
          );
        }
        return emitVectorStorageRead(context.nameFor(root!), param.valueType, index);
      }
      if (mode === "value" && param?.valueType === "complex64" && expression.target.kind === "identifier") {
        return emitPointerStorageRead(param, index, context.ir, context);
      }
      if (mode === "lvalue" && param?.valueType === "complex64" && expression.target.kind === "identifier") {
        return `${context.nameFor(root!)}[u32(${index})]`;
      }
      if (global && expression.target.kind === "identifier") {
        if (mode === "value" && isCudaVectorType(global.valueType) && context.ir.atomicDeviceGlobals.includes(global.name)) {
          return emitDeviceGlobalVectorFlatRead(
            global,
            vectorStorageBase(index, cudaVectorLaneCount(global.valueType)),
            global.valueType,
            context.ir,
            context,
          );
        }
        return mode === "value"
          ? emitDeviceGlobalPointerRead(global, index, context.ir, context)
          : `${context.nameFor(root!)}[${index}]`;
      }
      const access = `${emitExpression(expression.target, context, "lvalue")}[${index}]`;
      if (mode === "value" && param?.valueType === "bool") return `(${access} != 0u)`;
      if (mode === "value" && param && context.ir.atomicParams.includes(param.name)) {
        const loaded = `atomicLoad(&${access})`;
        return param.valueType === "float" || param.valueType === "double" ? `bitcast<f32>(${loaded})` : loaded;
      }
      if (mode === "value" && root && context.isAtomicShared(root)) {
        const shared = sharedDeclarationFor(root, context);
        const loaded = `atomicLoad(&${access})`;
        return shared?.valueType === "float" || shared?.valueType === "double" ? `bitcast<f32>(${loaded})` : loaded;
      }
      return access;
    }
    case "call":
      return emitCall(expression, context);
    case "unary":
      if (expression.operator === "&") return `&${emitExpression(expression.argument, context, "lvalue")}`;
      if (expression.operator === "*") return emitDeref(expression.argument, context);
      if (expression.operator === "!") return `(!${emitTruthinessExpression(expression.argument, context)})`;
      return `(${expression.operator}${emitExpression(expression.argument, context)})`;
    case "binary": {
      const pointerComparison = emitPointerComparison(expression, context);
      if (pointerComparison) return pointerComparison;
      const pointerDifference = emitPointerDifference(expression, context);
      if (pointerDifference) return pointerDifference;
      const vectorArithmetic = emitVectorArithmetic(expression, context);
      if (vectorArithmetic) return vectorArithmetic;
      return emitScalarBinaryExpression(expression, context);
    }
    case "conditional":
      return emitConditionalExpression(expression, context);
    case "assignment":
      return emitAssignment(expression, context);
    case "update":
      return emitUpdateExpression(expression, context);
    case "sequence":
      throw featureError("unsupported-sequence-expression", "comma expressions are only supported in for-loop clauses");
  }
}

function emitExpressionStatement(expression: CudaLiteExpression, context: EmitContext): string {
  if (expression.kind === "number" || expression.kind === "string") return "";
  const source = emitExpression(expression, context);
  if (expression.kind === "call") {
    const name = expressionName(expression.callee);
    if (isAtomicCasCallName(name)) return `_ = ${source.replace(/\.old_value$/u, "")}`;
    if (isAtomicExchangeCallName(name)) return `_ = ${source.replace(/^bitcast<f32>\((atomicExchange\(.*\))\)$/u, "$1")}`;
    if (isAtomicReturnCallName(name)) return `_ = ${source}`;
  }
  return source;
}

function emitConditionalExpression(expression: CudaLiteConditionalExpression, context: EmitContext): string {
  const valueType = expressionValueTypeForEmit(expression, context);
  const alternate = valueType
    ? emitExpressionAsValueType(expression.alternate, valueType, context)
    : emitExpression(expression.alternate, context);
  const consequent = valueType
    ? emitExpressionAsValueType(expression.consequent, valueType, context)
    : emitExpression(expression.consequent, context);
  return `select(${alternate}, ${consequent}, ${emitTruthinessExpression(expression.condition, context)})`;
}

function emitForLoopWithContinuing(
  statement: Extract<CudaLiteStatement, { kind: "for" }>,
  context: EmitContext,
  indentLevel: number,
  activeFlag?: string,
): string[] {
  const prefix = indent(indentLevel);
  const lines: string[] = [];
  if (statement.init?.kind === "var") {
    lines.push(`${prefix}${emitForVar(statement.init, context)};`);
  } else if (statement.init) {
    for (const expression of sequenceItems(statement.init)) lines.push(`${prefix}${emitExpression(expression, context)};`);
  }
  lines.push(`${prefix}loop {`);
  if (statement.condition) lines.push(`${indent(indentLevel + 1)}if (!(${emitTruthinessExpression(statement.condition, context)})) { break; }`);
  lines.push(...emitStatementSequence(statement.body, context, indentLevel + 1, activeFlag ? { activeFlag } : undefined));
  if (statement.update) {
    lines.push(`${indent(indentLevel + 1)}continuing {`);
    for (const expression of sequenceItems(statement.update)) {
      lines.push(`${indent(indentLevel + 2)}${emitExpression(expression, context)};`);
    }
    lines.push(`${indent(indentLevel + 1)}}`);
  }
  lines.push(`${prefix}}`);
  return lines;
}

function emitForLoopWithActiveBreak(
  statement: Extract<CudaLiteStatement, { kind: "for" }>,
  context: EmitContext,
  indentLevel: number,
  activeFlag: string,
): string[] {
  if (statement.update?.kind === "sequence" || statement.init?.kind === "sequence") {
    return emitForLoopWithContinuing(statement, context, indentLevel, activeFlag);
  }
  const prefix = indent(indentLevel);
  const init = statement.init?.kind === "var"
    ? emitForVar(statement.init, context)
    : statement.init
      ? emitExpression(statement.init, context)
      : "";
  const condition = statement.condition ? emitTruthinessExpression(statement.condition, context) : "true";
  const update = statement.update ? emitExpression(statement.update, context) : "";
  const lines = [`${prefix}for (${init}; ${condition}; ${update}) {`];
  lines.push(...emitStatementSequence(statement.body, context, indentLevel + 1, { activeFlag }));
  lines.push(`${prefix}}`);
  return lines;
}

function emitWhileLoopWithEarlyBreakBeforeBarrier(
  statement: Extract<CudaLiteStatement, { kind: "while" }>,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  const active = context.nameFor(`bg_loop_active_${statement.span.start}`);
  const breakIndex = statement.body.findIndex((child, index) =>
    splitIfTrailingBreak(child) !== undefined && statement.body.slice(index + 1).some((item) => statementContainsBarrierLike(item, context))
  );
  const lines = [`${prefix}{`, `${indent(indentLevel + 1)}var ${active}: bool = true;`];
  const condition = emitTruthinessExpression(statement.condition, context);
  if (condition === "true" || condition === "(1 != 0)" || condition === "1") {
    const iter = context.nameFor(`bg_loop_iter_${statement.span.start}`);
    lines.push(`${indent(indentLevel + 1)}for (var ${iter}: u32 = 0u; ${iter} < ${barrierWhileIterationBound(statement)}u; ${iter} = ${iter} + 1u) {`);
    lines.push(...emitStatementSequence(statement.body, context, indentLevel + 2, { activeFlag: active }));
    lines.push(`${indent(indentLevel + 1)}}`);
    lines.push(`${prefix}}`);
    return lines;
  } else {
    lines.push(`${indent(indentLevel + 1)}while ((${condition}) && ${active}) {`);
  }
  lines.push(...statement.body.slice(0, breakIndex).flatMap((child) => emitStatement(child, context, indentLevel + 2)));
  for (let index = breakIndex; index < statement.body.length; index++) {
    const child = statement.body[index]!;
    const split = splitIfTrailingBreak(child);
    if (split) {
      lines.push(...emitIfTrailingBreakAsActiveFlag(split, active, context, indentLevel + 2));
    } else {
      lines.push(...emitStatementWithActiveFlag(child, active, context, indentLevel + 2));
    }
  }
  if (condition === "true" || condition === "(1 != 0)" || condition === "1") {
    lines.push(`${indent(indentLevel + 2)}if (!(${active})) { break; }`);
  }
  lines.push(`${indent(indentLevel + 1)}}`);
  lines.push(`${prefix}}`);
  return lines;
}

function barrierWhileIterationBound(_statement: Extract<CudaLiteStatement, { kind: "while" }>): number {
  return 256;
}

function emitBoundedBarrierForLoop(
  statement: Extract<CudaLiteStatement, { kind: "for" }>,
  context: EmitContext,
  indentLevel: number,
  activeFlag?: string,
): string[] {
  const outerPrefix = indent(indentLevel);
  const prefix = indent(indentLevel + 1);
  const loopActive = context.nameFor(`bg_barrier_loop_active_${statement.span.start}`);
  const loopIter = context.nameFor(`bg_barrier_loop_iter_${statement.span.start}`);
  const active = loopActive;
  const updateActive = loopActive;
  const lines: string[] = [];
  lines.push(`${outerPrefix}{`);
  if (statement.init?.kind === "var") {
    lines.push(`${prefix}${emitForVar(statement.init, context)};`);
  } else if (statement.init) {
    for (const expression of sequenceItems(statement.init)) lines.push(`${prefix}${emitExpression(expression, context)};`);
  }
  lines.push(`${prefix}var ${loopActive}: bool = ${activeFlag ?? "true"};`);
  lines.push(`${prefix}for (var ${loopIter}: u32 = 0u; ${loopIter} < ${barrierLoopIterationBound(statement, context)}; ${loopIter} = ${loopIter} + 1u) {`);
  if (statement.condition) {
    lines.push(`${indent(indentLevel + 1)}${loopActive} = (${loopActive} && (${emitBarrierLoopActiveCondition(statement.condition, context)}));`);
  }
  lines.push(...emitStatementSequence(statement.body, context, indentLevel + 1, { activeFlag: active }));
  if (statement.update) {
    for (const expression of sequenceItems(statement.update)) {
      const predicated = emitPredicatedScalarExpressionStatement(expression, updateActive, context, indentLevel + 1);
      if (predicated) {
        lines.push(...predicated);
      } else {
        lines.push(`${indent(indentLevel + 1)}if (${active}) {`);
        lines.push(`${indent(indentLevel + 2)}${emitExpression(expression, context)};`);
        lines.push(`${indent(indentLevel + 1)}}`);
      }
    }
  }
  lines.push(`${prefix}}`);
  lines.push(`${outerPrefix}}`);
  return lines;
}

function emitBarrierLoopActiveCondition(expression: CudaLiteExpression, context: EmitContext): string {
  if (
    expression.kind === "call" &&
    expression.callee.kind === "member" &&
    expression.callee.property === "any" &&
    expression.callee.object.kind === "identifier" &&
    context.cooperativeGroupFor(expression.callee.object.name) &&
    expression.args[0]
  ) {
    return emitTruthinessExpression(expression.args[0], context);
  }
  return emitTruthinessExpression(expression, context);
}

function barrierLoopIterationBound(statement: Extract<CudaLiteStatement, { kind: "for" }>, context: EmitContext): string {
  const dynamicBound = dynamicBarrierLoopIterationBound(statement, context);
  if (dynamicBound) return dynamicBound;
  const update = statement.update;
  if (update?.kind === "assignment" && (update.operator === "<<=" || update.operator === ">>=")) return "32u";
  if (update?.kind === "assignment" && update.operator === "=" && update.right.kind === "binary") {
    if (update.right.operator === "<<" || update.right.operator === ">>") return "32u";
    if (update.right.operator === "+" && update.right.right.kind === "number" && Number(update.right.right.value) >= 16) return "128u";
  }
  return "256u";
}

function dynamicBarrierLoopIterationBound(statement: Extract<CudaLiteStatement, { kind: "for" }>, context: EmitContext): string | undefined {
  const loopName = forLoopVariableName(statement);
  if (!loopName || statement.condition?.kind !== "binary") return undefined;
  if (statement.condition.left.kind !== "identifier" || statement.condition.left.name !== loopName) return undefined;
  if (statement.condition.operator !== "<" && statement.condition.operator !== "<=") return undefined;
  if (!expressionIsUniformForLoopBound(statement.condition.right, context)) return undefined;
  const step = positiveLoopStepExpression(statement, loopName, context);
  if (!step) return undefined;
  const upper = statement.condition.operator === "<="
    ? `(${emitExpression(statement.condition.right, context)} + 1)`
    : emitExpression(statement.condition.right, context);
  const safeStep = `max(1, ${step})`;
  return `max(1u, u32(max(0, (((${upper}) + ${safeStep} - 1) / ${safeStep}))))`;
}

function expressionIsUniformForLoopBound(expression: CudaLiteExpression, context: EmitContext): boolean {
  switch (expression.kind) {
    case "number":
      return true;
    case "identifier":
      return context.paramFor(expression.name) !== undefined;
    case "member":
      return expression.object.kind === "identifier" &&
        (expression.object.name === "blockIdx" || expression.object.name === "blockDim" || expression.object.name === "gridDim");
    case "unary":
      return expressionIsUniformForLoopBound(expression.argument, context);
    case "cast":
      return expressionIsUniformForLoopBound(expression.expression, context);
    case "binary":
      return expressionIsUniformForLoopBound(expression.left, context) && expressionIsUniformForLoopBound(expression.right, context);
    case "conditional":
      return expressionIsUniformForLoopBound(expression.condition, context) &&
        expressionIsUniformForLoopBound(expression.consequent, context) &&
        expressionIsUniformForLoopBound(expression.alternate, context);
    default:
      return false;
  }
}

function forLoopVariableName(statement: Extract<CudaLiteStatement, { kind: "for" }>): string | undefined {
  if (statement.init?.kind === "var") return statement.init.name;
  if (statement.init?.kind === "assignment" && statement.init.left.kind === "identifier") return statement.init.left.name;
  return undefined;
}

function positiveLoopStepExpression(
  statement: Extract<CudaLiteStatement, { kind: "for" }>,
  loopName: string,
  context: EmitContext,
): string | undefined {
  const update = statement.update;
  if (!update) return undefined;
  if (update.kind === "update" && update.argument.kind === "identifier" && update.argument.name === loopName) {
    return update.operator === "++" ? "1" : undefined;
  }
  if (update.kind !== "assignment" || update.left.kind !== "identifier" || update.left.name !== loopName) return undefined;
  if (update.operator === "+=") return expressionIsUniformForLoopBound(update.right, context) ? emitExpression(update.right, context) : undefined;
  if (update.operator !== "=" || update.right.kind !== "binary" || update.right.operator !== "+") return undefined;
  if (update.right.left.kind === "identifier" && update.right.left.name === loopName) {
    return expressionIsUniformForLoopBound(update.right.right, context) ? emitExpression(update.right.right, context) : undefined;
  }
  if (update.right.right.kind === "identifier" && update.right.right.name === loopName) {
    return expressionIsUniformForLoopBound(update.right.left, context) ? emitExpression(update.right.left, context) : undefined;
  }
  return undefined;
}

function sequenceItems(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  return expression.kind === "sequence" ? expression.expressions : [expression];
}

function emitPredicatedScalarExpressionStatement(
  expression: CudaLiteExpression,
  activeFlag: string,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (expression.kind === "assignment") {
    return emitPredicatedScalarAssignment(expression, activeFlag, context, indentLevel);
  }
  if (expression.kind !== "update" || expression.argument.kind !== "identifier") return undefined;
  const root = rootIdentifier(expression.argument);
  if (!root || !context.isLocalName(root)) return undefined;
  const valueType = expressionValueTypeForEmit(expression.argument, context);
  if (valueType !== "uint" && valueType !== "int") return undefined;
  const left = emitExpression(expression.argument, context, "lvalue");
  const current = emitExpressionAsValueType(expression.argument, valueType, context);
  const step = valueType === "uint" ? "1u" : "1";
  const operator = expression.operator === "++" ? "+" : "-";
  return [`${indent(indentLevel)}${left} = select(${current}, (${current} ${operator} ${step}), ${activeFlag});`];
}

function emitAssignmentStatement(
  expression: CudaLiteAssignmentExpression,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const prefix = indent(indentLevel);
  const nested = splitNestedAssignmentExpression(expression.right);
  if (!nested) return [`${prefix}${emitAssignment(expression, context)};`];
  return [
    ...emitAssignmentStatement(nested.assignment, context, indentLevel),
    `${prefix}${emitAssignment({ ...expression, right: nested.replacement }, context)};`,
  ];
}

function splitNestedAssignmentExpression(
  expression: CudaLiteExpression,
): { readonly assignment: CudaLiteAssignmentExpression; readonly replacement: CudaLiteExpression } | undefined {
  if (expression.kind === "assignment") return { assignment: expression, replacement: expression.left };
  if (expression.kind === "cast") {
    const nested = splitNestedAssignmentExpression(expression.expression);
    if (!nested) return undefined;
    return {
      assignment: nested.assignment,
      replacement: { ...expression, expression: nested.replacement },
    };
  }
  return undefined;
}

function emitPredicatedScalarAssignmentStatement(
  statement: CudaLiteStatement,
  activeFlag: string,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (statement.kind !== "expr" || statement.expression.kind !== "assignment") return undefined;
  return emitPredicatedScalarAssignment(statement.expression, activeFlag, context, indentLevel);
}

function emitGuardedAssignmentWithHoistedUniformRhs(
  statement: CudaLiteStatement,
  guardSource: string,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (statement.kind !== "expr" || statement.expression.kind !== "assignment") return undefined;
  const expression = statement.expression;
  if (expression.operator !== "=") return undefined;
  if (!expressionContainsSubgroupCall(expression.right)) return undefined;
  if (expressionContainsAssignment(expression.right) || expressionContainsSideEffectingCall(expression.right)) return undefined;
  const valueType = expressionValueTypeForEmit(expression.left, context) ?? expressionValueTypeForEmit(expression.right, context);
  if (!valueType || valueType === "void" || valueType === "complex64") return undefined;
  const prefix = indent(indentLevel);
  const tempName = context.nameFor(`bg_guarded_value_${expression.span.start}`);
  const replacement: CudaLiteExpression = { kind: "identifier", name: tempName, span: expression.right.span };
  return [
    `${prefix}let ${tempName}: ${wgslScalar(valueType)} = ${emitExpressionAsValueType(expression.right, valueType, context)};`,
    `${prefix}if (${guardSource}) {`,
    `${indent(indentLevel + 1)}${emitAssignment({ ...expression, right: replacement }, context)};`,
    `${prefix}}`,
  ];
}

function emitLocalConstantAssignmentStatement(
  statement: CudaLiteStatement,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (statement.kind !== "expr" || statement.expression.kind !== "assignment" || statement.expression.operator !== "=") return undefined;
  const root = rootIdentifier(statement.expression.left);
  if (!root || !context.isLocalName(root)) return undefined;
  if (
    context.paramFor(root) ||
    context.deviceGlobalFor(root) ||
    context.isAtomicShared(root) ||
    sharedDeclarationFor(root, context) ||
    localArrayForStorageView(root, statement.expression.left.span, context) ||
    storageViewLValue(statement.expression.left, context) ||
    scalarStorageViewLValue(statement.expression.left, context) ||
    scalarParamStorageViewLValue(statement.expression.left, context) ||
    devicePointerLValue(statement.expression.left, context) ||
    poolAccessForIndex(statement.expression.left, context)
  ) {
    return undefined;
  }
  if (!expressionIsLocalConstantSafe(statement.expression.right, context)) return undefined;
  return [`${indent(indentLevel)}${emitExpression(statement.expression.left, context, "lvalue")} = ${emitExpression(statement.expression.right, context)};`];
}

function expressionIsLocalConstantSafe(expression: CudaLiteExpression, context: EmitContext): boolean {
  switch (expression.kind) {
    case "number":
    case "string":
      return true;
    case "identifier":
      return CUDA_NAMED_CONSTANTS.has(expression.name) ||
        context.isLocalName(expression.name) ||
        (context.paramFor(expression.name) !== undefined && context.paramFor(expression.name)?.pointer !== true) ||
        context.isUniformScalar(expression.name);
    case "member":
      return expressionIsLocalConstantSafe(expression.object, context) ||
        (expression.object.kind === "identifier" &&
          (expression.object.name === "threadIdx" ||
            expression.object.name === "blockIdx" ||
            expression.object.name === "blockDim" ||
            expression.object.name === "gridDim"));
    case "cast":
      return expressionIsLocalConstantSafe(expression.expression, context);
    case "unary":
      return expressionIsLocalConstantSafe(expression.argument, context);
    case "binary":
      return expressionIsLocalConstantSafe(expression.left, context) && expressionIsLocalConstantSafe(expression.right, context);
    case "conditional":
      return expressionIsLocalConstantSafe(expression.condition, context) &&
        expressionIsLocalConstantSafe(expression.consequent, context) &&
        expressionIsLocalConstantSafe(expression.alternate, context);
    case "call": {
      const name = expressionName(expression.callee);
      return name !== undefined &&
        (cudaVectorConstructorType(name) !== undefined || name.startsWith("make_")) &&
        expression.args.every((arg) => expressionIsLocalConstantSafe(arg, context));
    }
    default:
      return false;
  }
}

function emitPredicatedScalarAssignment(
  expression: CudaLiteAssignmentExpression,
  activeFlag: string,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (expressionContainsAssignment(expression.right)) return undefined;
  const root = rootIdentifier(expression.left);
  if (!root || !context.isLocalName(root)) return undefined;
  if (
    context.paramFor(root) ||
    context.deviceGlobalFor(root) ||
    context.isAtomicShared(root) ||
    storageViewLValue(expression.left, context) ||
    scalarStorageViewLValue(expression.left, context) ||
    scalarParamStorageViewLValue(expression.left, context) ||
    devicePointerLValue(expression.left, context) ||
    poolAccessForIndex(expression.left, context)
  ) {
    return undefined;
  }
  if (expression.left.kind !== "identifier" && expression.left.kind !== "member") {
    const localArray = localArrayForStorageView(root, expression.left.span, context);
    if (!localArray || expression.left.kind !== "index") return undefined;
  }
  const valueType = expressionValueTypeForEmit(expression.left, context) ?? context.localValueTypeFor(root);
  if (!valueType || valueType === "void" || valueType === "complex64") return undefined;
  const left = emitExpression(expression.left, context, "lvalue");
  const current = emitExpressionAsValueType(expression.left, valueType, context);
  const value = predicatedScalarAssignmentValue(expression, valueType, context);
  if (!value) return undefined;
  if (expressionContainsSideEffectingCall(expression.right)) {
    const prefix = indent(indentLevel);
    return [
      `${prefix}if (${activeFlag}) {`,
      `${indent(indentLevel + 1)}${left} = ${value};`,
      `${prefix}}`,
    ];
  }
  if (predicatedValueNeedsUniformCallHoist(value)) {
    const prefix = indent(indentLevel);
    const tempName = context.nameFor(`bg_predicated_value_${expression.span.start}`);
    return [
      `${prefix}let ${tempName}: ${wgslScalar(valueType)} = ${value};`,
      `${prefix}${left} = select(${current}, ${tempName}, ${activeFlag});`,
    ];
  }
  return [`${indent(indentLevel)}${left} = select(${current}, ${value}, ${activeFlag});`];
}

function predicatedValueNeedsUniformCallHoist(value: string): boolean {
  return /\b(?:subgroup[A-Z][A-Za-z0-9_]*|bg_warp_(?:reduce|shuffle)_[A-Za-z0-9_]+)\s*\(/u.test(value);
}

function expressionContainsSideEffectingCall(expression: CudaLiteExpression): boolean {
  let found = false;
  walkCudaLiteExpressions([{ kind: "expr", expression, span: expression.span }], (item) => {
    if (item.kind !== "call") return;
    const name = expressionName(item.callee);
    if (
      name !== undefined &&
      (name.startsWith("atomic") ||
        CUDA_CACHE_HINT_STORES.has(name) ||
        name === "cudaMemcpy" ||
        name === "cudaMemcpyAsync" ||
        name === "cudaMemcpyPeerAsync")
    ) {
      found = true;
    }
  });
  return found;
}

function predicatedScalarAssignmentValue(
  expression: CudaLiteAssignmentExpression,
  valueType: CudaLiteScalarType,
  context: EmitContext,
): string | undefined {
  if (expressionContainsAssignment(expression.right)) return undefined;
  if (expression.operator === "=") {
    if (
      (isCudaVectorType(valueType) || valueType === "complex64") &&
      expression.right.kind === "identifier" &&
      context.paramFor(expression.right.name) === undefined &&
      context.deviceGlobalFor(expression.right.name) === undefined &&
      !context.isUniformScalar(expression.right.name)
    ) {
      return emitExpression(expression.right, context);
    }
    return emitExpressionAsValueType(expression.right, valueType, context);
  }
  const operator = expression.operator.slice(0, -1);
  if (!["+", "-", "*", "/", "<<", ">>", "&", "|", "^"].includes(operator)) return undefined;
  const left = emitExpressionAsValueType(expression.left, valueType, context);
  if (operator === "<<" || operator === ">>") {
    return `(${left} ${operator} ${emitExpressionAsWgslScalar(expression.right, "u32", context)})`;
  }
  if (operator === "&" || operator === "|" || operator === "^") {
    const localLeftType = expression.left.kind === "identifier" ? context.localValueTypeFor(expression.left.name) : undefined;
    const rightScalar = expressionWgslScalarType(expression.right, context);
    const target = valueType === "uint" || localLeftType === "uint" || rightScalar === "u32" ||
      (operator === "|" && expression.left.kind === "identifier")
      ? "u32"
      : "i32";
    const value = `(${emitExpressionAsWgslScalar(expression.left, target, context)} ${operator} ${emitExpressionAsWgslScalar(expression.right, target, context)})`;
    return target === "u32" ? value : valueType === "uint" ? `u32(${value})` : `i32(${value})`;
  }
  const right = emitExpressionAsValueType(expression.right, valueType, context);
  return `(${left} ${operator} ${right})`;
}

function emitFillRegsStatement(
  expression: CudaLiteExpression,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (expression.kind !== "call") return undefined;
  const name = expressionName(expression.callee);
  if (name !== "fill_1D_regs" && name !== "fill_2D_regs" && name !== "fill_3D_regs") return undefined;
  const target = expression.args[0];
  const value = expression.args[1];
  if (target?.kind !== "identifier" || !value) return undefined;
  const array = context.localArrayFor(target.name, target.span);
  if (!array) return undefined;
  return emitLocalArrayFill(context.nameFor(target.name), array.dimensions, emitExpression(value, context), indentLevel);
}

interface EmittedMatrixTile {
  readonly name: string;
  readonly spec: MatrixTileResolvedSpec;
  readonly base: string;
}

function emitWmmaStatement(
  expression: CudaLiteExpression,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (expression.kind !== "call") return undefined;
  const builtin = wmmaBuiltinName(expressionName(expression.callee));
  if (!builtin) return undefined;
  switch (builtin) {
    case "fill_fragment":
      return emitWmmaFillFragment(expression, context, indentLevel);
    case "load_matrix_sync":
      return emitWmmaLoadMatrixSync(expression, context, indentLevel);
    case "mma_sync":
      return emitWmmaMmaSync(expression, context, indentLevel);
    case "store_matrix_sync":
      return emitWmmaStoreMatrixSync(expression, context, indentLevel);
  }
}

function emitWmmaFillFragment(
  expression: CudaLiteCallExpression,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const tile = emitMatrixTileRef(expression.args[0], context, "wmma::fill_fragment");
  const value = emitMatrixTileValueForStore(emitExpression(expression.args[1]!, context), tile.spec);
  const index = `bg_wmma_i_${indentLevel}`;
  const prefix = indent(indentLevel);
  return [
    `${prefix}for (var ${index}: u32 = 0u; ${index} < ${matrixTileElementCount(tile.spec)}u; ${index} = ${index} + 1u) {`,
    `${indent(indentLevel + 1)}${emitMatrixTileAccess(tile, index)} = ${value};`,
    `${prefix}}`,
  ];
}

function emitWmmaLoadMatrixSync(
  expression: CudaLiteCallExpression,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const tile = emitMatrixTileRef(expression.args[0], context, "wmma::load_matrix_sync");
  const source = devicePointerArgumentParts(expression.args[1]!, context);
  if (!source) throw featureError("unsupported-wmma-pointer-operand", "wmma::load_matrix_sync source expects storage/shared pointer");
  const stride = `u32(${emitExpression(expression.args[2]!, context)})`;
  const layout = emitMatrixTileLayoutForCall(expression.args[3], tile.spec.layout ?? "row_major");
  const [rows, cols] = matrixTileRowsCols(tile.spec);
  const row = `bg_wmma_row_${indentLevel}`;
  const col = `bg_wmma_col_${indentLevel}`;
  const prefix = indent(indentLevel);
  const memIndex = emitMatrixTileMemoryIndex(row, col, stride, layout);
  const tileIndex = `(${row} * ${cols}u + ${col})`;
  const read = `${pointerReadHelperName(tile.spec.valueType)}(${source.buffer}, (${source.base} + ${memIndex}))`;
  return [
    `${prefix}for (var ${row}: u32 = 0u; ${row} < ${rows}u; ${row} = ${row} + 1u) {`,
    `${indent(indentLevel + 1)}for (var ${col}: u32 = 0u; ${col} < ${cols}u; ${col} = ${col} + 1u) {`,
    `${indent(indentLevel + 2)}${emitMatrixTileAccess(tile, tileIndex)} = ${emitMatrixTileValueForStore(read, tile.spec)};`,
    `${indent(indentLevel + 1)}}`,
    `${prefix}}`,
  ];
}

function emitWmmaMmaSync(
  expression: CudaLiteCallExpression,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const dst = emitMatrixTileRef(expression.args[0], context, "wmma::mma_sync destination");
  const a = emitMatrixTileRef(expression.args[1], context, "wmma::mma_sync A");
  const b = emitMatrixTileRef(expression.args[2], context, "wmma::mma_sync B");
  const c = emitMatrixTileRef(expression.args[3], context, "wmma::mma_sync accumulator");
  const row = `bg_wmma_row_${indentLevel}`;
  const col = `bg_wmma_col_${indentLevel}`;
  const kk = `bg_wmma_k_${indentLevel}`;
  const sum = `bg_wmma_sum_${indentLevel}`;
  const prefix = indent(indentLevel);
  const dstIndex = `(${row} * ${dst.spec.n}u + ${col})`;
  const aIndex = `(${row} * ${dst.spec.k}u + ${kk})`;
  const bIndex = `(${kk} * ${dst.spec.n}u + ${col})`;
  const integerMma = dst.spec.tileValueType === "s32" &&
    isMatrixTileByteValueType(a.spec.tileValueType) &&
    isMatrixTileByteValueType(b.spec.tileValueType);
  const sumType = integerMma ? "i32" : "f32";
  const sumInit = integerMma
    ? emitMatrixTileIntegerValue(emitMatrixTileAccess(c, dstIndex), c.spec)
    : `f32(${emitMatrixTileAccess(c, dstIndex)})`;
  const multiply = integerMma
    ? `(${emitMatrixTileIntegerValue(emitMatrixTileAccess(a, aIndex), a.spec)} * ${emitMatrixTileIntegerValue(emitMatrixTileAccess(b, bIndex), b.spec)})`
    : `f32(${emitMatrixTileAccess(a, aIndex)}) * f32(${emitMatrixTileAccess(b, bIndex)})`;
  return [
    `${prefix}for (var ${row}: u32 = 0u; ${row} < ${dst.spec.m}u; ${row} = ${row} + 1u) {`,
    `${indent(indentLevel + 1)}for (var ${col}: u32 = 0u; ${col} < ${dst.spec.n}u; ${col} = ${col} + 1u) {`,
    `${indent(indentLevel + 2)}var ${sum}: ${sumType} = ${sumInit};`,
    `${indent(indentLevel + 2)}for (var ${kk}: u32 = 0u; ${kk} < ${dst.spec.k}u; ${kk} = ${kk} + 1u) {`,
    `${indent(indentLevel + 3)}${sum} = ${sum} + ${multiply};`,
    `${indent(indentLevel + 2)}}`,
    `${indent(indentLevel + 2)}${emitMatrixTileAccess(dst, dstIndex)} = ${emitMatrixTileValueForStore(sum, dst.spec)};`,
    `${indent(indentLevel + 1)}}`,
    `${prefix}}`,
  ];
}

function emitWmmaStoreMatrixSync(
  expression: CudaLiteCallExpression,
  context: EmitContext,
  indentLevel: number,
): string[] {
  const target = devicePointerArgumentParts(expression.args[0]!, context);
  if (!target) throw featureError("unsupported-wmma-pointer-operand", "wmma::store_matrix_sync destination expects storage/shared pointer");
  const tile = emitMatrixTileRef(expression.args[1], context, "wmma::store_matrix_sync fragment");
  const stride = `u32(${emitExpression(expression.args[2]!, context)})`;
  const layout = emitMatrixTileLayoutForCall(expression.args[3], "mem_row_major");
  const [rows, cols] = matrixTileRowsCols(tile.spec);
  const row = `bg_wmma_row_${indentLevel}`;
  const col = `bg_wmma_col_${indentLevel}`;
  const prefix = indent(indentLevel);
  const memIndex = emitMatrixTileMemoryIndex(row, col, stride, layout);
  const tileIndex = `(${row} * ${cols}u + ${col})`;
  return [
    `${prefix}for (var ${row}: u32 = 0u; ${row} < ${rows}u; ${row} = ${row} + 1u) {`,
    `${indent(indentLevel + 1)}for (var ${col}: u32 = 0u; ${col} < ${cols}u; ${col} = ${col} + 1u) {`,
    `${indent(indentLevel + 2)}${pointerWriteHelperName(tile.spec.valueType)}(${target.buffer}, (${target.base} + ${memIndex}), ${emitMatrixTileAccess(tile, tileIndex)});`,
    `${indent(indentLevel + 1)}}`,
    `${prefix}}`,
  ];
}

function emitMatrixTileRef(
  expression: CudaLiteExpression | undefined,
  context: EmitContext,
  label: string,
): EmittedMatrixTile {
  if (!expression) throw featureError("unsupported-wmma-fragment-operand", `${label} expects WMMA fragment argument`);
  const ref = matrixTileReference(expression);
  if (!ref) throw featureError("unsupported-wmma-fragment-operand", `${label} expects WMMA fragment argument`);
  const declaration = context.localArrayFor(ref.root);
  const spec = declaration?.matrixTile ? resolveMatrixTileSpec(declaration.matrixTile) : undefined;
  if (!declaration || !spec) throw featureError("unsupported-wmma-fragment-operand", `'${ref.root}' is not a WMMA fragment`);
  const count = matrixTileElementCount(spec);
  const base = emitMatrixTileBase(ref.indices, declaration.dimensions, count, context);
  return { name: context.nameFor(ref.root), spec, base };
}

function emitMatrixTileBase(
  indices: readonly CudaLiteExpression[],
  dimensions: readonly number[],
  elementCount: number,
  context: EmitContext,
): string {
  if (indices.length !== dimensions.length) return "0u";
  if (indices.length === 0) return "0u";
  const terms = indices.map((index, axis) => {
    const stride = dimensions.slice(axis + 1).reduce((product, dimension) => product * dimension, elementCount);
    const value = `u32(${emitExpression(index, context)})`;
    return stride === 1 ? value : `(${value} * ${stride}u)`;
  });
  return terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
}

function emitMatrixTileAccess(tile: EmittedMatrixTile, index: string): string {
  return tile.base === "0u" ? `${tile.name}[${index}]` : `${tile.name}[(${tile.base} + ${index})]`;
}

function matrixTileRowsCols(tile: MatrixTileResolvedSpec): readonly [number, number] {
  if (tile.role === "matrix_a") return [tile.m, tile.k];
  if (tile.role === "matrix_b") return [tile.k, tile.n];
  return [tile.m, tile.n];
}

function emitMatrixTileMemoryIndex(row: string, col: string, stride: string, layout: MatrixTileLayout): string {
  return layout === "col_major" || layout === "mem_col_major"
    ? `(${col} * ${stride} + ${row})`
    : `(${row} * ${stride} + ${col})`;
}

function emitMatrixTileLayoutForCall(
  expression: CudaLiteExpression | undefined,
  fallback: MatrixTileLayout,
): MatrixTileLayout {
  if (!expression) return fallback;
  return normalizeMatrixTileLayout(expressionName(expression)) ?? fallback;
}

function emitMatrixTileValueForStore(value: string, tile: MatrixTileResolvedSpec): string {
  switch (tile.tileValueType) {
    case "f16":
      return `f16(${value})`;
    case "u8":
      return `(u32(${value}) & 0xffu)`;
    case "s8":
    case "s32":
      return `i32(${value})`;
    default:
      return value;
  }
}

function emitMatrixTileIntegerValue(value: string, tile: MatrixTileResolvedSpec): string {
  if (tile.tileValueType === "u8") return `i32(u32(${value}) & 0xffu)`;
  return `i32(${value})`;
}

function emitCpAsyncStatement(
  expression: CudaLiteExpression,
  context: EmitContext,
  indentLevel: number,
): string[] | undefined {
  if (expression.kind !== "call") return undefined;
  const name = expressionName(expression.callee);
  const prefix = indent(indentLevel);
  if (isCpAsyncFenceCall(name)) return [`${prefix}// cp.async fence omitted: ${name}`];
  if (!isCpAsyncCopyCall(name)) return undefined;
  const [dst, src, bytes] = expression.args;
  if (!dst || !src) return [`${prefix}// cp.async omitted: missing pointer operand`];
  const dstParts = devicePointerArgumentParts(dst, context);
  const srcParts = devicePointerArgumentParts(src, context);
  if (!dstParts || !srcParts) return [`${prefix}// cp.async byte-address copy omitted: ${name}`];
  const valueType = devicePointerValueTypeForExpression(src, context);
  const count = cpAsyncElementCount(bytes, valueType);
  const lines: string[] = [];
  for (let index = 0; index < count; index++) {
    const srcIndex = index === 0 ? srcParts.base : `(${srcParts.base} + ${index}u)`;
    const dstIndex = index === 0 ? dstParts.base : `(${dstParts.base} + ${index}u)`;
    lines.push(`${prefix}${pointerWriteHelperName(valueType)}(${dstParts.buffer}, ${dstIndex}, ${pointerReadHelperName(valueType)}(${srcParts.buffer}, ${srcIndex}));`);
  }
  return lines;
}

function isCpAsyncCopyCall(name: string | undefined): boolean {
  return name === "CP_ASYNC_CA" || name === "CP_ASYNC_CG" || name === "CP_ASYNC_BULK";
}

function isCpAsyncFenceCall(name: string | undefined): boolean {
  return name === "CP_ASYNC_COMMIT_GROUP" ||
    name === "CP_ASYNC_WAIT_ALL" ||
    name === "CP_ASYNC_WAIT_GROUP" ||
    name === "CP_ASYNC_BULK_COMMIT_GROUP" ||
    name === "CP_ASYNC_BULK_WAIT_ALL" ||
    name === "CP_ASYNC_BULK_WAIT_GROUP";
}

function cpAsyncElementCount(bytes: CudaLiteExpression | undefined, valueType: CudaLiteScalarType): number {
  if (bytes?.kind !== "number") return 1;
  const elementBytes = wgslElementByteSize(valueType);
  if (elementBytes <= 0) return 1;
  return Math.max(1, Math.min(16, Math.floor(bytes.value / elementBytes)));
}

function emitVectorConversionConstructor(
  targetType: CudaLiteVectorType,
  expression: CudaLiteExpression,
  context: EmitContext,
): string {
  const sourceType = expressionValueTypeForEmit(expression, context) ??
    (expression.kind === "identifier" ? context.localValueTypeFor(expression.name) : undefined);
  const value = emitExpression(expression, context);
  if (isComplexFloat2RepresentationPair(sourceType, targetType)) return value;
  if (!isCudaVectorType(sourceType)) {
    return emitVectorSplat(targetType, castExpressionToVectorScalar(value, targetType));
  }
  const fields = ["x", "y", "z", "w"];
  const scalar = cudaVectorScalarType(targetType) ?? "float";
  const lanes = Array.from({ length: cudaVectorLaneCount(targetType) }, (_unused, index) =>
    index < cudaVectorLaneCount(sourceType) ? `${wgslScalar(scalar)}(${value}.${fields[index]!})` : zeroValue(scalar));
  return `${wgslScalar(targetType)}(${lanes.join(", ")})`;
}

function isComplexFloat2RepresentationPair(
  sourceType: CudaLiteScalarType | undefined,
  targetType: CudaLiteScalarType,
): boolean {
  return (sourceType === "complex64" && targetType === "float2") ||
    (sourceType === "float2" && targetType === "complex64");
}

function emitMixedVectorConstructor(
  targetType: CudaLiteVectorType,
  expressions: readonly CudaLiteExpression[],
  context: EmitContext,
): string {
  const targetScalar = cudaVectorScalarType(targetType) ?? "float";
  if (!expressions.some((expression) => isCudaVectorType(expressionValueTypeForEmit(expression, context)))) {
    return emitVectorConstructor(targetType, expressions.map((expression) => emitExpressionAsValueType(expression, targetScalar, context)));
  }
  const lanes: string[] = [];
  for (const expression of expressions) {
    const sourceType = expressionValueTypeForEmit(expression, context);
    const value = emitExpression(expression, context);
    if (isCudaVectorType(sourceType)) {
      for (let lane = 0; lane < cudaVectorLaneCount(sourceType); lane++) {
        lanes.push(castExpressionToVectorScalar(`${value}.${vectorFieldName(lane)}`, targetType));
      }
    } else {
      lanes.push(castExpressionToVectorScalar(value, targetType));
    }
  }
  while (lanes.length < cudaVectorLaneCount(targetType)) lanes.push(zeroValue(targetScalar));
  return `${wgslScalar(targetType)}(${lanes.slice(0, cudaVectorLaneCount(targetType)).join(", ")})`;
}

function emitLocalArrayFill(
  name: string,
  dimensions: readonly number[],
  value: string,
  indentLevel: number,
  indexes: readonly string[] = [],
): string[] {
  if (indexes.length === dimensions.length) {
    return [`${indent(indentLevel)}${name}${indexes.map((index) => `[${index}]`).join("")} = ${value};`];
  }
  const loopName = `fill_${name}_${indexes.length}`;
  const lines = [
    `${indent(indentLevel)}for (var ${loopName}: i32 = 0; ${loopName} < ${dimensions[indexes.length] ?? 0}; ${loopName} = ${loopName} + 1) {`,
  ];
  lines.push(...emitLocalArrayFill(name, dimensions, value, indentLevel + 1, [...indexes, loopName]));
  lines.push(`${indent(indentLevel)}}`);
  return lines;
}

function emitPointerAliasIndex(alias: PointerAlias, index: CudaLiteExpression, context: EmitContext): string {
  return emitPointerAliasIndexImpl(alias, index, context, storageViewCallbacks);
}

function createStorageView(
  rootName: string,
  baseIndex: CudaLiteExpression,
  index: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: EmitContext,
): StorageView {
  return createStorageViewImpl(rootName, baseIndex, index, valueType, context, storageViewCallbacks);
}

function emitPointerIndex(rootName: string, index: CudaLiteExpression, context: EmitContext): string {
  return emitPointerIndexImpl(rootName, index, context, storageViewCallbacks);
}

function pointerBaseExpression(rootName: string, context: EmitContext): string | undefined {
  if (context.devicePointerParamFor(rootName)) return context.nameFor(`${rootName}_base`);
  const mutable = context.mutablePointerBaseFor(rootName);
  if (mutable) return mutable;
  const base = context.pointerBaseOffsetFieldFor(rootName);
  return base ? `${UNIFORM_PARAMS_NAME}.${context.nameFor(base)}` : undefined;
}

interface PointerComparisonTerm {
  readonly kind: "static" | "token" | "address";
  readonly value?: string;
  readonly buffer?: string;
  readonly base?: string;
  readonly pointerish: boolean;
  readonly nullish: boolean;
  readonly symbol?: string;
}

function emitPointerComparison(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): string | undefined {
  if (expression.operator !== "==" && expression.operator !== "!=") return undefined;
  const left = pointerComparisonTerm(expression.left, context);
  const right = pointerComparisonTerm(expression.right, context);
  if (!left || !right) return undefined;
  if (!left.pointerish && !right.pointerish && !left.nullish && !right.nullish) return undefined;

  if (left.kind === "static" && right.kind === "static") {
    const equal = staticPointerTermEqual(left, right);
    if (equal === undefined) {
      throw featureError(
        "unsupported-pointer-pointer-comparison",
        "WGSL output supports pointer comparison only against NULL/nullptr or the same pointer symbol",
      );
    }
    return expression.operator === "==" ? String(equal) : String(!equal);
  }

  if (left.kind === "address" && right.kind === "address") {
    const equal = `((${left.buffer} == ${right.buffer}) && (${left.base} == ${right.base}))`;
    return expression.operator === "==" ? equal : `(!${equal})`;
  }

  if (left.kind === "token" && right.kind === "token") {
    return `(${left.value} ${expression.operator} ${right.value})`;
  }

  const token = left.kind === "token" ? left : right.kind === "token" ? right : undefined;
  const address = left.kind === "address" ? left : right.kind === "address" ? right : undefined;
  const stat = left.kind === "static" ? left : right.kind === "static" ? right : undefined;
  if (token && stat?.nullish) {
    return `(${token.value} ${expression.operator} 0u)`;
  }
  if (address && stat?.nullish) {
    const buffer = address.buffer ?? "0u";
    if (/^\d+u$/u.test(buffer)) {
      const equal = buffer === NULL_DEVICE_POINTER_BUFFER;
      return expression.operator === "==" ? String(equal) : String(!equal);
    }
    const equal = `(${buffer} == ${NULL_DEVICE_POINTER_BUFFER})`;
    return expression.operator === "==" ? equal : `(!${equal})`;
  }
  throw featureError(
    "unsupported-pointer-pointer-comparison",
    "WGSL output supports pointer comparison only inside the same pointer address model",
  );
}

function emitPointerDifference(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): string | undefined {
  if (expression.operator !== "-") return undefined;
  if (!isPointerDifferenceOperand(expression.left, context) || !isPointerDifferenceOperand(expression.right, context)) return undefined;
  const left = devicePointerArgumentParts(expression.left, context);
  const right = devicePointerArgumentParts(expression.right, context);
  if (!left || !right) return undefined;
  const diff = `(i32(${left.base}) - i32(${right.base}))`;
  return left.buffer === right.buffer
    ? diff
    : `select(0, ${diff}, (${left.buffer} == ${right.buffer}))`;
}

function isPointerDifferenceOperand(expression: CudaLiteExpression, context: EmitContext): boolean {
  if (isNullPointerExpression(expression)) return true;
  if (expression.kind === "conditional") {
    return isPointerDifferenceOperand(expression.consequent, context) || isPointerDifferenceOperand(expression.alternate, context);
  }
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    const pointer = expression.args[0];
    return pointer !== undefined && isPointerDifferenceOperand(pointer, context);
  }
  if (expression.kind === "identifier") {
    return context.localPointerHandleFor(expression.name) !== undefined ||
      context.devicePointerParamFor(expression.name) !== undefined ||
      context.localPointerArrayFor(expression.name, expression.span) !== undefined ||
      context.pointerAliasFor(expression.name, expression.span) !== undefined;
  }
  if (expression.kind === "index" && expression.target.kind === "identifier") {
    return context.localPointerArrayFor(expression.target.name, expression.target.span) !== undefined;
  }
  if (expression.kind === "cast" && expression.pointer) return isPointerDifferenceOperand(expression.expression, context);
  if (expression.kind === "unary" && expression.operator === "&") return true;
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    return isPointerDifferenceOperand(expression.left, context);
  }
  return false;
}

function staticPointerTermEqual(left: PointerComparisonTerm, right: PointerComparisonTerm): boolean | undefined {
  if (left.nullish || right.nullish) return left.nullish === right.nullish && left.value === right.value;
  if (left.symbol && right.symbol && left.symbol === right.symbol) return true;
  return undefined;
}

function isNullPointerExpression(expression: CudaLiteExpression): boolean {
  if (expression.kind === "number") return expression.value === 0;
  if (expression.kind !== "identifier") return false;
  if (expression.name === "nullptr" || expression.name === "NULL") return true;
  const namedConstant = CUDA_NAMED_CONSTANTS.get(expression.name);
  return namedConstant?.valueType === "voidptr" && namedConstant.value === 0;
}

function pointerComparisonTerm(expression: CudaLiteExpression, context: EmitContext): PointerComparisonTerm | undefined {
  if (expression.kind === "identifier") {
    if (expression.name === "nullptr") return { kind: "static", value: "null", pointerish: false, nullish: true };
    const namedConstant = CUDA_NAMED_CONSTANTS.get(expression.name);
    if (namedConstant?.valueType === "voidptr" && namedConstant.value === 0) {
      return { kind: "static", value: "null", pointerish: false, nullish: true };
    }
    if (context.poolPointerFor(expression.name)) {
      return { kind: "token", value: emitIdentifier(expression.name, context), pointerish: true, nullish: false, symbol: expression.name };
    }
    const localType = context.localValueTypeFor(expression.name);
    if (localType === "voidptr") {
      return { kind: "token", value: emitIdentifier(expression.name, context), pointerish: true, nullish: false, symbol: expression.name };
    }
  }
  if (expression.kind === "number" && expression.value === 0) {
    return { kind: "static", value: "null", pointerish: false, nullish: true };
  }
  if (expression.kind === "cast" && expression.pointer) {
    return pointerComparisonTerm(expression.expression, context);
  }
  if (expression.kind === "call") {
    const pool = poolPointerForAllocationCall(expression);
    if (pool) return { kind: "token", value: emitCall(expression, context), pointerish: true, nullish: false };
  }
  if (expression.kind === "identifier" && context.deviceGlobalFor(expression.name)?.dimensions.length === 0) {
    return undefined;
  }
  const address = devicePointerArgumentParts(expression, context);
  if (address) {
    const symbol = expressionName(expression);
    return {
      kind: "address",
      buffer: address.buffer,
      base: address.base,
      pointerish: true,
      nullish: false,
      ...(symbol === undefined ? {} : { symbol }),
    };
  }
  return undefined;
}

function emitIdentifier(name: string, context: EmitContext, mode: EmitMode = "value"): string {
  if (name === "nullptr") return "0u";
  const namedConstant = CUDA_NAMED_CONSTANTS.get(name);
  if (namedConstant) return namedConstant.wgsl;
  if (name === "threadIdx" || name === "blockIdx" || name === "blockDim" || name === "gridDim") return name;
  if (context.isAtomicShared(name) && mode === "value") {
    const shared = sharedDeclarationFor(name, context);
    const loaded = `atomicLoad(&${context.nameFor(name)})`;
    return shared?.valueType === "float" ? `bitcast<f32>(${loaded})` : loaded;
  }
  if (context.isLocalName(name)) return context.nameFor(name);
  const global = context.deviceGlobalFor(name);
  if (global) {
    if (mode === "lvalue") return `${context.nameFor(name)}[0u]`;
    return emitDeviceGlobalPointerRead(global, "0u", context.ir, context);
  }
  const constant = context.ir.constants.find((item) => item.name === name);
  if (constant && isExternalConstantBinding(constant) && constant.dimensions.length === 0) {
    return emitConstantPointerRead(constant, "0u", context);
  }
  const param = context.paramFor(name);
  if ((param && !param.pointer && !isSurfaceParam(param)) || context.isUniformScalar(name)) {
    return emitUniformScalarRead(name, context);
  }
  return context.nameFor(name);
}

function emitUniformScalarRead(name: string, context: EmitContext): string {
  return context.uniformScalarTypeFor(name) === "bool"
    ? `(${UNIFORM_PARAMS_NAME}.${context.nameFor(name)} != 0u)`
    : `${UNIFORM_PARAMS_NAME}.${context.nameFor(name)}`;
}

function sharedDeclarationFor(name: string, context: EmitContext): CudaLiteVarDecl | undefined {
  return context.ir.sharedDeclarations.find((item) => item.name === name);
}

function localArrayForStorageView(name: string, span: SourceSpan | undefined, context: EmitContext): CudaLiteVarDecl | undefined {
  return localArrayForStorageViewImpl(name, span, context);
}

function emitDeref(expression: CudaLiteExpression, context: EmitContext): string {
  const localBitReinterpret = emitLocalBitReinterpretDeref(expression, context);
  if (localBitReinterpret) return localBitReinterpret;
  const localPointer = localPointerArrayLocalAccess(expression, context);
  if (localPointer) return localPointer;
  if (expression.kind === "identifier") {
    const alias = flattenedPointerAlias(expression.name, expression.span, context);
    if (alias) {
      const pointerParam = context.devicePointerParamFor(alias.rootName);
      if (pointerParam) {
        const valueType = alias.valueType ?? pointerParam.valueType;
        return `${pointerReadHelperName(valueType)}(${context.nameFor(`${alias.rootName}_buffer`)}, ${emitPointerAliasIndex(alias, zeroExpression(expression.span), context)})`;
      }
      if (alias.valueType) {
        const index = emitPointerAliasIndex(alias, zeroExpression(expression.span), context);
        const shared = sharedDeclarationFor(alias.rootName, context);
        if (shared) return emitSharedPointerRead(shared, index, context.ir, context, alias.valueType);
        const local = localArrayForStorageView(alias.rootName, expression.span, context);
        if (local) return emitLocalPointerRead(local, index, alias.valueType, context);
        const global = context.deviceGlobalFor(alias.rootName);
        if (global) return emitDeviceGlobalPointerRead(global, index, context.ir, context, alias.valueType);
      }
      return `${context.nameFor(alias.rootName)}[${emitPointerAliasIndex(alias, zeroExpression(expression.span), context)}]`;
    }
    const param = context.paramFor(expression.name);
    if (param?.pointer && !context.devicePointerParamFor(expression.name)) {
      if (context.isLocalName(expression.name)) return `*${context.nameFor(expression.name)}`;
      const index = emitPointerIndex(expression.name, zeroExpression(expression.span), context);
      const access = `${context.nameFor(expression.name)}[${index}]`;
      if (context.ir.atomicParams.includes(param.name)) {
        const loaded = `atomicLoad(&${access})`;
        return param.valueType === "float" || param.valueType === "double" ? `bitcast<f32>(${loaded})` : loaded;
      }
      return emitPointerStorageRead(param, index, context.ir, context);
    }
  }
  if (expression.kind === "cast" && expression.pointer) {
    const poolPointer = poolPointerExpressionInfo(expression.expression, context);
    if (poolPointer) {
      return emitPoolRead({
        poolName: poolPointer.poolName,
        pointerExpression: expression.expression,
        indexExpression: poolZeroIndex(expression.span),
        valueType: expression.valueType,
        ...(poolPointer.rawBuffer ? { rawBuffer: true } : {}),
      }, context, (item) => emitExpression(item, context));
    }
  }
  const storageView = storageViewForPointerExpression(expression, zeroExpression(expression.span), context);
  if (storageView && isCudaVectorType(storageView.valueType)) {
    const shared = sharedDeclarationFor(storageView.rootName, context);
    if (shared) return emitSharedPointerRead(shared, storageView.index, context.ir, context, storageView.valueType, storageView.subElementLane);
    const local = localArrayForStorageView(storageView.rootName, expression.span, context);
    if (local) return emitLocalPointerRead(local, storageView.index, storageView.valueType, context, storageView.subElementLane);
    return emitVectorStorageReadAt(context.nameFor(storageView.rootName), storageView.valueType, storageView.index);
  }
  const parts = devicePointerArgumentParts(expression, context);
  if (parts) {
    const valueType = devicePointerValueTypeForExpression(expression, context);
    return `${pointerReadHelperName(valueType)}(${parts.buffer}, ${parts.base})`;
  }
  return `*${emitExpression(expression, context)}`;
}

function emitLocalVectorAddressScalarAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string | undefined {
  if (
    expression.left.kind !== "unary" ||
    expression.left.operator !== "*" ||
    expression.left.argument.kind !== "cast" ||
    !expression.left.argument.pointer ||
    expression.left.argument.expression.kind !== "unary" ||
    expression.left.argument.expression.operator !== "&" ||
    expression.left.argument.expression.argument.kind !== "identifier"
  ) {
    return undefined;
  }
  const name = expression.left.argument.expression.argument.name;
  const vectorType = context.localValueTypeFor(name);
  if (!isCudaVectorType(vectorType)) return undefined;
  const localName = context.nameFor(name);
  const scalarType = expression.left.argument.valueType;
  const right = emitExpressionAsValueType(expression.right, scalarType, context);
  const current = `${localName}[0u]`;
  const value = expression.operator === "=" ? right : `(${current} ${expression.operator.slice(0, -1)} ${right})`;
  return `${localName} = ${emitVectorLaneSetExpression(localName, vectorType, 0, value)}`;
}

function emitLocalBitReinterpretDeref(expression: CudaLiteExpression, context: EmitContext): string | undefined {
  if (expression.kind !== "cast" || !expression.pointer) return undefined;
  const inner = expression.expression;
  if (inner.kind !== "unary" || inner.operator !== "&" || inner.argument.kind !== "identifier") return undefined;
  const sourceName = inner.argument.name;
  const sourceType = context.localValueTypeFor(sourceName);
  if (sourceType === undefined) return undefined;
  const source = context.nameFor(sourceName);
  if ((expression.valueType === "uint" || expression.valueType === "int") && sourceType === "bf162") {
    const packed = emitBf162PackExpression(source);
    return expression.valueType === "uint" ? packed : `bitcast<i32>(${packed})`;
  }
  if (expression.valueType === "bf162" && (sourceType === "uint" || sourceType === "int")) {
    const bits = sourceType === "uint" ? source : `bitcast<u32>(${source})`;
    return emitBf162UnpackExpression(bits);
  }
  return undefined;
}

function localPointerArrayLocalAccess(expression: CudaLiteExpression, context: EmitContext): string | undefined {
  if (expression.kind !== "index" || expression.target.kind !== "identifier") return undefined;
  const root = context.localPointerArrayRootFor(expression.target.name, expression.target.span);
  if (!root) return undefined;
  return `${context.nameFor(root.name)}[${context.nameFor(`${expression.target.name}_base`)}[u32(${emitExpression(expression.index, context)})]]`;
}

function emitMember(expression: Extract<CudaLiteExpression, { kind: "member" }>, context: EmitContext): string {
  const matrixMember = emitMatrixTileMemberExpression(expression, context);
  if (matrixMember) return matrixMember;
  if (expression.property === "size") {
    const valueType = expressionValueTypeForEmit(expression.object, context);
    if (isCudaVectorType(valueType)) return String(cudaVectorLaneCount(valueType));
  }
  const storageView = storageViewLValue(expression, context);
  if (storageView?.fieldIndex !== undefined) {
    const shared = storageView.rootName ? sharedDeclarationFor(storageView.rootName, context) : undefined;
    if (shared) return `${emitSharedPointerRead(shared, storageView.index, context.ir, context, storageView.valueType, storageView.subElementLane)}.${storageView.field}`;
    const local = storageView.rootName ? localArrayForStorageView(storageView.rootName, expression.span, context) : undefined;
    if (local) return `${emitLocalPointerRead(local, storageView.index, storageView.valueType, context, storageView.subElementLane)}.${storageView.field}`;
    return `${storageView.name}[${storageView.index} + ${storageView.fieldIndex}u]`;
  }
  const objectName = expressionName(expression.object);
  const axisIndex = expression.property === "x" ? 0 : expression.property === "y" ? 1 : 2;
  switch (objectName) {
    case "threadIdx":
      if (context.ir.workgroupSize[axisIndex] === 1) return "0";
      return `i32(local_id.${expression.property})`;
    case "blockIdx":
      return `i32(workgroup_id.${expression.property})`;
    case "blockDim":
      return String(context.ir.workgroupSize[axisIndex]);
    case "gridDim":
      return `i32(num_workgroups.${expression.property})`;
    default:
      return `${emitExpression(expression.object, context)}.${expression.property}`;
  }
}

function emitMatrixTileMemberExpression(
  expression: Extract<CudaLiteExpression, { kind: "member" }>,
  context: EmitContext,
): string | undefined {
  const ref = matrixTileReference(expression.object);
  if (!ref) return undefined;
  const declaration = context.localArrayFor(ref.root);
  const spec = declaration?.matrixTile ? resolveMatrixTileSpec(declaration.matrixTile) : undefined;
  if (!declaration || !spec) return undefined;
  if (expression.property === "num_elements") return String(matrixTileElementCount(spec));
  if (expression.property === "x") throw featureError("unsupported-wmma-fragment-member", "WMMA fragment lane storage requires indexed access");
  throw featureError("unsupported-wmma-fragment-member", `unsupported WMMA fragment member '${expression.property}'`);
}

function emitMatrixTileLaneAccessExpression(
  expression: Extract<CudaLiteExpression, { kind: "index" }>,
  context: EmitContext,
): string | undefined {
  if (expression.target.kind !== "member" || expression.target.property !== "x") return undefined;
  const ref = matrixTileReference(expression.target.object);
  if (!ref) return undefined;
  const declaration = context.localArrayFor(ref.root);
  const spec = declaration?.matrixTile ? resolveMatrixTileSpec(declaration.matrixTile) : undefined;
  if (!declaration || !spec) return undefined;
  const base = emitMatrixTileBase(ref.indices, declaration.dimensions, matrixTileElementCount(spec), context);
  const lane = `u32(${emitExpression(expression.index, context)})`;
  return base === "0u"
    ? `${context.nameFor(ref.root)}[${lane}]`
    : `${context.nameFor(ref.root)}[(${base} + ${lane})]`;
}

function expressionValueTypeForEmit(expression: CudaLiteExpression, context: EmitContext): CudaLiteScalarType | undefined {
  if (context.expressionValueTypes.has(expression)) return context.expressionValueTypes.get(expression);
  const valueType = uncachedExpressionValueTypeForEmit(expression, context);
  context.expressionValueTypes.set(expression, valueType);
  return valueType;
}

function uncachedExpressionValueTypeForEmit(expression: CudaLiteExpression, context: EmitContext): CudaLiteScalarType | undefined {
  if (expression.kind === "number") {
    if (numberLiteralHasFloatSyntax(expression.raw)) return "float";
    if (numberLiteralHasUnsignedSuffix(expression.raw)) return "uint";
    return "int";
  }
  if (expression.kind === "identifier") {
    return context.localValueTypeFor(expression.name) ??
      context.paramFor(expression.name)?.valueType ??
      context.deviceGlobalFor(expression.name)?.valueType ??
      sharedDeclarationFor(expression.name, context)?.valueType ??
      context.ir.constants.find((item) => item.name === expression.name)?.valueType ??
      context.uniformScalarTypeFor(expression.name);
  }
  if (expression.kind === "cast") return expression.valueType;
  if (expression.kind === "call") {
    const name = expressionName(expression.callee);
    if (name === "subgroupAny" || name === "subgroupAll") return "bool";
    if (name === "subgroupBallot") return "uint";
    if (expression.callee.kind === "member" && (expression.callee.property === "any" || expression.callee.property === "all")) return "bool";
    if (expression.callee.kind === "member" && expression.callee.property === "ballot") return "uint";
    if (name !== undefined && isTextureReadCall(name)) return expression.templateValueType ?? "float";
    if (name === "surf1Dread" || name === "surf2Dread" || name === "surf2DLayeredread" || name === "surf3Dread") return expression.templateValueType ?? "float";
    const atomicReturnType = atomicCallReturnValueType(name, expression.args, context);
    if (atomicReturnType !== undefined) return atomicReturnType;
    const cooperativeReturnType = name === undefined ? undefined : cooperativeReductionReturnType(name, expression.args, context);
    if (cooperativeReturnType !== undefined) return cooperativeReturnType;
    if (name !== undefined) {
      const fn = context.deviceFunctionFor(name, expression.args.length);
      if (fn) return fn.returnType;
    }
    if (name === "abs" || name === "min" || name === "max" || name === "fminf" || name === "fmaxf") {
      const vectorType = expression.args.map((arg) => expressionValueTypeForEmit(arg, context)).find(isCudaVectorType);
      if (vectorType) return vectorType;
    }
    if (name === "lerp") {
      const left = expression.args[0] ? expressionValueTypeForEmit(expression.args[0], context) : undefined;
      const right = expression.args[1] ? expressionValueTypeForEmit(expression.args[1], context) : undefined;
      if (isCudaVectorType(left)) return left;
      if (isCudaVectorType(right)) return right;
    }
    if (name === "fma" || name === "fmaf" || name === "__fmaf_rn") {
      return expression.args.map((arg) => expressionValueTypeForEmit(arg, context)).find(isCudaVectorType) ?? "float";
    }
    if (name === "normalize") {
      return expression.args[0] ? expressionValueTypeForEmit(expression.args[0], context) : undefined;
    }
    if (name === "cross") {
      return expression.args.map((arg) => expressionValueTypeForEmit(arg, context)).find(isCudaVectorType);
    }
    if (name === "dot" || name === "length") return "float";
    if (name !== undefined && CUDA_CACHE_HINT_LOADS.has(name)) {
      return expression.args[0] ? devicePointerValueTypeForExpression(expression.args[0], context) : undefined;
    }
    if (name === "__halves2bfloat162") return "bf162";
    const intrinsic = name ? CUDA_INTRINSICS_BY_NAME.get(name) : undefined;
    if (intrinsic?.returnType === "argument1") return expression.args[0]
      ? expressionValueTypeForEmit(expression.args[0], context)
      : undefined;
    if (intrinsic?.returnType !== undefined) return intrinsic.returnType;
    if (name === "abs" || name === "min" || name === "max") {
      return expression.args.reduce<CudaLiteScalarType | undefined>(
        (type, arg) => promotedCudaScalarType(type, expressionValueTypeForEmit(arg, context)),
        undefined,
      );
    }
    if (name === "sqrt" || name === "sqrtf" || name === "exp" || name === "expf" || name === "log" || name === "logf") return "float";
    return name ? cudaVectorConstructorType(name) : undefined;
  }
  if (expression.kind === "index") {
    const pointerLvalue = devicePointerLValue(expression, context);
    if (pointerLvalue) return pointerLvalue.valueType;
    const storageView = storageViewLValue(expression, context);
    if (storageView) return storageView.valueType;
    if (expression.target.kind === "identifier") {
      const localType = context.localValueTypeFor(expression.target.name);
      if (isCudaVectorType(localType)) return cudaVectorScalarType(localType);
    }
    const root = rootIdentifier(expression);
    const localArray = root ? localArrayForStorageView(root, expression.span, context) : undefined;
    if (localArray) return localArray.valueType;
    const shared = root ? sharedDeclarationFor(root, context) : undefined;
    if (shared) return shared.valueType;
    return expressionValueTypeForEmit(expression.target, context);
  }
  if (expression.kind === "member") {
    if (expression.property === "size") return "int";
    const objectName = expressionName(expression.object);
    if ((objectName === "threadIdx" || objectName === "blockIdx" || objectName === "blockDim" || objectName === "gridDim") &&
      (expression.property === "x" || expression.property === "y" || expression.property === "z")) {
      return "int";
    }
    const objectType = expressionValueTypeForEmit(expression.object, context);
    if (isCudaVectorType(objectType)) return cudaVectorScalarType(objectType);
    if (objectType === "complex64") return "float";
    return objectType;
  }
  if (expression.kind === "binary") {
    const vectorType = vectorArithmeticTypeForEmit(expression, context);
    if (vectorType) return vectorType;
    if (isComparisonOperator(expression.operator) || expression.operator === "&&" || expression.operator === "||") return "bool";
    return promotedCudaScalarType(
      expressionValueTypeForEmit(expression.left, context),
      expressionValueTypeForEmit(expression.right, context),
    );
  }
  if (expression.kind === "unary") {
    if (expression.operator === "!") return "bool";
    if (expression.operator === "*") {
      const storageView = storageViewForPointerExpression(expression.argument, zeroExpression(expression.span), context);
      if (storageView) return storageView.valueType;
      if (expression.argument.kind === "identifier") {
        const alias = flattenedPointerAlias(expression.argument.name, expression.argument.span, context);
        if (alias?.valueType) return alias.valueType;
        const param = context.paramFor(expression.argument.name);
        if (param?.pointer) return param.valueType;
      }
      return devicePointerValueTypeForExpression(expression.argument, context);
    }
    return expressionValueTypeForEmit(expression.argument, context);
  }
  if (expression.kind === "conditional") {
    const consequent = expressionValueTypeForEmit(expression.consequent, context);
    const alternate = expressionValueTypeForEmit(expression.alternate, context);
    if (isCudaVectorType(consequent) && consequent === alternate) return consequent;
    return promotedCudaScalarType(
      consequent,
      alternate,
    );
  }
  return undefined;
}

function atomicCallReturnValueType(
  name: string | undefined,
  args: readonly CudaLiteExpression[],
  context: EmitContext,
): CudaLiteScalarType | undefined {
  if (name === "atomicInc" || name === "atomicInc_system" || name === "atomicDec" || name === "atomicDec_system") {
    return "uint";
  }
  if (
    name === "atomicAdd" ||
    name === "atomicSub" ||
    name === "atomicMin" ||
    name === "atomicMax" ||
    name === "atomicMaxFloat" ||
    name === "atomicExch" ||
    name === "atomicCAS" ||
    name === "atomicAnd" ||
    name === "atomicOr" ||
    name === "atomicXor" ||
    name === "atomicAdd_system" ||
    name === "atomicSub_system" ||
    name === "atomicMin_system" ||
    name === "atomicMax_system" ||
    name === "atomicExch_system" ||
    name === "atomicCAS_system" ||
    name === "atomicAnd_system" ||
    name === "atomicOr_system" ||
    name === "atomicXor_system"
  ) {
    return atomicTargetValueType(args[0], context);
  }
  return undefined;
}

function atomicTargetValueType(
  target: CudaLiteExpression | undefined,
  context: EmitContext,
): CudaLiteScalarType | undefined {
  if (target === undefined) return undefined;
  if (target.kind === "cast" && target.pointer) return target.valueType;
  if (target.kind === "unary" && target.operator === "&") return expressionValueTypeForEmit(target.argument, context);
  return devicePointerValueTypeForExpression(target, context) ?? expressionValueTypeForEmit(target, context);
}

function cooperativeReductionReturnType(
  name: string,
  args: readonly CudaLiteExpression[],
  context: EmitContext,
): CudaLiteScalarType | undefined {
  if (name.endsWith("::reduce")) {
    return expressionValueTypeForEmitWithLocalFallback(args[1], context);
  }
  switch (name) {
    case "bg_subgroup_add":
    case "blockReduce":
      return args[0] ? expressionValueTypeForEmit(args[0], context) : undefined;
    case "warpReduceSum":
    case "warp_reduce_sum":
    case "warp_reduce_sum_f32":
    case "warp_reduce_sum_f16":
    case "warp_reduce_sum_f16_f16":
    case "warp_reduce_sum_f16_f32":
    case "warp_reduce_sum_i8_i32":
    case "warp_reduce_sum_i32_i32":
    case "warpReduceMax":
    case "warp_reduce_max":
    case "warp_reduce_max_f32":
    case "warpReduceMin":
    case "warp_reduce_min": {
      const value = args.length === 2 ? args[1] : args[0];
      return value ? expressionValueTypeForEmit(value, context) : undefined;
    }
    case "__reduce_add_sync":
      return args[1] ? expressionValueTypeForEmit(args[1], context) : undefined;
    default:
      return undefined;
  }
}

function expressionValueTypeForEmitWithLocalFallback(
  expression: CudaLiteExpression | undefined,
  context: EmitContext,
): CudaLiteScalarType | undefined {
  if (!expression) return undefined;
  return expressionValueTypeForEmit(expression, context) ??
    (expression.kind === "identifier" ? context.localValueTypeFor(expression.name) : undefined);
}

function vectorArithmeticTypeForEmit(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): CudaLiteVectorType | undefined {
  if (!isVectorArithmeticOperator(expression.operator)) return undefined;
  const left = expressionValueTypeForEmit(expression.left, context);
  const right = expressionValueTypeForEmit(expression.right, context);
  if (isCudaVectorType(left) && isCudaVectorType(right)) return left === right ? left : undefined;
  if (isCudaVectorType(left)) return left;
  return isCudaVectorType(right) ? right : undefined;
}

function emitVectorArithmetic(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): string | undefined {
  const vectorType = vectorArithmeticTypeForEmit(expression, context);
  if (!vectorType) return undefined;
  const left = emitExpressionAsVectorOperand(expression.left, vectorType, context);
  const right = emitExpressionAsVectorOperand(expression.right, vectorType, context);
  return `(${left} ${expression.operator} ${right})`;
}

function emitScalarBinaryExpression(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): string {
  const threadIndexSimplification = emitThreadIndexRangeBinarySimplification(expression, context);
  if (threadIndexSimplification !== undefined) return threadIndexSimplification;
  if (expression.operator === "&&" || expression.operator === "||") {
    return `(${emitTruthinessExpression(expression.left, context)} ${expression.operator} ${emitTruthinessExpression(expression.right, context)})`;
  }
  if (isComparisonOperator(expression.operator)) {
    const leftScalar = expressionWgslScalarType(expression.left, context);
    const rightScalar = expressionWgslScalarType(expression.right, context);
    if ((leftScalar === "bool" && rightScalar !== "bool") || (rightScalar === "bool" && leftScalar !== "bool")) {
      return `(${emitTruthinessExpression(expression.left, context)} ${expression.operator} ${emitTruthinessExpression(expression.right, context)})`;
    }
  }
  if (isShiftOperator(expression.operator)) {
    const leftType = integerBinaryTargetType(expression, context);
    const left = leftType ? emitExpressionAsWgslScalar(expression.left, leftType, context) : emitExpression(expression.left, context);
    const right = isAbstractIntegerLiteral(expression.right)
      ? emitExpression(expression.right, context)
      : emitExpressionAsWgslScalar(expression.right, "u32", context);
    return `(${left} ${expression.operator} ${right})`;
  }
  if (isBitwiseOperator(expression.operator)) {
    const target = expression.operator === "|" && expression.left.kind === "identifier"
      ? "u32"
      : integerBinaryTargetType(expression, context);
    const left = target ? emitExpressionAsWgslScalar(expression.left, target, context) : emitExpression(expression.left, context);
    const right = target ? emitExpressionAsWgslScalar(expression.right, target, context) : emitExpression(expression.right, context);
    return `(${left} ${expression.operator} ${right})`;
  }
  const target = promotedWgslScalarTypeForBinary(expression, context);
  const left = target ? emitExpressionAsWgslScalar(expression.left, target, context) : emitExpression(expression.left, context);
  const right = target ? emitExpressionAsWgslScalar(expression.right, target, context) : emitExpression(expression.right, context);
  return `(${left} ${expression.operator} ${right})`;
}

function emitThreadIndexRangeBinarySimplification(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): string | undefined {
  if (expression.operator !== "/" && expression.operator !== "%") return undefined;
  if (!isIntegerNumberLiteral(expression.right)) return undefined;
  const divisor = expression.right.value;
  if (!Number.isFinite(divisor) || divisor <= 0) return undefined;
  const axisIndex = threadIndexAxis(expression.left);
  if (axisIndex === undefined) return undefined;
  const axisSize = context.ir.workgroupSize[axisIndex];
  if (divisor < axisSize) return undefined;
  if (expression.operator === "/") return "0";
  return emitExpression(expression.left, context);
}

function threadIndexAxis(expression: CudaLiteExpression): 0 | 1 | 2 | undefined {
  if (expression.kind !== "member" || expressionName(expression.object) !== "threadIdx") return undefined;
  switch (expression.property) {
    case "x":
      return 0;
    case "y":
      return 1;
    case "z":
      return 2;
    default:
      return undefined;
  }
}

function integerBinaryTargetType(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): "i32" | "u32" | undefined {
  const left = expressionWgslScalarType(expression.left, context);
  const right = expressionWgslScalarType(expression.right, context);
  if (left === "u32" || right === "u32") return "u32";
  if (left === "i32" || right === "i32") return "i32";
  return undefined;
}

function promotedWgslScalarTypeForBinary(
  expression: Extract<CudaLiteExpression, { kind: "binary" }>,
  context: EmitContext,
): "f32" | "f16" | "i32" | "u32" | undefined {
  if (!isScalarPromotionOperator(expression.operator)) return undefined;
  const left = expressionWgslScalarType(expression.left, context);
  const right = expressionWgslScalarType(expression.right, context);
  if (left === undefined || right === undefined || left === right) return undefined;
  if (left === "f32" || right === "f32") return "f32";
  if (left === "f16" || right === "f16") return "f16";
  if (left === "u32" || right === "u32") return "u32";
  if (left === "i32" || right === "i32") return "i32";
  return undefined;
}

function emitExpressionAsWgslScalar(
  expression: CudaLiteExpression,
  target: "f32" | "f16" | "i32" | "u32",
  context: EmitContext,
): string {
  if (target === "u32" && isIntegerNumberLiteral(expression)) return emitNumberLiteralAsU32(expression.raw);
  if (target === "i32" && isIntegerNumberLiteral(expression) && expression.value > 2147483647 && expression.value <= 0xffffffff) {
    return `bitcast<i32>(${emitNumberLiteralAsU32(expression.raw)})`;
  }
  const value = emitExpression(expression, context);
  if (isCudaVectorType(expressionValueTypeForEmit(expression, context))) return `${target}(${value}.x)`;
  return expressionWgslScalarType(expression, context) === target ? value : `${target}(${value})`;
}

function expressionWgslScalarType(
  expression: CudaLiteExpression,
  context: EmitContext,
): "f32" | "f16" | "i32" | "u32" | "bool" | undefined {
  const valueType = expressionValueTypeForEmit(expression, context);
  if (valueType === undefined || isCudaVectorType(valueType)) return undefined;
  return cudaScalarWgslType(valueType);
}

function isScalarPromotionOperator(operator: string): boolean {
  return isVectorArithmeticOperator(operator) || operator === "%" || isComparisonOperator(operator);
}

function emitExpressionAsVectorOperand(
  expression: CudaLiteExpression,
  vectorType: CudaLiteVectorType,
  context: EmitContext,
): string {
  const value = emitExpression(expression, context);
  return expressionValueTypeForEmit(expression, context) === vectorType
    ? value
    : emitVectorSplat(vectorType, castExpressionToVectorScalar(value, vectorType));
}

function emitVectorMinMaxCall(
  expression: CudaLiteCallExpression,
  name: string | undefined,
  context: EmitContext,
): string | undefined {
  if (name !== "min" && name !== "max" && name !== "fminf" && name !== "fmaxf") return undefined;
  const vectorType = expression.args
    .map((arg) => expressionValueTypeForEmit(arg, context))
    .find(isCudaVectorType);
  if (!vectorType) return undefined;
  const op = name === "min" || name === "fminf" ? "min" : "max";
  return `${op}(${expression.args.map((arg) => emitExpressionAsVectorOperand(arg, vectorType, context)).join(", ")})`;
}

function emitVectorLerpCall(
  expression: CudaLiteCallExpression,
  name: string | undefined,
  context: EmitContext,
): string | undefined {
  if (name !== "lerp") return undefined;
  const vectorType = expression.args
    .slice(0, 2)
    .map((arg) => expressionValueTypeForEmit(arg, context))
    .find(isCudaVectorType);
  if (!vectorType) return undefined;
  const left = expression.args[0] ? emitExpressionAsVectorOperand(expression.args[0], vectorType, context) : emitVectorSplat(vectorType, "0.0");
  const right = expression.args[1] ? emitExpressionAsVectorOperand(expression.args[1], vectorType, context) : emitVectorSplat(vectorType, "0.0");
  const t = expression.args[2] ? emitExpression(expression.args[2], context) : "0.0";
  return `fma(${emitVectorSplat(vectorType, castExpressionToVectorScalar(t, vectorType))}, (${right} - ${left}), ${left})`;
}

function emitVectorFmaCall(
  expression: CudaLiteCallExpression,
  name: string | undefined,
  context: EmitContext,
): string | undefined {
  if (name !== "fma" && name !== "fmaf" && name !== "__fmaf_rn") return undefined;
  const vectorType = expression.args
    .map((arg) => expressionValueTypeForEmit(arg, context))
    .find(isCudaVectorType);
  if (!vectorType) return undefined;
  return `fma(${expression.args.map((arg) => emitExpressionAsVectorOperand(arg, vectorType, context)).join(", ")})`;
}

function emitFrexpCall(expression: CudaLiteCallExpression, context: EmitContext): string {
  const value = expression.args[0] ? emitExpressionAsValueType(expression.args[0], "float", context) : "0.0";
  const exponent = expression.args[1] ? emitExpression(expression.args[1], context) : "&__bg_missing_frexp_exp";
  return `bg_frexp(${value}, ${exponent})`;
}

interface VectorLaneAddressCast {
  readonly vectorName: string;
  readonly vectorType: CudaLiteScalarType;
  readonly laneIndex: string;
  readonly castType: CudaLiteScalarType;
}

function vectorLaneAddressCast(expression: CudaLiteExpression | undefined, context: EmitContext): VectorLaneAddressCast | undefined {
  if (
    expression?.kind === "index" &&
    expression.target.kind === "cast" &&
    expression.target.pointer &&
    expression.target.expression.kind === "unary" &&
    expression.target.expression.operator === "&" &&
    expression.target.expression.argument.kind === "identifier"
  ) {
    const vectorName = expression.target.expression.argument.name;
    const vectorType = context.localValueTypeFor(vectorName);
    if (!isCudaVectorType(vectorType)) return undefined;
    return {
      vectorName: context.nameFor(vectorName),
      vectorType,
      laneIndex: `u32(${emitExpression(expression.index, context)})`,
      castType: expression.target.valueType,
    };
  }
  if (!expression || expression.kind !== "cast" || expression.pointer) return undefined;
  const address = expression.expression;
  if (address.kind !== "unary" || address.operator !== "&" || address.argument.kind !== "index") return undefined;
  const target = address.argument.target;
  if (target.kind !== "identifier") return undefined;
  const vectorType = context.localValueTypeFor(target.name);
  if (!isCudaVectorType(vectorType)) return undefined;
  return {
    vectorName: context.nameFor(target.name),
    vectorType,
    laneIndex: `u32(${emitExpression(address.argument.index, context)})`,
    castType: expression.valueType,
  };
}

function localVectorAddressViewForIndex(
  expression: Extract<CudaLiteExpression, { kind: "index" }>,
  context: EmitContext,
): string | undefined {
  if (
    expression.target.kind !== "cast" ||
    !expression.target.pointer ||
    expression.target.expression.kind !== "unary" ||
    expression.target.expression.operator !== "&" ||
    expression.target.expression.argument.kind !== "identifier"
  ) {
    return undefined;
  }
  const sourceName = expression.target.expression.argument.name;
  const sourceType = context.localValueTypeFor(sourceName);
  if (sourceType !== expression.target.valueType || !isCudaVectorType(sourceType)) return undefined;
  const source = context.nameFor(sourceName);
  if (expression.index.kind === "number" && expression.index.value === 0) return source;
  return `${source}[u32(${emitExpression(expression.index, context)})]`;
}

function emitVectorLaneAddressRead(view: VectorLaneAddressCast): string {
  const scalarType = cudaVectorScalarType(view.vectorType) ?? "float";
  const lane = `${view.vectorName}[${view.laneIndex}]`;
  if (scalarType === view.castType) return lane;
  const source = cudaScalarWgslType(scalarType);
  const target = cudaScalarWgslType(view.castType);
  if (source && target && source !== "bool" && target !== "bool") return `bitcast<${target}>(${lane})`;
  return `${wgslScalar(view.castType)}(${lane})`;
}

function emitVectorLaneAddressFromFloatWrite(
  view: VectorLaneAddressCast,
  value: string,
): string {
  const scalarType = cudaVectorScalarType(view.vectorType) ?? "float";
  const laneValue = scalarType === "uint" && view.castType === "float"
    ? `((bitcast<u32>(f32(${value})) + 0x8000u) & 0xffff0000u)`
    : emitExpressionAsWgslScalarText(value, scalarType);
  return `${view.vectorName} = ${emitVectorLaneSetExpression(view.vectorName, view.vectorType, view.laneIndex, laneValue)}`;
}

function curandStateAddressSpace(
  expression: CudaLiteExpression | undefined,
  context: EmitContext,
): "function" | "storage" | undefined {
  if (expression?.kind !== "unary" || expression.operator !== "&") return undefined;
  const root = rootIdentifier(expression.argument);
  if (!root) return undefined;
  const param = context.paramFor(root);
  if (param?.pointer) return "storage";
  if (context.deviceGlobalFor(root)) return "storage";
  return "function";
}

function emitCall(expression: CudaLiteCallExpression, context: EmitContext): string {
  const name = expressionName(expression.callee);
  const cooperativeGroupCall = emitCooperativeGroupCall(expression, context, cooperativeCallbacks(context));
  if (cooperativeGroupCall !== undefined) return cooperativeGroupCall;
  if (wmmaBuiltinName(name)) return "0";
  if (name === "to_float") {
    const lane = vectorLaneAddressCast(expression.args[0], context);
    if (lane) return emitVectorLaneAddressRead(lane);
  }
  if (name === "from_float") {
    const lane = vectorLaneAddressCast(expression.args[0], context);
    const value = expression.args[1];
    if (lane && value) return emitVectorLaneAddressFromFloatWrite(lane, emitExpressionAsValueType(value, "float", context));
  }
  const deviceFunction = name ? context.deviceFunctionFor(name, expression.args.length) : undefined;
  if (deviceFunction) {
    const args = emitDeviceFunctionCallArgs(expression, deviceFunction, context);
    return `${context.nameFor(deviceFunctionLinkName(deviceFunction, context.ir))}(${[...args, "local_id", "workgroup_id", "num_workgroups"].join(", ")})`;
  }
  const args = expression.args.map((arg) => emitExpression(arg, context));
  const vectorFma = emitVectorFmaCall(expression, name, context);
  if (vectorFma !== undefined) return vectorFma;
  const vectorMinMax = emitVectorMinMaxCall(expression, name, context);
  if (vectorMinMax !== undefined) return vectorMinMax;
  const vectorLerp = emitVectorLerpCall(expression, name, context);
  if (vectorLerp !== undefined) return vectorLerp;
  if (name === "frexp" || name === "frexpf") return emitFrexpCall(expression, context);
  const intrinsic = name ? CUDA_INTRINSICS_BY_NAME.get(name) : undefined;
  if (intrinsic?.emitWgsl) {
    const intrinsicArgs = intrinsicNeedsFloatArgs(name)
      ? expression.args.map((arg) => emitExpressionAsFloatArgument(arg, context))
      : args;
    return intrinsic.emitWgsl(intrinsicArgs);
  }
  const vectorConstructor = name ? cudaVectorConstructorType(name) : undefined;
  if (vectorConstructor) {
    return expression.args.length === 1
      ? emitVectorConversionConstructor(vectorConstructor, expression.args[0]!, context)
      : emitMixedVectorConstructor(vectorConstructor, expression.args, context);
  }
  if (name === "__halves2bfloat162") return `${wgslScalar("bf162")}(${args.join(", ")})`;
  if (isPointerIdentityCall(name)) return expression.args[0] ? emitExpression(expression.args[0], context) : "0u";
  if (name !== undefined && CUDA_CACHE_HINT_LOADS.has(name)) {
    const target = expression.args[0];
    if (!target) return "0";
    const storageView = storageViewForPointerExpression(target, zeroExpression(target.span), context);
    if (storageView) {
      const shared = sharedDeclarationFor(storageView.rootName, context);
      if (shared) return emitSharedPointerRead(shared, storageView.index, context.ir, context, storageView.valueType, storageView.subElementLane);
      const local = localArrayForStorageView(storageView.rootName, target.span, context);
      if (local) return emitLocalPointerRead(local, storageView.index, storageView.valueType, context, storageView.subElementLane);
      const param = context.paramFor(storageView.rootName);
      if (param) return emitPointerStorageRead(param, storageView.index, context.ir, context, storageView.valueType);
      const global = context.deviceGlobalFor(storageView.rootName);
      if (global) return emitDeviceGlobalPointerRead(global, storageView.index, context.ir, context, storageView.valueType);
      if (isCudaVectorType(storageView.valueType)) {
        return emitVectorStorageReadAt(context.nameFor(storageView.rootName), storageView.valueType, storageView.index);
      }
    }
    const parts = devicePointerArgumentParts(target, context);
    if (!parts) throw featureError("unsupported-device-pointer-param", `${name} expects a storage pointer or derived storage address`);
    const valueType = devicePointerValueTypeForExpression(target, context);
    return `${pointerReadHelperName(valueType)}(${parts.buffer}, ${parts.base})`;
  }
  if (name !== undefined && CUDA_CACHE_HINT_STORES.has(name)) {
    const target = expression.args[0];
    const value = expression.args[1];
    if (!target || !value) return "0";
    const storageView = storageViewForPointerExpression(target, zeroExpression(target.span), context);
    if (storageView) {
      const written = emitExpressionAsValueType(value, storageView.valueType, context);
      const shared = sharedDeclarationFor(storageView.rootName, context);
      if (shared) return emitSharedPointerWrite(shared, storageView.index, written, context.ir, context, storageView.valueType, storageView.subElementLane);
      const local = localArrayForStorageView(storageView.rootName, target.span, context);
      if (local) return emitLocalPointerWrite(local, storageView.index, written, storageView.valueType, context, storageView.subElementLane);
      const param = context.paramFor(storageView.rootName);
      if (param) return emitPointerStorageWrite(param, storageView.index, written, context.ir, context, storageView.valueType);
      const global = context.deviceGlobalFor(storageView.rootName);
      if (global) return emitDeviceGlobalPointerWrite(global, storageView.index, written, context.ir, context, storageView.valueType);
      if (isCudaVectorType(storageView.valueType)) {
        return emitVectorStorageWriteAt(context.nameFor(storageView.rootName), storageView.valueType, storageView.index, written);
      }
    }
    const parts = devicePointerArgumentParts(target, context);
    if (!parts) throw featureError("unsupported-device-pointer-param", `${name} expects a storage pointer or derived storage address`);
    const valueType = devicePointerValueTypeForExpression(target, context);
    return `${pointerWriteHelperName(valueType)}(${parts.buffer}, ${parts.base}, ${emitExpressionAsValueType(value, valueType, context)})`;
  }
  if (name === "__cvta_generic_to_shared") {
    const target = expression.args[0];
    if (!target) return "0u";
    const sharedIndex = emitSharedAddressIndex(target, context.ir, (expression) => emitExpression(expression, context));
    if (sharedIndex) return `u32(${sharedIndex})`;
    const parts = devicePointerArgumentParts(target, context);
    if (!parts) throw featureError("unsupported-device-pointer-param", "__cvta_generic_to_shared expects a storage or shared pointer");
    return `u32(${parts.base})`;
  }
  if (isCpAsyncCopyCall(name) || isCpAsyncFenceCall(name)) return "0";
  switch (name) {
    case "__syncthreads":
    case "__syncwarp":
      return "workgroupBarrier()";
    case "__threadfence":
      return "storageBarrier()";
    case "__trap":
      return "0";
    case "clock":
      return "i32(workgroup_id.x * 104729u + workgroup_id.y * 1009u + workgroup_id.z * 97u + local_id.x + local_id.y * 31u + local_id.z * 7u)";
    case "clock64":
      return "i32(workgroup_id.x * 104729u + workgroup_id.y * 1009u + workgroup_id.z * 97u + local_id.x + local_id.y * 31u + local_id.z * 7u)";
    case "cudaDeviceSynchronize":
    case "cudaStreamCreate":
    case "cudaStreamCreateWithFlags":
    case "cudaStreamDestroy":
    case "cudaStreamSynchronize":
    case "cudaEventCreate":
    case "cudaEventCreateWithFlags":
    case "cudaEventDestroy":
    case "cudaEventRecord":
    case "cudaEventSynchronize":
      return "0";
    case "cudaMemcpy":
      return "0";
    case "cudaMemcpyAsync":
      return "0";
    case "cudaMemcpyPeerAsync":
      return "0";
    case "cudaGraphSetConditional":
      return "0";
    case "min":
    case "max":
      return `${name}(${args.join(", ")})`;
    case "div_ceil":
      return `(((${args[0] ?? "0"} + ${args[1] ?? "1"}) - 1) / ${args[1] ?? "1"})`;
    case "bg_subgroup_add":
      if (context.subgroupMode === "scalar") return args[0] ?? "0";
      return `subgroupAdd(${args.join(", ")})`;
    case "warpReduceSum":
    case "warp_reduce_sum":
    case "warp_reduce_sum_f32":
    case "warp_reduce_sum_f16":
    case "warp_reduce_sum_f16_f16":
    case "warp_reduce_sum_f16_f32":
    case "warp_reduce_sum_i8_i32":
    case "warp_reduce_sum_i32_i32":
      if (context.subgroupMode === "scalar") return args.length === 2 ? args[1] ?? "0" : args[0] ?? "0";
      return emitScalarWarpReduceCall("sum", expression, context, cooperativeCallbacks(context));
    case "blockReduce":
      if (context.subgroupMode === "scalar") return args[0] ?? "0";
      return `subgroupAdd(${args[0] ?? "0"})`;
    case "__reduce_add_sync":
      if (context.subgroupMode === "scalar") return args[1] ?? "0";
      return emitScalarWarpReduceCall("sum", expression, context, cooperativeCallbacks(context));
    case "warpReduceMax":
    case "warp_reduce_max":
    case "warp_reduce_max_f32":
      if (context.subgroupMode === "scalar") return args.length === 2 ? args[1] ?? "0" : args[0] ?? "0";
      return emitScalarWarpReduceCall("max", expression, context, cooperativeCallbacks(context));
    case "warpReduceMin":
    case "warp_reduce_min":
      if (context.subgroupMode === "scalar") return args.length === 2 ? args[1] ?? "0" : args[0] ?? "0";
      return emitScalarWarpReduceCall("min", expression, context, cooperativeCallbacks(context));
    case "__shfl_sync":
      if (context.subgroupMode === "scalar") return args[1] ?? "0";
      return emitScalarWarpShuffleCall("sync", expression, context, cooperativeCallbacks(context));
    case "__shfl_down_sync":
      if (context.subgroupMode === "scalar") return args[1] ?? "0";
      return emitScalarWarpShuffleCall("down", expression, context, cooperativeCallbacks(context));
    case "__shfl_up_sync":
      if (context.subgroupMode === "scalar") return args[1] ?? "0";
      return emitScalarWarpShuffleCall("up", expression, context, cooperativeCallbacks(context));
    case "__shfl_xor_sync":
      if (context.subgroupMode === "scalar") return args[1] ?? "0";
      return emitScalarWarpShuffleCall("xor", expression, context, cooperativeCallbacks(context));
    case "__any_sync": {
      const predicate = expression.args[1] ? emitTruthinessExpression(expression.args[1], context) : "false";
      if (context.subgroupMode === "scalar") return `select(0u, 1u, ${predicate})`;
      return `select(0u, 1u, subgroupAny(${predicate}))`;
    }
    case "__all_sync": {
      const predicate = expression.args[1] ? emitTruthinessExpression(expression.args[1], context) : "false";
      if (context.subgroupMode === "scalar") return `select(0u, 1u, ${predicate})`;
      return `select(0u, 1u, subgroupAll(${predicate}))`;
    }
    case "__ballot_sync": {
      const predicate = expression.args[1] ? emitTruthinessExpression(expression.args[1], context) : "false";
      if (context.subgroupMode === "scalar") return `select(0u, 1u, ${predicate})`;
      return `subgroupBallot(${predicate}).x`;
    }
    case "tex1D":
    case "tex1Dfetch":
    case "tex2D":
    case "tex2DLod":
    case "tex2DLayered":
    case "tex3D":
    case "texCubemap":
      if (expression.args.length >= 2 && expression.args[0]?.kind === "identifier") {
        const textureSurface = textureSurfaceContext(context);
        const textureArgs = textureReadArgsForEmit(expression, textureSurface);
        if (isTextureBindingName(expression.args[0].name, textureSurface)) {
          const suffix = textureReadHelperSuffix(expression.templateValueType);
          return `bg_tex2d_${suffix}_${expression.args[0].name}(${textureArgs.join(", ")})`;
        }
        return emitTextureReadExpression(
          emitTextureArgument(expression.args[0], textureSurface),
          textureArgs,
          expression.templateValueType,
        );
      }
      return `${name}(${args.join(", ")})`;
    case "surf1Dread":
      if (expression.args.length === 2) {
        const targetType = expression.templateValueType ?? "float";
        return emitSurfaceReadExpression(expression.args[0]!, expression.args[1]!, zeroExpression(expression.span), undefined, targetType, textureSurfaceContext(context));
      }
      if (expression.args.length >= 3) {
        const target = expression.args[0]!;
        const lvalue = target.kind === "unary" && target.operator === "&" ? target.argument : target;
        const targetType = expressionValueTypeForEmit(lvalue, context) ?? "float";
        const value = emitSurfaceReadExpression(expression.args[1]!, expression.args[2]!, zeroExpression(expression.span), undefined, targetType, textureSurfaceContext(context));
        return `${emitExpression(lvalue, context, "lvalue")} = ${value}`;
      }
      return `${name}(${args.join(", ")})`;
    case "surf2Dread":
    case "surf2DLayeredread":
    case "surf3Dread":
      if (expression.args.length === 3) {
        const targetType = expression.templateValueType ?? "float";
        return emitSurfaceReadExpression(expression.args[0]!, expression.args[1]!, expression.args[2]!, undefined, targetType, textureSurfaceContext(context));
      }
      if ((name === "surf2DLayeredread" || name === "surf3Dread") && expression.args.length === 4) {
        const targetType = expression.templateValueType ?? "float";
        return emitSurfaceReadExpression(expression.args[0]!, expression.args[1]!, expression.args[2]!, expression.args[3]!, targetType, textureSurfaceContext(context));
      }
      if (expression.args.length >= 4) {
        const target = expression.args[0]!;
        const lvalue = target.kind === "unary" && target.operator === "&" ? target.argument : target;
        const targetType = expressionValueTypeForEmit(lvalue, context) ?? "float";
        const z = name === "surf2DLayeredread" || name === "surf3Dread" ? expression.args[4] : undefined;
        const value = emitSurfaceReadExpression(expression.args[1]!, expression.args[2]!, expression.args[3]!, z, targetType, textureSurfaceContext(context));
        return `${emitExpression(lvalue, context, "lvalue")} = ${value}`;
      }
      return `${name}(${args.join(", ")})`;
    case "surf2Dwrite":
    case "surf1Dwrite":
    case "surf2DLayeredwrite":
    case "surf3Dwrite":
      if (expression.args.length >= 3) {
        const y = name === "surf1Dwrite"
          ? "0"
          : name === "surf2DLayeredwrite"
            ? emitExpression(expression.args[3]!, context)
            : emitExpression(expression.args[3]!, context);
        const z = name === "surf3Dwrite" || name === "surf2DLayeredwrite" ? emitExpression(expression.args[4]!, context) : "0";
        return emitSurfaceWriteExpression(expression.args[1]!, expression.args[0]!, expression.args[2]!, y, z, textureSurfaceContext(context));
      }
      return `${name}(${args.join(", ")})`;
    case "deviceAllocate":
    case "streamOrderedAllocate":
      if (expression.args.length === 4 && expression.args[0]?.kind === "identifier" && expression.args[1]?.kind === "identifier") {
        return `${rawPoolHelperName(expression.args[0].name, expression.args[1].name)}(u32(${emitExpression(expression.args[2]!, context)}), u32(${emitExpression(expression.args[3]!, context)}))`;
      }
      if (expression.args.length >= 2 && expression.args[0]?.kind === "identifier") {
        return `bg_pool_alloc_${expression.args[0].name}(u32(${emitExpression(expression.args[1]!, context)}))`;
      }
      if (expression.args.length >= 2 && expression.args[0]?.kind === "unary" && expression.args[0].operator === "&" && expression.args[0].argument.kind === "identifier") {
        return `bg_pool_alloc_${expression.args[0].argument.name}(u32(${emitExpression(expression.args[1]!, context)}))`;
      }
      return "0u";
    case "sizeof":
      if (expression.args[0]?.kind === "identifier") return String(sizeofCudaType(expression.args[0].name) ?? 4);
      return "4";
    case "alignof":
      if (expression.args[0]?.kind === "identifier") return String(alignofCudaType(expression.args[0].name) ?? 4);
      return "4";
    case "vec_at":
      return `(${args[0] ?? "vec4<f32>()"}[u32(${args[1] ?? "0"})])`;
    case "dot":
      return `dot(${args[0] ?? "vec2<f32>()"}, ${args[1] ?? "vec2<f32>()"})`;
    case "length":
      return `length(${args[0] ?? "vec2<f32>()"})`;
    case "normalize":
      return `normalize(${args[0] ?? "vec2<f32>()"})`;
    case "cross":
      return `cross(${args[0] ?? "vec3<f32>()"}, ${args[1] ?? "vec3<f32>()"})`;
    case "curand_init": {
      const helper = curandStateAddressSpace(expression.args[3], context) === "storage" ? "bg_curand_init_storage" : "bg_curand_init";
      return `${helper}(u32(${args[0] ?? "0"}), u32(${args[1] ?? "0"}), u32(${args[2] ?? "0"}), ${args[3] ?? "&state"})`;
    }
    case "curand_uniform":
    case "curand_uniform_double": {
      const helper = curandStateAddressSpace(expression.args[0], context) === "storage" ? "bg_curand_uniform_storage" : "bg_curand_uniform";
      return `${helper}(${args[0] ?? "&state"})`;
    }
    case "curand_normal":
    case "curand_normal_double": {
      const helper = curandStateAddressSpace(expression.args[0], context) === "storage" ? "bg_curand_normal_storage" : "bg_curand_normal";
      return `${helper}(${args[0] ?? "&state"})`;
    }
    case "atomicAdd":
    case "atomicAdd_system":
      return emitAtomicCallImpl("atomicAdd", expression, context, args, atomicCallbacks);
    case "atomicSub":
    case "atomicSub_system":
      return emitAtomicCallImpl("atomicSub", expression, context, args, atomicCallbacks);
    case "atomicMin":
    case "atomicMin_system":
      return emitAtomicCallImpl("atomicMin", expression, context, args, atomicCallbacks);
    case "atomicMax":
    case "atomicMax_system":
      return emitAtomicCallImpl("atomicMax", expression, context, args, atomicCallbacks);
    case "atomicMaxFloat":
      return emitAtomicCallImpl("atomicMax", expression, context, args, atomicCallbacks);
    case "atomicAnd":
    case "atomicAnd_system":
      return emitAtomicCallImpl("atomicAnd", expression, context, args, atomicCallbacks);
    case "atomicOr":
    case "atomicOr_system":
      return emitAtomicCallImpl("atomicOr", expression, context, args, atomicCallbacks);
    case "atomicXor":
    case "atomicXor_system":
      return emitAtomicCallImpl("atomicXor", expression, context, args, atomicCallbacks);
    case "atomicInc":
    case "atomicInc_system":
      return emitAtomicCallImpl("atomicInc", expression, context, args, atomicCallbacks);
    case "atomicDec":
    case "atomicDec_system":
      return emitAtomicCallImpl("atomicDec", expression, context, args, atomicCallbacks);
    case "atomicExch":
    case "atomicExch_system":
      return emitAtomicCallImpl("atomicExchange", expression, context, args, atomicCallbacks);
    case "atomicCAS":
    case "atomicCAS_system":
      return emitAtomicCasCallImpl(expression, context, args, atomicCallbacks);
    default:
      return `${emitExpression(expression.callee, context)}(${args.join(", ")})`;
  }
}

function emitDeviceFunctionCallArgs(
  expression: CudaLiteCallExpression,
  deviceFunction: CudaLiteDeviceFunction,
  context: EmitContext,
): string[] {
  return deviceFunction.params.flatMap((param, index) => {
    const arg = expression.args[index];
    if (param.cooperativeGroupKind !== undefined) {
      return param.cooperativeGroupKind === "thread"
        ? [emitCooperativeGroupTileSizeArgument(arg, context)]
        : [];
    }
    if (param.valueType === "texture2d") return [emitTextureArgument(arg, textureSurfaceContext(context))];
    if (param.valueType === "surface2d") return [emitSurfaceArgument(arg, textureSurfaceContext(context))];
    if (!arg) return param.pointer ? ["0u", "0u"] : [zeroValue(param.valueType)];
    return param.pointer
      ? usesFunctionLocalPointerParam(deviceFunction, param, context.ir)
        ? [emitFunctionLocalPointerArgument(arg, context)]
        : emitDevicePointerArgument(arg, context)
      : [emitExpressionAsValueType(arg, param.valueType, context)];
  });
}

function emitCooperativeGroupTileSizeArgument(
  arg: CudaLiteExpression | undefined,
  context: EmitContext,
): string {
  const blockSize = `${context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2]}u`;
  if (arg?.kind !== "identifier") return blockSize;
  const group = context.cooperativeGroupFor(arg.name);
  if (!group) return blockSize;
  if (group.groupKind === "tile") return `${group.tileSize ?? 32}u`;
  if (group.groupKind === "thread" && group.dynamicTileSizeName) return group.dynamicTileSizeName;
  return blockSize;
}

function emitExpressionAsValueType(
  expression: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: EmitContext,
): string {
  if (expression.kind === "initializer" && isCudaVectorType(valueType)) {
    return emitVectorConstructor(
      valueType,
      expression.elements.map((element) => emitExpressionAsValueType(element, cudaVectorScalarType(valueType) ?? "float", context)),
    );
  }
  if (valueType === "uint") return emitExpressionAsWgslScalar(expression, "u32", context);
  if (valueType === "int") return emitExpressionAsWgslScalar(expression, "i32", context);
  if ((isCudaVectorType(valueType) || valueType === "complex64") && expression.kind === "unary" && expression.operator === "*") {
    const parts = devicePointerArgumentParts(expression.argument, context);
    if (parts) return `${pointerReadHelperName(valueType)}(${parts.buffer}, ${parts.base})`;
  }
  if (isCudaVectorType(valueType) && expression.kind === "index") {
    const root = rootIdentifier(expression);
    const pointerParam = root && expression.target.kind === "identifier" ? context.devicePointerParamFor(root) : undefined;
    if (
      root !== undefined &&
      pointerParam &&
      context.mutablePointerBaseFor(root) === context.nameFor(`${root}_base`)
    ) {
      return `${pointerReadHelperName(valueType)}(${context.nameFor(`${pointerParam.name}_buffer`)}, (${context.nameFor(`${pointerParam.name}_base`)} + ${emitDevicePointerIndexDelta(expression.index, pointerParam.valueType, context)}))`;
    }
    const storageView = storageViewLValue(expression, context);
    if (storageView && isCudaVectorType(storageView.valueType)) {
      const shared = storageView.rootName ? sharedDeclarationFor(storageView.rootName, context) : undefined;
      if (shared) return emitSharedPointerRead(shared, storageView.index, context.ir, context, storageView.valueType, storageView.subElementLane);
      const local = storageView.rootName ? localArrayForStorageView(storageView.rootName, expression.span, context) : undefined;
      if (local) return emitLocalPointerRead(local, storageView.index, storageView.valueType, context, storageView.subElementLane);
      const param = storageView.rootName ? context.paramFor(storageView.rootName) : undefined;
      if (param) return emitPointerStorageRead(param, storageView.index, context.ir, context, storageView.valueType);
      const global = storageView.rootName ? context.deviceGlobalFor(storageView.rootName) : undefined;
      if (global && context.ir.atomicDeviceGlobals.includes(global.name)) {
        return emitDeviceGlobalPointerRead(global, storageView.index, context.ir, context, storageView.valueType);
      }
      return emitVectorStorageReadAt(storageView.name, storageView.valueType, storageView.index);
    }
    const index = root ? emitPointerIndex(root, expression.index, context) : undefined;
    const shared = root ? sharedDeclarationFor(root, context) : undefined;
    if (shared && isCudaVectorType(shared.valueType)) return emitSharedPointerRead(shared, index!, context.ir, context, valueType);
    const local = root ? localArrayForStorageView(root, expression.span, context) : undefined;
    if (local && isCudaVectorType(local.valueType)) return emitLocalPointerRead(local, index!, valueType, context);
    const param = root ? context.paramFor(root) : undefined;
    if (param) return emitPointerStorageRead(param, index!, context.ir, context, valueType);
  }
  const value = emitExpression(expression, context);
  const sourceType = expressionValueTypeForEmit(expression, context);
  if (valueType === "void") return value;
  if (sourceType === valueType || isComplexFloat2RepresentationPair(sourceType, valueType)) return value;
  if (valueType === "complex64") {
    if (isCudaVectorType(sourceType)) {
      const y = cudaVectorLaneCount(sourceType) > 1 ? `f32(${value}.y)` : "0.0";
      return `vec2<f32>(f32(${value}.x), ${y})`;
    }
    return `vec2<f32>(${emitExpressionAsValueType(expression, "float", context)}, 0.0)`;
  }
  if (sourceType === "complex64") return `${wgslScalar(valueType)}(${value}.x)`;
  if (isCudaVectorType(sourceType)) return `${wgslScalar(valueType)}(${value}.x)`;
  if (
    isCudaVectorType(valueType) &&
    expression.kind === "identifier" &&
    context.paramFor(expression.name) === undefined &&
    context.deviceGlobalFor(expression.name) === undefined &&
    !context.isUniformScalar(expression.name)
  ) {
    return value;
  }
  if (isCudaVectorType(valueType)) return sourceType !== undefined
    ? emitVectorConversionConstructor(valueType, expression, context)
    : emitVectorSplat(valueType, castExpressionToVectorScalar(value, valueType));
  if (valueType === "bool") return emitTruthinessExpression(expression, context);
  return `${wgslScalar(valueType)}(${value})`;
}

const FLOAT_ARG_INTRINSICS = new Set([
  "sqrt", "sqrtf", "exp", "expf", "__expf", "log", "logf", "__logf",
  "fabs", "fabsf", "floor", "floorf", "ceil", "ceilf", "round", "roundf",
  "rintf", "trunc", "truncf", "sin", "sinf", "__sinf", "cos", "cosf",
  "__cosf", "tan", "tanf", "__tanf", "asin", "asinf", "acos", "acosf",
  "atan", "atanf", "tanh", "tanhf", "cosh", "coshf", "isinf", "isnan",
  "isNan", "rsqrt",
  "rsqrtf", "__frcp_rn", "__saturatef", "pow", "powf", "atan2", "atan2f",
  "fmin", "fminf", "fmax", "fmaxf", "fma", "fmaf", "__fmaf_rn", "lerp",
]);

function intrinsicNeedsFloatArgs(name: string | undefined): boolean {
  return name !== undefined && FLOAT_ARG_INTRINSICS.has(name);
}

function emitExpressionAsFloatArgument(expression: CudaLiteExpression, context: EmitContext): string {
  return expressionValueTypeForEmit(expression, context) === "float"
    ? emitExpression(expression, context)
    : emitExpressionAsValueType(expression, "float", context);
}

function emitInlineU32Output(target: CudaLiteExpression, expression: string, context: EmitContext): string {
  const valueType = expressionValueTypeForEmit(target, context);
  if (valueType === "int") return `i32(${expression})`;
  return expression;
}

function flattenedPointerAlias(
  name: string,
  span: CudaLiteExpression["span"],
  context: EmitContext,
): PointerAlias | undefined {
  return flattenedPointerAliasImpl(name, span, context, storageViewCallbacks);
}

function devicePointerMemberRootName(expression: CudaLiteExpression): string | undefined {
  let cursor: CudaLiteExpression = expression;
  while (true) {
    if (cursor.kind === "identifier") return cursor.name;
    if (cursor.kind === "member") {
      cursor = cursor.object;
      continue;
    }
    if (cursor.kind === "index") {
      cursor = cursor.target;
      continue;
    }
    if (cursor.kind === "unary" && cursor.operator === "*") {
      cursor = cursor.argument;
      continue;
    }
    if (cursor.kind === "cast") {
      cursor = cursor.expression;
      continue;
    }
    return undefined;
  }
}

function emitDevicePointerMemberAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string | undefined {
  if (expression.left.kind !== "member") return undefined;
  const rootName = devicePointerMemberRootName(expression.left);
  if (rootName === undefined || context.devicePointerParamFor(rootName) === undefined) return undefined;
  const pointerLvalue = devicePointerLValue(expression.left, context);
  if (!pointerLvalue || pointerLvalue.fieldIndex === undefined) return undefined;
  if (!isCudaVectorType(pointerLvalue.valueType)) {
    throw featureError("unsupported-vector-assignment", "member assignment through pointer expects a CUDA vector pointer");
  }
  const read = `${pointerReadHelperName(pointerLvalue.valueType)}(${pointerLvalue.buffer}, ${pointerLvalue.index})`;
  const field = vectorFieldName(pointerLvalue.fieldIndex);
  const scalar = cudaVectorScalarType(pointerLvalue.valueType) ?? "float";
  const right = emitExpressionAsValueType(expression.right, scalar, context);
  const currentLane = `${read}.${field}`;
  const laneValue = expression.operator === "="
    ? right
    : `(${currentLane} ${expression.operator.slice(0, -1)} ${right})`;
  const value = emitVectorLaneSetExpression(read, pointerLvalue.valueType, pointerLvalue.fieldIndex, laneValue);
  return `${pointerWriteHelperName(pointerLvalue.valueType)}(${pointerLvalue.buffer}, ${pointerLvalue.index}, ${value})`;
}

function emitAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string {
  const directPointerIndexAssignment = emitDirectPointerIndexAssignment(expression, context);
  if (directPointerIndexAssignment) return directPointerIndexAssignment;
  const pointerArrayAssignment = emitLocalPointerArrayAssignment(expression, context);
  if (pointerArrayAssignment) return pointerArrayAssignment;
  const pointerRebase = emitPointerRebaseAssignment(expression, context);
  if (pointerRebase) return pointerRebase;
  const localVectorAddressScalarAssignment = emitLocalVectorAddressScalarAssignment(expression, context);
  if (localVectorAddressScalarAssignment) return localVectorAddressScalarAssignment;
  const directStorageScalarAssignment = emitDirectStorageScalarAssignment(expression, context);
  if (directStorageScalarAssignment) return directStorageScalarAssignment;
  const localVectorAssignment = emitLocalVectorIndexAssignment(expression, context);
  if (localVectorAssignment) return localVectorAssignment;
  const pointerMemberAssignment = emitDevicePointerMemberAssignment(expression, context);
  if (pointerMemberAssignment) return pointerMemberAssignment;
  const storageView = storageViewLValue(expression.left, context);
  if (storageView && isCudaVectorType(storageView.valueType)) {
    const shared = storageView.rootName ? sharedDeclarationFor(storageView.rootName, context) : undefined;
    const local = storageView.rootName ? localArrayForStorageView(storageView.rootName, expression.left.span, context) : undefined;
    const right = emitExpressionAsValueType(expression.right, storageView.valueType, context);
    if (shared || local) {
      const read = (): string => shared
        ? emitSharedPointerRead(shared, storageView.index, context.ir, context, storageView.valueType, storageView.subElementLane)
        : emitLocalPointerRead(local!, storageView.index, storageView.valueType, context, storageView.subElementLane);
      const write = (value: string): string => shared
        ? emitSharedPointerWrite(shared, storageView.index, value, context.ir, context, storageView.valueType, storageView.subElementLane)
        : emitLocalPointerWrite(local!, storageView.index, value, storageView.valueType, context, storageView.subElementLane);
      if (storageView.field) {
        const scalar = cudaVectorScalarType(storageView.valueType) ?? "float";
        const currentVector = read();
        const current = `${currentVector}.${storageView.field}`;
        const laneValue = expression.operator === "="
          ? emitExpressionAsValueType(expression.right, scalar, context)
          : `(${current} ${expression.operator.slice(0, -1)} ${emitExpressionAsValueType(expression.right, scalar, context)})`;
        return write(emitVectorLaneSetExpression(currentVector, storageView.valueType, storageView.fieldIndex ?? 0, laneValue));
      }
      if (expression.operator !== "=") {
        const current = read();
        const op = expression.operator.slice(0, -1);
        const vectorRight = emitExpressionAsVectorOperand(expression.right, storageView.valueType as CudaLiteVectorType, context);
        return write(`(${current} ${op} ${vectorRight})`);
      }
      return write(right);
    }
    const param = storageView.rootName ? context.paramFor(storageView.rootName) : undefined;
    if (param && context.ir.atomicParams.includes(param.name)) {
      const write = (value: string): string =>
        emitPointerStorageWrite(param, storageView.index, value, context.ir, context, storageView.valueType);
      if (storageView.field) {
        const scalar = cudaVectorScalarType(storageView.valueType) ?? "float";
        const currentVector = emitPointerStorageRead(param, storageView.index, context.ir, context, storageView.valueType);
        const current = `${currentVector}.${storageView.field}`;
        const laneValue = expression.operator === "="
          ? emitExpressionAsValueType(expression.right, scalar, context)
          : `(${current} ${expression.operator.slice(0, -1)} ${emitExpressionAsValueType(expression.right, scalar, context)})`;
        return write(emitVectorLaneSetExpression(currentVector, storageView.valueType, storageView.fieldIndex ?? 0, laneValue));
      }
      if (expression.operator !== "=") {
        const current = emitPointerStorageRead(param, storageView.index, context.ir, context, storageView.valueType);
        const op = expression.operator.slice(0, -1);
        const vectorRight = emitExpressionAsVectorOperand(expression.right, storageView.valueType as CudaLiteVectorType, context);
        return write(`(${current} ${op} ${vectorRight})`);
      }
      return write(right);
    }
    const global = storageView.rootName ? context.deviceGlobalFor(storageView.rootName) : undefined;
    if (global && context.ir.atomicDeviceGlobals.includes(global.name)) {
      const write = (value: string): string =>
        emitDeviceGlobalPointerWrite(global, storageView.index, value, context.ir, context, storageView.valueType);
      if (storageView.field) {
        const scalar = cudaVectorScalarType(storageView.valueType) ?? "float";
        const currentVector = emitDeviceGlobalPointerRead(global, storageView.index, context.ir, context, storageView.valueType);
        const current = `${currentVector}.${storageView.field}`;
        const laneValue = expression.operator === "="
          ? emitExpressionAsValueType(expression.right, scalar, context)
          : `(${current} ${expression.operator.slice(0, -1)} ${emitExpressionAsValueType(expression.right, scalar, context)})`;
        return write(emitVectorLaneSetExpression(currentVector, storageView.valueType, storageView.fieldIndex ?? 0, laneValue));
      }
      if (expression.operator !== "=") {
        const current = emitDeviceGlobalPointerRead(global, storageView.index, context.ir, context, storageView.valueType);
        const op = expression.operator.slice(0, -1);
        const vectorRight = emitExpressionAsVectorOperand(expression.right, storageView.valueType as CudaLiteVectorType, context);
        return write(`(${current} ${op} ${vectorRight})`);
      }
      return write(right);
    }
    if (storageView.field) {
      const current = `${storageView.name}[${storageView.index} + ${storageView.fieldIndex ?? 0}u]`;
      const scalar = cudaVectorScalarType(storageView.valueType) ?? "float";
      const laneRight = emitExpressionAsValueType(expression.right, scalar, context);
      if (expression.operator !== "=") {
        const op = expression.operator.slice(0, -1);
        return `${current} = (${current} ${op} ${laneRight})`;
      }
      return emitVectorStorageFieldWriteAt(storageView.name, storageView.index, storageView.fieldIndex ?? 0, laneRight);
    }
    if (expression.operator !== "=") {
      const current = emitVectorStorageReadAt(storageView.name, storageView.valueType, storageView.index);
      const op = expression.operator.slice(0, -1);
      const vectorRight = emitExpressionAsVectorOperand(expression.right, storageView.valueType as CudaLiteVectorType, context);
      return emitVectorStorageWriteAt(storageView.name, storageView.valueType, storageView.index, `(${current} ${op} ${vectorRight})`);
    }
    return emitVectorStorageWriteAt(storageView.name, storageView.valueType, storageView.index, right);
  }
  const poolAccess = poolAccessForIndex(expression.left, context);
  if (poolAccess) return emitPoolAssignment(poolAccess, expression.operator, expression.right, context, (item) => emitExpression(item, context));
  const localVectorWholeAssignment = emitLocalVectorAssignment(expression, context);
  if (localVectorWholeAssignment) return localVectorWholeAssignment;
  const vectorAssignment = emitVectorAssignment(expression, context);
  if (vectorAssignment) return vectorAssignment;
  const scalarStorageView = scalarStorageViewLValue(expression.left, context);
  if (scalarStorageView) {
    const current = scalarStorageView.addressSpace === "shared"
      ? emitSharedPointerRead(scalarStorageView.root, scalarStorageView.index, context.ir, context, scalarStorageView.valueType, scalarStorageView.subElementLane)
      : emitLocalPointerRead(scalarStorageView.root, scalarStorageView.index, scalarStorageView.valueType, context, scalarStorageView.subElementLane);
    const right = emitExpressionAsValueType(expression.right, scalarStorageView.valueType, context);
    const value = expression.operator === "="
      ? right
      : `(${current} ${expression.operator.slice(0, -1)} ${right})`;
    return scalarStorageView.addressSpace === "shared"
      ? emitSharedPointerWrite(scalarStorageView.root, scalarStorageView.index, value, context.ir, context, scalarStorageView.valueType, scalarStorageView.subElementLane)
      : emitLocalPointerWrite(scalarStorageView.root, scalarStorageView.index, value, scalarStorageView.valueType, context, scalarStorageView.subElementLane);
  }
  const scalarParamStorageView = scalarParamStorageViewLValue(expression.left, context);
  if (scalarParamStorageView) {
    const current = scalarParamStorageView.addressSpace === "global"
      ? emitDeviceGlobalPointerRead(scalarParamStorageView.root, scalarParamStorageView.index, context.ir, context, scalarParamStorageView.valueType)
      : emitPointerStorageRead(scalarParamStorageView.root, scalarParamStorageView.index, context.ir, context, scalarParamStorageView.valueType);
    const right = emitExpressionAsValueType(expression.right, scalarParamStorageView.valueType, context);
    const value = expression.operator === "="
      ? right
      : `(${current} ${expression.operator.slice(0, -1)} ${right})`;
    return scalarParamStorageView.addressSpace === "global"
      ? emitDeviceGlobalPointerWrite(scalarParamStorageView.root, scalarParamStorageView.index, value, context.ir, context, scalarParamStorageView.valueType)
      : emitPointerStorageWrite(scalarParamStorageView.root, scalarParamStorageView.index, value, context.ir, context, scalarParamStorageView.valueType);
  }
  const pointerLvalue = devicePointerLValue(expression.left, context);
  if (pointerLvalue) {
    const read = `${pointerReadHelperName(pointerLvalue.valueType)}(${pointerLvalue.buffer}, ${pointerLvalue.index})`;
    if (pointerLvalue.fieldIndex !== undefined) {
      if (!isCudaVectorType(pointerLvalue.valueType)) {
        throw featureError("unsupported-vector-assignment", "member assignment through pointer expects a CUDA vector pointer");
      }
      const field = vectorFieldName(pointerLvalue.fieldIndex);
      const scalar = cudaVectorScalarType(pointerLvalue.valueType) ?? "float";
      const right = emitExpressionAsValueType(expression.right, scalar, context);
      const currentLane = `${read}.${field}`;
      const laneValue = expression.operator === "="
        ? right
        : `(${currentLane} ${expression.operator.slice(0, -1)} ${right})`;
      return `${pointerWriteHelperName(pointerLvalue.valueType)}(${pointerLvalue.buffer}, ${pointerLvalue.index}, ${emitVectorLaneSetExpression(read, pointerLvalue.valueType, pointerLvalue.fieldIndex, laneValue)})`;
    }
    const right = emitPointerAssignmentValue(expression.right, pointerLvalue.valueType, context);
    const value = expression.operator === "="
      ? right
      : expression.operator === "+="
        ? `(${read} + ${right})`
        : expression.operator === "-="
          ? `(${read} - ${right})`
          : expression.operator === "*="
            ? `(${read} * ${right})`
            : expression.operator === "/="
              ? `(${read} / ${right})`
              : expression.operator === "<<="
                ? `(${read} << ${right})`
                : expression.operator === ">>="
                  ? `(${read} >> ${right})`
                  : expression.operator === "&="
                    ? `(${read} & ${right})`
                    : expression.operator === "|="
                      ? `(${read} | ${right})`
                      : `(${read} ^ ${right})`;
    return `${pointerWriteHelperName(pointerLvalue.valueType)}(${pointerLvalue.buffer}, ${pointerLvalue.index}, ${value})`;
  }
  const root = rootIdentifier(expression.left);
  const param = root ? context.paramFor(root) : undefined;
  const global = root ? context.deviceGlobalFor(root) : undefined;
  const sharedScalar = root ? sharedDeclarationFor(root, context) : undefined;
  if (param?.pointer && param.valueType === "complex64" && expression.left.kind === "index" && expression.left.target.kind === "identifier") {
    const index = emitPointerIndex(param.name, expression.left.index, context);
    const current = emitPointerStorageRead(param, index, context.ir, context);
    const right = emitExpressionAsValueType(expression.right, "complex64", context);
    const value = expression.operator === "="
      ? right
      : expression.operator === "+="
        ? `(${current} + ${right})`
        : expression.operator === "-="
          ? `(${current} - ${right})`
          : expression.operator === "*="
            ? `(${current} * ${right})`
            : expression.operator === "/="
              ? `(${current} / ${right})`
              : undefined;
    if (value) return emitPointerStorageWrite(param, index, value, context.ir, context);
  }
  if (root && context.isAtomicShared(root)) {
    const value = emitExpression(expression.right, context);
    const target = emitExpression(expression.left, context, "lvalue");
    const shared = sharedDeclarationFor(root, context);
    if (shared?.valueType === "float" || shared?.valueType === "double") {
      if (expression.operator === "=") return `atomicStore(&${target}, bitcast<u32>(${value}))`;
      if (expression.operator === "+=") return `bg_atomicAdd_f32_workgroup(&${target}, ${value})`;
      if (expression.operator === "-=") return `bg_atomicSub_f32_workgroup(&${target}, ${value})`;
    } else {
      if (expression.operator === "=") return `atomicStore(&${target}, ${value})`;
      if (expression.operator === "+=") return `atomicAdd(&${target}, ${value})`;
      if (expression.operator === "-=") return `atomicSub(&${target}, ${value})`;
    }
  }
  if (param && context.ir.atomicParams.includes(param.name)) {
    const value = emitExpression(expression.right, context);
    const target = emitExpression(expression.left, context, "lvalue");
    if (param.valueType === "float" || param.valueType === "double") {
      if (expression.operator === "=") return `atomicStore(&${target}, bitcast<u32>(${value}))`;
      if (expression.operator === "+=") return `bg_atomicAdd_f32(&${target}, ${value})`;
      if (expression.operator === "-=") return `bg_atomicSub_f32(&${target}, ${value})`;
    } else {
      if (expression.operator === "=") return `atomicStore(&${target}, ${value})`;
      if (expression.operator === "+=") return `atomicAdd(&${target}, ${value})`;
    }
  }
  if (global && context.ir.atomicDeviceGlobals.includes(global.name)) {
    const value = emitExpression(expression.right, context);
    const target = emitExpression(expression.left, context, "lvalue");
    if (global.valueType === "float" || global.valueType === "double") {
      if (expression.operator === "=") return `atomicStore(&${target}, bitcast<u32>(${value}))`;
      if (expression.operator === "+=") return `bg_atomicAdd_f32(&${target}, ${value})`;
      if (expression.operator === "-=") return `bg_atomicSub_f32(&${target}, ${value})`;
    } else {
      if (expression.operator === "=") return `atomicStore(&${target}, ${value})`;
      if (expression.operator === "+=") return `atomicAdd(&${target}, ${value})`;
      if (expression.operator === "-=") return `atomicSub(&${target}, ${value})`;
    }
  }
  const left = emitExpression(expression.left, context, "lvalue");
  const directStorageElementType = directStorageElementValueTypeForEmit(expression.left, context);
  const leftType = directStorageElementType ?? sharedScalar?.valueType ?? expressionValueTypeForEmit(expression.left, context);
  const right = expression.operator === "=" && directStorageElementType
    ? emitExpressionAsStorageElementValueType(expression.right, directStorageElementType, context)
    : expression.operator === "=" && leftType === "bool"
    ? emitExpressionAsValueType(expression.right, "bool", context)
    : expression.operator === "=" && isCudaVectorType(leftType)
    ? emitExpressionAsValueType(expression.right, leftType, context)
    : expression.operator === "=" && (leftType === "uint" || leftType === "int") && expression.right.kind === "binary" &&
      (isBitwiseOperator(expression.right.operator) || isShiftOperator(expression.right.operator))
    ? emitExpressionAsWgslScalar(expression.right, leftType === "uint" ? "u32" : "i32", context)
    : expression.operator === "=" && shouldCastDirectAssignment(expression.right, leftType, context)
    ? emitExpressionAsValueType(expression.right, leftType, context)
    : emitExpression(expression.right, context);
  const scalarParamLocal = root !== undefined && param?.pointer === false && context.isLocalName(root);
  if (((param?.valueType === "bool" && !scalarParamLocal) || global?.valueType === "bool") && (expression.left.kind === "index" || expression.left.kind === "identifier")) {
    if (expression.operator === "=") return `${left} = select(0u, 1u, ${right})`;
  }
  if (expression.operator === "<<=") return `${left} = (${left} << ${emitExpressionAsWgslScalar(expression.right, "u32", context)})`;
  if (expression.operator === ">>=") return `${left} = (${left} >> ${emitExpressionAsWgslScalar(expression.right, "u32", context)})`;
  if (expression.operator === "&=" || expression.operator === "|=" || expression.operator === "^=") {
    const localLeftType = expression.left.kind === "identifier" ? context.localValueTypeFor(expression.left.name) : undefined;
    const rightScalar = expressionWgslScalarType(expression.right, context);
    const target = (localLeftType ?? leftType) === "uint" || rightScalar === "u32" ||
      (expression.operator === "|=" && expression.left.kind === "identifier")
      ? "u32"
      : "i32";
    return `${left} = (${left} ${expression.operator.slice(0, -1)} ${emitExpressionAsWgslScalar(expression.right, target, context)})`;
  }
  const scalarCompound = emitScalarCompoundAssignment(expression, left, context);
  if (scalarCompound) return scalarCompound;
  return `${left} ${expression.operator} ${right}`;
}

function emitDirectStorageScalarAssignment(
  expression: CudaLiteAssignmentExpression,
  context: EmitContext,
): string | undefined {
  if (expressionContainsAssignment(expression.right)) return undefined;
  const root = directLValueRootName(expression.left);
  if (!root) return undefined;
  if (context.devicePointerParamFor(root)) return undefined;
  const param = context.paramFor(root);
  const global = context.deviceGlobalFor(root);
  if (
    expression.left.kind === "index" &&
    expression.left.target.kind === "identifier" &&
    ((param?.pointer && !isCudaVectorType(param.valueType)) || (global !== undefined && !isCudaVectorType(global.valueType)))
  ) {
    const valueType = param?.valueType ?? global?.valueType;
    if (!valueType) return undefined;
    const target = emitExpression(expression.left, context, "lvalue");
    const current = emitExpressionAsStorageElementValueType(expression.left, valueType, context);
    const right = emitExpressionAsStorageElementValueType(expression.right, valueType, context);
    if ((param && context.ir.atomicParams.includes(param.name)) || (global && context.ir.atomicDeviceGlobals.includes(global.name))) {
      const atomicRight = valueType === "float" || valueType === "double"
        ? emitStorageCarrierAsU32(right, valueType)
        : right;
      if (expression.operator === "=") return `atomicStore(&${target}, ${atomicRight})`;
      const op = expression.operator.slice(0, -1);
      const value = op === "<<" || op === ">>"
        ? `(${current} ${op} ${emitExpressionAsWgslScalar(expression.right, "u32", context)})`
        : `(${current} ${op} ${right})`;
      const atomicValue = valueType === "float" || valueType === "double"
        ? emitStorageCarrierAsU32(value, valueType)
        : value;
      return `atomicStore(&${target}, ${atomicValue})`;
    }
    if (expression.operator === "=") return `${target} = ${right}`;
    const op = expression.operator.slice(0, -1);
    const value = op === "<<" || op === ">>"
      ? `(${current} ${op} ${emitExpressionAsWgslScalar(expression.right, "u32", context)})`
      : `(${current} ${op} ${right})`;
    return `${target} = ${value}`;
  }
  const sharedDeclaration = sharedDeclarationFor(root, context);
  const declaration = sharedDeclaration ?? localArrayForStorageView(root, expression.left.span, context);
  const valueType = declaration?.valueType;
  if (!valueType || isCudaVectorType(valueType)) return undefined;
  const target = emitExpression(expression.left, context, "lvalue");
  const current = emitExpressionAsStorageElementValueType(expression.left, valueType, context);
  const right = emitExpressionAsStorageElementValueType(expression.right, valueType, context);
  if (sharedDeclaration && context.isAtomicShared(root)) {
    const atomicValue = valueType === "float" || valueType === "double"
      ? emitStorageCarrierAsU32(right, valueType)
      : right;
    if (expression.operator === "=") return `atomicStore(&${target}, ${atomicValue})`;
    const op = expression.operator.slice(0, -1);
    const value = op === "<<" || op === ">>"
      ? `(${current} ${op} ${emitExpressionAsWgslScalar(expression.right, "u32", context)})`
      : `(${current} ${op} ${right})`;
    const atomicCompoundValue = valueType === "float" || valueType === "double"
      ? emitStorageCarrierAsU32(value, valueType)
      : value;
    return `atomicStore(&${target}, ${atomicCompoundValue})`;
  }
  if (expression.operator === "=") return `${target} = ${right}`;
  const op = expression.operator.slice(0, -1);
  const value = op === "<<" || op === ">>"
    ? `(${current} ${op} ${emitExpressionAsWgslScalar(expression.right, "u32", context)})`
    : `(${current} ${op} ${right})`;
  return `${target} = ${value}`;
}

function expressionContainsAssignment(expression: CudaLiteExpression): boolean {
  if (expression.kind === "assignment") return true;
  return expressionChildren(expression).some(expressionContainsAssignment);
}

function shouldCastDirectAssignment(
  right: CudaLiteExpression,
  leftType: CudaLiteScalarType | undefined,
  context: EmitContext,
): leftType is CudaLiteScalarType {
  if (leftType === undefined || leftType === "void" || leftType === "bool" || leftType === "complex64" || isCudaVectorType(leftType)) return false;
  const rightType = expressionValueTypeForEmit(right, context);
  if (rightType === undefined || rightType === leftType) return false;
  if ((leftType === "uint" || leftType === "int") && right.kind === "number" && !numberLiteralHasFloatSyntax(right.raw)) return false;
  return true;
}

function directStorageElementValueTypeForEmit(
  expression: CudaLiteExpression,
  context: EmitContext,
): CudaLiteScalarType | undefined {
  if (expression.kind !== "index" && expression.kind !== "member") return undefined;
  const root = directLValueRootName(expression);
  if (!root) return undefined;
  const declaration = sharedDeclarationFor(root, context) ?? localArrayForStorageView(root, expression.span, context);
  if (!declaration) return undefined;
  const rootType = declaration.valueType;
  if (expression.kind === "member" && isCudaVectorType(rootType)) return cudaVectorScalarType(rootType);
  return rootType;
}

function directLValueRootName(expression: CudaLiteExpression): string | undefined {
  let cursor: CudaLiteExpression = expression;
  while (true) {
    if (cursor.kind === "identifier") return cursor.name;
    if (cursor.kind === "index") {
      cursor = cursor.target;
      continue;
    }
    if (cursor.kind === "member") {
      cursor = cursor.object;
      continue;
    }
    if (cursor.kind === "cast") {
      cursor = cursor.expression;
      continue;
    }
    return undefined;
  }
}

function emitExpressionAsStorageElementValueType(
  expression: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: EmitContext,
): string {
  if (isCudaVectorType(valueType) || valueType === "void" || valueType === "complex64") {
    return emitExpressionAsValueType(expression, valueType, context);
  }
  if (valueType === "bool") return emitExpressionAsValueType(expression, "bool", context);
  if (valueType === "uint") {
    if (isIntegerNumberLiteral(expression)) return emitNumberLiteralAsU32(expression.raw);
    return `u32(${emitExpression(expression, context)})`;
  }
  if (valueType === "int") {
    if (isIntegerNumberLiteral(expression) && expression.value > 2147483647 && expression.value <= 0xffffffff) {
      return `bitcast<i32>(${emitNumberLiteralAsU32(expression.raw)})`;
    }
    return `i32(${emitExpression(expression, context)})`;
  }
  return `${wgslScalar(valueType)}(${emitExpression(expression, context)})`;
}

function emitScalarCompoundAssignment(
  expression: CudaLiteAssignmentExpression,
  left: string,
  context: EmitContext,
): string | undefined {
  const operator = expression.operator === "+="
    ? "+"
    : expression.operator === "-="
      ? "-"
      : expression.operator === "*="
        ? "*"
        : expression.operator === "/="
          ? "/"
          : undefined;
  if (!operator) return undefined;
  const leftType = expressionWgslScalarType(expression.left, context);
  if (!leftType || leftType === "bool") return undefined;
  const rightType = expressionWgslScalarType(expression.right, context);
  const binary: Extract<CudaLiteExpression, { kind: "binary" }> = {
    kind: "binary",
    operator,
    left: expression.left,
    right: expression.right,
    span: expression.span,
  };
  const operationType = promotedWgslScalarTypeForBinary(binary, context) ?? leftType;
  if (operationType === leftType && (rightType === undefined || rightType === leftType)) {
    return undefined;
  }
  const value = `(${emitExpressionAsWgslScalar(expression.left, operationType, context)} ${operator} ${emitExpressionAsWgslScalar(expression.right, operationType, context)})`;
  return `${left} = ${operationType === leftType ? value : `${leftType}(${value})`}`;
}

function emitLocalPointerArrayAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string | undefined {
  if (expression.left.kind !== "index" || expression.left.target.kind !== "identifier") return undefined;
  const pointerArray = context.localPointerArrayFor(expression.left.target.name, expression.left.target.span);
  if (!pointerArray) return undefined;
  if (expression.operator !== "=") {
    throw featureError("unsupported-pointer-assignment", "CUDA pointer-array elements support direct assignment only");
  }
  const localRoot = context.localPointerArrayRootFor(pointerArray.name, expression.left.target.span);
  if (localRoot) {
    const base = localPointerArrayLocalBase(expression.right, localRoot, context);
    if (!base) {
      throw featureError("unsupported-pointer-assignment", "CUDA local pointer-array assignment expects an address inside one local array");
    }
    const index = `u32(${emitExpression(expression.left.index, context)})`;
    return `${context.nameFor(`${pointerArray.name}_base`)}[${index}] = ${base}`;
  }
  const parts = devicePointerArgumentParts(expression.right, context);
  if (!parts) {
    throw featureError("unsupported-pointer-assignment", "CUDA pointer-array assignment expects a modeled storage or shared pointer");
  }
  const index = `u32(${emitExpression(expression.left.index, context)})`;
  return `${context.nameFor(`${pointerArray.name}_buffer`)}[${index}] = ${parts.buffer}; ${context.nameFor(`${pointerArray.name}_base`)}[${index}] = ${parts.base}`;
}

function localPointerArrayLocalBase(
  expression: CudaLiteExpression,
  root: CudaLiteVarDecl,
  context: EmitContext,
): string | undefined {
  if (expression.kind === "cast" && expression.pointer) return localPointerArrayLocalBase(expression.expression, root, context);
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    return expression.args[0] ? localPointerArrayLocalBase(expression.args[0], root, context) : undefined;
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const base = localPointerArrayLocalBase(expression.left, root, context);
    if (!base) return undefined;
    const delta = `u32(${emitExpression(expression.right, context)})`;
    return expression.operator === "+" ? `(${base} + ${delta})` : `(${base} - ${delta})`;
  }
  if (expression.kind !== "unary" || expression.operator !== "&") return undefined;
  return localArrayFlatIndexExpression(expression.argument, root, context);
}

function localArrayFlatIndexExpression(
  expression: CudaLiteExpression,
  root: CudaLiteVarDecl,
  context: EmitContext,
): string | undefined {
  const indices: CudaLiteExpression[] = [];
  let cursor = expression;
  while (cursor.kind === "index") {
    indices.unshift(cursor.index);
    cursor = cursor.target;
  }
  if (cursor.kind !== "identifier" || cursor.name !== root.name) return undefined;
  if (indices.length === 0) return "0u";
  const dimensions = root.dimensions;
  if (indices.length !== dimensions.length) return undefined;
  const terms = indices.map((index, axis) => {
    const stride = dimensions.slice(axis + 1).reduce((product, dimension) => product * dimension, 1);
    const value = `u32(${emitExpression(index, context)})`;
    return stride === 1 ? value : `(${value} * ${stride}u)`;
  });
  return terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
}

function emitLocalVectorIndexAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string | undefined {
  if (expression.left.kind !== "index" || expression.left.target.kind !== "identifier") return undefined;
  const name = expression.left.target.name;
  const type = context.localValueTypeFor(name);
  if (!isCudaVectorType(type)) return undefined;
  const index = `u32(${emitExpression(expression.left.index, context)})`;
  const scalar = cudaVectorScalarType(type) ?? "float";
  const right = emitExpressionAsValueType(expression.right, scalar, context);
  const localName = context.nameFor(name);
  const current = `${localName}[${index}]`;
  const value = expression.operator === "=" ? right : `(${current} ${expression.operator.slice(0, -1)} ${right})`;
  return `${localName} = ${emitVectorLaneSet(localName, type, index, value)}`;
}

function emitLocalVectorAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string | undefined {
  if (expression.left.kind !== "identifier") return undefined;
  const name = expression.left.name;
  const type = context.localValueTypeFor(name);
  if (!isCudaVectorType(type)) return undefined;
  const left = context.nameFor(name);
  if (expression.operator === "=") {
    return `${left} = ${emitExpressionAsValueType(expression.right, type, context)}`;
  }
  const op = expression.operator.slice(0, -1);
  if (!isVectorArithmeticOperator(op)) return undefined;
  const right = emitExpressionAsVectorOperand(expression.right, type, context);
  return `${left} = (${left} ${op} ${right})`;
}

function emitVectorLaneSet(name: string, type: CudaLiteScalarType, index: string, value: string): string {
  return emitVectorLaneSetExpression(name, type, index, value);
}

function emitPointerRebaseAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string | undefined {
  if (expression.left.kind !== "identifier") return undefined;
  const handle = context.localPointerHandleFor(expression.left.name);
  if (handle) {
    const buffer = context.nameFor(`${handle.name}_buffer`);
    const base = context.nameFor(`${handle.name}_base`);
    if (expression.operator === "=") {
      const parts = devicePointerArgumentParts(expression.right, context);
      if (!parts) throw featureError("unsupported-pointer-assignment", "CUDA pointer assignment expects a modeled storage or shared pointer");
      return `${buffer} = ${parts.buffer}; ${base} = ${parts.base}`;
    }
    if (expression.operator === "+=" || expression.operator === "-=") {
      const delta = `u32(${emitExpression(expression.right, context)})`;
      const op = expression.operator === "+=" ? "+" : "-";
      return `${base} = (${base} ${op} ${delta})`;
    }
    return undefined;
  }
  const base = context.mutablePointerBaseFor(expression.left.name);
  if (!base) return undefined;
  if (expression.operator === "=") {
    const parts = devicePointerArgumentParts(expression.right, context);
    if (!parts) throw featureError("unsupported-pointer-assignment", "CUDA pointer assignment expects a modeled storage or shared pointer");
    const expectedBuffer = context.storagePointerIdFor(expression.left.name);
    if (expectedBuffer !== undefined && parts.buffer !== `${expectedBuffer}u`) {
      throw featureError("unsupported-pointer-assignment", "storage pointer parameter assignment must stay within the same buffer");
    }
    return `${base} = ${parts.base}`;
  }
  if (expression.operator !== "+=" && expression.operator !== "-=") return undefined;
  const delta = `u32(${emitExpression(expression.right, context)})`;
  const op = expression.operator === "+=" ? "+" : "-";
  return `${base} = (${base} ${op} ${delta})`;
}

function emitPointerRebaseUpdate(expression: CudaLiteExpression, context: EmitContext): string | undefined {
  if (expression.kind !== "update" || expression.argument.kind !== "identifier") return undefined;
  const base = context.mutablePointerBaseFor(expression.argument.name);
  if (!base) return undefined;
  return `${base} = (${base} ${expression.operator === "++" ? "+" : "-"} 1u)`;
}

function emitUpdateExpression(
  expression: Extract<CudaLiteExpression, { kind: "update" }>,
  context: EmitContext,
): string {
  const pointerRebase = emitPointerRebaseUpdate(expression, context);
  if (pointerRebase) return pointerRebase;
  const directStorageUpdate = emitDirectStorageScalarUpdate(expression, context);
  if (directStorageUpdate) return directStorageUpdate;
  const pointerLvalue = devicePointerLValue(expression.argument, context);
  if (pointerLvalue) {
    const read = `${pointerReadHelperName(pointerLvalue.valueType)}(${pointerLvalue.buffer}, ${pointerLvalue.index})`;
    const delta = updateDeltaForValueType(pointerLvalue.valueType);
    const value = `(${read} ${expression.operator === "++" ? "+" : "-"} ${delta})`;
    return `${pointerWriteHelperName(pointerLvalue.valueType)}(${pointerLvalue.buffer}, ${pointerLvalue.index}, ${value})`;
  }
  const target = emitExpression(expression.argument, context, "lvalue");
  const type = expressionValueTypeForEmit(expression.argument, context);
  const delta = updateDeltaForValueType(type);
  return `${target} = (${target} ${expression.operator === "++" ? "+" : "-"} ${delta})`;
}

function emitDirectStorageScalarUpdate(
  expression: Extract<CudaLiteExpression, { kind: "update" }>,
  context: EmitContext,
): string | undefined {
  const root = directLValueRootName(expression.argument);
  if (!root || context.devicePointerParamFor(root)) return undefined;
  const param = context.paramFor(root);
  const global = context.deviceGlobalFor(root);
  if (
    expression.argument.kind !== "index" ||
    expression.argument.target.kind !== "identifier" ||
    !((param?.pointer && !isCudaVectorType(param.valueType) && !context.ir.atomicParams.includes(param.name)) ||
      (global !== undefined && !isCudaVectorType(global.valueType) && !context.ir.atomicDeviceGlobals.includes(global.name)))
  ) {
    return undefined;
  }
  const valueType = param?.valueType ?? global?.valueType;
  if (!valueType) return undefined;
  const target = emitExpression(expression.argument, context, "lvalue");
  const current = emitExpressionAsStorageElementValueType(expression.argument, valueType, context);
  return `${target} = (${current} ${expression.operator === "++" ? "+" : "-"} ${updateDeltaForValueType(valueType)})`;
}

function emitVectorAssignment(expression: CudaLiteAssignmentExpression, context: EmitContext): string | undefined {
  const direct = vectorStorageLValue(expression.left, context);
  if (!direct) return undefined;
  const shared = direct.rootName ? sharedDeclarationFor(direct.rootName, context) : undefined;
  const local = direct.rootName ? localArrayForStorageView(direct.rootName, expression.left.span, context) : undefined;
  const param = direct.rootName ? context.paramFor(direct.rootName) : undefined;
  const global = direct.rootName ? context.deviceGlobalFor(direct.rootName) : undefined;
  const atomicVectorParam = param && context.ir.atomicParams.includes(param.name) ? param : undefined;
  const atomicVectorGlobal = global && context.ir.atomicDeviceGlobals.includes(global.name) ? global : undefined;
  if (atomicVectorParam || atomicVectorGlobal) {
    const scalar = cudaVectorScalarType(direct.valueType) ?? "float";
    if (direct.fieldIndex !== undefined) {
      const index = `(${vectorStorageBase(direct.index, direct.lanes)} + ${direct.fieldIndex}u)`;
      const current = atomicVectorParam
        ? emitPointerStorageRead(atomicVectorParam, index, context.ir, context, scalar)
        : emitDeviceGlobalPointerRead(atomicVectorGlobal!, index, context.ir, context, scalar);
      const right = emitExpressionAsValueType(expression.right, scalar, context);
      const value = expression.operator === "="
        ? right
        : `(${current} ${expression.operator.slice(0, -1)} ${right})`;
      return atomicVectorParam
        ? emitPointerStorageWrite(atomicVectorParam, index, value, context.ir, context, scalar)
        : emitDeviceGlobalPointerWrite(atomicVectorGlobal!, index, value, context.ir, context, scalar);
    }
    const right = emitExpressionAsValueType(expression.right, direct.valueType, context);
    const value = expression.operator === "="
      ? right
      : `(${emitAtomicVectorDirectRead(atomicVectorParam, atomicVectorGlobal, direct, context)} ${expression.operator.slice(0, -1)} ${emitExpressionAsVectorOperand(expression.right, direct.valueType as CudaLiteVectorType, context)})`;
    return emitAtomicVectorDirectWrite(atomicVectorParam, atomicVectorGlobal, direct, value, context);
  }
  if ((shared || local) && direct.fieldIndex !== undefined) {
    const scalar = cudaVectorScalarType(direct.valueType) ?? "float";
    const currentVector = shared
      ? emitSharedPointerRead(shared, direct.index, context.ir, context, direct.valueType)
      : emitLocalPointerRead(local!, direct.index, direct.valueType, context);
    const current = `${currentVector}.${direct.field}`;
    const right = emitExpressionAsValueType(expression.right, scalar, context);
    const laneValue = expression.operator === "="
      ? right
      : `(${current} ${expression.operator.slice(0, -1)} ${right})`;
    const value = emitVectorLaneSetExpression(currentVector, direct.valueType, direct.fieldIndex, laneValue);
    return shared
      ? emitSharedPointerWrite(shared, direct.index, value, context.ir, context, direct.valueType)
      : emitLocalPointerWrite(local!, direct.index, value, direct.valueType, context);
  }
  const right = direct.field
    ? emitExpressionAsValueType(expression.right, cudaVectorScalarType(direct.valueType) ?? "float", context)
    : emitExpressionAsValueType(expression.right, direct.valueType, context);
  if (direct.field) {
    if (expression.operator !== "=") {
      const current = `${direct.name}[${vectorStorageBase(direct.index, direct.lanes)} + ${direct.fieldIndex ?? 0}u]`;
      const op = expression.operator.slice(0, -1);
      return `${current} = (${current} ${op} ${right})`;
    }
    return emitVectorStorageFieldWrite(direct.name, direct.valueType, direct.index, direct.field, right);
  }
  if (expression.operator !== "=") {
    const current = emitVectorStorageRead(direct.name, direct.valueType, direct.index);
    const op = expression.operator.slice(0, -1);
    const vectorRight = emitExpressionAsVectorOperand(expression.right, direct.valueType as CudaLiteVectorType, context);
    return emitVectorStorageWrite(direct.name, direct.valueType, direct.index, `(${current} ${op} ${vectorRight})`);
  }
  return emitVectorStorageWrite(direct.name, direct.valueType, direct.index, right);
}

function emitAtomicVectorDirectRead(
  param: CudaLiteParam | undefined,
  global: CudaLiteDeviceGlobal | undefined,
  direct: VectorStorageLValue,
  context: EmitContext,
): string {
  const scalar = cudaVectorScalarType(direct.valueType) ?? "float";
  const base = vectorStorageBase(direct.index, direct.lanes);
  const values = Array.from({ length: direct.lanes }, (_, lane) => {
    const index = `(${base} + ${lane}u)`;
    return param
      ? emitPointerStorageRead(param, index, context.ir, context, scalar)
      : emitDeviceGlobalPointerRead(global!, index, context.ir, context, scalar);
  });
  return `${wgslScalar(direct.valueType)}(${values.join(", ")})`;
}

function emitAtomicVectorDirectWrite(
  param: CudaLiteParam | undefined,
  global: CudaLiteDeviceGlobal | undefined,
  direct: VectorStorageLValue,
  value: string,
  context: EmitContext,
): string {
  const scalar = cudaVectorScalarType(direct.valueType) ?? "float";
  const base = vectorStorageBase(direct.index, direct.lanes);
  return Array.from({ length: direct.lanes }, (_, lane) => {
    const index = `(${base} + ${lane}u)`;
    const laneValue = `${value}.${vectorFieldName(lane)}`;
    return param
      ? emitPointerStorageWrite(param, index, laneValue, context.ir, context, scalar)
      : emitDeviceGlobalPointerWrite(global!, index, laneValue, context.ir, context, scalar);
  }).join("; ");
}

function vectorStorageLValue(expression: CudaLiteExpression, context: EmitContext): VectorStorageLValue | undefined {
  return vectorStorageLValueImpl(expression, context, storageViewCallbacks);
}

function storageViewLValue(expression: CudaLiteExpression, context: EmitContext): VectorStorageLValue | undefined {
  return storageViewLValueImpl(expression, context, storageViewCallbacks);
}

function scalarStorageViewLValue(expression: CudaLiteExpression, context: EmitContext): ScalarStorageViewLValue | undefined {
  return scalarStorageViewLValueImpl(expression, context, storageViewCallbacks);
}

function scalarParamStorageViewLValue(expression: CudaLiteExpression, context: EmitContext): ScalarParamStorageViewLValue | undefined {
  return scalarParamStorageViewLValueImpl(expression, context, storageViewCallbacks);
}

function storageViewForPointerExpression(
  pointer: CudaLiteExpression,
  index: CudaLiteExpression,
  context: EmitContext,
): StorageView | undefined {
  return storageViewForPointerExpressionImpl(pointer, index, context, storageViewCallbacks);
}

function noopCallComment(expression: CudaLiteExpression): string | undefined {
  if (expression.kind !== "call") return undefined;
  switch (expressionName(expression.callee)) {
    case "assert":
      return "assert omitted: WebGPU has no device abort";
    case "__trap":
      return "__trap omitted: WebGPU has no device abort";
    case "printf":
      return "printf omitted: WebGPU has no device stdout";
    case "cudaDeviceSynchronize":
      return "cudaDeviceSynchronize omitted: WebGPU dispatch completion is host-managed";
    case "cudaStreamCreate":
    case "cudaStreamCreateWithFlags":
    case "cudaStreamDestroy":
    case "cudaStreamSynchronize":
    case "cudaEventCreate":
    case "cudaEventCreateWithFlags":
    case "cudaEventDestroy":
    case "cudaEventRecord":
    case "cudaEventSynchronize":
      return `${expressionName(expression.callee)} omitted: WebGPU stream/event orchestration is host-managed`;
    case "cudaMemcpy":
      return "cudaMemcpy omitted: WebGPU copy orchestration is host-managed";
    case "cudaMemcpyAsync":
      return "cudaMemcpyAsync omitted: WebGPU copy orchestration is host-managed";
    case "cudaMemcpyPeerAsync":
      return "cudaMemcpyPeerAsync omitted: WebGPU copy orchestration is host-managed";
    case "cudaGraphSetConditional":
      return "cudaGraphSetConditional omitted: CUDA graph conditional scheduling is host-managed";
    default:
      return undefined;
  }
}

function isEmittedPointerVar(statement: CudaLiteVarDecl, context: EmitContext): boolean {
  return statement.valueType === "voidptr" || context.poolPointerFor(statement.name) !== undefined;
}

function indent(level: number): string {
  return "  ".repeat(level);
}

function featureError(code: string, message: string): CudaLiteCompilerError {
  const span: SourceSpan = { start: 0, end: 0, line: 1, column: 1 };
  return new CudaLiteCompilerError(message, [{ code, severity: "error", message, span }]);
}
