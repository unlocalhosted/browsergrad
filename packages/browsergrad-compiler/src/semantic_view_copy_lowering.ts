import { wireIntegerToBigInt } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  assertSemanticIntegerInterval,
  lowerSemanticIndexExpressionWithInterval,
  lowerSemanticPredicateExpression,
  semanticScalarBinary,
  semanticScalarCast,
  semanticScalarLiteral,
  SemanticIndexMapLoweringError,
  unflattenSemanticRowMajorIndex,
  type SemanticCoordinateExpression,
  type SemanticIntegerInterval,
} from "./semantic_index_map_lowering.js";
import {
  semanticIdKey,
  semanticIdsEqual,
  semanticMemoryIdFromSymbol,
} from "./semantic_ids.js";
import type {
  CudaLiteSemanticSymbol,
  SemanticExpression,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir_types.js";
import type { RuntimeLoweredSemanticKernelIr } from "./semantic_runtime_lowering.js";
import {
  CudaLiteViewCopyBindingError,
  unwrapPreparedCudaLiteViewCopyBinding,
  type PreparedCudaLiteViewCopyBinding,
  type PreparedCudaLiteViewCopyBindingRecord,
} from "./semantic_view_copy_bindings.js";
import type { CompileCudaLiteOptions, SourceSpan } from "./types.js";

const I32_MAX = (1n << 31n) - 1n;
const U32_MAX = (1n << 32n) - 1n;
const F32_BYTES = 4n;

interface ViewCopyLoweringContext {
  readonly prepared: PreparedCudaLiteViewCopyBinding;
  readonly record: PreparedCudaLiteViewCopyBindingRecord;
  readonly sourceParameter: CudaLiteSemanticSymbol;
  readonly destinationParameter: CudaLiteSemanticSymbol;
  readonly sourceParameterId: string;
  readonly destinationParameterId: string;
  readonly logicalShape: readonly bigint[];
  readonly logicalElementCount: bigint;
  readonly dimensionValues: ReadonlyMap<string, bigint>;
  copyStores: number;
  sourceBranches: number;
}

export function lowerCudaLiteViewCopyBinding(
  ir: RuntimeLoweredSemanticKernelIr,
  prepared: PreparedCudaLiteViewCopyBinding,
  options: CompileCudaLiteOptions,
): RuntimeLoweredSemanticKernelIr {
  try {
    return lowerCudaLiteViewCopyBindingUnchecked(ir, prepared, options);
  } catch (error) {
    if (error instanceof CudaLiteViewCopyBindingError && error.span === undefined) {
      throw new CudaLiteViewCopyBindingError(error.code, error.path, error.message, { cause: error }, ir.span);
    }
    throw error;
  }
}

function lowerCudaLiteViewCopyBindingUnchecked(
  ir: RuntimeLoweredSemanticKernelIr,
  prepared: PreparedCudaLiteViewCopyBinding,
  options: CompileCudaLiteOptions,
): RuntimeLoweredSemanticKernelIr {
  const record = unwrapPreparedCudaLiteViewCopyBinding(prepared);
  const sourceParameter = requiredParameter(ir, prepared.sourceParameter, "$.sourceParameter");
  const destinationParameter = requiredParameter(ir, prepared.destinationParameter, "$.destinationParameter");
  validateKernelSurface(ir, sourceParameter, destinationParameter, options);
  const logicalShape = record.specialization.logicalShape;
  const logicalElementCount = record.specialization.elementCount;
  requireU32(logicalElementCount, "$.operation", "logical element count");
  validateAccessorRange(record.specialization.source.allocationByteLength, "$.operation.source");
  validateAccessorRange(record.specialization.destination.allocationByteLength, "$.operation.destination");
  const context: ViewCopyLoweringContext = {
    prepared,
    record,
    sourceParameter,
    destinationParameter,
    sourceParameterId: semanticIdKey(sourceParameter.id),
    destinationParameterId: semanticIdKey(destinationParameter.id),
    logicalShape,
    logicalElementCount,
    dimensionValues: new Map(Object.entries(record.specialization.bindings).map(([id, value]) => [id, wireIntegerToBigInt(value)])),
    copyStores: 0,
    sourceBranches: 0,
  };
  validateNoPointerAliases(ir, context);
  const operations = mapOperations(ir.operations, context, [], "$.kernelIr.operations");
  if (context.copyStores !== 1) {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-UNSUPPORTED-SOURCE",
      "$.kernelIr.operations",
      `view-copy source must contain exactly one materializing store; found ${context.copyStores}`,
    );
  }
  if (context.sourceBranches !== 1) {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-UNSUPPORTED-SOURCE",
      "$.kernelIr.operations",
      `view-copy source must contain exactly one source branch; found ${context.sourceBranches}`,
    );
  }
  return rewriteRawWordCarriers({ ...ir, operations }, context);
}

function requiredParameter(
  ir: RuntimeLoweredSemanticKernelIr,
  name: string,
  path: string,
): CudaLiteSemanticSymbol {
  const parameter = ir.params.find((candidate) => candidate.name === name);
  if (parameter === undefined) {
    fail("BG-COMPILER-VIEW-COPY-BINDING-UNKNOWN-PARAMETER", path, `kernel has no parameter named ${name}`);
  }
  return parameter;
}

function validateKernelSurface(
  ir: RuntimeLoweredSemanticKernelIr,
  source: CudaLiteSemanticSymbol,
  destination: CudaLiteSemanticSymbol,
  options: CompileCudaLiteOptions,
): void {
  if (ir.params.length !== 2 || !ir.params.every((parameter) => parameter === source || parameter === destination)) {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-UNSUPPORTED-SOURCE",
      "$.kernelIr.params",
      "initial view-copy binding requires exactly the authorized source and destination parameters",
    );
  }
  if (ir.functions.length !== 0) {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-UNSUPPORTED-SOURCE",
      "$.kernelIr.functions",
      "initial view-copy binding does not admit device helper functions",
    );
  }
  validateRootParameter(source, true, "$.sourceParameter");
  validateRootParameter(destination, false, "$.destinationParameter");
  if (options.pointerBaseOffsets?.[source.name] !== undefined || options.pointerBaseOffsets?.[destination.name] !== undefined) {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-UNSUPPORTED-PARAMETER",
      "$.options.pointerBaseOffsets",
      "view-copy parameters cannot also use legacy pointerBaseOffsets",
    );
  }
}

function validateRootParameter(parameter: CudaLiteSemanticSymbol, source: boolean, path: string): void {
  if (
    parameter.kind !== "param" ||
    parameter.pointer !== true ||
    parameter.addressSpace !== "storage" ||
    parameter.valueType !== "float" ||
    parameter.dimensions.length !== 0 ||
    parameter.pointerRuntimeState === true ||
    parameter.pointerMayBeNull === true ||
    parameter.pointerBaseIndices !== undefined ||
    parameter.pointerBaseIsScalarLane === true ||
    parameter.pointerBaseUnitBytes !== undefined ||
    parameter.pointerValid !== undefined ||
    parameter.pointerSelection !== undefined ||
    parameter.pointerArrayAliases !== undefined ||
    parameter.pointerParamAlias !== undefined ||
    parameter.pointerMemoryAlias !== undefined ||
    parameter.pointerRoot !== undefined ||
    parameter.pointerCarrierValueType !== undefined ||
    parameter.packedByteLanes !== undefined ||
    (source ? parameter.constant !== true : parameter.constant === true)
  ) {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-UNSUPPORTED-PARAMETER",
      path,
      `${parameter.name} must be an unaliased ${source ? "const " : ""}float* global-storage root`,
    );
  }
}

function validateNoPointerAliases(ir: RuntimeLoweredSemanticKernelIr, context: ViewCopyLoweringContext): void {
  for (const [index, symbol] of ir.symbols.entries()) {
    const key = semanticIdKey(symbol.id);
    if (key === context.sourceParameterId || key === context.destinationParameterId) continue;
    if (symbol.pointer === true) {
      fail(
        "BG-COMPILER-VIEW-COPY-BINDING-UNSUPPORTED-SOURCE",
        `$.kernelIr.symbols[${index}]`,
        `view-copy source cannot introduce pointer ${symbol.name}`,
      );
    }
  }
}

function mapOperations(
  operations: readonly SemanticKernelIrOperation[],
  context: ViewCopyLoweringContext,
  guards: readonly SemanticExpression[],
  path: string,
): readonly SemanticKernelIrOperation[] {
  return operations.map((operation, index) => mapOperation(operation, context, guards, `${path}[${index}]`));
}

function mapOperation(
  operation: SemanticKernelIrOperation,
  context: ViewCopyLoweringContext,
  guards: readonly SemanticExpression[],
  path: string,
): SemanticKernelIrOperation {
  switch (operation.kind) {
    case "declare":
      if (operation.target.pointer === true || operation.init !== undefined && boundParameterInExpression(operation.init, context)) {
        unsupported(path, "pointer or bound-buffer declaration");
      }
      return operation;
    case "dim3-declare":
      if (operation.args.some((arg) => boundParameterInExpression(arg, context))) unsupported(path, "bound-buffer dim3 declaration");
      return operation;
    case "branch": {
      if (boundParameterInExpression(operation.condition, context)) unsupported(`${path}.condition`, "bound-buffer branch condition");
      context.sourceBranches += 1;
      if (context.sourceBranches > 1 || operation.alternate.length !== 0) {
        unsupported(path, "multiple, nested, or alternate source branches");
      }
      return {
        ...operation,
        consequent: mapOperations(operation.consequent, context, [...guards, operation.condition], `${path}.consequent`),
        alternate: [],
      };
    }
    case "block": return {
      ...operation,
      body: mapOperations(operation.body, context, guards, `${path}.body`),
    };
    case "store": return lowerMaterializingStore(operation, context, guards, path);
    case "return": unsupported(path, "return operation");
    default: unsupported(path, `${operation.kind} operation`);
  }
}

function lowerMaterializingStore(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "store" }>,
  context: ViewCopyLoweringContext,
  guards: readonly SemanticExpression[],
  path: string,
): SemanticKernelIrOperation {
  if (semanticIdKey(operation.target.baseId) !== context.destinationParameterId) {
    unsupported(path, `store to ${operation.target.base}`);
  }
  if (context.copyStores !== 0) unsupported(path, "multiple destination stores");
  if (operation.operator !== "=") unsupported(`${path}.operator`, `destination store operator ${operation.operator}`);
  validateDirectMemoryRef(operation.target, context.destinationParameterId, `${path}.target`);
  if (
    operation.value.kind !== "index" ||
    operation.value.target.kind !== "symbol" ||
    semanticIdKey(operation.value.target.id) !== context.sourceParameterId ||
    operation.value.addressSpace !== "storage" ||
    operation.value.valueType !== "float"
  ) {
    unsupported(`${path}.value`, "non-direct source expression");
  }
  if (operation.reads.length !== 1) unsupported(`${path}.reads`, `${operation.reads.length} source reads`);
  const sourceRead = operation.reads[0];
  if (sourceRead === undefined) throw new Error("internal: source read disappeared");
  validateDirectMemoryRef(sourceRead, context.sourceParameterId, `${path}.reads[0]`);
  const destinationFlat = operation.target.indices[0];
  const sourceFlat = operation.value.index;
  const traceFlat = sourceRead.indices[0];
  if (
    destinationFlat === undefined || sourceFlat === undefined || traceFlat === undefined ||
    !sameFlatSymbol(destinationFlat, sourceFlat) || !sameFlatSymbol(sourceFlat, traceFlat)
  ) {
    unsupported(path, "source and destination do not use one identical flat index symbol");
  }
  if (guards.length !== 1 || !isExactLogicalDomainGuard(guards[0]!, sourceFlat, context.logicalElementCount)) {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-MISSING-GUARD",
      path,
      `materializing store must be dominated only by ${sourceFlat.name} < ${context.logicalElementCount}u`,
      operation.span,
    );
  }

  const coordinates = lowerCoordinates(sourceFlat, context, `${path}.index`);
  const sourcePredicate = withIndexMapErrors(() => lowerSemanticPredicateExpression(
    context.record.sourceIndexMap.inBounds,
    coordinates,
    context.dimensionValues,
    "int",
    operation.span,
    `${path}.sourcePredicate`,
  ), `${path}.sourcePredicate`, operation.span);
  const sourceWord = rootWordExpression(
    context.record.sourceIndexMap,
    context.record.specialization.source.viewByteOffset,
    context.record.specialization.source.allocationByteLength,
    coordinates,
    context,
    operation.span,
    `${path}.source`,
    true,
  );
  const destinationWord = rootWordExpression(
    context.record.destinationIndexMap,
    context.record.specialization.destination.viewByteOffset,
    context.record.specialization.destination.allocationByteLength,
    coordinates,
    context,
    operation.span,
    `${path}.destination`,
    false,
  );
  const rawSourceRead = rawWordMemoryRef(sourceRead, sourceWord);
  const rawDestination = rawWordMemoryRef(operation.target, destinationWord);
  const actualStore: Extract<SemanticKernelIrOperation, { readonly kind: "store" }> = {
    ...operation,
    target: rawDestination,
    value: {
      ...operation.value,
      target: { ...operation.value.target, valueType: "uint" },
      index: sourceWord,
      valueType: "uint",
    },
    reads: [rawSourceRead],
  };
  const policy = context.record.specialization.operation.source.invalidSource;
  const alternate = policy.kind === "reject" ? [] : [{
    ...operation,
    target: rawDestination,
    value: semanticScalarLiteral(BigInt(`0x${policy.value.bits}`), "uint", operation.span, `${path}.fill`),
    reads: [],
  } satisfies Extract<SemanticKernelIrOperation, { readonly kind: "store" }>];
  context.copyStores += 1;
  return {
    kind: "block",
    body: [{
      kind: "branch",
      condition: sourcePredicate,
      consequent: [actualStore],
      alternate,
      span: operation.span,
    }],
    span: operation.span,
  };
}

function validateDirectMemoryRef(ref: SemanticMemoryRef, expectedParameterId: string, path: string): void {
  if (
    semanticIdKey(ref.baseId) !== expectedParameterId ||
    ref.addressSpace !== "storage" ||
    ref.valueType !== "float" ||
    ref.containerValueType !== "float" ||
    ref.indices.length !== 1 ||
    ref.fields.length !== 0 ||
    ref.pointerBaseIsScalarLane === true ||
    ref.pointerBaseUnitBytes !== undefined ||
    ref.packedByteLanes !== undefined
  ) {
    unsupported(path, "typed, fielded, aliased, vectorized, or multi-index memory reference");
  }
}

function lowerCoordinates(
  flat: Extract<SemanticExpression, { readonly kind: "symbol" }>,
  context: ViewCopyLoweringContext,
  path: string,
): readonly SemanticCoordinateExpression[] {
  return withIndexMapErrors(
    () => unflattenSemanticRowMajorIndex(flat, context.logicalShape, "int", path),
    path,
    flat.span,
  );
}

function rootWordExpression(
  indexMap: PreparedCudaLiteViewCopyBindingRecord["sourceIndexMap"],
  viewByteOffset: bigint,
  allocationByteLength: bigint,
  coordinates: readonly SemanticCoordinateExpression[],
  context: ViewCopyLoweringContext,
  span: SourceSpan,
  path: string,
  guardedSource: boolean,
): SemanticExpression {
  if (indexMap.locationUnit !== "element") {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-UNSUPPORTED-SOURCE",
      `${path}.indexMap.locationUnit`,
      "initial view-copy binding requires element-unit index maps",
      span,
    );
  }
  if (viewByteOffset % F32_BYTES !== 0n) {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-UNSUPPORTED-SOURCE",
      `${path}.viewByteOffset`,
      "f32 view byte offset is not word aligned",
      span,
    );
  }
  const lowered = withIndexMapErrors(() => lowerSemanticIndexExpressionWithInterval(
    indexMap.location,
    coordinates,
    context.dimensionValues,
    "int",
    span,
    `${path}.indexMap.location`,
  ), `${path}.indexMap.location`, span);
  const wordOffset = viewByteOffset / F32_BYTES;
  const interval = {
    minimum: lowered.interval.minimum + wordOffset,
    maximum: lowered.interval.maximum + wordOffset,
  };
  withIndexMapErrors(() => assertSemanticIntegerInterval(interval, "int", `${path}.rootWord`), `${path}.rootWord`, span);
  const rootWord = wordOffset === 0n
    ? lowered.expression
    : semanticScalarBinary(
        "+",
        semanticScalarLiteral(wordOffset, "int", span, `${path}.viewByteOffset`),
        lowered.expression,
        "int",
        span,
      );
  if (!guardedSource) validateDestinationInterval(interval, allocationByteLength, path, span);
  return semanticScalarCast(rootWord, "uint", span);
}

function validateDestinationInterval(
  interval: SemanticIntegerInterval,
  allocationByteLength: bigint,
  path: string,
  span: SourceSpan,
): void {
  const allocationWords = allocationByteLength / F32_BYTES;
  if (interval.minimum < 0n || interval.maximum >= allocationWords) {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-INTEGER-RANGE",
      `${path}.rootWord`,
      `destination word interval [${interval.minimum}, ${interval.maximum}] escapes ${allocationWords} words`,
      span,
    );
  }
}

function validateAccessorRange(allocationByteLength: bigint, path: string): void {
  if (allocationByteLength <= 0n || allocationByteLength % F32_BYTES !== 0n) {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-INTEGER-RANGE",
      `${path}.allocationByteLength`,
      "view-copy allocation must be a positive whole number of f32 words",
    );
  }
  const maximumWord = allocationByteLength / F32_BYTES - 1n;
  if (maximumWord > I32_MAX || maximumWord > U32_MAX) {
    fail(
      "BG-COMPILER-VIEW-COPY-BINDING-INTEGER-RANGE",
      `${path}.allocationByteLength`,
      `allocation word index ${maximumWord} is outside the signed compiler address profile`,
    );
  }
}

function rawWordMemoryRef(ref: SemanticMemoryRef, index: SemanticExpression): SemanticMemoryRef {
  return {
    ...ref,
    valueType: "uint",
    containerValueType: "uint",
    indices: [index],
  };
}

function rewriteRawWordCarriers(
  ir: RuntimeLoweredSemanticKernelIr,
  context: ViewCopyLoweringContext,
): RuntimeLoweredSemanticKernelIr {
  const rewrite = (symbol: CudaLiteSemanticSymbol): CudaLiteSemanticSymbol => {
    const key = semanticIdKey(symbol.id);
    return key === context.sourceParameterId || key === context.destinationParameterId
      ? { ...symbol, valueType: "uint" }
      : symbol;
  };
  return {
    ...ir,
    params: ir.params.map(rewrite),
    symbols: ir.symbols.map(rewrite),
    memory: ir.memory.map(rewrite),
    launchableEntries: ir.launchableEntries.map((entry) => ({
      ...entry,
      params: entry.params.map(rewrite),
    })),
  };
}

function sameFlatSymbol(
  left: SemanticExpression,
  right: SemanticExpression,
): left is Extract<SemanticExpression, { readonly kind: "symbol" }> {
  return left.kind === "symbol" && right.kind === "symbol" &&
    left.valueType === "uint" && right.valueType === "uint" && semanticIdsEqual(left.id, right.id);
}

function isExactLogicalDomainGuard(
  condition: SemanticExpression,
  flat: Extract<SemanticExpression, { readonly kind: "symbol" }>,
  elementCount: bigint,
): boolean {
  return condition.kind === "binary" && condition.operator === "<" &&
    condition.left.kind === "symbol" && semanticIdsEqual(condition.left.id, flat.id) &&
    condition.right.kind === "literal" && condition.right.literalKind === "number" &&
    Number.isSafeInteger(condition.right.value) && BigInt(condition.right.value) === elementCount;
}

function boundParameterInExpression(
  expression: SemanticExpression,
  context: ViewCopyLoweringContext,
): boolean {
  if (expression.kind === "symbol") return isBoundId(expression.id, context);
  if (expression.kind === "pointer-valid") return isBoundId(expression.pointerId, context);
  switch (expression.kind) {
    case "literal": return false;
    case "member": return boundParameterInExpression(expression.object, context);
    case "index": return boundParameterInExpression(expression.target, context) || boundParameterInExpression(expression.index, context);
    case "call": return boundParameterInExpression(expression.callee, context) || expression.args.some((arg) => boundParameterInExpression(arg, context));
    case "texture-read": return [expression.texture, expression.x, expression.y, ...(expression.z === undefined ? [] : [expression.z])]
      .some((value) => boundParameterInExpression(value, context));
    case "surface-read": return [expression.surface, expression.xBytes, expression.y, ...(expression.z === undefined ? [] : [expression.z])]
      .some((value) => boundParameterInExpression(value, context));
    case "cast": return boundParameterInExpression(expression.expression, context);
    case "unary": return boundParameterInExpression(expression.argument, context);
    case "binary": return boundParameterInExpression(expression.left, context) || boundParameterInExpression(expression.right, context);
    case "conditional": return [expression.condition, expression.consequent, expression.alternate]
      .some((value) => boundParameterInExpression(value, context));
    case "assignment": return boundParameterInExpression(expression.target, context) || boundParameterInExpression(expression.value, context);
    case "update": return boundParameterInExpression(expression.argument, context);
    case "initializer": return expression.elements.some((value) => boundParameterInExpression(value, context));
    case "sequence": return expression.expressions.some((value) => boundParameterInExpression(value, context));
  }
}

function isBoundId(id: CudaLiteSemanticSymbol["id"], context: ViewCopyLoweringContext): boolean {
  const key = semanticIdKey(id);
  return key === context.sourceParameterId || key === context.destinationParameterId ||
    semanticIdsEqual(semanticMemoryIdFromSymbol(id), semanticMemoryIdFromSymbol(context.sourceParameter.id)) ||
    semanticIdsEqual(semanticMemoryIdFromSymbol(id), semanticMemoryIdFromSymbol(context.destinationParameter.id));
}

function requireU32(value: bigint, path: string, name: string): void {
  if (value < 0n || value > U32_MAX) {
    fail("BG-COMPILER-VIEW-COPY-BINDING-INTEGER-RANGE", path, `${name} ${value} is outside u32`);
  }
}

function withIndexMapErrors<T>(fn: () => T, path: string, span: SourceSpan): T {
  try {
    return fn();
  } catch (cause) {
    if (!(cause instanceof SemanticIndexMapLoweringError)) throw cause;
    const code = cause.code === "integer-range"
      ? "BG-COMPILER-VIEW-COPY-BINDING-INTEGER-RANGE"
      : "BG-COMPILER-VIEW-COPY-BINDING-UNSUPPORTED-SOURCE";
    throw new CudaLiteViewCopyBindingError(code, cause.path || path, cause.message, { cause }, span);
  }
}

function unsupported(path: string, use: string): never {
  fail(
    "BG-COMPILER-VIEW-COPY-BINDING-UNSUPPORTED-SOURCE",
    path,
    `initial view-copy binding does not support ${use}`,
  );
}

function fail(
  code: ConstructorParameters<typeof CudaLiteViewCopyBindingError>[0],
  path: string,
  message: string,
  span?: SourceSpan,
): never {
  throw new CudaLiteViewCopyBindingError(code, path, message, undefined, span);
}
