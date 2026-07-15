import type { IndexExpr } from "@unlocalhosted/browsergrad-semantic-core/layout";
import { wireIntegerToBigInt } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { isSemanticKernelIrOperation } from "./semantic_ir.js";
import type {
  CudaLiteSemanticSymbol,
  SemanticExpression,
  SemanticKernelIrOperation,
  SemanticMatrixTileRef,
  SemanticMemoryRef,
  SemanticPointerAlias,
} from "./semantic_ir_types.js";
import {
  semanticIdKey,
  semanticIdsEqual,
  semanticMemoryIdFromSymbol,
} from "./semantic_ids.js";
import { walkSemanticOperations } from "./semantic_ir_walk.js";
import type { RuntimeLoweredSemanticKernelIr } from "./semantic_runtime_lowering.js";
import type { CompileCudaLiteOptions, SourceSpan } from "./types.js";
import {
  CudaLiteLayoutBindingError,
  unwrapPreparedCudaLiteLayoutBindings,
  type PreparedCudaLiteLayoutBindingRecord,
  type PreparedCudaLiteLayoutBindings,
} from "./semantic_layout_bindings.js";

const U32_MAX = (1n << 32n) - 1n;

interface IntegerInterval {
  readonly minimum: bigint;
  readonly maximum: bigint;
}

interface BindingContext {
  readonly record: PreparedCudaLiteLayoutBindingRecord;
  readonly parameter: CudaLiteSemanticSymbol;
  readonly parameterId: string;
  readonly logicalShape: readonly bigint[];
  readonly logicalElementCount: bigint;
  readonly dimensionValues: ReadonlyMap<string, bigint>;
  readonly coordinateIntervals: readonly IntegerInterval[];
  readonly mutatedSymbols: ReadonlySet<string>;
  uses: number;
}

interface LoweringContext {
  readonly bindingsByParameterId: ReadonlyMap<string, BindingContext>;
}

type MemoryRefMode = "direct-read" | "write" | "unsupported";
type ExpressionMode = "read" | "write" | "unsupported";

export function lowerCudaLiteLayoutBindings(
  ir: RuntimeLoweredSemanticKernelIr,
  prepared: PreparedCudaLiteLayoutBindings,
  options: CompileCudaLiteOptions,
): RuntimeLoweredSemanticKernelIr {
  try {
    return lowerCudaLiteLayoutBindingsUnchecked(ir, prepared, options);
  } catch (error) {
    if (error instanceof CudaLiteLayoutBindingError && error.span === undefined) {
      throw new CudaLiteLayoutBindingError(error.code, error.path, error.message, { cause: error }, ir.span);
    }
    throw error;
  }
}

function lowerCudaLiteLayoutBindingsUnchecked(
  ir: RuntimeLoweredSemanticKernelIr,
  prepared: PreparedCudaLiteLayoutBindings,
  options: CompileCudaLiteOptions,
): RuntimeLoweredSemanticKernelIr {
  const preparedRecord = unwrapPreparedCudaLiteLayoutBindings(prepared);
  const mutatedSymbols = collectMutatedSymbols(ir);
  const bindings = preparedRecord.bindings.map((record, index) => (
    createBindingContext(ir, record, mutatedSymbols, options, `$.bindings[${index}]`)
  ));
  validateNoAliasMetadata(ir, bindings);
  const context: LoweringContext = {
    bindingsByParameterId: new Map(bindings.map((binding) => [binding.parameterId, binding])),
  };
  const operations = mapOperations(ir.operations, context, [], "$.kernelIr.operations");
  const functions = ir.functions.map((fn, index) => ({
    ...fn,
    body: mapOperations(fn.body, context, [], `$.kernelIr.functions[${index}].body`),
  }));
  for (const [index, binding] of bindings.entries()) {
    if (binding.uses === 0) {
      fail(
        "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-USE",
        `$.bindings[${index}].parameter`,
        `bound parameter ${binding.parameter.name} has no direct indexed reads`,
      );
    }
  }
  return { ...ir, operations, functions };
}

function createBindingContext(
  ir: RuntimeLoweredSemanticKernelIr,
  record: PreparedCudaLiteLayoutBindingRecord,
  mutatedSymbols: ReadonlySet<string>,
  options: CompileCudaLiteOptions,
  path: string,
): BindingContext {
  const parameter = ir.params.find((candidate) => candidate.name === record.summary.parameter);
  if (parameter === undefined) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-UNKNOWN-PARAMETER",
      `${path}.parameter`,
      `kernel has no parameter named ${record.summary.parameter}`,
    );
  }
  if (
    parameter.pointer !== true ||
    parameter.constant !== true ||
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
    parameter.packedByteLanes !== undefined
  ) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-PARAMETER",
      `${path}.parameter`,
      `parameter ${parameter.name} must be an unaliased const float* global-storage root`,
    );
  }
  if (record.summary.dtype !== "f32" || record.summary.dtypeBytes !== 4) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-DTYPE",
      `${path}.viewId`,
      `initial compiler layout binding requires f32; view uses ${record.summary.dtype}`,
    );
  }
  if (options.pointerBaseOffsets?.[parameter.name] !== undefined) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-POINTER-OFFSET",
      `$.options.pointerBaseOffsets.${parameter.name}`,
      "layout-bound parameters cannot also use legacy pointerBaseOffsets",
    );
  }
  if (record.indexMap.inBounds.kind !== "bool" || record.indexMap.inBounds.value !== true) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-PREDICATE",
      `${path}.viewId`,
      "initial compiler layout binding requires an always-true verified index-map predicate",
    );
  }

  const logicalShape = record.summary.logicalShape.map((value) => BigInt(value));
  if (logicalShape.length < 2 || logicalShape.length > 3 || logicalShape.some((extent) => extent <= 0n)) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX-MAP",
      `${path}.viewId`,
      "initial compiler layout binding requires a nonempty static rank-2 or rank-3 view",
    );
  }
  const logicalElementCount = logicalShape.reduce((product, extent) => product * extent, 1n);
  requireU32(logicalElementCount, `${path}.viewId`, "logical element count");
  const dimensionValues = new Map(Object.entries(record.summary.dimensionBindings).map(([id, value]) => [id, BigInt(value)]));
  const coordinateIntervals = logicalShape.map((extent) => ({ minimum: 0n, maximum: extent - 1n }));
  const location = indexInterval(record.indexMap.location, coordinateIntervals, dimensionValues, `${path}.indexMap.location`);
  if (location.minimum < 0n) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-INTEGER-RANGE",
      `${path}.indexMap.location`,
      "initial unsigned compiler lowering cannot represent a negative physical location",
    );
  }
  const viewByteOffset = BigInt(record.summary.viewByteOffset);
  const allocationByteLength = BigInt(record.summary.allocationByteLength);
  const dtypeBytes = BigInt(record.summary.dtypeBytes);
  const rootByteMinimum = viewByteOffset + (record.summary.locationUnit === "element" ? location.minimum * dtypeBytes : location.minimum);
  const rootByteMaximumExclusive = viewByteOffset +
    (record.summary.locationUnit === "element" ? location.maximum * dtypeBytes : location.maximum) +
    dtypeBytes;
  if (rootByteMinimum < 0n || rootByteMaximumExclusive > allocationByteLength) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX-MAP",
      `${path}.indexMap.location`,
      "verified logical domain is not wholly contained in the bound allocation",
    );
  }
  const combinedModulo = record.summary.locationUnit === "element"
    ? viewByteOffset % dtypeBytes
    : combineModulo(
      viewByteOffset % dtypeBytes,
      indexModulo(record.indexMap.location, dtypeBytes, dimensionValues),
      dtypeBytes,
    );
  if (combinedModulo !== 0n) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX-MAP",
      `${path}.indexMap.location`,
      "physical byte location is not statically divisible by the f32 element width",
    );
  }
  const physicalMaximum = (rootByteMaximumExclusive - dtypeBytes) / dtypeBytes;
  requireU32(physicalMaximum, `${path}.indexMap.location`, "physical element index");
  if (record.summary.locationUnit === "byte") {
    requireU32(viewByteOffset + location.maximum, `${path}.indexMap.location`, "byte-address lowering intermediate");
  }

  return {
    record,
    parameter,
    parameterId: semanticIdKey(parameter.id),
    logicalShape,
    logicalElementCount,
    dimensionValues,
    coordinateIntervals,
    mutatedSymbols,
    uses: 0,
  };
}

function validateNoAliasMetadata(ir: RuntimeLoweredSemanticKernelIr, bindings: readonly BindingContext[]): void {
  const boundMemoryIds = new Map(bindings.map((binding) => [
    binding.parameterId,
    semanticMemoryIdFromSymbol(binding.parameter.id),
  ]));
  for (const [index, symbol] of ir.symbols.entries()) {
    if (boundMemoryIds.has(semanticIdKey(symbol.id))) continue;
    for (const binding of bindings) {
      const memoryId = boundMemoryIds.get(binding.parameterId);
      if (memoryId === undefined) continue;
      const aliasesParameter = symbol.pointerParamAlias !== undefined &&
        semanticIdKey(symbol.pointerParamAlias) === binding.parameterId;
      const aliasesMemory = symbol.pointerRoot !== undefined && semanticIdsEqual(symbol.pointerRoot, memoryId) ||
        symbol.pointerMemoryAlias !== undefined && semanticIdsEqual(symbol.pointerMemoryAlias, memoryId);
      const aliasesSelection = symbol.pointerSelection !== undefined && (
        pointerAliasMemory(symbol.pointerSelection.consequent, memoryId) ||
        pointerAliasMemory(symbol.pointerSelection.alternate, memoryId)
      );
      if (aliasesParameter || aliasesMemory || aliasesSelection || pointerAliasesMemory(symbol.pointerArrayAliases, memoryId)) {
        fail(
          "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-USE",
          `$.kernelIr.symbols[${index}]`,
          `parameter ${binding.parameter.name} is aliased or forwarded through ${symbol.name}`,
        );
      }
    }
  }
}

function pointerAliasesMemory(
  aliases: CudaLiteSemanticSymbol["pointerArrayAliases"],
  memoryId: ReturnType<typeof semanticMemoryIdFromSymbol>,
): boolean {
  return aliases?.some((alias) => {
    if (alias === undefined) return false;
    return pointerAliasMemory(alias, memoryId);
  }) === true;
}

function pointerAliasMemory(
  alias: SemanticPointerAlias,
  memoryId: ReturnType<typeof semanticMemoryIdFromSymbol>,
): boolean {
  if (alias.pointerRoot !== undefined && semanticIdsEqual(alias.pointerRoot, memoryId)) return true;
  if (alias.pointerSelection === undefined) return false;
  return pointerAliasMemory(alias.pointerSelection.consequent, memoryId) ||
    pointerAliasMemory(alias.pointerSelection.alternate, memoryId);
}

function collectMutatedSymbols(ir: RuntimeLoweredSemanticKernelIr): ReadonlySet<string> {
  const mutated = new Set<string>();
  const markSymbol = (symbol: Extract<SemanticExpression, { readonly kind: "symbol" }>): void => {
    mutated.add(semanticIdKey(symbol.id));
    mutated.add(semanticIdKey(semanticMemoryIdFromSymbol(symbol.id)));
  };
  const markWrittenExpression = (expression: SemanticExpression): void => {
    switch (expression.kind) {
      case "symbol": markSymbol(expression); return;
      case "member": markWrittenExpression(expression.object); return;
      case "index": markWrittenExpression(expression.target); return;
      case "cast": markWrittenExpression(expression.expression); return;
      case "unary": markWrittenExpression(expression.argument); return;
      case "assignment": markWrittenExpression(expression.target); return;
      case "update": markWrittenExpression(expression.argument); return;
      case "conditional":
        markWrittenExpression(expression.consequent);
        markWrittenExpression(expression.alternate);
        return;
      case "sequence":
        for (const item of expression.expressions) markWrittenExpression(item);
        return;
      case "literal":
      case "call":
      case "texture-read":
      case "surface-read":
      case "binary":
      case "initializer": return;
    }
  };
  const visit = (expression: SemanticExpression): void => {
    if (expression.kind === "assignment" && expression.target.kind === "symbol") {
      markSymbol(expression.target);
    }
    if (expression.kind === "update" && expression.argument.kind === "symbol") {
      markSymbol(expression.argument);
    }
    if (expression.kind === "unary" && expression.operator === "&") markWrittenExpression(expression.argument);
  };
  const collectOperationWrites = (operations: readonly SemanticKernelIrOperation[]): void => {
    for (const operation of operations) {
      switch (operation.kind) {
        case "store": mutated.add(semanticIdKey(operation.target.baseId)); break;
        case "copy": mutated.add(semanticIdKey(operation.target.baseId)); break;
        case "matrix-store": mutated.add(semanticIdKey(operation.target.baseId)); break;
        case "atomic":
          if (operation.target !== undefined) mutated.add(semanticIdKey(operation.target.baseId));
          break;
        case "surface-read-store": markWrittenExpression(operation.target); break;
        case "call":
          if (operation.result !== undefined) markSymbol(operation.result);
          break;
        case "inline-asm":
          for (const output of operation.outputs) markWrittenExpression(output);
          break;
        case "branch":
          collectOperationWrites(operation.consequent);
          collectOperationWrites(operation.alternate);
          break;
        case "block": collectOperationWrites(operation.body); break;
        case "loop":
          if (operation.init !== undefined && isSemanticKernelIrOperation(operation.init)) {
            collectOperationWrites([operation.init]);
          }
          collectOperationWrites(operation.body);
          if (operation.continuing !== undefined) collectOperationWrites(operation.continuing);
          break;
        default: break;
      }
    }
  };
  walkSemanticOperations(ir.operations, visit);
  collectOperationWrites(ir.operations);
  for (const fn of ir.functions) {
    walkSemanticOperations(fn.body, visit);
    collectOperationWrites(fn.body);
  }
  return mutated;
}

function mapOperations(
  operations: readonly SemanticKernelIrOperation[],
  context: LoweringContext,
  guards: readonly SemanticExpression[],
  path: string,
): readonly SemanticKernelIrOperation[] {
  return operations.map((operation, index) => mapOperation(operation, context, guards, `${path}[${index}]`));
}

function mapOperation(
  operation: SemanticKernelIrOperation,
  context: LoweringContext,
  guards: readonly SemanticExpression[],
  path: string,
): SemanticKernelIrOperation {
  switch (operation.kind) {
    case "declare": return operation.init === undefined
      ? operation
      : { ...operation, init: mapExpression(operation.init, context, guards, "read", `${path}.init`) };
    case "dim3-declare": return {
      ...operation,
      args: operation.args.map((arg, index) => mapExpression(arg, context, guards, "read", `${path}.args[${index}]`)),
    };
    case "cooperative-group-declare": return operation.declaration.partitionPredicate === undefined
      ? operation
      : {
          ...operation,
          declaration: {
            ...operation.declaration,
            partitionPredicate: mapExpression(operation.declaration.partitionPredicate, context, guards, "read", `${path}.declaration.partitionPredicate`),
          },
        };
    case "load": return { ...operation, source: mapMemoryRef(operation.source, context, guards, "direct-read", `${path}.source`) };
    case "store": return {
      ...operation,
      target: mapMemoryRef(operation.target, context, guards, "write", `${path}.target`),
      value: mapExpression(operation.value, context, guards, "read", `${path}.value`),
      reads: operation.reads.map((ref, index) => mapMemoryRef(ref, context, guards, "direct-read", `${path}.reads[${index}]`)),
    };
    case "copy": return {
      ...operation,
      source: mapMemoryRef(operation.source, context, guards, "unsupported", `${path}.source`),
      target: mapMemoryRef(operation.target, context, guards, "unsupported", `${path}.target`),
    };
    case "copy-fence": return operation;
    case "matrix-fill": return {
      ...operation,
      fragment: mapMatrixRef(operation.fragment, context, guards, `${path}.fragment`),
      value: mapExpression(operation.value, context, guards, "read", `${path}.value`),
    };
    case "matrix-load": return {
      ...operation,
      fragment: mapMatrixRef(operation.fragment, context, guards, `${path}.fragment`),
      source: mapMemoryRef(operation.source, context, guards, "unsupported", `${path}.source`),
      stride: mapExpression(operation.stride, context, guards, "read", `${path}.stride`),
    };
    case "matrix-mma": return {
      ...operation,
      destination: mapMatrixRef(operation.destination, context, guards, `${path}.destination`),
      a: mapMatrixRef(operation.a, context, guards, `${path}.a`),
      b: mapMatrixRef(operation.b, context, guards, `${path}.b`),
      accumulator: mapMatrixRef(operation.accumulator, context, guards, `${path}.accumulator`),
    };
    case "matrix-store": return {
      ...operation,
      target: mapMemoryRef(operation.target, context, guards, "unsupported", `${path}.target`),
      fragment: mapMatrixRef(operation.fragment, context, guards, `${path}.fragment`),
      stride: mapExpression(operation.stride, context, guards, "read", `${path}.stride`),
    };
    case "surface-write": return {
      ...operation,
      surface: mapExpression(operation.surface, context, guards, "unsupported", `${path}.surface`),
      value: mapExpression(operation.value, context, guards, "read", `${path}.value`),
      xBytes: mapExpression(operation.xBytes, context, guards, "read", `${path}.xBytes`),
      y: mapExpression(operation.y, context, guards, "read", `${path}.y`),
      ...(operation.z === undefined ? {} : { z: mapExpression(operation.z, context, guards, "read", `${path}.z`) }),
    };
    case "surface-read-store": return {
      ...operation,
      target: mapExpression(operation.target, context, guards, "write", `${path}.target`),
      surface: mapExpression(operation.surface, context, guards, "read", `${path}.surface`),
      xBytes: mapExpression(operation.xBytes, context, guards, "read", `${path}.xBytes`),
      y: mapExpression(operation.y, context, guards, "read", `${path}.y`),
      ...(operation.z === undefined ? {} : { z: mapExpression(operation.z, context, guards, "read", `${path}.z`) }),
    };
    case "atomic": return {
      ...operation,
      ...(operation.target === undefined ? {} : { target: mapMemoryRef(operation.target, context, guards, "unsupported", `${path}.target`) }),
      args: operation.args.map((arg, index) => mapExpression(arg, context, guards, "unsupported", `${path}.args[${index}]`)),
    };
    case "call": return {
      ...operation,
      args: operation.args.map((arg, index) => mapExpression(arg, context, guards, "read", `${path}.args[${index}]`)),
      reads: operation.reads.map((ref, index) => mapMemoryRef(ref, context, guards, "unsupported", `${path}.reads[${index}]`)),
    };
    case "runtime-copy": return {
      ...operation,
      args: operation.args.map((arg, index) => mapExpression(arg, context, guards, "unsupported", `${path}.args[${index}]`)),
    };
    case "pool-allocate": return {
      ...operation,
      sizeBytes: mapExpression(operation.sizeBytes, context, guards, "read", `${path}.sizeBytes`),
      pool: operation.pool.kind === "device-pool" ? operation.pool : {
        ...operation.pool,
        data: mapMemoryRef(operation.pool.data, context, guards, "unsupported", `${path}.pool.data`),
        offset: mapMemoryRef(operation.pool.offset, context, guards, "unsupported", `${path}.pool.offset`),
        capacityBytes: mapExpression(operation.pool.capacityBytes, context, guards, "read", `${path}.pool.capacityBytes`),
      },
    };
    case "pointer-rebind": return {
      ...operation,
      source: mapMemoryRef(operation.source, context, guards, "unsupported", `${path}.source`),
    };
    case "pointer-array-rebind": return {
      ...operation,
      slot: mapExpression(operation.slot, context, guards, "read", `${path}.slot`),
      source: mapMemoryRef(operation.source, context, guards, "unsupported", `${path}.source`),
    };
    case "expression": return {
      ...operation,
      expression: mapExpression(operation.expression, context, guards, "read", `${path}.expression`),
    };
    case "branch": {
      const condition = mapExpression(operation.condition, context, guards, "read", `${path}.condition`);
      return {
        ...operation,
        condition,
        consequent: mapOperations(operation.consequent, context, [...guards, condition], `${path}.consequent`),
        alternate: mapOperations(operation.alternate, context, guards, `${path}.alternate`),
      };
    }
    case "block": return {
      ...operation,
      body: mapOperations(operation.body, context, guards, `${path}.body`),
    };
    case "loop": return {
      ...operation,
      ...(operation.init === undefined ? {} : {
        init: isSemanticKernelIrOperation(operation.init)
          ? mapOperation(operation.init, context, guards, `${path}.init`)
          : mapExpression(operation.init, context, guards, "read", `${path}.init`),
      }),
      ...(operation.condition === undefined ? {} : {
        condition: mapExpression(operation.condition, context, guards, "read", `${path}.condition`),
      }),
      ...(operation.update === undefined ? {} : {
        update: mapExpression(operation.update, context, guards, "read", `${path}.update`),
      }),
      body: mapOperations(operation.body, context, guards, `${path}.body`),
      ...(operation.continuing === undefined ? {} : {
        continuing: mapOperations(operation.continuing, context, guards, `${path}.continuing`),
      }),
    };
    case "barrier":
    case "fence":
    case "break":
    case "continue": return operation;
    case "device-launch": return {
      ...operation,
      launch: {
        ...operation.launch,
        grid: operation.launch.grid.map((value, index) => mapExpression(value, context, guards, "read", `${path}.launch.grid[${index}]`)),
        block: operation.launch.block.map((value, index) => mapExpression(value, context, guards, "read", `${path}.launch.block[${index}]`)),
        args: operation.launch.args.map((value, index) => mapExpression(value, context, guards, "unsupported", `${path}.launch.args[${index}]`)),
      },
    };
    case "inline-asm": return {
      ...operation,
      outputs: operation.outputs.map((value, index) => mapExpression(value, context, guards, "unsupported", `${path}.outputs[${index}]`)),
      inputs: operation.inputs.map((value, index) => mapExpression(value, context, guards, "unsupported", `${path}.inputs[${index}]`)),
    };
    case "return": return operation.value === undefined
      ? operation
      : { ...operation, value: mapExpression(operation.value, context, guards, "read", `${path}.value`) };
  }
}

function mapMemoryRef(
  ref: SemanticMemoryRef,
  context: LoweringContext,
  guards: readonly SemanticExpression[],
  mode: MemoryRefMode,
  path: string,
): SemanticMemoryRef {
  const binding = context.bindingsByParameterId.get(semanticIdKey(ref.baseId));
  if (binding === undefined) {
    return {
      ...ref,
      indices: ref.indices.map((index, offset) => mapExpression(index, context, guards, "read", `${path}.indices[${offset}]`)),
    };
  }
  if (mode !== "direct-read") unsupportedUse(binding, path, mode === "write" ? "write" : "non-direct memory operation");
  if (
    ref.addressSpace !== "storage" ||
    ref.valueType !== "float" ||
    ref.containerValueType !== "float" ||
    ref.indices.length !== 1 ||
    ref.fields.length !== 0 ||
    ref.pointerBaseIsScalarLane === true ||
    ref.pointerBaseUnitBytes !== undefined ||
    ref.packedByteLanes !== undefined
  ) {
    unsupportedUse(binding, path, "typed, fielded, vectorized, or multi-index access");
  }
  const flat = ref.indices[0];
  if (flat === undefined) throw new Error("internal: verified direct read lost its index");
  binding.uses += 1;
  return { ...ref, indices: [physicalIndexExpression(binding, flat, guards, `${path}.indices[0]`)] };
}

function mapMatrixRef<T extends SemanticMatrixTileRef>(
  ref: T,
  context: LoweringContext,
  guards: readonly SemanticExpression[],
  path: string,
): T {
  return {
    ...ref,
    indices: ref.indices.map((index, offset) => mapExpression(index, context, guards, "read", `${path}.indices[${offset}]`)),
  };
}

function mapExpression(
  expression: SemanticExpression,
  context: LoweringContext,
  guards: readonly SemanticExpression[],
  mode: ExpressionMode,
  path: string,
): SemanticExpression {
  if (expression.kind === "symbol") {
    const binding = context.bindingsByParameterId.get(semanticIdKey(expression.id));
    if (binding !== undefined) unsupportedUse(binding, path, mode === "read" ? "bare pointer use" : mode);
    return expression;
  }
  if (expression.kind === "pointer-valid") {
    const binding = context.bindingsByParameterId.get(semanticIdKey(expression.pointerId));
    if (binding !== undefined) unsupportedUse(binding, path, "pointer validity or nullability test");
    return expression;
  }
  if (expression.kind === "index") {
    if (expression.target.kind === "symbol") {
      const binding = context.bindingsByParameterId.get(semanticIdKey(expression.target.id));
      if (binding !== undefined) {
        if (mode !== "read") unsupportedUse(binding, path, mode === "write" ? "write" : "non-direct indexed use");
        binding.uses += 1;
        return {
          ...expression,
          index: physicalIndexExpression(binding, expression.index, guards, `${path}.index`),
        };
      }
    }
    const nested = boundParameterInExpression(expression.target, context);
    if (nested !== undefined) unsupportedUse(nested, `${path}.target`, "nested, cast, or rebased pointer indexing");
    return {
      ...expression,
      target: mapExpression(expression.target, context, guards, mode, `${path}.target`),
      index: mapExpression(expression.index, context, guards, "read", `${path}.index`),
    };
  }
  switch (expression.kind) {
    case "literal": return expression;
    case "member": return { ...expression, object: mapExpression(expression.object, context, guards, mode, `${path}.object`) };
    case "call": return {
      ...expression,
      callee: mapExpression(expression.callee, context, guards, "unsupported", `${path}.callee`),
      args: expression.args.map((arg, index) => mapExpression(arg, context, guards, mode, `${path}.args[${index}]`)),
    };
    case "texture-read": return {
      ...expression,
      texture: mapExpression(expression.texture, context, guards, "unsupported", `${path}.texture`),
      x: mapExpression(expression.x, context, guards, "read", `${path}.x`),
      y: mapExpression(expression.y, context, guards, "read", `${path}.y`),
      ...(expression.z === undefined ? {} : { z: mapExpression(expression.z, context, guards, "read", `${path}.z`) }),
    };
    case "surface-read": return {
      ...expression,
      surface: mapExpression(expression.surface, context, guards, "unsupported", `${path}.surface`),
      xBytes: mapExpression(expression.xBytes, context, guards, "read", `${path}.xBytes`),
      y: mapExpression(expression.y, context, guards, "read", `${path}.y`),
      ...(expression.z === undefined ? {} : { z: mapExpression(expression.z, context, guards, "read", `${path}.z`) }),
    };
    case "cast": return { ...expression, expression: mapExpression(expression.expression, context, guards, mode, `${path}.expression`) };
    case "unary": return { ...expression, argument: mapExpression(expression.argument, context, guards, mode, `${path}.argument`) };
    case "binary": return {
      ...expression,
      left: mapExpression(expression.left, context, guards, mode, `${path}.left`),
      right: mapExpression(expression.right, context, guards, mode, `${path}.right`),
    };
    case "conditional": return {
      ...expression,
      condition: mapExpression(expression.condition, context, guards, "read", `${path}.condition`),
      consequent: mapExpression(expression.consequent, context, guards, mode, `${path}.consequent`),
      alternate: mapExpression(expression.alternate, context, guards, mode, `${path}.alternate`),
    };
    case "assignment": return {
      ...expression,
      target: mapExpression(expression.target, context, guards, "write", `${path}.target`),
      value: mapExpression(expression.value, context, guards, "read", `${path}.value`),
    };
    case "update": return { ...expression, argument: mapExpression(expression.argument, context, guards, "write", `${path}.argument`) };
    case "initializer": return {
      ...expression,
      elements: expression.elements.map((item, index) => mapExpression(item, context, guards, mode, `${path}.elements[${index}]`)),
    };
    case "sequence": return {
      ...expression,
      expressions: expression.expressions.map((item, index) => mapExpression(item, context, guards, mode, `${path}.expressions[${index}]`)),
    };
  }
}

function physicalIndexExpression(
  binding: BindingContext,
  flat: SemanticExpression,
  guards: readonly SemanticExpression[],
  path: string,
): SemanticExpression {
  if (flat.kind !== "symbol" || flat.valueType !== "uint") {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX",
      path,
      "flat logical index must be one unmodified uint symbol",
    );
  }
  if (
    binding.mutatedSymbols.has(semanticIdKey(flat.id)) ||
    binding.mutatedSymbols.has(semanticIdKey(semanticMemoryIdFromSymbol(flat.id)))
  ) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX",
      path,
      `logical index ${flat.name} is mutated and cannot carry a stable dominance proof`,
    );
  }
  if (!guards.some((guard) => containsLogicalDomainGuard(guard, flat, binding.logicalElementCount))) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-MISSING-GUARD",
      path,
      `direct read must be dominated by ${flat.name} < ${binding.logicalElementCount}u`,
    );
  }
  const coordinates = unflattenLogicalIndex(flat, binding.logicalShape);
  let location = lowerIndexExpression(binding.record.indexMap.location, coordinates, binding.dimensionValues, path);
  const byteOffset = BigInt(binding.record.summary.viewByteOffset);
  if (binding.record.summary.locationUnit === "element") {
    const elementOffset = byteOffset / BigInt(binding.record.summary.dtypeBytes);
    if (elementOffset !== 0n) location = uintBinary("+", uintLiteral(elementOffset, flat.span), location, flat.span);
    return location;
  }
  if (byteOffset !== 0n) location = uintBinary("+", uintLiteral(byteOffset, flat.span), location, flat.span);
  return uintCast(
    uintBinary("/", location, uintLiteral(BigInt(binding.record.summary.dtypeBytes), flat.span), flat.span),
    flat.span,
  );
}

function unflattenLogicalIndex(flat: SemanticExpression, shape: readonly bigint[]): readonly SemanticExpression[] {
  return shape.map((extent, axis) => {
    const stride = shape.slice(axis + 1).reduce((product, value) => product * value, 1n);
    let coordinate = stride === 1n
      ? flat
      : uintCast(uintBinary("/", flat, uintLiteral(stride, flat.span), flat.span), flat.span);
    if (axis > 0 && extent > 1n) coordinate = uintBinary("%", coordinate, uintLiteral(extent, flat.span), flat.span);
    if (extent === 1n) coordinate = uintLiteral(0n, flat.span);
    return coordinate;
  });
}

function lowerIndexExpression(
  expression: IndexExpr,
  coordinates: readonly SemanticExpression[],
  dimensions: ReadonlyMap<string, bigint>,
  path: string,
): SemanticExpression {
  switch (expression.kind) {
    case "const": return uintLiteral(requireU32(wireIntegerToBigInt(expression.value), path, "index constant"), coordinates[0]!.span);
    case "coordinate": {
      const coordinate = coordinates[expression.axis];
      if (coordinate === undefined) fail("BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX-MAP", path, `coordinate axis ${expression.axis} is unavailable`);
      return coordinate;
    }
    case "dimension": {
      const value = dimensions.get(expression.symbolId);
      if (value === undefined) fail("BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX-MAP", path, `dimension ${expression.symbolId} was not specialized`);
      return uintLiteral(requireU32(value, path, "dimension value"), coordinates[0]!.span);
    }
    case "add": {
      const terms = expression.terms.map((term, index) => lowerIndexExpression(term, coordinates, dimensions, `${path}.terms[${index}]`));
      const first = terms[0];
      if (first === undefined) throw new Error("internal: verified add expression is empty");
      return terms.slice(1).reduce((sum, term) => uintBinary("+", sum, term, sum.span), first);
    }
    case "mul": return uintBinary(
      "*",
      lowerIndexExpression(expression.lhs, coordinates, dimensions, `${path}.lhs`),
      lowerIndexExpression(expression.rhs, coordinates, dimensions, `${path}.rhs`),
      coordinates[0]!.span,
    );
    case "floorDiv":
    case "ceilDiv":
    case "mod":
    case "min":
    case "max":
      fail(
        "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX-MAP",
        path,
        `initial positive-affine compiler profile does not lower ${expression.kind}`,
      );
  }
}

function indexInterval(
  expression: IndexExpr,
  coordinates: readonly IntegerInterval[],
  dimensions: ReadonlyMap<string, bigint>,
  path: string,
): IntegerInterval {
  let interval: IntegerInterval;
  switch (expression.kind) {
    case "const": {
      const value = wireIntegerToBigInt(expression.value);
      interval = { minimum: value, maximum: value };
      break;
    }
    case "coordinate": {
      const value = coordinates[expression.axis];
      if (value === undefined) fail("BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX-MAP", path, `coordinate axis ${expression.axis} is unavailable`);
      interval = value;
      break;
    }
    case "dimension": {
      const value = dimensions.get(expression.symbolId);
      if (value === undefined) fail("BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX-MAP", path, `dimension ${expression.symbolId} was not specialized`);
      interval = { minimum: value, maximum: value };
      break;
    }
    case "add": {
      const values = expression.terms.map((term, index) => indexInterval(term, coordinates, dimensions, `${path}.terms[${index}]`));
      interval = values.reduce((sum, value) => ({
        minimum: sum.minimum + value.minimum,
        maximum: sum.maximum + value.maximum,
      }), { minimum: 0n, maximum: 0n });
      break;
    }
    case "mul": {
      const lhs = indexInterval(expression.lhs, coordinates, dimensions, `${path}.lhs`);
      const rhs = indexInterval(expression.rhs, coordinates, dimensions, `${path}.rhs`);
      if (lhs.minimum < 0n || rhs.minimum < 0n) {
        fail("BG-COMPILER-LAYOUT-BINDING-INTEGER-RANGE", path, "unsigned affine multiplication requires nonnegative operands");
      }
      interval = { minimum: lhs.minimum * rhs.minimum, maximum: lhs.maximum * rhs.maximum };
      break;
    }
    case "floorDiv":
    case "ceilDiv":
    case "mod":
    case "min":
    case "max":
      fail(
        "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX-MAP",
        path,
        `initial positive-affine compiler profile does not analyze ${expression.kind}`,
      );
  }
  if (interval.minimum < 0n || interval.maximum > U32_MAX) {
    fail(
      "BG-COMPILER-LAYOUT-BINDING-INTEGER-RANGE",
      path,
      `index interval [${interval.minimum}, ${interval.maximum}] is outside u32`,
    );
  }
  return interval;
}

function indexModulo(
  expression: IndexExpr,
  divisor: bigint,
  dimensions: ReadonlyMap<string, bigint>,
): bigint | undefined {
  switch (expression.kind) {
    case "const": return euclideanModulo(wireIntegerToBigInt(expression.value), divisor);
    case "dimension": {
      const value = dimensions.get(expression.symbolId);
      return value === undefined ? undefined : euclideanModulo(value, divisor);
    }
    case "coordinate": return divisor === 1n ? 0n : undefined;
    case "add": {
      let result = 0n;
      for (const term of expression.terms) {
        const value = indexModulo(term, divisor, dimensions);
        if (value === undefined) return undefined;
        result = euclideanModulo(result + value, divisor);
      }
      return result;
    }
    case "mul": {
      const lhs = indexModulo(expression.lhs, divisor, dimensions);
      const rhs = indexModulo(expression.rhs, divisor, dimensions);
      if (lhs === 0n || rhs === 0n) return 0n;
      return lhs === undefined || rhs === undefined ? undefined : euclideanModulo(lhs * rhs, divisor);
    }
    case "floorDiv":
    case "ceilDiv":
    case "mod":
    case "min":
    case "max": return undefined;
  }
}

function combineModulo(left: bigint, right: bigint | undefined, divisor: bigint): bigint | undefined {
  return right === undefined ? undefined : euclideanModulo(left + right, divisor);
}

function euclideanModulo(value: bigint, divisor: bigint): bigint {
  const remainder = value % divisor;
  return remainder < 0n ? remainder + divisor : remainder;
}

function containsLogicalDomainGuard(
  condition: SemanticExpression,
  flat: Extract<SemanticExpression, { readonly kind: "symbol" }>,
  elementCount: bigint,
): boolean {
  if (condition.kind !== "binary") return false;
  if (condition.operator === "&&") {
    return containsLogicalDomainGuard(condition.left, flat, elementCount) ||
      containsLogicalDomainGuard(condition.right, flat, elementCount);
  }
  if (condition.left.kind !== "symbol" || semanticIdKey(condition.left.id) !== semanticIdKey(flat.id)) return false;
  if (condition.right.kind !== "literal" || condition.right.literalKind !== "number") return false;
  if (!Number.isSafeInteger(condition.right.value) || BigInt(condition.right.value) < 0n) return false;
  const limit = BigInt(condition.right.value);
  return condition.operator === "<" && limit === elementCount ||
    condition.operator === "<=" && limit + 1n === elementCount;
}

function boundParameterInExpression(
  expression: SemanticExpression,
  context: LoweringContext,
): BindingContext | undefined {
  if (expression.kind === "symbol") return context.bindingsByParameterId.get(semanticIdKey(expression.id));
  if (expression.kind === "pointer-valid") return context.bindingsByParameterId.get(semanticIdKey(expression.pointerId));
  switch (expression.kind) {
    case "literal": return undefined;
    case "member": return boundParameterInExpression(expression.object, context);
    case "index": return boundParameterInExpression(expression.target, context) ?? boundParameterInExpression(expression.index, context);
    case "call": return boundParameterInExpression(expression.callee, context) ?? firstBound(expression.args, context);
    case "texture-read": return firstBound([expression.texture, expression.x, expression.y, ...(expression.z === undefined ? [] : [expression.z])], context);
    case "surface-read": return firstBound([expression.surface, expression.xBytes, expression.y, ...(expression.z === undefined ? [] : [expression.z])], context);
    case "cast": return boundParameterInExpression(expression.expression, context);
    case "unary": return boundParameterInExpression(expression.argument, context);
    case "binary": return boundParameterInExpression(expression.left, context) ?? boundParameterInExpression(expression.right, context);
    case "conditional": return firstBound([expression.condition, expression.consequent, expression.alternate], context);
    case "assignment": return boundParameterInExpression(expression.target, context) ?? boundParameterInExpression(expression.value, context);
    case "update": return boundParameterInExpression(expression.argument, context);
    case "initializer": return firstBound(expression.elements, context);
    case "sequence": return firstBound(expression.expressions, context);
  }
}

function firstBound(expressions: readonly SemanticExpression[], context: LoweringContext): BindingContext | undefined {
  for (const expression of expressions) {
    const binding = boundParameterInExpression(expression, context);
    if (binding !== undefined) return binding;
  }
  return undefined;
}

function uintLiteral(value: bigint, span: SourceSpan): SemanticExpression {
  return {
    kind: "literal",
    literalKind: "number",
    value: Number(requireU32(value, "$", "generated literal")),
    valueType: "uint",
    span,
  };
}

function uintBinary(
  operator: string,
  left: SemanticExpression,
  right: SemanticExpression,
  span: SourceSpan,
): SemanticExpression {
  return { kind: "binary", operator, left, right, valueType: "uint", span };
}

function uintCast(expression: SemanticExpression, span: SourceSpan): SemanticExpression {
  return { kind: "cast", valueType: "uint", pointer: false, expression, span };
}

function requireU32(value: bigint, path: string, name: string): bigint {
  if (value < 0n || value > U32_MAX) {
    fail("BG-COMPILER-LAYOUT-BINDING-INTEGER-RANGE", path, `${name} ${value} is outside u32`);
  }
  return value;
}

function unsupportedUse(binding: BindingContext, path: string, use: string): never {
  fail(
    "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-USE",
    path,
    `layout-bound parameter ${binding.parameter.name} does not support ${use}`,
  );
}

function fail(
  code: ConstructorParameters<typeof CudaLiteLayoutBindingError>[0],
  path: string,
  message: string,
  span?: SourceSpan,
): never {
  throw new CudaLiteLayoutBindingError(code, path, message, undefined, span);
}
