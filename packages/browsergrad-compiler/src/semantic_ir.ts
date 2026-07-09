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
import {
  cudaLiteDimensionStride as dimensionStride,
  cudaLiteTotalElements as totalElements,
} from "./cuda_lite_values.js";
import { cudaBfloat16IntrinsicReturnType } from "./cuda_bfloat16_intrinsics.js";
import {
  isCudaSemanticSurfaceWriteCallName,
  isCudaTexture2DReadCallName,
} from "./cuda_texture_surface_calls.js";
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
  CUDA_BARRIER_CALL_NAMES,
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
import { classifyInlineAsm, type PtxSpecialU32Register } from "./features/inline_ptx/model.js";
import { alignofCudaType, sizeofCudaType } from "./type_layout.js";
import { cudaVectorLaneCount, cudaVectorScalarType, cudaVectorSwizzleType, isCudaVectorType } from "./vector_types.js";
import { SEMANTIC_LOCAL_ARRAY_FILL_CALLS } from "./semantic_builtin_calls.js";
import { SEMANTIC_CURAND_CALLS } from "./semantic_curand_intrinsics.js";

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
  readonly pointerAddressSpace?: SemanticAddressSpace;
  readonly pointerBaseIndices?: readonly SemanticExpression[];
  readonly pointerValid?: SemanticExpression;
  readonly constant?: boolean;
  readonly initialized?: boolean;
  readonly init?: SemanticExpression;
  readonly dimensions: readonly number[];
  readonly addressSpace: SemanticAddressSpace;
  readonly span: SourceSpan;
}

export interface CudaLiteSemanticFunction {
  readonly name: string;
  readonly returnType: CudaLiteScalarType;
  readonly params: readonly CudaLiteSemanticSymbol[];
  readonly body: readonly SemanticKernelIrOperation[];
  readonly span: SourceSpan;
}

export interface CudaLiteSemanticModel {
  readonly kind: "cuda-lite-semantic-model";
  readonly kernelName: string;
  readonly span: SourceSpan;
  readonly params: readonly CudaLiteSemanticSymbol[];
  readonly symbols: readonly CudaLiteSemanticSymbol[];
  readonly functions: readonly CudaLiteSemanticFunction[];
  readonly requiredFeatures: readonly string[];
}

export interface SemanticMemoryRef {
  readonly base: string;
  readonly addressSpace: SemanticAddressSpace;
  readonly valueType?: CudaLiteScalarType;
  readonly containerValueType?: CudaLiteScalarType;
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
      readonly callee: "tex2D" | "tex2DLod";
      readonly texture: SemanticExpression;
      readonly x: SemanticExpression;
      readonly y: SemanticExpression;
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
  | { readonly kind: "cooperative-group-declare"; readonly declaration: CudaLiteCooperativeGroupDecl; readonly span: SourceSpan }
  | { readonly kind: "load"; readonly source: SemanticMemoryRef; readonly span: SourceSpan }
  | { readonly kind: "store"; readonly target: SemanticMemoryRef; readonly value: SemanticExpression; readonly operator: string; readonly reads: readonly SemanticMemoryRef[]; readonly span: SourceSpan }
  | { readonly kind: "surface-write"; readonly surface: SemanticExpression; readonly value: SemanticExpression; readonly xBytes: SemanticExpression; readonly y: SemanticExpression; readonly z?: SemanticExpression; readonly span: SourceSpan }
  | { readonly kind: "surface-read-store"; readonly target: SemanticExpression; readonly surface: SemanticExpression; readonly xBytes: SemanticExpression; readonly y: SemanticExpression; readonly z?: SemanticExpression; readonly valueType?: CudaLiteScalarType; readonly span: SourceSpan }
  | { readonly kind: "atomic"; readonly callee: string; readonly target?: SemanticMemoryRef; readonly args: readonly SemanticExpression[]; readonly span: SourceSpan }
  | { readonly kind: "call"; readonly callee: string; readonly args: readonly SemanticExpression[]; readonly reads: readonly SemanticMemoryRef[]; readonly span: SourceSpan }
  | { readonly kind: "expression"; readonly expression: SemanticExpression; readonly span: SourceSpan }
  | { readonly kind: "branch"; readonly condition: SemanticExpression; readonly consequent: readonly SemanticKernelIrOperation[]; readonly alternate: readonly SemanticKernelIrOperation[]; readonly span: SourceSpan }
  | { readonly kind: "loop"; readonly loopKind: "for" | "while" | "do-while"; readonly init?: SemanticKernelIrOperation | SemanticExpression; readonly condition?: SemanticExpression; readonly update?: SemanticExpression; readonly body: readonly SemanticKernelIrOperation[]; readonly span: SourceSpan }
  | { readonly kind: "barrier"; readonly callee: string; readonly groupName?: string; readonly span: SourceSpan }
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
    case "load":
      walkSemanticMemoryRef(operation.source, visitExpression);
      return;
    case "store":
      walkSemanticMemoryRef(operation.target, visitExpression);
      walkSemanticExpression(operation.value, visitExpression);
      for (const read of operation.reads) walkSemanticMemoryRef(read, visitExpression);
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
    case "cooperative-group-declare":
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
const BARRIER_CALLS: ReadonlySet<string> = new Set([...CUDA_BARRIER_CALL_NAMES, "grid.sync", "cg::sync"]);
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
  return {
    kind: "cuda-lite-semantic-model",
    kernelName: analysis.kernel.name,
    span: analysis.kernel.span,
    params,
    symbols: [...params, ...constants, ...deviceGlobals, ...textures],
    functions,
    requiredFeatures: analysis.requiredFeatures,
  };
}

export function lowerSemanticModelToKernelIr(
  analysis: CudaLiteAnalysis,
  semantic: CudaLiteSemanticModel,
  options: { readonly workgroupSize?: readonly [number, number, number] } = {},
): SemanticKernelIrModule {
  const functionSymbols = semantic.functions.map(symbolForSemanticFunctionDeclaration);
  const scope = new Map([...semantic.symbols, ...functionSymbols].map((symbol) => [symbol.name, symbol]));
  const operations = lowerStatements(analysis.kernel.body, scope);
  const localMemory = collectDeclaredMemory(operations);
  const reachable = collectReachableAnalysisNames(analysis);
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
      ),
      ...localMemory,
    ],
    functions: semantic.functions.filter((fn) => reachable.functionNames.has(fn.name)),
    operations,
    requiredFeatures: semantic.requiredFeatures,
    workgroupSize: normalizeWorkgroupSize(options.workgroupSize ?? DEFAULT_WORKGROUP_SIZE),
  };
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
  return out;
}

function lowerStatementOperations(
  statement: CudaLiteStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] {
  const mathOutVarDecl = semanticMathOutVarDeclOperations(statement, scope);
  const mathOutAssignment = semanticMathOutAssignmentOperations(statement, scope);
  const mathOutCall = semanticMathOutCallStatementOperations(statement, scope);
  return mathOutVarDecl ?? mathOutAssignment ?? mathOutCall ?? [lowerStatement(statement, scope)];
}

function lowerInlineAsmSpecialRegisterAssignment(
  statement: CudaLiteAsmStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
): SemanticKernelIrOperation | undefined {
  const op = classifyInlineAsm(statement.template);
  const outputs = statement.outputs ?? (statement.output === undefined ? [] : [statement.output]);
  if (op?.kind !== "special-register-u32" || statement.inputs.length !== 0 || outputs.length !== 1) return undefined;
  const target = lowerExpression(outputs[0]!, scope);
  return {
    kind: "expression",
    expression: {
      kind: "assignment",
      operator: "=",
      target,
      value: semanticSpecialRegisterExpression(op.register, statement.span),
      valueType: expressionValueType(target) ?? "uint",
      span: statement.span,
    },
    span: statement.span,
  };
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
      return { kind: "cooperative-group-declare", declaration: statement, span: statement.span };
    case "kernel-launch":
      return { kind: "device-launch", launch: lowerDeviceLaunch(statement, scope), span: statement.span };
    case "asm": {
      const registerAssignment = lowerInlineAsmSpecialRegisterAssignment(statement, scope);
      if (registerAssignment) return registerAssignment;
      return { kind: "inline-asm", statement, span: statement.span };
    }
    case "expr": {
      const aliasAssignment = localPointerAliasUpdate(statement.expression, scope);
      if (aliasAssignment) return { kind: "expression", expression: zeroExpression(statement.span), span: statement.span };
      const expression = lowerExpression(statement.expression, scope);
      if (expression.kind === "call" && expression.callee.kind === "symbol" && BARRIER_CALLS.has(expression.callee.name)) {
        const groupName = barrierGroupName(expression);
        return {
          kind: "barrier",
          callee: expression.callee.name,
          ...(groupName === undefined ? {} : { groupName }),
          span: statement.span,
        };
      }
      if (expression.kind === "call" && expression.callee.kind === "symbol" && FENCE_CALLS.has(expression.callee.name)) {
        return { kind: "fence", callee: expression.callee.name, span: statement.span };
      }
      if (expression.kind === "assignment") {
        if (statement.expression.kind === "assignment") {
          const mathOutAssignment = semanticMathOutAssignmentBlock(statement.expression, expression, scope, statement.span);
          if (mathOutAssignment) return mathOutAssignment;
        }
        const target = memoryRefFromExpression(expression.target);
        if (target) {
          return {
            kind: "store",
            target,
            value: expression.value,
            operator: expression.operator,
            reads: collectMemoryRefs(expression.value),
            span: statement.span,
          };
        }
      }
      if (expression.kind === "call" && expression.callee.kind === "symbol") {
        if (isCudaSemanticSurfaceWriteCallName(expression.callee.name) && expression.args.length >= 4) {
          return {
            kind: "surface-write",
            value: expression.args[0]!,
            surface: expression.args[1]!,
            xBytes: expression.args[2]!,
            y: expression.args[3]!,
            ...(expression.args[4] === undefined ? {} : { z: expression.args[4]! }),
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
      return {
        kind: "symbol",
        name: expression.name,
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
      const preservePointerArgs = expression.callee.kind === "identifier" &&
        (SEMANTIC_LOCAL_ARRAY_FILL_CALLS.has(expression.callee.name) || SEMANTIC_CURAND_CALLS.has(expression.callee.name));
      const args = expression.args.map((arg) => preservePointerArgs
        ? lowerExpression(arg, scope)
        : pointerAliasValueExpression(arg, scope, arg.span) ?? lowerExpression(arg, scope));
      if (expression.callee.kind === "identifier" && CUDA_CACHE_HINT_LOADS.has(expression.callee.name)) {
        const load = cacheHintLoadExpression(expression, scope);
        if (load) return load;
      }
      if (
        expression.callee.kind === "identifier" &&
        isCudaTexture2DReadCallName(expression.callee.name) &&
        args.length >= 3
      ) {
        return {
          kind: "texture-read",
          callee: expression.callee.name as "tex2D" | "tex2DLod",
          texture: args[0]!,
          x: args[1]!,
          y: args[2]!,
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
            : expression.templateValueType ?? semanticIntrinsicReturnType(expression.callee.kind === "identifier" ? expression.callee.name : undefined, args) ?? expressionValueType(callee) ?? expressionValueType(args[0])),
        span: expression.span,
      };
    }
    case "cast":
      return {
        kind: "cast",
        valueType: expression.valueType,
        pointer: expression.pointer ?? false,
        expression: lowerExpression(expression.expression, scope),
        span: expression.span,
      };
    case "unary": {
      const aliased = expression.operator === "*" ? localPointerAliasDerefExpression(expression.argument, scope, expression.span) : undefined;
      if (aliased) return aliased;
      const argument = lowerExpression(expression.argument, scope);
      return { kind: "unary", operator: expression.operator, argument, ...optionalValueType(expression.operator === "&" ? "voidptr" : expressionValueType(argument)), span: expression.span };
    }
    case "binary": {
      const pointerDifference = localPointerAliasDifferenceExpression(expression, scope);
      if (pointerDifference) return pointerDifference;
      const pointerComparison = localPointerAliasComparisonExpression(expression, scope);
      if (pointerComparison) return pointerComparison;
      const left = lowerExpression(expression.left, scope);
      const right = lowerExpression(expression.right, scope);
      return { kind: "binary", operator: expression.operator, left, right, ...optionalValueType(COMPARISON_OPERATORS.has(expression.operator) ? "bool" : expressionValueType(left) ?? expressionValueType(right)), span: expression.span };
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
  const exponentTarget = source.args[1] === undefined ? undefined : pointerAliasValueExpression(source.args[1], scope, source.args[1].span);
  if (value === undefined || !exponentTarget || !semanticExpressionSideEffectFree(value)) return undefined;
  const exponentRef = memoryRefFromExpression(exponentTarget);
  if (!exponentRef) return undefined;
  const temp = tempScalarSymbol("__bg.frexp.value", span, "float");
  const tempValue = semanticSymbolExpression(temp, value.span);
  return {
    sideEffects: [
      { kind: "declare", target: temp, init: value, span },
      storeOperation(exponentRef, unaryIntCallExpression("__bg_frexp_exponent", tempValue, expression.span), span),
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
  const target = source.args[1] === undefined ? undefined : pointerAliasValueExpression(source.args[1], scope, source.args[1].span);
  if (value === undefined || !target) return undefined;
  const targetRef = memoryRefFromExpression(target);
  if (!targetRef) return undefined;
  const exponent = frexpExponentForFiniteNumber(value);
  return {
    kind: "store",
    target: targetRef,
    value: intNumberExpression(exponent, expression.span),
    operator: "=",
    reads: [],
    span,
  };
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
        semanticExpressionSideEffectFree(expression.y);
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
    span,
  };
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
    indices: parts.indices,
    fields: parts.fields,
    span: expression.span,
  };
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
  const params = fn.params.map(symbolForFunctionParam);
  for (const param of params) scope.set(param.name, param);
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

function symbolForFunctionParam(param: CudaLiteParam): CudaLiteSemanticSymbol {
  const symbol = symbolForParam(param);
  if (symbol.addressSpace === "texture" || symbol.addressSpace === "surface") return symbol;
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
): Pick<CudaLiteSemanticSymbol, "pointerRoot" | "pointerAddressSpace" | "pointerBaseIndices" | "pointerValid"> | undefined {
  if (!expression) return undefined;
  if (expression.kind === "cast" && expression.pointer) return localPointerAliasForInitializer(expression.expression, scope);
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
      alternate.pointerBaseIndices?.length !== 1
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
    };
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const left = localPointerAliasForInitializer(expression.left, scope);
    if (left?.pointerRoot && semanticPointerAliasAddressSpaceSupported(left.pointerAddressSpace) && left.pointerBaseIndices?.length === 1) {
      const right = lowerExpression(expression.right, scope);
      return {
        pointerRoot: left.pointerRoot,
        pointerAddressSpace: left.pointerAddressSpace,
        pointerBaseIndices: [expression.operator === "+"
          ? addIndexExpressions(left.pointerBaseIndices[0]!, right, expression.span)
          : subtractIndexExpressions(left.pointerBaseIndices[0]!, right, expression.span)],
      };
    }
    if (expression.operator === "+") {
      const right = localPointerAliasForInitializer(expression.right, scope);
      if (right?.pointerRoot && semanticPointerAliasAddressSpaceSupported(right.pointerAddressSpace) && right.pointerBaseIndices?.length === 1) {
        const leftOffset = lowerExpression(expression.left, scope);
        return {
          pointerRoot: right.pointerRoot,
          pointerAddressSpace: right.pointerAddressSpace,
          pointerBaseIndices: [addIndexExpressions(right.pointerBaseIndices[0]!, leftOffset, expression.span)],
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
      };
    }
    if (root?.kind === "param" && root.pointer && root.addressSpace === "storage") {
      return {
        pointerRoot: root.name,
        pointerAddressSpace: root.addressSpace,
        pointerBaseIndices: [zeroExpression(expression.span)],
      };
    }
    if (!root || root.kind !== "local" || root.dimensions.length !== 1 || root.pointer) return undefined;
    return {
      pointerRoot: root.name,
      pointerAddressSpace: root.addressSpace,
      pointerBaseIndices: [zeroExpression(expression.span)],
    };
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

function semanticPointerAliasAddressSpaceSupported(addressSpace: SemanticAddressSpace | undefined): addressSpace is "local" | "storage" {
  return addressSpace === "local" || addressSpace === "storage";
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
    if (statement.kind !== "expr") continue;
    const expression = statement.expression;
    if (expression.kind !== "assignment" || expression.operator !== "=" || expression.left.kind !== "identifier" || expression.left.name !== name) continue;
    const alias = localPointerAliasForInitializer(expression.right, scope);
    return alias !== undefined && semanticPointerAliasAddressSpaceSupported(alias.pointerAddressSpace);
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
    const one = { kind: "literal" as const, literalKind: "number" as const, value: 1, valueType: "int" as const, span: expression.span };
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
    const delta = lowerExpression(expression.right, scope);
    const index = expression.operator === "+="
      ? addIndexExpressions(target.pointerBaseIndices[0]!, delta, expression.span)
      : subtractIndexExpressions(target.pointerBaseIndices[0]!, delta, expression.span);
    scope.set(target.name, { ...target, pointerBaseIndices: [index] });
    return true;
  }
  return false;
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
  const ref = localArrayRefFromExpression(expression, scope);
  if (ref) {
    if (ref.root.dimensions.length === 0) return undefined;
    return { root: ref.root, indices: [flatIndexExpressionForDimensions(ref.root.dimensions, ref.indices, expression.span)] };
  }
  if (expression.kind !== "index" || expression.target.kind !== "identifier") return undefined;
  const target = scope.get(expression.target.name);
  if (target?.kind === "param" && target.pointer && target.addressSpace === "storage") {
    return {
      root: target,
      indices: [lowerExpression(expression.index, scope)],
    };
  }
  if (target?.pointerRoot && semanticPointerAliasAddressSpaceSupported(target.pointerAddressSpace) && target.pointerBaseIndices?.length === 1) {
    const root = scope.get(target.pointerRoot);
    if (!root || !semanticPointerAliasAddressSpaceSupported(root.addressSpace)) return undefined;
    return {
      root,
      indices: [addIndexExpressions(target.pointerBaseIndices[0]!, lowerExpression(expression.index, scope), expression.span)],
    };
  }
  return undefined;
}

function localArrayRefFromExpression(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): { readonly root: CudaLiteSemanticSymbol; readonly indices: readonly SemanticExpression[] } | undefined {
  if (expression.kind !== "index") return undefined;
  if (expression.target.kind === "identifier") {
    const root = scope.get(expression.target.name);
    if (!root || root.kind !== "local" || root.dimensions.length === 0 || root.pointer) return undefined;
    return { root, indices: [lowerExpression(expression.index, scope)] };
  }
  const target = localArrayRefFromExpression(expression.target, scope);
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
    pointerAliasElementOffset(aliasValueType, root.valueType, lowerExpression(expression.index, scope), expression.index.span),
    expression.index.span,
  );
  return {
    kind: "index",
    target,
    index,
    ...optionalValueType(aliasValueType),
    addressSpace: root.addressSpace,
    span: expression.span,
  };
}

function pointerAliasTargetValueType(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): CudaLiteScalarType | undefined {
  if (expression.kind === "cast" && expression.pointer) return expression.valueType;
  if (expression.kind === "identifier") return scope.get(expression.name)?.valueType;
  return undefined;
}

function pointerAliasElementOffset(
  aliasType: CudaLiteScalarType | undefined,
  rootType: CudaLiteScalarType | undefined,
  index: SemanticExpression,
  span: SourceSpan,
): SemanticExpression {
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
  if (expression.kind !== "identifier") return undefined;
  const symbol = scope.get(expression.name);
  if (symbol?.kind === "param" && symbol.pointer && semanticPointerAliasAddressSpaceSupported(symbol.addressSpace)) {
    return {
      kind: "index",
      target: semanticSymbolExpression(symbol, expression.span),
      index: zeroExpression(expression.span),
      ...optionalValueType(symbol.valueType),
      addressSpace: symbol.addressSpace,
      span,
    };
  }
  if (!symbol?.pointerRoot || !semanticPointerAliasAddressSpaceSupported(symbol.pointerAddressSpace) || !symbol.pointerBaseIndices || symbol.pointerBaseIndices.length !== 1) return undefined;
  const root = scope.get(symbol.pointerRoot);
  if (!root) return undefined;
  return {
    kind: "index",
    target: semanticSymbolExpression(root, expression.span),
    index: symbol.pointerBaseIndices[0]!,
    ...optionalValueType(symbol.valueType ?? root.valueType),
    addressSpace: root.addressSpace,
    span,
  };
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
  const zero = zeroExpression(span);
  return {
    kind: "binary",
    operator: value ? "==" : "!=",
    left: zero,
    right: zero,
    valueType: "bool",
    span,
  };
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
  if (property === "x" || property === "y" || property === "z") {
    if (object.kind === "symbol" && (object.name === "threadIdx" || object.name === "blockIdx" || object.name === "blockDim" || object.name === "gridDim")) return "uint";
  }
  const objectType = expressionValueType(object);
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

function expressionValueType(expression: SemanticExpression | undefined): CudaLiteScalarType | undefined {
  if (!expression || expression.kind === "initializer") return undefined;
  return "valueType" in expression ? expression.valueType : undefined;
}

function semanticIntrinsicReturnType(name: string | undefined, args: readonly SemanticExpression[]): CudaLiteScalarType | undefined {
  if (name === undefined) return undefined;
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
  if (/^0x/iu.test(raw)) return /(?:[uU][lL]*|[lL]+[uU][lL]*)$/u.test(raw) ? "uint" : "int";
  return /[.eE]|[fF]$/u.test(raw) ? "float" : /(?:[uU][lL]*|[lL]+[uU][lL]*)$/u.test(raw) ? "uint" : "int";
}

function normalizeWorkgroupSize(value: readonly [number, number, number]): [number, number, number] {
  return [normalizeDimension(value[0]), normalizeDimension(value[1]), normalizeDimension(value[2])];
}

function normalizeDimension(value: number): number {
  return Number.isFinite(value) && Number.isInteger(value) && value > 0 ? value : 1;
}
