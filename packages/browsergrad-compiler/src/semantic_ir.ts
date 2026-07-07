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
import { CUDA_CACHE_HINT_LOADS, CUDA_CACHE_HINT_STORES } from "./intrinsics.js";

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
      readonly callee: "surf2Dread" | "surf2DLayeredread";
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
  | { readonly kind: "barrier"; readonly callee: string; readonly span: SourceSpan }
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

const DEFAULT_WORKGROUP_SIZE: KernelLaunch["blockDim"] = [256, 1, 1];
const COMPARISON_OPERATORS = new Set(["<", "<=", ">", ">=", "==", "!=", "&&", "||"]);
const POINTER_ORDER_OPERATORS = new Set(["<", "<=", ">", ">=", "==", "!="]);
const BARRIER_CALLS = new Set(["__syncthreads", "__syncwarp", "grid.sync", "cg::sync"]);
const ATOMIC_CALL_PREFIX = "atomic";
const TEXTURE_2D_READ_CALLS = new Set(["tex2D", "tex2DLod"]);
const SURFACE_WRITE_CALLS = new Set(["surf2Dwrite", "surf2DLayeredwrite"]);

export function createCudaLiteSemanticModel(analysis: CudaLiteAnalysis): CudaLiteSemanticModel {
  const params = analysis.kernel.params.map(symbolForParam);
  const constants = analysis.constants.map(symbolForConstant);
  const deviceGlobals = analysis.deviceGlobals.map(symbolForDeviceGlobal);
  const textures = analysis.textures.map(symbolForTexture);
  const globalScope = new Map([...params, ...constants, ...deviceGlobals, ...textures].map((symbol) => [symbol.name, symbol]));
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
  const scope = new Map(semantic.symbols.map((symbol) => [symbol.name, symbol]));
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
    case "asm":
      return { kind: "inline-asm", statement, span: statement.span };
    case "expr": {
      const aliasAssignment = localPointerAliasUpdate(statement.expression, scope);
      if (aliasAssignment) return { kind: "expression", expression: zeroExpression(statement.span), span: statement.span };
      const expression = lowerExpression(statement.expression, scope);
      if (expression.kind === "call" && expression.callee.kind === "symbol" && BARRIER_CALLS.has(expression.callee.name)) {
        return { kind: "barrier", callee: expression.callee.name, span: statement.span };
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
        if (SURFACE_WRITE_CALLS.has(expression.callee.name) && expression.args.length >= 4) {
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
          (expression.callee.name === "surf2DLayeredread" && expression.args.length === 5)
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
        ...optionalValueType(expressionValueType(target)),
        addressSpace: expressionAddressSpace(target),
        span: expression.span,
      };
    }
    case "call": {
      const args = expression.args.map((arg) => lowerExpression(arg, scope));
      if (expression.callee.kind === "identifier" && CUDA_CACHE_HINT_LOADS.has(expression.callee.name)) {
        const load = cacheHintLoadExpression(expression, scope);
        if (load) return load;
      }
      if (
        expression.callee.kind === "identifier" &&
        TEXTURE_2D_READ_CALLS.has(expression.callee.name) &&
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
          expression.callee.name === "surf2DLayeredread" && args.length === 4)
      ) {
        return {
          kind: "surface-read",
          callee: expression.callee.name as "surf2Dread" | "surf2DLayeredread",
          surface: args[0]!,
          xBytes: args[1]!,
          y: args[2]!,
          ...(args[3] === undefined ? {} : { z: args[3]! }),
          valueType: expression.templateValueType ?? "float",
          span: expression.span,
        };
      }
      return {
        kind: "call",
        callee: lowerExpression(expression.callee, scope),
        args,
        ...(expression.templateValueType === undefined ? {} : { templateValueType: expression.templateValueType }),
        ...optionalValueType(expression.templateValueType ?? expressionValueType(args[0])),
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
    statement.valueType !== "float" ||
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
  if (!target) return undefined;
  return semanticMathOutAssignmentStores(statement.expression.right, expression.value, target, scope, statement.span);
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

function isSincosCallName(name: string): boolean {
  return name === "sincos" || name === "sincosf" || name === "__sincosf" || name === "sincospi" || name === "sincospif";
}

function isSincosPiCallName(name: string): boolean {
  return name === "sincospi" || name === "sincospif";
}

function isModfCallName(name: string): boolean {
  return name === "modf" || name === "modff";
}

function isRemquoCallName(name: string): boolean {
  return name === "remquo" || name === "remquof";
}

function isFrexpCallName(name: string): boolean {
  return name === "frexp" || name === "frexpf";
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
  return {
    kind: "index",
    target: semanticSymbolExpression(root, span),
    index: alias.pointerBaseIndices[0]!,
    ...optionalValueType(root.valueType),
    addressSpace: root.addressSpace,
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
    indices: parts.indices,
    fields: parts.fields,
    span: expression.span,
  };
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
    return expression.name === "__builtin_assume_aligned" || expression.name === "ct::assume_aligned" ? expression.name : undefined;
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
  if (expression.target.kind !== "identifier") return undefined;
  const symbol = scope.get(expression.target.name);
  if (!symbol?.pointerRoot || !semanticPointerAliasAddressSpaceSupported(symbol.pointerAddressSpace) || !symbol.pointerBaseIndices || symbol.pointerBaseIndices.length !== 1) return undefined;
  const root = scope.get(symbol.pointerRoot);
  if (!root) return undefined;
  const target = semanticSymbolExpression(root, expression.target.span);
  const index = addIndexExpressions(symbol.pointerBaseIndices[0]!, lowerExpression(expression.index, scope), expression.index.span);
  return {
    kind: "index",
    target,
    index,
    ...optionalValueType(symbol.valueType ?? root.valueType),
    addressSpace: root.addressSpace,
    span: expression.span,
  };
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
    const stride = dimensions.slice(offset + 1).reduce((product, dimension) => product * dimension, 1);
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

function optionalValueType(valueType: CudaLiteScalarType | undefined): { readonly valueType?: CudaLiteScalarType } {
  return valueType === undefined ? {} : { valueType };
}

function numberLiteralType(raw: string): CudaLiteScalarType {
  return /[.eE]|f$/u.test(raw) ? "float" : raw.endsWith("u") || raw.endsWith("U") ? "uint" : "int";
}

function normalizeWorkgroupSize(value: readonly [number, number, number]): [number, number, number] {
  return [normalizeDimension(value[0]), normalizeDimension(value[1]), normalizeDimension(value[2])];
}

function normalizeDimension(value: number): number {
  return Number.isFinite(value) && Number.isInteger(value) && value > 0 ? value : 1;
}
