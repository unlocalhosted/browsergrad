import type {
  CudaLiteAnalysis,
  CudaLiteAsmStatement,
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
  readonly constant?: boolean;
  readonly initialized?: boolean;
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
    out.push(lowerStatement(statement, scope));
  }
  return out;
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
      if (target.pointerRoot && target.pointerAddressSpace === "local" && target.pointerBaseIndices) {
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
          ? lowerStatement(statement.init, loopScope)
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
): Pick<CudaLiteSemanticSymbol, "pointerRoot" | "pointerAddressSpace" | "pointerBaseIndices"> | undefined {
  if (!expression) return undefined;
  if (expression.kind === "cast" && expression.pointer) return localPointerAliasForInitializer(expression.expression, scope);
  if (expression.kind === "conditional") {
    const consequent = localPointerAliasForInitializer(expression.consequent, scope);
    const alternate = localPointerAliasForInitializer(expression.alternate, scope);
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
        condition: lowerExpression(expression.condition, scope),
        consequent: consequent.pointerBaseIndices[0]!,
        alternate: alternate.pointerBaseIndices[0]!,
        valueType: "int",
        span: expression.span,
      }],
    };
  }
  if (expression.kind === "identifier") {
    const root = scope.get(expression.name);
    if (root?.pointerRoot && root.pointerAddressSpace === "local" && root.pointerBaseIndices?.length === 1) {
      return {
        pointerRoot: root.pointerRoot,
        pointerAddressSpace: root.pointerAddressSpace,
        pointerBaseIndices: root.pointerBaseIndices,
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
  const ref = localPointerAliasRoot(expression.argument, scope);
  if (!ref || ref.root.addressSpace !== "local" || ref.root.dimensions.length !== 1) return undefined;
  return {
    pointerRoot: ref.root.name,
    pointerAddressSpace: ref.root.addressSpace,
    pointerBaseIndices: ref.indices,
  };
}

function isLocalPointerAliasPlaceholder(statement: CudaLiteStatement): statement is Extract<CudaLiteStatement, { readonly kind: "var" }> {
  return statement.kind === "var" &&
    statement.storage === "local" &&
    statement.pointer &&
    statement.dimensions.length === 0 &&
    statement.init === undefined;
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
    return alias !== undefined && alias.pointerAddressSpace === "local";
  }
  return false;
}

function localPointerAliasUpdate(
  expression: CudaLiteExpression,
  scope: Map<string, CudaLiteSemanticSymbol>,
): boolean {
  if (expression.kind !== "assignment" || expression.left.kind !== "identifier") return false;
  const target = scope.get(expression.left.name);
  if (!target || target.kind !== "local" || !target.pointer || target.dimensions.length > 0) return false;
  if (expression.operator === "=") {
    const alias = localPointerAliasForInitializer(expression.right, scope);
    if (!alias || alias.pointerAddressSpace !== "local") return false;
    scope.set(target.name, { ...target, ...alias });
    return true;
  }
  if ((expression.operator === "+=" || expression.operator === "-=") && target.pointerRoot && target.pointerAddressSpace === "local" && target.pointerBaseIndices?.length === 1) {
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
      next.pointerAddressSpace !== "local" ||
      next.pointerBaseIndices?.length !== 1
    ) {
      continue;
    }
    parent.set(name, {
      ...current,
      pointerRoot: next.pointerRoot,
      pointerAddressSpace: "local",
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
      left.pointerAddressSpace !== "local" ||
      right.pointerAddressSpace !== "local" ||
      left.pointerBaseIndices?.length !== 1 ||
      right.pointerBaseIndices?.length !== 1
    ) {
      continue;
    }
    parent.set(name, {
      ...current,
      pointerRoot: left.pointerRoot,
      pointerAddressSpace: "local",
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
  if (expression.kind !== "index" || expression.target.kind !== "identifier") return undefined;
  const root = scope.get(expression.target.name);
  if (!root || root.kind !== "local" || root.dimensions.length !== 1 || root.pointer) return undefined;
  return { root, indices: [lowerExpression(expression.index, scope)] };
}

function localPointerAliasIndexExpression(
  expression: Extract<CudaLiteExpression, { readonly kind: "index" }>,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): SemanticExpression | undefined {
  if (expression.target.kind !== "identifier") return undefined;
  const symbol = scope.get(expression.target.name);
  if (!symbol?.pointerRoot || symbol.pointerAddressSpace !== "local" || !symbol.pointerBaseIndices || symbol.pointerBaseIndices.length !== 1) return undefined;
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
  if (!symbol?.pointerRoot || symbol.pointerAddressSpace !== "local" || !symbol.pointerBaseIndices || symbol.pointerBaseIndices.length !== 1) return undefined;
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
  const left = localPointerAliasScalarIndex(expression.left, scope);
  const right = localPointerAliasScalarIndex(expression.right, scope);
  if (!left || !right || left.root !== right.root) return undefined;
  return {
    kind: "binary",
    operator: expression.operator,
    left: left.index,
    right: right.index,
    valueType: "bool",
    span: expression.span,
  };
}

function localPointerAliasScalarIndex(
  expression: CudaLiteExpression,
  scope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): { readonly root: string; readonly index: SemanticExpression } | undefined {
  if (expression.kind !== "identifier") return undefined;
  const symbol = scope.get(expression.name);
  if (!symbol?.pointerRoot || symbol.pointerAddressSpace !== "local" || !symbol.pointerBaseIndices || symbol.pointerBaseIndices.length !== 1) return undefined;
  return { root: symbol.pointerRoot, index: symbol.pointerBaseIndices[0]! };
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

function isZeroLiteral(expression: SemanticExpression): boolean {
  return expression.kind === "literal" && expression.literalKind === "number" && expression.value === 0;
}

function zeroExpression(span: SourceSpan): SemanticExpression {
  return { kind: "literal", literalKind: "number", value: 0, valueType: "int", span };
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
