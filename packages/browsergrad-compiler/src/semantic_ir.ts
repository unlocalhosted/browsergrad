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
  readonly constant?: boolean;
  readonly dimensions: readonly number[];
  readonly addressSpace: SemanticAddressSpace;
  readonly span: SourceSpan;
}

export interface CudaLiteSemanticFunction {
  readonly name: string;
  readonly returnType: CudaLiteScalarType;
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
  readonly requiredFeatures: readonly string[];
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
const BARRIER_CALLS = new Set(["__syncthreads", "__syncwarp", "grid.sync", "cg::sync"]);

export function createCudaLiteSemanticModel(analysis: CudaLiteAnalysis): CudaLiteSemanticModel {
  const params = analysis.kernel.params.map(symbolForParam);
  const constants = analysis.constants.map(symbolForConstant);
  const deviceGlobals = analysis.deviceGlobals.map(symbolForDeviceGlobal);
  const textures = analysis.textures.map(symbolForTexture);
  const functions = analysis.functions.map(symbolForFunction);
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
  return {
    kind: "semantic-kernel-ir",
    name: analysis.kernel.name,
    span: analysis.kernel.span,
    params: semantic.params,
    memory: [
      ...semantic.symbols.filter((symbol) => symbol.kind !== "param" && symbol.kind !== "function"),
      ...localMemory,
    ],
    functions: semantic.functions,
    operations,
    requiredFeatures: semantic.requiredFeatures,
    workgroupSize: normalizeWorkgroupSize(options.workgroupSize ?? DEFAULT_WORKGROUP_SIZE),
  };
}

function lowerStatements(
  statements: readonly CudaLiteStatement[],
  parentScope: ReadonlyMap<string, CudaLiteSemanticSymbol>,
): readonly SemanticKernelIrOperation[] {
  const scope = new Map(parentScope);
  const out: SemanticKernelIrOperation[] = [];
  for (const statement of statements) {
    out.push(lowerStatement(statement, scope));
  }
  return out;
}

function lowerStatement(
  statement: CudaLiteStatement,
  scope: Map<string, CudaLiteSemanticSymbol>,
): SemanticKernelIrOperation {
  switch (statement.kind) {
    case "block":
      return { kind: "block", body: lowerStatements(statement.body, scope), span: statement.span };
    case "var": {
      const target = symbolForVar(statement);
      scope.set(target.name, target);
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
      const expression = lowerExpression(statement.expression, scope);
      if (expression.kind === "call" && expression.callee.kind === "symbol" && BARRIER_CALLS.has(expression.callee.name)) {
        return { kind: "barrier", callee: expression.callee.name, span: statement.span };
      }
      return { kind: "expression", expression, span: statement.span };
    }
    case "if":
      return {
        kind: "branch",
        condition: lowerExpression(statement.condition, scope),
        consequent: lowerStatements(statement.consequent, scope),
        alternate: lowerStatements(statement.alternate ?? [], scope),
        span: statement.span,
      };
    case "for":
      return {
        kind: "loop",
        loopKind: "for",
        ...(statement.init === undefined ? {} : { init: statement.init.kind === "var" ? lowerStatement(statement.init, new Map(scope)) : lowerExpression(statement.init, scope) }),
        ...(statement.condition === undefined ? {} : { condition: lowerExpression(statement.condition, scope) }),
        ...(statement.update === undefined ? {} : { update: lowerExpression(statement.update, scope) }),
        body: lowerStatements(statement.body, scope),
        span: statement.span,
      };
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
      const argument = lowerExpression(expression.argument, scope);
      return { kind: "unary", operator: expression.operator, argument, ...optionalValueType(expression.operator === "&" ? "voidptr" : expressionValueType(argument)), span: expression.span };
    }
    case "binary": {
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

function symbolForFunction(fn: CudaLiteDeviceFunction): CudaLiteSemanticFunction {
  return {
    name: fn.name,
    returnType: fn.returnType,
    params: fn.params.map(symbolForParam),
    span: fn.span,
  };
}

function symbolForVar(statement: CudaLiteVarDecl): CudaLiteSemanticSymbol {
  return {
    name: statement.name,
    kind: statement.storage === "shared" ? "shared" : "local",
    valueType: statement.valueType,
    pointer: statement.pointer,
    constant: false,
    dimensions: statement.dimensions,
    addressSpace: statement.storage,
    span: statement.span,
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
