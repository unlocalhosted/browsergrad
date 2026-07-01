import { walkCudaLiteExpressions } from "./ast_queries.js";
import { expressionName, rootIdentifier } from "./analyzer.js";
import { matrixTileStorageDimensions } from "./matrix_tiles.js";
import {
  type CudaLiteCallExpression,
  type CudaLiteExpression,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type CudaLiteStatement,
  type CudaLiteVarDecl,
  type KernelIrModule,
  type SourceSpan,
} from "./types.js";

export interface PointerAlias {
  readonly rootName: string;
  readonly baseIndex: CudaLiteExpression;
  readonly valueType?: CudaLiteScalarType;
  readonly declarationSpan?: SourceSpan;
}

export interface PoolPointerAlias {
  readonly poolName: string;
  readonly offsetName?: string;
  readonly rawBuffer?: boolean;
}

export interface RawPoolAllocator {
  readonly baseName: string;
  readonly offsetName: string;
}

export function collectLocalNames(statements: readonly CudaLiteStatement[]): ReadonlySet<string> {
  const names = new Set<string>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" || item.kind === "dim3" || item.kind === "cooperative-group") names.add(item.name);
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var") names.add(item.init.name);
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return names;
}

export function collectMutableScalarParams(
  statements: readonly CudaLiteStatement[],
  params: readonly CudaLiteParam[],
): readonly CudaLiteParam[] {
  const paramByName = new Map(params.filter(isMutableKernelValueParam).map((param) => [param.name, param] as const));
  const mutated = new Set<string>();
  walkCudaLiteExpressions(statements, (expression) => {
    const name = mutatedIdentifierName(expression);
    if (name && paramByName.has(name)) mutated.add(name);
  });
  return params.filter((param) => mutated.has(param.name));
}

export function collectLocalArrays(statements: readonly CudaLiteStatement[]): ReadonlyMap<string, CudaLiteVarDecl> {
  const arrays = new Map<string, CudaLiteVarDecl>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.storage === "local" && (item.dimensions.length > 0 || item.matrixTile)) arrays.set(item.name, item);
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.storage === "local" && (item.init.dimensions.length > 0 || item.init.matrixTile)) {
          arrays.set(item.init.name, item.init);
        }
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return arrays;
}

export function collectLocalArrayDeclarations(statements: readonly CudaLiteStatement[]): readonly CudaLiteVarDecl[] {
  const declarations: CudaLiteVarDecl[] = [];
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.storage === "local" && (item.dimensions.length > 0 || item.matrixTile)) {
        declarations.push(item);
      }
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.storage === "local" && (item.init.dimensions.length > 0 || item.init.matrixTile)) {
          declarations.push(item.init);
        }
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return declarations.sort((left, right) => left.span.start - right.span.start);
}

export function localArrayDeclarationFor(
  declarations: readonly CudaLiteVarDecl[],
  name: string,
  span?: SourceSpan,
): CudaLiteVarDecl | undefined {
  let candidate: CudaLiteVarDecl | undefined;
  for (const declaration of declarations) {
    if (declaration.name !== name) continue;
    if (span && declaration.span.start > span.start) break;
    candidate = declaration;
  }
  return candidate;
}

export function collectLocalPointerArrayRoots(statements: readonly CudaLiteStatement[]): ReadonlyMap<string, CudaLiteVarDecl> {
  interface PointerArrayState {
    readonly declaration: CudaLiteVarDecl;
    root?: CudaLiteVarDecl;
    invalid: boolean;
    sawAssignment: boolean;
  }

  const states: PointerArrayState[] = [];
  const scanExpression = (
    expression: CudaLiteExpression,
    arrays: ReadonlyMap<string, CudaLiteVarDecl>,
    pointerArrays: ReadonlyMap<string, PointerArrayState>,
  ): void => {
    if (expression.kind === "assignment" && expression.left.kind === "index" && expression.left.target.kind === "identifier") {
      const state = pointerArrays.get(expression.left.target.name);
      if (state) {
        state.sawAssignment = true;
        const root = localArrayAddressRoot(expression.right, arrays);
        if (!root || (state.root !== undefined && state.root !== root)) {
          state.invalid = true;
        } else {
          state.root = root;
        }
      }
    }
    for (const child of expressionChildren(expression)) scanExpression(child, arrays, pointerArrays);
  };
  const walk = (
    items: readonly CudaLiteStatement[],
    inheritedArrays: ReadonlyMap<string, CudaLiteVarDecl>,
    inheritedPointerArrays: ReadonlyMap<string, PointerArrayState>,
  ): void => {
    const arrays = new Map(inheritedArrays);
    const pointerArrays = new Map(inheritedPointerArrays);
    for (const item of items) {
      switch (item.kind) {
        case "var":
          if (item.storage === "local" && item.dimensions.length > 0) {
            if (isLocalPointerArrayDecl(item)) {
              const state: PointerArrayState = { declaration: item, invalid: false, sawAssignment: false };
              pointerArrays.set(item.name, state);
              states.push(state);
            } else {
              arrays.set(item.name, item);
              pointerArrays.delete(item.name);
            }
          }
          if (item.init) scanExpression(item.init, arrays, pointerArrays);
          break;
        case "expr":
          scanExpression(item.expression, arrays, pointerArrays);
          break;
        case "if":
          scanExpression(item.condition, arrays, pointerArrays);
          walk(item.consequent, new Map(arrays), new Map(pointerArrays));
          if (item.alternate) walk(item.alternate, new Map(arrays), new Map(pointerArrays));
          break;
        case "for": {
          const loopArrays = new Map(arrays);
          const loopPointerArrays = new Map(pointerArrays);
          if (item.init?.kind === "var") {
            const init = item.init;
            if (init.storage === "local" && init.dimensions.length > 0) {
              if (isLocalPointerArrayDecl(init)) {
                const state: PointerArrayState = { declaration: init, invalid: false, sawAssignment: false };
                loopPointerArrays.set(init.name, state);
                states.push(state);
              } else {
                loopArrays.set(init.name, init);
                loopPointerArrays.delete(init.name);
              }
            }
            if (init.init) scanExpression(init.init, loopArrays, loopPointerArrays);
          } else if (item.init) {
            scanExpression(item.init, loopArrays, loopPointerArrays);
          }
          if (item.condition) scanExpression(item.condition, loopArrays, loopPointerArrays);
          if (item.update) scanExpression(item.update, loopArrays, loopPointerArrays);
          walk(item.body, loopArrays, loopPointerArrays);
          break;
        }
        case "while":
          scanExpression(item.condition, arrays, pointerArrays);
          walk(item.body, new Map(arrays), new Map(pointerArrays));
          break;
        case "do-while":
          walk(item.body, new Map(arrays), new Map(pointerArrays));
          scanExpression(item.condition, arrays, pointerArrays);
          break;
        case "block":
          walk(item.body, new Map(arrays), new Map(pointerArrays));
          break;
        case "kernel-launch":
          for (const expression of [...item.grid, ...item.block, ...item.args]) scanExpression(expression, arrays, pointerArrays);
          break;
        case "asm":
          if (item.output) scanExpression(item.output, arrays, pointerArrays);
          for (const output of item.outputs ?? []) scanExpression(output, arrays, pointerArrays);
          for (const input of item.inputs) scanExpression(input, arrays, pointerArrays);
          break;
        case "return":
          if (item.value) scanExpression(item.value, arrays, pointerArrays);
          break;
        case "dim3":
          for (const arg of item.args) scanExpression(arg, arrays, pointerArrays);
          break;
        case "cooperative-group":
        case "continue":
        case "break":
          break;
      }
    }
  };
  walk(statements, new Map(), new Map());
  const out = new Map<string, CudaLiteVarDecl>();
  for (const state of states) {
    if (state.sawAssignment && !state.invalid && state.root && !state.root.pointer) {
      out.set(state.declaration.name, state.root);
    }
  }
  return out;
}

export function collectLocalValueTypes(statements: readonly CudaLiteStatement[]): ReadonlyMap<string, CudaLiteScalarType> {
  const types = new Map<string, CudaLiteScalarType>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.storage === "local" && !item.pointer && item.dimensions.length === 0) {
        types.set(item.name, item.valueType);
      }
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.storage === "local" && !item.init.pointer && item.init.dimensions.length === 0) {
          types.set(item.init.name, item.init.valueType);
        }
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return types;
}

export function collectLocalPointerHandles(
  statements: readonly CudaLiteStatement[],
  poolPointers: ReadonlyMap<string, PoolPointerAlias> = collectPoolPointers(statements),
  structuredPointerRoots: ReadonlySet<string> = new Set(),
): ReadonlyMap<string, CudaLiteVarDecl> {
  const mutableNames = collectMutableLocalPointerNames(statements);
  const handles = new Map<string, CudaLiteVarDecl>();
  const needsHandle = (statement: CudaLiteVarDecl): boolean =>
    statement.pointer &&
    statement.storage === "local" &&
    !poolPointers.has(statement.name) &&
    (mutableNames.has(statement.name) ||
      needsStructuredAddressHandle(statement.init, structuredPointerRoots) ||
      needsDynamicPointerHandle(statement.init));
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && needsHandle(item)) {
        handles.set(item.name, item);
      }
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && needsHandle(item.init)) {
          handles.set(item.init.name, item.init);
        }
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return handles;
}

export function structuredPointerHandleRoots(ir: KernelIrModule): ReadonlySet<string> {
  return new Set([
    ...ir.constants.map((constant) => constant.name),
    ...ir.deviceGlobals.map((global) => global.name),
    ...ir.sharedDeclarations.map((shared) => shared.name),
  ]);
}

export function collectPointerAliases(
  statements: readonly CudaLiteStatement[],
  skipNames: ReadonlySet<string> = new Set(),
): ReadonlyMap<string, readonly PointerAlias[]> {
  const aliases = new Map<string, PointerAlias[]>();
  const addAlias = (statement: CudaLiteVarDecl, alias: PointerAlias): void => {
    const list = aliases.get(statement.name) ?? [];
    list.push({ ...alias, declarationSpan: statement.span });
    aliases.set(statement.name, list);
  };
  const walk = (items: readonly CudaLiteStatement[], inheritedArrays: ReadonlyMap<string, CudaLiteVarDecl>): void => {
    const arrays = new Map(inheritedArrays);
    for (const item of items) {
      if (item.kind === "var" && item.storage === "local" && (item.dimensions.length > 0 || item.matrixTile)) {
        arrays.set(item.name, item);
      }
      if (item.kind === "var" && item.pointer && !skipNames.has(item.name)) {
        const alias = pointerAliasForVar(item, arrays);
        if (alias) addAlias(item, alias);
      }
      if (item.kind === "dim3" || item.kind === "cooperative-group" || item.kind === "kernel-launch") continue;
      if (item.kind === "if") {
        walk(item.consequent, arrays);
        if (item.alternate) walk(item.alternate, arrays);
      }
      if (item.kind === "for") {
        const loopArrays = new Map(arrays);
        if (item.init?.kind === "var" && item.init.storage === "local" && (item.init.dimensions.length > 0 || item.init.matrixTile)) {
          loopArrays.set(item.init.name, item.init);
        }
        if (item.init?.kind === "var" && item.init.pointer && !skipNames.has(item.init.name)) {
          const alias = pointerAliasForVar(item.init, loopArrays);
          if (alias) addAlias(item.init, alias);
        }
        walk(item.body, loopArrays);
      }
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body, arrays);
    }
  };
  walk(statements, new Map());
  return aliases;
}

export function pointerAliasDeclarationFor(
  aliases: ReadonlyMap<string, readonly PointerAlias[]>,
  name: string,
  span?: SourceSpan,
): PointerAlias | undefined {
  const candidates = aliases.get(name);
  if (!candidates || candidates.length === 0) return undefined;
  if (!span) return candidates[candidates.length - 1];
  for (let index = candidates.length - 1; index >= 0; index--) {
    const candidate = candidates[index]!;
    if (!candidate.declarationSpan || candidate.declarationSpan.start <= span.start) return candidate;
  }
  return candidates[0];
}

export function collectMutableStoragePointerBases(
  statements: readonly CudaLiteStatement[],
  pointerParamNames: ReadonlySet<string>,
): readonly string[] {
  const names = new Set<string>();
  walkCudaLiteExpressions(statements, (expression) => {
    if (expression.kind === "assignment" && expression.left.kind === "identifier") {
      if ((expression.operator === "=" || expression.operator === "+=" || expression.operator === "-=") && pointerParamNames.has(expression.left.name)) {
        names.add(expression.left.name);
      }
      return;
    }
    if (expression.kind === "update" && expression.argument.kind === "identifier") {
      if (pointerParamNames.has(expression.argument.name)) names.add(expression.argument.name);
    }
  });
  return [...names].sort();
}

export function collectPoolPointers(statements: readonly CudaLiteStatement[]): ReadonlyMap<string, PoolPointerAlias> {
  const aliases = new Map<string, PoolPointerAlias>();
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.pointer) {
        const alias = poolPointerForInitializer(item.init, aliases);
        if (alias) aliases.set(item.name, alias);
      }
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.pointer) {
          const alias = poolPointerForInitializer(item.init.init, aliases);
          if (alias) aliases.set(item.init.name, alias);
        }
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walk(item.body);
    }
  };
  walk(statements);
  return aliases;
}

export function poolPointerForAllocationCall(call: CudaLiteCallExpression): PoolPointerAlias | undefined {
  const name = expressionName(call.callee);
  if (name !== "deviceAllocate" && name !== "streamOrderedAllocate") return undefined;
  if (call.args.length === 4) {
    const base = call.args[0];
    const offset = call.args[1];
    return base?.kind === "identifier" && offset?.kind === "identifier"
      ? { poolName: base.name, offsetName: offset.name, rawBuffer: true }
      : undefined;
  }
  const pool = call.args[0];
  if (pool?.kind === "unary" && pool.operator === "&" && pool.argument.kind === "identifier") {
    return { poolName: pool.argument.name };
  }
  return pool?.kind === "identifier" ? { poolName: pool.name } : undefined;
}

export function collectRawPoolAllocators(statements: readonly CudaLiteStatement[]): readonly RawPoolAllocator[] {
  const allocators = new Map<string, RawPoolAllocator>();
  const visitExpression = (expression: CudaLiteExpression): void => {
    if (expression.kind === "call") {
      const alias = poolPointerForAllocationCall(expression);
      if (alias?.rawBuffer && alias.offsetName) {
        const key = `${alias.poolName}\0${alias.offsetName}`;
        allocators.set(key, { baseName: alias.poolName, offsetName: alias.offsetName });
      }
      visitExpression(expression.callee);
      for (const arg of expression.args) visitExpression(arg);
      return;
    }
    if (expression.kind === "cast") visitExpression(expression.expression);
    else if (expression.kind === "member") visitExpression(expression.object);
    else if (expression.kind === "index") {
      visitExpression(expression.target);
      visitExpression(expression.index);
    } else if (expression.kind === "unary" || expression.kind === "update") visitExpression(expression.argument);
    else if (expression.kind === "binary") {
      visitExpression(expression.left);
      visitExpression(expression.right);
    } else if (expression.kind === "conditional") {
      visitExpression(expression.condition);
      visitExpression(expression.consequent);
      visitExpression(expression.alternate);
    } else if (expression.kind === "assignment") {
      visitExpression(expression.left);
      visitExpression(expression.right);
    } else if (expression.kind === "sequence") {
      for (const item of expression.expressions) visitExpression(item);
    }
  };
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.init) visitExpression(item.init);
      if (item.kind === "expr") visitExpression(item.expression);
      if (item.kind === "if") {
        visitExpression(item.condition);
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.init) visitExpression(item.init.init);
        else if (item.init && item.init.kind !== "var") visitExpression(item.init);
        if (item.condition) visitExpression(item.condition);
        if (item.update) visitExpression(item.update);
        walk(item.body);
      }
      if (item.kind === "while" || item.kind === "do-while") {
        walk(item.body);
        visitExpression(item.condition);
      }
      if (item.kind === "block") walk(item.body);
      if (item.kind === "return" && item.value) visitExpression(item.value);
    }
  };
  walk(statements);
  return [...allocators.values()];
}

export function pointerAliasForPointerExpression(
  expression: CudaLiteExpression | undefined,
  valueType: CudaLiteScalarType,
  localArrays: ReadonlyMap<string, CudaLiteVarDecl> = new Map(),
): PointerAlias | undefined {
  if (!expression) return undefined;
  if (expression.kind === "cast" && expression.pointer) return pointerAliasForPointerExpression(expression.expression, valueType, localArrays);
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    return pointerAliasForPointerExpression(expression.args[0], valueType, localArrays);
  }
  if (expression.kind === "unary" && expression.operator === "&") {
    const target = expression.argument;
    const local = localArrayAddressAlias(target, valueType, localArrays);
    if (local) return local;
    if (target.kind === "index" && target.target.kind === "identifier") {
      return { rootName: target.target.name, baseIndex: target.index, valueType };
    }
    if (target.kind === "identifier") {
      return { rootName: target.name, baseIndex: zeroExpression(target.span), valueType };
    }
    return undefined;
  }
  if (expression.kind === "identifier") {
    return { rootName: expression.name, baseIndex: zeroExpression(expression.span), valueType };
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    const base = pointerAliasForPointerExpression(expression.left, valueType, localArrays);
    if (!base) return undefined;
    return {
      ...base,
      baseIndex: expression.operator === "+"
        ? { kind: "binary", operator: "+", left: base.baseIndex, right: expression.right, span: expression.span }
        : { kind: "binary", operator: "-", left: base.baseIndex, right: expression.right, span: expression.span },
    };
  }
  return undefined;
}

export function flatLocalArrayIndexExpression(
  indices: readonly CudaLiteExpression[],
  dimensions: readonly number[],
  span: SourceSpan,
): CudaLiteExpression {
  if (indices.length === 0) return zeroExpression(span);
  const terms = indices.map((index, axis) => {
    const stride = dimensions.slice(axis + 1).reduce((product, dimension) => product * dimension, 1);
    if (stride === 1) return index;
    return {
      kind: "binary",
      operator: "*",
      left: index,
      right: { kind: "number", value: stride, raw: String(stride), span: index.span },
      span: index.span,
    } satisfies CudaLiteExpression;
  });
  return terms.reduce<CudaLiteExpression>((left, right) => ({
    kind: "binary",
    operator: "+",
    left,
    right,
    span,
  }), zeroExpression(span));
}

export function isPointerIdentityCall(name: string | undefined): boolean {
  return name === "__builtin_assume_aligned" || name === "ct::assume_aligned";
}

export function zeroExpression(span: CudaLiteExpression["span"]): CudaLiteExpression {
  return { kind: "number", value: 0, raw: "0", span };
}

export function expressionChildren(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  switch (expression.kind) {
    case "call":
      return [expression.callee, ...expression.args];
    case "initializer":
      return expression.elements;
    case "cast":
      return [expression.expression];
    case "member":
      return [expression.object];
    case "index":
      return [expression.target, expression.index];
    case "unary":
    case "update":
      return [expression.argument];
    case "binary":
      return [expression.left, expression.right];
    case "conditional":
      return [expression.condition, expression.consequent, expression.alternate];
    case "assignment":
      return [expression.left, expression.right];
    case "sequence":
      return expression.expressions;
    case "number":
    case "string":
    case "identifier":
      return [];
  }
}

export function isLocalPointerArrayDecl(statement: CudaLiteVarDecl): boolean {
  return statement.pointer && statement.storage === "local" && statement.dimensions.length > 0;
}

function mutatedIdentifierName(expression: CudaLiteExpression): string | undefined {
  if (expression.kind === "assignment" && expression.left.kind === "identifier") return expression.left.name;
  if (expression.kind === "update" && expression.argument.kind === "identifier") return expression.argument.name;
  return undefined;
}

function isMutableKernelValueParam(param: CudaLiteParam): boolean {
  return !param.pointer &&
    param.cooperativeGroupKind === undefined &&
    param.valueType !== "surface2d" &&
    param.valueType !== "texture2d" &&
    param.valueType !== "devicepool" &&
    param.valueType !== "voidptr";
}

function needsStructuredAddressHandle(
  expression: CudaLiteExpression | undefined,
  structuredPointerRoots: ReadonlySet<string>,
): boolean {
  if (!expression) return false;
  if (expression.kind === "cast" && expression.pointer) return needsStructuredAddressHandle(expression.expression, structuredPointerRoots);
  if (expression.kind === "conditional") {
    return needsStructuredAddressHandle(expression.consequent, structuredPointerRoots) ||
      needsStructuredAddressHandle(expression.alternate, structuredPointerRoots);
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    return needsStructuredAddressHandle(expression.left, structuredPointerRoots);
  }
  if (expression.kind !== "unary" || expression.operator !== "&") return false;
  let depth = 0;
  let cursor = expression.argument;
  while (cursor.kind === "index") {
    depth++;
    cursor = cursor.target;
  }
  return depth > 0 && cursor.kind === "identifier" && structuredPointerRoots.has(cursor.name);
}

function needsDynamicPointerHandle(expression: CudaLiteExpression | undefined): boolean {
  if (!expression) return false;
  if (expression.kind === "cast" && expression.pointer) return needsDynamicPointerHandle(expression.expression);
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    return needsDynamicPointerHandle(expression.args[0]);
  }
  if (expression.kind === "conditional") return true;
  return false;
}

function collectMutableLocalPointerNames(statements: readonly CudaLiteStatement[]): ReadonlySet<string> {
  const declared = new Set<string>();
  const mutated = new Set<string>();
  const walkStatements = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.pointer && item.storage === "local") declared.add(item.name);
      if (item.kind === "for" && item.init?.kind === "var" && item.init.pointer && item.init.storage === "local") declared.add(item.init.name);
      if (item.kind === "if") {
        walkStatements(item.consequent);
        if (item.alternate) walkStatements(item.alternate);
      }
      if (item.kind === "for") walkStatements(item.body);
      if (item.kind === "while" || item.kind === "do-while" || item.kind === "block") walkStatements(item.body);
    }
  };
  walkStatements(statements);
  walkCudaLiteExpressions(statements, (expression) => {
    if (expression.kind === "assignment" && expression.left.kind === "identifier" && declared.has(expression.left.name)) {
      mutated.add(expression.left.name);
    }
    if (expression.kind === "update" && expression.argument.kind === "identifier" && declared.has(expression.argument.name)) {
      mutated.add(expression.argument.name);
    }
  });
  return mutated;
}

function poolPointerForInitializer(
  init: CudaLiteExpression | undefined,
  aliases: ReadonlyMap<string, PoolPointerAlias>,
): PoolPointerAlias | undefined {
  if (!init) return undefined;
  if (init.kind === "call") return poolPointerForAllocationCall(init);
  if (init.kind === "cast" && init.pointer) return poolPointerForInitializer(init.expression, aliases);
  if (init.kind === "identifier") return aliases.get(init.name);
  return undefined;
}

function pointerAliasForVar(
  statement: CudaLiteVarDecl,
  localArrays: ReadonlyMap<string, CudaLiteVarDecl> = new Map(),
): PointerAlias | undefined {
  const init = statement.init;
  const view = pointerAliasForPointerExpression(init, statement.valueType, localArrays);
  if (view) return view;
  if (init?.kind !== "unary" || init.operator !== "&") return undefined;
  const target = init.argument;
  if (target.kind !== "index" || target.target.kind !== "identifier") return undefined;
  return {
    rootName: target.target.name,
    baseIndex: target.index,
  };
}

function localArrayAddressAlias(
  expression: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  localArrays: ReadonlyMap<string, CudaLiteVarDecl>,
): PointerAlias | undefined {
  const indices: CudaLiteExpression[] = [];
  let cursor = expression;
  while (cursor.kind === "index") {
    indices.unshift(cursor.index);
    cursor = cursor.target;
  }
  if (cursor.kind !== "identifier") return undefined;
  const declaration = localArrays.get(cursor.name);
  if (!declaration) return undefined;
  return {
    rootName: cursor.name,
    baseIndex: flatLocalArrayIndexExpression(indices, matrixTileStorageDimensions(declaration), expression.span),
    valueType,
  };
}

function localArrayAddressRoot(
  expression: CudaLiteExpression,
  arrays: ReadonlyMap<string, CudaLiteVarDecl>,
): CudaLiteVarDecl | undefined {
  if (expression.kind === "cast" && expression.pointer) return localArrayAddressRoot(expression.expression, arrays);
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    return expression.args[0] ? localArrayAddressRoot(expression.args[0], arrays) : undefined;
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    return localArrayAddressRoot(expression.left, arrays);
  }
  if (expression.kind === "conditional") {
    const consequent = localArrayAddressRoot(expression.consequent, arrays);
    const alternate = localArrayAddressRoot(expression.alternate, arrays);
    return consequent && consequent === alternate ? consequent : undefined;
  }
  if (expression.kind !== "unary" || expression.operator !== "&") return undefined;
  const root = rootIdentifier(expression.argument);
  const declaration = root ? arrays.get(root) : undefined;
  return declaration && !declaration.pointer ? declaration : undefined;
}
