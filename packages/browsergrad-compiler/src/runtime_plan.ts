import { walkCudaLiteExpressions } from "./ast_queries.js";
import { lowerAnalyzedCudaLiteToKernelIr } from "./analyzer.js";
import type {
  SemanticExpression,
  SemanticKernelIrOperation,
} from "./semantic_ir.js";
import type {
  CompiledCudaLiteKernel,
  CudaLiteExpression,
  CudaLiteStatement,
  CudaLiteVarDecl,
  KernelIrModule,
  SourceSpan,
} from "./types.js";

export type CudaRuntimeOperationKind =
  | "device-launch"
  | "device-sync"
  | "grid-sync"
  | "runtime-copy";

export interface CudaRuntimeOperation {
  readonly kind: CudaRuntimeOperationKind;
  readonly span: SourceSpan;
  readonly label: string;
}

export interface CudaRuntimePlan {
  readonly operations: readonly CudaRuntimeOperation[];
  readonly requiresHostOrchestration: boolean;
  readonly canRunSingleDispatchWebGpu: boolean;
  readonly referenceAvailable: boolean;
}

export type CudaGridSyncPhasePlan =
  | {
      readonly supported: true;
      readonly modules: readonly KernelIrModule[];
    }
  | {
      readonly supported: false;
      readonly reason: string;
      readonly modules: readonly [];
    };

export function createCudaRuntimePlan(
  compiled: CompiledCudaLiteKernel,
): CudaRuntimePlan {
  const operations = collectSemanticRuntimeOperations(compiled.kernelIr.operations);
  return {
    operations,
    requiresHostOrchestration: operations.length > 0,
    canRunSingleDispatchWebGpu: operations.length === 0,
    referenceAvailable: operations.every((operation) => REFERENCE_RUNTIME_OPERATIONS.has(operation.kind)),
  };
}

function collectSemanticRuntimeOperations(operations: readonly SemanticKernelIrOperation[]): readonly CudaRuntimeOperation[] {
  const runtime: CudaRuntimeOperation[] = [];
  const cooperativeGroups = new Map<string, Extract<CudaLiteStatement, { kind: "cooperative-group" }>["groupKind"]>();
  visitSemanticOperations(operations, (operation) => {
    if (operation.kind === "cooperative-group-declare") cooperativeGroups.set(operation.declaration.name, operation.declaration.groupKind);
    if (operation.kind === "device-launch") {
      runtime.push({
        kind: "device-launch",
        span: operation.span,
        label: `${operation.launch.callee}<<<...>>>`,
      });
      return;
    }
    if (
      operation.kind === "barrier" &&
      (operation.callee === "grid.sync" || operation.callee === "cg::sync") &&
      operation.groupName !== undefined &&
      cooperativeGroups.get(operation.groupName) === "grid"
    ) {
      runtime.push({
        kind: "grid-sync",
        span: operation.span,
        label: "grid.sync()",
      });
      return;
    }
    if (operation.kind === "call" || operation.kind === "atomic") {
      const runtimeOperation = runtimeOperationForSemanticCall(operation.callee, operation.span);
      if (runtimeOperation) runtime.push(runtimeOperation);
    }
    for (const expression of semanticExpressionsForOperation(operation)) {
      visitSemanticExpression(expression, (item) => {
        const runtimeOperation = runtimeOperationForSemanticExpression(item, cooperativeGroups);
        if (runtimeOperation) runtime.push(runtimeOperation);
      });
    }
  });
  const unique: CudaRuntimeOperation[] = [...uniqueRuntimeOperations(runtime)];
  return unique.sort((left: CudaRuntimeOperation, right: CudaRuntimeOperation) => left.span.start - right.span.start);
}

const REFERENCE_RUNTIME_OPERATIONS: ReadonlySet<CudaRuntimeOperationKind> = new Set([
  "device-launch",
  "device-sync",
  "grid-sync",
  "runtime-copy",
]);

export function createCudaGridSyncPhasePlan(compiled: CompiledCudaLiteKernel): CudaGridSyncPhasePlan {
  const ir = lowerAnalyzedCudaLiteToKernelIr(compiled.analysis, {
    workgroupSize: compiled.kernelIr.workgroupSize,
    ...(compiled.dynamicSharedMemory === undefined ? {} : { dynamicSharedMemory: compiled.dynamicSharedMemory }),
  });
  const operations = collectRuntimeOperations(ir.body);
  const runtimePlan: CudaRuntimePlan = {
    operations,
    requiresHostOrchestration: operations.length > 0,
    canRunSingleDispatchWebGpu: operations.length === 0,
    referenceAvailable: operations.every((operation) => REFERENCE_RUNTIME_OPERATIONS.has(operation.kind)),
  };
  if (runtimePlan.operations.length === 0) return { supported: true, modules: [ir] };
  const unsupported = runtimePlan.operations.find((operation) => operation.kind !== "grid-sync");
  if (unsupported) {
    return { supported: false, reason: `${unsupported.kind} cannot be split into grid-sync phases`, modules: [] };
  }

  const cooperativeGroups = collectCooperativeGroups(ir.body);
  const topLevelGridSyncs = new Set(
    ir.body
      .filter((statement) => isGridSyncStatement(statement, cooperativeGroups))
      .map((statement) => statement.span.start),
  );
  if (topLevelGridSyncs.size !== runtimePlan.operations.length) {
    return { supported: false, reason: "grid.sync() must be a top-level uniform statement for WebGPU phase splitting", modules: [] };
  }

  const rawPhases: CudaLiteStatement[][] = [[]];
  for (const statement of ir.body) {
    if (isGridSyncStatement(statement, cooperativeGroups)) {
      rawPhases.push([]);
      continue;
    }
    rawPhases[rawPhases.length - 1]!.push(statement);
  }

  const phaseReplay = replayableDeclarationsForPhases(rawPhases, ir);
  const safety = validateGridSyncPhaseSafety(rawPhases, ir, phaseReplay);
  if (safety) return { supported: false, reason: safety, modules: [] };

  const groupDeclarations = ir.body.filter((statement): statement is Extract<CudaLiteStatement, { kind: "cooperative-group" }> =>
    statement.kind === "cooperative-group"
  );
  return {
    supported: true,
    modules: rawPhases.map((phase, index): KernelIrModule => ({
      ...ir,
      name: `${ir.name}_grid_phase_${index}`,
      body: withCooperativeGroupMetadata(groupDeclarations, phaseReplay[index] ?? [], phase),
    })),
  };
}

function collectRuntimeOperations(statements: readonly CudaLiteStatement[]): readonly CudaRuntimeOperation[] {
  const operations: CudaRuntimeOperation[] = [];
  const cooperativeGroups = new Map<string, Extract<CudaLiteStatement, { kind: "cooperative-group" }>["groupKind"]>();
  visitStatements(statements, (statement) => {
    if (statement.kind === "cooperative-group") cooperativeGroups.set(statement.name, statement.groupKind);
    if (statement.kind === "kernel-launch") {
      operations.push({
        kind: "device-launch",
        span: statement.span,
        label: `${statement.callee}<<<...>>>`,
      });
    }
  });
  walkCudaLiteExpressions(statements, (expression) => {
    const operation = runtimeOperationForExpression(expression, cooperativeGroups);
    if (operation) operations.push(operation);
  });
  return operations.sort((left, right) => left.span.start - right.span.start);
}

function runtimeOperationForExpression(
  expression: CudaLiteExpression,
  cooperativeGroups: ReadonlyMap<string, Extract<CudaLiteStatement, { kind: "cooperative-group" }>["groupKind"]>,
): CudaRuntimeOperation | undefined {
  if (expression.kind !== "call") return undefined;
  if (expression.callee.kind === "identifier") {
    if (isHostManagedRuntimeNoopCall(expression.callee.name)) {
      return {
        kind: "device-sync",
        span: expression.span,
        label: `${expression.callee.name}()`,
      };
    }
    if (isCudaRuntimeCopyCall(expression.callee.name)) {
      return {
        kind: "runtime-copy",
        span: expression.span,
        label: `${expression.callee.name}(...)`,
      };
    }
  }
  if (
    expression.callee.kind !== "member" ||
    expression.callee.property !== "sync" ||
    expression.callee.object.kind !== "identifier" ||
    cooperativeGroups.get(expression.callee.object.name) !== "grid"
  ) {
    const group = namespaceSyncGroupName(expression);
    if (!group || cooperativeGroups.get(group) !== "grid") return undefined;
  }
  return {
    kind: "grid-sync",
    span: expression.span,
    label: "grid.sync()",
  };
}

function visitSemanticOperations(
  operations: readonly SemanticKernelIrOperation[],
  visit: (operation: SemanticKernelIrOperation) => void,
): void {
  for (const operation of operations) {
    visit(operation);
    if (operation.kind === "block") visitSemanticOperations(operation.body, visit);
    else if (operation.kind === "branch") {
      visitSemanticOperations(operation.consequent, visit);
      visitSemanticOperations(operation.alternate, visit);
    } else if (operation.kind === "loop") {
      if (operation.init && isSemanticOperation(operation.init)) visitSemanticOperations([operation.init], visit);
      visitSemanticOperations(operation.body, visit);
    }
  }
}

function isSemanticOperation(value: SemanticKernelIrOperation | SemanticExpression): value is SemanticKernelIrOperation {
  if (value.kind === "call") return "reads" in value;
  return value.kind === "declare" ||
    value.kind === "dim3-declare" ||
    value.kind === "cooperative-group-declare" ||
    value.kind === "load" ||
    value.kind === "store" ||
    value.kind === "surface-write" ||
    value.kind === "surface-read-store" ||
    value.kind === "atomic" ||
    value.kind === "expression" ||
    value.kind === "branch" ||
    value.kind === "loop" ||
    value.kind === "barrier" ||
    value.kind === "fence" ||
    value.kind === "device-launch" ||
    value.kind === "inline-asm" ||
    value.kind === "return" ||
    value.kind === "continue" ||
    value.kind === "break" ||
    value.kind === "block";
}

function semanticExpressionsForOperation(operation: SemanticKernelIrOperation): readonly SemanticExpression[] {
  switch (operation.kind) {
    case "declare":
      return operation.init ? [operation.init] : [];
    case "dim3-declare":
      return operation.args;
    case "load":
      return operation.source.indices;
    case "store":
      return [operation.value, ...operation.target.indices, ...operation.reads.flatMap((read) => read.indices)];
    case "surface-write":
      return [operation.surface, operation.value, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : [])];
    case "surface-read-store":
      return [operation.target, operation.surface, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : [])];
    case "atomic":
      return [...operation.args, ...(operation.target?.indices ?? [])];
    case "call":
      return operation.args;
    case "expression":
      return [operation.expression];
    case "branch":
      return [operation.condition];
    case "loop":
      return [
        ...(!operation.init || isSemanticOperation(operation.init) ? [] : [operation.init]),
        ...(operation.condition ? [operation.condition] : []),
        ...(operation.update ? [operation.update] : []),
      ];
    case "device-launch":
      return [...operation.launch.grid, ...operation.launch.block, ...operation.launch.args];
    case "return":
      return operation.value ? [operation.value] : [];
    case "cooperative-group-declare":
    case "barrier":
    case "fence":
    case "inline-asm":
    case "continue":
    case "break":
    case "block":
      return [];
  }
}

function visitSemanticExpression(
  expression: SemanticExpression,
  visit: (expression: SemanticExpression) => void,
): void {
  visit(expression);
  switch (expression.kind) {
    case "literal":
    case "symbol":
      return;
    case "member":
      visitSemanticExpression(expression.object, visit);
      return;
    case "index":
      visitSemanticExpression(expression.target, visit);
      visitSemanticExpression(expression.index, visit);
      return;
    case "call":
      visitSemanticExpression(expression.callee, visit);
      for (const arg of expression.args) visitSemanticExpression(arg, visit);
      return;
    case "texture-read":
      visitSemanticExpression(expression.texture, visit);
      visitSemanticExpression(expression.x, visit);
      visitSemanticExpression(expression.y, visit);
      return;
    case "surface-read":
      visitSemanticExpression(expression.surface, visit);
      visitSemanticExpression(expression.xBytes, visit);
      visitSemanticExpression(expression.y, visit);
      if (expression.z) visitSemanticExpression(expression.z, visit);
      return;
    case "cast":
      visitSemanticExpression(expression.expression, visit);
      return;
    case "unary":
    case "update":
      visitSemanticExpression(expression.argument, visit);
      return;
    case "binary":
      visitSemanticExpression(expression.left, visit);
      visitSemanticExpression(expression.right, visit);
      return;
    case "conditional":
      visitSemanticExpression(expression.condition, visit);
      visitSemanticExpression(expression.consequent, visit);
      visitSemanticExpression(expression.alternate, visit);
      return;
    case "assignment":
      visitSemanticExpression(expression.target, visit);
      visitSemanticExpression(expression.value, visit);
      return;
    case "initializer":
      for (const element of expression.elements) visitSemanticExpression(element, visit);
      return;
    case "sequence":
      for (const item of expression.expressions) visitSemanticExpression(item, visit);
      return;
  }
}

function runtimeOperationForSemanticExpression(
  expression: SemanticExpression,
  cooperativeGroups: ReadonlyMap<string, Extract<CudaLiteStatement, { kind: "cooperative-group" }>["groupKind"]>,
): CudaRuntimeOperation | undefined {
  if (expression.kind !== "call") return undefined;
  const callName = semanticCallName(expression.callee);
  const runtimeOperation = callName ? runtimeOperationForSemanticCall(callName, expression.span) : undefined;
  if (runtimeOperation) return runtimeOperation;
  const syncGroup = semanticSyncGroupName(expression);
  if (syncGroup && cooperativeGroups.get(syncGroup) === "grid") {
    return {
      kind: "grid-sync",
      span: expression.span,
      label: "grid.sync()",
    };
  }
  return undefined;
}

function runtimeOperationForSemanticCall(name: string, span: SourceSpan): CudaRuntimeOperation | undefined {
  if (isHostManagedRuntimeNoopCall(name)) {
    return {
      kind: "device-sync",
      span,
      label: `${name}()`,
    };
  }
  if (isCudaRuntimeCopyCall(name)) {
    return {
      kind: "runtime-copy",
      span,
      label: `${name}(...)`,
    };
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

function semanticSyncGroupName(expression: Extract<SemanticExpression, { readonly kind: "call" }>): string | undefined {
  if (expression.callee.kind === "member" && expression.callee.property === "sync" && expression.callee.object.kind === "symbol") {
    return expression.callee.object.name;
  }
  if (expression.callee.kind === "symbol" && expression.callee.name.endsWith("::sync")) {
    const group = expression.args[0];
    return group?.kind === "symbol" ? group.name : undefined;
  }
  return undefined;
}

function uniqueRuntimeOperations(operations: readonly CudaRuntimeOperation[]): readonly CudaRuntimeOperation[] {
  const seen = new Set<string>();
  const out: CudaRuntimeOperation[] = [];
  for (const operation of operations) {
    const key = `${operation.kind}:${operation.span.start}:${operation.span.end}:${operation.label}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(operation);
  }
  return out;
}

function isCudaRuntimeCopyCall(name: string): boolean {
  return name === "cudaMemcpy" ||
    name === "cudaMemcpyAsync" ||
    name === "cudaMemcpy2D" ||
    name === "cudaMemcpy2DAsync" ||
    name === "cudaMemcpyPeer" ||
    name === "cudaMemcpyPeerAsync" ||
    name === "cudaMemcpyToSymbol" ||
    name === "cudaMemcpyToSymbolAsync" ||
    name === "cudaMemcpyFromSymbol" ||
    name === "cudaMemcpyFromSymbolAsync" ||
    name === "cudaMemset" ||
    name === "cudaMemsetAsync" ||
    name === "cudaMemset2D" ||
    name === "cudaMemset2DAsync";
}

function isHostManagedRuntimeNoopCall(name: string): boolean {
  return name === "cudaDeviceSynchronize" ||
    name === "cudaCtxResetPersistingL2Cache" ||
    name === "cudaDeviceReset" ||
    name === "cudaThreadExit" ||
    name === "cudaThreadSynchronize" ||
    name === "cudaDeviceGetAttribute" ||
    name === "cudaDeviceGetLimit" ||
    name === "cudaThreadGetLimit" ||
    name === "cudaDeviceSetLimit" ||
    name === "cudaThreadSetLimit" ||
    name === "cudaDeviceCanAccessPeer" ||
    name === "cudaDeviceEnablePeerAccess" ||
    name === "cudaDeviceDisablePeerAccess" ||
    name === "cudaGetDeviceFlags" ||
    name === "cudaSetDeviceFlags" ||
    name === "cudaMemGetInfo" ||
    name === "cudaOccupancyMaxActiveBlocksPerMultiprocessor" ||
    name === "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags" ||
    name === "cudaOccupancyMaxPotentialBlockSize" ||
    name === "cudaOccupancyMaxPotentialBlockSizeWithFlags" ||
    name === "cudaDeviceGetCacheConfig" ||
    name === "cudaDeviceSetCacheConfig" ||
    name === "cudaDeviceGetSharedMemConfig" ||
    name === "cudaThreadGetCacheConfig" ||
    name === "cudaDeviceSetSharedMemConfig" ||
    name === "cudaThreadSetCacheConfig" ||
    name === "cudaThreadExchangeStreamCaptureMode" ||
    name === "cudaDeviceGetStreamPriorityRange" ||
    name === "cudaFree" ||
    name === "cudaFreeAsync" ||
    name === "cudaMemAdvise" ||
    name === "cudaMemPrefetchAsync" ||
    name === "cudaStreamAttachMemAsync" ||
    name === "cudaStreamCreate" ||
    name === "cudaStreamCreateWithFlags" ||
    name === "cudaStreamCreateWithPriority" ||
    name === "cudaStreamDestroy" ||
    name === "cudaStreamGetDevice" ||
    name === "cudaStreamGetFlags" ||
    name === "cudaStreamGetId" ||
    name === "cudaStreamGetPriority" ||
    name === "cudaStreamIsCapturing" ||
    name === "cudaStreamGetCaptureInfo" ||
    name === "cudaStreamBeginCapture" ||
    name === "cudaStreamEndCapture" ||
    name === "cudaStreamUpdateCaptureDependencies" ||
    name === "cudaGraphCreate" ||
    name === "cudaGraphInstantiate" ||
    name === "cudaGraphInstantiateWithFlags" ||
    name === "cudaGraphUpload" ||
    name === "cudaGraphExecUpdate" ||
    name === "cudaGraphDestroy" ||
    name === "cudaGraphExecDestroy" ||
    name === "cudaStreamQuery" ||
    name === "cudaStreamSynchronize" ||
    name === "cudaStreamWaitEvent" ||
    name === "cudaSetDevice" ||
    name === "cudaGetDevice" ||
    name === "cudaGetDeviceCount" ||
    name === "cudaRuntimeGetVersion" ||
    name === "cudaDriverGetVersion" ||
    name === "cudaFuncSetAttribute" ||
    name === "cudaFuncSetCacheConfig" ||
    name === "cudaFuncSetSharedMemConfig" ||
    name === "cudaGetLastError" ||
    name === "cudaPeekAtLastError" ||
    name === "cudaProfilerStart" ||
    name === "cudaProfilerStop" ||
    name === "cudaEventCreate" ||
    name === "cudaEventCreateWithFlags" ||
    name === "cudaEventDestroy" ||
    name === "cudaEventQuery" ||
    name === "cudaEventRecord" ||
    name === "cudaEventRecordWithFlags" ||
    name === "cudaEventSynchronize";
}

function collectCooperativeGroups(
  statements: readonly CudaLiteStatement[],
): ReadonlyMap<string, Extract<CudaLiteStatement, { kind: "cooperative-group" }>["groupKind"]> {
  const groups = new Map<string, Extract<CudaLiteStatement, { kind: "cooperative-group" }>["groupKind"]>();
  visitStatements(statements, (statement) => {
    if (statement.kind === "cooperative-group") groups.set(statement.name, statement.groupKind);
  });
  return groups;
}

function isGridSyncStatement(
  statement: CudaLiteStatement,
  cooperativeGroups: ReadonlyMap<string, Extract<CudaLiteStatement, { kind: "cooperative-group" }>["groupKind"]>,
): boolean {
  if (statement.kind !== "expr") return false;
  const expression = statement.expression;
  if (expression.kind !== "call") return false;
  const namespaceGroup = namespaceSyncGroupName(expression);
  if (namespaceGroup) return cooperativeGroups.get(namespaceGroup) === "grid";
  return expression.kind === "call" &&
    expression.callee.kind === "member" &&
    expression.callee.property === "sync" &&
    expression.callee.object.kind === "identifier" &&
    cooperativeGroups.get(expression.callee.object.name) === "grid";
}

function namespaceSyncGroupName(expression: CudaLiteExpression): string | undefined {
  if (expression.kind !== "call" || expression.callee.kind !== "identifier" || !expression.callee.name.endsWith("::sync")) return undefined;
  const group = expression.args[0];
  return group?.kind === "identifier" ? group.name : undefined;
}

function validateGridSyncPhaseSafety(
  phases: readonly (readonly CudaLiteStatement[])[],
  ir: KernelIrModule,
  phaseReplay: readonly (readonly CudaLiteVarDecl[])[],
): string | undefined {
  const previousLocals = new Set<string>();
  const previousSharedAccesses = new Set<string>();
  const sharedNames = new Set(ir.sharedDeclarations.map((declaration) => declaration.name));
  const globals = new Set([
    ...ir.params.map((param) => param.name),
    ...ir.constants.map((constant) => constant.name),
    ...ir.deviceGlobals.map((global) => global.name),
    ...ir.textures.map((texture) => texture.name),
    ...ir.functions.map((fn) => fn.name),
    "threadIdx",
    "blockIdx",
    "blockDim",
    "gridDim",
    "nullptr",
  ]);

  for (let phaseIndex = 0; phaseIndex < phases.length; phaseIndex++) {
    const phase = phases[phaseIndex]!;
    if (phaseIndex < phases.length - 1 && phaseHasReturn(phase)) {
      return "return before grid.sync() cannot be replayed safely across WebGPU phases";
    }

    const refs = identifiersReferencedBy(phase);
    const sharedAccesses = firstSharedAccessesByName(phase, sharedNames);
    const currentLocals = localsDeclaredBy(phase);
    const replayedLocals = new Set((phaseReplay[phaseIndex] ?? []).map((declaration) => declaration.name));
    for (const name of refs) {
      if (previousLocals.has(name) && !currentLocals.has(name) && !globals.has(name) && !replayedLocals.has(name)) {
        return `local '${name}' crosses grid.sync(); WebGPU phases cannot preserve private thread state`;
      }
    }

    for (const [name, firstAccess] of sharedAccesses) {
      if (previousSharedAccesses.has(name) && firstAccess !== "write") {
        return `shared memory '${name}' is read before rewrite after grid.sync(); WebGPU workgroup memory cannot persist across dispatches`;
      }
      previousSharedAccesses.add(name);
    }
    for (const name of currentLocals) previousLocals.add(name);
  }
  return undefined;
}

function withCooperativeGroupMetadata(
  declarations: readonly Extract<CudaLiteStatement, { kind: "cooperative-group" }>[],
  replay: readonly CudaLiteVarDecl[],
  phase: readonly CudaLiteStatement[],
): readonly CudaLiteStatement[] {
  const phaseGroups = new Set(phase.filter((statement) => statement.kind === "cooperative-group").map((statement) => statement.name));
  const phaseLocals = localsDeclaredBy(phase);
  return [
    ...declarations.filter((declaration) => !phaseGroups.has(declaration.name)),
    ...replay.filter((declaration) => !phaseLocals.has(declaration.name)),
    ...phase,
  ];
}

function replayableDeclarationsForPhases(
  phases: readonly (readonly CudaLiteStatement[])[],
  ir: KernelIrModule,
): readonly (readonly CudaLiteVarDecl[])[] {
  const globals = new Set([
    ...ir.params.map((param) => param.name),
    ...ir.constants.map((constant) => constant.name),
    ...ir.deviceGlobals.map((global) => global.name),
    ...ir.textures.map((texture) => texture.name),
    ...ir.functions.map((fn) => fn.name),
    "threadIdx",
    "blockIdx",
    "blockDim",
    "gridDim",
    "nullptr",
  ]);
  const available = new Map<string, CudaLiteVarDecl>();
  const out: CudaLiteVarDecl[][] = [];
  for (const phase of phases) {
    out.push([...replayClosureFor(phase, available)]);
    for (const declaration of topLevelReplayableLocalDeclarations(phase, globals, new Set(available.keys()))) {
      available.set(declaration.name, declaration);
    }
  }
  return out;
}

function replayClosureFor(
  phase: readonly CudaLiteStatement[],
  available: ReadonlyMap<string, CudaLiteVarDecl>,
): readonly CudaLiteVarDecl[] {
  const needed = new Set(identifiersReferencedBy(phase));
  const emitted = new Set<string>();
  const visiting = new Set<string>();
  const ordered: CudaLiteVarDecl[] = [];
  const emit = (name: string): void => {
    if (emitted.has(name) || visiting.has(name)) return;
    const declaration = available.get(name);
    if (!declaration) return;
    visiting.add(name);
    const initRefs = declaration.init ? identifiersReferencedByExpression(declaration.init) : new Set<string>();
    for (const ref of initRefs) emit(ref);
    visiting.delete(name);
    ordered.push(declaration);
    emitted.add(name);
  };
  for (const name of needed) emit(name);
  return ordered;
}

function topLevelReplayableLocalDeclarations(
  phase: readonly CudaLiteStatement[],
  globals: ReadonlySet<string>,
  priorReplayable: ReadonlySet<string>,
): readonly CudaLiteVarDecl[] {
  const replayable = new Set(priorReplayable);
  const declarations: CudaLiteVarDecl[] = [];
  for (const statement of phase) {
    if (
      statement.kind === "var" &&
      statement.storage === "local" &&
      !statement.pointer &&
      statement.dimensions.length === 0 &&
      statement.init &&
      isReplayableExpression(statement.init, globals, replayable)
    ) {
      declarations.push(statement);
      replayable.add(statement.name);
    }
  }
  return declarations;
}

function isReplayableExpression(
  expression: CudaLiteExpression,
  globals: ReadonlySet<string>,
  replayableLocals: ReadonlySet<string>,
): boolean {
  switch (expression.kind) {
    case "number":
    case "string":
      return true;
    case "initializer":
      return expression.elements.every((element) => isReplayableExpression(element, globals, replayableLocals));
    case "identifier":
      return globals.has(expression.name) || replayableLocals.has(expression.name);
    case "member":
      return isReplayableExpression(expression.object, globals, replayableLocals);
    case "cast":
      return isReplayableExpression(expression.expression, globals, replayableLocals);
    case "unary":
      return expression.operator !== "*" && expression.operator !== "&" &&
        isReplayableExpression(expression.argument, globals, replayableLocals);
    case "binary":
      return isReplayableExpression(expression.left, globals, replayableLocals) &&
        isReplayableExpression(expression.right, globals, replayableLocals);
    case "conditional":
      return isReplayableExpression(expression.condition, globals, replayableLocals) &&
        isReplayableExpression(expression.consequent, globals, replayableLocals) &&
        isReplayableExpression(expression.alternate, globals, replayableLocals);
    case "sequence":
      return false;
    case "index":
    case "call":
    case "assignment":
    case "update":
      return false;
  }
}

function localsDeclaredBy(statements: readonly CudaLiteStatement[]): ReadonlySet<string> {
  const locals = new Set<string>();
  visitStatements(statements, (statement) => {
    if ((statement.kind === "var" && statement.storage === "local") || statement.kind === "dim3") locals.add(statement.name);
    if (statement.kind === "for" && statement.init?.kind === "var" && statement.init.storage === "local") locals.add(statement.init.name);
  });
  return locals;
}

function identifiersReferencedBy(statements: readonly CudaLiteStatement[]): ReadonlySet<string> {
  const refs = new Set<string>();
  walkCudaLiteExpressions(statements, (expression) => {
    if (expression.kind === "identifier") refs.add(expression.name);
  });
  return refs;
}

function identifiersReferencedByExpression(expression: CudaLiteExpression): ReadonlySet<string> {
  const refs = new Set<string>();
  const visit = (item: CudaLiteExpression): void => {
    switch (item.kind) {
      case "identifier":
        refs.add(item.name);
        return;
      case "number":
      case "string":
        return;
      case "cast":
        visit(item.expression);
        return;
      case "member":
        visit(item.object);
        return;
      case "index":
        visit(item.target);
        visit(item.index);
        return;
      case "call":
        visit(item.callee);
        for (const arg of item.args) visit(arg);
        return;
      case "unary":
      case "update":
        visit(item.argument);
        return;
      case "binary":
        visit(item.left);
        visit(item.right);
        return;
      case "conditional":
        visit(item.condition);
        visit(item.consequent);
        visit(item.alternate);
        return;
      case "assignment":
        visit(item.left);
        visit(item.right);
        return;
      case "sequence":
        for (const expression of item.expressions) visit(expression);
        return;
    }
  };
  visit(expression);
  return refs;
}

type SharedAccess = "read" | "write";

function firstSharedAccessesByName(
  statements: readonly CudaLiteStatement[],
  sharedNames: ReadonlySet<string>,
): ReadonlyMap<string, SharedAccess> {
  const accesses = new Map<string, SharedAccess>();
  const record = (expression: CudaLiteExpression, access: SharedAccess): void => {
    const name = expressionRoot(expression);
    if (name && sharedNames.has(name) && !accesses.has(name)) accesses.set(name, access);
  };
  const visitLvalueIndexes = (expression: CudaLiteExpression): void => {
    if (expression.kind === "index") {
      visitExpression(expression.index);
      visitLvalueIndexes(expression.target);
    } else if (expression.kind === "member") {
      visitLvalueIndexes(expression.object);
    } else if (expression.kind === "unary" && expression.operator === "*") {
      visitExpression(expression.argument);
    }
  };
  const visitExpression = (expression: CudaLiteExpression): void => {
    switch (expression.kind) {
      case "identifier":
      case "number":
      case "string":
        record(expression, "read");
        return;
      case "cast":
        visitExpression(expression.expression);
        return;
      case "member":
        visitExpression(expression.object);
        return;
      case "index":
        record(expression, "read");
        visitLvalueIndexes(expression);
        return;
      case "call":
        visitExpression(expression.callee);
        for (const arg of expression.args) visitExpression(arg);
        return;
      case "unary":
        if (expression.operator === "&") {
          visitLvalueIndexes(expression.argument);
          return;
        }
        visitExpression(expression.argument);
        return;
      case "binary":
        visitExpression(expression.left);
        visitExpression(expression.right);
        return;
      case "conditional":
        visitExpression(expression.condition);
        visitExpression(expression.consequent);
        visitExpression(expression.alternate);
        return;
      case "assignment":
        if (expression.operator !== "=") record(expression.left, "read");
        visitLvalueIndexes(expression.left);
        visitExpression(expression.right);
        record(expression.left, "write");
        return;
      case "update":
        record(expression.argument, "read");
        visitLvalueIndexes(expression.argument);
        record(expression.argument, "write");
        return;
      case "sequence":
        for (const item of expression.expressions) visitExpression(item);
        return;
    }
  };
  const visit = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      switch (item.kind) {
        case "var":
          if (item.init) visitExpression(item.init);
          break;
        case "expr":
          visitExpression(item.expression);
          break;
        case "if":
          visitExpression(item.condition);
          visit(item.consequent);
          if (item.alternate) visit(item.alternate);
          break;
        case "for":
          if (item.init?.kind === "var") {
            if (item.init.init) visitExpression(item.init.init);
          } else if (item.init) {
            visitExpression(item.init);
          }
          if (item.condition) visitExpression(item.condition);
          visit(item.body);
          if (item.update) visitExpression(item.update);
          break;
        case "while":
          visitExpression(item.condition);
          visit(item.body);
          break;
        case "do-while":
          visit(item.body);
          visitExpression(item.condition);
          break;
        case "return":
          if (item.value) visitExpression(item.value);
          break;
        case "dim3":
        case "cooperative-group":
        case "kernel-launch":
        case "asm":
        case "continue":
        case "break":
          break;
      }
    }
  };
  visit(statements);
  return accesses;
}

function expressionRoot(expression: CudaLiteExpression): string | undefined {
  switch (expression.kind) {
    case "identifier":
      return expression.name;
    case "member":
      return expressionRoot(expression.object);
    case "index":
      return expressionRoot(expression.target);
    case "unary":
      return expression.operator === "*" || expression.operator === "&" ? expressionRoot(expression.argument) : undefined;
    default:
      return undefined;
  }
}

function phaseHasReturn(statements: readonly CudaLiteStatement[]): boolean {
  let hasReturn = false;
  visitStatements(statements, (statement) => {
    if (statement.kind === "return") hasReturn = true;
  });
  return hasReturn;
}

function visitStatements(
  statements: readonly CudaLiteStatement[],
  visit: (statement: CudaLiteStatement) => void,
): void {
  for (const statement of statements) {
    visit(statement);
    if (statement.kind === "if") {
      visitStatements(statement.consequent, visit);
      if (statement.alternate) visitStatements(statement.alternate, visit);
    }
    if (statement.kind === "for" || statement.kind === "while" || statement.kind === "do-while" || statement.kind === "block") {
      visitStatements(statement.body, visit);
    }
  }
}
