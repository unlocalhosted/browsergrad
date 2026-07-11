import {
  isSemanticKernelIrOperation,
  semanticExpressionChildren,
  semanticOperationExpressions,
} from "./semantic_ir_walk.js";
import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";

export type SemanticGridSyncPhasePlan =
  | {
      readonly supported: true;
      readonly phases: readonly SemanticKernelIrModule[];
    }
  | {
      readonly supported: false;
      readonly reason: string;
      readonly phases: readonly [];
    };

export function createSemanticGridSyncPhasePlan(
  ir: SemanticKernelIrModule,
): SemanticGridSyncPhasePlan {
  const groups = semanticCooperativeGroups(ir.operations);
  const topLevelSyncs = new Set(
    ir.operations
      .filter((operation) => isSemanticGridSyncOperation(operation, groups))
      .map((operation) => operation.span.start),
  );
  const allSyncs = collectSemanticGridSyncOperations(ir.operations, groups);
  if (allSyncs.length === 0) return { supported: true, phases: [ir] };
  if (topLevelSyncs.size !== allSyncs.length) {
    return {
      supported: false,
      reason: "grid.sync() must be a top-level uniform statement for WebGPU phase splitting",
      phases: [],
    };
  }

  const rawPhases: SemanticKernelIrOperation[][] = [[]];
  for (const operation of ir.operations) {
    if (isSemanticGridSyncOperation(operation, groups)) {
      rawPhases.push([]);
      continue;
    }
    rawPhases[rawPhases.length - 1]!.push(operation);
  }

  const phaseReplay = semanticReplayableDeclarationsForPhases(rawPhases, ir);
  const safety = validateSemanticGridSyncPhaseSafety(rawPhases, ir, phaseReplay);
  if (safety) return { supported: false, reason: safety, phases: [] };

  const declarations = ir.operations.filter((operation): operation is Extract<SemanticKernelIrOperation, { readonly kind: "cooperative-group-declare" }> =>
    operation.kind === "cooperative-group-declare",
  );
  return {
    supported: true,
    phases: rawPhases.map((phase, index) => ({
      ...ir,
      name: `${ir.name}_grid_phase_${index}`,
      operations: withSemanticCooperativeGroupMetadata(declarations, phaseReplay[index] ?? [], phase),
    })),
  };
}

export function isSemanticGridSyncOperation(
  operation: SemanticKernelIrOperation,
  _groups: ReadonlyMap<string, string>,
): boolean {
  return operation.kind === "barrier" && operation.scope === "grid";
}

function semanticCooperativeGroups(
  operations: readonly SemanticKernelIrOperation[],
): ReadonlyMap<string, string> {
  const groups = new Map<string, string>();
  visitSemanticOperations(operations, (operation) => {
    if (operation.kind === "cooperative-group-declare") {
      groups.set(operation.declaration.name, operation.declaration.groupKind);
    }
  });
  return groups;
}

function collectSemanticGridSyncOperations(
  operations: readonly SemanticKernelIrOperation[],
  groups: ReadonlyMap<string, string>,
): readonly SemanticKernelIrOperation[] {
  const out: SemanticKernelIrOperation[] = [];
  visitSemanticOperations(operations, (operation) => {
    if (isSemanticGridSyncOperation(operation, groups)) out.push(operation);
  });
  return out;
}

function validateSemanticGridSyncPhaseSafety(
  phases: readonly (readonly SemanticKernelIrOperation[])[],
  ir: SemanticKernelIrModule,
  phaseReplay: readonly (readonly Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>[])[],
): string | undefined {
  const previousLocals = new Set<string>();
  const previousSharedAccesses = new Set<string>();
  const sharedNames = new Set(ir.memory.filter((symbol) => symbol.addressSpace === "shared").map((symbol) => symbol.name));
  const globals = semanticGridSyncGlobals(ir);

  for (let phaseIndex = 0; phaseIndex < phases.length; phaseIndex++) {
    const phase = phases[phaseIndex]!;
    if (phaseIndex < phases.length - 1 && semanticPhaseHasReturn(phase)) {
      return "return before grid.sync() cannot be replayed safely across WebGPU phases";
    }

    const refs = semanticIdentifiersReferencedBy(phase);
    const sharedAccesses = firstSemanticSharedAccessesByName(phase, sharedNames);
    const currentLocals = semanticLocalsDeclaredBy(phase);
    const replayedLocals = new Set((phaseReplay[phaseIndex] ?? []).map((operation) => operation.target.name));
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

function semanticGridSyncGlobals(ir: SemanticKernelIrModule): ReadonlySet<string> {
  return new Set([
    ...ir.params.map((param) => param.name),
    ...ir.memory.filter((symbol) => symbol.addressSpace !== "local" && symbol.addressSpace !== "shared").map((symbol) => symbol.name),
    ...ir.functions.map((fn) => fn.name),
    "threadIdx",
    "blockIdx",
    "blockDim",
    "gridDim",
    "nullptr",
    "NULL",
  ]);
}

function withSemanticCooperativeGroupMetadata(
  declarations: readonly Extract<SemanticKernelIrOperation, { readonly kind: "cooperative-group-declare" }>[],
  replay: readonly Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>[],
  phase: readonly SemanticKernelIrOperation[],
): readonly SemanticKernelIrOperation[] {
  const phaseGroups = new Set(phase.filter((operation) => operation.kind === "cooperative-group-declare").map((operation) => operation.declaration.name));
  const phaseLocals = semanticLocalsDeclaredBy(phase);
  return [
    ...declarations.filter((operation) => !phaseGroups.has(operation.declaration.name)),
    ...replay.filter((operation) => !phaseLocals.has(operation.target.name)),
    ...phase,
  ];
}

function semanticReplayableDeclarationsForPhases(
  phases: readonly (readonly SemanticKernelIrOperation[])[],
  ir: SemanticKernelIrModule,
): readonly (readonly Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>[] )[] {
  const globals = semanticGridSyncGlobals(ir);
  const available = new Map<string, Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>>();
  const out: Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>[][] = [];
  for (const phase of phases) {
    out.push([...semanticReplayClosureFor(phase, available)]);
    for (const operation of topLevelSemanticReplayableDeclarations(phase, globals, new Set(available.keys()))) {
      available.set(operation.target.name, operation);
    }
  }
  return out;
}

function semanticReplayClosureFor(
  phase: readonly SemanticKernelIrOperation[],
  available: ReadonlyMap<string, Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>>,
): readonly Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>[] {
  const needed = semanticIdentifiersReferencedBy(phase);
  const emitted = new Set<string>();
  const visiting = new Set<string>();
  const ordered: Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>[] = [];
  const emit = (name: string): void => {
    if (emitted.has(name) || visiting.has(name)) return;
    const declaration = available.get(name);
    if (!declaration) return;
    visiting.add(name);
    if (declaration.init) {
      for (const ref of semanticIdentifiersReferencedByExpression(declaration.init)) emit(ref);
    }
    visiting.delete(name);
    ordered.push(declaration);
    emitted.add(name);
  };
  for (const name of needed) emit(name);
  return ordered;
}

function topLevelSemanticReplayableDeclarations(
  phase: readonly SemanticKernelIrOperation[],
  globals: ReadonlySet<string>,
  priorReplayable: ReadonlySet<string>,
): readonly Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>[] {
  const replayable = new Set(priorReplayable);
  const declarations: Extract<SemanticKernelIrOperation, { readonly kind: "declare" }>[] = [];
  for (const operation of phase) {
    if (
      operation.kind === "declare" &&
      operation.target.addressSpace === "local" &&
      !operation.target.pointer &&
      operation.target.dimensions.length === 0 &&
      operation.init &&
      isReplayableSemanticExpression(operation.init, globals, replayable)
    ) {
      declarations.push(operation);
      replayable.add(operation.target.name);
    }
  }
  return declarations;
}

function isReplayableSemanticExpression(
  expression: SemanticExpression,
  globals: ReadonlySet<string>,
  replayableLocals: ReadonlySet<string>,
): boolean {
  switch (expression.kind) {
    case "literal":
      return true;
    case "initializer":
      return expression.elements.every((element) => isReplayableSemanticExpression(element, globals, replayableLocals));
    case "symbol":
      return globals.has(expression.name) || replayableLocals.has(expression.name);
    case "member":
      return isReplayableSemanticExpression(expression.object, globals, replayableLocals);
    case "cast":
      return !expression.pointer && isReplayableSemanticExpression(expression.expression, globals, replayableLocals);
    case "unary":
      return expression.operator !== "*" && expression.operator !== "&" &&
        isReplayableSemanticExpression(expression.argument, globals, replayableLocals);
    case "binary":
      return isReplayableSemanticExpression(expression.left, globals, replayableLocals) &&
        isReplayableSemanticExpression(expression.right, globals, replayableLocals);
    case "conditional":
      return isReplayableSemanticExpression(expression.condition, globals, replayableLocals) &&
        isReplayableSemanticExpression(expression.consequent, globals, replayableLocals) &&
        isReplayableSemanticExpression(expression.alternate, globals, replayableLocals);
    case "index":
    case "call":
    case "texture-read":
    case "surface-read":
    case "assignment":
    case "update":
    case "sequence":
      return false;
  }
}

function semanticLocalsDeclaredBy(operations: readonly SemanticKernelIrOperation[]): ReadonlySet<string> {
  const locals = new Set<string>();
  visitSemanticOperations(operations, (operation) => {
    if (operation.kind === "declare" && operation.target.addressSpace === "local") locals.add(operation.target.name);
    if (operation.kind === "dim3-declare") locals.add(operation.name);
  });
  return locals;
}

function semanticIdentifiersReferencedBy(operations: readonly SemanticKernelIrOperation[]): ReadonlySet<string> {
  const refs = new Set<string>();
  visitSemanticOperations(operations, (operation) => {
    for (const expression of semanticOperationExpressions(operation)) {
      for (const name of semanticIdentifiersReferencedByExpression(expression)) refs.add(name);
    }
    for (const memoryRef of semanticOperationMemoryRefs(operation)) {
      refs.add(memoryRef.base);
      for (const index of memoryRef.indices) {
        for (const name of semanticIdentifiersReferencedByExpression(index)) refs.add(name);
      }
    }
  });
  return refs;
}

function semanticIdentifiersReferencedByExpression(expression: SemanticExpression): ReadonlySet<string> {
  const refs = new Set<string>();
  const visit = (item: SemanticExpression): void => {
    if (item.kind === "symbol") refs.add(item.name);
    for (const child of semanticExpressionChildren(item)) visit(child);
  };
  visit(expression);
  return refs;
}

type SharedAccess = "read" | "write";

function firstSemanticSharedAccessesByName(
  operations: readonly SemanticKernelIrOperation[],
  sharedNames: ReadonlySet<string>,
): ReadonlyMap<string, SharedAccess> {
  const accesses = new Map<string, SharedAccess>();
  const recordRef = (ref: SemanticMemoryRef, access: SharedAccess): void => {
    if (ref.addressSpace === "shared" && sharedNames.has(ref.base) && !accesses.has(ref.base)) accesses.set(ref.base, access);
  };
  const visitExpression = (expression: SemanticExpression, access: SharedAccess = "read"): void => {
    if (expression.kind === "index" && expression.addressSpace === "shared") {
      const root = semanticExpressionRoot(expression);
      if (root && sharedNames.has(root) && !accesses.has(root)) accesses.set(root, access);
    }
    if (expression.kind === "assignment") {
      if (expression.operator !== "=") visitExpression(expression.target, "read");
      visitExpression(expression.target, "write");
      visitExpression(expression.value);
      return;
    }
    if (expression.kind === "update") {
      visitExpression(expression.argument, "read");
      visitExpression(expression.argument, "write");
      return;
    }
    for (const child of semanticExpressionChildren(expression)) visitExpression(child);
  };
  visitSemanticOperations(operations, (operation) => {
    if (operation.kind === "load") recordRef(operation.source, "read");
    if (operation.kind === "store") {
      if (operation.operator !== "=") recordRef(operation.target, "read");
      recordRef(operation.target, "write");
      for (const read of operation.reads) recordRef(read, "read");
    }
    if (operation.kind === "atomic" && operation.target) {
      recordRef(operation.target, "read");
      recordRef(operation.target, "write");
    }
    for (const expression of semanticOperationExpressions(operation)) visitExpression(expression);
  });
  return accesses;
}

function semanticExpressionRoot(expression: SemanticExpression): string | undefined {
  if (expression.kind === "symbol") return expression.name;
  if (expression.kind === "member" || expression.kind === "index") return semanticExpressionRoot(expression.kind === "member" ? expression.object : expression.target);
  if (expression.kind === "unary" && (expression.operator === "*" || expression.operator === "&")) return semanticExpressionRoot(expression.argument);
  return undefined;
}

function semanticPhaseHasReturn(operations: readonly SemanticKernelIrOperation[]): boolean {
  let hasReturn = false;
  visitSemanticOperations(operations, (operation) => {
    if (operation.kind === "return") hasReturn = true;
  });
  return hasReturn;
}

function semanticOperationMemoryRefs(operation: SemanticKernelIrOperation): readonly SemanticMemoryRef[] {
  if (operation.kind === "load") return [operation.source];
  if (operation.kind === "store") return [operation.target, ...operation.reads];
  if (operation.kind === "atomic" && operation.target) return [operation.target];
  if (operation.kind === "call") return operation.reads;
  return [];
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
      if (operation.init && isSemanticKernelIrOperation(operation.init)) visitSemanticOperations([operation.init], visit);
      visitSemanticOperations(operation.body, visit);
    }
  }
}
