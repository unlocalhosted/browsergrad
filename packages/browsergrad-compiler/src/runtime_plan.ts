import { walkCudaLiteExpressions } from "./ast_queries.js";
import type {
  CompiledCudaLiteKernel,
  CudaLiteExpression,
  CudaLiteStatement,
  KernelIrModule,
  SourceSpan,
} from "./types.js";

export type CudaRuntimeOperationKind =
  | "device-launch"
  | "device-sync"
  | "grid-sync"
  | "peer-copy";

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
  compiled: Pick<CompiledCudaLiteKernel, "ir">,
): CudaRuntimePlan {
  const operations = collectRuntimeOperations(compiled.ir.body);
  return {
    operations,
    requiresHostOrchestration: operations.length > 0,
    canRunSingleDispatchWebGpu: operations.length === 0,
    referenceAvailable: operations.every((operation) => REFERENCE_RUNTIME_OPERATIONS.has(operation.kind)),
  };
}

const REFERENCE_RUNTIME_OPERATIONS: ReadonlySet<CudaRuntimeOperationKind> = new Set([
  "device-launch",
  "device-sync",
  "grid-sync",
  "peer-copy",
]);

export function createCudaGridSyncPhasePlan(ir: KernelIrModule): CudaGridSyncPhasePlan {
  const runtimePlan = createCudaRuntimePlan({ ir });
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

  const safety = validateGridSyncPhaseSafety(rawPhases, ir);
  if (safety) return { supported: false, reason: safety, modules: [] };

  const groupDeclarations = ir.body.filter((statement): statement is Extract<CudaLiteStatement, { kind: "cooperative-group" }> =>
    statement.kind === "cooperative-group"
  );
  return {
    supported: true,
    modules: rawPhases.map((phase, index): KernelIrModule => ({
      ...ir,
      name: `${ir.name}_grid_phase_${index}`,
      body: withCooperativeGroupMetadata(groupDeclarations, phase),
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
    if (expression.callee.name === "cudaDeviceSynchronize") {
      return {
        kind: "device-sync",
        span: expression.span,
        label: "cudaDeviceSynchronize()",
      };
    }
    if (expression.callee.name === "cudaMemcpyPeerAsync") {
      return {
        kind: "peer-copy",
        span: expression.span,
        label: "cudaMemcpyPeerAsync(...)",
      };
    }
  }
  if (
    expression.callee.kind !== "member" ||
    expression.callee.property !== "sync" ||
    expression.callee.object.kind !== "identifier" ||
    cooperativeGroups.get(expression.callee.object.name) !== "grid"
  ) {
    return undefined;
  }
  return {
    kind: "grid-sync",
    span: expression.span,
    label: "grid.sync()",
  };
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
  return expression.kind === "call" &&
    expression.callee.kind === "member" &&
    expression.callee.property === "sync" &&
    expression.callee.object.kind === "identifier" &&
    cooperativeGroups.get(expression.callee.object.name) === "grid";
}

function validateGridSyncPhaseSafety(
  phases: readonly (readonly CudaLiteStatement[])[],
  ir: KernelIrModule,
): string | undefined {
  const previousLocals = new Set<string>();
  const previousSharedRefs = new Set<string>();
  const sharedNames = new Set(ir.sharedDeclarations.map((declaration) => declaration.name));
  const globals = new Set([
    ...ir.params.map((param) => param.name),
    ...ir.constants.map((constant) => constant.name),
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
    const currentLocals = localsDeclaredBy(phase);
    for (const name of refs) {
      if (previousLocals.has(name) && !currentLocals.has(name) && !globals.has(name)) {
        return `local '${name}' crosses grid.sync(); WebGPU phases cannot preserve private thread state`;
      }
      if (sharedNames.has(name) && previousSharedRefs.has(name)) {
        return `shared memory '${name}' crosses grid.sync(); WebGPU workgroup memory cannot persist across dispatches`;
      }
    }

    for (const name of refs) {
      if (sharedNames.has(name)) previousSharedRefs.add(name);
    }
    for (const name of currentLocals) previousLocals.add(name);
  }
  return undefined;
}

function withCooperativeGroupMetadata(
  declarations: readonly Extract<CudaLiteStatement, { kind: "cooperative-group" }>[],
  phase: readonly CudaLiteStatement[],
): readonly CudaLiteStatement[] {
  const phaseGroups = new Set(phase.filter((statement) => statement.kind === "cooperative-group").map((statement) => statement.name));
  return [
    ...declarations.filter((declaration) => !phaseGroups.has(declaration.name)),
    ...phase,
  ];
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
    if (statement.kind === "for") visitStatements(statement.body, visit);
  }
}
