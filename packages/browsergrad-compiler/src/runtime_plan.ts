import { walkCudaLiteExpressions } from "./ast_queries.js";
import type {
  CompiledCudaLiteKernel,
  CudaLiteExpression,
  CudaLiteStatement,
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
