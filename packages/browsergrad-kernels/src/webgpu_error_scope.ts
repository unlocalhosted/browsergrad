export type ScopedWebGpuIssueKind =
  | "validation"
  | "out-of-memory"
  | "internal"
  | "device-lost"
  | "error-scope"
  | "operation";

export class ScopedWebGpuIssueError extends Error {
  constructor(
    readonly kind: ScopedWebGpuIssueKind,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "ScopedWebGpuIssueError";
  }
}

export interface ScopedWebGpuIssueOptions<T> {
  readonly cleanup?: (value: T) => void;
  /** Optional async completion started synchronously by `issue` (for mapAsync). */
  readonly completion?: (value: T) => Promise<unknown>;
}

/**
 * Capture errors from a synchronous WebGPU issue phase. Validation, OOM, and
 * internal scopes are pushed outer-to-inner immediately before `issue`; every
 * pop is invoked synchronously in the finally path in strict LIFO order. No
 * value escapes until every pop has settled or device loss wins the race.
 */
export async function issueWithWebGpuErrorScopes<T>(
  gpu: Pick<GPUDevice, "pushErrorScope" | "popErrorScope" | "lost">,
  path: string,
  issue: () => T,
  options: ScopedWebGpuIssueOptions<T> = {},
): Promise<T> {
  const pushed: GPUErrorFilter[] = [];
  let value: T | undefined;
  let issueCompleted = false;
  let operationError: unknown;
  let popAttempts: readonly PopAttempt[] = [];
  try {
    for (const scope of ["internal", "out-of-memory", "validation"] as const) {
      gpu.pushErrorScope(scope);
      pushed.push(scope);
    }
    value = issue();
    issueCompleted = true;
  } catch (error) {
    operationError = error;
  } finally {
    // popErrorScope removes a scope when called. Invoke every pop before
    // yielding so no later issue can be accidentally captured by these scopes.
    popAttempts = Object.freeze(
      [...pushed].reverse().map((scope) => ({
        scope,
        result: popAttempt(gpu),
      })),
    );
  }

  const scopeSettlement = Promise.all(popAttempts.map(({ result }) => result));
  let operationSettlement: Promise<OperationSettlement>;
  if (issueCompleted && options.completion !== undefined) {
    try {
      operationSettlement = Promise.resolve(options.completion(value as T)).then(
        () => ({ kind: "completed" as const }),
        (error: unknown) => ({ kind: "failed" as const, error }),
      );
    } catch (error) {
      operationSettlement = Promise.resolve({ kind: "failed" as const, error });
    }
  } else {
    operationSettlement = Promise.resolve({ kind: "completed" as const });
  }
  const settlement = await Promise.race([
    gpu.lost.then((info) => ({ kind: "lost" as const, info })),
    Promise.all([scopeSettlement, operationSettlement]).then(
      ([scopes, operation]) => ({ kind: "scopes" as const, scopes, operation }),
    ),
  ]);
  if (settlement.kind === "lost") {
    cleanup(value, issueCompleted, options.cleanup);
    throw new ScopedWebGpuIssueError(
      "device-lost",
      path,
      `WebGPU device lost (${settlement.info.reason}): ${settlement.info.message}`,
    );
  }

  for (const [index, attempt] of settlement.scopes.entries()) {
    if (attempt.failure !== undefined) {
      cleanup(value, issueCompleted, options.cleanup);
      throw new ScopedWebGpuIssueError(
        "error-scope",
        path,
        `popErrorScope(${popAttempts[index]!.scope}) failed: ${message(attempt.failure)}`,
        { cause: attempt.failure },
      );
    }
  }
  for (const [index, attempt] of settlement.scopes.entries()) {
    if (attempt.value !== null) {
      cleanup(value, issueCompleted, options.cleanup);
      const scope = popAttempts[index]!.scope;
      throw new ScopedWebGpuIssueError(
        scope,
        path,
        `${scope} GPU error: ${attempt.value.message}`,
      );
    }
  }
  if (settlement.operation.kind === "failed") {
    cleanup(value, issueCompleted, options.cleanup);
    throw new ScopedWebGpuIssueError(
      "operation",
      path,
      message(settlement.operation.error),
      { cause: settlement.operation.error },
    );
  }
  if (operationError !== undefined) {
    cleanup(value, issueCompleted, options.cleanup);
    throw new ScopedWebGpuIssueError(
      "operation",
      path,
      message(operationError),
      { cause: operationError },
    );
  }
  return value as T;
}

interface PopAttempt {
  readonly scope: GPUErrorFilter;
  readonly result: Promise<PopResult>;
}

type OperationSettlement =
  | Readonly<{ kind: "completed" }>
  | Readonly<{ kind: "failed"; error: unknown }>;

interface PopResult {
  readonly value: GPUError | null;
  readonly failure?: unknown;
}

function popAttempt(
  gpu: Pick<GPUDevice, "popErrorScope">,
): Promise<PopResult> {
  try {
    return gpu.popErrorScope().then(
      (value) => ({ value }),
      (failure: unknown) => ({ value: null, failure }),
    );
  } catch (failure) {
    return Promise.resolve({ value: null, failure });
  }
}

function cleanup<T>(
  value: T | undefined,
  issueCompleted: boolean,
  action: ((value: T) => void) | undefined,
): void {
  if (!issueCompleted || action === undefined) return;
  try {
    action(value as T);
  } catch {
    // Preserve the authoritative GPU/operation failure.
  }
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
