import type { KernelDevice } from "./types.js";
import {
  prepareWgslKernelProgramSequence,
  WgslPipelineCreationError,
  WgslShaderCreationError,
  type WgslKernelRunInput,
  type WgslKernelRunResult,
  type WgslKernelSequenceStep,
} from "./wgsl_program.js";

export type SemanticWebGpuHostIssue =
  | "shader"
  | "pipeline"
  | "validation"
  | "out-of-memory"
  | "internal"
  | "device-lost"
  | "error-scope"
  | "execution";

export class SemanticWebGpuHostError extends Error {
  constructor(
    readonly issue: SemanticWebGpuHostIssue,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "SemanticWebGpuHostError";
  }
}

/**
 * Owns pipeline preparation, dispatch, readback, LIFO error scopes, device-loss
 * races, and prepared-sequence cleanup for semantic host execution paths.
 */
export async function runPreparedSemanticWebGpuHostReadback(
  device: KernelDevice,
  steps: readonly WgslKernelSequenceStep[],
  input: WgslKernelRunInput,
): Promise<WgslKernelRunResult> {
  const gpu = device.gpu;
  pushErrorScopes(gpu);
  const preparation = prepareWgslKernelProgramSequence(device, steps, input).then(
    (value) => ({ kind: "completed", value }) as const,
    (error: unknown) => ({ kind: "failed", error }) as const,
  );
  const preparationScopes = popAllErrorScopes(gpu);
  const preparationOutcome = await Promise.race([
    Promise.all([preparation, preparationScopes]).then(
      ([outcome, scopes]) => ({ kind: "settled", outcome, scopes }) as const,
    ),
    gpu.lost.then((info) => ({ kind: "lost", info }) as const),
  ]);
  if (preparationOutcome.kind === "lost") {
    void preparation.then((outcome) => {
      if (outcome.kind === "completed") outcome.value.destroy();
    });
    throw new SemanticWebGpuHostError(
      "device-lost",
      "$.device",
      `WebGPU device lost (${preparationOutcome.info.reason}): ${preparationOutcome.info.message}`,
    );
  }
  if (preparationOutcome.outcome.kind !== "completed") {
    classifyPhase(preparationOutcome.outcome, preparationOutcome.scopes, "$.pipeline");
    throw new SemanticWebGpuHostError(
      "internal",
      "$.pipeline",
      "pipeline preparation failed without a classified diagnostic",
    );
  }
  const sequence = preparationOutcome.outcome.value;
  try {
    classifyPhase(preparationOutcome.outcome, preparationOutcome.scopes, "$.pipeline");
  } catch (error) {
    sequence.destroy();
    throw error;
  }

  pushErrorScopes(gpu);
  const execution = sequence.run().then(
    (value) => ({ kind: "completed", value }) as const,
    (error: unknown) => ({ kind: "failed", error }) as const,
  );
  const executionScopes = popAllErrorScopes(gpu);
  try {
    const executionOutcome = await Promise.race([
      Promise.all([execution, executionScopes]).then(
        ([outcome, scopes]) => ({ kind: "settled", outcome, scopes }) as const,
      ),
      gpu.lost.then((info) => ({ kind: "lost", info }) as const),
    ]);
    if (executionOutcome.kind === "lost") {
      throw new SemanticWebGpuHostError(
        "device-lost",
        "$.device",
        `WebGPU device lost (${executionOutcome.info.reason}): ${executionOutcome.info.message}`,
      );
    }
    classifyPhase(executionOutcome.outcome, executionOutcome.scopes, "$.dispatch");
    if (executionOutcome.outcome.kind !== "completed") {
      throw new SemanticWebGpuHostError(
        "internal",
        "$.dispatch",
        "dispatch failed without a classified diagnostic",
      );
    }
    return executionOutcome.outcome.value;
  } finally {
    sequence.destroy();
  }
}

type PhaseOutcome<T> =
  | { readonly kind: "completed"; readonly value: T }
  | { readonly kind: "failed"; readonly error: unknown };

interface ErrorScopeResults {
  readonly validation: ErrorScopeAttempt;
  readonly outOfMemory: ErrorScopeAttempt;
  readonly internal: ErrorScopeAttempt;
}

interface ErrorScopeAttempt {
  readonly value: GPUError | null;
  readonly failure?: unknown;
}

function pushErrorScopes(gpu: GPUDevice): void {
  gpu.pushErrorScope("internal");
  gpu.pushErrorScope("out-of-memory");
  gpu.pushErrorScope("validation");
}

async function popAllErrorScopes(gpu: GPUDevice): Promise<ErrorScopeResults> {
  const validation = popErrorScopeAttempt(gpu);
  const outOfMemory = popErrorScopeAttempt(gpu);
  const internal = popErrorScopeAttempt(gpu);
  const [validationResult, outOfMemoryResult, internalResult] = await Promise.all([
    validation,
    outOfMemory,
    internal,
  ]);
  return Object.freeze({
    validation: validationResult,
    outOfMemory: outOfMemoryResult,
    internal: internalResult,
  });
}

async function popErrorScopeAttempt(gpu: GPUDevice): Promise<ErrorScopeAttempt> {
  try {
    return { value: await gpu.popErrorScope() };
  } catch (failure) {
    return { value: null, failure };
  }
}

function classifyPhase<T>(
  outcome: PhaseOutcome<T>,
  scopes: ErrorScopeResults,
  path: string,
): void {
  const scopeFailure = scopes.validation.failure
    ?? scopes.outOfMemory.failure
    ?? scopes.internal.failure;
  if (scopeFailure !== undefined) {
    throw new SemanticWebGpuHostError(
      "error-scope",
      "$.device.errorScope",
      message(scopeFailure),
      { cause: scopeFailure },
    );
  }
  if (scopes.outOfMemory.value !== null) {
    throw new SemanticWebGpuHostError(
      "out-of-memory",
      path,
      scopes.outOfMemory.value.message,
    );
  }
  if (scopes.internal.value !== null) {
    throw new SemanticWebGpuHostError("internal", path, scopes.internal.value.message);
  }
  if (outcome.kind === "failed") {
    if (outcome.error instanceof WgslShaderCreationError) {
      throw new SemanticWebGpuHostError(
        "shader",
        "$.shaderModule",
        outcome.error.message,
        { cause: outcome.error },
      );
    }
    if (outcome.error instanceof WgslPipelineCreationError) {
      throw new SemanticWebGpuHostError(
        "pipeline",
        "$.pipeline",
        outcome.error.message,
        { cause: outcome.error },
      );
    }
  }
  if (scopes.validation.value !== null) {
    throw new SemanticWebGpuHostError("validation", path, scopes.validation.value.message);
  }
  if (outcome.kind === "failed") {
    throw new SemanticWebGpuHostError(
      "execution",
      path,
      message(outcome.error),
      { cause: outcome.error },
    );
  }
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
