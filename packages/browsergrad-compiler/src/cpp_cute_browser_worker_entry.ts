import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL,
  type CppCuteBrowserWorkerControllerFailureMessage,
  type CppCuteBrowserWorkerControllerInboundMessage,
  type CppCuteBrowserWorkerControllerTerminalMessage,
  type CppCuteBrowserWorkerEntryFailurePhase,
} from "./cpp_cute_browser_worker_messages.js";
import {
  prepareCppCuteBrowserWorkerRuntimeBinding,
  startCppCuteBrowserWorkerRuntime,
} from "./cpp_cute_browser_worker_runtime.js";
import {
  discardCppCuteBrowserWorkerRealmInput,
  inspectCppCuteBrowserWorkerRealmInput,
  reconstructCppCuteBrowserWorkerTransfer,
  type PreparedCppCuteBrowserWorkerRealmInput,
  type CppCuteBrowserWorkerTransferMessage,
} from "./cpp_cute_browser_worker_transfer.js";

export const CPP_CUTE_BROWSER_WORKER_ENTRY_PROTOCOL =
  "browsergrad.compiler.cpp-cute.browser-worker-entry@1";

const FAILURE_CODE = /^BG-[A-Z0-9-]+$/u;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_AGGREGATE_ERROR = AggregateError;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_REGEXP_TEST = RegExp.prototype.test;
const NATIVE_REFLECT_APPLY = Reflect.apply;

export type CppCuteBrowserWorkerTerminalEmitter = (
  message: CppCuteBrowserWorkerControllerInboundMessage,
  transfer: readonly ArrayBuffer[],
) => void;

export interface CppCuteBrowserWorkerEntryMessageEvent {
  readonly data: unknown;
}

export type CppCuteBrowserWorkerEntryMessageListener = (
  event: CppCuteBrowserWorkerEntryMessageEvent,
) => void;

/** Captured dedicated-Worker effects. No fetch/import/module-loader authority. */
export interface CppCuteBrowserWorkerEntryScope {
  readonly addEventListener: (
    type: "message",
    listener: CppCuteBrowserWorkerEntryMessageListener,
  ) => void;
  readonly removeEventListener: (
    type: "message",
    listener: CppCuteBrowserWorkerEntryMessageListener,
  ) => void;
  readonly postMessage: (
    message: CppCuteBrowserWorkerControllerInboundMessage,
    transfer: readonly ArrayBuffer[],
  ) => void;
  readonly queueMicrotask: (callback: () => void) => void;
}

interface CppCuteBrowserWorkerEntryEffects {
  readonly addEventListener: CppCuteBrowserWorkerEntryScope["addEventListener"];
  readonly removeEventListener: CppCuteBrowserWorkerEntryScope["removeEventListener"];
  readonly postMessage: CppCuteBrowserWorkerEntryScope["postMessage"];
  readonly queueMicrotask: CppCuteBrowserWorkerEntryScope["queueMicrotask"];
}

/** Installs one one-shot launch listener in an already authenticated Worker. */
export function installCppCuteBrowserWorkerEntry(
  scope: CppCuteBrowserWorkerEntryScope,
): void {
  const effects = entryEffects(scope);
  const onMessage: CppCuteBrowserWorkerEntryMessageListener = (event) => {
    NATIVE_REFLECT_APPLY(effects.removeEventListener, scope, ["message", onMessage]);
    void handleCppCuteBrowserWorkerTransfer(
      event.data as CppCuteBrowserWorkerTransferMessage,
      (message, transfer) => NATIVE_REFLECT_APPLY(
        effects.postMessage,
        scope,
        [message, transfer],
      ),
    ).catch((cause: unknown) => {
      NATIVE_REFLECT_APPLY(effects.queueMicrotask, scope, [() => {
        throw cause;
      }]);
    });
  };
  NATIVE_REFLECT_APPLY(effects.addEventListener, scope, ["message", onMessage]);
}

/**
 * Consumes one already-transferred launch inside the dedicated Worker realm.
 * Invalid launch envelopes reject before a trusted invocation/nonce exists;
 * the installed entrypoint converts that rejection into a Worker error event.
 */
export async function handleCppCuteBrowserWorkerTransfer(
  message: CppCuteBrowserWorkerTransferMessage,
  emitTerminal: CppCuteBrowserWorkerTerminalEmitter,
): Promise<void> {
  const realmInput = await reconstructCppCuteBrowserWorkerTransfer(message);
  const identity = NATIVE_OBJECT_FREEZE({
    invocationId: realmInput.invocationId,
    invocationNonceSha256: realmInput.invocationNonceSha256,
  });
  let binding: Awaited<ReturnType<typeof prepareCppCuteBrowserWorkerRuntimeBinding>>;
  try {
    binding = await prepareCppCuteBrowserWorkerRuntimeBinding({ realmInput });
  } catch (cause) {
    const settledCause = settleUnadoptedRealmInput(realmInput, cause);
    emitTerminal(terminalFailure(settledCause, "runtime-adoption", identity), []);
    return;
  }
  let terminal: ReturnType<typeof terminalSuccess>;
  try {
    const result = await startCppCuteBrowserWorkerRuntime(binding);
    terminal = terminalSuccess(result, identity);
  } catch (cause) {
    emitTerminal(terminalFailure(cause, "runtime-start", identity), []);
    return;
  }
  emitTerminal(terminal.message, terminal.transfer);
}

export type CppCuteBrowserWorkerEntryErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-ENTRY-INTERNAL";

export class CppCuteBrowserWorkerEntryError extends Error {
  constructor(
    readonly code: CppCuteBrowserWorkerEntryErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserWorkerEntryError";
  }
}

function terminalFailure(
  cause: unknown,
  phase: CppCuteBrowserWorkerEntryFailurePhase,
  identity: {
    readonly invocationId: string;
    readonly invocationNonceSha256: string;
  },
): CppCuteBrowserWorkerControllerFailureMessage {
  const candidate = errorProjection(cause, phase);
  return NATIVE_OBJECT_FREEZE({
    kind: "browsergrad-cpp-cute-worker-failure",
    version: 1,
    controllerProtocol: CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL,
    invocationId: identity.invocationId,
    invocationNonceSha256: identity.invocationNonceSha256,
    phase,
    failureCode: candidate.code,
    failurePath: candidate.path,
    workerExecutionObserved: false,
    loweringAuthorityMinted: false,
  });
}

function terminalSuccess(
  result: Awaited<ReturnType<typeof startCppCuteBrowserWorkerRuntime>>,
  identity: {
    readonly invocationId: string;
    readonly invocationNonceSha256: string;
  },
): {
  readonly message: CppCuteBrowserWorkerControllerTerminalMessage;
  readonly transfer: readonly ArrayBuffer[];
} {
  if (result.kind !== "browsergrad-cpp-cute-runtime-result") {
    throw new CppCuteBrowserWorkerEntryError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-ENTRY-INTERNAL",
      "$.runtime.result.kind",
      "runtime returned an unknown terminal result kind",
    );
  }
  const controlBytes = terminalBytes(result.controlBytes, "$.runtime.result.controlBytes");
  const artifactBytes = terminalBytes(result.artifactBytes, "$.runtime.result.artifactBytes");
  return NATIVE_OBJECT_FREEZE({
    message: NATIVE_OBJECT_FREEZE({
      kind: "browsergrad-cpp-cute-worker-terminal",
      version: 1,
      controllerProtocol: CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL,
      invocationId: identity.invocationId,
      invocationNonceSha256: identity.invocationNonceSha256,
      controlBytes,
      artifactBytes,
    }),
    transfer: NATIVE_OBJECT_FREEZE([
      controlBytes.buffer as ArrayBuffer,
      artifactBytes.buffer as ArrayBuffer,
    ]),
  });
}

function terminalBytes(value: unknown, path: string): Uint8Array<ArrayBuffer> {
  try {
    const inspection = inspectUnsharedPlainUint8Array(value);
    return copyInspectedUnsharedUint8Array(value, inspection) as Uint8Array<ArrayBuffer>;
  } catch (cause) {
    throw new CppCuteBrowserWorkerEntryError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-ENTRY-INTERNAL",
      path,
      "runtime terminal bytes are not transferable plain bytes",
      { cause },
    );
  }
}

function settleUnadoptedRealmInput(
  realmInput: PreparedCppCuteBrowserWorkerRealmInput,
  primaryCause: unknown,
): unknown {
  try {
    if (inspectCppCuteBrowserWorkerRealmInput(realmInput).state === "prepared") {
      discardCppCuteBrowserWorkerRealmInput(realmInput);
    }
    return primaryCause;
  } catch (cleanupCause) {
    return new NATIVE_AGGREGATE_ERROR(
      [primaryCause, cleanupCause],
      "Worker runtime adoption and realm-input settlement failed",
    );
  }
}

function errorProjection(
  cause: unknown,
  phase: CppCuteBrowserWorkerEntryFailurePhase,
): { readonly code: string; readonly path: string } {
  if (typeof cause === "object" && cause !== null) {
    let code: unknown;
    let path: unknown;
    try {
      const descriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(cause);
      code = descriptors["code"]?.value;
      path = descriptors["path"]?.value;
    } catch {
      code = undefined;
      path = undefined;
    }
    if (typeof code === "string" && typeof path === "string" &&
        NATIVE_REFLECT_APPLY(NATIVE_REGEXP_TEST, FAILURE_CODE, [code]) === true &&
        validFailurePath(path)) {
      return NATIVE_OBJECT_FREEZE({ code, path });
    }
  }
  return NATIVE_OBJECT_FREEZE({
    code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-ENTRY-INTERNAL",
    path: phase === "runtime-adoption" ? "$.runtime.adoption" : "$.runtime.start",
  });
}

function validFailurePath(value: string): boolean {
  if (!value.startsWith("$") || value.length > 512) return false;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code <= 0x1f || code === 0x7f) return false;
  }
  return true;
}

function entryEffects(
  scope: CppCuteBrowserWorkerEntryScope,
): CppCuteBrowserWorkerEntryEffects {
  if (typeof scope !== "object" || scope === null) {
    throw new CppCuteBrowserWorkerEntryError(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-ENTRY-INTERNAL",
      "$.scope",
      "dedicated Worker scope is unavailable",
    );
  }
  const result: Record<string, unknown> = {};
  for (const key of [
    "addEventListener",
    "removeEventListener",
    "postMessage",
    "queueMicrotask",
  ] as const) {
    let operation: unknown;
    try {
      operation = scope[key];
    } catch (cause) {
      throw new CppCuteBrowserWorkerEntryError(
        "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-ENTRY-INTERNAL",
        `$.scope.${key}`,
        "dedicated Worker scope operation is unreadable",
        { cause },
      );
    }
    if (typeof operation !== "function") {
      throw new CppCuteBrowserWorkerEntryError(
        "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-ENTRY-INTERNAL",
        `$.scope.${key}`,
        "dedicated Worker scope operation is unavailable",
      );
    }
    result[key] = operation;
  }
  return NATIVE_OBJECT_FREEZE(result) as unknown as CppCuteBrowserWorkerEntryEffects;
}
