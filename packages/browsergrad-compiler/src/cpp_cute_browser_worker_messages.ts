export const CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL =
  "browsergrad.compiler.cpp-cute.browser-worker-controller@2";

export interface CppCuteBrowserWorkerControllerTerminalMessage {
  readonly kind: "browsergrad-cpp-cute-worker-terminal";
  readonly version: 2;
  readonly controllerProtocol: typeof CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL;
  readonly invocationId: string;
  readonly invocationNonceSha256: string;
  readonly controlBytes: Uint8Array;
  readonly artifactBytes: Uint8Array;
}

export type CppCuteBrowserWorkerEntryFailurePhase =
  | "runtime-adoption"
  | "runtime-start";

/**
 * Authenticated only by the host controller's ownership of the exact Worker
 * event source. This reports infrastructure failure; it is never a compiler
 * result frame or lowering authority.
 */
export interface CppCuteBrowserWorkerControllerFailureMessage {
  readonly kind: "browsergrad-cpp-cute-worker-failure";
  readonly version: 2;
  readonly controllerProtocol: typeof CPP_CUTE_BROWSER_WORKER_CONTROLLER_PROTOCOL;
  readonly invocationId: string;
  readonly invocationNonceSha256: string;
  readonly phase: CppCuteBrowserWorkerEntryFailurePhase;
  readonly failureCode: string;
  readonly failurePath: string;
  readonly failureDetail: string;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
}

export type CppCuteBrowserWorkerControllerInboundMessage =
  | CppCuteBrowserWorkerControllerTerminalMessage
  | CppCuteBrowserWorkerControllerFailureMessage;
