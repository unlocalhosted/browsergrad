import type {
  CppCuteBrowserWorkerTransferMessage,
} from "./cpp_cute_browser_worker_transfer.js";

const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_CONSTRUCT = Reflect.construct;
const CAPTURED_BLOB = typeof globalThis.Blob === "function" ? globalThis.Blob : undefined;
const CAPTURED_WORKER = typeof globalThis.Worker === "function" ? globalThis.Worker : undefined;
const CAPTURED_EVENT_TARGET = typeof globalThis.EventTarget === "function"
  ? globalThis.EventTarget
  : undefined;
const CAPTURED_URL = typeof globalThis.URL === "function" ? globalThis.URL : undefined;
const CAPTURED_PERFORMANCE = globalThis.performance;
const CAPTURED_CREATE_OBJECT_URL = CAPTURED_URL?.createObjectURL;
const CAPTURED_REVOKE_OBJECT_URL = CAPTURED_URL?.revokeObjectURL;
const CAPTURED_PERFORMANCE_NOW = CAPTURED_PERFORMANCE?.now;
const CAPTURED_SET_TIMEOUT = globalThis.setTimeout;
const CAPTURED_CLEAR_TIMEOUT = globalThis.clearTimeout;
const CAPTURED_WORKER_POST_MESSAGE = CAPTURED_WORKER?.prototype.postMessage;
const CAPTURED_WORKER_TERMINATE = CAPTURED_WORKER?.prototype.terminate;
const CAPTURED_ADD_EVENT_LISTENER = CAPTURED_EVENT_TARGET?.prototype.addEventListener;
const CAPTURED_REMOVE_EVENT_LISTENER = CAPTURED_EVENT_TARGET?.prototype.removeEventListener;

export interface CppCuteBrowserCapturedPlatformWorker {
  postMessage(message: CppCuteBrowserWorkerTransferMessage, transfer: readonly ArrayBuffer[]): void;
  addEventListener(type: "message", listener: (event: { readonly data: unknown }) => void): void;
  addEventListener(type: "error" | "messageerror", listener: (event: unknown) => void): void;
  removeEventListener(type: "message", listener: (event: { readonly data: unknown }) => void): void;
  removeEventListener(type: "error" | "messageerror", listener: (event: unknown) => void): void;
  terminate(): void;
}

/** Exact browser effect surface captured before any invocation input is read. */
export interface CppCuteBrowserCapturedPlatform {
  readonly authority: "module-captured-browser-platform-effects";
  readonly createModuleBlobUrl: (verifiedWorkerModuleBytes: Uint8Array) => string;
  readonly createModuleWorker: (
    blobUrl: string,
    workerName: string,
  ) => CppCuteBrowserCapturedPlatformWorker;
  readonly revokeModuleBlobUrl: (blobUrl: string) => void;
  readonly monotonicNowMilliseconds: () => number;
  readonly setHostTimeout: (callback: () => void, delayMilliseconds: number) => unknown;
  readonly clearHostTimeout: (handle: unknown) => void;
  readonly capturedBeforeInvocation: true;
  readonly callerEffectsAccepted: false;
}

const CAPTURED_PLATFORM = capturePlatform();

/** Returns the immutable module-captured platform, or undefined outside a browser Worker realm. */
export function getCppCuteBrowserCapturedPlatform(): CppCuteBrowserCapturedPlatform | undefined {
  return CAPTURED_PLATFORM;
}

function capturePlatform(): CppCuteBrowserCapturedPlatform | undefined {
  if (CAPTURED_BLOB === undefined || CAPTURED_WORKER === undefined ||
      CAPTURED_CREATE_OBJECT_URL === undefined || CAPTURED_REVOKE_OBJECT_URL === undefined ||
      CAPTURED_PERFORMANCE === undefined || CAPTURED_PERFORMANCE_NOW === undefined ||
      typeof CAPTURED_SET_TIMEOUT !== "function" || typeof CAPTURED_CLEAR_TIMEOUT !== "function" ||
      CAPTURED_WORKER_POST_MESSAGE === undefined || CAPTURED_WORKER_TERMINATE === undefined ||
      CAPTURED_ADD_EVENT_LISTENER === undefined || CAPTURED_REMOVE_EVENT_LISTENER === undefined) {
    return undefined;
  }
  const BlobConstructor = CAPTURED_BLOB;
  const WorkerConstructor = CAPTURED_WORKER;
  const createObjectUrl = CAPTURED_CREATE_OBJECT_URL;
  const revokeObjectUrl = CAPTURED_REVOKE_OBJECT_URL;
  const performanceObject = CAPTURED_PERFORMANCE;
  const performanceNow = CAPTURED_PERFORMANCE_NOW;
  const setTimeoutFunction = CAPTURED_SET_TIMEOUT;
  const clearTimeoutFunction = CAPTURED_CLEAR_TIMEOUT;
  const postMessage = CAPTURED_WORKER_POST_MESSAGE;
  const terminate = CAPTURED_WORKER_TERMINATE;
  const addEventListener = CAPTURED_ADD_EVENT_LISTENER;
  const removeEventListener = CAPTURED_REMOVE_EVENT_LISTENER;
  return Object.freeze({
    authority: "module-captured-browser-platform-effects",
    createModuleBlobUrl: (bytes: Uint8Array): string => {
      const blob = nativeConstruct(BlobConstructor, [
        [new Uint8Array(bytes)],
        { type: "text/javascript" },
      ]) as Blob;
      return nativeApply(createObjectUrl, CAPTURED_URL, [blob]) as string;
    },
    createModuleWorker: (blobUrl: string, workerName: string) => {
      const worker = nativeConstruct(WorkerConstructor, [
        blobUrl,
        { type: "module", name: workerName },
      ]) as Worker;
      return Object.freeze({
        postMessage: (
          message: CppCuteBrowserWorkerTransferMessage,
          transfer: readonly ArrayBuffer[],
        ): void => {
          nativeApply(postMessage, worker, [message, transfer]);
        },
        addEventListener: (type: string, listener: EventListener): void => {
          nativeApply(addEventListener, worker, [type, listener]);
        },
        removeEventListener: (type: string, listener: EventListener): void => {
          nativeApply(removeEventListener, worker, [type, listener]);
        },
        terminate: (): void => {
          nativeApply(terminate, worker, []);
        },
      }) as CppCuteBrowserCapturedPlatformWorker;
    },
    revokeModuleBlobUrl: (blobUrl: string): void => {
      nativeApply(revokeObjectUrl, CAPTURED_URL, [blobUrl]);
    },
    monotonicNowMilliseconds: (): number =>
      nativeApply(performanceNow, performanceObject, []) as number,
    setHostTimeout: (callback: () => void, delayMilliseconds: number): unknown =>
      nativeApply(setTimeoutFunction, globalThis, [callback, delayMilliseconds]),
    clearHostTimeout: (handle: unknown): void => {
      nativeApply(clearTimeoutFunction, globalThis, [handle]);
    },
    capturedBeforeInvocation: true,
    callerEffectsAccepted: false,
  });
}

function nativeApply(
  callable: (...arguments_: never[]) => unknown,
  receiver: unknown,
  arguments_: readonly unknown[],
): unknown {
  return NATIVE_REFLECT_APPLY(callable, receiver, arguments_);
}

function nativeConstruct(
  constructor: new (...arguments_: never[]) => unknown,
  arguments_: readonly unknown[],
): unknown {
  return NATIVE_REFLECT_CONSTRUCT(constructor, arguments_);
}
