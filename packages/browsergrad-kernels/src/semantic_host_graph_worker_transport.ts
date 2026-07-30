import {
  decodeHostGraphArtifact,
  type VerifiedHostGraphArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/graph";
import {
  decodeKernelArtifact,
  type VerifiedKernelArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  decodeLayoutArtifact,
  type VerifiedLayoutArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  assertJsonValue,
  copyVerifiedArtifactWireBytes,
  decodeWireJson,
  deepFreezeJson,
  hashSemanticArtifact,
  isJsonObject,
  parseWireU64,
  SemanticSchemaError,
  type JsonValue,
  type VerifiedArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createDevice } from "./device.js";
import {
  SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
  SEMANTIC_HOST_GRAPH_WEBGPU_PROFILE,
  SemanticHostGraphWebGpuError,
  prepareSemanticHostGraphWebGpu,
  runSemanticHostGraphWebGpu,
  type SemanticHostGraphWebGpuControlBinding,
  type SemanticHostGraphWebGpuExecutionRequest,
  type SemanticHostGraphWebGpuInputBinding,
  type SemanticHostGraphWebGpuOutputBinding,
  type SemanticHostGraphWebGpuTrace,
} from "./semantic_host_graph.js";

export const SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL =
  "browsergrad.host-graph.browser-worker-transport@1" as const;
export const SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION = 1;
export const SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACTS = 256;
export const SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACT_BYTES =
  64 * 1024 * 1024;
export const SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_INPUT_BYTES =
  256 * 1024 * 1024;
export const SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_OUTPUT_BYTES =
  256 * 1024 * 1024;
export const SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_WALL_TIME_MS = 300_000;

const DEFAULT_WALL_TIME_MS = 60_000;
const REQUEST_ID = /^bg\.host-graph\.worker\.[0-9a-f]{32}$/u;
const FAILURE_CODE = /^BG-[A-Z0-9-]+$/u;
const HASH = /^[0-9a-f]{64}$/u;
const WORKER_MODULE_URL = new URL(
  "./semantic_host_graph_worker_module.js",
  import.meta.url,
);

const OBJECT_PROTOTYPE = Object.prototype;
const ARRAY_PROTOTYPE = Array.prototype;
const UINT8_ARRAY_PROTOTYPE = Uint8Array.prototype;
const TYPED_ARRAY_PROTOTYPE =
  Object.getPrototypeOf(UINT8_ARRAY_PROTOTYPE) as object;
const OBJECT_GET_OWN_PROPERTY_DESCRIPTORS =
  Object.getOwnPropertyDescriptors;
const OBJECT_GET_OWN_PROPERTY_DESCRIPTOR =
  Object.getOwnPropertyDescriptor;
const OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const OBJECT_FREEZE = Object.freeze;
const REFLECT_OWN_KEYS = Reflect.ownKeys;
const REFLECT_APPLY = Reflect.apply;
const UINT8_SET = Uint8Array.prototype.set;
const UINT8_BUFFER_GETTER = requiredGetter(
  TYPED_ARRAY_PROTOTYPE,
  "buffer",
);
const UINT8_BYTE_OFFSET_GETTER = requiredGetter(
  TYPED_ARRAY_PROTOTYPE,
  "byteOffset",
);
const UINT8_BYTE_LENGTH_GETTER = requiredGetter(
  TYPED_ARRAY_PROTOTYPE,
  "byteLength",
);
const ARRAY_BUFFER_SLICE = ArrayBuffer.prototype.slice;
const SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER =
  typeof SharedArrayBuffer === "undefined"
    ? undefined
    : OBJECT_GET_OWN_PROPERTY_DESCRIPTOR(
        SharedArrayBuffer.prototype,
        "byteLength",
      )?.get;
const ABORTED_GETTER =
  typeof AbortSignal === "undefined"
    ? undefined
    : OBJECT_GET_OWN_PROPERTY_DESCRIPTOR(
        AbortSignal.prototype,
        "aborted",
      )?.get;
const CAPTURED_WORKER = globalThis.Worker;
const CAPTURED_SET_TIMEOUT = globalThis.setTimeout.bind(globalThis);
const CAPTURED_CLEAR_TIMEOUT = globalThis.clearTimeout.bind(globalThis);
const CAPTURED_GET_RANDOM_VALUES =
  globalThis.crypto?.getRandomValues.bind(globalThis.crypto);

export type SemanticHostGraphBrowserWorkerTransportErrorCode =
  | "BG-WEBGPU-GRAPH-WORKER-INVALID"
  | "BG-WEBGPU-GRAPH-WORKER-CAPABILITY"
  | "BG-WEBGPU-GRAPH-WORKER-RESOURCE-LIMIT"
  | "BG-WEBGPU-GRAPH-WORKER-CANCELLED"
  | "BG-WEBGPU-GRAPH-WORKER-TIMEOUT"
  | "BG-WEBGPU-GRAPH-WORKER-ERROR"
  | "BG-WEBGPU-GRAPH-WORKER-TERMINAL"
  | "BG-WEBGPU-GRAPH-WORKER-FAILURE"
  | "BG-WEBGPU-GRAPH-WORKER-INTERNAL";

export class SemanticHostGraphBrowserWorkerTransportError extends Error {
  constructor(
    readonly code: SemanticHostGraphBrowserWorkerTransportErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "SemanticHostGraphBrowserWorkerTransportError";
  }
}

export class SemanticHostGraphBrowserWorkerReportedError
  extends SemanticHostGraphBrowserWorkerTransportError {
  constructor(readonly failure: SemanticHostGraphWorkerFailureMessage) {
    super(
      "BG-WEBGPU-GRAPH-WORKER-FAILURE",
      "$.terminal",
      `Worker failed during ${failure.phase}: ${failure.failureCode} at ` +
        `${failure.failurePath}: ${failure.failureDetail}`,
    );
    this.name = "SemanticHostGraphBrowserWorkerReportedError";
  }
}

export interface ExecuteSemanticHostGraphBrowserWorkerInput {
  readonly graphArtifact: VerifiedHostGraphArtifact;
  readonly kernelArtifacts: readonly VerifiedKernelArtifact[];
  readonly layoutArtifacts: readonly VerifiedLayoutArtifact[];
  readonly request: SemanticHostGraphWebGpuExecutionRequest;
}

export interface ExecuteSemanticHostGraphBrowserWorkerOptions {
  readonly signal?: AbortSignal;
  readonly timeoutMs?: number;
}

export interface SemanticHostGraphBrowserWorkerTransportTrace {
  readonly profile:
    typeof SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL;
  readonly topology: "single-dedicated-browser-worker";
  readonly backendProfile: typeof SEMANTIC_HOST_GRAPH_WEBGPU_PROFILE;
  readonly backendVersion:
    typeof SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION;
  readonly requestId: string;
  readonly artifactByteLength: number;
  readonly inputByteLength: number;
  readonly outputByteLength: number;
  readonly acceptedTerminalMessages: 1;
  readonly workerExecutionObserved: true;
  readonly workerLifecycle: "one-shot-terminated";
}

export interface SemanticHostGraphBrowserWorkerExecutionResult {
  readonly outputs: readonly SemanticHostGraphWebGpuOutputBinding[];
  readonly backendTrace: SemanticHostGraphWebGpuTrace;
  readonly transportTrace: SemanticHostGraphBrowserWorkerTransportTrace;
}

interface SemanticHostGraphWorkerRequestMessage {
  readonly kind: "browsergrad-host-graph-worker-request";
  readonly version:
    typeof SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION;
  readonly protocol:
    typeof SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL;
  readonly requestId: string;
  readonly graphArtifactBytes: Uint8Array;
  readonly kernelArtifactBytes: readonly Uint8Array[];
  readonly layoutArtifactBytes: readonly Uint8Array[];
  readonly inputs: readonly SemanticHostGraphWebGpuInputBinding[];
  readonly controls: readonly SemanticHostGraphWebGpuControlBinding[];
}

interface SemanticHostGraphWorkerSuccessMessage {
  readonly kind: "browsergrad-host-graph-worker-success";
  readonly version:
    typeof SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION;
  readonly protocol:
    typeof SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL;
  readonly requestId: string;
  readonly workerExecutionObserved: true;
  readonly outputs: readonly SemanticHostGraphWebGpuOutputBinding[];
  readonly trace: SemanticHostGraphWebGpuTrace;
}

export interface SemanticHostGraphWorkerFailureMessage {
  readonly kind: "browsergrad-host-graph-worker-failure";
  readonly version:
    typeof SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION;
  readonly protocol:
    typeof SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL;
  readonly requestId: string;
  readonly phase:
    | "artifact-verification"
    | "device-acquisition"
    | "preparation-execution";
  readonly failureCode: string;
  readonly failurePath: string;
  readonly failureDetail: string;
  readonly workerExecutionObserved: false;
}

type SemanticHostGraphWorkerTerminalMessage =
  | SemanticHostGraphWorkerSuccessMessage
  | SemanticHostGraphWorkerFailureMessage;

export interface SemanticHostGraphWorkerLike {
  postMessage(
    message: SemanticHostGraphWorkerRequestMessage,
    transfer: readonly ArrayBuffer[],
  ): void;
  addEventListener(
    type: "message",
    listener: (event: { readonly data: unknown }) => void,
  ): void;
  addEventListener(
    type: "error" | "messageerror",
    listener: (event: unknown) => void,
  ): void;
  removeEventListener(
    type: "message",
    listener: (event: { readonly data: unknown }) => void,
  ): void;
  removeEventListener(
    type: "error" | "messageerror",
    listener: (event: unknown) => void,
  ): void;
  terminate(): void;
}

export interface SemanticHostGraphWorkerTransportPlatform {
  readonly createWorker: (
    moduleUrl: URL,
    name: string,
  ) => SemanticHostGraphWorkerLike;
  readonly nextRequestId: () => string;
  readonly setTimeout: (callback: () => void, delayMs: number) => unknown;
  readonly clearTimeout: (handle: unknown) => void;
}

interface PreparedTransportLaunch {
  readonly message: SemanticHostGraphWorkerRequestMessage;
  readonly transfer: readonly ArrayBuffer[];
  readonly artifactByteLength: number;
  readonly inputByteLength: number;
}

/** Execute one graph in one package-owned, one-shot dedicated browser Worker. */
export function executeSemanticHostGraphBrowserWorker(
  input: ExecuteSemanticHostGraphBrowserWorkerInput,
  options: ExecuteSemanticHostGraphBrowserWorkerOptions = {},
): Promise<SemanticHostGraphBrowserWorkerExecutionResult> {
  if (CAPTURED_WORKER === undefined) {
    return Promise.reject(transportError(
      "BG-WEBGPU-GRAPH-WORKER-CAPABILITY",
      "$.runtime.Worker",
      "dedicated module Worker is unavailable in this realm",
    ));
  }
  if (CAPTURED_GET_RANDOM_VALUES === undefined) {
    return Promise.reject(transportError(
      "BG-WEBGPU-GRAPH-WORKER-CAPABILITY",
      "$.runtime.crypto",
      "cryptographic request ID generation is unavailable",
    ));
  }
  const random = CAPTURED_GET_RANDOM_VALUES;
  return executeSemanticHostGraphBrowserWorkerWithPlatform(
    input,
    options,
    OBJECT_FREEZE({
      createWorker: (moduleUrl: URL, name: string) => {
        const worker = new CAPTURED_WORKER(
          moduleUrl,
          { type: "module", name },
        );
        return {
          postMessage: (
            message: SemanticHostGraphWorkerRequestMessage,
            transfer: readonly ArrayBuffer[],
          ) =>
            worker.postMessage(message, [...transfer]),
          addEventListener: (
            type: "message" | "error" | "messageerror",
            listener:
              | ((event: { readonly data: unknown }) => void)
              | ((event: unknown) => void),
          ) => worker.addEventListener(
            type,
            listener as EventListener,
          ),
          removeEventListener: (
            type: "message" | "error" | "messageerror",
            listener:
              | ((event: { readonly data: unknown }) => void)
              | ((event: unknown) => void),
          ) => worker.removeEventListener(
            type,
            listener as EventListener,
          ),
          terminate: () => worker.terminate(),
        } as SemanticHostGraphWorkerLike;
      },
      nextRequestId: () => {
        const bytes = new Uint8Array(16);
        random(bytes);
        let hex = "";
        for (const byte of bytes) hex += byte.toString(16).padStart(2, "0");
        return `bg.host-graph.worker.${hex}`;
      },
      setTimeout: (callback: () => void, delayMs: number) =>
        CAPTURED_SET_TIMEOUT(callback, delayMs),
      clearTimeout: (handle: unknown) =>
        CAPTURED_CLEAR_TIMEOUT(handle as ReturnType<typeof setTimeout>),
    }),
  );
}

/** @internal Exact effect injection for lifecycle and hostile-frame tests. */
export function executeSemanticHostGraphBrowserWorkerWithPlatform(
  input: ExecuteSemanticHostGraphBrowserWorkerInput,
  options: ExecuteSemanticHostGraphBrowserWorkerOptions,
  platform: SemanticHostGraphWorkerTransportPlatform,
): Promise<SemanticHostGraphBrowserWorkerExecutionResult> {
  let launch: PreparedTransportLaunch;
  let normalized: {
    readonly signal?: AbortSignal;
    readonly timeoutMs: number;
  };
  let requestId: string;
  try {
    normalized = normalizeOptions(options);
    throwIfAborted(normalized.signal);
    requestId = platform.nextRequestId();
    if (!REQUEST_ID.test(requestId)) {
      invalid("$.platform.nextRequestId", "platform returned an invalid request ID");
    }
    launch = prepareLaunch(input, requestId);
  } catch (cause) {
    return Promise.reject(asTransportError(cause));
  }

  let worker: SemanticHostGraphWorkerLike;
  try {
    worker = platform.createWorker(
      WORKER_MODULE_URL,
      `browsergrad-host-graph-${requestId.slice(-12)}`,
    );
  } catch (cause) {
    return Promise.reject(transportError(
      "BG-WEBGPU-GRAPH-WORKER-ERROR",
      "$.worker",
      "failed to create the owned module Worker",
      cause,
    ));
  }

  return new Promise((resolve, reject) => {
    let settled = false;
    let timer: unknown;
    const finish = (
      outcome:
        | { readonly result: SemanticHostGraphBrowserWorkerExecutionResult }
        | { readonly error: SemanticHostGraphBrowserWorkerTransportError },
    ): void => {
      if (settled) return;
      settled = true;
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
      worker.removeEventListener("messageerror", onMessageError);
      normalized.signal?.removeEventListener("abort", onAbort);
      if (timer !== undefined) platform.clearTimeout(timer);
      worker.terminate();
      if ("result" in outcome) resolve(outcome.result);
      else reject(outcome.error);
    };
    const onMessage = (event: { readonly data: unknown }): void => {
      try {
        const terminal = parseTerminal(event.data, requestId);
        if (terminal.kind === "browsergrad-host-graph-worker-failure") {
          finish({
            error: new SemanticHostGraphBrowserWorkerReportedError(terminal),
          });
          return;
        }
        const outputByteLength = terminal.outputs.reduce(
          (sum, output) => sum + output.bytes.byteLength,
          0,
        );
        finish({
          result: OBJECT_FREEZE({
            outputs: terminal.outputs,
            backendTrace: terminal.trace,
            transportTrace: OBJECT_FREEZE({
              profile:
                SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
              topology: "single-dedicated-browser-worker",
              backendProfile: SEMANTIC_HOST_GRAPH_WEBGPU_PROFILE,
              backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
              requestId,
              artifactByteLength: launch.artifactByteLength,
              inputByteLength: launch.inputByteLength,
              outputByteLength,
              acceptedTerminalMessages: 1,
              workerExecutionObserved: true,
              workerLifecycle: "one-shot-terminated",
            }),
          }),
        });
      } catch (cause) {
        finish({
          error: transportError(
            "BG-WEBGPU-GRAPH-WORKER-TERMINAL",
            "$.terminal",
            "owned Worker returned an invalid terminal frame",
            cause,
          ),
        });
      }
    };
    const onError = (): void => finish({
      error: transportError(
        "BG-WEBGPU-GRAPH-WORKER-ERROR",
        "$.worker.error",
        "owned Worker emitted an error before its terminal frame",
      ),
    });
    const onMessageError = (): void => finish({
      error: transportError(
        "BG-WEBGPU-GRAPH-WORKER-TERMINAL",
        "$.worker.messageerror",
        "owned Worker emitted an unreadable terminal frame",
      ),
    });
    const onAbort = (): void => finish({
      error: transportError(
        "BG-WEBGPU-GRAPH-WORKER-CANCELLED",
        "$.options.signal",
        "browser Worker execution was cancelled",
      ),
    });

    try {
      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", onError);
      worker.addEventListener("messageerror", onMessageError);
      normalized.signal?.addEventListener("abort", onAbort, { once: true });
      if (normalized.signal?.aborted === true) {
        onAbort();
        return;
      }
      timer = platform.setTimeout(() => finish({
        error: transportError(
          "BG-WEBGPU-GRAPH-WORKER-TIMEOUT",
          "$.options.timeoutMs",
          "owned Worker exceeded the host wall-time limit",
        ),
      }), normalized.timeoutMs);
      worker.postMessage(launch.message, launch.transfer);
    } catch (cause) {
      finish({
        error: transportError(
          "BG-WEBGPU-GRAPH-WORKER-ERROR",
          "$.worker",
          "owned Worker launch failed",
          cause,
        ),
      });
    }
  });
}

export interface SemanticHostGraphWorkerEntryScope {
  addEventListener(
    type: "message",
    listener: (event: { readonly data: unknown }) => void,
  ): void;
  removeEventListener(
    type: "message",
    listener: (event: { readonly data: unknown }) => void,
  ): void;
  postMessage(
    message: SemanticHostGraphWorkerTerminalMessage,
    transfer: readonly ArrayBuffer[],
  ): void;
  close(): void;
  queueMicrotask(callback: () => void): void;
}

/** @internal Install the one-shot endpoint inside a dedicated Worker realm. */
export function installSemanticHostGraphWorkerEntry(
  scope: SemanticHostGraphWorkerEntryScope,
): void {
  const onMessage = (event: { readonly data: unknown }): void => {
    scope.removeEventListener("message", onMessage);
    const requestId = requestIdFromUnknown(event.data);
    void handleSemanticHostGraphWorkerRequest(event.data)
      .then(({ message, transfer }) => {
        scope.postMessage(message, transfer);
        scope.close();
      })
      .catch((cause: unknown) => {
        if (requestId === undefined) {
          scope.queueMicrotask(() => {
            throw cause;
          });
          return;
        }
        scope.postMessage(workerFailure(requestId, cause), []);
        scope.close();
      });
  };
  scope.addEventListener("message", onMessage);
}

/** @internal Verify, prepare, and execute one transferred request. */
export async function handleSemanticHostGraphWorkerRequest(
  value: unknown,
): Promise<{
  readonly message: SemanticHostGraphWorkerSuccessMessage;
  readonly transfer: readonly ArrayBuffer[];
}> {
  const request = parseRequest(value);
  let phase: SemanticHostGraphWorkerFailureMessage["phase"] =
    "artifact-verification";
  let device: Awaited<ReturnType<typeof createDevice>> | undefined;
  try {
    const layoutArtifacts = await Promise.all(
      request.layoutArtifactBytes.map((bytes) =>
        decodeLayoutArtifact(bytes)),
    );
    const layoutsByHash = new Map<string, VerifiedLayoutArtifact>();
    for (const layout of layoutArtifacts) {
      layoutsByHash.set(await hashSemanticArtifact(layout), layout);
    }
    const kernelArtifacts: VerifiedKernelArtifact[] = [];
    for (let index = 0; index < request.kernelArtifactBytes.length; index += 1) {
      const bytes = request.kernelArtifactBytes[index] as Uint8Array;
      const layoutHash = kernelLayoutSemanticHash(bytes, index);
      const layout = layoutsByHash.get(layoutHash);
      if (layout === undefined) {
        throw transportError(
          "BG-WEBGPU-GRAPH-WORKER-INVALID",
          `$.kernelArtifactBytes[${index}]`,
          "kernel artifact references a layout absent from this request",
        );
      }
      kernelArtifacts.push(await decodeKernelArtifact(bytes, { layout }));
    }
    const graphArtifact = await decodeHostGraphArtifact(
      request.graphArtifactBytes,
      { kernelArtifacts, layoutArtifacts },
    );

    phase = "device-acquisition";
    device = await createDevice();
    phase = "preparation-execution";
    const prepared = await prepareSemanticHostGraphWebGpu(graphArtifact, {
      kernelArtifacts,
      layoutArtifacts,
    });
    const result = await runSemanticHostGraphWebGpu(
      device,
      prepared,
      {
        inputs: request.inputs,
        ...(request.controls.length === 0
          ? {}
          : { controls: request.controls }),
      },
    );
    const transfer = result.outputs.map((output) =>
      output.bytes.buffer as ArrayBuffer);
    return OBJECT_FREEZE({
      message: OBJECT_FREEZE({
        kind: "browsergrad-host-graph-worker-success",
        version:
          SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION,
        protocol:
          SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
        requestId: request.requestId,
        workerExecutionObserved: true,
        outputs: result.outputs,
        trace: result.trace,
      }),
      transfer: OBJECT_FREEZE(transfer),
    });
  } catch (cause) {
    throw workerPhaseError(phase, cause);
  } finally {
    if (device !== undefined) {
      device.clearCache();
      device.gpu.destroy();
    }
  }
}

class SemanticHostGraphWorkerPhaseError extends Error {
  constructor(
    readonly phase: SemanticHostGraphWorkerFailureMessage["phase"],
    readonly failureCode: string,
    readonly failurePath: string,
    readonly failureDetail: string,
    options?: ErrorOptions,
  ) {
    super(failureDetail, options);
    this.name = "SemanticHostGraphWorkerPhaseError";
  }
}

function prepareLaunch(
  input: ExecuteSemanticHostGraphBrowserWorkerInput,
  requestId: string,
): PreparedTransportLaunch {
  const object = exactRecord(
    input,
    "$.input",
    ["graphArtifact", "kernelArtifacts", "layoutArtifacts", "request"],
  );
  const kernelArtifacts = denseArray(
    object.kernelArtifacts,
    "$.input.kernelArtifacts",
    SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACTS,
  ) as readonly VerifiedKernelArtifact[];
  const layoutArtifacts = denseArray(
    object.layoutArtifacts,
    "$.input.layoutArtifacts",
    SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACTS,
  ) as readonly VerifiedLayoutArtifact[];
  const graphArtifactBytes = artifactBytes(
    object.graphArtifact as VerifiedArtifact<JsonValue>,
    "$.input.graphArtifact",
  );
  const kernelArtifactBytes = kernelArtifacts.map((artifact, index) =>
    artifactBytes(
      artifact as VerifiedArtifact<JsonValue>,
      `$.input.kernelArtifacts[${index}]`,
    ));
  const layoutArtifactBytes = layoutArtifacts.map((artifact, index) =>
    artifactBytes(
      artifact as VerifiedArtifact<JsonValue>,
      `$.input.layoutArtifacts[${index}]`,
    ));
  const allArtifactBytes = [
    graphArtifactBytes,
    ...kernelArtifactBytes,
    ...layoutArtifactBytes,
  ];
  const artifactByteLength = sumBytes(
    allArtifactBytes,
    SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACT_BYTES,
    "$.input.artifacts",
  );
  const request = exactRecord(
    object.request,
    "$.input.request",
    ["inputs", "controls"],
    ["inputs"],
  );
  const inputs = denseArray(
    request.inputs,
    "$.input.request.inputs",
    SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACTS,
  ).map((binding, index) => captureInput(
    binding,
    `$.input.request.inputs[${index}]`,
  ));
  const controls = request.controls === undefined
    ? []
    : denseArray(
        request.controls,
        "$.input.request.controls",
        SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACTS,
      ).map((binding, index) => captureControl(
        binding,
        `$.input.request.controls[${index}]`,
      ));
  const inputByteLength = sumBytes(
    inputs.map((input) => input.bytes),
    SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_INPUT_BYTES,
    "$.input.request.inputs",
  );
  const message = OBJECT_FREEZE({
    kind: "browsergrad-host-graph-worker-request" as const,
    version: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION,
    protocol: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
    requestId,
    graphArtifactBytes,
    kernelArtifactBytes: OBJECT_FREEZE(kernelArtifactBytes),
    layoutArtifactBytes: OBJECT_FREEZE(layoutArtifactBytes),
    inputs: OBJECT_FREEZE(inputs),
    controls: OBJECT_FREEZE(controls),
  });
  return OBJECT_FREEZE({
    message,
    transfer: OBJECT_FREEZE([
      ...allArtifactBytes.map((bytes) => bytes.buffer as ArrayBuffer),
      ...inputs.map((input) => input.bytes.buffer as ArrayBuffer),
    ]),
    artifactByteLength,
    inputByteLength,
  });
}

function parseRequest(value: unknown): SemanticHostGraphWorkerRequestMessage {
  const object = exactRecord(value, "$", [
    "kind",
    "version",
    "protocol",
    "requestId",
    "graphArtifactBytes",
    "kernelArtifactBytes",
    "layoutArtifactBytes",
    "inputs",
    "controls",
  ]);
  if (object.kind !== "browsergrad-host-graph-worker-request") {
    invalid("$.kind", "unexpected Worker request kind");
  }
  if (
    object.version !==
      SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION ||
    object.protocol !==
      SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL
  ) {
    invalid("$.protocol", "unsupported Worker transport protocol");
  }
  const requestId = requireRequestId(object.requestId, "$.requestId");
  const graphArtifactBytes = validateTransferredBytes(
    object.graphArtifactBytes,
    "$.graphArtifactBytes",
  );
  const kernelArtifactBytes = denseArray(
    object.kernelArtifactBytes,
    "$.kernelArtifactBytes",
    SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACTS,
  ).map((bytes, index) => validateTransferredBytes(
    bytes,
    `$.kernelArtifactBytes[${index}]`,
  ));
  const layoutArtifactBytes = denseArray(
    object.layoutArtifactBytes,
    "$.layoutArtifactBytes",
    SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACTS,
  ).map((bytes, index) => validateTransferredBytes(
    bytes,
    `$.layoutArtifactBytes[${index}]`,
  ));
  sumBytes(
    [graphArtifactBytes, ...kernelArtifactBytes, ...layoutArtifactBytes],
    SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACT_BYTES,
    "$.artifacts",
  );
  const inputs = denseArray(
    object.inputs,
    "$.inputs",
    SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACTS,
  ).map((binding, index) => validateTransferredInput(
    binding,
    `$.inputs[${index}]`,
  ));
  sumBytes(
    inputs.map((input) => input.bytes),
    SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_INPUT_BYTES,
    "$.inputs",
  );
  const controls = denseArray(
    object.controls,
    "$.controls",
    SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACTS,
  ).map((binding, index) => captureControl(
    binding,
    `$.controls[${index}]`,
  ));
  return OBJECT_FREEZE({
    kind: "browsergrad-host-graph-worker-request",
    version: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION,
    protocol: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
    requestId,
    graphArtifactBytes,
    kernelArtifactBytes: OBJECT_FREEZE(kernelArtifactBytes),
    layoutArtifactBytes: OBJECT_FREEZE(layoutArtifactBytes),
    inputs: OBJECT_FREEZE(inputs),
    controls: OBJECT_FREEZE(controls),
  });
}

function parseTerminal(
  value: unknown,
  requestId: string,
): SemanticHostGraphWorkerTerminalMessage {
  const kind = recordKind(value, "$.kind");
  if (kind === "browsergrad-host-graph-worker-failure") {
    const object = exactRecord(value, "$", [
      "kind",
      "version",
      "protocol",
      "requestId",
      "phase",
      "failureCode",
      "failurePath",
      "failureDetail",
      "workerExecutionObserved",
    ]);
    requireTerminalHeader(object, requestId);
    if (
      object.phase !== "artifact-verification" &&
      object.phase !== "device-acquisition" &&
      object.phase !== "preparation-execution"
    ) {
      invalid("$.phase", "unknown Worker failure phase");
    }
    if (
      typeof object.failureCode !== "string" ||
      !FAILURE_CODE.test(object.failureCode)
    ) {
      invalid("$.failureCode", "invalid Worker failure code");
    }
    if (
      typeof object.failurePath !== "string" ||
      object.failurePath.length === 0 ||
      typeof object.failureDetail !== "string" ||
      object.failureDetail.length === 0 ||
      object.failureDetail.length > 2_048
    ) {
      invalid("$.failureDetail", "invalid Worker failure projection");
    }
    if (object.workerExecutionObserved !== false) {
      invalid(
        "$.workerExecutionObserved",
        "failed Worker must not claim execution",
      );
    }
    return OBJECT_FREEZE({
      kind,
      version: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION,
      protocol: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
      requestId,
      phase: object.phase,
      failureCode: object.failureCode,
      failurePath: object.failurePath,
      failureDetail: object.failureDetail,
      workerExecutionObserved: false,
    });
  }
  if (kind !== "browsergrad-host-graph-worker-success") {
    invalid("$.kind", "unknown Worker terminal kind");
  }
  const object = exactRecord(value, "$", [
    "kind",
    "version",
    "protocol",
    "requestId",
    "workerExecutionObserved",
    "outputs",
    "trace",
  ]);
  requireTerminalHeader(object, requestId);
  if (object.workerExecutionObserved !== true) {
    invalid(
      "$.workerExecutionObserved",
      "successful Worker terminal must prove execution",
    );
  }
  const outputs = denseArray(
    object.outputs,
    "$.outputs",
    SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACTS,
  ).map((output, index) => validateTransferredOutput(
    output,
    `$.outputs[${index}]`,
  ));
  sumBytes(
    outputs.map((output) => output.bytes),
    SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_OUTPUT_BYTES,
    "$.outputs",
  );
  const trace = validateTrace(object.trace);
  return OBJECT_FREEZE({
    kind,
    version: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION,
    protocol: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
    requestId,
    workerExecutionObserved: true,
    outputs: OBJECT_FREEZE(outputs),
    trace,
  });
}

function validateTrace(value: unknown): SemanticHostGraphWebGpuTrace {
  const object = exactRecord(value, "$.trace", [
    "profile",
    "backendVersion",
    "graphSemanticHash",
    "pipelineIdentityHash",
    "backendSpecializationHash",
    "failureModel",
    "executedNodeIds",
    "expandedStepCount",
    "dispatchStepCount",
    "copyStepCount",
    "materializationCount",
    "completedEventIds",
    "completedRepeats",
    "completedDynamicDispatches",
    "completedConditionals",
    "midGraphFeedbackCount",
    "midGraphFeedbackStageCount",
    "collectiveReductionStepCount",
    "collectiveReplicationStepCount",
    "wgslModuleHashes",
    "plannedTransientGpuBytes",
    "plannedTransientHostBytes",
    "plannedTransientWorkingSetBytes",
    "maxTransientWorkingSetBytes",
    "submitted",
    "device",
  ]);
  if (
    object.profile !== SEMANTIC_HOST_GRAPH_WEBGPU_PROFILE ||
    object.backendVersion !== SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION ||
    object.failureModel !== "fail-stop-no-partial-output-commit" ||
    object.submitted !== true
  ) {
    invalid("$.trace", "terminal trace does not identify submitted backend work");
  }
  for (const field of [
    "graphSemanticHash",
    "pipelineIdentityHash",
    "backendSpecializationHash",
  ] as const) {
    if (typeof object[field] !== "string" || !HASH.test(object[field])) {
      invalid(`$.trace.${field}`, "trace hash must be lowercase SHA-256");
    }
  }
  for (const field of [
    "expandedStepCount",
    "dispatchStepCount",
    "copyStepCount",
    "materializationCount",
    "midGraphFeedbackCount",
    "midGraphFeedbackStageCount",
    "collectiveReductionStepCount",
    "collectiveReplicationStepCount",
  ] as const) {
    nonnegativeInteger(object[field], `$.trace.${field}`);
  }
  for (const field of [
    "plannedTransientGpuBytes",
    "plannedTransientHostBytes",
    "plannedTransientWorkingSetBytes",
    "maxTransientWorkingSetBytes",
  ] as const) {
    parseWireU64(object[field], `$.trace.${field}`);
  }
  validateStringArray(
    object.executedNodeIds,
    "$.trace.executedNodeIds",
    16_384,
    true,
  );
  validateStringArray(
    object.completedEventIds,
    "$.trace.completedEventIds",
    16_384,
    true,
  );
  const moduleHashes = validateStringArray(
    object.wgslModuleHashes,
    "$.trace.wgslModuleHashes",
    16_384,
    true,
  );
  if (moduleHashes.some((hash) => !HASH.test(hash))) {
    invalid("$.trace.wgslModuleHashes", "WGSL module hashes must be SHA-256");
  }
  for (const field of [
    "completedRepeats",
    "completedDynamicDispatches",
    "completedConditionals",
  ] as const) {
    denseArray(object[field], `$.trace.${field}`, 16_384);
  }
  const device = exactRecord(
    object.device,
    "$.trace.device",
    ["features", "limits"],
  );
  validateStringArray(
    device.features,
    "$.trace.device.features",
    256,
    true,
  );
  const limits = exactRecord(
    device.limits,
    "$.trace.device.limits",
    [
      "maxBufferSize",
      "maxStorageBufferBindingSize",
      "maxComputeWorkgroupsPerDimension",
      "maxComputeInvocationsPerWorkgroup",
      "maxComputeWorkgroupSizeX",
      "maxBindingsPerBindGroup",
      "maxStorageBuffersPerShaderStage",
      "maxUniformBuffersPerShaderStage",
    ],
  );
  for (const [name, limit] of Object.entries(limits)) {
    nonnegativeInteger(limit, `$.trace.device.limits.${name}`);
  }
  try {
    assertJsonValue(object);
    return deepFreezeJson(
      object as unknown as JsonValue,
    ) as unknown as SemanticHostGraphWebGpuTrace;
  } catch (cause) {
    throw transportError(
      "BG-WEBGPU-GRAPH-WORKER-INVALID",
      "$.trace",
      "terminal trace is not a bounded canonical data tree",
      cause,
    );
  }
}

function captureInput(
  value: unknown,
  path: string,
): SemanticHostGraphWebGpuInputBinding {
  const object = exactRecord(value, path, ["rank", "resourceId", "bytes"]);
  const rank = parseWireU64(object.rank, `${path}.rank`);
  const resourceId = nonemptyString(object.resourceId, `${path}.resourceId`);
  return OBJECT_FREEZE({
    rank,
    resourceId,
    bytes: copyBytes(object.bytes, `${path}.bytes`),
  });
}

function validateTransferredInput(
  value: unknown,
  path: string,
): SemanticHostGraphWebGpuInputBinding {
  const object = exactRecord(value, path, ["rank", "resourceId", "bytes"]);
  return OBJECT_FREEZE({
    rank: parseWireU64(object.rank, `${path}.rank`),
    resourceId: nonemptyString(object.resourceId, `${path}.resourceId`),
    bytes: validateTransferredBytes(object.bytes, `${path}.bytes`),
  });
}

function captureControl(
  value: unknown,
  path: string,
): SemanticHostGraphWebGpuControlBinding {
  const object = exactRecord(value, path, ["controlId", "value"]);
  return OBJECT_FREEZE({
    controlId: nonemptyString(object.controlId, `${path}.controlId`),
    value: parseWireU64(object.value, `${path}.value`),
  });
}

function validateTransferredOutput(
  value: unknown,
  path: string,
): SemanticHostGraphWebGpuOutputBinding {
  const object = exactRecord(value, path, ["rank", "resourceId", "bytes"]);
  return OBJECT_FREEZE({
    rank: parseWireU64(object.rank, `${path}.rank`),
    resourceId: nonemptyString(object.resourceId, `${path}.resourceId`),
    bytes: validateTransferredBytes(object.bytes, `${path}.bytes`),
  });
}

function artifactBytes(
  artifact: VerifiedArtifact<JsonValue>,
  path: string,
): Uint8Array {
  try {
    const bytes = copyVerifiedArtifactWireBytes(artifact);
    if (bytes.byteLength > SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_ARTIFACT_BYTES) {
      resource(path, "artifact exceeds the Worker transport byte limit");
    }
    return bytes;
  } catch (cause) {
    if (cause instanceof SemanticHostGraphBrowserWorkerTransportError) {
      throw cause;
    }
    throw transportError(
      "BG-WEBGPU-GRAPH-WORKER-INVALID",
      path,
      "expected an opaque verifier-issued semantic artifact",
      cause,
    );
  }
}

function kernelLayoutSemanticHash(bytes: Uint8Array, index: number): string {
  const value = decodeWireJson(bytes);
  if (
    !isJsonObject(value) ||
    value.payload === undefined ||
    !isJsonObject(value.payload)
  ) {
    invalid(
      `$.kernelArtifactBytes[${index}]`,
      "kernel artifact envelope payload is missing",
    );
  }
  const hash = value.payload.layoutSemanticHash;
  if (typeof hash !== "string" || !HASH.test(hash)) {
    invalid(
      `$.kernelArtifactBytes[${index}].payload.layoutSemanticHash`,
      "kernel layout semantic hash is invalid",
    );
  }
  return hash;
}

function workerFailure(
  requestId: string,
  cause: unknown,
): SemanticHostGraphWorkerFailureMessage {
  const failure = cause instanceof SemanticHostGraphWorkerPhaseError
    ? cause
    : workerPhaseError("preparation-execution", cause);
  return OBJECT_FREEZE({
    kind: "browsergrad-host-graph-worker-failure",
    version: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION,
    protocol: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
    requestId,
    phase: failure.phase,
    failureCode: failure.failureCode,
    failurePath: failure.failurePath,
    failureDetail: failure.failureDetail,
    workerExecutionObserved: false,
  });
}

function workerPhaseError(
  phase: SemanticHostGraphWorkerFailureMessage["phase"],
  cause: unknown,
): SemanticHostGraphWorkerPhaseError {
  if (cause instanceof SemanticHostGraphWorkerPhaseError) return cause;
  if (cause instanceof SemanticHostGraphWebGpuError) {
    return new SemanticHostGraphWorkerPhaseError(
      phase,
      cause.code,
      cause.path,
      boundedDetail(cause.message),
      { cause },
    );
  }
  if (cause instanceof SemanticSchemaError) {
    return new SemanticHostGraphWorkerPhaseError(
      phase,
      cause.diagnostic.code,
      cause.diagnostic.path ?? "$",
      boundedDetail(cause.diagnostic.message),
      { cause },
    );
  }
  if (cause instanceof SemanticHostGraphBrowserWorkerTransportError) {
    return new SemanticHostGraphWorkerPhaseError(
      phase,
      cause.code,
      cause.path,
      boundedDetail(cause.message),
      { cause },
    );
  }
  return new SemanticHostGraphWorkerPhaseError(
    phase,
    "BG-WEBGPU-GRAPH-WORKER-INTERNAL",
    "$.worker",
    boundedDetail(
      cause instanceof Error
        ? cause.message
        : "unknown Worker execution failure",
    ),
    { cause },
  );
}

function normalizeOptions(
  options: ExecuteSemanticHostGraphBrowserWorkerOptions,
): { readonly signal?: AbortSignal; readonly timeoutMs: number } {
  const object = exactRecord(
    options,
    "$.options",
    ["signal", "timeoutMs"],
    [],
  );
  const signal = object.signal === undefined
    ? undefined
    : requireAbortSignal(object.signal, "$.options.signal");
  const timeoutMs = object.timeoutMs === undefined
    ? DEFAULT_WALL_TIME_MS
    : positiveInteger(
        object.timeoutMs,
        SEMANTIC_HOST_GRAPH_BROWSER_WORKER_MAX_WALL_TIME_MS,
        "$.options.timeoutMs",
      );
  return OBJECT_FREEZE({
    ...(signal === undefined ? {} : { signal }),
    timeoutMs,
  });
}

function requireAbortSignal(value: unknown, path: string): AbortSignal {
  if (ABORTED_GETTER === undefined) invalid(path, "AbortSignal is unavailable");
  try {
    REFLECT_APPLY(ABORTED_GETTER, value, []);
  } catch {
    invalid(path, "expected a native AbortSignal");
  }
  return value as AbortSignal;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    throw transportError(
      "BG-WEBGPU-GRAPH-WORKER-CANCELLED",
      "$.options.signal",
      "browser Worker execution was cancelled before launch",
    );
  }
}

function validateTransferredBytes(value: unknown, path: string): Uint8Array {
  const slots = nativeUint8Slots(value, path);
  if (isSharedBuffer(slots.buffer)) {
    invalid(path, "shared bytes require an explicit synchronization contract");
  }
  return value as Uint8Array;
}

function copyBytes(value: unknown, path: string): Uint8Array {
  const slots = nativeUint8Slots(value, path);
  if (isSharedBuffer(slots.buffer)) {
    invalid(path, "shared bytes require an explicit synchronization contract");
  }
  const copy = new Uint8Array(slots.byteLength);
  REFLECT_APPLY(UINT8_SET, copy, [
    new Uint8Array(slots.buffer, slots.byteOffset, slots.byteLength),
  ]);
  return copy;
}

function nativeUint8Slots(
  value: unknown,
  path: string,
): {
  readonly buffer: ArrayBufferLike;
  readonly byteOffset: number;
  readonly byteLength: number;
} {
  if (
    typeof value !== "object" ||
    value === null ||
    OBJECT_GET_PROTOTYPE_OF(value) !== UINT8_ARRAY_PROTOTYPE
  ) {
    invalid(path, "expected a plain Uint8Array");
  }
  try {
    const buffer = REFLECT_APPLY(UINT8_BUFFER_GETTER, value, []) as
      ArrayBufferLike;
    const byteOffset = REFLECT_APPLY(
      UINT8_BYTE_OFFSET_GETTER,
      value,
      [],
    ) as number;
    const byteLength = REFLECT_APPLY(
      UINT8_BYTE_LENGTH_GETTER,
      value,
      [],
    ) as number;
    if (!isSharedBuffer(buffer)) {
      REFLECT_APPLY(ARRAY_BUFFER_SLICE, buffer, [0, 0]);
    }
    return OBJECT_FREEZE({ buffer, byteOffset, byteLength });
  } catch (cause) {
    throw transportError(
      "BG-WEBGPU-GRAPH-WORKER-INVALID",
      path,
      "bytes do not expose attached native Uint8Array slots",
      cause,
    );
  }
}

function isSharedBuffer(value: ArrayBufferLike): boolean {
  if (SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined) return false;
  try {
    REFLECT_APPLY(SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER, value, []);
    return true;
  } catch {
    return false;
  }
}

function denseArray(
  value: unknown,
  path: string,
  maximum: number,
): readonly unknown[] {
  if (
    !Array.isArray(value) ||
    OBJECT_GET_PROTOTYPE_OF(value) !== ARRAY_PROTOTYPE
  ) {
    invalid(path, "expected a plain dense array");
  }
  if (value.length > maximum) resource(path, `array exceeds limit ${maximum}`);
  const keys = REFLECT_OWN_KEYS(value);
  if (
    keys.length !== value.length + 1 ||
    keys.some((key) => {
      if (key === "length") return false;
      if (typeof key !== "string" || !/^(?:0|[1-9][0-9]*)$/u.test(key)) {
        return true;
      }
      const index = Number(key);
      return index < 0 || index >= value.length || String(index) !== key;
    })
  ) {
    invalid(path, "array must be dense and contain no named properties");
  }
  const descriptors = OBJECT_GET_OWN_PROPERTY_DESCRIPTORS(value);
  const result: unknown[] = [];
  for (let index = 0; index < value.length; index += 1) {
    const descriptor = descriptors[String(index)];
    if (
      descriptor === undefined ||
      descriptor.enumerable !== true ||
      !("value" in descriptor)
    ) {
      invalid(`${path}[${index}]`, "array element must be an enumerable data property");
    }
    result.push(descriptor.value);
  }
  return result;
}

function exactRecord(
  value: unknown,
  path: string,
  allowed: readonly string[],
  required: readonly string[] = allowed,
): Readonly<Record<string, unknown>> {
  if (
    typeof value !== "object" ||
    value === null ||
    Array.isArray(value)
  ) {
    invalid(path, "expected a plain object");
  }
  const prototype = OBJECT_GET_PROTOTYPE_OF(value);
  if (prototype !== OBJECT_PROTOTYPE && prototype !== null) {
    invalid(path, "expected a plain object");
  }
  const keys = REFLECT_OWN_KEYS(value);
  if (keys.some((key) => typeof key !== "string")) {
    invalid(path, "object must not contain symbol properties");
  }
  const unknown = keys.filter((key) =>
    typeof key === "string" && !allowed.includes(key));
  if (unknown.length > 0) {
    invalid(path, `object has unknown fields: ${unknown.sort().join(", ")}`);
  }
  const descriptors = OBJECT_GET_OWN_PROPERTY_DESCRIPTORS(value);
  const result: Record<string, unknown> = {};
  for (const key of allowed) {
    const descriptor = descriptors[key];
    if (descriptor === undefined) {
      if (required.includes(key)) invalid(`${path}.${key}`, "field is required");
      continue;
    }
    if (descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "field must be an enumerable data property");
    }
    result[key] = descriptor.value;
  }
  return OBJECT_FREEZE(result);
}

function requireTerminalHeader(
  object: Readonly<Record<string, unknown>>,
  requestId: string,
): void {
  if (
    object.version !==
      SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_VERSION ||
    object.protocol !==
      SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL
  ) {
    invalid("$.protocol", "unsupported Worker terminal protocol");
  }
  if (object.requestId !== requestId) {
    invalid("$.requestId", "Worker terminal request ID mismatch");
  }
}

function requestIdFromUnknown(value: unknown): string | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return undefined;
  }
  const descriptor = OBJECT_GET_OWN_PROPERTY_DESCRIPTOR(value, "requestId");
  return descriptor !== undefined &&
      "value" in descriptor &&
      typeof descriptor.value === "string" &&
      REQUEST_ID.test(descriptor.value)
    ? descriptor.value
    : undefined;
}

function requireRequestId(value: unknown, path: string): string {
  if (typeof value !== "string" || !REQUEST_ID.test(value)) {
    invalid(path, "invalid Worker request ID");
  }
  return value;
}

function recordKind(value: unknown, path: string): unknown {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    invalid(path, "terminal frame must be an object");
  }
  const descriptor = OBJECT_GET_OWN_PROPERTY_DESCRIPTOR(value, "kind");
  if (
    descriptor === undefined ||
    descriptor.enumerable !== true ||
    !("value" in descriptor)
  ) {
    invalid(path, "terminal kind must be an enumerable data property");
  }
  return descriptor.value;
}

function sumBytes(
  values: readonly Uint8Array[],
  maximum: number,
  path: string,
): number {
  let total = 0;
  for (const value of values) {
    total += value.byteLength;
    if (!Number.isSafeInteger(total) || total > maximum) {
      resource(path, `byte length exceeds limit ${maximum}`);
    }
  }
  return total;
}

function positiveInteger(value: unknown, maximum: number, path: string): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value <= 0 ||
    value > maximum
  ) {
    invalid(path, `expected a positive integer no greater than ${maximum}`);
  }
  return value;
}

function nonnegativeInteger(value: unknown, path: string): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < 0
  ) {
    invalid(path, "expected a non-negative safe integer");
  }
  return value;
}

function validateStringArray(
  value: unknown,
  path: string,
  maximum: number,
  requireNonempty: boolean,
): readonly string[] {
  const values = denseArray(value, path, maximum);
  return values.map((item, index) => {
    if (
      typeof item !== "string" ||
      item.length > 128 ||
      (requireNonempty && item.length === 0)
    ) {
      invalid(`${path}[${index}]`, "expected a bounded string");
    }
    return item;
  });
}

function nonemptyString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0 || value.length > 128) {
    invalid(path, "expected a non-empty string of at most 128 characters");
  }
  return value;
}

function boundedDetail(value: string): string {
  const normalized = value.length === 0 ? "Worker operation failed" : value;
  return normalized.length <= 2_048
    ? normalized
    : `${normalized.slice(0, 2_045)}...`;
}

function requiredGetter(
  target: object,
  name: string,
): (this: unknown) => unknown {
  const getter = OBJECT_GET_OWN_PROPERTY_DESCRIPTOR(target, name)?.get;
  if (getter === undefined) throw new Error(`missing typed-array ${name} getter`);
  return getter;
}

function resource(path: string, message: string): never {
  throw transportError(
    "BG-WEBGPU-GRAPH-WORKER-RESOURCE-LIMIT",
    path,
    message,
  );
}

function invalid(path: string, message: string): never {
  throw transportError(
    "BG-WEBGPU-GRAPH-WORKER-INVALID",
    path,
    message,
  );
}

function asTransportError(
  cause: unknown,
): SemanticHostGraphBrowserWorkerTransportError {
  return cause instanceof SemanticHostGraphBrowserWorkerTransportError
    ? cause
    : transportError(
        "BG-WEBGPU-GRAPH-WORKER-INVALID",
        "$",
        "invalid browser Worker execution request",
        cause,
      );
}

function transportError(
  code: SemanticHostGraphBrowserWorkerTransportErrorCode,
  path: string,
  message: string,
  cause?: unknown,
): SemanticHostGraphBrowserWorkerTransportError {
  return new SemanticHostGraphBrowserWorkerTransportError(
    code,
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}
