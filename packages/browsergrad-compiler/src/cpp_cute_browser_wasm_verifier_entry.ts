import {
  canonicalJsonBytes,
  sha256Hex,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  decodeCppCuteBrowserRuntimeAbiManifest,
} from "./cpp_cute_browser_runtime_abi.js";
import {
  unwrapPreparedCppCuteBrowserWasmConformance,
  verifyCppCuteBrowserWasmConformance,
} from "./cpp_cute_browser_wasm_inspection.js";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
  CPP_CUTE_BROWSER_WASM_VERIFIER_MINOR,
  CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
  CPP_CUTE_BROWSER_WASM_VERIFIER_REPORT_BYTE_LIMIT,
  type CppCuteBrowserWasmVerifierFailureMessage,
  type CppCuteBrowserWasmVerifierFailurePhase,
  type CppCuteBrowserWasmVerifierLaunchMessage,
  type CppCuteBrowserWasmVerifierReportSummary,
  type CppCuteBrowserWasmVerifierSuccessMessage,
  type CppCuteBrowserWasmVerifierTerminalMessage,
} from "./cpp_cute_browser_wasm_verifier_messages.js";

export const CPP_CUTE_BROWSER_WASM_VERIFIER_ENTRY_PROTOCOL =
  "browsergrad.compiler.cpp-cute.browser-wasm-verifier-entry@1";

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const REQUEST_ID = /^bg\.cpp\.browser-wasm-verifier-request\.sha256\.[0-9a-f]{64}$/u;
const ASSET_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u;
const ABI_ID = /^bg\.cpp\.browser-runtime-abi\.sha256\.[0-9a-f]{64}$/u;
const FAILURE_CODE = /^BG-[A-Z0-9-]+$/u;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTOR = Object.getOwnPropertyDescriptor;
const NATIVE_GET_PROTOTYPE_OF = Object.getPrototypeOf;

export interface CppCuteBrowserWasmVerifierEntryMessageEvent {
  readonly data: unknown;
}

export type CppCuteBrowserWasmVerifierEntryMessageListener = (
  event: CppCuteBrowserWasmVerifierEntryMessageEvent,
) => void;

/** Captured one-shot Worker effects. No fetch or module-loader authority exists. */
export interface CppCuteBrowserWasmVerifierEntryScope {
  readonly addEventListener: (
    type: "message",
    listener: CppCuteBrowserWasmVerifierEntryMessageListener,
  ) => void;
  readonly removeEventListener: (
    type: "message",
    listener: CppCuteBrowserWasmVerifierEntryMessageListener,
  ) => void;
  readonly postMessage: (
    message: CppCuteBrowserWasmVerifierTerminalMessage,
    transfer: readonly ArrayBuffer[],
  ) => void;
  readonly queueMicrotask: (callback: () => void) => void;
}

interface LaunchIdentity {
  readonly requestId: string;
  readonly invocationNonceSha256: string;
}

export type CppCuteBrowserWasmVerifierEntryErrorCode =
  "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-ENTRY-INVALID";
const ENTRY_INVALID_CODE: CppCuteBrowserWasmVerifierEntryErrorCode =
  "BG-COMPILER-CPP-CUTE-BROWSER-WASM-VERIFIER-ENTRY-INVALID";

export class CppCuteBrowserWasmVerifierEntryError extends Error {
  constructor(
    readonly code: CppCuteBrowserWasmVerifierEntryErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserWasmVerifierEntryError";
  }
}

export function installCppCuteBrowserWasmVerifierEntry(
  scope: CppCuteBrowserWasmVerifierEntryScope,
): void {
  const effects = exactScope(scope);
  const onMessage: CppCuteBrowserWasmVerifierEntryMessageListener = (event) => {
    NATIVE_REFLECT_APPLY(effects.removeEventListener, scope, ["message", onMessage]);
    void handleCppCuteBrowserWasmVerifierLaunch(event.data, (message, transfer) => {
      NATIVE_REFLECT_APPLY(effects.postMessage, scope, [message, transfer]);
    }).catch((cause: unknown) => {
      NATIVE_REFLECT_APPLY(effects.queueMicrotask, scope, [() => { throw cause; }]);
    });
  };
  NATIVE_REFLECT_APPLY(effects.addEventListener, scope, ["message", onMessage]);
}

/**
 * Runs strict ABI preparation, raw-Wasm inspection, engine validation, and
 * hashing entirely inside the disposable Worker realm.
 */
export async function handleCppCuteBrowserWasmVerifierLaunch(
  value: unknown,
  emitTerminal: (
    message: CppCuteBrowserWasmVerifierTerminalMessage,
    transfer: readonly ArrayBuffer[],
  ) => void,
): Promise<void> {
  const launch = parseLaunch(value);
  let terminalAttempted = false;
  const emitOnce = (
    message: CppCuteBrowserWasmVerifierTerminalMessage,
    transfer: readonly ArrayBuffer[],
  ): void => {
    if (terminalAttempted) {
      invalid("$.terminal", "verifier entry attempted more than one terminal emission");
    }
    // Set before entering caller code: an emitter that pushes and then throws
    // must never trigger a fallback terminal attempt.
    terminalAttempted = true;
    emitTerminal(message, transfer);
  };
  const identity = Object.freeze({
    requestId: launch.requestId,
    invocationNonceSha256: launch.invocationNonceSha256,
  });
  let runtimeAbi: Awaited<ReturnType<typeof decodeCppCuteBrowserRuntimeAbiManifest>>;
  try {
    runtimeAbi = await decodeCppCuteBrowserRuntimeAbiManifest(launch.runtimeAbiManifestBytes);
    if (runtimeAbi.manifestId !== launch.expectedRuntimeAbiManifestId ||
        runtimeAbi.contractSha256 !== launch.expectedRuntimeAbiContractSha256 ||
        runtimeAbi.resourceSha256 !== launch.expectedRuntimeAbiResourceSha256) {
      invalid(
        "$.runtimeAbiManifestBytes",
        "decoded runtime-ABI identities differ from the host-owned launch binding",
      );
    }
  } catch (cause) {
    emitOnce(failureMessage(identity, "runtime-abi", cause), []);
    return;
  }

  let report: ReturnType<typeof unwrapPreparedCppCuteBrowserWasmConformance>["summary"];
  try {
    const conformance = await verifyCppCuteBrowserWasmConformance(
      launch.wasmBytes,
      runtimeAbi,
      {
        maxModuleByteLength: launch.expectedWasmByteLength,
        maxOperations: launch.maxOperations,
      },
    );
    report = unwrapPreparedCppCuteBrowserWasmConformance(conformance).summary;
    if (report.wasmSha256 !== launch.expectedWasmSha256 ||
        report.wasmByteLength !== launch.expectedWasmByteLength ||
        report.runtimeAbiManifestId !== launch.expectedRuntimeAbiManifestId ||
        report.runtimeAbiContractSha256 !== launch.expectedRuntimeAbiContractSha256) {
      invalid("$.wasmBytes", "verified report differs from the host-owned launch binding");
    }
  } catch (cause) {
    emitOnce(failureMessage(identity, "raw-wasm", cause), []);
    return;
  }

  try {
    // Do not return the potentially large structural projection to the host.
    // Its canonical hash remains bound while the copy-safe result stays small.
    const summary: CppCuteBrowserWasmVerifierReportSummary = Object.freeze({
      authority: report.authority,
      wasmSha256: report.wasmSha256,
      wasmByteLength: report.wasmByteLength,
      observedProjectionSha256: report.observedProjectionSha256,
      runtimeAbiManifestId: report.runtimeAbiManifestId,
      runtimeAbiContractSha256: report.runtimeAbiContractSha256,
      exactInterfaceConformance: true,
      mismatches: Object.freeze([]) as readonly [],
      rawWasmVerified: true,
      workerExecutionReady: false,
      releaseReady: false,
    });
    const reportBytes = canonicalJsonBytes(summary as unknown as JsonValue);
    if (reportBytes.byteLength > CPP_CUTE_BROWSER_WASM_VERIFIER_REPORT_BYTE_LIMIT) {
      invalid(
        "$.reportBytes",
        `canonical report exceeds ${CPP_CUTE_BROWSER_WASM_VERIFIER_REPORT_BYTE_LIMIT} bytes`,
      );
    }
    const transferable = new Uint8Array(reportBytes) as Uint8Array<ArrayBuffer>;
    const terminal: CppCuteBrowserWasmVerifierSuccessMessage = Object.freeze({
      kind: "browsergrad-cpp-cute-wasm-verifier-success",
      version: CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
      protocol: CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
      ...identity,
      reportByteLength: transferable.byteLength,
      reportSha256: await sha256Hex(transferable),
      reportBytes: transferable,
      rawWasmVerified: true,
      verifierWorkerSelfAttested: false,
      productionConformanceAuthorityMinted: false,
      releaseReady: false,
    });
    emitOnce(terminal, Object.freeze([transferable.buffer]));
  } catch (cause) {
    if (terminalAttempted) throw cause;
    emitOnce(failureMessage(identity, "report-encoding", cause), []);
  }
}

function parseLaunch(value: unknown): CppCuteBrowserWasmVerifierLaunchMessage {
  const data = exactDataRecord(value, "$", [
    "kind", "version", "protocol", "requestId", "invocationNonceSha256", "wasmAssetId",
    "expectedWasmSha256", "expectedWasmByteLength", "expectedRuntimeAbiManifestId",
    "expectedRuntimeAbiContractSha256", "expectedRuntimeAbiResourceSha256", "maxOperations",
    "runtimeAbiManifestBytes", "wasmBytes",
  ]);
  literal(data["kind"], "browsergrad-cpp-cute-wasm-verifier-launch", "$.kind");
  literal(data["protocol"], CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL, "$.protocol");
  const version = exactDataRecord(data["version"], "$.version", ["major", "minor"]);
  literal(version["major"], CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR, "$.version.major");
  literal(version["minor"], CPP_CUTE_BROWSER_WASM_VERIFIER_MINOR, "$.version.minor");
  const expectedWasmByteLength = positiveInteger(
    data["expectedWasmByteLength"],
    "$.expectedWasmByteLength",
  );
  const wasmBytes = exactBytes(data["wasmBytes"], "$.wasmBytes");
  if (wasmBytes.byteLength !== expectedWasmByteLength) {
    invalid("$.wasmBytes", "transferred Wasm byte length differs from the expected binding");
  }
  return Object.freeze({
    kind: "browsergrad-cpp-cute-wasm-verifier-launch",
    version: Object.freeze({
      major: CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
      minor: CPP_CUTE_BROWSER_WASM_VERIFIER_MINOR,
    }),
    protocol: CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
    requestId: pattern(data["requestId"], REQUEST_ID, "$.requestId"),
    invocationNonceSha256: pattern(
      data["invocationNonceSha256"],
      SHA256_HEX,
      "$.invocationNonceSha256",
    ),
    wasmAssetId: pattern(data["wasmAssetId"], ASSET_ID, "$.wasmAssetId"),
    expectedWasmSha256: pattern(data["expectedWasmSha256"], SHA256_HEX, "$.expectedWasmSha256"),
    expectedWasmByteLength,
    expectedRuntimeAbiManifestId: pattern(
      data["expectedRuntimeAbiManifestId"],
      ABI_ID,
      "$.expectedRuntimeAbiManifestId",
    ),
    expectedRuntimeAbiContractSha256: pattern(
      data["expectedRuntimeAbiContractSha256"],
      SHA256_HEX,
      "$.expectedRuntimeAbiContractSha256",
    ),
    expectedRuntimeAbiResourceSha256: pattern(
      data["expectedRuntimeAbiResourceSha256"],
      SHA256_HEX,
      "$.expectedRuntimeAbiResourceSha256",
    ),
    maxOperations: positiveInteger(data["maxOperations"], "$.maxOperations"),
    runtimeAbiManifestBytes: exactBytes(
      data["runtimeAbiManifestBytes"],
      "$.runtimeAbiManifestBytes",
    ),
    wasmBytes,
  });
}

function failureMessage(
  identity: LaunchIdentity,
  phase: CppCuteBrowserWasmVerifierFailurePhase,
  cause: unknown,
): CppCuteBrowserWasmVerifierFailureMessage {
  const projection = errorProjection(cause, phase);
  return Object.freeze({
    kind: "browsergrad-cpp-cute-wasm-verifier-failure",
    version: CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR,
    protocol: CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL,
    ...identity,
    phase,
    failureCode: projection.code,
    failurePath: projection.path,
    rawWasmVerified: false,
    verifierWorkerSelfAttested: false,
    productionConformanceAuthorityMinted: false,
    releaseReady: false,
  });
}

function errorProjection(
  cause: unknown,
  phase: CppCuteBrowserWasmVerifierFailurePhase,
): { readonly code: string; readonly path: string } {
  if (typeof cause === "object" && cause !== null) {
    try {
      const descriptors = Object.getOwnPropertyDescriptors(cause);
      const code = descriptors["code"]?.value;
      const path = descriptors["path"]?.value;
      if (typeof code === "string" && FAILURE_CODE.test(code) &&
          typeof path === "string" && validPath(path)) {
        return Object.freeze({ code, path });
      }
    } catch {
      // The bounded internal fallback below does not expose untrusted errors.
    }
  }
  return Object.freeze({
    code: ENTRY_INVALID_CODE,
    path: phase === "runtime-abi" ? "$.runtimeAbiManifestBytes" :
      phase === "raw-wasm" ? "$.wasmBytes" : "$.reportBytes",
  });
}

function exactScope(scope: CppCuteBrowserWasmVerifierEntryScope): CppCuteBrowserWasmVerifierEntryScope {
  if (typeof scope !== "object" || scope === null) invalid("$.scope", "expected Worker scope");
  const addEventListener = snapshotScopeCallable(scope, "addEventListener");
  const removeEventListener = snapshotScopeCallable(scope, "removeEventListener");
  const postMessage = snapshotScopeCallable(scope, "postMessage");
  const queueMicrotask = snapshotScopeCallable(scope, "queueMicrotask");
  return Object.freeze({
    addEventListener: addEventListener as CppCuteBrowserWasmVerifierEntryScope["addEventListener"],
    removeEventListener: removeEventListener as CppCuteBrowserWasmVerifierEntryScope["removeEventListener"],
    postMessage: postMessage as CppCuteBrowserWasmVerifierEntryScope["postMessage"],
    queueMicrotask: queueMicrotask as CppCuteBrowserWasmVerifierEntryScope["queueMicrotask"],
  });
}

function snapshotScopeCallable(scope: object, key: string): (...arguments_: never[]) => unknown {
  let owner: object | null = scope;
  let depth = 0;
  while (owner !== null && depth < 32) {
    const descriptor = NATIVE_REFLECT_APPLY(
      NATIVE_GET_OWN_PROPERTY_DESCRIPTOR,
      Object,
      [owner, key],
    ) as PropertyDescriptor | undefined;
    if (descriptor !== undefined) {
      if (!("value" in descriptor) || typeof descriptor.value !== "function") {
        invalid(`$.scope.${key}`, "Worker effect must be a getter-free callable data property");
      }
      return descriptor.value as (...arguments_: never[]) => unknown;
    }
    owner = NATIVE_REFLECT_APPLY(NATIVE_GET_PROTOTYPE_OF, Object, [owner]) as object | null;
    depth += 1;
  }
  invalid(`$.scope.${key}`, "missing bounded getter-free Worker effect");
}

function exactDataRecord(
  value: unknown,
  path: string,
  keys: readonly string[],
): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null ||
      (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null)) {
    invalid(path, "expected plain data object");
  }
  const ownKeys = Reflect.ownKeys(value);
  if (ownKeys.length !== keys.length ||
      ownKeys.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(path, `expected exactly data fields ${keys.join(", ")}`);
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const result: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "field must be one enumerable data property");
    }
    result[key] = descriptor.value;
  }
  return Object.freeze(result);
}

function exactBytes(value: unknown, path: string): Uint8Array {
  try {
    const inspection = inspectUnsharedPlainUint8Array(value);
    return copyInspectedUnsharedUint8Array(value, inspection);
  } catch (cause) {
    invalid(path, "expected unshared plain Uint8Array bytes", { cause });
  }
}

function positiveInteger(value: unknown, path: string): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    invalid(path, "expected positive safe integer");
  }
  return value as number;
}

function pattern(value: unknown, expected: RegExp, path: string): string {
  if (typeof value !== "string" || !expected.test(value)) {
    invalid(path, `string does not match ${expected.source}`);
  }
  return value;
}

function literal(value: unknown, expected: unknown, path: string): void {
  if (value !== expected) invalid(path, `expected ${String(expected)}`);
}

function validPath(value: string): boolean {
  if (!value.startsWith("$") || value.length > 512) return false;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code <= 0x1f || code === 0x7f) return false;
  }
  return true;
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  throw new CppCuteBrowserWasmVerifierEntryError(
    ENTRY_INVALID_CODE,
    path,
    message,
    options,
  );
}
