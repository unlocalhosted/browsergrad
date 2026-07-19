import type { CppCuteBrowserWasmInspectionReport } from "./cpp_cute_browser_wasm_inspection.js";

export const CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL =
  "browsergrad.compiler.cpp-cute.browser-wasm-verifier@1";

export const CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR = 1;
export const CPP_CUTE_BROWSER_WASM_VERIFIER_MINOR = 0;
/** The Worker returns only the conformance summary, never its large projection. */
export const CPP_CUTE_BROWSER_WASM_VERIFIER_REPORT_BYTE_LIMIT = 64 * 1024;

/**
 * One-shot launch envelope transferred to the disposable verifier Worker.
 * The host owns every expected identity; the Worker may only confirm them.
 */
export interface CppCuteBrowserWasmVerifierLaunchMessage {
  readonly kind: "browsergrad-cpp-cute-wasm-verifier-launch";
  readonly version: {
    readonly major: typeof CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR;
    readonly minor: typeof CPP_CUTE_BROWSER_WASM_VERIFIER_MINOR;
  };
  readonly protocol: typeof CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL;
  readonly requestId: string;
  readonly invocationNonceSha256: string;
  readonly wasmAssetId: string;
  readonly expectedWasmSha256: string;
  readonly expectedWasmByteLength: number;
  readonly expectedRuntimeAbiManifestId: string;
  readonly expectedRuntimeAbiContractSha256: string;
  readonly expectedRuntimeAbiResourceSha256: string;
  readonly maxOperations: number;
  readonly runtimeAbiManifestBytes: Uint8Array;
  readonly wasmBytes: Uint8Array;
}

/** Copy-safe report subset retained by the host after strict wire validation. */
export interface CppCuteBrowserWasmVerifierReportSummary {
  readonly authority: CppCuteBrowserWasmInspectionReport["authority"];
  readonly wasmSha256: string;
  readonly wasmByteLength: number;
  readonly observedProjectionSha256: string;
  readonly runtimeAbiManifestId: string;
  readonly runtimeAbiContractSha256: string;
  readonly exactInterfaceConformance: true;
  readonly mismatches: readonly [];
  readonly rawWasmVerified: true;
  readonly workerExecutionReady: false;
  readonly releaseReady: false;
}

export interface CppCuteBrowserWasmVerifierSuccessMessage {
  readonly kind: "browsergrad-cpp-cute-wasm-verifier-success";
  readonly version: typeof CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR;
  readonly protocol: typeof CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL;
  readonly requestId: string;
  readonly invocationNonceSha256: string;
  readonly reportByteLength: number;
  readonly reportSha256: string;
  readonly reportBytes: Uint8Array;
  readonly rawWasmVerified: true;
  readonly verifierWorkerSelfAttested: false;
  readonly productionConformanceAuthorityMinted: false;
  readonly releaseReady: false;
}

export type CppCuteBrowserWasmVerifierFailurePhase =
  | "runtime-abi"
  | "raw-wasm"
  | "report-encoding";

export interface CppCuteBrowserWasmVerifierFailureMessage {
  readonly kind: "browsergrad-cpp-cute-wasm-verifier-failure";
  readonly version: typeof CPP_CUTE_BROWSER_WASM_VERIFIER_MAJOR;
  readonly protocol: typeof CPP_CUTE_BROWSER_WASM_VERIFIER_PROTOCOL;
  readonly requestId: string;
  readonly invocationNonceSha256: string;
  readonly phase: CppCuteBrowserWasmVerifierFailurePhase;
  readonly failureCode: string;
  readonly failurePath: string;
  readonly rawWasmVerified: false;
  readonly verifierWorkerSelfAttested: false;
  readonly productionConformanceAuthorityMinted: false;
  readonly releaseReady: false;
}

export type CppCuteBrowserWasmVerifierTerminalMessage =
  | CppCuteBrowserWasmVerifierSuccessMessage
  | CppCuteBrowserWasmVerifierFailureMessage;
