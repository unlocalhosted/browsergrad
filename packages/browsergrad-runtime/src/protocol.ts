/**
 * Internal RPC protocol between the main-thread client and the worker.
 *
 * NOT exported from index.ts. Anything in this file is private and may
 * change between any two versions. The public contract is in `./types.ts`.
 *
 * Wire format: every message is a discriminated union with a `kind` field.
 * `id` correlates requests with responses; `id: 0` is reserved for
 * unsolicited events (init progress) that don't correspond to a request.
 */

import type {
  Artifact,
  Assertion,
  ExecError,
  PackageProgressEvent,
} from "./types.js";

/* ────────────────────────────────────────────────────────────
 * Client → Worker
 * ──────────────────────────────────────────────────────────── */

export type ClientToWorker =
  | InitRequest
  | FsWriteRequest
  | FsReadRequest
  | ExecRequest
  | ClearNamespaceRequest;

export interface InitRequest {
  readonly id: number;
  readonly kind: "init";
  readonly pyodideIndexURL: string;
  readonly packages: readonly string[];
  /**
   * SharedArrayBuffer-backed Uint8Array for cooperative cancel.
   * Written by client (value 2 = SIGINT). Read by Pyodide via setInterruptBuffer.
   * Absent → worker runs in terminate-only cancel mode.
   */
  readonly interruptBuffer?: Uint8Array;
}

export interface FsWriteRequest {
  readonly id: number;
  readonly kind: "fs.write";
  readonly path: string;
  readonly content: string;
}

export interface FsReadRequest {
  readonly id: number;
  readonly kind: "fs.read";
  readonly path: string;
}

export interface ExecRequest {
  readonly id: number;
  readonly kind: "exec";
  readonly code: string;
}

export interface ClearNamespaceRequest {
  readonly id: number;
  readonly kind: "clearNamespace";
}

/* ────────────────────────────────────────────────────────────
 * Worker → Client
 * ──────────────────────────────────────────────────────────── */

export type WorkerToClient =
  | InitProgressEvent
  | InitDoneResponse
  | FsWriteResponse
  | FsReadResponse
  | ExecStdoutEvent
  | ExecStderrEvent
  | ExecAssertionEvent
  | ExecArtifactEvent
  | ExecDoneResponse
  | ClearNamespaceResponse
  | ErrorResponse;

export interface InitProgressEvent {
  readonly id: 0;
  readonly kind: "init:progress";
  readonly event: PackageProgressEvent;
}

export interface InitDoneResponse {
  readonly id: number;
  readonly kind: "init:done";
}

export interface FsWriteResponse {
  readonly id: number;
  readonly kind: "fs.write:done";
}

export interface FsReadResponse {
  readonly id: number;
  readonly kind: "fs.read:done";
  readonly content: string;
}

export interface ExecStdoutEvent {
  readonly id: number;
  readonly kind: "exec:stdout";
  readonly chunk: string;
}

export interface ExecStderrEvent {
  readonly id: number;
  readonly kind: "exec:stderr";
  readonly chunk: string;
}

export interface ExecAssertionEvent {
  readonly id: number;
  readonly kind: "exec:assertion";
  readonly assertion: Assertion;
}

export interface ExecArtifactEvent {
  readonly id: number;
  readonly kind: "exec:artifact";
  readonly artifact: Artifact;
}

export interface ExecDoneResponse {
  readonly id: number;
  readonly kind: "exec:done";
  readonly ok: boolean;
  readonly durationMs: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly error: ExecError | null;
}

export interface ClearNamespaceResponse {
  readonly id: number;
  readonly kind: "clearNamespace:done";
}

export interface ErrorResponse {
  readonly id: number;
  readonly kind: "error";
  readonly message: string;
}
