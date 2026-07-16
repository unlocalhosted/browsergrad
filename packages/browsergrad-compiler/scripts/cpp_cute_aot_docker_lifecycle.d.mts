import type { AuthorizedCppCuteAotOciMetadata } from "../dist/cpp_cute_aot_oci.js";
import type {
  VerifiedCppCuteAotOfflineResult,
} from "../dist/cpp_cute_aot_runner_plan.js";
import type {
  BoundedChildProcessRequest,
  BoundedChildProcessResult,
} from "./cpp_cute_aot_docker_process.mjs";
import type {
  ObservedCppCuteAotLocalDockerImage,
} from "./cpp_cute_aot_docker_shell.mjs";

export type CppCuteAotDockerRunErrorCode =
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-HOST"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-INVALID"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-STAGING"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-PROCESS"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CREATE"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CONTAINER-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-START"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-TERMINAL"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CLEANUP"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-UNVERIFIED";

export class CppCuteAotDockerRunError extends Error {
  readonly code: CppCuteAotDockerRunErrorCode;
  readonly path: string;
}

declare const completedCppCuteAotDockerRunBrand: unique symbol;

export interface CompletedCppCuteAotDockerRun {
  readonly [completedCppCuteAotDockerRunBrand]: true;
  readonly jobId: string;
  readonly profileHash: string;
  readonly executionPlanSha256: string;
  readonly containerId: string;
  readonly imageId: string;
  readonly frontendOutcome: "accepted" | "rejected";
}

export interface CompletedCppCuteAotDockerRunRecord {
  readonly observedImage: ObservedCppCuteAotLocalDockerImage;
  readonly result: VerifiedCppCuteAotOfflineResult;
  readonly evidence: object;
}

export function executeCppCuteAotDockerRun(
  authorizedMetadata: AuthorizedCppCuteAotOciMetadata,
  options?: { readonly signal?: AbortSignal },
): Promise<CompletedCppCuteAotDockerRun>;

export function __executeCppCuteAotDockerRunWithProcessForTest(
  authorizedMetadata: AuthorizedCppCuteAotOciMetadata,
  processAdapter: (
    request: BoundedChildProcessRequest,
  ) => Promise<BoundedChildProcessResult>,
  options?: { readonly signal?: AbortSignal },
): Promise<CompletedCppCuteAotDockerRun>;

export function unwrapCompletedCppCuteAotDockerRun(
  completed: CompletedCppCuteAotDockerRun,
): CompletedCppCuteAotDockerRunRecord;

export function __unwrapCompletedCppCuteAotDockerRunForTest(
  completed: CompletedCppCuteAotDockerRun,
): CompletedCppCuteAotDockerRunRecord;
