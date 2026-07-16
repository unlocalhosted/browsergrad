import type { AuthorizedCppCuteAotOciMetadata } from "../dist/cpp_cute_aot_oci.js";
import type {
  BoundedChildProcessRequest,
  BoundedChildProcessResult,
} from "./cpp_cute_aot_docker_process.mjs";

export type CppCuteAotDockerImageErrorCode =
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-INVALID"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-PROCESS"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-EXIT"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-OUTPUT"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-CLEANUP"
  | "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-UNVERIFIED";

export class CppCuteAotDockerImageError extends Error {
  readonly code: CppCuteAotDockerImageErrorCode;
  readonly path: string;
}

declare const observedCppCuteAotLocalDockerImageBrand: unique symbol;

export interface ObservedCppCuteAotLocalDockerImage {
  readonly [observedCppCuteAotLocalDockerImageBrand]: true;
  readonly jobId: string;
  readonly profileHash: string;
  readonly executionPlanSha256: string;
  readonly imageReference: string;
  readonly manifestDigest: string;
  readonly imageId: string;
  readonly configDigest: string;
  readonly platform: "linux/amd64";
  readonly dockerClientVersion: "29.6.1";
  readonly dockerEngineVersion: "29.6.1";
  readonly dockerRequestApiVersion: "1.49";
  readonly dockerEngineApiVersion: "1.55";
  readonly dockerEngineMinApiVersion: "1.40";
  readonly dockerImageStore: "containerd";
  readonly layerCount: number;
  readonly totalLayerBytes: number;
}

export interface ObservedCppCuteAotLocalDockerImageRecord {
  readonly authorizedMetadata: AuthorizedCppCuteAotOciMetadata;
  readonly shellSession: object;
  readonly repoDigests: readonly string[];
}

export interface DockerObservationSession {
  readonly runRoot: string;
  readonly configDirectory: string;
  readonly homeDirectory: string;
  readonly authorizedMetadata: AuthorizedCppCuteAotOciMetadata;
  readonly processAdapter: (
    request: BoundedChildProcessRequest,
  ) => Promise<BoundedChildProcessResult>;
  readonly preserveRunRoot: () => void;
  readonly signal?: AbortSignal;
}

export interface CppCuteAotDockerImageSessionResult<T> {
  readonly observed: ObservedCppCuteAotLocalDockerImage;
  readonly value: T;
}

export function observeCppCuteAotLocalDockerImage(
  authorizedMetadata: AuthorizedCppCuteAotOciMetadata,
  options?: { readonly signal?: AbortSignal },
): Promise<ObservedCppCuteAotLocalDockerImage>;

export function __observeCppCuteAotLocalDockerImageWithProcessForTest(
  authorizedMetadata: AuthorizedCppCuteAotOciMetadata,
  processAdapter: (
    request: BoundedChildProcessRequest,
  ) => Promise<BoundedChildProcessResult>,
  options?: { readonly signal?: AbortSignal },
): Promise<ObservedCppCuteAotLocalDockerImage>;

export function __runCppCuteAotLocalDockerImageSession<T>(
  authorizedMetadata: AuthorizedCppCuteAotOciMetadata,
  continuation: (session: DockerObservationSession) => Promise<T>,
  options?: { readonly signal?: AbortSignal },
): Promise<Readonly<CppCuteAotDockerImageSessionResult<T>>>;

export function __runCppCuteAotLocalDockerImageSessionWithProcessForTest<T>(
  authorizedMetadata: AuthorizedCppCuteAotOciMetadata,
  processAdapter: (
    request: BoundedChildProcessRequest,
  ) => Promise<BoundedChildProcessResult>,
  continuation: (session: DockerObservationSession) => Promise<T>,
  options?: { readonly signal?: AbortSignal },
): Promise<Readonly<CppCuteAotDockerImageSessionResult<T>>>;

export function unwrapObservedCppCuteAotLocalDockerImage(
  observed: ObservedCppCuteAotLocalDockerImage,
): ObservedCppCuteAotLocalDockerImageRecord;

export function __unwrapObservedCppCuteAotLocalDockerImageForTest(
  observed: ObservedCppCuteAotLocalDockerImage,
): ObservedCppCuteAotLocalDockerImageRecord;
