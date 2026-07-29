import type {
  PrepareSemanticViewCopyWgslRequest,
  PreparedSemanticViewCopyWgsl,
  SemanticViewCopyResidentRunOptions,
  SemanticViewCopyResidentSource,
} from "./semantic_view_copy.js";
import type { VerifiedLayoutArtifact } from "@unlocalhosted/browsergrad-semantic-core/layout";
import type { VerifiedKernelArtifact } from "@unlocalhosted/browsergrad-semantic-core/kernel";
import type { DirectDispatchResult } from "./runner.js";
import type { KernelDevice } from "./types.js";

type ResidentIssuer = (
  device: KernelDevice,
  prepared: PreparedSemanticViewCopyWgsl,
  source: SemanticViewCopyResidentSource,
  options?: SemanticViewCopyResidentRunOptions,
) => DirectDispatchResult;

let residentIssuer: ResidentIssuer | undefined;

export type SemanticViewCopyDynamicLaunchMode =
  | "linear-prefix"
  | "rectangular-prefix";

export interface PreparedSemanticViewCopyDynamicWgsl
  extends PreparedSemanticViewCopyWgsl {
  readonly dynamicLaunchMode: SemanticViewCopyDynamicLaunchMode;
  readonly dynamicUniformName: string;
  readonly dynamicUniformByteLength: 4 | 16 | 32;
}

type DynamicViewCopyPreparer = (
  layoutArtifact: VerifiedLayoutArtifact,
  kernelArtifact: VerifiedKernelArtifact,
  request: PrepareSemanticViewCopyWgslRequest,
  mode: SemanticViewCopyDynamicLaunchMode,
) => Promise<PreparedSemanticViewCopyDynamicWgsl>;

let dynamicViewCopyPreparer: DynamicViewCopyPreparer | undefined;

/** @internal Registers the module-authorized synchronous issue capability once. */
export function registerPreparedSemanticViewCopyResidentIssuer(
  issuer: ResidentIssuer,
): void {
  if (residentIssuer !== undefined && residentIssuer !== issuer) {
    throw new Error("semantic view-copy resident issuer was registered twice");
  }
  residentIssuer = issuer;
}

/** @internal Registers the canonical dynamic-domain WGSL lowerer once. */
export function registerSemanticViewCopyDynamicPreparer(
  preparer: DynamicViewCopyPreparer,
): void {
  if (
    dynamicViewCopyPreparer !== undefined &&
    dynamicViewCopyPreparer !== preparer
  ) {
    throw new Error(
      "semantic view-copy dynamic preparer was registered twice",
    );
  }
  dynamicViewCopyPreparer = preparer;
}

/** @internal Prepares one canonical view-copy with a runtime launch guard. */
export function prepareSemanticViewCopyDynamicWgsl(
  layoutArtifact: VerifiedLayoutArtifact,
  kernelArtifact: VerifiedKernelArtifact,
  request: PrepareSemanticViewCopyWgslRequest,
  mode: SemanticViewCopyDynamicLaunchMode,
): Promise<PreparedSemanticViewCopyDynamicWgsl> {
  if (dynamicViewCopyPreparer === undefined) {
    throw new Error(
      "semantic view-copy dynamic preparer is not initialized",
    );
  }
  return dynamicViewCopyPreparer(
    layoutArtifact,
    kernelArtifact,
    request,
    mode,
  );
}

/**
 * @internal Synchronous GPU issue seam used only under tensor-plan-owned error
 * scopes. This module has no public package export.
 */
export function issuePreparedSemanticViewCopyResidentUnchecked(
  device: KernelDevice,
  prepared: PreparedSemanticViewCopyWgsl,
  source: SemanticViewCopyResidentSource,
  options?: SemanticViewCopyResidentRunOptions,
): DirectDispatchResult {
  if (residentIssuer === undefined) {
    throw new Error("semantic view-copy resident issuer is not initialized");
  }
  return residentIssuer(device, prepared, source, options);
}
