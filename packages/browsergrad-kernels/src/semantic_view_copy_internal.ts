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

export interface PreparedSemanticViewCopyDynamicPrefixWgsl
  extends PreparedSemanticViewCopyWgsl {
  readonly dynamicPrefixUniformName: string;
}

type DynamicPrefixPreparer = (
  layoutArtifact: VerifiedLayoutArtifact,
  kernelArtifact: VerifiedKernelArtifact,
  request: PrepareSemanticViewCopyWgslRequest,
) => Promise<PreparedSemanticViewCopyDynamicPrefixWgsl>;

let dynamicPrefixPreparer: DynamicPrefixPreparer | undefined;

/** @internal Registers the module-authorized synchronous issue capability once. */
export function registerPreparedSemanticViewCopyResidentIssuer(
  issuer: ResidentIssuer,
): void {
  if (residentIssuer !== undefined && residentIssuer !== issuer) {
    throw new Error("semantic view-copy resident issuer was registered twice");
  }
  residentIssuer = issuer;
}

/** @internal Registers the canonical runtime-prefix WGSL lowerer once. */
export function registerSemanticViewCopyDynamicPrefixPreparer(
  preparer: DynamicPrefixPreparer,
): void {
  if (
    dynamicPrefixPreparer !== undefined &&
    dynamicPrefixPreparer !== preparer
  ) {
    throw new Error(
      "semantic view-copy dynamic-prefix preparer was registered twice",
    );
  }
  dynamicPrefixPreparer = preparer;
}

/** @internal Prepares one canonical view-copy with a runtime prefix guard. */
export function prepareSemanticViewCopyDynamicPrefixWgsl(
  layoutArtifact: VerifiedLayoutArtifact,
  kernelArtifact: VerifiedKernelArtifact,
  request: PrepareSemanticViewCopyWgslRequest,
): Promise<PreparedSemanticViewCopyDynamicPrefixWgsl> {
  if (dynamicPrefixPreparer === undefined) {
    throw new Error(
      "semantic view-copy dynamic-prefix preparer is not initialized",
    );
  }
  return dynamicPrefixPreparer(layoutArtifact, kernelArtifact, request);
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
