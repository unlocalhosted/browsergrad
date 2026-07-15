import type {
  PreparedSemanticViewCopyWgsl,
  SemanticViewCopyResidentRunOptions,
  SemanticViewCopyResidentSource,
} from "./semantic_view_copy.js";
import type { DirectDispatchResult } from "./runner.js";
import type { KernelDevice } from "./types.js";

type ResidentIssuer = (
  device: KernelDevice,
  prepared: PreparedSemanticViewCopyWgsl,
  source: SemanticViewCopyResidentSource,
  options?: SemanticViewCopyResidentRunOptions,
) => DirectDispatchResult;

let residentIssuer: ResidentIssuer | undefined;

/** @internal Registers the module-authorized synchronous issue capability once. */
export function registerPreparedSemanticViewCopyResidentIssuer(
  issuer: ResidentIssuer,
): void {
  if (residentIssuer !== undefined && residentIssuer !== issuer) {
    throw new Error("semantic view-copy resident issuer was registered twice");
  }
  residentIssuer = issuer;
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
