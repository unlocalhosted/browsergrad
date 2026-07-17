import {
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE,
} from "./resources/cpp_cute_semantic_adapter_manifest_v1.js";

const TEMPORAL_POLICY =
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE.body.temporalMacros;
const WARNING_REGISTRY =
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE.body.warningPolicyRegistry;

export const CPP_CUTE_FRONTEND_TEMPORAL_MACRO_POLICY_ID =
  TEMPORAL_POLICY.policyId;
export const CPP_CUTE_FRONTEND_WARNING_POLICY_REGISTRY_ID =
  WARNING_REGISTRY.registryId;
export const CPP_CUTE_FRONTEND_WARNING_BASELINE =
  WARNING_REGISTRY.compilerBaseline;
export const CPP_CUTE_FRONTEND_TEMPORAL_MACRO_NAMES =
  TEMPORAL_POLICY.macroNames;
export const CPP_CUTE_FRONTEND_WARNING_POLICY_MAPPINGS =
  WARNING_REGISTRY.entries;

export type CppCuteFrontendTemporalMacroName =
  (typeof CPP_CUTE_FRONTEND_TEMPORAL_MACRO_NAMES)[number];
export type CppCuteFrontendWarningPolicyId =
  (typeof CPP_CUTE_FRONTEND_WARNING_POLICY_MAPPINGS)[number]["policyId"];
export type CppCuteFrontendWarningDisposition = "ignore" | "warn" | "error";

export function isCppCuteFrontendTemporalMacroName(
  name: string,
): name is CppCuteFrontendTemporalMacroName {
  return CPP_CUTE_FRONTEND_TEMPORAL_MACRO_NAMES.some((candidate) => candidate === name);
}

export function isCppCuteFrontendReservedMacroName(name: string): boolean {
  return name === "defined" || name.startsWith("_") || name.includes("__");
}

export function cppCuteFrontendWarningPolicyMapping(
  id: string,
): (typeof CPP_CUTE_FRONTEND_WARNING_POLICY_MAPPINGS)[number] | undefined {
  return CPP_CUTE_FRONTEND_WARNING_POLICY_MAPPINGS.find(
    (mapping) => mapping.policyId === id,
  );
}

/** Exact trusted argv elements from package canonical policy authority. */
export function cppCuteFrontendWarningArguments(
  id: CppCuteFrontendWarningPolicyId,
  disposition: CppCuteFrontendWarningDisposition,
): readonly string[] {
  const mapping = cppCuteFrontendWarningPolicyMapping(id);
  if (mapping === undefined) throw new TypeError(`unknown warning policy ${id}`);
  return Object.freeze([...mapping.argv[disposition]]);
}
