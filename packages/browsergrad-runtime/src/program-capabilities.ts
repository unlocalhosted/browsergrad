import {
  createLoweringDecision,
  type LoweringDecision,
  type LoweringDecisionInput,
  type LoweringDecisionSubject,
  type SemanticBackendDefinition,
  type SemanticCapabilityDefinition,
} from "@unlocalhosted/browsergrad-semantic-core/capability";

import {
  SEMANTIC_BACKEND_DEFINITIONS,
  SEMANTIC_CAPABILITY_DEFINITIONS,
} from "./program-capability-registry.generated.js";

export const PROGRAM_CAPABILITY_SUPPORT_VIEW_SCHEMA =
  "browsergrad.program-capability-support-view" as const;
export const PROGRAM_CAPABILITY_SUPPORT_VIEW_SCHEMA_VERSION = 1 as const;

export interface ProgramLoweringDecisionRecord
  extends Omit<LoweringDecisionInput, "subject"> {
  readonly capabilityId: string;
  readonly backendId: string;
}

export interface ProgramCapabilitySupportViewInput {
  readonly viewId: string;
  readonly subject: LoweringDecisionSubject;
  readonly decisions: readonly ProgramLoweringDecisionRecord[];
}

export interface ProgramCapabilitySupportView {
  readonly schema: typeof PROGRAM_CAPABILITY_SUPPORT_VIEW_SCHEMA;
  readonly schemaVersion: typeof PROGRAM_CAPABILITY_SUPPORT_VIEW_SCHEMA_VERSION;
  readonly viewId: string;
  readonly subject: LoweringDecisionSubject;
  readonly capabilities: readonly SemanticCapabilityDefinition[];
  readonly backends: readonly SemanticBackendDefinition[];
  readonly decisions: readonly LoweringDecision[];
}

const CAPABILITIES_BY_ID = new Map(
  SEMANTIC_CAPABILITY_DEFINITIONS.map((definition) => [
    definition.capabilityId,
    definition,
  ]),
);
const BACKENDS_BY_ID = new Map(
  SEMANTIC_BACKEND_DEFINITIONS.map((definition) => [
    definition.backendId,
    definition,
  ]),
);
const CANONICAL_ID =
  /^[a-z][a-z0-9]*(?:\.[a-z][a-z0-9-]*)+$/u;

export function semanticCapabilityDefinitions():
  readonly SemanticCapabilityDefinition[] {
  return SEMANTIC_CAPABILITY_DEFINITIONS;
}

export function semanticBackendDefinitions():
  readonly SemanticBackendDefinition[] {
  return SEMANTIC_BACKEND_DEFINITIONS;
}

export function createProgramCapabilitySupportView(
  input: ProgramCapabilitySupportViewInput,
): ProgramCapabilitySupportView {
  if (
    typeof input.viewId !== "string"
    || !CANONICAL_ID.test(input.viewId)
  ) {
    throw new TypeError(
      `program capability support viewId is malformed: ${String(input.viewId)}`,
    );
  }
  if (!Array.isArray(input.decisions) || input.decisions.length === 0) {
    throw new TypeError(
      "program capability support view requires at least one lowering decision",
    );
  }
  const observed = new Set<string>();
  const decisions = input.decisions.map((record) => {
    const capability = CAPABILITIES_BY_ID.get(record.capabilityId);
    if (capability === undefined) {
      throw new TypeError(
        `program lowering decision references unknown capability: ${record.capabilityId}`,
      );
    }
    const backend = BACKENDS_BY_ID.get(record.backendId);
    if (backend === undefined) {
      throw new TypeError(
        `program lowering decision references unknown backend: ${record.backendId}`,
      );
    }
    const identity = `${capability.capabilityId}\0${backend.backendId}`;
    if (observed.has(identity)) {
      throw new TypeError(
        `program capability support view has duplicate capability/backend decision: ${capability.capabilityId} / ${backend.backendId}`,
      );
    }
    observed.add(identity);
    return createLoweringDecision(capability, backend, {
      subject: input.subject,
      executionTier: record.executionTier,
      state: record.state,
      ...(record.preservationLevel === undefined
        ? {}
        : { preservationLevel: record.preservationLevel }),
      ...(record.requiredFeatures === undefined
        ? {}
        : { requiredFeatures: [...record.requiredFeatures] }),
      ...(record.requiredLimits === undefined
        ? {}
        : { requiredLimits: { ...record.requiredLimits } }),
      ...(record.runtimeGuardIds === undefined
        ? {}
        : { runtimeGuardIds: [...record.runtimeGuardIds] }),
      ...(record.legalizationIds === undefined
        ? {}
        : { legalizationIds: [...record.legalizationIds] }),
      ...(record.numericalPolicyId === undefined
        ? {}
        : { numericalPolicyId: record.numericalPolicyId }),
      ...(record.reasonCode === undefined
        ? {}
        : { reasonCode: record.reasonCode }),
    });
  });
  decisions.sort((left, right) =>
    left.capabilityId.localeCompare(right.capabilityId)
    || left.backendId.localeCompare(right.backendId)
  );
  const capabilityIds = new Set(
    decisions.map((decision) => decision.capabilityId),
  );
  const backendIds = new Set(
    decisions.map((decision) => decision.backendId),
  );
  return Object.freeze({
    schema: PROGRAM_CAPABILITY_SUPPORT_VIEW_SCHEMA,
    schemaVersion: PROGRAM_CAPABILITY_SUPPORT_VIEW_SCHEMA_VERSION,
    viewId: input.viewId,
    subject: decisions[0]!.subject,
    capabilities: Object.freeze(
      SEMANTIC_CAPABILITY_DEFINITIONS.filter((definition) =>
        capabilityIds.has(definition.capabilityId)
      ),
    ),
    backends: Object.freeze(
      SEMANTIC_BACKEND_DEFINITIONS.filter((definition) =>
        backendIds.has(definition.backendId)
      ),
    ),
    decisions: Object.freeze(decisions),
  });
}
