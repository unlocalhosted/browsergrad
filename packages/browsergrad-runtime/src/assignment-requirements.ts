import {
  ASSIGNMENT_REQUIREMENT_RESOLUTION_SCHEMA,
  ASSIGNMENT_REQUIREMENT_SCHEMA_VERSION,
  createAssignmentRequirementResolution,
  type AssignmentRequirementDefinition,
  type AssignmentRequirementProviderInput,
  type AssignmentRequirementResolution,
} from "@unlocalhosted/browsergrad-semantic-core/requirement";

import { createAssignmentCapabilityEnvironment, evaluateAssignmentCapabilities } from "./assignment-capabilities.js";
import { ASSIGNMENT_REQUIREMENT_DEFINITIONS } from "./assignment-requirement-registry.generated.js";
import type {
  AssignmentCapabilityEnvironment,
  AssignmentCapabilityEvaluation,
  AssignmentProfile,
} from "./assignment-types.js";

export const ASSIGNMENT_REQUIREMENT_ENVIRONMENT_SCHEMA =
  "browsergrad.assignment-requirement-environment" as const;
export const ASSIGNMENT_REQUIREMENT_ENVIRONMENT_SCHEMA_VERSION = 1 as const;

export interface AssignmentRequirementProviderRecord
  extends AssignmentRequirementProviderInput {
  readonly requirementId: string;
}

export interface AssignmentRequirementResolutionEnvironmentInput {
  readonly environmentId: string;
  readonly providers?: readonly AssignmentRequirementProviderRecord[];
}

export interface AssignmentRequirementResolutionEnvironment {
  readonly schema: typeof ASSIGNMENT_REQUIREMENT_ENVIRONMENT_SCHEMA;
  readonly schemaVersion: typeof ASSIGNMENT_REQUIREMENT_ENVIRONMENT_SCHEMA_VERSION;
  readonly environmentId: string;
  readonly resolutions: readonly AssignmentRequirementResolution[];
}

const DEFINITIONS_BY_ID = new Map(
  ASSIGNMENT_REQUIREMENT_DEFINITIONS.map((definition) => [
    definition.requirementId,
    definition,
  ]),
);

export function assignmentRequirementDefinitions():
  readonly AssignmentRequirementDefinition[] {
  return ASSIGNMENT_REQUIREMENT_DEFINITIONS;
}

export function assignmentRequirementDefinition(
  requirementId: string,
): AssignmentRequirementDefinition | undefined {
  return DEFINITIONS_BY_ID.get(requirementId);
}

export function createAssignmentRequirementResolutionEnvironment(
  input: AssignmentRequirementResolutionEnvironmentInput,
): AssignmentRequirementResolutionEnvironment {
  const environmentId = requireNonEmpty(input.environmentId, "environmentId");
  const providers = new Map<string, AssignmentRequirementProviderInput>();
  for (const provider of input.providers ?? []) {
    const requirementId = requireNonEmpty(
      provider.requirementId,
      "requirementId",
    );
    if (!DEFINITIONS_BY_ID.has(requirementId)) {
      throw new TypeError(
        `assignment requirement provider references unknown requirement: ${requirementId}`,
      );
    }
    if (providers.has(requirementId)) {
      throw new TypeError(
        `assignment requirement has more than one provider: ${requirementId}`,
      );
    }
    providers.set(requirementId, {
      providerId: provider.providerId,
      mode: provider.mode,
      ...(provider.evidenceIds === undefined
        ? {}
        : { evidenceIds: [...provider.evidenceIds] }),
    });
  }

  const resolutions = ASSIGNMENT_REQUIREMENT_DEFINITIONS.map((definition) => {
    const provider = providers.get(definition.requirementId);
    return createAssignmentRequirementResolution(
      definition,
      provider === undefined
        ? { environmentId, status: "unavailable" }
        : { environmentId, status: "available", provider },
    );
  });
  return Object.freeze({
    schema: ASSIGNMENT_REQUIREMENT_ENVIRONMENT_SCHEMA,
    schemaVersion: ASSIGNMENT_REQUIREMENT_ENVIRONMENT_SCHEMA_VERSION,
    environmentId,
    resolutions: Object.freeze(resolutions),
  });
}

export function assignmentCapabilityEnvironmentFromRequirementResolutions(
  environment: AssignmentRequirementResolutionEnvironment,
): AssignmentCapabilityEnvironment {
  validateResolutionEnvironment(environment);
  return createAssignmentCapabilityEnvironment({
    browserCapabilities: availableRequirementIds(environment, "browser"),
    simulatedCapabilities: availableRequirementIds(environment, "simulated"),
    externalCapabilities: availableRequirementIds(environment, "external"),
  });
}

export function evaluateAssignmentRequirementResolutions(
  profile: AssignmentProfile,
  environment: AssignmentRequirementResolutionEnvironment,
): AssignmentCapabilityEvaluation {
  return evaluateAssignmentCapabilities(
    profile,
    assignmentCapabilityEnvironmentFromRequirementResolutions(environment),
  );
}

function availableRequirementIds(
  environment: AssignmentRequirementResolutionEnvironment,
  mode: "browser" | "simulated" | "external",
): string[] {
  return environment.resolutions.flatMap((resolution) =>
    resolution.status === "available" && resolution.provider.mode === mode
      ? [resolution.requirementId]
      : []
  );
}

function validateResolutionEnvironment(
  environment: AssignmentRequirementResolutionEnvironment,
): void {
  if (
    environment.schema !== ASSIGNMENT_REQUIREMENT_ENVIRONMENT_SCHEMA
    || environment.schemaVersion
      !== ASSIGNMENT_REQUIREMENT_ENVIRONMENT_SCHEMA_VERSION
  ) {
    throw new TypeError(
      "assignment requirement resolution environment schema is not supported",
    );
  }
  const environmentId = requireNonEmpty(
    environment.environmentId,
    "environmentId",
  );
  if (
    !Array.isArray(environment.resolutions)
    || environment.resolutions.length !== ASSIGNMENT_REQUIREMENT_DEFINITIONS.length
  ) {
    throw new TypeError(
      "assignment requirement resolution environment must resolve every registered definition exactly once",
    );
  }
  const observed = new Set<string>();
  for (const resolution of environment.resolutions) {
    const definition = DEFINITIONS_BY_ID.get(resolution.requirementId);
    if (definition === undefined || observed.has(resolution.requirementId)) {
      throw new TypeError(
        `assignment requirement resolution is unknown or duplicated: ${resolution.requirementId}`,
      );
    }
    if (
      resolution.schema !== ASSIGNMENT_REQUIREMENT_RESOLUTION_SCHEMA
      || resolution.schemaVersion !== ASSIGNMENT_REQUIREMENT_SCHEMA_VERSION
      || (
        resolution.status !== "available"
        && resolution.status !== "unavailable"
      )
      || resolution.environmentId !== environmentId
      || resolution.definitionVersion !== definition.semanticVersion
      || resolution.kind !== definition.kind
      || resolution.capabilityId !== definition.capabilityId
    ) {
      throw new TypeError(
        `assignment requirement resolution does not match its definition: ${resolution.requirementId}`,
      );
    }
    createAssignmentRequirementResolution(
      definition,
      resolution.status === "available"
        ? {
            environmentId,
            status: "available",
            provider: resolution.provider,
          }
        : {
            environmentId,
            status: "unavailable",
            ...(resolution.diagnostic === undefined
              ? {}
              : { diagnostic: resolution.diagnostic }),
          },
    );
    observed.add(resolution.requirementId);
  }
}

function requireNonEmpty(value: string, field: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(
      `assignment requirement environment ${field} must be a non-empty string`,
    );
  }
  return value;
}
