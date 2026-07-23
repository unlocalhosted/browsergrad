import { describe, expect, it } from "vitest";

import {
  ASSIGNMENT_REQUIREMENT_DEFINITION_SCHEMA,
  ASSIGNMENT_REQUIREMENT_RESOLUTION_SCHEMA,
  createAssignmentRequirementDefinition,
  createAssignmentRequirementResolution,
} from "../../src/requirement.js";

const DEFINITION = createAssignmentRequirementDefinition({
  requirementId: "webgpu",
  semanticVersion: "0.1.0",
  kind: "device-feature",
  owner: "@unlocalhosted/browsergrad-runtime",
  lifecycle: "legacy",
  meaning: "A named environment provider reports WebGPU availability.",
});

describe("assignment requirement protocol", () => {
  it("creates immutable versioned definitions", () => {
    expect(DEFINITION).toEqual({
      schema: ASSIGNMENT_REQUIREMENT_DEFINITION_SCHEMA,
      schemaVersion: 1,
      requirementId: "webgpu",
      semanticVersion: "0.1.0",
      kind: "device-feature",
      owner: "@unlocalhosted/browsergrad-runtime",
      lifecycle: "legacy",
      meaning: "A named environment provider reports WebGPU availability.",
    });
    expect(Object.isFrozen(DEFINITION)).toBe(true);
  });

  it("binds available resolutions to one provider and sorted evidence", () => {
    const resolution = createAssignmentRequirementResolution(DEFINITION, {
      environmentId: "browser.local",
      status: "available",
      provider: {
        providerId: "navigator.gpu",
        mode: "browser",
        evidenceIds: ["probe.adapter", "probe.api", "probe.adapter"],
      },
    });
    expect(resolution).toEqual({
      schema: ASSIGNMENT_REQUIREMENT_RESOLUTION_SCHEMA,
      schemaVersion: 1,
      environmentId: "browser.local",
      requirementId: "webgpu",
      definitionVersion: "0.1.0",
      kind: "device-feature",
      status: "available",
      provider: {
        providerId: "navigator.gpu",
        mode: "browser",
        evidenceIds: ["probe.adapter", "probe.api"],
      },
    });
    expect(Object.isFrozen(resolution)).toBe(true);
    expect(
      resolution.status === "available"
        && Object.isFrozen(resolution.provider)
        && Object.isFrozen(resolution.provider.evidenceIds),
    ).toBe(true);
  });

  it("represents absence without inventing a provider or evidence", () => {
    expect(createAssignmentRequirementResolution(DEFINITION, {
      environmentId: "browser.local",
      status: "unavailable",
      diagnostic: "  WebGPU probe returned no adapter  ",
    })).toEqual({
      schema: ASSIGNMENT_REQUIREMENT_RESOLUTION_SCHEMA,
      schemaVersion: 1,
      environmentId: "browser.local",
      requirementId: "webgpu",
      definitionVersion: "0.1.0",
      kind: "device-feature",
      status: "unavailable",
      diagnostic: "WebGPU probe returned no adapter",
    });
  });

  it("rejects malformed definitions and unbound availability", () => {
    expect(() => createAssignmentRequirementDefinition({
      ...DEFINITION,
      requirementId: "Web GPU",
    })).toThrow(/requirementId is malformed/u);
    expect(() => createAssignmentRequirementDefinition({
      ...DEFINITION,
      semanticVersion: "latest",
    })).toThrow(/semanticVersion is malformed/u);
    expect(() => createAssignmentRequirementDefinition({
      ...DEFINITION,
      capabilityId: "browsergrad.layout.index-map",
    })).toThrow(/requires kind semantic-feature/u);
    expect(() => createAssignmentRequirementDefinition({
      ...DEFINITION,
      kind: "semantic-feature",
      capabilityId: "index-map",
    })).toThrow(/capabilityId is malformed/u);
    expect(() => createAssignmentRequirementResolution(DEFINITION, {
      environmentId: "",
      status: "unavailable",
    })).toThrow(/environmentId must be a non-empty string/u);
    expect(() => createAssignmentRequirementResolution(DEFINITION, {
      environmentId: "browser.local",
      status: "available",
      provider: {
        providerId: "",
        mode: "browser",
      },
    })).toThrow(/providerId must be a non-empty string/u);
  });
});
