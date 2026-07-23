import { describe, expect, it } from "vitest";

import { VALID_PROFILE } from "./assignment-fixtures";
import {
  ASSIGNMENT_REQUIREMENT_ENVIRONMENT_SCHEMA,
  assignmentCapabilityEnvironmentFromRequirementResolutions,
  assignmentRequirementDefinition,
  assignmentRequirementDefinitions,
  createAssignmentRequirementResolutionEnvironment,
  evaluateAssignmentRequirementResolutions,
  parseAssignmentProfile,
  type AssignmentRequirementResolutionEnvironment,
} from "../src/index";

describe("assignment requirement resolutions", () => {
  it("exposes the generated immutable vocabulary registry", () => {
    const definitions = assignmentRequirementDefinitions();
    expect(definitions).toHaveLength(53);
    expect(Object.isFrozen(definitions)).toBe(true);
    expect(definitions.map((definition) => definition.requirementId)).toEqual(
      definitions
        .map((definition) => definition.requirementId)
        .sort((left, right) => left.localeCompare(right)),
    );
    expect(new Set(
      definitions.map((definition) => definition.requirementId),
    )).toHaveLength(53);
    expect(assignmentRequirementDefinition("webgpu")).toMatchObject({
      requirementId: "webgpu",
      semanticVersion: "0.1.0",
      kind: "device-feature",
      lifecycle: "legacy",
    });
    expect(assignmentRequirementDefinition("not-registered")).toBeUndefined();
  });

  it("binds availability to explicit providers and snapshots evidence", () => {
    const evidenceIds = ["probe.navigator-gpu", "probe.adapter"];
    const environment = createAssignmentRequirementResolutionEnvironment({
      environmentId: "browser.local",
      providers: [
        {
          requirementId: "webgpu",
          providerId: "navigator.gpu",
          mode: "browser",
          evidenceIds,
        },
        {
          requirementId: "native-cuda-external",
          providerId: "runner.production",
          mode: "external",
        },
      ],
    });
    evidenceIds.push("mutation.after-construction");

    expect(environment.schema).toBe(
      ASSIGNMENT_REQUIREMENT_ENVIRONMENT_SCHEMA,
    );
    expect(Object.isFrozen(environment)).toBe(true);
    expect(Object.isFrozen(environment.resolutions)).toBe(true);
    expect(environment.resolutions).toHaveLength(53);
    expect(environment.resolutions.find(
      (resolution) => resolution.requirementId === "webgpu",
    )).toMatchObject({
      environmentId: "browser.local",
      status: "available",
      provider: {
        providerId: "navigator.gpu",
        mode: "browser",
        evidenceIds: ["probe.adapter", "probe.navigator-gpu"],
      },
    });
    expect(environment.resolutions.find(
      (resolution) => resolution.requirementId === "pyodide",
    )).toMatchObject({
      environmentId: "browser.local",
      status: "unavailable",
    });
  });

  it("derives legacy capability evaluation only from available resolutions", () => {
    const parsed = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "execution",
          kind: "capability",
          options: {
            requires: ["pyodide"],
            any_of: [["webgpu"], ["native-cuda-external"]],
          },
        },
      ],
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;

    const definitionsOnly =
      createAssignmentRequirementResolutionEnvironment({
        environmentId: "browser.local",
      });
    expect(
      evaluateAssignmentRequirementResolutions(
        parsed.profile,
        definitionsOnly,
      ).ok,
    ).toBe(false);

    const resolved = createAssignmentRequirementResolutionEnvironment({
      environmentId: "browser.local",
      providers: [
        {
          requirementId: "pyodide",
          providerId: "worker.pyodide",
          mode: "browser",
        },
        {
          requirementId: "native-cuda-external",
          providerId: "runner.native",
          mode: "external",
        },
      ],
    });
    expect(
      assignmentCapabilityEnvironmentFromRequirementResolutions(resolved),
    ).toEqual({
      capabilities: ["native-cuda-external", "pyodide"],
      capabilityModes: {
        "native-cuda-external": "external",
        pyodide: "browser",
      },
    });
    expect(
      evaluateAssignmentRequirementResolutions(parsed.profile, resolved),
    ).toMatchObject({
      ok: true,
      gates: [{
        status: "external-only",
        selectedAnyOf: ["native-cuda-external"],
      }],
    });
  });

  it("rejects unknown, duplicate, and structurally forged resolutions", () => {
    expect(() => createAssignmentRequirementResolutionEnvironment({
      environmentId: "browser.local",
      providers: [{
        requirementId: "not-registered",
        providerId: "fixture",
        mode: "simulated",
      }],
    })).toThrow(/unknown requirement/u);
    expect(() => createAssignmentRequirementResolutionEnvironment({
      environmentId: "browser.local",
      providers: [
        {
          requirementId: "webgpu",
          providerId: "navigator.gpu",
          mode: "browser",
        },
        {
          requirementId: "webgpu",
          providerId: "fixture.webgpu",
          mode: "simulated",
        },
      ],
    })).toThrow(/more than one provider/u);

    const valid = createAssignmentRequirementResolutionEnvironment({
      environmentId: "browser.local",
    });
    const missing = {
      ...valid,
      resolutions: valid.resolutions.slice(1),
    } satisfies AssignmentRequirementResolutionEnvironment;
    expect(() =>
      assignmentCapabilityEnvironmentFromRequirementResolutions(missing)
    ).toThrow(/resolve every registered definition/u);

    const wrongEnvironment = {
      ...valid,
      resolutions: valid.resolutions.map((resolution, index) =>
        index === 0
          ? { ...resolution, environmentId: "browser.other" }
          : resolution
      ),
    } as AssignmentRequirementResolutionEnvironment;
    expect(() =>
      assignmentCapabilityEnvironmentFromRequirementResolutions(
        wrongEnvironment,
      )
    ).toThrow(/does not match its definition/u);
  });
});
