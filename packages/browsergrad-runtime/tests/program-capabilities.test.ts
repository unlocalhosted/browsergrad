import { describe, expect, it } from "vitest";

import {
  PROGRAM_CAPABILITY_SUPPORT_VIEW_SCHEMA,
  createProgramCapabilitySupportView,
  semanticBackendDefinitions,
  semanticCapabilityDefinitions,
} from "../src/index";

describe("program capability support views", () => {
  it("exposes immutable generated definitions without inferring support", () => {
    expect(semanticCapabilityDefinitions()).toHaveLength(1);
    expect(semanticBackendDefinitions()).toHaveLength(3);
    expect(Object.isFrozen(semanticCapabilityDefinitions())).toBe(true);
    expect(semanticCapabilityDefinitions()[0]).toMatchObject({
      capabilityId: "browsergrad.layout.index-map",
      semanticVersion: "1.0.0",
      operationVersion: "browsergrad.layout@1",
    });
    expect(
      semanticCapabilityDefinitions().some(
        (definition) => "state" in definition,
      ),
    ).toBe(false);
  });

  it("builds a deterministic view only from program-scoped decisions", () => {
    const requiredFeatures = ["webgpu"];
    const view = createProgramCapabilitySupportView({
      viewId: "browsergrad.support.transpose",
      subject: {
        kind: "program",
        programId: "browsergrad.program.transpose",
      },
      decisions: [
        {
          capabilityId: "browsergrad.layout.index-map",
          backendId: "browsergrad.kernels.webgpu",
          executionTier: "webgpu-core",
          state: "conditional",
          preservationLevel: "observable-equivalent",
          requiredFeatures,
          runtimeGuardIds: ["browsergrad.guard.device-features"],
        },
        {
          capabilityId: "browsergrad.layout.index-map",
          backendId: "browsergrad.compiler.semantic-reference",
          executionTier: "semantic-reference",
          state: "supported",
          preservationLevel: "observable-equivalent",
        },
      ],
    });
    requiredFeatures.push("mutation.after-construction");

    expect(view.schema).toBe(PROGRAM_CAPABILITY_SUPPORT_VIEW_SCHEMA);
    expect(view.subject).toEqual({
      kind: "program",
      programId: "browsergrad.program.transpose",
    });
    expect(view.capabilities.map(({ capabilityId }) => capabilityId)).toEqual([
      "browsergrad.layout.index-map",
    ]);
    expect(view.backends.map(({ backendId }) => backendId)).toEqual([
      "browsergrad.compiler.semantic-reference",
      "browsergrad.kernels.webgpu",
    ]);
    expect(view.decisions.map((decision) => [
      decision.backendId,
      decision.state,
      decision.requiredFeatures,
    ])).toEqual([
      ["browsergrad.compiler.semantic-reference", "supported", []],
      ["browsergrad.kernels.webgpu", "conditional", ["webgpu"]],
    ]);
    expect(Object.isFrozen(view)).toBe(true);
    expect(Object.isFrozen(view.decisions)).toBe(true);
  });

  it("keeps explicit refusals distinct from missing definitions", () => {
    const view = createProgramCapabilitySupportView({
      viewId: "browsergrad.support.refusal",
      subject: { kind: "artifact", artifactHash: "b".repeat(64) },
      decisions: [{
        capabilityId: "browsergrad.layout.index-map",
        backendId: "browsergrad.kernels.webgpu",
        executionTier: "webgpu-core",
        state: "unsupported",
        reasonCode: "BG-LOWERING-NO-PROFILE",
      }],
    });
    expect(view.decisions[0]).toMatchObject({
      state: "unsupported",
      reasonCode: "BG-LOWERING-NO-PROFILE",
    });
    expect(view.decisions[0]).not.toHaveProperty("preservationLevel");
  });

  it("rejects empty, unknown, duplicate, and unbound decision sets", () => {
    expect(() => createProgramCapabilitySupportView({
      viewId: "browsergrad.support.empty",
      subject: {
        kind: "program",
        programId: "browsergrad.program.empty",
      },
      decisions: [],
    })).toThrow(/at least one lowering decision/u);
    expect(() => createProgramCapabilitySupportView({
      viewId: "browsergrad.support.unknown",
      subject: {
        kind: "program",
        programId: "browsergrad.program.unknown",
      },
      decisions: [{
        capabilityId: "browsergrad.capability.missing",
        backendId: "browsergrad.kernels.webgpu",
        executionTier: "webgpu-core",
        state: "unsupported",
        reasonCode: "BG-LOWERING-NO-DEFINITION",
      }],
    })).toThrow(/unknown capability/u);
    expect(() => createProgramCapabilitySupportView({
      viewId: "browsergrad.support.duplicate",
      subject: {
        kind: "program",
        programId: "browsergrad.program.duplicate",
      },
      decisions: [
        {
          capabilityId: "browsergrad.layout.index-map",
          backendId: "browsergrad.kernels.webgpu",
          executionTier: "webgpu-core",
          state: "unsupported",
          reasonCode: "BG-LOWERING-NO-PROFILE",
        },
        {
          capabilityId: "browsergrad.layout.index-map",
          backendId: "browsergrad.kernels.webgpu",
          executionTier: "webgpu-enhanced",
          state: "unknown",
          reasonCode: "BG-LOWERING-NOT-EVALUATED",
        },
      ],
    })).toThrow(/duplicate capability\/backend decision/u);
  });
});
