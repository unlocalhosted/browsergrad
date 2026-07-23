import { describe, expect, it } from "vitest";

import {
  LOWERING_DECISION_SCHEMA,
  createLoweringDecision,
  createSemanticBackendDefinition,
  createSemanticCapabilityDefinition,
} from "../../src/capability.js";

const CAPABILITY = createSemanticCapabilityDefinition({
  capabilityId: "browsergrad.layout.index-map",
  semanticVersion: "1.0.0",
  operationVersion: "browsergrad.layout@1",
  preservationLevels: [
    "portable-relegalized",
    "observable-equivalent",
    "observable-equivalent",
  ],
  owner: "@unlocalhosted/browsergrad-semantic-core",
  evidenceIds: ["layout.trace", "layout.verify"],
});

const BACKEND = createSemanticBackendDefinition({
  backendId: "browsergrad.kernels.webgpu",
  semanticVersion: "0.1.0",
  owner: "@unlocalhosted/browsergrad-kernels",
  executionTiers: ["webgpu-enhanced", "webgpu-core"],
  evidenceIds: ["webgpu.real-device"],
});

describe("program-scoped lowering decisions", () => {
  it("creates immutable sorted capability and backend definitions", () => {
    expect(CAPABILITY).toMatchObject({
      schema: "browsergrad.semantic-capability-definition",
      schemaVersion: 1,
      capabilityId: "browsergrad.layout.index-map",
      preservationLevels: [
        "observable-equivalent",
        "portable-relegalized",
      ],
    });
    expect(BACKEND).toMatchObject({
      schema: "browsergrad.semantic-backend-definition",
      schemaVersion: 1,
      backendId: "browsergrad.kernels.webgpu",
      executionTiers: ["webgpu-core", "webgpu-enhanced"],
    });
    expect(Object.isFrozen(CAPABILITY)).toBe(true);
    expect(Object.isFrozen(CAPABILITY.preservationLevels)).toBe(true);
    expect(Object.isFrozen(BACKEND.executionTiers)).toBe(true);
  });

  it("binds a conditional decision to one program, capability, and backend", () => {
    const decision = createLoweringDecision(CAPABILITY, BACKEND, {
      subject: {
        kind: "program",
        programId: "browsergrad.program.transpose",
      },
      executionTier: "webgpu-core",
      state: "conditional",
      preservationLevel: "portable-relegalized",
      requiredFeatures: ["webgpu", "shader-f16", "webgpu"],
      requiredLimits: {
        maxStorageBufferBindingSize: 4096,
        maxComputeInvocationsPerWorkgroup: 64,
      },
      runtimeGuardIds: ["browsergrad.guard.device-limits"],
      legalizationIds: ["browsergrad.legalize.layout-index-map"],
      numericalPolicyId: "browsergrad.numerical.exact-bits",
    });

    expect(decision).toEqual({
      schema: LOWERING_DECISION_SCHEMA,
      schemaVersion: 1,
      subject: {
        kind: "program",
        programId: "browsergrad.program.transpose",
      },
      capabilityId: "browsergrad.layout.index-map",
      capabilityVersion: "1.0.0",
      backendId: "browsergrad.kernels.webgpu",
      backendVersion: "0.1.0",
      executionTier: "webgpu-core",
      state: "conditional",
      preservationLevel: "portable-relegalized",
      requiredFeatures: ["shader-f16", "webgpu"],
      requiredLimits: {
        maxComputeInvocationsPerWorkgroup: 64,
        maxStorageBufferBindingSize: 4096,
      },
      runtimeGuardIds: ["browsergrad.guard.device-limits"],
      legalizationIds: ["browsergrad.legalize.layout-index-map"],
      numericalPolicyId: "browsergrad.numerical.exact-bits",
    });
    expect(Object.isFrozen(decision)).toBe(true);
    expect(Object.isFrozen(decision.subject)).toBe(true);
    expect(Object.isFrozen(decision.requiredLimits)).toBe(true);
  });

  it("represents a backend-tier refusal without a preservation claim", () => {
    expect(createLoweringDecision(CAPABILITY, BACKEND, {
      subject: { kind: "artifact", artifactHash: "a".repeat(64) },
      executionTier: "webgpu-core",
      state: "unsupported",
      reasonCode: "BG-LOWERING-NO-PORTABLE-PROFILE",
    })).toMatchObject({
      subject: { kind: "artifact", artifactHash: "a".repeat(64) },
      state: "unsupported",
      reasonCode: "BG-LOWERING-NO-PORTABLE-PROFILE",
      requiredFeatures: [],
      requiredLimits: {},
      runtimeGuardIds: [],
      legalizationIds: [],
    });
  });

  it("rejects static claims, unbound conditions, and incompatible tiers", () => {
    expect(() => createLoweringDecision(CAPABILITY, BACKEND, {
      subject: {
        kind: "program",
        programId: "browsergrad.program.transpose",
      },
      executionTier: "semantic-reference",
      state: "supported",
      preservationLevel: "observable-equivalent",
    })).toThrow(/execution tier/u);
    expect(() => createLoweringDecision(CAPABILITY, BACKEND, {
      subject: {
        kind: "program",
        programId: "browsergrad.program.transpose",
      },
      executionTier: "webgpu-core",
      state: "conditional",
      preservationLevel: "observable-equivalent",
    })).toThrow(/requires a feature, limit, or runtime guard/u);
    expect(() => createLoweringDecision(CAPABILITY, BACKEND, {
      subject: {
        kind: "program",
        programId: "browsergrad.program.transpose",
      },
      executionTier: "webgpu-core",
      state: "unknown",
      preservationLevel: "observable-equivalent",
      reasonCode: "BG-LOWERING-NOT-EVALUATED",
    })).toThrow(/cannot claim preservation/u);
    expect(() => createLoweringDecision(CAPABILITY, BACKEND, {
      subject: { kind: "artifact", artifactHash: "not-a-hash" },
      executionTier: "webgpu-core",
      state: "unsupported",
      reasonCode: "BG-LOWERING-NO-PROFILE",
    })).toThrow(/artifactHash/u);
  });
});
