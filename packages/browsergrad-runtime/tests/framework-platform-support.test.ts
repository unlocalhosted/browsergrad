import { describe, expect, it } from "vitest";

import {
  FRAMEWORK_PLATFORM_SUPPORT_VIEW_SCHEMA,
  createAssignmentRequirementResolutionEnvironment,
  createFrameworkPlatformSupportView,
  type FrameworkPlatformSupportSourceInput,
} from "../src/index";

const JIT_SOURCE: FrameworkPlatformSupportSourceInput = {
  frameworkId: "browsergrad.jit",
  frameworkVersion: "0.9.0",
  contractSchema: "browsergrad.jit.framework-operation-contracts",
  contractVersion: { major: 1, minor: 0 },
  operations: [{
    operationId: "browsergrad.jit.framework.tensor.expand.v1",
    publicSurface: "Tensor.expand",
    implementationId: "BROADCAST_TO",
    semanticState: "typed",
    shapeContract: "broadcast-compatible-shape",
    dtypeContract: "preserve-input",
    decisions: {
      cpu: "supported-numpy-owning-broadcast",
      closureAutograd: "supported-unbroadcast",
      symbolicVjp: "supported-unbroadcast",
      functionalGrad: "supported-via-symbolic-vjp",
      vmap: "supported-leading-batch-axis",
      onnxExport: "supported-opset17-expand",
      tensorPlan: "supported-materializing-or-resident",
      webgpu: "eligible-positive-stride-profile",
      residency: "conditional-resident",
      materialization: "cpu-owning-array",
    },
    legacyOperationId: "jit.custom.expand.v0",
  }],
};

function createView(
  frameworks: readonly FrameworkPlatformSupportSourceInput[] = [JIT_SOURCE],
) {
  return createFrameworkPlatformSupportView({
    viewId: "browsergrad.support.platform.transpose",
    requirements: createAssignmentRequirementResolutionEnvironment({
      environmentId: "browser.local",
      providers: [{
        requirementId: "webgpu",
        providerId: "navigator.gpu",
        mode: "browser",
        evidenceIds: ["browser.features"],
      }],
    }),
    program: {
      viewId: "browsergrad.support.program.transpose",
      subject: {
        kind: "program",
        programId: "browsergrad.program.transpose",
      },
      decisions: [{
        capabilityId: "browsergrad.layout.index-map",
        backendId: "browsergrad.kernels.webgpu",
        executionTier: "webgpu-core",
        state: "conditional",
        preservationLevel: "observable-equivalent",
        requiredFeatures: ["webgpu"],
        runtimeGuardIds: ["browsergrad.guard.device-features"],
      }],
    },
    frameworks,
  });
}

describe("framework platform support views", () => {
  it("keeps requirements, program decisions, and framework contracts distinct", () => {
    const view = createView();

    expect(view.schema).toBe(FRAMEWORK_PLATFORM_SUPPORT_VIEW_SCHEMA);
    expect(view.environmentId).toBe("browser.local");
    expect(view.subject).toEqual({
      kind: "program",
      programId: "browsergrad.program.transpose",
    });
    expect(
      view.requirements.resolutions.find(
        ({ requirementId }) => requirementId === "webgpu",
      ),
    ).toMatchObject({
      status: "available",
      provider: { providerId: "navigator.gpu", mode: "browser" },
    });
    expect(view.programSupport.decisions[0]).toMatchObject({
      state: "conditional",
      preservationLevel: "observable-equivalent",
    });
    expect(view.frameworks[0]!.operations[0]!.decisions.webgpu).toBe(
      "eligible-positive-stride-profile",
    );
    expect(view.frameworks[0]!.operations[0]).not.toHaveProperty("supported");
    expect(Object.isFrozen(view)).toBe(true);
    expect(Object.isFrozen(view.requirements.resolutions)).toBe(true);
    expect(Object.isFrozen(view.programSupport.decisions)).toBe(true);
    expect(Object.isFrozen(view.frameworks[0]!.operations)).toBe(true);
  });

  it("normalizes detached input and deterministically orders providers", () => {
    const decisions = { ...JIT_SOURCE.operations[0]!.decisions };
    const other: FrameworkPlatformSupportSourceInput = {
      ...JIT_SOURCE,
      frameworkId: "browsergrad.grad",
      frameworkVersion: "0.5.2",
      contractSchema: "browsergrad.grad.compatibility-contracts",
      operations: [{
        ...JIT_SOURCE.operations[0]!,
        operationId: "browsergrad.grad.tensor.expand.v1",
        decisions,
      }],
    };
    const view = createView([JIT_SOURCE, other]);
    decisions.cpu = "mutation-after-construction";

    expect(view.frameworks.map(({ frameworkId }) => frameworkId)).toEqual([
      "browsergrad.grad",
      "browsergrad.jit",
    ]);
    expect(view.frameworks[0]!.operations[0]!.decisions.cpu).not.toBe(
      "mutation-after-construction",
    );
  });

  it("rejects empty sources, duplicate identities, and open decision maps", () => {
    expect(() => createView([])).toThrow(/1\.\.16 framework sources/u);
    expect(() => createView([JIT_SOURCE, JIT_SOURCE])).toThrow(
      /duplicate framework/u,
    );
    expect(() => createView([{
      ...JIT_SOURCE,
      operations: [
        JIT_SOURCE.operations[0]!,
        JIT_SOURCE.operations[0]!,
      ],
    }])).toThrow(/duplicate operation/u);
    expect(() => createView([{
      ...JIT_SOURCE,
      operations: [{
        ...JIT_SOURCE.operations[0]!,
        decisions: {
          ...JIT_SOURCE.operations[0]!.decisions,
          supported: "true",
        } as typeof JIT_SOURCE.operations[0]["decisions"],
      }],
    }])).toThrow(/fields are not registered/u);
  });
});
