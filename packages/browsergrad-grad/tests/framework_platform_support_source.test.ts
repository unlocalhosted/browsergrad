import { describe, expect, it } from "vitest";

import {
  GRAD_FRAMEWORK_CONTRACT_SCHEMA,
  GRAD_FRAMEWORK_ID,
  frameworkPlatformSupportSource,
} from "../src/index";

describe("generated Grad framework platform support", () => {
  it("projects every frozen verified eager behavior without device substitution", () => {
    const source = frameworkPlatformSupportSource();

    expect(source).toMatchObject({
      frameworkId: GRAD_FRAMEWORK_ID,
      contractSchema: GRAD_FRAMEWORK_CONTRACT_SCHEMA,
      contractVersion: { major: 2, minor: 0 },
    });
    expect(source.operations).toHaveLength(22);
    expect(source.operations.every(
      ({ semanticState }) => semanticState === "verified-eager-contract",
    )).toBe(true);
    expect(source.operations.every(
      ({ decisions }) =>
        decisions.symbolicVjp === "not-applicable-eager"
        && decisions.functionalGrad === "not-applicable-eager"
        && decisions.vmap === "not-applicable-eager"
        && decisions.onnxExport === "not-applicable-eager"
        && decisions.tensorPlan === "not-applicable-eager-framework"
        && decisions.webgpu === "refused-numpy-reference-only",
    )).toBe(true);
  });

  it("retains explicit refusal, view, dtype, and autograd decisions", () => {
    const byId = new Map(
      frameworkPlatformSupportSource().operations.map((operation) => [
        operation.operationId,
        operation,
      ]),
    );

    expect(
      byId.get("browsergrad.grad.dtype.bf16-rejected.v1"),
    ).toMatchObject({
      decisions: {
        cpu: "refused-reject-unsupported-bfloat16-before-allocation",
        closureAutograd: "not-applicable",
        residency: "not-applicable-refused-before-execution",
        materialization: "not-applicable",
      },
    });
    expect(byId.get("browsergrad.grad.view.expand.v1")).toMatchObject({
      dtypeContract: "preserves",
      decisions: {
        cpu: "supported-pytorch-shaped-compatibility",
        closureAutograd: "supported-eager-graph-edge",
        residency: "pyodide-wasm-linear-memory",
        materialization: "none-view-or-identity",
      },
    });
    expect(
      byId.get("browsergrad.grad.conversion.to-cross-floating.v1"),
    ).toMatchObject({
      decisions: {
        closureAutograd: "supported-eager-graph-edge",
        materialization: "always-owning",
      },
    });
  });

  it("returns detached deterministic sources", () => {
    const first = frameworkPlatformSupportSource();
    const second = frameworkPlatformSupportSource();
    (first.operations[0]!.decisions as Record<string, string>).cpu = "forged";
    (first.operations as unknown[]).push({ forged: true });

    expect(first.operations).toHaveLength(23);
    expect(second.operations).toHaveLength(22);
    expect(second.operations[0]!.decisions.cpu).not.toBe("forged");
    expect(second.operations.map(({ operationId }) => operationId)).toEqual(
      second.operations.map(({ operationId }) => operationId).sort(),
    );
  });
});
