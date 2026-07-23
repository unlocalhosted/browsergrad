import { describe, expect, it } from "vitest";

import {
  JIT_FRAMEWORK_ID,
  JIT_FRAMEWORK_OPERATION_SUPPORT_SCHEMA,
  frameworkOperationSupport,
  frameworkPlatformSupportSource,
} from "../src/index";
import { FRAMEWORK_OPERATION_CONTRACTS_JSON } from "../src/python/framework-operation-contracts.v1.generated";

describe("JavaScript framework-operation support source", () => {
  it("projects the exact executable registry without inferring availability", () => {
    const support = frameworkOperationSupport();
    const source = frameworkPlatformSupportSource();
    const registry = JSON.parse(FRAMEWORK_OPERATION_CONTRACTS_JSON) as unknown;

    expect(support).toEqual(registry);
    expect(support).toMatchObject({
      schema: JIT_FRAMEWORK_OPERATION_SUPPORT_SCHEMA,
      version: { major: 1, minor: 0 },
    });
    expect(support.operations).toHaveLength(36);
    expect(source).toMatchObject({
      frameworkId: JIT_FRAMEWORK_ID,
      contractSchema: JIT_FRAMEWORK_OPERATION_SUPPORT_SCHEMA,
      contractVersion: { major: 1, minor: 0 },
    });
    expect(source.operations).toHaveLength(36);
    expect(source.operations[0]).toHaveProperty("operationId");
    expect(source.operations[0]).toHaveProperty("decisions.cpu");
    expect(source.operations[0]).not.toHaveProperty("supported");
    expect(source.operations[0]).not.toHaveProperty("available");
  });

  it("returns detached tables and platform projections", () => {
    const first = frameworkOperationSupport();
    const second = frameworkOperationSupport();
    const firstSource = frameworkPlatformSupportSource();
    const secondSource = frameworkPlatformSupportSource();

    (first.operations[0]!.decisions as Record<string, string>).cpu = "forged";
    (first.operations as unknown[]).push({ forged: true });
    (firstSource.operations[0]!.decisions as Record<string, string>).webgpu =
      "forged";

    expect(first.operations).toHaveLength(37);
    expect(second.operations).toHaveLength(36);
    expect(second.operations[0]!.decisions.cpu).not.toBe("forged");
    expect(secondSource.operations[0]!.decisions.webgpu).not.toBe("forged");
  });

  it("keeps the platform projection deterministically ordered", () => {
    const source = frameworkPlatformSupportSource();
    const ids = source.operations.map(({ operationId }) => operationId);
    expect(ids).toEqual([...ids].sort());
    expect(new Set(ids).size).toBe(ids.length);
  });
});
