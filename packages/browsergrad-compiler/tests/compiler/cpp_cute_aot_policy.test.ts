import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { hashCanonicalJson } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_AOT_SANDBOX_POLICY_SHA256,
  CPP_CUTE_AOT_SANDBOX_POLICY_V1,
  computeCppCuteAotExecutionPlanHash,
  verifyCppCuteAotSandboxPolicyIdentity,
} from "../../src/cpp_cute_aot_policy.js";
import { createCppCuteProvenanceFixture } from "./support/cpp_cute_provenance_fixtures.js";

describe("C++/CuTe AOT sandbox policy", () => {
  it("keeps one checked-in canonical policy and deterministic execution plan", async () => {
    const path = fileURLToPath(new URL(
      "../../../../tools/cpp-cute-aot/sandbox-policy.v1.json",
      import.meta.url,
    ));
    const checkedIn = JSON.parse(await readFile(path, "utf8")) as unknown;
    expect(checkedIn).toEqual(CPP_CUTE_AOT_SANDBOX_POLICY_V1);
    expect(await hashCanonicalJson(CPP_CUTE_AOT_SANDBOX_POLICY_V1)).toBe(
      CPP_CUTE_AOT_SANDBOX_POLICY_SHA256,
    );
    await expect(verifyCppCuteAotSandboxPolicyIdentity()).resolves.toBeUndefined();
    expect(Object.isFrozen(CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.arguments)).toBe(true);
    expect(Object.isFrozen(CPP_CUTE_AOT_SANDBOX_POLICY_V1.decoding.artifact)).toBe(true);
    expect(() => {
      (CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.arguments as unknown as string[]).push("--escape");
    }).toThrowError(TypeError);
    expect(() => {
      (CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.user as { uid: number }).uid = 0;
    }).toThrowError(TypeError);

    const fixture = await createCppCuteProvenanceFixture();
    expect(await computeCppCuteAotExecutionPlanHash(fixture.job)).toBe(
      fixture.receipt.invocation.executionPlanSha256,
    );
    expect(fixture.receipt.invocation.executionPlanSha256).toBe(
      "8b8b3809df6e20dde842a3488553b2f84093426aca0c95f7b1bcfb53b04091c7",
    );
  });
});
