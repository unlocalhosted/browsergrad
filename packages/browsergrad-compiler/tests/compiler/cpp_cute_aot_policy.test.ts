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

    const fixture = await createCppCuteProvenanceFixture();
    expect(await computeCppCuteAotExecutionPlanHash(fixture.job)).toBe(
      fixture.receipt.invocation.executionPlanSha256,
    );
    expect(fixture.receipt.invocation.executionPlanSha256).toBe(
      "18d1393909e147e16f4b31777da36c674b5651b5085600cb1546cb869781b1f0",
    );
  });
});
