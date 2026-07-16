import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import {
  hashCanonicalJson,
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  unwrapPreparedCppCuteAotExecutionEnvironment,
} from "../../src/cpp_cute_aot_environment.js";
import {
  CPP_CUTE_AOT_SECCOMP_PROFILE_BYTE_LENGTH,
  CPP_CUTE_AOT_SECCOMP_PROFILE_SHA256,
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
    const seccompPath = fileURLToPath(new URL(
      "../../../../tools/cpp-cute-aot/seccomp.v1.json",
      import.meta.url,
    ));
    const seccompLockPath = fileURLToPath(new URL(
      "../../../../tools/cpp-cute-aot/seccomp.v1.lock.json",
      import.meta.url,
    ));
    const seccompLock = JSON.parse(await readFile(seccompLockPath, "utf8")) as unknown;
    expect(seccompLock).toEqual({
      schema: "browsergrad.compiler.cpp-cute.seccomp-provenance-lock",
      version: { major: 1, minor: 0 },
      upstream: {
        repository: "https://github.com/moby/profiles",
        revision: {
          algorithm: "git-sha1",
          value: "f9bc03ec19b2dc4c091449b08e88f85c0caa9f0b",
        },
        path: "seccomp/default.json",
      },
      resource: {
        path: "seccomp.v1.json",
        sha256: CPP_CUTE_AOT_SECCOMP_PROFILE_SHA256,
        byteLength: CPP_CUTE_AOT_SECCOMP_PROFILE_BYTE_LENGTH,
      },
    });
    const checkedSeccomp = await readFile(seccompPath);
    expect(checkedSeccomp.byteLength).toBe(CPP_CUTE_AOT_SECCOMP_PROFILE_BYTE_LENGTH);
    expect(await sha256Hex(checkedSeccomp)).toBe(CPP_CUTE_AOT_SECCOMP_PROFILE_SHA256);
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
    const environment = unwrapPreparedCppCuteAotExecutionEnvironment(
      fixture.executionEnvironment,
    ).manifest;
    expect(environment.body.runtime.seccomp.profileSha256).toBe(
      CPP_CUTE_AOT_SECCOMP_PROFILE_SHA256,
    );
    expect(CPP_CUTE_AOT_SANDBOX_POLICY_V1.privileges.seccomp).toEqual({
      request: "checked-private-snapshot",
      sourceSha256: CPP_CUTE_AOT_SECCOMP_PROFILE_SHA256,
      sourceByteLength: CPP_CUTE_AOT_SECCOMP_PROFILE_BYTE_LENGTH,
      effectiveProfile: "requires-external-run-evidence",
    });
    expect(await computeCppCuteAotExecutionPlanHash(
      fixture.job,
      fixture.executionEnvironment,
    )).toBe(
      fixture.receipt.invocation.executionPlanSha256,
    );
    expect(fixture.receipt.invocation.executionPlanSha256).toBe(
      "684616abea77112a4351bd0d64dbdd5cef0a908c60ac6c923a21d50e2944d7da",
    );
  });
});
