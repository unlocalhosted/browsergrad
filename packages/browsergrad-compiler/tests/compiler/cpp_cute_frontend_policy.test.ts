import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_FRONTEND_TEMPORAL_MACRO_NAMES,
  CPP_CUTE_FRONTEND_TEMPORAL_MACRO_POLICY_ID,
  CPP_CUTE_FRONTEND_WARNING_BASELINE,
  CPP_CUTE_FRONTEND_WARNING_POLICY_MAPPINGS,
  CPP_CUTE_FRONTEND_WARNING_POLICY_REGISTRY_ID,
  cppCuteFrontendWarningArguments,
  type CppCuteFrontendWarningDisposition,
} from "../../src/cpp_cute_frontend_compiler_policy.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  cloneCppCuteProfileInput,
  createCppCuteProfileInput,
} from "./support/cpp_cute_frontend_fixtures.js";

describe("C++/CuTe closed compiler policy", () => {
  it("binds temporal and diagnostic policy selections into the compilation contract", async () => {
    const prepared = await prepareCppCuteFrontendProfile(createCppCuteProfileInput());
    const contract = unwrapPreparedCppCuteFrontendProfile(prepared).compilationContract;

    expect(contract.language.preprocessing).toEqual({
      temporalMacros: {
        policyId: CPP_CUTE_FRONTEND_TEMPORAL_MACRO_POLICY_ID,
        mode: "reject",
      },
    });
    expect(contract.language.diagnostics).toEqual({
      warningRegistryId: CPP_CUTE_FRONTEND_WARNING_POLICY_REGISTRY_ID,
      baseline: CPP_CUTE_FRONTEND_WARNING_BASELINE,
    });
    expect(contract.compatibility.unsupportedSourceFeatures).toContain(
      "cxx:temporal-macros@1",
    );
  });

  it("rejects missing or unknown policy authority", async () => {
    const missing = cloneCppCuteProfileInput();
    delete (missing["language"] as Record<string, unknown>)["preprocessing"];
    await expect(prepareCppCuteFrontendProfile(missing)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      path: "$.language",
    });

    const temporal = cloneCppCuteProfileInput();
    const temporalLanguage = temporal["language"] as Record<string, unknown>;
    const preprocessing = temporalLanguage["preprocessing"] as Record<string, unknown>;
    const temporalMacros = preprocessing["temporalMacros"] as Record<string, unknown>;
    temporalMacros["policyId"] = "browsergrad.compiler.cpp-cute.temporal-macros.pin@1";
    await expect(prepareCppCuteFrontendProfile(temporal)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      path: "$.language.preprocessing.temporalMacros.policyId",
    });

    const diagnostics = cloneCppCuteProfileInput();
    const diagnosticLanguage = diagnostics["language"] as Record<string, unknown>;
    const diagnosticPolicy = diagnosticLanguage["diagnostics"] as Record<string, unknown>;
    diagnosticPolicy["warningRegistryId"] = "browsergrad.compiler.cpp-cute.open-warnings@1";
    await expect(prepareCppCuteFrontendProfile(diagnostics)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      path: "$.language.diagnostics.warningRegistryId",
    });

    const baseline = cloneCppCuteProfileInput();
    const baselineLanguage = baseline["language"] as Record<string, unknown>;
    const baselinePolicy = baselineLanguage["diagnostics"] as Record<string, unknown>;
    baselinePolicy["baseline"] = "suppress-all";
    await expect(prepareCppCuteFrontendProfile(baseline)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      path: "$.language.diagnostics.baseline",
    });
  });

  it("rejects temporal macro overrides and contradictory compatibility claims", async () => {
    for (const macroName of CPP_CUTE_FRONTEND_TEMPORAL_MACRO_NAMES) {
      for (const option of [
        { kind: "define", name: macroName, value: "forged" },
        { kind: "undefine", name: macroName },
      ]) {
        const input = cloneCppCuteProfileInput();
        (input["language"] as Record<string, unknown>)["options"] = [option];
        await expect(prepareCppCuteFrontendProfile(input)).rejects.toMatchObject({
          code: "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
          path: "$.language.options[0].name",
        });
      }
    }

    for (const name of ["_PRIVATE", "name__implementation", "defined"]) {
      const input = cloneCppCuteProfileInput();
      (input["language"] as Record<string, unknown>)["options"] = [{
        kind: "define",
        name,
        value: "1",
      }];
      await expect(prepareCppCuteFrontendProfile(input)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
        path: "$.language.options[0].name",
      });
    }

    const missingUnsupported = cloneCppCuteProfileInput();
    (missingUnsupported["compatibility"] as Record<string, unknown>)["unsupportedSourceFeatures"] = [];
    await expect(prepareCppCuteFrontendProfile(missingUnsupported)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      path: "$.compatibility.unsupportedSourceFeatures",
    });

    const contradictory = cloneCppCuteProfileInput();
    (contradictory["compatibility"] as Record<string, unknown>)["supportedSourceFeatures"] = [
      "cuda:language@1",
      "cute:layout-algebra@1",
      "cxx:templates@1",
      "cxx:temporal-macros@1",
    ];
    await expect(prepareCppCuteFrontendProfile(contradictory)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      path: "$.compatibility.supportedSourceFeatures",
    });
  });

  it("closes warning IDs and materializes exact non-shell argv elements", async () => {
    const dispositions: readonly CppCuteFrontendWarningDisposition[] = [
      "ignore",
      "warn",
      "error",
    ];
    for (const mapping of CPP_CUTE_FRONTEND_WARNING_POLICY_MAPPINGS) {
      for (const disposition of dispositions) {
        const input = cloneCppCuteProfileInput();
        (input["language"] as Record<string, unknown>)["options"] = [{
          kind: "warning-policy",
          id: mapping.policyId,
          disposition,
        }];
        const prepared = await prepareCppCuteFrontendProfile(input);
        expect(unwrapPreparedCppCuteFrontendProfile(prepared).profile.language.options).toEqual([{
          kind: "warning-policy",
          id: mapping.policyId,
          disposition,
        }]);

        const group = mapping.clangDiagnosticGroup;
        const expected = disposition === "ignore"
          ? [`-Wno-${group}`]
          : disposition === "warn"
            ? [`-W${group}`, `-Wno-error=${group}`]
            : [`-W${group}`, `-Werror=${group}`];
        expect(cppCuteFrontendWarningArguments(mapping.policyId, disposition)).toEqual(expected);
      }
    }

    const unknown = cloneCppCuteProfileInput();
    (unknown["language"] as Record<string, unknown>)["options"] = [{
      kind: "warning-policy",
      id: "clang.everything",
      disposition: "error",
    }];
    await expect(prepareCppCuteFrontendProfile(unknown)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROFILE-INVALID",
      path: "$.language.options[0].id",
    });
  });

  it("preserves warning order and hashes policy semantics, not deployment", async () => {
    const ordered = cloneCppCuteProfileInput();
    (ordered["language"] as Record<string, unknown>)["options"] = [
      { kind: "warning-policy", id: "clang.unused-variable", disposition: "error" },
      { kind: "warning-policy", id: "clang.sign-compare", disposition: "warn" },
    ];
    const first = await prepareCppCuteFrontendProfile(ordered);

    const reversed = structuredClone(ordered) as typeof ordered;
    ((reversed["language"] as Record<string, unknown>)["options"] as unknown[]).reverse();
    const second = await prepareCppCuteFrontendProfile(reversed);
    expect(second.compilationContractHash).not.toBe(first.compilationContractHash);

    const deploymentOnly = structuredClone(ordered) as typeof ordered;
    const deployment = deploymentOnly["deployment"] as Record<string, unknown>;
    (deployment["runner"] as Record<string, unknown>)["binarySha256"] = "8".repeat(64);
    const third = await prepareCppCuteFrontendProfile(deploymentOnly);
    expect(third.compilationContractHash).toBe(first.compilationContractHash);
  });
});
