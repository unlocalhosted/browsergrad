import { describe, expect, it } from "vitest";

import {
  CppCuteBrowserFullDistributionMaterializationError,
  parseCppCuteBrowserFullDistributionMaterializationArguments,
  requireCppCuteBrowserDeterministicDistributionAuthority,
  requireCppCuteBrowserFullDistributionMaterializationAuthority,
} from "./cpp_cute_browser_full_distribution_materialization.mjs";

describe("full browser C++/CuTe distribution materialization", () => {
  it("parses the closed deterministic and finalization operations", () => {
    expect(parseCppCuteBrowserFullDistributionMaterializationArguments([
      "--operation=materialize-deterministic",
      "--output-root=/private/tmp/browsergrad-distribution",
      "--producer-policy=/private/tmp/producer-policy.json",
      "--profile-output=/private/tmp/browsergrad-profile.json",
      "--wasm=/private/tmp/clang-extractor.wasm",
    ])).toEqual({
      operation: "materialize-deterministic",
      "output-root": "/private/tmp/browsergrad-distribution",
      "producer-policy": "/private/tmp/producer-policy.json",
      "profile-output": "/private/tmp/browsergrad-profile.json",
      wasm: "/private/tmp/clang-extractor.wasm",
    });
    expect(parseCppCuteBrowserFullDistributionMaterializationArguments([
      "--operation=finalize",
      "--output-root=/private/tmp/browsergrad-distribution",
      "--producer-policy=/private/tmp/producer-policy.json",
      "--profile=/private/tmp/browsergrad-profile.json",
      "--trust-store=/private/tmp/producer-trust-store.json",
      "--envelope=/private/tmp/build-provenance.dsse.json",
    ])).toEqual({
      operation: "finalize",
      "output-root": "/private/tmp/browsergrad-distribution",
      "producer-policy": "/private/tmp/producer-policy.json",
      profile: "/private/tmp/browsergrad-profile.json",
      "trust-store": "/private/tmp/producer-trust-store.json",
      envelope: "/private/tmp/build-provenance.dsse.json",
    });
  });

  it("rejects operation widening, duplicates, and noncanonical paths", () => {
    expect(() =>
      parseCppCuteBrowserFullDistributionMaterializationArguments([
        "--operation=materialize-deterministic",
        "--output-root=/private/tmp/one",
        "--producer-policy=/private/tmp/policy.json",
        "--profile-output=/private/tmp/profile.json",
        "--wasm=/private/tmp/wasm",
        "--extra=/private/tmp/extra",
      ]),
    ).toThrow(CppCuteBrowserFullDistributionMaterializationError);
    expect(() =>
      parseCppCuteBrowserFullDistributionMaterializationArguments([
        "--operation=finalize",
        "--operation=finalize",
      ]),
    ).toThrow("duplicate --operation");
    expect(() =>
      parseCppCuteBrowserFullDistributionMaterializationArguments([
        "--operation=materialize-deterministic",
        "--output-root=/private/tmp/../tmp/one",
        "--producer-policy=/private/tmp/policy.json",
        "--profile-output=/private/tmp/profile.json",
        "--wasm=/private/tmp/wasm",
      ]),
    ).toThrow("canonical absolute");
  });

  it("rejects forged deterministic and full-distribution authorities", () => {
    expect(() =>
      requireCppCuteBrowserDeterministicDistributionAuthority(
        Object.freeze({}) as never,
      ),
    ).toThrow("materializer-issued deterministic");
    expect(() =>
      requireCppCuteBrowserFullDistributionMaterializationAuthority(
        Object.freeze({}) as never,
      ),
    ).toThrow("materializer-issued full-distribution");
  });
});
