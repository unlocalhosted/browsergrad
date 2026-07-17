import {
  deepFreezeJson,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

/**
 * Closed semantic-policy contract consumed by every Clang 22.1.8 producer.
 *
 * This resource defines policy selection and policy-to-argv lowering. It does
 * not prove that a native or Wasm producer loaded the resource, installed the
 * required preprocessor callback, or invoked Clang with the recorded argv.
 */
const CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_VALUE = {
  schema: "browsergrad.compiler.cpp-cute.semantic-adapter-manifest",
  version: { major: 1, minor: 0 },
  manifestId: "bg.cpp.semantic-adapter.sha256.77d2e58c18d0df8e8a8aef7fa5742f8e9ae82912e692efab5507cf14112bceb0",
  body: {
    semanticAdapterId: "browsergrad.compiler.cpp-cute.clang-semantic-adapter@1",
    clang: {
      compilerId: "clang",
      version: "22.1.8",
    },
    temporalMacros: {
      policyId: "browsergrad.compiler.cpp-cute.temporal-macros.reject@1",
      mode: "reject",
      macroNames: ["__DATE__", "__TIMESTAMP__", "__TIME__"],
      consultation: "forbidden",
      mutation: "forbidden",
      enforcement: "preprocessor-callback-before-expansion",
      diagnosticCodes: {
        consultation: "browsergrad.cpp-cute:temporal-macro-forbidden",
        mutation: "browsergrad.cpp-cute:temporal-macro-mutation-forbidden",
      },
      defenseInDepthArgv: [
        "-Werror=builtin-macro-redefined",
        "-Werror=date-time",
        "-Werror=macro-redefined",
      ],
    },
    warningPolicyRegistry: {
      registryId: "browsergrad.compiler.cpp-cute.clang-warning-registry@1",
      compilerBaseline: "compiler-default",
      unknownPolicy: "reject",
      reservedClangDiagnosticGroups: [
        "builtin-macro-redefined",
        "date-time",
        "macro-redefined",
      ],
      entries: [
        {
          policyId: "clang.deprecated-declarations",
          clangDiagnosticGroup: "deprecated-declarations",
          argv: {
            ignore: ["-Wno-deprecated-declarations"],
            warn: ["-Wdeprecated-declarations", "-Wno-error=deprecated-declarations"],
            error: ["-Wdeprecated-declarations", "-Werror=deprecated-declarations"],
          },
        },
        {
          policyId: "clang.sign-compare",
          clangDiagnosticGroup: "sign-compare",
          argv: {
            ignore: ["-Wno-sign-compare"],
            warn: ["-Wsign-compare", "-Wno-error=sign-compare"],
            error: ["-Wsign-compare", "-Werror=sign-compare"],
          },
        },
        {
          policyId: "clang.unknown-pragmas",
          clangDiagnosticGroup: "unknown-pragmas",
          argv: {
            ignore: ["-Wno-unknown-pragmas"],
            warn: ["-Wunknown-pragmas", "-Wno-error=unknown-pragmas"],
            error: ["-Wunknown-pragmas", "-Werror=unknown-pragmas"],
          },
        },
        {
          policyId: "clang.unused-function",
          clangDiagnosticGroup: "unused-function",
          argv: {
            ignore: ["-Wno-unused-function"],
            warn: ["-Wunused-function", "-Wno-error=unused-function"],
            error: ["-Wunused-function", "-Werror=unused-function"],
          },
        },
        {
          policyId: "clang.unused-parameter",
          clangDiagnosticGroup: "unused-parameter",
          argv: {
            ignore: ["-Wno-unused-parameter"],
            warn: ["-Wunused-parameter", "-Wno-error=unused-parameter"],
            error: ["-Wunused-parameter", "-Werror=unused-parameter"],
          },
        },
        {
          policyId: "clang.unused-variable",
          clangDiagnosticGroup: "unused-variable",
          argv: {
            ignore: ["-Wno-unused-variable"],
            warn: ["-Wunused-variable", "-Wno-error=unused-variable"],
            error: ["-Wunused-variable", "-Werror=unused-variable"],
          },
        },
      ],
    },
  },
} as const satisfies JsonObject;

export type CppCuteSemanticAdapterManifestV1Resource =
  typeof CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_VALUE;

export const CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE = deepFreezeJson(
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_VALUE,
) as unknown as CppCuteSemanticAdapterManifestV1Resource;
