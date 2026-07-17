#include "extractor/BrowserGradCppCuteCommandLine.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace browsergrad::cpp_cute;

#define BG_CHECK(condition)                                                  \
  do {                                                                       \
    if (!(condition)) {                                                      \
      std::fprintf(stderr, "command-line check failed at line %d: %s\n",  \
                   __LINE__, #condition);                                    \
      return 1;                                                              \
    }                                                                        \
  } while (false)

struct WarningVector {
  std::string_view external_id;
  std::string_view clang_group;
};

bool rejects(std::span<const CompilerPolicyOption> options,
             CommandLineMaterializationStatus status,
             std::uint32_t error_option_index) {
  const MaterializedCompilerPolicy result =
      materialize_compiler_policy_command_line(options);
  return result.status == status &&
         result.error_option_index == error_option_index &&
         result.arguments.empty();
}

int run_valid_policy_tests() {
  constexpr std::array<WarningVector, 6U> kWarnings = {{
      {"clang.deprecated-declarations", "deprecated-declarations"},
      {"clang.sign-compare", "sign-compare"},
      {"clang.unknown-pragmas", "unknown-pragmas"},
      {"clang.unused-function", "unused-function"},
      {"clang.unused-parameter", "unused-parameter"},
      {"clang.unused-variable", "unused-variable"},
  }};
  BG_CHECK(compiler_temporal_macro_policy_id() ==
           "browsergrad.compiler.cpp-cute.temporal-macros.reject@1");
  BG_CHECK(compiler_warning_policy_registry_id() ==
           "browsergrad.compiler.cpp-cute.clang-warning-registry@1");
  for (const WarningVector& warning : kWarnings) {
    const std::array<CompilerPolicyOption, 1U> ignore = {
        CompilerWarningPolicyOption{warning.external_id,
                                    WarningDisposition::kIgnore},
    };
    const MaterializedCompilerPolicy ignored =
        materialize_compiler_policy_command_line(ignore);
    BG_CHECK(ignored.status == CommandLineMaterializationStatus::kValid);
    BG_CHECK(ignored.arguments ==
             std::vector<std::string>{"-Wno-" +
                                      std::string(warning.clang_group)});

    const std::array<CompilerPolicyOption, 1U> warn = {
        CompilerWarningPolicyOption{warning.external_id,
                                    WarningDisposition::kWarn},
    };
    const MaterializedCompilerPolicy warned =
        materialize_compiler_policy_command_line(warn);
    BG_CHECK(warned.status == CommandLineMaterializationStatus::kValid);
    const std::vector<std::string> expected_warned = {
        "-W" + std::string(warning.clang_group),
        "-Wno-error=" + std::string(warning.clang_group),
    };
    BG_CHECK(warned.arguments == expected_warned);

    const std::array<CompilerPolicyOption, 1U> error = {
        CompilerWarningPolicyOption{warning.external_id,
                                    WarningDisposition::kError},
    };
    const MaterializedCompilerPolicy errored =
        materialize_compiler_policy_command_line(error);
    BG_CHECK(errored.status == CommandLineMaterializationStatus::kValid);
    const std::vector<std::string> expected_errored = {
        "-W" + std::string(warning.clang_group),
        "-Werror=" + std::string(warning.clang_group),
    };
    BG_CHECK(errored.arguments == expected_errored);
  }

  const std::array<CompilerPolicyOption, 4U> policy = {
      CompilerDefineOption{"CUTE_ENABLED", std::nullopt},
      CompilerDefineOption{"MESSAGE", "x y;$(not-a-shell)"},
      CompilerUndefineOption{"LEGACY_MODE"},
      CompilerWarningPolicyOption{"clang.unused-parameter",
                                  WarningDisposition::kWarn},
  };
  const MaterializedCompilerPolicy materialized =
      materialize_compiler_policy_command_line(policy);
  BG_CHECK(materialized.status == CommandLineMaterializationStatus::kValid);
  BG_CHECK(materialized.error_option_index == policy.size());
  const std::vector<std::string> expected_policy = {
      "-DCUTE_ENABLED",
      "-DMESSAGE=x y;$(not-a-shell)",
      "-ULEGACY_MODE",
      "-Wunused-parameter",
      "-Wno-error=unused-parameter",
  };
  BG_CHECK(materialized.arguments == expected_policy);

  const std::array<CompilerPolicyOption, 0U> empty = {};
  const MaterializedCompilerPolicy no_arguments =
      materialize_compiler_policy_command_line(empty);
  BG_CHECK(no_arguments.status == CommandLineMaterializationStatus::kValid);
  BG_CHECK(no_arguments.error_option_index == 0U);
  BG_CHECK(no_arguments.arguments.empty());
  return 0;
}

int run_rejection_tests() {
  const std::array<CompilerPolicyOption, 1U> unknown_warning = {
      CompilerWarningPolicyOption{"clang.everything",
                                  WarningDisposition::kError},
  };
  BG_CHECK(rejects(unknown_warning,
                   CommandLineMaterializationStatus::kUnknownWarningId, 0U));

  const std::array<CompilerPolicyOption, 1U> unnamespaced_warning = {
      CompilerWarningPolicyOption{"unused-parameter",
                                  WarningDisposition::kWarn},
  };
  BG_CHECK(rejects(unnamespaced_warning,
                   CommandLineMaterializationStatus::kUnknownWarningId, 0U));

  const std::array<CompilerPolicyOption, 1U> unknown_disposition = {
      CompilerWarningPolicyOption{
          "clang.unused-variable", static_cast<WarningDisposition>(0xffU)},
  };
  BG_CHECK(rejects(
      unknown_disposition,
      CommandLineMaterializationStatus::kUnknownWarningDisposition, 0U));

  const std::array<CompilerPolicyOption, 2U> duplicate_warning = {
      CompilerWarningPolicyOption{"clang.sign-compare",
                                  WarningDisposition::kWarn},
      CompilerWarningPolicyOption{"clang.sign-compare",
                                  WarningDisposition::kError},
  };
  BG_CHECK(rejects(duplicate_warning,
                   CommandLineMaterializationStatus::kDuplicateOption, 1U));

  const std::array<CompilerPolicyOption, 2U> duplicate_define = {
      CompilerDefineOption{"MODE", "1"},
      CompilerDefineOption{"MODE", "2"},
  };
  BG_CHECK(rejects(duplicate_define,
                   CommandLineMaterializationStatus::kDuplicateOption, 1U));

  const std::array<CompilerPolicyOption, 2U> duplicate_undefine = {
      CompilerUndefineOption{"MODE"},
      CompilerUndefineOption{"MODE"},
  };
  BG_CHECK(rejects(duplicate_undefine,
                   CommandLineMaterializationStatus::kDuplicateOption, 1U));

  const std::array<CompilerPolicyOption, 2U> define_then_undefine = {
      CompilerDefineOption{"MODE", "1"},
      CompilerUndefineOption{"MODE"},
  };
  BG_CHECK(rejects(
      define_then_undefine,
      CommandLineMaterializationStatus::kConflictingMacroAction, 1U));

  const std::array<CompilerPolicyOption, 2U> undefine_then_define = {
      CompilerUndefineOption{"MODE"},
      CompilerDefineOption{"MODE", "1"},
  };
  BG_CHECK(rejects(
      undefine_then_define,
      CommandLineMaterializationStatus::kConflictingMacroAction, 1U));

  for (const std::string_view name : {"_PRIVATE", "name__impl", "defined"}) {
    const std::array<CompilerPolicyOption, 1U> define = {
        CompilerDefineOption{name, "1"},
    };
    BG_CHECK(rejects(define,
                     CommandLineMaterializationStatus::kReservedMacroName,
                     0U));
    const std::array<CompilerPolicyOption, 1U> undefine = {
        CompilerUndefineOption{name},
    };
    BG_CHECK(rejects(undefine,
                     CommandLineMaterializationStatus::kReservedMacroName,
                     0U));
  }

  for (const std::string_view name :
       {"__DATE__", "__TIME__", "__TIMESTAMP__"}) {
    const std::array<CompilerPolicyOption, 1U> define = {
        CompilerDefineOption{name, "redacted"},
    };
    BG_CHECK(rejects(define,
                     CommandLineMaterializationStatus::kTemporalMacroName,
                     0U));
    const std::array<CompilerPolicyOption, 1U> undefine = {
        CompilerUndefineOption{name},
    };
    BG_CHECK(rejects(undefine,
                     CommandLineMaterializationStatus::kTemporalMacroName,
                     0U));
  }

  for (const std::string_view name : {"", "9MODE", "HAS-DASH", "é"}) {
    const std::array<CompilerPolicyOption, 1U> invalid_name = {
        CompilerDefineOption{name, "1"},
    };
    BG_CHECK(rejects(invalid_name,
                     CommandLineMaterializationStatus::kInvalidMacroName,
                     0U));
  }

  const std::array<CompilerPolicyOption, 1U> empty_value = {
      CompilerDefineOption{"MODE", ""},
  };
  BG_CHECK(rejects(empty_value,
                   CommandLineMaterializationStatus::kInvalidMacroValue, 0U));
  constexpr std::string_view kNulValue("x\0y", 3U);
  const std::array<CompilerPolicyOption, 1U> nul_value = {
      CompilerDefineOption{"MODE", kNulValue},
  };
  BG_CHECK(rejects(nul_value,
                   CommandLineMaterializationStatus::kInvalidMacroValue, 0U));
  const std::string oversized_value(1'025U, 'x');
  const std::array<CompilerPolicyOption, 1U> oversized = {
      CompilerDefineOption{"MODE", oversized_value},
  };
  BG_CHECK(rejects(oversized,
                   CommandLineMaterializationStatus::kInvalidMacroValue, 0U));

  const CompilerPolicyOption repeated =
      CompilerDefineOption{"MODE", std::nullopt};
  const std::vector<CompilerPolicyOption> too_many(4'097U, repeated);
  BG_CHECK(rejects(too_many,
                   CommandLineMaterializationStatus::kTooManyOptions,
                   4'096U));
  return 0;
}

}  // namespace

int main() {
  if (run_valid_policy_tests() != 0) return 1;
  return run_rejection_tests();
}
