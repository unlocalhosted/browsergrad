#include "extractor/BrowserGradCppCutePreprocessorPolicy.h"

#include <array>
#include <cstdio>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

using namespace browsergrad::cpp_cute;

#define BG_CHECK(condition)                                                   \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::fprintf(stderr, "preprocessor policy check failed at line %d: "   \
                           "%s\n",                                           \
                   __LINE__, #condition);                                     \
      return 1;                                                               \
    }                                                                         \
  } while (false)

int run_preprocessor_policy_tests() {
  static_assert(
      !std::is_copy_constructible_v<CppCutePreprocessorPolicyState>);
  static_assert(!std::is_copy_assignable_v<CppCutePreprocessorPolicyState>);
  static_assert(
      !std::is_move_constructible_v<CppCutePreprocessorPolicyState>);
  static_assert(!std::is_move_assignable_v<CppCutePreprocessorPolicyState>);

  constexpr std::array temporal_macros{
      std::pair{"__DATE__", TemporalMacroKind::kDate},
      std::pair{"__TIMESTAMP__", TemporalMacroKind::kTimestamp},
      std::pair{"__TIME__", TemporalMacroKind::kTime},
  };
  for (const auto& [name, kind] : temporal_macros) {
    const auto classified = classify_temporal_macro(name);
    BG_CHECK(classified.has_value());
    BG_CHECK(*classified == kind);
    BG_CHECK(temporal_macro_name(kind) == name);
  }

  // Exact identifier classification leaves comments, literals, aliases,
  // case variants, and neighboring builtins inert until Clang reports a real
  // macro event for one of the three names.
  constexpr std::array inert_spellings{
      "DATE",
      "__date__",
      "__DATE",
      "__DATE___",
      "__FILE__",
      "\"__DATE__\"",
      "// __TIME__",
      "/* __TIMESTAMP__ */",
  };
  for (const std::string_view spelling : inert_spellings) {
    BG_CHECK(!classify_temporal_macro(spelling).has_value());
  }

  constexpr std::array consultation_uses{
      TemporalMacroUse::kExpansion,
      TemporalMacroUse::kDefined,
      TemporalMacroUse::kIfdef,
      TemporalMacroUse::kIfndef,
      TemporalMacroUse::kElifdef,
      TemporalMacroUse::kElifndef,
  };
  for (const TemporalMacroUse use : consultation_uses) {
    BG_CHECK(!temporal_macro_use_is_mutation(use));
    BG_CHECK(temporal_macro_diagnostic_code(use) ==
             kTemporalMacroForbiddenDiagnosticCode);
  }
  constexpr std::array mutation_uses{
      TemporalMacroUse::kDefine,
      TemporalMacroUse::kUndefine,
  };
  for (const TemporalMacroUse use : mutation_uses) {
    BG_CHECK(temporal_macro_use_is_mutation(use));
    BG_CHECK(temporal_macro_diagnostic_code(use) ==
             kTemporalMacroMutationForbiddenDiagnosticCode);
  }

  BG_CHECK(kCppCuteTemporalMacroPolicyId ==
           "browsergrad.compiler.cpp-cute.temporal-macros.reject@1");
  BG_CHECK(std::string_view(kTemporalMacroForbiddenDiagnosticMessage)
               .starts_with(kTemporalMacroForbiddenDiagnosticCode));
  BG_CHECK(std::string_view(kTemporalMacroMutationForbiddenDiagnosticMessage)
               .starts_with(
                   kTemporalMacroMutationForbiddenDiagnosticCode));
  BG_CHECK(std::string_view(kTemporalMacroForbiddenDiagnosticMessage)
               .find("%0") != std::string_view::npos);
  BG_CHECK(std::string_view(kTemporalMacroMutationForbiddenDiagnosticMessage)
               .find("%0") != std::string_view::npos);
  BG_CHECK(kTemporalMacroRejectedRecoveryEpoch == 0U);

  CppCutePreprocessorPolicyState state;
  BG_CHECK(!state.failed());
  BG_CHECK(state.violation_count() == 0U);
  BG_CHECK(!state.first_violation().has_value());

  CppCutePreprocessorPolicyInstallation absent;
  BG_CHECK(!static_cast<bool>(absent));
  BG_CHECK(absent.status ==
           CppCutePreprocessorPolicyInstallStatus::kMissingPreprocessor);
  CppCutePreprocessorPolicyInstallation installed{
      CppCutePreprocessorPolicyInstallStatus::kInstalled,
      std::make_shared<CppCutePreprocessorPolicyState>(),
  };
  BG_CHECK(static_cast<bool>(installed));

  return 0;
}

}  // namespace

int main() { return run_preprocessor_policy_tests(); }
