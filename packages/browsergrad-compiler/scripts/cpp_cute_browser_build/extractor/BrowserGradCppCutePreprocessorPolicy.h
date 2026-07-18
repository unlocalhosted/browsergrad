#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>

namespace clang {
class CompilerInstance;
}

namespace browsergrad::cpp_cute {

inline constexpr std::string_view kCppCuteTemporalMacroPolicyId =
    "browsergrad.compiler.cpp-cute.temporal-macros.reject@1";
inline constexpr std::string_view kTemporalMacroForbiddenDiagnosticCode =
    "browsergrad.cpp-cute:temporal-macro-forbidden";
inline constexpr std::string_view
    kTemporalMacroMutationForbiddenDiagnosticCode =
        "browsergrad.cpp-cute:temporal-macro-mutation-forbidden";
inline constexpr char kTemporalMacroForbiddenDiagnosticMessage[] =
    "browsergrad.cpp-cute:temporal-macro-forbidden: temporal predefined "
    "macro '%0' is forbidden by the compilation contract";
inline constexpr char kTemporalMacroMutationForbiddenDiagnosticMessage[] =
    "browsergrad.cpp-cute:temporal-macro-mutation-forbidden: source "
    "mutation of temporal predefined macro '%0' is forbidden by the "
    "compilation contract";
inline constexpr std::uint64_t kTemporalMacroRejectedRecoveryEpoch = 0U;

enum class TemporalMacroKind : std::uint8_t {
  kDate,
  kTimestamp,
  kTime,
};

enum class TemporalMacroUse : std::uint8_t {
  kExpansion,
  kDefined,
  kIfdef,
  kIfndef,
  kElifdef,
  kElifndef,
  kDefine,
  kUndefine,
};

struct TemporalMacroViolation {
  TemporalMacroKind macro = TemporalMacroKind::kDate;
  TemporalMacroUse use = TemporalMacroUse::kExpansion;
};

constexpr std::optional<TemporalMacroKind> classify_temporal_macro(
    std::string_view name) noexcept {
  if (name == "__DATE__") return TemporalMacroKind::kDate;
  if (name == "__TIMESTAMP__") return TemporalMacroKind::kTimestamp;
  if (name == "__TIME__") return TemporalMacroKind::kTime;
  return std::nullopt;
}

constexpr std::string_view temporal_macro_name(
    TemporalMacroKind macro) noexcept {
  switch (macro) {
    case TemporalMacroKind::kDate:
      return "__DATE__";
    case TemporalMacroKind::kTimestamp:
      return "__TIMESTAMP__";
    case TemporalMacroKind::kTime:
      return "__TIME__";
  }
  return {};
}

constexpr bool temporal_macro_use_is_mutation(TemporalMacroUse use) noexcept {
  return use == TemporalMacroUse::kDefine ||
         use == TemporalMacroUse::kUndefine;
}

constexpr std::string_view temporal_macro_diagnostic_code(
    TemporalMacroUse use) noexcept {
  return temporal_macro_use_is_mutation(use)
             ? kTemporalMacroMutationForbiddenDiagnosticCode
             : kTemporalMacroForbiddenDiagnosticCode;
}

class CppCuteTemporalMacroCallbacks;

class CppCutePreprocessorPolicyState final {
 public:
  CppCutePreprocessorPolicyState() = default;
  CppCutePreprocessorPolicyState(const CppCutePreprocessorPolicyState&) =
      delete;
  CppCutePreprocessorPolicyState& operator=(
      const CppCutePreprocessorPolicyState&) = delete;
  CppCutePreprocessorPolicyState(CppCutePreprocessorPolicyState&&) = delete;
  CppCutePreprocessorPolicyState& operator=(
      CppCutePreprocessorPolicyState&&) = delete;

  bool failed() const noexcept { return violation_count_ != 0U; }
  std::uint32_t violation_count() const noexcept { return violation_count_; }
  std::optional<TemporalMacroViolation> first_violation() const noexcept {
    return first_violation_;
  }

 private:
  friend class CppCuteTemporalMacroCallbacks;

  void record(TemporalMacroKind macro, TemporalMacroUse use) noexcept {
    if (!first_violation_.has_value()) {
      first_violation_ = TemporalMacroViolation{macro, use};
    }
    if (violation_count_ != std::numeric_limits<std::uint32_t>::max()) {
      ++violation_count_;
    }
  }

  std::optional<TemporalMacroViolation> first_violation_;
  std::uint32_t violation_count_ = 0U;
};

enum class CppCutePreprocessorPolicyInstallStatus : std::uint8_t {
  kInstalled,
  kMissingDiagnostics,
  kMissingPreprocessor,
};

struct CppCutePreprocessorPolicyInstallation {
  CppCutePreprocessorPolicyInstallStatus status =
      CppCutePreprocessorPolicyInstallStatus::kMissingPreprocessor;
  std::shared_ptr<CppCutePreprocessorPolicyState> state;
  std::uint32_t consultation_diagnostic_id = 0U;
  std::uint32_t mutation_diagnostic_id = 0U;

  explicit operator bool() const noexcept {
    return status == CppCutePreprocessorPolicyInstallStatus::kInstalled &&
           state != nullptr;
  }
};

/**
 * Installs the reject-only temporal macro policy into one initialized Clang
 * CompilerInstance. Call exactly once for each independent CUDA semantic pass,
 * after Clang creates its Preprocessor and before it begins lexing the input.
 *
 * The callback owns a shared pass state so the caller can reject the pass even
 * if Clang suppresses diagnostic rendering after reaching its error limit.
 */
CppCutePreprocessorPolicyInstallation install_cpp_cute_preprocessor_policy(
    clang::CompilerInstance& compiler);

}  // namespace browsergrad::cpp_cute
