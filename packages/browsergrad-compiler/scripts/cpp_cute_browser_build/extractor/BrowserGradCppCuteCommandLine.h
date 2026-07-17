#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace browsergrad::cpp_cute {

enum class WarningDisposition : std::uint8_t {
  kIgnore,
  kWarn,
  kError,
};

struct CompilerDefineOption {
  std::string_view name;
  std::optional<std::string_view> value;
};

struct CompilerUndefineOption {
  std::string_view name;
};

struct CompilerWarningPolicyOption {
  /** Closed external ID, for example clang.unused-parameter. */
  std::string_view id;
  WarningDisposition disposition = WarningDisposition::kWarn;
};

/** Typed policy only. Raw argv and shell command strings are not accepted. */
using CompilerPolicyOption =
    std::variant<CompilerDefineOption, CompilerUndefineOption,
                 CompilerWarningPolicyOption>;

enum class CommandLineMaterializationStatus : std::uint8_t {
  kValid,
  kTooManyOptions,
  kInvalidMacroName,
  kInvalidMacroValue,
  kReservedMacroName,
  kTemporalMacroName,
  kUnknownWarningId,
  kUnknownWarningDisposition,
  kDuplicateOption,
  kConflictingMacroAction,
};

struct MaterializedCompilerPolicy {
  CommandLineMaterializationStatus status =
      CommandLineMaterializationStatus::kValid;
  std::uint32_t error_option_index = 0U;
  /** Exact owning argv elements. Never a shell-escaped or joined command. */
  std::vector<std::string> arguments;
};

std::string_view compiler_temporal_macro_policy_id() noexcept;
std::string_view compiler_warning_policy_registry_id() noexcept;

/**
 * Materializes closed compiler policy into exact owning argv elements.
 *
 * On rejection, arguments is empty and error_option_index identifies the
 * first rejected option. Input order is preserved for valid policy.
 */
MaterializedCompilerPolicy materialize_compiler_policy_command_line(
    std::span<const CompilerPolicyOption> options);

}  // namespace browsergrad::cpp_cute
