#include "BrowserGradCppCuteCommandLine.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace browsergrad::cpp_cute {
namespace {

constexpr std::size_t kMaximumOptionCount = 4'096U;
constexpr std::size_t kMaximumMacroValueByteLength = 1'024U;

struct WarningMapping {
  std::string_view external_id;
  std::string_view clang_group;
  std::array<std::string_view, 1U> ignore_arguments;
  std::array<std::string_view, 2U> warn_arguments;
  std::array<std::string_view, 2U> error_arguments;
};

#include "BrowserGradCppCuteCommandLinePolicy.inc"

enum class MacroAction : std::uint8_t {
  kDefine,
  kUndefine,
};

bool is_ascii_letter(std::uint8_t value) noexcept {
  return (value >= static_cast<std::uint8_t>('A') &&
          value <= static_cast<std::uint8_t>('Z')) ||
         (value >= static_cast<std::uint8_t>('a') &&
          value <= static_cast<std::uint8_t>('z'));
}

bool is_ascii_digit(std::uint8_t value) noexcept {
  return value >= static_cast<std::uint8_t>('0') &&
         value <= static_cast<std::uint8_t>('9');
}

bool is_macro_name(std::string_view name) noexcept {
  if (name.empty()) return false;
  const auto first = static_cast<std::uint8_t>(name.front());
  if (!is_ascii_letter(first) && first != static_cast<std::uint8_t>('_')) {
    return false;
  }
  for (const char character : name.substr(1U)) {
    const auto value = static_cast<std::uint8_t>(character);
    if (!is_ascii_letter(value) && !is_ascii_digit(value) &&
        value != static_cast<std::uint8_t>('_')) {
      return false;
    }
  }
  return true;
}

bool is_temporal_macro(std::string_view name) noexcept {
  for (const std::string_view temporal_name : kTemporalMacroNames) {
    if (name == temporal_name) return true;
  }
  return false;
}

bool is_reserved_macro(std::string_view name) noexcept {
  return name == "defined" || name.front() == '_' ||
         name.find("__") != std::string_view::npos;
}

bool valid_macro_value(std::string_view value) noexcept {
  return !value.empty() && value.size() <= kMaximumMacroValueByteLength &&
         value.find('\0') == std::string_view::npos;
}

const WarningMapping* warning_mapping(std::string_view external_id) noexcept {
  for (const WarningMapping& mapping : kWarningMappings) {
    if (mapping.external_id == external_id) return &mapping;
  }
  return nullptr;
}

template <std::size_t Size>
void append_warning_arguments(
    std::vector<std::string>& arguments,
    const std::array<std::string_view, Size>& generated_arguments) {
  for (const std::string_view argument : generated_arguments) {
    arguments.emplace_back(argument);
  }
}

void append_macro_argument(std::vector<std::string>& arguments, char action,
                           std::string_view name,
                           std::optional<std::string_view> value) {
  std::size_t byte_length = 2U + name.size();
  if (value.has_value()) byte_length += 1U + value->size();
  std::string argument;
  argument.reserve(byte_length);
  argument.push_back('-');
  argument.push_back(action);
  argument.append(name);
  if (value.has_value()) {
    argument.push_back('=');
    argument.append(*value);
  }
  arguments.push_back(std::move(argument));
}

MaterializedCompilerPolicy reject(CommandLineMaterializationStatus status,
                                  std::size_t option_index) {
  MaterializedCompilerPolicy result;
  result.status = status;
  result.error_option_index = static_cast<std::uint32_t>(option_index);
  return result;
}

CommandLineMaterializationStatus validate_macro_name(
    std::string_view name) noexcept {
  if (!is_macro_name(name)) {
    return CommandLineMaterializationStatus::kInvalidMacroName;
  }
  if (is_temporal_macro(name)) {
    return CommandLineMaterializationStatus::kTemporalMacroName;
  }
  if (is_reserved_macro(name)) {
    return CommandLineMaterializationStatus::kReservedMacroName;
  }
  return CommandLineMaterializationStatus::kValid;
}

}  // namespace

std::string_view compiler_temporal_macro_policy_id() noexcept {
  return kTemporalMacroPolicyId;
}

std::string_view compiler_warning_policy_registry_id() noexcept {
  return kWarningPolicyRegistryId;
}

MaterializedCompilerPolicy materialize_compiler_policy_command_line(
    std::span<const CompilerPolicyOption> options) {
  if (options.size() > kMaximumOptionCount) {
    return reject(CommandLineMaterializationStatus::kTooManyOptions,
                  kMaximumOptionCount);
  }

  MaterializedCompilerPolicy result;
  result.arguments.reserve(options.size() * 2U);
  std::unordered_map<std::string_view, MacroAction> macro_actions;
  std::unordered_set<std::string_view> warning_ids;
  macro_actions.reserve(options.size());
  warning_ids.reserve(options.size());

  for (std::size_t index = 0U; index < options.size(); ++index) {
    const CompilerPolicyOption& option = options[index];
    if (const auto* define = std::get_if<CompilerDefineOption>(&option)) {
      const CommandLineMaterializationStatus name_status =
          validate_macro_name(define->name);
      if (name_status != CommandLineMaterializationStatus::kValid) {
        return reject(name_status, index);
      }
      if (define->value.has_value() && !valid_macro_value(*define->value)) {
        return reject(CommandLineMaterializationStatus::kInvalidMacroValue,
                      index);
      }
      const auto [entry, inserted] =
          macro_actions.emplace(define->name, MacroAction::kDefine);
      if (!inserted) {
        return reject(
            entry->second == MacroAction::kDefine
                ? CommandLineMaterializationStatus::kDuplicateOption
                : CommandLineMaterializationStatus::kConflictingMacroAction,
            index);
      }
      append_macro_argument(result.arguments, 'D', define->name,
                            define->value);
      continue;
    }
    if (const auto* undefine = std::get_if<CompilerUndefineOption>(&option)) {
      const CommandLineMaterializationStatus name_status =
          validate_macro_name(undefine->name);
      if (name_status != CommandLineMaterializationStatus::kValid) {
        return reject(name_status, index);
      }
      const auto [entry, inserted] =
          macro_actions.emplace(undefine->name, MacroAction::kUndefine);
      if (!inserted) {
        return reject(
            entry->second == MacroAction::kUndefine
                ? CommandLineMaterializationStatus::kDuplicateOption
                : CommandLineMaterializationStatus::kConflictingMacroAction,
            index);
      }
      append_macro_argument(result.arguments, 'U', undefine->name,
                            std::nullopt);
      continue;
    }

    const auto& warning = std::get<CompilerWarningPolicyOption>(option);
    const WarningMapping* mapping = warning_mapping(warning.id);
    if (mapping == nullptr) {
      return reject(CommandLineMaterializationStatus::kUnknownWarningId,
                    index);
    }
    if (!warning_ids.insert(warning.id).second) {
      return reject(CommandLineMaterializationStatus::kDuplicateOption,
                    index);
    }
    switch (warning.disposition) {
      case WarningDisposition::kIgnore:
        append_warning_arguments(result.arguments, mapping->ignore_arguments);
        break;
      case WarningDisposition::kWarn:
        append_warning_arguments(result.arguments, mapping->warn_arguments);
        break;
      case WarningDisposition::kError:
        append_warning_arguments(result.arguments, mapping->error_arguments);
        break;
      default:
        return reject(
            CommandLineMaterializationStatus::kUnknownWarningDisposition,
            index);
    }
  }
  result.error_option_index = static_cast<std::uint32_t>(options.size());
  return result;
}

}  // namespace browsergrad::cpp_cute
