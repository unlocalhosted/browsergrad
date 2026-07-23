#include "BrowserGradCppCuteInvocation.h"

#include "BrowserGradCppCuteVirtualPath.h"

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

constexpr std::string_view kCanonicalArgv0 = "clang++";
constexpr std::string_view kPinnedClangVersion = "22.1.8";
constexpr std::string_view kCudaDeviceTriple = "nvptx64-nvidia-cuda";
constexpr std::size_t kMaximumIncludeRootCount = 256U;
constexpr std::uint32_t kMaximumCompilerOptionCount = 4'096U;
constexpr std::uint32_t kMaximumErrorLimit = 999'999U;
constexpr std::size_t kMaximumArgumentCount = 20'000U;
constexpr std::size_t kMaximumArgumentByteLength = 32U * 1024U * 1024U;

// The generated include is the single native authority for temporal defense
// argv. Warning lowering itself remains owned by BrowserGradCppCuteCommandLine.
struct WarningMapping {
  std::string_view external_id;
  std::string_view clang_group;
  std::array<std::string_view, 1U> ignore_arguments;
  std::array<std::string_view, 2U> warn_arguments;
  std::array<std::string_view, 2U> error_arguments;
};

#include "BrowserGradCppCuteCommandLinePolicy.inc"

static_assert(kTemporalMacroNames.size() == 3U);
static_assert(kReservedClangDiagnosticGroups.size() ==
              kTemporalDefenseInDepthArgv.size());
static_assert(kWarningMappings.size() > 0U);

struct IncludeRootRecord {
  std::string_view virtual_path;
};

bool is_ascii_letter(unsigned char value) noexcept {
  return (value >= static_cast<unsigned char>('A') &&
          value <= static_cast<unsigned char>('Z')) ||
         (value >= static_cast<unsigned char>('a') &&
          value <= static_cast<unsigned char>('z'));
}

bool is_ascii_digit(unsigned char value) noexcept {
  return value >= static_cast<unsigned char>('0') &&
         value <= static_cast<unsigned char>('9');
}

bool is_ascii_lower(unsigned char value) noexcept {
  return value >= static_cast<unsigned char>('a') &&
         value <= static_cast<unsigned char>('z');
}

bool valid_identifier(std::string_view value) noexcept {
  if (value.empty() || value.size() > 128U ||
      !is_ascii_lower(static_cast<unsigned char>(value.front()))) {
    return false;
  }
  for (const char character : value.substr(1U)) {
    const auto byte = static_cast<unsigned char>(character);
    if (!is_ascii_lower(byte) && !is_ascii_digit(byte) && byte != '.' &&
        byte != '_' && byte != '-') {
      return false;
    }
  }
  return true;
}

bool valid_triple(std::string_view value) noexcept {
  if (value.empty() || value.size() > 256U || value.front() == '-' ||
      value.back() == '-' || value.find('-') == std::string_view::npos) {
    return false;
  }
  for (const char character : value) {
    const auto byte = static_cast<unsigned char>(character);
    if (!is_ascii_letter(byte) && !is_ascii_digit(byte) && byte != '_' &&
        byte != '.' && byte != '+' && byte != '-') {
      return false;
    }
  }
  return true;
}

bool valid_device_architecture(std::string_view value) noexcept {
  if (value.size() != 5U && value.size() != 6U) return false;
  if (!value.starts_with("sm_") || value[3] < '1' || value[3] > '9' ||
      value[4] < '0' || value[4] > '9') {
    return false;
  }
  return value.size() == 5U ||
         (value[5] >= 'a' && value[5] <= 'z');
}

MaterializedCppCuteInvocation reject(
    InvocationMaterializationStatus status, InvocationErrorField field,
    std::size_t index = 0U,
    CommandLineMaterializationStatus policy_status =
        CommandLineMaterializationStatus::kValid) {
  MaterializedCppCuteInvocation result;
  result.status = status;
  result.error_field = field;
  result.error_index = static_cast<std::uint32_t>(index);
  result.policy_status = policy_status;
  return result;
}

bool checked_add(std::size_t& total, std::size_t value,
                 std::size_t maximum) noexcept {
  if (value > maximum || total > maximum - value) return false;
  total += value;
  return true;
}

bool arguments_within_budget(
    std::span<const std::string> arguments) noexcept {
  if (arguments.size() > kMaximumArgumentCount) return false;
  std::size_t total = 0U;
  for (const std::string& argument : arguments) {
    if (!checked_add(total, argument.size(), kMaximumArgumentByteLength)) {
      return false;
    }
  }
  return true;
}

std::string joined_argument(std::string_view prefix, std::string_view value) {
  std::string result;
  result.reserve(prefix.size() + value.size());
  result.append(prefix);
  result.append(value);
  return result;
}

std::uint32_t compiler_option_ordinal(
    const InvocationCompilerOption& option) {
  return std::visit(
      [](const auto& value) { return value.compiler_option_ordinal; },
      option);
}

CommandLineMaterializationStatus append_policy_option(
    std::vector<std::string>& arguments,
    const CompilerPolicyOption& option) {
  const std::array single = {option};
  MaterializedCompilerPolicy materialized =
      materialize_compiler_policy_command_line(single);
  if (materialized.status != CommandLineMaterializationStatus::kValid) {
    return materialized.status;
  }
  for (std::string& argument : materialized.arguments) {
    arguments.push_back(std::move(argument));
  }
  return CommandLineMaterializationStatus::kValid;
}

}  // namespace

std::string_view cpp_cute_invocation_argv0() noexcept {
  return kCanonicalArgv0;
}

std::string_view cpp_cute_invocation_clang_version() noexcept {
  return kPinnedClangVersion;
}

MaterializedCppCuteInvocation materialize_cpp_cute_invocation(
    const CppCuteInvocationInput& input) {
  if (input.clang_version != kPinnedClangVersion) {
    return reject(InvocationMaterializationStatus::kWrongClangVersion,
                  InvocationErrorField::kClangVersion);
  }
  if (!cpp_cute_valid_canonical_virtual_path(input.main_source_virtual_path) ||
      input.main_source_virtual_path == "/") {
    return reject(InvocationMaterializationStatus::kInvalidMainSourcePath,
                  InvocationErrorField::kMainSource);
  }

  std::string_view host_triple;
  switch (input.semantic_pass.pass) {
    case CudaSemanticPass::kDeviceExtraction:
      if (input.semantic_pass.ordinal != 0U) {
        return reject(InvocationMaterializationStatus::kInvalidSemanticPass,
                      InvocationErrorField::kSemanticPass);
      }
      if (input.semantic_pass.target_triple != kCudaDeviceTriple) {
        return reject(InvocationMaterializationStatus::kInvalidTargetTriple,
                      InvocationErrorField::kSemanticPass);
      }
      if (!valid_triple(input.semantic_pass.auxiliary_target_triple) ||
          input.semantic_pass.auxiliary_target_triple == kCudaDeviceTriple) {
        return reject(
            InvocationMaterializationStatus::kInvalidAuxiliaryTargetTriple,
            InvocationErrorField::kSemanticPass);
      }
      host_triple = input.semantic_pass.auxiliary_target_triple;
      break;
    case CudaSemanticPass::kHostValidation:
      if (input.semantic_pass.ordinal != 1U) {
        return reject(InvocationMaterializationStatus::kInvalidSemanticPass,
                      InvocationErrorField::kSemanticPass);
      }
      if (!valid_triple(input.semantic_pass.target_triple) ||
          input.semantic_pass.target_triple == kCudaDeviceTriple) {
        return reject(InvocationMaterializationStatus::kInvalidTargetTriple,
                      InvocationErrorField::kSemanticPass);
      }
      if (input.semantic_pass.auxiliary_target_triple != kCudaDeviceTriple) {
        return reject(
            InvocationMaterializationStatus::kInvalidAuxiliaryTargetTriple,
            InvocationErrorField::kSemanticPass);
      }
      host_triple = input.semantic_pass.target_triple;
      break;
    default:
      return reject(InvocationMaterializationStatus::kInvalidSemanticPass,
                    InvocationErrorField::kSemanticPass);
  }
  if (!valid_device_architecture(
          input.semantic_pass.device_architecture)) {
    return reject(
        InvocationMaterializationStatus::kInvalidDeviceArchitecture,
        InvocationErrorField::kSemanticPass);
  }
  if (!cpp_cute_valid_canonical_virtual_path(
          input.resource_directory_virtual_path) ||
      input.resource_directory_virtual_path == "/" ||
      input.resource_directory_virtual_path.size() +
              std::string_view("/include").size() >
          kCppCuteMaximumVirtualPathByteLength) {
    return reject(
        InvocationMaterializationStatus::kInvalidResourceDirectoryPath,
        InvocationErrorField::kResourceDirectory);
  }
  if (!cpp_cute_valid_canonical_virtual_path(
          input.cuda_toolkit_root_virtual_path) ||
      input.cuda_toolkit_root_virtual_path == "/" ||
      input.cuda_toolkit_root_virtual_path.size() +
              std::string_view("/include").size() >
          kCppCuteMaximumVirtualPathByteLength) {
    return reject(
        InvocationMaterializationStatus::kInvalidCudaToolkitRootPath,
        InvocationErrorField::kCudaToolkit);
  }
  if (input.include_roots.empty()) {
    return reject(InvocationMaterializationStatus::kMissingIncludeRoots,
                  InvocationErrorField::kIncludeRoot);
  }
  if (input.include_roots.size() > kMaximumIncludeRootCount) {
    return reject(InvocationMaterializationStatus::kTooManyIncludeRoots,
                  InvocationErrorField::kIncludeRoot,
                  kMaximumIncludeRootCount);
  }

  std::unordered_map<std::string_view, IncludeRootRecord> roots_by_id;
  std::unordered_set<std::string_view> root_paths;
  roots_by_id.reserve(input.include_roots.size());
  root_paths.reserve(input.include_roots.size());
  const std::string resource_include_path = joined_argument(
      input.resource_directory_virtual_path, "/include");
  const std::string cuda_include_path = joined_argument(
      input.cuda_toolkit_root_virtual_path, "/include");
  bool has_resource_include_root = false;
  bool has_cuda_include_root = false;
  bool has_main_source_include_root = false;
  for (std::size_t index = 0U; index < input.include_roots.size(); ++index) {
    const InvocationIncludeRoot& root = input.include_roots[index];
    if (root.ordinal != index) {
      return reject(
          InvocationMaterializationStatus::kInvalidIncludeRootOrdinal,
          InvocationErrorField::kIncludeRoot, index);
    }
    if (!valid_identifier(root.id)) {
      return reject(InvocationMaterializationStatus::kInvalidIncludeRootId,
                    InvocationErrorField::kIncludeRoot, index);
    }
    if (root.mode != InvocationIncludeMode::kQuote &&
        root.mode != InvocationIncludeMode::kSystem) {
      return reject(InvocationMaterializationStatus::kInvalidIncludeRootMode,
                    InvocationErrorField::kIncludeRoot, index);
    }
    if (!cpp_cute_valid_canonical_virtual_path(root.virtual_path)) {
      return reject(InvocationMaterializationStatus::kInvalidIncludeRootPath,
                    InvocationErrorField::kIncludeRoot, index);
    }
    if (!roots_by_id
             .emplace(root.id, IncludeRootRecord{root.virtual_path})
             .second) {
      return reject(InvocationMaterializationStatus::kDuplicateIncludeRootId,
                    InvocationErrorField::kIncludeRoot, index);
    }
    if (!root_paths.insert(root.virtual_path).second) {
      return reject(
          InvocationMaterializationStatus::kDuplicateIncludeRootPath,
          InvocationErrorField::kIncludeRoot, index);
    }
    if (root.virtual_path == resource_include_path) {
      if (root.mode != InvocationIncludeMode::kSystem) {
        return reject(
            InvocationMaterializationStatus::kInvalidResourceIncludeRoot,
            InvocationErrorField::kIncludeRoot, index);
      }
      has_resource_include_root = true;
    }
    if (root.virtual_path == cuda_include_path) {
      if (root.mode != InvocationIncludeMode::kSystem) {
        return reject(
            InvocationMaterializationStatus::kInvalidCudaToolkitIncludeRoot,
            InvocationErrorField::kIncludeRoot, index);
      }
      has_cuda_include_root = true;
    }
    if (root.mode == InvocationIncludeMode::kQuote &&
        cpp_cute_virtual_path_contains(root.virtual_path,
                                       input.main_source_virtual_path)) {
      has_main_source_include_root = true;
    }
  }
  if (!has_resource_include_root) {
    return reject(
        InvocationMaterializationStatus::kMissingResourceIncludeRoot,
        InvocationErrorField::kIncludeRoot);
  }
  if (!has_cuda_include_root) {
    return reject(
        InvocationMaterializationStatus::kMissingCudaToolkitIncludeRoot,
        InvocationErrorField::kCudaToolkit);
  }
  if (!has_main_source_include_root) {
    return reject(
        InvocationMaterializationStatus::kMissingMainSourceIncludeRoot,
        InvocationErrorField::kIncludeRoot);
  }

  if (input.compiler_options.size() > kMaximumCompilerOptionCount) {
    return reject(InvocationMaterializationStatus::kTooManyCompilerOptions,
                  InvocationErrorField::kCompilerOption,
                  kMaximumCompilerOptionCount);
  }
  std::unordered_set<std::string_view> forced_paths;
  forced_paths.reserve(input.compiler_options.size());
  std::vector<CompilerPolicyOption> policy_options;
  std::vector<std::size_t> policy_option_indexes;
  policy_options.reserve(input.compiler_options.size());
  policy_option_indexes.reserve(input.compiler_options.size());
  bool has_syntax_only = false;
  bool has_error_limit = false;
  for (std::size_t index = 0U; index < input.compiler_options.size(); ++index) {
    const InvocationCompilerOption& option = input.compiler_options[index];
    if (option.valueless_by_exception()) {
      return reject(
          InvocationMaterializationStatus::kInvalidCompilerOption,
          InvocationErrorField::kCompilerOption, index);
    }
    if (compiler_option_ordinal(option) != index) {
      return reject(
          InvocationMaterializationStatus::kInvalidCompilerOptionOrdinal,
          InvocationErrorField::kCompilerOption, index);
    }
    if (const auto* define = std::get_if<InvocationDefineOption>(&option)) {
      policy_options.emplace_back(define->option);
      policy_option_indexes.push_back(index);
      continue;
    }
    if (const auto* undefine =
            std::get_if<InvocationUndefineOption>(&option)) {
      policy_options.emplace_back(undefine->option);
      policy_option_indexes.push_back(index);
      continue;
    }
    if (const auto* warning = std::get_if<InvocationWarningOption>(&option)) {
      policy_options.emplace_back(warning->option);
      policy_option_indexes.push_back(index);
      continue;
    }
    if (std::get_if<InvocationSyntaxOnlyOption>(&option) != nullptr) {
      if (has_syntax_only) {
        return reject(
            InvocationMaterializationStatus::kDuplicateSyntaxOnlyOption,
            InvocationErrorField::kCompilerOption, index);
      }
      has_syntax_only = true;
      continue;
    }
    if (const auto* error_limit =
            std::get_if<InvocationErrorLimitOption>(&option)) {
      if (has_error_limit) {
        return reject(
            InvocationMaterializationStatus::kDuplicateErrorLimitOption,
            InvocationErrorField::kCompilerOption, index);
      }
      if (error_limit->value == 0U ||
          error_limit->value > kMaximumErrorLimit) {
        return reject(InvocationMaterializationStatus::kInvalidErrorLimit,
                      InvocationErrorField::kCompilerOption, index);
      }
      has_error_limit = true;
      continue;
    }

    const auto* forced = std::get_if<InvocationForcedInclude>(&option);
    if (forced == nullptr) {
      return reject(InvocationMaterializationStatus::kInvalidCompilerOption,
                    InvocationErrorField::kCompilerOption, index);
    }
    const auto root = roots_by_id.find(forced->include_root_id);
    if (root == roots_by_id.end()) {
      return reject(
          InvocationMaterializationStatus::kUnknownForcedIncludeRoot,
          InvocationErrorField::kCompilerOption, index);
    }
    if (!cpp_cute_valid_canonical_virtual_path(forced->virtual_path) ||
        forced->virtual_path == "/") {
      return reject(
          InvocationMaterializationStatus::kInvalidForcedIncludePath,
          InvocationErrorField::kCompilerOption, index);
    }
    if (forced->virtual_path == root->second.virtual_path ||
        !cpp_cute_virtual_path_contains(root->second.virtual_path,
                                        forced->virtual_path)) {
      return reject(
          InvocationMaterializationStatus::kForcedIncludeOutsideRoot,
          InvocationErrorField::kCompilerOption, index);
    }
    if (!forced_paths.insert(forced->virtual_path).second) {
      return reject(
          InvocationMaterializationStatus::kDuplicateForcedIncludePath,
          InvocationErrorField::kCompilerOption, index);
    }
    if (forced->virtual_path == input.main_source_virtual_path) {
      return reject(
          InvocationMaterializationStatus::kForcedIncludeConflictsWithMainSource,
          InvocationErrorField::kCompilerOption, index);
    }
  }
  if (!has_syntax_only) {
    return reject(InvocationMaterializationStatus::kMissingSyntaxOnlyOption,
                  InvocationErrorField::kCompilerOption,
                  input.compiler_options.size());
  }
  if (!has_error_limit) {
    return reject(InvocationMaterializationStatus::kMissingErrorLimitOption,
                  InvocationErrorField::kCompilerOption,
                  input.compiler_options.size());
  }

  MaterializedCompilerPolicy policy =
      materialize_compiler_policy_command_line(policy_options);
  if (policy.status != CommandLineMaterializationStatus::kValid) {
    const std::size_t policy_index = policy.error_option_index;
    const std::size_t input_index =
        policy_index < policy_option_indexes.size()
            ? policy_option_indexes[policy_index]
            : input.compiler_options.size();
    return reject(InvocationMaterializationStatus::kCompilerPolicyRejected,
                  InvocationErrorField::kCompilerOption, input_index,
                  policy.status);
  }

  MaterializedCppCuteInvocation result;
  const std::size_t expected_argument_count =
      17U + input.include_roots.size() * 2U +
      input.compiler_options.size() * 2U +
      kTemporalDefenseInDepthArgv.size();
  if (expected_argument_count > kMaximumArgumentCount) {
    return reject(InvocationMaterializationStatus::kArgumentBudgetExceeded,
                  InvocationErrorField::kArguments);
  }
  result.arguments.reserve(expected_argument_count);
  result.arguments.emplace_back(kCanonicalArgv0);
  result.arguments.emplace_back("--no-default-config");
  result.arguments.emplace_back("-x");
  result.arguments.emplace_back("cuda");
  result.arguments.emplace_back("-std=c++17");
  result.arguments.emplace_back(
      input.semantic_pass.pass == CudaSemanticPass::kDeviceExtraction
          ? "--cuda-device-only"
          : "--cuda-host-only");
  result.arguments.push_back(joined_argument("--target=", host_triple));
  result.arguments.push_back(joined_argument(
      "--cuda-gpu-arch=", input.semantic_pass.device_architecture));
  result.arguments.emplace_back("-resource-dir");
  result.arguments.emplace_back(input.resource_directory_virtual_path);
  result.arguments.push_back(joined_argument(
      "--cuda-path=", input.cuda_toolkit_root_virtual_path));
  result.arguments.emplace_back("--cuda-path-ignore-env");
  result.arguments.emplace_back("-nostdinc");
  result.arguments.emplace_back("-nostdinc++");
  result.arguments.emplace_back("-nogpuinc");
  result.arguments.emplace_back("-nogpulib");

  for (const InvocationIncludeRoot& root : input.include_roots) {
    result.arguments.emplace_back(root.mode == InvocationIncludeMode::kQuote
                                      ? "-iquote"
                                      : "-isystem");
    result.arguments.emplace_back(root.virtual_path);
  }
  for (const std::string_view argument : kTemporalDefenseInDepthArgv) {
    result.arguments.emplace_back(argument);
  }
  for (const InvocationCompilerOption& option : input.compiler_options) {
    if (const auto* define = std::get_if<InvocationDefineOption>(&option)) {
      const auto status = append_policy_option(result.arguments,
                                                define->option);
      if (status != CommandLineMaterializationStatus::kValid) {
        return reject(
            InvocationMaterializationStatus::kCompilerPolicyRejected,
            InvocationErrorField::kCompilerOption,
            define->compiler_option_ordinal, status);
      }
      continue;
    }
    if (const auto* undefine =
            std::get_if<InvocationUndefineOption>(&option)) {
      const auto status = append_policy_option(result.arguments,
                                                undefine->option);
      if (status != CommandLineMaterializationStatus::kValid) {
        return reject(
            InvocationMaterializationStatus::kCompilerPolicyRejected,
            InvocationErrorField::kCompilerOption,
            undefine->compiler_option_ordinal, status);
      }
      continue;
    }
    if (const auto* warning = std::get_if<InvocationWarningOption>(&option)) {
      const auto status = append_policy_option(result.arguments,
                                                warning->option);
      if (status != CommandLineMaterializationStatus::kValid) {
        return reject(
            InvocationMaterializationStatus::kCompilerPolicyRejected,
            InvocationErrorField::kCompilerOption,
            warning->compiler_option_ordinal, status);
      }
      continue;
    }
    if (const auto* forced = std::get_if<InvocationForcedInclude>(&option)) {
      result.arguments.emplace_back("-include");
      result.arguments.emplace_back(forced->virtual_path);
      continue;
    }
    if (std::get_if<InvocationSyntaxOnlyOption>(&option) != nullptr) {
      result.arguments.emplace_back("-fsyntax-only");
      continue;
    }
    const auto& error_limit = std::get<InvocationErrorLimitOption>(option);
    result.arguments.push_back(joined_argument(
        "-ferror-limit=", std::to_string(error_limit.value)));
  }
  result.arguments.emplace_back(input.main_source_virtual_path);

  if (!arguments_within_budget(result.arguments)) {
    return reject(InvocationMaterializationStatus::kArgumentBudgetExceeded,
                  InvocationErrorField::kArguments);
  }
  return result;
}

}  // namespace browsergrad::cpp_cute
