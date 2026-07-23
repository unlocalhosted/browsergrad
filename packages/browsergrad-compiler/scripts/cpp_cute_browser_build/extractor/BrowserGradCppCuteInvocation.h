#pragma once

#include "BrowserGradCppCuteCommandLine.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace browsergrad::cpp_cute {

enum class CudaSemanticPass : std::uint8_t {
  kDeviceExtraction = 0,
  kHostValidation = 1,
};

enum class InvocationIncludeMode : std::uint8_t {
  kQuote = 0,
  kSystem = 1,
};

struct InvocationSemanticPass {
  /** Must be zero for device extraction and one for host validation. */
  std::uint32_t ordinal = 0U;
  CudaSemanticPass pass = CudaSemanticPass::kDeviceExtraction;
  /** Primary cc1 triple recorded for this pass. */
  std::string_view target_triple;
  /** Auxiliary cc1 triple recorded for this pass. */
  std::string_view auxiliary_target_triple;
  /** Closed Clang CUDA target CPU spelling, for example sm_80. */
  std::string_view device_architecture;
};

struct InvocationIncludeRoot {
  /** Array position is semantic and must equal this ordinal. */
  std::uint32_t ordinal = 0U;
  std::string_view id;
  InvocationIncludeMode mode = InvocationIncludeMode::kQuote;
  std::string_view virtual_path;
};

struct InvocationForcedInclude {
  /** Ordinal of the forced-include option in the source profile. */
  std::uint32_t compiler_option_ordinal = 0U;
  std::string_view include_root_id;
  std::string_view virtual_path;
};

struct InvocationDefineOption {
  std::uint32_t compiler_option_ordinal = 0U;
  CompilerDefineOption option;
};

struct InvocationUndefineOption {
  std::uint32_t compiler_option_ordinal = 0U;
  CompilerUndefineOption option;
};

struct InvocationWarningOption {
  std::uint32_t compiler_option_ordinal = 0U;
  CompilerWarningPolicyOption option;
};

struct InvocationSyntaxOnlyOption {
  std::uint32_t compiler_option_ordinal = 0U;
};

struct InvocationErrorLimitOption {
  std::uint32_t compiler_option_ordinal = 0U;
  std::uint32_t value = 0U;
};

/** Exact closed counterpart of one ordered frontend-profile option. */
using InvocationCompilerOption =
    std::variant<InvocationDefineOption, InvocationUndefineOption,
                 InvocationWarningOption, InvocationForcedInclude,
                 InvocationSyntaxOnlyOption, InvocationErrorLimitOption>;

/**
 * Sealed typed input for one Clang 22.1.8 CUDA semantic pass.
 *
 * There is intentionally no raw argv, shell command, language-mode, standard,
 * ambient search path, or linker option escape hatch.
 */
struct CppCuteInvocationInput {
  std::string_view clang_version;
  std::string_view main_source_virtual_path;
  InvocationSemanticPass semantic_pass;
  std::string_view resource_directory_virtual_path;
  std::string_view cuda_toolkit_root_virtual_path;
  std::span<const InvocationIncludeRoot> include_roots;
  /** Array position and each option's compiler_option_ordinal must match. */
  std::span<const InvocationCompilerOption> compiler_options;
};

enum class InvocationMaterializationStatus : std::uint8_t {
  kValid,
  kWrongClangVersion,
  kInvalidMainSourcePath,
  kInvalidSemanticPass,
  kInvalidTargetTriple,
  kInvalidAuxiliaryTargetTriple,
  kInvalidDeviceArchitecture,
  kInvalidResourceDirectoryPath,
  kInvalidCudaToolkitRootPath,
  kMissingCudaToolkitIncludeRoot,
  kInvalidCudaToolkitIncludeRoot,
  kMissingIncludeRoots,
  kTooManyIncludeRoots,
  kInvalidIncludeRootOrdinal,
  kInvalidIncludeRootId,
  kInvalidIncludeRootMode,
  kInvalidIncludeRootPath,
  kDuplicateIncludeRootId,
  kDuplicateIncludeRootPath,
  kMissingResourceIncludeRoot,
  kInvalidResourceIncludeRoot,
  kMissingMainSourceIncludeRoot,
  kTooManyCompilerOptions,
  kInvalidCompilerOption,
  kInvalidCompilerOptionOrdinal,
  kUnknownForcedIncludeRoot,
  kInvalidForcedIncludePath,
  kForcedIncludeOutsideRoot,
  kDuplicateForcedIncludePath,
  kForcedIncludeConflictsWithMainSource,
  kMissingSyntaxOnlyOption,
  kDuplicateSyntaxOnlyOption,
  kMissingErrorLimitOption,
  kDuplicateErrorLimitOption,
  kInvalidErrorLimit,
  kCompilerPolicyRejected,
  kArgumentBudgetExceeded,
};

enum class InvocationErrorField : std::uint8_t {
  kNone,
  kClangVersion,
  kMainSource,
  kSemanticPass,
  kResourceDirectory,
  kCudaToolkit,
  kIncludeRoot,
  kCompilerOption,
  kArguments,
};

struct MaterializedCppCuteInvocation {
  InvocationMaterializationStatus status =
      InvocationMaterializationStatus::kValid;
  InvocationErrorField error_field = InvocationErrorField::kNone;
  std::uint32_t error_index = 0U;
  CommandLineMaterializationStatus policy_status =
      CommandLineMaterializationStatus::kValid;
  /** Exact owning argv. It is never joined, escaped, or passed through a shell. */
  std::vector<std::string> arguments;
};

std::string_view cpp_cute_invocation_argv0() noexcept;
std::string_view cpp_cute_invocation_clang_version() noexcept;

/**
 * Builds one exact driver argv or rejects with an empty argv.
 *
 * Compiler options are emitted in their exact array/ordinal order. Exactly one
 * syntax-only option and one positive error-limit option are required; neither
 * is silently synthesized by this layer.
 *
 * Device extraction validates the device primary and host auxiliary triples,
 * then supplies the host triple to the CUDA driver. Clang 22.1.8 derives the
 * NVPTX cc1 job from --cuda-device-only and --cuda-gpu-arch. Supplying NVPTX
 * directly through --target would incorrectly request an NVPTX host toolchain.
 */
MaterializedCppCuteInvocation materialize_cpp_cute_invocation(
    const CppCuteInvocationInput& input);

}  // namespace browsergrad::cpp_cute
