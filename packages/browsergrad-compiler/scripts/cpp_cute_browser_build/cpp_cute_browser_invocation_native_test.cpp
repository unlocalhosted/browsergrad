#include "extractor/BrowserGradCppCuteInvocation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace browsergrad::cpp_cute;

#define BG_CHECK(condition)                                                   \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::fprintf(stderr, "invocation check failed at line %d: %s\n",       \
                   __LINE__, #condition);                                     \
      return 1;                                                               \
    }                                                                         \
  } while (false)

constexpr std::array<InvocationIncludeRoot, 4U> kIncludeRoots = {{
    {0U, "workspace", InvocationIncludeMode::kQuote, "/workspace/project"},
    {1U, "clang-resource", InvocationIncludeMode::kSystem,
     "/toolchain/clang/lib/clang/22/include"},
    {2U, "cuda", InvocationIncludeMode::kSystem,
     "/toolchain/cuda/include"},
    {3U, "cxx-stdlib", InvocationIncludeMode::kSystem,
     "/toolchain/cxx/include/c++/v1"},
}};

const std::array<InvocationCompilerOption, 7U> kCompilerOptions = {{
    InvocationDefineOption{0U, {"MESSAGE", "x y;$(not-a-shell)"}},
    InvocationForcedInclude{
        1U, "clang-resource",
        "/toolchain/clang/lib/clang/22/include/__clang_cuda_runtime_wrapper.h"},
    InvocationWarningOption{
        2U, {"clang.unused-parameter", WarningDisposition::kWarn}},
    InvocationSyntaxOnlyOption{3U},
    InvocationUndefineOption{4U, {"LEGACY_MODE"}},
    InvocationErrorLimitOption{5U, 64U},
    InvocationForcedInclude{
        6U, "workspace", "/workspace/project/browsergrad_prelude.h"},
}};

InvocationSemanticPass device_pass() {
  return {0U, CudaSemanticPass::kDeviceExtraction,
          "nvptx64-nvidia-cuda", "x86_64-unknown-linux-gnu", "sm_80"};
}

InvocationSemanticPass host_pass() {
  return {1U, CudaSemanticPass::kHostValidation,
          "x86_64-unknown-linux-gnu", "nvptx64-nvidia-cuda", "sm_80"};
}

CppCuteInvocationInput valid_input(
    std::span<const InvocationIncludeRoot> include_roots = kIncludeRoots,
    std::span<const InvocationCompilerOption> compiler_options =
        kCompilerOptions) {
  return {
      "22.1.8",
      "/workspace/project/kernel.cu",
      device_pass(),
      "/toolchain/clang/lib/clang/22",
      include_roots,
      compiler_options,
  };
}

bool rejects(
    const CppCuteInvocationInput& input,
    InvocationMaterializationStatus status,
    InvocationErrorField field,
    std::uint32_t error_index = 0U,
    CommandLineMaterializationStatus policy_status =
        CommandLineMaterializationStatus::kValid) {
  const MaterializedCppCuteInvocation result =
      materialize_cpp_cute_invocation(input);
  return result.status == status && result.error_field == field &&
         result.error_index == error_index &&
         result.policy_status == policy_status && result.arguments.empty();
}

int run_exact_order_tests() {
  BG_CHECK(cpp_cute_invocation_argv0() == "clang++");
  BG_CHECK(cpp_cute_invocation_clang_version() == "22.1.8");

  const MaterializedCppCuteInvocation device =
      materialize_cpp_cute_invocation(valid_input());
  BG_CHECK(device.status == InvocationMaterializationStatus::kValid);
  BG_CHECK(device.error_field == InvocationErrorField::kNone);
  const std::vector<std::string> expected = {
      "clang++",
      "--no-default-config",
      "-x",
      "cuda",
      "-std=c++17",
      "--cuda-device-only",
      "--target=x86_64-unknown-linux-gnu",
      "--cuda-gpu-arch=sm_80",
      "-resource-dir",
      "/toolchain/clang/lib/clang/22",
      "--cuda-path-ignore-env",
      "-nostdinc",
      "-nostdinc++",
      "-nogpuinc",
      "-nogpulib",
      "-iquote",
      "/workspace/project",
      "-isystem",
      "/toolchain/clang/lib/clang/22/include",
      "-isystem",
      "/toolchain/cuda/include",
      "-isystem",
      "/toolchain/cxx/include/c++/v1",
      "-Werror=builtin-macro-redefined",
      "-Werror=date-time",
      "-Werror=macro-redefined",
      "-DMESSAGE=x y;$(not-a-shell)",
      "-include",
      "/toolchain/clang/lib/clang/22/include/__clang_cuda_runtime_wrapper.h",
      "-Wunused-parameter",
      "-Wno-error=unused-parameter",
      "-fsyntax-only",
      "-ULEGACY_MODE",
      "-ferror-limit=64",
      "-include",
      "/workspace/project/browsergrad_prelude.h",
      "/workspace/project/kernel.cu",
  };
  BG_CHECK(device.arguments == expected);

  CppCuteInvocationInput host_input = valid_input();
  host_input.semantic_pass = host_pass();
  const MaterializedCppCuteInvocation host =
      materialize_cpp_cute_invocation(host_input);
  BG_CHECK(host.status == InvocationMaterializationStatus::kValid);
  std::vector<std::string> expected_host = expected;
  expected_host[5] = "--cuda-host-only";
  BG_CHECK(host.arguments == expected_host);
  return 0;
}

MaterializedCppCuteInvocation build_from_temporary_storage() {
  std::string compiler = "22.1.8";
  std::string host_triple = "x86_64-unknown-linux-gnu";
  std::string architecture = "sm_90a";
  std::string source_root = "/workspace/x;$(literal)";
  std::string main_source = source_root + "/--cuda-host-only.cu";
  std::string resource_directory = "/toolchain/clang/lib/clang/22";
  std::string resource_include = resource_directory + "/include";
  std::string forced_path = source_root + "/prelude with spaces.h";
  std::string macro_name = "MESSAGE";
  std::string macro_value = "x y;$(still-not-a-shell)";
  const std::array roots = {
      InvocationIncludeRoot{0U, "workspace", InvocationIncludeMode::kQuote,
                            source_root},
      InvocationIncludeRoot{1U, "clang-resource",
                            InvocationIncludeMode::kSystem,
                            resource_include},
  };
  const std::array<InvocationCompilerOption, 4U> options = {{
      InvocationDefineOption{0U, {macro_name, macro_value}},
      InvocationForcedInclude{1U, "workspace", forced_path},
      InvocationSyntaxOnlyOption{2U},
      InvocationErrorLimitOption{3U, 20U},
  }};
  CppCuteInvocationInput input{
      compiler,
      main_source,
      {0U, CudaSemanticPass::kDeviceExtraction,
       "nvptx64-nvidia-cuda", host_triple, architecture},
      resource_directory,
      roots,
      options,
  };
  return materialize_cpp_cute_invocation(input);
}

int run_ownership_and_no_shell_tests() {
  const MaterializedCppCuteInvocation result =
      build_from_temporary_storage();
  BG_CHECK(result.status == InvocationMaterializationStatus::kValid);
  BG_CHECK(result.arguments.front() == "clang++");
  BG_CHECK(result.arguments.back() ==
           "/workspace/x;$(literal)/--cuda-host-only.cu");
  BG_CHECK(result.arguments[6] == "--target=x86_64-unknown-linux-gnu");
  BG_CHECK(result.arguments[7] == "--cuda-gpu-arch=sm_90a");
  BG_CHECK(result.arguments[16] == "/workspace/x;$(literal)");
  BG_CHECK(result.arguments[22] ==
           "-DMESSAGE=x y;$(still-not-a-shell)");
  BG_CHECK(result.arguments[24] ==
           "/workspace/x;$(literal)/prelude with spaces.h");
  BG_CHECK(result.arguments[25] == "-fsyntax-only");
  BG_CHECK(result.arguments[26] == "-ferror-limit=20");
  return 0;
}

int run_identity_and_pass_rejection_tests() {
  CppCuteInvocationInput input = valid_input();
  input.clang_version = "22.1.7";
  BG_CHECK(rejects(input,
                   InvocationMaterializationStatus::kWrongClangVersion,
                   InvocationErrorField::kClangVersion));

  input = valid_input();
  input.main_source_virtual_path = "workspace/kernel.cu";
  BG_CHECK(rejects(input,
                   InvocationMaterializationStatus::kInvalidMainSourcePath,
                   InvocationErrorField::kMainSource));
  input.main_source_virtual_path = "/workspace/../kernel.cu";
  BG_CHECK(rejects(input,
                   InvocationMaterializationStatus::kInvalidMainSourcePath,
                   InvocationErrorField::kMainSource));
  std::string nul_path = "/workspace/project/";
  nul_path.push_back('\0');
  nul_path.append("kernel.cu");
  input.main_source_virtual_path = nul_path;
  BG_CHECK(rejects(input,
                   InvocationMaterializationStatus::kInvalidMainSourcePath,
                   InvocationErrorField::kMainSource));

  input = valid_input();
  input.semantic_pass.ordinal = 1U;
  BG_CHECK(rejects(input,
                   InvocationMaterializationStatus::kInvalidSemanticPass,
                   InvocationErrorField::kSemanticPass));
  input = valid_input();
  input.semantic_pass.pass = static_cast<CudaSemanticPass>(0xffU);
  BG_CHECK(rejects(input,
                   InvocationMaterializationStatus::kInvalidSemanticPass,
                   InvocationErrorField::kSemanticPass));
  input = valid_input();
  input.semantic_pass.target_triple = "x86_64-unknown-linux-gnu";
  BG_CHECK(rejects(input,
                   InvocationMaterializationStatus::kInvalidTargetTriple,
                   InvocationErrorField::kSemanticPass));
  input = valid_input();
  input.semantic_pass.auxiliary_target_triple = "nvptx64-nvidia-cuda";
  BG_CHECK(rejects(
      input,
      InvocationMaterializationStatus::kInvalidAuxiliaryTargetTriple,
      InvocationErrorField::kSemanticPass));
  input = valid_input();
  input.semantic_pass.device_architecture = "sm_080";
  BG_CHECK(rejects(
      input, InvocationMaterializationStatus::kInvalidDeviceArchitecture,
      InvocationErrorField::kSemanticPass));

  input = valid_input();
  input.semantic_pass = host_pass();
  input.semantic_pass.target_triple = "--config=ambient";
  BG_CHECK(rejects(input,
                   InvocationMaterializationStatus::kInvalidTargetTriple,
                   InvocationErrorField::kSemanticPass));
  input = valid_input();
  input.semantic_pass = host_pass();
  input.semantic_pass.auxiliary_target_triple = "amdgcn-amd-amdhsa";
  BG_CHECK(rejects(
      input,
      InvocationMaterializationStatus::kInvalidAuxiliaryTargetTriple,
      InvocationErrorField::kSemanticPass));

  input = valid_input();
  input.resource_directory_virtual_path = "/toolchain/./clang";
  BG_CHECK(rejects(
      input,
      InvocationMaterializationStatus::kInvalidResourceDirectoryPath,
      InvocationErrorField::kResourceDirectory));
  const std::string oversized_resource(4'097U, 'x');
  std::string absolute_oversized_resource = "/" + oversized_resource;
  input.resource_directory_virtual_path = absolute_oversized_resource;
  BG_CHECK(rejects(
      input,
      InvocationMaterializationStatus::kInvalidResourceDirectoryPath,
      InvocationErrorField::kResourceDirectory));
  return 0;
}

int run_include_root_rejection_tests() {
  constexpr std::array<InvocationIncludeRoot, 0U> empty = {};
  BG_CHECK(rejects(valid_input(empty),
                   InvocationMaterializationStatus::kMissingIncludeRoots,
                   InvocationErrorField::kIncludeRoot));
  const std::vector<InvocationIncludeRoot> too_many(257U, kIncludeRoots[0]);
  BG_CHECK(rejects(valid_input(too_many),
                   InvocationMaterializationStatus::kTooManyIncludeRoots,
                   InvocationErrorField::kIncludeRoot, 256U));

  auto roots = kIncludeRoots;
  roots[0].ordinal = 1U;
  BG_CHECK(rejects(valid_input(roots),
                   InvocationMaterializationStatus::kInvalidIncludeRootOrdinal,
                   InvocationErrorField::kIncludeRoot, 0U));
  roots = kIncludeRoots;
  roots[0].id = "9workspace";
  BG_CHECK(rejects(valid_input(roots),
                   InvocationMaterializationStatus::kInvalidIncludeRootId,
                   InvocationErrorField::kIncludeRoot, 0U));
  roots = kIncludeRoots;
  roots[0].mode = static_cast<InvocationIncludeMode>(0xffU);
  BG_CHECK(rejects(valid_input(roots),
                   InvocationMaterializationStatus::kInvalidIncludeRootMode,
                   InvocationErrorField::kIncludeRoot, 0U));
  roots = kIncludeRoots;
  roots[0].virtual_path = "/workspace//project";
  BG_CHECK(rejects(valid_input(roots),
                   InvocationMaterializationStatus::kInvalidIncludeRootPath,
                   InvocationErrorField::kIncludeRoot, 0U));
  roots = kIncludeRoots;
  roots[1].id = "workspace";
  BG_CHECK(rejects(valid_input(roots),
                   InvocationMaterializationStatus::kDuplicateIncludeRootId,
                   InvocationErrorField::kIncludeRoot, 1U));
  roots = kIncludeRoots;
  roots[2].virtual_path = roots[1].virtual_path;
  BG_CHECK(rejects(valid_input(roots),
                   InvocationMaterializationStatus::kDuplicateIncludeRootPath,
                   InvocationErrorField::kIncludeRoot, 2U));

  constexpr std::array missing_resource = {
      InvocationIncludeRoot{0U, "workspace", InvocationIncludeMode::kQuote,
                            "/workspace/project"},
  };
  BG_CHECK(rejects(
      valid_input(missing_resource),
      InvocationMaterializationStatus::kMissingResourceIncludeRoot,
      InvocationErrorField::kIncludeRoot));
  roots = kIncludeRoots;
  roots[1].mode = InvocationIncludeMode::kQuote;
  BG_CHECK(rejects(
      valid_input(roots),
      InvocationMaterializationStatus::kInvalidResourceIncludeRoot,
      InvocationErrorField::kIncludeRoot, 1U));
  roots = kIncludeRoots;
  roots[0].mode = InvocationIncludeMode::kSystem;
  BG_CHECK(rejects(
      valid_input(roots),
      InvocationMaterializationStatus::kMissingMainSourceIncludeRoot,
      InvocationErrorField::kIncludeRoot));
  return 0;
}

int run_ordered_option_rejection_tests() {
  const std::vector<InvocationCompilerOption> too_many(
      4'097U, InvocationSyntaxOnlyOption{0U});
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, too_many),
      InvocationMaterializationStatus::kTooManyCompilerOptions,
      InvocationErrorField::kCompilerOption, 4'096U));

  auto options = kCompilerOptions;
  std::get<InvocationForcedInclude>(options[1]).compiler_option_ordinal = 2U;
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, options),
      InvocationMaterializationStatus::kInvalidCompilerOptionOrdinal,
      InvocationErrorField::kCompilerOption, 1U));

  const std::array<InvocationCompilerOption, 1U> missing_syntax = {{
      InvocationErrorLimitOption{0U, 64U},
  }};
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, missing_syntax),
      InvocationMaterializationStatus::kMissingSyntaxOnlyOption,
      InvocationErrorField::kCompilerOption, 1U));
  const std::array<InvocationCompilerOption, 3U> duplicate_syntax = {{
      InvocationSyntaxOnlyOption{0U},
      InvocationSyntaxOnlyOption{1U},
      InvocationErrorLimitOption{2U, 64U},
  }};
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, duplicate_syntax),
      InvocationMaterializationStatus::kDuplicateSyntaxOnlyOption,
      InvocationErrorField::kCompilerOption, 1U));
  const std::array<InvocationCompilerOption, 1U> missing_error = {{
      InvocationSyntaxOnlyOption{0U},
  }};
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, missing_error),
      InvocationMaterializationStatus::kMissingErrorLimitOption,
      InvocationErrorField::kCompilerOption, 1U));
  const std::array<InvocationCompilerOption, 3U> duplicate_error = {{
      InvocationSyntaxOnlyOption{0U},
      InvocationErrorLimitOption{1U, 64U},
      InvocationErrorLimitOption{2U, 32U},
  }};
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, duplicate_error),
      InvocationMaterializationStatus::kDuplicateErrorLimitOption,
      InvocationErrorField::kCompilerOption, 2U));
  const std::array<InvocationCompilerOption, 2U> invalid_error = {{
      InvocationSyntaxOnlyOption{0U},
      InvocationErrorLimitOption{1U, 1'000'000U},
  }};
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, invalid_error),
      InvocationMaterializationStatus::kInvalidErrorLimit,
      InvocationErrorField::kCompilerOption, 1U));
  const std::array<InvocationCompilerOption, 2U> zero_error = {{
      InvocationSyntaxOnlyOption{0U},
      InvocationErrorLimitOption{1U, 0U},
  }};
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, zero_error),
      InvocationMaterializationStatus::kInvalidErrorLimit,
      InvocationErrorField::kCompilerOption, 1U));
  return 0;
}

std::array<InvocationCompilerOption, 3U> forced_options(
    InvocationForcedInclude forced) {
  forced.compiler_option_ordinal = 0U;
  return {{forced, InvocationSyntaxOnlyOption{1U},
           InvocationErrorLimitOption{2U, 64U}}};
}

int run_forced_and_policy_rejection_tests() {
  auto options = forced_options(
      InvocationForcedInclude{0U, "unknown", "/workspace/project/x.h"});
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, options),
      InvocationMaterializationStatus::kUnknownForcedIncludeRoot,
      InvocationErrorField::kCompilerOption, 0U));
  options = forced_options(InvocationForcedInclude{
      0U, "workspace", "/workspace/../escape.h"});
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, options),
      InvocationMaterializationStatus::kInvalidForcedIncludePath,
      InvocationErrorField::kCompilerOption, 0U));
  options = forced_options(InvocationForcedInclude{
      0U, "clang-resource", "/workspace/project/outside.h"});
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, options),
      InvocationMaterializationStatus::kForcedIncludeOutsideRoot,
      InvocationErrorField::kCompilerOption, 0U));
  options = forced_options(InvocationForcedInclude{
      0U, "clang-resource", "/toolchain/clang/lib/clang/22/include"});
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, options),
      InvocationMaterializationStatus::kForcedIncludeOutsideRoot,
      InvocationErrorField::kCompilerOption, 0U));

  const std::array<InvocationCompilerOption, 4U> duplicate_forced = {{
      InvocationForcedInclude{0U, "workspace",
                              "/workspace/project/prelude.h"},
      InvocationForcedInclude{1U, "workspace",
                              "/workspace/project/prelude.h"},
      InvocationSyntaxOnlyOption{2U},
      InvocationErrorLimitOption{3U, 64U},
  }};
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, duplicate_forced),
      InvocationMaterializationStatus::kDuplicateForcedIncludePath,
      InvocationErrorField::kCompilerOption, 1U));
  options = forced_options(InvocationForcedInclude{
      0U, "workspace", "/workspace/project/kernel.cu"});
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, options),
      InvocationMaterializationStatus::kForcedIncludeConflictsWithMainSource,
      InvocationErrorField::kCompilerOption, 0U));

  const std::array<InvocationCompilerOption, 5U> conflicting_policy = {{
      InvocationDefineOption{0U, {"MODE", "1"}},
      InvocationForcedInclude{1U, "workspace",
                              "/workspace/project/prelude.h"},
      InvocationSyntaxOnlyOption{2U},
      InvocationErrorLimitOption{3U, 64U},
      InvocationUndefineOption{4U, {"MODE"}},
  }};
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, conflicting_policy),
      InvocationMaterializationStatus::kCompilerPolicyRejected,
      InvocationErrorField::kCompilerOption, 4U,
      CommandLineMaterializationStatus::kConflictingMacroAction));
  const std::array<InvocationCompilerOption, 3U> temporal_policy = {{
      InvocationSyntaxOnlyOption{0U},
      InvocationErrorLimitOption{1U, 64U},
      InvocationDefineOption{2U, {"__DATE__", "redacted"}},
  }};
  BG_CHECK(rejects(
      valid_input(kIncludeRoots, temporal_policy),
      InvocationMaterializationStatus::kCompilerPolicyRejected,
      InvocationErrorField::kCompilerOption, 2U,
      CommandLineMaterializationStatus::kTemporalMacroName));
  return 0;
}

}  // namespace

int main() {
  if (run_exact_order_tests() != 0) return 1;
  if (run_ownership_and_no_shell_tests() != 0) return 1;
  if (run_identity_and_pass_rejection_tests() != 0) return 1;
  if (run_include_root_rejection_tests() != 0) return 1;
  if (run_ordered_option_rejection_tests() != 0) return 1;
  return run_forced_and_policy_rejection_tests();
}
