#include "BrowserGradCppCuteCompilePlan.h"

#include <charconv>
#include <new>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace browsergrad::cpp_cute {
namespace {

bool parse_positive_u32(const std::string_view value,
                        std::uint32_t& output) noexcept {
  if (value.empty() || value.front() == '0') return false;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(),
                                      output);
  return parsed.ec == std::errc{} &&
         parsed.ptr == value.data() + value.size() && output != 0U;
}

std::optional<WarningDisposition> warning_disposition(
    const std::string_view value) noexcept {
  if (value == "ignore") return WarningDisposition::kIgnore;
  if (value == "warn") return WarningDisposition::kWarn;
  if (value == "error") return WarningDisposition::kError;
  return std::nullopt;
}

bool translate_options(const DecodedCompileSession& session,
                       std::vector<InvocationCompilerOption>& output) {
  output.reserve(session.compiler_option_count());
  for (std::size_t index = 0U; index < session.compiler_option_count();
       ++index) {
    const CompilerOptionView option = session.compiler_option(index);
    if (option.ordinal != index) return false;
    switch (option.kind) {
      case CompilerOptionKind::kDefine:
        output.emplace_back(InvocationDefineOption{
            option.ordinal,
            {option.name_or_id,
             option.has_value
                 ? std::optional<std::string_view>(
                       option.value_or_disposition)
                 : std::nullopt},
        });
        break;
      case CompilerOptionKind::kUndefine:
        output.emplace_back(InvocationUndefineOption{
            option.ordinal,
            {option.name_or_id},
        });
        break;
      case CompilerOptionKind::kWarningPolicy: {
        const std::optional<WarningDisposition> disposition =
            warning_disposition(option.value_or_disposition);
        if (!disposition.has_value()) return false;
        output.emplace_back(InvocationWarningOption{
            option.ordinal,
            {option.name_or_id, *disposition},
        });
        break;
      }
      case CompilerOptionKind::kForcedInclude:
        output.emplace_back(InvocationForcedInclude{
            option.ordinal,
            option.include_root_id,
            option.virtual_path,
        });
        break;
      case CompilerOptionKind::kFrontendOption:
        if (option.name_or_id == "syntax-only" && !option.has_value) {
          output.emplace_back(InvocationSyntaxOnlyOption{option.ordinal});
        } else if (option.name_or_id == "error-limit" && option.has_value) {
          std::uint32_t value = 0U;
          if (!parse_positive_u32(option.value_or_disposition, value)) {
            return false;
          }
          output.emplace_back(
              InvocationErrorLimitOption{option.ordinal, value});
        } else {
          return false;
        }
        break;
    }
  }
  return true;
}

bool translate_include_roots(const DecodedCompileSession& session,
                             std::vector<InvocationIncludeRoot>& output) {
  output.reserve(session.include_root_count());
  for (std::size_t index = 0U; index < session.include_root_count(); ++index) {
    const IncludeRootView root = session.include_root(index);
    InvocationIncludeMode mode;
    if (root.mode == "quote") {
      mode = InvocationIncludeMode::kQuote;
    } else if (root.mode == "system") {
      mode = InvocationIncludeMode::kSystem;
    } else {
      return false;
    }
    output.push_back({static_cast<std::uint32_t>(index), root.include_root_id,
                      mode, root.virtual_path});
  }
  return true;
}

bool translate_pass(const SemanticPassView& pass,
                    InvocationSemanticPass& output) noexcept {
  CudaSemanticPass kind;
  if (pass.ordinal == 0U && pass.pass_id == "cuda-device-sema" &&
      pass.domain == "device" && pass.role == "semantic-extraction" &&
      pass.invocation_mode == "cuda-device-only") {
    kind = CudaSemanticPass::kDeviceExtraction;
  } else if (pass.ordinal == 1U && pass.pass_id == "cuda-host-sema" &&
             pass.domain == "host" && pass.role == "validation" &&
             pass.invocation_mode == "cuda-host-only") {
    kind = CudaSemanticPass::kHostValidation;
  } else {
    return false;
  }
  output = {pass.ordinal, kind, pass.target_triple,
            pass.auxiliary_target_triple, pass.device_architecture};
  return true;
}

void append_frontend_resource_limits(
    const DecodedCompileSession& session,
    std::vector<std::string>& arguments) {
  // Clang's built-in limits bound one evaluation/depth chain. The separately
  // instrumented record enforces aggregate work across both semantic passes.
  arguments.push_back("-fconstexpr-steps=" +
                      std::to_string(session.request_semantic_limit(
                          CompileSemanticLimit::kConstexprSteps)));
  arguments.push_back("-ftemplate-depth=" +
                      std::to_string(session.request_semantic_limit(
                          CompileSemanticLimit::kTemplateDepth)));
}

}  // namespace

struct PreparedCppCuteCompilePlan::Impl final {
  std::string compilation_contract_hash;
  std::uint32_t maximum_output_byte_length = 0U;
  std::uint32_t maximum_diagnostic_count = 0U;
  std::array<std::vector<std::string>, 2U> arguments;
};

PreparedCppCuteCompilePlan::PreparedCppCuteCompilePlan(
    std::unique_ptr<const Impl> impl)
    : implementation_(std::move(impl)) {}

PreparedCppCuteCompilePlan::~PreparedCppCuteCompilePlan() = default;

std::string_view
PreparedCppCuteCompilePlan::compilation_contract_hash() const noexcept {
  return implementation_->compilation_contract_hash;
}

std::uint32_t
PreparedCppCuteCompilePlan::maximum_output_byte_length() const noexcept {
  return implementation_->maximum_output_byte_length;
}

std::uint32_t
PreparedCppCuteCompilePlan::maximum_diagnostic_count() const noexcept {
  return implementation_->maximum_diagnostic_count;
}

std::span<const std::string>
PreparedCppCuteCompilePlan::device_arguments() const noexcept {
  return implementation_->arguments[0U];
}

std::span<const std::string>
PreparedCppCuteCompilePlan::host_arguments() const noexcept {
  return implementation_->arguments[1U];
}

PrepareCppCuteCompilePlanResult prepare_cpp_cute_compile_plan(
    const DecodedCompileSession& session) noexcept {
  PrepareCppCuteCompilePlanResult result;
  try {
    if (session.semantic_pass_count() != 2U ||
        session.maximum_output_byte_length() == 0U ||
        session.request_semantic_limit(CompileSemanticLimit::kDiagnostics) == 0U) {
      result.status = CompilePlanStatus::kInvalidSessionData;
      return result;
    }
    std::vector<InvocationIncludeRoot> include_roots;
    std::vector<InvocationCompilerOption> compiler_options;
    if (!translate_include_roots(session, include_roots) ||
        !translate_options(session, compiler_options)) {
      result.status = CompilePlanStatus::kInvalidSessionData;
      return result;
    }

    auto implementation = std::make_unique<PreparedCppCuteCompilePlan::Impl>();
    implementation->compilation_contract_hash =
        session.compilation_contract_hash();
    implementation->maximum_output_byte_length =
        session.maximum_output_byte_length();
    implementation->maximum_diagnostic_count = static_cast<std::uint32_t>(
        session.request_semantic_limit(CompileSemanticLimit::kDiagnostics));
    for (std::size_t index = 0U; index < 2U; ++index) {
      InvocationSemanticPass semantic_pass;
      if (!translate_pass(session.semantic_pass(index), semantic_pass)) {
        result.status = CompilePlanStatus::kInvalidSessionData;
        result.failure.pass_ordinal = static_cast<std::uint32_t>(index);
        return result;
      }
      MaterializedCppCuteInvocation materialized =
          materialize_cpp_cute_invocation({
              session.compiler_version(),
              session.main_virtual_path(),
              semantic_pass,
              session.compiler_resource_directory_virtual_path(),
              include_roots,
              compiler_options,
          });
      if (materialized.status != InvocationMaterializationStatus::kValid) {
        result.status = CompilePlanStatus::kInvocationRejected;
        result.failure = {
            static_cast<std::uint32_t>(index), materialized.status,
            materialized.error_field, materialized.error_index};
        return result;
      }
      append_frontend_resource_limits(session, materialized.arguments);
      implementation->arguments[index] = std::move(materialized.arguments);
    }
    std::unique_ptr<const PreparedCppCuteCompilePlan::Impl> immutable(
        implementation.release());
    result.plan = std::unique_ptr<PreparedCppCuteCompilePlan>(
        new PreparedCppCuteCompilePlan(std::move(immutable)));
    result.status = CompilePlanStatus::kReady;
    return result;
  } catch (const std::bad_alloc&) {
    result.status = CompilePlanStatus::kResourceLimit;
    return result;
  } catch (...) {
    result.status = CompilePlanStatus::kInternalError;
    return result;
  }
}

}  // namespace browsergrad::cpp_cute
