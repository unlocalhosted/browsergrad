#pragma once

#include "BrowserGradCppCuteCompileSession.h"
#include "BrowserGradCppCuteInvocation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace browsergrad::cpp_cute {

enum class CompilePlanStatus : std::uint8_t {
  kReady,
  kInvalidSessionData,
  kInvocationRejected,
  kResourceLimit,
  kInternalError,
};

struct CompilePlanFailure {
  std::uint32_t pass_ordinal = 0U;
  InvocationMaterializationStatus invocation_status =
      InvocationMaterializationStatus::kValid;
  InvocationErrorField invocation_field = InvocationErrorField::kNone;
  std::uint32_t invocation_index = 0U;
};

struct PrepareCppCuteCompilePlanResult;

/** Owning device-first/host-second argv authority for one decoded session. */
class PreparedCppCuteCompilePlan final {
 public:
  ~PreparedCppCuteCompilePlan();

  PreparedCppCuteCompilePlan(const PreparedCppCuteCompilePlan&) = delete;
  PreparedCppCuteCompilePlan& operator=(const PreparedCppCuteCompilePlan&) =
      delete;
  PreparedCppCuteCompilePlan(PreparedCppCuteCompilePlan&&) = delete;
  PreparedCppCuteCompilePlan& operator=(PreparedCppCuteCompilePlan&&) = delete;

  std::string_view compilation_contract_hash() const noexcept;
  std::uint32_t maximum_output_byte_length() const noexcept;
  std::uint32_t maximum_diagnostic_count() const noexcept;
  std::span<const std::string> device_arguments() const noexcept;
  std::span<const std::string> host_arguments() const noexcept;

 private:
  struct Impl;
  explicit PreparedCppCuteCompilePlan(std::unique_ptr<const Impl> impl);
  std::unique_ptr<const Impl> implementation_;

  friend struct PrepareCppCuteCompilePlanResult;
  friend PrepareCppCuteCompilePlanResult prepare_cpp_cute_compile_plan(
      const DecodedCompileSession& session) noexcept;
};

struct PrepareCppCuteCompilePlanResult {
  CompilePlanStatus status = CompilePlanStatus::kInternalError;
  CompilePlanFailure failure{};
  std::unique_ptr<PreparedCppCuteCompilePlan> plan;

  PrepareCppCuteCompilePlanResult() = default;
  PrepareCppCuteCompilePlanResult(PrepareCppCuteCompilePlanResult&&) noexcept =
      default;
  PrepareCppCuteCompilePlanResult& operator=(
      PrepareCppCuteCompilePlanResult&&) noexcept = default;
  PrepareCppCuteCompilePlanResult(const PrepareCppCuteCompilePlanResult&) =
      delete;
  PrepareCppCuteCompilePlanResult& operator=(
      const PrepareCppCuteCompilePlanResult&) = delete;
};

PrepareCppCuteCompilePlanResult prepare_cpp_cute_compile_plan(
    const DecodedCompileSession& session) noexcept;

}  // namespace browsergrad::cpp_cute
