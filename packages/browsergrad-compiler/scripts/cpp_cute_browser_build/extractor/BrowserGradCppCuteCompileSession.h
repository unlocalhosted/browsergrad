#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "BrowserGradCppCuteRuntime.h"

namespace browsergrad::cpp_cute {

enum class CompileSessionDecodeStatus : std::uint8_t {
  kReady,
  kInvalidFrame,
  kAbiMismatch,
  kResourceLimit,
  kInternalError,
};

enum class CompileSessionRegion : std::uint8_t {
  kNone,
  kProfile,
  kRequest,
  kCrossRegion,
};

/** Stable, non-attacker-controlled admission failure categories. */
enum class CompileSessionDecodeReason : std::uint16_t {
  kNone,
  kCanonicalJson,
  kSchema,
  kUnsupportedVersion,
  kUnsupportedDeployment,
  kRuntimeAbi,
  kSemanticAdapter,
  kCompilerIdentity,
  kIdentityMismatch,
  kContractMismatch,
  kLimit,
  kAllocation,
  kHash,
};

struct CompileSessionDecodeFailure {
  CompileSessionRegion region = CompileSessionRegion::kNone;
  CompileSessionDecodeReason reason = CompileSessionDecodeReason::kNone;
  std::uint32_t byte_offset = 0;
};

enum class CompilerOptionKind : std::uint8_t {
  kDefine,
  kUndefine,
  kFrontendOption,
  kWarningPolicy,
  kForcedInclude,
};

struct CompilerOptionView {
  CompilerOptionKind kind = CompilerOptionKind::kDefine;
  std::uint32_t ordinal = 0;
  std::string_view name_or_id;
  std::string_view value_or_disposition;
  std::string_view include_root_id;
  std::string_view virtual_path;
  bool has_value = false;
};

struct SemanticPassView {
  std::uint32_t ordinal = 0;
  std::string_view pass_id;
  std::string_view domain;
  std::string_view role;
  std::string_view invocation_mode;
  std::string_view target_triple;
  std::string_view auxiliary_target_triple;
  std::string_view device_architecture;
};

struct IncludeRootView {
  std::uint32_t ordinal = 0;
  std::string_view include_root_id;
  std::string_view mode;
  std::string_view virtual_path;
  std::string_view manifest_sha256;
  std::string_view owner_kind;
  std::string_view dependency_id;
};

struct SourceFileView {
  std::string_view file_id;
  std::string_view role;
  std::string_view virtual_path;
  std::string_view content_sha256;
  std::string_view byte_length;
  std::string_view include_root_id;
  bool has_include_root = false;
};

struct EntryRequestView {
  std::string_view request_id;
  std::string_view kind;
  std::string_view declaration_kind;
  std::string_view virtual_path;
  std::string_view begin_byte;
  std::string_view end_byte;
  std::string_view token_sha256;
};

struct CompileSessionDecodeResult;

/** Stable indices in the versioned profile/request semantic-limit vector. */
enum class CompileSemanticLimit : std::uint8_t {
  kSourceFiles = 0U,
  kHeaderFiles = 2U,
  kIncludeDepth = 4U,
  kMacroExpansions = 5U,
  kPreprocessedTokens = 6U,
  kAstNodes = 7U,
  kConstexprSteps = 8U,
  kTemplateInstantiations = 9U,
  kTemplateDepth = 10U,
  kDiagnostics = 18U,
};

/**
 * Immutable, owning admission authority.
 *
 * A decoded session proves canonical profile/request schema and identity only.
 * It deliberately does not prove VFS source bytes, a Clang invocation, either
 * CUDA semantic pass, or artifact-v3 authority.
 */
class DecodedCompileSession final {
 public:
  ~DecodedCompileSession();

  DecodedCompileSession(const DecodedCompileSession&) = delete;
  DecodedCompileSession& operator=(const DecodedCompileSession&) = delete;
  DecodedCompileSession(DecodedCompileSession&&) = delete;
  DecodedCompileSession& operator=(DecodedCompileSession&&) = delete;

  std::string_view profile_id() const noexcept;
  std::string_view profile_hash() const noexcept;
  std::string_view compilation_contract_hash() const noexcept;
  std::string_view compiler_version() const noexcept;
  std::string_view compiler_resource_directory_virtual_path() const noexcept;
  std::string_view request_id() const noexcept;
  std::string_view request_hash() const noexcept;
  std::string_view main_virtual_path() const noexcept;
  std::uint32_t maximum_output_byte_length() const noexcept;
  std::uint64_t request_semantic_limit(CompileSemanticLimit limit) const noexcept;

  std::size_t compiler_option_count() const noexcept;
  CompilerOptionView compiler_option(std::size_t index) const noexcept;
  std::size_t semantic_pass_count() const noexcept;
  SemanticPassView semantic_pass(std::size_t index) const noexcept;
  std::size_t include_root_count() const noexcept;
  IncludeRootView include_root(std::size_t index) const noexcept;
  std::size_t source_file_count() const noexcept;
  SourceFileView source_file(std::size_t index) const noexcept;
  EntryRequestView entry_request() const noexcept;

 private:
  struct Impl;
  explicit DecodedCompileSession(std::unique_ptr<const Impl> implementation);
  std::unique_ptr<const Impl> implementation_;

  friend struct CompileSessionDecodeResult;
  friend CompileSessionDecodeResult decode_compile_session(
      const ValidatedInputFrameRegions& regions) noexcept;
};

struct CompileSessionDecodeResult {
  CompileSessionDecodeStatus status = CompileSessionDecodeStatus::kInternalError;
  CompileSessionDecodeFailure failure{};
  std::unique_ptr<DecodedCompileSession> session;

  CompileSessionDecodeResult() = default;
  CompileSessionDecodeResult(CompileSessionDecodeResult&&) noexcept = default;
  CompileSessionDecodeResult& operator=(CompileSessionDecodeResult&&) noexcept = default;
  CompileSessionDecodeResult(const CompileSessionDecodeResult&) = delete;
  CompileSessionDecodeResult& operator=(const CompileSessionDecodeResult&) = delete;
};

CompileSessionDecodeResult decode_compile_session(
    const ValidatedInputFrameRegions& regions) noexcept;

}  // namespace browsergrad::cpp_cute
