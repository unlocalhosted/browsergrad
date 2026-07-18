#pragma once

#include "BrowserGradCppCuteCompilePlan.h"
#include "BrowserGradCppCuteCompileSession.h"
#include "BrowserGradCppCuteDiagnostics.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace browsergrad::cpp_cute {

enum class ProducerReviewStatus : std::uint8_t {
  kReviewComplete,
  kReviewCompleteWithBlockingDiagnostics,
  kInvalidPlan,
  kResourceLimit,
  kVfsError,
  kInternalError,
};

enum class ProducerIncludeKind : std::uint8_t {
  kSourceQuote,
  kSourceAngle,
  kCompilerForced,
};

struct ProducerIncludeEdgeObservation {
  ProducerIncludeKind kind = ProducerIncludeKind::kSourceQuote;
  std::string including_file_path;
  std::string resolved_file_path;
  std::string spelling;
  std::uint64_t directive_start_byte_offset = 0U;
  std::uint64_t directive_end_byte_offset = 0U;
  std::uint32_t compiler_option_ordinal = 0U;
};

struct ProducerIntegerHierarchy {
  bool tuple = false;
  std::int64_t value = 0;
  std::vector<ProducerIntegerHierarchy> elements;
};

struct ProducerLayoutObservation {
  bool selected = false;
  bool resolved_layout_type = false;
  bool resolved_static_affine_layout = false;
  std::string canonical_usr;
  std::string canonical_name;
  std::string canonical_type;
  std::string initializer_callee;
  std::uint32_t identity_begin_byte = 0U;
  std::uint32_t identity_end_byte = 0U;
  std::uint32_t rank = 0U;
  std::uint32_t leaf_rank = 0U;
  std::int64_t size = 0;
  std::int64_t cosize = 0;
  ProducerIntegerHierarchy shape;
  ProducerIntegerHierarchy stride;
};

struct ProducerOpenedFileObservation {
  std::string virtual_path;
  std::string content_sha256;
  std::uint64_t byte_length = 0U;
};

struct ProducerDiagnosticObservation {
  RawDiagnosticStage stage = RawDiagnosticStage::kArtifactExtractor;
  RawDiagnosticSeverity severity = RawDiagnosticSeverity::kIgnored;
  bool custom = false;
  CustomDiagnosticCode custom_code =
      CustomDiagnosticCode::kDiagnosticNormalizationFailed;
  std::uint32_t raw_diagnostic_id = 0U;
  std::string rendered_message;
  bool has_source_location = false;
  std::string virtual_path;
  std::uint64_t byte_offset = 0U;
};

struct ProducerPassObservation {
  bool invocation_succeeded = false;
  bool policy_installed = false;
  bool policy_failed = false;
  bool vfs_failed = false;
  std::uint32_t policy_violation_count = 0U;
  std::uint32_t clang_error_count = 0U;
  bool diagnostic_capture_failed = false;
  std::vector<ProducerDiagnosticObservation> diagnostics;
  std::vector<std::string> opened_file_paths;
  std::vector<ProducerOpenedFileObservation> opened_files;
  std::vector<ProducerIncludeEdgeObservation> include_edges;
  ProducerLayoutObservation layout;
};

struct ProducerReviewResult {
  ProducerReviewStatus status = ProducerReviewStatus::kInternalError;
  std::uint32_t completed_pass_count = 0U;
  std::uint32_t blocking_diagnostic_pass_count = 0U;
  bool shared_surface_converged = false;
  std::array<ProducerPassObservation, 2U> passes;
};

/** Device-first/host-second review path; it cannot mint Artifact V3. */
ProducerReviewResult run_cpp_cute_producer_review(
    const PreparedCppCuteCompilePlan& plan,
    const DecodedCompileSession& session) noexcept;

}  // namespace browsergrad::cpp_cute
