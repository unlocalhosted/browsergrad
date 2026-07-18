#pragma once

#include "BrowserGradCppCuteDiagnostics.h"
#include "BrowserGradCppCuteImportedVfs.h"
#include "BrowserGradCppCutePreprocessorPolicy.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace browsergrad::cpp_cute {

struct SourceAnchor {
  std::string virtual_path;
  std::uint32_t begin_byte = 0;
  std::uint32_t end_byte = 0;
};

struct LayoutIntegerHierarchy {
  bool tuple = false;
  std::int64_t value = 0;
  std::vector<LayoutIntegerHierarchy> elements;
};

struct LayoutTrace {
  bool selected = false;
  bool resolved_layout_type = false;
  bool resolved_static_affine_layout = false;
  std::string canonical_usr;
  std::string canonical_name;
  std::string canonical_type;
  std::string initializer_callee;
  std::uint32_t identity_begin_byte = 0;
  std::uint32_t identity_end_byte = 0;
  std::uint32_t rank = 0U;
  std::uint32_t leaf_rank = 0U;
  std::int64_t size = 0;
  std::int64_t cosize = 0;
  LayoutIntegerHierarchy shape;
  LayoutIntegerHierarchy stride;
};

struct ClangForcedIncludeObservation {
  std::string_view virtual_path;
  std::uint32_t compiler_option_ordinal =
      kImportedVfsNoCompilerOptionOrdinal;
};

struct ClangDiagnosticObservation {
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

struct ClangPassReview {
  bool invocation_succeeded = false;
  CppCutePreprocessorPolicyInstallStatus policy_install_status =
      CppCutePreprocessorPolicyInstallStatus::kMissingPreprocessor;
  bool policy_failed = false;
  bool vfs_failed = false;
  std::uint32_t policy_violation_count = 0U;
  std::uint32_t clang_error_count = 0U;
  bool diagnostic_capture_failed = false;
  std::vector<ClangDiagnosticObservation> diagnostics;
  ImportedVfsPassObservation vfs_observation;
  LayoutTrace layout_trace;
};

/**
 * Executes one independently owned Clang semantic pass over the closed VFS.
 * Every call creates fresh VFS-observation, include-edge, preprocessing-policy,
 * diagnostic-count, and semantic-extraction state.
 */
bool run_cpp_cute_clang_pass_for_review(
    const std::vector<std::string>& command_line,
    const SourceAnchor& anchor,
    std::span<const ClangForcedIncludeObservation> forced_includes,
    ImportedVfsObservationLimits observation_limits,
    std::uint32_t maximum_diagnostic_count,
    std::uint32_t maximum_diagnostic_byte_length,
    ClangPassReview& review);

}  // namespace browsergrad::cpp_cute
