#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "BrowserGradCppCuteDiagnostics.h"
#include "BrowserGradCppCuteImportedVfs.h"
#include "BrowserGradCppCutePreprocessorPolicy.h"

namespace browsergrad::cpp_cute {

enum class SourceAnchorKind : std::uint8_t {
  kLayoutVariable,
  kViewCopyFunction,
};

struct SourceAnchor {
  std::string virtual_path;
  std::uint32_t begin_byte = 0;
  std::uint32_t end_byte = 0;
  SourceAnchorKind kind = SourceAnchorKind::kLayoutVariable;
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

enum class ViewCopyScalarKind : std::uint8_t {
  kUnsupported,
  kFloat32,
  kSignedInt32,
  kUnsignedInt32,
};

struct ViewCopyParameterTrace {
  bool resolved_pointer = false;
  ViewCopyScalarKind scalar_kind = ViewCopyScalarKind::kUnsupported;
  bool pointee_const = false;
  std::uint32_t ordinal = 0U;
  std::string canonical_usr;
  std::string canonical_name;
  std::string canonical_type;
  std::uint32_t declaration_begin_byte = 0U;
  std::uint32_t declaration_end_byte = 0U;
  std::uint32_t identity_begin_byte = 0U;
  std::uint32_t identity_end_byte = 0U;
};

struct ViewCopyTensorTrace {
  bool resolved_tensor_type = false;
  bool resolved_static_affine_layout = false;
  bool initializer_parameter_bound = false;
  bool engine_pointee_const = false;
  std::uint32_t engine_parameter_ordinal = 0U;
  std::string canonical_usr;
  std::string canonical_name;
  std::string canonical_type;
  std::string tensor_template_path;
  std::string initializer_callee_usr;
  std::string initializer_callee_name;
  std::string initializer_callee_path;
  std::string layout_canonical_type;
  std::string layout_template_path;
  std::uint32_t declaration_begin_byte = 0U;
  std::uint32_t declaration_end_byte = 0U;
  std::uint32_t identity_begin_byte = 0U;
  std::uint32_t identity_end_byte = 0U;
  std::uint32_t rank = 0U;
  std::uint32_t leaf_rank = 0U;
  std::int64_t size = 0;
  std::int64_t cosize = 0;
  LayoutIntegerHierarchy shape;
  LayoutIntegerHierarchy stride;
};

struct ViewCopyTrace {
  bool selected = false;
  bool ambiguous = false;
  bool resolved_function = false;
  bool resolved_copy = false;
  bool cuda_host = false;
  bool cuda_device = false;
  bool cuda_global = false;
  bool force_inline = false;
  std::string canonical_usr;
  std::string canonical_name;
  std::string canonical_type;
  std::string copy_callee_usr;
  std::string copy_callee_name;
  std::string copy_callee_path;
  std::uint32_t declaration_begin_byte = 0U;
  std::uint32_t declaration_end_byte = 0U;
  std::uint32_t identity_begin_byte = 0U;
  std::uint32_t identity_end_byte = 0U;
  std::uint32_t copy_begin_byte = 0U;
  std::uint32_t copy_end_byte = 0U;
  std::uint32_t source_tensor_ordinal = 0U;
  std::uint32_t destination_tensor_ordinal = 0U;
  std::vector<ViewCopyParameterTrace> parameters;
  std::vector<ViewCopyTensorTrace> tensors;
};

struct ClangForcedIncludeObservation {
  std::string_view virtual_path;
  std::uint32_t compiler_option_ordinal = kImportedVfsNoCompilerOptionOrdinal;
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
  ImportedVfsObserverFailure vfs_failure = ImportedVfsObserverFailure::kNone;
  std::uint32_t policy_violation_count = 0U;
  std::uint32_t clang_error_count = 0U;
  bool diagnostic_capture_failed = false;
  bool frontend_work_limit_exceeded = false;
  std::vector<ClangDiagnosticObservation> diagnostics;
  ImportedVfsPassObservation vfs_observation;
  LayoutTrace layout_trace;
  ViewCopyTrace view_copy_trace;
};

/**
 * Executes one independently owned Clang semantic pass over the closed VFS.
 * Every call creates fresh VFS-observation, include-edge, preprocessing-policy,
 * diagnostic-count, and semantic-extraction state.
 */
bool run_cpp_cute_clang_pass_for_review(
    const std::vector<std::string>& command_line, const SourceAnchor& anchor,
    std::span<const ClangForcedIncludeObservation> forced_includes,
    ImportedVfsObservationLimits observation_limits,
    std::uint32_t maximum_diagnostic_count,
    std::uint32_t maximum_diagnostic_byte_length, ClangPassReview& review);

}  // namespace browsergrad::cpp_cute
